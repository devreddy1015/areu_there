"""Recorder / replayer.

Covers the pcap round trip, captures written by other tools (tcpdump on a
gateway, a mirror port, `-i any`), a live loopback recording, per-node
delivery statistics, and the guarantee the recorder exists for: running the
pipeline from a recording gives exactly the readings it gave live.
"""

import contextlib
import io
import math
import os
import socket
import struct
import tempfile
import threading
import time
import unittest

from wifisense import SensingPipeline
from wifisense.cli import main as cli_main
from wifisense.sources import SyntheticConfig, SyntheticCsiSource
from wifisense.sources.esp32_udp import build_esp32_frame, parse_esp32_frame
from wifisense.sources.recorder import (
    LINKTYPE_ETHERNET,
    LINKTYPE_IPV4,
    LINKTYPE_LINUX_SLL,
    LINKTYPE_LINUX_SLL2,
    LINKTYPE_NULL,
    LINKTYPE_RAW,
    PCAP_MAGIC_NS,
    PcapWriter,
    ReadStats,
    RecordedPacket,
    iter_frames,
    link_stats,
    read_pcap,
    record_udp,
    replay_udp,
)

T0_NS = 1_760_000_000_123_456_789
SRC = ("10.0.0.21", 40001)
DST = ("10.0.0.1", 5566)


# -- helpers ---------------------------------------------------------------


def quantize(frame):
    """CsiFrame -> int8 (imag, real) pairs, imaginary first like ESP-IDF."""

    def clamp(v):
        return max(-128, min(127, round(v)))

    return [
        (clamp(a * math.sin(p)), clamp(a * math.cos(p)))
        for a, p in zip(frame.amplitudes, frame.phases)
    ]


def synthetic_payloads(n, node_id=1, rate_hz=20.0, occupied=True, seed=7):
    cfg = SyntheticConfig(sample_rate_hz=rate_hz, occupied=occupied, seed=seed)
    source = SyntheticCsiSource(cfg)
    return [
        build_esp32_frame(quantize(f), node_id=node_id, sequence=i)
        for i, f in enumerate(source.frames(n))
    ]


def tiny_frame(node_id, sequence):
    return build_esp32_frame([(1, 2), (3, 4)], node_id=node_id, sequence=sequence)


def ipv4_udp(payload, src=SRC, dst=DST, *, options=b"", proto=17, frag=0):
    """Hand-built IPv4 (+ UDP) packet, independent of the writer under test."""
    assert len(options) % 4 == 0
    ihl = 5 + len(options) // 4
    body = struct.pack("!HHHH", src[1], dst[1], 8 + len(payload), 0) + payload
    total = ihl * 4 + len(body)
    ip = struct.pack(
        "!BBHHHBBH4s4s",
        0x40 | ihl, 0, total, 1, frag, 64, proto, 0,
        socket.inet_aton(src[0]), socket.inet_aton(dst[0]),
    )
    return ip + options + body


def pcap_bytes(records, *, linktype=LINKTYPE_RAW, endian="<", nano=True):
    """records: (t_ns, data) or (t_ns, data, orig_len)."""
    magic = 0xA1B23C4D if nano else 0xA1B2C3D4
    out = bytearray(struct.pack(endian + "IHHiIII", magic, 2, 4, 0, 0, 65535, linktype))
    for t_ns, data, *rest in records:
        orig = rest[0] if rest else len(data)
        sec, rem = divmod(t_ns, 1_000_000_000)
        frac = rem if nano else rem // 1000
        out += struct.pack(endian + "IIII", sec, frac, len(data), orig) + data
    return bytes(out)


class TempDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def path(self, name="rec.pcap"):
        return os.path.join(self.dir, name)

    def put(self, data, name="in.pcap"):
        p = self.path(name)
        with open(p, "wb") as f:
            f.write(data)
        return p

    def read_all(self, path, port=None):
        st = ReadStats()
        return list(read_pcap(path, port, st)), st


# -- writer / reader round trip -------------------------------------------


class TestPcapRoundTrip(TempDir):
    def test_payloads_and_ns_timestamps_are_exact(self):
        sent = [
            RecordedPacket(T0_NS, SRC, DST, b""),
            RecordedPacket(T0_NS + 1, ("192.168.4.7", 1), DST, b"\x00"),
            RecordedPacket(T0_NS + 999_999_999, SRC, ("0.0.0.0", 5566), tiny_frame(3, 9)),
            RecordedPacket(T0_NS + 10**9, SRC, DST, bytes(range(256)) * 5),
            RecordedPacket(T0_NS + 2 * 10**9, SRC, DST, b"\xff" * 65507),
        ]
        with PcapWriter(self.path()) as w:
            for p in sent:
                w.write(p.t_ns, p.src, p.dst, p.payload)
        got, st = self.read_all(self.path())
        self.assertEqual(got, sent)
        self.assertEqual(st.packets, len(sent))
        self.assertTrue(st.nanosecond)
        self.assertFalse(st.truncated)
        self.assertAlmostEqual(got[1].t_s, (T0_NS + 1) / 1e9)

    def test_global_header_is_nanosecond_little_endian_raw(self):
        with PcapWriter(self.path()):
            pass
        with open(self.path(), "rb") as f:
            head = f.read()
        self.assertEqual(len(head), 24)
        magic, vmaj, vmin, _zone, _sig, snaplen, linktype = struct.unpack("<IHHiIII", head)
        self.assertEqual(magic, PCAP_MAGIC_NS)
        self.assertEqual((vmaj, vmin, snaplen, linktype), (2, 4, 65535, 101))

    def test_synthesized_ip_header_is_valid(self):
        """Wireshark flags a bad IPv4 checksum; UDP checksum 0 is legal."""
        payload = tiny_frame(1, 1)
        with PcapWriter(self.path()) as w:
            w.write(T0_NS, SRC, DST, payload)
        with open(self.path(), "rb") as f:
            data = f.read()[24 + 16 :]
        ip = data[:20]
        words = sum(struct.unpack("!10H", ip))
        while words >> 16:
            words = (words & 0xFFFF) + (words >> 16)
        self.assertEqual(words, 0xFFFF, "IPv4 header checksum does not verify")
        self.assertEqual(ip[0], 0x45)
        self.assertEqual(ip[8], 64)   # TTL
        self.assertEqual(ip[9], 17)   # UDP
        self.assertEqual(struct.unpack("!H", ip[2:4])[0], 28 + len(payload))
        sport, dport, ulen, csum = struct.unpack("!HHHH", data[20:28])
        self.assertEqual((sport, dport, ulen, csum), (SRC[1], DST[1], 8 + len(payload), 0))
        self.assertEqual(data[28:], payload)

    def test_refuses_to_overwrite_a_recording(self):
        with PcapWriter(self.path()) as w:
            w.write(T0_NS, SRC, DST, b"keep me")
        with self.assertRaises(FileExistsError):
            PcapWriter(self.path())
        self.assertEqual(self.read_all(self.path())[0][0].payload, b"keep me")
        with PcapWriter(self.path(), overwrite=True):
            pass
        self.assertEqual(self.read_all(self.path())[0], [])

    def test_rejects_payload_larger_than_an_ipv4_datagram(self):
        with PcapWriter(self.path()) as w:
            with self.assertRaises(ValueError):
                w.write(T0_NS, SRC, DST, b"\x00" * 65508)

    def test_flush_reaches_disk_while_still_open(self):
        """What the recorder relies on so a kill loses at most ~1 s."""
        with PcapWriter(self.path(), flush_interval_s=0.05) as w:
            w.write(T0_NS, SRC, DST, b"first")
            time.sleep(0.06)
            w.flush_if_due()  # the recorder calls this when idle
            got, st = self.read_all(self.path())
            self.assertEqual([p.payload for p in got], [b"first"])
            self.assertFalse(st.truncated)


# -- captures from other tools ---------------------------------------------


class TestForeignCaptures(TempDir):
    PAYLOAD = tiny_frame(5, 77)

    def assert_one(self, path, payload=PAYLOAD, t_ns=T0_NS):
        got, st = self.read_all(path)
        self.assertEqual(len(got), 1, st)
        self.assertEqual(got[0].payload, payload)
        self.assertEqual(got[0].src, SRC)
        self.assertEqual(got[0].dst, DST)
        self.assertEqual(got[0].t_ns, t_ns)
        return st

    def test_microsecond_magic(self):
        path = self.put(pcap_bytes([(T0_NS, ipv4_udp(self.PAYLOAD))], nano=False))
        st = self.assert_one(path, t_ns=T0_NS // 1000 * 1000)
        self.assertFalse(st.nanosecond)

    def test_big_endian_files(self):
        for nano in (False, True):
            with self.subTest(nano=nano):
                path = self.put(
                    pcap_bytes([(T0_NS, ipv4_udp(self.PAYLOAD))], endian=">", nano=nano),
                    name=f"be{nano}.pcap",
                )
                self.assert_one(path, t_ns=T0_NS if nano else T0_NS // 1000 * 1000)

    def test_ethernet_with_vlan_tag_padding_and_fcs(self):
        payload = b"abcd"  # short enough that the frame needs padding
        eth = b"\x02" * 6 + b"\x04" * 6 + b"\x81\x00\x00\x2a" + b"\x08\x00"
        frame = eth + ipv4_udp(payload)
        frame += b"\x00" * max(0, 60 - len(frame)) + b"\xde\xad\xbe\xef"  # pad + FCS
        path = self.put(pcap_bytes([(T0_NS, frame)], linktype=LINKTYPE_ETHERNET))
        self.assert_one(path, payload=payload)

    def test_ethernet_untagged_and_double_tagged(self):
        plain = b"\x02" * 12 + b"\x08\x00" + ipv4_udp(self.PAYLOAD)
        qinq = b"\x02" * 12 + b"\x88\xa8\x00\x01" + b"\x81\x00\x00\x02" + b"\x08\x00"
        qinq += ipv4_udp(self.PAYLOAD)
        for name, frame in (("plain", plain), ("qinq", qinq)):
            with self.subTest(name):
                path = self.put(
                    pcap_bytes([(T0_NS, frame)], linktype=LINKTYPE_ETHERNET), name=name
                )
                self.assert_one(path)

    def test_linux_cooked_v1_and_v2(self):
        """`tcpdump -i any` writes SLL (older) or SLL2 (libpcap >= 1.10)."""
        sll = struct.pack("!HHH8sH", 0, 1, 6, b"\x02" * 6, 0x0800)
        sll2 = struct.pack("!HHIHBB8s", 0x0800, 0, 3, 1, 0, 6, b"\x02" * 6)
        for name, linktype, head in (
            ("sll", LINKTYPE_LINUX_SLL, sll),
            ("sll2", LINKTYPE_LINUX_SLL2, sll2),
        ):
            with self.subTest(name):
                path = self.put(
                    pcap_bytes([(T0_NS, head + ipv4_udp(self.PAYLOAD))], linktype=linktype),
                    name=name,
                )
                self.assert_one(path)

    def test_bsd_loopback_in_either_byte_order_and_raw_ipv4_228(self):
        cases = (
            ("null-le", LINKTYPE_NULL, struct.pack("<I", 2)),
            ("null-be", LINKTYPE_NULL, struct.pack(">I", 2)),
            ("ipv4", LINKTYPE_IPV4, b""),
        )
        for name, linktype, head in cases:
            with self.subTest(name):
                path = self.put(
                    pcap_bytes([(T0_NS, head + ipv4_udp(self.PAYLOAD))], linktype=linktype),
                    name=name,
                )
                self.assert_one(path)

    def test_ip_options_are_skipped_over(self):
        for options in (b"\x01\x01\x01\x00", b"\x94\x04\x00\x00" + b"\x01" * 4):
            with self.subTest(ihl=5 + len(options) // 4):
                path = self.put(
                    pcap_bytes([(T0_NS, ipv4_udp(self.PAYLOAD, options=options))]),
                    name=f"opt{len(options)}",
                )
                self.assert_one(path)

    def test_non_udp_fragments_and_non_ipv4_are_skipped_and_counted(self):
        good = ipv4_udp(self.PAYLOAD)
        ipv6 = b"\x60" + b"\x00" * 47
        records = [
            (T0_NS, ipv4_udp(b"tcp", proto=6)),
            (T0_NS, ipv4_udp(b"icmp", proto=1)),
            (T0_NS, ipv4_udp(self.PAYLOAD, frag=0x2000)),   # first fragment (MF)
            (T0_NS, ipv4_udp(self.PAYLOAD, frag=0x0005)),   # later fragment
            (T0_NS, ipv6),
            (T0_NS, b""),
            (T0_NS, good),
        ]
        path = self.put(pcap_bytes(records))
        st = self.assert_one(path)
        self.assertEqual(
            (st.records, st.not_udp, st.fragments, st.not_ipv4, st.skipped),
            (7, 2, 2, 2, 6),
        )
        arp = b"\x02" * 12 + b"\x08\x06" + b"\x00" * 28
        path = self.put(pcap_bytes([(T0_NS, arp)], linktype=LINKTYPE_ETHERNET), "arp")
        self.assertEqual(self.read_all(path)[1].not_ipv4, 1)

    def test_destination_port_filter(self):
        records = [
            (T0_NS, ipv4_udp(b"dns", dst=("10.0.0.1", 53))),
            (T0_NS, ipv4_udp(self.PAYLOAD)),
        ]
        path = self.put(pcap_bytes(records))
        got, st = self.read_all(path, port=5566)
        self.assertEqual([p.payload for p in got], [self.PAYLOAD])
        self.assertEqual(st.other_port, 1)
        self.assertEqual(len(self.read_all(path)[0]), 2)

    def test_snapped_capture_is_skipped_not_misreported(self):
        """`tcpdump -s 64` keeps a prefix; a cut datagram is not a bad frame."""
        full = ipv4_udp(self.PAYLOAD + b"\x00" * 100)
        path = self.put(pcap_bytes([(T0_NS, full[:64], len(full))]))
        got, st = self.read_all(path)
        self.assertEqual(got, [])
        self.assertEqual(st.snapped, 1)

    def test_truncated_final_record_ends_cleanly(self):
        payloads = [tiny_frame(1, i) for i in range(3)]
        with PcapWriter(self.path("full.pcap")) as w:
            for i, p in enumerate(payloads):
                w.write(T0_NS + i, SRC, DST, p)
        with open(self.path("full.pcap"), "rb") as f:
            full = f.read()
        last_record = 16 + 28 + len(payloads[-1])
        for keep in (10, 16 + 20):  # mid record header, mid packet data
            with self.subTest(keep=keep):
                cut = self.put(full[: len(full) - last_record + keep], f"cut{keep}")
                got, st = self.read_all(cut)
                self.assertEqual([p.payload for p in got], payloads[:2])
                self.assertTrue(st.truncated)
                self.assertEqual(st.truncated_bytes, keep)
                frames = list(iter_frames(cut))
                self.assertEqual(len(frames), 2)

    def test_header_only_file_is_empty_not_truncated(self):
        path = self.put(pcap_bytes([]))
        got, st = self.read_all(path)
        self.assertEqual(got, [])
        self.assertFalse(st.truncated)

    def test_bad_files_raise_with_a_reason(self):
        cases = {
            "pcapng": (b"\x0a\x0d\x0d\x0a" + b"\x00" * 28, "pcapng"),
            "text": (b"hello, this is not a capture file", "not a pcap"),
            "short": (b"\xd4\xc3\xb2\xa1", "not a pcap"),
            "linktype": (pcap_bytes([], linktype=105), "link type 105"),
            "corrupt": (pcap_bytes([]) + struct.pack("<IIII", 1, 0, 10**7, 10**7), "corrupt"),
        }
        for name, (data, reason) in cases.items():
            with self.subTest(name):
                with self.assertRaises(ValueError) as ctx:
                    list(read_pcap(self.put(data, name)))
                self.assertIn(reason, str(ctx.exception))


# -- live UDP --------------------------------------------------------------


class RecorderThread:
    """Runs record_udp on 127.0.0.1:0 and exposes the bound port."""

    def __init__(self, path, **kwargs):
        self.ready = threading.Event()
        self.stop = threading.Event()
        self.addr = None
        self.result = None
        self.error = None

        def on_ready(addr):
            self.addr = addr
            self.ready.set()

        def run():
            try:
                self.result = record_udp(
                    path, port=0, host="127.0.0.1", stop_event=self.stop,
                    on_ready=on_ready, **kwargs,
                )
            except BaseException as exc:  # surfaced by join()
                self.error = exc
                self.ready.set()

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()
        if not self.ready.wait(5.0) or self.error:
            raise RuntimeError(f"recorder did not start: {self.error}")

    def join(self, timeout=10.0):
        self.thread.join(timeout)
        if self.thread.is_alive():
            self.stop.set()
            self.thread.join(2.0)
            raise AssertionError("recorder did not stop on its own")
        if self.error:
            raise self.error
        return self.result


def send_all(payloads, addr):
    sender = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        sender.bind(("127.0.0.1", 0))
        for i, p in enumerate(payloads):
            sender.sendto(p, addr)
            if i % 16 == 15:
                time.sleep(0.002)  # stay well inside the receive buffer
        return sender.getsockname()
    finally:
        sender.close()


class TestLiveLoopback(TempDir):
    def test_records_every_datagram_including_malformed(self):
        frames = synthetic_payloads(40, node_id=3)
        malformed = [b"", b"garbage", frames[0][:25]]
        sent = frames[:20] + malformed + frames[20:]
        rec = RecorderThread(self.path(), max_packets=len(sent))
        before = time.time_ns()
        sender_addr = send_all(sent, rec.addr)
        stats = rec.join()
        after = time.time_ns()

        self.assertEqual(stats.packets, len(sent))
        self.assertEqual(stats.malformed, len(malformed))
        self.assertEqual(stats.nodes, {3: len(frames)})
        self.assertEqual(stats.payload_bytes, sum(map(len, sent)))
        if stats.kernel_drops is not None:
            self.assertEqual(stats.kernel_drops, 0)

        got, st = self.read_all(self.path())
        self.assertEqual([p.payload for p in got], sent)
        self.assertFalse(st.truncated)
        for p in got:
            self.assertEqual(p.src, sender_addr)
            self.assertEqual(p.dst, rec.addr)
            self.assertTrue(before <= p.t_ns <= after)
        self.assertEqual([p.t_ns for p in got], sorted(p.t_ns for p in got))

        fst = ReadStats()
        decoded = [f for _t, f in iter_frames(self.path(), stats=fst)]
        self.assertEqual(decoded, [parse_esp32_frame(p) for p in frames])
        self.assertEqual(fst.malformed, len(malformed))

    def test_stops_on_duration_with_a_valid_empty_file(self):
        start = time.monotonic()
        stats = record_udp(self.path(), port=0, host="127.0.0.1", duration_s=0.3)
        self.assertGreaterEqual(time.monotonic() - start, 0.29)
        self.assertEqual(stats.packets, 0)
        self.assertEqual(self.read_all(self.path())[0], [])

    def test_stop_event(self):
        rec = RecorderThread(self.path())
        send_all([b"one"], rec.addr)
        time.sleep(0.1)
        rec.stop.set()
        self.assertEqual(rec.join(2.0).packets, 1)

    def test_replay_resends_payloads_byte_exact(self):
        payloads = synthetic_payloads(30, node_id=2) + [b"malformed on purpose"]
        with PcapWriter(self.path()) as w:
            for i, p in enumerate(payloads):
                w.write(T0_NS + i * 1_000_000, SRC, DST, p)
        rec = RecorderThread(self.path("again.pcap"), max_packets=len(payloads))
        sent = replay_udp(self.path(), *rec.addr, speed=0)
        rec.join()
        self.assertEqual(sent, len(payloads))
        again, _ = self.read_all(self.path("again.pcap"))
        self.assertEqual([p.payload for p in again], payloads)

    def test_replay_preserves_timing_scaled_by_speed(self):
        with PcapWriter(self.path()) as w:
            for i in range(6):
                w.write(T0_NS + i * 40_000_000, SRC, DST, b"x")  # 200 ms span
        sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sink.bind(("127.0.0.1", 0))
        try:
            elapsed = {}
            for speed in (1.0, 2.0):
                start = time.monotonic()
                replay_udp(self.path(), *sink.getsockname(), speed=speed)
                elapsed[speed] = time.monotonic() - start
            self.assertGreaterEqual(elapsed[1.0], 0.19)
            self.assertGreaterEqual(elapsed[2.0], 0.095)
            self.assertLess(elapsed[2.0], elapsed[1.0])
        finally:
            sink.close()


# -- per-node delivery statistics -------------------------------------------


class TestLinkStats(TempDir):
    def write(self, entries):
        """entries: (t_ns, node_id, sequence[, src_ip]) or (t_ns, raw_bytes)."""
        with PcapWriter(self.path(), overwrite=True) as w:
            for e in entries:
                if isinstance(e[1], bytes):
                    w.write(e[0], SRC, DST, e[1])
                else:
                    src = (e[3], 4000) if len(e) > 3 else SRC
                    w.write(e[0], src, DST, tiny_frame(e[1], e[2]))
        st = ReadStats()
        return link_stats(self.path(), stats=st), st

    def test_rate_and_inter_arrival_from_timestamps(self):
        links, _ = self.write([(T0_NS + i * 50_000_000, 1, i) for i in range(200)])
        ls = links[1]
        self.assertEqual(ls.packets, 200)
        self.assertAlmostEqual(ls.duration_s, 199 * 0.05)
        self.assertAlmostEqual(ls.rate_hz, 20.0)
        self.assertAlmostEqual(ls.dt_median_ms, 50.0)
        self.assertAlmostEqual(ls.dt_p95_ms, 50.0)
        self.assertEqual((ls.lost, ls.loss_pct, ls.resets), (0, 0.0, 0))

    def test_jitter_shows_in_p95_not_median(self):
        t, entries = T0_NS, []
        for i in range(101):
            entries.append((t, 1, i))
            t += 45_000_000 if i % 10 else 95_000_000  # every 10th gap is long
        ls = self.write(entries)[0][1]
        self.assertAlmostEqual(ls.dt_median_ms, 45.0)
        self.assertAlmostEqual(ls.dt_p95_ms, 95.0)
        self.assertAlmostEqual(ls.dt_max_ms, 95.0)

    def test_sequence_gap_is_loss(self):
        entries = [(T0_NS + i * 50_000_000, 1, i) for i in range(200) if not 50 <= i < 55]
        ls = self.write(entries)[0][1]
        self.assertEqual(ls.packets, 195)
        self.assertEqual(ls.lost, 5)
        self.assertAlmostEqual(ls.loss_pct, 2.5)
        self.assertEqual(ls.resets, 0)
        self.assertAlmostEqual(ls.dt_max_ms, 300.0)  # the outage, as seen

    def test_reboot_is_a_reset_not_loss(self):
        first = [(T0_NS + i * 50_000_000, 1, 1000 + i) for i in range(100)]
        t_boot = T0_NS + 100 * 50_000_000 + 2_000_000_000
        second = [(t_boot + i * 50_000_000, 1, i) for i in range(100)]
        ls = self.write(first + second)[0][1]
        self.assertEqual((ls.resets, ls.lost, ls.loss_pct), (1, 0, 0.0))
        # A boot loop that dies after a few packets is still a reset.
        short = [(T0_NS + i * 50_000_000, 1, i % 4) for i in range(12)]
        ls = self.write(short)[0][1]
        self.assertEqual((ls.resets, ls.lost), (2, 0))
        # Firmware that seeds its counter arbitrarily: an implausible forward
        # jump (days of packets) is a reset too, not a 50M-packet loss.
        ls = self.write([(T0_NS + i, 1, s) for i, s in enumerate([0, 1, 50_000_000, 50_000_001])])[0][1]
        self.assertEqual((ls.resets, ls.lost), (1, 0))

    def test_u32_wrap_is_not_loss(self):
        start = 0xFFFFFFFF - 9
        entries = [(T0_NS + i * 50_000_000, 1, (start + i) & 0xFFFFFFFF) for i in range(30)]
        ls = self.write(entries)[0][1]
        self.assertEqual((ls.lost, ls.resets), (0, 0))
        # ...but a gap straddling the wrap is counted exactly.
        seqs = [0xFFFFFFFD, 0xFFFFFFFE, 2, 3]  # 0xFFFFFFFF, 0, 1 missing
        ls = self.write([(T0_NS + i, 1, s) for i, s in enumerate(seqs)])[0][1]
        self.assertEqual((ls.lost, ls.resets), (3, 0))

    def test_duplicates_and_late_packets_are_not_loss_or_resets(self):
        seqs = [0, 1, 2, 4, 3, 5, 5, 6, 7]
        ls = self.write([(T0_NS + i, 1, s) for i, s in enumerate(seqs)])[0][1]
        self.assertEqual(
            (ls.lost, ls.resets, ls.duplicates, ls.reordered), (0, 0, 1, 1)
        )
        self.assertEqual(ls.loss_pct, 0.0)

    def test_nodes_are_separate_and_malformed_is_counted_per_file(self):
        entries = []
        for i in range(40):
            entries.append((T0_NS + i * 50_000_000, 1, i))
            if i % 2 == 0:
                entries.append((T0_NS + i * 50_000_000 + 1_000, 2, i // 2))
        entries.insert(10, (T0_NS + 9 * 50_000_000 + 5, b"not a frame"))
        entries.insert(20, (T0_NS + 18 * 50_000_000 + 5, tiny_frame(1, 0)[:22]))
        links, st = self.write(entries)
        self.assertEqual(sorted(links), [1, 2])
        self.assertAlmostEqual(links[1].rate_hz, 20.0)
        self.assertAlmostEqual(links[2].rate_hz, 10.0)
        self.assertEqual((links[1].lost, links[2].lost), (0, 0))
        self.assertEqual(st.malformed, 2)
        self.assertEqual(links[1].subcarriers, (2,))

    def test_two_boards_with_one_node_id_are_visible(self):
        entries = [(T0_NS + i, 7, i, f"10.0.0.{50 + i % 2}") for i in range(10)]
        ls = self.write(entries)[0][7]
        self.assertEqual(ls.sources, ("10.0.0.50", "10.0.0.51"))


# -- the reason this module exists ------------------------------------------


class TestReplayDeterminism(TempDir):
    def test_pipeline_from_recording_matches_pipeline_on_same_frames(self):
        """Every real-data bug is reproducible from a recording: the pipeline
        must see exactly the same frames, in the same order, with nothing
        lost or altered by the capture, and malformed or foreign datagrams
        in the file must not perturb it."""
        fs = 20.0
        payloads = synthetic_payloads(700, node_id=4, rate_hz=fs, seed=11)
        lost = {120, 121, 400}
        kept = [p for i, p in enumerate(payloads) if i not in lost]

        with PcapWriter(self.path()) as w:
            for i, p in enumerate(kept):
                t = T0_NS + i * 50_000_000 + (i * 7919 % 5) * 1_000_000  # jitter
                w.write(t, SRC, DST, p)
                if i == 200:
                    w.write(t + 1, ("10.0.0.99", 9), DST, b"\x01\x00\x11\xc5 junk")
                if i % 50 == 0:
                    w.write(t + 2, SRC, DST, tiny_frame(9, i))

        def run(frames):
            pipeline = SensingPipeline(
                n_subcarriers=56, sample_rate_hz=fs, calibration_s=5.0,
                suppress_vitals_when_empty=False,
            )
            return [pipeline.process(f.to_csi_frame(fs)) for f in frames]

        direct = run(parse_esp32_frame(p) for p in kept)
        st = ReadStats()
        replayed = run(f for _t, f in iter_frames(self.path(), stats=st) if f.node_id == 4)

        self.assertEqual(len(replayed), len(kept))
        self.assertEqual(replayed, direct)
        self.assertEqual(st.malformed, 1)
        self.assertTrue(
            any(r.respiratory_rate.status.value != "unavailable" for r in direct),
            "test never reached the vitals path",
        )


# -- CLI --------------------------------------------------------------------


class TestCli(TempDir):
    def make_recording(self):
        """Node 1 at 20 Hz (3 frames lost), node 2 at 10 Hz, one malformed."""
        node1 = synthetic_payloads(400, node_id=1, occupied=False, seed=3)
        node2 = synthetic_payloads(200, node_id=2, occupied=False, seed=4)
        with PcapWriter(self.path()) as w:
            for i, p in enumerate(node1):
                t = T0_NS + i * 50_000_000
                if not 100 <= i < 103:
                    w.write(t, ("10.0.0.11", 4000), DST, p)
                if i % 2 == 0:
                    w.write(t + 7_000_000, ("10.0.0.12", 4000), DST, node2[i // 2])
                if i == 300:
                    w.write(t + 9_000_000, ("10.0.0.13", 4000), DST, b"garbage")
        return self.path()

    def cli(self, *argv):
        out, self.stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(self.stderr):
            rc = cli_main(list(argv))
        return rc, out.getvalue()

    def test_stats(self):
        rc, out = self.cli("stats", self.make_recording())
        self.assertEqual(rc, 0)
        rows = {
            line.split()[0]: line.split()
            for line in out.splitlines()
            if line.strip()[:1].isdigit()
        }
        node, packets, _span, rate, lost, loss_pct, resets = rows["1"][:7]
        self.assertEqual((packets, rate, lost, loss_pct, resets), ("397", "19.85", "3", "0.75", "0"))
        self.assertEqual(rows["2"][1], "200")
        self.assertEqual(rows["2"][4:7], ["0", "0.00", "0"])
        self.assertIn("1 malformed", out)
        self.assertIn("truncated tail: no", out)
        self.assertIn("resets", out)

    def test_replay_runs_pipeline_on_busiest_node_at_measured_rate(self):
        rc, out = self.cli("replay", self.make_recording(), "--calibration-s", "2")
        self.assertEqual(rc, 0)
        self.assertIn("node 1 (397 frames", out)
        self.assertIn("rate: 20.000 Hz measured", out)
        printed = [line for line in out.splitlines() if line.startswith("rec=")]
        self.assertEqual(len(printed), 20)  # once per recorded second
        self.assertIn("1 malformed", out)

    def test_replay_explicit_node_and_rate(self):
        rc, out = self.cli(
            "replay", self.make_recording(), "--node", "2", "--rate", "10",
            "--calibration-s", "2",
        )
        self.assertEqual(rc, 0)
        self.assertIn("rate: 10 Hz (from --rate", out)
        self.assertIn("replayed 200 frames from node 2", out)
        rc, _ = self.cli("replay", self.path(), "--node", "42")
        self.assertEqual(rc, 1)
        self.assertIn("nodes present: 1, 2", self.stderr.getvalue())

    def test_replay_send_over_udp(self):
        path = self.make_recording()
        expected = [p.payload for p in read_pcap(path)]
        sink = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sink.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1 << 20)
        sink.bind(("127.0.0.1", 0))
        sink.settimeout(2.0)
        try:
            host, port = sink.getsockname()
            rc, out = self.cli("replay", path, "--send", f"{host}:{port}", "--speed", "0")
            self.assertEqual(rc, 0)
            self.assertIn(f"sent {len(expected)} datagrams", out)
            got = [sink.recv(65535) for _ in expected]
        finally:
            sink.close()
        self.assertEqual(got, expected)

    def test_record_with_duration_and_overwrite_guard(self):
        out_path = self.path("cli.pcap")
        args = ("record", "--host", "127.0.0.1", "--port", "0", "--out", out_path,
                "--duration-s", "0.2")
        rc, out = self.cli(*args)
        self.assertEqual(rc, 0)
        self.assertIn("stopped: 0 packets", out)
        rc, _ = self.cli(*args)
        self.assertEqual(rc, 1)
        self.assertIn("--overwrite", self.stderr.getvalue())
        self.assertEqual(self.cli(*args, "--overwrite")[0], 0)


if __name__ == "__main__":
    unittest.main()
