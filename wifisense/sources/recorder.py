"""Raw UDP packet recorder and replayer.

Every bug found on real hardware has to be reproducible offline, so the
gateway records *datagrams*, not parsed frames. A datagram the parser
rejects today (bad magic, truncated payload, a firmware build with a
different header) is exactly the evidence needed to fix the parser
tomorrow, so malformed datagrams are recorded too. The same recordings are
what achieved per-node rate, loss and jitter are measured from
(CAMPUS_SENSING_PROMPT.md §3.4, §5.2).

File format: classic libpcap, not pcapng
----------------------------------------
Recordings open directly in Wireshark, and the reader accepts what
`tcpdump -w` writes, so a capture taken with stock tools on a gateway or a
mirror port replays here unchanged.

The writer emits nanosecond-resolution, little-endian pcap (magic
0xA1B23C4D, v2.4, snaplen 65535) with LINKTYPE_RAW. Each record is a
synthesized IPv4 + UDP header around the payload, so the sender's address
and port survive. The IPv4 header checksum is correct; the UDP checksum is
zero, which IPv4 defines as "not computed". The original link layer is not
kept: it never reaches a UDP socket.

The reader accepts µs and ns files in either byte order, with the link
types a gateway capture is likely to have: Ethernet (with 802.1Q/802.1ad
tags), raw IPv4 (101, 228), Linux cooked v1/v2 (113, 276, which is what
`tcpdump -i any` writes) and BSD loopback (0). It skips, and counts,
anything that is not an unfragmented IPv4 UDP datagram. The kernel
reassembles fragments before a socket sees them, but tcpdump captures them
raw; reassembling here is not worth the code for ~150-byte CSI frames.

Timestamps
----------
pcap stores wall-clock (epoch) time, so records are stamped with
`time.time_ns()` the moment the receive call returns. Every rate and jitter
figure is computed from *differences* between timestamps, so a wall-clock
step during a recording (NTP correcting the clock) shows up as one bogus
inter-arrival gap. That is a known caveat; run chrony/ntpd in slew mode on
the gateway. The stamp is taken in user space and includes Python
scheduling latency (typically well under 1 ms). Packets that queued while
the recorder was busy get stamped close together, which inflates the p95
jitter figure rather than hiding it.

Crash safety
------------
The writer flushes at least once a second, including while idle, so
killing the recorder loses at most ~1 s of packets. The reader treats a
half-written final record as the end of the file rather than an error.
There is no fsync: a power cut on the gateway can lose more than that,
depending on the filesystem.
"""

from __future__ import annotations

import math
import os
import socket
import struct
import threading
import time
from array import array
from dataclasses import dataclass, field
from typing import Callable, Iterator

from .esp32_udp import Esp32Frame, parse_esp32_frame

__all__ = [
    "RecordedPacket",
    "ReadStats",
    "PcapWriter",
    "read_pcap",
    "iter_frames",
    "RecordStats",
    "record_udp",
    "replay_udp",
    "LinkStats",
    "link_stats",
]

PCAP_MAGIC_US = 0xA1B2C3D4
PCAP_MAGIC_NS = 0xA1B23C4D
_PCAPNG_MAGIC = 0x0A0D0D0A

LINKTYPE_NULL = 0
LINKTYPE_ETHERNET = 1
LINKTYPE_RAW = 101
LINKTYPE_LINUX_SLL = 113
LINKTYPE_IPV4 = 228
LINKTYPE_LINUX_SLL2 = 276
_SUPPORTED_LINKTYPES = (
    LINKTYPE_NULL,
    LINKTYPE_ETHERNET,
    LINKTYPE_RAW,
    LINKTYPE_LINUX_SLL,
    LINKTYPE_IPV4,
    LINKTYPE_LINUX_SLL2,
)

SNAPLEN = 65535
MAX_UDP_PAYLOAD = 65535 - 20 - 8  # largest payload one IPv4 datagram can carry

_GLOBAL_HEADER = struct.Struct("<IHHiIII")
_RECORD_HEADER = struct.Struct("<IIII")
# version/IHL, TOS, total length, id, flags/fragment, TTL, protocol,
# header checksum, src, dst | UDP: sport, dport, length, checksum
_IPV4_UDP = struct.Struct("!BBHHHBBH4s4sHHHH")
_UDP = struct.Struct("!HHH")
# libpcap's own upper bound on a record. Anything larger is not a capture
# that was cut short, it is a corrupt length field, and there is no way to
# resynchronise after one.
_MAX_RECORD = 262144
_ETHERTYPE_IPV4 = 0x0800
_VLAN_TPIDS = (0x8100, 0x88A8, 0x9100)
_AF_INET = 2
_U32 = 0xFFFFFFFF

# One receive buffer large enough for any IPv4 datagram, so nothing is
# ever silently truncated by the recorder itself.
_RECV_BUFSIZE = 65535
# How often the receive loop wakes up with nothing to do, to notice a stop
# request, a deadline, or a pending flush.
_POLL_S = 0.2
# Ask for a large socket buffer so a stall in Python (GC, slow disk) is
# absorbed by the kernel rather than dropped. Linux caps this at
# net.core.rmem_max; raise that sysctl for big meshes.
_RCVBUF_BYTES = 4 << 20


# --------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class RecordedPacket:
    """One UDP datagram exactly as a socket received it.

    `t_ns` is the gateway's wall-clock receive time in epoch nanoseconds.
    """

    t_ns: int
    src: tuple[str, int]
    dst: tuple[str, int]
    payload: bytes

    @property
    def t_s(self) -> float:
        """Receive time in epoch seconds.

        A float carries about 0.2 µs of resolution at today's epoch; use
        `t_ns` when exact arithmetic matters.
        """
        return self.t_ns / 1e9


def _ipv4_checksum(header: bytes | bytearray) -> int:
    """RFC 791 header checksum: one's complement of the one's complement sum."""
    total = sum(struct.unpack(f"!{len(header) // 2}H", header))
    while total >> 16:
        total = (total & 0xFFFF) + (total >> 16)
    return ~total & 0xFFFF


class PcapWriter:
    """Append UDP datagrams to a new nanosecond pcap file.

    Refuses to overwrite an existing file unless `overwrite=True`: a
    recording is often the only copy of a session that cannot be repeated,
    and re-running yesterday's command line should not destroy it.

    Data is flushed to the OS at least every `flush_interval_s`, both on
    `write` and on `flush_if_due` (which the recorder calls while idle), so
    a killed process loses at most that much.
    """

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        overwrite: bool = False,
        flush_interval_s: float = 1.0,
    ) -> None:
        self.path = os.fspath(path)
        self.flush_interval_s = flush_interval_s
        self.packets = 0
        self._ip_id = 0
        self._f = open(self.path, "wb" if overwrite else "xb")
        self._f.write(
            _GLOBAL_HEADER.pack(PCAP_MAGIC_NS, 2, 4, 0, 0, SNAPLEN, LINKTYPE_RAW)
        )
        self._f.flush()
        self._last_flush = time.monotonic()

    def write(
        self,
        t_ns: int,
        src: tuple[str, int],
        dst: tuple[str, int],
        payload: bytes,
    ) -> None:
        """Append one datagram received at epoch time `t_ns`."""
        if len(payload) > MAX_UDP_PAYLOAD:
            raise ValueError(
                f"payload of {len(payload)} bytes does not fit one IPv4 datagram"
            )
        total = 28 + len(payload)
        headers = bytearray(
            _IPV4_UDP.pack(
                0x45,  # IPv4, 20-byte header
                0,
                total,
                self._ip_id,
                0x4000,  # don't-fragment: the record is a whole datagram
                64,
                17,  # UDP
                0,  # checksum, filled in below
                socket.inet_pton(socket.AF_INET, src[0]),
                socket.inet_pton(socket.AF_INET, dst[0]),
                src[1],
                dst[1],
                8 + len(payload),
                0,  # UDP checksum 0 = not computed (legal for IPv4)
            )
        )
        headers[10:12] = _ipv4_checksum(headers[:20]).to_bytes(2, "big")
        sec, nsec = divmod(t_ns, 1_000_000_000)
        self._f.write(_RECORD_HEADER.pack(sec, nsec, total, total) + headers + payload)
        self._ip_id = (self._ip_id + 1) & 0xFFFF
        self.packets += 1
        self.flush_if_due()

    def flush_if_due(self) -> None:
        """Flush if `flush_interval_s` has passed since the last flush."""
        if time.monotonic() - self._last_flush >= self.flush_interval_s:
            self.flush()

    def flush(self) -> None:
        self._f.flush()
        self._last_flush = time.monotonic()

    def close(self) -> None:
        if not self._f.closed:
            self._f.close()

    def __enter__(self) -> PcapWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


# --------------------------------------------------------------------------
# Reading
# --------------------------------------------------------------------------


@dataclass
class ReadStats:
    """What a pass over a capture saw, including everything it skipped.

    Pass one in to `read_pcap`, `iter_frames`, `link_stats` or
    `replay_udp` and it is filled in as the file is read.
    """

    records: int = 0
    """pcap records read."""
    packets: int = 0
    """UDP datagrams delivered to the caller."""
    not_ipv4: int = 0
    """Records that are not a parseable IPv4 packet (ARP, IPv6, ...)."""
    not_udp: int = 0
    fragments: int = 0
    """IPv4 fragments. Not reassembled; see the module docstring."""
    other_port: int = 0
    """UDP datagrams to a port other than the requested one."""
    snapped: int = 0
    """Datagrams the capture holds fewer bytes of than the datagram has
    (tcpdump run with a small -s snaplen, or inconsistent lengths)."""
    malformed: int = 0
    """Datagrams that failed `parse_esp32_frame` (frame-level readers only)."""
    truncated: bool = False
    """The file ends part-way through a record (recorder killed mid-write)."""
    truncated_bytes: int = 0
    linktype: int | None = None
    nanosecond: bool = False

    @property
    def skipped(self) -> int:
        """Records that were not delivered as UDP datagrams."""
        return (
            self.not_ipv4
            + self.not_udp
            + self.fragments
            + self.other_port
            + self.snapped
        )


def _open_header(head: bytes, path: str) -> tuple[str, int]:
    """Return (struct byte-order prefix, ns per fractional-second unit)."""
    if len(head) < 24:
        raise ValueError(f"{path}: not a pcap file (only {len(head)} bytes)")
    for endian in ("<", ">"):
        (magic,) = struct.unpack_from(endian + "I", head)
        if magic == PCAP_MAGIC_US:
            return endian, 1000
        if magic == PCAP_MAGIC_NS:
            return endian, 1
    (magic,) = struct.unpack_from("<I", head)
    if magic == _PCAPNG_MAGIC:
        raise ValueError(
            f"{path}: pcapng is not supported; convert it with "
            f"`editcap -F pcap in.pcapng out.pcap` or capture with `tcpdump -w`"
        )
    raise ValueError(f"{path}: not a pcap file (magic 0x{magic:08X})")


def _ipv4_offset(linktype: int, data: bytes) -> int | None:
    """Offset of the IPv4 header inside one record, or None if not IPv4."""
    if linktype in (LINKTYPE_RAW, LINKTYPE_IPV4):
        return 0
    if linktype == LINKTYPE_ETHERNET:
        offset = 12
        for _ in range(3):  # at most two stacked VLAN tags, then the type
            if len(data) < offset + 2:
                return None
            ethertype = int.from_bytes(data[offset : offset + 2], "big")
            if ethertype not in _VLAN_TPIDS:
                return offset + 2 if ethertype == _ETHERTYPE_IPV4 else None
            offset += 4
        return None
    if linktype == LINKTYPE_LINUX_SLL:
        if len(data) < 16:
            return None
        return 16 if int.from_bytes(data[14:16], "big") == _ETHERTYPE_IPV4 else None
    if linktype == LINKTYPE_LINUX_SLL2:
        if len(data) < 20:
            return None
        return 20 if int.from_bytes(data[0:2], "big") == _ETHERTYPE_IPV4 else None
    if linktype == LINKTYPE_NULL:
        # The address family is in the *capturing host's* byte order,
        # which need not match the file's, so accept either.
        if len(data) < 4:
            return None
        family = data[:4]
        if _AF_INET in (
            int.from_bytes(family, "little"),
            int.from_bytes(family, "big"),
        ):
            return 4
        return None
    return None


def read_pcap(
    path: str | os.PathLike[str],
    port: int | None = None,
    stats: ReadStats | None = None,
) -> Iterator[RecordedPacket]:
    """Yield every unfragmented IPv4 UDP datagram in a pcap file.

    `port` keeps only datagrams sent *to* that UDP port, which is what you
    want for a tcpdump capture that also caught DNS, mDNS and the rest.
    Everything skipped is counted in `stats`. A truncated final record ends
    iteration cleanly and sets `stats.truncated`; a record whose length
    field is impossible raises `ValueError`, because nothing after it can
    be trusted.

    IP header checksums are deliberately not verified: captures taken on
    the sending host routinely carry garbage checksums (checksum offload).
    """
    st = stats if stats is not None else ReadStats()
    path = os.fspath(path)
    with open(path, "rb") as f:
        head = f.read(24)
        endian, frac_ns = _open_header(head, path)
        _magic, _vmaj, _vmin, _zone, _sigfigs, _snaplen, network = struct.unpack(
            endian + "IHHiIII", head
        )
        # The top bits of this field can carry FCS-length information.
        linktype = network & 0x03FFFFFF
        if linktype not in _SUPPORTED_LINKTYPES:
            raise ValueError(f"{path}: unsupported pcap link type {linktype}")
        st.linktype = linktype
        st.nanosecond = frac_ns == 1
        record = struct.Struct(endian + "IIII")

        while True:
            offset = f.tell()
            hdr = f.read(16)
            if not hdr:
                return
            if len(hdr) < 16:
                st.truncated = True
                st.truncated_bytes = len(hdr)
                return
            sec, frac, incl_len, _orig_len = record.unpack(hdr)
            if incl_len > _MAX_RECORD:
                raise ValueError(
                    f"{path}: corrupt record at byte {offset}: "
                    f"length {incl_len} exceeds {_MAX_RECORD}"
                )
            data = f.read(incl_len)
            if len(data) < incl_len:
                st.truncated = True
                st.truncated_bytes = 16 + len(data)
                return
            st.records += 1

            ip_at = _ipv4_offset(linktype, data)
            if ip_at is None or len(data) < ip_at + 20 or data[ip_at] >> 4 != 4:
                st.not_ipv4 += 1
                continue
            ihl = (data[ip_at] & 0x0F) * 4
            if ihl < 20 or len(data) < ip_at + ihl:
                st.not_ipv4 += 1
                continue
            if int.from_bytes(data[ip_at + 6 : ip_at + 8], "big") & 0x3FFF:
                st.fragments += 1  # more-fragments flag or non-zero offset
                continue
            if data[ip_at + 9] != 17:
                st.not_udp += 1
                continue

            # Ethernet pads short frames and may carry an FCS, so trust the
            # IP total length over the record length where it is set.
            ip_total = int.from_bytes(data[ip_at + 2 : ip_at + 4], "big")
            ip_end = len(data) if ip_total == 0 else min(len(data), ip_at + ip_total)
            udp_at = ip_at + ihl
            if ip_end < udp_at + 8:
                st.snapped += 1
                continue
            sport, dport, udp_len = _UDP.unpack_from(data, udp_at)
            if port is not None and dport != port:
                st.other_port += 1
                continue
            payload_end = udp_at + udp_len
            if udp_len < 8 or payload_end > ip_end:
                st.snapped += 1
                continue

            st.packets += 1
            yield RecordedPacket(
                t_ns=sec * 1_000_000_000 + frac * frac_ns,
                src=(socket.inet_ntoa(data[ip_at + 12 : ip_at + 16]), sport),
                dst=(socket.inet_ntoa(data[ip_at + 16 : ip_at + 20]), dport),
                payload=data[udp_at + 8 : payload_end],
            )


def iter_frames(
    path: str | os.PathLike[str],
    port: int | None = None,
    stats: ReadStats | None = None,
) -> Iterator[tuple[float, Esp32Frame]]:
    """Yield `(receive time in epoch seconds, frame)` for every valid raw-CSI
    datagram, counting the rest in `stats.malformed`.

    This is the offline twin of `listen_udp`: same parser, same skip rule,
    so a pipeline fed from here sees exactly what it would have seen live.
    """
    st = stats if stats is not None else ReadStats()
    for pkt in read_pcap(path, port, st):
        try:
            frame = parse_esp32_frame(pkt.payload)
        except ValueError:
            st.malformed += 1
            continue
        yield pkt.t_s, frame


# --------------------------------------------------------------------------
# Live recording and replay
# --------------------------------------------------------------------------


@dataclass
class RecordStats:
    """What `record_udp` captured. Live-updated while recording."""

    path: str
    bound: tuple[str, int]
    packets: int = 0
    payload_bytes: int = 0
    malformed: int = 0
    """Datagrams recorded that do not parse as raw-CSI (still on disk)."""
    nodes: dict[int, int] = field(default_factory=dict)
    """Valid frames per node_id."""
    elapsed_s: float = 0.0
    kernel_drops: int | None = None
    """Datagrams the kernel discarded because the recorder's socket buffer
    was full (Linux only; None where it cannot be read). Non-zero means the
    gateway lost packets that the radio delivered, and the loss figures
    from `link_stats` will overstate the RF loss by this much."""


def _kernel_drops(sock: socket.socket) -> int | None:
    """Read this socket's drop counter from /proc/net/udp (Linux only)."""
    try:
        inode = str(os.fstat(sock.fileno()).st_ino)
        with open("/proc/net/udp", encoding="ascii") as f:
            next(f, None)
            for line in f:
                cols = line.split()
                if len(cols) > 12 and cols[9] == inode:
                    return int(cols[12])
    except (OSError, ValueError):
        return None
    return None


def _enable_pktinfo(sock: socket.socket) -> bool:
    """Ask the kernel to report each datagram's destination address.

    A socket bound to 0.0.0.0 cannot otherwise tell a unicast datagram from
    a broadcast one, and that difference matters: 802.11 sends broadcast at
    a low basic rate with no retries, so a node misconfigured to broadcast
    shows far worse loss. Recording the real destination makes that
    visible in the capture.
    """
    opt = getattr(socket, "IP_PKTINFO", None)
    if opt is None:
        return False
    try:
        sock.setsockopt(socket.IPPROTO_IP, opt, 1)
    except OSError:
        return False
    return True


def _receive(
    sock: socket.socket, pktinfo: bool, bound_ip: str
) -> tuple[bytes, tuple[str, int], str]:
    if pktinfo:
        data, ancdata, _flags, src = sock.recvmsg(
            _RECV_BUFSIZE, socket.CMSG_SPACE(12)
        )
        dst_ip = bound_ip
        for level, ctype, cdata in ancdata:
            # struct in_pktinfo { int ifindex; in_addr spec_dst; in_addr addr; }
            if level == socket.IPPROTO_IP and ctype == socket.IP_PKTINFO:
                if len(cdata) >= 12:
                    dst_ip = socket.inet_ntoa(cdata[8:12])
        return data, src, dst_ip
    data, src = sock.recvfrom(_RECV_BUFSIZE)
    return data, src, bound_ip


def record_udp(
    path: str | os.PathLike[str],
    port: int = 5566,
    host: str = "0.0.0.0",
    duration_s: float | None = None,
    max_packets: int | None = None,
    stop_event: threading.Event | None = None,
    on_ready: Callable[[tuple[str, int]], None] | None = None,
    on_progress: Callable[[RecordStats], None] | None = None,
    progress_interval_s: float = 5.0,
    overwrite: bool = False,
) -> RecordStats:
    """Record every datagram arriving on `host:port` to a new pcap file.

    Stops after `duration_s`, after `max_packets`, or within ~0.2 s of
    `stop_event` being set, whichever comes first; with none of them it
    records until interrupted. A KeyboardInterrupt still closes the file
    cleanly before propagating, but you lose the returned stats, so prefer
    `stop_event` for a graceful stop.

    `on_ready(bound_addr)` is called once the socket is bound and the file
    is open, which lets a caller bind port 0 and learn the real port.
    `on_progress(stats)` is called every `progress_interval_s`.

    Malformed datagrams are recorded like any other; they are only counted
    here. The socket deliberately does *not* set SO_REUSEADDR: with it,
    Linux would let `wifisense listen` bind the same port and silently
    split the datagrams between the two processes, which would read as
    packet loss. Without it the second bind fails loudly.
    """
    path = os.fspath(path)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RCVBUF_BYTES)
        except OSError:
            pass
        sock.bind((host, port))
        sock.settimeout(_POLL_S)
        pktinfo = _enable_pktinfo(sock)
        bound = sock.getsockname()
        bound = (bound[0], bound[1])
        stats = RecordStats(path=path, bound=bound)
        drops_at_start = _kernel_drops(sock)

        def refresh(now: float) -> None:
            stats.elapsed_s = now - start
            drops = _kernel_drops(sock)
            if drops is not None and drops_at_start is not None:
                stats.kernel_drops = drops - drops_at_start

        with PcapWriter(path, overwrite=overwrite) as writer:
            start = time.monotonic()
            deadline = None if duration_s is None else start + duration_s
            next_progress = start + progress_interval_s
            if on_ready is not None:
                on_ready(bound)

            while True:
                now = time.monotonic()
                if stop_event is not None and stop_event.is_set():
                    break
                if max_packets is not None and stats.packets >= max_packets:
                    break
                if deadline is not None and now >= deadline:
                    break
                if on_progress is not None and now >= next_progress:
                    refresh(now)
                    on_progress(stats)
                    next_progress = now + progress_interval_s

                try:
                    payload, src, dst_ip = _receive(sock, pktinfo, bound[0])
                except (TimeoutError, socket.timeout):
                    writer.flush_if_due()
                    continue
                t_ns = time.time_ns()
                writer.write(t_ns, (src[0], src[1]), (dst_ip, bound[1]), payload)

                stats.packets += 1
                stats.payload_bytes += len(payload)
                try:
                    frame = parse_esp32_frame(payload)
                except ValueError:
                    stats.malformed += 1
                else:
                    stats.nodes[frame.node_id] = stats.nodes.get(frame.node_id, 0) + 1

        refresh(time.monotonic())
        return stats
    finally:
        sock.close()


def replay_udp(
    path: str | os.PathLike[str],
    host: str,
    port: int,
    speed: float = 1.0,
    filter_port: int | None = None,
    stats: ReadStats | None = None,
) -> int:
    """Re-send every recorded payload to `host:port`; return how many were sent.

    Inter-arrival timing is preserved, divided by `speed`; `speed=0` sends
    as fast as possible (expect the receiver to drop some if its socket
    buffer is small). Malformed datagrams are re-sent too, which is the
    point: the live path must survive them. `filter_port` keeps only
    datagrams that were originally sent to that port.

    The schedule is anchored to the first packet rather than accumulated
    sleep by sleep, so timing errors do not drift. A forward wall-clock
    step in the recording replays as a pause of the same length.
    """
    if speed < 0:
        raise ValueError(f"speed must be >= 0, got {speed}")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        try:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        except OSError:
            pass
        sent = 0
        t0_rec: int | None = None
        t0_wall = 0.0
        for pkt in read_pcap(path, filter_port, stats):
            if speed > 0:
                if t0_rec is None:
                    t0_rec = pkt.t_ns
                    t0_wall = time.monotonic()
                delay = t0_wall + (pkt.t_ns - t0_rec) / 1e9 / speed - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
            sock.sendto(pkt.payload, (host, port))
            sent += 1
        return sent
    finally:
        sock.close()


# --------------------------------------------------------------------------
# Per-node delivery statistics
# --------------------------------------------------------------------------

# A backwards sequence step is a late (reordered) packet only if it fills a
# gap at most this many packets behind; anything else is a reset.
_REORDER_WINDOW = 64
# A forward jump larger than this is not loss: at 20 Hz it is over nine days
# of packets. Firmware that seeds its counter arbitrarily after a reboot is
# the plausible cause, so count it as a reset.
_MAX_FORWARD_GAP = 1 << 24


@dataclass
class LinkStats:
    """Achieved delivery for one node, as measured at the gateway.

    Rates and inter-arrival times come from receive timestamps, so they
    describe what the gateway actually got, which is the number that
    decides whether a link can sustain vitals (>= 10 Hz, §3.4). Loss comes
    from raw-CSI sequence gaps (u32, wrap-aware). A sequence that jumps
    backwards without filling a recent gap is counted as a *reset* (node
    reboot), not as loss or as negative loss. Packets lost immediately
    before a reset, or before the first packet of the file, are invisible.

    Keyed by `node_id` for now. Once the multi-link header carries `tx_id`,
    the key becomes the (rx, tx) link.
    """

    node_id: int
    packets: int = 0
    duration_s: float = 0.0
    rate_hz: float = 0.0
    lost: int = 0
    loss_pct: float = 0.0
    resets: int = 0
    duplicates: int = 0
    reordered: int = 0
    dt_median_ms: float = math.nan
    dt_p95_ms: float = math.nan
    dt_max_ms: float = math.nan
    subcarriers: tuple[int, ...] = ()
    """Distinct subcarrier counts seen, in order of first appearance."""
    sources: tuple[str, ...] = ()
    """Sender IPs. More than one usually means two boards flashed with the
    same node_id, whose interleaved sequences read as constant resets."""
    gain_frames: int = 0
    """Frames carrying a receiver gain (header bytes 18-19 not both zero,
    spec/wire.md §1). 0 means the firmware did not report one."""
    agc_moves: int = 0
    """Received frames whose AGC gain differed from the frame received
    before it, in arrival order (PROTOCOL_S5 §1.1). A lost frame between
    two received ones can hide a move and its reversal."""
    fft_moves: int = 0
    agc_min: int | None = None
    agc_max: int | None = None


def _percentile(sorted_values: list[int], q: float) -> float:
    """Linear-interpolated percentile (numpy's default method)."""
    if not sorted_values:
        return math.nan
    pos = (len(sorted_values) - 1) * q / 100.0
    lo = int(pos)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (pos - lo)


class _LinkAccumulator:
    """Streaming sequence/timing bookkeeping for one node."""

    def __init__(self, node_id: int) -> None:
        self.node_id = node_id
        self.packets = 0
        self.first_t_ns = 0
        self.last_t_ns = 0
        self.prev_seq: int | None = None
        self.lost = 0
        self.resets = 0
        self.duplicates = 0
        self.reordered = 0
        self.missing: set[int] = set()
        self.dts = array("q")
        self.subcarriers: list[int] = []
        self.sources: list[str] = []
        self.gain_frames = 0
        self.agc_moves = 0
        self.fft_moves = 0
        self.agc_seen: list[int] = []
        self.prev_gain: tuple[int, int] | None = None

    def add(self, t_ns: int, frame: Esp32Frame, src_ip: str) -> None:
        if self.packets:
            self.dts.append(t_ns - self.last_t_ns)
        else:
            self.first_t_ns = t_ns
        self.last_t_ns = t_ns
        self.packets += 1
        if frame.n_subcarriers not in self.subcarriers:
            self.subcarriers.append(frame.n_subcarriers)
        if src_ip not in self.sources:
            self.sources.append(src_ip)
        self._gain(frame.agc_gain, frame.fft_gain)
        self._sequence(frame.sequence)

    def _gain(self, agc: int, fft: int) -> None:
        if (agc, fft) != (0, 0):
            self.gain_frames += 1
            if not self.agc_seen:
                self.agc_seen = [agc, agc]
            self.agc_seen = [min(self.agc_seen[0], agc), max(self.agc_seen[1], agc)]
        if self.prev_gain is not None:
            self.agc_moves += agc != self.prev_gain[0]
            self.fft_moves += fft != self.prev_gain[1]
        self.prev_gain = (agc, fft)

    def _sequence(self, seq: int) -> None:
        prev = self.prev_seq
        if prev is None:
            self.prev_seq = seq
            return
        step = (seq - prev) & _U32
        if step == 0:
            self.duplicates += 1
            return
        if step < 0x80000000 and step <= _MAX_FORWARD_GAP:
            gap = step - 1  # modular, so a u32 wrap is not loss
            if gap:
                self.lost += gap
                if gap <= _REORDER_WINDOW:
                    self.missing.update((prev + i) & _U32 for i in range(1, step))
            self.prev_seq = seq
        elif seq in self.missing and ((prev - seq) & _U32) <= _REORDER_WINDOW:
            # Arrived late into a gap already counted as lost: un-count it.
            self.missing.discard(seq)
            self.lost -= 1
            self.reordered += 1
            return
        else:
            self.resets += 1
            self.missing.clear()
            self.prev_seq = seq
        if len(self.missing) > 4 * _REORDER_WINDOW:
            cur = self.prev_seq
            self.missing = {
                s for s in self.missing if ((cur - s) & _U32) <= _REORDER_WINDOW
            }

    def finish(self) -> LinkStats:
        duration_s = (self.last_t_ns - self.first_t_ns) / 1e9
        dts = sorted(self.dts)
        received = self.packets - self.duplicates
        expected = received + self.lost
        return LinkStats(
            node_id=self.node_id,
            packets=self.packets,
            duration_s=duration_s,
            rate_hz=(self.packets - 1) / duration_s if duration_s > 0 else 0.0,
            lost=self.lost,
            loss_pct=100.0 * self.lost / expected if expected else 0.0,
            resets=self.resets,
            duplicates=self.duplicates,
            reordered=self.reordered,
            dt_median_ms=_percentile(dts, 50.0) / 1e6,
            dt_p95_ms=_percentile(dts, 95.0) / 1e6,
            dt_max_ms=dts[-1] / 1e6 if dts else math.nan,
            subcarriers=tuple(self.subcarriers),
            sources=tuple(self.sources),
            gain_frames=self.gain_frames,
            agc_moves=self.agc_moves if self.gain_frames else 0,
            fft_moves=self.fft_moves if self.gain_frames else 0,
            agc_min=self.agc_seen[0] if self.agc_seen else None,
            agc_max=self.agc_seen[1] if self.agc_seen else None,
        )


def link_stats(
    path: str | os.PathLike[str],
    port: int | None = None,
    stats: ReadStats | None = None,
) -> dict[int, LinkStats]:
    """Per-node packet rate, loss and inter-arrival jitter for a recording.

    Malformed datagrams cannot be attributed to a node, so they are counted
    file-wide in `stats.malformed`, alongside everything else `read_pcap`
    skipped and whether the file ended in a truncated record.
    """
    st = stats if stats is not None else ReadStats()
    nodes: dict[int, _LinkAccumulator] = {}
    for pkt in read_pcap(path, port, st):
        try:
            frame = parse_esp32_frame(pkt.payload)
        except ValueError:
            st.malformed += 1
            continue
        acc = nodes.get(frame.node_id)
        if acc is None:
            acc = nodes[frame.node_id] = _LinkAccumulator(frame.node_id)
        acc.add(pkt.t_ns, frame, pkt.src[0])
    return {node_id: nodes[node_id].finish() for node_id in sorted(nodes)}
