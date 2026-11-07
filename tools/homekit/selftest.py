"""`homekit selftest`: is each board sending, and how fast?

Three places to look, for the three moments of hardware day:

    selftest --serial /dev/ttyUSB0   the board's own log, right after flashing
    selftest                         the running gateway's /api/links
    selftest --udp                   listen on UDP 5566 yourself, no gateway

Each ends in one line per node: "frames arriving at X Hz from node N".
Healthy is what docs/HARDWARE.md §7 calls healthy: a CSI rate within a
few percent of 20 Hz and at least 10 Hz, FeatureFrames at 2 Hz, nothing
dropped. The serial log is never saved and never echoed whole: the WiFi
driver prints the network's name in it.
"""

from __future__ import annotations

import re
import socket
import struct
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

from homekit import ports
from homekit.gateway import TOKEN_FILE, Host

REPO = Path(__file__).resolve().parents[2]
MIN_CSI_HZ = 10.0  # README: at least 10 Hz sustained before trusting a link
FRAME_HZ = 2.0  # CONFIG_SAANS_FRAME_RATE_HZ in every profile
MIN_FRAME_HZ = 0.75 * FRAME_HZ
FEATURE_MAGIC = 0x534E4153
RAW_MAGIC = 0xC5110001
ANSI = re.compile(r"\x1b\[[0-9;]*m")

# The lines the firmware prints (edge/firmware/main, HARDWARE.md §7-8).
BANNER_ROLE = re.compile(r"role\s*:\s*(TX|RX)")
BANNER_ID = re.compile(r"node id\s*:\s*(\d+)")
RX_STATS = re.compile(r"csi=([\d.]+) Hz frames=(\d+) dropped=(\d+) send_fail=(\d+)")
TX_STATS = re.compile(r"sent=(\d+) failed=(\d+) rate=([\d.]+) Hz")
GOT_IP = re.compile(r"saans\.net: got ip (\S+)")
UPLINK = re.compile(r"uplink -> (\S+)")
HT20_OK = "ESP-NOW rate: HT20 MCS0 LGI"
# Each is a sentence from §8, in its order; the text is the firmware's.
WARNINGS = [
    ("no CSI for 1 s", "no CSI: is the TX on, and on the same channel?"),
    ("HT20 rate config FAILED", "the TX is sending non-HT frames (HARDWARE.md §8.2)"),
    ("FEWER SUBCARRIERS", "fewer subcarriers than the DSP expects (§8.3)"),
    ("buffer length changed", "the CSI layout changes mid-run (§8.4)"),
    ("but the DSP is configured for", "CSI faster than configured: set the TX rate equal (§8.5)"),
    ("HALTED", "the RX is NOT sensing: rebuild with matching config (§8.6)"),
    ("disconnected, retrying", "the RX cannot join the WiFi: SSID, password, channel (§8.7)"),
    ("bad gateway address", "the gateway address compiled in is not an IPv4 address"),
]


@dataclass
class SerialReport:
    role: str | None = None
    node: int | None = None
    ip: str | None = None
    uplink: str | None = None
    ht20: bool = False
    csi_hz: float | None = None
    frames: int | None = None
    dropped: int = 0
    send_fail: int = 0
    tx_hz: float | None = None
    tx_failed: int = 0
    capture: str | None = None
    warnings: list[str] = field(default_factory=list)

    def feed(self, line: str) -> None:
        line = ANSI.sub("", line)
        if m := BANNER_ROLE.search(line):
            self.role = m.group(1)
        if m := BANNER_ID.search(line):
            self.node = int(m.group(1))
        if m := GOT_IP.search(line):
            self.ip = m.group(1)
        if m := UPLINK.search(line):
            self.uplink = m.group(1)
        if HT20_OK in line:
            self.ht20 = True
        if m := RX_STATS.search(line):
            self.role = self.role or "RX"
            self.csi_hz, self.frames = float(m.group(1)), int(m.group(2))
            self.dropped, self.send_fail = int(m.group(3)), int(m.group(4))
        if m := TX_STATS.search(line):
            self.role = self.role or "TX"
            self.tx_failed, self.tx_hz = int(m.group(2)), float(m.group(3))
        if "capture: len=" in line:
            self.capture = line[line.index("capture: len="):]
        for needle, meaning in WARNINGS:
            if needle in line and meaning not in self.warnings:
                self.warnings.append(meaning)

    def done(self) -> bool:
        return self.csi_hz is not None or self.tx_hz is not None

    def verdict(self) -> tuple[bool, list[str]]:
        who = f"node {self.node}" if self.node is not None else "the board"
        out = [f"{who}: {w}" for w in self.warnings]
        if self.role == "TX" or self.tx_hz is not None:
            if self.tx_hz is None:
                return False, out + [f"{who} (TX): no 'sent=' line yet; it prints one a minute"]
            ok = self.tx_hz >= MIN_CSI_HZ and self.tx_failed == 0 and not self.warnings
            out.append(f"{who} (TX): beacons at {self.tx_hz:.1f} Hz, {self.tx_failed} failed"
                       + ("" if self.ht20 else "; HT20 line not seen (it prints at boot only)"))
            return ok, out
        if self.csi_hz is None:
            return False, out + [f"{who} (RX): no 'csi=' line yet; it prints one a minute"]
        ok = (self.csi_hz >= MIN_CSI_HZ and self.dropped == 0 and self.send_fail == 0
              and not self.warnings)
        out.append(f"{who} (RX): CSI arriving at {self.csi_hz:.1f} Hz, {self.frames} frames sent, "
                   f"{self.dropped} dropped, {self.send_fail} send failures")
        if self.ip or self.uplink:
            out.append(f"{who} (RX): WiFi address {self.ip or '?'}, sending to {self.uplink or '?'}")
        elif self.csi_hz is not None:
            out.append(f"{who} (RX): WiFi join not seen in this window (it prints at boot only)")
        if self.capture:
            out.append(f"{who} (RX): {self.capture}")
        return ok, out


def from_serial(dev: str, seconds: float) -> int:
    problem = ports.access_problem(dev)
    if problem:
        print(f"error: {problem}", file=sys.stderr)
        return 2
    rep = SerialReport()
    print(f"Reading {dev} for up to {seconds:.0f} s (the boards report once a minute) ...")
    ports.read_lines(dev, seconds, lambda line: (rep.feed(line), not rep.done())[1])
    return _print(*rep.verdict())


@dataclass(frozen=True)
class LinkRate:
    node: int
    room: str
    frame_hz: float
    csi_hz: float | None = None
    loss: float | None = None
    raw: bool = False

    def line(self) -> str:
        extra = []
        if self.csi_hz is not None:
            extra.append(f"CSI at {self.csi_hz:.1f} Hz on the node")
        if self.loss is not None:
            extra.append(f"loss {100 * self.loss:.1f}%")
        if self.raw:
            extra.append("raw mode ON")
        tail = f" ({'; '.join(extra)})" if extra else ""
        return f"frames arriving at {self.frame_hz:.1f} Hz from node {self.node}, {self.room}{tail}"

    def ok(self) -> bool:
        return self.frame_hz >= MIN_FRAME_HZ and (self.csi_hz is None or self.csi_hz >= MIN_CSI_HZ)


def rates_from_links(before: list, after: list, dt: float) -> list[LinkRate]:
    """Two /api/links snapshots `dt` apart into one rate per receiver."""
    def frames(snap):
        return {(sp["space_id"], l["link"]["rx"]): l for sp in snap for l in sp.get("links", [])}

    a, b = frames(before), frames(after)
    out = []
    for (room, rx), link in sorted(b.items(), key=lambda kv: kv[0][1]):
        f0 = a.get((room, rx), {}).get("frames", link.get("frames", 0))
        hz = max(0, link.get("frames", 0) - f0) / dt if dt > 0 else 0.0
        out.append(LinkRate(rx, room, hz, link.get("rate_hz") if link.get("up") else None,
                            link.get("loss"), bool(link.get("raw_mode"))))
    return out


def from_gateway(url: str, seconds: float, token_file: str) -> int:
    host = Host()
    try:
        token = Path(token_file).read_text().strip()
    except PermissionError:
        print(f"note: cannot read {token_file}; run with sudo if the gateway asks for sign-in")
        token = None
    except FileNotFoundError:
        token = None
    first = host.http_json(f"{url}/api/links", token)
    if first is None:
        if host.last_status in (401, 403):
            why = (f"the gateway asks for sign-in and {token_file} was not readable: "
                   "run this with sudo")
        else:
            why = "is the gateway running? sudo systemctl status saans-gateway"
        print(f"error: no answer from {url}/api/links ({why})", file=sys.stderr)
        return 2
    print(f"Counting frames at {url} for {seconds:.0f} s ...")
    t0 = time.monotonic()
    time.sleep(seconds)
    second = host.http_json(f"{url}/api/links", token) or []
    rates = rates_from_links(first, second, time.monotonic() - t0)
    lines = [r.line() + ("" if r.ok() else "  <-- too slow or silent") for r in rates]
    health = host.http_json(f"{url}/api/health", token) or {}
    for u in health.get("unknown_links", []):
        lines.append(f"node {u.get('rx')} is sending ({u.get('frames')} {u.get('kind')} frames"
                     f" from {u.get('last_src')}) but no room lists it: add it with --reconfigure")
    if not rates:
        lines.append("no rooms with receivers in the gateway's config")
    return _print(bool(rates) and all(r.ok() for r in rates) and not health.get("unknown_links"), lines)


def classify(data: bytes) -> tuple[str, int] | None:
    """(kind, node) for a datagram on the sensor port, or None if neither
    format. The headers only: spec/wire.md §1 and §2."""
    if len(data) < 6:
        return None
    (magic,) = struct.unpack_from("<I", data, 0)
    if magic == FEATURE_MAGIC and len(data) in (68, 84):
        return "feature", data[5]
    if magic == RAW_MAGIC and len(data) >= 20:
        return "raw", data[4]
    return None


def from_udp(port: int, seconds: float) -> int:
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.bind(("0.0.0.0", port))
    except OSError:
        print(f"error: UDP {port} is in use, most likely by the gateway: run selftest without --udp",
              file=sys.stderr)
        return 2
    counts: dict[tuple[str, int], int] = {}
    print(f"Listening on UDP {port} for {seconds:.0f} s ...")
    deadline = time.monotonic() + seconds
    with s:
        while (left := deadline - time.monotonic()) > 0:
            s.settimeout(left)
            try:
                data, _ = s.recvfrom(4096)
            except TimeoutError:
                break
            k = classify(data)
            if k:
                counts[k] = counts.get(k, 0) + 1
    nodes = sorted({n for _, n in counts})
    lines, ok = [], bool(nodes)
    for n in nodes:
        f = counts.get(("feature", n), 0) / seconds
        r = counts.get(("raw", n), 0) / seconds
        lines.append(f"frames arriving at {f:.1f} Hz from node {n}"
                     + (f", raw CSI at {r:.1f} Hz" if r else ""))
        ok &= f >= MIN_FRAME_HZ and (not r or r >= MIN_CSI_HZ)
    if not nodes:
        lines.append("nothing arrived: check the boards' 'uplink ->' address is this machine's")
    return _print(ok, lines)


def _print(ok: bool, lines: list[str]) -> int:
    for line in lines:
        print(f"  {line}")
    print("PASS" if ok else "NOT YET: see docs/HARDWARE.md §8, in order")
    return 0 if ok else 1


def run(args) -> int:
    if args.serial:
        return from_serial(args.serial, args.seconds or 75)
    if args.udp:
        return from_udp(args.port, args.seconds or 20)
    return from_gateway(args.url.rstrip("/"), args.seconds or 20, args.token_file or TOKEN_FILE)
