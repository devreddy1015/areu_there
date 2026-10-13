"""Raw CSI wire format (the debug format of spec/wire.md §1) and a UDP listener.

A fixed 20-byte header and int8 I/Q pairs: simple enough for any ESP32
firmware to emit, including the Saans node in raw mode.

Layout, little-endian throughout:

    [0..3]   magic 0xC5110001
    [4]      node id
    [5]      antenna count
    [6..7]   subcarrier count (u16)
    [8..11]  centre frequency in MHz (u32)
    [12..15] sequence number (u32)
    [16]     RSSI (i8, dBm)
    [17]     noise floor (i8, dBm)
    [18]     PPDU type (0=HT/legacy, 1=HE-SU, 2=HE-MU, 3=HE-TB, 0xFF=unknown)
    [19]     bandwidth/flags
    [20..]   I/Q payload: int8 pairs, IMAGINARY FIRST
"""

from __future__ import annotations

import socket
import struct
from dataclasses import dataclass
from typing import Iterator

from ..types import CsiFrame

__all__ = [
    "CSI_MAGIC",
    "CSI_HEADER_SIZE",
    "Esp32Frame",
    "parse_esp32_frame",
    "build_esp32_frame",
    "listen_udp",
    "SequenceGapFiller",
]

CSI_MAGIC = 0xC5110001
CSI_HEADER_SIZE = 20
_HEADER = struct.Struct("<IBBHIIbbBB")


@dataclass
class Esp32Frame:
    """One decoded CSI datagram."""

    node_id: int
    n_antennas: int
    n_subcarriers: int
    freq_mhz: int
    sequence: int
    rssi_dbm: int
    noise_floor_dbm: int
    ppdu_type: int
    flags: int
    iq: list[tuple[int, int]]

    def to_csi_frame(self, sample_rate_hz: float) -> CsiFrame:
        """Convert to the pipeline's frame type.

        `sample_rate_hz` is not carried on the wire — it is a property of
        how fast the node is being driven, so you must supply the rate you
        configured (or measure it from packet arrival times).
        """
        return CsiFrame.from_iq(
            self.iq,
            sample_index=self.sequence,
            sample_rate_hz=sample_rate_hz,
            rssi_dbm=self.rssi_dbm,
            node_id=self.node_id,
        )


def parse_esp32_frame(data: bytes) -> Esp32Frame:
    """Decode one datagram. Raises `ValueError` on anything malformed."""
    if len(data) < CSI_HEADER_SIZE:
        raise ValueError(
            f"short frame: {len(data)} bytes, need at least {CSI_HEADER_SIZE}"
        )

    (
        magic,
        node_id,
        n_antennas,
        n_subcarriers,
        freq_mhz,
        sequence,
        rssi,
        noise_floor,
        ppdu_type,
        flags,
    ) = _HEADER.unpack_from(data, 0)

    if magic != CSI_MAGIC:
        raise ValueError(f"bad magic 0x{magic:08X}, expected 0x{CSI_MAGIC:08X}")

    payload = data[CSI_HEADER_SIZE:]
    expected = n_subcarriers * n_antennas * 2
    if len(payload) < expected:
        raise ValueError(
            f"truncated payload: {len(payload)} bytes, "
            f"expected {expected} for {n_subcarriers} subcarriers "
            f"x {n_antennas} antennas"
        )

    # Imaginary component first — see module docstring. Swapping these
    # leaves amplitude correct and mirrors every phase, which survives
    # amplitude-only testing and then quietly breaks phase-based work.
    iq = [
        (
            int.from_bytes(payload[i * 2 : i * 2 + 1], "little", signed=True),
            int.from_bytes(payload[i * 2 + 1 : i * 2 + 2], "little", signed=True),
        )
        for i in range(n_subcarriers)
    ]

    return Esp32Frame(
        node_id=node_id,
        n_antennas=n_antennas,
        n_subcarriers=n_subcarriers,
        freq_mhz=freq_mhz,
        sequence=sequence,
        rssi_dbm=rssi,
        noise_floor_dbm=noise_floor,
        ppdu_type=ppdu_type,
        flags=flags,
        iq=iq,
    )


def build_esp32_frame(
    iq: list[tuple[int, int]],
    node_id: int = 0,
    sequence: int = 0,
    freq_mhz: int = 2437,
    rssi_dbm: int = -45,
    noise_floor_dbm: int = -92,
    n_antennas: int = 1,
) -> bytes:
    """Encode a datagram. Used by the tests to round-trip the parser."""
    header = _HEADER.pack(
        CSI_MAGIC,
        node_id,
        n_antennas,
        len(iq),
        freq_mhz,
        sequence,
        rssi_dbm,
        noise_floor_dbm,
        0,
        0,
    )
    payload = bytearray()
    for im, re in iq:
        payload.append(im & 0xFF)
        payload.append(re & 0xFF)
    return header + bytes(payload)


def listen_udp(
    port: int = 5566,
    host: str = "0.0.0.0",
    bufsize: int = 4096,
    timeout_s: float | None = None,
) -> Iterator[Esp32Frame]:
    """Yield frames from a UDP socket until interrupted.

    Malformed datagrams are skipped rather than raised: a node rebooting
    mid-stream should not take the pipeline down with it.
    """
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    # Deliberately no SO_REUSEADDR: with it, a second listener (or a
    # recorder) on the same port binds silently and the two split the
    # datagrams between them, which looks exactly like 50% packet loss.
    # UDP has no TIME_WAIT, so restarting does not need it either.
    sock.bind((host, port))
    if timeout_s is not None:
        sock.settimeout(timeout_s)
    try:
        while True:
            try:
                data, _addr = sock.recvfrom(bufsize)
            except socket.timeout:
                return
            try:
                yield parse_esp32_frame(data)
            except ValueError:
                continue
    finally:
        sock.close()


_SEQ_MOD = 1 << 32


class SequenceGapFiller:
    """Keep the time axis uniform across lost packets.

    Every stage downstream assumes one frame per 1/rate seconds. Drop 5%
    of packets and feed the survivors straight in, and the pipeline's
    clock runs 5% fast: a 15 BPM breather reads ~15.8 BPM, outside the
    0.5 BPM tolerance, with nothing to say why. So use the raw-CSI
    sequence number: for a short gap, emit linearly interpolated frames
    between the last frame and this one (amplitudes interpolated; phase
    held, since wrapped phase does not interpolate and nothing downstream
    uses it yet).

    Gaps longer than `max_gap` frames, backwards jumps (node reboot,
    reordering) and duplicates are not papered over: the frame passes
    through alone, duplicates are dropped, and each is counted so the
    operator can see the stream was discontinuous.
    """

    def __init__(self, max_gap: int = 20) -> None:
        if max_gap < 0:
            raise ValueError(f"max_gap must be >= 0, got {max_gap}")
        self.max_gap = max_gap
        self._last_seq: int | None = None
        self._last: CsiFrame | None = None
        self.filled = 0
        self.discontinuities = 0
        self.duplicates = 0

    def push(self, sequence: int, frame: CsiFrame) -> list[CsiFrame]:
        """Frames to process, in order: interpolated fill, then `frame`."""
        if self._last_seq is None or self._last is None:
            self._last_seq, self._last = sequence, frame
            return [frame]

        step = (sequence - self._last_seq) % _SEQ_MOD
        if step == 0:
            self.duplicates += 1
            return []
        out: list[CsiFrame] = []
        missing = step - 1
        if step > _SEQ_MOD // 2 or missing > self.max_gap:
            self.discontinuities += 1
        elif missing > 0 and self._last.n_subcarriers == frame.n_subcarriers:
            a0, a1 = self._last.amplitudes, frame.amplitudes
            for j in range(1, missing + 1):
                w = j / step
                out.append(
                    CsiFrame(
                        amplitudes=[x + (y - x) * w for x, y in zip(a0, a1)],
                        phases=list(self._last.phases),
                        sample_index=(self._last_seq + j) % _SEQ_MOD,
                        sample_rate_hz=frame.sample_rate_hz,
                        rssi_dbm=frame.rssi_dbm,
                        node_id=frame.node_id,
                    )
                )
            self.filled += missing
        elif missing > 0:
            self.discontinuities += 1
        out.append(frame)
        self._last_seq, self._last = sequence, frame
        return out
