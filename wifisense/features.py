"""FeatureFrame v1: what a Saans sensor node sends instead of raw CSI.

Reference implementation of `spec/wire.md` §2, and of §2b, FeatureFrame
v2: the same fields, signed (ADR-012). The C++ edge firmware and
the Rust gateway must produce the same frames from the same input; the
golden vectors in `spec/golden/` are generated from this file.

Why features and not raw CSI (ADR-001): raw CSI is enough to re-derive
breathing, heartbeat and, with more nodes, movement through walls. Sending
it to a server makes the server a store of health data whether anyone
meant it to be or not. A 68-byte frame twice a second carries exactly
what the health detectors need, with breathing- and heart-band fields
zeroed on the node unless the space has opted in.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import struct
import zlib
from collections.abc import Callable
from collections import deque
from dataclasses import dataclass, field

from .dsp import BiquadBandpass
from .pipeline import SensingPipeline
from .preprocess import subcarrier_bnr
from .sources.esp32_udp import SequenceGapFiller
from .types import CsiFrame, VitalStatus

__all__ = [
    "FEATURE_MAGIC",
    "FEATURE_VERSION",
    "FEATURE_FRAME_SIZE",
    "FeatureFrame",
    "FeatureExtractor",
    "decode_feature_frame",
    "FEATURE_VERSION_SIGNED",
    "SIGNED_FRAME_SIZE",
    "TAG_SIZE",
    "NODE_KEY_LABEL",
    "FrameAuthError",
    "SignedFrame",
    "ReplayGuard",
    "derive_node_key",
    "encode_signed",
    "decode_signed",
]

FEATURE_MAGIC = 0x534E4153  # bytes on the wire: b"SANS"
FEATURE_VERSION = 1
FEATURE_FRAME_SIZE = 68
_BODY = struct.Struct("<IBBBBIIHH9fIbbBB")  # 64 bytes, then u32 CRC
_CRC = struct.Struct("<I")
assert _BODY.size == 64

# FeatureFrame v2 (ADR-012): v1's first 63 bytes with version 2, then
# key_id in v1's reserved byte, the node's boot counter, and a tag.
FEATURE_VERSION_SIGNED = 2
SIGNED_FRAME_SIZE = 84
SIGNED_BODY_SIZE = 68  # what the tag covers
TAG_SIZE = 16
NODE_KEY_LABEL = b"saans frame key v1"
_BOOT = struct.Struct("<I")

FLAG_CALIBRATED = 1 << 0
FLAG_OCCUPIED = 1 << 1
FLAG_VITALS_ENABLED = 1 << 2
FLAG_RAW_MODE = 1 << 3
FLAG_BREATHING_VALID = 1 << 4
FLAG_HEART_VALID = 1 << 5
FLAG_DISCONTINUITY = 1 << 6

NOT_CALIBRATED = 0xFFFFFFFF
NOISE_UNKNOWN = -128

# Window lengths fixed by the spec, in seconds.
BREATH_RMS_WINDOW_S = 2.0
GROSS_MOTION_WINDOW_S = 1.0
LINK_STATS_WINDOW_S = 10.0
# Gross motion: above the breathing band and its first harmonics, below
# the Nyquist of a 10 Hz link. Limbs, posture changes, walking.
GROSS_MOTION_BAND_HZ = (1.0, 4.0)
# Two cascaded sections, not one. A single biquad's skirt is only
# 6 dB/octave: at 0.25 Hz it passes ~19% of the breathing component, and a
# still breather's breathing band is ~6x the empty-room level, so one
# section made sitting still read as 1.3x the empty-room motion level and
# reset every stillness clock (measured in the Rust evaluation, NOTES.md
# §8). Two sections put breathing ~28 dB down.
GROSS_MOTION_SECTIONS = 2
BREATH_BAND_HZ = (0.1, 0.5)


def _f32(x: float) -> float:
    """Round to float32, as the wire does."""
    return struct.unpack("<f", struct.pack("<f", x))[0]


@dataclass
class FeatureFrame:
    node_id: int = 0
    tx_id: int = 0xFF
    flags: int = 0
    seq: int = 0
    uptime_ms: int = 0
    csi_rate_chz: int = 0
    loss_permille: int = 0
    motion_energy: float = 0.0
    presence_threshold: float = 0.0
    gross_motion: float = 0.0
    breath_rms: float = 0.0
    breathing_bpm: float = 0.0
    breathing_conf: float = 0.0
    heart_bpm: float = 0.0
    heart_conf: float = 0.0
    best_bnr: float = 0.0
    calib_age_s: int = NOT_CALIBRATED
    rssi_dbm: int = 0
    noise_floor_dbm: int = NOISE_UNKNOWN
    n_subcarriers: int = 0
    version: int = FEATURE_VERSION

    def flag(self, bit: int) -> bool:
        return bool(self.flags & bit)

    def encode(self) -> bytes:
        body = self._body(self.version, 0)
        return body + _CRC.pack(zlib.crc32(body) & 0xFFFFFFFF)

    def _body(self, version: int, last_byte: int) -> bytes:
        """Bytes 0..63: every field, clamped as the spec says. The last byte
        is v1's `reserved` (0) or v2's `key_id`."""
        return _BODY.pack(
            FEATURE_MAGIC,
            version,
            self.node_id & 0xFF,
            self.tx_id & 0xFF,
            self.flags & 0xFF,
            self.seq & 0xFFFFFFFF,
            self.uptime_ms & 0xFFFFFFFF,
            max(0, min(0xFFFF, self.csi_rate_chz)),
            max(0, min(0xFFFF, self.loss_permille)),
            self.motion_energy,
            self.presence_threshold,
            self.gross_motion,
            self.breath_rms,
            self.breathing_bpm,
            self.breathing_conf,
            self.heart_bpm,
            self.heart_conf,
            self.best_bnr,
            self.calib_age_s & 0xFFFFFFFF,
            max(-128, min(127, self.rssi_dbm)),
            max(-128, min(127, self.noise_floor_dbm)),
            max(0, min(255, self.n_subcarriers)),
            last_byte,
        )

    def as_dict(self) -> dict:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


def decode_feature_frame(data: bytes) -> FeatureFrame:
    """Parse and validate one datagram. Raises ValueError on anything off."""
    if len(data) != FEATURE_FRAME_SIZE:
        raise ValueError(f"FeatureFrame must be {FEATURE_FRAME_SIZE} bytes, got {len(data)}")
    (crc,) = _CRC.unpack_from(data, 64)
    if zlib.crc32(data[:64]) & 0xFFFFFFFF != crc:
        raise ValueError("FeatureFrame CRC mismatch")
    v = _BODY.unpack_from(data, 0)
    if v[0] != FEATURE_MAGIC:
        raise ValueError(f"bad magic 0x{v[0]:08X}")
    if v[1] != FEATURE_VERSION:
        raise ValueError(f"unsupported FeatureFrame version {v[1]}")
    return _fields(v)


def _fields(v: tuple) -> FeatureFrame:
    return FeatureFrame(
        version=v[1], node_id=v[2], tx_id=v[3], flags=v[4], seq=v[5], uptime_ms=v[6],
        csi_rate_chz=v[7], loss_permille=v[8], motion_energy=v[9], presence_threshold=v[10],
        gross_motion=v[11], breath_rms=v[12], breathing_bpm=v[13], breathing_conf=v[14],
        heart_bpm=v[15], heart_conf=v[16], best_bnr=v[17], calib_age_s=v[18],
        rssi_dbm=v[19], noise_floor_dbm=v[20], n_subcarriers=v[21],
    )


# ---------------------------------------------------------------- v2, signed


class FrameAuthError(ValueError):
    """A v2 frame refused. `reason` is one of: size, magic, version, key
    (no site key with this key_id), tag (forged, corrupted, or signed with
    another key). The gateway counts each separately."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


@dataclass(frozen=True)
class SignedFrame:
    frame: FeatureFrame
    key_id: int
    boot: int


def derive_node_key(site_key: bytes, node_id: int) -> bytes:
    """The key node `node_id` is flashed with: HMAC-SHA256(site key,
    label || node_id). The gateway holds only site keys and derives these,
    so one node's key signs for that node and nothing else."""
    if len(site_key) != 32:
        raise ValueError(f"a site key is 32 bytes, got {len(site_key)}")
    if not 0 <= node_id <= 0xFF:
        raise ValueError(f"node_id must fit a byte, got {node_id}")
    return hmac.new(site_key, NODE_KEY_LABEL + bytes([node_id]), hashlib.sha256).digest()


def _tag(node_key: bytes, body: bytes) -> bytes:
    return hmac.new(node_key, body, hashlib.sha256).digest()[:TAG_SIZE]


def encode_signed(frame: FeatureFrame, *, node_key: bytes, key_id: int, boot: int) -> bytes:
    """FeatureFrame v2: the frame's fields, `key_id`, `boot`, and the first
    16 bytes of HMAC-SHA256 over everything before the tag. `frame.version`
    is ignored; the result is always version 2."""
    if len(node_key) != 32:
        raise ValueError(f"a node key is 32 bytes, got {len(node_key)}")
    if not 0 <= key_id <= 0xFF:
        raise ValueError(f"key_id must fit a byte, got {key_id}")
    if not 0 <= boot <= 0xFFFFFFFF:
        raise ValueError(f"boot must fit a u32, got {boot}")
    body = frame._body(FEATURE_VERSION_SIGNED, key_id) + _BOOT.pack(boot)
    return body + _tag(node_key, body)


def decode_signed(
    data: bytes, node_key_for: Callable[[int, int], bytes | None]
) -> SignedFrame:
    """Check, then parse, one v2 datagram.

    `node_key_for(key_id, node_id)` returns that node's key under that
    site key, or None if no such site key is held. Fields are read only
    after the tag matches, and the replay floor is the caller's
    (`ReplayGuard`), applied after this.
    """
    if len(data) != SIGNED_FRAME_SIZE:
        raise FrameAuthError("size", f"a signed FeatureFrame is {SIGNED_FRAME_SIZE} bytes, got {len(data)}")
    (magic,) = struct.unpack_from("<I", data, 0)
    if magic != FEATURE_MAGIC:
        raise FrameAuthError("magic", f"bad magic 0x{magic:08X}")
    if data[4] != FEATURE_VERSION_SIGNED:
        raise FrameAuthError("version", f"expected FeatureFrame version {FEATURE_VERSION_SIGNED}, got {data[4]}")
    key_id, node_id = data[63], data[5]
    node_key = node_key_for(key_id, node_id)
    if node_key is None:
        raise FrameAuthError("key", f"no site key with key_id {key_id}")
    body, tag = data[:SIGNED_BODY_SIZE], data[SIGNED_BODY_SIZE:]
    if not hmac.compare_digest(_tag(node_key, body), tag):
        raise FrameAuthError("tag", "the tag does not match")
    (boot,) = _BOOT.unpack_from(data, 64)
    return SignedFrame(frame=_fields(_BODY.unpack_from(data, 0)), key_id=key_id, boot=boot)


class ReplayGuard:
    """The highest (boot, seq) accepted per link (node_id, tx_id).

    A frame is fresh only if its pair is strictly greater, boot first: a
    replay is never greater than a frame already accepted, and a reboot
    raises `boot`, so `seq` restarting from 0 is fresh. A node whose boot
    counter goes backwards is refused until its floor is cleared, which is
    deliberate: from here it is indistinguishable from a replay.
    """

    def __init__(self, floors: dict[tuple[int, int], tuple[int, int]] | None = None) -> None:
        self._floors: dict[tuple[int, int], tuple[int, int]] = dict(floors or {})

    def accept(self, node_id: int, tx_id: int, *, boot: int, seq: int) -> bool:
        link, pair = (node_id, tx_id), (boot, seq)
        floor = self._floors.get(link)
        if floor is not None and pair <= floor:
            return False
        self._floors[link] = pair
        return True

    def floors(self) -> dict[tuple[int, int], tuple[int, int]]:
        return dict(self._floors)

    def to_json(self) -> str:
        return json.dumps(
            [{"node_id": n, "tx_id": t, "boot": b, "seq": q} for (n, t), (b, q) in sorted(self._floors.items())]
        )

    @classmethod
    def from_json(cls, text: str) -> ReplayGuard:
        return cls({(e["node_id"], e["tx_id"]): (e["boot"], e["seq"]) for e in json.loads(text)})


class _RollingRms:
    """RMS over the last `n` samples."""

    def __init__(self, n: int) -> None:
        self._sq: deque[float] = deque(maxlen=max(1, n))

    def push(self, x: float) -> None:
        self._sq.append(x * x)

    @property
    def value(self) -> float:
        # Summed afresh on each read (twice a second) rather than kept as a
        # running total: running-sum float drift over a night of samples
        # would make the three implementations disagree for no physical
        # reason.
        if not self._sq:
            return 0.0
        return math.sqrt(sum(self._sq) / len(self._sq))


class FeatureExtractor:
    """One link's CSI frames in, FeatureFrames out.

    Wraps the reference `SensingPipeline` and adds what the wire needs:
    gap filling, gross-motion and breathing-band RMS, link statistics,
    BNR link quality, consent gating and encoding.

    Call `push(frame, sequence, t_s)` for every *received* CSI frame. It
    returns zero or more FeatureFrames (normally zero; one every
    `round(fs / frame_rate_hz)` processed frames). Deterministic given its
    inputs: features are clocked by processed frames, not wall time.
    """

    def __init__(
        self,
        n_subcarriers: int = 56,
        sample_rate_hz: float = 20.0,
        node_id: int = 0,
        tx_id: int = 0xFF,
        frame_rate_hz: float = 2.0,
        vitals_enabled: bool = False,
        heart_enabled: bool = False,
        calibration_s: float = 30.0,
        max_gap_s: float = 1.0,
        subcarrier_selection: str = "variance",
        raw_mode: bool = False,
        bnr_window_s: float = 30.0,
        bnr_update_s: float = 5.0,
        bnr_rate_hz: float = 4.0,
        compute_bnr: bool = True,
    ) -> None:
        if sample_rate_hz < 10.0:
            raise ValueError(
                f"FeatureFrame needs >= 10 Hz CSI (got {sample_rate_hz:g}): the gross-motion "
                f"band reaches 4 Hz, and breathing needs it anyway (campus plan, constraint 4)"
            )
        if not 0.0 < frame_rate_hz <= sample_rate_hz:
            raise ValueError(f"frame_rate_hz must be in (0, {sample_rate_hz}], got {frame_rate_hz}")
        self.n_subcarriers = n_subcarriers
        self.fs = sample_rate_hz
        self.node_id = node_id
        self.tx_id = tx_id
        self.vitals_enabled = vitals_enabled
        self.heart_enabled = heart_enabled
        self.raw_mode = raw_mode

        self.pipeline = SensingPipeline(
            n_subcarriers=n_subcarriers,
            sample_rate_hz=sample_rate_hz,
            calibration_s=calibration_s,
            subcarrier_selection=subcarrier_selection,
        )
        self.gaps = SequenceGapFiller(max_gap=int(max_gap_s * sample_rate_hz))
        self._breath_filter = BiquadBandpass(*BREATH_BAND_HZ, sample_rate_hz)
        self._motion_filters = [BiquadBandpass(*GROSS_MOTION_BAND_HZ, sample_rate_hz) for _ in range(GROSS_MOTION_SECTIONS)]
        self._breath_rms = _RollingRms(round(sample_rate_hz * BREATH_RMS_WINDOW_S))
        self._motion_rms = _RollingRms(round(sample_rate_hz * GROSS_MOTION_WINDOW_S))
        self._every = max(1, round(sample_rate_hz / frame_rate_hz))

        # best_bnr is optional (spec/wire.md): its row store is half of a
        # node's per-link memory, and no detector reads it. A memory-bound
        # node sets compute_bnr=False and reports 0.0 ("not computed");
        # every other field is unaffected.
        self.compute_bnr = compute_bnr
        # BNR link quality: block-average residuals to ~bnr_rate_hz, score
        # the last bnr_window_s every bnr_update_s (same scheme as
        # RespirFiFusion, so best_bnr means the same thing as its selector).
        self._bnr_dec = max(1, int(round(sample_rate_hz / bnr_rate_hz)))
        self._bnr_rate = sample_rate_hz / self._bnr_dec
        self._bnr_rows: deque[list[float]] = deque(maxlen=max(8, int(bnr_window_s * self._bnr_rate)))
        self._bnr_acc = [0.0] * n_subcarriers
        self._bnr_n = 0
        self._bnr_every = max(1, int(round(bnr_update_s * sample_rate_hz)))
        self._best_bnr = 0.0

        self._processed = 0
        self._seq = 0
        self._t0: float | None = None
        self._calibrated_at: int | None = None
        self._arrivals: deque[tuple[float, int]] = deque()
        self._rssi_sum = 0
        self._rssi_n = 0
        self._noise_sum = 0
        self._noise_n = 0
        self._discontinuities_seen = 0
        self._last_reading = None

    # -- input -------------------------------------------------------------

    def push(
        self,
        frame: CsiFrame,
        sequence: int | None = None,
        t_s: float | None = None,
        noise_floor_dbm: int | None = None,
    ) -> list[FeatureFrame]:
        seq = frame.sample_index if sequence is None else sequence
        t = t_s if t_s is not None else self._processed / self.fs
        if self._t0 is None:
            self._t0 = t
        self._arrivals.append((t, seq))
        while self._arrivals and self._arrivals[0][0] < t - LINK_STATS_WINDOW_S:
            self._arrivals.popleft()
        self._rssi_sum += frame.rssi_dbm
        self._rssi_n += 1
        if noise_floor_dbm is not None:
            self._noise_sum += noise_floor_dbm
            self._noise_n += 1

        out: list[FeatureFrame] = []
        for f in self.gaps.push(seq, frame):
            emitted = self._process(f, t)
            if emitted is not None:
                out.append(emitted)
        return out

    def _process(self, frame: CsiFrame, t: float) -> FeatureFrame | None:
        p = self.pipeline
        reading = p.process(frame)
        self._last_reading = reading
        self._processed += 1
        if p.presence.calibrated and self._calibrated_at is None:
            self._calibrated_at = self._processed

        self._breath_rms.push(self._breath_filter.process(p.last_vitals_input))
        m = p.last_fused
        for section in self._motion_filters:
            m = section.process(m)
        self._motion_rms.push(m)

        if self.compute_bnr:
            self._bnr_step(p)

        if self._processed % self._every:
            return None
        return self._emit(t)

    def _bnr_step(self, p: SensingPipeline) -> None:
        n = min(len(p.last_residuals), self.n_subcarriers)
        for k in range(n):
            r = p.last_residuals[k]
            self._bnr_acc[k] += r if math.isfinite(r) else 0.0
        self._bnr_n += 1
        if self._bnr_n >= self._bnr_dec:
            self._bnr_rows.append([a / self._bnr_n for a in self._bnr_acc])
            self._bnr_acc = [0.0] * self.n_subcarriers
            self._bnr_n = 0
        if self._processed % self._bnr_every == 0 and len(self._bnr_rows) == self._bnr_rows.maxlen:
            bnr, _ = subcarrier_bnr(list(self._bnr_rows), self._bnr_rate, *BREATH_BAND_HZ)
            finite = [b for b in bnr if math.isfinite(b)]
            self._best_bnr = max(finite) if finite else 0.0

    # -- output ------------------------------------------------------------

    def _link_stats(self) -> tuple[int, int]:
        if len(self._arrivals) < 2:
            return 0, 0
        (t_first, s_first), (t_last, s_last) = self._arrivals[0], self._arrivals[-1]
        span = t_last - t_first
        rate = (len(self._arrivals) - 1) / span if span > 0.0 else 0.0
        expected = ((s_last - s_first) % (1 << 32)) + 1
        received = len(self._arrivals)
        loss = max(0.0, 1.0 - received / expected) if expected >= received else 0.0
        return int(round(rate * 100.0)), int(round(loss * 1000.0))

    def _emit(self, t: float) -> FeatureFrame:
        p = self.pipeline
        reading = self._last_reading
        presence = p.presence
        flags = 0
        if presence.calibrated:
            flags |= FLAG_CALIBRATED
        if reading.occupied:
            flags |= FLAG_OCCUPIED
        if self.vitals_enabled:
            flags |= FLAG_VITALS_ENABLED
        if self.raw_mode:
            flags |= FLAG_RAW_MODE
        if self.gaps.discontinuities > self._discontinuities_seen:
            flags |= FLAG_DISCONTINUITY
            self._discontinuities_seen = self.gaps.discontinuities

        usable = (VitalStatus.VALID, VitalStatus.DEGRADED)
        breathing_ok = self.vitals_enabled and reading.occupied and reading.respiratory_rate.status in usable
        heart_ok = breathing_ok and self.heart_enabled and reading.heart_rate.status in usable
        if breathing_ok:
            flags |= FLAG_BREATHING_VALID
        if heart_ok:
            flags |= FLAG_HEART_VALID

        rate_chz, loss_pm = self._link_stats()
        vit = self.vitals_enabled
        heart = vit and self.heart_enabled
        frame = FeatureFrame(
            node_id=self.node_id,
            tx_id=self.tx_id,
            flags=flags,
            seq=self._seq,
            uptime_ms=int(round((t - (self._t0 or 0.0)) * 1000.0)),
            csi_rate_chz=rate_chz,
            loss_permille=loss_pm,
            motion_energy=_f32(reading.motion_energy),
            presence_threshold=_f32(presence.threshold if presence.calibrated else 0.0),
            gross_motion=_f32(self._motion_rms.value),
            breath_rms=_f32(self._breath_rms.value) if vit else 0.0,
            breathing_bpm=_f32(reading.respiratory_rate.value_bpm) if vit else 0.0,
            breathing_conf=_f32(reading.respiratory_rate.confidence) if vit else 0.0,
            heart_bpm=_f32(reading.heart_rate.value_bpm) if heart else 0.0,
            heart_conf=_f32(reading.heart_rate.confidence) if heart else 0.0,
            best_bnr=_f32(self._best_bnr),
            calib_age_s=(
                int((self._processed - self._calibrated_at) / self.fs)
                if self._calibrated_at is not None
                else NOT_CALIBRATED
            ),
            rssi_dbm=int(round(self._rssi_sum / self._rssi_n)) if self._rssi_n else 0,
            noise_floor_dbm=int(round(self._noise_sum / self._noise_n)) if self._noise_n else NOISE_UNKNOWN,
            n_subcarriers=self.n_subcarriers,
        )
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        self._rssi_sum = self._rssi_n = 0
        self._noise_sum = self._noise_n = 0
        return frame
