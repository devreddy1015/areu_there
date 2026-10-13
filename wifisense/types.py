"""Core domain types for the WiFi sensing pipeline.

A CSI frame is one snapshot of the radio channel: for each OFDM subcarrier,
a complex number describing how that frequency was attenuated and
phase-shifted on its way from transmitter to receiver. People moving,
breathing, or simply existing in the path perturb those numbers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum


class VitalStatus(str, Enum):
    """How much to trust a vital-sign estimate."""

    VALID = "valid"
    DEGRADED = "degraded"
    UNRELIABLE = "unreliable"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class VitalEstimate:
    """A single vital-sign estimate in beats/breaths per minute."""

    value_bpm: float
    confidence: float
    status: VitalStatus

    @staticmethod
    def unavailable() -> "VitalEstimate":
        return VitalEstimate(0.0, 0.0, VitalStatus.UNAVAILABLE)

    @staticmethod
    def from_confidence(bpm: float, confidence: float) -> "VitalEstimate":
        """Grade an estimate by its spectral confidence.

        The thresholds are deliberately conservative: a band-limited
        periodogram peak can look convincing on pure noise, so anything
        under 0.40 is reported but labelled unreliable rather than hidden.
        """
        if confidence >= 0.70:
            status = VitalStatus.VALID
        elif confidence >= 0.40:
            status = VitalStatus.DEGRADED
        else:
            status = VitalStatus.UNRELIABLE
        return VitalEstimate(bpm, confidence, status)


@dataclass
class CsiFrame:
    """One CSI snapshot.

    `amplitudes` and `phases` are per-subcarrier, same length. Phase is in
    radians and is *wrapped* (-pi, pi]; unwrapping and CFO/SFO removal are
    the caller's problem, and matter a great deal for anything beyond
    amplitude-only sensing.
    """

    amplitudes: list[float]
    phases: list[float]
    sample_index: int
    sample_rate_hz: float
    rssi_dbm: int = 0
    node_id: int = 0

    def __post_init__(self) -> None:
        if len(self.amplitudes) != len(self.phases):
            raise ValueError(
                f"amplitude/phase length mismatch: "
                f"{len(self.amplitudes)} != {len(self.phases)}"
            )
        if self.sample_rate_hz <= 0.0:
            raise ValueError(f"sample_rate_hz must be positive, got {self.sample_rate_hz}")

    @property
    def n_subcarriers(self) -> int:
        return len(self.amplitudes)

    @classmethod
    def from_iq(
        cls,
        iq: list[tuple[int, int]],
        sample_index: int,
        sample_rate_hz: float,
        rssi_dbm: int = 0,
        node_id: int = 0,
    ) -> "CsiFrame":
        """Build a frame from (imag, real) integer pairs.

        Note the ordering: ESP-IDF hands back the imaginary component
        first. Getting this backwards leaves amplitude correct and phase
        mirrored, which is a genuinely nasty bug to chase later.
        """
        amplitudes = [math.hypot(float(im), float(re)) for im, re in iq]
        phases = [math.atan2(float(im), float(re)) for im, re in iq]
        return cls(amplitudes, phases, sample_index, sample_rate_hz, rssi_dbm, node_id)


@dataclass
class VitalReading:
    """Combined output of one pipeline step."""

    respiratory_rate: VitalEstimate
    heart_rate: VitalEstimate
    subcarrier_count: int
    signal_quality: float
    timestamp_s: float
    occupied: bool = False
    motion_energy: float = 0.0


@dataclass
class RunningVariance:
    """Welford's online variance — numerically stable, O(1) per sample.

    Used to weight subcarriers: the ones whose amplitude actually moves
    are the ones carrying body signal.
    """

    count: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> None:
        self.count += 1
        delta = x - self.mean
        self.mean += delta / self.count
        self.m2 += delta * (x - self.mean)

    @property
    def variance(self) -> float:
        return self.m2 / (self.count - 1) if self.count > 1 else 0.0

    @property
    def std(self) -> float:
        return math.sqrt(self.variance)

    def reset(self) -> None:
        self.count = 0
        self.mean = 0.0
        self.m2 = 0.0
