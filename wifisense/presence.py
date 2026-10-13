"""Presence detection by ambient calibration.

No model, no training, no labelled data: measure how much the channel
jitters in an empty room, then call anything significantly above that
"occupied". This is the honest baseline every WiFi-sensing claim should be
compared against, and in practice it is most of the usable signal.
"""

from __future__ import annotations

import math
from collections import deque

from .dsp import BiquadBandpass
from .types import RunningVariance

__all__ = ["PresenceDetector", "PresenceState"]


class PresenceState:
    """Presence decision plus the numbers behind it."""

    __slots__ = ("occupied", "energy", "threshold", "calibrated", "calibration_progress")

    def __init__(
        self,
        occupied: bool,
        energy: float,
        threshold: float,
        calibrated: bool,
        calibration_progress: float,
    ) -> None:
        self.occupied = occupied
        self.energy = energy
        self.threshold = threshold
        self.calibrated = calibrated
        self.calibration_progress = calibration_progress

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        if not self.calibrated:
            return f"PresenceState(calibrating {self.calibration_progress:.0%})"
        return (
            f"PresenceState(occupied={self.occupied}, "
            f"energy={self.energy:.4f}, threshold={self.threshold:.4f})"
        )


class PresenceDetector:
    """Adaptive-threshold presence detector.

    During calibration the room must be empty: the detector accumulates the
    mean and standard deviation of short-term motion energy and sets its
    threshold at `mean + sigma_mult * sigma`. After that, energy above the
    threshold means occupied.

    Hysteresis (`hold_s`) stops a motionless occupant from flickering the
    output off between breaths — the single most common complaint about
    naive presence detectors.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        calibration_s: float = 30.0,
        sigma_mult: float = 4.0,
        energy_window_s: float = 8.0,
        hold_s: float = 10.0,
        f_low: float = 0.1,
        f_high: float = 2.0,
        warmup_s: float | None = None,
    ) -> None:
        self.sample_rate_hz = sample_rate_hz
        self.sigma_mult = sigma_mult
        self.calibration_samples = max(int(sample_rate_hz * calibration_s), 1)
        self.hold_samples = max(int(sample_rate_hz * hold_s), 1)

        # Discard the filter's startup transient before measuring anything.
        #
        # A biquad centred near 0.1 Hz needs tens of seconds to settle, and
        # the static-suppression EMA baseline is still converging over the same
        # period. Calibrating through that transient measures the filter
        # waking up rather than the room, and sets a threshold so high that
        # an occupied room never crosses it -- the detector then reports
        # "empty" forever, which looks like a tuning problem and is not.
        # Five periods of the lowest frequency clears it.
        self.warmup_samples = (
            max(int(sample_rate_hz * warmup_s), 0)
            if warmup_s is not None
            else int(sample_rate_hz * 5.0 / f_low)
        )
        self._warmed = 0

        # Measure energy *in the band where bodies live*, not broadband.
        #
        # A still person's whole signature is a narrowband 0.1-0.5 Hz
        # breathing modulation. Sensor and multipath noise is broadband, and
        # typically larger. So broadband RMS barely separates an occupied
        # room from an empty one and the occupancy flag chatters -- the
        # single most visible failure of a naive detector. Bandpassing first
        # throws away the noise that carries no information about presence
        # and leaves the part that does.
        #
        # The upper edge reaches past breathing to 2 Hz so that gross motion
        # (walking, gestures) registers too, not just a sleeping subject.
        self._bandpass = BiquadBandpass(f_low, f_high, sample_rate_hz)

        # A floor under the calibrated threshold, derived rather than tuned.
        #
        # The energy is an RMS over an 8 s sliding window, so consecutive
        # values are almost the same number: 30 s of calibration holds only
        # a handful of independent windows, and their sample standard
        # deviation badly underestimates how far an empty room's energy
        # actually wanders. Measured on the Rust evaluation (NOTES.md §8):
        # the "mean + 4 sigma" threshold sat just 13% above the empty-room
        # mean and an empty room read as occupied 36% of the time.
        #
        # Theory gives the spread directly. Band-limited noise with
        # equivalent noise bandwidth B, averaged over T seconds, has an RMS
        # whose relative standard deviation is about 1/(2*sqrt(B*T)). A
        # second-order bandpass has ENBW = (pi/2) * its -3 dB bandwidth. So
        # the threshold is never below mean * (1 + sigma_mult * that).
        enbw = (math.pi / 2.0) * (f_high - f_low)
        self.min_threshold_ratio = 1.0 + sigma_mult / (2.0 * math.sqrt(enbw * energy_window_s))
        self._energy_window: deque[float] = deque(
            maxlen=max(int(sample_rate_hz * energy_window_s), 2)
        )
        self._calib = RunningVariance()
        self._threshold = math.inf
        self._calibrated = False
        self._samples_since_trigger = self.hold_samples

    def process(self, residual: float) -> PresenceState:
        """Feed one fused residual sample and get the current decision."""
        self._energy_window.append(self._bandpass.process(residual))

        # RMS of the band-limited residual is the motion energy proxy.
        n = len(self._energy_window)
        energy = (
            math.sqrt(sum(v * v for v in self._energy_window) / n) if n else 0.0
        )

        if self._warmed < self.warmup_samples:
            self._warmed += 1
            return PresenceState(
                occupied=False,
                energy=energy,
                threshold=self._threshold,
                calibrated=False,
                calibration_progress=0.0,
            )

        if not self._calibrated:
            self._calib.update(energy)
            if self._calib.count >= self.calibration_samples:
                self._threshold = max(
                    self._calib.mean + self.sigma_mult * self._calib.std,
                    self._calib.mean * self.min_threshold_ratio,
                    1e-6,
                )
                self._calibrated = True
            return PresenceState(
                occupied=False,
                energy=energy,
                threshold=self._threshold,
                calibrated=False,
                calibration_progress=self._calib.count / self.calibration_samples,
            )

        if energy > self._threshold:
            self._samples_since_trigger = 0
        else:
            self._samples_since_trigger += 1

        occupied = self._samples_since_trigger < self.hold_samples
        return PresenceState(
            occupied=occupied,
            energy=energy,
            threshold=self._threshold,
            calibrated=True,
            calibration_progress=1.0,
        )

    @property
    def calibrated(self) -> bool:
        return self._calibrated

    @property
    def warmup_s(self) -> float:
        """Seconds of input discarded before calibration begins."""
        return self.warmup_samples / self.sample_rate_hz

    @property
    def threshold(self) -> float:
        return self._threshold

    def recalibrate(self) -> None:
        """Start calibration over. The room must be empty again."""
        self._calib.reset()
        self._threshold = math.inf
        self._calibrated = False
        self._energy_window.clear()
        self._bandpass.reset()
        self._warmed = 0
        self._samples_since_trigger = self.hold_samples
