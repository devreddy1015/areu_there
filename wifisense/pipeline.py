"""End-to-end pipeline: CSI frames in, presence and vitals out."""

from __future__ import annotations

from .preprocess import RespirFiFusion, StaticSuppressor, fuse
from .presence import PresenceDetector
from .types import CsiFrame, VitalEstimate, VitalReading, VitalStatus
from .vitals import BreathingExtractor, HeartRateExtractor

__all__ = ["SensingPipeline"]


class SensingPipeline:
    """Wires the stages together.

        CSI frame
          -> static suppression (per-subcarrier baseline removal + weights)
          -> weighted fusion to a scalar
          -> presence detector  (ambient-calibrated threshold)
          -> breathing extractor (0.1-0.5 Hz bandpass + spectral peak)
          -> heart rate extractor (0.8-2.0 Hz bandpass + spectral peak)

    Vitals are suppressed while the room reads as empty: reporting a
    breathing rate for an unoccupied room is the classic way these systems
    manufacture impressive-looking numbers out of noise.

    `subcarrier_selection` picks how subcarriers are combined for the
    vitals path:

    - ``"variance"`` (default): `fuse` with the suppressor's variance
      weights. All weights are positive, so subcarriers whose amplitude
      moves in opposite directions can cancel.
    - ``"respirfi"``: RespirFi BNR selection, signed grouping and
      sign-flip averaging (`RespirFiFusion`), falling back to the variance
      weights whenever no subcarrier clears the BNR floor. On synthetic
      Fresnel blind spots at ESP32 bandwidth it is never worse than
      "variance" and slightly better (NOTES.md §6.5); off by default until
      compared on real recordings.

    Presence always uses the variance-weighted stream. It is calibrated on
    an empty room, where the selector has nothing to select, and its
    threshold must not change meaning when someone sits down.
    """

    def __init__(
        self,
        n_subcarriers: int = 56,
        sample_rate_hz: float = 20.0,
        calibration_s: float = 30.0,
        breathing_window_s: float | None = None,
        heart_window_s: float | None = None,
        suppress_vitals_when_empty: bool = True,
        subcarrier_selection: str = "variance",
    ) -> None:
        if subcarrier_selection not in ("variance", "respirfi"):
            raise ValueError(
                f"subcarrier_selection must be 'variance' or 'respirfi', got {subcarrier_selection!r}"
            )
        self.sample_rate_hz = sample_rate_hz
        self.suppress_vitals_when_empty = suppress_vitals_when_empty
        self.subcarrier_selection = subcarrier_selection

        self.suppressor = StaticSuppressor(n_subcarriers, sample_rate_hz)
        self.selector = (
            RespirFiFusion(n_subcarriers, sample_rate_hz)
            if subcarrier_selection == "respirfi"
            else None
        )
        self.presence = PresenceDetector(sample_rate_hz, calibration_s=calibration_s)
        self.breathing = BreathingExtractor(sample_rate_hz, window_s=breathing_window_s)
        self.heart = HeartRateExtractor(sample_rate_hz, window_s=heart_window_s)

        self._samples = 0
        # Intermediate signals of the most recent frame, exposed for the
        # FeatureFrame extractor (wifisense/features.py) so it never has to
        # re-run a stage and risk drifting from what the pipeline used.
        self.last_residuals: list[float] = []
        self.last_fused = 0.0
        self.last_vitals_input = 0.0

    def process(self, frame: CsiFrame) -> VitalReading:
        """Push one frame through every stage."""
        residuals, weights = self.suppressor.process(frame)
        fused = fuse(residuals, weights)

        presence_state = self.presence.process(fused)
        vitals_input = self.selector.process(residuals, weights) if self.selector else fused
        self.last_residuals, self.last_fused, self.last_vitals_input = residuals, fused, vitals_input
        self.breathing.push(vitals_input)
        self.heart.push(vitals_input)

        hide = self.suppress_vitals_when_empty and not presence_state.occupied
        if hide:
            breathing_est = VitalEstimate.unavailable()
            heart_est = VitalEstimate.unavailable()
        else:
            breathing_est = self.breathing.estimate()
            # The heart rate stage needs the breathing frequency to cancel
            # its harmonics and sidebands, but only trust it as a reference
            # when the breathing estimate itself is sound.
            breathing_hz = (
                breathing_est.value_bpm / 60.0
                if breathing_est.status
                in (VitalStatus.VALID, VitalStatus.DEGRADED)
                else None
            )
            heart_est = self.heart.estimate(breathing_hz)

        self._samples += 1
        return VitalReading(
            respiratory_rate=breathing_est,
            heart_rate=heart_est,
            subcarrier_count=len(residuals),
            signal_quality=min(1.0, self.breathing.fill_fraction),
            timestamp_s=self._samples / self.sample_rate_hz,
            occupied=presence_state.occupied,
            motion_energy=presence_state.energy,
        )

    def process_all(self, frames: list[CsiFrame]) -> VitalReading | None:
        """Run a batch and return only the final reading."""
        reading = None
        for frame in frames:
            reading = self.process(frame)
        return reading

    @property
    def calibrated(self) -> bool:
        return self.presence.calibrated

    def reset(self) -> None:
        self.suppressor.reset()
        if self.selector is not None:
            self.selector.reset()
        self.presence.recalibrate()
        self.breathing.reset()
        self.heart.reset()
        self._samples = 0
