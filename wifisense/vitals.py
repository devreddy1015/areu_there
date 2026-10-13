"""Vital-sign extraction from CSI residuals."""

from __future__ import annotations

from collections import deque

from .dsp import BiquadBandpass, band_peak, sideband_peak
from .types import VitalEstimate, VitalStatus

__all__ = ["BandVitalExtractor", "BreathingExtractor", "HeartRateExtractor", "MIN_CYCLES"]

# A periodogram cannot resolve a frequency it has not seen repeat. Three
# cycles is the practical floor for a stable estimate; two is enough to
# detect periodicity but the frequency estimate wanders badly.
MIN_CYCLES = 3.0


class BandVitalExtractor:
    """Bandpass + spectral peak-picking over a sliding window.

    The window length is derived from the *lowest* frequency in the band,
    not chosen arbitrarily. Resolving 0.1 Hz (a 10-second breath cycle)
    needs at least `MIN_CYCLES * 10 s = 30 s` of history; an estimate after
    10 seconds would cover less than two cycles at the bottom of the band
    and produce a number that is not yet meaningful.
    """

    def __init__(
        self,
        f_low: float,
        f_high: float,
        sample_rate_hz: float,
        window_s: float | None = None,
        resolution_hz: float | None = None,
    ) -> None:
        self.f_low = f_low
        self.f_high = f_high
        self.sample_rate_hz = sample_rate_hz

        self.min_window_s = MIN_CYCLES / f_low
        self.window_s = window_s if window_s is not None else self.min_window_s * 1.5
        if self.window_s < self.min_window_s:
            raise ValueError(
                f"window_s={self.window_s:.1f}s is too short to resolve "
                f"{f_low} Hz; need at least {self.min_window_s:.1f}s "
                f"({MIN_CYCLES:g} cycles)"
            )

        # Decimate before spectral analysis. Carrying 20 Hz samples to
        # resolve a 0.5 Hz band is 10x more arithmetic than the information
        # warrants: the band scan is O(window_samples) per candidate
        # frequency, so this is the difference between a pipeline that
        # keeps up in real time and one that does not -- and it is what
        # makes the same code viable on an ESP32.
        #
        # The bandpass ahead of it does the anti-alias duty. It is only a
        # single biquad (6 dB/octave), so target 4x the top of the band
        # rather than a bare Nyquist 2x to keep aliased residue well down.
        self.decimation = max(1, int(sample_rate_hz / (4.0 * f_high)))
        self.analysis_rate_hz = sample_rate_hz / self.decimation

        # Scanning much finer than a quarter of the true resolution (1/T)
        # buys nothing -- the window length, not the grid, bounds how well
        # two frequencies can be told apart.
        self.resolution_hz = (
            resolution_hz
            if resolution_hz is not None
            else max(1.0 / (4.0 * self.window_s), (f_high - f_low) / 400.0)
        )

        self._filter = BiquadBandpass(f_low, f_high, sample_rate_hz)
        self._max_samples = max(int(self.analysis_rate_hz * self.window_s), 1)
        self._min_samples = max(int(self.analysis_rate_hz * self.min_window_s), 1)
        self._history: deque[float] = deque(maxlen=self._max_samples)
        self._decimation_phase = 0

    def push(self, sample: float) -> None:
        """Filter one sample at the full rate; keep every Nth for analysis."""
        filtered = self._filter.process(sample)
        self._decimation_phase += 1
        if self._decimation_phase >= self.decimation:
            self._decimation_phase = 0
            self._history.append(filtered)

    def estimate(self) -> VitalEstimate:
        """Current estimate, or `unavailable` if the window is not yet full."""
        if len(self._history) < self._min_samples:
            return VitalEstimate.unavailable()

        peak = band_peak(
            list(self._history),
            self.analysis_rate_hz,
            self.f_low,
            self.f_high,
            resolution_hz=self.resolution_hz,
        )
        if peak is None:
            return VitalEstimate.unavailable()

        return VitalEstimate.from_confidence(peak.freq_hz * 60.0, peak.confidence)

    def process(self, sample: float) -> VitalEstimate:
        self.push(sample)
        return self.estimate()

    @property
    def ready(self) -> bool:
        return len(self._history) >= self._min_samples

    @property
    def fill_fraction(self) -> float:
        """How full the analysis window is, in [0, 1] — useful for a UI."""
        return min(1.0, len(self._history) / self._min_samples)

    def reset(self) -> None:
        self._history.clear()
        self._filter.reset()
        self._decimation_phase = 0


class BreathingExtractor(BandVitalExtractor):
    """Respiratory rate: 0.1-0.5 Hz, i.e. 6-30 breaths per minute.

    Covers resting adult (12-20 BPM) through infant (up to 30) and deep
    sleep (down to 6). Needs ~30 s of history at the low end.
    """

    def __init__(
        self,
        sample_rate_hz: float,
        window_s: float | None = None,
        f_low: float = 0.1,
        f_high: float = 0.5,
    ) -> None:
        super().__init__(f_low, f_high, sample_rate_hz, window_s)


class HeartRateExtractor(BandVitalExtractor):
    """Heart rate: 0.8-2.0 Hz, i.e. 48-120 BPM.

    Far harder than breathing. Chest-wall displacement from a heartbeat is
    roughly a tenth of that from a breath, and because CSI amplitude is a
    nonlinear function of displacement, breathing floods this band with
    harmonics (k * f_b) and intermodulation sidebands (f_h +/- k * f_b).
    The f_h + f_b sideband is frequently *stronger* than f_h itself.

    So pass the breathing frequency to `estimate()` and this uses
    `sideband_peak`, which requires a symmetric triplet and recovers the
    true carrier. Without it, the estimator falls back to plain peak
    picking, which on realistic input returns a sideband with no
    indication anything went wrong.

    Even with the triplet method: treat readings as provisional until you
    have validated them against a reference monitor, on your own hardware,
    with your own subjects. Nothing here is a medical device.
    """

    #: Resolving the cardiac carrier from its sidebands means resolving a
    #: spacing of one breathing period. At the slow end of the breathing
    #: band (0.1 Hz) that spacing is 0.1 Hz, needing >= 10 s of history just
    #: to separate them at all. The MIN_CYCLES rule alone would allow 5.6 s,
    #: which is enough to see a cardiac oscillation but nowhere near enough
    #: to tell it from a sideband. 60 s is the evidence-based default from
    #: `evaluate.py`; 90-120 s measurably improves the accept rate.
    DEFAULT_WINDOW_S = 60.0

    def __init__(
        self,
        sample_rate_hz: float,
        window_s: float | None = None,
        f_low: float = 0.8,
        f_high: float = 2.0,
    ) -> None:
        super().__init__(
            f_low,
            f_high,
            sample_rate_hz,
            window_s if window_s is not None else self.DEFAULT_WINDOW_S,
        )

    def estimate(self, breathing_hz: float | None = None) -> VitalEstimate:
        """Estimate heart rate, optionally suppressing breathing artefacts.

        `breathing_hz` should come from the breathing stage. When omitted,
        this degrades to the naive band peak and the result should be
        treated with corresponding suspicion.
        """
        if len(self._history) < self._min_samples:
            return VitalEstimate.unavailable()

        if breathing_hz is None or breathing_hz <= 0.0:
            return super().estimate()

        peak = sideband_peak(
            list(self._history),
            self.analysis_rate_hz,
            self.f_low,
            self.f_high,
            modulator_hz=breathing_hz,
            resolution_hz=self.resolution_hz,
        )
        if peak is None:
            return VitalEstimate.unavailable()
        return VitalEstimate.from_confidence(peak.freq_hz * 60.0, peak.confidence)

    def process(self, sample: float, breathing_hz: float | None = None) -> VitalEstimate:
        self.push(sample)
        return self.estimate(breathing_hz)
