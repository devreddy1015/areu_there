"""Signal processing primitives.

Pure stdlib on purpose. Everything here is a few dozen lines of float
arithmetic that ports directly to C for on-device use, which is the whole
point of picking Goertzel over an FFT: we only ever care about two narrow
bands, so evaluating the DFT at just those bins is both cheaper and
higher-resolution than binning the whole spectrum.
"""

from __future__ import annotations

import math

__all__ = [
    "BiquadBandpass",
    "goertzel_power",
    "hann_window",
    "SpectralPeak",
    "band_peak",
    "sideband_peak",
]


class BiquadBandpass:
    """Second-order constant-peak-gain bandpass (RBJ audio-EQ cookbook).

    An ad-hoc resonator with pole radius `r = 1 - bw/2` needs clamping to
    stay inside the unit circle, and its centre frequency drifts with the
    sample rate. A standard biquad costs the same per sample, is unconditionally stable for Q > 0, and puts the peak
    exactly where you asked for it.

    Implemented in transposed direct form II: the best-conditioned of the
    four classic topologies for float arithmetic.
    """

    def __init__(self, f_low: float, f_high: float, sample_rate_hz: float) -> None:
        if not 0.0 < f_low < f_high:
            raise ValueError(f"need 0 < f_low < f_high, got {f_low}, {f_high}")
        nyquist = sample_rate_hz / 2.0
        if f_high >= nyquist:
            raise ValueError(
                f"f_high ({f_high} Hz) must be below Nyquist ({nyquist} Hz); "
                f"raise the sample rate or lower the band"
            )

        self.f_low = f_low
        self.f_high = f_high
        self.sample_rate_hz = sample_rate_hz

        # Geometric mean is the correct centre for a log-symmetric response.
        f0 = math.sqrt(f_low * f_high)
        q = f0 / (f_high - f_low)

        w0 = 2.0 * math.pi * f0 / sample_rate_hz
        alpha = math.sin(w0) / (2.0 * q)
        cos_w0 = math.cos(w0)

        a0 = 1.0 + alpha
        self._b0 = alpha / a0
        self._b1 = 0.0
        self._b2 = -alpha / a0
        self._a1 = (-2.0 * cos_w0) / a0
        self._a2 = (1.0 - alpha) / a0

        self._s1 = 0.0
        self._s2 = 0.0

    @property
    def center_hz(self) -> float:
        return math.sqrt(self.f_low * self.f_high)

    def process(self, x: float) -> float:
        """Filter one sample.

        A non-finite input would otherwise poison the recurrence forever,
        so we reset rather than latch NaN.
        """
        if not math.isfinite(x):
            self.reset()
            return 0.0

        y = self._b0 * x + self._s1
        self._s1 = self._b1 * x - self._a1 * y + self._s2
        self._s2 = self._b2 * x - self._a2 * y

        if not math.isfinite(y):
            self.reset()
            return 0.0
        return y

    def process_all(self, xs: list[float]) -> list[float]:
        return [self.process(x) for x in xs]

    def reset(self) -> None:
        self._s1 = 0.0
        self._s2 = 0.0


def hann_window(n: int) -> list[float]:
    """Periodic Hann window.

    Without it, a strong breathing component leaks sidelobes across the
    whole band and buries the much weaker heartbeat peak.
    """
    if n <= 1:
        return [1.0] * max(n, 0)
    return [0.5 * (1.0 - math.cos(2.0 * math.pi * i / n)) for i in range(n)]


def goertzel_power(signal: list[float], freq_hz: float, sample_rate_hz: float) -> float:
    """|X(f)|^2 at an arbitrary (non-bin-aligned) frequency.

    This is the generalised Goertzel: the classic algorithm restricted to
    integer bins, with the final phase correction kept so that `freq_hz`
    need not land on a DFT bin. Cost is O(N) per frequency with two
    multiplies per sample, versus O(N log N) for a full FFT whose bin
    spacing we would then have to interpolate anyway.
    """
    n = len(signal)
    if n == 0 or sample_rate_hz <= 0.0:
        return 0.0

    k = freq_hz * n / sample_rate_hz
    w = 2.0 * math.pi * k / n
    cos_w = math.cos(w)
    sin_w = math.sin(w)
    coeff = 2.0 * cos_w

    s1 = 0.0
    s2 = 0.0
    for x in signal:
        s0 = x + coeff * s1 - s2
        s2 = s1
        s1 = s0

    real = s1 - s2 * cos_w
    imag = s2 * sin_w
    return real * real + imag * imag


class SpectralPeak:
    """Result of a band scan."""

    __slots__ = ("freq_hz", "power", "confidence")

    def __init__(self, freq_hz: float, power: float, confidence: float) -> None:
        self.freq_hz = freq_hz
        self.power = power
        self.confidence = confidence

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SpectralPeak(freq_hz={self.freq_hz:.4f}, "
            f"power={self.power:.4g}, confidence={self.confidence:.3f})"
        )


def band_peak(
    signal: list[float],
    sample_rate_hz: float,
    f_low: float,
    f_high: float,
    resolution_hz: float = 0.002,
    window: bool = True,
    notch_hz: list[float] | None = None,
    notch_width_hz: float = 0.0,
) -> SpectralPeak | None:
    """Find the dominant frequency inside [f_low, f_high].

    Returns the peak plus a confidence in [0, 1] defined as the fraction of
    in-band spectral energy concentrated at the peak. A clean sinusoid
    concentrates most of its energy in a couple of bins and scores high;
    band-limited noise spreads evenly and scores low. That ratio is what
    makes it possible to say "nobody is breathing here" instead of always
    reporting whatever the largest bin happened to be.

    Counting zero crossings would be cheaper, but it is biased by any residual DC offset, degrades badly once a second
    periodicity (the heartbeat) rides on the first, and yields no usable
    confidence signal.

    `notch_hz` excludes frequencies (within +/- `notch_width_hz`) from both
    the peak search and the confidence statistics. This is how the heart
    rate stage avoids locking onto a breathing harmonic — see
    `HeartRateExtractor`.
    """
    n = len(signal)
    if n == 0 or f_low <= 0.0 or f_high <= f_low or resolution_hz <= 0.0:
        return None

    if window:
        w = hann_window(n)
        signal = [s * wi for s, wi in zip(signal, w)]

    # Round, don't truncate. Every configuration in use puts this ratio on
    # an integer (72, 200, 288, 600), and truncation sits exactly on that
    # boundary: the same band in float32 computes 199.99998 and silently
    # drops the last grid point, shifting every confidence by ~0.5% (found
    # by the C++ port's randomised tests). Rounding gives the same count in
    # every precision.
    steps = int(round((f_high - f_low) / resolution_hz))
    if steps < 2:
        return None

    def is_notched(f: float) -> bool:
        if not notch_hz or notch_width_hz <= 0.0:
            return False
        return any(abs(f - nf) <= notch_width_hz for nf in notch_hz)

    freqs: list[float] = []
    powers: list[float] = []
    candidates: list[int] = []
    for i in range(steps + 1):
        f = f_low + i * resolution_hz
        freqs.append(f)
        powers.append(goertzel_power(signal, f, sample_rate_hz))
        if not is_notched(f):
            candidates.append(i)

    if not candidates:
        return None

    total = sum(powers[i] for i in candidates)
    if total <= 0.0:
        return None

    peak_i = max(candidates, key=powers.__getitem__)
    peak_f = freqs[peak_i]
    peak_p = powers[peak_i]

    # Parabolic interpolation across the peak for sub-resolution accuracy,
    # but never across a notched neighbour: that would pull the estimate
    # toward the frequency the caller asked to exclude.
    if 0 < peak_i < len(powers) - 1 and not is_notched(freqs[peak_i - 1]) and not is_notched(freqs[peak_i + 1]):
        y0, y1, y2 = powers[peak_i - 1], powers[peak_i], powers[peak_i + 1]
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            offset = 0.5 * (y0 - y2) / denom
            if -1.0 < offset < 1.0:
                peak_f = peak_f + offset * resolution_hz

    # Confidence, calibrated against what noise alone would produce.
    #
    # The naive "1 - mean/peak" scores band-limited noise around 0.75,
    # because the largest of M exponentially-distributed periodogram values
    # sits several times above their mean purely by chance. So compare the
    # observed peak-to-mean ratio against the ratio *expected from noise*.
    #
    # The number of statistically independent bins is set by the window
    # duration, not by how finely we scanned: n_ind = bandwidth * T. For
    # noise, E[peak/mean] over n_ind bins follows the Gumbel result
    # ln(n_ind) + gamma. For a pure tone, peak/mean approaches n_ind.
    # Rescaling between those two anchors puts noise near 0 and a clean
    # sinusoid near 1.
    mean_power = total / len(candidates)
    if peak_p <= 0.0 or mean_power <= 0.0:
        return SpectralPeak(peak_f, peak_p, 0.0)

    # A window function spreads even a perfect tone across several bins, so
    # the achievable peak-to-mean ratio is n_ind scaled down by the window's
    # equivalent noise bandwidth (1.5 bins for Hann, 1.0 for rectangular).
    # Without this correction a clean sinusoid tops out around 0.6.
    enbw = 1.5 if window else 1.0

    duration_s = n / sample_rate_hz
    n_ind = max((f_high - f_low) * duration_s, 2.0)
    ratio = peak_p / mean_power
    noise_ratio = math.log(n_ind) + 0.5772156649  # Euler-Mascheroni
    tone_ratio = n_ind / enbw
    span = tone_ratio - noise_ratio

    confidence = 0.0 if span <= 0.0 else (ratio - noise_ratio) / span
    confidence = max(0.0, min(1.0, confidence))

    return SpectralPeak(peak_f, peak_p, confidence)


def _noise_calibrated_confidence(
    peak_power: float,
    mean_power: float,
    n_samples: int,
    sample_rate_hz: float,
    f_low: float,
    f_high: float,
    enbw: float = 1.5,
) -> float:
    """Shared confidence scale — see the long note inside `band_peak`."""
    if peak_power <= 0.0 or mean_power <= 0.0 or sample_rate_hz <= 0.0:
        return 0.0
    duration_s = n_samples / sample_rate_hz
    n_ind = max((f_high - f_low) * duration_s, 2.0)
    noise_ratio = math.log(n_ind) + 0.5772156649
    span = (n_ind / enbw) - noise_ratio
    if span <= 0.0:
        return 0.0
    return max(0.0, min(1.0, ((peak_power / mean_power) - noise_ratio) / span))


def sideband_peak(
    signal: list[float],
    sample_rate_hz: float,
    f_low: float,
    f_high: float,
    modulator_hz: float,
    resolution_hz: float = 0.002,
    harmonic_tol_hz: float = 0.03,
) -> SpectralPeak | None:
    """Find a carrier hidden under a stronger modulating periodicity.

    Why this exists: CSI *amplitude* is the magnitude of a static multipath
    vector plus a body-reflected vector, so it is a nonlinear function of
    chest displacement. That nonlinearity does two things to the heart-rate
    band:

      1. It fills the band with harmonics of the breathing rate (k * f_b).
      2. It creates intermodulation sidebands at f_heart +/- k * f_b.

    On realistic data the f_heart + f_b sideband is often *stronger* than
    f_heart itself, so plain peak-picking in 0.8-2.0 Hz confidently returns
    the wrong number, and a zero-crossing counter has exactly this failure
    mode with no way to detect it.

    The fix exploits the structure: a true cardiac component appears as a
    symmetric triplet (f-f_b, f, f+f_b), while a spurious peak has no
    matching partner on both sides. Scoring candidates by
    `P(f) * sqrt(P(f-f_b) * P(f+f_b))` — a geometric mean, so *both*
    sidebands must be present — recovers the carrier. Breathing harmonics
    are excluded outright.

    Pass `modulator_hz` as the breathing frequency from the breathing stage.
    """
    n = len(signal)
    if n == 0 or modulator_hz <= 0.0 or f_high <= f_low or resolution_hz <= 0.0:
        return None

    w = hann_window(n)
    windowed = [s * wi for s, wi in zip(signal, w)]

    def power_at(f: float) -> float:
        return goertzel_power(windowed, f, sample_rate_hz) if f > 0.0 else 0.0

    def is_breathing_harmonic(f: float) -> bool:
        k = round(f / modulator_hz)
        return k >= 1 and abs(f - k * modulator_hz) <= harmonic_tol_hz

    # Round, don't truncate. Every configuration in use puts this ratio on
    # an integer (72, 200, 288, 600), and truncation sits exactly on that
    # boundary: the same band in float32 computes 199.99998 and silently
    # drops the last grid point, shifting every confidence by ~0.5% (found
    # by the C++ port's randomised tests). Rounding gives the same count in
    # every precision.
    steps = int(round((f_high - f_low) / resolution_hz))
    if steps < 2:
        return None

    scored: list[tuple[float, float]] = []
    for i in range(steps + 1):
        f = f_low + i * resolution_hz
        if is_breathing_harmonic(f):
            continue
        lower = power_at(f - modulator_hz)
        upper = power_at(f + modulator_hz)
        score = power_at(f) * math.sqrt(max(lower, 0.0) * max(upper, 0.0))
        scored.append((f, score))

    if not scored:
        return None

    best_f, best_score = max(scored, key=lambda fs: fs[1])
    if best_score <= 0.0:
        return None

    # Confidence here is peak *dominance*, not spectral concentration.
    #
    # The concentration measure used by `band_peak` is the wrong tool for
    # this band: a correct cardiac estimate still sits in a crowded
    # spectrum full of breathing harmonics and sidebands, so concentration
    # stays low even when the answer is right. What actually indicates a
    # trustworthy result is whether one triplet stands clear of every rival
    # triplet elsewhere in the band. So: compare the winning score against
    # the best score at least `separation_hz` away.
    separation_hz = max(modulator_hz * 0.5, 0.05)
    rivals = [s for f, s in scored if abs(f - best_f) > separation_hz]
    runner_up = max(rivals) if rivals else 0.0
    confidence = max(0.0, min(1.0, 1.0 - (runner_up / best_score)))

    return SpectralPeak(best_f, power_at(best_f), confidence)
