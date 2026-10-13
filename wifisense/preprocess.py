"""Static-environment suppression.

Walls, furniture and the direct path dominate CSI amplitude by orders of
magnitude over a breathing chest. Before anything else, subtract the slowly
varying baseline so that what remains is the body-modulated residual.
"""

from __future__ import annotations

import cmath
import math
from collections import deque

from .types import CsiFrame, RunningVariance

__all__ = [
    "StaticSuppressor",
    "corner_to_alpha",
    "alpha_to_corner",
    "fuse",
    "respirfi_select",
    "RespirFiFusion",
    "SubcarrierSelection",
    "symmetric_eigen",
    "subcarrier_bnr",
]


def corner_to_alpha(corner_hz: float, sample_rate_hz: float) -> float:
    """EMA smoothing factor giving a highpass corner at `corner_hz`.

    Subtracting an EMA from the signal is a first-order highpass:

        p[n] = a*x[n] + (1-a)*p[n-1]
        r[n] = x[n] - p[n-1]   =>   H(z) = (1 - z^-1) / (1 - (1-a)z^-1)

    whose -3 dB corner sits at roughly `a * fs / (2*pi)` for small `a`.
    Inverting that is the only safe way to choose `a`.

    This matters more than it looks. A fixed `alpha = 0.05` regardless of
    sample rate would, at 100 Hz, place the corner near 0.8 Hz, i.e. *above* the entire 0.1-0.5 Hz
    breathing band, attenuating the very signal the next stage is trying
    to measure by roughly 12 dB at 0.2 Hz. Tying `a` to the sample rate
    and to a corner below the band removes that self-inflicted loss.
    """
    if sample_rate_hz <= 0.0:
        raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
    alpha = 2.0 * math.pi * corner_hz / sample_rate_hz
    return min(max(alpha, 1e-6), 0.999)


def alpha_to_corner(alpha: float, sample_rate_hz: float) -> float:
    """Inverse of `corner_to_alpha`, for reporting what a given alpha does."""
    return alpha * sample_rate_hz / (2.0 * math.pi)


class StaticSuppressor:
    """Per-subcarrier baseline removal with variance-based weighting.

    Emits a residual vector (observed minus predicted baseline) plus a
    weight vector. Weights come from each subcarrier's running variance:
    subcarriers that actually move carry body signal, flat ones carry
    noise, and weighting by variance is a cheap approximation to a
    matched filter across the subcarrier axis.
    """

    def __init__(
        self,
        n_subcarriers: int,
        sample_rate_hz: float,
        corner_hz: float = 0.03,
    ) -> None:
        if n_subcarriers <= 0:
            raise ValueError(f"n_subcarriers must be positive, got {n_subcarriers}")

        self.n_subcarriers = n_subcarriers
        self.sample_rate_hz = sample_rate_hz
        self.corner_hz = corner_hz
        self.alpha = corner_to_alpha(corner_hz, sample_rate_hz)

        self._baseline = [0.0] * n_subcarriers
        self._seeded = [False] * n_subcarriers
        self._variance = [RunningVariance() for _ in range(n_subcarriers)]

    def process(self, frame: CsiFrame) -> tuple[list[float], list[float]]:
        """Return (residuals, weights) for one frame.

        Both lists are truncated to `min(frame.n_subcarriers, n_subcarriers)`.
        The first frame for a given subcarrier seeds its baseline and
        yields a zero residual — there is nothing to compare against yet.
        """
        n = min(frame.n_subcarriers, self.n_subcarriers)
        if n == 0:
            return [], []

        residuals = [0.0] * n
        for i in range(n):
            amp = frame.amplitudes[i]
            if not math.isfinite(amp):
                residuals[i] = 0.0
                continue

            if self._seeded[i]:
                residuals[i] = amp - self._baseline[i]
                self._baseline[i] = self.alpha * amp + (1.0 - self.alpha) * self._baseline[i]
            else:
                self._baseline[i] = amp
                self._seeded[i] = True
                residuals[i] = 0.0

            self._variance[i].update(residuals[i])

        return residuals, self.weights(n)

    def weights(self, n: int | None = None) -> list[float]:
        """Normalised per-subcarrier weights from running std deviation.

        Falls back to uniform weights until enough samples have accumulated
        for the variance estimate to mean anything.
        """
        count = self.n_subcarriers if n is None else min(n, self.n_subcarriers)
        stds = [self._variance[i].std for i in range(count)]
        total = sum(stds)
        if total <= 0.0:
            return [1.0 / count] * count if count else []
        return [s / total for s in stds]

    @property
    def effective_corner_hz(self) -> float:
        return alpha_to_corner(self.alpha, self.sample_rate_hz)

    def reset(self) -> None:
        self._baseline = [0.0] * self.n_subcarriers
        self._seeded = [False] * self.n_subcarriers
        for v in self._variance:
            v.reset()


def fuse(residuals: list[float], weights: list[float]) -> float:
    """Collapse per-subcarrier residuals into one scalar.

    Weights are renormalised over the subcarriers actually present, so a
    short frame does not silently scale the output down.
    """
    n = min(len(residuals), len(weights))
    if n == 0:
        return 0.0
    total_w = sum(weights[:n])
    if total_w <= 0.0:
        return sum(residuals[:n]) / n
    return sum(r * w for r, w in zip(residuals[:n], weights[:n])) / total_w


# ---------------------------------------------------------------------------
# RespirFi subcarrier selection (Wang et al., "High-Fidelity and
# Location-Robust Respiratory Waveform Monitoring with Single-Antenna WiFi",
# IEEE IoT-J 2026, arXiv:2604.20397, Section IV-C)
# ---------------------------------------------------------------------------
#
# Why the variance weighting above is not enough. Amplitude is
#
#     |h| = sqrt(|S|^2 + |B|^2 + 2|S||B| cos(theta)),
#
# so a breath moves |h| *up* on subcarriers where theta mod 2pi is in
# (0, pi) and *down* where it is in (pi, 2pi). `fuse` averages residuals
# with positive weights, so those two groups cancel each other. And where
# theta sits near 0 or pi the first-order response vanishes, leaving the
# second-order term at twice the breathing rate. RespirFi's answer:
#
#   1. score every subcarrier by breathing-to-noise ratio (BNR) and keep
#      those above eta = max(P80, 2)                         -- eq. 7, 8
#   2. cosine similarity between survivors, ternarised        -- eq. 9, 11
#   3. split into two anti-correlated groups (signed spectral clustering)
#                                                             -- eq. 10
#   4. drop weakly-affiliated subcarriers by score v_k        -- eq. 12
#   5. flip the sign of one group and average the z-scored series.
#
# Deviations from the paper, each forced by our setting and each a knob:
#
# - Band. The paper scores 0.16-0.5 Hz. We use the breathing extractor's
#   own band (0.1-0.5 Hz by default) so an 8 BPM breather is not rejected
#   by the selector and then searched for by the extractor.
# - BNR uses spectral *power*, not the magnitude sums of eq. 7. The paper
#   Savitzky-Golay + LOWESS smooths before scoring, which empties the
#   out-of-band bins; without that smoothing a magnitude sum is dominated
#   by how many noise bins there are. A clean 0.25 Hz tone at 30% noise
#   scores 0.66 by magnitude over 32 s at 4 Hz -- one strong peak loses to
#   51 out-of-band noise bins -- so the floor of 2 would reject a perfect
#   breathing subcarrier. With power, the floor means what the paper
#   intends: at least two thirds of the variance sits in the breathing
#   band. White noise scores (f_high - f_low) / (fs/2 - band) = 0.25.
# - Preprocessing: block-average to ~4 Hz (Nyquist 2 Hz) instead of
#   SG + LOWESS. Cheap, and it fixes what "out of band" means.
# - Similarity is computed on the in-band spectrum (equivalently, the
#   time-domain cosine similarity of the band-limited series, by Parseval)
#   so slow drift the static suppressor lets through cannot dominate it.
# - The ternarisation threshold and the v_k retention rule are not given
#   in the paper. Defaults are derived below, not tuned.
# - Partitioning maximises eq. 10 (= x^T W x) by its classic spectral
#   relaxation, the leading eigenvector of the ternary affinity, instead of
#   the matrix-power-mean signed Laplacian of the paper's ref. [46].
# - The paper's breathing-phase identification (Section IV-D, which way is
#   inhale) is not ported: rate estimation does not need it.
#
# What it cannot do: escape a blind spot that covers the whole band. The
# paper's own eq. 6 says subcarrier diversity guarantees an escape only
# when bandwidth * (d_body - d_LoS) >= 0.255 c. At the ~17.5 MHz an ESP32
# HT20 channel spans, that needs ~4.4 m of excess path. Nearer the link
# than that, every subcarrier can sit in the same null at once.


def _next_pow2(n: int) -> int:
    p = 1
    while p < n:
        p <<= 1
    return p


def _fft(x: list[complex]) -> list[complex]:
    """Iterative radix-2 FFT. `len(x)` must be a power of two.

    The selector needs the in-band bins for similarity and every bin for
    the BNR denominator, not just the two narrow bands Goertzel serves
    elsewhere, so a real FFT pays for itself here: 56 subcarriers x 128
    points is a few milliseconds.
    """
    n = len(x)
    a = list(x)
    j = 0
    for i in range(1, n):
        bit = n >> 1
        while j & bit:
            j ^= bit
            bit >>= 1
        j |= bit
        if i < j:
            a[i], a[j] = a[j], a[i]
    length = 2
    while length <= n:
        half = length // 2
        twiddles = [cmath.exp(-2j * math.pi * k / length) for k in range(half)]
        for start in range(0, n, length):
            for k in range(half):
                u = a[start + k]
                v = a[start + k + half] * twiddles[k]
                a[start + k] = u + v
                a[start + k + half] = u - v
        length <<= 1
    return a


def _percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (numpy's default definition)."""
    s = sorted(values)
    if not s:
        return 0.0
    pos = (len(s) - 1) * q / 100.0
    lo = int(math.floor(pos))
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (pos - lo)


def symmetric_eigen(matrix: list[list[float]], max_sweeps: int = 60) -> tuple[list[float], list[list[float]]]:
    """Eigen-decomposition of a small symmetric matrix by cyclic Jacobi.

    Returns `(eigenvalues, vectors)` where `vectors[i][j]` is component i
    of eigenvector j. Jacobi is slow for big matrices but unconditionally
    stable, exact to rounding, and twenty lines of arithmetic: the right
    trade for the <= ~20 x 20 affinity matrices the selector produces,
    and trivially portable to C.
    """
    n = len(matrix)
    a = [list(map(float, row)) for row in matrix]
    v = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]
    for _ in range(max_sweeps):
        off = sum(a[i][j] * a[i][j] for i in range(n) for j in range(i + 1, n))
        diag = sum(a[i][i] * a[i][i] for i in range(n))
        if off <= 1e-24 * max(diag, 1e-300):
            break
        for p in range(n):
            for q in range(p + 1, n):
                apq = a[p][q]
                if abs(apq) <= 1e-300:
                    continue
                theta = (a[q][q] - a[p][p]) / (2.0 * apq)
                t = 1.0 / (abs(theta) + math.sqrt(theta * theta + 1.0))
                if theta < 0.0:
                    t = -t
                c = 1.0 / math.sqrt(t * t + 1.0)
                s = t * c
                for k in range(n):
                    akp, akq = a[k][p], a[k][q]
                    a[k][p] = c * akp - s * akq
                    a[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = a[p][k], a[q][k]
                    a[p][k] = c * apk - s * aqk
                    a[q][k] = s * apk + c * aqk
                for k in range(n):
                    vkp, vkq = v[k][p], v[k][q]
                    v[k][p] = c * vkp - s * vkq
                    v[k][q] = s * vkp + c * vkq
    return [a[i][i] for i in range(n)], v


class SubcarrierSelection:
    """Outcome of one selection pass. Diagnostics are kept on purpose:
    on real data the first question is always *which* subcarriers it
    picked and why."""

    __slots__ = ("signs", "selected", "bnr", "bnr_threshold", "affinity_threshold", "candidates")

    def __init__(
        self,
        signs: dict[int, int],
        bnr: list[float],
        bnr_threshold: float,
        affinity_threshold: float,
        candidates: list[int],
    ) -> None:
        self.signs = signs                  # subcarrier -> +1 / -1 (retained only)
        self.selected = sorted(signs)
        self.bnr = bnr
        self.bnr_threshold = bnr_threshold
        self.affinity_threshold = affinity_threshold
        self.candidates = candidates        # passed the BNR stage

    @property
    def groups(self) -> tuple[list[int], list[int]]:
        return (
            [k for k in self.selected if self.signs[k] > 0],
            [k for k in self.selected if self.signs[k] < 0],
        )

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        g1, g2 = self.groups
        return (
            f"SubcarrierSelection(+{g1}, -{g2}, bnr_eta={self.bnr_threshold:.2f}, "
            f"candidates={len(self.candidates)})"
        )


def subcarrier_bnr(
    window: list[list[float]],
    sample_rate_hz: float,
    f_low: float = 0.1,
    f_high: float = 0.5,
) -> tuple[list[float], dict[int, list[complex]]]:
    """Per-subcarrier breathing-to-noise power ratio, plus in-band spectra.

    Each subcarrier's series is z-scored, zero-padded to a power of two and
    transformed; BNR is in-band power over out-of-band power (DC excluded).
    Returns `(bnr, spectra)` where `spectra[k]` holds subcarrier k's
    in-band bins (absent for non-finite or constant series). Also the
    `best_bnr` link-quality field of FeatureFrame v1 (spec/wire.md).
    """
    n_t = len(window)
    n_k = min((len(row) for row in window), default=0)
    if n_t < 8 or n_k == 0 or sample_rate_hz <= 0.0:
        return [0.0] * n_k, {}
    nfft = _next_pow2(n_t)
    bin_hz = sample_rate_hz / nfft
    half = nfft // 2
    in_band = [m for m in range(1, half + 1) if f_low <= m * bin_hz <= f_high]
    out_band = [m for m in range(1, half + 1) if not f_low <= m * bin_hz <= f_high]
    bnr = [0.0] * n_k
    spectra: dict[int, list[complex]] = {}
    if not in_band or not out_band:
        return bnr, spectra
    for k in range(n_k):
        col = [window[t][k] for t in range(n_t)]
        if not all(math.isfinite(x) for x in col):
            continue
        mean = sum(col) / n_t
        std = math.sqrt(sum((x - mean) ** 2 for x in col) / n_t)
        if std <= 0.0:
            continue
        z = [complex((x - mean) / std) for x in col] + [0j] * (nfft - n_t)
        spec = _fft(z)
        power = [abs(spec[m]) ** 2 for m in range(half + 1)]
        num = sum(power[m] for m in in_band)
        den = sum(power[m] for m in out_band)
        bnr[k] = num / den if den > 0.0 else float("inf")
        spectra[k] = [spec[m] for m in in_band]
    return bnr, spectra


def respirfi_select(
    window: list[list[float]],
    sample_rate_hz: float,
    f_low: float = 0.1,
    f_high: float = 0.5,
    bnr_floor: float = 2.0,
    bnr_percentile: float = 80.0,
    affinity_threshold: float | None = None,
    keep_fraction: float = 0.5,
) -> SubcarrierSelection | None:
    """Pick and sign subcarriers from a window of per-subcarrier samples.

    `window[t][k]` is subcarrier k at time t, sampled at `sample_rate_hz`
    (already low-passed / decimated; see `RespirFiFusion`). Returns None if
    no subcarrier clears the BNR threshold, which is the honest answer for
    an empty room or a whole-band blind spot.

    `affinity_threshold=None` derives the ternarisation level from
    significance rather than taste: band-limited noise of bandwidth B over
    T seconds has about 2BT independent samples, so the cosine similarity
    of two unrelated series has a standard deviation near 1/sqrt(2BT).
    Twice that is the ~95% level, ~0.39 for a 32 s window over 0.1-0.5 Hz.

    `keep_fraction` retains subcarriers whose eq. 12 score is at least that
    fraction of the best score. The paper only says "those with high
    similarity are retained".
    """
    if not window or not window[0]:
        return None
    n_t = len(window)
    n_k = min(len(row) for row in window)
    if n_t < 8 or sample_rate_hz <= 0.0:
        return None

    bnr, spectra = subcarrier_bnr(window, sample_rate_hz, f_low, f_high)
    if not spectra:
        return None

    eta = max(_percentile(bnr, bnr_percentile), bnr_floor)
    candidates = [k for k in range(n_k) if bnr[k] >= eta and k in spectra]

    if affinity_threshold is None:
        n_ind = 2.0 * (f_high - f_low) * n_t / sample_rate_hz
        affinity_threshold = min(0.9, 2.0 / math.sqrt(max(n_ind, 1.0)))

    if not candidates:
        return None
    if len(candidates) == 1:
        return SubcarrierSelection({candidates[0]: 1}, bnr, eta, affinity_threshold, candidates)

    # Cosine similarity of the in-band spectra (eq. 9, band-limited).
    energy = {k: math.sqrt(sum(abs(c) ** 2 for c in spectra[k])) for k in candidates}
    m = len(candidates)
    sim = [[0.0] * m for _ in range(m)]
    for i, ki in enumerate(candidates):
        sim[i][i] = 1.0
        for j in range(i + 1, m):
            kj = candidates[j]
            dot = sum((a * b.conjugate()).real for a, b in zip(spectra[ki], spectra[kj]))
            denom = energy[ki] * energy[kj]
            sim[i][j] = sim[j][i] = dot / denom if denom > 0.0 else 0.0

    # Ternary affinity (eq. 11), then the leading eigenvector of it as the
    # relaxation of max_x x^T W x over x in {-1, +1}^m (eq. 10).
    tern = [
        [1.0 if s >= affinity_threshold else -1.0 if s <= -affinity_threshold else 0.0 for s in row]
        for row in sim
    ]
    values, vectors = symmetric_eigen(tern)
    lead = max(range(m), key=values.__getitem__)
    x = [1 if vectors[i][lead] > 0.0 else -1 if vectors[i][lead] < 0.0 else 0 for i in range(m)]
    # Zero components (an isolated subcarrier) join whichever group they
    # agree with more; the score below discards them if that is nobody.
    for i in range(m):
        if x[i] == 0:
            pull = sum(sim[i][j] * x[j] for j in range(m) if j != i)
            x[i] = 1 if pull >= 0.0 else -1

    # Affiliation score (eq. 12): similarity to own group minus to the other.
    scores = [sum(sim[i][j] * x[i] * x[j] for j in range(m) if j != i) for i in range(m)]
    best = max(scores)
    if best <= 0.0:
        return None
    keep = [i for i in range(m) if scores[i] > 0.0 and scores[i] >= keep_fraction * best]

    # Orient so the larger (then stronger) group is positive. Arbitrary but
    # deterministic; streaming continuity is handled by the caller.
    plus = sum(scores[i] for i in keep if x[i] > 0)
    minus = sum(scores[i] for i in keep if x[i] < 0)
    flip = -1 if minus > plus else 1
    signs = {candidates[i]: flip * x[i] for i in keep}
    return SubcarrierSelection(signs, bnr, eta, affinity_threshold, candidates)


class RespirFiFusion:
    """Streaming wrapper: periodically re-select, fuse every frame.

    Selection is a windowed batch operation; the pipeline is per-frame. So
    residuals are block-averaged down to ~`analysis_rate_hz` into a ring
    buffer, `respirfi_select` re-runs every `update_s` seconds over the
    last `window_s`, and between updates each frame is fused with the
    current signed weights:

        fused = mean_sigma * mean_k( sign_k * r_k / sigma_k )

    i.e. the paper's z-score-flip-average, rescaled by the mean sigma of
    the chosen subcarriers so the output stays in amplitude units like
    `fuse` (the presence detector and the extractors never see a unit
    jump when selection kicks in).

    Two things the paper does not need but a stream does:

    - Orientation continuity. Group signs are arbitrary per pass. A sign
      flip mid-stream would turn the breathing extractor's history into a
      phase-reversed splice, so each new weight vector is oriented to
      agree with the previous one (or with the variance weights before the
      first pass) over the buffered window.
    - Holding. If a pass finds nothing (`None`), the previous weights are
      kept; before any pass succeeds, frames fall back to `fuse`.
    """

    def __init__(
        self,
        n_subcarriers: int,
        sample_rate_hz: float,
        f_low: float = 0.1,
        f_high: float = 0.5,
        window_s: float | None = None,
        update_s: float = 5.0,
        analysis_rate_hz: float = 4.0,
        **select_kwargs: float,
    ) -> None:
        if n_subcarriers <= 0:
            raise ValueError(f"n_subcarriers must be positive, got {n_subcarriers}")
        if sample_rate_hz <= 0.0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")
        self.n_subcarriers = n_subcarriers
        self.sample_rate_hz = sample_rate_hz
        self.f_low = f_low
        self.f_high = f_high
        # Same rule as the extractors: three cycles of the slowest rate.
        self.window_s = window_s if window_s is not None else 3.0 / f_low
        self.update_s = update_s
        self.decimation = max(1, int(round(sample_rate_hz / analysis_rate_hz)))
        self.analysis_rate_hz = sample_rate_hz / self.decimation
        if f_high >= self.analysis_rate_hz / 2.0:
            raise ValueError(
                f"analysis rate {self.analysis_rate_hz:g} Hz cannot represent {f_high} Hz; "
                f"raise analysis_rate_hz"
            )
        self.select_kwargs = select_kwargs

        self._window_len = max(8, int(self.window_s * self.analysis_rate_hz))
        self._update_every = max(1, int(round(update_s * sample_rate_hz)))
        self._buffer: deque[list[float]] = deque(maxlen=self._window_len)
        self._acc = [0.0] * n_subcarriers
        self._acc_n = 0
        self._frames = 0
        self._weights: list[float] | None = None
        self.selection: SubcarrierSelection | None = None
        self.passes = 0
        self.failed_passes = 0

    @property
    def weights(self) -> list[float] | None:
        """Current signed per-subcarrier weights (None before first pass)."""
        return None if self._weights is None else list(self._weights)

    def process(self, residuals: list[float], fallback_weights: list[float]) -> float:
        """Fuse one frame's residuals; occasionally re-run selection."""
        n = min(len(residuals), self.n_subcarriers)
        for k in range(n):
            r = residuals[k]
            self._acc[k] += r if math.isfinite(r) else 0.0
        self._acc_n += 1
        if self._acc_n >= self.decimation:
            self._buffer.append([a / self._acc_n for a in self._acc])
            self._acc = [0.0] * self.n_subcarriers
            self._acc_n = 0

        self._frames += 1
        if self._frames % self._update_every == 0 and len(self._buffer) >= self._window_len:
            self._reselect(fallback_weights)

        if self._weights is None:
            return fuse(residuals, fallback_weights)
        return sum(w * r for w, r in zip(self._weights[:n], residuals[:n]) if math.isfinite(r))

    def _reselect(self, fallback_weights: list[float]) -> None:
        window = list(self._buffer)
        self.passes += 1
        new = self._compute_weights(window)
        if new is None:
            self.failed_passes += 1
            return
        n = self.n_subcarriers

        # Orientation continuity against what the stream was carrying.
        if self._weights is not None:
            reference = self._weights
        else:
            total = sum(fallback_weights[:n])
            reference = [w / total for w in fallback_weights[:n]] if total > 0.0 else [1.0 / n] * n
        agreement = 0.0
        for row in window:
            a = sum(w * x for w, x in zip(new, row))
            b = sum(w * x for w, x in zip(reference, row))
            agreement += a * b
        if agreement < 0.0:
            new = [-w for w in new]

        self._weights = new

    def _compute_weights(self, window: list[list[float]]) -> list[float] | None:
        """Signed weights from one window, or None to keep the old ones.

        Separated out so evaluation baselines (PCA, oracle) can reuse the
        identical buffering, cadence and orientation logic and differ only
        in how they choose weights.
        """
        sel = respirfi_select(
            window, self.analysis_rate_hz, self.f_low, self.f_high, **self.select_kwargs
        )
        if sel is None:
            return None
        sigmas = {}
        for k in sel.selected:
            col = [row[k] for row in window]
            mean = sum(col) / len(col)
            sigmas[k] = math.sqrt(sum((x - mean) ** 2 for x in col) / len(col))
        live = [k for k in sel.selected if sigmas[k] > 0.0]
        if not live:
            return None
        mean_sigma = sum(sigmas[k] for k in live) / len(live)
        new = [0.0] * self.n_subcarriers
        for k in live:
            new[k] = sel.signs[k] * mean_sigma / (sigmas[k] * len(live))
        self.selection = sel
        return new

    def reset(self) -> None:
        self._buffer.clear()
        self._acc = [0.0] * self.n_subcarriers
        self._acc_n = 0
        self._frames = 0
        self._weights = None
        self.selection = None
        self.passes = 0
        self.failed_passes = 0
