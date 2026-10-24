"""Turning an RTI image into a position — or into an honest refusal.

This is the last stage of the localization chain: `invert` hands over a
flat image in `geometry.grid.pixels` order, and this module decides where
in the room the person is, how well it knows that, and — the part that
actually matters — *whether it knows at all*.

Why abstention is the headline feature and not an afterthought
--------------------------------------------------------------
The measurement feeding this chain is VRTI: the per-link **variance** of
the channel over a short window (Wilson & Patwari, IEEE TMC 2011). Variance
is a motion statistic. A person who stops walking stops producing it, and
within a window or two the image they were lighting up decays into the
same noise floor as an empty room. This is not a tuning problem; it is
what the sensor measures.

So a tracker that always returns coordinates is lying for a predictable
fraction of the time, and lying in the most damaging way available: a
plausible (x, y) inside the room, with no flag on it. Downstream that is
indistinguishable from a real fix. Every design choice below is pushed in
the direction of returning `None` instead:

  * significance is judged against the image's own robust noise floor, so
    it does not depend on the units the measurement happens to arrive in;
  * when that floor cannot be *measured* — a large block of pixels pinned
    at one exact value, which is what clamping, quantization and saturation
    all produce — the reading abstains instead of dividing by a denominator
    that has collapsed (see `extract_peak`);
  * the threshold is calibrated against measured noise-only images rather
    than guessed (see `DEFAULT_Z_THRESHOLD`);
  * every comparison that can see a NaN is written to fail *closed*, so a
    diverged upstream stage produces an abstention and not a confident fix;
  * while coasting through a gap the filter keeps running but `update`
    still returns `None` — the coast is state maintenance, never output;
  * a coast that runs longer than `max_coast_s` discards the state rather
    than extrapolating a walker who has almost certainly stopped.

What significance does NOT survive
----------------------------------
`z` is invariant to the scale of the measurement vector at fixed `alpha`
(the dual solve is linear in `y`, so scaling `y` scales peak and spread
together) and is near-indifferent to the prior length `delta_c`. It is NOT
invariant to the regularizer gain `alpha`, because `alpha` changes the
image's *shape* and not merely its scale. MEASURED, 8 nodes at 0.5 m pitch,
20 forward-projected targets at 15% link noise, delta_c = 0.5 m, only
`alpha` varying:

    alpha        0.1    1.0    10     100    300    1000
    median z    10.28  11.03  11.68   7.57   4.98   3.87
    detections  20/20  20/20  20/20  20/20  10/20   0/20

(delta_c really is near-irrelevant by comparison: median z stays in
10.7-11.2 across delta_c 0.4 to 2.0 m.) So a sweep that changes `alpha`
must re-histogram `z` and re-pick the threshold; that is why `extract_peak`
is public and returns the raw statistics rather than only a decision.

What this module does *not* do is decide that a quiet image means an empty
room. It cannot tell "nobody here" from "somebody here, holding still".
Both return `None`, and that ambiguity is real — resolving it needs the
breathing pipeline, not this one.

Pipeline inside `Tracker.update`
--------------------------------
    image -> robust background/spread -> peak pixel -> sub-pixel centroid
          -> significance test (abstain here) -> Kalman correct -> estimate

Pure stdlib, like the rest of the repo: the 4x4 filter is a handful of
float operations that port to C directly, and the only matrix inverse
needed is 2x2, which is written out algebraically.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .geometry import Geometry

__all__ = [
    "PositionEstimate",
    "PeakReading",
    "extract_peak",
    "Tracker",
    "DEFAULT_Z_THRESHOLD",
    "DEFAULT_CENTROID_RADIUS_M",
]


# Significance threshold on the robust z-score, in units of robust sigma.
#
# MEASURED, not guessed. Noise-only images were scored with exactly the
# `extract_peak` statistic (median background, MAD spread) and the fraction
# exceeding each threshold counted. 8 x 6 m room, 8 perimeter nodes.
#
# (a) iid Gaussian per pixel, 20000 images at 0.5 m pitch (P=192):
#
#       threshold   3.0     3.5     4.0     4.5     5.0     5.5    6.0
#       FP/image    0.270   0.068   0.013   0.0018  0.0002  5e-5   0
#
# (b) Spatially smoothed Gaussian — the realistic case, since the inversion
#     applies a prior C_pq = s^2 exp(-dist/delta_c) and its output is
#     correlated over delta_c. 6000 images per cell, threshold 5.0:
#
#       delta_c      0.4 m    0.7 m    1.0 m    1.5 m
#       FP (0.50 m)  0.0003   0.0013   0.0035   0.0067
#       FP (0.25 m)  0.0007   0.0023   0.0042   0.0085
#
# The surprise worth recording: smoothing makes this tail HEAVIER, not
# lighter. The intuition that correlated pixels mean fewer effective
# independent draws and therefore a tamer maximum is wrong here, because
# the MAD in the denominator is estimated from those same correlated
# pixels and so fluctuates far more than the iid case suggests. A
# threshold calibrated only on iid noise is optimistic by roughly an order
# of magnitude in false-positive rate. Hence 5.0 rather than the ~3.2 that
# sqrt(2 ln P) would suggest for P=192.
#
# (c) Pinned-pixel noise, the case that previously broke this module
#     outright. iid Gaussian clamped from below at its own f-quantile, so a
#     fraction f of pixels hold one identical value — what a non-negativity
#     clamp, a quantizer or a saturating sensor produces. 2000 images per
#     cell, threshold 5.0, with the pinned-block handling in `extract_peak`:
#
#       tie fraction f   0.00   0.30   0.50   0.60   0.70   0.74   >=0.75
#       FP (P=192)       2      0      1      6      17     31     0*
#       FP (P=768)       0      0      0      5      15     25     0*
#
#     * at or above 0.75 the image has too few free pixels to measure a
#       floor at all and `extract_peak` abstains outright.
#
# The residual false positives are not merely rare, they are *weak*: across
# all 48000 smoothed-noise images and all 32000 pinned-pixel images above,
# the largest confidence any false positive achieved was 0.387, i.e. below
# the 0.40 that `types.VitalEstimate` already grades UNRELIABLE. So the
# module's guarantee is the one the project needs — noise never produces a
# *confident* position — rather than the stronger one it cannot honestly
# make.
DEFAULT_Z_THRESHOLD = 5.0

# Radius of the centroid neighbourhood, in METRES rather than pixels, so
# that changing the grid pitch does not silently change the estimator.
#
# MEASURED, scanning a Gaussian blob across one pixel cell in the room
# interior (mean error in m; "nearest" is the peak pixel centre):
#
#   blob sigma   pitch   r=0.5 m   r=1.0 m   r=1.5 m   nearest
#      0.3 m     0.50     0.016     0.002     0.002     0.193
#      0.5 m     0.50     0.089     0.016     0.001     0.193
#      0.8 m     0.50     0.146     0.081     0.030     0.193
#      1.2 m     0.50     0.169     0.127     0.076     0.193
#
# A fixed 3x3 neighbourhood (r = one pitch) is a trap: on a blob wider than
# the pitch it pixel-locks, because truncating a wide blob at 3x3 clips
# more of whichever side extends further and drags the centroid back onto
# the peak pixel. At sigma = 1.2 m it barely beats nearest-pixel at all.
#
# Bigger is not simply better, though — with noise added, r=1.0 m wins.
# MEASURED over 400 random in-room targets, blob sigma 0.5 m, noise as a
# fraction of the peak, with the symmetric window described in
# `extract_peak`:
#
#   noise sd (frac of peak)   r=0.5 m   r=1.0 m   r=1.5 m   nearest
#      0.05, pitch 0.50        0.0988    0.0515    0.0583    0.2070
#      0.10, pitch 0.50        0.1123    0.0788    0.0945    0.2234
#      0.05, pitch 0.25        0.0750    0.0338    0.0334    0.1262
#      0.10, pitch 0.25        0.1006    0.0536    0.0570    0.1670
#
# because every extra ring of pixels admits more noise that survives the
# clamp at the background. 1.0 m is the compromise: near-optimal under
# realistic noise at both pitches.
#
# The wall band used to be the argument against a large radius, and is not
# any more. With the window clipped at the grid edge the radius mattered a
# great deal there and the centroid was worse than nearest-pixel outright;
# with the window shrunk symmetrically instead (see `extract_peak`) the
# radius barely matters in that band. MEASURED over 400 targets drawn from
# the outermost 0.5 m, noise 0.05:
#
#   pitch   r=0.5 m   r=1.0 m   r=1.5 m   nearest
#    0.50    0.1539    0.1426    0.1438    0.1999
#    0.25    0.0925    0.0773    0.0779    0.1207
#
# RESIDUAL BIAS, measured, and not removable by tuning this constant: the
# window is centred on the peak PIXEL rather than on the blob, so an
# off-centre blob gets a lopsided window and the centroid is dragged back
# toward that pixel. The signature is the classic pixel-locking S-curve —
# exactly zero on a pixel centre, worst at the midpoint between two
# centres, sign-flipping there as the peak changes hands. Worst |bias| is
# 0.0223 m at 0.5 m pitch (4.5% of a pitch) and 0.0184 m at 0.25 m pitch
# (7.3%). It is a bias, not noise: it does not average out over repeated
# measurements at one position, though it does partly cancel along a walk
# that crosses many cells. Removing it needs an iterated centroid that
# re-centres the window on its own estimate, or a model fit; neither is
# worth 2 cm against a forward model whose own error is far larger.
DEFAULT_CENTROID_RADIUS_M = 1.0

# Consistency constant turning a median absolute deviation into a sigma
# estimate for Gaussian data.
_MAD_TO_SIGMA = 1.4826

# A block of pixels holding one *exactly* equal value is a pinned block,
# not a sample of the noise: it is what a non-negativity clamp, a quantizer
# or a saturating sensor leaves behind. Once such a block is large enough to
# reach the middle of the sorted deviations it drives the MAD toward zero
# and the z-score toward infinity, so the block is excluded from the
# dispersion estimate and its value is taken as the background.
#
# 1/8 is MEASURED rather than chosen for tidiness. The failure is continuous
# in the block fraction f, not a cliff at f = 1/2: with a plain MAD the
# median deviation is the (0.5-f)/(1-f) quantile of the free pixels, so an
# f of 0.4 already reads the 17th percentile of the noise as its 50th. Pure
# Gaussian noise clamped at its own f-quantile, 3000 images per cell at
# P=192, counting images scoring z >= 5.0:
#
#   trigger at f >=   0.000  0.124  0.130  0.200  0.300  0.400  0.500
#     1/2 (naive)       2      2      2      2     130   2684*     2
#     1/8 (this)        2      2      0      0      0      0       2
#
#   * and 685 of the 2684 were CONFIDENT. The naive rule is not merely
#     imprecise, it manufactures certainty from noise.
#
# Below 1/8 the uncorrected MAD is biased by at most a factor
# (0.5-0.125)/(1-0.125) -> 0.43-quantile, about 1.2x, which the threshold
# absorbs; the table above shows no excess there.
_TIE_MIN_FRACTION = 0.125

# ... and the mirror of it: a floor estimated from a handful of pixels is
# not an estimate. Below this fraction of free (non-pinned) pixels the
# image is declared uninformative and `extract_peak` returns z = 0 rather
# than a ratio against a denominator nobody should believe. MEASURED (table
# above, extended): at f = 0.74 the false-positive rate is already 31/2000
# and climbing, so the cut sits just beyond it at f = 0.75.
_MIN_FREE_FRACTION = 0.25


@dataclass(frozen=True)
class PositionEstimate:
    """One accepted fix: where, how well known, and how much to believe it.

    `sigma_m` is the filter's own 1-sigma position uncertainty in metres,
    `sqrt(P[0][0] + P[1][1])` — the RMS of the two-axis error, which is the
    quantity directly comparable to the Euclidean error an evaluation
    script measures against ground truth. It is a *filter-internal*
    uncertainty: it reflects the assumed measurement and process noise, so
    it is only as honest as those assumptions, and it says nothing about
    bias from a wrong weight model.

    `confidence` is a property of the image, not of the filter — it scores
    how far the peak stood clear of the image's own noise floor. The two
    can disagree, and that is informative: a small `sigma_m` with a low
    `confidence` means the filter is coasting smoothly on weak evidence.
    """

    x: float
    y: float
    sigma_m: float
    confidence: float
    t: float = 0.0
    # The filter's position covariance at this estimate (spec/locate.md §3):
    # `P[0][0]`, `P[0][1]`, `P[1][1]` of the same matrix `sigma_m` is read
    # from. An estimate built without them has a zero covariance, so radius
    # 0, which is never published (spec/locate.md §8; min_plausible_radius_m
    # is validated > 0, §9).
    cov_xx: float = 0.0
    cov_xy: float = 0.0
    cov_yy: float = 0.0


@dataclass(frozen=True)
class PeakReading:
    """Raw image statistics, before any significance decision is applied.

    Separated from `Tracker` so the peak finder can be tested and swept on
    its own, and so an evaluation script can histogram `z` over a run to
    choose a threshold from data instead of inheriting this module's.

    `peak_index` is `None` — not -1 — when the image contained non-finite
    values and no peak could be located. -1 would be a valid Python index
    and would silently resolve to the far corner of the room; `None` makes
    `grid.pixels[reading.peak_index]` raise, which is the whole point of a
    sentinel.

    `x`/`y` are the sub-pixel centroid and are NaN in that same case; NaN
    rather than the room centre on purpose, because a silent fallback to
    the centre is exactly the failure mode that makes a broken localizer
    look like a working one.

    `tie_fraction` is the share of pixels sitting in a pinned block at one
    exactly-equal value, and is the diagnostic for *why* a finite image
    scored z = 0: at or above `1 - _MIN_FREE_FRACTION` the noise floor was
    not measurable at all. It is meaningful only when `n_nonfinite` is 0.
    """

    x: float
    y: float
    peak: float
    background: float
    spread: float
    z: float
    peak_index: int | None
    n_nonfinite: int = 0
    tie_fraction: float = 0.0


def _median(sorted_values: list[float]) -> float:
    """Median of an already-sorted list, written so it cannot overflow.

    `0.5 * (a + b)` is the obvious form and it is wrong at the top of the
    float range: a + b overflows to inf once the two straddle 1.8e308, so a
    perfectly flat image of 1e308 used to report an infinite background and
    hand a NaN z-score to a comparison that then accepted it. Halving each
    term first is exact (0.5 is a power of two) and the single remaining
    addition cannot overflow.
    """
    n = len(sorted_values)
    mid = n // 2
    if n % 2:
        return sorted_values[mid]
    return 0.5 * sorted_values[mid - 1] + 0.5 * sorted_values[mid]


def _dominant_run(sorted_values: list[float]) -> tuple[float, int]:
    """Value and length of the longest run of exactly-equal values.

    Exact equality rather than a tolerance, deliberately. The mechanisms
    that pin a block of pixels — a non-negativity clamp, a quantizer, a
    saturating sensor, an untouched region of a hand-built image — all
    produce bit-identical floats, while a genuinely noisy image produces
    distinct ones. A tolerance would need a scale, and every scale that is
    large enough to be useful is also large enough to swallow real
    low-contrast structure.
    """
    best_value = sorted_values[0]
    best_count = 1
    run_value = sorted_values[0]
    run_count = 1
    for value in sorted_values[1:]:
        if value == run_value:
            run_count += 1
        else:
            run_value = value
            run_count = 1
        if run_count > best_count:
            best_count = run_count
            best_value = run_value
    return best_value, best_count


def extract_peak(
    geometry: Geometry,
    image: list[float],
    radius_m: float = DEFAULT_CENTROID_RADIUS_M,
) -> PeakReading:
    """Locate the image maximum to sub-pixel accuracy and score it.

    Background and spread are a median and a MAD. Robust statistics rather
    than mean and standard deviation because a genuine detection is, by
    construction, a cluster of outliers: using the mean would let the target
    inflate the very noise floor it is being compared against, which costs
    sensitivity exactly when there is something to see.

    How much target the MAD tolerates, measured rather than quoted
    --------------------------------------------------------------
    The classical answer is "50%, and a blob never covers half the room",
    but that figure is the breakdown point — where the estimator fails
    completely — and the sensitivity loss is continuous and already severe
    well below it, because the target's own pixels inflate the denominator
    long before they dominate it. MEASURED on the default 8 x 6 m / 0.5 m
    geometry, one Gaussian target at (4.0, 3.0), sweeping its width
    (coverage is the share of pixels above half the peak):

        blob sigma   coverage    z        outcome
          1.2 m        12%       9.90     detected, confidence 0.495
          1.6 m        27%       3.37     ABSTAINS on a genuine target
          2.0 m        40%       2.20     abstains

    So the practical cliff is near 20% coverage, not 50%: a broad
    reconstruction suppresses its own significance. The honest caveat on
    reachability is that no image this broad came out of this repo's own
    `Inverter` at any delta_c from 0.4 to 2.0 m (median z stayed 10.7-11.2,
    detection 20/20), so this needs a forward model broader than the
    current pipeline produces — which is exactly the model-mismatch regime
    the inverse-crime caveat says real data will land in.

    Pinned pixels, and why the MAD alone is not enough
    -------------------------------------------------
    The MAD's other failure has nothing to do with the target. If a block
    of pixels holds one *exactly equal* value and that block reaches the
    middle of the sorted deviations, the MAD is measuring the width of the
    block, which is zero by construction — and z = (peak - background) /
    spread then explodes on an image carrying no information at all. Every
    realistic source of such a block is a thing a caller plausibly does:
    the standard RTI non-negativity clamp (attenuation cannot be negative),
    quantization, a saturating sensor. So the dominant pinned block, if it
    is at least `_TIE_MIN_FRACTION` of the image, is taken as the background
    and excluded from the dispersion, which leaves the free pixels to
    measure the floor. That estimate is exactly right when the block is a
    clamp at the noise centre (the free deviations are then the positive
    half of a symmetric distribution, whose median matches the full one)
    and conservative when it is not. If fewer than `_MIN_FREE_FRACTION` of
    the pixels are free, there is nothing left to measure the floor with
    and the reading abstains with z = 0 — never with a fallback dispersion,
    because any fallback that still returns a number turns "no information"
    into "strong detection" for free, which is the one thing this module
    exists to prevent.

    The centroid subtracts that background and clamps negatives to zero
    before weighting. Both halves matter. Without the subtraction a large
    positive pedestal drags every centroid onto its own peak pixel and the
    sub-pixel resolution is lost; without the clamp, negative pixels — which
    a regularized inversion produces as ringing around a positive blob —
    enter as negative mass and can throw the centroid off the target.
    """
    n_pixels = geometry.n_pixels
    if len(image) != n_pixels:
        raise ValueError(
            f"image has {len(image)} pixels, geometry has {n_pixels}; "
            f"images must be in geometry.grid.pixels order"
        )
    if n_pixels == 0:
        raise ValueError("geometry has no pixels")
    if not radius_m > 0.0:
        raise ValueError(f"radius_m must be positive, got {radius_m}")

    n_nonfinite = sum(1 for v in image if not math.isfinite(v))
    if n_nonfinite:
        # A diverged inversion must not be allowed to produce a position.
        # z = 0 guarantees every caller abstains; `n_nonfinite` says why.
        return PeakReading(
            x=math.nan,
            y=math.nan,
            peak=math.nan,
            background=math.nan,
            spread=0.0,
            z=0.0,
            peak_index=None,
            n_nonfinite=n_nonfinite,
        )

    grid = geometry.grid
    peak_index = max(range(n_pixels), key=image.__getitem__)
    peak = image[peak_index]

    ordered = sorted(image)
    pinned_value, pinned_count = _dominant_run(ordered)
    if pinned_count >= _TIE_MIN_FRACTION * n_pixels:
        background = pinned_value
        free = sorted(abs(v - background) for v in image if v != pinned_value)
    else:
        background = _median(ordered)
        free = sorted(abs(v - background) for v in image)
        pinned_count = 0
    tie_fraction = pinned_count / n_pixels

    if len(free) < _MIN_FREE_FRACTION * n_pixels:
        # Too few pixels left to measure a noise floor from. Abstain rather
        # than invent one: see the docstring.
        spread = 0.0
    else:
        spread = _MAD_TO_SIGMA * _median(free)

    if spread > 0.0 and math.isfinite(spread):
        z = (peak - background) / spread
        if not math.isfinite(z) or z < 0.0:
            z = 0.0
    else:
        spread = 0.0 if not math.isfinite(spread) else spread
        z = 0.0

    # Neighbourhood in pixels, derived from the metre radius per axis so a
    # non-square pitch stays physically symmetric. Floor, with a minimum of
    # one, so a requested neighbourhood is never degenerate on a coarse grid.
    rx = max(1, int(radius_m / grid.pitch_x_m))
    ry = max(1, int(radius_m / grid.pitch_y_m))

    ix0 = peak_index % grid.n_x
    iy0 = peak_index // grid.n_x

    # Shrink the window symmetrically at the grid edge instead of letting
    # the edge clip it on one side. A one-sided window has mass on one side
    # of the peak pixel and none on the other, so its centroid is dragged
    # systematically inward — toward the room centre, which is precisely the
    # artifact that makes a localizer score well on a random walk without
    # knowing anything. MEASURED, mean error over a scan of each pixel row
    # at blob sigma 0.5 m, clipped window vs symmetric vs nearest-pixel:
    #
    #   pitch 0.50  row 0  0.2568 / 0.1260 / 0.1906      (clipped LOSES)
    #               row 1  0.0638 / 0.0593 / 0.1906
    #   pitch 0.25  row 0  0.3070 / 0.0639 / 0.0953      (clipped LOSES)
    #               row 1  0.1792 / 0.0541 / 0.0953      (clipped LOSES)
    #               row 2  0.0869 / 0.0387 / 0.0953
    #
    # The interior is untouched (0.01532 m both ways at 0.5 m pitch, 0.01383
    # at 0.25 m) because `min` returns the requested radius there. In the
    # outermost cell the window collapses to the peak pixel itself and the
    # estimator degrades gracefully to nearest-pixel rather than to
    # something worse than nearest-pixel.
    rx = min(rx, ix0, grid.n_x - 1 - ix0)
    ry = min(ry, iy0, grid.n_y - 1 - iy0)

    sum_w = 0.0
    sum_wx = 0.0
    sum_wy = 0.0
    for dy in range(-ry, ry + 1):
        # No bounds test: rx and ry were clipped to the distance from the
        # peak pixel to each edge, so every index below is in range by
        # construction. That is the invariant that makes the window
        # symmetric, so it is worth stating structurally rather than
        # re-checking.
        row = (iy0 + dy) * grid.n_x
        for dx in range(-rx, rx + 1):
            index = row + ix0 + dx
            w = image[index] - background
            if w <= 0.0:
                continue
            px, py = grid.pixels[index]
            sum_w += w
            sum_wx += w * px
            sum_wy += w * py

    if sum_w > 0.0:
        cx = sum_wx / sum_w
        cy = sum_wy / sum_w
    else:
        # Peak is at or below the background: there is nothing to weight.
        # Fall back to the peak pixel centre. z will be <= 0 here, so this
        # reading is destined to be rejected anyway.
        cx, cy = grid.pixels[peak_index]

    return PeakReading(
        x=cx,
        y=cy,
        peak=peak,
        background=background,
        spread=spread,
        z=z,
        peak_index=peak_index,
        tie_fraction=tie_fraction,
    )


def _identity(n: int) -> list[list[float]]:
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


def _mat_mul(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    n = len(a)
    k = len(b)
    m = len(b[0])
    out = [[0.0] * m for _ in range(n)]
    for i in range(n):
        ai = a[i]
        oi = out[i]
        for p in range(k):
            aip = ai[p]
            if aip == 0.0:
                continue
            bp = b[p]
            for j in range(m):
                oi[j] += aip * bp[j]
    return out


def _mat_t(a: list[list[float]]) -> list[list[float]]:
    return [list(col) for col in zip(*a)]


def _mat_add(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[x + y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _mat_sub(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[x - y for x, y in zip(ra, rb)] for ra, rb in zip(a, b)]


def _symmetrize(a: list[list[float]]) -> list[list[float]]:
    """Average a matrix with its transpose.

    The Joseph update is already structurally symmetric, but rounding
    breaks that at the 1e-16 level and the asymmetry compounds over
    thousands of updates. One cheap averaging per step keeps the covariance
    exactly symmetric, which is what the sigma extraction assumes.
    """
    n = len(a)
    return [[0.5 * (a[i][j] + a[j][i]) for j in range(n)] for i in range(n)]


class Tracker:
    """Constant-velocity Kalman filter over (x, y, vx, vy), with abstention.

    State is `[x, y, vx, vy]` in metres and metres/second. The motion model
    is the standard continuous white-noise acceleration (CWNA) one: velocity
    is a random walk driven by white acceleration of power spectral density
    `accel_psd`, and position is its integral.

    Choosing `accel_psd` physically
    -------------------------------
    The useful reading of the PSD is that over an interval `dt` the model
    expects velocity to wander by `sqrt(accel_psd * dt)` m/s, i.e. an RMS
    acceleration of `sqrt(accel_psd / dt)`. At the default 0.1 m^2/s^3 and
    a 0.4 s update interval (8 nodes x 50 ms TDMA slots) that is 0.2 m/s of
    velocity change per step, an RMS acceleration of 0.5 m/s^2.

    That is the right order for the target this is built for. A person
    walking at the 0.3-1.2 m/s of the simulator's random-waypoint model
    does not accelerate hard; 0.5 m/s^2 lets the filter follow a normal
    turn while still averaging several measurements together on a straight
    leg. Raising it tracks turns faster and smooths less, lowering it the
    reverse — it is the single knob that sets that trade, and it is a
    *modelling* choice, not a measured constant. Someone tracking a running
    target should raise it.

    `meas_noise_m` defaults to the grid pitch, because the sub-pixel
    centroid's error under realistic noise measures at a fraction of a
    pitch (see `DEFAULT_CENTROID_RADIUS_M`), and the remaining error is
    dominated by the forward model's own bias rather than by the grid. It
    is inflated for weak detections by `(2 - confidence)`, so a
    barely-significant peak is trusted half as much as a clean one. That
    inflation is a heuristic — it has the right monotonicity but no
    calibration behind it.

    Abstain, coast, reset
    ---------------------
    `update` returns `None` whenever the image peak fails the significance
    test. While that is happening the filter still predicts forward, so the
    state ages and `P` grows through `Q`; `sigma_m` therefore rises during a
    gap and falls again on the first accepted measurement. The coasted
    state is deliberately *not* emitted — see the module docstring.

    After `max_coast_s` without an accepted measurement the state is thrown
    away and the next accepted measurement re-initializes the filter. The
    default 3.0 s is set by what coasting is worth: at a walking 1 m/s,
    3 s of extrapolation is 3 m in an 8 x 6 m room, and the prior velocity
    spread of 1 m/s means the prediction's own sigma has already grown to
    about 3 m by then — no better than a uniform guess over the room. More
    to the point, VRTI abstains *because the person stopped moving*, so the
    constant-velocity assumption is not merely uncertain after a long gap,
    it is specifically wrong. Resetting costs one measurement of
    reacquisition and removes the possibility of snapping back from a
    fictional extrapolated position.

    What that expiry must NOT do is fire on a slow but unbroken stream. The
    gap it measures is the time since the last *accepted* fix, which on a
    frame that itself carries a measurement is just the frame interval, so
    the test is against `max(max_coast_s, dt)`: one frame's own interval is
    never a gap, however long it is. Without that, a mesh whose per-link
    rate is below `1/max_coast_s` — 9 nodes at 1.4 Hz aggregate is 0.175 Hz
    per link, well inside this regime — would reset and re-initialize on
    every single frame, silently stop filtering, and pin `sigma_m` at its
    single-measurement value while `is_tracking` stayed True.
    """

    def __init__(
        self,
        geometry: Geometry,
        z_threshold: float = DEFAULT_Z_THRESHOLD,
        centroid_radius_m: float = DEFAULT_CENTROID_RADIUS_M,
        accel_psd: float = 0.1,
        meas_noise_m: float | None = None,
        init_speed_std_ms: float = 1.0,
        max_coast_s: float = 3.0,
        clamp_to_room: bool = True,
    ) -> None:
        # Written as `not (x > 0)` throughout rather than `x <= 0`, so a
        # NaN is REJECTED. Every comparison against a NaN is False, so the
        # natural spelling lets one straight through, and a NaN
        # `z_threshold` in particular would turn the significance test
        # into a coin that always lands the same way.
        if not z_threshold > 0.0:
            raise ValueError(f"z_threshold must be positive, got {z_threshold}")
        if not centroid_radius_m > 0.0:
            raise ValueError(
                f"centroid_radius_m must be positive, got {centroid_radius_m}"
            )
        if not accel_psd > 0.0:
            raise ValueError(f"accel_psd must be positive, got {accel_psd}")
        if not init_speed_std_ms > 0.0:
            raise ValueError(
                f"init_speed_std_ms must be positive, got {init_speed_std_ms}"
            )
        if not max_coast_s >= 0.0:
            raise ValueError(f"max_coast_s must be non-negative, got {max_coast_s}")

        grid = geometry.grid
        if meas_noise_m is None:
            meas_noise_m = 0.5 * (grid.pitch_x_m + grid.pitch_y_m)
        if not meas_noise_m > 0.0:
            raise ValueError(f"meas_noise_m must be positive, got {meas_noise_m}")

        self.geometry = geometry
        self.z_threshold = z_threshold
        self.centroid_radius_m = centroid_radius_m
        self.accel_psd = accel_psd
        self.meas_noise_m = meas_noise_m
        self.init_speed_std_ms = init_speed_std_ms
        self.max_coast_s = max_coast_s
        self.clamp_to_room = clamp_to_room

        self._state: list[float] | None = None
        self._cov: list[list[float]] | None = None
        # None until the first frame, so that a caller whose clock does not
        # start at zero does not have a spurious elapsed interval on its
        # first update.
        self._t: float | None = None
        self._last_fix_t: float | None = None

    # -- introspection -------------------------------------------------
    #
    # These expose the coasted state for diagnostics and for the evaluation
    # script. They are NOT an output channel: an estimate the tracker was
    # not willing to return from `update` is not one a caller should treat
    # as a fix.

    @property
    def is_tracking(self) -> bool:
        return self._state is not None

    @property
    def position(self) -> tuple[float, float] | None:
        return None if self._state is None else (self._state[0], self._state[1])

    @property
    def velocity(self) -> tuple[float, float] | None:
        return None if self._state is None else (self._state[2], self._state[3])

    @property
    def sigma_m(self) -> float | None:
        """Current 1-sigma position uncertainty, including while coasting."""
        if self._cov is None:
            return None
        return math.sqrt(max(self._cov[0][0] + self._cov[1][1], 0.0))

    def reset(self) -> None:
        """Forget the track. The next accepted measurement starts fresh."""
        self._state = None
        self._cov = None

    # -- filter internals ----------------------------------------------

    def _process_noise(self, dt: float) -> list[list[float]]:
        """CWNA process noise, per axis, assembled into the 4x4.

        Per axis the continuous model integrates to

            Q = q * [[dt^3/3, dt^2/2],
                     [dt^2/2, dt   ]]

        The off-diagonal position/velocity terms are the reason to use this
        rather than a diagonal fudge: they encode that a velocity error and
        the position error it causes are the *same* error seen twice. This
        is the exact integral of the stated model, and that — not a
        direction of conservatism — is the whole justification.

        An earlier version of this comment claimed that dropping the cross
        terms makes `sigma_m` optimistic. MEASURED, it does the opposite.
        Running the position/velocity recursion longhand at dt = 1.2 s,
        accel_psd = 0.1, R = 0.09, to steady state and then coasting one
        step, the position variance comes out at 0.3850 with the cross
        terms and 0.5025 without, i.e. the diagonal fudge is *pessimistic*
        here, because the correction it allows removes less velocity
        variance and the next coast then inflates position by more. The
        sign of that error depends on dt, q and R, which is precisely the
        argument for integrating the model rather than fudging it: a wrong
        Q is not wrong in a dependable direction.

        x and y are independent — the model has no reason to prefer any
        direction — so the two axes share no cross terms.
        """
        q = self.accel_psd
        q_pp = q * dt * dt * dt / 3.0
        q_pv = q * dt * dt / 2.0
        q_vv = q * dt
        return [
            [q_pp, 0.0, q_pv, 0.0],
            [0.0, q_pp, 0.0, q_pv],
            [q_pv, 0.0, q_vv, 0.0],
            [0.0, q_pv, 0.0, q_vv],
        ]

    def _predict(self, dt: float) -> None:
        assert self._state is not None and self._cov is not None
        x, y, vx, vy = self._state
        self._state = [x + vx * dt, y + vy * dt, vx, vy]

        f = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
        cov = _mat_mul(_mat_mul(f, self._cov), _mat_t(f))
        self._cov = _symmetrize(_mat_add(cov, self._process_noise(dt)))

    def _correct(self, zx: float, zy: float, r: float) -> None:
        """Kalman correction with H = [[1,0,0,0],[0,1,0,0]], R = r*I.

        The covariance update is the Joseph form,

            P = (I - KH) P (I - KH)^T + K R K^T

        rather than the cheaper `(I - KH) P`. The two are algebraically
        equal at the optimal gain, so the difference is purely numerical and
        only shows up once `(I - KH)` is close to singular — which is to say
        once R is tiny beside P and the gain saturates at one. MEASURED over
        300 steps of this exact filter at dt = 0.4, accel_psd = 0.1, varying
        only R, reporting the smallest covariance diagonal reached:

            r        Joseph      short form
            1e-6     9.997e-7    9.997e-7
            1e-12    1.000e-12   1.000e-12
            1e-18    1.000e-18   0.0        <- collapsed
            1e-24    1.000e-24   0.0        <- collapsed

        At the default `meas_noise_m` the two are indistinguishable, so this
        is insurance rather than a live fix; the claim is bounded to what
        the table shows, and a regression test pins the r = 1e-18 column.
        At 4x4 the extra multiplies cost microseconds, which is not a trade
        worth thinking about twice.
        """
        assert self._state is not None and self._cov is not None
        p = self._cov

        # S = H P H^T + R, a 2x2 taken straight from the position block.
        s00 = p[0][0] + r
        s01 = p[0][1]
        s10 = p[1][0]
        s11 = p[1][1] + r
        det = s00 * s11 - s01 * s10
        if not math.isfinite(det) or abs(det) <= 0.0:
            # Cannot happen with r > 0 and P positive semi-definite, but a
            # silent NaN here would corrupt every later estimate, so refuse
            # the correction instead of propagating it.
            return
        inv00 = s11 / det
        inv01 = -s01 / det
        inv10 = -s10 / det
        inv11 = s00 / det

        # K = P H^T S^-1; P H^T is just the first two columns of P.
        k = [
            [p[i][0] * inv00 + p[i][1] * inv10, p[i][0] * inv01 + p[i][1] * inv11]
            for i in range(4)
        ]

        innov_x = zx - self._state[0]
        innov_y = zy - self._state[1]
        self._state = [
            self._state[i] + k[i][0] * innov_x + k[i][1] * innov_y for i in range(4)
        ]

        # Joseph form.
        kh = [[k[i][0], k[i][1], 0.0, 0.0] for i in range(4)]
        i_kh = _mat_sub(_identity(4), kh)
        term1 = _mat_mul(_mat_mul(i_kh, p), _mat_t(i_kh))
        krk = [[r * (k[i][0] * k[j][0] + k[i][1] * k[j][1]) for j in range(4)] for i in range(4)]
        self._cov = _symmetrize(_mat_add(term1, krk))

    def _initialise(self, zx: float, zy: float, r: float) -> None:
        """Start a track from one measurement.

        Velocity is seeded at zero with a wide prior rather than guessed.
        `init_speed_std_ms` of 1.0 m/s covers the 0.3-1.2 m/s walking range
        in either direction on each axis, so the second measurement
        effectively determines the velocity and the filter is usable from
        the third onwards.
        """
        self._state = [zx, zy, 0.0, 0.0]
        v2 = self.init_speed_std_ms * self.init_speed_std_ms
        self._cov = [
            [r, 0.0, 0.0, 0.0],
            [0.0, r, 0.0, 0.0],
            [0.0, 0.0, v2, 0.0],
            [0.0, 0.0, 0.0, v2],
        ]

    # -- public API ----------------------------------------------------

    def confidence_from_z(self, z: float) -> float:
        """Map a robust z-score to a confidence in [0, 1].

        `1 - threshold/z`: zero exactly at the threshold, 0.5 at twice it,
        0.9 at ten times. Deliberately parameter-free given the threshold —
        there is no second constant to tune, and the scale inherits whatever
        calibration the threshold has.

        A non-finite z scores zero. `min(1.0, nan)` is `1.0` in CPython, so
        without the explicit test a NaN z — reachable from any upstream
        stage that diverges — would come back as maximal confidence.

        This is a statement about the *image*, not about the filter: it says
        how far the peak stood clear of the noise floor in this one frame.
        """
        if not math.isfinite(z) or z <= 0.0 or z < self.z_threshold:
            return 0.0
        return max(0.0, min(1.0, 1.0 - self.z_threshold / z))

    def update(self, image: list[float], t: float) -> PositionEstimate | None:
        """Feed one RTI image. Returns a fix, or `None` to abstain.

        `image` is a flat list in `geometry.grid.pixels` order. `t` is
        seconds, monotonically non-decreasing; a step backwards is treated
        as zero elapsed time rather than running the filter in reverse,
        which would shrink the covariance, and it does not advance the
        internal clock either, so one rewound frame cannot double-count the
        elapsed time on the next step.

        A non-finite `t` raises `ValueError`. It would otherwise poison the
        clock permanently: `nan > max_coast_s` is False, so the coast expiry
        — the one mechanism that stops noise sustaining a track — would be
        silently disabled from then on.
        """
        if not math.isfinite(t):
            raise ValueError(f"t must be finite, got {t}")

        reading = extract_peak(self.geometry, image, self.centroid_radius_m)

        if self._t is None:
            self._t = t
            self._last_fix_t = t
        assert self._last_fix_t is not None

        # Age the state to `t` first, so that both the coast-expiry test and
        # any correction happen at a consistent time.
        dt = t - self._t
        if self._state is not None:
            if dt > 0.0:
                self._predict(dt)
            # Expiry measures how far this frame sits from the last ACCEPTED
            # fix. The current frame's own interval is not a gap, however
            # long it is — see the class docstring — so the budget is at
            # least `dt`. The distance is taken as an absolute value so that
            # a clock which jumps BACKWARDS past the budget expires the
            # track too: a fix from 4 s in the caller's future is exactly as
            # stale as one from 4 s in its past, and without the `abs` a
            # sustained rewind disables the expiry entirely. Written as
            # `not (... <= ...)` so a NaN that somehow reached here resets
            # rather than coasting forever.
            if not (abs(t - self._last_fix_t) <= max(self.max_coast_s, dt)):
                self.reset()
        if t > self._t:
            self._t = t

        # NaN fails CLOSED: `nan < threshold` is False, which would ACCEPT
        # the frame, so the test is written as the positive condition.
        # `extract_peak` already forces z to 0.0 for any non-finite result
        # and the constructor rejects a non-finite threshold, so no input
        # reaches this line with a NaN today — mutating it back to
        # `reading.z < self.z_threshold` leaves the suite green. It stays
        # because the cost is nothing and the failure it guards is the one
        # this module is least able to afford.
        if not (reading.z >= self.z_threshold):
            return None
        if not (math.isfinite(reading.x) and math.isfinite(reading.y)):
            # `_correct` already refuses to propagate a non-finite det for
            # exactly this reason; the same instinct belongs on the
            # measurement path, where a NaN would otherwise be blended into
            # the state and never leave it.
            return None

        confidence = self.confidence_from_z(reading.z)

        # Inflate the measurement noise for a weak peak. Heuristic: a
        # zero-confidence (threshold-grazing) detection is trusted half as
        # much as a saturated one. Monotonic in the right direction,
        # uncalibrated in magnitude.
        sigma = self.meas_noise_m * (2.0 - confidence)
        r = sigma * sigma

        if self._state is None:
            self._initialise(reading.x, reading.y, r)
        else:
            self._correct(reading.x, reading.y, r)
        self._last_fix_t = self._t

        assert self._state is not None
        x, y = self._state[0], self._state[1]
        if self.clamp_to_room:
            # The motion model knows nothing about walls, so a correction
            # from a prior that drifted outside can land outside too: the
            # corrected position is a convex blend of prior and measurement,
            # and with R large beside P the blend stays near an outside
            # prior. Clamp the REPORTED position only; clamping the state
            # would make the filter nonlinear and quietly break the
            # covariance's meaning.
            room = self.geometry.room
            x = min(max(x, 0.0), room.width_m)
            y = min(max(y, 0.0), room.height_m)

        sigma_m = self.sigma_m
        assert sigma_m is not None and self._cov is not None
        cov = self._cov
        return PositionEstimate(
            x=x, y=y, sigma_m=sigma_m, confidence=confidence, t=t,
            cov_xx=cov[0][0], cov_xy=cov[0][1], cov_yy=cov[1][1],
        )
