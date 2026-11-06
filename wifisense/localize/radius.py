"""The published radius and the estimate/abstention records (spec/locate.md §8, §11).

Every position this package publishes is a point **with a radius**, or an
abstention with a reason; never a bare point (ADR-008 decision 5). The
radius is a *model* statement: a conservative 95% circle of a covariance,
times an uncalibrated `*_radius_scale`. Every estimate therefore says
`radius_calibrated: False` until Phase 5 measures coverage on held-out
runs.

New code (spec/locate.md §1 B): every formula is evaluated exactly as the
spec writes it, binary64, left to right, no `hypot`, and every comparison
that decides something fails closed on a NaN.
"""

from __future__ import annotations

import math

__all__ = [
    "K95",
    "CHI2_2_99",
    "LN10",
    "radius95",
    "publishable",
    "publishable_estimate",
    "estimate_record",
    "abstention",
    "REASONS",
]

# §1 constants, computed, never typed.
K95 = math.sqrt(-2.0 * math.log(0.05))      # 2.4477468306808166
CHI2_2_99 = -2.0 * math.log(0.01)           # 9.210340371976182
LN10 = math.log(10.0)

# Every abstention reason any estimator can give (§11).
REASONS = (
    "no_disturbance",
    "implausible_radius",
    "not_significant",
    "no_reports",
    "too_few_ranges",
    "degenerate",
    "not_converged",
    "gated",
)


def radius95(a: float, b: float, d: float) -> float:
    """§8: the 95% radius of the covariance `[[a, b], [b, d]]`.

    `h = 0.5*(a - d)`; `lmax = 0.5*(a + d) + sqrt(h*h + b*b)`;
    `mm = lmax if lmax > 0.0 else 0.0` (a NaN gives 0.0); `K95 * sqrt(mm)`.
    The circle holds at least 95% of a Gaussian with that covariance,
    whatever its shape (the largest eigenvalue's circle).
    """
    h = 0.5 * (a - d)
    # `h*h + b*b` is never negative; a NaN passes through `math.sqrt`.
    lmax = 0.5 * (a + d) + math.sqrt(h * h + b * b)
    mm = lmax if lmax > 0.0 else 0.0
    return K95 * math.sqrt(mm)


def publishable(r: float, min_plausible_radius_m: float) -> bool:
    """The radius half of §8's publication test: `isfinite(r) and r > 0.0
    and r >= min_plausible_radius_m` (a NaN fails it). `r > 0` holds even
    for a caller that passes a zero (or negative) minimum, so a bare point
    is never published (ADR-008 d5). An estimator publishes only through
    `publishable_estimate`, which also needs a finite position."""
    return math.isfinite(r) and r > 0.0 and r >= min_plausible_radius_m


def publishable_estimate(x: float, y: float, r: float, min_plausible_radius_m: float) -> bool:
    """§5 step 5, §7.4, §8: an estimate is published only if `x`, `y` and
    `r` are finite, `r > 0` and `r >= min_plausible_radius_m` (written so
    that a NaN fails); otherwise its estimator abstains
    `implausible_radius`."""
    return math.isfinite(x) and math.isfinite(y) and publishable(r, min_plausible_radius_m)


def estimate_record(
    x: float,
    y: float,
    radius_m: float,
    cov_xx: float,
    cov_xy: float,
    cov_yy: float,
    confidence: float | None = None,
) -> dict:
    """§11 **Estimate**: `{x, y, radius_m, cov_xx, cov_xy, cov_yy,
    confidence, radius_calibrated}`; `radius_calibrated` is False in
    Phase 4."""
    return {
        "x": x,
        "y": y,
        "radius_m": radius_m,
        "cov_xx": cov_xx,
        "cov_xy": cov_xy,
        "cov_yy": cov_yy,
        "confidence": confidence,
        "radius_calibrated": False,
    }


def abstention(reason: str) -> dict:
    """§11 **Abstention**: `{reason}`."""
    if reason not in REASONS:
        raise ValueError(f"reason: {reason!r} is not one of {', '.join(REASONS)}")
    return {"reason": reason}
