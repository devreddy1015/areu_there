"""Weighted-link-midpoint baseline (spec/locate.md §5): a position with a radius, or an abstention.

The baseline ADR-008 decision 9 asks every other person estimator to be
published beside: each disturbed link votes for its midpoint, weighted by
how disturbed it is. Two things make it more than `evaluate_localize.py`'s
version, which is left unchanged:

  * a **significance gate** (`wlm_gate_z`, `wlm_floor`): with no link over
    it the baseline abstains `no_disturbance` instead of falling back to
    the room centre;
  * a **covariance** and so a radius: a mixture in which each link
    contributes a uniform density along its segment (`a = d*d/12`) and
    across its ellipse (`c = b*b/3`, `b` the ellipse's semi-minor axis for
    the excess path `lambda_m`), plus the spread of the midpoints about the
    estimate. A non-finite position, or a radius that is non-finite, not
    above 0 or below `min_plausible_radius_m`, is never published
    (`implausible_radius`), whatever minimum the caller passes.

New code (spec/locate.md §1 B): every Σ is an explicit Neumaier sum in link
order, never the builtin `sum`; every formula is evaluated as written.
`Link.midpoint` and `Link.length_m` are ported code (CPython `math.hypot`).
"""

from __future__ import annotations

import math

from ..roomsim.numerics import Neumaier
from .geometry import Geometry
from .radius import abstention, estimate_record, publishable_estimate, radius95

__all__ = ["weighted_link_midpoint", "wlm_covariance_terms", "default_wlm_floor"]


def default_wlm_floor(signal_name: str) -> float:
    """§5/§9: 0.25 dB² for `rssi_dbm`, 0.0 for any other signal."""
    return 0.25 if signal_name == "rssi_dbm" else 0.0


def wlm_covariance_terms(link, lambda_m: float) -> tuple[float, float, float, float]:
    """`(a_l, c_l, ux, uy)` of one link (§5 step 4).

    For `d_l > 0`: `ux = (rx.x - tx.x) / d_l`, `uy = (rx.y - tx.y) / d_l`,
    `a_l = (d_l*d_l) / 12.0`, `b_l = 0.5 * sqrt((d_l + λ)*(d_l + λ) - d_l*d_l)`,
    `c_l = (b_l*b_l) / 3.0`; otherwise all four are 0.0.
    """
    d_l = link.length_m
    if not (d_l > 0.0):
        return (0.0, 0.0, 0.0, 0.0)
    ux = (link.rx[0] - link.tx[0]) / d_l
    uy = (link.rx[1] - link.tx[1]) / d_l
    a_l = (d_l * d_l) / 12.0
    dl = d_l + lambda_m
    b_l = 0.5 * math.sqrt(dl * dl - d_l * d_l)
    c_l = (b_l * b_l) / 3.0
    return (a_l, c_l, ux, uy)


def weighted_link_midpoint(
    geometry: Geometry,
    y: list[float],
    null_sigma: list[float],
    signal_name: str,
    wlm_gate_z: float = 3.0,
    wlm_floor: float | None = None,
    wlm_radius_scale: float = 1.0,
    min_plausible_radius_m: float = 0.3,
) -> dict:
    """§5. Returns a §11 Estimate dict (`confidence` None) or an Abstention.

    `y` and `null_sigma` are in canonical link order (length L);
    `wlm_floor` None means the signal's default (`default_wlm_floor`).
    """
    links = geometry.links
    n = len(links)
    if len(y) != n:
        raise ValueError(f"y has {len(y)} entries, geometry has {n} links")
    if len(null_sigma) != n:
        raise ValueError(f"null_sigma has {len(null_sigma)} entries, geometry has {n} links")
    floor = default_wlm_floor(signal_name) if wlm_floor is None else wlm_floor

    # 1. Gate (a NaN anywhere fails it).
    disturbed = False
    for l in range(n):
        s = null_sigma[l]
        if s > 0.0 and y[l] > wlm_gate_z * s + floor:
            disturbed = True
            break
    if not disturbed:
        return abstention("no_disturbance")

    # 2. Weights: Python's two-argument max(y_l, 0.0), every link.
    w = [0.0 if 0.0 > v else v for v in y]
    acc = Neumaier()
    for wl in w:
        acc.add(wl)
    W = acc.value()
    if not (W > 0.0):
        return abstention("no_disturbance")

    # 3. Estimate.
    mids = [lk.midpoint for lk in links]
    sx = Neumaier()
    sy = Neumaier()
    for l in range(n):
        sx.add(w[l] * mids[l][0])
        sy.add(w[l] * mids[l][1])
    xh = sx.value() / W
    yh = sy.value() / W

    # 4. Covariance: the per-link uniform mixture plus the midpoints' spread.
    lam = geometry.lambda_m
    axx = Neumaier()
    axy = Neumaier()
    ayy = Neumaier()
    for l in range(n):
        a_l, c_l, ux, uy = wlm_covariance_terms(links[l], lam)
        ex = mids[l][0] - xh
        ey = mids[l][1] - yh
        wl = w[l]
        axx.add(wl * ((((a_l * ux) * ux) + ((c_l * uy) * uy)) + ex * ex))
        axy.add(wl * ((((a_l * ux) * uy) - ((c_l * uy) * ux)) + ex * ey))
        ayy.add(wl * ((((a_l * uy) * uy) + ((c_l * ux) * ux)) + ey * ey))
    s_xx = axx.value() / W
    s_xy = axy.value() / W
    s_yy = ayy.value() / W

    # 5. Radius.
    r = wlm_radius_scale * radius95(s_xx, s_xy, s_yy)
    if not publishable_estimate(xh, yh, r, min_plausible_radius_m):
        return abstention("implausible_radius")
    return estimate_record(xh, yh, r, s_xx, s_xy, s_yy, None)
