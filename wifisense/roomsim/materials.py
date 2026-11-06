"""Materials (spec/roomsim.md §5): ITU-R P.2040 slabs and object constants.

Everything here is SIM: the ITU-R P.2040-3 Table 3 constants are
published, but treating a wall as one homogeneous slab, and the object
`rho`/`L_db` values, are modelling assumptions for Phase 10 to calibrate.
"""

from __future__ import annotations

from .numerics import (
    EPS0, TWO_PI, cadd, cdiv, cexp, cmul, cscale, csqrt, csub,
)

__all__ = [
    "ITU", "OBJECT_RHO", "OBJECT_LOSS_DB", "OBJECT_ITU",
    "eta", "fresnel", "slab", "Slab",
]

# §5.1: scene material -> (a, b, c, d) of ITU-R P.2040-3 Table 3.
ITU = {
    "concrete": (5.24, 0.0, 0.0462, 0.7822),
    "brick": (3.91, 0.0, 0.0238, 0.16),
    "drywall": (2.73, 0.0, 0.0085, 0.9395),
    "glass": (6.31, 0.0, 0.0036, 1.3394),
    "wood": (1.99, 0.0, 0.0047, 1.0718),
    "metal": (1.0, 0.0, 1e7, 0.0),
}

# §5.2: object material -> ITU material (fabric has none: it never
# reflects specularly), scatter rho, blockage loss per crossing (dB).
OBJECT_ITU = {"metal": "metal", "wood": "wood", "glass": "glass", "fabric": None}
OBJECT_RHO = {"metal": 1.0, "wood": 0.3, "glass": 0.2, "fabric": 0.1}
OBJECT_LOSS_DB = {"metal": 20.0, "wood": 4.0, "glass": 2.0, "fabric": 3.0}


def eta(material: str, carrier_hz: float):
    """Complex relative permittivity
    `(er, -(sigma / ((TWO_PI * carrier_hz) * EPS0)))`, with
    `er = a * pow(f_ghz, b)`, `sigma = c * pow(f_ghz, d)`."""
    a, b, c, d = ITU[material]
    f_ghz = carrier_hz / 1e9
    er = a * (f_ghz ** b)
    sigma = c * (f_ghz ** d)
    return (er, -(sigma / ((TWO_PI * carrier_hz) * EPS0)))


def fresnel(et, cos_t: float):
    """Single-interface coefficients `(s, G_TE, G_TM)` of §5.1."""
    sin2 = 1.0 - cos_t * cos_t
    s = csqrt(csub(et, (sin2, 0.0)))
    ct = (cos_t, 0.0)
    g_te = cdiv(csub(ct, s), cadd(ct, s))
    ec = cscale(et, cos_t)
    g_tm = cdiv(csub(ec, s), cadd(ec, s))
    return s, g_te, g_tm


def slab(G, s, cos_t: float, t: float, lam: float):
    """§5.1 slab of thickness `t`: returns `(R, T, q0)`.

    `T` includes `e^{+j q0}` (every path length counts the slab as free
    space); `R` refers to the incident face (§5.1).
    """
    k0t = TWO_PI * t / lam
    q = cscale(s, k0t)
    q0 = k0t * cos_t
    E2 = cexp(cmul((0.0, -2.0), q))
    GG = cmul(G, G)
    den = csub((1.0, 0.0), cmul(GG, E2))
    R = cdiv(cmul(G, csub((1.0, 0.0), E2)), den)
    T = cdiv(cmul(csub((1.0, 0.0), GG), cexp(cmul((0.0, -1.0), csub(q, (q0, 0.0))))), den)
    return R, T, q0


class Slab:
    """A wall, floor or ceiling as an ITU-R P.2040 slab (§5.1).

    `pol` is "TE" (walls) or "TM" (floor, ceiling). `R` is referenced to
    the incident face, where §6.1 places every bounce, so walls, floor and
    ceiling all reflect with it unchanged; `T` carries `e^{+j q0}` because
    a crossing is tested on the centreline and every path length counts the
    slab as free space.
    """

    __slots__ = ("material", "thickness", "pol", "eta", "lam")

    def __init__(self, material: str, thickness: float, pol: str, carrier_hz: float, lam: float) -> None:
        self.material = material
        self.thickness = thickness
        self.pol = pol
        self.eta = eta(material, carrier_hz)
        self.lam = lam

    def coefficients(self, cos_t: float):
        """`(G, R, T)` at `cos_t`."""
        s, g_te, g_tm = fresnel(self.eta, cos_t)
        G = g_te if self.pol == "TE" else g_tm
        R, T, _q0 = slab(G, s, cos_t, self.thickness, self.lam)
        return G, R, T

    def reflect(self, cos_t: float):
        return self.coefficients(cos_t)[1]

    def transmit(self, cos_t: float):
        return self.coefficients(cos_t)[2]
