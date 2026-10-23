"""Constants and the exact arithmetic helpers of spec/roomsim.md §1.

Every formula in Room Sim is evaluated exactly as the spec writes it, in
IEEE-754 binary64, one rounding per operator, so the Python reference and
the Rust crate agree to ~1e-9. That is why complex numbers here are plain
`(re, im)` tuples with the explicit helpers below and never Python's
`complex`, `cmath`, `math.hypot`, `math.fsum` or the built-in `sum` (whose
float behaviour changed in CPython 3.12).
"""

from __future__ import annotations

import math

__all__ = [
    "PI", "TWO_PI", "FOUR_PI", "FOUR_PI_1_5", "C", "EPS0", "MIN_RANGE_M",
    "Cx", "cadd", "csub", "cmul", "cscale", "cdiv", "csqrt", "cexpj", "cexp", "cabs2",
    "Neumaier", "neumaier", "round_half_even", "clamp", "sign", "pow10",
    "theta", "carrier_hz", "wavelength",
]

Cx = tuple  # (re, im)

# §1 constants, computed as written.
PI = math.pi
TWO_PI = 2.0 * PI
FOUR_PI = 4.0 * PI
FOUR_PI_1_5 = FOUR_PI * math.sqrt(FOUR_PI)
C = 299792458.0
EPS0 = 8.8541878128e-12
MIN_RANGE_M = 0.1


# -- complex helpers (§1) ---------------------------------------------------


def cadd(a, b):
    return (a[0] + b[0], a[1] + b[1])


def csub(a, b):
    return (a[0] - b[0], a[1] - b[1])


def cmul(a, b):
    return (a[0] * b[0] - a[1] * b[1], a[0] * b[1] + a[1] * b[0])


def cscale(a, s):
    return (a[0] * s, a[1] * s)


def cdiv(a, b):
    d = b[0] * b[0] + b[1] * b[1]
    return ((a[0] * b[0] + a[1] * b[1]) / d, (a[1] * b[0] - a[0] * b[1]) / d)


def csqrt(z):
    """Principal square root, §1's branch-by-branch formula."""
    zr, zi = z
    if zr == 0.0 and zi == 0.0:
        return (0.0, 0.0)
    m = math.sqrt(zr * zr + zi * zi)
    if zr >= 0.0:
        t = math.sqrt((m + zr) / 2.0)
        return (t, zi / (2.0 * t))
    t = math.sqrt((m - zr) / 2.0)
    return (abs(zi) / (2.0 * t), math.copysign(t, zi))


def cexpj(th):
    return (math.cos(th), math.sin(th))


def cexp(z):
    return cscale(cexpj(z[1]), math.exp(z[0]))


def cabs2(a):
    """`re*re + im*im`."""
    return a[0] * a[0] + a[1] * a[1]


# -- sums and rounding ------------------------------------------------------


class Neumaier:
    """Neumaier-compensated sum (§1), used for exactly the sums §1 lists."""

    __slots__ = ("hi", "lo")

    def __init__(self) -> None:
        self.hi = 0.0
        self.lo = 0.0

    def add(self, x: float) -> None:
        hi = self.hi
        t = hi + x
        if abs(hi) >= abs(x):
            self.lo += (hi - t) + x
        else:
            self.lo += (x - t) + hi
        self.hi = t

    def value(self) -> float:
        lo = self.lo
        if lo != 0.0 and math.isfinite(lo):
            return self.hi + lo
        return self.hi


def neumaier(xs) -> float:
    """Neumaier sum of `xs`, in the given order."""
    acc = Neumaier()
    for x in xs:
        acc.add(x)
    return acc.value()


def round_half_even(x: float) -> int:
    """Round to the nearest integer, ties to even (Python `round`)."""
    return round(x)


def clamp(x, lo, hi):
    """`min(max(x, lo), hi)`."""
    return min(max(x, lo), hi)


def sign(d: float) -> int:
    """1 if d > 0, -1 if d < 0, else 0 (-0.0 is 0)."""
    if d > 0.0:
        return 1
    if d < 0.0:
        return -1
    return 0


def pow10(y: float) -> float:
    """`10^y` as `pow(10.0, y)`."""
    return 10.0 ** y


def carrier_hz(channel: int) -> float:
    """§3.2: `(2412 + 5*(channel-1)) * 1e6`."""
    return float(2412 + 5 * (channel - 1)) * 1e6


def wavelength(carrier: float) -> float:
    """`LAMBDA = C / carrier_hz`."""
    return C / carrier


def theta(f: float, L: float) -> float:
    """Phase of a path of length L at frequency f: `((-TWO_PI) * f) * L / C`."""
    return (((-TWO_PI) * f) * L) / C
