"""Geometry primitives of spec/roomsim.md §7 (and the §3.1 frames).

Points are tuples: `(x, y)` on the floor plane, `(x, y, z)` in the room.
Every expression is written in the spec's evaluation order; lengths are
`sqrt(dx*dx + dy*dy)` (never `hypot`).
"""

from __future__ import annotations

import math

from .numerics import clamp

__all__ = [
    "dist3", "dist2", "dot2", "dot3", "unit3",
    "pdist", "inside", "inside_with_margin",
    "seg_cross_2d", "seg_seg_dist",
    "Box", "INSIDE_EDGE_TOL",
]

# §7.2: a point within this distance of an edge is inside.
INSIDE_EDGE_TOL = 1e-9


# -- §7.1 -------------------------------------------------------------------


def dist3(p, q) -> float:
    """`|P - Q| = sqrt((dx*dx + dy*dy) + dz*dz)`."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    dz = p[2] - q[2]
    return math.sqrt((dx * dx + dy * dy) + dz * dz)


def dist2(p, q) -> float:
    """xy length `sqrt(dx*dx + dy*dy)`."""
    dx = p[0] - q[0]
    dy = p[1] - q[1]
    return math.sqrt(dx * dx + dy * dy)


def dot2(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1]


def dot3(a, b) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def unit3(p, q):
    """Unit direction of the leg `p -> q`: `(dx/D, dy/D, dz/D)`."""
    dx = q[0] - p[0]
    dy = q[1] - p[1]
    dz = q[2] - p[2]
    d = math.sqrt((dx * dx + dy * dy) + dz * dz)
    return (dx / d, dy / d, dz / d)


# -- §7.2 -------------------------------------------------------------------


def pdist(p, a, b) -> float:
    """Distance from `p` to the segment `a`-`b` (§7.2)."""
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dd = dx * dx + dy * dy
    if dd == 0.0:
        return dist2(p, a)
    t = clamp(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / dd, 0.0, 1.0)
    q = (a[0] + t * dx, a[1] + t * dy)
    return dist2(p, q)


def inside(poly, p) -> bool:
    """§7.2 `inside(F, p)`: on an edge (within 1e-9) or by crossing number."""
    n = len(poly)
    for i in range(n):
        if pdist(p, poly[i], poly[(i + 1) % n]) <= INSIDE_EDGE_TOL:
            return True
    res = False
    px, py = p[0], p[1]
    for i in range(n):
        a = poly[i]
        b = poly[(i + 1) % n]
        if (a[1] > py) != (b[1] > py):
            if px < a[0] + (py - a[1]) * (b[0] - a[0]) / (b[1] - a[1]):
                res = not res
    return res


def inside_with_margin(poly, p, m: float) -> bool:
    """§7.2: `inside` and at least `m` from every edge."""
    if not inside(poly, p):
        return False
    n = len(poly)
    for i in range(n):
        if not (pdist(p, poly[i], poly[(i + 1) % n]) >= m):
            return False
    return True


# -- §7.3 -------------------------------------------------------------------


def seg_cross_2d(p, q, a, b) -> bool:
    """Do segments `p-q` and `a-b` cross: the §6.1 wall test with
    `0 <= s <= 1` and `0 <= v <= 1`."""
    rx = q[0] - p[0]
    ry = q[1] - p[1]
    wx = b[0] - a[0]
    wy = b[1] - a[1]
    den = rx * wy - ry * wx
    if den == 0.0:
        return False
    apx = a[0] - p[0]
    apy = a[1] - p[1]
    s = (apx * wy - apy * wx) / den
    v = (apx * ry - apy * rx) / den
    return 0.0 <= s <= 1.0 and 0.0 <= v <= 1.0


def seg_seg_dist(p, q, a, b) -> float:
    """§7.3 segment-segment distance: 0 if they cross, else the least of
    the four point-segment distances."""
    if seg_cross_2d(p, q, a, b):
        return 0.0
    return min(pdist(p, a, b), pdist(q, a, b), pdist(a, p, q), pdist(b, p, q))


# -- objects: §3.1 frames and §7.4 box tests ---------------------------------


class Box:
    """An upright box: centre `(cx, cy, cz)`, size `(sl, sb, sh)`, yaw `psi`.

    `psi = (yaw_deg * PI) / 180.0` exactly (§3.1).
    """

    __slots__ = ("cx", "cy", "cz", "sl", "sb", "sh", "psi", "cs", "sn")

    def __init__(self, pos, size, yaw_deg: float) -> None:
        self.cx, self.cy, self.cz = float(pos[0]), float(pos[1]), float(pos[2])
        self.sl, self.sb, self.sh = float(size[0]), float(size[1]), float(size[2])
        self.psi = (float(yaw_deg) * math.pi) / 180.0
        self.cs = math.cos(self.psi)
        self.sn = math.sin(self.psi)

    def to_world(self, lx: float, ly: float, lz: float):
        """Object-local to world (§3.1)."""
        cs, sn = self.cs, self.sn
        return (self.cx + (lx * cs - ly * sn), self.cy + (lx * sn + ly * cs), self.cz + lz)

    def rotate(self, lx: float, ly: float, lz: float):
        """Rotate a local direction into the world (§3.1 without the centre)."""
        cs, sn = self.cs, self.sn
        return ((lx * cs - ly * sn), (lx * sn + ly * cs), lz)

    def to_local(self, p):
        """World to object-local (§3.1). Accepts 2-D or 3-D points; a 2-D
        point gets `lz = 0.0`."""
        cs, sn = self.cs, self.sn
        dx = p[0] - self.cx
        dy = p[1] - self.cy
        lx = dx * cs + dy * sn
        ly = -dx * sn + dy * cs
        lz = (p[2] - self.cz) if len(p) > 2 else 0.0
        return (lx, ly, lz)

    def strictly_inside(self, p) -> bool:
        """§7.4 "strictly inside" (node validation)."""
        lx, ly, lz = self.to_local(p)
        return abs(lx) < self.sl / 2.0 and abs(ly) < self.sb / 2.0 and abs(lz) < self.sh / 2.0

    def footprint_contains(self, p) -> bool:
        """§8.2's "box footprint (not grown) contains the start":
        `|lx| <= sl/2` and `|ly| <= sb/2` (closed)."""
        lx, ly, _ = self.to_local(p)
        return abs(lx) <= self.sl / 2.0 and abs(ly) <= self.sb / 2.0

    def leg_meets(self, p, q) -> bool:
        """§7.4: does the 3-D leg `p -> q` meet the box?"""
        a = self.to_local(p)
        b = self.to_local(q)
        return _slab_meet(a, b, (self.sl / 2.0, self.sb / 2.0, self.sh / 2.0), 3)

    def leg_meets_2d(self, p, q, grow: float) -> bool:
        """§7.4, 2-D version (z dropped), against the footprint grown by
        `grow` on every side: half-extents `sl/2 + grow`, `sb/2 + grow`."""
        a = self.to_local(p)
        b = self.to_local(q)
        return _slab_meet(a, b, (self.sl / 2.0 + grow, self.sb / 2.0 + grow), 2)


def _slab_meet(a, b, half, naxes: int) -> bool:
    t_enter = 0.0
    t_exit = 1.0
    for ax in range(naxes):
        h = half[ax]
        p = a[ax]
        d = b[ax] - a[ax]
        if d == 0.0:
            if abs(p) > h:
                return False
            continue
        t1 = (-h - p) / d
        t2 = (h - p) / d
        lo = t1 if t1 < t2 else t2
        hi = t2 if t1 < t2 else t1
        t_enter = max(t_enter, lo)
        t_exit = min(t_exit, hi)
    return t_exit > t_enter
