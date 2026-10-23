"""The simulated room: a scene resolved against a configuration (§3.1, §4, §6.1).

`World` holds what every link shares: the footprint `F`, height `H`, the
footprint's bounding box, the walls (the scene's, or the footprint's edges
as implied walls, §3.1), the reflecting surfaces of §6.1, the objects with
their §3.1 frames, the node positions and the §4 frequency grid.
"""

from __future__ import annotations

import math

from .config import ConfigError
from .geometry import inside, pdist
from .materials import Slab
from .numerics import carrier_hz as _carrier_hz
from .numerics import sign, wavelength
from .scene import NODE_WALL_CLEARANCE_M

__all__ = ["World", "WallSurface", "FloorSurface", "CeilingSurface", "ObjectGeom", "BINS",
           "NODE_WALL_CLEARANCE_M"]

# §4: bins k in [-32, 31], ascending.
BINS = tuple(range(-32, 32))

# §3.1: NODE_WALL_CLEARANCE_M (saans-scene's 1 cm) is measured here from a
# wall's face (t/2 from its centreline), not from the centreline.


class WallSurface:
    """A wall `A -> B` of height `h`, thickness `t` (§3.1, §6.1).

    A bounce is taken on a face of the wall, chosen by a side
    `f in {+1, -1}`: the face line is the centreline moved by
    `o = f * (t / 2.0)` along `n`. Transmission is tested on the
    centreline (`hit` with `f = 0`, i.e. `o = 0`).
    """

    kind = "wall"
    __slots__ = ("index", "wall_index", "A", "B", "h", "t", "material", "implied",
                 "len", "e", "n", "slab")

    def __init__(self, index, wall_index, A, B, h, t, material, implied, carrier, lam):
        self.index = index
        self.wall_index = wall_index
        self.A = (float(A[0]), float(A[1]))
        self.B = (float(B[0]), float(B[1]))
        self.h = float(h)
        self.t = float(t)
        self.material = material
        self.implied = implied
        dx = self.B[0] - self.A[0]
        dy = self.B[1] - self.A[1]
        self.len = math.sqrt(dx * dx + dy * dy)
        self.e = (dx / self.len, dy / self.len)
        self.n = (-self.e[1], self.e[0], 0.0)
        self.slab = Slab(material, self.t, "TE", carrier, lam)

    def dc(self, p) -> float:
        """Signed distance from the centreline: `(P.x-A.x)*n.x + (P.y-A.y)*n.y`."""
        return (p[0] - self.A[0]) * self.n[0] + (p[1] - self.A[1]) * self.n[1]

    def offset(self, f: int) -> float:
        """`o = f * (t / 2.0)`: the face `f` (0 is the centreline)."""
        return float(f) * (self.t / 2.0)

    def side(self, p) -> int:
        return sign(self.dc(p))

    def front(self, f: int, p) -> bool:
        """`front_f(P)`: `sign(dc(P) - o) == f`."""
        return sign(self.dc(p) - self.offset(f)) == f

    def image(self, f: int, p):
        """`image_f(P)`: the mirror image of `p` in face `f`."""
        dd = 2.0 * (self.dc(p) - self.offset(f))
        return (p[0] - dd * self.n[0], p[1] - dd * self.n[1], p[2])

    def hit(self, p, q, f: int = 0):
        """`hit_f(P, Q)`: where segment `p -> q` meets face `f`, or None
        (§6.1); `f = 0` is the centreline (transmission, §6.9)."""
        o = self.offset(f)
        afx = self.A[0] + o * self.n[0]
        afy = self.A[1] + o * self.n[1]
        rx = q[0] - p[0]
        ry = q[1] - p[1]
        wx = self.B[0] - self.A[0]
        wy = self.B[1] - self.A[1]
        den = rx * wy - ry * wx
        if den == 0.0:
            return None
        apx = afx - p[0]
        apy = afy - p[1]
        s = (apx * wy - apy * wx) / den
        v = (apx * ry - apy * rx) / den
        if not (0.0 < s < 1.0 and 0.0 <= v <= 1.0):
            return None
        x = (p[0] + s * rx, p[1] + s * ry, p[2] + s * (q[2] - p[2]))
        if not (0.0 <= x[2] <= self.h):
            return None
        return x


class _HorizontalSurface:
    """Floor or ceiling: its plane is its face, so `f` (the side a bounce
    is taken on) changes nothing but `front`."""

    __slots__ = ("index", "z", "F", "material", "t", "slab")
    n = (0.0, 0.0, 1.0)

    def __init__(self, index, z, F, material, t, carrier, lam):
        self.index = index
        self.z = z
        self.F = F
        self.material = material
        self.t = float(t)
        self.slab = Slab(material, self.t, "TM", carrier, lam)

    def front(self, f: int, p) -> bool:
        """`front_f(P)`: `side(P) == f`."""
        return self.side(p) == f

    def _hit_z(self, p, q, z):
        dz = q[2] - p[2]
        if dz == 0.0:
            return None
        s = (z - p[2]) / dz
        if not (0.0 < s < 1.0):
            return None
        x = (p[0] + s * (q[0] - p[0]), p[1] + s * (q[1] - p[1]), z)
        if not inside(self.F, (x[0], x[1])):
            return None
        return x


class FloorSurface(_HorizontalSurface):
    """The floor, `z = 0` (§6.1)."""

    kind = "floor"
    __slots__ = ()

    def image(self, f: int, p):
        return (p[0], p[1], -p[2])

    def side(self, p) -> int:
        return sign(p[2])

    def hit(self, p, q, f: int = 0):
        return self._hit_z(p, q, 0.0)


class CeilingSurface(_HorizontalSurface):
    """The ceiling, `z = H` (§6.1)."""

    kind = "ceiling"
    __slots__ = ()

    def image(self, f: int, p):
        return (p[0], p[1], 2.0 * self.z - p[2])

    def side(self, p) -> int:
        return sign(self.z - p[2])

    def hit(self, p, q, f: int = 0):
        return self._hit_z(p, q, self.z)


class ObjectGeom:
    """A scene object with its §3.1 frame."""

    __slots__ = ("index", "kind", "material", "box", "is_opening")

    def __init__(self, index, obj):
        self.index = index
        self.kind = obj.kind
        self.material = obj.material
        self.box = obj.box()
        self.is_opening = obj.kind in ("door", "window")


class World:
    """A scene resolved against a `Config`: everything links share."""

    def __init__(self, scene, cfg) -> None:
        self.scene = scene
        self.F = [(float(x), float(y)) for x, y in scene.footprint()]
        self.H = float(scene.room.h_m)
        self.x0 = min(p[0] for p in self.F)
        self.x1 = max(p[0] for p in self.F)
        self.y0 = min(p[1] for p in self.F)
        self.y1 = max(p[1] for p in self.F)
        self.carrier_hz = _carrier_hz(cfg.channel)
        self.lam = wavelength(self.carrier_hz)
        self.freqs = [self.carrier_hz + float(k) * cfg.spacing_hz for k in BINS]

        self.implied_walls = len(scene.walls) == 0
        walls = []
        if scene.walls:
            for i, w in enumerate(scene.walls):
                h = self.H if w.height_m is None else w.height_m
                walls.append(WallSurface(i, i, w.a, w.b, h, w.thickness_m, w.material, False,
                                         self.carrier_hz, self.lam))
        else:
            n = len(self.F)
            for i in range(n):
                walls.append(WallSurface(i, i, self.F[i], self.F[(i + 1) % n], self.H,
                                         cfg.implied_wall_thickness_m, cfg.implied_wall_material, True,
                                         self.carrier_hz, self.lam))
        self.walls = walls
        surfaces = list(walls)
        if cfg.floor_ceiling:
            surfaces.append(FloorSurface(len(surfaces), 0.0, self.F, cfg.floor_material,
                                         cfg.slab_thickness_m, self.carrier_hz, self.lam))
            surfaces.append(CeilingSurface(len(surfaces), self.H, self.F, cfg.ceiling_material,
                                           cfg.slab_thickness_m, self.carrier_hz, self.lam))
        self.surfaces = surfaces
        self.objects = [ObjectGeom(i, o) for i, o in enumerate(scene.objects)]
        self.node_ids = [n.id for n in scene.nodes]
        self.node_pos = {n.id: (float(n.pos[0]), float(n.pos[1]), float(n.pos[2])) for n in scene.nodes}
        self._check_nodes_clear_of_walls()

    def _check_nodes_clear_of_walls(self) -> None:
        """§3.1: a node within a wall's thickness,
        `pdist(node.xy, A, B) < t/2 + 0.01` with `node.z <= h`, for any wall
        (implied walls included), is rejected naming `nodes`."""
        for nid in self.node_ids:
            p = self.node_pos[nid]
            for w in self.walls:
                d = pdist((p[0], p[1]), w.A, w.B)
                lim = w.t / 2.0 + NODE_WALL_CLEARANCE_M
                if p[2] <= w.h and d < lim:
                    which = (f"implied wall {w.wall_index} (footprint edge {w.wall_index})" if w.implied
                             else f"walls[{w.wall_index}]")
                    raise ConfigError(
                        "nodes",
                        f"node {nid} at ({p[0]:.3f}, {p[1]:.3f}, {p[2]:.3f}) m is within {which}'s thickness: "
                        f"{d:.3f} m from its centreline, less than t/2 + {NODE_WALL_CLEARANCE_M} = {lim:.3f} m "
                        f"(a radio cannot be inside a wall)",
                    )
