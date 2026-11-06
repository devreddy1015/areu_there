"""Where people are (spec/roomsim.md §9): walkability, starts, trajectories.

A walking person's trajectory is a list of contiguous walk and pause
segments drawn lazily from stream `200 + i`; it depends only on the draws,
never on the times it is evaluated at. Everything here is SIM.
"""

from __future__ import annotations

import math

from .geometry import dist2, inside, inside_with_margin, seg_seg_dist

__all__ = ["walkable_point", "walkable_leg", "door_start", "random_start", "Segment", "Trajectory"]


def _grow(cfg) -> float:
    return cfg.body_radius_m + cfg.object_clearance_m


def walkable_point(world, cfg, p) -> bool:
    """§9: inside with `walk_margin_m` and outside every non-opening
    object's footprint grown by `rg = body_radius_m + object_clearance_m`."""
    if not inside_with_margin(world.F, p, cfg.walk_margin_m):
        return False
    rg = _grow(cfg)
    for o in world.objects:
        if o.is_opening:
            continue
        b = o.box
        lx, ly, _ = b.to_local(p)
        if not (abs(lx) > b.sl / 2.0 + rg or abs(ly) > b.sb / 2.0 + rg):
            return False
    return True


def walkable_leg(world, cfg, a, b) -> bool:
    """§9: both ends walkable, at least `walk_margin_m` from every
    footprint edge (§7.3), and missing every grown object rectangle (§7.4,
    2-D)."""
    if not (walkable_point(world, cfg, a) and walkable_point(world, cfg, b)):
        return False
    F = world.F
    n = len(F)
    for i in range(n):
        if not (seg_seg_dist(a, b, F[i], F[(i + 1) % n]) >= cfg.walk_margin_m):
            return False
    rg = _grow(cfg)
    for o in world.objects:
        if o.is_opening:
            continue
        if o.box.leg_meets_2d(a, b, rg):
            return False
    return True


def door_start(world, cfg):
    """§9: the entry point at the first door, or None."""
    door = None
    for o in world.objects:
        if o.kind == "door":
            door = o
            break
    if door is None:
        return None
    box = door.box
    along_x = box.sl < box.sb  # the normal axis is the thinner horizontal extent
    direction = None
    for sgn in (1.0, -1.0):
        d = 0.5 * sgn
        p = box.to_world(d, 0.0, 0.0) if along_x else box.to_world(0.0, d, 0.0)
        if inside(world.F, (p[0], p[1])):
            direction = sgn
            break
    if direction is None:
        return None
    for n in range(1, 41):
        d = (0.05 * float(n)) * direction
        p = box.to_world(d, 0.0, 0.0) if along_x else box.to_world(0.0, d, 0.0)
        q = (p[0], p[1])
        if walkable_point(world, cfg, q):
            return q
    return None


def _draw_xy(rng, world, cfg):
    m = cfg.walk_margin_m
    x = rng.range(world.x0 + m, world.x1 - m)
    y = rng.range(world.y0 + m, world.y1 - m)
    return (x, y)


def random_start(rng, world, cfg):
    """§9: up to 1000 tries; the first walkable point, or None."""
    for _ in range(1000):
        p = _draw_xy(rng, world, cfg)
        if walkable_point(world, cfg, p):
            return p
    return None


class Segment:
    """A walk (`frm -> to` at speed `v`) or a pause (at `frm`)."""

    __slots__ = ("walk", "t0", "t1", "frm", "to", "v", "heading")

    def __init__(self, walk, t0, t1, frm, to, v, heading):
        self.walk = walk
        self.t0 = t0
        self.t1 = t1
        self.frm = frm
        self.to = to
        self.v = v
        self.heading = heading


class Trajectory:
    """§9 trajectory of one walking person, extended lazily from `rng`
    (stream `200 + i`, after any start draws)."""

    def __init__(self, start, rng, world, cfg) -> None:
        self.start = (float(start[0]), float(start[1]))
        self.rng = rng
        self.world = world
        self.cfg = cfg
        self.segs: list[Segment] = []
        self._cursor = 0
        self._last_heading = 0.0

    def _end(self):
        if not self.segs:
            return self.start
        s = self.segs[-1]
        return s.to

    def extend(self) -> None:
        """Append one walk (and its pause), or one pause if no try is
        accepted (§9)."""
        cfg = self.cfg
        rng = self.rng
        t0 = self.segs[-1].t1 if self.segs else 0.0
        cur = self._end()
        for _ in range(64):
            p = _draw_xy(rng, self.world, cfg)
            if not walkable_point(self.world, cfg, p):
                continue
            d = dist2(p, cur)
            if not (d >= 0.5):
                continue
            if not walkable_leg(self.world, cfg, cur, p):
                continue
            v = rng.range(cfg.speed_mps[0], cfg.speed_mps[1])
            pause = rng.range(cfg.pause_s[0], cfg.pause_s[1])
            heading = math.atan2(p[1] - cur[1], p[0] - cur[0])
            walk = Segment(True, t0, t0 + d / v, cur, p, v, heading)
            self.segs.append(walk)
            self._last_heading = heading
            pt1 = walk.t1 + pause
            if pt1 > walk.t1:
                self.segs.append(Segment(False, walk.t1, pt1, p, p, 0.0, heading))
            return
        pause = rng.range(cfg.pause_s[0], cfg.pause_s[1])
        self.segs.append(Segment(False, t0, t0 + max(pause, 0.5), cur, cur, 0.0, self._last_heading))

    def at(self, t: float):
        """`(x, y, heading, s, v)` at time `t`. Times must be nondecreasing
        across calls (packet order)."""
        while not self.segs or self.segs[-1].t1 <= t:
            self.extend()
        segs = self.segs
        i = self._cursor
        if t < segs[i].t0:
            i = 0
        while not (segs[i].t0 <= t < segs[i].t1):
            i += 1
        self._cursor = i
        s = segs[i]
        if s.walk:
            f = (t - s.t0) / (s.t1 - s.t0)
            x = s.frm[0] + (s.to[0] - s.frm[0]) * f
            y = s.frm[1] + (s.to[1] - s.frm[1]) * f
            return (x, y, s.heading, 1, s.v)
        return (s.frm[0], s.frm[1], s.heading, 0, 0.0)
