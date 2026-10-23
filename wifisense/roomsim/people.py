"""People (spec/roomsim.md §8): body parameters, mode and posture, body
points, and the §8.5 cylinder shadowing (three knife edges: two sides and
the top, ITU-R P.526's finite-width screen).

A body is 6..16 point scatterers and a vertical cylinder; gait is a
sinusoid and breathing moves the chest. All of it is SIM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .config import ConfigError
from .geometry import dist2, dist3
from .motion import Trajectory, door_start, random_start, walkable_point
from .numerics import PI, TWO_PI, Neumaier, clamp
from .rng import Rng

__all__ = [
    "BODY_TABLE", "POSTURES", "POSTURE_CODE", "Person", "build_people",
    "J1", "three_edge_J", "shadow_J", "shadow_factor",
]

# §8.3 body table: (x, y, z/height, weight, swing amp, swing phase, breath).
BODY_TABLE = (
    (0.12, 0.0, 0.72, 0.20, 0.0, 0.0, 1.0),     # chest
    (0.0, 0.0, 0.93, 0.08, 0.0, 0.0, 0.0),      # head
    (0.0, 0.0, 0.53, 0.15, 0.0, 0.0, 0.0),      # pelvis
    (-0.12, 0.0, 0.70, 0.15, 0.0, 0.0, 0.0),    # back
    (0.0, 0.22, 0.45, 0.04, 0.25, PI, 0.0),     # left hand
    (0.0, -0.22, 0.45, 0.04, 0.25, 0.0, 0.0),   # right hand
    (0.0, 0.10, 0.05, 0.04, 0.30, 0.0, 0.0),    # left foot
    (0.0, -0.10, 0.05, 0.04, 0.30, PI, 0.0),    # right foot
    (0.0, 0.20, 0.80, 0.05, 0.0, 0.0, 0.0),     # left shoulder
    (0.0, -0.20, 0.80, 0.05, 0.0, 0.0, 0.0),    # right shoulder
    (0.0, 0.10, 0.28, 0.04, 0.15, 0.0, 0.0),    # left knee
    (0.0, -0.10, 0.28, 0.04, 0.15, PI, 0.0),    # right knee
    (0.0, 0.22, 0.62, 0.03, 0.12, PI, 0.0),     # left elbow
    (0.0, -0.22, 0.62, 0.03, 0.12, 0.0, 0.0),   # right elbow
    (0.11, 0.0, 0.60, 0.08, 0.0, 0.0, 0.5),     # abdomen
    (0.0, 0.0, 0.85, 0.03, 0.0, 0.0, 0.0),      # neck
)

POSTURES = ("stand", "sit", "lie")
POSTURE_CODE = {"stand": 0, "sit": 1, "lie": 2}


@dataclass
class Person:
    """One simulated person, `people[index]`. No scene id is kept."""

    index: int
    height: float
    f_b: float
    p2p: float
    phi_b: float
    phi_g: float
    start: tuple
    walking: bool
    posture: str
    on_object: int            # first object whose footprint holds the start (stationary), or -1
    bed: object               # ObjectGeom of the bed for `lie`, else None
    heading0: float           # stationary heading (start_heading_rad)
    trajectory: Trajectory | None
    rcs: tuple                # per body point, body_rcs_m2 * w_j / W

    @property
    def top(self) -> float:
        """§8.5 cylinder top."""
        if self.posture == "lie":
            b = self.bed.box
            return (b.cz + b.sh / 2.0) + 0.25
        if self.posture == "sit":
            return 0.72 * self.height
        return self.height

    def state(self, t: float):
        """`(x, y, heading, s, v)` at `t` (§9)."""
        if self.trajectory is None:
            return (self.start[0], self.start[1], self.heading0, 0, 0.0)
        return self.trajectory.at(t)

    def body_points(self, t: float, state, n_points: int):
        """World positions of the first `n_points` body points at `t` (§8.3)."""
        px, py, hd, s, v = state
        sf = float(s)
        f_s = clamp(0.95 * math.sqrt(v / 1.1), 0.5, 1.2) if s == 1 else 0.0
        ch = math.cos(hd)
        sh = math.sin(hd)
        out = []
        lie = self.posture == "lie"
        if lie:
            box = self.bed.box
            if box.sl >= box.sb:
                ax, ay = box.cs, box.sn
            else:
                ax, ay = -box.sn, box.cs
            cx, cy = -ay, ax
            bed_top = (box.cz + box.sh / 2.0) + 0.12
        for j in range(n_points):
            x, y, zf, _w, amp, phase, breath = BODY_TABLE[j]
            sw = (amp * sf) * math.sin((TWO_PI * f_s) * t + self.phi_g + phase)
            br = (breath * (self.p2p / 2.0)) * math.sin((TWO_PI * self.f_b) * t + self.phi_b)
            bx = x + sw + br
            by = y
            bz = zf * self.height
            if lie:
                along = bz - 0.5 * self.height
                out.append((px + (along * ax + by * cx), py + (along * ay + by * cy), bed_top + bx))
                continue
            if self.posture == "sit":
                bz = 0.72 * bz
            out.append((px + (bx * ch - by * sh), py + (bx * sh + by * ch), bz))
        return out


def _first_containing_object(world, p):
    for o in world.objects:
        if o.box.footprint_contains(p):
            return o
    return None


def build_people(world, cfg, scene) -> list:
    """Body parameters (§8.1), starts (§9), mode and posture (§8.2) and
    trajectories for every simulated person. Raises ConfigError naming
    `people[i]` when no start can be found, or the override that is
    impossible."""
    starts = [p.start for p in scene.people]
    n = len(starts) if cfg.n_people is None else cfg.n_people
    if n <= len(starts):
        starts = starts[:n]
    else:
        starts = starts + [None] * (n - len(starts))
    weights = [BODY_TABLE[j][3] for j in range(cfg.body_points)]
    acc = Neumaier()
    for w in weights:
        acc.add(w)
    W = acc.value()
    rcs = tuple(cfg.body_rcs_m2 * w / W for w in weights)
    door = None
    out = []
    for i, st in enumerate(starts):
        name = f"people[{i}]"
        prng = Rng.stream(cfg.seed, 100 + i)
        height = prng.range(cfg.height_m[0], cfg.height_m[1])
        f_b = prng.range(cfg.breathing_hz[0], cfg.breathing_hz[1])
        p2p = prng.range(cfg.breathing_p2p_m[0], cfg.breathing_p2p_m[1])
        phi_b = prng.range(0.0, TWO_PI)
        phi_g = prng.range(0.0, TWO_PI)
        mrng = Rng.stream(cfg.seed, 200 + i)
        if st is not None:
            start = (float(st[0]), float(st[1]))
            walkable = walkable_point(world, cfg, start)
        else:
            if door is None:
                door = (door_start(world, cfg),)
            start = door[0]
            if start is None:
                start = random_start(mrng, world, cfg)
            if start is None:
                raise ConfigError(name, "no walkable start: no door start and none in 1000 tries")
            walkable = True
        mode = cfg.people_mode.get(i)
        if mode is None:
            walking = walkable
        elif mode == "walk":
            if not walkable:
                raise ConfigError("people_mode", f"{name} is set to walk but its start "
                                  f"({start[0]:.3f}, {start[1]:.3f}) is not walkable")
            walking = True
        else:
            walking = False
        on = None if walking else _first_containing_object(world, start)
        if walking:
            posture = "stand"
        elif on is None:
            posture = "stand"
        elif on.kind == "bed":
            posture = "lie"
        elif on.kind in ("chair", "sofa"):
            posture = "sit"
        else:
            posture = "stand"
        override = cfg.people_posture.get(i)
        if override is not None:
            if walking and override != "stand":
                raise ConfigError("people_posture", f"{name} walks, and a walking person stands")
            if override == "lie" and (on is None or on.kind != "bed"):
                raise ConfigError("people_posture", f"{name} is set to lie but no bed is under its start")
            posture = override
        bed = on if posture == "lie" else None
        traj = Trajectory(start, mrng, world, cfg) if walking else None
        out.append(Person(
            index=i, height=height, f_b=f_b, p2p=p2p, phi_b=phi_b, phi_g=phi_g,
            start=start, walking=walking, posture=posture,
            on_object=on.index if on is not None else -1, bed=bed,
            heading0=float(cfg.start_heading_rad.get(i, 0.0)),
            trajectory=traj, rcs=rcs,
        ))
    return out


# -- §8.5 shadowing ------------------------------------------------------------


def J1(nu: float) -> float:
    """ITU-R P.526 single knife edge loss, dB (§8.5)."""
    if nu <= -0.78:
        return 0.0
    w = nu - 0.1
    return 6.9 + 20.0 * math.log10((math.sqrt(w * w + 1.0) + nu) - 0.1)


def three_edge_J(nu_1: float, nu_2: float, nu_t: float, cap: float) -> float:
    """The P.526 finite-width-screen average over the near side edge
    `nu_1`, the far side edge `nu_2` and the top `nu_t` (§8.5):
    `clamp(-10*log10((10^(-J1(nu_1)/10) + 10^(-J1(nu_2)/10)) + 10^(-J1(nu_t)/10)), 0, cap)`."""
    a = 10.0 ** (-J1(nu_1) / 10.0)
    b = 10.0 ** (-J1(nu_2) / 10.0)
    c = 10.0 ** (-J1(nu_t) / 10.0)
    return clamp(-10.0 * math.log10((a + b) + c), 0.0, cap)


def shadow_J(P1, P2, c, top: float, radius: float, cap: float, lam: float):
    """§8.5 loss `J` (dB) of the leg `P1 -> P2` by a cylinder at `c`, or
    None for no loss."""
    rx = P2[0] - P1[0]
    ry = P2[1] - P1[1]
    L2 = rx * rx + ry * ry
    if L2 == 0.0:
        return None
    u = ((c[0] - P1[0]) * rx + (c[1] - P1[1]) * ry) / L2
    if u <= 0.0 or u >= 1.0:
        end = P1 if u <= 0.0 else P2
        if dist2(end, c) < radius and end[2] <= top:
            return cap
        return None
    D = dist3(P2, P1)
    zs = P1[2] + u * (P2[2] - P1[2])
    d_perp = dist2(c, (P1[0] + u * rx, P1[1] + u * ry))
    d1 = u * D
    d2 = (1.0 - u) * D
    k = math.sqrt(2.0 * (d1 + d2) / ((lam * d1) * d2))
    nu_1 = (radius - d_perp) * k
    nu_2 = (radius + d_perp) * k
    nu_t = (top - zs) * k
    return three_edge_J(nu_1, nu_2, nu_t, cap)


def shadow_factor(P1, P2, c, top: float, radius: float, cap: float, lam: float) -> float:
    """`10^(-J/20)` of one leg and one person (1.0 for no loss)."""
    J = shadow_J(P1, P2, c, top, radius, cap, lam)
    if J is None:
        return 1.0
    return 10.0 ** (-J / 20.0)
