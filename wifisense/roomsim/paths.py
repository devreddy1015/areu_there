"""Static paths of one link (spec/roomsim.md §6) and the §6.9 leg losses.

Paths, in this order: LoS (kind 0); order-1 surface bounces (1); order-2
bounces (2); object faces, specular (3) or point (4); clutter (5); the
diffuse tail (6). A path's length `L` is the Neumaier sum of its legs'
3-D lengths, in order. Everything here is SIM.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .geometry import dist3, dot3, inside_with_margin, unit3
from .materials import OBJECT_ITU, OBJECT_LOSS_DB, OBJECT_RHO, eta, fresnel
from .numerics import (
    C, FOUR_PI, FOUR_PI_1_5, MIN_RANGE_M, TWO_PI, Neumaier, cexpj, cmul, cscale,
)
from .rng import Rng

__all__ = [
    "KIND_LOS", "KIND_ORDER1", "KIND_ORDER2", "KIND_OBJ_SPECULAR", "KIND_OBJ_POINT",
    "KIND_CLUTTER", "KIND_TAIL", "FACE_NAMES",
    "StaticPath", "ClutterPoint", "draw_clutter", "path_length", "point_scatter_coef",
    "leg_losses", "static_paths", "tail_draws", "tail_terms", "pair_index",
]

KIND_LOS = 0
KIND_ORDER1 = 1
KIND_ORDER2 = 2
KIND_OBJ_SPECULAR = 3
KIND_OBJ_POINT = 4
KIND_CLUTTER = 5
KIND_TAIL = 6

FACE_NAMES = ("+x", "-x", "+y", "-y", "top")


@dataclass
class StaticPath:
    """One static path of a link.

    `coef0` is the coefficient before the §6.9 losses, `coef` the static
    coefficient `A_p` after them. `surf_i`/`surf_j` are §6.1 surface
    indices (-1 when unused), `obj`/`face` the object index and face
    number 0..4 (+x, -x, +y, -y, top) for kinds 3/4, `clutter` the
    clutter point index for kind 5, `tail` the term index for kind 6.
    `excl_walls[l]` are the wall indices leg `l` touches as bounce points;
    `excl_obj` the object whose face point the legs touch (or -1).
    """

    kind: int
    points: list
    L: float
    coef0: tuple
    coef: tuple = (0.0, 0.0)
    surf_i: int = -1
    surf_j: int = -1
    obj: int = -1
    face: int = -1
    clutter: int = -1
    tail: int = -1
    sigma: float = 0.0
    cos_t: tuple = ()
    excl_walls: list = field(default_factory=list)
    excl_obj: int = -1

    def legs(self):
        pts = self.points
        return [(pts[i], pts[i + 1]) for i in range(len(pts) - 1)]


@dataclass
class ClutterPoint:
    pos: tuple
    sigma: float


def path_length(points) -> float:
    """Neumaier sum of the legs' 3-D lengths, in order (§6)."""
    acc = Neumaier()
    for i in range(len(points) - 1):
        acc.add(dist3(points[i + 1], points[i]))
    return acc.value()


def draw_clutter(world, cfg) -> list:
    """§6.7: `clutter_count` points from stream 1. A point with no accepted
    position consumes its 2000 draws and is dropped."""
    rng = Rng.stream(cfg.seed, 1)
    lo, hi = cfg.clutter_rcs_m2
    ln_lo = math.log(lo)
    ln_hi = math.log(hi)
    out = []
    for _ in range(cfg.clutter_count):
        pos = None
        for _try in range(1000):
            x = rng.range(world.x0, world.x1)
            y = rng.range(world.y0, world.y1)
            if inside_with_margin(world.F, (x, y), cfg.clutter_margin_m):
                pos = (x, y)
                break
        if pos is None:
            continue
        z = rng.range(0.0, world.H)
        sigma = math.exp(rng.range(ln_lo, ln_hi))
        out.append(ClutterPoint((pos[0], pos[1], z), sigma))
    return out


def point_scatter_coef(Tp, P, Rp, sigma: float, lam: float):
    """§6.6: `(a, 0)`, `a = (LAMBDA * sqrt(sigma)) / ((FOUR_PI_1_5 * d1) * d2)`
    with both ranges clamped to `MIN_RANGE_M`."""
    d1 = max(dist3(Tp, P), MIN_RANGE_M)
    d2 = max(dist3(P, Rp), MIN_RANGE_M)
    a = (lam * math.sqrt(sigma)) / ((FOUR_PI_1_5 * d1) * d2)
    return (a, 0.0)


def leg_losses(coef, P, Q, world, cfg, excl_walls=(), excl_objs=()):
    """§6.9: multiply `coef` by the wall transmissions (walls in order) and
    then the object blockages (objects in order) of the leg `P -> Q`."""
    if cfg.transmission:
        u = None
        for w in world.walls:
            if w.wall_index in excl_walls:
                continue
            if w.hit(P, Q) is not None:
                if u is None:
                    u = unit3(P, Q)
                cos_t = abs(dot3(u, w.n))
                coef = cmul(coef, w.slab.transmit(cos_t))
    if cfg.object_blockage:
        for o in world.objects:
            if o.is_opening or o.index in excl_objs:
                continue
            if o.box.leg_meets(P, Q):
                coef = cscale(coef, 10.0 ** (-OBJECT_LOSS_DB[o.material] / 20.0))
    return coef


def _apply_losses(path: StaticPath, world, cfg) -> None:
    coef = path.coef0
    excl_objs = (path.excl_obj,) if path.excl_obj >= 0 else ()
    for li, (P, Q) in enumerate(path.legs()):
        coef = leg_losses(coef, P, Q, world, cfg, path.excl_walls[li], excl_objs)
    path.coef = coef


def _wall_id(surface) -> int:
    return surface.wall_index if surface.kind == "wall" else -1


def _ids(*ids):
    return tuple(i for i in ids if i >= 0)


def static_paths(world, cfg, Tp, Rp, clutter) -> list:
    """The kind 0..5 static paths of the link `Tp -> Rp`, in §6 order, with
    their §6.9 static coefficients. The tail (§6.8) is `tail_terms`."""
    lam = world.lam
    paths: list[StaticPath] = []

    # §6.2 line of sight
    pts = [Tp, Rp]
    L = path_length(pts)
    paths.append(StaticPath(KIND_LOS, pts, L, (lam / (FOUR_PI * L), 0.0), excl_walls=[()]))

    S = world.surfaces
    # §6.3 order 1: the bounce is on the face on Tp's side, f = side_i(Tp).
    if cfg.reflection_order >= 1:
        for si in S:
            f = si.side(Tp)
            if f == 0 or not (si.front(f, Tp) and si.front(f, Rp)):
                continue
            X = si.hit(si.image(f, Tp), Rp, f)
            if X is None:
                continue
            pts = [Tp, X, Rp]
            L = path_length(pts)
            cos_t = abs(dot3(unit3(Tp, X), si.n))
            coef0 = cscale(si.slab.reflect(cos_t), lam / (FOUR_PI * L))
            wi = _ids(_wall_id(si))
            paths.append(StaticPath(KIND_ORDER1, pts, L, coef0, surf_i=si.index, cos_t=(cos_t,),
                                    excl_walls=[wi, wi]))

    # §6.4 order 2: faces fi = side_i(Tp) and fj = side_j(Rp).
    if cfg.reflection_order >= 2:
        for si in S:
            fi = si.side(Tp)
            if fi == 0:
                continue
            T1 = si.image(fi, Tp)
            for sj in S:
                if sj is si:
                    continue
                fj = sj.side(Rp)
                if fj == 0:
                    continue
                T2 = sj.image(fj, T1)
                Xj = sj.hit(T2, Rp, fj)
                if Xj is None:
                    continue
                Xi = si.hit(T1, Xj, fi)
                if Xi is None:
                    continue
                if not (si.front(fi, Tp) and si.front(fi, Xj) and sj.front(fj, Xi) and sj.front(fj, Rp)):
                    continue
                pts = [Tp, Xi, Xj, Rp]
                L = path_length(pts)
                c1 = abs(dot3(unit3(Tp, Xi), si.n))
                c2 = abs(dot3(unit3(Xi, Xj), sj.n))
                coef0 = cscale(cmul(si.slab.reflect(c1), sj.slab.reflect(c2)), lam / (FOUR_PI * L))
                wi = _ids(_wall_id(si))
                wj = _ids(_wall_id(sj))
                paths.append(StaticPath(KIND_ORDER2, pts, L, coef0, surf_i=si.index, surf_j=sj.index,
                                        cos_t=(c1, c2), excl_walls=[wi, wi + wj, wj]))

    # §6.5 object faces
    for o in world.objects:
        if o.is_opening:
            continue
        paths.extend(_object_face_paths(o, world, cfg, Tp, Rp))

    # §6.7 clutter
    for ci, cp in enumerate(clutter):
        pts = [Tp, cp.pos, Rp]
        L = path_length(pts)
        paths.append(StaticPath(KIND_CLUTTER, pts, L, point_scatter_coef(Tp, cp.pos, Rp, cp.sigma, lam),
                                clutter=ci, sigma=cp.sigma, excl_walls=[(), ()]))

    for p in paths:
        _apply_losses(p, world, cfg)
    return paths


# Face table (§6.5): local centre (as multipliers of sl/2, sb/2, sh/2),
# local outward normal, in-plane extents.
_FACES = (
    ((1, 0, 0), (1.0, 0.0, 0.0), ("sb", "sh")),
    ((-1, 0, 0), (-1.0, 0.0, 0.0), ("sb", "sh")),
    ((0, 1, 0), (0.0, 1.0, 0.0), ("sl", "sh")),
    ((0, -1, 0), (0.0, -1.0, 0.0), ("sl", "sh")),
    ((0, 0, 1), (0.0, 0.0, 1.0), ("sl", "sb")),
)


def _face_centre_local(box, f: int):
    hl = box.sl / 2.0
    hb = box.sb / 2.0
    hh = box.sh / 2.0
    if f == 0:
        return (hl, 0.0, 0.0)
    if f == 1:
        return (-hl, 0.0, 0.0)
    if f == 2:
        return (0.0, hb, 0.0)
    if f == 3:
        return (0.0, -hb, 0.0)
    return (0.0, 0.0, hh)


def _on_face(box, f: int, X) -> bool:
    lx, ly, lz = box.to_local(X)
    if f <= 1:
        return abs(ly) <= box.sb / 2.0 and abs(lz) <= box.sh / 2.0
    if f <= 3:
        return abs(lx) <= box.sl / 2.0 and abs(lz) <= box.sh / 2.0
    return abs(lx) <= box.sl / 2.0 and abs(ly) <= box.sb / 2.0


def _object_face_paths(o, world, cfg, Tp, Rp) -> list:
    box = o.box
    lam = world.lam
    out = []
    itu = OBJECT_ITU[o.material]
    et = eta(itu, world.carrier_hz) if itu is not None else None
    for f in range(5):
        lc = _face_centre_local(box, f)
        Pf = box.to_world(lc[0], lc[1], lc[2])
        nl = _FACES[f][1]
        nf = box.rotate(nl[0], nl[1], nl[2])
        e1 = getattr(box, _FACES[f][2][0])
        e2 = getattr(box, _FACES[f][2][1])
        dT = dot3((Tp[0] - Pf[0], Tp[1] - Pf[1], Tp[2] - Pf[2]), nf)
        dR = dot3((Rp[0] - Pf[0], Rp[1] - Pf[1], Rp[2] - Pf[2]), nf)
        if not (dT > 0.0 and dR > 0.0):
            continue
        if cfg.object_specular and o.material != "fabric":
            d2T = 2.0 * dT
            Tq = (Tp[0] - d2T * nf[0], Tp[1] - d2T * nf[1], Tp[2] - d2T * nf[2])
            RT = (Rp[0] - Tq[0], Rp[1] - Tq[1], Rp[2] - Tq[2])
            den = dot3(RT, nf)
            if den != 0.0:
                s = dot3((Pf[0] - Tq[0], Pf[1] - Tq[1], Pf[2] - Tq[2]), nf) / den
                if 0.0 < s < 1.0:
                    X = (Tq[0] + s * RT[0], Tq[1] + s * RT[1], Tq[2] + s * RT[2])
                    if _on_face(box, f, X):
                        d1 = dist3(Tp, X)
                        d2 = dist3(X, Rp)
                        rF = math.sqrt(lam * d1 * d2 / (d1 + d2))
                        if e1 >= 2.0 * rF and e2 >= 2.0 * rF:
                            pts = [Tp, X, Rp]
                            L = path_length(pts)
                            cos_t = abs(dot3(unit3(Tp, X), nf))
                            _s, g_te, g_tm = fresnel(et, cos_t)
                            G = g_tm if f == 4 else g_te
                            out.append(StaticPath(KIND_OBJ_SPECULAR, pts, L, cscale(G, lam / (FOUR_PI * L)),
                                                  obj=o.index, face=f, cos_t=(cos_t,),
                                                  excl_walls=[(), ()], excl_obj=o.index))
                            continue
        if cfg.object_scatter:
            sigma = cfg.object_scatter_scale * OBJECT_RHO[o.material] * (e1 * e2)
            pts = [Tp, Pf, Rp]
            L = path_length(pts)
            out.append(StaticPath(KIND_OBJ_POINT, pts, L, point_scatter_coef(Tp, Pf, Rp, sigma, lam),
                                  obj=o.index, face=f, sigma=sigma, excl_walls=[(), ()], excl_obj=o.index))
    return out


def pair_index(node_ids, a: int, b: int) -> int:
    """Position of the unordered pair {a, b} in the list of all node-id
    pairs (a, b), a < b, sorted (§6.8)."""
    lo, hi = (a, b) if a < b else (b, a)
    ids = sorted(node_ids)
    idx = 0
    for i, x in enumerate(ids):
        for y in ids[i + 1:]:
            if (x, y) == (lo, hi):
                return idx
            idx += 1
    raise KeyError((a, b))


def tail_draws(cfg, pidx: int) -> list:
    """§6.8 draws of one unordered pair: `[(x_n, phi_n)]`."""
    rng = Rng.stream(cfg.seed, 100000 + pidx)
    cd = C * cfg.tail_decay_s
    out = []
    for _ in range(cfg.tail_count):
        x = -(cd * math.log(1.0 - rng.uniform()))
        phi = rng.range(0.0, TWO_PI)
        out.append((x, phi))
    return out


def tail_terms(cfg, paths, draws) -> list:
    """§6.8: the diffuse tail of a link from its kind 0..3 paths and its
    pair's draws: `tail_count` terms of equal amplitude
    `sqrt((10^(tail_power_db/10) * P_spec) / float(tail_count))`. Empty
    when `tail` is off (the draws still happen)."""
    if not cfg.tail:
        return []
    L_ref = None
    p_spec = Neumaier()
    for p in paths:
        if p.kind <= KIND_OBJ_SPECULAR:
            if L_ref is None or p.L > L_ref:
                L_ref = p.L
            p_spec.add(p.coef[0] * p.coef[0] + p.coef[1] * p.coef[1])
    P_spec = p_spec.value()
    out = []
    if not draws:
        return out
    # Every term has the same amplitude; the exponential delays alone give
    # the power-delay profile exp(-tau / tail_decay_s) (§6.8).
    a_tail = math.sqrt(((10.0 ** (cfg.tail_power_db / 10.0)) * P_spec) / float(cfg.tail_count))
    for n, (x, phi) in enumerate(draws):
        coef = cscale(cexpj(phi), a_tail)
        out.append(StaticPath(KIND_TAIL, [], L_ref + x, coef, coef, tail=n))
    return out
