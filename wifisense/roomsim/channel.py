"""The clean channel of one packet (spec/roomsim.md §10-11).

`LinkModel` holds a link's static paths (§6, computed once) and evaluates

    H_k = sum over static kind 0..5 paths p:  cmul(cscale(A_p, S_p(t)), cexpj(theta(f_k, L_p)))
        + sum over tail terms n:              cmul(A_n, cexpj(theta(f_k, L_n)))
        + sum over people q, points j:        cmul(cscale(A_qj(t), S_qj(t)), cexpj(theta(f_k, L_qj(t))))

with one Neumaier sum per component, in exactly that order. The static
phasors `cexpj(theta(f_k, L_p))` are memoised per link: the same values,
computed once.
"""

from __future__ import annotations

from dataclasses import dataclass

from .numerics import C, TWO_PI, cexpj, cscale
from .paths import leg_losses, path_length, point_scatter_coef, static_paths, tail_terms
from .people import shadow_factor

__all__ = ["PersonSnapshot", "LinkModel", "snapshot_people"]


@dataclass
class PersonSnapshot:
    """A person at one transmission time: state, body points, cylinder."""

    index: int
    x: float
    y: float
    heading: float
    s: int
    v: float
    points: list
    rcs: tuple
    top: float
    excl_obj: tuple   # object their own body legs skip (§6.9), if any
    posture: str


def snapshot_people(people, cfg, t: float) -> list:
    """Every person's §9 state and §8.3 body points at `t`."""
    out = []
    for p in people:
        st = p.state(t)
        pts = p.body_points(t, st, cfg.body_points)
        excl = ()
        if not p.walking and p.posture in ("sit", "lie") and p.on_object >= 0:
            excl = (p.on_object,)
        out.append(PersonSnapshot(p.index, st[0], st[1], st[2], st[3], st[4], pts, p.rcs, p.top, excl, p.posture))
    return out


def _theta(f: float, L: float) -> float:
    return (((-TWO_PI) * f) * L) / C


class LinkModel:
    """Static part of the link `tx -> rx` and its per-packet channel."""

    def __init__(self, world, cfg, tx: int, rx: int, clutter, tail_draws) -> None:
        self.world = world
        self.cfg = cfg
        self.tx = tx
        self.rx = rx
        self.Tp = world.node_pos[tx]
        self.Rp = world.node_pos[rx]
        self.paths = static_paths(world, cfg, self.Tp, self.Rp, clutter)
        self.tail = tail_terms(cfg, self.paths, tail_draws)
        freqs = world.freqs
        self._E = [[cexpj(_theta(f, p.L)) for f in freqs] for p in self.paths]
        self._Et = [[cexpj(_theta(f, p.L)) for f in freqs] for p in self.tail]
        self._legs = [p.legs() for p in self.paths]

    def static_coefficients(self, people) -> list:
        """`cscale(A_p, S_p(t))` of every kind 0..5 path (§8.5 shadowing by
        every person, legs outer, people inner)."""
        cfg = self.cfg
        lam = self.world.lam
        out = []
        do_shadow = cfg.shadowing and len(people) > 0
        for p, legs in zip(self.paths, self._legs):
            S = 1.0
            if do_shadow:
                for P1, P2 in legs:
                    for q in people:
                        S = S * shadow_factor(P1, P2, (q.x, q.y), q.top, cfg.body_radius_m,
                                              cfg.shadow_cap_db, lam)
            out.append(cscale(p.coef, S))
        return out

    def body_terms(self, people) -> list:
        """`(cscale(A_qj, S_qj), L_qj)` for every person and body point."""
        cfg = self.cfg
        world = self.world
        lam = world.lam
        Tp, Rp = self.Tp, self.Rp
        out = []
        for q in people:
            for j, P in enumerate(q.points):
                coef = point_scatter_coef(Tp, P, Rp, q.rcs[j], lam)
                coef = leg_losses(coef, Tp, P, world, cfg, (), q.excl_obj)
                coef = leg_losses(coef, P, Rp, world, cfg, (), q.excl_obj)
                S = 1.0
                if cfg.shadowing:
                    for P1, P2 in ((Tp, P), (P, Rp)):
                        for r in people:
                            if r.index == q.index:
                                continue
                            S = S * shadow_factor(P1, P2, (r.x, r.y), r.top, cfg.body_radius_m,
                                                  cfg.shadow_cap_db, lam)
                out.append((cscale(coef, S), path_length([Tp, P, Rp])))
        return out

    def channel(self, people) -> list:
        """`H_k` for the 64 bins, ascending k (§11)."""
        statics = self.static_coefficients(people)
        body = self.body_terms(people)
        tails = [p.coef for p in self.tail]
        E = self._E
        Et = self._Et
        freqs = self.world.freqs
        H = []
        for k in range(64):
            hr = lr = hi_ = li = 0.0
            # The static, tail and body terms, Neumaier per component (§1),
            # inlined for speed: exactly Neumaier.add / value().
            terms = []
            for i, A in enumerate(statics):
                e = E[i][k]
                terms.append((A[0] * e[0] - A[1] * e[1], A[0] * e[1] + A[1] * e[0]))
            for i, A in enumerate(tails):
                e = Et[i][k]
                terms.append((A[0] * e[0] - A[1] * e[1], A[0] * e[1] + A[1] * e[0]))
            f = freqs[k]
            for A, L in body:
                e = cexpj(_theta(f, L))
                terms.append((A[0] * e[0] - A[1] * e[1], A[0] * e[1] + A[1] * e[0]))
            for xr, xi in terms:
                t = hr + xr
                if abs(hr) >= abs(xr):
                    lr += (hr - t) + xr
                else:
                    lr += (xr - t) + hr
                hr = t
                t = hi_ + xi
                if abs(hi_) >= abs(xi):
                    li += (hi_ - t) + xi
                else:
                    li += (xi - t) + hi_
                hi_ = t
            re = hr + lr if (lr != 0.0 and lr - lr == 0.0) else hr
            im = hi_ + li if (li != 0.0 and li - li == 0.0) else hi_
            H.append((re, im))
        return H
