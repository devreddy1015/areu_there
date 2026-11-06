"""Room Sim's side of the Sionna audit (SIM): deterministic per-link paths.

Everything is read through the Python reference `wifisense.roomsim`,
read-only: `RoomSim(scene, cfg)` validates the run and draws the clutter and
tail exactly as a real run does, and `RoomSim.link_model(tx, rx)` computes
the link's static paths (spec/roomsim.md §6: kinds 0..5 with their §6.9
static coefficients `A_p`) and its diffuse tail (kind 6, §6.8). No people
are simulated and no impairment is applied (§11 with `people = []`, before
§12): this is the clean, static, empty-room channel.

Path kinds (§6): 0 LoS, 1 order-1 surface bounce, 2 order-2 bounce,
3 object specular face, 4 object point scatterer, 5 clutter, 6 tail.
The audit's "specular" set is kinds 0..3 (the paths Room Sim builds from
geometry and the slab/interface coefficients); kinds 4..6 are its
stochastic or heuristic parts, reported separately.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np

from wifisense.roomsim import Config, RoomSim
from wifisense.roomsim.numerics import C
from wifisense.roomsim.world import BINS

from .scene_to_sionna import LLTF_BINS

__all__ = ["RoomSimLink", "shell_config", "roomsim_links", "SPECULAR_KINDS", "KIND_NAMES",
           "expected_tail_moments"]

SPECULAR_KINDS = (0, 1, 2, 3)
KIND_NAMES = {0: "los", 1: "order1", 2: "order2", 3: "object_specular", 4: "object_point",
              5: "clutter", 6: "tail"}
_LLTF_INDEX = [BINS.index(k) for k in LLTF_BINS]


@dataclass
class RoomSimLink:
    """One Room Sim link `tx -> rx` (SIM): every path's kind, length `L`
    (m), static coefficient `A_p` (complex; tail terms carry their §6.8
    coefficient), §6.1 surface indices (`surf_i`, `surf_j`, -1 unused),
    object index and face (-1 unused); and `H` (the reference's own §11
    channel with no people) at the 52 LLTF bins."""

    tx: int
    rx: int
    kind: np.ndarray
    L: np.ndarray
    coef: np.ndarray
    surf_i: np.ndarray
    surf_j: np.ndarray
    obj: np.ndarray
    face: np.ndarray
    H: np.ndarray
    tail_power_db: float
    tail_decay_s: float

    def select(self, kinds) -> np.ndarray:
        return np.isin(self.kind, np.asarray(list(kinds)))

    @property
    def tau(self) -> np.ndarray:
        return self.L / C


def shell_config(cfg: Config) -> Config:
    """`cfg` with every object effect off (§3.2 `object_specular`,
    `object_scatter`, `object_blockage`): the walls, floor and ceiling alone,
    the Room Sim side of the audit's "shell" variant."""
    return dataclasses.replace(cfg, object_specular=False, object_scatter=False, object_blockage=False)


def roomsim_links(scene, cfg: Config, links=None) -> dict:
    """`{(tx, rx): RoomSimLink}` for `links` (default: every ordered pair of
    distinct node ids), from the reference's `RoomSim.link_model`."""
    sim = RoomSim(scene, cfg)
    ids = sorted(sim.world.node_ids)
    if links is None:
        links = [(a, b) for a in ids for b in ids if a != b]
    out = {}
    for tx, rx in links:
        lm = sim.link_model(tx, rx)
        ps = list(lm.paths) + list(lm.tail)
        H64 = lm.channel([])
        out[(tx, rx)] = RoomSimLink(
            tx=tx, rx=rx,
            kind=np.array([p.kind for p in ps], dtype=np.int64),
            L=np.array([p.L for p in ps], dtype=np.float64),
            coef=np.array([complex(p.coef[0], p.coef[1]) for p in ps], dtype=np.complex128),
            surf_i=np.array([p.surf_i for p in ps], dtype=np.int64),
            surf_j=np.array([p.surf_j for p in ps], dtype=np.int64),
            obj=np.array([p.obj for p in ps], dtype=np.int64),
            face=np.array([p.face for p in ps], dtype=np.int64),
            H=np.array([complex(H64[i][0], H64[i][1]) for i in _LLTF_INDEX], dtype=np.complex128),
            tail_power_db=float(cfg.tail_power_db) if cfg.tail else float("-inf"),
            tail_decay_s=float(cfg.tail_decay_s),
        )
    return out


def expected_tail_moments(link: RoomSimLink, anchor: str = "L_ref") -> tuple:
    """The link's power-weighted delay moments `(P, mean_tau, rms_tau)` with
    its tail replaced by its expectation (§6.8): an exponential power-delay
    profile of total `10^(tail_power_db/10) * P_spec`, decay `tail_decay_s`,
    starting at `L_ref / C` (`L_ref` the longest kind 0..3 path, `P_spec`
    their power). Kinds 0..5 enter as they are.

    `anchor="los"` starts the expected tail at the LoS delay instead: a
    what-if diagnostic of how much of the delay spread the anchor makes,
    not Room Sim's model."""
    det = link.kind <= 5
    p = np.abs(link.coef[det]) ** 2
    tau = link.tau[det]
    spec = link.kind <= 3
    P_spec = float(np.sum(np.abs(link.coef[spec]) ** 2))
    P = float(np.sum(p))
    m1 = float(np.sum(p * tau))
    m2 = float(np.sum(p * tau * tau))
    if np.isfinite(link.tail_power_db) and np.any(spec):
        if anchor == "L_ref":
            L_ref = float(np.max(link.L[spec]))
        elif anchor == "los":
            L_ref = float(np.min(link.L[link.kind == 0]))
        else:
            raise ValueError(anchor)
        t0 = L_ref / C
        td = link.tail_decay_s
        Pt = (10.0 ** (link.tail_power_db / 10.0)) * P_spec
        P += Pt
        m1 += Pt * (t0 + td)
        m2 += Pt * (t0 * t0 + 2.0 * t0 * td + 2.0 * td * td)
    mean = m1 / P
    return P, mean, float(np.sqrt(max(m2 / P - mean * mean, 0.0)))
