"""Room Sim v1, top level (spec/roomsim.md §10, §12-13).

    from wifisense.roomsim import Config, RoomSim, load_scene

    sim = RoomSim(load_scene("spec/scenes/bedroom.json"), Config(duration_s=2.0))
    for tr in sim.transmissions():      # §10 order: superframe m, slot i
        tr.truth                        # people and phone at tr.t (SIM truth)
        for pkt in tr.packets:          # receivers in id order, lost included
            pkt.H, pkt.buffer, pkt.datagram

Everything this produces is SIM: a physically motivated model, not a
measurement. Room geometry and furniture come from the scene file only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .channel import LinkModel, snapshot_people
from .config import Config
from .impair import (
    BUFFER_ACTIVE, BUFFER_K, LinkImpairment, datagram, draw_device, received_buffer,
)
from .paths import draw_clutter, pair_index, tail_draws
from .people import build_people
from .rng import Rng
from .world import World

__all__ = ["RoomSim", "Packet", "Truth", "PersonTruth", "Transmission", "Config", "LABEL"]

LABEL = "SIM"


@dataclass
class PersonTruth:
    """`people[index]` at one transmission: `walking` is the §8.3 walk
    flag `s` at that instant (False while pausing or stationary)."""

    x: float
    y: float
    heading: float
    walking: bool
    posture: str


@dataclass
class Truth:
    """One record per transmission `(m, i)`, lost or not (§13)."""

    m: int
    slot: int
    t: float
    tx: int
    people: list
    phone: tuple | None
    label: str = LABEL


@dataclass
class Packet:
    """One packet record (§13). Received-only fields are None when lost.

    `buffer` holds 128 `(re, im)` entries in buffer order (LLTF then
    HT-LTF, FFT order; `RoomSim.buffer_k`), int8 values when `quantize`.
    `agc_db` is the link's AGC state after this packet's step (§12.2);
    `draws` the packet's §12.2 draws (`impair.PacketDraws`).
    """

    t: float
    tx: int
    rx: int
    link_index: int
    lost: bool
    m: int
    slot: int
    agc_db: float
    seq: int | None = None
    rssi: int | None = None
    rssi_true: float | None = None
    noise_floor: int | None = None
    H: list | None = None
    buffer: list | None = None
    buffer_pre: list | None = None
    datagram: bytes | None = None
    draws: object = None
    label: str = LABEL


@dataclass
class Transmission:
    """One transmission `(m, i)`: its truth, its packet records and the
    people's §8.3 snapshots (body points) it was computed with."""

    m: int
    slot: int
    t: float
    tx: int
    truth: Truth
    packets: list = field(default_factory=list)
    snapshots: list = field(default_factory=list)


class RoomSim:
    """A scene and a configuration, validated, with the run-wide draws
    (clutter §6.7, devices §12.1, tail draws §6.8) made once."""

    def __init__(self, scene, cfg: Config | None = None) -> None:
        cfg = Config() if cfg is None else cfg
        cfg.validate(scene)
        self.scene = scene
        self.cfg = cfg
        self.world = World(scene, cfg)
        self.clutter = draw_clutter(self.world, cfg)
        ids = self.world.node_ids
        self.devices = {n: draw_device(cfg, n, Rng.stream(cfg.seed, 500 + n)) for n in ids}
        roles = {n: cfg.roles.get(n, "txrx") for n in ids}
        self.transmitters = [n for n in ids if roles[n] in ("tx", "txrx")]
        self.receivers = [n for n in ids if roles[n] in ("rx", "txrx")]
        self.links = sorted((a, b) for a in self.transmitters for b in self.receivers if a != b)
        self.link_index = {lk: i for i, lk in enumerate(self.links)}
        self.pairs = [(a, b) for i, a in enumerate(sorted(ids)) for b in sorted(ids)[i + 1:]]
        # §6.8: every pair's draws happen, whether or not `tail` is on.
        self._tail_draws = [tail_draws(cfg, pi) for pi in range(len(self.pairs))]
        self._models: dict = {}
        # Validate the people now (starts, overrides): a bad run fails here.
        self.people_template = build_people(self.world, cfg, scene)
        self.buffer_k = BUFFER_K
        self.buffer_active = BUFFER_ACTIVE

    @property
    def n_transmitters(self) -> int:
        return len(self.transmitters)

    def tx_period(self, node: int) -> int:
        return self.cfg.tx_period.get(node, 1)

    def per_link_rate_hz(self, tx: int) -> float:
        """§10: `1 / ((nT * slot_s) * tx_period[tx])`."""
        return 1.0 / ((float(self.n_transmitters) * self.cfg.slot_s) * float(self.tx_period(tx)))

    def link_model(self, tx: int, rx: int) -> LinkModel:
        """The static model of the link `tx -> rx` (any two distinct nodes)."""
        key = (tx, rx)
        if key not in self._models:
            pi = pair_index(self.world.node_ids, tx, rx)
            self._models[key] = LinkModel(self.world, self.cfg, tx, rx, self.clutter, self._tail_draws[pi])
        return self._models[key]

    def schedule(self):
        """§10: `(m, i, t, tx)` of every transmission with `t < duration_s`."""
        cfg = self.cfg
        nT = self.n_transmitters
        m = 0
        while True:
            for i, tx in enumerate(self.transmitters):
                t = float(m * nT + i) * cfg.slot_s
                if not (t < cfg.duration_s):
                    return
                if m % self.tx_period(tx) == 0:
                    yield m, i, t, tx
            m += 1

    def _truth(self, m, i, t, tx, snaps, people) -> Truth:
        pt = [PersonTruth(s.x, s.y, s.heading, s.s == 1, s.posture) for s in snaps]
        phone = None
        if self.scene.phone is not None:
            c = self.cfg.phone_carrier
            if c < len(snaps):
                s = snaps[c]
                bx, by, bz = self.cfg.phone_offset_m
                ch = math.cos(s.heading)
                sh = math.sin(s.heading)
                x = s.x + (bx * ch - by * sh)
                y = s.y + (bx * sh + by * ch)
                if s.posture == "sit":
                    z = 0.72 * bz
                elif s.posture == "lie":
                    b = people[c].bed.box
                    z = (b.cz + b.sh / 2.0) + 0.05
                else:
                    z = bz
                phone = (x, y, z)
            else:
                st = self.scene.phone.start
                phone = (float(st[0]), float(st[1]), 1.0)
        return Truth(m, i, t, tx, pt, phone)

    def transmissions(self, *, channel: bool = True):
        """Generate every transmission in §10 order with its truth and its
        packet records. Each call is a fresh, identical run. With
        `channel=False` only the truth and the draws are computed."""
        cfg = self.cfg
        people = build_people(self.world, cfg, self.scene)
        links = {}
        for (a, b), li in self.link_index.items():
            p = cfg.loss_prob_link.get(f"{a},{b}", cfg.loss_prob)
            links[(a, b)] = LinkImpairment(cfg, Rng.stream(cfg.seed, 1000 + li), float(p))
        seq = {n: 0 for n in self.receivers}
        for m, i, t, tx in self.schedule():
            snaps = snapshot_people(people, cfg, t)
            tr = Transmission(m, i, t, tx, self._truth(m, i, t, tx, snaps, people), snapshots=snaps)
            for rx in self.receivers:
                if rx == tx:
                    continue
                li = self.link_index[(tx, rx)]
                imp = links[(tx, rx)]
                draws, lost = imp.next_packet()
                pkt = Packet(t, tx, rx, li, lost, m, i, imp.agc, draws=draws)
                if not lost:
                    pkt.seq = seq[rx]
                    seq[rx] = (seq[rx] + 1) & 0xFFFFFFFF
                    dev_rx = self.devices[rx]
                    pkt.noise_floor = dev_rx.nf_dbm
                    if channel:
                        H = self.link_model(tx, rx).channel(snaps)
                        pkt.H = H
                        rt, rssi, pre, buf = received_buffer(cfg, H, self.devices[tx], dev_rx, draws, imp.agc)
                        pkt.rssi_true = rt
                        pkt.rssi = rssi
                        pkt.buffer_pre = pre
                        pkt.buffer = buf
                        if cfg.quantize:
                            pkt.datagram = datagram(cfg, rx, pkt.seq, rssi, dev_rx.nf_dbm, buf)
                tr.packets.append(pkt)
            yield tr

    def packets(self):
        """Every packet record, in §10 output order."""
        for tr in self.transmissions():
            yield from tr.packets

    def truth(self):
        """Every truth record (no channel is computed)."""
        for tr in self.transmissions(channel=False):
            yield tr.truth
