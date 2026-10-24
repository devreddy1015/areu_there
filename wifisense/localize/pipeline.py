"""Locate on Room Sim runs (spec/locate.md §4.3, §10, §11): ticks and phone epochs.

    from wifisense.roomsim import Config, load_scene
    from wifisense.localize.config import LocateConfig
    from wifisense.localize.pipeline import run_locate

    run = run_locate(load_scene("spec/scenes/bedroom.json"), Config(duration_s=30.0, seed=1))
    run.ticks     # one record per tick:  {label, t, wlm, rti, truth}
    run.epochs    # one per fired phone epoch: {label, t, rtt, rssi, reports, truth}

Everything here is **SIM**: its inputs are Room Sim packets and truth, its
geometry the scene. Each estimate is a position with a radius
(`radius_calibrated: False`) or an abstention with a reason. Truth is used
only to simulate the phone's radio and to score; no `Person.id` or
`Phone.id` appears in any record, log or error.

The person pipeline (§10.1): Room Sim packets become `CsiFrame`s (§4.3)
for one `LinkMeasurement` (default signal `RssiSignal`), calibrated on an
empty-room Room Sim run of the same room (§10.2 step 1); at each tick the
weighted-link-midpoint baseline (§5) and the ported RTI (`Inverter` then
`Tracker`) each give an estimate or abstain. The phone pipeline (§10.2
step 3) feeds every transmission's truth into `phone.PhoneRun`.

Refused, with an error naming the parameter: any label but "SIM" on a
Room Sim run (`label`; a `PersonPipeline` fed by a caller takes SIM, REAL
or SIM->REAL), a Room Sim config with any `roles` entry (the localize
link order is every ordered pair), a scene with two nodes at one `(x, y)`
(`nodes`), `quantize` false (no int8 buffer), a set `phone_seed`,
`tx_power_dbm` or `carrier_hz` that is not Room Sim's, a `tick_hz` above
one tick per slot or whose ticks stop advancing (`tick_hz`), a
transmitter whose `tx_period` means its links can never fill a window
(`tx_period`), and a calibration that fails (`calib_s`).
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

from ..roomsim import Config as RoomSimConfig
from ..roomsim import RoomSim
from ..roomsim.numerics import carrier_hz as roomsim_carrier_hz
from ..types import CsiFrame
from .baseline import weighted_link_midpoint
from .config import LocateConfig, LocateError, check_person_label, check_roomsim_label
from .geometry import Geometry, Node, PixelGrid, Room
from .invert import Inverter
from .measure import LinkMeasurement, LinkSignal, RssiSignal, StaticSuppressedSignal, TotalPowerSignal
from .phone import PhoneRun
from .radius import abstention, estimate_record, publishable_estimate, radius95
from .track import Tracker

__all__ = [
    "LocateRun",
    "PersonPipeline",
    "make_signal",
    "locate_geometry",
    "check_roomsim_input",
    "base_rate_hz",
    "check_tx_period",
    "frame_from_packet",
    "tick_time",
    "next_tick",
    "check_tick_hz",
    "calibration_config",
    "calibration_stream",
    "phone_nodes",
    "phone_walls",
    "phone_boxes",
    "person_cylinders",
    "tick_truth",
    "run_locate",
]


# ---------------------------------------------------------------------------
# inputs (§2) and glue (§4.3)
# ---------------------------------------------------------------------------


def make_signal(name: str) -> LinkSignal:
    """The `LinkSignal` named `name` (`LocateConfig.link_signal`)."""
    if name == "rssi_dbm":
        return RssiSignal()
    if name == "total_power_db":
        return TotalPowerSignal()
    if name == "static_suppressed_amplitude":
        return StaticSuppressedSignal()
    raise LocateError("link_signal", f"{name!r} is not a link signal")


def locate_geometry(scene, cfg: LocateConfig) -> Geometry:
    """§2: `Room(width_m = room.w_m, height_m = room.d_m)` (the footprint's
    bounding rectangle, origin (0, 0); `height_m` is the second horizontal
    axis), nodes `Node(id, x, y)` (z dropped), a `pitch_m` grid, `lambda_m`."""
    room = Room(width_m=float(scene.room.w_m), height_m=float(scene.room.d_m))
    nodes = [Node(n.id, float(n.pos[0]), float(n.pos[1])) for n in scene.nodes]
    return Geometry(room, nodes, PixelGrid(room, cfg.pitch_m), cfg.lambda_m)


def check_roomsim_input(scene, rs_cfg: RoomSimConfig) -> None:
    """§2, §4.3 refusals for the person pipeline: any `roles` entry
    (`roles`), two nodes at one `(x, y)` (`nodes`), `quantize` false
    (`quantize`)."""
    if rs_cfg.roles:
        raise LocateError(
            "roles",
            "the person pipeline needs every node to be txrx (the localize link order is every "
            "ordered pair); this config sets roles",
        )
    seen = {}
    for n in scene.nodes:
        xy = (float(n.pos[0]), float(n.pos[1]))
        if xy in seen:
            raise LocateError(
                "nodes",
                f"nodes {seen[xy]} and {n.id} share (x, y) = ({xy[0]:g}, {xy[1]:g}); the localize "
                "geometry is 2-D",
            )
        seen[xy] = n.id
    if not rs_cfg.quantize:
        raise LocateError("quantize", "the glue reads the int8 buffer, which quantize false does not produce")


def base_rate_hz(n_transmitters: int, slot_s: float) -> float:
    """§4.3: `rate_hz = 1.0 / (float(nT) * slot_s)`, Room Sim's base per-link rate."""
    den = float(n_transmitters) * slot_s
    if not (den > 0.0):
        raise LocateError("slot_s", f"nT * slot_s = {den!r} is not > 0")
    return 1.0 / den


def check_tx_period(transmitters, tx_period: dict, window_s: float, rate_hz: float, min_samples: int) -> None:
    """§4.1: refuse (naming `tx_period`) a transmitter with
    `floor(window_s * rate_hz / float(tx_period[tx])) + 1 < min_samples`."""
    for tx in transmitters:
        p = tx_period.get(tx, 1)
        if math.floor(window_s * rate_hz / float(p)) + 1 < min_samples:
            raise LocateError(
                "tx_period",
                f"node {tx} transmits every {p} superframes, so its links get at most "
                f"{math.floor(window_s * rate_hz / float(p)) + 1} samples in a {window_s:g} s window, "
                f"fewer than min_samples = {min_samples}",
            )


def frame_from_packet(pkt, fw_subcarriers: int, rate_hz: float) -> CsiFrame:
    """§4.3: one received Room Sim packet as a `CsiFrame`.

    `amplitudes[k] = sqrt(float(re*re + im*im))` over the first
    `fw_subcarriers` int8 entries of `buffer` (integer products, never
    `hypot`), `phases` all 0.0, `sample_index = seq`, `sample_rate_hz =
    rate_hz`, `rssi_dbm = rssi`, `node_id = rx`.
    """
    if pkt.lost or pkt.buffer is None:
        raise ValueError("a lost packet has no frame")
    amps = []
    for e in range(fw_subcarriers):
        re, im = pkt.buffer[e]
        re = int(re)
        im = int(im)
        amps.append(math.sqrt(float(re * re + im * im)))
    return CsiFrame(amps, [0.0] * fw_subcarriers, pkt.seq, rate_hz, pkt.rssi, pkt.rx)


def tick_time(k: int, window_s: float, tick_hz: float) -> float:
    """§10.1: `t_k = window_s + float(k) / tick_hz`, never accumulated."""
    return window_s + float(k) / tick_hz


def next_tick(k: int, t_k: float, window_s: float, tick_hz: float) -> tuple[int, float]:
    """§10.1: `(k + 1, t_{k+1})`, `t_{k+1}` computed from `k + 1`. A tick
    that does not advance (`not (t_{k+1} > t_k)`: `float(k) / tick_hz` no
    longer moves `window_s`) refuses the run naming `tick_hz`; its ticks
    would never reach `duration_s`."""
    k += 1
    t = tick_time(k, window_s, tick_hz)
    if not (t > t_k):
        raise LocateError("tick_hz", f"tick {k} does not advance past t = {t_k!r} s")
    return k, t


def check_tick_hz(tick_hz: float, slot_s: float) -> None:
    """§9: on a Room Sim run, `tick_hz <= 1.0 / slot_s` (at most one tick
    per slot), else refused naming `tick_hz`. `slot_s > 0` is Room Sim's
    own check, made before this one."""
    if not (tick_hz <= 1.0 / slot_s):
        raise LocateError("tick_hz", f"{tick_hz!r} Hz is more than one tick per {slot_s!r} s slot")


def calibration_config(rs_cfg: RoomSimConfig, calib_s: float) -> RoomSimConfig:
    """§10.2 step 1: same scene, config and seed; `n_people = 0`;
    `people_mode`, `people_posture`, `start_heading_rad` emptied;
    `duration_s = duration_s + calib_s`."""
    return dataclasses.replace(
        rs_cfg,
        n_people=0,
        people_mode={},
        people_posture={},
        start_heading_rad={},
        duration_s=rs_cfg.duration_s + calib_s,
    )


def calibration_stream(calib_sim: RoomSim, duration_s: float, fw_subcarriers: int, rate_hz: float):
    """The empty-room run's received packets with `t >= duration_s`, as
    `(t, tx, rx, frame)` (§10.2 step 1)."""
    for tr in calib_sim.transmissions():
        if not (tr.t >= duration_s):
            continue
        for pkt in tr.packets:
            if pkt.lost:
                continue
            yield pkt.t, pkt.tx, pkt.rx, frame_from_packet(pkt, fw_subcarriers, rate_hz)


def phone_nodes(scene) -> list:
    """`(node_id, x, y, z)` of every node, id order (§6.1)."""
    return sorted((n.id, float(n.pos[0]), float(n.pos[1]), float(n.pos[2])) for n in scene.nodes)


def phone_walls(world) -> list:
    """Room Sim's `W` (implied walls included) as `(ax, ay, bx, by, h)`."""
    return [(w.A[0], w.A[1], w.B[0], w.B[1], w.h) for w in world.walls]


def phone_boxes(world) -> list:
    """Every object that is not a door or window, as
    `(cx, cy, cz, sl, sb, sh, cs, sn)`."""
    out = []
    for o in world.objects:
        if o.is_opening:
            continue
        b = o.box
        out.append((b.cx, b.cy, b.cz, b.sl, b.sb, b.sh, b.cs, b.sn))
    return out


def person_cylinders(snapshots, body_radius_m: float) -> list:
    """Every person's §8.5 cylinder `(cx, cy, radius, top)` at one transmission."""
    return [(s.x, s.y, body_radius_m, s.top) for s in snapshots]


def tick_truth(truth) -> dict:
    """§11: a tick's truth, `{"people": [{x, y, walking, posture}, ...]}`
    (index order; no scene id)."""
    if truth is None:
        return {"people": []}
    return {
        "people": [
            {"x": p.x, "y": p.y, "walking": bool(p.walking), "posture": p.posture}
            for p in truth.people
        ]
    }


# ---------------------------------------------------------------------------
# the person pipeline (§10.1)
# ---------------------------------------------------------------------------


class PersonPipeline:
    """One measurement, inverter and tracker; `feed` packets in Room Sim
    order, `tick(t_k)` for the §10.1 estimates. `label` labels its records:
    "SIM", "REAL" or "SIM->REAL" (§1; never SCENE)."""

    def __init__(self, geometry: Geometry, cfg: LocateConfig, rate_hz: float, label: str = "SIM") -> None:
        self.label = check_person_label(label)
        cfg.validate()
        self.cfg = cfg
        self.geometry = geometry
        self.rate_hz = rate_hz
        self.signal = make_signal(cfg.link_signal)
        self.meas = LinkMeasurement(geometry, window_s=cfg.window_s, sample_rate_hz=rate_hz, signal=self.signal)
        self.inverter = Inverter(
            geometry, cfg.alpha, cfg.sigma, cfg.delta_c, warn_blind_pixels=False
        )
        self.tracker = Tracker(geometry, z_threshold=cfg.z_threshold)

    def calibrate(self, stream) -> None:
        """§4.2: `calibrate_from(stream, min_windows)`; a failure is the
        run's error naming `calib_s`."""
        try:
            self.meas.calibrate_from(stream, min_windows=self.cfg.min_windows)
        except ValueError as exc:
            raise LocateError("calib_s", f"the empty-room calibration failed: {exc}") from None

    def feed(self, pkt, fw_subcarriers: int) -> None:
        """§4.3: one Room Sim packet; lost packets are skipped."""
        if pkt.lost:
            return
        self.meas.update(pkt.t, pkt.tx, pkt.rx, frame_from_packet(pkt, fw_subcarriers, self.rate_hz))

    def estimates(self, t_k: float) -> tuple[dict, dict]:
        """§10.1 at tick time `t_k`: `(wlm, rti)` records."""
        cfg = self.cfg
        y = self.meas.vector(t_k)
        null_sigma = self.meas.baseline_std_vector()
        wlm = weighted_link_midpoint(
            self.geometry, y, null_sigma, self.signal.name,
            wlm_gate_z=cfg.wlm_gate_z, wlm_floor=cfg.wlm_floor_value(),
            wlm_radius_scale=cfg.wlm_radius_scale,
            min_plausible_radius_m=cfg.min_plausible_radius_m,
        )
        image = self.inverter.solve(y)
        est = self.tracker.update(image, t_k)
        if est is None:
            rti = abstention("not_significant")
        else:
            r = cfg.rti_radius_scale * radius95(est.cov_xx, est.cov_xy, est.cov_yy)
            if publishable_estimate(est.x, est.y, r, cfg.min_plausible_radius_m):
                rti = estimate_record(est.x, est.y, r, est.cov_xx, est.cov_xy, est.cov_yy, est.confidence)
            else:
                rti = abstention("implausible_radius")
        return wlm, rti

    def tick(self, t_k: float, truth) -> dict:
        """The §11 tick record `{label, t, wlm, rti, truth}`."""
        wlm, rti = self.estimates(t_k)
        return {"label": self.label, "t": t_k, "wlm": wlm, "rti": rti, "truth": tick_truth(truth)}


# ---------------------------------------------------------------------------
# the Room Sim run (§10.2)
# ---------------------------------------------------------------------------


@dataclass
class LocateRun:
    """The records of one run, and diagnostics (no scene ids).

    `diagnostics`: `label`, `link_signal`, `rate_hz`, `min_samples`,
    `n_links`, `calibrated_mask` (per link, canonical order: False for a
    link whose baseline was imputed), `blind_pixels` (count),
    `n_ticks`, `n_epochs`, `has_phone` (whether the scene has one)."""

    ticks: list = field(default_factory=list)
    epochs: list = field(default_factory=list)
    diagnostics: dict = field(default_factory=dict)


def run_locate(scene, rs_cfg: RoomSimConfig | None = None, cfg: LocateConfig | None = None,
               label: str = "SIM") -> LocateRun:
    """§10.2: calibration run, occupied run (one tick record per §10.1
    tick), and the phone (one epoch record per fired epoch, if the scene
    has a phone). Room Sim input is simulated, so `label` must be "SIM"
    (§1). A `phone_seed`, `tx_power_dbm` or `carrier_hz` set in `cfg` must
    equal Room Sim's (§9)."""
    check_roomsim_label(label)
    rs_cfg = RoomSimConfig() if rs_cfg is None else rs_cfg
    cfg = LocateConfig() if cfg is None else cfg
    cfg.validate()
    check_roomsim_input(scene, rs_cfg)
    sim = RoomSim(scene, rs_cfg)                   # validates the Room Sim config
    if cfg.phone_seed is not None and cfg.phone_seed != rs_cfg.seed:
        raise LocateError("phone_seed", "a set phone_seed must be the Room Sim run's seed")
    rcfg = cfg.resolved(rs_cfg.seed, rs_cfg.tx_power_dbm, roomsim_carrier_hz(rs_cfg.channel))
    check_tick_hz(rcfg.tick_hz, rs_cfg.slot_s)

    geometry = locate_geometry(scene, rcfg)
    rate_hz = base_rate_hz(sim.n_transmitters, rs_cfg.slot_s)
    person = PersonPipeline(geometry, rcfg, rate_hz, label)
    check_tx_period(sim.transmitters, rs_cfg.tx_period, rcfg.window_s, rate_hz, person.meas.min_samples)

    has_phone = scene.phone is not None
    phone = None
    if has_phone:
        phone = PhoneRun(phone_nodes(scene), phone_walls(sim.world), phone_boxes(sim.world), rcfg, label)

    # 1. Calibration on the empty room (same seed: the same room).
    fw = rs_cfg.fw_subcarriers
    duration_s = rs_cfg.duration_s
    calib_sim = RoomSim(scene, calibration_config(rs_cfg, rcfg.calib_s))
    person.calibrate(calibration_stream(calib_sim, duration_s, fw, rate_hz))

    # 2-3. The occupied run: packets into the measurement, ticks between
    # them, and every transmission's truth into the phone.
    run = LocateRun()
    k = 0
    t_k = tick_time(k, rcfg.window_s, rcfg.tick_hz)
    last_truth = None
    radius = rs_cfg.body_radius_m
    for tr in sim.transmissions():
        while t_k < duration_s and tr.t > t_k:
            run.ticks.append(person.tick(t_k, last_truth))
            k, t_k = next_tick(k, t_k, rcfg.window_s, rcfg.tick_hz)
        for pkt in tr.packets:
            person.feed(pkt, fw)
        last_truth = tr.truth
        if phone is not None and tr.truth.phone is not None:
            rec = phone.sample(tr.t, tr.truth.phone, person_cylinders(tr.snapshots, radius))
            if rec is not None:
                run.epochs.append(rec)
    while t_k < duration_s:
        run.ticks.append(person.tick(t_k, last_truth))
        k, t_k = next_tick(k, t_k, rcfg.window_s, rcfg.tick_hz)

    run.diagnostics = {
        "label": label,
        "link_signal": person.signal.name,
        "rate_hz": rate_hz,
        "min_samples": person.meas.min_samples,
        "n_links": geometry.n_links,
        "calibrated_mask": person.meas.calibrated_mask(),
        "blind_pixels": len(person.inverter.blind_pixels),
        "n_ticks": len(run.ticks),
        "n_epochs": len(run.epochs),
        "has_phone": has_phone,
    }
    return run
