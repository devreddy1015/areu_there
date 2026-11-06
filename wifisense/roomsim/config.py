"""Room Sim configuration (spec/roomsim.md §3.2) and its validation.

Every parameter is listed with its default. REAL values come from the
firmware or a published standard; everything marked SIM in §3.2 is a
modelling assumption to be calibrated in Phases 9-10.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass, field

from .scene import MAX_PEOPLE, WALL_MATERIALS

__all__ = ["Config", "ConfigError", "PAIR_FIELDS", "MAP_FIELDS"]

ROLES = ("txrx", "tx", "rx")
MODES = ("walk", "stationary")
POSTURES = ("stand", "sit", "lie")

# Every (lo, hi) pair: validated lo <= hi.
PAIR_FIELDS = (
    "clutter_rcs_m2", "height_m", "breathing_hz", "breathing_p2p_m", "speed_mps", "pause_s",
    "device_gain_db", "timing_offset_dev_s", "noise_floor_dbm", "rssi_range_dbm",
)
# Maps whose keys are node ids, person indices or "tx,rx" strings.
MAP_FIELDS = ("roles", "tx_period", "people_mode", "people_posture", "start_heading_rad", "loss_prob_link")
# Maps whose keys are integers, spelled in JSON as canonical decimal (§3.2).
INT_KEY_MAPS = ("roles", "tx_period", "people_mode", "people_posture", "start_heading_rad")
# Maps whose values are binary64 numbers.
FLOAT_VALUE_MAPS = ("start_heading_rad", "loss_prob_link")


def _canonical_key(a) -> bool:
    """§3.2: ASCII digits only, no sign, spaces, underscores or leading
    zeros (`"0"` is the only key starting with `0`)."""
    return isinstance(a, str) and a.isascii() and a.isdigit() and (a == "0" or a[0] != "0")


def _as_float(param: str, v):
    """A JSON integer given for a binary64 parameter becomes a float, as
    serde reads it; anything else (bools included) is left for validation."""
    if isinstance(v, int) and not isinstance(v, bool):
        try:
            return float(v)
        except OverflowError:
            raise ConfigError(param, f"{v} is outside the binary64 range") from None
    return v


class ConfigError(ValueError):
    """A configuration Room Sim rejects; the message names the parameter."""

    def __init__(self, param: str, msg: str) -> None:
        self.param = param
        super().__init__(f"{param}: {msg}")


@dataclass
class Config:
    """Every §3.2 parameter, with its default."""

    # run
    seed: int = 0
    duration_s: float = 10.0
    # radio
    channel: int = 6
    spacing_hz: float = 312500.0
    tx_power_dbm: float = 10.0
    # TDMA
    slot_s: float = 0.0125
    roles: dict = field(default_factory=dict)
    tx_period: dict = field(default_factory=dict)
    # surfaces
    reflection_order: int = 2
    floor_ceiling: bool = True
    floor_material: str = "concrete"
    ceiling_material: str = "concrete"
    slab_thickness_m: float = 0.2
    transmission: bool = True
    implied_wall_material: str = "drywall"
    implied_wall_thickness_m: float = 0.1
    # objects
    object_specular: bool = True
    object_scatter: bool = True
    object_scatter_scale: float = 1.0
    object_blockage: bool = True
    # clutter
    clutter_count: int = 24
    clutter_rcs_m2: tuple = (0.005, 0.2)
    clutter_margin_m: float = 0.1
    # tail
    tail: bool = True
    tail_count: int = 40
    tail_power_db: float = -15.0
    tail_decay_s: float = 20e-9
    # people
    n_people: int | None = None
    people_mode: dict = field(default_factory=dict)
    people_posture: dict = field(default_factory=dict)
    start_heading_rad: dict = field(default_factory=dict)
    body_points: int = 10
    body_rcs_m2: float = 0.5
    body_radius_m: float = 0.18
    height_m: tuple = (1.55, 1.85)
    breathing_hz: tuple = (0.1, 0.3)
    breathing_p2p_m: tuple = (0.004, 0.012)
    speed_mps: tuple = (0.3, 1.2)
    pause_s: tuple = (0.0, 20.0)
    walk_margin_m: float = 0.4
    object_clearance_m: float = 0.05
    shadowing: bool = True
    shadow_cap_db: float = 20.0
    # phone
    phone_carrier: int = 0
    phone_offset_m: tuple = (0.05, -0.25, 1.0)
    # impairments
    impair: bool = True
    loss_prob: float = 0.02
    loss_prob_link: dict = field(default_factory=dict)
    agc_jump_prob: float = 0.01
    agc_step_db: float = 3.0
    agc_up_prob: float = 0.5
    agc_initial_db: float = 0.0
    agc_max_db: float = 6.0
    device_gain_db: tuple = (-2.0, 2.0)
    cpe: bool = True
    timing_offset_s: float = 50e-9
    timing_offset_dev_s: tuple = (0.0, 0.0)
    ripple: bool = True
    ripple_amp: float = 0.05
    ripple_phase_rad: float = 0.1
    ripple_harmonics: int = 3
    ripple_period_bins: float = 64.0
    snr_noise: bool = True
    chip_noise_counts: float = 0.5
    noise_floor_dbm: tuple = (-97.0, -92.0)
    rssi_jitter_db: float = 1.0
    rssi_range_dbm: tuple = (-100.0, 0.0)
    target_counts: float = 24.0
    ltf_merge: bool = True
    ltf_merge_weight: float = 0.5
    smoothing: bool = False
    smoothing_kernel: tuple = (0.25, 0.5, 0.25)
    smoothing_edge: tuple = (2.0 / 3.0, 1.0 / 3.0)
    # output
    quantize: bool = True
    fw_subcarriers: int = 56

    # -- construction ---------------------------------------------------------

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Build from a JSON-like dict, as `saans-roomsim` reads JSON: lists
        become tuples; integer map keys must be canonical decimal strings
        (or ints); a JSON integer given for a binary64 parameter, pair
        element or map value becomes a float (bools and integer parameters
        are left alone, for validation to judge)."""
        names = {f.name for f in dataclasses.fields(cls)}
        kw = {}
        for k, v in d.items():
            if k not in names:
                raise ConfigError(k, "unknown parameter")
            if k in MAP_FIELDS:
                if not isinstance(v, dict):
                    raise ConfigError(k, "expected a map")
                if k in INT_KEY_MAPS:
                    out = {}
                    for a, b in v.items():
                        if isinstance(a, int) and not isinstance(a, bool) and a >= 0:
                            out[a] = b
                        elif _canonical_key(a):
                            out[int(a)] = b
                        else:
                            raise ConfigError(k, f"key {a!r} is not a canonical decimal integer "
                                                 "(ASCII digits, no sign, spaces, underscores or leading zeros)")
                    v = out
                else:
                    v = {str(a): b for a, b in v.items()}
                if k in FLOAT_VALUE_MAPS:
                    v = {a: _as_float(k, b) for a, b in v.items()}
            elif isinstance(v, list):
                v = tuple(_as_float(k, x) for x in v) if k in _TUPLE_FIELDS else tuple(v)
            elif k in _FLOAT_FIELDS:
                v = _as_float(k, v)
            kw[k] = v
        return cls(**kw)

    def to_dict(self) -> dict:
        """Every value, JSON-ready (maps keyed by strings)."""
        out = {}
        for f in dataclasses.fields(self):
            v = getattr(self, f.name)
            if isinstance(v, dict):
                v = {str(k): v[k] for k in sorted(v, key=lambda x: (str(type(x)), x))}
            elif isinstance(v, tuple):
                v = list(v)
            out[f.name] = v
        return out

    def non_default(self) -> dict:
        """The values that differ from the §3.2 defaults."""
        base = Config()
        return {f.name: getattr(self, f.name) for f in dataclasses.fields(self)
                if getattr(self, f.name) != getattr(base, f.name)}

    # -- validation -------------------------------------------------------------

    def validate(self, scene) -> None:
        """Reject a configuration this scene cannot run (§3.2 Validation,
        plus the checks that keep every formula defined)."""
        node_ids = [n.id for n in scene.nodes]

        def need(cond: bool, param: str, msg: str) -> None:
            if not cond:
                raise ConfigError(param, msg)

        def is_int(v) -> bool:
            return isinstance(v, int) and not isinstance(v, bool)

        def num(param: str, v) -> float:
            need(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v),
                 param, f"{v!r} is not a finite number")
            return float(v)

        def within(param: str, v, lo: float, hi: float) -> None:
            x = num(param, v)
            need(lo <= x <= hi, param, f"{x!r} is not in {lo:g}..{hi:g}")

        need(is_int(self.seed) and 0 <= self.seed < 1 << 64, "seed", "must be an integer in 0..2^64-1")
        need(num("duration_s", self.duration_s) >= 0.0, "duration_s", "must be >= 0")
        need(is_int(self.channel) and 1 <= self.channel <= 13, "channel", f"{self.channel!r} is not in 1..13")
        need(num("spacing_hz", self.spacing_hz) > 0.0, "spacing_hz", "must be > 0")
        within("tx_power_dbm", self.tx_power_dbm, -100.0, 60.0)
        need(num("slot_s", self.slot_s) > 0.0, "slot_s", "must be > 0")
        for nid in node_ids:
            need(0 <= nid <= 254, "nodes", f"node id {nid} is not in 0..254")
        for p in INT_KEY_MAPS:
            for k in getattr(self, p):
                need(is_int(k) and k >= 0, p, f"key {k!r} is not a non-negative integer")
        for k, v in self.roles.items():
            need(k in node_ids, "roles", f"key {k!r} is not a node id of this scene")
            need(v in ROLES, "roles", f"node {k}: {v!r} is not one of {', '.join(ROLES)}")
        for k, v in self.tx_period.items():
            need(k in node_ids, "tx_period", f"key {k!r} is not a node id of this scene")
            need(is_int(v) and v >= 1, "tx_period", f"node {k}: {v!r} is not an integer >= 1")
        role = {n: self.roles.get(n, "txrx") for n in node_ids}
        txs = [n for n in node_ids if role[n] in ("tx", "txrx")]
        rxs = [n for n in node_ids if role[n] in ("rx", "txrx")]
        need(any(a != b for a in txs for b in rxs), "roles",
             "no link: a run needs a transmitter and a different receiver (a one-node scene has none)")
        need(self.reflection_order in (0, 1, 2) and is_int(self.reflection_order),
             "reflection_order", "must be 0, 1 or 2")
        # Slab surfaces take the scene's wall materials; `metal` is for objects only (§5.1).
        for p in ("floor_material", "ceiling_material", "implied_wall_material"):
            need(getattr(self, p) in WALL_MATERIALS, p,
                 f"{getattr(self, p)!r} is not one of {', '.join(WALL_MATERIALS)}")
        need(num("slab_thickness_m", self.slab_thickness_m) > 0.0, "slab_thickness_m", "must be > 0")
        need(num("implied_wall_thickness_m", self.implied_wall_thickness_m) > 0.0,
             "implied_wall_thickness_m", "must be > 0")
        need(num("object_scatter_scale", self.object_scatter_scale) >= 0.0, "object_scatter_scale", "must be >= 0")
        need(is_int(self.clutter_count) and self.clutter_count >= 0, "clutter_count", "must be an integer >= 0")
        need(num("clutter_margin_m", self.clutter_margin_m) >= 0.0, "clutter_margin_m", "must be >= 0")
        need(is_int(self.tail_count) and self.tail_count >= 0, "tail_count", "must be an integer >= 0")
        within("tail_power_db", self.tail_power_db, -100.0, 30.0)
        need(num("tail_decay_s", self.tail_decay_s) > 0.0, "tail_decay_s", "must be > 0")

        for p in PAIR_FIELDS:
            v = getattr(self, p)
            need(isinstance(v, tuple) and len(v) == 2, p, "must be a (lo, hi) pair")
            lo, hi = num(p, v[0]), num(p, v[1])
            need(lo <= hi, p, f"lo {lo} > hi {hi}")
        need(self.clutter_rcs_m2[0] > 0.0, "clutter_rcs_m2", "must be > 0 (log-uniform)")
        need(self.speed_mps[0] > 0.0, "speed_mps", "must be > 0")
        need(self.pause_s[0] >= 0.0, "pause_s", "must be >= 0")
        for v in self.device_gain_db:
            within("device_gain_db", v, -60.0, 60.0)
        nf_lo, nf_hi = self.noise_floor_dbm
        need(-127.0 <= nf_lo and nf_hi <= -1.0, "noise_floor_dbm", "must lie within -127..-1")
        r_lo, r_hi = self.rssi_range_dbm
        need(float(r_lo).is_integer() and float(r_hi).is_integer() and -128 <= r_lo and r_hi <= 127,
             "rssi_range_dbm", "must be integers within -128..127 (an int8 on the wire)")

        n_scene = len(scene.people)
        if self.n_people is not None:
            need(is_int(self.n_people) and 0 <= self.n_people <= MAX_PEOPLE,
                 "n_people", f"{self.n_people!r} is not an integer in 0..{MAX_PEOPLE}")
        n = n_scene if self.n_people is None else self.n_people
        need(n <= MAX_PEOPLE, "n_people", f"{n} people: at most {MAX_PEOPLE}")
        for p, allowed in (("people_mode", MODES), ("people_posture", POSTURES)):
            for k, v in getattr(self, p).items():
                need(0 <= k < n, p, f"index {k} is not a simulated person (0..{n - 1})")
                need(v in allowed, p, f"people[{k}]: {v!r} is not one of {', '.join(allowed)}")
        for k, v in self.start_heading_rad.items():
            need(0 <= k < n, "start_heading_rad", f"index {k} is not a simulated person")
            num("start_heading_rad", v)
        need(is_int(self.body_points) and 6 <= self.body_points <= 16, "body_points",
             f"{self.body_points!r} is not in 6..16")
        need(num("body_rcs_m2", self.body_rcs_m2) >= 0.0, "body_rcs_m2", "must be >= 0")
        need(num("body_radius_m", self.body_radius_m) >= 0.0, "body_radius_m", "must be >= 0")
        need(num("walk_margin_m", self.walk_margin_m) >= 0.0, "walk_margin_m", "must be >= 0")
        num("object_clearance_m", self.object_clearance_m)
        need(num("shadow_cap_db", self.shadow_cap_db) >= 0.0, "shadow_cap_db", "must be >= 0")
        need(is_int(self.phone_carrier) and self.phone_carrier >= 0, "phone_carrier", "must be an integer >= 0")
        need(isinstance(self.phone_offset_m, tuple) and len(self.phone_offset_m) == 3,
             "phone_offset_m", "must be (x, y, z)")
        for v in self.phone_offset_m:
            num("phone_offset_m", v)

        for p in ("loss_prob", "agc_jump_prob", "agc_up_prob", "ltf_merge_weight"):
            v = num(p, getattr(self, p))
            need(0.0 <= v <= 1.0, p, f"{v} is not a probability in 0..1")
        for k, v in self.loss_prob_link.items():
            # Exactly f"{tx},{rx}" (the key §12.2 looks up): no spaces, signs or leading zeros.
            parts = str(k).split(",")
            ok = len(parts) == 2 and all(s.isascii() and s.isdigit() and str(int(s)) == s for s in parts)
            need(ok, "loss_prob_link", f"key {k!r} is not \"tx,rx\"")
            need(int(parts[0]) in node_ids and int(parts[1]) in node_ids, "loss_prob_link",
                 f"key {k!r} names a node this scene does not have")
            v = num("loss_prob_link", v)
            need(0.0 <= v <= 1.0, "loss_prob_link", f"{k}: {v} is not a probability")
        # Decibel and scale parameters stay where every formula is finite (§3.2).
        within("agc_step_db", self.agc_step_db, 0.0, 60.0)
        within("agc_max_db", self.agc_max_db, 0.0, 60.0)
        within("agc_initial_db", self.agc_initial_db, -60.0, 60.0)
        within("rssi_jitter_db", self.rssi_jitter_db, 0.0, 60.0)
        within("ripple_amp", self.ripple_amp, 0.0, 1.0)
        need(num("ripple_phase_rad", self.ripple_phase_rad) >= 0.0, "ripple_phase_rad", "must be >= 0")
        within("timing_offset_s", self.timing_offset_s, 0.0, 1e-3)
        for v in self.timing_offset_dev_s:
            within("timing_offset_dev_s", v, -1e-3, 1e-3)
        within("chip_noise_counts", self.chip_noise_counts, 0.0, 1000.0)
        within("target_counts", self.target_counts, 0.0, 1000.0)
        need(self.target_counts > 0.0, "target_counts", "must be > 0")
        for p, hi in (("breathing_p2p_m", 1.0), ("height_m", 3.0)):
            for v in getattr(self, p):
                within(p, v, 0.0, hi)
        need(is_int(self.ripple_harmonics) and self.ripple_harmonics >= 0, "ripple_harmonics",
             "must be an integer >= 0")
        need(abs(num("ripple_period_bins", self.ripple_period_bins)) >= 1.0, "ripple_period_bins",
             "|ripple_period_bins| must be >= 1 (it is a number of bins)")
        for p, n_ in (("smoothing_kernel", 3), ("smoothing_edge", 2)):
            v = getattr(self, p)
            need(isinstance(v, tuple) and len(v) == n_, p, f"must have {n_} taps")
            for x in v:
                num(p, x)
        need(is_int(self.fw_subcarriers) and 8 <= self.fw_subcarriers <= 64, "fw_subcarriers",
             f"{self.fw_subcarriers!r} is not in 8..64")
        for p in ("floor_ceiling", "transmission", "object_specular", "object_scatter", "object_blockage",
                  "tail", "shadowing", "impair", "cpe", "ripple", "snr_noise", "ltf_merge", "smoothing",
                  "quantize"):
            need(isinstance(getattr(self, p), bool), p, "must be true or false")


# binary64 parameters and tuples (every tuple element is binary64, as in
# saans-roomsim's Config): from_dict turns a JSON integer into a float.
_FLOAT_FIELDS = frozenset(f.name for f in dataclasses.fields(Config) if f.type == "float")
_TUPLE_FIELDS = frozenset(f.name for f in dataclasses.fields(Config) if f.type == "tuple")
