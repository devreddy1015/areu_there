"""Room Sim runs for calibration (spec/calibration.md §3): a parameter vector
applied to dataset rooms' stored configs, run with the Rust CLI
(`saans roomsim-export`, bit-identical to the Python reference), read back as
SIM windows through the shared reader and turned into calibration features.

Each (room, state) is one 30 s run:
- `empty`: `n_people = 0`;
- `still`: one person, `people_mode {0: stationary}`;
- `moving`: one person, `people_mode {0: walk}`.

Windows are 2 s long, ending every 2 s from 4 s to 30 s. Nothing here
changes Room Sim: every parameter is an existing config field
(spec/roomsim.md §3.2), and its defaults are the dataset's.
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import shutil
import subprocess
import tempfile

import numpy as np

from ml import preprocess as pp
from ml import roomsim_dataset as rd
from ml.calibrate import features as cf

__all__ = ["PARAMS", "NAMES", "DEFAULT_THETA", "SAANS", "DATASET", "to_theta", "to_unit", "apply", "rooms",
           "run_one", "simulate"]

SAANS = "/home/Aman/are_u_theone/are_U_there/rust/target/release/saans"
DATASET = "/home/Aman/are_u_theone/datasets/roomsim-v1"
SECONDS = 30
LENGTH_S = 2.0
#: (name, default, lo, hi, scale): the calibrated parameters (spec/calibration.md §3).
PARAMS = (
    ("tx_power_dbm", 10.0, -5.0, 20.0, "lin"),
    ("noise_shift_db", 0.0, -10.0, 20.0, "lin"),        # added to both ends of noise_floor_dbm
    ("chip_noise_counts", 0.5, 0.05, 4.0, "log"),
    ("tail_power_db", -15.0, -30.0, -3.0, "lin"),
    ("clutter_scale", 1.0, 0.1, 10.0, "log"),           # multiplies both ends of clutter_rcs_m2
    ("body_rcs_m2", 0.5, 0.05, 10.0, "log"),
    ("breathing_scale", 1.0, 0.5, 4.0, "log"),          # multiplies both ends of breathing_p2p_m
    ("rssi_jitter_db", 1.0, 0.2, 4.0, "log"),
    ("ripple_amp", 0.05, 0.0, 0.3, "lin"),
)
NAMES = tuple(p[0] for p in PARAMS)
DEFAULT_THETA = {p[0]: p[1] for p in PARAMS}


def to_theta(u) -> dict:
    """[0, 1]^P -> parameter values (log-uniform where the scale is log)."""
    u = np.clip(np.asarray(u, np.float64), 0.0, 1.0)
    out = {}
    for (name, _d, lo, hi, scale), x in zip(PARAMS, u):
        out[name] = float(math.exp(math.log(lo) + x * (math.log(hi) - math.log(lo)))) if scale == "log" \
            else float(lo + x * (hi - lo))
    return out


def to_unit(theta: dict) -> np.ndarray:
    out = []
    for name, _d, lo, hi, scale in PARAMS:
        v = float(theta[name])
        out.append((math.log(v) - math.log(lo)) / (math.log(hi) - math.log(lo)) if scale == "log"
                   else (v - lo) / (hi - lo))
    return np.asarray(out)


def apply(cfg: dict, theta: dict, state: str) -> dict:
    """A copy of a stored Room Sim config with `theta` and the state's people."""
    c = json.loads(json.dumps(cfg))
    c["duration_s"] = float(SECONDS)
    c["tx_power_dbm"] = float(theta["tx_power_dbm"])
    c["noise_floor_dbm"] = [float(v) + float(theta["noise_shift_db"]) for v in cfg["noise_floor_dbm"]]
    c["chip_noise_counts"] = float(theta["chip_noise_counts"])
    c["tail_power_db"] = float(theta["tail_power_db"])
    c["clutter_rcs_m2"] = [float(v) * float(theta["clutter_scale"]) for v in cfg["clutter_rcs_m2"]]
    c["body_rcs_m2"] = float(theta["body_rcs_m2"])
    c["breathing_p2p_m"] = [float(v) * float(theta["breathing_scale"]) for v in cfg["breathing_p2p_m"]]
    c["rssi_jitter_db"] = float(theta["rssi_jitter_db"])
    c["ripple_amp"] = float(theta["ripple_amp"])
    c["people_posture"] = {}
    c["start_heading_rad"] = {}
    if state == "empty":
        c["n_people"], c["people_mode"] = 0, {}
    elif state == "still":
        c["n_people"], c["people_mode"] = 1, {"0": "stationary"}
    elif state == "moving":
        c["n_people"], c["people_mode"] = 1, {"0": "walk"}
    else:
        raise ValueError(state)
    return c


def rooms(n: int, *, seed: int, min_nodes: int = 6, max_nodes: int = 10, exclude=()) -> list:
    """`n` train-split dataset rooms with `min_nodes..max_nodes` nodes, drawn with `seed` (SIM)."""
    m = rd.load_manifest(DATASET)
    ent = {int(e["index"]): e for e in m["scenes"]}
    pool = [i for i in rd.splits(m)["train"] if min_nodes <= int(ent[i]["n_nodes"]) <= max_nodes
            and i not in set(exclude)]
    rng = np.random.default_rng(seed)
    return sorted(int(i) for i in rng.choice(pool, size=n, replace=False))


def _scene_paths(index: int) -> tuple:
    sid = rd.scene_id(index) if hasattr(rd, "scene_id") else f"gen-{index:06d}"
    base = os.path.join(DATASET, "scenes", sid)
    return base + ".json", base + ".roomsim.json"


def run_one(task: tuple) -> dict:
    """One (theta, room, state) run -> `{"features": {group: values}, "ok": bool, "error": str}`."""
    theta, index, state, workdir = task
    scene, cfg_path = _scene_paths(index)
    with open(cfg_path, encoding="utf-8") as fh:
        cfg = apply(json.load(fh), theta, state)
    d = tempfile.mkdtemp(prefix="cal-", dir=workdir)
    try:
        cp = os.path.join(d, "cfg.json")
        with open(cp, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh)
        prefix = os.path.join(d, "run")
        r = subprocess.run([SAANS, "roomsim-export", scene, "--out", prefix, "--config", cp, "--overwrite"],
                           capture_output=True, text=True)
        if r.returncode != 0:
            return {"features": {}, "ok": False, "error": (r.stderr or r.stdout).strip()[-300:]}
        ends = [pp.sim_end_ns(int(2 * t) - 1) for t in range(4, SECONDS + 1, 2)]
        wins = pp.from_pcap(prefix + ".sim.pcap", LENGTH_S, ends, link_map=pp.truth_link_map(prefix + ".truth.jsonl"))
        parts = []
        for w in wins:
            if w.label != "SIM":
                raise AssertionError("a Room Sim run read back as not SIM")
            parts.append(cf.window_features(w, state))
        return {"features": cf.merge(parts), "ok": True, "error": ""}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def simulate(thetas: list, room_list: list, *, workers: int = 40, workdir: str = "/dev/shm") -> list:
    """For each theta: the merged features of every (room, state) run and the failures."""
    tasks = [(th, i, s, workdir) for th in thetas for i in room_list for s in cf.STATES]
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        res = pool.map(run_one, tasks, chunksize=1)
    out = []
    per = len(room_list) * len(cf.STATES)
    for j in range(len(thetas)):
        chunk = res[j * per:(j + 1) * per]
        out.append({"features": cf.merge([r["features"] for r in chunk if r["ok"]]),
                    "failed": [(room_list[q // len(cf.STATES)], cf.STATES[q % len(cf.STATES)], r["error"])
                               for q, r in enumerate(chunk) if not r["ok"]]})
    return out
