"""Phase 8 baselines (spec/model.md §3), each returning the predictions
contract of `ml.train.metrics` so that every method is scored by the same
code. Everything is **SIM**.

1. **Stored WLM and RTI** (`stored_predictions`): Locate v1's outcome at
   the window's end tick as the dataset stores it (spec/dataset.md §6: 1 s
   of RSSI, Locate's defaults, the idealised empty-room calibration).
   Presence = published; xy = the published centre with its radius
   (Locate's uncalibrated model); abstentions are `published = False`.
2. **Recomputed WLM and RTI** (`recompute_scene`, `recompute_dataset`,
   `recomputed_predictions`): the Python reference `wifisense.localize`
   run on the shard's stored RSSI streams (`csi_mask`, `rssi` on the TDMA
   grid, spec/dataset.md §4) after calibrating on the stored empty-room
   recording (`calib_*`), with `window_s` = the compared model's `ℓ`, at
   every label tick, on all links and on the link subsets of
   spec/model.md §5.6 ("tx0": links with `tx == 0`; "single01": the link
   (0, 1)). The glue is `run_locate`'s exactly (spec/locate.md §4.3, §10):
   received packets in Room Sim's order (transmission `n`, then receiver),
   `t = float(n) * slot_s`, the RSSI signal's validity gate (a frame with
   every int8 entry zero is dropped), ticks at `window_s + j / tick_hz`,
   and the scene file's float64 geometry. With `window_s = 1` and every
   link it reproduces the stored outcome bit for bit at every tick but the
   last (`t_k = S`: Locate also saw the transmission at exactly `t = S`,
   which the dataset does not store; spec/dataset.md §6) — tested.
   A link subset runs Locate on a geometry holding only those links
   (`subset_geometry`): the measurement, calibration, inverter and tracker
   see only them. The geometry is still known to WLM and RTI, so
   "single01_nogeo" scores the "single01" outcome.
   **Deviation, flagged:** the stored calibration recording is 15 s
   (`calib_s`), which holds 14 non-overlapping 1 s windows but only 6-7
   at 2 s and 2-3 at 4 s, fewer than Locate's `min_windows = 10`; so at
   `window_s = ℓ` the calibration asks for `min(10, floor(calib_s / ℓ) -
   1)` windows (10, 6, 2 at ℓ = 1, 2, 4) and its null sigma is noisier at
   longer windows.
3. **Variance features + random forest and MLP** (`window_features`,
   `link_variance_maps`, `fit_variance_baselines`): per window, from the
   cached preprocessing, each used link's spread `s_l` = mean of `sigma`
   over the 44 reference bins; features (44) = `s_l`'s mean, std
   (population), min, max, quartiles and 90th percentile over the used
   links; the node count; the room `w, d, h`; the 4 x 4 link-variance map
   in room-normalised coordinates (cell `(i, j)`, `i` along x and `j` along
   y, centre `((i + 0.5) / 4, (j + 0.5) / 4)`, the mean `s_l` over the
   used links whose normalised 2-D segment passes within 1/8 of the centre,
   flattened `4 i + j`) and the 4 x 4 map of those links' count (both 0
   where none). Trained on decision 5's windows of the train split: a
   scikit-learn random forest (300 trees, defaults otherwise) and an MLP
   (two hidden layers of 128, ReLU, Adam 1e-3, batch 512, early stopping
   on the val windows' loss, patience 5, at most 60 epochs) for presence,
   count, zone (per cell; the MLP's loss over reachable cells) and, on
   one-person windows, xy (room-normalised targets): the forest's radius
   is the 95th percentile of its trees' distances (m) to the forest's
   mean, the MLP predicts an isotropic Gaussian like the model (`sigma =
   0.1225 + softplus` m, radius `max(2.448 sigma, 0.3)`); both floored at
   0.3 m. Seeds fix the forest's and the MLP's randomness (3 seeds).
4. **Prior** (`prior_predictions`): the train split's class frequencies
   (presence, count, each zone cell: the decisions are the majority
   classes; AUC 0.5); xy the footprint's centroid (spec/locate.md §13's
   reference point) with radius the 95th percentile of its training errors
   on one-person train windows.

WLM and RTI use the idealised calibration the model is denied (spec/model.md
decision 4): their predictions carry `uses_calibration: True`.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import multiprocessing
import os
import sys
import time

import numpy as np

from ml import roomsim_dataset as rd
from ml.train import cache as cache_mod
from ml.train import metrics

__all__ = [
    "LABEL", "SUBSETS", "REASON_CODES", "N_FEATURES", "FEATURE_NAMES", "stored_predictions", "calib_min_windows",
    "subset_geometry", "locate_inputs", "recompute_scene", "recompute_dataset", "load_recomputed",
    "recomputed_predictions", "segment_distance", "link_variance_maps", "scene_incidence", "window_features",
    "fit_variance_baselines", "footprint_centroid", "prior_predictions", "baseline_methods", "save_predictions",
    "load_predictions",
]

LABEL = "SIM"
#: spec/model.md §5.6's link subsets for the recomputed Locate ("single01_nogeo" scores "single01").
SUBSETS = ("all", "tx0", "single01")
REASON_CODES = {"estimate": 0, "no_disturbance": 1, "not_significant": 2, "implausible_radius": 3,
                "warming_up": 4}
GRID = 4
FEATURE_NAMES = (["s_mean", "s_std", "s_min", "s_max", "s_q25", "s_q50", "s_q75", "s_q90", "n_nodes",
                  "room_w", "room_d", "room_h"]
                 + [f"var_map_{i}{j}" for i in range(GRID) for j in range(GRID)]
                 + [f"count_map_{i}{j}" for i in range(GRID) for j in range(GRID)])
N_FEATURES = len(FEATURE_NAMES)


# --------------------------------------------------------------------------
# 1. stored WLM / RTI
# --------------------------------------------------------------------------


def stored_predictions(c, truth: dict, which: str = "wlm") -> dict:
    """The stored WLM or RTI outcome (spec/dataset.md §6) at `truth`'s
    windows of cache `c`, as a one-estimate predictions dict."""
    if which not in ("wlm", "rti"):
        raise ValueError(f"which {which!r} is not 'wlm' or 'rti'")
    w = truth["window"]
    win = c.windows
    pub = win[f"{which}_reason"][w] == 0
    xy = np.where(pub[:, None], win[f"{which}_xy"][w].astype(np.float64), 0.0)
    rad = np.where(pub, win[f"{which}_radius"][w].astype(np.float64), metrics.MIN_RADIUS_M)
    return metrics.one_estimate_predictions(
        win["scene"][w], win["k"][w], xy, rad, pub, presence_prob=pub.astype(np.float64),
        name=f"{which}_stored", uses_calibration=True,
        note=f"stored Locate {which.upper()} at the end tick: 1 s of RSSI, defaults, idealised calibration")


# --------------------------------------------------------------------------
# 2. recomputed WLM / RTI (wifisense.localize)
# --------------------------------------------------------------------------


def calib_min_windows(calib_s: float, window_s: float, default: int = 10) -> int:
    """`min(default, floor(calib_s / window_s) - 1)`, at least 1: the
    non-overlapping windows the stored calibration recording surely holds
    (module docstring's deviation)."""
    return max(1, min(int(default), int(math.floor(calib_s / window_s + 1e-9)) - 1))


def subset_geometry(full, keep: set):
    """A copy of a localize `Geometry` whose links are only `keep` (set of
    `(tx, rx)`), in canonical order; nodes, grid and lambda unchanged."""
    from wifisense.localize.geometry import Geometry
    g = Geometry(full.room, full.nodes, full.grid, full.lambda_m)
    g.links = [lk for lk in g.links if (lk.tx_id, lk.rx_id) in keep]
    g._link_index = {(lk.tx_id, lk.rx_id): i for i, lk in enumerate(g.links)}
    if not g.links:
        raise ValueError("a link subset with no link")
    return g


class _Frame:
    """The part of a `CsiFrame` `RssiSignal` reads: `_frame_power`'s
    validity gate (some non-zero entry among the 56: one unit amplitude
    stands for "non-zero", exactly equivalent for finite int8 entries) and
    the RSSI."""

    __slots__ = ("amplitudes", "n_subcarriers", "rssi_dbm")

    def __init__(self, nonzero: bool, rssi: int):
        self.amplitudes = [1.0] if nonzero else [0.0]
        self.n_subcarriers = 1
        self.rssi_dbm = rssi


def _packets(scene: dict, mask, rssi, iq, m0: int, slot_s: float):
    """Received packets in Room Sim's order (transmission n, then receiver):
    `(n, t, tx, rx, rssi, nonzero)` arrays (spec/dataset.md §4.1)."""
    n_tx = int(scene["meta"]["n_tx"])
    links = scene["links"].astype(np.int64)
    slot = scene["tx_slot"].astype(np.int64)
    li, col = np.nonzero(np.asarray(mask) != 0)
    n = (col + int(m0)) * n_tx + slot[li]
    order = np.lexsort((li, n))
    li, col, n = li[order], col[order], n[order]
    nonzero = (np.asarray(iq)[li, :, col, :] != 0).any(axis=(1, 2))
    t = n.astype(np.float64) * slot_s
    return n, t, links[li, 0], links[li, 1], np.asarray(rssi)[li, col].astype(np.int64), nonzero


def locate_inputs(dataset_dir: str, index: int, manifest: dict | None = None) -> dict:
    """What recomputing Locate on SIM scene `index` needs: the shard, the
    scene file (float64 geometry, parsed by `wifisense.roomsim`), Locate's
    resolved config and Room Sim's `slot_s` and `duration_s`, the occupied
    and calibration packet streams."""
    from wifisense.localize.config import LocateConfig
    from wifisense.roomsim import load_scene as rs_load_scene
    scene = rd.load_scene(dataset_dir, index, manifest=manifest)
    sid = scene["meta"]["scene_id"]
    base = os.path.join(dataset_dir, "scenes", sid)
    rd.load_scene_json(dataset_dir, index, manifest=manifest)          # sha256 and provenance checked
    with open(base + ".locate.json", encoding="utf-8") as f:
        cfg = LocateConfig(**json.load(f))
    with open(base + ".roomsim.json", encoding="utf-8") as f:
        rs = json.load(f)
    slot_s, duration_s = float(rs["slot_s"]), float(rs["duration_s"])
    occupied = _packets(scene, scene["csi_mask"], scene["rssi"], scene["csi_iq"], 0, slot_s)
    calib = _packets(scene, scene["calib_mask_m"], scene["calib_rssi"], scene["calib_iq"],
                     int(scene["meta"]["calib_m0"]), slot_s)
    keep = calib[1] >= duration_s                       # spec/locate.md §10.2: t >= duration_s
    calib = tuple(a[keep] for a in calib)
    if rs.get("roles"):
        raise ValueError(f"scene {index}: Room Sim roles are set; the person pipeline needs every node txrx")
    tx_period = {int(k): int(v) for k, v in (rs.get("tx_period") or {}).items()}
    return {"scene": scene, "scene_obj": rs_load_scene(base + ".json"), "cfg": cfg, "slot_s": slot_s,
            "duration_s": duration_s, "tx_period": tx_period, "occupied": occupied, "calib": calib}


def _run_locate(inp: dict, geometry, inverter, window_s: float, min_windows: int, keep: set | None) -> dict:
    """Locate's person pipeline (spec/locate.md §10) on the stored streams:
    per label tick `k` the WLM and RTI reason code, centre and radius."""
    from wifisense.localize.pipeline import (PersonPipeline, base_rate_hz, check_tick_hz, check_tx_period,
                                             make_signal, next_tick, tick_time)
    from wifisense.localize.measure import LinkMeasurement
    from wifisense.localize.track import Tracker
    scene = inp["scene"]
    cfg = dataclasses.replace(inp["cfg"], window_s=float(window_s), min_windows=int(min_windows)).validate()
    slot_s, duration_s = inp["slot_s"], inp["duration_s"]
    check_tick_hz(cfg.tick_hz, slot_s)
    n_tx = int(scene["meta"]["n_tx"])
    rate = base_rate_hz(n_tx, slot_s)
    person = PersonPipeline.__new__(PersonPipeline)       # PersonPipeline with a shared, prebuilt inverter
    person.label = "SIM"
    person.cfg = cfg
    person.geometry = geometry
    person.rate_hz = rate
    person.signal = make_signal(cfg.link_signal)
    person.meas = LinkMeasurement(geometry, window_s=cfg.window_s, sample_rate_hz=rate, signal=person.signal)
    person.inverter = inverter
    person.tracker = Tracker(geometry, z_threshold=cfg.z_threshold)
    check_tx_period(range(n_tx), inp["tx_period"], cfg.window_s, rate, person.meas.min_samples)

    def select(stream):
        if keep is None:
            return stream
        sel = np.array([(int(a), int(b)) in keep for a, b in zip(stream[2], stream[3])], bool)
        return tuple(x[sel] for x in stream)
    cn, ct, ctx, crx, crs, cnz = select(inp["calib"])
    person.calibrate((float(t), int(a), int(b), _Frame(bool(z), int(r)))
                     for t, a, b, z, r in zip(ct, ctx, crx, cnz, crs))
    on, ot, otx, orx, ors, onz = select(inp["occupied"])
    n_ticks = len(scene["t_label"])
    out = {f"{e}_{f}": v for e in ("wlm", "rti") for f, v in
           (("reason", np.full(n_ticks, REASON_CODES["warming_up"], np.uint8)),
            ("xy", np.full((n_ticks, 2), np.nan)), ("radius", np.full(n_ticks, np.nan)))}
    first = int(round(2 * window_s)) - 1                 # label tick of Locate's tick j = 0 (t = window_s)
    meas_update = person.meas.update
    k = 0
    t_k = tick_time(0, cfg.window_s, cfg.tick_hz)
    p, total = 0, on.size

    def record(j, t_tick):
        wlm, rti = person.estimates(t_tick)
        kk = first + j
        if not 0 <= kk < n_ticks or abs(0.5 * (kk + 1) - t_tick) > 1e-9:
            raise AssertionError(f"Locate tick {j} at t = {t_tick} is not label tick {kk}")
        for name, r in (("wlm", wlm), ("rti", rti)):
            if "x" in r:
                out[f"{name}_reason"][kk] = 0
                out[f"{name}_xy"][kk] = (r["x"], r["y"])
                out[f"{name}_radius"][kk] = r["radius_m"]
            else:
                out[f"{name}_reason"][kk] = REASON_CODES[r["reason"]]
    while p < total:
        n0, t0 = on[p], ot[p]
        while t_k < duration_s and t0 > t_k:
            record(k, t_k)
            k, t_k = next_tick(k, t_k, cfg.window_s, cfg.tick_hz)
        q = p
        while q < total and on[q] == n0:
            meas_update(float(ot[q]), int(otx[q]), int(orx[q]), _Frame(bool(onz[q]), int(ors[q])))
            q += 1
        p = q
    while t_k < duration_s:
        record(k, t_k)
        k, t_k = next_tick(k, t_k, cfg.window_s, cfg.tick_hz)
    out["calibrated_links"] = int(sum(person.meas.calibrated_mask()))
    out["n_links"] = geometry.n_links
    out["min_windows"] = int(min_windows)
    return out


def _subset_keep(scene: dict, subset: str) -> set | None:
    links = [tuple(int(v) for v in lk) for lk in scene["links"]]
    if subset == "all":
        return None
    if subset == "tx0":
        return {lk for lk in links if lk[0] == 0}
    if subset == "single01":
        return {(0, 1)}
    raise ValueError(f"subset {subset!r} is not one of {SUBSETS}")


def recompute_scene(dataset_dir: str, index: int, lengths=(1.0, 2.0, 4.0), subsets=SUBSETS, *,
                    manifest: dict | None = None, inputs: dict | None = None) -> dict:
    """Locate (WLM and RTI) recomputed on SIM scene `index` for every window
    length in `lengths` and link subset in `subsets`: `{(length, subset):
    {"wlm_reason" [K] u8, "wlm_xy" [K, 2], "wlm_radius" [K], "rti_*",
    "min_windows", ...}}` over every label tick `k` (code 4, warming up,
    before Locate's first tick at `t = ℓ`)."""
    from wifisense.localize.invert import Inverter
    from wifisense.localize.pipeline import locate_geometry
    inp = locate_inputs(dataset_dir, index, manifest) if inputs is None else inputs
    cfg = inp["cfg"]
    full = locate_geometry(inp["scene_obj"], cfg)
    res = {}
    for subset in subsets:
        keep = _subset_keep(inp["scene"], subset)
        geom = full if keep is None else subset_geometry(full, keep)
        inverter = Inverter(geom, cfg.alpha, cfg.sigma, cfg.delta_c, warn_blind_pixels=False)
        for length in lengths:
            mw = calib_min_windows(cfg.calib_s, float(length), cfg.min_windows)
            res[(float(length), subset)] = _run_locate(inp, geom, inverter, float(length), mw, keep)
    return res


_RC: dict = {}


def _recompute_task(index: int):
    try:
        return index, recompute_scene(_RC["dir"], index, _RC["lengths"], _RC["subsets"], manifest=_RC["manifest"]), None
    except Exception as exc:                       # reported, never silently dropped
        return index, None, f"{type(exc).__name__}: {exc}"


def recompute_dataset(dataset_dir: str, scenes, lengths=(1.0, 2.0, 4.0), subsets=SUBSETS, *, workers: int = 90,
                      out_path: str | None = None, progress: bool = True) -> dict:
    """`recompute_scene` over `scenes` in parallel (fork). Returns (and with
    `out_path` saves as .npz) `{"scenes" [S], "lengths", "subsets",
    "<length>|<subset>|<field>": [S, K, ...]}` plus the label and the
    deviation note."""
    manifest = rd.load_manifest(dataset_dir)
    scenes = [int(s) for s in scenes]
    _RC.update(dir=dataset_dir, lengths=tuple(float(x) for x in lengths), subsets=tuple(subsets), manifest=manifest)
    t0 = time.time()
    results = {}
    errors = {}
    ctx = multiprocessing.get_context("fork")
    with ctx.Pool(max(1, int(workers))) as pool:
        for i, (idx, res, err) in enumerate(pool.imap_unordered(_recompute_task, scenes, chunksize=1)):
            if err is not None:
                errors[idx] = err
            else:
                results[idx] = res
            if progress and ((i + 1) % max(1, len(scenes) // 10) == 0 or i + 1 == len(scenes)):
                print(f"[recompute SIM] {i + 1}/{len(scenes)} scenes, {time.time() - t0:.0f} s", file=sys.stderr,
                      flush=True)
    if errors:
        raise RuntimeError(f"recompute failed on {len(errors)} scenes: {dict(list(errors.items())[:3])}")
    out = {"label": np.array(LABEL), "scenes": np.array(scenes, np.int32),
           "lengths": np.array(sorted(set(float(x) for x in lengths))), "subsets": np.array(list(subsets)),
           "dataset": np.array(os.path.basename(os.path.normpath(os.path.abspath(dataset_dir)))),
           "impairments": np.array(rd.impairments(manifest)),
           "note": np.array("Locate (wifisense.localize) recomputed on the stored RSSI streams and calibration; "
                            "min_windows = min(10, floor(calib_s / window_s) - 1); SIM")}
    for key in results[scenes[0]]:
        length, subset = key
        for field in results[scenes[0]][key]:
            vals = [results[s][key][field] for s in scenes]
            out[f"{length:g}|{subset}|{field}"] = np.stack([np.asarray(v) for v in vals])
    out["wall_s"] = np.array(time.time() - t0)
    if out_path:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
        np.savez(out_path, **out)
    return out


def load_recomputed(path: str) -> dict:
    """A `recompute_dataset` .npz as a dict (refuses one not labelled SIM)."""
    with np.load(path) as z:
        out = {k: z[k] for k in z.files}
    if str(out["label"]) != LABEL:
        raise ValueError(f"{path}: not labelled SIM")
    return out


def recomputed_predictions(rec: dict, truth: dict, length: float, subset: str = "all", which: str = "wlm") -> dict:
    """The recomputed WLM or RTI at `truth`'s windows as a one-estimate
    predictions dict; "single01_nogeo" scores the "single01" outcome
    (Locate always knows the geometry)."""
    sub = "single01" if subset == "single01_nogeo" else subset
    key = f"{float(length):g}|{sub}|{which}"
    pos = {int(s): i for i, s in enumerate(rec["scenes"])}
    try:
        rows = np.array([pos[int(s)] for s in truth["scene"]])
    except KeyError as exc:
        raise KeyError(f"scene {exc} was not recomputed") from None
    ks = truth["k"]
    reason = rec[key + "_reason"][rows, ks]
    pub = reason == 0
    xy = np.where(pub[:, None], rec[key + "_xy"][rows, ks], 0.0)
    rad = np.where(pub, rec[key + "_radius"][rows, ks], metrics.MIN_RADIUS_M)
    return metrics.one_estimate_predictions(
        truth["scene"], truth["k"], xy, rad, pub, presence_prob=pub.astype(np.float64),
        name=f"{which}_recomputed_{float(length):g}s_{sub}", uses_calibration=True,
        note=f"wifisense.localize {which.upper()} recomputed: window_s {float(length):g} s, links {sub}, "
             f"idealised calibration")


# --------------------------------------------------------------------------
# 3. variance features
# --------------------------------------------------------------------------


def segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points `p` [..., 2] to segments `a`-`b` [..., 2]
    (broadcast); a zero-length segment is its point."""
    ab = b - a
    ap = p - a
    den = (ab * ab).sum(-1)
    t = np.where(den > 0, (ap * ab).sum(-1) / np.where(den > 0, den, 1.0), 0.0)
    t = np.clip(t, 0.0, 1.0)
    q = a + t[..., None] * ab
    return np.sqrt(((p - q) ** 2).sum(-1))


def scene_incidence(link_pos: np.ndarray, room: np.ndarray) -> np.ndarray:
    """[L, 16] bool: link `l`'s room-normalised 2-D segment passes within
    1/8 of cell `(i, j)`'s centre `((i + 0.5) / 4, (j + 0.5) / 4)`
    (column `4 i + j`; `i` along x, `j` along y)."""
    lp = np.asarray(link_pos, np.float64)
    scale = np.asarray(room, np.float64)[:2]
    a = lp[:, 0, :2] / scale
    b = lp[:, 1, :2] / scale
    centres = np.array([((i + 0.5) / GRID, (j + 0.5) / GRID) for i in range(GRID) for j in range(GRID)])
    d = segment_distance(centres[None, :, :], a[:, None, :], b[:, None, :])
    return d <= 1.0 / (2 * GRID) + 1e-12


def link_variance_maps(link_pos: np.ndarray, room: np.ndarray, s: np.ndarray, used: np.ndarray) -> tuple:
    """spec/model.md §3.3's maps for one window: `(mean_map, count_map)`,
    each [4, 4] indexed `[i, j]` (`i` along x): the mean `s_l` over the
    used links passing within 1/8 of the cell centre, and their count;
    both 0 where none."""
    inc = scene_incidence(link_pos, room) & np.asarray(used, bool)[:, None]
    cnt = inc.sum(0).astype(np.float64)
    tot = (inc * np.asarray(s, np.float64)[:, None]).sum(0)
    mean = np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1.0), 0.0)
    return mean.reshape(GRID, GRID), cnt.reshape(GRID, GRID)


def _stats(s: np.ndarray, used: np.ndarray) -> np.ndarray:
    """[W, 8] the summary statistics of `s_l` over each window's used links (0 when none)."""
    w = s.shape[0]
    out = np.zeros((w, 8))
    n = used.sum(1)
    ok = n > 0
    if not ok.any():
        return out
    v = np.where(used, s, np.nan)
    with np.errstate(all="ignore"):
        out[ok, 0] = np.nanmean(v[ok], 1)
        out[ok, 1] = np.nanstd(v[ok], 1)
        out[ok, 2] = np.nanmin(v[ok], 1)
        out[ok, 3] = np.nanmax(v[ok], 1)
        out[ok, 4:8] = np.nanpercentile(v[ok], [25, 50, 75, 90], axis=1).T
    return out


def window_features(c, windows) -> np.ndarray:
    """[W, 44] float64 features (FEATURE_NAMES) of cache windows `windows`
    (spec/model.md §3.3; module docstring)."""
    windows = np.asarray(windows, np.int64)
    win = c.windows
    rows = c.rows
    out = np.zeros((windows.size, N_FEATURES))
    srow = win["scene_row"][windows]
    for s in np.unique(srow):
        sel = np.flatnonzero(srow == s)
        ws = windows[sel]
        n = int(c.scenes["n_links"][s])
        _, link_pos = c.scene_links(int(s))
        room = c.scenes["room"][s].astype(np.float64)
        inc = scene_incidence(link_pos, room).astype(np.float64)          # [L, 16]
        r0 = win["row_offset"][ws]
        idx = (r0[:, None] + np.arange(n)[None, :]).ravel()
        sig = np.asarray(rows["sigma"][idx], np.float32).astype(np.float64).reshape(ws.size, n, -1)
        sl = sig.mean(-1)                                                  # [W, L]
        used = np.asarray(rows["link_mask"][idx], bool).reshape(ws.size, n)
        f = np.zeros((ws.size, N_FEATURES))
        f[:, 0:8] = _stats(sl, used)
        f[:, 8] = c.scenes["n_nodes"][s]
        f[:, 9:12] = room
        cnt = used.astype(np.float64) @ inc                               # [W, 16]
        tot = (sl * used) @ inc
        f[:, 12:28] = np.where(cnt > 0, tot / np.where(cnt > 0, cnt, 1.0), 0.0)
        f[:, 28:44] = cnt
        out[sel] = f
    return out


def _labels(c, windows) -> dict:
    win = c.windows
    s = win["scene_row"][windows]
    return {"presence": win["presence"][windows].astype(np.int64), "count": win["count"][windows].astype(np.int64),
            "zone": win["zone_occupied"][windows].astype(np.int64), "reach": c.scenes["zone_reachable"][s],
            "xy": win["xy"][windows].astype(np.float64), "pm": win["person_mask"][windows],
            "room": c.scenes["room"][s].astype(np.float64)}


def _xy_target(lab: dict) -> tuple:
    """One-person windows' `(mask, normalised (x/w, y/d) [W, 2])`."""
    one = lab["count"] == 1
    idx = np.argmax(lab["pm"], 1)
    xy = lab["xy"][np.arange(idx.size), idx]
    return one, xy / lab["room"][:, :2]


def _proba1(clf, x: np.ndarray) -> np.ndarray:
    """P(class 1) of a fitted binary classifier (0 when it only saw class 0)."""
    p = clf.predict_proba(x)
    classes = list(clf.classes_)
    return p[:, classes.index(1)] if 1 in classes else np.zeros(x.shape[0])


def _forest(seed: int, xtr: np.ndarray, lab: dict, xev: dict, n_estimators: int, n_jobs: int) -> dict:
    """Random forests for presence, count, zone (multi-output) and xy; per
    evaluation set the probabilities, the xy estimate (m) and its radius."""
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    out = {name: {} for name in xev}
    kw = dict(n_estimators=n_estimators, random_state=seed, n_jobs=n_jobs)
    rf = RandomForestClassifier(**kw).fit(xtr, lab["presence"])
    for name, (x, _) in xev.items():
        out[name]["presence_prob"] = _proba1(rf, x)
    del rf
    rf = RandomForestClassifier(**kw).fit(xtr, lab["count"])
    for name, (x, _) in xev.items():
        p = rf.predict_proba(x)
        full = np.zeros((x.shape[0], 4))
        full[:, [int(cl) for cl in rf.classes_]] = p
        out[name]["count_prob"] = full
    del rf
    rf = RandomForestClassifier(**kw).fit(xtr, lab["zone"])
    for name, (x, _) in xev.items():
        ps = rf.predict_proba(x)
        out[name]["zone_prob"] = np.stack([
            p[:, list(cls).index(1)] if 1 in list(cls) else np.zeros(x.shape[0])
            for p, cls in zip(ps, rf.classes_)], 1)
    del rf
    one, target = _xy_target(lab)
    rf = RandomForestRegressor(**kw).fit(xtr[one], target[one])
    for name, (x, room) in xev.items():
        trees = np.stack([t.predict(x) for t in rf.estimators_])          # [n_trees, W, 2] normalised
        scale = room[:, :2][None]
        trees_m = trees * scale
        mean_m = trees_m.mean(0)
        dist = np.sqrt(((trees_m - mean_m[None]) ** 2).sum(-1))           # [n_trees, W]
        out[name]["xy"] = mean_m
        out[name]["radius"] = np.maximum(np.percentile(dist, 95, axis=0), metrics.MIN_RADIUS_M)
    del rf
    return out


def _mlp_fit(task: str, seed: int, xtr, ytr, xva, yva, aux_tr=None, aux_va=None, *, max_epochs: int = 60,
             patience: int = 5, batch: int = 512, lr: float = 1e-3, threads: int = 8):
    """Train one 2 x 128 MLP for `task` (presence, count, zone, xy) with
    early stopping on the val loss; returns the best state's predict fn."""
    import torch
    from torch import nn
    torch.set_num_threads(threads)
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    n_out = {"presence": 1, "count": 4, "zone": 9, "xy": 3}[task]
    net = nn.Sequential(nn.Linear(xtr.shape[1], 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                        nn.Linear(128, n_out)).double()
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    xt, xv = torch.from_numpy(xtr), torch.from_numpy(xva)
    yt, yv = torch.from_numpy(ytr), torch.from_numpy(yva)
    at = torch.from_numpy(aux_tr) if aux_tr is not None else None
    av = torch.from_numpy(aux_va) if aux_va is not None else None

    def loss_fn(out, y, aux):
        if task == "presence":
            return nn.functional.binary_cross_entropy_with_logits(out[:, 0], y.double())
        if task == "count":
            return nn.functional.cross_entropy(out, y.long())
        if task == "zone":
            el = nn.functional.binary_cross_entropy_with_logits(out, y.double(), reduction="none")
            m = aux.double()
            return (el * m).sum() / m.sum().clamp(min=1.0)
        # xy: aux = room [w, d]; y normalised -> metres
        centre = torch.sigmoid(out[:, :2]) * aux
        sig = 0.1225 + nn.functional.softplus(out[:, 2])
        err2 = ((centre - y * aux) ** 2).sum(1)
        return (2.0 * torch.log(sig) + err2 / (2.0 * sig * sig)).mean()

    best, best_state, bad = float("inf"), None, 0
    n = xt.shape[0]
    for _epoch in range(max_epochs):
        net.train()
        perm = torch.randperm(n, generator=g)
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(net(xt[idx]), yt[idx], at[idx] if at is not None else None)
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            vl = float(loss_fn(net(xv), yv, av))
        if vl < best - 1e-9:
            best, bad = vl, 0
            best_state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    net.load_state_dict(best_state)
    net.eval()

    def predict(x, room=None):
        with torch.no_grad():
            out = net(torch.from_numpy(x))
            if task == "presence":
                return torch.sigmoid(out[:, 0]).numpy()
            if task == "count":
                return torch.softmax(out, 1).numpy()
            if task == "zone":
                return torch.sigmoid(out).numpy()
            r = torch.from_numpy(room[:, :2])
            centre = (torch.sigmoid(out[:, :2]) * r).numpy()
            sig = (0.1225 + nn.functional.softplus(out[:, 2])).clamp(max=7.4).numpy()
            return centre, sig
    return predict, best


def _mlp(seed: int, xtr, lab, xva, lab_va, xev: dict, threads: int) -> dict:
    mu, sd = xtr.mean(0), xtr.std(0)
    sd = np.where(sd > 0, sd, 1.0)
    z = lambda x: (x - mu) / sd                                   # noqa: E731
    ztr, zva = z(xtr), z(xva)
    out = {name: {} for name in xev}
    info = {}
    for task, ytr, yva, atr, ava in (
            ("presence", lab["presence"].astype(np.float64), lab_va["presence"].astype(np.float64), None, None),
            ("count", lab["count"], lab_va["count"], None, None),
            ("zone", lab["zone"].astype(np.float64), lab_va["zone"].astype(np.float64),
             lab["reach"].astype(np.float64), lab_va["reach"].astype(np.float64))):
        pred, best = _mlp_fit(task, seed, ztr, ytr, zva, yva, atr, ava, threads=threads)
        info[task] = best
        for name, (x, _) in xev.items():
            out[name][f"{task}_prob"] = pred(z(x))
    one, tgt = _xy_target(lab)
    one_v, tgt_v = _xy_target(lab_va)
    pred, best = _mlp_fit("xy", seed, ztr[one], tgt[one], zva[one_v], tgt_v[one_v],
                          lab["room"][one][:, :2].copy(), lab_va["room"][one_v][:, :2].copy(), threads=threads)
    info["xy"] = best
    for name, (x, room) in xev.items():
        centre, sig = pred(z(x), room)
        out[name]["xy"] = centre
        out[name]["radius"] = np.maximum(metrics.R95 * sig, metrics.MIN_RADIUS_M)
    out["_val_loss"] = info
    return out


def fit_variance_baselines(c, eval_splits=("test",), seeds=(0, 1, 2), *, n_estimators: int = 300,
                           n_jobs: int = 90, mlp_threads: int = 16, train_windows=None, val_windows=None,
                           eval_windows: dict | None = None, kinds=("forest", "mlp")) -> dict:
    """Fit the variance-feature forest and MLP on decision 5's train
    windows (early stopping on the val windows) once per seed; returns
    `{split: {"forest": [pred per seed], "mlp": [...]}}`, predictions dicts
    (one-estimate kind) for each evaluation split's decision 5 windows."""
    tr = c.window_indices("train", min_t=metrics.EVAL_MIN_T) if train_windows is None else np.asarray(train_windows)
    va = c.window_indices("val", min_t=metrics.EVAL_MIN_T) if val_windows is None else np.asarray(val_windows)
    evw = eval_windows or {s: c.window_indices(s, min_t=metrics.EVAL_MIN_T) for s in eval_splits}
    xtr, xva = window_features(c, tr), window_features(c, va)
    xev = {s: (window_features(c, w), c.scenes["room"][c.windows["scene_row"][w]].astype(np.float64))
           for s, w in evw.items()}
    lab, lab_va = _labels(c, tr), _labels(c, va)
    results = {}
    # In process, one model at a time: each forest uses n_jobs threads, each MLP (about 5 s on 187k
    # windows) mlp_threads; no fork, so a caller that already initialised CUDA is safe.
    for s in seeds:
        if "forest" in kinds:
            results[("forest", s)] = _forest(s, xtr, lab, xev, n_estimators, n_jobs)
        if "mlp" in kinds:
            results[("mlp", s)] = _mlp(s, xtr, lab, xva, lab_va, xev, mlp_threads)
    out = {}
    for split, w in evw.items():
        win = c.windows
        out[split] = {}
        for kind in kinds:
            preds = []
            for s in seeds:
                r = results[(kind, s)][split]
                preds.append(metrics.one_estimate_predictions(
                    win["scene"][w], win["k"][w], r["xy"], r["radius"], None, presence_prob=r["presence_prob"],
                    count_prob=r["count_prob"], zone_prob=r["zone_prob"], name=f"{kind}_variance", seed=int(s),
                    uses_calibration=False,
                    val_loss=results[(kind, s)].get("_val_loss") if kind == "mlp" else None))
            out[split][kind] = preds
    return out


# --------------------------------------------------------------------------
# 4. prior
# --------------------------------------------------------------------------


def footprint_centroid(footprint: np.ndarray) -> np.ndarray:
    """The area centroid of a simple polygon [V, 2] (shoelace)."""
    p = np.asarray(footprint, np.float64)
    p = p[np.isfinite(p).all(1)]
    x, y = p[:, 0], p[:, 1]
    x1, y1 = np.roll(x, -1), np.roll(y, -1)
    cr = x * y1 - x1 * y
    a = cr.sum() / 2.0
    if a == 0:
        return p.mean(0)
    return np.array([((x + x1) * cr).sum() / (6 * a), ((y + y1) * cr).sum() / (6 * a)])


def prior_predictions(c, truth: dict, train_windows=None) -> dict:
    """The prior (spec/model.md §3.4) at `truth`'s windows: the train
    split's frequencies (presence, count, each zone cell over the windows
    where it is reachable) and the footprint centroid with radius the
    95th percentile of its errors on one-person train windows (>= 0.3 m)."""
    tr = c.window_indices("train", min_t=metrics.EVAL_MIN_T) if train_windows is None else np.asarray(train_windows)
    win, sc = c.windows, c.scenes
    lab = _labels(c, tr)
    p_presence = lab["presence"].mean()
    p_count = np.bincount(lab["count"], minlength=4) / lab["count"].size
    reach = lab["reach"]
    p_zone = (lab["zone"] * reach).sum(0) / np.maximum(reach.sum(0), 1)
    cent = np.array([footprint_centroid(sc["footprint"][s][: int(sc["n_vertices"][s])]) for s in range(sc["scene"].size)])
    one, _ = _xy_target(lab)
    srow = win["scene_row"][tr]
    idx = np.argmax(lab["pm"], 1)
    pxy = lab["xy"][np.arange(idx.size), idx]
    err = np.sqrt(((pxy[one] - cent[srow[one]]) ** 2).sum(1))
    radius = max(float(np.percentile(err, 95)), metrics.MIN_RADIUS_M)
    w = truth["window"]
    n = w.size
    return metrics.one_estimate_predictions(
        win["scene"][w], win["k"][w], cent[win["scene_row"][w]], np.full(n, radius), None,
        presence_prob=np.full(n, p_presence), count_prob=np.tile(p_count, (n, 1)),
        zone_prob=np.tile(p_zone, (n, 1)), name="prior", uses_calibration=False,
        note="train-split class frequencies; footprint centroid, radius = p95 of its train errors")


# --------------------------------------------------------------------------
# assembly, saving
# --------------------------------------------------------------------------


def baseline_methods(c, truth: dict, *, recomputed: dict | None = None, length: float | None = None,
                     variance: dict | None = None, split: str | None = None, prior: bool = True,
                     stored: bool | None = None) -> dict:
    """Every baseline's predictions at `truth`'s windows, ready for
    `metrics.evaluate`: name -> predictions dict (or list per seed).

    - `wlm_stored`, `rti_stored` (all links only: the stored outcome used
      every link; default on when `truth["subset"]` is "all");
    - with `recomputed` (`load_recomputed(...)`) and `length` (the model's
      ℓ): `wlm_recomputed`, `rti_recomputed` on `truth`'s subset;
    - `prior` (all links only);
    - with `variance` (`fit_variance_baselines(...)` or
      `load_predictions(...)` output) and `split`: `forest`, `mlp` (lists
      over seeds; all links only)."""
    subset = truth.get("subset", "all") or "all"
    out = {}
    if stored is None:
        stored = subset == "all"
    if stored:
        out["wlm_stored"] = stored_predictions(c, truth, "wlm")
        out["rti_stored"] = stored_predictions(c, truth, "rti")
    if recomputed is not None:
        if length is None:
            raise ValueError("recomputed baselines need the model's window length")
        out["wlm_recomputed"] = recomputed_predictions(recomputed, truth, length, subset, "wlm")
        out["rti_recomputed"] = recomputed_predictions(recomputed, truth, length, subset, "rti")
    if prior and subset == "all":
        out["prior"] = prior_predictions(c, truth)
    if variance is not None and subset == "all":
        block = variance[split] if split is not None else variance
        for kind, preds in block.items():
            out[kind] = [metrics.align(metrics.check_predictions(p), truth) for p in preds]
    return out


_ARRAY_FIELDS = ("scene", "k", "presence_prob", "count_prob", "zone_prob", "slot_exist", "slot_xy", "slot_sigma",
                 "slot_radius", "published")


def save_predictions(path: str, methods: dict) -> None:
    """Save `{name: predictions dict or list of them}` (or `{split: {...}}`)
    as one .npz, labelled SIM."""
    flat = {"label": np.array(LABEL)}
    meta = {}

    def put(prefix, preds):
        preds = preds if isinstance(preds, (list, tuple)) else [preds]
        meta[prefix] = len(preds)
        for i, p in enumerate(preds):
            for f in _ARRAY_FIELDS:
                if p.get(f) is not None:
                    flat[f"{prefix}|{i}|{f}"] = np.asarray(p[f])
            extra = {k: v for k, v in p.items() if k not in _ARRAY_FIELDS and not k.startswith("_")
                     and isinstance(v, (str, int, float, bool, type(None), dict))}
            flat[f"{prefix}|{i}|json"] = np.array(json.dumps(extra, default=str))
    for name, v in methods.items():
        if isinstance(v, dict) and "scene" not in v:
            for inner, preds in v.items():
                put(f"{name}/{inner}", preds)
        else:
            put(name, v)
    flat["index"] = np.array(json.dumps(meta))
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    np.savez(path, **flat)


def load_predictions(path: str) -> dict:
    """The inverse of `save_predictions` (nested `{split: {name: [...]}}`
    when saved so)."""
    with np.load(path) as z:
        if str(z["label"]) != LABEL:
            raise ValueError(f"{path}: not labelled SIM")
        index = json.loads(str(z["index"]))
        out: dict = {}
        for prefix, n in index.items():
            preds = []
            for i in range(n):
                p = json.loads(str(z[f"{prefix}|{i}|json"]))
                for f in _ARRAY_FIELDS:
                    key = f"{prefix}|{i}|{f}"
                    p[f] = z[key] if key in z.files else None
                preds.append(p)
            if "/" in prefix:
                outer, inner = prefix.split("/", 1)
                out.setdefault(outer, {})[inner] = preds
            else:
                out[prefix] = preds
    return out


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _default_recomputed_path(dataset_dir: str) -> str:
    name = os.path.basename(os.path.normpath(os.path.abspath(dataset_dir)))
    return os.path.join(cache_mod.DEFAULT_ROOT, name, "locate_recomputed.npz")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 8 baselines (SIM).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("recompute", help="recompute WLM/RTI with wifisense.localize")
    r.add_argument("dataset_dir")
    r.add_argument("--splits", nargs="+", default=["val", "test"])
    r.add_argument("--lengths", type=float, nargs="+", default=[1.0, 2.0, 4.0])
    r.add_argument("--subsets", nargs="+", default=list(SUBSETS))
    r.add_argument("--workers", type=int, default=90)
    r.add_argument("--out", default=None)
    v = sub.add_parser("variance", help="fit the variance-feature forest and MLP (3 seeds) on a cache")
    v.add_argument("cache_dir")
    v.add_argument("--splits", nargs="+", default=["val", "test"])
    v.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    v.add_argument("--trees", type=int, default=300)
    v.add_argument("--n-jobs", type=int, default=90)
    v.add_argument("--out", default=None, help="default: <cache_dir>/../variance_baselines_<ℓ>.npz")
    s = sub.add_parser("stored", help="score the stored WLM and RTI on a cache's split")
    s.add_argument("cache_dir")
    s.add_argument("--split", default="test")
    s.add_argument("--boot", type=int, default=metrics.N_BOOT)
    s.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.cmd == "recompute":
        m = rd.load_manifest(args.dataset_dir)
        sp = rd.splits(m)
        scenes = sorted(i for name in args.splits for i in sp[name])
        out = args.out or _default_recomputed_path(args.dataset_dir)
        res = recompute_dataset(args.dataset_dir, scenes, args.lengths, args.subsets, workers=args.workers,
                                out_path=out)
        print(json.dumps({"label": LABEL, "out": out, "scenes": len(scenes), "wall_s": float(res["wall_s"])}))
    elif args.cmd == "variance":
        c = cache_mod.open_cache(args.cache_dir)
        t0 = time.time()
        res = fit_variance_baselines(c, tuple(args.splits), tuple(args.seeds), n_estimators=args.trees,
                                     n_jobs=args.n_jobs)
        out = args.out or os.path.join(os.path.dirname(c.dir), f"variance_baselines_{c.length_s:g}.npz")
        save_predictions(out, res)
        print(json.dumps({"label": LABEL, "out": out, "wall_s": round(time.time() - t0, 1)}))
    else:
        c = cache_mod.open_cache(args.cache_dir)
        truth = metrics.truth_from_cache(c, args.split)
        res = metrics.evaluate({"wlm_stored": stored_predictions(c, truth, "wlm"),
                                "rti_stored": stored_predictions(c, truth, "rti")}, truth, n_boot=args.boot)
        text = json.dumps(res, indent=1)
        if args.out:
            with open(args.out, "w", encoding="utf-8") as f:
                f.write(text)
        else:
            print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
