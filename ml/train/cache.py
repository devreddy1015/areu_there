"""The Phase 8 window cache (spec/model.md §1 "Cache"): every second label
tick's window of a SIM dataset, preprocessed once, as float16 memory maps.

Everything in a cache is **SIM** (spec/dataset.md: generated rooms,
simulated people and CSI); `meta.json` says so and `open_cache` refuses a
cache that does not.

**What is cached.** For a dataset (`roomsim-v1`, `roomsim-v1-hwoff`) and a
window length `ℓ` (`length_s`), every label tick `k` with `k % 2 == 1` and
`t_k = 0.5 (k + 1) >= ℓ` ("every_second": the window ends at whole seconds,
so the brief's 2 s windows `k = 4w + 3` are all included, and decision 5's
evaluation windows, odd `k` with `t_k >= 4 s`, are a subset for every
`ℓ <= 4`). Each window is cut by `ml.preprocess.from_roomsim_shard` and
prepared by `ml.preprocess.preprocess` with spec/model.md decision 2's
configuration, `PreprocessConfig(bin_hz=5, ref_bins="shared44",
min_packets=3)` (phase off unless `phase=True`); only the 44 reference
subcarriers `R` are kept (SIM never carries the other 8). A cached value
is the float16 rounding of `preprocess`'s float32 output, bit for bit
(tested).

**Layout** (`<out_dir>/`, numpy `.npy` files opened with `mmap_mode`):

- row arrays, one row per (window, link): the window `w`'s `n_links[w]`
  rows start at `row_offset[w]` and follow the scene's link order
  (ascending `(tx, rx)`, the scene's `links` table):
  `amp` [R, 44, T] f16 (z-scores at `R`, 0 where masked), `mu`, `sigma`
  [R, 44] f16, `rssi` [R, T] f16 (dBm), `gain_db` [R, T] f16 (spec/model.md
  §5.5's ablation channel; not a model input otherwise), `sample_mask`,
  `rssi_mask` [R, T] bool, `bin_count` [R, T] u8, `link_mask` [R] bool
  (the link is used: at least `min_packets` non-empty bins); with
  `phase=True` also `phase` [R, 44, T, 2] f16 (cos, sin) and `phase_mask`
  [R, 44, T] bool.
- `windows.npz`, one entry per window, ordered by scene then `k`:
  `scene_row` (into the scene table), `scene` (the dataset's scene index),
  `k`, `t`, `row_offset`, `n_links`, `split` (0 train, 1 val, 2 test); the
  labels at tick `k` (spec/dataset.md §5): `presence`, `count`,
  `zone_occupied` [9], `xy` [3, 2] (m; **0 where absent**, `person_mask`
  says who is present), `person_mask`, `walking` (at `k`), `walked`
  (walking at some label tick in `(t_k - ℓ, t_k]`, spec/model.md §1),
  `posture` (255 absent), `zone` (-1 absent); the stored baselines at `k`
  (spec/dataset.md §6: Locate's outcome, 1 s of RSSI, defaults):
  `wlm_reason`, `wlm_xy` [2], `wlm_radius`, `rti_*` likewise (NaN unless
  published); and the preprocessing counts `links_used`, `links_dropped`,
  `zero_gain_packets`.
- `scenes.npz`, one entry per scene: `scene`, `split`, `n_nodes`,
  `n_links`, `l_shape`, `n_people`, `room` [3] (`w, d, h`, m), `footprint`
  [6, 2] (NaN-padded) and `n_vertices`, `zone_reachable` [9],
  `zone_floor_frac` [9], `link_offset` [S + 1] into the link table
  `links` [ΣL, 2] (tx, rx) and `link_pos` [ΣL, 2, 3] (f32, SCENE: the
  shard's `node_pos`, as `preprocess` reports them), `window_offset`
  [S + 1] into the window table.
- `meta.json`: the label, the dataset and its impairment profile, `ℓ`,
  `T`, the preprocessing configuration, the tick rule, counts, sizes and
  the build time.

    from ml.train import cache
    meta = cache.build_cache("/home/Aman/are_u_theone/datasets/roomsim-v1", 2.0, workers=90)
    c = cache.open_cache(meta["out_dir"])
    w = c.window_indices("test", min_t=4.0)[0]
    one = c.read_window(w)          # amp [L, 44, T] f16, labels, stored WLM/RTI, link geometry

Builds run in parallel (`multiprocessing`, fork): the parent sizes every
file from the manifest (a scene's link count and its ticks fix its rows),
creates the memory maps, and each worker preprocesses whole scenes and
writes its rows in place. A cache is written to `<out_dir>.partial` and
renamed when complete, so a directory without `meta.json` is unfinished.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import math
import multiprocessing
import os
import shutil
import sys
import time

import numpy as np

from ml import preprocess as pp
from ml import roomsim_dataset as rd

__all__ = [
    "LABEL", "CACHE_SPEC", "CACHE_VERSION", "DEFAULT_ROOT", "FALLBACK_ROOT", "PREPROCESS", "N_REF", "TICK_RULES",
    "SPLIT_CODES", "ROW_FILES", "REF_SETS", "row_files", "cache_ticks", "default_cache_dir", "estimate_bytes",
    "build_cache", "open_cache", "Cache",
]

LABEL = "SIM"
CACHE_SPEC = "model v1 window cache (spec/model.md §1)"
CACHE_VERSION = 1
#: spec/model.md §1: in /dev/shm when it fits, otherwise under the runs directory.
DEFAULT_ROOT = "/dev/shm/saans-phase8-cache"
FALLBACK_ROOT = "/home/Aman/are_u_theone/runs/phase8/cache"
#: spec/model.md decision 2: binned at 5 Hz over the shared 44 reference bins, phase off.
PREPROCESS = pp.PreprocessConfig(phase=False, bin_hz=5, ref_bins="shared44", min_packets=3)
N_REF = 44
#: The named reference sets a cache can keep (spec/preprocess.md §3): shared44 for Phase 8, shared40 for
#: Phase 9's SIM pre-training (the bins every Phase 9 source carries, spec/phase9.md §3).
REF_SETS = {"shared44": pp.SHARED44, "shared40": pp.SHARED40}
TICK_RULES = ("every_second", "every")
SPLIT_CODES = {"train": 0, "val": 1, "test": 2}
SPLIT_NAMES = ("train", "val", "test")
MAX_PEOPLE = 3

#: Row arrays: name -> (dtype, trailing shape as a function of T). `phase*` only with phase=True.
ROW_FILES = {
    "amp": (np.float16, lambda t: (N_REF, t)),
    "mu": (np.float16, lambda t: (N_REF,)),
    "sigma": (np.float16, lambda t: (N_REF,)),
    "rssi": (np.float16, lambda t: (t,)),
    "gain_db": (np.float16, lambda t: (t,)),
    "sample_mask": (np.bool_, lambda t: (t,)),
    "rssi_mask": (np.bool_, lambda t: (t,)),
    "bin_count": (np.uint8, lambda t: (t,)),
    "link_mask": (np.bool_, lambda t: ()),
}
PHASE_FILES = {
    "phase": (np.float16, lambda t: (N_REF, t, 2)),
    "phase_mask": (np.bool_, lambda t: (N_REF, t)),
}


def row_files(n_ref: int = N_REF, phase: bool = False) -> dict:
    """ROW_FILES (and with `phase` PHASE_FILES) for `n_ref` reference bins."""
    out = {
        "amp": (np.float16, lambda t: (n_ref, t)),
        "mu": (np.float16, lambda t: (n_ref,)),
        "sigma": (np.float16, lambda t: (n_ref,)),
    }
    out.update({k: v for k, v in ROW_FILES.items() if k not in out})
    if phase:
        out["phase"] = (np.float16, lambda t: (n_ref, t, 2))
        out["phase_mask"] = (np.bool_, lambda t: (n_ref, t))
    return out


# --------------------------------------------------------------------------
# ticks, paths, sizes
# --------------------------------------------------------------------------


def _halves(length_s: float) -> int:
    return rd._halves(length_s)


def cache_ticks(n_ticks: int, length_s: float, ticks: str = "every_second") -> list:
    """The label ticks a cache holds for a scene with `n_ticks` ticks: `k`
    with `t_k = 0.5 (k + 1) >= length_s` (`k >= 2 length_s - 1`) and, for
    "every_second", `k % 2 == 1` (windows ending at whole seconds, so the
    brief's 2 s windows `k = 4w + 3` are included); "every" keeps every
    tick. SIM."""
    if ticks not in TICK_RULES:
        raise ValueError(f"ticks {ticks!r} is not one of {TICK_RULES}")
    first = _halves(length_s) - 1
    step = 2 if ticks == "every_second" else 1
    start = first + (first % 2 == 0) if ticks == "every_second" else first
    return list(range(start, int(n_ticks), step))


def _length_name(length_s: float) -> str:
    return f"{float(length_s):g}"


def default_cache_dir(dataset: str, length_s: float, root: str = DEFAULT_ROOT, ref_bins: str = "shared44") -> str:
    """`<root>/<dataset name>/<ℓ>/`, e.g. `/dev/shm/saans-phase8-cache/roomsim-v1/2`; another reference
    set than shared44 adds it to the name (`roomsim-v1-shared40/2`)."""
    name = os.path.basename(os.path.normpath(os.path.abspath(os.fspath(dataset))))
    if ref_bins != "shared44":
        name = f"{name}-{ref_bins}"
    return os.path.join(root, name, _length_name(length_s))


def _scene_rows(manifest: dict, indices, length_s: float, ticks: str) -> list:
    """Per scene `(index, n_links, [ticks])`, from the manifest alone."""
    n_ticks = 2 * int(manifest["seconds"])
    out = []
    entries = {int(e["index"]): e for e in manifest["scenes"]}
    for i in indices:
        e = entries[int(i)]
        n_links = int(e.get("n_links", int(e["n_nodes"]) * (int(e["n_nodes"]) - 1)))
        out.append((int(i), n_links, cache_ticks(n_ticks, length_s, ticks)))
    return out


def _t_bins(length_s: float) -> int:
    nb, _ = pp.ref_impl.bin_layout(PREPROCESS.bin_hz, pp._length_ns(length_s))
    return int(nb)


def estimate_bytes(dataset_dir: str, length_s: float, ticks: str = "every_second", phase: bool = False,
                   scenes=None, ref_bins: str = "shared44") -> int:
    """The cache's size in bytes (row files; the tables add a few tens of MB)."""
    manifest = rd.load_manifest(dataset_dir)
    idx = scenes if scenes is not None else [int(e["index"]) for e in manifest["scenes"]]
    plan = _scene_rows(manifest, idx, length_s, ticks)
    rows = sum(n_links * len(ks) for _, n_links, ks in plan)
    t = _t_bins(length_s)
    files = row_files(len(REF_SETS[ref_bins]), phase)
    per_row = sum(np.dtype(dt).itemsize * int(np.prod(shape(t), dtype=np.int64)) for dt, shape in files.values())
    return int(rows * per_row)


def _free_bytes(path: str) -> int:
    while not os.path.exists(path):
        parent = os.path.dirname(path)
        if parent == path:
            break
        path = parent
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


# --------------------------------------------------------------------------
# the worker: one scene
# --------------------------------------------------------------------------

_WORKER: dict = {}


def _worker_init(partial_dir: str, files: dict, dataset_dir: str, manifest: dict, length_s: float,
                 cfg: pp.PreprocessConfig) -> None:
    _WORKER.clear()
    _WORKER.update(partial_dir=partial_dir, files=files, dataset_dir=dataset_dir, manifest=manifest,
                   length_s=length_s, cfg=cfg, maps={})


def _map(name: str) -> np.ndarray:
    maps = _WORKER["maps"]
    if name not in maps:
        maps[name] = np.load(os.path.join(_WORKER["partial_dir"], name + ".npy"), mmap_mode="r+")
    return maps[name]


def _footprint(scene: dict) -> tuple:
    fp = np.full((6, 2), np.nan, np.float32)
    v = scene["footprint"].astype(np.float32)
    fp[: len(v)] = v
    return fp, len(v)


def _walked(walking: np.ndarray, k: int, halves: int) -> np.ndarray:
    """spec/model.md §1: walking at some label tick in `(t_k - ℓ, t_k]`,
    i.e. ticks `k - 2ℓ + 1 .. k` (ticks are 0.5 s apart)."""
    lo = max(0, k - halves + 1)
    return walking[lo:k + 1].astype(bool).any(axis=0)


def scene_windows(dataset_dir: str, index: int, ticks_k, length_s: float, cfg: pp.PreprocessConfig = PREPROCESS,
                  *, manifest: dict | None = None, scene: dict | None = None):
    """Preprocess SIM scene `index`'s windows ending at `ticks_k`: returns
    `(scene, outs)` with `outs[j]` = `preprocess(from_roomsim_shard(...,
    ticks_k[j], length_s), cfg)`. What the cache stores, before the cast."""
    if scene is None:
        scene = rd.load_scene(dataset_dir, index, manifest=manifest)
    wins = pp.from_roomsim_shard(dataset_dir, index, list(ticks_k), length_s, manifest=manifest, scene=scene)
    return scene, [pp.preprocess(w, cfg) for w in wins]


def _scene_task(task: tuple) -> dict:
    """Preprocess one scene's windows, write its rows in place, return its
    labels and tables (small)."""
    scene_row, index, ks, row0 = task
    dataset_dir, manifest = _WORKER["dataset_dir"], _WORKER["manifest"]
    length_s, cfg = _WORKER["length_s"], _WORKER["cfg"]
    halves = _halves(length_s)
    scene, outs = scene_windows(dataset_dir, index, ks, length_s, cfg, manifest=manifest)
    n_links = int(scene["links"].shape[0])
    ref = outs[0]["ref"] if outs else np.zeros(52, bool)
    n_ref = len(REF_SETS[cfg.ref_bins])
    if outs and int(ref.sum()) != n_ref:
        raise ValueError(f"scene {index}: {int(ref.sum())} reference bins, not {n_ref}")
    nw = len(ks)
    t = _t_bins(length_s)
    row = row0
    phase = cfg.phase
    amp_m, mu_m, sig_m = _map("amp"), _map("mu"), _map("sigma")
    rssi_m, gain_m, sm_m, rm_m = _map("rssi"), _map("gain_db"), _map("sample_mask"), _map("rssi_mask")
    bc_m, lm_m = _map("bin_count"), _map("link_mask")
    if phase:
        ph_m, pm_m = _map("phase"), _map("phase_mask")
    counts = {"links_used": np.zeros(nw, np.int16), "links_dropped": np.zeros(nw, np.int16),
              "zero_gain_packets": np.zeros(nw, np.int32)}
    for j, out in enumerate(outs):
        if out["amp"].shape != (n_links, 52, t):
            raise ValueError(f"scene {index} k={ks[j]}: amp {out['amp'].shape}, expected {(n_links, 52, t)}")
        sl = slice(row, row + n_links)
        amp_m[sl] = out["amp"][:, ref, :].astype(np.float16)
        mu_m[sl] = out["mu"][:, ref].astype(np.float16)
        sig_m[sl] = out["sigma"][:, ref].astype(np.float16)
        rssi_m[sl] = out["rssi"].astype(np.float16)
        gain_m[sl] = out["gain_db"].astype(np.float16)
        sm_m[sl] = out["sample_mask"]
        rm_m[sl] = out["rssi_mask"]
        bc_m[sl] = np.minimum(out["bin_count"], 255).astype(np.uint8)
        lm_m[sl] = out["link_mask"]
        if phase:
            ph_m[sl] = out["phase"][:, ref].astype(np.float16)
            pm_m[sl] = out["phase_mask"][:, ref]
        counts["links_used"][j] = int(out["link_mask"].sum())
        counts["links_dropped"][j] = int(out["links_dropped"])
        counts["zero_gain_packets"][j] = int(out["zero_gain_packets"])
        row += n_links
    for m in _WORKER["maps"].values():
        m.flush()

    ks_arr = np.asarray(ks, np.int64)
    pmask = scene["person_mask"][ks_arr].astype(bool)
    xy = np.where(pmask[..., None], np.nan_to_num(scene["xy"][ks_arr].astype(np.float32), nan=0.0), 0.0)
    walked = np.stack([_walked(scene["walking"], int(k), halves) for k in ks]) if nw else np.zeros((0, 3), bool)
    walked &= pmask
    labels = {
        "k": ks_arr.astype(np.int32),
        "t": (0.5 * (ks_arr + 1)).astype(np.float32),
        "presence": scene["presence"][ks_arr].astype(np.uint8),
        "count": scene["count"][ks_arr].astype(np.uint8),
        "zone_occupied": scene["zone_occupied"][ks_arr].astype(np.uint8),
        "xy": xy.astype(np.float32),
        "person_mask": pmask,
        "walking": scene["walking"][ks_arr].astype(bool) & pmask,
        "walked": walked,
        "posture": scene["posture"][ks_arr].astype(np.uint8),
        "zone": scene["zone"][ks_arr].astype(np.int8),
    }
    for name in ("wlm", "rti"):
        labels[f"{name}_reason"] = scene[f"{name}_reason"][ks_arr].astype(np.uint8)
        labels[f"{name}_xy"] = scene[f"{name}_xy"][ks_arr].astype(np.float32)
        labels[f"{name}_radius"] = scene[f"{name}_radius"][ks_arr].astype(np.float32)
    labels.update(counts)
    links = scene["links"].astype(np.int16)
    pos = {int(i): p for i, p in zip(scene["node_ids"], scene["node_pos"])}
    link_pos = np.stack([np.stack([pos[int(a)], pos[int(b)]]) for a, b in links]).astype(np.float32)
    if outs and not np.array_equal(outs[0]["link_pos"], link_pos):
        raise ValueError(f"scene {index}: link positions differ from preprocess's")
    fp, n_v = _footprint(scene)
    entry = scene["entry"]
    info = {
        "scene_row": scene_row, "scene": int(index), "n_nodes": int(scene["meta"]["n_nodes"]),
        "n_links": n_links, "l_shape": bool(entry.get("l_shape", len(scene["footprint"]) == 6)),
        "n_people": int(scene["meta"]["n_people"]), "room": scene["room_size"].astype(np.float32),
        "footprint": fp, "n_vertices": n_v, "zone_reachable": scene["zone_reachable"].astype(bool),
        "zone_floor_frac": scene["zone_floor_frac"].astype(np.float32), "links": links, "link_pos": link_pos,
    }
    return {"info": info, "labels": labels}


# --------------------------------------------------------------------------
# build
# --------------------------------------------------------------------------


def build_cache(dataset_dir: str, length_s: float, out_dir: str | None = None, ticks: str = "every_second",
                workers: int | None = None, *, scenes=None, phase: bool = False, overwrite: bool = False,
                progress: bool = True, root: str | None = None, ref_bins: str = "shared44") -> dict:
    """Preprocess every cached window (`cache_ticks`) of SIM dataset
    `dataset_dir` at window length `length_s` into `out_dir` (default
    `default_cache_dir(dataset_dir, length_s)` under `/dev/shm`, or under
    `FALLBACK_ROOT` when `/dev/shm` lacks room), in parallel over scenes with
    `workers` processes (default: all cores but two). `scenes` restricts it
    to some scene indices (tests); `ref_bins` picks the kept reference set
    (REF_SETS; shared40 for Phase 9). Returns the cache's `meta.json` dict,
    with `out_dir`, the build time and the size. Refuses an existing cache
    unless `overwrite`. Everything written is SIM."""
    t_start = time.time()
    manifest = rd.load_manifest(dataset_dir)
    profile = rd.impairments(manifest)
    split_of = {}
    for name, idx in rd.splits(manifest).items():
        for i in idx:
            split_of[int(i)] = name
    indices = sorted(int(e["index"]) for e in manifest["scenes"]) if scenes is None else [int(i) for i in scenes]
    if len(set(indices)) != len(indices):
        raise ValueError("repeated scene index")
    if ref_bins not in REF_SETS:
        raise ValueError(f"ref_bins {ref_bins!r} is not one of {sorted(REF_SETS)}")
    cfg = pp.PreprocessConfig(phase=bool(phase), bin_hz=PREPROCESS.bin_hz, ref_bins=ref_bins,
                              min_packets=PREPROCESS.min_packets)
    t_bins = _t_bins(length_s)
    plan = _scene_rows(manifest, indices, length_s, ticks)
    n_rows = sum(n_links * len(ks) for _, n_links, ks in plan)
    n_windows = sum(len(ks) for _, _, ks in plan)
    need = estimate_bytes(dataset_dir, length_s, ticks, phase, indices, ref_bins)
    if out_dir is None:
        out_dir = default_cache_dir(dataset_dir, length_s, root or DEFAULT_ROOT, ref_bins)
        if root is None and need + (16 << 30) > _free_bytes(DEFAULT_ROOT):
            out_dir = default_cache_dir(dataset_dir, length_s, FALLBACK_ROOT, ref_bins)
    out_dir = os.path.abspath(out_dir)
    if os.path.exists(out_dir):
        if not overwrite:
            raise FileExistsError(f"{out_dir} exists; pass overwrite=True to rebuild it")
        shutil.rmtree(out_dir)
    if need + (1 << 30) > _free_bytes(os.path.dirname(out_dir)):
        raise OSError(f"{out_dir}: the cache needs about {need / 2**30:.1f} GiB, more than is free")
    partial = out_dir + ".partial"
    if os.path.exists(partial):
        shutil.rmtree(partial)
    os.makedirs(partial)
    files = row_files(len(REF_SETS[ref_bins]), phase)
    for name, (dt, shape) in files.items():
        m = np.lib.format.open_memmap(os.path.join(partial, name + ".npy"), mode="w+", dtype=dt,
                                      shape=(n_rows,) + shape(t_bins))
        m.flush()
        del m
    tasks = []
    row = 0
    for s, (i, n_links, ks) in enumerate(plan):
        tasks.append((s, i, ks, row))
        row += n_links * len(ks)
    workers = max(1, (os.cpu_count() or 2) - 2) if workers is None else max(1, int(workers))
    init = (partial, {k: None for k in files}, dataset_dir, manifest, float(length_s), cfg)
    results = [None] * len(tasks)
    done = 0
    step = max(1, len(tasks) // 20)
    if workers == 1:
        _worker_init(*init)
        it = map(_scene_task, tasks)
        for res in it:
            results[res["info"]["scene_row"]] = res
            done += 1
            if progress and (done % step == 0 or done == len(tasks)):
                _progress(done, len(tasks), t_start)
        _WORKER.clear()
    else:
        ctx = multiprocessing.get_context("fork")
        # Big scenes first, so the tail of the build is not one 12-node scene.
        order = sorted(tasks, key=lambda tk: -len(tk[2]) * _scene_links(plan, tk[0]))
        with ctx.Pool(workers, initializer=_worker_init, initargs=init) as pool:
            for res in pool.imap_unordered(_scene_task, order, chunksize=1):
                results[res["info"]["scene_row"]] = res
                done += 1
                if progress and (done % step == 0 or done == len(tasks)):
                    _progress(done, len(tasks), t_start)
    # tables
    infos = [r["info"] for r in results]
    labels = [r["labels"] for r in results]
    n_scenes = len(infos)
    link_offset = np.zeros(n_scenes + 1, np.int64)
    link_offset[1:] = np.cumsum([inf["n_links"] for inf in infos])
    window_offset = np.zeros(n_scenes + 1, np.int64)
    window_offset[1:] = np.cumsum([len(lb["k"]) for lb in labels])
    scenes_tab = {
        "scene": np.array([inf["scene"] for inf in infos], np.int32),
        "split": np.array([SPLIT_CODES[split_of[inf["scene"]]] for inf in infos], np.int8),
        "n_nodes": np.array([inf["n_nodes"] for inf in infos], np.int16),
        "n_links": np.array([inf["n_links"] for inf in infos], np.int16),
        "l_shape": np.array([inf["l_shape"] for inf in infos], bool),
        "n_people": np.array([inf["n_people"] for inf in infos], np.int8),
        "room": np.stack([inf["room"] for inf in infos]).astype(np.float32),
        "footprint": np.stack([inf["footprint"] for inf in infos]).astype(np.float32),
        "n_vertices": np.array([inf["n_vertices"] for inf in infos], np.int8),
        "zone_reachable": np.stack([inf["zone_reachable"] for inf in infos]).astype(bool),
        "zone_floor_frac": np.stack([inf["zone_floor_frac"] for inf in infos]).astype(np.float32),
        "link_offset": link_offset,
        "links": np.concatenate([inf["links"] for inf in infos]).astype(np.int16),
        "link_pos": np.concatenate([inf["link_pos"] for inf in infos]).astype(np.float32),
        "window_offset": window_offset,
    }
    win = {key: np.concatenate([lb[key] for lb in labels]) for key in labels[0]}
    n_per = np.array([len(lb["k"]) for lb in labels], np.int64)
    win["scene_row"] = np.repeat(np.arange(n_scenes, dtype=np.int32), n_per)
    win["scene"] = scenes_tab["scene"][win["scene_row"]]
    win["split"] = scenes_tab["split"][win["scene_row"]]
    win["n_links"] = scenes_tab["n_links"][win["scene_row"]].astype(np.int32)
    win["row_offset"] = np.zeros(n_windows, np.int64)
    win["row_offset"][1:] = np.cumsum(win["n_links"][:-1].astype(np.int64))
    if win["row_offset"][-1] + win["n_links"][-1] != n_rows:
        raise AssertionError("row bookkeeping")
    for s, task in enumerate(tasks):
        if win["row_offset"][window_offset[s]] != task[3]:
            raise AssertionError(f"row offset of scene row {s}")
    np.savez(os.path.join(partial, "scenes.npz"), **scenes_tab)
    np.savez(os.path.join(partial, "windows.npz"), **win)
    sizes = {name: os.path.getsize(os.path.join(partial, name)) for name in sorted(os.listdir(partial))}
    build_s = time.time() - t_start
    meta = {
        "label": LABEL, "spec": CACHE_SPEC, "cache_version": CACHE_VERSION,
        "dataset": os.path.basename(os.path.normpath(os.path.abspath(dataset_dir))),
        "dataset_dir": os.path.abspath(dataset_dir), "impairments": profile,
        "baseline_valid": bool(manifest["baseline_valid"]),
        "manifest_sha256": _sha256(os.path.join(dataset_dir, "manifest.json")),
        "length_s": float(length_s), "T": t_bins, "bin_hz": PREPROCESS.bin_hz,
        "preprocess": {"phase": cfg.phase, "bin_hz": cfg.bin_hz, "ref_bins": cfg.ref_bins,
                       "min_packets": cfg.min_packets},
        "ref_k": [int(k) for k in REF_SETS[ref_bins]], "n_ref": len(REF_SETS[ref_bins]),
        "ticks": ticks, "tick_rule": "k % 2 == 1 and t_k >= length_s" if ticks == "every_second"
        else "t_k >= length_s",
        "seconds": int(manifest["seconds"]), "n_scenes": n_scenes, "n_windows": int(n_windows),
        "n_rows": int(n_rows), "phase": bool(phase),
        "row_files": {name: {"dtype": np.dtype(dt).str, "shape": [n_rows] + list(shape(t_bins))}
                      for name, (dt, shape) in files.items()},
        "split_scenes": {name: int((scenes_tab["split"] == code).sum()) for name, code in SPLIT_CODES.items()},
        "split_windows": {name: int((win["split"] == code).sum()) for name, code in SPLIT_CODES.items()},
        "windows_without_links": int((win["links_used"] == 0).sum()),
        "links_dropped_total": int(win["links_dropped"].astype(np.int64).sum()),
        "bytes": int(sum(sizes.values())), "file_bytes": sizes, "build_s": round(build_s, 3),
        "workers": workers, "created": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
    }
    with open(os.path.join(partial, "meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1)
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)
    os.rename(partial, out_dir)
    meta["out_dir"] = out_dir
    return meta


def _scene_links(plan, s):
    return plan[s][1]


def _progress(done: int, total: int, t0: float) -> None:
    el = time.time() - t0
    eta = el / done * (total - done) if done else float("nan")
    print(f"[cache SIM] {done}/{total} scenes, {el:.0f} s elapsed, ETA {eta:.0f} s", file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# reading
# --------------------------------------------------------------------------


class Cache:
    """An opened SIM window cache: `meta` (meta.json), `scenes` and
    `windows` (the tables, in memory), `rows` (the row arrays, memory
    mapped read-only). Treat everything as read-only."""

    def __init__(self, cache_dir: str, mmap: bool = True):
        self.dir = os.path.abspath(cache_dir)
        path = os.path.join(self.dir, "meta.json")
        if not os.path.isfile(path):
            raise FileNotFoundError(f"{path}: no meta.json (an unfinished or missing cache)")
        with open(path, encoding="utf-8") as f:
            self.meta = json.load(f)
        if self.meta.get("label") != LABEL or self.meta.get("spec") != CACHE_SPEC:
            raise ValueError(f"{self.dir}: label {self.meta.get('label')!r} spec {self.meta.get('spec')!r}; "
                             f"refusing a cache not marked SIM / {CACHE_SPEC!r}")
        if self.meta.get("cache_version") != CACHE_VERSION:
            raise ValueError(f"{self.dir}: cache version {self.meta.get('cache_version')}, this reads {CACHE_VERSION}")
        with np.load(os.path.join(self.dir, "scenes.npz")) as z:
            self.scenes = {k: z[k] for k in z.files}
        with np.load(os.path.join(self.dir, "windows.npz")) as z:
            self.windows = {k: z[k] for k in z.files}
        self.T = int(self.meta["T"])
        self.length_s = float(self.meta["length_s"])
        self._mmap = mmap
        self._rows = None
        self.n_windows = int(self.meta["n_windows"])
        self._scene_row_of = {int(s): r for r, s in enumerate(self.scenes["scene"])}

    # Row arrays are opened lazily, so a forked DataLoader worker maps its own.
    @property
    def rows(self) -> dict:
        if self._rows is None:
            names = list(self.meta["row_files"])
            self._rows = {n: np.load(os.path.join(self.dir, n + ".npy"), mmap_mode="r" if self._mmap else None)
                          for n in names}
        return self._rows

    def __getstate__(self):
        state = dict(self.__dict__)
        state["_rows"] = None
        return state

    @property
    def label(self) -> str:
        return LABEL

    def scene_row(self, scene: int) -> int:
        return self._scene_row_of[int(scene)]

    def window_indices(self, split: str | None = None, *, min_t: float | None = None, scenes=None) -> np.ndarray:
        """Window numbers of `split` ("train", "val", "test" or None for all),
        optionally with `t_k >= min_t` (decision 5's evaluation windows:
        `min_t=4.0`) and restricted to some dataset scene indices."""
        keep = np.ones(self.n_windows, bool)
        if split is not None:
            keep &= self.windows["split"] == SPLIT_CODES[split]
        if min_t is not None:
            keep &= self.windows["t"] >= np.float32(min_t)
        if scenes is not None:
            keep &= np.isin(self.windows["scene"], np.asarray(list(scenes), np.int64))
        return np.flatnonzero(keep)

    def scene_links(self, scene_row: int) -> tuple:
        """`(links [L, 2] int16, link_pos [L, 2, 3] f32)` of a scene row."""
        a, b = self.scenes["link_offset"][scene_row], self.scenes["link_offset"][scene_row + 1]
        return self.scenes["links"][a:b], self.scenes["link_pos"][a:b]

    def read_window(self, w: int, names=None) -> dict:
        """One window: its row arrays (`amp` [L, 44, T] f16, `mu`, `sigma`
        [L, 44], `rssi`, `gain_db`, `sample_mask`, `rssi_mask`, `bin_count`
        [L, T], `link_mask` [L], and `phase`/`phase_mask` in a phase cache),
        the scene's `links`, `link_pos`, `room`, `zone_reachable`, and the
        window's labels and stored baselines. SIM."""
        w = int(w)
        r0 = int(self.windows["row_offset"][w])
        n = int(self.windows["n_links"][w])
        rows = self.rows
        names = list(rows) if names is None else names
        out = {name: np.asarray(rows[name][r0:r0 + n]) for name in names}
        s = int(self.windows["scene_row"][w])
        out["links"], out["link_pos"] = self.scene_links(s)
        for key in ("room", "zone_reachable", "n_nodes", "l_shape", "n_people", "footprint", "n_vertices"):
            out[key] = self.scenes[key][s]
        for key, arr in self.windows.items():
            out.setdefault(key, arr[w])
        out["label"] = LABEL
        return out


def open_cache(cache_dir: str, mmap: bool = True) -> Cache:
    """Open a SIM window cache written by `build_cache`."""
    return Cache(cache_dir, mmap=mmap)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Build the Phase 8 SIM window cache (spec/model.md §1).")
    ap.add_argument("dataset_dir")
    ap.add_argument("--length", type=float, nargs="+", required=True, help="window length(s) ℓ in s")
    ap.add_argument("--out", default=None, help="cache directory (only with one --length)")
    ap.add_argument("--root", default=None, help=f"root of the default directories (default {DEFAULT_ROOT})")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--ticks", default="every_second", choices=TICK_RULES)
    ap.add_argument("--phase", action="store_true")
    ap.add_argument("--ref-bins", default="shared44", choices=sorted(REF_SETS))
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    if args.out and len(args.length) != 1:
        ap.error("--out takes one --length")
    for length in args.length:
        meta = build_cache(args.dataset_dir, length, args.out, args.ticks, args.workers, phase=args.phase,
                           overwrite=args.overwrite, root=args.root, ref_bins=args.ref_bins)
        print(json.dumps({k: meta[k] for k in ("label", "out_dir", "dataset", "impairments", "length_s", "T",
                                                "n_scenes", "n_windows", "n_rows", "bytes", "build_s",
                                                "windows_without_links", "links_dropped_total")}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
