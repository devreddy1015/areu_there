"""The Phase 9 REAL window cache (spec/phase9.md §3): each OpenCSI session's
windows, preprocessed once and kept as float16 arrays in one `.npz` per
(session, `ℓ`), with a `meta.json`. Everything in it is **REAL**.

Windows end every 0.5 s inside each labelled segment (`ml.opencsi.window_ends`);
the evaluation windows are those on a 1 s step whose window starts at least
`EVAL_SKIP_S` after the first segment's start (the baselines' calibration
span, spec/phase9.md §5). Every session has the same 56 links in ascending
`(tx, rx)` order, so arrays are `[W, L, ...]` with `link_mask` saying which
links preprocessing used.

    from ml.phase9 import real_cache as rc
    meta = rc.build("envC_C3", 2.0)
    c = rc.load("envC_C3", 2.0)          # dict of arrays + meta, label REAL
"""

from __future__ import annotations

import json
import math
import multiprocessing
import os
import sys
import time

import numpy as np

from ml import opencsi as oc
from ml import preprocess as pp

__all__ = ["LABEL", "ROOT", "PREPROCESS", "TRAIN_HOP_S", "EVAL_HOP_S", "EVAL_SKIP_S", "build", "load", "path"]

LABEL = "REAL"
ROOT = "/home/Aman/are_u_theone/runs/phase9/real_cache"
#: spec/phase9.md §3: binned at 5 Hz over the 40 bins every Phase 9 source carries, phase off.
PREPROCESS = pp.PreprocessConfig(phase=False, bin_hz=5, ref_bins="shared40", min_packets=3)
TRAIN_HOP_S = 0.5
EVAL_HOP_S = 1.0
EVAL_SKIP_S = 90.0
_FORMAT = 1

_SESSION: dict = {}


def path(trial: str, length_s: float, root: str = ROOT) -> tuple:
    """`(<root>/<trial>_<ℓ>.npz, <root>/<trial>_<ℓ>.json)`."""
    stem = os.path.join(root, f"{trial}_{float(length_s):g}")
    return stem + ".npz", stem + ".json"


def _one(t_end_ns: int) -> dict:
    s, length_s = _SESSION["session"], _SESSION["length_s"]
    out = pp.preprocess(oc.window(s, t_end_ns, length_s), PREPROCESS)
    ref = out["ref"]
    return {
        "amp": out["amp"][:, ref, :].astype(np.float16), "mu": out["mu"][:, ref].astype(np.float16),
        "sigma": out["sigma"][:, ref].astype(np.float16), "rssi": out["rssi"].astype(np.float16),
        "sample_mask": out["sample_mask"], "rssi_mask": out["rssi_mask"],
        "bin_count": np.minimum(out["bin_count"], 255).astype(np.uint8), "link_mask": out["link_mask"],
        "zero_gain_packets": int(out["zero_gain_packets"]), "label": out["label"], "n_ref": int(ref.sum()),
    }


def build(trial: str, length_s: float, *, root: str = ROOT, opencsi_root: str = oc.DEFAULT_ROOT,
          workers: int | None = None, overwrite: bool = False, session: oc.Session | None = None) -> dict:
    """Preprocess every window of `trial` at `length_s` and write the cache.
    Returns its meta dict."""
    t0 = time.time()
    arrays_path, meta_path = path(trial, length_s, root)
    if os.path.exists(meta_path) and not overwrite:
        raise FileExistsError(f"{meta_path} exists; pass overwrite=True")
    s = oc.load_session(opencsi_root, trial) if session is None else session
    if s.label != LABEL:
        raise ValueError(f"{trial}: session labelled {s.label!r}, not REAL")
    ends = oc.window_ends(s, length_s, TRAIN_HOP_S)
    t_end = np.array([e.t_end_ns for e in ends], np.int64)
    seg = np.array([e.segment for e in ends], np.int16)
    _SESSION.clear()
    _SESSION.update(session=s, length_s=float(length_s))
    workers = max(1, (os.cpu_count() or 2) - 2) if workers is None else max(1, int(workers))
    if workers == 1:
        outs = [_one(int(t)) for t in t_end]
    else:
        with multiprocessing.get_context("fork").Pool(min(workers, len(ends))) as pool:
            outs = pool.map(_one, t_end.tolist(), chunksize=8)
    _SESSION.clear()
    if any(o["label"] != LABEL for o in outs):
        raise AssertionError("a REAL window came out with another label")
    keys = ("amp", "mu", "sigma", "rssi", "sample_mask", "rssi_mask", "bin_count", "link_mask")
    arr = {k: np.stack([o[k] for o in outs]) for k in keys}
    ln = oc._length_ns(length_s)
    first = s.segments[0].start_ns
    starts = np.array([s.segments[i].start_ns for i in seg], np.int64)
    step = (t_end - ln - starts) // oc._length_ns(TRAIN_HOP_S)
    hop_ratio = int(round(EVAL_HOP_S / TRAIN_HOP_S))
    eval_ = (step % hop_ratio == 0) & (t_end - ln >= first + oc._length_ns(EVAL_SKIP_S))
    lab = [oc.LABELS[s.segments[i].fine_label] for i in seg]
    arr.update({
        "t_end_ns": t_end, "segment": seg, "eval": eval_,
        "presence": np.array([p for p, _, _ in lab], np.uint8), "count": np.array([c for _, c, _ in lab], np.uint8),
        "moving": np.array([m for _, _, m in lab], bool), "links": s.links.astype(np.int16),
        "zero_gain_packets": np.array([o["zero_gain_packets"] for o in outs], np.int32),
    })
    os.makedirs(root, exist_ok=True)
    tmp = arrays_path + ".tmp.npz"
    np.savez(tmp, **arr)
    os.replace(tmp, arrays_path)
    n_ref = outs[0]["n_ref"] if outs else len(pp.SHARED40)
    meta = {
        "label": LABEL, "format": _FORMAT, "spec": "spec/phase9.md §3", "trial": trial,
        "environment": s.environment, "chip": s.chip, "layout": s.layout.name, "length_s": float(length_s),
        "T": int(arr["amp"].shape[-1]) if outs else None, "n_ref": n_ref, "ref_k": list(pp.SHARED40),
        "preprocess": {"phase": PREPROCESS.phase, "bin_hz": PREPROCESS.bin_hz, "ref_bins": PREPROCESS.ref_bins,
                       "min_packets": PREPROCESS.min_packets},
        "train_hop_s": TRAIN_HOP_S, "eval_hop_s": EVAL_HOP_S, "eval_skip_s": EVAL_SKIP_S,
        "n_windows": int(t_end.size), "n_eval": int(eval_.sum()), "n_links": int(s.links.shape[0]),
        "segments": [[g.start_ns, g.end_ns, g.fine_label] for g in s.segments], "room_m": list(s.room_m),
        "t0_s": s.t0_s, "session_stats": s.stats,
        "links_used_mean": float(arr["link_mask"].mean()) if outs else 0.0,
        "bins_filled_mean": float(arr["sample_mask"][arr["link_mask"]].mean()) if outs and arr["link_mask"].any()
        else 0.0, "build_s": round(time.time() - t0, 2),
    }
    with open(meta_path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(meta_path + ".tmp", meta_path)
    return meta


def load(trial: str, length_s: float, root: str = ROOT) -> dict:
    """A built cache: its arrays (in memory) and `meta`; refuses one not labelled REAL."""
    arrays_path, meta_path = path(trial, length_s, root)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("label") != LABEL or meta.get("format") != _FORMAT:
        raise ValueError(f"{meta_path}: label {meta.get('label')!r} format {meta.get('format')!r}; want REAL/{_FORMAT}")
    with np.load(arrays_path) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = meta
    return out


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Build the Phase 9 REAL window caches (spec/phase9.md §3).")
    ap.add_argument("--length", type=float, nargs="+", required=True)
    ap.add_argument("--trials", nargs="+", default=list(oc.TRIALS))
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    for trial in args.trials:
        s = oc.load_session(oc.DEFAULT_ROOT, trial)
        for length in args.length:
            m = build(trial, length, root=args.root, workers=args.workers, overwrite=args.overwrite, session=s)
            print(json.dumps({k: m[k] for k in ("label", "trial", "length_s", "T", "n_windows", "n_eval",
                                                 "links_used_mean", "bins_filled_mean", "build_s")}), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
