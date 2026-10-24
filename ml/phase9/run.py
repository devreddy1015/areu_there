"""Phase 9 runner (spec/phase9.md): folds, baselines, models, scoring.

    .venv/bin/python -m ml.phase9.run baselines --length 2      # DSP, WLM, variance forest/MLP, prior
    .venv/bin/python -m ml.phase9.run models --length 2 --pretrained RUN_DIR [RUN_DIR ...]
    .venv/bin/python -m ml.phase9.run report --length 2         # ml/results/phase9/*.json

Predictions are saved per (method, test session, fraction, seed) under
`RUNS/predictions/` as `.npz` (presence_prob, count_prob, auc_score) on the
session's evaluation windows, and scored together by `report`. Every result
carries its label: REAL for methods trained (or calibrated) on REAL data
only, SIM->REAL for anything a SIM-trained model produced.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

from ml import opencsi as oc
from ml.phase9 import baselines_real as br
from ml.phase9 import metrics_real as mr
from ml.phase9 import real_cache as rc
from ml.phase9 import real_data as rdm

__all__ = ["FOLDS", "FRACTIONS", "SEEDS", "RUNS", "RESULTS", "fold_of", "load_caches", "eval_windows",
           "run_baselines", "report"]

#: spec/phase9.md §4: leave one room out; the C6 session is test-only (fold A).
FOLDS = {
    "A": {"test": ["envA_S3", "envA_C6"], "train": ["envB_S3prime", "envC_C3"]},
    "B": {"test": ["envB_S3prime"], "train": ["envA_S3", "envC_C3"]},
    "C": {"test": ["envC_C3"], "train": ["envA_S3", "envB_S3prime"]},
}
FRACTIONS = (0.01, 0.1, 1.0)
SEEDS = (0, 1, 2)
RUNS = "/home/Aman/are_u_theone/runs/phase9"
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "phase9")


def fold_of(trial: str) -> str:
    for name, f in FOLDS.items():
        if trial in f["test"]:
            return name
    raise KeyError(trial)


def load_caches(length_s: float) -> dict:
    return {t: rc.load(t, length_s) for t in oc.TRIALS}


def eval_windows(cache: dict) -> np.ndarray:
    return np.flatnonzero(cache["eval"])


def _pred_path(method: str, trial: str, length_s: float, fraction=None, seed=None) -> str:
    parts = [method, trial, f"l{length_s:g}"]
    if fraction is not None:
        parts.append(f"f{fraction:g}")
    if seed is not None:
        parts.append(f"s{seed}")
    return os.path.join(RUNS, "predictions", "_".join(parts) + ".npz")


def save_pred(path: str, pred: dict, *, label: str, extra: dict | None = None) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    arrays = {"presence_prob": np.asarray(pred["presence_prob"], np.float64)}
    if pred.get("count_prob") is not None:
        arrays["count_prob"] = np.asarray(pred["count_prob"], np.float64)
    if pred.get("auc_score") is not None:
        arrays["auc_score"] = np.asarray(pred["auc_score"], np.float64)
    meta = {"label": label, **(extra or {})}
    np.savez(path, meta=json.dumps(meta), **arrays)


def load_pred(path: str) -> tuple:
    with np.load(path) as z:
        meta = json.loads(str(z["meta"]))
        pred = {"presence_prob": z["presence_prob"], "count_prob": z["count_prob"] if "count_prob" in z else None}
        if "auc_score" in z:
            pred["auc_score"] = z["auc_score"]
    return pred, meta


# --------------------------------------------------------------------------
# baselines
# --------------------------------------------------------------------------


def _train_features(caches: dict, trials: list, fraction: float, seed: int) -> tuple:
    xs, ps, cs = [], [], []
    for t in trials:
        c = caches[t]
        w = rdm.fraction_positions(c, fraction, seed)
        xs.append(br.variance_features(c["sigma"][w], c["link_mask"][w]))
        ps.append(c["presence"][w].astype(np.int64))
        cs.append(c["count"][w].astype(np.int64))
    return np.concatenate(xs), np.concatenate(ps), np.concatenate(cs)


def run_baselines(length_s: float, *, workers: int = 56, log=print) -> None:
    """DSP and WLM per test session (no training: calibrated on its first
    80 s); variance forest and MLP and the prior per fold, fraction and seed
    (trained on the fold's training sessions)."""
    caches = load_caches(length_s)
    for trial in oc.TRIALS:
        c = caches[trial]
        ev = eval_windows(c)
        ends = c["t_end_ns"][ev]
        s = oc.load_session(oc.DEFAULT_ROOT, trial)
        t0 = time.time()
        dsp = br.dsp_presence(s, ends, workers=workers)
        save_pred(_pred_path("dsp_majority", trial, length_s), dsp, label="REAL",
                  extra={"uses_calibration": True, "calib_s": br.CALIB_S})
        os.makedirs(os.path.join(RUNS, "predictions"), exist_ok=True)
        np.savez(os.path.join(RUNS, "predictions", f"dsp_links_{trial}_l{length_s:g}.npz"),
                 occupied=dsp["link_occupied"], ratio=dsp["link_ratio"])
        wlm = br.wlm_gate(s, ends, length_s)
        save_pred(_pred_path("wlm_gate", trial, length_s), wlm, label="REAL",
                  extra={"uses_calibration": True, "calib_s": br.CALIB_S, "rate_hz": wlm["rate_hz"],
                         "calibrated_links": wlm["calibrated_links"]})
        log(f"[phase9 REAL] {trial} l={length_s:g}: DSP and WLM in {time.time() - t0:.1f} s")
    for fold, f in FOLDS.items():
        for frac in FRACTIONS:
            for seed in SEEDS:
                x, p, cnt = _train_features(caches, f["train"], frac, seed)
                for trial in f["test"]:
                    c = caches[trial]
                    ev = eval_windows(c)
                    xe = br.variance_features(c["sigma"][ev], c["link_mask"][ev])
                    fits = br.fit_variance(x, p, cnt, xe, seed)
                    for name, pred in fits.items():
                        save_pred(_pred_path(f"var_{name}", trial, length_s, frac, seed), pred, label="REAL",
                                  extra={"fold": fold, "train": f["train"], "n_train": int(x.shape[0])})
                    save_pred(_pred_path("prior", trial, length_s, frac, seed), br.prior(p, cnt, ev.size),
                              label="REAL", extra={"fold": fold, "n_train": int(x.shape[0])})
            log(f"[phase9 REAL] fold {fold} f={frac:g}: variance baselines and prior, 3 seeds")


def sim_variance_features(cache_dir: str, split: str = "train", min_t: float = 4.0) -> tuple:
    """Geometry-free variance features (`baselines_real.variance_features`'s
    nine) of a SIM cache's windows of `split` with `t_k >= min_t`, with
    their presence and count. SIM."""
    from ml.train import cache as cache_mod
    from ml.train.baselines import _stats
    c = cache_mod.open_cache(cache_dir)
    windows = c.window_indices(split, min_t=min_t)
    win = c.windows
    srow = win["scene_row"][windows]
    feats = np.zeros((windows.size, 9))
    for sr in np.unique(srow):
        sel = np.flatnonzero(srow == sr)
        ws = windows[sel]
        n = int(c.scenes["n_links"][sr])
        idx = (win["row_offset"][ws][:, None] + np.arange(n)[None, :]).ravel()
        sl = np.asarray(c.rows["sigma"][idx], np.float32).astype(np.float64).reshape(ws.size, n, -1).mean(-1)
        used = np.asarray(c.rows["link_mask"][idx], bool).reshape(ws.size, n)
        feats[sel, :8] = _stats(sl, used)
        feats[sel, 8] = used.sum(1)
    return feats, win["presence"][windows].astype(np.int64), win["count"][windows].astype(np.int64)


def run_sim_variance(length_s: float, sim_cache_dir: str, *, log=print) -> None:
    """spec/phase9.md §5.3, SIM-trained: forest and MLP fitted on the SIM
    (shared40) cache's train windows, applied to every REAL session (SIM->REAL)."""
    x, p, cnt = sim_variance_features(sim_cache_dir)
    caches = load_caches(length_s)
    for seed in SEEDS:
        for trial in oc.TRIALS:
            c = caches[trial]
            ev = eval_windows(c)
            xe = br.variance_features(c["sigma"][ev], c["link_mask"][ev])
            fits = br.fit_variance(x, p, cnt, xe, seed)
            for name, pred in fits.items():
                save_pred(_pred_path(f"simvar_{name}", trial, length_s, seed=seed), pred, label="SIM->REAL",
                          extra={"train": "SIM " + sim_cache_dir, "n_train": int(x.shape[0])})
        log(f"[phase9 SIM->REAL] SIM-trained variance baselines, seed {seed}")


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------


def _collect(trial: str, length_s: float) -> tuple:
    """Every saved method for one test session: `{name: [pred per seed]}` and their labels."""
    methods, labels = {}, {}
    pdir = os.path.join(RUNS, "predictions")
    for fn in sorted(os.listdir(pdir)):
        if not fn.endswith(".npz") or fn.startswith("dsp_links_"):
            continue
        stem = fn[:-4]
        if f"_{trial}_l{length_s:g}" not in stem:
            continue
        method, rest = stem.split(f"_{trial}_l{length_s:g}")
        frac = None
        for part in rest.strip("_").split("_"):
            if part.startswith("f"):
                frac = part[1:]
        name = method if frac is None else f"{method}@{frac}"
        pred, meta = load_pred(os.path.join(pdir, fn))
        methods.setdefault(name, []).append(pred)
        if labels.setdefault(name, meta["label"]) != meta["label"]:
            raise ValueError(f"{name}: mixed labels")
    return methods, labels


def _dsp_per_link(trial: str, length_s: float, truth: dict) -> dict:
    path = os.path.join(RUNS, "predictions", f"dsp_links_{trial}_l{length_s:g}.npz")
    if not os.path.exists(path):
        return {}
    with np.load(path) as z:
        occ, ratio = z["occupied"], z["ratio"]
    figs = []
    one = np.ones((1, occ.shape[1]))
    for l in range(occ.shape[0]):
        if not np.isfinite(ratio[l]).any():
            continue
        pred = {"presence_prob": occ[l].astype(np.float64), "auc_score": np.nan_to_num(ratio[l], nan=0.0)}
        f = mr.figures(pred, truth, one)
        figs.append({k: float(v[0]) for k, v in f.items()})
    out = {}
    for key in ("presence|balanced_accuracy", "presence|auc", "presence|false_alarm", "presence|miss"):
        v = np.array([f[key] for f in figs], np.float64)
        v = v[np.isfinite(v)]
        if v.size:
            out[key] = {"median": float(np.median(v)), "q25": float(np.percentile(v, 25)),
                        "q75": float(np.percentile(v, 75)), "best": float(v.max()), "worst": float(v.min()),
                        "n_links": int(v.size)}
    return {"label": "REAL", "note": "each link as a one-link deployment: the spread over links, no interval",
            "figures": out}


def report(length_s: float, *, n_boot: int = mr.N_BOOT, out_dir: str = RESULTS, log=print) -> dict:
    caches = load_caches(length_s)
    os.makedirs(out_dir, exist_ok=True)
    allres = {}
    for trial in oc.TRIALS:
        c = caches[trial]
        ev = eval_windows(c)
        truth = rdm.truth(c, ev)
        methods, labels = _collect(trial, length_s)
        if not methods:
            continue
        res = mr.evaluate(methods, truth, label_of=labels, n_boot=n_boot)
        res["dsp_per_link"] = _dsp_per_link(trial, length_s, truth)
        res["_session"] = {"trial": trial, "fold": fold_of(trial), "label_note": "figures on REAL recordings",
                           "n_eval": int(ev.size), "length_s": length_s,
                           "windows": {"empty": int((~truth["presence"]).sum()),
                                       "still": int((truth["presence"] & ~truth["moving"]).sum()),
                                       "moving": int(truth["moving"].sum())}}
        allres[trial] = res
        with open(os.path.join(out_dir, f"{trial}_l{length_s:g}.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
        log(f"[phase9] {trial}: {len(methods)} methods scored")
    return allres


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Phase 9 runner (spec/phase9.md).")
    ap.add_argument("what", choices=("baselines", "simvar", "report"))
    ap.add_argument("--length", type=float, required=True)
    ap.add_argument("--sim-cache", default=None, help="the shared40 SIM cache (simvar)")
    ap.add_argument("--n-boot", type=int, default=mr.N_BOOT)
    args = ap.parse_args(argv)
    if args.what == "baselines":
        run_baselines(args.length)
    elif args.what == "simvar":
        run_sim_variance(args.length, args.sim_cache or f"/dev/shm/saans-phase8-cache/roomsim-v1-shared40/{args.length:g}")
    else:
        report(args.length, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
