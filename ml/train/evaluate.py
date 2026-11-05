"""Evaluate trained model v1 checkpoints on a split's evaluation windows
(spec/model.md decision 5, §4, §5): SIM.

    .venv/bin/python -m ml.train.evaluate --run RUN_DIR --split test
    .venv/bin/python -m ml.train.evaluate --run RUN_DIR --split test --subset tx0
    .venv/bin/python -m ml.train.evaluate --run R_s0 R_s1 R_s2 --split test --dataset roomsim-v1-hwoff --out DIR

Per run (one checkpoint, `best.pt` unless `--ckpt`): the model's predictions
on every evaluation window of the split (every second label tick with
`t_k >= 4 s`, the data side's `CacheDataset(mode="eval")`), optionally
on a link subset of spec/model.md §5.6 (`tx0`: the links whose transmitter
is node 0; `single01`: the link (0, 1); `single01_nogeo`: the same with the
geometry unknown), saved as the predictions contract (ml.train.metrics):
`scene`, `k` [N], `presence_prob` [N], `count_prob` [N, 4], `zone_prob`
[N, 9], `slot_exist` [N, 3], `slot_xy` [N, 3, 2] (m, room frame:
`centre * (w, d)`), `slot_sigma` [N, 3] (m), plus `slot_radius` (the
reported 95% radius, `max(2.448 sigma, 0.3)`), the cache `window` numbers and
the label "SIM", as `.npz`. Then every figure of spec/model.md §4 through
ml.train.metrics.evaluate (with several runs, the seeds' mean, std and the
scene-bootstrap interval of the seed-averaged figure) as JSON labelled SIM.
With `--baselines`, the evaluated dataset's own baselines (ml.train.baselines:
stored and recomputed WLM and RTI, prior, forest, MLP, as far as their files
exist; the missing ones are listed) are scored on the same windows and the
same resampled scenes, with every model-minus-baseline difference paired.

`--dataset` evaluates on another dataset's cache (the impairment cross
cases: an ON model on `roomsim-v1-hwoff` and back); the default is the
run's own. Outputs go to `<run>/eval/` for one run, `--out` for several.
With an unknown geometry (`single01_nogeo`) xy and zone are not scored and
`slot_xy` is in room-normalised units (the model is not given the room).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.train import model as mdl  # noqa: E402
from ml.train import train as tr  # noqa: E402

LABEL = "SIM"
SUBSETS = ("all", "tx0", "single01", "single01_nogeo")


def load_checkpoint(path: str, device="cpu"):
    """`(model, checkpoint dict)`; refuses a checkpoint not labelled SIM."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    if ck.get("label") != LABEL:
        raise ValueError(f"{path}: label {ck.get('label')!r}, not SIM")
    model = mdl.SaansModel(mdl.ModelConfig.from_dict(ck["model_config"]))
    model.load_state_dict(ck["model"])
    model.to(device).eval()
    return model, ck


def checkpoint_path(run: str, ckpt: str | None = None) -> str:
    if ckpt and os.path.isabs(ckpt):
        return ckpt
    if os.path.isfile(run):
        return run
    return os.path.join(run, ckpt or "best.pt")


def make_eval_source(run_cfg: dict, *, dataset: str | None, cache_dir: str | None, workers: int, batch: int,
                     synthetic: bool = False):
    """The run's data source, pointed at `dataset`/`cache_dir` when given."""
    cfg = dict(run_cfg)
    if dataset:
        cfg["dataset"] = dataset
        cfg["cache_dir"] = cache_dir
    elif cache_dir:
        cfg["cache_dir"] = cache_dir
    cfg["workers"] = workers
    cfg["eval_batch"] = batch
    if not synthetic:
        cfg["synthetic"] = None
    return tr.make_source(cfg)


@torch.no_grad()
def predict(model, loader, device, *, amp: bool = True, compiled: bool = False, pad_links: int = 0,
            batch: int | None = None, pack: bool = True) -> dict:
    """Run `model` over `loader`: the predictions contract (numpy) with
    the window ids, plus the eval-mode loss terms and the targets."""
    fwd = tr.Compiled(model, compiled and torch.device(device).type == "cuda")
    res = tr.run_eval(fwd, loader, torch.device(device), amp=amp and torch.device(device).type == "cuda",
                      weights=dict(mdl.DEFAULT_LOSS_WEIGHTS), matcher="enumerate",
                      n_links=pad_links if fwd.enabled else None, n_batch=batch if fwd.enabled else None,
                      pack=pack)
    return res


def contract_arrays(res: dict, subset: str) -> dict:
    """The arrays saved per run: the contract plus window numbers and the radius."""
    p, ids = res["predictions"], res["ids"]
    out = {"scene": ids["scene"], "k": ids["k"], **p, "slot_radius": mdl.radius95(p["slot_sigma"]).astype(np.float32)}
    if "window" in ids:
        out["window"] = ids["window"]
    out["geometry_known"] = res["targets"]["geometry_known"].astype(bool)
    out["label"] = np.array(LABEL)
    out["subset"] = np.array(subset)
    return out


def baseline_paths(cache) -> dict:
    """The data side's default baseline files for a cache's dataset and l
    (ml.train.baselines): the recomputed WLM/RTI
    (`<root>/<dataset>/locate_recomputed.npz`) and the variance-feature
    forest/MLP (`<root>/<dataset>/variance_baselines_<l>.npz`), looked up
    beside the cache, then under the data side's default and fallback roots
    (a phase cache shares its dataset's baselines: same windows)."""
    from ml.train import cache as cache_mod

    name = cache.meta["dataset"]
    dirs = [os.path.dirname(cache.dir)] + [os.path.join(r, name) for r in (cache_mod.DEFAULT_ROOT,
                                                                            cache_mod.FALLBACK_ROOT)]
    out = {}
    for key, fname in (("recomputed", "locate_recomputed.npz"),
                       ("variance", f"variance_baselines_{cache.length_s:g}.npz")):
        cands = [os.path.join(d, fname) for d in dirs]
        out[key] = next((c for c in cands if os.path.isfile(c)), cands[0])
    return out


def baseline_set(src, truth: dict, split: str, length_s: float, recomputed: str | None = None,
                 variance: str | None = None) -> tuple:
    """`(methods, included, missing)`: the evaluated dataset's baselines on
    `truth`'s windows (ml.train.baselines.baseline_methods): the stored WLM
    and RTI and the prior (all links), the recomputed WLM and RTI at the
    model's l on the same subset, the forest and MLP (all links), each when
    its file exists."""
    from ml.train import baselines as bl

    if not hasattr(src, "cache"):
        raise ValueError("baselines need the real cache")
    paths = baseline_paths(src.cache)
    recomputed = recomputed or paths["recomputed"]
    variance = variance or paths["variance"]
    missing = []
    rec = bl.load_recomputed(recomputed) if os.path.isfile(recomputed) else None
    if rec is None:
        missing.append({"recomputed": recomputed})
    var = None
    if os.path.isfile(variance):
        var = bl.load_predictions(variance)
        if split not in var:
            missing.append({"variance": f"{variance} has no {split!r} split"})
            var = None
    else:
        missing.append({"variance": variance})
    methods = bl.baseline_methods(src.cache, truth, recomputed=rec, length=length_s, variance=var,
                                  split=split if var is not None else None)
    return methods, sorted(methods), missing


def evaluate_runs(runs: list, split: str = "test", subset: str = "all", *, dataset: str | None = None,
                  cache_dir: str | None = None, ckpt: str | None = None, out_dir: str | None = None,
                  device: str = "cuda", workers: int = 4, batch: int = 256, n_boot: int = 2000, seed: int = 0,
                  compiled: bool = False, amp: bool = True, name: str | None = None, baselines: bool = False,
                  recomputed: str | None = None, variance: str | None = None) -> dict:
    """Evaluate one or more runs (seeds of one configuration) on the same
    windows; write `<name>.json` (and one `.npz` per run) to `out_dir`.
    With `baselines`, the evaluated dataset's baselines are scored on the
    same windows and resampled scenes, with the model as the reference of
    every paired difference. Returns the JSON dict."""
    if subset not in SUBSETS:
        raise ValueError(f"subset {subset!r} is not one of {SUBSETS}")
    if not runs:
        raise ValueError("no run to evaluate")
    dev = torch.device(device if (device != "cuda" or torch.cuda.is_available()) else "cpu")
    t0 = time.time()
    preds, entries, truth, src_desc = [], [], None, None
    run_dataset, first = None, None
    for run in runs:
        path = checkpoint_path(run, ckpt)
        model, ck = load_checkpoint(path, dev)
        rc = ck["run_config"]
        synthetic = rc.get("synthetic") is not None
        src = make_eval_source(rc, dataset=dataset, cache_dir=cache_dir, workers=workers, batch=batch,
                               synthetic=synthetic)
        loader = src.eval_loader(split, None if subset == "all" else subset)
        res = predict(model, loader, dev, amp=amp, compiled=compiled, pad_links=rc.get("pad_links") or 0,
                      batch=batch)
        arrays = contract_arrays(res, subset)
        this = (float(rc["window_s"]), bool(rc["phase"]), bool(rc["gain"]), rc["d"], rc["lr"], rc["dataset"])
        if first is None:
            first = this
            run_dataset = (rc["dataset"], rc["window_s"])
        elif this != first:
            raise ValueError(f"runs of different configurations: {first} and {this} (seeds of one config only)")
        desc = src.describe()
        src_desc = desc
        if truth is None:
            truth = _truth(src, res, split, subset, synthetic)
        preds.append(_contract(arrays))
        entries.append({"run_id": rc["run_id"], "checkpoint": os.path.abspath(path), "epoch": ck.get("epoch"),
                        "step": ck.get("step"), "val_loss": ck.get("val_loss"), "seed": rc["seed"],
                        "eval_loss": {k: v for k, v in res.items() if k in ("loss",) + tr.TERMS},
                        "arrays": arrays, "dataset_trained": rc["dataset"]})
    from ml.train import metrics as met

    methods = {"model": preds if len(preds) > 1 else preds[0]}
    included, missing, reference = [], [], None
    if baselines:
        base, included, missing = baseline_set(src, truth, split, float(run_dataset[1]), recomputed, variance)
        methods.update(base)
        reference = "model"
    metrics = met.evaluate(methods, truth, reference=reference, n_boot=n_boot, seed=seed)
    ds_name = (src_desc or {}).get("dataset", dataset or run_dataset[0])
    if name is None:
        base = os.path.basename(os.path.normpath(ds_name or "synthetic"))
        name = f"{base}_{split}_{subset}"
    if out_dir is None:
        if len(runs) != 1:
            raise ValueError("several runs need --out")
        out_dir = os.path.join(os.path.dirname(checkpoint_path(runs[0], ckpt)), "eval")
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for e in entries:
        f = os.path.join(out_dir, f"{name}__{e['run_id']}.npz" if len(entries) > 1 else f"{name}.npz")
        np.savez(f, **e.pop("arrays"))
        e["predictions"] = os.path.abspath(f)
        files.append(f)
    doc = {
        "label": LABEL, "spec": "spec/model.md §4, §5", "what": "model v1 on SIM evaluation windows",
        "split": split, "subset": subset, "dataset": ds_name, "length_s": run_dataset[1] if run_dataset else None,
        "trained_on": sorted({e["dataset_trained"] for e in entries}),
        "data": src_desc, "runs": entries, "n_seeds": len(entries),
        "baselines": {"included": included, "missing": missing} if baselines else None,
        "eval_windows": "every second label tick with t_k >= 4 s (spec/model.md decision 5)",
        "metrics": metrics, "time_s": round(time.time() - t0, 1),
    }
    tr._write_json(doc, os.path.join(out_dir, f"{name}.json"))
    doc["json"] = os.path.abspath(os.path.join(out_dir, f"{name}.json"))
    return doc


def _contract(arrays: dict) -> dict:
    from ml.train import metrics as met

    return met.predictions_from_slots(arrays["scene"], arrays["k"], arrays["presence_prob"], arrays["count_prob"],
                                      arrays["zone_prob"], arrays["slot_exist"], arrays["slot_xy"],
                                      arrays["slot_sigma"])


def _truth(src, res: dict, split: str, subset: str, synthetic: bool) -> dict:
    """The truth: from the cache (ml.train.metrics.truth_from_cache) or, for
    synthetic data, from the collected targets."""
    from ml.train import metrics as met

    if synthetic or not hasattr(src, "cache"):
        return tr.truth_of(res)
    return met.truth_from_cache(src.cache, split, subset=subset)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate model v1 runs on SIM evaluation windows (spec/model.md §4).")
    ap.add_argument("--run", nargs="+", required=True, help="run directories (or checkpoint files); several = seeds")
    ap.add_argument("--ckpt", default=None, help="checkpoint name inside each run (default best.pt)")
    ap.add_argument("--split", default="test", choices=("train", "val", "test"))
    ap.add_argument("--subset", default="all", choices=SUBSETS)
    ap.add_argument("--dataset", default=None, help="evaluate on this dataset's cache (default: the run's)")
    ap.add_argument("--cache-dir", default=None)
    ap.add_argument("--out", default=None, help="output directory (default <run>/eval for one run)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0, help="the bootstrap's seed")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--baselines", action="store_true",
                    help="score the evaluated dataset's baselines beside the model (paired differences)")
    ap.add_argument("--recomputed", default=None, help="recomputed WLM/RTI file (default: the data side's)")
    ap.add_argument("--variance", default=None, help="forest/MLP predictions file (default: the data side's)")
    args = ap.parse_args(argv)
    doc = evaluate_runs(args.run, args.split, args.subset, dataset=args.dataset, cache_dir=args.cache_dir,
                        ckpt=args.ckpt, out_dir=args.out, device=args.device, workers=args.workers,
                        batch=args.batch, n_boot=args.n_boot, seed=args.seed, compiled=args.compile, name=args.name,
                        baselines=args.baselines, recomputed=args.recomputed, variance=args.variance)
    flat = {}
    from ml.train import metrics as met

    figs = met.flat_figures(doc["metrics"]["methods"]["model"])
    for key, short in tr.HEADLINE.items():
        if key in figs:
            f = figs[key]
            flat[short] = {k: f[k] for k in ("value", "mean", "std", "ci95") if k in f}
    print(json.dumps({"label": LABEL, "json": doc["json"], "split": doc["split"], "subset": doc["subset"],
                      "dataset": doc["dataset"], "n_seeds": doc["n_seeds"], "baselines": doc["baselines"],
                      "headline": flat}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
