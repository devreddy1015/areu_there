"""The Phase 8 sweep and final runs (spec/model.md §5): SIM.

    .venv/bin/python -m ml.train.sweep --plan                 # list the 27 runs and the caches they need
    .venv/bin/python -m ml.train.sweep --concurrency 6        # train them (resume-safe), then the val table
    .venv/bin/python -m ml.train.sweep --collect [--intervals] # only the val table (with scene-bootstrap CIs)
    .venv/bin/python -m ml.train.sweep --final [--choose RUN_ID] --concurrency 6

**Sweep** (§5.1): lr {3e-4, 1e-3, 3e-3} x width D {64, 128, 256} x window
l {1, 2, 4} s, seed 0, on roomsim-v1, each a `ml.train.train` process; up
to `--concurrency` (default 9) at once on the one GPU, the costliest
first (GPU_MS). Measured: 9 runs keep the GPU about 97% busy; the sweep
needs about 7,200 GPU-seconds at 30 epochs a run (6,500 for training, 12%
more for the per-epoch validation), so about 2-2.5 h of wall time, less
when early stopping ends runs sooner. Resume-safe: a run
with `done.json` is skipped, a run with only `last.pt` resumes at its last
finished epoch (ml.train.train). `--collect` writes `sweep_table.json` and
`sweep_table.md` (validation only; selection by the validation loss) under
the sweep directory; `--intervals` adds each run's val figures with the
scene-bootstrap interval (ml.train.evaluate on val).

**Final** (§5.2-§5.6), for the chosen configuration (the lowest val loss,
or `--choose`) x seeds {0, 1, 2}:
- `main`: roomsim-v1; evaluated on its test split, on the link subsets
  `tx0`, `single01`, `single01_nogeo` (§5.6), and on roomsim-v1-hwoff's
  test split (the cross case);
- `hwoff`: trained on roomsim-v1-hwoff (§5.3); evaluated on its own test
  split and on roomsim-v1's (cross);
- `phase`: phase on, 2 x 44 more channels (§5.4), which needs a phase cache
  (`ml.train.cache ... --phase`, under PHASE_CACHE_ROOT);
- `gain`: `gain_db` minus its window mean as one more channel (§5.5; every
  cache stores `gain_db`).
Each evaluation aggregates the three seeds (mean, std and the bootstrap
interval of the seed-averaged figure; ml.train.metrics). A missing cache is
reported with the command that builds it; nothing is trained on data that
does not exist. The baselines' comparison (WLM, RTI, forest, MLP, prior on
the same windows) is the data side's (ml.train.baselines).
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import subprocess
import sys
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.train import train as tr  # noqa: E402

LABEL = "SIM"
DATASETS_ROOT = "/home/Aman/are_u_theone/datasets"
SWEEP_ROOT = os.path.join(tr.RUNS_ROOT, "sweep")
FINAL_ROOT = os.path.join(tr.RUNS_ROOT, "final")
CACHE_ROOT = "/dev/shm/saans-phase8-cache"
PHASE_CACHE_ROOT = "/dev/shm/saans-phase8-cache-phase"
GRID = {"lr": (3e-4, 1e-3, 3e-3), "d": (64, 128, 256), "window_s": (1.0, 2.0, 4.0)}
SEEDS = (0, 1, 2)
#: Measured GPU kernel time per train step (ms; compiled, fp16, packed links, B = 128, training link subsets;
#: Quadro RTX 8000, 2026-10-25): the cost of a run when runs share the GPU. For scheduling and estimates only.
GPU_MS = {(64, 1.0): 5.0, (128, 1.0): 7.1, (256, 1.0): 12.5, (64, 2.0): 5.8, (128, 2.0): 8.2,
          (256, 2.0): 14.9, (64, 4.0): 7.0, (128, 4.0): 10.3, (256, 4.0): 19.1}
#: Measured (packed links, 2026-10-25, the CPU shared with other jobs): 9 runs at once, one of each (D, l),
#: kept the GPU 97.5% busy (6 runs: 88%); 3 copies of the largest run were GPU-bound alone. GPU memory per
#: run 0.4-1.5 GiB allocated, 0.8-5.2 GiB reserved (about 23 GiB for 9 with their CUDA contexts).
DEFAULT_CONCURRENCY = 9


def cache_dir(dataset: str, window_s: float, phase: bool = False) -> str:
    root = PHASE_CACHE_ROOT if phase else CACHE_ROOT
    return os.path.join(root, os.path.basename(os.path.normpath(dataset)), f"{float(window_s):g}")


def build_command(dataset: str, window_s: float, phase: bool = False) -> str:
    cmd = (f".venv/bin/python -m ml.train.cache {os.path.join(DATASETS_ROOT, dataset)} --length {float(window_s):g}"
           f" --root {PHASE_CACHE_ROOT if phase else CACHE_ROOT}")
    return cmd + (" --phase" if phase else "")


def sweep_plan(dataset: str = "roomsim-v1", seed: int = 0, out_root: str = SWEEP_ROOT, base: dict | None = None) -> list:
    """The 27 sweep runs as resolved train configs (seed 0, §5.1)."""
    runs = []
    for window_s in GRID["window_s"]:
        for d in GRID["d"]:
            for lr in GRID["lr"]:
                cfg = dict(base or {}, dataset=dataset, window_s=window_s, d=d, lr=lr, seed=seed, out_root=out_root,
                           cache_dir=cache_dir(dataset, window_s))
                cfg.pop("run_id", None)
                runs.append(tr.resolve_config(cfg))
    return runs


def run_dir(cfg: dict) -> str:
    return os.path.join(cfg["out_root"], cfg["run_id"])


def is_done(cfg: dict) -> bool:
    return os.path.isfile(os.path.join(run_dir(cfg), "done.json"))


def pending(runs: list) -> list:
    return [r for r in runs if not is_done(r)]


def missing_caches(runs: list) -> list:
    """The caches the runs need that do not exist: (path, build command)."""
    out = {}
    for r in runs:
        if r.get("synthetic") is not None:
            continue
        path = r["cache_dir"] or cache_dir(r["dataset"], r["window_s"], r["phase"])
        if not os.path.isfile(os.path.join(path, "meta.json")):
            out[path] = build_command(r["dataset"], r["window_s"], r["phase"])
    return sorted(out.items())


def _cost(cfg: dict) -> float:
    return GPU_MS.get((cfg["d"], float(cfg["window_s"])), 15.0) * (1.5 if cfg["phase"] else 1.0)


def launch(runs: list, concurrency: int = DEFAULT_CONCURRENCY, log_dir: str | None = None, env: dict | None = None,
           python: str = sys.executable, poll_s: float = 2.0, verbose: bool = True) -> dict:
    """Train every unfinished run of `runs`, `concurrency` processes at a
    time, the costliest first (each process: `python -m ml.train.train
    --config <run>/launch.json`, its output in `log_dir/<run_id>.log`).
    Returns `{"launched", "finished", "failed": {run_id: exit code}}`."""
    runs = [tr.resolve_config(r) for r in runs]
    todo = sorted(pending(runs), key=_cost, reverse=True)
    res = {"launched": [], "finished": [], "failed": {}}
    if not todo:
        return res
    log_dir = log_dir or os.path.join(os.path.dirname(run_dir(todo[0])), "logs")
    os.makedirs(log_dir, exist_ok=True)
    penv = dict(os.environ, PYTHONUNBUFFERED="1", **(env or {}))
    penv["PYTHONPATH"] = ROOT + (os.pathsep + penv["PYTHONPATH"] if penv.get("PYTHONPATH") else "")
    running = {}
    t0 = time.time()
    queue = list(todo)
    try:
        _schedule(queue, running, res, concurrency, log_dir, penv, python, poll_s, verbose, t0)
    except BaseException:
        for rid, (proc, log, _cfg) in running.items():           # interrupted: stop the children too
            proc.terminate()                                      # (each resumes from last.pt next time)
        for rid, (proc, log, _cfg) in running.items():
            with contextlib.suppress(Exception):
                proc.wait(timeout=60)
            log.close()
        raise
    res["wall_s"] = time.time() - t0
    return res


def _schedule(queue, running, res, concurrency, log_dir, penv, python, poll_s, verbose, t0):
    while queue or running:
        while queue and len(running) < max(1, concurrency):
            cfg = queue.pop(0)
            os.makedirs(run_dir(cfg), exist_ok=True)
            path = os.path.join(run_dir(cfg), "launch.json")
            tr._write_json(cfg, path)
            log = open(os.path.join(log_dir, f"{cfg['run_id']}.log"), "a")
            proc = subprocess.Popen([python, "-m", "ml.train.train", "--config", path], cwd=ROOT, env=penv,
                                    stdout=log, stderr=subprocess.STDOUT)
            running[cfg["run_id"]] = (proc, log, cfg)
            res["launched"].append(cfg["run_id"])
            if verbose:
                print(f"[sweep SIM] start {cfg['run_id']} ({len(running)} running, {len(queue)} queued, "
                      f"{time.time() - t0:.0f} s)", flush=True)
        time.sleep(poll_s)
        for rid, (proc, log, cfg) in list(running.items()):
            code = proc.poll()
            if code is None:
                continue
            log.close()
            del running[rid]
            if code == 0 and is_done(cfg):
                res["finished"].append(rid)
            else:
                res["failed"][rid] = code
            if verbose:
                print(f"[sweep SIM] {'done' if code == 0 else f'FAILED ({code})'} {rid} "
                      f"({time.time() - t0:.0f} s)", flush=True)


def _best_epoch_record(cfg: dict) -> dict | None:
    path = os.path.join(run_dir(cfg), "log.jsonl")
    if not os.path.isfile(path):
        return None
    best = None
    with open(path) as f:
        for ln in f:
            if not ln.strip():
                continue
            rec = json.loads(ln)
            if best is None or rec["val"]["loss"] < best["val"]["loss"]:
                best = rec
    return best


def collect(runs: list, out_dir: str | None = None, intervals: dict | None = None) -> dict:
    """The validation table of finished runs, sorted by val loss (best
    first): config, best epoch, val loss and its terms, the val summary at
    the best epoch, steps, time, throughput and GPU memory. SIM, val only."""
    rows = []
    runs = [tr.resolve_config(r) for r in runs]
    for cfg in runs:
        if not is_done(cfg):
            continue
        with open(os.path.join(run_dir(cfg), "done.json")) as f:
            done = json.load(f)
        rec = _best_epoch_record(cfg) or {}
        row = {"run_id": cfg["run_id"], "lr": cfg["lr"], "d": cfg["d"], "window_s": cfg["window_s"],
               "seed": cfg["seed"], "dataset": cfg["dataset"], "best_val_loss": done.get("best_val_loss"),
               "best_epoch": done.get("best_epoch"), "stop": done.get("stop"), "steps": done.get("steps"),
               "elapsed_s": done.get("elapsed_s"), "n_params": done.get("n_params"), "compiled": done.get("compiled"),
               "val_terms": {k: v for k, v in rec.get("val", {}).items() if k != "n_windows"},
               "val_metrics": rec.get("val_metrics"), "samples_per_s": rec.get("throughput", {}).get("samples_per_s"),
               "gpu_max_allocated_mb": rec.get("gpu", {}).get("max_allocated_mb")}
        if intervals and cfg["run_id"] in intervals:
            row["val_intervals"] = intervals[cfg["run_id"]]
        rows.append(row)
    rows.sort(key=lambda r: (r["best_val_loss"] if r["best_val_loss"] is not None else float("inf")))
    table = {"label": LABEL, "spec": "spec/model.md §5.1", "split": "val",
             "selection": "lowest validation loss (spec/model.md §5.1); test untouched",
             "n_runs": len(runs), "n_finished": len(rows), "rows": rows,
             "best": rows[0]["run_id"] if rows else None}
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        tr._write_json(table, os.path.join(out_dir, "sweep_table.json"))
        with open(os.path.join(out_dir, "sweep_table.md"), "w") as f:
            f.write(markdown_table(table))
    return table


def markdown_table(table: dict) -> str:
    lines = ["Phase 8 sweep, validation split (SIM). Selection by the validation loss; test untouched.", "",
             "| rank | l (s) | D | lr | val loss | best epoch | presence acc | count acc | zone F1 | xy median (m) "
             "| xy recall | coverage | samples/s |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]

    def fmt(v, nd=3):
        return "" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))

    for i, r in enumerate(table["rows"]):
        m = r.get("val_metrics") or {}
        iv = r.get("val_intervals") or {}

        def cell(key, nd=3):
            s = fmt(m.get(key), nd)
            if key in iv and iv[key].get("ci95") and None not in iv[key]["ci95"]:
                lo, hi = iv[key]["ci95"]
                s += f" [{lo:.{nd}f}, {hi:.{nd}f}]"
            return s

        lines.append(f"| {i + 1} | {r['window_s']:g} | {r['d']} | {r['lr']:g} | {fmt(r['best_val_loss'], 4)} | "
                     f"{r['best_epoch']} | {cell('presence_acc')} | {cell('count_acc')} | {cell('zone_f1')} | "
                     f"{cell('xy_median_m', 2)} | {cell('xy_recall')} | {cell('radius_coverage')} | "
                     f"{fmt(r['samples_per_s'], 0)} |")
    return "\n".join(lines) + "\n"


def val_intervals(runs: list, n_boot: int = 2000, workers: int = 4, device: str = "cuda") -> dict:
    """Each finished run's headline val figures with the scene-bootstrap
    interval (ml.train.evaluate on val; written to `<run>/eval/`)."""
    from ml.train import evaluate as ev
    from ml.train import metrics as met

    out = {}
    for cfg in runs:
        if not is_done(cfg):
            continue
        doc = ev.evaluate_runs([run_dir(cfg)], "val", "all", device=device, workers=workers, n_boot=n_boot)
        figs = met.flat_figures(doc["metrics"]["methods"]["model"])
        out[cfg["run_id"]] = {short: {k: figs[key][k] for k in ("value", "ci95") if k in figs[key]}
                              for key, short in tr.HEADLINE.items() if key in figs}
    return out


# --------------------------------------------------------------------------
# final runs (§5.2-§5.6)
# --------------------------------------------------------------------------


def chosen_config(table: dict, runs: list, choose: str | None = None) -> dict:
    rid = choose or table.get("best")
    if not rid:
        raise SystemExit("no finished sweep run to choose from (run the sweep first, or pass --choose)")
    for cfg in runs:
        if cfg["run_id"] == rid:
            return cfg
    raise SystemExit(f"--choose {rid!r} is not a sweep run")


def final_plan(best: dict, seeds=SEEDS, out_root: str = FINAL_ROOT) -> dict:
    """The final trainings (name -> list of configs, one per seed) and the
    evaluations (each: the runs, the dataset evaluated, the split, the
    subset, an output name)."""
    keep = {k: best[k] for k in ("lr", "d", "window_s", "epochs", "batch", "weight_decay", "warmup_steps",
                                  "grad_clip", "patience", "dropout", "heads", "layers", "ticks_per_scene", "matcher",
                                  "loss_weights", "subset_rules", "compile", "amp", "workers", "eval_batch",
                                  "pad_links")}
    l = best["window_s"]
    variants = {
        "main": {"dataset": "roomsim-v1"},
        "hwoff": {"dataset": "roomsim-v1-hwoff"},
        "phase": {"dataset": "roomsim-v1", "phase": True},
        "gain": {"dataset": "roomsim-v1", "gain": True},
    }
    train = {}
    for name, v in variants.items():
        train[name] = []
        for s in seeds:
            cfg = dict(keep, **v, seed=s, out_root=out_root,
                       cache_dir=cache_dir(v["dataset"], l, v.get("phase", False)))
            cfg["run_id"] = f"{name}_l{l:g}_d{keep['d']}_lr{keep['lr']:g}_s{s}"
            train[name].append(tr.resolve_config(cfg))
    evals = []
    on, off = "roomsim-v1", "roomsim-v1-hwoff"
    for subset in ("all", "tx0", "single01", "single01_nogeo"):
        evals.append({"name": f"main_on_{subset}", "train": "main", "dataset": on, "subset": subset})
    evals += [
        {"name": "main_on_hwoff_all", "train": "main", "dataset": off, "subset": "all"},
        {"name": "hwoff_on_hwoff_all", "train": "hwoff", "dataset": off, "subset": "all"},
        {"name": "hwoff_on_on_all", "train": "hwoff", "dataset": on, "subset": "all"},
        {"name": "phase_on_all", "train": "phase", "dataset": on, "subset": "all", "phase": True},
        {"name": "gain_on_all", "train": "gain", "dataset": on, "subset": "all"},
    ]
    for e in evals:
        e["split"] = "test"
        e["cache_dir"] = cache_dir(e["dataset"], l, e.get("phase", False))
    return {"label": LABEL, "chosen": best["run_id"], "window_s": l, "train": train, "evals": evals,
            "out_root": out_root}


def run_final(plan: dict, concurrency: int, n_boot: int = 2000, workers: int = 4, verbose: bool = True) -> dict:
    """Train what the caches allow, then every evaluation whose runs finished."""
    all_runs = [c for runs in plan["train"].values() for c in runs]
    missing = missing_caches(all_runs) + [(e["cache_dir"], build_command(e["dataset"], plan["window_s"],
                                                                         e.get("phase", False)))
                                          for e in plan["evals"]
                                          if not os.path.isfile(os.path.join(e["cache_dir"], "meta.json"))]
    missing = sorted(set(missing))
    runnable = [c for c in all_runs if os.path.isfile(os.path.join(c["cache_dir"], "meta.json"))]
    report = {"label": LABEL, "chosen": plan["chosen"], "missing_caches": [{"cache": p, "build": cmd}
                                                                           for p, cmd in missing]}
    if verbose and missing:
        for p, cmd in missing:
            print(f"[final SIM] missing cache {p}: build with `{cmd}`", flush=True)
    report["train"] = launch(runnable, concurrency=concurrency, log_dir=os.path.join(plan["out_root"], "logs"),
                             verbose=verbose)
    from ml.train import evaluate as ev

    eval_dir = os.path.join(plan["out_root"], "eval")
    report["evals"] = {}
    for e in plan["evals"]:
        runs = plan["train"][e["train"]]
        if not all(is_done(c) for c in runs):
            report["evals"][e["name"]] = {"skipped": "training not finished (or its cache is missing)"}
            continue
        if not os.path.isfile(os.path.join(e["cache_dir"], "meta.json")):
            report["evals"][e["name"]] = {"skipped": f"missing cache {e['cache_dir']}"}
            continue
        out_json = os.path.join(eval_dir, f"{e['name']}.json")
        if os.path.isfile(out_json):
            report["evals"][e["name"]] = {"json": out_json, "cached": True}
            continue
        doc = ev.evaluate_runs([run_dir(c) for c in runs], e["split"], e["subset"], dataset=e["dataset"],
                               cache_dir=e["cache_dir"], out_dir=eval_dir, workers=workers, n_boot=n_boot,
                               name=e["name"], baselines=True)
        report["evals"][e["name"]] = {"json": doc["json"], "baselines": doc["baselines"]}
    tr._write_json(report, os.path.join(plan["out_root"], "final_report.json"))
    return report


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="The Phase 8 sweep and final runs (spec/model.md §5). SIM.")
    ap.add_argument("--plan", action="store_true", help="print the sweep plan and missing caches, run nothing")
    ap.add_argument("--collect", action="store_true", help="only write the val table")
    ap.add_argument("--intervals", action="store_true", help="with --collect: scene-bootstrap intervals on val")
    ap.add_argument("--final", action="store_true", help="the chosen config x 3 seeds and the ablations")
    ap.add_argument("--choose", default=None, help="with --final: a sweep run id (default: lowest val loss)")
    ap.add_argument("--concurrency", type=int, default=DEFAULT_CONCURRENCY)
    ap.add_argument("--sweep-root", default=SWEEP_ROOT)
    ap.add_argument("--final-root", default=FINAL_ROOT)
    ap.add_argument("--workers", type=int, default=4, help="data loader workers per run")
    ap.add_argument("--n-boot", type=int, default=2000)
    args = ap.parse_args(argv)
    runs = sweep_plan(out_root=args.sweep_root, base={"workers": args.workers})
    if args.plan:
        miss = missing_caches(runs)
        print(json.dumps({"label": LABEL, "runs": [r["run_id"] for r in runs],
                          "pending": [r["run_id"] for r in pending(runs)],
                          "missing_caches": [{"cache": p, "build": c} for p, c in miss]}, indent=1))
        return 0
    if args.final:
        table = collect(runs, args.sweep_root)
        best = chosen_config(table, runs, args.choose)
        plan = final_plan(best, out_root=args.final_root)
        report = run_final(plan, args.concurrency, n_boot=args.n_boot, workers=args.workers)
        print(json.dumps({k: report[k] for k in ("chosen", "missing_caches", "evals")}, indent=1))
        return 0
    if not args.collect:
        miss = missing_caches(runs)
        if miss:
            for p, cmd in miss:
                print(f"[sweep SIM] missing cache {p}: build with `{cmd}`", file=sys.stderr)
            return 2
        res = launch(runs, concurrency=args.concurrency, log_dir=os.path.join(args.sweep_root, "logs"))
        print(json.dumps({"label": LABEL, **{k: res[k] for k in ("finished", "failed")},
                          "wall_s": res.get("wall_s")}, indent=1))
    iv = val_intervals(runs, n_boot=args.n_boot, workers=args.workers) if args.intervals else None
    table = collect(runs, args.sweep_root, intervals=iv)
    print(markdown_table(table))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
