"""Run integrity (spec/model.md §2, §5; spec/phase9.md §4.1): the checks
that keep a run which only looks finished from becoming a result. SIM.

A sweep runs for hours, unattended, and its outputs are read by later
steps that cannot tell a good run from a broken one:
- a run whose fp16 steps are almost all skipped (the gradient scaler drops
  every non-finite step) or whose validation loss is non-finite has stopped
  learning, yet would still reach `done.json` (`skip_problem`,
  `best_problem`);
- a run id names only the dataset, window, width, lr, seed and the
  phase/gain ablations, so a finished run of another epoch count, batch,
  patience, dropout or loss weighting would be reused under the same name
  (`config_diff`, `done_problem`);
- a final configuration picked from a partial grid, or from a row without a
  finite loss, is not the sweep's choice (`rank_rows`, `pick_final`);
- an evaluation written before its checkpoints were retrained describes
  other weights (`stale_eval`);
- several "seeds" that are one run twice, or runs scored on different
  windows, would pass as a three-seed result (`check_seed_runs`);
- Phase 9 must start from P9-SIM checkpoints (`p9_pretrained_problems`).

Standard library only (no numpy, no torch), so these rules are tested
anywhere; ml.train.train, ml.train.sweep, ml.train.evaluate and
ml.phase9.run apply them.
"""

from __future__ import annotations

import hashlib
import json
import math
import os

#: Settings that do not change what a run computes (where it runs, how fast, how it is evaluated in batches):
#: a run resumed or reused with other values of these is the same run.
RESUME_IGNORED = frozenset({"overwrite", "workers", "device", "out_root", "compile", "compile_mode", "eval_batch",
                            "interrupt_after_epochs"})
#: Optimiser steps per skip check: one device sync per window.
SKIP_WINDOW = 50
#: The largest share of a window's steps that may be skipped. A frozen run skips them all; a healthy one skips
#: only while the gradient scaler halves its starting scale (2^16) to fit fp16 (ml/tests/test_train_smoke.py's
#: GPU run allows 5 in its first 300 steps), then about one step per 2,000. The scaler's state is first saved
#: with last.pt, so a start-up mistaken for a frozen run would fail again on every resume: half a window allows
#: 25 halvings, which only gradients near 10^7 would need.
MAX_SKIP_FRACTION = 0.5
#: P9-SIM (spec/phase9.md §4.1): Phase 8's recipe retrained on roomsim-v1 with the 40 reference bins the
#: public recordings share.
P9_DATASET = "roomsim-v1"
P9_REF_BINS = "shared40"


def finite(x) -> bool:
    """A real, finite number (not None, not a bool, not inf or NaN)."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _norm(v):
    """`v` as config.json stores it (tuples become lists), so a stored config compares equal to the live one."""
    return json.loads(json.dumps(v, sort_keys=True))


def config_diff(a: dict, b: dict, ignore=RESUME_IGNORED) -> list:
    """The settings, outside `ignore`, in which two run configs differ (a key missing from one differs)."""
    keys = (set(a) | set(b)) - set(ignore)
    return sorted(k for k in keys if (k in a) != (k in b) or (k in a and _norm(a[k]) != _norm(b[k])))


def same_run(a: dict, b: dict) -> bool:
    return not config_diff(a, b)


def _read_json(path: str):
    """`(doc, None)`, or `(None, reason)` when the file is missing or unreadable."""
    try:
        with open(path) as f:
            return json.load(f), None
    except FileNotFoundError:
        return None, "missing"
    except (OSError, ValueError) as e:
        return None, f"unreadable ({type(e).__name__}: {e})"[:200]


def _no_best(best_epoch, best_val) -> str | None:
    if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch < 0 or not finite(best_val):
        return f"no epoch reached a finite validation loss (best_epoch {best_epoch}, best_val_loss {best_val})"
    return None


def done_problem(run_dir: str, cfg: dict | None = None) -> str | None:
    """Why `run_dir` does not hold a finished, usable run (of `cfg`, when
    given); None when it does. Done means: `done.json` records a best epoch
    with a finite validation loss, `best.pt` exists, and (with `cfg`) the
    run's `config.json` equals `cfg` on every setting `cfg` names, outside
    RESUME_IGNORED."""
    done, err = _read_json(os.path.join(run_dir, "done.json"))
    if done is None:
        return "unfinished (no done.json)" if err == "missing" else f"done.json is {err}"
    problem = _no_best(done.get("best_epoch"), done.get("best_val_loss"))
    if problem:
        return problem
    if not os.path.isfile(os.path.join(run_dir, "best.pt")):
        return "done.json without best.pt"
    if cfg is None:
        return None
    stored, err = _read_json(os.path.join(run_dir, "config.json"))
    if stored is None:
        return f"config.json is {err}: the run's configuration cannot be checked"
    diff = config_diff({k: stored[k] for k in cfg if k in stored}, cfg)
    if diff:
        return f"trained with another configuration (differs in {', '.join(diff)})"
    return None


def skip_problem(skipped: int, steps: int, max_fraction: float = MAX_SKIP_FRACTION) -> str | None:
    """Why a window of `steps` optimiser steps, `skipped` of them skipped for
    non-finite gradients, means the run has diverged or frozen; None when it
    has not."""
    if steps <= 0 or skipped <= max_fraction * steps:
        return None
    return (f"{skipped} of {steps} optimiser steps skipped for non-finite gradients (more than "
            f"{max_fraction:.0%}): the run diverged or froze")


def best_problem(best_epoch, best_val, ckpt_epoch) -> str | None:
    """Why `done.json` may not be written: no finite best validation loss,
    or `best.pt` (`ckpt_epoch`: the epoch it holds, None when missing) is not
    the best epoch's. None when it may."""
    problem = _no_best(best_epoch, best_val)
    if problem:
        return problem
    if ckpt_epoch is None:
        return "best.pt is missing"
    if ckpt_epoch != best_epoch:
        return f"best.pt holds epoch {ckpt_epoch}, the best epoch is {best_epoch}"
    return None


def restore_best(last_epoch: int, best_epoch: int, ckpt_epoch) -> bool:
    """True when `best.pt` must be rewritten from `last.pt` on resume: the
    trainer saves `last.pt` before `best.pt`, so a run killed between the
    two has a best epoch that is the last one (whose model `last.pt` holds)
    and a `best.pt` that is older or missing."""
    return best_epoch >= 0 and best_epoch == last_epoch and ckpt_epoch != best_epoch


def rank_rows(rows: list) -> tuple:
    """`(ranked, dropped)`: the sweep rows with a finite `best_val_loss`,
    lowest first, and the run ids of the others."""
    ranked = sorted((r for r in rows if finite(r.get("best_val_loss"))), key=lambda r: r["best_val_loss"])
    dropped = [r["run_id"] for r in rows if not finite(r.get("best_val_loss"))]
    return ranked, dropped


def pick_final(table: dict, choose: str | None = None) -> str:
    """The run id the final runs take their configuration from: `choose`
    when given, else the sweep table's best, which is refused unless every
    sweep run finished with a finite loss (spec/model.md §5.1 selects from
    the whole grid)."""
    if choose:
        return choose
    missing = table["n_runs"] - table["n_finished"]
    if missing > 0 or table.get("not_done"):
        not_done = sorted((table.get("not_done") or {}).items())
        listed = "; ".join(f"{rid}: {why}" for rid, why in not_done[:5]) + ("; ..." if len(not_done) > 5 else "")
        raise SystemExit(f"{missing} of {table['n_runs']} sweep runs are not done ({listed}): finish the sweep "
                         f"(it resumes) or pass --choose RUN_ID")
    if not table.get("best"):
        raise SystemExit("no finished sweep run to choose from (run the sweep first, or pass --choose)")
    return table["best"]


def file_sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def stale_eval(path: str, checkpoints: list) -> str | None:
    """Why the evaluation JSON at `path` (ml.train.evaluate) no longer
    describes `checkpoints`, in order: another run list, or a checkpoint
    rewritten since it was scored (its sha256 differs, or none was
    recorded). None when it still does."""
    doc, err = _read_json(path)
    if doc is None:
        return f"{os.path.basename(path)} is {err}"
    runs = doc.get("runs") or []
    if len(runs) != len(checkpoints):
        return f"it scored {len(runs)} runs, not {len(checkpoints)}"
    for entry, ckpt in zip(runs, checkpoints):
        path_now = os.path.abspath(ckpt)
        if entry.get("checkpoint") != path_now:
            return f"it scored {entry.get('checkpoint')}, not {path_now}"
        if not entry.get("checkpoint_sha256"):
            return "it records no checkpoint hash (written before hashes were recorded)"
        if not os.path.isfile(ckpt):
            return f"{ckpt} is missing"
        if entry["checkpoint_sha256"] != file_sha256(ckpt):
            return f"{ckpt} changed since it was scored (retrained)"
    return None


def _as_list(x) -> list:
    return x.tolist() if hasattr(x, "tolist") else list(x)


def check_seed_runs(entries: list) -> None:
    """Refuse runs that are not distinct seeds scored on the same windows
    (each entry: `run_id`, `seed`, and the window ids `scene`, `k` in
    prediction order). The truth is taken once, from the first run, and the
    seeds' spread is the result: a repeated run would pass as a seed, and
    windows in another order would be scored against the wrong truth."""
    seeds = [e["seed"] for e in entries]
    repeated = sorted({s for s in seeds if seeds.count(s) > 1})
    if repeated:
        raise ValueError(f"runs {[e['run_id'] for e in entries]} repeat seed(s) {repeated}: several runs must be "
                         f"distinct seeds of one configuration")
    ref = entries[0]
    for e in entries[1:]:
        for key in ("scene", "k"):
            if _as_list(e[key]) != _as_list(ref[key]):
                raise ValueError(f"{e['run_id']} was scored on other windows than {ref['run_id']} ({key} differs)")


def recorded_ref_bins(run_dir: str) -> str | None:
    """The reference bins of the cache a run was trained on, as its
    `config.json` records them (`data.ref_bins`); None when it does not."""
    doc, _ = _read_json(os.path.join(run_dir, "config.json"))
    return ((doc or {}).get("data") or {}).get("ref_bins")


def p9_pretrained_problems(run_config: dict, length_s: float, seed: int, ref_bins: str | None) -> list:
    """Why a checkpoint's run config is not P9-SIM seed `seed` at window
    `length_s` (spec/phase9.md §4.1: roomsim-v1, the shared40 reference
    bins, phase and gain off); empty when it is."""
    problems = []
    window = run_config.get("window_s")
    if not finite(window) or float(window) != float(length_s):
        problems.append(f"window_s {window}, not {float(length_s):g}")
    dataset = os.path.basename(os.path.normpath(str(run_config.get("dataset"))))
    if dataset != P9_DATASET:
        problems.append(f"dataset {dataset!r}, not {P9_DATASET!r}")
    if run_config.get("synthetic") is not None:
        problems.append("trained on synthetic data")
    if ref_bins is None:
        problems.append(f"the run records no reference bins (P9-SIM is trained on a {P9_REF_BINS} cache)")
    elif ref_bins != P9_REF_BINS:
        problems.append(f"ref_bins {ref_bins!r}, not {P9_REF_BINS!r}")
    for key in ("phase", "gain"):
        if run_config.get(key):
            problems.append(f"{key} on")
    if run_config.get("seed") != seed:
        problems.append(f"seed {run_config.get('seed')}, not {seed}")
    return problems
