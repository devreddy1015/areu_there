"""Phase 9 metrics (spec/phase9.md §6): presence and count on one REAL test
session, with a moving-block bootstrap.

A method's predictions on the session's evaluation windows (in cache order)
are `{"presence_prob" [N], "count_prob" [N, 4] or None}` (and optionally
`auc_score` [N], the ranking AUC uses, for a baseline whose decision is
not a threshold on one probability), or a list of
them (one per seed). Decision rules are spec/model.md §4's: presence at
probability >= 0.5, count the argmax (a method without a count head
counts its presence decision). Nobody abstains on presence here, so
coverage is 1 and every result says so.

**Intervals.** Windows are 1 s apart and strongly dependent, so the
bootstrap resamples **blocks** of `BLOCK` consecutive evaluation windows
within each segment (the last block of a segment may be shorter): per
segment, as many blocks as it has, drawn with replacement, so every
resample keeps the label mix. `N_BOOT` resamples; each recomputes the
figure per seed and averages the seeds. A difference between two methods
is computed on the same resamples (paired).
"""

from __future__ import annotations

import numpy as np

from ml.train import metrics as m8

__all__ = ["BLOCK", "N_BOOT", "LABELS", "block_ids", "block_weights", "figures", "evaluate", "compare"]

BLOCK = 30
N_BOOT = 2000
LABELS = ("REAL", "SIM->REAL")


def block_ids(segment: np.ndarray, block: int = BLOCK) -> np.ndarray:
    """[N] the block of each evaluation window: consecutive runs of `block`
    windows within one segment (windows in time order)."""
    segment = np.asarray(segment)
    out = np.empty(segment.size, np.int64)
    nxt = 0
    for s in np.unique(segment):
        idx = np.flatnonzero(segment == s)
        out[idx] = nxt + np.arange(idx.size) // block
        nxt = int(out[idx].max()) + 1
    return out


def block_weights(segment: np.ndarray, n_boot: int = N_BOOT, seed: int = 0, block: int = BLOCK) -> np.ndarray:
    """[n_boot, N] window multiplicities of the moving-block bootstrap:
    within each segment, its `n_b` blocks drawn `n_b` times with replacement."""
    blocks = block_ids(segment, block)
    seg_of_block = np.array([np.asarray(segment)[blocks == b][0] for b in range(blocks.max() + 1)])
    counts = m8.bootstrap_counts(seg_of_block, n_boot=n_boot, seed=seed)         # [n_boot, n_blocks]
    return counts[:, blocks].astype(np.float64)


def _groups(truth: dict) -> dict:
    p = np.asarray(truth["presence"], bool)
    moving = np.asarray(truth["moving"], bool)
    lead = np.asarray(truth["leading_empty"], bool)
    return {"all": np.ones(p.size, bool), "empty": ~p, "empty_leading": ~p & lead, "empty_trailing": ~p & ~lead,
            "occupied": p, "still": p & ~moving, "moving": p & moving}


def figures(pred: dict, truth: dict, wt: np.ndarray) -> dict:
    """Every figure for weights `wt` [B, N] (B = 1 for the point value): `name -> [B]`."""
    wt = np.atleast_2d(wt)
    y = np.asarray(truth["presence"], bool)
    prob = np.asarray(pred["presence_prob"], np.float64)
    dec = prob >= 0.5
    g = _groups(truth)
    out = {}

    def rate(mask_num, mask_den):
        return m8._div((wt * mask_num).sum(1), (wt * mask_den).sum(1))
    out["presence|accuracy"] = rate(dec == y, np.ones_like(y))
    tpr = rate(dec & y, y)
    tnr = rate(~dec & ~y, ~y)
    out["presence|balanced_accuracy"] = 0.5 * (tpr + tnr)
    out["presence|false_alarm"] = rate(dec & g["empty"], g["empty"])
    out["presence|false_alarm_leading"] = rate(dec & g["empty_leading"], g["empty_leading"])
    out["presence|false_alarm_trailing"] = rate(dec & g["empty_trailing"], g["empty_trailing"])
    out["presence|miss"] = rate(~dec & g["occupied"], g["occupied"])
    out["presence|miss_still"] = rate(~dec & g["still"], g["still"])
    out["presence|miss_moving"] = rate(~dec & g["moving"], g["moving"])
    score = np.asarray(pred.get("auc_score", prob), np.float64)              # a baseline's own ranking score
    out["presence|auc"] = m8.weighted_auc(score, y, wt)
    for name, sel in (("still", g["still"] | g["empty"]), ("moving", g["moving"] | g["empty"])):
        out[f"presence|auc_{name}_vs_empty"] = m8.weighted_auc(score[sel], y[sel], wt[:, sel])
    count = np.asarray(truth["count"], np.int64)
    if pred.get("count_prob") is not None:
        c_hat = np.asarray(pred["count_prob"]).argmax(1)
    else:
        c_hat = dec.astype(np.int64)
    out["count|accuracy"] = rate(c_hat == count, np.ones_like(y))
    out["count|mae"] = m8._div((wt * np.abs(c_hat - count)).sum(1), wt.sum(1))
    out["coverage"] = np.ones(wt.shape[0])
    return out


def _as_list(p):
    return p if isinstance(p, (list, tuple)) else [p]


def evaluate(methods: dict, truth: dict, *, label_of: dict, n_boot: int = N_BOOT, seed: int = 0,
             block: int = BLOCK) -> dict:
    """Score every method on one session's evaluation windows: per figure
    the value (one seed) or mean ± std over seeds, and the bootstrap's 95%
    interval of the seed-averaged figure. `label_of[name]` is "REAL" or
    "SIM->REAL" (refused otherwise)."""
    n = len(truth["presence"])
    wt = block_weights(truth["segment"], n_boot, seed, block)
    one = np.ones((1, n))
    res = {}
    for name, preds in methods.items():
        lab = label_of[name]
        if lab not in LABELS:
            raise ValueError(f"{name}: label {lab!r} is not one of {LABELS}")
        preds = _as_list(preds)
        for p in preds:
            if len(p["presence_prob"]) != n:
                raise ValueError(f"{name}: {len(p['presence_prob'])} predictions for {n} windows")
        points = [figures(p, truth, one) for p in preds]
        boots = [figures(p, truth, wt) for p in preds]
        figs = {}
        for key in points[0]:
            pts = [float(pt[key][0]) for pt in points]
            boot = np.mean(np.stack([b[key] for b in boots]), axis=0)
            figs[key] = m8._summary(pts, boot)
        res[name] = {"label": lab, "n_seeds": len(preds), "n_windows": int(n), "figures": m8._nest(figs)}
    return res


def compare(a: list, b: list, truth: dict, *, n_boot: int = N_BOOT, seed: int = 0, block: int = BLOCK,
            keys=("presence|balanced_accuracy", "presence|auc", "presence|false_alarm", "presence|miss")) -> dict:
    """Paired differences `a - b` (seed-averaged) on the same resampled blocks: point and 95% interval."""
    n = len(truth["presence"])
    wt = block_weights(truth["segment"], n_boot, seed, block)
    one = np.ones((1, n))
    out = {}
    for key in keys:
        pa = np.mean([figures(p, truth, one)[key][0] for p in _as_list(a)])
        pb = np.mean([figures(p, truth, one)[key][0] for p in _as_list(b)])
        ba = np.mean(np.stack([figures(p, truth, wt)[key] for p in _as_list(a)]), axis=0)
        bb = np.mean(np.stack([figures(p, truth, wt)[key] for p in _as_list(b)]), axis=0)
        d = {"diff": m8._num(pa - pb)}
        d.update(m8._ci(ba - bb))
        out[key] = d
    return out
