"""Phase 8 metrics (spec/model.md §4) and the predictions contract. SIM.

Every figure here is **SIM** and every result dict says so (`"label":
"SIM"`): the windows are simulated (spec/dataset.md), and nothing here is
a claim about real rooms (ADR-008 d9, ADR-009).

**The predictions contract.** Every method — the model, the stored and
recomputed WLM and RTI, the forest, the MLP and the prior — hands its
evaluation-window outputs to `evaluate` as one dict of numpy arrays (N
windows), so all of them are scored by the same code:

| key | shape | meaning |
|---|---|---|
| `scene`, `k` | [N] int | the window: dataset scene index and end tick |
| `presence_prob` | [N] float | P(someone present); None if not applicable |
| `count_prob` | [N, 4] float | P(count = 0..3); None if not applicable |
| `zone_prob` | [N, 9] float | P(cell occupied), cells `3*row + col` (spec/dataset.md §5); None if n/a |
| `slot_exist` | [N, 3] float | existence probability of each xy slot; None if n/a |
| `slot_xy` | [N, 3, 2] float | slot centres, metres, room frame |
| `slot_sigma` | [N, 3] float | isotropic Gaussian sigma, metres |
| `slot_radius` | [N, 3] float | optional: the 95% radius (default `max(2.448 slot_sigma, 0.3)`) |
| `published` | [N] bool | optional: an abstaining method's publish flag (default all true) |
| `kind` | str | "slots" (the model: up to three estimates) or "one_estimate" (slot 0 is the method's one estimate) |
| `uses_calibration` | bool | optional: the method uses the idealised `calib_*` (WLM, RTI; decision 4) |

A one-estimate method puts its estimate in slot 0 with `slot_exist[:, 0]`
= `published` (1/0) and the other slots' existence 0; its presence is
`published` for WLM and RTI. `predictions_from_slots`,
`one_estimate_predictions` build these dicts.

**Decision rules** (spec/model.md §4, fixed before training): presence at
`presence_prob >= 0.5`; count the argmax; a reachable cell occupied at
`zone_prob >= 0.5`; a slot is an estimate at `slot_exist >= 0.5`.
Matching is Hungarian on the Euclidean distance in metres between the
estimates and the present people; a matched pair beyond the gate (2 m, also
reported at 1 m) is a miss and a false estimate.

**Figures.** Presence: accuracy, balanced accuracy, false-alarm rate
(empty windows called occupied), miss rate, AUC. Count: accuracy, MAE, the
4 x 4 confusion. Zone (reachable cells only): per-cell F1 averaged over the
cells whose F1 is defined, per-cell F1 and confusion counts, exact-set
accuracy. xy: median and p90 error of matched people, matching recall and
precision, the median 95% radius of the matched estimates (and of every
estimate) and its coverage (matched people inside it). Every figure by
window group: all, count = 0..3, walking (some present person `walked`),
still (people present, none walked), empty; xy per person also by the
person's own motion. **One-estimate protocol** (one-person windows only):
each method's one estimate — the model's slot with the highest existence
probability whatever the others say, the forest's/MLP's/prior's xy, WLM's
and RTI's published centre — scored on median and p90 error, single-cell
zone accuracy (the method's most probable reachable cell when it has a
zone head, else the cell of its estimate) and radius coverage; (a) on all
one-person windows, an abstaining method's coverage beside, (b) paired, on
exactly the windows an abstaining method published, with the paired
bootstrap interval of the difference and the window count.

**Uncertainty.** A scene-level bootstrap (spec/model.md §4, decision 6):
2,000 resamples of the scenes with replacement, stratified by the scene's
people count, every window of a drawn scene kept (each resample is a
multiplicity per scene, i.e. a weight per window); the figure is
recomputed per resample, per seed, averaged over seeds; the 95% interval
is the 2.5th-97.5th percentile of the resampled seed-averaged figure
(resamples where it is undefined are dropped and counted). Seeds: the mean
and the sample standard deviation (ddof 1) over seeds of the point
figures; a deterministic method (one prediction set) reports its value
with the interval, never a bare figure. Differences between methods are
paired: the same resampled scenes for both.

    truth = metrics.truth_from_cache(cache, "test")
    res = metrics.evaluate({"model": [p_seed0, p_seed1, p_seed2], "wlm_stored": [wlm]}, truth,
                           reference="model")
"""

from __future__ import annotations

import math

import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment

__all__ = [
    "LABEL", "R95", "MIN_RADIUS_M", "DECISION", "GATES_M", "N_BOOT", "WINDOW_GROUPS", "PERSON_GROUPS",
    "EVAL_MIN_T", "PREDICTION_KEYS", "truth_from_cache", "make_truth", "check_predictions", "align",
    "slot_radius", "zone_cell", "match", "predictions_from_slots", "one_estimate_predictions", "score",
    "bootstrap_counts", "weighted_quantile", "weighted_auc", "evaluate", "aggregate_seeds", "flat_figures",
]

LABEL = "SIM"
#: spec/model.md decision 3: the reported radius is max(2.448 sigma, 0.3 m).
R95 = 2.448
MIN_RADIUS_M = 0.3
DECISION = 0.5
GATES_M = (2.0, 1.0)
N_BOOT = 2000
EVAL_MIN_T = 4.0
N_ZONES = 9
N_SLOTS = 3
WINDOW_GROUPS = ("all", "count=0", "count=1", "count=2", "count=3", "walking", "still", "empty")
PERSON_GROUPS = ("all", "walking", "still")
ONE_GROUPS = ("all", "walking", "still")
PREDICTION_KEYS = ("scene", "k", "presence_prob", "count_prob", "zone_prob", "slot_exist", "slot_xy", "slot_sigma")


# --------------------------------------------------------------------------
# truth
# --------------------------------------------------------------------------


def make_truth(scene, k, count, xy, person_mask, walked, zone_occupied, zone_reachable, room, *, stratum=None,
               geometry_known=None, **info) -> dict:
    """A truth dict from arrays (N windows): `scene`, `k`, `count` [N],
    `xy` [N, 3, 2] (m), `person_mask`, `walked` [N, 3], `zone_occupied`,
    `zone_reachable` [N, 9], `room` [N, 3] (`w, d, h`). `stratum` is the
    bootstrap stratum per window (default: `count`, constant per scene in
    SIM); `geometry_known` [N] (default all true). SIM."""
    count = np.asarray(count, np.int64)
    n = count.shape[0]
    pm = np.asarray(person_mask, bool)
    t = {
        "label": LABEL,
        "scene": np.asarray(scene, np.int64), "k": np.asarray(k, np.int64), "count": count,
        "presence": count > 0,
        "xy": np.where(pm[..., None], np.asarray(xy, np.float64), 0.0),
        "person_mask": pm, "walked": np.asarray(walked, bool) & pm,
        "zone_occupied": np.asarray(zone_occupied, bool), "zone_reachable": np.asarray(zone_reachable, bool),
        "room": np.asarray(room, np.float64).reshape(n, -1),
        "stratum": np.asarray(count if stratum is None else stratum, np.int64),
        "geometry_known": np.ones(n, bool) if geometry_known is None else np.asarray(geometry_known, bool),
    }
    for key in ("scene", "k", "stratum", "geometry_known", "xy", "zone_occupied", "zone_reachable", "walked"):
        if t[key].shape[0] != n:
            raise ValueError(f"truth {key} has {t[key].shape[0]} windows, count has {n}")
    if t["xy"].shape[1:] != (3, 2) or t["zone_occupied"].shape[1:] != (N_ZONES,):
        raise ValueError("truth xy must be [N, 3, 2] and zone_occupied [N, 9]")
    if (t["person_mask"].sum(1) != count).any():
        raise ValueError("truth person_mask disagrees with count")
    t.update(info)
    order = np.lexsort((t["k"], t["scene"]))
    if not np.array_equal(order, np.arange(n)):
        t = {key: (v[order] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) else v) for key, v in t.items()}
    key = t["scene"] * (1 << 20) + t["k"]
    if (np.diff(key) <= 0).any():
        raise ValueError("truth has a repeated (scene, k)")
    _groups(t)
    return t


def _groups(t: dict) -> None:
    """Window groups (WINDOW_GROUPS) as masks, in place."""
    present = t["person_mask"]
    walking = (t["walked"] & present).any(1)
    c = t["count"]
    t["groups"] = {
        "all": np.ones(c.shape[0], bool), "count=0": c == 0, "count=1": c == 1, "count=2": c == 2,
        "count=3": c == 3, "walking": (c > 0) & walking, "still": (c > 0) & ~walking, "empty": c == 0,
    }
    # person zone cells (for the one-estimate protocol): the cell of each present person.
    w = np.maximum(t["room"][:, 0], 1e-12)[:, None]
    d = np.maximum(t["room"][:, 1], 1e-12)[:, None]
    col = np.clip(np.floor(3.0 * t["xy"][..., 0] / w), 0, 2)
    row = np.clip(np.floor(3.0 * t["xy"][..., 1] / d), 0, 2)
    t["person_cell"] = np.where(present, 3 * row + col, -1).astype(np.int64)
    # scenes and their strata (each scene one stratum; checked)
    scenes, pos = np.unique(t["scene"], return_inverse=True)
    t["scene_list"] = scenes
    t["scene_pos"] = pos.astype(np.int64)
    strata = np.full(scenes.size, -1, np.int64)
    strata[pos] = t["stratum"]
    if (strata[pos] != t["stratum"]).any():
        raise ValueError("a scene's windows lie in different bootstrap strata")
    t["scene_stratum"] = strata


def truth_from_cache(c, split: str = "test", subset: str = "all", min_t: float = EVAL_MIN_T, windows=None) -> dict:
    """The truth of decision 5's evaluation windows of `split` in a SIM
    window cache (`ml.train.cache.Cache`): every second label tick with
    `t_k >= min_t`. `subset` names the evaluation subset (spec/model.md
    §5.6); "single01_nogeo" marks geometry unknown (xy and zone not scored).
    The bootstrap stratum is the scene's people count."""
    w = c.window_indices(split, min_t=min_t) if windows is None else np.asarray(windows, np.int64)
    win, sc = c.windows, c.scenes
    s = win["scene_row"][w]
    known = np.full(w.size, subset != "single01_nogeo")
    return make_truth(
        win["scene"][w], win["k"][w], win["count"][w], win["xy"][w], win["person_mask"][w], win["walked"][w],
        win["zone_occupied"][w], sc["zone_reachable"][s], sc["room"][s], stratum=sc["n_people"][s],
        geometry_known=known, window=w, split=split, subset=subset, dataset=c.meta["dataset"],
        impairments=c.meta["impairments"], length_s=c.length_s, l_shape=sc["l_shape"][s],
        n_nodes=sc["n_nodes"][s])


# --------------------------------------------------------------------------
# predictions
# --------------------------------------------------------------------------


def slot_radius(pred: dict) -> np.ndarray | None:
    """[N, 3] 95% radii: `slot_radius` if given, else `max(2.448 slot_sigma, 0.3)`."""
    if pred.get("slot_radius") is not None:
        return np.asarray(pred["slot_radius"], np.float64)
    if pred.get("slot_sigma") is None:
        return None
    return np.maximum(R95 * np.asarray(pred["slot_sigma"], np.float64), MIN_RADIUS_M)


def check_predictions(pred: dict, n: int | None = None) -> dict:
    """Validate a predictions dict (module docstring) and return a copy
    with numpy arrays of fixed dtypes. Refuses a label other than SIM,
    wrong shapes, probabilities outside [0, 1], non-finite values where
    used, and a published estimate without a finite radius."""
    if pred.get("label", LABEL) != LABEL:
        raise ValueError(f"predictions labelled {pred.get('label')!r}; these metrics score SIM windows only")
    out = dict(pred)
    out["label"] = LABEL
    out.setdefault("kind", "slots")
    if out["kind"] not in ("slots", "one_estimate"):
        raise ValueError(f"kind {out['kind']!r} is not 'slots' or 'one_estimate'")
    out["scene"] = np.asarray(pred["scene"], np.int64)
    out["k"] = np.asarray(pred["k"], np.int64)
    n = out["scene"].shape[0] if n is None else n
    shapes = {"presence_prob": (n,), "count_prob": (n, 4), "zone_prob": (n, N_ZONES), "slot_exist": (n, N_SLOTS),
              "slot_xy": (n, N_SLOTS, 2), "slot_sigma": (n, N_SLOTS), "slot_radius": (n, N_SLOTS)}
    if out["k"].shape != (n,) or out["scene"].shape != (n,):
        raise ValueError(f"scene/k must be [{n}]")
    for key, shape in shapes.items():
        v = pred.get(key)
        if v is None:
            out[key] = None
            continue
        v = np.asarray(v, np.float64)
        if v.shape != shape:
            raise ValueError(f"{key} has shape {v.shape}, the contract says {shape}")
        out[key] = v
    for key in ("presence_prob", "count_prob", "zone_prob", "slot_exist"):
        v = out[key]
        if v is not None and (not np.isfinite(v).all() or (v < 0).any() or (v > 1).any()):
            raise ValueError(f"{key} must be probabilities in [0, 1]")
    pub = pred.get("published")
    out["published"] = np.ones(n, bool) if pub is None else np.asarray(pub, bool)
    if out["published"].shape != (n,):
        raise ValueError(f"published must be [{n}]")
    if out["slot_xy"] is not None:
        rad = slot_radius(out)
        if rad is None:
            raise ValueError("slot_xy needs slot_sigma or slot_radius")
        live = out["slot_exist"] >= DECISION if out["slot_exist"] is not None else np.ones((n, N_SLOTS), bool)
        if out["kind"] == "one_estimate":
            live = live.copy()
            live[:, 0] |= out["published"]
        if not np.isfinite(out["slot_xy"][live]).all() or not np.isfinite(rad[live]).all() or (rad[live] <= 0).any():
            raise ValueError("every estimate needs a finite centre and a finite positive radius (ADR-008 d5)")
        out["_radius"] = rad
    if out.get("slot_exist") is None and out.get("slot_xy") is not None:
        out["slot_exist"] = np.zeros((n, N_SLOTS))
        out["slot_exist"][:, 0] = out["published"]
    return out


def align(pred: dict, truth: dict) -> dict:
    """`pred` reordered to `truth`'s windows (by `(scene, k)`); refuses a
    missing or extra window."""
    key_t = truth["scene"] * (1 << 20) + truth["k"]
    key_p = np.asarray(pred["scene"], np.int64) * (1 << 20) + np.asarray(pred["k"], np.int64)
    order = np.argsort(key_p, kind="stable")
    if key_p.shape != key_t.shape or not np.array_equal(key_p[order], key_t):
        missing = np.setdiff1d(key_t, key_p).size
        extra = np.setdiff1d(key_p, key_t).size
        raise ValueError(f"predictions do not cover the truth's windows exactly ({missing} missing, {extra} extra, "
                         f"{key_p.size - np.unique(key_p).size} repeated)")
    n = key_t.size
    out = {}
    for key, v in pred.items():
        out[key] = v[order] if isinstance(v, np.ndarray) and v.shape[:1] == (n,) and v.ndim >= 1 else v
    return out


def zone_cell(xy: np.ndarray, room: np.ndarray) -> np.ndarray:
    """The 3 x 3 cell `3*row + col` of each point (spec/dataset.md §5:
    `col = min(floor(3x/w), 2)`, `row = min(floor(3y/d), 2)`, clamped at 0)."""
    xy = np.asarray(xy, np.float64)
    room = np.asarray(room, np.float64)
    col = np.clip(np.floor(3.0 * xy[..., 0] / room[..., 0]), 0, 2)
    row = np.clip(np.floor(3.0 * xy[..., 1] / room[..., 1]), 0, 2)
    return (3 * row + col).astype(np.int64)


def predictions_from_slots(scene, k, presence_prob, count_prob, zone_prob, slot_exist, slot_xy, slot_sigma,
                           **extra) -> dict:
    """The contract for a slot model (the model's `predictions()` output plus the window ids)."""
    return dict(extra, kind="slots", label=LABEL, scene=scene, k=k, presence_prob=presence_prob,
                count_prob=count_prob, zone_prob=zone_prob, slot_exist=slot_exist, slot_xy=slot_xy,
                slot_sigma=slot_sigma)


def one_estimate_predictions(scene, k, xy, radius, published=None, *, presence_prob=None, count_prob=None,
                             zone_prob=None, **extra) -> dict:
    """The contract for a one-estimate method: `xy` [N, 2] (m), `radius`
    [N] (its 95% radius, m), `published` [N] (default all). Unpublished
    rows may hold NaN. WLM and RTI pass `presence_prob=published`
    (spec/model.md §3.1: presence = published)."""
    xy = np.asarray(xy, np.float64)
    n = xy.shape[0]
    pub = np.ones(n, bool) if published is None else np.asarray(published, bool)
    sx = np.zeros((n, N_SLOTS, 2))
    sx[:, 0] = np.where(pub[:, None], xy, 0.0)
    rad = np.full((n, N_SLOTS), MIN_RADIUS_M)
    rad[:, 0] = np.where(pub, np.asarray(radius, np.float64), MIN_RADIUS_M)
    ex = np.zeros((n, N_SLOTS))
    ex[:, 0] = pub
    return dict(extra, kind="one_estimate", label=LABEL, scene=scene, k=k, presence_prob=presence_prob,
                count_prob=count_prob, zone_prob=zone_prob, slot_exist=ex, slot_xy=sx, slot_sigma=None,
                slot_radius=rad, published=pub)


# --------------------------------------------------------------------------
# matching and scoring
# --------------------------------------------------------------------------


def match(est_xy: np.ndarray, people_xy: np.ndarray) -> tuple:
    """Hungarian assignment (minimum total Euclidean distance, metres) of
    estimates to people: `(est_index, person_index, distance)` of the
    min(n_est, n_people) pairs. No gate: the caller applies it."""
    est_xy = np.asarray(est_xy, np.float64).reshape(-1, 2)
    people_xy = np.asarray(people_xy, np.float64).reshape(-1, 2)
    if est_xy.shape[0] == 0 or people_xy.shape[0] == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), np.zeros(0)
    d = np.sqrt(((est_xy[:, None, :] - people_xy[None, :, :]) ** 2).sum(-1))
    r, c = linear_sum_assignment(d)
    return r.astype(np.int64), c.astype(np.int64), d[r, c]


def _sorted_atoms(values: np.ndarray, windows: np.ndarray) -> tuple:
    order = np.argsort(values, kind="stable")
    return values[order], windows[order]


def score(pred: dict, truth: dict) -> dict:
    """Per-window (and per-person, per-estimate) atoms of one prediction
    set against the truth; the figures are weighted sums and quantiles of
    these (so a bootstrap resample is a weight per window)."""
    pred = align(check_predictions(pred), truth)
    n = truth["count"].shape[0]
    known = truth["geometry_known"]
    at = {"n": n, "kind": pred["kind"], "name": pred.get("name")}
    y = truth["presence"]
    if pred["presence_prob"] is not None:
        at["p"] = pred["presence_prob"]
        at["yhat"] = pred["presence_prob"] >= DECISION
    if pred["count_prob"] is not None:
        at["chat"] = np.argmax(pred["count_prob"], axis=1)
    reach = truth["zone_reachable"]
    if pred["zone_prob"] is not None and known.any():
        at["zhat"] = (pred["zone_prob"] >= DECISION) & reach
        at["zone_known"] = known
    if pred["slot_xy"] is not None and known.any():
        rad = pred["_radius"]
        exist = pred["slot_exist"] >= DECISION
        txy, tpm, twalk = truth["xy"], truth["person_mask"], truth["walked"]
        pair_rows = []        # (window, person, est, dist)
        for i in np.flatnonzero(known):
            e = np.flatnonzero(exist[i])
            p = np.flatnonzero(tpm[i])
            if e.size and p.size:
                r, c, dist = match(pred["slot_xy"][i, e], txy[i, p])
                for a, b, dd in zip(r, c, dist):
                    pair_rows.append((i, p[b], e[a], dd))
        pairs = np.array(pair_rows, np.float64).reshape(-1, 4)
        pw, pp, pe, pd = pairs[:, 0].astype(np.int64), pairs[:, 1].astype(np.int64), pairs[:, 2].astype(np.int64), pairs[:, 3]
        n_est = (exist & known[:, None]).sum(1).astype(np.float64)
        xy_at = {"n_est": n_est, "n_people": (tpm & known[:, None]).sum(1).astype(np.float64),
                 "n_people_walk": (tpm & twalk & known[:, None]).sum(1).astype(np.float64),
                 "n_people_still": (tpm & ~twalk & known[:, None]).sum(1).astype(np.float64)}
        er, ew = np.nonzero(exist & known[:, None])
        xy_at["est_radius"] = _sorted_atoms(rad[er, ew], er)
        for gate in GATES_M:
            ok = pd <= gate
            mw, mp, me, md = pw[ok], pp[ok], pe[ok], pd[ok]
            mrad = rad[mw, me]
            mcov = md <= mrad
            mwalk = twalk[mw, mp]
            g = {}
            g["n_matched"] = np.bincount(mw, minlength=n).astype(np.float64)
            g["n_covered"] = np.bincount(mw, weights=mcov.astype(np.float64), minlength=n)
            g["n_matched_walk"] = np.bincount(mw, weights=mwalk.astype(np.float64), minlength=n)
            g["n_matched_still"] = g["n_matched"] - g["n_matched_walk"]
            g["n_covered_walk"] = np.bincount(mw, weights=(mcov & mwalk).astype(np.float64), minlength=n)
            g["n_covered_still"] = g["n_covered"] - g["n_covered_walk"]
            g["err"] = _sorted_atoms(md, mw)
            g["err_walk"] = _sorted_atoms(md[mwalk], mw[mwalk])
            g["err_still"] = _sorted_atoms(md[~mwalk], mw[~mwalk])
            g["radius"] = _sorted_atoms(mrad, mw)
            xy_at[gate] = g
        at["xy"] = xy_at
    # one-estimate protocol: one-person windows with geometry known
    if pred["slot_xy"] is not None and known.any():
        one = (truth["count"] == 1) & known
        if pred["kind"] == "slots":
            best = np.argmax(pred["slot_exist"], axis=1)
            pub = np.ones(n, bool)
        else:
            best = np.zeros(n, np.int64)
            pub = pred["published"].copy()
        exy = pred["slot_xy"][np.arange(n), best]
        erad = pred["_radius"][np.arange(n), best]
        pidx = np.argmax(truth["person_mask"], axis=1)
        pxy = truth["xy"][np.arange(n), pidx]
        err = np.sqrt(((exy - pxy) ** 2).sum(1))
        if pred["zone_prob"] is not None:
            zp = np.where(reach, pred["zone_prob"], -1.0)
            cell = np.argmax(zp, axis=1)
        else:
            cell = zone_cell(np.where(np.isfinite(exy), exy, 0.0), truth["room"])
        hit = truth["zone_occupied"][np.arange(n), cell]
        at["one"] = {"mask": one, "published": pub & one, "err": np.where(pub & one, err, np.nan),
                     "radius": np.where(pub & one, erad, np.nan), "covered": pub & one & (err <= erad),
                     "hit": pub & one & hit, "walk": truth["walked"][np.arange(n), pidx] & one}
        # the single estimate's cell against zone_occupied on every window (spec/dataset.md §5's scoring note)
        at["zone1"] = {"published": pub & known, "hit": pub & known & hit}
    return at


# --------------------------------------------------------------------------
# weighted statistics
# --------------------------------------------------------------------------


def bootstrap_counts(scene_stratum: np.ndarray, n_boot: int = N_BOOT, seed: int = 0) -> np.ndarray:
    """[n_boot, S] scene multiplicities of a stratified scene bootstrap:
    within each stratum of `n_s` scenes, `n_s` draws with replacement."""
    scene_stratum = np.asarray(scene_stratum)
    s = scene_stratum.shape[0]
    rng = np.random.default_rng(seed)
    counts = np.zeros((n_boot, s), np.int32)
    rows = np.arange(n_boot)[:, None]
    for stratum in np.unique(scene_stratum):
        idx = np.flatnonzero(scene_stratum == stratum)
        draws = idx[rng.integers(0, idx.size, size=(n_boot, idx.size))]
        np.add.at(counts, (np.broadcast_to(rows, draws.shape), draws), 1)
    return counts


def weighted_quantile(values_sorted: np.ndarray, weights: np.ndarray, q: float) -> np.ndarray:
    """Per row of `weights` [B, M] (non-negative integers: multiplicities),
    the `q` quantile of `values_sorted` [M] (ascending) repeated by those
    multiplicities, with numpy's default linear interpolation (so unit
    weights give `np.percentile(values, 100 q)`). NaN for a row of zero
    total weight."""
    weights = np.atleast_2d(weights)
    b = weights.shape[0]
    if values_sorted.size == 0:
        return np.full(b, np.nan)
    cw = np.cumsum(weights, axis=1)
    tot = cw[:, -1]
    h = (tot - 1) * q
    jf = np.floor(h)
    frac = h - jf
    jc = np.minimum(jf + 1, np.maximum(tot - 1, 0))
    i_f = np.minimum((cw <= jf[:, None]).sum(1), values_sorted.size - 1)
    i_c = np.minimum((cw <= jc[:, None]).sum(1), values_sorted.size - 1)
    v = values_sorted[i_f] + frac * (values_sorted[i_c] - values_sorted[i_f])
    return np.where(tot > 0, v, np.nan)


def weighted_auc(score_: np.ndarray, y: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """Per row of `weights` [B, N], the ROC AUC of `score_` for labels `y`
    with those window weights (Mann-Whitney, ties count one half). NaN when
    a class has no weight."""
    weights = np.atleast_2d(weights).astype(np.float64)
    y = np.asarray(y, bool)
    uniq, inv = np.unique(np.asarray(score_, np.float64), return_inverse=True)
    n, g = y.size, uniq.size
    ind = sparse.csr_matrix((np.ones(n), (np.arange(n), inv)), shape=(n, g))
    pos = np.asarray((ind.T @ (weights * y).T).T)
    neg = np.asarray((ind.T @ (weights * ~y).T).T)
    below = np.cumsum(neg, axis=1) - neg
    num = (pos * (below + 0.5 * neg)).sum(1)
    den = pos.sum(1) * neg.sum(1)
    return np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan)


def _div(a, b):
    a = np.asarray(a, np.float64)
    b = np.asarray(b, np.float64)
    return np.where(b > 0, a / np.where(b > 0, b, 1.0), np.nan)


def _q(atoms: tuple, wt: np.ndarray, q: float, keep: np.ndarray | None = None) -> np.ndarray:
    vals, win = atoms
    if keep is not None:
        sel = keep[win]
        vals, win = vals[sel], win[sel]
    return weighted_quantile(vals, wt[:, win], q)


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def _figures(at: dict, truth: dict, wt: np.ndarray, groups=WINDOW_GROUPS) -> dict:
    """Every §4 figure of one scored prediction set under window weights
    `wt` [B, N]: name -> [B]. Names are `area|group|figure`."""
    out = {}
    y = truth["presence"]
    c = truth["count"]
    for gname in groups:
        m = truth["groups"][gname]
        if not m.any():
            continue
        w = wt[:, m]
        nw = w.sum(1)
        if "p" in at:
            yh, ym = at["yhat"][m], y[m]
            pos = w @ ym
            neg = nw - pos
            tp = w @ (yh & ym)
            tn = w @ (~yh & ~ym)
            tpr, tnr = _div(tp, pos), _div(tn, neg)
            out[f"presence|{gname}|accuracy"] = _div(tp + tn, nw)
            out[f"presence|{gname}|balanced_accuracy"] = 0.5 * (tpr + tnr)
            out[f"presence|{gname}|false_alarm_rate"] = 1.0 - tnr
            out[f"presence|{gname}|miss_rate"] = 1.0 - tpr
            out[f"presence|{gname}|auc"] = weighted_auc(at["p"][m], ym, w)
        if "chat" in at:
            ch, cm = at["chat"][m], c[m]
            out[f"count|{gname}|accuracy"] = _div(w @ (ch == cm), nw)
            out[f"count|{gname}|mae"] = _div(w @ np.abs(ch - cm).astype(np.float64), nw)
        if "zhat" in at:
            mk = m & at["zone_known"]
            if mk.any():
                wk = wt[:, mk]
                zh, zt, rc = at["zhat"][mk], truth["zone_occupied"][mk], truth["zone_reachable"][mk]
                tp = wk @ (zh & zt & rc)
                fp = wk @ (zh & ~zt & rc)
                fn = wk @ (~zh & zt & rc)
                f1 = _div(2 * tp, 2 * tp + fp + fn)               # [B, 9]
                with np.errstate(all="ignore"):
                    defined = np.isfinite(f1)
                    macro = np.where(defined.any(1), np.nansum(np.where(defined, f1, 0.0), 1)
                                     / np.maximum(defined.sum(1), 1), np.nan)
                out[f"zone|{gname}|f1_macro"] = macro
                exact = ((zh == zt) | ~rc).all(1)
                out[f"zone|{gname}|exact_set_accuracy"] = _div(wk @ exact, wk.sum(1))
                if gname == "all":
                    for cell in range(N_ZONES):
                        out[f"zone|all|f1_cell{cell}"] = f1[:, cell]
        if "zone1" in at:
            z1 = at["zone1"]
            mk = m & truth["geometry_known"]
            if mk.any():
                wk = wt[:, mk]
                npub = wk @ z1["published"][mk]
                out[f"zone1|{gname}|hit_rate"] = _div(wk @ z1["hit"][mk], npub)
                out[f"zone1|{gname}|coverage"] = _div(npub, wk.sum(1))
        if "xy" in at:
            xa = at["xy"]
            mk = m & truth["geometry_known"]
            if mk.any():
                wk = wt[:, mk]
                n_people = wk @ xa["n_people"][mk]
                n_est = wk @ xa["n_est"][mk]
                out[f"xy|{gname}|radius_median_all_estimates"] = _q(xa["est_radius"], wt, 0.5, mk)
                for gate in GATES_M:
                    g = xa[gate]
                    tag = f"xy{gate:g}m"
                    nm = wk @ g["n_matched"][mk]
                    out[f"{tag}|{gname}|median_error"] = _q(g["err"], wt, 0.5, mk)
                    out[f"{tag}|{gname}|p90_error"] = _q(g["err"], wt, 0.9, mk)
                    out[f"{tag}|{gname}|recall"] = _div(nm, n_people)
                    out[f"{tag}|{gname}|precision"] = _div(nm, n_est)
                    out[f"{tag}|{gname}|radius_median"] = _q(g["radius"], wt, 0.5, mk)
                    out[f"{tag}|{gname}|coverage"] = _div(wk @ g["n_covered"][mk], nm)
    if "xy" in at and truth["geometry_known"].any():
        xa = at["xy"]
        mk = truth["geometry_known"]
        wk = wt[:, mk]
        for gate in GATES_M:
            g = xa[gate]
            tag = f"xy{gate:g}m"
            for pg in ("walking", "still"):
                suffix = "walk" if pg == "walking" else "still"
                nm = wk @ g[f"n_matched_{suffix}"][mk]
                out[f"{tag}|person:{pg}|median_error"] = _q(g[f"err_{suffix}"], wt, 0.5, mk)
                out[f"{tag}|person:{pg}|p90_error"] = _q(g[f"err_{suffix}"], wt, 0.9, mk)
                out[f"{tag}|person:{pg}|recall"] = _div(nm, wk @ xa[f"n_people_{suffix}"][mk])
                out[f"{tag}|person:{pg}|coverage"] = _div(wk @ g[f"n_covered_{suffix}"][mk], nm)
    return out


def _one_figures(one: dict, wt: np.ndarray, restrict: np.ndarray | None = None, prefix: str = "one") -> dict:
    """The one-estimate protocol's figures on one-person windows (within
    `restrict`): coverage (published share), and over the published
    windows median and p90 error, zone accuracy, radius coverage and the
    median radius; by all / walking / still (the person's motion)."""
    out = {}
    base = one["mask"] if restrict is None else one["mask"] & restrict
    for gname in ONE_GROUPS:
        m = base if gname == "all" else base & (one["walk"] if gname == "walking" else ~one["walk"])
        if not m.any():
            continue
        w = wt[:, m]
        nw = w.sum(1)
        pub = one["published"][m]
        npub = w @ pub
        out[f"{prefix}|{gname}|coverage"] = _div(npub, nw)
        out[f"{prefix}|{gname}|n_published"] = npub
        pm = np.zeros_like(one["mask"])
        pm[np.flatnonzero(m)[pub]] = True
        if pm.any():
            errs = one["err"][pm]
            wins = np.flatnonzero(pm)
            order = np.argsort(errs, kind="stable")
            atoms = (errs[order], wins[order])
            out[f"{prefix}|{gname}|median_error"] = weighted_quantile(atoms[0], wt[:, atoms[1]], 0.5)
            out[f"{prefix}|{gname}|p90_error"] = weighted_quantile(atoms[0], wt[:, atoms[1]], 0.9)
            rad = one["radius"][pm]
            ro = np.argsort(rad, kind="stable")
            out[f"{prefix}|{gname}|radius_median"] = weighted_quantile(rad[ro], wt[:, wins[ro]], 0.5)
            wp = wt[:, pm]
            npw = wp.sum(1)
            out[f"{prefix}|{gname}|zone_accuracy"] = _div(wp @ one["hit"][pm], npw)
            out[f"{prefix}|{gname}|radius_coverage"] = _div(wp @ one["covered"][pm], npw)
    return out


def _point_tables(at: dict, truth: dict) -> dict:
    """Point-only tables: the 4 x 4 count confusion (rows truth, columns
    predicted) and each cell's zone confusion counts over windows where it
    is reachable (ADR-008 d8)."""
    out = {}
    if "chat" in at:
        conf = np.zeros((4, 4), np.int64)
        np.add.at(conf, (truth["count"], at["chat"]), 1)
        out["count_confusion"] = conf.tolist()
    if "zhat" in at:
        mk = at["zone_known"]
        zh, zt, rc = at["zhat"][mk], truth["zone_occupied"][mk], truth["zone_reachable"][mk]
        out["zone_cells"] = [{"cell": cell, "reachable_windows": int(rc[:, cell].sum()),
                              "tp": int((zh[:, cell] & zt[:, cell] & rc[:, cell]).sum()),
                              "fp": int((zh[:, cell] & ~zt[:, cell] & rc[:, cell]).sum()),
                              "fn": int((~zh[:, cell] & zt[:, cell] & rc[:, cell]).sum()),
                              "tn": int((~zh[:, cell] & ~zt[:, cell] & rc[:, cell]).sum())} for cell in range(N_ZONES)]
    return out


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------


def _ci(samples: np.ndarray) -> dict:
    ok = np.isfinite(samples)
    if not ok.any():
        return {"ci95": [None, None], "n_boot_valid": 0}
    lo, hi = np.percentile(samples[ok], [2.5, 97.5])
    return {"ci95": [float(lo), float(hi)], "n_boot_valid": int(ok.sum())}


def _num(x):
    x = float(x)
    return x if math.isfinite(x) else None


def _summary(points: list, boot: np.ndarray | None) -> dict:
    """One figure: value (one seed) or mean, std and the seed values, plus
    the bootstrap interval of the seed-averaged figure."""
    pts = np.array(points, np.float64)
    if pts.size == 1:
        d = {"value": _num(pts[0])}
    else:
        d = {"mean": _num(pts.mean()), "std": _num(pts.std(ddof=1)), "seeds": [_num(v) for v in pts]}
    if boot is not None:
        d.update(_ci(boot))
    return d


def _nest(flat: dict) -> dict:
    out: dict = {}
    for key, val in flat.items():
        parts = key.split("|")
        node = out
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = val
    return out


def flat_figures(result_method: dict) -> dict:
    """`area|group|figure` -> the figure dict, from one method's nested result."""
    flat = {}

    def walk(node, path):
        if isinstance(node, dict) and ("value" in node or "mean" in node or "diff" in node):
            flat["|".join(path)] = node
            return
        if isinstance(node, dict):
            for k2, v2 in node.items():
                walk(v2, path + [k2])
    walk(result_method.get("figures", {}), [])
    return flat


def evaluate(methods: dict, truth: dict, *, reference: str | None = None, n_boot: int = N_BOOT, seed: int = 0,
             chunk: int = 100, one_estimate: bool = True, groups=WINDOW_GROUPS) -> dict:
    """Score every method on the same windows and the same resampled scenes.

    `methods`: name -> one predictions dict (a deterministic method) or a
    list of them (one per seed). `reference` (e.g. "model"): also every
    paired difference reference - method of the figures both have, and
    the one-estimate protocol's (b) comparisons against every abstaining
    method. Returns a JSON-ready dict labelled SIM (module docstring)."""
    if truth.get("label") != LABEL:
        raise ValueError("truth must be labelled SIM")
    sets = {name: (v if isinstance(v, (list, tuple)) else [v]) for name, v in methods.items()}
    if reference is not None and reference not in sets:
        raise ValueError(f"reference {reference!r} is not among the methods")
    atoms = {name: [score(p, truth) for p in preds] for name, preds in sets.items()}
    meta = {name: {"kind": atoms[name][0]["kind"], "seeds": len(preds),
                   "uses_calibration": bool(preds[0].get("uses_calibration", False)),
                   "published_share": float(np.mean(check_predictions(preds[0])["published"]))}
            for name, preds in sets.items()}
    n = truth["count"].shape[0]
    # (b) paired one-estimate comparisons: every other method with an estimate, on the windows it
    # published (all one-person windows for a method that never abstains).
    paired_with = [name for name in sets if name != reference and "one" in atoms[name][0]]
    one_ok = one_estimate and any("one" in a[0] for a in atoms.values())

    def compute(wt):
        res = {}
        for name, ats in atoms.items():
            res[name] = []
            for a in ats:
                f = _figures(a, truth, wt, groups)
                if one_ok and "one" in a:
                    f.update(_one_figures(a["one"], wt))
                res[name].append(f)
        pairs = {}
        if one_ok and reference is not None and "one" in atoms[reference][0]:
            for other in paired_with:
                if "one" not in atoms[other][0]:
                    continue
                pubb = atoms[other][0]["one"]["published"]
                pairs[other] = ([_one_figures(a["one"], wt, pubb, "paired") for a in atoms[reference]],
                                [_one_figures(a["one"], wt, pubb, "paired") for a in atoms[other]])
        return res, pairs

    point_res, point_pairs = compute(np.ones((1, n)))
    counts = bootstrap_counts(truth["scene_stratum"], n_boot, seed) if n_boot else None
    boot = {name: [{} for _ in ats] for name, ats in atoms.items()}
    boot_pairs = {}
    if n_boot:
        for start in range(0, n_boot, chunk):
            wt = counts[start:start + chunk][:, truth["scene_pos"]].astype(np.float64)
            res, pairs = compute(wt)
            for name, per_seed in res.items():
                for s, f in enumerate(per_seed):
                    for key, val in f.items():
                        boot[name][s].setdefault(key, []).append(val)
            for other, (ra, rb) in pairs.items():
                bp = boot_pairs.setdefault(other, ([{} for _ in ra], [{} for _ in rb]))
                for side, lst in ((0, ra), (1, rb)):
                    for s, f in enumerate(lst):
                        for key, val in f.items():
                            bp[side][s].setdefault(key, []).append(val)

    def seed_mean(per_seed_boot, key):
        arrs = [np.concatenate(d[key]) for d in per_seed_boot if key in d]
        if len(arrs) != len(per_seed_boot):
            return None
        return np.mean(np.stack(arrs), axis=0)

    result = {
        "label": LABEL, "split": truth.get("split"), "dataset": truth.get("dataset"),
        "impairments": truth.get("impairments"), "length_s": truth.get("length_s"), "subset": truth.get("subset"),
        "n_windows": int(n), "n_scenes": int(truth["scene_list"].size),
        "n_windows_by_group": {g: int(m.sum()) for g, m in truth["groups"].items()},
        "n_scenes_by_stratum": {str(int(s)): int((truth["scene_stratum"] == s).sum())
                                for s in np.unique(truth["scene_stratum"])},
        "bootstrap": {"resamples": int(n_boot), "unit": "scene", "stratified_by": "people count",
                      "ci": 0.95, "interval": "percentile", "seed": int(seed),
                      "seeds": "per resample the figure is computed per seed and averaged over seeds"},
        "decision_rules": {"presence": ">= 0.5", "count": "argmax", "zone": "reachable cell >= 0.5",
                           "slot": "existence >= 0.5", "match": "Hungarian, Euclidean m",
                           "gates_m": list(GATES_M), "radius": "max(2.448 sigma, 0.3 m) unless given"},
        "methods": {},
    }
    for name, per_seed in point_res.items():
        keys = sorted(set().union(*[f.keys() for f in per_seed]))
        figs = {}
        for key in keys:
            pts = [f[key][0] if key in f else np.nan for f in per_seed]
            bs = seed_mean(boot[name], key) if n_boot else None
            figs[key] = _summary(pts, bs)
        tables = [_point_tables(a, truth) for a in atoms[name]]
        result["methods"][name] = dict(meta[name], label=LABEL, figures=_nest(figs),
                                       tables=tables[0] if len(tables) == 1 else {"seeds": tables})
    if reference is not None:
        diffs = {}
        ref_point = point_res[reference]
        for other in sets:
            if other == reference:
                continue
            shared = set().union(*[f.keys() for f in ref_point]) & set().union(*[f.keys() for f in point_res[other]])
            dd = {}
            for key in sorted(shared):
                if key.startswith("one|") or key.endswith("|n_published"):
                    continue
                a = np.mean([f[key][0] for f in ref_point])
                b = np.mean([f[key][0] for f in point_res[other]])
                entry = {"diff": _num(a - b), "reference": _num(a), "other": _num(b)}
                if n_boot:
                    ba, bb = seed_mean(boot[reference], key), seed_mean(boot[other], key)
                    if ba is not None and bb is not None:
                        entry.update(_ci(ba - bb))
                dd[key] = entry
            diffs[f"{reference} - {other}"] = _nest(dd)
        result["differences"] = diffs
        paired = {}
        for other, (ra, rb) in point_pairs.items():
            keys = sorted(set().union(*[f.keys() for f in ra]) & set().union(*[f.keys() for f in rb]))
            block = {"n_windows": int((atoms[other][0]["one"]["published"]).sum()), "label": LABEL}
            for key in keys:
                a = np.mean([f[key][0] for f in ra])
                b = np.mean([f[key][0] for f in rb])
                e = {reference: _num(a), other: _num(b), "diff": _num(a - b)}
                if n_boot:
                    ba = seed_mean(boot_pairs[other][0], key)
                    bb = seed_mean(boot_pairs[other][1], key)
                    if ba is not None and bb is not None:
                        e.update(_ci(ba - bb))
                        e[f"{reference}_ci95"] = _ci(ba)["ci95"]
                        e[f"{other}_ci95"] = _ci(bb)["ci95"]
                block[key] = e
            paired[f"{reference} vs {other}"] = _nest(block)
        result["one_estimate_paired"] = paired
    return result


def aggregate_seeds(preds: list, truth: dict, name: str = "model", **kw) -> dict:
    """One method's figures over its seeds: per figure the mean and std
    (ddof 1) over seeds and the bootstrap interval of the seed-averaged
    figure (same resamples for every seed). SIM."""
    res = evaluate({name: list(preds)}, truth, **kw)
    return res["methods"][name]
