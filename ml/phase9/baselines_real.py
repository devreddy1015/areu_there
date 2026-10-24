"""Phase 9 baselines on REAL sessions (spec/phase9.md §5), each returning
`{"presence_prob" [N], "count_prob" None or [N, 4], ...}` on a session's
evaluation windows (`ml.phase9.metrics_real`'s contract).

1. **DSP presence** (`dsp_presence`): the S0–S5 `SensingPipeline`'s
   presence path with its defaults — `StaticSuppressor` -> `fuse` ->
   `PresenceDetector` (50 s warm-up, 30 s of calibration, 10 s hold) —
   per link, on the 40 shared amplitudes of the session's (capped) packets
   held onto a 20 Hz grid from the first segment's start (zero-order hold:
   each tick takes the link's latest packet; ticks before its first packet
   are skipped). `SensingPipeline.process` also runs the vitals
   extractors, which never feed presence; they are skipped here, so the
   presence output is the pipeline's. The first 80 s (warm-up and
   calibration) lie inside the leading empty segment and outside every
   scored window. At each window end: per link `occupied` and the ratio
   energy / threshold; the room decision is the majority of links, its
   score (for AUC) the median ratio.
2. **WLM gate** (`wlm_gate`): spec/locate.md §5 step 1 on Locate's RSSI
   link signal (`wifisense.localize.measure.LinkMeasurement`, `window_s =
   ℓ`), the null sigma calibrated on the session's first 80 s
   (`calibrate_from`, 10 windows at least), the stream after it fed in time
   order. Presence = the gate fires (some link with `y > 3 null_sigma +
   0.25 dB²`); score = `max_l (y_l - 0.25) / null_sigma_l`. Node positions
   are unknown, so the measurement gets placeholder positions on the
   room's perimeter: the gate never reads them, and the WLM estimate and
   its radius are not applicable.
3. **Variance features** (`variance_features`, `fit_variance`): the
   geometry-free part of spec/model.md §3.3 — `s_l` (mean of `sigma` over
   the 40 reference bins) summarised over the used links (mean, std, min,
   max, quartiles, 90th percentile) and the used-link count; a random
   forest (300 trees) and an MLP (scikit-learn, two hidden layers of 128,
   200 epochs of Adam, no early stopping: no REAL validation room) for
   presence and count, per seed.
4. **Prior** (`prior`): the training windows' majority class.

DSP and WLM use the test session's own first 80 s as an empty-room
calibration, as a deployment does at install; the learned methods see no
test-room data.
"""

from __future__ import annotations

import math
import multiprocessing
import os

import numpy as np

from ml import opencsi as oc
from ml import preprocess as pp

__all__ = ["CALIB_S", "DSP_RATE_HZ", "dsp_link_states", "dsp_presence", "placeholder_geometry", "wlm_gate",
           "variance_features", "fit_variance", "prior"]

#: The baselines' calibration span from the first segment's start (DSP: 50 s warm-up + 30 s).
CALIB_S = 80.0
DSP_RATE_HZ = 20.0
_REF = np.isin(pp.K_GRID, pp.SHARED40)
_JOB: dict = {}


# --------------------------------------------------------------------------
# 1. DSP presence
# --------------------------------------------------------------------------


def zoh_ticks(t_link_ns: np.ndarray, t_start_ns: int, t_stop_ns: int, rate_hz: float = DSP_RATE_HZ) -> tuple:
    """Zero-order hold of one link's packets onto `t_start + j / rate`
    (`<= t_stop`): `(tick times [J] ns, packet index [J])`, ticks before the
    link's first packet dropped."""
    step = int(round(1e9 / rate_hz))
    ticks = np.arange(int(t_start_ns), int(t_stop_ns) + 1, step, dtype=np.int64)
    idx = np.searchsorted(t_link_ns, ticks, "right") - 1
    ok = idx >= 0
    return ticks[ok], idx[ok]


def _dsp_link(l: int) -> tuple:
    from wifisense.presence import PresenceDetector
    from wifisense.preprocess import StaticSuppressor, fuse
    from wifisense.types import CsiFrame
    s, start, ends = _JOB["session"], _JOB["start"], _JOB["ends"]
    a, b = s.offsets[l], s.offsets[l + 1]
    occ = np.zeros(ends.size, bool)
    ratio = np.full(ends.size, np.nan)
    if b == a:
        return l, occ, ratio
    t = s.t_ns[a:b]
    amp = np.abs(pp.to_grid(s.iq[a:b], s.layout))[:, _REF]                  # [n, 40] raw |h|
    ticks, idx = zoh_ticks(t, start, int(ends.max()))
    sup = StaticSuppressor(40, DSP_RATE_HZ)
    det = PresenceDetector(DSP_RATE_HZ)
    zeros = [0.0] * 40
    state = None
    e = 0
    for j, (tk, i) in enumerate(zip(ticks.tolist(), idx.tolist())):
        while e < ends.size and ends[e] < tk:                                # windows ending before this tick
            if state is not None and state.calibrated:
                occ[e] = state.occupied
                ratio[e] = state.energy / state.threshold
            e += 1
        frame = CsiFrame(amplitudes=amp[i].tolist(), phases=zeros, sample_index=j, sample_rate_hz=DSP_RATE_HZ)
        residuals, weights = sup.process(frame)
        state = det.process(fuse(residuals, weights))
    while e < ends.size:
        if state is not None and state.calibrated:
            occ[e] = state.occupied
            ratio[e] = state.energy / state.threshold
        e += 1
    return l, occ, ratio


def dsp_link_states(session: oc.Session, ends_ns: np.ndarray, workers: int = 56) -> tuple:
    """Per link and window end: `(occupied [L, N] bool, ratio [L, N])`
    (NaN ratio before a link is calibrated or with no packets)."""
    ends_ns = np.asarray(ends_ns, np.int64)
    if (np.diff(ends_ns) < 0).any():
        raise ValueError("window ends must be in time order")
    _JOB.clear()
    _JOB.update(session=session, start=int(session.segments[0].start_ns), ends=ends_ns)
    n_links = session.links.shape[0]
    occ = np.zeros((n_links, ends_ns.size), bool)
    ratio = np.full((n_links, ends_ns.size), np.nan)
    with multiprocessing.get_context("fork").Pool(max(1, min(workers, n_links))) as pool:
        for l, o, r in pool.imap_unordered(_dsp_link, range(n_links)):
            occ[l], ratio[l] = o, r
    _JOB.clear()
    return occ, ratio


def dsp_presence(session: oc.Session, ends_ns: np.ndarray, workers: int = 56) -> dict:
    """The DSP baseline's room decision (majority of calibrated links) and
    per-link decisions. Refuses a window end inside the calibration span."""
    start = int(session.segments[0].start_ns)
    if (np.asarray(ends_ns) < start + int(CALIB_S * 1e9)).any():
        raise ValueError("a window ends inside the DSP's warm-up and calibration span")
    occ, ratio = dsp_link_states(session, ends_ns, workers)
    live = np.isfinite(ratio)
    n_live = live.sum(0)
    votes = (occ & live).sum(0)
    decision = votes * 2 > n_live
    with np.errstate(all="ignore"):
        score = np.nanmedian(np.where(live, ratio, np.nan), axis=0)
    score = np.where(np.isfinite(score), score, 0.0)
    # presence_prob carries the decision exactly at the 0.5 rule and orders windows by the median ratio
    # within each decision (for AUC the score is the median ratio itself).
    prob = np.where(decision, 0.5 + 0.5 * _squash(score), 0.5 * _squash(score))
    return {"presence_prob": prob, "count_prob": None, "auc_score": score, "decision": decision,
            "link_occupied": occ, "link_ratio": ratio, "links_live": n_live, "uses_calibration": True}


def _squash(x: np.ndarray) -> np.ndarray:
    """A monotone map of [0, inf) into [0, 1)."""
    x = np.maximum(np.asarray(x, np.float64), 0.0)
    return x / (1.0 + x)


# --------------------------------------------------------------------------
# 2. WLM gate (RSSI)
# --------------------------------------------------------------------------


def placeholder_geometry(session: oc.Session):
    """A localize `Geometry` with the session's nodes evenly spaced on the
    perimeter of its declared L x W room. Positions are placeholders (the
    dataset has none); only the link order and count matter to the gate."""
    from wifisense.localize.geometry import Geometry, Node, Room
    length, width = session.room_m
    per = 2 * (length + width)
    nodes = []
    for i in range(session.n_nodes):
        d = per * (i + 0.5) / session.n_nodes
        if d < length:
            x, y = d, 0.0
        elif d < length + width:
            x, y = length, d - length
        elif d < 2 * length + width:
            x, y = length - (d - length - width), width
        else:
            x, y = 0.0, width - (d - 2 * length - width)
        nodes.append(Node(i, float(x), float(y)))
    return Geometry(Room(width_m=float(length), height_m=float(width)), nodes)


def wlm_gate(session: oc.Session, ends_ns: np.ndarray, length_s: float, *, gate_z: float = 3.0,
             floor_db2: float = 0.25, min_windows: int = 10) -> dict:
    """spec/locate.md §5 step 1 on the RSSI signal at every window end (module doc)."""
    from wifisense.localize.measure import LinkMeasurement, RssiSignal
    from ml.train.baselines import _Frame
    ends_ns = np.asarray(ends_ns, np.int64)
    start = int(session.segments[0].start_ns)
    cal_end = start + int(CALIB_S * 1e9)
    if (ends_ns < cal_end).any():
        raise ValueError("a window ends inside the calibration span")
    geom = placeholder_geometry(session)
    link = np.repeat(np.arange(session.links.shape[0]), np.diff(session.offsets))
    order = np.lexsort((link, session.t_ns))                                 # time order, links break ties
    t, li = session.t_ns[order], link[order]
    rssi = session.rssi[order].astype(np.int64)
    nonzero = (session.iq[order] != 0).any(axis=(1, 2))
    cal = (t >= start) & (t < cal_end)
    # Per-link packet rate in the calibration span (kept packets): sizes min_samples.
    rates = np.bincount(li[cal], minlength=session.links.shape[0]) / CALIB_S
    rate = float(np.median(rates[rates > 0]))
    meas = LinkMeasurement(geom, window_s=float(length_s), sample_rate_hz=rate, signal=RssiSignal())
    tx, rx = session.links[:, 0].astype(int), session.links[:, 1].astype(int)
    meas.calibrate_from(((float(tt) / 1e9, int(tx[l]), int(rx[l]), _Frame(bool(z), int(r)))
                         for tt, l, z, r in zip(t[cal], li[cal], nonzero[cal], rssi[cal])), min_windows=min_windows)
    null = np.asarray(meas.baseline_std_vector(), np.float64)
    after = np.flatnonzero(t >= cal_end)
    score = np.zeros(ends_ns.size)
    fired = np.zeros(ends_ns.size, bool)
    p = 0
    for e, t_end in enumerate(ends_ns.tolist()):
        while p < after.size and t[after[p]] < t_end:
            q = after[p]
            meas.update(float(t[q]) / 1e9, int(tx[li[q]]), int(rx[li[q]]), _Frame(bool(nonzero[q]), int(rssi[q])))
            p += 1
        y = np.asarray(meas.vector(float(t_end) / 1e9), np.float64)
        ok = null > 0
        fired[e] = bool((ok & (y > gate_z * null + floor_db2)).any())
        score[e] = float(((y[ok] - floor_db2) / null[ok]).max()) if ok.any() else 0.0
    prob = np.where(fired, 0.5 + 0.5 * _squash(score / gate_z), 0.5 * _squash(score / gate_z))
    return {"presence_prob": prob, "count_prob": None, "auc_score": score, "decision": fired,
            "calibrated_links": int(sum(meas.calibrated_mask())), "rate_hz": rate, "uses_calibration": True}


# --------------------------------------------------------------------------
# 3. Variance features, 4. prior
# --------------------------------------------------------------------------


def variance_features(sigma: np.ndarray, link_mask: np.ndarray) -> np.ndarray:
    """[W, 9]: the statistics of `s_l = mean(sigma over the reference bins)`
    over each window's used links (spec/model.md §3.3's first 8) and the
    used-link count. `sigma` [W, L, R] (float16 ok), `link_mask` [W, L]."""
    from ml.train.baselines import _stats
    s = np.asarray(sigma, np.float32).astype(np.float64).mean(-1)
    used = np.asarray(link_mask, bool)
    return np.concatenate([_stats(s, used), used.sum(1, keepdims=True).astype(np.float64)], axis=1)


def fit_variance(x_train: np.ndarray, presence: np.ndarray, count: np.ndarray, x_eval: np.ndarray, seed: int, *,
                 n_jobs: int = 16) -> dict:
    """Forest and MLP fitted on `(x_train, labels)`, applied to `x_eval`:
    `{"forest": pred, "mlp": pred}` with `presence_prob` and `count_prob`."""
    import warnings

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.exceptions import ConvergenceWarning
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    out = {}
    for name in ("forest", "mlp"):
        preds = {}
        for task, y in (("presence", presence), ("count", count)):
            if name == "forest":
                clf = RandomForestClassifier(n_estimators=300, random_state=seed, n_jobs=n_jobs)
            else:
                clf = make_pipeline(StandardScaler(), MLPClassifier((128, 128), max_iter=200, early_stopping=False,
                                                                     random_state=seed))
            y = np.asarray(y, np.int64)
            if np.unique(y).size < 2:
                p = np.zeros((x_eval.shape[0], 4))
                p[:, int(y[0])] = 1.0
            else:
                with warnings.catch_warnings():
                    # 200 epochs is the fixed recipe (no REAL validation room to stop on); sklearn warns when
                    # the loss is still moving then, which is expected and not an error.
                    warnings.simplefilter("ignore", ConvergenceWarning)
                    clf.fit(x_train, y)
                pr = clf.predict_proba(x_eval)
                p = np.zeros((x_eval.shape[0], 4))
                p[:, np.asarray(clf.classes_, np.int64)] = pr
            preds[task] = p
        out[name] = {"presence_prob": preds["presence"][:, 1], "count_prob": preds["count"], "uses_calibration": False}
    return out


def prior(presence_train: np.ndarray, count_train: np.ndarray, n_eval: int) -> dict:
    """The training windows' majority class (presence and count), constant."""
    pres = float(np.mean(presence_train) >= 0.5)
    c = int(np.bincount(np.asarray(count_train, np.int64), minlength=4).argmax())
    cp = np.zeros((n_eval, 4))
    cp[:, c] = 1.0
    return {"presence_prob": np.full(n_eval, pres), "count_prob": cp, "uses_calibration": False}
