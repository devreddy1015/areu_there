"""Metrics of the Phase 10a Sionna audit (spec/sionna_audit.md §4). All SIM.

Both simulators describe a path by a complex coefficient `a` (antenna
patterns, reflection/transmission coefficients and spreading, no
propagation phase) and a length `L = C * tau`; a path's term at frequency
`f` is `a * exp(-j 2 pi f tau)` in both (Room Sim §1: a longer path lags;
Sionna `Paths.cfr`). Everything here is numpy float64.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = ["C", "cfr", "power_db", "rms_delay_spread", "selectivity_db", "cfr_agreement",
           "summarize", "match_paths", "wrap_phase"]

C = 299792458.0


def cfr(a, tau, freqs) -> np.ndarray:
    """`H(f) = sum_i a_i exp(-j 2 pi f tau_i)` at every `f` in `freqs` (Hz)."""
    a = np.asarray(a, dtype=np.complex128)
    tau = np.asarray(tau, dtype=np.float64)
    freqs = np.asarray(freqs, dtype=np.float64)
    if a.size == 0:
        return np.zeros(freqs.shape, dtype=np.complex128)
    return np.exp(-2j * np.pi * np.outer(freqs, tau)) @ a


def power_db(p) -> float:
    """`10 log10(p)`, `-inf` for zero power."""
    p = float(p)
    return 10.0 * math.log10(p) if p > 0.0 else float("-inf")


def rms_delay_spread(p, tau) -> float:
    """Power-weighted RMS delay spread (s) of paths with powers `p` and
    delays `tau`: `sqrt(sum p (tau - mean)^2 / sum p)`. NaN with no power."""
    p = np.asarray(p, dtype=np.float64)
    tau = np.asarray(tau, dtype=np.float64)
    P = float(np.sum(p))
    if not P > 0.0:
        return float("nan")
    m = float(np.sum(p * tau)) / P
    return float(np.sqrt(np.sum(p * (tau - m) ** 2) / P))


def selectivity_db(H) -> float:
    """Frequency selectivity: the standard deviation (ddof 0) over the
    subcarriers of `20 log10 |H|`, dB. NaN if any `|H|` is 0."""
    mag = np.abs(np.asarray(H))
    if np.any(mag <= 0.0):
        return float("nan")
    return float(np.std(20.0 * np.log10(mag)))


def cfr_agreement(H_ref, H_other) -> dict:
    """CFR magnitude agreement across subcarriers, dB domain:
    `corr` the Pearson correlation of `20 log10 |H|` (NaN when either is
    flat to 1e-9 dB), `rms_db` the RMS of their difference after removing
    its mean (the mean level), `offset_db` that mean (`ref - other`)."""
    m1 = np.abs(np.asarray(H_ref))
    m2 = np.abs(np.asarray(H_other))
    if np.any(m1 <= 0.0) or np.any(m2 <= 0.0):
        return {"corr": float("nan"), "rms_db": float("nan"), "offset_db": float("nan")}
    d1 = 20.0 * np.log10(m1)
    d2 = 20.0 * np.log10(m2)
    diff = d1 - d2
    off = float(np.mean(diff))
    rms = float(np.sqrt(np.mean((diff - off) ** 2)))
    if np.std(d1) < 1e-9 or np.std(d2) < 1e-9:
        corr = float("nan")
    else:
        corr = float(np.corrcoef(d1, d2)[0, 1])
    return {"corr": corr, "rms_db": rms, "offset_db": off}


def summarize(values) -> dict:
    """Distribution of the finite values: n, median, p10, p90, mean, min,
    max (NaN when there are none), plus `n_nonfinite` (NaN or +-inf values,
    e.g. a link with no path in one simulator). Label SIM."""
    v = np.asarray(list(values), dtype=np.float64)
    fin = v[np.isfinite(v)]
    out = {"label": "SIM", "n": int(fin.size), "n_nonfinite": int(v.size - fin.size)}
    if fin.size:
        out.update(median=float(np.median(fin)), p10=float(np.percentile(fin, 10)),
                   p90=float(np.percentile(fin, 90)), mean=float(np.mean(fin)),
                   min=float(np.min(fin)), max=float(np.max(fin)))
    else:
        out.update(median=float("nan"), p10=float("nan"), p90=float("nan"), mean=float("nan"),
                   min=float("nan"), max=float("nan"))
    return out


def wrap_phase(x) -> np.ndarray:
    """Wrap radians to (-pi, pi]."""
    x = np.asarray(x, dtype=np.float64)
    return -((-x + np.pi) % (2.0 * np.pi) - np.pi)


def match_paths(rs_keys, rs_L, s_keys, s_L) -> dict:
    """Match Room Sim paths to Sionna paths by their sequence of reflecting
    shapes (refractions ignored).

    `rs_keys[i]`/`s_keys[j]` are tuples of shape names (`()` for the direct
    path). With both endpoints fixed, a sequence of mirror planes fixes the
    path (the image method), so a key names at most one path on each side;
    should one repeat, the nearest length wins. The length difference is
    returned, not used as a gate: a wall whose single Sionna surface sits on
    another face than Room Sim's bounce gives the same path a few
    centimetres longer or shorter. Returns `pairs` as `(i, j, dL)`
    (`dL = L_rs - L_s`, m) and the unmatched indices of each side."""
    by_key: dict = {}
    for j, k in enumerate(s_keys):
        by_key.setdefault(k, []).append(j)
    used = set()
    pairs = []
    rs_only = []
    for i, k in enumerate(rs_keys):
        best = None
        for j in by_key.get(k, ()):
            if j in used:
                continue
            d = float(rs_L[i]) - float(s_L[j])
            if best is None or abs(d) < abs(best[0]):
                best = (d, j)
        if best is None:
            rs_only.append(i)
        else:
            used.add(best[1])
            pairs.append((i, best[1], best[0]))
    s_only = [j for j in range(len(s_keys)) if j not in used]
    return {"pairs": pairs, "rs_only": rs_only, "s_only": s_only}
