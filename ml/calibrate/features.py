"""Calibration features (spec/calibration.md §2): per used link and window,
from the shared preprocessing (`bin_hz=None`, `ref_bins="shared40"`, so a
packet-level spread independent of how packets fall into bins), for SIM
and REAL alike:

| group | value |
|---|---|
| `log_sigma_<state>` | `log10` of the mean over the 40 reference bins of `sigma` (the window's per-subcarrier spread of the gain-normalised amplitude), per state: empty, still, moving |
| `fs_db_empty` | frequency selectivity: the std over the 40 bins of `20 log10(mu)` (dB), empty windows |
| `rssi_std_<state>` | the std of the window's RSSI values (dBm), empty and moving |
| `rssi_mean_empty` | the mean RSSI (dBm), empty windows |

A link contributes only when preprocessing used it (`min_packets = 3`).
"""

from __future__ import annotations

import numpy as np

from ml import preprocess as pp

__all__ = ["CONFIG", "GROUPS", "STATES", "window_features", "merge"]

CONFIG = pp.PreprocessConfig(phase=False, bin_hz=None, ref_bins="shared40", min_packets=3)
STATES = ("empty", "still", "moving")
GROUPS = ("log_sigma_empty", "log_sigma_still", "log_sigma_moving", "fs_db_empty", "rssi_std_empty",
          "rssi_std_moving", "rssi_mean_empty")


def window_features(win: pp.CsiWindow, state: str) -> dict:
    """`{group: values [n_used_links]}` of one window in `state` (empty, still, moving)."""
    if state not in STATES:
        raise ValueError(f"state {state!r} is not one of {STATES}")
    out = pp.preprocess(win, CONFIG, dtype=np.float64)
    ref = out["ref"]
    used = out["link_mask"]
    feats = {}
    sig = out["sigma"][used][:, ref].mean(1)
    feats[f"log_sigma_{state}"] = np.log10(np.maximum(sig, 1e-6))
    rs, rm = [], []
    for l in np.flatnonzero(used):
        r = win.rssi[l][win.rssi_mask[l]]
        rs.append(float(np.std(r, ddof=1)) if r.size >= 2 else np.nan)
        rm.append(float(np.mean(r)) if r.size else np.nan)
    rs, rm = np.asarray(rs), np.asarray(rm)
    if state == "empty":
        mu = out["mu"][used][:, ref]
        feats["fs_db_empty"] = np.std(20.0 * np.log10(np.maximum(mu, 1e-6)), axis=1, ddof=1)
        feats["rssi_std_empty"] = rs[np.isfinite(rs)]
        feats["rssi_mean_empty"] = rm[np.isfinite(rm)]
    elif state == "moving":
        feats["rssi_std_moving"] = rs[np.isfinite(rs)]
    return feats


def merge(parts: list) -> dict:
    """Concatenate `{group: values}` dicts; every group present (empty arrays allowed)."""
    out = {g: [] for g in GROUPS}
    for p in parts:
        for g, v in p.items():
            out[g].append(np.asarray(v, np.float64))
    return {g: (np.concatenate(v) if v else np.zeros(0)) for g, v in out.items()}
