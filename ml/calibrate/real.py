"""REAL calibration targets (spec/calibration.md §2): the calibration features
of OpenCSI's HT20 sessions, through Phase 9's adapter (rate cap included),
2 s windows every 2 s wholly inside each labelled segment. REAL."""

from __future__ import annotations

import multiprocessing

import numpy as np

from ml import opencsi as oc
from ml.calibrate import features as cf

__all__ = ["HT20", "STATE_OF", "session_features"]

HT20 = ("envA_S3", "envB_S3prime", "envC_C3")
STATE_OF = {"empty_room": "empty", "occupied_static": "still", "occupied_moving": "moving"}
_S: dict = {}


def _one(end) -> dict:
    s = _S["session"]
    w = oc.window(s, end.t_end_ns, 2.0)
    return cf.window_features(w, STATE_OF[s.segments[end.segment].fine_label])


def session_features(trial: str, *, workers: int = 32) -> dict:
    """`{group: values}` of one session's windows (REAL)."""
    s = oc.load_session(oc.DEFAULT_ROOT, trial)
    if s.label != "REAL":
        raise ValueError(f"{trial}: not REAL")
    ends = oc.window_ends(s, 2.0, 2.0)
    _S.clear()
    _S["session"] = s
    with multiprocessing.get_context("fork").Pool(workers) as pool:
        parts = pool.map(_one, ends, chunksize=8)
    _S.clear()
    return cf.merge(parts)
