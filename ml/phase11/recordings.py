"""The owner's recordings (spec/recording.md §2–§4): a session directory
(`capture.pcap`, `labels.jsonl`, optional `scene.json`) read with
`ml.preprocess.from_pcap` (today's firmware: one transmitter, so
`single_tx=True`; label REAL) into a window cache in **exactly Phase 9's
REAL cache format** (`ml.phase9.real_cache`), so `ml.phase9.real_data`,
`finetune.predict` and the variance baselines run on it unchanged.

    from ml.phase11 import recordings as rec
    meta = rec.build_cache("saans-data/recordings/bedroom/2026-11-02_1", 2.0)
    c = rec.load_cache("saans-data/recordings/bedroom/2026-11-02_1", 2.0)

Windows follow spec/phase9.md §3: every 0.5 s wholly inside one labelled
segment; evaluation windows every 1 s starting at least 90 s after the
first segment's start (the DSP baseline's calibration span).
"""

from __future__ import annotations

import json
import os
import time

import numpy as np

from ml import preprocess as pp
from ml.phase9 import real_cache as rc

__all__ = ["LABEL", "FINE_LABELS", "ROOT", "load_labels", "window_ends", "build_cache", "load_cache"]

LABEL = "REAL"
FINE_LABELS = {"empty_room": (False, 0, False), "occupied_static": (True, 1, False),
               "occupied_moving": (True, 1, True)}
ROOT = "/home/Aman/are_u_theone/runs/phase11/cache"
_NS = 1_000_000_000


def load_labels(path: str) -> list:
    """spec/recording.md §3: `[(start_ns, end_ns, fine_label, count, note)]` in time order;
    refuses an unknown label, a non-integer time, an empty or overlapping segment."""
    segs = []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            a = json.loads(line)
            s, e, lab = a.get("start_ns"), a.get("end_ns"), a.get("fine_label")
            if lab not in FINE_LABELS:
                raise ValueError(f"{path}:{i}: fine_label {lab!r} is not one of {sorted(FINE_LABELS)}")
            if not (isinstance(s, int) and isinstance(e, int)) or isinstance(s, bool) or isinstance(e, bool):
                raise ValueError(f"{path}:{i}: start_ns and end_ns must be integer epoch ns")
            if e <= s:
                raise ValueError(f"{path}:{i}: an empty segment")
            count = a.get("count", FINE_LABELS[lab][1])
            if not isinstance(count, int) or isinstance(count, bool) or not 0 <= count <= 3:
                raise ValueError(f"{path}:{i}: count {count!r} is not an integer 0..3")
            if (count == 0) != (lab == "empty_room"):
                raise ValueError(f"{path}:{i}: count {count} contradicts {lab}")
            segs.append((s, e, lab, count, str(a.get("note", ""))))
    segs.sort()
    for a, b in zip(segs, segs[1:]):
        if b[0] < a[1]:
            raise ValueError(f"{path}: segments overlap")
    if not segs:
        raise ValueError(f"{path}: no segment")
    return segs


def window_ends(segments: list, length_s: float, hop_s: float) -> list:
    """`[(t_end_ns, segment index)]`: `t_end = start + ℓ + j hop <= end`."""
    ln, hop = pp._length_ns(length_s), pp._length_ns(hop_s)
    out = []
    for i, (s, e, *_rest) in enumerate(segments):
        t = s + ln
        while t <= e:
            out.append((t, i))
            t += hop
    return out


def _paths(session_dir: str, length_s: float, root: str) -> tuple:
    name = os.path.relpath(os.path.abspath(session_dir), os.path.dirname(os.path.dirname(os.path.abspath(session_dir))))
    stem = os.path.join(root, name.replace(os.sep, "__") + f"_{float(length_s):g}")
    return stem + ".npz", stem + ".json", name


def build_cache(session_dir: str, length_s: float, *, root: str = ROOT, port: int = 5566,
                overwrite: bool = False) -> dict:
    """Read, window, preprocess and cache one session (REAL)."""
    t0 = time.time()
    arrays_path, meta_path, name = _paths(session_dir, length_s, root)
    if os.path.exists(meta_path) and not overwrite:
        raise FileExistsError(f"{meta_path} exists; pass overwrite=True")
    segs = load_labels(os.path.join(session_dir, "labels.jsonl"))
    scene_path = os.path.join(session_dir, "scene.json")
    scene = scene_path if os.path.exists(scene_path) else None
    ends = window_ends(segs, length_s, rc.TRAIN_HOP_S)
    if not ends:
        raise ValueError(f"{session_dir}: no segment is longer than the window")
    stats = pp.PcapStats()
    wins = pp.from_pcap(os.path.join(session_dir, "capture.pcap"), length_s, [t for t, _ in ends], port=port,
                        single_tx=True, label=LABEL, scene=scene, stats=stats)
    outs = []
    for w in wins:
        if w.label != LABEL:
            raise AssertionError(f"{session_dir}: a window labelled {w.label!r}")
        o = pp.preprocess(w, rc.PREPROCESS)
        ref = o["ref"]
        outs.append({
            "amp": o["amp"][:, ref, :].astype(np.float16), "mu": o["mu"][:, ref].astype(np.float16),
            "sigma": o["sigma"][:, ref].astype(np.float16), "rssi": o["rssi"].astype(np.float16),
            "sample_mask": o["sample_mask"], "rssi_mask": o["rssi_mask"],
            "bin_count": np.minimum(o["bin_count"], 255).astype(np.uint8), "link_mask": o["link_mask"],
        })
    arr = {k: np.stack([o[k] for o in outs]) for k in outs[0]}
    t_end = np.array([t for t, _ in ends], np.int64)
    seg = np.array([g for _, g in ends], np.int16)
    ln = pp._length_ns(length_s)
    starts = np.array([segs[g][0] for g in seg], np.int64)
    step = (t_end - ln - starts) // pp._length_ns(rc.TRAIN_HOP_S)
    hop_ratio = int(round(rc.EVAL_HOP_S / rc.TRAIN_HOP_S))
    eval_ = (step % hop_ratio == 0) & (t_end - ln >= segs[0][0] + pp._length_ns(rc.EVAL_SKIP_S))
    arr.update({
        "t_end_ns": t_end, "segment": seg, "eval": eval_,
        "presence": np.array([FINE_LABELS[segs[g][2]][0] for g in seg], np.uint8),
        "count": np.array([segs[g][3] for g in seg], np.uint8),
        "moving": np.array([FINE_LABELS[segs[g][2]][2] for g in seg], bool),
        "links": wins[0].links.astype(np.int16),
    })
    os.makedirs(root, exist_ok=True)
    np.savez(arrays_path + ".tmp.npz", **arr)
    os.replace(arrays_path + ".tmp.npz", arrays_path)
    room = [0.0, 0.0]
    if scene is not None:
        with open(scene, encoding="utf-8") as fh:
            sj = json.load(fh)
        room = [float(sj["room"]["w_m"]), float(sj["room"]["d_m"])] if "room" in sj else room
    meta = {
        "label": LABEL, "format": rc._FORMAT, "spec": "spec/recording.md §4", "trial": name,
        "environment": name.split(os.sep)[0], "chip": "owner firmware (edge/)", "layout": "esp32-fw56",
        "length_s": float(length_s), "T": int(arr["amp"].shape[-1]), "n_ref": int(arr["amp"].shape[2]),
        "ref_k": list(pp.SHARED40), "preprocess": {"phase": rc.PREPROCESS.phase, "bin_hz": rc.PREPROCESS.bin_hz,
                                                    "ref_bins": rc.PREPROCESS.ref_bins,
                                                    "min_packets": rc.PREPROCESS.min_packets},
        "train_hop_s": rc.TRAIN_HOP_S, "eval_hop_s": rc.EVAL_HOP_S, "eval_skip_s": rc.EVAL_SKIP_S,
        "n_windows": int(t_end.size), "n_eval": int(eval_.sum()), "n_links": int(arr["links"].shape[0]),
        "segments": [[s, e, lab] for s, e, lab, _c, _n in segs], "room_m": room, "t0_s": 0,
        "session_stats": {"records": stats.records, "malformed": stats.malformed, "duplicates": stats.duplicates},
        "build_s": round(time.time() - t0, 2),
    }
    with open(meta_path + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, indent=1)
    os.replace(meta_path + ".tmp", meta_path)
    return meta


def load_cache(session_dir: str, length_s: float, root: str = ROOT) -> dict:
    arrays_path, meta_path, _ = _paths(session_dir, length_s, root)
    with open(meta_path, encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("label") != LABEL:
        raise ValueError(f"{meta_path}: labelled {meta.get('label')!r}, not REAL")
    with np.load(arrays_path) as z:
        out = {k: z[k] for k in z.files}
    out["meta"] = meta
    return out
