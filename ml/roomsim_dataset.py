"""Load dataset v1 (spec/dataset.md), the scenes `saans gen-dataset` writes.

Everything here is **SIM**: the scene geometry was drawn at random, the
people and the CSI were simulated by Room Sim, and the baseline is Locate
run on those simulated packets. Nothing in a shard was measured in a room,
and every function below says so.

A dataset directory holds `manifest.json` (§7.3, written last: a directory
without it is an unfinished run) and one `.npz` shard per scene (§7.2).
`load_scene` checks the shard's sha256 against the manifest, its
`meta_json` label and spec string, and every member's name, dtype and
shape against §7.2 before it returns anything.

The CSI is stored as the ESP32 sends it: int8 `(imag, real)` pairs, one
column per TDMA superframe (§4.1). `amplitude` and `window` turn it into
the brief's `[links, subcarriers, time]` tensor without resampling (§4.3).
Masked window slots are NaN unless the caller asks for another fill.

Two things to know before training on it (spec/dataset.md §6, §12):

  * **AGC gain steps.** With impairments on, Room Sim's AGC steps a link's
    gain by about ±3 dB between packets, uniformly across subcarriers (in
    about 17% of 2 s link-windows). Raw amplitude variance therefore
    tracks gain steps as much as people; normalise each packet's gain
    (e.g. divide by its mean over active subcarriers) before reading
    motion into it. That normalisation is Phase 7's, not this loader's.
  * **`calib_*` is an idealised empty room.** The calibration run shares
    the occupied run's seed, so its clutter, devices and noise are the same
    streams exactly, with no drift. It is there to recompute the WLM
    baseline; a model that uses it sees more than a deployment would and is
    reported separately.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import operator
import os
from typing import NamedTuple

import numpy as np

__all__ = [
    "LABEL", "SPEC", "MEMBERS", "META_KEYS", "REASONS", "POSTURES", "SPLITS", "Window",
    "load_manifest", "load_scene", "load_scene_json", "impairments", "amplitude", "window", "labels_at",
    "baseline_at", "splits", "zone_of",
]

LABEL = "SIM"
SPEC = "dataset v1"

#: Transmissions per second, `1 / slot_s` (§4.1). The window and label
#: arithmetic below is written for it, so a shard with another `slot_s` is
#: refused rather than cut on the wrong grid.
TX_PER_S = 80
SLOT_S = 0.0125

N_SUB = 56
MAX_PEOPLE = 3
SPLITS = ("train", "val", "test")
#: The manifest's `impairments` profiles (§8), plus "off" for a manifest
#: from before the field existed whose `impair` is false (Room Sim's master
#: switch: no impairments and no noise).
IMPAIRMENTS = ("on", "hardware_off", "off")
#: `wlm_reason` / `rti_reason` codes (§6).
REASONS = ("estimate", "no_disturbance", "not_significant", "implausible_radius", "warming_up")
#: `posture` codes (§5); 255 is absent.
POSTURES = ("stand", "sit", "lie")

#: Every shard member in §7.2 order (§4.2, §5, §6, then the scene block):
#: name, numpy `dtype.str` (the npy header's `descr`), shape. A string in a
#: shape is a dimension bound by its first appearance and checked at every
#: later one: L links, M columns, K label ticks, W = S/2 windows, N nodes,
#: Mc calibration columns, V footprint vertices, B meta bytes.
MEMBERS = (
    ("csi_iq", "|i1", ("L", N_SUB, "M", 2)),
    ("csi_mask", "|u1", ("L", "M")),
    ("rssi", "|i1", ("L", "M")),
    ("tx_slot", "|u1", ("L",)),
    ("subcarrier_k", "|i1", (N_SUB,)),
    ("subcarrier_active", "|u1", (N_SUB,)),
    ("count", "|u1", ("K",)),
    ("presence", "|u1", ("K",)),
    ("xy", "<f4", ("K", MAX_PEOPLE, 2)),
    ("person_mask", "|u1", ("K", MAX_PEOPLE)),
    ("walking", "|u1", ("K", MAX_PEOPLE)),
    ("posture", "|u1", ("K", MAX_PEOPLE)),
    ("zone", "|i1", ("K", MAX_PEOPLE)),
    ("zone_occupied", "|u1", ("K", 9)),
    ("zone_single", "|i1", ("K",)),
    ("t_label", "<f8", ("K",)),
    ("zone_floor_frac", "<f4", (9,)),
    ("zone_reachable", "|u1", (9,)),
    ("wlm_xy", "<f4", ("K", 2)),
    ("rti_xy", "<f4", ("K", 2)),
    ("wlm_radius", "<f4", ("K",)),
    ("rti_radius", "<f4", ("K",)),
    ("wlm_reason", "|u1", ("K",)),
    ("rti_reason", "|u1", ("K",)),
    ("calib_baseline", "<f8", ("L",)),
    ("calib_null_sigma", "<f8", ("L",)),
    ("calib_mask", "|u1", ("L",)),
    ("calib_iq", "|i1", ("L", N_SUB, "Mc", 2)),
    ("calib_mask_m", "|u1", ("L", "Mc")),
    ("calib_rssi", "|i1", ("L", "Mc")),
    ("links", "|u1", ("L", 2)),
    ("node_ids", "|u1", ("N",)),
    ("node_pos", "<f4", ("N", 3)),
    ("room_size", "<f4", (3,)),
    ("footprint", "<f4", ("V", 2)),
    ("window_end_tick", "<i4", ("W",)),
    ("meta_json", "|u1", ("B",)),
)
#: The keys §7.2 puts in `meta_json`.
META_KEYS = ("label", "spec", "scene_index", "scene_id", "scene_sha256", "roomsim_seed", "n_nodes",
             "n_people", "slot_s", "n_tx", "columns", "calib_m0", "impair")


class Window(NamedTuple):
    """One window of SIM CSI ending at a label tick (§4.3).

    Slot `j` of every link holds that link's one transmission (if any) in
    the `j`-th of `T` consecutive superframe-long intervals ending at
    `t_k`, i.e. `n` in `[80 t_k - (T - j) N, 80 t_k - (T - 1 - j) N)`: the
    newest column is always slot `T - 1`, and a link with one column fewer
    than `T` has slot 0 masked. Links are thus aligned in time to within
    one superframe, never resampled.
    """

    amplitude: np.ndarray  #: [L, 56, T] float32, `fill` (NaN) where masked
    mask: np.ndarray       #: [L, T] bool, a received packet inside the window
    link_mask: np.ndarray  #: [L] bool, the link has at least one packet
    column: np.ndarray     #: [L, T] int64, the superframe `m` of each slot (may be < 0 near t = 0)
    inside: np.ndarray     #: [L, T] bool, the slot's transmission is in the window and the stream (sent,
    #:                        lost or not): `T` or `T - 1` per link, less only before the stream starts


def load_manifest(dataset_dir: str) -> dict:
    """Read and check `DIR/manifest.json` of a SIM dataset (§7.3).

    Refuses a directory without a manifest (an unfinished run), a manifest
    not labelled SIM or not `dataset v1`, one whose scene list has a
    repeated index or disagrees with `n_scenes`, an unknown `impairments`
    profile, and a missing `baseline_valid`. The data it describes are SIM.
    """
    path = os.path.join(dataset_dir, "manifest.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"{path}: no manifest (an unfinished or missing SIM dataset run, §7.3)")
    with open(path, encoding="utf-8") as f:
        manifest = json.load(f)
    _check_label(manifest, path)
    entries = manifest.get("scenes")
    if not isinstance(entries, list) or manifest.get("n_scenes", len(entries)) != len(entries):
        raise ValueError(f"{path}: 'scenes' is not the list of n_scenes = {manifest.get('n_scenes')} entries")
    indices = [int(e["index"]) for e in entries]
    if len(set(indices)) != len(indices):
        raise ValueError(f"{path}: repeated scene index")
    impairments(manifest)
    if not isinstance(manifest.get("baseline_valid"), bool):
        raise ValueError(f"{path}: 'baseline_valid' is not true or false (§8)")
    return manifest


def impairments(manifest: dict) -> str:
    """The SIM dataset's impairment profile (§8): "on", "hardware_off"
    (hardware impairments off, the receiver's noise kept), or "off" for an
    older manifest without the field whose `impair` is false."""
    profile = manifest.get("impairments")
    if profile is None:
        profile = "on" if manifest.get("impair", True) else "off"
    if profile not in IMPAIRMENTS:
        raise ValueError(f"manifest impairments {profile!r} is not one of {IMPAIRMENTS}")
    return profile


def load_scene(dataset_dir: str, index: int, manifest: dict | None = None) -> dict:
    """Load SIM scene `index`'s shard: every §7.2 member as a numpy array.

    Checks, before returning: the shard's sha256 equals the manifest
    entry's `shard_sha256`; `meta_json` says `label` SIM and `spec`
    `dataset v1`, has every §7.2 key and names this scene (index, id,
    scene sha256); every member is present with its §7.2 dtype and shape
    (and nothing else is); `slot_s` is 0.0125 s.
    Adds `scene["meta"]` (the parsed `meta_json`), `scene["entry"]` (the
    manifest entry) and `scene["dataset"]` (`impairments`, `baseline_valid`
    from the manifest). Pass `manifest` to skip re-reading it. All SIM.

    The CSI carries Room Sim's AGC gain steps (see the module docstring);
    `calib_*` is an idealised, drift-free empty room, for recomputing the
    WLM baseline only: a model that uses it is reported separately (§6).
    """
    if manifest is None:
        manifest = load_manifest(dataset_dir)
    entry = _entry(manifest, index)
    path = _dataset_path(dataset_dir, entry["shard"])
    with open(path, "rb") as f:
        data = f.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != entry["shard_sha256"]:
        raise ValueError(f"{path}: sha256 {digest} does not match the manifest's {entry['shard_sha256']}")
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        scene = {name: npz[name] for name in npz.files}
    dims = _check_members(scene, path)
    meta = json.loads(scene["meta_json"].tobytes().decode("utf-8"))
    _check_label(meta, path + " meta_json")
    missing = [key for key in META_KEYS if key not in meta]
    if missing:
        raise ValueError(f"{path}: meta_json lacks {missing} (§7.2)")
    named = (meta["scene_index"], meta["scene_id"], meta["scene_sha256"])
    if named != (entry["index"], entry["id"], entry["scene_sha256"]):
        raise ValueError(f"{path}: meta_json names scene {named}, the manifest "
                         f"{(entry['index'], entry['id'], entry['scene_sha256'])}")
    n_tx = int(meta["n_tx"])
    if meta["n_nodes"] != dims["N"] or n_tx != dims["N"] or dims["L"] != n_tx * (n_tx - 1):
        raise ValueError(f"{path}: {dims['N']} nodes, {dims['L']} links, meta n_nodes {meta['n_nodes']} "
                         f"n_tx {n_tx}: not the all-txrx mesh of §3")
    if abs(float(meta["slot_s"]) - SLOT_S) > 1e-12:
        raise ValueError(f"{path}: slot_s {meta['slot_s']}, this loader cuts the {SLOT_S} s grid only")
    if dims["M"] != meta["columns"] or dims["M"] != -(-40 * dims["K"] // n_tx):
        raise ValueError(f"{path}: {dims['M']} columns, meta {meta['columns']}, ceil(80 S / N) "
                         f"{-(-40 * dims['K'] // n_tx)}")
    if int(scene["tx_slot"].max(initial=0)) >= n_tx:
        raise ValueError(f"{path}: a tx_slot is not below n_tx = {n_tx}")
    scene["meta"] = meta
    scene["entry"] = entry
    scene["dataset"] = {"impairments": impairments(manifest), "baseline_valid": bool(manifest["baseline_valid"])}
    return scene


def load_scene_json(dataset_dir: str, index: int, manifest: dict | None = None) -> dict:
    """SIM scene `index`'s scene file, `DIR/scenes/<id>.json` (§7.1): the
    generated geometry (walls, furniture, doors, windows, nodes, people's
    starts) as parsed JSON. Checks its sha256 against the manifest's
    `scene_sha256` and that its provenance is "generated"."""
    if manifest is None:
        manifest = load_manifest(dataset_dir)
    entry = _entry(manifest, index)
    path = _dataset_path(dataset_dir, os.path.join("scenes", entry["id"] + ".json"))
    with open(path, "rb") as f:
        data = f.read()
    if hashlib.sha256(data).hexdigest() != entry["scene_sha256"]:
        raise ValueError(f"{path}: sha256 does not match the manifest's scene_sha256")
    doc = json.loads(data.decode("utf-8"))
    if doc.get("provenance") != "generated":
        raise ValueError(f"{path}: provenance {doc.get('provenance')!r}, not 'generated' (SIM)")
    return doc


def amplitude(iq: np.ndarray) -> np.ndarray:
    """SIM CSI amplitude, exactly as Locate computes it (§4.2, spec/locate.md §4.3).

    `iq[..., 0]` is the imaginary part and `iq[..., 1]` the real part, the
    datagram's order. Both are widened to int32 before squaring (so -128
    gives 128, not an int8 overflow), and the result is
    `sqrt(float64(re*re + im*im))` with shape `iq.shape[:-1]`.
    """
    iq = np.asarray(iq)
    if iq.dtype != np.int8 or iq.shape[-1:] != (2,):
        raise ValueError(f"amplitude: want int8 [..., 2] (imag, real), got {iq.dtype} {iq.shape}")
    im = iq[..., 0].astype(np.int32)
    re = iq[..., 1].astype(np.int32)
    return np.sqrt((re * re + im * im).astype(np.float64))


def window(scene: dict, k: int, length_s: float = 2.0, fill: float = np.nan) -> Window:
    """The SIM CSI window of length `length_s` ending at label tick `k` (§4.3).

    Per link `l`, holds the columns `m` whose transmission
    `n = m*N + tx_slot[l]` lies in `[80 (t_k - length_s), 80 t_k)`, chosen
    in integers (`80 t_k = 40 (k + 1)`). `T = ceil(80 length_s / N)`; a
    link with one column fewer has slot 0 masked (see `Window`), and slots
    before the stream starts (`k` near 0) are masked too. A masked slot's
    amplitude is `fill`, NaN by default so a forgotten mask fails loudly;
    pass `fill=0.0` for a model that consumes the mask itself. Null
    subcarriers are not masked here: find them by `subcarrier_active`.
    The float32 amplitudes are `amplitude()`'s float64 values rounded once.

    The amplitudes are raw: with impairments on they carry Room Sim's AGC
    gain steps (about ±3 dB, uniform across subcarriers, in about 17% of
    2 s link-windows), so normalise each packet's gain before reading
    motion from their variance (Phase 7).
    """
    k = _tick(scene, k)
    halves = _halves(length_s)
    n_tx = int(scene["meta"]["n_tx"])
    n_hi = 40 * (k + 1)                 # 80 t_k
    n_lo = n_hi - 40 * halves           # 80 (t_k - length_s)
    t = -(-40 * halves // n_tx)         # ceil(80 length_s / N)
    slot = scene["tx_slot"].astype(np.int64)
    last = (n_hi - 1 - slot) // n_tx    # each link's last column with n < n_hi (floor division)
    column = last[:, None] - np.arange(t - 1, -1, -1)
    n_cols = scene["csi_mask"].shape[1]
    inside = (column * n_tx + slot[:, None] >= n_lo) & (column >= 0) & (column < n_cols)
    rows = np.arange(len(slot))[:, None]
    col = np.clip(column, 0, n_cols - 1)
    mask = inside & (scene["csi_mask"][rows, col] != 0)
    iq = np.moveaxis(scene["csi_iq"][rows, :, col], 1, 2)   # [L, T, 56, 2] -> [L, 56, T, 2]
    amp = np.where(mask[:, None, :], amplitude(iq), fill).astype(np.float32)
    return Window(amp, mask, mask.any(axis=1), column, inside)


def labels_at(scene: dict, k: int) -> dict:
    """The SIM truth labels at label tick `k` (§5), unpadded.

    Returns `t`, `count`, `presence`, `zone_occupied` ([9] uint8, the zone
    target), `zone_single` (-1 unless exactly one person) and `people`: per
    present person `index`, `x`, `y` (m), `zone`, `walking`, `posture`
    (`stand`/`sit`/`lie`). Index order carries no meaning.
    """
    k = _tick(scene, k)
    people = []
    for i in range(MAX_PEOPLE):
        if scene["person_mask"][k, i]:
            x, y = (float(v) for v in scene["xy"][k, i])
            people.append({"index": i, "x": x, "y": y, "zone": int(scene["zone"][k, i]),
                           "walking": bool(scene["walking"][k, i]), "posture": POSTURES[scene["posture"][k, i]]})
    return {"label": LABEL, "k": k, "t": float(scene["t_label"][k]), "count": int(scene["count"][k]),
            "presence": bool(scene["presence"][k]), "zone_occupied": scene["zone_occupied"][k].copy(),
            "zone_single": int(scene["zone_single"][k]), "people": people}


def baseline_at(scene: dict, k: int) -> dict:
    """Locate v1's stored SIM baseline at label tick `k` (§6): WLM and RTI.

    Each is `{"published", "reason", "xy", "radius_m"}`; `xy` and
    `radius_m` are None unless published, and a published estimate always
    carries both (a shard where they disagree with the reason is refused),
    so no caller can draw a centre without its radius. `valid` is the
    manifest's `baseline_valid` (§8: every scene's calibration has some
    link with `null_sigma > 0`). The radius is Locate's uncalibrated
    model, and the calibration it used is idealised (§6). SIM.
    """
    k = _tick(scene, k)
    valid = bool(scene["dataset"]["baseline_valid"])
    return {"label": LABEL, "k": k, "t": float(scene["t_label"][k]), "valid": valid,
            "wlm": _estimate(scene, "wlm", k), "rti": _estimate(scene, "rti", k)}


def splits(manifest: dict) -> dict:
    """The manifest's fixed SIM scene splits (§7.3): `{"train", "val", "test"}`
    -> sorted scene indices. Checks they are disjoint, cover every scene and
    agree with each entry's `split`. Splits are by scene, never by window."""
    out = {name: sorted(int(i) for i in manifest["splits"][name]) for name in SPLITS}
    seen = [i for name in SPLITS for i in out[name]]
    entries = {int(e["index"]): e for e in manifest["scenes"]}
    if len(seen) != len(set(seen)) or set(seen) != set(entries):
        raise ValueError("manifest splits are not a partition of its scenes")
    for name in SPLITS:
        for i in out[name]:
            if entries[i].get("split") != name:
                raise ValueError(f"scene {i} is in split {name!r} but its entry says {entries[i].get('split')!r}")
    return out


def zone_of(x: float, y: float, w: float, d: float) -> int:
    """The SIM zone cell of `(x, y)` in the 3 x 3 grid over (0,0)..(w,d) (§5):
    `3*row + col`, `col = min(floor(3x/w), 2)`, `row = min(floor(3y/d), 2)`."""
    col = min(math.floor(3.0 * x / w), 2)
    row = min(math.floor(3.0 * y / d), 2)
    return 3 * row + col


def _check_label(doc: dict, where: str) -> None:
    """Refuse a manifest or meta block not labelled SIM / `dataset v1`."""
    if doc.get("label") != LABEL:
        raise ValueError(f"{where}: label {doc.get('label')!r}, not {LABEL!r}; refusing data not marked SIM")
    if doc.get("spec") != SPEC:
        raise ValueError(f"{where}: spec {doc.get('spec')!r}, this loader reads {SPEC!r}")


def _entry(manifest: dict, index: int) -> dict:
    """SIM scene `index`'s manifest entry."""
    index = operator.index(index)
    for e in manifest["scenes"]:
        if e["index"] == index:
            return e
    raise KeyError(f"no scene {index} in the manifest")


def _dataset_path(dataset_dir: str, name: str) -> str:
    """A SIM dataset file's path: `name` relative to the dataset (a bare
    shard name falls back to `shards/`); never outside the dataset."""
    root = os.path.realpath(dataset_dir)
    path = os.path.realpath(os.path.join(root, name))
    if not os.path.exists(path) and os.path.dirname(name) == "":
        path = os.path.realpath(os.path.join(root, "shards", name))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(f"{name!r} is outside {dataset_dir}")
    return path


def _check_members(scene: dict, where: str) -> dict:
    """§7.2: a SIM shard holds exactly `MEMBERS`, each with its dtype and
    shape; returns the bound dimensions."""
    names = [m[0] for m in MEMBERS]
    missing = [n for n in names if n not in scene]
    extra = [n for n in scene if n not in names]
    if missing or extra:
        raise ValueError(f"{where}: members missing {missing}, unexpected {extra} (§7.2)")
    dims: dict = {}
    for name, descr, shape in MEMBERS:
        arr = scene[name]
        if arr.dtype.str != descr or arr.ndim != len(shape):
            raise ValueError(f"{where}: {name} is {arr.dtype.str} {arr.shape}, §7.2 says {descr} {shape}")
        for want, got in zip(shape, arr.shape):
            want = dims.setdefault(want, got) if isinstance(want, str) else want
            if got != want:
                raise ValueError(f"{where}: {name} has shape {arr.shape}, §7.2 says {shape} with {dims}")
    if dims["K"] % 4 or dims["W"] != dims["K"] // 4 or dims["V"] not in (4, 6):
        raise ValueError(f"{where}: K = {dims['K']} ticks, {dims['W']} windows, {dims['V']} vertices (§7.2)")
    return dims


def _tick(scene: dict, k: int) -> int:
    """A SIM label tick index, checked against the scene's K."""
    k = operator.index(k)
    if not 0 <= k < len(scene["t_label"]):
        raise IndexError(f"label tick {k} outside 0..{len(scene['t_label']) - 1}")
    return k


def _halves(length_s: float) -> int:
    """A SIM window length in whole half-seconds (§4.3: a multiple of 0.5 s)."""
    halves = round(2 * length_s)
    if halves < 1 or 2 * length_s != halves:
        raise ValueError(f"window length {length_s!r} s is not a positive multiple of 0.5 s")
    return halves


def _estimate(scene: dict, name: str, k: int) -> dict:
    """One stored SIM baseline (`wlm` or `rti`) at tick `k`, checked."""
    reason = int(scene[f"{name}_reason"][k])
    xy = scene[f"{name}_xy"][k]
    radius = float(scene[f"{name}_radius"][k])
    if reason >= len(REASONS):
        raise ValueError(f"{name}_reason[{k}] = {reason} is not a §6 code")
    published = reason == 0
    # §6: centre and radius both finite when published, both NaN otherwise.
    finite = np.isfinite(xy).all() and math.isfinite(radius)
    if not (finite if published else np.isnan(xy).all() and math.isnan(radius)):
        raise ValueError(f"{name} at tick {k}: reason {REASONS[reason]} with centre {xy} radius {radius}")
    return {"published": published, "reason": REASONS[reason],
            "xy": (float(xy[0]), float(xy[1])) if published else None,
            "radius_m": radius if published else None}
