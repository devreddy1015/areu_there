"""Phase 8 data (spec/model.md §1, §2 "Training", §5.6): a torch `Dataset`
over a window cache (`ml.train.cache`), the training sampler, the link-
subset and unknown-geometry rules, and the batch. Everything is **SIM**.

**Windows.** `CacheDataset(cache, split, mode="train")` draws from every
cached window of the split (odd `k`, `t_k >= ℓ`); `mode="eval"` uses
decision 5's evaluation windows, every second label tick with `t_k >= 4 s`
(`EVAL_MIN_T`), in cache order (scene, then `k`), the same windows for
every `ℓ <= 4`.

**Sampling.** `TrainSampler` draws, each epoch, `TICKS_PER_SCENE` (64)
cached windows per train scene (without replacement; all of them if a
scene has fewer), shuffled across scenes, from `numpy.random.default_rng((seed,
epoch))`. It yields `(position, draw_seed)` pairs; the dataset takes the
link subset and the geometry flag from `default_rng(draw_seed)`, so an
epoch's samples depend on `(seed, epoch)` only, never on the number of
loader workers. `set_epoch(e)` jumps to epoch `e`; otherwise each new
iteration is the next epoch.

**Link subsets** (training, spec/model.md §2), over the window's *used*
links (preprocessing kept them: at least `min_packets` non-empty bins):
with probability 0.2 only the links of one transmitter, drawn uniformly
among the transmitters that have a used link; with probability 0.1 one
used link drawn uniformly; otherwise each used link is dropped with
probability 0.1, and if all were dropped one drawn uniformly is kept.
**Unknown geometry** with probability 0.1 (drawn after the subset):
`geometry_known` false. Evaluation subsets (spec/model.md §5.6): "all"
(every used link), "tx0" (used links with `tx == 0`, today's one-transmitter
firmware), "single01" (the link (0, 1) only), "single01_nogeo" (and
`geometry_known` false).

**The batch** (`collate`, spec/model.md §1; padded to the batch's largest
kept-link count `L`, at least 1). A sample keeps only its kept links, in
the scene's `(tx, rx)` order, so `link_mask[b, l]` is true exactly for
`l < ` the sample's kept count; padding is 0 (False) in every per-link
array.

| key | shape | dtype |
|---|---|---|
| `amp` | [B, L, 44, T] | float16 (z-scores at R, 0 where masked) |
| `mu`, `sigma` | [B, L, 44] | float16 |
| `sample_mask` | [B, L, T] | bool |
| `link_mask` | [B, L] | bool (present, used and kept) |
| `rssi` | [B, L, T] | float16 (dBm), with `rssi_mask` [B, L, T] bool |
| `link_pos` | [B, L, 2, 3] | float32 (tx, rx positions, m, SCENE) |
| `room` | [B, 3] | float32 (`w, d, h`, m, SCENE) |
| `geometry_known` | [B] | bool |
| `presence` | [B] | float32 |
| `count` | [B] | int64 (0-3) |
| `zone_occupied` | [B, 9] | float32 |
| `zone_reachable` | [B, 9] | bool |
| `xy` | [B, 3, 2] | float32 (m, 0 where absent) |
| `person_mask` | [B, 3] | bool |
| `walked` | [B, 3] | bool (walking at some label tick in `(t_k - ℓ, t_k]`) |
| `wlm_published`, `rti_published` | [B] | bool (the stored outcome at `k`) |
| `wlm_xy`, `rti_xy` | [B, 2] | float32 (m, 0 unless published) |
| `wlm_radius`, `rti_radius` | [B] | float32 (m, 0 unless published) |
| `meta` | list of B dicts | `scene` (dataset index), `k`, `n_nodes`, `l_shape`, `split`, `window` (cache row), `links` (kept `(tx, rx)`, [L_b, 2]), `subset` |

`link_pos` and `room` keep their true values when `geometry_known` is
false (so no consumer divides by a zeroed room); the model must not read
them then (spec/model.md §2.2), and the xy and zone losses skip those
samples. On request (`include=`), ablation inputs are added: "gain_db"
[B, L, T] float16 (spec/model.md §5.5), "phase" [B, L, 44, T, 2] float16
and "phase_mask" [B, L, 44, T] bool (§5.4; a cache built with
`phase=True`), "bin_count" [B, L, T] uint8, "links" [B, L, 2] int64 (-1
padding). They are absent otherwise, so the default batch is exactly
spec/model.md §1's.

    from ml.train import cache, data
    c = cache.open_cache("/dev/shm/saans-phase8-cache/roomsim-v1/2")
    train = data.CacheDataset(c, "train", mode="train")
    loader = data.make_loader(train, batch_size=128, sampler=data.TrainSampler(train, seed=0), num_workers=8)
    test = data.CacheDataset(c, "test", mode="eval", subset="tx0")
    for batch in data.make_loader(test, batch_size=256, num_workers=8): ...
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ml.train import cache as cache_mod

__all__ = [
    "LABEL", "EVAL_MIN_T", "TICKS_PER_SCENE", "P_ONE_TX", "P_ONE_LINK", "P_DROP", "P_UNKNOWN_GEOMETRY",
    "EVAL_SUBSETS", "EXTRA_KEYS", "BATCH_KEYS", "link_subset", "eval_subset", "CacheDataset", "TrainSampler",
    "collate", "make_loader", "to_device", "eval_window_indices",
]

LABEL = "SIM"
#: spec/model.md decision 5: evaluation windows are every second label tick with t_k >= 4 s.
EVAL_MIN_T = 4.0
#: spec/model.md §1: cached ticks drawn per train scene per epoch.
TICKS_PER_SCENE = 64
#: spec/model.md §2 "Training": link subsets and unknown geometry.
P_ONE_TX = 0.2
P_ONE_LINK = 0.1
P_DROP = 0.1
P_UNKNOWN_GEOMETRY = 0.1
EVAL_SUBSETS = ("all", "tx0", "single01", "single01_nogeo")
EXTRA_KEYS = ("gain_db", "phase", "bin_count", "links")
#: The batch keys of spec/model.md §1, in its table's order.
BATCH_KEYS = ("amp", "mu", "sigma", "sample_mask", "link_mask", "rssi", "rssi_mask", "link_pos", "room",
              "geometry_known", "presence", "count", "zone_occupied", "zone_reachable", "xy", "person_mask",
              "walked", "wlm_published", "wlm_xy", "wlm_radius", "rti_published", "rti_xy", "rti_radius", "meta")


# --------------------------------------------------------------------------
# the rules
# --------------------------------------------------------------------------


def link_subset(tx: np.ndarray, used: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """spec/model.md §2's training link subset over the used links: [L]
    bool, the links kept. Draw order (fixed, for reproducibility): one
    uniform for the branch, then the branch's draws."""
    used = np.asarray(used, bool)
    tx = np.asarray(tx)
    cand = np.flatnonzero(used)
    u = rng.random()
    if cand.size == 0:
        return used.copy()
    if u < P_ONE_TX:
        txs = np.unique(tx[cand])
        t = txs[rng.integers(txs.size)]
        return used & (tx == t)
    if u < P_ONE_TX + P_ONE_LINK:
        keep = np.zeros_like(used)
        keep[cand[rng.integers(cand.size)]] = True
        return keep
    keep = used & ~(rng.random(used.shape[0]) < P_DROP)
    if not keep.any():
        keep[cand[rng.integers(cand.size)]] = True
    return keep


def eval_subset(links: np.ndarray, used: np.ndarray, subset: str) -> tuple:
    """spec/model.md §5.6's evaluation subsets: `(keep [L] bool,
    geometry_known)`. "all": every used link; "tx0": used links with
    `tx == 0`; "single01" and "single01_nogeo": the used link (0, 1) only
    (none when preprocessing dropped it), the latter with geometry unknown."""
    used = np.asarray(used, bool)
    links = np.asarray(links)
    if subset in (None, "all"):
        return used.copy(), True
    if subset == "tx0":
        return used & (links[:, 0] == 0), True
    if subset in ("single01", "single01_nogeo"):
        return used & (links[:, 0] == 0) & (links[:, 1] == 1), subset == "single01"
    raise ValueError(f"subset {subset!r} is not one of {EVAL_SUBSETS}")


def eval_window_indices(c, split: str) -> np.ndarray:
    """Decision 5's evaluation windows of `split` in a cache: odd `k` with
    `t_k >= 4 s`, in cache order."""
    return c.window_indices(split, min_t=EVAL_MIN_T)


# --------------------------------------------------------------------------
# the dataset
# --------------------------------------------------------------------------


class CacheDataset(Dataset):
    """Samples of one split of a SIM window cache.

    `mode="train"`: every cached window of the split is a candidate (pair it
    with `TrainSampler`), the link-subset and unknown-geometry rules apply
    (`augment`, default on). `mode="eval"`: decision 5's windows (`t_k >=
    4 s`), deterministic, with an evaluation `subset` (EVAL_SUBSETS).
    `windows` overrides the window list (cache window numbers). `include`
    adds ablation inputs (EXTRA_KEYS). An integer key is a position in
    `self.windows` (no augmentation unless a draw seed comes with it); a
    `(position, draw_seed)` key (TrainSampler) applies the training rules
    with `default_rng(draw_seed)`.
    """

    def __init__(self, cache, split: str | None, *, mode: str = "eval", subset: str = "all", windows=None,
                 augment: bool | None = None, include=(), min_t: float | None = None):
        if isinstance(cache, str):
            cache = cache_mod.open_cache(cache)
        if mode not in ("train", "eval"):
            raise ValueError(f"mode {mode!r} is not 'train' or 'eval'")
        if subset not in EVAL_SUBSETS:
            raise ValueError(f"subset {subset!r} is not one of {EVAL_SUBSETS}")
        if mode == "train" and subset != "all":
            raise ValueError("training uses every link (the subsets are drawn by the training rule)")
        include = tuple(include)
        unknown = [k for k in include if k not in EXTRA_KEYS]
        if unknown:
            raise ValueError(f"include {unknown} not in {EXTRA_KEYS}")
        if "phase" in include and "phase" not in cache.meta["row_files"]:
            raise ValueError(f"{cache.dir}: not a phase cache (build_cache(..., phase=True))")
        self.cache = cache
        self.split = split
        self.mode = mode
        self.subset = subset
        self.augment = (mode == "train") if augment is None else bool(augment)
        self.include = include
        if windows is not None:
            self.windows = np.asarray(windows, np.int64)
        else:
            if min_t is None and mode == "eval":
                min_t = EVAL_MIN_T
            self.windows = cache.window_indices(split, min_t=min_t).astype(np.int64)
        self.T = cache.T
        self._split_names = cache_mod.SPLIT_NAMES

    @property
    def label(self) -> str:
        return LABEL

    def __len__(self) -> int:
        return int(self.windows.size)

    def scene_positions(self) -> dict:
        """dataset scene index -> positions in `self.windows` (for the sampler)."""
        scenes = self.cache.windows["scene"][self.windows]
        order = np.argsort(scenes, kind="stable")
        uniq, start = np.unique(scenes[order], return_index=True)
        bounds = list(start) + [order.size]
        return {int(s): order[bounds[i]:bounds[i + 1]] for i, s in enumerate(uniq)}

    def __getitem__(self, key) -> dict:
        if isinstance(key, (tuple, list)):
            pos, draw = int(key[0]), int(key[1])
            rng = np.random.default_rng(draw) if self.augment else None
        else:
            pos, rng = int(key), None
        return self.sample(int(self.windows[pos]), rng)

    def sample(self, w: int, rng: np.random.Generator | None = None) -> dict:
        """The sample of cache window `w`: the training rules with `rng`
        (when augmenting), else the dataset's evaluation subset."""
        c = self.cache
        win = c.windows
        rows = c.rows
        r0 = int(win["row_offset"][w])
        n = int(win["n_links"][w])
        s = int(win["scene_row"][w])
        links, link_pos = c.scene_links(s)
        used = np.asarray(rows["link_mask"][r0:r0 + n], bool)
        if rng is not None:
            keep = link_subset(links[:, 0], used, rng)
            known = not (rng.random() < P_UNKNOWN_GEOMETRY)
        else:
            keep, known = eval_subset(links, used, self.subset)
        idx = np.flatnonzero(keep)
        if idx.size == n:
            sel = slice(r0, r0 + n)
        else:
            sel = r0 + idx
        out = {
            "amp": np.asarray(rows["amp"][sel]),
            "mu": np.asarray(rows["mu"][sel]),
            "sigma": np.asarray(rows["sigma"][sel]),
            "sample_mask": np.asarray(rows["sample_mask"][sel]),
            "rssi": np.asarray(rows["rssi"][sel]),
            "rssi_mask": np.asarray(rows["rssi_mask"][sel]),
            "link_pos": link_pos[idx].astype(np.float32),
            "room": c.scenes["room"][s].astype(np.float32),
            "geometry_known": bool(known),
            "presence": np.float32(win["presence"][w]),
            "count": np.int64(win["count"][w]),
            "zone_occupied": win["zone_occupied"][w].astype(np.float32),
            "zone_reachable": c.scenes["zone_reachable"][s].astype(bool),
            "xy": win["xy"][w].astype(np.float32),
            "person_mask": win["person_mask"][w].astype(bool),
            "walked": win["walked"][w].astype(bool),
        }
        for name in ("wlm", "rti"):
            pub = int(win[f"{name}_reason"][w]) == 0
            out[f"{name}_published"] = pub
            out[f"{name}_xy"] = (win[f"{name}_xy"][w].astype(np.float32) if pub else np.zeros(2, np.float32))
            out[f"{name}_radius"] = np.float32(win[f"{name}_radius"][w]) if pub else np.float32(0.0)
        for name in self.include:
            if name == "phase":
                out["phase"] = np.asarray(rows["phase"][sel])
                out["phase_mask"] = np.asarray(rows["phase_mask"][sel])
            elif name == "links":
                out["links"] = links[idx].astype(np.int64)
            else:
                out[name] = np.asarray(rows[name][sel])
        out["meta"] = {
            "scene": int(win["scene"][w]), "k": int(win["k"][w]), "n_nodes": int(c.scenes["n_nodes"][s]),
            "l_shape": bool(c.scenes["l_shape"][s]), "split": self._split_names[int(win["split"][w])],
            "window": int(w), "links": links[idx].astype(np.int64),
            "subset": "train" if rng is not None else self.subset, "label": LABEL,
        }
        return out


# --------------------------------------------------------------------------
# sampling
# --------------------------------------------------------------------------


class TrainSampler(Sampler):
    """Each epoch: `ticks_per_scene` windows per scene of the dataset
    (without replacement), shuffled together, from `default_rng((seed,
    epoch))`; yields `(position, draw_seed)`. Each `__iter__` is the next
    epoch unless `set_epoch` was called since."""

    def __init__(self, dataset: CacheDataset, ticks_per_scene: int = TICKS_PER_SCENE, seed: int = 0):
        self.positions = dataset.scene_positions()
        self.ticks_per_scene = int(ticks_per_scene)
        self.seed = int(seed)
        self.epoch = 0
        self._len = sum(min(self.ticks_per_scene, len(p)) for p in self.positions.values())

    def set_epoch(self, epoch: int) -> None:
        """The next iteration is epoch `epoch`."""
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self._len

    def epoch_keys(self, epoch: int) -> list:
        """The `(position, draw_seed)` keys of epoch `epoch`."""
        rng = np.random.default_rng((self.seed, int(epoch)))
        picks = []
        for scene in sorted(self.positions):
            p = self.positions[scene]
            m = min(self.ticks_per_scene, p.size)
            picks.append(p[rng.choice(p.size, size=m, replace=False)])
        pos = np.concatenate(picks) if picks else np.zeros(0, np.int64)
        pos = pos[rng.permutation(pos.size)]
        draws = rng.integers(0, 2 ** 63 - 1, size=pos.size, dtype=np.int64)
        return list(zip(pos.tolist(), draws.tolist()))

    def __iter__(self):
        keys = self.epoch_keys(self.epoch)
        self.epoch += 1
        return iter(keys)


# --------------------------------------------------------------------------
# the batch
# --------------------------------------------------------------------------


def collate(samples: list) -> dict:
    """Pad a list of samples into spec/model.md §1's batch (module
    docstring): torch tensors, padded to the largest kept-link count (at
    least 1), padding zero/False, plus `meta` (a list)."""
    b = len(samples)
    if b == 0:
        raise ValueError("empty batch")
    lk = [s["amp"].shape[0] for s in samples]
    big = max(1, max(lk))
    t = samples[0]["amp"].shape[-1]
    nref = samples[0]["amp"].shape[1]
    amp = np.zeros((b, big, nref, t), np.float16)
    mu = np.zeros((b, big, nref), np.float16)
    sigma = np.zeros((b, big, nref), np.float16)
    smask = np.zeros((b, big, t), bool)
    lmask = np.zeros((b, big), bool)
    rssi = np.zeros((b, big, t), np.float16)
    rmask = np.zeros((b, big, t), bool)
    lpos = np.zeros((b, big, 2, 3), np.float32)
    extras = {}
    first = samples[0]
    if "gain_db" in first:
        extras["gain_db"] = np.zeros((b, big, t), np.float16)
    if "bin_count" in first:
        extras["bin_count"] = np.zeros((b, big, t), np.uint8)
    if "phase" in first:
        extras["phase"] = np.zeros((b, big, nref, t, 2), np.float16)
        extras["phase_mask"] = np.zeros((b, big, nref, t), bool)
    if "links" in first:
        extras["links"] = np.full((b, big, 2), -1, np.int64)
    for i, s in enumerate(samples):
        n = lk[i]
        if s["amp"].shape[-1] != t:
            raise ValueError("samples of one batch must share T (one cache, one ℓ)")
        amp[i, :n] = s["amp"]
        mu[i, :n] = s["mu"]
        sigma[i, :n] = s["sigma"]
        smask[i, :n] = s["sample_mask"]
        lmask[i, :n] = True
        rssi[i, :n] = s["rssi"]
        rmask[i, :n] = s["rssi_mask"]
        lpos[i, :n] = s["link_pos"]
        for name, arr in extras.items():
            arr[i, :n] = s[name]
    out = {
        "amp": torch.from_numpy(amp), "mu": torch.from_numpy(mu), "sigma": torch.from_numpy(sigma),
        "sample_mask": torch.from_numpy(smask), "link_mask": torch.from_numpy(lmask),
        "rssi": torch.from_numpy(rssi), "rssi_mask": torch.from_numpy(rmask),
        "link_pos": torch.from_numpy(lpos),
        "room": torch.from_numpy(np.stack([s["room"] for s in samples]).astype(np.float32)),
        "geometry_known": torch.tensor([bool(s["geometry_known"]) for s in samples], dtype=torch.bool),
        "presence": torch.tensor([float(s["presence"]) for s in samples], dtype=torch.float32),
        "count": torch.tensor([int(s["count"]) for s in samples], dtype=torch.int64),
        "zone_occupied": torch.from_numpy(np.stack([s["zone_occupied"] for s in samples]).astype(np.float32)),
        "zone_reachable": torch.from_numpy(np.stack([s["zone_reachable"] for s in samples]).astype(bool)),
        "xy": torch.from_numpy(np.stack([s["xy"] for s in samples]).astype(np.float32)),
        "person_mask": torch.from_numpy(np.stack([s["person_mask"] for s in samples]).astype(bool)),
        "walked": torch.from_numpy(np.stack([s["walked"] for s in samples]).astype(bool)),
    }
    for name in ("wlm", "rti"):
        out[f"{name}_published"] = torch.tensor([bool(s[f"{name}_published"]) for s in samples], dtype=torch.bool)
        out[f"{name}_xy"] = torch.from_numpy(np.stack([s[f"{name}_xy"] for s in samples]).astype(np.float32))
        out[f"{name}_radius"] = torch.tensor([float(s[f"{name}_radius"]) for s in samples], dtype=torch.float32)
    for name, arr in extras.items():
        out[name] = torch.from_numpy(arr)
    out["meta"] = [s["meta"] for s in samples]
    return out


def make_loader(dataset: CacheDataset, batch_size: int, *, sampler=None, num_workers: int = 8,
                pin_memory: bool = True, persistent_workers: bool | None = None, prefetch_factor: int | None = 4,
                drop_last: bool = False) -> DataLoader:
    """A DataLoader with `collate`. With no sampler the order is the
    dataset's (evaluation); pass `TrainSampler(dataset, seed=...)` to train.
    Workers are forked and map the cache read-only."""
    if persistent_workers is None:
        persistent_workers = num_workers > 0
    kw = {}
    if num_workers > 0 and prefetch_factor is not None:
        kw["prefetch_factor"] = prefetch_factor
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler, shuffle=False, num_workers=num_workers,
                      collate_fn=collate, pin_memory=pin_memory, persistent_workers=persistent_workers,
                      drop_last=drop_last, multiprocessing_context="fork" if num_workers > 0 else None, **kw)


def to_device(batch: dict, device, non_blocking: bool = True) -> dict:
    """Move a batch's tensors to `device` (meta stays a list)."""
    return {k: (v.to(device, non_blocking=non_blocking) if torch.is_tensor(v) else v) for k, v in batch.items()}
