"""Phase 9 data (spec/phase9.md §3–§4): REAL windows as the model's batch.

`RealDataset` serves windows of one or more REAL caches (`real_cache.load`)
as samples with exactly the keys of `ml.train.data.CacheDataset.sample`, so
`ml.train.data.collate` and the model take them unchanged:

- every window has `geometry_known` false (node positions are unknown);
  `link_pos` is 0 and `room` the declared L x W with a 2.5 m placeholder
  height (never read when the geometry is unknown; it only keeps a room
  that nothing divides by zero);
- labels: `presence`, `count` (0 or 1); `walked` = [moving, 0, 0];
  `person_mask` = [count > 0, 0, 0] with `xy` 0; `zone_occupied` 0 and
  `zone_reachable` all false (no zone is scored or trained);
- no stored WLM/RTI outcome (`*_published` false);
- training (`augment=True`) applies Phase 8's link-subset rule
  (`ml.train.data.link_subset`) over the window's used links; evaluation
  keeps every used link.

`fraction_positions` draws spec/phase9.md §4.5's fine-tuning blocks.
"""

from __future__ import annotations

import math
import zlib

import numpy as np
import torch

from ml.train import data as d8

__all__ = ["LABEL", "ROOM_HEIGHT_M", "RealDataset", "FixedStepSampler", "fraction_positions", "truth"]

LABEL = "REAL"
ROOM_HEIGHT_M = 2.5


class RealDataset(torch.utils.data.Dataset):
    """Windows `(cache index, window)` of REAL caches as model samples."""

    def __init__(self, caches: list, items, *, augment: bool = False):
        for c in caches:
            if c["meta"]["label"] != LABEL:
                raise ValueError(f"cache of {c['meta']['trial']} is labelled {c['meta']['label']!r}, not REAL")
        self.caches = caches
        self.items = [(int(a), int(b)) for a, b in items]
        self.augment = bool(augment)

    @property
    def label(self) -> str:
        return LABEL

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, key) -> dict:
        if isinstance(key, (tuple, list)):
            pos, draw = int(key[0]), int(key[1])
            rng = np.random.default_rng(draw) if self.augment else None
        else:
            pos, rng = int(key), None
        ci, w = self.items[pos]
        return self.sample(ci, w, rng)

    def sample(self, ci: int, w: int, rng: np.random.Generator | None = None) -> dict:
        c = self.caches[ci]
        links = c["links"]
        used = c["link_mask"][w].astype(bool)
        keep = d8.link_subset(links[:, 0], used, rng) if rng is not None else used
        idx = np.flatnonzero(keep)
        length, width = c["meta"]["room_m"]
        count = int(c["count"][w])
        out = {
            "amp": c["amp"][w, idx], "mu": c["mu"][w, idx], "sigma": c["sigma"][w, idx],
            "sample_mask": c["sample_mask"][w, idx], "rssi": c["rssi"][w, idx], "rssi_mask": c["rssi_mask"][w, idx],
            "link_pos": np.zeros((idx.size, 2, 3), np.float32),
            "room": np.array([length, width, ROOM_HEIGHT_M], np.float32),
            "geometry_known": False,
            "presence": np.float32(c["presence"][w]),
            "count": np.int64(count),
            "zone_occupied": np.zeros(9, np.float32),
            "zone_reachable": np.zeros(9, bool),
            "xy": np.zeros((3, 2), np.float32),
            "person_mask": np.array([count > 0, False, False]),
            "walked": np.array([bool(c["moving"][w]), False, False]),
        }
        for name in ("wlm", "rti"):
            out[f"{name}_published"] = False
            out[f"{name}_xy"] = np.zeros(2, np.float32)
            out[f"{name}_radius"] = np.float32(0.0)
        out["meta"] = {
            "scene": ci, "k": int(w), "n_nodes": int(links.max()) + 1, "l_shape": False, "split": "real", "window": int(w), "links": links[idx].astype(np.int64),
            "subset": "train" if rng is not None else "all", "label": LABEL, "trial": c["meta"]["trial"],
        }
        return out


class FixedStepSampler(torch.utils.data.Sampler):
    """`steps * batch` keys `(position, draw_seed)` drawn uniformly with
    replacement from `default_rng(seed)`: a fixed number of optimiser steps
    whatever the training set's size (spec/phase9.md §4.3)."""

    def __init__(self, n: int, steps: int, batch: int, seed: int):
        if n <= 0:
            raise ValueError("an empty training set")
        self.n, self.total, self.seed = int(n), int(steps) * int(batch), int(seed)

    def __len__(self) -> int:
        return self.total

    def __iter__(self):
        rng = np.random.default_rng((self.seed, 9))
        pos = rng.integers(0, self.n, size=self.total)
        draws = rng.integers(0, 2 ** 63 - 1, size=self.total, dtype=np.int64)
        return iter(list(zip(pos.tolist(), draws.tolist())))


def _offset_rng(seed: int, trial: str, segment: int) -> np.random.Generator:
    return np.random.default_rng((int(seed), zlib.crc32(trial.encode()), int(segment)))


def fraction_positions(cache: dict, fraction: float, seed: int) -> np.ndarray:
    """spec/phase9.md §4.5: from every labelled segment of one session's
    training windows, a contiguous block of `max(1, ceil(f n_segment))`
    windows (cache order is time order within a segment) at an offset drawn
    from `(seed, trial, segment)`. Returns window indices, ascending."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"fraction {fraction!r} is not in (0, 1]")
    seg = cache["segment"]
    out = []
    for g in np.unique(seg):
        idx = np.flatnonzero(seg == g)
        m = max(1, math.ceil(fraction * idx.size - 1e-9))
        start = int(_offset_rng(seed, cache["meta"]["trial"], int(g)).integers(0, idx.size - m + 1))
        out.append(idx[start:start + m])
    return np.sort(np.concatenate(out))


def truth(cache: dict, windows: np.ndarray) -> dict:
    """The metrics' truth for `windows` of one cache (spec/phase9.md §6)."""
    w = np.asarray(windows, np.int64)
    return {"presence": cache["presence"][w].astype(bool), "count": cache["count"][w].astype(np.int64),
            "moving": cache["moving"][w].astype(bool), "segment": cache["segment"][w].astype(np.int64),
            "leading_empty": cache["segment"][w] == 0, "t_end_ns": cache["t_end_ns"][w]}
