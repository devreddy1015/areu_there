"""Training of model v1 (spec/model.md §2 Training): SIM pre-training, Phase 8.

    .venv/bin/python -m ml.train.train --config CONFIG.json [--lr 1e-3 --d 128 --window-s 2 --seed 0 ...]
    .venv/bin/python -m ml.train.train --synthetic --device cpu --epochs 1     # smoke run on synthetic data

AdamW (weight decay 0.01), a cosine schedule after 500 linear warm-up steps,
batch 128, 30 epochs, gradient clipping at 1.0, fp16 autocast with a
gradient scaler (Turing has no bf16 tensor cores, spec/model.md decision 1),
`torch.compile` with an eager fallback, early stopping on the validation
loss (patience 5). Each run writes `<out_root>/<run_id>/`: `config.json` (the
resolved config, loss weights and link-subset rules included), `log.jsonl`
(one line per epoch: losses, a validation metrics summary, times,
throughput, GPU memory), `best.pt` (the best validation loss), `last.pt`
(resume state, every epoch) and `done.json` (written last: a run without it
is unfinished and resumes from `last.pt`).

Data come from the Phase 8 cache through `ml.train.data` (the data side's
Dataset and collate, which apply spec/model.md §2's link-subset and
unknown-geometry rules in training), or, with `--synthetic`, from
`SyntheticScenes` below: random rooms, nodes and people whose links'
`sigma`, `mu` and RSSI respond to the people near them — the exact §1 batch,
for the tests and smoke runs. Every number either produces is SIM.

Seeds fix the initialisation (`build_model(seed)`), the sampling and
link subsets (the dataset's generator, keyed by seed and epoch) and dropout
(`torch.manual_seed`). GPU kernels are not forced deterministic unless
`deterministic` is set (cuDNN autotuning and atomics may change the last
bits between runs); on the CPU a resumed run equals an uninterrupted one
bit for bit (tested).

For the compiled graph every batch's link axis is padded to 132 (static
shapes) and the link encoder runs on the real links only (`pack_links`,
model.pack_index, its length a dynamic dimension); both leave the outputs
unchanged (tested). Measured on the Quadro RTX 8000 (B = 128, D = 256,
l = 2 s, idle machine): 27.4 ms per step compiled and padded, 50.5 ms
eager; packed, 18.6 ms (6,900 samples/s) and 0.8 GB of GPU memory.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import json
import math
import os
import platform
import random
import socket
import sys
import time
import warnings
from dataclasses import asdict

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.train import model as mdl  # noqa: E402

LABEL = "SIM"
RUNS_ROOT = "/home/Aman/are_u_theone/runs/phase8"
MAX_LINKS = 132                      # 12 nodes x 11 (spec/dataset.md §2.4)
BIN_HZ = 5                           # spec/model.md decision 2: T = 5 l bins

DEFAULTS = {
    "run_id": None,                  # default: derived from dataset, window, width, lr, seed
    "out_root": RUNS_ROOT,
    "dataset": "roomsim-v1",         # or "roomsim-v1-hwoff" (ablation §5.3)
    "cache_dir": None,               # the Phase 8 cache (ml.train.cache); None: the data side's default
    "window_s": 2.0,                 # l, s
    "d": 128,
    "lr": 1e-3,
    "seed": 0,
    "epochs": 30,
    "batch": 128,
    "weight_decay": 0.01,
    "warmup_steps": 500,
    "grad_clip": 1.0,
    "amp": True,                     # fp16 autocast + GradScaler (CUDA only)
    "compile": True,                 # torch.compile, eager fallback on failure
    "compile_mode": "default",
    "patience": 5,
    "dropout": 0.1,
    "heads": 4,
    "layers": 2,
    "ticks_per_scene": 64,           # training draws per train scene per epoch (spec/model.md §1)
    "workers": 6,
    "eval_batch": 256,
    "matcher": "enumerate",          # or "scipy": the same minimum-cost assignment (model.py)
    "loss_weights": dict(mdl.DEFAULT_LOSS_WEIGHTS),
    "subset_rules": {"p_one_tx": 0.2, "p_one_link": 0.1, "p_drop_link": 0.1, "p_unknown_geometry": 0.1},
    "phase": False,                  # ablation §5.4 (needs a cache with phase)
    "gain": False,                   # ablation §5.5 (needs a cache with gain_db)
    "pad_links": MAX_LINKS,          # pad every batch's link axis to this (static shapes for compile); 0 = no
    "pack_links": True,              # encode only the real links (model.pack_index): same outputs, less work
    "pack_bucket": 256,              # the packed link count is padded to a multiple of this
    "max_steps": 0,                  # stop after this many optimiser steps (0 = no limit; smoke runs)
    "max_epoch_steps": 0,            # cap steps per epoch (0 = the full epoch; smoke runs)
    "val_max_batches": 0,            # cap validation batches (0 = every val window; smoke runs)
    "device": "cuda",
    "deterministic": False,
    "synthetic": None,               # a dict of SyntheticScenes sizes, or None for the real cache
    "overwrite": False,
    "interrupt_after_epochs": 0,     # tests: stop after this many epochs as if killed (no done.json)
}


# --------------------------------------------------------------------------
# Synthetic SIM-like scenes (tests, smoke runs): the exact §1 batch
# --------------------------------------------------------------------------


def _perimeter_points(rng: np.random.Generator, w: float, d: float, n: int) -> np.ndarray:
    """n points on the rectangle's edges, 5 cm inside, at least 0.5 m apart in (x, y) when possible."""
    pts = []
    per = 2 * (w + d)
    for _ in range(200 * n):
        s = rng.uniform(0, per)
        if s < w:
            p = (s, 0.05)
        elif s < w + d:
            p = (w - 0.05, s - w)
        elif s < 2 * w + d:
            p = (2 * w + d - s, d - 0.05)
        else:
            p = (0.05, per - s)
        if all((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2 >= 0.25 for q in pts):
            pts.append(p)
        if len(pts) == n:
            break
    while len(pts) < n:
        pts.append((rng.uniform(0, w), rng.uniform(0, d)))
    z = rng.uniform(0.8, 2.2, n)
    return np.column_stack([np.array(pts), z])


def _seg_dist(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Distance from points p [P, 2] to segments a-b [L, 2] -> [P, L]."""
    ab = b - a
    ap = p[:, None, :] - a[None]
    t = np.clip((ap * ab[None]).sum(-1) / np.maximum((ab * ab).sum(-1), 1e-9)[None], 0, 1)
    q = a[None] + t[..., None] * ab[None]
    return np.sqrt(((p[:, None, :] - q) ** 2).sum(-1))


class SyntheticScenes(torch.utils.data.Dataset):
    """Random SIM-like windows with the exact spec/model.md §1 sample layout
    (before collation): rooms 3-10 m, `n_nodes` nodes on the walls, every
    ordered link, 0-3 people. A person near a link's segment raises its
    `sigma`, moves its `mu` and its RSSI, more when walking; `amp` is unit
    z-scores. Learnable, not physical. Labelled SIM like everything here.

    Sample `i` of epoch `e` is drawn from `default_rng((seed, e, i))`, so
    the data are the same for any worker count. `train=True` applies
    spec/model.md §2's link-subset and unknown-geometry rules."""

    def __init__(self, n: int = 256, window_s: float = 2.0, n_nodes=(4, 6), seed: int = 0, train: bool = False,
                 subset_rules: dict | None = None, split: str = "train", phase: bool = False, gain: bool = False,
                 n_scenes: int | None = None):
        self.n = int(n)
        self.t = int(round(BIN_HZ * window_s))
        self.n_nodes = tuple(n_nodes)
        self.seed = int(seed)
        self.train = train
        self.rules = dict(DEFAULTS["subset_rules"], **(subset_rules or {}))
        self.split = split
        self.phase = phase
        self.gain = gain
        self.n_scenes = n_scenes or max(1, self.n // 8)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, i: int) -> dict:
        scene = i % self.n_scenes
        srng = np.random.default_rng((self.seed, 7, scene))                  # the scene: fixed across epochs
        k = 7 + 2 * (i // self.n_scenes)
        rng = np.random.default_rng((self.seed, 11, self.epoch if self.train else 0, i))
        w, d = srng.uniform(3, 10, 2)
        h = srng.uniform(2.5, 3.5)
        n = int(srng.integers(self.n_nodes[0], self.n_nodes[1] + 1))
        nodes = _perimeter_points(srng, w, d, n)
        links = np.array([(a, b) for a in range(n) for b in range(n) if a != b], dtype=np.int64)
        L, T, R = len(links), self.t, mdl.N_REF
        count = int(srng.integers(0, 4))
        xy = np.zeros((3, 2), np.float32)
        xy[:count] = np.column_stack([rng.uniform(0.3, w - 0.3, count), rng.uniform(0.3, d - 0.3, count)])
        walked = np.zeros(3, bool)
        walked[:count] = rng.random(count) < 0.5
        pmask = np.arange(3) < count
        a2, b2 = nodes[links[:, 0], :2], nodes[links[:, 1], :2]
        if count:
            dist = _seg_dist(xy[:count].astype(np.float64), a2, b2)          # [P, L]
            gainp = np.where(walked[:count], 1.5, 0.6)[:, None]
            dist_l = (gainp * np.exp(-dist ** 2 / (2 * 0.5 ** 2))).sum(0)
        else:
            dist_l = np.zeros(L)
        smask = rng.random((L, T)) > 0.01
        smask[:, :3] = True                                                  # at least min_packets bins
        amp = rng.normal(size=(L, R, T))
        amp = np.where(smask[:, None, :], amp, 0.0)
        nv = smask.sum(-1)[:, None, None]
        m = amp.sum(-1, keepdims=True) / nv
        s = np.sqrt((((amp - m) * smask[:, None, :]) ** 2).sum(-1, keepdims=True) / np.maximum(nv - 1, 1))
        amp = np.where(smask[:, None, :], (amp - m) / np.maximum(s, 1e-6), 0.0)
        base = srng.normal(0, 0.25, (L, R))
        mu = 1.0 + base + 0.05 * dist_l[:, None] * srng.normal(0, 1, (L, R))
        sigma = 0.02 * np.exp(rng.normal(0, 0.2, (L, R))) * (1.0 + 4.0 * dist_l[:, None])
        length = np.sqrt(((a2 - b2) ** 2).sum(-1))
        rssi = (-35.0 - 2.0 * length - 1.5 * dist_l)[:, None] + rng.normal(0, 0.3, (L, T)) * (1 + 2 * dist_l)[:, None]
        rssi = np.where(smask, np.round(rssi * 4) / 4, 0.0)
        link_mask = np.ones(L, bool)
        geometry_known = True
        if self.train:
            u = rng.random()
            r = self.rules
            if u < r["p_one_tx"]:
                tx = rng.integers(0, n)
                link_mask = links[:, 0] == tx
            elif u < r["p_one_tx"] + r["p_one_link"]:
                link_mask = np.zeros(L, bool)
                link_mask[rng.integers(0, L)] = True
            else:
                link_mask = rng.random(L) >= r["p_drop_link"]
                if not link_mask.any():
                    link_mask[rng.integers(0, L)] = True
            geometry_known = bool(rng.random() >= r["p_unknown_geometry"])
        keep = np.flatnonzero(link_mask)                                     # links dropped by the rule are removed
        zone_occ = np.zeros(9, np.float32)
        for p in range(count):
            col = min(int(3 * xy[p, 0] / w), 2)
            row = min(int(3 * xy[p, 1] / d), 2)
            zone_occ[3 * row + col] = 1
        reach = np.ones(9, bool)
        if srng.random() < 0.25:                                             # an L-shaped room's lost corner
            corner = int(srng.choice([0, 2, 6, 8]))
            if zone_occ[corner] == 0:
                reach[corner] = False
        out = {
            "amp": amp[keep].astype(np.float16),
            "mu": mu[keep].astype(np.float16),
            "sigma": sigma[keep].astype(np.float16),
            "sample_mask": smask[keep],
            "link_mask": np.ones(len(keep), bool),
            "rssi": rssi[keep].astype(np.float16),
            "rssi_mask": smask[keep].copy(),
            "link_pos": np.stack([nodes[links[keep, 0]], nodes[links[keep, 1]]], 1).astype(np.float32),
            "links": links[keep],
            "room": np.array([w, d, h], np.float32),
            "geometry_known": geometry_known,
            "presence": np.float32(count > 0),
            "count": count,
            "zone_occupied": zone_occ,
            "zone_reachable": reach,
            "xy": xy,
            "person_mask": pmask,
            "walked": walked,
            "wlm_published": False, "wlm_xy": np.zeros(2, np.float32), "wlm_radius": np.float32(0),
            "rti_published": False, "rti_xy": np.zeros(2, np.float32), "rti_radius": np.float32(0),
            "meta": {"scene": scene, "k": k, "n_nodes": n, "l_shape": bool(not reach.all()), "split": self.split},
        }
        if self.phase:
            ph = rng.uniform(-np.pi, np.pi, (len(keep), R, T))
            out["phase"] = np.stack([np.cos(ph), np.sin(ph)], -1).astype(np.float16)
            out["phase_mask"] = np.broadcast_to(smask[keep][:, None, :], (len(keep), R, T)).copy()
        if self.gain:
            out["gain_db"] = np.where(smask[keep], rng.normal(30, 1, (len(keep), T)), 0).astype(np.float16)
        return out


def collate_pad(samples: list) -> dict:
    """Stack samples, padding the link axis to the batch's largest link count
    (zeros / False), as the §1 batch is. `meta` stays a list."""
    n_links = max(s["amp"].shape[0] for s in samples)
    out = {}
    for key in samples[0]:
        vals = [s[key] for s in samples]
        if key == "meta":
            out[key] = list(vals)
            continue
        arrs = [np.asarray(v) for v in vals]
        if key in mdl.PER_LINK_KEYS:
            padded = []
            for a in arrs:
                pad = [(0, n_links - a.shape[0])] + [(0, 0)] * (a.ndim - 1)
                padded.append(np.pad(a, pad))
            arrs = padded
        out[key] = torch.from_numpy(np.stack(arrs))
    return out


# --------------------------------------------------------------------------
# Batches on the device
# --------------------------------------------------------------------------


def to_device(batch: dict, device, non_blocking: bool = True) -> dict:
    return {k: (v.to(device, non_blocking=non_blocking) if isinstance(v, torch.Tensor) else v)
            for k, v in batch.items()}


def forward_batch(b: dict, link_mask_cpu: torch.Tensor, *, n_links: int | None, n_batch: int | None, pack: bool,
                  bucket: int, compiled: bool) -> dict:
    """The forward pass's view of a device batch: links padded to `n_links`
    and samples to `n_batch` (static shapes for a compiled graph), plus
    `pack_index` (from the loader's CPU `link_mask`, no device sync) when
    `pack`; its length is marked dynamic for torch.compile."""
    fb = pad_batch(b, n_links=n_links, n_batch=n_batch)
    if pack:
        idx = mdl.pack_index(link_mask_cpu, fb["amp"].shape[1], bucket=bucket)
        fb["pack_index"] = idx.to(b["amp"].device, non_blocking=True)
        if compiled:
            torch._dynamo.mark_dynamic(fb["pack_index"], 0)
    return fb


def pad_batch(batch: dict, n_links: int | None = None, n_batch: int | None = None) -> dict:
    """Pad per-link tensors' link axis to `n_links` and every tensor's batch
    axis to `n_batch` (padding: zeros, False; a padded sample has no link and
    unknown geometry). For the forward pass only: the model's outputs for real
    samples do not change (tested); losses use the unpadded batch."""
    out = dict(batch)
    for key, v in batch.items():
        if not isinstance(v, torch.Tensor):
            continue
        if n_links and key in mdl.PER_LINK_KEYS and v.shape[1] < n_links:
            pad = torch.zeros((v.shape[0], n_links - v.shape[1]) + tuple(v.shape[2:]), dtype=v.dtype, device=v.device)
            v = torch.cat([v, pad], dim=1)
        if n_batch and v.shape[0] < n_batch:
            pad = torch.zeros((n_batch - v.shape[0],) + tuple(v.shape[1:]), dtype=v.dtype, device=v.device)
            v = torch.cat([v, pad], dim=0)
        out[key] = v
    if n_batch and "room" in out and batch["room"].shape[0] < n_batch:
        out["room"] = out["room"].clone()
        out["room"][batch["room"].shape[0]:] = 1.0
    return out


# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------


class SyntheticSource:
    """Synthetic train/val/test sets (`DEFAULTS['synthetic']` sizes)."""

    def __init__(self, cfg: dict):
        s = dict({"n_train": 512, "n_val": 128, "n_test": 128, "n_nodes": [4, 6]}, **(cfg["synthetic"] or {}))
        self.cfg = cfg
        common = {"window_s": cfg["window_s"], "n_nodes": tuple(s["n_nodes"]), "phase": cfg["phase"],
                  "gain": cfg["gain"]}
        self.train_set = SyntheticScenes(s["n_train"], seed=cfg["seed"], train=True, split="train",
                                         subset_rules=cfg["subset_rules"], **common)
        self.eval_sets = {split: SyntheticScenes(s[f"n_{split}"], seed=cfg["seed"] + 1000 + j, train=False,
                                                 split=split, **common)
                          for j, split in enumerate(("val", "test"))}
        self.t = self.train_set.t

    def describe(self) -> dict:
        return {"kind": "synthetic", "dataset": "synthetic", "n_train": len(self.train_set),
                **{f"n_{k}": len(v) for k, v in self.eval_sets.items()}, "T": self.t}

    def train_loader(self, epoch: int):
        self.train_set.set_epoch(epoch)
        g = torch.Generator().manual_seed(_mix(self.cfg["seed"], epoch))
        return torch.utils.data.DataLoader(self.train_set, batch_size=self.cfg["batch"], shuffle=True,
                                           drop_last=len(self.train_set) >= self.cfg["batch"], generator=g,
                                           collate_fn=collate_pad, num_workers=0)

    def eval_loader(self, split: str, subset: str | None = None, workers: int | None = None):
        if subset not in (None, "all"):
            raise ValueError("link subsets are evaluated on the real cache only")
        return torch.utils.data.DataLoader(self.eval_sets[split], batch_size=self.cfg["eval_batch"], shuffle=False,
                                           collate_fn=collate_pad, num_workers=0)


def _mix(*xs) -> int:
    """A 63-bit seed from integers (stable across runs and platforms)."""
    return int(np.random.SeedSequence([int(x) & 0xFFFFFFFF for x in xs]).generate_state(2, np.uint32)
               .astype(np.uint64).dot(np.array([1, 1 << 32], np.uint64)) & ((1 << 63) - 1))


class CacheSource:
    """The Phase 8 window cache through the data side's `ml.train.data`:
    `CacheDataset` (training: every cached window of the train split with
    spec/model.md §2's link-subset and unknown-geometry rules; evaluation:
    decision 5's windows, optionally a §5.6 link subset), `TrainSampler`
    (64 windows per train scene per epoch, keyed by seed and epoch) and
    `make_loader`. The rules' probabilities are the data module's; the run
    config records them and refuses different ones."""

    def __init__(self, cfg: dict):
        from ml.train import cache as cache_mod
        from ml.train import data as data_mod

        self.cfg = cfg
        self.data = data_mod
        rules = {"p_one_tx": data_mod.P_ONE_TX, "p_one_link": data_mod.P_ONE_LINK, "p_drop_link": data_mod.P_DROP,
                 "p_unknown_geometry": data_mod.P_UNKNOWN_GEOMETRY}
        if any(abs(cfg["subset_rules"][k] - v) > 1e-12 for k, v in rules.items()):
            raise ValueError(f"subset_rules {cfg['subset_rules']} differ from ml.train.data's {rules}")
        cache_dir = cfg["cache_dir"] or cache_mod.default_cache_dir(cfg["dataset"], cfg["window_s"])
        self.cache = cache_mod.open_cache(cache_dir)
        meta = self.cache.meta
        if abs(float(meta["length_s"]) - float(cfg["window_s"])) > 1e-9:
            raise ValueError(f"{cache_dir} holds {meta['length_s']} s windows, the run wants {cfg['window_s']} s")
        if os.path.basename(os.path.normpath(cfg["dataset"])) != meta["dataset"]:
            raise ValueError(f"{cache_dir} is dataset {meta['dataset']!r}, the run names {cfg['dataset']!r}")
        if cfg["phase"] and not meta.get("phase"):
            raise ValueError(f"{cache_dir} has no phase (build it with ml.train.cache --phase --out DIR)")
        if meta.get("preprocess", {}).get("bin_hz") != BIN_HZ:
            raise ValueError(f"{cache_dir} is binned at {meta.get('preprocess', {}).get('bin_hz')} Hz, not {BIN_HZ}")
        self.include = (("phase",) if cfg["phase"] else ()) + (("gain_db",) if cfg["gain"] else ())
        self.train_set = data_mod.CacheDataset(self.cache, "train", mode="train", include=self.include)
        self.sampler = data_mod.TrainSampler(self.train_set, ticks_per_scene=cfg["ticks_per_scene"], seed=cfg["seed"])
        self.train_dl = data_mod.make_loader(self.train_set, cfg["batch"], sampler=self.sampler,
                                             num_workers=cfg["workers"], drop_last=True)
        self._eval = {}

    def describe(self) -> dict:
        m = self.cache.meta
        return {"kind": "cache", "cache_dir": self.cache.dir, "dataset": m["dataset"], "impairments": m["impairments"],
                "length_s": m["length_s"], "T": m["T"], "bin_hz": m["bin_hz"], "phase_cache": bool(m.get("phase")),
                "include": list(self.include), "n_train_windows": len(self.train_set),
                "train_samples_per_epoch": len(self.sampler),
                "subset_rules": {"p_one_tx": self.data.P_ONE_TX, "p_one_link": self.data.P_ONE_LINK,
                                 "p_drop_link": self.data.P_DROP, "p_unknown_geometry": self.data.P_UNKNOWN_GEOMETRY},
                "eval_min_t": self.data.EVAL_MIN_T, "manifest_sha256": m.get("manifest_sha256")}

    def train_loader(self, epoch: int):
        self.sampler.set_epoch(epoch)
        return self.train_dl

    def eval_loader(self, split: str, subset: str | None = None, workers: int | None = None):
        key = (split, subset or "all")
        if key not in self._eval:
            ds = self.data.CacheDataset(self.cache, split, mode="eval", subset=subset or "all", include=self.include)
            nw = max(1, min(4, self.cfg["workers"])) if workers is None else workers
            self._eval[key] = self.data.make_loader(ds, self.cfg["eval_batch"], num_workers=nw)
        return self._eval[key]


def make_source(cfg: dict):
    if cfg.get("synthetic") is not None:
        return SyntheticSource(cfg)
    return CacheSource(cfg)


# --------------------------------------------------------------------------
# Schedule, evaluation, metrics summary
# --------------------------------------------------------------------------


def lr_lambda(warmup: int, total: int):
    """Linear warm-up over `warmup` steps, then cosine to 0 at `total`."""
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        if total <= warmup:
            return 1.0
        p = min(1.0, (step - warmup) / (total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return f


TERMS = ("presence", "count", "zone", "xy_nll", "exist")
WARM_STEPS = 30                      # steps excluded from the steady-state throughput (compilation, warm-up)


@torch.no_grad()
def run_eval(fwd, loader, device, *, amp: bool, weights: dict, matcher: str, n_links: int | None = None,
             n_batch: int | None = None, max_batches: int = 0, collect: bool = True, pack: bool = False,
             bucket: int = 256) -> dict:
    """The validation loss over every window of `loader` (each term's exact
    mean over the whole set: its numerator and denominator are summed across
    batches), and, with `collect`, the predictions (spec/model.md contract)
    with the targets and window ids."""
    sums = {t: 0.0 for t in TERMS}
    dens = {t: 0.0 for t in TERMS}
    preds, targets, ids = [], [], []
    n = 0
    for bi, batch in enumerate(loader):
        if max_batches and bi >= max_batches:
            break
        b = to_device(batch, device)
        nb = b["amp"].shape[0]
        fb = forward_batch(b, batch["link_mask"], n_links=n_links, n_batch=n_batch if n_batch and nb < n_batch else None,
                           pack=pack, bucket=bucket, compiled=getattr(fwd, "enabled", False))
        with _autocast(device, amp):
            out = fwd(fb)
        out = {k: v[:nb].float() for k, v in out.items()}
        parts = loss_parts(out, b, matcher)
        for t in TERMS:
            sums[t] += float(parts[t][0])
            dens[t] += float(parts[t][1])
        n += nb
        if collect:
            preds.append(mdl.predictions(out))
            targets.append({k: b[k].cpu().numpy() for k in ("presence", "count", "zone_occupied", "zone_reachable",
                                                           "xy", "person_mask", "walked", "geometry_known", "room")
                            if k in b})
            ids.append(window_ids(batch))
    means = {t: sums[t] / max(dens[t], 1.0) for t in TERMS}
    res = {"loss": sum(weights[t] * means[t] for t in TERMS), **means, "n_windows": n}
    if collect:
        res["predictions"] = _concat(preds)
        res["targets"] = _concat(targets)
        res["ids"] = _concat(ids)
    return res


def loss_parts(out: dict, batch: dict, matcher: str) -> dict:
    """Each loss term as (sum, count) over the batch, so a whole set's mean is exact."""
    w = {t: 1.0 for t in TERMS}
    res = mdl.compute_loss(out, batch, w, matcher=matcher)
    known = batch["geometry_known"].bool()
    nb = float(batch["amp"].shape[0])
    n_zone = float((batch["zone_reachable"].bool() & known[:, None]).sum())
    n_slots = float(known.sum()) * out["slot_logit"].shape[1]
    n_people = float(res["n_matched"])
    return {"presence": (float(res["presence"]) * nb, nb), "count": (float(res["count"]) * nb, nb),
            "zone": (float(res["zone"]) * n_zone, n_zone), "xy_nll": (float(res["xy_nll"]) * n_people, n_people),
            "exist": (float(res["exist"]) * n_slots, n_slots)}


def window_ids(batch: dict) -> dict:
    """(scene, k) per window, plus n_nodes, from the batch's `meta` (a list of
    dicts, or a dict of arrays)."""
    meta = batch.get("meta")
    if meta is None:
        n = batch["amp"].shape[0]
        return {"scene": np.full(n, -1, np.int64), "k": np.full(n, -1, np.int64)}
    if isinstance(meta, dict):
        get = lambda key: np.asarray(meta[key])  # noqa: E731
    else:
        get = lambda key: np.array([_meta_get(m, key) for m in meta])  # noqa: E731
    out = {"scene": get("scene").astype(np.int64), "k": get("k").astype(np.int64)}
    for key in ("n_nodes", "window"):
        with contextlib.suppress(KeyError, TypeError, ValueError):
            out[key] = get(key).astype(np.int64)
    return out


_META_ALIASES = {"scene": ("scene", "scene_index", "scene_idx"), "k": ("k", "tick", "label_tick")}


def _meta_get(m, key):
    if isinstance(m, dict):
        for alias in _META_ALIASES.get(key, (key,)):
            if alias in m:
                return m[alias]
        raise KeyError(key)
    if hasattr(m, key):
        return getattr(m, key)
    raise KeyError(key)


def _concat(parts: list) -> dict:
    if not parts:
        return {}
    return {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}


def quick_metrics(preds: dict, tg: dict) -> dict:
    """A small validation summary for the training log (spec/model.md §4's
    decision rules; the reported figures come from ml.train.metrics through
    evaluate.py). SIM."""
    out = {}
    pres = tg["presence"] > 0.5
    pp = preds["presence_prob"] >= 0.5
    out["presence_acc"] = float((pp == pres).mean())
    if pres.any() and (~pres).any():
        out["presence_bal_acc"] = float(0.5 * ((pp[pres]).mean() + (~pp[~pres]).mean()))
    out["count_acc"] = float((preds["count_prob"].argmax(-1) == tg["count"]).mean())
    known = tg["geometry_known"].astype(bool) if "geometry_known" in tg else np.ones(len(pres), bool)
    reach = tg["zone_reachable"].astype(bool) & known[:, None]
    zp = preds["zone_prob"] >= 0.5
    zt = tg["zone_occupied"] > 0.5
    f1s = []
    for c in range(9):
        r = reach[:, c]
        if not r.any():
            continue
        tp = float((zp[r, c] & zt[r, c]).sum())
        fp = float((zp[r, c] & ~zt[r, c]).sum())
        fn = float((~zp[r, c] & zt[r, c]).sum())
        if tp + fp + fn > 0:
            f1s.append(2 * tp / (2 * tp + fp + fn))
    if f1s:
        out["zone_f1"] = float(np.mean(f1s))
    # xy: slots with existence >= 0.5, Hungarian to present people, a pair beyond 2 m a miss.
    from scipy.optimize import linear_sum_assignment

    errs, inside, n_est, n_people = [], [], 0, 0
    for i in range(len(pres)):
        if not known[i]:
            continue
        est = np.flatnonzero(preds["slot_exist"][i] >= 0.5)
        ppl = np.flatnonzero(tg["person_mask"][i])
        n_est += est.size
        n_people += ppl.size
        if not est.size or not ppl.size:
            continue
        dmat = np.sqrt(((tg["xy"][i][ppl][:, None] - preds["slot_xy"][i][est][None]) ** 2).sum(-1))
        r, c = linear_sum_assignment(dmat)
        for a, b in zip(r, c):
            if dmat[a, b] <= 2.0:
                errs.append(dmat[a, b])
                inside.append(dmat[a, b] <= float(mdl.radius95(preds["slot_sigma"][i][est[b]])))
    if n_people:
        out["xy_recall"] = len(errs) / n_people
    if n_est:
        out["xy_precision"] = len(errs) / n_est
    if errs:
        out["xy_median_m"] = float(np.median(errs))
        out["xy_p90_m"] = float(np.percentile(errs, 90))
        out["radius_coverage"] = float(np.mean(inside))
    return out


#: The validation summary logged per epoch: ml.train.metrics figure -> log name (point values, no bootstrap).
HEADLINE = {
    "presence|all|accuracy": "presence_acc", "presence|all|balanced_accuracy": "presence_bal_acc",
    "presence|all|auc": "presence_auc", "presence|still|accuracy": "presence_acc_still",
    "count|all|accuracy": "count_acc", "count|all|mae": "count_mae",
    "zone|all|f1_macro": "zone_f1", "zone|all|exact_set_accuracy": "zone_exact",
    "xy2m|all|median_error": "xy_median_m", "xy2m|all|p90_error": "xy_p90_m", "xy2m|all|recall": "xy_recall",
    "xy2m|all|precision": "xy_precision", "xy2m|all|coverage": "radius_coverage",
    "xy2m|all|radius_median": "radius_median_m", "one|all|median_error": "one_person_median_m",
}


def contract(res: dict) -> dict:
    """The predictions contract (ml.train.metrics) of `run_eval`'s output."""
    from ml.train import metrics as met

    p, ids = res["predictions"], res["ids"]
    return met.predictions_from_slots(ids["scene"], ids["k"], p["presence_prob"], p["count_prob"], p["zone_prob"],
                                      p["slot_exist"], p["slot_xy"], p["slot_sigma"])


def truth_of(res: dict) -> dict:
    """ml.train.metrics' truth from the targets `run_eval` collected (the
    bootstrap stratum is the people count, constant per scene in SIM)."""
    from ml.train import metrics as met

    tg, ids = res["targets"], res["ids"]
    return met.make_truth(ids["scene"], ids["k"], tg["count"], tg["xy"], tg["person_mask"], tg["walked"],
                          tg["zone_occupied"] > 0.5, tg["zone_reachable"], tg["room"],
                          geometry_known=tg.get("geometry_known"))


def metrics_summary(res: dict, cfg: dict | None = None) -> dict:
    """The epoch's validation summary: HEADLINE's point figures from
    ml.train.metrics (no bootstrap; evaluate.py reports the intervals), or
    `quick_metrics` if that fails (the error is logged). SIM."""
    try:
        from ml.train import metrics as met

        out = met.evaluate({"model": contract(res)}, truth_of(res), n_boot=0)
        flat = met.flat_figures(out["methods"]["model"])
        return {"source": "ml.train.metrics", **{name: flat[key].get("value") for key, name in HEADLINE.items()
                                                 if key in flat}}
    except Exception as e:  # noqa: BLE001 - a summary must never stop training
        return {"source": "train.quick_metrics", "metrics_error": f"{type(e).__name__}: {e}"[:300],
                **quick_metrics(res["predictions"], res["targets"])}


# --------------------------------------------------------------------------
# The run
# --------------------------------------------------------------------------


def _autocast(device, amp: bool):
    dev = torch.device(device)
    if amp and dev.type == "cuda":
        return torch.autocast("cuda", dtype=torch.float16)
    return contextlib.nullcontext()


def default_run_id(cfg: dict) -> str:
    ds = "hwoff" if cfg["dataset"].endswith("hwoff") else "on"
    extra = ("_phase" if cfg["phase"] else "") + ("_gain" if cfg["gain"] else "")
    syn = "_synthetic" if cfg.get("synthetic") is not None else ""
    return f"{ds}_l{cfg['window_s']:g}_d{cfg['d']}_lr{cfg['lr']:g}{extra}_s{cfg['seed']}{syn}"


def resolve_config(cfg: dict | None = None, **overrides) -> dict:
    out = json.loads(json.dumps(DEFAULTS))
    for src in (cfg or {}), overrides:
        for k, v in src.items():
            if k not in DEFAULTS:
                raise ValueError(f"unknown config key {k!r}")
            if v is None and k not in ("run_id", "cache_dir", "synthetic"):
                continue
            if k in ("loss_weights", "subset_rules") and v is not None:
                unknown = set(v) - set(DEFAULTS[k])
                if unknown:
                    raise ValueError(f"unknown {k} {sorted(unknown)}")
                out[k] = dict(out[k], **v)
            else:
                out[k] = v
    if out["window_s"] * BIN_HZ != int(out["window_s"] * BIN_HZ):
        raise ValueError(f"window_s {out['window_s']} is not a multiple of {1 / BIN_HZ} s")
    if out["matcher"] not in mdl.MATCHERS:
        raise ValueError(f"matcher {out['matcher']!r} is not one of {sorted(mdl.MATCHERS)}")
    if not out["run_id"]:
        out["run_id"] = default_run_id(out)
    return out


def model_config(cfg: dict) -> mdl.ModelConfig:
    return mdl.ModelConfig(d=cfg["d"], heads=cfg["heads"], layers=cfg["layers"], dropout=cfg["dropout"],
                           phase=cfg["phase"], gain=cfg["gain"])


def _rng_state() -> dict:
    st = {"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}
    if torch.cuda.is_available():
        st["cuda"] = torch.cuda.get_rng_state_all()
    return st


def _set_rng_state(st: dict) -> None:
    torch.set_rng_state(st["torch"])
    np.random.set_state(st["numpy"])
    random.setstate(st["python"])
    if "cuda" in st and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(st["cuda"])


_RESUME_IGNORED = {"overwrite", "workers", "device", "out_root", "compile", "compile_mode", "eval_batch",
                   "interrupt_after_epochs"}


def _same_run(a: dict, b: dict) -> bool:
    return {k: v for k, v in a.items() if k not in _RESUME_IGNORED} == \
        {k: v for k, v in b.items() if k not in _RESUME_IGNORED}


def _save(obj, path: str) -> None:
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


def _write_json(obj, path: str) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=1, default=_json_default)
        f.write("\n")
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"{type(o).__name__} is not JSON serialisable")


def _environment(device) -> dict:
    env = {"host": socket.gethostname(), "python": platform.python_version(), "torch": torch.__version__,
           "numpy": np.__version__, "pid": os.getpid()}
    dev = torch.device(device)
    if dev.type == "cuda" and torch.cuda.is_available():
        env["gpu"] = torch.cuda.get_device_name(dev)
    return env


class Compiled:
    """`torch.compile(model)` with an eager fallback: the first call that
    raises inside compilation switches this run to eager for good (logged)."""

    def __init__(self, model, enabled: bool, mode: str = "default"):
        self.model = model
        self.fn = model
        self.error = None
        self.enabled = enabled
        if enabled:
            try:
                self.fn = torch.compile(model, mode=None if mode == "default" else mode)
            except Exception as e:  # noqa: BLE001
                self.fail(e)

    def fail(self, e: Exception) -> None:
        self.error = f"{type(e).__name__}: {e}"[:2000]
        self.fn = self.model
        self.enabled = False
        warnings.warn(f"torch.compile failed, running eager: {self.error}")

    def __call__(self, batch):
        if not self.enabled:
            return self.model(batch)
        try:
            return self.fn(batch)
        except Exception as e:  # noqa: BLE001
            if not _is_compile_error(e):
                raise
            self.fail(e)
            return self.model(batch)


def _is_compile_error(e: Exception) -> bool:
    mod = type(e).__module__ or ""
    return mod.startswith("torch._dynamo") or mod.startswith("torch._inductor") or "inductor" in str(e).lower() \
        or "triton" in str(e).lower()


def train(cfg: dict | None = None, **overrides) -> dict:
    """Run (or resume) one training run; returns the `done.json` summary.
    Holds an exclusive lock on the run directory (`.lock`) while it runs: a
    second process on the same run is refused."""
    cfg = resolve_config(cfg, **overrides)
    run_dir = os.path.join(cfg["out_root"], cfg["run_id"])
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, ".lock"), "w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"{run_dir} is being trained by another process") from None
        lock.write(f"{os.getpid()}\n")
        lock.flush()
        return _train_locked(cfg, run_dir)


def _train_locked(cfg: dict, run_dir: str) -> dict:
    done_path = os.path.join(run_dir, "done.json")
    last_path = os.path.join(run_dir, "last.pt")
    log_path = os.path.join(run_dir, "log.jsonl")
    os.makedirs(run_dir, exist_ok=True)
    if cfg["overwrite"]:
        for name in ("done.json", "last.pt", "best.pt", "log.jsonl", "config.json"):
            with contextlib.suppress(FileNotFoundError):
                os.remove(os.path.join(run_dir, name))
    if os.path.exists(done_path):
        with open(done_path) as f:
            return json.load(f)

    device = torch.device(cfg["device"] if (cfg["device"] != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    # GradScaler skips the optimiser step on an overflow; the scheduler then warns about the order. Expected.
    warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")
    if cfg["deterministic"]:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True

    torch.manual_seed(_mix(cfg["seed"], 1))
    np.random.seed(_mix(cfg["seed"], 2) % (1 << 32))
    random.seed(_mix(cfg["seed"], 3))
    mcfg = model_config(cfg)
    model = mdl.build_model(mcfg, seed=cfg["seed"]).to(device)
    n_params = mdl.count_parameters(model)
    source = make_source(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = len(source.train_loader(0))
    if cfg["max_epoch_steps"]:
        steps_per_epoch = min(steps_per_epoch, cfg["max_epoch_steps"])
    total_steps = steps_per_epoch * cfg["epochs"]
    if cfg["max_steps"]:
        total_steps = min(total_steps, cfg["max_steps"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(cfg["warmup_steps"], total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, step, best, bad, elapsed0 = 0, 0, float("inf"), 0, 0.0
    best_epoch = -1
    if os.path.exists(last_path):
        st = torch.load(last_path, map_location="cpu", weights_only=False)
        if not _same_run(st["run_config"], cfg):
            raise SystemExit(f"{run_dir}: last.pt was written by another configuration; use --overwrite")
        model.load_state_dict(st["model"])
        opt.load_state_dict(st["optimizer"])
        sched.load_state_dict(st["scheduler"])
        scaler.load_state_dict(st["scaler"])
        _set_rng_state(st["rng"])
        start_epoch, step, best, bad = st["epoch"] + 1, st["step"], st["best_val"], st["bad_epochs"]
        best_epoch, elapsed0 = st["best_epoch"], st["elapsed_s"]
        _truncate_log(log_path, start_epoch)
    else:
        with contextlib.suppress(FileNotFoundError):
            os.remove(log_path)
    info = {"label": LABEL, "spec": "spec/model.md", **cfg, "model_config": asdict(mcfg), "n_params": n_params,
            "steps_per_epoch": steps_per_epoch, "total_steps": total_steps, "data": source.describe(),
            "amp_dtype": "float16" if use_amp else "float32", "environment": _environment(device)}
    _write_json(info, os.path.join(run_dir, "config.json"))

    fwd = Compiled(model, cfg["compile"] and device.type == "cuda", cfg["compile_mode"])
    pad_l = cfg["pad_links"] or None
    t_start = time.time() - elapsed0
    stop_reason = "epochs"
    for epoch in range(start_epoch, cfg["epochs"]):
        if cfg["max_steps"] and step >= cfg["max_steps"]:
            stop_reason = "max_steps"
            break
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sums = {t: torch.zeros((), device=device) for t in ("total",) + TERMS}
        n_steps, n_samples = 0, 0
        n_skipped = torch.zeros((), device=device)
        t0 = time.time()
        t_warm, n_warm = None, 0
        loader = source.train_loader(epoch)
        for batch in loader:
            if (cfg["max_epoch_steps"] and n_steps >= cfg["max_epoch_steps"]) or \
                    (cfg["max_steps"] and step >= cfg["max_steps"]):
                break
            b = to_device(batch, device)
            fb = forward_batch(b, batch["link_mask"], n_links=pad_l if fwd.enabled else None, n_batch=None,
                               pack=cfg["pack_links"], bucket=cfg["pack_bucket"], compiled=fwd.enabled)

            def train_step():
                opt.zero_grad(set_to_none=True)
                with _autocast(device, use_amp):
                    out = fwd(fb)
                losses = mdl.compute_loss(out, b, cfg["loss_weights"], matcher=cfg["matcher"])
                scaler.scale(losses["total"]).backward()
                return losses

            try:
                losses = train_step()
            except Exception as e:  # noqa: BLE001 - a compile failure in the backward pass surfaces here
                if not (fwd.enabled and _is_compile_error(e)):
                    raise
                fwd.fail(e)
                losses = train_step()
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            n_skipped += (~torch.isfinite(gnorm)).float()        # the scaler skips exactly these steps
            if use_amp:
                scaler.step(opt)                                  # no-op on a non-finite step
                scaler.update()
            elif bool(torch.isfinite(gnorm)):                     # fp32: skip a non-finite step ourselves
                opt.step()
            sched.step()
            for t in sums:
                sums[t] += losses[t].detach()
            n_steps += 1
            step += 1
            n_samples += int(b["amp"].shape[0])
            if n_steps == WARM_STEPS:                             # steady state: after compilation and warm-up
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                t_warm, n_warm = time.time(), n_samples
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_end = time.time()
        train_s = t_end - t0
        train_means = {t: float(v) / max(n_steps, 1) for t, v in sums.items()}

        model.eval()
        t1 = time.time()
        val = run_eval(fwd, source.eval_loader("val"), device, amp=use_amp, weights=cfg["loss_weights"],
                       matcher=cfg["matcher"], n_links=pad_l if fwd.enabled else None,
                       n_batch=cfg["eval_batch"] if (pad_l and fwd.enabled) else None,
                       max_batches=cfg["val_max_batches"], pack=cfg["pack_links"], bucket=cfg["pack_bucket"])
        summary = metrics_summary(val)
        val_s = time.time() - t1
        improved = val["loss"] < best
        if improved:
            best, bad, best_epoch = val["loss"], 0, epoch
            _save({"label": LABEL, "model": model.state_dict(), "model_config": asdict(mcfg), "run_config": cfg,
                   "epoch": epoch, "step": step, "val_loss": val["loss"]}, os.path.join(run_dir, "best.pt"))
        else:
            bad += 1
        rec = {
            "label": LABEL, "epoch": epoch, "step": step, "lr": sched.get_last_lr()[0],
            "train": train_means, "skipped_steps": int(n_skipped),
            "grad_scale": float(scaler.get_scale()) if use_amp else None,
            "val": {k: v for k, v in val.items() if k not in ("predictions", "targets", "ids")},
            "val_metrics": summary, "best": improved, "best_val": best, "bad_epochs": bad,
            "time": {"train_s": train_s, "val_s": val_s, "elapsed_s": time.time() - t_start},
            "throughput": {"steps_per_s": n_steps / max(train_s, 1e-9), "samples_per_s": n_samples / max(train_s, 1e-9),
                           "steady_samples_per_s": ((n_samples - n_warm) / max(t_end - t_warm, 1e-9)
                                                    if t_warm is not None and n_samples > n_warm else None)},
            "compiled": fwd.enabled, "compile_error": fwd.error,
        }
        if device.type == "cuda":
            rec["gpu"] = {"max_allocated_mb": torch.cuda.max_memory_allocated(device) / 2 ** 20,
                          "max_reserved_mb": torch.cuda.max_memory_reserved(device) / 2 ** 20}
        with open(log_path, "a") as f:
            f.write(json.dumps(rec, default=_json_default) + "\n")
        _save({"label": LABEL, "model": model.state_dict(), "optimizer": opt.state_dict(),
               "scheduler": sched.state_dict(), "scaler": scaler.state_dict(), "rng": _rng_state(),
               "run_config": cfg, "model_config": asdict(mcfg), "epoch": epoch, "step": step, "best_val": best,
               "bad_epochs": bad, "best_epoch": best_epoch, "elapsed_s": time.time() - t_start}, last_path)
        print(f"[{cfg['run_id']}] epoch {epoch} step {step} train {train_means['total']:.4f} "
              f"val {val['loss']:.4f}{' *' if improved else ''} ({train_s:.0f}+{val_s:.0f} s, "
              f"{rec['throughput']['samples_per_s']:.0f} samples/s) SIM", flush=True)
        if bad >= cfg["patience"]:
            stop_reason = "early_stopping"
            break
        if cfg["interrupt_after_epochs"] and epoch + 1 - start_epoch >= cfg["interrupt_after_epochs"]:
            return {"label": LABEL, "run_id": cfg["run_id"], "interrupted_after_epoch": epoch}
    else:
        stop_reason = "epochs"
    if cfg["max_steps"] and step >= cfg["max_steps"]:
        stop_reason = "max_steps"
    done = {"label": LABEL, "run_id": cfg["run_id"], "best_epoch": best_epoch, "best_val_loss": best,
            "steps": step, "stop": stop_reason, "elapsed_s": time.time() - t_start, "n_params": n_params,
            "compiled": fwd.enabled, "compile_error": fwd.error}
    _write_json(done, done_path)
    return done


def _truncate_log(path: str, n_epochs: int) -> None:
    """Keep the log's first `n_epochs` epoch lines (a resumed run rewrites the rest)."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        lines = [ln for ln in f if ln.strip()]
    keep = [ln for ln in lines if json.loads(ln).get("epoch", -1) < n_epochs]
    with open(path, "w") as f:
        f.writelines(keep)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def _bool(s: str) -> bool:
    if s.lower() in ("1", "true", "yes", "on"):
        return True
    if s.lower() in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"{s!r} is not a boolean")


def parse_args(argv=None) -> dict:
    ap = argparse.ArgumentParser(description="Train model v1 (spec/model.md). Everything it produces is SIM.")
    ap.add_argument("--config", help="a JSON file of DEFAULTS keys; command-line flags override it")
    for key, val in DEFAULTS.items():
        if key in ("loss_weights", "subset_rules", "synthetic"):
            continue
        flag = "--" + key.replace("_", "-")
        if isinstance(val, bool):
            ap.add_argument(flag, type=_bool, default=None, metavar="BOOL")
        elif isinstance(val, int) and not isinstance(val, bool):
            ap.add_argument(flag, type=int, default=None)
        elif isinstance(val, float):
            ap.add_argument(flag, type=float, default=None)
        else:
            ap.add_argument(flag, default=None)
    ap.add_argument("--loss-weights", type=json.loads, default=None, help="JSON, e.g. '{\"xy_nll\": 0.5}'")
    ap.add_argument("--subset-rules", type=json.loads, default=None, help="JSON")
    ap.add_argument("--synthetic", nargs="?", const="{}", default=None,
                    help="train on synthetic data (optional JSON of sizes: n_train, n_val, n_test, n_nodes)")
    args = vars(ap.parse_args(argv))
    cfg = {}
    config_path = args.pop("config")
    if config_path:
        with open(config_path) as f:
            cfg = json.load(f)
    syn = args.pop("synthetic")
    if syn is not None:
        cfg["synthetic"] = json.loads(syn)
    for k, v in args.items():
        if v is not None:
            cfg[k] = v
    return cfg


def main(argv=None) -> int:
    cfg = parse_args(argv)
    done = train(cfg)
    print(json.dumps(done, indent=1, default=_json_default))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
