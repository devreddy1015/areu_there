"""Phase 9 fine-tuning and real-only training (spec/phase9.md §4.3–§4.4),
and prediction on REAL windows.

The recipe is fixed (spec/phase9.md §4.3, chosen before any REAL test
figure existed): AdamW (weight decay 0.01), `STEPS` = 600 steps of batch 64
(50 warm-up, cosine), fp16 autocast with a gradient scaler, gradient
clipping 1.0, Phase 8's link-subset rule, geometry always unknown, loss =
presence BCE + count CE (zone, xy and existence weights 0: no labels). No
early stopping: no REAL validation room is independent of the test room.
Fine-tuning starts from a P9-SIM checkpoint (lr 1e-4); real-only starts
from a random initialisation (seed `s`) at Phase 8's chosen lr. Eager mode
(the runs are short; compiling would cost more than it saves).

Every model this trains has seen REAL data: a fine-tuned model's results
are SIM->REAL (its pre-training was SIM), a real-only model's REAL.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch

from ml.phase9 import real_data as rdm
from ml.train import data as d8
from ml.train import model as mdl

__all__ = ["STEPS", "WARMUP", "BATCH", "LR_FINETUNE", "WEIGHT_DECAY", "GRAD_CLIP", "LOSS_WEIGHTS",
           "load_pretrained", "fresh_model", "train_real", "predict"]

STEPS = 600
WARMUP = 50
BATCH = 64
LR_FINETUNE = 1e-4
WEIGHT_DECAY = 0.01
GRAD_CLIP = 1.0
LOSS_WEIGHTS = {"presence": 1.0, "count": 1.0, "zone": 0.0, "xy_nll": 0.0, "exist": 0.0}


def load_pretrained(path: str, device="cpu") -> tuple:
    """A P9-SIM checkpoint (`best.pt` of `ml.train.train`): `(model, model_config, run_config)`.
    Refuses a checkpoint not labelled SIM."""
    st = torch.load(path, map_location="cpu", weights_only=False)
    if st.get("label") != "SIM":
        raise ValueError(f"{path}: checkpoint labelled {st.get('label')!r}, not SIM")
    model = mdl.build_model(st["model_config"])
    model.load_state_dict(st["model"])
    return model.to(device), st["model_config"], st.get("run_config", {})


def fresh_model(model_config: dict, seed: int, device="cpu") -> torch.nn.Module:
    """The same architecture from a random initialisation fixed by `seed`."""
    return mdl.build_model(model_config, seed=seed).to(device)


def _lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        p = min(1.0, (step - warmup) / max(1, total - warmup))
        return 0.5 * (1.0 + math.cos(math.pi * p))
    return f


def train_real(model: torch.nn.Module, caches: list, items, *, lr: float, seed: int, steps: int = STEPS,
               batch: int = BATCH, device="cuda", workers: int = 4, log_every: int = 100, log=None) -> dict:
    """Train `model` in place on REAL windows `items` ((cache index, window)
    pairs) for `steps` steps; returns a small log (losses, time, GPU memory)."""
    device = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = device.type == "cuda"
    torch.manual_seed(int(seed))
    ds = rdm.RealDataset(caches, items, augment=True)
    sampler = rdm.FixedStepSampler(len(ds), steps, batch, seed)
    loader = d8.make_loader(ds, batch, sampler=sampler, num_workers=workers, prefetch_factor=4, drop_last=False,
                            persistent_workers=False)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, _lr_lambda(WARMUP, steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.time()
    hist, skipped, step = [], 0, 0
    run = {"presence": 0.0, "count": 0.0, "n": 0}
    for b in loader:
        b = d8.to_device(b, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(b)
        out = {k: v.float() for k, v in out.items()}
        losses = mdl.compute_loss(out, b, LOSS_WEIGHTS)
        opt.zero_grad(set_to_none=True)
        scaler.scale(losses["total"]).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        scale = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if use_amp and scaler.get_scale() < scale:
            skipped += 1
        sched.step()
        step += 1
        run["presence"] += float(losses["presence"])
        run["count"] += float(losses["count"])
        run["n"] += 1
        if step % log_every == 0 or step == steps:
            hist.append({"step": step, "presence": run["presence"] / run["n"], "count": run["count"] / run["n"]})
            if log:
                log(f"    step {step}: presence {hist[-1]['presence']:.4f} count {hist[-1]['count']:.4f}")
            run = {"presence": 0.0, "count": 0.0, "n": 0}
    model.eval()
    mem = torch.cuda.max_memory_allocated(device) / 2**20 if device.type == "cuda" else 0.0
    return {"steps": step, "skipped_steps": skipped, "seconds": round(time.time() - t0, 2), "gpu_mb": round(mem, 1),
            "history": hist, "n_train": len(ds), "lr": lr, "seed": seed}


@torch.no_grad()
def predict(model: torch.nn.Module, cache: dict, windows, *, batch: int = 256, device="cuda") -> dict:
    """The model on REAL windows of one cache (every used link, geometry
    unknown): `presence_prob` [N], `count_prob` [N, 4]."""
    device = torch.device(device if (str(device) != "cuda" or torch.cuda.is_available()) else "cpu")
    use_amp = device.type == "cuda"
    model.eval()
    ds = rdm.RealDataset([cache], [(0, int(w)) for w in windows], augment=False)
    pres, cnt = [], []
    for i in range(0, len(ds), batch):
        b = d8.collate([ds[j] for j in range(i, min(len(ds), i + batch))])
        b = d8.to_device(b, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            out = model(b)
        p = mdl.predictions({k: v.float() for k, v in out.items()})
        pres.append(p["presence_prob"])
        cnt.append(p["count_prob"])
    return {"presence_prob": np.concatenate(pres).astype(np.float64), "count_prob": np.concatenate(cnt).astype(np.float64)}
