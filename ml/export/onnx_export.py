"""ONNX export of a trained model v1 and its check (Phase 12).

ADR-009 decision 4: an export is a research artefact. Loading it in the
gateway is a later ADR; `docs/onnx-ort.md` notes how that would read it.

The exported graph takes the §1 batch as ten explicit inputs (float32
where the batch has float16: the model reads every input as float32 anyway)
and returns six outputs; the batch `B` and link `L` axes are dynamic, `T`
(= 5 ℓ) and the reference-bin count are fixed by the checkpoint:

| input | shape | dtype |
|---|---|---|
| `amp` | [B, L, R, T] | float32 |
| `mu`, `sigma` | [B, L, R] | float32 |
| `sample_mask` | [B, L, T] | bool |
| `link_mask` | [B, L] | bool |
| `rssi` | [B, L, T] | float32 |
| `rssi_mask` | [B, L, T] | bool |
| `link_pos` | [B, L, 2, 3] | float32 |
| `room` | [B, 3] | float32 |
| `geometry_known` | [B] | bool |

Outputs: `presence_logit` [B], `count_logits` [B, 4], `zone_logits` [B, 9],
`slot_logit` [B, 3], `slot_xy` [B, 3, 2] (m), `slot_sigma` [B, 3] (m).

    .venv/bin/python -m ml.export.onnx_export CHECKPOINT.pt --out model.onnx --window-s 2 [--check-cache DIR]

`check` runs the export with onnxruntime (CPU) against the PyTorch model in
float32 on the same batches and reports the largest absolute difference
per output; the brief's bar is 1e-4.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from ml.train import model as mdl

__all__ = ["INPUTS", "OUTPUTS", "Exportable", "export", "check", "random_batch"]

INPUTS = ("amp", "mu", "sigma", "sample_mask", "link_mask", "rssi", "rssi_mask", "link_pos", "room", "geometry_known")
OUTPUTS = ("presence_logit", "count_logits", "zone_logits", "slot_logit", "slot_xy", "slot_sigma")
_BOOL = {"sample_mask", "link_mask", "rssi_mask", "geometry_known"}
TOLERANCE = 1e-4


class Exportable(torch.nn.Module):
    """The model with explicit tensor inputs and a tuple of outputs (what ONNX export traces)."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, amp, mu, sigma, sample_mask, link_mask, rssi, rssi_mask, link_pos, room, geometry_known):
        out = self.model({"amp": amp, "mu": mu, "sigma": sigma, "sample_mask": sample_mask, "link_mask": link_mask,
                          "rssi": rssi, "rssi_mask": rssi_mask, "link_pos": link_pos, "room": room,
                          "geometry_known": geometry_known})
        return tuple(out[k] for k in OUTPUTS)


def random_batch(n_ref: int, t: int, b: int = 4, n_links: int = 12, seed: int = 0) -> dict:
    """A batch with every case the graph must handle: padded links, masked bins, unknown geometry."""
    g = torch.Generator().manual_seed(seed)
    link_mask = torch.ones(b, n_links, dtype=torch.bool)
    link_mask[0, n_links // 2:] = False
    sample_mask = torch.rand(b, n_links, t, generator=g) > 0.1
    sample_mask &= link_mask[..., None]
    room = torch.tensor([[4.0, 3.5, 2.6]]).repeat(b, 1)
    return {
        "amp": torch.randn(b, n_links, n_ref, t, generator=g) * sample_mask[:, :, None, :],
        "mu": (1.0 + 0.2 * torch.randn(b, n_links, n_ref, generator=g)) * link_mask[..., None],
        "sigma": (0.05 * torch.rand(b, n_links, n_ref, generator=g)) * link_mask[..., None],
        "sample_mask": sample_mask, "link_mask": link_mask,
        "rssi": (-50 + 3 * torch.randn(b, n_links, t, generator=g)) * sample_mask,
        "rssi_mask": sample_mask.clone(),
        "link_pos": torch.rand(b, n_links, 2, 3, generator=g) * room[:, None, None, :] * link_mask[..., None, None],
        "room": room, "geometry_known": torch.tensor([True, False] * (b // 2) + [True] * (b % 2)),
    }


def _args(batch: dict) -> tuple:
    return tuple(batch[k].bool() if k in _BOOL else batch[k].float() for k in INPUTS)


def load_checkpoint(path: str) -> tuple:
    st = torch.load(path, map_location="cpu", weights_only=False)
    model = mdl.build_model(st["model_config"])
    model.load_state_dict(st["model"])
    model.eval()
    return model, st


def export(model: torch.nn.Module, out_path: str, t: int, *, opset: int = 17) -> dict:
    """Export `model` (eval mode, CPU, float32) for windows of `t` bins."""
    model = model.float().cpu().eval()
    wrapped = Exportable(model).eval()
    example = _args(random_batch(model.cfg.n_ref, t))
    dyn = {}
    for k in INPUTS:
        dyn[k] = {0: "B"} if k in ("room", "geometry_known") else {0: "B", 1: "L"}
    for k in OUTPUTS:
        dyn[k] = {0: "B"}
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    t0 = time.time()
    with torch.no_grad():
        torch.onnx.export(wrapped, example, out_path, input_names=list(INPUTS), output_names=list(OUTPUTS),
                          dynamic_axes=dyn, opset_version=opset, do_constant_folding=True, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(out_path))
    return {"path": out_path, "opset": opset, "t": t, "n_ref": model.cfg.n_ref, "bytes": os.path.getsize(out_path),
            "export_s": round(time.time() - t0, 2)}


def check(model: torch.nn.Module, onnx_path: str, batches: list, *, tolerance: float = TOLERANCE) -> dict:
    """The largest absolute difference per output between onnxruntime (CPU)
    and PyTorch (CPU, float32) over `batches`; `ok` when all are below `tolerance`."""
    import onnxruntime as ort
    model = model.float().cpu().eval()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    worst = {k: 0.0 for k in OUTPUTS}
    n = 0
    with torch.no_grad():
        for batch in batches:
            args = _args(batch)
            ref = Exportable(model)(*args)
            got = sess.run(list(OUTPUTS), {k: a.numpy() for k, a in zip(INPUTS, args)})
            for k, r, g in zip(OUTPUTS, ref, got):
                if r.shape != g.shape:
                    raise AssertionError(f"{k}: torch {tuple(r.shape)} vs onnx {g.shape}")
                worst[k] = max(worst[k], float(np.abs(r.numpy().astype(np.float64) - g.astype(np.float64)).max()))
            n += int(args[0].shape[0])
    return {"max_abs_diff": worst, "max": max(worst.values()), "tolerance": tolerance,
            "ok": all(v < tolerance for v in worst.values()), "windows": n}


def cache_batches(cache_dir: str, n_batches: int = 8, batch: int = 64, split: str = "test", seed: int = 0) -> list:
    """Real SIM batches from a Phase 8 cache (every link, geometry known) for `check`."""
    from ml.train import cache as cache_mod
    from ml.train import data as d8
    c = cache_mod.open_cache(cache_dir)
    ds = d8.CacheDataset(c, split, mode="eval")
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_batches):
        idx = rng.choice(len(ds), size=batch, replace=False)
        out.append(d8.collate([ds[int(i)] for i in idx]))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Export model v1 to ONNX and check it against PyTorch (Phase 12).")
    ap.add_argument("checkpoint")
    ap.add_argument("--out", required=True)
    ap.add_argument("--window-s", type=float, required=True)
    ap.add_argument("--opset", type=int, default=17)
    ap.add_argument("--check-cache", default=None, help="a Phase 8 cache dir for real check batches")
    args = ap.parse_args(argv)
    model, st = load_checkpoint(args.checkpoint)
    t = int(round(5 * args.window_s))
    info = export(model, args.out, t, opset=args.opset)
    batches = [random_batch(model.cfg.n_ref, t, b=8, n_links=n, seed=s) for s, n in enumerate((1, 5, 20, 56, 132))]
    if args.check_cache:
        batches += cache_batches(args.check_cache)
    res = check(model, args.out, batches)
    info.update({"check": res, "checkpoint": os.path.abspath(args.checkpoint), "label": st.get("label")})
    print(json.dumps(info, indent=1))
    return 0 if res["ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
