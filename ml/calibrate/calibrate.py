"""Differentiable calibration of Room Sim's global parameters (spec/calibration.md).

1. **Design.** A Latin-hypercube sample of the unit cube over `simulate.PARAMS`
   (plus the defaults), each run on the same fit rooms (common random numbers:
   the stored configs keep their seeds), giving every feature group's
   quantiles at `J` levels.
2. **Emulator.** A small torch MLP from the unit parameter vector to those
   quantiles. It is accurate where the design is dense, and differentiable.
3. **Fit**, per Phase 9 fold, on the fold's training sessions only (the test
   room never calibrates anything):
   - target: the REAL quantiles;
   - objective: the sum over groups of the 1-D Wasserstein-1 distance,
     approximated from the quantiles as `mean_j |q_sim(τ_j) - q_real(τ_j)|`,
     each group divided by its REAL interquartile range so the groups weigh
     alike;
   - Adam on `u = sigmoid(z)` from many starts.
4. **Check.** The fitted parameters and the defaults are simulated on
   *other* rooms. The exact `scipy.stats.wasserstein_distance` to the
   fold's training data and to its unseen test room is reported for both.

Labels: the parameters are SIM model parameters; every distance between a
SIM and a REAL distribution is SIM->REAL.

    .venv/bin/python -m ml.calibrate.calibrate --design 128 --fit-rooms 24 --check-rooms 24
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch

from ml.calibrate import features as cf
from ml.calibrate import real as cr
from ml.calibrate import simulate as cs

__all__ = ["J", "TAUS", "FOLDS", "quantiles", "lhs", "Emulator", "fit_emulator", "fit", "run"]

J = 20
TAUS = (np.arange(J) + 0.5) / J
#: Phase 9's folds (spec/phase9.md §4), HT20 sessions only.
FOLDS = {"A": {"train": ["envB_S3prime", "envC_C3"], "test": "envA_S3"},
         "B": {"train": ["envA_S3", "envC_C3"], "test": "envB_S3prime"},
         "C": {"train": ["envA_S3", "envB_S3prime"], "test": "envC_C3"}}
OUT = "/home/Aman/are_u_theone/runs/phase10/calibration"
RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "phase10")


def quantiles(feats: dict) -> np.ndarray:
    """[G, J] quantiles of every group at TAUS (NaN for an empty group)."""
    out = np.full((len(cf.GROUPS), J), np.nan)
    for g, name in enumerate(cf.GROUPS):
        v = feats[name]
        v = v[np.isfinite(v)]
        if v.size:
            out[g] = np.quantile(v, TAUS)
    return out


def lhs(n: int, d: int, seed: int) -> np.ndarray:
    """A Latin-hypercube sample of [0, 1]^d (each axis: one point per stratum)."""
    rng = np.random.default_rng(seed)
    u = (rng.random((n, d)) + np.arange(n)[:, None]) / n
    for k in range(d):
        u[:, k] = u[rng.permutation(n), k]
    return u


class Emulator(torch.nn.Module):
    def __init__(self, d_in: int, d_out: int, width: int = 256):
        super().__init__()
        self.net = torch.nn.Sequential(torch.nn.Linear(d_in, width), torch.nn.SiLU(), torch.nn.Linear(width, width),
                                       torch.nn.SiLU(), torch.nn.Linear(width, d_out))
        self.register_buffer("mean", torch.zeros(d_out))
        self.register_buffer("std", torch.ones(d_out))

    def forward(self, u):
        return self.net(u) * self.std + self.mean


def fit_emulator(u: np.ndarray, q: np.ndarray, *, steps: int = 4000, seed: int = 0, holdout: int = 0) -> tuple:
    """Fit on `(u [N, P], q [N, G*J])`; with `holdout` > 0 also report the
    held-out samples' error. Returns `(emulator, report)`."""
    torch.manual_seed(seed)
    n = u.shape[0]
    idx = np.random.default_rng(seed).permutation(n)
    te, tr = idx[:holdout], idx[holdout:]
    x = torch.tensor(u, dtype=torch.float32)
    y = torch.tensor(q, dtype=torch.float32)
    em = Emulator(u.shape[1], q.shape[1])
    em.mean.copy_(y[tr].mean(0))
    em.std.copy_(y[tr].std(0).clamp(min=1e-6))
    opt = torch.optim.Adam(em.parameters(), lr=3e-3, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        loss = (((em(x[tr]) - y[tr]) / em.std) ** 2).mean()
        loss.backward()
        opt.step()
        sched.step()
    rep = {"train_rmse_std": float(loss.detach().sqrt())}
    if holdout:
        with torch.no_grad():
            err = ((em(x[te]) - y[te]) / em.std).numpy()
            var = ((y[te] - y[tr].mean(0)) / em.std).numpy()
        rep["holdout_rmse_std"] = float(np.sqrt((err ** 2).mean()))
        rep["holdout_r2"] = float(1.0 - (err ** 2).sum() / max((var ** 2).sum(), 1e-12))
        g = len(cf.GROUPS)
        per = err.reshape(err.shape[0], g, J)
        pv = var.reshape(var.shape[0], g, J)
        rep["holdout_r2_per_group"] = {name: float(1.0 - (per[:, i] ** 2).sum() / max((pv[:, i] ** 2).sum(), 1e-12))
                                       for i, name in enumerate(cf.GROUPS)}
    return em, rep


def _objective(qhat: torch.Tensor, q_real: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    """Sum over groups of mean_j |q_sim - q_real| / scale_g (W1 from quantiles, per group scaled)."""
    g = len(cf.GROUPS)
    d = (qhat.reshape(-1, g, J) - q_real.reshape(1, g, J)).abs().mean(-1) / scale
    return d.sum(-1)


def fit(em: Emulator, q_real: np.ndarray, scale: np.ndarray, *, starts: int = 256, steps: int = 1500,
        seed: int = 0) -> dict:
    """Minimise the emulated objective over u in [0, 1]^P from `starts` random starts."""
    torch.manual_seed(seed)
    p = em.net[0].in_features
    z = torch.randn(starts, p, requires_grad=True)
    qr = torch.tensor(q_real.reshape(-1), dtype=torch.float32)
    sc = torch.tensor(scale, dtype=torch.float32)
    opt = torch.optim.Adam([z], lr=0.05)
    for _ in range(steps):
        opt.zero_grad()
        obj = _objective(em(torch.sigmoid(z)), qr, sc)
        obj.sum().backward()
        opt.step()
    with torch.no_grad():
        obj = _objective(em(torch.sigmoid(z)), qr, sc)
        best = int(torch.argmin(obj))
        u = torch.sigmoid(z[best]).numpy().astype(np.float64)
        top = torch.argsort(obj)[:10]
        spread = torch.sigmoid(z[top]).std(0).numpy()
    return {"u": u, "objective": float(obj[best]), "top10_u_std": spread.tolist()}


def w1(sim: dict, real: dict) -> dict:
    """Exact 1-D Wasserstein-1 per group between SIM and REAL samples, and scaled by REAL's IQR."""
    from scipy.stats import wasserstein_distance
    out = {}
    for g in cf.GROUPS:
        a, b = sim[g][np.isfinite(sim[g])], real[g][np.isfinite(real[g])]
        if a.size and b.size:
            iqr = float(np.subtract(*np.percentile(b, [75, 25]))) or 1.0
            d = float(wasserstein_distance(a, b))
            out[g] = {"w1": d, "w1_over_iqr": d / iqr}
    out["sum_over_iqr"] = float(sum(v["w1_over_iqr"] for v in out.values()))
    return out


def run(design: int = 128, fit_rooms: int = 24, check_rooms: int = 24, *, workers: int = 48, seed: int = 0,
        log=print) -> dict:
    torch.set_num_threads(4)                        # tiny tensors: more threads only spin
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    real = {t: cr.session_features(t) for t in cr.HT20}
    log(f"[cal REAL] features of {len(real)} sessions in {time.time() - t0:.0f} s")
    rooms_fit = cs.rooms(fit_rooms, seed=seed)
    rooms_check = cs.rooms(check_rooms, seed=seed + 1, exclude=rooms_fit)
    u = np.vstack([cs.to_unit(cs.DEFAULT_THETA)[None, :], lhs(design, len(cs.PARAMS), seed)])
    thetas = [cs.to_theta(x) for x in u]
    t1 = time.time()
    sims = cs.simulate(thetas, rooms_fit, workers=workers)
    failed = sum(len(s["failed"]) for s in sims)
    log(f"[cal SIM] {len(thetas)} parameter vectors x {fit_rooms} rooms x 3 states in {time.time() - t1:.0f} s, "
        f"{failed} runs failed")
    q = np.stack([quantiles(s["features"]) for s in sims])                   # [N, G, J]
    ok = np.isfinite(q).all(axis=(1, 2))
    np.savez(os.path.join(OUT, "design.npz"), u=u, q=q, ok=ok)
    em_cv, rep_cv = fit_emulator(u[ok], q[ok].reshape(int(ok.sum()), -1), holdout=16, seed=seed)
    em, rep = fit_emulator(u[ok], q[ok].reshape(int(ok.sum()), -1), seed=seed)
    log(f"[cal] emulator: held-out R2 {rep_cv['holdout_r2']:.3f}")
    folds = {}
    for name, f in FOLDS.items():
        target = cf.merge([real[t] for t in f["train"]])
        qr = quantiles(target)
        scale = np.array([max(float(np.subtract(*np.percentile(target[g][np.isfinite(target[g])], [75, 25]))), 1e-3)
                          for g in cf.GROUPS])
        res = fit(em, qr, scale, seed=seed)
        theta = cs.to_theta(res["u"])
        with torch.no_grad():
            obj0 = float(_objective(em(torch.tensor(cs.to_unit(cs.DEFAULT_THETA)[None, :], dtype=torch.float32)),
                                    torch.tensor(qr.reshape(-1), dtype=torch.float32),
                                    torch.tensor(scale, dtype=torch.float32))[0])
        folds[name] = {"train_sessions": f["train"], "test_session": f["test"], "theta": theta, "u": res["u"].tolist(),
                       "emulated_objective": res["objective"], "emulated_objective_default": obj0,
                       "top10_u_std": res["top10_u_std"],
                       "at_bound": [cs.NAMES[i] for i, x in enumerate(res["u"]) if x < 0.02 or x > 0.98]}
        log(f"[cal] fold {name}: emulated objective {obj0:.3f} (defaults) -> {res['objective']:.3f}")
    # check on other rooms: defaults and every fold's fit
    t2 = time.time()
    check = cs.simulate([cs.DEFAULT_THETA] + [folds[n]["theta"] for n in FOLDS], rooms_check, workers=workers)
    log(f"[cal SIM] check runs in {time.time() - t2:.0f} s")
    default_feats = check[0]["features"]
    for j, (name, f) in enumerate(FOLDS.items()):
        fitted = check[1 + j]["features"]
        train_real = cf.merge([real[t] for t in f["train"]])
        test_real = real[f["test"]]
        folds[name]["check"] = {
            "label": "SIM->REAL",
            "train_sessions": {"default": w1(default_feats, train_real), "calibrated": w1(fitted, train_real)},
            "test_session": {"default": w1(default_feats, test_real), "calibrated": w1(fitted, test_real)},
            "failed_runs": len(check[1 + j]["failed"]),
        }
    summary = {
        "label": "SIM (parameters) / SIM->REAL (distances)", "spec": "spec/calibration.md",
        "params": [{"name": p[0], "default": p[1], "lo": p[2], "hi": p[3], "scale": p[4]} for p in cs.PARAMS],
        "groups": list(cf.GROUPS), "quantile_levels": J, "design": int(u.shape[0]), "design_ok": int(ok.sum()),
        "fit_rooms": rooms_fit, "check_rooms": rooms_check, "seconds_per_run": cs.SECONDS,
        "failed_design_runs": failed, "emulator": {"holdout": rep_cv, "final": rep}, "folds": folds,
        "real_quantiles": {t: quantiles(real[t]).tolist() for t in real},
        "default_check_quantiles": quantiles(default_feats).tolist(), "wall_s": round(time.time() - t0, 1),
    }
    torch.save(em.state_dict(), os.path.join(OUT, "emulator.pt"))
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Calibrate Room Sim's global parameters against OpenCSI (Phase 10b).")
    ap.add_argument("--design", type=int, default=128)
    ap.add_argument("--fit-rooms", type=int, default=24)
    ap.add_argument("--check-rooms", type=int, default=24)
    ap.add_argument("--workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(RESULTS, "calibration.json"))
    args = ap.parse_args(argv)
    summary = run(args.design, args.fit_rooms, args.check_rooms, workers=args.workers, seed=args.seed)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({n: {"theta": f["theta"], "check": {k: v["calibrated"]["sum_over_iqr"] for k, v in
                                                          f["check"].items() if isinstance(v, dict) and "calibrated" in v}}
                      for n, f in summary["folds"].items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
