#!/usr/bin/env python3
"""Honest evaluation of RTI localization on synthetic multi-link data.

Everything printed here is SYNTHETIC. It establishes whether the inversion
and tracker are mathematically sound and how they scale. It establishes
nothing about real radios, real rooms or real bodies.

Four design choices make this an honest harness rather than a flattering one:

1. **Trivial baselines run on the same data.** A localizer in a small room
   can look good purely because the room is small. The weighted-link-midpoint
   baseline in particular is a strong, nearly-free estimator, and RTI has to
   beat it to earn its complexity. During development it beat RTI outright
   until two real bugs were found, so it stays in the table permanently.

2. **Moving and still are scored separately, and BOTH report accuracy.**
   An earlier version of this script scored the still case only by whether
   the tracker reported a position, treating any report as "false
   confidence". That was wrong: a still person still breathes, so the links
   are still modulated, and the reports turned out to be accurate. Measuring
   "did it answer" without "was it right" is how you mistake a capability
   for a bug. Both are now reported.

3. **The shadowing-to-scattering ratio is swept**, because it decides
   whether RTI applies at all. Wilson & Patwari validated RTI on RSS over
   802.15.4, where a body SHADOWS the link. WiFi CSI amplitude is dominated
   by coherent multipath instead, which is not monotonic in distance from
   the line of sight. See `shadow_s0_db` below.

4. **Position comes from the image centroid, not its argmax.** RTI images
   are blurry blobs; argmax on a 0.5 m grid quantises and is noisy. Using
   argmax understated RTI by ~0.5 m during development.

    python3 evaluate_localize.py            # full sweep, a few minutes
    python3 evaluate_localize.py --quick    # one seed, shorter runs
"""

from __future__ import annotations

import argparse
import math
import statistics
import time
import warnings
from dataclasses import dataclass, replace

from wifisense.localize.geometry import Geometry, PixelGrid, Room, perimeter_nodes
from wifisense.localize.invert import Inverter
from wifisense.localize.measure import (
    LinkMeasurement,
    StaticSuppressedSignal,
    TotalPowerSignal,
)
from wifisense.localize.sim import MultiLinkSimulator, SimConfig
from wifisense.localize.track import Tracker

ROOM = Room(8.0, 6.0)


@dataclass(frozen=True)
class Config:
    n_nodes: int = 8
    pitch_m: float = 0.5
    lambda_m: float = 0.6
    slot_ms: float = 20.0
    noise_std: float = 0.6
    packet_loss: float = 0.0
    shadow_s0_db: float = 30.0
    signal: str = "power"               # "power" (RSS-like) | "suppressed"
    alpha: float = 1.0
    window_s: float = 1.0
    z_threshold: float = 5.0

    def label(self) -> str:
        return (
            f"N={self.n_nodes} pitch={self.pitch_m} lam={self.lambda_m} "
            f"slot={self.slot_ms:g}ms S0={self.shadow_s0_db:g}dB sig={self.signal}"
        )


@dataclass
class Result:
    cfg: Config
    n_links: int
    n_pixels: int
    per_link_hz: float

    moving_median: float | None
    moving_p90: float | None
    moving_coverage: float

    still_median: float | None
    still_report_rate: float

    base_centre: float | None
    base_maxlink: float | None
    base_weighted: float | None

    solve_ms: float


def build(cfg: Config) -> tuple[Geometry, Inverter]:
    geom = Geometry(
        ROOM,
        perimeter_nodes(ROOM, cfg.n_nodes, inset_m=0.1),
        PixelGrid(ROOM, cfg.pitch_m),
        lambda_m=cfg.lambda_m,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # blind pixels reported separately
        inv = Inverter(geom, alpha=cfg.alpha, sigma=1.0, delta_c=0.5)
    return geom, inv


def baselines(geom: Geometry, y: list[float]) -> dict[str, tuple[float, float]]:
    """The three trivial predictors, from the same measurement vector."""
    centre = ROOM.center
    best = max(range(len(y)), key=y.__getitem__) if y else 0
    maxlink = geom.links[best].midpoint if y else centre
    total = sum(v for v in y if v > 0.0)
    if total > 0.0:
        wx = sum(max(v, 0.0) * l.midpoint[0] for v, l in zip(y, geom.links)) / total
        wy = sum(max(v, 0.0) * l.midpoint[1] for v, l in zip(y, geom.links)) / total
        weighted = (wx, wy)
    else:
        weighted = centre
    return {"centre": centre, "maxlink": maxlink, "weighted": weighted}


def run_config(cfg: Config, seed: int, duration_s: float, calib_s: float) -> Result:
    geom, inv = build(cfg)
    scfg = SimConfig(
        slot_ms=cfg.slot_ms,
        noise_std=cfg.noise_std,
        packet_loss=cfg.packet_loss,
        shadow_s0_db=cfg.shadow_s0_db,
        seed=seed,
    )

    empty = MultiLinkSimulator(geom, replace(scfg, occupied=False))
    per_link_hz = empty.per_link_rate_hz
    signal = TotalPowerSignal() if cfg.signal == "power" else StaticSuppressedSignal()
    meas = LinkMeasurement(
        geom, window_s=cfg.window_s, sample_rate_hz=per_link_hz, signal=signal
    )
    meas.calibrate_from(empty.stream(calib_s))
    if not meas.is_calibrated:
        raise RuntimeError(f"calibration failed: {cfg.label()}")

    sim = MultiLinkSimulator(geom, replace(scfg, occupied=True))
    tracker = Tracker(geom, z_threshold=cfg.z_threshold)

    period = 1.0 / per_link_hz
    nxt = calib_s + cfg.window_s
    mv_err: list[float] = []
    mv_total = 0
    st_err: list[float] = []
    st_total = 0
    base_err: dict[str, list[float]] = {"centre": [], "maxlink": [], "weighted": []}
    solve_t = 0.0
    solves = 0

    for t, tx, rx, frame in sim.stream(duration_s):
        meas.update(t, tx, rx, frame)
        if t < nxt:
            continue
        nxt = t + period

        y = meas.vector()
        t0 = time.perf_counter()
        image = inv.solve(y)
        solve_t += time.perf_counter() - t0
        solves += 1

        est = tracker.update(image, t)
        truth = sim.truth_at(t)

        if truth.moving:
            mv_total += 1
            if est is not None:
                mv_err.append(math.dist((est.x, est.y), (truth.x, truth.y)))
                for name, p in baselines(geom, y).items():
                    base_err[name].append(math.dist(p, (truth.x, truth.y)))
        else:
            st_total += 1
            if est is not None:
                st_err.append(math.dist((est.x, est.y), (truth.x, truth.y)))

    def med(xs: list[float]) -> float | None:
        return statistics.median(xs) if xs else None

    def p90(xs: list[float]) -> float | None:
        return sorted(xs)[min(int(0.9 * len(xs)), len(xs) - 1)] if xs else None

    return Result(
        cfg=cfg,
        n_links=geom.n_links,
        n_pixels=geom.n_pixels,
        per_link_hz=per_link_hz,
        moving_median=med(mv_err),
        moving_p90=p90(mv_err),
        moving_coverage=(len(mv_err) / mv_total) if mv_total else 0.0,
        still_median=med(st_err),
        still_report_rate=(len(st_err) / st_total) if st_total else 0.0,
        base_centre=med(base_err["centre"]),
        base_maxlink=med(base_err["maxlink"]),
        base_weighted=med(base_err["weighted"]),
        solve_ms=(solve_t / solves * 1000.0) if solves else 0.0,
    )


def merge(rs: list[Result]) -> Result:
    def m(a: str) -> float | None:
        v = [getattr(r, a) for r in rs if getattr(r, a) is not None]
        return statistics.median(v) if v else None

    b = rs[0]
    return Result(
        cfg=b.cfg, n_links=b.n_links, n_pixels=b.n_pixels, per_link_hz=b.per_link_hz,
        moving_median=m("moving_median"), moving_p90=m("moving_p90"),
        moving_coverage=m("moving_coverage") or 0.0,
        still_median=m("still_median"), still_report_rate=m("still_report_rate") or 0.0,
        base_centre=m("base_centre"), base_maxlink=m("base_maxlink"),
        base_weighted=m("base_weighted"), solve_ms=m("solve_ms") or 0.0,
    )


def f(v: float | None, spec: str = "6.2f") -> str:
    return "   n/a" if v is None else format(v, spec)


def table(title: str, rows: list[tuple[str, Result]]) -> None:
    print()
    print(title)
    print("-" * 120)
    print(f"{'variant':<20} {'lnk':>4} {'px':>5} {'Hz':>6} | "
          f"{'MOVE med':>8} {'p90':>6} {'cov':>5} | {'STILL med':>9} {'rate':>5} | "
          f"{'centre':>7} {'maxlnk':>7} {'wgt':>7} | {'ms':>5}")
    print("-" * 120)
    for label, r in rows:
        beat = ""
        if r.moving_median is not None and r.base_weighted is not None:
            beat = " *" if r.moving_median < r.base_weighted else "  "
        print(f"{label:<20} {r.n_links:>4} {r.n_pixels:>5} {r.per_link_hz:>6.2f} | "
              f"{f(r.moving_median,'8.2f')} {f(r.moving_p90,'6.2f')} {r.moving_coverage:>4.0%} | "
              f"{f(r.still_median,'9.2f')} {r.still_report_rate:>4.0%} | "
              f"{f(r.base_centre,'7.2f')} {f(r.base_maxlink,'7.2f')} "
              f"{f(r.base_weighted,'7.2f')}{beat}| {r.solve_ms:>5.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--duration-s", type=float, default=None)
    ap.add_argument("--seeds", type=int, default=None)
    args = ap.parse_args()

    dur = args.duration_s or (50.0 if args.quick else 120.0)
    nseed = args.seeds or (1 if args.quick else 3)
    calib = 20.0
    seeds = [11 + 7 * i for i in range(nseed)]
    base = Config()

    sweeps = [
        ("NODE COUNT  (how many radios do we need?)",
         [(f"N={n}", replace(base, n_nodes=n)) for n in (4, 6, 8, 12)]),
        ("LINK SIGNAL  (this is the big one - see notes below)",
         [("total power (RSS)", replace(base, signal="power")),
          ("static-suppressed", replace(base, signal="suppressed"))]),
        ("SHADOWING S0  (does RTI's premise even hold?)",
         [(f"S0={s:g}dB", replace(base, shadow_s0_db=s)) for s in (0.0, 3.0, 10.0, 30.0, 100.0)]),
        ("ELLIPSE WIDTH lambda",
         [(f"lambda={l}", replace(base, lambda_m=l)) for l in (0.3, 0.6, 1.0)]),
        ("GRID PITCH",
         [(f"pitch={p}", replace(base, pitch_m=p)) for p in (0.25, 0.5, 1.0)]),
        ("SLOT TIME  (TDMA rate penalty)",
         [(f"slot={s:g}ms", replace(base, slot_ms=s)) for s in (10.0, 20.0, 40.0)]),
        ("NOISE",
         [(f"noise={n}", replace(base, noise_std=n)) for n in (0.3, 0.6, 1.2)]),
        ("PACKET LOSS",
         [(f"loss={l:.0%}", replace(base, packet_loss=l)) for l in (0.0, 0.15, 0.30)]),
        ("ABSTENTION THRESHOLD",
         [(f"z={z:g}", replace(base, z_threshold=z)) for z in (3.0, 5.0, 8.0)]),
    ]

    print("=" * 120)
    print("RTI LOCALIZATION EVALUATION  ---  SYNTHETIC DATA ONLY")
    print("=" * 120)
    print(f"room {ROOM.width_m:g}x{ROOM.height_m:g} m | {dur:g}s runs x {nseed} seed(s) | "
          f"{calib:g}s empty-room calibration")
    print(f"baseline config: {base.label()}")
    print()
    print("MOVE med/p90 = error in metres while walking.  cov = fraction of moving time answered.")
    print("STILL med/rate = error and report rate while standing still (the subject still breathes,")
    print("              so these reports are not automatically wrong - judge them by the error).")
    print("centre/maxlnk/wgt = trivial baselines on the same answered steps.")
    print("              '*' marks rows where RTI beats the weighted-midpoint baseline.")
    print("              If RTI cannot beat 'wgt', it is not earning its complexity.")

    t0 = time.perf_counter()
    for title, variants in sweeps:
        rows = [(lbl, merge([run_config(c, s, dur, calib) for s in seeds]))
                for lbl, c in variants]
        table(title, rows)

    print()
    print("=" * 120)
    print(f"wall clock: {time.perf_counter() - t0:.1f}s")
    print("SYNTHETIC. Says the maths is sound and how it scales. Says nothing about real hardware.")
    print("=" * 120)


if __name__ == "__main__":
    main()
