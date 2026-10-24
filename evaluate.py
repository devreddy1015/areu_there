#!/usr/bin/env python3
"""Ground-truth evaluation sweep.

Runs the pipeline against synthetic scenes whose true breathing and heart
rates are known by construction, and reports accuracy *and* whether the
reported confidence actually predicts correctness.

That second question is the one that matters. An estimator which is wrong
sometimes is fine if it knows when; an estimator which is confidently
wrong is not usable for anything.

Two suites:

  legacy  The original sweep: 8 breathing x 8 heart rates in the original
          random-phase scene. `--fusion respirfi` runs the same sweep
          through RespirFi subcarrier selection instead of `fuse`.
  nulls   Breathing in physically modelled rooms with Fresnel-zone blind
          spots (scene="geometric"), comparing subcarrier-fusion methods
          on identical data. Asks one question: does RespirFi selection
          improve breathing accuracy at ESP32 bandwidth?

  headroom  Ground truth only, no estimation: in how many positions do
          subcarriers disagree in sign, or all sit in a blind spot? Bounds
          what any subcarrier-fusion method could gain.

  height  Ground truth only: the same questions in the box model
          (room_height_m), paired position for position with the 2-D
          scene, one change at a time: the corner-image fix, the heights,
          then the floor and ceiling.

    python3 evaluate.py [--suite all|legacy|nulls|headroom|height] [--quick] [--workers N]
    python3 evaluate.py --suite legacy [--fusion legacy|respirfi] [--window-s 60] [--duration-s 120]
"""

from __future__ import annotations

import argparse
import itertools
import math
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor

from wifisense.dsp import band_peak, sideband_peak
from wifisense.preprocess import RespirFiFusion, StaticSuppressor, _fft, _next_pow2, fuse
from wifisense.sources import SyntheticCsiSource, SyntheticConfig
from wifisense.vitals import BreathingExtractor

BREATHING_BPM = [8, 10, 12, 14, 16, 18, 20, 24]
HEART_BPM = [50, 55, 60, 68, 75, 85, 95, 110]

BREATHING_TOL_BPM = 0.5
HEART_TOL_BPM = 3.0
CONFIDENCE_GATE = 0.40  # matches VitalEstimate.from_confidence


def run_case(br: float, hr: float, duration_s: float, window_s: float, fusion: str = "legacy"):
    cfg = SyntheticConfig(breathing_bpm=br, heart_bpm=hr)
    source = SyntheticCsiSource(cfg)
    fs = cfg.sample_rate_hz
    suppressor = StaticSuppressor(cfg.n_subcarriers, fs)

    if fusion == "respirfi":
        rf = RespirFiFusion(cfg.n_subcarriers, fs)
        fused = [rf.process(*suppressor.process(f)) for f in source.seconds(duration_s)]
    else:
        fused = [fuse(*suppressor.process(f)) for f in source.seconds(duration_s)]
    tail = fused[-int(window_s * fs) :]

    breathing = band_peak(tail, fs, 0.1, 0.5)
    if breathing is None:
        return None
    heart = sideband_peak(tail, fs, 0.8, 2.0, breathing.freq_hz)
    return breathing, heart


def legacy_suite(args: argparse.Namespace) -> None:
    print("=" * 72)
    print(f"SUITE legacy (original random-phase scene)   fusion: {args.fusion}")
    print("=" * 72)
    rows = []
    for br, hr in itertools.product(BREATHING_BPM, HEART_BPM):
        result = run_case(float(br), float(hr), args.duration_s, args.window_s, args.fusion)
        if result is None:
            continue
        breathing, heart = result
        rows.append(
            {
                "br_true": br,
                "hr_true": hr,
                "br_err": abs(breathing.freq_hz * 60.0 - br),
                "br_conf": breathing.confidence,
                "hr_err": abs(heart.freq_hz * 60.0 - hr) if heart else float("inf"),
                "hr_conf": heart.confidence if heart else 0.0,
            }
        )

    n = len(rows)
    print(f"scenarios: {n}   window: {args.window_s:g}s   duration: {args.duration_s:g}s")
    print()

    br_errs = [r["br_err"] for r in rows]
    br_ok = sum(1 for e in br_errs if e < BREATHING_TOL_BPM)
    print("BREATHING")
    print(f"  mean abs error : {sum(br_errs)/n:.4f} BPM")
    print(f"  max  abs error : {max(br_errs):.4f} BPM")
    print(f"  within {BREATHING_TOL_BPM} BPM  : {br_ok}/{n} ({100*br_ok/n:.0f}%)")
    print()

    hr_errs = [r["hr_err"] for r in rows]
    hr_ok = sum(1 for e in hr_errs if e < HEART_TOL_BPM)
    print("HEART RATE (ungated)")
    print(f"  within {HEART_TOL_BPM} BPM    : {hr_ok}/{n} ({100*hr_ok/n:.0f}%)")
    print()

    print("HEART RATE by reported confidence")
    for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]:
        sel = [r for r in rows if lo <= r["hr_conf"] < hi]
        if not sel:
            continue
        ok = sum(1 for r in sel if r["hr_err"] < HEART_TOL_BPM)
        mean_err = sum(r["hr_err"] for r in sel) / len(sel)
        print(
            f"  conf {lo:.1f}-{hi:.1f} : {len(sel):3d} cases, "
            f"{ok:3d} correct ({100*ok/len(sel):3.0f}%), mean err {mean_err:7.2f} BPM"
        )
    print()

    accepted = [r for r in rows if r["hr_conf"] >= CONFIDENCE_GATE]
    if accepted:
        ok = sum(1 for r in accepted if r["hr_err"] < HEART_TOL_BPM)
        print(f"HEART RATE gated at confidence >= {CONFIDENCE_GATE}")
        print(f"  accepted       : {len(accepted)}/{n} ({100*len(accepted)/n:.0f}% of scenarios)")
        print(f"  correct        : {ok}/{len(accepted)} ({100*ok/len(accepted):.0f}%)")
        print(f"  mean abs error : {sum(r['hr_err'] for r in accepted)/len(accepted):.2f} BPM")
    print()
    print("Synthetic data only. These numbers say the DSP is sound; they say")
    print("nothing about real hardware, real rooms, or real bodies.")
    print()


# ---------------------------------------------------------------------------
# Suite: nulls. Fresnel-zone blind spots in physically modelled rooms.
# ---------------------------------------------------------------------------
#
# Every case is one room instance (seeded furniture layout), one seated
# position, one breathing rate. All fusion methods see the *same* frames.
# Nothing here was tuned on these cases: the RespirFi defaults are the
# paper's or derived (see preprocess.py), and case seeds start at 10000,
# away from anything inspected during development.

# Multipath richness. The more the static phase rotates across the band,
# the more subcarriers can disagree in sign -- the case selection exists for.
ROOMS = {
    "clear-LoS": dict(n_scatterers=6),
    "attenuated-LoS": dict(n_scatterers=12, reflection_order=2, los_attenuation_db=6.0),
    "rich-NLoS": dict(n_scatterers=12, reflection_order=2, los_attenuation_db=15.0, wall_reflection=0.6),
}
BODIES = {"point": dict(), "torso": dict(body_points=5, body_width_m=0.3)}
# Complex noise std per component, against |LoS| = 40: 1.5% (the original
# synthetic default) to 20% amplitude jitter. Real ESP32 CSI sits
# somewhere in 5-20%; P1 will measure it.
NOISES = (0.6, 2.0, 4.0, 8.0)
METHODS = ("legacy", "respirfi", "respirfi-nofloor", "pca", "oracle")
EVAL_TIMES_S = (75.0, 90.0, 105.0, 120.0)
NULL_SENS = 0.2       # |d|h|/dx| (normalised) below this = no first-order response


class PcaFusion(RespirFiFusion):
    """Baseline: first principal component of the band-limited residuals.

    The textbook way to combine subcarriers whose signs disagree: the
    leading eigenvector of their covariance carries the signs for free and
    uses every subcarrier, where RespirFi keeps ~20%. Shares buffering,
    cadence and orientation logic with RespirFiFusion so only the weight
    choice differs.
    """

    def _compute_weights(self, window):
        n_t, n_k = len(window), self.n_subcarriers
        nfft = _next_pow2(n_t)
        bin_hz = self.analysis_rate_hz / nfft
        band = [m for m in range(1, nfft // 2 + 1) if self.f_low <= m * bin_hz <= self.f_high]
        spectra = []
        for k in range(n_k):
            col = [row[k] for row in window]
            mean = sum(col) / n_t
            spec = _fft([complex(x - mean) for x in col] + [0j] * (nfft - n_t))
            spectra.append([spec[m] for m in band])
        cov = [[0.0] * n_k for _ in range(n_k)]
        for i in range(n_k):
            for j in range(i, n_k):
                c = sum((a * b.conjugate()).real for a, b in zip(spectra[i], spectra[j]))
                cov[i][j] = cov[j][i] = c
        v = [1.0] * n_k
        for _ in range(100):
            w = [sum(cov[i][j] * v[j] for j in range(n_k)) for i in range(n_k)]
            norm = math.sqrt(sum(x * x for x in w))
            if norm == 0.0:
                return None
            v = [x / norm for x in w]
        total = sum(abs(x) for x in v)
        return [x / total for x in v] if total > 0.0 else None


def oracle_weights(src: SyntheticCsiSource) -> list[float]:
    """Matched-filter weights from the simulator's ground truth.

    w_k = d|h_k|/dx (signed): the best any linear subcarrier fusion can do
    with equal noise per subcarrier. An upper bound, not a method -- a
    receiver cannot know these.
    """
    out = []
    for k, (s, b) in enumerate(zip(src._static, src._body)):
        h0 = s + b
        pts = src._body_points[k] if src._body_points else [b]
        gains = src._body_gains if src._body_gains else [src._path_gain]
        dh = sum((1j * g * bp for bp, g in zip(pts, gains)), 0j)
        out.append((h0.conjugate() * dh).real / abs(h0) / src._wavelength(k) if abs(h0) else 0.0)
    total = sum(abs(x) for x in out)
    return [x / total for x in out] if total > 0.0 else [1.0 / len(out)] * len(out)


def _scene_config(room: str, body: str, noise: float, seed: int, xy, bpm: float,
                  room_height_m: float | None = None) -> SyntheticConfig:
    return SyntheticConfig(
        scene="geometric",
        body_xy_m=xy,
        breathing_bpm=bpm,
        noise_std=noise,
        seed=seed,
        room_height_m=room_height_m,
        **ROOMS[room],
        **BODIES[body],
    )


def make_cases(n_random: int, n_null: int, n_partial: int, room_height_m: float | None = None) -> list[dict]:
    """Uniformly placed scenes, plus scenes *forced* into blind spots.

    Uniform placement lands in a whole-band null only ~3-6% of the time,
    too rarely to measure anything there, so null and partial-null
    positions are also found by rejection sampling on the simulator's
    ground-truth sensitivity. Results are always broken down by bin, so
    the oversampling never inflates or deflates the uniform numbers.
    """
    cases = []
    case_id = 0
    for room, body, noise in itertools.product(ROOMS, BODIES, NOISES):
        rng = random.Random(f"{room}/{body}/{noise}")
        wanted = [("uniform", None)] * n_random + [("null", (0.0, NULL_SENS))] * n_null + [
            ("partial", (NULL_SENS, 0.5))
        ] * n_partial
        for kind, band in wanted:
            case_id += 1
            seed = 10_000 + case_id
            bpm = rng.uniform(8.0, 24.0)
            for _ in range(5000):
                xy = (rng.uniform(0.6, 5.4), rng.uniform(0.6, 4.4))
                if band is None:
                    break
                src = SyntheticCsiSource(_scene_config(room, body, noise, seed, xy, bpm, room_height_m))
                if band[0] <= max(src.sensitivity()) < band[1]:
                    break
            else:
                continue
            cases.append(dict(room=room, body=body, noise=noise, seed=seed, xy=xy, bpm=bpm, kind=kind,
                              room_height_m=room_height_m))
    return cases


def run_null_case(case: dict) -> dict:
    cfg = _scene_config(case["room"], case["body"], case["noise"], case["seed"], case["xy"], case["bpm"],
                        case["room_height_m"])
    src = SyntheticCsiSource(cfg)
    fs = cfg.sample_rate_hz
    n = cfg.n_subcarriers

    sens_signed = oracle_weights(src)
    sens = src.sensitivity()
    pos = sum(1 for k, w in enumerate(sens_signed) if w > 0 and sens[k] >= NULL_SENS)
    neg = sum(1 for k, w in enumerate(sens_signed) if w < 0 and sens[k] >= NULL_SENS)

    suppressor = StaticSuppressor(n, fs)
    fusions = {
        "respirfi": RespirFiFusion(n, fs),
        "respirfi-nofloor": RespirFiFusion(n, fs, bnr_floor=0.0),
        "pca": PcaFusion(n, fs),
    }
    oracle = sens_signed
    extractors = {m: BreathingExtractor(fs) for m in METHODS}
    eval_frames = {int(t * fs) for t in EVAL_TIMES_S}
    estimates = {m: [] for m in METHODS}

    for i, frame in enumerate(src.seconds(max(EVAL_TIMES_S)), start=1):
        r, w = suppressor.process(frame)
        extractors["legacy"].push(fuse(r, w))
        for name, fu in fusions.items():
            extractors[name].push(fu.process(r, w))
        extractors["oracle"].push(sum(a * b for a, b in zip(oracle, r)))
        if i in eval_frames:
            for m in METHODS:
                e = extractors[m].estimate()
                estimates[m].append((e.value_bpm, e.confidence))

    rf = fusions["respirfi"]
    return dict(
        case,
        max_sens=max(sens),
        mixed_sign=bool(pos and neg),
        estimates=estimates,
        rf_engaged=(rf.passes - rf.failed_passes) / rf.passes if rf.passes else 0.0,
        rf_nofloor_engaged=(
            (fusions["respirfi-nofloor"].passes - fusions["respirfi-nofloor"].failed_passes)
            / fusions["respirfi-nofloor"].passes
            if fusions["respirfi-nofloor"].passes
            else 0.0
        ),
    )


def _score(results: list[dict], method: str) -> dict:
    errs, doubled, gated, gated_ok, conf_wrong = [], 0, 0, 0, 0
    for r in results:
        truth = r["bpm"]
        for bpm, conf in r["estimates"][method]:
            err = abs(bpm - truth)
            errs.append(err)
            doubled += abs(bpm - 2.0 * truth) < 1.0
            if conf >= CONFIDENCE_GATE:
                gated += 1
                gated_ok += err < BREATHING_TOL_BPM
            if conf >= 0.7 and err >= 1.0:
                conf_wrong += 1
    n = len(errs)
    errs.sort()
    return dict(
        n=n,
        ok=sum(e < BREATHING_TOL_BPM for e in errs) / n if n else float("nan"),
        ok1=sum(e < 1.0 for e in errs) / n if n else float("nan"),
        median=errs[n // 2] if n else float("nan"),
        doubled=doubled / n if n else float("nan"),
        coverage=gated / n if n else float("nan"),
        gated_acc=gated_ok / gated if gated else float("nan"),
        conf_wrong=conf_wrong / n if n else float("nan"),
    )


def _table(title: str, results: list[dict]) -> None:
    if not results:
        return
    n_obs = len(results) * len(EVAL_TIMES_S)
    print(f"\n{title}   ({len(results)} scenes x {len(EVAL_TIMES_S)} readings = {n_obs})")
    print(f"  {'method':17s} {'<0.5BPM':>8s} {'<1BPM':>7s} {'median':>7s} {'2x rate':>8s} {'conf>=.4':>9s} {'acc|gate':>9s} {'conf-wrong':>11s}")
    for m in METHODS:
        s = _score(results, m)
        print(
            f"  {m:17s} {100*s['ok']:7.1f}% {100*s['ok1']:6.1f}% {s['median']:7.3f} "
            f"{100*s['doubled']:7.1f}% {100*s['coverage']:8.1f}% {100*s['gated_acc']:8.1f}% {100*s['conf_wrong']:10.1f}%"
        )


#: "Right", for the reliability tables: the S5 gate's tolerance.
RELIABILITY_TOL_BPM = 1.0
RELIABILITY_BINS = 10


def reliability(pairs: list[tuple[float, bool]], bins: int = RELIABILITY_BINS) -> tuple[list[tuple], float]:
    """How far a confidence is from a probability.

    `pairs` is (confidence, was_right) per reading. Returns one row per
    non-empty bin -- (lo, hi, n, mean confidence, share right) -- and the
    expected calibration error: the reading-weighted mean of
    |share right - mean confidence| over the bins. 0 is a confidence that
    means what it says; the breathing confidence is the fraction of in-band
    energy at the peak, which was never built to be a probability, and this
    measures by how much it is not one.
    """
    acc = [[0, 0.0, 0] for _ in range(bins)]
    for conf, ok in pairs:
        b = min(max(int(conf * bins), 0), bins - 1)
        acc[b][0] += 1
        acc[b][1] += conf
        acc[b][2] += bool(ok)
    total = sum(a[0] for a in acc)
    rows, ece = [], 0.0
    for b, (n, csum, right) in enumerate(acc):
        if not n:
            continue
        mean_conf, share = csum / n, right / n
        rows.append((b / bins, (b + 1) / bins, n, mean_conf, share))
        ece += n / total * abs(share - mean_conf)
    return rows, ece


def _reliability_table(title: str, results: list[dict], method: str = "legacy") -> None:
    pairs = [
        (conf, abs(bpm - r["bpm"]) < RELIABILITY_TOL_BPM)
        for r in results
        for bpm, conf in r["estimates"][method]
    ]
    if not pairs:
        return
    rows, ece = reliability(pairs)
    print(f"\n{title}: {len(pairs)} readings, fusion {method}, right = within {RELIABILITY_TOL_BPM:g} BPM")
    print(f"  {'confidence':>12s} {'readings':>9s} {'mean conf':>10s} {'share right':>12s} {'gap':>7s}")
    for lo, hi, n, mc, sh in rows:
        print(f"  {lo:5.1f}-{hi:<5.1f}  {n:9d} {mc:10.3f} {sh:12.3f} {sh - mc:+7.3f}")
    print(f"  expected calibration error: {ece:.3f}")


def _paired(results: list[dict], a: str, b: str) -> tuple[int, int, int]:
    """(a right & b wrong, b right & a wrong, both) over all readings."""
    a_only = b_only = both = 0
    for r in results:
        for (ea, _), (eb, _) in zip(r["estimates"][a], r["estimates"][b]):
            ok_a = abs(ea - r["bpm"]) < BREATHING_TOL_BPM
            ok_b = abs(eb - r["bpm"]) < BREATHING_TOL_BPM
            a_only += ok_a and not ok_b
            b_only += ok_b and not ok_a
            both += ok_a and ok_b
    return a_only, b_only, both


def nulls_suite(args: argparse.Namespace) -> None:
    if args.quick:
        cases = make_cases(3, 2, 2, args.room_height)
    else:
        cases = make_cases(args.scenes, args.scenes // 2, args.scenes // 2, args.room_height)
    workers = args.workers or min(len(cases), os.cpu_count() or 1)
    print("=" * 72)
    print("SUITE nulls -- SYNTHETIC. Fresnel blind spots in geometric rooms, ESP32 HT20")
    print(f"(56 subcarriers x 312.5 kHz = 17.5 MHz). {len(cases)} scenes: rooms {list(ROOMS)},")
    print(f"bodies {list(BODIES)}, noise {list(NOISES)}; breathing U(8, 24) BPM;")
    print("per cell: uniform positions plus positions forced into (partial) nulls;")
    print(f"readings at t = {', '.join(f'{t:g}' for t in EVAL_TIMES_S)} s. Workers: {workers}.")
    if args.room_height is not None:
        print(f"BOX MODEL: rooms {args.room_height:g} m high, concrete floor and ceiling, nodes at")
        print("1.2 m, chests at 1.0 m (seated), furniture 0.3-1.8 m. Same seeds as the 2-D")
        print("study, so the furniture footprints match; positions are drawn afresh.")
    print("=" * 72)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(run_null_case, cases, chunksize=1))
    print(f"ran in {time.time() - t0:.0f}s")

    nulls = [r for r in results if r["max_sens"] < NULL_SENS]
    partial = [r for r in results if NULL_SENS <= r["max_sens"] < 0.5]
    clear = [r for r in results if r["max_sens"] >= 0.5]
    mixed = [r for r in results if r["mixed_sign"]]

    print(
        f"\nscene mix: whole-band null {len(nulls)}, partial {len(partial)}, clear {len(clear)}; "
        f"mixed-sign subcarriers {len(mixed)} of {len(results)}"
    )
    eng = sum(r["rf_engaged"] for r in results) / len(results)
    eng_nf = sum(r["rf_nofloor_engaged"] for r in results) / len(results)
    print(
        f"RespirFi passes that selected anything: {100*eng:.0f}% with the paper's BNR floor of 2, "
        f"{100*eng_nf:.0f}% without it (otherwise it falls back to `fuse`)"
    )

    uniform = [r for r in results if r["kind"] == "uniform"]
    _table("UNIFORM PLACEMENT (unbiased mix of positions)", uniform)
    _table(f"WHOLE-BAND NULL (no subcarrier with sensitivity >= {NULL_SENS})", nulls)
    _table("PARTIAL NULL (best subcarrier 0.2-0.5)", partial)
    _table("CLEAR (best subcarrier >= 0.5)", clear)
    _table("MIXED-SIGN SUBCARRIERS (where sign-flip fusion can matter)", mixed)
    for room in ROOMS:
        _table(f"ROOM {room} (uniform placement)", [r for r in uniform if r["room"] == room])
    for body in BODIES:
        _table(f"BODY {body} (uniform placement)", [r for r in uniform if r["body"] == body])
    for noise in NOISES:
        _table(f"NOISE std {noise} (uniform placement)", [r for r in uniform if r["noise"] == noise])
        _table(f"NOISE std {noise} (partial nulls)", [r for r in partial if r["noise"] == noise])

    print("\nPAIRED vs legacy (readings within 0.5 BPM): +X right & legacy wrong / -legacy right & X wrong")
    for title, subset in (("uniform", uniform), ("partial", partial), ("null", nulls), ("mixed-sign", mixed)):
        cells = []
        for m in METHODS[1:]:
            a, b, _ = _paired(subset, m, "legacy")
            cells.append(f"{m} +{a}/-{b}")
        print(f"  {title:10s}  " + "   ".join(cells))

    print("\nCONFIDENCE vs ACCURACY. The breathing confidence is the share of in-band")
    print("energy at the peak, a gate rather than a probability; these tables measure")
    print("how far it is from one. 'gap' = share right - mean confidence: negative is")
    print("overconfident. Uniform placement is the honest population; the forced nulls")
    print("are over-sampled by design and shown apart, where overconfidence would live.")
    _reliability_table("RELIABILITY, uniform placement", uniform)
    _reliability_table(f"RELIABILITY, whole-band nulls", nulls)

    print()
    print("Columns: '2x rate' = estimate within 1 BPM of twice the truth (the")
    print("blind-spot signature); 'conf>=.4' = readings that pass the DEGRADED gate;")
    print("'acc|gate' = accuracy among those; 'conf-wrong' = confidence >= 0.7 yet")
    print(">= 1 BPM off. 'oracle' uses the simulator's true per-subcarrier response")
    print("and bounds what any linear fusion could achieve; it is not a method.")
    print("SYNTHETIC: geometry, noise and body models are simplifications. These")
    print("numbers compare fusion methods on identical data; they do not predict")
    print("accuracy on real hardware.")


HEADROOM_ROOMS = {
    "clear LoS (order 1, 6 scatterers)": dict(n_scatterers=6),
    "order 2, LoS -6 dB, 12 scatterers": dict(n_scatterers=12, reflection_order=2, los_attenuation_db=6.0),
    "order 2, LoS -15 dB, 12 scatterers": dict(n_scatterers=12, reflection_order=2, los_attenuation_db=15.0),
    "same, Gamma 0.6 (richest)": dict(n_scatterers=12, reflection_order=2, los_attenuation_db=15.0, wall_reflection=0.6),
}


def headroom_suite(args: argparse.Namespace) -> None:
    n = 100 if args.quick else 300
    print("=" * 72)
    print(f"SUITE headroom -- SYNTHETIC ground truth, {n} uniform positions per room")
    print("=" * 72)
    print(f"  {'room':36s} {'mixed-sign':>10s} {'whole-null':>10s} {'avg-eff p10':>11s} {'p50':>5s}")
    for name, kw in HEADROOM_ROOMS.items():
        rng = random.Random(0)
        mixed = whole = 0
        eff = []
        for i in range(n):
            xy = (rng.uniform(0.6, 5.4), rng.uniform(0.6, 4.4))
            src = SyntheticCsiSource(SyntheticConfig(scene="geometric", body_xy_m=xy, seed=1000 + i, **kw))
            sens = src.sensitivity()
            signed = [math.copysign(m, w) for m, w in zip(sens, oracle_weights(src))]
            pos = sum(1 for v in signed if v > NULL_SENS)
            neg = sum(1 for v in signed if v < -NULL_SENS)
            mixed += bool(pos and neg)
            whole += max(sens) < NULL_SENS
            eff.append(abs(sum(signed)) / max(sum(sens), 1e-12))
        eff.sort()
        print(f"  {name:36s} {100*mixed/n:9.1f}% {100*whole/n:9.1f}% {eff[n//10]:11.2f} {eff[n//2]:5.2f}")
    print()
    print("mixed-sign: subcarriers with sensitivity >= 0.2 on both sides of zero, the")
    print("only case sign-flip fusion can fix. whole-null: no subcarrier >= 0.2, which no")
    print("subcarrier selection can fix. avg-eff: fraction of the breathing response a")
    print("plain average keeps (1 = no cancellation). The original random scene scores")
    print(f"{_random_scene_efficiency():.2f}.")
    print()


HEIGHT_ROOMS = {
    "clear LoS (order 1, 6 scatterers)": dict(n_scatterers=6),
    "order 2, LoS -6 dB, 12 scatterers": dict(n_scatterers=12, reflection_order=2, los_attenuation_db=6.0),
}
ROOM_HEIGHT_M = 2.7  # scenes/bedroom.json
# One change per row, so each row's difference from the one above is that
# change alone.
HEIGHT_MODELS = {
    "2-D slice (published model)": {},
    "box, all at 1.2 m, floor+ceiling 0": dict(
        room_height_m=ROOM_HEIGHT_M, tx_z_m=1.2, rx_z_m=1.2, body_z_m=1.2,
        floor_reflection=0.0, ceiling_reflection=0.0, scatterer_z_m=(1.2, 1.2)),
    "+ chest 1.0 m, furniture 0.3-1.8 m": dict(
        room_height_m=ROOM_HEIGHT_M, floor_reflection=0.0, ceiling_reflection=0.0),
    "+ concrete floor and ceiling": dict(room_height_m=ROOM_HEIGHT_M),
}
CHEST_HEIGHTS = {"lying, 0.6 m": 0.6, "seated, 1.0 m": 1.0, "standing, 1.3 m": 1.3}


def _ground_truth(src: SyntheticCsiSource) -> tuple[bool, bool, float, float]:
    """(mixed sign, whole-band null, average efficiency, Rician K in dB)."""
    sens = src.sensitivity()
    signed = [math.copysign(m, w) for m, w in zip(sens, oracle_weights(src))]
    pos = sum(1 for v in signed if v > NULL_SENS)
    neg = sum(1 for v in signed if v < -NULL_SENS)
    eff = abs(sum(signed)) / max(sum(sens), 1e-12)
    los = src.static_paths[0][0] ** 2
    rest = sum(a * a for a, _ in src.static_paths[1:])
    k_db = 10.0 * math.log10(los / rest) if rest > 0.0 else float("inf")
    return bool(pos and neg), max(sens) < NULL_SENS, eff, k_db


def height_suite(args: argparse.Namespace) -> None:
    n = 300 if args.quick else 1000
    print("=" * 72)
    print(f"SUITE height -- SYNTHETIC ground truth, {n} uniform positions per room,")
    print(f"the same positions and furniture in every row; box {ROOM_HEIGHT_M} m high,")
    print("nodes at 1.2 m unless a row says otherwise")
    print("=" * 72)
    for room, kw in HEIGHT_ROOMS.items():
        rng = random.Random(0)
        positions = [(rng.uniform(0.6, 5.4), rng.uniform(0.6, 4.4)) for _ in range(n)]
        print(f"\n{room}")
        print(f"  {'model':38s} {'mixed-sign':>10s} {'whole-null':>10s} {'avg-eff p10':>11s} {'p50':>5s} {'K p50':>7s}")
        nulls = {}
        for name, model in HEIGHT_MODELS.items():
            rows = [
                _ground_truth(SyntheticCsiSource(SyntheticConfig(scene="geometric", body_xy_m=xy, seed=1000 + i, **kw, **model)))
                for i, xy in enumerate(positions)
            ]
            nulls[name] = {i for i, r in enumerate(rows) if r[1]}
            eff = sorted(r[2] for r in rows)
            k = sorted(r[3] for r in rows)
            print(
                f"  {name:38s} {100*sum(r[0] for r in rows)/n:9.1f}% {100*len(nulls[name])/n:9.1f}% "
                f"{eff[n//10]:11.2f} {eff[n//2]:5.2f} {k[n//2]:5.1f}dB"
            )
        first, last = list(HEIGHT_MODELS)[0], list(HEIGHT_MODELS)[-1]
        print(f"  whole-band nulls in both the 2-D slice and the full box: {len(nulls[first] & nulls[last])} "
              f"of {len(nulls[first])} and {len(nulls[last])}")

    print(f"\nCHEST HEIGHT, full box (concrete floor and ceiling), {list(HEIGHT_ROOMS)[0]}")
    print(f"  {'chest':38s} {'mixed-sign':>10s} {'whole-null':>10s} {'avg-eff p10':>11s} {'p50':>5s}")
    rng = random.Random(0)
    positions = [(rng.uniform(0.6, 5.4), rng.uniform(0.6, 4.4)) for _ in range(n)]
    for name, z in CHEST_HEIGHTS.items():
        rows = [
            _ground_truth(SyntheticCsiSource(SyntheticConfig(
                scene="geometric", body_xy_m=xy, seed=1000 + i, room_height_m=ROOM_HEIGHT_M, body_z_m=z,
                **list(HEIGHT_ROOMS.values())[0])))
            for i, xy in enumerate(positions)
        ]
        eff = sorted(r[2] for r in rows)
        print(f"  {name:38s} {100*sum(r[0] for r in rows)/n:9.1f}% {100*sum(r[1] for r in rows)/n:9.1f}% "
              f"{eff[n//10]:11.2f} {eff[n//2]:5.2f}")
    print()
    print("Columns as in the headroom suite. K: line-of-sight power over the summed")
    print("power of every other static path (walls, floor, ceiling, furniture), the")
    print("median over positions. The floor and ceiling use one |Gamma| at every angle")
    print("of incidence, as the walls always have; a real floor bounce at these angles")
    print("depends on polarisation. SYNTHETIC: this compares two models of a room, not")
    print("a model with a room.")
    print()


def _random_scene_efficiency() -> float:
    src = SyntheticCsiSource(SyntheticConfig())
    sens = src.sensitivity()
    signed = []
    for k, (st, b) in enumerate(zip(src._static, src._body)):
        h0 = st + b
        signed.append(math.copysign(sens[k], (h0.conjugate() * 1j * b).real))
    return abs(sum(signed)) / sum(sens)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", choices=("all", "legacy", "nulls", "headroom", "height"), default="all")
    ap.add_argument("--fusion", choices=("legacy", "respirfi"), default="legacy",
                    help="subcarrier fusion for the legacy suite")
    ap.add_argument("--window-s", type=float, default=60.0)
    ap.add_argument("--duration-s", type=float, default=120.0)
    ap.add_argument("--scenes", type=int, default=20,
                    help="nulls suite: uniform scenes per room/body/noise cell (plus half as many forced into nulls and partial nulls)")
    ap.add_argument("--quick", action="store_true", help="nulls suite: 3 uniform + 2 null + 2 partial per cell")
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--room-height", type=float, default=None, metavar="M",
                    help="nulls suite: run in the box model, rooms M metres high (default: the 2-D slice)")
    args = ap.parse_args()

    if args.suite in ("all", "legacy"):
        legacy_suite(args)
    if args.suite in ("all", "headroom"):
        headroom_suite(args)
    if args.suite in ("all", "height"):
        height_suite(args)
    if args.suite in ("all", "nulls"):
        nulls_suite(args)


if __name__ == "__main__":
    main()
