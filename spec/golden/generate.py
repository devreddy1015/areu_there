#!/usr/bin/env python3
"""Generate golden vectors from the Python reference implementation.

    $(.venv/bin/uv python find 3.14) -I spec/golden/generate.py

(CPython 3.14 exactly: the committed vectors are 3.14's; main() refuses
any other version.)

Every Saans implementation (Rust gateway, C++ edge) must reproduce these
outputs from these inputs. See SAANS_PLAN.md §4 for tolerances and
spec/wire.md §3 for the rules. Never edit the output by hand.

Format, per vector set NAME (chosen so C++ needs no JSON parser):

    NAME.kv              one "key value" per line: scalars (ints, floats, words)
    NAME__ARRAY.f64      raw little-endian float64, one file per array
    NAME.json            the same scalars plus {array: [file, length]} for tools

Integer and enum arrays are stored as float64 (exact below 2^53). Vital
status codes: 0 unavailable, 1 unreliable, 2 degraded, 3 valid.
"""

from __future__ import annotations

import json
import math
import os
import random
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from wifisense.dsp import BiquadBandpass, band_peak, goertzel_power, sideband_peak  # noqa: E402
from wifisense.features import FeatureExtractor, FeatureFrame  # noqa: E402
from wifisense.preprocess import (  # noqa: E402
    StaticSuppressor,
    _fft,
    corner_to_alpha,
    fuse,
    respirfi_select,
    subcarrier_bnr,
)
from wifisense.presence import PresenceDetector  # noqa: E402
from wifisense.sources import SyntheticConfig, SyntheticCsiSource  # noqa: E402
from wifisense.sources.esp32_udp import SequenceGapFiller  # noqa: E402
from wifisense.types import CsiFrame, VitalStatus  # noqa: E402
from wifisense.vitals import BreathingExtractor, HeartRateExtractor  # noqa: E402

STATUS_CODE = {
    VitalStatus.UNAVAILABLE: 0,
    VitalStatus.UNRELIABLE: 1,
    VitalStatus.DEGRADED: 2,
    VitalStatus.VALID: 3,
}

MANIFEST: list[str] = []


def emit(name: str, scalars: dict, arrays: dict[str, list[float]]) -> None:
    meta = {"name": name, "scalars": scalars, "arrays": {}}
    with open(os.path.join(HERE, f"{name}.kv"), "w") as kv:
        for k, v in scalars.items():
            if isinstance(v, bool):
                v = int(v)
            if isinstance(v, float):
                v = repr(v)
            kv.write(f"{k} {v}\n")
        for k, arr in arrays.items():
            fname = f"{name}__{k}.f64"
            with open(os.path.join(HERE, fname), "wb") as fh:
                fh.write(struct.pack(f"<{len(arr)}d", *[float(x) for x in arr]))
            meta["arrays"][k] = [fname, len(arr)]
            kv.write(f"array.{k} {len(arr)}\n")
    with open(os.path.join(HERE, f"{name}.json"), "w") as fh:
        json.dump(meta, fh, indent=1)
    MANIFEST.append(name)
    sizes = ", ".join(f"{k}[{len(v)}]" for k, v in arrays.items())
    print(f"  {name:28s} {sizes}")


def test_signal(n: int, fs: float, seed: int) -> list[float]:
    """Breathing-like tone + harmonic + cardiac tone + noise + a DC step."""
    rng = random.Random(seed)
    out = []
    for i in range(n):
        t = i / fs
        x = 1.0 * math.sin(2 * math.pi * 0.25 * t) + 0.3 * math.sin(2 * math.pi * 0.5 * t + 0.4)
        x += 0.08 * math.sin(2 * math.pi * 1.2 * t) + rng.gauss(0.0, 0.2)
        x += 3.0 if i > n // 2 else 0.0
        out.append(x)
    return out


# -- primitives ---------------------------------------------------------------


def gen_biquad() -> None:
    fs = 20.0
    x = test_signal(2400, fs, seed=1)
    for name, (lo, hi) in (("biquad_breath", (0.1, 0.5)), ("biquad_presence", (0.1, 2.0)), ("biquad_motion", (1.0, 4.0))):
        f = BiquadBandpass(lo, hi, fs)
        y = [f.process(v) for v in x]
        emit(name, {"f_low": lo, "f_high": hi, "fs": fs}, {"input": x, "output": y})


def gen_goertzel() -> None:
    fs = 4.0
    x = test_signal(240, fs, seed=2)
    freqs = [0.1 + 0.0137 * i for i in range(60)]
    emit("goertzel", {"fs": fs}, {"input": x, "freqs": freqs, "power": [goertzel_power(x, f, fs) for f in freqs]})


def gen_band_peak() -> None:
    rng = random.Random(3)
    cases = []
    # (signal, fs, f_low, f_high, resolution, window, notch list, notch width)
    fs = 2.0
    n = 90
    tone = [math.sin(2 * math.pi * 0.233 * i / fs) for i in range(n)]
    noisy = [v + rng.gauss(0, 0.6) for v in tone]
    noise = [rng.gauss(0, 1.0) for _ in range(n)]
    cases.append(("tone", tone, fs, 0.1, 0.5, 0.002, True, [], 0.0))
    cases.append(("noisy_tone", noisy, fs, 0.1, 0.5, 0.002, True, [], 0.0))
    cases.append(("noise", noise, fs, 0.1, 0.5, 0.002, True, [], 0.0))
    cases.append(("rect_window", noisy, fs, 0.1, 0.5, 0.005, False, [], 0.0))
    cases.append(("notched", noisy, fs, 0.1, 0.5, 0.002, True, [0.233], 0.02))
    # Pins the grid rule: (0.5 - 0.1) / res = 199.99999995 in f64. Rounding
    # gives 201 grid points; the old truncation gave 200.
    cases.append(("grid_rule", noisy, fs, 0.1, 0.5, 0.0020000000005, True, [], 0.0))
    # Pins notch-aware interpolation: the peak's right neighbour is notched
    # and, if used, would pull the estimate toward the notch by an in-range
    # offset. The rule returns the grid point instead.
    near = [math.sin(2 * math.pi * 0.2505 * i / fs) for i in range(n)]
    cases.append(("notch_neighbour", near, fs, 0.1, 0.5, 0.002, True, [0.252], 0.0005))
    for name, sig, fs_, lo, hi, res, win, notch, nw in cases:
        p = band_peak(sig, fs_, lo, hi, resolution_hz=res, window=win, notch_hz=notch or None, notch_width_hz=nw)
        emit(
            f"band_peak_{name}",
            {
                "fs": fs_, "f_low": lo, "f_high": hi, "resolution_hz": res, "window": int(win),
                "notch_width_hz": nw, "found": int(p is not None),
                "freq_hz": p.freq_hz if p else 0.0, "power": p.power if p else 0.0,
                "confidence": p.confidence if p else 0.0,
            },
            {"input": sig, "notch_hz": notch},
        )


def gen_sideband() -> None:
    cfg = SyntheticConfig(breathing_bpm=15.0, heart_bpm=68.0, n_subcarriers=16)
    src = SyntheticCsiSource(cfg)
    sup = StaticSuppressor(16, cfg.sample_rate_hz)
    fused = [fuse(*sup.process(f)) for f in src.seconds(90)]
    hr = HeartRateExtractor(cfg.sample_rate_hz)
    for v in fused:
        hr.push(v)
    sig = list(hr._history)
    p = sideband_peak(sig, hr.analysis_rate_hz, 0.8, 2.0, modulator_hz=0.25, resolution_hz=hr.resolution_hz)
    emit(
        "sideband_peak",
        {"fs": hr.analysis_rate_hz, "f_low": 0.8, "f_high": 2.0, "modulator_hz": 0.25,
         "resolution_hz": hr.resolution_hz, "harmonic_tol_hz": 0.03,
         "freq_hz": p.freq_hz, "power": p.power, "confidence": p.confidence},
        {"input": sig},
    )


def gen_fft_bnr() -> None:
    rng = random.Random(4)
    x = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(64)]
    y = _fft(x)
    emit("fft64", {"n": 64}, {"in_re": [c.real for c in x], "in_im": [c.imag for c in x],
                              "out_re": [c.real for c in y], "out_im": [c.imag for c in y]})
    # BNR on a 120 x 8 window: 3 breathing subcarriers (two signs), a doubled
    # one, noise, a constant (must score 0) and a slow drift.
    rate, n_t = 4.0, 120
    window = []
    for i in range(n_t):
        t = i / rate
        b = math.sin(2 * math.pi * 0.25 * t)
        window.append([
            b + rng.gauss(0, 0.3), -b + rng.gauss(0, 0.3), 0.5 * b + rng.gauss(0, 0.3),
            math.cos(2 * math.pi * 0.5 * t) + rng.gauss(0, 0.3), rng.gauss(0, 1.0),
            7.0, 0.01 * i + rng.gauss(0, 0.05), b + rng.gauss(0, 1.0),
        ])
    bnr, _ = subcarrier_bnr(window, rate, 0.1, 0.5)
    sel = respirfi_select(window, rate, bnr_percentile=0.0)
    signs = [0.0] * 8
    if sel:
        for k, s in sel.signs.items():
            signs[k] = float(s)
    emit("bnr", {"rate_hz": rate, "f_low": 0.1, "f_high": 0.5, "rows": n_t, "cols": 8},
         {"window": [v for row in window for v in row], "bnr": bnr})
    emit("respirfi_select", {"rate_hz": rate, "f_low": 0.1, "f_high": 0.5, "rows": n_t, "cols": 8,
                             "bnr_percentile": 0.0, "bnr_floor": 2.0, "keep_fraction": 0.5,
                             "affinity_threshold": sel.affinity_threshold if sel else 0.0,
                             "selected": int(sel is not None)},
         {"window": [v for row in window for v in row], "signs": signs})


# -- stages --------------------------------------------------------------------


def synthetic_frames(n_sub: int, seconds_empty: float, seconds_occupied: float, bpm: float, seed: int):
    cfg = SyntheticConfig(n_subcarriers=n_sub, breathing_bpm=bpm, heart_bpm=68.0, seed=seed)
    src = SyntheticCsiSource(cfg)
    cfg.occupied = False
    frames = src.seconds(seconds_empty)
    cfg.occupied = True
    frames += src.seconds(seconds_occupied)
    return cfg, frames


def gen_static_suppressor() -> None:
    cfg, frames = synthetic_frames(8, 10, 20, 15.0, seed=5)
    fs = cfg.sample_rate_hz
    sup = StaticSuppressor(8, fs, corner_hz=0.03)
    res, wts, fused = [], [], []
    amps = []
    for f in frames:
        r, w = sup.process(f)
        amps += f.amplitudes
        res += r
        wts += w
        fused.append(fuse(r, w))
    emit("static_suppressor", {"n_subcarriers": 8, "fs": fs, "corner_hz": 0.03,
                               "alpha": corner_to_alpha(0.03, fs), "frames": len(frames)},
         {"amplitudes": amps, "residuals": res, "weights": wts, "fused": fused})


def fused_stream(seconds_empty=70.0, seconds_occupied=110.0, bpm=13.0, seed=6):
    cfg, frames = synthetic_frames(16, seconds_empty, seconds_occupied, bpm, seed)
    sup = StaticSuppressor(16, cfg.sample_rate_hz)
    return cfg, [fuse(*sup.process(f)) for f in frames]


def gen_presence() -> None:
    cfg, fused = fused_stream()
    d = PresenceDetector(cfg.sample_rate_hz, calibration_s=10.0)
    occ, en, th, cal = [], [], [], []
    for v in fused:
        s = d.process(v)
        occ.append(int(s.occupied)); en.append(s.energy); th.append(s.threshold); cal.append(int(s.calibrated))
    emit("presence", {"fs": cfg.sample_rate_hz, "calibration_s": 10.0, "warmup_s": d.warmup_s,
                      "f_low": 0.1, "f_high": 2.0, "sigma_mult": d.sigma_mult,
                      "hold_s": d.hold_samples / cfg.sample_rate_hz, "energy_window_s": 8.0,
                      "threshold_before_calibration": "inf"},
         {"input": fused, "occupied": occ, "energy": en, "threshold": th, "calibrated": cal})


def gen_extractors() -> None:
    cfg, fused = fused_stream()
    fs = cfg.sample_rate_hz
    br = BreathingExtractor(fs)
    hr = HeartRateExtractor(fs)
    every = int(fs)  # one estimate per second
    out = {k: [] for k in ("br_bpm", "br_conf", "br_status", "hr_bpm", "hr_conf", "hr_status", "at")}
    for i, v in enumerate(fused):
        br.push(v)
        hr.push(v)
        if (i + 1) % every == 0:
            b = br.estimate()
            bh = b.value_bpm / 60.0 if b.status in (VitalStatus.VALID, VitalStatus.DEGRADED) else None
            h = hr.estimate(bh)
            out["at"].append(i)
            out["br_bpm"].append(b.value_bpm); out["br_conf"].append(b.confidence); out["br_status"].append(STATUS_CODE[b.status])
            out["hr_bpm"].append(h.value_bpm); out["hr_conf"].append(h.confidence); out["hr_status"].append(STATUS_CODE[h.status])
    emit("extractors", {"fs": fs, "br_window_s": br.window_s, "br_decimation": br.decimation,
                        "br_resolution_hz": br.resolution_hz, "hr_window_s": hr.window_s,
                        "hr_decimation": hr.decimation, "hr_resolution_hz": hr.resolution_hz},
         {"input": fused, **out})


def gen_gapfill() -> None:
    seqs = [0, 1, 2, 5, 6, 6, 7, 40, 41, 0xFFFFFFFE, 0xFFFFFFFF, 1, 2, 3]
    g = SequenceGapFiller(max_gap=5)
    in_amp, out_amp, out_seq, out_from = [], [], [], []
    for i, s in enumerate(seqs):
        a = float(i) * 1.5 + 1.0
        in_amp.append(a)
        for f in g.push(s, CsiFrame([a, 2.0 * a], [0.0, 0.0], s, 20.0)):
            out_amp.append(f.amplitudes[0]); out_seq.append(f.sample_index); out_from.append(i)
    emit("gapfill", {"max_gap": 5, "filled": g.filled, "discontinuities": g.discontinuities, "duplicates": g.duplicates},
         {"seq": seqs, "in_amp": in_amp, "out_amp": out_amp, "out_seq": out_seq, "out_from_input": out_from})


# -- end to end ----------------------------------------------------------------

FEATURE_FIELDS = [
    "node_id", "tx_id", "flags", "seq", "uptime_ms", "csi_rate_chz", "loss_permille",
    "motion_energy", "presence_threshold", "gross_motion", "breath_rms", "breathing_bpm",
    "breathing_conf", "heart_bpm", "heart_conf", "best_bnr", "calib_age_s", "rssi_dbm",
    "noise_floor_dbm", "n_subcarriers",
]


def gen_features() -> None:
    """A link watching an empty room calibrate, then a seated breather, with
    packet loss, one long gap and RSSI variation; two consent settings."""
    n_sub, fs = 32, 20.0
    cfg, frames = synthetic_frames(n_sub, 70.0, 100.0, 16.0, seed=7)
    rng = random.Random(8)
    seq, t_s, keep_amp, rssi = [], [], [], []
    for i, f in enumerate(frames):
        if rng.random() < 0.03 or 2400 <= i < 2440:  # 3% loss and one 2 s outage
            continue
        seq.append(i)
        t_s.append(i / fs + rng.uniform(0.0, 0.004))
        rssi.append(-45 + int(rng.gauss(0, 2)))
        keep_amp.append(f.amplitudes)
    for name, vit, heart in (("features_vitals", True, True), ("features_private", False, False)):
        fx = FeatureExtractor(n_subcarriers=n_sub, sample_rate_hz=fs, node_id=3, tx_id=7,
                              vitals_enabled=vit, heart_enabled=heart, calibration_s=10.0)
        out: list[FeatureFrame] = []
        for s, t, a, r in zip(seq, t_s, keep_amp, rssi):
            out += fx.push(CsiFrame(a, [0.0] * n_sub, s, fs, rssi_dbm=r, node_id=3), s, t, noise_floor_dbm=-92)
        arrays = {k: [float(getattr(fr, k)) for fr in out] for k in FEATURE_FIELDS}
        arrays["bytes"] = [b for fr in out for b in fr.encode()]
        emit(name, {"n_subcarriers": n_sub, "fs": fs, "node_id": 3, "tx_id": 7, "frame_rate_hz": 2.0,
                    "vitals_enabled": int(vit), "heart_enabled": int(heart), "calibration_s": 10.0,
                    "max_gap_s": 1.0, "noise_floor_dbm": -92, "frames_out": len(out), "frame_size": 68},
             {"seq_in": seq, "t_in": t_s, "rssi_in": rssi,
              "amplitudes_in": [v for a in keep_amp for v in a], **arrays})


def gen_codec() -> None:
    frames = [
        FeatureFrame(),
        FeatureFrame(node_id=12, tx_id=3, flags=0b0011_0111, seq=0xFFFFFFFF, uptime_ms=123456789,
                     csi_rate_chz=1987, loss_permille=31, motion_energy=0.15625, presence_threshold=0.05,
                     gross_motion=0.0625, breath_rms=0.2, breathing_bpm=15.25, breathing_conf=0.93,
                     heart_bpm=68.5, heart_conf=0.41, best_bnr=42.0, calib_age_s=3600, rssi_dbm=-61,
                     noise_floor_dbm=-95, n_subcarriers=56),
    ]
    arrays = {k: [float(getattr(fr, k)) for fr in frames] for k in FEATURE_FIELDS}
    arrays["bytes"] = [b for fr in frames for b in fr.encode()]
    emit("feature_codec", {"frames": len(frames), "frame_size": 68, "magic": 0x534E4153}, arrays)


def gen_sim_geometric() -> None:
    """The geometric scene's physics, noise-free, for the Rust simulator.

    Scatterer positions and RCS are recorded explicitly: the Rust side
    cannot reproduce Python's Mersenne Twister, and should not need to.
    """
    cfg = SyntheticConfig(
        scene="geometric", n_subcarriers=56, noise_std=0.0, seed=11, n_scatterers=4,
        reflection_order=2, los_attenuation_db=3.0, wall_reflection=0.45,
        body_points=3, body_width_m=0.3, body_xy_m=(2.6, 3.4), breathing_bpm=14.0,
    )
    src = SyntheticCsiSource(cfg)
    frames = src.frames(200)
    emit("sim_geometric", {
        "n_subcarriers": 56, "fs": cfg.sample_rate_hz, "carrier_hz": cfg.carrier_hz,
        "subcarrier_spacing_hz": cfg.subcarrier_spacing_hz, "room_w": cfg.room_m[0], "room_h": cfg.room_m[1],
        "tx_x": cfg.tx_xy_m[0], "tx_y": cfg.tx_xy_m[1], "rx_x": cfg.rx_xy_m[0], "rx_y": cfg.rx_xy_m[1],
        "body_x": cfg.body_xy_m[0], "body_y": cfg.body_xy_m[1], "body_points": cfg.body_points,
        "body_width_m": cfg.body_width_m, "body_rcs_m2": cfg.body_rcs_m2, "wall_reflection": cfg.wall_reflection,
        "reflection_order": cfg.reflection_order, "los_attenuation_db": cfg.los_attenuation_db,
        "static_magnitude": cfg.static_magnitude, "breathing_bpm": cfg.breathing_bpm,
        "breathing_amplitude_m": cfg.breathing_amplitude_m, "heart_bpm": cfg.heart_bpm,
        "heart_amplitude_m": cfg.heart_amplitude_m, "excess_path_m": src.excess_path_m,
    }, {
        "scatterer_x": [p[0] for p in src.scatterers], "scatterer_y": [p[1] for p in src.scatterers],
        "scatterer_rcs": src.scatterer_rcs,
        "amplitudes": [a for f in frames for a in f.amplitudes],
        "phases": [a for f in frames for a in f.phases],
        "sensitivity": src.sensitivity(),
    })


# --------------------------------------------------------------------------
# Room Sim v1 (spec/roomsim.md §14). Everything below is SIM.
#
# Common conventions for every roomsim_* set:
#   * scalars `scene` (repo-relative path), `scene_sha256`, `seed`,
#     `duration_s` where a run is involved, and `cfg.<name>` for EVERY
#     config value that differs from the §3.2 default (maps as
#     `cfg.<name>.<key>`, pairs as `cfg.<name>.0` / `.1`, bools as 0/1);
#     roomsim_physics_switches holds two runs and puts its run prefix in
#     front of all but `scene`/`scene_sha256` (`a_seed`, `a_cfg.<name>`);
#   * complex arrays are NAME_re / NAME_im; bins ascending k = -32..31;
#     "packet-major" means packet 0's values first;
#   * ints, bools and enums are float64: path kind codes 0 LoS, 1 order-1,
#     2 order-2, 3 object specular, 4 object point, 5 clutter, 6 tail;
#     posture codes 0 stand, 1 sit, 2 lie; -1 means "none";
#   * material codes follow the §5.1 table: 0 concrete, 1 brick,
#     2 drywall, 3 glass, 4 wood, 5 metal.
# --------------------------------------------------------------------------

ROOMSIM_MATERIALS = ("concrete", "brick", "drywall", "glass", "wood", "metal")
ROOMSIM_REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def _roomsim_cfg_scalars(cfg) -> dict:
    out = {}
    for k, v in cfg.non_default().items():
        if isinstance(v, dict):
            for kk in sorted(v, key=str):
                vv = v[kk]
                out[f"cfg.{k}.{kk}"] = int(vv) if isinstance(vv, bool) else vv
        elif isinstance(v, tuple):
            for i, vv in enumerate(v):
                out[f"cfg.{k}.{i}"] = vv
        elif isinstance(v, bool):
            out[f"cfg.{k}"] = int(v)
        elif v is not None:
            out[f"cfg.{k}"] = v
    return out


def _roomsim_run_scalars(rel_scene: str, sim, links=None) -> dict:
    out = {"label": "SIM", "spec": "roomsim_v1", "scene": rel_scene, "scene_sha256": sim.scene.sha256,
           "seed": sim.cfg.seed, "duration_s": sim.cfg.duration_s,
           "carrier_hz": sim.world.carrier_hz, "lambda": sim.world.lam}
    out.update(_roomsim_cfg_scalars(sim.cfg))
    if links is not None:
        out["n_links"] = len(links)
        out["links"] = ",".join(f"{a}-{b}" for a, b in links)
        for i, (a, b) in enumerate(links):
            out[f"link{i}_tx"] = a
            out[f"link{i}_rx"] = b
    return out


def _roomsim_paths_arrays(sim, links, prefix: str = "") -> dict:
    """Static paths of `links` (see gen_roomsim_paths for the layout)."""
    from wifisense.roomsim.paths import pair_index

    a: dict = {k: [] for k in (
        "path_link", "kind", "surf_i", "surf_j", "obj", "face", "clutter", "tail", "npts", "L",
        "coef_re", "coef_im", "coef0_re", "coef0_im", "sigma", "pts_x", "pts_y", "pts_z",
        "link_npaths", "H0_re", "H0_im", "tail_x", "tail_phi")}
    for li, (tx, rx) in enumerate(links):
        lm = sim.link_model(tx, rx)
        allp = lm.paths + lm.tail
        a["link_npaths"].append(len(allp))
        for p in allp:
            a["path_link"].append(li)
            a["kind"].append(p.kind)
            a["surf_i"].append(p.surf_i)
            a["surf_j"].append(p.surf_j)
            a["obj"].append(p.obj)
            a["face"].append(p.face)
            a["clutter"].append(p.clutter)
            a["tail"].append(p.tail)
            a["npts"].append(len(p.points))
            a["L"].append(p.L)
            a["coef_re"].append(p.coef[0])
            a["coef_im"].append(p.coef[1])
            a["coef0_re"].append(p.coef0[0])
            a["coef0_im"].append(p.coef0[1])
            a["sigma"].append(p.sigma)
            for q in p.points:
                a["pts_x"].append(q[0])
                a["pts_y"].append(q[1])
                a["pts_z"].append(q[2])
        for h in lm.channel([]):
            a["H0_re"].append(h[0])
            a["H0_im"].append(h[1])
        for x, phi in sim._tail_draws[pair_index(sim.world.node_ids, tx, rx)]:
            a["tail_x"].append(x)
            a["tail_phi"].append(phi)
    a["clutter_x"] = [c.pos[0] for c in sim.clutter]
    a["clutter_y"] = [c.pos[1] for c in sim.clutter]
    a["clutter_z"] = [c.pos[2] for c in sim.clutter]
    a["clutter_sigma"] = [c.sigma for c in sim.clutter]
    return {prefix + k: v for k, v in a.items()}


def gen_roomsim_materials() -> None:
    """§5: every material x thickness x angle x polarisation, channel 6.

    Inputs: `cos_t[4]` = cos((deg * PI) / 180.0) for deg = 0, 30, 60, 85
    (use these values, do not recompute them); `thickness[2]` = 0.1, 0.23.
    `eta_re/_im[6]`: eta per material code 0..5 (§5.1 table order).

    Main table, 96 rows, row r = ((mat * 2 + ti) * 4 + ai) * 2 + pol
    (pol 0 = TE, 1 = TM): columns `row_material`, `row_thickness`,
    `row_cos_t`, `row_pol` and the coefficients `G`, `R`, `T` (each
    _re/_im), where G is G_TE or G_TM by `pol` and R and T the slab of §5.1
    with that G. R is referenced to the incident face and is what every
    surface reflects with (walls included, §5.1); there is no R_wall.

    Object single-interface G (§5.2), 24 rows, r = (om * 4 + ai) * 2 + pol
    over object materials om = 0 metal, 1 wood, 2 glass: `obj_material`
    (§5.1 material code), `obj_cos_t`, `obj_pol`, `obj_G_re/_im`.
    """
    from wifisense.roomsim.materials import eta, fresnel, slab
    from wifisense.roomsim.numerics import C, carrier_hz

    fc = carrier_hz(6)
    lam = C / fc
    degs = (0.0, 30.0, 60.0, 85.0)
    cos_t = [math.cos((d * math.pi) / 180.0) for d in degs]
    thick = (0.1, 0.23)
    arr: dict = {"cos_t": cos_t, "thickness": list(thick), "eta_re": [], "eta_im": []}
    for k in ("row_material", "row_thickness", "row_cos_t", "row_pol", "G_re", "G_im", "R_re", "R_im",
              "T_re", "T_im"):
        arr[k] = []
    for mi, m in enumerate(ROOMSIM_MATERIALS):
        et = eta(m, fc)
        arr["eta_re"].append(et[0])
        arr["eta_im"].append(et[1])
        for t in thick:
            for ct in cos_t:
                s, g_te, g_tm = fresnel(et, ct)
                for pol, G in ((0, g_te), (1, g_tm)):
                    R, T, _q0 = slab(G, s, ct, t, lam)
                    for k, v in (("row_material", mi), ("row_thickness", t), ("row_cos_t", ct), ("row_pol", pol),
                                 ("G_re", G[0]), ("G_im", G[1]), ("R_re", R[0]), ("R_im", R[1]),
                                 ("T_re", T[0]), ("T_im", T[1])):
                        arr[k].append(v)
    for k in ("obj_material", "obj_cos_t", "obj_pol", "obj_G_re", "obj_G_im"):
        arr[k] = []
    for m in ("metal", "wood", "glass"):
        et = eta(m, fc)
        for ct in cos_t:
            _s, g_te, g_tm = fresnel(et, ct)
            for pol, G in ((0, g_te), (1, g_tm)):
                for k, v in (("obj_material", ROOMSIM_MATERIALS.index(m)), ("obj_cos_t", ct), ("obj_pol", pol),
                             ("obj_G_re", G[0]), ("obj_G_im", G[1])):
                    arr[k].append(v)
    scalars = {"label": "SIM", "spec": "roomsim_v1", "channel": 6, "carrier_hz": fc, "lambda": lam,
               "n_materials": len(ROOMSIM_MATERIALS), "n_angles": len(degs), "n_thickness": len(thick)}
    for i, m in enumerate(ROOMSIM_MATERIALS):
        scalars[f"material_{i}"] = m
    for i, d in enumerate(degs):
        scalars[f"angle_deg_{i}"] = d
    emit("roomsim_materials", scalars, arr)


def gen_roomsim_shadow() -> None:
    """§8.5: the knife edge, the three-edge J and whole legs, LAMBDA at channel 6.

    `nu[17]`, `J1[17]`: nu = -1.0 + 0.25 * float(n), n = 0..16.
    Grid, 216 rows r = (a * 6 + b) * 6 + c over
    v = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0): `grid_nu_1` = v[a] (near side
    edge), `grid_nu_2` = v[b] (far side edge), `grid_nu_t` = v[c] (top),
    `grid_J` = the three-edge J of §8.5 with cap `shadow_cap_db` (20.0).
    Legs, one row each: inputs `leg_p1x/_p1y/_p1z`, `leg_p2x/_p2y/_p2z`,
    cylinder centre `leg_cx/_cy`, `leg_top`, radius `body_radius_m`
    (0.18); outputs `leg_J` (dB; 0.0 for "no loss") and
    `leg_factor` = 10^(-J/20) (1.0 for "no loss").
    """
    from wifisense.roomsim.numerics import C, carrier_hz
    from wifisense.roomsim.people import J1, shadow_J, three_edge_J

    lam = C / carrier_hz(6)
    nus = [-1.0 + 0.25 * float(n) for n in range(17)]
    vals = (-1.0, -0.5, 0.0, 0.5, 1.0, 2.0)
    arr = {"nu": nus, "J1": [J1(v) for v in nus], "grid_nu_1": [], "grid_nu_2": [], "grid_nu_t": [],
           "grid_J": []}
    for a in vals:
        for b in vals:
            for c in vals:
                arr["grid_nu_1"].append(a)
                arr["grid_nu_2"].append(b)
                arr["grid_nu_t"].append(c)
                arr["grid_J"].append(three_edge_J(a, b, c, 20.0))
    legs = [
        # (P1, P2, c, top)
        ((0.15, 0.15, 1.2), (4.05, 0.15, 1.2), (2.1, 0.15), 1.7),     # on the axis, mid-way
        ((0.15, 0.15, 1.2), (4.05, 0.15, 1.2), (1.0, 0.3), 1.7),      # beside the axis
        ((0.15, 0.15, 1.2), (4.05, 0.15, 1.2), (3.0, 0.6), 1.6),      # further off
        ((0.15, 0.15, 1.2), (4.05, 3.45, 1.2), (2.0, 1.8), 1.0),      # under the leg
        ((0.3, 0.3, 1.6), (2.0, 5.0, 0.9), (1.2, 2.8), 1.75),         # sloping leg
        ((0.3, 0.3, 1.6), (2.0, 5.0, 0.9), (2.05, 5.1), 1.75),        # end inside the cylinder
        ((0.3, 0.3, 1.6), (2.0, 5.0, 0.9), (2.05, 5.1), 0.5),         # end above the top
        ((0.3, 0.3, 1.6), (2.0, 5.0, 0.9), (0.2, 0.0), 1.75),         # behind the start
        ((1.0, 1.0, 0.2), (1.0, 1.0, 2.5), (1.0, 1.0), 1.75),         # vertical leg: L2 == 0
        ((5.7, 0.3, 1.4), (3.5, 5.0, 1.4), (4.6, 2.65), 1.62),        # exactly on the axis
    ]
    for k in ("leg_p1x", "leg_p1y", "leg_p1z", "leg_p2x", "leg_p2y", "leg_p2z", "leg_cx", "leg_cy", "leg_top",
              "leg_J", "leg_factor"):
        arr[k] = []
    for P1, P2, c, top in legs:
        J = shadow_J(P1, P2, c, top, 0.18, 20.0, lam)
        for k, v in zip(("leg_p1x", "leg_p1y", "leg_p1z", "leg_p2x", "leg_p2y", "leg_p2z", "leg_cx", "leg_cy",
                         "leg_top"), (*P1, *P2, *c, top)):
            arr[k].append(v)
        arr["leg_J"].append(0.0 if J is None else J)
        arr["leg_factor"].append(1.0 if J is None else 10.0 ** (-J / 20.0))
    emit("roomsim_shadow", {"label": "SIM", "spec": "roomsim_v1", "channel": 6, "lambda": lam,
                            "shadow_cap_db": 20.0, "body_radius_m": 0.18, "n_legs": len(legs)}, arr)


def _gen_roomsim_paths(name: str, rel_scene: str, links) -> None:
    from wifisense.roomsim import Config, RoomSim, load_scene

    sim = RoomSim(load_scene(os.path.join(ROOMSIM_REPO, rel_scene)), Config(duration_s=0.0))
    scalars = _roomsim_run_scalars(rel_scene, sim, links)
    scalars["n_clutter"] = len(sim.clutter)
    scalars["n_surfaces"] = len(sim.world.surfaces)
    scalars["n_walls"] = len(sim.world.walls)
    scalars["implied_walls"] = int(sim.world.implied_walls)
    emit(name, scalars, _roomsim_paths_arrays(sim, links))


def gen_roomsim_paths() -> None:
    """Bedroom static paths, links (0,1) and (0,3), default config.

    One row per path, links in order, each link's paths in §6 order with
    its tail terms (kind 6) last: `path_link` (index into the set's link
    list), `kind`, `surf_i`, `surf_j` (§6.1 surface index: walls in order,
    then floor, then ceiling; -1 unused), `obj`, `face` (object index and
    face 0..4 = +x, -x, +y, -y, top, for kinds 3/4), `clutter` (kind 5),
    `tail` (term n, kind 6), `npts`, `L`, `coef_re/_im` (the static
    coefficient A_p after §6.9; tail: its coefficient), `coef0_re/_im`
    (before §6.9), `sigma` (RCS for kinds 4/5, else 0). `pts_x/_y/_z`:
    every path's points [Tp, ..., Rp] concatenated (npts each; 0 for the
    tail). `link_npaths` per link. `H0_re/_im`: per link, 64 bins, the
    channel with NO people (static paths and tail only). `tail_x`,
    `tail_phi`: per link, the tail_count draws of its pair (§6.8).
    `clutter_x/_y/_z/_sigma`: the run's kept clutter points (§6.7).
    """
    _gen_roomsim_paths("roomsim_paths", "spec/scenes/bedroom.json", [(0, 1), (0, 3)])


def gen_roomsim_paths_hall() -> None:
    """Hall static paths, links (1,6) and (0,7); layout as roomsim_paths."""
    _gen_roomsim_paths("roomsim_paths_hall", "spec/scenes/hall-l-shaped.json", [(1, 6), (0, 7)])


def gen_roomsim_paths_specular() -> None:
    """Object-specular faces (§6.5 step 1, kind 3), which no other set has:
    spec/scenes/fixtures/roomsim_specular.json (implied walls), links
    (0,3), (3,0) and (1,2); layout as roomsim_paths. Link (0,3) and its
    reverse reflect off a metal face at yaw 0 and one at yaw 80 (TE), a
    glass table top at yaw 10 (TM) and a wood face at yaw -5 (TE).
    """
    _gen_roomsim_paths("roomsim_paths_specular", "spec/scenes/fixtures/roomsim_specular.json",
                       [(0, 3), (3, 0), (1, 2)])


def gen_roomsim_motion() -> None:
    """Hall, people[0] walking, default config: §8.1, §9 and §8.3.

    Samples n = 0..299 at `t` = float(n) / 10.0: `x`, `y`, `heading`,
    `s` (walk flag), `v` (leg speed, 0 in a pause) and the body points
    `body_x/_y/_z` (sample-major, `body_points` per sample, table order).
    Segments generated by then, in order: `seg_walk` (1 walk, 0 pause),
    `seg_t0`, `seg_t1`, `seg_x0`, `seg_y0` (from), `seg_x1`, `seg_y1` (to),
    `seg_v`, `seg_heading`. Scalars: the §8.1 draws and the start.
    """
    from wifisense.roomsim import Config, RoomSim, load_scene

    rel = "spec/scenes/hall-l-shaped.json"
    sim = RoomSim(load_scene(os.path.join(ROOMSIM_REPO, rel)), Config(duration_s=0.0))
    p = sim.people_template[0]
    arr: dict = {k: [] for k in ("t", "x", "y", "heading", "s", "v", "body_x", "body_y", "body_z")}
    for n in range(300):
        t = float(n) / 10.0
        st = p.state(t)
        for k, v in zip(("t", "x", "y", "heading", "s", "v"), (t, *st)):
            arr[k].append(v)
        for q in p.body_points(t, st, sim.cfg.body_points):
            arr["body_x"].append(q[0])
            arr["body_y"].append(q[1])
            arr["body_z"].append(q[2])
    for k in ("seg_walk", "seg_t0", "seg_t1", "seg_x0", "seg_y0", "seg_x1", "seg_y1", "seg_v", "seg_heading"):
        arr[k] = []
    for g in p.trajectory.segs:
        for k, v in zip(("seg_walk", "seg_t0", "seg_t1", "seg_x0", "seg_y0", "seg_x1", "seg_y1", "seg_v",
                         "seg_heading"), (int(g.walk), g.t0, g.t1, g.frm[0], g.frm[1], g.to[0], g.to[1], g.v,
                                          g.heading)):
            arr[k].append(v)
    scalars = _roomsim_run_scalars(rel, sim)
    scalars.update({"person": 0, "walking": int(p.walking), "start_x": p.start[0], "start_y": p.start[1],
                    "height": p.height, "f_b": p.f_b, "p2p": p.p2p, "phi_b": p.phi_b, "phi_g": p.phi_g,
                    "n_samples": 300, "n_segments": len(p.trajectory.segs)})
    emit("roomsim_motion", scalars, arr)


def _roomsim_truth_arrays(trs, with_body: bool) -> dict:
    a: dict = {k: [] for k in ("tr_t", "tr_m", "tr_slot", "tr_tx", "p_x", "p_y", "p_heading", "p_walking",
                               "p_posture", "phone_x", "phone_y", "phone_z")}
    if with_body:
        a.update({"body_x": [], "body_y": [], "body_z": []})
    code = {"stand": 0, "sit": 1, "lie": 2}
    for tr in trs:
        a["tr_t"].append(tr.t)
        a["tr_m"].append(tr.m)
        a["tr_slot"].append(tr.slot)
        a["tr_tx"].append(tr.tx)
        for p in tr.truth.people:
            a["p_x"].append(p.x)
            a["p_y"].append(p.y)
            a["p_heading"].append(p.heading)
            a["p_walking"].append(int(p.walking))
            a["p_posture"].append(code[p.posture])
        ph = tr.truth.phone
        if ph is not None:
            a["phone_x"].append(ph[0])
            a["phone_y"].append(ph[1])
            a["phone_z"].append(ph[2])
        if with_body:
            for snap in tr.snapshots:
                for q in snap.points:
                    a["body_x"].append(q[0])
                    a["body_y"].append(q[1])
                    a["body_z"].append(q[2])
    return a


def _roomsim_packet_arrays(pkts) -> dict:
    a: dict = {k: [] for k in ("t", "m", "slot", "tx", "rx", "link_index", "H_re", "H_im")}
    for p in pkts:
        for k, v in (("t", p.t), ("m", p.m), ("slot", p.slot), ("tx", p.tx), ("rx", p.rx),
                     ("link_index", p.link_index)):
            a[k].append(v)
        for h in p.H:
            a["H_re"].append(h[0])
            a["H_im"].append(h[1])
    return a


def _roomsim_bedroom_sim(impair: bool):
    from wifisense.roomsim import Config, RoomSim, load_scene

    rel = "spec/scenes/bedroom.json"
    sim = RoomSim(load_scene(os.path.join(ROOMSIM_REPO, rel)), Config(seed=0, duration_s=0.5, impair=impair))
    return rel, sim


def gen_roomsim_channel() -> None:
    """Bedroom, seed 0, people[0] stationary lying on the bed, 0.5 s,
    `impair` false (so no packet is lost and every packet has H).

    Per packet, §10 order (120 packets): `t`, `m`, `slot`, `tx`, `rx`,
    `link_index`, `rssi_true`, `rssi`, `seq`; `H_re/_im` packet-major,
    64 bins ascending k. Per transmission (40): `tr_t`, `tr_m`, `tr_slot`,
    `tr_tx`, people[0]'s `p_x`, `p_y`, `p_heading`, `p_walking`,
    `p_posture` (code), the phone `phone_x/_y/_z`, and the body points
    `body_x/_y/_z` (transmission-major, `body_points` each, table order).
    """
    rel, sim = _roomsim_bedroom_sim(False)
    trs = list(sim.transmissions())
    pkts = [p for tr in trs for p in tr.packets]
    arr = _roomsim_packet_arrays(pkts)
    arr["rssi_true"] = [p.rssi_true for p in pkts]
    arr["rssi"] = [p.rssi for p in pkts]
    arr["seq"] = [p.seq for p in pkts]
    arr.update(_roomsim_truth_arrays(trs, True))
    scalars = _roomsim_run_scalars(rel, sim)
    scalars.update({"n_packets": len(pkts), "n_transmissions": len(trs), "n_people": 1,
                    "body_points": sim.cfg.body_points})
    emit("roomsim_channel", scalars, arr)


def gen_roomsim_impair() -> None:
    """The roomsim_channel packets with the default impairments (smoothing
    off): the same scene, seed, duration and (t, tx, rx) order.

    Per packet (all 120, lost included): `t`, `tx`, `rx`, `link_index`,
    `lost`, `agc_db` (after this packet's step), `seq` (-1 if lost),
    `rssi` and `rssi_true` and `noise_floor` (0 if lost), the §12.2
    draws `u_loss`, `u_jump`, `u_dir`, `phi0`, `tau`, `rj`, and `noise`
    (256 per packet, packet-major: n_re, n_im per buffer entry).
    Per received packet only, in order: `buffer_pre_re/_im` and
    `buffer_re/_im` (128 entries each, buffer order: LLTF then HT-LTF, FFT
    order k = 0..31, -32..-1), and the datagrams: `dgram` = every byte as
    0..255, concatenated, with `dgram_len` per received packet.
    `buffer_k[128]`, `buffer_active[128]`. Devices, per node id ascending
    (`dev_node`): `dev_alpha/_beta/_gamma/_delta` (node-major,
    ripple_harmonics each), `dev_g_db`, `dev_nf`, `dev_tau`, `dev_rdev_re/_im`
    (node-major, 64 bins ascending k).
    """
    from wifisense.roomsim import BUFFER_ACTIVE, BUFFER_K

    rel, sim = _roomsim_bedroom_sim(True)
    pkts = list(sim.packets())
    a: dict = {k: [] for k in ("t", "tx", "rx", "link_index", "lost", "agc_db", "seq", "rssi", "rssi_true",
                               "noise_floor", "u_loss", "u_jump", "u_dir", "phi0", "tau", "rj", "noise",
                               "buffer_pre_re", "buffer_pre_im", "buffer_re", "buffer_im", "dgram", "dgram_len")}
    for p in pkts:
        d = p.draws
        for k, v in (("t", p.t), ("tx", p.tx), ("rx", p.rx), ("link_index", p.link_index), ("lost", int(p.lost)),
                     ("agc_db", p.agc_db), ("seq", -1 if p.lost else p.seq), ("rssi", 0 if p.lost else p.rssi),
                     ("rssi_true", 0.0 if p.lost else p.rssi_true),
                     ("noise_floor", 0 if p.lost else p.noise_floor), ("u_loss", d.u_loss), ("u_jump", d.u_jump),
                     ("u_dir", d.u_dir), ("phi0", d.phi0), ("tau", d.tau), ("rj", d.rj)):
            a[k].append(v)
        a["noise"].extend(d.noise)
        if p.lost:
            continue
        for re, im in p.buffer_pre:
            a["buffer_pre_re"].append(re)
            a["buffer_pre_im"].append(im)
        for re, im in p.buffer:
            a["buffer_re"].append(re)
            a["buffer_im"].append(im)
        a["dgram"].extend(p.datagram)
        a["dgram_len"].append(len(p.datagram))
    a["buffer_k"] = list(BUFFER_K)
    a["buffer_active"] = [int(x) for x in BUFFER_ACTIVE]
    for k in ("dev_node", "dev_alpha", "dev_beta", "dev_gamma", "dev_delta", "dev_g_db", "dev_nf", "dev_tau",
              "dev_rdev_re", "dev_rdev_im"):
        a[k] = []
    for nid in sorted(sim.devices):
        dv = sim.devices[nid]
        a["dev_node"].append(nid)
        a["dev_alpha"].extend(dv.alpha)
        a["dev_beta"].extend(dv.beta)
        a["dev_gamma"].extend(dv.gamma)
        a["dev_delta"].extend(dv.delta)
        a["dev_g_db"].append(dv.g_dev_db)
        a["dev_nf"].append(dv.nf_dbm)
        a["dev_tau"].append(dv.tau_dev)
        for z in dv.rdev:
            a["dev_rdev_re"].append(z[0])
            a["dev_rdev_im"].append(z[1])
    scalars = _roomsim_run_scalars(rel, sim)
    scalars.update({"n_packets": len(pkts), "n_received": len(a["dgram_len"]), "n_lost": sum(a["lost"]),
                    "n_nodes": len(sim.devices)})
    emit("roomsim_impair", scalars, a)


def gen_roomsim_multi() -> None:
    """spec/scenes/fixtures/roomsim_multi.json: no walls (implied walls), a
    30-degree metal cabinet, a door, 3 walking people ("third" enters at
    the door); tx_period {1: 2}, 2 s, `impair` false (so every packet has H).

    Packets of links (0,1), (1,2), (2,0) only, in §10 order: `t`, `m`,
    `slot`, `tx`, `rx`, `link_index` (in the run's link list), `H_re/_im`
    packet-major. Truth, every transmission: `tr_t`, `tr_m`, `tr_slot`,
    `tr_tx`; people (transmission-major, 3 each): `p_x`, `p_y`,
    `p_heading`, `p_walking`, `p_posture`; `phone_x/_y/_z`. Static paths
    of the same three links with the prefix `paths_` in the roomsim_paths
    layout. Scalars: each person's start and §8.1 draws.
    """
    from wifisense.roomsim import Config, RoomSim, load_scene

    rel = "spec/scenes/fixtures/roomsim_multi.json"
    links = [(0, 1), (1, 2), (2, 0)]
    sim = RoomSim(load_scene(os.path.join(ROOMSIM_REPO, rel)),
                  Config(seed=0, duration_s=2.0, tx_period={1: 2}, impair=False))
    trs = list(sim.transmissions())
    pkts = [p for tr in trs for p in tr.packets if (p.tx, p.rx) in links]
    arr = _roomsim_packet_arrays(pkts)
    arr.update(_roomsim_truth_arrays(trs, False))
    arr.update(_roomsim_paths_arrays(sim, links, prefix="paths_"))
    scalars = _roomsim_run_scalars(rel, sim, links)
    scalars.update({"n_packets": len(pkts), "n_transmissions": len(trs), "n_people": len(sim.people_template),
                    "implied_walls": int(sim.world.implied_walls), "n_walls": len(sim.world.walls)})
    for p in sim.people_template:
        i = p.index
        scalars.update({f"person{i}_start_x": p.start[0], f"person{i}_start_y": p.start[1],
                        f"person{i}_walking": int(p.walking), f"person{i}_height": p.height,
                        f"person{i}_f_b": p.f_b, f"person{i}_p2p": p.p2p, f"person{i}_phi_b": p.phi_b,
                        f"person{i}_phi_g": p.phi_g})
    emit("roomsim_multi", scalars, arr)

def _roomsim_switches_config(**kw):
    """The roomsim_switches configuration: its non-default impairment,
    scheduling and output branches."""
    from wifisense.roomsim import Config

    base = dict(
        seed=5, duration_s=0.5, channel=11, fw_subcarriers=64, body_points=16,
        roles={0: "tx", 7: "rx"}, tx_period={3: 2}, loss_prob_link={"0,7": 1.0, "2,1": 0.5},
        phone_carrier=1, start_heading_rad={1: 0.7},
        smoothing=True, cpe=False, ripple=False, ltf_merge=False, snr_noise=False,
        timing_offset_dev_s=(-2e-08, 3e-08),
    )
    base.update(kw)
    return Config(**base)


def _roomsim_received_arrays(pkts, prefix: str = "") -> dict:
    """Per packet (lost included) and per received packet arrays shared by
    roomsim_switches (see its docstring)."""
    a: dict = {k: [] for k in ("t", "m", "slot", "tx", "rx", "link_index", "lost", "agc_db", "seq", "rssi",
                               "rssi_true", "noise_floor", "buffer_pre_re", "buffer_pre_im", "buffer_re",
                               "buffer_im", "dgram", "dgram_len")}
    for p in pkts:
        for k, v in (("t", p.t), ("m", p.m), ("slot", p.slot), ("tx", p.tx), ("rx", p.rx),
                     ("link_index", p.link_index), ("lost", int(p.lost)), ("agc_db", p.agc_db),
                     ("seq", -1 if p.lost else p.seq), ("rssi", 0 if p.lost else p.rssi),
                     ("rssi_true", 0.0 if p.lost else p.rssi_true),
                     ("noise_floor", 0 if p.lost else p.noise_floor)):
            a[k].append(v)
        if p.lost:
            continue
        for re, im in p.buffer_pre:
            a["buffer_pre_re"].append(re)
            a["buffer_pre_im"].append(im)
        for re, im in p.buffer:
            a["buffer_re"].append(re)
            a["buffer_im"].append(im)
        if p.datagram is not None:
            a["dgram"].extend(p.datagram)
            a["dgram_len"].append(len(p.datagram))
    return {prefix + k: v for k, v in a.items()}


def gen_roomsim_switches() -> None:
    """Hall-l-shaped, impaired, with the non-default impairment, scheduling
    and output branches the other sets leave at their defaults (the
    geometry, object, tail, shadowing and people switches are in
    roomsim_physics_switches): `seed` 5, `channel` 11, `fw_subcarriers`
    64, `body_points` 16, `roles` {0: tx, 7: rx} (nodes 1..6 txrx, so
    transmitters 0..6 and receivers 1..7), `tx_period` {3: 2},
    `loss_prob_link` {"0,7": 1.0, "2,1": 0.5}, `phone_carrier` 1 (the
    seated people[1], on the sofa: stationary, `sit`), `start_heading_rad`
    {1: 0.7}, `smoothing` true, `cpe`, `ripple`, `ltf_merge` and
    `snr_noise` false, `timing_offset_dev_s` (-2e-08, 3e-08); 0.5 s. The
    `cfg.*` scalars rebuild it.

    Per packet (all, lost included), §10 order: `t`, `m`, `slot`, `tx`,
    `rx`, `link_index`, `lost`, `agc_db` (after this packet's step), `seq`
    (-1 if lost), `rssi` (0 if lost), `rssi_true` (0.0 if lost),
    `noise_floor` (0 if lost). Per received packet only, in order:
    `buffer_pre_re/_im` and `buffer_re/_im` (128 entries each, buffer
    order: LLTF then HT-LTF, FFT order), `dgram` (every datagram byte as
    0..255, concatenated) with `dgram_len` per received packet.
    Truth, per transmission: `tr_t`, `tr_m`, `tr_slot`, `tr_tx`; people
    (transmission-major, 2 each): `p_x`, `p_y`, `p_heading`, `p_walking`,
    `p_posture` (code); the phone `phone_x/_y/_z` (one per transmission);
    body points `body_x/_y/_z` (transmission-major, people in order, 16
    each, table order). Devices, node id ascending (`dev_node`):
    `dev_g_db`, `dev_nf`, `dev_tau` (`ripple` is off, so `Rdev = (1, 0)`).

    The `nq_` arrays are a second run of the same configuration with
    `quantize` false and `duration_s` = `nq_duration_s` (0.05): per packet
    `nq_t`, `nq_m`, `nq_slot`, `nq_tx`, `nq_rx`, `nq_link_index`,
    `nq_lost`, `nq_agc_db`, `nq_seq`, `nq_rssi`, `nq_rssi_true`,
    `nq_noise_floor` (lost as above); per received packet
    `nq_buffer_pre_re/_im` and `nq_buffer_re/_im` (the float buffer, equal
    to buffer_pre). It emits no datagram: `nq_dgram` and `nq_dgram_len`
    are empty (zero-length files) and `nq_n_datagrams` is 0. Its truth is
    the first `nq_duration_s` of the main run's and is not stored again.

    Scalars besides `cfg.*`: `n_packets`, `n_received`, `n_lost`,
    `n_transmissions`, `n_people`, `n_nodes`, `n_links`, `n_transmitters`,
    `body_points`; `nq_duration_s`, `nq_quantize` (0), `nq_n_packets`,
    `nq_n_received`, `nq_n_datagrams`; per person `person{i}_walking`
    (0/1), `person{i}_posture` (word), `person{i}_height`. Link (0,7) has
    `loss_prob_link` 1.0, so every packet on it is lost.
    """
    from wifisense.roomsim import RoomSim, load_scene

    rel = "spec/scenes/hall-l-shaped.json"
    scene = load_scene(os.path.join(ROOMSIM_REPO, rel))
    sim = RoomSim(scene, _roomsim_switches_config())
    trs = list(sim.transmissions())
    pkts = [p for tr in trs for p in tr.packets]
    arr = _roomsim_received_arrays(pkts)
    arr.update(_roomsim_truth_arrays(trs, True))
    for k in ("dev_node", "dev_g_db", "dev_nf", "dev_tau"):
        arr[k] = []
    for nid in sorted(sim.devices):
        dv = sim.devices[nid]
        arr["dev_node"].append(nid)
        arr["dev_g_db"].append(dv.g_dev_db)
        arr["dev_nf"].append(dv.nf_dbm)
        arr["dev_tau"].append(dv.tau_dev)
    nq_duration = 0.05
    nq = RoomSim(scene, _roomsim_switches_config(quantize=False, duration_s=nq_duration))
    nq_pkts = list(nq.packets())
    arr.update(_roomsim_received_arrays(nq_pkts, prefix="nq_"))
    scalars = _roomsim_run_scalars(rel, sim)
    scalars.update({
        "n_packets": len(pkts), "n_received": len(arr["dgram_len"]), "n_lost": sum(arr["lost"]),
        "n_transmissions": len(trs), "n_people": len(sim.people_template), "n_nodes": len(sim.devices),
        "n_links": len(sim.links), "n_transmitters": len(sim.transmitters),
        "body_points": sim.cfg.body_points,
        "nq_duration_s": nq_duration, "nq_quantize": 0, "nq_n_packets": len(nq_pkts),
        "nq_n_received": sum(1 for p in nq_pkts if not p.lost),
        "nq_n_datagrams": sum(1 for p in nq_pkts if p.datagram is not None),
    })
    for p in sim.people_template:
        i = p.index
        scalars.update({f"person{i}_walking": int(p.walking), f"person{i}_posture": p.posture,
                        f"person{i}_height": p.height})
    emit("roomsim_switches", scalars, arr)


def gen_roomsim_export() -> None:
    """`export_run` on the bedroom, seed 0, 0.25 s, `loss_prob` 0.25 (the
    rest default): what `saans-cli/tests/roomsim_export.rs` compares with
    `saans roomsim-export spec/scenes/bedroom.json` on the same config.

    `pcap`: every byte of PREFIX.sim.pcap as 0..255 (`pcap_len` bytes).
    The truth file's values, one row per line (transmission, §10 order):
    `tr_m`, `tr_slot`, `tr_t`, `tr_tx`, `tr_npackets` (its packet
    records), `has_phone` (1, or 0 for `"phone": null`) and
    `phone_x/_y/_z` (0.0 when there is none); people, transmission-major
    (`n_people` each): `p_x`, `p_y`, `p_heading`, `p_walking` (0/1),
    `p_posture` (code 0 stand, 1 sit, 2 lie); packets, transmission-major
    (`tr_npackets` each, receivers in id order): `pk_rx`,
    `pk_link_index`, `pk_lost` (0/1), `pk_seq`, `pk_rssi`, `pk_rssi_true`
    (for a lost packet, whose three JSON values are null, the sentinels
    are `pk_seq` -1, `pk_rssi` 0 and `pk_rssi_true` 0.0; `pk_lost` says
    which). Every truth line also has `"label": "SIM"` and each person
    `"name": "people[i]"` (checked here, not stored).

    The manifest, as scalars: `counts_transmissions`, `counts_packets`,
    `counts_lost`, `counts_datagrams`; `gateway_faithful` (0/1);
    `receivers` (the `receiver_rate_hz` keys in file order, ascending,
    comma-separated, e.g. "0,1,2,3"), `n_receivers`, and
    `receiver_rate_hz.<rx>` per receiver (Hz). As arrays, per link in link
    order: `link_tx`, `link_rx`, `link_rate_hz` (`per_link_rate_hz`),
    `link_packets`, `link_lost`, `link_datagrams`. Scalars `pcap_len`,
    `n_transmissions` (truth lines), `n_people`, `n_packets` (packet
    records).
    """
    import tempfile

    from wifisense.roomsim import Config, RoomSim, export_run, load_scene

    rel = "spec/scenes/bedroom.json"
    sim = RoomSim(load_scene(os.path.join(ROOMSIM_REPO, rel)), Config(duration_s=0.25, loss_prob=0.25))
    code = {"stand": 0, "sit": 1, "lie": 2}
    with tempfile.TemporaryDirectory() as d:
        info = export_run(sim, os.path.join(d, "export"))
        with open(info["pcap"], "rb") as fh:
            pcap = fh.read()
        with open(info["truth"], encoding="utf-8") as fh:
            lines = [json.loads(x) for x in fh]
        with open(info["manifest"], encoding="utf-8") as fh:
            man = json.load(fh)
    arr: dict = {"pcap": list(pcap)}
    for k in ("tr_m", "tr_slot", "tr_t", "tr_tx", "tr_npackets", "has_phone", "phone_x", "phone_y", "phone_z",
              "p_x", "p_y", "p_heading", "p_walking", "p_posture",
              "pk_rx", "pk_link_index", "pk_lost", "pk_seq", "pk_rssi", "pk_rssi_true"):
        arr[k] = []
    for rec in lines:
        assert rec["label"] == "SIM"
        for k in ("m", "slot", "t", "tx"):
            arr["tr_" + k].append(rec[k])
        arr["tr_npackets"].append(len(rec["packets"]))
        ph = rec["phone"]
        arr["has_phone"].append(0 if ph is None else 1)
        for i, k in enumerate(("phone_x", "phone_y", "phone_z")):
            arr[k].append(0.0 if ph is None else ph[i])
        for i, p in enumerate(rec["people"]):
            assert p["name"] == f"people[{i}]"
            for k in ("x", "y", "heading"):
                arr["p_" + k].append(p[k])
            arr["p_walking"].append(int(p["walking"]))
            arr["p_posture"].append(code[p["posture"]])
        for pk in rec["packets"]:
            arr["pk_rx"].append(pk["rx"])
            arr["pk_link_index"].append(pk["link_index"])
            arr["pk_lost"].append(int(pk["lost"]))
            arr["pk_seq"].append(-1 if pk["seq"] is None else pk["seq"])
            arr["pk_rssi"].append(0 if pk["rssi"] is None else pk["rssi"])
            arr["pk_rssi_true"].append(0.0 if pk["rssi_true"] is None else pk["rssi_true"])
    for k in ("link_tx", "link_rx", "link_rate_hz", "link_packets", "link_lost", "link_datagrams"):
        arr[k] = []
    for lk in man["links"]:
        for k, v in (("link_tx", lk["tx"]), ("link_rx", lk["rx"]), ("link_rate_hz", lk["per_link_rate_hz"]),
                     ("link_packets", lk["packets"]), ("link_lost", lk["lost"]),
                     ("link_datagrams", lk["datagrams"])):
            arr[k].append(v)
    assert man["label"] == "SIM" and man["scene_sha256"] == sim.scene.sha256
    scalars = _roomsim_run_scalars(rel, sim)
    scalars.update({"pcap_len": len(pcap), "n_transmissions": len(lines),
                    "n_people": len(lines[0]["people"]) if lines else 0,
                    "n_packets": len(arr["pk_rx"])})
    for k in ("transmissions", "packets", "lost", "datagrams"):
        scalars["counts_" + k] = man["counts"][k]
    scalars["gateway_faithful"] = int(man["gateway_faithful"])
    rates = man["receiver_rate_hz"]
    scalars["receivers"] = ",".join(rates)
    scalars["n_receivers"] = len(rates)
    for k, v in rates.items():
        scalars[f"receiver_rate_hz.{k}"] = v
    emit("roomsim_export", scalars, arr)


#: The roomsim_physics_switches runs: (prefix, config keywords, links).
ROOMSIM_PHYSICS_RUNS = (
    ("a", dict(seed=1, duration_s=0.5, impair=False, reflection_order=1, floor_ceiling=False,
               object_specular=False, object_scatter_scale=0.5, clutter_count=5, tail=False,
               n_people=3, people_mode={2: "stationary"}, people_posture={1: "stand"},
               start_heading_rad={2: 2.0}),
     ((4, 6), (0, 1), (0, 5))),
    ("b", dict(seed=2, duration_s=0.5, impair=False, reflection_order=0, transmission=False,
               object_scatter=False, object_blockage=False, shadowing=False),
     ((5, 7), (1, 6), (0, 1))),
)


def gen_roomsim_physics_switches() -> None:
    """The non-default geometry, object, tail, shadowing and people
    branches, on spec/scenes/hall-l-shaped.json with `impair` false (so
    every packet has H): two runs, variant `a` and variant `b`, 0.5 s each.

    Variant `a` (seed 1): `reflection_order` 1, `floor_ceiling` false (6
    surfaces, the walls), `object_specular` false (the long table's top,
    specular on link (4,6) by default, becomes a kind-4 point),
    `object_scatter_scale` 0.5, `clutter_count` 5, `tail` false,
    `n_people` 3 (people[2] has no scene start and enters at the door, at
    (0.4, 1.5)), `people_mode` {2: stationary} (it stays there, standing),
    `start_heading_rad` {2: 2.0}, `people_posture` {1: stand} (the sitter
    on the sofa stands on it, so the sofa now blocks its body legs);
    links (4,6), (0,1), (0,5). Variant `b` (seed 2): `reflection_order` 0,
    `transmission` false (link (1,6) crosses the re-entrant walls),
    `object_scatter` false (only the table top's kind-3 path is left of
    the objects, link (5,7)), `object_blockage` false, `shadowing` false;
    the scene's two people (people[0] walks, people[1] sits on the sofa);
    links (5,7), (1,6), (0,1). Every switch changes H on at least one of
    its run's links. The rest is default in both.

    Layout. Top-level scalars: `label`, `spec`, `scene`, `scene_sha256`,
    `carrier_hz`, `lambda` (both runs are on channel 6), `n_variants` (2),
    `variants` ("a,b", the prefixes in order). Everything else belongs to
    one run and carries its prefix `V_` (V = a or b): scalars and arrays
    alike, so `a_kind` is run a's `kind`. Per run, scalars `V_seed`,
    `V_duration_s`, `V_cfg.*` (every non-default value of that run, by
    the common convention with the prefix in front: `a_cfg.n_people 3`,
    `a_cfg.people_mode.2 stationary`, `b_cfg.transmission 0`, ...; they
    rebuild the run's Config), `V_n_links`, `V_links` ("4-6,0-1,0-5"),
    `V_link{i}_tx`, `V_link{i}_rx`, `V_n_clutter`, `V_n_surfaces`,
    `V_n_walls`, `V_n_packets` (stored packets, below),
    `V_n_transmissions`, `V_n_people`, `V_body_points`, and per person i:
    `V_person{i}_start_x`, `_start_y`, `_walking` (0/1, the §8.2 mode),
    `_posture` (word), `_on_object` (§8.2 object index in file order, -1
    none; -1 for a walking person), `_heading0` (`start_heading_rad`,
    default 0.0), `_height`, `_f_b`, `_p2p`, `_phi_b`, `_phi_g` (§8.1).

    Per run, arrays:
    * static paths of its links, in the roomsim_paths layout with the
      prefix: `V_path_link`, `V_kind`, `V_surf_i`, `V_surf_j`, `V_obj`,
      `V_face`, `V_clutter`, `V_tail`, `V_npts`, `V_L`, `V_coef_re/_im`,
      `V_coef0_re/_im`, `V_sigma`, `V_pts_x/_y/_z`, `V_link_npaths`,
      `V_H0_re/_im` (64 per link, no people), `V_tail_x`, `V_tail_phi`
      (`tail_count` = 40 draws per link, made whether or not `tail` is on,
      §6.8), `V_clutter_x/_y/_z/_sigma` (the run's kept clutter points);
    * every packet of its links, §10 order (transmissions in order, then
      receivers by id): `V_t`, `V_m`, `V_slot`, `V_tx`, `V_rx`,
      `V_link_index` (in the run's full link list), `V_H_re/_im`
      (packet-major, 64 bins ascending k);
    * truth, every transmission of the run (all transmitters): `V_tr_t`,
      `V_tr_m`, `V_tr_slot`, `V_tr_tx`; people (transmission-major,
      `V_n_people` each): `V_p_x`, `V_p_y`, `V_p_heading`, `V_p_walking`
      (the §8.3 walk flag at that instant), `V_p_posture` (code); the
      phone `V_phone_x/_y/_z` (one per transmission; carried by people[0]);
      body points `V_body_x/_y/_z` (transmission-major, people in order,
      `V_body_points` each, table order).
    """
    from wifisense.roomsim import Config, RoomSim, load_scene

    rel = "spec/scenes/hall-l-shaped.json"
    scene = load_scene(os.path.join(ROOMSIM_REPO, rel))
    scalars: dict = {"label": "SIM", "spec": "roomsim_v1", "scene": rel, "scene_sha256": scene.sha256}
    runs = []
    for v, kw, links in ROOMSIM_PHYSICS_RUNS:
        sim = RoomSim(scene, Config(**kw))
        runs.append((v, sim, list(links)))
    scalars.update({"carrier_hz": runs[0][1].world.carrier_hz, "lambda": runs[0][1].world.lam,
                    "n_variants": len(runs), "variants": ",".join(v for v, _s, _l in runs)})
    arr: dict = {}
    for v, sim, links in runs:
        assert sim.world.carrier_hz == scalars["carrier_hz"] and sim.world.lam == scalars["lambda"]
        trs = list(sim.transmissions())
        pkts = [p for tr in trs for p in tr.packets if (p.tx, p.rx) in links]
        a = _roomsim_paths_arrays(sim, links)
        a.update(_roomsim_packet_arrays(pkts))
        a.update(_roomsim_truth_arrays(trs, True))
        sc = {"seed": sim.cfg.seed, "duration_s": sim.cfg.duration_s}
        sc.update(_roomsim_cfg_scalars(sim.cfg))
        sc.update({"n_links": len(links), "links": ",".join(f"{x}-{y}" for x, y in links)})
        for i, (x, y) in enumerate(links):
            sc[f"link{i}_tx"] = x
            sc[f"link{i}_rx"] = y
        sc.update({"n_clutter": len(sim.clutter), "n_surfaces": len(sim.world.surfaces),
                   "n_walls": len(sim.world.walls), "n_packets": len(pkts), "n_transmissions": len(trs),
                   "n_people": len(sim.people_template), "body_points": sim.cfg.body_points})
        for p in sim.people_template:
            i = p.index
            sc.update({f"person{i}_start_x": p.start[0], f"person{i}_start_y": p.start[1],
                       f"person{i}_walking": int(p.walking), f"person{i}_posture": p.posture,
                       f"person{i}_on_object": p.on_object, f"person{i}_heading0": p.heading0,
                       f"person{i}_height": p.height, f"person{i}_f_b": p.f_b, f"person{i}_p2p": p.p2p,
                       f"person{i}_phi_b": p.phi_b, f"person{i}_phi_g": p.phi_g})
        scalars.update({f"{v}_{k}": x for k, x in sc.items()})
        arr.update({f"{v}_{k}": x for k, x in a.items()})
    emit("roomsim_physics_switches", scalars, arr)


# --------------------------------------------------------------------------
# Locate v1 (spec/locate.md §12). `localize_*` sets pin the ported RTI
# modules (wifisense/localize/geometry.py, invert.py, measure.py, track.py)
# and CPython's two-argument math.hypot; `locate_*` sets pin the new pieces
# (radius, baseline, phone) and the Room Sim pipeline. Everything is SIM.
#
# Common conventions for every localize_* and locate_* set:
#   * scalars `label` ("SIM") and `spec` ("locate_v1"); a set whose inputs
#     come from a scene file has `scene` (repo-relative) and `scene_sha256`
#     (of the file's bytes); random inputs are recorded arrays, never draws
#     the reader must reproduce (Rng streams are named where the set pins
#     the stream itself);
#   * EVERY array NAME has a scalar `cmp.NAME` saying how it is compared:
#       input       a recorded input: use it as given;
#       exact       identical (discrete values, flags, codes, counts,
#                   indices, uniform draws; a NaN matches a NaN);
#       bits        bit-identical, sign of zero and NaN included (py_hypot);
#       rule        spec/roomsim.md §14's rule |a - b| <= 1e-9*|b| + 1e-12*M,
#                   M the largest finite |b_j| of the whole array;
#       rule:N      the same rule with M taken over each run of N
#                   consecutive values (one row, one tick, one epoch);
#       rule:each   M = |b| (one value per group: a pure relative test);
#       rule:ptr:P  groups of P[i] consecutive values, P another array of
#                   the set (CSR row lengths);
#       tie:V       an integer RSSI: identical unless the pre-rounding value
#                   V[i] (same index) is within 1e-6 of a .5 tie;
#       le:B        a rounding-level diagnostic: both values <= B (a number,
#                   or an array of the set read at the same index);
#     a non-finite b matches only the same non-finite a. Scalars named in
#     a `cmp.NAME` entry follow it too; other float scalars compare by the
#     rule with M = |b|, and every other scalar is identical;
#   * reason codes: -1 published, else the index in REASONS (scalar
#     `reasons`): 0 no_disturbance, 1 implausible_radius, 2 not_significant,
#     3 no_reports, 4 too_few_ranges, 5 degenerate, 6 not_converged,
#     7 gated;
#   * an absent value (an abstention's position, a failed report, a None)
#     is stored as 0.0, and a reason or a present flag says which;
#   * error codes (EXC_CODE, by Python exception class): 0 none,
#     1 ValueError, 2 KeyError, 3 IndexError, 4 MalformedMatrixError,
#     5 AsymmetricMatrixError, 6 NotPositiveDefiniteError,
#     7 IllConditionedError, 8 LocateError; each set lists its error cases
#     in order and describes each in a scalar `err.<i>`;
#   * posture codes 0 stand, 1 sit, 2 lie; signal codes 0 rssi_dbm,
#     1 total_power_db, 2 static_suppressed_amplitude;
#   * a set holding several runs or geometries lists their prefixes in
#     `variants` (comma-separated) and puts `<prefix>_` in front of each of
#     their scalars and arrays (the roomsim_physics_switches convention);
#   * ints, bools and codes are stored as float64 (exact below 2^53).
#
# Several sets depend on CPython >= 3.12's Neumaier-compensated builtin
# `sum()` (spec/locate.md §1 A): generate on CPython 3.14 only.
# --------------------------------------------------------------------------

LOCATE_REPO = ROOMSIM_REPO
POSTURE_CODE = {"stand": 0, "sit": 1, "lie": 2}
SIGNAL_CODE = {"rssi_dbm": 0, "total_power_db": 1, "static_suppressed_amplitude": 2}


def _lemit(name: str, scalars: dict, arrays: dict, cmp: dict) -> None:
    """emit() for a localize_/locate_ set: adds `label`, `spec`, `reasons`
    and one `cmp.NAME` per array (and per named scalar)."""
    from wifisense.localize.radius import REASONS

    missing = sorted(set(arrays) - set(cmp))
    assert not missing, f"{name}: arrays without cmp: {missing}"
    for k, v in cmp.items():
        assert k in arrays or k in scalars, f"{name}: cmp for unknown {k}"
        assert v in ("input", "exact", "bits", "rule", "rule:each") or v.startswith(
            ("rule:", "tie:", "le:")), f"{name}: cmp {v}"
        if v.startswith("rule:ptr:"):
            ptr = v[len("rule:ptr:"):]
            assert ptr in arrays and int(sum(arrays[ptr])) == len(arrays[k]), f"{name}: {k} groups by {ptr}"
        elif v.startswith("rule:") and v != "rule:each":
            n = int(v[5:])
            assert n > 0 and len(arrays[k]) % n == 0, f"{name}: {k} is not a multiple of {n}"
        elif v.startswith(("tie:", "le:")):
            ref = v.split(":", 1)[1]
            if ref in arrays:
                assert len(arrays[ref]) == len(arrays[k]), f"{name}: {k} vs {ref}"
            else:
                float(ref)
    out = {"label": "SIM", "spec": "locate_v1", "reasons": ",".join(REASONS)}
    out.update(scalars)
    for k in sorted(cmp):
        out[f"cmp.{k}"] = cmp[k]
    emit(name, out, arrays)


def _exc_code(fn) -> int:
    """EXC_CODE of calling `fn()` (0 if it returns)."""
    from wifisense.localize.config import LocateError
    from wifisense.localize.invert import (
        AsymmetricMatrixError, IllConditionedError, MalformedMatrixError, NotPositiveDefiniteError,
    )

    try:
        fn()
    except IllConditionedError:
        return 7
    except NotPositiveDefiniteError:
        return 6
    except AsymmetricMatrixError:
        return 5
    except MalformedMatrixError:
        return 4
    except LocateError:
        return 8
    except IndexError:
        return 3
    except KeyError:
        return 2
    except ValueError:
        return 1
    return 0


def _err_cases(cases) -> tuple[dict, list]:
    """(`err.<i>` scalars, codes) for `[(description, fn), ...]`."""
    sc = {"n_err": len(cases)}
    codes = []
    for i, (desc, fn) in enumerate(cases):
        sc[f"err.{i}"] = desc
        codes.append(_exc_code(fn))
    return sc, codes


def _scene_scalars(rel: str) -> dict:
    import hashlib

    with open(os.path.join(LOCATE_REPO, rel), "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    return {"scene": rel, "scene_sha256": digest}


def _reason_code(reason) -> int:
    from wifisense.localize.radius import REASONS

    return -1 if reason is None else REASONS.index(reason)


def _est_row(rec: dict) -> tuple:
    """(reason code, x, y, radius_m, cov_xx, cov_xy, cov_yy, confidence)
    of a §11 Estimate or Abstention; 0.0 for absent values."""
    if "reason" in rec:
        return (_reason_code(rec["reason"]), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    assert rec["radius_calibrated"] is False
    conf = rec["confidence"]
    return (-1, rec["x"], rec["y"], rec["radius_m"], rec["cov_xx"], rec["cov_xy"], rec["cov_yy"],
            0.0 if conf is None else conf)


EST_FIELDS = ("reason", "x", "y", "radius", "cov_xx", "cov_xy", "cov_yy", "confidence")
EST_CMP = {"reason": "exact", "x": "rule", "y": "rule", "radius": "rule:each", "cov_xx": "rule:each",
           "cov_xy": "rule:each", "cov_yy": "rule:each", "confidence": "rule:each"}


def _rng_draws(stream: int, n: int, lo: float, hi: float) -> list[float]:
    from wifisense.roomsim.rng import Rng

    rng = Rng.stream(0, stream)
    return [rng.range(lo, hi) for _ in range(n)]


def _rng_gauss(stream: int, n: int) -> list[float]:
    from wifisense.roomsim.rng import Rng

    rng = Rng.stream(0, stream)
    return [rng.gauss() for _ in range(n)]


# -- localize_*: the ported RTI ------------------------------------------------


def gen_localize_py_hypot() -> None:
    """CPython 3.14's two-argument `math.hypot` (Modules/mathmodule.c
    `vector_norm`), which Rust ports as `py_hypot` (spec/locate.md §1 A,
    §12). Every comparison is bit for bit.

    Edge pairs `a[i]`, `b[i]` -> `h[i]` (`cmp bits`): every ordered pair of
    the 35 values in scalar `edge_values` order (zeros of both signs,
    subnormals, the normal/subnormal boundary, 2^-511, 2^511, 2^1023, the
    largest finite, infinities, a NaN, ordinary values), then ratios near
    2^-27..2^-29 and 2^25..2^29 with their 1-ulp neighbours, both orders,
    and equal magnitudes of both signs.

    Random pairs: `rng = Rng.stream(0, 320000)` (saans_sim's generator);
    for i in 0..n_random-1: `a = rng.range(-10.0, 10.0)`, then
    `b = rng.range(-10.0, 10.0)`, `h = hypot(a, b)`. `sha256_random` is the
    SHA-256 (hex) of the n_random results as little-endian f64, in order;
    `rand_a`, `rand_b`, `rand_h` are the first 64 (`input`/`input`/`bits`)
    for debugging; `n_random_not_sqrt` counts results that differ from
    `sqrt(a*a + b*b)` (why it is not libm's or the naive formula).
    """
    import hashlib

    from wifisense.roomsim.rng import Rng

    vals = [0.0, -0.0, 5e-324, -5e-324, 1e-320, 2.225073858507201e-308, 2.2250738585072014e-308, 1e-300,
            2.0 ** -511, 1e-154, 1e-10, 2.0 ** -27, 0.5, 1.0, -1.0, 1.0000000000000002, 1.5, 2.0, 3.0, 4.0,
            -4.0, 5.0, 0.1, 0.3, 1e10, 2.0 ** 27, 2.0 ** 511, 1e154, 1e300, 2.0 ** 1023,
            1.7976931348623157e308, -1.7976931348623157e308, math.inf, -math.inf, math.nan]
    pairs = [(x, y) for x in vals for y in vals]
    for k in (25, 26, 27, 28, 29):
        for base in (1.0, 3.7, 1e-200, 1e200):
            small = base * 2.0 ** -k
            for s in (small, math.nextafter(small, 0.0), math.nextafter(small, math.inf)):
                pairs.append((base, s))
                pairs.append((-s, base))
            big = base * 2.0 ** k
            pairs.append((base, big))
            pairs.append((big, -base))
    for x in (0.7, 1e-310, 3e-160, 2.5e155, 1e307, 123456.789):
        pairs.append((x, x))
        pairs.append((x, -x))
        pairs.append((x, math.nextafter(x, math.inf)))
    a = [p[0] for p in pairs]
    b = [p[1] for p in pairs]
    h = [math.hypot(x, y) for x, y in pairs]

    n = 1_000_000
    rng = Rng.stream(0, 320000)
    sha = hashlib.sha256()
    pack = struct.Struct("<d").pack
    ra, rb, rh = [], [], []
    differs = 0
    for i in range(n):
        x = rng.range(-10.0, 10.0)
        y = rng.range(-10.0, 10.0)
        r = math.hypot(x, y)
        if r != math.sqrt(x * x + y * y):
            differs += 1
        sha.update(pack(r))
        if i < 64:
            ra.append(x)
            rb.append(y)
            rh.append(r)
    _lemit("localize_py_hypot",
           {"n_edge": len(pairs), "edge_values": ",".join(repr(v) for v in vals), "n_random": n,
            "rng_seed": 0, "rng_stream": 320000, "range_lo": -10.0, "range_hi": 10.0,
            "sha256_random": sha.hexdigest(), "n_random_not_sqrt": differs},
           {"a": a, "b": b, "h": h, "rand_a": ra, "rand_b": rb, "rand_h": rh},
           {"a": "input", "b": "input", "h": "bits", "rand_a": "input", "rand_b": "input", "rand_h": "bits"})


#: localize_perimeter's perimeter_nodes cases: (W, H, n, inset, offset).
LOCALIZE_PERIMETER_CASES = (
    (8.0, 6.0, 8, 0.1, 0.0), (2.0, 10.0, 6, 0.1, 0.0), (8.0, 6.0, 7, 0.1, 1.7), (8.0, 6.0, 5, 0.0, -3.3),
    (4.2, 3.6, 4, 0.15, 0.0), (8.0, 6.0, 12, 0.1, 0.0), (8.0, 6.0, 3, 0.1, -1e-17),
)
#: localize_perimeter's PixelGrid cases: (W, H, pitch).
LOCALIZE_GRID_CASES = (
    (8.0, 6.0, 0.5), (8.0, 6.0, 0.25), (8.0, 6.0, 0.7), (5.0, 5.0, 2.0), (7.0, 7.0, 2.0), (4.2, 3.6, 0.5),
    (6.0, 5.0, 0.5), (9.0, 7.0, 0.5), (8.0, 6.0, math.inf), (3.0, 2.0, 100.0), (2.5, 0.75, 0.5),
    (2.5, 1.25, 0.5),
)


def gen_localize_perimeter() -> None:
    """geometry.py's Room, perimeter_nodes, links, PixelGrid, nearest_index
    and ellipse_weight (spec/locate.md §3).

    perimeter_nodes cases (`pc_*`, `input`): `pc_room_w`, `pc_room_h`,
    `pc_n`, `pc_inset`, `pc_offset` (case 6's offset -1e-17 lands on the
    perimeter itself under Python's float %). Nodes, case-major (`pc_n`
    each, walk order): `node_id` (exact), `node_x`, `node_y`. Links of
    `Geometry(room, those nodes)` (every ordered pair, canonical order),
    case-major (`pc_nlinks` each): `link_tx`, `link_rx` (exact),
    `link_len` (Node.distance_to: CPython hypot), `link_mid_x`,
    `link_mid_y`.

    PixelGrid cases (`g_room_w`, `g_room_h`, `g_pitch`, `input`; the
    half-even pins: 5x5 at 2.0 gives n = 2, 7x7 at 2.0 gives n = 4,
    0.75/0.5 gives n_y = 2 and 1.25/0.5 gives n_y = 2; 0.7 on 8x6 is the
    anisotropic 11x9; an infinite pitch gives 1x1): `g_n_x`, `g_n_y`,
    `g_n_pixels` (exact), `g_pitch_x`, `g_pitch_y`, `g_pixel_area`;
    pixels grid-major, row-major y-outer (`g_n_pixels` each): `pixel_x`,
    `pixel_y`.

    nearest_index cases: `ni_grid` (index into the grid cases), `ni_x`,
    `ni_y` (input) -> `ni_index` (exact).

    ellipse_weight / excess_path_m cases: link `ew_ax, ew_ay -> ew_bx,
    ew_by` (tx, rx; `Link.length_m` by Node.distance_to, except the
    zero-length case 3, built with length 0.0), point `ew_px`, `ew_py`,
    `ew_lambda` (input) -> `ew_weight`, `ew_excess` (rule:each). Case 0/1
    is the strict-boundary pin: excess exactly 1.0 against lambda 1.0
    (weight 0) and nextafter(1.0, 2.0) (weight 0.5).

    Error cases (`err_code`, exact), in order: see the `err.<i>` scalars.
    """
    from wifisense.localize.geometry import Geometry, Link, Node, PixelGrid, Room, ellipse_weight, perimeter_nodes

    arr: dict = {k: [] for k in ("pc_room_w", "pc_room_h", "pc_n", "pc_inset", "pc_offset", "node_id", "node_x",
                                 "node_y", "pc_nlinks", "link_tx", "link_rx", "link_len", "link_mid_x",
                                 "link_mid_y")}
    for W, H, n, inset, off in LOCALIZE_PERIMETER_CASES:
        room = Room(W, H)
        nodes = perimeter_nodes(room, n, inset, off)
        for k, v in (("pc_room_w", W), ("pc_room_h", H), ("pc_n", n), ("pc_inset", inset), ("pc_offset", off)):
            arr[k].append(v)
        for nd in nodes:
            arr["node_id"].append(nd.node_id)
            arr["node_x"].append(nd.x)
            arr["node_y"].append(nd.y)
        geom = Geometry(room, nodes, PixelGrid(room, 0.5), 0.3)
        arr["pc_nlinks"].append(len(geom.links))
        for lk in geom.links:
            arr["link_tx"].append(lk.tx_id)
            arr["link_rx"].append(lk.rx_id)
            arr["link_len"].append(lk.length_m)
            arr["link_mid_x"].append(lk.midpoint[0])
            arr["link_mid_y"].append(lk.midpoint[1])
    for k in ("g_room_w", "g_room_h", "g_pitch", "g_n_x", "g_n_y", "g_n_pixels", "g_pitch_x", "g_pitch_y",
              "g_pixel_area", "pixel_x", "pixel_y"):
        arr[k] = []
    grids = []
    for W, H, pitch in LOCALIZE_GRID_CASES:
        g = PixelGrid(Room(W, H), pitch)
        grids.append(g)
        for k, v in (("g_room_w", W), ("g_room_h", H), ("g_pitch", pitch), ("g_n_x", g.n_x), ("g_n_y", g.n_y),
                     ("g_n_pixels", len(g)), ("g_pitch_x", g.pitch_x_m), ("g_pitch_y", g.pitch_y_m),
                     ("g_pixel_area", g.pixel_area_m2)):
            arr[k].append(v)
        for px, py in g.pixels:
            arr["pixel_x"].append(px)
            arr["pixel_y"].append(py)
    pts = ((0.0, 0.0), (8.0, 6.0), (-1.0, -1.0), (100.0, 3.0), (3.99, 2.01), (4.0, 3.0),
           (0.7272727272727273, 0.6666666666666666), (7.999999, 5.999999), (-0.3, 2.2), (0.5, 0.49999999999999994))
    for k in ("ni_grid", "ni_x", "ni_y", "ni_index"):
        arr[k] = []
    for gi in (0, 2, 8):
        for x, y in pts:
            arr["ni_grid"].append(gi)
            arr["ni_x"].append(x)
            arr["ni_y"].append(y)
            arr["ni_index"].append(grids[gi].nearest_index(x, y))
    ew = [((0.0, 0.0), (4.0, 0.0), (-0.5, 0.0), 1.0), ((0.0, 0.0), (4.0, 0.0), (-0.5, 0.0), math.nextafter(1.0, 2.0)),
          ((0.0, 0.0), (4.0, 0.0), (2.0, 0.0), 0.001), ((1.0, 1.0), (1.0, 1.0), (1.0, 1.0), 0.3),
          ((0.1, 0.1), (7.9, 5.9), (4.0, 3.0), 0.3), ((0.1, 0.1), (7.9, 5.9), (4.2, 2.7), 0.3),
          ((0.0, 0.0), (4.0, 0.0), (2.0, 1.5), 1.0), ((0.0, 0.0), (4.0, 0.0), (2.0, 1.4999999), 1.0)]
    for k in ("ew_ax", "ew_ay", "ew_bx", "ew_by", "ew_px", "ew_py", "ew_lambda", "ew_weight", "ew_excess"):
        arr[k] = []
    for i, (A, B, P, lam) in enumerate(ew):
        length = 0.0 if i == 3 else Node(0, A[0], A[1]).distance_to(B[0], B[1])
        lk = Link(0, 1, A, B, length)
        for k, v in (("ew_ax", A[0]), ("ew_ay", A[1]), ("ew_bx", B[0]), ("ew_by", B[1]), ("ew_px", P[0]),
                     ("ew_py", P[1]), ("ew_lambda", lam), ("ew_weight", ellipse_weight(lk, P[0], P[1], lam)),
                     ("ew_excess", lk.excess_path_m(P[0], P[1]))):
            arr[k].append(v)
    room = Room(8.0, 6.0)
    g8 = Geometry.perimeter(8)
    two = [Node(0, 1.0, 1.0)]
    errs = [
        ("Room(0.0, 6.0)", lambda: Room(0.0, 6.0)),
        ("Room(8.0, nan)", lambda: Room(8.0, math.nan)),
        ("perimeter_nodes(Room(8,6), 1)", lambda: perimeter_nodes(room, 1)),
        ("perimeter_nodes(Room(8,6), 4, inset_m=-0.1)", lambda: perimeter_nodes(room, 4, -0.1)),
        ("perimeter_nodes(Room(8,6), 4, inset_m=3.0)", lambda: perimeter_nodes(room, 4, 3.0)),
        ("PixelGrid(Room(8,6), nan)", lambda: PixelGrid(room, math.nan)),
        ("PixelGrid(Room(8,6), 0.0)", lambda: PixelGrid(room, 0.0)),
        ("PixelGrid(Room(8,6), -0.5)", lambda: PixelGrid(room, -0.5)),
        ("PixelGrid(Room(8,6), 0.5).pixel_index(16, 0)", lambda: PixelGrid(room, 0.5).pixel_index(16, 0)),
        ("PixelGrid(Room(8,6), 0.5).pixel_index(-1, 0)", lambda: PixelGrid(room, 0.5).pixel_index(-1, 0)),
        ("PixelGrid(Room(8,6), 0.5).nearest_index(nan, 1.0)",
         lambda: PixelGrid(room, 0.5).nearest_index(math.nan, 1.0)),
        ("Geometry(Room(8,6), [Node(0, 1, 1)])", lambda: Geometry(room, two)),
        ("Geometry(Room(8,6), nodes 0 at (1,1) and 0 at (2,2))",
         lambda: Geometry(room, [Node(0, 1.0, 1.0), Node(0, 2.0, 2.0)])),
        ("Geometry(Room(8,6), nodes 0 and 1 both at (1,1))",
         lambda: Geometry(room, [Node(0, 1.0, 1.0), Node(1, 1.0, 1.0)])),
        ("Geometry(Room(8,6), nodes 0 (1,1), 1 (2,2), lambda_m=0.0)",
         lambda: Geometry(room, [Node(0, 1.0, 1.0), Node(1, 2.0, 2.0)], lambda_m=0.0)),
        ("Geometry(Room(8,6), nodes 0 (1,1), 1 (2,2), lambda_m=nan)",
         lambda: Geometry(room, [Node(0, 1.0, 1.0), Node(1, 2.0, 2.0)], lambda_m=math.nan)),
        ("Geometry.perimeter(8).link_index(0, 0)", lambda: g8.link_index(0, 0)),
        ("Geometry.perimeter(8).node(99)", lambda: g8.node(99)),
        ("ellipse_weight(link (0,0)-(4,0), 1.0, 1.0, 0.0)",
         lambda: ellipse_weight(Link(0, 1, (0.0, 0.0), (4.0, 0.0), 4.0), 1.0, 1.0, 0.0)),
    ]
    esc, codes = _err_cases(errs)
    arr["err_code"] = codes
    cmp = {k: "input" for k in ("pc_room_w", "pc_room_h", "pc_n", "pc_inset", "pc_offset", "g_room_w", "g_room_h",
                                "g_pitch", "ni_grid", "ni_x", "ni_y", "ew_ax", "ew_ay", "ew_bx", "ew_by", "ew_px",
                                "ew_py", "ew_lambda")}
    cmp.update({k: "exact" for k in ("node_id", "pc_nlinks", "link_tx", "link_rx", "g_n_x", "g_n_y", "g_n_pixels",
                                     "ni_index", "err_code")})
    cmp.update({"node_x": "rule", "node_y": "rule", "link_len": "rule", "link_mid_x": "rule",
                "link_mid_y": "rule", "g_pitch_x": "rule:each", "g_pitch_y": "rule:each",
                "g_pixel_area": "rule:each", "pixel_x": "rule:ptr:g_n_pixels", "pixel_y": "rule:ptr:g_n_pixels",
                "ew_weight": "rule:each", "ew_excess": "rule:each"})
    sc = {"n_perimeter_cases": len(LOCALIZE_PERIMETER_CASES), "n_grid_cases": len(LOCALIZE_GRID_CASES),
          "n_ni": len(arr["ni_index"]), "n_ew": len(ew)}
    sc.update(esc)
    _lemit("localize_perimeter", sc, arr, cmp)


def _min_boundary_margin(geom) -> float:
    """The smallest |d_tx + d_rx - d_l - lambda| over every link with
    d_l > 0 and every pixel (d by CPython hypot, as geometry.py's _build):
    how far the nearest pixel is from flipping its support membership."""
    lam = geom.lambda_m
    dist = {n.node_id: [math.hypot(n.x - px, n.y - py) for px, py in geom.grid.pixels] for n in geom.nodes}
    margin = math.inf
    for lk in geom.links:
        if not lk.length_m > 0.0:
            continue
        dt, dr = dist[lk.tx_id], dist[lk.rx_id]
        for p in range(geom.n_pixels):
            m = abs(dt[p] + dr[p] - lk.length_m - lam)
            if m < margin:
                margin = m
    return margin


def _weights_arrays(geom, prefix: str = "") -> tuple[dict, dict, dict]:
    """(scalars, arrays, cmp) of one Geometry for the localize_weights sets."""
    lam = geom.lambda_m
    sup = geom.supports()
    a: dict = {"node_id": [n.node_id for n in geom.nodes], "node_x": [n.x for n in geom.nodes],
               "node_y": [n.y for n in geom.nodes],
               "link_tx": [lk.tx_id for lk in geom.links], "link_rx": [lk.rx_id for lk in geom.links],
               "link_len": [lk.length_m for lk in geom.links],
               "link_w": [(1.0 / math.sqrt(lk.length_m)) if lk.length_m > 0.0 else 0.0 for lk in geom.links],
               "support_len": [len(s) for s in sup], "support_idx": [p for s in sup for p in s],
               "pixel_link_counts": list(geom.pixel_link_counts())}
    a["blind_pixels"] = [p for p, c in enumerate(a["pixel_link_counts"]) if c == 0]
    margin = _min_boundary_margin(geom)
    w = geom.weight_matrix()
    for li, s in enumerate(sup):
        nz = [p for p, v in enumerate(w[li]) if v != 0.0]
        assert nz == list(s), "dense W agrees with the supports"
    g = geom.grid
    sc = {"room_w": g.room.width_m, "room_h": g.room.height_m, "pitch_m": g.pitch_m, "lambda_m": lam,
          "n_x": g.n_x, "n_y": g.n_y, "n_nodes": geom.n_nodes, "n_links": geom.n_links,
          "n_pixels": geom.n_pixels, "nnz": len(a["support_idx"]), "n_blind": len(a["blind_pixels"]),
          "min_boundary_margin": margin}
    cmp = {"node_id": "input", "node_x": "input", "node_y": "input", "link_tx": "exact", "link_rx": "exact",
           "link_len": "rule", "link_w": "rule", "support_len": "exact", "support_idx": "exact",
           "pixel_link_counts": "exact", "blind_pixels": "exact"}
    return ({prefix + k: v for k, v in sc.items()}, {prefix + k: v for k, v in a.items()},
            {prefix + k: v for k, v in cmp.items()})


def gen_localize_weights() -> None:
    """geometry.py's weight model (Wilson & Patwari ellipse, strict `<`) on
    four geometries, `variants` "default,aniso,edge,edgeup":

    * `default`: Geometry.perimeter(8) on Room(8, 6), pitch 0.5, lambda 0.3;
    * `aniso`: Geometry.perimeter(8), pitch 0.7 (the 11x9 grid), lambda 0.6;
    * `edge`, `edgeup`: Room(3, 1), nodes 0 (0.25, 0.25), 1 (2.25, 0.25),
      pitch 0.5, lambda 1.0 and nextafter(1.0, 2.0): pixel (2.75, 0.25) has
      excess exactly 1.0, outside the first support and inside the second
      (their `min_boundary_margin` is 0 on purpose).

    Per variant V (prefix `V_`): scalars `room_w`, `room_h`, `pitch_m`,
    `lambda_m`, `n_x`, `n_y`, `n_nodes`, `n_links`, `n_pixels`, `nnz`,
    `n_blind`, `min_boundary_margin` (the smallest
    |d_tx + d_rx - d_l - lambda| over every link with d_l > 0 and every
    pixel, d by CPython hypot: how far the nearest pixel is from flipping;
    a mismatch in a support with a margin of a few ulp is a hypot
    difference, not a logic error). Arrays: `node_id`, `node_x`, `node_y`
    (input); per link, canonical order: `link_tx`, `link_rx` (exact),
    `link_len`, `link_w` (1/sqrt(d_l), 0 for d_l <= 0); the supports as
    CSR: `support_len` per link and `support_idx` (ascending pixel indices,
    link-major); `pixel_link_counts` per pixel; `blind_pixels` (pixels no
    link covers, ascending).
    """
    from wifisense.localize.geometry import Geometry, Node, PixelGrid, Room

    room = Room(8.0, 6.0)
    geoms = [("default", Geometry.perimeter(8, room, 0.5, 0.3)),
             ("aniso", Geometry.perimeter(8, room, 0.7, 0.6))]
    r31 = Room(3.0, 1.0)
    nodes = [Node(0, 0.25, 0.25), Node(1, 2.25, 0.25)]
    geoms.append(("edge", Geometry(r31, nodes, PixelGrid(r31, 0.5), 1.0)))
    geoms.append(("edgeup", Geometry(r31, nodes, PixelGrid(r31, 0.5), math.nextafter(1.0, 2.0))))
    sc: dict = {"n_variants": len(geoms), "variants": ",".join(v for v, _g in geoms)}
    arr: dict = {}
    cmp: dict = {}
    for v, g in geoms:
        s, a, c = _weights_arrays(g, v + "_")
        sc.update(s)
        arr.update(a)
        cmp.update(c)
    _lemit("localize_weights", sc, arr, cmp)


def _gen_localize_weights_scene(name: str, rel: str) -> None:
    from wifisense.localize.config import LocateConfig
    from wifisense.localize.pipeline import locate_geometry
    from wifisense.roomsim import load_scene

    geom = locate_geometry(load_scene(os.path.join(LOCATE_REPO, rel)), LocateConfig())
    s, a, c = _weights_arrays(geom)
    sc = _scene_scalars(rel)
    sc.update(s)
    _lemit(name, sc, a, c)


def gen_localize_weights_bedroom() -> None:
    """The bedroom's nodes (spec/scenes/bedroom.json) as Locate builds them
    (spec/locate.md §2: Room(room.w_m, room.d_m), z dropped, pitch 0.5,
    lambda 0.3); layout as one localize_weights variant, unprefixed."""
    _gen_localize_weights_scene("localize_weights_bedroom", "spec/scenes/bedroom.json")


def gen_localize_weights_office() -> None:
    """The office's 6 nodes (spec/scenes/office.json); layout as
    localize_weights_bedroom."""
    _gen_localize_weights_scene("localize_weights_office", "spec/scenes/office.json")


def gen_localize_weights_hall() -> None:
    """The L-shaped hall's 8 nodes on its 9 x 7 bounding rectangle
    (spec/scenes/hall-l-shaped.json, tiled whole, §13); layout as
    localize_weights_bedroom."""
    _gen_localize_weights_scene("localize_weights_hall", "spec/scenes/hall-l-shaped.json")


def gen_localize_prior() -> None:
    """invert.py's SpatialPrior (the Toeplitz exponential kernel, dix with
    pitch_x and diy with pitch_y), `variants` "aniso,default":

    * `aniso`: PixelGrid(Room(8, 6), 0.7) (11 x 9, pitch 0.7273 x 0.6667),
      sigma 1.7, delta_c 0.65;
    * `default`: PixelGrid(Room(8, 6), 0.5) (16 x 12), sigma 1.0, delta_c 0.5.

    Per variant V: scalars `n_x`, `n_y`, `pitch_x`, `pitch_y`, `sigma`,
    `delta_c`, `sigma2`; `kern` (n_x * n_y, index dix * n_y + diy:
    sigma2 * exp(-hypot(dix * pitch_x, diy * pitch_y) / delta_c), from
    integer offsets); `row_p` (input) and `rows` (row(p) for each, P each,
    row-major); `entry_p`, `entry_q` (input) -> `entry`.
    Error cases (`err_code`): see `err.<i>`.
    """
    from wifisense.localize.geometry import PixelGrid, Room
    from wifisense.localize.invert import SpatialPrior

    room = Room(8.0, 6.0)
    sc: dict = {}
    arr: dict = {}
    cmp: dict = {}
    variants = (("aniso", PixelGrid(room, 0.7), 1.7, 0.65, (0, 50, 98)),
                ("default", PixelGrid(room, 0.5), 1.0, 0.5, (0, 95, 191)))
    for v, g, sigma, dc, rows in variants:
        pr = SpatialPrior(g, sigma, dc)
        n = g.n_x * g.n_y
        kern = [pr.entry(0, diy * g.n_x + dix) for dix in range(g.n_x) for diy in range(g.n_y)]
        ep = [0, 0, 5, n - 1, 17, n // 2, 3]
        eq = [0, n - 1, 5 + g.n_x, 0, 17 + 2 * g.n_x + 3, n // 2 - g.n_x - 1, 3 + g.n_x * (g.n_y - 1)]
        a = {"kern": kern, "row_p": list(rows), "rows": [x for p in rows for x in pr.row(p)],
             "entry_p": ep, "entry_q": eq, "entry": [pr.entry(p, q) for p, q in zip(ep, eq)]}
        sc.update({f"{v}_{k}": x for k, x in (("n_x", g.n_x), ("n_y", g.n_y), ("pitch_x", g.pitch_x_m),
                                               ("pitch_y", g.pitch_y_m), ("sigma", sigma), ("delta_c", dc),
                                               ("sigma2", pr.sigma2))})
        arr.update({f"{v}_{k}": x for k, x in a.items()})
        cmp.update({f"{v}_kern": "rule", f"{v}_row_p": "input", f"{v}_rows": f"rule:{n}",
                    f"{v}_entry_p": "input", f"{v}_entry_q": "input", f"{v}_entry": "rule"})
    g = PixelGrid(room, 0.5)
    pr = SpatialPrior(g)
    errs = [
        ("SpatialPrior(16x12 grid, sigma=0.0)", lambda: SpatialPrior(g, 0.0)),
        ("SpatialPrior(16x12 grid, sigma=nan)", lambda: SpatialPrior(g, math.nan)),
        ("SpatialPrior(16x12 grid, sigma=inf)", lambda: SpatialPrior(g, math.inf)),
        ("SpatialPrior(16x12 grid, sigma=1e200)", lambda: SpatialPrior(g, 1e200)),
        ("SpatialPrior(16x12 grid, delta_c=0.0)", lambda: SpatialPrior(g, 1.0, 0.0)),
        ("SpatialPrior(16x12 grid, delta_c=-1.0)", lambda: SpatialPrior(g, 1.0, -1.0)),
        ("SpatialPrior(16x12 grid, delta_c=inf)", lambda: SpatialPrior(g, 1.0, math.inf)),
        ("SpatialPrior(16x12 grid).entry(192, 0)", lambda: pr.entry(192, 0)),
        ("SpatialPrior(16x12 grid).entry(0, -1)", lambda: pr.entry(0, -1)),
        ("SpatialPrior(16x12 grid).row(192)", lambda: pr.row(192)),
    ]
    esc, codes = _err_cases(errs)
    sc.update(esc)
    arr["err_code"] = codes
    cmp["err_code"] = "exact"
    sc.update({"n_variants": 2, "variants": "aniso,default"})
    _lemit("localize_prior", sc, arr, cmp)


def gen_localize_cholesky() -> None:
    """invert.py's cholesky_factor, cholesky_solve, cholesky_inverse.

    Cases, in order (sizes `m_n`, `m_nn` = n*n, exact): 0 the hand 3x3
    [[4,2,0],[2,5,2],[0,2,5]] (inverse exactly dyadic) with b = [1,2,3];
    1 a recorded 12x12 SPD matrix (B Bᵀ + 0.5 I, B from Rng.stream(0,
    330001).range(-1, 1)) with a recorded b; 2 [[4]]; 3 a matrix
    asymmetric by 1e-12 (inside symmetry_tol: accepted); 4 the asymmetric
    [[1e9, 1], [2, 1e9]]; 5 a NaN at a[0][1]; 6 an inf on the diagonal;
    7 the indefinite [[1, 2], [2, 1]] (pivot 1); 8 diag(1, 1, -1) (pivot
    2); 9 the 2x2 zero matrix; 10 the empty matrix; 11 [[-1]] (pivot 0).

    Inputs (row-major, case-major): `m_a` (`m_nn` each), `m_b` (`m_n`
    each). Outputs: `m_code` (EXC_CODE of cholesky_factor: 0 ok, 4
    malformed, 5 asymmetric, 6 not positive definite), `m_pivot` (the
    failing pivot index the NotPositiveDefiniteError names, -1 if none
    or another error); for code 0, `m_g` (the lower factor G, n x n with
    zeros above the diagonal), `m_x` = cholesky_solve(A, b), `m_inv` =
    cholesky_inverse(A) (exactly symmetric); zeros for a failed case.
    `err_code`: cholesky_solve_factored with a wrong-length b (`err.<i>`).
    """
    import re

    from wifisense.localize.invert import (
        NotPositiveDefiniteError, cholesky_factor, cholesky_inverse, cholesky_solve, cholesky_solve_factored,
    )

    rnd = _rng_draws(330001, 12 * 12 + 12, -1.0, 1.0)
    B = [rnd[i * 12:(i + 1) * 12] for i in range(12)]
    spd = []
    for i in range(12):
        row = []
        for j in range(12):
            acc = 0.0
            for k in range(12):
                acc = acc + B[i][k] * B[j][k]
            row.append(acc + (0.5 if i == j else 0.0))
        spd.append(row)
    cases = [
        ([[4.0, 2.0, 0.0], [2.0, 5.0, 2.0], [0.0, 2.0, 5.0]], [1.0, 2.0, 3.0]),
        (spd, rnd[144:]),
        ([[4.0]], [2.0]),
        ([[2.0, 1.0 + 1e-12], [1.0, 3.0]], [1.0, -1.0]),
        ([[1e9, 1.0], [2.0, 1e9]], [1.0, 1.0]),
        ([[1.0, math.nan], [0.0, 1.0]], [1.0, 1.0]),
        ([[math.inf, 0.0], [0.0, 1.0]], [1.0, 1.0]),
        ([[1.0, 2.0], [2.0, 1.0]], [1.0, 1.0]),
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -1.0]], [1.0, 1.0, 1.0]),
        ([[0.0, 0.0], [0.0, 0.0]], [1.0, 1.0]),
        ([], []),
        ([[-1.0]], [1.0]),
    ]
    arr: dict = {k: [] for k in ("m_n", "m_nn", "m_a", "m_b", "m_code", "m_pivot", "m_g", "m_x", "m_inv")}
    for A, b in cases:
        n = len(A)
        arr["m_n"].append(n)
        arr["m_nn"].append(n * n)
        arr["m_a"].extend(v for row in A for v in row)
        arr["m_b"].extend(b)
        pivot = -1
        try:
            g = cholesky_factor(A)
            code = 0
        except NotPositiveDefiniteError as exc:
            code = 6
            m = re.search(r"pivot (\d+) of", str(exc))
            pivot = int(m.group(1)) if m else -1
        except Exception:
            code = _exc_code(lambda: cholesky_factor(A))
        arr["m_code"].append(code)
        arr["m_pivot"].append(pivot)
        if code == 0:
            arr["m_g"].extend(v for row in g for v in row)
            arr["m_x"].extend(cholesky_solve(A, b))
            arr["m_inv"].extend(v for row in cholesky_inverse(A) for v in row)
        else:
            arr["m_g"].extend([0.0] * (n * n))
            arr["m_x"].extend([0.0] * n)
            arr["m_inv"].extend([0.0] * (n * n))
    g0 = cholesky_factor(cases[0][0])
    esc, codes = _err_cases([
        ("cholesky_solve_factored(G of case 0, [1.0, 2.0])", lambda: cholesky_solve_factored(g0, [1.0, 2.0])),
    ])
    arr["err_code"] = codes
    sc = {"n_cases": len(cases), "symmetry_tol": 1e-9}
    sc.update(esc)
    _lemit("localize_cholesky", sc, arr,
           {"m_n": "input", "m_nn": "input", "m_a": "input", "m_b": "input", "m_code": "exact", "m_pivot": "exact",
            "m_g": "rule:ptr:m_nn", "m_x": "rule:ptr:m_n", "m_inv": "rule:ptr:m_nn", "err_code": "exact"})


def _blob(geom, cx: float, cy: float, width: float = 0.4, amp: float = 1.0) -> list[float]:
    """A Gaussian blob image (an input generator; the images are recorded)."""
    return [amp * math.exp(-((px - cx) * (px - cx) + (py - cy) * (py - cy)) / (2.0 * width * width))
            for px, py in geom.grid.pixels]


def _inverter_set(name: str, geom, alpha: float, sigma: float, delta_c: float, ys: list, doc_cases: str) -> None:
    import warnings

    from wifisense.localize.invert import Inverter

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inv = Inverter(geom, alpha, sigma, delta_c)
    L, P, G = geom.n_links, geom.n_pixels, inv.n_independent_links
    xs = [inv.solve(y) for y in ys]
    arr = {"node_id": [n.node_id for n in geom.nodes], "node_x": [n.x for n in geom.nodes],
           "node_y": [n.y for n in geom.nodes], "blind_pixels": list(inv.blind_pixels),
           "group_len": [len(g) for g in inv.link_groups], "group_idx": [i for g in inv.link_groups for i in g],
           "group_scale": list(inv._group_scale),
           "a_plus_alpha": [v for row in inv._a_plus_alpha for v in row],
           "projection": [v for row in inv.projection for v in row],
           "y": [v for y in ys for v in y], "x": [v for x in xs for v in x]}
    sc = {"room_w": geom.room.width_m, "room_h": geom.room.height_m, "pitch_m": geom.grid.pitch_m,
          "lambda_m": geom.lambda_m, "n_x": geom.grid.n_x, "n_y": geom.grid.n_y, "alpha": alpha, "sigma": sigma,
          "delta_c": delta_c, "residual_tol": 1e-6, "n_links": L, "n_pixels": P, "n_independent_links": G,
          "n_blind": len(inv.blind_pixels), "asymmetry": inv.asymmetry, "mean_diag_a": inv.mean_diag_a,
          "min_diag_a": inv.min_diag_a, "inverse_residual": inv.inverse_residual,
          "projection_slots": inv.projection_slots, "projection_bytes": inv.projection_bytes,
          "n_cases": len(ys), "cases": doc_cases, "min_boundary_margin": _min_boundary_margin(geom)}
    cmp = {"node_id": "input", "node_x": "input", "node_y": "input", "blind_pixels": "exact", "group_len": "exact",
           "group_idx": "exact", "group_scale": "rule", "a_plus_alpha": f"rule:{G}", "projection": f"rule:{G}",
           "y": "input", "x": f"rule:{P}", "asymmetry": "le:1e-07", "inverse_residual": "le:1e-06"}
    _lemit(name, sc, arr, cmp)


def _inverter_cases(geom, inv_seed: int) -> tuple[list, str]:
    """Recorded y vectors: forward-projected blobs, two blobs, zero, a
    noisy blob, pure noise, an antisymmetric y and mixed signs."""
    from wifisense.localize.invert import forward_project

    W, H = geom.room.width_m, geom.room.height_m
    L = geom.n_links
    noise = _rng_gauss(inv_seed, 3 * L)
    ys, names = [], []
    for fx, fy in ((0.3, 0.35), (0.55, 0.6), (0.85, 0.2)):
        ys.append(forward_project(geom, _blob(geom, fx * W, fy * H)))
        names.append(f"blob({fx}W,{fy}H)")
    two = [a + b for a, b in zip(_blob(geom, 0.25 * W, 0.7 * H), _blob(geom, 0.7 * W, 0.3 * H, 0.5, 0.6))]
    ys.append(forward_project(geom, two))
    names.append("two blobs")
    ys.append([0.0] * L)
    names.append("zero")
    b = forward_project(geom, _blob(geom, 0.45 * W, 0.5 * H))
    ys.append([v + 0.05 * g for v, g in zip(b, noise[:L])])
    names.append("noisy blob")
    ys.append([0.1 * g for g in noise[L:2 * L]])
    names.append("noise")
    anti = [0.0] * L
    for i, lk in enumerate(geom.links):
        if lk.tx_id < lk.rx_id:
            anti[i] = noise[2 * L + i]
            anti[geom.link_index(lk.rx_id, lk.tx_id)] = -noise[2 * L + i]
    ys.append(anti)
    names.append("antisymmetric")
    ys.append([(-1.0) ** i * 0.3 * (i % 5) for i in range(L)])
    names.append("mixed signs")
    return ys, ";".join(names)


def gen_localize_inverter_default() -> None:
    """invert.py's Inverter on Geometry.perimeter(8) (Room(8, 6), pitch 0.5,
    lambda 0.3), alpha 1, sigma 1, delta_c 0.5, residual_tol 1e-6.

    Scalars: the geometry (`room_w`, `room_h`, `pitch_m`, `lambda_m`,
    `n_x`, `n_y`, `min_boundary_margin` as localize_weights), `alpha`,
    `sigma`, `delta_c`, `residual_tol`, `n_links`,
    `n_pixels`, `n_independent_links` (G, the rows after the duplicate-row
    collapse), `n_blind`, `mean_diag_a`, `min_diag_a`, `projection_slots`,
    `projection_bytes`, and the rounding-level diagnostics `asymmetry`
    (`le:1e-07`, the guard) and `inverse_residual` (`le:1e-06`,
    residual_tol); `n_cases`, `cases` (the case names, ';'-separated).
    Arrays: `node_id`, `node_x`, `node_y` (input: the geometry's nodes);
    `blind_pixels`; the groups as CSR, first-occurrence order:
    `group_len`, `group_idx` (link indices, ascending within a group);
    `group_scale` (1/sqrt(len)); `a_plus_alpha` (the symmetrised G x G
    matrix with the ridge, row-major, `Inverter._a_plus_alpha`);
    `projection` (P x G, pixel-major); solve cases: `y` (input, case-major,
    n_links each) and `x` = solve(y) (case-major, n_pixels each).
    """
    from wifisense.localize.geometry import Geometry, Room

    geom = Geometry.perimeter(8, Room(8.0, 6.0), 0.5, 0.3)
    ys, names = _inverter_cases(geom, 330002)
    _inverter_set("localize_inverter_default", geom, 1.0, 1.0, 0.5, ys, names)


def gen_localize_inverter_aniso() -> None:
    """The Inverter on the anisotropic grid: Geometry.perimeter(8) on
    Room(8, 6), pitch 0.7 (11 x 9), lambda 0.6, alpha 1, sigma 1.3,
    delta_c 0.6; layout as localize_inverter_default."""
    from wifisense.localize.geometry import Geometry, Room

    geom = Geometry.perimeter(8, Room(8.0, 6.0), 0.7, 0.6)
    ys, names = _inverter_cases(geom, 330003)
    _inverter_set("localize_inverter_aniso", geom, 1.0, 1.3, 0.6, ys, names)


def gen_localize_inverter_fine() -> None:
    """The Inverter at scale: Geometry.perimeter(12) on Room(8, 6), pitch
    0.25 (32 x 24 = 768 pixels, 132 links, 66 rows), lambda 0.3, alpha 1,
    sigma 1, delta_c 0.5; layout as localize_inverter_default."""
    from wifisense.localize.geometry import Geometry, Room

    geom = Geometry.perimeter(12, Room(8.0, 6.0), 0.25, 0.3)
    ys, names = _inverter_cases(geom, 330004)
    _inverter_set("localize_inverter_fine", geom, 1.0, 1.0, 0.5, ys, names)


def gen_localize_rankstarved() -> None:
    """The Inverter's refusals on a rank-starved geometry: 12 perimeter
    nodes on Room(8, 6), pitch 2.0 (4 x 3 = 12 pixels), lambda 0.3: 132
    links collapse to 51 rows (non-reciprocal links with identical rows
    merge too). Scalars: the geometry, `sigma`, `delta_c`, `n_links`,
    `n_pixels`, `n_independent_links`, `n_cases`, `min_boundary_margin`
    (as localize_weights).

    Groups: `group_len`, `group_idx` (CSR), `group_scale`. Construction
    cases (`c_*`), in order: alpha 1e-2, 1e-4, 1e-6, 1e-8 (accepted, the
    back-check residual growing); 1e-10 (IllConditionedError from the
    back-check); 1e-10 with residual_tol 1e-3 (accepted); 5e-324 with
    check_inverse false (IllConditionedError from the factorisation);
    1.0 with check_inverse false (accepted, residual None); 0.0, nan, inf
    (ValueError). Inputs `c_alpha`, `c_residual_tol`, `c_check` (0/1).
    Outputs `c_code` (EXC_CODE), `c_has_residual` (0 when None or
    refused), `c_residual` (`le:c_residual_tol`: a rounding-level
    quantity), `c_mean_diag_a`, `c_min_diag_a` (0.0 when refused),
    `c_asymmetry` (`le:1e-07`).
    """
    import warnings

    from wifisense.localize.geometry import Geometry, PixelGrid, Room, perimeter_nodes
    from wifisense.localize.invert import Inverter

    room = Room(8.0, 6.0)
    geom = Geometry(room, perimeter_nodes(room, 12, 0.1), PixelGrid(room, 2.0), 0.3)
    cases = [(1e-2, 1e-6, True), (1e-4, 1e-6, True), (1e-6, 1e-6, True), (1e-8, 1e-6, True), (1e-10, 1e-6, True),
             (1e-10, 1e-3, True), (5e-324, 1e-6, False), (1.0, 1e-6, False), (0.0, 1e-6, True),
             (math.nan, 1e-6, True), (math.inf, 1e-6, True)]
    arr: dict = {k: [] for k in ("c_alpha", "c_residual_tol", "c_check", "c_code", "c_has_residual", "c_residual",
                                 "c_mean_diag_a", "c_min_diag_a", "c_asymmetry")}
    first = None
    for alpha, tol, check in cases:
        made = []

        def build(alpha=alpha, tol=tol, check=check, made=made):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                made.append(Inverter(geom, alpha, residual_tol=tol, check_inverse=check))

        code = _exc_code(build)
        inv = made[0] if made else None
        if inv is not None and first is None:
            first = inv
        res = None if inv is None else inv.inverse_residual
        for k, v in (("c_alpha", alpha), ("c_residual_tol", tol), ("c_check", int(check)), ("c_code", code),
                     ("c_has_residual", int(res is not None)), ("c_residual", 0.0 if res is None else res),
                     ("c_mean_diag_a", 0.0 if inv is None else inv.mean_diag_a),
                     ("c_min_diag_a", 0.0 if inv is None else inv.min_diag_a),
                     ("c_asymmetry", 0.0 if inv is None else inv.asymmetry)):
            arr[k].append(v)
    arr["group_len"] = [len(g) for g in first.link_groups]
    arr["group_idx"] = [i for g in first.link_groups for i in g]
    arr["group_scale"] = list(first._group_scale)
    _lemit("localize_rankstarved",
           {"room_w": 8.0, "room_h": 6.0, "n_nodes": 12, "inset_m": 0.1, "pitch_m": 2.0, "lambda_m": 0.3,
            "sigma": 1.0, "delta_c": 0.5, "n_links": geom.n_links, "n_pixels": geom.n_pixels,
            "n_independent_links": first.n_independent_links, "n_cases": len(cases),
            "min_boundary_margin": _min_boundary_margin(geom)},
           arr,
           {"c_alpha": "input", "c_residual_tol": "input", "c_check": "input", "c_code": "exact",
            "c_has_residual": "exact", "c_residual": "le:c_residual_tol", "c_mean_diag_a": "rule:each",
            "c_min_diag_a": "rule:each", "c_asymmetry": "le:1e-07", "group_len": "exact", "group_idx": "exact",
            "group_scale": "rule"})


def gen_localize_forward() -> None:
    """invert.py's forward_project on Geometry.perimeter(6) (Room(8, 6),
    pitch 0.5, lambda 0.5; 30 links): y_l = (1/sqrt(d_l)) * sum(image over
    its support), Neumaier, all links in canonical order. Scalars: the
    geometry, `n_links`, `n_pixels`, `n_cases`, `min_boundary_margin` (as
    localize_weights).

    Nodes `node_x`, `node_y` (input, ids 0..5). `image` (input, 3 x 192: a
    blob plus recorded noise, recorded values of both signs, zeros) -> `y`
    (3 x 30, `rule:30`). `err_code`: a 191-pixel image (`err.<i>`).
    """
    from wifisense.localize.geometry import Geometry, Room
    from wifisense.localize.invert import forward_project

    geom = Geometry.perimeter(6, Room(8.0, 6.0), 0.5, 0.5)
    P = geom.n_pixels
    g = _rng_gauss(330005, 2 * P)
    imgs = [[v + 0.1 * e for v, e in zip(_blob(geom, 3.0, 2.0, 0.6), g[:P])], g[P:], [0.0] * P]
    esc, codes = _err_cases([("forward_project(geometry, 191 zeros)", lambda: forward_project(geom, [0.0] * (P - 1)))])
    sc = {"room_w": 8.0, "room_h": 6.0, "n_nodes": 6, "pitch_m": 0.5, "lambda_m": 0.5, "n_links": geom.n_links,
          "n_pixels": P, "n_cases": len(imgs), "min_boundary_margin": _min_boundary_margin(geom)}
    sc.update(esc)
    _lemit("localize_forward", sc,
           {"node_x": [n.x for n in geom.nodes], "node_y": [n.y for n in geom.nodes],
            "image": [v for im in imgs for v in im], "y": [v for im in imgs for v in forward_project(geom, im)],
            "err_code": codes},
           {"node_x": "input", "node_y": "input", "image": "input", "y": f"rule:{geom.n_links}", "err_code": "exact"})


def gen_localize_peak() -> None:
    """track.py's extract_peak on Geometry.perimeter(8) (16 x 12 pixels).

    Cases (`image` input, case-major, 192 each; `radius_m` input), in
    order: 0-3 inverted blobs (the default Inverter's solve of
    forward-projected blobs at (2.0, 2.0), (4.1, 3.3), (7.6, 5.6) near a
    corner, (0.6, 0.4)); 4 the solve of recorded noise; 5-10 case 4 with
    its k lowest pixels set to exactly 0.0, k = 23, 24 (the 0.125 tie
    threshold), 38, 143, 144 (the 0.75 free-fraction threshold), 145;
    11 case 0 with a NaN at pixel 7; 12 case 0 with +inf at pixel 3;
    13 all 0.5; 14 case 1 with pixels 40 and 150 both set to its maximum
    (the first wins); 15 case 0 negated; 16-18 cases 0, 1, 2 with radius
    0.5, 2.0, 0.25; 19 case 4 with 60 pixels pinned, alternately 0.0 and
    -0.0 (one tie run). Radius 1.0 unless stated.

    Outputs (PeakReading): `x`, `y`, `peak`, `background` (NaN in the
    non-finite cases), `spread`, `z`, `peak_index` (-1 for None),
    `n_nonfinite`, `tie_fraction`. `err_code` (`err.<i>`).
    """
    import warnings

    from wifisense.localize.geometry import Geometry
    from wifisense.localize.invert import Inverter, forward_project
    from wifisense.localize.track import extract_peak

    geom = Geometry.perimeter(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inv = Inverter(geom)
    P = geom.n_pixels
    base = [inv.solve(forward_project(geom, _blob(geom, cx, cy))) for cx, cy in
            ((2.0, 2.0), (4.1, 3.3), (7.6, 5.6), (0.6, 0.4))]
    noise = inv.solve([0.1 * g for g in _rng_gauss(330006, geom.n_links)])
    cases = [(im, 1.0) for im in base] + [(noise, 1.0)]
    order = sorted(range(P), key=lambda p: (noise[p], p))
    for k in (23, 24, 38, 143, 144, 145):
        im = list(noise)
        for p in order[:k]:
            im[p] = 0.0
        cases.append((im, 1.0))
    im = list(base[0])
    im[7] = math.nan
    cases.append((im, 1.0))
    im = list(base[0])
    im[3] = math.inf
    cases.append((im, 1.0))
    cases.append(([0.5] * P, 1.0))
    im = list(base[1])
    top = max(im)
    im[40] = top
    im[150] = top
    cases.append((im, 1.0))
    cases.append(([-v for v in base[0]], 1.0))
    cases += [(base[0], 0.5), (base[1], 2.0), (base[2], 0.25)]
    im = list(noise)
    for i, p in enumerate(order[:60]):
        im[p] = 0.0 if i % 2 == 0 else -0.0
    cases.append((im, 1.0))
    arr: dict = {k: [] for k in ("image", "radius_m", "x", "y", "peak", "background", "spread", "z", "peak_index",
                                 "n_nonfinite", "tie_fraction")}
    for im, r in cases:
        pk = extract_peak(geom, im, r)
        arr["image"].extend(im)
        arr["radius_m"].append(r)
        for k in ("x", "y", "peak", "background", "spread", "z", "n_nonfinite", "tie_fraction"):
            arr[k].append(getattr(pk, k))
        arr["peak_index"].append(-1 if pk.peak_index is None else pk.peak_index)
    esc, codes = _err_cases([
        ("extract_peak(geometry, 191 zeros)", lambda: extract_peak(geom, [0.0] * (P - 1))),
        ("extract_peak(geometry, case 0, radius_m=0.0)", lambda: extract_peak(geom, base[0], 0.0)),
        ("extract_peak(geometry, case 0, radius_m=nan)", lambda: extract_peak(geom, base[0], math.nan)),
    ])
    arr["err_code"] = codes
    sc = {"n_cases": len(cases), "n_pixels": P, "n_x": geom.grid.n_x, "n_y": geom.grid.n_y}
    sc.update(esc)
    _lemit("localize_peak", sc, arr,
           {"image": "input", "radius_m": "input", "x": "rule", "y": "rule", "peak": "rule:each",
            "background": "rule:each", "spread": "rule:each", "z": "rule:each", "peak_index": "exact",
            "n_nonfinite": "exact", "tie_fraction": "exact", "err_code": "exact"})


#: localize_tracker's two trackers: (prefix, Tracker keyword arguments).
LOCALIZE_TRACKERS = (
    ("a", {}),
    ("b", dict(z_threshold=4.0, centroid_radius_m=0.75, accel_psd=0.5, meas_noise_m=0.3, init_speed_std_ms=0.5,
               max_coast_s=1.0, clamp_to_room=False)),
)


def gen_localize_tracker() -> None:
    """track.py's Tracker over a recorded image sequence on
    Geometry.perimeter(8) (16 x 12 pixels), two trackers, `variants` "a,b":
    `a` the defaults; `b` z_threshold 4.0, centroid_radius_m 0.75,
    accel_psd 0.5, meas_noise_m 0.3, init_speed_std_ms 0.5, max_coast_s
    1.0, clamp_to_room false.

    Inputs: `t` (step times; one repeated time, one step backwards, a 4 s
    gap) and `image` (step-major, 192 each: inverted noisy blobs moving
    across the room, noise-only steps, one all-NaN step, a blob at the wall
    edge). Per tracker V, per step: `V_est` (1 if update returned an
    estimate), `V_x`, `V_y`, `V_sigma_m`, `V_confidence`, `V_cov_xx`,
    `V_cov_xy`, `V_cov_yy` (PositionEstimate; 0.0 when None);
    `V_tracking` (is_tracking after the step) and, when tracking,
    `V_pos_x`, `V_pos_y`, `V_vel_x`, `V_vel_y`, `V_sigma` (the position,
    velocity and sigma_m properties; 0.0 otherwise). Per step `z` (the
    image's extract_peak z, radius 1.0; for reading the trace).
    confidence_from_z: `cz_z` (input) -> `V_cz` per tracker.
    """
    import warnings

    from wifisense.localize.geometry import Geometry
    from wifisense.localize.invert import Inverter, forward_project
    from wifisense.localize.track import Tracker, extract_peak

    geom = Geometry.perimeter(8)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        inv = Inverter(geom)
    L, P = geom.n_links, geom.n_pixels
    noise = _rng_gauss(330007, 60 * L)
    steps = []
    t = 0.0
    for i in range(60):
        if i < 20:
            cx, cy = 1.5 + 0.2 * i, 1.5 + 0.1 * i
        elif i < 25:
            cx = None                      # nobody: noise only
        elif i < 45:
            cx, cy = 6.0 - 0.15 * (i - 25), 4.5 - 0.1 * (i - 25)
        else:
            cx, cy = 7.75, 0.25 + 0.3 * (i - 45)     # along the wall
        dt = 0.4
        if i == 25:
            dt = 4.0                       # a gap past every max_coast_s
        if i == 30:
            dt = 0.0                       # a repeated time
        if i == 33:
            dt = -0.3                      # backwards
        t = t + dt if i else 0.0
        e = noise[i * L:(i + 1) * L]
        if i == 38:
            img = [math.nan] * P
        elif cx is None:
            img = inv.solve([0.1 * v for v in e])
        else:
            y = forward_project(geom, _blob(geom, cx, cy))
            img = inv.solve([a + 0.03 * v for a, v in zip(y, e)])
        steps.append((t, img))
    arr: dict = {"t": [s[0] for s in steps], "image": [v for s in steps for v in s[1]],
                 "z": [extract_peak(geom, s[1]).z for s in steps], "cz_z": [-1.0, 0.0, 3.9, 4.0, 5.0, 7.5, 1e300,
                                                                               math.inf, math.nan]}
    cmp = {"t": "input", "image": "input", "z": "rule:each", "cz_z": "input"}
    for v, kw in LOCALIZE_TRACKERS:
        tr = Tracker(geom, **kw)
        a = {k: [] for k in ("est", "x", "y", "sigma_m", "confidence", "cov_xx", "cov_xy", "cov_yy", "tracking",
                             "pos_x", "pos_y", "vel_x", "vel_y", "sigma")}
        for ts, img in steps:
            est = tr.update(img, ts)
            a["est"].append(int(est is not None))
            for k in ("x", "y", "sigma_m", "confidence", "cov_xx", "cov_xy", "cov_yy"):
                a[k].append(0.0 if est is None else getattr(est, k))
            a["tracking"].append(int(tr.is_tracking))
            pos, vel = tr.position, tr.velocity
            a["pos_x"].append(pos[0] if pos else 0.0)
            a["pos_y"].append(pos[1] if pos else 0.0)
            a["vel_x"].append(vel[0] if vel else 0.0)
            a["vel_y"].append(vel[1] if vel else 0.0)
            a["sigma"].append(tr.sigma_m if tr.sigma_m is not None else 0.0)
        a["cz"] = [tr.confidence_from_z(z) for z in arr["cz_z"]]
        arr.update({f"{v}_{k}": x for k, x in a.items()})
        cmp.update({f"{v}_est": "exact", f"{v}_tracking": "exact", f"{v}_x": "rule", f"{v}_y": "rule",
                    f"{v}_pos_x": "rule", f"{v}_pos_y": "rule", f"{v}_vel_x": "rule", f"{v}_vel_y": "rule",
                    f"{v}_cz": "rule:each"})
        for k in ("sigma_m", "confidence", "cov_xx", "cov_xy", "cov_yy", "sigma"):
            cmp[f"{v}_{k}"] = "rule:each"
    sc = {"n_steps": len(steps), "n_pixels": P, "n_variants": len(LOCALIZE_TRACKERS),
          "variants": ",".join(v for v, _k in LOCALIZE_TRACKERS)}
    for v, kw in LOCALIZE_TRACKERS:
        for k, x in kw.items():
            sc[f"{v}_tracker.{k}"] = int(x) if isinstance(x, bool) else x
    _lemit("localize_tracker", sc, arr, cmp)


def _measure_streams():
    """The recorded localize_measure streams: (calibration frames, occupied
    frames, query points), each frame (t, tx, rx, seq, rssi, amplitudes)."""
    from wifisense.roomsim.rng import Rng

    rng = Rng.stream(0, 330010)
    n_sub = 8
    links = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
    base = {lk: 1.0 + 0.5 * i for i, lk in enumerate(links)}
    rssi0 = {lk: -40 - 3 * i for i, lk in enumerate(links)}

    def amps(lk, scale):
        return [base[lk] * scale * (1.0 + 0.03 * rng.gauss()) * (1.0 + 0.1 * k) for k in range(n_sub)]

    def rssi(lk, jitter):
        if lk == (1, 2):
            return rssi0[lk]                         # a constant link: zero baseline
        return rssi0[lk] + int(round(jitter * rng.gauss()))

    calib = []
    for m in range(150):                            # 15 s at 10 Hz per link
        for i, tx in enumerate((0, 1, 2)):
            if tx == 2 and m >= 50:
                continue                             # node 2 calibrates for 5 s only
            t = (m * 3 + i) / 30.0
            for rx in (0, 1, 2):
                if rx == tx:
                    continue
                a = amps((tx, rx), 1.0)
                if (tx, rx) == (0, 1) and m == 70:
                    a[3] = math.nan                  # dropped by the validity gate
                calib.append((t, tx, rx, m, rssi((tx, rx), 1.0), a))
    occ = []
    queries = []
    t0 = 100.0
    for m in range(120):                            # 12 s
        for i, tx in enumerate((0, 1, 2)):
            if tx == 2 and 80 <= m < 95:
                continue                             # node 2 silent: its links age out
            t = t0 + (m * 3 + i) / 30.0
            for rx in (0, 1, 2):
                if rx == tx:
                    continue
                lk = (tx, rx)
                scale = 1.0
                jit = 1.0
                if lk in ((0, 1), (1, 0)) and 30 <= m < 80:
                    scale = 1.0 + 0.4 * math.sin(0.9 * m)      # a person on the link
                    jit = 3.0
                a = amps(lk, scale)
                tt = t
                if lk == (2, 0) and m >= 50:
                    tt = t - 5.0                     # a restarted clock on one link
                if lk == (0, 2) and m == 20:
                    a = [0.0] * n_sub                # zero power: dropped
                if lk == (1, 2) and m == 25:
                    a[0] = math.nan                  # non-finite: dropped
                occ.append((tt, tx, rx, m, rssi(lk, jit), a))
                if lk == (1, 0) and m == 33:
                    occ.append((t - 0.25, tx, rx, m, rssi(lk, jit), amps(lk, scale)))   # out of order
                if lk == (2, 1) and m == 40:
                    occ.append((t - 2.5, tx, rx, m, rssi(lk, jit), amps(lk, 1.0)))     # one stale sample
                if lk == (0, 1) and m == 100:
                    for _ in range(45):               # a burst at one time: the window cap
                        occ.append((t, tx, rx, m, rssi(lk, jit), amps(lk, 1.0)))
        if m % 5 == 4:
            qt = t0 + (m * 3 + 2) / 30.0 + 0.01
            queries.append((len(occ), None if (m // 5) % 4 == 3 else qt))
    queries.append((len(occ), t0 + 12.0 + 2.5))      # the stream stopped: windows age out
    return n_sub, calib, occ, queries


#: localize_measure's runs: (prefix, signal, calibration_mode, kwargs, use set_baseline).
LOCALIZE_MEASURE_RUNS = (
    ("tp", "total_power_db", "subtract", {}, False),
    ("ss", "static_suppressed_amplitude", "subtract", {}, False),
    ("rs", "rssi_dbm", "subtract", {}, False),
    ("tpn", "total_power_db", "normalize", {}, False),
    ("rsn", "rssi_dbm", "normalize", {}, False),
    ("rsc", "rssi_dbm", "subtract", {"clamp_negative": True}, False),
    ("rsm", "rssi_dbm", "subtract", {"window_s": 0.75, "min_samples": 4}, False),
    ("given", "total_power_db", "subtract", {}, True),
)


def gen_localize_measure() -> None:
    """measure.py's LinkMeasurement with each LinkSignal on recorded frames.

    Geometry: Room(4, 3), nodes 0 (0.2, 0.2), 1 (3.8, 0.2), 2 (2.0, 2.8),
    pitch 0.5, lambda 0.3: 6 links, canonical order (0,1), (0,2), (1,0),
    (1,2), (2,0), (2,1). `sample_rate_hz` 10.0 (3 transmitters, 1/30 s
    slots), `window_s` 1.0 unless a run says otherwise, 8 subcarriers.

    Frames (input, in feed order): calibration `c_t`, `c_tx`, `c_rx`,
    `c_seq` (the frame's sample_index), `c_rssi`, `c_amp` (frame-major, 8
    each); occupied `o_t`, `o_tx`, `o_rx`, `o_seq`, `o_rssi`, `o_amp`
    likewise. Every frame's phases are 0.0, its sample_rate_hz 10.0 and
    its node_id its rx. Calibration: 15 s; node 2 transmits only for its first
    5 s (its links get too few windows and are imputed); link (1,2)'s RSSI
    is constant (a zero RSSI baseline); one NaN frame. Occupied (t from
    100 s): links (0,1)/(1,0) disturbed from 3 s to 8 s; a zero-power and a
    NaN frame (dropped); one sample 0.25 s out of order; one stale sample
    2.5 s back; link (2,0)'s clock jumps back 5 s from 5 s on (a restart
    after two stale samples); node 2 silent from 8.0 s to 9.5 s; a burst of
    45 frames at one time on (0,1) (window cap 40).

    Queries (input): `q_after` (occupied frames fed before it), `q_has_t`
    (0: vector() without t), `q_t`. Every query calls vector(t),
    sample_counts(t) and ready_mask(t), in that order.

    Runs, `variants` (each from a fresh LinkMeasurement fed the same
    frames): `tp` TotalPowerSignal; `ss` StaticSuppressedSignal(0.03);
    `rs` RssiSignal; `tpn` total power, NORMALIZE; `rsn` RSSI, NORMALIZE
    (the constant link's zero baseline: y 0.0 and not ready); `rsc` RSSI
    with clamp_negative; `rsm` RSSI with window_s 0.75 and min_samples 4;
    `given` total power with set_baseline(`input_baseline`,
    `input_baseline_std`) (input) instead of calibrate_from; the others
    calibrate_from(calibration frames, min_windows=10). Per run V: scalars
    `V_min_samples`, `V_is_calibrated`, `V_window_s`, `V_signal` (code),
    `V_mode`, `V_clamp_negative`, `V_set_baseline` (1 for `given`);
    `V_baseline`, `V_baseline_std`, `V_baseline_raw_std`,
    `V_calibrated_mask` (after calibration); per query (query-major, 6
    each): `V_vector`, `V_counts`, `V_ready`; `V_counters` (3 per query:
    clock_restarts, stale_samples, window_overflows).
    Error cases (`err_code`, `err.<i>`).
    """
    from wifisense.localize.geometry import Geometry, Node, PixelGrid, Room
    from wifisense.localize.measure import LinkMeasurement
    from wifisense.localize.pipeline import make_signal
    from wifisense.types import CsiFrame

    room = Room(4.0, 3.0)
    geom = Geometry(room, [Node(0, 0.2, 0.2), Node(1, 3.8, 0.2), Node(2, 2.0, 2.8)], PixelGrid(room, 0.5), 0.3)
    rate = 10.0
    n_sub, calib, occ, queries = _measure_streams()

    def frames(seq):
        for t, tx, rx, s, r, a in seq:
            yield t, tx, rx, CsiFrame(list(a), [0.0] * n_sub, s, rate, r, rx)

    arr: dict = {}
    for p, seq in (("c", calib), ("o", occ)):
        arr[f"{p}_t"] = [f[0] for f in seq]
        arr[f"{p}_tx"] = [f[1] for f in seq]
        arr[f"{p}_rx"] = [f[2] for f in seq]
        arr[f"{p}_seq"] = [f[3] for f in seq]
        arr[f"{p}_rssi"] = [f[4] for f in seq]
        arr[f"{p}_amp"] = [v for f in seq for v in f[5]]
    arr["q_after"] = [q[0] for q in queries]
    arr["q_has_t"] = [int(q[1] is not None) for q in queries]
    arr["q_t"] = [0.0 if q[1] is None else q[1] for q in queries]
    set_b = [0.02, 0.03, 0.04, 0.0, 0.05, 0.06]
    set_s = [0.01, 0.01, 0.02, 0.0, 0.02, 0.03]
    arr["input_baseline"] = set_b
    arr["input_baseline_std"] = set_s
    cmp = {k: "input" for k in arr}
    assert not any(k.startswith(r[0] + "_") for k in arr for r in LOCALIZE_MEASURE_RUNS), "a run prefix shadows an input"
    sc: dict = {"n_links": 6, "n_sub": n_sub, "sample_rate_hz": rate, "room_w": 4.0, "room_h": 3.0,
                "pitch_m": 0.5, "lambda_m": 0.3, "node_x": "0.2,3.8,2.0", "node_y": "0.2,0.2,2.8",
                "min_windows": 10, "n_calib": len(calib), "n_occ": len(occ), "n_queries": len(queries),
                "n_variants": len(LOCALIZE_MEASURE_RUNS),
                "variants": ",".join(r[0] for r in LOCALIZE_MEASURE_RUNS)}
    for v, sig, mode, kw, use_set in LOCALIZE_MEASURE_RUNS:
        meas = LinkMeasurement(geom, window_s=kw.get("window_s", 1.0), sample_rate_hz=rate, signal=make_signal(sig),
                               calibration_mode=mode, min_samples=kw.get("min_samples"),
                               clamp_negative=kw.get("clamp_negative", False))
        if use_set:
            meas.set_baseline(set_b, set_s)
        else:
            meas.calibrate_from(frames(calib), min_windows=10)
        a = {"baseline": meas.baseline_vector(), "baseline_std": meas.baseline_std_vector(),
             "baseline_raw_std": meas.baseline_raw_std_vector(),
             "calibrated_mask": [int(b) for b in meas.calibrated_mask()],
             "vector": [], "counts": [], "ready": [], "counters": []}
        fed = 0
        it = frames(occ)
        for after, qt in queries:
            while fed < after:
                t, tx, rx, fr = next(it)
                meas.update(t, tx, rx, fr)
                fed += 1
            a["vector"].extend(meas.vector(qt))
            a["counts"].extend(meas.sample_counts(qt))
            a["ready"].extend(int(b) for b in meas.ready_mask(qt))
            a["counters"].extend((meas.clock_restarts, meas.stale_samples, meas.window_overflows))
        arr.update({f"{v}_{k}": x for k, x in a.items()})
        cmp.update({f"{v}_baseline": "rule", f"{v}_baseline_std": "rule", f"{v}_baseline_raw_std": "rule",
                    f"{v}_calibrated_mask": "exact", f"{v}_vector": "rule:6", f"{v}_counts": "exact",
                    f"{v}_ready": "exact", f"{v}_counters": "exact"})
        sc.update({f"{v}_min_samples": meas.min_samples, f"{v}_is_calibrated": int(meas.is_calibrated),
                   f"{v}_window_s": meas.window_s, f"{v}_signal": SIGNAL_CODE[sig], f"{v}_mode": mode,
                   f"{v}_clamp_negative": int(meas.clamp_negative), f"{v}_set_baseline": int(use_set)})
    m0 = LinkMeasurement(geom, sample_rate_hz=rate)
    fr0 = CsiFrame([1.0] * n_sub, [0.0] * n_sub, 0, rate, -40, 1)
    errs = [
        ("LinkMeasurement(geometry, window_s=0.0, sample_rate_hz=10.0)",
         lambda: LinkMeasurement(geom, window_s=0.0, sample_rate_hz=rate)),
        ("LinkMeasurement(geometry, sample_rate_hz=0.0)", lambda: LinkMeasurement(geom, sample_rate_hz=0.0)),
        ("LinkMeasurement(geometry, sample_rate_hz=10.0, min_samples=1)",
         lambda: LinkMeasurement(geom, sample_rate_hz=rate, min_samples=1)),
        ("LinkMeasurement(geometry, sample_rate_hz=10.0, calibration_mode='median')",
         lambda: LinkMeasurement(geom, sample_rate_hz=rate, calibration_mode="median")),
        ("update(0.0, 0, 0, frame)", lambda: m0.update(0.0, 0, 0, fr0)),
        ("update(0.0, 0, 7, frame)", lambda: m0.update(0.0, 0, 7, fr0)),
        ("set_baseline([0.1] * 5)", lambda: m0.set_baseline([0.1] * 5)),
        ("set_baseline([0.1] * 5 + [-0.1])", lambda: m0.set_baseline([0.1] * 5 + [-0.1])),
        ("set_baseline([0.1] * 5 + [nan])", lambda: m0.set_baseline([0.1] * 5 + [math.nan])),
        ("set_baseline([0.1] * 6, [0.1] * 5)", lambda: m0.set_baseline([0.1] * 6, [0.1] * 5)),
        ("calibrate_from(empty stream)", lambda: LinkMeasurement(geom, sample_rate_hz=rate).calibrate_from([])),
        ("CsiFrame([1.0, 2.0], [0.0], 0, 10.0)", lambda: CsiFrame([1.0, 2.0], [0.0], 0, 10.0)),
        ("CsiFrame([1.0], [0.0], 0, 0.0)", lambda: CsiFrame([1.0], [0.0], 0, 0.0)),
    ]
    esc, codes = _err_cases(errs)
    sc.update(esc)
    arr["err_code"] = codes
    cmp["err_code"] = "exact"
    _lemit("localize_measure", sc, arr, cmp)


# -- locate_*: the new pieces -------------------------------------------------


def gen_locate_radius() -> None:
    """spec/locate.md §8: radius95(a, b, d) of [[a, b], [b, d]] and the
    publication test.

    Scalars: the §1 constants `K95`, `CHI2_2_99`, `LN10`; `min_plausible_radius_m`
    (0.3). Cases (input `a`, `b`, `d`), in order: identity, diag(4, 1),
    [[2, 1], [1, 2]], zero, a NaN, inf - inf (a = d = inf), a = inf,
    negative definite, an indefinite matrix with a positive eigenvalue,
    diag(0.01, 0.01) (below the minimum), the matrix whose radius is
    nearest 0.3, entries of 1e300 (h*h overflows), 1e-300 (underflows),
    a negative b, then 12 recorded random PSD matrices. Outputs `r`
    (`rule:each`) and `pub` (publishable at 0.3: exact).
    """
    from wifisense.localize.radius import CHI2_2_99, K95, LN10, publishable, radius95

    t = (0.3 / K95) * (0.3 / K95)
    cases = [(1.0, 0.0, 1.0), (4.0, 0.0, 1.0), (2.0, 1.0, 2.0), (0.0, 0.0, 0.0), (math.nan, 0.0, 1.0),
             (math.inf, 0.0, math.inf), (math.inf, 0.0, 1.0), (-1.0, 0.0, -1.0), (1.0, 0.0, -3.0),
             (0.01, 0.0, 0.01), (t, 0.0, t), (1e300, 0.5e300, 2e300), (1e-300, 0.0, 3e-300), (2.0, -1.5, 1.5)]
    rnd = _rng_draws(330020, 36, -1.0, 1.0)
    for i in range(12):
        p, q, s = rnd[3 * i:3 * i + 3]
        a = p * p + 0.1 * q * q
        d = q * q + 0.05
        cases.append((a, s * math.sqrt(a * d) * 0.9, d))
    rs = [radius95(a, b, d) for a, b, d in cases]
    _lemit("locate_radius",
           {"K95": K95, "CHI2_2_99": CHI2_2_99, "LN10": LN10, "min_plausible_radius_m": 0.3, "n_cases": len(cases)},
           {"a": [c[0] for c in cases], "b": [c[1] for c in cases], "d": [c[2] for c in cases], "r": rs,
            "pub": [int(publishable(r, 0.3)) for r in rs]},
           {"a": "input", "b": "input", "d": "input", "r": "rule:each", "pub": "exact"})


def gen_locate_wlm() -> None:
    """spec/locate.md §5: the weighted-link-midpoint baseline.

    Geometries (`variants` "g0,g1", prefixes `g0_`, `g1_`): g0 a 4 x 4 m room
    with nodes 0 (0,0), 1 (4,0), 2 (4,4), 3 (0,4) (two diagonals), pitch
    0.5, lambda 0.3; g1 the bedroom's nodes as Locate builds them (scene
    below; Room(4.2, 3.6), z dropped, pitch 0.5, lambda 0.3). Per
    geometry V: scalars `V_room_w`, `V_room_h`; `node_x`, `node_y` (input,
    id order 0..3) and per link,
    canonical order, the §5 step-4 terms `a_l`, `c_l`, `ux`, `uy` and
    `mid_x`, `mid_y` (Link.midpoint), `len` (Link.length_m: CPython hypot).

    Cases (input): `geom` (0/1), `signal` (code), `gate_z`,
    `floor_default` (1: wlm_floor None, the signal's default), `floor`,
    `radius_scale`, `min_radius`; `y`, `null_sigma` (case-major, 12
    each). Outputs (§11 Estimate or Abstention, `confidence` always None):
    `reason`, `x`, `y_hat`, `radius`, `cov_xx`, `cov_xy`, `cov_yy` (0.0
    when abstaining). Cases, in order: one diagonal; one diagonal in both
    directions; both diagonals equally (the centre); every link; nothing
    over the gate; the disturbed link's null_sigma 0 (never passes); NaN in
    the disturbed y; NaN null_sigma; NaN gate; a NaN y elsewhere (the
    weight sum fails); radius scale 0, NaN, inf; minimum radius 100; y 0.3
    against null_sigma 0.02 on rssi_dbm (fails its 0.25 floor), on
    total_power_db (passes) and with an explicit floor 0.5 (fails);
    mixed-sign y; then 8 recorded random cases on g1; then three with
    minimum radius 0 (a raw argument here; §9 refuses it in a config):
    the disturbed y inf, and y 1e308 in both directions (the weight sum
    overflows), each giving a non-finite position whose NaN covariance
    has radius 0 >= 0, so only §8's finite x̂, ŷ abstains them
    (`implausible_radius`); and the one-diagonal case (published). Then
    three more raw calls on the one-diagonal case whose radius is not
    > 0 though it is >= the minimum, each abstaining only by §8's
    `r > 0` (`implausible_radius`): radius scale 0 and -0.0 (r 0.0 and
    -0.0) with minimum radius 0, and radius scale -1 (r about -4.0) with
    minimum radius -10.
    """
    from wifisense.localize.baseline import weighted_link_midpoint, wlm_covariance_terms
    from wifisense.localize.config import LocateConfig
    from wifisense.localize.geometry import Geometry, Node, PixelGrid, Room
    from wifisense.localize.pipeline import locate_geometry
    from wifisense.localize.radius import radius95
    from wifisense.roomsim import load_scene

    rel = "spec/scenes/bedroom.json"
    r4 = Room(4.0, 4.0)
    g0 = Geometry(r4, [Node(0, 0.0, 0.0), Node(1, 4.0, 0.0), Node(2, 4.0, 4.0), Node(3, 0.0, 4.0)],
                  PixelGrid(r4, 0.5), 0.3)
    g1 = locate_geometry(load_scene(os.path.join(LOCATE_REPO, rel)), LocateConfig())
    geoms = (g0, g1)
    arr: dict = {}
    cmp: dict = {}
    for gi, g in enumerate(geoms):
        terms = [wlm_covariance_terms(lk, g.lambda_m) for lk in g.links]
        a = {"node_x": [n.x for n in g.nodes], "node_y": [n.y for n in g.nodes],
             "a_l": [t[0] for t in terms], "c_l": [t[1] for t in terms], "ux": [t[2] for t in terms],
             "uy": [t[3] for t in terms], "mid_x": [lk.midpoint[0] for lk in g.links],
             "mid_y": [lk.midpoint[1] for lk in g.links], "len": [lk.length_m for lk in g.links]}
        arr.update({f"g{gi}_{k}": v for k, v in a.items()})
        cmp.update({f"g{gi}_{k}": ("input" if k.startswith("node") else "rule") for k in a})
    L = g0.n_links
    i02, i20, i13, i31 = g0.link_index(0, 2), g0.link_index(2, 0), g0.link_index(1, 3), g0.link_index(3, 1)

    def yv(pairs, base=0.0):
        y = [base] * L
        for i, v in pairs:
            y[i] = v
        return y

    s01 = [0.1] * L
    D = dict(signal="rssi_dbm", gate_z=3.0, floor=None, scale=1.0, minr=0.3)
    cases = [
        (0, yv([(i02, 2.0)]), s01, D),
        (0, yv([(i02, 2.0), (i20, 2.0)]), s01, D),
        (0, yv([(i02, 1.5), (i20, 1.5), (i13, 1.5), (i31, 1.5)]), s01, D),
        (0, [1.0 + 0.1 * i for i in range(L)], s01, D),
        (0, yv([(i02, 0.5)]), s01, D),
        (0, yv([(i02, 2.0)]), [0.0 if i == i02 else 0.1 for i in range(L)], D),
        (0, yv([(i02, math.nan), (i13, 0.3)]), s01, D),
        (0, yv([(i02, 2.0)]), [math.nan if i == i02 else 0.1 for i in range(L)], D),
        (0, yv([(i02, 2.0)]), s01, dict(D, gate_z=math.nan)),
        (0, yv([(i02, 2.0), (i13, math.nan)]), s01, D),
        (0, yv([(i02, 2.0)]), s01, dict(D, scale=0.0)),
        (0, yv([(i02, 2.0)]), s01, dict(D, scale=math.nan)),
        (0, yv([(i02, 2.0)]), s01, dict(D, scale=math.inf)),
        (0, yv([(i02, 2.0)]), s01, dict(D, minr=100.0)),
        (0, yv([(i02, 0.3)]), [0.02] * L, D),
        (0, yv([(i02, 0.3)]), [0.02] * L, dict(D, signal="total_power_db")),
        (0, yv([(i02, 0.3)]), [0.02] * L, dict(D, signal="static_suppressed_amplitude", floor=0.5)),
        (0, yv([(i02, 2.0), (i13, 0.7)], -0.2), s01, D),
    ]
    L1 = g1.n_links
    rnd = _rng_draws(330021, 8 * 2 * L1, 0.0, 1.0)
    for c in range(8):
        y = [3.0 * (rnd[c * 2 * L1 + i] - 0.3) for i in range(L1)]
        s = [0.05 + 0.3 * rnd[c * 2 * L1 + L1 + i] for i in range(L1)]
        cases.append((1, y, s, dict(D, gate_z=1.0 + 0.5 * c, signal=("rssi_dbm", "total_power_db")[c % 2])))
    n_before_min0 = len(cases)
    cases += [
        (0, yv([(i02, math.inf)]), s01, dict(D, minr=0.0)),
        (0, yv([(i02, 1e308), (i20, 1e308)]), s01, dict(D, minr=0.0)),
        (0, yv([(i02, 2.0)]), s01, dict(D, minr=0.0)),
    ]
    n_before_r_pos = len(cases)
    cases += [
        (0, yv([(i02, 2.0)]), s01, dict(D, scale=0.0, minr=0.0)),
        (0, yv([(i02, 2.0)]), s01, dict(D, scale=-0.0, minr=0.0)),
        (0, yv([(i02, 2.0)]), s01, dict(D, scale=-1.0, minr=-10.0)),
    ]
    keys = ("geom", "signal", "gate_z", "floor_default", "floor", "radius_scale", "min_radius", "y", "null_sigma",
            "reason", "x", "y_hat", "radius", "cov_xx", "cov_xy", "cov_yy")
    for k in keys:
        arr[k] = []
    for gi, y, s, o in cases:
        rec = weighted_link_midpoint(geoms[gi], y, s, o["signal"], wlm_gate_z=o["gate_z"], wlm_floor=o["floor"],
                                     wlm_radius_scale=o["scale"], min_plausible_radius_m=o["minr"])
        row = _est_row(rec)
        for k, v in (("geom", gi), ("signal", SIGNAL_CODE[o["signal"]]), ("gate_z", o["gate_z"]),
                     ("floor_default", int(o["floor"] is None)), ("floor", 0.0 if o["floor"] is None else o["floor"]),
                     ("radius_scale", o["scale"]), ("min_radius", o["minr"]), ("reason", row[0]), ("x", row[1]),
                     ("y_hat", row[2]), ("radius", row[3]), ("cov_xx", row[4]), ("cov_xy", row[5]),
                     ("cov_yy", row[6])):
            arr[k].append(v)
        arr["y"].extend(y)
        arr["null_sigma"].extend(s)
    # The minimum-radius-0 cases pin §8's finite-position term: refuse to
    # write the set if they stop doing so.
    implausible = _reason_code("implausible_radius")
    assert arr["reason"][n_before_min0:n_before_r_pos] == [implausible, implausible, -1], \
        arr["reason"][n_before_min0:n_before_r_pos]
    # The r > 0 cases: abstained although r >= the minimum (refuse to write
    # the set if §8's r > 0 stops deciding them).
    for (_gi, _y, _s, o), reason in zip(cases[n_before_r_pos:], arr["reason"][n_before_r_pos:]):
        r = o["scale"] * radius95(arr["cov_xx"][n_before_min0 + 2], arr["cov_xy"][n_before_min0 + 2],
                                  arr["cov_yy"][n_before_min0 + 2])
        assert r >= o["minr"] and not r > 0.0 and reason == implausible, (o, r, reason)
    cmp.update({k: "input" for k in keys[:9]})
    cmp.update({"reason": "exact", "x": "rule", "y_hat": "rule", "radius": "rule:each", "cov_xx": "rule:each",
                "cov_xy": "rule:each", "cov_yy": "rule:each"})
    sc = _scene_scalars(rel)
    sc.update({"n_cases": len(cases), "n_links": L, "lambda_m": 0.3, "n_variants": 2, "variants": "g0,g1",
               "g0_room_w": 4.0, "g0_room_h": 4.0,
               "g1_room_w": g1.room.width_m, "g1_room_h": g1.room.height_m})
    _lemit("locate_wlm", sc, arr, cmp)


def _hall_world():
    from wifisense.roomsim import Config, RoomSim, load_scene

    rel = "spec/scenes/hall-l-shaped.json"
    scene = load_scene(os.path.join(LOCATE_REPO, rel))
    return rel, scene, RoomSim(scene, Config(duration_s=0.0))


def gen_locate_los() -> None:
    """spec/locate.md §6.2's line-of-sight primitives, one row each:
    `kind` (0 wall_blocks, 1 box_blocks, 2 cylinder_blocks), the
    primitive `prim` (8 per row: wall (ax, ay, bx, by, h, 0, 0, 0); box
    (cx, cy, cz, sl, sb, sh, cs, sn); cylinder (cx, cy, radius, top, 0, 0,
    0, 0)), the segment `p` and `q` (3 per row: x, y, z) -> `blocked`
    (exact). Rows: the hand cases (a wall's centreline, a doorway gap, over
    the top, parallel; a box at yaw 0 and 90 degrees; a cylinder below
    and above its top, beside it, nearer end inside, a vertical segment),
    then every wall and box of the hall (Room Sim's W, implied walls
    included; objects that are not doors or windows) and three cylinders
    against 40 recorded random segments each.
    """
    from wifisense.localize.phone import box_blocks, cylinder_blocks, wall_blocks
    from wifisense.localize.pipeline import phone_boxes, phone_walls

    rel, _scene, sim = _hall_world()
    w0 = (0.0, 3.0, 2.0, 3.0, 2.7)
    table = (1.0, 2.0, 0.375, 1.2, 0.6, 0.75, 1.0, 0.0)
    rot = (1.0, 2.0, 0.375, 1.2, 0.6, 0.75, 0.0, 1.0)
    cyl = (2.0, 2.0, 0.18, 1.7)
    rows = [
        (0, w0, (1.0, 1.0, 1.0), (1.0, 5.0, 1.2)), (0, (3.0, 3.0, 6.0, 3.0, 2.7), (2.5, 1.0, 1.0), (2.5, 5.0, 1.2)),
        (0, w0, (2.5, 1.0, 1.0), (2.5, 5.0, 1.2)), (0, w0, (1.0, 1.0, 3.0), (1.0, 5.0, 3.0)),
        (0, w0, (0.5, 3.0, 1.0), (1.5, 3.0, 1.0)), (0, w0, (2.0, 1.0, 1.0), (2.0, 5.0, 1.0)),
        (0, w0, (1.0, 3.0, 1.0), (1.0, 5.0, 1.0)),
        (1, table, (0.0, 2.0, 0.3), (2.0, 2.0, 0.3)), (1, table, (0.0, 2.0, 1.0), (2.0, 2.0, 1.0)),
        (1, rot, (1.0, 1.0, 0.3), (1.0, 3.0, 0.3)), (1, rot, (0.5, 1.0, 0.3), (0.5, 3.0, 0.3)),
        (1, table, (1.0, 2.0, 0.3), (1.0, 2.0, 0.6)), (1, table, (0.4, 1.0, 0.75), (0.4, 3.0, 0.75)),
        (2, cyl, (1.0, 2.0, 1.0), (3.0, 2.0, 1.2)), (2, cyl, (1.0, 2.0, 1.9), (3.0, 2.0, 1.9)),
        (2, cyl, (1.0, 2.3, 1.0), (3.0, 2.3, 1.0)), (2, cyl, (2.05, 2.0, 1.0), (4.0, 2.0, 1.0)),
        (2, cyl, (0.0, 2.0, 1.0), (2.05, 2.0, 1.0)), (2, cyl, (0.0, 2.0, 1.0), (1.5, 2.0, 1.0)),
        (2, cyl, (2.0, 2.0, 0.5), (2.0, 2.0, 1.5)), (2, cyl, (2.05, 2.0, 1.8), (4.0, 2.0, 1.0)),
    ]
    prims = [(0, w) for w in phone_walls(sim.world)] + [(1, b) for b in phone_boxes(sim.world)]
    prims += [(2, (2.0, 5.8, 0.18, 1.75)), (2, (7.0, 0.6, 0.18, 1.2)), (2, (4.5, 1.5, 0.25, 0.9))]
    rnd = _rng_draws(330030, len(prims) * 40 * 6, 0.0, 1.0)
    i = 0
    for kind, pr in prims:
        for _ in range(40):
            u = rnd[i:i + 6]
            i += 6
            rows.append((kind, pr, (9.0 * u[0], 7.0 * u[1], 0.3 + 2.0 * u[2]),
                         (9.0 * u[3], 7.0 * u[4], 0.3 + 2.0 * u[5])))
    fn = (wall_blocks, box_blocks, cylinder_blocks)
    arr = {"kind": [], "prim": [], "p": [], "q": [], "blocked": []}
    for kind, pr, p, q in rows:
        pr = tuple(float(v) for v in pr)
        arr["kind"].append(kind)
        arr["prim"].extend(pr + (0.0,) * (8 - len(pr)))
        arr["p"].extend(p)
        arr["q"].extend(q)
        arr["blocked"].append(int(fn[kind](p, q, pr)))
    sc = _scene_scalars(rel)
    sc.update({"n_rows": len(rows), "n_walls": len(phone_walls(sim.world)), "n_boxes": len(phone_boxes(sim.world))})
    _lemit("locate_los", sc, arr,
           {"kind": "input", "prim": "input", "p": "input", "q": "input", "blocked": "exact"})


def _locate_cfg_scalars(cfg, prefix: str = "locate.") -> dict:
    """`<prefix><name>` for every LocateConfig value that differs from its default."""
    import dataclasses as dc

    from wifisense.localize.config import LocateConfig

    d = LocateConfig()
    out = {}
    for f in dc.fields(LocateConfig):
        v = getattr(cfg, f.name)
        if v != getattr(d, f.name) and v is not None:
            out[prefix + f.name] = v
    return out


def _phone_trajectory():
    """localize_phone_run's samples: (t, (px, py, pz), cylinders)."""
    way = [(1.0, 1.0), (7.5, 1.5), (2.0, 1.5), (2.0, 6.0), (3.5, 5.0)]
    seg = []
    for (ax, ay), (bx, by) in zip(way, way[1:]):
        seg.append((ax, ay, bx, by, math.sqrt((bx - ax) * (bx - ax) + (by - ay) * (by - ay))))
    total = 0.0
    for s in seg:
        total = total + s[4]
    out = []
    for j in range(1200):                            # 15 s at 12.5 ms
        t = float(j) * 0.0125
        if 5.0 < t < 5.9:
            continue                                  # no truth samples: epochs passed over
        s = (t * 1.1) % total
        for ax, ay, bx, by, ln in seg:
            if s <= ln:
                f = s / ln
                px, py = ax + f * (bx - ax), ay + f * (by - ay)
                break
            s -= ln
        pz = 1.0 + 0.05 * math.sin(2.0 * t)
        other = (2.0 + 1.5 * math.sin(0.4 * t), 5.0 + 1.0 * math.cos(0.3 * t), 0.18, 1.7)
        cyls = [(px - 0.05, py + 0.25, 0.18, 1.75), other, (7.0, 0.6, 0.18, 1.2)]
        out.append((t, (px, py, pz), cyls))
    return out


#: locate_phone_run's runs: (prefix, LocateConfig keyword arguments).
LOCATE_PHONE_RUNS = (
    ("a", dict(phone_seed=11)),
    ("b", dict(phone_seed=12, rtt_outlier_p=0.3, rtt_fail_p=0.3, rssi_fail_p=0.3, rtt_nlos_bias_m=2.0,
               phone_rate_hz=10.0, phone_max_coast_s=0.5, huber_k=1.0)),
    ("c", dict(phone_seed=55, rtt_sigma_m=4.0, rtt_fail_p=0.5, rssi_fail_p=0.3, rssi_shadow_sigma_db=1.0,
               rssi_device_sigma_db=1e308, min_plausible_radius_m=2.6, phone_radius_scale=0.5,
               wlm_radius_scale=1.5, rti_radius_scale=2.0)),
)


def _check_phone_run_edges(a: dict, nn: int) -> None:
    """locate_phone_run variant c's coverage (its docstring): every edge it
    exists to pin is reached, or the set is not written."""
    v_bad = [not math.isfinite(x) for x in a["rssi_v"]]
    assert any(v_bad), "c: no non-finite RSSI v"
    assert all(p == 0 for p, bad in zip(a["rssi_present"], v_bad) if bad), "c: a report from a non-finite v"
    dbm = [q for q, p in zip(a["rssi_dbm"], a["rssi_present"]) if p]
    assert -100 in dbm and 0 in dbm, "c: the clamp is not reached at both ends"
    for m in ("rtt", "rssi"):
        got = set(a[f"{m}_reason"])
        assert {-1, _reason_code("implausible_radius")} <= got, f"c: {m} reasons {sorted(got)}"
    fix = set(a["rtt_fix_reason"])
    need = {_reason_code(r) for r in ("no_reports", "degenerate", "not_converged")}
    assert need <= fix, f"c: rtt fix reasons {sorted(fix)}"
    for e, r in enumerate(a["rtt_fix_reason"]):
        if r == _reason_code("degenerate"):
            ids = [i for i in range(nn) if a["rtt_present"][e * nn + i]]
            # The hall's collinear triples: x = 0.3 and x + y = 7.0.
            assert ids in ([0, 5, 6], [3, 4, 6]), f"c: degenerate epoch {e} from nodes {ids}"


def gen_locate_phone_run() -> None:
    """spec/locate.md §6-§7 end to end on plain data: the simulated phone in
    the L-shaped hall (nodes, walls and boxes from the scene below) on a
    synthetic trajectory with recorded blockers.

    Inputs: nodes `node_id`, `node_x`, `node_y`, `node_z` (id order);
    walls `wall` (5 per wall: ax, ay, bx, by, h; Room Sim's W, implied
    walls included); boxes `box` (8 per box: cx, cy, cz, sl, sb, sh, cs,
    sn; objects that are not doors or windows); truth samples `s_t`,
    `s_x`, `s_y`, `s_z` (the phone) with `s_ncyl` cylinders each, `cyl`
    (4 per cylinder: cx, cy, radius, top; sample-major: the carrier, a
    walker, a seated person). Samples every 12.5 ms for 15 s except
    5.0 < t < 5.9 (epochs passed over are dropped). Scalars
    `tx_power_dbm`, `carrier_hz` (Room Sim's channel 6).

    Runs (`variants` "a,b,c"), each `PhoneRun(nodes, walls, boxes,
    LocateConfig(**kw).resolved(0, tx_power_dbm, carrier_hz))` fed every
    sample; `V_cfg.<name>` lists kw, `V_phone_seed`, `V_rssi_p0_dbm`,
    `V_pl0`. `a`: defaults, phone_seed 11. `b`: phone_seed 12,
    rtt_outlier_p 0.3, rtt_fail_p 0.3, rssi_fail_p 0.3, rtt_nlos_bias_m
    2.0, phone_rate_hz 10, phone_max_coast_s 0.5, huber_k 1.0. `c` (the
    edges, from a run): phone_seed 55, rtt_sigma_m 4.0, rtt_fail_p 0.5,
    rssi_fail_p 0.3, rssi_shadow_sigma_db 1.0, rssi_device_sigma_db 1e308,
    min_plausible_radius_m 2.6, phone_radius_scale 0.5 (wlm_radius_scale
    1.5 and rti_radius_scale 2.0, which the phone must not read). Its
    device offsets are +-inf on three nodes (|g| > 1.797: v is not
    finite, no RSSI report, S still computed, §6.5) and finite but 3e306
    to 1.7e308 in size on the rest (every report clamped to 0 or -100);
    RTT and RSSI both publish and abstain `implausible_radius`, and RTT
    also reaches `no_reports`, `degenerate` (only nodes 3, 4, 6,
    collinear on x + y = 7, report) and `not_converged`. The generator
    refuses to write the set if any of these is lost.
    Per node: `V_offset_m` (off_n), `V_device_db` (dev_n).
    Per fired epoch: `V_ep_j` (the sample index), `V_ep_t`.
    Per epoch per node (epoch-major, node id order; `rule:8` = one
    epoch): `V_blocked`; `V_d` (true 3-D distance); the draws in §6.3
    order, RTT stream `V_u_fail`, `V_g`, `V_u_out`, `V_u_mag`, `V_u_b`,
    RSSI stream `V_u_fail_s`, `V_g_s`, `V_u_j` (uniforms exact, gauss by
    the rule); `V_rtt_value` (r before the failure test), `V_rtt_present`,
    `V_rtt_m`; `V_rssi_v` (v before rounding), `V_rssi_present`,
    `V_rssi_dbm` (`tie:V_rssi_v`); `V_shadow` (S after this epoch).
    Per epoch per method M (rtt, rssi): the fix `V_M_fix_reason`,
    `V_M_fix_x`, `V_M_fix_y`, `V_M_fix_inv00`, `V_M_fix_inv01`,
    `V_M_fix_inv11` (inv(N)), `V_M_fix_r00`, `V_M_fix_r01`, `V_M_fix_r11`
    (R), `V_M_fix_chi2` (0.0 for rssi), `V_M_fix_iter`; the filter step
    `V_M_kf_reason`, `V_M_kf_x`, `V_M_kf_y`, `V_M_kf_p00`, `V_M_kf_p01`,
    `V_M_kf_p11`, `V_M_kf_started`, `V_M_kf_discarded`; the published
    record `V_M_reason`, `V_M_x`, `V_M_y`, `V_M_radius`, `V_M_cov_xx`,
    `V_M_cov_xy`, `V_M_cov_yy` (P + beta^2 inv(N)).

    The epoch clock alone: `clk_t` (input, rate 5 Hz from e_next 0) ->
    `clk_fired`, `clk_e_next` (after each call), exact. The last four
    times (1e7 twice, 10000000.2, 12345678.9) take §6.1's skip-ahead,
    which must end where the plain loop would.
    """
    from wifisense.localize.config import LocateConfig
    from wifisense.localize.phone import EpochClock, PhoneRun
    from wifisense.localize.pipeline import phone_boxes, phone_nodes, phone_walls
    from wifisense.roomsim.numerics import carrier_hz

    rel, scene, sim = _hall_world()
    nodes = phone_nodes(scene)
    walls = phone_walls(sim.world)
    boxes = phone_boxes(sim.world)
    samples = _phone_trajectory()
    txp, fc = 10.0, carrier_hz(6)
    nn = len(nodes)
    arr: dict = {"node_id": [n[0] for n in nodes], "node_x": [n[1] for n in nodes], "node_y": [n[2] for n in nodes],
                 "node_z": [n[3] for n in nodes], "wall": [float(v) for w in walls for v in w],
                 "box": [float(v) for b in boxes for v in b], "s_t": [s[0] for s in samples],
                 "s_x": [s[1][0] for s in samples], "s_y": [s[1][1] for s in samples],
                 "s_z": [s[1][2] for s in samples], "s_ncyl": [len(s[2]) for s in samples],
                 "cyl": [v for s in samples for c in s[2] for v in c]}
    cmp = {k: "input" for k in arr}
    sc = _scene_scalars(rel)
    sc.update({"n_nodes": nn, "n_walls": len(walls), "n_boxes": len(boxes), "n_samples": len(samples),
               "tx_power_dbm": txp, "carrier_hz": fc, "n_variants": len(LOCATE_PHONE_RUNS),
               "variants": ",".join(v for v, _k in LOCATE_PHONE_RUNS)})
    node_keys = ("blocked", "d", "u_fail", "g", "u_out", "u_mag", "u_b", "u_fail_s", "g_s", "u_j", "rtt_value",
                 "rtt_present", "rtt_m", "rssi_v", "rssi_present", "rssi_dbm", "shadow")
    fix_keys = ("fix_reason", "fix_x", "fix_y", "fix_inv00", "fix_inv01", "fix_inv11", "fix_r00", "fix_r01",
                "fix_r11", "fix_chi2", "fix_iter")
    kf_keys = ("kf_reason", "kf_x", "kf_y", "kf_p00", "kf_p01", "kf_p11", "kf_started", "kf_discarded")
    pub_keys = ("reason", "x", "y", "radius", "cov_xx", "cov_xy", "cov_yy")
    for v, kw in LOCATE_PHONE_RUNS:
        cfg = LocateConfig(**kw).resolved(0, txp, fc)
        run = PhoneRun(nodes, walls, boxes, cfg)
        a: dict = {"offset_m": [run.radio.offset_m[n[0]] for n in nodes],
                   "device_db": [run.radio.device_db[n[0]] for n in nodes], "ep_j": [], "ep_t": []}
        for k in node_keys:
            a[k] = []
        for m in ("rtt", "rssi"):
            for k in fix_keys + kf_keys + pub_keys:
                a[f"{m}_{k}"] = []
        for j, (t, ph, cyls) in enumerate(samples):
            rec = run.sample(t, ph, cyls)
            if rec is None:
                continue
            a["ep_j"].append(j)
            a["ep_t"].append(t)
            for r in run.last_reports:
                d = r.rtt_draws + r.rssi_draws
                for k, x in (("blocked", int(r.blocked)), ("d", r.d), ("u_fail", d[0]), ("g", d[1]), ("u_out", d[2]),
                             ("u_mag", d[3]), ("u_b", d[4]), ("u_fail_s", d[5]), ("g_s", d[6]), ("u_j", d[7]),
                             ("rtt_value", r.rtt_value), ("rtt_present", int(r.rtt_m is not None)),
                             ("rtt_m", 0.0 if r.rtt_m is None else r.rtt_m), ("rssi_v", r.rssi_v),
                             ("rssi_present", int(r.rssi_dbm is not None)),
                             ("rssi_dbm", 0 if r.rssi_dbm is None else r.rssi_dbm), ("shadow", r.shadow_db)):
                    a[k].append(x)
            assert [x["node"] for x in rec["reports"]] == [n[0] for n in nodes]
            for m, meth in (("rtt", run.rtt), ("rssi", run.rssi)):
                fx, st = meth.last_fix, meth.last_step
                a[f"{m}_fix_reason"].append(_reason_code(fx.reason))
                for k in ("x", "y", "inv00", "inv01", "inv11", "r00", "r01", "r11", "chi2"):
                    x = getattr(fx, k)
                    a[f"{m}_fix_{k}"].append(0.0 if x is None else x)
                a[f"{m}_fix_iter"].append(0 if fx.iterations is None else fx.iterations)
                a[f"{m}_kf_reason"].append(_reason_code(st.reason))
                for k in ("x", "y", "p00", "p01", "p11"):
                    x = getattr(st, k)
                    a[f"{m}_kf_{k}"].append(0.0 if x is None else x)
                a[f"{m}_kf_started"].append(int(st.started))
                a[f"{m}_kf_discarded"].append(int(st.discarded))
                row = _est_row(rec[m])
                for k, x in zip(pub_keys, row):
                    a[f"{m}_{k}"].append(x)
        if v == "c":
            _check_phone_run_edges(a, nn)
        arr.update({f"{v}_{k}": x for k, x in a.items()})
        c = {"offset_m": "rule", "device_db": "rule", "ep_j": "exact", "ep_t": "exact", "blocked": "exact",
             "d": f"rule:{nn}", "u_fail": "exact", "g": "rule", "u_out": "exact", "u_mag": "exact", "u_b": "exact",
             "u_fail_s": "exact", "g_s": "rule", "u_j": "exact", "rtt_value": f"rule:{nn}", "rtt_present": "exact",
             "rtt_m": f"rule:{nn}", "rssi_v": f"rule:{nn}", "rssi_present": "exact",
             "rssi_dbm": f"tie:{v}_rssi_v", "shadow": "rule"}
        for m in ("rtt", "rssi"):
            c.update({f"{m}_fix_reason": "exact", f"{m}_fix_x": "rule", f"{m}_fix_y": "rule",
                      f"{m}_fix_iter": "exact", f"{m}_kf_reason": "exact", f"{m}_kf_x": "rule", f"{m}_kf_y": "rule",
                      f"{m}_kf_started": "exact", f"{m}_kf_discarded": "exact", f"{m}_reason": "exact",
                      f"{m}_x": "rule", f"{m}_y": "rule"})
            for k in ("fix_inv00", "fix_inv01", "fix_inv11", "fix_r00", "fix_r01", "fix_r11", "fix_chi2", "kf_p00",
                      "kf_p01", "kf_p11", "radius", "cov_xx", "cov_xy", "cov_yy"):
                c[f"{m}_{k}"] = "rule:each"
        cmp.update({f"{v}_{k}": x for k, x in c.items()})
        sc.update(_locate_cfg_scalars(LocateConfig(**kw), f"{v}_cfg."))
        sc.update({f"{v}_phone_seed": cfg.phone_seed, f"{v}_rssi_p0_dbm": cfg.rssi_p0_dbm, f"{v}_pl0": run.radio.pl0,
                   f"{v}_n_epochs": len(a["ep_j"])})
    clk = EpochClock(5.0)
    ct = [0.0, 0.1, 0.2, 0.25, 0.9, 0.95, 1.0, 1.0, math.nan, math.inf, 1.19, 1.2, 1.4000000000000001, 7.0,
          1e7, 1e7, 10000000.2, 12345678.9]
    fired, en = [], []
    for t in ct:
        fired.append(int(clk.fires(t)))
        en.append(clk.e_next)
    arr.update({"clk_t": ct, "clk_fired": fired, "clk_e_next": en})
    cmp.update({"clk_t": "input", "clk_fired": "exact", "clk_e_next": "exact"})
    _lemit("locate_phone_run", sc, arr, cmp)


#: locate_phone_fix's node sets: 0 the bedroom's 4 nodes (scene), 1 the
#: office's collinear nodes 0, 4, 1 (literal positions), 2 the office's 6,
#: 3 the office's collinear nodes 3, 5, 2, 4 three nodes on the slanted
#: line y = 0.3 x + 0.61 (as doubles their det is 4.5e-13, not 0).
LOCATE_FIX_OFFICE = ((0, 0.2, 0.2, 1.5), (1, 5.8, 0.2, 1.5), (2, 5.8, 4.8, 1.5), (3, 0.2, 4.8, 1.5),
                     (4, 3.0, 0.2, 1.5), (5, 3.0, 4.8, 1.5))
LOCATE_FIX_SLANTED = ((0, 0.3, 0.7, 1.5), (1, 2.3, 1.3, 1.5), (2, 5.3, 2.2, 1.5))
# The office's nodes 0, 4, 1 with two y's written 0.1 + 0.2 (one ulp above
# 0.3): collinear to rounding on a horizontal line.
LOCATE_FIX_OFFICE_ULP = ((0, 0.2, 0.1 + 0.2, 1.5), (4, 3.0, 0.1 + 0.2, 1.5), (1, 5.8, 0.3, 1.5))


def gen_locate_phone_fix() -> None:
    """spec/locate.md §7.1-7.2: one epoch's RTT and RSSI fixes
    (`rtt_fix`, `rssi_fix`), each case on its own configuration.

    Node sets (`ns_n` nodes each; `ns_id`, `ns_x`, `ns_y`, `ns_z`, input):
    0 the bedroom's 4 nodes (scene below); 1 the office's nodes 0, 4, 1
    (collinear on y = 0.2); 2 the office's 6 nodes; 3 the office's nodes
    3, 5, 2 (collinear on y = 4.8); 4 three nodes on the slanted line
    y = 0.3 x + 0.61, (0.3, 0.7), (2.3, 1.3), (5.3, 2.2), z 1.5, whose
    anchor det is rounding noise (4.5e-13), not 0; 5 three office-like
    nodes, ids 0, 4, 1 at (0.2, 0.1 + 0.2), (3.0, 0.1 + 0.2), (5.8, 0.3),
    z 1.5: on the line y = 0.3 but for one ulp (anchor n11 1.2e-32, det
    3.9e-31). Sets 1 to 3 are spec/scenes/office.json's positions and set
    5 is made from them, written into generate.py, so they carry no scene
    digest.

    Cases (input): `method` (0 rtt, 1 rssi), `ns`, `zh`
    (phone_assumed_z_m), `sigma` (rtt_sigma_m for RTT,
    rssi_shadow_sigma_db for RSSI), `huber_k`, `p0` (rssi_p0_dbm),
    `n_exp` (rssi_est_exponent); reports `n_rep` per case, `rep_node`,
    `rep_value` (case-major, in the order given: the fix sorts them).
    Outputs (Fix): `reason` (-1 a fix), `x`, `y`, `inv00`, `inv01`,
    `inv11` (inv(N) of the final pass), `r00`, `r01`, `r11` (R), `chi2`
    (RTT; 0.0 for RSSI), `iterations` (0 without a fix); 0.0 when there is
    no fix.

    RTT cases, in order: noiseless ranges to (1.7, 2.1, 1.2) with zh 1.2;
    to (3.2, 0.9, 1.0) with zh 1.0; the first with node 2's range +5 m
    (one-sided Huber, k 1.345) and the same with k 1e9 (least squares;
    scalars `huber_shift_m`, `ls_shift_m`: each fix's distance from the
    truth); no reports; one; two; the collinear set (degenerate); the
    reports (0, 0.0), (1, 2.0), (2, 1.0), (3, 2.0) at zh 1.0 (Gauss-Newton
    zig-zags: 97 iterations, a fix within §7.1's 200, not_converged under
    the old 50); the first case's reports reversed; a range shorter than
    |dz| (h = 0 at the start); 8 recorded noisy cases on node set 2.
    RSSI cases: noiseless float values from the estimator's own model at
    (1.7, 2.1) and (3.0, 1.0) (zh 1.2); the first rounded half-even; no
    reports; two; collinear; -100 and 0 dBm extremes; 8 recorded noisy
    integer cases on node set 2.

    Then the §7 collinear rule and the §7.1-7.2 limits, appended (the
    cases above keep their indices). The collinear sweep, RTT first then
    RSSI, each over node set 1 then node set 3: the truths
    (0.5 i, 0.5 j) for i = 1..11 (outer), j = 1..9 (inner), RTT with
    noiseless ranges from z 1.0 at zh 1.0, RSSI with the model's values
    at zh 1.2 rounded half-even; every one is `degenerate`. Then, RTT
    (zh 1.0): node set 1 at (2.9, 4.5) and (2.9, 4.9), where the old
    centroid start published a mirror fix 8.6 and 9.4 m off; node set 4
    at (2.0, 3.0), (4.0, 0.5), (1.0, 2.5) (degenerate); the bedroom
    reports (0, 0.0), (1, 0.0), (2, 0.5), (3, 5.0) (288 iterations:
    not_converged). RSSI (zh 1.2, rounded): node set 1 at (3.0, 3.8) and
    node set 3 at (3.0, 3.9), where the old start published fixes 7.0 and
    1.8 m off; node set 4 at the same three truths (degenerate); the
    bedroom reports (0, -30), (1, -30), (2, -50), (3, -60) at zh 1.0 (645
    iterations: not_converged); the start guard on the bedroom: every
    report -4000 dBm (each weight 10^-400 underflows to 0: the sum is not
    > 0) and node 0 at +4000 dBm with the rest -50 (10^400 overflows: the
    sum is not finite), both `degenerate`. The generator refuses to write
    the set if any of these outcomes changes.

    Then §7's collinear rule at any line direction, appended: node set 5
    with the truth (2.9, 4.5, 1.0) at zh 1.0: RTT noiseless ranges; RSSI
    the model's float values; RSSI those rounded half-even. All three are
    `degenerate`. The earlier rule `not (det > 1e-9 * (n00*n11))` let
    this set through (n11 is itself rounding-sized), and RTT then
    published the mirror fix (2.9, -3.9) and the rounded RSSI one near
    (3.0, -3.87), each 8.4 m off (the float RSSI did not converge). The
    generator refuses to write the set unless the earlier rule would have
    accepted these anchors and the current one gives `degenerate`.
    """
    from wifisense.localize.config import LocateConfig, default_rssi_p0_dbm
    from wifisense.localize.phone import _anchor_normal, rssi_fix, rtt_fix
    from wifisense.localize.pipeline import phone_nodes
    from wifisense.roomsim import load_scene
    from wifisense.roomsim.numerics import carrier_hz

    rel = "spec/scenes/bedroom.json"
    bed = phone_nodes(load_scene(os.path.join(LOCATE_REPO, rel)))
    sets = [bed, [LOCATE_FIX_OFFICE[i] for i in (0, 4, 1)], list(LOCATE_FIX_OFFICE),
            [LOCATE_FIX_OFFICE[i] for i in (3, 5, 2)], list(LOCATE_FIX_SLANTED), list(LOCATE_FIX_OFFICE_ULP)]
    p0 = default_rssi_p0_dbm(10.0, carrier_hz(6))

    def ranges(ns, x, y, z):
        return [(n[0], math.sqrt(((x - n[1]) * (x - n[1]) + (y - n[2]) * (y - n[2])) + (z - n[3]) * (z - n[3])))
                for n in sets[ns]]

    def model(ns, x, y, zh, nexp=2.0):
        out = []
        for n in sets[ns]:
            dx, dy, dz = x - n[1], y - n[2], zh - n[3]
            rho = math.sqrt((dx * dx + dy * dy) + dz * dz)
            out.append((n[0], p0 - (10.0 * nexp) * math.log10(rho)))
        return out

    R = dict(zh=1.0, sigma=1.0, k=1.345, p0=p0, n=2.0)
    S = dict(zh=1.2, sigma=4.0, k=1.345, p0=p0, n=2.0)
    base = ranges(0, 1.7, 2.1, 1.2)
    long5 = [(i, r + 5.0 if i == 2 else r) for i, r in base]
    cases = [
        (0, 0, base, dict(R, zh=1.2)),
        (0, 0, ranges(0, 3.2, 0.9, 1.0), R),
        (0, 0, long5, dict(R, zh=1.2)),
        (0, 0, long5, dict(R, zh=1.2, k=1e9)),
        (0, 0, [], R),
        (0, 0, base[:1], R),
        (0, 0, base[:2], R),
        (0, 1, ranges(1, 2.0, 2.5, 1.0), R),
        (0, 0, [(0, 0.0), (1, 2.0), (2, 1.0), (3, 2.0)], R),
        (0, 0, list(reversed(base)), dict(R, zh=1.2)),
        (0, 0, [(0, 0.5), (1, 3.9), (2, 4.6), (3, 2.8)], dict(R, zh=0.0)),
    ]
    rnd = _rng_draws(330040, 8 * 3, 0.0, 1.0)
    g = _rng_gauss(330041, 8 * 6 * 2)
    for c in range(8):
        tx, ty, tz = 0.5 + 5.0 * rnd[3 * c], 0.5 + 4.0 * rnd[3 * c + 1], 0.8 + 0.8 * rnd[3 * c + 2]
        rr = [(i, max(0.0, r + g[6 * c + i] + (4.0 if (i + c) % 5 == 0 else 0.0))) for i, r in ranges(2, tx, ty, tz)]
        cases.append((0, 2, rr, dict(R, k=(1.345, 2.0)[c % 2])))
    sm = model(0, 1.7, 2.1, 1.2)
    cases += [
        (1, 0, sm, S),
        (1, 0, model(0, 3.0, 1.0, 1.2), S),
        (1, 0, [(i, float(round(v))) for i, v in sm], S),
        (1, 0, [], S),
        (1, 0, sm[:2], S),
        (1, 1, model(1, 2.0, 2.5, 1.2), S),
        (1, 0, [(0, -100.0), (1, 0.0), (2, -100.0), (3, -50.0)], S),
    ]
    for c in range(8):
        tx, ty = 0.5 + 5.0 * rnd[3 * c + 1], 0.5 + 4.0 * rnd[3 * c]
        vals = [(i, float(round(v + 4.0 * g[48 + 6 * c + i]))) for i, v in model(2, tx, ty, 1.0)]
        cases.append((1, 2, vals, dict(S, zh=1.0, n=(2.0, 2.5)[c % 2])))
    # Appended: §7's collinear rule and the §7.1-7.2 limits (docstring).
    n_before_r1 = len(cases)
    grid = [(0.5 * i, 0.5 * j) for i in range(1, 12) for j in range(1, 10)]
    for ns in (1, 3):
        cases += [(0, ns, ranges(ns, x, y, 1.0), R) for x, y in grid]
    for ns in (1, 3):
        cases += [(1, ns, [(i, float(round(v))) for i, v in model(ns, x, y, 1.2)], S) for x, y in grid]
    slanted = ((2.0, 3.0), (4.0, 0.5), (1.0, 2.5))
    cases += [(0, 1, ranges(1, 2.9, 4.5, 1.0), R), (0, 1, ranges(1, 2.9, 4.9, 1.0), R)]
    cases += [(0, 4, ranges(4, x, y, 1.0), R) for x, y in slanted]
    cases += [(0, 0, [(0, 0.0), (1, 0.0), (2, 0.5), (3, 5.0)], R)]
    cases += [(1, 1, [(i, float(round(v))) for i, v in model(1, 3.0, 3.8, 1.2)], S),
              (1, 3, [(i, float(round(v))) for i, v in model(3, 3.0, 3.9, 1.2)], S)]
    cases += [(1, 4, [(i, float(round(v))) for i, v in model(4, x, y, 1.2)], S) for x, y in slanted]
    cases += [(1, 0, [(0, -30.0), (1, -30.0), (2, -50.0), (3, -60.0)], dict(S, zh=1.0)),
              (1, 0, [(0, -4000.0), (1, -4000.0), (2, -4000.0), (3, -4000.0)], S),
              (1, 0, [(0, 4000.0), (1, -50.0), (2, -50.0), (3, -50.0)], S)]
    # Appended: the collinear rule at any line direction (docstring).
    n_before_r2 = len(cases)
    ulp_rssi = model(5, 2.9, 4.5, 1.0)
    cases += [(0, 5, ranges(5, 2.9, 4.5, 1.0), R), (1, 5, ulp_rssi, dict(S, zh=1.0)),
              (1, 5, [(i, float(round(v))) for i, v in ulp_rssi], dict(S, zh=1.0))]
    keys_in = ("method", "ns", "zh", "sigma", "huber_k", "p0", "n_exp", "n_rep", "rep_node", "rep_value")
    keys_out = ("reason", "x", "y", "inv00", "inv01", "inv11", "r00", "r01", "r11", "chi2", "iterations")
    arr: dict = {k: [] for k in keys_in + keys_out}
    fixes = []
    for meth, ns, reps, o in cases:
        if meth == 0:
            cfg = LocateConfig(rtt_sigma_m=o["sigma"], huber_k=o["k"], phone_assumed_z_m=o["zh"])
            fx = rtt_fix(reps, sets[ns], cfg)
        else:
            cfg = LocateConfig(rssi_shadow_sigma_db=o["sigma"], huber_k=o["k"], phone_assumed_z_m=o["zh"],
                               rssi_p0_dbm=o["p0"], rssi_est_exponent=o["n"])
            fx = rssi_fix(reps, sets[ns], cfg)
        fixes.append(fx)
        for k, v in (("method", meth), ("ns", ns), ("zh", o["zh"]), ("sigma", o["sigma"]), ("huber_k", o["k"]),
                     ("p0", o["p0"]), ("n_exp", o["n"]), ("n_rep", len(reps)), ("reason", _reason_code(fx.reason)),
                     ("iterations", 0 if fx.iterations is None else fx.iterations)):
            arr[k].append(v)
        arr["rep_node"].extend(r[0] for r in reps)
        arr["rep_value"].extend(r[1] for r in reps)
        for k in ("x", "y", "inv00", "inv01", "inv11", "r00", "r01", "r11", "chi2"):
            v = getattr(fx, k)
            arr[k].append(0.0 if v is None else v)
    for k, ns in (("ns_n", None), ("ns_id", 0), ("ns_x", 1), ("ns_y", 2), ("ns_z", 3)):
        arr[k] = [len(s) for s in sets] if ns is None else [n[ns] for s in sets for n in s]
    # The pins this set exists for: refuse to write it if one is lost.
    assert fixes[8].ok and fixes[8].iterations == 97, fixes[8]
    for (meth, ns, reps, _o), fx in zip(cases, fixes):
        if ns in (1, 3, 4, 5) and len(reps) >= 3:
            assert fx.reason == "degenerate", (meth, ns, reps, fx)
    tail = [fx.reason for fx in fixes[n_before_r1 + 4 * len(grid):n_before_r2]]
    assert tail == ["degenerate"] * 5 + ["not_converged"] + ["degenerate"] * 5 + ["not_converged"] + \
        ["degenerate"] * 2, tail
    a00, _a01, a11, det = _anchor_normal([(*n, 0.0) for n in sorted(sets[5])])
    assert det > 1e-9 * (a00 * a11) and not det > 1e-9 * ((a00 + a11) * (a00 + a11)), (a00, a11, det)
    assert [fx.reason for fx in fixes[n_before_r2:]] == ["degenerate"] * 3, fixes[n_before_r2:]

    def off(fx):
        dx, dy = fx.x - 1.7, fx.y - 2.1
        return math.sqrt(dx * dx + dy * dy)

    sc = _scene_scalars(rel)
    sc.update({"n_cases": len(cases), "huber_shift_m": off(fixes[2]), "ls_shift_m": off(fixes[3]),
               "n_node_sets": len(sets)})
    cmp = {k: "input" for k in keys_in + ("ns_n", "ns_id", "ns_x", "ns_y", "ns_z")}
    cmp.update({"reason": "exact", "iterations": "exact", "x": "rule", "y": "rule"})
    for k in ("inv00", "inv01", "inv11", "r00", "r01", "r11", "chi2"):
        cmp[k] = "rule:each"
    _lemit("locate_phone_fix", sc, arr, cmp)


def gen_locate_phone_kf() -> None:
    """spec/locate.md §7.3: PhoneKalman (default configuration: q 1.0,
    phone_init_speed_std 1.0, phone_max_coast_s 3.0) over a recorded
    sequence of fixes, one step per row, in order.

    Inputs: `t`; the fix `f_ok` (0: no fix, reason `f_reason`), `f_x`,
    `f_y`, `f_r00`, `f_r01`, `f_r11` (R; 0.0 without a fix). The sequence
    (times dyadic so that t - t_acc is exact): start; two updates; a fix
    10 sigma off (gated, predicted state kept); a no-fix epoch (nothing
    predicted); an update predicting from the gated epoch's t_f; a fix
    with d2 = CHI2_2_99 * (1 - 1e-6) (accepted) and one with
    CHI2_2_99 * (1 + 1e-6) (gated); no-fix epochs up to t - t_acc = 3.0
    exactly (kept); a fix at 3.25 (discarded, then restarted by it); a run
    of far-off fixes ending in expiry and a restart; a no-fix epoch that
    discards; a start; R with a negative entry (det(S) <= 0: gated); a NaN
    fix (gated); an update.

    Outputs per step (KalmanStep): `reason` (-1 accepted or started),
    `x`, `y`, `p00`, `p01`, `p11`, `started`, `discarded`; the filter after
    the step: `exists`, `s` (4 per step: x, y, vx, vy), `P` (16 per step,
    row-major), `t_f`, `t_acc` (0.0 when it does not exist).
    """
    import copy

    from wifisense.localize.config import LocateConfig
    from wifisense.localize.phone import Fix, PhoneKalman
    from wifisense.localize.radius import CHI2_2_99

    cfg = LocateConfig()
    kf = PhoneKalman(cfg)

    def fix(x, y, r00=0.25, r01=0.02, r11=0.3):
        return Fix(None, x, y, 0.0, 0.0, 0.0, r00, r01, r11, None, 1)

    def gate_fix(t, factor):
        probe = copy.deepcopy(kf)
        probe.step(t, fix(-1e6, -1e6))                 # gated: leaves the prediction in `probe`
        S00 = probe.P[0][0] + 0.25
        S01 = probe.P[0][1] + 0.02
        S11 = probe.P[1][1] + 0.3
        det = S00 * S11 - S01 * S01
        si00 = S11 / det
        nu = math.sqrt(CHI2_2_99 * factor / si00)
        return fix(probe.s[0] + nu, probe.s[1])

    nofix = Fix("too_few_ranges")
    seq = [(0.0, fix(1.0, 1.0)), (0.25, fix(1.1, 1.0)), (0.5, fix(1.2, 1.05)), (0.75, fix(6.0, 1.0)),
           (1.0, Fix("degenerate")), (1.25, fix(1.4, 1.1))]
    rows = []

    def run(t, fx):
        st = kf.step(t, fx)
        rows.append((t, fx, st, kf.exists, list(kf.s) if kf.exists else [0.0] * 4,
                     [v for r in kf.P for v in r] if kf.exists else [0.0] * 16,
                     kf.t_f if kf.exists else 0.0, kf.t_acc if kf.exists else 0.0))

    for t, fx in seq:
        run(t, fx)
    run(1.5, gate_fix(1.5, 1.0 - 1e-6))
    run(1.75, gate_fix(1.75, 1.0 + 1e-6))
    for t in (2.0, 3.0, 4.5):
        run(t, nofix)
    run(4.75, fix(2.0, 2.0))
    run(5.0, fix(2.1, 2.0))
    for t in (5.5, 6.5, 7.5, 8.0):
        run(t, fix(40.0, 40.0, 0.01, 0.0, 0.01))
    run(8.25, fix(40.0, 40.0))
    run(8.5, fix(40.05, 40.0))
    run(12.0, Fix("not_converged"))
    run(12.25, fix(3.0, 3.0))
    run(12.5, fix(3.0, 3.0, -10.0, 0.0, 1.0))
    run(12.75, fix(math.nan, 3.0))
    run(13.0, fix(3.1, 3.05))
    arr: dict = {k: [] for k in ("t", "f_ok", "f_reason", "f_x", "f_y", "f_r00", "f_r01", "f_r11", "reason", "x",
                                 "y", "p00", "p01", "p11", "started", "discarded", "exists", "s", "P", "t_f",
                                 "t_acc")}
    for t, fx, st, ex, s, P, tf, ta in rows:
        for k, v in (("t", t), ("f_ok", int(fx.ok)), ("f_reason", _reason_code(fx.reason)),
                     ("f_x", fx.x if fx.ok else 0.0), ("f_y", fx.y if fx.ok else 0.0),
                     ("f_r00", fx.r00 if fx.ok else 0.0), ("f_r01", fx.r01 if fx.ok else 0.0),
                     ("f_r11", fx.r11 if fx.ok else 0.0), ("reason", _reason_code(st.reason)),
                     ("started", int(st.started)), ("discarded", int(st.discarded)), ("exists", int(ex)),
                     ("t_f", tf), ("t_acc", ta)):
            arr[k].append(v)
        for k in ("x", "y", "p00", "p01", "p11"):
            v = getattr(st, k)
            arr[k].append(0.0 if v is None else v)
        arr["s"].extend(s)
        arr["P"].extend(P)
    cmp = {k: "input" for k in ("t", "f_ok", "f_reason", "f_x", "f_y", "f_r00", "f_r01", "f_r11")}
    cmp.update({"reason": "exact", "started": "exact", "discarded": "exact", "exists": "exact", "t_f": "exact",
                "t_acc": "exact", "x": "rule", "y": "rule", "p00": "rule:each", "p01": "rule:each",
                "p11": "rule:each", "s": "rule:4", "P": "rule:16"})
    _lemit("locate_phone_kf", {"n_steps": len(rows), "phone_accel_psd": cfg.phone_accel_psd,
                               "phone_init_speed_std": cfg.phone_init_speed_std,
                               "phone_max_coast_s": cfg.phone_max_coast_s, "CHI2_2_99": CHI2_2_99}, arr, cmp)


def _locate_capture_run(scene, rs_cfg):
    """run_locate(scene, rs_cfg) with the person and phone pipelines
    recording what each tick and epoch used: (run, person pipeline,
    per-tick (record, y, image, tracking), per-epoch (record, reports,
    rtt fix, rtt step, rssi fix, rssi step))."""
    from wifisense.localize import pipeline as pl

    cap: dict = {"person": None, "ticks": [], "epochs": []}

    class RecordingPerson(pl.PersonPipeline):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            cap["person"] = self
            cur: dict = {}
            vec, solve = self.meas.vector, self.inverter.solve

            def vector(t=None):
                out = vec(t)
                cur["y"] = list(out)
                return out

            def solve_(y):
                out = solve(y)
                cur["image"] = list(out)
                return out

            self.meas.vector = vector
            self.inverter.solve = solve_
            self._cur = cur

        def tick(self, t_k, truth):
            self._cur.clear()
            rec = super().tick(t_k, truth)
            cap["ticks"].append((rec, self._cur["y"], self._cur["image"], self.tracker.is_tracking))
            return rec

    class RecordingPhone(pl.PhoneRun):
        def sample(self, t, phone, cylinders):
            rec = super().sample(t, phone, cylinders)
            if rec is not None:
                cap["epochs"].append((rec, list(self.last_reports), self.rtt.last_fix, self.rtt.last_step,
                                      self.rssi.last_fix, self.rssi.last_step))
            return rec

    saved = (pl.PersonPipeline, pl.PhoneRun)
    pl.PersonPipeline, pl.PhoneRun = RecordingPerson, RecordingPhone
    try:
        run = pl.run_locate(scene, rs_cfg)
    finally:
        pl.PersonPipeline, pl.PhoneRun = saved
    assert [c[0] for c in cap["ticks"]] == run.ticks and [c[0] for c in cap["epochs"]] == run.epochs
    return run, cap["person"], cap["ticks"], cap["epochs"]


def _gen_locate_pipeline(name: str, rel: str, rs_kw: dict) -> None:
    from wifisense.localize.config import LocateConfig
    from wifisense.localize.track import extract_peak
    from wifisense.roomsim import Config, RoomSim, load_scene

    scene = load_scene(os.path.join(LOCATE_REPO, rel))
    rs_cfg = Config(**rs_kw)
    run, person, ticks, epochs = _locate_capture_run(scene, rs_cfg)
    geom = person.geometry
    cfg = person.cfg
    L, P = geom.n_links, geom.n_pixels
    n_people = len(RoomSim(scene, Config(**dict(rs_kw, duration_s=0.0))).people_template)
    arr: dict = {"link_tx": [lk.tx_id for lk in geom.links], "link_rx": [lk.rx_id for lk in geom.links],
                 "baseline": person.meas.baseline_vector(), "null_sigma": person.meas.baseline_std_vector(),
                 "calibrated_mask": [int(b) for b in person.meas.calibrated_mask()],
                 "blind_pixels": list(person.inverter.blind_pixels)}
    for k in ("tk_t", "tk_y", "tk_image", "tk_z", "tk_tracking", "tk_p_x", "tk_p_y", "tk_p_walking",
              "tk_p_posture"):
        arr[k] = []
    for e in ("wlm", "rti"):
        for f in EST_FIELDS:
            arr[f"tk_{e}_{f}"] = []
    for rec, y, image, tracking in ticks:
        assert rec["label"] == "SIM"
        arr["tk_t"].append(rec["t"])
        arr["tk_y"].extend(y)
        arr["tk_image"].extend(image)
        arr["tk_z"].append(extract_peak(geom, image).z)
        arr["tk_tracking"].append(int(tracking))
        for e in ("wlm", "rti"):
            for f, v in zip(EST_FIELDS, _est_row(rec[e])):
                arr[f"tk_{e}_{f}"].append(v)
        people = rec["truth"]["people"]
        assert len(people) == n_people
        for p in people:
            arr["tk_p_x"].append(p["x"])
            arr["tk_p_y"].append(p["y"])
            arr["tk_p_walking"].append(int(p["walking"]))
            arr["tk_p_posture"].append(POSTURE_CODE[p["posture"]])
    nn = len(scene.nodes)
    for k in ("ep_t", "ep_x", "ep_y", "ep_z", "rp_node", "rp_blocked", "rp_rtt_present", "rp_rtt_m",
              "rp_rssi_present", "rp_rssi_dbm", "rp_rssi_v"):
        arr[k] = []
    for m in ("rtt", "rssi"):
        for f in EST_FIELDS[:7]:
            arr[f"ep_{m}_{f}"] = []
        arr[f"ep_{m}_fix_reason"] = []
        arr[f"ep_{m}_kf_reason"] = []
    for rec, reports, rfix, rstep, sfix, sstep in epochs:
        assert rec["label"] == "SIM"
        arr["ep_t"].append(rec["t"])
        for c in ("x", "y", "z"):
            arr["ep_" + c].append(rec["truth"][c])
        for r, nr in zip(rec["reports"], reports):
            assert r["node"] == nr.node and r["rssi_dbm"] == nr.rssi_dbm and r["rtt_m"] == nr.rtt_m
            arr["rp_node"].append(r["node"])
            arr["rp_blocked"].append(int(r["blocked"]))
            arr["rp_rtt_present"].append(int(r["rtt_m"] is not None))
            arr["rp_rtt_m"].append(0.0 if r["rtt_m"] is None else r["rtt_m"])
            arr["rp_rssi_present"].append(int(r["rssi_dbm"] is not None))
            arr["rp_rssi_dbm"].append(0 if r["rssi_dbm"] is None else r["rssi_dbm"])
            arr["rp_rssi_v"].append(nr.rssi_v)
        for m, fx, st in (("rtt", rfix, rstep), ("rssi", sfix, sstep)):
            for f, v in zip(EST_FIELDS[:7], _est_row(rec[m])):
                arr[f"ep_{m}_{f}"].append(v)
            arr[f"ep_{m}_fix_reason"].append(_reason_code(fx.reason))
            arr[f"ep_{m}_kf_reason"].append(_reason_code(st.reason))
    cmp = {"link_tx": "exact", "link_rx": "exact", "baseline": "rule", "null_sigma": "rule",
           "calibrated_mask": "exact", "blind_pixels": "exact", "tk_t": "exact", "tk_y": f"rule:{L}",
           "tk_image": f"rule:{P}", "tk_z": "rule:each", "tk_tracking": "exact", "tk_p_x": "rule", "tk_p_y": "rule",
           "tk_p_walking": "exact", "tk_p_posture": "exact", "ep_t": "rule", "ep_x": "rule", "ep_y": "rule",
           "ep_z": "rule", "rp_node": "exact", "rp_blocked": "exact", "rp_rtt_present": "exact",
           "rp_rtt_m": f"rule:{nn}", "rp_rssi_present": "exact", "rp_rssi_dbm": "tie:rp_rssi_v",
           "rp_rssi_v": f"rule:{nn}"}
    for e in ("wlm", "rti"):
        for f in EST_FIELDS:
            cmp[f"tk_{e}_{f}"] = EST_CMP[f]
    for m in ("rtt", "rssi"):
        for f in EST_FIELDS[:7]:
            cmp[f"ep_{m}_{f}"] = EST_CMP[f]
        cmp[f"ep_{m}_fix_reason"] = "exact"
        cmp[f"ep_{m}_kf_reason"] = "exact"
    d = run.diagnostics
    sc = _scene_scalars(rel)
    sc.update({"seed": rs_cfg.seed, "duration_s": rs_cfg.duration_s})
    sc.update(_roomsim_cfg_scalars(rs_cfg))
    sc.update(_locate_cfg_scalars(LocateConfig()))
    sc.update({"person_label": "SIM", "calib_s": cfg.calib_s, "window_s": cfg.window_s, "tick_hz": cfg.tick_hz,
               "link_signal": cfg.link_signal, "phone_seed": cfg.phone_seed, "tx_power_dbm": cfg.tx_power_dbm,
               "carrier_hz": cfg.carrier_hz, "rssi_p0_dbm": cfg.rssi_p0_dbm, "wlm_floor": cfg.wlm_floor,
               "rate_hz": d["rate_hz"], "min_samples": d["min_samples"], "n_links": L, "n_pixels": P,
               "n_x": geom.grid.n_x, "n_y": geom.grid.n_y, "n_blind": d["blind_pixels"], "n_ticks": d["n_ticks"],
               "n_epochs": d["n_epochs"], "has_phone": int(d["has_phone"]), "n_people": n_people, "n_nodes": nn,
               "n_independent_links": person.inverter.n_independent_links})
    _lemit(name, sc, arr, cmp)


#: The pipeline sets' Room Sim configurations (the rest default; seed 0).
LOCATE_PIPELINE_RUNS = (
    ("locate_pipeline_bedroom", "spec/scenes/bedroom.json", dict(duration_s=10.0, n_people=2)),
    ("locate_pipeline_office", "spec/scenes/office.json", dict(duration_s=10.0, n_people=3)),
)


def gen_locate_pipeline_bedroom() -> None:
    """spec/locate.md §10-§11: `run_locate(scene, Config(**cfg.*))` with
    the default LocateConfig and label "SIM": the empty-room calibration
    run (same seed, `duration_s + calib_s`), then 10 s occupied, every
    tick record and every phone epoch record. Here
    spec/scenes/bedroom.json with `n_people` 2: the sleeper lying on the
    bed (it carries the phone) and people[1] walking in from the door.

    Scalars: `scene`, `scene_sha256`, `seed`, `duration_s`, `cfg.*` (Room
    Sim's non-default values, the roomsim convention), `locate.*` (none:
    the default LocateConfig), the resolved values `calib_s`, `window_s`,
    `tick_hz`, `link_signal`, `phone_seed`, `tx_power_dbm`, `carrier_hz`,
    `rssi_p0_dbm`, `wlm_floor`; the run's `rate_hz`, `min_samples`,
    `n_links`, `n_pixels`, `n_x`, `n_y`, `n_blind`, `n_independent_links`,
    `n_ticks`, `n_epochs`, `has_phone`, `n_people`, `n_nodes`.

    Calibration: `link_tx`, `link_rx` (canonical order), `baseline`,
    `null_sigma` (baseline_std_vector), `calibrated_mask`; `blind_pixels`.

    Ticks (t_k = window_s + k / tick_hz): `tk_t`; the inputs the tick used,
    `tk_y` (meas.vector(t_k), n_links per tick) and `tk_image`
    (inverter.solve(y), n_pixels per tick); `tk_z` (extract_peak z of the
    image) and `tk_tracking` (tracker.is_tracking after the tick), for
    reading the trace; WLM `tk_wlm_reason`, `tk_wlm_x`, `tk_wlm_y`,
    `tk_wlm_radius`, `tk_wlm_cov_xx`, `tk_wlm_cov_xy`, `tk_wlm_cov_yy`,
    `tk_wlm_confidence` (always 0.0: None); RTI the same with `tk_rti_`
    (`tk_rti_confidence` the tracker's); truth, tick-major (`n_people`
    each, index order): `tk_p_x`, `tk_p_y`, `tk_p_walking`,
    `tk_p_posture`.

    Phone epochs: `ep_t`, the truth `ep_x`, `ep_y`, `ep_z`; reports,
    epoch-major (`n_nodes` each, id order): `rp_node`, `rp_blocked`,
    `rp_rtt_present`, `rp_rtt_m`, `rp_rssi_present`, `rp_rssi_dbm`
    (`tie:rp_rssi_v`), `rp_rssi_v` (the pre-rounding RSSI, for the §12
    tie rule); per method M (rtt, rssi): `ep_M_reason`, `ep_M_x`,
    `ep_M_y`, `ep_M_radius`, `ep_M_cov_xx`, `ep_M_cov_xy`, `ep_M_cov_yy`,
    and for diagnosis `ep_M_fix_reason`, `ep_M_kf_reason`.
    """
    _gen_locate_pipeline(*LOCATE_PIPELINE_RUNS[0])


def gen_locate_pipeline_office() -> None:
    """spec/scenes/office.json with `n_people` 3: the two seated occupants
    (people[0] carries the phone) and people[2] walking in from the door;
    otherwise as locate_pipeline_bedroom, whose layout this set shares."""
    _gen_locate_pipeline(*LOCATE_PIPELINE_RUNS[1])


def main() -> None:
    # The committed vectors are CPython 3.14's (spec/locate.md §12): the
    # localize_ sets depend on 3.12+'s Neumaier-compensated builtin sum()
    # (§1 A), and other versions rewrite other sets (sim_geometric differs
    # on 3.12). Any other interpreter would delete every vector and write
    # different bytes, so it is refused before anything is touched.
    if sys.implementation.name != "cpython" or sys.version_info[:2] != (3, 14):
        sys.exit("generate.py must run on CPython 3.14 (the committed vectors are 3.14's): "
                 "$(.venv/bin/uv python find 3.14) -I spec/golden/generate.py")
    for f in os.listdir(HERE):
        if f.endswith((".kv", ".f64", ".json")):
            os.remove(os.path.join(HERE, f))
    print("golden vectors ->", HERE)
    gen_biquad()
    gen_goertzel()
    gen_band_peak()
    gen_sideband()
    gen_fft_bnr()
    gen_static_suppressor()
    gen_presence()
    gen_extractors()
    gen_gapfill()
    gen_codec()
    gen_features()
    gen_sim_geometric()
    gen_roomsim_materials()
    gen_roomsim_shadow()
    gen_roomsim_paths()
    gen_roomsim_paths_hall()
    gen_roomsim_paths_specular()
    gen_roomsim_motion()
    gen_roomsim_channel()
    gen_roomsim_impair()
    gen_roomsim_multi()
    gen_roomsim_switches()
    gen_roomsim_export()
    gen_roomsim_physics_switches()
    gen_localize_py_hypot()
    gen_localize_perimeter()
    gen_localize_weights()
    gen_localize_weights_bedroom()
    gen_localize_weights_office()
    gen_localize_weights_hall()
    gen_localize_prior()
    gen_localize_cholesky()
    gen_localize_inverter_default()
    gen_localize_inverter_aniso()
    gen_localize_inverter_fine()
    gen_localize_rankstarved()
    gen_localize_forward()
    gen_localize_peak()
    gen_localize_tracker()
    gen_localize_measure()
    gen_locate_radius()
    gen_locate_wlm()
    gen_locate_los()
    gen_locate_phone_run()
    gen_locate_phone_fix()
    gen_locate_phone_kf()
    gen_locate_pipeline_bedroom()
    gen_locate_pipeline_office()
    with open(os.path.join(HERE, "MANIFEST"), "w") as fh:
        fh.write("\n".join(MANIFEST) + "\n")
    print(f"{len(MANIFEST)} vector sets")


if __name__ == "__main__":
    main()
