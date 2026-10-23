#!/usr/bin/env python3
"""Generate golden vectors from the Python reference implementation.

    python3 spec/golden/generate.py

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


def main() -> None:
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
    with open(os.path.join(HERE, "MANIFEST"), "w") as fh:
        fh.write("\n".join(MANIFEST) + "\n")
    print(f"{len(MANIFEST)} vector sets")


if __name__ == "__main__":
    main()
