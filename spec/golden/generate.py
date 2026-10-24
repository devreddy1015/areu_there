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
import hashlib  # noqa: E402

from wifisense.features import (  # noqa: E402
    FeatureExtractor,
    FeatureFrame,
    derive_node_key,
    encode_signed,
)
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


def gen_codec_signed() -> None:
    """FeatureFrame v2 (ADR-012): node-key derivation and signed frames,
    byte-exact. The site key is a fixed hash so the vectors are stable;
    per frame, key_id and boot cover 0 and the top of their ranges."""
    site_key = hashlib.sha256(b"saans golden site key").digest()
    frames = [
        FeatureFrame(),
        FeatureFrame(node_id=12, tx_id=3, flags=0b0011_0111, seq=0xFFFFFFFF, uptime_ms=123456789,
                     csi_rate_chz=1987, loss_permille=31, motion_energy=0.15625, presence_threshold=0.05,
                     gross_motion=0.0625, breath_rms=0.2, breathing_bpm=15.25, breathing_conf=0.93,
                     heart_bpm=68.5, heart_conf=0.41, best_bnr=42.0, calib_age_s=3600, rssi_dbm=-61,
                     noise_floor_dbm=-95, n_subcarriers=56),
    ]
    key_ids, boots = [0, 255], [0, 0xFFFFFFFF]
    node_keys = [derive_node_key(site_key, fr.node_id) for fr in frames]
    signed = [encode_signed(fr, node_key=k, key_id=i, boot=b)
              for fr, k, i, b in zip(frames, node_keys, key_ids, boots)]
    arrays = {k: [float(getattr(fr, k)) for fr in frames] for k in FEATURE_FIELDS}
    arrays.update({
        "site_key": list(site_key),
        "node_keys": [b for k in node_keys for b in k],
        "key_id": key_ids,
        "boot": boots,
        "bytes": [b for d in signed for b in d],
    })
    emit("feature_codec_signed", {"frames": len(frames), "frame_size": 84, "tag_size": 16,
                                  "magic": 0x534E4153, "version": 2}, arrays)


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


def gen_sim_geometric_3d() -> None:
    """The geometric scene in a box (room_height_m set), noise-free.

    Exercises what the 2-D set cannot: floor and ceiling bounces with
    different Gammas, order-2 images across all six faces, nodes at two
    heights, a chest below both, furniture at drawn heights.
    """
    cfg = SyntheticConfig(
        scene="geometric", n_subcarriers=56, noise_std=0.0, seed=11, n_scatterers=4,
        reflection_order=2, los_attenuation_db=3.0, wall_reflection=0.45,
        body_points=3, body_width_m=0.3, body_xy_m=(2.6, 3.4), breathing_bpm=14.0,
        room_height_m=2.7, tx_z_m=1.2, rx_z_m=0.8, body_z_m=1.0,
        floor_reflection=0.395, ceiling_reflection=0.247,
    )
    src = SyntheticCsiSource(cfg)
    frames = src.frames(200)
    emit("sim_geometric_3d", {
        "n_subcarriers": 56, "fs": cfg.sample_rate_hz, "carrier_hz": cfg.carrier_hz,
        "subcarrier_spacing_hz": cfg.subcarrier_spacing_hz, "room_w": cfg.room_m[0], "room_h": cfg.room_m[1],
        "room_z": cfg.room_height_m, "tx_x": cfg.tx_xy_m[0], "tx_y": cfg.tx_xy_m[1], "tx_z": cfg.tx_z_m,
        "rx_x": cfg.rx_xy_m[0], "rx_y": cfg.rx_xy_m[1], "rx_z": cfg.rx_z_m,
        "body_x": cfg.body_xy_m[0], "body_y": cfg.body_xy_m[1], "body_z": cfg.body_z_m,
        "body_points": cfg.body_points, "body_width_m": cfg.body_width_m, "body_rcs_m2": cfg.body_rcs_m2,
        "wall_reflection": cfg.wall_reflection, "floor_reflection": cfg.floor_reflection,
        "ceiling_reflection": cfg.ceiling_reflection, "reflection_order": cfg.reflection_order,
        "los_attenuation_db": cfg.los_attenuation_db, "static_magnitude": cfg.static_magnitude,
        "breathing_bpm": cfg.breathing_bpm, "breathing_amplitude_m": cfg.breathing_amplitude_m,
        "heart_bpm": cfg.heart_bpm, "heart_amplitude_m": cfg.heart_amplitude_m,
        "excess_path_m": src.excess_path_m, "n_static_paths": len(src.static_paths),
    }, {
        "scatterer_x": [p[0] for p in src.scatterers], "scatterer_y": [p[1] for p in src.scatterers],
        "scatterer_z": [p[2] for p in src.scatterers], "scatterer_rcs": src.scatterer_rcs,
        "body_gain": src._body_gains,
        "amplitudes": [a for f in frames for a in f.amplitudes],
        "phases": [a for f in frames for a in f.phases],
        "sensitivity": src.sensitivity(),
    })


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
    gen_codec_signed()
    gen_features()
    gen_sim_geometric()
    gen_sim_geometric_3d()
    with open(os.path.join(HERE, "MANIFEST"), "w") as fh:
        fh.write("\n".join(MANIFEST) + "\n")
    print(f"{len(MANIFEST)} vector sets")


if __name__ == "__main__":
    main()
