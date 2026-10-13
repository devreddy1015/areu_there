#!/usr/bin/env python3
"""Python reference throughput on the same input as `cargo bench -p saans-dsp`."""

from __future__ import annotations

import os
import struct
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..")))

from wifisense.features import FeatureExtractor  # noqa: E402
from wifisense.types import CsiFrame  # noqa: E402


def f64(name: str) -> list[float]:
    data = open(os.path.join(HERE, name), "rb").read()
    return list(struct.unpack(f"<{len(data) // 8}d", data))


n, fs = 32, 20.0
seq, t, amps = f64("features_vitals__seq_in.f64"), f64("features_vitals__t_in.f64"), f64("features_vitals__amplitudes_in.f64")
frames = [CsiFrame(amps[i * n:(i + 1) * n], [0.0] * n, int(seq[i]), fs) for i in range(len(seq))]
start = time.perf_counter()
fx = FeatureExtractor(n_subcarriers=n, sample_rate_hz=fs, vitals_enabled=True, heart_enabled=True, calibration_s=10.0)
emitted = sum(len(fx.push(f, int(seq[i]), t[i], -92)) for i, f in enumerate(frames))
el = time.perf_counter() - start
fps = len(frames) / el
print(f"python reference FeatureExtractor: {len(frames)} CSI frames ({n} subcarriers, {fs:g} Hz) in {el:.2f} s")
print(f"  {fps:.0f} frames/s = {fps / fs:.1f}x real time for one link; {emitted} FeatureFrames")
