import cmath
import math
import random
import unittest

from wifisense import SensingPipeline
from wifisense.preprocess import (
    RespirFiFusion,
    StaticSuppressor,
    _fft,
    _percentile,
    fuse,
    respirfi_select,
    symmetric_eigen,
)
from wifisense.sources import SyntheticCsiSource, SyntheticConfig
from wifisense.types import VitalStatus
from wifisense.vitals import BreathingExtractor

RATE = 4.0       # analysis rate the selector sees
F_B = 0.25       # 15 BPM


def window(groups, seconds=32.0, noise=0.1, seed=0):
    """`groups` is a list of per-subcarrier generators f(t) -> float."""
    rng = random.Random(seed)
    n = int(seconds * RATE)
    return [[g(i / RATE) + rng.gauss(0.0, noise) for g in groups] for i in range(n)]


def breath(sign):
    return lambda t: sign * math.sin(2 * math.pi * F_B * t)


def doubled(t):
    """A subcarrier on a Fresnel-zone boundary: only the second-order term
    survives, at twice the breathing rate."""
    return math.cos(2 * math.pi * 2 * F_B * t)


def quiet(t):
    return 0.0


class TestLinearAlgebraHelpers(unittest.TestCase):
    def test_fft_matches_naive_dft(self):
        rng = random.Random(1)
        x = [complex(rng.gauss(0, 1), rng.gauss(0, 1)) for _ in range(16)]
        fast = _fft(x)
        for m in range(16):
            slow = sum(x[n] * cmath.exp(-2j * math.pi * m * n / 16) for n in range(16))
            self.assertAlmostEqual(abs(fast[m] - slow), 0.0, places=9)

    def test_symmetric_eigen_reconstructs_matrix(self):
        rng = random.Random(2)
        n = 7
        a = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i, n):
                a[i][j] = a[j][i] = rng.uniform(-1, 1)
        values, v = symmetric_eigen(a)
        for i in range(n):
            for j in range(n):
                rebuilt = sum(v[i][k] * values[k] * v[j][k] for k in range(n))
                self.assertAlmostEqual(rebuilt, a[i][j], places=9)
                dot = sum(v[k][i] * v[k][j] for k in range(n))
                self.assertAlmostEqual(dot, 1.0 if i == j else 0.0, places=9)

    def test_percentile_matches_linear_interpolation(self):
        self.assertAlmostEqual(_percentile([1, 2, 3, 4, 5], 80), 4.2)
        self.assertAlmostEqual(_percentile([7.0], 80), 7.0)


class TestRespirFiSelect(unittest.TestCase):
    def test_splits_opposite_trends_into_signed_groups(self):
        """Case 1 and Case 2 subcarriers move in opposite directions. They
        must end up in opposite groups, i.e. with opposite signs."""
        groups = [breath(+1)] * 5 + [breath(-1)] * 5
        sel = respirfi_select(window(groups), RATE, bnr_percentile=0.0)
        self.assertIsNotNone(sel)
        self.assertEqual(sel.selected, list(range(10)))
        signs = [sel.signs[k] for k in range(10)]
        self.assertEqual(len(set(signs[:5])), 1)
        self.assertEqual(len(set(signs[5:])), 1)
        self.assertNotEqual(signs[0], signs[5])

    def test_discards_doubled_and_empty_subcarriers(self):
        """Blind-spot subcarriers (2x rate) and silent ones are unrelated to
        the breathing waveform and must not be averaged in."""
        groups = [breath(+1)] * 4 + [breath(-1)] * 4 + [doubled] * 3 + [quiet] * 3
        sel = respirfi_select(window(groups, noise=0.2), RATE, bnr_percentile=0.0, bnr_floor=0.0)
        self.assertIsNotNone(sel)
        for k in range(8, 14):
            self.assertNotIn(k, sel.selected)
        self.assertEqual(sorted(sel.selected), list(range(8)))

    def test_bnr_percentile_keeps_top_fifth(self):
        groups = [breath(+1)] * 20
        sel = respirfi_select(window(groups, noise=0.5), RATE)
        self.assertIsNotNone(sel)
        self.assertLessEqual(len(sel.candidates), 5)

    def test_noise_only_returns_none(self):
        """An empty room has nothing above the BNR floor of 2. Saying so is
        the point; selecting the least-bad noise would be worse."""
        sel = respirfi_select(window([quiet] * 20, noise=1.0, seed=3), RATE)
        self.assertIsNone(sel)

    def test_breathing_band_scores_higher_than_noise(self):
        groups = [breath(+1)] + [quiet]
        sel = respirfi_select(window(groups, noise=0.3), RATE, bnr_percentile=0.0, bnr_floor=0.0)
        self.assertGreater(sel.bnr[0], 2.0)
        self.assertLess(sel.bnr[1], 1.0)

    def test_derived_affinity_threshold_is_significance_level(self):
        sel = respirfi_select(window([breath(+1)] * 3), RATE, bnr_percentile=0.0)
        n_ind = 2 * 0.4 * 32.0
        self.assertAlmostEqual(sel.affinity_threshold, 2 / math.sqrt(n_ind), places=6)


def opposed_frames(seconds, fs=20.0, n=16, seed=4):
    """Half the subcarriers rise with each breath, half fall: plain
    averaging cancels them almost exactly."""
    rng = random.Random(seed)
    out = []
    for i in range(int(seconds * fs)):
        b = math.sin(2 * math.pi * F_B * i / fs)
        out.append([40.0 + (1.0 if k < n // 2 else -1.0) * b + rng.gauss(0, 0.05) for k in range(n)])
    return out


class TestRespirFiFusion(unittest.TestCase):
    def test_falls_back_to_fuse_before_first_pass(self):
        fu = RespirFiFusion(4, 20.0)
        self.assertAlmostEqual(fu.process([1.0, 2.0, 3.0, 4.0], [0.25] * 4), 2.5)
        self.assertIsNone(fu.weights)

    def test_recovers_breathing_that_variance_fusion_cancels(self):
        fs = 20.0
        sup = StaticSuppressor(16, fs)
        fu = RespirFiFusion(16, fs, bnr_percentile=0.0)
        legacy, signed = [], []
        for amps in opposed_frames(120, fs):
            from wifisense.types import CsiFrame

            r, w = sup.process(CsiFrame(amps, [0.0] * 16, 0, fs))
            legacy.append(fuse(r, w))
            signed.append(fu.process(r, w))
        tail = slice(-int(30 * fs), None)
        rms = lambda xs: math.sqrt(sum(x * x for x in xs) / len(xs))
        self.assertGreater(rms(signed[tail]), 10 * rms(legacy[tail]))
        ext = BreathingExtractor(fs)
        for x in signed:
            ext.push(x)
        self.assertAlmostEqual(ext.estimate().value_bpm, 15.0, delta=0.3)

    def test_orientation_is_continuous_across_passes(self):
        """Group signs are arbitrary per pass; the stream must not flip."""
        fs = 20.0
        sup = StaticSuppressor(16, fs)
        fu = RespirFiFusion(16, fs, bnr_percentile=0.0, update_s=2.0)
        from wifisense.types import CsiFrame

        history = []
        for amps in opposed_frames(120, fs, seed=5):
            r, w = sup.process(CsiFrame(amps, [0.0] * 16, 0, fs))
            fu.process(r, w)
            if fu.weights is not None and (not history or fu.weights != history[-1]):
                history.append(fu.weights)
        self.assertGreater(len(history), 5)
        for a, b in zip(history, history[1:]):
            self.assertGreater(sum(x * y for x, y in zip(a, b)), 0.0)

    def test_rejects_band_above_analysis_nyquist(self):
        with self.assertRaises(ValueError):
            RespirFiFusion(4, 20.0, f_high=3.0)


class TestPipelineFlag(unittest.TestCase):
    def test_rejects_unknown_selection(self):
        with self.assertRaises(ValueError):
            SensingPipeline(subcarrier_selection="magic")

    def test_end_to_end_with_respirfi(self):
        cfg = SyntheticConfig(breathing_bpm=15.0)
        source = SyntheticCsiSource(cfg)
        pipeline = SensingPipeline(calibration_s=20.0, subcarrier_selection="respirfi")
        cfg.occupied = False
        for frame in source.seconds(pipeline.presence.warmup_s + 30.0):
            pipeline.process(frame)
        cfg.occupied = True
        reading = None
        for frame in source.seconds(120):
            reading = pipeline.process(frame)
        self.assertTrue(reading.occupied)
        self.assertAlmostEqual(reading.respiratory_rate.value_bpm, 15.0, delta=0.5)
        self.assertEqual(reading.respiratory_rate.status, VitalStatus.VALID)
        self.assertIsNotNone(pipeline.selector.selection)


if __name__ == "__main__":
    unittest.main()
