import unittest

from wifisense.types import VitalStatus
from wifisense.vitals import (
    MIN_CYCLES,
    BreathingExtractor,
    HeartRateExtractor,
)
from wifisense.preprocess import StaticSuppressor, fuse
from wifisense.sources import SyntheticCsiSource, SyntheticConfig


def fused_stream(cfg, duration_s):
    source = SyntheticCsiSource(cfg)
    suppressor = StaticSuppressor(cfg.n_subcarriers, cfg.sample_rate_hz)
    return [fuse(*suppressor.process(f)) for f in source.seconds(duration_s)]


class TestWindowSizing(unittest.TestCase):
    def test_minimum_window_derives_from_lowest_frequency(self):
        """Resolving 0.1 Hz needs MIN_CYCLES * 10 s of history; 10 s would
        be under two cycles there."""
        ext = BreathingExtractor(20.0)
        self.assertAlmostEqual(ext.min_window_s, MIN_CYCLES / 0.1)
        self.assertGreaterEqual(ext.min_window_s, 30.0)

    def test_rejects_window_shorter_than_resolvable(self):
        with self.assertRaises(ValueError):
            BreathingExtractor(20.0, window_s=10.0)

    def test_heart_rate_default_window_exceeds_cycle_minimum(self):
        """MIN_CYCLES alone would allow ~5.6 s, far too short to separate a
        cardiac carrier from its breathing sidebands."""
        ext = HeartRateExtractor(20.0)
        self.assertGreaterEqual(ext.window_s, 30.0)

    def test_not_ready_before_window_fills(self):
        ext = BreathingExtractor(20.0)
        for _ in range(100):
            ext.push(0.0)
        self.assertFalse(ext.ready)
        self.assertEqual(ext.estimate().status, VitalStatus.UNAVAILABLE)


class TestBreathingExtractor(unittest.TestCase):
    def test_recovers_known_rate(self):
        cfg = SyntheticConfig(breathing_bpm=15.0)
        ext = BreathingExtractor(cfg.sample_rate_hz)
        for sample in fused_stream(cfg, 120):
            ext.push(sample)
        est = ext.estimate()
        self.assertAlmostEqual(est.value_bpm, 15.0, delta=0.5)
        self.assertEqual(est.status, VitalStatus.VALID)

    def test_tracks_across_the_band(self):
        for bpm in (8.0, 12.0, 20.0, 24.0):
            cfg = SyntheticConfig(breathing_bpm=bpm)
            ext = BreathingExtractor(cfg.sample_rate_hz)
            for sample in fused_stream(cfg, 120):
                ext.push(sample)
            self.assertAlmostEqual(ext.estimate().value_bpm, bpm, delta=0.5)

    def test_empty_room_yields_low_confidence(self):
        cfg = SyntheticConfig(occupied=False)
        ext = BreathingExtractor(cfg.sample_rate_hz)
        for sample in fused_stream(cfg, 120):
            ext.push(sample)
        est = ext.estimate()
        self.assertLess(est.confidence, 0.4)
        self.assertEqual(est.status, VitalStatus.UNRELIABLE)

    def test_reset_clears_history(self):
        cfg = SyntheticConfig()
        ext = BreathingExtractor(cfg.sample_rate_hz)
        for sample in fused_stream(cfg, 120):
            ext.push(sample)
        self.assertTrue(ext.ready)
        ext.reset()
        self.assertFalse(ext.ready)


class TestHeartRateExtractor(unittest.TestCase):
    def test_sideband_correction_beats_naive_peak(self):
        """With the breathing frequency supplied, the estimator must not
        return the f_h + f_b sideband that plain peak picking locks onto."""
        cfg = SyntheticConfig(breathing_bpm=15.0, heart_bpm=68.0)
        samples = fused_stream(cfg, 150)

        ext = HeartRateExtractor(cfg.sample_rate_hz, window_s=120.0)
        for s in samples:
            ext.push(s)

        corrected = ext.estimate(breathing_hz=15.0 / 60.0)
        self.assertAlmostEqual(corrected.value_bpm, 68.0, delta=3.0)

        naive = ext.estimate(breathing_hz=None)
        self.assertGreater(abs(naive.value_bpm - 68.0), 3.0)

    def test_low_confidence_is_labelled_unreliable(self):
        """The gate that makes this usable: wrong answers must not be
        reported as valid. See evaluate.py for the measured relationship."""
        cfg = SyntheticConfig(occupied=False)
        ext = HeartRateExtractor(cfg.sample_rate_hz, window_s=120.0)
        for s in fused_stream(cfg, 150):
            ext.push(s)
        est = ext.estimate(breathing_hz=0.25)
        self.assertIn(
            est.status, (VitalStatus.UNRELIABLE, VitalStatus.UNAVAILABLE)
        )


if __name__ == "__main__":
    unittest.main()
