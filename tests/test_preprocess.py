import math
import unittest

from wifisense.preprocess import (
    StaticSuppressor,
    alpha_to_corner,
    corner_to_alpha,
    fuse,
)
from wifisense.types import CsiFrame


def frame(amps, fs=20.0, idx=0):
    return CsiFrame(list(amps), [0.0] * len(amps), idx, fs)


class TestCornerAlphaMapping(unittest.TestCase):
    def test_roundtrip(self):
        for fs in (10.0, 20.0, 100.0):
            for corner in (0.01, 0.03, 0.1):
                self.assertAlmostEqual(
                    alpha_to_corner(corner_to_alpha(corner, fs), fs), corner, places=9
                )

    def test_alpha_scales_with_sample_rate(self):
        """Same corner at double the rate needs half the alpha."""
        self.assertAlmostEqual(
            corner_to_alpha(0.03, 40.0), corner_to_alpha(0.03, 20.0) / 2, places=9
        )

    def test_fixed_alpha_puts_corner_above_breathing_band(self):
        """Why alpha must track the sample rate.

        A fixed alpha = 0.05 at 100 Hz puts the highpass corner near
        0.8 Hz -- above the whole 0.1-0.5 Hz breathing band it is meant to
        preserve. This test pins the arithmetic so the reasoning survives.
        """
        self.assertGreater(alpha_to_corner(0.05, 100.0), 0.5)
        # The sample-rate-aware default stays well below the band.
        self.assertLess(alpha_to_corner(corner_to_alpha(0.03, 100.0), 100.0), 0.1)

    def test_rejects_bad_sample_rate(self):
        with self.assertRaises(ValueError):
            corner_to_alpha(0.03, 0.0)


class TestStaticSuppressor(unittest.TestCase):
    def test_rejects_zero_subcarriers(self):
        with self.assertRaises(ValueError):
            StaticSuppressor(0, 20.0)

    def test_first_frame_residual_is_zero(self):
        s = StaticSuppressor(3, 20.0)
        residuals, _ = s.process(frame([1.0, 2.0, 3.0]))
        self.assertEqual(residuals, [0.0, 0.0, 0.0])

    def test_static_signal_converges_to_zero(self):
        s = StaticSuppressor(2, 20.0, corner_hz=0.5)
        f = frame([5.0, 10.0])
        for _ in range(2000):
            residuals, _ = s.process(f)
        for r in residuals:
            self.assertLess(abs(r), 0.01)

    def test_preserves_in_band_oscillation(self):
        """The point of the whole stage: kill the static component, keep
        the breathing modulation."""
        fs, f_b = 20.0, 0.25
        s = StaticSuppressor(1, fs, corner_hz=0.03)
        residuals = []
        for i in range(int(fs * 120)):
            amp = 100.0 + 1.0 * math.sin(2 * math.pi * f_b * i / fs)
            r, _ = s.process(frame([amp], fs, i))
            residuals.append(r[0])
        tail = residuals[len(residuals) // 2 :]
        peak_to_peak = max(tail) - min(tail)
        self.assertGreater(peak_to_peak, 1.0)   # 1.0 amplitude -> ~2.0 p2p
        self.assertLess(abs(sum(tail) / len(tail)), 0.05)  # DC removed

    def test_handles_non_finite_amplitude(self):
        s = StaticSuppressor(2, 20.0)
        s.process(frame([1.0, 2.0]))
        residuals, _ = s.process(frame([float("nan"), 2.0]))
        self.assertTrue(all(math.isfinite(r) for r in residuals))

    def test_weights_favor_moving_subcarriers(self):
        s = StaticSuppressor(2, 20.0, corner_hz=0.03)
        for i in range(400):
            moving = 10.0 + 5.0 * math.sin(2 * math.pi * 0.25 * i / 20.0)
            _, weights = s.process(frame([moving, 10.0], 20.0, i))
        self.assertGreater(weights[0], weights[1])

    def test_reset_clears_state(self):
        s = StaticSuppressor(2, 20.0)
        s.process(frame([1.0, 2.0]))
        s.reset()
        residuals, _ = s.process(frame([1.0, 2.0]))
        self.assertEqual(residuals, [0.0, 0.0])


class TestFuse(unittest.TestCase):
    def test_weighted_average(self):
        self.assertAlmostEqual(fuse([1.0, 3.0], [0.5, 0.5]), 2.0)

    def test_renormalizes_partial_weights(self):
        """Weights summing to less than 1 must not scale the result down."""
        self.assertAlmostEqual(fuse([2.0, 2.0], [0.1, 0.1]), 2.0)

    def test_empty_inputs(self):
        self.assertEqual(fuse([], []), 0.0)

    def test_zero_weights_fall_back_to_mean(self):
        self.assertAlmostEqual(fuse([1.0, 3.0], [0.0, 0.0]), 2.0)


if __name__ == "__main__":
    unittest.main()
