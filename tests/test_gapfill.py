import random
import unittest

from wifisense.preprocess import StaticSuppressor, fuse
from wifisense.sources import SyntheticCsiSource, SyntheticConfig
from wifisense.sources.esp32_udp import SequenceGapFiller
from wifisense.types import CsiFrame
from wifisense.vitals import BreathingExtractor


def f(amp, seq=0):
    return CsiFrame([amp, 2 * amp], [0.0, 0.0], seq, 20.0)


class TestSequenceGapFiller(unittest.TestCase):
    def test_contiguous_frames_pass_through(self):
        g = SequenceGapFiller()
        for seq in range(5):
            self.assertEqual(len(g.push(seq, f(1.0, seq))), 1)
        self.assertEqual(g.filled, 0)

    def test_short_gap_is_linearly_interpolated(self):
        g = SequenceGapFiller(max_gap=5)
        g.push(10, f(1.0))
        out = g.push(13, f(4.0))
        self.assertEqual([x.amplitudes[0] for x in out], [2.0, 3.0, 4.0])
        self.assertEqual([x.sample_index for x in out[:2]], [11, 12])
        self.assertEqual(g.filled, 2)

    def test_long_gap_is_a_discontinuity_not_invented_data(self):
        g = SequenceGapFiller(max_gap=5)
        g.push(0, f(1.0))
        out = g.push(100, f(2.0))
        self.assertEqual(len(out), 1)
        self.assertEqual(g.discontinuities, 1)

    def test_duplicates_dropped_and_reboots_pass_through(self):
        g = SequenceGapFiller()
        g.push(50, f(1.0))
        self.assertEqual(g.push(50, f(1.0)), [])
        self.assertEqual(g.duplicates, 1)
        self.assertEqual(len(g.push(0, f(1.0))), 1)  # node rebooted
        self.assertEqual(g.discontinuities, 1)

    def test_u32_wrap_is_contiguous(self):
        g = SequenceGapFiller()
        g.push(0xFFFFFFFF, f(1.0))
        out = g.push(1, f(3.0))
        self.assertEqual([x.amplitudes[0] for x in out], [2.0, 3.0])
        self.assertEqual(g.discontinuities, 0)

    def test_packet_loss_no_longer_biases_breathing_rate(self):
        """Regression: feeding survivors of 15% packet loss straight in
        compresses time, and 15 BPM read as 17.8 BPM *at confidence 0.98*.
        With gap filling it reads 15.0."""
        rng = random.Random(1)
        frames = [x for x in SyntheticCsiSource(SyntheticConfig(breathing_bpm=15)).seconds(120) if rng.random() >= 0.15]

        def estimate(fill):
            sup, ex, g = StaticSuppressor(56, 20.0), BreathingExtractor(20.0), SequenceGapFiller(20)
            for x in frames:
                for y in g.push(x.sample_index, x) if fill else [x]:
                    ex.push(fuse(*sup.process(y)))
            return ex.estimate().value_bpm

        self.assertGreater(estimate(fill=False), 16.5)
        self.assertAlmostEqual(estimate(fill=True), 15.0, delta=0.2)


if __name__ == "__main__":
    unittest.main()
