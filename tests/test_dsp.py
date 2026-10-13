import math
import random
import unittest

from wifisense.dsp import (
    BiquadBandpass,
    band_peak,
    goertzel_power,
    hann_window,
    sideband_peak,
)


def sine(freq_hz, fs, duration_s, amplitude=1.0, phase=0.0):
    n = int(fs * duration_s)
    return [amplitude * math.sin(2 * math.pi * freq_hz * i / fs + phase) for i in range(n)]


class TestBiquadBandpass(unittest.TestCase):
    def test_rejects_inverted_band(self):
        with self.assertRaises(ValueError):
            BiquadBandpass(0.5, 0.1, 20.0)

    def test_rejects_band_above_nyquist(self):
        with self.assertRaises(ValueError):
            BiquadBandpass(0.1, 15.0, 20.0)

    def test_center_is_geometric_mean(self):
        f = BiquadBandpass(0.1, 0.4, 20.0)
        self.assertAlmostEqual(f.center_hz, math.sqrt(0.04), places=9)

    @staticmethod
    def _gain(f_low, f_high, fs, freq, duration_s=400.0):
        """Measured |H(freq)| for a unit sine, ignoring the settling transient."""
        filt = BiquadBandpass(f_low, f_high, fs)
        out = filt.process_all(sine(freq, fs, duration_s))
        tail = out[len(out) // 2 :]
        return math.sqrt(sum(x * x for x in tail) / len(tail)) * math.sqrt(2)

    def test_unity_gain_at_center(self):
        """Constant-peak-gain topology: |H| = 1 at the geometric centre."""
        self.assertAlmostEqual(self._gain(0.1, 0.5, 20.0, math.sqrt(0.05)), 1.0, places=2)

    def test_minus_3db_at_band_edges(self):
        """The real specification. Q is derived so the requested edges land
        on the half-power points; if the centre or Q were wrong these would
        drift immediately."""
        for edge in (0.1, 0.5):
            self.assertAlmostEqual(
                self._gain(0.1, 0.5, 20.0, edge), 1 / math.sqrt(2), places=2
            )

    def test_rolloff_is_monotonic_out_of_band(self):
        """A single biquad gives 6 dB/octave, so rejection is gradual --
        about 8x at 3 Hz, not the 100x a steeper filter would give. Cascade
        sections if an application needs more."""
        gains = [self._gain(0.1, 0.5, 20.0, f) for f in (1.0, 2.0, 3.0, 5.0, 8.0)]
        self.assertEqual(gains, sorted(gains, reverse=True))
        self.assertLess(gains[2], 0.15)  # 3 Hz
        self.assertLess(gains[-1], 0.05)  # 8 Hz

    def test_remains_stable_over_long_run(self):
        """An ad-hoc resonator diverges without a pole clamp; a biquad
        with Q > 0 is unconditionally stable, so assert that directly."""
        f = BiquadBandpass(0.1, 0.5, 20.0)
        out = 0.0
        for _ in range(200_000):
            out = f.process(1.0)
        self.assertTrue(math.isfinite(out))
        self.assertLess(abs(out), 1e6)

    def test_recovers_from_nan_input(self):
        f = BiquadBandpass(0.1, 0.5, 20.0)
        for v in sine(0.25, 20.0, 10):
            f.process(v)
        f.process(float("nan"))
        after = [f.process(v) for v in sine(0.25, 20.0, 30)]
        self.assertTrue(all(math.isfinite(v) for v in after))
        self.assertGreater(max(abs(v) for v in after), 0.0)


class TestGoertzel(unittest.TestCase):
    def test_peaks_at_signal_frequency(self):
        fs = 20.0
        sig = sine(0.25, fs, 60)
        at_signal = goertzel_power(sig, 0.25, fs)
        off_signal = goertzel_power(sig, 0.40, fs)
        self.assertGreater(at_signal, 100 * off_signal)

    def test_zero_for_empty_input(self):
        self.assertEqual(goertzel_power([], 1.0, 20.0), 0.0)

    def test_matches_naive_dft(self):
        """Generalised Goertzel must equal a direct DFT evaluation at an
        arbitrary, non-bin-aligned frequency."""
        fs, f = 20.0, 0.2345
        sig = sine(0.25, fs, 20)
        n = len(sig)
        k = f * n / fs
        real = sum(x * math.cos(2 * math.pi * k * i / n) for i, x in enumerate(sig))
        imag = -sum(x * math.sin(2 * math.pi * k * i / n) for i, x in enumerate(sig))
        self.assertAlmostEqual(
            goertzel_power(sig, f, fs) / (real * real + imag * imag), 1.0, places=6
        )


class TestHannWindow(unittest.TestCase):
    def test_endpoints_and_length(self):
        w = hann_window(64)
        self.assertEqual(len(w), 64)
        self.assertAlmostEqual(w[0], 0.0, places=12)
        self.assertAlmostEqual(max(w), 1.0, places=6)

    def test_degenerate_lengths(self):
        self.assertEqual(hann_window(0), [])
        self.assertEqual(hann_window(1), [1.0])


class TestBandPeak(unittest.TestCase):
    def test_finds_known_frequency(self):
        peak = band_peak(sine(0.25, 20.0, 60), 20.0, 0.1, 0.5)
        self.assertIsNotNone(peak)
        self.assertAlmostEqual(peak.freq_hz, 0.25, places=2)

    def test_subresolution_accuracy(self):
        """Parabolic interpolation should beat the scan grid spacing."""
        peak = band_peak(sine(0.2637, 20.0, 120), 20.0, 0.1, 0.5, resolution_hz=0.01)
        self.assertLess(abs(peak.freq_hz - 0.2637), 0.01)

    def test_confidence_high_for_clean_tone(self):
        peak = band_peak(sine(0.25, 20.0, 60), 20.0, 0.1, 0.5)
        self.assertGreater(peak.confidence, 0.9)

    def test_confidence_low_for_noise(self):
        """The naive '1 - mean/peak' scores band-limited noise around 0.75.
        Noise-calibrated confidence must put it near zero instead."""
        rng = random.Random(7)
        noise = [rng.gauss(0, 1) for _ in range(1200)]
        peak = band_peak(noise, 20.0, 0.1, 0.5)
        self.assertLess(peak.confidence, 0.25)

    def test_confidence_decreases_with_noise(self):
        rng = random.Random(11)
        clean = sine(0.25, 20.0, 60)
        noisy = [v + rng.gauss(0, 3.0) for v in clean]
        self.assertGreater(
            band_peak(clean, 20.0, 0.1, 0.5).confidence,
            band_peak(noisy, 20.0, 0.1, 0.5).confidence,
        )

    def test_notch_excludes_frequency(self):
        peak = band_peak(
            sine(0.25, 20.0, 60), 20.0, 0.1, 0.5, notch_hz=[0.25], notch_width_hz=0.05
        )
        self.assertGreater(abs(peak.freq_hz - 0.25), 0.04)

    def test_returns_none_on_empty(self):
        self.assertIsNone(band_peak([], 20.0, 0.1, 0.5))


class TestSidebandPeak(unittest.TestCase):
    def test_recovers_carrier_from_stronger_sideband(self):
        """The regression that motivates this function: when the
        f_h + f_b intermodulation sideband is stronger than the carrier,
        plain peak picking returns the sideband. A symmetric-triplet
        search must still return the carrier."""
        fs, f_b, f_h = 20.0, 0.25, 1.1333
        n = int(fs * 120)
        sig = []
        for i in range(n):
            t = i / fs
            sig.append(
                0.4 * math.sin(2 * math.pi * f_h * t)
                + 0.9 * math.sin(2 * math.pi * (f_h + f_b) * t)  # stronger sideband
                + 0.4 * math.sin(2 * math.pi * (f_h - f_b) * t)
            )
        naive = band_peak(sig, fs, 0.8, 2.0)
        self.assertAlmostEqual(naive.freq_hz, f_h + f_b, places=2)  # naive is fooled

        corrected = sideband_peak(sig, fs, 0.8, 2.0, modulator_hz=f_b)
        self.assertAlmostEqual(corrected.freq_hz, f_h, places=2)

    def test_rejects_invalid_modulator(self):
        self.assertIsNone(sideband_peak(sine(1.1, 20.0, 60), 20.0, 0.8, 2.0, 0.0))

    def test_returns_none_on_empty(self):
        self.assertIsNone(sideband_peak([], 20.0, 0.8, 2.0, 0.25))


if __name__ == "__main__":
    unittest.main()
