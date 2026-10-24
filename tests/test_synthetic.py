import math
import os
import struct
import unittest

from wifisense.dsp import band_peak
from wifisense.preprocess import StaticSuppressor, fuse
from wifisense.sources import SyntheticCsiSource, SyntheticConfig
from wifisense.sources.synthetic import SPEED_OF_LIGHT


def geo(**kw):
    return SyntheticConfig(scene="geometric", **kw)


REFERENCE = os.path.join(os.path.dirname(__file__), "data", "random_scene_200.f64")

# Two orders tighter than the 1e-9 the Rust port is held to (ADR-002), and
# the same bound spec/golden/check_drift.py uses, for the same reason.
TOL = 1e-11


class TestRandomSceneUnchanged(unittest.TestCase):
    def test_default_output_matches_the_original(self):
        """The original evaluation numbers and every existing test depend on
        this sequence: the default scene must not change.

        This used to be a SHA-256 of `repr()` of the output, taken from the
        pre-P0a source, and it held until CI moved to Ubuntu 26.04. A digest
        of floats is a statement about one libm: math.sin and math.cos are
        not required to be correctly rounded, two platforms may disagree in
        the last place, and one ULP anywhere in 22,400 values changes every
        bit of the hash. The reference file is that same output -- written
        from a run whose digest still matched the pin -- compared
        numerically: amplitudes against their own scale, phases as wrapped
        angles, both at 1e-11. Any change to the generator moves values by
        far more than that; a different libm moves them by about 1e-15.
        """
        frames = SyntheticCsiSource(SyntheticConfig()).frames(200)
        amps = [a for f in frames for a in f.amplitudes]
        phases = [p for f in frames for p in f.phases]
        with open(REFERENCE, "rb") as fh:
            raw = fh.read()
        ref = struct.unpack(f"<{len(raw) // 8}d", raw)
        self.assertEqual(len(ref), len(amps) + len(phases), "frame count or subcarrier count changed")
        ref_amps, ref_phases = ref[: len(amps)], ref[len(amps):]

        scale = max(abs(a) for a in ref_amps)
        worst_amp = max(abs(a - b) for a, b in zip(amps, ref_amps)) / scale
        # A phase one ULP either side of +/-pi wraps to the other end; compare
        # the angle between them, not the difference of the numbers.
        worst_phase = max(
            abs((a - b + math.pi) % (2.0 * math.pi) - math.pi) for a, b in zip(phases, ref_phases)
        ) / math.pi
        self.assertLess(worst_amp, TOL, f"amplitudes moved by {worst_amp:.3e} of their scale")
        self.assertLess(worst_phase, TOL, f"phases moved by {worst_phase:.3e} of pi")

    def test_rejects_unknown_scene(self):
        with self.assertRaises(ValueError):
            SyntheticCsiSource(SyntheticConfig(scene="moon"))


class TestGeometricScene(unittest.TestCase):
    def test_deterministic_under_seed(self):
        a = SyntheticCsiSource(geo(n_scatterers=5, seed=9)).frames(20)
        b = SyntheticCsiSource(geo(n_scatterers=5, seed=9)).frames(20)
        self.assertEqual([f.amplitudes for f in a], [f.amplitudes for f in b])

    def test_positions_must_be_inside_room(self):
        with self.assertRaises(ValueError):
            SyntheticCsiSource(geo(body_xy_m=(7.0, 1.0)))

    def test_excess_path(self):
        src = SyntheticCsiSource(geo(tx_xy_m=(1.0, 2.0), rx_xy_m=(5.0, 2.0), body_xy_m=(3.0, 3.5)))
        self.assertAlmostEqual(src.excess_path_m, 2 * math.hypot(2.0, 1.5) - 4.0, places=9)

    def test_los_magnitude_scaled(self):
        """With no reflections, scatterers or body, |h| is exactly the LoS
        magnitude on every subcarrier."""
        cfg = geo(wall_reflection=0.0, noise_std=0.0, occupied=False)
        frame = SyntheticCsiSource(cfg).next_frame()
        for a in frame.amplitudes:
            self.assertAlmostEqual(a, cfg.static_magnitude, places=9)

    def test_los_attenuation(self):
        cfg = geo(wall_reflection=0.0, noise_std=0.0, occupied=False, los_attenuation_db=20.0)
        frame = SyntheticCsiSource(cfg).next_frame()
        self.assertAlmostEqual(frame.amplitudes[0], cfg.static_magnitude / 10.0, places=9)

    def test_second_order_reflections_change_static_channel(self):
        a = SyntheticCsiSource(geo(noise_std=0.0, occupied=False)).next_frame()
        b = SyntheticCsiSource(geo(noise_std=0.0, occupied=False, reflection_order=2)).next_frame()
        self.assertNotAlmostEqual(a.amplitudes[10], b.amplitudes[10], places=6)

    def test_static_phase_rotates_with_frequency_coherently(self):
        """Pure LoS: phase advances by exactly -2*pi*spacing*d/c per subcarrier."""
        cfg = geo(wall_reflection=0.0, noise_std=0.0, occupied=False)
        frame = SyntheticCsiSource(cfg).next_frame()
        d = math.dist(cfg.tx_xy_m, cfg.rx_xy_m)
        step = -2 * math.pi * cfg.subcarrier_spacing_hz * d / SPEED_OF_LIGHT
        diff = math.remainder(frame.phases[1] - frame.phases[0] - step, 2 * math.pi)
        self.assertAlmostEqual(diff, 0.0, places=9)

    def test_sensitivity_bounded(self):
        for y in (3.0, 3.5, 4.2):
            for kw in ({}, dict(body_points=5, body_width_m=0.3)):
                s = SyntheticCsiSource(geo(body_xy_m=(3.0, y), n_scatterers=4, seed=3, **kw)).sensitivity()
                self.assertTrue(all(0.0 <= v <= 1.0 + 1e-12 for v in s))

    def test_blind_spot_doubles_the_apparent_rate(self):
        """The physics the whole null suite rests on. On a Fresnel-zone
        boundary the first-order response vanishes and the amplitude
        oscillates at twice the breathing rate; one zone over, it doesn't."""
        f_b = 0.2
        found = {}
        for i in range(400):
            y = 2.9 + i * 0.0025
            src = SyntheticCsiSource(geo(body_xy_m=(3.0, y), noise_std=0.0, breathing_bpm=60 * f_b, heart_amplitude_m=0.0))
            peak = max(src.sensitivity())
            if peak < 0.05 and "null" not in found:
                found["null"] = src
            if peak > 0.9 and "clear" not in found:
                found["clear"] = src
            if len(found) == 2:
                break
        self.assertEqual(set(found), {"null", "clear"})

        def dominant(src):
            sup = StaticSuppressor(src.config.n_subcarriers, src.config.sample_rate_hz)
            x = [fuse(*sup.process(f)) for f in src.seconds(90)][-60 * 20 :]
            return band_peak(x, 20.0, 0.1, 0.6).freq_hz

        self.assertAlmostEqual(dominant(found["clear"]), f_b, delta=0.01)
        self.assertAlmostEqual(dominant(found["null"]), 2 * f_b, delta=0.01)

    def test_extended_body_keeps_the_null(self):
        """A torso facing the link spreads along the Fresnel ellipse, where
        excess path changes only to second order, and moves as one block:
        extent shifts the null, it does not remove it."""
        lowest = min(
            max(SyntheticCsiSource(geo(body_xy_m=(3.0, 2.9 + i * 0.0025), body_points=5, body_width_m=0.3)).sensitivity())
            for i in range(120)
        )
        self.assertLess(lowest, 0.1)


if __name__ == "__main__":
    unittest.main()
