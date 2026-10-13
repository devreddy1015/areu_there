import struct
import unittest

from wifisense.features import (
    FEATURE_FRAME_SIZE,
    FLAG_BREATHING_VALID,
    FLAG_CALIBRATED,
    FLAG_OCCUPIED,
    FLAG_VITALS_ENABLED,
    NOT_CALIBRATED,
    FeatureExtractor,
    FeatureFrame,
    decode_feature_frame,
)
from wifisense.sources import SyntheticCsiSource, SyntheticConfig


def run(vitals: bool, heart: bool = False, seconds_occupied: float = 80.0):
    cfg = SyntheticConfig(n_subcarriers=16, breathing_bpm=15.0)
    src = SyntheticCsiSource(cfg)
    fx = FeatureExtractor(n_subcarriers=16, vitals_enabled=vitals, heart_enabled=heart, calibration_s=10.0)
    out = []
    cfg.occupied = False
    i = 0
    for f in src.seconds(65):
        out += fx.push(f, i, i / 20.0)
        i += 1
    cfg.occupied = True
    for f in src.seconds(seconds_occupied):
        out += fx.push(f, i, i / 20.0)
        i += 1
    return out


class TestCodec(unittest.TestCase):
    def test_size_magic_and_roundtrip(self):
        f = FeatureFrame(node_id=4, tx_id=9, flags=0b10111, seq=77, breathing_bpm=14.5, calib_age_s=12)
        b = f.encode()
        self.assertEqual(len(b), FEATURE_FRAME_SIZE)
        self.assertEqual(b[:4], b"SANS")
        d = decode_feature_frame(b)
        self.assertEqual((d.node_id, d.tx_id, d.flags, d.seq, d.calib_age_s), (4, 9, 0b10111, 77, 12))
        self.assertEqual(d.breathing_bpm, 14.5)

    def test_crc_catches_corruption(self):
        b = bytearray(FeatureFrame().encode())
        b[30] ^= 0x01
        with self.assertRaises(ValueError):
            decode_feature_frame(bytes(b))

    def test_rejects_wrong_version_and_size(self):
        with self.assertRaises(ValueError):
            decode_feature_frame(FeatureFrame().encode()[:-1])
        f = FeatureFrame(version=2).encode()
        with self.assertRaises(ValueError):
            decode_feature_frame(f)

    def test_floats_are_float32_on_the_wire(self):
        d = decode_feature_frame(FeatureFrame(motion_energy=0.1).encode())
        self.assertEqual(d.motion_energy, struct.unpack("<f", struct.pack("<f", 0.1))[0])


class TestExtractor(unittest.TestCase):
    def test_two_frames_per_second_after_calibration(self):
        frames = run(vitals=True)
        self.assertEqual(len(frames), int((65 + 80) * 2))
        self.assertEqual(frames[0].calib_age_s, NOT_CALIBRATED)
        last = frames[-1]
        self.assertTrue(last.flags & FLAG_CALIBRATED)
        self.assertTrue(last.flags & FLAG_OCCUPIED)
        self.assertTrue(last.flags & FLAG_BREATHING_VALID)
        self.assertAlmostEqual(last.breathing_bpm, 15.0, delta=0.5)
        self.assertAlmostEqual(last.csi_rate_chz / 100.0, 20.0, delta=0.1)

    def test_consent_gate_zeroes_vitals_on_the_node(self):
        """ADR-001/005: without consent, nothing derived from the
        breathing or heart band may leave the extractor."""
        for f in run(vitals=False):
            self.assertFalse(f.flags & FLAG_VITALS_ENABLED)
            self.assertFalse(f.flags & FLAG_BREATHING_VALID)
            self.assertEqual((f.breath_rms, f.breathing_bpm, f.breathing_conf, f.heart_bpm, f.heart_conf), (0.0,) * 5)

    def test_heart_needs_its_own_consent(self):
        for f in run(vitals=True, heart=False):
            self.assertEqual((f.heart_bpm, f.heart_conf), (0.0, 0.0))

    def test_presence_still_works_without_vitals(self):
        self.assertTrue(run(vitals=False)[-1].flags & FLAG_OCCUPIED)

    def test_deterministic(self):
        a = [f.encode() for f in run(vitals=True, seconds_occupied=10)]
        b = [f.encode() for f in run(vitals=True, seconds_occupied=10)]
        self.assertEqual(a, b)

    def test_rejects_rates_too_slow_for_the_spec(self):
        with self.assertRaises(ValueError):
            FeatureExtractor(sample_rate_hz=8.0)


if __name__ == "__main__":
    unittest.main()
