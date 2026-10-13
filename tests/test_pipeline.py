import math
import unittest

from wifisense import SensingPipeline
from wifisense.presence import PresenceDetector
from wifisense.sources import SyntheticCsiSource, SyntheticConfig
from wifisense.types import VitalStatus

FS = 20.0
MOTION_HZ = 0.25  # inside the 0.1-2.0 Hz presence band


def motion(n, amplitude=1.0, fs=FS, freq=MOTION_HZ):
    """In-band stimulus. Note this must be *in band*: the detector
    bandpasses before measuring energy, so an alternating +/-1 sequence
    (which is fs/2) is rejected entirely and reads as an empty room."""
    return [amplitude * math.sin(2 * math.pi * freq * i / fs) for i in range(n)]


def quiet(n, level=1e-4):
    return [level] * n


class TestPresenceDetector(unittest.TestCase):
    def _calibrated(self, **kw):
        """A detector past warmup and calibration on a quiet room.

        `warmup_s=0` isolates the threshold logic from filter settling;
        `test_warmup_is_required_before_calibration` covers the real path.
        """
        params = dict(calibration_s=5.0, warmup_s=0.0, hold_s=10.0)
        params.update(kw)
        d = PresenceDetector(FS, **params)
        for v in quiet(int(FS * 6)):
            d.process(v)
        return d

    def test_requires_calibration_before_reporting(self):
        d = PresenceDetector(FS, calibration_s=5.0, warmup_s=0.0)
        state = d.process(0.1)
        self.assertFalse(state.calibrated)
        self.assertFalse(state.occupied)

    def test_warmup_is_required_before_calibration(self):
        """Calibrating through the bandpass transient measures the filter
        waking up, not the room, and yields a threshold an occupied room
        never crosses."""
        d = PresenceDetector(FS, calibration_s=5.0, warmup_s=30.0)
        for v in quiet(int(FS * 20)):
            d.process(v)
        self.assertFalse(d.calibrated)  # still inside warmup
        for v in quiet(int(FS * 20)):
            d.process(v)
        self.assertTrue(d.calibrated)

    def test_default_warmup_covers_several_low_frequency_periods(self):
        d = PresenceDetector(FS, f_low=0.1)
        self.assertGreaterEqual(d.warmup_s, 30.0)

    def test_calibrates_on_quiet_input(self):
        self.assertTrue(self._calibrated().calibrated)

    def test_detects_in_band_energy(self):
        d = self._calibrated()
        state = None
        for v in motion(int(FS * 30)):
            state = d.process(v)
        self.assertTrue(state.occupied)
        self.assertGreater(state.energy, state.threshold)

    def test_ignores_out_of_band_energy(self):
        """A 10 Hz jitter is not a person. Broadband RMS would trigger on
        it; band-limited energy must not."""
        d = self._calibrated()
        state = None
        for i in range(int(FS * 30)):
            state = d.process(1.0 if i % 2 else -1.0)
        self.assertFalse(state.occupied)

    def test_hysteresis_holds_through_a_quiet_gap(self):
        """A motionless occupant must not flicker the output off between
        breaths -- the usual complaint about naive detectors."""
        d = self._calibrated(hold_s=20.0)
        for v in motion(int(FS * 30)):
            d.process(v)
        state = None
        for v in quiet(int(FS * 5)):  # 5 s quiet, inside the 20 s hold
            state = d.process(v)
        self.assertTrue(state.occupied)

    def test_releases_after_hold_expires(self):
        d = self._calibrated(hold_s=2.0)
        for v in motion(int(FS * 30)):
            d.process(v)
        state = None
        for v in quiet(int(FS * 30)):
            state = d.process(v)
        self.assertFalse(state.occupied)

    def test_recalibrate_resets(self):
        d = self._calibrated()
        self.assertTrue(d.calibrated)
        d.recalibrate()
        self.assertFalse(d.calibrated)


class TestSensingPipeline(unittest.TestCase):
    def _run(self, cfg, calib_s=20.0, occupied_s=150.0):
        source = SyntheticCsiSource(cfg)
        pipeline = SensingPipeline(
            n_subcarriers=cfg.n_subcarriers,
            sample_rate_hz=cfg.sample_rate_hz,
            calibration_s=calib_s,
        )
        # The empty phase must cover filter warmup *and* calibration.
        empty_s = pipeline.presence.warmup_s + calib_s + 10.0
        cfg.occupied = False
        for frame in source.seconds(empty_s):
            pipeline.process(frame)
        cfg.occupied = True
        reading = None
        for frame in source.seconds(occupied_s):
            reading = pipeline.process(frame)
        return pipeline, source, reading

    def test_end_to_end_breathing(self):
        cfg = SyntheticConfig(breathing_bpm=15.0, heart_bpm=68.0)
        pipeline, _, reading = self._run(cfg)
        self.assertTrue(pipeline.calibrated)
        self.assertTrue(reading.occupied)
        self.assertAlmostEqual(reading.respiratory_rate.value_bpm, 15.0, delta=0.5)
        self.assertEqual(reading.respiratory_rate.status, VitalStatus.VALID)

    def test_presence_is_stable_while_occupied(self):
        """Occupancy must not chatter while someone is simply sitting there."""
        cfg = SyntheticConfig()
        source = SyntheticCsiSource(cfg)
        pipeline = SensingPipeline(
            n_subcarriers=cfg.n_subcarriers,
            sample_rate_hz=cfg.sample_rate_hz,
            calibration_s=20.0,
        )
        cfg.occupied = False
        for frame in source.seconds(pipeline.presence.warmup_s + 30.0):
            pipeline.process(frame)
        cfg.occupied = True
        for frame in source.seconds(30):  # let energy rise
            pipeline.process(frame)
        states = [pipeline.process(f).occupied for f in source.seconds(90)]
        self.assertTrue(all(states), f"occupancy flickered: {states.count(False)} drops")

    def test_vitals_suppressed_when_room_empties(self):
        """Reporting a breathing rate for an empty room is how these systems
        manufacture impressive numbers out of noise."""
        cfg = SyntheticConfig()
        pipeline, source, _ = self._run(cfg)
        cfg.occupied = False
        reading = None
        for frame in source.seconds(90):
            reading = pipeline.process(frame)
        self.assertFalse(reading.occupied)
        self.assertEqual(reading.respiratory_rate.status, VitalStatus.UNAVAILABLE)
        self.assertEqual(reading.heart_rate.status, VitalStatus.UNAVAILABLE)

    def test_reading_carries_metadata(self):
        cfg = SyntheticConfig()
        _, _, reading = self._run(cfg)
        self.assertEqual(reading.subcarrier_count, cfg.n_subcarriers)
        self.assertGreater(reading.timestamp_s, 0.0)
        self.assertGreaterEqual(reading.signal_quality, 0.0)
        self.assertLessEqual(reading.signal_quality, 1.0)

    def test_reset_returns_to_uncalibrated(self):
        cfg = SyntheticConfig()
        pipeline, _, _ = self._run(cfg)
        pipeline.reset()
        self.assertFalse(pipeline.calibrated)


if __name__ == "__main__":
    unittest.main()
