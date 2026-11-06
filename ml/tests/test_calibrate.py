"""ml/calibrate (spec/calibration.md): parameter maps, the config overrides,
quantiles and the Latin hypercube, the emulator and the fit on a toy
problem with a known answer, and one real Room Sim run when the release
CLI and the dataset exist."""

import json
import os
import unittest

import numpy as np
import torch

from ml.calibrate import calibrate as cc
from ml.calibrate import features as cf
from ml.calibrate import simulate as cs


class TestParams(unittest.TestCase):
    def test_unit_round_trip_and_defaults_inside(self):
        u = cs.to_unit(cs.DEFAULT_THETA)
        self.assertTrue(((u >= 0) & (u <= 1)).all())
        back = cs.to_theta(u)
        for name in cs.NAMES:
            self.assertAlmostEqual(back[name], cs.DEFAULT_THETA[name], places=9)
        lo = cs.to_theta(np.zeros(len(cs.PARAMS)))
        hi = cs.to_theta(np.ones(len(cs.PARAMS)))
        for name, _d, a, b, _s in cs.PARAMS:
            self.assertAlmostEqual(lo[name], a, places=9)
            self.assertAlmostEqual(hi[name], b, places=9)

    def test_apply_touches_only_the_listed_fields(self):
        cfg = {"duration_s": 120.5, "tx_power_dbm": 10.0, "noise_floor_dbm": [-97.0, -92.0], "chip_noise_counts": 0.5,
               "tail_power_db": -15.0, "clutter_rcs_m2": [0.005, 0.2], "body_rcs_m2": 0.5,
               "breathing_p2p_m": [0.004, 0.012], "rssi_jitter_db": 1.0, "ripple_amp": 0.05, "seed": 7,
               "n_people": 2, "people_mode": {"1": "walk"}, "people_posture": {"0": "sit"}, "start_heading_rad": {},
               "other": "kept"}
        th = dict(cs.DEFAULT_THETA, noise_shift_db=5.0, clutter_scale=2.0, breathing_scale=3.0)
        c = cs.apply(cfg, th, "moving")
        self.assertEqual(c["noise_floor_dbm"], [-92.0, -87.0])
        self.assertEqual(c["clutter_rcs_m2"], [0.01, 0.4])
        self.assertAlmostEqual(c["breathing_p2p_m"][1], 0.036)
        self.assertEqual((c["n_people"], c["people_mode"], c["people_posture"]), (1, {"0": "walk"}, {}))
        self.assertEqual((c["seed"], c["other"], c["duration_s"]), (7, "kept", float(cs.SECONDS)))
        self.assertEqual(cfg["noise_floor_dbm"], [-97.0, -92.0])                 # the input is not modified
        self.assertEqual(cs.apply(cfg, th, "empty")["n_people"], 0)
        self.assertEqual(cs.apply(cfg, th, "still")["people_mode"], {"0": "stationary"})


class TestDesign(unittest.TestCase):
    def test_lhs_one_point_per_stratum(self):
        u = cc.lhs(16, 3, seed=1)
        for k in range(3):
            self.assertEqual(sorted(np.floor(u[:, k] * 16).astype(int).tolist()), list(range(16)))

    def test_quantiles(self):
        f = {g: np.zeros(0) for g in cf.GROUPS}
        f["log_sigma_empty"] = np.arange(100, dtype=float)
        q = cc.quantiles(f)
        self.assertEqual(q.shape, (len(cf.GROUPS), cc.J))
        np.testing.assert_allclose(q[0], np.quantile(np.arange(100.0), cc.TAUS))
        self.assertTrue(np.isnan(q[1]).all())


class TestFitToy(unittest.TestCase):
    def test_recovers_a_known_parameter(self):
        """Quantiles of N(mu(u), 1) per group with mu linear in u: the fit finds the true u."""
        torch.set_num_threads(2)
        rng = np.random.default_rng(0)
        g, p = len(cf.GROUPS), 2
        z = np.quantile(rng.standard_normal(200000), cc.TAUS)

        def q_of(u):
            mu = np.array([3 * u[0], -2 * u[1], u[0] + u[1], 0.5 * u[0], 2 * u[1], u[0], -u[1]])[:g]
            return (mu[:, None] + z[None, :]).reshape(-1)
        u = cc.lhs(64, p, seed=2)
        q = np.stack([q_of(x) for x in u])
        em, rep = cc.fit_emulator(u, q, steps=2500, holdout=8, seed=0)
        self.assertGreater(rep["holdout_r2"], 0.99)
        truth = np.array([0.3, 0.7])
        res = cc.fit(em, q_of(truth).reshape(g, cc.J), np.ones(g), starts=32, steps=800, seed=0)
        np.testing.assert_allclose(res["u"], truth, atol=0.05)


@unittest.skipUnless(os.path.exists(cs.SAANS) and os.path.isdir(cs.DATASET), "needs the release CLI and roomsim-v1")
class TestOneRun(unittest.TestCase):
    def test_a_room_runs_and_gives_every_feature(self):
        room = cs.rooms(1, seed=5)[0]
        res = cs.run_one((cs.DEFAULT_THETA, room, "empty", "/dev/shm"))
        self.assertTrue(res["ok"], res["error"])
        f = res["features"]
        for gname in ("log_sigma_empty", "fs_db_empty", "rssi_std_empty", "rssi_mean_empty"):
            self.assertGreater(f[gname].size, 0, gname)
            self.assertTrue(np.isfinite(f[gname]).all())
        self.assertEqual(f["log_sigma_moving"].size, 0)


if __name__ == "__main__":
    unittest.main()
