"""Phase 8 baselines (spec/model.md §3, §7): the variance features and the
link-variance and count maps by hand-built cases, the prior, the stored
baselines' mapping onto the predictions contract, the forest and MLP
plumbing, and — on the real dataset (SAANS_DATASET_DIR) — the recomputed
WLM and RTI equal to the stored outcome when `window_s` = 1 s and every
link is kept. SIM.

Run from the repo root: .venv/bin/python -m unittest discover -s ml/tests
"""

import math
import os
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml import roomsim_dataset as rd  # noqa: E402
from ml.tests import test_roomsim_dataset as trd  # noqa: E402
from ml.tests import test_train_cache as ttc  # noqa: E402
from ml.train import baselines as bl  # noqa: E402
from ml.train import cache  # noqa: E402
from ml.train import metrics as M  # noqa: E402

DATASET = os.environ.get("SAANS_DATASET_DIR")


def lp(*segments):
    """link_pos [L, 2, 3] from 2-D segments ((x0, y0), (x1, y1)) at z = 1."""
    return np.array([[[a[0], a[1], 1.0], [b[0], b[1], 1.0]] for a, b in segments])


class TestMaps(unittest.TestCase):
    def test_segment_distance(self):
        a, b = np.array([0.0, 0.0]), np.array([1.0, 0.0])
        self.assertAlmostEqual(float(M.np.asarray(bl.segment_distance(np.array([0.5, 0.3]), a, b))), 0.3)
        self.assertAlmostEqual(float(bl.segment_distance(np.array([2.0, 0.0]), a, b)), 1.0)      # beyond b
        self.assertAlmostEqual(float(bl.segment_distance(np.array([-3.0, 4.0]), a, b)), 5.0)     # before a
        self.assertAlmostEqual(float(bl.segment_distance(np.array([3.0, 4.0]), a, a)), 5.0)      # a point

    def test_maps_by_hand(self):
        # 4 x 4 m room, normalised = metres / 4. Cell (i, j) centre ((i + .5)/4, (j + .5)/4).
        # A: y = 0.5 m (0.125 normalised), x 0..4: through every j = 0 centre; j = 1 centres 0.25 away.
        # B: the diagonal: through (i, i); (i, i +- 1) are 0.25/sqrt(2) = 0.177 > 1/8 away.
        # C: (0.2, 3.8)-(0.3, 3.9): its end (0.05, 0.95) is sqrt(2) * 0.075 = 0.106 from (0, 3)'s centre.
        links = lp(((0, 0.5), (4, 0.5)), ((0, 0), (4, 4)), ((0.2, 3.8), (0.3, 3.9)))
        room = np.array([4.0, 4.0, 2.5])
        s = np.array([1.0, 2.0, 4.0])
        mean, cnt = bl.link_variance_maps(links, room, s, np.ones(3, bool))
        want_cnt = np.zeros((4, 4))
        want_mean = np.zeros((4, 4))
        for i in range(4):
            want_cnt[i, 0] += 1
            want_cnt[i, i] += 1
        want_cnt[0, 3] = 1
        want_mean[0, 0] = 1.5                                   # A and B
        want_mean[1:, 0] = 1.0                                  # A
        for i in range(1, 4):
            want_mean[i, i] = 2.0                               # B
        want_mean[0, 3] = 4.0                                   # C
        np.testing.assert_array_equal(cnt, want_cnt)
        np.testing.assert_allclose(mean, want_mean)
        # an unused link counts nowhere
        mean, cnt = bl.link_variance_maps(links, room, s, np.array([True, True, False]))
        self.assertEqual((mean[0, 3], cnt[0, 3]), (0.0, 0.0))
        # within 1/8 is inclusive: a line at normalised y = 0.25 touches j = 0 and j = 1 centres (0.125 away)
        mean, cnt = bl.link_variance_maps(lp(((0, 1.0), (4, 1.0))), room, np.array([3.0]), np.ones(1, bool))
        np.testing.assert_array_equal(cnt[:, :2], np.ones((4, 2)))
        np.testing.assert_array_equal(cnt[:, 2:], np.zeros((4, 2)))
        # a non-square room normalises each axis by its own length
        mean, cnt = bl.link_variance_maps(lp(((0, 1.0), (8, 1.0))), np.array([8.0, 8.0, 3]), np.array([1.0]),
                                          np.ones(1, bool))
        np.testing.assert_array_equal(cnt[:, 0], np.ones(4))
        self.assertEqual(cnt.sum(), 4)

    def test_stats_by_hand(self):
        s = np.array([[1.0, 2.0, 3.0, 4.0, 9.0]])
        used = np.array([[True, True, True, True, False]])
        st = bl._stats(s, used)[0]
        np.testing.assert_allclose(st, [2.5, math.sqrt(1.25), 1, 4, 1.75, 2.5, 3.25, 3.7])
        self.assertTrue((bl._stats(s, np.zeros_like(used)) == 0).all())
        self.assertEqual(len(bl.FEATURE_NAMES), bl.N_FEATURES)
        self.assertEqual(bl.N_FEATURES, 8 + 1 + 3 + 16 + 16)

    def test_centroid(self):
        np.testing.assert_allclose(bl.footprint_centroid([(0, 0), (6, 0), (6, 4), (0, 4)]), (3, 2))
        # L: (0..6 x 0..2.7) area 16.2 at (3, 1.35) and (0..3.6 x 2.7..4.5) area 6.48 at (1.8, 3.6)
        l_shape = [(0, 0), (6, 0), (6, 2.7), (3.6, 2.7), (3.6, 4.5), (0, 4.5)]
        want = ((16.2 * 3 + 6.48 * 1.8) / 22.68, (16.2 * 1.35 + 6.48 * 3.6) / 22.68)
        np.testing.assert_allclose(bl.footprint_centroid(l_shape), want)
        padded = np.full((6, 2), np.nan)
        padded[:4] = [(0, 0), (6, 0), (6, 4), (0, 4)]
        np.testing.assert_allclose(bl.footprint_centroid(padded), (3, 2))

    def test_calib_min_windows(self):
        self.assertEqual([bl.calib_min_windows(15.0, w) for w in (0.5, 1.0, 2.0, 4.0)], [10, 10, 6, 2])


class TestSubsetGeometry(unittest.TestCase):
    def test_subset(self):
        from wifisense.localize.geometry import Geometry, Node, PixelGrid, Room
        from wifisense.localize.invert import Inverter
        room = Room(4.0, 3.0)
        nodes = [Node(0, 0.2, 0.2), Node(1, 3.8, 0.2), Node(2, 3.8, 2.8), Node(3, 0.2, 2.8)]
        full = Geometry(room, nodes, PixelGrid(room, 0.5), 0.3)
        g = bl.subset_geometry(full, {(0, 1), (0, 3), (2, 0)})
        self.assertEqual([(lk.tx_id, lk.rx_id) for lk in g.links], [(0, 1), (0, 3), (2, 0)])
        self.assertEqual(g.link_index(2, 0), 2)
        with self.assertRaises(KeyError):
            g.link_index(1, 0)
        self.assertEqual(len(g.supports()), 3)
        inv = Inverter(g, warn_blind_pixels=False)
        self.assertEqual(len(inv.solve([1.0, 0.0, 0.0])), g.n_pixels)
        one = bl.subset_geometry(full, {(0, 1)})
        self.assertEqual(one.n_links, 1)
        with self.assertRaises(ValueError):
            bl.subset_geometry(full, {(5, 6)})


class SynthCase(unittest.TestCase):
    """A synthetic SIM dataset of 12 s scenes (eval windows k = 7..23), all people counts in each split."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = os.path.join(cls._tmp.name, "synth-sim")
        specs = [{"n_nodes": 4 + (i % 3), "n_people": i % 4} for i in range(12)]
        split_of = {i: "train" for i in range(8)}
        split_of.update({8: "test", 9: "val", 10: "val", 11: "test"})      # test: 0 and 3 people
        cls.arrays = ttc.write_synthetic(cls.dataset, specs, split_of, seconds=12)
        meta = cache.build_cache(cls.dataset, 2.0, os.path.join(cls._tmp.name, "cache"), workers=2, progress=False)
        cls.c = cache.open_cache(meta["out_dir"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestFeaturesOnCache(SynthCase):
    def test_window_features(self):
        c = self.c
        ws = c.window_indices("train", min_t=4.0)
        x = bl.window_features(c, ws)
        self.assertEqual(x.shape, (ws.size, bl.N_FEATURES))
        for row, w in zip(x[::7], ws[::7]):
            one = c.read_window(int(w))
            s = one["sigma"].astype(np.float32).astype(np.float64).mean(1)
            used = one["link_mask"]
            mean, cnt = bl.link_variance_maps(one["link_pos"], one["room"], s, used)
            np.testing.assert_allclose(row[12:28], mean.ravel())
            np.testing.assert_array_equal(row[28:44], cnt.ravel())
            np.testing.assert_allclose(row[0:8], bl._stats(s[None], used[None])[0])
            self.assertEqual(row[8], one["n_nodes"])
            np.testing.assert_allclose(row[9:12], one["room"])

    def test_stored(self):
        truth = M.truth_from_cache(self.c, "test")
        p = bl.stored_predictions(self.c, truth, "wlm")
        # make_arrays: WLM published at odd k except 7, centre (2 + 0.1 k, 2), radius 1.5
        k = truth["k"]
        want_pub = (k % 2 == 1) & (k != 7)
        np.testing.assert_array_equal(p["published"], want_pub)
        np.testing.assert_array_equal(p["presence_prob"], want_pub.astype(float))
        np.testing.assert_allclose(p["slot_xy"][want_pub, 0], np.stack([2 + 0.1 * k[want_pub], np.full(want_pub.sum(), 2.0)], 1),
                                   rtol=1e-6)
        np.testing.assert_allclose(p["slot_radius"][want_pub, 0], 1.5)
        self.assertTrue(p["uses_calibration"])
        q = bl.stored_predictions(self.c, truth, "rti")
        self.assertFalse(q["published"].any())                # RTI published at k = 3 only
        M.evaluate({"wlm": p, "rti": q}, truth, n_boot=20)

    def test_prior(self):
        truth = M.truth_from_cache(self.c, "test")
        p = bl.prior_predictions(self.c, truth)
        tr = self.c.window_indices("train", min_t=4.0)
        cnt = self.c.windows["count"][tr]
        self.assertAlmostEqual(p["presence_prob"][0], float((cnt > 0).mean()))
        np.testing.assert_allclose(p["count_prob"][0], np.bincount(cnt, minlength=4) / cnt.size)
        self.assertTrue((p["slot_xy"][:, 0] == np.array([trd.W / 2, trd.D / 2])).all())   # rectangle centroid
        one = cnt == 1
        xy = self.c.windows["xy"][tr][one, 0]
        err = np.hypot(xy[:, 0] - trd.W / 2, xy[:, 1] - trd.D / 2)
        self.assertAlmostEqual(p["slot_radius"][0, 0], max(np.percentile(err, 95), 0.3), places=5)
        res = M.evaluate({"prior": p}, truth, n_boot=0)
        f = res["methods"]["prior"]["figures"]["presence"]["all"]
        self.assertAlmostEqual(f["auc"]["value"], 0.5)                                   # a constant score

    def test_variance_models(self):
        res = bl.fit_variance_baselines(self.c, ("test",), seeds=(0, 1), n_estimators=5, n_jobs=2, mlp_threads=1)
        truth = M.truth_from_cache(self.c, "test")
        for kind in ("forest", "mlp"):
            preds = res["test"][kind]
            self.assertEqual(len(preds), 2)
            for p in preds:
                q = M.check_predictions(p)
                self.assertEqual(q["kind"], "one_estimate")
                self.assertTrue(q["published"].all())
                self.assertTrue((q["slot_radius"][:, 0] >= 0.3).all())
                np.testing.assert_allclose(q["count_prob"].sum(1), 1.0)
                self.assertTrue(((q["zone_prob"] >= 0) & (q["zone_prob"] <= 1)).all())
                w = truth["room"][:, :2]
                self.assertTrue(((q["slot_xy"][:, 0] >= 0) & (q["slot_xy"][:, 0] <= w)).all())
        # deterministic for a seed
        again = bl.fit_variance_baselines(self.c, ("test",), seeds=(0,), n_estimators=5, n_jobs=2, mlp_threads=1)
        for kind in ("forest", "mlp"):
            np.testing.assert_array_equal(again["test"][kind][0]["slot_xy"], res["test"][kind][0]["slot_xy"])
            np.testing.assert_array_equal(again["test"][kind][0]["presence_prob"], res["test"][kind][0]["presence_prob"])
        out = M.evaluate(bl.baseline_methods(self.c, truth, variance=res, split="test"), truth, n_boot=20,
                         reference="forest")
        self.assertEqual(set(out["methods"]), {"wlm_stored", "rti_stored", "prior", "forest", "mlp"})
        self.assertEqual(out["methods"]["forest"]["seeds"], 2)
        # save and load round trip
        path = os.path.join(self._tmp.name, "vb.npz")
        bl.save_predictions(path, res)
        back = bl.load_predictions(path)
        np.testing.assert_array_equal(back["test"]["mlp"][1]["slot_radius"], res["test"]["mlp"][1]["slot_radius"])
        self.assertEqual(back["test"]["forest"][0]["seed"], 0)


@unittest.skipUnless(DATASET, "SAANS_DATASET_DIR names no real dataset")
class TestRecomputedEqualsStored(unittest.TestCase):
    """spec/model.md §7: the recomputed WLM (and RTI) equal the stored ones at window_s = 1 s, every link."""

    def scenes(self):
        m = rd.load_manifest(DATASET)
        by = {}
        for e in m["scenes"]:
            by.setdefault((e["n_nodes"], e["l_shape"]), e["index"])
        return m, [by[(4, True)], by[(9, False)], by[(12, False)]]

    def test_equal(self):
        m, scenes = self.scenes()
        for idx in scenes:
            res = bl.recompute_scene(DATASET, idx, lengths=(1.0,), subsets=("all",), manifest=m)[(1.0, "all")]
            scene = rd.load_scene(DATASET, idx, manifest=m)
            last = len(scene["t_label"]) - 1
            ks = np.arange(1, last)                 # every tick but warming up (k = 0) and t = S (spec/dataset.md §6)
            for name in ("wlm", "rti"):
                np.testing.assert_array_equal(res[f"{name}_reason"][ks], scene[f"{name}_reason"][ks], f"{idx} {name}")
                pub = scene[f"{name}_reason"][ks] == 0
                np.testing.assert_array_equal(res[f"{name}_xy"][ks][pub].astype(np.float32), scene[f"{name}_xy"][ks][pub])
                np.testing.assert_array_equal(res[f"{name}_radius"][ks][pub].astype(np.float32),
                                              scene[f"{name}_radius"][ks][pub])
                self.assertEqual(int(res[f"{name}_reason"][0]), 4)
            self.assertEqual(res["min_windows"], 10)

    def test_calibration_equals_stored(self):
        from wifisense.localize.pipeline import PersonPipeline, base_rate_hz, locate_geometry
        m, scenes = self.scenes()
        for idx in scenes:
            inp = bl.locate_inputs(DATASET, idx, m)
            cfg = inp["cfg"]
            geom = locate_geometry(inp["scene_obj"], cfg)
            person = PersonPipeline(geom, cfg, base_rate_hz(int(inp["scene"]["meta"]["n_tx"]), inp["slot_s"]))
            n, t, tx, rx, rssi, nz = inp["calib"]
            person.calibrate((float(a), int(b), int(c), bl._Frame(bool(d), int(e))) for a, b, c, d, e in
                             zip(t, tx, rx, nz, rssi))
            np.testing.assert_array_equal(person.meas.baseline_vector(), inp["scene"]["calib_baseline"])
            np.testing.assert_array_equal(person.meas.baseline_std_vector(), inp["scene"]["calib_null_sigma"])
            np.testing.assert_array_equal(person.meas.calibrated_mask(), inp["scene"]["calib_mask"].astype(bool))

    def test_subsets_and_lengths_run(self):
        m, scenes = self.scenes()
        res = bl.recompute_scene(DATASET, scenes[1], lengths=(2.0, 4.0), subsets=bl.SUBSETS, manifest=m)
        self.assertEqual(set(res), {(w, s) for w in (2.0, 4.0) for s in bl.SUBSETS})
        for (w, s), r in res.items():
            first = int(2 * w) - 1
            self.assertTrue((r["wlm_reason"][:first] == 4).all())
            self.assertTrue((r["wlm_reason"][first:] != 4).all())
            pub = r["wlm_reason"] == 0
            self.assertTrue((r["wlm_radius"][pub] >= 0.3).all())
            self.assertEqual(r["n_links"], {"all": 72, "tx0": 8, "single01": 1}[s])
            self.assertEqual(r["min_windows"], {2.0: 6, 4.0: 2}[w])


if __name__ == "__main__":
    unittest.main()
