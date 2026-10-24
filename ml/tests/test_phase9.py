"""ml/phase9 (spec/phase9.md): fraction blocks, the block bootstrap, the
figures, the DSP zero-order hold and the REAL dataset's samples, on
hand-built cases; the REAL cache against `preprocess` when the dataset and
caches exist."""

import os
import unittest

import numpy as np

from ml.phase9 import baselines_real as br
from ml.phase9 import metrics_real as mr
from ml.phase9 import real_data as rdm


def _cache(n_per_seg=(10, 6, 4, 4), n_links=3, t=5, trial="envT_S3"):
    seg = np.concatenate([np.full(n, g, np.int16) for g, n in enumerate(n_per_seg)])
    w = seg.size
    pres = np.isin(seg, (1, 2)).astype(np.uint8)
    rng = np.random.default_rng(0)
    return {
        "segment": seg, "presence": pres, "count": pres.copy(), "moving": seg == 2,
        "t_end_ns": np.arange(w, dtype=np.int64) * 500_000_000, "eval": np.ones(w, bool),
        "links": np.array([[0, 1], [1, 0], [2, 0]], np.int16)[:n_links],
        "link_mask": np.ones((w, n_links), bool),
        "amp": rng.standard_normal((w, n_links, 40, t)).astype(np.float16),
        "mu": np.ones((w, n_links, 40), np.float16), "sigma": np.full((w, n_links, 40), 0.1, np.float16),
        "rssi": np.full((w, n_links, t), -40, np.float16), "sample_mask": np.ones((w, n_links, t), bool),
        "rssi_mask": np.ones((w, n_links, t), bool),
        "meta": {"label": "REAL", "trial": trial, "room_m": [5.0, 4.0]},
    }


class TestFractions(unittest.TestCase):
    def test_one_contiguous_block_per_segment(self):
        c = _cache()
        for f, want in ((0.01, [1, 1, 1, 1]), (0.5, [5, 3, 2, 2]), (1.0, [10, 6, 4, 4])):
            pos = rdm.fraction_positions(c, f, seed=3)
            counts = [int((c["segment"][pos] == g).sum()) for g in range(4)]
            self.assertEqual(counts, want)
            for g in range(4):
                p = pos[c["segment"][pos] == g]
                self.assertTrue((np.diff(p) == 1).all())                     # contiguous
        a = rdm.fraction_positions(c, 0.5, seed=1)
        self.assertTrue(np.array_equal(a, rdm.fraction_positions(c, 0.5, seed=1)))   # deterministic

    def test_refuses_a_bad_fraction(self):
        with self.assertRaises(ValueError):
            rdm.fraction_positions(_cache(), 0.0, 0)


class TestBlocks(unittest.TestCase):
    def test_blocks_stay_inside_segments(self):
        seg = np.array([0] * 7 + [1] * 3 + [2] * 5)
        b = mr.block_ids(seg, block=3)
        self.assertEqual(b.tolist(), [0, 0, 0, 1, 1, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5])

    def test_weights_keep_each_segment_size_in_blocks(self):
        seg = np.array([0] * 9 + [1] * 6)
        wt = mr.block_weights(seg, n_boot=50, seed=1, block=3)
        self.assertEqual(wt.shape, (50, 15))
        # every resample draws 3 blocks of 3 in segment 0 and 2 blocks of 3 in segment 1
        np.testing.assert_array_equal(wt[:, :9].sum(1), 9)
        np.testing.assert_array_equal(wt[:, 9:].sum(1), 6)
        # windows of one block share a weight
        np.testing.assert_array_equal(wt[:, 0], wt[:, 2])


class TestFigures(unittest.TestCase):
    def test_hand_built(self):
        truth = {"presence": np.array([0, 0, 1, 1, 1, 0], bool), "count": np.array([0, 0, 1, 1, 1, 0]),
                 "moving": np.array([0, 0, 0, 1, 1, 0], bool), "segment": np.array([0, 0, 1, 2, 2, 3]),
                 "leading_empty": np.array([1, 1, 0, 0, 0, 0], bool)}
        pred = {"presence_prob": np.array([0.1, 0.6, 0.4, 0.9, 0.8, 0.2]), "count_prob": None}
        f = {k: float(v[0]) for k, v in mr.figures(pred, truth, np.ones((1, 6))).items()}
        self.assertAlmostEqual(f["presence|accuracy"], 4 / 6)
        self.assertAlmostEqual(f["presence|false_alarm"], 1 / 3)
        self.assertAlmostEqual(f["presence|false_alarm_leading"], 1 / 2)
        self.assertAlmostEqual(f["presence|false_alarm_trailing"], 0.0)
        self.assertAlmostEqual(f["presence|miss_still"], 1.0)
        self.assertAlmostEqual(f["presence|miss_moving"], 0.0)
        self.assertAlmostEqual(f["presence|balanced_accuracy"], 0.5 * (2 / 3 + 2 / 3))
        # AUC: positives 0.4, 0.9, 0.8 against negatives 0.1, 0.6, 0.2: 8 of 9 pairs
        self.assertAlmostEqual(f["presence|auc"], 8 / 9)
        self.assertAlmostEqual(f["count|accuracy"], 4 / 6)
        self.assertAlmostEqual(f["coverage"], 1.0)

    def test_auc_score_overrides_the_probability_for_ranking(self):
        truth = {"presence": np.array([0, 1], bool), "count": np.array([0, 1]), "moving": np.array([0, 1], bool),
                 "segment": np.array([0, 1]), "leading_empty": np.array([1, 0], bool)}
        pred = {"presence_prob": np.array([0.9, 0.1]), "auc_score": np.array([0.0, 5.0])}
        f = mr.figures(pred, truth, np.ones((1, 2)))
        self.assertEqual(float(f["presence|auc"][0]), 1.0)
        self.assertEqual(float(f["presence|accuracy"][0]), 0.0)

    def test_evaluate_refuses_an_unlabelled_method(self):
        truth = {"presence": np.array([0, 1], bool), "count": np.array([0, 1]), "moving": np.array([0, 1], bool),
                 "segment": np.array([0, 1]), "leading_empty": np.array([1, 0], bool)}
        with self.assertRaises(ValueError):
            mr.evaluate({"m": {"presence_prob": np.array([0.2, 0.8])}}, truth, label_of={"m": "SIM"}, n_boot=10)
        r = mr.evaluate({"m": [{"presence_prob": np.array([0.2, 0.8])}] * 3}, truth, label_of={"m": "SIM->REAL"},
                        n_boot=10)
        self.assertEqual(r["m"]["label"], "SIM->REAL")
        self.assertEqual(r["m"]["n_seeds"], 3)


class TestDspHold(unittest.TestCase):
    def test_zero_order_hold(self):
        t = np.array([120, 130, 400], np.int64) * 1_000_000                  # ms -> ns
        ticks, idx = br.zoh_ticks(t, 0, 500_000_000, rate_hz=20.0)
        # ticks every 50 ms from 0; the first at or after 120 ms is 150 ms
        self.assertEqual((ticks // 1_000_000).tolist(), [150, 200, 250, 300, 350, 400, 450, 500])
        self.assertEqual(idx.tolist(), [1, 1, 1, 1, 1, 2, 2, 2])


class TestRealDataset(unittest.TestCase):
    def test_samples_have_the_model_batch_keys_and_unknown_geometry(self):
        from ml.train import data as d8
        c = _cache()
        ds = rdm.RealDataset([c], [(0, w) for w in range(c["segment"].size)], augment=True)
        keys = list(iter(rdm.FixedStepSampler(len(ds), steps=2, batch=4, seed=0)))
        self.assertEqual(len(keys), 8)
        batch = d8.collate([ds[k] for k in keys])
        for key in d8.BATCH_KEYS:
            self.assertIn(key, batch)
        self.assertFalse(batch["geometry_known"].any())
        self.assertFalse(batch["zone_reachable"].any())
        self.assertTrue(all(m["label"] == "REAL" for m in batch["meta"]))
        s = ds.sample(0, 17)                                                 # segment 2: moving
        self.assertEqual((float(s["presence"]), int(s["count"])), (1.0, 1))
        self.assertTrue(s["walked"][0])

    def test_refuses_a_cache_not_labelled_real(self):
        c = _cache()
        c["meta"] = dict(c["meta"], label="SIM")
        with self.assertRaises(ValueError):
            rdm.RealDataset([c], [(0, 0)])


@unittest.skipUnless(os.environ.get("SAANS_OPENCSI") and os.path.isdir("/home/Aman/are_u_theone/runs/phase9/real_cache"),
                     "set SAANS_OPENCSI=1 with the REAL caches built")
class TestRealCache(unittest.TestCase):
    def test_a_cached_window_equals_preprocess(self):
        from ml import opencsi as oc
        from ml import preprocess as pp
        from ml.phase9 import real_cache as rc
        c = rc.load("envC_C3", 2.0)
        s = oc.load_session(oc.DEFAULT_ROOT, "envC_C3")
        for w in (0, 700, c["segment"].size - 1):
            out = pp.preprocess(oc.window(s, int(c["t_end_ns"][w]), 2.0), rc.PREPROCESS)
            ref = out["ref"]
            np.testing.assert_array_equal(c["amp"][w], out["amp"][:, ref].astype(np.float16))
            np.testing.assert_array_equal(c["sigma"][w], out["sigma"][:, ref].astype(np.float16))
            np.testing.assert_array_equal(c["link_mask"][w], out["link_mask"])
        self.assertEqual(c["meta"]["label"], "REAL")
        ev = c["eval"]
        first = c["meta"]["segments"][0][0]
        self.assertTrue((c["t_end_ns"][ev] - 2_000_000_000 >= first + 90_000_000_000).all())


if __name__ == "__main__":
    unittest.main()
