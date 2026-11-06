"""Phase 8 metrics (spec/model.md §4, §7): Hungarian matching, every
figure and the scene bootstrap against hand-built cases. SIM.

The hand-built case (`case()`): 6 windows in 3 scenes of a 6 x 4 m room
(cell 2 unreachable): scene 10 empty (w0, w1), scene 11 one person at
(1, 1) who walked in w2 and not in w3, scene 12 two still people at
(1, 1) and (4, 3) (w4, w5). Every expected number below is worked out by
hand in the comments.

Run from the repo root: .venv/bin/python -m unittest discover -s ml/tests
"""

import os
import sys
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.train import metrics as M  # noqa: E402

R = M.R95


def case():
    """The hand-built truth and a slot model's predictions (module docstring)."""
    scene = [10, 10, 11, 11, 12, 12]
    k = [7, 9, 7, 9, 7, 9]
    count = [0, 0, 1, 1, 2, 2]
    xy = np.zeros((6, 3, 2))
    pm = np.zeros((6, 3), bool)
    for w in (2, 3, 4, 5):
        xy[w, 0] = (1, 1)
        pm[w, 0] = True
    for w in (4, 5):
        xy[w, 1] = (4, 3)
        pm[w, 1] = True
    walked = np.zeros((6, 3), bool)
    walked[2, 0] = True
    occ = np.zeros((6, 9), bool)
    occ[2:, 0] = True                                         # (1, 1): col floor(1/2) = 0, row floor(3/4) = 0
    occ[4:, 8] = True                                         # (4, 3): col 2, row floor(9/4) = 2
    reach = np.ones((6, 9), bool)
    reach[:, 2] = False
    truth = M.make_truth(scene, k, count, xy, pm, walked, occ, reach, np.tile([6.0, 4.0, 3.0], (6, 1)),
                         split="test", dataset="hand")
    pres = np.array([0.2, 0.7, 0.9, 0.4, 0.8, 0.6])
    cp = np.full((6, 4), 0.1)
    for w, c in enumerate([0, 1, 1, 2, 2, 0]):              # argmax
        cp[w, c] = 0.7
    zp = np.full((6, 9), 0.1)
    zp[1, 4] = 0.8                                            # w1 {4}
    zp[2, 0] = 0.9                                            # w2 {0}
    zp[3, 0], zp[3, 2] = 0.9, 0.95                            # w3 {0, 2}; 2 unreachable
    zp[4, 0] = 0.7                                            # w4 {0}
    zp[5, 0], zp[5, 8] = 0.6, 0.9                             # w5 {0, 8}
    ex = np.zeros((6, 3))
    sxy = np.zeros((6, 3, 2))
    sig = np.full((6, 3), 0.5)
    ex[1, 0], sxy[1, 0] = 0.6, (3, 2)                         # false estimate in an empty window
    ex[2, :2], sxy[2, 0], sxy[2, 1], sig[2, :2] = (0.9, 0.7), (1.3, 1.4), (5, 3), (0.25, 0.1)
    ex[3], sxy[3, 1], sig[3, 1] = (0.3, 0.8, 0.2), (1, 2.5), 0.5
    ex[4, :2], sxy[4, 0], sxy[4, 1], sig[4, :2] = (0.9, 0.9), (1, 1.2), (4.5, 3), (0.3, 0.3)
    ex[5, 0], sxy[5, 0], sig[5, 0] = 0.6, (2.4, 2), 1.0
    pred = M.predictions_from_slots(scene, k, pres, cp, zp, ex, sxy, sig, name="model")
    return truth, pred


def wlm_case(truth):
    """WLM-like: published at w0 (a false alarm) and w2 (centre (1, 1), radius 3)."""
    xy = np.zeros((6, 2))
    xy[0] = (3, 2)
    xy[2] = (1, 1)
    pub = np.array([1, 0, 1, 0, 0, 0], bool)
    return M.one_estimate_predictions(truth["scene"], truth["k"], xy, np.full(6, 3.0), pub,
                                      presence_prob=pub.astype(float), name="wlm", uses_calibration=True)


def fig(res, name, path):
    node = res["methods"][name]["figures"]
    for p in path.split("|"):
        node = node[p]
    return node["value"] if "value" in node else node["mean"]


class TestMatch(unittest.TestCase):
    def test_hungarian_beats_greedy(self):
        # A=(0,0), B=(2,0); P=(1,0), Q=(3.5,0): greedy may take B-P (1) then A-Q (3.5) = 4.5;
        # the optimum is A-P (1) + B-Q (1.5) = 2.5.
        r, c, d = M.match([(0, 0), (2, 0)], [(1, 0), (3.5, 0)])
        self.assertEqual(sorted(zip(r.tolist(), c.tolist())), [(0, 0), (1, 1)])
        np.testing.assert_allclose(sorted(d), [1.0, 1.5])

    def test_rectangular_and_empty(self):
        r, c, d = M.match([(0, 0), (10, 0), (5, 5)], [(5, 4)])
        self.assertEqual((r.tolist(), c.tolist()), ([2], [0]))
        np.testing.assert_allclose(d, [1.0])
        r, c, d = M.match(np.zeros((0, 2)), [(1, 1)])
        self.assertEqual(r.size, 0)

    def test_radius_and_cells(self):
        p = {"slot_sigma": np.array([[0.1, 0.2, 1.0]])}
        np.testing.assert_allclose(M.slot_radius(p), [[0.3, R * 0.2, R]])          # max(2.448 sigma, 0.3)
        p = {"slot_sigma": None, "slot_radius": np.array([[0.7, 0.8, 0.9]])}
        np.testing.assert_allclose(M.slot_radius(p), [[0.7, 0.8, 0.9]])
        # spec/dataset.md §5: col = min(floor(3x/w), 2), row = min(floor(3y/d), 2)
        cells = M.zone_cell(np.array([[0.1, 0.1], [5.99, 0.1], [6.0, 4.0], [2.0, 4 / 3], [-0.1, 2.0]]),
                            np.array([6.0, 4.0, 3.0]))
        self.assertEqual(cells.tolist(), [0, 2, 8, 4, 3])


class TestFigures(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.truth, cls.pred = case()
        cls.res = M.evaluate({"model": cls.pred, "wlm": wlm_case(cls.truth)}, cls.truth, reference="model",
                             n_boot=200, seed=1)

    def g(self, path, name="model"):
        return fig(self.res, name, path)

    def test_labels(self):
        self.assertEqual(self.res["label"], "SIM")
        self.assertEqual(self.res["methods"]["model"]["label"], "SIM")
        self.assertTrue(self.res["methods"]["wlm"]["uses_calibration"])
        self.assertEqual(self.res["n_windows_by_group"],
                         {"all": 6, "count=0": 2, "count=1": 2, "count=2": 2, "count=3": 0, "walking": 1,
                          "still": 3, "empty": 2})

    def test_presence(self):
        # y = [0,0,1,1,1,1], yhat = [0,1,1,0,1,1]
        self.assertAlmostEqual(self.g("presence|all|accuracy"), 4 / 6)
        self.assertAlmostEqual(self.g("presence|all|miss_rate"), 1 / 4)
        self.assertAlmostEqual(self.g("presence|all|false_alarm_rate"), 1 / 2)
        self.assertAlmostEqual(self.g("presence|all|balanced_accuracy"), 0.625)
        # positives {0.9, 0.4, 0.8, 0.6} vs negatives {0.2, 0.7}: 2 + 1 + 2 + 1 = 6 of 8 pairs
        self.assertAlmostEqual(self.g("presence|all|auc"), 0.75)
        self.assertAlmostEqual(self.g("presence|count=0|accuracy"), 0.5)
        self.assertIsNone(self.g("presence|count=0|balanced_accuracy"))     # one class: undefined
        self.assertAlmostEqual(self.g("presence|still|accuracy"), 2 / 3)       # w3 missed
        # WLM: presence = published = [1,0,1,0,0,0]: correct w1, w2
        self.assertAlmostEqual(self.g("presence|all|accuracy", "wlm"), 2 / 6)

    def test_count(self):
        self.assertAlmostEqual(self.g("count|all|accuracy"), 0.5)
        self.assertAlmostEqual(self.g("count|all|mae"), 4 / 6)
        conf = self.res["methods"]["model"]["tables"]["count_confusion"]
        want = np.zeros((4, 4), int)
        for t, p in ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2), (2, 0)):
            want[t, p] += 1
        self.assertEqual(conf, want.tolist())

    def test_zone(self):
        # cell 0: TP 4 -> F1 1; cell 4: FP 1 -> F1 0; cell 8: TP 1, FN 1 -> 2/3; others undefined
        self.assertAlmostEqual(self.g("zone|all|f1_macro"), (1 + 0 + 2 / 3) / 3)
        self.assertAlmostEqual(self.g("zone|all|f1_cell0"), 1.0)
        self.assertAlmostEqual(self.g("zone|all|f1_cell8"), 2 / 3)
        self.assertIsNone(self.g("zone|all|f1_cell2"))                         # unreachable everywhere
        # exact set over reachable cells: w0, w2, w3 (cell 2 ignored), w5
        self.assertAlmostEqual(self.g("zone|all|exact_set_accuracy"), 4 / 6)
        cells = self.res["methods"]["model"]["tables"]["zone_cells"]
        self.assertEqual((cells[8]["tp"], cells[8]["fn"], cells[8]["fp"]), (1, 1, 0))
        self.assertEqual(cells[2]["reachable_windows"], 0)
        self.assertEqual(cells[4]["fp"], 1)

    def test_xy(self):
        # 2 m gate: matched errors w2 0.5, w3 1.5, w4 0.2 and 0.5, w5 1.7205; 6 people, 7 estimates
        e5 = np.hypot(1.4, 1.0)
        errs = [0.5, 1.5, 0.2, 0.5, e5]
        self.assertAlmostEqual(self.g("xy2m|all|median_error"), 0.5)
        self.assertAlmostEqual(self.g("xy2m|all|p90_error"), np.percentile(errs, 90))
        self.assertAlmostEqual(self.g("xy2m|all|recall"), 5 / 6)
        self.assertAlmostEqual(self.g("xy2m|all|precision"), 5 / 7)
        # covered: 0.5 <= 2.448*0.25, 1.5 > 2.448*0.5, 0.2, 0.5 <= 0.734, 1.72 <= 2.448
        self.assertAlmostEqual(self.g("xy2m|all|coverage"), 4 / 5)
        self.assertAlmostEqual(self.g("xy2m|all|radius_median"), R * 0.3)
        # 1 m gate: w3 (1.5) and w5 (1.72) become a miss and a false estimate
        self.assertAlmostEqual(self.g("xy1m|all|recall"), 3 / 6)
        self.assertAlmostEqual(self.g("xy1m|all|precision"), 3 / 7)
        self.assertAlmostEqual(self.g("xy1m|all|median_error"), 0.5)
        # groups: the empty windows' one estimate is false (precision 0, recall undefined)
        self.assertAlmostEqual(self.g("xy2m|empty|precision"), 0.0)
        self.assertIsNone(self.g("xy2m|empty|recall"))
        self.assertAlmostEqual(self.g("xy2m|walking|median_error"), 0.5)
        # per person: the walker (w2) matched at 0.5; still people 4 of 5 matched
        self.assertAlmostEqual(self.g("xy2m|person:walking|recall"), 1.0)
        self.assertAlmostEqual(self.g("xy2m|person:still|recall"), 4 / 5)
        self.assertAlmostEqual(self.g("xy2m|person:still|median_error"), np.median([1.5, 0.2, 0.5, e5]))
        # WLM against the nearest present person: w2 at 0 m; w0's estimate is false
        self.assertAlmostEqual(self.g("xy2m|all|median_error", "wlm"), 0.0)
        self.assertAlmostEqual(self.g("xy2m|all|precision", "wlm"), 1 / 2)
        self.assertAlmostEqual(self.g("xy2m|all|recall", "wlm"), 1 / 6)

    def test_one_estimate(self):
        # model: best slots w2 slot0 (err 0.5, r 0.612, covered), w3 slot1 (err 1.5, r 1.224, not covered);
        # zone: most probable reachable cell 0 in both (w3's cell 2 is unreachable): hits
        self.assertAlmostEqual(self.g("one|all|median_error"), 1.0)
        self.assertAlmostEqual(self.g("one|all|p90_error"), 1.4)
        self.assertAlmostEqual(self.g("one|all|zone_accuracy"), 1.0)
        self.assertAlmostEqual(self.g("one|all|radius_coverage"), 0.5)
        self.assertAlmostEqual(self.g("one|all|coverage"), 1.0)
        self.assertAlmostEqual(self.g("one|walking|median_error"), 0.5)
        # WLM (a): publishes w2 of the one-person windows {w2, w3}
        self.assertAlmostEqual(self.g("one|all|coverage", "wlm"), 0.5)
        self.assertAlmostEqual(self.g("one|all|median_error", "wlm"), 0.0)
        self.assertAlmostEqual(self.g("one|all|zone_accuracy", "wlm"), 1.0)
        self.assertAlmostEqual(self.g("one|all|radius_coverage", "wlm"), 1.0)
        # (b) paired on WLM's published one-person windows {w2}: 0.5 vs 0.0
        pair = self.res["one_estimate_paired"]["model vs wlm"]
        self.assertEqual(pair["n_windows"], 1)
        self.assertEqual(pair["label"], "SIM")
        e = pair["paired"]["all"]["median_error"]
        self.assertAlmostEqual(e["model"], 0.5)
        self.assertAlmostEqual(e["wlm"], 0.0)
        self.assertAlmostEqual(e["diff"], 0.5)
        self.assertIn("ci95", e)

    def test_zone1_and_differences(self):
        # single-estimate cell vs zone_occupied: WLM published w0 (empty: miss) and w2 (cell 0: hit)
        self.assertAlmostEqual(self.g("zone1|all|hit_rate", "wlm"), 0.5)
        self.assertAlmostEqual(self.g("zone1|all|coverage", "wlm"), 2 / 6)
        d = self.res["differences"]["model - wlm"]["presence"]["all"]["accuracy"]
        self.assertAlmostEqual(d["diff"], 4 / 6 - 2 / 6)
        self.assertEqual(len(d["ci95"]), 2)


class TestWeights(unittest.TestCase):
    def test_weighted_quantile_is_repeated_percentile(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            v = np.sort(rng.normal(size=rng.integers(1, 30)))
            w = rng.integers(0, 4, size=(5, v.size))
            w[0] = 1
            for q in (0.5, 0.9, 0.0, 1.0):
                got = M.weighted_quantile(v, w, q)
                for row in range(5):
                    rep = np.repeat(v, w[row])
                    want = np.percentile(rep, 100 * q) if rep.size else np.nan
                    if rep.size:
                        self.assertAlmostEqual(got[row], want, places=12)
                    else:
                        self.assertTrue(np.isnan(got[row]))

    def test_weighted_auc(self):
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(1)
        s = np.round(rng.random(40), 1)                       # ties
        y = rng.random(40) < 0.4
        w = rng.integers(0, 3, size=(4, 40))
        got = M.weighted_auc(s, y, w)
        for row in range(4):
            keep = w[row] > 0
            want = roc_auc_score(y[keep], s[keep], sample_weight=w[row][keep])
            self.assertAlmostEqual(got[row], want, places=12)
        self.assertTrue(np.isnan(M.weighted_auc(s, np.zeros(40, bool), np.ones((1, 40))))[0])

    def test_resample_equals_duplicated_windows(self):
        """A resample's weights equal scoring the drawn scenes' windows duplicated (every window kept)."""
        truth, pred = case()
        at = M.score(pred, truth)
        counts = np.array([[2, 0, 1]])                        # scene 10 twice, 11 never, 12 once
        wt = counts[:, truth["scene_pos"]].astype(float)
        weighted = M._figures(at, truth, wt)
        weighted.update(M._one_figures(at["one"], wt))
        rows = [w for w in range(6) for _ in range(int(wt[0, w]))]
        copies = {}
        new_scene = []
        for w in rows:
            copies[w] = copies.get(w, 0) + 1
            new_scene.append(truth["scene"][w] * 10 + copies[w])
        sub = M.make_truth(new_scene, truth["k"][rows], truth["count"][rows], truth["xy"][rows],
                           truth["person_mask"][rows], truth["walked"][rows], truth["zone_occupied"][rows],
                           truth["zone_reachable"][rows], truth["room"][rows])
        p2 = {key: (v[rows] if isinstance(v, np.ndarray) and v.shape[:1] == (6,) else v) for key, v in pred.items()}
        p2["scene"] = np.array(new_scene)
        p2["k"] = truth["k"][rows]
        at2 = M.score(p2, sub)
        plain = M._figures(at2, sub, np.ones((1, len(rows))))
        for key, val in plain.items():
            a, b = weighted[key][0], val[0]
            self.assertTrue((np.isnan(a) and np.isnan(b)) or abs(a - b) < 1e-12, key)


class TestBootstrap(unittest.TestCase):
    def test_counts_stratified(self):
        strata = np.array([0, 0, 0, 1, 1, 2])
        c = M.bootstrap_counts(strata, 500, seed=3)
        self.assertEqual(c.shape, (500, 6))
        self.assertTrue((c[:, :3].sum(1) == 3).all() and (c[:, 3:5].sum(1) == 2).all() and (c[:, 5] == 1).all())
        np.testing.assert_array_equal(c, M.bootstrap_counts(strata, 500, seed=3))

    def two_scene_truth(self):
        # one stratum, two scenes of 10 windows: scene 1 all called right, scene 2 all wrong
        n = 20
        scene = [1] * 10 + [2] * 10
        k = list(range(1, 20, 2)) * 2
        count = [1] * n
        xy = np.zeros((n, 3, 2))
        xy[:, 0] = (1, 1)
        pm = np.zeros((n, 3), bool)
        pm[:, 0] = True
        occ = np.zeros((n, 9), bool)
        occ[:, 0] = True
        truth = M.make_truth(scene, k, count, xy, pm, np.zeros((n, 3), bool), occ, np.ones((n, 9), bool),
                             np.tile([6.0, 4.0, 3.0], (n, 1)))
        pres = np.array([0.9] * 10 + [0.1] * 10)
        pred = {"scene": scene, "k": k, "presence_prob": pres, "count_prob": None, "zone_prob": None,
                "slot_exist": None, "slot_xy": None, "slot_sigma": None}
        return truth, pred

    def test_scene_bootstrap_by_hand(self):
        """Resampled accuracy is 1, 0.5, 0 with probabilities 1/4, 1/2, 1/4: interval [0, 1]."""
        truth, pred = self.two_scene_truth()
        at = M.score(pred, truth)
        c = M.bootstrap_counts(truth["scene_stratum"], 4000, seed=0)
        acc = M._figures(at, truth, c[:, truth["scene_pos"]].astype(float))["presence|all|accuracy"]
        vals, freq = np.unique(acc, return_counts=True)
        self.assertEqual(vals.tolist(), [0.0, 0.5, 1.0])
        np.testing.assert_allclose(freq / 4000, [0.25, 0.5, 0.25], atol=0.03)
        res = M.evaluate({"m": pred}, truth, n_boot=2000, seed=0)
        a = res["methods"]["m"]["figures"]["presence"]["all"]["accuracy"]
        self.assertAlmostEqual(a["value"], 0.5)
        self.assertEqual(a["ci95"], [0.0, 1.0])
        self.assertEqual(a["n_boot_valid"], 2000)
        self.assertEqual(res["bootstrap"]["resamples"], 2000)

    def test_paired_difference_of_identical_methods_is_zero(self):
        truth, pred = self.two_scene_truth()
        res = M.evaluate({"a": pred, "b": dict(pred)}, truth, reference="a", n_boot=300)
        d = res["differences"]["a - b"]["presence"]["all"]["accuracy"]
        self.assertEqual(d["diff"], 0.0)
        self.assertEqual(d["ci95"], [0.0, 0.0])

    def test_seeds(self):
        truth, pred = self.two_scene_truth()
        p2 = dict(pred, presence_prob=np.array([0.9] * 20))     # all correct
        res = M.aggregate_seeds([pred, p2], truth, n_boot=500)
        a = res["figures"]["presence"]["all"]["accuracy"]
        self.assertAlmostEqual(a["mean"], 0.75)
        self.assertAlmostEqual(a["std"], np.std([0.5, 1.0], ddof=1))
        self.assertEqual(a["seeds"], [0.5, 1.0])
        # per resample the seed mean is (x + 1) / 2 with x in {0, 0.5, 1}: interval [0.5, 1]
        self.assertEqual(a["ci95"], [0.5, 1.0])
        self.assertEqual(res["seeds"], 2)


class TestContract(unittest.TestCase):
    def test_refusals(self):
        truth, pred = case()
        with self.assertRaises(ValueError):
            M.check_predictions(dict(pred, label="REAL"))
        with self.assertRaises(ValueError):
            M.check_predictions(dict(pred, presence_prob=pred["presence_prob"] + 1))
        with self.assertRaises(ValueError):
            M.check_predictions(dict(pred, count_prob=np.zeros((6, 3))))
        bad = dict(pred, slot_sigma=None)
        with self.assertRaises(ValueError):
            M.check_predictions(bad)
        bad = dict(pred, slot_xy=np.where(np.ones((6, 3, 2), bool), np.nan, 0.0))
        with self.assertRaises(ValueError):
            M.check_predictions(bad)
        short = {key: (v[:5] if isinstance(v, np.ndarray) else v) for key, v in pred.items()}
        short["scene"], short["k"] = short["scene"][:5], short["k"][:5]
        with self.assertRaises(ValueError):
            M.align(M.check_predictions(short), truth)
        with self.assertRaises(ValueError):
            M.evaluate({"m": pred}, dict(truth, label="REAL"), n_boot=0)

    def test_alignment_reorders(self):
        truth, pred = case()
        order = np.array([5, 3, 1, 0, 2, 4])
        shuffled = {key: (v[order] if isinstance(v, np.ndarray) and v.shape[:1] == (6,) else v)
                    for key, v in pred.items()}
        shuffled["scene"] = np.asarray(pred["scene"])[order]
        shuffled["k"] = np.asarray(pred["k"])[order]
        a = M.evaluate({"m": pred}, truth, n_boot=0)
        b = M.evaluate({"m": shuffled}, truth, n_boot=0)
        self.assertEqual(a["methods"]["m"]["figures"], b["methods"]["m"]["figures"])

    def test_nogeo_skips_xy_and_zone(self):
        truth, pred = case()
        t2 = M.make_truth(truth["scene"], truth["k"], truth["count"], truth["xy"], truth["person_mask"],
                          truth["walked"], truth["zone_occupied"], truth["zone_reachable"], truth["room"],
                          geometry_known=np.zeros(6, bool), subset="single01_nogeo")
        res = M.evaluate({"m": pred}, t2, n_boot=0)
        figs = res["methods"]["m"]["figures"]
        self.assertIn("presence", figs)
        self.assertIn("count", figs)
        for area in ("zone", "xy2m", "xy1m", "one", "zone1"):
            self.assertNotIn(area, figs)

    def test_flat_figures(self):
        truth, pred = case()
        res = M.evaluate({"m": pred}, truth, n_boot=0)
        flat = M.flat_figures(res["methods"]["m"])
        self.assertAlmostEqual(flat["presence|all|accuracy"]["value"], 4 / 6)


if __name__ == "__main__":
    unittest.main()
