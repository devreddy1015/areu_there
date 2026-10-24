"""Phase 8 data (spec/model.md §1, §2 "Training", §5.6, §7): the batch
contract, the link-subset and unknown-geometry rules, the evaluation
subsets and windows, and the training sampler. SIM.

The rules are checked by their frequencies over many seeded draws (bounds
are about 5 binomial standard deviations) and exactly where they are
exact; the batch against spec/model.md §1's table key by key.

Run from the repo root: .venv/bin/python -m unittest discover -s ml/tests
"""

import os
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.tests import test_train_cache as ttc  # noqa: E402  (synthetic SIM datasets)
from ml.train import cache, data  # noqa: E402

#: spec/model.md §1's batch: key -> (dtype, shape with B, L, T symbols).
SPEC_BATCH = {
    "amp": (torch.float16, ("B", "L", 44, "T")),
    "mu": (torch.float16, ("B", "L", 44)),
    "sigma": (torch.float16, ("B", "L", 44)),
    "sample_mask": (torch.bool, ("B", "L", "T")),
    "link_mask": (torch.bool, ("B", "L")),
    "rssi": (torch.float16, ("B", "L", "T")),
    "rssi_mask": (torch.bool, ("B", "L", "T")),
    "link_pos": (torch.float32, ("B", "L", 2, 3)),
    "room": (torch.float32, ("B", 3)),
    "geometry_known": (torch.bool, ("B",)),
    "presence": (torch.float32, ("B",)),
    "count": (torch.int64, ("B",)),
    "zone_occupied": (torch.float32, ("B", 9)),
    "zone_reachable": (torch.bool, ("B", 9)),
    "xy": (torch.float32, ("B", 3, 2)),
    "person_mask": (torch.bool, ("B", 3)),
    "walked": (torch.bool, ("B", 3)),
    "wlm_published": (torch.bool, ("B",)),
    "wlm_xy": (torch.float32, ("B", 2)),
    "wlm_radius": (torch.float32, ("B",)),
    "rti_published": (torch.bool, ("B",)),
    "rti_xy": (torch.float32, ("B", 2)),
    "rti_radius": (torch.float32, ("B",)),
}


def mesh_links(n):
    return np.array([(a, b) for a in range(n) for b in range(n) if a != b])


class TestLinkSubsetRule(unittest.TestCase):
    """spec/model.md §2: p 0.2 one transmitter's links, p 0.1 one link, else drop each with p 0.1 (>= 1 kept)."""

    def classify(self, keep, used, tx):
        k = np.flatnonzero(keep)
        if k.size == 1:
            return "one_link"
        if np.unique(tx[k]).size == 1 and np.array_equal(keep, used & (tx == tx[k[0]])):
            return "one_tx"
        return "drop"

    def test_frequencies(self):
        links = mesh_links(12)
        tx = links[:, 0]
        used = np.ones(len(links), bool)
        rng = np.random.default_rng(0)
        n = 20000
        kinds = {"one_tx": 0, "one_link": 0, "drop": 0}
        kept_frac = []
        for _ in range(n):
            keep = data.link_subset(tx, used, rng)
            self.assertTrue(keep.any())
            kind = self.classify(keep, used, tx)
            kinds[kind] += 1
            if kind == "drop":
                kept_frac.append(keep.mean())
        for kind, p in (("one_tx", 0.2), ("one_link", 0.1), ("drop", 0.7)):
            sd = np.sqrt(p * (1 - p) / n)
            self.assertLess(abs(kinds[kind] / n - p), 5 * sd, kinds)
        self.assertLess(abs(np.mean(kept_frac) - 0.9), 0.005)

    def test_only_used_links_and_at_least_one(self):
        links = mesh_links(5)
        tx = links[:, 0]
        rng = np.random.default_rng(1)
        used = rng.random(len(links)) < 0.5
        used[:4] = False                                  # transmitter 0 has no used link
        seen_tx = set()
        for _ in range(3000):
            keep = data.link_subset(tx, used, rng)
            self.assertFalse((keep & ~used).any())
            self.assertTrue(keep.any())
            k = np.flatnonzero(keep)
            if np.unique(tx[k]).size == 1 and np.array_equal(keep, used & (tx == tx[k[0]])) and k.size > 1:
                seen_tx.add(int(tx[k[0]]))
        self.assertNotIn(0, seen_tx)                      # a transmitter without used links is never drawn
        # all-drop branch: one used link survives
        one = np.zeros(len(links), bool)
        one[7] = True
        for _ in range(200):
            self.assertTrue(np.array_equal(data.link_subset(tx, one, rng), one))
        none = np.zeros(len(links), bool)
        self.assertFalse(data.link_subset(tx, none, rng).any())

    def test_deterministic(self):
        links = mesh_links(8)
        a = [data.link_subset(links[:, 0], np.ones(56, bool), np.random.default_rng(5)) for _ in range(3)]
        self.assertTrue(all(np.array_equal(a[0], x) for x in a))


class TestEvalSubsets(unittest.TestCase):
    def test_subsets(self):
        links = mesh_links(4)
        used = np.ones(12, bool)
        used[1] = False                                   # (0, 2) unused
        keep, known = data.eval_subset(links, used, "all")
        self.assertTrue(np.array_equal(keep, used) and known)
        keep, known = data.eval_subset(links, used, "tx0")
        self.assertEqual(links[keep].tolist(), [[0, 1], [0, 3]])
        self.assertTrue(known)
        keep, known = data.eval_subset(links, used, "single01")
        self.assertEqual(links[keep].tolist(), [[0, 1]])
        self.assertTrue(known)
        keep, known = data.eval_subset(links, used, "single01_nogeo")
        self.assertEqual(links[keep].tolist(), [[0, 1]])
        self.assertFalse(known)
        used[0] = False
        keep, _ = data.eval_subset(links, used, "single01")
        self.assertFalse(keep.any())
        with self.assertRaises(ValueError):
            data.eval_subset(links, used, "tx1")


class DataCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.dataset = os.path.join(cls._tmp.name, "synth-sim")
        ttc.write_synthetic(cls.dataset)
        meta = cache.build_cache(cls.dataset, 2.0, os.path.join(cls._tmp.name, "cache"), workers=1, progress=False)
        cls.c = cache.open_cache(meta["out_dir"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestDataset(DataCase):
    def test_windows(self):
        ev = data.CacheDataset(self.c, "test", mode="eval")
        k = self.c.windows["k"][ev.windows]
        self.assertTrue((k % 2 == 1).all() and (0.5 * (k + 1) >= 4.0).all())
        self.assertEqual(len(ev), 2 * 5)                  # 2 test scenes x k in 7..15
        np.testing.assert_array_equal(ev.windows, data.eval_window_indices(self.c, "test"))
        tr = data.CacheDataset(self.c, "train", mode="train")
        self.assertEqual(len(tr), 2 * 7)                  # every cached window (t_k >= ℓ)
        with self.assertRaises(ValueError):
            data.CacheDataset(self.c, "train", mode="train", subset="tx0")

    def test_sample_rows_and_labels(self):
        ev = data.CacheDataset(self.c, "val", mode="eval")
        for i in range(len(ev)):
            s = ev[i]
            w = int(ev.windows[i])
            one = self.c.read_window(w)
            used = one["link_mask"]
            np.testing.assert_array_equal(s["amp"], one["amp"][used])
            np.testing.assert_array_equal(s["mu"], one["mu"][used])
            np.testing.assert_array_equal(s["rssi"], one["rssi"][used])
            np.testing.assert_array_equal(s["link_pos"], one["link_pos"][used])
            np.testing.assert_array_equal(s["meta"]["links"], one["links"][used])
            self.assertTrue(s["geometry_known"])
            self.assertEqual(s["meta"]["scene"], int(one["scene"]))
            self.assertEqual(s["meta"]["k"], int(one["k"]))
            self.assertEqual(s["meta"]["split"], "val")
            self.assertEqual(s["count"], int(one["count"]))
            np.testing.assert_array_equal(s["walked"], one["walked"])
            pub = int(one["wlm_reason"]) == 0
            self.assertEqual(s["wlm_published"], pub)
            if pub:
                np.testing.assert_array_equal(s["wlm_xy"], one["wlm_xy"])
            else:
                self.assertTrue((s["wlm_xy"] == 0).all() and s["wlm_radius"] == 0)

    def test_eval_subsets(self):
        for subset, want in (("tx0", lambda l: l[:, 0] == 0), ("single01", lambda l: (l[:, 0] == 0) & (l[:, 1] == 1))):
            ds = data.CacheDataset(self.c, "test", mode="eval", subset=subset)
            for i in range(len(ds)):
                s = ds[i]
                self.assertTrue(want(s["meta"]["links"]).all())
                one = self.c.read_window(int(ds.windows[i]))
                self.assertEqual(len(s["meta"]["links"]), int((want(one["links"]) & one["link_mask"]).sum()))
                self.assertTrue(s["geometry_known"])
        ds = data.CacheDataset(self.c, "test", mode="eval", subset="single01_nogeo")
        self.assertTrue(all(not ds[i]["geometry_known"] for i in range(len(ds))))

    def test_unknown_geometry_rate(self):
        tr = data.CacheDataset(self.c, "train", mode="train")
        n = 6000
        known = np.array([tr[(i % len(tr), 1000 + i)]["geometry_known"] for i in range(n)])
        p = 0.1
        self.assertLess(abs((~known).mean() - p), 5 * np.sqrt(p * (1 - p) / n))
        # the same draw seed gives the same sample
        a, b = tr[(3, 77)], tr[(3, 77)]
        np.testing.assert_array_equal(a["meta"]["links"], b["meta"]["links"])
        self.assertEqual(a["geometry_known"], b["geometry_known"])
        # an integer key is not augmented
        s = tr[3]
        self.assertTrue(s["geometry_known"])
        self.assertEqual(len(s["meta"]["links"]), int(self.c.read_window(int(tr.windows[3]))["link_mask"].sum()))


class TestCollate(DataCase):
    def check_batch(self, b, n, include=()):
        extra = {"gain_db", "bin_count", "links"} & set(include)
        if "phase" in include:
            extra |= {"phase", "phase_mask"}
        self.assertEqual(set(b), set(SPEC_BATCH) | {"meta"} | extra)
        self.assertEqual(tuple(data.BATCH_KEYS), tuple(SPEC_BATCH) + ("meta",))
        big = b["amp"].shape[1]
        sizes = {"B": n, "L": big, "T": 10}
        for key, (dt, shape) in SPEC_BATCH.items():
            self.assertEqual(b[key].dtype, dt, key)
            self.assertEqual(tuple(b[key].shape), tuple(sizes.get(d, d) for d in shape), key)
        self.assertIsInstance(b["meta"], list)
        self.assertEqual(len(b["meta"]), n)
        kept = np.array([len(m["links"]) for m in b["meta"]])
        self.assertEqual(big, max(1, kept.max()))
        lm = b["link_mask"].numpy()
        for i in range(n):
            self.assertTrue(lm[i, :kept[i]].all() and not lm[i, kept[i]:].any())
            # padding is zero in every per-link array
            for key in ("amp", "mu", "sigma", "rssi", "link_pos"):
                self.assertTrue((b[key][i, kept[i]:] == 0).all(), key)
            for key in ("sample_mask", "rssi_mask"):
                self.assertFalse(b[key][i, kept[i]:].any(), key)
        for key in ("amp", "mu", "sigma", "rssi", "link_pos", "room", "xy", "wlm_xy", "rti_xy"):
            self.assertTrue(torch.isfinite(b[key].float()).all(), key)
        # xy is 0 where absent
        self.assertTrue((b["xy"][~b["person_mask"]] == 0).all())

    def test_eval_batch(self):
        ev = data.CacheDataset(self.c, "test", mode="eval")
        b = data.collate([ev[i] for i in range(len(ev))])
        self.check_batch(b, len(ev))
        self.assertTrue(b["geometry_known"].all())

    def test_train_batch_and_extras(self):
        tr = data.CacheDataset(self.c, "train", mode="train", include=("gain_db", "bin_count", "links"))
        keys = data.TrainSampler(tr, seed=3).epoch_keys(0)
        b = data.collate([tr[k] for k in keys])
        self.check_batch(b, len(keys), include=("gain_db", "bin_count", "links"))
        self.assertEqual(b["gain_db"].dtype, torch.float16)
        lk = b["links"].numpy()
        for i, m in enumerate(b["meta"]):
            n = len(m["links"])
            np.testing.assert_array_equal(lk[i, :n], m["links"])
            self.assertTrue((lk[i, n:] == -1).all())

    def test_empty_subset_pads_to_one(self):
        ev = data.CacheDataset(self.c, "test", mode="eval", subset="single01")
        s = dict(ev[0])
        for key in ("amp", "mu", "sigma", "sample_mask", "rssi", "rssi_mask", "link_pos"):
            s[key] = s[key][:0]
        s["meta"] = dict(s["meta"], links=s["meta"]["links"][:0])
        b = data.collate([s, s])
        self.assertEqual(b["amp"].shape[1], 1)
        self.assertFalse(b["link_mask"].any())

    def test_phase_requires_phase_cache(self):
        with self.assertRaises(ValueError):
            data.CacheDataset(self.c, "test", include=("phase",))
        with self.assertRaises(ValueError):
            data.CacheDataset(self.c, "test", include=("nope",))


class TestSampler(DataCase):
    def test_epochs(self):
        tr = data.CacheDataset(self.c, "train", mode="train")
        sm = data.TrainSampler(tr, ticks_per_scene=3, seed=11)
        self.assertEqual(len(sm), 2 * 3)
        e0 = list(sm)
        e1 = list(sm)
        self.assertNotEqual(e0, e1)
        self.assertEqual(e0, data.TrainSampler(tr, ticks_per_scene=3, seed=11).epoch_keys(0))
        sm.set_epoch(0)
        self.assertEqual(list(sm), e0)
        self.assertEqual(list(sm), e1)                    # the next iteration continues from epoch 1
        for keys in (e0, e1):
            pos = [p for p, _ in keys]
            scenes = self.c.windows["scene"][tr.windows[pos]]
            for s in np.unique(scenes):
                self.assertEqual(int((scenes == s).sum()), 3)
            self.assertEqual(len(set(pos)), len(pos))     # without replacement
        self.assertNotEqual(e0, data.TrainSampler(tr, ticks_per_scene=3, seed=12).epoch_keys(0))
        # a scene with fewer windows than ticks_per_scene gives all of them
        self.assertEqual(len(data.TrainSampler(tr, ticks_per_scene=64).epoch_keys(0)), len(tr))
        self.assertEqual(data.TICKS_PER_SCENE, 64)

    def test_loader_workers_do_not_change_batches(self):
        tr = data.CacheDataset(self.c, "train", mode="train")
        out = []
        for workers in (0, 2):
            sm = data.TrainSampler(tr, ticks_per_scene=5, seed=4)
            ld = data.make_loader(tr, 4, sampler=sm, num_workers=workers, pin_memory=False)
            out.append([b for b in ld])
        self.assertEqual(len(out[0]), len(out[1]))
        for b0, b1 in zip(*out):
            for key in SPEC_BATCH:
                self.assertTrue(torch.equal(b0[key], b1[key]), key)
            self.assertEqual([m["window"] for m in b0["meta"]], [m["window"] for m in b1["meta"]])

    def test_to_device(self):
        ev = data.CacheDataset(self.c, "test")
        b = data.to_device(data.collate([ev[0], ev[1]]), "cpu")
        self.assertIsInstance(b["meta"], list)
        self.assertEqual(b["amp"].device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
