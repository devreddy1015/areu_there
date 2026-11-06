"""The Phase 8 window cache (spec/model.md §1, §7 "Cache and batch"): SIM.

Synthetic SIM datasets are written with the dataset tests' shard builder
(`test_roomsim_dataset.make_arrays` / `write_dataset`) with random int8
CSI and RSSI, so `preprocess` has something to normalise. A cached window
must equal `ml.preprocess.preprocess` on the same window to float16
rounding (here: exactly the float16 cast of its float32 output), labels
and stored baselines must be the shard's at the end tick, and `walked` is
checked against a hand-built walking pattern.

If SAANS_DATASET_DIR names the real dataset, a ~10-scene real cache is
built and checked the same way (skipped otherwise).

Run from the repo root: .venv/bin/python -m unittest discover -s ml/tests
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml import preprocess as pp  # noqa: E402
from ml import roomsim_dataset as rd  # noqa: E402
from ml.tests import test_roomsim_dataset as trd  # noqa: E402  (synthetic SIM shards)
from ml.train import cache  # noqa: E402

DATASET = os.environ.get("SAANS_DATASET_DIR")
SPLIT_OF = {0: "train", 1: "val", 2: "test", 3: "train", 4: "test"}
#: Synthetic scenes: nodes, people, lost packets. Scene 4's link 0 keeps only columns 5 and 30 of its
#: first 80 (4 s at 4 nodes), so its 2 s windows ending at k = 3, 5, 7 have fewer than 3 non-empty bins.
SPECS = ({"n_nodes": 4, "n_people": 0}, {"n_nodes": 5, "n_people": 1}, {"n_nodes": 4, "n_people": 2},
         {"n_nodes": 6, "n_people": 3},
         {"n_nodes": 4, "n_people": 1, "lost": tuple((0, m) for m in range(80) if m not in (5, 30))})
SECONDS = 8
#: Person 0's walking pattern (every scene with people): walking at tick 2 and tick 9 only.
WALK_TICKS = (2, 9)


def synthetic_arrays(i, spec, seconds=SECONDS):
    """One synthetic SIM shard's members with random CSI (int8, active
    subcarriers, received slots) and RSSI, and person 0 walking only at
    WALK_TICKS."""
    a = trd.make_arrays(index=i, seconds=seconds, **spec)
    rng = np.random.default_rng(100 + i)
    iq = rng.integers(-40, 41, size=a["csi_iq"].shape).astype(np.int8)
    active = np.array(trd.ACTIVE, bool)[None, :, None, None]
    mask = a["csi_mask"].astype(bool)[:, None, :, None]
    a["csi_iq"] = np.where(active & mask, iq, 0).astype(np.int8)
    a["rssi"] = np.where(a["csi_mask"] != 0, rng.integers(-80, -30, size=a["rssi"].shape), 0).astype(np.int8)
    walking = np.zeros_like(a["walking"])
    if spec.get("n_people", 0) > 0:
        for k in WALK_TICKS:
            walking[k, 0] = 1
    a["walking"] = walking
    return a


def write_synthetic(root, specs=SPECS, split_of=None, seconds=SECONDS):
    """A synthetic SIM dataset under `root` (scene i in `split_of[i]`)."""
    arrays = [synthetic_arrays(i, s, seconds) for i, s in enumerate(specs)]
    trd.write_dataset(root, arrays, split_of=split_of or SPLIT_OF, seconds=seconds)
    return arrays


def build(dataset, length, out, **kw):
    kw.setdefault("workers", 1)
    kw.setdefault("progress", False)
    return cache.build_cache(dataset, length, out, **kw)


class TestTicks(unittest.TestCase):
    def test_every_second(self):
        # k odd and t_k = 0.5 (k + 1) >= length: k >= 2 length - 1.
        self.assertEqual(cache.cache_ticks(16, 1.0), [1, 3, 5, 7, 9, 11, 13, 15])
        self.assertEqual(cache.cache_ticks(16, 2.0), [3, 5, 7, 9, 11, 13, 15])
        self.assertEqual(cache.cache_ticks(16, 4.0), [7, 9, 11, 13, 15])
        self.assertEqual(cache.cache_ticks(16, 1.5), [3, 5, 7, 9, 11, 13, 15])
        self.assertEqual(cache.cache_ticks(16, 0.5), [1, 3, 5, 7, 9, 11, 13, 15])
        # the brief's 2 s windows k = 4w + 3 are all included
        self.assertTrue({4 * w + 3 for w in range(4)} <= set(cache.cache_ticks(16, 2.0)))
        # decision 5's evaluation windows (odd k, t_k >= 4) are cached for every length <= 4
        ev = [k for k in range(240) if k % 2 == 1 and 0.5 * (k + 1) >= 4.0]
        for length in (1.0, 2.0, 4.0):
            self.assertTrue(set(ev) <= set(cache.cache_ticks(240, length)))
        self.assertEqual(len(cache.cache_ticks(240, 2.0)), 119)

    def test_every(self):
        self.assertEqual(cache.cache_ticks(10, 2.0, "every"), [3, 4, 5, 6, 7, 8, 9])
        with self.assertRaises(ValueError):
            cache.cache_ticks(10, 2.0, "odd")

    def test_default_dir(self):
        self.assertEqual(cache.default_cache_dir("/x/roomsim-v1/", 2.0), "/dev/shm/saans-phase8-cache/roomsim-v1/2")
        self.assertEqual(cache.default_cache_dir("roomsim-v1-hwoff", 1.5, "/r"), "/r/roomsim-v1-hwoff/1.5")


class SyntheticCase(unittest.TestCase):
    """A synthetic SIM dataset and its ℓ = 2 cache (workers 1), shared by the class."""

    length = 2.0

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.root = cls._tmp.name
        cls.dataset = os.path.join(cls.root, "synth-sim")
        cls.arrays = write_synthetic(cls.dataset)
        cls.meta = build(cls.dataset, cls.length, os.path.join(cls.root, "cache"))
        cls.c = cache.open_cache(cls.meta["out_dir"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()


class TestBuild(SyntheticCase):
    def test_meta(self):
        m = self.meta
        self.assertEqual(m["label"], "SIM")
        self.assertEqual(m["T"], 10)
        self.assertEqual(m["preprocess"], {"phase": False, "bin_hz": 5, "ref_bins": "shared44", "min_packets": 3})
        self.assertEqual(m["ref_k"], list(range(-26, -8)) + list(range(1, 27)))
        n_links = [s["n_nodes"] * (s["n_nodes"] - 1) for s in SPECS]
        self.assertEqual(m["n_windows"], 7 * len(SPECS))
        self.assertEqual(m["n_rows"], 7 * sum(n_links))
        self.assertEqual(m["split_scenes"], {"train": 2, "val": 1, "test": 2})
        self.assertEqual(m["impairments"], "on")
        with open(os.path.join(m["out_dir"], "meta.json"), encoding="utf-8") as f:
            self.assertEqual(json.load(f)["label"], "SIM")
        self.assertFalse(os.path.exists(m["out_dir"] + ".partial"))

    def test_files(self):
        c = self.c
        n = self.meta["n_rows"]
        want = {"amp": (np.float16, (n, 44, 10)), "mu": (np.float16, (n, 44)), "sigma": (np.float16, (n, 44)),
                "rssi": (np.float16, (n, 10)), "gain_db": (np.float16, (n, 10)), "sample_mask": (np.bool_, (n, 10)),
                "rssi_mask": (np.bool_, (n, 10)), "bin_count": (np.uint8, (n, 10)), "link_mask": (np.bool_, (n,))}
        self.assertEqual(set(c.rows), set(want))
        for name, (dt, shape) in want.items():
            self.assertEqual(c.rows[name].dtype, np.dtype(dt), name)
            self.assertEqual(c.rows[name].shape, shape, name)
            if c.rows[name].dtype.kind == "f":
                self.assertTrue(np.isfinite(np.asarray(c.rows[name], np.float32)).all(), name)
        # amp, rssi and gain are 0 wherever the sample mask is off
        sm = np.asarray(c.rows["sample_mask"])
        self.assertTrue((np.asarray(c.rows["amp"]).transpose(0, 2, 1)[~sm] == 0).all())
        self.assertTrue((np.asarray(c.rows["rssi"])[~sm] == 0).all())

    def test_equals_preprocess(self):
        """Every cached window is preprocess()'s float32 output cast to float16 (spec/model.md §7)."""
        c = self.c
        for w in range(c.n_windows):
            one = c.read_window(w)
            win = pp.from_roomsim_shard(self.dataset, int(one["scene"]), int(one["k"]), self.length)
            out = pp.preprocess(win, cache.PREPROCESS)
            ref = out["ref"]
            self.assertEqual(int(ref.sum()), 44)
            for key in ("amp", "mu", "sigma"):
                np.testing.assert_array_equal(one[key], out[key][:, ref].astype(np.float16), err_msg=f"{key} w={w}")
            for key in ("rssi", "gain_db"):
                np.testing.assert_array_equal(one[key], out[key].astype(np.float16), err_msg=f"{key} w={w}")
            for key in ("sample_mask", "rssi_mask", "link_mask"):
                np.testing.assert_array_equal(one[key], out[key], err_msg=f"{key} w={w}")
            np.testing.assert_array_equal(one["bin_count"], out["bin_count"])
            np.testing.assert_array_equal(one["links"], out["links"])
            np.testing.assert_array_equal(one["link_pos"], out["link_pos"])
            self.assertEqual(int(one["links_dropped"]), out["links_dropped"])

    def test_dropped_link_masked(self):
        """Scene 4's link 0 loses most packets: some 2 s windows drop it (link_mask false, rows 0)."""
        c = self.c
        ws = c.window_indices(scenes=[4])
        lm = np.array([c.read_window(w)["link_mask"][0] for w in ws])
        self.assertEqual(c.windows["k"][ws][~lm].tolist(), [3, 5, 7])
        for w in ws[~lm]:
            one = c.read_window(w)
            self.assertFalse(one["sample_mask"][0].any())
            self.assertTrue((one["amp"][0] == 0).all() and (one["mu"][0] == 0).all() and (one["sigma"][0] == 0).all())
            self.assertGreaterEqual(int(one["links_dropped"]), 1)

    def test_labels(self):
        c = self.c
        for w in range(c.n_windows):
            one = c.read_window(w)
            s, k = int(one["scene"]), int(one["k"])
            a = self.arrays[s]
            self.assertEqual(int(one["count"]), int(a["count"][k]))
            self.assertEqual(int(one["presence"]), int(a["presence"][k]))
            np.testing.assert_array_equal(one["zone_occupied"], a["zone_occupied"][k])
            pm = a["person_mask"][k].astype(bool)
            np.testing.assert_array_equal(one["person_mask"], pm)
            np.testing.assert_array_equal(one["xy"][pm], a["xy"][k][pm])
            self.assertTrue((one["xy"][~pm] == 0).all())                      # 0 where absent, never NaN
            np.testing.assert_array_equal(one["posture"], a["posture"][k])
            np.testing.assert_array_equal(one["zone"], a["zone"][k])
            self.assertAlmostEqual(float(one["t"]), 0.5 * (k + 1))
            np.testing.assert_array_equal(one["zone_reachable"], a["zone_reachable"].astype(bool))
            np.testing.assert_array_equal(one["room"], a["room_size"])
            for name in ("wlm", "rti"):
                self.assertEqual(int(one[f"{name}_reason"]), int(a[f"{name}_reason"][k]))
                np.testing.assert_array_equal(one[f"{name}_xy"], a[f"{name}_xy"][k])     # NaN unless published
                np.testing.assert_array_equal(one[f"{name}_radius"], a[f"{name}_radius"][k])

    def test_walked(self):
        """walked = walking at some label tick in (t_k - 2, t_k], i.e. ticks k-3..k (person 0 walks at 2 and 9)."""
        c = self.c
        for w in range(c.n_windows):
            one = c.read_window(w)
            k = int(one["k"])
            n_people = int(one["count"])
            want = n_people > 0 and any(k - 3 <= t <= k for t in WALK_TICKS)
            self.assertEqual(bool(one["walked"][0]), want, f"k={k}")
            self.assertFalse(one["walked"][1:].any())
            self.assertEqual(bool(one["walking"][0]), n_people > 0 and k in WALK_TICKS)

    def test_tables(self):
        c = self.c
        sc = c.scenes
        self.assertEqual(sc["scene"].tolist(), list(range(len(SPECS))))
        self.assertEqual(sc["split"].tolist(), [cache.SPLIT_CODES[SPLIT_OF[i]] for i in range(len(SPECS))])
        for s, a in enumerate(self.arrays):
            links, link_pos = c.scene_links(s)
            np.testing.assert_array_equal(links, a["links"])
            pos = {int(i): p for i, p in zip(a["node_ids"], a["node_pos"])}
            for (tx, rx), lp in zip(links, link_pos):
                np.testing.assert_array_equal(lp, np.stack([pos[int(tx)], pos[int(rx)]]))
            self.assertEqual(int(sc["n_vertices"][s]), 4)
            np.testing.assert_array_equal(sc["footprint"][s][:4], a["footprint"])
            self.assertTrue(np.isnan(sc["footprint"][s][4:]).all())
            self.assertEqual(int(sc["n_people"][s]), SPECS[s]["n_people"])
        win = c.windows
        self.assertTrue((np.diff(win["row_offset"]) == win["n_links"][:-1]).all())
        self.assertEqual(int(win["row_offset"][-1] + win["n_links"][-1]), self.meta["n_rows"])
        self.assertTrue((np.diff(sc["window_offset"]) == 7).all())

    def test_window_indices(self):
        c = self.c
        ev = c.window_indices("test", min_t=4.0)
        self.assertTrue((c.windows["split"][ev] == 2).all())
        self.assertEqual(sorted(set(c.windows["k"][ev].tolist())), [7, 9, 11, 13, 15])
        self.assertEqual(ev.size, 2 * 5)
        self.assertEqual(c.window_indices().size, c.n_windows)

    def test_workers_identical(self):
        """A parallel build (fork, 3 workers) writes the same bytes as a serial one."""
        out = os.path.join(self.root, "cache-w3")
        meta = build(self.dataset, self.length, out, workers=3)
        for name in sorted(os.listdir(out)):
            if name == "meta.json":
                continue
            with open(os.path.join(out, name), "rb") as f1, open(os.path.join(self.meta["out_dir"], name), "rb") as f2:
                self.assertEqual(f1.read(), f2.read(), name)
        self.assertEqual(meta["n_rows"], self.meta["n_rows"])

    def test_refusals(self):
        with self.assertRaises(FileExistsError):
            build(self.dataset, self.length, self.meta["out_dir"])
        bad = os.path.join(self.root, "not-sim")
        shutil.copytree(self.meta["out_dir"], bad)
        with open(os.path.join(bad, "meta.json"), encoding="utf-8") as f:
            m = json.load(f)
        m["label"] = "REAL"
        with open(os.path.join(bad, "meta.json"), "w", encoding="utf-8") as f:
            json.dump(m, f)
        with self.assertRaises(ValueError):
            cache.open_cache(bad)
        os.remove(os.path.join(bad, "meta.json"))
        with self.assertRaises(FileNotFoundError):
            cache.open_cache(bad)

    def test_pickle_reopens_rows(self):
        """A pickled cache (a spawned loader worker) maps its rows itself."""
        import pickle
        c2 = pickle.loads(pickle.dumps(self.c))
        self.assertIsNone(c2._rows)
        np.testing.assert_array_equal(c2.rows["amp"][:5], self.c.rows["amp"][:5])


class TestOtherBuilds(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = self._tmp.name
        self.dataset = os.path.join(self.root, "synth-sim")
        write_synthetic(self.dataset, SPECS[:3])

    def tearDown(self):
        self._tmp.cleanup()

    def test_one_second_and_phase(self):
        meta = build(self.dataset, 1.0, os.path.join(self.root, "c1"), phase=True)
        c = cache.open_cache(meta["out_dir"])
        self.assertEqual(c.T, 5)
        self.assertEqual(sorted(set(c.windows["k"].tolist())), [1, 3, 5, 7, 9, 11, 13, 15])
        self.assertIn("phase", c.rows)
        self.assertEqual(c.rows["phase"].shape, (meta["n_rows"], 44, 5, 2))
        cfg = pp.PreprocessConfig(phase=True, bin_hz=5, ref_bins="shared44", min_packets=3)
        for w in (0, 5, c.n_windows - 1):
            one = c.read_window(w)
            out = pp.preprocess(pp.from_roomsim_shard(self.dataset, int(one["scene"]), int(one["k"]), 1.0), cfg)
            ref = out["ref"]
            np.testing.assert_array_equal(one["phase"], out["phase"][:, ref].astype(np.float16))
            np.testing.assert_array_equal(one["phase_mask"], out["phase_mask"][:, ref])
            # phase on leaves the amplitude path unchanged
            plain = pp.preprocess(pp.from_roomsim_shard(self.dataset, int(one["scene"]), int(one["k"]), 1.0),
                                  cache.PREPROCESS)
            np.testing.assert_array_equal(one["amp"], plain["amp"][:, ref].astype(np.float16))

    def test_scene_subset(self):
        meta = build(self.dataset, 2.0, os.path.join(self.root, "c2"), scenes=[2, 0])
        c = cache.open_cache(meta["out_dir"])
        self.assertEqual(c.scenes["scene"].tolist(), [2, 0])
        self.assertEqual(c.n_windows, 14)
        one = c.read_window(0)
        self.assertEqual(int(one["scene"]), 2)
        win = pp.from_roomsim_shard(self.dataset, 2, int(one["k"]), 2.0)
        out = pp.preprocess(win, cache.PREPROCESS)
        np.testing.assert_array_equal(one["amp"], out["amp"][:, out["ref"]].astype(np.float16))


@unittest.skipUnless(DATASET, "SAANS_DATASET_DIR names no real dataset")
class TestRealCache(unittest.TestCase):
    """~10 real SIM scenes at ℓ = 2 (parallel build): cache == preprocess on sampled windows."""

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        manifest = rd.load_manifest(DATASET)
        sp = rd.splits(manifest)
        cls.scenes = sp["train"][:4] + sp["val"][:3] + sp["test"][:3]
        cls.meta = cache.build_cache(DATASET, 2.0, os.path.join(cls._tmp.name, "real"), workers=8,
                                     scenes=cls.scenes, progress=False)
        cls.c = cache.open_cache(cls.meta["out_dir"])

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_counts(self):
        self.assertEqual(self.meta["n_windows"], 119 * len(self.scenes))
        self.assertEqual(self.c.window_indices("test", min_t=4.0).size, 117 * 3)
        self.assertEqual(self.meta["label"], "SIM")

    def test_equals_preprocess(self):
        rng = np.random.default_rng(0)
        for w in rng.choice(self.c.n_windows, 12, replace=False):
            one = self.c.read_window(int(w))
            out = pp.preprocess(pp.from_roomsim_shard(DATASET, int(one["scene"]), int(one["k"]), 2.0),
                                cache.PREPROCESS)
            ref = out["ref"]
            for key in ("amp", "mu", "sigma"):
                np.testing.assert_array_equal(one[key], out[key][:, ref].astype(np.float16))
            np.testing.assert_array_equal(one["rssi"], out["rssi"].astype(np.float16))
            np.testing.assert_array_equal(one["sample_mask"], out["sample_mask"])
            np.testing.assert_array_equal(one["link_mask"], out["link_mask"])
            scene = rd.load_scene(DATASET, int(one["scene"]))
            lab = rd.labels_at(scene, int(one["k"]))
            self.assertEqual(int(one["count"]), lab["count"])
            base = rd.baseline_at(scene, int(one["k"]))
            self.assertEqual(int(one["wlm_reason"]) == 0, base["wlm"]["published"])


if __name__ == "__main__":
    unittest.main()
