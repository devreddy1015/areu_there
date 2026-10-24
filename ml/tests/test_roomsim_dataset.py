"""Dataset v1's Python loader and sample plot (spec/dataset.md §10, §11 Format).

Synthetic shards are built here with numpy.savez (uncompressed, STORED,
the §7.2 member names and dtypes) on hand-chosen TDMA grids: the CSI's
real part is the column `m + 1` and its imaginary part the link index, so
a window's amplitude says which column landed in which slot. Every window
case below has its answer worked out by hand in its comment, and a
brute-force check redoes §4.3 with exact fractions for N = 4..12.

If SAANS_DATASET_DIR names a real dataset, TestRealDataset also loads
every shard and checks it byte-level and against §7.2's shapes, dtypes and
meta (skipped otherwise). Everything here is SIM.

Run from the repo root: .venv/bin/python -m unittest discover -s ml/tests
"""

import contextlib
import hashlib
import io
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
import zipfile
from fractions import Fraction

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml import plot_sample  # noqa: E402  (selects the Agg backend)
from ml import roomsim_dataset as rd  # noqa: E402

#: §7.2's members in order with their npy `descr`, transcribed here
#: independently of `rd.MEMBERS` (TestFormat compares the two).
SPEC_72 = (
    ("csi_iq", "|i1"), ("csi_mask", "|u1"), ("rssi", "|i1"), ("tx_slot", "|u1"),
    ("subcarrier_k", "|i1"), ("subcarrier_active", "|u1"),
    ("count", "|u1"), ("presence", "|u1"), ("xy", "<f4"), ("person_mask", "|u1"), ("walking", "|u1"),
    ("posture", "|u1"), ("zone", "|i1"), ("zone_occupied", "|u1"), ("zone_single", "|i1"), ("t_label", "<f8"),
    ("zone_floor_frac", "<f4"), ("zone_reachable", "|u1"),
    ("wlm_xy", "<f4"), ("rti_xy", "<f4"), ("wlm_radius", "<f4"), ("rti_radius", "<f4"),
    ("wlm_reason", "|u1"), ("rti_reason", "|u1"),
    ("calib_baseline", "<f8"), ("calib_null_sigma", "<f8"), ("calib_mask", "|u1"),
    ("calib_iq", "|i1"), ("calib_mask_m", "|u1"), ("calib_rssi", "|i1"),
    ("links", "|u1"), ("node_ids", "|u1"), ("node_pos", "<f4"), ("room_size", "<f4"), ("footprint", "<f4"),
    ("window_end_tick", "<i4"), ("meta_json", "|u1"),
)

W, D, H = 6.0, 4.5, 2.7
BUFFER_K = list(range(32)) + list(range(-32, -8))      # LLTF bins 0..31, -32..-9
ACTIVE = [1 if 1 <= abs(k) <= 26 else 0 for k in BUFFER_K]  # 44 ones, 12 DC and guard nulls
#: The L's cut corner (top right, extents 0.4 w and 0.4 d): cell 8 is all
#: cut, cells 5 and 7 lose a 0.2 strip, cell 4 a 0.2 x 0.2 corner.
L_FOOTPRINT = [(0, 0), (W, 0), (W, 0.6 * D), (0.6 * W, 0.6 * D), (0.6 * W, D), (0, D)]
L_FLOOR_FRAC = [1, 1, 1, 1, 0.96, 0.8, 1, 0.8, 0]


def person_at(i, k):
    """Hand-built SIM people: P0 walks along y = 1, P1 sits, P2 lies.
    Returns ((x, y), walking, posture)."""
    if i == 0:
        return (0.5 + 0.25 * k, 1.0), 1, 0
    if i == 1:
        return (2.5, 3.0), 0, 1
    return (1.0, 4.0), 0, 2


def zone(x, y):
    return 3 * min(int(3 * y / D), 2) + min(int(3 * x / W), 2)


def make_arrays(index=0, n_nodes=4, seconds=4, n_people=1, lost=(), l_shape=False, impair=True, mc=6):
    """One synthetic SIM shard's members, in §7.2 order. Link l = (tx, rx)
    in tx-major order, `tx_slot[l] = tx`; column m of link l holds
    `(imag, real) = (l % 100, m % 120 + 1)` on active subcarriers when
    received; `(l, m)` in `lost` is lost. WLM: warming up at k = 0,
    published at odd k but 7 (implausible_radius), else no_disturbance;
    RTI: published at k = 3 only."""
    n, s, p = n_nodes, seconds, n_people
    k_ticks, cols = 2 * s, -(-80 * s // n_nodes)
    links = [(tx, rx) for tx in range(n) for rx in range(n) if rx != tx]
    n_links = len(links)
    tx_slot = np.array([tx for tx, _ in links])
    mask = np.arange(cols)[None, :] * n + tx_slot[:, None] < 80 * s
    for l, m in lost:
        mask[l, m] = False
    iq = np.zeros((n_links, 56, cols, 2), np.int8)
    iq[..., 0] = (np.arange(n_links) % 100)[:, None, None]
    iq[..., 1] = (np.arange(cols) % 120 + 1)[None, None, :]
    iq *= np.array(ACTIVE, np.int8)[None, :, None, None]
    iq *= mask[:, None, :, None].astype(np.int8)

    xy = np.full((k_ticks, 3, 2), np.nan)
    person_mask = np.zeros((k_ticks, 3))
    walking = np.zeros((k_ticks, 3))
    posture = np.full((k_ticks, 3), 255)
    zones = np.full((k_ticks, 3), -1)
    occupied = np.zeros((k_ticks, 9))
    for k in range(k_ticks):
        for i in range(p):
            (x, y), walk, post = person_at(i, k)
            xy[k, i] = (x, y)
            person_mask[k, i], walking[k, i], posture[k, i], zones[k, i] = 1, walk, post, zone(x, y)
            occupied[k, zone(x, y)] = 1
    frac = L_FLOOR_FRAC if l_shape else [1] * 9

    wlm_reason = [4 if k == 0 else 3 if k == 7 else 0 if k % 2 else 1 for k in range(k_ticks)]
    rti_reason = [4 if k == 0 else 0 if k == 3 else 2 for k in range(k_ticks)]
    wlm_xy = [(2.0 + 0.1 * k, 2.0) if r == 0 else (np.nan, np.nan) for k, r in enumerate(wlm_reason)]
    rti_xy = [(3.0, 2.5) if r == 0 else (np.nan, np.nan) for r in rti_reason]
    angle = 2 * np.pi * np.arange(n) / n + 0.3
    node_pos = np.stack([W / 2 + 0.42 * W * np.cos(angle), D / 2 + 0.42 * D * np.sin(angle), np.full(n, 1.2)], 1)
    meta = {"label": "SIM", "spec": "dataset v1", "scene_index": index, "scene_id": f"gen-{index:06}",
            "scene_sha256": hashlib.sha256(f"scene {index}".encode()).hexdigest(), "roomsim_seed": 1000 + index,
            "n_nodes": n, "n_people": p, "slot_s": 0.0125, "n_tx": n, "columns": cols,
            "calib_m0": cols + 4, "impair": impair}
    values = {
        "csi_iq": iq, "csi_mask": mask, "rssi": np.where(mask, -50, 0), "tx_slot": tx_slot,
        "subcarrier_k": BUFFER_K, "subcarrier_active": ACTIVE,
        "count": [p] * k_ticks, "presence": [int(p > 0)] * k_ticks, "xy": xy, "person_mask": person_mask,
        "walking": walking, "posture": posture, "zone": zones, "zone_occupied": occupied,
        "zone_single": [zones[k, 0] if p == 1 else -1 for k in range(k_ticks)],
        "t_label": [0.5 * (k + 1) for k in range(k_ticks)],
        "zone_floor_frac": frac, "zone_reachable": [int(f > 0) for f in frac],
        "wlm_xy": wlm_xy, "rti_xy": rti_xy,
        "wlm_radius": [1.5 if r == 0 else np.nan for r in wlm_reason],
        "rti_radius": [2.0 if r == 0 else np.nan for r in rti_reason],
        "wlm_reason": wlm_reason, "rti_reason": rti_reason,
        "calib_baseline": [-50.0] * n_links, "calib_null_sigma": [0.5] * n_links, "calib_mask": [1] * n_links,
        "calib_iq": np.zeros((n_links, 56, mc, 2)), "calib_mask_m": np.ones((n_links, mc)),
        "calib_rssi": np.full((n_links, mc), -50),
        "links": links, "node_ids": range(n), "node_pos": node_pos, "room_size": (W, D, H),
        "footprint": L_FOOTPRINT if l_shape else [(0, 0), (W, 0), (W, D), (0, D)],
        "window_end_tick": [4 * w + 3 for w in range(s // 2)],
        "meta_json": np.frombuffer(json.dumps(meta).encode(), np.uint8),
    }
    return {name: np.asarray(values[name], dtype=np.dtype(descr)) for name, descr in SPEC_72}


def in_memory(arrays):
    """A scene dict as `load_scene` returns it, without the files."""
    return dict(arrays, meta=json.loads(arrays["meta_json"].tobytes()))


def sha256(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def write_dataset(root, scenes, split_of=None, **manifest_fields):
    """Write `scenes` (make_arrays kwargs, or ready member dicts) as a SIM
    dataset under `root`; scene i goes to `split_of[i]` (default train)."""
    os.makedirs(os.path.join(root, "shards"))
    entries = []
    for i, spec in enumerate(scenes):
        arrays = spec if "csi_iq" in spec else make_arrays(index=i, **spec)
        shard = f"shards/gen-{i:06}.npz"
        np.savez(os.path.join(root, shard), **arrays)
        meta = json.loads(arrays["meta_json"].tobytes())
        entries.append({"index": i, "id": f"gen-{i:06}", "split": (split_of or {}).get(i, "train"),
                        "shard": shard, "shard_sha256": sha256(os.path.join(root, shard)),
                        "scene_sha256": meta["scene_sha256"], "n_nodes": meta["n_nodes"],
                        "n_people": meta["n_people"]})
    manifest = {"label": "SIM", "spec": "dataset v1", "seed": 0, "seconds": 4, "label_hz": 2,
                "impair": True, "baseline_valid": True, "from": None,
                "splits": {name: [e["index"] for e in entries if e["split"] == name] for name in rd.SPLITS},
                "scenes": entries}
    manifest.update(manifest_fields)
    with open(os.path.join(root, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f)
    return manifest


class TempDir(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()


class TestFormat(unittest.TestCase):
    def test_members_match_spec(self):
        """The loader's §7.2 table and this file's agree, name, order and dtype."""
        self.assertEqual([(m[0], m[1]) for m in rd.MEMBERS], list(SPEC_72))

    def test_synthetic_shard_is_stored_npy(self):
        """np.savez writes the members STORED, in order, as numpy loads them."""
        with tempfile.TemporaryDirectory() as tmp:
            write_dataset(tmp, [{}])
            with zipfile.ZipFile(os.path.join(tmp, "shards", "gen-000000.npz")) as z:
                self.assertEqual(z.namelist(), [n + ".npy" for n, _ in SPEC_72])
                self.assertTrue(all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist()))


class TestAmplitude(unittest.TestCase):
    def test_hand_values(self):
        """(imag, real) pairs, widened before squaring: -128 is 128, never an
        int8 overflow; (-128, -128) is sqrt(32768)."""
        iq = np.array([[3, 4], [-3, -4], [0, -128], [-128, 0], [-128, -128], [127, -128], [0, 0], [1, 0]], np.int8)
        want = [5.0, 5.0, 128.0, 128.0, math.sqrt(32768.0), math.sqrt(32513.0), 0.0, 1.0]
        got = rd.amplitude(iq)
        self.assertEqual(got.dtype, np.float64)
        self.assertEqual(got.tolist(), want)
        self.assertEqual(want[4], 181.01933598375618)

    def test_shape_and_refusals(self):
        self.assertEqual(rd.amplitude(np.zeros((2, 56, 3, 2), np.int8)).shape, (2, 56, 3))
        with self.assertRaises(ValueError):
            rd.amplitude(np.zeros((4, 2), np.int16))
        with self.assertRaises(ValueError):
            rd.amplitude(np.zeros((4, 3), np.int8))


class TestLoad(TempDir):
    def test_round_trip(self):
        manifest = write_dataset(self.dir, [{"n_people": 2}, {"n_nodes": 6, "l_shape": True}])
        want = make_arrays(index=1, n_nodes=6, l_shape=True)
        scene = rd.load_scene(self.dir, 1, manifest=rd.load_manifest(self.dir))
        for name, _ in SPEC_72:
            np.testing.assert_array_equal(scene[name], want[name], err_msg=name)
            self.assertEqual(scene[name].dtype, want[name].dtype, name)
        self.assertEqual(scene["meta"]["scene_id"], "gen-000001")
        self.assertEqual(scene["meta"]["label"], "SIM")
        self.assertEqual(scene["entry"], manifest["scenes"][1])
        self.assertEqual(rd.load_scene(self.dir, 0)["meta"]["n_people"], 2)

    def test_manifest_label_and_spec(self):
        with self.assertRaisesRegex(FileNotFoundError, "unfinished"):
            rd.load_manifest(self.dir)
        write_dataset(self.dir, [{}], label="REAL")
        with self.assertRaisesRegex(ValueError, "SIM"):
            rd.load_manifest(self.dir)
        with self.assertRaisesRegex(ValueError, "SIM"):
            rd.load_scene(self.dir, 0)

    def test_manifest_spec(self):
        write_dataset(self.dir, [{}], spec="dataset v2")
        with self.assertRaisesRegex(ValueError, "dataset v1"):
            rd.load_manifest(self.dir)

    def test_bad_sha_refused(self):
        write_dataset(self.dir, [{}])
        path = os.path.join(self.dir, "shards", "gen-000000.npz")
        with open(path, "r+b") as f:     # one bit of one CSI byte
            f.seek(200)
            byte = f.read(1)[0]
            f.seek(200)
            f.write(bytes([byte ^ 1]))
        with self.assertRaisesRegex(ValueError, "sha256"):
            rd.load_scene(self.dir, 0)

    def test_meta_label_refused(self):
        """A shard whose meta_json is not SIM is refused even when its
        sha256 is the manifest's."""
        arrays = make_arrays()
        meta = json.loads(arrays["meta_json"].tobytes())
        meta["label"] = "REAL"
        arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        write_dataset(self.dir, [arrays])
        with self.assertRaisesRegex(ValueError, "not 'SIM'"):
            rd.load_scene(self.dir, 0)

    def test_wrong_members_refused(self):
        bad_dtype = make_arrays()
        bad_dtype["xy"] = bad_dtype["xy"].astype(np.float64)
        bad_shape = make_arrays()
        bad_shape["rssi"] = bad_shape["rssi"][:, :-1]
        missing = make_arrays()
        del missing["rti_xy"]
        extra = dict(make_arrays(), phase=np.zeros(3, np.float32))
        write_dataset(self.dir, [bad_dtype, bad_shape, missing, extra])
        for i, what in enumerate(("xy is <f8", "rssi has shape", "missing \\['rti_xy'\\]", "unexpected \\['phase'\\]")):
            with self.subTest(i=i), self.assertRaisesRegex(ValueError, what):
                rd.load_scene(self.dir, i)

    def test_meta_must_name_the_scene(self):
        write_dataset(self.dir, [make_arrays(index=5)])   # meta says scene 5, the manifest 0
        with self.assertRaisesRegex(ValueError, "names scene \\(5, 'gen-000005'"):
            rd.load_scene(self.dir, 0)

    def test_meta_keys(self):
        arrays = make_arrays()
        meta = json.loads(arrays["meta_json"].tobytes())
        del meta["calib_m0"]
        arrays["meta_json"] = np.frombuffer(json.dumps(meta).encode(), np.uint8)
        write_dataset(self.dir, [arrays])
        with self.assertRaisesRegex(ValueError, "lacks \\['calib_m0'\\]"):
            rd.load_scene(self.dir, 0)

    def test_shard_outside_dataset_refused(self):
        manifest = write_dataset(self.dir, [{}])
        manifest["scenes"][0]["shard"] = "../elsewhere.npz"
        with self.assertRaisesRegex(ValueError, "outside"):
            rd.load_scene(self.dir, 0, manifest=manifest)
        with self.assertRaises(KeyError):
            rd.load_scene(self.dir, 3, manifest=manifest)


class TestWindow(unittest.TestCase):
    def check_slots(self, win, link, columns, valid):
        """Slot j of `link` is column `columns[j]`, received iff `valid[j]`,
        with amplitude sqrt((l % 100)^2 + (m + 1)^2) on active subcarriers, 0 on nulls."""
        np.testing.assert_array_equal(win.column[link], columns)
        np.testing.assert_array_equal(win.mask[link], valid)
        active = np.array(ACTIVE, bool)
        for j, (m, ok) in enumerate(zip(columns, valid)):
            got = win.amplitude[link, :, j]
            if ok:
                want = np.float32(math.sqrt(float((link % 100) ** 2 + (m + 1) ** 2)))
                self.assertTrue((got[active] == want).all(), (link, j))
                self.assertTrue((got[~active] == 0).all())
            else:
                self.assertTrue(np.isnan(got).all(), (link, j))

    def test_t_by_hand(self):
        """§4.3: T = ceil(80 l / N) is 40 at 4 nodes, 20 at 8, 14 at 12 for l = 2."""
        for n, t in ((4, 40), (8, 20), (12, 14), (6, 27), (5, 32)):
            win = rd.window(in_memory(make_arrays(n_nodes=n)), 7, 2.0)
            self.assertEqual(win.amplitude.shape, (n * (n - 1), 56, t))
            self.assertEqual(win.amplitude.dtype, np.float32)
            self.assertEqual(win.mask.shape, (n * (n - 1), t))
            self.assertEqual(win.link_mask.shape, (n * (n - 1),))

    def test_n4_whole_columns(self):
        """N = 4, k = 7 (t = 4 s = S), l = 2: n in [160, 320), so every slot
        s takes m = 40..79 (4*79 + 3 = 319): 40 columns, none masked."""
        scene = in_memory(make_arrays(n_nodes=4))
        win = rd.window(scene, 7)
        for link in (0, 5, 11):
            self.check_slots(win, link, list(range(40, 80)), [True] * 40)
        self.assertTrue(win.link_mask.all())

    def test_n12_partial_column(self):
        """N = 12, k = 3 (t = 2), l = 2: n in [0, 160), 160 = 13*12 + 4.
        Slots 0..3 hold m = 0..13 (12*13 + 3 = 159); slots 4..11 only
        m = 0..12 (12*13 + 4 = 160 is out), so their slot 0 is column -1,
        masked."""
        scene = in_memory(make_arrays(n_nodes=12))
        win = rd.window(scene, 3)
        tx = scene["links"][:, 0]
        for link in (0, 11 * 3, 11 * 3 + 10):          # tx 0, tx 3, tx 3
            self.check_slots(win, link, list(range(0, 14)), [True] * 14)
        for link in (11 * 4, 11 * 7 + 2, 11 * 11 + 10):   # tx 4, 7, 11
            self.check_slots(win, link, list(range(-1, 13)), [False] + [True] * 13)
        np.testing.assert_array_equal(win.mask.sum(axis=1), np.where(tx < 4, 14, 13))

    def test_n12_last_window_and_last_column(self):
        """N = 12, k = 7 (t = 4 = S), l = 2: n in [160, 320); 160 = 13*12 + 4,
        so slots 4..7 take 14 columns m = 13..26, slots 0..3 13 columns
        m = 14..26 (12*13 + 3 = 159 is out), slots 8..11 13 columns m = 13..25:
        the last column 26 (n = 312 + s) is past 80 S = 320 for s >= 8 and is
        never read for them."""
        scene = in_memory(make_arrays(n_nodes=12))
        self.assertEqual(scene["csi_mask"].shape[1], 27)                        # ceil(320/12)
        self.assertEqual(scene["csi_mask"][:, 26].tolist(), [1 if tx < 8 else 0 for tx, _ in scene["links"]])
        win = rd.window(scene, 7)
        self.check_slots(win, 11 * 5, list(range(13, 27)), [True] * 14)            # tx 5
        self.check_slots(win, 11 * 2 + 1, list(range(13, 27)), [False] + [True] * 13)  # tx 2
        self.check_slots(win, 11 * 9, list(range(12, 26)), [False] + [True] * 13)      # tx 9

    def test_n6_partial_column(self):
        """N = 6, k = 3, l = 2: T = ceil(160/6) = 27, 160 = 26*6 + 4: slots
        0..3 hold m = 0..26, slots 4 and 5 m = 0..25 (slot 0 masked)."""
        scene = in_memory(make_arrays(n_nodes=6))
        win = rd.window(scene, 3)
        self.check_slots(win, 5 * 3, list(range(0, 27)), [True] * 27)              # tx 3
        self.check_slots(win, 5 * 4 + 1, list(range(-1, 26)), [False] + [True] * 26)  # tx 4
        self.check_slots(win, 5 * 5 + 4, list(range(-1, 26)), [False] + [True] * 26)  # tx 5

    def test_half_second(self):
        """N = 12, k = 0, l = 0.5: n in [0, 40), T = ceil(40/12) = 4;
        slots 0..3 hold m = 0..3, slots 4..11 m = 0..2."""
        scene = in_memory(make_arrays(n_nodes=12))
        win = rd.window(scene, 0, 0.5)
        self.check_slots(win, 11 * 3, [0, 1, 2, 3], [True] * 4)
        self.check_slots(win, 11 * 4, [-1, 0, 1, 2], [False, True, True, True])

    def test_k0_edge(self):
        """N = 4, k = 0 (t = 0.5), l = 2: n in [-120, 40). Only m = 0..9
        exist; the first 30 slots are before the stream (m = -30..-1)."""
        scene = in_memory(make_arrays(n_nodes=4))
        win = rd.window(scene, 0, 2.0)
        for link in (0, 7):
            self.check_slots(win, link, list(range(-30, 10)), [False] * 30 + [True] * 10)
        self.assertTrue(win.link_mask.all())

    def test_lost_packets(self):
        """Lost packets are masked; a link with every window packet lost
        is out of link_mask."""
        lost = [(0, 75)] + [(5, m) for m in range(40, 80)]
        scene = in_memory(make_arrays(n_nodes=4, lost=lost))
        win = rd.window(scene, 7)
        self.check_slots(win, 0, list(range(40, 80)), [m != 75 for m in range(40, 80)])
        self.assertEqual(win.link_mask.tolist(), [link != 5 for link in range(12)])
        self.assertTrue(np.isnan(win.amplitude[5]).all())
        filled = rd.window(scene, 7, fill=0.0)
        self.assertTrue((filled.amplitude[5] == 0).all())
        self.assertFalse(np.isnan(filled.amplitude).any())

    def test_brute_force(self):
        """§4.3 with exact fractions for N = 4..12, every tick and length, and
        random losses: the received slots are exactly the columns whose n is
        in [80 (t_k - l), 80 t_k); each link has T or T - 1 of them."""
        rng = np.random.default_rng(7)
        for n in range(4, 13):
            arrays = make_arrays(n_nodes=n)
            arrays["csi_mask"] &= (rng.random(arrays["csi_mask"].shape) > 0.2).astype(np.uint8)
            scene = in_memory(arrays)
            cols = arrays["csi_mask"].shape[1]
            for length in (0.5, 1.0, 1.5, 2.0, 3.0):
                t = math.ceil(Fraction(80) * Fraction(length) / n)
                for k in range(8):
                    t_k = Fraction(k + 1, 2)
                    lo, hi = 80 * (t_k - Fraction(length)), 80 * t_k
                    win = rd.window(scene, k, length)
                    self.assertEqual(win.mask.shape, (n * (n - 1), t))
                    for link, slot in enumerate(arrays["tx_slot"].tolist()):
                        span = [m for m in range(-100, cols + 100) if lo <= m * n + slot < hi]
                        self.assertIn(len(span), (t, t - 1) if (80 * Fraction(length)) % n else (t,))
                        want = {m for m in span if 0 <= m < cols and arrays["csi_mask"][link, m]}
                        got = set(win.column[link][win.mask[link]].tolist())
                        self.assertEqual(got, want, (n, length, k, link))
                        self.assertEqual(win.column[link][-1], span[-1])

    def test_bad_arguments(self):
        scene = in_memory(make_arrays())
        for k in (-1, 8):
            with self.assertRaises(IndexError):
                rd.window(scene, k)
        for length in (0, 0.3, -0.5, 1.25):
            with self.assertRaises(ValueError):
                rd.window(scene, 3, length)
        self.assertEqual(rd.window(scene, np.int32(3), 1).mask.shape, (12, 20))


class TestLabelsBaseline(unittest.TestCase):
    def test_labels_two_people(self):
        """k = 3: P0 walking at (1.25, 1.0), zone 0; P1 sitting at
        (2.5, 3.0), zone 3*2 + 1 = 7."""
        lab = rd.labels_at(in_memory(make_arrays(n_people=2)), 3)
        self.assertEqual((lab["label"], lab["t"], lab["count"], lab["presence"]), ("SIM", 2.0, 2, True))
        self.assertEqual(lab["zone_occupied"].tolist(), [1, 0, 0, 0, 0, 0, 0, 1, 0])
        self.assertEqual(lab["zone_single"], -1)
        self.assertEqual(lab["people"], [
            {"index": 0, "x": 1.25, "y": 1.0, "zone": 0, "walking": True, "posture": "stand"},
            {"index": 1, "x": 2.5, "y": 3.0, "zone": 7, "walking": False, "posture": "sit"}])

    def test_labels_one_and_none(self):
        one = rd.labels_at(in_memory(make_arrays(n_people=1)), 7)     # P0 at (2.25, 1.0): zone 1
        self.assertEqual((one["count"], one["zone_single"], one["people"][0]["zone"]), (1, 1, 1))
        none = rd.labels_at(in_memory(make_arrays(n_people=0)), 7)
        self.assertEqual((none["count"], none["presence"], none["people"], none["zone_single"]), (0, False, [], -1))
        self.assertEqual(rd.zone_of(2.25, 1.0, W, D), 1)
        self.assertEqual(rd.zone_of(W, D, W, D), 8)            # the far edge stays in the last cell

    def test_baseline(self):
        scene = in_memory(make_arrays())
        b0 = rd.baseline_at(scene, 0)
        self.assertEqual(b0["wlm"], {"published": False, "reason": "warming_up", "xy": None, "radius_m": None})
        self.assertEqual(b0["rti"]["reason"], "warming_up")
        b3 = rd.baseline_at(scene, 3)
        self.assertEqual(b3["wlm"], {"published": True, "reason": "estimate", "xy": (np.float32(2.3).item(), 2.0),
                                     "radius_m": 1.5})
        self.assertEqual(b3["rti"], {"published": True, "reason": "estimate", "xy": (3.0, 2.5), "radius_m": 2.0})
        self.assertEqual(rd.baseline_at(scene, 7)["wlm"]["reason"], "implausible_radius")
        self.assertEqual(rd.baseline_at(scene, 2)["rti"]["reason"], "not_significant")
        self.assertTrue(b3["valid"])
        self.assertFalse(rd.baseline_at(in_memory(make_arrays(impair=False)), 3)["valid"])

    def test_baseline_never_a_centre_without_radius(self):
        no_radius = make_arrays()
        no_radius["wlm_radius"][3] = np.nan
        stray_centre = make_arrays()
        stray_centre["rti_xy"][2] = (1.0, 1.0)
        bad_code = make_arrays()
        bad_code["wlm_reason"][2] = 9
        for arrays, k in ((no_radius, 3), (stray_centre, 2), (bad_code, 2)):
            with self.assertRaises(ValueError):
                rd.baseline_at(in_memory(arrays), k)


class TestSplits(TempDir):
    def test_splits(self):
        manifest = write_dataset(self.dir, [{}] * 5, split_of={3: "val", 1: "test"})
        self.assertEqual(rd.splits(manifest), {"train": [0, 2, 4], "val": [3], "test": [1]})

    def test_inconsistent_splits_refused(self):
        manifest = write_dataset(self.dir, [{}] * 3, split_of={2: "test"})
        cases = []
        overlap = json.loads(json.dumps(manifest))
        overlap["splits"]["val"] = [0]
        cases.append(overlap)
        missing = json.loads(json.dumps(manifest))
        missing["splits"]["train"] = [0]
        cases.append(missing)
        disagree = json.loads(json.dumps(manifest))
        disagree["scenes"][2]["split"] = "val"
        cases.append(disagree)
        for case in cases:
            with self.assertRaises(ValueError):
                rd.splits(case)


class TestPlot(TempDir):
    def setUp(self):
        super().setUp()
        write_dataset(self.dir, [{"n_people": 2}, {"n_nodes": 6, "n_people": 1, "l_shape": True,
                                                   "lost": [(3, m) for m in range(30, 54)]}])

    def assert_png(self, path):
        with open(path, "rb") as f:
            data = f.read()
        self.assertEqual(data[:8], b"\x89PNG\r\n\x1a\n")
        self.assertGreater(len(data), 20_000)

    def test_published_and_abstained(self):
        """Window 0 ends at k = 3 (WLM published), window 1 at k = 7 (WLM
        abstained, implausible_radius); scene 1 is an L with a lost link."""
        for scene, w in ((0, 0), (1, 1)):
            out = os.path.join(self.dir, f"s{scene}w{w}.png")
            plot_sample.plot_sample(self.dir, scene, w, out)
            self.assert_png(out)
        out = os.path.join(self.dir, "cli.png")
        with contextlib.redirect_stdout(io.StringIO()) as said:
            argv = [self.dir, "--scene", "1", "--window", "0", "--out", out, "--links", "5"]
            self.assertEqual(plot_sample.main(argv), 0)
        self.assertIn("SIM", said.getvalue())
        self.assert_png(out)
        with self.assertRaises(IndexError):
            plot_sample.plot_sample(self.dir, 0, 2, out)

    def test_script_runs_isolated(self):
        """`python -I ml/plot_sample.py ...` finds its package without the
        caller's sys.path."""
        out = os.path.join(self.dir, "script.png")
        script = os.path.join(ROOT, "ml", "plot_sample.py")
        run = subprocess.run([sys.executable, "-I", script, self.dir, "--scene", "0", "--window", "1", "--out", out],
                             capture_output=True, text=True, cwd=self.dir,
                             env=dict(os.environ, MPLCONFIGDIR=self.dir), timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertIn("SIM", run.stdout)
        self.assert_png(out)


@unittest.skipUnless(os.environ.get("SAANS_DATASET_DIR"), "SAANS_DATASET_DIR names no real dataset")
class TestRealDataset(unittest.TestCase):
    """Every shard of a real `saans gen-dataset` output against §7.2 (and
    §4, §5, §6's invariants): numpy loads every member with the right shape
    and dtype. Reads the whole dataset (about 9 MB per scene)."""

    def test_every_shard(self):
        root = os.environ["SAANS_DATASET_DIR"]
        manifest = rd.load_manifest(root)
        rd.splits(manifest)
        s = manifest["seconds"]
        k_ticks = 2 * s
        for entry in manifest["scenes"]:
            with self.subTest(scene=entry["index"]):
                scene = rd.load_scene(root, entry["index"], manifest=manifest)
                self.check_bytes(root, entry)
                self.check_shapes(scene, manifest, entry, k_ticks)
                self.check_labels(scene, k_ticks)
                for k in (3, k_ticks - 1):
                    n = scene["meta"]["n_tx"]
                    self.assertEqual(rd.window(scene, k).amplitude.shape, (n * (n - 1), 56, -(-160 // n)))

    def check_bytes(self, root, entry):
        """§7.2's bytes: STORED members in order, npy 1.0, C order, data at a
        multiple of 64."""
        with zipfile.ZipFile(rd._shard_path(root, entry["shard"])) as z:
            self.assertEqual(z.namelist(), [n + ".npy" for n, _ in SPEC_72])
            for info, (name, descr) in zip(z.infolist(), SPEC_72):
                self.assertEqual(info.compress_type, zipfile.ZIP_STORED, name)
                with z.open(info) as f:
                    self.assertEqual(np.lib.format.read_magic(f), (1, 0), name)
                    _, fortran, dtype = np.lib.format.read_array_header_1_0(f)
                    self.assertFalse(fortran, name)
                    self.assertEqual(dtype.str, descr, name)
                    self.assertEqual(f.tell() % 64, 0, name)

    def check_shapes(self, scene, manifest, entry, k_ticks):
        meta = scene["meta"]
        n = meta["n_nodes"]
        n_links, cols, mc = n * (n - 1), -(-80 * manifest["seconds"] // n), scene["calib_mask_m"].shape[1]
        want = {"csi_iq": (n_links, 56, cols, 2), "csi_mask": (n_links, cols), "rssi": (n_links, cols),
                "tx_slot": (n_links,), "subcarrier_k": (56,), "subcarrier_active": (56,),
                "xy": (k_ticks, 3, 2), "zone_occupied": (k_ticks, 9), "wlm_xy": (k_ticks, 2),
                "calib_iq": (n_links, 56, mc, 2), "calib_rssi": (n_links, mc), "links": (n_links, 2),
                "node_ids": (n,), "node_pos": (n, 3), "room_size": (3,), "zone_floor_frac": (9,),
                "window_end_tick": (manifest["seconds"] // 2,)}
        for name in ("count", "presence", "zone_single", "t_label", "wlm_radius", "rti_radius",
                     "wlm_reason", "rti_reason"):
            want[name] = (k_ticks,)
        for name in ("person_mask", "walking", "posture", "zone"):
            want[name] = (k_ticks, 3)
        for name in ("calib_baseline", "calib_null_sigma", "calib_mask"):
            want[name] = (n_links,)
        for name, shape in want.items():
            self.assertEqual(scene[name].shape, shape, name)
        self.assertIn(scene["footprint"].shape, ((4, 2), (6, 2)))
        self.assertEqual(scene["calib_mask_m"].shape, (n_links, mc))
        for key in ("scene_index", "scene_id", "scene_sha256", "roomsim_seed", "n_nodes", "n_people",
                    "slot_s", "n_tx", "columns", "calib_m0", "impair"):
            self.assertIn(key, meta)
        self.assertEqual((meta["columns"], meta["n_people"], meta["impair"], meta["scene_sha256"]),
                         (cols, entry["n_people"], manifest["impair"], entry["scene_sha256"]))
        scene_file = os.path.join(os.environ["SAANS_DATASET_DIR"], "scenes", entry["id"] + ".json")
        if os.path.exists(scene_file):
            self.assertEqual(sha256(scene_file), meta["scene_sha256"])

    def check_labels(self, scene, k_ticks):
        received = scene["csi_mask"].astype(bool)
        self.assertTrue((scene["csi_iq"][~np.repeat(received[:, None, :], 56, axis=1)] == 0).all())
        self.assertTrue((scene["rssi"][~received] == 0).all())
        self.assertTrue(((scene["rssi"][received] >= -100) & (scene["rssi"][received] <= 0)).all())
        self.assertEqual(scene["subcarrier_k"].tolist(), BUFFER_K)
        self.assertEqual(scene["subcarrier_active"].tolist(), ACTIVE)
        self.assertEqual(scene["t_label"].tolist(), [0.5 * (k + 1) for k in range(k_ticks)])
        self.assertEqual(scene["window_end_tick"].tolist(), [4 * w + 3 for w in range(k_ticks // 4)])
        count = scene["count"]
        self.assertTrue((count == count[0]).all() and count[0] <= 3)
        np.testing.assert_array_equal(scene["presence"], count > 0)
        present = np.arange(3)[None, :] < count[:, None]
        np.testing.assert_array_equal(scene["person_mask"], present)
        np.testing.assert_array_equal(np.isnan(scene["xy"]).any(axis=2), ~present)
        np.testing.assert_array_equal(scene["posture"] == 255, ~present)
        np.testing.assert_array_equal(scene["zone"] == -1, ~present)
        occupied = np.zeros((k_ticks, 9), np.uint8)
        for k, i in zip(*np.nonzero(present)):
            occupied[k, scene["zone"][k, i]] = 1
        np.testing.assert_array_equal(scene["zone_occupied"], occupied)
        np.testing.assert_array_equal(scene["zone_single"], np.where(count == 1, scene["zone"][:, 0], -1))
        np.testing.assert_array_equal(scene["zone_reachable"], scene["zone_floor_frac"] > 0)
        self.assertEqual((scene["wlm_reason"][0], scene["rti_reason"][0]), (4, 4))
        for k in range(k_ticks):
            rd.baseline_at(scene, k)          # a published estimate always has its radius


if __name__ == "__main__":
    unittest.main()
