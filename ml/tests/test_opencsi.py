"""ml/opencsi.py (spec/phase9.md §2) on a small hand-built trial, and on the real dataset when present."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from ml import opencsi as oc
from ml import preprocess as pp

MACS = ("AA:00:00:00:00:03", "AA:00:00:00:00:01", "AA:00:00:00:00:02")   # node ids by sorted MAC: 2, 0, 1
T0 = 1_700_000_000


def _iq52(scale: int, *, entries: int = 52) -> list:
    """52 (imag, real) pairs in the OpenCSI HT20 layout: entry 0 and 27..37 zero, the rest non-zero."""
    out = []
    for e in range(entries):
        if e == 0 or 27 <= e <= 37:
            out.append([0, 0])
        else:
            out.append([(e % 7) - 3, scale + (e % 5)])
    return out


def _record(tx: int, rx: int, ts: str, scale: int = 20, rssi: int = -40, fw: int = 1000, entries: int = 52) -> str:
    """One raw line; `ts` is written verbatim (the adapter must parse it exactly)."""
    iq = _iq52(scale, entries=entries) if entries == 52 else [[1, 2]] * entries
    return ('{"record_id":"x","trial_id":"T","seq":1,"wall_ts":%s,"server_ts":%s,"firmware_ts_us":%d,'
            '"tx_mac":"%s","rx_mac":"%s","rssi":%d,"noise_floor":0,"snr":-40.0,"num_subcarriers":%d,'
            '"bandwidth":"HT20","iq_data":%s,"amplitude_mean":1.0,"amplitude_std":0.5,"amplitude_var":0.25}'
            % (ts, ts, fw, MACS[tx], MACS[rx], rssi, entries, json.dumps(iq, separators=(",", ":"))))


class _Trial:
    """A temporary dataset root with one HT20 trial, registered in `oc.TRIALS` while it lives."""

    NAME = "envT_S3"

    def __init__(self, lines: list, segments: list):
        import zstandard
        self.root = tempfile.mkdtemp(prefix="opencsi-test-")
        os.makedirs(os.path.join(self.root, "environments"))
        with open(os.path.join(self.root, "environments", "envT.json"), "w") as fh:
            json.dump({"environment_id": "envT", "dimensions_m": {"length": 5.5, "width": 4.25}}, fh)
        tdir = os.path.join(self.root, "trials", self.NAME)
        os.makedirs(os.path.join(tdir, "raw"))
        with open(os.path.join(tdir, "annotations.jsonl"), "w") as fh:
            for start, end, label in segments:
                fh.write(json.dumps({"start_ts": start, "end_ts": end, "fine_label": label, "note": label}) + "\n")
        with open(os.path.join(tdir, "raw", "csi_raw.jsonl.zst"), "wb") as fh:
            fh.write(zstandard.ZstdCompressor().compress("".join(lines).encode()))
        oc.TRIALS[self.NAME] = ("envT", "test", "opencsi-lltf52")

    def close(self):
        oc.TRIALS.pop(self.NAME, None)
        shutil.rmtree(self.root, ignore_errors=True)


def _lines() -> list:
    """Node 2 -> 0 (MACS[0] -> MACS[1]) at 10 Hz over 0..9.9 s; node 0 -> 1 a burst of 10 in one 200 ms cell;
    one exact duplicate, one other-layout record and a cut-off last line."""
    out = []
    for i in range(100):
        out.append(_record(0, 1, f"{T0 + i // 10}.{i % 10}000001", scale=20 + (i % 3), fw=1000 + i) + "\n")
    for j in range(10):
        out.append(_record(1, 2, f"{T0 + 3}.{j:03d}", scale=30, fw=5000 + j) + "\n")
    out.append(out[5])                                                     # exact duplicate
    out.append(_record(1, 2, f"{T0 + 4}.5", entries=256) + "\n")             # another layout: skipped
    out.append(_record(0, 1, f"{T0 + 9}.95")[:150])                         # cut off: malformed
    return out


SEGMENTS = [(T0 + 0.5, T0 + 4.5, "empty_room"), (T0 + 5.0, T0 + 9.75, "occupied_moving")]


class TestLayouts(unittest.TestCase):
    def test_lltf52_is_the_first_52_entries_of_the_lltf_buffer(self):
        lay = oc.LAYOUTS["opencsi-lltf52"]
        self.assertEqual(lay.entries, 52)
        self.assertEqual(lay.k[1], 1)
        self.assertEqual(lay.k[26], 26)
        self.assertEqual(lay.k[38], -26)
        self.assertEqual(lay.k[51], -13)
        self.assertTrue(all(lay.grid[e] is None for e in [0] + list(range(27, 38))))
        carried_k = sorted(int(pp.K_GRID[b]) for b in np.flatnonzero(lay.carried))
        self.assertEqual(carried_k, sorted(pp.SHARED40))
        self.assertEqual(len(carried_k), 40)

    def test_he256_tone_4k_is_subcarrier_k(self):
        lay = oc.LAYOUTS["opencsi-he256"]
        self.assertEqual(lay.entries, 256)
        self.assertTrue(lay.carried.all())
        for k in (-26, -1, 1, 13, 26):
            self.assertEqual(lay.k[128 + 4 * k], k)
        self.assertIsNone(lay.k[128 + 2])
        self.assertIsNone(lay.k[128])

    def test_pairs_are_imag_then_real(self):
        lay = oc.LAYOUTS["opencsi-lltf52"]
        entries = np.zeros((52, 2), np.int8)
        entries[1] = (3, -7)                                                 # (imag, real) at k = 1
        grid = pp.to_grid(entries, lay)
        self.assertEqual(grid[pp.GRID_K.index(1)], complex(-7, 3))

    def test_shared40_refuses_a_source_without_it(self):
        lacking = np.ones(52, bool)
        lacking[pp.GRID_K.index(-13)] = False
        with self.assertRaises(ValueError):
            pp._ref_mask("shared40", lacking)
        self.assertEqual(int(pp._ref_mask("shared40", oc.LAYOUTS["opencsi-lltf52"].carried).sum()), 40)
        with self.assertRaises(ValueError):
            pp._ref_mask("shared44", oc.LAYOUTS["opencsi-lltf52"].carried)


class TestCap(unittest.TestCase):
    def test_at_most_cap_evenly_spaced_first_and_last_kept(self):
        for n in range(1, 25):
            keep = oc._cap_keep(np.zeros(n, np.int64), np.arange(n, dtype=np.int64), 4, 10**9)
            ranks = np.flatnonzero(keep).tolist()
            if n <= 4:
                self.assertEqual(ranks, list(range(n)))
            else:
                self.assertEqual(len(ranks), 4, n)
                self.assertEqual((ranks[0], ranks[-1]), (0, n - 1))
                want = [(2 * i * (n - 1) + 3) // 6 for i in range(4)]
                self.assertEqual(ranks, want)

    def test_cells_and_links_are_separate(self):
        link = np.array([0] * 6 + [1] * 6)
        t = np.array([0, 1, 2, 3, 4, 5, 0, 1, 2, 10, 11, 12], np.int64)
        keep = oc._cap_keep(link, t, 2, 10)
        self.assertEqual(np.flatnonzero(keep).tolist(), [0, 5, 6, 8, 9, 11])


class TestSession(unittest.TestCase):
    def setUp(self):
        self.trial = _Trial(_lines(), SEGMENTS)
        self.cache = tempfile.mkdtemp(prefix="opencsi-cache-")

    def tearDown(self):
        self.trial.close()
        shutil.rmtree(self.cache, ignore_errors=True)

    def test_parse_counts_nodes_and_exact_times(self):
        s = oc.load_session(self.trial.root, _Trial.NAME, cache_dir=None, cap=None)
        self.assertEqual(s.label, "REAL")
        self.assertEqual((s.stats["malformed"], s.stats["duplicates"], s.stats["other_layout"]), (1, 1, 1))
        self.assertEqual(s.stats["records"], 110)
        self.assertEqual(s.n_nodes, 3)
        # MACS[0] (...03) is node 2, MACS[1] (...01) node 0, MACS[2] (...02) node 1.
        self.assertEqual(s.links.tolist(), [[0, 1], [2, 0]])
        self.assertEqual(s.t0_s, T0)
        l20 = s.t_ns[s.offsets[1]:s.offsets[2]]
        self.assertEqual(int(l20[0]), 100)                                    # "1700000000.0000001" -> 100 ns, exact
        self.assertEqual(int(l20[13]), 1_300_000_100)
        self.assertTrue((np.diff(l20) > 0).all())
        self.assertEqual(s.room_m, (5.5, 4.25))
        self.assertEqual([g.fine_label for g in s.segments], ["empty_room", "occupied_moving"])
        self.assertEqual(s.segments[0].start_ns, 500_000_000)

    def test_cap_applies_per_link_cell(self):
        s = oc.load_session(self.trial.root, _Trial.NAME, cache_dir=None)
        self.assertEqual(s.stats["capped_out"], 6)                            # the burst of 10 keeps 4
        burst = s.t_ns[s.offsets[0]:s.offsets[1]]
        self.assertEqual(((burst - 3_000_000_000) // 1_000_000).tolist(), [0, 3, 6, 9])

    def test_cache_round_trip(self):
        a = oc.load_session(self.trial.root, _Trial.NAME, cache_dir=self.cache)
        real = oc.parse_trial
        oc.parse_trial = lambda *args, **kw: (_ for _ in ()).throw(AssertionError("re-parsed"))
        try:
            b = oc.load_session(self.trial.root, _Trial.NAME, cache_dir=self.cache)
        finally:
            oc.parse_trial = real
        for name in ("links", "offsets", "t_ns", "iq", "rssi"):
            np.testing.assert_array_equal(getattr(a, name), getattr(b, name))
        self.assertEqual(a.segments, b.segments)
        self.assertEqual(a.stats, b.stats)

    def test_windows_lie_inside_one_segment(self):
        s = oc.load_session(self.trial.root, _Trial.NAME, cache_dir=None)
        ends = oc.window_ends(s, 2.0, 0.5)
        for e in ends:
            seg = s.segments[e.segment]
            self.assertGreaterEqual(e.t_end_ns - 2_000_000_000, seg.start_ns)
            self.assertLessEqual(e.t_end_ns, seg.end_ns)
        # 4.0 s and 4.75 s segments: floor((dur - 2) / 0.5) + 1 windows each.
        self.assertEqual(sum(e.segment == 0 for e in ends), 5)
        self.assertEqual(sum(e.segment == 1 for e in ends), 6)
        lab = oc.labels(s, ends[-1])
        self.assertEqual((lab["presence"], lab["count"], lab["moving"]), (True, 1, True))
        self.assertEqual(oc.labels(s, ends[0])["presence"], False)
        self.assertFalse(lab["positions_known"])

    def test_window_record(self):
        s = oc.load_session(self.trial.root, _Trial.NAME, cache_dir=None)
        w = oc.window(s, 4_000_000_000, 2.0)                                  # [2, 4) s
        self.assertEqual(w.label, "REAL")
        self.assertEqual(w.source, f"opencsi:{_Trial.NAME}")
        self.assertEqual(w.window_end, f"t_end_ns={T0 * 10**9 + 4_000_000_000}")
        self.assertFalse(w.link_pos_known.any())
        self.assertEqual(w.received.sum(axis=1).tolist(), [4, 20])           # the capped burst; 10 Hz x 2 s
        self.assertEqual(w.csi.shape, (2, 20, 52))
        self.assertTrue(w.received[0, -4:].all() and not w.received[0, :-4].any())   # right-aligned
        t = w.t_rel_ns[w.received]
        self.assertTrue(((t >= -2_000_000_000) & (t < 0)).all())
        out = pp.preprocess(w, pp.PreprocessConfig(bin_hz=5, ref_bins="shared40"))
        self.assertEqual(out["label"], "REAL")
        self.assertEqual(int(out["ref"].sum()), 40)
        # The burst's 4 packets fill one 200 ms bin, under min_packets = 3 bins: that link is not used.
        self.assertEqual(out["link_mask"].tolist(), [False, True])

    def test_a_link_silent_in_a_window_has_nothing_received(self):
        s = oc.load_session(self.trial.root, _Trial.NAME, cache_dir=None)
        w = oc.window(s, 9_000_000_000, 1.0)
        self.assertEqual(w.received.sum(axis=1).tolist(), [0, 10])


@unittest.skipUnless(os.environ.get("SAANS_OPENCSI") and os.path.isdir(oc.DEFAULT_ROOT),
                     "set SAANS_OPENCSI=1 with the dataset at oc.DEFAULT_ROOT")
class TestRealDataset(unittest.TestCase):
    def test_every_trial_loads(self):
        for trial, (_env, _chip, lay) in oc.TRIALS.items():
            if trial.startswith("envT"):
                continue
            s = oc.load_session(oc.DEFAULT_ROOT, trial)
            self.assertEqual((s.n_nodes, s.links.shape[0]), (8, 56), trial)
            self.assertEqual([g.fine_label for g in s.segments],
                             ["empty_room", "occupied_static", "occupied_moving", "empty_room"], trial)
            self.assertLessEqual(s.stats["malformed"], 1)
            self.assertEqual(int(s.layout.carried.sum()), 40 if lay == "opencsi-lltf52" else 52)


if __name__ == "__main__":
    unittest.main()
