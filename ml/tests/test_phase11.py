"""ml/phase11 (spec/recording.md) on a hand-built recording: one transmitter
heard by two receivers through today's datagram, labels, the cache in
Phase 9's format. The fixture is synthetic; it is labelled REAL only
because that is the path a recording takes (no figure comes from it)."""

import json
import os
import shutil
import tempfile
import unittest

import numpy as np

from ml.phase11 import recordings as rec
from ml.phase9 import real_data as rdm
from wifisense.sources.esp32_udp import build_esp32_frame
from wifisense.sources.recorder import PcapWriter

T0 = 1_790_500_000_000_000_000


def _iq(rng, moving: bool) -> list:
    """56 entries in the fw56 layout, (imag, real); entry 0 and 27..37 zero (DC and guards)."""
    out = []
    for e in range(56):
        if e == 0 or 27 <= e <= 37:
            out.append((0, 0))
        else:
            a = 20 + (8 if moving else 1) * rng.standard_normal()
            out.append((int(np.clip(round(0.3 * a), -127, 127)), int(np.clip(round(a), -127, 127))))
    return out


def _session(dirpath: str) -> None:
    rng = np.random.default_rng(0)
    with PcapWriter(os.path.join(dirpath, "capture.pcap")) as w:
        seq = {1: 0, 2: 0}
        for i in range(121 * 20):                                           # 20 Hz for 121 s
            t = T0 + i * 50_000_000
            moving = t >= T0 + 101 * 10**9
            for node in (1, 2):
                frame = build_esp32_frame(_iq(rng, moving), node_id=node, sequence=seq[node], rssi_dbm=-50)
                seq[node] += 1
                w.write(t + node * 1_000_000, (f"192.168.1.{10 + node}", 5566), ("192.168.1.2", 5566), frame)
    with open(os.path.join(dirpath, "labels.jsonl"), "w") as fh:
        fh.write(json.dumps({"start_ns": T0 + 10**9, "end_ns": T0 + 100 * 10**9, "fine_label": "empty_room"}) + "\n")
        fh.write(json.dumps({"start_ns": T0 + 101 * 10**9, "end_ns": T0 + 120 * 10**9,
                             "fine_label": "occupied_moving", "note": "walking"}) + "\n")


class TestLabels(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp(prefix="p11-labels-")

    def tearDown(self):
        shutil.rmtree(self.dir, ignore_errors=True)

    def _write(self, rows):
        p = os.path.join(self.dir, "labels.jsonl")
        with open(p, "w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
        return p

    def test_refusals(self):
        bad = [
            [{"start_ns": 0, "end_ns": 10, "fine_label": "sleeping"}],
            [{"start_ns": 0.5, "end_ns": 10, "fine_label": "empty_room"}],
            [{"start_ns": 10, "end_ns": 10, "fine_label": "empty_room"}],
            [{"start_ns": 0, "end_ns": 10, "fine_label": "empty_room"},
             {"start_ns": 5, "end_ns": 20, "fine_label": "occupied_static"}],
            [{"start_ns": 0, "end_ns": 10, "fine_label": "empty_room", "count": 1}],
        ]
        for rows in bad:
            with self.assertRaises(ValueError, msg=rows):
                rec.load_labels(self._write(rows))

    def test_defaults_and_order(self):
        p = self._write([{"start_ns": 20, "end_ns": 30, "fine_label": "occupied_static"},
                         {"start_ns": 0, "end_ns": 10, "fine_label": "empty_room"}])
        segs = rec.load_labels(p)
        self.assertEqual([s[2] for s in segs], ["empty_room", "occupied_static"])
        self.assertEqual([s[3] for s in segs], [0, 1])


class TestCache(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.base = tempfile.mkdtemp(prefix="p11-")
        cls.session = os.path.join(cls.base, "room", "2026-11-02_1")
        os.makedirs(cls.session)
        _session(cls.session)
        cls.root = os.path.join(cls.base, "cache")
        cls.meta = rec.build_cache(cls.session, 2.0, root=cls.root)
        cls.c = rec.load_cache(cls.session, 2.0, root=cls.root)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.base, ignore_errors=True)

    def test_label_links_and_windows(self):
        c = self.c
        self.assertEqual(self.meta["label"], "REAL")
        self.assertEqual(c["links"].tolist(), [[-1, 1], [-1, 2]])
        # 99 s and 19 s segments, 2 s windows every 0.5 s
        self.assertEqual(int((c["segment"] == 0).sum()), 195)
        self.assertEqual(int((c["segment"] == 1).sum()), 35)
        self.assertTrue((c["presence"][c["segment"] == 1] == 1).all())
        self.assertTrue(c["moving"][c["segment"] == 1].all())
        ev = c["eval"]
        self.assertTrue((c["t_end_ns"][ev] - 2 * 10**9 >= T0 + 91 * 10**9).all())
        self.assertGreater(int(ev.sum()), 0)
        self.assertEqual(c["amp"].shape[2], 40)

    def test_phase9_tools_read_it(self):
        from ml.train import data as d8
        c = self.c
        ds = rdm.RealDataset([c], [(0, w) for w in range(c["segment"].size)])
        batch = d8.collate([ds[i] for i in range(4)])
        self.assertEqual(tuple(batch["amp"].shape[2:]), (40, 10))
        self.assertFalse(batch["geometry_known"].any())


if __name__ == "__main__":
    unittest.main()
