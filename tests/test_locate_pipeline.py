"""spec/locate.md §4.3, §10, §11: Locate on Room Sim runs.

The refusals (each naming its parameter), the glue from Room Sim packets
to `CsiFrame`s, the tick clock, the calibration run's construction, and a
short end-to-end run on the bedroom checking the records' structure, their
labels, `radius_calibrated` false, and that no scene id appears anywhere.
Room Sim runs here use a cheap physics configuration (no reflections,
tail, clutter or object paths) to stay fast; that changes no rule under
test. Everything is SIM.
"""

from __future__ import annotations

import json
import math
import os
import unittest

from wifisense.localize.config import LocateConfig, LocateError
from wifisense.localize.geometry import Geometry
from wifisense.localize.measure import RssiSignal
from wifisense.localize.pipeline import (
    PersonPipeline, base_rate_hz, calibration_config, check_tick_hz, check_tx_period, frame_from_packet,
    locate_geometry, next_tick, phone_boxes, phone_walls, run_locate, tick_time,
)
from wifisense.localize.radius import radius95
from wifisense.roomsim import Config, RoomSim, load_scene, parse_scene
from wifisense.types import CsiFrame

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
BEDROOM = os.path.join(ROOT, "spec", "scenes", "bedroom.json")

CHEAP = dict(reflection_order=0, floor_ceiling=False, clutter_count=0, tail=False,
             object_specular=False, object_scatter=False, body_points=6)


def small_scene(nodes=None, phone=True):
    doc = {
        "schema_version": 1, "name": "test-room", "provenance": "hand_written",
        "room": {"w_m": 4.0, "d_m": 3.0, "h_m": 2.7},
        "nodes": nodes or [
            {"id": 0, "pos": [0.3, 0.3, 1.2]}, {"id": 1, "pos": [3.7, 0.3, 1.2]},
            {"id": 2, "pos": [3.7, 2.7, 1.2]}, {"id": 3, "pos": [0.3, 2.7, 1.2]},
        ],
        "people": [{"id": "walker-x", "start": [2.0, 1.5]}],
    }
    if phone:
        doc["phone"] = {"id": "phone-x", "start": [2.0, 1.5]}
    return parse_scene(json.dumps(doc))


class TestRefusals(unittest.TestCase):
    def assertRefused(self, param, scene, rs_cfg, cfg=None, label="SIM"):
        with self.assertRaises(LocateError) as cm:
            run_locate(scene, rs_cfg, cfg, label)
        self.assertEqual(cm.exception.param, param)
        self.assertTrue(str(cm.exception).startswith(param + ":"))
        for word in ("walker-x", "phone-x"):
            self.assertNotIn(word, str(cm.exception))
        return cm.exception

    def test_roles(self):
        self.assertRefused("roles", small_scene(), Config(duration_s=1.0, roles={0: "txrx"}))
        self.assertRefused("roles", small_scene(), Config(duration_s=1.0, roles={1: "rx"}))

    def test_stacked_nodes(self):
        nodes = [{"id": 0, "pos": [0.3, 0.3, 0.5]}, {"id": 1, "pos": [0.3, 0.3, 2.0]},
                 {"id": 2, "pos": [3.7, 2.7, 1.2]}]
        self.assertRefused("nodes", small_scene(nodes), Config(duration_s=1.0))

    def test_quantize(self):
        self.assertRefused("quantize", small_scene(), Config(duration_s=1.0, quantize=False))

    def test_tx_period(self):
        # 4 transmitters at 12.5 ms: 20 Hz per link, min_samples 10.
        # tx_period 2: floor(10.0) + 1 = 11 >= 10, accepted; 3: 7 < 10, refused.
        check_tx_period([0, 1], {0: 2}, 1.0, 20.0, 10)
        with self.assertRaises(LocateError) as cm:
            check_tx_period([0, 1], {1: 3}, 1.0, 20.0, 10)
        self.assertEqual(cm.exception.param, "tx_period")
        self.assertRefused("tx_period", small_scene(), Config(duration_s=1.0, tx_period={2: 3}, **CHEAP))

    def test_calibration_too_short(self):
        self.assertRefused("calib_s", small_scene(), Config(duration_s=0.5, **CHEAP),
                           LocateConfig(calib_s=3.0))

    def test_config_and_label(self):
        self.assertRefused("window_s", small_scene(), Config(duration_s=1.0), LocateConfig(window_s=NAN))
        self.assertRefused("label", small_scene(), Config(duration_s=1.0), label="LIVE")

    def test_phone_refuses_a_non_sim_label(self):
        self.assertRefused("label", small_scene(), Config(duration_s=0.5, **CHEAP),
                           LocateConfig(calib_s=11.0), label="REAL")

    def test_room_sim_run_is_only_sim(self):
        """§1: Room Sim input is SIM, so a run without a phone refuses every
        other label too, without echoing it."""
        for label in ("REAL", "SCENE", "SIM->REAL", "sim", None, 3):
            with self.subTest(label=label):
                err = self.assertRefused("label", small_scene(phone=False), Config(duration_s=0.5, **CHEAP),
                                         LocateConfig(calib_s=11.0), label=label)
                self.assertNotIn(repr(label), str(err))

    def test_set_room_sim_values_must_match(self):
        """§9: phone_seed, tx_power_dbm and carrier_hz, when set, must be
        Room Sim's."""
        rs = Config(duration_s=0.5, seed=4, **CHEAP)
        self.assertRefused("phone_seed", small_scene(), rs, LocateConfig(phone_seed=5))
        self.assertRefused("tx_power_dbm", small_scene(), rs, LocateConfig(tx_power_dbm=12.0))
        self.assertRefused("carrier_hz", small_scene(), rs, LocateConfig(carrier_hz=2412e6))

    def test_tick_hz_at_most_one_per_slot(self):
        """§9: tick_hz <= 1.0 / slot_s on a Room Sim run (80 Hz here)."""
        self.assertRefused("tick_hz", small_scene(phone=False), Config(duration_s=0.5, **CHEAP),
                           LocateConfig(tick_hz=1e300))
        self.assertRefused("tick_hz", small_scene(phone=False), Config(duration_s=0.5, **CHEAP),
                           LocateConfig(tick_hz=80.00000000000001))
        self.assertRefused("tick_hz", small_scene(phone=False), Config(duration_s=0.5, slot_s=0.025, **CHEAP),
                           LocateConfig(tick_hz=41.0))
        check_tick_hz(80.0, 0.0125)
        with self.assertRaises(LocateError) as cm:
            check_tick_hz(NAN, 0.0125)
        self.assertEqual(cm.exception.param, "tick_hz")

    def test_one_tick_per_slot_runs(self):
        out = run_locate(small_scene(phone=False), Config(duration_s=1.25, **CHEAP),
                         LocateConfig(calib_s=11.0, tick_hz=80.0))
        ts = [r["t"] for r in out.ticks]
        self.assertEqual(ts, [1.0 + k / 80.0 for k in range(20)])
        self.assertTrue(all(b > a for a, b in zip(ts, ts[1:])))


NAN = float("nan")


class TestGlue(unittest.TestCase):
    def test_rate_and_ticks(self):
        self.assertEqual(base_rate_hz(4, 0.0125), 1.0 / (4.0 * 0.0125))
        ticks = []
        k = 0
        while tick_time(k, 1.0, 2.0) < 3.0:
            ticks.append(tick_time(k, 1.0, 2.0))
            k += 1
        self.assertEqual(ticks, [1.0, 1.5, 2.0, 2.5])
        # Computed, never accumulated.
        self.assertEqual(tick_time(10, 1.0, 3.0), 1.0 + 10.0 / 3.0)
        with self.assertRaises(LocateError):
            base_rate_hz(0, 0.0125)

    def test_a_tick_that_does_not_advance_is_refused(self):
        """§10.1: `not (t_{k+1} > t_k)` refuses the run naming tick_hz."""
        self.assertEqual(next_tick(0, 1.0, 1.0, 2.0), (1, 1.5))
        self.assertEqual(next_tick(9, tick_time(9, 1.0, 3.0), 1.0, 3.0), (10, 1.0 + 10.0 / 3.0))
        for k, window_s, tick_hz in ((0, 1.0, 1e300), (0, 1.0, 1e17), (0, 1e17, 2.0), (5, 1.0, NAN)):
            with self.subTest(window_s=window_s, tick_hz=tick_hz):
                with self.assertRaises(LocateError) as cm:
                    next_tick(k, tick_time(k, window_s, tick_hz), window_s, tick_hz)
                self.assertEqual(cm.exception.param, "tick_hz")

    def test_frame_from_packet(self):
        sim = RoomSim(small_scene(), Config(duration_s=0.05, **CHEAP))
        pkts = [p for tr in sim.transmissions() for p in tr.packets if not p.lost]
        self.assertTrue(pkts)
        for pkt in pkts[:6]:
            fr = frame_from_packet(pkt, 56, 20.0)
            self.assertEqual(fr.n_subcarriers, 56)
            for k in range(56):
                re, im = pkt.buffer[k]
                self.assertEqual(fr.amplitudes[k], math.sqrt(float(re * re + im * im)))
            self.assertEqual(fr.phases, [0.0] * 56)
            self.assertEqual((fr.sample_index, fr.sample_rate_hz, fr.rssi_dbm, fr.node_id),
                             (pkt.seq, 20.0, pkt.rssi, pkt.rx))
            self.assertEqual(RssiSignal().new_state(20.0).push(fr), float(pkt.rssi))

    def test_calibration_config(self):
        rs = Config(seed=9, duration_s=6.0, people_mode={0: "walk"}, people_posture={0: "stand"},
                    start_heading_rad={0: 1.0})
        c = calibration_config(rs, 15.0)
        self.assertEqual((c.seed, c.n_people, c.people_mode, c.people_posture, c.start_heading_rad),
                         (9, 0, {}, {}, {}))
        self.assertEqual(c.duration_s, 21.0)
        self.assertEqual(rs.duration_s, 6.0)

    def test_geometry_and_blockers(self):
        scene = load_scene(BEDROOM)
        g = locate_geometry(scene, LocateConfig())
        self.assertEqual((g.room.width_m, g.room.height_m), (4.2, 3.6))
        self.assertEqual(g.n_links, 12)
        sim = RoomSim(scene, Config(duration_s=0.0))
        self.assertEqual(len(phone_walls(sim.world)), 4)          # implied walls
        kinds = [o.kind for o in sim.world.objects]
        self.assertEqual(len(phone_boxes(sim.world)), sum(1 for k in kinds if k not in ("door", "window")))


class TestPersonTick(unittest.TestCase):
    """§10.1 on synthetic RSSI: the links through one point are stirred."""

    @staticmethod
    def pipeline(label="SIM", **kw):
        g = Geometry.perimeter(8)
        pp = PersonPipeline(g, LocateConfig(**kw), 20.0, label)
        n = g.n_links
        pp.meas.set_baseline([0.0] * n, [0.1] * n)
        p = (3.0, 2.5)
        for i in range(21):
            t = 0.05 * i
            for lk in g.links:
                hit = lk.excess_path_m(*p) < g.lambda_m
                r = (-40 if i % 2 else -46) if hit else -40
                pp.meas.update(t, lk.tx_id, lk.rx_id, CsiFrame([1.0] * 4, [0.0] * 4, i, 20.0, r, lk.rx_id))
        return pp

    def test_wlm_and_rti_estimates(self):
        pp = self.pipeline()
        wlm, rti = pp.estimates(1.0)
        self.assertNotIn("reason", wlm)
        self.assertNotIn("reason", rti)
        self.assertIsNone(wlm["confidence"])
        self.assertGreater(rti["confidence"], 0.0)
        est = pp.tracker
        self.assertEqual(rti["cov_xx"], est._cov[0][0])
        self.assertEqual(rti["radius_m"], radius95(rti["cov_xx"], rti["cov_xy"], rti["cov_yy"]))
        self.assertLess(math.sqrt((rti["x"] - 3.0) ** 2 + (rti["y"] - 2.5) ** 2), 1.0)
        rec = pp.tick(1.0, None)
        self.assertEqual(rec["truth"], {"people": []})

    def test_radius_scale_wiring(self):
        """Each `*_radius_scale` scales only its own estimator (swapping
        any two would fail here)."""
        base_w, base_r = self.pipeline().estimates(1.0)
        for kw in ({"wlm_radius_scale": 1.5}, {"rti_radius_scale": 2.0}, {"phone_radius_scale": 0.75},
                   {"wlm_radius_scale": 1.5, "rti_radius_scale": 2.0, "phone_radius_scale": 0.75},
                   {"wlm_radius_scale": 2.0, "rti_radius_scale": 1.5, "phone_radius_scale": 3.0}):
            with self.subTest(**kw):
                wlm, rti = self.pipeline(**kw).estimates(1.0)
                sw, sr = kw.get("wlm_radius_scale", 1.0), kw.get("rti_radius_scale", 1.0)
                self.assertEqual(wlm["radius_m"], sw * radius95(wlm["cov_xx"], wlm["cov_xy"], wlm["cov_yy"]))
                self.assertEqual(rti["radius_m"], sr * radius95(rti["cov_xx"], rti["cov_xy"], rti["cov_yy"]))
                self.assertEqual(wlm["radius_m"], sw * base_w["radius_m"])
                self.assertEqual(rti["radius_m"], sr * base_r["radius_m"])
                self.assertEqual((wlm["x"], rti["x"], rti["cov_xx"]), (base_w["x"], base_r["x"], base_r["cov_xx"]))

    def test_person_labels(self):
        """§1: a person pipeline fed by a caller takes SIM, REAL or
        SIM->REAL; SCENE labels geometry, never an estimate."""
        for label in ("SIM", "REAL", "SIM->REAL"):
            self.assertEqual(self.pipeline(label).tick(1.0, None)["label"], label)
        for label in ("SCENE", "LIVE", "sim", None):
            with self.subTest(label=label):
                with self.assertRaises(LocateError) as cm:
                    PersonPipeline(Geometry.perimeter(8), LocateConfig(), 20.0, label)
                self.assertEqual(cm.exception.param, "label")
                self.assertNotIn(repr(label), str(cm.exception))

    def test_implausible_radius_keeps_the_corrected_tracker(self):
        pp = self.pipeline(min_plausible_radius_m=100.0)
        wlm, rti = pp.estimates(1.0)
        self.assertEqual(wlm, {"reason": "implausible_radius"})
        self.assertEqual(rti, {"reason": "implausible_radius"})
        self.assertTrue(pp.tracker.is_tracking)

    def test_quiet_room_abstains(self):
        g = Geometry.perimeter(8)
        pp = PersonPipeline(g, LocateConfig(), 20.0)
        pp.meas.set_baseline([0.0] * g.n_links, [0.1] * g.n_links)
        wlm, rti = pp.estimates(1.0)
        self.assertEqual((wlm, rti), ({"reason": "no_disturbance"}, {"reason": "not_significant"}))


class TestBedroomRun(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.scene = load_scene(BEDROOM)
        cls.out = run_locate(cls.scene, Config(duration_s=6.0, seed=3, **CHEAP), LocateConfig(calib_s=12.0))

    def test_ticks(self):
        ticks = self.out.ticks
        self.assertEqual([r["t"] for r in ticks], [1.0 + k / 2.0 for k in range(10)])
        for r in ticks:
            self.assertEqual(list(r), ["label", "t", "wlm", "rti", "truth"])
            self.assertEqual(r["label"], "SIM")
            self.assertEqual(list(r["truth"]), ["people"])
            self.assertEqual(len(r["truth"]["people"]), 1)
            self.assertEqual(list(r["truth"]["people"][0]), ["x", "y", "walking", "posture"])
            self.assertIn(r["wlm"].get("reason", "estimate"), ("estimate", "no_disturbance", "implausible_radius"))
            self.assertIn(r["rti"].get("reason", "estimate"), ("estimate", "not_significant", "implausible_radius"))

    def test_epochs(self):
        epochs = self.out.epochs
        self.assertEqual(len(epochs), 30)                        # 5 Hz for 6 s
        for e, r in enumerate(epochs):
            self.assertEqual(list(r), ["label", "t", "rtt", "rssi", "reports", "truth"])
            self.assertEqual(r["label"], "SIM")
            self.assertGreaterEqual(r["t"], e / 5.0)
            self.assertEqual([x["node"] for x in r["reports"]], [0, 1, 2, 3])
            self.assertEqual(list(r["truth"]), ["x", "y", "z"])

    def test_every_estimate_has_a_plausible_radius(self):
        n = 0
        for rec, keys in [(t, ("wlm", "rti")) for t in self.out.ticks] + \
                         [(e, ("rtt", "rssi")) for e in self.out.epochs]:
            for k in keys:
                est = rec[k]
                if "reason" in est:
                    self.assertEqual(list(est), ["reason"])
                    continue
                n += 1
                self.assertEqual(list(est), ["x", "y", "radius_m", "cov_xx", "cov_xy", "cov_yy",
                                             "confidence", "radius_calibrated"])
                self.assertIs(est["radius_calibrated"], False)
                self.assertTrue(math.isfinite(est["radius_m"]))
                self.assertGreaterEqual(est["radius_m"], 0.3)
        self.assertGreater(n, 0)

    def test_no_scene_ids_anywhere(self):
        text = json.dumps({"ticks": self.out.ticks, "epochs": self.out.epochs,
                           "diagnostics": self.out.diagnostics})
        with open(BEDROOM, encoding="utf-8") as fh:
            doc = json.load(fh)
        ids = [p["id"] for p in doc["people"]] + [doc["phone"]["id"]]
        self.assertEqual(ids, ["sleeper", "phone"])
        for word in ids:
            self.assertNotIn(json.dumps(word), text)
        self.assertNotIn("sleeper", text)

    def test_diagnostics(self):
        d = self.out.diagnostics
        self.assertEqual(d["label"], "SIM")
        self.assertEqual(d["link_signal"], "rssi_dbm")
        self.assertEqual(d["rate_hz"], 20.0)
        self.assertEqual(d["min_samples"], 10)
        self.assertEqual(len(d["calibrated_mask"]), 12)


if __name__ == "__main__":
    unittest.main()
