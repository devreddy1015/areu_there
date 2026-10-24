"""Room Sim inputs: the scene port of saans-scene's validator plus the §3.1
rejections, and the §3.2 configuration validation."""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roomsim_helpers import BEDROOM, HALL, MULTI, ROOT, make_scene, rect_walls, scene_dict  # noqa: E402

from wifisense.roomsim import Config, ConfigError, RoomSim, SceneError, load_scene, parse_scene  # noqa: E402


def problems(doc) -> list:
    try:
        parse_scene(json.dumps(doc) if not isinstance(doc, str) else doc)
    except SceneError as e:
        return e.problems
    return []


class TestSceneFiles(unittest.TestCase):
    def test_example_scenes_load(self):
        for name in ("bedroom.json", "office.json", "hall-l-shaped.json"):
            sc = load_scene(os.path.join(ROOT, "spec", "scenes", name))
            self.assertTrue(sc.nodes)
            self.assertEqual(len(sc.sha256), 64)
        sc = load_scene(MULTI)
        self.assertEqual(sc.walls, ())
        self.assertEqual([p.start for p in sc.people], [(1.5, 1.5), (4.0, 3.5), None])

    def test_nodes_sorted_and_no_person_ids_kept(self):
        doc = scene_dict(nodes=[{"id": 3, "pos": [1.0, 1.0, 1.0]}, {"id": 1, "pos": [2.0, 2.0, 1.0]}],
                         people=[{"id": "alice", "start": [3.0, 3.0]}], phone={"id": "p", "start": [1, 1]})
        sc = parse_scene(json.dumps(doc))
        self.assertEqual([n.id for n in sc.nodes], [1, 3])
        self.assertNotIn("alice", repr(sc))


class TestSceneRejections(unittest.TestCase):
    def test_room_sim_rejections(self):
        # a node strictly inside an object's box
        box = {"type": "almirah", "pos": [1.0, 2.5, 1.0], "size": [0.6, 0.6, 2.0], "material": "wood"}
        self.assertTrue(any("strictly inside" in p for p in problems(scene_dict(objects=[box]))))
        # on the box's face is not strictly inside
        box2 = dict(box, pos=[1.3, 2.5, 1.0])
        self.assertEqual(problems(scene_dict(objects=[box2])), [])
        # two nodes at one position
        nodes = [{"id": 0, "pos": [1.0, 2.5, 1.2]}, {"id": 1, "pos": [1.0, 2.5, 1.2]}]
        self.assertTrue(any("same position" in p for p in problems(scene_dict(nodes=nodes))))
        # more than 100 people
        self.assertEqual(problems(scene_dict(people=[{}] * 100)), [])
        self.assertTrue(any("at most 100" in p for p in problems(scene_dict(people=[{}] * 101))))

    def test_validate_rs_checks(self):
        cases = [
            (dict(scene_dict(), schema_version=2), "schema_version"),
            (dict(scene_dict(), name="  "), "name is empty"),
            (scene_dict(w=0.0), "room.w_m"),
            (scene_dict(polygon=[[0, 0], [6, 0]]), "at least 3"),
            (scene_dict(polygon=[[0, 0], [6, 5], [6, 0], [0, 5]]), "cross or touch"),
            (scene_dict(polygon=[[0, 0], [6, 0], [6, 4], [0, 4]]), "bounding box"),
            (scene_dict(walls=[{"from": [0, 0], "to": [0, 0], "material": "brick", "thickness_m": 0.1}]), "zero length"),
            (scene_dict(walls=[{"from": [0, 0], "to": [6, 0], "material": "brick", "thickness_m": 1.5}]), "thickness"),
            (scene_dict(walls=[{"from": [0, 0], "to": [6, 0], "material": "brick", "thickness_m": 0.1,
                                "height_m": 3.0}]), "height_m"),
            (scene_dict(walls=[{"from": [0, 0], "to": [7, 0], "material": "brick", "thickness_m": 0.1}]), "outside the room"),
            (scene_dict(objects=[{"type": "bed", "pos": [5.8, 2.5, 0.25], "size": [1.0, 1.0, 0.5],
                                  "material": "fabric"}]), "leaves the room"),
            (scene_dict(objects=[{"type": "bed", "pos": [3, 2.5, 2.5], "size": [1.0, 1.0, 0.5],
                                  "material": "fabric"}]), "above the"),
            (scene_dict(objects=[{"type": "bed", "pos": [3, 2.5, 0.1], "size": [1.0, 1.0, 0.5],
                                  "material": "fabric"}]), "below the floor"),
            (scene_dict(objects=[{"type": "bed", "pos": [3, 2.5, 0.25], "size": [0.0, 1.0, 0.5],
                                  "material": "fabric"}]), "greater than zero"),
            (scene_dict(nodes=[]), "nodes is empty"),
            (scene_dict(nodes=[{"id": 0, "pos": [1, 1, 1]}, {"id": 0, "pos": [2, 2, 1]}]), "share id"),
            (scene_dict(nodes=[{"id": 0, "pos": [0.005, 1, 1]}]), "outside the room footprint or within"),
            (scene_dict(nodes=[{"id": 0, "pos": [1, 1, 3.0]}]), "height"),
            (scene_dict(people=[{"start": [7.0, 1.0]}]), "people[0].start"),
            (scene_dict(phone={"start": [7.0, 1.0]}), "phone.start"),
        ]
        for doc, needle in cases:
            got = problems(doc)
            self.assertTrue(any(needle in p for p in got), f"{needle!r} not in {got}")

    def test_deserialisation(self):
        base = scene_dict()
        bad = [
            {k: v for k, v in base.items() if k != "provenance"},
            dict(base, provenance="guessed"),
            dict(base, nodes=[{"id": 256, "pos": [1, 1, 1]}]),
            dict(base, nodes=[{"id": 1.0, "pos": [1, 1, 1]}]),
            dict(base, nodes=[{"id": 1, "pos": [1, 1]}]),
            dict(base, walls=None),
            dict(base, objects=[{"type": "lamp", "pos": [1, 1, 1], "size": [1, 1, 1], "material": "wood"}]),
            dict(base, objects=[{"type": "bed", "pos": [3, 2.5, 0.25], "size": [1, 1, 0.5], "material": "stone"}]),
        ]
        for doc in bad:
            self.assertTrue(problems(doc), doc)
        # Every provenance the scene format names is accepted, the
        # generated one (spec/dataset.md) included.
        for provenance in ("surveyed", "roomplan_import", "hand_written", "generated"):
            self.assertEqual(problems(dict(base, provenance=provenance)), [], provenance)
        self.assertTrue(problems('{"schema_version": 1, "schema_version": 1}'))
        self.assertTrue(problems("not json"))
        self.assertTrue(problems('{"x": NaN}'))
        # unknown keys are ignored, as serde does
        self.assertEqual(problems(dict(base, comment="hello")), [])


class TestNodeInWall(unittest.TestCase):
    """§3.1: a node within a wall's thickness, pdist(node.xy, A, B) <
    t/2 + 0.01 with node.z <= h, implied walls included, is an error naming
    `nodes` (and the node and wall)."""

    def run_with(self, nodes, walls=None, **cfg):
        sc = make_scene(nodes=nodes, walls=walls)
        return RoomSim(sc, Config(duration_s=0.0, **cfg))

    def rejects(self, nodes, walls=None, needle="", **cfg):
        with self.assertRaises(ConfigError) as cm:
            self.run_with(nodes, walls, **cfg)
        self.assertEqual(cm.exception.param, "nodes")
        self.assertIn("node 0 ", str(cm.exception))
        self.assertIn(needle, str(cm.exception))

    @staticmethod
    def two(p0):
        return [{"id": 0, "pos": p0}, {"id": 1, "pos": [5.0, 2.5, 1.2]}]

    def test_listed_walls(self):
        walls = rect_walls(6.0, 5.0)            # 0.23 m: the limit is 0.115 + 0.01
        self.rejects(self.two([0.12, 2.5, 1.2]), walls, "walls[3]")
        self.run_with(self.two([0.13, 2.5, 1.2]), walls)
        self.rejects(self.two([2.0, 4.88, 1.2]), walls, "walls[2]")
        # above a short wall the node is clear of it; at its top it is not
        short = walls[:3] + [dict(walls[3], height_m=1.0)]
        self.run_with(self.two([0.12, 2.5, 1.2]), short)
        self.rejects(self.two([0.12, 2.5, 1.0]), short, "walls[3]")

    def test_interior_wall_and_its_ends(self):
        part = {"from": [3.0, 1.0], "to": [3.0, 4.0], "material": "brick", "thickness_m": 0.2}
        walls = rect_walls(6.0, 5.0) + [part]   # the limit is 0.1 + 0.01
        self.rejects(self.two([3.105, 2.5, 1.2]), walls, "walls[4]")
        self.rejects(self.two([2.9, 2.5, 1.2]), walls, "walls[4]")
        self.run_with(self.two([2.88, 2.5, 1.2]), walls)
        self.rejects(self.two([3.0, 4.05, 1.2]), walls, "walls[4]")   # past the end: distance to the end
        self.run_with(self.two([3.0, 4.2, 1.2]), walls)

    def test_implied_walls(self):
        # drywall 0.1 m by default: the limit is 0.05 + 0.01
        self.rejects(self.two([0.05, 2.5, 1.2]), None, "implied wall 3")
        self.run_with(self.two([0.07, 2.5, 1.2]))
        self.run_with(self.two([0.05, 2.5, 1.2]), implied_wall_thickness_m=0.02)
        self.rejects(self.two([0.07, 2.5, 1.2]), None, "implied wall 3", implied_wall_thickness_m=0.2)


class TestConfigValidation(unittest.TestCase):
    def setUp(self):
        self.scene = load_scene(BEDROOM)

    def rejects(self, param, **kw):
        with self.assertRaises(ConfigError) as cm:
            RoomSim(self.scene, Config(duration_s=0.0, **kw))
        self.assertEqual(cm.exception.param, param, str(cm.exception))
        self.assertIn(param, str(cm.exception))

    def test_defaults(self):
        c = Config()
        self.assertEqual((c.seed, c.duration_s, c.channel, c.spacing_hz), (0, 10.0, 6, 312500.0))
        self.assertEqual(c.reflection_order, 2)
        self.assertFalse(c.smoothing)
        self.assertEqual(c.smoothing_edge, (2.0 / 3.0, 1.0 / 3.0))
        self.assertEqual(c.fw_subcarriers, 56)
        self.assertEqual(c.non_default(), {})
        self.assertEqual(Config.from_dict({"tx_period": {"1": 2}, "height_m": [1.6, 1.7]}).non_default(),
                         {"tx_period": {1: 2}, "height_m": (1.6, 1.7)})
        RoomSim(self.scene, Config(duration_s=0.0))

    def test_rejections(self):
        self.rejects("channel", channel=0)
        self.rejects("channel", channel=14)
        self.rejects("fw_subcarriers", fw_subcarriers=7)
        self.rejects("fw_subcarriers", fw_subcarriers=65)
        self.rejects("noise_floor_dbm", noise_floor_dbm=(-128.0, -92.0))
        self.rejects("noise_floor_dbm", noise_floor_dbm=(-97.0, 0.0))
        self.rejects("noise_floor_dbm", noise_floor_dbm=(-90.0, -95.0))
        self.rejects("tx_period", tx_period={1: 0})
        self.rejects("tx_period", tx_period={1: 1.5})
        self.rejects("tx_period", tx_period={9: 2})
        self.rejects("roles", roles={9: "tx"})
        self.rejects("roles", roles={0: "listener"})
        self.rejects("roles", roles={0: "rx", 1: "rx", 2: "rx", 3: "rx"})
        self.rejects("roles", roles={0: "tx", 1: "tx", 2: "tx", 3: "tx"})
        self.rejects("body_points", body_points=5)
        self.rejects("body_points", body_points=17)
        self.rejects("n_people", n_people=101)
        self.rejects("height_m", height_m=(1.9, 1.6))
        self.rejects("speed_mps", speed_mps=(1.0, 0.5))
        self.rejects("reflection_order", reflection_order=3)
        self.rejects("loss_prob_link", loss_prob_link={"0,9": 0.5})
        # A key is exactly f"{tx},{rx}": one that would never be looked up is an error.
        self.rejects("loss_prob_link", loss_prob_link={"0, 1": 0.5})
        self.rejects("loss_prob_link", loss_prob_link={"00,1": 0.5})
        self.rejects("loss_prob_link", loss_prob_link={"-0,1": 0.5})
        # Slab surfaces are wall materials; metal is for objects (§5.1).
        self.rejects("floor_material", floor_material="metal")
        self.rejects("ceiling_material", ceiling_material="metal")
        self.rejects("implied_wall_material", implied_wall_material="metal")
        self.rejects("start_heading_rad", start_heading_rad={1: 0.5})
        self.rejects("people_posture", people_posture={0: "sit", 3: "sit"})
        self.rejects("people_mode", people_mode={0: "walk"})  # the sleeper's start is not walkable

    def test_a_run_needs_a_link(self):
        sc = make_scene(nodes=[{"id": 0, "pos": [1.0, 2.5, 1.2]}])
        with self.assertRaises(ConfigError) as cm:
            RoomSim(sc, Config(duration_s=0.0))
        self.assertEqual(cm.exception.param, "roles")
        self.assertIn("one-node", str(cm.exception))
        for roles in ({0: "rx", 1: "rx", 2: "rx", 3: "rx"}, {0: "tx", 1: "tx", 2: "tx", 3: "tx"}):
            self.rejects("roles", roles=roles)
        # one transmitter and one other receiver is enough, either way round
        sim = RoomSim(self.scene, Config(duration_s=0.0, roles={0: "txrx", 1: "tx", 2: "tx", 3: "tx"}))
        self.assertEqual(sim.links, [(1, 0), (2, 0), (3, 0)])
        sc2 = make_scene()
        self.assertEqual(RoomSim(sc2, Config(duration_s=0.0, roles={0: "txrx", 1: "rx"})).links, [(0, 1)])

    def test_integer_map_keys_are_canonical_decimal(self):
        good = {"roles": {"0": "tx", "1": "rx", "2": "rx", "3": "rx"}, "tx_period": {"0": 2, "3": 1},
                "people_mode": {"0": "stationary"}, "people_posture": {"0": "lie"},
                "start_heading_rad": {"0": 1}}
        cfg = Config.from_dict(dict(good, duration_s=0.0))
        self.assertEqual(cfg.roles, {0: "tx", 1: "rx", 2: "rx", 3: "rx"})
        self.assertEqual(cfg.tx_period, {0: 2, 3: 1})
        RoomSim(self.scene, cfg)
        for name, value in (("roles", "rx"), ("tx_period", 2), ("people_mode", "stationary"),
                            ("people_posture", "lie"), ("start_heading_rad", 0.5)):
            for key in ("01", "00", "+1", "-1", " 1", "1 ", "1_0", "1.0", "", "\u0661", "0x1", "1e0"):
                with self.assertRaises(ConfigError, msg=f"{name} {key!r}") as cm:
                    Config.from_dict({name: {key: value}})
                self.assertEqual(cm.exception.param, name)
            # a Config built directly with a string key is rejected by validate()
            with self.assertRaises(ConfigError) as cm:
                RoomSim(self.scene, Config(duration_s=0.0, **{name: {"0": value}}))
            self.assertEqual(cm.exception.param, name)
        # "10" is canonical (and then not a node of this scene)
        self.assertEqual(Config.from_dict({"tx_period": {"10": 2}}).tx_period, {10: 2})

    def test_from_dict_coerces_binary64_fields_only(self):
        cfg = Config.from_dict({
            "duration_s": 1, "tx_power_dbm": 10, "slot_s": 1, "ripple_period_bins": 64, "tail_decay_s": 1,
            "height_m": [1, 2], "rssi_range_dbm": [-100, 0], "phone_offset_m": [0, 0, 1],
            "smoothing_kernel": [0, 1, 0], "start_heading_rad": {"0": 1}, "loss_prob_link": {"0,1": 1},
            "seed": 3, "tail_count": 7, "channel": 6, "fw_subcarriers": 56, "tx_period": {"1": 2},
        })
        for name in ("duration_s", "tx_power_dbm", "slot_s", "ripple_period_bins", "tail_decay_s"):
            self.assertIs(type(getattr(cfg, name)), float, name)
        for name in ("height_m", "rssi_range_dbm", "phone_offset_m", "smoothing_kernel"):
            self.assertTrue(all(type(x) is float for x in getattr(cfg, name)), name)
        self.assertIs(type(cfg.start_heading_rad[0]), float)
        self.assertIs(type(cfg.loss_prob_link["0,1"]), float)
        for name in ("seed", "tail_count", "channel", "fw_subcarriers"):
            self.assertIs(type(getattr(cfg, name)), int, name)
        self.assertIs(type(cfg.tx_period[1]), int)
        # bools stay bools (and are rejected); integer fields keep floats (and reject them)
        self.assertIs(Config.from_dict({"duration_s": True}).duration_s, True)
        self.assertIs(Config.from_dict({"height_m": [True, 2]}).height_m[0], True)
        for d, param in (({"duration_s": True}, "duration_s"), ({"height_m": [True, 2.0]}, "height_m"),
                         ({"tail_count": 3.0}, "tail_count"), ({"seed": 1.0}, "seed"),
                         ({"tx_period": {"1": 2.0}}, "tx_period")):
            with self.assertRaises(ConfigError) as cm:
                RoomSim(self.scene, Config.from_dict(dict(d, duration_s=d.get("duration_s", 0.0))))
            self.assertEqual(cm.exception.param, param)
        with self.assertRaises(ConfigError) as cm:
            Config.from_dict({"tx_power_dbm": 10 ** 400})
        self.assertEqual(cm.exception.param, "tx_power_dbm")

    def test_decibel_and_scale_bounds(self):
        def up(x):
            return math.nextafter(x, math.inf)

        def down(x):
            return math.nextafter(x, -math.inf)

        bad = [
            ("tx_power_dbm", 60.5), ("tx_power_dbm", -100.5), ("tail_power_db", 30.5), ("tail_power_db", -100.5),
            ("agc_step_db", -0.1), ("agc_step_db", 60.5), ("agc_max_db", -0.1), ("agc_max_db", 60.5),
            ("agc_initial_db", 60.5), ("agc_initial_db", -60.5),
            ("device_gain_db", (-60.5, 0.0)), ("device_gain_db", (0.0, 60.5)),
            ("rssi_jitter_db", -0.1), ("rssi_jitter_db", 60.5),
            ("ripple_amp", -0.01), ("ripple_phase_rad", -0.01), ("timing_offset_s", -1e-9),
            ("chip_noise_counts", -0.1), ("target_counts", 0.0), ("target_counts", -1.0),
            ("ripple_period_bins", 0.0), ("ripple_period_bins", 0.5), ("ripple_period_bins", -0.999),
            # the upper bounds that keep every formula finite, one ulp outside each end
            ("ripple_amp", up(1.0)), ("ripple_amp", down(0.0)),
            ("timing_offset_s", up(1e-3)), ("timing_offset_s", down(0.0)),
            ("timing_offset_dev_s", (down(-1e-3), 0.0)), ("timing_offset_dev_s", (0.0, up(1e-3))),
            ("chip_noise_counts", up(1000.0)), ("chip_noise_counts", down(0.0)),
            ("target_counts", up(1000.0)), ("target_counts", 1e300),
            ("breathing_p2p_m", (down(0.0), 0.01)), ("breathing_p2p_m", (0.004, up(1.0))),
            ("height_m", (down(0.0), 1.8)), ("height_m", (1.6, up(3.0))),
        ]
        for name, v in bad:
            self.rejects(name, **{name: v})
        good = [
            ("tx_power_dbm", 60.0), ("tx_power_dbm", -100.0), ("tail_power_db", 30.0), ("tail_power_db", -100.0),
            ("agc_step_db", 0.0), ("agc_max_db", 60.0), ("agc_initial_db", -60.0), ("device_gain_db", (-60.0, 60.0)),
            ("rssi_jitter_db", 0.0), ("ripple_amp", 0.0), ("ripple_phase_rad", 0.0), ("timing_offset_s", 0.0),
            ("chip_noise_counts", 0.0), ("target_counts", 1e-3), ("ripple_period_bins", 1.0),
            ("ripple_period_bins", -1.0), ("ripple_period_bins", -64.0),
            ("ripple_amp", 1.0), ("timing_offset_s", 1e-3), ("timing_offset_dev_s", (-1e-3, 1e-3)),
            ("chip_noise_counts", 1000.0), ("target_counts", 1000.0), ("target_counts", 5e-324),
            ("breathing_p2p_m", (0.0, 1.0)), ("height_m", (0.0, 3.0)),
        ]
        for name, v in good:
            RoomSim(self.scene, Config(duration_s=0.0, **{name: v}))

    def test_runs_stay_finite_at_the_bounds(self):
        # Every accepted extreme runs to datagrams: no inf or NaN reaches the int8 cast.
        for kw in (dict(ripple_amp=1.0, timing_offset_s=1e-3, timing_offset_dev_s=(-1e-3, 1e-3),
                        chip_noise_counts=1000.0, target_counts=1000.0, breathing_p2p_m=(1.0, 1.0),
                        height_m=(3.0, 3.0)),
                   dict(chip_noise_counts=0.0, target_counts=5e-324, breathing_p2p_m=(0.0, 0.0),
                        height_m=(0.0, 0.0))):
            got = [p for p in RoomSim(self.scene, Config(duration_s=0.1, **kw)).packets() if not p.lost]
            self.assertTrue(got)
            for p in got:
                self.assertTrue(math.isfinite(p.rssi_true), kw)
                self.assertTrue(all(math.isfinite(v) for re_im in p.buffer_pre for v in re_im), kw)
                self.assertTrue(p.datagram, kw)

    def test_node_id_255_is_rejected(self):
        sc = make_scene(nodes=[{"id": 0, "pos": [1.0, 2.5, 1.2]}, {"id": 255, "pos": [5.0, 2.5, 1.2]}])
        with self.assertRaises(ConfigError) as cm:
            RoomSim(sc, Config(duration_s=0.0))
        self.assertIn("0..254", str(cm.exception))

    def test_lie_needs_a_bed(self):
        sc = load_scene(HALL)
        with self.assertRaises(ConfigError) as cm:
            RoomSim(sc, Config(duration_s=0.0, people_mode={1: "stationary"}, people_posture={1: "lie"}))
        self.assertEqual(cm.exception.param, "people_posture")


if __name__ == "__main__":
    unittest.main()
