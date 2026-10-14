"""Scene files: the loader, the checks a schema cannot make, and the
depth/elevation conversion.

Three things are worth saying about what is tested here and why.

First, the JSON Schema in `scenes/scene.schema.json` ships for external
tooling and the loader validates by hand, because this repository has no
install step and `jsonschema` is not available. Two descriptions of one
format drift. `TestSchemaAgreement` reads the schema file and asserts
that its enums, its version constant and its required keys still match
the loader's, so the drift is caught here rather than by a user whose
file passes one and fails the other.

Second, the one collision this format has with the existing code is the
word "height". `localize.geometry.Room.height_m` is the second
HORIZONTAL axis; the scene's `height_m` is elevation. Every example scene
has three different numbers for width, depth and elevation precisely so
that a test of the conversion cannot pass by accident.

Third, the material table is checkable arithmetic, not a magic constant
block: the ITU-R P.2040 coefficients are in the code and the evaluated
values are in `scenes/materials.json`, and `TestMaterials` recomputes one
from the other.

Whole file runs in well under a second.
"""

from __future__ import annotations

import json
import math
import os
import unittest

from wifisense.localize.geometry import Geometry, Node, Room
from wifisense.scene import (
    CONTAIN_TOL_M,
    MATERIAL_REFERENCE_GHZ,
    MATERIALS,
    OBJECT_MATERIALS,
    OBJECT_TYPES,
    SCHEMA_VERSION,
    WALL_MATERIALS,
    Scene,
    SceneError,
    load_scene,
    material_constants,
    parse_scene,
    scenes_dir,
    to_geometry,
    to_nodes,
    to_room,
)

SCENES = scenes_dir()
EXAMPLES = ("bedroom", "office", "hall-l")


def scene_path(name: str) -> str:
    return os.path.join(SCENES, name + ".json")


def scene_text(name: str) -> str:
    with open(scene_path(name), "r", encoding="utf-8") as handle:
        return handle.read()


def load(name: str) -> Scene:
    return load_scene(scene_path(name))


class TestExampleScenes(unittest.TestCase):
    def test_every_example_loads(self):
        for name in EXAMPLES:
            with self.subTest(scene=name):
                scene = load(name)
                self.assertEqual(scene.schema_version, SCHEMA_VERSION)
                self.assertTrue(scene.walls)
                self.assertTrue(scene.objects)
                self.assertGreaterEqual(len(scene.nodes), 4)
                self.assertIsNotNone(scene.phone)

    def test_the_advertised_dimensions(self):
        self.assertEqual(
            [(s.room.width_m, s.room.depth_m, s.room.height_m) for s in map(load, EXAMPLES)],
            [(4.2, 3.6, 2.7), (6.0, 5.0, 2.9), (8.0, 6.0, 3.0)],
        )

    def test_furniture_is_a_plausible_size(self):
        # The kind of thing a scene gets wrong by a factor of ten without
        # anybody noticing, and not a schema's business.
        bedroom = load("bedroom")
        bed = next(o for o in bedroom.objects if o.type == "bed")
        self.assertAlmostEqual(bed.size_m[0], 1.9, delta=0.2)
        self.assertAlmostEqual(bed.size_m[1], 1.4, delta=0.2)
        door = next(o for o in bedroom.objects if o.type == "door")
        self.assertAlmostEqual(door.size_m[0], 0.8, delta=0.15)
        self.assertTrue(1.9 < door.size_m[2] < 2.2, door.size_m[2])

    def test_a_rotated_bed_keeps_its_footprint(self):
        # The bedroom's bed is at 90 degrees, so its plan footprint is
        # 1.4 x 1.9 and not 1.9 x 1.4. A loader that ignored rotation_deg
        # would put a corner at x = -0.2 and still think it was inside.
        bed = next(o for o in load("bedroom").objects if o.type == "bed")
        self.assertEqual(bed.rotation_deg, 90.0)
        xs = [c[0] for c in bed.footprint]
        ys = [c[1] for c in bed.footprint]
        self.assertAlmostEqual(max(xs) - min(xs), 1.4, places=9)
        self.assertAlmostEqual(max(ys) - min(ys), 1.9, places=9)
        self.assertGreater(min(xs), 0.0)

    def test_nodes_are_where_someone_could_mount_them(self):
        for name in EXAMPLES:
            with self.subTest(scene=name):
                scene = load(name)
                for node in scene.nodes:
                    # Chest height, which is the height the 2-D stack
                    # assumes, and the only height that makes dropping z
                    # harmless.
                    self.assertAlmostEqual(node.pos_m[2], 1.2, places=9)
                self.assertEqual(scene.node_z_spread_m, 0.0)

    def test_the_l_shaped_hall_is_smaller_than_its_bounding_box(self):
        hall = load("hall-l")
        self.assertAlmostEqual(hall.room.area_m2, 35.75, places=9)
        self.assertLess(hall.room.area_m2, hall.room.width_m * hall.room.depth_m)
        self.assertFalse(hall.room.contains(7.0, 5.0))  # the missing arm
        self.assertTrue(hall.room.contains(7.0, 1.0))
        self.assertTrue(hall.room.contains(0.0, 0.0))  # a corner is inside

    def test_the_other_two_rooms_are_rectangles(self):
        for name in ("bedroom", "office"):
            with self.subTest(scene=name):
                scene = load(name)
                self.assertEqual(scene.room.polygon_m, ())
                self.assertAlmostEqual(
                    scene.room.area_m2, scene.room.width_m * scene.room.depth_m, places=9
                )


class TestDepthIsNotElevation(unittest.TestCase):
    """The conversion that silently rotates a room when it is wrong."""

    def test_depth_maps_to_the_rooms_second_horizontal_axis(self):
        for name in EXAMPLES:
            with self.subTest(scene=name):
                scene = load(name)
                room = to_room(scene)
                self.assertIsInstance(room, Room)
                self.assertEqual(room.width_m, scene.room.width_m)
                self.assertEqual(room.height_m, scene.room.depth_m)
                # The trap: every example gives width, depth and elevation
                # three different values, so swapping the last two cannot
                # pass this test.
                self.assertNotEqual(scene.room.depth_m, scene.room.height_m)
                self.assertNotEqual(room.height_m, scene.room.height_m)

    def test_the_area_agrees_with_the_bounding_box(self):
        # Room.area_m2 is width x height, and that has to be the scene's
        # bounding box, not width x elevation.
        for name in EXAMPLES:
            with self.subTest(scene=name):
                scene = load(name)
                self.assertAlmostEqual(
                    to_room(scene).area_m2,
                    scene.room.width_m * scene.room.depth_m,
                    places=9,
                )

    def test_nodes_convert_and_keep_their_ids(self):
        for name in EXAMPLES:
            with self.subTest(scene=name):
                scene = load(name)
                nodes = to_nodes(scene)
                self.assertEqual(len(nodes), len(scene.nodes))
                for converted, original in zip(nodes, scene.nodes):
                    self.assertIsInstance(converted, Node)
                    self.assertEqual(converted.node_id, original.id)
                    self.assertEqual(converted.position, original.pos_m[:2])

    def test_geometry_builds_and_every_node_is_in_the_room(self):
        for name in EXAMPLES:
            with self.subTest(scene=name):
                scene = load(name)
                geometry = to_geometry(scene)
                self.assertIsInstance(geometry, Geometry)
                self.assertEqual(geometry.n_nodes, len(scene.nodes))
                self.assertEqual(geometry.n_links, len(scene.nodes) * (len(scene.nodes) - 1))
                for node in geometry.nodes:
                    self.assertTrue(geometry.room.contains(node.x, node.y))

    def test_a_deployment_needs_two_nodes(self):
        raw = json.loads(scene_text("bedroom"))
        raw["nodes"] = raw["nodes"][:1]
        scene = parse_scene(json.dumps(raw))
        with self.assertRaises(SceneError) as caught:
            to_geometry(scene)
        self.assertIn("at least 2 nodes", str(caught.exception))


class TestValidation(unittest.TestCase):
    """What a JSON Schema cannot say."""

    def bedroom_with(self, **changes) -> dict:
        raw = json.loads(scene_text("bedroom"))
        raw.update(changes)
        return raw

    def assert_refused(self, raw: dict, fragment: str) -> str:
        with self.assertRaises(SceneError) as caught:
            parse_scene(json.dumps(raw))
        message = str(caught.exception)
        self.assertIn(fragment, message)
        return message

    def test_a_negative_dimension(self):
        raw = self.bedroom_with()
        raw["room"]["width_m"] = -4.2
        self.assert_refused(raw, "must be positive")

    def test_a_zero_dimension(self):
        raw = self.bedroom_with()
        raw["room"]["height_m"] = 0.0
        self.assert_refused(raw, "must be positive")

    def test_furniture_outside_the_room(self):
        raw = self.bedroom_with()
        almirah = next(o for o in raw["objects"] if o["id"] == "almirah")
        almirah["centre_m"] = [4.5, 1.2, 1.0]
        self.assert_refused(raw, "outside the room")

    def test_furniture_through_the_ceiling(self):
        raw = self.bedroom_with()
        almirah = next(o for o in raw["objects"] if o["id"] == "almirah")
        almirah["centre_m"] = [3.88, 1.2, 2.0]
        self.assert_refused(raw, "in z")

    def test_a_node_outside_the_room(self):
        raw = self.bedroom_with()
        raw["nodes"][1]["pos_m"] = [4.3, 2.4, 1.2]
        self.assert_refused(raw, "outside the room")

    def test_a_node_above_the_ceiling(self):
        raw = self.bedroom_with()
        raw["nodes"][1]["pos_m"] = [4.1, 2.4, 3.5]
        self.assert_refused(raw, "z = 3.500")

    def test_two_nodes_at_one_point(self):
        raw = self.bedroom_with()
        raw["nodes"][1]["pos_m"] = list(raw["nodes"][0]["pos_m"])
        self.assert_refused(raw, "same point")

    def test_two_nodes_a_micron_apart_are_the_same_point(self):
        raw = self.bedroom_with()
        first = raw["nodes"][0]["pos_m"]
        raw["nodes"][1]["pos_m"] = [first[0] + 1e-9, first[1], first[2]]
        self.assert_refused(raw, "same point")

    def test_a_duplicate_node_id(self):
        raw = self.bedroom_with()
        raw["nodes"][1]["id"] = raw["nodes"][0]["id"]
        self.assert_refused(raw, "already used")

    def test_a_node_id_outside_one_byte(self):
        raw = self.bedroom_with()
        raw["nodes"][0]["id"] = 256
        self.assert_refused(raw, "0..255")

    def test_a_node_outside_the_l_but_inside_its_bounding_box(self):
        # (7.0, 5.0) is inside the 8 x 6 box and outside the floor. A
        # loader that checked the box would take it.
        raw = json.loads(scene_text("hall-l"))
        raw["nodes"][0]["pos_m"] = [7.0, 5.0, 1.2]
        self.assert_refused(raw, "outside the room")

    def test_a_polygon_that_disagrees_with_the_room(self):
        raw = json.loads(scene_text("hall-l"))
        raw["room"]["width_m"] = 9.0
        self.assert_refused(raw, "bounding box")

    def test_a_clockwise_polygon(self):
        raw = json.loads(scene_text("hall-l"))
        raw["room"]["polygon_m"] = list(reversed(raw["room"]["polygon_m"]))
        self.assert_refused(raw, "counter-clockwise")

    def test_a_polygon_with_two_vertices(self):
        raw = json.loads(scene_text("hall-l"))
        raw["room"]["polygon_m"] = [[0.0, 0.0], [8.0, 6.0]]
        self.assert_refused(raw, "at least 3 vertices")

    def test_an_unknown_key(self):
        raw = self.bedroom_with(colour="blue")
        self.assert_refused(raw, "unknown key")

    def test_an_unknown_key_inside_a_node(self):
        raw = self.bedroom_with()
        raw["nodes"][0]["mac"] = "aa:bb:cc:dd:ee:ff"
        self.assert_refused(raw, "unknown key")

    def test_a_missing_required_key(self):
        raw = self.bedroom_with()
        del raw["room"]["depth_m"]
        self.assert_refused(raw, "missing depth_m")

    def test_a_future_schema_version_is_refused_not_guessed_at(self):
        raw = self.bedroom_with(schema_version=2)
        self.assert_refused(raw, "Refusing")

    def test_a_material_that_is_not_in_the_table(self):
        raw = self.bedroom_with()
        raw["walls"][0]["material"] = "cheese"
        message = self.assert_refused(raw, "cheese")
        self.assertIn("concrete", message)

    def test_a_wall_material_on_an_object(self):
        # 'concrete' is a wall material, not a furniture material, and the
        # two enums are deliberately different.
        raw = self.bedroom_with()
        raw["objects"][0]["material"] = "concrete"
        self.assert_refused(raw, "concrete")

    def test_an_object_material_on_a_wall(self):
        raw = self.bedroom_with()
        raw["walls"][0]["material"] = "fabric"
        self.assert_refused(raw, "fabric")

    def test_nan_is_refused(self):
        # CPython's json accepts NaN and Infinity as an extension. A room
        # 4.2 m wide and NaN deep would load and then poison every
        # distance computed from it.
        text = scene_text("bedroom").replace('"depth_m": 3.6', '"depth_m": NaN')
        with self.assertRaises(SceneError) as caught:
            parse_scene(text)
        self.assertIn("finite", str(caught.exception))

    def test_a_number_too_large_for_a_float_is_refused(self):
        # 1e400 does not go through parse_constant: it parses as a float
        # and comes out as inf.
        text = scene_text("bedroom").replace('"depth_m": 3.6', '"depth_m": 1e400')
        with self.assertRaises(SceneError) as caught:
            parse_scene(text)
        self.assertIn("finite", str(caught.exception))

    def test_a_boolean_is_not_a_number(self):
        raw = self.bedroom_with()
        raw["room"]["depth_m"] = True
        self.assert_refused(raw, "expected a number")

    def test_broken_json_names_the_file(self):
        with self.assertRaises(SceneError) as caught:
            parse_scene("{oops", source_path="/tmp/x.json")
        self.assertIn("/tmp/x.json", str(caught.exception))

    def test_the_error_carries_the_path_to_the_problem(self):
        raw = self.bedroom_with()
        raw["nodes"][2]["pos_m"] = [1.6, 0.1]
        self.assert_refused(raw, "scene.nodes[2].pos_m")

    def test_a_scene_with_no_nodes_is_still_a_valid_scene(self):
        # A room imported from a scan has no sensors in it yet.
        raw = self.bedroom_with(nodes=[], people=[])
        del raw["phone"]
        scene = parse_scene(json.dumps(raw))
        self.assertEqual(scene.nodes, ())
        self.assertEqual(scene.node_z_spread_m, 0.0)

    def test_a_door_in_the_plane_of_its_wall_is_inside(self):
        # The containment test has to admit the boundary: the bedroom door
        # spans y = 0.0 to 0.06 and z = 0.0 to 2.05, both touching.
        door = next(o for o in load("bedroom").objects if o.type == "door")
        self.assertAlmostEqual(min(c[1] for c in door.footprint), 0.0, places=9)
        self.assertAlmostEqual(door.z_extent_m[0], 0.0, places=9)
        self.assertLess(CONTAIN_TOL_M, 1e-3)


class TestMaterials(unittest.TestCase):
    """ITU-R P.2040 Table 3, as arithmetic rather than as a constant block."""

    def setUp(self):
        with open(os.path.join(SCENES, "materials.json"), "r", encoding="utf-8") as handle:
            self.doc = json.load(handle)

    def test_the_file_and_the_table_hold_the_same_coefficients(self):
        self.assertEqual(self.doc["evaluated_at_ghz"], MATERIAL_REFERENCE_GHZ)
        for name, entry in self.doc["materials"].items():
            with self.subTest(material=name):
                constants = material_constants(name)
                if entry["itu_p2040"] is None:
                    # fabric: no row in the Recommendation, so no
                    # coefficients here either.
                    self.assertIsNone(constants)
                    self.assertEqual(entry["at_2437_mhz"], None)
                    continue
                self.assertIsNotNone(constants)
                for key in ("a", "b", "c", "d"):
                    self.assertEqual(entry["itu_p2040"][key], getattr(constants, key))
                self.assertEqual(
                    tuple(entry["itu_p2040"]["valid_ghz"]), constants.valid_ghz
                )

    def test_the_stored_values_are_what_the_coefficients_give(self):
        # Stored to 9 significant digits, so 1e-8 relative is the tightest
        # honest comparison. This checks that the file was evaluated from
        # these coefficients; it is not a physics tolerance.
        for name, entry in self.doc["materials"].items():
            if entry["at_2437_mhz"] is None:
                continue
            constants = material_constants(name)
            at = entry["at_2437_mhz"]
            f = MATERIAL_REFERENCE_GHZ
            for key, got in (
                ("epsilon_r", constants.epsilon_r(f)),
                ("sigma_s_per_m", constants.sigma_s_per_m(f)),
                ("epsilon_imag", constants.epsilon_imag(f)),
                ("reflection_normal", constants.reflection_normal(f)),
            ):
                with self.subTest(material=name, quantity=key):
                    want = at[key]
                    self.assertLessEqual(
                        abs(want - got), 1e-8 * max(abs(want), abs(got)), f"{name}.{key}"
                    )

    def test_every_material_is_valid_at_the_reference_frequency(self):
        for name, constants in MATERIALS.items():
            with self.subTest(material=name):
                self.assertTrue(constants.in_range(MATERIAL_REFERENCE_GHZ))

    def test_epsilon_imag_is_the_published_conversion(self):
        # eps'' = sigma / (2 pi f eps0), which is 17.975... sigma / f_GHz.
        for name, constants in MATERIALS.items():
            with self.subTest(material=name):
                f = MATERIAL_REFERENCE_GHZ
                expected = 17.975103584522344 * constants.sigma_s_per_m(f) / f
                self.assertAlmostEqual(
                    constants.epsilon_imag(f) / expected, 1.0, places=9
                )

    def test_reflection_is_ordered_the_way_materials_are(self):
        f = MATERIAL_REFERENCE_GHZ
        gamma = {n: m.reflection_normal(f) for n, m in MATERIALS.items()}
        self.assertGreater(gamma["metal"], 0.99)
        self.assertLess(gamma["wood"], gamma["drywall"])
        self.assertLess(gamma["drywall"], gamma["brick"])
        self.assertLess(gamma["brick"], gamma["concrete"])
        self.assertLess(gamma["concrete"], gamma["glass"])
        self.assertLess(gamma["glass"], gamma["metal"])

    def test_concrete_is_near_the_simulators_own_wall_reflection(self):
        # Not a coincidence worth hiding: the simulator's scalar
        # wall_reflection has been 0.4 since before this table existed,
        # and concrete at normal incidence is 0.395.
        gamma = MATERIALS["concrete"].reflection_normal(MATERIAL_REFERENCE_GHZ)
        self.assertAlmostEqual(gamma, 0.395, places=3)

    def test_fabric_has_a_label_and_no_physics(self):
        self.assertIn("fabric", OBJECT_MATERIALS)
        self.assertIsNone(material_constants("fabric"))
        self.assertNotIn("fabric", MATERIALS)
        with self.assertRaises(KeyError):
            material_constants("cheese")

    def test_every_scene_material_is_accounted_for(self):
        for name in WALL_MATERIALS | OBJECT_MATERIALS:
            with self.subTest(material=name):
                # Either it has constants or it is a known gap; never a
                # KeyError, which is what an unlisted material raises.
                material_constants(name)
        # Every wall material has a P.2040 row. Only an object material
        # may be missing one.
        for name in WALL_MATERIALS:
            self.assertIsNotNone(material_constants(name), name)


class TestSchemaAgreement(unittest.TestCase):
    """The schema file ships for other tools; the loader checks by hand."""

    def setUp(self):
        with open(os.path.join(SCENES, "scene.schema.json"), "r", encoding="utf-8") as handle:
            self.schema = json.load(handle)

    def test_the_dialect_is_the_one_claimed(self):
        self.assertEqual(
            self.schema["$schema"], "https://json-schema.org/draft/2020-12/schema"
        )

    def test_the_version_constant_matches(self):
        self.assertEqual(
            self.schema["properties"]["schema_version"]["const"], SCHEMA_VERSION
        )

    def test_the_material_enums_match(self):
        defs = self.schema["$defs"]
        self.assertEqual(set(defs["wall_material"]["enum"]), set(WALL_MATERIALS))
        self.assertEqual(set(defs["object_material"]["enum"]), set(OBJECT_MATERIALS))

    def test_the_object_types_match(self):
        enum = self.schema["$defs"]["object"]["properties"]["type"]["enum"]
        self.assertEqual(set(enum), set(OBJECT_TYPES))

    def test_the_required_keys_match(self):
        defs = self.schema["$defs"]
        self.assertEqual(set(self.schema["required"]), {"schema_version", "room"})
        self.assertEqual(
            set(defs["room"]["required"]), {"width_m", "depth_m", "height_m"}
        )
        self.assertEqual(
            set(defs["wall"]["required"]),
            {"from_m", "to_m", "material", "thickness_m"},
        )
        self.assertEqual(
            set(defs["object"]["required"]), {"type", "centre_m", "size_m", "material"}
        )
        self.assertEqual(set(defs["node"]["required"]), {"id", "pos_m"})

    def test_the_schema_closes_every_object(self):
        # additionalProperties: false is what makes a typo an error rather
        # than a silently ignored key, and the loader does the same.
        self.assertFalse(self.schema["additionalProperties"])
        for name, definition in self.schema["$defs"].items():
            if definition.get("type") == "object":
                with self.subTest(definition=name):
                    self.assertFalse(definition["additionalProperties"])

    def test_the_examples_validate_against_the_schemas_own_enums(self):
        # Not a full schema check — there is no validator here — but the
        # parts most likely to drift.
        defs = self.schema["$defs"]
        for name in EXAMPLES:
            raw = json.loads(scene_text(name))
            with self.subTest(scene=name):
                self.assertEqual(
                    raw["schema_version"],
                    self.schema["properties"]["schema_version"]["const"],
                )
                for wall in raw.get("walls", []):
                    self.assertIn(wall["material"], defs["wall_material"]["enum"])
                for obj in raw.get("objects", []):
                    self.assertIn(obj["material"], defs["object_material"]["enum"])
                    self.assertIn(obj["type"], defs["object"]["properties"]["type"]["enum"])


class TestPlanGeometry(unittest.TestCase):
    """The containment test, against numbers derived in the test."""

    def test_a_point_in_a_square(self):
        square = ((0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0))
        from wifisense.scene import _contains_point

        self.assertTrue(_contains_point(square, 1.0, 1.0, 0.0))
        self.assertFalse(_contains_point(square, 3.0, 1.0, 0.0))
        # The boundary is admitted by the tolerance, not by the ray cast.
        self.assertTrue(_contains_point(square, 0.0, 1.0, 1e-6))
        self.assertFalse(_contains_point(square, -0.1, 1.0, 1e-6))

    def test_the_l_shape_by_hand(self):
        hall = load("hall-l")
        # Worked out from the polygon in the file, not from the loader.
        inside = [(0.5, 0.5), (7.5, 0.5), (4.0, 5.5), (4.4, 2.6)]
        outside = [(5.0, 3.0), (7.9, 5.9), (4.6, 2.6), (-0.1, 1.0)]
        for x, y in inside:
            self.assertTrue(hall.room.contains(x, y), f"({x}, {y}) should be inside")
        for x, y in outside:
            self.assertFalse(hall.room.contains(x, y), f"({x}, {y}) should be outside")

    def test_a_rotation_turns_the_box_counter_clockwise(self):
        from wifisense.scene import _rotated_footprint

        corners = _rotated_footprint(0.0, 0.0, 2.0, 1.0, 90.0)
        # Width along +x becomes width along +y.
        self.assertAlmostEqual(max(c[0] for c in corners), 0.5, places=9)
        self.assertAlmostEqual(max(c[1] for c in corners), 1.0, places=9)
        # The first corner, (-1, -0.5), goes to (0.5, -1).
        self.assertAlmostEqual(corners[0][0], 0.5, places=9)
        self.assertAlmostEqual(corners[0][1], -1.0, places=9)

    def test_the_signed_area_of_a_known_polygon(self):
        from wifisense.scene import _signed_area

        square = ((0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0))
        self.assertAlmostEqual(_signed_area(square), 6.0, places=9)
        self.assertAlmostEqual(_signed_area(tuple(reversed(square))), -6.0, places=9)


class TestRoundTripThroughJson(unittest.TestCase):
    def test_reparsing_an_example_gives_an_equal_scene(self):
        for name in EXAMPLES:
            with self.subTest(scene=name):
                first = parse_scene(scene_text(name))
                second = parse_scene(scene_text(name))
                self.assertEqual(first, second)

    def test_the_source_path_is_not_part_of_the_comparison(self):
        from_text = parse_scene(scene_text("bedroom"))
        from_file = load("bedroom")
        self.assertEqual(from_text, from_file)
        self.assertIsNone(from_text.source_path)
        self.assertTrue(from_file.source_path.endswith("bedroom.json"))


if __name__ == "__main__":
    unittest.main()
