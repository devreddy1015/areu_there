"""The RoomPlan importer stub: its mapping table, and what it admits to.

WHAT THESE TESTS CAN AND CANNOT SHOW. The fixture below is hand-written
to the shape the importer expects. NO REAL ROOMPLAN EXPORT HAS BEEN RUN
THROUGH ANY OF THIS, so a passing test here says the importer is
self-consistent — it maps the categories it says it maps, it reports what
it drops, it produces a scene the loader accepts — and says nothing about
whether the format assumptions in `roomplan_import.UNVERIFIED` are right.
The coordinate test in particular checks the convention the importer
implements, not the convention Apple uses. That distinction is the whole
reason the module is called a stub.

What the tests do defend is the part that is not guesswork: the mapping
table is complete, nothing is dropped silently, and every invented value
is in the report.
"""

from __future__ import annotations

import math
import unittest

from wifisense.roomplan_import import (
    DEFAULT_OPENING_THICKNESS_M,
    DEFAULT_WALL_MATERIAL,
    DEFAULT_WALL_THICKNESS_M,
    OBJECT_CATEGORY_MAP,
    SURFACE_CATEGORY_MAP,
    UNVERIFIED,
    import_roomplan,
)
from wifisense.scene import (
    OBJECT_MATERIALS,
    OBJECT_TYPES,
    SceneError,
    scene_from_mapping,
)


def transform(tx: float, ty: float, tz: float, yaw_deg: float = 0.0) -> list[float]:
    """A column-major 4x4 with a yaw about ARKit's +Y and a translation.

    Written here rather than taken from the importer, so the importer's
    reading of a transform is checked against a matrix built from the
    definition instead of against itself.
    """
    theta = math.radians(yaw_deg)
    c, s = math.cos(theta), math.sin(theta)
    return [
        c, 0.0, -s, 0.0,     # column 0: the local X axis
        0.0, 1.0, 0.0, 0.0,  # column 1: up
        s, 0.0, c, 0.0,      # column 2
        tx, ty, tz, 1.0,     # column 3: the translation
    ]


def surface(ident, category, dims, tx, ty, tz, yaw=0.0):
    return {
        "identifier": ident,
        "category": category,
        "dimensions": list(dims),
        "transform": transform(tx, ty, tz, yaw),
        "confidence": "high",
    }


def fixture() -> dict:
    """A 4 x 3 m room, 2.7 m high, in a scan whose origin is not a corner.

    ARKit x runs 10 to 14, ARKit z runs -5 to -2, so the scene's origin
    has to be moved by (10, 2) and nothing lands at a negative coordinate
    by luck.
    """
    return {
        "identifier": "00000000-0000-0000-0000-000000000000",
        "version": 2,
        "walls": [
            surface("wall-south", "wall", (4.0, 2.7, 0.1), 12.0, 1.35, -2.0),
            surface("wall-north", "wall", (4.0, 2.7, 0.1), 12.0, 1.35, -5.0),
            surface("wall-west", "wall", (3.0, 2.7, 0.1), 10.0, 1.35, -3.5, 90.0),
            surface("wall-east", "wall", (3.0, 2.7, 0.1), 14.0, 1.35, -3.5, 90.0),
        ],
        "doors": [
            # Swift encodes a case with associated values as an object.
            surface("door-1", {"door": {"isOpen": False}}, (0.8, 2.05, 0.05), 11.5, 1.025, -2.0),
        ],
        "windows": [
            surface("window-1", "window", (1.2, 1.3, 0.05), 10.0, 1.5, -3.5, 90.0),
        ],
        "openings": [
            surface("opening-1", "opening", (0.9, 2.1, 0.05), 14.0, 1.05, -3.0, 90.0),
        ],
        "objects": [
            surface("bed-1", "bed", (1.9, 0.55, 1.4), 11.0, 0.275, -3.0),
            surface("table-1", {"table": {}}, (1.2, 0.75, 0.6), 13.0, 0.375, -3.5),
            surface("storage-1", "storage", (1.0, 2.0, 0.6), 11.0, 1.0, -4.7),
            surface("toilet-1", "toilet", (0.4, 0.8, 0.7), 13.5, 0.4, -4.5),
            surface("hammock-1", "hammock", (2.0, 0.4, 0.8), 12.0, 1.0, -4.0),
            {
                "identifier": "broken-1",
                "category": "chair",
                "dimensions": [0.5, 0.9, 0.5],
                "transform": [1.0, 2.0, 3.0],
            },
        ],
    }


class TestMappingTable(unittest.TestCase):
    """The part that is not guesswork."""

    def test_every_mapped_object_type_is_a_real_scene_type(self):
        for category, mapped in OBJECT_CATEGORY_MAP.items():
            with self.subTest(category=category):
                if mapped is not None:
                    self.assertIn(mapped, OBJECT_TYPES)

    def test_every_mapped_surface_is_a_wall_or_a_real_scene_type(self):
        for category, mapped in SURFACE_CATEGORY_MAP.items():
            with self.subTest(category=category):
                if mapped is not None:
                    self.assertIn(mapped, OBJECT_TYPES | {"wall"})

    def test_the_furnishing_categories_are_all_covered(self):
        # The iOS 16/17 CapturedRoom.Object.Category cases, written out
        # here so that adding one to the importer without adding it here
        # is a failure rather than a silent gap.
        expected = {
            "storage", "refrigerator", "stove", "bed", "sink", "washerDryer",
            "toilet", "bathtub", "oven", "dishwasher", "table", "sofa",
            "chair", "fireplace", "television", "stairs",
        }
        self.assertEqual(set(OBJECT_CATEGORY_MAP), expected)

    def test_the_five_types_a_room_scan_can_fill(self):
        mapped = {k: v for k, v in OBJECT_CATEGORY_MAP.items() if v is not None}
        self.assertEqual(
            mapped,
            {"bed": "bed", "table": "table", "chair": "chair", "sofa": "sofa", "storage": "almirah"},
        )

    def test_the_unverified_list_is_not_empty(self):
        # If somebody ever verifies these against a real export they
        # should shorten this list, and the test is the reminder that it
        # is a list of open questions rather than documentation.
        self.assertGreaterEqual(len(UNVERIFIED), 5)


class TestImport(unittest.TestCase):
    def setUp(self):
        self.scene_dict, self.report = import_roomplan(fixture())

    def test_the_output_is_a_scene_the_loader_accepts(self):
        self.assertIsNone(self.report.validation_error, self.report.validation_error)
        scene = scene_from_mapping(self.scene_dict)
        self.assertEqual(scene.schema_version, 1)

    def test_the_room_is_the_size_the_walls_say(self):
        room = self.scene_dict["room"]
        self.assertAlmostEqual(room["width_m"], 4.0, places=9)
        self.assertAlmostEqual(room["depth_m"], 3.0, places=9)
        self.assertAlmostEqual(room["height_m"], 2.7, places=9)

    def test_the_origin_moves_to_a_corner(self):
        # The scan's origin was at ARKit (0, 0, 0), which is outside the
        # room. Every wall endpoint should now be at 0 or at the far wall.
        xs = [c[0] for w in self.scene_dict["walls"] for c in (w["from_m"], w["to_m"])]
        ys = [c[1] for w in self.scene_dict["walls"] for c in (w["from_m"], w["to_m"])]
        self.assertAlmostEqual(min(xs), 0.0, places=9)
        self.assertAlmostEqual(min(ys), 0.0, places=9)
        self.assertAlmostEqual(max(xs), 4.0, places=9)
        self.assertAlmostEqual(max(ys), 3.0, places=9)

    def test_the_axis_convention_is_the_one_the_module_documents(self):
        # THIS CHECKS A CONVENTION, NOT REALITY. The importer reads
        # scene (x, y, z) = (ark_x, -ark_z, ark_y); the bed's ARKit
        # translation was (11, 0.275, -3) and the origin moved by
        # (10, 2), so it lands at (1, 1, 0.275). If Apple's handedness
        # turns out to be the other way, this test still passes and the
        # room is still mirrored.
        bed = next(o for o in self.scene_dict["objects"] if o["type"] == "bed")
        self.assertAlmostEqual(bed["centre_m"][0], 1.0, places=9)
        self.assertAlmostEqual(bed["centre_m"][1], 1.0, places=9)
        self.assertAlmostEqual(bed["centre_m"][2], 0.275, places=9)

    def test_dimensions_are_reordered_to_width_depth_height(self):
        # RoomPlan gives (width, height, depth); the scene wants
        # (width, depth, height). The bed was (1.9, 0.55, 1.4).
        bed = next(o for o in self.scene_dict["objects"] if o["type"] == "bed")
        self.assertEqual(bed["size_m"], [1.9, 1.4, 0.55])

    def test_a_rotated_wall_keeps_its_length(self):
        west = next(w for w in self.scene_dict["walls"] if w["id"] == "wall-west")
        length = math.dist(west["from_m"], west["to_m"])
        self.assertAlmostEqual(length, 3.0, places=9)
        self.assertAlmostEqual(west["from_m"][0], 0.0, places=9)
        self.assertAlmostEqual(west["to_m"][0], 0.0, places=9)

    def test_both_category_encodings_are_read(self):
        # "bed" as a string, {"table": {}} as an object, and a door as
        # {"door": {"isOpen": false}}.
        types = {o["type"] for o in self.scene_dict["objects"]}
        self.assertIn("bed", types)
        self.assertIn("table", types)
        self.assertIn("door", types)

    def test_storage_becomes_an_almirah(self):
        almirah = next(o for o in self.scene_dict["objects"] if o["type"] == "almirah")
        self.assertEqual(almirah["id"], "storage-1")
        self.assertEqual(almirah["size_m"], [1.0, 0.6, 2.0])

    def test_an_opening_in_a_wall_is_reported_and_dropped(self):
        skipped = {n.what for n in self.report.of_kind("skipped")}
        self.assertIn("opening-1", skipped)
        self.assertNotIn("opening", {o["type"] for o in self.scene_dict["objects"]})

    def test_a_category_with_no_counterpart_says_so(self):
        note = next(n for n in self.report.of_kind("skipped") if n.what == "toilet-1")
        self.assertIn("toilet", note.detail)
        self.assertIn("no counterpart", note.detail)

    def test_an_unrecognised_category_is_reported_not_guessed(self):
        note = next(n for n in self.report.of_kind("unknown") if n.what == "hammock-1")
        self.assertIn("hammock", note.detail)
        self.assertNotIn("hammock", {o["id"] for o in self.scene_dict["objects"]})

    def test_an_unreadable_transform_does_not_stop_the_import(self):
        note = next(n for n in self.report.of_kind("skipped") if n.what == "broken-1")
        self.assertIn("transform", note.detail)
        self.assertEqual(len(self.scene_dict["walls"]), 4)

    def test_every_material_is_a_default_and_is_reported(self):
        for wall in self.scene_dict["walls"]:
            self.assertEqual(wall["material"], DEFAULT_WALL_MATERIAL)
            self.assertEqual(wall["thickness_m"], DEFAULT_WALL_THICKNESS_M)
        for obj in self.scene_dict["objects"]:
            self.assertIn(obj["material"], OBJECT_MATERIALS)
        details = " ".join(n.detail for n in self.report.of_kind("default"))
        self.assertIn("RoomPlan reports neither", details)
        self.assertIn("no materials at all", details)

    def test_an_opening_is_moved_inside_the_floor_and_the_move_is_reported(self):
        # RoomPlan puts a door's centre in the plane of its wall, where
        # half the box is outside the room. The importer moves it in by
        # half its thickness, and says so.
        door = next(o for o in self.scene_dict["objects"] if o["type"] == "door")
        self.assertAlmostEqual(door["centre_m"][1], 0.5 * DEFAULT_OPENING_THICKNESS_M, places=9)
        window = next(o for o in self.scene_dict["objects"] if o["type"] == "window")
        self.assertAlmostEqual(window["centre_m"][0], 0.5 * DEFAULT_OPENING_THICKNESS_M, places=9)
        warnings = " ".join(n.detail for n in self.report.of_kind("warning"))
        self.assertIn("towards the middle of the room", warnings)

    def test_the_floor_outline_is_lost_and_the_report_says_so(self):
        self.assertNotIn("polygon_m", self.scene_dict["room"])
        warnings = {n.what for n in self.report.of_kind("warning")}
        self.assertIn("room.polygon_m", warnings)

    def test_there_are_no_nodes_and_the_report_says_so(self):
        self.assertEqual(self.scene_dict["nodes"], [])
        warnings = {n.what for n in self.report.of_kind("warning")}
        self.assertIn("nodes", warnings)

    def test_the_description_says_where_the_numbers_came_from(self):
        description = self.scene_dict["description"]
        self.assertIn("RoomPlan", description)
        self.assertIn("never been run against a", description)
        self.assertIn("default", description)

    def test_the_report_prints_the_open_questions(self):
        text = str(self.report)
        self.assertIn("UNVERIFIED", text)
        for item in UNVERIFIED:
            self.assertIn(item, text)

    def test_an_unknown_top_level_key_is_reported(self):
        raw = fixture()
        raw["roomTemperature"] = 21.0
        _, report = import_roomplan(raw)
        self.assertTrue(
            any("roomTemperature" in n.what for n in report.of_kind("unknown")),
            [str(n) for n in report.notes],
        )


class TestBadInput(unittest.TestCase):
    def test_an_empty_export_is_an_error_and_not_a_crash(self):
        with self.assertRaises(SceneError) as caught:
            import_roomplan({})
        self.assertIn("usable transform", str(caught.exception))

    def test_a_list_is_not_an_export(self):
        with self.assertRaises(SceneError):
            import_roomplan([])

    def test_a_scan_of_only_furniture_still_produces_a_room(self):
        # No walls at all: the extent comes from the furniture and the
        # height from the default, which the report has to say.
        raw = fixture()
        raw["walls"] = []
        raw["doors"] = []
        raw["windows"] = []
        scene_dict, report = import_roomplan(raw)
        self.assertEqual(scene_dict["walls"], [])
        self.assertGreater(scene_dict["room"]["width_m"], 0.0)
        details = {n.what for n in report.of_kind("default")}
        self.assertIn("room.height_m", details)


if __name__ == "__main__":
    unittest.main()
