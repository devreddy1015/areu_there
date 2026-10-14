"""Apple RoomPlan export -> Saans scene. A STUB, and unverified.

RoomPlan is a LiDAR scan on an iPhone or iPad. It is the quickest honest
way to get a room's shape into `scenes/*.json` without a tape measure,
and it is worth being explicit about why it is here: the geometry comes
from a laser, not from WiFi. Nothing an ESP32 measures could produce any
of it.

WHAT "STUB" MEANS HERE, EXACTLY

The mapping table below is real and complete: every RoomPlan surface and
object category I know of appears in it, either mapped to one of our
seven object types or listed with the reason it is not. Unknown
categories are reported, never dropped quietly.

The rest is unverified. NO ROOMPLAN EXPORT HAS BEEN RUN THROUGH THIS.
Every item in `UNVERIFIED` below is an assumption about a file format I
have not had a sample of, and each is the kind of assumption that
produces a room of the right size and the wrong orientation. Treat a
scene this produces as a first draft to check by eye, not as a
measurement.

It does not round-trip. There is no scene -> RoomPlan direction, and the
forward direction loses things: openings, every object category with no
counterpart here, the floor outline of a non-rectangular room, and all
of the materials, which RoomPlan does not report at all.

WHAT ROOMPLAN DOES NOT GIVE YOU

  * Materials. Not estimated, not guessed at, not present in the export.
    Everything this importer writes for `material` is a DEFAULT chosen
    here, and every one of them is listed in the report. A viewer must
    not present them as scanned.
  * Wall thickness. RoomPlan models a wall as a plane. The thickness in
    the output is a default.
  * A floor outline. This importer writes the bounding box, so an
    L-shaped room comes out rectangular and about a third too large. Fix
    `room.polygon_m` by hand; the loader will check that its bounding box
    still matches the stated width and depth.
  * Node positions. RoomPlan has never heard of an ESP32. `nodes` comes
    out empty and the scene will not convert to a `Geometry` until
    somebody surveys them.

USE

    import json
    from wifisense.roomplan_import import import_roomplan
    from wifisense.scene import scene_from_mapping

    scene_dict, report = import_roomplan(json.load(open("Room.json")))
    print(report)                      # read this before anything else
    scene = scene_from_mapping(scene_dict)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from .scene import SceneError, scene_from_mapping

__all__ = [
    "OBJECT_CATEGORY_MAP",
    "SURFACE_CATEGORY_MAP",
    "UNVERIFIED",
    "DEFAULT_WALL_THICKNESS_M",
    "DEFAULT_WALL_MATERIAL",
    "DEFAULT_OPENING_THICKNESS_M",
    "DEFAULT_ROOM_HEIGHT_M",
    "ImportNote",
    "ImportReport",
    "import_roomplan",
]


# Every assumption that a real export would settle in five minutes, and
# that nothing here has tested. Printed by the report so it travels with
# the output.
UNVERIFIED = (
    "transform is 16 floats in COLUMN-major order, so the translation is "
    "elements 12, 13, 14 (simd_float4x4 is column-major; a JSON encoder "
    "that flattened it row-major would put the translation at 3, 7, 11 and "
    "every object would land on the wrong side of the room)",
    "ARKit axes are +X right, +Y up, -Z forward, so scene (x, y, z) = "
    "(ark_x, -ark_z, ark_y). The sign on ark_z is the one that keeps the "
    "frame right-handed; get it wrong and the room is mirrored, which is "
    "hard to see in a plan view and obvious the first time a door is on "
    "the wrong wall",
    "dimensions is (width, height, depth) in the item's own frame, so our "
    "size_m is (dimensions.x, dimensions.z, dimensions.y)",
    "rotation about the vertical axis is atan2(-m[2], m[0]) from the "
    "transform's first column",
    "the category field may be a plain string or a single-key object "
    "(Swift encodes a case with associated values, such as door(isOpen:), "
    "as an object). Both are accepted here; which one a given iOS version "
    "writes is unconfirmed",
    "the top-level keys are walls, doors, windows, openings, objects and "
    "floors; older exports may not have floors, newer ones may add keys "
    "this importer will report as unknown",
    "wall dimensions.x is the wall's length and dimensions.y its height",
)

# RoomPlan reports neither of these. They are defaults, and every use of
# one is listed in the report.
DEFAULT_WALL_THICKNESS_M = 0.115
DEFAULT_WALL_MATERIAL = "drywall"
DEFAULT_OPENING_THICKNESS_M = 0.06
DEFAULT_ROOM_HEIGHT_M = 2.7

# RoomPlan surface categories -> what we do with them.
SURFACE_CATEGORY_MAP: dict[str, str | None] = {
    "wall": "wall",
    "door": "door",
    "window": "window",
    # An opening is a doorway with no door in it: a hole, with nothing to
    # draw and nothing to attenuate. Dropping it loses the fact that the
    # wall is not continuous there, which matters to the radio.
    "opening": None,
    # The floor surface would be the right source for room.polygon_m. Its
    # outline is not in the fields this importer reads, so the floor is
    # used for the bounding box and nothing else.
    "floor": None,
}

# RoomPlan object categories (CapturedRoom.Object.Category as of the
# iOS 16/17 API) -> one of our seven object types, or None with the
# reason. A later iOS may add cases; those are reported as unknown.
OBJECT_CATEGORY_MAP: dict[str, str | None] = {
    "bed": "bed",
    "table": "table",
    "chair": "chair",
    "sofa": "sofa",
    # A wardrobe, a cupboard, a bookshelf: RoomPlan calls them all
    # storage, and "almirah" is this format's name for that box.
    "storage": "almirah",
    # Everything below has no counterpart among our seven types. They are
    # skipped rather than forced into "table", because a scene that calls
    # a toilet a table is worse than a scene that is missing a toilet.
    "refrigerator": None,
    "stove": None,
    "oven": None,
    "dishwasher": None,
    "washerDryer": None,
    "sink": None,
    "toilet": None,
    "bathtub": None,
    "fireplace": None,
    "television": None,
    "stairs": None,
}


@dataclass(frozen=True)
class ImportNote:
    """One thing the reader has to know about this import.

    `kind` is "skipped", "default", "unknown" or "warning"; `what`
    identifies the item; `detail` says why.
    """

    kind: str
    what: str
    detail: str

    def __str__(self) -> str:
        return f"{self.kind}: {self.what} - {self.detail}"


@dataclass
class ImportReport:
    """What the importer assumed, dropped and could not understand.

    This is not logging. Nothing in the output scene says which of its
    numbers were scanned and which were invented here, so the report is
    the only record of the difference, and a caller that throws it away
    has thrown away the part that says the materials are made up.
    """

    notes: list[ImportNote] = field(default_factory=list)
    counts: dict[str, int] = field(default_factory=dict)
    validation_error: str | None = None

    def add(self, kind: str, what: str, detail: str) -> None:
        self.notes.append(ImportNote(kind, what, detail))

    def count(self, key: str, n: int = 1) -> None:
        self.counts[key] = self.counts.get(key, 0) + n

    def of_kind(self, kind: str) -> list[ImportNote]:
        return [n for n in self.notes if n.kind == kind]

    def __str__(self) -> str:
        lines = ["RoomPlan import (UNVERIFIED: no real export has been tested)"]
        if self.counts:
            lines.append(
                "  kept: " + ", ".join(f"{k} {v}" for k, v in sorted(self.counts.items()))
            )
        for note in self.notes:
            lines.append(f"  {note}")
        if self.validation_error:
            lines.append(f"  INVALID SCENE: {self.validation_error}")
        lines.append("  assumptions not yet checked against a real file:")
        lines.extend(f"    - {u}" for u in UNVERIFIED)
        return "\n".join(lines)


def _category_name(value: Any) -> str | None:
    """The category's name, whether it is a string or a one-key object.

    Swift encodes an enum case without associated values as a string and
    one with them (`door(isOpen:)`) as an object with a single key. Which
    form a given export uses is one of the unverified items.
    """
    if isinstance(value, str):
        return value
    if isinstance(value, dict) and len(value) == 1:
        return next(iter(value))
    return None


def _transform_translation(transform: Any) -> tuple[float, float, float] | None:
    """(x, y, z) from a flat 16-float column-major transform."""
    if not isinstance(transform, list) or len(transform) != 16:
        return None
    try:
        values = [float(v) for v in transform]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return (values[12], values[13], values[14])


def _transform_yaw_deg(transform: Any) -> float:
    """Rotation about the vertical axis, degrees counter-clockwise in scene axes.

    The first column of the matrix is the item's local X axis in world
    coordinates. For a rotation by theta about ARKit's +Y that column is
    (cos theta, 0, -sin theta), so theta = atan2(-m[2], m[0]). Scene y is
    -ark_z, which flips the sign back, so the same theta is a
    counter-clockwise rotation in the scene's plan view.
    """
    values = transform if isinstance(transform, list) and len(transform) == 16 else None
    if values is None:
        return 0.0
    try:
        m0, m2 = float(values[0]), float(values[2])
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(m0) and math.isfinite(m2)):
        return 0.0
    return math.degrees(math.atan2(-m2, m0))


def _dimensions(value: Any) -> tuple[float, float, float] | None:
    """(width, height, depth) as RoomPlan orders them."""
    if not isinstance(value, list) or len(value) != 3:
        return None
    try:
        values = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in values):
        return None
    return (values[0], values[1], values[2])


def _to_scene_axes(ark: tuple[float, float, float]) -> tuple[float, float, float]:
    """ARKit (x, y, z) -> scene (x, y, z). See UNVERIFIED."""
    return (ark[0], -ark[2], ark[1])


def _identifier(item: dict, fallback: str) -> str:
    value = item.get("identifier")
    return value if isinstance(value, str) and value else fallback


def import_roomplan(
    raw: Any,
    *,
    wall_material: str = DEFAULT_WALL_MATERIAL,
    wall_thickness_m: float = DEFAULT_WALL_THICKNESS_M,
    name: str | None = None,
) -> tuple[dict, ImportReport]:
    """Convert a parsed RoomPlan export into a scene dictionary.

    Returns the dictionary and a report. It does not raise on a scene it
    cannot fully understand: an unreadable item becomes a note and the
    rest of the room still comes through, because a scan with one bad
    transform is still most of a room. It does validate its own output
    and puts any failure in `report.validation_error` rather than
    raising, since a scan whose furniture overhangs a wall by two
    centimetres is a normal scan and a human has to look at it either
    way.
    """
    report = ImportReport()
    if not isinstance(raw, dict):
        raise SceneError(f"RoomPlan export: expected an object at the top level")

    known_keys = {
        "identifier",
        "version",
        "walls",
        "doors",
        "windows",
        "openings",
        "objects",
        "floors",
        "sections",
        "story",
    }
    for key in sorted(set(raw.keys()) - known_keys):
        report.add("unknown", f"top-level key {key!r}", "not read by this importer")

    walls_raw = raw.get("walls") if isinstance(raw.get("walls"), list) else []
    floors_raw = raw.get("floors") if isinstance(raw.get("floors"), list) else []
    openings_raw = raw.get("openings") if isinstance(raw.get("openings"), list) else []

    # Pass 1: everything's position in scene axes, so the bounding box is
    # known before anything is written. The origin of a RoomPlan scan is
    # wherever the capture started, which is not a corner of the room.
    placed: list[tuple[str, dict, tuple[float, float, float], tuple[float, float, float]]] = []
    for group, items in (
        ("wall", walls_raw),
        ("floor", floors_raw),
        ("door", raw.get("doors") if isinstance(raw.get("doors"), list) else []),
        ("window", raw.get("windows") if isinstance(raw.get("windows"), list) else []),
        ("object", raw.get("objects") if isinstance(raw.get("objects"), list) else []),
    ):
        for i, item in enumerate(items):
            label = f"{group}[{i}]"
            if not isinstance(item, dict):
                report.add("skipped", label, "not an object")
                continue
            ident = _identifier(item, label)
            translation = _transform_translation(item.get("transform"))
            dims = _dimensions(item.get("dimensions"))
            if translation is None:
                report.add("skipped", ident, "transform is not 16 finite numbers")
                continue
            if dims is None:
                report.add("skipped", ident, "dimensions is not 3 finite numbers")
                continue
            placed.append((group, item, _to_scene_axes(translation), dims))

    for i, item in enumerate(openings_raw):
        ident = _identifier(item if isinstance(item, dict) else {}, f"opening[{i}]")
        report.add(
            "skipped",
            ident,
            "RoomPlan opening: a gap in a wall with nothing to draw. The scene "
            "format has no type for it, and the wall it interrupts is written "
            "as continuous",
        )

    if not placed:
        raise SceneError("RoomPlan export: nothing with a usable transform in it")

    # Pass 2: wall centrelines, still in the scan's own coordinates. They
    # are worked out before the origin is chosen because a wall's ENDPOINT
    # is what reaches the corner of the room; its centre never does, and a
    # bounding box taken from centres alone comes out half a wall too
    # small and pushes everything to negative coordinates.
    wall_lines: list[tuple[str, tuple[float, float], tuple[float, float], float]] = []
    for group, item, position, dims in placed:
        if group != "wall":
            continue
        ident = _identifier(item, "wall")
        category = _category_name(item.get("category"))
        if category is not None and SURFACE_CATEGORY_MAP.get(category) != "wall":
            report.add("unknown", ident, f"wall list holds category {category!r}")
        length, height = dims[0], dims[1]
        if length <= 0.0:
            report.add("skipped", ident, f"wall length is {length:g} m")
            continue
        yaw = math.radians(_transform_yaw_deg(item.get("transform")))
        half = 0.5 * length
        direction = (math.cos(yaw), math.sin(yaw))
        wall_lines.append(
            (
                ident,
                (position[0] - half * direction[0], position[1] - half * direction[1]),
                (position[0] + half * direction[0], position[1] + half * direction[1]),
                height,
            )
        )

    points: list[tuple[float, float]] = [(p[0], p[1]) for _, _, p, _ in placed]
    for _, start, end, _ in wall_lines:
        points.append(start)
        points.append(end)
    min_x = min(p[0] for p in points)
    min_y = min(p[1] for p in points)
    width = max(p[0] for p in points) - min_x
    depth = max(p[1] for p in points) - min_y
    if width <= 0.0 or depth <= 0.0:
        raise SceneError(
            f"RoomPlan export: the scan's bounding box is {width:g} x {depth:g} m"
        )
    room_centre = (0.5 * width, 0.5 * depth)

    def to_floor(point: tuple[float, ...]) -> tuple[float, ...]:
        return (point[0] - min_x, point[1] - min_y, *point[2:])

    walls: list[dict] = []
    wall_heights: list[float] = []
    for ident, start, end, height in wall_lines:
        walls.append(
            {
                "id": ident[:64],
                "from_m": list(to_floor(start)),
                "to_m": list(to_floor(end)),
                "material": wall_material,
                "thickness_m": wall_thickness_m,
            }
        )
        if height > 0.0:
            wall_heights.append(height)
        report.count("walls")

    if walls:
        report.add(
            "default",
            f"{len(walls)} walls",
            f"material {wall_material!r} and thickness {wall_thickness_m:g} m: "
            f"RoomPlan reports neither",
        )

    height_m = max(wall_heights) if wall_heights else DEFAULT_ROOM_HEIGHT_M
    if not wall_heights:
        report.add(
            "default",
            "room.height_m",
            f"{DEFAULT_ROOM_HEIGHT_M:g} m: no wall carried a usable height",
        )
    report.add(
        "warning",
        "room.polygon_m",
        "not written. The floor outline is not read from the export, so a "
        "room that is not rectangular comes out as its bounding box and is "
        "too large. Fix it by hand.",
    )

    # Pass 3: doors, windows and furniture.
    objects: list[dict] = []
    defaulted_materials = 0
    nudged_openings = 0
    for group, item, position, dims in placed:
        if group in ("wall", "floor"):
            continue
        ident = _identifier(item, group)
        category = _category_name(item.get("category"))
        if category is None:
            report.add("skipped", ident, "category is neither a string nor a one-key object")
            continue

        if group in ("door", "window"):
            scene_type = SURFACE_CATEGORY_MAP.get(category)
            if scene_type not in ("door", "window"):
                report.add("unknown", ident, f"{group} list holds category {category!r}")
                continue
            size = [dims[0], DEFAULT_OPENING_THICKNESS_M, dims[1]]
            material = "wood" if scene_type == "door" else "glass"
            defaulted_materials += 1
        else:
            if category not in OBJECT_CATEGORY_MAP:
                report.add(
                    "unknown",
                    ident,
                    f"object category {category!r} is not in this importer's table; "
                    f"a newer iOS may have added it",
                )
                continue
            scene_type = OBJECT_CATEGORY_MAP[category]
            if scene_type is None:
                report.add(
                    "skipped",
                    ident,
                    f"RoomPlan category {category!r} has no counterpart among the "
                    f"scene's seven object types",
                )
                continue
            # dimensions is (width, height, depth); the scene wants
            # (width, depth, height).
            size = [dims[0], dims[2], dims[1]]
            material = "wood"
            defaulted_materials += 1

        if min(size) <= 0.0:
            report.add("skipped", ident, f"size {size} has a non-positive side")
            continue

        centre = to_floor(position)
        rotation = _transform_yaw_deg(item.get("transform"))

        if scene_type in ("door", "window"):
            # RoomPlan models a door as a plane in the wall, so its centre
            # is ON the boundary and half the leaf is outside the room.
            # The scene format wants a box inside the floor, so the
            # opening is moved half its thickness towards the middle of
            # the room. A few centimetres, and a change to a scanned
            # position, which is why it is reported.
            theta = math.radians(rotation)
            normal = (-math.sin(theta), math.cos(theta))
            towards = (room_centre[0] - centre[0], room_centre[1] - centre[1])
            sign = 1.0 if normal[0] * towards[0] + normal[1] * towards[1] >= 0.0 else -1.0
            shift = sign * 0.5 * size[1]
            centre = (centre[0] + shift * normal[0], centre[1] + shift * normal[1], centre[2])
            nudged_openings += 1

        objects.append(
            {
                "id": ident[:64],
                "type": scene_type,
                # RoomPlan's transform sits at the item's centre, including
                # in height, so z is used as it comes.
                "centre_m": [centre[0], centre[1], centre[2]],
                "size_m": size,
                "rotation_deg": rotation,
                "material": material,
            }
        )
        report.count(scene_type + "s")

    if defaulted_materials:
        report.add(
            "default",
            f"{defaulted_materials} object materials",
            "RoomPlan reports no materials at all. Doors are written as wood, "
            "windows as glass, furniture as wood. None of that was measured.",
        )
    if nudged_openings:
        report.add(
            "warning",
            f"{nudged_openings} doors and windows",
            f"moved {0.5 * DEFAULT_OPENING_THICKNESS_M:g} m towards the middle of "
            f"the room. RoomPlan puts an opening's centre in the plane of its "
            f"wall, where half the box would be outside the floor; the scene "
            f"format wants it inside. This changes a scanned position.",
        )
    report.add(
        "warning",
        "nodes",
        "empty: RoomPlan does not know where the sensors are. Survey them and "
        "add them before this scene can become a deployment.",
    )

    scene: dict[str, Any] = {
        "schema_version": 1,
        "name": name or f"Imported room, {width:.1f} x {depth:.1f} m",
        "description": (
            "Imported from an Apple RoomPlan LiDAR scan by "
            "wifisense/roomplan_import.py, which has never been run against a "
            "real export. Geometry is from the scan; every material and every "
            "thickness is a default chosen by the importer, and the floor "
            "outline is the bounding box. Check it against the room before "
            "trusting it."
        ),
        "room": {
            "width_m": width,
            "depth_m": depth,
            "height_m": height_m,
        },
        "walls": walls,
        "objects": objects,
        "nodes": [],
        "people": [],
    }

    try:
        scene_from_mapping(scene)
    except SceneError as exc:
        report.validation_error = str(exc)

    return scene, report
