"""Scene files: the room as DEFINED, loaded, checked and converted.

A scene is where the room comes from. It is typed in, measured with a
tape, or imported from a LiDAR scan (`wifisense.roomplan_import`). It
never comes from WiFi: an ESP32 at 20 MHz on one antenna cannot measure a
room's dimensions and cannot recognise a bed, so anything displayed from
this file is SCENE data and has to be labelled that way, separately from
a SIM estimate and from a LIVE measurement.

The format is `scenes/scene.schema.json` (JSON Schema draft 2020-12) and
the examples are `scenes/bedroom.json`, `scenes/office.json`,
`scenes/hall-l.json`.

WHY THE VALIDATION IS HAND-ROLLED. This repository is pure standard
library and has no install step, and that is a property worth keeping:
`python3 -m wifisense demo` works on a fresh machine. `jsonschema` would
be the obvious dependency and it is not available, so the schema file is
shipped for external tooling — editors, CI in another language, anyone
checking a scene without this package — and the checking here is written
out by hand against the same rules. The two can drift. `tests/test_scene.py`
is what stops them: it reads the schema file and asserts that the enums,
the required-key lists and the version constant in this module still
match it.

The checks here also go beyond what a schema can say. A schema can
require a positive `width_m`; it cannot say that a bed is inside the
room, that two nodes are not at the same point, or that a polygon's
bounding box matches the stated width and depth.

AXES, AND THE ONE COLLISION THAT MATTERS

    scene x -> width_m    floor plan, metres, origin at a floor corner
    scene y -> depth_m    the SECOND HORIZONTAL axis of the floor plan
    scene z -> height_m   ELEVATION, up

`wifisense.localize.geometry.Room` is a 2-D plan-view type whose two
fields are `width_m` and `height_m`, and its own docstring says its
`height_m` is the second horizontal axis, not elevation. So:

    scene room.depth_m   -> geometry.Room.height_m
    scene room.height_m  -> dropped; the 2-D types have nowhere to put it

`to_room` is the only place that mapping happens, and it is the only
place it should happen. Writing `Room(width_m=scene.room.width_m,
height_m=scene.room.height_m)` instead produces a room of the right type
and the wrong shape — a 4.2 x 3.6 m bedroom becomes 4.2 x 2.7 m — and
nothing downstream can tell, because both numbers are plausible. That is
why `tests/test_scene.py` checks the conversion against the depth rather
than against the elevation, with the elevation deliberately different
from both.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence

from .localize.geometry import Geometry, Node, PixelGrid, Room

__all__ = [
    "SCHEMA_VERSION",
    "WALL_MATERIALS",
    "OBJECT_MATERIALS",
    "OBJECT_TYPES",
    "MATERIALS",
    "MaterialConstants",
    "SceneError",
    "SceneRoom",
    "Wall",
    "SceneObject",
    "SceneNode",
    "Person",
    "Phone",
    "Scene",
    "load_scene",
    "parse_scene",
    "scene_from_mapping",
    "material_constants",
    "to_room",
    "to_nodes",
    "to_geometry",
    "scenes_dir",
]

# A loader that does not know a version refuses the file. The version
# exists for the case a schema cannot catch: a field that keeps its name
# and changes its meaning.
SCHEMA_VERSION = 1

WALL_MATERIALS = frozenset({"concrete", "brick", "drywall", "glass", "wood"})
OBJECT_MATERIALS = frozenset({"wood", "metal", "fabric", "glass"})
OBJECT_TYPES = frozenset(
    {"bed", "table", "chair", "almirah", "sofa", "door", "window"}
)

# Containment slack. A door leaf sits exactly in the plane of its wall and
# a window's outer face is exactly on the boundary, so "inside" has to
# admit the boundary itself; this is floating-point slack, not a licence
# to place furniture outside the room.
CONTAIN_TOL_M = 1e-6

# Two nodes closer than this are the same point as far as the link
# geometry is concerned: the link between them has zero length and
# `geometry.Geometry` rejects it anyway, with a less specific message.
NODE_MIN_SEPARATION_M = 1e-6

# The channel the rest of the repository uses (802.11 channel 6).
MATERIAL_REFERENCE_GHZ = 2.437

# Vacuum permittivity, CODATA 2018, as used for eps'' = sigma / (2 pi f eps0).
EPSILON_0_F_PER_M = 8.8541878128e-12


class SceneError(ValueError):
    """A scene file that cannot be loaded, with the path to the problem.

    A `ValueError`, because that is what a caller parsing a file already
    catches, and because every one of these is a bad value rather than a
    missing file or a broken pipe.
    """


# --------------------------------------------------------------------------
# Materials
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class MaterialConstants:
    """ITU-R P.2040 Table 3 coefficients for one material.

        eps_r' = a * f_GHz^b          sigma = c * f_GHz^d   [S/m]

    Stored as coefficients rather than as numbers evaluated at one
    frequency, so the table stays checkable and so a 5 GHz radio gets the
    right values instead of the 2.4 GHz ones. `scenes/materials.json`
    carries the same coefficients plus their values at 2.437 GHz, and
    `tests/test_scene.py` recomputes those from here.

    `valid_ghz` is the Recommendation's stated frequency range for the
    row. Outside it the fit is an extrapolation, so `evaluate` says so
    rather than quietly returning a number.
    """

    name: str
    a: float
    b: float
    c: float
    d: float
    valid_ghz: tuple[float, float]

    def epsilon_r(self, f_ghz: float = MATERIAL_REFERENCE_GHZ) -> float:
        """Real relative permittivity at `f_ghz`."""
        return self.a * f_ghz**self.b

    def sigma_s_per_m(self, f_ghz: float = MATERIAL_REFERENCE_GHZ) -> float:
        """Conductivity in siemens per metre at `f_ghz`."""
        return self.c * f_ghz**self.d

    def epsilon_imag(self, f_ghz: float = MATERIAL_REFERENCE_GHZ) -> float:
        """Imaginary relative permittivity: sigma / (2 pi f eps0)."""
        f_hz = f_ghz * 1e9
        return self.sigma_s_per_m(f_ghz) / (2.0 * math.pi * f_hz * EPSILON_0_F_PER_M)

    def reflection_normal(self, f_ghz: float = MATERIAL_REFERENCE_GHZ) -> float:
        """|Gamma| at normal incidence, air onto a half-space of this material.

        Gamma = (1 - sqrt(eta)) / (1 + sqrt(eta)) with eta = eps' - j eps''.

        This is ONE interface, not a wall. A real wall is a slab with two
        interfaces and internal reflections (ITU-R P.2040 section 5.2),
        and for something as thin as a sheet of drywall the slab response
        differs substantially from this and swings with thickness. The
        number is here because it is the quantity the simulator's scalar
        `wall_reflection` stands for, so it says which materials are in
        the right ballpark for that knob — concrete comes out at 0.395
        against the simulator's long-standing 0.4 — and not because it is
        the reflectivity of a finished wall.
        """
        eta = complex(self.epsilon_r(f_ghz), -self.epsilon_imag(f_ghz))
        root = _complex_sqrt(eta)
        return abs((1.0 - root) / (1.0 + root))

    def in_range(self, f_ghz: float = MATERIAL_REFERENCE_GHZ) -> bool:
        lo, hi = self.valid_ghz
        return lo <= f_ghz <= hi


def _complex_sqrt(z: complex) -> complex:
    """Principal square root, without importing cmath for one call."""
    r = abs(z)
    if r == 0.0:
        return 0j
    re = math.sqrt((r + z.real) / 2.0)
    im = math.copysign(math.sqrt((r - z.real) / 2.0), z.imag)
    return complex(re, im)


# ITU-R P.2040 Table 3. `fabric` is deliberately absent: the table has no
# row for it and no sourced value is available here, so a sofa carries
# "fabric" as a label for the viewer and any physics layer must refuse it
# rather than use an invented permittivity. See scenes/materials.json.
MATERIALS: dict[str, MaterialConstants] = {
    "concrete": MaterialConstants("concrete", 5.24, 0.0, 0.0462, 0.7822, (1.0, 100.0)),
    "brick": MaterialConstants("brick", 3.91, 0.0, 0.0238, 0.16, (1.0, 10.0)),
    "drywall": MaterialConstants("drywall", 2.73, 0.0, 0.0085, 0.9395, (1.0, 100.0)),
    "glass": MaterialConstants("glass", 6.31, 0.0, 0.0036, 1.3394, (0.1, 100.0)),
    "wood": MaterialConstants("wood", 1.99, 0.0, 0.0047, 1.0718, (0.001, 100.0)),
    "metal": MaterialConstants("metal", 1.0, 0.0, 1.0e7, 0.0, (1.0, 100.0)),
}

MATERIALS_WITHOUT_CONSTANTS = frozenset({"fabric"})


def material_constants(name: str) -> MaterialConstants | None:
    """Electrical constants for a material name, or None where there are none.

    Returns None for `fabric`, which is a valid scene material with no
    ITU-R P.2040 row. Raises `KeyError` for a name that is not a scene
    material at all, because that is a typo rather than a known gap.
    """
    if name in MATERIALS:
        return MATERIALS[name]
    if name in MATERIALS_WITHOUT_CONSTANTS:
        return None
    raise KeyError(f"not a scene material: {name!r}")


# --------------------------------------------------------------------------
# The scene types
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class SceneRoom:
    """The floor plan's bounds.

    Named `SceneRoom`, not `Room`, because `localize.geometry.Room` is a
    different type with a different meaning for `height_m`, and the two
    meet in `to_room`. Importing both under one name is how a room ends
    up rotated.

    `width_m` x `depth_m` is the clear interior floor. `height_m` is
    floor to ceiling. `polygon_m` is the floor outline for a room that is
    not a rectangle; when it is present its bounding box is the same
    `width_m` x `depth_m`, so a reader that ignores it still gets the
    extent right, and `area_m2` reports the polygon's real area.
    """

    width_m: float
    depth_m: float
    height_m: float
    polygon_m: tuple[tuple[float, float], ...] = ()

    @property
    def footprint(self) -> tuple[tuple[float, float], ...]:
        """The floor outline, counter-clockwise: the polygon or the box."""
        if self.polygon_m:
            return self.polygon_m
        w, d = self.width_m, self.depth_m
        return ((0.0, 0.0), (w, 0.0), (w, d), (0.0, d))

    @property
    def area_m2(self) -> float:
        """Floor area. For an L-shape this is less than width x depth."""
        return abs(_signed_area(self.footprint))

    def contains(self, x: float, y: float, tol: float = CONTAIN_TOL_M) -> bool:
        return _contains_point(self.footprint, x, y, tol)


@dataclass(frozen=True)
class Wall:
    """One wall segment in plan view, with the material the radio sees.

    `from_m` and `to_m` are the centreline. `thickness_m` is the full
    thickness about it. A perimeter wall's centreline runs along the
    boundary of the clear interior, so a viewer draws its thickness
    outwards; an interior partition has no outward side and is drawn
    symmetric about the segment.

    `height_m` defaults to the room's, which is what `effective_height_m`
    resolves.
    """

    from_m: tuple[float, float]
    to_m: tuple[float, float]
    material: str
    thickness_m: float
    id: str | None = None
    height_m: float | None = None

    @property
    def length_m(self) -> float:
        return math.hypot(self.to_m[0] - self.from_m[0], self.to_m[1] - self.from_m[1])

    def effective_height_m(self, room: SceneRoom) -> float:
        return room.height_m if self.height_m is None else self.height_m


@dataclass(frozen=True)
class SceneObject:
    """One upright box of furniture, or a door or a window.

    `size_m` is (width, depth, height) in the object's own frame, before
    `rotation_deg` turns it counter-clockwise about the vertical axis.
    `centre_m` includes z: a 2.05 m door has its centre at z = 1.025.
    """

    type: str
    centre_m: tuple[float, float, float]
    size_m: tuple[float, float, float]
    material: str
    rotation_deg: float = 0.0
    id: str | None = None

    @property
    def footprint(self) -> tuple[tuple[float, float], ...]:
        """The four plan-view corners after rotation, counter-clockwise."""
        return _rotated_footprint(
            self.centre_m[0],
            self.centre_m[1],
            self.size_m[0],
            self.size_m[1],
            self.rotation_deg,
        )

    @property
    def z_extent_m(self) -> tuple[float, float]:
        half = 0.5 * self.size_m[2]
        return (self.centre_m[2] - half, self.centre_m[2] + half)


@dataclass(frozen=True)
class SceneNode:
    """One ESP32 at a surveyed position.

    The position is an input, never an output. Nothing in this project
    can work out where a node is: a single-chain ESP32 has no angle of
    arrival and phase sanitization destroys absolute ranging, so the
    coordinates are surveyed by hand and any error in them lands directly
    in whatever is computed from them (NOTES-localize.md section 1).
    """

    id: int
    pos_m: tuple[float, float, float]
    label: str | None = None


@dataclass(frozen=True)
class Person:
    """A simulated person's starting position, in plan view only.

    No z: the sensing stack is 2-D and assumes chests at a common height.
    `height_m` is for drawing.
    """

    pos_m: tuple[float, float]
    height_m: float = 1.7
    id: str | None = None


@dataclass(frozen=True)
class Phone:
    """The one tracked handset's starting position.

    It has a z because a phone in a pocket and a phone on a table are at
    different heights. The 2-D estimators ignore it.
    """

    pos_m: tuple[float, float, float]
    id: str | None = None


@dataclass(frozen=True)
class Scene:
    """A whole scene file, parsed and checked."""

    room: SceneRoom
    schema_version: int = SCHEMA_VERSION
    name: str | None = None
    description: str | None = None
    walls: tuple[Wall, ...] = ()
    objects: tuple[SceneObject, ...] = ()
    nodes: tuple[SceneNode, ...] = ()
    people: tuple[Person, ...] = ()
    phone: Phone | None = None
    source_path: str | None = field(default=None, compare=False)

    @property
    def node_z_spread_m(self) -> float:
        """Largest difference in node elevation, 0.0 with fewer than two.

        The 2-D conversion drops z. That is only honest while this is
        small: `geometry.Room` assumes nodes and chests at roughly one
        height, which is the usual plan-view simplification. A scene that
        mounts one node at the skirting board and another above a door
        has broken that assumption, and the number to look at is this
        one.
        """
        if len(self.nodes) < 2:
            return 0.0
        zs = [n.pos_m[2] for n in self.nodes]
        return max(zs) - min(zs)


# --------------------------------------------------------------------------
# Geometry helpers
# --------------------------------------------------------------------------


def _signed_area(poly: Sequence[tuple[float, float]]) -> float:
    """Shoelace area; positive when the vertices run counter-clockwise."""
    total = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def _point_in_polygon(poly: Sequence[tuple[float, float]], x: float, y: float) -> bool:
    """Ray casting, with the boundary left undefined on purpose.

    A point exactly on an edge may come out either way here, which is why
    `_contains_point` tests the boundary separately rather than relying
    on this. Doors and windows sit exactly in the plane of their wall, so
    boundary points are the normal case, not an edge case.
    """
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y):
            t = (y - y0) / (y1 - y0)
            if x < x0 + t * (x1 - x0):
                inside = not inside
    return inside


def _distance_to_segment(
    px: float, py: float, ax: float, ay: float, bx: float, by: float
) -> float:
    dx, dy = bx - ax, by - ay
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return math.hypot(px - ax, py - ay)
    t = ((px - ax) * dx + (py - ay) * dy) / length_sq
    t = min(1.0, max(0.0, t))
    return math.hypot(px - (ax + t * dx), py - (ay + t * dy))


def _distance_to_boundary(
    poly: Sequence[tuple[float, float]], x: float, y: float
) -> float:
    n = len(poly)
    return min(
        _distance_to_segment(x, y, poly[i][0], poly[i][1], poly[(i + 1) % n][0], poly[(i + 1) % n][1])
        for i in range(n)
    )


def _contains_point(
    poly: Sequence[tuple[float, float]], x: float, y: float, tol: float
) -> bool:
    """Inside the polygon, or within `tol` metres of its boundary."""
    if _point_in_polygon(poly, x, y):
        return True
    return _distance_to_boundary(poly, x, y) <= tol


def _check_in_box(x: float, y: float, room: SceneRoom, path: str) -> None:
    """Inside the bounding box, which is where a wall is allowed to be.

    Walls are checked against the box rather than against the floor
    polygon because a wall runs ALONG the floor's edge, and a point on
    the edge is exactly the case point-in-polygon leaves undefined.
    """
    if not (
        -CONTAIN_TOL_M <= x <= room.width_m + CONTAIN_TOL_M
        and -CONTAIN_TOL_M <= y <= room.depth_m + CONTAIN_TOL_M
    ):
        raise SceneError(
            f"{path}: ({x:.3f}, {y:.3f}) m is outside the room's "
            f"{room.width_m:g} x {room.depth_m:g} m bounds"
        )


def _rotated_footprint(
    cx: float, cy: float, w: float, d: float, rotation_deg: float
) -> tuple[tuple[float, float], ...]:
    """The four plan corners of a `w` x `d` box centred at (cx, cy)."""
    theta = math.radians(rotation_deg)
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    half_w, half_d = 0.5 * w, 0.5 * d
    corners = ((-half_w, -half_d), (half_w, -half_d), (half_w, half_d), (-half_w, half_d))
    return tuple(
        (cx + ox * cos_t - oy * sin_t, cy + ox * sin_t + oy * cos_t)
        for ox, oy in corners
    )


# --------------------------------------------------------------------------
# Parsing and validation, by hand
# --------------------------------------------------------------------------


def _reject_constant(token: str) -> float:
    raise SceneError(
        f"{token} is not valid JSON and not a position: "
        f"every number in a scene must be finite"
    )


def _check_keys(obj: dict, required: Iterable[str], optional: Iterable[str], path: str) -> None:
    required = set(required)
    allowed = required | set(optional)
    missing = sorted(required - obj.keys())
    if missing:
        raise SceneError(f"{path}: missing {', '.join(missing)}")
    extra = sorted(obj.keys() - allowed)
    if extra:
        raise SceneError(
            f"{path}: unknown {'key' if len(extra) == 1 else 'keys'} "
            f"{', '.join(repr(k) for k in extra)} "
            f"(allowed: {', '.join(sorted(allowed))})"
        )


def _obj(value: Any, path: str) -> dict:
    if not isinstance(value, dict):
        raise SceneError(f"{path}: expected an object, got {_kind(value)}")
    return value


def _list(value: Any, path: str) -> list:
    if not isinstance(value, list):
        raise SceneError(f"{path}: expected an array, got {_kind(value)}")
    return value


def _kind(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "a boolean"
    if isinstance(value, (int, float)):
        return "a number"
    if isinstance(value, str):
        return "a string"
    if isinstance(value, list):
        return "an array"
    if isinstance(value, dict):
        return "an object"
    return type(value).__name__


def _number(value: Any, path: str) -> float:
    # bool is an int in Python; `true` is not a measurement.
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise SceneError(f"{path}: expected a number, got {_kind(value)}")
    out = float(value)
    if not math.isfinite(out):
        raise SceneError(f"{path}: expected a finite number, got {out}")
    return out


def _positive(value: Any, path: str) -> float:
    out = _number(value, path)
    if out <= 0.0:
        raise SceneError(f"{path}: must be positive, got {out:g}")
    return out


def _string(value: Any, path: str, max_length: int = 120) -> str:
    if not isinstance(value, str):
        raise SceneError(f"{path}: expected a string, got {_kind(value)}")
    if len(value) > max_length:
        raise SceneError(f"{path}: longer than {max_length} characters")
    return value


def _enum(value: Any, allowed: Iterable[str], path: str) -> str:
    out = _string(value, path, max_length=64)
    allowed = set(allowed)
    if out not in allowed:
        raise SceneError(
            f"{path}: {out!r} is not one of {', '.join(sorted(allowed))}"
        )
    return out


def _point(value: Any, n: int, path: str) -> tuple[float, ...]:
    items = _list(value, path)
    if len(items) != n:
        raise SceneError(f"{path}: expected {n} numbers, got {len(items)}")
    return tuple(_number(v, f"{path}[{i}]") for i, v in enumerate(items))


def scene_from_mapping(raw: Any, source_path: str | None = None) -> Scene:
    """Build and check a `Scene` from already-parsed JSON.

    Separate from `parse_scene` so a caller holding a dict — a test, a
    future HTTP endpoint, the RoomPlan importer checking its own output —
    can validate without a round trip through text.
    """
    root = _obj(raw, "scene")
    _check_keys(
        root,
        required=("schema_version", "room"),
        optional=("name", "description", "walls", "objects", "nodes", "people", "phone"),
        path="scene",
    )

    version = root["schema_version"]
    if isinstance(version, bool) or not isinstance(version, int):
        raise SceneError(f"scene.schema_version: expected an integer, got {_kind(version)}")
    if version != SCHEMA_VERSION:
        raise SceneError(
            f"scene.schema_version is {version}; this loader reads version "
            f"{SCHEMA_VERSION} only. Refusing rather than guessing at a "
            f"format it has not been told about."
        )

    room = _room_from(_obj(root["room"], "scene.room"))
    footprint = room.footprint

    walls = tuple(
        _wall_from(_obj(w, f"scene.walls[{i}]"), room, f"scene.walls[{i}]")
        for i, w in enumerate(_list(root.get("walls", []), "scene.walls"))
    )
    objects = tuple(
        _object_from(_obj(o, f"scene.objects[{i}]"), room, footprint, f"scene.objects[{i}]")
        for i, o in enumerate(_list(root.get("objects", []), "scene.objects"))
    )
    nodes = tuple(
        _node_from(_obj(n, f"scene.nodes[{i}]"), room, footprint, f"scene.nodes[{i}]")
        for i, n in enumerate(_list(root.get("nodes", []), "scene.nodes"))
    )
    _check_nodes_distinct(nodes)

    people = tuple(
        _person_from(_obj(p, f"scene.people[{i}]"), footprint, f"scene.people[{i}]")
        for i, p in enumerate(_list(root.get("people", []), "scene.people"))
    )

    phone = None
    if root.get("phone") is not None:
        phone = _phone_from(_obj(root["phone"], "scene.phone"), room, footprint, "scene.phone")

    return Scene(
        room=room,
        schema_version=version,
        name=_string(root["name"], "scene.name") if "name" in root else None,
        description=(
            _string(root["description"], "scene.description", max_length=4000)
            if "description" in root
            else None
        ),
        walls=walls,
        objects=objects,
        nodes=nodes,
        people=people,
        phone=phone,
        source_path=source_path,
    )


def _room_from(raw: dict) -> SceneRoom:
    _check_keys(
        raw,
        required=("width_m", "depth_m", "height_m"),
        optional=("polygon_m",),
        path="scene.room",
    )
    width = _positive(raw["width_m"], "scene.room.width_m")
    depth = _positive(raw["depth_m"], "scene.room.depth_m")
    height = _positive(raw["height_m"], "scene.room.height_m")

    polygon: tuple[tuple[float, float], ...] = ()
    if "polygon_m" in raw:
        verts = _list(raw["polygon_m"], "scene.room.polygon_m")
        if len(verts) < 3:
            raise SceneError(
                f"scene.room.polygon_m: a floor needs at least 3 vertices, got {len(verts)}"
            )
        polygon = tuple(
            _point(v, 2, f"scene.room.polygon_m[{i}]") for i, v in enumerate(verts)
        )  # type: ignore[assignment]
        _check_polygon(polygon, width, depth)

    return SceneRoom(width_m=width, depth_m=depth, height_m=height, polygon_m=polygon)


def _check_polygon(
    polygon: Sequence[tuple[float, float]], width: float, depth: float
) -> None:
    """A polygon has to agree with the width and depth it is declared with.

    Not a schema's business and not a nicety: `width_m` and `depth_m`
    stay the authority on the room's extent, so a viewer that cannot draw
    a polygon still gets the bounds right. If the two disagree there is
    no way to tell from the file which one is wrong.

    Self-intersection is NOT checked. A figure-of-eight floor passes here
    and then behaves strangely in the point-in-polygon test. It would
    cost an O(n^2) segment sweep to catch and no example needs it yet.
    """
    n = len(polygon)
    for i in range(n):
        x0, y0 = polygon[i]
        x1, y1 = polygon[(i + 1) % n]
        if math.hypot(x1 - x0, y1 - y0) <= CONTAIN_TOL_M:
            raise SceneError(
                f"scene.room.polygon_m[{i}] and [{(i + 1) % n}] are the same point"
            )

    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    bounds = ((min(xs), 0.0), (max(xs), width), (min(ys), 0.0), (max(ys), depth))
    names = ("min x", "max x", "min y", "max y")
    expected_names = ("0", "width_m", "0", "depth_m")
    for (got, want), what, want_name in zip(bounds, names, expected_names):
        if abs(got - want) > 1e-6:
            raise SceneError(
                f"scene.room.polygon_m: {what} is {got:g}, but {want_name} is "
                f"{want:g}. The polygon's bounding box must be the stated "
                f"width_m x depth_m."
            )

    area = _signed_area(polygon)
    if area <= 0.0:
        raise SceneError(
            f"scene.room.polygon_m: vertices must run counter-clockwise and "
            f"enclose an area; signed area is {area:g} m2"
        )


def _wall_from(raw: dict, room: SceneRoom, path: str) -> Wall:
    _check_keys(
        raw,
        required=("from_m", "to_m", "material", "thickness_m"),
        optional=("id", "height_m"),
        path=path,
    )
    start = _point(raw["from_m"], 2, f"{path}.from_m")
    end = _point(raw["to_m"], 2, f"{path}.to_m")
    if math.hypot(end[0] - start[0], end[1] - start[1]) <= CONTAIN_TOL_M:
        raise SceneError(f"{path}: from_m and to_m are the same point")

    # Wall endpoints are checked against the bounding box, not the
    # polygon: a wall lies ON the boundary, where point-in-polygon is
    # undefined, and an interior partition lies inside it.
    for name, point in (("from_m", start), ("to_m", end)):
        _check_in_box(point[0], point[1], room, f"{path}.{name}")

    thickness = _positive(raw["thickness_m"], f"{path}.thickness_m")
    if thickness > 2.0:
        raise SceneError(f"{path}.thickness_m: {thickness:g} m is not a wall")

    height = None
    if "height_m" in raw:
        height = _positive(raw["height_m"], f"{path}.height_m")
        if height > room.height_m + CONTAIN_TOL_M:
            raise SceneError(
                f"{path}.height_m: {height:g} m is taller than the room's "
                f"{room.height_m:g} m"
            )

    return Wall(
        from_m=(start[0], start[1]),
        to_m=(end[0], end[1]),
        material=_enum(raw["material"], WALL_MATERIALS, f"{path}.material"),
        thickness_m=thickness,
        id=_string(raw["id"], f"{path}.id", max_length=64) if "id" in raw else None,
        height_m=height,
    )


def _object_from(
    raw: dict, room: SceneRoom, footprint: Sequence[tuple[float, float]], path: str
) -> SceneObject:
    _check_keys(
        raw,
        required=("type", "centre_m", "size_m", "material"),
        optional=("id", "rotation_deg"),
        path=path,
    )
    centre = _point(raw["centre_m"], 3, f"{path}.centre_m")
    size = _list(raw["size_m"], f"{path}.size_m")
    if len(size) != 3:
        raise SceneError(f"{path}.size_m: expected 3 numbers, got {len(size)}")
    sizes = tuple(_positive(v, f"{path}.size_m[{i}]") for i, v in enumerate(size))

    rotation = 0.0
    if "rotation_deg" in raw:
        rotation = _number(raw["rotation_deg"], f"{path}.rotation_deg")
        if not -360.0 <= rotation <= 360.0:
            raise SceneError(
                f"{path}.rotation_deg: {rotation:g} is outside -360..360"
            )

    obj = SceneObject(
        type=_enum(raw["type"], OBJECT_TYPES, f"{path}.type"),
        centre_m=(centre[0], centre[1], centre[2]),
        size_m=(sizes[0], sizes[1], sizes[2]),
        material=_enum(raw["material"], OBJECT_MATERIALS, f"{path}.material"),
        rotation_deg=rotation,
        id=_string(raw["id"], f"{path}.id", max_length=64) if "id" in raw else None,
    )

    for i, (x, y) in enumerate(obj.footprint):
        if not _contains_point(footprint, x, y, CONTAIN_TOL_M):
            raise SceneError(
                f"{path}: corner {i} of the {obj.type} is at "
                f"({x:.3f}, {y:.3f}) m, outside the room"
            )
    low, high = obj.z_extent_m
    if low < -CONTAIN_TOL_M or high > room.height_m + CONTAIN_TOL_M:
        raise SceneError(
            f"{path}: the {obj.type} spans {low:.3f} to {high:.3f} m in z, "
            f"outside the room's 0 to {room.height_m:g} m"
        )
    return obj


def _node_from(
    raw: dict, room: SceneRoom, footprint: Sequence[tuple[float, float]], path: str
) -> SceneNode:
    _check_keys(raw, required=("id", "pos_m"), optional=("label",), path=path)
    node_id = raw["id"]
    if isinstance(node_id, bool) or not isinstance(node_id, int):
        raise SceneError(f"{path}.id: expected an integer, got {_kind(node_id)}")
    if not 0 <= node_id <= 255:
        # 0..255 because the wire format and the simulator both carry a
        # node id in one byte.
        raise SceneError(f"{path}.id: {node_id} is outside 0..255")

    pos = _point(raw["pos_m"], 3, f"{path}.pos_m")
    if not _contains_point(footprint, pos[0], pos[1], CONTAIN_TOL_M):
        raise SceneError(
            f"{path}: node {node_id} is at ({pos[0]:.3f}, {pos[1]:.3f}) m, "
            f"outside the room"
        )
    if pos[2] < -CONTAIN_TOL_M or pos[2] > room.height_m + CONTAIN_TOL_M:
        raise SceneError(
            f"{path}: node {node_id} is at z = {pos[2]:.3f} m, outside the "
            f"room's 0 to {room.height_m:g} m"
        )
    return SceneNode(
        id=node_id,
        pos_m=(pos[0], pos[1], pos[2]),
        label=_string(raw["label"], f"{path}.label", max_length=64) if "label" in raw else None,
    )


def _check_nodes_distinct(nodes: Sequence[SceneNode]) -> None:
    seen: dict[int, int] = {}
    for i, node in enumerate(nodes):
        if node.id in seen:
            raise SceneError(
                f"scene.nodes[{i}]: node id {node.id} is already used by "
                f"scene.nodes[{seen[node.id]}]"
            )
        seen[node.id] = i

    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            a, b = nodes[i].pos_m, nodes[j].pos_m
            gap = math.dist(a, b)
            if gap <= NODE_MIN_SEPARATION_M:
                # A zero-length link has no geometry: its ellipse
                # degenerates and 1 / sqrt(0) is not a weight.
                raise SceneError(
                    f"scene.nodes[{i}] and scene.nodes[{j}] (ids {nodes[i].id} "
                    f"and {nodes[j].id}) are {gap:g} m apart, which is the same "
                    f"point; the link between them would be degenerate"
                )


def _person_from(
    raw: dict, footprint: Sequence[tuple[float, float]], path: str
) -> Person:
    _check_keys(raw, required=("pos_m",), optional=("id", "height_m"), path=path)
    pos = _point(raw["pos_m"], 2, f"{path}.pos_m")
    if not _contains_point(footprint, pos[0], pos[1], CONTAIN_TOL_M):
        raise SceneError(
            f"{path}: person is at ({pos[0]:.3f}, {pos[1]:.3f}) m, outside the room"
        )
    height = 1.7
    if "height_m" in raw:
        height = _positive(raw["height_m"], f"{path}.height_m")
        if height > 2.5:
            raise SceneError(f"{path}.height_m: {height:g} m is not a person")
    return Person(
        pos_m=(pos[0], pos[1]),
        height_m=height,
        id=_string(raw["id"], f"{path}.id", max_length=64) if "id" in raw else None,
    )


def _phone_from(
    raw: dict, room: SceneRoom, footprint: Sequence[tuple[float, float]], path: str
) -> Phone:
    _check_keys(raw, required=("pos_m",), optional=("id",), path=path)
    pos = _point(raw["pos_m"], 3, f"{path}.pos_m")
    if not _contains_point(footprint, pos[0], pos[1], CONTAIN_TOL_M):
        raise SceneError(
            f"{path}: phone is at ({pos[0]:.3f}, {pos[1]:.3f}) m, outside the room"
        )
    if pos[2] < -CONTAIN_TOL_M or pos[2] > room.height_m + CONTAIN_TOL_M:
        raise SceneError(
            f"{path}: phone is at z = {pos[2]:.3f} m, outside the room's 0 to "
            f"{room.height_m:g} m"
        )
    return Phone(
        pos_m=(pos[0], pos[1], pos[2]),
        id=_string(raw["id"], f"{path}.id", max_length=64) if "id" in raw else None,
    )


def parse_scene(text: str, source_path: str | None = None) -> Scene:
    """Parse and check a scene from JSON text.

    `NaN` and `Infinity` are rejected: CPython's `json` accepts them by
    default as an extension, and a room 4.2 m wide and NaN deep would
    otherwise load and then poison every distance computed from it.
    """
    try:
        raw = json.loads(text, parse_constant=_reject_constant)
    except json.JSONDecodeError as exc:
        where = f"{source_path}: " if source_path else ""
        raise SceneError(f"{where}not valid JSON: {exc}") from None
    return scene_from_mapping(raw, source_path=source_path)


def load_scene(path: str | os.PathLike[str]) -> Scene:
    """Read, parse and check a scene file."""
    path = os.fspath(path)
    with open(path, "r", encoding="utf-8") as handle:
        return parse_scene(handle.read(), source_path=path)


def scenes_dir() -> str:
    """The repository's `scenes/` directory, beside the package."""
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scenes")


# --------------------------------------------------------------------------
# Conversion to the 2-D localization types
# --------------------------------------------------------------------------


def to_room(scene: Scene) -> Room:
    """The scene's floor plan as a `localize.geometry.Room`.

    THE ONE PLACE THE TWO MEANINGS OF "HEIGHT" MEET.

        scene.room.width_m   -> Room.width_m     both are x
        scene.room.depth_m   -> Room.height_m    both are the second
                                                 HORIZONTAL axis
        scene.room.height_m  -> dropped          elevation; the 2-D type
                                                 has nowhere to put it

    `Room.height_m` is not elevation. Its own docstring says so, and this
    function exists so that nothing else in the codebase has to remember
    it. The failure it prevents is silent: pass the elevation instead and
    a 4.2 x 3.6 m bedroom becomes a 4.2 x 2.7 m one that still loads,
    still solves and is wrong by a quarter of a metre everywhere.

    An L-shaped room flattens to its bounding box here, because
    `geometry.Room` is a rectangle. Pixels in the missing arm are part of
    the grid and the caller is responsible for knowing that; use
    `scene.room.contains` to test a point against the real floor.
    """
    return Room(width_m=scene.room.width_m, height_m=scene.room.depth_m)


def to_nodes(scene: Scene) -> list[Node]:
    """The scene's nodes as `localize.geometry.Node`, dropping elevation.

    The localization stack is 2-D: nodes and chests are assumed to be at
    roughly one height. `Scene.node_z_spread_m` is the number that says
    whether a given scene honours that; the three example scenes mount
    every node at 1.2 m, so the spread is zero and the projection costs
    nothing.
    """
    return [Node(node_id=n.id, x=n.pos_m[0], y=n.pos_m[1]) for n in scene.nodes]


def to_geometry(scene: Scene, pitch_m: float = 0.5, lambda_m: float = 0.3) -> Geometry:
    """Scene -> the full `localize.geometry.Geometry` deployment.

    Nothing new happens here: it is `to_room` and `to_nodes` with the
    grid and the ellipse width the caller wants. It is separate so that
    the depth/elevation mapping stays in one function, and so a caller
    sweeping `lambda_m` does not re-derive the room each time.

    `Geometry` needs at least two nodes, so a scene with fewer is a
    `SceneError` here rather than a `ValueError` from three frames down.
    """
    if len(scene.nodes) < 2:
        raise SceneError(
            f"a deployment needs at least 2 nodes to form a link; "
            f"{scene.name or scene.source_path or 'this scene'} has "
            f"{len(scene.nodes)}"
        )
    room = to_room(scene)
    return Geometry(
        room=room,
        nodes=to_nodes(scene),
        grid=PixelGrid(room, pitch_m),
        lambda_m=lambda_m,
    )
