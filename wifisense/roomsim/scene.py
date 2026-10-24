"""Scene loading and validation for Room Sim (spec/roomsim.md §3.1).

A scene is read only (ADR-008 decision 1): this module parses
`spec/scene.schema.json` files with the standard `json` module, ports
every check of `rust/crates/saans-scene/src/validate.rs` (deserialisation
included: required fields, types, enum names, u8 node ids, duplicate keys
rejected and unknown keys ignored, as serde does), and adds the §3.1
rejections: two nodes at one position, a node strictly inside an object's
box (§7.4), more than 100 people. The last §3.1 rejection, a node within a
wall's thickness, needs the configuration (implied walls take
`implied_wall_thickness_m`), so `world.World` makes it, naming `nodes`.

Room geometry and furniture come from the scene file and nowhere else;
nothing here is, or may be, inferred from WiFi.

The validator's own geometry (`_on_segment`, `_point_in_polygon`, ...)
follows `saans_scene::geom`, with `sqrt(dx*dx + dy*dy)` where Rust uses
`hypot` (Room Sim never uses `hypot`, §1); the two agree except within an
ulp of a 1e-9 tolerance boundary.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass

from .geometry import Box

__all__ = [
    "SceneError", "Wall", "Object", "Node", "Person", "Phone", "Room", "Scene",
    "load_scene", "parse_scene", "validate_scene",
    "SCHEMA_VERSION", "MAX_PEOPLE", "WALL_MATERIALS", "OBJECT_MATERIALS", "OBJECT_KINDS",
]

SCHEMA_VERSION = 1
EPS_M = 1e-9
MAX_WALL_THICKNESS_M = 1.0
NODE_WALL_CLEARANCE_M = 0.01
MAX_PEOPLE = 100

PROVENANCES = ("surveyed", "roomplan_import", "hand_written", "generated")
WALL_MATERIALS = ("concrete", "brick", "drywall", "glass", "wood")
OBJECT_KINDS = ("bed", "table", "chair", "almirah", "sofa", "door", "window")
OBJECT_MATERIALS = ("wood", "metal", "fabric", "glass")


class SceneError(ValueError):
    """A scene Room Sim must reject. `problems` lists every one found."""

    def __init__(self, label: str, problems: list[str]) -> None:
        self.label = label
        self.problems = list(problems)
        lines = "\n".join(f"  - {p}" for p in self.problems)
        super().__init__(f"{label}: {len(self.problems)} problem(s) with this scene:\n{lines}")


@dataclass(frozen=True)
class Room:
    w_m: float
    d_m: float
    h_m: float
    polygon: tuple | None = None


@dataclass(frozen=True)
class Wall:
    a: tuple
    b: tuple
    material: str
    thickness_m: float
    height_m: float | None = None


@dataclass(frozen=True)
class Object:
    kind: str
    pos: tuple
    size: tuple
    yaw_deg: float
    material: str

    def box(self) -> Box:
        return Box(self.pos, self.size, self.yaw_deg)

    @property
    def is_opening(self) -> bool:
        """Doors and windows are openings in a wall, not furniture (§6.5)."""
        return self.kind in ("door", "window")


@dataclass(frozen=True)
class Node:
    id: int
    pos: tuple


@dataclass(frozen=True)
class Person:
    """A person's scene start, or None. Their scene `id` is deliberately
    not kept: no Person.id appears in any Room Sim output (§3.1)."""

    start: tuple | None


@dataclass(frozen=True)
class Phone:
    start: tuple


@dataclass(frozen=True)
class Scene:
    """A validated scene. `nodes` are sorted by id ascending (§3.1)."""

    name: str
    provenance: str
    room: Room
    walls: tuple
    objects: tuple
    nodes: tuple
    people: tuple
    phone: Phone | None
    source: str = "<memory>"
    sha256: str = ""

    def footprint(self) -> list:
        """`room.polygon`, or the `w x d` rectangle (§3.1)."""
        if self.room.polygon is not None:
            return [tuple(p) for p in self.room.polygon]
        w, d = self.room.w_m, self.room.d_m
        return [(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)]

    def node(self, node_id: int) -> Node:
        for n in self.nodes:
            if n.id == node_id:
                return n
        raise KeyError(node_id)


# --------------------------------------------------------------------------
# Deserialisation (serde semantics)
# --------------------------------------------------------------------------


class _Bad(Exception):
    pass


def _reject_constant(name: str):
    raise _Bad(f"{name} is not valid JSON (serde_json rejects it)")


def _no_duplicates(pairs):
    out = {}
    for k, v in pairs:
        if k in out:
            raise _Bad(f"duplicate field `{k}`")
        out[k] = v
    return out


def _num(v, where: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _Bad(f"{where}: expected a number")
    return float(v)


def _uint(v, where: str, hi: int) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise _Bad(f"{where}: expected an integer")
    if v < 0 or v > hi:
        raise _Bad(f"{where}: {v} is out of range 0..{hi}")
    return v


def _str(v, where: str) -> str:
    if not isinstance(v, str):
        raise _Bad(f"{where}: expected a string")
    return v


def _enum(v, where: str, allowed) -> str:
    s = _str(v, where)
    if s not in allowed:
        raise _Bad(f"{where}: unknown variant `{s}`, expected one of {', '.join(allowed)}")
    return s


def _obj(v, where: str) -> dict:
    if not isinstance(v, dict):
        raise _Bad(f"{where}: expected an object")
    return v


def _req(d: dict, key: str, where: str):
    if key not in d:
        raise _Bad(f"{where}: missing field `{key}`")
    return d[key]


def _tuple(v, n: int, where: str) -> tuple:
    if not isinstance(v, list) or len(v) != n:
        raise _Bad(f"{where}: expected an array of {n} numbers")
    return tuple(_num(x, f"{where}[{i}]") for i, x in enumerate(v))


def _seq(d: dict, key: str, where: str) -> list:
    if key not in d:
        return []
    v = d[key]
    if not isinstance(v, list):
        raise _Bad(f"{where}.{key}: expected an array")
    return v


def _opt(d: dict, key: str):
    """An `Option` field: absent and `null` are both None (serde)."""
    return d.get(key)


def _deserialize(doc) -> dict:
    top = _obj(doc, "scene")
    out: dict = {}
    out["schema_version"] = _uint(_req(top, "schema_version", "scene"), "schema_version", 0xFFFFFFFF)
    out["name"] = _str(_req(top, "name", "scene"), "name")
    out["provenance"] = _enum(_req(top, "provenance", "scene"), "provenance", PROVENANCES)
    room = _obj(_req(top, "room", "scene"), "room")
    poly = _opt(room, "polygon")
    if poly is not None:
        if not isinstance(poly, list):
            raise _Bad("room.polygon: expected an array")
        poly = tuple(_tuple(p, 2, f"room.polygon[{i}]") for i, p in enumerate(poly))
    out["room"] = Room(
        w_m=_num(_req(room, "w_m", "room"), "room.w_m"),
        d_m=_num(_req(room, "d_m", "room"), "room.d_m"),
        h_m=_num(_req(room, "h_m", "room"), "room.h_m"),
        polygon=poly,
    )
    walls = []
    for i, w in enumerate(_seq(top, "walls", "scene")):
        where = f"walls[{i}]"
        w = _obj(w, where)
        h = _opt(w, "height_m")
        walls.append(Wall(
            a=_tuple(_req(w, "from", where), 2, f"{where}.from"),
            b=_tuple(_req(w, "to", where), 2, f"{where}.to"),
            material=_enum(_req(w, "material", where), f"{where}.material", WALL_MATERIALS),
            thickness_m=_num(_req(w, "thickness_m", where), f"{where}.thickness_m"),
            height_m=None if h is None else _num(h, f"{where}.height_m"),
        ))
    out["walls"] = tuple(walls)
    objects = []
    for i, o in enumerate(_seq(top, "objects", "scene")):
        where = f"objects[{i}]"
        o = _obj(o, where)
        if o.get("id") is not None:
            _str(o["id"], f"{where}.id")
        yaw = o.get("yaw_deg", 0.0)
        objects.append(Object(
            kind=_enum(_req(o, "type", where), f"{where}.type", OBJECT_KINDS),
            pos=_tuple(_req(o, "pos", where), 3, f"{where}.pos"),
            size=_tuple(_req(o, "size", where), 3, f"{where}.size"),
            yaw_deg=_num(yaw, f"{where}.yaw_deg"),
            material=_enum(_req(o, "material", where), f"{where}.material", OBJECT_MATERIALS),
        ))
    out["objects"] = tuple(objects)
    if "nodes" not in top:
        raise _Bad("scene: missing field `nodes`")
    nodes = []
    for i, n in enumerate(_seq(top, "nodes", "scene")):
        where = f"nodes[{i}]"
        n = _obj(n, where)
        nodes.append(Node(
            id=_uint(_req(n, "id", where), f"{where}.id", 255),
            pos=_tuple(_req(n, "pos", where), 3, f"{where}.pos"),
        ))
    out["nodes_file_order"] = nodes
    people = []
    for i, p in enumerate(_seq(top, "people", "scene")):
        where = f"people[{i}]"
        p = _obj(p, where)
        if p.get("id") is not None:
            _str(p["id"], f"{where}.id")
        s = _opt(p, "start")
        people.append(Person(start=None if s is None else _tuple(s, 2, f"{where}.start")))
    out["people"] = tuple(people)
    ph = _opt(top, "phone")
    if ph is not None:
        ph = _obj(ph, "phone")
        if ph.get("id") is not None:
            _str(ph["id"], "phone.id")
        out["phone"] = Phone(start=_tuple(_req(ph, "start", "phone"), 2, "phone.start"))
    else:
        out["phone"] = None
    return out


# --------------------------------------------------------------------------
# saans_scene::geom, ported
# --------------------------------------------------------------------------


def _len2(dx: float, dy: float) -> float:
    return math.sqrt(dx * dx + dy * dy)


def _cross(a, b, c) -> float:
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _on_segment(a, b, p) -> bool:
    ln = _len2(b[0] - a[0], b[1] - a[1])
    if ln <= EPS_M:
        return _len2(p[0] - a[0], p[1] - a[1]) <= EPS_M
    if abs(_cross(a, b, p) / ln) > EPS_M:
        return False
    t = ((p[0] - a[0]) * (b[0] - a[0]) + (p[1] - a[1]) * (b[1] - a[1])) / (ln * ln)
    return -EPS_M <= t <= 1.0 + EPS_M


def _segments_intersect(a, b, c, d) -> bool:
    d1, d2 = _cross(c, d, a), _cross(c, d, b)
    d3, d4 = _cross(a, b, c), _cross(a, b, d)
    if ((d1 > 0.0 and d2 < 0.0) or (d1 < 0.0 and d2 > 0.0)) and ((d3 > 0.0 and d4 < 0.0) or (d3 < 0.0 and d4 > 0.0)):
        return True
    return _on_segment(c, d, a) or _on_segment(c, d, b) or _on_segment(a, b, c) or _on_segment(a, b, d)


def _self_intersection(poly):
    n = len(poly)
    for i in range(n):
        a, b = poly[i], poly[(i + 1) % n]
        for j in range(i + 1, n):
            if j == i + 1 or (i == 0 and j == n - 1):
                continue
            c, d = poly[j], poly[(j + 1) % n]
            if _segments_intersect(a, b, c, d):
                return (i, j)
    return None


def _area(poly) -> float:
    n = len(poly)
    acc = 0.0
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        acc += x0 * y1 - x1 * y0
    return abs(acc) / 2.0


def _bbox(poly):
    x0 = y0 = math.inf
    x1 = y1 = -math.inf
    for x, y in poly:
        x0 = min(x0, x)
        y0 = min(y0, y)
        x1 = max(x1, x)
        y1 = max(y1, y)
    return ((x0, y0), (x1, y1))


def _point_in_polygon(poly, p) -> bool:
    n = len(poly)
    for i in range(n):
        if _on_segment(poly[i], poly[(i + 1) % n], p):
            return True
    res = False
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > p[1]) != (y1 > p[1]):
            t = (p[1] - y0) / (y1 - y0)
            if p[0] < x0 + t * (x1 - x0):
                res = not res
    return res


def _distance_to_segment(a, b, p) -> float:
    dx, dy = b[0] - a[0], b[1] - a[1]
    len2 = dx * dx + dy * dy
    if len2 <= EPS_M * EPS_M:
        return _len2(p[0] - a[0], p[1] - a[1])
    t = min(max(((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len2, 0.0), 1.0)
    return _len2(p[0] - (a[0] + t * dx), p[1] - (a[1] + t * dy))


def _point_inside_with_margin(poly, p, margin) -> bool:
    if not _point_in_polygon(poly, p):
        return False
    n = len(poly)
    for i in range(n):
        if _distance_to_segment(poly[i], poly[(i + 1) % n], p) < margin:
            return False
    return True


def _finite(t) -> bool:
    return all(math.isfinite(x) for x in t)


def _footprint_bbox(o: Object):
    t = (o.yaw_deg * math.pi) / 180.0
    c, s = abs(math.cos(t)), abs(math.sin(t))
    hx = (o.size[0] * c + o.size[1] * s) / 2.0
    hy = (o.size[0] * s + o.size[1] * c) / 2.0
    return ((o.pos[0] - hx, o.pos[1] - hy), (o.pos[0] + hx, o.pos[1] + hy))


def _object_inside(poly, o: Object) -> bool:
    if o.is_opening:
        return _point_in_polygon(poly, (o.pos[0], o.pos[1]))
    lo, hi = _footprint_bbox(o)
    t = (o.yaw_deg * math.pi) / 180.0
    c, s = math.cos(t), math.sin(t)
    hx, hy = o.size[0] / 2.0, o.size[1] / 2.0
    px, py = o.pos[0], o.pos[1]
    corners = [
        (px + hx * c - hy * s, py + hx * s + hy * c),
        (px - hx * c - hy * s, py - hx * s + hy * c),
        (px + hx * c + hy * s, py + hx * s - hy * c),
        (px - hx * c + hy * s, py - hx * s - hy * c),
        lo,
        hi,
    ]
    n = len(corners) if abs(o.yaw_deg) < EPS_M else 4
    return all(_point_in_polygon(poly, p) for p in corners[:n])


def _fmt(x: float) -> str:
    return f"{x:.3f}"


# --------------------------------------------------------------------------
# Validation: validate.rs, then the §3.1 extras
# --------------------------------------------------------------------------


def validate_scene(d: dict) -> list[str]:
    """Every problem with a deserialised scene, in reading order
    (`Scene::validate`), followed by the §3.1 Room Sim rejections."""
    errs: list[str] = []
    if d["schema_version"] != SCHEMA_VERSION:
        errs.append(f"schema_version {d['schema_version']} is not supported (this build reads version {SCHEMA_VERSION})")
    if d["name"].strip() == "":
        errs.append("name is empty: a scene is referred to by name in reports and the demo")
    r: Room = d["room"]
    bad_dim = False
    for fld, v in (("w_m", r.w_m), ("d_m", r.d_m), ("h_m", r.h_m)):
        if not math.isfinite(v) or v <= 0.0:
            errs.append(f"room.{fld} = {v} m: must be finite and greater than zero")
            bad_dim = True
    if bad_dim:
        return errs

    if r.polygon is not None:
        poly = r.polygon
        if len(poly) < 3:
            errs.append(f"room.polygon has {len(poly)} vertices: a floor outline needs at least 3")
            return errs
        for i, p in enumerate(poly):
            if not _finite(p):
                errs.append(f"room.polygon[{i}] is not a finite coordinate")
                return errs
        si = _self_intersection(poly)
        if si is not None:
            errs.append(f"room.polygon edges {si[0]} and {si[1]} cross or touch: the floor outline must be a simple polygon")
            return errs
        if _area(poly) <= EPS_M:
            errs.append("room.polygon encloses no area")
            return errs
        got = _bbox(poly)
        want = ((0.0, 0.0), (r.w_m, r.d_m))
        off = max(abs(got[0][0] - want[0][0]), abs(got[0][1] - want[0][1]),
                  abs(got[1][0] - want[1][0]), abs(got[1][1] - want[1][1]))
        if off > 1e-6:
            errs.append(
                f"room.polygon spans ({_fmt(got[0][0])}, {_fmt(got[0][1])})..({_fmt(got[1][0])}, {_fmt(got[1][1])}) "
                f"but room is {_fmt(r.w_m)} x {_fmt(r.d_m)} m: the outline's bounding box must be (0, 0)..(w_m, d_m)"
            )
        poly = list(poly)
    else:
        poly = [(0.0, 0.0), (r.w_m, 0.0), (r.w_m, r.d_m), (0.0, r.d_m)]

    for i, w in enumerate(d["walls"]):
        if not _finite(w.a) or not _finite(w.b):
            errs.append(f"walls[{i}] has a non-finite endpoint")
            continue
        if _len2(w.b[0] - w.a[0], w.b[1] - w.a[1]) <= EPS_M:
            errs.append(f"walls[{i}] has zero length")
        t = w.thickness_m
        if not math.isfinite(t) or t <= 0.0 or t > MAX_WALL_THICKNESS_M:
            errs.append(f"walls[{i}].thickness_m = {t}: must be finite, greater than zero and at most {MAX_WALL_THICKNESS_M} m")
        if w.height_m is not None:
            h = w.height_m
            if not math.isfinite(h) or h <= 0.0 or h > r.h_m + EPS_M:
                errs.append(f"walls[{i}].height_m = {h} m: must be finite, greater than zero and at most the room height {r.h_m} m")
        for p in (w.a, w.b):
            if not _point_in_polygon(poly, p):
                errs.append(f"walls[{i}] endpoint ({_fmt(p[0])}, {_fmt(p[1])}) is outside the room footprint")

    for i, o in enumerate(d["objects"]):
        if not _finite(o.pos) or not _finite(o.size) or not math.isfinite(o.yaw_deg):
            errs.append(f"objects[{i}] has a non-finite position, size or rotation")
            continue
        if o.size[0] <= 0.0 or o.size[1] <= 0.0 or o.size[2] <= 0.0:
            errs.append(f"objects[{i}].size = ({o.size[0]}, {o.size[1]}, {o.size[2]}) m: every extent must be finite and greater than zero")
            continue
        if not _object_inside(poly, o):
            lo, hi = _footprint_bbox(o)
            errs.append(f"objects[{i}] covers ({_fmt(lo[0])}, {_fmt(lo[1])})..({_fmt(hi[0])}, {_fmt(hi[1])}) m, which leaves the room footprint")
        z0 = o.pos[2] - o.size[2] / 2.0
        z1 = o.pos[2] + o.size[2] / 2.0
        if z0 < -EPS_M:
            errs.append(f"objects[{i}] reaches {z0:.3f} m, below the floor")
        if z1 > r.h_m + EPS_M:
            errs.append(f"objects[{i}] reaches {z1:.3f} m, above the {r.h_m} m ceiling")

    nodes = d["nodes_file_order"]
    if not nodes:
        errs.append("nodes is empty: a scene with no ESP32 has nothing to simulate")
    seen: list[int] = []
    for i, n in enumerate(nodes):
        if n.id in seen:
            errs.append(f"two nodes share id {n.id}: ids address links as (tx, rx) and must be unique")
        else:
            seen.append(n.id)
        if not _finite(n.pos):
            errs.append(f"nodes[{i}] (id {n.id}) has a non-finite position")
            continue
        if not _point_inside_with_margin(poly, (n.pos[0], n.pos[1]), NODE_WALL_CLEARANCE_M):
            errs.append(f"node {n.id} at ({_fmt(n.pos[0])}, {_fmt(n.pos[1])}) m is outside the room footprint or within {NODE_WALL_CLEARANCE_M} m of a wall")
        if n.pos[2] < 0.0 or n.pos[2] > r.h_m + EPS_M:
            errs.append(f"node {n.id} is at z = {n.pos[2]} m, outside the room's 0..{r.h_m} m height")

    for i, p in enumerate(d["people"]):
        if p.start is not None:
            if not _finite(p.start) or not _point_in_polygon(poly, p.start):
                errs.append(f"people[{i}].start ({_fmt(p.start[0])}, {_fmt(p.start[1])}) m is outside the room footprint")
    ph = d["phone"]
    if ph is not None:
        if not _finite(ph.start) or not _point_in_polygon(poly, ph.start):
            errs.append(f"phone.start ({_fmt(ph.start[0])}, {_fmt(ph.start[1])}) m is outside the room footprint")

    # §3.1: Room Sim's own rejections.
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if nodes[i].pos == nodes[j].pos:
                errs.append(f"nodes {nodes[i].id} and {nodes[j].id} are at the same position: a link needs two places")
    for n in nodes:
        if not _finite(n.pos):
            continue
        for k, o in enumerate(d["objects"]):
            if not (_finite(o.pos) and _finite(o.size) and math.isfinite(o.yaw_deg)):
                continue
            if o.box().strictly_inside(n.pos):
                errs.append(f"node {n.id} is strictly inside objects[{k}] ({o.kind}): a radio cannot be inside furniture")
    if len(d["people"]) > MAX_PEOPLE:
        errs.append(f"people has {len(d['people'])} entries: Room Sim simulates at most {MAX_PEOPLE}")
    return errs


def parse_scene(text: str | bytes, label: str = "<memory>") -> Scene:
    """Parse and validate scene JSON. Raises `SceneError` listing every
    problem."""
    raw = text.encode("utf-8") if isinstance(text, str) else bytes(text)
    try:
        doc = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates, parse_constant=_reject_constant)
        d = _deserialize(doc)
    except _Bad as e:
        raise SceneError(label, [str(e)]) from None
    except (ValueError, UnicodeDecodeError) as e:
        raise SceneError(label, [f"not JSON: {e}"]) from None
    errs = validate_scene(d)
    if errs:
        raise SceneError(label, errs)
    nodes = tuple(sorted(d["nodes_file_order"], key=lambda n: n.id))
    return Scene(
        name=d["name"], provenance=d["provenance"], room=d["room"], walls=d["walls"],
        objects=d["objects"], nodes=nodes, people=d["people"], phone=d["phone"],
        source=label, sha256=hashlib.sha256(raw).hexdigest(),
    )


def load_scene(path: str | os.PathLike[str]) -> Scene:
    """Read, parse and validate a scene file."""
    path = os.fspath(path)
    with open(path, "rb") as fh:
        raw = fh.read()
    return parse_scene(raw, label=path)
