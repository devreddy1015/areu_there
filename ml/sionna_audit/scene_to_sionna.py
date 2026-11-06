"""Room Sim scene -> Sionna RT 2.2.0 scene, for the Phase 10a audit (SIM).

Builds a Sionna RT scene programmatically from a Room Sim scene file
(spec/scene.schema.json) and a Room Sim configuration (spec/roomsim.md
§3.2): a Mitsuba XML file plus one ASCII PLY mesh per shape, written to a
directory, then loaded with `sionna.rt.load_scene(..., merge_shapes=False)`
so every wall, the floor, the ceiling and every piece of furniture stays its
own scene object and every Sionna path can be traced back to Room Sim's
surface or object index.

The geometry is read from Room Sim's own resolved world
(`wifisense.roomsim.world.World`, read-only), so implied walls, wall
heights, the footprint, the object frames (§3.1) and the node positions are
exactly the ones Room Sim uses. What maps exactly and what does not
(spec/sionna_audit.md §3 lists every item):

* **Walls** are single flat quads (Sionna's own model for a wall: a surface
  carrying an ITU-R P.2040 slab material with the wall's thickness). A wall
  is placed on the face Room Sim reflects from (§6.1: `t/2` from the
  centreline on the nodes' side), spanning Room Sim's face line exactly
  (untrimmed at corners, as §6.1 says) from `z = 0` to its height. A wall
  with nodes on both sides has no single such face and is placed on its
  centreline (an approximation, recorded in `approximations`).
* **Floor and ceiling** are the footprint polygon at `z = 0` and `z = H`
  (ear-clipped into triangles), with `floor_material`/`ceiling_material`
  and `slab_thickness_m`.
* **Materials**: Room Sim's ITU names map one-to-one onto Sionna's
  `ITURadioMaterial` types (`drywall` -> `plasterboard`); the constants are
  the same ITU-R P.2040-3 Table 3 rows (checked in the tests).
* **Furniture** (every object but doors and windows, which Room Sim ignores,
  §6.5) becomes a closed box mesh with its material (the bottom face is
  dropped when it lies on the floor, to avoid two coplanar surfaces). Room
  Sim's object model has no exact Sionna equivalent (single-interface
  reflection *and* a fixed blockage loss per crossing): `object_model`
  chooses `"halfspace"` (default: a thickness at which the slab reflection
  equals Room Sim's single-interface `G` to 1e-4, so the box is opaque) or
  `"panel"` (thin physical panels). `fabric` has no ITU material; it becomes
  a custom non-reflecting material (`eps_r = 1`) whose conductivity gives
  Room Sim's 3 dB per crossing at normal incidence.
* **Antennas**: Sionna's `iso` pattern with `V` polarisation at every node,
  Room Sim's "isotropic and vertically polarised" antennas (§16); the LoS
  coefficient is then exactly `LAMBDA / (FOUR_PI * L)` in both.
* **Frequencies**: the scene frequency is Room Sim's carrier (materials are
  evaluated there, as Room Sim does, §16); the 52 LLTF subcarriers are
  `carrier + k * spacing`, `k = -26..-1, 1..26` (§4).

Everything this module builds is SIM. Sionna and Mitsuba are imported lazily
(`configure_backend`), so the geometry and material code runs without them.
"""

from __future__ import annotations

import glob
import math
import os
import re
from dataclasses import dataclass, field

import numpy as np

from wifisense.roomsim import materials as rs_materials
from wifisense.roomsim.config import Config
from wifisense.roomsim.numerics import EPS0, TWO_PI, C
from wifisense.roomsim.world import World

__all__ = [
    "LABEL", "WALL_ITU", "OBJECT_ITU", "OBJECT_MODELS", "PANEL_THICKNESS_M", "HALFSPACE_RESIDUAL",
    "LLTF_BINS", "lltf_frequencies", "MaterialSpec", "ShapeSpec", "ConvertedScene", "convert",
    "write_scene", "triangulate", "box_mesh", "fabric_conductivity", "halfspace_thickness",
    "find_libllvm", "configure_backend", "load_sionna_scene", "SionnaLink", "solve_paths",
    "INTERACTION",
]

LABEL = "SIM"

#: Room Sim wall/floor/ceiling material -> Sionna ITU material type (§5.1).
WALL_ITU = {"concrete": "concrete", "brick": "brick", "drywall": "plasterboard", "glass": "glass", "wood": "wood"}
#: Room Sim object material -> Sionna ITU material type; fabric has none (§5.2).
OBJECT_ITU = {"metal": "metal", "wood": "wood", "glass": "glass", "fabric": None}
OBJECT_MODELS = ("halfspace", "panel")
#: `object_model="panel"`: physical panel thickness per object material, m
#: (sheet metal, a furniture board, a glass top). SIM assumptions.
PANEL_THICKNESS_M = {"metal": 0.002, "wood": 0.02, "glass": 0.01}
#: `object_model="halfspace"`: the slab's internal round trip `|e^{-j2q}|`
#: at normal incidence, so its reflection equals the single interface `G`
#: to this relative accuracy at every angle. At 1e-12 a face also passes
#: `|T|^2 < 1e-12`, so Sionna never samples a refraction into it (a rarely
#: sampled one would only fill its candidate hash table, whose collisions
#: drop real candidates at random).
HALFSPACE_RESIDUAL = 1e-12
#: Fabric's stand-in slab thickness, m (its conductivity is solved for).
FABRIC_THICKNESS_M = 0.1

#: The 52 LLTF bins (§4): k = -26..-1, 1..26.
LLTF_BINS = tuple(list(range(-26, 0)) + list(range(1, 27)))

#: Sionna's InteractionType codes (sionna/rt/constants.py).
INTERACTION = {"none": 0, "specular": 1, "diffuse": 2, "refraction": 4, "diffraction": 8}
INVALID_SHAPE = 0xFFFFFFFF


def lltf_frequencies(carrier_hz: float, spacing_hz: float = 312500.0) -> np.ndarray:
    """The 52 LLTF subcarrier frequencies, Hz, `carrier + float(k) * spacing`
    for `k` in `LLTF_BINS` (spec/roomsim.md §4), ascending."""
    return np.array([carrier_hz + float(k) * spacing_hz for k in LLTF_BINS], dtype=np.float64)


# -- materials ---------------------------------------------------------------


@dataclass(frozen=True)
class MaterialSpec:
    """One Sionna radio material. `itu_type` names an ITU-R P.2040 material
    (`ITURadioMaterial`); otherwise `eps_r`/`sigma` define a custom
    `RadioMaterial` (fabric only)."""

    id: str
    itu_type: str | None
    thickness_m: float
    eps_r: float | None = None
    sigma_s_m: float | None = None
    scattering_coefficient: float = 0.0
    source: str = ""          # what in Room Sim it stands for


@dataclass
class ShapeSpec:
    """One Sionna scene object. `role` is wall/floor/ceiling/object;
    `index` the Room Sim wall index or object index (-1 for floor/ceiling);
    `surface` Room Sim's §6.1 surface index (-1 for objects)."""

    name: str
    role: str
    index: int
    surface: int
    material: str
    vertices: np.ndarray
    faces: np.ndarray
    note: str = ""


@dataclass
class ConvertedScene:
    """A Room Sim scene converted for Sionna RT (SIM)."""

    name: str
    carrier_hz: float
    spacing_hz: float
    wavelength_m: float
    height_m: float
    footprint: list
    nodes: dict                # node id -> (x, y, z), ids ascending
    shapes: list               # ShapeSpec
    materials: dict            # id -> MaterialSpec
    surface_names: dict        # Room Sim §6.1 surface index -> shape name
    object_names: dict         # Room Sim object index -> shape name (furniture only)
    approximations: list = field(default_factory=list)
    options: dict = field(default_factory=dict)
    label: str = LABEL

    @property
    def frequencies(self) -> np.ndarray:
        return lltf_frequencies(self.carrier_hz, self.spacing_hz)

    def shape(self, name: str) -> ShapeSpec:
        for s in self.shapes:
            if s.name == name:
                return s
        raise KeyError(name)


def _eta(eps_r: float, sigma: float, carrier_hz: float):
    """Room Sim's complex permittivity `(er, -(sigma / ((TWO_PI * f) * EPS0)))`."""
    return (eps_r, -(sigma / ((TWO_PI * carrier_hz) * EPS0)))


def halfspace_thickness(material: str, carrier_hz: float, residual: float = HALFSPACE_RESIDUAL) -> float:
    """Slab thickness at which the internal round trip `|e^{-j2q}|` at normal
    incidence is `residual`, so the §5.1 slab `R` equals the single-interface
    `G` to that relative accuracy at every angle (the loss per unit length
    is smallest at normal incidence). `material` is an ITU table key."""
    et = rs_materials.eta(material, carrier_hz)
    s, _g_te, _g_tm = rs_materials.fresnel(et, 1.0)
    k0 = TWO_PI * carrier_hz / C
    att = abs(s[1])                 # |Im sqrt(eta)|, Np per unit k0*d
    if att <= 0.0:
        raise ValueError(f"{material} is lossless: no thickness makes it a half-space")
    return math.log(1.0 / residual) / (2.0 * k0 * att)


def fabric_conductivity(loss_db_per_crossing: float, carrier_hz: float,
                        thickness_m: float = FABRIC_THICKNESS_M) -> float:
    """Conductivity (S/m) of a non-reflecting stand-in for fabric
    (`eps_r = 1`): two faces of a box, each a slab of `thickness_m`, lose
    `loss_db_per_crossing` together at normal incidence, i.e. one face loses
    half of it. Solved by bisection on Room Sim's own §5.1 slab `T`."""
    lam = C / carrier_hz
    target = 10.0 ** (-(loss_db_per_crossing / 2.0) / 20.0)     # |T| of one face

    def t_abs(sigma: float) -> float:
        et = _eta(1.0, sigma, carrier_hz)
        s, g_te, _ = rs_materials.fresnel(et, 1.0)
        _R, T, _q0 = rs_materials.slab(g_te, s, 1.0, thickness_m, lam)
        return math.sqrt(T[0] * T[0] + T[1] * T[1])

    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if t_abs(mid) > target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# -- geometry ----------------------------------------------------------------


def _polygon_area(poly) -> float:
    a = 0.0
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        a += x0 * y1 - x1 * y0
    return 0.5 * a


def _cross2(o, a, b) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def triangulate(poly) -> list:
    """Ear-clipping triangulation of a simple polygon `[(x, y), ...]`
    (either orientation). Returns index triples, counter-clockwise.
    Collinear vertices are skipped."""
    pts = [(float(x), float(y)) for x, y in poly]
    n = len(pts)
    if n < 3:
        raise ValueError("a polygon needs 3 vertices")
    idx = list(range(n)) if _polygon_area(pts) > 0.0 else list(range(n - 1, -1, -1))
    # Drop collinear vertices (they add nothing and are never ears).
    changed = True
    while changed and len(idx) > 3:
        changed = False
        for k in range(len(idx)):
            a, b, c = pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % len(idx)]]
            if abs(_cross2(a, b, c)) <= 1e-12:
                del idx[k]
                changed = True
                break
    tris = []
    while len(idx) > 3:
        m = len(idx)
        for k in range(m):
            i0, i1, i2 = idx[k - 1], idx[k], idx[(k + 1) % m]
            a, b, c = pts[i0], pts[i1], pts[i2]
            if _cross2(a, b, c) <= 1e-12:
                continue            # reflex (or degenerate): not an ear
            inside = False
            for j in idx:
                if j in (i0, i1, i2):
                    continue
                p = pts[j]
                if _cross2(a, b, p) >= 0.0 and _cross2(b, c, p) >= 0.0 and _cross2(c, a, p) >= 0.0:
                    inside = True
                    break
            if inside:
                continue
            tris.append((i0, i1, i2))
            del idx[k]
            break
        else:
            raise ValueError("ear clipping failed: the polygon is not simple")
    tris.append((idx[0], idx[1], idx[2]))
    return tris


def box_mesh(box, drop_bottom: bool):
    """A Room Sim object box (`geometry.Box`, §3.1) as 8 vertices and 12
    outward triangles (10 when `drop_bottom`), taken to the world with
    Room Sim's own `Box.to_world`."""
    hl, hb, hh = box.sl / 2.0, box.sb / 2.0, box.sh / 2.0
    local = [(-hl, -hb, -hh), (hl, -hb, -hh), (hl, hb, -hh), (-hl, hb, -hh),
             (-hl, -hb, hh), (hl, -hb, hh), (hl, hb, hh), (-hl, hb, hh)]
    V = np.array([box.to_world(*p) for p in local], dtype=np.float64)
    faces = [(4, 5, 6), (4, 6, 7),           # top
             (0, 1, 5), (0, 5, 4),           # -y
             (1, 2, 6), (1, 6, 5),           # +x
             (2, 3, 7), (2, 7, 6),           # +y
             (3, 0, 4), (3, 4, 7)]           # -x
    if not drop_bottom:
        faces += [(0, 3, 2), (0, 2, 1)]      # bottom
    return V, np.array(faces, dtype=np.int64)


def _wall_face_offset(wall, node_pos, footprint) -> tuple:
    """The signed offset `o` of the Sionna surface standing for a wall, and
    a note when it is an approximation.

    Room Sim reflects from the face on the transmitter's side (§6.1,
    `f = side(Tp)`), so a wall with every node on one side of its line has
    one face that every Room Sim bounce uses: `o = f * t/2`, exact. A wall
    with nodes on both sides of its line (an L's re-entrant walls) is
    reflected from both faces in Room Sim, a single Sionna surface cannot
    be: it goes on the face toward the room (the footprint's interior), so
    bounces from inside are exact and Room Sim's bounces off its other face
    (reached through another wall) are up to `t` longer or shorter. A wall
    off the footprint boundary with nodes on both sides goes on its
    centreline."""
    from wifisense.roomsim.geometry import inside

    sides = {wall.side(p) for p in node_pos.values()} - {0}
    if len(sides) == 1:
        f = sides.pop()
        return float(f) * (wall.t / 2.0), ""
    mx, my = (wall.A[0] + wall.B[0]) / 2.0, (wall.A[1] + wall.B[1]) / 2.0
    eps = wall.t / 2.0 + 1e-3
    ins = [inside(footprint, (mx + s * eps * wall.n[0], my + s * eps * wall.n[1])) for s in (1.0, -1.0)]
    if ins[0] != ins[1]:
        f = 1 if ins[0] else -1
        return float(f) * (wall.t / 2.0), (
            f"walls[{wall.wall_index}] has nodes on both sides of its line: placed on its room-facing face, so "
            f"Room Sim's bounces off its other face (reached through another wall) differ by up to t = "
            f"{wall.t} m in path length")
    return 0.0, (f"walls[{wall.wall_index}] has nodes on both sides: placed on its centreline, so its "
                 f"reflections are {wall.t / 2.0:.3f} m off Room Sim's faces (§6.1)")


def convert(scene, cfg: Config | None = None, *, furniture: bool = True, object_model: str = "halfspace",
            scattering_coefficient: float = 0.0) -> ConvertedScene:
    """Convert a parsed Room Sim scene (`wifisense.roomsim.Scene`) under
    `cfg` (Room Sim defaults if None) into a `ConvertedScene` (SIM).

    `furniture=False` leaves every object out (the "shell" variant: walls,
    floor and ceiling only). `object_model` is `"halfspace"` or `"panel"`
    (module docstring). `scattering_coefficient` is Sionna's `S` on every
    material (0: no diffuse reflection). Doors and windows are never
    converted (Room Sim ignores them as obstacles, spec/roomsim.md §6.5).
    """
    if object_model not in OBJECT_MODELS:
        raise ValueError(f"object_model must be one of {OBJECT_MODELS}")
    if not 0.0 <= scattering_coefficient <= 1.0:
        raise ValueError("scattering_coefficient must be in [0, 1]")
    cfg = Config() if cfg is None else cfg
    world = World(scene, cfg)            # Room Sim's own resolved geometry (read-only)
    carrier = world.carrier_hz
    S = float(scattering_coefficient)
    materials: dict = {}
    shapes: list = []
    approx: list = []
    surface_names: dict = {}
    object_names: dict = {}

    def material(mid: str, itu_type, thickness, source, eps_r=None, sigma=None) -> str:
        if mid not in materials:
            materials[mid] = MaterialSpec(mid, itu_type, float(thickness), eps_r, sigma, S, source)
        return mid

    # Walls (explicit or implied), §3.1 / §6.1.
    for w in world.walls:
        o, note = _wall_face_offset(w, world.node_pos, world.F)
        if note:
            approx.append(note)
        ax, ay = w.A[0] + o * w.n[0], w.A[1] + o * w.n[1]
        bx, by = w.B[0] + o * w.n[0], w.B[1] + o * w.n[1]
        V = np.array([(ax, ay, 0.0), (bx, by, 0.0), (bx, by, w.h), (ax, ay, w.h)], dtype=np.float64)
        F = np.array([(0, 1, 2), (0, 2, 3)], dtype=np.int64)
        mid = material(f"rm-wall-{w.material}-{w.t:.4f}", WALL_ITU[w.material], w.t,
                       f"{'implied ' if w.implied else ''}wall material {w.material}, t = {w.t} m")
        name = f"wall-{w.wall_index}"
        shapes.append(ShapeSpec(name, "wall", w.wall_index, w.index, mid, V, F,
                                note=f"face offset {o:+.4f} m from the centreline"))
        surface_names[w.index] = name

    # Floor and ceiling (§6.1), the footprint polygon.
    if cfg.floor_ceiling:
        F2 = world.F
        tris = np.array(triangulate(F2), dtype=np.int64)
        for s in world.surfaces[len(world.walls):]:
            z = 0.0 if s.kind == "floor" else world.H
            V = np.array([(x, y, z) for x, y in F2], dtype=np.float64)
            mat = cfg.floor_material if s.kind == "floor" else cfg.ceiling_material
            mid = material(f"rm-{s.kind}-{mat}-{s.t:.4f}", WALL_ITU[mat], s.t,
                           f"{s.kind}_material {mat}, slab_thickness_m = {s.t} m")
            shapes.append(ShapeSpec(s.kind, s.kind, -1, s.index, mid, V, tris))
            surface_names[s.index] = s.kind

    # Furniture (§5.2, §6.5): every object but doors and windows.
    if furniture:
        fabric_sigma = None
        for o in world.objects:
            if o.is_opening:
                continue
            m = o.material
            if m == "fabric":
                if fabric_sigma is None:
                    loss = rs_materials.OBJECT_LOSS_DB["fabric"]
                    fabric_sigma = fabric_conductivity(loss, carrier)
                mid = material("rm-obj-fabric", None, FABRIC_THICKNESS_M,
                               f"object material fabric (no ITU material): eps_r 1, sigma {fabric_sigma:.6g} S/m, "
                               f"{FABRIC_THICKNESS_M} m per face = {rs_materials.OBJECT_LOSS_DB['fabric']} dB per "
                               "crossing at normal incidence", eps_r=1.0, sigma=fabric_sigma)
            elif object_model == "halfspace":
                itu = OBJECT_ITU[m]
                d = halfspace_thickness(itu, carrier) if m != "metal" else 0.01
                mid = material(f"rm-obj-{m}-halfspace", itu, d,
                               f"object material {m}: ITU {itu}, {d:.4g} m thick so R = single-interface G")
            else:
                itu = OBJECT_ITU[m]
                d = PANEL_THICKNESS_M[m]
                mid = material(f"rm-obj-{m}-panel", itu, d, f"object material {m}: ITU {itu}, {d} m panels")
            box = o.box
            drop_bottom = (box.cz - box.sh / 2.0) <= 1e-9
            V, F = box_mesh(box, drop_bottom)
            name = f"obj-{o.index}"
            shapes.append(ShapeSpec(name, "object", o.index, -1, mid, V, F,
                                    note=f"{o.kind} ({m}){', no bottom face (on the floor)' if drop_bottom else ''}"))
            object_names[o.index] = name
        if any(o.material == "fabric" and not o.is_opening for o in world.objects):
            approx.append("fabric has no ITU material: a non-reflecting stand-in (eps_r 1) losing Room Sim's "
                          "3 dB per crossing at normal incidence")
        if object_model == "halfspace" and any(not o.is_opening and o.material != "fabric" for o in world.objects):
            approx.append("furniture faces are half-spaces (Room Sim's single-interface G): the boxes are opaque "
                          "in Sionna, where Room Sim charges a fixed L_db per crossing")
    openings = [o.index for o in world.objects if o.is_opening]
    if openings:
        approx.append(f"objects {openings} (doors/windows) are not converted: Room Sim models the wall whole (§6.5)")

    return ConvertedScene(
        name=scene.name, carrier_hz=carrier, spacing_hz=float(cfg.spacing_hz), wavelength_m=world.lam,
        height_m=world.H, footprint=list(world.F),
        nodes={nid: world.node_pos[nid] for nid in sorted(world.node_ids)},
        shapes=shapes, materials=materials, surface_names=surface_names, object_names=object_names,
        approximations=approx,
        options={"furniture": furniture, "object_model": object_model, "scattering_coefficient": S,
                 "floor_ceiling": bool(cfg.floor_ceiling)},
    )


# -- writing -----------------------------------------------------------------


def _write_ply(path: str, V: np.ndarray, F: np.ndarray) -> None:
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(V)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(F)}\nproperty list uchar int vertex_indices\nend_header\n")
        for v in V:
            f.write(f"{float(v[0])!r} {float(v[1])!r} {float(v[2])!r}\n")
        for t in F:
            f.write(f"3 {int(t[0])} {int(t[1])} {int(t[2])}\n")


def _material_xml(m: MaterialSpec) -> str:
    if m.itu_type is not None:
        return (f'  <bsdf type="itu-radio-material" id="{m.id}">\n'
                f'    <string name="type" value="{m.itu_type}"/>\n'
                f'    <float name="thickness" value="{m.thickness_m!r}"/>\n'
                f'    <float name="scattering_coefficient" value="{m.scattering_coefficient!r}"/>\n'
                f'  </bsdf>\n')
    return (f'  <bsdf type="radio-material" id="{m.id}">\n'
            f'    <float name="relative_permittivity" value="{m.eps_r!r}"/>\n'
            f'    <float name="conductivity" value="{m.sigma_s_m!r}"/>\n'
            f'    <float name="thickness" value="{m.thickness_m!r}"/>\n'
            f'    <float name="scattering_coefficient" value="{m.scattering_coefficient!r}"/>\n'
            f'  </bsdf>\n')


def write_scene(conv: ConvertedScene, out_dir: str) -> str:
    """Write `scene.xml` and one PLY per shape into `out_dir`; returns the
    XML path. Mesh paths in the XML are absolute."""
    os.makedirs(out_dir, exist_ok=True)
    mesh_dir = os.path.join(out_dir, "meshes")
    os.makedirs(mesh_dir, exist_ok=True)
    parts = ['<scene version="2.1.0">\n', f"  <!-- {conv.label}: Room Sim scene {conv.name!r} converted for "
             "Sionna RT 2.2.0 (ml/sionna_audit) -->\n"]
    for m in conv.materials.values():
        parts.append(_material_xml(m))
    for s in conv.shapes:
        ply = os.path.abspath(os.path.join(mesh_dir, f"{s.name}.ply"))
        _write_ply(ply, s.vertices, s.faces)
        parts.append(f'  <shape type="ply" id="{s.name}">\n'
                     f'    <string name="filename" value="{ply}"/>\n'
                     f'    <boolean name="face_normals" value="true"/>\n'
                     f'    <ref id="{s.material}" name="bsdf"/>\n'
                     f'  </shape>\n')
    parts.append("</scene>\n")
    path = os.path.join(out_dir, "scene.xml")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(parts))
    return path


# -- the Sionna backend ------------------------------------------------------

_BACKEND: dict = {}


def find_libllvm() -> str | None:
    """A libLLVM for Dr.Jit's LLVM backend: `DRJIT_LIBLLVM_PATH` if set, else
    the newest major version >= 17 found in the usual places (Dr.Jit 1.5
    needs >= 15, and LLVM 15 cannot lower the `llvm.maximum` it emits).
    Linker scripts (tiny text files) are skipped."""
    env = os.environ.get("DRJIT_LIBLLVM_PATH")
    if env:
        return env
    pats = [os.path.expanduser("~/.rustup/toolchains/*/lib/libLLVM*.so*"),
            "/usr/lib/llvm-*/lib/libLLVM*.so*", "/usr/lib/x86_64-linux-gnu/libLLVM*.so*",
            "/usr/local/lib/libLLVM*.so*", "/opt/*/lib/libLLVM*.so*"]
    best = None
    for pat in pats:
        for p in glob.glob(pat):
            try:
                if os.path.getsize(p) < 1_000_000:
                    continue
            except OSError:
                continue
            m = re.search(r"libLLVM[.-](?:so\.)?(\d+)", os.path.basename(p))
            if not m:
                continue
            ver = int(m.group(1))
            if ver >= 17 and (best is None or ver > best[0]):
                best = (ver, p)
    return best[1] if best else None


def configure_backend(variant: str | None = None) -> str:
    """Select the Mitsuba variant before `sionna.rt` is imported and return
    it. Default `llvm_ad_mono_polarized` (CPU), or `SIONNA_AUDIT_VARIANT`.
    On the audit machine the CUDA variant aborts the process in OptiX's
    compiler (driver 545: an unimplemented PTX intrinsic), which no
    `try` can catch, so it is never tried automatically."""
    if _BACKEND:
        return _BACKEND["variant"]
    variant = variant or os.environ.get("SIONNA_AUDIT_VARIANT", "llvm_ad_mono_polarized")
    if variant.startswith("llvm") and not os.environ.get("DRJIT_LIBLLVM_PATH"):
        lib = find_libllvm()
        if lib:
            os.environ["DRJIT_LIBLLVM_PATH"] = lib
    import mitsuba as mi
    mi.set_variant(variant)
    import sionna.rt  # noqa: F401  (registers the radio materials for this variant)
    _BACKEND["variant"] = variant
    _BACKEND["libllvm"] = os.environ.get("DRJIT_LIBLLVM_PATH")
    return variant


def backend_info() -> dict:
    """The selected variant and libLLVM path (after `configure_backend`)."""
    return dict(_BACKEND)


def load_sionna_scene(conv: ConvertedScene, out_dir: str, node_ids=None):
    """Write `conv` to `out_dir` and load it into Sionna RT with every node in
    `node_ids` (default: all) as both a transmitter `tx-<id>` and a receiver
    `rx-<id>`: iso pattern, V polarisation, scene frequency = carrier."""
    configure_backend()
    import mitsuba as mi
    import sionna.rt as rt

    path = write_scene(conv, out_dir)
    scene = rt.load_scene(path, merge_shapes=False)
    scene.frequency = conv.carrier_hz
    scene.tx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    scene.rx_array = rt.PlanarArray(num_rows=1, num_cols=1, pattern="iso", polarization="V")
    ids = sorted(conv.nodes) if node_ids is None else list(node_ids)
    for nid in ids:
        x, y, z = conv.nodes[nid]
        scene.add(rt.Transmitter(name=f"tx-{nid}", position=mi.Point3f(x, y, z)))
    for nid in ids:
        x, y, z = conv.nodes[nid]
        scene.add(rt.Receiver(name=f"rx-{nid}", position=mi.Point3f(x, y, z)))
    return scene


@dataclass
class SionnaLink:
    """Sionna's paths of one link `tx -> rx` (SIM): complex coefficients `a`
    (iso/V antennas, unitless), delays `tau` (s), and per path its
    interaction types `inter` and shape ids `objs` (`[P, D]`, 0 / INVALID
    past the path's last interaction); `names` maps shape ids to names."""

    tx: int
    rx: int
    a: np.ndarray
    tau: np.ndarray
    inter: np.ndarray
    objs: np.ndarray
    names: dict

    def n_of(self, kind: str) -> np.ndarray:
        """Per path, how many interactions of `kind` it has."""
        if self.inter.size == 0:
            return np.zeros(len(self.a), dtype=np.int64)
        return np.sum(self.inter == INTERACTION[kind], axis=1).astype(np.int64)

    def reflections(self, idx=None) -> list:
        """Per path (or per path in `idx`), the names of the shapes it
        reflects from specularly, in order (refractions skipped)."""
        code = INTERACTION["specular"]
        rows = range(len(self.a)) if idx is None else idx
        return [tuple(self.names.get(int(o), f"?{int(o)}")
                      for t, o in zip(self.inter[p], self.objs[p]) if t == code) for p in rows]

    def sequence(self, p: int) -> list:
        """Path `p` as `[(interaction type, shape name), ...]`."""
        return [(int(t), self.names.get(int(o), f"?{int(o)}")) for t, o in zip(self.inter[p], self.objs[p])
                if t != 0]

    def touches(self, prefix: str) -> np.ndarray:
        """Per path, whether any interaction is with a shape named `prefix*`."""
        ids = [i for i, n in self.names.items() if n.startswith(prefix)]
        if self.inter.size == 0 or not ids:
            return np.zeros(len(self.a), dtype=bool)
        return np.any(np.isin(self.objs, ids) & (self.inter != 0), axis=1)


class _CandidateCapture:
    """Wraps Sionna's candidate generator to read how many candidates it
    stored: when that total is below `max_num_paths_per_src`, no source hit
    its cap, so no candidate was dropped for lack of buffer."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.stored = None

    def __call__(self, **kwargs):
        buf = self.inner(**kwargs)
        self.stored = int(buf.paths_counter.numpy()[0])
        return buf


def _extract(paths, scene, tx_ids, rx_ids, out: dict) -> None:
    """Copy every ordered pair `tx != rx` of a solved `Paths` into `out`."""
    tx_names = list(scene.transmitters.keys())
    rx_names = list(scene.receivers.keys())
    if tx_names != [f"tx-{i}" for i in tx_ids] or rx_names != [f"rx-{i}" for i in rx_ids]:
        raise RuntimeError("transmitter/receiver order differs from the node ids")
    id_to_name = {int(o.object_id): name for name, o in scene.objects.items()}
    tau = paths.tau.numpy().astype(np.float64)                                    # [rx, tx, P]
    if tau.shape[-1] == 0:
        for tx in tx_ids:
            for rx in rx_ids:
                if tx != rx:
                    out[(tx, rx)] = SionnaLink(tx, rx, np.zeros(0, complex), np.zeros(0),
                                               np.zeros((0, 0), np.uint32), np.zeros((0, 0), np.uint32),
                                               id_to_name)
        return
    a_re, a_im = paths.a
    a = a_re.numpy().astype(np.float64) + 1j * a_im.numpy().astype(np.float64)   # [rx, 1, tx, 1, P]
    valid = paths.valid.numpy().astype(bool)
    if valid.ndim == 5:
        valid = valid[:, 0, :, 0, :]
    inter = paths.interactions.numpy()     # [D, rx, tx, P]
    objs = paths.objects.numpy()
    for i, tx in enumerate(tx_ids):
        for j, rx in enumerate(rx_ids):
            if tx == rx:
                continue
            idx = np.nonzero(valid[j, i, :] & (tau[j, i, :] >= 0.0))[0]
            it = np.ascontiguousarray(inter[:, j, i, idx].T).astype(np.uint32)
            ob = np.ascontiguousarray(objs[:, j, i, idx].T).astype(np.uint32)
            # Zero everything after a path's first NONE (Sionna leaves it so already).
            dead = np.cumsum(it == 0, axis=1) > 0
            it[dead] = 0
            out[(tx, rx)] = SionnaLink(tx, rx, a[j, 0, i, 0, idx], tau[j, i, idx], it, ob, id_to_name)


def solve_paths(scene, node_ids, *, max_depth: int, samples_per_src: int, max_num_paths_per_src: int,
                diffuse: bool = False, refraction: bool = True, seed: int = 42,
                deterministic: bool = False, one_receiver_per_solve: bool = True,
                stats: dict | None = None) -> dict:
    """Run Sionna's `PathSolver` on a scene from `load_sionna_scene` and
    return `{(tx, rx): SionnaLink}` for every ordered pair `tx != rx` of
    `node_ids`. Specular reflection and LoS are on, diffraction off;
    `diffuse` and `refraction` as given.

    `one_receiver_per_solve` (default) solves once per receiver with every
    transmitter present. With several receivers in one solve, Sionna 2.2.0's
    candidate generator was seen to drop some specular chains for whole
    (source, target) blocks (spec/sionna_audit.md §3.4: in a 9-node room it
    never stored floor -> ceiling for 37 of 72 links, while the same chain
    was found in every one-receiver solve); one receiver per solve costs a
    ray launch per receiver and removes that.

    `stats`, if given, receives the largest number of stored candidates of
    any solve and whether any solve could have dropped one for lack of buffer
    (`truncated`); a truncated specular run raises (a diffuse run is a Monte
    Carlo estimate either way)."""
    configure_backend()
    import mitsuba as mi
    import sionna.rt as rt

    ids = list(node_ids)
    positions = {}
    for name, dev in scene.receivers.items():
        positions[int(name.split("-", 1)[1])] = [float(v) for v in np.asarray(dev.position.numpy()).ravel()]
    out: dict = {}
    worst = 0
    groups = [[r] for r in ids] if one_receiver_per_solve else [ids]
    for rx_ids in groups:
        if one_receiver_per_solve:
            for name in list(scene.receivers.keys()):
                scene.remove(name)
            for r in rx_ids:
                scene.add(rt.Receiver(name=f"rx-{r}", position=mi.Point3f(*positions[r])))
        solver = rt.PathSolver(deterministic=deterministic)
        cap = _CandidateCapture(solver._candidate_generator)
        solver._candidate_generator = cap
        paths = solver(scene, max_depth=max_depth, max_num_paths_per_src=max_num_paths_per_src,
                       samples_per_src=samples_per_src, synthetic_array=True, los=True,
                       specular_reflection=True, diffuse_reflection=diffuse, refraction=refraction,
                       diffraction=False, edge_diffraction=False, seed=seed)
        worst = max(worst, cap.stored or 0)
        if worst >= max_num_paths_per_src and not diffuse:
            raise RuntimeError(f"Sionna stored {worst} candidates, at the per-source cap "
                               f"{max_num_paths_per_src}: raise max_num_paths_per_src")
        _extract(paths, scene, ids, rx_ids, out)
        del paths
    if one_receiver_per_solve:          # leave the scene as load_sionna_scene made it
        for name in list(scene.receivers.keys()):
            scene.remove(name)
        for r in ids:
            scene.add(rt.Receiver(name=f"rx-{r}", position=mi.Point3f(*positions[r])))
    if stats is not None:
        stats.update({"candidates_stored_max": worst, "max_num_paths_per_src": max_num_paths_per_src,
                      "truncated": bool(worst >= max_num_paths_per_src), "solves": len(groups)})
    return out
