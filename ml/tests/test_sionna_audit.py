"""Phase 10a: the Sionna RT audit of Room Sim (spec/sionna_audit.md). All SIM.

What is checked:

* the 52 LLTF subcarrier frequencies, bit for bit against Room Sim's grid;
* the §5.1 slab coefficients: Room Sim's reference against an independent
  transcription of spec/roomsim.md §5.1 and its §15 ground truths, and
  (with Sionna) against Sionna RT 2.2.0's own `itu_coefficients_single_layer_slab`
  and ITU table;
* the converter on a hand-built one-wall room: geometry, face placement,
  materials, the XML and meshes; with Sionna, the ray-traced paths and CFR
  against Room Sim's own static paths and channel;
* the metrics; and a tiny audit run (a hand-written scene, and one dataset
  scene when the SIM dataset is present).

Tests that need Sionna skip cleanly when it cannot be imported (or its
Dr.Jit backend finds no libLLVM). Run from the repo root:

    .venv/bin/python -m unittest ml.tests.test_sionna_audit
"""

import argparse
import cmath
import json
import math
import os
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.sionna_audit import metrics as M  # noqa: E402
from ml.sionna_audit import roomsim_side as rss  # noqa: E402
from ml.sionna_audit import scene_to_sionna as s2s  # noqa: E402
from wifisense.roomsim import Config, load_scene, parse_scene  # noqa: E402
from wifisense.roomsim import materials as rs_materials  # noqa: E402
from wifisense.roomsim.numerics import EPS0, C, carrier_hz  # noqa: E402
from wifisense.roomsim.world import BINS, World  # noqa: E402

DATASET = os.environ.get("SAANS_DATASET_DIR", "/home/Aman/are_u_theone/datasets/roomsim-v1")
BEDROOM = os.path.join(ROOT, "spec", "scenes", "bedroom.json")

_SIONNA = {}


def sionna_rt():
    """`sionna.rt` with the audit's backend, or None (reason in _SIONNA)."""
    if "rt" not in _SIONNA:
        try:
            s2s.configure_backend()
            import sionna.rt as rt
            _SIONNA["rt"] = rt
        except Exception as e:  # noqa: BLE001  (ImportError, no libLLVM, ...)
            _SIONNA["rt"] = None
            _SIONNA["why"] = f"{type(e).__name__}: {e}"
    return _SIONNA["rt"]


def needs_sionna(cls):
    """Skip a test class when Sionna RT 2.2.0 cannot be used here."""
    orig = cls.setUpClass

    @classmethod
    def setUpClass(klass):
        if sionna_rt() is None:
            raise unittest.SkipTest(f"Sionna RT unavailable ({_SIONNA.get('why')})")
        orig.__func__(klass)

    cls.setUpClass = setUpClass
    return cls


def one_wall_scene(nodes=((3.0, 2.0, 1.5), (7.0, 2.0, 1.5)), wall_y=5.0, material="concrete", t=0.2):
    """A hand-built room (10 x 10 x 3 m) whose only listed wall runs along
    `y = wall_y` (so no wall is implied, spec/roomsim.md §3.1)."""
    doc = {
        "schema_version": 1, "name": "one-wall", "provenance": "hand_written",
        "room": {"w_m": 10.0, "d_m": 10.0, "h_m": 3.0},
        "walls": [{"from": [0.0, wall_y], "to": [10.0, wall_y], "material": material, "thickness_m": t}],
        "nodes": [{"id": i, "pos": list(p)} for i, p in enumerate(nodes)],
    }
    return parse_scene(json.dumps(doc))


# -- independent §5.1 transcription (complex arithmetic) ---------------------------


def p2040(material: str, cos_t: float, t: float, f: float, pol: str):
    """spec/roomsim.md §5.1 written out with Python complex numbers:
    `(G, R, T, q0)`, `T` with the spec's `e^{+j q0}`."""
    a, b, c, d = rs_materials.ITU[material]
    fg = f / 1e9
    eta = complex(a * fg ** b, -(c * fg ** d) / (2 * math.pi * f * EPS0))
    lam = C / f
    s = cmath.sqrt(eta - (1 - cos_t * cos_t))
    G = (cos_t - s) / (cos_t + s) if pol == "TE" else (eta * cos_t - s) / (eta * cos_t + s)
    k0t = 2 * math.pi * t / lam
    q = s * k0t
    q0 = k0t * cos_t
    E2 = cmath.exp(-2j * q)
    den = 1 - G * G * E2
    R = G * (1 - E2) / den
    T = (1 - G * G) * cmath.exp(-1j * (q - q0)) / den
    return G, R, T, q0


class TestFrequencies(unittest.TestCase):
    """The 52 LLTF subcarriers (spec/roomsim.md §4)."""

    def test_bins(self):
        self.assertEqual(len(s2s.LLTF_BINS), 52)
        self.assertEqual(list(s2s.LLTF_BINS), list(range(-26, 0)) + list(range(1, 27)))
        self.assertNotIn(0, s2s.LLTF_BINS)

    def test_frequencies_are_roomsims(self):
        for ch in (1, 6, 11, 13):
            cfg = Config(channel=ch)
            world = World(one_wall_scene(), cfg)
            f = s2s.lltf_frequencies(world.carrier_hz, cfg.spacing_hz)
            want = [world.freqs[BINS.index(k)] for k in s2s.LLTF_BINS]
            self.assertEqual(f.tolist(), want)          # bit for bit
        f = s2s.lltf_frequencies(carrier_hz(6))
        self.assertEqual(carrier_hz(6), 2437e6)
        self.assertTrue(np.allclose(np.diff(f)[np.diff(f) < 400e3], 312500.0))
        self.assertAlmostEqual(f[0], 2437e6 - 26 * 312500.0)
        self.assertAlmostEqual(f[-1], 2437e6 + 26 * 312500.0)
        self.assertAlmostEqual(f[26] - f[25], 2 * 312500.0)   # the DC gap


class TestSlabFormula(unittest.TestCase):
    """Room Sim's §5.1 slab against the spec's text and §15 (no Sionna)."""

    f = 2437e6

    def test_reference_equals_spec_text(self):
        lam = C / self.f
        for mat in rs_materials.ITU:
            et = rs_materials.eta(mat, self.f)
            for t in (0.008, 0.1, 0.2, 0.25):
                for ang in (0.0, 15.0, 30.0, 45.0, 60.0, 75.0, 85.0, 89.0):
                    c = math.cos(math.radians(ang))
                    s, g_te, g_tm = rs_materials.fresnel(et, c)
                    for pol, G in (("TE", g_te), ("TM", g_tm)):
                        R, T, q0 = rs_materials.slab(G, s, c, t, lam)
                        G2, R2, T2, q02 = p2040(mat, c, t, self.f, pol)
                        self.assertLess(abs(complex(*G) - G2), 1e-12)
                        self.assertLess(abs(complex(*R) - R2), 1e-9, (mat, t, ang, pol))
                        self.assertLess(abs(complex(*T) - T2), 1e-9, (mat, t, ang, pol))
                        self.assertAlmostEqual(q0, q02, places=12)

    def test_section_15_ground_truths(self):
        lam = C / self.f
        for mat in ("concrete", "brick", "drywall", "glass", "wood"):
            et = rs_materials.eta(mat, self.f)
            s, g_te, g_tm = rs_materials.fresnel(et, 1.0)
            self.assertLess(abs(complex(*g_tm) + complex(*g_te)), 1e-12)     # G_TM = -G_TE at normal incidence
            sq = cmath.sqrt(complex(*et))
            self.assertLess(abs(complex(*g_te) - (1 - sq) / (1 + sq)), 1e-12)
        # A lossless slab conserves energy; a lossless half-wave slab does not reflect.
        for er in (2.0, 4.0, 6.31):
            et = (er, 0.0)
            for ang in (0.0, 30.0, 60.0):
                c = math.cos(math.radians(ang))
                s, g_te, g_tm = rs_materials.fresnel(et, c)
                for G in (g_te, g_tm):
                    R, T, _q0 = rs_materials.slab(G, s, c, 0.13, lam)
                    self.assertAlmostEqual(abs(complex(*R)) ** 2 + abs(complex(*T)) ** 2, 1.0, places=12)
            t_half = lam / (2.0 * math.sqrt(er))
            s, g_te, _g = rs_materials.fresnel(et, 1.0)
            R, _T, _q0 = rs_materials.slab(g_te, s, 1.0, t_half, lam)
            self.assertLess(abs(complex(*R)), 1e-12)

    def test_printed_p2040_differs_by_q0_only(self):
        """The printed P.2040 transmission (and Sionna's) lacks Room Sim's
        e^{+j q0}: equal magnitude, phase differing by exactly q0 (§5.1)."""
        lam = C / self.f
        for mat in ("concrete", "drywall", "glass"):
            et = rs_materials.eta(mat, self.f)
            for ang in (0.0, 40.0, 80.0):
                c = math.cos(math.radians(ang))
                s, g_te, g_tm = rs_materials.fresnel(et, c)
                for G in (g_te, g_tm):
                    _R, T, q0 = rs_materials.slab(G, s, c, 0.2, lam)
                    Gc = complex(*G)
                    q = complex(*s) * (2 * math.pi * 0.2 / lam)
                    T_printed = (1 - Gc * Gc) * cmath.exp(-1j * q) / (1 - Gc * Gc * cmath.exp(-2j * q))
                    self.assertAlmostEqual(abs(complex(*T)), abs(T_printed), places=12)
                    d = float(M.wrap_phase(cmath.phase(complex(*T)) - cmath.phase(T_printed) - q0))
                    self.assertAlmostEqual(d, 0.0, places=9)

    def test_halfspace_thickness_reproduces_single_interface(self):
        lam = C / self.f
        for mat in ("wood", "glass"):
            d = s2s.halfspace_thickness(mat, self.f)
            et = rs_materials.eta(mat, self.f)
            for ang in (0.0, 30.0, 60.0, 85.0):
                c = math.cos(math.radians(ang))
                s, g_te, g_tm = rs_materials.fresnel(et, c)
                for G in (g_te, g_tm):
                    R, T, _q0 = rs_materials.slab(G, s, c, d, lam)
                    self.assertLess(abs(complex(*R) - complex(*G)), 1e-9)
                    self.assertLess(abs(complex(*T)), 1e-5)        # opaque

    def test_fabric_stand_in(self):
        lam = C / self.f
        sigma = s2s.fabric_conductivity(3.0, self.f)
        et = (1.0, -(sigma / ((2 * math.pi * self.f) * EPS0)))
        s, g_te, g_tm = rs_materials.fresnel(et, 1.0)
        R, T, _q0 = rs_materials.slab(g_te, s, 1.0, s2s.FABRIC_THICKNESS_M, lam)
        self.assertAlmostEqual(20 * math.log10(abs(complex(*T))), -1.5, places=6)   # two faces: 3 dB
        self.assertLess(20 * math.log10(abs(complex(*R))), -30.0)                  # it does not reflect


@needs_sionna
class TestSlabAgainstSionna(unittest.TestCase):
    """Room Sim's materials and slab against Sionna RT 2.2.0's."""

    @classmethod
    def setUpClass(cls):
        pass

    def test_itu_tables_agree(self):
        from sionna.rt.radio_materials.itu import ITU_MATERIALS_PROPERTIES
        f_ghz = 2.437
        mapping = dict(s2s.WALL_ITU)
        mapping["metal"] = "metal"
        for rs_name, itu in mapping.items():
            rows = [p for (lo, hi), p in ITU_MATERIALS_PROPERTIES[itu].items() if lo <= f_ghz <= hi]
            self.assertEqual(len(rows) >= 1, True, itu)
            self.assertEqual(tuple(rows[0]), tuple(rs_materials.ITU[rs_name]), (rs_name, itu))

    def test_slab_coefficients_agree(self):
        from ml.sionna_audit import audit as A
        res = A.slab_formula_check(2437e6, thicknesses=(0.1, 0.2), angles_deg=(0.0, 20.0, 45.0, 70.0, 85.0))
        w = res["worst"]
        self.assertLess(w["R_abs_err"], 1e-4)
        self.assertLess(w["R_db_err"], 0.01)
        self.assertLess(w["T_db_err"], 0.01)
        self.assertLess(w["T_phase_minus_q0_err"], 1e-3)       # Room Sim's T = Sionna's T e^{+j q0}
        for o in res["objects_halfspace"]:
            self.assertLess(o["max_abs_err_G_vs_sionna_R"], 1e-5, o)

    def test_raytraced_one_surface(self):
        from ml.sionna_audit import audit as A
        res = A.raytraced_slab_check(2437e6, material="brick", thickness=0.15, angles_deg=(0.0, 50.0))
        for r in res["rows"]:
            if r["path"] == "reflection_tilted":
                # Room Sim's TE-only wall coefficient (§16): Sionna lies between TE and TM.
                lo, hi = sorted((r["roomsim_db"], r["roomsim_tm_db"]))
                self.assertTrue(lo - 0.01 <= r["sionna_db"] <= hi + 0.01, r)
            else:
                self.assertLess(abs(r["diff_db"]), 0.01, r)
                self.assertLess(abs(r["diff_phase_rad"]), 1e-3, r)


class TestConverter(unittest.TestCase):
    """Room Sim scene -> Sionna scene description (no Sionna needed)."""

    def test_one_wall_geometry_and_material(self):
        scene = one_wall_scene()
        conv = s2s.convert(scene, Config(floor_ceiling=False))
        self.assertEqual(conv.label, "SIM")
        self.assertEqual([s.name for s in conv.shapes], ["wall-0"])
        w = conv.shapes[0]
        # Both nodes are at y = 2 < 5: the bounce face is t/2 = 0.1 m toward them (§6.1).
        self.assertTrue(np.allclose(w.vertices[:, 1], 4.9))
        self.assertEqual(sorted(set(w.vertices[:, 0].tolist())), [0.0, 10.0])
        self.assertEqual(sorted(set(w.vertices[:, 2].tolist())), [0.0, 3.0])
        self.assertEqual(len(w.faces), 2)
        m = conv.materials[w.material]
        self.assertEqual((m.itu_type, m.thickness_m, m.scattering_coefficient), ("concrete", 0.2, 0.0))
        self.assertEqual(conv.surface_names, {0: "wall-0"})
        self.assertEqual(conv.approximations, [])
        self.assertEqual(conv.nodes, {0: (3.0, 2.0, 1.5), 1: (7.0, 2.0, 1.5)})
        self.assertAlmostEqual(conv.carrier_hz, 2437e6)

    def test_other_side_and_both_sides(self):
        conv = s2s.convert(one_wall_scene(nodes=((3.0, 8.0, 1.5), (7.0, 7.0, 1.5))), Config(floor_ceiling=False))
        self.assertTrue(np.allclose(conv.shapes[0].vertices[:, 1], 5.1))
        conv = s2s.convert(one_wall_scene(nodes=((3.0, 2.0, 1.5), (7.0, 7.0, 1.5))), Config(floor_ceiling=False))
        self.assertTrue(np.allclose(conv.shapes[0].vertices[:, 1], 5.0))       # interior wall: centreline
        self.assertEqual(len(conv.approximations), 1)
        self.assertIn("centreline", conv.approximations[0])

    def test_materials_map(self):
        for mat, itu in (("drywall", "plasterboard"), ("brick", "brick"), ("glass", "glass"), ("wood", "wood")):
            conv = s2s.convert(one_wall_scene(material=mat, t=0.05), Config(floor_ceiling=False))
            m = conv.materials[conv.shapes[0].material]
            self.assertEqual((m.itu_type, m.thickness_m), (itu, 0.05))

    def test_floor_ceiling_and_furniture(self):
        scene = load_scene(BEDROOM)
        cfg = Config(floor_material="wood", ceiling_material="drywall", slab_thickness_m=0.25)
        conv = s2s.convert(scene, cfg)
        names = [s.name for s in conv.shapes]
        self.assertEqual(names[:6], ["wall-0", "wall-1", "wall-2", "wall-3", "floor", "ceiling"])
        floor, ceil = conv.shape("floor"), conv.shape("ceiling")
        self.assertTrue(np.all(floor.vertices[:, 2] == 0.0))
        self.assertTrue(np.all(ceil.vertices[:, 2] == scene.room.h_m))
        self.assertEqual((conv.materials[floor.material].itu_type, conv.materials[floor.material].thickness_m),
                         ("wood", 0.25))
        self.assertEqual(conv.materials[ceil.material].itu_type, "plasterboard")
        # Bedroom objects: bed, almirah, table, chair, door, window: the last two are openings.
        self.assertEqual(sorted(conv.object_names), [0, 1, 2, 3])
        self.assertTrue(any("doors/windows" in a for a in conv.approximations))
        world = World(scene, cfg)
        for oi, name in conv.object_names.items():
            sh = conv.shape(name)
            box = world.objects[oi].box
            self.assertEqual(len(sh.faces), 10)          # on the floor: no bottom face
            corners = {tuple(np.round(box.to_world(sx * box.sl / 2, sy * box.sb / 2, sz * box.sh / 2), 12))
                       for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)}
            self.assertEqual({tuple(np.round(v, 12)) for v in sh.vertices}, corners)
        # Materials per object model; fabric is the custom stand-in in both.
        bed = conv.materials[conv.shape("obj-0").material]
        self.assertIsNone(bed.itu_type)
        self.assertEqual(bed.eps_r, 1.0)
        wood_hs = conv.materials[conv.shape("obj-1").material]
        self.assertEqual(wood_hs.itu_type, "wood")
        self.assertGreater(wood_hs.thickness_m, 1.0)
        panel = s2s.convert(scene, cfg, object_model="panel")
        self.assertEqual(panel.materials[panel.shape("obj-1").material].thickness_m, s2s.PANEL_THICKNESS_M["wood"])
        shell = s2s.convert(scene, cfg, furniture=False)
        self.assertEqual([s.name for s in shell.shapes], names[:6])

    def test_triangulate(self):
        L = [(0.0, 0.0), (9.0, 0.0), (9.0, 3.0), (4.0, 3.0), (4.0, 7.0), (0.0, 7.0)]
        for poly in (L, L[::-1], [(0.0, 0.0), (5.0, 0.0), (5.0, 4.0), (0.0, 4.0)]):
            tris = s2s.triangulate(poly)
            self.assertEqual(len(tris), len(poly) - 2)
            area = sum(abs(s2s._cross2(poly[a], poly[b], poly[c])) / 2.0 for a, b, c in tris)
            self.assertAlmostEqual(area, abs(s2s._polygon_area(poly)), places=12)
            for a, b, c in tris:
                self.assertGreater(s2s._cross2(poly[a], poly[b], poly[c]), 0.0)   # counter-clockwise

    def test_l_shaped_reentrant_walls(self):
        hall = load_scene(os.path.join(ROOT, "spec", "scenes", "hall-l-shaped.json"))
        conv = s2s.convert(hall, Config())
        notes = " ".join(conv.approximations)
        self.assertIn("walls[2]", notes)
        self.assertIn("walls[3]", notes)
        # walls[2] runs y = 3 from x = 9 to 4, the room below it: its face is at 3 - 0.115/2.
        self.assertTrue(np.allclose(conv.shape("wall-2").vertices[:, 1], 3.0 - 0.115 / 2.0))
        self.assertTrue(np.allclose(conv.shape("wall-3").vertices[:, 0], 4.0 - 0.115 / 2.0))

    def test_write_scene(self):
        conv = s2s.convert(load_scene(BEDROOM), Config())
        with tempfile.TemporaryDirectory() as d:
            path = s2s.write_scene(conv, d)
            root = ET.parse(path).getroot()
            shapes = {e.get("id") for e in root.findall("shape")}
            self.assertEqual(shapes, {s.name for s in conv.shapes})
            bsdfs = {e.get("id"): e.get("type") for e in root.findall("bsdf")}
            self.assertEqual(set(bsdfs), set(conv.materials))
            self.assertEqual(bsdfs["rm-obj-fabric"], "radio-material")
            for s in conv.shapes:
                with open(os.path.join(d, "meshes", f"{s.name}.ply"), encoding="ascii") as f:
                    head = f.read(400)
                self.assertIn(f"element vertex {len(s.vertices)}", head)
                self.assertIn(f"element face {len(s.faces)}", head)


@needs_sionna
class TestConverterWithSionna(unittest.TestCase):
    """The one-wall room ray-traced by Sionna against Room Sim's paths."""

    @classmethod
    def setUpClass(cls):
        cls.scene = one_wall_scene()
        cls.cfg = Config(floor_ceiling=False, clutter_count=0, tail=False)
        cls.conv = s2s.convert(cls.scene, cls.cfg)
        cls.tmp = tempfile.TemporaryDirectory()
        cls.sc = s2s.load_sionna_scene(cls.conv, cls.tmp.name)
        cls.links = s2s.solve_paths(cls.sc, [0, 1], max_depth=2, samples_per_src=100_000,
                                    max_num_paths_per_src=100_000)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def test_paths_match_roomsim(self):
        R = rss.roomsim_links(self.scene, self.cfg)[(0, 1)]
        S = self.links[(0, 1)]
        self.assertEqual(sorted(R.kind.tolist()), [0, 1])
        self.assertEqual(sorted(S.n_of("specular").tolist()), [0, 1])
        for kind, nref in ((0, 0), (1, 1)):
            i = int(np.nonzero(R.kind == kind)[0][0])
            j = int(np.nonzero(S.n_of("specular") == nref)[0][0])
            self.assertAlmostEqual(R.L[i], S.tau[j] * C, delta=1e-4)
            self.assertLess(abs(R.coef[i] - S.a[j]) / abs(R.coef[i]), 1e-5)
        self.assertEqual(S.reflections(), [(), ("wall-0",)] if S.n_of("specular")[0] == 0 else [("wall-0",), ()])

    def test_cfr_matches_roomsim_channel(self):
        """Room Sim's own §11 channel (no people, no clutter, no tail) at the
        52 LLTF bins against the CFR of Sionna's paths."""
        R = rss.roomsim_links(self.scene, self.cfg)[(0, 1)]
        S = self.links[(0, 1)]
        H_s = M.cfr(S.a, S.tau, self.conv.frequencies)
        self.assertLess(np.max(np.abs(H_s - R.H)) / np.max(np.abs(R.H)), 1e-4)
        self.assertLess(np.max(np.abs(M.cfr(R.coef, R.tau, self.conv.frequencies) - R.H)) / np.max(np.abs(R.H)),
                        1e-12)

    def test_sionna_cfr_convention(self):
        """Sionna's `Paths.cfr` takes baseband offsets and applies the
        carrier phase itself: it equals the passband sum used here."""
        import mitsuba as mi
        import sionna.rt as rt
        sc = self.sc
        for name in list(sc.receivers):
            sc.remove(name)
        sc.add(rt.Receiver(name="rx-1", position=mi.Point3f(*self.conv.nodes[1])))
        try:
            paths = rt.PathSolver()(sc, max_depth=1, samples_per_src=100_000, max_num_paths_per_src=100_000)
            k = np.array(s2s.LLTF_BINS, dtype=np.float64) * 312500.0
            Hc = paths.cfr(frequencies=k, normalize_delays=False, out_type="numpy")   # complex ndarray
            H_sionna = np.asarray(Hc)[0, 0, 0, 0, 0, :]          # rx 1, tx 0
            S = self.links[(0, 1)]
            H = M.cfr(S.a, S.tau, self.conv.frequencies)
            self.assertLess(np.max(np.abs(H_sionna - H)) / np.max(np.abs(H)), 1e-3)
        finally:
            for name in list(sc.receivers):
                sc.remove(name)
            for nid in (0, 1):
                sc.add(rt.Receiver(name=f"rx-{nid}", position=mi.Point3f(*self.conv.nodes[nid])))


class TestMetrics(unittest.TestCase):

    def test_delay_spread(self):
        self.assertAlmostEqual(M.rms_delay_spread([1.0, 1.0], [0.0, 10e-9]), 5e-9, places=18)
        self.assertEqual(M.rms_delay_spread([2.0], [7e-9]), 0.0)
        self.assertTrue(math.isnan(M.rms_delay_spread([], [])))

    def test_selectivity_and_agreement(self):
        f = s2s.lltf_frequencies(2437e6)
        H1 = M.cfr([1e-3], [10e-9], f)
        self.assertAlmostEqual(M.selectivity_db(H1), 0.0, places=9)
        H2 = M.cfr([1e-3, 5e-4], [10e-9, 40e-9], f)
        a = M.cfr_agreement(H2, H2)
        self.assertAlmostEqual(a["corr"], 1.0, places=12)
        self.assertAlmostEqual(a["rms_db"], 0.0, places=12)
        b = M.cfr_agreement(H2, 2.0 * H2)
        self.assertAlmostEqual(b["offset_db"], -20 * math.log10(2.0), places=9)
        self.assertAlmostEqual(b["rms_db"], 0.0, places=9)
        self.assertTrue(math.isnan(M.cfr_agreement(H1, H1)["corr"]))       # flat: undefined

    def test_match_paths(self):
        m = M.match_paths([(), ("w",), ("w", "f")], [1.0, 2.0, 3.0], [("w",), (), ("x",)], [2.01, 1.0, 4.0])
        self.assertEqual([(i, j) for i, j, _d in m["pairs"]], [(0, 1), (1, 0)])
        self.assertAlmostEqual(m["pairs"][1][2], -0.01)
        self.assertEqual(m["rs_only"], [2])
        self.assertEqual(m["s_only"], [2])

    def test_summarize(self):
        s = M.summarize([1.0, 2.0, 3.0, float("nan"), float("inf")])
        self.assertEqual((s["label"], s["n"], s["n_nonfinite"], s["median"]), ("SIM", 3, 2, 2.0))


@needs_sionna
class TestTinyAudit(unittest.TestCase):
    """A small audit: the hand-written bedroom, then one SIM dataset scene."""

    @classmethod
    def setUpClass(cls):
        pass

    def _opts(self, **kw):
        from ml.sionna_audit import audit as A
        o = argparse.Namespace(variants=["shell", "furnished"], spec_depth=4, spec_samples=200_000,
                               spec_cap=1_000_000, diffuse_s=[0.3], diffuse_depth=2, diffuse_samples=20_000,
                               diffuse_cap=2_000_000, sionna_seed=A.SIONNA_SEED)
        for k, v in kw.items():
            setattr(o, k, v)
        return o

    def test_bedroom(self):
        from ml.sionna_audit import audit as A
        scene = load_scene(BEDROOM)
        cfg = Config()
        res = A.audit_scene(scene, cfg, {"id": "bedroom"}, self._opts(), log=lambda *_a: None)
        shell = res["variants"]["shell"]
        self.assertEqual(len(shell["records"]), 12)
        for r in shell["records"]:
            self.assertLess(abs(r["d_los_db"]), 1e-3, r)              # the same LoS in both
            self.assertEqual(r["n_s_only"], 0, r)
            self.assertLess(abs(r["d_spec_db"]), 1.0, r)
        los = [m for m in shell["matched"] if m[0] == "los"]
        self.assertEqual(len(los), 12)
        order1 = [m for m in shell["matched"] if m[0] == "order1_floor_ceiling"]
        self.assertTrue(order1 and all(abs(m[1]) < 1e-3 for m in order1))   # pure TM: exact
        furn = res["variants"]["furnished"]
        self.assertEqual(len(furn["records"]), 12)
        full = res["full"]["S0.3"]
        self.assertEqual(len(full), 12)
        for r in full:
            self.assertGreater(r["ds_rs_full_ns"], 0.0)
            self.assertGreater(r["ds_s_full_ns"], 0.0)
            self.assertLess(r["roomsim_cfr_check_rel_err"], 1e-9)       # numpy CFR = Room Sim's own H
        # Summaries carry the label.
        summ = A.summarize_records(shell["records"], A.SPEC_METRICS)
        self.assertTrue(all(v["label"] == "SIM" for v in summ.values()))
        self.assertTrue(all(v["label"] == "SIM" for v in A.summarize_matched(shell["matched"]).values()))

    @unittest.skipUnless(os.path.isfile(os.path.join(DATASET, "manifest.json")), "SIM dataset v1 not present")
    def test_dataset_run(self):
        from ml import roomsim_dataset as rd
        from ml.sionna_audit import audit as A
        manifest = rd.load_manifest(DATASET)
        chosen, _forced = A.choose_scenes(manifest)
        self.assertEqual(chosen, A.choose_scenes(manifest)[0])          # fixed seed
        entries = {int(e["index"]): e for e in manifest["scenes"]}
        self.assertTrue(any(entries[i]["l_shape"] for i in chosen))
        self.assertTrue(all(entries[i]["split"] == "test" for i in chosen))
        small = min(rd.splits(manifest)["test"], key=lambda i: (entries[i]["n_nodes"], i))
        with tempfile.TemporaryDirectory() as d:
            out = os.path.join(d, "audit.json")
            md = os.path.join(d, "audit.md")
            argv = ["--dataset", DATASET, "--indices", str(small), "--quick", "--variants", "shell",
                    "--diffuse-s", "--no-slab-check", "--out", out, "--markdown", md]
            with open(os.devnull, "w") as null:
                old = sys.stdout
                sys.stdout = null
                try:
                    A.main(argv)
                finally:
                    sys.stdout = old
            with open(out, encoding="utf-8") as f:
                doc = json.load(f)
            self.assertEqual(doc["label"], "SIM")
            self.assertEqual(doc["variants"]["shell"]["label"], "SIM")
            self.assertEqual([s["index"] for s in doc["dataset"]["scenes"]], [small])
            o = doc["variants"]["shell"]["overall"]
            self.assertLess(abs(o["d_los_db"]["median"]), 1e-3)
            with open(md, encoding="utf-8") as f:
                self.assertIn("All numbers SIM", f.read())


if __name__ == "__main__":
    unittest.main()
