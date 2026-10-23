"""Room Sim physics against independent ground truth (spec/roomsim.md §15).

Materials (§5), the three-edge knife-edge shadow (§8.5), first-order
images in the incident faces of a rectangle's walls (§6.1, §6.3), the
specular law, the diffuse tail's amplitude (§6.8), walls and objects on a
link, and a person on the line of sight.
"""

import json
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roomsim_helpers import BEDROOM, HALL, MULTI, OFFICE, make_scene, rect_walls  # noqa: E402

from wifisense.roomsim import Config, RoomSim, load_scene, parse_scene  # noqa: E402
from wifisense.roomsim.channel import PersonSnapshot  # noqa: E402
from wifisense.roomsim.geometry import dot3, unit3  # noqa: E402
from wifisense.roomsim.materials import ITU, Slab, eta, fresnel, slab  # noqa: E402
from wifisense.roomsim.numerics import (  # noqa: E402
    C, FOUR_PI, cabs2, cadd, carrier_hz, cdiv, cmul, cscale, csqrt, csub,
)
from wifisense.roomsim.paths import (  # noqa: E402
    KIND_LOS, KIND_OBJ_POINT, KIND_OBJ_SPECULAR, KIND_ORDER1, KIND_ORDER2, KIND_TAIL,
)
from wifisense.roomsim.people import J1, shadow_J, three_edge_J  # noqa: E402

F6 = carrier_hz(6)
LAM = C / F6
ANGLES = (0.0, 30.0, 60.0, 85.0)


def cos_deg(a):
    return math.cos((a * math.pi) / 180.0)


def cabs(z):
    return math.sqrt(cabs2(z))


def knife_edge(nu):
    """ITU-R P.526 single edge, written out here (not the simulator's J1)."""
    if nu <= -0.78:
        return 0.0
    return 6.9 + 20.0 * math.log10(math.sqrt((nu - 0.1) ** 2 + 1.0) + nu - 0.1)


def finite_screen_J(nu_1, nu_2, nu_t, cap):
    """P.526's finite-width screen over three edges, clamped to 0..cap."""
    s = sum(10.0 ** (-knife_edge(nu) / 10.0) for nu in (nu_1, nu_2, nu_t))
    return min(max(-10.0 * math.log10(s), 0.0), cap)


class TestMaterials(unittest.TestCase):
    def test_normal_incidence_closed_form(self):
        """G_TE = (1 - sqrt(eta)) / (1 + sqrt(eta)) and G_TM = -G_TE."""
        for m in ITU:
            et = eta(m, F6)
            _s, g_te, g_tm = fresnel(et, 1.0)
            r = csqrt(et)
            want = cdiv(csub((1.0, 0.0), r), cadd((1.0, 0.0), r))
            self.assertLess(cabs(csub(g_te, want)), 1e-12, m)
            self.assertLess(cabs(cadd(g_tm, g_te)), 1e-12, m)

    def test_lossless_slab_conserves_energy(self):
        for er in (1.5, 2.73, 5.24, 9.0):
            for a in ANGLES:
                ct = cos_deg(a)
                s, g_te, g_tm = fresnel((er, 0.0), ct)
                for G in (g_te, g_tm):
                    for t in (0.05, 0.1, 0.23, 0.4):
                        R, T, _q0 = slab(G, s, ct, t, LAM)
                        self.assertAlmostEqual(cabs2(R) + cabs2(T), 1.0, delta=1e-12)

    def test_half_wave_slab_is_transparent_to_reflection(self):
        for er in (2.73, 4.0, 6.31):
            t = LAM / (2.0 * math.sqrt(er))
            s, g_te, g_tm = fresnel((er, 0.0), 1.0)
            for G in (g_te, g_tm):
                R, T, _ = slab(G, s, 1.0, t, LAM)
                self.assertLess(cabs(R), 1e-12)
                self.assertAlmostEqual(cabs(T), 1.0, delta=1e-12)

    def test_free_space_slab(self):
        """eta = (1, 0): no reflection, and T = 1 (T carries e^{+j q0})."""
        for a in ANGLES + (45.0, 89.0):
            ct = cos_deg(a)
            s, g_te, g_tm = fresnel((1.0, 0.0), ct)
            for G in (g_te, g_tm):
                for t in (0.1, 0.23):
                    R, T, _ = slab(G, s, ct, t, LAM)
                    self.assertLess(cabs(R), 1e-12)
                    self.assertLess(cabs(csub(T, (1.0, 0.0))), 1e-12)

    def test_lossy_slabs_lose_energy_and_metal_reflects(self):
        for m in ("concrete", "brick", "drywall", "glass", "wood"):
            sl = Slab(m, 0.23, "TE", F6, LAM)
            for a in ANGLES:
                _G, R, T = sl.coefficients(cos_deg(a))
                self.assertLess(cabs2(R) + cabs2(T), 1.0)
        _G, R, T = Slab("metal", 0.1, "TE", F6, LAM).coefficients(1.0)
        self.assertGreater(cabs(R), 0.999)
        self.assertLess(cabs(T), 1e-12)

    def test_every_surface_reflects_with_the_plain_slab_R(self):
        """§5.1: R refers to the incident face, where §6.1 takes every
        bounce, so walls, floor and ceiling all use the slab's R as is."""
        for m in ("brick", "concrete", "drywall"):
            for pol in ("TE", "TM"):
                sl = Slab(m, 0.23, pol, F6, LAM)
                for a in ANGLES:
                    ct = cos_deg(a)
                    s_, g_te, g_tm = fresnel(eta(m, F6), ct)
                    R, T, _q0 = slab(g_te if pol == "TE" else g_tm, s_, ct, 0.23, LAM)
                    self.assertEqual(sl.reflect(ct), R)
                    self.assertEqual(sl.transmit(ct), T)

    def test_concrete_wall_attenuates_more_than_drywall(self):
        def link_power(material):
            walls = rect_walls(6.0, 5.0) + [{"from": [3.0, 0.0], "to": [3.0, 5.0], "material": material,
                                             "thickness_m": 0.15}]
            sc = make_scene(walls=walls)
            cfg = Config(duration_s=0.0, clutter_count=0, tail=False)
            lm = RoomSim(sc, cfg).link_model(0, 1)
            los = [p for p in lm.paths if p.kind == KIND_LOS][0]
            H = lm.channel([])
            return cabs(los.coef), sum(cabs2(h) for h in H) / 64.0

        los_c, p_c = link_power("concrete")
        los_d, p_d = link_power("drywall")
        self.assertLess(los_c, los_d)
        self.assertLess(p_c, p_d)


class TestShadow(unittest.TestCase):
    def test_J1_at_zero(self):
        want = 6.9 + 20.0 * math.log10(math.sqrt(1.01) - 0.1)
        self.assertEqual(J1(0.0), want)
        self.assertAlmostEqual(J1(0.0), 6.03, delta=0.005)

    def test_J1_shape(self):
        self.assertEqual(J1(-0.78), 0.0)
        self.assertEqual(J1(-2.0), 0.0)
        prev = J1(-0.77)
        for i in range(1, 40):
            v = J1(-0.77 + 0.1 * i)
            self.assertGreater(v, prev)
            prev = v

    def test_three_edges_and_cap(self):
        # every edge clear: each term is 1, -10 log10(3) < 0 clamps to 0
        self.assertEqual(three_edge_J(-5.0, -5.0, -5.0, 20.0), 0.0)
        # one edge clear is enough for no loss
        self.assertEqual(three_edge_J(-1.0, 3.0, 3.0, 20.0), 0.0)
        self.assertEqual(three_edge_J(3.0, 3.0, -1.0, 20.0), 0.0)
        self.assertEqual(three_edge_J(10.0, 10.0, 10.0, 20.0), 20.0)
        # three equal edges: J = J1 - 10 log10(3)
        self.assertAlmostEqual(three_edge_J(1.0, 1.0, 1.0, 50.0), J1(1.0) - 10.0 * math.log10(3.0), delta=1e-12)
        for nus in ((0.3, 0.9, 2.0), (-0.5, 1.5, 0.7), (1.2, 1.4, -0.7), (2.5, 2.5, 2.5)):
            self.assertAlmostEqual(three_edge_J(*nus, 20.0), finite_screen_J(*nus, 20.0), delta=1e-12)

    def test_through_the_axis_of_a_4m_link(self):
        """§8.5: a leg through the axis below the head loses about 8.7 dB."""
        J = shadow_J((0.0, 0.0, 1.2), (4.0, 0.0, 1.2), (2.0, 0.0), 1.7, 0.18, 20.0, LAM)
        self.assertAlmostEqual(J, 8.7, delta=0.15)

    def test_cylinder_geometry(self):
        P1, P2 = (0.0, 0.0, 1.2), (4.0, 0.0, 1.2)
        # on the leg, mid-way: d_perp = 0, so both side edges are r * k
        J = shadow_J(P1, P2, (2.0, 0.0), 1.7, 0.18, 20.0, LAM)
        k = math.sqrt(2.0 * 4.0 / (LAM * 2.0 * 2.0))
        self.assertAlmostEqual(J, finite_screen_J(0.18 * k, 0.18 * k, 0.5 * k, 20.0), delta=1e-12)
        # off the axis by d = 0.1 at u* = 1/4: near edge (r - d) * k, far edge (r + d) * k
        k = math.sqrt(2.0 * 4.0 / ((LAM * 1.0) * 3.0))
        J = shadow_J(P1, P2, (1.0, 0.1), 1.7, 0.18, 20.0, LAM)
        self.assertAlmostEqual(J, finite_screen_J(0.08 * k, 0.28 * k, 0.5 * k, 20.0), delta=1e-12)
        # beside the body (d > r) the near edge is clear: no loss
        self.assertEqual(shadow_J(P1, P2, (2.0, 0.6), 1.7, 0.18, 20.0, LAM), 0.0)
        # beside the leg's ends, no loss; inside the cylinder at an end, the cap
        self.assertIsNone(shadow_J(P1, P2, (-1.0, 0.0), 1.7, 0.18, 20.0, LAM))
        self.assertEqual(shadow_J(P1, P2, (4.05, 0.0), 1.7, 0.18, 20.0, LAM), 20.0)
        self.assertIsNone(shadow_J(P1, P2, (4.05, 0.0), 1.0, 0.18, 20.0, LAM))  # end above the top
        self.assertIsNone(shadow_J((1.0, 1.0, 0.0), (1.0, 1.0, 2.0), (1.0, 1.0), 1.7, 0.18, 20.0, LAM))


class TestPaths(unittest.TestCase):
    def test_rectangle_first_order_images(self):
        """Order-1 bounces in the bedroom are taken on each wall's inner
        face, t/2 in from its centreline (§6.1): they have the closed-form
        lengths of the face images, lie on the face, obey the specular law
        and reflect with the plain slab R (§5.1)."""
        sc = load_scene(BEDROOM)
        sim = RoomSim(sc, Config(duration_s=0.0))
        W, D, H = 4.2, 3.6, 2.7
        # walls 0..2 brick 0.23 m (y = 0, x = W, y = D), wall 3 drywall 0.1 m (x = 0)
        tb, td = 0.23, 0.1
        for tx, rx in ((0, 1), (0, 2), (1, 3), (2, 3)):
            T = sim.world.node_pos[tx]
            R = sim.world.node_pos[rx]
            lm = sim.link_model(tx, rx)
            dx, dy, dz = R[0] - T[0], R[1] - T[1], R[2] - T[2]
            closed = {
                0: math.sqrt((dx * dx + (R[1] + T[1] - tb) ** 2) + dz * dz),              # face y = tb/2
                1: math.sqrt(((2 * W - tb - R[0] - T[0]) ** 2 + dy * dy) + dz * dz),      # face x = W - tb/2
                2: math.sqrt((dx * dx + (2 * D - tb - R[1] - T[1]) ** 2) + dz * dz),      # face y = D - tb/2
                3: math.sqrt(((R[0] + T[0] - td) ** 2 + dy * dy) + dz * dz),              # face x = td/2
                4: math.sqrt((dx * dx + dy * dy) + (R[2] + T[2]) ** 2),                   # floor
                5: math.sqrt((dx * dx + dy * dy) + (2 * H - R[2] - T[2]) ** 2),           # ceiling
            }
            faces = {0: (1, tb / 2.0), 1: (0, W - tb / 2.0), 2: (1, D - tb / 2.0), 3: (0, td / 2.0),
                     4: (2, 0.0), 5: (2, H)}
            got = {p.surf_i: p for p in lm.paths if p.kind == KIND_ORDER1}
            self.assertEqual(sorted(got), [0, 1, 2, 3, 4, 5])
            for i, L in closed.items():
                p = got[i]
                self.assertAlmostEqual(p.L, L, delta=1e-12)
                axis, coord = faces[i]
                self.assertAlmostEqual(p.points[1][axis], coord, delta=1e-12)
                s = sim.world.surfaces[i]
                n = s.n
                u_in = unit3(p.points[0], p.points[1])
                u_out = unit3(p.points[1], p.points[2])
                # specular: u_out = u_in - 2 (u_in . n) n
                d = dot3(u_in, n)
                mirror = (u_in[0] - 2 * d * n[0], u_in[1] - 2 * d * n[1], u_in[2] - 2 * d * n[2])
                for a, b in zip(mirror, u_out):
                    self.assertAlmostEqual(a, b, delta=1e-12)
                ct = abs(d)
                self.assertEqual(p.coef0, cscale(s.slab.reflect(ct), LAM / (FOUR_PI * p.L)))

    def test_a_wall_bounces_on_the_face_each_side_sees(self):
        """§6.1/§6.3: a partition reflects on its +n face for nodes on that
        side and on its -n face for nodes on the other; a link across it
        has no bounce on it, and its LoS crosses it with T (§6.9)."""
        part = {"from": [3.0, 1.0], "to": [3.0, 4.0], "material": "brick", "thickness_m": 0.2}
        nodes = [{"id": 0, "pos": [1.0, 2.5, 1.2]}, {"id": 1, "pos": [2.0, 1.5, 1.2]},
                 {"id": 2, "pos": [5.0, 2.5, 1.2]}, {"id": 3, "pos": [4.0, 3.5, 1.2]}]
        sc = make_scene(walls=rect_walls(6.0, 5.0) + [part], nodes=nodes)
        sim = RoomSim(sc, Config(duration_s=0.0, clutter_count=0, tail=False))
        wall = sim.world.surfaces[4]
        self.assertEqual(wall.n, (-1.0, 0.0, 0.0))   # +n points to x < 3
        for (a, b), face_x, img_x in (((0, 1), 2.9, 2.0 * 2.9 - 1.0), ((2, 3), 3.1, 2.0 * 3.1 - 5.0)):
            lm = sim.link_model(a, b)
            (p,) = [q for q in lm.paths if q.kind == KIND_ORDER1 and q.surf_i == 4]
            T, R = lm.Tp, lm.Rp
            self.assertAlmostEqual(p.points[1][0], face_x, delta=1e-12)
            self.assertTrue(1.0 <= p.points[1][1] <= 4.0)
            want = math.sqrt((R[0] - img_x) ** 2 + (R[1] - T[1]) ** 2)
            self.assertAlmostEqual(p.L, want, delta=1e-12)
            # the bounce is on this wall only: no other wall's T on its legs
            self.assertEqual(p.coef, p.coef0)
        across = sim.link_model(0, 2)
        self.assertEqual([q for q in across.paths if q.kind == KIND_ORDER1 and q.surf_i == 4], [])
        # (an order-2 path may still bounce off the far wall and then the
        # partition's other face, crossing the partition on its first leg)
        for q in across.paths:
            if q.kind == KIND_ORDER2 and q.surf_j == 4:
                self.assertAlmostEqual(q.points[2][0], 3.1, delta=1e-12)
        los = [q for q in across.paths if q.kind == KIND_LOS][0]
        u = unit3(across.Tp, across.Rp)
        self.assertEqual(los.coef, cmul(los.coef0, wall.slab.transmit(abs(dot3(u, wall.n)))))

    def test_tail_terms_share_one_amplitude(self):
        """§6.8: every tail term has amplitude
        sqrt((10^(tail_power_db/10) * P_spec) / tail_count), so the tail's
        power is exactly 10^(tail_power_db/10) of the specular power."""
        sim = RoomSim(load_scene(BEDROOM), Config(duration_s=0.0))
        cfg = sim.cfg
        for tx, rx in ((0, 1), (2, 0)):
            lm = sim.link_model(tx, rx)
            spec = [p for p in lm.paths if p.kind <= KIND_OBJ_SPECULAR]
            P_spec = math.fsum(cabs2(p.coef) for p in spec)
            L_ref = max(p.L for p in spec)
            self.assertEqual(len(lm.tail), cfg.tail_count)
            a = math.sqrt(10.0 ** (cfg.tail_power_db / 10.0) * P_spec / cfg.tail_count)
            for p in lm.tail:
                self.assertEqual(p.kind, KIND_TAIL)
                self.assertAlmostEqual(cabs(p.coef), a, delta=1e-14 * a)
                self.assertGreater(p.L, L_ref)
            P_tail = math.fsum(cabs2(p.coef) for p in lm.tail)
            self.assertAlmostEqual(P_tail / P_spec, 10.0 ** (cfg.tail_power_db / 10.0), delta=1e-12)
        # reciprocal: both directions of a pair share the draws and the amplitude
        ab, ba = sim.link_model(0, 2).tail, sim.link_model(2, 0).tail
        for p, q in zip(ab, ba):
            self.assertAlmostEqual(p.L, q.L, delta=1e-12 * p.L)
            self.assertAlmostEqual(cabs(csub(p.coef, q.coef)), 0.0, delta=1e-12 * cabs(p.coef))

    def test_order2_paths_obey_the_specular_law(self):
        sim = RoomSim(load_scene(BEDROOM), Config(duration_s=0.0))
        lm = sim.link_model(0, 2)
        o2 = [p for p in lm.paths if p.kind == 2]
        self.assertGreater(len(o2), 10)
        for p in o2:
            for leg, surf in ((0, p.surf_i), (1, p.surf_j)):
                n = sim.world.surfaces[surf].n
                u_in = unit3(p.points[leg], p.points[leg + 1])
                u_out = unit3(p.points[leg + 1], p.points[leg + 2])
                d = dot3(u_in, n)
                for c in range(3):
                    self.assertAlmostEqual(u_in[c] - 2 * d * n[c], u_out[c], delta=1e-12)

    def test_metal_box_on_the_los_weakens_the_link(self):
        box = {"type": "almirah", "pos": [3.0, 2.5, 1.0], "size": [0.6, 0.6, 2.0], "yaw_deg": 0.0,
               "material": "metal"}
        cfg = Config(duration_s=0.0, clutter_count=0, tail=False)
        with_box = RoomSim(make_scene(walls=rect_walls(6.0, 5.0), objects=[box]), cfg).link_model(0, 1)
        without = RoomSim(make_scene(walls=rect_walls(6.0, 5.0)), cfg).link_model(0, 1)
        los_b = [p for p in with_box.paths if p.kind == KIND_LOS][0]
        los_n = [p for p in without.paths if p.kind == KIND_LOS][0]
        self.assertAlmostEqual(cabs(los_b.coef) / cabs(los_n.coef), 10.0 ** (-20.0 / 20.0), delta=1e-12)
        p_b = sum(cabs2(h) for h in with_box.channel([]))
        p_n = sum(cabs2(h) for h in without.channel([]))
        self.assertLess(p_b, p_n)
        # No face radiates through the box: a face takes part only with both
        # nodes in front of it, and a segment that crosses a convex box has
        # its ends on the inner side of at least one face plane each, so no
        # face of a box the LoS crosses can have both nodes in front.
        Tp, Rp = with_box.Tp, with_box.Rp
        for p in with_box.paths:
            self.assertNotIn(p.kind, (KIND_OBJ_SPECULAR, KIND_OBJ_POINT))
        # ...whereas a box beside the LoS does light faces.
        beside = dict(box, pos=[3.0, 1.0, 1.0])
        lm = RoomSim(make_scene(walls=rect_walls(6.0, 5.0), objects=[beside]), cfg).link_model(0, 1)
        faces = [p for p in lm.paths if p.kind in (KIND_OBJ_SPECULAR, KIND_OBJ_POINT)]
        self.assertEqual([p.face for p in faces], [2])
        for p in faces:
            n = lm.world.objects[0].box.rotate(0.0, 1.0, 0.0)
            pf = p.points[1] if p.kind == KIND_OBJ_POINT else (3.0, 1.3, 1.0)
            self.assertGreater(dot3((Tp[0] - pf[0], Tp[1] - pf[1], Tp[2] - pf[2]), n), 0.0)
            self.assertGreater(dot3((Rp[0] - pf[0], Rp[1] - pf[1], Rp[2] - pf[2]), n), 0.0)

    def test_person_on_the_los_shadows_the_los_and_paths_through_them(self):
        """§15: a person on a link's LoS lowers the LoS by exactly
        10^(-J/20), J the three-edge §8.5 loss computed here independently;
        a path whose legs all pass well clear of them in plan, or above
        their top, is unchanged; a path whose legs pass through the
        cylinder (the floor bounce at their feet) is shadowed too."""
        sc = make_scene(walls=rect_walls(6.0, 5.0))
        cfg = Config(duration_s=0.0, clutter_count=0, tail=False)
        lm = RoomSim(sc, cfg).link_model(0, 1)
        r, top = 0.18, 1.7
        clean = lm.static_coefficients([])
        for cy in (2.5, 2.6):
            c = (3.0, cy)
            q = PersonSnapshot(0, c[0], c[1], 0.0, 0, 0.0, [], (), top, (), "stand")
            shadowed = lm.static_coefficients([q])
            # the LoS (1, 2.5, 1.2) -> (5, 2.5, 1.2): u* = 1/2, d1 = d2 = 2, d_perp = |cy - 2.5|
            k = math.sqrt(2.0 * 4.0 / (LAM * 2.0 * 2.0))
            d = abs(cy - 2.5)
            J = finite_screen_J((r - d) * k, (r + d) * k, (top - 1.2) * k, 20.0)
            self.assertGreater(J, 5.0)
            n_clear = n_through = 0
            for p, a, b in zip(lm.paths, shadowed, clean):
                if p.kind == KIND_LOS:
                    self.assertAlmostEqual(cabs(a) / cabs(b), 10.0 ** (-J / 20.0), delta=1e-12)
                    self.assertAlmostEqual(math.atan2(a[1], a[0]), math.atan2(b[1], b[0]), delta=1e-12)
                    continue
                legs = list(zip(p.points[:-1], p.points[1:]))
                status = [self._leg_vs_cylinder(P1, P2, c, r, top) for P1, P2 in legs]
                if all(st == "clear" for st in status):
                    n_clear += 1
                    self.assertEqual(a, b, (p.kind, p.surf_i, p.surf_j))
                elif any(st == "through" for st in status):
                    n_through += 1
                    self.assertLess(cabs(a), cabs(b), (p.kind, p.surf_i, p.surf_j))
            self.assertGreaterEqual(n_clear, 5)
            self.assertGreater(n_through, 0)
        # the floor bounce passes the person's feet: shadowed
        floor = [i for i, p in enumerate(lm.paths) if p.kind == KIND_ORDER1 and p.surf_i == 4][0]
        q = PersonSnapshot(0, 3.0, 2.5, 0.0, 0, 0.0, [], (), top, (), "stand")
        self.assertLess(cabs(lm.static_coefficients([q])[floor]), cabs(clean[floor]))

    @staticmethod
    def _leg_vs_cylinder(P1, P2, c, r, top):
        """'clear' when no §8.5 edge can block the leg (every point of it
        farther than r + m from c in plan, or its plan-closest point higher
        than top + m, m = 0.78 * sqrt(LAMBDA * D / 8) the largest margin a
        P.526 edge needs), 'through' when it passes inside the cylinder
        between its ends, else 'near'."""
        rx, ry = P2[0] - P1[0], P2[1] - P1[1]
        L2 = rx * rx + ry * ry
        if L2 == 0.0:
            return "clear"
        D = math.sqrt(L2 + (P2[2] - P1[2]) ** 2)
        m = 0.78 * math.sqrt(LAM * D / 8.0)
        u = ((c[0] - P1[0]) * rx + (c[1] - P1[1]) * ry) / L2
        uc = min(max(u, 0.0), 1.0)
        px, py = P1[0] + uc * rx - c[0], P1[1] + uc * ry - c[1]
        dist = math.sqrt(px * px + py * py)
        z = P1[2] + uc * (P2[2] - P1[2])
        if dist >= r + m or z >= top + m:
            return "clear"
        if not 0.0 < u < 1.0:
            return "clear" if (dist >= r or z > top) else "near"
        if dist < r and z < top:
            return "through"
        return "near"

    def test_reciprocity(self):
        """H(a->b) == H(b->a) before impairments, people included (§15).

        The shipped scenes' round coordinates put some paths exactly on the
        edge between two surfaces or a leg exactly on a box edge, admitted
        one way and not the other by the last bit (bedroom (0, 1), the
        multi fixture's (0, 2), ...), so every node is moved a few
        centimetres first and every pair is checked.
        """
        from wifisense.roomsim.channel import snapshot_people
        from wifisense.roomsim.people import build_people
        for path in (BEDROOM, HALL, OFFICE, MULTI):
            with open(path) as fh:
                doc = json.load(fh)
            for i, n in enumerate(doc["nodes"]):
                x, y, z = n["pos"]
                n["pos"] = [x + 0.019 * math.sin(1.7 * i + 0.3), y + 0.019 * math.cos(2.3 * i + 0.1),
                            z + 0.013 * math.sin(0.9 * i + 1.0)]
            sim = RoomSim(parse_scene(json.dumps(doc)), Config(duration_s=0.0))
            people = build_people(sim.world, sim.cfg, sim.scene)
            ids = sim.world.node_ids
            for t in (0.0, 1.3):
                snaps = snapshot_people(people, sim.cfg, t)
                for i, a in enumerate(ids):
                    for b in ids[i + 1:]:
                        hab = sim.link_model(a, b).channel(snaps)
                        hba = sim.link_model(b, a).channel(snaps)
                        M = max(cabs(h) for h in hab)
                        for x, y in zip(hab, hba):
                            self.assertLess(cabs(csub(x, y)), 1e-9 * M, (path, a, b, t))


if __name__ == "__main__":
    unittest.main()
