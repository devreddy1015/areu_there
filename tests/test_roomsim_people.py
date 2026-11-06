"""Room Sim people and motion (spec/roomsim.md §8-9)."""

import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roomsim_helpers import BEDROOM, HALL, MULTI, OFFICE, make_scene, rect_walls  # noqa: E402

from wifisense.roomsim import Config, RoomSim, load_scene  # noqa: E402
from wifisense.roomsim.channel import snapshot_people  # noqa: E402
from wifisense.roomsim.motion import walkable_leg, walkable_point  # noqa: E402
from wifisense.roomsim.people import BODY_TABLE  # noqa: E402


def point_in_poly(poly, p):
    """Independent even-odd test (no shared code with the simulator)."""
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x0, y0 = poly[i]
        x1, y1 = poly[(i + 1) % n]
        if (y0 > y) != (y1 > y) and x < x0 + (y - y0) * (x1 - x0) / (y1 - y0):
            inside = not inside
    return inside


def seg_point_dist(p, a, b):
    ax, ay = a
    bx, by = b
    dx, dy = bx - ax, by - ay
    t = max(0.0, min(1.0, ((p[0] - ax) * dx + (p[1] - ay) * dy) / (dx * dx + dy * dy)))
    ex, ey = p[0] - (ax + t * dx), p[1] - (ay + t * dy)
    return math.sqrt(ex * ex + ey * ey)


class TestModes(unittest.TestCase):
    def test_bedroom_sleeper_is_stationary_and_lying(self):
        sim = RoomSim(load_scene(BEDROOM), Config(duration_s=1.0))
        (p,) = sim.people_template
        self.assertFalse(p.walking)
        self.assertEqual(p.posture, "lie")
        self.assertEqual(sim.scene.objects[p.on_object].kind, "bed")
        truths = list(sim.truth())
        self.assertEqual(len(truths), 80)
        for tr in truths:
            pt = tr.people[0]
            self.assertEqual((pt.x, pt.y, pt.walking, pt.posture), (1.1, 2.6, False, "lie"))
        # body points lie on the bed: z = top + 0.12 + bx, |bx| <= 0.12 + p2p/2
        snaps = snapshot_people(sim.people_template, sim.cfg, 0.37)
        for pt in snaps[0].points:
            self.assertGreaterEqual(pt[2], 0.5 - 1e-12)
            self.assertLess(pt[2], 0.5 + 0.12 + 0.12 + 0.01)
        self.assertEqual(snaps[0].excl_obj, (p.on_object,))
        self.assertEqual(snaps[0].top, 0.75)

    def test_hall_walker_walks(self):
        sim = RoomSim(load_scene(HALL), Config(duration_s=20.0))
        walker = sim.people_template[0]
        self.assertTrue(walker.walking)
        self.assertEqual(walker.posture, "stand")
        truths = list(sim.truth())
        xs = {(round(tr.people[0].x, 6), round(tr.people[0].y, 6)) for tr in truths}
        self.assertGreater(len(xs), 100)
        self.assertTrue(any(tr.people[0].walking for tr in truths))

    def test_third_person_enters_at_the_door(self):
        sim = RoomSim(load_scene(MULTI), Config(duration_s=0.0))
        third = sim.people_template[2]
        self.assertTrue(third.walking)
        # the door is at (3.0, 0.0), 0.9 x 0.1: its normal axis is local y,
        # +y is into the room; the first walkable point is 0.4 m in
        # (walk_margin_m from the y = 0 edge).
        self.assertEqual(third.start, (3.0, 0.05 * 8.0))
        self.assertFalse(walkable_point(sim.world, sim.cfg, (3.0, 0.05 * 7.0)))

    def test_shipped_scenes_seat_their_occupants(self):
        """The office's occupants start on their chairs and the hall's
        people[1] on the sofa: stationary and sitting (§8.2)."""
        sim = RoomSim(load_scene(OFFICE), Config(duration_s=0.0))
        self.assertEqual(len(sim.people_template), 2)
        for p in sim.people_template:
            self.assertEqual((p.walking, p.posture), (False, "sit"))
            self.assertEqual(sim.scene.objects[p.on_object].kind, "chair")
        sim = RoomSim(load_scene(HALL), Config(duration_s=0.1))
        walker, sitter = sim.people_template
        self.assertEqual((walker.walking, walker.posture), (True, "stand"))
        self.assertEqual((sitter.walking, sitter.posture), (False, "sit"))
        self.assertEqual(sim.scene.objects[sitter.on_object].kind, "sofa")
        self.assertEqual(sitter.start, (7.0, 0.6))
        tr = next(sim.truth())
        self.assertEqual((tr.people[1].walking, tr.people[1].posture), (False, "sit"))

    def test_sitting_and_overrides(self):
        chair = {"type": "chair", "pos": [3.0, 1.5, 0.45], "size": [0.5, 0.5, 0.9], "material": "wood"}
        sc = make_scene(walls=rect_walls(6.0, 5.0), objects=[chair], people=[{"start": [3.0, 1.5]}, {"start": [2.0, 4.0]}])
        sim = RoomSim(sc, Config(duration_s=0.0, people_mode={1: "stationary"}, start_heading_rad={1: 1.0}))
        sitter, stander = sim.people_template
        self.assertEqual((sitter.walking, sitter.posture), (False, "sit"))
        self.assertEqual((stander.walking, stander.posture, stander.heading0), (False, "stand", 1.0))
        snaps = snapshot_people(sim.people_template, sim.cfg, 0.0)
        self.assertEqual(snaps[0].excl_obj, (0,))
        self.assertEqual(snaps[1].excl_obj, ())
        self.assertAlmostEqual(snaps[0].top, 0.72 * sitter.height, delta=1e-15)
        for j, pt in enumerate(snaps[0].points):
            self.assertAlmostEqual(pt[2], 0.72 * (BODY_TABLE[j][2] * sitter.height), delta=1e-15)
        sim2 = RoomSim(sc, Config(duration_s=0.0, people_mode={1: "stationary"}, people_posture={0: "stand"}))
        self.assertEqual(sim2.people_template[0].posture, "stand")

    def test_n_people_adds_door_walkers(self):
        sim = RoomSim(load_scene(BEDROOM), Config(duration_s=0.0, n_people=3))
        ps = sim.people_template
        self.assertEqual(len(ps), 3)
        self.assertEqual(ps[1].start, ps[2].start)  # both enter at the door
        self.assertTrue(ps[1].walking and ps[2].walking)
        sim0 = RoomSim(load_scene(BEDROOM), Config(duration_s=0.0, n_people=0))
        self.assertEqual(sim0.people_template, [])


class TestBody(unittest.TestCase):
    def test_rcs_and_breathing(self):
        sim = RoomSim(load_scene(BEDROOM), Config(duration_s=0.0, body_points=16))
        p = sim.people_template[0]
        self.assertAlmostEqual(sum(p.rcs), 0.5, delta=1e-15)
        self.assertTrue(1.55 <= p.height < 1.85)
        self.assertTrue(0.1 <= p.f_b < 0.3)
        # the chest rises and falls by p2p/2 on the bed (z = top + 0.12 + bx)
        zs = [snapshot_people([p], sim.cfg, 0.05 * i)[0].points[0][2] for i in range(400)]
        amp = (max(zs) - min(zs)) / 2.0
        self.assertAlmostEqual(amp, p.p2p / 2.0, delta=p.p2p * 0.02)

    # §8.3 rows written out here (x, y, z/height, swing amp, swing phase, breath):
    # chest, left hand, right hand, left foot, right foot.
    ROWS = {0: (0.12, 0.0, 0.72, 0.0, 0.0, 1.0), 4: (0.0, 0.22, 0.45, 0.25, math.pi, 0.0),
            5: (0.0, -0.22, 0.45, 0.25, 0.0, 0.0), 6: (0.0, 0.10, 0.05, 0.30, 0.0, 0.0),
            7: (0.0, -0.10, 0.05, 0.30, math.pi, 0.0)}

    @staticmethod
    def body_frame(pt, x, y, hd):
        """World point back to the body frame (forward, left)."""
        dx, dy = pt[0] - x, pt[1] - y
        return (dx * math.cos(hd) + dy * math.sin(hd), -dx * math.sin(hd) + dy * math.cos(hd))

    def test_walking_swing(self):
        """§8.3: sw = (amp * s) * sin((2 pi f_s) t + phi_g + phase), f_s =
        clamp(0.95 sqrt(v / 1.1), 0.5, 1.2) while walking: the hands swing in
        opposite directions, and nothing swings in a pause or for a
        stationary person."""
        sim = RoomSim(load_scene(HALL), Config(duration_s=0.0))
        p = sim.people_template[0]
        p.state(60.0)
        walk = [g for g in p.trajectory.segs if g.walk][0]
        pause = [g for g in p.trajectory.segs if not g.walk][0]
        swung = 0
        for f in (0.2, 0.45, 0.7):
            t = walk.t0 + f * (walk.t1 - walk.t0)
            st = p.state(t)
            x, y, hd, s, v = st
            self.assertEqual((s, v), (1, walk.v))
            pts = p.body_points(t, st, 10)
            self.assertEqual(len(pts), 10)
            f_s = min(max(0.95 * math.sqrt(v / 1.1), 0.5), 1.2)
            br = (p.p2p / 2.0) * math.sin(2.0 * math.pi * p.f_b * t + p.phi_b)
            for j, (bx, by, zf, amp, phase, breath) in self.ROWS.items():
                sw = amp * math.sin(2.0 * math.pi * f_s * t + p.phi_g + phase)
                fx, fy = self.body_frame(pts[j], x, y, hd)
                self.assertAlmostEqual(fx, bx + sw + breath * br, delta=1e-12)
                self.assertAlmostEqual(fy, by, delta=1e-12)
                self.assertAlmostEqual(pts[j][2], zf * p.height, delta=1e-12)
            left = self.body_frame(pts[4], x, y, hd)[0]
            right = self.body_frame(pts[5], x, y, hd)[0]
            self.assertAlmostEqual(left, -right, delta=1e-12)
            swung += abs(left) > 0.01
        self.assertGreater(swung, 0)
        # a pause: s = 0, the hands and feet hang at x = 0
        t = 0.5 * (pause.t0 + pause.t1)
        st = p.state(t)
        self.assertEqual(st[3], 0)
        pts = p.body_points(t, st, 10)
        for j in (4, 5, 6, 7):
            fx, _fy = self.body_frame(pts[j], st[0], st[1], st[2])
            self.assertAlmostEqual(fx, 0.0, delta=1e-12)
        # a stationary person (the hall's seated people[1]) never swings
        sitter = sim.people_template[1]
        self.assertFalse(sitter.walking)
        for t in (0.0, 0.37, 2.9):
            st = sitter.state(t)
            self.assertEqual(st[3], 0)
            pts = sitter.body_points(t, st, 10)
            for j in (4, 5, 6, 7):
                fx, _fy = self.body_frame(pts[j], st[0], st[1], st[2])
                self.assertAlmostEqual(fx, 0.0, delta=1e-12)


class TestTrajectories(unittest.TestCase):
    def test_waypoints_and_legs_avoid_furniture(self):
        for path in (HALL, MULTI):
            sim = RoomSim(load_scene(path), Config(duration_s=0.0))
            cfg = sim.cfg
            rg = cfg.body_radius_m + cfg.object_clearance_m
            F = sim.world.F
            walkers = [p for p in sim.people_template if p.walking]
            self.assertTrue(walkers)
            for person in sim.people_template:
                traj = person.trajectory
                if not person.walking:
                    # e.g. the hall's people[1], seated on the sofa (§8.2)
                    self.assertIsNone(traj)
                    continue
                self.assertIsNotNone(traj)
                traj.at(300.0)
                prev_t1 = 0.0
                walks = pauses = 0
                for seg in traj.segs:
                    self.assertEqual(seg.t0, prev_t1)
                    self.assertGreater(seg.t1, seg.t0)
                    prev_t1 = seg.t1
                    if seg.walk:
                        walks += 1
                        self.assertTrue(0.3 <= seg.v < 1.2)
                        self.assertTrue(walkable_leg(sim.world, cfg, seg.frm, seg.to))
                        ddx, ddy = seg.to[0] - seg.frm[0], seg.to[1] - seg.frm[1]
                        d = math.sqrt(ddx * ddx + ddy * ddy)
                        self.assertGreaterEqual(d, 0.5)
                        self.assertAlmostEqual(seg.t1 - seg.t0, d / seg.v, delta=1e-9)
                        # independent check along the leg
                        for i in range(41):
                            f = i / 40.0
                            q = (seg.frm[0] + (seg.to[0] - seg.frm[0]) * f, seg.frm[1] + (seg.to[1] - seg.frm[1]) * f)
                            self.assertTrue(point_in_poly(F, q))
                            for k in range(len(F)):
                                self.assertGreaterEqual(seg_point_dist(q, F[k], F[(k + 1) % len(F)]),
                                                        cfg.walk_margin_m - 1e-9)
                            for o in sim.world.objects:
                                if o.is_opening:
                                    continue
                                lx, ly, _ = o.box.to_local(q)
                                self.assertTrue(abs(lx) > o.box.sl / 2 + rg - 1e-9 or abs(ly) > o.box.sb / 2 + rg - 1e-9)
                    else:
                        pauses += 1
                        self.assertTrue(0.0 < seg.t1 - seg.t0 < 20.0 + 1e-9)
                self.assertGreater(walks, 5)
                self.assertGreater(pauses, 1)

    def test_positions_follow_the_segments(self):
        sim = RoomSim(load_scene(HALL), Config(duration_s=0.0))
        p = sim.people_template[0]
        for n in range(300):
            t = float(n) / 10.0
            x, y, hd, s, v = p.state(t)
            seg = [g for g in p.trajectory.segs if g.t0 <= t < g.t1][0]
            if seg.walk:
                self.assertEqual(s, 1)
                self.assertEqual(hd, math.atan2(seg.to[1] - seg.frm[1], seg.to[0] - seg.frm[0]))
            else:
                self.assertEqual((s, x, y), (0, seg.frm[0], seg.frm[1]))

    def test_evaluation_times_do_not_change_the_trajectory(self):
        a = RoomSim(load_scene(HALL), Config(duration_s=0.0)).people_template[0]
        b = RoomSim(load_scene(HALL), Config(duration_s=0.0)).people_template[0]
        a.state(100.0)
        for n in range(1000):
            b.state(n * 0.1)
        n = min(len(a.trajectory.segs), len(b.trajectory.segs))
        self.assertGreater(n, 5)
        for sa, sb in zip(a.trajectory.segs[:n], b.trajectory.segs[:n]):
            self.assertEqual((sa.t0, sa.t1, sa.frm, sa.to), (sb.t0, sb.t1, sb.frm, sb.to))


class TestPhone(unittest.TestCase):
    def test_phone_is_carried(self):
        sim = RoomSim(load_scene(MULTI), Config(duration_s=0.5))
        for tr in sim.truth():
            p = tr.people[0]
            ch, sh = math.cos(p.heading), math.sin(p.heading)
            self.assertEqual(tr.phone, (p.x + (0.05 * ch - (-0.25) * sh), p.y + (0.05 * sh + (-0.25) * ch), 1.0))
        sim = RoomSim(load_scene(MULTI), Config(duration_s=0.1, phone_carrier=7))
        self.assertEqual(next(sim.truth()).phone, (1.5, 1.5, 1.0))


if __name__ == "__main__":
    unittest.main()
