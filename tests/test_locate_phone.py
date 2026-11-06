"""spec/locate.md §6-§7: the simulated phone, its fixes and its filters.

Known-geometry checks (§12a) and the ordering rules the Rust port must
match: the epoch firing rule, the RNG streams and their draw order, the
shadowing that advances on failed reports, and the Kalman filter's steps
(coast reset at exactly `phone_max_coast_s`, a rejected fix keeps the
predicted state, no predict without a fix). Everything is SIM.
"""

from __future__ import annotations

import math
import unittest

from wifisense.localize.config import LocateConfig, LocateError
from wifisense.localize.phone import (
    EpochClock, Fix, PhoneKalman, PhoneMethod, PhoneRadio, PhoneRun, box_blocks,
    cylinder_blocks, rssi_fix, rtt_fix, run_phone, segment_blocked, wall_blocks,
)
from wifisense.localize.radius import CHI2_2_99, radius95
from wifisense.roomsim.rng import Rng

NAN = float("nan")
BEDROOM = [(0, 0.15, 0.15, 1.2), (1, 4.05, 0.15, 1.2), (2, 4.05, 3.45, 1.2), (3, 0.15, 3.45, 1.2)]
OFFICE = [(0, 0.2, 0.2, 1.5), (1, 5.8, 0.2, 1.5), (2, 5.8, 4.8, 1.5), (3, 0.2, 4.8, 1.5),
          (4, 3.0, 0.2, 1.5), (5, 3.0, 4.8, 1.5)]
HALL = [(0, 0.3, 0.3, 1.6), (1, 8.7, 0.3, 1.6), (2, 8.7, 2.7, 1.6), (3, 4.3, 2.7, 1.6),
        (4, 3.7, 3.3, 1.6), (5, 0.3, 3.3, 1.6), (6, 0.3, 6.7, 1.6), (7, 3.7, 6.7, 1.6)]
CARRIER = 2437e6


def cfg(**kw) -> LocateConfig:
    return LocateConfig(**kw).resolved(11, 10.0, CARRIER)


def assert_names(case, param, fn, *args, **kw):
    """`fn(*args, **kw)` raises `LocateError` naming `param`."""
    with case.assertRaises(LocateError) as cm:
        fn(*args, **kw)
    case.assertEqual(cm.exception.param, param)
    case.assertTrue(str(cm.exception).startswith(param + ":"), str(cm.exception))
    return cm.exception


def dist3(p, n):
    return math.sqrt((p[0] - n[1]) ** 2 + (p[1] - n[2]) ** 2 + (p[2] - n[3]) ** 2)


class TestLineOfSight(unittest.TestCase):
    def test_wall_centreline_blocks_and_a_doorway_does_not(self):
        # An interior wall along y = 3 with an opening from x = 2 to x = 3.
        walls = [(0.0, 3.0, 2.0, 3.0, 2.7), (3.0, 3.0, 6.0, 3.0, 2.7)]
        self.assertTrue(segment_blocked((1.0, 1.0, 1.0), (1.0, 5.0, 1.2), walls, [], []))
        self.assertFalse(segment_blocked((2.5, 1.0, 1.0), (2.5, 5.0, 1.2), walls, [], []))
        # Over the wall's top, not blocked; parallel to it, not blocked.
        self.assertFalse(wall_blocks((1.0, 1.0, 3.0), (1.0, 5.0, 3.0), walls[0]))
        self.assertFalse(wall_blocks((0.5, 3.0, 1.0), (1.5, 3.0, 1.0), walls[0]))

    def test_box(self):
        table = (1.0, 2.0, 0.375, 1.2, 0.6, 0.75, 1.0, 0.0)   # yaw 0
        self.assertTrue(box_blocks((0.0, 2.0, 0.3), (2.0, 2.0, 0.3), table))
        self.assertFalse(box_blocks((0.0, 2.0, 1.0), (2.0, 2.0, 1.0), table))
        # Yawed 90 degrees: the long side now runs along y.
        rot = (1.0, 2.0, 0.375, 1.2, 0.6, 0.75, 0.0, 1.0)
        self.assertTrue(box_blocks((1.0, 1.0, 0.3), (1.0, 3.0, 0.3), rot))
        self.assertFalse(box_blocks((0.5, 1.0, 0.3), (0.5, 3.0, 0.3), rot))

    def test_cylinder_below_and_above_its_top(self):
        cyl = (2.0, 2.0, 0.18, 1.7)
        self.assertTrue(cylinder_blocks((1.0, 2.0, 1.0), (3.0, 2.0, 1.2), cyl))
        self.assertFalse(cylinder_blocks((1.0, 2.0, 1.9), (3.0, 2.0, 1.9), cyl))
        self.assertFalse(cylinder_blocks((1.0, 2.3, 1.0), (3.0, 2.3, 1.0), cyl))
        # Nearer end inside the cylinder.
        self.assertTrue(cylinder_blocks((2.05, 2.0, 1.0), (4.0, 2.0, 1.0), cyl))
        self.assertTrue(cylinder_blocks((0.0, 2.0, 1.0), (2.05, 2.0, 1.0), cyl))
        self.assertFalse(cylinder_blocks((0.0, 2.0, 1.0), (1.5, 2.0, 1.0), cyl))
        # A vertical segment (L2 == 0) is never blocked.
        self.assertFalse(cylinder_blocks((2.0, 2.0, 0.5), (2.0, 2.0, 1.5), cyl))


class TestEpochClock(unittest.TestCase):
    def test_firing_rule(self):
        clk = EpochClock(5.0)
        times = [0.0, 0.1, 0.2, 0.25, 0.9, 0.95, 1.0, 1.0]
        fired = [clk.fires(t) for t in times]
        self.assertEqual(fired, [True, False, True, False, True, False, True, False])
        self.assertEqual(clk.e_next, 6)      # 0.4, 0.6 and 0.8 were passed over

    def test_nan_never_fires(self):
        clk = EpochClock(5.0)
        self.assertFalse(clk.fires(NAN))
        self.assertFalse(clk.fires(float("inf")))
        self.assertEqual(clk.e_next, 0)
        with self.assertRaises(LocateError):
            EpochClock(0.0)

    def test_skip_ahead_is_the_loop(self):
        """A long gap is skipped ahead (no t * rate iterations) and ends
        where the plain §6.1 loop ends."""
        for rate in (5.0, 10.0, 3.0):
            for t in (0.0, 0.2, 0.39999999999999997, 0.4, 1.0, 7.3, 1234.5678, 99999.99999):
                with self.subTest(rate=rate, t=t):
                    clk = EpochClock(rate)
                    clk.fires(0.0)
                    e = clk.e_next
                    fired = t >= float(e) / rate
                    while t >= float(e) / rate:
                        e += 1
                    self.assertEqual(clk.fires(t), fired)
                    self.assertEqual(clk.e_next, e)
        clk = EpochClock(5.0)
        self.assertTrue(clk.fires(1e12))                  # 5e12 iterations without the skip
        self.assertEqual(clk.e_next, 5 * 10 ** 12 + 1)
        self.assertFalse(clk.fires(1e12))

    def test_epoch_counter_range_is_refused(self):
        clk = EpochClock(5.0)
        assert_names(self, "t", clk.fires, 2.0 ** 53 / 5.0)
        assert_names(self, "t", clk.fires, 1e308)        # t * rate overflows to inf
        self.assertTrue(clk.fires(2.0 ** 53 / 5.0 * 0.5))


class TestRadio(unittest.TestCase):
    def test_streams_and_draw_order(self):
        c = cfg()
        radio = PhoneRadio(BEDROOM, c)
        for nid, *_ in BEDROOM:
            rt = Rng.stream(11, 300000 + nid)
            rs = Rng.stream(11, 310000 + nid)
            self.assertEqual(radio.offset_m[nid], c.rtt_offset_sigma_m * rt.gauss())
            self.assertEqual(radio.device_db[nid], c.rssi_device_sigma_db * rs.gauss())
        refs = {nid: (Rng.stream(11, 300000 + nid), Rng.stream(11, 310000 + nid)) for nid, *_ in BEDROOM}
        for nid in refs:
            refs[nid][0].gauss()
            refs[nid][1].gauss()
        for e, blocked in enumerate(([False] * 4, [True, False, True, False], [True] * 4)):
            reps = radio.epoch((1.0 + 0.1 * e, 2.0, 1.0), blocked)
            for rep in reps:
                rt, rs = refs[rep.node]
                self.assertEqual(rep.rtt_draws,
                                 (rt.uniform(), rt.gauss(), rt.uniform(), rt.uniform(), rt.uniform()))
                self.assertEqual(rep.rssi_draws, (rs.uniform(), rs.gauss(), rs.uniform()))

    def test_rtt_and_rssi_formulas(self):
        c = cfg(rtt_outlier_p=0.0, rtt_fail_p=0.0, rssi_fail_p=0.0)
        radio = PhoneRadio(BEDROOM, c)
        p = (1.3, 2.1, 1.0)
        reps = radio.epoch(p, [False, True, False, True])
        lam = 299792458.0 / CARRIER
        pl0 = 20.0 * math.log10(4.0 * math.pi / lam)
        for rep, node in zip(reps, BEDROOM):
            u_fail, g, u_out, u_mag, u_b = rep.rtt_draws
            d = dist3(p, node)
            self.assertAlmostEqual(rep.d, d, places=12)
            if rep.blocked:
                r = d + radio.offset_m[rep.node] + 1.5 * g - 1.0 * math.log(1.0 - u_b)
            else:
                r = d + radio.offset_m[rep.node] + g
            self.assertAlmostEqual(rep.rtt_m, max(r, 0.0), places=12)
            _, g_s, u_j = rep.rssi_draws
            self.assertEqual(rep.shadow_db, 4.0 * g_s)            # first epoch
            dd = max(d, 0.1)
            pl = pl0 + 20.0 * math.log10(dd)                     # d < breakpoint
            v = 10.0 - pl - (5.0 if rep.blocked else 0.0) + rep.shadow_db + radio.device_db[rep.node] \
                + (-1.0 + 2.0 * u_j)
            self.assertAlmostEqual(rep.rssi_v, v, places=10)
            self.assertEqual(rep.rssi_dbm, min(max(round(rep.rssi_v), -100), 0))
            self.assertIsInstance(rep.rssi_dbm, int)

    def test_shadowing_advances_on_failed_reports(self):
        ok = PhoneRadio(BEDROOM, cfg(rssi_fail_p=0.0, rtt_fail_p=0.0))
        failing = PhoneRadio(BEDROOM, cfg(rssi_fail_p=1.0, rtt_fail_p=1.0))
        path = [(1.0, 1.0, 1.0), (1.5, 1.2, 1.0), (1.5, 1.2, 1.0), (3.0, 2.5, 1.0)]
        prev = {}
        for e, p in enumerate(path):
            a = ok.epoch(p, [False] * 4)
            b = failing.epoch(p, [False] * 4)
            for ra, rb in zip(a, b):
                self.assertIsNone(rb.rssi_dbm)
                self.assertIsNone(rb.rtt_m)
                self.assertIsNotNone(ra.rssi_dbm)
                self.assertEqual(ra.shadow_db, rb.shadow_db)
                g_s = ra.rssi_draws[1]
                if e == 0:
                    expect = 4.0 * g_s
                else:
                    q = path[e - 1]
                    disp = math.sqrt((p[0] - q[0]) ** 2 + (p[1] - q[1]) ** 2)
                    rho = math.exp(-(disp / 6.0))
                    expect = rho * prev[ra.node] + math.sqrt(1.0 - rho * rho) * (4.0 * g_s)
                self.assertEqual(ra.shadow_db, expect)
                prev[ra.node] = ra.shadow_db

    def test_rssi_clamp_both_bounds(self):
        """§6.5's clamp(round(v), -100, 0) at both ends, set by tx_power_dbm
        (node 12 m away, beyond the breakpoint)."""
        node = [(0, 13.0, 1.0, 1.0)]
        kw = dict(rssi_fail_p=0.0, rssi_shadow_sigma_db=1e-9, rssi_device_sigma_db=0.0, rssi_jitter_db=0.0)
        low = PhoneRadio(node, LocateConfig(**kw).resolved(11, -80.0, CARRIER)).epoch((1.0, 1.0, 1.0), [False])[0]
        self.assertLess(low.rssi_v, -100.5)
        self.assertEqual(low.rssi_dbm, -100)
        high = PhoneRadio(node, LocateConfig(**kw).resolved(11, 80.0, CARRIER)).epoch((1.0, 1.0, 1.0), [False])[0]
        self.assertGreater(high.rssi_v, 0.5)
        self.assertEqual(high.rssi_dbm, 0)
        for rep in (low, high):
            self.assertIsInstance(rep.rssi_dbm, int)

    def test_non_finite_v_gives_no_rssi_report(self):
        """§6.5: no report when v is not finite (here 2.0*rssi_jitter_db
        overflows); the draws and S are kept as computed."""
        ok = PhoneRadio(BEDROOM, cfg(rssi_fail_p=0.0))
        bad = PhoneRadio(BEDROOM, cfg(rssi_fail_p=0.0, rssi_jitter_db=1e308))
        for p in ((1.0, 1.0, 1.0), (1.5, 1.2, 1.0), (1.5, 1.2, 1.0)):
            for ra, rb in zip(ok.epoch(p, [False] * 4), bad.epoch(p, [False] * 4)):
                self.assertFalse(math.isfinite(rb.rssi_v))
                self.assertIsNone(rb.rssi_dbm)
                self.assertIsNotNone(ra.rssi_dbm)
                self.assertEqual(rb.rssi_draws, ra.rssi_draws)
                self.assertEqual(rb.rtt_draws, ra.rtt_draws)
                self.assertEqual(rb.shadow_db, ra.shadow_db)
                self.assertEqual(rb.rtt_m, ra.rtt_m)

    def test_nodes_contract(self):
        """§6.1: nodes in id order (a sorted copy), distinct, at least one."""
        radio = PhoneRadio(list(reversed(BEDROOM)), cfg())
        self.assertEqual(radio.nodes, BEDROOM)
        self.assertEqual([r.node for r in radio.epoch((1.0, 1.0, 1.0), [False] * 4)], [0, 1, 2, 3])
        assert_names(self, "nodes", PhoneRadio, [], cfg())
        assert_names(self, "nodes", PhoneRadio, BEDROOM + [BEDROOM[2]], cfg())
        assert_names(self, "nodes", PhoneRadio, [(0, NAN, 0.0, 1.0)], cfg())

    def test_phone_and_distance_must_be_finite(self):
        radio = PhoneRadio(BEDROOM, cfg())
        assert_names(self, "phone", radio.epoch, (NAN, 1.0, 1.0), [False] * 4)
        # Finite coordinates whose distance overflows (§6.1).
        assert_names(self, "phone", radio.epoch, (1e160, 0.0, 1.0), [False] * 4)

    def test_label_must_be_sim(self):
        for label in ("REAL", "SIM->REAL", "SCENE", "sim", None):
            with self.subTest(label=label):
                with self.assertRaises(LocateError) as cm:
                    PhoneRadio(BEDROOM, cfg(), label=label)
                self.assertEqual(cm.exception.param, "label")
                self.assertNotIn(repr(label), str(cm.exception))       # never echoed
                with self.assertRaises(LocateError):
                    rtt_fix([], BEDROOM, cfg(), label=label)
                with self.assertRaises(LocateError):
                    rssi_fix([], BEDROOM, cfg(), label=label)
                with self.assertRaises(LocateError):
                    PhoneKalman(cfg(), label=label)
                with self.assertRaises(LocateError):
                    PhoneMethod("rtt", BEDROOM, cfg(), label=label)
                with self.assertRaises(LocateError):
                    run_phone([], BEDROOM, [], [], cfg(), label=label)

    def test_unresolved_config_is_refused(self):
        """Refused naming the first unset Room Sim value; `rssi_p0_dbm`
        unset is not refused (it is derived where it is used)."""
        assert_names(self, "phone_seed", PhoneRadio, BEDROOM, LocateConfig())
        assert_names(self, "carrier_hz", PhoneRun, BEDROOM, [], [],
                     LocateConfig(phone_seed=11, tx_power_dbm=10.0))
        assert_names(self, "tx_power_dbm", PhoneMethod, "rssi", BEDROOM, LocateConfig(phone_seed=11))

    def test_unset_rssi_p0_is_derived(self):
        """A plain-data config with `phone_seed`, `tx_power_dbm` and
        `carrier_hz` set and `rssi_p0_dbm` unset runs (as the Rust port
        does), with the records of the resolved config: `rssi_p0_dbm`
        unset is `default_rssi_p0_dbm(tx_power_dbm, carrier_hz)`."""
        plain = LocateConfig(phone_seed=11, tx_power_dbm=10.0, carrier_hz=CARRIER)
        self.assertIsNone(plain.rssi_p0_dbm)
        samples = TestPhoneRun().samples()
        walls = [(0.0, 0.0, 4.2, 0.0, 2.7)]
        recs = run_phone(samples, BEDROOM, walls, [], plain)
        self.assertEqual(recs, run_phone(samples, BEDROOM, walls, [], cfg()))
        self.assertTrue(any("reason" not in r["rssi"] for r in recs), recs)
        p = (1.3, 2.1, 1.0)
        reps = [(n[0], cfg().rssi_p0_dbm - 20.0 * math.log10(dist3(p, n))) for n in BEDROOM]
        self.assertEqual(rssi_fix(reps, BEDROOM, plain), rssi_fix(reps, BEDROOM, cfg()))
        # Nothing to derive it from: refused naming it.
        assert_names(self, "rssi_p0_dbm", rssi_fix, reps, BEDROOM, LocateConfig())
        assert_names(self, "rssi_p0_dbm", rssi_fix, reps, BEDROOM, LocateConfig(tx_power_dbm=10.0))


class TestRttFix(unittest.TestCase):
    def test_noiseless_bedroom_recovers_the_truth(self):
        for p in ((1.3, 2.1, 1.0), (3.0, 0.9, 0.55), (0.7, 3.0, 1.6)):
            with self.subTest(p=p):
                c = cfg(phone_assumed_z_m=p[2])
                fix = rtt_fix([(n[0], dist3(p, n)) for n in BEDROOM], BEDROOM, c)
                self.assertTrue(fix.ok, fix)
                self.assertLess(math.sqrt((fix.x - p[0]) ** 2 + (fix.y - p[1]) ** 2), 1e-6)
                self.assertLess(fix.chi2, 1e-9)
                self.assertGreater(radius95(fix.r00, fix.r01, fix.r11), 0.0)
                self.assertEqual(fix.r00, 1.0 * fix.inv00)
                self.assertEqual(fix.r01, 1.0 * fix.inv01)

    def test_office_collinear_nodes_are_degenerate_everywhere(self):
        """§12a: the office's collinear nodes 0, 4, 1 (and 3, 5, 2) alone
        give `degenerate` for every phone position on a 0.1 m grid over
        the 6 x 5 m room, for both fixes (§7's collinear rule)."""
        c = cfg(phone_assumed_z_m=1.5)
        n_fix = 0
        for ids in ((0, 4, 1), (3, 5, 2)):
            line = [n for n in OFFICE if n[0] in ids]
            for i in range(61):
                for j in range(51):
                    p = (0.1 * i, 0.1 * j, 1.0)
                    rtt = rtt_fix([(n[0], dist3(p, n)) for n in line], OFFICE, c)
                    rssi = rssi_fix([(n[0], round(c.rssi_p0_dbm - 20.0 * math.log10(max(dist3(p, n), 0.1))))
                                     for n in line], OFFICE, c)
                    if rtt.reason != "degenerate" or rssi.reason != "degenerate":
                        self.fail(f"nodes {ids}, phone {p}: rtt {rtt.reason}, rssi {rssi.reason}")
                    n_fix += 2
        self.assertEqual(n_fix, 2 * 2 * 61 * 51)
        # A fourth, off-line node makes the same reports solvable.
        p = (2.9, 4.5, 1.5)
        four = [n for n in OFFICE if n[0] in (0, 4, 1, 5)]
        self.assertTrue(rtt_fix([(n[0], dist3(p, n)) for n in four], OFFICE, c).ok)

    def test_collinear_rule_does_not_depend_on_the_line_direction(self):
        """§7's collinear rule `not (det > 1e-9 * ((n00 + n11)*(n00 + n11)))`
        on office-like nodes collinear to rounding on an axis-aligned line
        (`0.1 + 0.2` is 0.30000000000000004): noiseless reports from the
        truth (2.9, 4.5, 1.0) give `degenerate` for both fixes and both
        methods. The old test `det > 1e-9 * (n00*n11)` passed them (n11 is
        itself rounding-sized) and published a mirror image 8.4 m off."""
        nodes = [(0, 0.2, 0.1 + 0.2, 1.5), (4, 3.0, 0.1 + 0.2, 1.5), (1, 5.8, 0.3, 1.5)]
        c = cfg(phone_assumed_z_m=1.0)
        p = (2.9, 4.5, 1.0)
        rtt = [(n[0], dist3(p, n)) for n in nodes]
        rssi = [(n[0], c.rssi_p0_dbm - 20.0 * math.log10(dist3(p, n))) for n in nodes]
        # The case the old rule let through (plain sums: 2 terms each).
        a = [(2.0 * (n[1] - nodes[0][1]), 2.0 * (n[2] - nodes[0][2])) for n in nodes[1:]]
        n00 = a[0][0] * a[0][0] + a[1][0] * a[1][0]
        n01 = a[0][0] * a[0][1] + a[1][0] * a[1][1]
        n11 = a[0][1] * a[0][1] + a[1][1] * a[1][1]
        det = n00 * n11 - n01 * n01
        self.assertTrue(det > 1e-9 * (n00 * n11))
        self.assertFalse(det > 1e-9 * ((n00 + n11) * (n00 + n11)))
        self.assertEqual(rtt_fix(rtt, nodes, c).reason, "degenerate")
        self.assertEqual(rssi_fix(rssi, nodes, c).reason, "degenerate")
        self.assertEqual(rssi_fix([(nid, round(v)) for nid, v in rssi], nodes, c).reason, "degenerate")
        self.assertEqual(PhoneMethod("rtt", nodes, c).step(0.0, rtt), {"reason": "degenerate"})
        self.assertEqual(PhoneMethod("rssi", nodes, c).step(0.0, rssi), {"reason": "degenerate"})
        # Bit-equal y: degenerate under both rules.
        flat = [(0, 0.2, 0.3, 1.5), (4, 3.0, 0.3, 1.5), (1, 5.8, 0.3, 1.5)]
        self.assertEqual(rtt_fix([(n[0], dist3(p, n)) for n in flat], flat, c).reason, "degenerate")
        # Anchors on one line at any angle, collinear to rounding.
        for k in range(24):
            th = math.radians(7.5 * k)
            ux, uy = math.cos(th), math.sin(th)
            line = [(i, 0.5 + s * ux, 0.7 + s * uy, 1.5) for i, s in enumerate((0.0, 2.3, 5.9))]
            q = (0.5 - 2.0 * uy + 1.0 * ux, 0.7 + 2.0 * ux + 1.0 * uy, 1.0)
            with self.subTest(angle_deg=7.5 * k):
                self.assertEqual(rtt_fix([(n[0], dist3(q, n)) for n in line], line, c).reason, "degenerate")
                self.assertEqual(rssi_fix([(n[0], c.rssi_p0_dbm - 20.0 * math.log10(dist3(q, n)))
                                           for n in line], line, c).reason, "degenerate")

    def test_hall_triples(self):
        """§7: the hall's collinear triples (nodes 0, 5, 6 on x = 0.3 and
        3, 4, 6 on x + y = 7) are `degenerate` at every phone position on a
        0.5 m grid, for both fixes; the thin but valid triangle 1, 3, 4
        (node 3 about 0.21 m off the 5.8 m line from 1 to 4) gives a fix at
        every one."""
        c = cfg(phone_assumed_z_m=1.0)
        grid = [(0.5 * i, 0.5 * j, 1.0) for i in range(19) for j in range(15)]
        for ids, want in (((0, 5, 6), "degenerate"), ((3, 4, 6), "degenerate"), ((1, 3, 4), None)):
            tri = [n for n in HALL if n[0] in ids]
            for p in grid:
                rtt = rtt_fix([(n[0], dist3(p, n)) for n in tri], HALL, c)
                rssi = rssi_fix([(n[0], c.rssi_p0_dbm - 20.0 * math.log10(max(dist3(p, n), 0.1)))
                                 for n in tri], HALL, c)
                if rtt.reason != want or rssi.reason != want:
                    self.fail(f"nodes {ids}, phone {p}: rtt {rtt.reason}, rssi {rssi.reason}")

    def test_one_sided_huber_beats_least_squares_on_a_long_range(self):
        p = (1.3, 2.1, 1.0)
        reps = [(n[0], dist3(p, n) + (5.0 if n[0] == 2 else 0.0)) for n in BEDROOM]
        huber = rtt_fix(reps, BEDROOM, cfg())
        plain = rtt_fix(reps, BEDROOM, cfg(huber_k=1e12))
        eh = math.sqrt((huber.x - p[0]) ** 2 + (huber.y - p[1]) ** 2)
        ep = math.sqrt((plain.x - p[0]) ** 2 + (plain.y - p[1]) ** 2)
        print(f"\n  SIM: one range +5 m: one-sided Huber error {eh:.3f} m, least squares {ep:.3f} m")
        self.assertLess(eh, ep)

    def test_reasons(self):
        c = cfg()
        self.assertEqual(rtt_fix([], BEDROOM, c).reason, "no_reports")
        self.assertEqual(rtt_fix([(0, 1.0), (1, 2.0)], BEDROOM, c).reason, "too_few_ranges")
        # Inconsistent ranges: Gauss-Newton zig-zags slowly. This set needs
        # 97 iterations (over the old limit of 50, within 200) ...
        slow = rtt_fix([(0, 0.0), (1, 2.0), (2, 1.0), (3, 2.0)], BEDROOM, c)
        self.assertTrue(slow.ok, slow)
        self.assertEqual(slow.iterations, 97)
        # ... and this one 288 (§7.1's limit is 200).
        self.assertEqual(rtt_fix([(0, 0.0), (1, 0.0), (2, 0.5), (3, 5.0)], BEDROOM, c).reason, "not_converged")

    def test_reports_contract(self):
        """§7: reports name distinct known nodes with finite values, else
        the fix is refused naming `reports` (both fixes)."""
        c = cfg()
        for fix in (rtt_fix, rssi_fix):
            for reps in ([(9, -1.0), (1, -2.0), (2, -3.0)],                  # not a node
                         [(0, -1.0), (1, -2.0), (0, -3.0)],                  # node 0 twice
                         [(0, -1.0), (1, NAN), (2, -3.0)],
                         [(0, -1.0), (1, float("inf")), (2, -3.0)],
                         [(0, float("-inf"))]):
                with self.subTest(fix=fix.__name__, reports=reps):
                    assert_names(self, "reports", fix, reps, BEDROOM, c)

    def test_report_order_does_not_matter(self):
        p = (1.3, 2.1, 1.0)
        reps = [(n[0], dist3(p, n) + 0.1 * n[0]) for n in BEDROOM]
        a = rtt_fix(reps, BEDROOM, cfg())
        b = rtt_fix(list(reversed(reps)), list(reversed(BEDROOM)), cfg())
        self.assertEqual(a, b)


class TestRssiFix(unittest.TestCase):
    def test_noiseless_model_recovers_the_truth(self):
        c = cfg()
        for p in ((1.3, 2.1), (3.2, 0.8), (2.0, 3.0)):
            with self.subTest(p=p):
                reps = []
                for nid, nx, ny, nz in BEDROOM:
                    rho = math.sqrt((p[0] - nx) ** 2 + (p[1] - ny) ** 2 + (c.phone_assumed_z_m - nz) ** 2)
                    reps.append((nid, c.rssi_p0_dbm - 10.0 * c.rssi_est_exponent * math.log10(rho)))
                fix = rssi_fix(reps, BEDROOM, c)
                self.assertTrue(fix.ok, fix)
                self.assertLess(math.sqrt((fix.x - p[0]) ** 2 + (fix.y - p[1]) ** 2), 1e-6)
                self.assertIsNone(fix.chi2)
                self.assertEqual(fix.r00, 16.0 * fix.inv00)

    def test_reasons(self):
        c = cfg()
        self.assertEqual(rssi_fix([], BEDROOM, c).reason, "no_reports")
        self.assertEqual(rssi_fix([(0, -40), (1, -50)], BEDROOM, c).reason, "too_few_ranges")
        line = [(0, -40), (4, -45), (1, -50)]
        self.assertEqual(rssi_fix(line, OFFICE, cfg(phone_assumed_z_m=1.5)).reason, "degenerate")
        # 645 iterations to converge: over §7.1's 200.
        self.assertEqual(rssi_fix([(0, -30), (1, -30), (2, -50), (3, -60)], BEDROOM, c).reason, "not_converged")

    def test_start_weight_guard(self):
        """§7.2: a weight sum that is not finite and > 0 is `degenerate`,
        decided by the comparison, never by an exception."""
        c = cfg()
        self.assertEqual(rssi_fix([(0, -4000.0), (1, -4000.0), (2, -4000.0)], BEDROOM, c).reason, "degenerate")
        self.assertEqual(rssi_fix([(0, -40.0), (1, 4000.0), (2, -50.0)], BEDROOM, c).reason, "degenerate")
        self.assertEqual(rssi_fix([(0, 1e300), (1, 1e300), (2, 1e300)], BEDROOM, c).reason, "degenerate")


def good_fix(x, y, var=0.25):
    return Fix(None, x, y, var, 0.0, var, var, 0.0, var, None, 3)


class TestKalman(unittest.TestCase):
    def test_start_is_ungated(self):
        kf = PhoneKalman(cfg())
        st = kf.step(1.0, good_fix(2.0, 1.0))
        self.assertTrue(st.started)
        self.assertEqual((st.x, st.y, st.p00, st.p01, st.p11), (2.0, 1.0, 0.25, 0.0, 0.25))
        self.assertEqual(kf.s, [2.0, 1.0, 0.0, 0.0])
        self.assertEqual(kf.P[2][2], 1.0)
        self.assertEqual((kf.t_f, kf.t_acc), (1.0, 1.0))

    def test_ten_sigma_fix_is_gated_and_the_prediction_kept(self):
        kf = PhoneKalman(cfg())
        kf.step(1.0, good_fix(2.0, 1.0))
        # 10 sigma of R (sigma_R = 0.5 m): 5 m off; d2 = 25 / S00 = 46.1 > 9.21.
        st = kf.step(1.2, good_fix(2.0 + 10.0 * 0.5, 1.0))
        self.assertEqual(st.reason, "gated")
        self.assertIsNone(st.x)
        # The predicted state and covariance are kept; t_acc is not moved.
        dt = 1.2 - 1.0
        self.assertEqual(kf.s, [2.0, 1.0, 0.0, 0.0])
        self.assertAlmostEqual(kf.P[0][0], 0.25 + dt * dt * 1.0 + 1.0 * dt ** 3 / 3.0, places=12)
        self.assertAlmostEqual(kf.P[0][2], dt * 1.0 + 1.0 * dt * dt / 2.0, places=12)
        self.assertEqual((kf.t_f, kf.t_acc), (1.2, 1.0))

    def test_gate_threshold(self):
        kf = PhoneKalman(cfg(phone_accel_psd=1e-9, phone_init_speed_std=1e-9))
        kf.step(0.0, good_fix(0.0, 0.0, 1.0))
        # S ~ 2 I: d2 = nu^2 / 2; just inside and just outside chi2(2, 0.99).
        inside = math.sqrt(2.0 * CHI2_2_99) * 0.999
        self.assertIsNone(kf.step(1e-6, good_fix(inside, 0.0, 1.0)).reason)
        kf = PhoneKalman(cfg(phone_accel_psd=1e-9, phone_init_speed_std=1e-9))
        kf.step(0.0, good_fix(0.0, 0.0, 1.0))
        outside = math.sqrt(2.0 * CHI2_2_99) * 1.001
        self.assertEqual(kf.step(1e-6, good_fix(outside, 0.0, 1.0)).reason, "gated")

    def test_coast_reset_at_exactly_max_coast(self):
        kf = PhoneKalman(cfg())
        kf.step(1.0, good_fix(2.0, 1.0))
        st = kf.step(4.0, good_fix(2.1, 1.0))              # t - t_acc == 3.0: kept
        self.assertFalse(st.discarded)
        self.assertFalse(st.started)
        self.assertIsNone(st.reason)
        self.assertEqual(kf.t_acc, 4.0)
        st = kf.step(7.25, good_fix(2.0, 1.0))             # 3.25 > 3.0: discarded, restarted
        self.assertTrue(st.discarded)
        self.assertTrue(st.started)
        self.assertEqual((st.x, st.y), (2.0, 1.0))

    def test_discard_happens_on_an_epoch_without_a_fix(self):
        kf = PhoneKalman(cfg())
        kf.step(1.0, good_fix(2.0, 1.0))
        st = kf.step(5.0, Fix("no_reports"))
        self.assertEqual(st.reason, "no_reports")
        self.assertTrue(st.discarded)
        self.assertFalse(kf.exists)

    def test_no_predict_without_a_fix(self):
        kf = PhoneKalman(cfg())
        kf.step(1.0, good_fix(2.0, 1.0))
        kf.step(1.2, good_fix(2.1, 1.1))
        before = ([list(r) for r in kf.P], list(kf.s), kf.t_f, kf.t_acc)
        for reason in ("no_reports", "too_few_ranges", "degenerate", "not_converged"):
            st = kf.step(1.4, Fix(reason))
            self.assertEqual(st.reason, reason)
            self.assertEqual(([list(r) for r in kf.P], list(kf.s), kf.t_f, kf.t_acc), before)
        # The next predict runs from t_f = 1.2, not from 1.4.
        kf.step(1.6, good_fix(2.1, 1.1))
        self.assertEqual(kf.t_f, 1.6)

    def test_rejected_streak_then_coast_expiry(self):
        kf = PhoneKalman(cfg())
        kf.step(0.0, good_fix(1.0, 1.0))
        for t in (1.0, 2.0, 3.0):
            self.assertEqual(kf.step(t, good_fix(40.0, 1.0)).reason, "gated")
        st = kf.step(3.5, good_fix(40.0, 1.0))
        self.assertTrue(st.discarded and st.started)
        self.assertEqual(st.x, 40.0)


class TestPhoneMethod(unittest.TestCase):
    def test_published_covariance_adds_b(self):
        c = cfg(phone_assumed_z_m=1.0)
        m = PhoneMethod("rtt", BEDROOM, c)
        p = (1.3, 2.1, 1.0)
        rec = m.step(0.0, [(n[0], dist3(p, n)) for n in BEDROOM])
        fix = m.last_fix
        bb = 0.5 * 0.5
        self.assertEqual(rec["cov_xx"], fix.r00 + bb * fix.inv00)
        self.assertEqual(rec["cov_xy"], fix.r01 + bb * fix.inv01)
        self.assertEqual(rec["cov_yy"], fix.r11 + bb * fix.inv11)
        self.assertEqual(rec["radius_m"], radius95(rec["cov_xx"], rec["cov_xy"], rec["cov_yy"]))
        self.assertIs(rec["radius_calibrated"], False)
        self.assertIsNone(rec["confidence"])
        # B never enters the filter.
        self.assertEqual(m.kf.P[0][0], fix.r00)

    def test_implausible_radius_still_starts_the_filter(self):
        m = PhoneMethod("rtt", BEDROOM, cfg(min_plausible_radius_m=100.0))
        p = (1.3, 2.1, 1.0)
        rec = m.step(0.0, [(n[0], dist3(p, n)) for n in BEDROOM])
        self.assertEqual(rec, {"reason": "implausible_radius"})
        self.assertTrue(m.kf.exists)

    def test_reasons_in_order(self):
        m = PhoneMethod("rssi", BEDROOM, cfg())
        self.assertEqual(m.step(0.0, [(0, None), (1, None)]), {"reason": "no_reports"})
        self.assertEqual(m.step(0.2, [(0, -40), (1, -50), (2, None)]), {"reason": "too_few_ranges"})
        m = PhoneMethod("rtt", BEDROOM, cfg())
        self.assertEqual(m.step(0.0, [(0, 0.0), (1, 0.0), (2, 0.5), (3, 5.0)]), {"reason": "not_converged"})

    def test_gated_epoch_publishes_nothing(self):
        """§12a: a fix far off the prediction is `gated`: the record is the
        abstention alone."""
        m = PhoneMethod("rtt", BEDROOM, cfg())
        p, q = (0.8, 0.7, 1.0), (3.6, 3.0, 1.0)
        self.assertNotIn("reason", m.step(0.0, [(n[0], dist3(p, n)) for n in BEDROOM]))
        self.assertEqual(m.step(0.2, [(n[0], dist3(q, n)) for n in BEDROOM]), {"reason": "gated"})

    def test_radius_scale_wiring(self):
        """Only `phone_radius_scale` scales a phone radius (the other two
        scales are set to other values and must not enter)."""
        p = (1.3, 2.1, 1.0)
        reports = {"rtt": [(n[0], dist3(p, n)) for n in BEDROOM]}
        c = cfg()
        reports["rssi"] = [(n[0], c.rssi_p0_dbm - 20.0 * math.log10(dist3(p, n))) for n in BEDROOM]
        for method in ("rtt", "rssi"):
            with self.subTest(method=method):
                base = PhoneMethod(method, BEDROOM, cfg()).step(0.0, reports[method])
                for kw, scale in (({"phone_radius_scale": 0.75}, 0.75),
                                  ({"wlm_radius_scale": 1.5, "rti_radius_scale": 2.0}, 1.0),
                                  ({"phone_radius_scale": 1.25, "wlm_radius_scale": 0.5,
                                    "rti_radius_scale": 3.0}, 1.25)):
                    rec = PhoneMethod(method, BEDROOM, cfg(**kw)).step(0.0, reports[method])
                    self.assertEqual(rec["radius_m"], scale * radius95(rec["cov_xx"], rec["cov_xy"], rec["cov_yy"]))
                    self.assertEqual((rec["x"], rec["y"], rec["cov_xx"]), (base["x"], base["y"], base["cov_xx"]))
                    self.assertEqual(rec["radius_m"], scale * base["radius_m"])


class TestPhoneRun(unittest.TestCase):
    def samples(self, n=40, dt=0.0125):
        out = []
        for j in range(n):
            t = float(j) * dt
            p = (1.0 + 0.5 * t, 1.5 + 0.2 * t, 1.0)
            out.append((t, p, [(p[0] - 0.05, p[1] + 0.25, 0.18, 1.7)]))
        return out

    def test_records(self):
        c = cfg()
        walls = [(0.0, 0.0, 4.2, 0.0, 2.7), (4.2, 0.0, 4.2, 3.6, 2.7), (4.2, 3.6, 0.0, 3.6, 2.7),
                 (0.0, 3.6, 0.0, 0.0, 2.7)]
        recs = run_phone(self.samples(), BEDROOM, walls, [], c)
        self.assertEqual([r["t"] for r in recs], [0.0, 0.2, 0.4])
        for r in recs:
            self.assertEqual(list(r), ["label", "t", "rtt", "rssi", "reports", "truth"])
            self.assertEqual(r["label"], "SIM")
            self.assertEqual([e["node"] for e in r["reports"]], [0, 1, 2, 3])
            for e in r["reports"]:
                self.assertEqual(list(e), ["node", "rtt_m", "rssi_dbm", "blocked"])
            self.assertEqual(list(r["truth"]), ["x", "y", "z"])
            for k in ("rtt", "rssi"):
                est = r[k]
                if "reason" not in est:
                    self.assertTrue(math.isfinite(est["radius_m"]))
                    self.assertGreaterEqual(est["radius_m"], c.min_plausible_radius_m)
        # Deterministic.
        self.assertEqual(recs, run_phone(self.samples(), BEDROOM, walls, [], c))

    def test_blocked_flag_follows_the_geometry(self):
        c = cfg()
        run = PhoneRun(BEDROOM, [], [], c)
        p = (2.0, 1.8, 1.0)
        rec = run.sample(0.0, p, [(1.0, 1.0, 0.3, 1.8)])    # between the phone and node 0
        self.assertEqual([e["blocked"] for e in rec["reports"]], [True, False, False, False])

    def test_truth_sample_contract(self):
        """§6.1: each refusal names its input."""
        c = cfg()
        walls = [(0.0, 0.0, 4.2, 0.0, 2.7)]
        box = (1.0, 2.0, 0.375, 1.2, 0.6, 0.75, 1.0, 0.0)
        assert_names(self, "walls", PhoneRun, BEDROOM, [(0.0, 0.0, NAN, 0.0, 2.7)], [], c)
        assert_names(self, "walls", PhoneRun, BEDROOM, [(0.0, 0.0, 4.2, 0.0)], [], c)
        assert_names(self, "objects", PhoneRun, BEDROOM, walls, [box[:7] + (float("inf"),)], c)
        assert_names(self, "nodes", PhoneRun, [], walls, [box], c)
        assert_names(self, "nodes", PhoneRun, BEDROOM + BEDROOM[:1], walls, [box], c)
        cyl = [(2.0, 2.0, 0.18, 1.7)]
        p = (1.0, 1.0, 1.0)
        for t, ph, cy, param in ((NAN, p, cyl, "t"), (float("inf"), p, cyl, "t"),
                                 (2.0 ** 53 / 5.0, p, cyl, "t"), (0.5, (1.0, NAN, 1.0), cyl, "phone"),
                                 (0.5, (1e160, 1.0, 1.0), cyl, "phone"),
                                 (0.5, p, [(2.0, NAN, 0.18, 1.7)], "cylinders")):
            with self.subTest(t=t, phone=ph, cylinders=cy):
                run = PhoneRun(BEDROOM, walls, [box], c)
                assert_names(self, param, run.sample, t, ph, cy)
        run = PhoneRun(BEDROOM, walls, [box], c)
        self.assertIsNotNone(run.sample(0.0, p, cyl))
        self.assertIsNone(run.sample(0.0, p, cyl))           # t_j >= t_{j-1}: equal is fine
        self.assertIsNotNone(run.sample(0.25, p, cyl))
        assert_names(self, "t", run.sample, 0.2, p, cyl)      # decreasing
        # §6.1: a node distance that overflows is refused at every sample,
        # firing or not, before any state changes (a refused sample changes
        # nothing: the next good sample fires as if it had not been fed).
        run = PhoneRun(BEDROOM, walls, [box], c)
        self.assertIsNotNone(run.sample(0.0, p, cyl))
        self.assertEqual((run.clock.e_next, run.last_t), (1, 0.0))
        far = (1e160, 1.0, 1.0)
        for t in (0.05, 0.3):             # 0.05 would not fire; 0.3 would
            with self.subTest(t=t):
                assert_names(self, "phone", run.sample, t, far, cyl)
                self.assertEqual((run.clock.e_next, run.last_t), (1, 0.0))
                self.assertEqual(run.radio._prev_xy, (p[0], p[1]))
        rec = run.sample(0.31, p, cyl)
        self.assertIsNotNone(rec)
        self.assertEqual((rec["t"], run.clock.e_next, run.last_t), (0.31, 2, 0.31))
        self.assertIsNotNone(run.sample(0.4, p, cyl))
        # The same refusals on a run fed the samples one by one are its
        # records with the refused samples left out.
        good = [(0.0, p, cyl), (0.31, p, cyl), (0.4, p, cyl)]
        self.assertEqual([r["t"] for r in run_phone(good, BEDROOM, walls, [box], c)], [0.0, 0.31, 0.4])
        # A first sample far in the run skips ahead (no t * rate loop).
        run = PhoneRun(list(reversed(BEDROOM)), walls, [box], c)
        rec = run.sample(1e12, p, cyl)
        self.assertEqual((rec["t"], [e["node"] for e in rec["reports"]]), (1e12, [0, 1, 2, 3]))
        self.assertEqual(run.clock.e_next, 5 * 10 ** 12 + 1)

    def test_no_scene_strings(self):
        recs = run_phone(self.samples(), BEDROOM, [], [], cfg())
        text = repr(recs)
        for word in ("phone", "sleeper", "id"):
            self.assertNotIn(f"'{word}'", text)


if __name__ == "__main__":
    unittest.main()
