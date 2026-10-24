"""spec/locate.md §3, §5, §8, §9: RssiSignal, the tracker's covariance
fields, the radius, the weighted-link-midpoint baseline and the config.

Expected values are built here from the spec's formulas, not from the
module's helpers, so a transposed index cannot agree with itself.
Everything is SIM.
"""

from __future__ import annotations

import math
import unittest

from wifisense.localize.baseline import default_wlm_floor, weighted_link_midpoint
from wifisense.localize.config import LocateConfig, LocateError
from wifisense.localize.geometry import Geometry, Node, Room
from wifisense.localize.measure import LinkMeasurement, RssiSignal
from wifisense.localize.radius import (
    CHI2_2_99, K95, LN10, abstention, estimate_record, publishable, publishable_estimate, radius95,
)
from wifisense.localize.track import PositionEstimate, Tracker
from wifisense.types import CsiFrame

NAN = float("nan")
INF = float("inf")


def square_geometry(side: float = 4.0, lambda_m: float = 0.3) -> Geometry:
    room = Room(side, side)
    nodes = [Node(0, 0.0, 0.0), Node(1, side, 0.0), Node(2, side, side), Node(3, 0.0, side)]
    return Geometry(room, nodes, lambda_m=lambda_m)


def frame(rssi: int, amps=None) -> CsiFrame:
    amps = [3.0, 4.0, 5.0] if amps is None else amps
    return CsiFrame(list(amps), [0.0] * len(amps), 0, 20.0, rssi, 1)


class TestRssiSignal(unittest.TestCase):
    def test_name_and_value(self):
        sig = RssiSignal()
        self.assertEqual(sig.name, "rssi_dbm")
        st = sig.new_state(20.0)
        v = st.push(frame(-47))
        self.assertIsInstance(v, float)
        self.assertEqual(v, -47.0)

    def test_shared_validity_gate_drops_bad_frames(self):
        st = RssiSignal().new_state(20.0)
        self.assertIsNone(st.push(frame(-40, [])))                 # empty
        self.assertIsNone(st.push(frame(-40, [1.0, NAN])))         # non-finite amplitude
        self.assertIsNone(st.push(frame(-40, [0.0, 0.0])))         # zero power
        self.assertIsNone(st.push(frame(-40, [1e200] * 4)))        # power overflows
        self.assertEqual(st.push(frame(-40, [0.0, 1.0])), -40.0)   # one zero bin is fine

    def test_measurement_variance_of_rssi(self):
        geom = square_geometry()
        meas = LinkMeasurement(geom, window_s=1.0, sample_rate_hz=10.0, signal=RssiSignal())
        vals = [-40, -42, -40, -42, -40, -42]
        for i, v in enumerate(vals):
            meas.update(0.1 * i, 0, 1, frame(v))
        y = meas.vector(0.5)
        mean = sum(vals) / len(vals)
        expect = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
        self.assertAlmostEqual(y[geom.link_index(0, 1)], expect, places=12)


class TestPositionEstimateCovariance(unittest.TestCase):
    def test_default_is_zero(self):
        est = PositionEstimate(1.0, 2.0, 0.5, 0.9)
        self.assertEqual((est.cov_xx, est.cov_xy, est.cov_yy), (0.0, 0.0, 0.0))
        self.assertEqual(radius95(est.cov_xx, est.cov_xy, est.cov_yy), 0.0)

    def test_update_sets_the_matrix_sigma_is_read_from(self):
        geom = Geometry.perimeter(8)
        tracker = Tracker(geom)
        px = [(i, p) for i, p in enumerate(geom.grid.pixels)]

        def blob(cx, cy):
            return [math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / 0.5) for _, (x, y) in px]

        for k, t in enumerate((0.0, 0.4, 0.8, 1.2)):
            est = tracker.update(blob(3.0 + 0.2 * k, 3.0), t)
            self.assertIsNotNone(est)
            cov = tracker._cov
            self.assertEqual(est.cov_xx, cov[0][0])
            self.assertEqual(est.cov_xy, cov[0][1])
            self.assertEqual(est.cov_yy, cov[1][1])
            self.assertEqual(est.sigma_m, math.sqrt(max(est.cov_xx + est.cov_yy, 0.0)))
            self.assertGreater(radius95(est.cov_xx, est.cov_xy, est.cov_yy), 0.0)


class TestRadius(unittest.TestCase):
    def test_constants_are_computed(self):
        self.assertEqual(K95, math.sqrt(-2.0 * math.log(0.05)))
        self.assertEqual(K95, 2.4477468306808166)
        self.assertEqual(CHI2_2_99, 9.210340371976182)
        self.assertEqual(LN10, math.log(10.0))

    def test_known_matrices(self):
        self.assertEqual(radius95(1.0, 0.0, 1.0), K95)
        self.assertEqual(radius95(4.0, 0.0, 1.0), 2.0 * K95)
        self.assertEqual(radius95(1.0, 0.0, 4.0), 2.0 * K95)
        self.assertAlmostEqual(radius95(2.0, 1.0, 2.0), K95 * math.sqrt(3.0), places=14)

    def test_zero_matrix_is_not_publishable(self):
        r = radius95(0.0, 0.0, 0.0)
        self.assertEqual(r, 0.0)
        self.assertFalse(publishable(r, 0.3))

    def test_nan_and_negative_give_zero(self):
        self.assertEqual(radius95(NAN, 0.0, 1.0), 0.0)
        self.assertEqual(radius95(INF, 0.0, -INF), 0.0)       # inf - inf
        self.assertEqual(radius95(-1.0, 0.0, -2.0), 0.0)

    def test_publication_test_fails_closed(self):
        self.assertFalse(publishable(NAN, 0.3))
        self.assertFalse(publishable(INF, 0.3))
        self.assertFalse(publishable(0.29, 0.3))
        self.assertTrue(publishable(0.3, 0.3))
        self.assertTrue(publishable(0.31, 0.3))

    def test_zero_radius_is_never_published(self):
        """§8: every publication test also needs `r > 0`, so a caller that
        passes a zero (or negative) minimum still never publishes a bare
        point (ADR-008 d5)."""
        for minr in (0.0, -0.0, -10.0, NAN):
            with self.subTest(minr=minr):
                self.assertFalse(publishable(0.0, minr))
                self.assertFalse(publishable(-0.0, minr))
                self.assertFalse(publishable(-2.5, minr))
                self.assertFalse(publishable_estimate(1.0, 2.0, 0.0, minr))
        self.assertTrue(publishable(5e-324, 0.0))
        self.assertTrue(publishable_estimate(1.0, 2.0, 5e-324, -10.0))

    def test_publication_needs_a_finite_position(self):
        """§8: x, y and r finite, r > 0 and r >= min_plausible_radius_m."""
        self.assertTrue(publishable_estimate(1.0, -2.0, 0.3, 0.3))
        for x, y, r in ((NAN, 0.0, 1.0), (0.0, NAN, 1.0), (INF, 0.0, 1.0), (0.0, -INF, 1.0),
                        (0.0, 0.0, NAN), (0.0, 0.0, INF), (0.0, 0.0, 0.29)):
            with self.subTest(x=x, y=y, r=r):
                self.assertFalse(publishable_estimate(x, y, r, 0.3))
        self.assertFalse(publishable_estimate(NAN, NAN, 0.0, 0.0))

    def test_records(self):
        rec = estimate_record(1.0, 2.0, 0.5, 0.1, 0.0, 0.2)
        self.assertEqual(
            list(rec),
            ["x", "y", "radius_m", "cov_xx", "cov_xy", "cov_yy", "confidence", "radius_calibrated"],
        )
        self.assertIsNone(rec["confidence"])
        self.assertIs(rec["radius_calibrated"], False)
        self.assertEqual(abstention("gated"), {"reason": "gated"})
        with self.assertRaises(ValueError):
            abstention("because")


class TestWeightedLinkMidpoint(unittest.TestCase):
    def setUp(self):
        self.geom = square_geometry()
        self.L = self.geom.n_links
        self.sig = [0.1] * self.L

    def expected_cov(self, link):
        lam = self.geom.lambda_m
        d = math.sqrt((link.rx[0] - link.tx[0]) ** 2 + (link.rx[1] - link.tx[1]) ** 2)
        ux = (link.rx[0] - link.tx[0]) / d
        uy = (link.rx[1] - link.tx[1]) / d
        a = d * d / 12.0
        b = 0.5 * math.sqrt((d + lam) ** 2 - d * d)
        c = b * b / 3.0
        # a·uuᵀ + c·vvᵀ with v = (-uy, ux)
        return (a * ux * ux + c * uy * uy, a * ux * uy - c * uy * ux, a * uy * uy + c * ux * ux)

    def test_one_link_gives_its_midpoint_and_ellipse_covariance(self):
        for tx, rx in ((0, 1), (0, 2), (3, 1), (2, 3)):
            with self.subTest(link=(tx, rx)):
                y = [0.0] * self.L
                li = self.geom.link_index(tx, rx)
                y[li] = 2.0
                y[(li + 1) % self.L] = -0.7          # negative y weighs nothing
                rec = weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm")
                self.assertNotIn("reason", rec)
                link = self.geom.links[li]
                self.assertEqual((rec["x"], rec["y"]), link.midpoint)
                exx, exy, eyy = self.expected_cov(link)
                self.assertAlmostEqual(rec["cov_xx"], exx, places=12)
                self.assertAlmostEqual(rec["cov_xy"], exy, places=12)
                self.assertAlmostEqual(rec["cov_yy"], eyy, places=12)
                self.assertAlmostEqual(rec["radius_m"], radius95(exx, exy, eyy), places=12)
                self.assertIsNone(rec["confidence"])
                self.assertIs(rec["radius_calibrated"], False)

    def test_diagonal_link_has_an_off_diagonal_term(self):
        y = [0.0] * self.L
        li = self.geom.link_index(0, 2)
        y[li] = 1.0
        rec = weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm")
        d = 4.0 * math.sqrt(2.0)
        a = d * d / 12.0
        b = 0.5 * math.sqrt((d + 0.3) ** 2 - d * d)
        c = b * b / 3.0
        self.assertAlmostEqual(rec["cov_xy"], 0.5 * (a - c), places=12)

    def test_two_diagonals_give_the_centre(self):
        y = [0.0] * self.L
        y[self.geom.link_index(0, 2)] = 1.5
        y[self.geom.link_index(1, 3)] = 1.5
        rec = weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm")
        self.assertEqual((rec["x"], rec["y"]), (2.0, 2.0))
        self.assertAlmostEqual(rec["cov_xy"], 0.0, places=12)
        self.assertAlmostEqual(rec["cov_xx"], rec["cov_yy"], places=12)

    def test_nothing_over_the_gate_abstains(self):
        y = [0.5] * self.L      # 0.5 <= 3*0.1 + 0.25
        self.assertEqual(weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm"),
                         {"reason": "no_disturbance"})
        self.assertEqual(weighted_link_midpoint(self.geom, [0.0] * self.L, self.sig, "rssi_dbm"),
                         {"reason": "no_disturbance"})

    def test_floor_depends_on_the_signal(self):
        self.assertEqual(default_wlm_floor("rssi_dbm"), 0.25)
        self.assertEqual(default_wlm_floor("total_power_db"), 0.0)
        y = [0.0] * self.L
        y[0] = 0.5              # over 3*0.1 = 0.3, under 0.3 + 0.25
        self.assertEqual(weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm")["reason"],
                         "no_disturbance")
        self.assertNotIn("reason", weighted_link_midpoint(self.geom, y, self.sig, "total_power_db"))
        self.assertNotIn("reason", weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm",
                                                          wlm_floor=0.0))

    def test_zero_null_sigma_never_gates(self):
        y = [5.0] * self.L
        rec = weighted_link_midpoint(self.geom, y, [0.0] * self.L, "rssi_dbm")
        self.assertEqual(rec, {"reason": "no_disturbance"})

    def test_nan_fails_closed(self):
        y = [0.0] * self.L
        y[0] = NAN
        self.assertEqual(weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm")["reason"],
                         "no_disturbance")
        y = [1.0] * self.L
        self.assertEqual(weighted_link_midpoint(self.geom, y, [NAN] * self.L, "rssi_dbm")["reason"],
                         "no_disturbance")
        # A disturbed link and a NaN elsewhere: the weight sum is NaN.
        y = [0.0] * self.L
        y[0] = 2.0
        y[1] = NAN
        self.assertEqual(weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm")["reason"],
                         "no_disturbance")
        y[1] = 0.0
        self.assertEqual(weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm",
                                                wlm_gate_z=NAN)["reason"], "no_disturbance")

    def test_implausible_radius(self):
        y = [0.0] * self.L
        y[0] = 2.0
        for kw in ({"min_plausible_radius_m": 100.0}, {"wlm_radius_scale": 0.0},
                   {"wlm_radius_scale": NAN}, {"wlm_radius_scale": INF}):
            with self.subTest(**{k: repr(v) for k, v in kw.items()}):
                rec = weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm", **kw)
                self.assertEqual(rec, {"reason": "implausible_radius"})

    def test_zero_radius_raw_call_abstains(self):
        """§5 step 5 and §8's `r > 0`: a raw library call with a zero
        minimum (§9 validates it > 0 only in a config) and a zero or
        negative radius abstains `implausible_radius`; the same call with a
        positive radius publishes."""
        y = [0.0] * self.L
        y[0] = 2.0
        for scale, minr in ((0.0, 0.0), (-0.0, 0.0), (-1.0, -10.0), (0.0, -1.0)):
            with self.subTest(scale=scale, minr=minr):
                rec = weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm",
                                             wlm_radius_scale=scale, min_plausible_radius_m=minr)
                self.assertEqual(rec, {"reason": "implausible_radius"})
        rec = weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm", min_plausible_radius_m=0.0)
        self.assertNotIn("reason", rec)
        self.assertGreater(rec["radius_m"], 0.0)

    def test_non_finite_position_is_never_published(self):
        """§5 step 5: a weight sum that overflows makes x̂ NaN (radius 0):
        abstain, even at a minimum radius of 0 that only a direct call can
        pass (§9 validates it > 0)."""
        for y in ([1e308] * self.L, [INF] + [0.0] * (self.L - 1)):
            for minr in (0.0, -1.0, 0.3):
                with self.subTest(y0=y[0], minr=minr):
                    rec = weighted_link_midpoint(self.geom, y, self.sig, "rssi_dbm", min_plausible_radius_m=minr)
                    self.assertEqual(rec, {"reason": "implausible_radius"})

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            weighted_link_midpoint(self.geom, [0.0], self.sig, "rssi_dbm")
        with self.assertRaises(ValueError):
            weighted_link_midpoint(self.geom, [0.0] * self.L, [0.1], "rssi_dbm")


class TestConfig(unittest.TestCase):
    def assertNames(self, param, **kw):
        with self.assertRaises(LocateError) as cm:
            LocateConfig(**kw).validate()
        self.assertEqual(cm.exception.param, param)
        self.assertTrue(str(cm.exception).startswith(param + ":"))

    def test_defaults_validate(self):
        cfg = LocateConfig().validate()
        self.assertEqual(cfg.wlm_floor_value(), 0.25)
        self.assertEqual(LocateConfig(link_signal="total_power_db").wlm_floor_value(), 0.0)

    def test_every_number_finite(self):
        import dataclasses
        for f in dataclasses.fields(LocateConfig):
            if f.name in ("min_windows", "link_signal", "phone_seed"):
                continue
            for bad in (NAN, INF, -INF, "1.0", True):
                with self.subTest(param=f.name, value=bad):
                    self.assertNames(f.name, **{f.name: bad})

    def test_positive(self):
        # min_plausible_radius_m and the three scales too: a zero radius
        # would publish a bare point (ADR-008 d5); carrier_hz guards C / f.
        for p in ("rtt_sigma_m", "rssi_shadow_sigma_db", "rssi_shadow_corr_m", "rssi_breakpoint_m",
                  "phone_rate_hz", "huber_k", "rssi_far_exponent", "rssi_est_exponent",
                  "phone_accel_psd", "phone_init_speed_std", "tick_hz", "calib_s", "window_s",
                  "carrier_hz", "min_plausible_radius_m", "wlm_radius_scale", "rti_radius_scale",
                  "phone_radius_scale"):
            with self.subTest(param=p):
                self.assertNames(p, **{p: 0.0})
                self.assertNames(p, **{p: -0.0})
                self.assertNames(p, **{p: -1.0})
                LocateConfig(**{p: 5e-324}).validate()

    def test_non_negative(self):
        for p in ("rtt_nlos_sigma_scale", "rtt_nlos_bias_m", "rtt_offset_sigma_m", "rtt_persist_sigma_m",
                  "rssi_nlos_loss_db", "rssi_device_sigma_db", "rssi_persist_sigma_db", "rssi_jitter_db",
                  "phone_max_coast_s", "wlm_gate_z", "wlm_floor"):
            with self.subTest(param=p):
                self.assertNames(p, **{p: -1e-9})
                LocateConfig(**{p: 0.0}).validate()

    def test_probabilities(self):
        for p in ("rtt_outlier_p", "rtt_fail_p", "rssi_fail_p"):
            with self.subTest(param=p):
                self.assertNames(p, **{p: -0.01})
                self.assertNames(p, **{p: 1.01})
                LocateConfig(**{p: 0.0}).validate()
                LocateConfig(**{p: 1.0}).validate()

    def test_outlier_bounds(self):
        self.assertNames("rtt_outlier_lo_m", rtt_outlier_lo_m=9.0)
        LocateConfig(rtt_outlier_lo_m=8.0).validate()

    def test_integers_and_names(self):
        self.assertNames("phone_seed", phone_seed=-1)
        self.assertNames("phone_seed", phone_seed=1 << 64)
        self.assertNames("phone_seed", phone_seed=1.0)
        LocateConfig(phone_seed=(1 << 64) - 1).validate()
        self.assertNames("min_windows", min_windows=0)
        self.assertNames("min_windows", min_windows=10.0)
        self.assertNames("min_windows", min_windows=True)
        self.assertNames("min_windows", min_windows=-3)
        LocateConfig(min_windows=1).validate()
        self.assertNames("link_signal", link_signal="rssi")
        self.assertNames("link_signal", link_signal=None)
        for name in ("rssi_dbm", "total_power_db", "static_suppressed_amplitude"):
            LocateConfig(link_signal=name).validate()
        self.assertNames("carrier_hz", carrier_hz=0.0)
        # No measurement sample_rate_hz parameter (§9: derived, §4.3).
        with self.assertRaises(TypeError):
            LocateConfig(sample_rate_hz=20.0)

    def test_resolved(self):
        cfg = LocateConfig().resolved(42, 10.0, 2437e6)
        self.assertEqual(cfg.phone_seed, 42)
        self.assertEqual(cfg.wlm_floor, 0.25)
        lam = 299792458.0 / 2437e6
        self.assertEqual(cfg.rssi_p0_dbm, 10.0 - 20.0 * math.log10((4.0 * math.pi) / lam))
        self.assertEqual(LocateConfig(phone_seed=7).resolved(42, 10.0, 2437e6).phone_seed, 7)
        with self.assertRaises(LocateError) as cm:
            LocateConfig(tx_power_dbm=3.0).resolved(42, 10.0, 2437e6)
        self.assertEqual(cm.exception.param, "tx_power_dbm")
        with self.assertRaises(LocateError) as cm:
            LocateConfig(carrier_hz=2412e6).resolved(42, 10.0, 2437e6)
        self.assertEqual(cm.exception.param, "carrier_hz")
        same = LocateConfig(tx_power_dbm=10.0, carrier_hz=2437e6).resolved(42, 10.0, 2437e6)
        self.assertEqual((same.tx_power_dbm, same.carrier_hz), (10.0, 2437e6))
        self.assertEqual(LocateConfig(rssi_p0_dbm=-35.0).resolved(42, 10.0, 2437e6).rssi_p0_dbm, -35.0)

    def test_require_resolved_needs_only_room_sim_values(self):
        """§9: `require_resolved` names the first unset of `phone_seed`,
        `tx_power_dbm`, `carrier_hz` (as the Rust `Config::require_resolved`
        does); `rssi_p0_dbm` unset is derived where it is used."""
        for kw, param in (({}, "phone_seed"), ({"phone_seed": 1}, "tx_power_dbm"),
                          ({"phone_seed": 1, "tx_power_dbm": 10.0}, "carrier_hz"),
                          ({"tx_power_dbm": 10.0, "carrier_hz": 2437e6}, "phone_seed")):
            with self.subTest(**kw):
                with self.assertRaises(LocateError) as cm:
                    LocateConfig(**kw).require_resolved()
                self.assertEqual(cm.exception.param, param)
        plain = LocateConfig(phone_seed=1, tx_power_dbm=10.0, carrier_hz=2437e6)
        self.assertIs(plain.require_resolved(), plain)
        self.assertIsNone(plain.rssi_p0_dbm)

    def test_rssi_p0_value(self):
        """`rssi_p0_dbm`, else its §9 default from `tx_power_dbm` and
        `carrier_hz` (the value `resolved` fills), else refused naming
        `rssi_p0_dbm`."""
        lam = 299792458.0 / 2437e6
        want = 10.0 - 20.0 * math.log10((4.0 * math.pi) / lam)
        plain = LocateConfig(phone_seed=1, tx_power_dbm=10.0, carrier_hz=2437e6)
        self.assertEqual(plain.rssi_p0_value(), want)
        self.assertEqual(plain.rssi_p0_value(), LocateConfig().resolved(1, 10.0, 2437e6).rssi_p0_dbm)
        self.assertEqual(LocateConfig(rssi_p0_dbm=-35.0).rssi_p0_value(), -35.0)
        for kw in ({}, {"tx_power_dbm": 10.0}, {"carrier_hz": 2437e6}):
            with self.subTest(**kw):
                with self.assertRaises(LocateError) as cm:
                    LocateConfig(**kw).rssi_p0_value()
                self.assertEqual(cm.exception.param, "rssi_p0_dbm")


if __name__ == "__main__":
    unittest.main()
