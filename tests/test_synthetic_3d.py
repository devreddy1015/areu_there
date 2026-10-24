"""The geometric scene with height: floor, ceiling and mounting heights.

`room_height_m=None` is the 2-D model, and must stay exactly as it was;
everything else here is the 3-D box. The 2-D tests live in
test_synthetic.py.
"""

import cmath
import json
import math
import os
import unittest

from wifisense.sources import SyntheticCsiSource, SyntheticConfig
from wifisense.sources.synthetic import SPEED_OF_LIGHT, _box_images

MATERIALS = os.path.join(os.path.dirname(__file__), "..", "scenes", "materials.json")


def geo(**kw):
    return SyntheticConfig(**{"scene": "geometric", "noise_std": 0.0, **kw})


def box(**kw):
    return geo(room_height_m=2.7, **kw)


# A box that is the 2-D model in disguise: every height the same, floor
# and ceiling absorbing, furniture at that height too.
FLAT = dict(room_height_m=2.7, tx_z_m=1.2, rx_z_m=1.2, body_z_m=1.2,
            floor_reflection=0.0, ceiling_reflection=0.0, scatterer_z_m=(1.2, 1.2))


def channel(src):
    return [s + b for s, b in zip(src._static, src._body)]


def worst_relative(a, b):
    scale = max(abs(x) for x in a)
    return max(abs(x - y) for x, y in zip(a, b)) / scale


class TestTwoDimensionalModelUnchanged(unittest.TestCase):
    def test_height_fields_are_ignored_without_a_room_height(self):
        """Every number published from the 2-D scene depends on this."""
        kw = dict(n_scatterers=5, seed=4, reflection_order=2, body_points=3, body_width_m=0.3)
        a = SyntheticCsiSource(geo(**kw)).frames(5)
        b = SyntheticCsiSource(geo(tx_z_m=0.4, rx_z_m=2.0, body_z_m=0.6, floor_reflection=0.9,
                                   ceiling_reflection=0.1, scatterer_z_m=(0.1, 0.2), **kw)).frames(5)
        self.assertEqual([f.amplitudes for f in a], [f.amplitudes for f in b])
        self.assertEqual([f.phases for f in a], [f.phases for f in b])


class TestBoxImages(unittest.TestCase):
    def test_every_image_once(self):
        """The number of lattice points with 1 <= |nx|+|ny|+|nz| <= N is
        6, 24, 62 for N = 1, 2, 3. A breadth-first mirror search that only
        refuses to mirror straight back finds some of them twice."""
        p = (1.1, 2.3, 0.7)
        for order, count in ((0, 0), (1, 6), (2, 24), (3, 62)):
            images = _box_images(p, (6.0, 5.0, 2.7), order, 0.4, 0.3, 0.2)
            self.assertEqual(len(images), count)
            self.assertEqual(len({tuple(round(c, 9) for c in q) for q, _ in images}), count)

    def test_first_order_faces(self):
        p, size = (1.0, 2.0, 1.2), (6.0, 5.0, 2.7)
        got = {tuple(round(c, 12) for c in q): g for q, g in _box_images(p, size, 1, 0.4, 0.3, 0.2)}
        self.assertEqual(got, {
            (-1.0, 2.0, 1.2): -0.4, (11.0, 2.0, 1.2): -0.4,
            (1.0, -2.0, 1.2): -0.4, (1.0, 8.0, 1.2): -0.4,
            (1.0, 2.0, -1.2): -0.3,                       # floor
            (1.0, 2.0, round(5.4 - 1.2, 12)): -0.2,       # ceiling
        })

    def test_second_order_coefficients(self):
        """Floor then ceiling, either way round, is one bounce off each."""
        p, size = (1.0, 2.0, 1.2), (6.0, 5.0, 2.7)
        got = {tuple(round(c, 12) for c in q): g for q, g in _box_images(p, size, 2, 0.4, 0.3, 0.2)}
        self.assertAlmostEqual(got[(1.0, 2.0, round(5.4 + 1.2, 12))], 0.3 * 0.2, places=15)
        self.assertAlmostEqual(got[(1.0, 2.0, round(-5.4 + 1.2, 12))], 0.3 * 0.2, places=15)
        self.assertAlmostEqual(got[(-1.0, 2.0, -1.2)], 0.4 * 0.3, places=15)   # west wall + floor
        self.assertAlmostEqual(got[(-1.0, -2.0, 1.2)], 0.4 * 0.4, places=15)   # a corner, once

    def test_path_order_is_by_bounces(self):
        """A port sums in this order to reproduce the golden vectors, so
        the order is part of the spec: by bounces, then nx, ny, nz."""
        images = _box_images((1.0, 2.0, 1.2), (6.0, 5.0, 2.7), 2, 0.4, 0.3, 0.2)
        first = [tuple(round(c, 9) for c in q) for q, _ in images[:6]]
        self.assertEqual(first, [
            (-1.0, 2.0, 1.2),   # (-1, 0, 0)
            (1.0, -2.0, 1.2),   # (0, -1, 0)
            (1.0, 2.0, -1.2),   # (0, 0, -1)
            (1.0, 2.0, 4.2),    # (0, 0, 1)
            (1.0, 8.0, 1.2),    # (0, 1, 0)
            (11.0, 2.0, 1.2),   # (1, 0, 0)
        ])
        self.assertEqual(images[6][0], (-11.0, 2.0, 1.2))   # (-2, 0, 0) opens order 2


class TestReducesToTheFlatModel(unittest.TestCase):
    def test_order_one(self):
        """A flat box is the 2-D model: same paths, same body, same
        furniture footprint. Only the summation order differs, so the
        bound is a few ULP of the channel's scale."""
        kw = dict(n_scatterers=4, seed=7, body_points=3, body_width_m=0.3, body_xy_m=(2.6, 3.4))
        a = SyntheticCsiSource(geo(**kw))
        b = SyntheticCsiSource(geo(**FLAT, **kw))
        self.assertLess(worst_relative(channel(a), channel(b)), 1e-13)
        for x, y in zip(a.sensitivity(), b.sensitivity()):
            self.assertAlmostEqual(x, y, places=12)
        self.assertAlmostEqual(a.excess_path_m, b.excess_path_m, places=12)

    def test_order_two_counts_each_corner_once(self):
        """The 2-D model reaches the four corner images (one bounce off an
        x wall and one off a y wall) by two routes each, x-then-y and
        y-then-x, and adds both. A corner reflection is one specular path.
        The box adds it once, so the difference is exactly one copy of
        each corner path."""
        kw = dict(reflection_order=2, wall_reflection=0.45)
        cfg2 = geo(**kw)
        a = SyntheticCsiSource(cfg2)
        b = SyntheticCsiSource(geo(**FLAT, **kw))
        (tx, ty), rx = cfg2.tx_xy_m, cfg2.rx_xy_m
        w, h = cfg2.room_m
        scale = cfg2.static_magnitude * math.dist(cfg2.tx_xy_m, rx)
        corners = [(cx, cy) for cx in (-tx, 2 * w - tx) for cy in (-ty, 2 * h - ty)]
        for k in range(cfg2.n_subcarriers):
            f = a._frequency(k)
            extra = sum(scale * 0.45 ** 2 / math.dist(c, rx) * cmath.exp(-2j * math.pi * f * math.dist(c, rx) / SPEED_OF_LIGHT)
                        for c in corners)
            self.assertLess(abs(a._static[k] - extra - b._static[k]), 1e-12 * abs(a._static[k]))


class TestHeights(unittest.TestCase):
    def test_floor_and_ceiling_bounce(self):
        """A 3 m link at 1.2 m in a 2.7 m room: the floor bounce is 3.84 m
        long and the ceiling bounce 3.35 m, each with its own -Gamma."""
        src = SyntheticCsiSource(box(tx_xy_m=(1.5, 2.5), rx_xy_m=(4.5, 2.5), occupied=False,
                                     floor_reflection=0.39, ceiling_reflection=0.25))
        lengths = {round(l, 9): a for a, l in src.static_paths}
        floor, ceiling = math.hypot(3.0, 2.4), math.hypot(3.0, 2 * (2.7 - 1.2))
        self.assertAlmostEqual(lengths[round(floor, 9)] * floor, -0.39, places=12)
        self.assertAlmostEqual(lengths[round(ceiling, 9)] * ceiling, -0.25, places=12)
        self.assertEqual(src.static_paths[0], (1.0 / 3.0, 3.0))

    def test_los_uses_the_slant_range(self):
        cfg = box(wall_reflection=0.0, floor_reflection=0.0, ceiling_reflection=0.0, occupied=False,
                  tx_xy_m=(1.0, 1.0), rx_xy_m=(4.0, 1.0), tx_z_m=0.5, rx_z_m=2.5)
        src = SyntheticCsiSource(cfg)
        d = math.hypot(3.0, 2.0)
        step = -2 * math.pi * cfg.subcarrier_spacing_hz * d / SPEED_OF_LIGHT
        self.assertAlmostEqual(math.remainder(cmath.phase(src._static[1]) - cmath.phase(src._static[0]) - step,
                                              2 * math.pi), 0.0, places=9)
        for s in src._static:
            self.assertAlmostEqual(abs(s), cfg.static_magnitude, places=9)

    def test_excess_path(self):
        src = SyntheticCsiSource(box(tx_xy_m=(1.0, 2.0), rx_xy_m=(5.0, 2.0), body_xy_m=(3.0, 3.5),
                                     tx_z_m=1.2, rx_z_m=1.2, body_z_m=1.0))
        self.assertAlmostEqual(src.excess_path_m, 2 * math.sqrt(4.0 + 2.25 + 0.04) - 4.0, places=12)

    def test_chest_gain_level_with_the_link_is_the_2d_gain(self):
        a = SyntheticCsiSource(geo(body_xy_m=(2.2, 3.9)))
        b = SyntheticCsiSource(geo(**FLAT, body_xy_m=(2.2, 3.9)))
        self.assertAlmostEqual(a._body_gains[0], b._body_gains[0], places=12)

    def test_chest_moves_horizontally(self):
        """Gain is the horizontal part of (u_tx + u_rx), so a chest below
        the link loses some of it, and one directly below the midpoint has
        none at first order: moving sideways changes both legs equally and
        only to second order."""
        tx, rx = (1.0, 2.5), (5.0, 2.5)
        level = SyntheticCsiSource(box(tx_xy_m=tx, rx_xy_m=rx, tx_z_m=2.0, rx_z_m=2.0, body_z_m=2.0,
                                       body_xy_m=(3.0, 4.0)))._body_gains[0]
        below = SyntheticCsiSource(box(tx_xy_m=tx, rx_xy_m=rx, tx_z_m=2.0, rx_z_m=2.0, body_z_m=1.0,
                                       body_xy_m=(3.0, 4.0)))._body_gains[0]
        d = math.sqrt(4.0 + 2.25 + 1.0)
        self.assertAlmostEqual(below, 2 * 1.5 / d, places=12)
        self.assertAlmostEqual(level, 2 * 1.5 / math.sqrt(4.0 + 2.25), places=12)
        under = SyntheticCsiSource(box(tx_xy_m=tx, rx_xy_m=rx, tx_z_m=2.0, rx_z_m=2.0, body_z_m=1.0,
                                       body_xy_m=(3.0, 2.5)))
        self.assertEqual(under._body_gains[0], 0.0)
        self.assertTrue(all(v == 0.0 for v in under.sensitivity()))

    def test_torso_spreads_horizontally_at_chest_height(self):
        src = SyntheticCsiSource(box(body_points=5, body_width_m=0.4, body_z_m=1.1, body_xy_m=(3.0, 3.6)))
        self.assertEqual(len(src.body_point_positions), 5)
        self.assertTrue(all(p[2] == 1.1 for p in src.body_point_positions))
        first, last = src.body_point_positions[0], src.body_point_positions[-1]
        self.assertAlmostEqual(math.dist(first, last), 0.4, places=12)

    def test_positions_must_be_inside_the_box(self):
        for kw in (dict(body_z_m=2.8), dict(tx_z_m=0.0), dict(rx_z_m=-0.1), dict(room_height_m=0.0)):
            with self.assertRaises(ValueError, msg=kw):
                SyntheticCsiSource(geo(**{"room_height_m": 2.7, **kw}))

    def test_furniture_footprint_matches_the_2d_scene(self):
        """Heights are drawn after every 2-D draw, so a seed puts the same
        furniture in the same place in both models: a paired comparison."""
        a = SyntheticCsiSource(geo(n_scatterers=6, seed=21))
        b = SyntheticCsiSource(box(n_scatterers=6, seed=21, scatterer_z_m=(0.3, 1.8)))
        self.assertEqual([p[:2] for p in a.scatterers], [p[:2] for p in b.scatterers])
        self.assertEqual(a.scatterer_rcs, b.scatterer_rcs)
        self.assertTrue(all(0.3 <= p[2] <= 1.8 for p in b.scatterers))
        self.assertEqual(len({p[2] for p in b.scatterers}), 6)

    def test_sensitivity_bounded(self):
        for z in (0.6, 1.0, 1.35):
            for kw in ({}, dict(body_points=5, body_width_m=0.3)):
                s = SyntheticCsiSource(box(body_z_m=z, n_scatterers=4, seed=3, reflection_order=2, **kw)).sensitivity()
                self.assertTrue(all(0.0 <= v <= 1.0 + 1e-12 for v in s))

    def test_default_reflections_are_concrete(self):
        """The defaults are the ITU-R P.2040 concrete row the repository
        already carries, not numbers typed from memory."""
        with open(MATERIALS) as fh:
            concrete = json.load(fh)["materials"]["concrete"]["at_2437_mhz"]["reflection_normal"]
        cfg = SyntheticConfig()
        self.assertEqual(cfg.floor_reflection, round(concrete, 3))
        self.assertEqual(cfg.ceiling_reflection, round(concrete, 3))

    def test_deterministic_under_seed(self):
        a = SyntheticCsiSource(box(n_scatterers=5, seed=9, noise_std=0.6)).frames(10)
        b = SyntheticCsiSource(box(n_scatterers=5, seed=9, noise_std=0.6)).frames(10)
        self.assertEqual([f.amplitudes for f in a], [f.amplitudes for f in b])



class TestLyingChest(unittest.TestCase):
    """chest_axis="vertical": a person lying on their back, whose chest
    rises and falls instead of moving forward and back."""

    def test_gain_is_the_vertical_part(self):
        """Gain is (u_tx + u_rx) . z: largest straight below the link, the
        place an upright chest has none at all."""
        tx, rx = (1.0, 2.5), (5.0, 2.5)
        kw = dict(tx_xy_m=tx, rx_xy_m=rx, tx_z_m=1.2, rx_z_m=1.2, body_z_m=0.2, body_xy_m=(3.0, 2.5))
        lying = SyntheticCsiSource(box(chest_axis="vertical", **kw))
        upright = SyntheticCsiSource(box(**kw))
        self.assertAlmostEqual(lying._body_gains[0], 2 * 1.0 / math.sqrt(4.0 + 1.0), places=12)
        self.assertEqual(upright._body_gains[0], 0.0)
        self.assertGreater(max(lying.sensitivity()), 0.0)

    def test_level_with_the_link_it_has_none(self):
        src = SyntheticCsiSource(box(chest_axis="vertical", tx_z_m=1.0, rx_z_m=1.0, body_z_m=1.0))
        self.assertTrue(all(abs(g) < 1e-12 for g in src._body_gains))

    def test_nodes_below_the_chest_reverse_the_sign(self):
        """A chest above both nodes (a top bunk) lengthens the path as it
        rises: the gain is negative, and sensitivity still lands in [0, 1]."""
        src = SyntheticCsiSource(box(chest_axis="vertical", body_z_m=2.0, body_points=5,
                                     body_width_m=0.3, n_scatterers=4, seed=3))
        self.assertTrue(all(g < 0.0 for g in src._body_gains))
        self.assertTrue(all(0.0 <= v <= 1.0 + 1e-12 for v in src.sensitivity()))

    def test_torso_still_spreads_horizontally(self):
        src = SyntheticCsiSource(box(chest_axis="vertical", body_points=3, body_width_m=0.4, body_z_m=0.2))
        self.assertTrue(all(p[2] == 0.2 for p in src.body_point_positions))
        self.assertAlmostEqual(math.dist(src.body_point_positions[0], src.body_point_positions[-1]), 0.4,
                               places=12)

    def test_only_in_a_box_and_only_two_axes(self):
        with self.assertRaises(ValueError):
            SyntheticCsiSource(geo(chest_axis="vertical"))
        with self.assertRaises(ValueError):
            SyntheticCsiSource(box(chest_axis="sideways"))

    def test_horizontal_is_the_default(self):
        cfg = SyntheticConfig()
        self.assertEqual(cfg.chest_axis, "horizontal")
        a = SyntheticCsiSource(box(n_scatterers=3, seed=2)).frames(3)
        b = SyntheticCsiSource(box(n_scatterers=3, seed=2, chest_axis="horizontal")).frames(3)
        self.assertEqual([f.amplitudes for f in a], [f.amplitudes for f in b])


if __name__ == "__main__":
    unittest.main()
