"""Numerical tests for the RTI geometry and ellipse weight model.

The geometry layer is the one part of the localization stack with
unambiguous ground truth — distances are distances — so these tests check
real numbers against independently derived ones rather than against the
implementation's own helpers. In particular the perimeter spacing test
inverts the arc-length walk here, in the test file, instead of calling
anything in `geometry`; checking a mapping against itself would prove
nothing.

Whole file runs in well under a second.
"""

from __future__ import annotations

import math
import random
import unittest

from wifisense.localize.geometry import (
    Geometry,
    Link,
    Node,
    PixelGrid,
    Room,
    ellipse_weight,
    perimeter_nodes,
)

INSET = 0.1


def arc_length_on_inset_rect(room: Room, inset: float, x: float, y: float) -> float:
    """Independent inverse of the perimeter walk, for the spacing test.

    Given a point known to lie on the inset rectangle, return how far it
    is counter-clockwise from the bottom-left inset corner. Written from
    the geometry of a rectangle, not from `perimeter_nodes`.
    """
    x0 = y0 = inset
    x1 = room.width_m - inset
    y1 = room.height_m - inset
    w = x1 - x0
    h = y1 - y0
    tol = 1e-9

    if abs(y - y0) < tol and x0 - tol <= x <= x1 + tol:
        return x - x0
    if abs(x - x1) < tol and y0 - tol <= y <= y1 + tol:
        return w + (y - y0)
    if abs(y - y1) < tol and x0 - tol <= x <= x1 + tol:
        return w + h + (x1 - x)
    if abs(x - x0) < tol and y0 - tol <= y <= y1 + tol:
        return 2.0 * w + h + (y1 - y)
    raise AssertionError(f"({x}, {y}) is not on the inset rectangle at all")


class TestRoom(unittest.TestCase):
    def test_defaults_are_eight_by_six(self):
        room = Room()
        self.assertEqual((room.width_m, room.height_m), (8.0, 6.0))
        self.assertEqual(room.area_m2, 48.0)
        self.assertEqual(room.perimeter_m, 28.0)
        self.assertEqual(room.center, (4.0, 3.0))

    def test_rejects_non_positive_dimensions(self):
        for w, h in ((0.0, 6.0), (8.0, 0.0), (-1.0, 6.0)):
            with self.assertRaises(ValueError):
                Room(w, h)

    def test_contains(self):
        room = Room()
        self.assertTrue(room.contains(0.0, 0.0))
        self.assertTrue(room.contains(8.0, 6.0))
        self.assertFalse(room.contains(8.01, 3.0))
        self.assertTrue(room.contains(8.01, 3.0, tol=0.05))


class TestPerimeterPlacement(unittest.TestCase):
    """Even spacing around a rectangle means even ARC LENGTH.

    The failure these guard against is per-edge placement (N/4 nodes on
    each side), which on an 8 x 6 room stretches the spacing on the long
    edges by 1.33x relative to the short ones.
    """

    def test_spacing_is_equal_arc_length_for_many_n(self):
        room = Room()
        inset_perimeter = 2.0 * ((8.0 - 2 * INSET) + (6.0 - 2 * INSET))
        for n in (2, 3, 4, 5, 6, 7, 8, 11, 12, 13, 17):
            with self.subTest(n=n):
                nodes = perimeter_nodes(room, n, INSET)
                self.assertEqual(len(nodes), n)
                s = [arc_length_on_inset_rect(room, INSET, nd.x, nd.y) for nd in nodes]
                step = inset_perimeter / n
                gaps = [(s[(i + 1) % n] - s[i]) % inset_perimeter for i in range(n)]
                for gap in gaps:
                    self.assertAlmostEqual(gap, step, places=9)

    def test_long_edges_get_more_nodes_than_short_edges(self):
        """The property that separates arc-length from per-edge placement.

        A per-edge implementation puts N/4 nodes on each side, so an 8 x 6
        room would get 2/2/2/2 for N=8. Equal arc length must instead give
        the 7.8 m edges more nodes than the 5.8 m edges — here 3/1/3/1.
        Asserting the counts directly is what makes the preceding spacing
        test non-vacuous.
        """
        room = Room()
        w = room.width_m - 2 * INSET
        h = room.height_m - 2 * INSET
        nodes = perimeter_nodes(room, 8, INSET)
        s = [arc_length_on_inset_rect(room, INSET, nd.x, nd.y) for nd in nodes]
        per_edge = [0, 0, 0, 0]
        for value in s:
            if value < w:
                per_edge[0] += 1
            elif value < w + h:
                per_edge[1] += 1
            elif value < 2 * w + h:
                per_edge[2] += 1
            else:
                per_edge[3] += 1
        self.assertEqual(per_edge, [3, 1, 3, 1])
        self.assertNotEqual(per_edge, [2, 2, 2, 2])

    def test_odd_n_positions_are_all_distinct(self):
        for n in (3, 5, 7, 11, 13):
            nodes = perimeter_nodes(Room(), n, INSET)
            s = [
                arc_length_on_inset_rect(Room(), INSET, nd.x, nd.y) for nd in nodes
            ]
            self.assertEqual(len(set(round(v, 9) for v in s)), n)
            self.assertAlmostEqual(s[0], 0.0, places=9)

    def test_nodes_lie_inside_the_room_and_on_the_inset_rectangle(self):
        room = Room()
        for n in (4, 7, 12):
            for node in perimeter_nodes(room, n, INSET):
                self.assertTrue(room.contains(node.x, node.y))
                # Raises if the node is off the inset rectangle.
                arc_length_on_inset_rect(room, INSET, node.x, node.y)

    def test_ids_are_sequential_from_zero(self):
        nodes = perimeter_nodes(Room(), 9, INSET)
        self.assertEqual([n.node_id for n in nodes], list(range(9)))

    def test_start_offset_rotates_the_pattern_but_keeps_spacing(self):
        room = Room()
        inset_perimeter = 2.0 * ((8.0 - 2 * INSET) + (6.0 - 2 * INSET))
        offset = inset_perimeter / 16.0
        nodes = perimeter_nodes(room, 8, INSET, start_offset_m=offset)
        s = [arc_length_on_inset_rect(room, INSET, nd.x, nd.y) for nd in nodes]
        self.assertAlmostEqual(s[0], offset, places=9)
        gaps = [(s[(i + 1) % 8] - s[i]) % inset_perimeter for i in range(8)]
        for gap in gaps:
            self.assertAlmostEqual(gap, inset_perimeter / 8.0, places=9)

    def test_non_square_room_spacing_crosses_corners(self):
        """A 2 x 10 room is where per-edge placement fails loudest."""
        room = Room(2.0, 10.0)
        inset_perimeter = 2.0 * ((2.0 - 2 * INSET) + (10.0 - 2 * INSET))
        nodes = perimeter_nodes(room, 6, INSET)
        s = [arc_length_on_inset_rect(room, INSET, nd.x, nd.y) for nd in nodes]
        gaps = [(s[(i + 1) % 6] - s[i]) % inset_perimeter for i in range(6)]
        for gap in gaps:
            self.assertAlmostEqual(gap, inset_perimeter / 6.0, places=9)

    def test_rejects_bad_arguments(self):
        with self.assertRaises(ValueError):
            perimeter_nodes(Room(), 1, INSET)
        with self.assertRaises(ValueError):
            perimeter_nodes(Room(), 4, -0.1)
        with self.assertRaises(ValueError):
            perimeter_nodes(Room(8.0, 6.0), 4, 3.5)  # inset collapses the 6 m side


class TestLinks(unittest.TestCase):
    def test_link_count_is_n_times_n_minus_one(self):
        for n in (2, 3, 4, 6, 8, 12):
            with self.subTest(n=n):
                g = Geometry.perimeter(n_nodes=n)
                self.assertEqual(g.n_links, n * (n - 1))
                self.assertEqual(len(g.links), n * (n - 1))

    def test_no_self_links_and_no_duplicates(self):
        g = Geometry.perimeter(n_nodes=8)
        pairs = [(l.tx_id, l.rx_id) for l in g.links]
        self.assertEqual(len(set(pairs)), len(pairs))
        self.assertFalse(any(tx == rx for tx, rx in pairs))

    def test_canonical_order_is_sorted_by_tx_then_rx(self):
        g = Geometry.perimeter(n_nodes=6)
        pairs = [(l.tx_id, l.rx_id) for l in g.links]
        self.assertEqual(pairs, sorted(pairs))

    def test_link_index_round_trips(self):
        g = Geometry.perimeter(n_nodes=8)
        for i, link in enumerate(g.links):
            self.assertEqual(g.link_index(link.tx_id, link.rx_id), i)

    def test_link_index_rejects_unknown_pairs(self):
        g = Geometry.perimeter(n_nodes=4)
        with self.assertRaises(KeyError):
            g.link_index(0, 0)
        with self.assertRaises(KeyError):
            g.link_index(0, 99)

    def test_length_matches_euclidean_distance(self):
        g = Geometry.perimeter(n_nodes=7)
        for link in g.links:
            expected = math.hypot(link.tx[0] - link.rx[0], link.tx[1] - link.rx[1])
            self.assertAlmostEqual(link.length_m, expected, places=12)
            self.assertEqual(link.d_l, link.length_m)

    def test_excess_path_is_zero_on_the_segment(self):
        g = Geometry.perimeter(n_nodes=6)
        for link in g.links[:10]:
            for t in (0.0, 0.1, 0.5, 0.73, 1.0):
                x = link.tx[0] + t * (link.rx[0] - link.tx[0])
                y = link.tx[1] + t * (link.rx[1] - link.tx[1])
                self.assertAlmostEqual(link.excess_path_m(x, y), 0.0, places=12)

    def test_excess_path_is_positive_off_the_segment(self):
        link = Link(0, 1, (0.0, 0.0), (4.0, 0.0), 4.0)
        self.assertGreater(link.excess_path_m(2.0, 1.0), 0.0)
        self.assertGreater(link.excess_path_m(-1.0, 0.0), 0.0)
        self.assertAlmostEqual(link.excess_path_m(-1.0, 0.0), 2.0, places=12)


class TestEllipseMembership(unittest.TestCase):
    LINK = Link(0, 1, (0.0, 0.0), (4.0, 0.0), 4.0)

    def test_line_of_sight_pixels_are_always_inside(self):
        """Excess is 0 on the segment, so any positive lambda admits it.

        Swept over real deployment links and down to a 1 mm lambda,
        because this is the invariant the whole model rests on: if the
        direct path itself could fall outside the ellipse, a person
        standing in a link would be invisible to that link.
        """
        g = Geometry.perimeter(n_nodes=8)
        for link in g.links:
            for t in (0.0, 0.25, 0.5, 0.9, 1.0):
                x = link.tx[0] + t * (link.rx[0] - link.tx[0])
                y = link.tx[1] + t * (link.rx[1] - link.tx[1])
                for lam in (0.001, 0.05, 0.3, 2.0):
                    self.assertGreater(
                        ellipse_weight(link, x, y, lam),
                        0.0,
                        msg=f"link {link.tx_id}->{link.rx_id} t={t} lambda={lam}",
                    )

    def test_weight_value_is_one_over_sqrt_length(self):
        w = ellipse_weight(self.LINK, 2.0, 0.0, 0.5)
        self.assertAlmostEqual(w, 1.0 / math.sqrt(4.0), places=12)

    def test_far_pixels_are_always_outside(self):
        g = Geometry.perimeter(n_nodes=8)
        far_points = [(-50.0, -50.0), (100.0, 0.0), (4.0, 80.0)]
        for link in g.links:
            for x, y in far_points:
                self.assertEqual(ellipse_weight(link, x, y, 3.0), 0.0)

    def test_boundary_is_exactly_excluded(self):
        """`excess < lambda` is strict, and the boundary case is exact.

        The pixel at (-0.5, 0) on a link from (0,0) to (4,0) gives
        d1 = 0.5 and d2 = 4.5 with no rounding at all (hypot with a zero
        component is exact), so the excess is exactly 1.0. At lambda
        exactly 1.0 the pixel is out; one ulp above, it is in.
        """
        excess = self.LINK.excess_path_m(-0.5, 0.0)
        self.assertEqual(excess, 1.0)
        self.assertEqual(ellipse_weight(self.LINK, -0.5, 0.0, 1.0), 0.0)
        just_over = math.nextafter(1.0, 2.0)
        self.assertGreater(ellipse_weight(self.LINK, -0.5, 0.0, just_over), 0.0)
        just_under = math.nextafter(1.0, 0.0)
        self.assertEqual(ellipse_weight(self.LINK, -0.5, 0.0, just_under), 0.0)

    def test_membership_is_monotone_across_the_boundary(self):
        """Walking outward along the major axis crosses once and stays out."""
        lam = 1.0
        inside = [ellipse_weight(self.LINK, x, 0.0, lam) > 0.0
                  for x in (-0.49, -0.3, 0.0, 2.0, 4.0, 4.3, 4.49)]
        self.assertTrue(all(inside))
        outside = [ellipse_weight(self.LINK, x, 0.0, lam) == 0.0
                   for x in (-0.51, -1.0, -5.0, 4.51, 5.0, 20.0)]
        self.assertTrue(all(outside))

    def test_minor_axis_half_width_matches_the_ellipse_formula(self):
        """b = sqrt(a^2 - c^2) with a = (d_l + lambda)/2 and c = d_l/2.

        For d_l = 4 and lambda = 1, a = 2.5, c = 2, so b = 1.5. Checking
        the perpendicular extent independently of the collinear case
        catches a model that thresholded, say, perpendicular distance to
        the line instead of summed path length.
        """
        lam = 1.0
        a = (4.0 + lam) / 2.0
        b = math.sqrt(a * a - 2.0 * 2.0)
        self.assertAlmostEqual(b, 1.5, places=12)
        self.assertGreater(ellipse_weight(self.LINK, 2.0, b - 1e-6, lam), 0.0)
        self.assertEqual(ellipse_weight(self.LINK, 2.0, b + 1e-6, lam), 0.0)

    def test_focus_pixel_is_inside(self):
        self.assertGreater(ellipse_weight(self.LINK, 0.0, 0.0, 0.01), 0.0)
        self.assertGreater(ellipse_weight(self.LINK, 4.0, 0.0, 0.01), 0.0)

    def test_rejects_non_positive_lambda(self):
        for lam in (0.0, -0.5):
            with self.assertRaises(ValueError):
                ellipse_weight(self.LINK, 2.0, 0.0, lam)

    def test_degenerate_zero_length_link_has_zero_weight(self):
        """1/sqrt(0) is not a weight; guard rather than emit inf."""
        degenerate = Link(0, 1, (1.0, 1.0), (1.0, 1.0), 0.0)
        self.assertEqual(ellipse_weight(degenerate, 1.0, 1.0, 1.0), 0.0)


class TestSymmetry(unittest.TestCase):
    """The ellipse does not care which end transmits.

    Equality here is exact, not approximate: d(tx,p) + d(p,rx) versus
    d(rx,p) + d(p,tx) is `u + v` versus `v + u`, and IEEE 754 addition is
    commutative; hypot of exactly negated arguments gives the identical
    result. If this ever becomes merely approximate, something has started
    accumulating state per direction.
    """

    def test_ellipse_weight_is_direction_independent(self):
        rng = random.Random(20261013)
        for _ in range(400):
            tx = (rng.uniform(0.0, 8.0), rng.uniform(0.0, 6.0))
            rx = (rng.uniform(0.0, 8.0), rng.uniform(0.0, 6.0))
            d = math.hypot(tx[0] - rx[0], tx[1] - rx[1])
            if d < 0.5:
                continue
            fwd = Link(0, 1, tx, rx, d)
            rev = Link(1, 0, rx, tx, d)
            px = rng.uniform(-1.0, 9.0)
            py = rng.uniform(-1.0, 7.0)
            lam = rng.choice([0.05, 0.3, 0.75, 2.0])
            self.assertEqual(
                ellipse_weight(fwd, px, py, lam),
                ellipse_weight(rev, px, py, lam),
            )

    def test_weight_matrix_rows_are_identical_for_reversed_links(self):
        g = Geometry.perimeter(n_nodes=8, pitch_m=0.5, lambda_m=0.3)
        w = g.weight_matrix()
        for link in g.links:
            fwd = w[g.link_index(link.tx_id, link.rx_id)]
            rev = w[g.link_index(link.rx_id, link.tx_id)]
            self.assertEqual(fwd, rev)

    def test_link_lengths_are_identical_for_reversed_links(self):
        g = Geometry.perimeter(n_nodes=7)
        for link in g.links:
            other = g.links[g.link_index(link.rx_id, link.tx_id)]
            self.assertEqual(link.length_m, other.length_m)


class TestPixelGrid(unittest.TestCase):
    def test_default_pitch_gives_expected_count(self):
        grid = PixelGrid(Room(), 0.5)
        self.assertEqual((grid.n_x, grid.n_y), (16, 12))
        self.assertEqual(len(grid.pixels), 192)

    def test_quarter_metre_pitch_quadruples_the_count(self):
        coarse = PixelGrid(Room(), 0.5)
        fine = PixelGrid(Room(), 0.25)
        self.assertEqual((fine.n_x, fine.n_y), (32, 24))
        self.assertEqual(len(fine.pixels), 768)
        self.assertEqual(len(fine.pixels), 4 * len(coarse.pixels))

    def test_pitch_count_relationship_over_a_sweep(self):
        room = Room()
        for pitch, expected in ((2.0, 4 * 3), (1.0, 8 * 6), (0.5, 16 * 12),
                                (0.25, 32 * 24), (0.2, 40 * 30)):
            with self.subTest(pitch=pitch):
                self.assertEqual(len(PixelGrid(room, pitch).pixels), expected)

    def test_all_pixel_centres_are_inside_the_room(self):
        room = Room()
        for pitch in (1.0, 0.5, 0.25, 0.7, 0.33):
            grid = PixelGrid(room, pitch)
            for x, y in grid.pixels:
                self.assertTrue(room.contains(x, y), msg=f"({x}, {y}) pitch={pitch}")
                self.assertGreater(x, 0.0)
                self.assertGreater(y, 0.0)

    def test_grid_tiles_the_room_exactly(self):
        """Realised pitch is derived so n * pitch == the room dimension."""
        room = Room()
        for pitch in (0.5, 0.25, 0.7, 0.33):
            grid = PixelGrid(room, pitch)
            self.assertAlmostEqual(grid.n_x * grid.pitch_x_m, room.width_m, places=12)
            self.assertAlmostEqual(grid.n_y * grid.pitch_y_m, room.height_m, places=12)
            self.assertAlmostEqual(
                len(grid.pixels) * grid.pixel_area_m2, room.area_m2, places=9
            )

    def test_non_divisor_pitch_is_adjusted_not_truncated(self):
        grid = PixelGrid(Room(), 0.7)
        self.assertEqual((grid.n_x, grid.n_y), (11, 9))
        self.assertNotAlmostEqual(grid.pitch_x_m, 0.7, places=3)
        self.assertAlmostEqual(grid.pitch_x_m, 8.0 / 11.0, places=12)

    def test_ordering_is_row_major_in_y(self):
        grid = PixelGrid(Room(), 0.5)
        self.assertEqual(grid.pixels[0], (0.25, 0.25))
        self.assertEqual(grid.pixels[1], (0.75, 0.25))
        self.assertEqual(grid.pixels[grid.n_x], (0.25, 0.75))
        for iy in range(grid.n_y):
            for ix in range(grid.n_x):
                self.assertEqual(grid.pixel_index(ix, iy), iy * grid.n_x + ix)

    def test_pixel_index_rejects_out_of_range(self):
        grid = PixelGrid(Room(), 0.5)
        with self.assertRaises(IndexError):
            grid.pixel_index(16, 0)
        with self.assertRaises(IndexError):
            grid.pixel_index(0, -1)

    def test_nearest_index_round_trips_every_centre(self):
        grid = PixelGrid(Room(), 0.25)
        for i, (x, y) in enumerate(grid.pixels):
            self.assertEqual(grid.nearest_index(x, y), i)

    def test_nearest_index_clamps_outside_the_room(self):
        grid = PixelGrid(Room(), 0.5)
        self.assertEqual(grid.nearest_index(-3.0, -3.0), 0)
        self.assertEqual(grid.nearest_index(99.0, 99.0), len(grid.pixels) - 1)

    def test_rejects_non_positive_pitch(self):
        for pitch in (0.0, -0.5):
            with self.assertRaises(ValueError):
                PixelGrid(Room(), pitch)


class TestWeightMatrix(unittest.TestCase):
    def test_shape_is_links_by_pixels(self):
        g = Geometry.perimeter(n_nodes=8, pitch_m=0.5)
        w = g.weight_matrix()
        self.assertEqual(len(w), g.n_links)
        self.assertEqual(len(w), 56)
        for row in w:
            self.assertEqual(len(row), g.n_pixels)
            self.assertEqual(len(row), 192)

    def test_agrees_with_ellipse_weight_entry_by_entry(self):
        """The fast path precomputes node-pixel distances; prove it did not
        change the answer relative to the straightforward formula."""
        g = Geometry.perimeter(n_nodes=6, pitch_m=0.5, lambda_m=0.4)
        w = g.weight_matrix()
        for i, link in enumerate(g.links):
            for p, (x, y) in enumerate(g.grid.pixels):
                self.assertEqual(w[i][p], ellipse_weight(link, x, y, g.lambda_m))

    def test_nonzero_entries_all_equal_one_over_sqrt_length(self):
        g = Geometry.perimeter(n_nodes=8, pitch_m=0.5)
        w = g.weight_matrix()
        for i, link in enumerate(g.links):
            expected = 1.0 / math.sqrt(link.length_m)
            for value in w[i]:
                self.assertIn(value, (0.0, expected))

    def test_supports_match_the_dense_matrix(self):
        g = Geometry.perimeter(n_nodes=6, pitch_m=0.5)
        w = g.weight_matrix()
        for row, support in zip(w, g.supports()):
            self.assertEqual(
                support, [p for p, value in enumerate(row) if value != 0.0]
            )
            self.assertEqual(support, sorted(support))

    def test_cached_between_calls(self):
        g = Geometry.perimeter(n_nodes=4, pitch_m=0.5)
        self.assertIs(g.weight_matrix(), g.weight_matrix())

    def test_every_link_sees_at_least_one_pixel(self):
        """A link whose ellipse is empty contributes a row of zeros and
        makes W C W^T singular in that direction."""
        for n in (4, 6, 8, 12):
            for pitch in (0.5, 0.25):
                g = Geometry.perimeter(n_nodes=n, pitch_m=pitch, lambda_m=0.3)
                for i, support in enumerate(g.supports()):
                    self.assertGreater(
                        len(support), 0, msg=f"n={n} pitch={pitch} link {i}"
                    )

    def test_matrix_is_sparse_in_practice(self):
        """Documented as 16-21% nonzero; this is why `supports` exists."""
        g = Geometry.perimeter(n_nodes=8, pitch_m=0.25, lambda_m=0.3)
        nonzero = sum(len(s) for s in g.supports())
        fraction = nonzero / (g.n_links * g.n_pixels)
        self.assertLess(fraction, 0.25)
        self.assertGreater(fraction, 0.10)


class TestLambdaMonotonicity(unittest.TestCase):
    LAMBDAS = (0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)

    def test_larger_lambda_includes_strictly_more_pixels(self):
        base = Geometry.perimeter(n_nodes=8, pitch_m=0.5, lambda_m=0.3)
        counts = []
        for lam in self.LAMBDAS:
            g = base.with_lambda(lam)
            counts.append(sum(len(s) for s in g.supports()))
        for earlier, later in zip(counts, counts[1:]):
            self.assertLess(earlier, later)

    def test_smaller_lambda_support_is_a_subset_of_larger(self):
        """Strictly more is necessary but not sufficient — the sets must
        also nest, or the ellipse test is not a monotone threshold."""
        base = Geometry.perimeter(n_nodes=6, pitch_m=0.5, lambda_m=0.3)
        previous = None
        for lam in self.LAMBDAS:
            sets = [set(s) for s in base.with_lambda(lam).supports()]
            if previous is not None:
                for small, large in zip(previous, sets):
                    self.assertTrue(small <= large)
            previous = sets

    def test_with_lambda_preserves_room_nodes_and_grid(self):
        base = Geometry.perimeter(n_nodes=8, pitch_m=0.25, lambda_m=0.3)
        other = base.with_lambda(1.0)
        self.assertIs(other.grid, base.grid)
        self.assertEqual(other.lambda_m, 1.0)
        self.assertEqual(
            [n.position for n in other.nodes], [n.position for n in base.nodes]
        )
        self.assertEqual(
            [(l.tx_id, l.rx_id) for l in other.links],
            [(l.tx_id, l.rx_id) for l in base.links],
        )


class TestCoverage(unittest.TestCase):
    """Locks the blind-spot numbers documented on `pixel_link_counts`.

    These are the honest limitation of the model, so they are asserted
    rather than left as prose that can drift out of date.
    """

    def test_default_lambda_leaves_the_two_unoccupied_corners_blind(self):
        g = Geometry.perimeter(n_nodes=8, pitch_m=0.5, lambda_m=0.3)
        counts = g.pixel_link_counts()
        holes = [g.grid.pixels[i] for i, c in enumerate(counts) if c == 0]
        self.assertEqual(len(holes), 2)
        self.assertEqual(sorted(holes), [(0.25, 5.75), (7.75, 0.25)])

    def test_wider_lambda_closes_the_corner_holes(self):
        g = Geometry.perimeter(n_nodes=8, pitch_m=0.5, lambda_m=0.75)
        self.assertEqual(g.pixel_link_counts().count(0), 0)
        fine = Geometry.perimeter(n_nodes=8, pitch_m=0.25, lambda_m=1.0)
        self.assertEqual(fine.pixel_link_counts().count(0), 0)

    def test_more_nodes_does_not_fix_the_corners(self):
        """N=12 has the same two holes as N=8: the corners lie outside the
        convex hull of inset nodes, and no extra link reaches around."""
        for n in (8, 12):
            g = Geometry.perimeter(n_nodes=n, pitch_m=0.5, lambda_m=0.3)
            self.assertEqual(g.pixel_link_counts().count(0), 2, msg=f"n={n}")

    def test_four_nodes_is_blind_in_the_interior_too(self):
        """The qualitative failure: 12 links cannot cover an 8 x 6 room."""
        g = Geometry.perimeter(n_nodes=4, pitch_m=0.5, lambda_m=0.3)
        counts = g.pixel_link_counts()
        holes = [g.grid.pixels[i] for i, c in enumerate(counts) if c == 0]
        self.assertEqual(len(holes), 16)
        interior = [
            (x, y) for x, y in holes if 1.0 < x < 7.0 and 1.0 < y < 5.0
        ]
        self.assertGreater(len(interior), 0)

    def test_coverage_counts_sum_to_total_support(self):
        g = Geometry.perimeter(n_nodes=6, pitch_m=0.5, lambda_m=0.5)
        self.assertEqual(
            sum(g.pixel_link_counts()), sum(len(s) for s in g.supports())
        )


class TestGeometryValidation(unittest.TestCase):
    def test_rejects_fewer_than_two_nodes(self):
        with self.assertRaises(ValueError):
            Geometry(Room(), [Node(0, 1.0, 1.0)])

    def test_rejects_duplicate_node_ids(self):
        with self.assertRaises(ValueError):
            Geometry(Room(), [Node(0, 1.0, 1.0), Node(0, 2.0, 2.0)])

    def test_rejects_coincident_nodes(self):
        with self.assertRaises(ValueError):
            Geometry(Room(), [Node(0, 1.0, 1.0), Node(1, 1.0, 1.0)])

    def test_rejects_non_positive_lambda(self):
        with self.assertRaises(ValueError):
            Geometry.perimeter(n_nodes=4, lambda_m=0.0)

    def test_node_lookup(self):
        g = Geometry.perimeter(n_nodes=5)
        self.assertEqual(g.node(3).node_id, 3)
        with self.assertRaises(KeyError):
            g.node(99)

    def test_counts_are_consistent(self):
        g = Geometry.perimeter(n_nodes=8, pitch_m=0.25)
        self.assertEqual(g.n_nodes, 8)
        self.assertEqual(g.n_pixels, len(g.grid.pixels))
        self.assertEqual(g.n_links, len(g.links))

    def test_accepts_hand_placed_nodes_with_arbitrary_ids(self):
        nodes = [Node(7, 0.5, 0.5), Node(3, 7.5, 0.5), Node(11, 4.0, 5.5)]
        g = Geometry(Room(), nodes)
        self.assertEqual(g.n_links, 6)
        self.assertEqual(
            [(l.tx_id, l.rx_id) for l in g.links],
            [(3, 7), (3, 11), (7, 3), (7, 11), (11, 3), (11, 7)],
        )
        self.assertEqual(g.link_index(11, 7), 5)


if __name__ == "__main__":
    unittest.main()
