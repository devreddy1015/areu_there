"""Numerical tests for the dual-form regularized RTI inversion.

The failure mode this module is guarding against is not a crash. It is an
inversion that runs, returns a smooth plausible image, and puts the person
in the wrong place — a mirrored image from a transposed index, or a solver
that quietly returns garbage on an ill-conditioned matrix. So the tests
here check numbers against ground truth derived OUTSIDE the module:

  * the Cholesky solver against a 3 x 3 inverse computed by hand on paper
    (cofactors over a determinant of exactly 64, so every entry is an
    exact binary fraction and the comparison can be near-exact), and
    against a Gaussian elimination with partial pivoting written from
    scratch in this file;
  * the Toeplitz `SpatialPrior` against the direct exp(-dist/delta_c)
    formula evaluated on the actual pixel coordinates, ON A GRID WHOSE
    TWO PITCHES DIFFER. That last part is load-bearing and was missing:
    every grid in this file used to be square-pitched, and a mutant that
    swapped pitch_x for pitch_y in the kernel — the exact wrong-axis typo
    — passed all 52 tests (`ran=52 failures=0 errors=0`). It is a 3-5%
    systematic error in every entry of C, stretched along the wrong room
    axis, and `PixelGrid` reaches unequal pitches on 7 of 13 ordinary
    pitch values for the default room;
  * the dual form against a primal solve, x = (W^T W + alpha C^-1)^-1 W^T y,
    built here with the test file's own Gaussian elimination and the
    module's dense C. That is the single most valuable test in the file:
    the two routes share no code beyond the prior, so agreement to
    rounding means the push-through identity really was implemented. It
    also pins the module's deduplication of reciprocal links, because the
    primal here is assembled from the FULL dense W with both directions
    present — if collapsing the duplicate rows changed the estimator by
    anything beyond rounding, this test is where it would show.

Whole file runs in about 5 s.
"""

from __future__ import annotations

import math
import random
import sys
import unittest
import unittest.mock
import warnings

from wifisense.localize import invert as invert_module
from wifisense.localize.geometry import Geometry, PixelGrid, Room, perimeter_nodes
from wifisense.localize.invert import (
    AsymmetricMatrixError,
    BlindPixelWarning,
    IllConditionedError,
    Inverter,
    LinearAlgebraError,
    MalformedMatrixError,
    NotPositiveDefiniteError,
    SpatialPrior,
    cholesky_factor,
    cholesky_inverse,
    cholesky_solve,
    forward_project,
)

# ---------------------------------------------------------------------------
# Independent reference implementations. Nothing below imports from invert.py.
# ---------------------------------------------------------------------------


def gauss_solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Gauss-Jordan with partial pivoting, written independently of the module.

    Deliberately a different algorithm from the module's Cholesky, so that
    agreement between them is evidence rather than a tautology. Pivoting on
    the largest magnitude in the column is what makes it usable on the
    non-symmetric primal matrix too.
    """
    n = len(a)
    m = [list(row) + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        pivot = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[pivot][col]) < 1e-300:
            raise ZeroDivisionError(f"singular at column {col}")
        m[col], m[pivot] = m[pivot], m[col]
        pv = m[col][col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] / pv
            if f != 0.0:
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def gauss_inverse(a: list[list[float]]) -> list[list[float]]:
    n = len(a)
    cols = [
        gauss_solve(a, [1.0 if i == j else 0.0 for i in range(n)]) for j in range(n)
    ]
    return [[cols[j][i] for j in range(n)] for i in range(n)]


def matvec(a: list[list[float]], x: list[float]) -> list[float]:
    return [sum(ai * xi for ai, xi in zip(row, x)) for row in a]


def random_spd(n: int, rng: random.Random) -> list[list[float]]:
    """A random symmetric positive definite matrix, via M M^T + n I."""
    m = [[rng.uniform(-1.0, 1.0) for _ in range(n)] for _ in range(n)]
    return [
        [sum(m[i][k] * m[j][k] for k in range(n)) + (n if i == j else 0.0) for j in range(n)]
        for i in range(n)
    ]


def gaussian_blob(
    geometry: Geometry, cx: float, cy: float, width_m: float = 0.4
) -> list[float]:
    """A known ground-truth image: one Gaussian bump centred at (cx, cy)."""
    two_w2 = 2.0 * width_m * width_m
    return [
        math.exp(-((px - cx) ** 2 + (py - cy) ** 2) / two_w2)
        for px, py in geometry.grid.pixels
    ]


def peak_index(image: list[float]) -> int:
    return max(range(len(image)), key=image.__getitem__)


def quiet_inverter(*args, **kwargs) -> Inverter:
    """Build an Inverter, suppressing the structural blind-corner warning.

    The warning is correct and wanted in production; it is noise here,
    except in the one test that asserts it fires.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", BlindPixelWarning)
        return Inverter(*args, **kwargs)


# The 3 x 3 system whose inverse was computed by hand (cofactor expansion).
# det(HAND_A) = 64 exactly, so every entry of the inverse is a dyadic
# rational and representable in binary floating point with no error at all.
HAND_A = [
    [4.0, 2.0, 0.0],
    [2.0, 5.0, 2.0],
    [0.0, 2.0, 5.0],
]
HAND_A_INV = [
    [21.0 / 64.0, -10.0 / 64.0, 4.0 / 64.0],
    [-10.0 / 64.0, 20.0 / 64.0, -8.0 / 64.0],
    [4.0 / 64.0, -8.0 / 64.0, 16.0 / 64.0],
]


class TestCholesky(unittest.TestCase):
    def test_factor_reconstructs_the_matrix(self):
        g = cholesky_factor(HAND_A)
        for i in range(3):
            for j in range(3):
                got = sum(g[i][k] * g[j][k] for k in range(3))
                self.assertAlmostEqual(got, HAND_A[i][j], places=13)

    def test_factor_is_lower_triangular(self):
        g = cholesky_factor(HAND_A)
        for i in range(3):
            for j in range(i + 1, 3):
                self.assertEqual(g[i][j], 0.0)

    def test_solve_matches_hand_computed_inverse(self):
        # B^-1 [1, 2, 3]^T = [13, 6, 36]/64, verified on paper by
        # multiplying back: A @ [13, 6, 36]/64 == [1, 2, 3] exactly.
        x = cholesky_solve(HAND_A, [1.0, 2.0, 3.0])
        for got, want in zip(x, (13.0 / 64.0, 6.0 / 64.0, 36.0 / 64.0)):
            self.assertAlmostEqual(got, want, places=14)

    def test_inverse_matches_hand_computed_inverse_entrywise(self):
        inv = cholesky_inverse(HAND_A)
        for i in range(3):
            for j in range(3):
                self.assertAlmostEqual(inv[i][j], HAND_A_INV[i][j], places=14)

    def test_inverse_is_exactly_symmetric(self):
        # cholesky_inverse symmetrizes, so this is ==, not almostEqual.
        inv = cholesky_inverse(HAND_A)
        for i in range(3):
            for j in range(3):
                self.assertEqual(inv[i][j], inv[j][i])

    def test_agrees_with_independent_gaussian_elimination(self):
        rng = random.Random(20261013)
        worst = 0.0
        for _ in range(40):
            n = rng.randint(2, 12)
            a = random_spd(n, rng)
            b = [rng.gauss(0.0, 1.0) for _ in range(n)]
            x_chol = cholesky_solve(a, b)
            x_gauss = gauss_solve(a, b)
            scale = max(max(abs(v) for v in x_gauss), 1e-30)
            worst = max(worst, max(abs(p - q) for p, q in zip(x_chol, x_gauss)) / scale)
        # Two different algorithms on well-conditioned SPD input should
        # agree to a few ulp; 1e-10 is loose enough not to be flaky and
        # tight enough that a real bug cannot hide under it.
        self.assertLess(worst, 1e-10, f"worst relative disagreement {worst:.3e}")

    def test_rejects_indefinite_matrix_loudly(self):
        with self.assertRaises(NotPositiveDefiniteError):
            cholesky_factor([[1.0, 2.0], [2.0, 1.0]])

    def test_rejects_singular_matrix_loudly(self):
        with self.assertRaises(NotPositiveDefiniteError):
            cholesky_factor([[1.0, 1.0], [1.0, 1.0]])

    def test_rejects_negative_diagonal(self):
        with self.assertRaises(NotPositiveDefiniteError):
            cholesky_factor([[-1.0, 0.0], [0.0, -1.0]])

    def test_rejects_asymmetric_matrix(self):
        # The transposed-index guard. This matrix is factorable if you
        # ignore the upper triangle, which is exactly the trap.
        with self.assertRaises(ValueError) as ctx:
            cholesky_factor([[4.0, 2.0], [3.0, 5.0]])
        self.assertIn("symmetric", str(ctx.exception))

    def test_every_refusal_is_catchable_as_a_linear_algebra_error(self):
        # A sweep driver wraps a configuration in one `except
        # LinearAlgebraError` and records the failure. That only works if
        # every refusal is one: the asymmetry refusal used to be a bare
        # ValueError, so such a driver crashed on it while catching the
        # Inverter's refusal for the very same condition.
        cases = [
            ([[4.0, 2.0], [3.0, 5.0]], AsymmetricMatrixError),
            ([[1.0, 0.0]], MalformedMatrixError),
            ([], MalformedMatrixError),
            ([[float("nan"), 0.0], [0.0, 1.0]], MalformedMatrixError),
            ([[1.0, 2.0], [2.0, 1.0]], NotPositiveDefiniteError),
        ]
        for matrix, expected in cases:
            with self.subTest(expected=expected.__name__):
                with self.assertRaises(LinearAlgebraError):
                    cholesky_factor(matrix)
                with self.assertRaises(expected):
                    cholesky_factor(matrix)

    def test_shape_and_finiteness_refusals_stay_value_errors(self):
        # Backward compatibility: these were plain ValueErrors before
        # MalformedMatrixError existed, and callers catch that.
        for matrix in ([[1.0, 0.0]], [], [[float("inf"), 0.0], [0.0, 1.0]]):
            with self.subTest(matrix=matrix):
                with self.assertRaises(ValueError):
                    cholesky_factor(matrix)

    def test_rejects_non_square_and_empty(self):
        with self.assertRaises(ValueError):
            cholesky_factor([[1.0, 0.0]])
        with self.assertRaises(ValueError):
            cholesky_factor([])

    def test_rejects_a_nan_wherever_it_sits(self):
        # REGRESSION. A NaN anywhere except a[0][0] used to defeat BOTH
        # guards at once. `max` keeps its running result unless the next
        # value compares greater and `nan > x` is always False, so the
        # scale scan skipped it and isfinite(scale) passed; `max(0.0,
        # nan)` is 0.0, so the symmetry scan read a NaN-against-number
        # pair as zero asymmetry. cholesky_factor then returned
        # [[2,0,0],[1,2,0],[0,1,2]] for
        #   [[4, nan, 0], [2, 5, nan], [0, 2, 5]]
        # with no exception at all — a clean factor of the LOWER TRIANGLE
        # MIRRORED, i.e. of HAND_A, a matrix the caller never passed.
        # Moving the same NaN to a[0][0] did raise, which is how we know
        # the guard was meant to fire and only worked by position.
        nan = float("nan")
        mirrored = [[4.0, nan, 0.0], [2.0, 5.0, nan], [0.0, 2.0, 5.0]]
        with self.assertRaises(MalformedMatrixError) as ctx:
            cholesky_factor(mirrored)
        self.assertIn("non-finite", str(ctx.exception))
        # The message must name WHICH entry, because "somewhere in a
        # 56 x 56 matrix" is not a diagnosis.
        self.assertIn("a[0][1]", str(ctx.exception))

        n = 4
        for i in range(n):
            for j in range(n):
                for bad in (nan, float("inf"), float("-inf")):
                    with self.subTest(i=i, j=j, bad=bad):
                        a = [
                            [4.0 if r == c else 1.0 for c in range(n)]
                            for r in range(n)
                        ]
                        a[i][j] = bad
                        with self.assertRaises(MalformedMatrixError):
                            cholesky_factor(a)

    def test_symmetry_tolerance_is_relative_to_the_pair_not_the_matrix(self):
        # REGRESSION. The tolerance used to be relative to the largest
        # entry anywhere in the matrix, so an asymmetry among the small
        # entries of a matrix with a large diagonal passed unnoticed:
        # worst = 1.0, scale = 1e9, and 1.0 > 1e-9 * 1e9 is False. The
        # factorisation then read the lower triangle and silently replaced
        # a[0][1] = 1.0 with a[1][0] = 2.0.
        with self.assertRaises(AsymmetricMatrixError):
            cholesky_factor([[1e9, 1.0], [2.0, 1e9]])

    def test_symmetry_tolerance_boundary(self):
        # Pins the guard's actual sensitivity, which no test used to
        # touch: test_rejects_asymmetric_matrix sits eight orders of
        # magnitude past the threshold. Tolerance is
        # symmetry_tol * max(|a_ij|, |a_ji|) plus a few ulp of the matrix
        # scale, so at this scale the floor is negligible and 1e-9
        # relative is the live edge.
        base, tol = 1.0, 1e-9
        for factor, should_raise in ((0.5, False), (0.9, False), (10.0, True)):
            with self.subTest(factor=factor):
                off = base * (1.0 + factor * tol)
                a = [[4.0, base, 0.0], [off, 5.0, 0.0], [0.0, 0.0, 6.0]]
                if should_raise:
                    with self.assertRaises(AsymmetricMatrixError):
                        cholesky_factor(a, symmetry_tol=tol)
                else:
                    cholesky_factor(a, symmetry_tol=tol)

    def test_absolute_floor_admits_honest_rounding_noise(self):
        # The other side of the per-pair tolerance: an entry that ought to
        # be zero emerges from a long accumulation at a few ulp of the
        # MATRIX scale, where relative agreement is meaningless. Demanding
        # it would reject arithmetic that is doing nothing wrong.
        eps = 2.220446049250313e-16
        a = [[1e6, 0.0, 0.0], [3.0 * eps * 1e6, 1e6, 0.0], [0.0, 0.0, 1e6]]
        cholesky_factor(a)

    def test_error_message_names_the_failing_pivot(self):
        with self.assertRaises(NotPositiveDefiniteError) as ctx:
            cholesky_factor([[2.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, -1.0]])
        self.assertIn("pivot 2", str(ctx.exception))


class TestSpatialPrior(unittest.TestCase):
    """The Toeplitz shortcut must equal the definition, entry for entry."""

    def test_matches_direct_formula_on_pixel_coordinates(self):
        # The grids are deliberately one square-pitched and one NOT.
        # `PixelGrid` rounds n_x and n_y independently, so Room(8, 6) at a
        # 0.7 m target gives an 11 x 9 grid at 0.7273 x 0.6667 m — and
        # until that case was added, the module's one axis-dependent
        # expression, hypot(dix * pitch_x, diy * pitch_y), was never
        # exercised with the two pitches different. A mutant that swapped
        # them passed all 52 tests. On the 4 x 3 / 0.5 m grid alone the
        # swap is a no-op (worst error exactly 0.0); on the 11 x 9 grid at
        # delta_c = 0.65 m it moves C by up to 0.032 against a unit
        # diagonal, which is 30% of the median differing entry and a
        # factor of 2.5 on the far-field tail — the prior stretched along
        # the wrong room axis.
        for room, pitch in ((Room(4.0, 3.0), 0.5), (Room(8.0, 6.0), 0.7)):
            grid = PixelGrid(room, pitch)
            with self.subTest(room=(room.width_m, room.height_m), pitch=pitch):
                sigma, delta_c = 1.7, 0.65
                prior = SpatialPrior(grid, sigma, delta_c)
                worst = 0.0
                for p, (px, py) in enumerate(grid.pixels):
                    for q, (qx, qy) in enumerate(grid.pixels):
                        want = sigma * sigma * math.exp(
                            -math.hypot(px - qx, py - qy) / delta_c
                        )
                        worst = max(worst, abs(prior.entry(p, q) - want))
                # Same expression, same inputs up to the grid being exactly
                # regular, so this is rounding-level, not approximation-level.
                self.assertLess(worst, 1e-12, f"worst entry error {worst:.3e}")

    def test_the_anisotropic_grid_really_is_anisotropic(self):
        # Guard on the guard above. If PixelGrid's rounding ever changes
        # and this grid comes out square-pitched, the test that pins the
        # kernel silently stops testing the axis pairing, exactly as it
        # did before. Fail here instead, where the cause is obvious.
        grid = PixelGrid(Room(8.0, 6.0), 0.7)
        self.assertEqual((grid.n_x, grid.n_y), (11, 9))
        self.assertNotAlmostEqual(grid.pitch_x_m, grid.pitch_y_m, places=3)

    def test_one_pixel_steps_differ_between_the_two_axes(self):
        # The cheapest possible direct check that dix pairs with pitch_x
        # and diy with pitch_y. A one-pixel step in x must cost
        # exp(-pitch_x / delta_c) and in y exp(-pitch_y / delta_c); the
        # wrong-axis mutant swaps precisely these two numbers, and no
        # permutation of the pixel index can fake it either, because the
        # two values are different.
        grid = PixelGrid(Room(8.0, 6.0), 0.7)
        delta_c = 0.5
        prior = SpatialPrior(grid, 1.0, delta_c)
        p = grid.pixel_index(3, 3)
        self.assertAlmostEqual(
            prior.entry(p, p + 1), math.exp(-grid.pitch_x_m / delta_c), places=14
        )
        self.assertAlmostEqual(
            prior.entry(p, p + grid.n_x),
            math.exp(-grid.pitch_y_m / delta_c),
            places=14,
        )
        self.assertNotAlmostEqual(
            prior.entry(p, p + 1), prior.entry(p, p + grid.n_x), places=3
        )

    def test_row_agrees_with_entry_and_with_dense(self):
        grid = PixelGrid(Room(4.0, 3.0), 0.5)
        prior = SpatialPrior(grid, 1.0, 0.4)
        dense = prior.dense()
        for p in range(len(grid.pixels)):
            row = prior.row(p)
            self.assertEqual(len(row), len(grid.pixels))
            self.assertEqual(row, dense[p])
            for q in range(len(grid.pixels)):
                self.assertEqual(row[q], prior.entry(p, q))

    def test_is_symmetric_and_has_sigma_squared_on_the_diagonal(self):
        grid = PixelGrid(Room(4.0, 3.0), 0.5)
        prior = SpatialPrior(grid, 2.5, 0.4)
        for p in range(len(grid.pixels)):
            self.assertAlmostEqual(prior.entry(p, p), 6.25, places=14)
            for q in range(len(grid.pixels)):
                self.assertEqual(prior.entry(p, q), prior.entry(q, p))

    def test_is_positive_definite(self):
        # exp(-r/delta_c) is the Matern-1/2 kernel, positive definite in
        # the plane. If this ever fails, the dual form's claim that
        # W C W^T + alpha I is SPD fails with it.
        grid = PixelGrid(Room(3.0, 2.0), 0.5)
        for delta_c in (0.2, 0.5, 1.0, 3.0):
            cholesky_factor(SpatialPrior(grid, 1.0, delta_c).dense())

    def test_decays_with_distance(self):
        grid = PixelGrid(Room(4.0, 1.0), 0.5)
        prior = SpatialPrior(grid, 1.0, 0.5)
        values = [prior.entry(0, q) for q in range(grid.n_x)]
        for a, b in zip(values, values[1:]):
            self.assertGreater(a, b)

    def test_rejects_bad_parameters(self):
        grid = PixelGrid(Room(2.0, 2.0), 1.0)
        for sigma, delta_c in ((0.0, 0.5), (-1.0, 0.5), (1.0, 0.0), (1.0, -0.5)):
            with self.assertRaises(ValueError):
                SpatialPrior(grid, sigma, delta_c)

    def test_rejects_non_finite_parameters_by_name(self):
        # REGRESSION. `value <= 0.0` is False for both NaN and +inf, so
        # the bare comparison passed them into C and they surfaced much
        # later as "matrix contains a non-finite entry" — an error about
        # the matrix, not about the argument. delta_c = inf was not
        # caught at all: exp(-d/inf) = 1 makes C all-ones and rank one,
        # and the inverter returned a meaningless image with no warning.
        grid = PixelGrid(Room(2.0, 2.0), 1.0)
        nan, inf = float("nan"), float("inf")
        for name, sigma, delta_c in (
            ("sigma", nan, 0.5),
            ("sigma", inf, 0.5),
            ("delta_c", 1.0, nan),
            ("delta_c", 1.0, inf),
        ):
            with self.subTest(name=name, sigma=sigma, delta_c=delta_c):
                with self.assertRaises(ValueError) as ctx:
                    SpatialPrior(grid, sigma, delta_c)
                self.assertIn(name, str(ctx.exception))

    def test_rejects_a_sigma_that_overflows_when_squared(self):
        # 1e200 is finite and positive, and its square is not. A sigma
        # sweep would otherwise have a silent cliff.
        grid = PixelGrid(Room(2.0, 2.0), 1.0)
        with self.assertRaises(ValueError) as ctx:
            SpatialPrior(grid, 1e200, 0.5)
        self.assertIn("sigma", str(ctx.exception))

    def test_entry_and_row_agree_about_what_an_invalid_index_is(self):
        # REGRESSION. entry() did no bounds checking, so a negative index
        # wrapped through Python's negative list indexing and returned a
        # plausible covariance for the wrong pair: on this 48-pixel grid
        # entry(-1, 0) returned 0.0008493257047191695 (pixel 41's value)
        # and entry(-5, 3) returned exp(-1) = 0.3678..., both thoroughly
        # believable. row(-1) raised IndexError on the same object. A
        # caller indexing relative to an image peak — a sub-pixel centroid
        # at a wall-adjacent peak — reaches ix - 1 as a matter of course.
        grid = PixelGrid(Room(4.0, 3.0), 0.5)
        n = len(grid.pixels)
        self.assertEqual(n, 48)
        prior = SpatialPrior(grid, 1.0, 0.5)
        for bad in (-1, -5, -n, n, n + 1, 2 * n):
            with self.subTest(index=bad):
                with self.assertRaises(IndexError):
                    prior.entry(bad, 0)
                with self.assertRaises(IndexError):
                    prior.entry(0, bad)
                with self.assertRaises(IndexError):
                    prior.row(bad)
        # And the valid extremes still work, so the guard is not off by one.
        self.assertGreater(prior.entry(0, n - 1), 0.0)
        self.assertEqual(len(prior.row(n - 1)), n)


class TestForwardProject(unittest.TestCase):
    def test_matches_the_dense_weight_matrix(self):
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        weights = geom.weight_matrix()
        rng = random.Random(11)
        image = [rng.uniform(0.0, 1.0) for _ in range(geom.n_pixels)]
        want = matvec(weights, image)
        got = forward_project(geom, image)
        self.assertEqual(len(got), geom.n_links)
        for a, b in zip(got, want):
            self.assertAlmostEqual(a, b, places=12)

    def test_rejects_wrong_length_image(self):
        geom = Geometry.perimeter(4, pitch_m=1.0, lambda_m=1.0)
        with self.assertRaises(ValueError):
            forward_project(geom, [0.0] * (geom.n_pixels + 1))


class TestDualEqualsPrimal(unittest.TestCase):
    """The most valuable test in the file.

    x = C W^T (W C W^T + alpha I)^-1 y   must equal
    x = (W^T W + alpha C^-1)^-1 W^T y

    on a geometry small enough that the P x P primal is computable. The
    primal is assembled here from the dense W, the dense C and this file's
    own Gaussian elimination; it shares no solver code with the module.
    """

    @staticmethod
    def small_geometry() -> Geometry:
        # 3 x 2 m, 4 nodes, 24 pixels at 0.5 m, lambda wide enough that
        # every pixel is covered (checked below) so the comparison is not
        # dominated by structurally blind pixels.
        room = Room(3.0, 2.0)
        return Geometry(room, perimeter_nodes(room, 4, 0.1), PixelGrid(room, 0.5), 0.8)

    def test_geometry_is_fully_covered(self):
        geom = self.small_geometry()
        self.assertEqual(geom.n_pixels, 24)
        self.assertEqual(geom.n_links, 12)
        self.assertGreater(min(geom.pixel_link_counts()), 0)

    def test_dual_equals_primal(self):
        geom = self.small_geometry()
        weights = geom.weight_matrix()
        n_p, n_l = geom.n_pixels, geom.n_links
        rng = random.Random(4242)

        settings = (
            (1.0, 1.0, 0.5),
            (0.05, 2.0, 1.5),
            (10.0, 0.5, 0.25),
            (0.5, 1.0, 1.0),
        )
        for alpha, sigma, delta_c in settings:
            with self.subTest(alpha=alpha, sigma=sigma, delta_c=delta_c):
                inverter = quiet_inverter(geom, alpha, sigma, delta_c)

                cov = SpatialPrior(geom.grid, sigma, delta_c).dense()
                cov_inv = gauss_inverse(cov)
                primal = [
                    [
                        sum(weights[l][i] * weights[l][j] for l in range(n_l))
                        + alpha * cov_inv[i][j]
                        for j in range(n_p)
                    ]
                    for i in range(n_p)
                ]

                y = [rng.gauss(0.0, 1.0) for _ in range(n_l)]
                wty = [
                    sum(weights[l][i] * y[l] for l in range(n_l)) for i in range(n_p)
                ]
                x_primal = gauss_solve(primal, wty)
                x_dual = inverter.solve(y)

                scale = max(max(abs(v) for v in x_primal), 1e-30)
                worst = max(abs(a - b) for a, b in zip(x_primal, x_dual)) / scale
                # Measured: 9e-16 to 7e-13 across these settings. The loose
                # end is alpha=0.05 with delta_c=1.5, where C is nearly
                # rank-deficient and the PRIMAL (which must invert it) is
                # the inaccurate side of the comparison, not the dual.
                self.assertLess(
                    worst,
                    1e-9,
                    f"dual and primal disagree by relative {worst:.3e}",
                )


class TestBlobRecovery(unittest.TestCase):
    """THE CORE TEST: forward-project a known blob, invert, find it again.

    Ground truth is a single Gaussian bump at a known pixel centre. y is
    built with the same W the inverter uses, so this isolates the
    inversion from any question about whether the ellipse model is a good
    description of radio physics — it is not testing physics, it is
    testing that the linear algebra round-trips.
    """

    # The two exhaustive sweeps below are the expensive part of this file,
    # so each is run once per class and the assertions read the result.
    # Running them per test method doubled the file's wall clock for no
    # extra coverage.
    coarse: list[tuple[int, int, int, int]]
    fine: list[tuple[int, int, int, int]]

    @classmethod
    def setUpClass(cls) -> None:
        cls.coarse_geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        cls.fine_geom = Geometry.perimeter(8, pitch_m=0.25, lambda_m=0.3)
        cls.coarse = cls.sweep(cls.coarse_geom, 0.4)
        cls.fine = cls.sweep(cls.fine_geom, 0.3)

    @staticmethod
    def sweep(geom: Geometry, width_m: float) -> list[tuple[int, int, int, int]]:
        """Place a blob on every pixel in turn; record where it comes back.

        Returns (ix, iy, peak_ix, peak_iy) per pixel, in canonical order.
        """
        grid = geom.grid
        inverter = quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=0.5)
        out = []
        for iy in range(grid.n_y):
            for ix in range(grid.n_x):
                cx, cy = grid.pixels[iy * grid.n_x + ix]
                image = inverter.solve(
                    forward_project(geom, gaussian_blob(geom, cx, cy, width_m))
                )
                k = peak_index(image)
                out.append((ix, iy, k % grid.n_x, k // grid.n_x))
        return out

    def test_every_pixel_of_the_standard_deployment_is_recovered(self):
        # 8 nodes, 0.5 m pitch: the default deployment. No exclusions at
        # all, including the two structurally blind corners.
        failures = [
            ((ix, iy), (kx, ky))
            for ix, iy, kx, ky in self.coarse
            if max(abs(kx - ix), abs(ky - iy)) > 1
        ]
        self.assertEqual(len(self.coarse), self.coarse_geom.n_pixels)
        self.assertEqual(
            failures, [], f"{len(failures)} of {len(self.coarse)} blobs misplaced"
        )

    def test_peak_is_usually_exactly_right_not_merely_within_one_pixel(self):
        # "Within one pixel" is a weak bar on a 0.5 m grid. MEASURED by
        # driving this class's own sweep: the argmax is the exact blob
        # pixel for 184 of 192 positions (95.8%). The eight misses are all
        # on the outer ring and all off by exactly one pixel —
        #   (14,0)->(14,1) (15,0)->(15,1) (15,2)->(14,2) (15,3)->(14,3)
        #   (0,8)->(1,8)   (0,9)->(1,9)   (0,11)->(0,10) (1,11)->(1,10)
        # which is the same wall effect the 0.25 m grid shows in
        # test_fine_grid_wall_ring_is_the_known_failure, one pitch coarser.
        # An earlier comment here claimed 192/192; it did not reproduce.
        # The 90% floor stands, now justified by a measured 95.8% rather
        # than by an imagined 100%.
        exact = sum(1 for ix, iy, kx, ky in self.coarse if (ix, iy) == (kx, ky))
        self.assertGreaterEqual(
            exact, int(0.90 * len(self.coarse)), f"{exact}/{len(self.coarse)} exact"
        )
        off_by_one = [
            (ix, iy, kx, ky)
            for ix, iy, kx, ky in self.coarse
            if (ix, iy) != (kx, ky)
        ]
        self.assertTrue(
            all(max(abs(kx - ix), abs(ky - iy)) == 1 for ix, iy, kx, ky in off_by_one),
            f"a miss moved by more than one pixel: {off_by_one}",
        )
        # Every miss on the outer ring. If an interior pixel ever joins
        # them that is a different bug from the wall effect.
        grid = self.coarse_geom.grid
        interior = [
            (ix, iy)
            for ix, iy, _, _ in off_by_one
            if 0 < ix < grid.n_x - 1 and 0 < iy < grid.n_y - 1
        ]
        self.assertEqual(interior, [], f"interior misses: {interior}")

    def test_fine_grid_interior_is_recovered(self):
        # 0.25 m pitch. MEASURED: the interior (every pixel at least 0.5 m
        # from a wall) is recovered exactly, 560/560. The outermost ring is
        # NOT — see test_fine_grid_wall_ring_is_the_known_failure below,
        # which documents it rather than hiding it behind a loose bound.
        geom = self.fine_geom
        grid = geom.grid
        checked = 0
        failures = []
        for ix, iy, kx, ky in self.fine:
            cx, cy = grid.pixels[iy * grid.n_x + ix]
            margin = min(cx, cy, geom.room.width_m - cx, geom.room.height_m - cy)
            if margin < 0.5:
                continue
            checked += 1
            if max(abs(kx - ix), abs(ky - iy)) > 1:
                failures.append((ix, iy))
        self.assertGreater(checked, 500)
        self.assertEqual(failures, [], f"{len(failures)}/{checked} interior misplaced")

    def test_fine_grid_wall_ring_is_the_known_failure(self):
        # Pinned so it is a documented property and not a surprise later.
        # MEASURED at 0.25 m / lambda 0.3 / alpha 1.0: 14 of 768 positions
        # land more than one pixel away, every one of them within 0.38 m of
        # a wall and clustered on the two corners that `geometry`'s
        # pixel_link_counts documents as structurally blind.
        geom = self.fine_geom
        grid = geom.grid
        margins = []
        for ix, iy, kx, ky in self.fine:
            if max(abs(kx - ix), abs(ky - iy)) <= 1:
                continue
            cx, cy = grid.pixels[iy * grid.n_x + ix]
            margins.append(
                min(cx, cy, geom.room.width_m - cx, geom.room.height_m - cy)
            )
        self.assertLessEqual(len(margins), 20, "more misplacements than measured")
        # THIS is the assertion doing the work: the wall ring is a pinned,
        # documented property, so it must still be there. If it vanishes,
        # something changed the estimator and the rest of this file's
        # measured numbers are stale too.
        self.assertTrue(margins, "expected the known wall-ring failures to still exist")
        # Tightened from 0.5 m, where it was vacuous: the interior test
        # above skips exactly the pixels with margin < 0.5 and demands
        # zero misplacements in the rest, so any failure at margin >= 0.5
        # fails that test first and this one could never be the assertion
        # that caught it. MEASURED: every wall-ring failure sits within
        # 0.375 m of a wall, so 0.4 bites before the interior boundary.
        self.assertLess(
            max(margins),
            0.4,
            "a misplacement appeared away from the wall; that is a new bug",
        )

    def test_recovery_survives_noise_on_the_link_vector(self):
        # Gaussian noise at 20% of the RMS of y. MEASURED median peak error
        # 0.00 m, p90 0.00 m over 192 positions, so asserting a median of
        # zero and a p90 of one pixel is honest rather than generous.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        grid = geom.grid
        inverter = quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=0.5)
        rng = random.Random(99)

        clean = [
            forward_project(geom, gaussian_blob(geom, cx, cy))
            for cx, cy in grid.pixels
        ]
        rms = math.sqrt(
            sum(sum(v * v for v in y) / len(y) for y in clean) / len(clean)
        )
        errors = []
        for p, (cx, cy) in enumerate(grid.pixels):
            noisy = [v + rng.gauss(0.0, 0.2 * rms) for v in clean[p]]
            k = peak_index(inverter.solve(noisy))
            errors.append(math.hypot(grid.pixels[k][0] - cx, grid.pixels[k][1] - cy))
        errors.sort()
        self.assertEqual(errors[len(errors) // 2], 0.0)
        self.assertLessEqual(errors[int(0.9 * len(errors))], 0.5)

    def test_noise_degradation_curve_is_pinned_across_seeds(self):
        # The module docstring used to claim recovery to within one pixel
        # was universal "with Gaussian noise added to y at up to 50% of
        # its RMS", and nothing tested above 20% with a single seed. The
        # claim is false. MEASURED here, seeds 0-4, 192 blob positions
        # each, counting positions whose argmax is more than one pixel
        # (Chebyshev index distance) from the true pixel:
        #
        #     noise   misses per seed 0-4      worst displacement
        #      20%    1, 0, 0, 0, 1            1.12 m
        #      30%    1, 0, 1, 0, 1            1.12 m
        #      50%    1, 1, 1, 2, 1            2.00 m
        #
        # So it is not even universal at 20%; seed 99, which the test
        # above happens to use, gives 0. Every miss sits on a structurally
        # blind corner. This test pins the curve as a bounded known
        # property, the way test_fine_grid_wall_ring_is_the_known_failure
        # pins the wall ring, because the shape of this curve between 20%
        # and 50% is what track.py's abstention threshold gets sized
        # against. The bounds are one above the measured maxima.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        grid = geom.grid
        n_x = grid.n_x
        inverter = quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=0.5)
        clean = [
            forward_project(geom, gaussian_blob(geom, cx, cy, 0.4))
            for cx, cy in grid.pixels
        ]
        rms = math.sqrt(
            sum(sum(v * v for v in y) / len(y) for y in clean) / len(clean)
        )
        budget = {0.2: (2, 1.5), 0.3: (2, 1.5), 0.5: (3, 2.5)}
        for fraction, (max_misses, max_error_m) in budget.items():
            for seed in range(5):
                with self.subTest(noise=fraction, seed=seed):
                    rng = random.Random(seed)
                    misses, worst = 0, 0.0
                    for p, (cx, cy) in enumerate(grid.pixels):
                        noisy = [v + rng.gauss(0.0, fraction * rms) for v in clean[p]]
                        k = peak_index(inverter.solve(noisy))
                        if max(abs(k % n_x - p % n_x), abs(k // n_x - p // n_x)) > 1:
                            misses += 1
                            worst = max(
                                worst,
                                math.hypot(
                                    grid.pixels[k][0] - cx, grid.pixels[k][1] - cy
                                ),
                            )
                    self.assertLessEqual(
                        misses, max_misses, f"{misses} misplaced at {fraction:.0%}"
                    )
                    self.assertLessEqual(
                        worst, max_error_m, f"worst {worst:.2f} m at {fraction:.0%}"
                    )

    def test_every_alpha_in_the_sweep_range_recovers_every_pixel(self):
        # The half of the docstring claim that DOES hold, now covered.
        # MEASURED: 0 of 192 positions outside one pixel at alpha in
        # {0.1, 0.2, 0.5, 1, 2, 5, 10, 20}. The alpha axis is flat for
        # clean y, which is why the sweep in evaluate_localize has to vary
        # the noise-to-signal ratio to learn anything about alpha.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        grid = geom.grid
        n_x = grid.n_x
        clean = [
            forward_project(geom, gaussian_blob(geom, cx, cy, 0.4))
            for cx, cy in grid.pixels
        ]
        for alpha in (0.1, 1.0, 20.0):
            with self.subTest(alpha=alpha):
                inverter = quiet_inverter(geom, alpha=alpha, sigma=1.0, delta_c=0.5)
                misses = [
                    p
                    for p in range(len(clean))
                    for k in (peak_index(inverter.solve(clean[p])),)
                    if max(abs(k % n_x - p % n_x), abs(k // n_x - p // n_x)) > 1
                ]
                self.assertEqual(misses, [], f"{len(misses)}/192 at alpha={alpha}")

    def test_two_blobs_give_two_local_maxima(self):
        # A sanity check that the prior is not simply collapsing everything
        # onto the room centre, which is the failure mode that makes a
        # tracker look good for the wrong reason.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        grid = geom.grid
        inverter = quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=0.5)
        left = gaussian_blob(geom, 2.0, 3.0, 0.4)
        right = gaussian_blob(geom, 6.0, 3.0, 0.4)
        image = inverter.solve(
            forward_project(geom, [a + b for a, b in zip(left, right)])
        )
        centre = grid.nearest_index(*geom.room.center)
        peak_left = max(
            range(len(image)), key=lambda p: image[p] if grid.pixels[p][0] < 4.0 else -1e30
        )
        peak_right = max(
            range(len(image)), key=lambda p: image[p] if grid.pixels[p][0] > 4.0 else -1e30
        )
        self.assertLess(abs(grid.pixels[peak_left][0] - 2.0), 0.6)
        self.assertLess(abs(grid.pixels[peak_right][0] - 6.0), 0.6)
        self.assertGreater(image[peak_left], image[centre])
        self.assertGreater(image[peak_right], image[centre])


class TestRegularizationSmooths(unittest.TestCase):
    def test_increasing_alpha_lowers_peak_to_rms(self):
        # Peak-to-RMS is the sharpness measure: it is scale invariant, so
        # it cannot be fooled by alpha simply shrinking the whole image
        # (which it also does — see the peak column in the probe output).
        # MEASURED over six decades of alpha: 4.480, 4.468, 4.364, 3.914,
        # 3.151, 2.394, 2.234. Strictly decreasing at every step.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        y = forward_project(geom, gaussian_blob(geom, 4.0, 3.0))
        ratios = []
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 1e4):
            image = quiet_inverter(geom, alpha=alpha, sigma=1.0, delta_c=0.5).solve(y)
            rms = math.sqrt(sum(v * v for v in image) / len(image))
            ratios.append(max(image) / rms)
        for a, b in zip(ratios, ratios[1:]):
            self.assertLess(b, a, f"peak/rms not monotone in alpha: {ratios}")
        self.assertLess(ratios[-1], 0.6 * ratios[0])

    def test_increasing_alpha_shrinks_the_image(self):
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        y = forward_project(geom, gaussian_blob(geom, 4.0, 3.0))
        peaks = [
            max(quiet_inverter(geom, alpha=a, sigma=1.0, delta_c=0.5).solve(y))
            for a in (0.1, 1.0, 10.0, 100.0, 1000.0)
        ]
        for a, b in zip(peaks, peaks[1:]):
            self.assertLess(b, a, f"peak not monotone in alpha: {peaks}")

    def test_correlation_length_has_an_optimum_rather_than_a_direction(self):
        # This test originally asserted that a larger delta_c always
        # smears the image. IT DOES NOT, and the first run said so.
        #
        # MEASURED: concentration of |x| within 1 m of the true blob, at
        # alpha = 1.0 on the 0.5 m grid, against delta_c =
        #   0.05  0.10  0.25  0.40  0.50  0.75  1.00  1.50  2.00  4.00  8.00
        #   .329  .333  .393  .429  .434  .436  .434  .433  .433  .427  .408
        # — unimodal, peaking around one to two pixel pitches. The reason
        # is that delta_c is not a smoothing strength, it is the prior's
        # correlation LENGTH, and it competes with two other length
        # scales. Below the pixel pitch, C is nearly diagonal, the prior
        # stops coupling neighbours at all, and the image goes ringy.
        # Above the room scale, C approaches rank one and every image
        # becomes the same smear. In between, the resolution is set by the
        # ellipse width and the link geometry, not by delta_c, which is
        # why the middle of the curve is almost flat.
        #
        # So the honest assertion is unimodality with the optimum near the
        # pixel pitch, not monotonicity in either direction.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        y = forward_project(geom, gaussian_blob(geom, 4.0, 3.0))

        def concentration(delta_c: float) -> float:
            image = quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=delta_c).solve(y)
            total = sum(abs(v) for v in image)
            inside = sum(
                abs(v)
                for v, (px, py) in zip(image, geom.grid.pixels)
                if math.hypot(px - 4.0, py - 3.0) <= 1.0
            )
            return inside / total

        sweep = [0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0]
        values = [concentration(d) for d in sweep]
        best = sweep[max(range(len(sweep)), key=values.__getitem__)]

        self.assertGreaterEqual(best, 0.4, f"optimum at delta_c={best}: {values}")
        self.assertLessEqual(best, 1.5, f"optimum at delta_c={best}: {values}")
        # Both ends must be genuinely worse, or "optimum" means nothing.
        self.assertLess(values[0], max(values) - 0.05, f"{values}")
        self.assertLess(values[-1], max(values) - 0.01, f"{values}")

    def test_long_correlation_length_degrades_conditioning(self):
        # The other half of why delta_c cannot simply be turned up: C
        # approaches rank one, and the dual system inherits it. MEASURED
        # back-check residual max |B B^-1 - I| on the 0.5 m grid:
        #   delta_c 0.25 -> 1.1e-15,  1.0 -> 1.7e-14,  8.0 -> 3.0e-13.
        # Still far from dangerous, but moving two orders of magnitude
        # over the sweep, and in the direction that eventually bites.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        residuals = [
            quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=d).inverse_residual
            for d in (0.25, 1.0, 8.0)
        ]
        for a, b in zip(residuals, residuals[1:]):
            self.assertGreater(b, a, f"residual not growing with delta_c: {residuals}")
        self.assertLess(max(residuals), 1e-9)

    def test_the_conditioning_trend_stops_past_the_room_scale(self):
        # The limit of the claim above, which IllConditionedError's advice
        # used to state as though it held everywhere. It does not: the
        # residual plateaus once C is effectively rank one, so "lower
        # delta_c" is not a general remedy for a conditioning complaint.
        # MEASURED on the 0.5 m grid at alpha = 1: 8 m -> 3.0e-13,
        # 50 m -> 1.9e-13, 500 m -> 3.9e-13. Flat and non-monotone, all
        # within a factor of 2.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        tail = [
            quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=d).inverse_residual
            for d in (8.0, 50.0, 500.0)
        ]
        self.assertFalse(
            all(a < b for a, b in zip(tail, tail[1:])),
            f"residual still monotone past the room scale: {tail}",
        )
        self.assertLess(max(tail) / min(tail), 5.0, f"not a plateau: {tail}")


class TestSymmetry(unittest.TestCase):
    """The deployment's symmetries must survive the inversion.

    A 6 x 6 m room with 4 nodes lands one node on each corner of the inset
    square, so the deployment is invariant under three distinct symmetries:
    the vertical mirror, the horizontal mirror, and the diagonal flip. The
    pixel grid at a 1 m pitch is invariant under all three too. So a
    ground-truth image with those symmetries must invert to an image with
    those symmetries.

    BE PRECISE ABOUT WHAT THIS CATCHES, because this class used to claim
    more than it delivers. It catches an estimator that breaks a symmetry
    the deployment has — a weight, a prior or a solve that treats one half
    of the room differently from the other. It does NOT catch a
    transposed pixel index. VERIFIED by mutation: a `SpatialPrior` that
    reads the pixel index as ix*n_y+iy instead of iy*n_x+ix leaves all
    three tests here passing, because on a deployment invariant under the
    diagonal flip the transposition IS that flip, so it commutes straight
    through the estimator and the image comes back symmetric anyway. The
    same mutation leaves `Inverter.asymmetry` at 2e-16 and 5e-16, so that
    guard is silent too.

    What does catch it: `test_matches_direct_formula_on_pixel_coordinates`
    on a grid whose two pitches differ (no permutation can fake a kernel
    that is anisotropic the right way round), and the exhaustive blob
    sweep in `TestBlobRecovery`. Nine tests in those two classes fail
    under the mutation; none of the three here do.
    """

    @staticmethod
    def square_geometry() -> Geometry:
        return Geometry.perimeter(4, room=Room(6.0, 6.0), pitch_m=1.0, lambda_m=2.0)

    def test_deployment_really_is_symmetric(self):
        geom = self.square_geometry()
        self.assertEqual(geom.grid.n_x, geom.grid.n_y)
        self.assertGreater(min(geom.pixel_link_counts()), 0)
        corners = {(0.1, 0.1), (5.9, 0.1), (5.9, 5.9), (0.1, 5.9)}
        self.assertEqual({(round(n.x, 6), round(n.y, 6)) for n in geom.nodes}, corners)

    def test_centred_blob_gives_a_fully_symmetric_image(self):
        geom = self.square_geometry()
        inverter = quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=0.8)
        image = inverter.solve(
            forward_project(geom, gaussian_blob(geom, 3.0, 3.0, 1.0))
        )
        n_x, n_y = geom.grid.n_x, geom.grid.n_y
        scale = max(abs(v) for v in image)
        self.assertGreater(scale, 1e-6, "image is empty; test proves nothing")

        for iy in range(n_y):
            for ix in range(n_x):
                here = image[iy * n_x + ix]
                self.assertAlmostEqual(
                    here, image[iy * n_x + (n_x - 1 - ix)], delta=1e-12 + 1e-9 * scale
                )
                self.assertAlmostEqual(
                    here, image[(n_y - 1 - iy) * n_x + ix], delta=1e-12 + 1e-9 * scale
                )
                # The diagonal flip. NOT a transposed-index detector — on
                # a deployment this symmetric the transposition is the
                # flip, so it commutes through; see the class docstring.
                self.assertAlmostEqual(
                    here, image[ix * n_x + iy], delta=1e-12 + 1e-9 * scale
                )

    def test_mirrored_blob_gives_a_mirrored_image(self):
        geom = self.square_geometry()
        inverter = quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=0.8)
        left = inverter.solve(forward_project(geom, gaussian_blob(geom, 1.5, 2.5, 0.6)))
        right = inverter.solve(forward_project(geom, gaussian_blob(geom, 4.5, 2.5, 0.6)))
        n_x, n_y = geom.grid.n_x, geom.grid.n_y
        scale = max(abs(v) for v in left)
        self.assertGreater(scale, 1e-6)
        for iy in range(n_y):
            for ix in range(n_x):
                self.assertAlmostEqual(
                    left[iy * n_x + ix],
                    right[iy * n_x + (n_x - 1 - ix)],
                    delta=1e-12 + 1e-9 * scale,
                )

    def test_asymmetric_room_does_not_accidentally_pass(self):
        # Guard on the guard: on an 8 x 6 room the diagonal flip is NOT a
        # symmetry, so an implementation that returned a symmetric image
        # regardless would be caught here.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        inverter = quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=0.5)
        image = inverter.solve(forward_project(geom, gaussian_blob(geom, 2.0, 1.0)))
        n_x = geom.grid.n_x
        mirrored = max(
            abs(image[iy * n_x + ix] - image[iy * n_x + (n_x - 1 - ix)])
            for iy in range(geom.grid.n_y)
            for ix in range(n_x)
        )
        self.assertGreater(mirrored, 1e-3 * max(abs(v) for v in image))


class TestInverterContract(unittest.TestCase):
    def test_solve_returns_one_value_per_pixel(self):
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        inverter = quiet_inverter(geom)
        out = inverter.solve([0.0] * geom.n_links)
        self.assertEqual(len(out), geom.n_pixels)
        self.assertEqual(len(out), len(geom.grid.pixels))

    def test_projection_has_shape_pixels_by_independent_links(self):
        # The projection's second axis is the number of DISTINCT rows of
        # W, not the link count: reciprocal links have identical rows and
        # are collapsed. `solve` still takes y in full canonical link
        # order, which is the shape `measure` emits.
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        inverter = quiet_inverter(geom)
        self.assertEqual(len(inverter.projection), geom.n_pixels)
        for row in inverter.projection:
            self.assertEqual(len(row), inverter.n_independent_links)
        self.assertEqual(inverter.n_links, geom.n_links)
        self.assertEqual(inverter.n_independent_links, geom.n_links // 2)

    def test_solve_is_linear(self):
        # The projection is a fixed matrix, so solve must be exactly
        # linear. Any hidden per-call state shows up here immediately.
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        inverter = quiet_inverter(geom)
        rng = random.Random(5)
        y1 = [rng.gauss(0.0, 1.0) for _ in range(geom.n_links)]
        y2 = [rng.gauss(0.0, 1.0) for _ in range(geom.n_links)]
        a, b = 2.5, -1.75
        combined = inverter.solve([a * u + b * v for u, v in zip(y1, y2)])
        x1, x2 = inverter.solve(y1), inverter.solve(y2)
        separate = [a * u + b * v for u, v in zip(x1, x2)]
        scale = max(abs(v) for v in combined)
        for p, q in zip(combined, separate):
            self.assertAlmostEqual(p, q, delta=1e-12 * max(scale, 1.0))

    def test_sigma_is_redundant_with_alpha(self):
        # C = sigma^2 C0 substituted into the dual form leaves
        # x = C0 W^T (W C0 W^T + (alpha/sigma^2) I)^-1 y, so holding
        # alpha/sigma^2 fixed must reproduce the image exactly.
        # MEASURED: 0.0 relative difference for sigma in {0.25, 0.5, 2},
        # 1.2e-14 at sigma = 10. This is why the evaluation sweep should
        # fix sigma = 1 and vary alpha alone.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        y = forward_project(geom, gaussian_blob(geom, 2.5, 4.0))
        reference = quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=0.5).solve(y)
        scale = max(abs(v) for v in reference)

        for sigma in (0.25, 0.5, 2.0, 3.0, 10.0):
            with self.subTest(sigma=sigma):
                image = quiet_inverter(
                    geom, alpha=1.0 * sigma * sigma, sigma=sigma, delta_c=0.5
                ).solve(y)
                worst = max(abs(a - b) for a, b in zip(reference, image)) / scale
                self.assertLess(worst, 1e-12, f"relative difference {worst:.3e}")

        # Control: if the ratio is NOT held fixed the image must change,
        # otherwise the test above would pass on a solver that ignores
        # both parameters.
        off_ratio = quiet_inverter(geom, alpha=1.0, sigma=3.0, delta_c=0.5).solve(y)
        changed = max(abs(a - b) for a, b in zip(reference, off_ratio)) / scale
        self.assertGreater(changed, 1e-3, "sigma had no effect at all")

    def test_image_is_invariant_to_the_scale_of_y(self):
        # Exact linearity means the peak cannot move when y is rescaled.
        # This is what lets `measure` hand over raw variance changes of
        # order 1e-4 with no normalisation step, and lets alpha be tuned
        # independently of the measurement's units.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        inverter = quiet_inverter(geom, alpha=1.0, sigma=1.0, delta_c=0.5)
        y = forward_project(geom, gaussian_blob(geom, 2.5, 4.0))
        base = inverter.solve(y)
        base_peak = peak_index(base)
        base_scale = max(abs(v) for v in base)

        for k in (1e-6, 1e-3, 1e3, 1e6):
            with self.subTest(scale=k):
                image = inverter.solve([k * v for v in y])
                self.assertEqual(peak_index(image), base_peak)
                worst = max(
                    abs(k * a - b) for a, b in zip(base, image)
                ) / (k * base_scale)
                self.assertLess(worst, 1e-12, f"relative difference {worst:.3e}")

    def test_zero_measurement_gives_zero_image(self):
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        image = quiet_inverter(geom).solve([0.0] * geom.n_links)
        self.assertEqual(set(image), {0.0})

    def test_repeated_solves_are_identical(self):
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        inverter = quiet_inverter(geom)
        y = forward_project(geom, gaussian_blob(geom, 3.0, 2.0))
        self.assertEqual(inverter.solve(y), inverter.solve(y))

    def test_rejects_wrong_length_y(self):
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        inverter = quiet_inverter(geom)
        for n in (geom.n_links - 1, geom.n_links + 1, 0):
            with self.assertRaises(ValueError):
                inverter.solve([0.0] * n)

    def test_rejects_non_finite_y(self):
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        inverter = quiet_inverter(geom)
        for bad in (float("nan"), float("inf"), float("-inf")):
            y = [0.0] * geom.n_links
            y[3] = bad
            with self.assertRaises(ValueError):
                inverter.solve(y)

    def test_rejects_non_positive_alpha(self):
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        for alpha in (0.0, -1.0):
            with self.assertRaises(ValueError):
                quiet_inverter(geom, alpha=alpha)

    def test_rejects_non_finite_constructor_parameters_by_name(self):
        # REGRESSION. `alpha <= 0.0` is False for NaN and for +inf, so
        # both used to sail through the constructor and surface ~0.2 s
        # later from inside cholesky_factor as "matrix contains a
        # non-finite entry" — an error naming the matrix, not the
        # argument. Compare alpha=-1.0, which always said "alpha must be
        # positive". The message must name the parameter, which is the
        # whole point of catching it here.
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        nan, inf = float("nan"), float("inf")
        cases = [
            ("alpha", {"alpha": nan}),
            ("alpha", {"alpha": inf}),
            ("sigma", {"sigma": nan}),
            ("sigma", {"sigma": inf}),
            ("sigma", {"sigma": 1e200}),
            ("delta_c", {"delta_c": nan}),
            ("delta_c", {"delta_c": inf}),
        ]
        for name, kwargs in cases:
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError) as ctx:
                    quiet_inverter(geom, **kwargs)
                message = str(ctx.exception)
                self.assertIn(name, message)
                self.assertNotIn("non-finite entry", message)

    def test_rejects_bad_prior_parameters(self):
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        with self.assertRaises(ValueError):
            quiet_inverter(geom, sigma=0.0)
        with self.assertRaises(ValueError):
            quiet_inverter(geom, delta_c=0.0)

    def test_forward_method_matches_module_function(self):
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        inverter = quiet_inverter(geom)
        image = gaussian_blob(geom, 3.0, 2.0)
        self.assertEqual(inverter.forward(image), forward_project(geom, image))

    def test_warns_about_structurally_blind_pixels(self):
        # The default 8-node 0.5 m deployment has two blind corners, which
        # geometry.pixel_link_counts documents. The warning must fire.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        self.assertGreater(sum(1 for c in geom.pixel_link_counts() if c == 0), 0)
        with self.assertWarns(BlindPixelWarning):
            inverter = Inverter(geom, alpha=1.0)
        self.assertEqual(
            len(inverter.blind_pixels),
            sum(1 for c in geom.pixel_link_counts() if c == 0),
        )

    def test_does_not_warn_when_every_pixel_is_covered(self):
        # lambda = 0.75 m closes the corner holes at a 0.5 m pitch, per the
        # coverage table in geometry.pixel_link_counts.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.75)
        self.assertGreater(min(geom.pixel_link_counts()), 0)
        with warnings.catch_warnings():
            warnings.simplefilter("error", BlindPixelWarning)
            Inverter(geom, alpha=1.0)

    def test_dual_system_is_well_conditioned_and_symmetric(self):
        # The diagnostics the Inverter records about its own numerics.
        # MEASURED: asymmetry 4e-16 to 9e-16, inverse residual 1e-15 to
        # 3e-15 across the three standard deployments.
        for n_nodes, pitch in ((8, 0.5), (8, 0.25), (12, 0.5)):
            with self.subTest(n_nodes=n_nodes, pitch=pitch):
                geom = Geometry.perimeter(n_nodes, pitch_m=pitch, lambda_m=0.3)
                inverter = quiet_inverter(geom, alpha=1.0)
                self.assertLess(inverter.asymmetry, 1e-12)
                self.assertLess(inverter.inverse_residual, 1e-9)
                self.assertGreater(inverter.mean_diag_a, 0.0)
                self.assertGreater(inverter.min_diag_a, 0.0)

    def test_alpha_appears_on_the_diagonal_of_the_dual_system(self):
        # Pins the ridge down where it belongs: V C V^T + alpha I, not
        # alpha scaled by anything. Deduplication must not have quietly
        # rescaled alpha — the whole point of the sqrt(m) row weighting is
        # that the ridge keeps its meaning.
        geom = Geometry.perimeter(4, pitch_m=1.0, lambda_m=1.5)
        i1 = quiet_inverter(geom, alpha=1.0)
        a1, a2 = i1._a_plus_alpha, quiet_inverter(geom, alpha=5.0)._a_plus_alpha
        n = i1.n_independent_links
        self.assertEqual(len(a1), n)
        for i in range(n):
            for j in range(n):
                expected = a1[i][j] + (4.0 if i == j else 0.0)
                self.assertAlmostEqual(a2[i][j], expected, delta=1e-9)


class TestReciprocalLinkDeduplication(unittest.TestCase):
    """Reciprocal links are one row of W, and the module must treat them so.

    `geometry.Link` is ordered, but `excess_path_m` and `length_m` are
    both symmetric in (tx, rx), so W's rows for (i, j) and (j, i) are
    identical element for element. Keeping both made W C W^T rank
    deficient by L/2 — PSD, never PD, with the ridge as the only thing
    holding the factorisation up — and cost 4x in the dominant P*L^2 term.

    The collapse has to be exactly information-preserving, which is the
    point of these tests. The strongest evidence is elsewhere:
    `TestDualEqualsPrimal` builds the primal from the FULL dense W, both
    directions present, and still agrees to rounding.
    """

    def test_reciprocal_links_really_do_share_a_row_of_w(self):
        # The premise. If geometry ever makes the two directions differ —
        # a direction-dependent weight, say — the deduplication must stop
        # merging them, and this test is where that shows up first.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        weights = geom.weight_matrix()
        pairs = 0
        for i, link in enumerate(geom.links):
            j = geom.link_index(link.rx_id, link.tx_id)
            if j <= i:
                continue
            pairs += 1
            self.assertEqual(weights[i], weights[j], f"link {i} vs {j}")
        self.assertEqual(pairs, geom.n_links // 2)

    def test_the_inverter_collapses_them(self):
        for n_nodes, pitch in ((6, 0.5), (8, 0.5), (8, 0.25), (12, 0.5)):
            with self.subTest(n_nodes=n_nodes, pitch=pitch):
                geom = Geometry.perimeter(n_nodes, pitch_m=pitch, lambda_m=0.3)
                inverter = quiet_inverter(geom, alpha=1.0)
                self.assertEqual(inverter.n_independent_links, geom.n_links // 2)
                # Every group is exactly one reciprocal pair, and the
                # groups partition the links.
                seen = sorted(i for g in inverter.link_groups for i in g)
                self.assertEqual(seen, list(range(geom.n_links)))
                for group in inverter.link_groups:
                    self.assertEqual(len(group), 2)
                    a, b = (geom.links[i] for i in group)
                    self.assertEqual((a.tx_id, a.rx_id), (b.rx_id, b.tx_id))

    def test_the_collapsed_system_is_positive_definite_without_any_ridge(self):
        # The structural claim. Before the collapse, W C W^T had L/2
        # exactly-zero eigenvalues and this factorisation was impossible;
        # MEASURED now, it succeeds with no ridge term at all at 6, 8 and
        # 12 nodes and both pitches. That is what makes alpha a genuine
        # regulariser rather than the only thing holding the matrix up.
        for n_nodes, pitch in ((6, 0.5), (8, 0.5), (12, 0.5)):
            with self.subTest(n_nodes=n_nodes, pitch=pitch):
                inverter = quiet_inverter(
                    Geometry.perimeter(n_nodes, pitch_m=pitch, lambda_m=0.3),
                    alpha=1.0,
                )
                ridgeless = [list(row) for row in inverter._a_plus_alpha]
                for i in range(len(ridgeless)):
                    ridgeless[i][i] -= 1.0
                cholesky_factor(ridgeless)

    def test_a_vanishing_ridge_no_longer_breaks_the_factorisation(self):
        # REGRESSION, and the clearest symptom of the old rank
        # deficiency: Inverter(8 nodes, alpha=1e-15) used to raise
        # NotPositiveDefiniteError, whose docstring sent the reader off to
        # hunt a transposed index that was not there. MEASURED now: the
        # same deployment survives alpha = 1e-300.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        for alpha in (1e-15, 1e-100, 1e-300):
            with self.subTest(alpha=alpha):
                quiet_inverter(geom, alpha=alpha, check_inverse=False)

    def test_only_the_sum_of_the_two_directions_reaches_the_image(self):
        # The honest consequence, documented on Inverter and pinned here:
        # the antisymmetric part of y lies in the null space of the
        # forward model, so moving disturbance between the two directions
        # of a link changes nothing. `measure` should expect noise
        # averaging from the second direction, not extra information.
        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        inverter = quiet_inverter(geom, alpha=1.0)
        rng = random.Random(31337)
        y = [rng.gauss(0.0, 1.0) for _ in range(geom.n_links)]

        swapped = list(y)
        for i, link in enumerate(geom.links):
            swapped[geom.link_index(link.rx_id, link.tx_id)] = y[i]
        averaged = [0.5 * (a + b) for a, b in zip(y, swapped)]

        base = inverter.solve(y)
        scale = max(abs(v) for v in base)
        self.assertGreater(scale, 0.0)
        for label, other in (("swapped", swapped), ("averaged", averaged)):
            with self.subTest(variant=label):
                worst = max(
                    abs(a - b) for a, b in zip(base, inverter.solve(other))
                ) / scale
                self.assertLess(worst, 1e-12, f"{label} differs by {worst:.3e}")

        # Control: a change that is NOT purely antisymmetric must move the
        # image, or the test above would pass on a solver ignoring y.
        bumped = list(y)
        bumped[0] += 1.0
        bumped[geom.link_index(geom.links[0].rx_id, geom.links[0].tx_id)] += 1.0
        moved = max(
            abs(a - b) for a, b in zip(base, inverter.solve(bumped))
        ) / scale
        self.assertGreater(moved, 1e-3, "solve ignored a symmetric change in y")


class TestDiagnosticsAndRefusals(unittest.TestCase):
    """The module's own safety claims, exercised rather than asserted.

    `IllConditionedError` was imported into this file and never raised in
    it: neither of its two paths had a test, nor did `residual_tol` or
    `check_inverse`. A future edit that inverted the back-check
    comparison, raised the tolerance or dropped the guard would have left
    the suite green and the module's central claim — fail loudly rather
    than silently — gone.
    """

    @staticmethod
    def rank_starved_geometry() -> Geometry:
        # 12 nodes but a 2 m pitch: 132 links over 12 pixels, so V C V^T
        # has rank at most 12 and is massively rank deficient whatever the
        # deduplication does. (It collapses 132 links to 51 rows here, not
        # the 66 the reciprocal pairs alone account for, because at this
        # pitch several genuinely distinct links cover exactly the same
        # pixels — which is the deduplication working as specified.)
        room = Room(8.0, 6.0)
        return Geometry(room, perimeter_nodes(room, 12, 0.1), PixelGrid(room, 2.0), 0.3)

    def test_the_back_check_refuses_an_uninvertible_system(self):
        geom = self.rank_starved_geometry()
        self.assertEqual(geom.n_links, 132)
        self.assertEqual(geom.n_pixels, 12)
        with self.assertRaises(IllConditionedError) as ctx:
            quiet_inverter(geom, alpha=1e-10)
        self.assertIn("B B^-1", str(ctx.exception))

    def test_the_back_check_degrades_smoothly_before_it_refuses(self):
        # MEASURED on the same geometry: 1e-2 -> 3.3e-14, 1e-4 -> 5.4e-12,
        # 1e-6 -> 5.3e-10, 1e-8 -> 4.3e-08, then the refusal at 1e-10.
        # The diagnostic is informative well before it trips, which is
        # what makes it usable as a sweep health signal.
        geom = self.rank_starved_geometry()
        residuals = [
            quiet_inverter(geom, alpha=a).inverse_residual
            for a in (1e-2, 1e-4, 1e-6, 1e-8)
        ]
        for a, b in zip(residuals, residuals[1:]):
            self.assertGreater(b, a, f"residual not growing as alpha falls: {residuals}")
        self.assertLess(residuals[0], 1e-12)

    def test_residual_tol_is_honoured(self):
        # The parameter that controls the refusal, which nothing used to
        # touch. Loosening it past the measured residual must turn the
        # refusal above back into a successful construction.
        geom = self.rank_starved_geometry()
        inverter = quiet_inverter(geom, alpha=1e-10, residual_tol=1e-3)
        self.assertGreater(inverter.inverse_residual, 1e-6)

    def test_the_refusal_names_the_effective_ridge_not_just_alpha(self):
        # sigma and alpha act only through alpha/sigma^2, so advice to
        # "raise alpha" is useless to anyone sweeping sigma. The message
        # has to quote the ratio and the scale it is measured against.
        geom = self.rank_starved_geometry()
        with self.assertRaises(IllConditionedError) as ctx:
            quiet_inverter(geom, alpha=1e-10, sigma=1.0)
        message = str(ctx.exception)
        self.assertIn("alpha/sigma^2", message)
        self.assertIn("mean diagonal", message)

    def test_an_indefinite_factorisation_is_reported_against_alpha(self):
        # A Cholesky failure inside Inverter means the ridge vanished into
        # rounding, not that an index is transposed, so it must not reach
        # the caller as NotPositiveDefiniteError with its "bug upstream"
        # docstring. Driven on the rank-starved geometry with the smallest
        # positive float there is, which is the cleanest reachable
        # instance of "the ridge is below the matrix's rounding level".
        geom = self.rank_starved_geometry()
        with self.assertRaises(IllConditionedError) as ctx:
            quiet_inverter(geom, alpha=5e-324, check_inverse=False)
        self.assertIn("alpha", str(ctx.exception))
        # Still a LinearAlgebraError, so one handler covers every refusal.
        self.assertIsInstance(ctx.exception, LinearAlgebraError)

    def test_check_inverse_false_skips_the_back_check_without_lying(self):
        # REGRESSION. It used to set inverse_residual to NaN, documented
        # nowhere and exercised by no test. Every comparison against NaN
        # is False, so `residual > tol` read as healthy and `residual <
        # tol` read as sick — the diagnostic did not merely lose
        # information, it inverted its meaning. None makes the same
        # comparison raise instead.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        skipped = quiet_inverter(geom, alpha=1.0, check_inverse=False)
        checked = quiet_inverter(geom, alpha=1.0, check_inverse=True)
        self.assertIsNone(skipped.inverse_residual)
        with self.assertRaises(TypeError):
            skipped.inverse_residual > 1e-9  # noqa: B015 - the point of the test
        self.assertIsInstance(checked.inverse_residual, float)
        # Skipping the check must not change the answer.
        self.assertEqual(skipped.projection, checked.projection)

    def test_projection_bytes_counts_the_float_objects_too(self):
        # REGRESSION. projection_bytes used to return 8 * P * L and claim
        # that was "what the Python process actually occupies". That
        # reasoning is borrowed from geometry.weight_matrix, where it
        # holds because every nonzero in a row is the same object. It does
        # not transfer: every entry of the projection comes out of its own
        # dot product and is a distinct float, so the real figure is 4.3x
        # higher and the ESP32 sizing was drawn from the wrong number.
        geom = Geometry.perimeter(8, pitch_m=0.5, lambda_m=0.3)
        inverter = quiet_inverter(geom, alpha=1.0)
        entries = inverter.n_pixels * inverter.n_independent_links
        self.assertEqual(inverter.projection_slots, 8 * entries)
        self.assertEqual(inverter.projection_bytes, 32 * entries)

        distinct = {id(v) for row in inverter.projection for v in row}
        # The premise of the 24-byte-per-entry term: no sharing at all.
        # MEASURED 10752 distinct objects for 10752 entries.
        self.assertGreater(len(distinct) / entries, 0.99)

        measured = sys.getsizeof(inverter.projection)
        measured += sum(sys.getsizeof(row) for row in inverter.projection)
        measured += sum(
            sys.getsizeof(v)
            for row in inverter.projection
            for v in row
        )
        # The formula deliberately omits per-row list headers, which do
        # not survive a port to C. MEASURED 186 KB against a reported
        # 168 KB, so the shortfall is ~10% and the old 4.3x gap is gone.
        self.assertLess(inverter.projection_bytes, measured)
        self.assertGreater(inverter.projection_bytes, 0.8 * measured)

    def test_the_asymmetry_guard_fires_when_the_prior_is_not_symmetric(self):
        # The guard had no test at all, and forcing asymmetry = 0.0 after
        # construction left the whole suite green, so nothing proved it
        # could fire. It cannot fire from the module's own arithmetic —
        # A = V C V^T is built from one `supports()` call and one
        # symmetric C, so it is symmetric by construction, which is why
        # Inverter documents it as a belt-and-braces invariant rather than
        # as a detector of any bug reachable today. What it does catch is
        # a C that is not symmetric, which is what an index order
        # disagreeing inside the prior would produce. Injected here
        # because there is no other way in.
        class SkewedPrior(SpatialPrior):
            __slots__ = ()

            def row(self, p: int) -> list[float]:
                base = SpatialPrior.row(self, p)
                return [v * (1.0 + 0.3 * (q < p)) for q, v in enumerate(base)]

        geom = Geometry.perimeter(6, pitch_m=0.5, lambda_m=0.5)
        with unittest.mock.patch.object(invert_module, "SpatialPrior", SkewedPrior):
            with self.assertRaises(IllConditionedError) as ctx:
                quiet_inverter(geom, alpha=1.0)
        self.assertIn("asymmetric", str(ctx.exception))

    def test_the_asymmetry_diagnostic_is_at_rounding_level_when_it_should_be(self):
        # The other side: with the real prior the diagnostic must stay at
        # rounding, across deployments. MEASURED 2.6e-16 to 1.4e-15.
        for n_nodes, pitch in ((8, 0.5), (8, 0.25), (12, 0.25)):
            with self.subTest(n_nodes=n_nodes, pitch=pitch):
                inverter = quiet_inverter(
                    Geometry.perimeter(n_nodes, pitch_m=pitch, lambda_m=0.3),
                    alpha=1.0,
                    check_inverse=False,
                )
                self.assertLess(inverter.asymmetry, 1e-14)


if __name__ == "__main__":
    unittest.main()
