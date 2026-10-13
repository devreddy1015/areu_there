"""Regularized inversion: per-link disturbance vector -> spatial image.

This is the mathematical core of RTI. `measure` produces a vector y with
one number per link saying how disturbed that link is; `geometry` supplies
the forward model W saying which pixels each link can see. This module
answers the only question that matters: given y, what image x explains it?

The honest answer is "many images do". An 8-node mesh gives 56 links; a
0.5 m grid over an 8 x 6 m room gives 192 pixels, and a 0.25 m grid gives
768. The system W x = y is therefore underdetermined by a factor of 3 to
14, and least squares alone would hand back whatever salt-and-pepper
pattern fits the noise best. The fix is a prior that says neighbouring
pixels are correlated:

    C_pq = sigma^2 * exp(-dist(p, q) / delta_c)

Minimising ||y - W x||^2 + alpha * x^T C^-1 x then gives

    x_hat = (W^T W + alpha C^-1)^-1 W^T y                        [primal]
          = C W^T (W C W^T + alpha I)^-1 y                       [dual]

The two are algebraically identical — push-through:
(W^T W + alpha C^-1) C W^T = W^T (W C W^T + alpha I) — but they cost
wildly different amounts. The primal inverts a P x P matrix (768^3 / 3 =
1.5e8 pure-Python operations, plus it needs C^-1, which is a second P x P
inversion of a matrix that is nearly singular whenever delta_c is large).
The dual inverts an L x L matrix: 28 to 66 for the deployments we care
about, i.e. 1e4 to 1e5 operations, and C is never inverted at all. On an
underdetermined problem the dual form is not an optimisation, it is the
difference between feasible and not.

RECIPROCAL LINKS ARE ONE ROW, NOT TWO. `geometry.Link` is ordered, so an
N-node mesh has N*(N-1) links — but `excess_path_m` and `length_m` are
both symmetric in (tx, rx), so the rows of W for (i, j) and (j, i) are
*identical*, element for element. MEASURED at 6, 8 and 12 nodes across
pitches {0.25, 0.5, 1.0} m and lambdas {0.3, 0.75, 1.5} m — 27
configurations — exactly half the rows are duplicates of another row,
every time. Feeding both copies to the dual form is not merely wasteful,
it is structurally wrong:

  * W C W^T then has L/2 exactly-zero eigenvalues. It is PSD, never PD,
    and the ridge alpha I is the only thing holding the factorisation up.
    At alpha = 1e-15 on the 8-node deployment the old code fell over with
    `NotPositiveDefiniteError` on a perfectly well-formed input.
  * The antisymmetric part of y — the difference between the two
    directions of a link — lies in the null space of both W C W^T and
    C W^T, so it is annihilated exactly. The second direction cannot
    contribute one bit of geometric information through this estimator.
    (It is still worth measuring: averaging the two directions halves the
    variance of the noise on that row. What it cannot do is see anything
    the first direction could not.)
  * Cost. The dominant P*L^2 projection term is 4x larger than it needs
    to be, and the L^3 factorisation plus back-check 8x.

So W's duplicate rows are collapsed at construction. Writing V for the
deduplicated forward model, with row g scaled by sqrt(m_g) where m_g is
how many links share it:

    W^T W = V^T V      and      W^T y = V^T r,   r_g = (sum y_i)/sqrt(m_g)

which makes x = C V^T (V C V^T + alpha I)^-1 r the SAME ESTIMATOR, with
the same alpha, on a matrix that is genuinely positive definite. MEASURED:
the deduplicated V C V^T factors with no ridge at all at 6, 8 and 12 nodes
and both pitches, and the image agrees with the old 56-row form to a
relative 7e-15 at alpha = 1 (the disagreement is pure rounding, and it
grows to 3e-9 at alpha = 1e-6 because the old rank-deficient form is the
inaccurate side of the comparison). `solve` still takes y in
`geometry.n_links` canonical link order, so nothing upstream changes.

So: build the P x L projection

    Pi = C V^T (V C V^T + alpha I)^-1

once at construction, and make `solve` one reduction of y plus a single
matrix-vector product. That is also the shape this has to take on an
ESP32 eventually — the device holds Pi and nothing else, no W, no C, no
factorisation.

COMPLEXITY, with P pixels, L deduplicated rows (half the link count) and
nnz = sum over rows of the number of pixels inside that link's ellipse
(16-21% of L*P at the settings in `geometry.pixel_link_counts`):

    construction   O(P^2 + P*nnz + nnz*L + L^3 + P*L^2)
                   dominated in practice by P*nnz (forming C V^T) and
                   P*L^2 (forming Pi). Measured figures are on `Inverter`.
    solve          O(P*L) — one dense matrix-vector product, no branches.

Three numerical notes, because this is where a pure-Python RTI goes
quietly wrong rather than loudly wrong:

  * C is never formed as a P x P array. On a regular grid dist(p, q)
    depends only on the index offsets, so C is two-level Toeplitz and
    `SpatialPrior` stores O(n_x^2 * n_y) numbers instead of O(P^2) —
    5 MB saved at a 0.25 m pitch, and no accuracy given up: the entries
    are bit-identical to the direct formula.
  * exp(-r / delta_c) is the Matern-1/2 covariance, which is strictly
    positive definite in the plane for distinct points. So C is PD, and
    A = V C V^T is PD as long as V has independent rows — which is why
    the duplicate rows above had to go. Cholesky failing here is
    therefore informative but NOT proof of an upstream bug: two links
    whose ellipses nearly coincide would also do it (exactly coinciding
    ones are merged, so they cannot), and so would an alpha so far below
    `mean_diag_a` that the ridge disappears into rounding (MEASURED: the
    8-node 0.5 m deployment
    survives alpha = 1e-300 after deduplication, where before it failed
    at 1e-15). `Inverter` therefore converts a Cholesky failure into
    `IllConditionedError` quoting alpha against mean_diag_a, rather than
    sending the reader off to hunt a transposed index.
  * Index order is checked where checking actually proves something. W is
    L x P, C V^T is P x L, V C V^T is L x L, Pi is P x L, and a
    transposed index here produces a mirrored image that looks entirely
    reasonable — the classic RTI bug. `Inverter.asymmetry` measures how
    far V C V^T is from symmetric and refuses beyond rounding. Be precise
    about what that catches: an A assembled with the two index orders
    disagreeing. It does NOT catch a permuted pixel index, because C is
    symmetric under any permutation of the pixel index and so is
    V Pi C Pi^T V^T — VERIFIED by mutation, a prior that reads the pixel
    index as ix*n_y+iy leaves asymmetry at 3e-16. What catches that is
    `test_localize_invert`'s entry-for-entry check of the prior against
    exp(-dist/delta_c) on the real pixel coordinates, on a grid whose two
    pitches differ, plus the exhaustive blob sweep. MEASURED on the three
    standard deployments: relative asymmetry 2.6e-16 to 1.4e-15, and the
    back-check max |B B^-1 - I| 2.2e-15 to 6.2e-14 — against 9.6e-15 to
    4.4e-13 for the same three before the duplicate rows were collapsed,
    measured in the same process. Removing the rank deficiency improved
    the conditioning by 4x to 18x as well as the speed.
"""

from __future__ import annotations

import math
import sys
import warnings
from operator import add, mul

from .geometry import Geometry, PixelGrid

__all__ = [
    "LinearAlgebraError",
    "MalformedMatrixError",
    "AsymmetricMatrixError",
    "NotPositiveDefiniteError",
    "IllConditionedError",
    "BlindPixelWarning",
    "cholesky_factor",
    "cholesky_solve_factored",
    "cholesky_solve",
    "cholesky_inverse",
    "SpatialPrior",
    "forward_project",
    "Inverter",
]


class LinearAlgebraError(Exception):
    """Base class for the solver's refusals.

    Every refusal ABOUT A MATRIX derives from this — shape, finiteness,
    symmetry, definiteness, conditioning — so a parameter sweep can wrap a
    configuration in one `except LinearAlgebraError` and record it as a
    failed cell instead of crashing. Before this covered the asymmetry
    case, such a driver caught `Inverter`'s refusal for exactly that
    condition and then crashed on `cholesky_factor`'s.

    The shape, finiteness and symmetry refusals ALSO derive from
    `ValueError`, because they were plain ValueErrors before these classes
    existed and callers catching that must keep working.

    A bad CONSTRUCTOR ARGUMENT — alpha, sigma or delta_c non-positive or
    non-finite — is deliberately not one of these. It is a caller bug, not
    a property of the data, and a sweep that quietly records it as a
    failed cell would hide a typo in the sweep itself.
    """


class MalformedMatrixError(LinearAlgebraError, ValueError):
    """The input was not a square matrix of finite numbers."""


class AsymmetricMatrixError(LinearAlgebraError, ValueError):
    """A matrix offered as symmetric is not symmetric.

    Its own class rather than a bare ValueError because this is the
    transposed-index trap, the single most likely way to hand the solver
    a matrix that factors cleanly and answers a different question.
    """


class NotPositiveDefiniteError(LinearAlgebraError):
    """Cholesky hit a non-positive pivot.

    Loud on purpose, but not self-evidently a bug: it means the matrix as
    presented is not positive definite, which has three plausible causes
    in this package. Duplicate rows in the forward model, a ridge so small
    it vanishes into rounding against `mean_diag_a`, and a genuine
    upstream index error. `Inverter` rules the first out by construction —
    identical rows are merged before A is built, and MEASURED on a
    deliberately rank-starved 12-node 2 m-pitch deployment that collapses
    132 links to 51 rows, not just the 66 the reciprocal pairs account for
    — and it names the second when it re-raises this as
    `IllConditionedError`. What reaches a caller is therefore the third,
    or rows that are near-duplicates rather than exact ones.
    """


class IllConditionedError(LinearAlgebraError):
    """A matrix was factored but the result does not survive a back-check."""


class BlindPixelWarning(UserWarning):
    """Some pixels are inside no link's ellipse and can never light up."""


def _require_positive_finite(name: str, value: float) -> float:
    """Reject non-positive AND non-finite parameters, naming the parameter.

    The obvious `if value <= 0.0: raise` lets both NaN and +inf through,
    because every comparison against a NaN is False and +inf is not <= 0.
    They then propagate silently into C, A and the factorisation and
    surface a fifth of a second later as "matrix contains a non-finite
    entry" — an error that points at the matrix rather than at the
    argument, which is precisely the traceability problem `solve`'s
    per-index check on y was written to avoid. Costs one call per
    construction; the construction behind it costs 30 ms.
    """
    if not (value > 0.0) or not math.isfinite(value):
        raise ValueError(f"{name} must be positive and finite, got {value}")
    return value


def _dot(a: list[float], b: list[float]) -> float:
    """Inner product at C speed.

    `sum(map(mul, a, b))` keeps the whole loop inside the interpreter's
    C layer; the obvious generator expression is roughly 3x slower, and
    this is called P*L times when building the projection.
    """
    return sum(map(mul, a, b))


def cholesky_factor(
    a: list[list[float]],
    symmetry_tol: float = 1e-9,
) -> list[list[float]]:
    """Lower-triangular G with G G^T = A, for symmetric positive definite A.

    Raises rather than limping. `NotPositiveDefiniteError` carries the
    pivot index and the value that failed, because "which pivot" is the
    single most useful piece of information when chasing why a covariance
    came out indefinite.

    The symmetry check is not pedantry: feeding this a matrix that was
    meant to be V C V^T but was accidentally built with the pixel and link
    indices swapped produces something that is usually still factorable,
    and the resulting image is mirrored but otherwise plausible. Checking
    here catches it at the cause instead of in a tracker trace three
    modules later. Note that `Inverter` symmetrizes its own matrix
    explicitly before calling this, so what this guard protects is every
    OTHER caller of the module's exported solver — inside `Inverter` the
    asymmetry seen here is exactly 0.0 by construction.

    Two things the obvious implementations of those checks get wrong, both
    of which this one is written around:

      * NON-FINITE ENTRIES CANNOT BE FOUND WITH max(). CPython's `max`
        keeps its running result unless the next value compares greater,
        and `nan > x` is always False, so a NaN anywhere except the first
        position is skipped and `isfinite(scale)` passes. The same trick
        defeats a `worst = max(worst, abs(a[i][j] - a[j][i]))` symmetry
        scan: `max(0.0, nan)` is 0.0, so a NaN-against-number pair reads
        as zero asymmetry. Both guards then pass, the factorisation reads
        only the lower triangle, and the caller gets a clean factor of the
        lower triangle mirrored — a different matrix, with no exception.
        So the scan below tests `isfinite` explicitly, per entry.
      * `symmetry_tol` IS PER PAIR, not against the largest entry in the
        whole matrix. A global scale lets an asymmetry hide among the
        small entries of a matrix with a large diagonal: with the global
        form, [[1e9, 1.0], [2.0, 1e9]] has a relative asymmetry of exactly
        1e-9 and squeaks through. The absolute floor added to the per-pair
        tolerance is what keeps the strict version usable — an entry that
        ought to be zero emerges from a long accumulation at a few ulp of
        the matrix scale, and demanding relative agreement there would
        reject honest arithmetic. The comparison is written negated
        (`not (d <= tol)`) so that a NaN difference is a refusal; the
        natural `d > tol` spelling accepts exactly the input the check
        exists to reject. A difference exactly equal to the tolerance is
        accepted.
    """
    n = len(a)
    if n == 0:
        raise MalformedMatrixError("cannot factor an empty matrix")
    for i, row in enumerate(a):
        if len(row) != n:
            raise MalformedMatrixError(
                f"matrix is not square: row {i} has {len(row)} entries, need {n}"
            )

    scale = 0.0
    for i, row in enumerate(a):
        # `all(map(isfinite, ...))` runs in C and, unlike max(), cannot be
        # stepped over by a NaN. Only once a row is known finite is it
        # safe to ask max() for its largest magnitude.
        if not all(map(math.isfinite, row)):
            j = next(j for j, v in enumerate(row) if not math.isfinite(v))
            raise MalformedMatrixError(
                f"a[{i}][{j}] is {row[j]}; the matrix contains a non-finite "
                f"entry. Cholesky would otherwise return a plausible-looking "
                f"factor of a different matrix"
            )
        largest = max(map(abs, row))
        if largest > scale:
            scale = largest
    if scale <= 0.0:
        raise NotPositiveDefiniteError("matrix is entirely zero")

    # A few ulp of the matrix scale: the size of the rounding error a
    # legitimate symmetric accumulation leaves behind in an entry that
    # ought to be zero.
    floor = 8.0 * sys.float_info.epsilon * scale
    for i in range(n):
        ai = a[i]
        for j in range(i):
            upper = a[j][i]
            d = abs(ai[j] - upper)
            if not (d <= symmetry_tol * max(abs(ai[j]), abs(upper)) + floor):
                raise AsymmetricMatrixError(
                    f"matrix is not symmetric: a[{i}][{j}] = {ai[j]:.6e} but "
                    f"a[{j}][{i}] = {upper:.6e}, differing by {d:.3e} against "
                    f"a tolerance of {symmetry_tol:.1e} relative to the pair "
                    f"plus {floor:.3e}; this usually means a transposed index "
                    f"upstream"
                )

    g: list[list[float]] = [[0.0] * n for _ in range(n)]
    for i in range(n):
        gi = g[i]
        ai = a[i]
        for j in range(i):
            gj = g[j]
            gi[j] = (ai[j] - _dot(gi[:j], gj[:j])) / gj[j]
        diag = ai[i] - _dot(gi[:i], gi[:i])
        if not math.isfinite(diag) or diag <= 0.0:
            raise NotPositiveDefiniteError(
                f"pivot {i} of {n} is {diag:.6e}, which is not positive; "
                f"the matrix is not positive definite "
                f"(original diagonal entry was {ai[i]:.6e})"
            )
        gi[i] = math.sqrt(diag)
    return g


def cholesky_solve_factored(g: list[list[float]], b: list[float]) -> list[float]:
    """Solve G G^T x = b given the factor G from `cholesky_factor`."""
    n = len(g)
    if len(b) != n:
        raise ValueError(f"right-hand side has length {len(b)}, need {n}")

    # Forward substitution: G z = b.
    z = [0.0] * n
    for i in range(n):
        gi = g[i]
        z[i] = (b[i] - _dot(gi[:i], z[:i])) / gi[i]

    # Back substitution: G^T x = z. G^T's row i is column i of G, which is
    # strided, so walk it explicitly rather than slicing.
    x = [0.0] * n
    for i in range(n - 1, -1, -1):
        s = z[i]
        for k in range(i + 1, n):
            s -= g[k][i] * x[k]
        x[i] = s / g[i][i]
    return x


def cholesky_solve(a: list[list[float]], b: list[float]) -> list[float]:
    """Solve A x = b for symmetric positive definite A. Convenience wrapper."""
    return cholesky_solve_factored(cholesky_factor(a), b)


def cholesky_inverse(a: list[list[float]]) -> list[list[float]]:
    """A^-1 for symmetric positive definite A, symmetrized.

    Forming an explicit inverse is normally the wrong move — a triangular
    solve per right-hand side is both faster and more accurate. It is the
    right move here because the inverse is consumed by `Inverter` as one
    factor of a P x L product with P in the hundreds to thousands: a
    triangular solve per pixel costs the same O(P L^2) but with a serial
    dependency chain per element, whereas an explicit inverse turns the
    whole thing into P*L dot products that run at C speed.

    The result is symmetrized, because A^-1 is symmetric in exact
    arithmetic and the column-by-column solves accumulate in different
    orders. The asymmetry is at rounding level; averaging it away costs
    nothing and keeps downstream symmetry tests meaningful.
    """
    g = cholesky_factor(a)
    n = len(g)
    cols: list[list[float]] = []
    for j in range(n):
        e = [0.0] * n
        e[j] = 1.0
        cols.append(cholesky_solve_factored(g, e))
    # cols[j] is column j, so inv[i][j] = cols[j][i].
    return [
        [0.5 * (cols[j][i] + cols[i][j]) for j in range(n)]
        for i in range(n)
    ]


class SpatialPrior:
    """C_pq = sigma^2 * exp(-dist(p, q) / delta_c) over a `PixelGrid`.

    The prior is what makes an underdetermined inversion produce a blob
    instead of noise. `delta_c` is the correlation length in metres: it
    sets how wide a single pixel's influence spreads.

    MEASURED (8 nodes, 0.5 m pitch, alpha = 1, blob at the room centre):
    the fraction of |x| landing within 1 m of the true blob is UNIMODAL in
    delta_c, not monotone —

        delta_c  0.05  0.10  0.25  0.40  0.50  0.75  1.00  2.00  4.00  8.00
        conc.    .329  .333  .393  .429  .434  .436  .434  .433  .427  .408

    peaking at one to two pixel pitches and nearly flat across the middle,
    because in that range the resolution is set by the ellipse width and
    the link geometry rather than by the prior. Below the pixel pitch, C
    goes nearly diagonal, the prior stops coupling neighbours at all and
    the image goes ringy. Far above it, C approaches rank one, every image
    becomes the same smear, and the back-check residual on the dual solve
    grows from 1.1e-15 at delta_c = 0.25 m to 3.0e-13 at 8 m. (It stops
    growing there: 1.9e-13 at 50 m and 3.9e-13 at 500 m. Do not read the
    residual as a monotone function of delta_c past the room scale.) So
    delta_c is a knob with an interior optimum near the pixel pitch, not
    a smoothing strength to be turned up.

    `sigma` only ever enters as C = sigma^2 * C0, which makes it redundant
    with `alpha` — see the sweep note on `Inverter`. It is kept because
    the shared interface contract names it.

    THE KERNEL IS ANISOTROPIC WHENEVER THE GRID IS. `PixelGrid` rounds n_x
    and n_y independently, so a target pitch that divides neither room
    dimension gives pitch_x != pitch_y — 7 of 13 ordinary pitch values do
    on the default 8 x 6 m room, e.g. 0.7 m gives an 11 x 9 grid at
    0.7273 x 0.6667 m. The offset-to-metres step below must therefore pair
    dix with pitch_x and diy with pitch_y. MEASURED on that 11 x 9 grid at
    delta_c = 0.5 m, swapping them moves C by up to 0.030 against a unit
    diagonal — 3% of the prior's full scale, but a factor of 2.4 on the
    far-field entries and 39% on the median differing entry — stretched
    along the wrong room axis. It is invisible on any square-pitch grid,
    so the test that pins this kernel entry-for-entry uses a grid where
    the two pitches differ; before that grid existed the swap passed all
    52 tests in the file.

    NEVER MATERIALISED AS P x P. On a regular grid dist(p, q) depends only
    on (|ix_p - ix_q|, |iy_p - iy_q|), so C is two-level Toeplitz. What is
    stored is a table of n_y strips of n_x rows of n_x floats, which `row`
    assembles by concatenation. At a 0.25 m pitch in an 8 x 6 m room that
    is 24576 list slots (~200 KB) against 589824 entries (~5 MB) for the
    dense form, and the values are bit-identical because they come from
    the same expression — see `test_localize_invert`, which checks every
    entry against the direct formula on the actual pixel coordinates.

    The Toeplitz structure is a property of `PixelGrid`, which forces a
    uniform pitch so the grid tiles the room exactly. Anything with
    irregular pixel spacing would need the dense form and is not supported.
    """

    __slots__ = ("grid", "sigma", "delta_c", "sigma2", "_n_x", "_n_y", "_strips")

    def __init__(self, grid: PixelGrid, sigma: float = 1.0, delta_c: float = 0.5) -> None:
        _require_positive_finite("sigma", sigma)
        _require_positive_finite("delta_c", delta_c)

        self.grid = grid
        self.sigma = sigma
        self.delta_c = delta_c
        self.sigma2 = sigma * sigma
        if not math.isfinite(self.sigma2):
            # sigma = 1e200 is finite and positive and squares to inf, so
            # a sigma sweep has a silent cliff unless this is caught here
            # rather than as a non-finite entry in C half a second later.
            raise ValueError(
                f"sigma = {sigma} overflows to infinity when squared; "
                f"sigma is redundant with alpha anyway (see Inverter), so "
                f"fix sigma = 1 and scale alpha instead"
            )

        n_x = self._n_x = grid.n_x
        n_y = self._n_y = grid.n_y
        px = grid.pitch_x_m
        py = grid.pitch_y_m

        # Build the distinct values once so every strip shares the same
        # float objects; the strips then cost 8 bytes a slot, not 32.
        # dix pairs with px and diy with py — see the anisotropy note above.
        kern = [
            [
                self.sigma2 * math.exp(-math.hypot(dix * px, diy * py) / delta_c)
                for diy in range(n_y)
            ]
            for dix in range(n_x)
        ]
        self._strips: list[list[list[float]]] = [
            [[kern[abs(ixp - ix)][diy] for ix in range(n_x)] for ixp in range(n_x)]
            for diy in range(n_y)
        ]

    def entry(self, p: int, q: int) -> float:
        """C_pq for two canonical pixel indices.

        Bounds-checked, the same as `row`. Without the check a negative
        index wraps through Python's negative indexing and returns a
        perfectly plausible covariance for a different pair of pixels —
        `entry(-1, 0)` on a 48-pixel grid used to answer for pixel 41.
        That matters because a caller reading C around an image peak
        (a sub-pixel centroid at a wall-adjacent peak, say) steps to
        ix - 1 as a matter of course, and a wrong-but-smooth covariance is
        the hardest kind of error to notice downstream.
        """
        n_x = self._n_x
        n = n_x * self._n_y
        if not (0 <= p < n and 0 <= q < n):
            raise IndexError(f"pixel pair ({p}, {q}) outside grid of {n}")
        return self._strips[abs(p // n_x - q // n_x)][p % n_x][q % n_x]

    def row(self, p: int) -> list[float]:
        """Row p of C as a flat list in canonical pixel order."""
        n_x = self._n_x
        if not 0 <= p < n_x * self._n_y:
            raise IndexError(f"pixel {p} outside grid of {n_x * self._n_y}")
        ixp = p % n_x
        iyp = p // n_x
        strips = self._strips
        out: list[float] = []
        extend = out.extend
        for iy in range(self._n_y):
            extend(strips[abs(iyp - iy)][ixp])
        return out

    def dense(self) -> list[list[float]]:
        """The full P x P matrix. For tests and small primal solves only."""
        return [self.row(p) for p in range(self._n_x * self._n_y)]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"SpatialPrior(sigma={self.sigma:g}, delta_c={self.delta_c:g} m, "
            f"grid={self._n_x}x{self._n_y})"
        )


def forward_project(geometry: Geometry, image: list[float]) -> list[float]:
    """y = W x: what per-link vector would this image produce?

    The forward model, in canonical link order — all `geometry.n_links` of
    them, including both directions of every link, because this is what a
    radio would report. (The inversion collapses the duplicate rows
    internally; see `Inverter`. A round trip through both is still exact.)

    It exists here rather than in `geometry` because it is the inversion's
    own round trip: a test that builds y from a known x with this function
    and recovers x with `Inverter.solve` is checking the inversion, not
    the weight model.

    Iterates `geometry.supports()` rather than `weight_matrix()`, so the
    cost is O(nnz) rather than O(L*P) and no dense W is ever built.
    """
    if len(image) != geometry.n_pixels:
        raise ValueError(
            f"image has {len(image)} pixels, geometry has {geometry.n_pixels}"
        )
    out: list[float] = []
    for link, support in zip(geometry.links, geometry.supports()):
        w = 1.0 / math.sqrt(link.length_m) if link.length_m > 0.0 else 0.0
        out.append(w * sum(map(image.__getitem__, support)))
    return out


class Inverter:
    """Dual-form regularized least squares, precomputed per geometry.

    Construction does all the work; `solve` is one reduction of y followed
    by a matrix-vector product against the cached P x L projection. Treat
    an instance as immutable — alpha, sigma and delta_c are baked into the
    projection, so a sweep builds a new `Inverter` per setting.

    L HERE IS THE NUMBER OF DISTINCT ROWS OF W, which is half the link
    count: reciprocal links have identical rows and are collapsed, exactly
    and without changing the estimator (the module docstring derives it).
    `solve` still takes y of length `geometry.n_links`; `n_links` and
    `n_independent_links` are both exposed, and `link_groups` says which
    link indices were merged into each row.

    MEASURED, CPython 3.14.7, 8 x 6 m room, lambda = 0.3 m, sigma = 1.0,
    delta_c = 0.5 m, wall clock on an otherwise busy 20-core box; `solve`
    is the mean of 20 calls. To reproduce: build the Geometry, time the
    Inverter constructor, then time `solve` on a forward-projected blob.

        nodes links rows pitch  pixels  construct   solve  P*L vals  mean diag A
          8     56   28  0.50 m    192    0.009 s  0.12 ms     5376      39.9
          8     56   28  0.25 m    768    0.056 s  0.47 ms    21504     590.6
         12    132   66  0.25 m    768    0.175 s  1.08 ms    50688     580.0

    Those are 4.8x, 3.4x and 5.2x faster to construct than the identical
    arithmetic over all 56/56/132 rows (0.043 / 0.193 / 0.904 s, measured
    in the same process, best of three each), and half the projection. The
    speedup beats the 4x the P*L^2 term predicts because the L^3 inverse
    and its O(L^3) back-check fall by 8x.

    Construction is a one-off per deployment and per parameter setting; a
    live tracker at 1-5 Hz pays only the solve column, and 1.1 ms at the
    largest setting leaves the update rate limited by the radio, not by
    this.

    The projection size is the number that matters for the eventual ESP32
    port. Those P*L counts are 21 KB, 84 KB and 198 KB as float32. CPython
    is far heavier than a C array and `projection_bytes` says so honestly:
    every entry is a DISTINCT float object, so the resident cost is the
    8-byte list slot plus a 24-byte float, 32 bytes an entry — 168 KB,
    672 KB and 1584 KB. (MEASURED with sys.getsizeof over the outer list,
    every row and every entry: 186 KB, 745 KB and 1693 KB; the extra 7-11%
    is per-row list headers, which the formula deliberately leaves out
    because it does not survive a port.) The 8-node cases fit an ESP32 in
    float32; the 12-node 0.25 m case wants 198 KB of a ~320 KB DRAM
    budget, which leaves too little for anything else and argues for
    PSRAM. Worth knowing before choosing a grid pitch.

    `alpha` trades fit against smoothness. It has the UNITS OF THE
    DIAGONAL of V C V^T, so it is not dimensionless and a value tuned at
    one pitch does not transfer to another — note the mean-diagonal column
    above moving by 15x between a 0.5 m and a 0.25 m grid at the same node
    count, because a finer grid puts proportionally more pixels inside
    each ellipse. `mean_diag_a` is therefore exposed after construction,
    and a sweep is best parameterised as a fraction of it. (Collapsing the
    duplicate rows doubled `mean_diag_a`, because a merged row carries
    sqrt(2) times the weight of a single link. The estimator at a given
    alpha is unchanged; only this reported scale moved.)

    The default alpha = 1.0 is about 2.5% of `mean_diag_a` for the
    standard 8-node 0.5 m deployment with sigma = 1. MEASURED on SYNTHETIC
    forward-projected blobs (no simulator, no radio), a blob of width
    0.4 m placed on each of the 192 pixel centres in turn, "recovered"
    meaning the argmax is within one pixel in index distance:

        clean y     192/192 at every alpha in {0.1, 0.2, 0.5, 1, 2, 5,
                    10, 20}. The alpha axis is genuinely flat.
        20% noise   191 to 192 of 192 over seeds 0-4 (one miss on seeds
                    0 and 4), worst displacement 1.12 m.
        30% noise   191 to 192 of 192, worst 1.12 m.
        50% noise   190 to 191 of 192, worst 2.00 m.

    Noise is Gaussian at that fraction of the RMS of the clean y over all
    192 positions. Every miss sits on the structurally blind corners. An
    earlier version of this docstring claimed the recovery was universal
    at up to 50% noise; it is not, and the figures above replace it —
    `test_noise_degradation_curve_is_pinned_across_seeds` runs exactly
    this sweep, so the claim and the test cannot drift apart again.

    TWO PROPERTIES THAT MATTER FOR THE PARAMETER SWEEP, both verified to
    machine precision in `test_localize_invert`:

      * SIGMA IS REDUNDANT. Substituting C = sigma^2 C0 into the dual form
        gives x = C0 V^T (V C0 V^T + (alpha/sigma^2) I)^-1 y, so the image
        depends on sigma and alpha ONLY through alpha/sigma^2. Sweeping
        them as two axes wastes the whole sigma dimension. Fix sigma = 1
        and sweep alpha. (`sigma` stays in the signature because the
        shared interface contract specifies it.) Note the corollary for
        anyone who sweeps sigma anyway: raising sigma lowers the EFFECTIVE
        ridge, so it is sigma, not alpha or delta_c, that trips the
        back-check — which is why `IllConditionedError` quotes
        alpha/sigma^2 rather than alpha.
      * THE IMAGE IS EXACTLY LINEAR IN y, so y's absolute scale cancels
        out of the peak location entirely — rescaling y by 1e-6 or 1e6
        moves no peak and changes no relative value by more than 2e-16.
        `measure` emits variance changes of order 1e-4 and that needs no
        normalising before inversion, and alpha does not have to be
        retuned to match it. What alpha does have to be swept against is
        the NOISE-TO-SIGNAL RATIO within y, and the pixel pitch.

    ONE PROPERTY THAT MATTERS FOR `measure`: the two directions of a link
    are summed, and only their sum reaches the image. Anything that
    distinguishes (i -> j) from (j -> i) — a different noise floor, a
    different antenna orientation — is averaged away, because the two
    directions have identical rows in W and no estimator built on W can
    separate them. Measuring both is still worth it for the noise
    averaging; expecting independent geometric information from it is not.
    """

    def __init__(
        self,
        geometry: Geometry,
        alpha: float = 1.0,
        sigma: float = 1.0,
        delta_c: float = 0.5,
        *,
        residual_tol: float = 1e-6,
        check_inverse: bool = True,
        warn_blind_pixels: bool = True,
    ) -> None:
        # alpha = 0 is not merely ill-advised, it is wrong: V C V^T is
        # L x L but built from an L x P V with L < P, so it is singular
        # whenever two rows are close enough. The ridge is what makes the
        # dual form well posed. Non-finite is rejected by name here rather
        # than as an opaque non-finite entry inside the factorisation.
        _require_positive_finite("alpha", alpha)

        self.geometry = geometry
        self.alpha = alpha
        self.prior = SpatialPrior(geometry.grid, sigma, delta_c)
        self.sigma = self.prior.sigma
        self.delta_c = self.prior.delta_c
        self.n_links = geometry.n_links
        self.n_pixels = geometry.n_pixels

        counts = geometry.pixel_link_counts()
        self.blind_pixels: list[int] = [p for p, c in enumerate(counts) if c == 0]
        if self.blind_pixels and warn_blind_pixels:
            # Not an error: the two unoccupied corners of the standard
            # 8-node deployment are structurally blind (geometry's
            # `pixel_link_counts` documents why), and refusing would make
            # the default configuration unusable. But the inversion will
            # still return a smooth confident value there, drawn purely
            # from the prior and from neighbouring pixels, so anyone
            # reading a peak out of this image needs to know.
            warnings.warn(
                f"{len(self.blind_pixels)} of {self.n_pixels} pixels are inside "
                f"no link's ellipse and cannot be reconstructed; their value "
                f"comes entirely from the spatial prior. Raise lambda_m or add "
                f"nodes. First few: {self.blind_pixels[:5]}",
                BlindPixelWarning,
                stacklevel=2,
            )

        supports = geometry.supports()
        link_w = [
            1.0 / math.sqrt(link.length_m) if link.length_m > 0.0 else 0.0
            for link in geometry.links
        ]

        # --- collapse duplicate rows of W -------------------------------
        #
        # Two links with the same weight and the same support have the
        # same row of W, and reciprocal links always do. Grouping on the
        # row itself rather than on (tx, rx) ordering keeps this exact
        # whatever `geometry` does: anything that is genuinely a distinct
        # row stays a distinct row, so the collapse can only ever remove
        # redundancy, never information. An empty support means a row of
        # zeros regardless of the weight, so those merge too.
        #
        # Row g of V is then sqrt(m_g) times the shared W row, and the
        # matching reduction of y is r_g = (sum of the group) / sqrt(m_g);
        # together those reproduce W^T W and W^T y exactly, so the
        # estimator is untouched. See the module docstring for the algebra.
        seen: dict[tuple[tuple[int, ...], float], int] = {}
        self.link_groups: list[list[int]] = []
        for i, (w, support) in enumerate(zip(link_w, supports)):
            key = (tuple(support), w) if support else ((), 0.0)
            at = seen.get(key)
            if at is None:
                seen[key] = at = len(self.link_groups)
                self.link_groups.append([])
            self.link_groups[at].append(i)

        self.n_independent_links = n_rows = len(self.link_groups)
        reps = [group[0] for group in self.link_groups]
        row_support = [supports[r] for r in reps]
        row_w = [
            link_w[r] * math.sqrt(len(group))
            for r, group in zip(reps, self.link_groups)
        ]
        # Reciprocal of the same sqrt, applied to y in `solve`.
        self._group_scale = [1.0 / math.sqrt(len(g)) for g in self.link_groups]

        # --- M = C V^T, shape P x L -------------------------------------
        #
        # M[p][g] = sum_q C[p][q] V[g][q], and V row g is the constant
        # row_w[g] on its support and zero elsewhere, so the inner sum
        # runs over the support only. Looping p outermost means each row
        # of C is built once and reused across all L rows.
        prior_row = self.prior.row
        m: list[list[float]] = []
        for p in range(self.n_pixels):
            row = prior_row(p)
            get = row.__getitem__
            m.append([w * sum(map(get, s)) for w, s in zip(row_w, row_support)])
        # m stays a local. It is the same size as `projection` (P x L), and
        # keeping it on the instance would double the resident memory of
        # every Inverter for something nothing reads after construction.

        # --- A = V M, shape L x L ---------------------------------------
        #
        # A[g][k] = row_w[g] * sum over p in support(g) of M[p][k]. The
        # accumulation is a vectorised list add so the L-long inner loop
        # stays in C.
        a: list[list[float]] = []
        for w, support in zip(row_w, row_support):
            acc = [0.0] * n_rows
            for p in support:
                acc = list(map(add, acc, m[p]))
            a.append([w * v for v in acc])

        # --- symmetry check, then ridge ---------------------------------
        #
        # A = V C V^T with C symmetric is exactly symmetric in exact
        # arithmetic, so any asymmetry beyond rounding means M and A
        # disagree about which index is which. Read this guard for exactly
        # what it is: it catches an asymmetric CONSTRUCTION of A. It does
        # NOT catch a permuted pixel index, because C survives any pixel
        # permutation symmetric and so does V C V^T — verified by
        # mutation, see the module docstring. It has to run here, before
        # the symmetrization below, because after that there is nothing
        # left to measure: `cholesky_factor`'s own symmetry check sees
        # exactly 0.0 on everything this class hands it.
        scale = max((abs(v) for row in a for v in row), default=0.0)
        worst = 0.0
        for i in range(n_rows):
            for j in range(i):
                worst = max(worst, abs(a[i][j] - a[j][i]))
        self.asymmetry = worst / scale if scale > 0.0 else 0.0
        if self.asymmetry > 1e-7:
            raise IllConditionedError(
                f"V C V^T is asymmetric by a relative {self.asymmetry:.3e}, "
                f"far beyond rounding. The link/pixel index order is wrong."
            )

        diag = [a[i][i] for i in range(n_rows)]
        self.mean_diag_a = sum(diag) / len(diag) if diag else 0.0
        self.min_diag_a = min(diag) if diag else 0.0

        for i in range(n_rows):
            row_i = a[i]
            for j in range(i):
                mean = 0.5 * (row_i[j] + a[j][i])
                row_i[j] = mean
                a[j][i] = mean
            row_i[i] += alpha
        self._a_plus_alpha = a

        # --- Pi = M (A + alpha I)^-1, shape P x L ------------------------
        try:
            b_inv = cholesky_inverse(a)
        except NotPositiveDefiniteError as exc:
            # Reachable without any upstream bug: exactly duplicate rows
            # are gone, but near-duplicates are not, and an alpha far
            # enough below the diagonal vanishes into rounding. Say which,
            # with the numbers, rather than letting the reader follow
            # NotPositiveDefiniteError's advice and hunt an index bug that
            # is not there.
            ratio = alpha / self.mean_diag_a if self.mean_diag_a > 0.0 else float("inf")
            raise IllConditionedError(
                f"(V C V^T + alpha I) came out numerically indefinite: {exc}. "
                f"alpha = {alpha:g} is {ratio:.2e} of the mean diagonal "
                f"{self.mean_diag_a:.4g}, so the ridge is at or below the "
                f"rounding level of the matrix it is regularising. Raise "
                f"alpha, or lower sigma (only alpha/sigma^2 matters)."
            ) from exc

        #: max |B B^-1 - I| from the back-check, or None when
        #: `check_inverse=False` skipped it. None rather than NaN on
        #: purpose: every comparison against a NaN is False, so a caller
        #: testing `inverse_residual > tol` would read a skipped check as
        #: a healthy inverter, and `< tol` would read it as a sick one.
        #: None makes the same comparison raise TypeError.
        self.inverse_residual: float | None = (
            _max_inverse_residual(a, b_inv) if check_inverse else None
        )
        if self.inverse_residual is not None and self.inverse_residual > residual_tol:
            effective = alpha / self.prior.sigma2
            raise IllConditionedError(
                f"(V C V^T + alpha I) is too ill-conditioned to invert "
                f"reliably: max |B B^-1 - I| = {self.inverse_residual:.3e} "
                f"exceeds {residual_tol:.1e}. The knob is the EFFECTIVE ridge "
                f"alpha/sigma^2 = {effective:.3e} against a mean diagonal of "
                f"{self.mean_diag_a:.4g} (sigma and alpha act only through "
                f"that ratio, so raising alpha at a large sigma buys nothing "
                f"— lower sigma or raise alpha until the ratio moves). "
                f"delta_c = {self.delta_c:g} m is a second, weaker axis: a "
                f"correlation length above the room scale makes C nearly rank "
                f"one, though the residual is not monotone in it past ~8 m."
            )

        # B^-1 is symmetric, so its column k is its row k; no transpose.
        self.projection: list[list[float]] = [
            [_dot(m_p, b_inv[k]) for k in range(n_rows)] for m_p in m
        ]

    def solve(self, y: list[float]) -> list[float]:
        """x_hat = Pi r(y): the image, in canonical pixel order.

        `y` is in canonical LINK order, all `geometry.n_links` of them.
        The reduction r sums each group of duplicate-row links and divides
        by sqrt(group size), which is what makes the collapsed system the
        same estimator as the full one; with the usual reciprocal pairs it
        is the sum of the two directions over sqrt(2). Then one dense
        matrix-vector product, O(P*L), with no intermediate matrix.

        Non-finite inputs are rejected rather than propagated: a single
        NaN in y would otherwise turn the entire image to NaN and the
        tracker would report an abstention for a reason nobody could trace
        back to one bad link.
        """
        if len(y) != self.n_links:
            raise ValueError(
                f"y has {len(y)} entries, geometry has {self.n_links} links"
            )
        for i, v in enumerate(y):
            if not math.isfinite(v):
                raise ValueError(f"y[{i}] is {v}; link vector must be finite")
        get = y.__getitem__
        r = [
            s * sum(map(get, group))
            for s, group in zip(self._group_scale, self.link_groups)
        ]
        return [_dot(row, r) for row in self.projection]

    def forward(self, image: list[float]) -> list[float]:
        """y = W x. The forward model matching this inverter's geometry."""
        return forward_project(self.geometry, image)

    @property
    def projection_slots(self) -> int:
        """Pointer-array size of the cached projection: 8 bytes per entry.

        This is the figure `geometry.weight_matrix` reports for W, where
        it is also the whole story because every nonzero in a row of W is
        literally the same 1/sqrt(d_l) object. It is NOT the whole story
        here — see `projection_bytes`.
        """
        return 8 * self.n_pixels * self.n_independent_links

    @property
    def projection_bytes(self) -> int:
        """Resident size of the cached projection in CPython.

        32 bytes an entry: the 8-byte list slot PLUS the 24-byte float it
        points at. Unlike W, no two entries of the projection share a
        float object — each one comes out of its own `_dot` call —
        MEASURED at 10752 distinct objects for 10752 entries on the 8-node
        0.5 m deployment, zero sharing. Reporting only the slots, as an
        earlier version did, understated the real figure by 4.3x and sized
        the ESP32 conclusion against a number the process never occupies.

        A C or ESP32 port holding the same P*L values as float32 needs
        4 * P * L, an eighth of this; as float64, a quarter.
        """
        return 32 * self.n_pixels * self.n_independent_links

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Inverter(links={self.n_links} -> {self.n_independent_links} rows, "
            f"pixels={self.n_pixels}, alpha={self.alpha:g}, "
            f"sigma={self.sigma:g}, delta_c={self.delta_c:g} m, "
            f"mean_diag_A={self.mean_diag_a:.4g})"
        )


def _max_inverse_residual(
    a: list[list[float]], a_inv: list[list[float]]
) -> float:
    """max |A A^-1 - I|, the back-check on the explicit inverse.

    Costs O(L^3), which at L = 66 is about 2.9e5 multiply-adds — a few
    tens of milliseconds, paid once per `Inverter`. Worth it: it is the
    only direct evidence that the factorisation actually inverted the
    matrix rather than merely finishing without raising. For a
    well-conditioned A the residual sits near condition(A) * 2.2e-16.
    """
    n = len(a)
    worst = 0.0
    for i in range(n):
        ai = a[i]
        for j in range(n):
            # a_inv is symmetric, so its column j is its row j.
            v = _dot(ai, a_inv[j])
            worst = max(worst, abs(v - (1.0 if i == j else 0.0)))
    return worst
