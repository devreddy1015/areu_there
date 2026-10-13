"""Tests for `wifisense.localize.track`.

The load-bearing tests here are the abstention ones. A tracker that
reports a position for a motionless or absent person is the specific
failure this project exists to avoid, so "noise never produces a
confident fix" is tested across hundreds of seeds and against five noise
models, not with one hand-picked example.

Everything is compared against ground truth that is constructed here, not
against the module's own helpers, so a sign error or an index transposition
inside `track.py` cannot agree with itself and pass.

A note on test *power*, because several tests here were rewritten after a
mutation pass showed they had none. A test that only reads the module's
output in a scenario where the behaviour under test never fires proves
nothing, however green it is. Where a test is the sole coverage of a design
decision, it now states the mutant it kills:

    R inflation          test_weak_detections_are_trusted_less
    negative clamp       test_negative_ringing_does_not_drag_the_centroid
    room clamp           test_room_clamp_binds_on_an_emitted_frame
    Q cross terms        test_process_noise_matches_the_cwna_integral
    Joseph form          test_joseph_form_survives_a_vanishing_r
    pinned-block floor   TestDegenerateImages, throughout
    symmetric window     test_centroid_beats_nearest_pixel_against_the_wall
"""

from __future__ import annotations

import math
import random
import statistics
import unittest

from wifisense.localize.geometry import Geometry
from wifisense.localize.track import (
    DEFAULT_Z_THRESHOLD,
    PeakReading,
    PositionEstimate,
    Tracker,
    extract_peak,
)


def gaussian_blob(
    geometry: Geometry,
    cx: float,
    cy: float,
    sigma_m: float = 0.5,
    amplitude: float = 1.0,
    pedestal: float = 0.0,
) -> list[float]:
    """An image with a single Gaussian bump at (cx, cy).

    Written out here rather than imported so the tests have an independent
    notion of "the blob is at (cx, cy)".
    """
    two_s2 = 2.0 * sigma_m * sigma_m
    return [
        pedestal + amplitude * math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / two_s2)
        for x, y in geometry.grid.pixels
    ]


def iid_noise_image(geometry: Geometry, rng: random.Random) -> list[float]:
    return [rng.gauss(0.0, 1.0) for _ in range(geometry.n_pixels)]


def nearest_pixel(geometry: Geometry, reading: PeakReading) -> tuple[float, float]:
    """The peak pixel centre, with the sentinel contract enforced.

    `peak_index` is `None` when no peak could be located, and the point of
    using `None` rather than -1 is that this indexing raises instead of
    quietly returning the far corner of the room. Asserting it here pins
    that contract everywhere the baseline is used.
    """
    assert reading.peak_index is not None, "reading has no valid peak index"
    return geometry.grid.pixels[reading.peak_index]


def smoothing_kernel(geometry: Geometry, delta_c: float) -> list[tuple[list[tuple[int, float]], float]]:
    """Neighbour lists for an exp(-d/delta_c) smoother, precomputed once.

    Models what the inversion's spatial prior C_pq = s^2 exp(-dist/delta_c)
    does to the image: output pixels are correlated over `delta_c`. This is
    the realistic noise model, and it is the harder one — see the
    calibration note on `DEFAULT_Z_THRESHOLD`.
    """
    pixels = geometry.grid.pixels
    cutoff = 3.0 * delta_c
    kernel = []
    for xi, yi in pixels:
        neighbours = []
        for j, (xj, yj) in enumerate(pixels):
            d = math.hypot(xi - xj, yi - yj)
            if d <= cutoff:
                neighbours.append((j, math.exp(-d / delta_c)))
        norm = math.sqrt(sum(w * w for _, w in neighbours))
        kernel.append((neighbours, norm))
    return kernel


def smoothed_noise_image(kernel, n_pixels: int, rng: random.Random) -> list[float]:
    """Unit-variance spatially correlated noise."""
    raw = [rng.gauss(0.0, 1.0) for _ in range(n_pixels)]
    return [
        sum(w * raw[j] for j, w in neighbours) / norm if norm > 0.0 else 0.0
        for neighbours, norm in kernel
    ]


class TestAbstention(unittest.TestCase):
    """The guarantee the module exists for."""

    def setUp(self) -> None:
        self.geometry = Geometry.perimeter(8, pitch_m=0.5)

    def test_flat_image_abstains(self) -> None:
        """Zero spread means zero information, whatever the level.

        1e308 is in the sweep on purpose. `_median` used to average two
        values as `0.5 * (a + b)`, which overflows to inf for any level
        above ~9e307, making the background inf, the spread inf and the
        z-score inf/inf = nan — and `nan < threshold` is False, so the
        frame was ACCEPTED with confidence 1.0. A flat image is the most
        information-free input there is; it must never produce a fix at any
        magnitude the float type can hold.
        """
        for level in (0.0, 1.0, -3.5, 1e9, 1e308, -1e308, 8.99e307):
            with self.subTest(level=level):
                tracker = Tracker(self.geometry)
                image = [level] * self.geometry.n_pixels
                self.assertIsNone(tracker.update(image, 0.0))
                self.assertFalse(tracker.is_tracking)
                reading = extract_peak(self.geometry, image)
                self.assertTrue(math.isfinite(reading.background))
                self.assertEqual(reading.z, 0.0)

    def test_nonfinite_image_abstains(self) -> None:
        """A diverged inversion must not yield a position."""
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(bad=bad):
                image = gaussian_blob(self.geometry, 4.0, 3.0)
                image[17] = bad
                reading = extract_peak(self.geometry, image)
                self.assertEqual(reading.z, 0.0)
                self.assertEqual(reading.n_nonfinite, 1)
                self.assertTrue(math.isnan(reading.x))
                tracker = Tracker(self.geometry)
                self.assertIsNone(tracker.update(image, 0.0))

    def test_no_peak_index_is_none_not_a_valid_index(self) -> None:
        """-1 is a valid Python index; the sentinel must not be one.

        With `peak_index = -1` the idiom this file itself uses,
        `grid.pixels[reading.peak_index]`, returned (7.75, 5.75) — a
        well-formed in-room position in the far corner — for an image that
        had diverged. Silently plausible is the worst available failure.
        """
        image = gaussian_blob(self.geometry, 4.0, 3.0)
        image[17] = float("nan")
        reading = extract_peak(self.geometry, image)
        self.assertIsNone(reading.peak_index)
        with self.assertRaises(TypeError):
            self.geometry.grid.pixels[reading.peak_index]  # type: ignore[index]

    def test_iid_noise_never_yields_a_confident_fix(self) -> None:
        """400 independent pure-noise images, one fresh tracker each.

        Measured false-positive rate for iid noise at threshold 5.0 is
        2e-4 per image, so 400 trials should normally produce none at all.
        The assertion is deliberately stated as a rate rather than "zero",
        because a threshold test on random data has a tail by construction
        and a test that forbids the tail is a test that will one day be
        silenced rather than believed. What is asserted absolutely is the
        property that matters: nothing that slips through is CONFIDENT.
        """
        fixes = []
        trials = 400
        for seed in range(trials):
            rng = random.Random(20000 + seed)
            image = iid_noise_image(self.geometry, rng)
            tracker = Tracker(self.geometry)
            estimate = tracker.update(image, 0.0)
            if estimate is not None:
                fixes.append(estimate)

        self.assertLessEqual(
            len(fixes),
            trials // 50,
            f"{len(fixes)}/{trials} pure-noise images produced a fix",
        )
        for estimate in fixes:
            # 0.40 is where types.VitalEstimate already grades an estimate
            # UNRELIABLE, so a leaker below it is one the rest of the repo
            # would refuse to act on.
            self.assertLess(
                estimate.confidence,
                0.40,
                f"noise produced a fix with confidence {estimate.confidence:.3f}",
            )

    def test_spatially_smoothed_noise_never_yields_a_confident_fix(self) -> None:
        """The realistic case: noise correlated over the prior length.

        Smoothing makes this harder, not easier — the MAD in the z-score's
        denominator is estimated from the same correlated pixels and so
        fluctuates much more than in the iid case.
        """
        for delta_c in (0.5, 1.0):
            with self.subTest(delta_c=delta_c):
                kernel = smoothing_kernel(self.geometry, delta_c)
                fixes = []
                trials = 200
                for seed in range(trials):
                    rng = random.Random(70000 + 1000 * int(delta_c * 10) + seed)
                    image = smoothed_noise_image(kernel, self.geometry.n_pixels, rng)
                    tracker = Tracker(self.geometry)
                    estimate = tracker.update(image, 0.0)
                    if estimate is not None:
                        fixes.append(estimate)

                self.assertLessEqual(
                    len(fixes),
                    trials // 20,
                    f"{len(fixes)}/{trials} smoothed-noise images produced a fix",
                )
                for estimate in fixes:
                    self.assertLess(estimate.confidence, 0.40)

    def test_noise_stream_never_builds_a_sustained_track(self) -> None:
        """A continuous stream of noise must not accumulate into a track.

        The single-frame tests above cannot catch a filter that latches:
        one lucky frame initializes it and the coast then keeps it alive.
        The coast expiry is what prevents that, and this checks it end to
        end over a 200 s run.
        """
        rng = random.Random(4242)
        tracker = Tracker(self.geometry)
        dt = 0.4
        fixes = 0
        consecutive = 0
        worst_consecutive = 0
        for i in range(500):
            image = iid_noise_image(self.geometry, rng)
            estimate = tracker.update(image, i * dt)
            if estimate is None:
                consecutive = 0
            else:
                fixes += 1
                consecutive += 1
                worst_consecutive = max(worst_consecutive, consecutive)
                self.assertLess(estimate.confidence, 0.40)

        self.assertLessEqual(fixes, 10, f"{fixes}/500 noise frames produced a fix")
        self.assertLessEqual(
            worst_consecutive, 2, "noise sustained a multi-frame track"
        )
        self.assertFalse(
            tracker.is_tracking, "tracker still holds a track after 200 s of noise"
        )

    def test_a_real_blob_is_not_abstained_on(self) -> None:
        """The counterweight: the threshold must not reject everything.

        An abstention test suite that only checks abstention is passed by
        `return None`, so this pins the other side.
        """
        tracker = Tracker(self.geometry)
        image = gaussian_blob(self.geometry, 4.2, 3.3, sigma_m=0.5)
        estimate = tracker.update(image, 0.0)
        self.assertIsNotNone(estimate)
        assert estimate is not None
        self.assertGreater(estimate.confidence, 0.9)

    def test_blob_buried_in_noise_degrades_gracefully(self) -> None:
        """Sweep SNR: confidence should fall with it, not cliff to a lie."""
        previous = 1.1
        for amplitude in (8.0, 4.0, 2.0, 1.0, 0.5, 0.25):
            rng = random.Random(99)
            image = gaussian_blob(self.geometry, 4.0, 3.0, amplitude=amplitude)
            image = [v + rng.gauss(0.0, 1.0) for v in image]
            tracker = Tracker(self.geometry)
            estimate = tracker.update(image, 0.0)
            confidence = 0.0 if estimate is None else estimate.confidence
            self.assertLessEqual(
                confidence, previous + 1e-9, f"confidence rose as SNR fell (a={amplitude})"
            )
            previous = confidence

    def test_broad_targets_suppress_their_own_significance(self) -> None:
        """Pin the measured sensitivity limit, not the textbook one.

        The MAD's classical 50% breakdown point is not the operating limit:
        the target's own pixels inflate the denominator long before they
        dominate it, so sensitivity dies near 20% coverage. That is a real
        limitation of this estimator and it is documented in `extract_peak`;
        this test keeps the documentation and the code from drifting apart.
        """
        measured = []
        for blob_sigma in (1.2, 1.6, 2.0):
            image = gaussian_blob(self.geometry, 4.0, 3.0, sigma_m=blob_sigma)
            peak = max(image)
            coverage = sum(1 for v in image if v > 0.5 * peak) / len(image)
            reading = extract_peak(self.geometry, image)
            measured.append((blob_sigma, coverage, reading.z))

        # Coverage rises and significance falls with it, monotonically.
        for (_, cov_a, z_a), (_, cov_b, z_b) in zip(measured, measured[1:]):
            self.assertGreater(cov_b, cov_a)
            self.assertLess(z_b, z_a)
        # The documented cliff: detected at 12% coverage, gone by 27%.
        self.assertGreater(measured[0][2], DEFAULT_Z_THRESHOLD)
        self.assertLess(measured[1][2], DEFAULT_Z_THRESHOLD)
        self.assertLess(measured[0][1], 0.20)
        self.assertGreater(measured[1][1], 0.20)


class TestDegenerateImages(unittest.TestCase):
    """Images whose own noise floor cannot be measured.

    Every case here used to produce a maximally confident position from an
    image carrying no localizable signal — the exact failure the module's
    docstring says it exists to prevent. The common mechanism is a block of
    pixels pinned at one identical value: it drives the MAD toward zero,
    and z = (peak - background) / spread then explodes. Each test states
    how a real caller reaches the input.
    """

    def setUp(self) -> None:
        self.geometry = Geometry.perimeter(8, pitch_m=0.5)

    def test_single_spike_on_a_flat_image_abstains(self) -> None:
        """One pixel above an otherwise flat image carries no information.

        The old fallback divided by a mean absolute deviation, which for a
        one-spike image is itself proportional to the spike over the pixel
        count — so z came out at a constant 153.2 (confidence 0.967) for a
        spike of 1e-300, 1.0 or 1e9 alike, and the constant GREW with the
        grid size. An image numerically indistinguishable from the flat one
        that `test_flat_image_abstains` requires to return None scored a
        near-certain position.
        """
        for pitch, n_expected in ((0.5, 192), (0.25, 768)):
            geometry = Geometry.perimeter(8, pitch_m=pitch)
            self.assertEqual(geometry.n_pixels, n_expected)
            for spike in (1e-300, 1e-12, 1.0, 1e9):
                with self.subTest(pitch=pitch, spike=spike):
                    image = [0.0] * geometry.n_pixels
                    image[100] = spike
                    reading = extract_peak(geometry, image)
                    self.assertEqual(reading.z, 0.0, f"z={reading.z} on a lone spike")
                    self.assertGreater(reading.tie_fraction, 0.9)
                    self.assertIsNone(Tracker(geometry).update(image, 0.0))

    def test_rectified_noise_abstains(self) -> None:
        """`max(v, 0.0)` is the standard RTI non-negativity clamp.

        Attenuation cannot be negative, so rectifying the image is a
        natural thing for a caller to do — the module itself clamps
        negatives inside the centroid. It leaves half the pixels at exactly
        zero, which used to collapse the MAD: 278/300 seeds produced a fix
        and 162 of them exceeded confidence 0.40.
        """
        fixes = []
        trials = 300
        for seed in range(trials):
            rng = random.Random(90000 + seed)
            image = [max(0.0, rng.gauss(0.0, 1.0)) for _ in range(self.geometry.n_pixels)]
            estimate = Tracker(self.geometry).update(image, 0.0)
            if estimate is not None:
                fixes.append(estimate)
        self.assertLessEqual(
            len(fixes), trials // 50, f"{len(fixes)}/{trials} rectified-noise fixes"
        )
        for estimate in fixes:
            self.assertLess(estimate.confidence, 0.40)

    def test_half_the_pixels_zeroed_abstains(self) -> None:
        """A tie block anywhere in the sorted order, not just at the floor.

        Half the pixels zeroed and half pure N(0,1), shuffled, so the tie
        block sits in the middle of the sorted deviations. The MAD then
        lands on the smallest free deviation rather than inside the block,
        which is how this case slipped past a guard written as
        `spread <= 0.0`: the measured spread was 1.41e-05 against a true
        noise sd of 1.0, a 70,000x under-estimate, and z reached 180925.
        300/300 seeds produced a fix with confidence >= 0.40.
        """
        fixes = []
        trials = 300
        for seed in range(trials):
            rng = random.Random(40000 + seed)
            half = self.geometry.n_pixels // 2
            image = [0.0] * half + [rng.gauss(0.0, 1.0) for _ in range(half)]
            rng.shuffle(image)
            reading = extract_peak(self.geometry, image)
            # The floor estimate must be within a factor of two of the truth.
            self.assertGreater(reading.spread, 0.5)
            self.assertLess(reading.spread, 2.0)
            estimate = Tracker(self.geometry).update(image, 0.0)
            if estimate is not None:
                fixes.append(estimate)
        self.assertLessEqual(len(fixes), trials // 50, f"{len(fixes)}/{trials} fixes")
        for estimate in fixes:
            self.assertLess(estimate.confidence, 0.40)

    def test_quantized_noise_abstains(self) -> None:
        """Coarse quantization pins most pixels in one bin.

        N(0,1) quantized to a step of 4 sd. Previously 1974/2000 seeds
        produced a fix and 1971 of them were confident.
        """
        fixes = []
        trials = 300
        for seed in range(trials):
            rng = random.Random(5000 + seed)
            image = [
                4.0 * round(rng.gauss(0.0, 1.0) / 4.0)
                for _ in range(self.geometry.n_pixels)
            ]
            estimate = Tracker(self.geometry).update(image, 0.0)
            if estimate is not None:
                fixes.append(estimate)
        self.assertLessEqual(
            len(fixes), trials // 50, f"{len(fixes)}/{trials} quantized-noise fixes"
        )
        for estimate in fixes:
            self.assertLess(estimate.confidence, 0.40)

    def test_near_degenerate_spread_is_not_significance(self) -> None:
        """A deterministic witness, so this cannot drift with an RNG.

        96 zeros, one pixel at 1e-9, 94 at 1.0 and one at 2.0. The old
        median averaged across the tie boundary to a background of 5e-10,
        the MAD collapsed to 7.4e-10 and z came out at 2.7e9 — confidence
        0.999999998 for an image whose only outlier is 2x the bulk.
        """
        image = [0.0] * 96 + [1e-9] + [1.0] * 94 + [2.0]
        self.assertEqual(len(image), self.geometry.n_pixels)
        reading = extract_peak(self.geometry, image)
        # The floor is the spacing of the bulk, about 1.0, not 1e-9.
        self.assertGreater(reading.spread, 0.5)
        self.assertLess(reading.z, DEFAULT_Z_THRESHOLD)
        self.assertIsNone(Tracker(self.geometry).update(image, 0.0))

    def test_z_responds_to_spike_height_when_there_is_a_floor(self) -> None:
        """The other half of the contract: z must still mean something.

        The degenerate-image tests above are all passed by `return 0.0`, so
        this pins that z tracks the signal whenever a floor exists at all.
        """
        rng = random.Random(7)
        floor = [rng.gauss(0.0, 1.0) for _ in range(self.geometry.n_pixels)]
        previous = 0.0
        for height in (3.0, 6.0, 12.0, 24.0):
            image = list(floor)
            image[100] = height
            reading = extract_peak(self.geometry, image)
            self.assertGreater(reading.z, previous)
            previous = reading.z
        self.assertGreater(previous, DEFAULT_Z_THRESHOLD)

    def test_background_of_huge_values_does_not_overflow(self) -> None:
        """The median of an even-length list must not add two big floats.

        `0.5 * (a + b)` overflows to inf once a + b passes 1.8e308, which
        made the background inf, every deviation inf and the z-score
        inf/inf = nan. Halving first is exact and cannot overflow. Checked
        against a midpoint computed here, not against the module's.
        """
        image = [1.0e308 + i * 3.0e305 for i in range(self.geometry.n_pixels)]
        lo, hi = image[95], image[96]
        reading = extract_peak(self.geometry, image)
        self.assertTrue(math.isfinite(reading.background))
        self.assertAlmostEqual(
            reading.background / 1e308, (0.5 * lo + 0.5 * hi) / 1e308, places=9
        )
        self.assertTrue(math.isfinite(reading.spread))
        self.assertLess(reading.z, DEFAULT_Z_THRESHOLD)

    def test_a_centroid_that_overflows_is_not_emitted(self) -> None:
        """A finite image can still produce a non-finite centroid.

        The weighted sums are products of pixel values and coordinates, so
        values near the top of the float range overflow them and the
        centroid comes back NaN while the image itself is entirely finite —
        `extract_peak`'s non-finite-image guard never sees it, and the
        z-score is enormous rather than zero. Without a finiteness test on
        the measurement the filter initialises at NaN, and `clamp_to_room`
        does not rescue it: min/max propagate NaN rather than clipping it.
        """
        rng = random.Random(2)
        image = [0.0] * 100 + [abs(rng.gauss(0.0, 1.0)) for _ in range(92)]
        for index in (191, 190, 189, 175, 159):
            image[index] = 1.7e308
        reading = extract_peak(self.geometry, image)
        self.assertTrue(all(math.isfinite(v) for v in image))
        self.assertFalse(math.isfinite(reading.x))
        self.assertGreater(reading.z, DEFAULT_Z_THRESHOLD)

        tracker = Tracker(self.geometry)
        self.assertIsNone(tracker.update(image, 0.0))
        self.assertFalse(tracker.is_tracking)

    def test_mixed_extremes_do_not_produce_a_position(self) -> None:
        """Half at -1e308, half at +1e308: no floor, and no NaN either.

        This used to return x=nan, y=nan with confidence 1.0 and then
        poison the filter permanently — 60 subsequent clean blobs all came
        back as NaN while sigma_m SHRANK, the filter growing steadily more
        certain about nothing.
        """
        tracker = Tracker(self.geometry)
        image = [-1e308] * 97 + [1e308] * 95
        self.assertIsNone(tracker.update(image, 0.0))
        self.assertFalse(tracker.is_tracking)

        # And recovery is immediate: the poison frame left nothing behind.
        for i in range(1, 6):
            estimate = tracker.update(
                gaussian_blob(self.geometry, 4.0, 3.0), i * 0.4
            )
            self.assertIsNotNone(estimate)
            assert estimate is not None
            self.assertTrue(math.isfinite(estimate.x))
            self.assertTrue(math.isfinite(estimate.y))
        assert estimate is not None
        self.assertLess(math.hypot(estimate.x - 4.0, estimate.y - 3.0), 0.35)

    def test_a_clamped_image_with_a_real_target_still_detects(self) -> None:
        """Abstaining on pinned images must not cost real detections.

        The pinned-block handling would be easy to make safe and useless.
        A rectified image containing an actual blob has to come through,
        because rectification is a thing callers do to real images.
        """
        rng = random.Random(1234)
        detections = 0
        trials = 20
        errors = []
        for _ in range(trials):
            cx = rng.uniform(1.5, 6.5)
            cy = rng.uniform(1.5, 4.5)
            image = gaussian_blob(self.geometry, cx, cy, sigma_m=0.5, amplitude=4.0)
            image = [max(0.0, v + rng.gauss(0.0, 0.5)) for v in image]
            estimate = Tracker(self.geometry).update(image, 0.0)
            if estimate is not None:
                detections += 1
                errors.append(math.hypot(estimate.x - cx, estimate.y - cy))
        self.assertGreaterEqual(detections, trials - 1, "clamping killed real targets")
        self.assertLess(statistics.median(errors), 0.5)


class TestSubPixelCentroid(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = Geometry.perimeter(8, pitch_m=0.5)

    def test_beats_nearest_pixel_across_a_whole_cell(self) -> None:
        """Scan the true position across one pixel cell and compare.

        Nearest-pixel error averages a quarter of the diagonal over a cell;
        the centroid has to beat that by a wide margin to be worth having.
        """
        grid = self.geometry.grid
        pitch = grid.pitch_x_m
        centroid_errors = []
        nearest_errors = []
        for i in range(10):
            for j in range(10):
                cx = 4.0 + i * pitch / 10.0
                cy = 3.0 + j * pitch / 10.0
                image = gaussian_blob(self.geometry, cx, cy, sigma_m=0.5)
                reading = extract_peak(self.geometry, image)
                centroid_errors.append(math.hypot(reading.x - cx, reading.y - cy))
                nx, ny = nearest_pixel(self.geometry, reading)
                nearest_errors.append(math.hypot(nx - cx, ny - cy))

        centroid_mean = statistics.fmean(centroid_errors)
        nearest_mean = statistics.fmean(nearest_errors)
        self.assertLess(centroid_mean, nearest_mean / 3.0)
        # The headline claim: resolution is not quantised to the grid.
        self.assertLess(max(centroid_errors), 0.25 * pitch)

    def test_beats_nearest_pixel_at_a_finer_pitch(self) -> None:
        """Same claim at 0.25 m, where the grid is already fairly fine."""
        geometry = Geometry.perimeter(8, pitch_m=0.25)
        grid = geometry.grid
        centroid_errors = []
        nearest_errors = []
        for i in range(8):
            cx = 4.0 + i * grid.pitch_x_m / 8.0
            cy = 3.0 + i * grid.pitch_y_m / 8.0
            image = gaussian_blob(geometry, cx, cy, sigma_m=0.4)
            reading = extract_peak(geometry, image)
            centroid_errors.append(math.hypot(reading.x - cx, reading.y - cy))
            nx, ny = nearest_pixel(geometry, reading)
            nearest_errors.append(math.hypot(nx - cx, ny - cy))
        self.assertLess(statistics.fmean(centroid_errors), statistics.fmean(nearest_errors))

    def test_centroid_beats_nearest_pixel_against_the_wall(self) -> None:
        """The band no other test covered, and the one that used to lose.

        Clipping the neighbourhood at the grid edge leaves a one-sided
        window whose centroid is dragged inward, toward the room centre.
        MEASURED with the clipped window, mean error over the outermost
        pixel row against nearest-pixel's: 0.2568 vs 0.1906 at 0.5 m pitch,
        0.3070 vs 0.0953 at 0.25 m — losing by up to 3x, and losing by MORE
        on the finer grid. Shrinking the window symmetrically instead gives
        0.1260 and 0.0639.

        The requirement is the spec's: the sub-pixel centroid must beat the
        nearest-pixel baseline it exists to improve on, everywhere in the
        room and not merely two cells in from the wall.
        """
        for pitch in (0.5, 0.25):
            geometry = Geometry.perimeter(8, pitch_m=pitch)
            grid = geometry.grid
            for row in (0, 1):
                with self.subTest(pitch=pitch, row=row):
                    y0 = grid.pixels[row * grid.n_x][1]
                    centroid_errors = []
                    nearest_errors = []
                    for k in range(8):
                        for j in range(8):
                            cy = y0 - pitch / 2.0 + (k + 0.5) * pitch / 8.0
                            cx = 4.0 - pitch / 2.0 + (j + 0.5) * pitch / 8.0
                            image = gaussian_blob(geometry, cx, cy, sigma_m=0.5)
                            reading = extract_peak(geometry, image)
                            centroid_errors.append(
                                math.hypot(reading.x - cx, reading.y - cy)
                            )
                            nx, ny = nearest_pixel(geometry, reading)
                            nearest_errors.append(math.hypot(nx - cx, ny - cy))
                    centroid_mean = statistics.fmean(centroid_errors)
                    nearest_mean = statistics.fmean(nearest_errors)
                    self.assertLess(
                        centroid_mean,
                        nearest_mean,
                        f"centroid {centroid_mean:.4f} loses to nearest-pixel "
                        f"{nearest_mean:.4f} in row {row} at pitch {pitch}",
                    )

    def test_blob_on_the_corner_pixel_centre_is_exact(self) -> None:
        """Zero pixel-locking bias by construction, so all error is window.

        A blob sitting exactly on the corner pixel's centre has no sub-pixel
        offset to estimate, so a window that is symmetric about the peak
        pixel must return that centre exactly. The clipped window returned
        (0.502, 0.502) against a truth of (0.250, 0.250) — 0.356 m of pure
        inward drag, where the trivial nearest-pixel baseline scored 0.000.
        """
        for pitch in (0.5, 0.25):
            with self.subTest(pitch=pitch):
                geometry = Geometry.perimeter(8, pitch_m=pitch)
                for corner in (0, geometry.grid.n_x - 1, -1):
                    px, py = geometry.grid.pixels[corner]
                    image = gaussian_blob(geometry, px, py, sigma_m=0.5)
                    reading = extract_peak(geometry, image)
                    self.assertAlmostEqual(reading.x, px, delta=1e-9)
                    self.assertAlmostEqual(reading.y, py, delta=1e-9)

    def test_blob_exactly_between_two_pixel_centres(self) -> None:
        """The case nearest-pixel cannot get right even in principle.

        Placed on the midpoint between two columns, the peak pixel is a
        coin flip and is wrong by half a pitch either way. The centroid
        lands within a few percent of a pitch — not exactly on the
        midpoint, and the residual error is systematic rather than random:
        see `test_pixel_locking_bias_is_bounded` for what it is and why.
        """
        grid = self.geometry.grid
        left = grid.pixels[grid.pixel_index(7, 6)]
        right = grid.pixels[grid.pixel_index(8, 6)]
        cx = 0.5 * (left[0] + right[0])
        cy = left[1]
        image = gaussian_blob(self.geometry, cx, cy, sigma_m=0.5)
        reading = extract_peak(self.geometry, image)
        self.assertAlmostEqual(reading.x, cx, delta=0.05 * grid.pitch_x_m)
        self.assertAlmostEqual(reading.y, cy, delta=0.05 * grid.pitch_y_m)
        nearest_x = nearest_pixel(self.geometry, reading)[0]
        self.assertAlmostEqual(abs(nearest_x - cx), 0.5 * grid.pitch_x_m, delta=1e-9)

    def test_pixel_locking_bias_is_bounded(self) -> None:
        """Characterise the centroid's one systematic error.

        The neighbourhood is centred on the peak PIXEL, not on the blob, so
        once the blob sits off-centre the window is lopsided and the
        centroid is dragged back toward the peak pixel. The signature is
        the classic pixel-locking S-curve: exactly zero when the blob sits
        on a pixel centre, growing to a maximum at the midpoint between
        two centres, and flipping sign there as the peak pixel changes
        hands.

        This is a bias, not noise — it does not average away over repeated
        measurements at one spot — so it is worth bounding explicitly
        rather than leaving it inside an aggregate error figure.

        MEASURED worst |bias| scanning across one cell: 0.0223 m at 0.5 m
        pitch (4.5% of a pitch), 0.0184 m at 0.25 m pitch (7.3%).
        """
        for pitch, start, limit in ((0.5, 3.75, 0.05), (0.25, 3.875, 0.08)):
            with self.subTest(pitch=pitch):
                geometry = Geometry.perimeter(8, pitch_m=pitch)
                biases = []
                for k in range(21):
                    cx = start + (k / 20.0) * pitch
                    reading = extract_peak(
                        geometry, gaussian_blob(geometry, cx, 3.125, sigma_m=0.5)
                    )
                    biases.append(reading.x - cx)

                worst = max(abs(b) for b in biases)
                self.assertLess(worst, limit * pitch)
                # Zero at both pixel centres, where the window is symmetric.
                self.assertAlmostEqual(biases[0], 0.0, delta=1e-6)
                self.assertAlmostEqual(biases[-1], 0.0, delta=1e-6)
                # Sign flips across the midpoint.
                self.assertLess(biases[5], 0.0)
                self.assertGreater(biases[15], 0.0)

    def test_not_pulled_towards_the_room_centre(self) -> None:
        """Guards the failure mode P0B_TASK warns about explicitly.

        A tracker that quietly regresses to the room centre scores well on
        a random walk purely because the centre is a good guess. A corner
        blob must stay in its corner.

        The close-in targets are the point. With the window clipped at the
        grid edge the errors here were 0.3149 m at (0.30, 0.30), 0.3561 at
        (0.25, 0.25) and 0.4896 at (0.10, 0.10) — all failing the 0.3 m
        bound — while (0.6, 0.6) scraped through at 0.1455 and made the
        test look like coverage it was not.
        """
        room_cx, room_cy = self.geometry.room.center
        targets = (
            (1.0, 1.0),
            (7.0, 1.0),
            (1.0, 5.0),
            (7.0, 5.0),
            (0.6, 0.6),
            (0.30, 0.30),
            (0.25, 0.25),
            (0.10, 0.10),
            (7.9, 5.9),
        )
        for cx, cy in targets:
            with self.subTest(target=(cx, cy)):
                image = gaussian_blob(self.geometry, cx, cy, sigma_m=0.5)
                reading = extract_peak(self.geometry, image)
                error = math.hypot(reading.x - cx, reading.y - cy)
                self.assertLess(error, 0.3)
                # And strictly closer to the target than to the centre.
                to_centre = math.hypot(reading.x - room_cx, reading.y - room_cy)
                self.assertLess(error, to_centre)

    def _wall_walk_bias(self, wall_y: float) -> float:
        """Signed mean y-error of a straight walk at a fixed distance in."""
        tracker = Tracker(self.geometry, meas_noise_m=0.25)
        biases = []
        for i in range(26):
            t = i * 0.4
            cx = 1.0 + 0.6 * t
            if cx > 7.0:
                break
            estimate = tracker.update(
                gaussian_blob(self.geometry, cx, wall_y, sigma_m=0.5), t
            )
            if estimate is not None and i >= 5:
                biases.append(estimate.y - wall_y)
        self.assertGreater(len(biases), 15)
        return statistics.fmean(biases)

    def test_wall_walk_has_no_systematic_inward_bias(self) -> None:
        """End to end: a bias does not average out, so measure its sign.

        Aggregate error bounds hide a bias, because it is small beside the
        per-frame spread; the signed mean is what exposes it. MEASURED on
        the raw centroid along wall-hugging lines in the outermost pixel
        row (whose centre is y = 0.25 m), clipped window vs symmetric vs
        the nearest-pixel baseline:

            y = 0.25   +0.2518   0.0000   0.0000
            y = 0.40   +0.1676  -0.1500  -0.1500
            y = 0.10   +0.3462  +0.1500  +0.1500

        The clipped window pulls INWARD at every y, which is the artifact
        that lets a tracker score well on a random walk without knowing
        anything. The symmetric window carries exactly the nearest-pixel
        quantization bias in that row and nothing on top of it, so its sign
        follows the grid rather than the room centre and it averages to zero
        across the row.
        """
        on_centre = self._wall_walk_bias(0.25)
        self.assertLess(
            abs(on_centre),
            0.01,
            f"a walk along the row-0 pixel centre is biased by {on_centre:+.4f} m",
        )

        # Half a pitch either side of that centre, the only residual is the
        # quantization of a one-pixel-tall window — equal and opposite.
        above = self._wall_walk_bias(0.40)
        below = self._wall_walk_bias(0.10)
        self.assertAlmostEqual(above, -0.15, delta=0.02)
        self.assertAlmostEqual(below, +0.15, delta=0.02)
        self.assertLess(
            abs(above + below),
            0.02,
            "the outermost row has a net bias, not just grid quantization",
        )

    def test_negative_ringing_does_not_drag_the_centroid(self) -> None:
        """Regularized inversions ring negative around a positive blob.

        Unclamped, those negative pixels enter the weighted sum as negative
        mass and drag the centroid off the target.

        The tolerance is 0.05 m, not 0.2 m, and that is the whole test.
        MEASURED on this exact image: clamped 0.0180 m, unclamped 0.1882 m.
        At 0.2 m the unclamped version passed too, so deleting the clamp
        (`if w <= 0.0: continue`) left the entire suite green.
        """
        cx, cy = 4.2, 3.2
        image = gaussian_blob(self.geometry, cx, cy, sigma_m=0.4)
        for i, (x, y) in enumerate(self.geometry.grid.pixels):
            d = math.hypot(x - cx, y - cy)
            if 0.8 < d < 1.8:
                image[i] -= 0.4
        reading = extract_peak(self.geometry, image)
        self.assertLess(math.hypot(reading.x - cx, reading.y - cy), 0.05)
        self.assertTrue(self.geometry.room.contains(reading.x, reading.y))

    def test_pedestal_does_not_destroy_resolution(self) -> None:
        """A constant offset is background, and must be subtracted out.

        The tolerance is 1e-6 m rather than bit-equality. Adding 1000 to a
        blob whose tail runs down to 1e-22 destroys that tail below float
        resolution — 14.6% of the pixels land on exactly 1000.0 — so the
        two images are genuinely not the same data any more and asking for
        agreement at 5e-10 m was asking about floats, not about the
        estimator. 1e-6 m is 2e-6 of a pitch, and MEASURED, dropping the
        background subtraction entirely moves this centroid by 0.0707 m, so
        the assertion still has five orders of magnitude of headroom over
        the mutant it is there to catch.
        """
        cx, cy = 4.2, 3.3
        plain = extract_peak(self.geometry, gaussian_blob(self.geometry, cx, cy))
        raised = extract_peak(
            self.geometry, gaussian_blob(self.geometry, cx, cy, pedestal=1000.0)
        )
        self.assertAlmostEqual(plain.x, raised.x, delta=1e-6)
        self.assertAlmostEqual(plain.y, raised.y, delta=1e-6)
        # Both must still be near the truth; a centroid pinned to the peak
        # pixel would sit on a cell centre and satisfy the equality above.
        self.assertLess(math.hypot(raised.x - cx, raised.y - cy), 0.05)


class TestKalmanTracking(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = Geometry.perimeter(8, pitch_m=0.5)
        self.dt = 0.4  # 8 nodes x 50 ms TDMA slots -> 2.5 Hz per link

    def _walk(
        self,
        seed: int,
        pos_noise_m: float,
        n_steps: int = 60,
        speed: float = 0.6,
        dt: float | None = None,
        **tracker_kwargs,
    ) -> tuple[list[float], list[float]]:
        """Constant-velocity walk; returns (raw errors, tracked errors).

        The per-frame measurement noise is injected by displacing the blob
        CENTRE, which lumps together everything that makes a real image
        peak land off-target. `meas_noise_m` is set to the injected level,
        because the claim under test is that the filter smooths when told
        the truth about its measurements, not that the default guess is right.
        """
        dt = self.dt if dt is None else dt
        rng = random.Random(seed)
        tracker = Tracker(self.geometry, meas_noise_m=pos_noise_m, **tracker_kwargs)
        raw_errors: list[float] = []
        tracked_errors: list[float] = []
        # The walk is parameterised by displacement PER FRAME, not per
        # second, so sweeping `dt` changes the sampling rate and not how far
        # across the room the walker gets before running out of floor.
        speed = speed * self.dt / dt
        vx, vy = speed * 0.894, speed * 0.447  # unit-ish direction
        for i in range(n_steps):
            t = i * dt
            cx = 1.0 + vx * t
            cy = 1.0 + vy * t
            if not self.geometry.room.contains(cx, cy, tol=-1.0):
                break
            mx = cx + rng.gauss(0.0, pos_noise_m)
            my = cy + rng.gauss(0.0, pos_noise_m)
            image = gaussian_blob(self.geometry, mx, my, sigma_m=0.5)
            reading = extract_peak(self.geometry, image)
            estimate = tracker.update(image, t)
            self.assertIsNotNone(estimate)
            assert estimate is not None
            raw_errors.append(math.hypot(reading.x - cx, reading.y - cy))
            tracked_errors.append(math.hypot(estimate.x - cx, estimate.y - cy))
        return raw_errors, tracked_errors

    def test_filter_smooths_noisy_measurements(self) -> None:
        """The central claim: tracked error beats raw measurement error.

        Pooled over seeds, and also required to hold for most seeds
        individually — a filter that helps on average while wrecking one
        run in three is not a filter anyone should ship.

        Samples before index 5 are dropped. That is the acquisition
        transient: the filter starts with zero velocity and a 1 m/s prior,
        so it needs a few measurements before the velocity estimate means
        anything, and including that window measures initialization rather
        than smoothing.
        """
        for pos_noise in (0.15, 0.25, 0.40):
            with self.subTest(pos_noise=pos_noise):
                pooled_raw: list[float] = []
                pooled_tracked: list[float] = []
                wins = 0
                seeds = range(6)
                for seed in seeds:
                    raw, tracked = self._walk(seed, pos_noise)
                    raw, tracked = raw[5:], tracked[5:]
                    self.assertGreater(len(raw), 15)
                    pooled_raw.extend(raw)
                    pooled_tracked.extend(tracked)
                    if statistics.fmean(tracked) < statistics.fmean(raw):
                        wins += 1

                raw_mean = statistics.fmean(pooled_raw)
                tracked_mean = statistics.fmean(pooled_tracked)
                self.assertLess(
                    tracked_mean,
                    raw_mean,
                    f"pooled: tracked {tracked_mean:.4f} >= raw {raw_mean:.4f}",
                )
                self.assertGreaterEqual(wins, len(seeds) - 1)

    def test_filter_still_smooths_at_any_max_coast_s(self) -> None:
        """An unbroken stream of fixes is never a gap, however slow.

        The coast expiry used to be evaluated against `_last_fix_t`, which
        is always at least one frame stale, so it measured the inter-frame
        interval instead of an abstention gap. Once the frame interval
        exceeded `max_coast_s` the state was discarded and re-initialised on
        every single frame although nothing had ever abstained: `update`
        returned the raw centroid, `sigma_m` pinned at the single-measurement
        value sqrt(2)*meas_noise_m, and `is_tracking` stayed True with
        nothing logged. MEASURED at defaults, a 0.6 m/s walker with clean
        blobs: dt = 2.9 s gave raw 0.3130 / tracked 0.2959, dt = 3.01 s gave
        0.3126 / 0.3126 — byte-identical, no filtering at all.

        Per-link rates below 1/max_coast_s are not exotic: RESEARCH.md's
        9-node mesh runs at about 0.175 Hz per link.
        """
        cases = (
            (0.4, 3.0),
            (2.9, 3.0),
            (3.01, 3.0),  # frame interval just past the coast budget
            (5.7, 3.0),  # the 9-node-mesh regime
            (0.4, 1.0),
            (0.4, 0.4),  # exactly equal, a floating-point knife edge
            (0.4, 0.0),  # "reset on any gap" must still filter a stream
        )
        for dt, max_coast_s in cases:
            with self.subTest(dt=dt, max_coast_s=max_coast_s):
                raw, tracked = self._walk(
                    seed=3,
                    pos_noise_m=0.25,
                    n_steps=40,
                    dt=dt,
                    max_coast_s=max_coast_s,
                )
                self.assertGreater(len(raw), 8)
                raw, tracked = raw[3:], tracked[3:]
                raw_mean = statistics.fmean(raw)
                tracked_mean = statistics.fmean(tracked)
                self.assertLess(
                    tracked_mean,
                    raw_mean,
                    f"dt={dt} max_coast_s={max_coast_s}: tracked "
                    f"{tracked_mean:.6f} did not beat raw {raw_mean:.6f}",
                )
                self.assertNotAlmostEqual(
                    tracked_mean, raw_mean, places=9, msg="filter stopped filtering"
                )

    def test_straight_walk_tracked_with_small_error(self) -> None:
        """Absolute accuracy, not just relative improvement."""
        _, tracked = self._walk(seed=11, pos_noise_m=0.25)
        tracked = tracked[5:]
        median_error = statistics.median(tracked)
        self.assertLess(median_error, 0.25)
        self.assertLess(max(tracked), 0.75)

    def test_velocity_converges_on_a_constant_velocity_walk(self) -> None:
        """The state's velocity must be right, not merely its position.

        Without this, a filter that simply low-passes the measurements and
        keeps vx = vy = 0 would pass the smoothing test, and would then
        coast to a standstill through every gap.
        """
        tracker = Tracker(self.geometry, meas_noise_m=0.15)
        vx, vy = 0.5, 0.25
        for i in range(50):
            t = i * self.dt
            cx, cy = 1.0 + vx * t, 1.0 + vy * t
            if cx > 7.0 or cy > 5.0:
                break
            tracker.update(gaussian_blob(self.geometry, cx, cy), t)
        velocity = tracker.velocity
        self.assertIsNotNone(velocity)
        assert velocity is not None
        self.assertAlmostEqual(velocity[0], vx, delta=0.08)
        self.assertAlmostEqual(velocity[1], vy, delta=0.08)

    def test_room_clamp_binds_on_an_emitted_frame(self) -> None:
        """Exercise the clamp, do not merely observe output that is inside.

        The previous version of this test only ever read the clamped output
        in a scenario where the unclamped state never came within 0.39 m of
        a wall, so the assertion was satisfied by the data and deleting the
        clamp left the suite green.

        Reaching the clamp needs a prior that has coasted outside AND a
        gain small enough that the correction does not simply snap back to
        the in-room measurement — that is, R large beside P. A 1.2 m/s
        walker into the right wall, a 0.8 s coast, then a fix, with
        `meas_noise_m` of 1.0 m.
        """
        tracker = Tracker(self.geometry, meas_noise_m=1.0, max_coast_s=3.0)
        t = 0.0
        for _ in range(14):
            cx = 1.0 + 1.2 * t
            if cx > 7.6:
                break
            tracker.update(gaussian_blob(self.geometry, cx, 3.0), t)
            t += 0.4
        for _ in range(2):  # 0.8 s of abstention, inside max_coast_s
            self.assertIsNone(tracker.update([0.0] * self.geometry.n_pixels, t))
            t += 0.4
        estimate = tracker.update(gaussian_blob(self.geometry, 7.75, 3.0), t)

        self.assertIsNotNone(estimate)
        assert estimate is not None
        raw = tracker.position
        assert raw is not None
        room = self.geometry.room
        self.assertGreater(
            raw[0], room.width_m, "scenario no longer drives the state out of the room"
        )
        self.assertAlmostEqual(estimate.x, room.width_m, places=12)
        self.assertTrue(room.contains(estimate.x, estimate.y))

    def test_estimates_stay_inside_the_room(self) -> None:
        room = self.geometry.room
        tracker = Tracker(self.geometry, meas_noise_m=0.4)
        rng = random.Random(5)
        for i in range(60):
            t = i * self.dt
            cx = 0.4 + 0.1 * i
            cy = 0.4
            if cx > 7.6:
                break
            image = gaussian_blob(self.geometry, cx + rng.gauss(0, 0.4), cy, sigma_m=0.5)
            estimate = tracker.update(image, t)
            if estimate is not None:
                self.assertTrue(
                    room.contains(estimate.x, estimate.y),
                    f"estimate ({estimate.x:.2f}, {estimate.y:.2f}) outside room",
                )


class TestAbstentionGaps(unittest.TestCase):
    """Coasting, covariance growth, and the reset."""

    def setUp(self) -> None:
        self.geometry = Geometry.perimeter(8, pitch_m=0.5)
        self.dt = 0.4

    def _run_with_gap(
        self,
        gap_start: int,
        gap_len: int,
        n_steps: int = 40,
        max_coast_s: float = 3.0,
        seed: int = 3,
    ):
        """Walk a straight line, replacing a stretch of frames with noise.

        Noise frames rather than blank ones, because that is what a VRTI
        image of a person who stopped walking actually looks like.
        """
        rng = random.Random(seed)
        tracker = Tracker(self.geometry, meas_noise_m=0.2, max_coast_s=max_coast_s)
        vx, vy = 0.5, 0.25
        log = []
        for i in range(n_steps):
            t = i * self.dt
            cx, cy = 1.0 + vx * t, 1.0 + vy * t
            if cx > 7.0 or cy > 5.0:
                break
            in_gap = gap_start <= i < gap_start + gap_len
            if in_gap:
                image = iid_noise_image(self.geometry, rng)
            else:
                image = gaussian_blob(self.geometry, cx, cy, sigma_m=0.5)
            estimate = tracker.update(image, t)
            log.append(
                {
                    "i": i,
                    "t": t,
                    "truth": (cx, cy),
                    "in_gap": in_gap,
                    "estimate": estimate,
                    "sigma": tracker.sigma_m,
                    "tracking": tracker.is_tracking,
                }
            )
        return log

    def test_gap_frames_abstain(self) -> None:
        log = self._run_with_gap(gap_start=10, gap_len=4)
        for row in log:
            if row["in_gap"]:
                self.assertIsNone(
                    row["estimate"], f"emitted a fix at i={row['i']} during a gap"
                )

    def test_resume_after_short_gap_does_not_jump(self) -> None:
        """The whole point of coasting instead of resetting.

        The error on the first frame back must be comparable to the steady
        state, not a snap from a stale position.
        """
        log = self._run_with_gap(gap_start=12, gap_len=4)
        resumed = [
            r for r in log if not r["in_gap"] and r["i"] > 15 and r["estimate"] is not None
        ]
        self.assertGreater(len(resumed), 3)
        first = resumed[0]
        estimate = first["estimate"]
        cx, cy = first["truth"]
        resume_error = math.hypot(estimate.x - cx, estimate.y - cy)

        pre_gap = [r for r in log if r["i"] < 12 and r["estimate"] is not None][5:]
        steady = statistics.fmean(
            math.hypot(r["estimate"].x - r["truth"][0], r["estimate"].y - r["truth"][1])
            for r in pre_gap
        )
        self.assertLess(
            resume_error,
            max(4.0 * steady, 0.5),
            f"snapped on resume: {resume_error:.3f} m vs steady {steady:.3f} m",
        )

    def test_sigma_grows_during_gap_and_shrinks_on_resume(self) -> None:
        log = self._run_with_gap(gap_start=12, gap_len=4)

        before = [r["sigma"] for r in log if r["i"] == 11][0]
        gap_sigmas = [r["sigma"] for r in log if r["in_gap"]]
        self.assertEqual(len(gap_sigmas), 4)

        # Monotonically increasing through the gap: each coasted step adds
        # Q and subtracts nothing.
        self.assertGreater(gap_sigmas[0], before)
        for earlier, later in zip(gap_sigmas, gap_sigmas[1:]):
            self.assertGreater(later, earlier)

        after = [r for r in log if r["i"] >= 16 and r["estimate"] is not None]
        self.assertGreater(len(after), 2)
        self.assertLess(after[0]["estimate"].sigma_m, gap_sigmas[-1])
        # And it settles back to roughly where it was before the gap.
        self.assertLess(after[2]["estimate"].sigma_m, 1.5 * before)

    def test_long_gap_resets_instead_of_coasting(self) -> None:
        """Past `max_coast_s` the constant-velocity story is just wrong."""
        log = self._run_with_gap(gap_start=8, gap_len=14, max_coast_s=3.0, n_steps=40)
        gap_rows = [r for r in log if r["in_gap"]]
        self.assertTrue(gap_rows[0]["tracking"], "reset immediately, did not coast")
        self.assertFalse(gap_rows[-1]["tracking"], "still tracking after a 5.6 s gap")
        self.assertIsNone(gap_rows[-1]["sigma"])

        # The reset must happen at the documented time, not whenever.
        last_fix_t = max(r["t"] for r in log if r["estimate"] is not None and r["i"] < 8)
        first_dropped = min(r["t"] for r in gap_rows if not r["tracking"])
        self.assertGreater(first_dropped - last_fix_t, 3.0)
        self.assertLessEqual(first_dropped - last_fix_t, 3.0 + self.dt + 1e-9)

    def test_reacquires_cleanly_after_a_reset(self) -> None:
        """A fresh start must land on the measurement, not blend with a ghost."""
        log = self._run_with_gap(gap_start=8, gap_len=14, max_coast_s=3.0, n_steps=40)
        after = [r for r in log if r["i"] >= 22 and r["estimate"] is not None]
        self.assertGreater(len(after), 2)
        first = after[0]
        error = math.hypot(
            first["estimate"].x - first["truth"][0],
            first["estimate"].y - first["truth"][1],
        )
        self.assertLess(error, 0.4, "reacquisition did not land on the measurement")


class TestClockSafety(unittest.TestCase):
    """The timestamp is an input, and a bad one must not poison the state."""

    def setUp(self) -> None:
        self.geometry = Geometry.perimeter(8, pitch_m=0.5)

    def _seed_track(self, tracker: Tracker, n: int = 5) -> float:
        t = 0.0
        for i in range(n):
            tracker.update(gaussian_blob(self.geometry, 1.0 + 0.2 * i, 1.0), t)
            t += 0.4
        return t

    def test_non_finite_timestamp_is_rejected(self) -> None:
        """Raise, rather than silently disabling the coast expiry forever.

        A single accepted frame carrying a NaN `t` used to set
        `_last_fix_t = nan`, after which `nan > max_coast_s` was False for
        every later frame and the reset never fired again. MEASURED: five
        good blobs, one NaN-timestamped frame, then 2500 guaranteed-abstain
        images spanning 1000 s left `is_tracking` True at (491.1, 1.0) —
        61 room-widths outside an 8 m room — with sigma_m at 8171 m.
        """
        for bad in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(t=bad):
                tracker = Tracker(self.geometry, meas_noise_m=0.3)
                self._seed_track(tracker)
                with self.assertRaises(ValueError):
                    tracker.update(gaussian_blob(self.geometry, 2.0, 1.0), bad)
                # And the rejected frame changed nothing.
                self.assertTrue(tracker.is_tracking)
                position = tracker.position
                assert position is not None
                self.assertTrue(self.geometry.room.contains(*position))

    def test_coast_expiry_survives_a_rejected_timestamp(self) -> None:
        """After the bad frame, the expiry must still fire on a real gap."""
        tracker = Tracker(self.geometry, meas_noise_m=0.3, max_coast_s=3.0)
        t = self._seed_track(tracker)
        with self.assertRaises(ValueError):
            tracker.update(gaussian_blob(self.geometry, 2.0, 1.0), float("nan"))
        for i in range(20):
            self.assertIsNone(
                tracker.update([0.0] * self.geometry.n_pixels, t + i * 0.4)
            )
        self.assertFalse(tracker.is_tracking, "coast expiry never fired")

    def test_rewound_clock_does_not_sustain_a_track(self) -> None:
        """A clock that walks backwards must not defeat the expiry.

        `self._t = t` used to be assigned unconditionally, so one rewound
        frame double-counted the elapsed time on the next step and a
        sustained rewind kept `t - _last_fix_t` negative forever. MEASURED:
        five good blobs then 60 noise frames at t = 0.0, -0.4, -0.8, ...
        left `is_tracking` True with sigma_m unchanged at 0.3304 after what
        should have been 24 s of abstention.
        """
        tracker = Tracker(self.geometry, meas_noise_m=0.3, max_coast_s=3.0)
        self._seed_track(tracker)
        rng = random.Random(7)
        for i in range(60):
            image = iid_noise_image(self.geometry, rng)
            self.assertIsNone(tracker.update(image, -0.4 * i))
        self.assertFalse(
            tracker.is_tracking, "a rewound clock kept the track alive forever"
        )

    def test_rewound_frame_does_not_inflate_the_next_interval(self) -> None:
        """One backwards step must not double-count elapsed time after it.

        Stated as an exact equivalence, because an inequality has no power
        here: `self._t = t` used to be assigned unconditionally, so the
        frame AFTER a rewind saw dt = 0.7 s instead of 0.4 s and predicted
        too far. A rewound frame is defined to contribute zero elapsed time,
        which means a run containing one is bit-identical to the same run
        with that frame's timestamp replaced by the current clock.
        """
        image = gaussian_blob(self.geometry, 4.0, 3.0)

        def run(times) -> float:
            tracker = Tracker(self.geometry, meas_noise_m=0.3)
            for t in times:
                tracker.update(image, t)
            sigma = tracker.sigma_m
            assert sigma is not None
            return sigma

        rewound = run((0.0, 0.4, 0.1, 0.8, 1.2))
        stalled = run((0.0, 0.4, 0.4, 0.8, 1.2))
        self.assertEqual(rewound, stalled)
        # And the extra frame did something, so the equality is not trivial.
        self.assertNotEqual(rewound, run((0.0, 0.4, 0.8, 1.2)))

    def test_time_going_backwards_does_not_shrink_covariance(self) -> None:
        """A rewound clock must not be read as negative elapsed time."""
        tracker = Tracker(self.geometry, meas_noise_m=0.3)
        image = gaussian_blob(self.geometry, 4.0, 3.0)
        tracker.update(image, 10.0)
        tracker.update(image, 10.4)
        before = tracker.sigma_m
        assert before is not None
        tracker.update(image, 9.0)
        after = tracker.sigma_m
        assert after is not None
        self.assertTrue(math.isfinite(after))
        self.assertLessEqual(after, before + 1e-9)
        self.assertGreater(after, 0.0)

    def test_clock_need_not_start_at_zero(self) -> None:
        """A caller's clock is its own; a late epoch is not a 1000 s gap."""
        tracker = Tracker(self.geometry, meas_noise_m=0.3, max_coast_s=3.0)
        for i in range(6):
            estimate = tracker.update(
                gaussian_blob(self.geometry, 4.0 + 0.1 * i, 3.0), 1000.0 + i * 0.4
            )
            self.assertIsNotNone(estimate)
        self.assertTrue(tracker.is_tracking)


class TestNumericsAndApi(unittest.TestCase):
    def setUp(self) -> None:
        self.geometry = Geometry.perimeter(8, pitch_m=0.5)

    @staticmethod
    def _leading_minors(cov: list[list[float]]) -> list[float]:
        """Determinants of the leading principal submatrices, by expansion.

        All strictly positive is Sylvester's criterion for positive
        definiteness — the property the Joseph form is there to preserve,
        and one that checking the diagonal alone cannot see.
        """
        minors = []
        for k in range(1, len(cov) + 1):
            sub = [row[:k] for row in cov[:k]]
            # Plain Gaussian elimination without pivoting; the matrix is
            # symmetric positive definite if this succeeds with positive
            # pivots, which is exactly what is being asserted.
            m = [row[:] for row in sub]
            det = 1.0
            for i in range(k):
                pivot = m[i][i]
                det *= pivot
                if pivot == 0.0:
                    break
                for r in range(i + 1, k):
                    factor = m[r][i] / pivot
                    for c in range(i, k):
                        m[r][c] -= factor * m[i][c]
            minors.append(det)
        return minors

    def test_covariance_stays_symmetric_and_positive_definite(self) -> None:
        """Joseph form plus symmetrization, over thousands of updates.

        Positive DEFINITE, via leading minors, not merely positive on the
        diagonal: a covariance can have four positive diagonal entries and
        still be indefinite, and that is the state the Joseph form exists
        to avoid. The run length is the one the module docstring claims
        ("thousands of updates"), not the 400 that was tested before.
        """
        tracker = Tracker(self.geometry, meas_noise_m=0.3)
        rng = random.Random(1)
        for i in range(3000):
            t = i * 0.4
            cx = 1.0 + 3.0 * (1.0 + math.sin(0.05 * t))
            cy = 1.0 + 2.0 * (1.0 + math.cos(0.07 * t))
            image = gaussian_blob(
                self.geometry, cx + rng.gauss(0, 0.1), cy + rng.gauss(0, 0.1)
            )
            tracker.update(image, t)
            if i % 50 and i != 2999:
                continue
            cov = tracker._cov
            self.assertIsNotNone(cov)
            assert cov is not None
            for a in range(4):
                self.assertTrue(math.isfinite(cov[a][a]))
                for b in range(4):
                    self.assertEqual(cov[a][b], cov[b][a])
            for order, minor in enumerate(self._leading_minors(cov), start=1):
                self.assertGreater(
                    minor, 0.0, f"leading minor {order} is {minor} at step {i}"
                )
        sigma = tracker.sigma_m
        self.assertIsNotNone(sigma)
        assert sigma is not None
        self.assertTrue(math.isfinite(sigma))
        self.assertLess(sigma, 1.0)

    def test_process_noise_matches_the_cwna_integral(self) -> None:
        """Pin Q against the closed form, cross terms included.

        The position/velocity cross terms are argued for at length in
        `_process_noise` — they are why `sigma_m` is not optimistic — and
        setting `q_pv = 0.0` used to leave the whole suite green. Comparing
        the predicted covariance against the integral computed here, from
        the model and not from the code, kills that mutant on the first
        assertion.

            P' = F P F^T + Q,  Q = q [[dt^3/3, dt^2/2], [dt^2/2, dt]]
        """
        dt = 0.7
        q = 0.1
        r = 0.09
        v2 = 1.0
        tracker = Tracker(
            self.geometry, meas_noise_m=math.sqrt(r), accel_psd=q, init_speed_std_ms=1.0
        )
        tracker._initialise(4.0, 3.0, r)
        tracker._predict(dt)
        cov = tracker._cov
        assert cov is not None

        expected_pp = r + dt * dt * v2 + q * dt**3 / 3.0
        expected_pv = dt * v2 + q * dt * dt / 2.0
        expected_vv = v2 + q * dt
        for axis, vel in ((0, 2), (1, 3)):
            self.assertAlmostEqual(cov[axis][axis], expected_pp, places=12)
            self.assertAlmostEqual(cov[axis][vel], expected_pv, places=12)
            self.assertAlmostEqual(cov[vel][vel], expected_vv, places=12)
        # The two axes are independent; nothing may couple them.
        self.assertEqual(cov[0][1], 0.0)
        self.assertEqual(cov[0][3], 0.0)
        self.assertEqual(cov[1][2], 0.0)
        self.assertEqual(cov[2][3], 0.0)

    def test_dropping_the_q_cross_terms_changes_the_steady_state(self) -> None:
        """Pin the consequence of the cross terms, measured not assumed.

        The module used to claim that dropping them makes `sigma_m`
        optimistic. It does the opposite: the correction a diagonal Q allows
        removes less velocity variance, so the NEXT coast inflates position
        by more. Measured here against a longhand recursion of the same
        model — the filter's own numbers never enter the comparison — the
        position variance after a coast is 0.3850 with the cross terms and
        0.5025 without, at dt = 1.2 s, q = 0.1, R = 0.09.

        The point of the test is not the direction but that the two are
        far apart: a diagonal Q is a different model, not a simplification,
        and `q_pv = 0.0` used to leave every test in this file green.
        """
        dt = 1.2
        q = 0.1
        r = 0.09

        def coasted_position_variance(cross: float, steps: int) -> float:
            """One axis of the CWNA recursion, written out from the model."""
            pp, pv, vv = r, 0.0, 1.0
            for _ in range(steps):
                pp, pv, vv = (
                    pp + 2.0 * dt * pv + dt * dt * vv + q * dt**3 / 3.0,
                    pv + dt * vv + cross * q * dt * dt / 2.0,
                    vv + q * dt,
                )
                gain = pp / (pp + r)
                k_v = pv / (pp + r)
                pp, pv, vv = (
                    (1.0 - gain) ** 2 * pp + gain * gain * r,
                    (1.0 - gain) * pv,
                    vv - k_v * pv,
                )
            return pp + 2.0 * dt * pv + dt * dt * vv + q * dt**3 / 3.0

        steps = 5
        with_cross = coasted_position_variance(1.0, steps)
        without_cross = coasted_position_variance(0.0, steps)
        self.assertLess(with_cross, without_cross)
        self.assertGreater(without_cross - with_cross, 0.1)

        # ... and the shipped filter reproduces the with-cross recursion.
        tracker = Tracker(self.geometry, meas_noise_m=math.sqrt(r), accel_psd=q)
        tracker._initialise(4.0, 3.0, r)
        for _ in range(steps):
            tracker._predict(dt)
            tracker._correct(4.0, 3.0, r)
        tracker._predict(dt)
        cov = tracker._cov
        assert cov is not None
        self.assertAlmostEqual(cov[0][0], with_cross, places=12)
        self.assertAlmostEqual(cov[1][1], with_cross, places=12)

    def test_joseph_form_survives_a_vanishing_r(self) -> None:
        """The regime where the short form actually breaks.

        `(I - KH) P` and the Joseph form are algebraically equal at the
        optimal gain, so no amount of ordinary tracking distinguishes them —
        which is why the shipped suite could delete the `K R K^T` term and
        stay green. They part company once R is negligible beside P and the
        gain saturates: the short form's cancellation drives a diagonal to
        exactly zero and then negative. MEASURED over 300 steps at
        dt = 0.4: identical at r = 1e-12, and at r = 1e-18 the short form
        reaches 0.0 while the Joseph form holds 1e-18.

        This is insurance, not a live fix at default settings, and the
        assertion is bounded to exactly that claim.
        """
        tracker = Tracker(self.geometry, meas_noise_m=1e-9, accel_psd=0.1)
        tracker._initialise(4.0, 3.0, 1e-18)
        for _ in range(300):
            tracker._predict(0.4)
            tracker._correct(4.0, 3.0, 1e-18)
            cov = tracker._cov
            assert cov is not None
            for a in range(4):
                self.assertGreater(
                    cov[a][a], 0.0, "covariance diagonal collapsed under a tiny R"
                )
        cov = tracker._cov
        assert cov is not None
        for minor in self._leading_minors(cov):
            self.assertGreater(minor, 0.0)

    def test_repeated_identical_updates_are_deterministic(self) -> None:
        image = gaussian_blob(self.geometry, 3.3, 2.7)
        results = []
        for _ in range(3):
            tracker = Tracker(self.geometry, meas_noise_m=0.3)
            run = []
            for i in range(20):
                estimate = tracker.update(image, i * 0.4)
                run.append(None if estimate is None else (estimate.x, estimate.y, estimate.sigma_m))
            results.append(run)
        self.assertEqual(results[0], results[1])
        self.assertEqual(results[1], results[2])

    def test_confidence_is_monotonic_and_bounded(self) -> None:
        tracker = Tracker(self.geometry)
        self.assertEqual(tracker.confidence_from_z(0.0), 0.0)
        self.assertEqual(tracker.confidence_from_z(-5.0), 0.0)
        self.assertEqual(tracker.confidence_from_z(DEFAULT_Z_THRESHOLD - 0.01), 0.0)
        self.assertAlmostEqual(tracker.confidence_from_z(DEFAULT_Z_THRESHOLD), 0.0)
        self.assertAlmostEqual(tracker.confidence_from_z(2.0 * DEFAULT_Z_THRESHOLD), 0.5)
        previous = -1.0
        for z in (5.0, 6.0, 8.0, 10.0, 25.0, 100.0, 1e6):
            value = tracker.confidence_from_z(z)
            self.assertGreaterEqual(value, previous)
            self.assertLessEqual(value, 1.0)
            previous = value

    def test_confidence_of_a_non_finite_z_is_zero(self) -> None:
        """`min(1.0, nan)` is 1.0 in CPython, so this needs its own test.

        Without the explicit guard, a NaN z — which a diverged upstream
        stage can produce — came back as confidence 1.0, the single most
        damaging value the function can return.
        """
        tracker = Tracker(self.geometry)
        for z in (float("nan"), float("inf"), float("-inf")):
            with self.subTest(z=z):
                self.assertEqual(tracker.confidence_from_z(z), 0.0)

    def test_image_length_is_validated(self) -> None:
        tracker = Tracker(self.geometry)
        with self.assertRaises(ValueError):
            tracker.update([0.0] * (self.geometry.n_pixels + 1), 0.0)
        with self.assertRaises(ValueError):
            extract_peak(self.geometry, [0.0] * 3)

    def test_constructor_rejects_nonsense(self) -> None:
        """Including NaN, which every `x <= 0` test silently admits.

        `nan <= 0.0` is False, so the natural spelling of each guard let a
        NaN through. A NaN `z_threshold` is the worst of them: both
        `z >= nan` and `z < nan` are False, so the significance test stops
        depending on the image at all.
        """
        nan = float("nan")
        for kwargs in (
            {"z_threshold": 0.0},
            {"z_threshold": -1.0},
            {"z_threshold": nan},
            {"centroid_radius_m": 0.0},
            {"centroid_radius_m": nan},
            {"accel_psd": 0.0},
            {"accel_psd": nan},
            {"meas_noise_m": -0.5},
            {"meas_noise_m": nan},
            {"init_speed_std_ms": 0.0},
            {"init_speed_std_ms": nan},
            {"max_coast_s": -1.0},
            {"max_coast_s": nan},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    Tracker(self.geometry, **kwargs)
        for radius in (0.0, -1.0, nan):
            with self.assertRaises(ValueError):
                extract_peak(
                    self.geometry, [0.0] * self.geometry.n_pixels, radius_m=radius
                )

    def test_reset_clears_the_track(self) -> None:
        tracker = Tracker(self.geometry)
        tracker.update(gaussian_blob(self.geometry, 4.0, 3.0), 0.0)
        self.assertTrue(tracker.is_tracking)
        self.assertIsNotNone(tracker.position)
        tracker.reset()
        self.assertFalse(tracker.is_tracking)
        self.assertIsNone(tracker.position)
        self.assertIsNone(tracker.velocity)
        self.assertIsNone(tracker.sigma_m)

    def test_measurement_noise_defaults_to_the_grid_pitch(self) -> None:
        for pitch in (0.5, 0.25):
            geometry = Geometry.perimeter(8, pitch_m=pitch)
            tracker = Tracker(geometry)
            self.assertAlmostEqual(tracker.meas_noise_m, pitch)

    def test_weak_detections_are_trusted_less(self) -> None:
        """R inflation: a threshold-grazing peak must move the state less.

        Asserted on the NORMALISED gain, (posterior - prior) / (measurement
        - prior), not on the absolute posterior. The two measurements land
        in different places, so comparing absolute positions compares the
        measurements rather than the weighting: with the inflation deleted
        the old assertion still read 1.4368 < 1.4515 and passed on a
        0.0147 m margin, about 35x too insensitive to notice.

        MEASURED gains: 0.737 for the strong peak, 0.483 for the weak one.
        """
        offset = 2.0

        def step(amplitude: float, noise_sd: float, seed: int):
            tracker = Tracker(self.geometry, meas_noise_m=0.3)
            tracker.update(gaussian_blob(self.geometry, 4.0, 3.0), 0.0)
            prior = tracker.position
            assert prior is not None
            rng = random.Random(seed)
            image = gaussian_blob(
                self.geometry, 4.0 + offset, 3.0, sigma_m=0.5, amplitude=amplitude
            )
            image = [v + rng.gauss(0.0, noise_sd) for v in image]
            reading = extract_peak(self.geometry, image)
            estimate = tracker.update(image, 0.4)
            if estimate is None:
                return prior, reading, None, None
            gain = (estimate.x - prior[0]) / (reading.x - prior[0])
            return prior, reading, estimate, gain

        _, _, strong, strong_gain = step(20.0, 0.0, 1)
        self.assertIsNotNone(strong)
        assert strong is not None and strong_gain is not None

        # Find a noise level that leaves the peak just above threshold.
        weak = None
        weak_gain = None
        for sd in (0.5, 0.7, 0.9, 1.1, 1.3):
            _, _, weak, weak_gain = step(3.0, sd, 2)
            if weak is not None and weak.confidence < 0.35:
                break
        self.assertIsNotNone(weak, "could not construct a weak-but-significant peak")
        assert weak is not None and weak_gain is not None
        self.assertLess(weak.confidence, strong.confidence)
        self.assertLess(
            weak_gain,
            0.9 * strong_gain,
            f"weak gain {weak_gain:.4f} is not meaningfully below "
            f"strong gain {strong_gain:.4f}",
        )

    def test_dataclasses_expose_the_contract_fields(self) -> None:
        tracker = Tracker(self.geometry)
        estimate = tracker.update(gaussian_blob(self.geometry, 4.0, 3.0), 1.25)
        self.assertIsInstance(estimate, PositionEstimate)
        assert estimate is not None
        for field in ("x", "y", "sigma_m", "confidence"):
            self.assertTrue(hasattr(estimate, field))
            self.assertTrue(math.isfinite(getattr(estimate, field)))
        self.assertEqual(estimate.t, 1.25)
        reading = extract_peak(self.geometry, gaussian_blob(self.geometry, 4.0, 3.0))
        self.assertIsInstance(reading, PeakReading)
        self.assertIsNotNone(reading.peak_index)
        self.assertEqual(reading.tie_fraction, 0.0)

    def test_works_at_other_node_counts_and_pitches(self) -> None:
        """Nothing here should depend on the 8-node/0.5 m default."""
        for n_nodes in (4, 6, 12):
            for pitch in (0.5, 0.25):
                with self.subTest(n_nodes=n_nodes, pitch=pitch):
                    geometry = Geometry.perimeter(n_nodes, pitch_m=pitch)
                    tracker = Tracker(geometry)
                    estimate = tracker.update(gaussian_blob(geometry, 4.1, 3.1), 0.0)
                    self.assertIsNotNone(estimate)
                    assert estimate is not None
                    self.assertLess(math.hypot(estimate.x - 4.1, estimate.y - 3.1), 0.35)
                    self.assertIsNone(
                        tracker.update([0.0] * geometry.n_pixels, 0.4)
                    )


class TestAgainstRealInvertedImages(unittest.TestCase):
    """End-to-end against `invert`, not against hand-built blobs.

    Everything above feeds the tracker Gaussian bumps this file wrote
    itself, which proves the estimator is self-consistent but says nothing
    about whether the significance threshold is in the right place for the
    images it will actually receive. This closes that gap.

    Skipped rather than failed if `invert` cannot be imported or its
    signature has moved: it belongs to another session and is still in
    flux, and a red test here would be reporting their state, not mine.

    NOTE on what this does and does not show. `forward_project` builds y
    with exactly the weight model the `Inverter` inverts, so this is an
    inverse crime: it bounds the estimator's own error, not the error on
    real data, where the ellipse model is wrong in ways no amount of
    regularization fixes. Read the numbers as a floor.
    """

    def setUp(self) -> None:
        try:
            from wifisense.localize.invert import Inverter, forward_project
        except Exception as exc:  # pragma: no cover - depends on a peer module
            self.skipTest(f"wifisense.localize.invert unavailable: {exc}")
        self.Inverter = Inverter
        self.forward_project = forward_project
        self.geometry = Geometry.perimeter(8, pitch_m=0.5)
        try:
            self.inverter = Inverter(
                self.geometry, alpha=1.0, sigma=1.0, delta_c=0.5, warn_blind_pixels=False
            )
        except TypeError as exc:  # pragma: no cover - signature drift
            self.skipTest(f"Inverter signature changed: {exc}")

    def _target_field(self, cx: float, cy: float, extent: float = 0.4) -> list[float]:
        two_s2 = 2.0 * extent * extent
        return [
            math.exp(-((x - cx) ** 2 + (y - cy) ** 2) / two_s2)
            for x, y in self.geometry.grid.pixels
        ]

    def _noise_scale(self) -> float:
        probe = self.forward_project(self.geometry, self._target_field(4.0, 3.0))
        return 0.15 * (statistics.fmean(abs(v) for v in probe) or 1.0)

    def test_fires_on_a_real_target_and_abstains_on_an_empty_room(self) -> None:
        """The threshold must sit between the two, and it does.

        MEASURED over 60 trials at N=8, 0.5 m pitch, alpha=1, delta_c=0.5:
        median z was 11.0 with a target present and 2.6 on an empty room,
        against a threshold of 5.0. Coverage 100%, median error 0.10 m,
        p90 0.22 m.
        """
        rng = random.Random(17)
        noise = self._noise_scale()

        errors = []
        detections = 0
        trials = 20
        for _ in range(trials):
            cx = rng.uniform(1.5, 6.5)
            cy = rng.uniform(1.5, 4.5)
            y = self.forward_project(self.geometry, self._target_field(cx, cy))
            y = [v + rng.gauss(0.0, noise) for v in y]
            tracker = Tracker(self.geometry)
            estimate = tracker.update(self.inverter.solve(y), 0.0)
            if estimate is not None:
                detections += 1
                errors.append(math.hypot(estimate.x - cx, estimate.y - cy))

        self.assertGreaterEqual(
            detections, int(0.9 * trials), "threshold is too high to see a real target"
        )
        self.assertLess(statistics.median(errors), 0.5)

        empty_fixes = []
        for _ in range(trials):
            y = [rng.gauss(0.0, noise) for _ in range(self.geometry.n_links)]
            tracker = Tracker(self.geometry)
            estimate = tracker.update(self.inverter.solve(y), 0.0)
            if estimate is not None:
                empty_fixes.append(estimate)

        self.assertLessEqual(
            len(empty_fixes), 2, f"{len(empty_fixes)}/{trials} empty rooms produced a fix"
        )
        for estimate in empty_fixes:
            self.assertLess(estimate.confidence, 0.40)

    def test_a_clamped_empty_room_does_not_produce_a_fix(self) -> None:
        """The realistic route into the degenerate-image failure.

        Attenuation cannot be negative, so rectifying the inverted image is
        something a caller does without a second thought — and it is what
        turned a harmless empty-room image into a confident position.
        MEASURED over 500 empty rooms at these settings: 6 fixes and 0
        confident when passed raw, against 441 fixes and 261 CONFIDENT
        (max 0.999) once `max(v, 0.0)` was applied.
        """
        rng = random.Random(55003)
        noise = self._noise_scale()
        raw_fixes = 0
        clamped_fixes = []
        trials = 60
        for _ in range(trials):
            y = [rng.gauss(0.0, noise) for _ in range(self.geometry.n_links)]
            image = self.inverter.solve(y)
            if Tracker(self.geometry).update(image, 0.0) is not None:
                raw_fixes += 1
            estimate = Tracker(self.geometry).update([max(v, 0.0) for v in image], 0.0)
            if estimate is not None:
                clamped_fixes.append(estimate)

        self.assertLessEqual(raw_fixes, trials // 10)
        self.assertLessEqual(
            len(clamped_fixes),
            trials // 10,
            f"{len(clamped_fixes)}/{trials} clamped empty rooms produced a fix",
        )
        for estimate in clamped_fixes:
            self.assertLess(estimate.confidence, 0.40)

    def test_significance_is_scale_invariant(self) -> None:
        """Doubling the measurement units must not change any decision.

        The whole reason significance is a ratio against the image's own
        spread is that nothing upstream pins the units: `measure` can
        report variance in whatever the CSI amplitudes happen to be. A
        threshold that moved with the units would need recalibrating for
        every configuration.

        Note the claim is about the scale of `y` at fixed `alpha`. It does
        NOT extend to `alpha` itself, which changes the image's shape — see
        the module docstring for the measured alpha sweep.
        """
        y = self.forward_project(self.geometry, self._target_field(3.4, 2.6))
        base = extract_peak(self.geometry, self.inverter.solve(y))
        for factor in (1e-6, 0.5, 2.0, 1e6):
            scaled = extract_peak(
                self.geometry, self.inverter.solve([v * factor for v in y])
            )
            self.assertAlmostEqual(scaled.z, base.z, delta=1e-6 * max(base.z, 1.0))
            self.assertAlmostEqual(scaled.x, base.x, places=9)
            self.assertAlmostEqual(scaled.y, base.y, places=9)

    def test_significance_is_not_invariant_to_alpha(self) -> None:
        """Pin the corrected claim, since the wrong one was documented.

        The module used to say significance "does not depend on the
        measurement units or the regularizer gain". The first half holds
        and is tested above; the second half is false, and it mattered
        because `z_threshold` is a swept config field in
        evaluate_localize.py and the docstring told the sweeper the
        calibrated 5.0 carried across alpha. MEASURED median z: 11.03 at
        alpha=1, 7.57 at alpha=100, 3.87 at alpha=1000, where coverage on
        real targets falls from 20/20 to 0/20.
        """
        rng = random.Random(17)
        noise = self._noise_scale()
        targets = [
            (rng.uniform(1.5, 6.5), rng.uniform(1.5, 4.5)) for _ in range(8)
        ]
        measurements = [
            [
                v + rng.gauss(0.0, noise)
                for v in self.forward_project(self.geometry, self._target_field(cx, cy))
            ]
            for cx, cy in targets
        ]

        medians = {}
        for alpha in (1.0, 1000.0):
            try:
                inverter = self.Inverter(
                    self.geometry,
                    alpha=alpha,
                    sigma=1.0,
                    delta_c=0.5,
                    warn_blind_pixels=False,
                )
            except TypeError as exc:  # pragma: no cover - signature drift
                self.skipTest(f"Inverter signature changed: {exc}")
            medians[alpha] = statistics.median(
                extract_peak(self.geometry, inverter.solve(y)).z for y in measurements
            )

        self.assertGreater(medians[1.0], DEFAULT_Z_THRESHOLD)
        self.assertLess(medians[1000.0], DEFAULT_Z_THRESHOLD)
        self.assertLess(
            medians[1000.0],
            0.5 * medians[1.0],
            "z turned out to be insensitive to alpha after all; "
            "the module docstring's alpha table needs re-measuring",
        )


if __name__ == "__main__":
    unittest.main()
