"""Numerical tests for the VRTI per-link measurement stage.

These tests drive `LinkMeasurement` with a small physics-based CSI
generator defined *here*, in the test file, rather than with `sim.py`.
Two reasons. The obvious one is that `sim.py` belongs to another session
and may not exist yet. The better one is that a measurement stage checked
against the simulator it will ship beside only proves the two agree; a
generator written independently from the path geometry can disagree, and
that is the point of a test.

The generator is deliberately simple and is NOT a substitute for `sim.py`:
line of sight with distance-dependent shadowing, three fixed static
reflections, one body-scattered path, additive complex Gaussian noise and
a random per-packet common phase. It is enough to make a link's variance
rise when a body approaches it, which is the only physics this module is
supposed to react to.

Every number printed by `TestSignalStrategyComparison` is SYNTHETIC and is
a property of that generator, not of real hardware.

`TestClockDomains` drives the generator differently from everything else
here: it stamps links with per-node clock offsets, reordered packets and
restarted streams, because a mesh of free-running ESP32s does all three
and a test harness that gives every link one identical timestamp cannot
see any of it. `stream`'s `slot_s` is there for the same reason.

Whole file runs in about 3 s.
"""

from __future__ import annotations

import cmath
import math
import random
import time
import unittest

from wifisense.localize.geometry import Geometry, Node, Room
from wifisense.localize.measure import (
    CalibrationMode,
    LinkMeasurement,
    LinkSignal,
    LinkSignalState,
    StaticSuppressedSignal,
    TotalPowerSignal,
)
from wifisense.preprocess import StaticSuppressor
from wifisense.types import CsiFrame

# ---------------------------------------------------------------------------
# Synthetic CSI generator (test fixture, see module docstring)
# ---------------------------------------------------------------------------

C_M_S = 299_792_458.0
CENTER_HZ = 2.437e9
SPACING_HZ = 312.5e3
N_SUB = 56

# Per-link rate for 8 nodes at a 10 ms TDMA slot: 1 / (8 * 0.010).
RATE_HZ = 12.5

# Shadowing model S(delta) = S0 * exp(-delta / delta0) dB, delta = excess path.
S0_DB = 6.0
DELTA0_M = 0.3
# Effective body scattering, metres. A human at 2.4 GHz is a large scatterer.
BODY_RCS_M = 0.8

# Reference RMS |H| for an unoccupied link in this room, used to turn an
# SNR in dB into a noise sigma. Measured once by `_reference_h_rms`.
_H_RMS_CACHE: dict[tuple, float] = {}


def geometry_key(geometry: Geometry) -> tuple:
    """A content-based cache key for a Geometry.

    Deliberately NOT `id(geometry)`. Several test classes build a fresh
    geometry per test method and let it die, so CPython is free to hand a
    later, structurally different geometry the same address --- at which
    point an `id()`-keyed cache serves frames, or a noise scale, computed
    for the wrong room. That failure is invisible, depends on allocation
    order, and would only ever show up as an unreproducible flake.
    """
    return (
        geometry.room.width_m,
        geometry.room.height_m,
        tuple((n.node_id, n.x, n.y) for n in geometry.nodes),
    )


def _channel(paths: list[tuple[float, float]], n_sub: int) -> list[complex]:
    """Sum `(amplitude, path_length)` contributions over subcarriers.

    Each path contributes `a * exp(-j 2 pi f_k L / c)`. Evaluated by a
    geometric recurrence rather than one `cmath.exp` per subcarrier,
    because `f_k` is an arithmetic progression and so the phasor is a
    geometric one: two transcendental calls per path instead of 56. The
    whole test file is 15x faster for it, and the drift over 56 complex
    multiplies is around 1e-15.
    """
    f_first = CENTER_HZ - 0.5 * (n_sub - 1) * SPACING_HZ
    out = [0j] * n_sub
    for amplitude, length in paths:
        if amplitude == 0.0:
            continue
        current = amplitude * cmath.exp(-2j * math.pi * f_first * length / C_M_S)
        ratio = cmath.exp(-2j * math.pi * SPACING_HZ * length / C_M_S)
        for k in range(n_sub):
            out[k] += current
            current *= ratio
    return out


def _link_paths(
    tx: tuple[float, float],
    rx: tuple[float, float],
    person: tuple[float, float] | None,
    s0_db: float = S0_DB,
) -> list[tuple[float, float]]:
    """Propagation paths for one link, with or without a person."""
    d = math.dist(tx, rx)
    paths: list[tuple[float, float]] = []

    shadow_gain = 1.0
    if person is not None:
        d1 = math.dist(tx, person)
        d2 = math.dist(person, rx)
        excess = d1 + d2 - d
        attenuation_db = s0_db * math.exp(-excess / DELTA0_M)
        shadow_gain = 10.0 ** (-attenuation_db / 20.0)
        # Body-scattered path: amplitude ~ 1/(d1 d2), delay from d1 + d2.
        paths.append((BODY_RCS_M / (d1 * d2), d1 + d2))

    paths.append((shadow_gain / d, d))
    # Fixed first-order wall reflections. Static, so they contribute to the
    # mean channel but not to its variance -- which is exactly what the
    # empty-room calibration has to learn and remove.
    for excess_m, coefficient in ((1.7, 0.40), (3.1, 0.30), (4.9, 0.22)):
        reflected = d + excess_m
        paths.append((coefficient / reflected, reflected))
    return paths


def _reference_h_rms(geometry: Geometry, n_sub: int) -> float:
    """Mean RMS |H| over the empty-room links, for turning SNR into sigma."""
    key = (geometry_key(geometry), n_sub)
    cached = _H_RMS_CACHE.get(key)
    if cached is not None:
        return cached
    totals = []
    for link in geometry.links:
        h = _channel(_link_paths(link.tx, link.rx, None), n_sub)
        totals.append(math.sqrt(sum(abs(v) ** 2 for v in h) / n_sub))
    value = sum(totals) / len(totals)
    _H_RMS_CACHE[key] = value
    return value


def sigma_for_snr(geometry: Geometry, snr_db: float, n_sub: int = N_SUB) -> float:
    """Per-component noise sigma giving roughly `snr_db` on an empty link."""
    return _reference_h_rms(geometry, n_sub) * 10.0 ** (-snr_db / 20.0)


def make_frame(
    rng: random.Random,
    link,
    person: tuple[float, float] | None,
    sigma: float,
    sample_index: int,
    n_sub: int = N_SUB,
    s0_db: float = S0_DB,
) -> CsiFrame:
    """One CSI snapshot for one link."""
    h = _channel(_link_paths(link.tx, link.rx, person, s0_db), n_sub)
    # Random common phase per packet (CFO). It rotates every subcarrier
    # equally, so it leaves amplitude untouched -- which is precisely why
    # this module is amplitude-only. Included so the frames are not
    # accidentally phase-coherent in a way real hardware never is.
    rotation = cmath.exp(1j * rng.uniform(-math.pi, math.pi))
    amplitudes = []
    phases = []
    for k in range(n_sub):
        noise = complex(rng.gauss(0.0, sigma), rng.gauss(0.0, sigma))
        value = (h[k] + noise) * rotation
        amplitudes.append(abs(value))
        phases.append(cmath.phase(value))
    return CsiFrame(amplitudes, phases, sample_index, RATE_HZ, node_id=link.rx_id)


def stream(
    geometry: Geometry,
    person_at,
    seed: int,
    duration_s: float,
    sigma: float,
    n_sub: int = N_SUB,
    s0_db: float = S0_DB,
    t0: float = 0.0,
    slot_s: float = 0.0,
) -> list[tuple[float, int, int, CsiFrame]]:
    """A materialised `(t, tx_id, rx_id, frame)` stream, TDMA-ordered.

    Materialised rather than generated so that two strategies can be fed
    byte-identical input; a generator would re-randomise on the second
    pass and quietly invalidate every comparison in this file.

    `slot_s` is the TDMA slot length. With the default 0.0 every link in
    a superframe carries one identical timestamp, which is convenient and
    is also the one case where per-link and global windowing cannot be
    told apart. Pass the real slot length (`1 / (rate * n_nodes)`, the
    model in P0B_TASK §3.1: node *i* transmits in slot *i*) and the links
    spread across the superframe the way they actually do.
    """
    rng = random.Random(seed)
    period = 1.0 / RATE_HZ
    out = []
    for i in range(int(duration_s * RATE_HZ)):
        for link in geometry.links:
            t = t0 + i * period + link.tx_id * slot_s
            out.append(
                (t, link.tx_id, link.rx_id,
                 make_frame(rng, link, person_at(t), sigma, i, n_sub, s0_db))
            )
    return out


def empty_room(t: float) -> None:
    return None


def two_node_geometry() -> Geometry:
    """One link straight across the room at y = 3, plus its reverse."""
    return Geometry(Room(8.0, 6.0), [Node(0, 0.1, 3.0), Node(1, 7.9, 3.0)])


def mean_std(values: list[float]) -> tuple[float, float]:
    """Two-pass mean and sample standard deviation, written independently.

    Deliberately not `RunningVariance`: several tests check the module's
    Welford output against this, and checking an implementation against
    itself proves nothing.
    """
    n = len(values)
    if n < 2:
        raise AssertionError(f"need at least 2 values, got {n}")
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    return mean, math.sqrt(variance)


def feed(measurement: LinkMeasurement, frames) -> None:
    for t, tx_id, rx_id, frame in frames:
        measurement.update(t, tx_id, rx_id, frame)


def time_averaged_vector(measurement: LinkMeasurement, frames, warmup_s=1.5):
    """Mean of `y` per link over every completed superframe.

    Averaging over time is what separates the *systematic* per-link offset
    from the per-window estimator noise. A single `vector()` call mixes the
    two at comparable magnitude --- a 13-sample variance has roughly 40%
    relative spread --- so a test that compares single snapshots cannot
    tell whether calibration removed the offset or merely reshuffled the
    noise. Over ~140 windows the noise falls by about an order of
    magnitude and the offset stands alone.
    """
    n_links = measurement.geometry.n_links
    totals = [0.0] * n_links
    count = 0
    current_t = None

    def sample():
        nonlocal count
        values = measurement.vector()
        for i in range(n_links):
            totals[i] += values[i]
        count += 1

    for t, tx_id, rx_id, frame in frames:
        if current_t is not None and t != current_t and current_t >= warmup_s:
            sample()
        current_t = t
        measurement.update(t, tx_id, rx_id, frame)
    if current_t is not None and current_t >= warmup_s:
        sample()

    if count == 0:
        raise AssertionError("no completed superframes after warmup")
    return [total / count for total in totals], count


def collect_link_y(
    measurement: LinkMeasurement,
    frames,
    tx_id: int,
    rx_id: int,
    warmup_s: float = 1.5,
) -> list[float]:
    """Feed `frames`, sampling `y` for one link once per superframe."""
    index = measurement.geometry.link_index(tx_id, rx_id)
    out = []
    for t, tx, rx, frame in frames:
        measurement.update(t, tx, rx, frame)
        if tx == tx_id and rx == rx_id and t >= warmup_s:
            out.append(measurement.vector()[index])
    return out


# A corpus cache: frame generation dominates this file's runtime, and the
# comparison tests reuse the same streams across strategies by design.
_CORPUS: dict[tuple, list] = {}


def corpus(geometry, person_at, seed, duration_s, sigma, n_sub=N_SUB, s0_db=S0_DB,
           slot_s=0.0):
    key = (geometry_key(geometry), person_at.__name__, seed, duration_s,
           sigma, n_sub, s0_db, slot_s)
    cached = _CORPUS.get(key)
    if cached is None:
        cached = stream(geometry, person_at, seed, duration_s, sigma, n_sub, s0_db,
                        slot_s=slot_s)
        _CORPUS[key] = cached
    return cached


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------


def walking_across(t: float) -> tuple[float, float]:
    """Pacing ACROSS the two-node link, peak 1.1 m/s, staying in the room.

    Across, not along. Moving along a link's axis near its midpoint barely
    changes `d1 + d2` --- the derivative is `(x-x_tx)/d1 - (x_rx-x)/d2`,
    which vanishes at the midpoint --- so an axial walk is nearly invisible
    to that link no matter how fast it is. Crossing the line is the
    strong-disturbance direction, and it is what a tracker must see.
    """
    return (4.0, 3.0 + 2.0 * math.sin(2.0 * math.pi * t / 11.4))


def breathing_still(t: float) -> tuple[float, float]:
    """Standing on the link, chest moving 5 mm at 0.25 Hz. Nothing else."""
    return (4.0, 3.4 + 0.005 * math.sin(2.0 * math.pi * 0.25 * t))


def walking_corner(t: float) -> tuple[float, float]:
    """A slow continuous wander confined to the bottom-left corner."""
    return (
        0.8 + 0.4 * math.sin(2.0 * math.pi * 0.17 * t),
        1.1 + 0.4 * math.cos(2.0 * math.pi * 0.11 * t),
    )


# ---------------------------------------------------------------------------


class TestVectorShape(unittest.TestCase):
    """The output vector is the contract. Its length is never negotiable."""

    def setUp(self):
        self.geometry = Geometry.perimeter(4)
        self.measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)

    def test_length_with_no_data_at_all(self):
        vector = self.measurement.vector()
        self.assertEqual(len(vector), self.geometry.n_links)
        self.assertEqual(len(vector), 12)
        self.assertTrue(all(v == 0.0 for v in vector))

    def test_length_when_only_some_links_report(self):
        sigma = sigma_for_snr(self.geometry, 25.0)
        rng = random.Random(3)
        fed = {(0, 1), (2, 3)}
        for i in range(60):
            t = i / RATE_HZ
            for link in self.geometry.links:
                if (link.tx_id, link.rx_id) not in fed:
                    continue
                self.measurement.update(
                    t, link.tx_id, link.rx_id,
                    make_frame(rng, link, None, sigma, i),
                )

        vector = self.measurement.vector()
        self.assertEqual(len(vector), 12)

        ready = self.measurement.ready_mask()
        for index, link in enumerate(self.geometry.links):
            if (link.tx_id, link.rx_id) in fed:
                self.assertTrue(ready[index], f"{link.tx_id}->{link.rx_id} not ready")
            else:
                # Unobserved links are exactly 0.0 -- the neutral element of
                # the data term, documented as "no evidence of disturbance".
                self.assertFalse(ready[index])
                self.assertEqual(vector[index], 0.0)

    def test_unknown_link_raises_rather_than_being_dropped(self):
        rng = random.Random(4)
        frame = make_frame(rng, self.geometry.links[0], None, 0.01, 0)
        with self.assertRaises(KeyError):
            self.measurement.update(0.0, 0, 99, frame)

    def test_ready_mask_separates_dead_links_from_quiet_ones(self):
        """Every link quiet, every link ready -- and 'quiet' bounded by data.

        The bound is relative to the measured spread of these same raw
        variances, not an absolute number. An absolute one is untestable
        here: `vector()` is uncalibrated dB^2, whose scale is a property
        of the generator, so any round number large enough to be safe is
        also large enough to pass for an implementation reporting 25x the
        correct disturbance on every link.
        """
        sigma = sigma_for_snr(self.geometry, 25.0)
        feed(self.measurement, corpus(self.geometry, empty_room, 5, 4.0, sigma, 16))
        vector = self.measurement.vector()
        ready = self.measurement.ready_mask()
        # Every link is quiet, and every link is ready. A caller cannot tell
        # "quiet" from "dead" out of `vector()` alone, which is why this
        # mask exists.
        self.assertTrue(all(ready))
        median = sorted(vector)[len(vector) // 2]
        self.assertGreater(median, 0.0, "an empty room still has a noise floor")
        for index, value in enumerate(vector):
            self.assertLess(
                value, 5.0 * median,
                f"link {index} reads {value:.4g} against a median noise floor "
                f"of {median:.4g} on an EMPTY room",
            )
            self.assertGreater(value, 0.0)


class TestWindowNumerics(unittest.TestCase):
    """Windowing and the Welford recomputation, checked against two-pass."""

    def setUp(self):
        self.geometry = two_node_geometry()
        self.sigma = sigma_for_snr(self.geometry, 25.0)

    def test_windowed_variance_matches_independent_two_pass(self):
        """The cached Welford result equals a two-pass variance of the window.

        This is the test that would catch an incremental 'remove a sample'
        optimisation going wrong, which is the single most tempting change
        to make to this module.
        """
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        rng = random.Random(11)
        link = self.geometry.links[self.geometry.link_index(0, 1)]

        pushed: list[tuple[float, float]] = []
        for i in range(80):
            t = i / RATE_HZ
            frame = make_frame(rng, link, None, self.sigma, i)
            measurement.update(t, 0, 1, frame)
            # Reproduce the strategy's scalar independently: total power in dB.
            pushed.append((t, 10.0 * math.log10(sum(a * a for a in frame.amplitudes))))

            in_window = [v for (ts, v) in pushed if ts >= t - 1.0]
            if len(in_window) < measurement.min_samples:
                continue
            _, std = mean_std(in_window)
            self.assertAlmostEqual(
                measurement.vector()[measurement.geometry.link_index(0, 1)],
                std * std,
                places=12,
                msg=f"window variance mismatch at sample {i}",
            )

    def test_sample_count_tracks_the_time_window(self):
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        rng = random.Random(12)
        link = self.geometry.links[0]
        for i in range(60):
            measurement.update(i / RATE_HZ, 0, 1, make_frame(rng, link, None, self.sigma, i))
        count = measurement.sample_counts()[self.geometry.link_index(0, 1)]
        # 1.0 s at 12.5 Hz, boundary sample kept: 12 intervals + the endpoint.
        self.assertEqual(count, 13)

    def test_an_under_sampled_link_abstains_at_exactly_min_samples_minus_one(self):
        """The `min_samples` gate is load-bearing, so pin it from both sides.

        Nothing else in this file does: the other tests only ever exercise
        links with ZERO samples, whose variance is 0.0 under any
        threshold, so dropping the gate to 2 leaves them all passing. A
        two-sample variance is `(x1-x2)^2/2` with 141% relative spread,
        and `ready_mask()` saying True about one tells the inversion a
        row of noise is real.
        """
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        index = self.geometry.link_index(0, 1)
        link = self.geometry.links[index]
        rng = random.Random(17)
        self.assertGreaterEqual(measurement.min_samples, 3)

        for i in range(measurement.min_samples - 1):
            measurement.update(i / RATE_HZ, 0, 1,
                               make_frame(rng, link, None, self.sigma, i))
        self.assertEqual(measurement.sample_counts()[index], measurement.min_samples - 1)
        self.assertFalse(measurement.ready_mask()[index])
        self.assertEqual(measurement.vector()[index], 0.0)

        i = measurement.min_samples - 1
        measurement.update(i / RATE_HZ, 0, 1, make_frame(rng, link, None, self.sigma, i))
        self.assertTrue(measurement.ready_mask()[index])
        self.assertGreater(measurement.vector()[index], 0.0)

    def test_min_samples_never_falls_below_three(self):
        """Short windows and slow links must not reach a 2-sample variance."""
        for window_s, rate_hz in ((0.1, 12.5), (1.0, 2.0), (0.5, 4.0), (0.05, 1.0)):
            with self.subTest(window_s=window_s, rate_hz=rate_hz):
                measurement = LinkMeasurement(self.geometry, window_s, rate_hz)
                self.assertGreaterEqual(measurement.min_samples, 3)

    def test_sample_count_does_not_flicker_on_exactly_periodic_input(self):
        """The strict-`<` boundary rule has to survive floating point.

        The cutoff is a subtraction and the sample times are a division,
        so whenever `window_s * sample_rate_hz` is an integer a sample
        lands exactly on the boundary and the raw comparison puts it on
        either side depending on the last ulp. The documented convention
        is that it is kept; the count must therefore be constant, not
        oscillating between k and k+1 while the variance oscillates with
        it. 12.5 Hz / 1.0 s -- the configuration every other test uses --
        is precisely the case that cannot see this.
        """
        link = self.geometry.links[0]
        for rate_hz, window_s, expected in ((10.0, 1.0, 11), (20.0, 1.0, 21),
                                            (12.5, 0.8, 11)):
            with self.subTest(rate_hz=rate_hz, window_s=window_s):
                measurement = LinkMeasurement(self.geometry, window_s, rate_hz)
                rng = random.Random(18)
                seen = set()
                for i in range(80):
                    measurement.update(i / rate_hz, 0, 1,
                                       make_frame(rng, link, None, self.sigma, i, 8))
                    if i > 40:
                        seen.add(measurement.sample_counts()[
                            self.geometry.link_index(0, 1)])
                self.assertEqual(seen, {expected})

    def test_the_memory_cap_is_counted_rather_than_silent(self):
        """An under-declared sample rate shortens the window. Say so.

        `_max_window` is a memory backstop sized from the DECLARED rate,
        so declaring 12.5 Hz and feeding 100 Hz turns the advertised 1.0 s
        window into 0.5 s. That cannot be fixed from inside -- the rate is
        an argument, not a measurement -- but it must not be invisible.
        """
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        link = self.geometry.links[0]
        rng = random.Random(19)
        for i in range(300):
            measurement.update(i / 100.0, 0, 1,
                               make_frame(rng, link, None, self.sigma, i, 8))
        index = self.geometry.link_index(0, 1)
        count = measurement.sample_counts()[index]
        self.assertGreater(
            measurement.window_overflows, 0,
            "the cap evicted samples without counting them",
        )
        # And the shortening is real, which is what the counter is warning
        # about: 50 samples at the true 100 Hz is half the window asked for.
        self.assertLess(count / 100.0, 0.75 * measurement.window_s)

    def test_disturbance_is_forgotten_after_about_one_window(self):
        """A disturbance that ends is still visible at 0.3 s and gone at 1.3 s.

        Both halves matter. Forgetting too slowly smears a moving target;
        forgetting too fast starves the variance estimate. One window is
        the specified memory, so the test brackets it.

        Note what the departure itself does: a person leaving is a step
        change in the channel, and a step inside a variance window is a
        *large* disturbance --- larger than the walking that preceded it.
        So the comparison is against the calibrated null, not against the
        walking level, and the first probe is taken 0.3 s after departure
        when the step is still inside the window.
        """
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        measurement.calibrate_from(
            corpus(self.geometry, empty_room, 21, 25.0, self.sigma)
        )
        index = self.geometry.link_index(0, 1)
        null_sigma = measurement.baseline_std_vector()[index]

        rng = random.Random(13)
        link = self.geometry.links[index]
        step = 1.0 / RATE_HZ
        t = 0.0

        for i in range(int(4.0 * RATE_HZ)):
            person = (4.0, 3.0 + 2.0 * math.sin(2.0 * math.pi * t / 11.4))
            measurement.update(t, 0, 1, make_frame(rng, link, person, self.sigma, i))
            t += step
        during = measurement.vector()[index]
        self.assertGreater(
            during / null_sigma, 20.0,
            "the disturbance itself was never significant, so nothing is proved",
        )

        departure = t
        shortly_after = None
        i = 0
        while t < departure + 1.4:
            measurement.update(t, 0, 1, make_frame(rng, link, None, self.sigma, i))
            if shortly_after is None and t >= departure + 0.3:
                shortly_after = measurement.vector()[index]
            t += step
            i += 1
        well_after = measurement.vector()[index]

        self.assertGreater(
            shortly_after / null_sigma, 20.0,
            "0.3 s after departure the window should still hold the disturbance",
        )
        self.assertLess(
            well_after / null_sigma, 6.0,
            f"1.4 s after departure y is still {well_after/null_sigma:.1f} sigma; "
            f"the window is not forgetting",
        )

    def test_link_that_stops_reporting_decays_on_the_global_clock(self):
        """A dead link must fade to 0.0, not hold a stale disturbance.

        Pruning runs against the newest timestamp on ANY link, which is what
        makes this work: link (0,1) receives nothing after the cutover, so
        only the clock carried by link (1,0) can retire its samples.
        """
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        rng = random.Random(14)
        forward = self.geometry.links[self.geometry.link_index(0, 1)]
        reverse = self.geometry.links[self.geometry.link_index(1, 0)]

        t = 0.0
        step = 1.0 / RATE_HZ
        for i in range(40):
            person = (2.0 + 0.7 * t, 3.2)
            measurement.update(t, 0, 1, make_frame(rng, forward, person, self.sigma, i))
            measurement.update(t, 1, 0, make_frame(rng, reverse, person, self.sigma, i))
            t += step

        index = self.geometry.link_index(0, 1)
        self.assertGreater(measurement.vector()[index], 0.0)

        # Now only the reverse link keeps transmitting.
        for i in range(40):
            measurement.update(t, 1, 0, make_frame(rng, reverse, None, self.sigma, i))
            t += step

        self.assertEqual(measurement.vector()[index], 0.0)
        self.assertEqual(measurement.sample_counts()[index], 0)

    def test_clock_restart_recovers_instead_of_failing_silently(self):
        """A stream whose timestamps restart must not zero the output forever.

        Without the discontinuity check this is a total, silent failure:
        every new sample arrives older than the cutoff and is pruned on
        contact, so `vector()` returns zeros and the inversion reads a
        confidently empty room.
        """
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        rng = random.Random(15)
        link = self.geometry.links[0]

        for i in range(60):
            measurement.update(100.0 + i / RATE_HZ, 0, 1,
                               make_frame(rng, link, None, self.sigma, i))
        self.assertTrue(measurement.ready_mask()[self.geometry.link_index(0, 1)])

        # A second recording, clock back at zero.
        for i in range(60):
            measurement.update(i / RATE_HZ, 0, 1,
                               make_frame(rng, link, None, self.sigma, i))
        self.assertTrue(
            measurement.ready_mask()[self.geometry.link_index(0, 1)],
            "clock restart left the link permanently starved",
        )


class TestClockDomains(unittest.TestCase):
    """One link's timing trouble must stay on that link.

    The deployment this is written for is N ESP32s, each stamping its own
    receptions from its own free-running clock, with no NTP between them.
    Three things follow and all three used to be broken by a single
    global pruning clock:

      * a constant inter-node offset is normal, not an error, and must
        cost nothing;
      * a reordered or retransmitted packet is normal too --- the
        recorder in this repo keeps a `reordered` counter and a
        64-packet reorder window --- and must cost at most one sample on
        one link;
      * a genuine stream restart must be recovered from, per link, and
        must be counted so that a mesh quietly falling apart is
        distinguishable from a quiet room.
    """

    def setUp(self):
        self.geometry = Geometry.perimeter(4)
        self.sigma = sigma_for_snr(self.geometry, 22.0, 16)
        self.frames = corpus(self.geometry, walking_corner, 43, 6.0, self.sigma, 16)

    def _fed(self) -> LinkMeasurement:
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        feed(measurement, self.frames)
        return measurement

    def test_one_stale_packet_perturbs_nothing(self):
        """A retransmit carrying its original capture time costs one sample.

        It used to blank all twelve links: `sample_counts` went from
        [13]*12 to [1, 0, 0, ...] and `vector()` to all exact zeros, which
        is the "inversion reads a confidently empty room" failure the
        check was added to prevent, reached from one reordered packet.
        """
        reference = self._fed()
        counts = reference.sample_counts()
        expected = reference.vector()
        self.assertTrue(all(c == 13 for c in counts), counts)
        self.assertGreater(max(abs(v) for v in expected), 1.0)

        measurement = self._fed()
        t_latest = self.frames[-1][0]
        measurement.update(t_latest - 1.5, 0, 1, self.frames[0][3])

        self.assertEqual(measurement.sample_counts(), counts)
        self.assertTrue(all(measurement.ready_mask()))
        for index, (got, want) in enumerate(zip(measurement.vector(), expected)):
            self.assertAlmostEqual(got, want, places=12, msg=f"link {index} moved")
        self.assertEqual(measurement.stale_samples, 1)
        self.assertEqual(measurement.clock_restarts, 0)

    def test_sustained_stale_packets_never_blind_the_mesh(self):
        """One stale packet per superframe, forever. The image must survive.

        Measured on the old global check, the fraction of ready links fell
        from 1.00 to 0.00 as the stale rate rose; the mesh went
        permanently dark with no exception and no flag.
        """
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        previous_t = None
        for t, tx_id, rx_id, frame in self.frames:
            measurement.update(t, tx_id, rx_id, frame)
            if previous_t is not None and t != previous_t:
                measurement.update(t - 1.7, 0, 1, frame)
            previous_t = t
        self.assertTrue(all(measurement.ready_mask()))
        self.assertTrue(all(c == 13 for c in measurement.sample_counts()))
        self.assertGreater(measurement.stale_samples, 50)
        self.assertEqual(measurement.clock_restarts, 0)

    def test_a_constant_node_clock_offset_costs_nothing(self):
        """Node 1's clock is 1.5 s slow. Both link directions must still work.

        Each link is individually monotonic here, so this satisfies the
        module's stated contract exactly --- and under a single global
        clock it produced `sample_counts() == [0, 1]`, `ready_mask() ==
        [False, False]` and `vector() == [0.0, 0.0]` permanently, for a
        person walking straight across the link.
        """
        geometry = two_node_geometry()
        sigma = sigma_for_snr(geometry, 25.0)
        synced = LinkMeasurement(geometry, 1.0, RATE_HZ)
        skewed = LinkMeasurement(geometry, 1.0, RATE_HZ)
        forward = geometry.links[geometry.link_index(0, 1)]
        reverse = geometry.links[geometry.link_index(1, 0)]

        rng = random.Random(16)
        offset = -1.5  # node 1 receives link (0, 1) and stamps it 1.5 s slow
        for i in range(120):
            t = i / RATE_HZ
            person = (4.0, 3.0 + 2.0 * math.sin(2.0 * math.pi * t / 11.4))
            one = make_frame(rng, forward, person, sigma, i)
            two = make_frame(rng, reverse, person, sigma, i)
            synced.update(t, 0, 1, one)
            synced.update(t, 1, 0, two)
            skewed.update(t + offset, 0, 1, one)
            skewed.update(t, 1, 0, two)

        self.assertTrue(all(skewed.ready_mask()), skewed.sample_counts())
        self.assertEqual(skewed.sample_counts(), synced.sample_counts())
        self.assertEqual(skewed.clock_restarts, 0)
        self.assertEqual(skewed.stale_samples, 0)
        for index, (got, want) in enumerate(zip(skewed.vector(), synced.vector())):
            self.assertAlmostEqual(
                got, want, places=12,
                msg=f"link {index}: a constant offset changed the measurement",
            )

    def test_one_node_restarting_its_clock_does_not_touch_the_others(self):
        """Node 0's link reboots to t = 0 while the mesh carries on."""
        measurement = self._fed()
        rebooted = self.geometry.link_index(0, 1)
        others = [i for i in range(self.geometry.n_links) if i != rebooted]
        link = self.geometry.links[rebooted]
        rng = random.Random(20)

        mesh_t = self.frames[-1][0] + 1.0 / RATE_HZ
        for i in range(30):
            for index in others:
                other = self.geometry.links[index]
                measurement.update(mesh_t, other.tx_id, other.rx_id,
                                   make_frame(rng, other, None, self.sigma, i, 16))
            measurement.update(i / RATE_HZ, 0, 1,
                               make_frame(rng, link, None, self.sigma, i, 16))
            mesh_t += 1.0 / RATE_HZ
            counts = measurement.sample_counts()
            for index in others:
                self.assertEqual(
                    counts[index], 13,
                    f"link {index} lost samples because link (0,1) restarted",
                )

        self.assertEqual(measurement.clock_restarts, 1)
        self.assertTrue(measurement.ready_mask()[rebooted],
                        "the restarted link never recovered")

    def test_a_whole_stream_restart_recovers_on_every_link(self):
        """A second recording, clock back at zero: converge on the fresh answer."""
        replayed = self._fed()
        feed(replayed, self.frames)

        fresh = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        feed(fresh, self.frames)

        self.assertEqual(replayed.clock_restarts, self.geometry.n_links)
        self.assertTrue(all(replayed.ready_mask()))
        self.assertEqual(replayed.sample_counts(), fresh.sample_counts())
        for index, (got, want) in enumerate(zip(replayed.vector(), fresh.vector())):
            self.assertAlmostEqual(
                got, want, places=12,
                msg=f"link {index} did not converge on the restarted stream",
            )

    def test_a_late_sample_never_outlives_its_window(self):
        """Out of order by 0.95 s, so no restart -- it must still age out on time.

        Appending blindly at the tail and pruning only the head let such a
        sample survive to 1.91 s with a 1.0 s window, counted in every
        variance in between: the stretched-window bias the time window
        exists to prevent, reintroduced by the sample that needed it most.
        """
        geometry = two_node_geometry()
        sigma = sigma_for_snr(geometry, 25.0)
        measurement = LinkMeasurement(geometry, 1.0, RATE_HZ)
        link = geometry.links[0]
        index = geometry.link_index(0, 1)
        rng = random.Random(21)

        step = 1.0 / RATE_HZ
        t = 0.0
        for i in range(30):
            measurement.update(t, 0, 1, make_frame(rng, link, None, sigma, i))
            t += step
        measurement.update(t - step - 0.95, 0, 1,
                           make_frame(rng, link, None, sigma, 99))

        worst = 0.0
        for i in range(30):
            measurement.update(t, 0, 1, make_frame(rng, link, None, sigma, i))
            oldest = min(ts for ts, _ in measurement._windows[index])
            worst = max(worst, t - oldest)
            t += step
        self.assertLessEqual(
            worst, measurement.window_s + 1e-9,
            f"a sample survived to {worst:.3f} s in a "
            f"{measurement.window_s:.1f} s window",
        )

    def test_a_stopped_stream_reads_as_stopped_once_told_the_time(self):
        """Nothing ages between packets unless the caller says what time it is."""
        geometry = two_node_geometry()
        sigma = sigma_for_snr(geometry, 25.0)
        measurement = LinkMeasurement(geometry, 1.0, RATE_HZ)
        rng = random.Random(22)

        t = 0.0
        for i in range(40):
            for link in geometry.links:
                measurement.update(t, link.tx_id, link.rx_id,
                                   make_frame(rng, link, (4.0, 3.2), sigma, i))
            t += 1.0 / RATE_HZ

        self.assertTrue(all(measurement.ready_mask()))
        held = measurement.vector()
        self.assertGreater(max(held), 0.0)
        # Same call with no time argument: unchanged, as before.
        self.assertEqual(measurement.vector(), held)

        later = t + 3600.0
        self.assertEqual(measurement.vector(later), [0.0] * geometry.n_links)
        self.assertFalse(any(measurement.ready_mask(later)))
        self.assertEqual(measurement.sample_counts(later), [0] * geometry.n_links)

    def test_tdma_slot_timestamps_give_every_link_the_same_window(self):
        """Under the real slot model links do NOT share one timestamp.

        Node *i* transmits in slot *i*, so a link's samples sit up to
        (N-1) slots from its neighbours'. Pruning every link from one
        global timestamp then makes window length a function of node id
        --- measured as 12 samples for slots 0-2 against 13 for slots 3-7
        on an 8-node mesh, an 8% across-link difference fixed by nothing
        but wiring order. RTI compares links against each other, so that
        is a spatial bias, not extra smoothing.
        """
        n_nodes = 4
        slot_s = 1.0 / (RATE_HZ * n_nodes)
        geometry = Geometry.perimeter(n_nodes)
        sigma = sigma_for_snr(geometry, 22.0, 16)
        measurement = LinkMeasurement(geometry, 1.0, RATE_HZ)
        feed(measurement, corpus(geometry, empty_room, 23, 5.0, sigma, 16,
                                 slot_s=slot_s))

        counts = measurement.sample_counts()
        self.assertEqual(
            len(set(counts)), 1,
            f"window length varies with TDMA slot: "
            f"{ {geometry.links[i].tx_id: c for i, c in enumerate(counts)} }",
        )
        self.assertTrue(all(measurement.ready_mask()))


class TestCalibration(unittest.TestCase):
    """Empty-room calibration: the thing that makes `y` mean anything."""

    def setUp(self):
        self.geometry = Geometry.perimeter(4)
        self.sigma = sigma_for_snr(self.geometry, 22.0, 16)

    def test_calibration_removes_unequal_per_link_offsets(self):
        """Links have genuinely different empty-room variance; it must go.

        The baseline offsets here are not synthetic bookkeeping: links in
        this geometry differ in length by more than 2x, so their noise
        floors differ by design. Before calibration the vector is a
        picture of that; after it, it is a picture of nothing.
        """
        calibration_s, fresh_s = 40.0, 24.0
        calibration = corpus(self.geometry, empty_room, 31, calibration_s, self.sigma, 16)
        fresh = corpus(self.geometry, empty_room, 32, fresh_s, self.sigma, 16)

        raw = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        uncalibrated, n_windows = time_averaged_vector(raw, fresh)
        self.assertFalse(raw.is_calibrated)
        self.assertGreater(n_windows, 50)

        calibrated = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        calibrated.calibrate_from(calibration)
        self.assertTrue(calibrated.is_calibrated)
        residual, _ = time_averaged_vector(calibrated, fresh)

        baseline = calibrated.baseline_vector()
        # The offsets really are unequal: spread across links must be a
        # large fraction of their mean, or this test proves nothing.
        mean_b, std_b = mean_std(baseline)
        self.assertGreater(
            std_b / mean_b, 0.25,
            "per-link baselines are too similar for this test to be meaningful",
        )

        # Uncalibrated output carries that spread; calibrated output does not.
        _, spread_raw = mean_std(uncalibrated)
        _, spread_residual = mean_std(residual)
        self.assertLess(
            spread_residual, 0.30 * spread_raw,
            f"calibration barely changed the across-link spread: "
            f"{spread_raw:.4g} -> {spread_residual:.4g}",
        )

        # What is left must be sampling noise, not a surviving bias --- the
        # distinction the rest of this test cannot make on its own, because
        # a 4x reduction is equally consistent with "works, noise-limited"
        # and "half-works". Calibration cannot beat its own sampling error:
        # with roughly `duration / window` independent windows on each side,
        # the floor is `null_sigma * sqrt(1/n_cal + 1/n_fresh)`. Measured
        # across 20/40/80 s calibrations the residual tracks this floor and
        # falls with it while `spread_raw` stays put, which is what makes it
        # noise rather than bias.
        null_sigma = calibrated.baseline_std_vector()
        median_sigma = sorted(null_sigma)[len(null_sigma) // 2]
        floor = median_sigma * math.sqrt(1.0 / calibration_s + 1.0 / fresh_s)
        self.assertLess(
            spread_residual, 1.5 * floor,
            f"residual spread {spread_residual:.4g} exceeds the sampling-noise "
            f"floor {floor:.4g}; something systematic survived calibration",
        )
        for index, value in enumerate(residual):
            self.assertLess(
                abs(value), 0.75 * null_sigma[index],
                f"link {index} time-averaged residual {value:.4g} is "
                f"{abs(value)/null_sigma[index]:.2f} of its null sigma",
            )

    def test_baseline_std_predicts_the_spread_of_y_on_fresh_empty_data(self):
        """`baseline_std_vector` is the null sigma of the `y` actually emitted.

        If it does not predict the spread of `y` in an empty room, the
        significance test downstream is built on sand. Both calibration
        modes, because they emit `y` in different units and only one of
        them used to be described correctly: NORMALIZE divides by the
        baseline, so its null sigma is `spread / baseline`. Returning the
        raw spread there predicted 0.0094 against an observed 0.399 --- a
        factor of 42, and one that differs link to link, so no downstream
        scale factor could absorb it. The suite passed 27/27 with the
        documented contract broken in half the supported configurations
        because this test only ever ran the default mode.
        """
        calibration = corpus(self.geometry, empty_room, 31, 20.0, self.sigma, 16)
        fresh = corpus(self.geometry, empty_room, 33, 20.0, self.sigma, 16)
        index = self.geometry.link_index(0, 1)

        for mode in CalibrationMode:
            with self.subTest(mode=mode.value):
                calibrated = LinkMeasurement(
                    self.geometry, 1.0, RATE_HZ, calibration_mode=mode
                )
                calibrated.calibrate_from(calibration)
                predicted = calibrated.baseline_std_vector()

                observed_y = collect_link_y(calibrated, fresh, 0, 1)
                _, observed = mean_std(observed_y)
                ratio = observed / predicted[index]
                self.assertTrue(
                    0.5 < ratio < 2.0,
                    f"{mode.value}: predicted null sigma {predicted[index]:.4g}, "
                    f"observed {observed:.4g} (ratio {ratio:.2f})",
                )

        # And the raw spread is still reachable, because that is the half
        # of the pair `set_baseline` takes back.
        raw = LinkMeasurement(self.geometry, 1.0, RATE_HZ,
                              calibration_mode=CalibrationMode.NORMALIZE)
        raw.calibrate_from(calibration)
        for scaled, unscaled, base in zip(raw.baseline_std_vector(),
                                          raw.baseline_raw_std_vector(),
                                          raw.baseline_vector()):
            self.assertAlmostEqual(scaled, unscaled / base, places=12)

    def test_calibration_survives_two_recordings_spliced_together(self):
        """Concatenated recordings are the documented way to lengthen a run.

        `sim.py`'s `stream()` restarts its clock at t = 0 on every call,
        and the calibration docstring invites exactly that input. With no
        discontinuity handling on this path the second recording's
        samples arrive out of order behind a head that is never old
        enough to prune, so the window grows to the `_max_window` cap ---
        measured at 50 samples, a 4 s window where 1 s was asked for ---
        and the variance is computed ACROSS the splice. A 3 dB gain
        change between the two recordings (a different day, a different
        AGC setting) then inflated every baseline by 11-13x, silently,
        with `calibrated_mask()` all True; every subsequent `y` reads an
        order of magnitude too low, so a real person never clears
        threshold.
        """
        first = corpus(self.geometry, empty_room, 31, 20.0, self.sigma, 16)
        second = corpus(self.geometry, empty_room, 34, 20.0, self.sigma, 16)
        continuous = [(t + 20.0, tx, rx, f) for t, tx, rx, f in second]

        one_clock = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        one_clock.calibrate_from(list(first) + continuous)

        restarted = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        restarted.calibrate_from(list(first) + list(second))
        self.assertEqual(restarted.clock_restarts, self.geometry.n_links)
        self.assertTrue(all(restarted.calibrated_mask()))

        for index, (got, want) in enumerate(zip(restarted.baseline_vector(),
                                                one_clock.baseline_vector())):
            self.assertLess(
                abs(got - want), 0.10 * want,
                f"link {index}: restarting the clock moved the baseline "
                f"{want:.4g} -> {got:.4g}",
            )

        # The same splice with a 3 dB gain step across it. In dB a gain
        # change is a constant offset, so it must leave the VARIANCE
        # alone; anything else means the discontinuity leaked in.
        gain = 10.0 ** (3.0 / 20.0)
        louder = [
            (t, tx, rx, CsiFrame([a * gain for a in f.amplitudes], f.phases,
                                 f.sample_index, f.sample_rate_hz, node_id=f.node_id))
            for t, tx, rx, f in second
        ]
        stepped = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        stepped.calibrate_from(list(first) + louder)
        for index, (got, want) in enumerate(zip(stepped.baseline_vector(),
                                                restarted.baseline_vector())):
            self.assertLess(
                abs(got - want), 0.05 * want,
                f"link {index}: a 3 dB gain step changed the baseline "
                f"{want:.4g} -> {got:.4g}",
            )

    def test_a_calibration_too_short_to_mean_anything_is_rejected(self):
        """`min_windows` has to count independent windows, not samples.

        Counting per-sample estimates made the default `min_windows=10`
        reachable in `min_samples + 10` samples --- about 1.3 s, one
        window --- and a calibration that short is accepted with
        `calibrated_mask()` all True while its null sigma is 2.4x too
        small. Measured on fresh EMPTY-room data, 17% of windows then
        cleared a +3 sigma gate built on it, against the 0.1% such a gate
        promises.
        """
        for duration_s in (1.3, 2.0, 5.0):
            with self.subTest(duration_s=duration_s):
                short = corpus(self.geometry, empty_room, 31, duration_s,
                               self.sigma, 16)
                measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
                with self.assertRaises(ValueError):
                    measurement.calibrate_from(short)
                self.assertFalse(measurement.is_calibrated)

        # 20 s is enough, and the windows it counted are independent ones:
        # roughly one per `window_s`, not one per sample.
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        measurement.calibrate_from(
            corpus(self.geometry, empty_room, 31, 20.0, self.sigma, 16)
        )
        self.assertTrue(all(measurement.calibrated_mask()))
        with self.assertRaises(ValueError):
            LinkMeasurement(self.geometry, 1.0, RATE_HZ).calibrate_from(
                corpus(self.geometry, empty_room, 31, 20.0, self.sigma, 16),
                min_windows=200,
            )

    def test_uncalibrated_links_are_imputed_and_flagged(self):
        """A link absent from calibration gets the median, and says so.

        Leaving it at 0.0 would make it read as permanently disturbed and
        put a fixed bright ellipse in every image.
        """
        full = corpus(self.geometry, empty_room, 31, 20.0, self.sigma, 16)
        missing = {(0, 1), (1, 0)}
        partial = [row for row in full if (row[1], row[2]) not in missing]

        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        measurement.calibrate_from(partial)

        mask = measurement.calibrated_mask()
        baseline = measurement.baseline_vector()
        for tx_id, rx_id in missing:
            index = self.geometry.link_index(tx_id, rx_id)
            self.assertFalse(mask[index])
            self.assertGreater(baseline[index], 0.0, "imputed baseline must not be 0")

        measured = [baseline[i] for i, ok in enumerate(mask) if ok]
        self.assertTrue(all(mask[i] for i, l in enumerate(self.geometry.links)
                            if (l.tx_id, l.rx_id) not in missing))
        self.assertGreaterEqual(baseline[self.geometry.link_index(0, 1)], min(measured))
        self.assertLessEqual(baseline[self.geometry.link_index(0, 1)], max(measured))

    def test_calibration_on_an_empty_stream_raises(self):
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        with self.assertRaises(ValueError):
            measurement.calibrate_from([])

    def test_set_baseline_round_trips(self):
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        baseline = [0.1 * (i + 1) for i in range(self.geometry.n_links)]
        sigmas = [0.01 * (i + 1) for i in range(self.geometry.n_links)]
        measurement.set_baseline(baseline, sigmas)
        self.assertTrue(measurement.is_calibrated)
        self.assertEqual(measurement.baseline_vector(), baseline)
        self.assertEqual(measurement.baseline_std_vector(), sigmas)
        self.assertTrue(all(measurement.calibrated_mask()))

        with self.assertRaises(ValueError):
            measurement.set_baseline([0.0, 1.0])

    def test_set_baseline_refuses_a_corrupt_calibration(self):
        """The restore-from-disk path is where corrupt input arrives.

        A NaN entry makes SUBTRACT-mode `y` NaN forever and slips past the
        `clamp_negative` comparison; a negative entry is not a variance.
        Both used to be accepted wholesale, with `calibrated_mask()`
        affirming every link afterwards.
        """
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        n_links = self.geometry.n_links
        for bad in (float("nan"), float("inf"), -1.0):
            with self.subTest(bad=bad):
                values = [1.0] * n_links
                values[2] = bad
                with self.assertRaises(ValueError):
                    measurement.set_baseline(values, [0.1] * n_links)
                with self.assertRaises(ValueError):
                    measurement.set_baseline([1.0] * n_links,
                                             [bad] + [0.1] * (n_links - 1))
        self.assertFalse(measurement.is_calibrated)

    def test_a_baseline_with_no_sigma_does_not_claim_to_be_calibrated(self):
        """Omitting `baseline_std` leaves the null sigma unknown. Say so.

        It used to report `calibrated_mask()` all True while
        `baseline_std_vector()` was a vector of exact zeros, so a
        downstream `z = y / sigma` divides by zero on every link and
        nothing in the API says why.
        """
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        measurement.set_baseline([1.0] * self.geometry.n_links)
        self.assertTrue(measurement.is_calibrated)
        self.assertFalse(any(measurement.calibrated_mask()))
        self.assertEqual(measurement.baseline_std_vector(),
                         [0.0] * self.geometry.n_links)

    def test_a_link_that_normalize_cannot_use_is_not_ready(self):
        """A zero baseline in NORMALIZE pins `y` at 0.0 forever. Flag it.

        That link then votes "nobody is on my path" on every image no
        matter what it receives, which is exactly the blind-versus-quiet
        distinction `ready_mask()` exists to make. It reported healthy.
        """
        n_links = self.geometry.n_links
        baseline = [1.0] * n_links
        baseline[0] = 0.0
        measurement = LinkMeasurement(
            self.geometry, 1.0, RATE_HZ, calibration_mode=CalibrationMode.NORMALIZE
        )
        measurement.set_baseline(baseline, [0.1] * n_links)
        feed(measurement, corpus(self.geometry, walking_corner, 43, 4.0, self.sigma, 16))

        self.assertFalse(measurement.calibrated_mask()[0])
        self.assertFalse(measurement.ready_mask()[0])
        self.assertEqual(measurement.vector()[0], 0.0)
        # Its neighbours are unaffected: this is a per-link verdict.
        self.assertTrue(all(measurement.ready_mask()[1:]))

    def test_reset_keeps_calibration_by_default(self):
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        measurement.set_baseline([1.0] * self.geometry.n_links)
        feed(measurement, corpus(self.geometry, empty_room, 31, 4.0, self.sigma, 16))

        measurement.reset()
        self.assertTrue(measurement.is_calibrated)
        self.assertTrue(all(c == 0 for c in measurement.sample_counts()))

        measurement.reset(keep_calibration=False)
        self.assertFalse(measurement.is_calibrated)


class TestDisturbanceResponse(unittest.TestCase):
    """The core claim: a disturbed link goes positive, a quiet one does not."""

    @classmethod
    def setUpClass(cls):
        cls.geometry = Geometry.perimeter(4)
        cls.sigma = sigma_for_snr(cls.geometry, 22.0, 16)
        cls.calibration = corpus(cls.geometry, empty_room, 41, 20.0, cls.sigma, 16)

    def _calibrated(self, **kwargs):
        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ, **kwargs)
        measurement.calibrate_from(self.calibration)
        return measurement

    def test_empty_room_gives_approximately_zero_on_every_link(self):
        measurement = self._calibrated()
        feed(measurement, corpus(self.geometry, empty_room, 42, 12.0, self.sigma, 16))
        vector = measurement.vector()
        null_sigma = measurement.baseline_std_vector()
        for index, value in enumerate(vector):
            self.assertLess(
                abs(value) / null_sigma[index], 6.0,
                f"link {index}: |y| = {abs(value):.4g} is "
                f"{abs(value)/null_sigma[index]:.1f} sigma on an EMPTY room",
            )

    def test_near_link_goes_positive_and_far_link_stays_small(self):
        """A body in the corner lights the links it stands on, not the rest."""
        measurement = self._calibrated()
        feed(measurement, corpus(self.geometry, walking_corner, 43, 12.0, self.sigma, 16))
        vector = measurement.vector()
        null_sigma = measurement.baseline_std_vector()

        person = walking_corner(12.0 - 1.0 / RATE_HZ)
        excess = [link.excess_path_m(*person) for link in self.geometry.links]
        near = min(range(len(excess)), key=excess.__getitem__)
        far = max(range(len(excess)), key=excess.__getitem__)

        self.assertGreater(
            vector[near], 0.0,
            f"link on the body (excess {excess[near]:.2f} m) did not go positive",
        )
        self.assertGreater(
            vector[near] / null_sigma[near], 20.0,
            "disturbance is not significant against the calibrated null",
        )
        self.assertLess(
            vector[far], 0.25 * vector[near],
            f"far link (excess {excess[far]:.2f} m) y={vector[far]:.4g} is not "
            f"small against the near link y={vector[near]:.4g}",
        )

    def test_negative_entries_are_allowed_by_default(self):
        """Unclamped is the documented default; prove it is really unclamped.

        Clamping turns zero-mean estimator noise into a positive floor on
        every link at once, which the inversion reads as a faint
        disturbance filling the room.
        """
        measurement = self._calibrated()
        index = self.geometry.link_index(0, 1)
        values = collect_link_y(
            measurement, corpus(self.geometry, empty_room, 42, 12.0, self.sigma, 16), 0, 1
        )
        negatives = sum(1 for v in values if v < 0.0)
        self.assertGreater(
            negatives, 0.2 * len(values),
            "an empty room should put roughly half the windows below baseline",
        )
        mean, _ = mean_std(values)
        self.assertLess(abs(mean), 0.5 * mean_std(values)[1],
                        "unclamped null should be near zero-mean")

    def test_clamp_negative_option_removes_them(self):
        measurement = self._calibrated(clamp_negative=True)
        values = collect_link_y(
            measurement, corpus(self.geometry, empty_room, 42, 12.0, self.sigma, 16), 0, 1
        )
        self.assertTrue(all(v >= 0.0 for v in values))
        mean, _ = mean_std(values)
        # And demonstrate the cost: the clamped null is biased positive.
        self.assertGreater(mean, 0.0)

    def test_normalize_mode_is_fractional_excess_variance(self):
        subtract = self._calibrated(calibration_mode=CalibrationMode.SUBTRACT)
        normalize = self._calibrated(calibration_mode=CalibrationMode.NORMALIZE)
        frames = corpus(self.geometry, walking_corner, 43, 12.0, self.sigma, 16)
        feed(subtract, frames)
        feed(normalize, frames)

        baseline = subtract.baseline_vector()
        y_sub = subtract.vector()
        y_norm = normalize.vector()
        for index in range(self.geometry.n_links):
            self.assertAlmostEqual(
                y_norm[index], y_sub[index] / baseline[index], places=9,
                msg=f"link {index}: normalize is not subtract/baseline",
            )

    def test_normalize_equalises_link_scales(self):
        """The reason NORMALIZE exists: it tames unequal link noise floors."""
        subtract = self._calibrated(calibration_mode=CalibrationMode.SUBTRACT)
        normalize = self._calibrated(calibration_mode=CalibrationMode.NORMALIZE)
        frames = corpus(self.geometry, empty_room, 42, 12.0, self.sigma, 16)

        spread_sub = []
        spread_norm = []
        for tx_id, rx_id in ((0, 1), (0, 2), (1, 3)):
            subtract.reset()
            spread_sub.append(
                mean_std(collect_link_y(subtract, frames, tx_id, rx_id))[1]
            )
            normalize.reset()
            spread_norm.append(
                mean_std(collect_link_y(normalize, frames, tx_id, rx_id))[1]
            )

        ratio_sub = max(spread_sub) / min(spread_sub)
        ratio_norm = max(spread_norm) / min(spread_norm)
        self.assertLess(
            ratio_norm, ratio_sub,
            f"normalize did not equalise noise scales: {ratio_sub:.2f} -> "
            f"{ratio_norm:.2f}",
        )


class TestDroppedSamples(unittest.TestCase):
    """A bad frame must leave no trace, not a sentinel-shaped outlier."""

    def setUp(self):
        self.geometry = two_node_geometry()
        self.sigma = sigma_for_snr(self.geometry, 25.0)

    def test_nan_and_zero_frames_are_dropped_not_substituted(self):
        for signal in (TotalPowerSignal(), StaticSuppressedSignal()):
            with self.subTest(signal=signal.name):
                clean = LinkMeasurement(self.geometry, 1.0, RATE_HZ, signal=signal)
                dirty = LinkMeasurement(self.geometry, 1.0, RATE_HZ, signal=signal)
                rng = random.Random(51)
                link = self.geometry.links[0]

                frames = [make_frame(rng, link, None, self.sigma, i) for i in range(40)]
                for i, frame in enumerate(frames):
                    clean.update(i / RATE_HZ, 0, 1, frame)
                    dirty.update(i / RATE_HZ, 0, 1, frame)
                    if i == 20:
                        bad = CsiFrame([float("nan")] * N_SUB, [0.0] * N_SUB,
                                       i, RATE_HZ, node_id=1)
                        dirty.update(i / RATE_HZ, 0, 1, bad)
                        zero = CsiFrame([0.0] * N_SUB, [0.0] * N_SUB,
                                        i, RATE_HZ, node_id=1)
                        dirty.update(i / RATE_HZ, 0, 1, zero)

                index = self.geometry.link_index(0, 1)
                self.assertEqual(
                    clean.sample_counts()[index], dirty.sample_counts()[index],
                    "a dropped frame changed the window length",
                )
                self.assertAlmostEqual(
                    clean.vector()[index], dirty.vector()[index], places=9,
                    msg="a dropped frame perturbed the variance",
                )

    def test_a_frame_whose_power_overflows_is_dropped_too(self):
        """Every amplitude finite, their squares not. That must not pass.

        `[1e200] * 56` sums to `inf`, which clears a `total > 0.0` gate,
        and `10*log10(inf)` is `inf`. One of those in a window makes
        Welford's `m2` NaN, and NaN also slips past the `clamp_negative`
        comparison, so `vector()` returns NaN while `ready_mask()` says
        True. During calibration it is permanent: the baseline is NaN and
        every later `y` on that link is NaN forever.
        """
        overflowing = CsiFrame([1e200] * N_SUB, [0.0] * N_SUB, 0, RATE_HZ, node_id=1)
        self.assertTrue(all(math.isfinite(a) for a in overflowing.amplitudes))

        for signal in (TotalPowerSignal(), StaticSuppressedSignal()):
            with self.subTest(signal=signal.name):
                clean = LinkMeasurement(self.geometry, 1.0, RATE_HZ, signal=signal)
                dirty = LinkMeasurement(self.geometry, 1.0, RATE_HZ, signal=signal)
                rng = random.Random(52)
                link = self.geometry.links[0]
                index = self.geometry.link_index(0, 1)

                for i in range(20):
                    frame = make_frame(rng, link, None, self.sigma, i)
                    clean.update(i / RATE_HZ, 0, 1, frame)
                    dirty.update(i / RATE_HZ, 0, 1, frame)
                    if i == 10:
                        dirty.update(i / RATE_HZ, 0, 1, overflowing)

                self.assertEqual(clean.sample_counts()[index],
                                 dirty.sample_counts()[index])
                self.assertTrue(math.isfinite(dirty.vector()[index]))
                self.assertAlmostEqual(clean.vector()[index], dirty.vector()[index],
                                       places=9)

        # And the calibration path, where the damage used to be permanent.
        rng = random.Random(53)
        stream_with_overflow = []
        for i in range(300):
            t = i / RATE_HZ
            for link in self.geometry.links:
                stream_with_overflow.append(
                    (t, link.tx_id, link.rx_id,
                     make_frame(rng, link, None, self.sigma, i))
                )
            if i == 100:
                stream_with_overflow.append((t, 0, 1, overflowing))

        measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ)
        measurement.calibrate_from(stream_with_overflow)
        for value in measurement.baseline_vector() + measurement.baseline_std_vector():
            self.assertTrue(math.isfinite(value), "an overflowing frame reached the baseline")
        self.assertTrue(all(measurement.calibrated_mask()))

    def test_a_change_in_subcarrier_count_is_not_a_disturbance(self):
        """ESP-IDF hands back different CSI lengths. That is not motion.

        Summing `amp^2` over whatever subcarriers arrive makes the dB
        level a function of the COUNT: 56 -> 52 is a 0.32 dB step, which
        inside a variance window whose genuine spread is an order of
        magnitude smaller reads as a large, confident disturbance ---
        measured at 19.2 null sigma for 56 -> 52 and 81.2 for 56 -> 48,
        against the 6 sigma this suite calls "approximately zero" and the
        20 sigma it calls "a real person". The room was empty throughout.
        """
        calibration = corpus(self.geometry, empty_room, 71, 25.0, self.sigma)
        for signal in (TotalPowerSignal(), StaticSuppressedSignal()):
            for n_short in (52, 48):
                with self.subTest(signal=signal.name, n_short=n_short):
                    measurement = LinkMeasurement(self.geometry, 1.0, RATE_HZ,
                                                  signal=signal)
                    measurement.calibrate_from(calibration)
                    index = self.geometry.link_index(0, 1)
                    null_sigma = measurement.baseline_std_vector()[index]
                    link = self.geometry.links[index]
                    rng = random.Random(54)

                    t = 0.0
                    for i in range(40):
                        measurement.update(t, 0, 1,
                                           make_frame(rng, link, None, self.sigma, i))
                        t += 1.0 / RATE_HZ
                    steady = measurement.vector()[index] / null_sigma

                    peak = abs(steady)
                    for i in range(2 * int(RATE_HZ)):
                        full = make_frame(rng, link, None, self.sigma, i)
                        short = CsiFrame(full.amplitudes[:n_short],
                                         full.phases[:n_short], i, RATE_HZ, node_id=1)
                        measurement.update(t, 0, 1, short)
                        t += 1.0 / RATE_HZ
                        peak = max(peak, abs(measurement.vector()[index] / null_sigma))

                    self.assertLess(
                        peak, 6.0,
                        f"{signal.name}: switching 56 -> {n_short} subcarriers in an "
                        f"EMPTY room peaked at {peak:.1f} null sigma (steady "
                        f"{steady:.1f})",
                    )


class TestDeterminism(unittest.TestCase):
    def test_identical_input_gives_identical_output(self):
        geometry = Geometry.perimeter(4)
        sigma = sigma_for_snr(geometry, 22.0, 16)
        frames = corpus(geometry, walking_corner, 61, 8.0, sigma, 16)
        results = []
        for _ in range(2):
            measurement = LinkMeasurement(geometry, 1.0, RATE_HZ)
            feed(measurement, frames)
            results.append(measurement.vector())
        self.assertEqual(results[0], results[1])


# ---------------------------------------------------------------------------
# The deliverable: strategy (a) vs strategy (b) on identical input.
# ---------------------------------------------------------------------------


class _LinearPowerState(LinkSignalState):
    """Total received power in LINEAR units -- the control, not a product."""

    __slots__ = ()

    def push(self, frame: CsiFrame) -> float | None:
        total = sum(a * a for a in frame.amplitudes)
        return total if total > 0.0 else None


class _LinearPowerSignal(LinkSignal):
    """Deliberately wrong variant, kept as a regression guard. See below."""

    __slots__ = ()

    @property
    def name(self) -> str:
        return "total_power_linear"

    def new_state(self, sample_rate_hz: float) -> LinkSignalState:
        return _LinearPowerState()


class _UnweightedMeanState(LinkSignalState):
    """`StaticSuppressor` residuals averaged with NO weighting -- the control.

    Identical to `_StaticSuppressedState` except that `fuse`'s
    variance-derived weights are replaced by a flat mean, so the
    difference between the two measures exactly what the weighting buys
    and nothing else.
    """

    __slots__ = ("_sample_rate_hz", "_corner_hz", "_suppressor")

    def __init__(self, sample_rate_hz: float, corner_hz: float) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._corner_hz = corner_hz
        self._suppressor = None

    def push(self, frame: CsiFrame) -> float | None:
        if frame.n_subcarriers == 0:
            return None
        if not all(math.isfinite(a) for a in frame.amplitudes):
            return None
        if sum(a * a for a in frame.amplitudes) <= 0.0:
            return None
        if self._suppressor is None:
            self._suppressor = StaticSuppressor(
                frame.n_subcarriers, self._sample_rate_hz, self._corner_hz
            )
            self._suppressor.process(frame)
            return None
        residuals, _ = self._suppressor.process(frame)
        if not residuals:
            return None
        value = sum(residuals) / len(residuals)
        return value if math.isfinite(value) else None

    def reset(self) -> None:
        self._suppressor = None


class _UnweightedMeanSignal(LinkSignal):
    """Control for the claim that `fuse`'s weighting barely matters here."""

    __slots__ = ("corner_hz",)

    def __init__(self, corner_hz: float = 0.03) -> None:
        self.corner_hz = corner_hz

    @property
    def name(self) -> str:
        return "unweighted_mean"

    def new_state(self, sample_rate_hz: float) -> LinkSignalState:
        return _UnweightedMeanState(sample_rate_hz, self.corner_hz)


def d_prime(measurement_factory, calibration, null, disturbed, tx_id, rx_id):
    """Detectability of the disturbance against the empty-room null.

    `(mean y when disturbed - mean y when quiet) / sd(y when quiet)`.

    This is the only fair way to compare the two strategies: their outputs
    are in different units entirely (dB^2 against squared linear amplitude),
    so raw `y` magnitudes are not comparable and the ratio of standard
    deviations is. d' is what actually matters to the inversion, which only
    ever sees a link rise above its own noise.
    """
    measurement = measurement_factory()
    measurement.calibrate_from(calibration)
    quiet = collect_link_y(measurement, null, tx_id, rx_id)
    loud = collect_link_y(measurement, disturbed, tx_id, rx_id)
    mean_quiet, sd_quiet = mean_std(quiet)
    mean_loud, _ = mean_std(loud)
    if sd_quiet <= 0.0:
        return float("inf")
    return (mean_loud - mean_quiet) / sd_quiet


class TestSignalStrategyComparison(unittest.TestCase):
    """Both strategies respond, and this reports how their sensitivity differs.

    SYNTHETIC throughout. Every number below is a property of the generator
    at the top of this file -- in particular of its fixed-gain receiver,
    which has no AGC -- and not a measurement of hardware.
    """

    @classmethod
    def setUpClass(cls):
        cls.geometry = two_node_geometry()
        cls.rows: list[tuple] = []

    @classmethod
    def tearDownClass(cls):
        if not cls.rows:
            return
        print("\n\n  SYNTHETIC: link-signal sensitivity, d' against the "
              "empty-room null")
        print(f"  {'scenario':<10}{'SNR dB':>7}{'strategy':>28}{'d_prime':>10}")
        print("  " + "-" * 55)
        for scenario, snr, name, value in cls.rows:
            print(f"  {scenario:<10}{snr:>7.0f}{name:>28}{value:>10.1f}")
        print()

    def _record(self, scenario, snr_db, name, value):
        type(self).rows.append((scenario, snr_db, name, value))

    def test_both_strategies_respond_to_gross_motion(self):
        """Walking: both respond, and (a) leads (b) by about 1.3-1.6x.

        Measured across four independent seed sets at 30/20/10 dB, the
        ratio stayed in 1.27-1.60 every time, so the ordering is a property
        of the signals and not of one lucky draw. The headline is that
        static suppression plus 56-subcarrier fusion does not merely fail
        to help for gross motion --- it costs a little.

        Note the absolute scale: d' here runs into the hundreds and
        thousands, which means the walking regime is signal-dominated and
        the measurement stage is nowhere near the bottleneck. These numbers
        separate the strategies but say nothing about achievable
        localization accuracy, which the inversion and the geometry set.
        """
        for snr_db in (30.0, 20.0, 10.0):
            sigma = sigma_for_snr(self.geometry, snr_db)
            calibration = corpus(self.geometry, empty_room, 71, 25.0, sigma)
            null = corpus(self.geometry, empty_room, 72, 25.0, sigma)
            disturbed = corpus(self.geometry, walking_across, 73, 25.0, sigma)

            scores = {}
            for signal in (TotalPowerSignal(), StaticSuppressedSignal()):
                value = d_prime(
                    lambda s=signal: LinkMeasurement(self.geometry, 1.0, RATE_HZ, signal=s),
                    calibration, null, disturbed, 0, 1,
                )
                scores[signal.name] = value
                self._record("walking", snr_db, signal.name, value)
                self.assertGreater(
                    value, 5.0,
                    f"{signal.name} failed to respond to walking at {snr_db:.0f} dB",
                )

            # The finding: static suppression plus 56-subcarrier fusion is a
            # small net LOSS against plain total power for gross motion.
            ratio = (scores["total_power_db"]
                     / scores["static_suppressed_amplitude"])
            self.assertTrue(
                1.1 < ratio < 2.2,
                f"total_power_db / static_suppressed was {ratio:.2f} at "
                f"{snr_db:.0f} dB, outside the 1.27-1.60 measured over four "
                f"seed sets",
            )

    def test_weak_disturbance_separates_the_strategies(self):
        """A still, breathing person: (a) holds up, (b) collapses with SNR.

        Read the mechanism before reading the numbers --- see
        `test_dB_is_required_not_cosmetic` and the class docstring. Most of
        (a)'s advantage here is NOT chest-motion sensitivity: it is the
        shadowing-induced power drop raising the dB-domain noise variance
        on an attenuated link. That still yields a positive `y` for a
        present person, which is more than VRTI classically promises, but
        it is not motion detection and it depends on the receiver having
        no AGC.
        """
        scores: dict[tuple[float, str], float] = {}
        for snr_db in (30.0, 20.0):
            sigma = sigma_for_snr(self.geometry, snr_db)
            calibration = corpus(self.geometry, empty_room, 71, 25.0, sigma)
            null = corpus(self.geometry, empty_room, 72, 25.0, sigma)
            disturbed = corpus(self.geometry, breathing_still, 74, 25.0, sigma)

            for signal in (TotalPowerSignal(), StaticSuppressedSignal()):
                value = d_prime(
                    lambda s=signal: LinkMeasurement(self.geometry, 1.0, RATE_HZ, signal=s),
                    calibration, null, disturbed, 0, 1,
                )
                scores[(snr_db, signal.name)] = value
                self._record("breathing", snr_db, signal.name, value)

        # At 30 dB both see the still person.
        self.assertGreater(scores[(30.0, "total_power_db")], 5.0)
        self.assertGreater(scores[(30.0, "static_suppressed_amplitude")], 2.0)
        # By 20 dB only (a) still does, and by a wide margin.
        self.assertGreater(
            scores[(20.0, "total_power_db")],
            3.0 * scores[(20.0, "static_suppressed_amplitude")],
            "expected total power to lead decisively on the weak disturbance",
        )

    def test_fuse_weighting_is_nearly_irrelevant_on_this_bandwidth(self):
        """Replacing `fuse` with a flat mean costs a few percent, not nothing.

        `StaticSuppressedSignal`'s docstring reasons from RESEARCH.md §5a
        that at 17.5 MHz occupied bandwidth the subcarriers move together,
        so variance-derived weights have almost nothing to select between.
        That reasoning is sound and the conclusion holds --- but the
        number has to be in the repo, not in someone's memory: it was
        documented as "under 1%" and measures at 2.0-4.7%, with the
        weighting consistently helping rather than being neutral.

        The band asserted here is deliberately wide. The point is the
        order of magnitude: a few percent, nowhere near the 1.3-1.6x by
        which plain total power beats both of them.
        """
        rows = []
        for scenario, person_at, seed in (("walking", walking_across, 73),
                                          ("breathing", breathing_still, 74)):
            for snr_db in (20.0, 10.0):
                sigma = sigma_for_snr(self.geometry, snr_db)
                calibration = corpus(self.geometry, empty_room, 71, 25.0, sigma)
                null = corpus(self.geometry, empty_room, 72, 25.0, sigma)
                disturbed = corpus(self.geometry, person_at, seed, 25.0, sigma)

                scores = {}
                for signal in (StaticSuppressedSignal(), _UnweightedMeanSignal()):
                    scores[signal.name] = d_prime(
                        lambda s=signal: LinkMeasurement(
                            self.geometry, 1.0, RATE_HZ, signal=s),
                        calibration, null, disturbed, 0, 1,
                    )
                weighted = scores["static_suppressed_amplitude"]
                flat = scores["unweighted_mean"]
                self.assertGreater(weighted, 0.0)
                self._record(scenario, snr_db, "unweighted_mean(control)", flat)
                rows.append((scenario, snr_db, 100.0 * (weighted - flat) / weighted))

        for scenario, snr_db, gain_pct in rows:
            self.assertTrue(
                0.0 < gain_pct < 10.0,
                f"{scenario} at {snr_db:.0f} dB: weighting changed d' by "
                f"{gain_pct:+.1f}%, outside the measured 2.0-4.7% band",
            )

    def test_dB_is_required_not_cosmetic(self):
        """Linear power gives a NEGATIVE d': the wrong sign, not just worse.

        This is the regression guard for the single most consequential
        choice in `TotalPowerSignal`. A body shadowing a link drops its mean
        power by several dB. Variance of LINEAR power scales with the square
        of mean power, so the shadowed link's variance goes DOWN --- the
        inversion sees negative disturbance and pushes the image AWAY from
        the person. Working in dB makes the measurement a fractional power
        change, which is scale-free and keeps the sign right.

        What makes this a trap rather than merely a worse option is the
        second half of the test: linear power is strongly POSITIVE for a
        walking person. It would sail through any test built on gross
        motion and invert only for someone nearly stationary --- the exact
        case a localizer must not get confidently backwards. Measured
        negative on all 12 breathing runs across four seed sets (-0.7 to
        -2.4) and positive on all 12 walking runs.
        """
        snr_db = 20.0
        sigma = sigma_for_snr(self.geometry, snr_db)
        calibration = corpus(self.geometry, empty_room, 71, 25.0, sigma)
        null = corpus(self.geometry, empty_room, 72, 25.0, sigma)
        breathing = corpus(self.geometry, breathing_still, 74, 25.0, sigma)
        walking = corpus(self.geometry, walking_across, 73, 25.0, sigma)

        def score(signal, disturbed):
            return d_prime(
                lambda s=signal: LinkMeasurement(self.geometry, 1.0, RATE_HZ, signal=s),
                calibration, null, disturbed, 0, 1,
            )

        in_db = score(TotalPowerSignal(), breathing)
        linear_breathing = score(_LinearPowerSignal(), breathing)
        linear_walking = score(_LinearPowerSignal(), walking)
        self._record("breathing", snr_db, "total_power_linear(control)",
                     linear_breathing)

        self.assertGreater(in_db, 3.0)
        self.assertLess(
            linear_breathing, 0.0,
            f"linear power was expected to invert sign on a near-stationary "
            f"person, got d'={linear_breathing:.2f}",
        )
        self.assertGreater(
            linear_walking, 10.0,
            f"linear power should look fine on walking --- that is what makes "
            f"it a trap --- but scored d'={linear_walking:.2f}",
        )


class TestCost(unittest.TestCase):
    """Compute cost, measured. Printed because P2 has to budget for it.

    The number that matters is the COLD `vector()`: in the pipeline a
    superframe of `update()` precedes every image, so every link is dirty
    and Welford genuinely re-runs. Timing `vector()` in a tight loop
    instead measures the dirty-flag cache after the first iteration --- it
    came out 13x cheaper, identical for both strategies, and told nobody
    anything about the cost it was labelled with.
    """

    def test_report_update_and_vector_cost(self):
        geometry = Geometry.perimeter(8)
        sigma = sigma_for_snr(geometry, 22.0, N_SUB)
        rng = random.Random(81)
        superframes = [
            [(i / RATE_HZ, link.tx_id, link.rx_id,
              make_frame(rng, link, (4.0, 3.0), sigma, i))
             for link in geometry.links]
            for i in range(30)
        ]
        frames = [row for superframe in superframes for row in superframe]

        results = []
        for signal in (TotalPowerSignal(), StaticSuppressedSignal()):
            measurement = LinkMeasurement(geometry, 1.0, RATE_HZ, signal=signal)
            start = time.perf_counter()
            for t, tx_id, rx_id, frame in frames:
                measurement.update(t, tx_id, rx_id, frame)
            update_s = (time.perf_counter() - start) / len(frames)

            # Cold: every link dirty, which is the state a superframe of
            # `update()` leaves behind and therefore the only state the
            # pipeline ever calls `vector()` in. Forced through the dirty
            # flags rather than by feeding a superframe first, so the
            # number is the image cost alone with no update cost mixed in
            # and no subtraction to get it wrong.
            n_links = geometry.n_links
            repeats = 200
            start = time.perf_counter()
            for _ in range(repeats):
                measurement._dirty = [True] * n_links
                measurement.vector()
            cold_s = (time.perf_counter() - start) / repeats
            start = time.perf_counter()
            for _ in range(repeats):
                measurement._dirty = [True] * n_links
            cold_s -= (time.perf_counter() - start) / repeats

            start = time.perf_counter()
            for _ in range(200):
                measurement.vector()
            cached_s = (time.perf_counter() - start) / 200

            results.append((signal.name, update_s, cold_s, cached_s))
            self.assertGreater(len(measurement.vector()), 0)
            # The cache really is a cache: if these ever match, the dirty
            # flags have stopped working and this column is meaningless.
            self.assertLess(cached_s, 0.5 * cold_s)

        print(f"\n\n  SYNTHETIC: cost at {geometry.n_nodes} nodes / "
              f"{geometry.n_links} links / {N_SUB} subcarriers, CPython")
        print(f"  {'strategy':>28}{'us/update':>12}{'ms/vector()':>14}"
              f"{'ms cached':>12}")
        print("  " + "-" * 66)
        for name, update_s, cold_s, cached_s in results:
            print(f"  {name:>28}{update_s*1e6:>12.1f}{cold_s*1e3:>14.3f}"
                  f"{cached_s*1e3:>12.3f}")
        print("  ms/vector() is one image with every link dirty, which is the\n"
              "  state after a superframe of updates and so the only one the\n"
              "  pipeline is ever in.")
        print()


if __name__ == "__main__":
    unittest.main()
