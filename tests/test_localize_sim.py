"""Tests for the multi-link synthetic CSI source.

The simulator is the ground truth every other localization module is scored
against, so these tests are deliberately about *numbers*, not about plumbing:
the TDMA rate law computed from the emitted stream, shadowing measured
against an explicitly constructed geometry, breathing recovered with the
project's own spectral estimator, packet loss counted against its binomial
error bar.

Several tests assert on physics that is counter-intuitive and would
otherwise be mistaken for a bug — above all the forward-scatter null, where a
person standing exactly on a link's line of sight produces *no* breathing
modulation on that link at all.
"""

from __future__ import annotations

import cmath
import math
import unittest

from wifisense.dsp import band_peak
from wifisense.localize.geometry import Geometry, Node, PixelGrid, Room
from wifisense.localize.sim import (
    MultiLinkSimulator,
    RandomWaypointMotion,
    SimConfig,
    StaticMotion,
    WaypointMotion,
    slot_ms_for_aggregate_rate,
    slot_ms_for_per_link_rate,
)


def perpendicular_distance(link, x: float, y: float) -> float:
    """Distance from (x, y) to the infinite line through a link's endpoints.

    Independent of the simulator's own excess-path machinery on purpose: a
    test that selects links with the same function the module under test uses
    to shadow them proves less than it looks.
    """
    (ax, ay), (bx, by) = link.tx, link.rx
    return abs((bx - ax) * (ay - y) - (ax - x) * (by - ay)) / link.length_m


def two_node_geometry() -> Geometry:
    """One horizontal link at y = 3, from (1, 3) to (7, 3).

    Built by hand rather than from `Geometry.perimeter` so the line of sight
    is a line a test can reason about in its head.
    """
    room = Room(8.0, 6.0)
    return Geometry(
        room,
        [Node(0, 1.0, 3.0), Node(1, 7.0, 3.0)],
        PixelGrid(room, 0.5),
        lambda_m=0.3,
    )


def mean_amplitude_series(
    sim: MultiLinkSimulator, duration_s: float
) -> dict[tuple[int, int], list[float]]:
    """Per-link time series of the subcarrier-mean amplitude."""
    series: dict[tuple[int, int], list[float]] = {}
    for _t, tx, rx, frame in sim.stream(duration_s):
        series.setdefault((tx, rx), []).append(
            sum(frame.amplitudes) / len(frame.amplitudes)
        )
    return series


class TestTdmaTiming(unittest.TestCase):
    """The rate law is the whole point of the phase; test it from the stream."""

    def test_per_link_rate_equals_one_over_n_times_slot(self) -> None:
        for n_nodes in (4, 6, 8):
            for slot_ms in (10.0, 20.0, 50.0):
                with self.subTest(n_nodes=n_nodes, slot_ms=slot_ms):
                    geo = Geometry.perimeter(n_nodes)
                    sim = MultiLinkSimulator(
                        geo, SimConfig(seed=1, slot_ms=slot_ms, packet_loss=0.0)
                    )
                    expected = 1.0 / (n_nodes * slot_ms / 1000.0)

                    stamps: dict[tuple[int, int], list[float]] = {}
                    for t, tx, rx, frame in sim.stream(4.0):
                        stamps.setdefault((tx, rx), []).append(t)
                        self.assertAlmostEqual(frame.sample_rate_hz, expected, places=9)

                    self.assertEqual(len(stamps), geo.n_links)
                    for link_id, ts in stamps.items():
                        self.assertGreater(len(ts), 2, link_id)
                        # Measured from the stream, not from the config.
                        rate = (len(ts) - 1) / (ts[-1] - ts[0])
                        self.assertAlmostEqual(rate, expected, places=6)
                        for a, b in zip(ts, ts[1:]):
                            self.assertAlmostEqual(b - a, 1.0 / expected, places=9)

    def test_superframe_structure(self) -> None:
        """One transmitter per slot, everyone else receives, once per frame."""
        geo = Geometry.perimeter(6)
        sim = MultiLinkSimulator(geo, SimConfig(seed=2, packet_loss=0.0))

        by_time: dict[float, list[tuple[int, int, int]]] = {}
        for t, tx, rx, frame in sim.stream(sim.superframe_s * 3.0):
            self.assertEqual(frame.node_id, rx, "node_id must be the receiver")
            by_time.setdefault(t, []).append((tx, rx, frame.sample_index))

        self.assertEqual(len(by_time), 6 * 3, "one slot per node per superframe")
        for t, packets in by_time.items():
            transmitters = {tx for tx, _rx, _m in packets}
            receivers = [rx for _tx, rx, _m in packets]
            self.assertEqual(len(transmitters), 1, f"t={t} has >1 transmitter")
            self.assertEqual(len(receivers), 5, "every other node receives")
            self.assertEqual(len(set(receivers)), 5, "no duplicate receivers")
            self.assertNotIn(transmitters.pop(), receivers, "a node cannot hear itself")
            # sample_index is the superframe counter, so it is shared by the
            # whole superframe and increments once per superframe.
            self.assertEqual(len({m for _tx, _rx, m in packets}), 1)

        indices = sorted({m for pkts in by_time.values() for _tx, _rx, m in pkts})
        self.assertEqual(indices, [0, 1, 2])

    def test_every_ordered_pair_is_sampled_once_per_superframe(self) -> None:
        geo = Geometry.perimeter(5)
        sim = MultiLinkSimulator(geo, SimConfig(seed=3, packet_loss=0.0))
        seen = [(tx, rx) for _t, tx, rx, _f in sim.stream(sim.superframe_s)]
        self.assertEqual(len(seen), geo.n_links)
        self.assertEqual(
            sorted(seen), sorted((l.tx_id, l.rx_id) for l in geo.links)
        )

    def test_adding_nodes_costs_per_link_rate(self) -> None:
        """The trade RESEARCH.md §0 warns about, stated as an assertion."""
        rates = {}
        for n_nodes in (4, 6, 8, 12):
            sim = MultiLinkSimulator(
                Geometry.perimeter(n_nodes), SimConfig(seed=4, slot_ms=20.0)
            )
            rates[n_nodes] = (sim.per_link_rate_hz, sim.aggregate_packet_rate_hz)

        per_link = [rates[n][0] for n in (4, 6, 8, 12)]
        aggregate = [rates[n][1] for n in (4, 6, 8, 12)]
        self.assertEqual(per_link, sorted(per_link, reverse=True))
        self.assertEqual(aggregate, sorted(aggregate))
        self.assertAlmostEqual(rates[8][0], 6.25, places=9)
        self.assertAlmostEqual(rates[12][0], 1.0 / (12 * 0.02), places=9)

    def test_slot_for_target_rate(self) -> None:
        """Both helpers, round-tripped through the simulator they configure.

        Not `assertAlmostEqual(helper(9, 1.4), 79.365)`, which only restates
        1000/(9*1.4) and is what let the module quote RESEARCH.md §0's
        "~1.4 Hz AGGREGATE CSI across 9 nodes" through the PER-LINK helper.
        That reading is 72x too fast — it describes a mesh running at
        100.8 Hz aggregate — and since the published mesh is the only
        real-world anchor the node-count/rate sweep has, it would have made
        every row of the sweep answer the wrong question.
        """
        self.assertAlmostEqual(slot_ms_for_per_link_rate(8, 6.25), 20.0, places=9)
        for n_nodes in (4, 9, 16):
            for rate in (0.5, 2.0, 12.5):
                with self.subTest(n_nodes=n_nodes, rate=rate):
                    sim = MultiLinkSimulator(
                        Geometry.perimeter(n_nodes),
                        SimConfig(
                            seed=5, slot_ms=slot_ms_for_per_link_rate(n_nodes, rate)
                        ),
                    )
                    self.assertAlmostEqual(sim.per_link_rate_hz, rate, places=9)

                    sim = MultiLinkSimulator(
                        Geometry.perimeter(n_nodes),
                        SimConfig(
                            seed=5, slot_ms=slot_ms_for_aggregate_rate(n_nodes, rate)
                        ),
                    )
                    self.assertAlmostEqual(sim.aggregate_packet_rate_hz, rate, places=9)

        # RESEARCH.md §0's figure, read the way RESEARCH.md writes it.
        research = MultiLinkSimulator(
            Geometry.perimeter(9),
            SimConfig(seed=5, slot_ms=slot_ms_for_aggregate_rate(9, 1.4)),
        )
        self.assertAlmostEqual(research.aggregate_packet_rate_hz, 1.4, places=9)
        self.assertAlmostEqual(research.config.slot_ms, 5714.286, places=3)
        self.assertAlmostEqual(research.per_link_rate_hz, 0.019444, places=6)
        # The other reading, for contrast: 72x faster, and not what was
        # measured. The default 8-node/20 ms config is faster still.
        self.assertAlmostEqual(slot_ms_for_per_link_rate(9, 1.4), 79.365, places=3)
        wrong = MultiLinkSimulator(
            Geometry.perimeter(9), SimConfig(seed=5, slot_ms=79.365)
        )
        self.assertAlmostEqual(wrong.aggregate_packet_rate_hz, 100.8, places=1)
        self.assertAlmostEqual(
            wrong.aggregate_packet_rate_hz / research.aggregate_packet_rate_hz, 72.0, places=1
        )

    def test_every_frame_carries_the_configured_subcarriers(self) -> None:
        """K = 56 is a spec requirement and nothing asserted the frame width.

        Every consumer divides by `len(frame.amplitudes)`, so a frame that
        silently carried half the band would look like a perfectly ordinary
        one — `zip(amplitudes, phases)` truncates without complaint, and an
        average over 28 subcarriers is still an average. Asserted against
        the config rather than against 56, plus 56 once for the default, so
        the claim is "the frame is as configured" and not "the constant is
        still the constant".
        """
        self.assertEqual(SimConfig().n_subcarriers, 56)
        for k in (56, 20):
            with self.subTest(n_subcarriers=k):
                cfg = SimConfig(seed=1, n_subcarriers=k, packet_loss=0.0)
                sim = MultiLinkSimulator(Geometry.perimeter(4), cfg)
                n = 0
                for _t, tx, rx, frame in sim.stream(1.0):
                    self.assertEqual(len(frame.amplitudes), k)
                    self.assertEqual(len(frame.phases), k)
                    n += 1
                self.assertGreater(n, 50)
                self.assertEqual(len(sim.noiseless_channel(0, 1, 4.0, 3.0)), k)

    def test_subcarrier_grid_is_an_ht20_layout(self) -> None:
        """A DC null and a uniform spacing either side, read off the phases.

        The grid used to be built as `carrier - (K/2)*df + k*df`, which is
        half a subcarrier low and puts a sample exactly at the channel
        centre, where a real 802.11n HT20 symbol carries a null. Harmless to
        an amplitude-only pipeline, but this module is the index convention
        every other one inherits, and a subcarrier-domain transform or a
        replay of real ESP32 captures would inherit the offset with it.

        Asserted from the emitted physics rather than from a private list:
        with one clean link and nothing else in the model, the phase is
        -2*pi*f_k*d/c, so the step between neighbouring subcarriers is
        constant — except across the DC gap, where the frequency jumps by
        two subcarriers and so does the phase.
        """
        geo = two_node_geometry()
        sim = MultiLinkSimulator(
            geo, SimConfig(seed=1, wall_reflection=0.0, occupied=False)
        )
        phases = [cmath.phase(z) for z in sim.noiseless_channel(0, 1, 0.0, 0.0)]
        steps = [
            (b - a + math.pi) % (2.0 * math.pi) - math.pi
            for a, b in zip(phases, phases[1:])
        ]

        k = SimConfig().n_subcarriers
        gap = k // 2 - 1  # between subcarrier -1 and subcarrier +1
        unit = steps[0]
        for i, step in enumerate(steps):
            want = 2.0 * unit if i == gap else unit
            self.assertAlmostEqual(step, want, places=9, msg=f"step {i}")
        # The null sits at the centre: 28 subcarriers each side of it.
        self.assertEqual(gap, 27)
        self.assertEqual(len(steps) - 1 - gap, 27)


class TestDeterminism(unittest.TestCase):
    def test_identical_seed_gives_identical_stream(self) -> None:
        def run() -> list[tuple]:
            sim = MultiLinkSimulator(Geometry.perimeter(4), SimConfig(seed=77))
            return [
                (t, tx, rx, tuple(f.amplitudes), tuple(f.phases), f.rssi_dbm)
                for t, tx, rx, f in sim.stream(5.0)
            ]

        first, second = run(), run()
        self.assertGreater(len(first), 100)
        self.assertEqual(first, second)

    def test_different_seed_gives_different_stream(self) -> None:
        def run(seed: int) -> list[tuple]:
            sim = MultiLinkSimulator(Geometry.perimeter(4), SimConfig(seed=seed))
            return [(t, tx, rx, tuple(f.amplitudes)) for t, tx, rx, f in sim.stream(5.0)]

        self.assertNotEqual(run(77), run(78))

    def test_reset_rewinds_exactly(self) -> None:
        sim = MultiLinkSimulator(Geometry.perimeter(4), SimConfig(seed=9))
        first = [(t, tx, rx, tuple(f.amplitudes)) for t, tx, rx, f in sim.stream(3.0)]
        truth_first = [(g.x, g.y, g.moving) for g in sim.ground_truth(3.0)]
        sim.reset()
        second = [(t, tx, rx, tuple(f.amplitudes)) for t, tx, rx, f in sim.stream(3.0)]
        truth_second = [(g.x, g.y, g.moving) for g in sim.ground_truth(3.0)]
        self.assertEqual(first, second)
        self.assertEqual(truth_first, truth_second)

    def test_trajectory_is_independent_of_noise_and_loss(self) -> None:
        """Changing one knob must not silently change the experiment.

        The RNG streams are separate so that a sweep over packet loss or
        noise compares the same walk. Without this, every row of a sweep
        table is a different person walking a different path.
        """
        base = [
            (g.t, g.x, g.y, g.moving)
            for g in MultiLinkSimulator(
                Geometry.perimeter(6), SimConfig(seed=31)
            ).ground_truth(60.0, rate_hz=5.0)
        ]
        for cfg in (
            SimConfig(seed=31, packet_loss=0.5),
            SimConfig(seed=31, noise_std=12.0),
            SimConfig(seed=31, cfo=False),
            SimConfig(seed=31, slot_ms=97.0),
        ):
            with self.subTest(cfg=cfg.packet_loss):
                sim = MultiLinkSimulator(Geometry.perimeter(6), cfg)
                other = [
                    (g.t, g.x, g.y, g.moving)
                    for g in sim.ground_truth(60.0, rate_hz=5.0)
                ]
                self.assertEqual(base, other)

    def test_surviving_packets_do_not_depend_on_loss_rate(self) -> None:
        """Noise draws are consumed for dropped packets too, by design."""
        kept = {}
        for loss in (0.0, 0.3):
            sim = MultiLinkSimulator(
                Geometry.perimeter(4), SimConfig(seed=55, packet_loss=loss)
            )
            kept[loss] = {
                (t, tx, rx): tuple(f.amplitudes) for t, tx, rx, f in sim.stream(3.0)
            }
        common = set(kept[0.0]) & set(kept[0.3])
        self.assertGreater(len(common), 100)
        self.assertLess(len(kept[0.3]), len(kept[0.0]))
        for key in common:
            self.assertEqual(kept[0.0][key], kept[0.3][key])


class TestPacketLoss(unittest.TestCase):
    def test_loss_rate_matches_configuration(self) -> None:
        geo = Geometry.perimeter(4)
        duration = 20.0
        # Counted from a lossless run rather than computed as
        # int(duration / superframe) * n_links, which is one float rounding
        # away from being off by a whole superframe.
        scheduled = sum(
            1
            for _ in MultiLinkSimulator(
                geo, SimConfig(seed=13, packet_loss=0.0)
            ).stream(duration)
        )
        for loss in (0.0, 0.05, 0.2):
            with self.subTest(loss=loss):
                sim = MultiLinkSimulator(geo, SimConfig(seed=13, packet_loss=loss))
                emitted = sum(1 for _ in sim.stream(duration))
                observed = 1.0 - emitted / scheduled
                # Binomial 4-sigma band: with ~3000 scheduled packets this is
                # about +/- 0.029 at p = 0.2, so the test is sensitive to a
                # loss rate that is wrong by more than ~15% relative.
                sigma = math.sqrt(max(loss * (1.0 - loss), 1e-12) / scheduled)
                self.assertLess(
                    abs(observed - loss),
                    4.0 * sigma + 1e-12,
                    f"loss {observed:.4f} vs configured {loss}",
                )

    def test_loss_appears_as_gaps_in_sample_index(self) -> None:
        sim = MultiLinkSimulator(
            Geometry.perimeter(4), SimConfig(seed=14, packet_loss=0.3)
        )
        indices: dict[tuple[int, int], list[int]] = {}
        for _t, tx, rx, frame in sim.stream(20.0):
            indices.setdefault((tx, rx), []).append(frame.sample_index)

        gaps = sum(
            1
            for ts in indices.values()
            for a, b in zip(ts, ts[1:])
            if b - a > 1
        )
        self.assertGreater(gaps, 50, "dropped packets must leave index gaps")
        for ts in indices.values():
            self.assertEqual(ts, sorted(ts))
            self.assertEqual(len(ts), len(set(ts)))


class TestShadowing(unittest.TestCase):
    def test_law_endpoints_and_decay_constant(self) -> None:
        sim = MultiLinkSimulator(
            two_node_geometry(),
            SimConfig(seed=1, shadow_s0_db=6.0, shadow_delta0_m=0.2),
        )
        self.assertAlmostEqual(sim.shadowing_db(0.0), 6.0, places=12)
        self.assertAlmostEqual(sim.shadowing_db(0.2), 6.0 / math.e, places=12)
        self.assertAlmostEqual(sim.shadowing_db(0.4), 6.0 / math.e**2, places=12)
        self.assertLess(sim.shadowing_db(5.0), 1e-9)
        # Negative excess is geometrically impossible but reachable by
        # cancellation; it must clamp, not amplify.
        self.assertAlmostEqual(sim.shadowing_db(-1e-15), 6.0, places=12)

    def test_shadowing_increases_as_body_nears_line_of_sight(self) -> None:
        """Explicit geometry: one link along y = 3, body walking onto it."""
        geo = two_node_geometry()
        sim = MultiLinkSimulator(geo, SimConfig(seed=1))
        link = geo.links[geo.link_index(0, 1)]

        offsets = [3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1, 0.0]
        excess = [link.excess_path_m(4.0, 3.0 + d) for d in offsets]
        loss_db = [sim.link_shadowing_db(0, 1, 4.0, 3.0 + d) for d in offsets]

        for a, b in zip(excess, excess[1:]):
            self.assertGreater(a, b, "excess path must shrink toward the LoS")
        for a, b in zip(loss_db, loss_db[1:]):
            self.assertLess(a, b, "shadowing must grow toward the LoS")
        self.assertAlmostEqual(loss_db[-1], 6.0, places=12)
        self.assertLess(loss_db[0], 0.01, "3 m away the link must be clear")
        # Symmetric about the line, and the same in both link directions.
        self.assertAlmostEqual(
            sim.link_shadowing_db(0, 1, 4.0, 3.5),
            sim.link_shadowing_db(0, 1, 4.0, 2.5),
            places=12,
        )
        self.assertAlmostEqual(
            sim.link_shadowing_db(0, 1, 4.0, 3.5),
            sim.link_shadowing_db(1, 0, 4.0, 3.5),
            places=12,
        )

    def test_isolated_line_of_sight_power_falls_monotonically(self) -> None:
        """The mechanism alone: no walls, no body scatter, no noise.

        With those switched off the received amplitude is exactly the
        shadowed direct path, so monotonicity is exact rather than
        statistical. The full model is checked separately — there the
        body-scattered vector adds with a position-dependent phase and the
        monotonicity is genuinely not exact, which is physics and not a
        defect.
        """
        geo = two_node_geometry()
        cfg = SimConfig(
            seed=1,
            packet_loss=0.0,
            noise_std=0.0,
            cfo=False,
            wall_reflection=0.0,
            body_rcs_m2=0.0,
        )
        measured = []
        for offset in (3.0, 2.0, 1.5, 1.0, 0.75, 0.5, 0.25, 0.1, 0.0):
            sim = MultiLinkSimulator(geo, cfg, motion=StaticMotion(4.0, 3.0 + offset))
            _t, _tx, _rx, frame = next(iter(sim.stream(1.0)))
            measured.append(sum(frame.amplitudes) / len(frame.amplitudes))

        for a, b in zip(measured, measured[1:]):
            self.assertGreater(a, b)
        # On the line the direct path is down by S0 = 6 dB, less the residual
        # (2.4e-5 dB) the exponential still has 3 m away — which is why this
        # is not quite a factor of exactly 10^(-6/20).
        sim = MultiLinkSimulator(geo, cfg)
        expected = 10.0 ** (
            -(
                sim.link_shadowing_db(0, 1, 4.0, 3.0)
                - sim.link_shadowing_db(0, 1, 4.0, 6.0)
            )
            / 20.0
        )
        self.assertAlmostEqual(measured[-1] / measured[0], expected, places=9)
        self.assertAlmostEqual(expected, 10.0 ** (-6.0 / 20.0), places=5)

    def test_full_model_shadows_only_the_links_that_pass_through(self) -> None:
        """Eight nodes, person standing still: who loses power, and who does not.

        Compares the same seed with the person present and absent, so the
        noise realisation is identical and the difference is the body. One
        position, through the full noisy stream; `TestDeliveredBlockage`
        sweeps the room noiselessly, which is the assertion with teeth. The
        two thresholds here are measured, not aspirational — see that class
        for where they come from.
        """
        geo = Geometry.perimeter(8)
        px, py = 5.0, 2.0

        def mean_power(occupied: bool) -> dict[tuple[int, int], float]:
            sim = MultiLinkSimulator(
                geo,
                SimConfig(seed=9, packet_loss=0.0, occupied=occupied),
                motion=StaticMotion(px, py),
            )
            total: dict[tuple[int, int], list[float]] = {}
            for _t, tx, rx, frame in sim.stream(8.0):
                p = sum(a * a for a in frame.amplitudes) / len(frame.amplitudes)
                acc = total.setdefault((tx, rx), [0.0, 0.0])
                acc[0] += p
                acc[1] += 1.0
            return {k: v[0] / v[1] for k, v in total.items()}

        occupied, empty = mean_power(True), mean_power(False)
        near, far = [], []
        for key, power in occupied.items():
            link = geo.links[geo.link_index(*key)]
            change_db = 10.0 * math.log10(power / empty[key])
            excess = link.excess_path_m(px, py)
            if excess < 0.15:
                near.append(change_db)
            elif excess > 1.5:
                far.append(change_db)

        self.assertGreaterEqual(len(near), 4)
        self.assertGreaterEqual(len(far), 20)
        self.assertLess(max(near), -1.0, "every blocked link must lose power")
        # On the MAGNITUDE of the disturbance, not on a signed mean. A
        # signed mean over 34 links cancels: the links here individually
        # span -0.54 to +1.37 dB and average +0.05, so a bound on the mean
        # would pass with the far set alternating at +5 and -5 dB. The
        # claim worth making is the weaker true one — distant links ARE
        # disturbed by scatter, they are just disturbed less.
        self.assertLess(
            sum(abs(c) for c in far) / len(far),
            0.5,
            "distant links are disturbed, but only by scatter",
        )
        self.assertLess(
            max(abs(c) for c in far),
            2.0,
            "no distant link may be disturbed like a blocked one",
        )


class TestDeliveredBlockage(unittest.TestCase):
    """What the forward model actually delivers when a body blocks a link.

    The configured S0 is an input; the delivered attenuation is what every
    downstream module is scored on, and the two were not the same thing.
    The body-scattered term used to be a positive real coefficient, so on a
    line of sight — where d1 + d2 == d exactly, at every subcarrier — it was
    locked in phase with the direct path and filled in the shadow it was
    supposed to accompany: a configured 6 dB arrived as 3.96 dB on a 6 m
    link and as +2.27 dB, i.e. a BRIGHTER link, on a 1 m one. Nothing failed,
    because this is the ground-truth generator: it surfaced downstream as a
    localizer scored against an observable whose depth fell with link length
    and whose sign flipped near nodes.

    Hence this class. Everything here is noiseless and position-swept,
    because the defect was a function of geometry and a single probe
    position cannot see it.
    """

    def mean_power(self, sim: MultiLinkSimulator, tx: int, rx: int, x: float, y: float) -> float:
        return sum(abs(z) ** 2 for z in sim.noiseless_channel(tx, rx, x, y))

    def test_on_the_line_of_sight_the_delivered_depth_is_the_configured_one(self) -> None:
        """With the walls off, blockage on the LoS must be exactly S0.

        The forward-scatter taper is what makes this exact: on the line of
        sight the scattered path is zero, so the only body term left is the
        shadowing law. If the taper or the pi flip goes, this fails at every
        link length, and it fails worst on the short links — which is what
        confounds a node-count sweep, because adding nodes shortens the
        adjacent-node links.
        """
        for s0 in (3.0, 6.0, 10.0):
            for length in (7.0, 6.0, 4.0, 3.0, 2.0, 1.5, 1.0, 0.6):
                with self.subTest(s0=s0, length=length):
                    room = Room(8.0, 6.0)
                    geo = Geometry(
                        room,
                        [Node(0, 0.5, 3.0), Node(1, 0.5 + length, 3.0)],
                        PixelGrid(room, 0.5),
                        lambda_m=0.3,
                    )
                    sim = MultiLinkSimulator(
                        geo, SimConfig(seed=1, wall_reflection=0.0, shadow_s0_db=s0)
                    )
                    clear = self.mean_power(sim, 0, 1, -1e3, -1e3)
                    blocked = self.mean_power(sim, 0, 1, 0.5 + length / 2.0, 3.0)
                    delivered = -10.0 * math.log10(blocked / clear)
                    # Exact to 7e-8 dB, and the residual is floating point,
                    # not modelling: two unit vectors that should cancel
                    # exactly leave |a + b| ~ 1e-8, so the tapered scatter
                    # is ~4e-8 against a direct path of ~30.
                    self.assertAlmostEqual(delivered, s0, places=6)

    def test_no_blocked_link_anywhere_in_the_room_gains_power(self) -> None:
        """Sweep the body over the walkable region, not over one point.

        A blocked link losing power is the single property RTI rests on,
        and it is a property of (link, position) pairs, so it has to be
        swept. 'Blocked' is excess < 0.15 m, which on a 3.4 m link is a
        body within ~0.3 m of the line of sight.

        Measured on the finest grid (0.05 m, 146,936 pairs): the worst
        blocked link loses 1.105 dB and the median loses 2.36 dB. The old
        model's worst case in this band was +1.638 dB — it made a blocked
        link brighter — so the bound below is the assertion that was
        missing, not a restatement of one that passed.
        """
        geo = Geometry.perimeter(8)
        sim = MultiLinkSimulator(geo, SimConfig(seed=1))
        clear = {
            (l.tx_id, l.rx_id): self.mean_power(sim, l.tx_id, l.rx_id, -1e3, -1e3)
            for l in geo.links
        }

        n_pairs = 0
        worst = -99.0
        worst_at = None
        deepest = 0.0
        for i in range(10, 151, 5):  # 0.5 .. 7.5 m, the walkable region
            for j in range(10, 111, 5):  # 0.5 .. 5.5 m
                x, y = 0.05 * i, 0.05 * j
                for link in geo.links:
                    if link.excess_path_m(x, y) >= 0.15:
                        continue
                    n_pairs += 1
                    key = (link.tx_id, link.rx_id)
                    change = 10.0 * math.log10(
                        self.mean_power(sim, *key, x, y) / clear[key]
                    )
                    deepest = min(deepest, change)
                    if change > worst:
                        worst, worst_at = change, (key, round(x, 2), round(y, 2))

        self.assertGreater(n_pairs, 3000)
        self.assertLess(
            worst, -1.0, f"a blocked link gained or barely lost power at {worst_at}"
        )
        self.assertLess(deepest, -5.0, "some link must be deeply blocked")

    def test_attenuation_is_monotone_across_a_perpendicular_cut(self) -> None:
        """Walk across a link and the loss must grow all the way in.

        With the DEFAULT config — walls on, body scatter on — not with the
        body term switched off, which is how the old suite checked
        monotonicity and why the old model's local minimum ON the line of
        sight survived. There the deepest attenuation sat on an off-axis
        ring: 2.054 dB on the line against 3.284 dB at 0.4 m off it.

        Three claims, because the honest statement is not simply "monotone":

          1. the deepest point of the cut is ON the line of sight;
          2. sampled every 0.25 m the loss falls strictly outward;
          3. sampled every 0.125 m it does NOT, and the wobble is bounded
             by 0.25 dB (measured worst: 0.19 dB).

        Claim 3 is the scattered path adding as a vector: its phase turns
        through 2*pi per 0.123 m of excess, so a fine cut ripples. Asserting
        exact monotonicity would be asserting a fiction; asserting the size
        of the ripple pins it. Claims 1 and 2 are what the old model broke.

        Cross-room (diameter) links only. A link hugging a wall has a wall
        image almost exactly as long and as strong as its direct path — for
        `Geometry.perimeter(8)`'s link 5->6, 3.406 m against 3.400 m — and
        shadowing attenuates the direct path alone, so the delivered profile
        there is dominated by an unshadowed image and ripples by up to
        2.02 dB. That is the documented wall limitation showing through, not
        the body term, and a test that mixed those links in would be testing
        it rather than this.
        """
        for n_nodes in (6, 8, 12):
            geo = Geometry.perimeter(n_nodes)
            sim = MultiLinkSimulator(geo, SimConfig(seed=1))
            for tx in range(n_nodes // 2):
                rx = tx + n_nodes // 2
                with self.subTest(n_nodes=n_nodes, link=(tx, rx)):
                    link = geo.links[geo.link_index(tx, rx)]
                    (ax, ay), (bx, by) = link.tx, link.rx
                    mx, my = link.midpoint
                    ux = -(by - ay) / link.length_m
                    uy = (bx - ax) / link.length_m
                    clear = self.mean_power(sim, tx, rx, -1e3, -1e3)

                    fine = []
                    for step in range(11):
                        r = 0.125 * step
                        p = self.mean_power(sim, tx, rx, mx + r * ux, my + r * uy)
                        fine.append(-10.0 * math.log10(p / clear))

                    self.assertGreater(fine[0], 1.5, "on the line it must be blocked")
                    self.assertEqual(
                        max(fine),
                        fine[0],
                        f"deepest attenuation must be ON the line: {fine}",
                    )
                    coarse = fine[::2]
                    for a, b in zip(coarse, coarse[1:]):
                        self.assertGreater(a, b, f"loss must fall outward: {coarse}")
                    for a, b in zip(fine, fine[1:]):
                        self.assertLess(b - a, 0.25, f"ripple too large: {fine}")

    def test_blocked_links_stand_clear_of_distant_ones(self) -> None:
        """Spatial selectivity, swept over positions and stated per link.

        The old version of this check probed one hand-picked interior point
        and compared SIGNED means over link sets, both of which hide the
        thing being claimed: at (3.75, 0.50) — inside the default walkable
        region — a 'near' link gained 4.56 dB and the test could not see it.

        What is asserted here is what the model actually delivers, measured:
        every blocked link loses at least 1 dB at every position, while the
        mean |disturbance| on distant links stays near 0.2 dB. Note what is
        NOT asserted: that every distant link is disturbed less than every
        blocked one. That is false and it is honest physics — a body 0.9 m
        from a node disturbs the links through that node by up to 2.96 dB
        without blocking them. It happens at 14 of 607 positions, and it is
        the ceiling on the contrast any RTI inversion can get from this data.
        """
        geo = Geometry.perimeter(8)
        sim = MultiLinkSimulator(geo, SimConfig(seed=1))
        clear = {
            (l.tx_id, l.rx_id): self.mean_power(sim, l.tx_id, l.rx_id, -1e3, -1e3)
            for l in geo.links
        }

        checked = 0
        for i in range(10, 151, 10):  # 0.5 .. 7.5 m on a 0.5 m grid
            for j in range(10, 111, 10):
                x, y = 0.05 * i, 0.05 * j
                near, far = [], []
                for link in geo.links:
                    excess = link.excess_path_m(x, y)
                    if not (excess < 0.15 or excess > 1.5):
                        continue
                    key = (link.tx_id, link.rx_id)
                    change = 10.0 * math.log10(
                        self.mean_power(sim, *key, x, y) / clear[key]
                    )
                    (near if excess < 0.15 else far).append(change)
                if len(near) < 2:
                    continue
                checked += 1
                with self.subTest(x=round(x, 2), y=round(y, 2)):
                    self.assertLess(max(near), -1.0, "a blocked link gained power")
                    self.assertLess(
                        sum(abs(c) for c in far) / len(far),
                        0.6,
                        "distant links must stay much quieter than blocked ones",
                    )
                    self.assertGreater(
                        sum(abs(c) for c in near) / len(near),
                        2.0 * sum(abs(c) for c in far) / len(far),
                        "blocked links must stand clear of unblocked ones",
                    )
        self.assertGreater(checked, 100)

    def test_public_shadowing_matches_the_loss_the_channel_applies(self) -> None:
        """`link_shadowing_db` is a predictor of the forward model, or it is a lie.

        It used to disagree inside 5 cm of a node, because `_channel`
        clamped d1 and d2 to 0.05 m and then computed the excess from the
        CLAMPED distances — so the applied loss went DOWN as the body closed
        the last few centimetres onto a node, while the public method said
        6.000 dB. Body ON a node is the case to test, because that is where
        the clamp used to fire.
        """
        geo = two_node_geometry()
        cfg = SimConfig(
            seed=1, wall_reflection=0.0, body_rcs_m2=0.0, noise_std=0.0, cfo=False
        )
        sim = MultiLinkSimulator(geo, cfg)
        clear = sim.noiseless_channel(0, 1, -1e3, -1e3)
        base = sum(abs(z) ** 2 for z in clear)

        for r in (0.0, 0.01, 0.03, 0.05, 0.1, 0.5, 1.0, 3.0):
            with self.subTest(r=r):
                x = 1.0 + r
                applied = -10.0 * math.log10(
                    sum(abs(z) ** 2 for z in sim.noiseless_channel(0, 1, x, 3.0)) / base
                )
                self.assertAlmostEqual(applied, sim.link_shadowing_db(0, 1, x, 3.0), places=9)

        # And the law stays monotone onto the node rather than turning over.
        on_axis = [sim.link_shadowing_db(0, 1, 1.0 + r, 3.0) for r in (0.5, 0.2, 0.05, 0.0)]
        for a, b in zip(on_axis, on_axis[1:]):
            self.assertLessEqual(a, b)
        self.assertAlmostEqual(on_axis[-1], 6.0, places=12)


class TestBodyScattering(unittest.TestCase):
    def test_breathing_is_detectable_on_a_nearby_link(self) -> None:
        """A standing person still modulates the links beside them.

        Measured with the project's own `band_peak` on the subcarrier-mean
        amplitude, after removing the mean — the baseline is ~40 units and
        the breathing modulation is ~1, so without that removal the window's
        DC leakage buries the band and the estimator returns the lowest bin
        with high confidence. Downstream, `StaticSuppressor` is what does
        this; here it is one subtraction, kept explicit.
        """
        geo = Geometry.perimeter(6)
        px, py = 3.0, 4.5

        def best_near_link(amplitude_m: float) -> tuple[float, float]:
            sim = MultiLinkSimulator(
                geo,
                SimConfig(seed=11, packet_loss=0.0, breathing_amplitude_m=amplitude_m),
                motion=StaticMotion(px, py),
            )
            best = (0.0, 0.0)
            for (tx, rx), xs in mean_amplitude_series(sim, 40.0).items():
                link = geo.links[geo.link_index(tx, rx)]
                if perpendicular_distance(link, px, py) > 1.6:
                    continue
                mean = sum(xs) / len(xs)
                peak = band_peak(
                    [v - mean for v in xs],
                    sim.per_link_rate_hz,
                    0.10,
                    0.50,
                    resolution_hz=0.002,
                )
                if peak is None or abs(peak.freq_hz - 0.25) > 0.02:
                    continue
                if peak.confidence > best[0]:
                    best = (peak.confidence, peak.freq_hz)
            return best

        confidence, freq = best_near_link(0.005)
        self.assertGreaterEqual(
            confidence, 0.9, "5 mm of chest motion must be visible on some near link"
        )
        self.assertAlmostEqual(freq, 15.0 / 60.0, places=2)

        # Control: with the chest held still, nothing in the band survives.
        self.assertLess(best_near_link(0.0)[0], 0.5)

    def test_body_on_the_line_of_sight_shows_no_breathing(self) -> None:
        """The forward-scatter null, asserted because it looks like a bug.

        Standing exactly between the two nodes, chest displacement along the
        bisector of the tx-body-rx angle changes d1 and d2 by equal and
        opposite amounts, so the scattered path length d1 + d2 is stationary
        and there is no first-order breathing signature at all. This is the
        link with the *strongest* shadowing and the *weakest* vital signal,
        and the two therefore want different links.
        """
        geo = two_node_geometry()
        sim = MultiLinkSimulator(geo, SimConfig(seed=1))

        on_line = [
            sim.noiseless_channel(0, 1, 4.0, 3.0, d) for d in (-0.005, 0.0, 0.005)
        ]
        for h in on_line[1:]:
            for a, b in zip(h, on_line[0]):
                self.assertAlmostEqual(abs(a), abs(b), places=9)

        # A metre to the side, the same displacement is plainly visible.
        off_line = [
            sim.noiseless_channel(0, 1, 4.0, 4.0, d) for d in (-0.005, 0.005)
        ]
        swing = max(abs(abs(a) - abs(b)) for a, b in zip(*off_line))
        self.assertGreater(swing, 0.05)

    def test_empty_room_ignores_the_person(self) -> None:
        """`occupied=False` is the calibration stream `measure` needs."""
        geo = Geometry.perimeter(4)
        cfg_a = SimConfig(seed=21, occupied=False, packet_loss=0.0)
        cfg_b = SimConfig(seed=21, occupied=False, packet_loss=0.0)
        a = MultiLinkSimulator(geo, cfg_a, motion=StaticMotion(4.0, 3.0))
        b = MultiLinkSimulator(geo, cfg_b, motion=StaticMotion(0.7, 0.7))
        stream_a = [(t, tx, rx, tuple(f.amplitudes)) for t, tx, rx, f in a.stream(2.0)]
        stream_b = [(t, tx, rx, tuple(f.amplitudes)) for t, tx, rx, f in b.stream(2.0)]
        self.assertEqual(stream_a, stream_b)

        occupied = MultiLinkSimulator(
            geo, SimConfig(seed=21, packet_loss=0.0), motion=StaticMotion(4.0, 3.0)
        )
        stream_c = [
            (t, tx, rx, tuple(f.amplitudes)) for t, tx, rx, f in occupied.stream(2.0)
        ]
        self.assertNotEqual(stream_a, stream_c)

    def test_the_scattered_path_vanishes_on_the_line_of_sight(self) -> None:
        """The forward-scatter taper, asserted through the public channel.

        On the line of sight the shadowing law already represents the body —
        a shadow is the forward-scattered field cancelling the incident one
        — so the explicit scattered path has to be zero there or the torso
        is counted twice. It used to be counted twice, in phase, which is
        why a blocked link could come out brighter than an empty one.
        """
        geo = two_node_geometry()

        def channel(rcs: float, x: float, y: float) -> list[complex]:
            sim = MultiLinkSimulator(geo, SimConfig(seed=1, body_rcs_m2=rcs))
            return sim.noiseless_channel(0, 1, x, y, 0.004)

        for x in (1.5, 2.0, 4.0, 6.5):
            with self.subTest(x=x):
                scattering = channel(0.5, x, 3.0)
                none = channel(0.0, x, 3.0)
                self.assertLess(
                    max(abs(a - b) for a, b in zip(scattering, none)),
                    1e-7,
                    "the scattered path must be zero on the line of sight",
                )

        # A metre off it the term is plainly present: 3.8% of the direct
        # path at 1 m off the midpoint of this 6 m link.
        off = channel(0.5, 4.0, 4.0)
        bare = channel(0.0, 4.0, 4.0)
        self.assertGreater(max(abs(a - b) for a, b in zip(off, bare)), 1.0)

    def test_rendered_breathing_matches_the_configured_chest_motion(self) -> None:
        """The chest displacement `stream` computes, against the forward model.

        `TestForwardModel` checks the channel given a displacement; this
        checks the displacement, which is the one step of the vitals chain
        nothing exercised. The gap mattered: scaling the chest motion down
        20x left the whole suite passing, because the only breathing
        assertion was on a spectral CONFIDENCE, which saturates near 1.0 for
        any clean sinusoid above the noise floor and so constrains the
        5 mm -> CSI transfer only to within a factor of about 20.

        Here the expected amplitude is recomputed from the config —
        amplitude, rate and phase — and compared sample by sample, so a
        scale error, a factor of two, a sign slip or a wrong frequency all
        fail immediately.
        """
        geo = Geometry.perimeter(6)
        px, py = 3.0, 4.5
        cfg = SimConfig(
            seed=11,
            packet_loss=0.0,
            noise_std=0.0,
            cfo=False,
            breathing_bpm=18.0,
            breathing_phase_rad=0.7,
        )
        sim = MultiLinkSimulator(geo, cfg, motion=StaticMotion(px, py))

        checked = 0
        for t, tx, rx, frame in sim.stream(6.0):
            disp = cfg.breathing_amplitude_m * math.sin(
                2.0 * math.pi * cfg.breathing_hz * t + cfg.breathing_phase_rad
            )
            want = sim.noiseless_channel(tx, rx, px, py, disp)
            for got, w in zip(frame.amplitudes, want):
                self.assertAlmostEqual(got, abs(w), places=9)
            checked += 1
        self.assertGreater(checked, 500)

    def test_breathing_modulation_depth_matches_a_two_ray_prediction(self) -> None:
        """The absolute depth, against a closed form derived outside the module.

        With the walls off a link is two rays — the shadowed direct path and
        the tapered, pi-flipped scattered one — so

            |H_k(t)| = sqrt(A0^2 + B^2 + 2*A0*B*cos(psi_k(t)))

        with A0 the shadowed direct amplitude, B the (negative) scattered
        one and psi_k = 2*pi*f_k*(d1 + d2 - d)/c. Everything in it comes
        from the config and the public `link_shadowing_db`, not from
        `_channel`, so this pins the 5 mm -> amplitude transfer the vitals
        SNR arguments rest on to an absolute number rather than to a
        spectral confidence.

        Chest direction is set explicitly: at (4, 4) the bisector of this
        link points along -y, so a chest facing +y is the aligned case.
        """
        room = Room(8.0, 6.0)
        geo = Geometry(
            room,
            [Node(0, 1.0, 3.0), Node(1, 7.0, 3.0)],
            PixelGrid(room, 0.5),
            lambda_m=0.3,
        )
        px, py = 4.0, 4.0
        cfg = SimConfig(
            seed=1,
            packet_loss=0.0,
            noise_std=0.0,
            cfo=False,
            wall_reflection=0.0,
            chest_direction_rad=math.pi / 2.0,
        )
        sim = MultiLinkSimulator(geo, cfg, motion=StaticMotion(px, py))
        series = [sum(f.amplitudes) / len(f.amplitudes) for _t, _tx, _rx, f in sim.stream(24.0)]
        measured = max(series) - min(series)

        c = 299_792_458.0
        link = geo.links[geo.link_index(0, 1)]
        (ax, ay), (bx, by) = link.tx, link.rx
        d = link.length_m
        d1 = math.hypot(ax - px, ay - py)
        d2 = math.hypot(px - bx, py - by)
        sx = (ax - px) / d1 + (bx - px) / d2
        sy = (ay - py) / d1 + (by - py) / d2
        projection = sx * math.cos(cfg.chest_direction_rad) + sy * math.sin(
            cfg.chest_direction_rad
        )
        scale = cfg.reference_magnitude * cfg.reference_distance_m
        sigma = math.sqrt(cfg.body_rcs_m2 / (4.0 * math.pi))
        a0 = 10.0 ** (-sim.link_shadowing_db(0, 1, px, py) / 20.0) * scale / d
        b = (
            -0.5
            * math.hypot(sx, sy)
            * scale
            * sigma
            / (
                math.hypot(d1, cfg.body_radius_m) * math.hypot(d2, cfg.body_radius_m)
            )
        )
        n_low = cfg.n_subcarriers // 2
        freqs = [
            cfg.carrier_hz + k * cfg.subcarrier_spacing_hz
            for k in list(range(-n_low, 0)) + list(range(1, cfg.n_subcarriers - n_low + 1))
        ]

        predicted_series = []
        for step in range(400):
            t = step * 4.0 / 400.0
            disp = cfg.breathing_amplitude_m * math.sin(
                2.0 * math.pi * cfg.breathing_hz * t
            )
            path = d1 + d2 - projection * disp
            predicted_series.append(
                sum(
                    math.sqrt(
                        a0 * a0
                        + b * b
                        + 2.0 * a0 * b * math.cos(2.0 * math.pi * f * (path - d) / c)
                    )
                    for f in freqs
                )
                / len(freqs)
            )
        predicted = max(predicted_series) - min(predicted_series)

        # Both sides must be a real signal, not two matching zeros.
        self.assertGreater(predicted, 0.25)
        self.assertAlmostEqual(measured / predicted, 1.0, delta=0.02)
        # 5 mm of chest wall on a 30-unit baseline: ~1% modulation depth.
        self.assertAlmostEqual(measured, 0.2994, delta=0.01)

    def test_the_chest_has_one_direction_for_the_whole_run(self) -> None:
        """One torso moves one way, so some links see no breathing at all.

        Giving every link the displacement along its OWN bisector hands each
        one its maximum possible sensitivity, which is the most generous
        assumption available and overstates per-link breathing by up to 20x
        (`Geometry.perimeter(6)`, person at (3.0, 4.5), link 1->4: 0.602 m/m
        along the bisector against 0.030 m/m for a chest facing +y). The
        forward-scatter null survives a fixed direction, because a + b is
        zero on the line of sight whatever the chest does, but a second null
        family appears — and it is real, so it is modelled.
        """
        room = Room(8.0, 6.0)
        geo = Geometry(
            room,
            [Node(0, 1.0, 3.0), Node(1, 7.0, 3.0)],
            PixelGrid(room, 0.5),
            lambda_m=0.3,
        )

        def depth(chest_rad: float) -> float:
            sim = MultiLinkSimulator(
                geo,
                SimConfig(
                    seed=1,
                    packet_loss=0.0,
                    noise_std=0.0,
                    cfo=False,
                    chest_direction_rad=chest_rad,
                ),
                motion=StaticMotion(4.0, 4.0),
            )
            xs = [sum(f.amplitudes) / len(f.amplitudes) for _t, _tx, _rx, f in sim.stream(24.0)]
            return max(xs) - min(xs)

        # At (4, 4) this link's bisector points along -y.
        self.assertGreater(depth(math.pi / 2.0), 0.25, "aligned chest must show")
        self.assertLess(depth(0.0), 1e-9, "a perpendicular chest must not")

        # Drawn from the seed alone, so sweeping anything else keeps it.
        drawn = MultiLinkSimulator(geo, SimConfig(seed=404)).chest_direction_rad
        for cfg in (
            SimConfig(seed=404, noise_std=3.0),
            SimConfig(seed=404, packet_loss=0.4),
            SimConfig(seed=404, slot_ms=71.0),
        ):
            self.assertEqual(
                MultiLinkSimulator(geo, cfg).chest_direction_rad, drawn
            )
        self.assertNotEqual(
            MultiLinkSimulator(geo, SimConfig(seed=405)).chest_direction_rad, drawn
        )

    def test_breathing_is_rendered_while_the_person_walks(self) -> None:
        """A deliberate deviation from P0B_TASK.md 3.1, pinned on purpose.

        The spec asks for chest modulation "while the person stands still".
        This module applies it continuously, because switching it at every
        pause boundary puts a discontinuity into the signal that a
        downstream detector reads as an event. That is a defensible choice
        but it is a choice, and it was recorded only in a docstring — so a
        later "fix" that gated it on `moving` would have passed the suite.
        It should have to be a decision instead.
        """
        geo = Geometry.perimeter(6)
        motion = WaypointMotion([(2.0, 2.0), (6.0, 4.0)], speed_mps=0.6)
        base = dict(seed=5, packet_loss=0.0, noise_std=0.0, cfo=False)

        def run(amplitude: float) -> list[tuple[float, int, int, tuple[float, ...]]]:
            sim = MultiLinkSimulator(
                geo,
                SimConfig(breathing_amplitude_m=amplitude, **base),
                motion=WaypointMotion([(2.0, 2.0), (6.0, 4.0)], speed_mps=0.6),
            )
            return [
                (t, tx, rx, tuple(f.amplitudes))
                for t, tx, rx, f in sim.stream(4.0)
                if sim.truth_at(t).moving
            ]

        still, breathing = run(0.0), run(0.005)
        self.assertGreater(len(still), 500, "the person must be walking here")
        self.assertTrue(all(motion.at(t).moving for t, _tx, _rx, _a in still))
        self.assertNotEqual(still, breathing)


class TestReciprocity(unittest.TestCase):
    def test_the_two_directions_of_a_link_carry_the_same_signal(self) -> None:
        """Pins a LIMITATION, so that removing it has to be deliberate.

        geometry.py keeps all N(N-1) ordered links on the grounds that tx
        and rx differ in noise floor, antenna orientation and slot, i.e.
        that they are two independent observations. This module does not
        model any of that: the image sum is reciprocal by construction and
        both body terms are symmetric in d1/d2, so before noise and CFO the
        two directions are bit-identical. The measurement vector of length
        N(N-1) therefore carries N(N-1)/2 distinct signals plus independent
        noise, and any evaluation reporting "56 links at 8 nodes" is
        overstating the independent information by 2x.

        Asserted rather than left implicit because it is exactly the kind of
        fact a reader assumes away, and because the dual-form L x L solve
        downstream has duplicated rows because of it.
        """
        geo = Geometry.perimeter(8)
        sim = MultiLinkSimulator(geo, SimConfig(seed=1, noise_std=0.0, cfo=False))
        worst = 0.0
        for i in range(8):
            for j in range(i + 1, 8):
                forward = sim.noiseless_channel(i, j, 5.0, 2.0, 0.003)
                backward = sim.noiseless_channel(j, i, 5.0, 2.0, 0.003)
                worst = max(worst, max(abs(a - b) for a, b in zip(forward, backward)))
        self.assertLess(worst, 1e-9, "links are reciprocal in this model")

        # Only the receiver chain tells them apart.
        noisy = MultiLinkSimulator(geo, SimConfig(seed=1, packet_loss=0.0))
        frames: dict[tuple[int, int], tuple[float, ...]] = {}
        for _t, tx, rx, frame in noisy.stream(noisy.superframe_s * 2.0):
            frames.setdefault((tx, rx), tuple(frame.amplitudes))
        self.assertNotEqual(frames[(0, 3)], frames[(3, 0)])


class TestStreamFollowsTheWalk(unittest.TestCase):
    """The CSI and the ground truth must describe the same person.

    Every other physics test here drives the simulator with `StaticMotion`
    or calls `noiseless_channel` directly, and the motion tests look at
    trajectories without ever looking at a frame. That left
    `truth = self.motion.at(t)` inside `stream` completely unconstrained:
    rendering the person where they will be two seconds later, or freezing
    them at their t = 0 position for the whole run, both passed the entire
    suite. For the module P0B_TASK.md calls the ground-truth generator
    everything else is scored against, a 2 s skew is ~1.5 m of systematic
    error at 0.75 m/s — every tracker would be scored against the wrong
    truth and the sweep would conclude that RTI does not work.
    """

    def test_the_fade_happens_when_the_person_crosses_the_line_of_sight(self) -> None:
        geo = two_node_geometry()  # (1, 3) -> (7, 3)
        # Crosses y = 3 at t = 4.0 s, walking 1 -> 5 m in y at 0.5 m/s.
        motion = WaypointMotion([(4.0, 1.0), (4.0, 5.0)], speed_mps=0.5)
        cfg = SimConfig(
            seed=1,
            packet_loss=0.0,
            noise_std=0.0,
            cfo=False,
            wall_reflection=0.0,
            body_rcs_m2=0.0,
        )
        sim = MultiLinkSimulator(geo, cfg, motion=motion)

        series = [
            (t, sum(f.amplitudes) / len(f.amplitudes))
            for t, tx, rx, f in sim.stream(8.0)
            if (tx, rx) == (0, 1)
        ]
        self.assertGreater(len(series), 100)
        t_min = min(series, key=lambda p: p[1])[0]
        self.assertAlmostEqual(
            t_min,
            4.0,
            delta=sim.superframe_s,
            msg=f"deepest fade at t={t_min}, person crosses the LoS at t=4.0",
        )
        amps = [a for _t, a in series]
        self.assertGreater((max(amps) - min(amps)) / max(amps), 0.1)

    def test_rendered_position_agrees_with_truth_at(self) -> None:
        """Frame by frame, the loss on the link is the loss at truth_at(t)."""
        geo = two_node_geometry()
        cfg = SimConfig(
            seed=2,
            packet_loss=0.0,
            noise_std=0.0,
            cfo=False,
            wall_reflection=0.0,
            body_rcs_m2=0.0,
        )
        sim = MultiLinkSimulator(geo, cfg)  # the default random walk
        clear = sim.noiseless_channel(0, 1, -1e3, -1e3)
        reference = abs(clear[0])

        checked = 0
        for t, tx, rx, frame in sim.stream(30.0):
            if (tx, rx) != (0, 1):
                continue
            truth = sim.truth_at(t)
            measured = -20.0 * math.log10(
                (sum(frame.amplitudes) / len(frame.amplitudes)) / reference
            )
            self.assertAlmostEqual(
                measured, sim.link_shadowing_db(0, 1, truth.x, truth.y), places=6
            )
            checked += 1
        self.assertGreater(checked, 500)


class TestForwardModel(unittest.TestCase):
    def test_channel_matches_an_independent_implementation(self) -> None:
        """The optimised channel against a deliberately naive rewrite.

        `_channel` makes substitutions for speed that a reader has to take
        on trust: the per-subcarrier phasor is accumulated as a geometric
        sequence with one double step across the DC null rather than
        recomputed, and the scattering geometry is read off the sum of two
        unit vectors instead of an angle. Those are exact identities, which
        is precisely why a silent error in either would be invisible — the
        output would still look like a plausible channel. So this reference
        is written out the slow, obvious way, from the spec rather than from
        the module: explicit `acos`, an explicit angle for the chest
        direction, one `exp` per subcarrier, and the HT20 frequency grid
        rebuilt from the config rather than read off the simulator.
        """
        geo = Geometry.perimeter(6)
        cfg = SimConfig(seed=1)
        sim = MultiLinkSimulator(geo, cfg)

        c = 299_792_458.0
        k_count = cfg.n_subcarriers
        # HT20: K subcarriers symmetric about the carrier, none at DC.
        indices = [k for k in range(-(k_count // 2), k_count // 2 + 1) if k != 0]
        self.assertEqual(len(indices), k_count)
        freqs = [cfg.carrier_hz + k * cfg.subcarrier_spacing_hz for k in indices]
        scale = cfg.reference_magnitude * cfg.reference_distance_m
        sigma = math.sqrt(cfg.body_rcs_m2 / (4.0 * math.pi))
        width, height = geo.room.width_m, geo.room.height_m
        chest = sim.chest_direction_rad

        def reference(link, px: float, py: float, disp: float) -> list[complex]:
            d = link.length_m
            (ax, ay), (bx, by) = link.tx, link.rx
            d1 = math.hypot(ax - px, ay - py)
            d2 = math.hypot(px - bx, py - by)
            loss = cfg.shadow_s0_db * math.exp(
                -max(d1 + d2 - d, 0.0) / cfg.shadow_delta0_m
            )
            gain = 10.0 ** (-loss / 20.0)

            # Scattering angle the long way round, from the dot product and
            # an arccos, and the chest projection from the two bearings.
            cos_beta = ((ax - px) * (bx - px) + (ay - py) * (by - py)) / (d1 * d2)
            cos_beta = max(-1.0, min(1.0, cos_beta))
            beta = math.acos(cos_beta)
            taper = abs(math.cos(beta / 2.0))
            to_tx = math.atan2(ay - py, ax - px)
            to_rx = math.atan2(by - py, bx - px)
            projection = math.cos(to_tx - chest) + math.cos(to_rx - chest)
            body_len = d1 + d2 - projection * disp
            body = (
                -taper
                * scale
                * sigma
                / (
                    math.sqrt(d1 * d1 + cfg.body_radius_m**2)
                    * math.sqrt(d2 * d2 + cfg.body_radius_m**2)
                )
            )

            out = []
            for f in freqs:
                h = gain * (scale / d) * cmath.exp(-2j * math.pi * f * d / c)
                for ix, iy in (
                    (-ax, ay),
                    (2.0 * width - ax, ay),
                    (ax, -ay),
                    (ax, 2.0 * height - ay),
                ):
                    length = math.hypot(ix - bx, iy - by)
                    h += (
                        -cfg.wall_reflection
                        * scale
                        / length
                        * cmath.exp(-2j * math.pi * f * length / c)
                    )
                h += body * cmath.exp(-2j * math.pi * f * body_len / c)
                out.append(h)
            return out

        worst = 0.0
        for link in geo.links[:12]:
            for px, py, disp in (
                (4.0, 3.0, 0.0),
                (2.3, 1.7, 0.005),
                (6.5, 4.9, -0.005),
                (0.3, 5.7, 0.002),
            ):
                got = sim.noiseless_channel(link.tx_id, link.rx_id, px, py, disp)
                want = reference(link, px, py, disp)
                self.assertEqual(len(got), k_count)
                worst = max(worst, max(abs(a - b) for a, b in zip(got, want)))

        # Channel magnitudes are of order 30, so 1e-7 is ~3e-9 relative:
        # floating point, not modelling. The measured residual is 4.7e-8 and
        # all of it comes from ONE position, (4.0, 3.0), which lies exactly
        # on link 1->4's line of sight — and from the reference, not the
        # module. There cos(beta) = -1 + 2.2e-16, so the reference's
        # `acos` returns pi - 2.1e-8 and its `cos(beta/2)` is 1.05e-8: the
        # square root of a rounding error. The module's |a + b| / 2 gives
        # 1.7e-16. That is the whole argument for the vector form, and it
        # shows up here as the reference being the less accurate side.
        self.assertLess(worst, 1e-7)


class TestReceiverChain(unittest.TestCase):
    def test_noise_is_circular_complex_gaussian(self) -> None:
        """Checks the hand-rolled Box-Muller against the channel it rides on.

        The polar form replaced two `random.gauss` calls for speed, so the
        distribution it produces is this module's responsibility rather than
        the standard library's.
        """
        geo = two_node_geometry()
        sigma = 0.6
        sim = MultiLinkSimulator(
            geo,
            SimConfig(
                seed=17,
                packet_loss=0.0,
                occupied=False,
                cfo=False,
                noise_std=sigma,
            ),
        )
        clean = sim.noiseless_channel(0, 1, 0.0, 0.0)
        reals: list[float] = []
        imags: list[float] = []
        for _t, _tx, _rx, frame in sim.stream(40.0):
            for k, (amp, phase) in enumerate(zip(frame.amplitudes, frame.phases)):
                n = cmath.rect(amp, phase) - clean[k]
                reals.append(n.real)
                imags.append(n.imag)

        count = len(reals)
        self.assertGreater(count, 20_000)
        for name, xs in (("real", reals), ("imag", imags)):
            mean = sum(xs) / count
            var = sum((x - mean) ** 2 for x in xs) / (count - 1)
            # 4 standard errors on each statistic.
            self.assertLess(abs(mean), 4.0 * sigma / math.sqrt(count), name)
            self.assertLess(
                abs(math.sqrt(var) - sigma),
                4.0 * sigma / math.sqrt(2.0 * count),
                name,
            )
        cross = sum(r * i for r, i in zip(reals, imags)) / count
        self.assertLess(abs(cross), 4.0 * sigma * sigma / math.sqrt(count))

        # Total noise power is 2*sigma^2, not sigma^2 — it is complex. The
        # SNR quoted for the defaults follows from that: 10*log10(40^2 /
        # (2*0.6^2)) = 33.5 dB, and counting one quadrature overstates it by
        # exactly 3 dB. Asserted because it is the number a reader uses to
        # decide whether a downstream result is noise-limited.
        power = sum(r * r + i * i for r, i in zip(reals, imags)) / count
        self.assertAlmostEqual(power, 2.0 * sigma * sigma, delta=0.02)
        snr_db = 10.0 * math.log10(40.0**2 / (2.0 * 0.6**2))
        self.assertAlmostEqual(snr_db, 33.47, places=2)

    def test_cfo_rotates_phase_without_touching_amplitude(self) -> None:
        """Why the pipeline is amplitude-only (RESEARCH.md §5a), as a test."""
        geo = Geometry.perimeter(4)
        out = {}
        for cfo in (False, True):
            sim = MultiLinkSimulator(
                geo, SimConfig(seed=23, packet_loss=0.0, noise_std=0.0, cfo=cfo)
            )
            out[cfo] = [
                (tuple(f.amplitudes), tuple(f.phases)) for _t, _tx, _rx, f in sim.stream(2.0)
            ]

        self.assertEqual(len(out[False]), len(out[True]))
        rotations = []
        for (amp_a, ph_a), (amp_b, ph_b) in zip(out[False], out[True]):
            for a, b in zip(amp_a, amp_b):
                self.assertAlmostEqual(a, b, places=9)
            offsets = [
                (pb - pa + math.pi) % (2.0 * math.pi) - math.pi
                for pa, pb in zip(ph_a, ph_b)
            ]
            # One common rotation for the whole packet, not per subcarrier.
            self.assertLess(max(offsets) - min(offsets), 1e-9)
            rotations.append(offsets[0])
        self.assertGreater(max(rotations) - min(rotations), 3.0, "CFO must vary")

    def test_noise_is_independent_across_the_receivers_of_one_broadcast(self) -> None:
        """"Per reception, not per transmission" — asserted, not just claimed.

        `test_noise_is_circular_complex_gaussian` uses a two-node geometry,
        which has exactly one receiver per slot, so it structurally cannot
        see a noise vector shared by the receivers of one broadcast. Hoisting
        the noise draw out of the receiver loop left all 31 old tests
        passing. It matters beyond realism: RTI inversion assumes the L link
        measurements are independent observations, and slot-correlated noise
        makes the regularized solve look better conditioned than any real
        mesh will be.
        """
        geo = Geometry.perimeter(4)
        sim = MultiLinkSimulator(
            geo,
            SimConfig(seed=17, packet_loss=0.0, occupied=False, cfo=False, noise_std=0.6),
        )
        clean = {
            (tx, rx): sim.noiseless_channel(tx, rx, 0.0, 0.0)
            for tx in range(4)
            for rx in range(4)
            if tx != rx
        }

        by_slot: dict[float, dict[tuple[int, int], list[float]]] = {}
        for t, tx, rx, frame in sim.stream(30.0):
            residual = [
                (cmath.rect(a, p) - clean[(tx, rx)][k]).real
                for k, (a, p) in enumerate(zip(frame.amplitudes, frame.phases))
            ]
            by_slot.setdefault(t, {})[(tx, rx)] = residual

        for first, second in (((0, 1), (0, 2)), ((0, 1), (0, 3)), ((1, 0), (1, 2))):
            with self.subTest(pair=(first, second)):
                xs, ys = [], []
                for slot in by_slot.values():
                    if first in slot and second in slot:
                        xs.extend(slot[first])
                        ys.extend(slot[second])
                n = len(xs)
                self.assertGreater(n, 20_000)
                cross = sum(a * b for a, b in zip(xs, ys)) / n
                norm = math.sqrt(
                    (sum(a * a for a in xs) / n) * (sum(b * b for b in ys) / n)
                )
                # Correlation coefficient, against 4 standard errors of zero.
                # A shared noise vector would put this at exactly 1.
                self.assertLess(abs(cross / norm), 4.0 / math.sqrt(n))

    def test_packet_loss_is_per_reception_not_per_broadcast(self) -> None:
        """One broadcast can be heard by some receivers and missed by others.

        Counted as the fraction of slots in which EVERY receiver survived:
        (1-p)^(n-1) for per-reception loss, (1-p) if a lost broadcast took
        all its receptions with it. The existing loss tests only count
        packets and gaps, both of which bursty slot-level loss satisfies
        just as well.
        """
        geo = Geometry.perimeter(4)
        duration, loss = 40.0, 0.3
        slots = {
            t for t, _tx, _rx, _f in MultiLinkSimulator(
                geo, SimConfig(seed=13, packet_loss=0.0)
            ).stream(duration)
        }
        sim = MultiLinkSimulator(geo, SimConfig(seed=13, packet_loss=loss))
        received: dict[float, int] = {}
        for t, _tx, _rx, _f in sim.stream(duration):
            received[t] = received.get(t, 0) + 1

        n_slots = len(slots)
        self.assertGreater(n_slots, 1500)
        intact = sum(1 for t in slots if received.get(t, 0) == geo.n_nodes - 1)
        observed = intact / n_slots
        expected = (1.0 - loss) ** (geo.n_nodes - 1)
        sigma = math.sqrt(expected * (1.0 - expected) / n_slots)
        self.assertLess(
            abs(observed - expected),
            4.0 * sigma,
            f"{observed:.4f} intact slots against {expected:.4f} expected; "
            f"slot-level loss would give {1.0 - loss:.4f}",
        )

    def test_cfo_is_drawn_per_packet_not_per_slot(self) -> None:
        """Two receivers of the same broadcast have unrelated oscillators.

        Their PLLs free-run independently (RESEARCH.md §0 — it is why two
        ESP32s cannot form a coherent array), so a common per-slot phase
        would be modelling a mesh that does not exist. Drawing phi once per
        slot also passed the whole old suite, because the CFO test measures
        the spread across a run and never across one slot.
        """
        geo = Geometry.perimeter(4)
        sim = MultiLinkSimulator(
            geo, SimConfig(seed=23, packet_loss=0.0, noise_std=0.0, occupied=False)
        )
        clean = {
            (tx, rx): sim.noiseless_channel(tx, rx, 0.0, 0.0)
            for tx in range(4)
            for rx in range(4)
            if tx != rx
        }

        spreads = []
        for_slot: dict[float, list[float]] = {}
        for t, tx, rx, frame in sim.stream(10.0):
            rot = frame.phases[0] - cmath.phase(clean[(tx, rx)][0])
            for_slot.setdefault(t, []).append((rot + math.pi) % (2.0 * math.pi) - math.pi)
        for rotations in for_slot.values():
            self.assertEqual(len(rotations), 3)
            spreads.append(max(rotations) - min(rotations))

        self.assertGreater(len(spreads), 100)
        self.assertEqual(
            sum(1 for s in spreads if s < 1e-9),
            0,
            "the receivers of a slot must not share one CFO draw",
        )
        # Three uniform draws on (-pi, pi] have an expected spread of pi.
        self.assertGreater(sum(spreads) / len(spreads), 2.0)

    def test_rssi_tracks_received_power(self) -> None:
        geo = Geometry.perimeter(8)
        sim = MultiLinkSimulator(geo, SimConfig(seed=29, packet_loss=0.0))
        samples = []
        for _t, _tx, _rx, frame in sim.stream(2.0):
            power = sum(a * a for a in frame.amplitudes) / len(frame.amplitudes)
            samples.append((10.0 * math.log10(power), frame.rssi_dbm))

        for db, rssi in samples:
            self.assertIsInstance(rssi, int)
            self.assertGreaterEqual(rssi, -100)
            self.assertLessEqual(rssi, 0)
            # Quantised to 1 dB around a fixed calibration offset.
            self.assertLessEqual(abs(rssi - (db - 45.0 - 20.0 * math.log10(40.0))), 0.5)
        self.assertGreater(len({r for _d, r in samples}), 3, "RSSI must vary by link")


class TestGroundTruth(unittest.TestCase):
    def test_walk_stays_in_room_and_respects_speed_limits(self) -> None:
        room = Room(8.0, 6.0)
        sim = MultiLinkSimulator(
            Geometry.perimeter(8),
            SimConfig(seed=41, speed_mps=(0.3, 1.2), waypoint_margin_m=0.5),
        )
        dt = 0.02
        truth = list(sim.ground_truth(300.0, rate_hz=1.0 / dt))
        self.assertEqual(len(truth), 15_000)

        speeds = []
        for a, b in zip(truth, truth[1:]):
            self.assertTrue(room.contains(b.x, b.y, tol=0.0))
            self.assertGreaterEqual(b.x, 0.5 - 1e-9)
            self.assertLessEqual(b.x, 7.5 + 1e-9)
            step = math.hypot(b.x - a.x, b.y - a.y)
            self.assertLessEqual(step, 1.2 * dt + 1e-9, "faster than speed_max")
            if a.moving and b.moving:
                speeds.append(step / dt)
            if not a.moving and not b.moving:
                self.assertAlmostEqual(step, 0.0, places=12, msg="paused and drifting")

        self.assertGreater(len(speeds), 1000)
        slow = sum(1 for s in speeds if s < 0.3 - 1e-9)
        # Pairs straddling a segment boundary average two speeds, or part of a
        # pause, and legitimately fall below the minimum; everything else must
        # sit inside the configured band.
        self.assertLess(slow / len(speeds), 0.05)

        moving_fraction = sum(1 for g in truth if g.moving) / len(truth)
        self.assertGreater(moving_fraction, 0.1)
        self.assertLess(moving_fraction, 0.9)

    def test_random_access_matches_the_sequential_stream(self) -> None:
        sim = MultiLinkSimulator(Geometry.perimeter(4), SimConfig(seed=43))
        forward = list(sim.ground_truth(50.0, rate_hz=4.0))
        # Query backwards: the lazily extended trajectory must not depend on
        # the order it is asked about.
        backward = [sim.truth_at(g.t) for g in reversed(forward)]
        self.assertEqual(forward, list(reversed(backward)))

    def test_static_motion_never_moves(self) -> None:
        motion = StaticMotion(2.5, 1.5)
        for t in (0.0, 1.0, 1e3, 1e6):
            g = motion.at(t)
            self.assertEqual((g.x, g.y, g.moving), (2.5, 1.5, False))

    def test_waypoint_motion_is_exact(self) -> None:
        motion = WaypointMotion([(0.0, 0.0), (4.0, 0.0)], speed_mps=2.0)
        self.assertEqual(motion.at(0.0).x, 0.0)
        self.assertAlmostEqual(motion.at(1.0).x, 2.0, places=12)
        self.assertTrue(motion.at(1.0).moving)
        self.assertAlmostEqual(motion.at(2.0).x, 4.0, places=12)
        # Held still after the script runs out.
        self.assertFalse(motion.at(5.0).moving)
        self.assertAlmostEqual(motion.at(5.0).x, 4.0, places=12)

    def test_waypoint_pause_holds_position(self) -> None:
        motion = WaypointMotion(
            [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)], speed_mps=1.0, pause_s=3.0
        )
        self.assertTrue(motion.at(0.5).moving)
        for t in (1.0, 2.5, 3.9):
            g = motion.at(t)
            self.assertFalse(g.moving, t)
            self.assertAlmostEqual(g.x, 1.0, places=12)
        self.assertTrue(motion.at(4.5).moving)

    def test_random_waypoint_reset_is_reproducible(self) -> None:
        motion = RandomWaypointMotion(Room(8.0, 6.0), seed="abc")
        first = [motion.at(t) for t in (0.0, 5.0, 25.0, 60.0)]
        motion.reset()
        self.assertEqual(first, [motion.at(t) for t in (0.0, 5.0, 25.0, 60.0)])


class TestConfigValidation(unittest.TestCase):
    def test_rejects_impossible_settings(self) -> None:
        for kwargs in (
            {"slot_ms": 0.0},
            {"packet_loss": 1.0},
            {"packet_loss": -0.1},
            {"n_subcarriers": 0},
            {"noise_std": -1.0},
            {"shadow_delta0_m": 0.0},
            {"wall_reflection": 1.0},
            {"body_rcs_m2": -0.5},
            {"body_radius_m": 0.0},
            # A negative margin is the dangerous one: it passes every other
            # check and widens the uniform waypoint draw past the walls, so
            # the ground-truth stream reports a person outside the room —
            # which the pixel grid clamps and nothing raises on, leaving it
            # to surface as unexplained tracker bias.
            {"waypoint_margin_m": -1.5},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    SimConfig(**kwargs)

    def test_rejects_impossible_motion(self) -> None:
        room = Room(8.0, 6.0)
        with self.assertRaises(ValueError):
            RandomWaypointMotion(room, speed_mps=(0.0, 1.0))
        with self.assertRaises(ValueError):
            RandomWaypointMotion(room, speed_mps=(1.2, 0.3))
        with self.assertRaises(ValueError):
            RandomWaypointMotion(room, margin_m=3.0)
        with self.assertRaises(ValueError):
            RandomWaypointMotion(room, margin_m=-2.0)
        with self.assertRaises(ValueError):
            WaypointMotion([(1.0, 1.0)], speed_mps=0.0)
        with self.assertRaises(ValueError):
            slot_ms_for_per_link_rate(1, 5.0)
        with self.assertRaises(ValueError):
            slot_ms_for_per_link_rate(8, 0.0)
        with self.assertRaises(ValueError):
            slot_ms_for_aggregate_rate(1, 5.0)
        with self.assertRaises(ValueError):
            slot_ms_for_aggregate_rate(8, 0.0)

    def test_a_path_of_coincident_waypoints_means_standing_there(self) -> None:
        """[A, A] is how a test author writes "stand at A", not an error.

        Every leg is zero-length and the default pause is zero, so nothing
        was pushed and the segment list came out empty; `at()` then indexed
        it and raised IndexError — from inside `stream`, several modules
        away from the waypoint list that caused it.
        """
        for points in (
            [(2.0, 3.0), (2.0, 3.0)],
            [(1.0, 1.0), (1.0, 1.0), (1.0, 1.0)],
        ):
            with self.subTest(points=points):
                motion = WaypointMotion(points, speed_mps=1.0)
                for t in (0.0, 1.0, 1e6):
                    g = motion.at(t)
                    self.assertEqual((g.x, g.y, g.moving), (*points[0], False))

        sim = MultiLinkSimulator(
            Geometry.perimeter(4),
            SimConfig(seed=1),
            motion=WaypointMotion([(2.0, 2.0), (2.0, 2.0)]),
        )
        _t, _tx, _rx, frame = next(iter(sim.stream(1.0)))
        self.assertEqual(len(frame.amplitudes), 56)

    def test_extreme_amplitudes_do_not_break_the_rssi_conversion(self) -> None:
        """`round()` raises on inf and NaN, and it ran before the clamp.

        Any configuration that overflows a subcarrier amplitude past ~1.3e154
        squares to inf in the power sum, and the stream then died with
        "cannot convert float infinity to integer" rather than reporting a
        pegged RSSI. All of these pass `SimConfig.__post_init__`.
        """
        for kwargs in (
            {"noise_std": 1e200},
            {"reference_magnitude": 1e200},
            {"noise_std": 0.0, "reference_magnitude": 1e-200},
        ):
            with self.subTest(**kwargs):
                sim = MultiLinkSimulator(
                    Geometry.perimeter(4), SimConfig(seed=1, packet_loss=0.0, **kwargs)
                )
                _t, _tx, _rx, frame = next(iter(sim.stream(0.1)))
                self.assertIsInstance(frame.rssi_dbm, int)
                self.assertGreaterEqual(frame.rssi_dbm, -100)
                self.assertLessEqual(frame.rssi_dbm, 0)

    def test_a_non_finite_body_position_is_rejected_where_it_enters(self) -> None:
        """NaN survives every `<` comparison, so it has to be caught by name.

        It used to travel the whole chain — geometry, channel, power sum —
        and surface as "cannot convert float NaN to integer" from the RSSI
        quantiser, which is as far from the cause as it is possible to get.
        """
        geo = Geometry.perimeter(4)
        for x, y in (
            (float("nan"), 3.0),
            (4.0, float("nan")),
            (float("inf"), 3.0),
            (4.0, float("-inf")),
        ):
            with self.subTest(x=x, y=y):
                sim = MultiLinkSimulator(
                    geo, SimConfig(seed=1, packet_loss=0.0), motion=StaticMotion(x, y)
                )
                with self.assertRaises(ValueError):
                    next(iter(sim.stream(0.1)))
                with self.assertRaises(ValueError):
                    sim.noiseless_channel(0, 1, x, y)


if __name__ == "__main__":
    unittest.main()
