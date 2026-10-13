"""Multi-link synthetic CSI: the ground truth everything else is scored against.

This module is the measuring stick. `measure`, `invert` and `track` are all
evaluated against the trajectory this file invents, so an error here does not
show up as a failing test — it shows up as a localizer that confidently
reports coordinates derived from physics that does not exist. That asymmetry
is why this file carries more explanation per line than the rest of the
package, and why every modelling shortcut is named in the docstrings rather
than buried.

What is modelled, per link and per subcarrier (K = 56 on the 802.11n HT20
layout — subcarriers -28..-1 and +1..+28 at 312.5 kHz around 2.437 GHz,
with no sample at DC, where a real HT20 symbol carries a null):

  1. Line of sight, amplitude ~ 1/d, attenuated by a body-shadowing gain.
  2. First-order wall reflections by the image method, four walls, amplitude
     reflection coefficient ~0.4 with a pi flip. Static in time.
  3. One body-scattered path, amplitude ~ 1/(d1*d2), carrying a pi flip and
     a forward-direction taper (see below), with chest-displacement
     (breathing) modulation of its path length.
  4. Complex Gaussian receiver noise plus a uniform random common phase per
     packet, which is what carrier frequency offset looks like once it has
     wrapped a few hundred times between packets.

TERMS 1 AND 3 ARE ONE TORSO COUNTED TWICE, AND THAT IS THE TRAP. Shadowing
is a *loss* on the direct path; the scattered path is a *vector addition*
whose phase sweeps with position, and it is what carries breathing. Away
from the line of sight they are genuinely different mechanisms. In the
forward direction they are not: a shadow *is* the forward-scattered field
cancelling the incident one, so a model that applies a shadowing law and
then adds an untapered scattered path has written the body down twice.

An earlier revision of this file did exactly that, with a positive real
scattering coefficient. On the line of sight d1 + d2 == d, so that term was
*exactly in phase* with the direct path at every subcarrier and the body
filled in its own shadow: a configured S0 = 6 dB was delivered as 3.96 dB
on a 6 m link, 0.91 dB on a 2 m link and +2.27 dB — a *brighter* link — on
a 1 m one. In the walkable region of an eight-node room, 17.5% of (link,
position) pairs with the body inside the link's 0.3 m excess ellipse failed
this module's own "a blocked link loses more than 1 dB" criterion and 2.3%
of them came out BRIGHTER, by up to 4.56 dB. Both repairs below are
physics, not tuning:

  * The scattered amplitude carries a pi flip, like the wall images.
    Forward scatter is extinction; an out-of-phase forward field is what
    creates a shadow at all, so the scatter must deepen it, never fill it.
  * It is tapered by |cos(beta/2)|, beta being the tx-body-rx angle, which
    vanishes in the forward direction where the shadowing law has already
    accounted for the body. The taper is what removes the double count, and
    it is the same geometric factor that already governs the breathing
    sensitivity, so it adds no new null family: a body on the line of sight
    was always breathing-invisible.

What the model now delivers, which is the number downstream sees rather
than the one in the config (SYNTHETIC, measured by
tests/test_localize_sim.py::TestDeliveredBlockage):

  * body on the line of sight, walls off: exactly the configured S0 —
    6.000 dB at every link length from 0.6 m to 7 m.
  * body on the line of sight, walls on (`Geometry.perimeter(8)`, walkable
    region, 0.05 m grid): median 3.2 dB of a configured 6 dB, worst case
    1.47 dB. The missing depth is not the body term: shadowing attenuates
    the direct path only (see WHAT IS NOT MODELLED), so the unshadowed wall
    reflections dilute it.
  * no link with the body inside its 0.15 m excess ellipse gains power
    anywhere in the walkable region; the worst such link still loses
    1.11 dB. Widening the ellipse to 0.3 m, 0.05% of pairs do gain, by up
    to 0.78 dB, and every one of them is within 0.75 m of a node, where a
    point-scatterer model is out of its domain either way.

TIMING IS NOT A DETAIL. Nodes share one channel by TDMA: a superframe is
`n_nodes * slot_ms`, node i transmits in slot i, everyone else receives, so
every link is sampled exactly once per superframe and the per-link rate is
`1 / (n_nodes * slot)`. Adding nodes buys links and costs rate, and that
trade is the entire question phase P0b exists to answer. RESEARCH.md §0
records the one published multi-node ESP32 mesh: TDMA across 9 nodes gave
~1.4 Hz AGGREGATE CSI, at which ambulatory motion was detected ~100% of the
time but sedentary presence only 38%. Aggregate, not per link, and the
difference is a factor of 72, so it is worth writing both readings out:

  * 1.4 Hz aggregate across 9 nodes is a 5714 ms slot — see
    `slot_ms_for_aggregate_rate` — at which each of the 72 links is
    sampled once every 51 s, i.e. 0.019 Hz per link.
  * 1.4 Hz *per link* across 9 nodes would be a 79 ms slot and 100.8 Hz
    aggregate, which is not what was measured.

The 20 ms default here is 350 Hz aggregate on 8 nodes, i.e. 250x the only
published mesh's total CSI throughput; it assumes firmware-level scheduling
rather than a backend round trip. Sweep the slot across both readings; a
result quoted at the default is a result about hardware nobody has built.

DETERMINISM. One `seed` fixes everything, through four independent named
RNG streams (motion, noise, CFO, packet loss). Independent on purpose: a
sweep over `packet_loss` or `noise_std` must not reshuffle the walk, or the
configurations being compared are not the same experiment. The noise draws
for a packet are consumed even when that packet is dropped, so the noise
realisation of surviving packets is also independent of the loss rate.

WHAT IS NOT MODELLED — read this before believing a result:

  * 2-D. No floor or ceiling bounce, which in a real room is a strong,
    always-present reflection. The static multipath here is correspondingly
    cleaner than reality.
  * Shadowing attenuates only the direct path, not the wall reflections,
    though a body standing in a corner blocks those too. Shadowing is
    therefore under-modelled for links whose reflections matter most.
  * The body is one point scatterer plus one scalar shadowing law. A real
    torso is an extended, orientation-dependent, partly diffracting object,
    and the empirical spread of body-blocking loss at 2.4 GHz is several dB
    at any fixed geometry. Here it is a deterministic function of position.
    The point-scatterer amplitude is softened by `body_radius_m` rather
    than left to diverge, but inside ~0.5 m of a node the model is out of
    its validity domain whatever the softening, because the far-field
    distance of a 0.4 m torso at 12.3 cm is 2.6 m.
  * LINKS ARE RECIPROCAL. Before noise and CFO the two directions of a pair
    are bit-identical (measured: max |H(i->j) - H(j->i)| = 6.4e-13 over
    `Geometry.perimeter(8)`), because the image sum obeys
    |M(A)-B| = |A-M(B)| and both body terms are symmetric in d1/d2. The
    deployment keeps all N(N-1) ordered links — geometry.py argues they are
    two independent observations because tx and rx differ in noise floor,
    antenna orientation and slot — but this file models none of that
    asymmetry. Only the noise, the CFO draw and the loss draw differ, so
    the measurement vector of length N(N-1) carries N(N-1)/2 distinct
    signals plus independent noise, and an L x L dual-form solve on it has
    exactly duplicated rows held apart only by the regularizer. Report the
    effective link count, not the nominal one. (A per-node transmit-power
    offset would break the bit-identity without fixing this: the body's dB
    effect would still be the same in both directions, so it would hide the
    redundancy rather than remove it. Hence a documented limitation and not
    a fix.)
  * One chest direction, drawn once per run (`chest_direction_rad`), but a
    single scalar displacement: the torso translates, it does not expand.
    Links whose tx-body-rx bisector is perpendicular to that direction see
    no breathing, which is a second null family on top of the forward
    scatter null. It is modelled because the alternative — giving every
    link its own best-case direction, as this file used to — overstates the
    per-link breathing sensitivity by up to 20x (measured on
    `Geometry.perimeter(6)` with a person at (3.0, 4.5): link 1->4 reads
    0.602 m/m with a per-link bisector against 0.030 m/m for a chest facing
    +y). Downstream vitals results inherit whichever assumption is here, so
    it is the pessimistic-but-real one.
  * Breathing is rendered on every packet, including while the person
    walks, which P0B_TASK.md 3.1 term 3 does not ask for. Deliberate — the
    alternative is a discontinuity at every pause boundary that a
    downstream detector reads as an event — and pinned by a test, so that
    "fixing" it has to be a decision rather than an accident.
  * No AGC. Real ESP32 frames carry a per-packet gain step that moves the
    whole amplitude vector; it is the largest single confound for any
    variance-based measurement and it is absent here, so `measure` will look
    better on this data than on a radio. `rssi_dbm` is quantised to 1 dB,
    which is the one piece of that realism that is present.
  * Packet loss is i.i.d.; real loss is bursty.
  * No furniture scatterers, so static multipath is smoother across the band
    than a real room's.
"""

from __future__ import annotations

import cmath
import math
import random
from bisect import bisect_right
from collections.abc import Iterator
from dataclasses import dataclass
from typing import NamedTuple

from ..types import CsiFrame
from .geometry import Geometry, Room

__all__ = [
    "SPEED_OF_LIGHT",
    "GroundTruth",
    "StaticMotion",
    "WaypointMotion",
    "RandomWaypointMotion",
    "SimConfig",
    "MultiLinkSimulator",
    "slot_ms_for_per_link_rate",
    "slot_ms_for_aggregate_rate",
]

SPEED_OF_LIGHT = 299_792_458.0

# What `SimConfig.reference_magnitude` corresponds to in RSSI. Fixes the
# otherwise arbitrary offset between the simulator's linear amplitude units
# and the dBm the ESP32 reports, so that a mid-length link reads about the
# -45 dBm a node a few metres away actually reports.
_RSSI_REFERENCE_DBM = -45.0

_TWO_PI = 2.0 * math.pi


def slot_ms_for_per_link_rate(n_nodes: int, rate_hz: float) -> float:
    """Slot length giving `rate_hz` on EACH LINK with `n_nodes` in the ring.

    The inverse of the per-link rate law `1 / (n_nodes * slot)`, provided
    because the design question runs this way round: "breathing needs 2 Hz
    per link at 8 nodes — what slot do I need?" (62.5 ms).

    This is NOT the function to reach for when a figure is quoted as an
    aggregate. RESEARCH.md §0's "~1.4 Hz aggregate CSI across 9 nodes" run
    through here returns 79.4 ms, a configuration whose aggregate rate is
    100.8 Hz — 72x the measurement it was meant to reproduce. Use
    `slot_ms_for_aggregate_rate` for that, and see the module docstring.
    """
    if n_nodes < 2:
        raise ValueError(f"need at least 2 nodes, got {n_nodes}")
    if rate_hz <= 0.0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    return 1000.0 / (n_nodes * rate_hz)


def slot_ms_for_aggregate_rate(n_nodes: int, rate_hz: float) -> float:
    """Slot length giving `rate_hz` of CSI in TOTAL across every link.

    The aggregate rate is `n_links / superframe = n*(n-1) / (n*slot)`, so
    the n cancels and the slot is `(n_nodes - 1) / rate_hz` — one broadcast
    per slot is heard by n-1 receivers, and that is the only thing airtime
    buys. The two helpers exist as a pair so the published mesh's figure
    cannot quietly be read as the other quantity: 1.4 Hz aggregate across 9
    nodes is a 5714 ms slot, against 79 ms for 1.4 Hz per link.
    """
    if n_nodes < 2:
        raise ValueError(f"need at least 2 nodes, got {n_nodes}")
    if rate_hz <= 0.0:
        raise ValueError(f"rate_hz must be positive, got {rate_hz}")
    return 1000.0 * (n_nodes - 1) / rate_hz


# --------------------------------------------------------------------------
# Ground truth and motion
# --------------------------------------------------------------------------


class GroundTruth(NamedTuple):
    """Where the person is at time `t`, and whether they are walking.

    A NamedTuple rather than the frozen dataclass used elsewhere in the
    project, because the shared interface contract specifies this stream as
    `(t, x, y, moving)`. This way both readings work: a consumer can unpack
    it positionally as the contract writes it, or reach for `.moving` and
    say what it means.

    `moving` is the flag the whole evaluation hinges on. VRTI measures
    variance, so it sees motion and is blind to a person standing still;
    scoring "while moving" and "while still" separately — rather than
    averaging them into one number — is the difference between an honest
    result and a flattering one.
    """

    t: float
    x: float
    y: float
    moving: bool


class _SegmentMotion:
    """Piecewise-linear trajectory evaluated at an arbitrary time.

    Random access matters: packets arrive at TDMA slot instants that no
    motion model would naturally choose, and the evaluation wants ground
    truth on its own grid. So motion is stored as contiguous segments and
    `at()` interpolates, rather than being stepped forward with the stream.
    """

    __slots__ = ("_segments", "_starts")

    def __init__(self) -> None:
        # (t0, t1, x0, y0, x1, y1, moving), contiguous and in time order.
        self._segments: list[tuple[float, float, float, float, float, float, bool]] = []
        self._starts: list[float] = []

    def _push(
        self,
        t0: float,
        t1: float,
        x0: float,
        y0: float,
        x1: float,
        y1: float,
        moving: bool,
    ) -> None:
        self._segments.append((t0, t1, x0, y0, x1, y1, moving))
        self._starts.append(t0)

    @property
    def _end(self) -> float:
        return self._segments[-1][1] if self._segments else 0.0

    def _extend_to(self, t: float) -> None:
        """Generate segments until the trajectory covers `t`. May be a no-op."""

    def at(self, t: float) -> GroundTruth:
        if t < 0.0:
            t = 0.0
        if not self._segments or t >= self._end:
            self._extend_to(t)
        i = bisect_right(self._starts, t) - 1
        if i < 0:
            i = 0
        t0, t1, x0, y0, x1, y1, moving = self._segments[i]
        if t >= t1:
            # Past the end of a finite trajectory: hold the last position.
            return GroundTruth(t, x1, y1, False)
        span = t1 - t0
        f = 0.0 if span <= 0.0 else (t - t0) / span
        return GroundTruth(t, x0 + f * (x1 - x0), y0 + f * (y1 - y0), moving)

    def reset(self) -> None:
        """Return to t = 0 reproducibly."""


class StaticMotion(_SegmentMotion):
    """A person who never moves. Breathes, does not walk.

    The adversarial case for VRTI and the one worth testing hardest: a
    variance-based image goes dark, and a tracker that reports a position
    anyway is reporting the room's geometry, not the person.
    """

    def __init__(self, x: float, y: float) -> None:
        super().__init__()
        self._push(0.0, math.inf, x, y, x, y, False)


class WaypointMotion(_SegmentMotion):
    """A scripted path at constant speed, with an optional pause at each stop.

    Deterministic without an RNG, which is what a test wants when it needs to
    know the true position analytically — a straight-line walk for a Kalman
    filter, or a pass at a known offset from one link's line of sight.
    """

    def __init__(
        self,
        points: list[tuple[float, float]],
        speed_mps: float = 1.0,
        pause_s: float = 0.0,
    ) -> None:
        super().__init__()
        if len(points) < 1:
            raise ValueError("need at least one waypoint")
        if speed_mps <= 0.0:
            raise ValueError(f"speed_mps must be positive, got {speed_mps}")
        if pause_s < 0.0:
            raise ValueError(f"pause_s must be non-negative, got {pause_s}")

        t = 0.0
        x, y = points[0]
        if len(points) == 1:
            self._push(0.0, math.inf, x, y, x, y, False)
            return
        for nx, ny in points[1:]:
            dist = math.hypot(nx - x, ny - y)
            if dist > 0.0:
                dt = dist / speed_mps
                self._push(t, t + dt, x, y, nx, ny, True)
                t += dt
                x, y = nx, ny
            if pause_s > 0.0:
                self._push(t, t + pause_s, x, y, x, y, False)
                t += pause_s

        if not self._segments:
            # Every leg was zero-length and there is no pause to stand in,
            # e.g. [(2, 3), (2, 3)]. Without this the segment list is empty
            # and `at()` indexes it, so a test author writing "stand at A,
            # then walk to B" as [A, A, B] gets an IndexError from inside
            # the stream rather than a position. Hold, matching StaticMotion
            # — a path of coincident waypoints means standing there.
            self._push(0.0, math.inf, x, y, x, y, False)


class RandomWaypointMotion(_SegmentMotion):
    """Random waypoint: walk to a uniformly drawn point, pause, repeat.

    Segments are generated lazily as time is queried, from this object's own
    seeded RNG, so the trajectory depends on the seed alone and not on how
    far ahead anything happened to look.

    The default 0-20 s pause means the person is standing still for more than
    half the run. That is deliberate and it is the realistic case: it makes
    "coverage" (the fraction of time the tracker does not abstain) a number
    with teeth, instead of one measured on a subject who conveniently never
    stops walking.

    Known bias: a uniform-waypoint random walk concentrates near the room
    centre, which also happens to be where the link density is highest and
    where the "always guess the centre" baseline wins. Any evaluation using
    this model has to report the centre baseline alongside, or it will
    mistake the sampling bias for skill.
    """

    __slots__ = ("_room", "_seed", "_rng", "_speed", "_pause", "_margin", "_x", "_y", "_start")

    def __init__(
        self,
        room: Room,
        seed: int | str = 0,
        speed_mps: tuple[float, float] = (0.3, 1.2),
        pause_s: tuple[float, float] = (0.0, 20.0),
        margin_m: float = 0.5,
        start: tuple[float, float] | None = None,
    ) -> None:
        super().__init__()
        if not 0.0 < speed_mps[0] <= speed_mps[1]:
            raise ValueError(f"need 0 < speed_min <= speed_max, got {speed_mps}")
        if not 0.0 <= pause_s[0] <= pause_s[1]:
            raise ValueError(f"need 0 <= pause_min <= pause_max, got {pause_s}")
        # Bounded from below as well as above: a negative margin widens the
        # uniform draw past the walls and the ground-truth stream then
        # reports a person outside the room, which no pixel grid can
        # represent and nothing downstream raises on — it surfaces as
        # unexplained tracker bias. `perimeter_nodes` guards `inset_m` the
        # same way for the same reason.
        if margin_m < 0.0:
            raise ValueError(f"margin_m must be non-negative, got {margin_m}")
        if 2.0 * margin_m >= min(room.width_m, room.height_m):
            raise ValueError(
                f"margin_m={margin_m} leaves no room to walk in a "
                f"{room.width_m} x {room.height_m} room"
            )

        self._room = room
        self._seed = seed
        self._speed = speed_mps
        self._pause = pause_s
        self._margin = margin_m
        self._start = start
        self._rng = random.Random(seed)
        self._x, self._y = start if start is not None else self._draw_point()

    def _draw_point(self) -> tuple[float, float]:
        m = self._margin
        return (
            self._rng.uniform(m, self._room.width_m - m),
            self._rng.uniform(m, self._room.height_m - m),
        )

    def _next_waypoint(self) -> tuple[float, float]:
        # A zero-length walk would be a zero-duration segment and the
        # extension loop would never advance. Redrawing is the clean fix; the
        # draw is continuous so the retry essentially never runs twice.
        for _ in range(16):
            x, y = self._draw_point()
            if math.hypot(x - self._x, y - self._y) > 1e-6:
                return x, y
        return (self._x + 1e-3, self._y)

    def _extend_to(self, t: float) -> None:
        while not self._segments or self._end <= t:
            t0 = self._end
            x1, y1 = self._next_waypoint()
            speed = self._rng.uniform(*self._speed)
            dt = math.hypot(x1 - self._x, y1 - self._y) / speed
            self._push(t0, t0 + dt, self._x, self._y, x1, y1, True)
            self._x, self._y = x1, y1
            pause = self._rng.uniform(*self._pause)
            if pause > 0.0:
                self._push(t0 + dt, t0 + dt + pause, x1, y1, x1, y1, False)

    def reset(self) -> None:
        self._segments.clear()
        self._starts.clear()
        self._rng = random.Random(self._seed)
        self._x, self._y = (
            self._start if self._start is not None else self._draw_point()
        )


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------


@dataclass
class SimConfig:
    """Scene, radio and timing parameters. Defaults are justified in-line."""

    # -- OFDM channel: one ESP32 HT20 channel ------------------------------
    n_subcarriers: int = 56
    carrier_hz: float = 2.437e9
    subcarrier_spacing_hz: float = 312_500.0

    # -- TDMA --------------------------------------------------------------
    # 20 ms is firmware-scheduled broadcast: a CSI packet is ~1 ms on air, so
    # this is generous for the radio and tight for the scheduler. The one
    # published mesh (RESEARCH.md §0) effectively ran ~79 ms. Sweep it.
    slot_ms: float = 20.0
    # Per reception, not per transmission: one broadcast can be heard by some
    # receivers and missed by others, which is what a shared channel does.
    packet_loss: float = 0.02

    # -- Amplitude scale ---------------------------------------------------
    # Line of sight is `reference_magnitude` at `reference_distance_m` and
    # falls as 1/d, so a 2 m link reads ~100 and an 8 m link ~25 — the order
    # of magnitude the ESP32's int8 CSI pairs actually occupy.
    reference_magnitude: float = 40.0
    reference_distance_m: float = 5.0

    # -- Static scene ------------------------------------------------------
    # Amplitude reflection coefficient; drywall and brick at 2.4 GHz sit
    # around 0.3-0.5. Applied with a pi flip, as a dielectric reflection at
    # grazing incidence has.
    wall_reflection: float = 0.4

    # -- Body --------------------------------------------------------------
    # Bistatic RCS of a standing adult torso at 2.4 GHz. Published values
    # scatter across roughly 0.1-1 m^2 depending on aspect.
    #
    # The scattered path is tapered to zero in the forward direction (see
    # the module docstring), so the old justification for 0.5 — "~13% of
    # line of sight for a person mid-way along a 6 m link" — no longer
    # describes anything the model emits, and it was measured against the
    # UNSHADOWED direct path in any case. Re-derived against the shadowed
    # path it actually adds to: a person 1 m off the midpoint of a 6 m link
    # scatters 3.8% of the unshadowed direct path and 4.3% of the shadowed
    # one, rising to 9.0% at 2 m off. That is the regime the vitals
    # pipeline works in, and it is where the number should be set, because
    # on the line of sight this term is zero by construction.
    body_rcs_m2: float = 0.5
    # Half-extent of the torso, used to soften 1/(d1*d2) in the scattered
    # amplitude only. A point scatterer diverges when you stand on it; a
    # body cannot be approached closer than its own radius, so
    # d -> hypot(d, body_radius_m) is a physical regularisation rather than
    # a clamp. It must not touch the shadowing excess or the scattering
    # angle: those are geometry, and rewriting them (as a hard 0.05 m clamp
    # on d1 and d2 used to) makes the applied loss disagree with
    # `link_shadowing_db` and resurrects breathing inside the forward null.
    body_radius_m: float = 0.25
    # Direction the chest wall displaces along, radians, +x = 0. None draws
    # one per run from the seed. One chest has ONE direction; giving every
    # link its own bisector is the maximal assumption and overstates
    # per-link breathing sensitivity by up to 20x.
    chest_direction_rad: float | None = None
    occupied: bool = True

    # -- Shadowing: S(delta) = S0 * exp(-delta / delta0), in dB ------------
    # S0 = 6 dB is mid-range for a torso blocking a 2.4 GHz link.
    #
    # delta0 = 0.2 m sets how sharply that decays with the detour length
    # delta = d1 + d2 - d. The geometry is worth doing: for a person offset
    # laterally by r at the midpoint of a link of length d, delta ~ 2r^2/d,
    # so on a 6 m link the 1/e point delta = 0.2 m sits at r = 0.77 m. That
    # is the right scale — comparable to the first Fresnel zone radius
    # (sqrt(lambda*d)/2 = 0.43 m at 6 m) and to the metre-ish extent over
    # which blocking is observed in practice. Note the consequence: because
    # delta falls as 1/d, long links have a *wider* shadowing footprint in
    # metres than short ones, which is the same geometric fact the ellipse
    # weight model encodes with a constant excess threshold.
    shadow_s0_db: float = 6.0
    shadow_delta0_m: float = 0.2

    # -- Breathing ---------------------------------------------------------
    breathing_bpm: float = 15.0
    breathing_amplitude_m: float = 0.005  # 5 mm chest wall displacement
    breathing_phase_rad: float = 0.0

    # -- Receiver ----------------------------------------------------------
    # Per quadrature: the noise is circular complex with variance sigma^2 in
    # each of real and imaginary, so its total power is 2*sigma^2 = 0.72.
    # Against a reference magnitude of 40 (power 1600) that is 33.5 dB SNR
    # on a mid-length link, not the 36.5 dB a one-quadrature count gives.
    # The 3 dB matters: it is the number a reader uses to decide whether a
    # downstream result is noise-limited.
    noise_std: float = 0.6
    # A free-running CFO wraps many times between packets 160 ms apart, so a
    # uniform common phase is the honest model. It leaves amplitude untouched
    # — which is exactly why RESEARCH.md §5a's amplitude-only stance holds,
    # and why anything phase-based downstream must be provably CFO-immune.
    cfo: bool = True

    # -- Motion (used only for the default RandomWaypointMotion) -----------
    speed_mps: tuple[float, float] = (0.3, 1.2)
    pause_s: tuple[float, float] = (0.0, 20.0)
    waypoint_margin_m: float = 0.5

    seed: int = 1234

    def __post_init__(self) -> None:
        if self.n_subcarriers <= 0:
            raise ValueError(f"n_subcarriers must be positive, got {self.n_subcarriers}")
        if self.subcarrier_spacing_hz <= 0.0 or self.carrier_hz <= 0.0:
            raise ValueError("carrier and subcarrier spacing must be positive")
        if self.slot_ms <= 0.0:
            raise ValueError(f"slot_ms must be positive, got {self.slot_ms}")
        if not 0.0 <= self.packet_loss < 1.0:
            raise ValueError(f"packet_loss must be in [0, 1), got {self.packet_loss}")
        if self.reference_magnitude <= 0.0 or self.reference_distance_m <= 0.0:
            raise ValueError("reference magnitude and distance must be positive")
        if not 0.0 <= self.wall_reflection < 1.0:
            raise ValueError(
                f"wall_reflection must be in [0, 1), got {self.wall_reflection}"
            )
        if self.body_rcs_m2 < 0.0:
            raise ValueError(f"body_rcs_m2 must be non-negative, got {self.body_rcs_m2}")
        if self.body_radius_m <= 0.0:
            raise ValueError(
                f"body_radius_m must be positive, got {self.body_radius_m}"
            )
        if self.shadow_s0_db < 0.0:
            raise ValueError(f"shadow_s0_db must be non-negative, got {self.shadow_s0_db}")
        if self.shadow_delta0_m <= 0.0:
            raise ValueError(
                f"shadow_delta0_m must be positive, got {self.shadow_delta0_m}"
            )
        if self.breathing_bpm <= 0.0:
            raise ValueError(f"breathing_bpm must be positive, got {self.breathing_bpm}")
        if self.breathing_amplitude_m < 0.0:
            raise ValueError("breathing_amplitude_m must be non-negative")
        if self.noise_std < 0.0:
            raise ValueError(f"noise_std must be non-negative, got {self.noise_std}")
        # Checked here as well as in RandomWaypointMotion so the failure is
        # reported at config time. A negative margin is otherwise accepted
        # by both and shows up only as ground truth outside the room.
        if self.waypoint_margin_m < 0.0:
            raise ValueError(
                f"waypoint_margin_m must be non-negative, got {self.waypoint_margin_m}"
            )

    @property
    def breathing_hz(self) -> float:
        return self.breathing_bpm / 60.0


# --------------------------------------------------------------------------
# The simulator
# --------------------------------------------------------------------------


class MultiLinkSimulator:
    """A TDMA mesh of single-antenna nodes watching one person.

    Emits `(t, tx_id, rx_id, CsiFrame)` in time order, with `node_id` set to
    the receiver and `sample_rate_hz` set to the *per-link* rate — the rate
    that link's own time series actually arrives at, which is what
    `StaticSuppressor` and `BiquadBandpass` need to see. Setting it to the
    aggregate packet rate instead would silently mistune every downstream
    filter by a factor of n_nodes, so it is worth stating twice.
    """

    def __init__(
        self,
        geometry: Geometry,
        config: SimConfig | None = None,
        motion: _SegmentMotion | None = None,
    ) -> None:
        self.geometry = geometry
        self.config = config if config is not None else SimConfig()
        cfg = self.config

        # Slot order is by node id, so the schedule is a property of the
        # deployment and not of whatever order the node list came in.
        self._tx_order = sorted(n.node_id for n in geometry.nodes)
        self._n = len(self._tx_order)

        self.motion = (
            motion
            if motion is not None
            else RandomWaypointMotion(
                geometry.room,
                seed=f"{cfg.seed}:motion",
                speed_mps=cfg.speed_mps,
                pause_s=cfg.pause_s,
                margin_m=cfg.waypoint_margin_m,
            )
        )

        self._k = cfg.n_subcarriers
        # The 802.11n HT20 layout: K subcarriers at indices -K/2..-1 and
        # 1..K/2, i.e. symmetric about the carrier with a NULL at DC, which
        # is where a real HT20 symbol carries no energy. The obvious
        # alternative — f0 = carrier - (K/2)*df and K contiguous steps — is
        # neither centred (it sits half a subcarrier low) nor null at DC,
        # and this file is the index convention every other module and any
        # future replay of real ESP32 captures inherits.
        self._df = cfg.subcarrier_spacing_hz
        n_low = self._k // 2
        self._sc_index = [i for i in range(-n_low, 0)] + [
            i for i in range(1, self._k - n_low + 1)
        ]
        self._freqs = [cfg.carrier_hz + i * self._df for i in self._sc_index]
        self._f0 = self._freqs[0]
        # Where the uniform index step breaks, for the phasor recursion in
        # `_channel`: after the last negative subcarrier the index jumps by
        # two, across the DC null. -1 for K = 1, which never matches.
        self._dc_gap_at = n_low - 1

        self._scale = cfg.reference_magnitude * cfg.reference_distance_m
        self._body_sigma = math.sqrt(cfg.body_rcs_m2 / (4.0 * math.pi))
        self._body_radius = cfg.body_radius_m
        # One chest direction for the whole run. Its own RNG stream, not the
        # motion one, so that fixing it or sweeping anything else does not
        # reshuffle the walk the configurations are being compared on.
        if cfg.chest_direction_rad is None:
            angle = random.Random(f"{cfg.seed}:chest").uniform(-math.pi, math.pi)
        else:
            angle = cfg.chest_direction_rad
        self.chest_direction_rad = angle
        self._chest = (math.cos(angle), math.sin(angle))
        self._rssi_offset_db = _RSSI_REFERENCE_DBM - 20.0 * math.log10(
            cfg.reference_magnitude
        )

        self._links = geometry.links
        self._link_of = {(l.tx_id, l.rx_id): i for i, l in enumerate(self._links)}
        self._los: list[list[complex]] = []
        self._walls: list[list[complex]] = []
        # The empty-room channel per link, built on first use. It is constant
        # in time, and `measure`'s calibration pass streams nothing else, so
        # caching it turns that whole path into a lookup. Lazy rather than
        # eager because an occupied run never touches it and it would
        # otherwise add a third L x K complex array to the footprint.
        self._static: list[list[complex] | None] = []
        self._build_static()

        self._noise_rng = random.Random(f"{cfg.seed}:noise")
        self._cfo_rng = random.Random(f"{cfg.seed}:cfo")
        self._loss_rng = random.Random(f"{cfg.seed}:loss")

    # -- construction ------------------------------------------------------

    def _build_static(self) -> None:
        """Precompute the time-invariant part of every link's channel.

        Walls do not move, so the four image paths are computed once per link
        and reused for every packet; only the body term and the shadowing
        gain on the direct path change with time. Line of sight is kept
        *separate* from the wall sum precisely because shadowing multiplies
        one and not the other.

        Cost is 5*L*K complex exponentials at construction, one per path:
        15,680 for 8 nodes, measured at 3.9 ms, and 8.9 ms for 36,960 paths
        at 12 nodes (CPython 3.14.7, best of seven). The alternative is
        recomputing five static paths on every one of the ~41,000 packets a
        120 s eight-node run emits, which at a measured 33 us per packet
        would dominate the stream.
        """
        cfg = self.config
        room = self.geometry.room
        w, h = room.width_m, room.height_m
        c = SPEED_OF_LIGHT

        for link in self._links:
            d = link.length_m
            tx_x, tx_y = link.tx
            rx_x, rx_y = link.rx

            los_coeff = self._scale / d
            los = [
                los_coeff * cmath.exp(-2j * math.pi * f * d / c) for f in self._freqs
            ]

            # Image method: mirror the transmitter in each of the four walls.
            # For a convex room with both ends inside, the specular point
            # always lands on the wall segment, so no visibility test is
            # needed. First order only: second-order images are ~R^2 = 0.16
            # of the direct path and are left out deliberately.
            walls = [0j] * self._k
            for ix, iy in (
                (-tx_x, tx_y),
                (2.0 * w - tx_x, tx_y),
                (tx_x, -tx_y),
                (tx_x, 2.0 * h - tx_y),
            ):
                length = math.hypot(ix - rx_x, iy - rx_y)
                if length <= 0.0:
                    continue
                coeff = -cfg.wall_reflection * self._scale / length
                for k, f in enumerate(self._freqs):
                    walls[k] += coeff * cmath.exp(-2j * math.pi * f * length / c)

            self._los.append(los)
            self._walls.append(walls)
            self._static.append(None)

    def _empty_room_channel(self, li: int) -> list[complex]:
        """The link's channel with nobody in the room; cached per link."""
        cached = self._static[li]
        if cached is None:
            los, walls = self._los[li], self._walls[li]
            cached = [los[k] + walls[k] for k in range(self._k)]
            self._static[li] = cached
        return cached

    # -- timing ------------------------------------------------------------

    @property
    def n_nodes(self) -> int:
        return self._n

    @property
    def n_links(self) -> int:
        return len(self._links)

    @property
    def slot_s(self) -> float:
        return self.config.slot_ms / 1000.0

    @property
    def superframe_s(self) -> float:
        return self._n * self.slot_s

    @property
    def per_link_rate_hz(self) -> float:
        """1 / (n_nodes * slot). The number the whole evaluation turns on."""
        return 1.0 / self.superframe_s

    @property
    def aggregate_packet_rate_hz(self) -> float:
        """Packets per second across all links, before loss.

        Rises with n_nodes while the per-link rate falls — the two halves of
        the trade. More nodes means more links to invert from and less of a
        time series on each one.
        """
        return self.n_links / self.superframe_s

    # -- physics, exposed for testing and diagnostics ----------------------

    def shadowing_db(self, excess_m: float) -> float:
        """S(delta) = S0 * exp(-delta / delta0), the direct-path loss in dB.

        `excess_m` is d1 + d2 - d, so zero means the body is standing exactly
        on the line of sight and the loss is the full S0. Tiny negative
        values from floating-point cancellation are clamped rather than
        allowed to amplify the signal.
        """
        cfg = self.config
        if excess_m <= 0.0:
            return cfg.shadow_s0_db
        return cfg.shadow_s0_db * math.exp(-excess_m / cfg.shadow_delta0_m)

    def link_shadowing_db(self, tx_id: int, rx_id: int, x: float, y: float) -> float:
        """Direct-path loss on one link with the body at (x, y)."""
        link = self._links[self._link_of[(tx_id, rx_id)]]
        return self.shadowing_db(link.excess_path_m(x, y))

    def noiseless_channel(
        self,
        tx_id: int,
        rx_id: int,
        x: float,
        y: float,
        displacement_m: float = 0.0,
    ) -> list[complex]:
        """The per-subcarrier channel with no noise and no CFO.

        Exactly what `stream` would emit for this link and body position
        before the receiver is applied — so with `config.occupied` False it
        is the empty room, and (x, y) is ignored. The deterministic core of
        the model, exposed so a test can assert on the physics instead of on
        a noisy average, and so `measure` can check its calibration against
        the forward model rather than against another estimate.
        """
        li = self._link_of[(tx_id, rx_id)]
        if not self.config.occupied:
            # A copy, because the cached empty-room channel is live state.
            return list(self._empty_room_channel(li))
        return self._channel(li, x, y, displacement_m)

    def _channel(
        self, li: int, px: float, py: float, disp: float
    ) -> list[complex]:
        link = self._links[li]
        los = self._los[li]
        walls = self._walls[li]
        tx_x, tx_y = link.tx
        rx_x, rx_y = link.rx

        # NaN or inf here would travel the whole chain — the comparisons
        # below are all False for NaN, so it would pass through the
        # geometry, the channel and `_rssi_dbm` and only surface as a
        # ValueError from int(). Reject it at the one door body coordinates
        # come through.
        if not (-math.inf < px < math.inf and -math.inf < py < math.inf):
            raise ValueError(f"body position must be finite, got ({px}, {py})")

        d1 = math.hypot(tx_x - px, tx_y - py)
        d2 = math.hypot(px - rx_x, py - rx_y)

        # TRUE distances, not softened ones: the shadowing excess is
        # geometry, and computing it from a clamped d1/d2 makes the applied
        # loss disagree with the public `link_shadowing_db` inside the clamp
        # radius, and makes it fall as the body closes the last few cm onto
        # a node — the opposite of what the law says.
        gain = 10.0 ** (-self.shadowing_db(d1 + d2 - link.length_m) / 20.0)

        # Unit vectors from the body to each end. Their sum carries both
        # pieces of scattering geometry at once:
        #
        #   |a + b| = 2*|cos(beta/2)| for the tx-body-rx angle beta, and
        #   d(d1 + d2)/d(displacement along u) = -(a + b) . u.
        #
        # Taking the vector sum directly rather than going through
        # sqrt(2*(1 + cos beta)) also avoids the cancellation in 1 + cos
        # beta exactly where it matters, near beta = pi.
        #
        # The chest displaces along ONE direction for the whole run, not
        # along each link's own bisector: a torso that moved along every
        # bisector at once would hand every link its maximum possible
        # sensitivity. Forward scatter is still null whatever the direction,
        # because a + b = 0 identically on the line of sight.
        if d1 > 0.0 and d2 > 0.0:
            sx = (tx_x - px) / d1 + (rx_x - px) / d2
            sy = (tx_y - py) / d1 + (rx_y - py) / d2
        else:
            # Body exactly on a node: the scattering angle is undefined and
            # the forward-scatter limit is the defensible reading of it.
            sx = sy = 0.0
        ux, uy = self._chest
        path = d1 + d2 - (sx * ux + sy * uy) * disp

        # Forward-scatter taper, |cos(beta/2)|, and a pi flip.
        #
        # The flip is extinction: a forward-scattered field opposes the
        # incident one, which is what makes a shadow. The taper is what
        # keeps the two body terms from being the same physics twice —
        # where the body is in the way, the shadowing law has already
        # accounted for it, so the explicit scattered path must fade out.
        # Without the taper this term is exactly in phase with the direct
        # path on the line of sight (d1 + d2 == d there, at every
        # subcarrier) and fills in the shadow it is supposed to accompany.
        #
        # The softening of 1/(d1*d2) is separate and applies to the
        # amplitude only: a torso has a radius, so d -> hypot(d, r) rather
        # than a clamp that would rewrite the geometry above.
        r = self._body_radius
        d1e = math.hypot(d1, r)
        d2e = math.hypot(d2, r)
        body_coeff = -0.5 * math.hypot(sx, sy) * self._scale * self._body_sigma / (
            d1e * d2e
        )

        # exp(-j*2*pi*f_k*path/c) over the uniform part of the frequency
        # grid is a geometric sequence: one exponential and K complex
        # multiplies instead of K exponentials, with one double step across
        # the DC null. Unit-modulus accumulation over 56 terms drifts by
        # ~56 ulp, which is ~1e-14 of the phase — irrelevant beside a noise
        # floor at 1e-2.
        step = cmath.exp(-2j * math.pi * self._df * path / SPEED_OF_LIGHT)
        step_dc = step * step
        phasor = cmath.exp(-2j * math.pi * self._f0 * path / SPEED_OF_LIGHT)
        dc_gap_at = self._dc_gap_at

        out: list[complex] = []
        for k in range(self._k):
            out.append(gain * los[k] + walls[k] + body_coeff * phasor)
            phasor *= step_dc if k == dc_gap_at else step
        return out

    # -- ground truth ------------------------------------------------------

    def truth_at(self, t: float) -> GroundTruth:
        """Where the person is at `t`. Random access; cheap.

        Meaningless when `config.occupied` is False: the motion model still
        runs, but nothing is rendered into the channel.
        """
        return self.motion.at(t)

    def ground_truth(self, duration_s: float, rate_hz: float = 10.0) -> Iterator[GroundTruth]:
        """The parallel ground-truth stream, on its own uniform grid."""
        if rate_hz <= 0.0:
            raise ValueError(f"rate_hz must be positive, got {rate_hz}")
        dt = 1.0 / rate_hz
        i = 0
        while True:
            t = i * dt
            if t >= duration_s:
                return
            yield self.motion.at(t)
            i += 1

    # -- the stream --------------------------------------------------------

    def stream(self, duration_s: float) -> Iterator[tuple[float, int, int, CsiFrame]]:
        """Yield `(t, tx_id, rx_id, frame)` for `duration_s` of airtime.

        Superframe m, slot i is at t = (m*n + i)*slot, and every node except
        the transmitter receives that one broadcast — so all n-1 frames in a
        slot share a timestamp and a body position, and differ only in
        geometry, noise and CFO. That shared instant is physically right and
        also the main saving: the person's position and chest displacement
        are evaluated once per slot, not once per link.

        Calling this twice restarts the clock at t = 0 but *not* the RNG
        streams, so the second run is a different noise realisation of the
        same walk. Call `reset()` first for a bit-identical repeat.
        """
        cfg = self.config
        noise_draw = self._noise_rng.random
        uniform = self._cfo_rng.uniform
        loss_draw = self._loss_rng.random
        sigma = cfg.noise_std
        loss = cfg.packet_loss
        rate = self.per_link_rate_hz
        slot = self.slot_s
        n = self._n
        kk = self._k
        two_pi_fb = 2.0 * math.pi * cfg.breathing_hz
        occupied = cfg.occupied

        m = 0
        while True:
            for i, tx_id in enumerate(self._tx_order):
                t = (m * n + i) * slot
                if t >= duration_s:
                    return

                if occupied:
                    truth = self.motion.at(t)
                    px, py = truth.x, truth.y
                    # The chest keeps moving while the person walks; it is
                    # just swamped by the translation. Modelling it
                    # continuously avoids a discontinuity at every pause
                    # boundary that a "breathe only when still" switch would
                    # introduce, and that a downstream detector would see as
                    # a spurious event.
                    disp = cfg.breathing_amplitude_m * math.sin(
                        two_pi_fb * t + cfg.breathing_phase_rad
                    )

                for rx_id in self._tx_order:
                    if rx_id == tx_id:
                        continue

                    dropped = loss_draw() < loss
                    # Noise is drawn whether or not the packet survives, so
                    # that a sweep over packet_loss leaves the surviving
                    # packets bit-identical. Costs 2K wasted draws per lost
                    # packet; buys a controlled experiment.
                    #
                    # Box-Muller in polar form: a Rayleigh magnitude with a
                    # uniform phase *is* a circular complex Gaussian, with
                    # real and imaginary parts i.i.d. N(0, sigma^2). Two
                    # uniform draws instead of two `gauss` calls, which is
                    # 3x faster here and makes the circular symmetry
                    # explicit rather than incidental.
                    noise = [
                        cmath.rect(
                            sigma * math.sqrt(-2.0 * math.log(1.0 - noise_draw())),
                            _TWO_PI * noise_draw(),
                        )
                        for _ in range(kk)
                    ]
                    phi = uniform(-math.pi, math.pi) if cfg.cfo else 0.0
                    if dropped:
                        continue

                    li = self._link_of[(tx_id, rx_id)]
                    if occupied:
                        chan = self._channel(li, px, py, disp)
                    else:
                        chan = self._empty_room_channel(li)

                    rot = cmath.rect(1.0, phi)
                    amplitudes: list[float] = []
                    phases: list[float] = []
                    power = 0.0
                    for k in range(kk):
                        z = chan[k] * rot + noise[k]
                        a = abs(z)
                        amplitudes.append(a)
                        phases.append(cmath.phase(z))
                        power += a * a

                    yield (
                        t,
                        tx_id,
                        rx_id,
                        CsiFrame(
                            amplitudes=amplitudes,
                            phases=phases,
                            sample_index=m,
                            sample_rate_hz=rate,
                            rssi_dbm=self._rssi_dbm(power / kk),
                            node_id=rx_id,
                        ),
                    )
            m += 1

    def _rssi_dbm(self, mean_power: float) -> int:
        """Mean subcarrier power as the integer dBm a radio would report.

        Quantised to 1 dB because that is what the hardware gives. It matters:
        1 dB is a 12% amplitude step, large enough that an RSS-only
        measurement of a distant link can be entirely quantisation noise —
        which is a large part of why VRTI works on *variance* over a window
        rather than on instantaneous RSS.

        Written defensively because it is the one place in the stream that
        converts to int, so every non-finite value produced anywhere
        upstream lands here as an OverflowError or a ValueError from
        `round()` rather than as a number a caller can see is wrong. NaN
        fails `> 0.0` and takes the floor branch; inf is clamped before the
        conversion, not after — `round()` raises on both, so clamping its
        result cannot help.
        """
        if not mean_power > 0.0:  # catches 0, negatives and NaN
            return -100
        if mean_power == math.inf:
            return 0
        dbm = 10.0 * math.log10(mean_power) + self._rssi_offset_db
        return int(round(max(-100.0, min(0.0, dbm))))

    def reset(self) -> None:
        """Rewind to t = 0 with the original seed and trajectory."""
        cfg = self.config
        self._noise_rng = random.Random(f"{cfg.seed}:noise")
        self._cfo_rng = random.Random(f"{cfg.seed}:cfo")
        self._loss_rng = random.Random(f"{cfg.seed}:loss")
        self.motion.reset()

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"MultiLinkSimulator(nodes={self._n}, links={self.n_links}, "
            f"slot={self.config.slot_ms:g} ms, "
            f"per_link={self.per_link_rate_hz:.3g} Hz, "
            f"seed={self.config.seed})"
        )
