"""Per-link disturbance measurement: the `y` vector that RTI inverts.

Radio tomographic imaging solves `y = W x` for a pixel image `x`, where
`W` is the geometric forward model (`geometry.py`) and `y` is one scalar
per link saying "how much was this link disturbed". This module produces
`y`, and it is the only place in the localization stack where radio meets
arithmetic.

**Why variance and not mean.** The original RTI (Wilson & Patwari, IEEE
TMC 2010) imaged the *shift* in mean RSS caused by a body shadowing a
link. That works, but it needs an accurate empty-room mean per link and
drifts with temperature, furniture and anything else that moves once and
stays moved. VRTI (Wilson & Patwari, IEEE TMC 2011) images the *variance*
of each link over a short window instead. Two consequences, and both are
the reason this module exists in the form it does:

  * Variance responds to *motion*, not occupancy. A person who stops
    moving disappears. That is a limitation to be surfaced honestly
    (`track.py` abstains), not papered over.
  * Variance survives walls. A multipath-rich non-line-of-sight link
    still has its fading statistics stirred by a moving body even when
    the mean is dominated by reflections. This is why VRTI is the
    through-wall variant.

**Three link signals** (`RssiSignal`, added for Room Sim input in Phase 4,
is the datagram's RSSI; spec/locate.md §3). **The first two, because the
evaluation has to compare them.** The
`LinkSignal` strategy picks what scalar gets fed to the variance:

  (a) `TotalPowerSignal` — total received power in dB, summed across
      subcarriers. The RSS-like baseline the RTI papers actually used.
  (b) `StaticSuppressedSignal` — `StaticSuppressor` residuals fused
      across subcarriers. The CSI-specific alternative.

They are a strategy pair rather than two code paths because the only
honest way to answer "is CSI worth the trouble over RSS" is to run both
through identical windowing, identical calibration and identical
numerics, and compare. Anything that differs between them other than the
scalar extraction contaminates that comparison.

**Why (a) is in dB.** Variance of *linear* power scales with the square
of mean power, so a 2 m link and an 8 m link — about 12 dB apart in
line-of-sight power — would enter the least-squares fit with wildly
different weights for no reason except their length. `W` already carries
a `1/sqrt(d_l)` length normalisation and would be double-counting. dB
makes the measurement a *fractional* power change, which is comparable
across links, and it is what the papers' RSS-in-dBm measurements were.

**Why averaging subcarriers is noise averaging, not diversity** (and so
why (b) is not obviously better than (a)). RESEARCH.md §5a: on an ESP32's
17.5 MHz occupied bandwidth, escaping a Fresnel blind spot by subcarrier
selection needs the body-reflected path to exceed line-of-sight by about
4.4 m, which does not happen for anyone near the direct path in a normal
room. The whole band therefore sits at essentially one Fresnel phase: the
56 subcarriers move *together*. Signed averaging across them is then the
right operation — it keeps the coherent body signal and averages down the
independent per-subcarrier noise — but it buys no blind-spot diversity,
and a link in a null stays in a null. On a 160 MHz NIC the same signed
average would partially cancel and this code would need revisiting.

**Why Welford is recomputed rather than updated.** `RunningVariance` is
Welford's algorithm, which is incremental and has no exact removal: the
"subtract a sample" form reintroduces precisely the catastrophic
cancellation Welford exists to avoid, and can return a negative `m2`. A
sliding window needs removal. So the window is kept as a deque of raw
samples and Welford is re-run over its contents whenever the variance is
actually asked for, with a per-link dirty flag so it happens once per
`vector()` call and not once per sample. At 56 links and ~13 samples per
window that is ~730 updates per image. MEASURED on CPython, 8 nodes and
56 links with every link dirty — the state the pipeline is always in,
because a superframe of `update()` precedes every image — that is
0.11-0.14 ms per `vector()`, against 0.010 ms when every link is clean
and the cache answers. Correctness is bought for a tenth of a
millisecond per image.

**Why the clock is per link.** Each link keeps its own newest timestamp
and prunes against it, and a separate monotone *progress* clock — the
largest forward step any link has made — ages out links that have gone
quiet. The obvious cheaper design, one global "newest timestamp seen"
used as every link's pruning origin, is wrong in a way that only shows
up on a real mesh: N ESP32s each stamp their own receptions, so a node
whose clock is half a second off has every one of its links silently
truncated, and a node a window off has them blanked outright. Worse,
with one global clock a *single* late packet retires samples on every
other link at once. One late packet must cost at most one link; a
persistent per-node offset must cost nothing at all. See `_prune`.

**Why the baseline is subtracted and not divided.** See
`CalibrationMode`.
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from enum import Enum

from ..preprocess import StaticSuppressor, fuse
from ..types import CsiFrame, RunningVariance
from .geometry import Geometry

__all__ = [
    "CalibrationMode",
    "LinkSignal",
    "LinkSignalState",
    "TotalPowerSignal",
    "StaticSuppressedSignal",
    "RssiSignal",
    "LinkMeasurement",
]


class CalibrationMode(str, Enum):
    """What to do with the empty-room baseline variance.

    SUBTRACT (`y = var - baseline`) is the default and is the
    model-correct choice. Variance is additive over independent
    contributions, so an empty-room noise floor and a body disturbance
    add: `var = sigma_noise^2 + delta`. Subtracting recovers `delta`,
    which is linear in the disturbance, which is what `y = W x` assumes.
    Nothing else preserves that linearity.

    NORMALIZE (`y = var / baseline - 1`) divides out the noise floor
    instead, giving fractional excess variance. The `- 1` is deliberate:
    it keeps "undisturbed" at 0, matching SUBTRACT's convention and the
    0.0 that an unobserved link reports, so a dead link never looks like
    *negative* disturbance.

    NORMALIZE is not model-correct — it rescales row `l` of the
    measurement without rescaling row `l` of `W`, which is a different
    forward model. It is offered anyway because it buys something real:
    the dual-form inversion regularizes with a single scalar `alpha`
    across all links, which assumes every link's measurement noise has
    the same magnitude. Link noise floors in a real room span a wide
    range, so that assumption is badly violated, and the noisiest links
    then dominate the fit. Normalizing is an approximate whitening that
    makes the assumption roughly true.

    So: SUBTRACT if you trust `alpha` to be tuned for your worst link,
    NORMALIZE if the link noise floors are very unequal. Measure both.
    """

    SUBTRACT = "subtract"
    NORMALIZE = "normalize"


def _median(values: list[float]) -> float:
    """Median of a non-empty list.

    Hand-rolled rather than `statistics.median` only to keep the C port
    honest — this file is meant to be transliterable. It is used once per
    calibration, never in the sample path.
    """
    if not values:
        raise ValueError("median of an empty list")
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


class LinkSignalState(ABC):
    """Per-link running state for one `LinkSignal` strategy.

    One instance per link, because any useful signal extraction is
    stateful (a static-suppression baseline, at minimum) and link state
    must never leak between links — two links sharing a suppressor would
    cross-contaminate their baselines and produce a disturbance on every
    link whenever any one of them changed.
    """

    __slots__ = ()

    @abstractmethod
    def push(self, frame: CsiFrame) -> float | None:
        """Reduce one frame to a scalar, or None to drop the sample.

        Returning None rather than a sentinel number matters: a dropped
        frame must leave *no* trace in the window. A sentinel such as 0.0
        or -120 dB would be a large outlier in a 12-sample variance and
        would light that link up as a false disturbance.
        """

    def reset(self) -> None:
        """Forget all state. Default is a no-op for stateless signals."""


class LinkSignal(ABC):
    """Strategy: how to turn a link's CSI frames into a scalar time series.

    A factory rather than a plain function because the extraction is
    stateful per link; `LinkMeasurement` calls `new_state` once per link
    and never shares the result.
    """

    __slots__ = ()

    @property
    @abstractmethod
    def name(self) -> str:
        """Short label, used when reporting which signal produced a result."""

    @abstractmethod
    def new_state(self, sample_rate_hz: float) -> LinkSignalState:
        """Fresh per-link state. `sample_rate_hz` is the PER-LINK rate."""


def _frame_power(frame: CsiFrame) -> float | None:
    """MEAN received power per subcarrier, or None if the frame is unusable.

    The single validity gate both strategies share, so that "which frames
    count" cannot drift apart between them and contaminate the comparison
    they exist to support.

    **Mean, not total.** The obvious implementation sums `amp^2` over
    whatever subcarriers the frame carries, and that makes the dB value a
    function of the subcarrier COUNT as well as of the channel: 52
    subcarriers instead of 56 reads 10*log10(52/56) = -0.32 dB lower for
    pure bookkeeping reasons. ESP-IDF hands back different CSI lengths
    for LLTF-only and LLTF+HT-LTF receptions (`sources/recorder.py`
    accumulates the distinct counts it sees per stream, precisely because
    they vary), and a 0.32 dB step inside a variance window whose genuine
    per-sample spread is an order of magnitude smaller is a large false
    disturbance. MEASURED on the test file's two-node channel at 25 dB
    SNR: a mid-stream switch from 56 to 52 subcarriers peaked at 19.2
    null sigma with the sum and 0.9 with the mean; 56 to 48, 81.2 against
    1.3. Dividing by the count shifts every link by one constant dB
    offset, which calibration absorbs, and leaves short frames usable
    instead of discarding them.

    A frame is unusable if it carries no subcarriers, if any amplitude is
    non-finite, or if the total power is zero or overflows to infinity.
    The infinity check is not theoretical: every amplitude in
    `[1e200] * 56` is finite while their squares sum to `inf`, which then
    makes Welford's `m2` NaN, and NaN slips through the `clamp_negative`
    comparison and — if it lands during calibration — poisons that link's
    baseline permanently.

    One bad subcarrier condemns the whole frame rather than being
    skipped, because skipping changes which subcarriers the average
    covers while pretending nothing happened.
    """
    n = frame.n_subcarriers
    if n == 0:
        return None
    total = 0.0
    for amp in frame.amplitudes:
        if not math.isfinite(amp):
            return None
        total += amp * amp
    if not (total > 0.0 and math.isfinite(total)):
        return None
    return total / n


class _TotalPowerState(LinkSignalState):
    """Stateless: mean received power per subcarrier, in dB."""

    __slots__ = ()

    def push(self, frame: CsiFrame) -> float | None:
        mean_power = _frame_power(frame)
        if mean_power is None:
            return None
        value = 10.0 * math.log10(mean_power)
        # The same output guard `_StaticSuppressedState` has. Nothing
        # finite and positive can make `log10` non-finite, so this is belt
        # and braces -- but the two strategies are meant to share one
        # validity contract, and a gate that exists on one side only is
        # how that contract quietly stops being true.
        return value if math.isfinite(value) else None


class TotalPowerSignal(LinkSignal):
    """(a) Total received power in dB -- the RSS-like RTI baseline.

    This is the signal with published RTI accuracy behind it. It needs no
    warm-up, no state and no tuning, which makes it the right default and
    the right thing for the CSI-specific alternative to have to beat.
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        return "total_power_db"

    def new_state(self, sample_rate_hz: float) -> LinkSignalState:
        return _TotalPowerState()


class _StaticSuppressedState(LinkSignalState):
    """Static-suppressed amplitude residual, fused across subcarriers."""

    __slots__ = ("_sample_rate_hz", "_corner_hz", "_suppressor")

    def __init__(self, sample_rate_hz: float, corner_hz: float) -> None:
        self._sample_rate_hz = sample_rate_hz
        self._corner_hz = corner_hz
        # Built lazily on the first frame: the subcarrier count comes from
        # the data rather than from a constructor argument that could
        # disagree with it.
        self._suppressor: StaticSuppressor | None = None

    def push(self, frame: CsiFrame) -> float | None:
        if frame.n_subcarriers == 0:
            return None

        # Validate BEFORE the suppressor sees the frame. This gate is not
        # redundant with StaticSuppressor's own non-finite guard, and the
        # difference matters: that guard zeroes the offending subcarrier's
        # residual but still lets every *finite* subcarrier of a corrupt
        # frame update its EMA baseline. An all-zero frame is the bad case
        # -- every amplitude is finite, so it sails through, produces a
        # residual of minus the whole baseline on all 56 subcarriers, and
        # drags the baselines toward zero for the next several seconds.
        # One corrupt packet would then paint a phantom disturbance on that
        # link for most of a minute.
        if _frame_power(frame) is None:
            return None

        if self._suppressor is None:
            self._suppressor = StaticSuppressor(
                frame.n_subcarriers, self._sample_rate_hz, self._corner_hz
            )
            # Seed the EMA baselines and discard. StaticSuppressor returns
            # an all-zero residual for the first frame on each subcarrier
            # by construction; feeding that zero into the window would be
            # a fabricated sample, not a measured one.
            self._suppressor.process(frame)
            return None

        residuals, weights = self._suppressor.process(frame)
        value = fuse(residuals, weights)
        return value if math.isfinite(value) else None

    def reset(self) -> None:
        self._suppressor = None


class StaticSuppressedSignal(LinkSignal):
    """(b) Static-suppressed CSI amplitude, fused across subcarriers.

    `StaticSuppressor` removes each subcarrier's slowly varying baseline;
    `fuse` collapses the residuals to one number weighted by each
    subcarrier's running standard deviation, which is a cheap matched
    filter along the subcarrier axis.

    It did not beat (a). MEASURED on the synthetic channel in
    `tests/test_localize_measure.py`, as detectability `d'` against the
    empty-room null, over four independent seed sets:

      * gross motion (walking): (a) leads by 1.27-1.60x at every SNR
        tested (30/20/10 dB). Static suppression plus 56-subcarrier
        fusion is a small net *loss*, not a gain.
      * weak disturbance (a still, breathing person): (a) leads by 4-5x
        at 20 dB SNR, and by 10 dB (b) has collapsed to d' = 0.06 -- it
        has stopped detecting the person in any useful sense.

    Three things explain that, and all three are worth knowing before
    reaching for this strategy again.

    First, the fusion weighting buys almost nothing here. Replacing
    `fuse` with a plain unweighted mean across the same `StaticSuppressor`
    residuals costs 2.0-4.7% of `d'` -- it helps, but marginally, and
    nowhere near enough to matter against (a)'s 1.3-1.6x lead.
    `test_fuse_weighting_is_nearly_irrelevant_on_this_bandwidth` is that
    control, so the number is rerunnable rather than remembered. It
    follows from RESEARCH.md §5a: at 17.5 MHz occupied bandwidth every
    subcarrier carries the same signal at nearly the same Fresnel phase
    and a similar noise level, so the variance-derived weights are
    near-uniform by construction and have very little to select between.
    `fuse`'s weighting earns its keep in the breathing pipeline, where
    subcarriers genuinely differ; it has almost no purchase here.

    Second, the weights are learned online, so the map from frames to
    scalar is non-stationary -- uniform at first, variance-weighted
    later. Calibration has to run long enough for them to settle, or the
    baseline is measured through a different filter than the live run.

    Third, and the real asymmetry: (b) is a LINEAR amplitude residual, so
    its variance scales with the link's absolute signal level, while (a)
    is in dB and so is scale-free. A body shadowing a link drops its
    amplitude, which drags (b)'s variance down at the same time as the
    body's motion pushes it up. See `CalibrationMode` and the dB note in
    the module docstring --- this is the same effect that makes a linear
    power signal invert sign outright.

    Kept, despite losing, because the comparison is a deliverable and
    because the result is bandwidth-dependent: on a 160 MHz NIC the
    subcarriers decorrelate, the weighting starts to mean something, and
    this strategy may well win. Re-measure before assuming either way.
    """

    __slots__ = ("corner_hz",)

    def __init__(self, corner_hz: float = 0.03) -> None:
        self.corner_hz = corner_hz

    @property
    def name(self) -> str:
        return "static_suppressed_amplitude"

    def new_state(self, sample_rate_hz: float) -> LinkSignalState:
        return _StaticSuppressedState(sample_rate_hz, self.corner_hz)


class _RssiState(LinkSignalState):
    """Stateless: the frame's reported RSSI, in dBm (spec/locate.md §3)."""

    __slots__ = ()

    def push(self, frame: CsiFrame) -> float | None:
        # The shared validity gate first, so that "which frames count" is
        # the same for every strategy: an empty or non-finite frame is
        # dropped here exactly as `TotalPowerSignal` drops it.
        if _frame_power(frame) is None:
            return None
        return float(frame.rssi_dbm)


class RssiSignal(LinkSignal):
    """(c) The datagram's RSSI in dBm -- the AGC-free link signal.

    Room Sim normalises each packet's CSI to a fixed level with a
    random-step AGC (spec/roomsim.md §12.3), as the ESP32 does, so a
    per-packet total-power signal mostly sees the AGC's steps, not the
    bodies. The RSSI the receiver reports is computed before that gain and
    is untouched by it, which makes it the default signal for Room Sim
    input (spec/locate.md decision 2).

    Stateless. The per-frame value is `float(frame.rssi_dbm)` for every
    frame `_frame_power` accepts: integer dBm, so its variance over a
    window is a narrow signal (spec/locate.md §5, §13).
    """

    __slots__ = ()

    @property
    def name(self) -> str:
        return "rssi_dbm"

    def new_state(self, sample_rate_hz: float) -> LinkSignalState:
        return _RssiState()


class LinkMeasurement:
    """Sliding-window per-link variance change, in canonical link order.

    Feed it `(t, tx_id, rx_id, frame)` as the TDMA stream delivers them;
    ask it for `vector()` whenever the inversion wants an image. The
    returned list is always `len(geometry.links)` long and indexed by
    `geometry.link_index(tx, rx)`.

    **Windowing is by time, not by sample count.** The physical quantity
    is "how much did this link move in the last second", and under packet
    loss a count-based window silently stretches: a link losing half its
    packets would average over two seconds while its neighbours average
    over one. Because RTI compares links against each other, that is not
    extra smoothing but a *spatial* bias — the lossy link's response lags
    and blurs relative to the healthy ones, dragging the image toward
    where the person used to be. A time window keeps every link aligned
    and lets the sample count vary instead, which costs only estimator
    noise on the links that deserve it.

    **Links with too little data report 0.0.** Zero is the correct prior:
    `y = 0` is the neutral element of the data term, so an unobserved
    link neither attracts nor repels the image. Be aware of the
    conflation this creates — a dead link and a genuinely quiet link are
    indistinguishable in the output, so a dead link effectively votes
    "nobody is on my path". `ready_mask()` exists so the inversion can
    tell the two apart and drop the rows if it wants to.

    **Unobserved links decay, without one clock ruling them all.** Each
    link prunes against its own newest timestamp, so a node whose clock
    disagrees with its neighbours' keeps a correct window of its own
    data. Liveness is carried separately, by a monotone progress clock
    that advances with the largest forward step any link has made: a link
    that stops reporting is aged out by everyone else's progress and
    fades to 0.0 after one window rather than holding a stale
    disturbance. The two together give the property that matters on a
    mesh of free-running nodes — one link's timing trouble stays on that
    link. `clock_restarts`, `stale_samples` and `window_overflows` count
    the events, because a mesh that is quietly degrading this way is
    otherwise indistinguishable from a quiet room.

    **Nothing happens between packets unless you say so.** The clock only
    advances when a sample arrives, so if the whole mesh stops delivering
    — an AP reboot, a crashed collector, the end of a file replay —
    `vector()` keeps returning the last image it computed. Pass `t` to
    `vector()`, `sample_counts()` or `ready_mask()` to say what time it
    is now; the windows then age out properly and a tracker stops being
    told about a person who left.
    """

    def __init__(
        self,
        geometry: Geometry,
        window_s: float = 1.0,
        sample_rate_hz: float = 12.5,
        signal: LinkSignal | None = None,
        calibration_mode: CalibrationMode | str = CalibrationMode.SUBTRACT,
        min_samples: int | None = None,
        clamp_negative: bool = False,
    ) -> None:
        """
        `window_s` is the VRTI window, 0.5-2 s in the literature; 1 s by
        default. Shorter follows motion better and estimates variance
        worse, and that trade is the whole tuning story of this module.

        `sample_rate_hz` is the PER-LINK rate, which under TDMA is
        `1 / (n_nodes * slot_s)` and not the aggregate packet rate. The
        default 12.5 Hz is 8 nodes at a 10 ms slot. It must match the
        deployment: it sizes `min_samples` and sets the static-suppression
        corner, and getting it wrong silently mistunes both.

        `sample_rate_hz` also sets a hard cap on window length,
        `max(8, ceil(4 * window_s * sample_rate_hz))` samples, which is a
        memory backstop against a caller whose clock never advances. It
        is four windows of slack, so it does not trigger on a link that
        is merely healthy — but a rate under-declared by 4x or more turns
        it into a silent shortening of `window_s` itself. Declaring 12.5
        Hz and feeding 100 Hz gives a window holding 0.50 s, not 1.00 s.
        `window_overflows` counts every sample the cap evicts, so the
        condition is detectable rather than merely suffered.

        `clamp_negative` defaults to False, which surprises people. A
        12-sample variance estimate has about 43% relative spread, so in
        an empty room roughly half the links land *below* their baseline
        and produce negative `y`. Clamping those to zero turns zero-mean
        estimator noise into a rectified positive floor on every link at
        once, which the inversion reads as a faint disturbance filling the
        whole room. Least squares wants zero-mean noise, so the honest
        default is to let the negatives through.
        """
        if window_s <= 0.0:
            raise ValueError(f"window_s must be positive, got {window_s}")
        if sample_rate_hz <= 0.0:
            raise ValueError(f"sample_rate_hz must be positive, got {sample_rate_hz}")

        self.geometry = geometry
        self.window_s = window_s
        self.sample_rate_hz = sample_rate_hz
        self.signal: LinkSignal = TotalPowerSignal() if signal is None else signal
        self.calibration_mode = CalibrationMode(calibration_mode)
        self.clamp_negative = clamp_negative

        expected = window_s * sample_rate_hz
        if min_samples is None:
            # Half the nominal count: a link that lost more than half its
            # packets in the window abstains rather than reporting a
            # variance from three samples. Never below 3, because a
            # two-sample variance is (x1-x2)^2/2 and has 141% relative
            # spread -- it is noise with a plausible shape.
            self.min_samples = max(3, int(round(0.5 * expected)))
        else:
            self.min_samples = min_samples
        if self.min_samples < 2:
            raise ValueError(f"min_samples must be at least 2, got {self.min_samples}")

        n_links = geometry.n_links
        self._windows: list[deque[tuple[float, float]]] = [
            deque() for _ in range(n_links)
        ]
        self._states: list[LinkSignalState] = [
            self.signal.new_state(sample_rate_hz) for _ in range(n_links)
        ]
        self._cached_var = [0.0] * n_links
        self._dirty = [True] * n_links

        self._baseline = [0.0] * n_links
        self._baseline_std = [0.0] * n_links
        self._baseline_ok = [False] * n_links
        self._calibrated = False

        # Per-link clock state. `_t_last` is that link's own newest
        # timestamp and is the origin it prunes against; `_progress_at` is
        # the mesh progress clock as of its last sample, which is what
        # ages it out when it goes quiet. `_pending_restart` holds the
        # first sample of a suspected stream restart until a second one
        # confirms it -- see `_ingest`.
        self._t_last: list[float | None] = [None] * n_links
        self._progress_at: list[float] = [0.0] * n_links
        self._pending_restart: list[float | None] = [None] * n_links
        self._progress = 0.0
        self._t_stream: float | None = None

        # Diagnostics. A mesh whose nodes are drifting apart, or a
        # collector dropping packets into the wrong order, degrades this
        # module's output without raising anything; these counters are the
        # only way to tell that apart from a quiet room.
        self.clock_restarts = 0
        self.stale_samples = 0
        self.window_overflows = 0

        # Hard cap on window length, so a caller whose clock never
        # advances (a replay with a frozen clock, a unit test) grows
        # memory instead of silently never pruning. Four windows of slack
        # never triggers on a correctly declared rate; `window_overflows`
        # says when it did.
        self._max_window = max(8, int(math.ceil(4.0 * expected)))

    # ---- ingest -------------------------------------------------------

    def update(self, t: float, tx_id: int, rx_id: int, frame: CsiFrame) -> None:
        """Absorb one packet. Unknown links raise rather than being dropped.

        A `(tx, rx)` pair absent from the geometry means the caller and
        this object disagree about the deployment, which is a bug that
        would otherwise show up as a quietly incomplete image. The
        `KeyError` from `geometry.link_index` is deliberately left to
        propagate.

        Timestamps are assumed non-decreasing per link, as TDMA delivers
        them, and are NOT assumed comparable *between* links: nodes free-
        run their own clocks. What a late or backwards timestamp costs is
        described on `_ingest`, and the short version is that it costs
        that link and no other.
        """
        index = self.geometry.link_index(tx_id, rx_id)

        value = self._states[index].push(frame)
        if value is None:
            return

        self._ingest(index, t, value)

    def _ingest(self, index: int, t: float, value: float) -> bool:
        """Place one extracted scalar into its link's window.

        Shared by `update` and `calibrate_from` so that the two cannot
        drift apart — a calibration path with weaker clock handling than
        the live path learns its baseline through a different filter than
        the one the live run uses, which is worse than either path being
        wrong on its own. Returns whether the sample entered the window.

        Three timestamp cases, and the split between them is the whole
        point:

          * **Forward, or out of order by less than one window.** Kept,
            and inserted in time order rather than appended, so the deque
            stays sorted and pruning (which only ever inspects the head)
            retires it at its true age. Appending blindly lets a sample
            that arrived 0.9 s late survive to nearly twice `window_s`,
            which is exactly the stretched-window spatial bias the time
            window exists to prevent.
          * **More than one window behind this link's own clock, once.**
            Dropped, and remembered. A single stale packet is a
            retransmit or a reordered buffer — `sources/recorder.py`
            keeps a `reordered` counter and a 64-packet reorder window,
            so this codebase expects them — and the correct response is
            to lose that one sample, not to lose state.
          * **Twice in a row, the second no older than the first.** That
            link's stream has restarted: a replay, a reconnect, a
            rebooted node, a second recording. Two samples agreeing on a
            new, forward-running clock is the cheapest evidence that
            distinguishes a restart from a stray packet, and it costs one
            sample to collect.
            Only then is its window cleared, and only ITS window. The
            earlier design restarted every link on the first such sample,
            which turned one late packet into a blank image on all
            N*(N-1) links and made any inter-node clock offset larger
            than `window_s` blank the room permanently. A per-link
            decision costs one dropped sample and bounds the damage to
            the link that actually has a problem.
        """
        last = self._t_last[index]
        restarted = False

        if last is not None and t < last - self.window_s:
            pending = self._pending_restart[index]
            if pending is None or t < pending:
                self._pending_restart[index] = t
                self.stale_samples += 1
                return False
            self._restart_link(index)
            last = None
            restarted = True

        self._pending_restart[index] = None

        # Advance the mesh progress clock. It is a monotone count of
        # elapsed time assembled from per-link forward steps, never from
        # a comparison between two links' timestamps, so a constant
        # offset between nodes cancels out of it entirely. One sample can
        # move it by at most one window: a single corrupt far-future
        # timestamp then costs one window on every link rather than
        # retiring the whole mesh's data for however long the corruption
        # claimed.
        step = 0.0 if last is None else t - last
        if step < 0.0:
            step = 0.0
        elif step > self.window_s:
            step = self.window_s
        reached = self._progress_at[index] + step
        if reached > self._progress:
            self._progress = reached
        self._progress_at[index] = self._progress

        if last is None or t > last:
            self._t_last[index] = t
        # `_t_stream` is "what time the stream thinks it is", the
        # reference `_advance_to` measures a caller's `t` against. It
        # follows the newest timestamp, and follows a restart backwards,
        # but is not dragged back by a merely out-of-order sample --- that
        # would make the next `vector(t)` age the windows by the lateness
        # of that packet on top of the elapsed time.
        if self._t_stream is None or t > self._t_stream or restarted:
            self._t_stream = t

        self._insert(index, t, value)
        self._dirty[index] = True
        self._prune(index)
        return True

    def _insert(self, index: int, t: float, value: float) -> None:
        """Append in time order, scanning back from the tail.

        Linear in the number of samples that arrived out of order, which
        under TDMA is zero, so the common path is one comparison and an
        append. The window is a dozen samples long; a backward scan is
        cheap here and transliterates to a C ring buffer without drama.
        """
        window = self._windows[index]
        if window and t < window[-1][0]:
            position = len(window)
            while position > 0 and window[position - 1][0] > t:
                position -= 1
            window.insert(position, (t, value))
        else:
            window.append((t, value))

    def _restart_link(self, index: int) -> None:
        """Forget one link's window because its stream restarted.

        The window has to go: its samples carry timestamps from the old
        clock, which after a restart are all *newer* than the new cutoff
        and would therefore never be pruned, leaving the variance
        straddling two recordings forever.

        Signal-extraction state survives on purpose: a reconnect does not
        change the room, and a cold `StaticSuppressor` would spend its
        first seconds emitting a settling transient.
        """
        self._windows[index].clear()
        self._cached_var[index] = 0.0
        self._dirty[index] = True
        self._t_last[index] = None
        self._progress_at[index] = self._progress
        self.clock_restarts += 1

    def _clear_windows(self) -> None:
        """Drop all windowed samples and all clock state, keeping signals."""
        n_links = self.geometry.n_links
        for window in self._windows:
            window.clear()
        self._dirty = [True] * n_links
        self._cached_var = [0.0] * n_links
        self._t_last = [None] * n_links
        self._progress_at = [0.0] * n_links
        self._pending_restart = [None] * n_links
        self._progress = 0.0
        self._t_stream = None

    def _advance_to(self, t: float | None) -> None:
        """Move the progress clock to wall-clock `t` without a sample.

        For the case the sample-driven clock cannot see: the stream stops
        entirely. `t` is on the same clock as the frames most recently
        fed in.

        Not capped, unlike `_ingest`'s per-sample step. The cap there
        bounds the damage a single corrupt *packet* can do; `t` here is
        the caller's own statement about the present, and a caller who
        knows the stream died an hour ago means it. Monotone, so calling
        `vector(t)` twice with the same `t` ages nothing twice.
        """
        if t is None or self._t_stream is None:
            return
        step = t - self._t_stream
        if step <= 0.0:
            return
        self._t_stream = t
        self._progress += step

    def _prune(self, index: int) -> None:
        """Drop samples older than one window on this link's own clock.

        The cutoff has two terms:

            (its own newest timestamp - window_s) + how much mesh
            progress has happened since it last reported

        The first keeps the window a true `window_s` of that link's data
        whatever its node's clock offset is. The second is what retires a
        link that has gone silent: its neighbours' progress walks the
        cutoff forward until the window empties, one window after its
        last packet. When every clock agrees the second term is zero
        between packets and the two collapse to the obvious global rule.

        A sample exactly `window_s` old is kept. That convention is
        arbitrary but has to be fixed, and it has to survive floating
        point: the cutoff is a subtraction and the sample times are
        usually a division, so "exactly `window_s` old" lands on either
        side of the comparison depending on the last ulp, and the sample
        count flickers between `k` and `k+1` on exactly periodic input —
        MEASURED at 10 Hz / 1.0 s, 12.5 Hz / 0.8 s and 20 Hz / 1.0 s. A
        tolerance of 1e-9 relative keeps the boundary sample in every
        time, and is nine orders of magnitude below any sample period
        this code will see, so it can never retain a sample that is
        genuinely too old.
        """
        last = self._t_last[index]
        if last is None:
            return
        window = self._windows[index]
        cutoff = last - self.window_s + (self._progress - self._progress_at[index])
        tolerance = 1e-9 * max(abs(last), self.window_s)
        while window and window[0][0] < cutoff - tolerance:
            window.popleft()
            self._dirty[index] = True
        while len(window) > self._max_window:
            window.popleft()
            self._dirty[index] = True
            self.window_overflows += 1

    def _variance(self, index: int) -> float:
        """Welford over the current window contents, cached until dirtied."""
        if self._dirty[index]:
            accumulator = RunningVariance()
            for _, value in self._windows[index]:
                accumulator.update(value)
            self._cached_var[index] = accumulator.variance
            self._dirty[index] = False
        return self._cached_var[index]

    # ---- output -------------------------------------------------------

    def vector(self, t: float | None = None) -> list[float]:
        """The per-link disturbance vector `y`, in `geometry.links` order.

        Always `len(geometry.links)` long. Entries for links without
        `min_samples` samples in the current window are 0.0.

        `t` is "the time is now this", for callers that know it — the
        frame time a replay has reached, or a wall clock. Without it the
        windows only age when samples arrive, so a stream that stops
        leaves the last image standing indefinitely and a tracker goes on
        being told about a person who has left. With it, a silent mesh
        reads as silent after one window. Omitted, the windows age on
        sample arrivals alone, which is right for a caller that is
        feeding this object as fast as packets come in.

        If `is_calibrated` is False this returns *raw* windowed variance
        with no baseline removed. That is deliberately not zeros: a caller
        who knows the baseline is uniform (a simulation, a test) can still
        use it. But in a real room the per-link noise floors differ by
        orders of magnitude, and an uncalibrated image is a picture of the
        link geometry rather than of a person. Check `is_calibrated`.
        """
        self._advance_to(t)
        out = [0.0] * self.geometry.n_links
        subtract = self.calibration_mode is CalibrationMode.SUBTRACT

        for index in range(len(out)):
            self._prune(index)
            if len(self._windows[index]) < self.min_samples:
                continue

            variance = self._variance(index)
            if not self._calibrated:
                y = variance
            elif subtract:
                y = variance - self._baseline[index]
            else:
                baseline = self._baseline[index]
                # A zero or negative baseline means calibration saw a link
                # that never moved at all. Dividing by it is meaningless,
                # so fall back to "no evidence" rather than to infinity.
                y = (variance / baseline - 1.0) if baseline > 0.0 else 0.0

            if self.clamp_negative and y < 0.0:
                y = 0.0
            out[index] = y

        return out

    def sample_counts(self, t: float | None = None) -> list[int]:
        """Samples currently in each link's window, in canonical order."""
        self._advance_to(t)
        for index in range(self.geometry.n_links):
            self._prune(index)
        return [len(window) for window in self._windows]

    def ready_mask(self, t: float | None = None) -> list[bool]:
        """Per link: does it have enough samples for `vector()` to be real?

        The inversion wants this. A 0.0 in `vector()` means either "quiet"
        or "no data", and only this tells them apart; dropping the
        not-ready rows from the solve is strictly better than feeding them
        a fabricated zero.

        A link whose baseline is not positive in NORMALIZE mode is also
        not ready, for the same reason: `vector()` has no meaningful
        division to do and falls back to 0.0, so the link votes "nobody
        is on my path" on every image no matter what it receives. That is
        a blind link wearing a quiet link's clothes, which is precisely
        the distinction this mask was introduced to make.
        """
        counts = self.sample_counts(t)
        normalize = self._calibrated and (
            self.calibration_mode is CalibrationMode.NORMALIZE
        )
        return [
            count >= self.min_samples
            and not (normalize and not self._baseline[index] > 0.0)
            for index, count in enumerate(counts)
        ]

    # ---- calibration --------------------------------------------------

    @property
    def is_calibrated(self) -> bool:
        return self._calibrated

    def calibrate_from(
        self,
        stream: Iterable[tuple[float, int, int, CsiFrame]],
        min_windows: int = 10,
    ) -> None:
        """Learn the empty-room baseline variance per link.

        `stream` is the same `(t, tx_id, rx_id, frame)` shape `sim.py`
        yields, so an empty-room run can be piped straight in. **The room
        must genuinely be empty**: anything that moves during calibration
        is learned as normal and is invisible afterwards.

        The baseline is the **mean** of the windowed variances seen, not
        the median. The quantity being subtracted is an expectation, and
        the sample variance is right-skewed (chi-squared-like), so its
        median sits below its mean — calibrating on the median would leave
        a positive bias on every link. The cost is that a single transient
        during calibration pulls the mean up; that is the price of being
        unbiased, and the fix is a clean calibration run, not a robust
        estimator that biases every link to avoid one bad one.

        **One estimate per window, not one per sample.** A window is
        accumulated only once `window_s` of that link's own clock has
        passed since the last one, so the estimates do not overlap and
        `min_windows` counts what its name says. Accumulating on every
        sample instead — the obvious implementation, since a sample is
        when the variance changes — inflates the count by the window's
        sample count, and the default `min_windows=10` is then satisfied
        by `min_samples + 10` samples, about 1.3 s at the default rate.
        MEASURED: a 1.3 s calibration accepted that way reports
        `calibrated_mask()` all True while its null sigma is 2.4x too
        small, and 17% of fresh EMPTY-room windows then clear a +3 sigma
        gate that should admit 0.1%. The gate has to count independent
        evidence or it is not a gate.

        So `min_windows` is a duration in disguise: the default 10, at
        the default 1 s window, asks for roughly 11 s of clean stream on
        every link before that link's baseline is believed. Raise it for
        a calibration you intend to keep — the null sigma it produces is
        what every later significance decision is measured against.

        Non-overlapping windows are still not quite independent — the
        channel drifts slowly, so neighbouring windows share a little —
        so treat `min_windows` as an upper bound on the effective sample
        size rather than an exact one.

        Links that never accumulate `min_windows` estimates get the median
        baseline of the links that did, and are flagged in
        `calibrated_mask()`. Imputing beats leaving them at 0.0, which
        would guarantee they read as permanently disturbed and would put a
        fixed bright ellipse in every image.

        Signal-extraction state (static-suppression baselines) is
        deliberately *not* reset afterwards: it is warm, and resetting it
        would make the first frames of the live run produce a spurious
        settling transient on every link at once.
        """
        n_links = self.geometry.n_links

        # Clear the sample windows first. If this object has been running
        # with someone in the room, the windows hold contaminated samples
        # that would otherwise land in the first calibration estimates.
        self._clear_windows()

        accumulators = [RunningVariance() for _ in range(n_links)]
        # When this link's next window estimate is due, on its own clock.
        due: list[float | None] = [None] * n_links

        for t, tx_id, rx_id, frame in stream:
            index = self.geometry.link_index(tx_id, rx_id)
            value = self._states[index].push(frame)
            if value is None:
                continue
            if not self._ingest(index, t, value):
                continue

            when = due[index]
            if when is None or t < when - self.window_s:
                # The link's first sample, or its clock restarted and
                # `_ingest` has just emptied the window. Either way there
                # is no window yet; wait a full `window_s` for one, so no
                # estimate is ever taken from a half-filled window.
                due[index] = t + self.window_s
                continue
            if t < when:
                continue
            if len(self._windows[index]) < self.min_samples:
                continue

            accumulators[index].update(self._variance(index))
            due[index] = t + self.window_s

        ok = [acc.count >= min_windows for acc in accumulators]
        if not any(ok):
            best = max((acc.count for acc in accumulators), default=0)
            raise ValueError(
                f"calibration saw no link reach {min_windows} independent "
                f"windows (best was {best}); the stream covers less than "
                f"{min_windows * self.window_s:g} s per link, or the geometry "
                f"does not match it"
            )

        fallback_mean = _median([accumulators[i].mean for i in range(n_links) if ok[i]])
        fallback_std = _median([accumulators[i].std for i in range(n_links) if ok[i]])

        for index in range(n_links):
            if ok[index]:
                self._baseline[index] = accumulators[index].mean
                self._baseline_std[index] = accumulators[index].std
            else:
                self._baseline[index] = fallback_mean
                self._baseline_std[index] = fallback_std

        self._baseline_ok = ok
        self._calibrated = True

        # Leave no calibration samples, and no calibration clock, behind.
        # The live stream often restarts its timestamps at zero (a second
        # recording, a replay, a test), and while `_ingest` now recovers
        # from that by itself it costs a dropped sample per link to do so.
        # The cost here is one window of warm-up after a calibration that
        # took tens of seconds, which is not a cost.
        self._clear_windows()

    def set_baseline(
        self,
        baseline: list[float],
        baseline_std: list[float] | None = None,
    ) -> None:
        """Install a saved baseline, bypassing a live calibration run.

        For restoring yesterday's calibration from disk, and for tests
        that want an exactly known baseline rather than a measured one.
        Pair it with `baseline_vector()` and `baseline_raw_std_vector()`,
        which are the two quantities in the units this method expects.

        Validated, because "restored from disk" is exactly where a
        truncated write or a half-finished calibration arrives from, and
        an unchecked one is invisible afterwards. A non-finite entry
        makes SUBTRACT-mode `y` NaN forever and slips past the
        `clamp_negative` test; a negative one is not a variance at all.
        Both raise. A zero entry is merely useless — in NORMALIZE mode it
        pins the link at `y = 0.0` — so it is accepted but marked not-ok
        in `calibrated_mask()` rather than silently affirmed.

        Omitting `baseline_std` leaves the null sigma unknown, which is
        also marked not-ok: a downstream significance test dividing by a
        vector of exact zeros gets infinities on every link, and a mask
        that says "all calibrated" while that is true is the worst of
        both worlds.
        """
        n_links = self.geometry.n_links
        if len(baseline) != n_links:
            raise ValueError(
                f"baseline has {len(baseline)} entries, geometry has {n_links} links"
            )
        if baseline_std is not None and len(baseline_std) != n_links:
            raise ValueError(
                f"baseline_std has {len(baseline_std)} entries, "
                f"geometry has {n_links} links"
            )
        for index, value in enumerate(baseline):
            if not math.isfinite(value):
                raise ValueError(f"baseline[{index}] is {value}, not a finite number")
            if value < 0.0:
                raise ValueError(
                    f"baseline[{index}] is {value}; a variance is not negative"
                )
        if baseline_std is not None:
            for index, value in enumerate(baseline_std):
                if not math.isfinite(value):
                    raise ValueError(
                        f"baseline_std[{index}] is {value}, not a finite number"
                    )
                if value < 0.0:
                    raise ValueError(
                        f"baseline_std[{index}] is {value}; a deviation is not negative"
                    )

        self._baseline = list(baseline)
        if baseline_std is None:
            self._baseline_std = [0.0] * n_links
            self._baseline_ok = [False] * n_links
        else:
            self._baseline_std = list(baseline_std)
            self._baseline_ok = [value > 0.0 for value in baseline]
        self._calibrated = True

    def baseline_vector(self) -> list[float]:
        """Per-link empty-room variance, in canonical order. Save this."""
        return list(self._baseline)

    def baseline_std_vector(self) -> list[float]:
        """Per-link standard deviation of `y` under the null "nobody is there".

        In the units `vector()` emits, which is the only version of this
        number that is safe to divide a `y` by, and it therefore depends
        on `calibration_mode`. SUBTRACT returns `y = var - baseline`, so
        the null sigma is the spread of the windowed variance. NORMALIZE
        returns `y = var / baseline - 1`, so it is that spread divided by
        the baseline — a factor `1 / baseline` larger, and a factor that
        varies across links by construction, since unequal link noise
        floors are the entire reason NORMALIZE exists. MEASURED at 22 dB
        SNR on a four-node perimeter: returning the raw spread in
        NORMALIZE mode predicts 0.0094 against an observed sd of 0.399, a
        factor of 42, and a 5-sigma gate built on it reads an empty room
        as 58% occupied.

        Nothing downstream can derive this — only this module ever sees
        the empty room — so it is exported rather than kept private. For
        saving a calibration to disk use `baseline_raw_std_vector()`,
        which is what `set_baseline` takes back.
        """
        if self.calibration_mode is CalibrationMode.SUBTRACT:
            return list(self._baseline_std)
        return [
            (std / base) if base > 0.0 else 0.0
            for std, base in zip(self._baseline_std, self._baseline)
        ]

    def baseline_raw_std_vector(self) -> list[float]:
        """Per-link spread of the empty-room windowed variance, unscaled.

        The mode-independent form, for persistence: `baseline_vector()`
        and this round-trip through `set_baseline`. `baseline_std_vector`
        is the one to divide a `y` by.
        """
        return list(self._baseline_std)

    def calibrated_mask(self) -> list[bool]:
        """Per link: is its calibration real, rather than imputed or unusable?

        False means the baseline was imputed from the median of the other
        links, or was installed without a null sigma, or is not a usable
        positive variance. All three are "do not trust this link's `y`
        quantitatively", which is the question a caller is asking.
        """
        return list(self._baseline_ok)

    # ---- lifecycle ----------------------------------------------------

    def reset(self, keep_calibration: bool = True) -> None:
        """Drop all windowed data and signal state.

        Calibration is kept by default: the baseline describes the room,
        not the stream, and survives a reconnect. The diagnostic counters
        are kept too — they count what this object has seen in its life,
        and zeroing them on a reconnect would hide the reconnects.
        """
        self._clear_windows()
        for state in self._states:
            state.reset()

        if not keep_calibration:
            n_links = self.geometry.n_links
            self._baseline = [0.0] * n_links
            self._baseline_std = [0.0] * n_links
            self._baseline_ok = [False] * n_links
            self._calibrated = False

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"LinkMeasurement(links={self.geometry.n_links}, "
            f"signal={self.signal.name}, window={self.window_s:g} s @ "
            f"{self.sample_rate_hz:g} Hz, min_samples={self.min_samples}, "
            f"mode={self.calibration_mode.value}, "
            f"calibrated={self._calibrated})"
        )
