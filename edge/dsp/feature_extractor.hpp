// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — one link's CSI frames in, FeatureFrames out.
//
// Port of `FeatureExtractor` in `wifisense/features.py`. Header-only, no
// allocation after construction, float32 in the signal path. This is the
// top-level object `edge/firmware` drives: everything else in this directory
// hangs off it.
//
// Wraps `SensingPipeline` and adds what the wire needs: gap filling,
// gross-motion and breathing-band RMS, link statistics, BNR link quality,
// consent gating and encoding.
//
// Call `push(frame, sequence, t_s)` for every *received* CSI frame. It returns
// zero or more FeatureFrames (normally zero; one every
// `round(fs / frame_rate_hz)` PROCESSED frames).
//
// ---------------------------------------------------------------------------
// FEATURES ARE CLOCKED BY PROCESSED FRAMES, NOT BY WALL TIME
// ---------------------------------------------------------------------------
// spec/wire.md §2 "Timing" is explicit: "Features are sampled every
// `round(fs / frame_rate_hz)` CSI frames (after gap filling), not by wall
// clock, so they are deterministic given the input." That is what makes a
// golden vector possible at all — a wall-clock trigger would emit a different
// number of frames on every run and on every board. The arrival timestamps
// still matter, but only as DATA: they feed `uptime_ms` and the 10 s link
// statistics, never the decision to emit.
//
// ---------------------------------------------------------------------------
// WHERE DOUBLE IS ALLOWED, AND WHY IT IS ONLY HERE
// ---------------------------------------------------------------------------
// ADR-003 is float32, and every signal path in this library obeys it. Two
// kinds of quantity are the documented exception (SAANS_PLAN §3.1, "time
// accumulators may be double or integer microseconds, per the uptime_ms
// tolerance"):
//
//   1. ARRIVAL TIMESTAMPS. `uptime_ms` must agree with the reference to within
//      1 ms over the node's whole uptime. A float32 second-count cannot: it
//      resolves 1 ms only up to 2^24 ms ~ 4.7 hours, after which `t - t0`
//      quantises to 2 ms, then 4, and the field drifts away from the reference
//      for reasons that have nothing to do with the DSP. These are 2 Hz
//      bookkeeping, not hot-path arithmetic.
//
//   2. INIT-TIME SAMPLE COUNTS. `round(fs * window_s)` and friends are
//      computed in double for the reason vitals.hpp spells out under WHERE THE
//      DERIVED COUNTS ARE COMPUTED: both implementations see the same `fs` and
//      the same duration, so there is no parameter gap for a float32 product
//      to close and the rounding is pure added error — in the direction that
//      matters, because an off-by-one here moves a window by a whole sample.
//
// Everything else — the filters, the RMS windows, the BNR — is float32.
//
// ---------------------------------------------------------------------------
// ROUNDING IS PYTHON'S, NOT C's
// ---------------------------------------------------------------------------
// The reference reaches for `round()` in eight places, five of which land in
// fields the contract calls EXACT (`rssi_dbm`, `noise_floor_dbm`) or
// within-one (`uptime_ms`, `csi_rate_chz`, `loss_permille`). Python's `round`
// is round-half-to-EVEN; C's `round()` is round-half-away-from-zero. They
// differ on every exact half, which is not an exotic input here: an RSSI mean
// of -44.5 over an even number of frames is an ordinary Tuesday. See
// `round_half_even`.
//
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "bnr.hpp"
#include "core.hpp"
#include "feature_frame.hpp"
#include "gapfill.hpp"
#include "pipeline.hpp"

namespace saans {

// ---------------------------------------------------------------------------
// Window lengths fixed by the spec, in seconds.
// ---------------------------------------------------------------------------

inline constexpr float kBreathRmsWindowS   = 2.0f;
inline constexpr float kGrossMotionWindowS = 1.0f;
inline constexpr float kLinkStatsWindowS   = 10.0f;

/// Gross motion: above the breathing band and its first harmonics, below the
/// Nyquist of a 10 Hz link. Limbs, posture changes, walking.
inline constexpr float kGrossMotionLowHz  = 1.0f;
inline constexpr float kGrossMotionHighHz = 4.0f;

/// Biquad sections cascaded for the gross-motion band.
///
/// One RBJ section is 6 dB/octave, which leaves plenty of the breathing band
/// leaking into a measurement meant to mean "limbs and walking, not breathing".
/// Two identical sections double the roll-off and let `gross_motion` separate
/// the two the way the detectors in SAANS_PLAN 2.1 assume. Matches
/// GROSS_MOTION_SECTIONS in the reference.
inline constexpr std::size_t kGrossMotionSections = 2;
inline constexpr float kBreathBandLowHz   = 0.1f;
inline constexpr float kBreathBandHighHz  = 0.5f;

namespace detail {

/// Python's `round()`: round half to EVEN, returning an integer.
///
/// CPython's `float.__round__` with no ndigits is C `round()` corrected on the
/// halfway case (`floatobject.c`: `if (fabs(x - rounded) == 0.5) rounded =
/// 2.0 * round(x / 2.0);`), which is round-half-to-even exactly. `nearbyint`
/// in the default FE_TONEAREST mode is the same function, so that is what is
/// used rather than a hand-rolled correction that would have to be argued
/// about. Nothing in this library changes the rounding mode, and the build
/// does not pass -ffast-math (which would licence the compiler to assume it).
///
/// Saturates rather than invoking undefined behaviour for an absurd input; a
/// bare cast of a huge double to an integer type is UB and the inputs here are
/// a caller's wall-clock timestamps.
inline std::int64_t round_half_even(double v) {
    if (!std::isfinite(v)) {
        return 0;
    }
    const double r = std::nearbyint(v);
    if (r >= 9.2233720368547758e18) {
        return INT64_MAX;
    }
    if (r <= -9.2233720368547758e18) {
        return INT64_MIN;
    }
    return static_cast<std::int64_t>(r);
}

/// Python's `int(v)` on a non-negative double — truncation toward zero — with
/// a floor, matching the `max(1, int(...))` and `max(8, int(...))` idioms the
/// reference uses for its derived sample counts.
inline std::uint32_t truncate_floor(double v, std::uint32_t floor_value) {
    if (!std::isfinite(v) || v <= static_cast<double>(floor_value)) {
        return floor_value;
    }
    if (v >= 4294967295.0) {
        return 0xFFFFFFFFu;
    }
    const std::uint32_t t = static_cast<std::uint32_t>(v);
    return (t < floor_value) ? floor_value : t;
}

/// `max(floor_value, round(v))` — the same idiom with Python's rounding
/// instead of its truncation. Kept separate rather than merged behind a flag
/// because which one the reference used at each site is the whole question.
inline std::uint32_t round_floor(double v, std::uint32_t floor_value) {
    const std::int64_t r = round_half_even(v);
    if (r <= static_cast<std::int64_t>(floor_value)) {
        return floor_value;
    }
    if (r >= 4294967295LL) {
        return 0xFFFFFFFFu;
    }
    return static_cast<std::uint32_t>(r);
}

/// Saturating narrow to the int32 the FeatureFrame holds. The frame's own
/// encoder applies the reference's `max(lo, min(hi, x))` clamps; this only
/// stops an absurd intermediate from wrapping on the way there.
inline std::int32_t to_i32(std::int64_t v) {
    if (v > 2147483647LL) {
        return 2147483647;
    }
    if (v < -2147483648LL) {
        return -2147483647 - 1;
    }
    return static_cast<std::int32_t>(v);
}

// ---------------------------------------------------------------------------
// RollingRms
// ---------------------------------------------------------------------------

/// RMS over the last `n` samples — `features._RollingRms`.
///
/// RE-SUMMED ON EVERY READ, DELIBERATELY. The reference's own comment:
///
///     "Summed afresh on each read (twice a second) rather than kept as a
///      running total: running-sum float drift over a night of samples would
///      make the three implementations disagree for no physical reason."
///
/// That argument is STRONGER in float32, not weaker, so the re-summing stays
/// and a running total is not an optimisation to reach for here. The sum is
/// `NeumaierSum` because CPython >= 3.12's builtin `sum()` is itself
/// compensated and the reference leans on it; see core.hpp.
///
/// The reference pushes `x * x` into the deque and sums the squares, so this
/// stores the square too. Squaring at read time instead would be the same
/// single rounding, but storing it keeps the two implementations doing the
/// same operations at the same moments, which is the whole point of the
/// re-summing.
template <std::size_t MaxSamples>
class RollingRms {
    static_assert(MaxSamples >= 1, "a rolling RMS needs at least one sample");

  public:
    /// `deque(maxlen=max(1, n))`. False (and inert) if the window does not fit.
    bool configure(std::uint32_t n) {
        const std::uint32_t want = (n < 1u) ? 1u : n;
        if (want > MaxSamples) {
            return false;
        }
        cap_ = want;
        sq_.clear();
        return true;
    }

    void push(float x) { sq_.push(x * x); }

    float value() const {
        // Only the newest `cap_` squares are the reference's deque; the ring
        // may be wider because its capacity is a compile-time RAM budget.
        const std::size_t held = sq_.size();
        const std::size_t n    = (held < cap_) ? held : static_cast<std::size_t>(cap_);
        if (n == 0) {
            return 0.0f;  // `if not self._sq: return 0.0`
        }
        const std::size_t base = held - n;
        NeumaierSum       acc;
        for (std::size_t i = 0; i < n; ++i) {
            acc.add(sq_[base + i]);
        }
        const float mean_sq = acc.value() / static_cast<float>(n);
        // A sum of squares cannot be negative; the guard only stops a
        // compensation underflow becoming a NaN, and sqrtf(0) == 0 either way.
        return sqrtf(mean_sq < 0.0f ? 0.0f : mean_sq);
    }

    void clear() { sq_.clear(); }

    std::uint32_t capacity_configured() const { return cap_; }

  private:
    RingBuffer<float, MaxSamples> sq_;
    std::uint32_t                 cap_ = 1;
};

// ---------------------------------------------------------------------------
// BnrRowStore
// ---------------------------------------------------------------------------

/// Everything the BNR link-quality path needs, in one member.
///
/// It is here, rather than spelled out in the extractor, so that there can be
/// a `Rows == 0` specialisation holding NOTHING. The row store is the single
/// largest term in a link's budget — 120 x 56 floats is 26,880 B, more than a
/// third of the object — and a node that does not report `best_bnr` should not
/// pay for it. A runtime flag alone cannot do that: the array is a member
/// either way.
template <std::size_t Rows, std::size_t Cols>
struct BnrRowStore {
    static constexpr bool kPresent = true;

    /// Block-averaged residuals, oldest to newest within the ring the
    /// extractor's head and size index.
    float rows[Rows][Cols]{};
    /// Oldest-to-newest row pointers handed to `subcarrier_bnr_rows`, which
    /// reads them in place — so the analysis window is never copied.
    const float* ptrs[Rows]{};
    /// `_bnr_acc`: the partial block being averaged.
    float acc[Cols]{};
    /// Where `subcarrier_bnr_rows` writes one ratio per subcarrier.
    float out[Cols]{};
    /// The zero-padded transform. `next_pow2(Rows)` floats twice over.
    BnrWorkspace<Rows> work{};
};

/// `MaxBnrRows == 0`: this build does not compute BNR at all.
///
/// The storage is not merely unused, it is ABSENT — which is the whole point,
/// and is why the extractor reaches for `if constexpr` rather than a runtime
/// branch around every use. `best_bnr` then reads 0.0 on the wire, which
/// spec/wire.md defines as UNKNOWN rather than as a bad link.
template <std::size_t Cols>
struct BnrRowStore<0, Cols> {
    static constexpr bool kPresent = false;
};

}  // namespace detail

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Every knob `FeatureExtractor.__init__` takes, with the reference's defaults.
///
/// `subcarrier_selection` is absent: the C++ pipeline implements the
/// "variance" path only (pipeline.hpp says why), and a field that could only
/// hold one value would be a promise this port cannot keep.
struct FeatureExtractorConfig {
    std::size_t  n_subcarriers  = 56;
    float        sample_rate_hz = 20.0f;
    std::uint8_t node_id        = 0;
    std::uint8_t tx_id          = 0xFF;
    float        frame_rate_hz  = 2.0f;
    bool         vitals_enabled = false;
    bool         heart_enabled  = false;
    float        calibration_s  = 30.0f;
    float        max_gap_s      = 1.0f;
    bool         raw_mode       = false;
    float        bnr_window_s   = 30.0f;
    float        bnr_update_s   = 5.0f;
    float        bnr_rate_hz    = 4.0f;

    /// Whether to compute `best_bnr` at all — `FeatureExtractor(compute_bnr=)`
    /// in the reference, and `FeatureExtractorConfig.compute_bnr` in the Rust
    /// port. DEFAULT TRUE, which is what every golden vector was generated
    /// with.
    ///
    /// With it false the accumulate-and-score step is skipped entirely and
    /// `best_bnr` stays 0.0, which spec/wire.md defines as UNKNOWN — "before
    /// the first window, or on a memory-constrained node that does not compute
    /// it". NO OTHER FIELD MOVES: the BNR path reads the pipeline's residuals
    /// and feeds nothing back into it, so the rest of the frame is bit-
    /// identical with it on or off. The spec owner verified that across 280
    /// frames; `bench/size_report.cpp` re-checks it here, field by field, over
    /// the 336 frames of the `features_vitals` recording.
    ///
    /// The memory it saves is only saved when `MaxBnrRows == 0` as well — a
    /// runtime flag cannot un-declare a member array. See the template
    /// parameters below.
    bool compute_bnr = true;
};

// ---------------------------------------------------------------------------
// FeatureExtractor
// ---------------------------------------------------------------------------

/// One link's CSI frames in, FeatureFrames out.
///
/// ---------------------------------------------------------------------------
/// THE TEMPLATE PARAMETERS ARE THE NODE'S RAM BUDGET
/// ---------------------------------------------------------------------------
/// Every buffer is a member array, so `sizeof(FeatureExtractor)` is the whole
/// per-link cost and the linker can see it. At the defaults below, on a 32-bit
/// target, the dominant terms are:
///
///   bnr rows      MaxBnrRows * MaxSubcarriers    128 * 64 * 4  = 32.0 KB
///   pipeline      see BasicSensingPipeline                      17.6 KB
///   gap arena     MaxFill * MaxSubcarriers       32 * 64 * 4   =  8.0 KB
///   arrivals      MaxArrivals * 12                             =  6.0 KB
///   rms windows   2 * MaxRmsSamples * 4                        =  2.0 KB
///                                                              ~ 73 KB/link
///
/// MEASURED, not estimated: `sizeof` says 74648 bytes at the defaults and
/// 47608 for `BasicFeatureExtractor<32, 128, 256, 64, 32>`, on a 64-bit host.
/// The Xtensa figure is a little smaller (32-bit `size_t` and pointers) and is
/// the one to budget from; `test_feature_extractor.cpp` drives both shapes and
/// requires them to emit byte-identical frames, so trimming the parameters is
/// a RAM decision and never a behaviour one.
///
/// A node that knows it sees 32 subcarriers should say so —
/// `BasicFeatureExtractor<32>` halves the first and third lines outright. The
/// defaults are sized for an ESP32 HT20 channel's 56 usable subcarriers with
/// room to spare, because a configuration that does not fit goes INERT rather
/// than being silently shortened, and a dead link is not what anyone wants to
/// discover in the field.
///
/// THE DEFAULTS ARE DELIBERATELY LOOSE, AND A MULTI-LINK NODE SHOULD NOT USE
/// THEM. Every one of them is sized for a generic configuration rather than
/// for one sample rate, because a default that only suits 20 Hz is a trap for
/// whoever next runs this at 50. `edge/dsp/node_config.hpp` derives the tight
/// set from the node's actual constants and static_asserts each one against
/// the requirement it has to meet. Measured by `bench/run_size_report.sh` at
/// 56 subcarriers and 20 Hz, ON THE S3: 67,432 bytes a link becomes 51,808
/// tightened, 45,408 once the spectral scratch is shared rather than owned
/// (pipeline.hpp) — plus one 6,404-byte arena for the node — and 16,584 with
/// `best_bnr` compiled out. Two links in 171 KB, then three, then ten. The
/// host figures are a little larger and the same script prints them.
///
/// `MaxBnrRows` must cover `max(8, int(bnr_window_s * bnr_rate_hz))` (120 at
/// the node's configuration) AND the FFT it feeds, whose length is
/// `next_pow2(rows)` — so a power of two is the natural choice, though any
/// value that covers the window works and 120 is what the node needs.
///
/// `MaxBnrRows == 0` is the special case: NO BNR ON THIS BUILD. The row store
/// disappears entirely (detail::BnrRowStore's specialisation holds nothing),
/// `configure()` refuses a config that asks for `compute_bnr`, and `best_bnr`
/// goes out as 0.0 — which spec/wire.md defines as "unknown", not as a bad
/// link, precisely so a memory-bound node can say it. Nothing downstream reads
/// the field except system health.
///
/// `MaxArrivals` bounds the 10 s link-statistics ring, and it is a PEAK
/// OCCUPANCY rather than a window length. Two entries more than `fs * 10`:
///
///   * the window is CLOSED AT BOTH ENDS — `push_impl` pops while
///     `t_front < t - 10`, so the arrival exactly 10 s old is kept — and
///   * THE PUSH PRECEDES THE PRUNE, so the ring transiently holds the whole
///     closed window plus the arrival that is about to evict its oldest
///     entry.
///
/// Neither is slack. A 200-entry window dropped 30 arrivals for capacity on
/// `spec/golden/features_vitals`, which moves `csi_rate_chz` and
/// `loss_permille`; a 201-entry one drops none on that recording but forces an
/// eviction on EVERY frame of a steady, lossless 20 Hz link, which leaves
/// `arrival_overflows` permanently non-zero and unable to signal anything.
/// `node_config.hpp`'s `derive::arrival_peak` is where the figure is derived,
/// with the jitter headroom a real link needs on top of it; the stock 512
/// covers any rate up to ~51 Hz. See `arrival_overflows` for what happens past
/// the capacity; it is the one place this port degrades rather than refusing,
/// because the alternative — dropping the newest arrival — would make the loss
/// figure report the buffer instead of the link.
///
/// ERROR HANDLING. The reference raises ValueError for a sample rate below
/// 10 Hz and for a frame rate outside `(0, fs]`. Built -fno-exceptions, both
/// come back as `configure()` returning false and the extractor staying INERT:
/// `valid()` is false and `push` returns nothing. The same answer is given for
/// a configuration that does not fit the template parameters above, which has
/// no analogue in the reference — a silently shortened BNR window would report
/// a link quality computed over the wrong span and nothing downstream could
/// tell.
template <std::size_t MaxSubcarriers = 64, std::size_t MaxBnrRows = 128,
          std::size_t MaxArrivals = 512, std::size_t MaxRmsSamples = 256,
          std::size_t MaxFill = 32, std::size_t BreathHistory = 192,
          std::size_t HeartHistory = 960, std::size_t MaxEnergySamples = 256,
          class Scratch = OwnedScratch<scratch_floats_for(BreathHistory, HeartHistory)>>
class BasicFeatureExtractor {
    static_assert(MaxSubcarriers > 0, "MaxSubcarriers must be non-zero");
    // 0 is not "a very small window", it is NO WINDOW — the build has no BNR
    // in it at all. Everything in between is still refused: the reference's
    // `max(8, ...)` floor means a store of 1..7 rows can never be filled, so
    // `best_bnr` would sit at 0.0 forever while the RAM was spent anyway.
    static_assert(MaxBnrRows == 0 || MaxBnrRows >= 8,
                  "MaxBnrRows is either 0 (no BNR compiled in) or at least the reference's "
                  "window floor of 8 rows");
    static_assert(MaxArrivals >= 2, "link statistics need two arrivals");

  public:
    using Pipeline  = BasicSensingPipeline<MaxSubcarriers, BreathHistory, HeartHistory,
                                          MaxEnergySamples, Scratch>;
    using GapFiller = SequenceGapFiller<MaxSubcarriers, MaxFill>;

    /// Whether this BUILD can compute `best_bnr` at all. `config().compute_bnr`
    /// is the separate question of whether this INSTANCE does.
    static constexpr bool kBnrCompiledIn = (MaxBnrRows > 0);
    static_assert(detail::BnrRowStore<MaxBnrRows, MaxSubcarriers>::kPresent ==
                      kBnrCompiledIn,
                  "the row store and the flag that guards every use of it disagree");

    /// Frames emitted by one `push`. Views into the extractor's own storage,
    /// valid until the next `push` — the same discipline `SequenceGapFiller`
    /// already requires. Normally empty.
    struct Output {
        const FeatureFrame* frames = nullptr;
        std::size_t         count  = 0;

        const FeatureFrame* begin() const { return frames; }
        const FeatureFrame* end() const { return frames + count; }
        bool                empty() const { return count == 0; }
        const FeatureFrame& operator[](std::size_t i) const { return frames[i]; }
    };

    BasicFeatureExtractor() = default;

    explicit BasicFeatureExtractor(const FeatureExtractorConfig& cfg) { (void)configure(cfg); }

    bool configure(const FeatureExtractorConfig& cfg) {
        valid_   = false;
        bnr_on_  = false;
        cfg_     = cfg;

        // "FeatureFrame needs >= 10 Hz CSI: the gross-motion band reaches
        //  4 Hz, and breathing needs it anyway (campus plan, constraint 4)"
        if (!(cfg.sample_rate_hz >= 10.0f)) {
            return false;
        }
        if (!(cfg.frame_rate_hz > 0.0f) || !(cfg.frame_rate_hz <= cfg.sample_rate_hz)) {
            return false;
        }
        if (cfg.n_subcarriers == 0 || cfg.n_subcarriers > MaxSubcarriers) {
            return false;
        }

        const double fs = static_cast<double>(cfg.sample_rate_hz);

        if (!pipeline_.configure(cfg.n_subcarriers, cfg.sample_rate_hz, cfg.calibration_s)) {
            return false;
        }
        if (!breath_filter_.configure(kBreathBandLowHz, kBreathBandHighHz, cfg.sample_rate_hz)) {
            return false;
        }
        for (BiquadBandpass& section : motion_filters_) {
            if (!section.configure(kGrossMotionLowHz, kGrossMotionHighHz,
                                   cfg.sample_rate_hz)) {
                return false;
            }
        }

        // `_RollingRms(round(sample_rate_hz * WINDOW_S))`, and note the deque's
        // own `max(1, n)` floor is applied inside RollingRms::configure.
        if (!breath_rms_.configure(
                detail::round_floor(fs * static_cast<double>(kBreathRmsWindowS), 0u))) {
            return false;
        }
        if (!motion_rms_.configure(
                detail::round_floor(fs * static_cast<double>(kGrossMotionWindowS), 0u))) {
            return false;
        }

        // `_every = max(1, round(sample_rate_hz / frame_rate_hz))`.
        every_ = detail::round_floor(fs / static_cast<double>(cfg.frame_rate_hz), 1u);

        // `SequenceGapFiller(max_gap=int(max_gap_s * sample_rate_hz))`.
        // `int()`, not `round()`: the reference truncates here and rounds
        // elsewhere, and which it did is exactly what a port must not guess.
        const std::uint32_t max_gap =
            detail::truncate_floor(static_cast<double>(cfg.max_gap_s) * fs, 0u);
        gaps_ = GapFiller(max_gap);
        if (max_gap > MaxFill) {
            // SequenceGapFiller clamps silently (gapfill.hpp, THE CAPACITY
            // CONTRACT) which is right for a filler used directly, but here it
            // would make `max_gap_s` mean something different from what the
            // caller asked and from what the reference does. Refuse instead.
            return false;
        }

        // BNR link quality: block-average residuals to ~bnr_rate_hz, score the
        // last bnr_window_s every bnr_update_s (same scheme as RespirFiFusion,
        // so best_bnr means the same thing as its selector).
        //
        // The derived counts below are computed even when `compute_bnr` is
        // false, exactly as `__init__` computes them before `_bnr_step` is ever
        // reached — including this refusal, which stands in for the
        // ZeroDivisionError the reference would raise. Only the CAPACITY check
        // is conditional, because a store that is never written cannot
        // overflow.
        if (!(cfg.bnr_rate_hz > 0.0f)) {
            return false;
        }
        // Asked for a link-quality figure from a build that has no room store.
        // Refusing is the only honest answer: silently reporting 0.0 would be
        // indistinguishable on the wire from "the window has not filled yet",
        // and silently turning the flag off would hide a misconfigured node.
        if (cfg.compute_bnr && !kBnrCompiledIn) {
            return false;
        }
        bnr_on_ = cfg.compute_bnr && kBnrCompiledIn;
        bnr_dec_ = detail::round_floor(fs / static_cast<double>(cfg.bnr_rate_hz), 1u);
        // `_bnr_rate = sample_rate_hz / self._bnr_dec`. FLOAT32: it is used as
        // a sample rate by the float32 transform below, the same way
        // `analysis_rate_hz_` is in vitals.hpp.
        bnr_rate_ = cfg.sample_rate_hz / static_cast<float>(bnr_dec_);
        // `maxlen=max(8, int(bnr_window_s * self._bnr_rate))`, with the f64
        // `_bnr_rate` the reference forms — seconds times a rate, so DOUBLE.
        bnr_cap_ = detail::truncate_floor(
            static_cast<double>(cfg.bnr_window_s) * (fs / static_cast<double>(bnr_dec_)), 8u);
        if (bnr_on_ && bnr_cap_ > MaxBnrRows) {
            return false;
        }
        // `_bnr_every = max(1, int(round(bnr_update_s * sample_rate_hz)))`
        bnr_every_ = detail::round_floor(static_cast<double>(cfg.bnr_update_s) * fs, 1u);

        reset();
        valid_ = true;
        return true;
    }

    bool valid() const { return valid_; }

    const FeatureExtractorConfig& config() const { return cfg_; }
    std::uint32_t                 frames_per_feature() const { return every_; }
    std::uint32_t                 processed() const { return processed_; }
    float                         best_bnr() const { return best_bnr_; }

    /// Whether this instance is actually computing `best_bnr`. False means the
    /// field on the wire is 0.0 = UNKNOWN for the whole run, by configuration
    /// or because the build has no row store — not because the link is bad.
    bool computes_bnr() const { return bnr_on_; }
    Pipeline&                     pipeline() { return pipeline_; }
    const Pipeline&               pipeline() const { return pipeline_; }
    const GapFiller&              gaps() const { return gaps_; }

    /// How many arrivals were dropped from the 10 s link-statistics window
    /// because `MaxArrivals` was reached. Nonzero means `csi_rate_chz` and
    /// `loss_permille` are computed over a shorter span than the spec's 10 s —
    /// still the right density, but no longer the reference's window. It is a
    /// counter rather than a refusal because the link is fine; the budget is
    /// not.
    ///
    /// IT ONLY MEANS THAT IF ZERO IS REACHABLE. `MaxArrivals` has to cover the
    /// PEAK occupancy — see the template parameters above — and a capacity one
    /// short of it evicts on every frame of a perfectly healthy link, at which
    /// point this counter is noise rather than a signal. The node's capacity is
    /// derived for that peak with jitter headroom on top
    /// (`node_config.hpp::derive`), and `test_node_config.cpp` drives a steady,
    /// exactly-nominal stream and requires zero, so the day this becomes
    /// non-zero in the field it means what it says: the link is delivering
    /// faster than the node was configured for.
    std::uint32_t arrival_overflows = 0;

    // -- input --------------------------------------------------------------

    /// Push one *received* CSI frame. Gap filling happens FIRST: the pipeline
    /// never sees a stream with holes in it.
    Output push(const CsiFrame& frame, std::uint32_t sequence, double t_s,
                int noise_floor_dbm) {
        return push_impl(frame, sequence, t_s, true, true, noise_floor_dbm);
    }

    Output push(const CsiFrame& frame, std::uint32_t sequence, double t_s) {
        return push_impl(frame, sequence, t_s, true, false, 0);
    }

    /// `t_s=None` in the reference: the arrival time is taken to be
    /// `self._processed / self.fs`, i.e. a perfectly regular clock.
    Output push(const CsiFrame& frame, std::uint32_t sequence) {
        return push_impl(frame, sequence, 0.0, false, false, 0);
    }

    /// `sequence=None`: the frame's own sample index.
    Output push(const CsiFrame& frame) {
        return push_impl(frame, frame.sample_index, 0.0, false, false, 0);
    }

    /// Encode one emitted frame into 68 bytes.
    ///
    /// The frames `push` returns are ALREADY consent-gated — `_emit` writes
    /// 0.0 into the vitals fields and withholds their flags, which is the
    /// privacy boundary ADR-005 asks for and is what `features_private` pins.
    /// This still runs `apply_consent_gate`, which only ever clears, so it is
    /// a no-op on a correctly built frame and a backstop on any other: the
    /// one call a sender can reach for without having to have got `_emit`
    /// right. See feature_frame.hpp.
    void encode(const FeatureFrame& f, std::uint8_t* out) const {
        VitalsConsent consent;
        consent.vitals_enabled = cfg_.vitals_enabled;
        consent.heart_enabled  = cfg_.heart_enabled;
        encode_feature_frame_gated(f, consent, out);
    }

    /// Everything back to construction except the configuration.
    void reset() {
        pipeline_.reset();
        gaps_.reset();
        breath_filter_.reset();
        for (BiquadBandpass& section : motion_filters_) {
            section.reset();
        }
        breath_rms_.clear();
        motion_rms_.clear();

        if constexpr (kBnrCompiledIn) {
            for (std::size_t k = 0; k < MaxSubcarriers; ++k) {
                bnr_.acc[k] = 0.0f;
            }
        }
        bnr_n_    = 0;
        bnr_head_ = 0;
        bnr_size_ = 0;
        best_bnr_ = 0.0f;

        processed_            = 0;
        seq_                  = 0;
        have_t0_              = false;
        t0_                   = 0.0;
        calibrated_at_        = 0;
        have_calibrated_at_   = false;
        arrivals_head_        = 0;
        arrivals_size_        = 0;
        arrival_overflows     = 0;
        rssi_sum_             = 0;
        rssi_n_               = 0;
        noise_sum_            = 0;
        noise_n_              = 0;
        discontinuities_seen_ = 0;
    }

  private:
    // -- push ---------------------------------------------------------------

    Output push_impl(const CsiFrame& frame, std::uint32_t sequence, double t_s, bool have_t,
                     bool have_noise, int noise_floor_dbm) {
        Output out;
        out.frames = out_frames_;
        if (!valid_) {
            return out;
        }

        // `t = t_s if t_s is not None else self._processed / self.fs`
        const double t = have_t ? t_s
                                : (static_cast<double>(processed_) /
                                   static_cast<double>(cfg_.sample_rate_hz));
        if (!have_t0_) {
            t0_      = t;
            have_t0_ = true;
        }

        arrivals_push(t, sequence);
        // `while self._arrivals and self._arrivals[0][0] < t - LINK_STATS_WINDOW_S`
        const double cutoff = t - static_cast<double>(kLinkStatsWindowS);
        while (arrivals_size_ > 0 && arrivals_t_[arrivals_index(0)] < cutoff) {
            arrivals_pop_front();
        }

        rssi_sum_ += static_cast<std::int64_t>(frame.rssi_dbm);
        ++rssi_n_;
        if (have_noise) {
            noise_sum_ += static_cast<std::int64_t>(noise_floor_dbm);
            ++noise_n_;
        }

        // GAP FILLING HAPPENS FIRST, before the pipeline sees anything.
        const typename GapFiller::Output filled = gaps_.push(sequence, frame, arena_);
        for (std::size_t i = 0; i < filled.count; ++i) {
            if (process_one(filled[i], t)) {
                out_frames_[out.count] = emitted_;
                ++out.count;
            }
        }
        return out;
    }

    /// `_process`: one frame through the pipeline and everything hung off it.
    /// Returns true when a FeatureFrame was produced (left in `emitted_`).
    bool process_one(const CsiFrame& frame, double t) {
        // The STATEFUL half only; the spectral estimates are deferred to
        // `_emit`, which is the frame they would have been read on anyway.
        // See THE push / reading SPLIT in pipeline.hpp — it is exact, not an
        // approximation.
        (void)pipeline_.push(frame);
        ++processed_;

        if (pipeline_.presence().calibrated() && !have_calibrated_at_) {
            calibrated_at_      = processed_;
            have_calibrated_at_ = true;
        }

        breath_rms_.push(breath_filter_.process(pipeline_.last_vitals_input()));
        // Cascade: each section feeds the next, exactly as the reference's
        // `for section in self._motion_filters` loop does. The sections are
        // identical, so only consistency of order matters.
        float m = pipeline_.last_fused();
        for (BiquadBandpass& section : motion_filters_) {
            m = section.process(m);
        }
        motion_rms_.push(m);

        // `if self.compute_bnr: self._bnr_step(p)`. The `if constexpr` is what
        // keeps `accumulate_bnr` from being instantiated at all on a build with
        // no row store — it would not compile there, which is exactly the
        // guarantee wanted: the code that would touch the absent array is not
        // in the binary.
        if constexpr (kBnrCompiledIn) {
            if (bnr_on_) {
                accumulate_bnr();
            }
        }

        // `if self._processed % self._every: return None`
        if (processed_ % every_ != 0u) {
            return false;
        }
        emitted_ = emit(t);
        return true;
    }

    // -- BNR link quality ---------------------------------------------------

    void accumulate_bnr() {
        const std::size_t n_cfg = cfg_.n_subcarriers;
        const std::size_t have  = pipeline_.last_residual_count();
        const std::size_t n     = (have < n_cfg) ? have : n_cfg;
        const float*      res   = pipeline_.last_residuals();

        // `self._bnr_acc[k] += r if math.isfinite(r) else 0.0` — a plain
        // running sum of at most `bnr_dec_` terms, which is what the reference
        // writes. NOT compensated: this one is not a `sum()` call, so a
        // NeumaierSum here would be a different algorithm from the reference's
        // rather than the same one.
        for (std::size_t k = 0; k < n; ++k) {
            const float r = res[k];
            bnr_.acc[k] += is_finite(r) ? r : 0.0f;
        }
        ++bnr_n_;

        if (bnr_n_ >= bnr_dec_) {
            // `[a / self._bnr_n for a in self._bnr_acc]` — the WHOLE
            // accumulator, `n_subcarriers` wide, not just the `n` updated
            // above. A short frame leaves its tail at zero, which becomes a
            // constant column and is skipped by `subcarrier_bnr`.
            float* const  row = bnr_.rows[bnr_head_];
            const float   n_f   = static_cast<float>(bnr_n_);
            for (std::size_t k = 0; k < n_cfg; ++k) {
                row[k]       = bnr_.acc[k] / n_f;
                bnr_.acc[k]  = 0.0f;  // `self._bnr_acc = [0.0] * n_subcarriers`
            }
            bnr_head_ = (bnr_head_ + 1u) % bnr_cap_;
            if (bnr_size_ < bnr_cap_) {
                ++bnr_size_;
            }
            bnr_n_ = 0;
        }

        // `if self._processed % self._bnr_every == 0 and len(rows) == maxlen`
        if (processed_ % bnr_every_ == 0u && bnr_size_ == bnr_cap_) {
            score_bnr();
        }
    }

    void score_bnr() {
        // Oldest-to-newest row pointers; `subcarrier_bnr_rows` reads them in
        // place, so nothing is copied and no second window buffer exists.
        const std::size_t oldest = (bnr_head_ + bnr_cap_ - bnr_size_) % bnr_cap_;
        for (std::size_t i = 0; i < bnr_size_; ++i) {
            bnr_.ptrs[i] = bnr_.rows[(oldest + i) % bnr_cap_];
        }

        SubcarrierBnrParams params;
        params.f_low  = kBreathBandLowHz;
        params.f_high = kBreathBandHighHz;

        const SubcarrierBnrResult r =
            subcarrier_bnr_rows(bnr_.ptrs, bnr_size_, cfg_.n_subcarriers, bnr_rate_,
                                bnr_.out, MaxSubcarriers, bnr_.work.re, bnr_.work.im,
                                bnr_.work.capacity(), params);
        if (!r.ok) {
            // A port-level refusal (a buffer too small), never one of the
            // reference's own guards — those come back ok with zeros. Leave
            // `best_bnr` as it was rather than inventing a 0 that would read
            // as "this link went blind".
            return;
        }

        // `finite = [b for b in bnr if math.isfinite(b)]; max(finite) if finite
        //  else 0.0` — note that +inf is EXCLUDED, not treated as the best
        //  possible link. bnr.hpp explains why it is +inf in the first place.
        float best       = 0.0f;
        bool  have_any   = false;
        for (std::size_t k = 0; k < cfg_.n_subcarriers; ++k) {
            const float b = bnr_.out[k];
            if (!is_finite(b)) {
                continue;
            }
            if (!have_any || b > best) {
                best     = b;
                have_any = true;
            }
        }
        best_bnr_ = have_any ? best : 0.0f;
    }

    // -- link statistics ----------------------------------------------------

    std::size_t arrivals_index(std::size_t i) const {
        return (arrivals_head_ + MaxArrivals - arrivals_size_ + i) % MaxArrivals;
    }

    void arrivals_push(double t, std::uint32_t seq) {
        if (arrivals_size_ == MaxArrivals) {
            // See `arrival_overflows`. Dropping the OLDEST keeps the window a
            // (shorter) suffix of the reference's, so the rate stays a rate;
            // dropping the newest would make the loss figure a property of
            // this buffer.
            arrivals_pop_front();
            ++arrival_overflows;
        }
        arrivals_t_[arrivals_head_]   = t;
        arrivals_seq_[arrivals_head_] = seq;
        arrivals_head_                = (arrivals_head_ + 1u) % MaxArrivals;
        ++arrivals_size_;
    }

    void arrivals_pop_front() {
        if (arrivals_size_ > 0) {
            --arrivals_size_;
        }
    }

    /// `_link_stats`: achieved CSI rate (Hz x 100) and sequence loss (per
    /// mille) over the arrivals still inside the 10 s window.
    ///
    /// DOUBLE throughout, under the time-accumulator exception: `span` is a
    /// difference of two wall-clock timestamps and the whole point of the
    /// window is that it stays meaningful after days of uptime.
    void link_stats(std::int32_t* rate_chz, std::int32_t* loss_permille) const {
        if (arrivals_size_ < 2) {
            *rate_chz      = 0;
            *loss_permille = 0;
            return;
        }
        const std::size_t   i_first = arrivals_index(0);
        const std::size_t   i_last  = arrivals_index(arrivals_size_ - 1u);
        const double        t_first = arrivals_t_[i_first];
        const double        t_last  = arrivals_t_[i_last];
        const std::uint32_t s_first = arrivals_seq_[i_first];
        const std::uint32_t s_last  = arrivals_seq_[i_last];

        const double span = t_last - t_first;
        const double rate =
            (span > 0.0) ? (static_cast<double>(arrivals_size_ - 1u) / span) : 0.0;

        // `expected = ((s_last - s_first) % (1 << 32)) + 1` — the modulus is
        // the natural wrap of uint32, so a counter that rolled over mid-window
        // still gives the right span. +1 because both endpoints are inclusive.
        const std::uint64_t expected =
            static_cast<std::uint64_t>(static_cast<std::uint32_t>(s_last - s_first)) + 1u;
        const std::uint64_t received = static_cast<std::uint64_t>(arrivals_size_);
        double              loss     = 0.0;
        if (expected >= received) {
            loss = 1.0 - static_cast<double>(received) / static_cast<double>(expected);
            if (loss < 0.0) {
                loss = 0.0;  // `max(0.0, ...)`
            }
        }
        *rate_chz      = detail::to_i32(detail::round_half_even(rate * 100.0));
        *loss_permille = detail::to_i32(detail::round_half_even(loss * 1000.0));
    }

    // -- output -------------------------------------------------------------

    FeatureFrame emit(double t) {
        // THE ONE PLACE THE SPECTRAL ESTIMATES ARE PAID FOR. See pipeline.hpp.
        const VitalReading reading = pipeline_.reading();
        auto&              presence = pipeline_.presence();

        std::uint8_t flags = 0;
        if (presence.calibrated()) {
            flags = static_cast<std::uint8_t>(flags | kFlagCalibrated);
        }
        if (reading.occupied) {
            flags = static_cast<std::uint8_t>(flags | kFlagOccupied);
        }
        if (cfg_.vitals_enabled) {
            flags = static_cast<std::uint8_t>(flags | kFlagVitalsEnabled);
        }
        if (cfg_.raw_mode) {
            flags = static_cast<std::uint8_t>(flags | kFlagRawMode);
        }
        if (gaps_.discontinuities > discontinuities_seen_) {
            flags = static_cast<std::uint8_t>(flags | kFlagDiscontinuity);
            discontinuities_seen_ = gaps_.discontinuities;
        }

        const bool breathing_usable =
            reading.respiratory_rate.status == VitalStatus::Valid ||
            reading.respiratory_rate.status == VitalStatus::Degraded;
        const bool heart_usable = reading.heart_rate.status == VitalStatus::Valid ||
                                  reading.heart_rate.status == VitalStatus::Degraded;
        const bool breathing_ok = cfg_.vitals_enabled && reading.occupied && breathing_usable;
        const bool heart_ok     = breathing_ok && cfg_.heart_enabled && heart_usable;
        if (breathing_ok) {
            flags = static_cast<std::uint8_t>(flags | kFlagBreathingValid);
        }
        if (heart_ok) {
            flags = static_cast<std::uint8_t>(flags | kFlagHeartValid);
        }

        std::int32_t rate_chz = 0;
        std::int32_t loss_pm  = 0;
        link_stats(&rate_chz, &loss_pm);

        // THE GATE. `vit` and `heart` decide what is even computed into the
        // frame, exactly as the reference does: the zero is written here, on
        // the node, not filtered out later. ADR-001, ADR-005, spec/wire.md
        // "Gating rule".
        const bool vit   = cfg_.vitals_enabled;
        const bool heart = vit && cfg_.heart_enabled;

        FeatureFrame f;
        f.node_id            = cfg_.node_id;
        f.tx_id              = cfg_.tx_id;
        f.flags              = flags;
        f.seq                = seq_;
        f.uptime_ms          = static_cast<std::uint32_t>(
            static_cast<std::uint64_t>(detail::round_half_even((t - t0_) * 1000.0)) &
            0xFFFFFFFFull);
        f.csi_rate_chz       = rate_chz;
        f.loss_permille      = loss_pm;
        f.motion_energy      = reading.motion_energy;
        f.presence_threshold = presence.calibrated() ? presence.threshold() : 0.0f;
        f.gross_motion       = motion_rms_.value();
        f.breath_rms         = vit ? breath_rms_.value() : 0.0f;
        f.breathing_bpm      = vit ? reading.respiratory_rate.value_bpm : 0.0f;
        f.breathing_conf     = vit ? reading.respiratory_rate.confidence : 0.0f;
        f.heart_bpm          = heart ? reading.heart_rate.value_bpm : 0.0f;
        f.heart_conf         = heart ? reading.heart_rate.confidence : 0.0f;
        f.best_bnr           = best_bnr_;
        f.calib_age_s        = calib_age_s();
        f.rssi_dbm           = mean_or(rssi_sum_, rssi_n_, 0);
        f.noise_floor_dbm    = mean_or(noise_sum_, noise_n_, kNoiseUnknown);
        f.n_subcarriers      = static_cast<std::int32_t>(cfg_.n_subcarriers);

        seq_       = seq_ + 1u;  // `(self._seq + 1) & 0xFFFFFFFF` — uint32 wraps
        rssi_sum_  = 0;
        rssi_n_    = 0;
        noise_sum_ = 0;
        noise_n_   = 0;
        return f;
    }

    /// `int((self._processed - self._calibrated_at) / self.fs)`, or the
    /// not-calibrated sentinel. Truncation, not rounding: the field is "whole
    /// seconds since calibration completed" and the reference writes `int()`.
    std::uint32_t calib_age_s() const {
        if (!have_calibrated_at_) {
            return kNotCalibrated;
        }
        const double age = static_cast<double>(processed_ - calibrated_at_) /
                           static_cast<double>(cfg_.sample_rate_hz);
        if (!(age > 0.0)) {
            return 0;
        }
        if (age >= 4294967295.0) {
            return 0xFFFFFFFEu;  // anything but the sentinel
        }
        return static_cast<std::uint32_t>(age);
    }

    /// `int(round(sum / n)) if n else fallback`, with Python's rounding.
    static std::int32_t mean_or(std::int64_t sum, std::uint32_t n, std::int32_t fallback) {
        if (n == 0u) {
            return fallback;
        }
        return detail::to_i32(detail::round_half_even(static_cast<double>(sum) /
                                                      static_cast<double>(n)));
    }

    // -- state --------------------------------------------------------------

    FeatureExtractorConfig cfg_;
    Pipeline               pipeline_;
    GapFiller              gaps_{20};
    typename GapFiller::Arena arena_{};

    BiquadBandpass                      breath_filter_;
    BiquadBandpass                      motion_filters_[kGrossMotionSections];
    detail::RollingRms<MaxRmsSamples>   breath_rms_;
    detail::RollingRms<MaxRmsSamples>   motion_rms_;

    // BNR link quality. The store is EMPTY when MaxBnrRows == 0; the scalars
    // below stay either way, because 36 bytes of bookkeeping is not worth a
    // second specialisation and `configure()` still has to compute them to
    // refuse the configurations the reference refuses.
    detail::BnrRowStore<MaxBnrRows, MaxSubcarriers> bnr_{};
    std::size_t               bnr_head_ = 0;
    std::size_t               bnr_size_ = 0;
    std::size_t               bnr_cap_  = 8;
    std::uint32_t             bnr_dec_  = 1;
    std::uint32_t             bnr_every_ = 1;
    std::uint32_t             bnr_n_    = 0;
    float                     bnr_rate_ = 0.0f;
    float                     best_bnr_ = 0.0f;

    // Link statistics: a 10 s window of arrival timestamps and sequence
    // numbers, held as parallel arrays so the timestamps stay 8-byte aligned
    // without padding a 12-byte struct out to 16.
    double        arrivals_t_[MaxArrivals]{};
    std::uint32_t arrivals_seq_[MaxArrivals]{};
    std::size_t   arrivals_head_ = 0;
    std::size_t   arrivals_size_ = 0;

    FeatureFrame out_frames_[MaxFill + 1]{};
    FeatureFrame emitted_{};

    std::uint32_t every_      = 1;
    std::uint32_t processed_  = 0;
    std::uint32_t seq_        = 0;
    double        t0_         = 0.0;
    bool          have_t0_    = false;
    std::uint32_t calibrated_at_      = 0;
    bool          have_calibrated_at_ = false;

    std::int64_t  rssi_sum_  = 0;
    std::uint32_t rssi_n_    = 0;
    std::int64_t  noise_sum_ = 0;
    std::uint32_t noise_n_   = 0;

    std::uint32_t discontinuities_seen_ = 0;
    bool          valid_                = false;
    /// `cfg_.compute_bnr` AND a build that has somewhere to put the rows.
    /// Resolved once in `configure()` so the hot path is one bool test.
    bool          bnr_on_               = false;
};

/// The GENERIC budget — see THE TEMPLATE PARAMETERS ARE THE NODE'S RAM BUDGET
/// above before instantiating this on a part that does not have ~73 KB of SRAM
/// to spare for one link. It is what the tests and the benchmark drive,
/// because they must work at whatever sample rate a vector names.
///
/// The NODE's budget is `SaansNodeExtractor` in `edge/dsp/node_config.hpp`,
/// which is this object with every capacity cut to what the node's own
/// constants require and the proof of that written down as static_asserts.
using FeatureExtractor = BasicFeatureExtractor<>;

}  // namespace saans
