// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — the NODE's shape, as opposed to the library's defaults.
//
// `BasicFeatureExtractor`'s template arguments are a RAM budget, and its
// defaults are deliberately generic: they have to hold any sample rate a
// golden vector or a bench run might name, so every one of them is sized for
// the worst case of a parameter the node has already pinned. At 56 subcarriers
// and 20 Hz that costs 67,432 bytes a link on the ESP32-S3 (69,040 on a
// 64-bit host), which with ~171 KB of free DIRAM is two links and no third.
//
// This header derives the capacities from the node's OWN constants and proves
// each one with a static_assert, so the tightening is checkable rather than
// hopeful. Nothing here changes a value the DSP computes:
//
//   * A capacity BOUNDS what can be stored. If the configuration fits, the
//     same samples go through the same code in the same order and every field
//     is bit-identical — which is why `test_feature_extractor.cpp` can drive a
//     stock and a tightly-sized instantiation over the same recording and
//     require byte-identical frames. A capacity is only ever unsafe when it is
//     set BELOW what the configuration needs, and that is what the
//     static_asserts below exist to prevent. `configure()` is the second net:
//     it returns false and leaves the extractor INERT rather than quietly
//     analysing a shortened window.
//
//   * The spectral scratch is per-CALL working space, not per-link state, so
//     the node shares ONE arena across every link. See the long note in
//     pipeline.hpp — in particular that the sharing assumes a single-threaded
//     sensing path, and that a second sensing core needs a second arena.
//
// ---------------------------------------------------------------------------
// WHY THE DEFAULTS ARE NOT SIMPLY CHANGED TO THESE NUMBERS
// ---------------------------------------------------------------------------
// Because they are only right at 20 Hz. `MaxRmsSamples = 40` is correct for
// this node and wrong by a factor of 2.5 for the same code at 50 Hz, where it
// would make `configure()` refuse — loudly, but at boot, on a ceiling. A
// library default that suits exactly one deployment is a trap for whoever next
// instantiates it; the node's numbers belong at the node, which is here.
//
// ---------------------------------------------------------------------------
// THE DERIVATIONS ARE MIRRORS, AND THAT IS A LIABILITY WORTH NAMING
// ---------------------------------------------------------------------------
// `constexpr` cannot call the library's own sizing helpers
// (`BandVitalExtractor::required_history`, `required_energy_samples`,
// `FeatureExtractor::configure`): they are ordinary runtime functions, and
// `std::isfinite` is not a constant expression. So `derive` below restates
// each derivation operation for operation, including the float32/double split
// each one uses, and each restatement cites the line it mirrors.
//
// A mirror can drift from the thing it mirrors, and ONE OF THEM DID: the
// arrival window was derived as the length of the pruned window when what the
// ring has to hold is the peak BEFORE the prune, so a healthy 20 Hz link
// evicted an arrival on every frame and `arrival_overflows` — the counter
// whose whole job is to say the window has been shortened — sat permanently
// non-zero. The static_asserts below did not catch it because they compared
// each capacity against the same `derive::` call that defined it. Three things
// keep the next one from being silent:
//
//   * `configure()` re-derives everything at run time and refuses what does
//     not fit, so drift costs an INERT link rather than a wrong number — for
//     every capacity except the arrival ring, which degrades instead (see
//     `derive::arrival_peak`).
//   * the proofs in `NodeCapacities` compare against LITERALS for the node's
//     pinned configuration, not against `derive::` again.
//   * `test/test_node_config.cpp` closes the loop the only way it can be
//     closed: at run time, against the library's own sizing helpers, swept
//     over Kconfig's whole rate range — plus a steady nominal stream that
//     requires `arrival_overflows == 0`, and a field-by-field comparison of
//     the node alias against the stock instantiation over the golden
//     recording. `bench/size_report.cpp` does the same comparison for the
//     memory claims, but `make check` does not run it.
//
#include <cstddef>
#include <cstdint>

#include "feature_extractor.hpp"
#include "pipeline.hpp"
#include "presence.hpp"
#include "vitals.hpp"

namespace saans {
namespace node {

// ---------------------------------------------------------------------------
// The node's constants
// ---------------------------------------------------------------------------
//
// Every number the capacities are derived from, named once. Where the library
// already names one, this asserts against it rather than restating it, so a
// change over there surfaces here as a compile error instead of as a capacity
// that quietly no longer fits.

/// `FeatureExtractorConfig::bnr_window_s` / `bnr_rate_hz` defaults.
inline constexpr float kBnrWindowS = 30.0f;
inline constexpr float kBnrRateHz  = 4.0f;

/// `FeatureExtractorConfig::max_gap_s` default — one second of interpolation
/// before a hole becomes a discontinuity.
inline constexpr float kMaxGapS = 1.0f;

/// The breathing band and its analysis window. `BandVitalExtractor` resolves
/// a window of "not given" to `(kMinCycles / f_low) * 1.5`, which at 0.1 Hz is
/// 45 s; `SensingPipeline` passes "not given", so 45 s is what the node runs.
inline constexpr float kBreathWindowS = (3.0f / kBreathBandLowHz) * 1.5f;

/// The cardiac band, as `BasicSensingPipeline::configure` hard-codes it.
///
/// NOT assertable: they are default arguments at the call site, and a
/// static_assert cannot see one. If pipeline.hpp's `heart_.configure(0.8f,
/// 2.0f, ...)` ever changes, the heart history below silently sizes for the
/// old band — which `configure()` would then refuse at run time, because
/// `max_samples_ > MaxHistory` makes the extractor inert rather than short.
inline constexpr float kHeartBandLowHz  = 0.8f;
inline constexpr float kHeartBandHighHz = 2.0f;

/// `BasicHeartRateExtractor::kDefaultWindowS`, asserted rather than restated.
inline constexpr float kHeartWindowS = 60.0f;

/// `BasicPresenceDetector::configure`'s `energy_window_s` default. Same
/// caveat as the cardiac band: a default argument, so only a comment binds it.
inline constexpr float kEnergyWindowS = 8.0f;

static_assert(kHeartWindowS == BasicHeartRateExtractor<600>::kDefaultWindowS,
              "the cardiac analysis window moved in vitals.hpp; re-derive kHeartHistory");

// ---------------------------------------------------------------------------
// derive — compile-time mirrors of the library's runtime sizing
// ---------------------------------------------------------------------------

namespace derive {

/// Python's `int(v)` with a floor — `detail::truncate_count` (presence.hpp)
/// and `detail::truncate_floor` (feature_extractor.hpp), which are the same
/// function under two names.
///
/// The `isfinite` guard those carry is dropped: every argument here is a
/// literal expression the compiler evaluates, so a NaN would be a bug in this
/// header rather than a value arriving from a caller. The `v != v` test keeps
/// the answer defined anyway, because a wrong capacity is the one failure mode
/// this file exists to prevent.
constexpr std::uint32_t trunc_floor(double v, std::uint32_t floor_value) {
    if (v != v || v <= static_cast<double>(floor_value)) {
        return floor_value;
    }
    const std::uint32_t t = static_cast<std::uint32_t>(v);
    return (t < floor_value) ? floor_value : t;
}

/// `max(floor_value, round(v))` with PYTHON's rounding — half to EVEN, as
/// `detail::round_floor` does via `nearbyint`. For positive `v` only, which is
/// all a capacity derivation ever sees.
constexpr std::uint32_t round_floor(double v, std::uint32_t floor_value) {
    if (v != v || v <= 0.0) {
        return floor_value;
    }
    const std::uint64_t down = static_cast<std::uint64_t>(v);
    const double        frac = v - static_cast<double>(down);
    std::uint64_t       r    = down;
    if (frac > 0.5) {
        r = down + 1u;
    } else if (frac == 0.5) {
        r = (down % 2u == 0u) ? down : down + 1u;  // ties to even
    }
    return (r <= static_cast<std::uint64_t>(floor_value)) ? floor_value
                                                          : static_cast<std::uint32_t>(r);
}

/// `BandVitalExtractor::decimation_for` — `max(1, int(fs / (4 * f_high)))`,
/// FLOAT32 on purpose (vitals.hpp, WHERE THE DERIVED COUNTS ARE COMPUTED).
constexpr std::uint32_t decimation(float f_high, float sample_rate_hz) {
    return trunc_floor(static_cast<double>(sample_rate_hz / (4.0f * f_high)), 1u);
}

/// `BandVitalExtractor::required_history` — `int(analysis_rate * window_s)`
/// with the analysis rate taken as the f64 ratio, not the float32 one.
constexpr std::uint32_t band_history(float f_high, float sample_rate_hz, float window_s) {
    const std::uint32_t dec = decimation(f_high, sample_rate_hz);
    const double        ar =
        static_cast<double>(sample_rate_hz) / static_cast<double>(dec);
    return trunc_floor(ar * static_cast<double>(window_s), 1u);
}

/// `BasicPresenceDetector::required_energy_samples` — `max(2, int(fs * w))`.
constexpr std::uint32_t energy_samples(float sample_rate_hz, float window_s) {
    return trunc_floor(static_cast<double>(sample_rate_hz) * static_cast<double>(window_s),
                       2u);
}

/// `FeatureExtractor::configure`'s `_bnr_dec` and the `maxlen` it feeds:
/// `max(8, int(bnr_window_s * (fs / max(1, round(fs / bnr_rate_hz)))))`.
///
/// Note which rounding goes where — `round` for the decimation, `int` for the
/// row count. Getting that backwards at 20 Hz still gives 120; at 18 Hz it
/// gives 135 where the extractor wants 120, and the store is then a third too
/// big for no reason anyone could find later.
constexpr std::uint32_t bnr_rows(float sample_rate_hz, float window_s, float rate_hz) {
    const double        fs  = static_cast<double>(sample_rate_hz);
    const std::uint32_t dec = round_floor(fs / static_cast<double>(rate_hz), 1u);
    return trunc_floor(static_cast<double>(window_s) * (fs / static_cast<double>(dec)), 8u);
}

/// `_RollingRms(round(fs * window_s))`, with the deque's own `max(1, n)`.
constexpr std::uint32_t rms_samples(float sample_rate_hz, float window_s) {
    return round_floor(static_cast<double>(sample_rate_hz) * static_cast<double>(window_s),
                       1u);
}

/// `SequenceGapFiller(max_gap=int(max_gap_s * fs))`. `int()`, not `round()` —
/// the reference truncates here and rounds elsewhere, and `configure()`
/// refuses outright when `max_gap > MaxFill` rather than clamping, so this
/// must be the truncating one.
constexpr std::uint32_t gap_fill(float sample_rate_hz, float max_gap_s) {
    return trunc_floor(static_cast<double>(max_gap_s) * static_cast<double>(sample_rate_hz),
                       0u);
}

/// PEAK occupancy of the link-statistics ring for a steady stream at
/// `arrival_rate_hz` — not the length of the window.
///
/// Two separate `+1`s, and both are counts rather than margins:
///
///   * The window is CLOSED AT BOTH ENDS. `_link_stats` pops while
///     `t_front < t_now - 10`, so at a steady rate the entry exactly 10 s old
///     is KEPT and the trimmed window holds `rate * 10 + 1` arrivals, not
///     `rate * 10`. Sizing it at 200 for a 20 Hz node looks right and is not:
///     measured on `spec/golden/features_vitals`, a 200-entry window dropped
///     30 arrivals for capacity, which moves `csi_rate_chz` and
///     `loss_permille` off the reference's figures.
///
///   * THE PUSH PRECEDES THE PRUNE. `push_impl` calls `arrivals_push()` and
///     only then runs the `while (... < t - 10)` loop, exactly as the
///     reference appends before it pops (features.py:278-280). So the ring
///     has to hold the whole closed window PLUS the arrival that is about to
///     push the oldest one out of it. 201 at 20 Hz is therefore one short:
///     the forced-eviction branch fires on EVERY frame of a healthy, steady
///     20 Hz link (measured: 2031 overflows in 2400 frames). The emitted
///     fields survive — the forced pop is the pop the trim would have done —
///     but `arrival_overflows` is then permanently non-zero and can no longer
///     signal the condition it exists for.
///
/// Reordering push and prune would save the slot and is NOT done: the order
/// is the reference's, `t_first` is read out of the buffer, and a port that
/// reorders a window's maintenance to save 12 bytes is exactly the kind of
/// "equivalent" rewrite that turns out not to be.
constexpr std::uint32_t arrival_peak(double arrival_rate_hz, float window_s) {
    return round_floor(arrival_rate_hz * static_cast<double>(window_s), 1u) + 2u;
}

/// How much faster than its configured rate a link's arrivals are allowed to
/// run before the window starts dropping entries the trim would have kept.
///
/// NOT ROUND-UP SLACK. The count inside a fixed 10 s span is a COUNT OVER A
/// JITTERED PROCESS, so it fluctuates above its own mean even when the mean is
/// exactly nominal. Measured peak occupancy at 20 Hz over 40,000 arrivals,
/// against a nominal 202: +3 at +-10% inter-arrival jitter, +6 at +-20%
/// (which is the golden recording's own shape), +18 at +-50%. At 25% the
/// margin is 50 slots, 600 bytes a link, and covers all three with room.
///
/// WHAT IT DOES NOT COVER: a radio delivering at a different rate from the one
/// the DSP was configured for. That is not jitter, it is a misconfiguration —
/// at 50 Hz into a 20 Hz configuration the vitals bands, the gap filler and
/// the sample clock are all wrong too, and a bigger arrival ring would only
/// make the link statistics look healthy while they said nothing true. It is
/// `arrival_overflows` that reports it; see the Kconfig help for
/// SAANS_CSI_RATE_HZ, which says so where the operator sets the rate.
inline constexpr double kArrivalRateHeadroom = 1.25;

/// The capacity the node compiles in: the peak at the configured rate plus the
/// jitter headroom above.
constexpr std::uint32_t arrivals(float sample_rate_hz, float window_s) {
    return arrival_peak(static_cast<double>(sample_rate_hz) * kArrivalRateHeadroom,
                        window_s);
}

}  // namespace derive

// ---------------------------------------------------------------------------
// NodeCapacities
// ---------------------------------------------------------------------------

/// Every capacity `BasicFeatureExtractor` takes, derived for one node's
/// configuration, with the proof attached.
///
/// `SampleRateHz` is an integer because that is what Kconfig gives
/// (`CONFIG_SAANS_CSI_RATE_HZ`) and because C++17 cannot take a float as a
/// non-type template parameter. The derivations themselves still run on the
/// float the DSP will see.
template <std::size_t NSubcarriers, unsigned SampleRateHz>
struct NodeCapacities {
    static_assert(NSubcarriers > 0, "a link with no subcarriers is not a link");
    static_assert(SampleRateHz >= 10,
                  "the gross-motion band reaches 4 Hz; features.py refuses below 10 Hz");

    static constexpr float kSampleRateHz = static_cast<float>(SampleRateHz);

    static constexpr std::size_t kSubcarriers = NSubcarriers;

    static constexpr std::size_t kBnrRows =
        derive::bnr_rows(kSampleRateHz, kBnrWindowS, kBnrRateHz);

    static constexpr std::size_t kArrivals =
        derive::arrivals(kSampleRateHz, kLinkStatsWindowS);

    /// One window serves both rolling RMS objects, so it is sized for the
    /// longer of them — 2 s of breathing, not 1 s of gross motion.
    static constexpr std::size_t kRmsSamples =
        (derive::rms_samples(kSampleRateHz, kBreathRmsWindowS) >
         derive::rms_samples(kSampleRateHz, kGrossMotionWindowS))
            ? derive::rms_samples(kSampleRateHz, kBreathRmsWindowS)
            : derive::rms_samples(kSampleRateHz, kGrossMotionWindowS);

    static constexpr std::size_t kFill = derive::gap_fill(kSampleRateHz, kMaxGapS);

    static constexpr std::size_t kBreathHistory =
        derive::band_history(kBreathBandHighHz, kSampleRateHz, kBreathWindowS);

    static constexpr std::size_t kHeartHistory =
        derive::band_history(kHeartBandHighHz, kSampleRateHz, kHeartWindowS);

    static constexpr std::size_t kEnergySamples =
        derive::energy_samples(kSampleRateHz, kEnergyWindowS);

    /// The shared arena's size, for a caller declaring one by hand.
    static constexpr std::size_t kScratchFloats =
        scratch_floats_for(kBreathHistory, kHeartHistory);

    // -- the proofs ---------------------------------------------------------
    //
    // WHAT A static_assert HERE CAN AND CANNOT PROVE, SAID PLAINLY.
    //
    // Every capacity above is INITIALISED from `derive::`. Asserting it back
    // against the same `derive::` call with the same arguments — which is what
    // this block used to do — expands to `derive::x(a, b) >= derive::x(a, b)`,
    // and no value of the derivation can falsify that. It catches a capacity
    // somebody edited by hand and nothing else. In particular it cannot catch
    // the failure this file names as its own liability at the top: DRIFT
    // BETWEEN THE MIRROR AND THE LIBRARY. `derive::arrivals` was wrong by one
    // for exactly that reason and the assert passed over it.
    //
    // So the proofs below do two different jobs, and neither is a restatement:
    //
    //   1. Here, against LITERALS — the figures for the one configuration this
    //      node ships, worked out from the reference by hand and written down
    //      rather than recomputed. A `derive` that drifts stops agreeing with
    //      them and the build stops. They are guarded on the rate because they
    //      are only true at 20 Hz, and `>=` rather than `==` because rounding
    //      a capacity UP (to a power of two, say) is a RAM decision and stays
    //      legal; rounding one down is the bug.
    //
    //   2. At RUN TIME, in `test/test_node_config.cpp`, against the library's
    //      own sizing helpers — `BandVitalExtractor::required_history`,
    //      `BasicPresenceDetector::required_energy_samples`, and a `configure()`
    //      of the node alias — swept across Kconfig's whole 10..200 Hz range.
    //      `constexpr` cannot call those (they are ordinary runtime functions
    //      and `std::isfinite` is not a constant expression), which is why the
    //      mirror exists at all and why the check that closes it is a test.
    //      That test also drives a steady, exactly-nominal stream through the
    //      node's shape and requires `arrival_overflows == 0`, which is the
    //      check the arrivals defect needed and no static_assert could be.

    /// True for the configuration the literals below are worked out for.
    static constexpr bool kPinned = (SampleRateHz == 20);

    static_assert(!kPinned || kBnrRows >= 120,
                  "30 s of BNR rows at 20 Hz decimated to 4 Hz is 120; the row store is "
                  "shorter than bnr_window_s and configure() would refuse");
    static_assert(kBnrRows >= 8, "the reference's BNR window floor is 8 rows");
    static_assert(!kPinned || kArrivals >= 202,
                  "the 10 s link-statistics window at 20 Hz peaks at 202 entries — "
                  "20 * 10 closed at both ends, plus the arrival pushed before the trim "
                  "runs. Below that the ring evicts on a healthy link and "
                  "arrival_overflows stops meaning anything");
    static_assert(!kPinned || kRmsSamples >= 40,
                  "2 s of rolling RMS at 20 Hz is 40 samples (the breathing window, which "
                  "is the longer of the two sharing this capacity)");
    static_assert(!kPinned || kFill >= 20,
                  "max_gap_s = 1 s at 20 Hz is 20 frames of interpolation arena; "
                  "configure() refuses outright when max_gap exceeds it");
    static_assert(!kPinned || kBreathHistory >= 90,
                  "the breathing band decimates 20 Hz by 10, so its 45 s window is 90 "
                  "samples");
    static_assert(!kPinned || kHeartHistory >= 600,
                  "the cardiac band decimates 20 Hz by 2, so its 60 s window is 600 "
                  "samples");
    static_assert(!kPinned || kEnergySamples >= 160,
                  "the presence energy window is 8 s, which at 20 Hz is 160 samples");
    static_assert(!kPinned || kScratchFloats >= 1601,
                  "one reading() wants 2 * 600 + 401 floats at the node's cardiac history");

    // Rate-independent shape checks. These are not restatements either: they
    // are the relations the capacities have to each other and to the window
    // they serve, and each one would survive a `derive` that drifted.
    static_assert(kRmsSamples >= derive::rms_samples(kSampleRateHz, kGrossMotionWindowS),
                  "one window serves both rolling RMS objects, so it must be sized for "
                  "the LONGER of them");
    static_assert(kHeartHistory > kBreathHistory,
                  "the cardiac band decimates less than the breathing band and runs a "
                  "longer window, so its ring is the larger of the two");
    static_assert(kScratchFloats >= 2u * kHeartHistory,
                  "the spectral arena holds a linearised history and band_peak's windowed "
                  "copy of it, so it is at least twice the longest ring");
    static_assert(kArrivals >= derive::arrival_peak(static_cast<double>(kSampleRateHz),
                                                    kLinkStatsWindowS),
                  "the arrival ring must hold the peak at the NOMINAL rate before any "
                  "headroom is counted");
    static_assert(kArrivals >= 2, "link statistics need two arrivals");
};

// ---------------------------------------------------------------------------
// The node's extractor
// ---------------------------------------------------------------------------

/// Tag for the node's single spectral arena, so that `SharedScratch` for the
/// node is a distinct type from any other shared arena in the binary.
struct NodeSpectralArena;

/// The arena every link on this node computes its spectral estimates in.
///
/// ONE array, not one per link. SINGLE-THREADED: see pipeline.hpp. Two links
/// on two cores sharing this will interleave and produce two plausible wrong
/// readings with nothing in the output to say so.
template <std::size_t NSubcarriers, unsigned SampleRateHz>
using NodeArena =
    SharedScratch<NodeCapacities<NSubcarriers, SampleRateHz>::kScratchFloats,
                  NodeSpectralArena>;

/// One link on a Saans node: every capacity cut to what this node's constants
/// require, and the spectral scratch shared with every other link.
///
/// `ComputeBnr == false` drops the BNR row store from the TYPE — `MaxBnrRows`
/// becomes 0 — which is the only way to get the memory back; the runtime
/// `FeatureExtractorConfig::compute_bnr` flag alone still declares the array.
/// With it off, `best_bnr` goes out as 0.0, which spec/wire.md defines as
/// UNKNOWN rather than as a bad link, and no other field moves.
///
/// `Scratch` is parameterised only so that a measurement can instantiate the
/// owned-scratch form for comparison; the node wants the default.
template <std::size_t NSubcarriers, unsigned SampleRateHz, bool ComputeBnr = true,
          class Scratch = NodeArena<NSubcarriers, SampleRateHz>>
using NodeExtractor =
    BasicFeatureExtractor<NSubcarriers,
                          ComputeBnr ? NodeCapacities<NSubcarriers, SampleRateHz>::kBnrRows
                                     : 0u,
                          NodeCapacities<NSubcarriers, SampleRateHz>::kArrivals,
                          NodeCapacities<NSubcarriers, SampleRateHz>::kRmsSamples,
                          NodeCapacities<NSubcarriers, SampleRateHz>::kFill,
                          NodeCapacities<NSubcarriers, SampleRateHz>::kBreathHistory,
                          NodeCapacities<NSubcarriers, SampleRateHz>::kHeartHistory,
                          NodeCapacities<NSubcarriers, SampleRateHz>::kEnergySamples,
                          Scratch>;

/// THE node build: an ESP32 HT20 channel's 56 usable subcarriers at the 20 Hz
/// the firmware configures, with link quality on.
///
/// 45,408 bytes a link on the ESP32-S3, plus ONE 6,404-byte arena for the
/// whole node, against 67,432 for `BasicFeatureExtractor<56>` — three links in
/// 171 KB of free DIRAM rather than two. Built with `ComputeBnr = false` it is
/// 16,584 a link, which is ten. The firmware chooses between them with
/// `CONFIG_SAANS_COMPUTE_BNR`.
///
/// Those are TARGET figures, read out of an object file built by the
/// firmware's own toolchain: `bench/run_size_report.sh --target`. The same
/// script without `--target` prints the host figures, which are larger (8-byte
/// pointers) and are the ones every other `sizeof` in these headers quotes.
using SaansNodeExtractor = NodeExtractor<56, 20>;

/// The same node without `best_bnr`. Named so a size report and a Kconfig
/// help text can point at something.
using SaansNodeExtractorNoBnr = NodeExtractor<56, 20, false>;

}  // namespace node
}  // namespace saans
