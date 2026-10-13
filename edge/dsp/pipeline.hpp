// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — the stages wired together.
//
// Port of `wifisense/pipeline.py`. Header-only, no allocation after init,
// float32 throughout.
//
//     CSI frame
//       -> static suppression (per-subcarrier baseline removal + weights)
//       -> weighted fusion to a scalar
//       -> presence detector  (ambient-calibrated threshold)
//       -> breathing extractor (0.1-0.5 Hz bandpass + spectral peak)
//       -> heart rate extractor (0.8-2.0 Hz bandpass + spectral peak)
//
// Vitals are suppressed while the room reads as empty: reporting a breathing
// rate for an unoccupied room is the classic way these systems manufacture
// impressive-looking numbers out of noise.
//
// ---------------------------------------------------------------------------
// THE push / reading SPLIT — THIS IS THE ESTIMATE THROTTLE
// ---------------------------------------------------------------------------
// The reference's `process()` does two very different things on every frame:
//
//   1. the STATEFUL updates — suppressor, fuse, presence, and pushing one
//      sample into each vital extractor's decimating filter. O(n_subcarriers),
//      microseconds, and it must happen for every frame or the time axis is
//      wrong.
//
//   2. the SPECTRAL ESTIMATES — `band_peak` over 73 candidate frequencies and
//      `sideband_peak` over 289, each a Goertzel pass over the whole analysis
//      window. At the node's configuration that is ~6k and ~174k multiply-adds
//      respectively, three orders of magnitude more than (1), and it is a PURE
//      FUNCTION of the state (1) just updated.
//
// `FeatureExtractor` reads a reading twice a second and discards the other
// nineteen. Consecutive frames differ by one sample in hundreds, so computing
// nineteen estimates to throw them away is most of the node's CSI-path budget
// spent on arithmetic nobody looks at — the difference between keeping up in
// real time and not (measured: bench/bench_extractor.cpp reports both).
//
// So the two halves are separate entry points. `push(frame)` is (1) and
// `reading()` is (2). Because (2) reads only state that (1) has already
// settled, deferring it is EXACT, not an approximation: `process(frame)` is
// still spelled `push(frame); return reading();` and still matches the
// reference call for call, and a caller that wants every frame's estimate pays
// exactly what the reference pays. The Rust port (`rust/crates/saans-dsp/
// src/pipeline.rs`) makes the same split for the same reason, which is the
// second reading that confirms it is the intended shape.
//
// NOTE ON THE BRIEF. The task this port was written to describes the throttle
// as "the reference recomputes the spectral estimate only every
// `estimate_interval_s`". There is no `estimate_interval_s` in the reference —
// `grep` finds the name nowhere in `wifisense/`, `rust/` or `spec/` — and a
// time-based throttle would NOT be exact: it would hold an estimate computed
// from a stale window across frames, which the golden vectors would catch
// immediately (`features_vitals` pins `breathing_bpm` for all 336 frames). The
// split above is the throttle that is both free and exact, so it is the one
// implemented. Raised in `spec_questions`.
//
// ---------------------------------------------------------------------------
// THE SPECTRAL SCRATCH IS PER-CALL WORKING SPACE, NOT PER-LINK STATE
// ---------------------------------------------------------------------------
// `reading()` needs `kScratchFloats` floats to linearise the history into and
// to hold `sideband_peak`'s score grid — 6.4 KB at the node's configuration,
// 9.3 KB at the defaults. NOTHING in it survives the call: every read of a
// scratch float in this pipeline is preceded, within the same `reading()`, by
// the write that put it there. (`estimate()` is `const` on both extractors for
// exactly that reason; the scratch is an out-parameter, not a member it reads
// back later.)
//
// So on a node carrying several links it is the one large buffer that does NOT
// have to be multiplied by the link count. `Scratch` is the template parameter
// that says where it comes from:
//
//   OwnedScratch<N>      a member array. The default, and what the tests and
//                        the benchmark instantiate: a standalone extractor
//                        must work without the caller having to arrange an
//                        arena first.
//   SharedScratch<N,Tag> one static arena, shared by every pipeline that names
//                        the same N and Tag. This is the multi-link node.
//   BorrowedScratch      a span the caller hands over with `bind_scratch()`,
//                        for a node that wants the arena in a particular
//                        region. Binding invalidates the pipeline, so the span
//                        can never change behind a `configure()` that has
//                        already approved it.
//
// Sharing changes no value anywhere: the same samples go through the same code
// in the same order and each call rewrites what it reads.
//
// *** THE SHARING ASSUMES ONE CALLER AT A TIME. ***
//
// A shared arena is sound because a single-threaded node never has two
// pipelines inside `reading()` at once. That is the WHOLE basis of it. Two
// links being serviced from two FreeRTOS tasks — or from a task and an ISR —
// with a `SharedScratch` between them will interleave their writes into the
// same floats and produce two plausible, wrong vital estimates, with no crash
// and nothing in the output that says so. If the node ever grows a second
// sensing core, give each core its own arena (that is what `Tag` is for) or go
// back to `OwnedScratch`. Do not reach for a mutex: `reading()` is the
// expensive call, so serialising it would serialise the node.
//
// `configure()` refuses a scratch smaller than `kScratchFloats`, and a
// compile-time policy is static_assert'd as well, so an undersized arena is
// never a silent overrun. It is also never a silent SHORTENING: the vital
// extractors return Unavailable rather than analysing what fits. And because
// the one span that can change at run time can only change through
// `bind_scratch()`, which clears `valid_`, the arena `reading()` writes into
// is always one `configure()` has seen the full extent of.
//
// ---------------------------------------------------------------------------
// WHAT IS NOT PORTED: subcarrier_selection = "respirfi"
// ---------------------------------------------------------------------------
// The reference offers a second vitals path, `RespirFiFusion`, chosen by
// `subcarrier_selection="respirfi"`. `edge/dsp` has the BNR half of it
// (bnr.hpp) but not `respirfi_select` or the sign-flip averaging, so this
// pipeline implements the "variance" path only — the reference's default, the
// one every golden vector uses, and the one `SAANS_PLAN` §2 says stays off
// until it has been compared on real recordings. `selection()` reports it so
// the firmware cannot believe it asked for something it did not get.
//
#include <cstddef>
#include <cstdint>

#include "core.hpp"
#include "preprocess.hpp"
#include "presence.hpp"
#include "vitals.hpp"

namespace saans {

/// Which vitals path the pipeline runs. Only `Variance` is implemented here;
/// see the file header.
enum class SubcarrierSelection : std::uint8_t {
    Variance = 0,
    RespirFi = 1,
};

// ---------------------------------------------------------------------------
// Spectral scratch providers
// ---------------------------------------------------------------------------
//
// Each provides `data()`, `size()`, and `kStaticFloats` — the capacity when it
// is known at compile time, and 0 when it is not. `BasicSensingPipeline`
// static_asserts the first and checks the second in `configure()`, so both
// kinds answer the same question and neither can overrun.

/// `band_peak`'s and `sideband_peak`'s largest possible candidate grid.
///
/// The bound is not obvious, so it is worth stating: the default resolution is
/// `max(1 / (4 * window_s), (f_high - f_low) / 400)`, and the second term alone
/// caps the count at `400 + 1`. A caller who passes an explicit
/// `resolution_hz` finer than that can exceed it — `estimate()` then returns
/// Unavailable for a short buffer rather than scribbling, the same stance
/// `band_peak` takes.
inline constexpr std::size_t kMaxScanGrid = 401;

/// THE SCRATCH BUDGET, derived rather than guessed.
///
/// `BandVitalExtractor::estimate` wants `2 * n` floats (the linearised history,
/// then `band_peak`'s windowed copy) and the sideband form wants `2 * n + grid`
/// (its score grid cannot be elided — the confidence compares the winner
/// against the best RIVAL triplet, which is not known until the scan has
/// finished). `n` is bounded by each ring, and the grid by `kMaxScanGrid`.
///
/// A free function so that a caller can size a shared arena without first
/// having to name a pipeline type — which it cannot do, because the pipeline
/// type is the thing being parameterised by the arena.
constexpr std::size_t scratch_floats_for(std::size_t breath_history,
                                         std::size_t heart_history) {
    return (2u * breath_history > 2u * heart_history + kMaxScanGrid)
               ? 2u * breath_history
               : 2u * heart_history + kMaxScanGrid;
}

/// A member array: one scratch per pipeline. The default, because a pipeline
/// that needs nothing arranged for it is the one that cannot be misused.
template <std::size_t Floats>
struct OwnedScratch {
    static constexpr std::size_t kStaticFloats = Floats;

    float*      data() { return buf_; }
    std::size_t size() const { return Floats; }

  private:
    float buf_[Floats]{};
};

/// Distinguishes arenas that must not be shared with each other — one per
/// sensing core, if a node ever grows a second one. See the file header.
struct DefaultScratchArena;

/// ONE arena for every pipeline naming the same `Floats` and `Tag`. The
/// object itself is empty; the storage is a single static array.
///
/// SINGLE-THREADED ONLY — the file header says what goes wrong otherwise, and
/// it does not announce itself.
template <std::size_t Floats, class Tag = DefaultScratchArena>
struct SharedScratch {
    static constexpr std::size_t kStaticFloats = Floats;

    float*      data() const { return storage; }
    std::size_t size() const { return Floats; }

    /// ONE array in `.bss`, zero-initialised at startup like any other static.
    /// `inline` (C++17) is what gives the member a DEFINITION here rather than
    /// just a declaration needing an out-of-class one; every translation unit
    /// that names this instantiation then shares the same array, which is the
    /// entire point of this provider.
    static inline float storage[Floats]{};
};

template <std::size_t, std::size_t, std::size_t, std::size_t, class>
class BasicSensingPipeline;

/// A span the caller owns, for a node that wants the arena somewhere specific.
/// Unbound, or bound too short, is a `configure()` failure — see the file
/// header; never a silent overrun.
///
/// THE SPAN IS PART OF THE CONFIGURATION, which is why `bind` is private and
/// only the pipeline can reach it. It used to be public, and the pipeline
/// handed out a mutable reference to this object, so a caller could bind a
/// large arena, let `configure()` approve it, and then rebind a short one — at
/// which point `reading()` wrote past the end of it with no check at any
/// level, because `reading()` deliberately passes `kScratchFloats` rather than
/// `scratch_.size()` (it must, or a larger borrowed arena would shift a
/// refusal boundary). A comment saying "bind first" was the only thing
/// enforcing it. Now the only way in is `BasicSensingPipeline::bind_scratch`,
/// which clears `valid_`, so a rebind costs a `configure()` and the arena is
/// re-checked every time it changes.
struct BorrowedScratch {
    /// Not known at compile time, so the pipeline's static_assert stands down
    /// and the runtime check in `configure()` carries it.
    static constexpr std::size_t kStaticFloats = 0;

    float*      data() { return data_; }
    std::size_t size() const { return size_; }

  private:
    template <std::size_t, std::size_t, std::size_t, std::size_t, class>
    friend class BasicSensingPipeline;

    void bind(float* p, std::size_t n) {
        data_ = p;
        size_ = (p == nullptr) ? 0u : n;
    }

    float*      data_ = nullptr;
    std::size_t size_ = 0;
};

/// Combined output of one pipeline step — `wifisense.types.VitalReading`.
struct VitalReading {
    VitalEstimate respiratory_rate;
    VitalEstimate heart_rate;
    std::size_t   subcarrier_count = 0;
    float         signal_quality   = 0.0f;
    /// `self._samples / self.sample_rate_hz`.
    ///
    /// DOUBLE, under ADR-003's documented time-accumulator exception. The
    /// sample counter grows for the whole uptime of the node and a float32
    /// quotient stops being able to resolve one sample period at 2^24 samples
    /// — 9.7 days at 20 Hz — after which this field would advance in visible
    /// steps while the pipeline underneath it was perfectly healthy. It is
    /// computed once per reading (2 Hz in the firmware), not per frame, so the
    /// emulation cost ADR-003 is defending against is not paid in the hot path.
    double timestamp_s = 0.0;
    bool   occupied    = false;
    float  motion_energy = 0.0f;
};

/// Wires the stages together.
///
/// `MaxSubcarriers`, `BreathHistory` and `HeartHistory` are the compile-time
/// RAM budget; the defaults are the ones the stock extractors carry
/// (vitals.hpp explains how each was derived). `configure()` returns false and
/// leaves the pipeline INERT if a configuration does not fit, rather than
/// quietly analysing a shorter window — see the ERROR HANDLING notes in
/// presence.hpp and vitals.hpp, which this follows.
template <std::size_t MaxSubcarriers = 64, std::size_t BreathHistory = 192,
          std::size_t HeartHistory = 960, std::size_t MaxEnergySamples = 256,
          class Scratch = OwnedScratch<scratch_floats_for(BreathHistory, HeartHistory)>>
class BasicSensingPipeline {
    static_assert(MaxSubcarriers > 0, "MaxSubcarriers must be non-zero");

  public:
    using Breathing    = BasicBreathingExtractor<BreathHistory>;
    using Heart        = BasicHeartRateExtractor<HeartHistory>;
    using Presence     = BasicPresenceDetector<MaxEnergySamples>;
    using ScratchType  = Scratch;

    /// See `scratch_floats_for` and the file header. Kept as a member constant
    /// because callers size their own buffers from it.
    static constexpr std::size_t kMaxScanGrid   = saans::kMaxScanGrid;
    static constexpr std::size_t kScratchFloats =
        scratch_floats_for(BreathHistory, HeartHistory);

    /// A compile-time-sized arena that does not fit is a compile error here
    /// rather than an inert pipeline at boot, because the one thing the caller
    /// cannot do is notice it in the field. `kStaticFloats == 0` means the
    /// size is only known at run time; `configure()` checks that case.
    static_assert(Scratch::kStaticFloats == 0u || Scratch::kStaticFloats >= kScratchFloats,
                  "the spectral scratch is smaller than this pipeline's configuration needs; "
                  "size it with scratch_floats_for(BreathHistory, HeartHistory)");

    BasicSensingPipeline() = default;

    /// Defaults match `SensingPipeline.__init__`. A NEGATIVE window means
    /// "not given" — Python's `None` — as everywhere else in this library.
    explicit BasicSensingPipeline(std::size_t n_subcarriers, float sample_rate_hz = 20.0f,
                                  float calibration_s = 30.0f,
                                  float breathing_window_s = -1.0f,
                                  float heart_window_s     = -1.0f,
                                  bool  suppress_vitals_when_empty = true) {
        (void)configure(n_subcarriers, sample_rate_hz, calibration_s, breathing_window_s,
                        heart_window_s, suppress_vitals_when_empty);
    }

    bool configure(std::size_t n_subcarriers, float sample_rate_hz = 20.0f,
                   float calibration_s = 30.0f, float breathing_window_s = -1.0f,
                   float heart_window_s = -1.0f, bool suppress_vitals_when_empty = true,
                   SubcarrierSelection selection = SubcarrierSelection::Variance) {
        valid_                      = false;
        sample_rate_hz_             = sample_rate_hz;
        suppress_vitals_when_empty_ = suppress_vitals_when_empty;
        selection_                  = selection;

        // The reference raises ValueError for anything but these two words;
        // "respirfi" is a valid request this port cannot honour, and silently
        // running the variance path under its name would be a lie the firmware
        // could not detect. See the file header.
        if (selection != SubcarrierSelection::Variance) {
            return false;
        }
        // A BORROWED arena that was never bound, or bound too short. The
        // compile-time providers were already settled by the static_assert
        // above; this is the other half of the same promise, and it must be
        // checked HERE rather than at the first `reading()` — a pipeline that
        // pushes happily for thirty seconds and then reports Unavailable
        // forever is indistinguishable from an empty room.
        if (scratch_.data() == nullptr || scratch_.size() < kScratchFloats) {
            return false;
        }
        if (!suppressor_.configure(n_subcarriers, sample_rate_hz)) {
            return false;
        }
        if (!presence_.configure(sample_rate_hz, calibration_s)) {
            return false;
        }
        if (!breathing_.configure(0.1f, 0.5f, sample_rate_hz, breathing_window_s)) {
            return false;
        }
        if (!heart_.configure(0.8f, 2.0f, sample_rate_hz, heart_window_s)) {
            return false;
        }
        reset();
        valid_ = true;
        return true;
    }

    /// Point a `BorrowedScratch` pipeline at its arena. Only compiles for that
    /// provider; the other two own or share their storage and have nothing to
    /// bind.
    ///
    /// IT INVALIDATES THE PIPELINE, deliberately. The span is as much a part of
    /// the configuration as the sample rate, so changing it means
    /// `configure()` has to run again and re-check it — which is the whole
    /// defence against a short arena. Binding before the first `configure()`
    /// is the ordinary use and costs nothing extra; binding after one is no
    /// longer a silent overrun, it is an inert pipeline until the caller
    /// reconfigures, and `valid()` says so.
    void bind_scratch(float* p, std::size_t n) {
        valid_ = false;
        scratch_.bind(p, n);
    }

    /// The scratch provider, for reading its address or extent. There is no
    /// mutating path through here: see `bind_scratch`.
    Scratch&       scratch() { return scratch_; }
    const Scratch& scratch() const { return scratch_; }

    bool                valid() const { return valid_; }
    SubcarrierSelection selection() const { return selection_; }
    float               sample_rate_hz() const { return sample_rate_hz_; }
    bool                calibrated() const { return presence_.calibrated(); }

    // -- the per-frame half -------------------------------------------------

    /// Every stateful update for one frame; no spectral estimation.
    ///
    /// Returns the presence state, which is the only part of a reading that is
    /// cheap enough to want on every frame.
    PresenceState push(const CsiFrame& frame) {
        if (!valid_) {
            return PresenceState{};
        }
        const std::size_t n = suppressor_.process(frame, residuals_, weights_);
        const float       fused = fuse(residuals_, weights_, n);

        const PresenceState state = presence_.process(fused);

        // `vitals_input = self.selector.process(...) if self.selector else fused`.
        // The selector is always None here; see the file header.
        const float vitals_input = fused;

        last_n_            = n;
        last_fused_        = fused;
        last_vitals_input_ = vitals_input;
        last_presence_     = state;

        breathing_.push(vitals_input);
        heart_.push(vitals_input);

        ++samples_;
        return state;
    }

    // -- the on-demand half -------------------------------------------------

    /// The reading the reference would have returned for the last pushed frame.
    ///
    /// NOT const: it clobbers the scratch buffer. Nothing is allocated.
    ///
    /// With a `SharedScratch` this call is the critical section — see the file
    /// header. It is not re-entrant and it is not thread-safe; one caller at a
    /// time, across every link on the node.
    VitalReading reading() {
        VitalReading out;
        if (!valid_) {
            return out;
        }
        // The capacity passed below is `kScratchFloats`, not `scratch_.size()`:
        // a borrowed arena may be larger, and the estimators must see the same
        // bound whichever provider supplied the floats. It gates refusal only,
        // never arithmetic.
        float* const scratch = scratch_.data();

        // `hide = self.suppress_vitals_when_empty and not presence_state.occupied`
        const bool hide = suppress_vitals_when_empty_ && !last_presence_.occupied;
        if (!hide) {
            out.respiratory_rate = breathing_.estimate(scratch, kScratchFloats);
            // The heart rate stage needs the breathing frequency to cancel its
            // harmonics and sidebands, but only trust it as a reference when
            // the breathing estimate itself is sound.
            //
            // `None` becomes a non-positive hertz, which is the branch
            // `HeartRateExtractor::estimate` takes for "not supplied" — the
            // same fallback to a naive band peak the reference takes.
            const bool usable = out.respiratory_rate.status == VitalStatus::Valid ||
                                out.respiratory_rate.status == VitalStatus::Degraded;
            const float breathing_hz =
                usable ? (out.respiratory_rate.value_bpm / 60.0f) : 0.0f;
            out.heart_rate = heart_.estimate(breathing_hz, scratch, kScratchFloats);
        }

        out.subcarrier_count = last_n_;
        // `min(1.0, self.breathing.fill_fraction)` — fill_fraction already
        // clamps, so the outer min is the reference being explicit.
        const float fill    = breathing_.fill_fraction();
        out.signal_quality  = (fill < 1.0f) ? fill : 1.0f;
        out.timestamp_s     = (sample_rate_hz_ > 0.0f)
                                  ? static_cast<double>(samples_) /
                                        static_cast<double>(sample_rate_hz_)
                                  : 0.0;
        out.occupied        = last_presence_.occupied;
        out.motion_energy   = last_presence_.energy;
        return out;
    }

    /// Push one frame through every stage — the reference's `process`.
    VitalReading process(const CsiFrame& frame) {
        (void)push(frame);
        return reading();
    }

    // -- intermediates the FeatureFrame extractor reads ---------------------
    //
    // "Intermediate signals of the most recent frame, exposed for the
    //  FeatureFrame extractor (wifisense/features.py) so it never has to
    //  re-run a stage and risk drifting from what the pipeline used."
    //
    // `last_residuals()` is a view into the pipeline's own buffer and is valid
    // until the next `push`. Nothing copies it.

    const float*  last_residuals() const { return residuals_; }
    std::size_t   last_residual_count() const { return last_n_; }
    float         last_fused() const { return last_fused_; }
    float         last_vitals_input() const { return last_vitals_input_; }
    PresenceState last_presence() const { return last_presence_; }
    std::uint32_t samples() const { return samples_; }

    Presence&        presence() { return presence_; }
    const Presence&  presence() const { return presence_; }
    Breathing&       breathing() { return breathing_; }
    const Breathing& breathing() const { return breathing_; }
    Heart&           heart() { return heart_; }
    const Heart&     heart() const { return heart_; }
    const StaticSuppressor<MaxSubcarriers>& suppressor() const { return suppressor_; }

    /// Everything back to construction. `presence.recalibrate()` is the
    /// reference's spelling — the room must be empty again.
    void reset() {
        suppressor_.reset();
        presence_.recalibrate();
        breathing_.reset();
        heart_.reset();
        samples_           = 0;
        last_n_            = 0;
        last_fused_        = 0.0f;
        last_vitals_input_ = 0.0f;
        last_presence_     = PresenceState{};
    }

  private:
    StaticSuppressor<MaxSubcarriers> suppressor_;
    Presence                         presence_;
    Breathing                        breathing_;
    Heart                            heart_;

    float residuals_[MaxSubcarriers]{};
    float weights_[MaxSubcarriers]{};

    /// Per-CALL working space. With the default `OwnedScratch` this IS the
    /// array and costs `4 * kScratchFloats` per pipeline; with `SharedScratch`
    /// it is an empty object and the arena is shared by the whole node. See
    /// the file header before changing it.
    Scratch scratch_{};

    float               sample_rate_hz_             = 0.0f;
    bool                suppress_vitals_when_empty_ = true;
    SubcarrierSelection selection_                  = SubcarrierSelection::Variance;
    bool                valid_                      = false;

    std::uint32_t samples_           = 0;
    std::size_t   last_n_            = 0;
    float         last_fused_        = 0.0f;
    float         last_vitals_input_ = 0.0f;
    PresenceState last_presence_{};
};

/// The stock budget: 64 subcarriers, the stock breathing and heart rings and
/// the stock 8 s energy window, with its OWN scratch. 17976 bytes on a 64-bit
/// host, of which the 2321-float scratch is 9284 B — half the object, and the
/// half a multi-link node should be sharing. See `edge/dsp/node_config.hpp`
/// for the node's actual shape; a link that only ever sees 56 subcarriers, or
/// a node that does not run the cardiac channel, should instantiate
/// `BasicSensingPipeline<>` directly with its own numbers.
using SensingPipeline = BasicSensingPipeline<>;

}  // namespace saans
