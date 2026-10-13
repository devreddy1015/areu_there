// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — presence detection by ambient calibration.
//
// Port of `wifisense/presence.py`. Header-only, no allocation after init,
// float32 throughout.
//
// No model, no training, no labelled data: measure how much the channel jitters
// in an empty room, then call anything significantly above that "occupied".
// This is the honest baseline every WiFi-sensing claim should be compared
// against, and in practice it is most of the usable signal.
//
// ---------------------------------------------------------------------------
// WHAT THIS FILE OWES THE REFERENCE
// ---------------------------------------------------------------------------
// Three design decisions in `presence.py` are load-bearing and are reproduced
// here exactly, each with the reasoning that produced them. They are not
// tuning; getting any of them wrong turns the detector into one that reports a
// plausible-looking number and is always wrong in the same direction:
//
//   1. Energy is measured on the BAND-LIMITED residual, not broadband.
//   2. A WARMUP discards input before calibration statistics accumulate.
//   3. The threshold is +inf until calibration completes.
//
// See the comments at each site.
//
#include <limits>

#include "biquad.hpp"
#include "core.hpp"

namespace saans {

// ---------------------------------------------------------------------------
// RunningVariance
// ---------------------------------------------------------------------------

// Guarded so that a later move of this type into core.hpp (where it arguably
// belongs — `wifisense/types.py` keeps it alongside CsiFrame) does not become a
// redefinition. core.hpp is shared with other stages and is not edited from
// here.
#ifndef SAANS_HAVE_RUNNING_VARIANCE
#define SAANS_HAVE_RUNNING_VARIANCE 1

/// Welford's online variance — numerically stable, O(1) per sample.
///
/// Port of `wifisense.types.RunningVariance`. The naive `E[x^2] - E[x]^2` form
/// is the obvious alternative: it forms the difference of two nearly equal
/// quantities, and Welford never does.
///
/// HOW MUCH THAT IS WORTH, MEASURED, because the honest answer depends entirely
/// on the spread of the data and an earlier draft of this comment called the
/// naive form "unusable here" without saying at what spread:
///
///   sigma/mean    naive variance error    Welford
///   8.9e-2 (the `presence` vector)  1.6e-4   3.7e-8
///   1.0e-2                          1.8e-4   8.4e-8
///   1.0e-3                          1.9e-1   1.8e-7
///   1.0e-4                          6.0e+1   8.2e-6
///
/// So on the golden vector the naive form is 25x worse end to end — the settled
/// threshold moves from 9.5e-8 to 2.4e-6 relative — and still lands 40x inside
/// the 1e-4 contract. It becomes unusable, in the ordinary sense of the word,
/// one decade further in: a calibration run in a quiet room (sigma/mean ~1e-3,
/// the condition this comment has always named as the tripwire) has the naive
/// form 19% wrong while Welford is still good to 2e-7. That row is now a test
/// — see `test_running_variance_spread` — rather than an assertion in a
/// comment, because the vector itself does not reach the regime where the
/// choice matters.
///
/// FLOAT32. The cancellation that is left lives in `delta = x - mean_`. At
/// sigma/mean = 0.089 that costs ~3.5 bits of the 24 available, leaving the
/// variance good to ~4e-8 relative over the 200-sample calibration, and the
/// dependence is on the SPREAD, not the window length: at sigma/mean ~1e-4 it
/// has cost ~13 bits and Welford itself is down to 8e-6.
struct RunningVariance {
    std::uint32_t count = 0;
    float         mean  = 0.0f;
    float         m2    = 0.0f;

    void update(float x) {
        ++count;
        const float delta = x - mean;
        mean += delta / static_cast<float>(count);
        // `(x - mean)` reads the UPDATED mean. That asymmetry is the whole
        // trick, not a typo in the reference: it is what makes the increment
        // exactly the change in the sum of squared deviations.
        m2 += delta * (x - mean);
    }

    float variance() const {
        return (count > 1u) ? m2 / static_cast<float>(count - 1u) : 0.0f;
    }

    float std_dev() const { return sqrtf(variance()); }

    void reset() {
        count = 0;
        mean  = 0.0f;
        m2    = 0.0f;
    }
};

#endif  // SAANS_HAVE_RUNNING_VARIANCE

// ---------------------------------------------------------------------------
// PresenceState
// ---------------------------------------------------------------------------

/// Presence decision plus the numbers behind it.
///
/// Same five fields as `wifisense.presence.PresenceState`, and a POD so it can
/// be returned by value from the CSI callback without touching the heap.
struct PresenceState {
    bool  occupied             = false;
    float energy               = 0.0f;
    /// +inf until calibration completes. The `presence` golden vector pins
    /// this: `threshold_before_calibration inf`.
    float threshold            = std::numeric_limits<float>::infinity();
    bool  calibrated           = false;
    float calibration_progress = 0.0f;
};

namespace detail {

/// Python's `int(v)` — truncation toward zero — then `max(..., floor_value)`.
///
/// Spelled out rather than written inline at four call sites because the three
/// sample counts below are derived this way and an off-by-one in any of them
/// moves a transition in the golden vector by a whole sample. A non-finite or
/// absurd product saturates instead of invoking undefined behaviour, which a
/// bare cast would.
///
/// TAKES A DOUBLE. The reference truncates an f64 product; this port is handed
/// float32 parameters, and vitals.hpp's "WHERE THE DERIVED COUNTS ARE
/// COMPUTED" applies here unchanged:
///
///   * `fs * seconds` (calibration, hold, an explicit warmup, the energy
///     window) is formed in DOUBLE. Both sides see the same `fs` and the same
///     duration, so there is no parameter gap to close and a float32 product
///     only adds error — and it adds it in the direction that matters. At
///     fs = 5.1 Hz, `5.0999999046325684f * 10.0f` rounds UP to exactly 51.0f
///     and truncates to 51 where the reference's 50.999999046325684
///     truncates to 50. Over 5-100 Hz in 0.1 Hz steps that was 380 of 951
///     rates wrong on all three of warmup, calibration and hold.
///
///   * The DERIVED warmup `fs * 5 / f_low` divides by a BAND EDGE, so it is
///     formed in double and rounded ONCE to float32 before truncating. The
///     double part stops `5.1f * 5.0f` rounding up to 25.5f; the final float32
///     round absorbs the 1.5e-8 representation gap between this port's `0.1f`
///     and the reference's `0.1`, which a straight double evaluation would
///     carry through as `int(999.9999850988388) == 999` against the vector's
///     1000.
///
/// A float argument would promote silently; -Wdouble-promotion makes that a
/// build error instead, so every call site states its own precision.
inline std::uint32_t truncate_count(double v, std::uint32_t floor_value) {
    if (!std::isfinite(v) || v <= static_cast<double>(floor_value)) {
        return floor_value;
    }
    if (v >= 4294967295.0) {
        return 0xFFFFFFFFu;
    }
    const std::uint32_t t = static_cast<std::uint32_t>(v);
    return (t < floor_value) ? floor_value : t;
}

}  // namespace detail

// ---------------------------------------------------------------------------
// PresenceDetector
// ---------------------------------------------------------------------------

/// Adaptive-threshold presence detector.
///
/// During calibration the room must be empty: the detector accumulates the mean
/// and standard deviation of short-term motion energy and sets its threshold at
/// `mean + sigma_mult * sigma`. After that, energy above the threshold means
/// occupied.
///
/// Hysteresis (`hold_s`) stops a motionless occupant from flickering the output
/// off between breaths — the single most common complaint about naive presence
/// detectors.
///
/// `MaxEnergySamples` is the compile-time ceiling on the energy window, in
/// samples. The reference's `deque(maxlen=int(fs * energy_window_s))` becomes a
/// member array of this size, so the node's RAM cost is visible to the linker.
/// Pick it with `required_energy_samples()` next to where the sample rate is
/// chosen, and static_assert on it; `valid()` reports the same condition at
/// runtime.
///
/// ERROR HANDLING. The library is built -fno-exceptions, so, as with
/// BiquadBandpass, a configuration that cannot be honoured leaves the detector
/// INERT rather than approximated: `valid()` is false and `process` returns a
/// not-occupied, not-calibrated state with a +inf threshold. The alternative —
/// silently shortening the energy window to whatever fits — is worse than
/// dead, because the detector then calibrates a threshold against the wrong
/// statistic and nothing downstream can tell that it did.
template <std::size_t MaxEnergySamples>
class BasicPresenceDetector {
    static_assert(MaxEnergySamples >= 2, "the energy window needs at least 2 samples");

  public:
    /// How many energy samples a given configuration needs — exactly
    /// `max(int(fs * energy_window_s), 2)` in the reference. Call it where the
    /// sample rate is chosen to size `MaxEnergySamples`; `valid()` reports the
    /// same condition after construction. Not constexpr only because it goes
    /// through `std::isfinite` to reject a nonsense product.
    static std::uint32_t required_energy_samples(float sample_rate_hz, float energy_window_s) {
        return detail::truncate_count(
            static_cast<double>(sample_rate_hz) * static_cast<double>(energy_window_s), 2u);
    }

    /// All defaults match `PresenceDetector.__init__`. A NEGATIVE `warmup_s`
    /// means "not given" — Python's `None` — and derives the warmup from
    /// `f_low`. Zero is NOT the sentinel: `warmup_s=0.0` is a caller asking for
    /// no warmup at all, which the reference honours
    /// (`max(int(fs * 0.0), 0) == 0`), and conflating the two would silently
    /// give such a caller 50 seconds of discarded input. A negative warmup is
    /// the only spelling with no meaning in the reference, which is why it is
    /// the one borrowed for the sentinel.
    /// Inert until `configure` succeeds. Declaring the convenience
    /// constructor below suppresses the implicit default one, and
    /// `SensingPipeline` holds a detector by value and configures it
    /// afterwards; without this it would have to invent a placeholder sample
    /// rate in its member initialiser list.
    BasicPresenceDetector() = default;

    explicit BasicPresenceDetector(float sample_rate_hz, float calibration_s = 30.0f,
                                   float sigma_mult = 4.0f, float energy_window_s = 8.0f,
                                   float hold_s = 10.0f, float f_low = 0.1f,
                                   float f_high = 2.0f, float warmup_s = -1.0f) {
        (void)configure(sample_rate_hz, calibration_s, sigma_mult, energy_window_s, hold_s,
                        f_low, f_high, warmup_s);
    }

    bool configure(float sample_rate_hz, float calibration_s = 30.0f, float sigma_mult = 4.0f,
                   float energy_window_s = 8.0f, float hold_s = 10.0f, float f_low = 0.1f,
                   float f_high = 2.0f, float warmup_s = -1.0f) {
        valid_           = false;
        sample_rate_hz_  = sample_rate_hz;
        sigma_mult_      = sigma_mult;
        f_low_           = f_low;
        f_high_          = f_high;
        energy_window_s_ = energy_window_s;

        // MINIMUM THRESHOLD RATIO — a floor under the threshold, derived from
        // the statistics of the energy signal rather than tuned.
        //
        // The energy is a sliding RMS over `energy_window_s`. A calibration run
        // of 30 s therefore holds only a few INDEPENDENT windows, so the sample
        // standard deviation is badly underestimated and `mean + k*sigma` lands
        // barely above the mean. Measured consequence (found by the Rust
        // end-to-end evaluation, not by inspection): the threshold sat ~13%
        // above the empty-room mean and an empty room read as occupied 36% of
        // the time.
        //
        // For band-limited noise of bandwidth B observed for T seconds, the
        // relative standard deviation of the RMS is ~1/(2*sqrt(B*T)). Taking B
        // as the equivalent noise bandwidth of the presence bandpass gives a
        // floor that scales correctly with band and window instead of a
        // magic number:
        //
        //     enbw  = (pi/2) * (f_high - f_low)
        //     ratio = 1 + sigma_mult / (2 * sqrt(enbw * energy_window_s))
        //
        // = 1.4093 at the defaults (0.1-2.0 Hz, sigma_mult 4, 8 s window).
        {
            const float enbw = 1.57079632679489662f * (f_high - f_low);
            const float bt    = enbw * energy_window_s;
            min_threshold_ratio_ =
                (bt > 0.0f) ? 1.0f + sigma_mult / (2.0f * sqrtf(bt)) : 1.0f;
        }

        // DOUBLE products — see truncate_count's note above.
        calibration_samples_ = detail::truncate_count(
            static_cast<double>(sample_rate_hz) * static_cast<double>(calibration_s), 1u);
        hold_samples_ = detail::truncate_count(
            static_cast<double>(sample_rate_hz) * static_cast<double>(hold_s), 1u);

        // WARMUP: discard the filter's startup transient before measuring
        // anything.
        //
        // A biquad centred near 0.1 Hz needs tens of seconds to settle, and the
        // upstream EMA baseline is still converging over the same period.
        // Calibrating through that transient measures the filter waking up
        // rather than the room, and sets a threshold so high that an occupied
        // room never crosses it -- the detector then reports "empty" forever,
        // which looks like a tuning problem and is not. Five periods of the
        // lowest frequency clears it.
        //
        // The two spellings differ: an explicit warmup floors at 0, the derived
        // one does not take a max at all. Kept apart because the derived form
        // can never be negative for a valid band anyway, and collapsing them
        // would hide that. They also differ in PRECISION, and that is not
        // cosmetic either — the derived form divides by a band edge, so it
        // rounds once to float32 before truncating where the explicit form does
        // not. truncate_count's note above gives the measurement for each.
        //
        // The derived form is spelled `fs * (5 / f_low)` rather than the
        // reference's `fs * 5 / f_low`. That REASSOCIATION is deliberate and
        // was chosen by measurement, not taste. "Five periods of f_low" is a
        // count of seconds; forming it once in float32 rounds the band edge's
        // 1.5e-8 representation error back onto the value the reference's f64
        // literal means — `5.0f / 0.1f` is exactly 50.0f — and the remaining
        // multiply by the sample rate is then sample-rate arithmetic, which
        // belongs in double for the reason above. Swept over 5-100 Hz in
        // 0.1 Hz steps against the reference:
        //
        //   f_low                 0.05  0.08  0.1  0.125  0.2  0.25  0.4  0.5
        //   fs*5/f_low all-f32     380    96  380    380  190   380   96  380
        //   fs*5/f_low all-f64     233    96  233    380  116   380   59  380
        //   fs*(5/f_low), as here    0     0    0      0    0     0    0    0
        //
        // RESIDUAL, recorded rather than hidden: where `5 / f_low` is itself
        // not representable — f_low = 0.15, 0.3, 0.6 — nothing recovers the
        // reference's integer, because its f64 quotient and this port's float32
        // one straddle the truncation boundary from opposite sides. There the
        // warmup can be one sample out at ~18% of sample rates (168 of 951 at
        // f_low = 0.15), which shifts the whole calibration window by one
        // sample. It is in `float32_deviations`, and `spec_questions` asks for
        // the warmup to be pinned in SAMPLES in the spec, which would end the
        // ambiguity for every band edge at once.
        const float warmup_seconds_per_hz = 5.0f / f_low;
        warmup_samples_ =
            (warmup_s >= 0.0f)
                ? detail::truncate_count(
                      static_cast<double>(sample_rate_hz) * static_cast<double>(warmup_s), 0u)
                : detail::truncate_count(static_cast<double>(sample_rate_hz) *
                                             static_cast<double>(warmup_seconds_per_hz),
                                         0u);

        // Measure energy *in the band where bodies live*, not broadband.
        //
        // A still person's whole signature is a narrowband 0.1-0.5 Hz breathing
        // modulation. Sensor and multipath noise is broadband, and typically
        // larger. So broadband RMS barely separates an occupied room from an
        // empty one and the occupancy flag chatters -- the single most visible
        // failure of a naive detector. Bandpassing first throws away the noise
        // that carries no information about presence and leaves the part that
        // does. Measured on the reference, this took the separation between an
        // empty and an occupied room from ~1.1x to ~4x.
        //
        // The upper edge reaches past breathing to 2 Hz so that gross motion
        // (walking, gestures) registers too, not just a sleeping subject.
        if (!bandpass_.configure(f_low, f_high, sample_rate_hz)) {
            return false;
        }

        energy_window_cap_ = required_energy_samples(sample_rate_hz, energy_window_s);
        if (energy_window_cap_ > MaxEnergySamples) {
            // See ERROR HANDLING above: inert, never silently shortened.
            return false;
        }

        reset_state();
        valid_ = true;
        return true;
    }

    bool valid() const { return valid_; }

    /// Feed one fused residual sample and get the current decision.
    PresenceState process(float residual) {
        if (!valid_) {
            return PresenceState{};
        }

        energy_window_.push(bandpass_.process(residual));

        // RMS of the band-limited residual is the motion energy proxy.
        const float energy = window_rms();

        if (warmed_ < warmup_samples_) {
            ++warmed_;
            return PresenceState{false, energy, threshold_, false, 0.0f};
        }

        if (!calibrated_) {
            calib_.update(energy);
            if (calib_.count >= calibration_samples_) {
                // max(mean + k*sigma, mean*ratio, 1e-6) — the reference's
                // three-way max. The middle term is the derived floor; see
                // min_threshold_ratio_ in configure().
                const float t_sigma = calib_.mean + sigma_mult_ * calib_.std_dev();
                const float t_floor = calib_.mean * min_threshold_ratio_;
                float t = (t_sigma > t_floor) ? t_sigma : t_floor;
                if (!(t > 1e-6f)) {
                    t = 1e-6f;
                }
                threshold_  = t;
                calibrated_ = true;
            }
            // NOTE the ordering, which the golden vector pins and which is easy
            // to "tidy" into a bug: the sample that COMPLETES calibration still
            // returns `calibrated = false`, while already carrying the finite
            // threshold it just computed. On `presence` that is sample 1199 for
            // the threshold and 1200 for the flag.
            return PresenceState{false, energy, threshold_, false,
                                 static_cast<float>(calib_.count) /
                                     static_cast<float>(calibration_samples_)};
        }

        if (energy > threshold_) {
            samples_since_trigger_ = 0;
        } else if (samples_since_trigger_ < hold_samples_) {
            // DELIBERATE DEVIATION, behaviour-preserving. The reference
            // increments without bound. Once the counter has reached
            // `hold_samples_` the comparison below can no longer change, so
            // saturating there is the same function of the input -- and it
            // removes a uint32 overflow that would, after ~6.8 years at 20 Hz,
            // silently report an empty room as occupied for one hold period.
            ++samples_since_trigger_;
        }

        const bool occupied = samples_since_trigger_ < hold_samples_;
        return PresenceState{occupied, energy, threshold_, true, 1.0f};
    }

    bool  calibrated() const { return calibrated_; }
    float threshold() const { return threshold_; }

    /// Seconds of input discarded before calibration begins.
    float warmup_s() const {
        return (sample_rate_hz_ > 0.0f)
                   ? static_cast<float>(warmup_samples_) / sample_rate_hz_
                   : 0.0f;
    }

    /// Reported back the way the reference's generator reads it
    /// (`d.hold_samples / fs`), so a rounded `hold_s` round-trips.
    float hold_s() const {
        return (sample_rate_hz_ > 0.0f)
                   ? static_cast<float>(hold_samples_) / sample_rate_hz_
                   : 0.0f;
    }

    float         sample_rate_hz() const { return sample_rate_hz_; }
    float         sigma_mult() const { return sigma_mult_; }
    float         f_low() const { return f_low_; }
    float         f_high() const { return f_high_; }
    float         energy_window_s() const { return energy_window_s_; }
    std::uint32_t calibration_samples() const { return calibration_samples_; }
    std::uint32_t hold_samples() const { return hold_samples_; }
    std::uint32_t warmup_samples() const { return warmup_samples_; }
    std::uint32_t energy_window_capacity() const { return energy_window_cap_; }

    /// Start calibration over. The room must be empty again.
    void recalibrate() {
        bandpass_.reset();
        reset_state();
    }

  private:
    void reset_state() {
        calib_.reset();
        threshold_             = std::numeric_limits<float>::infinity();
        calibrated_            = false;
        energy_window_.clear();
        warmed_                = 0;
        samples_since_trigger_ = hold_samples_;
    }

    /// `sqrt(sum(v * v for v in window) / n)`, compensated.
    ///
    /// This is `saans::rolling_rms` (core.hpp) with the same arithmetic in the
    /// same order, written out rather than called because the ring's storage
    /// wraps and linearising 160 floats on every CSI frame to get a flat view
    /// would cost more than the sum itself. `RingBuffer::operator[]` is
    /// oldest-to-newest, which is the order `sum()` walks the deque in, and
    /// float addition is not associative, so that correspondence is required
    /// rather than cosmetic.
    ///
    /// COMPENSATED, because that is the ALGORITHM the reference runs, not
    /// because of the drift size. CPython >= 3.12 compensates `sum()` over
    /// floats, and the reference leans on it hard enough to re-sum the whole
    /// window on every read rather than keep a running total, specifically so
    /// the two implementations cannot drift apart. Matching it is the only way
    /// this port can claim to be the same function.
    ///
    /// AND THE DRIFT IS SMALL, which is worth saying plainly because an earlier
    /// draft of this comment claimed "a naive float32 total over 160 squares
    /// drifts ~1.5e-6 relative" and that number is wrong for this window.
    /// Measured on the `presence` vector, swapping the compensation for a plain
    /// `acc += v * v` moves the worst per-sample energy deviation from 8.62e-07
    /// to 9.34e-07 and the worst threshold deviation from 9.54e-08 to 1.31e-07
    /// — a ~1e-7 effect, an order of magnitude smaller than was claimed and
    /// three orders inside the contract. The 160 squares are all the same sign
    /// and within a decade of each other, which is the easy case for a naive
    /// total; the compensation earns its keep on algorithm parity and on a
    /// window that is not this well conditioned, not on this vector's numbers.
    /// `test_presence_rms_compensation` now measures both and asserts the
    /// direction, so the claim is checked rather than asserted in a comment.
    float window_rms() const {
        // Only the newest `energy_window_cap_` samples are the reference's
        // deque; the ring itself may be larger because its capacity is a
        // compile-time budget, not the configured window.
        const std::size_t held = energy_window_.size();
        const std::size_t n    = (held < energy_window_cap_)
                                     ? held
                                     : static_cast<std::size_t>(energy_window_cap_);
        if (n == 0) {
            return 0.0f;  // `if n else 0.0` in the reference
        }
        const std::size_t base = held - n;

        NeumaierSum acc;
        for (std::size_t i = 0; i < n; ++i) {
            const float v = energy_window_[base + i];
            acc.add(v * v);
        }
        const float mean_sq = acc.value() / static_cast<float>(n);
        // A sum of squares cannot be negative; the guard only stops a
        // compensation underflow from becoming a NaN. sqrtf(0) == 0 either way,
        // so this agrees with the reference's unguarded math.sqrt.
        return sqrtf(mean_sq < 0.0f ? 0.0f : mean_sq);
    }

    BiquadBandpass                       bandpass_;
    RingBuffer<float, MaxEnergySamples>  energy_window_;
    RunningVariance                      calib_;

    float sample_rate_hz_  = 0.0f;
    float sigma_mult_      = 0.0f;
    /// Derived floor on the threshold, as a multiple of the calibration mean.
    /// 1.0 means "no floor"; see configure().
    float min_threshold_ratio_ = 1.0f;
    float f_low_           = 0.0f;
    float f_high_          = 0.0f;
    float energy_window_s_ = 0.0f;
    float threshold_       = std::numeric_limits<float>::infinity();

    std::uint32_t calibration_samples_   = 1;
    std::uint32_t hold_samples_          = 1;
    std::uint32_t warmup_samples_        = 0;
    std::uint32_t energy_window_cap_     = 2;
    std::uint32_t warmed_                = 0;
    std::uint32_t samples_since_trigger_ = 1;

    bool calibrated_ = false;
    bool valid_      = false;
};

/// The default budget: 256 energy samples, which covers the reference's 8 s
/// window at any sample rate up to 32 Hz (the node runs at 20). A different
/// rate or window takes `BasicPresenceDetector<N>` directly.
using PresenceDetector = BasicPresenceDetector<256>;

}  // namespace saans
