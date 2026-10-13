// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — vital-sign extraction from CSI residuals.
//
// Port of `wifisense/vitals.py`. Header-only, no allocation after init,
// float32 throughout. Every buffer is either a member array sized by a template
// parameter or scratch handed in by the caller.
//
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "biquad.hpp"
#include "core.hpp"
#include "spectral.hpp"

namespace saans {

/// A periodogram cannot resolve a frequency it has not seen repeat. Three
/// cycles is the practical floor for a stable estimate; two is enough to detect
/// periodicity but the frequency estimate wanders badly.
inline constexpr float kMinCycles = 3.0f;

// ---------------------------------------------------------------------------
// VitalStatus / VitalEstimate
// ---------------------------------------------------------------------------

/// How much to trust a vital-sign estimate.
///
/// The underlying values are not arbitrary: they are the codes
/// `spec/golden/generate.py` writes into the `*_status` arrays, so a parity
/// test can compare the enum directly against the vector. Renumbering this
/// breaks every status comparison in the suite silently.
enum class VitalStatus : std::uint8_t {
    Unavailable = 0,
    Unreliable  = 1,
    Degraded    = 2,
    Valid       = 3,
};

/// A single vital-sign estimate in beats/breaths per minute.
struct VitalEstimate {
    float       value_bpm  = 0.0f;
    float       confidence = 0.0f;
    VitalStatus status     = VitalStatus::Unavailable;

    static VitalEstimate unavailable() { return VitalEstimate{}; }

    /// Grade an estimate by its spectral confidence.
    ///
    /// The thresholds are deliberately conservative: a band-limited periodogram
    /// peak can look convincing on pure noise, so anything under 0.40 is
    /// reported but labelled unreliable rather than hidden.
    ///
    /// FLOAT32 DEVIATION. The reference compares against the f64 literals 0.70
    /// and 0.40; `0.70f` is 0.699999988079071 and `0.40f` is 0.400000005960464.
    /// A confidence landing in those ~6e-8-wide slivers grades one step
    /// differently from the reference. It is not removable — the confidence
    /// itself is a float32 quantity carrying ~1e-6 of its own uncertainty, two
    /// orders wider than the gap — so a status disagreement should be read as
    /// "the confidence sat on the threshold", and the port's own confidence
    /// value, which IS compared numerically, is the thing to look at.
    static VitalEstimate from_confidence(float bpm, float confidence) {
        VitalStatus s = VitalStatus::Unreliable;
        if (confidence >= 0.70f) {
            s = VitalStatus::Valid;
        } else if (confidence >= 0.40f) {
            s = VitalStatus::Degraded;
        }
        return VitalEstimate{bpm, confidence, s};
    }
};

// ---------------------------------------------------------------------------
// BandVitalExtractor
// ---------------------------------------------------------------------------

/// Bandpass + decimate + spectral peak-picking over a sliding window.
///
/// The window length is derived from the *lowest* frequency in the band, not
/// chosen arbitrarily. Resolving 0.1 Hz (a 10-second breath cycle) needs at
/// least `kMinCycles * 10 s = 30 s` of history. Accepting an estimate after,
/// say, 10 seconds would be less than two cycles at the bottom of the band and
/// would produce a number that is not yet meaningful.
///
/// `MaxHistory` is the compile-time ceiling on the decimated analysis window,
/// in samples: `int(analysis_rate_hz * window_s)`. Use `required_history()` to
/// size it; `valid()` reports a configuration that does not fit.
///
/// NO EXCEPTIONS, SO NO ValueError. The reference raises when `window_s` is
/// shorter than `min_window_s`. Here `configure` returns false and leaves the
/// extractor INERT — `push` is a no-op and `estimate` is `unavailable()` —
/// because an extractor running on a too-short window still returns a
/// confident-looking BPM, and that is the one failure mode worse than silence.
///
/// ---------------------------------------------------------------------------
/// WHERE THE DERIVED COUNTS ARE COMPUTED, AND WHY IT IS SPLIT
/// ---------------------------------------------------------------------------
/// `decimation`, `max_samples` and `min_samples` are `int()` truncations. An
/// off-by-one in any of them moves the readiness boundary or the analysis
/// window by a whole decimated sample, which is a different vital sign, not a
/// rounding difference. The reference truncates an f64 product; this port is
/// handed float32 parameters, and the two ways of closing that gap pull in
/// OPPOSITE directions, so each site picks the one that matches:
///
///   * SAMPLE-RATE products (`analysis_rate * window_s`) are computed in
///     DOUBLE. `sample_rate_hz` and the window lengths reach both
///     implementations as the same value, so there is no parameter gap to
///     close and every float32 rounding on the way is pure added error. At
///     25 Hz the float32 chain gives 25.0f/3 = 8.3333330154f and
///     8.3333330154f * 60.0f = 499.99997f -> 499, where the reference's
///     8.333333333333334 * 60.0 = 500.00000000000006 -> 500. This is an
///     init-time cost only, so ADR-003's float32-in-the-hot-path rule is
///     untouched — biquad.hpp records the same trade-off for its coefficients.
///
///   * BAND-EDGE quantities (`min_window_s = 3 / f_low`, `window_s`,
///     `resolution_hz`, `decimation = fs / (4 * f_high)`) stay in FLOAT32.
///     Here the gap is parameter REPRESENTATION, not arithmetic: this port's
///     `0.1f` is 0.100000001490116, the reference's `0.1` is 0.1, and
///     `3.0 / 0.1` is exactly 30.0 in f64 while `3.0 / 0.100000001490116` is
///     29.999999552965164. Float32's coarser grid rounds that back onto 30.0f;
///     a double evaluation carries the 1.5e-8 error through and
///     `int(2.0 * 44.99999932944774)` is 89 where the reference says 90 — it
///     would break the `extractors` vector outright. The rounding is only
///     absorbing an error smaller than half a float32 ulp (6e-8 relative); it
///     is not making the arithmetic more accurate, and it is recorded in
///     `float32_deviations` as the knife edge it is.
///
/// The same split is applied in presence.hpp. `scan_steps` (spectral.hpp) is a
/// third site on the same knife edge; see THE SCAN GRID below.
template <std::size_t MaxHistory>
class BandVitalExtractor {
    static_assert(MaxHistory >= 2, "analysis history needs at least 2 samples");

  public:
    /// How many decimated samples a configuration needs. A NEGATIVE
    /// `window_s` means "not given", exactly as `None` does in the reference.
    ///
    /// Mirrors `configure()`'s derivation operation for operation, including
    /// the double/float32 split above. It has to: this is the documented way to
    /// size `MaxHistory`, so an answer derived any other way sizes the ring for
    /// a window the extractor will not use, and `valid()` then comes back
    /// false for a configuration the caller was told would fit.
    ///
    /// SUBCLASS DEFAULTS. `window_s < 0` resolves to THIS class's default,
    /// `min_window_s * 1.5`. A subclass with a different default window must
    /// override this (BasicHeartRateExtractor does) or callers sizing a heart
    /// extractor from it get the 5.625 s answer for a 60 s class.
    static std::uint32_t required_history(float f_low, float f_high, float sample_rate_hz,
                                          float window_s = -1.0f) {
        const float min_window_s = kMinCycles / f_low;
        const float ws           = (window_s >= 0.0f) ? window_s : min_window_s * 1.5f;
        const std::uint32_t dec  = decimation_for(f_high, sample_rate_hz);
        const double        ar =
            static_cast<double>(sample_rate_hz) / static_cast<double>(dec);
        return truncate_count(ar * static_cast<double>(ws), 1u);
    }

    BandVitalExtractor() = default;

    /// A NEGATIVE `window_s` or `resolution_hz` means "not given" — Python's
    /// `None`. Zero is NOT the sentinel, because zero is a value the reference
    /// gives a distinct meaning to and it is not the default: `window_s=0.0`
    /// is below `min_window_s` and raises, and `resolution_hz=0.0` makes
    /// `band_peak` return nothing. Both of those stay reachable here (inert
    /// and `unavailable()` respectively), which they would not be if zero were
    /// read as "use the default".
    BandVitalExtractor(float f_low, float f_high, float sample_rate_hz,
                       float window_s = -1.0f, float resolution_hz = -1.0f) {
        (void)configure(f_low, f_high, sample_rate_hz, window_s, resolution_hz);
    }

    bool configure(float f_low, float f_high, float sample_rate_hz, float window_s = -1.0f,
                   float resolution_hz = -1.0f) {
        valid_          = false;
        f_low_          = f_low;
        f_high_         = f_high;
        sample_rate_hz_ = sample_rate_hz;

        if (!(f_low > 0.0f) || !(f_high > f_low) || !(sample_rate_hz > 0.0f)) {
            return false;
        }

        min_window_s_ = kMinCycles / f_low;
        window_s_     = (window_s >= 0.0f) ? window_s : min_window_s_ * 1.5f;
        if (window_s_ < min_window_s_) {
            // "window_s=... is too short to resolve <f_low> Hz; need at least
            // <min_window_s>s (3 cycles)" — the reference's ValueError.
            return false;
        }

        // Decimate before spectral analysis. Carrying 20 Hz samples to resolve
        // a 0.5 Hz band is 10x more arithmetic than the information warrants:
        // the band scan is O(window_samples) per candidate frequency, so this
        // is the difference between a pipeline that keeps up in real time and
        // one that does not -- and it is what makes the same code viable on an
        // ESP32 at all.
        //
        // The bandpass ahead of it does the anti-alias duty. It is only a
        // single biquad (6 dB/octave), so target 4x the top of the band rather
        // than a bare Nyquist 2x to keep aliased residue well down.
        decimation_       = decimation_for(f_high, sample_rate_hz);
        analysis_rate_hz_ = sample_rate_hz / static_cast<float>(decimation_);

        // Scanning much finer than a quarter of the true resolution (1/T) buys
        // nothing -- the window length, not the grid, bounds how well two
        // frequencies can be told apart.
        //
        // THE SCAN GRID. These three numbers -- f_low, f_high, resolution_hz --
        // are the entire input to `scan_steps` (spectral.hpp), which truncates
        // `(f_high - f_low) / resolution_hz` to get the candidate count, and
        // band_peak's confidence divides the peak by the mean over exactly that
        // many candidates. spectral.hpp carries a FRAGILITY WARNING scoped to
        // "all three configurations in the spec"; this is the layer that CHOOSES
        // the three numbers, so the warning belongs here too. For the stock
        // bands the float32 subtraction and division round back onto the
        // reference's integer (72 breathing, 288 heart, both pinned in
        // test_vitals.cpp) -- but that is the same luck as the band-edge case
        // above, not a property of the arithmetic. A caller who supplies a
        // different window_s or band can land on a different count from the
        // reference: HeartRateExtractor(20.0f, 40.0f) scans 192 steps where the
        // reference scans 191, which moves the confidence by roughly 1/steps.
        // The only real fix is an integer grid count in the spec; it is raised
        // in `spec_questions` and must not be "fixed" by promoting the division
        // to double, which loses the last grid point on all three stock
        // configurations (72 -> 71, 288 -> 287) and breaks the vectors.
        if (resolution_hz >= 0.0f) {
            resolution_hz_ = resolution_hz;
        } else {
            const float by_window = 1.0f / (4.0f * window_s_);
            const float by_band   = (f_high - f_low) / 400.0f;
            resolution_hz_        = (by_window > by_band) ? by_window : by_band;
        }

        if (!filter_.configure(f_low, f_high, sample_rate_hz)) {
            return false;
        }

        // DOUBLE, deliberately — see WHERE THE DERIVED COUNTS ARE COMPUTED.
        // `analysis_rate_hz_` is kept as the float32 the spectral stage runs
        // on; the count uses the f64 ratio the reference truncates, because
        // `fs` and the decimation are bit-identical on both sides and the
        // float32 rounding would only add error.
        const double analysis_rate_d =
            static_cast<double>(sample_rate_hz) / static_cast<double>(decimation_);
        max_samples_ = truncate_count(analysis_rate_d * static_cast<double>(window_s_), 1u);
        min_samples_ =
            truncate_count(analysis_rate_d * static_cast<double>(min_window_s_), 1u);
        if (max_samples_ > MaxHistory) {
            return false;  // see NO EXCEPTIONS above: inert, never truncated
        }

        reset();
        valid_ = true;
        return true;
    }

    bool valid() const { return valid_; }

    /// Filter one sample at the full rate; keep every Nth for analysis.
    void push(float sample) {
        if (!valid_) {
            return;
        }
        const float filtered = filter_.process(sample);
        ++phase_;
        if (phase_ >= decimation_) {
            phase_ = 0;
            history_.push(filtered);
        }
    }

    /// Decimated samples currently available for analysis.
    std::size_t history_size() const {
        const std::size_t held = history_.size();
        const std::size_t cap  = static_cast<std::size_t>(max_samples_);
        return (held < cap) ? held : cap;
    }

    bool ready() const { return history_size() >= static_cast<std::size_t>(min_samples_); }

    /// How full the analysis window is, in [0, 1] — useful for a UI.
    float fill_fraction() const {
        const float f = static_cast<float>(history_size()) / static_cast<float>(min_samples_);
        return (f < 1.0f) ? f : 1.0f;
    }

    /// Scratch floats `estimate()` needs: the linearised history plus the
    /// windowed copy `band_peak` writes.
    std::size_t required_scratch() const {
        return 2u * static_cast<std::size_t>(max_samples_);
    }

    /// Current estimate, or `unavailable()` if the window is not yet full.
    ///
    /// SCRATCH. `band_peak` wants a flat array and the ring's storage wraps, so
    /// the history is linearised into the first `history_size()` floats of
    /// `scratch` — the exact analogue of `list(self._history)` in the reference
    /// — and the windowed copy goes in the second half. Nothing is allocated;
    /// `required_scratch()` says how much to supply.
    VitalEstimate estimate(float* scratch, std::size_t scratch_capacity) const {
        const std::size_t n = history_size();
        if (!valid_ || n < static_cast<std::size_t>(min_samples_)) {
            return VitalEstimate::unavailable();
        }
        if (scratch == nullptr || scratch_capacity < 2u * n) {
            // A short buffer would analyse the wrong window, which is worse
            // than analysing none.
            return VitalEstimate::unavailable();
        }
        linearise(scratch, n);

        BandPeakParams bp;
        bp.f_low         = f_low_;
        bp.f_high        = f_high_;
        bp.resolution_hz = resolution_hz_;
        bp.window        = true;

        const std::optional<SpectralPeak> peak =
            band_peak(scratch, n, analysis_rate_hz_, bp, scratch + n, scratch_capacity - n);
        if (!peak.has_value()) {
            return VitalEstimate::unavailable();
        }
        return VitalEstimate::from_confidence(peak->freq_hz * 60.0f, peak->confidence);
    }

    VitalEstimate process(float sample, float* scratch, std::size_t scratch_capacity) {
        push(sample);
        return estimate(scratch, scratch_capacity);
    }

    void reset() {
        history_.clear();
        filter_.reset();
        phase_ = 0;
    }

    float         f_low() const { return f_low_; }
    float         f_high() const { return f_high_; }
    float         sample_rate_hz() const { return sample_rate_hz_; }
    float         min_window_s() const { return min_window_s_; }
    float         window_s() const { return window_s_; }
    std::uint32_t decimation() const { return decimation_; }
    float         analysis_rate_hz() const { return analysis_rate_hz_; }
    float         resolution_hz() const { return resolution_hz_; }
    std::uint32_t max_samples() const { return max_samples_; }
    std::uint32_t min_samples() const { return min_samples_; }

  protected:
    /// Copy the newest `n` decimated samples, oldest-to-newest, into `dst`.
    ///
    /// The ring may hold more than the configured window, because its capacity
    /// is a compile-time RAM budget rather than the configuration; only the
    /// newest `max_samples_` are the reference's `deque(maxlen=...)`.
    void linearise(float* dst, std::size_t n) const {
        const std::size_t base = history_.size() - n;
        for (std::size_t i = 0; i < n; ++i) {
            dst[i] = history_[base + i];
        }
    }

    /// `int((f_high - f_low) / resolution_hz) + 1` — the sideband scan's score
    /// grid. Negative steps mean the scan will decline to run.
    std::size_t score_grid() const {
        const long steps = scan_steps(f_low_, f_high_, resolution_hz_);
        return (steps >= 0) ? static_cast<std::size_t>(steps) + 1u : 0u;
    }

    /// Python's `int(v)` then `max(v, floor)`. presence.hpp carries the same
    /// helper; it is duplicated rather than hoisted into core.hpp because
    /// core.hpp is shared with stages owned elsewhere and is not edited from
    /// here. A class-scope static rather than a second `saans::detail` free
    /// function, so the two definitions cannot collide when both headers are
    /// included.
    ///
    /// TAKES A DOUBLE, and every caller says in its own cast which precision
    /// the product was formed in — see WHERE THE DERIVED COUNTS ARE COMPUTED.
    /// A float argument would promote silently and -Wdouble-promotion would
    /// catch it, which is the point: there is no accidental spelling.
    static std::uint32_t truncate_count(double v, std::uint32_t floor_value) {
        if (!std::isfinite(v) || v <= static_cast<double>(floor_value)) {
            return floor_value;
        }
        if (v >= 4294967295.0) {
            return 0xFFFFFFFFu;
        }
        const std::uint32_t t = static_cast<std::uint32_t>(v);
        return (t < floor_value) ? floor_value : t;
    }

    /// `max(1, int(fs / (4 * f_high)))`.
    ///
    /// FLOAT32 on purpose: `f_high` is a band edge, so the gap to the reference
    /// is parameter representation rather than arithmetic and the float32
    /// rounding absorbs it. See WHERE THE DERIVED COUNTS ARE COMPUTED.
    static std::uint32_t decimation_for(float f_high, float sample_rate_hz) {
        return truncate_count(static_cast<double>(sample_rate_hz / (4.0f * f_high)), 1u);
    }

    BiquadBandpass                 filter_;
    RingBuffer<float, MaxHistory>  history_;

    float f_low_            = 0.0f;
    float f_high_           = 0.0f;
    float sample_rate_hz_   = 0.0f;
    float min_window_s_     = 0.0f;
    float window_s_         = 0.0f;
    float analysis_rate_hz_ = 0.0f;
    float resolution_hz_    = 0.0f;

    std::uint32_t decimation_  = 1;
    std::uint32_t max_samples_ = 1;
    std::uint32_t min_samples_ = 1;
    std::uint32_t phase_       = 0;

    bool valid_ = false;
};

// ---------------------------------------------------------------------------
// BreathingExtractor
// ---------------------------------------------------------------------------

/// Respiratory rate: 0.1-0.5 Hz, i.e. 6-30 breaths per minute.
///
/// Covers resting adult (12-20 BPM) through infant (up to 30) and deep sleep
/// (down to 6). Needs ~30 s of history at the low end, which at the node's
/// 20 Hz and a decimation of 10 is 60 decimated samples; the default 45 s
/// window is 90.
template <std::size_t MaxHistory>
class BasicBreathingExtractor : public BandVitalExtractor<MaxHistory> {
  public:
    /// Inert until `configure` succeeds, exactly as the base is. Needed
    /// because declaring the convenience constructor below suppresses the
    /// implicit one, and `SensingPipeline` holds these by value and configures
    /// them afterwards — without this it would have to invent a placeholder
    /// sample rate in its member initialiser list.
    BasicBreathingExtractor() = default;

    explicit BasicBreathingExtractor(float sample_rate_hz, float window_s = -1.0f,
                                     float f_low = 0.1f, float f_high = 0.5f)
        : BandVitalExtractor<MaxHistory>(f_low, f_high, sample_rate_hz, window_s) {}
};

/// 192 decimated samples: the default 45 s window at EVERY capture rate the
/// band is usable at, with the bound stated rather than asserted.
///
/// `decimation = int(fs / 2)` pins the analysis rate to the interval
/// `[2, fs/floor(fs/2))`, whose supremum is 4 Hz — approached as fs -> 4 from
/// below, where the decimation has not yet stepped to 2. So the worst case is
/// `int(4 * 45) = 180` and 192 covers the whole usable range (f_high = 0.5 Hz
/// needs fs > 1 Hz for the biquad to configure at all). The previous 128 was
/// sized for "the rate the node runs at" and went INERT between 5.7 and 6 Hz,
/// where the analysis rate approaches 3 Hz and the window wants 134 samples;
/// that is 768 bytes of SRAM to remove a silent dead band, which is the right
/// way round. At the node's 20 Hz the configured window is still 90.
using BreathingExtractor = BasicBreathingExtractor<192>;

// ---------------------------------------------------------------------------
// HeartRateExtractor
// ---------------------------------------------------------------------------

/// Heart rate: 0.8-2.0 Hz, i.e. 48-120 BPM.
///
/// Far harder than breathing. Chest-wall displacement from a heartbeat is
/// roughly a tenth of that from a breath, and because CSI amplitude is a
/// nonlinear function of displacement, breathing floods this band with
/// harmonics (k * f_b) and intermodulation sidebands (f_h +/- k * f_b). The
/// f_h + f_b sideband is frequently *stronger* than f_h itself.
///
/// So pass the breathing frequency to `estimate()` and this uses
/// `sideband_peak`, which requires a symmetric triplet and recovers the true
/// carrier. Without it, the estimator falls back to plain peak picking, which
/// on realistic input returns a sideband with no indication anything went
/// wrong.
///
/// Even with the triplet method: treat readings as provisional until you have
/// validated them against a reference monitor, on your own hardware, with your
/// own subjects. Nothing here is a medical device.
template <std::size_t MaxHistory>
class BasicHeartRateExtractor : public BandVitalExtractor<MaxHistory> {
    using Base = BandVitalExtractor<MaxHistory>;

  public:
    /// Resolving the cardiac carrier from its sidebands means resolving a
    /// spacing of one breathing period. At the slow end of the breathing band
    /// (0.1 Hz) that spacing is 0.1 Hz, needing >= 10 s of history just to
    /// separate them at all. The kMinCycles rule alone would allow 3.75 s,
    /// which is enough to see a cardiac oscillation but nowhere near enough to
    /// tell it from a sideband. 60 s is the evidence-based default from
    /// `evaluate.py`; 90-120 s measurably improves the accept rate.
    static constexpr float kDefaultWindowS = 60.0f;

    /// Inert until `configure` succeeds — see the note on
    /// `BasicBreathingExtractor`'s default constructor. Safe against the 60 s
    /// invariant below: a default-constructed extractor is not `valid()`, and
    /// the only way to make it valid is `configure`, which substitutes
    /// `kDefaultWindowS` itself.
    BasicHeartRateExtractor() = default;

    explicit BasicHeartRateExtractor(float sample_rate_hz, float window_s = -1.0f,
                                     float f_low = 0.8f, float f_high = 2.0f)
        : Base(f_low, f_high, sample_rate_hz,
               (window_s >= 0.0f) ? window_s : kDefaultWindowS) {}

    /// THE 60 s DEFAULT IS AN INVARIANT OF THE CLASS, NOT OF ITS CONSTRUCTOR.
    ///
    /// `BandVitalExtractor::configure` is public and resolves a negative
    /// `window_s` to ITS default, `min_window_s * 1.5` = 5.625 s. Inheriting it
    /// unchanged would mean that re-configuring — the only way to change the
    /// band or the sample rate after construction — silently discards the one
    /// thing this subclass exists to enforce, and unlike an undersized ring it
    /// does not go inert: it stays `valid()` and answers. Measured, at 20 Hz on
    /// a clean cardiac+breathing stream, a reconfigured instance reported
    /// 69.33 BPM / confidence 0.48 / Degraded from 56 decimated samples where
    /// the untouched one reported 66.00 BPM / 0.9999 / Valid — a 3.33 BPM error
    /// wearing a believable grade, which is the sideband-for-carrier
    /// substitution the class comment above says the 60 s default is there to
    /// prevent. The reference has no equivalent hole because `__init__` is its
    /// only entry point and it substitutes DEFAULT_WINDOW_S for None there.
    ///
    /// This OVERRIDES rather than hides anything virtual — the base is a
    /// non-polymorphic template, so a `BandVitalExtractor<N>&` reference to one
    /// of these would still reach the base version. Nothing in the library does
    /// that (there are no base-class pointers anywhere in the port) and adding
    /// a vtable to a class that lives in a static node budget to defend against
    /// a call nobody makes is the wrong trade.
    bool configure(float f_low, float f_high, float sample_rate_hz, float window_s = -1.0f,
                   float resolution_hz = -1.0f) {
        return Base::configure(f_low, f_high, sample_rate_hz,
                               (window_s >= 0.0f) ? window_s : kDefaultWindowS,
                               resolution_hz);
    }

    /// The same substitution for the sizing helper, for the same reason.
    ///
    /// The base's static resolves "not given" to the BASE default, so the
    /// inherited version answers 56 for a class that needs 600 at 20 Hz — a
    /// 10.7x undersize handed to a caller who followed the header's own
    /// instruction to size `MaxHistory` with it. The resulting extractor is
    /// inert, and because `estimate()` returns Unavailable both while the
    /// window fills and when the object is dead, the caller cannot tell the two
    /// apart without calling `valid()`.
    static std::uint32_t required_history(float f_low, float f_high, float sample_rate_hz,
                                          float window_s = -1.0f) {
        return Base::required_history(f_low, f_high, sample_rate_hz,
                                      (window_s >= 0.0f) ? window_s : kDefaultWindowS);
    }

    /// Linearised history + `sideband_peak`'s windowed copy + its score grid.
    /// The grid cannot be elided the way `band_peak`'s periodogram was: the
    /// confidence compares the winner against the best RIVAL triplet, which is
    /// not known until the scan has finished.
    std::size_t required_scratch() const {
        return Base::required_scratch() + Base::score_grid();
    }

    /// Estimate heart rate, optionally suppressing breathing artefacts.
    ///
    /// `breathing_hz` should come from the breathing stage. A value of zero or
    /// less means "not supplied" and is the same branch the reference takes for
    /// `None or <= 0.0`: it degrades to the naive band peak, and the result
    /// should be treated with corresponding suspicion.
    VitalEstimate estimate(float breathing_hz, float* scratch,
                           std::size_t scratch_capacity) const {
        const std::size_t n = Base::history_size();
        if (!Base::valid() || n < static_cast<std::size_t>(Base::min_samples())) {
            return VitalEstimate::unavailable();
        }
        if (breathing_hz <= 0.0f) {
            return Base::estimate(scratch, scratch_capacity);
        }

        const std::size_t grid = Base::score_grid();
        if (scratch == nullptr || grid == 0u || scratch_capacity < 2u * n + grid) {
            return VitalEstimate::unavailable();
        }
        Base::linearise(scratch, n);

        SidebandPeakParams sp;
        sp.f_low           = Base::f_low();
        sp.f_high          = Base::f_high();
        sp.modulator_hz    = breathing_hz;
        sp.resolution_hz   = Base::resolution_hz();
        sp.harmonic_tol_hz = 0.03f;  // the reference's default; vitals.py never overrides it

        const std::optional<SpectralPeak> peak =
            sideband_peak(scratch, n, Base::analysis_rate_hz(), sp, scratch + n, n,
                          scratch + 2u * n, scratch_capacity - 2u * n);
        if (!peak.has_value()) {
            return VitalEstimate::unavailable();
        }
        return VitalEstimate::from_confidence(peak->freq_hz * 60.0f, peak->confidence);
    }

    /// No breathing frequency available — the naive band peak.
    VitalEstimate estimate(float* scratch, std::size_t scratch_capacity) const {
        return estimate(0.0f, scratch, scratch_capacity);
    }

    VitalEstimate process(float sample, float breathing_hz, float* scratch,
                          std::size_t scratch_capacity) {
        Base::push(sample);
        return estimate(breathing_hz, scratch, scratch_capacity);
    }
};

/// 960 decimated samples: the 60 s default at EVERY capture rate, with the
/// bound derived rather than hoped for.
///
/// The old 640 was justified as "600 at the node's 20 Hz, with headroom for the
/// slower analysis rates a higher capture rate produces". That reasoning is
/// backwards: `decimation = int(fs / 8)` holds the analysis rate in
/// `[8, fs/floor(fs/8))`, which is LARGEST just below each multiple of 8 — the
/// supremum is 16 Hz, approached as fs -> 16 from below. So the real worst case
/// is `int(16 * 60) = 960`, and 640 went INERT over two bands of perfectly
/// ordinary capture rates: 10.7-15.9 Hz (decimation collapses to 1, so the
/// analysis rate is the capture rate: 12 Hz needs 720, 15 Hz needs 900) and
/// 21.4-23.9 Hz (decimation 2: 22 Hz needs 660). Breathing and presence stay
/// valid at all of those, so the node kept reporting presence and breathing
/// normally while the cardiac channel went permanently silent — the shape of
/// problem SAANS_PLAN calls "looks like a tuning problem and is not".
///
/// 960 floats is 3.84 KB of the S3's internal SRAM, 1.25 KB more than 640.
///
/// THE FAILURE IS STILL QUIET, so check it at init. An extractor whose window
/// does not fit is inert, and `estimate()` returns Unavailable, which is the
/// same answer it gives while the window is filling. `valid()` is the only
/// thing that distinguishes them, and `required_history()` is how to size a
/// `BasicHeartRateExtractor<N>` for a rate outside this default. A distinct
/// status code would be louder, but the status values are the reference's wire
/// codes (see VitalStatus) and inventing a fourth is not this port's to make;
/// it is raised in `spec_questions`.
using HeartRateExtractor = BasicHeartRateExtractor<960>;

}  // namespace saans
