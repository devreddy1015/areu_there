// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — static-environment suppression.
//
// Port of the first half of `wifisense/preprocess.py` (StaticSuppressor,
// corner_to_alpha, alpha_to_corner, fuse) plus the Welford accumulator
// `wifisense/types.py` calls `RunningVariance`. Header-only, no allocation,
// float32 throughout.
//
// Walls, furniture and the direct path dominate CSI amplitude by orders of
// magnitude over a breathing chest. Before anything else, subtract the slowly
// varying baseline so that what remains is the body-modulated residual.
//
// ---------------------------------------------------------------------------
// THE FLOAT32 PROBLEM THIS FILE EXISTS TO SOLVE
// ---------------------------------------------------------------------------
// Everything else in this library processes a signal that is already centred
// on zero. This stage does not: it subtracts two numbers that are deliberately
// almost equal. On the `static_suppressor` vector the amplitudes sit near 50
// and the residual near 1 — the subtraction throws away the top ~5.6 bits of
// the mantissa, so whatever error the baseline carries is amplified by ~50x
// when it lands in the residual.
//
// In f64 that is free: 1e-16 relative on a baseline of 50 is 5e-15 absolute,
// which is 5e-15 of a residual of 1. In float32 the same chain gives
// 6e-8 * 50 = 3e-6 absolute — and the reference's own update,
//
//     baseline = alpha*amp + (1 - alpha)*baseline
//
// rounds three times at the magnitude of `baseline` on every single frame.
// Transcribed literally into float32 it sits at 2.3e-5 of full scale over the
// 600-frame vector: inside the 1e-4 contract, but with only 4x of margin.
//
// WHAT THAT ERROR DOES NOT DO IS GROW. An earlier revision of this comment
// claimed it random-walks as sqrt(n) and that "a node runs for days"; that was
// wrong, and acting on it compensated the half of this stage that was already
// stable (see the WelfordVariance note below for the half that was not). The
// EMA is a contraction, b[n+1] = (1-a)*b[n] + a*x[n+1], so a rounding error
// injected into the baseline decays geometrically with time constant 1/alpha
// (~106 frames at the vector's 0.03 Hz / 20 Hz) and the accumulated error
// saturates at ~ulp(b)/sqrt(2*alpha) instead. Measured on a steady 20 Hz
// amplitude-50 stream, max |residual error| against an f64 EMA:
//
//                        n = 4,800      n = 20,000,000     ratio
//     literal            1.81e-04       2.06e-04           1.13
//     incremental        2.90e-05       4.17e-05           1.44
//     compensated        6.16e-08       6.73e-08           1.09
//
// (sqrt(n) over that 4,167x longer stream would predict 64.5.) All three
// saturate. `test_ema_soak` in the test file pins exactly that: the
// compensated form's error at 1e6 frames must be within 2x of its error at
// 1e5, which is the "does not grow with uptime" claim stated as a check.
//
// So the deviation is kept on its real and sufficient merit — a ~3000x smaller
// STEADY-STATE error on a stage whose subtraction cancels five and a half bits
// — not on a growth argument. The same three forms, measured on the golden
// vector as max |err| against the f64 reference as a fraction of the residual
// series' own full scale:
//
//     literal      baseline = a*x + (1-a)*b                     2.35e-05
//     incremental  baseline = b + a*(x - b)                     3.27e-06
//     compensated  the same, with b held as a hi+lo pair         4.65e-07
//
// The incremental form is algebraically identical and rounds once at the
// magnitude of `b` instead of three times. The compensation (`detail::
// CompensatedEma`) then keeps the ~24 bits that fall off the bottom of `b` and
// subtracts them *after* the cancellation, where they are no longer dwarfed.
// 4.65e-07 is the floor set by quantising the f64 input amplitudes to float32
// at all (3.79e-06 absolute, 4.43e-07 of full scale), so the port is now as
// close to the reference as a float32 node can be, with 200x of margin. Both
// deviations are arithmetic rearrangements of one line.
//
// They do move one BRANCH, and that is written up at `weights()`: on a
// perfectly static channel the rearranged update leaves the baseline bit-exact,
// so every residual is exactly zero and the uniform-weight fallback fires,
// where the reference's f64 round-off does not quite cancel and concentrates
// all weight on one subcarrier. Parity is not claimed in that regime; see the
// note there and `test_static_channel_divergence`.
//
// THE ACCUMULATOR THAT REALLY DOES GROW is `WelfordVariance::m2_`. It is
// ~(count-1)*variance, so it grows without bound for the whole uptime of the
// node — nothing resets the suppressor on a schedule (`pipeline.py:132`
// `reset()` is a manual API) — while the per-frame increment stays at the scale
// of the variance. Once ulp(m2)/2 exceeds a typical increment the adds
// stagnate. An earlier revision left it as a plain `+=` on the grounds that
// "Welford is already self-correcting"; Welford protects against catastrophic
// CANCELLATION, not against accumulator stagnation, and the 600-frame vector is
// 30 seconds and structurally cannot reach the stagnation regime. Measured
// against an f64 Welford on identical float32 input (8 subcarriers, 20 Hz,
// amplitudes near 50, residual std spread 2:1):
//
//                              plain +=        NeumaierSum
//     std() rel err @  1.0e6    1.20e-04         < 1e-7
//     std() rel err @  5.0e6    1.67e-03         < 1e-6
//     weights, strong-sample rel, first breach of the 1e-4 contract:
//                               frame 1,865,000 (25.9 h)     not reached
//
// So `m2_` is held in the `NeumaierSum` that core.hpp already provides and
// that this file already uses in `weights()` and `fuse()`. Welford's
// recurrence, its O(1) cost and the reference's "do not substitute a sum of
// squares" promise are all untouched; only the accumulator's bookkeeping
// changes. `mean_` needs nothing: it is a converging average, not a growing
// sum. This is a float32 deviation from the f64 reference's arithmetic and is
// recorded as one — see `test_variance_soak`, which fails for a plain `+=`.
//
// It is a mitigation, not a cure: NeumaierSum's own correction term stagnates
// in turn. Driving the shipped accumulator against an f64 Welford on a unit
// residual stream, `std()` relative error reaches 9.3e-08 at 69 h, 2.6e-05 at
// 278 h, 2.6e-04 at 556 h — so the contract is crossed somewhere in the third
// week — and 5.0e-03 at 931 h. Uncompensated the same crossing is at ~14 h, so
// this buys about a factor of 30 in uptime and no more. Bounding the
// accumulator (a windowed or periodically re-based variance) would be a
// behavioural change to the reference and is raised as a spec question rather
// than taken unilaterally here.
//
// The two places a sum is taken over a list — the weight normaliser and
// `fuse` — use NeumaierSum because CPython's `sum()` is itself compensated
// from 3.12 and the reference relies on that (see core.hpp).
//
// ---------------------------------------------------------------------------
// WHY THE WELFORD TYPE HERE IS NOT CALLED `RunningVariance`
// ---------------------------------------------------------------------------
// `wifisense/types.py` keeps one `RunningVariance` and both this stage and
// `presence.py` import it. In this tree `edge/dsp/presence.hpp` defines a
// `saans::RunningVariance` of its own, with a different interface (public
// `count`/`mean`/`m2` fields and `std_dev()`), because core.hpp — where the
// type belongs, and where presence.hpp's own guard comment anticipates it
// landing — is shared with other stages and is not edited from this module.
//
// Defining a second `saans::RunningVariance` here made including both headers
// in one translation unit a hard compile error in either order, which the
// on-node pipeline needs to do: `pipeline.py:63,83` feeds `StaticSuppressor
// .process` straight into `PresenceDetector.process`. Reusing presence.hpp's
// definition under its `SAANS_HAVE_RUNNING_VARIANCE` guard would have been the
// tidy answer, but it cannot carry the compensated accumulator above: whichever
// header a TU included first would win, so the node's variance arithmetic —
// and with it the 1e-4 contract past a day of uptime — would depend on include
// order. A silently order-dependent numeric result is worse than a duplicated
// type, so this one takes a distinct name and leaves `RunningVariance` alone.
//
// The real fix is one compensated Welford in core.hpp used by both stages.
// That is a cross-module edit and is in `spec_questions`.
//
#include <limits>

#include "core.hpp"

namespace saans {

namespace detail {

/// An EMA baseline carried as an unevaluated sum of two floats.
///
/// Why not `NeumaierSum` from core.hpp: that type deliberately exposes only
/// `value()`, the two halves already folded together and rounded back to one
/// float. Folding is exactly what must NOT happen here — the whole point is to
/// subtract the high half first (which is exact, by Sterbenz: the amplitude and
/// its baseline are within a factor of two of each other) and only then take
/// the low half off the small difference that survives. A folded baseline would
/// make `residual()` identical to the plain incremental form.
///
/// The compensation arithmetic itself is Neumaier's, for the reason core.hpp
/// gives: whichever operand is larger in magnitude keeps its bits in the sum,
/// so the correction is recovered from the other side.
struct CompensatedEma {
    float hi = 0.0f;
    float lo = 0.0f;

    void clear() {
        hi = 0.0f;
        lo = 0.0f;
    }

    /// First frame for this subcarrier: the baseline *is* the sample, exactly,
    /// so there is nothing for the low half to hold yet.
    void seed(float x) {
        hi = x;
        lo = 0.0f;
    }

    /// `x - baseline`, the reference's `amp - self._baseline[i]`.
    float residual(float x) const { return (x - hi) - lo; }

    /// `baseline = alpha*x + (1-alpha)*baseline`, rearranged to
    /// `baseline += alpha * (x - baseline)` and accumulated with compensation.
    /// See the file header for the three measurements that chose this form.
    void advance(float alpha, float r) {
        const float inc = alpha * r;
        const float t   = hi + inc;
        lo += (fabsf(hi) >= fabsf(inc)) ? ((hi - t) + inc) : ((inc - t) + hi);
        hi = t;
    }
};

}  // namespace detail

// ---------------------------------------------------------------------------
// corner_to_alpha / alpha_to_corner
// ---------------------------------------------------------------------------

/// EMA smoothing factor giving a highpass corner at `corner_hz`.
///
/// Subtracting an EMA from the signal is a first-order highpass:
///
///     p[n] = a*x[n] + (1-a)*p[n-1]
///     r[n] = x[n] - p[n-1]   =>   H(z) = (1 - z^-1) / (1 - (1-a)z^-1)
///
/// whose -3 dB corner sits at roughly `a * fs / (2*pi)` for small `a`.
/// Inverting that is the only safe way to choose `a`.
///
/// This matters more than it looks, and a fixed `alpha` is the trap. Hardcode
/// `alpha = 0.05` regardless of sample rate and at 100 Hz the corner lands near
/// 0.8 Hz, i.e. *above* the entire 0.1-0.5 Hz breathing band, attenuating the
/// very signal the next stage is trying to measure by roughly 12 dB at 0.2 Hz.
/// Tying `a` to the sample rate and to a corner below the band
/// removes that self-inflicted loss.
///
/// ERROR HANDLING. Python raises ValueError for a non-positive sample rate.
/// With -fno-exceptions there is nothing to raise, so this returns the floor of
/// the reference's own clamp, 1e-6: an EMA that never adapts, which makes the
/// residual the raw amplitude and is obviously wrong downstream rather than
/// quietly wrong. The only caller in this library, `StaticSuppressor::
/// configure`, rejects that sample rate before ever getting here.
///
/// Operation order is the reference's: ((2*pi) * corner_hz) / fs. `2.0f * kPi`
/// is exact (a power-of-two scaling), so the float32 result differs from the
/// f64 one by 2.8e-08 relative on the golden vector's 0.03 Hz at 20 Hz — the
/// representation error of pi, nothing more.
inline float corner_to_alpha(float corner_hz, float sample_rate_hz) {
    if (!(sample_rate_hz > 0.0f)) {
        return 1e-6f;
    }
    // clamp() propagates NaN where Python's min(max(...)) would too, because
    // both comparisons against an unordered value are false. Unreachable from
    // a configured suppressor; left agreeing with the reference anyway.
    return clamp(2.0f * kPi * corner_hz / sample_rate_hz, 1e-6f, 0.999f);
}

/// Inverse of `corner_to_alpha`, for reporting what a given alpha does.
///
/// Note what this is NOT: it multiplies by the sample rate and divides by the
/// constant `2*pi`, so it never divides by anything a caller controls. On a
/// zeroed alpha and sample rate it returns +0.0, not NaN. `StaticSuppressor::
/// effective_corner_hz` takes care of making "no corner configured" a value a
/// caller can tell apart from a corner of 0 Hz; this function cannot.
inline float alpha_to_corner(float alpha, float sample_rate_hz) {
    return alpha * sample_rate_hz / (2.0f * kPi);
}

// ---------------------------------------------------------------------------
// WelfordVariance
// ---------------------------------------------------------------------------

/// Welford's online variance — numerically stable, O(1) per sample.
///
/// Port of `wifisense.types.RunningVariance`, under a different name; the file
/// header says why, and `spec_questions` carries the request to hoist one
/// shared copy into core.hpp.
///
/// Used to weight subcarriers: the ones whose amplitude actually moves are the
/// ones carrying body signal.
///
/// Welford rather than a sum of squares is not a style preference. The sum of
/// squares of a series whose mean is near zero still grows without bound while
/// the variance does not, so in float32 the subtraction `E[x^2] - E[x]^2` would
/// cancel catastrophically within minutes; Welford never forms either term.
/// `count`, `mean` and `m2` are exposed because the reference exposes them and
/// because `m2` is the first thing to look at when a weight goes strange.
///
/// FLOAT32 DEVIATION. `m2_` is a `NeumaierSum`, not a bare float. It is the one
/// quantity here that grows for the whole uptime of the node, and in float32 a
/// plain `+=` stagnates: measured against an f64 Welford, `std()` is 1.2e-04
/// relative off at 1e6 samples (13.9 h at 20 Hz) and 1.7e-03 at 5e6 — past the
/// 1e-4 contract, which SAANS_PLAN §4 is explicit is a bug and not rounding.
/// Compensated, the same stream stays below 1e-6 relative throughout. The
/// recurrence, the operand order and the O(1) cost are the reference's; only
/// the accumulator's bookkeeping differs. See the file header for the full
/// table and for what this still does not fix (the contract is crossed again
/// in the third week of continuous uptime).
///
/// The accessor keeps the reference's name, `std()`, even though a member by
/// that name shadows the namespace for unqualified lookup. `std::size_t` below
/// still resolves: the name before a `::` is looked up among namespaces, types
/// and class templates only, so a member *function* called `std` is skipped.
/// Checked on g++ 16.1 and clang++ 22.1, both orderings, both inside and
/// outside member bodies — the alternative was to rename the accessor and
/// diverge from the Python for no reason.
class WelfordVariance {
  public:
    void update(float x) {
        ++count_;
        const float delta = x - mean_;
        mean_ += delta / static_cast<float>(count_);
        // The second difference uses the UPDATED mean. That is what makes this
        // Welford and not a wrong one-pass formula, so it is spelled out rather
        // than reusing `delta`. The increment is non-negative by construction:
        // the updated mean lies between the old mean and `x`, so both factors
        // carry the same sign.
        m2_.add(delta * (x - mean_));
    }

    /// Sample variance (n - 1), as the reference. One sample carries no
    /// information about spread, so it reports zero rather than dividing by it.
    float variance() const {
        return (count_ > 1u) ? m2_.value() / static_cast<float>(count_ - 1u) : 0.0f;
    }

    /// `math.sqrt(self.variance)`. The clamp mirrors `core.hpp::rolling_rms`:
    /// `m2_` is a sum of non-negative terms, so a negative variance can only
    /// come from the compensation term underflowing, and sqrtf(0) == 0 agrees
    /// with the reference either way. It stops that returning a NaN that would
    /// then poison every weight through the normaliser.
    float std() const {
        const float v = variance();
        return sqrtf(v < 0.0f ? 0.0f : v);
    }

    std::size_t count() const { return count_; }
    float       mean() const { return mean_; }
    float       m2() const { return m2_.value(); }

    void reset() {
        count_ = 0u;
        mean_  = 0.0f;
        m2_.reset();
    }

  private:
    std::size_t count_ = 0u;
    float       mean_  = 0.0f;
    NeumaierSum m2_;
};

// ---------------------------------------------------------------------------
// StaticSuppressor
// ---------------------------------------------------------------------------

/// Per-subcarrier baseline removal with variance-based weighting.
///
/// Emits a residual vector (observed minus predicted baseline) plus a weight
/// vector. Weights come from each subcarrier's running variance: subcarriers
/// that actually move carry body signal, flat ones carry noise, and weighting
/// by variance is a cheap approximation to a matched filter across the
/// subcarrier axis.
///
/// `MaxSubcarriers` sizes the per-subcarrier state as member arrays — 64 covers
/// an ESP32 HT20 channel's 56 usable subcarriers with room to spare, at
/// 64 * (2 floats + 1 bool + WelfordVariance) ~ 1.5 KB. The caller owns the
/// output buffers; nothing here allocates, ever (ADR-003).
///
/// ERROR HANDLING. Python raises ValueError for `n_subcarriers <= 0`, and
/// `corner_to_alpha` raises for a non-positive sample rate. Built
/// -fno-exceptions, `configure` reports both by returning false and leaving the
/// suppressor INERT: `valid()` is false, every accessor reads back as
/// unconfigured, and `process` and `weights` write nothing and return 0. The
/// same answer is given for `n_subcarriers > MaxSubcarriers`, which has no
/// analogue in the reference — silently clamping to the array size would
/// analyse a truncated channel and report it as a whole one, which is the kind
/// of wrong that survives review.
template <std::size_t MaxSubcarriers = 64>
class StaticSuppressor {
    static_assert(MaxSubcarriers > 0, "MaxSubcarriers must be non-zero");

  public:
    /// Inert until `configure` succeeds.
    StaticSuppressor() = default;

    /// Convenience form. Check `valid()`; there is no exception to catch.
    StaticSuppressor(std::size_t n_subcarriers, float sample_rate_hz,
                     float corner_hz = 0.03f) {
        (void)configure(n_subcarriers, sample_rate_hz, corner_hz);
    }

    /// Size the suppressor and clear its state. Returns false (and leaves it
    /// inert) for the two inputs the reference rejects, plus the one the
    /// fixed-capacity storage forces.
    ///
    /// A REJECTED configure leaves nothing of the old configuration behind.
    /// Python cannot get here — a failed `__init__` leaves no object to read —
    /// so there is no reference behaviour to match, and the alternative is
    /// worse than it looks: keeping the previous `alpha_`/`n_` would have
    /// `effective_corner_hz()` report a healthy 0.03 Hz corner on an
    /// 8-subcarrier channel for a stage that is in fact emitting nothing. A
    /// telemetry path that reads an accessor without first checking `valid()`
    /// should see an unconfigured suppressor, not a stale plausible one.
    bool configure(std::size_t n_subcarriers, float sample_rate_hz,
                   float corner_hz = 0.03f) {
        reset();
        valid_          = false;
        n_              = 0;
        sample_rate_hz_ = 0.0f;
        corner_hz_      = 0.0f;
        alpha_          = 0.0f;
        if (n_subcarriers == 0 || n_subcarriers > MaxSubcarriers) {
            return false;
        }
        if (!(sample_rate_hz > 0.0f)) {
            return false;
        }
        n_              = n_subcarriers;
        sample_rate_hz_ = sample_rate_hz;
        corner_hz_      = corner_hz;
        alpha_          = corner_to_alpha(corner_hz, sample_rate_hz);
        valid_          = true;
        return true;
    }

    bool        valid() const { return valid_; }
    std::size_t n_subcarriers() const { return n_; }
    float       sample_rate_hz() const { return sample_rate_hz_; }
    float       corner_hz() const { return corner_hz_; }
    float       alpha() const { return alpha_; }

    /// What the chosen alpha actually puts the corner at, for telemetry.
    ///
    /// NaN when the suppressor is inert, explicitly and not as a by-product of
    /// the arithmetic: `alpha_to_corner` divides only by the constant `2*pi`,
    /// so an unconfigured suppressor would otherwise report +0.0 and a caller
    /// would read that as a genuine, configured corner of 0 Hz. `valid()` is
    /// still the right predicate; this just makes the telemetry value itself
    /// un-mistakable.
    float effective_corner_hz() const {
        if (!valid_) {
            return std::numeric_limits<float>::quiet_NaN();
        }
        return alpha_to_corner(alpha_, sample_rate_hz_);
    }

    static constexpr std::size_t capacity() { return MaxSubcarriers; }

    /// Residuals and weights for one frame; returns how many of each were
    /// written, which is `min(frame.n_subcarriers, n_subcarriers())`.
    ///
    /// BOTH BUFFERS BELONG TO THE CALLER, MUST HOLD THAT MANY FLOATS, AND MUST
    /// NOT OVERLAP. Nothing is written past the returned count. The reference
    /// returns two fresh lists per frame; at 20 Hz on a node that is two
    /// allocations and two frees per frame, which is the whole reason this
    /// signature looks the way it does — and the reason the overlap question
    /// exists at all, since `residuals, weights = sup.process(frame)` cannot
    /// express it.
    ///
    /// Passing one buffer for both is rejected rather than diagnosed by
    /// corruption: the weights are computed from the variances this call just
    /// updated, so they can only be written after the residuals, and an aliased
    /// buffer would come back holding weights where the caller expects
    /// residuals, with everything downstream — `fuse`, presence, both vital
    /// extractors — then running on the weight vector twice over. The check is
    /// one pointer compare and takes the same "write nothing, return 0" stance
    /// as the null and inert guards. It is deliberately NOT `__restrict`:
    /// marking the parameters restrict would license the compiler to assume the
    /// comparison is false and drop it, which is the opposite of what is
    /// wanted. Partial overlap is still a caller error and is not detectable
    /// this cheaply.
    ///
    /// The first frame for a given subcarrier seeds its baseline and yields a
    /// zero residual — there is nothing to compare against yet.
    ///
    /// `frame.phases` is not required: this stage is amplitude-only, so
    /// demanding `frame.valid()` would reject a perfectly usable
    /// amplitude-only frame (the Rust port's golden test builds exactly such
    /// frames).
    std::size_t process(const CsiFrame& frame, float* residuals_out, float* weights_out) {
        if (!valid_ || residuals_out == nullptr || weights_out == nullptr ||
            residuals_out == weights_out || frame.amplitudes == nullptr) {
            return 0;
        }
        const std::size_t n = (frame.n_subcarriers < n_) ? frame.n_subcarriers : n_;
        if (n == 0) {
            return 0;
        }

        for (std::size_t i = 0; i < n; ++i) {
            const float amp = frame.amplitudes[i];
            if (!is_finite(amp)) {
                // A dropped or garbage subcarrier contributes nothing and must
                // not poison the baseline. The reference skips the variance
                // update too, so a flapping subcarrier's weight decays rather
                // than being pinned by zeros.
                //
                // This runs whether or not the subcarrier has been seeded. An
                // unseeded one stays unseeded: letting a NaN reach `seed()`
                // would make the baseline NaN for the lifetime of the node,
                // and `residual()` can never recover from that — the fused
                // scalar the presence detector and both vital extractors
                // consume would be permanently NaN. A node whose first frames
                // carry a dropped subcarrier is the ordinary startup case, not
                // an exotic one.
                residuals_out[i] = 0.0f;
                continue;
            }

            float r;
            if (seeded_[i]) {
                r = baseline_[i].residual(amp);
                baseline_[i].advance(alpha_, r);
            } else {
                baseline_[i].seed(amp);
                seeded_[i] = true;
                r          = 0.0f;
            }
            residuals_out[i] = r;
            var_[i].update(r);
        }

        return weights(weights_out, n);
    }

    /// Normalised per-subcarrier weights from running std deviation.
    ///
    /// Falls back to uniform weights until enough samples have accumulated for
    /// the variance estimate to mean anything — on frame 0 every residual is
    /// zero by construction, so every std is zero and this is the only answer
    /// that is not a division by zero.
    ///
    /// Writes the stds into `out` and normalises them in place, where the
    /// reference builds a list and then a second one. Same arithmetic, same
    /// order, no scratch. Nothing is written past the returned count.
    ///
    /// Inert suppressors return 0 and write nothing, the same as `process`.
    /// The reference has no such state; this method is public, and before
    /// `configure` was made to clear its fields on rejection it returned a
    /// count derived from a STALE `n_` and filled the buffer with plausible
    /// uniform weights — which a caller that checks only the returned count,
    /// the pattern `process` teaches, would accept. The `!valid_` test is now
    /// belt-and-braces: an inert suppressor also has `n_ == 0`, which forces
    /// the same answer one line below. It is kept because "inert writes
    /// nothing" should be stated here rather than inferred from an invariant
    /// held somewhere else.
    ///
    /// WHERE PARITY IS NOT CLAIMED. On a perfectly static channel — the empty
    /// room this stage is named for, and reachable on an ESP32 whose CSI is
    /// quantised int8 pairs that repeat exactly on a still link — this branch
    /// and the reference's disagree completely, and no tolerance can bridge it.
    /// The rearranged baseline update leaves `hi` bit-identical when the
    /// residual is exactly 0, so every std is exactly 0, `total <= 0.0f` fires
    /// and the weights are uniform. The reference's f64
    /// `alpha*amp + (1-alpha)*baseline` does NOT return a constant exactly: it
    /// accumulates ~7e-15 per frame, leaves this fallback after two frames and
    /// concentrates all weight on whichever subcarrier drifted first. On
    /// 50 frames of constant amplitudes {50, 17.3, 0.125, 1e6} at 20 Hz the
    /// port gives [0.25, 0.25, 0.25, 0.25], the reference [1, 0, 0, 0] and a
    /// literal float32 transcription [5.2e-05, 1.6e-05, 0, 0.99993] — three
    /// implementations, three answers, because what is being reported there is
    /// each one's own round-off and not a property of the channel.
    /// `fuse` agrees to ~1e-13 either way, which bounds the blast radius, but
    /// `RespirFiFusion` consumes the weight vector as its fallback_weights.
    /// Pinned by `test_static_channel_divergence`; raised in `spec_questions`,
    /// because only a vector can decide what all three ports should do here.
    std::size_t weights(float* out, std::size_t n) const {
        if (!valid_ || out == nullptr) {
            return 0;
        }
        const std::size_t count = (n < n_) ? n : n_;
        if (count == 0) {
            return 0;
        }
        NeumaierSum acc;
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = var_[i].std();
            acc.add(out[i]);
        }
        const float total = acc.value();
        if (total <= 0.0f) {
            const float uniform = 1.0f / static_cast<float>(count);
            for (std::size_t i = 0; i < count; ++i) {
                out[i] = uniform;
            }
            return count;
        }
        for (std::size_t i = 0; i < count; ++i) {
            out[i] = out[i] / total;
        }
        return count;
    }

    /// Read-only view of a subcarrier's variance state. Out of range is a
    /// programming error, as everywhere else in this library.
    const WelfordVariance& variance(std::size_t i) const { return var_[i]; }

    /// Clears baselines, seeds and variances; keeps the configuration, as the
    /// reference's `reset` does. Every slot is cleared, not just the first
    /// `n_`, so a later `configure` to a wider channel cannot inherit state
    /// from a narrower one.
    void reset() {
        for (std::size_t i = 0; i < MaxSubcarriers; ++i) {
            baseline_[i].clear();
            seeded_[i] = false;
            var_[i].reset();
        }
    }

  private:
    detail::CompensatedEma baseline_[MaxSubcarriers]{};
    bool                   seeded_[MaxSubcarriers]{};
    WelfordVariance        var_[MaxSubcarriers]{};

    std::size_t n_              = 0;
    float       sample_rate_hz_ = 0.0f;
    float       corner_hz_      = 0.0f;
    float       alpha_          = 0.0f;
    bool        valid_          = false;
};

// ---------------------------------------------------------------------------
// fuse
// ---------------------------------------------------------------------------

/// Collapse per-subcarrier residuals into one scalar.
///
/// Weights are renormalised over the subcarriers actually present, so a short
/// frame does not silently scale the output down.
///
/// `n` is the count the caller got back from `process`; the reference derives
/// it as `min(len(residuals), len(weights))`, which here would be a pair of
/// lengths nobody carries around.
///
/// Both sums are compensated for the same reason core.hpp gives: the reference
/// adds with CPython's `sum()`, which is itself Neumaier from 3.12, and this is
/// the scalar every downstream stage sees.
inline float fuse(const float* residuals, const float* weights, std::size_t n) {
    if (residuals == nullptr || weights == nullptr || n == 0) {
        return 0.0f;
    }
    NeumaierSum wsum;
    for (std::size_t i = 0; i < n; ++i) {
        wsum.add(weights[i]);
    }
    const float total_w = wsum.value();
    if (total_w <= 0.0f) {
        // No weight anywhere: the plain mean is the only unbiased answer, and
        // it keeps the output in amplitude units instead of collapsing to zero.
        NeumaierSum rsum;
        for (std::size_t i = 0; i < n; ++i) {
            rsum.add(residuals[i]);
        }
        return rsum.value() / static_cast<float>(n);
    }
    NeumaierSum num;
    for (std::size_t i = 0; i < n; ++i) {
        num.add(residuals[i] * weights[i]);
    }
    return num.value() / total_w;
}

}  // namespace saans
