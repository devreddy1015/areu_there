// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — second-order bandpass.
//
// Port of `BiquadBandpass` in `wifisense/dsp.py`. Header-only, no allocation,
// float32 throughout.
//
#include "core.hpp"

namespace saans {

/// Second-order constant-peak-gain bandpass (RBJ audio-EQ cookbook).
///
/// Chosen over an ad-hoc resonator of the form `r = 1 - bw/2`, which needs its
/// pole radius clamped to stay inside the unit circle and whose centre
/// frequency drifts with the sample rate. A standard biquad costs the same per
/// sample, is unconditionally stable for Q > 0, and puts the -3 dB points
/// exactly on the band edges you asked for.
///
/// Implemented in transposed direct form II: the best-conditioned of the four
/// classic topologies for float arithmetic. That choice was already worth
/// making in f64; at float32, with a 0.1-0.5 Hz band at 20 Hz sampling (poles
/// at radius 0.939, a ~16-sample memory), it is the difference between a
/// filter that tracks the reference and one that does not.
///
/// ERROR HANDLING. Python raises ValueError for f_low >= f_high or an f_high
/// at or above Nyquist. The library is built -fno-exceptions, so `configure`
/// reports the same two conditions by returning false and leaving the filter
/// INERT: `valid()` is false and `process` passes 0 out. A silently
/// mis-designed filter downstream would be far worse than a dead one, because
/// its output still looks like a plausible breathing waveform.
class BiquadBandpass {
  public:
    /// Inert until `configure` succeeds.
    BiquadBandpass() = default;

    /// Convenience form. Check `valid()`; there is no exception to catch.
    BiquadBandpass(float f_low, float f_high, float sample_rate_hz) {
        (void)configure(f_low, f_high, sample_rate_hz);
    }

    /// Design the filter and clear its state. Returns false (and leaves the
    /// filter inert) for the two inputs the reference rejects.
    bool configure(float f_low, float f_high, float sample_rate_hz) {
        reset();
        valid_ = false;

        // "need 0 < f_low < f_high"
        if (!(f_low > 0.0f) || !(f_low < f_high)) {
            return false;
        }
        // "f_high must be below Nyquist; raise the sample rate or lower the
        // band". Note this also rejects a non-positive sample rate, since
        // then nyquist <= 0 <= f_high.
        const float nyquist = sample_rate_hz / 2.0f;
        if (!(f_high < nyquist)) {
            return false;
        }

        f_low_          = f_low;
        f_high_         = f_high;
        sample_rate_hz_ = sample_rate_hz;

        // Geometric mean is the correct centre for a log-symmetric response.
        const float f0 = sqrtf(f_low * f_high);
        const float q  = f0 / (f_high - f_low);

        // Operation order copied from the reference: ((2*pi) * f0) / fs. The
        // same three values multiplied in another order round differently in
        // float32, and the coefficients sit in a feedback loop where a 1 ulp
        // shift in a1 moves the pole and persists.
        //
        // FLOAT32 DEVIATION, measured on the golden vectors. Designing these
        // coefficients in float32 rather than computing them in f64 and
        // rounding once costs accuracy only where the pole sits closest to the
        // unit circle — the breathing band:
        //
        //   biquad_breath   (0.1-0.5 Hz @ 20 Hz)  1.33e-5 vs 4.59e-6 full-scale
        //   biquad_presence (0.1-2.0 Hz @ 20 Hz)  1.16e-6 vs 1.23e-6
        //   biquad_motion   (1.0-4.0 Hz @ 20 Hz)  1.46e-7 vs 1.46e-7
        //
        // 1.33e-5 is 7.5x inside the 1e-4 contract, so the all-float design
        // stands: ADR-003 says float32, and this is the only place a double
        // would otherwise creep in. The f64 design is a one-line change and an
        // init-time-only cost if a later stage ever needs that factor of three
        // back — the measurement is recorded here so that decision is informed.
        const float w0    = 2.0f * kPi * f0 / sample_rate_hz;
        const float alpha = sinf(w0) / (2.0f * q);
        const float cos_w0 = cosf(w0);

        const float a0 = 1.0f + alpha;
        b0_ = alpha / a0;
        b1_ = 0.0f;
        b2_ = -alpha / a0;
        a1_ = (-2.0f * cos_w0) / a0;
        a2_ = (1.0f - alpha) / a0;

        valid_ = true;
        return true;
    }

    bool valid() const { return valid_; }

    float f_low() const { return f_low_; }
    float f_high() const { return f_high_; }
    float sample_rate_hz() const { return sample_rate_hz_; }

    /// Geometric-mean centre, recomputed the way the reference's property does.
    float center_hz() const { return sqrtf(f_low_ * f_high_); }

    /// Filter one sample.
    ///
    /// A non-finite input would otherwise poison the recurrence forever, so we
    /// reset rather than latch NaN. The second check catches the case where a
    /// finite input drives a finite state to infinity — on a CSI stream that
    /// means the amplitude series just stepped by something absurd, and the
    /// honest answer is to start the filter again rather than emit garbage for
    /// the next several minutes.
    float process(float x) {
        if (!valid_) {
            return 0.0f;
        }
        if (!is_finite(x)) {
            reset();
            return 0.0f;
        }

        // Transposed direct form II. b1_ is exactly zero; it is kept as a
        // named term rather than folded away so the expression matches the
        // reference line for line, and because 0.0f * x is exact for every
        // finite x (we have already rejected the rest).
        const float y = b0_ * x + s1_;
        s1_ = b1_ * x - a1_ * y + s2_;
        s2_ = b2_ * x - a2_ * y;

        if (!is_finite(y)) {
            reset();
            return 0.0f;
        }
        return y;
    }

    /// `process_all` over a caller-provided output buffer. In-place is allowed
    /// (`out == in`). There is no allocating overload on purpose.
    void process_all(const float* in, float* out, std::size_t n) {
        for (std::size_t i = 0; i < n; ++i) {
            out[i] = process(in[i]);
        }
    }

    void reset() {
        s1_ = 0.0f;
        s2_ = 0.0f;
    }

  private:
    float b0_ = 0.0f;
    float b1_ = 0.0f;
    float b2_ = 0.0f;
    float a1_ = 0.0f;
    float a2_ = 0.0f;

    float s1_ = 0.0f;
    float s2_ = 0.0f;

    float f_low_          = 0.0f;
    float f_high_         = 0.0f;
    float sample_rate_hz_ = 0.0f;
    bool  valid_          = false;
};

}  // namespace saans
