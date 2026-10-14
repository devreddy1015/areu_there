// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — Goertzel, Hann, band scan, sideband scan.
//
// Port of the rest of `wifisense/dsp.py`. Header-only, no allocation after
// init, float32 throughout. Every function that needs working memory takes it
// from the caller and says how much; nothing here touches the heap.
//
// Pure arithmetic on purpose. Goertzel rather than an FFT because we only ever
// care about two narrow bands, so evaluating the DFT at just those points is
// both cheaper and higher-resolution than binning the whole spectrum and then
// interpolating.
//
#include <cassert>
#include <cmath>
#include <cstddef>
#include <optional>

#include "core.hpp"

namespace saans {

/// Result of a band scan. Same three fields as `wifisense.dsp.SpectralPeak`.
/// Absence of a result is `std::optional`'s empty state — which is storage
/// inline, no allocation, and forces the caller to deal with the "insufficient
/// data returns nothing" paths the reference has throughout.
struct SpectralPeak {
    float freq_hz    = 0.0f;
    float power      = 0.0f;
    float confidence = 0.0f;
};

namespace detail {

/// One sample of the periodic Hann window, computed the way `hann_window`
/// would have stored it. Used by the scans so that windowing needs only the
/// one output buffer instead of a second one for the window itself; the
/// multiply sees the identical float either way.
inline float hann_at(std::size_t i, std::size_t n) {
    if (n <= 1) {
        return 1.0f;
    }
    return 0.5f * (1.0f - cosf(2.0f * kPi * static_cast<float>(i) / static_cast<float>(n)));
}

}  // namespace detail

/// Periodic Hann window into a caller buffer of `n` floats.
///
/// Without it, a strong breathing component leaks sidelobes across the whole
/// band and buries the much weaker heartbeat peak.
///
/// `n <= 1` fills with 1.0, matching the reference's degenerate case.
inline void hann_window(float* out, std::size_t n) {
    if (out == nullptr) {
        return;
    }
    for (std::size_t i = 0; i < n; ++i) {
        out[i] = detail::hann_at(i, n);
    }
}

/// |X(f)|^2 at an arbitrary (non-bin-aligned) frequency.
///
/// This is the generalised Goertzel: the classic algorithm restricted to
/// integer bins, with the final phase correction kept so that `freq_hz` need
/// not land on a DFT bin. Cost is O(N) per frequency with two multiplies per
/// sample, versus O(N log N) for a full FFT whose bin spacing we would then
/// have to interpolate anyway.
///
/// FLOAT32 NOTE. `k` is computed and then immediately divided out again
/// (`w = 2*pi*k/n`), which is algebraically just `2*pi*f/fs`. The redundant
/// round trip is kept because the reference does it: in float32 the two forms
/// differ in the last bits of `w`, and `w` sets the resonator's pole angle,
/// where a 1 ulp shift accumulates over N samples.
inline float goertzel_power(const float* signal, std::size_t n, float freq_hz,
                            float sample_rate_hz) {
    if (signal == nullptr || n == 0 || sample_rate_hz <= 0.0f) {
        return 0.0f;
    }

    const float nf = static_cast<float>(n);
    const float k  = freq_hz * nf / sample_rate_hz;
    const float w  = 2.0f * kPi * k / nf;

    const float cos_w = cosf(w);
    const float sin_w = sinf(w);
    const float coeff = 2.0f * cos_w;

    float s1 = 0.0f;
    float s2 = 0.0f;
    for (std::size_t i = 0; i < n; ++i) {
        const float s0 = signal[i] + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }

    const float real = s1 - s2 * cos_w;
    const float imag = s2 * sin_w;
    return real * real + imag * imag;
}

/// `int((f_high - f_low) / resolution_hz)` — the number of scan steps, exposed
/// so callers can size their scratch (the grid has `steps + 1` points).
///
/// FRAGILITY WARNING, and the one place this port is knowingly walking a knife
/// edge. The reference truncates a float division, so the step count is
/// sensitive to the last bits of its three arguments. Worked example, the
/// breathing band:
///
///     Python, f64:            (0.5 - 0.1) / 0.002  = 200.00000000000003 -> 200
///     f32 inputs, f64 divide: (0.5f - 0.1f)/0.002f = 199.99998975545216 -> 199
///     f32 inputs, f32 divide: (0.5f - 0.1f)/0.002f = 200.0f             -> 200
///
/// So the division MUST stay in float32: promoting it to double to "be more
/// accurate" loses the last grid point and shifts both the candidate mean and
/// the confidence. The same holds for the heart band (288 vs 287). This is
/// luck as much as design — the float32 rounding happens to land on the same
/// integer the f64 reference lands on for all three configurations in the
/// spec. It is recorded in `spec_questions`: an exact grid count
/// (`round((f_high - f_low) / resolution_hz)`, or an integer bin count in the
/// spec) would make this robust instead of lucky.
///
/// Returns -1 for a non-finite or negative quotient; the scans treat that as
/// "fewer than 2 steps" and return nothing, as the reference does.
inline long scan_steps(float f_low, float f_high, float resolution_hz) {
    // THE DIVISION STAYS IN FLOAT32 (SAANS_PLAN section 4, port rules).
    // Promoting it "for accuracy" is actively wrong here: with float32 band
    // edges, (0.5f - 0.1f)/0.002f is 200.0f in float32 but 199.99998975545216
    // once promoted, and the rounding below would then still be fine -- but
    // the rule exists because the reference computes in one precision and we
    // must not introduce a second.
    const float q = (f_high - f_low) / resolution_hz;
    if (!is_finite(q) || q < 0.0f || q > 1.0e8f) {
        return -1;
    }
    // ROUND, do not truncate. The reference is now
    // `steps = int(round((f_high - f_low) / resolution_hz))`.
    //
    // Truncating put the step count on a knife edge: over 756 round-number
    // (f_low, f_high, resolution) combinations the truncated count disagreed
    // with the f64 reference in 223 cases, and an off-by-one step changes the
    // candidate count and the mean power, moving the confidence by ~0.5% --
    // about 50x the tolerance. The configurations in the vectors happened to
    // agree, so the suite was green by luck rather than by design.
    //
    // nearbyintf, NOT roundf: Python's round() is round-half-to-EVEN, and
    // roundf is round-half-away-from-zero. nearbyintf in the default rounding
    // mode is half-to-even and matches. This is a spec'd port rule, not a
    // preference.
    return static_cast<long>(nearbyintf(q));
}

// ---------------------------------------------------------------------------
// band_peak
// ---------------------------------------------------------------------------

/// Everything `band_peak` takes besides the signal itself.
struct BandPeakParams {
    float f_low         = 0.0f;
    float f_high        = 0.0f;
    float resolution_hz = 0.002f;
    bool  window        = true;

    /// Frequencies excluded (within +/- `notch_width_hz`) from both the peak
    /// search and the confidence statistics. Non-owning, like everything else
    /// here. This is how the heart rate stage avoids locking onto a breathing
    /// harmonic — see `HeartRateExtractor`.
    const float* notch_hz       = nullptr;
    std::size_t  n_notch        = 0;
    float        notch_width_hz = 0.0f;
};

namespace detail {

inline bool is_notched(float f, const BandPeakParams& p) {
    if (p.notch_hz == nullptr || p.n_notch == 0 || p.notch_width_hz <= 0.0f) {
        return false;
    }
    for (std::size_t j = 0; j < p.n_notch; ++j) {
        if (fabsf(f - p.notch_hz[j]) <= p.notch_width_hz) {
            return true;
        }
    }
    return false;
}

}  // namespace detail

/// Find the dominant frequency inside [f_low, f_high].
///
/// Returns the peak plus a confidence in [0, 1]. A clean sinusoid concentrates
/// its energy in a couple of bins and scores high; band-limited noise spreads
/// evenly and scores low. That ratio is what makes it possible to say "nobody
/// is breathing here" instead of always reporting whatever the largest bin
/// happened to be.
///
/// Counting zero crossings would be cheaper. It is not used here because it is
/// biased by any residual DC offset, degrades badly once a second periodicity
/// (the heartbeat) rides on the first, and yields no usable confidence signal.
///
/// SCRATCH. When `p.window` is true the caller must supply `scratch` with room
/// for `n` floats, which receives the windowed copy of the signal; when it is
/// false no scratch is needed and `scratch` may be null. The periodogram
/// itself is NOT stored: the scan keeps a compensated running total and a
/// three-point sliding window around the best candidate, which is all the
/// parabolic interpolation needs. That removes an O(steps) buffer from the
/// node's RAM budget and, with it, the only capacity limit this function would
/// otherwise have had.
inline std::optional<SpectralPeak> band_peak(const float* signal, std::size_t n,
                                             float sample_rate_hz, const BandPeakParams& p,
                                             float* scratch = nullptr,
                                             std::size_t scratch_capacity = 0) {
    if (signal == nullptr || n == 0 || p.f_low <= 0.0f || p.f_high <= p.f_low ||
        p.resolution_hz <= 0.0f) {
        return std::nullopt;
    }

    const float* src = signal;
    if (p.window) {
        // A short buffer would silently analyse the wrong window, which is the
        // one failure mode worse than a crash here.
        assert(scratch != nullptr && scratch_capacity >= n && "band_peak: scratch too small");
        if (scratch == nullptr || scratch_capacity < n) {
            return std::nullopt;
        }
        for (std::size_t i = 0; i < n; ++i) {
            scratch[i] = signal[i] * detail::hann_at(i, n);
        }
        src = scratch;
    }

    const long steps = scan_steps(p.f_low, p.f_high, p.resolution_hz);
    if (steps < 2) {
        return std::nullopt;
    }

    // Compensated, because the confidence divides the peak by the mean of
    // exactly this sum — see NeumaierSum in core.hpp.
    NeumaierSum total_acc;
    std::size_t n_candidates = 0;

    bool  have_peak = false;
    long  peak_i    = 0;
    float peak_p    = 0.0f;

    float y_prev       = 0.0f;  // periodogram value at i-1, notched or not
    bool  notched_prev = false; // and whether that i-1 was notched
    float y_peak_minus = 0.0f;
    float y_peak_plus  = 0.0f;
    bool  peak_minus_notched = false;
    bool  peak_plus_notched  = false;
    bool  want_plus    = false;

    for (long i = 0; i <= steps; ++i) {
        const float f  = p.f_low + static_cast<float>(i) * p.resolution_hz;
        const float pw = goertzel_power(src, n, f, sample_rate_hz);
        const bool  notched = detail::is_notched(f, p);

        // Record the right-hand neighbour of whatever is currently the best.
        // Done before the peak update so that a new best taking over at this
        // index still gets its own neighbour recorded next time round.
        if (want_plus) {
            y_peak_plus        = pw;
            peak_plus_notched  = notched;
            want_plus          = false;
        }

        if (!notched) {
            total_acc.add(pw);
            ++n_candidates;
            // Strictly greater: Python's `max(candidates, key=...)` keeps the
            // FIRST maximal element, and ties are not hypothetical on a flat
            // or saturated spectrum.
            if (!have_peak || pw > peak_p) {
                have_peak     = true;
                peak_i        = i;
                peak_p        = pw;
                y_peak_minus       = y_prev;
                peak_minus_notched = notched_prev;
                want_plus          = true;
            }
        }

        y_prev       = pw;
        notched_prev = notched;
    }

    if (!have_peak) {  // every grid point was notched out
        return std::nullopt;
    }

    const float total = total_acc.value();
    if (total <= 0.0f) {
        return std::nullopt;
    }

    float peak_f = p.f_low + static_cast<float>(peak_i) * p.resolution_hz;

    // Parabolic interpolation across the peak for sub-resolution accuracy.
    //
    // SKIPPED IF EITHER NEIGHBOUR IS NOTCHED. Interpolating across a notched
    // bin uses a power the caller explicitly asked to exclude, and it can drag
    // the estimate INTO the notch -- the opposite of what a notch is for. The
    // old behaviour was saved only by the `-1 < offset < 1` guard rejecting an
    // out-of-range offset, which is correct by accident: a slightly different
    // spectrum yields an in-range offset and the estimate walks into the
    // excluded band. The reference now returns the bare grid point instead.
    if (peak_i > 0 && peak_i < steps && !peak_minus_notched && !peak_plus_notched) {
        const float y0 = y_peak_minus;
        const float y1 = peak_p;
        const float y2 = y_peak_plus;
        const float denom = y0 - 2.0f * y1 + y2;
        if (denom != 0.0f) {
            const float offset = 0.5f * (y0 - y2) / denom;
            if (offset > -1.0f && offset < 1.0f) {
                peak_f = peak_f + offset * p.resolution_hz;
            }
        }
    }

    // Confidence, calibrated against what noise alone would produce.
    //
    // The naive "1 - mean/peak" scores band-limited noise around 0.75, because
    // the largest of M exponentially-distributed periodogram values sits
    // several times above their mean purely by chance. So compare the observed
    // peak-to-mean ratio against the ratio *expected from noise*.
    //
    // The number of statistically independent bins is set by the window
    // duration, not by how finely we scanned: n_ind = bandwidth * T. For noise,
    // E[peak/mean] over n_ind bins follows the Gumbel result ln(n_ind) + gamma.
    // For a pure tone, peak/mean approaches n_ind. Rescaling between those two
    // anchors puts noise near 0 and a clean sinusoid near 1.
    const float mean_power = total / static_cast<float>(n_candidates);
    if (peak_p <= 0.0f || mean_power <= 0.0f) {
        return SpectralPeak{peak_f, peak_p, 0.0f};
    }

    // A window function spreads even a perfect tone across several bins, so the
    // achievable peak-to-mean ratio is n_ind scaled down by the window's
    // equivalent noise bandwidth (1.5 bins for Hann, 1.0 for rectangular).
    // Without this correction a clean sinusoid tops out around 0.6.
    const float enbw = p.window ? 1.5f : 1.0f;

    const float duration_s = static_cast<float>(n) / sample_rate_hz;
    float       n_ind      = (p.f_high - p.f_low) * duration_s;
    if (n_ind < 2.0f) {  // `max(..., 2.0)`; NaN falls through unchanged, as in Python
        n_ind = 2.0f;
    }
    const float ratio       = peak_p / mean_power;
    const float noise_ratio = logf(n_ind) + kEulerMascheroni;
    const float tone_ratio  = n_ind / enbw;
    const float span        = tone_ratio - noise_ratio;

    float confidence = (span <= 0.0f) ? 0.0f : (ratio - noise_ratio) / span;
    confidence       = clamp(confidence, 0.0f, 1.0f);

    return SpectralPeak{peak_f, peak_p, confidence};
}

// ---------------------------------------------------------------------------
// sideband_peak
// ---------------------------------------------------------------------------

/// Everything `sideband_peak` takes besides the signal itself.
struct SidebandPeakParams {
    float f_low           = 0.0f;
    float f_high          = 0.0f;
    /// The breathing frequency, from the breathing stage.
    float modulator_hz    = 0.0f;
    float resolution_hz   = 0.002f;
    float harmonic_tol_hz = 0.03f;
};

/// Find a carrier hidden under a stronger modulating periodicity.
///
/// Why this exists: CSI *amplitude* is the magnitude of a static multipath
/// vector plus a body-reflected vector, so it is a nonlinear function of chest
/// displacement. That nonlinearity does two things to the heart-rate band:
///
///   1. It fills the band with harmonics of the breathing rate (k * f_b).
///   2. It creates intermodulation sidebands at f_heart +/- k * f_b.
///
/// On realistic data the f_heart + f_b sideband is often *stronger* than
/// f_heart itself, so plain peak-picking in 0.8-2.0 Hz confidently returns the
/// wrong number. A zero-crossing counter has exactly this failure mode and no
/// way to detect it, which is why this scans the spectrum instead.
///
/// The fix exploits the structure: a true cardiac component appears as a
/// symmetric triplet (f-f_b, f, f+f_b), while a spurious peak has no matching
/// partner on both sides. Scoring candidates by
/// `P(f) * sqrt(P(f-f_b) * P(f+f_b))` — a geometric mean, so *both* sidebands
/// must be present — recovers the carrier. Breathing harmonics are excluded
/// outright.
///
/// SCRATCH. `windowed` needs room for `n` floats. `scores` needs room for
/// `scan_steps(...) + 1` floats and cannot be elided the way band_peak's
/// periodogram was: the confidence is a comparison against the best RIVAL
/// triplet, which is not known until the scan has finished. Skipped (harmonic)
/// grid points are marked with a negative sentinel rather than compacted, so
/// the array index stays the grid index.
inline std::optional<SpectralPeak> sideband_peak(const float* signal, std::size_t n,
                                                 float sample_rate_hz,
                                                 const SidebandPeakParams& p, float* windowed,
                                                 std::size_t windowed_capacity, float* scores,
                                                 std::size_t scores_capacity) {
    if (signal == nullptr || n == 0 || p.modulator_hz <= 0.0f || p.f_high <= p.f_low ||
        p.resolution_hz <= 0.0f) {
        return std::nullopt;
    }

    assert(windowed != nullptr && windowed_capacity >= n && "sideband_peak: windowed too small");
    if (windowed == nullptr || windowed_capacity < n) {
        return std::nullopt;
    }
    for (std::size_t i = 0; i < n; ++i) {
        windowed[i] = signal[i] * detail::hann_at(i, n);
    }

    const long steps = scan_steps(p.f_low, p.f_high, p.resolution_hz);
    if (steps < 2) {
        return std::nullopt;
    }

    const std::size_t grid = static_cast<std::size_t>(steps) + 1u;
    assert(scores != nullptr && scores_capacity >= grid && "sideband_peak: scores too small");
    if (scores == nullptr || scores_capacity < grid) {
        return std::nullopt;
    }

    const auto power_at = [&](float f) -> float {
        return (f > 0.0f) ? goertzel_power(windowed, n, f, sample_rate_hz) : 0.0f;
    };

    // `round` here must be round-half-to-EVEN, which is what Python's round()
    // does for floats and what nearbyintf does under the default rounding
    // mode. roundf() rounds half away from zero and would classify a grid
    // point sitting exactly half way between two breathing harmonics
    // differently from the reference.
    const auto is_breathing_harmonic = [&](float f) -> bool {
        const float k = nearbyintf(f / p.modulator_hz);
        return k >= 1.0f && fabsf(f - k * p.modulator_hz) <= p.harmonic_tol_hz;
    };

    constexpr float kSkipped = -1.0f;  // scores are products of powers: never negative

    bool  have_best  = false;
    long  best_i     = 0;
    float best_score = 0.0f;

    for (long i = 0; i <= steps; ++i) {
        const float f = p.f_low + static_cast<float>(i) * p.resolution_hz;
        if (is_breathing_harmonic(f)) {
            scores[static_cast<std::size_t>(i)] = kSkipped;
            continue;
        }
        const float lower = power_at(f - p.modulator_hz);
        const float upper = power_at(f + p.modulator_hz);
        const float score =
            power_at(f) * sqrtf((lower > 0.0f ? lower : 0.0f) * (upper > 0.0f ? upper : 0.0f));
        scores[static_cast<std::size_t>(i)] = score;

        // First maximal element wins, as Python's max() does.
        if (!have_best || score > best_score) {
            have_best  = true;
            best_i     = i;
            best_score = score;
        }
    }

    if (!have_best) {  // every grid point was a breathing harmonic
        return std::nullopt;
    }
    if (best_score <= 0.0f) {
        return std::nullopt;
    }

    const float best_f = p.f_low + static_cast<float>(best_i) * p.resolution_hz;

    // Confidence here is peak *dominance*, not spectral concentration.
    //
    // The concentration measure used by `band_peak` is the wrong tool for this
    // band: a correct cardiac estimate still sits in a crowded spectrum full of
    // breathing harmonics and sidebands, so concentration stays low even when
    // the answer is right. What actually indicates a trustworthy result is
    // whether one triplet stands clear of every rival triplet elsewhere in the
    // band. So: compare the winning score against the best score at least
    // `separation_hz` away.
    const float half_mod       = p.modulator_hz * 0.5f;
    const float separation_hz  = (half_mod > 0.05f) ? half_mod : 0.05f;

    float runner_up = 0.0f;
    for (long i = 0; i <= steps; ++i) {
        const float s = scores[static_cast<std::size_t>(i)];
        if (s == kSkipped) {
            continue;
        }
        const float f = p.f_low + static_cast<float>(i) * p.resolution_hz;
        if (fabsf(f - best_f) > separation_hz && s > runner_up) {
            runner_up = s;
        }
    }

    const float confidence = clamp(1.0f - (runner_up / best_score), 0.0f, 1.0f);

    // Recomputed rather than cached from the scan, as the reference does. The
    // Goertzel is deterministic, so this is the same float; it is spelled out
    // to keep the port line-for-line with `return SpectralPeak(best_f,
    // power_at(best_f), confidence)`.
    return SpectralPeak{best_f, power_at(best_f), confidence};
}

}  // namespace saans
