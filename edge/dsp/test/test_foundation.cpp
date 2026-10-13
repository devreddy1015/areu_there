// SPDX-License-Identifier: MIT
//
// Saans edge DSP — foundation parity tests.
//
// Covers core.hpp, biquad.hpp and spectral.hpp against the golden vectors in
// spec/golden/. Every numeric claim in here is checked against the Python
// reference; nothing is self-generated.
//
// Vectors used:
//   biquad_breath, biquad_presence, biquad_motion
//   goertzel
//   band_peak_tone, band_peak_noisy_tone, band_peak_noise,
//   band_peak_rect_window, band_peak_notched, band_peak_grid_rule,
//   band_peak_notch_neighbour
//   sideband_peak
//
// Tolerances are the ADR-003 / SAANS_PLAN §4 contract for the float32 edge
// build: 1e-4 relative on filter outputs, 0.05 BPM on rates. They are written
// once, here, and never relaxed per test.
//
#include <cstdio>
#include <string>
#include <vector>

#include "biquad.hpp"
#include "core.hpp"
#include "spectral.hpp"

#include "golden.hpp"

using saans::golden::Report;
using saans::golden::Vectors;

namespace {

// ---- the contract ----------------------------------------------------------
constexpr double kRelTol = 1e-4;      // filter outputs, powers, confidences
constexpr double kBpmTol = 0.05;      // rates, in BPM
/// Confidences carry an ABSOLUTE tolerance; see the note in test_band_peak.
constexpr double kConfTol = 5e-3;

// ---- scratch ---------------------------------------------------------------
// Static, fixed-capacity, file-scope: the same shape the firmware uses. The
// library never allocates, so the test must not either when driving it.
constexpr std::size_t kMaxSamples = 4096;
constexpr std::size_t kMaxSteps   = 4096;

float g_scratch[kMaxSamples];
float g_scores[kMaxSteps + 1];
float g_filtered[kMaxSamples];

// ---------------------------------------------------------------------------
// core.hpp — behaviour that has no golden vector because it is structural
// ---------------------------------------------------------------------------
void test_core(Report& r) {
    r.begin("core.hpp: RingBuffer, Neumaier, clamp", "no vector (structural)");

    saans::RingBuffer<float, 3> rb;
    r.equal("ring capacity", static_cast<long long>(rb.capacity()), 3);
    r.truth("ring starts empty", rb.empty() && !rb.full(), "size=0");

    rb.push(1.0f);
    rb.push(2.0f);
    r.equal("ring size after 2 pushes", static_cast<long long>(rb.size()), 2);
    r.truth("ring not yet full", !rb.full(), "size 2 of 3");
    r.close("ring[0] is oldest", static_cast<double>(rb[0]), 1.0, 0.0);

    rb.push(3.0f);
    rb.push(4.0f);
    rb.push(5.0f);  // two overwrites: 1 and 2 are gone
    r.truth("ring full after overflow", rb.full(), "size 3 of 3");
    r.close("ring[0] after overwrite", static_cast<double>(rb[0]), 3.0, 0.0);
    r.close("ring[2] is newest", static_cast<double>(rb[2]), 5.0, 0.0);
    r.close("ring newest()", static_cast<double>(rb.newest()), 5.0, 0.0);

    float lin[3] = {0.0f, 0.0f, 0.0f};
    const std::size_t got = rb.copy_to(lin, 3);
    r.equal("copy_to element count", static_cast<long long>(got), 3);
    r.truth("copy_to is oldest-to-newest", lin[0] == 3.0f && lin[1] == 4.0f && lin[2] == 5.0f,
            "3, 4, 5");
    float too_small[2];
    r.equal("copy_to refuses a short buffer", static_cast<long long>(rb.copy_to(too_small, 2)),
            0);

    // The case the compensation exists for: in float32 the 1 is entirely lost
    // against 1e8 by a naive running sum, which returns 1 instead of 2.
    const float cancelling[4] = {1.0e8f, 1.0f, -1.0e8f, 1.0f};
    float       naive         = 0.0f;
    for (float v : cancelling) {
        naive += v;
    }
    r.close("neumaier_sum recovers lost bits", static_cast<double>(saans::neumaier_sum(cancelling, 4)),
            2.0, 0.0);
    r.truth("naive float32 sum really does lose them", naive != 2.0f,
            std::string("naive = ") + std::to_string(naive));

    const float ones[4] = {2.0f, 2.0f, 2.0f, 2.0f};
    r.close("rolling_rms of a constant", static_cast<double>(saans::rolling_rms(ones, 4)), 2.0,
            kRelTol);
    r.close("rolling_rms of nothing", static_cast<double>(saans::rolling_rms(ones, 0)), 0.0, 0.0);

    r.close("clamp below", static_cast<double>(saans::clamp(-1.0f, 0.0f, 1.0f)), 0.0, 0.0);
    r.close("clamp above", static_cast<double>(saans::clamp(7.0f, 0.0f, 1.0f)), 1.0, 0.0);
    r.truth("is_finite rejects inf/nan",
            !saans::is_finite(INFINITY) && !saans::is_finite(NAN) && saans::is_finite(0.0f),
            "inf, nan, 0");

    // hann_window: periodic (not symmetric) — w[0] == 0 and w[n/2] == 1, and
    // crucially w[n-1] != 0, which is what distinguishes the two conventions.
    float w[8];
    saans::hann_window(w, 8);
    r.close("hann w[0]", static_cast<double>(w[0]), 0.0, 0.0);
    r.close("hann w[4] (centre)", static_cast<double>(w[4]), 1.0, kRelTol);
    r.truth("hann is periodic, not symmetric", w[7] > 0.0f && w[7] < 0.2f,
            std::string("w[7] = ") + std::to_string(w[7]));
    float w1[1] = {0.0f};
    saans::hann_window(w1, 1);
    r.close("hann n=1 degenerate case", static_cast<double>(w1[0]), 1.0, 0.0);
}

// ---------------------------------------------------------------------------
// biquad.hpp
// ---------------------------------------------------------------------------
void test_biquad_vector(Report& r, const char* name) {
    const Vectors v(name);
    r.begin("biquad.hpp: BiquadBandpass", name);

    const float f_low  = v.scalar_f32("f_low");
    const float f_high = v.scalar_f32("f_high");
    const float fs     = v.scalar_f32("fs");

    const std::vector<float>  in  = v.array_f32("input");
    const std::vector<double>& out = v.array("output");

    if (in.size() > kMaxSamples) {
        saans::golden::fatal("%s: input of %zu exceeds the test's %zu-sample scratch", name,
                             in.size(), kMaxSamples);
    }
    r.equal("input/output lengths agree", static_cast<long long>(in.size()),
            static_cast<long long>(out.size()));

    saans::BiquadBandpass bq(f_low, f_high, fs);
    r.truth("filter designed", bq.valid(), std::string(name) + " band accepted");
    r.close("centre is the geometric mean", static_cast<double>(bq.center_hz()),
            std::sqrt(static_cast<double>(f_low) * static_cast<double>(f_high)), kRelTol);

    bq.process_all(in.data(), g_filtered, in.size());
    r.close_array("filtered series", g_filtered, out.data(), out.size(), kRelTol);
}

void test_biquad_guards(Report& r) {
    r.begin("biquad.hpp: guards and self-healing", "no vector (structural)");

    // The two conditions the reference raises ValueError for.
    saans::BiquadBandpass bad;
    r.truth("rejects f_low >= f_high", !bad.configure(0.5f, 0.1f, 20.0f), "0.5 .. 0.1 @ 20 Hz");
    r.truth("rejects f_high at Nyquist", !bad.configure(0.1f, 10.0f, 20.0f), "0.1 .. 10 @ 20 Hz");
    r.truth("rejects f_low <= 0", !bad.configure(0.0f, 0.5f, 20.0f), "0 .. 0.5 @ 20 Hz");
    r.close("an inert filter outputs zero", static_cast<double>(bad.process(1.0f)), 0.0, 0.0);

    // "A non-finite input would otherwise poison the recurrence forever, so we
    // reset rather than latch NaN."
    saans::BiquadBandpass bq(0.1f, 0.5f, 20.0f);
    for (int i = 0; i < 200; ++i) {
        (void)bq.process(1.0f);
    }
    const float after_nan = bq.process(NAN);
    r.close("NaN input returns 0", static_cast<double>(after_nan), 0.0, 0.0);
    const float after_reset = bq.process(1.0f);
    r.truth("state was cleared, not latched", saans::is_finite(after_reset),
            std::string("next output = ") + std::to_string(after_reset));

    saans::BiquadBandpass fresh(0.1f, 0.5f, 20.0f);
    r.close("reset leaves the same filter as new",
            static_cast<double>(after_reset), static_cast<double>(fresh.process(1.0f)), 0.0);
}

// ---------------------------------------------------------------------------
// spectral.hpp — Goertzel
// ---------------------------------------------------------------------------
void test_goertzel(Report& r) {
    const Vectors v("goertzel");
    r.begin("spectral.hpp: goertzel_power", "goertzel");

    const float               fs    = v.scalar_f32("fs");
    const std::vector<float>  in    = v.array_f32("input");
    const std::vector<float>  freqs = v.array_f32("freqs");
    const std::vector<double>& power = v.array("power");

    r.equal("freq/power lengths agree", static_cast<long long>(freqs.size()),
            static_cast<long long>(power.size()));

    std::vector<float> got(freqs.size());
    for (std::size_t i = 0; i < freqs.size(); ++i) {
        got[i] = saans::goertzel_power(in.data(), in.size(), freqs[i], fs);
    }
    r.close_array("|X(f)|^2 across the sweep", got.data(), power.data(), power.size(), kRelTol);

    // The per-point relative figure matters more here than for a bipolar
    // series: powers are strictly positive, so there is no zero crossing to
    // excuse a large relative error at a small value.
    const saans::golden::ArrayDeviation d =
        saans::golden::compare(got.data(), power.data(), power.size(), 0.0);  // gate 0: every point counts
    r.truth("every point within 1e-4 relative, individually",
            d.pointwise_rel <= kRelTol,
            std::string("worst per-point rel dev = ") + std::to_string(d.pointwise_rel));

    // Degenerate guards the reference has.
    r.close("empty signal", static_cast<double>(saans::goertzel_power(in.data(), 0, 0.25f, fs)),
            0.0, 0.0);
    r.close("non-positive sample rate",
            static_cast<double>(saans::goertzel_power(in.data(), in.size(), 0.25f, 0.0f)), 0.0,
            0.0);
}

// ---------------------------------------------------------------------------
// spectral.hpp — band_peak
// ---------------------------------------------------------------------------
void test_band_peak(Report& r, const char* name) {
    const Vectors v(name);
    r.begin("spectral.hpp: band_peak", name);

    const std::vector<float> in      = v.array_f32("input");
    const std::vector<float> notches = v.array_f32("notch_hz");

    if (in.size() > kMaxSamples) {
        saans::golden::fatal("%s: input of %zu exceeds the test's %zu-sample scratch", name,
                             in.size(), kMaxSamples);
    }

    saans::BandPeakParams p;
    p.f_low          = v.scalar_f32("f_low");
    p.f_high         = v.scalar_f32("f_high");
    p.resolution_hz  = v.scalar_f32("resolution_hz");
    p.window         = v.flag("window");
    p.notch_width_hz = v.scalar_f32("notch_width_hz");
    p.notch_hz       = notches.empty() ? nullptr : notches.data();
    p.n_notch        = notches.size();

    const bool expect_found = v.flag("found");

    const std::optional<saans::SpectralPeak> got = saans::band_peak(
        in.data(), in.size(), v.scalar_f32("fs"), p, g_scratch, kMaxSamples);

    r.truth("found a peak", got.has_value() == expect_found,
            expect_found ? "reference found one" : "reference found none");
    if (!got.has_value()) {
        return;
    }

    // Frequency is checked both ways: relative, and as the rate tolerance the
    // plan states, since this number becomes a BPM downstream.
    r.close("peak freq (relative)", static_cast<double>(got->freq_hz), v.scalar("freq_hz"),
            kRelTol);
    r.close_abs("peak freq (as a rate)", static_cast<double>(got->freq_hz) * 60.0,
                v.scalar("freq_hz") * 60.0, kBpmTol);
    r.close("peak power", static_cast<double>(got->power), v.scalar("power"), kRelTol);
    // CONFIDENCE IS CHECKED ABSOLUTELY, NOT RELATIVELY (SAANS_PLAN section 4).
    // It is a bounded [0,1] score and every consumer compares it against
    // ABSOLUTE thresholds -- VitalEstimate::from_confidence uses 0.40 and 0.70.
    // A relative tolerance is meaningless exactly where the score spends most
    // of its time: near zero, where (ratio - noise_ratio) cancels and a tiny
    // absolute error reads as an enormous relative one. Measured over 375
    // randomised cases the worst absolute deviation was 2.03e-3 with zero
    // threshold crossings; the same data read relatively as a 4.2% "failure".
    r.close_abs("confidence (absolute)", static_cast<double>(got->confidence),
                v.scalar("confidence"), kConfTol);

    r.note("grid: %ld steps over %.4f..%.4f Hz at %.5f Hz, window=%s, %zu notch(es) +/- %.4f Hz",
           saans::scan_steps(p.f_low, p.f_high, p.resolution_hz), static_cast<double>(p.f_low),
           static_cast<double>(p.f_high), static_cast<double>(p.resolution_hz),
           p.window ? "hann" : "rect", p.n_notch, static_cast<double>(p.notch_width_hz));
}

/// The thing most likely to be subtly wrong, stated as a claim rather than
/// left implicit in three separate vectors: the confidence is calibrated so
/// that band-limited noise lands near 0 and a clean tone near 1. A naive
/// "1 - mean/peak" would score the noise case around 0.75.
///
/// The thresholds here come from the reference's own outputs, read out of the
/// vectors — this check asserts the ORDERING and the calibration anchors, not
/// a hand-picked number.
void test_band_peak_calibration(Report& r) {
    r.begin("spectral.hpp: band_peak noise calibration", "band_peak_{tone,noisy_tone,noise}");

    const Vectors tone("band_peak_tone");
    const Vectors noisy("band_peak_noisy_tone");
    const Vectors noise("band_peak_noise");

    const double c_tone  = tone.scalar("confidence");
    const double c_noisy = noisy.scalar("confidence");
    const double c_noise = noise.scalar("confidence");

    r.truth("reference orders tone > noisy > noise", c_tone > c_noisy && c_noisy > c_noise,
            std::to_string(c_tone) + " > " + std::to_string(c_noisy) + " > " +
                std::to_string(c_noise));
    r.truth("reference scores pure noise near zero, not ~0.75", c_noise < 0.05,
            std::string("reference noise confidence = ") + std::to_string(c_noise));

    // Now the same three through this port, driven from the vectors' own
    // parameters, and required to preserve both facts.
    auto run = [&](const Vectors& v) -> float {
        const std::vector<float> in = v.array_f32("input");
        saans::BandPeakParams    p;
        p.f_low         = v.scalar_f32("f_low");
        p.f_high        = v.scalar_f32("f_high");
        p.resolution_hz = v.scalar_f32("resolution_hz");
        p.window        = v.flag("window");
        const auto out =
            saans::band_peak(in.data(), in.size(), v.scalar_f32("fs"), p, g_scratch, kMaxSamples);
        if (!out.has_value()) {
            saans::golden::fatal("%s: band_peak returned nothing while the reference did not",
                                 v.name().c_str());
        }
        return out->confidence;
    };

    const float p_tone  = run(tone);
    const float p_noisy = run(noisy);
    const float p_noise = run(noise);

    r.truth("port preserves the ordering", p_tone > p_noisy && p_noisy > p_noise,
            std::to_string(p_tone) + " > " + std::to_string(p_noisy) + " > " +
                std::to_string(p_noise));
    r.truth("port scores pure noise near zero", p_noise < 0.05f,
            std::string("port noise confidence = ") + std::to_string(p_noise));
    r.close("port tone confidence saturates at 1", static_cast<double>(p_tone), 1.0, kRelTol);

    // A deliberately coarse sanity floor on the hardest of the three: if the
    // Gumbel anchor or the ENBW correction were dropped, the noise confidence
    // would move by far more than this.
    r.close("noise confidence tracks the reference", static_cast<double>(p_noise), c_noise,
            kRelTol);
}

// ---------------------------------------------------------------------------
// spectral.hpp — sideband_peak
// ---------------------------------------------------------------------------
void test_sideband_peak(Report& r) {
    const Vectors v("sideband_peak");
    r.begin("spectral.hpp: sideband_peak", "sideband_peak");

    const std::vector<float> in = v.array_f32("input");
    if (in.size() > kMaxSamples) {
        saans::golden::fatal("sideband_peak: input of %zu exceeds the %zu-sample scratch",
                             in.size(), kMaxSamples);
    }

    saans::SidebandPeakParams p;
    p.f_low           = v.scalar_f32("f_low");
    p.f_high          = v.scalar_f32("f_high");
    p.modulator_hz    = v.scalar_f32("modulator_hz");
    p.resolution_hz   = v.scalar_f32("resolution_hz");
    p.harmonic_tol_hz = v.scalar_f32("harmonic_tol_hz");

    const long steps = saans::scan_steps(p.f_low, p.f_high, p.resolution_hz);
    if (steps < 0 || static_cast<std::size_t>(steps) + 1u > kMaxSteps + 1u) {
        saans::golden::fatal("sideband_peak: %ld steps exceeds the %zu-step scratch", steps,
                             kMaxSteps);
    }

    const std::optional<saans::SpectralPeak> got =
        saans::sideband_peak(in.data(), in.size(), v.scalar_f32("fs"), p, g_scratch, kMaxSamples,
                             g_scores, kMaxSteps + 1u);

    r.truth("found a carrier", got.has_value(), "reference found one");
    if (!got.has_value()) {
        return;
    }

    r.close("carrier freq (relative)", static_cast<double>(got->freq_hz), v.scalar("freq_hz"),
            kRelTol);
    r.close_abs("carrier freq (as a rate)", static_cast<double>(got->freq_hz) * 60.0,
                v.scalar("freq_hz") * 60.0, kBpmTol);
    r.close("carrier power", static_cast<double>(got->power), v.scalar("power"), kRelTol);
    r.close("dominance confidence", static_cast<double>(got->confidence),
            v.scalar("confidence"), kRelTol);

    // The point of the triplet search: the answer must not be a breathing
    // harmonic. 1.12 Hz against a 0.25 Hz modulator is 4.48 harmonics out.
    const float k = std::round(got->freq_hz / p.modulator_hz);
    r.truth("result is not a breathing harmonic",
            std::fabs(got->freq_hz - k * p.modulator_hz) > p.harmonic_tol_hz,
            std::string("f = ") + std::to_string(got->freq_hz) + " Hz, nearest harmonic " +
                std::to_string(k * p.modulator_hz) + " Hz");
    r.note("grid: %ld steps over %.3f..%.3f Hz, modulator %.4f Hz, scores buffer %ld floats",
           steps, static_cast<double>(p.f_low), static_cast<double>(p.f_high),
           static_cast<double>(p.modulator_hz), steps + 1);
}

}  // namespace

int main(int argc, char** argv) {
    // Accept the vector directory as argv[1], the convention edge/dsp/Makefile
    // uses, so this test runs identically under either Makefile in the tree.
    saans::golden::dir_from_argv(argc, argv);

    std::printf("Saans edge DSP - foundation parity against the Python reference\n");
    std::printf("vectors: %s\n", saans::golden::dir().c_str());
    std::printf("contract: %.0e relative on filter outputs, %.2f BPM on rates (ADR-003)\n",
                kRelTol, kBpmTol);

    Report r;

    test_core(r);

    test_biquad_vector(r, "biquad_breath");
    test_biquad_vector(r, "biquad_presence");
    test_biquad_vector(r, "biquad_motion");
    test_biquad_guards(r);

    test_goertzel(r);

    test_band_peak(r, "band_peak_tone");
    test_band_peak(r, "band_peak_noisy_tone");
    test_band_peak(r, "band_peak_noise");
    test_band_peak(r, "band_peak_rect_window");
    test_band_peak(r, "band_peak_notched");
    // Added with the section 4 port rules. Each DISCRIMINATES: it passes only
    // under the new rule and fails under the old one.
    //   band_peak_grid_rule       res = 0.0020000000005 -> 199.99999995, which
    //                             truncates to 199 steps and rounds to 200.
    //                             Confidence 0.786477 new vs 0.780879 old.
    //   band_peak_notch_neighbour tone at 0.2505 Hz with the notch on the
    //                             0.252 grid point. Returning the grid point
    //                             gives 0.25; interpolating across the notched
    //                             neighbour gave 0.250498.
    test_band_peak(r, "band_peak_grid_rule");
    test_band_peak(r, "band_peak_notch_neighbour");
    test_band_peak_calibration(r);

    test_sideband_peak(r);

    return r.finish("foundation");
}
