// SPDX-License-Identifier: MIT
//
// Saans edge DSP — static-suppression parity tests.
//
// Covers preprocess.hpp (corner_to_alpha, alpha_to_corner, WelfordVariance,
// StaticSuppressor, fuse) against the golden vectors in spec/golden/.
//
// Vectors used:
//   static_suppressor   (amplitudes -> residuals, weights, fused)
//
// Tolerance is the ADR-003 / SAANS_PLAN §4 contract for the float32 edge
// build: 1e-4 relative on filter outputs. It is written once, here, and never
// relaxed per test. A vector that cannot be found is a hard failure inside
// golden.hpp; there is no skip path in this file either.
//
// ---------------------------------------------------------------------------
// WHAT THE 600-FRAME VECTOR CANNOT SEE, AND WHAT IS DONE ABOUT IT
// ---------------------------------------------------------------------------
// `static_suppressor` is 600 frames at 20 Hz: thirty seconds. Both float32
// deviations in preprocess.hpp are about what happens over DAYS, so the vector
// is structurally incapable of catching a regression in either — deleting the
// baseline compensation, or the variance compensation, or replacing either
// NeumaierSum with a plain `+=`, all left the earlier revision of this file
// green at 67/67. Three self-contained soaks below close that: they need no
// vector, because they compare the shipped float32 code against an f64
// transcription of the reference computed in the test itself, over 1e6-1.5e6
// frames. Each one also asserts that the UNCOMPENSATED form breaches the
// contract on the same stream, so the deviation stays load-bearing rather than
// becoming forty lines of arithmetic nobody can justify deleting.
//
// The soaks dominate the runtime of this binary (~2 s). That is the price of
// testing an uptime claim at all.
//
// ---------------------------------------------------------------------------
// WHY THIS FILE INCLUDES EVERY HEADER IN edge/dsp
// ---------------------------------------------------------------------------
// preprocess.hpp and presence.hpp each define their own Welford accumulator,
// for the reason preprocess.hpp's header explains, and an earlier revision gave
// both the same name — which made including both in one translation unit a hard
// compile error in either order, under both toolchains. No test binary happened
// to include both, so the suite was green while edge/firmware could not be
// built at all: `pipeline.py:63,83` feeds the suppressor straight into the
// presence detector. The include block below is the regression test for that.
// It costs a little compile time and it fails loudly, in CI, instead of in the
// firmware build.
//
#include <cmath>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>

// Every header in edge/dsp, alphabetically, so a later type collision between
// any two of them fails here. Do not prune this list to "the ones this test
// uses" — that is exactly what hid the collision.
#include "biquad.hpp"
#include "bnr.hpp"
#include "core.hpp"
#include "crc32.hpp"
#include "feature_frame.hpp"
#include "fft.hpp"
#include "gapfill.hpp"
#include "preprocess.hpp"
#include "presence.hpp"
#include "spectral.hpp"
#include "vitals.hpp"

#include "golden.hpp"

using saans::golden::Report;
using saans::golden::Vectors;

namespace {

// ---- the contract ----------------------------------------------------------
constexpr double kRelTol = 1e-4;  // residuals, weights, fused scalar

// Soak floors. Both are TIGHTER than the contract, deliberately: a soak exists
// to catch an accumulator starting to walk long before it reaches 1e-4, and
// the shipped code clears each of them by more than an order of magnitude.
// Neither is a relaxation of anything.
constexpr double kSoakRelFloor = 1e-5;  // relative, on std() and on weights
constexpr double kSoakAbsFloor = 1e-6;  // absolute, on a residual of full scale ~2.3

// Two float32 ulps, relative. This is the tightest claim that can be made about
// any sum that is rounded back to a float at the end, so it is what the two
// compensated-summation checks assert: the shipped code must land inside it and
// a naive `+=` over the same terms must not.
constexpr double kTwoUlpRel = 2.4e-7;

// ---- scratch ---------------------------------------------------------------
// Static, fixed-capacity, file-scope: the same shape the firmware uses. The
// library never allocates, so the code driving it must not either. (The golden
// data itself arrives in std::vectors from golden.hpp, which is host-only test
// code and says so.)
constexpr std::size_t kMaxSub = 64;

float g_residuals[kMaxSub];
float g_weights[kMaxSub];

/// Fill both scratch buffers with a value no computation can produce, so that
/// "the slot still holds the canary" distinguishes a write from a stale value.
/// Several checks below are worthless without this: `g_residuals` is file
/// scope, and an assertion that a slot is 0.0 passes just as happily when the
/// previous frame left it at 0.0 and nothing was written at all.
void poison(float v = -99.0f) {
    for (std::size_t i = 0; i < kMaxSub; ++i) {
        g_residuals[i] = v;
        g_weights[i]   = v;
    }
}

// ---- deterministic test signal ---------------------------------------------
// A 32-bit LCG, not <random>: the soaks must produce the same stream on every
// compiler and every run, and the comparison is float32-against-f64 on
// BIT-IDENTICAL input, so the generator has to be exactly reproducible.
struct Lcg {
    std::uint32_t s = 12345u;
    float         next() {  // uniform in [-0.5, 0.5)
        s = s * 1664525u + 1013904223u;
        return static_cast<float>((s >> 8) & 0xFFFFFFu) / 16777216.0f - 0.5f;
    }
};

constexpr double kTwoPi = 6.283185307179586476925286766559;

/// One amplitude sample at frame `t`: a static pedestal near 50, a 0.25 Hz
/// breathing term and a little noise. Rounded to float32 before it is handed to
/// anyone, so the f32 port and the f64 reference see the same bits and every
/// difference that follows is arithmetic, not input.
float soak_amplitude(long long t, float gain, float level, Lcg& rng) {
    const double ph = kTwoPi * 0.25 * static_cast<double>(t) / 20.0;
    return level + gain * sinf(static_cast<float>(ph)) + 0.3f * rng.next();
}

/// A number into a std::string for the `truth` details below. Ten significant
/// digits, because std::to_string prints six DECIMALS — which renders every
/// error a soak cares about as "0.000000", and two totals that differ in the
/// eighth digit as the same number.
std::string sci(double v) {
    char buf[40];
    std::snprintf(buf, sizeof(buf), "%.10g", v);
    return std::string(buf);
}

// ---- f64 transcriptions of the reference, for the soaks --------------------
// Straight from wifisense/types.py and wifisense/preprocess.py, in double, with
// the reference's operation order. These are the "what the Python would have
// said" side of every soak comparison. Host-only test code: doubles here are
// the point, not a slip.

struct RefWelford {
    long long count = 0;
    double    mean  = 0.0;
    double    m2    = 0.0;

    void update(double x) {
        ++count;
        const double delta = x - mean;
        mean += delta / static_cast<double>(count);
        m2 += delta * (x - mean);
    }
    double variance() const {
        return (count > 1) ? m2 / static_cast<double>(count - 1) : 0.0;
    }
    double std_() const { return std::sqrt(variance()); }
};

/// The float32 accumulator preprocess.hpp used to ship: Welford with a plain
/// `+=` on m2. Kept here, and only here, so each soak can assert that the
/// compensation it is testing actually changes the answer.
struct PlainWelford {
    std::size_t count = 0;
    float       mean  = 0.0f;
    float       m2    = 0.0f;

    void update(float x) {
        ++count;
        const float delta = x - mean;
        mean += delta / static_cast<float>(count);
        m2 += delta * (x - mean);
    }
    float variance() const {
        return (count > 1u) ? m2 / static_cast<float>(count - 1u) : 0.0f;
    }
    float std_() const { return sqrtf(variance()); }
};

/// f64 transcription of StaticSuppressor, fixed at four subcarriers.
struct RefSuppressor {
    static constexpr std::size_t kN = 4;

    double     alpha = 0.0;
    double     baseline[kN]{};
    bool       seeded[kN]{};
    RefWelford var[kN];

    RefSuppressor(double fs, double corner) {
        const double a = kTwoPi * corner / fs;
        alpha          = (a < 1e-6) ? 1e-6 : ((a > 0.999) ? 0.999 : a);
    }

    void process(const float* amps, double* residuals, double* weights) {
        for (std::size_t i = 0; i < kN; ++i) {
            const double amp = static_cast<double>(amps[i]);
            if (!std::isfinite(amp)) {
                residuals[i] = 0.0;
                continue;
            }
            if (seeded[i]) {
                residuals[i] = amp - baseline[i];
                baseline[i]  = alpha * amp + (1.0 - alpha) * baseline[i];
            } else {
                baseline[i]  = amp;
                seeded[i]    = true;
                residuals[i] = 0.0;
            }
            var[i].update(residuals[i]);
        }
        double total = 0.0;
        for (std::size_t i = 0; i < kN; ++i) {
            weights[i] = var[i].std_();
            total += weights[i];
        }
        if (total <= 0.0) {
            for (std::size_t i = 0; i < kN; ++i) {
                weights[i] = 1.0 / static_cast<double>(kN);
            }
            return;
        }
        for (std::size_t i = 0; i < kN; ++i) {
            weights[i] /= total;
        }
    }
};

// ---------------------------------------------------------------------------
// Header coexistence — the collision regression, made visible in the report
// ---------------------------------------------------------------------------
void test_header_coexistence(Report& r) {
    r.begin("edge/dsp headers coexist in one TU", "no vector (structural)");

    // That this file compiles at all is the assertion; these two lines make it
    // show up as a check rather than as an invisible property of the build.
    // Both Welford types must be usable side by side in one TU, because the
    // firmware uses both stages in one binary.
    saans::WelfordVariance from_preprocess;
    saans::RunningVariance from_presence;
    from_preprocess.update(1.0f);
    from_preprocess.update(3.0f);
    from_presence.update(1.0f);
    from_presence.update(3.0f);

    r.close("preprocess's WelfordVariance::std()", static_cast<double>(from_preprocess.std()),
            std::sqrt(2.0), kRelTol);
    r.close("presence's RunningVariance::std_dev()", static_cast<double>(from_presence.std_dev()),
            std::sqrt(2.0), kRelTol);
    r.truth("both stages' accumulators exist in one translation unit", true,
            "every header in edge/dsp is included above");
    r.note("the two are separate types on purpose: only preprocess's carries the compensated "
           "m2 accumulator, and a shared one under presence.hpp's include guard would make that "
           "depend on include order (see preprocess.hpp's header, and spec_questions)");
}

// ---------------------------------------------------------------------------
// corner_to_alpha / alpha_to_corner
// ---------------------------------------------------------------------------
void test_corner(Report& r) {
    r.begin("corner_to_alpha / alpha_to_corner", "static_suppressor + structural");

    const Vectors v("static_suppressor");
    const float   fs     = v.scalar_f32("fs");
    const float   corner = v.scalar_f32("corner_hz");

    // The reference's own derivation, checked against the number it wrote.
    // float32 costs 2.8e-08 relative here: the representation error of pi.
    r.close("alpha(0.03 Hz @ 20 Hz)", static_cast<double>(saans::corner_to_alpha(corner, fs)),
            v.scalar("alpha"), kRelTol);

    // Round trip. This is the property the function exists for: the corner you
    // asked for is the corner you get.
    const float a = saans::corner_to_alpha(corner, fs);
    r.close("alpha -> corner round trip", static_cast<double>(saans::alpha_to_corner(a, fs)),
            static_cast<double>(corner), kRelTol);

    // The clamp the reference applies, at both ends.
    r.close("clamped high (1000 Hz corner)",
            static_cast<double>(saans::corner_to_alpha(1000.0f, 20.0f)), 0.999, kRelTol);
    r.close("clamped low (0 Hz corner)",
            static_cast<double>(saans::corner_to_alpha(0.0f, 20.0f)), 1e-6, kRelTol);

    // Python raises ValueError for a non-positive sample rate; -fno-exceptions
    // substitutes the floor of that same clamp, an EMA that never adapts.
    r.close("fs <= 0 yields the inert alpha",
            static_cast<double>(saans::corner_to_alpha(0.03f, 0.0f)), 1e-6, kRelTol);
    r.close("fs < 0 yields the inert alpha",
            static_cast<double>(saans::corner_to_alpha(0.03f, -20.0f)), 1e-6, kRelTol);

    // `alpha_to_corner` divides only by the constant 2*pi, never by anything a
    // caller supplies, so a zeroed alpha and sample rate give +0.0 and not a
    // NaN. The handoff claimed the opposite; it is asserted here so the claim
    // and the code cannot drift apart again. Telling "unconfigured" apart from
    // "a 0 Hz corner" is StaticSuppressor::effective_corner_hz's job, below.
    r.close_abs("alpha_to_corner(0, 0) is +0, not NaN",
                static_cast<double>(saans::alpha_to_corner(0.0f, 0.0f)), 0.0, 0.0);

    // What a hardcoded alpha = 0.05 actually does at 100 Hz: a corner at
    // 0.796 Hz, above the whole breathing band. The docstring makes this
    // claim, so it should be checkable rather than taken on trust.
    r.close("fixed alpha=0.05 @ 100 Hz corner",
            static_cast<double>(saans::alpha_to_corner(0.05f, 100.0f)),
            0.05 * 100.0 / (2.0 * 3.14159265358979323846), kRelTol);
    r.truth("that corner sits above the 0.5 Hz band edge",
            saans::alpha_to_corner(0.05f, 100.0f) > 0.5f,
            "0.796 Hz vs a 0.1-0.5 Hz breathing band");
}

// ---------------------------------------------------------------------------
// WelfordVariance — Welford, no vector (structural)
// ---------------------------------------------------------------------------
void test_welford_variance(Report& r) {
    r.begin("WelfordVariance: Welford", "no vector (structural)");

    saans::WelfordVariance v;
    r.equal("fresh count", static_cast<long long>(v.count()), 0);
    r.close_abs("fresh variance", static_cast<double>(v.variance()), 0.0, 0.0);
    r.close_abs("fresh std", static_cast<double>(v.std()), 0.0, 0.0);
    r.close_abs("fresh m2", static_cast<double>(v.m2()), 0.0, 0.0);

    // One sample carries no information about spread: the reference reports
    // zero rather than dividing by (count - 1) == 0.
    v.update(42.0f);
    r.equal("count after one sample", static_cast<long long>(v.count()), 1);
    r.close_abs("mean after one sample", static_cast<double>(v.mean()), 42.0, 0.0);
    r.close_abs("variance of one sample is 0", static_cast<double>(v.variance()), 0.0, 0.0);

    // Textbook set: mean 5, sample variance (n-1) = 32/7.
    v.reset();
    const float sample[8] = {2.0f, 4.0f, 4.0f, 4.0f, 5.0f, 5.0f, 7.0f, 9.0f};
    for (std::size_t i = 0; i < 8; ++i) {
        v.update(sample[i]);
    }
    r.equal("count", static_cast<long long>(v.count()), 8);
    r.close("mean", static_cast<double>(v.mean()), 5.0, kRelTol);
    r.close("m2 (sum of squared deviations)", static_cast<double>(v.m2()), 32.0, kRelTol);
    r.close("sample variance (n-1)", static_cast<double>(v.variance()), 32.0 / 7.0, kRelTol);
    r.close("std", static_cast<double>(v.std()), 2.1380899352993947, kRelTol);

    // A constant series has zero spread — the case that drives the uniform
    // weight fallback on frame 0.
    v.reset();
    for (std::size_t i = 0; i < 16; ++i) {
        v.update(0.0f);
    }
    r.close_abs("variance of a constant series", static_cast<double>(v.variance()), 0.0, 0.0);
    r.close_abs("std of a constant series", static_cast<double>(v.std()), 0.0, 0.0);

    v.reset();
    r.equal("count after reset", static_cast<long long>(v.count()), 0);
    r.close_abs("m2 after reset", static_cast<double>(v.m2()), 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// StaticSuppressor guards — the paths Python spells as `raise`
// ---------------------------------------------------------------------------
void test_suppressor_guards(Report& r) {
    r.begin("StaticSuppressor: guards and truncation", "no vector (structural)");

    saans::StaticSuppressor<4> s;
    r.truth("default-constructed is inert", !s.valid(), "valid() == false");
    // The handoff claimed this returned NaN by arithmetic accident; it did not,
    // it returned +0.0, which a caller reads as a configured 0 Hz corner.
    // Now it is a NaN on purpose.
    r.truth("inert effective_corner_hz is NaN, not 0", std::isnan(s.effective_corner_hz()),
            "explicit quiet NaN, not 0/0");

    // Python: "n_subcarriers must be positive".
    r.truth("n_subcarriers == 0 rejected", !s.configure(0, 20.0f), "configure returned false");
    // No analogue in the reference: fixed-capacity storage. Clamping silently
    // would analyse a truncated channel and report it as a whole one.
    r.truth("n_subcarriers > capacity rejected", !s.configure(5, 20.0f),
            "capacity 4, asked for 5");
    // Python: corner_to_alpha raises for a non-positive sample rate.
    r.truth("sample_rate_hz == 0 rejected", !s.configure(4, 0.0f), "configure returned false");
    r.truth("sample_rate_hz < 0 rejected", !s.configure(4, -20.0f), "configure returned false");

    // An inert suppressor writes nothing rather than emitting a plausible
    // residual, the same stance biquad.hpp takes.
    const float     amps[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    saans::CsiFrame f;
    f.amplitudes     = amps;
    f.n_subcarriers  = 4;
    f.sample_rate_hz = 20.0f;
    poison();
    r.equal("inert process writes nothing",
            static_cast<long long>(s.process(f, g_residuals, g_weights)), 0);
    r.close_abs("inert process left the buffer alone", static_cast<double>(g_residuals[0]),
                -99.0, 0.0);

    // The mirror of that for `weights`, which is public and was NOT guarded.
    // Unguarded, and with `configure` leaving its fields behind on a rejection,
    // it returned a count derived from a stale n_ and filled the buffer with
    // plausible uniform weights, which a caller following the pattern
    // `process` teaches — check the returned count — would accept. Two things
    // were wrong and both are fixed, so either fix alone now produces the
    // right answer here: this asserts the BEHAVIOUR, and deleting just the
    // `!valid_` test in `weights()` will not fail it.
    poison();
    r.equal("inert weights writes nothing", static_cast<long long>(s.weights(g_weights, 4)), 0);
    r.close_abs("inert weights left the buffer alone", static_cast<double>(g_weights[0]), -99.0,
                0.0);

    // A REJECTED configure must not leave the previous configuration readable.
    // Telemetry that reads an accessor without checking valid() would otherwise
    // report a healthy 8-subcarrier 0.03 Hz suppressor for a stage emitting
    // nothing.
    saans::StaticSuppressor<8> stale(8, 20.0f, 0.03f);
    r.truth("configured before the rejection", stale.valid(), "valid() == true");
    r.truth("configure(99) rejected", !stale.configure(99, 20.0f), "capacity 8, asked for 99");
    r.equal("rejected configure clears n_subcarriers",
            static_cast<long long>(stale.n_subcarriers()), 0);
    r.close_abs("rejected configure clears alpha", static_cast<double>(stale.alpha()), 0.0, 0.0);
    r.close_abs("rejected configure clears corner_hz", static_cast<double>(stale.corner_hz()),
                0.0, 0.0);
    r.close_abs("rejected configure clears sample_rate_hz",
                static_cast<double>(stale.sample_rate_hz()), 0.0, 0.0);
    r.truth("rejected configure makes effective_corner_hz NaN",
            std::isnan(stale.effective_corner_hz()), "no stale 0.03 Hz corner");

    r.truth("configure(4, 20 Hz) accepted", s.configure(4, 20.0f), "valid() == true");
    r.equal("capacity", static_cast<long long>(s.capacity()), 4);
    r.equal("n_subcarriers", static_cast<long long>(s.n_subcarriers()), 4);

    // "Both lists are truncated to min(frame.n_subcarriers, n_subcarriers)."
    // and — the part the docstring states and nothing checked — NOTHING is
    // written past the returned count. A caller is told the buffers must hold
    // `min(frame.n, n_subcarriers())` floats, so sizing by the frame width is
    // within the contract and an overrun would corrupt whatever follows.
    f.n_subcarriers = 2;
    poison();
    r.equal("short frame truncates to the frame",
            static_cast<long long>(s.process(f, g_residuals, g_weights)), 2);
    bool res_clean = true, wts_clean = true;
    for (std::size_t i = 2; i < kMaxSub; ++i) {
        res_clean = res_clean && (g_residuals[i] == -99.0f);
        wts_clean = wts_clean && (g_weights[i] == -99.0f);
    }
    r.truth("short frame wrote no residual past the returned count", res_clean,
            "slots 2..63 still hold the canary");
    r.truth("short frame wrote no weight past the returned count", wts_clean,
            "slots 2..63 still hold the canary");

    f.n_subcarriers = 4;
    r.equal("long frame truncates to the suppressor",
            static_cast<long long>(s.process(f, g_residuals, g_weights)), 4);
    f.n_subcarriers = 0;
    poison();
    r.equal("empty frame yields nothing",
            static_cast<long long>(s.process(f, g_residuals, g_weights)), 0);
    r.close_abs("empty frame wrote nothing", static_cast<double>(g_residuals[0]), -99.0, 0.0);

    f.n_subcarriers = 4;
    f.amplitudes    = nullptr;
    poison();
    r.equal("null amplitudes yields nothing",
            static_cast<long long>(s.process(f, g_residuals, g_weights)), 0);
    r.close_abs("null amplitudes wrote nothing", static_cast<double>(g_residuals[0]), -99.0, 0.0);
}

// ---------------------------------------------------------------------------
// Aliased output buffers
// ---------------------------------------------------------------------------
void test_aliased_buffers(Report& r) {
    r.begin("StaticSuppressor: aliased output buffers", "no vector (structural)");

    // The weights can only be computed after the residuals, from the variances
    // this very call updated, so one buffer for both silently returns weights
    // where the caller expects residuals. The Python signature it is ported
    // from, `residuals, weights = sup.process(frame)`, cannot express the
    // hazard, so a caller transcribing from the reference has no reason to
    // suspect it. It is refused, not diagnosed by corruption.
    const float     f1[4] = {50.0f, 51.0f, 52.0f, 53.0f};
    const float     f2[4] = {51.0f, 51.0f, 52.0f, 53.0f};
    const float     f3[4] = {52.0f, 51.0f, 52.0f, 53.0f};
    const float*    seq[3] = {f1, f2, f3};
    saans::CsiFrame f;
    f.n_subcarriers  = 4;
    f.sample_rate_hz = 20.0f;

    saans::StaticSuppressor<8> good(4, 20.0f, 0.03f);
    float                      res[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    float                      wts[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (std::size_t k = 0; k < 3; ++k) {
        f.amplitudes = seq[k];
        (void)good.process(f, res, wts);
    }
    // With separate buffers the third frame's residual on subcarrier 0 is the
    // real differenced value and its weight is 1 (it is the only subcarrier
    // that has moved at all). Frame 0 seeds the baseline at 50; frame 1 gives
    // r = 1 and moves the baseline to 50 + alpha; frame 2 therefore gives
    // 52 - (50 + alpha) = 2 - alpha, derived here rather than transcribed so
    // the expected value cannot quietly stop meaning anything.
    const double alpha = static_cast<double>(saans::corner_to_alpha(0.03f, 20.0f));
    r.close("separate buffers: residual[0] is 2 - alpha", static_cast<double>(res[0]),
            2.0 - alpha, 1e-6);
    r.close_abs("separate buffers: weight[0]", static_cast<double>(wts[0]), 1.0, 1e-6);

    saans::StaticSuppressor<8> aliased(4, 20.0f, 0.03f);
    float                      buf[4] = {-99.0f, -99.0f, -99.0f, -99.0f};
    std::size_t                last   = 1;
    for (std::size_t k = 0; k < 3; ++k) {
        f.amplitudes = seq[k];
        last         = aliased.process(f, buf, buf);
    }
    r.equal("one buffer for both yields nothing", static_cast<long long>(last), 0);
    r.truth("and writes nothing at all",
            buf[0] == -99.0f && buf[1] == -99.0f && buf[2] == -99.0f && buf[3] == -99.0f,
            "all four slots still hold the canary");
    r.equal("a refused frame does not advance the state",
            static_cast<long long>(aliased.variance(0).count()), 0);
    r.note("unguarded, the aliased call returned {1, 0, 0, 0}: the weight vector sitting where "
           "the residuals should be, which fuse, presence and both vital extractors would then "
           "have consumed twice over");
}

// ---------------------------------------------------------------------------
// Seeding, non-finite amplitudes, reset — exact-arithmetic behaviour
// ---------------------------------------------------------------------------
void test_suppressor_edges(Report& r) {
    r.begin("StaticSuppressor: seeding and non-finite input", "no vector (structural)");

    // Every value below is exactly representable in float32, so these are
    // exact-equality checks, not tolerance checks.
    saans::StaticSuppressor<4> s(2, 20.0f, 0.03f);
    r.truth("configured", s.valid(), "valid() == true");

    const float     seed_amps[2] = {10.0f, 10.0f};
    saans::CsiFrame f;
    f.amplitudes     = seed_amps;
    f.n_subcarriers  = 2;
    f.sample_rate_hz = 20.0f;

    poison();
    std::size_t m = s.process(f, g_residuals, g_weights);
    r.equal("first frame writes both subcarriers", static_cast<long long>(m), 2);
    r.close_abs("first frame residual [0] is zero", static_cast<double>(g_residuals[0]), 0.0, 0.0);
    r.close_abs("first frame residual [1] is zero", static_cast<double>(g_residuals[1]), 0.0, 0.0);
    r.close_abs("first frame weight [0] is uniform", static_cast<double>(g_weights[0]), 0.5, 0.0);
    r.close_abs("first frame weight [1] is uniform", static_cast<double>(g_weights[1]), 0.5, 0.0);
    r.equal("seeding still updated the variance",
            static_cast<long long>(s.variance(0).count()), 1);

    // A non-finite amplitude yields a zero residual, does NOT move the
    // baseline, and does NOT update the variance ("the reference skips the
    // variance update too"). The slot is poisoned first: without that, this
    // assertion passes whether or not `process` writes anything, because the
    // preceding frame already left 0.0 there.
    const float bad_amps[2] = {std::numeric_limits<float>::quiet_NaN(), 11.0f};
    f.amplitudes            = bad_amps;
    poison(1234.5f);
    m = s.process(f, g_residuals, g_weights);
    r.equal("non-finite frame still writes both", static_cast<long long>(m), 2);
    r.close_abs("non-finite amplitude -> zero residual (written, not stale)",
                static_cast<double>(g_residuals[0]), 0.0, 0.0);
    r.equal("non-finite amplitude skips the variance update",
            static_cast<long long>(s.variance(0).count()), 1);
    r.equal("the finite neighbour did update",
            static_cast<long long>(s.variance(1).count()), 2);
    r.close_abs("finite neighbour residual", static_cast<double>(g_residuals[1]), 1.0, 0.0);

    // Infinity takes the same path, and the same poisoning applies.
    const float inf_amps[2] = {std::numeric_limits<float>::infinity(), 11.0f};
    f.amplitudes            = inf_amps;
    poison(1234.5f);
    (void)s.process(f, g_residuals, g_weights);
    r.close_abs("infinite amplitude -> zero residual (written, not stale)",
                static_cast<double>(g_residuals[0]), 0.0, 0.0);

    // Baseline for subcarrier 0 is therefore still exactly the seed, 10.0.
    const float good_amps[2] = {10.5f, 11.0f};
    f.amplitudes             = good_amps;
    (void)s.process(f, g_residuals, g_weights);
    r.close_abs("baseline survived the bad frames untouched",
                static_cast<double>(g_residuals[0]), 0.5, 0.0);

    // reset() clears state but keeps the configuration, so the next frame
    // re-seeds rather than differencing against a stale baseline.
    s.reset();
    r.truth("reset keeps the configuration", s.valid(), "valid() == true");
    r.equal("reset cleared the variance", static_cast<long long>(s.variance(0).count()), 0);
    (void)s.process(f, g_residuals, g_weights);
    r.close_abs("first frame after reset re-seeds", static_cast<double>(g_residuals[0]), 0.0, 0.0);
}

// ---------------------------------------------------------------------------
// A non-finite amplitude as a subcarrier's FIRST sample
// ---------------------------------------------------------------------------
void test_unseeded_non_finite(Report& r) {
    r.begin("StaticSuppressor: non-finite before the baseline is seeded",
            "no vector (structural)");

    // Every other non-finite check above is made on a subcarrier a clean first
    // frame already seeded. The unseeded branch — `!is_finite(amp)` reached
    // while `seeded_[i]` is still false — is the ORDINARY startup case: a node
    // whose first frames carry a dropped or garbage subcarrier. If a NaN ever
    // reaches `seed()` the baseline is NaN for the lifetime of the node,
    // `residual()` can never recover, and the fused scalar the presence
    // detector and both vital extractors consume is permanently NaN.
    saans::StaticSuppressor<4> s(2, 20.0f, 0.03f);
    const float     frame0[2] = {std::numeric_limits<float>::quiet_NaN(), 10.0f};
    const float     frame1[2] = {50.0f, 10.5f};
    const float     frame2[2] = {50.5f, 11.0f};
    saans::CsiFrame f;
    f.n_subcarriers  = 2;
    f.sample_rate_hz = 20.0f;

    f.amplitudes = frame0;
    poison(1234.5f);
    r.equal("frame 0 writes both subcarriers",
            static_cast<long long>(s.process(f, g_residuals, g_weights)), 2);
    r.close_abs("NaN on an unseeded subcarrier -> zero residual",
                static_cast<double>(g_residuals[0]), 0.0, 0.0);
    r.equal("and leaves it unseeded (no variance update)",
            static_cast<long long>(s.variance(0).count()), 0);

    f.amplitudes = frame1;
    (void)s.process(f, g_residuals, g_weights);
    r.close_abs("the next clean frame seeds it", static_cast<double>(g_residuals[0]), 0.0, 0.0);

    f.amplitudes = frame2;
    (void)s.process(f, g_residuals, g_weights);
    // 50.0 -> 50.5 is an ordinary differenced residual, exactly representable.
    r.close_abs("and the frame after that differences normally",
                static_cast<double>(g_residuals[0]), 0.5, 0.0);
    r.truth("the subcarrier is not permanently NaN", saans::is_finite(g_residuals[0]),
            "residual is 0.5, not NaN");
    r.truth("neither is the fused scalar",
            saans::is_finite(saans::fuse(g_residuals, g_weights, 2)), "fuse() is finite");
}

// ---------------------------------------------------------------------------
// A perfectly static channel — a DOCUMENTED divergence from the reference
// ---------------------------------------------------------------------------
void test_static_channel_divergence(Report& r) {
    r.begin("StaticSuppressor: perfectly static channel (documented divergence)",
            "no vector (divergence is pinned, not matched)");

    // This is the one regime where the port and the reference do not agree and
    // no tolerance can bridge them, so it is asserted explicitly rather than
    // left to be discovered. The rearranged baseline update leaves the
    // baseline bit-identical when the residual is exactly 0, so every std is 0,
    // the `total <= 0` fallback fires and the weights are uniform. The f64
    // reference's `alpha*amp + (1-alpha)*baseline` does not return a constant
    // exactly: it accumulates ~7e-15 per frame, leaves the fallback after two
    // frames and puts all the weight on whichever subcarrier drifted first.
    //
    // Measured, for amplitudes {50, 17.3, 0.125, 1e6} held for 50 frames:
    //   this port                      [0.25,    0.25,    0.25, 0.25   ]
    //   wifisense.preprocess (f64)     [1,       0,       0,    0      ]
    //   a literal float32 transcription[5.2e-05, 1.6e-05, 0,    0.99993]
    // Three implementations, three answers, because what each is reporting
    // there is its own round-off and not a property of the channel.
    // An ESP32's CSI is quantised int8 pairs, so a still link really does
    // repeat exactly and this is reachable, not hypothetical. `fuse` agrees to
    // ~1e-13 either way, which bounds the blast radius — but RespirFiFusion
    // consumes the weight vector as its fallback_weights, so it matters.
    // spec_questions asks for a static-channel vector to settle it.
    saans::StaticSuppressor<4> s(4, 20.0f, 0.03f);
    const float     amps[4] = {50.0f, 17.3f, 0.125f, 1.0e6f};
    saans::CsiFrame f;
    f.amplitudes     = amps;
    f.n_subcarriers  = 4;
    f.sample_rate_hz = 20.0f;

    poison();
    for (std::size_t k = 0; k < 50; ++k) {
        (void)s.process(f, g_residuals, g_weights);
    }
    bool all_zero = true, all_uniform = true;
    for (std::size_t i = 0; i < 4; ++i) {
        all_zero    = all_zero && (g_residuals[i] == 0.0f);
        all_uniform = all_uniform && (g_weights[i] == 0.25f);
    }
    r.truth("every residual is exactly zero after 50 constant frames", all_zero,
            "the compensated update is bit-exact on a constant input");
    r.truth("so the weights stay on the uniform fallback", all_uniform,
            "[0.25, 0.25, 0.25, 0.25]; the f64 reference gives [1, 0, 0, 0]");
    r.close_abs("and the fused scalar is zero",
                static_cast<double>(saans::fuse(g_residuals, g_weights, 4)), 0.0, 0.0);
    r.note("this is the module's ONLY stated exception to the 1e-4 contract; parity is not "
           "claimed on a channel that repeats bit-for-bit");
}

// ---------------------------------------------------------------------------
// fuse
// ---------------------------------------------------------------------------
void test_fuse(Report& r) {
    r.begin("fuse: renormalised weighted collapse", "no vector (structural)");

    const float residuals[4] = {1.0f, 2.0f, 3.0f, 4.0f};

    // Weights that do not sum to one are renormalised over what is present:
    // (1*1 + 2*1 + 3*2 + 4*4) / 8 = 25/8.
    const float weights[4] = {1.0f, 1.0f, 2.0f, 4.0f};
    r.close("weighted mean renormalises",
            static_cast<double>(saans::fuse(residuals, weights, 4)), 25.0 / 8.0, kRelTol);

    // "a short frame does not silently scale the output down": the same
    // weights over the first two subcarriers renormalise to 1/2 each.
    r.close("short frame renormalises too",
            static_cast<double>(saans::fuse(residuals, weights, 2)), 1.5, kRelTol);

    // Zero total weight falls back to the plain mean, keeping the output in
    // amplitude units instead of collapsing to zero.
    const float zero_w[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    r.close("zero total weight -> plain mean",
            static_cast<double>(saans::fuse(residuals, zero_w, 4)), 2.5, kRelTol);

    // Python's guard is `total_w <= 0.0`, so a negative total takes the same
    // branch rather than flipping the sign of the output.
    const float neg_w[4] = {-1.0f, -1.0f, -1.0f, -1.0f};
    r.close("negative total weight -> plain mean",
            static_cast<double>(saans::fuse(residuals, neg_w, 4)), 2.5, kRelTol);

    r.close_abs("n == 0 -> 0", static_cast<double>(saans::fuse(residuals, weights, 0)), 0.0, 0.0);
    r.close_abs("null residuals -> 0", static_cast<double>(saans::fuse(nullptr, weights, 4)), 0.0,
                0.0);
    r.close_abs("null weights -> 0", static_cast<double>(saans::fuse(residuals, nullptr, 4)), 0.0,
                0.0);

    // The compensated sums are a claim about matching CPython's own compensated
    // `sum()`, and until now nothing could tell them from a plain `+=` loop:
    // on the golden vector the two agree to the last bit, and swapping either
    // NeumaierSum for a naive loop left the suite at 67/67. An ill-conditioned
    // weight vector separates them. ulp(1e8) is 8, so each of the 63 weights of
    // 3.0 is below half an ulp and vanishes entirely from a naive total.
    float big_w[kMaxSub];
    float big_r[kMaxSub];
    big_w[0] = 1.0e8f;
    big_r[0] = 1.0f;
    for (std::size_t i = 1; i < kMaxSub; ++i) {
        big_w[i] = 3.0f;
        big_r[i] = 0.0f;
    }
    double num = 0.0, den = 0.0;  // the same float32 terms, summed exactly
    for (std::size_t i = 0; i < kMaxSub; ++i) {
        num += static_cast<double>(big_r[i]) * static_cast<double>(big_w[i]);
        den += static_cast<double>(big_w[i]);
    }
    const double exact = num / den;
    // What the same function would return with plain `+=` in place of both
    // NeumaierSums, computed here so the check is a comparison and not a
    // guess at where float32 happens to land.
    float naive_num = 0.0f, naive_den = 0.0f;
    for (std::size_t i = 0; i < kMaxSub; ++i) {
        naive_den += big_w[i];
        naive_num += big_r[i] * big_w[i];
    }
    const double naive = static_cast<double>(naive_num) / static_cast<double>(naive_den);

    r.close("compensated fuse is within two float32 ulps of the exact sum",
            static_cast<double>(saans::fuse(big_r, big_w, kMaxSub)), exact, kTwoUlpRel);
    r.truth("the deviation is load-bearing: a naive += is not",
            saans::golden::rel_dev(naive, exact) > kTwoUlpRel,
            "naive " + sci(naive) + " vs exact " + sci(exact) + ", rel dev " +
                sci(saans::golden::rel_dev(naive, exact)));
}

// ---------------------------------------------------------------------------
// The weight normaliser's own compensated sum
// ---------------------------------------------------------------------------
void test_weight_normaliser_sum(Report& r) {
    r.begin("StaticSuppressor::weights: compensated normaliser", "no vector (structural)");

    // Same argument as in `fuse`, at the other site that sums a list. One
    // subcarrier swinging by +/-1e4 and 63 swinging by +/-2e-4 gives stds
    // spanning seven decades; ulp(1e4) is 9.8e-04, so every one of the 63 small
    // stds is below half an ulp of the running total and a naive loop drops all
    // of them. The expected value is the port's OWN float32 stds renormalised
    // in f64 — this asserts the summation, not the variance.
    saans::StaticSuppressor<kMaxSub> s(kMaxSub, 20.0f, 0.03f);
    float                            amps[kMaxSub];
    saans::CsiFrame                  f;
    f.amplitudes     = amps;
    f.n_subcarriers  = kMaxSub;
    f.sample_rate_hz = 20.0f;

    for (std::size_t t = 0; t < 32; ++t) {
        const float swing = ((t & 1u) == 0u) ? 1.0f : -1.0f;
        amps[0]           = 1.0e4f * swing;
        for (std::size_t i = 1; i < kMaxSub; ++i) {
            amps[i] = 2.0e-4f * swing;
        }
        (void)s.process(f, g_residuals, g_weights);
    }

    double exact_total = 0.0;  // the same float32 stds, summed exactly
    float  naive_total = 0.0f;
    for (std::size_t i = 0; i < kMaxSub; ++i) {
        exact_total += static_cast<double>(s.variance(i).std());
        naive_total += s.variance(i).std();
    }
    r.truth("the stds really do span seven decades",
            s.variance(0).std() > 1.0e3f && s.variance(1).std() < 1.0e-3f,
            "subcarrier 0 ~1e4, the rest ~4e-4");
    r.close("weight[0] is within two float32 ulps of an exact renormalisation",
            static_cast<double>(g_weights[0]),
            static_cast<double>(s.variance(0).std()) / exact_total, kTwoUlpRel);
    r.close("weight[1] likewise", static_cast<double>(g_weights[1]),
            static_cast<double>(s.variance(1).std()) / exact_total, kTwoUlpRel);

    const double naive_w0 =
        static_cast<double>(s.variance(0).std()) / static_cast<double>(naive_total);
    r.truth("the deviation is load-bearing: a naive normaliser is not",
            saans::golden::rel_dev(naive_w0,
                                   static_cast<double>(s.variance(0).std()) / exact_total) >
                kTwoUlpRel,
            "naive total " + sci(static_cast<double>(naive_total)) + " vs exact " +
                sci(exact_total) + ", weight[0] rel dev " +
                sci(saans::golden::rel_dev(
                    naive_w0, static_cast<double>(s.variance(0).std()) / exact_total)));
}

// ---------------------------------------------------------------------------
// SOAK 1 — the baseline EMA, over a million frames
// ---------------------------------------------------------------------------
void test_ema_soak(Report& r) {
    r.begin("CompensatedEma: 1e6-frame soak against an f64 EMA", "no vector (self-contained)");

    // preprocess.hpp deviates from the reference's one-line baseline update
    // twice, and the 600-frame vector cannot tell any of the three forms apart:
    // reducing `residual()` to `return x - hi;` — which makes the whole
    // compensated low half dead code — left the suite at 67/67. This soak is
    // what makes the deviation falsifiable.
    //
    // Two assertions:
    //   1. the shipped form's worst error stays under an absolute floor, and
    //   2. its error at 1e6 frames is within 2x of its error at 1e5, which is
    //      the "saturates, does not grow with uptime" claim stated as a check.
    // Plus a third that the UNCOMPENSATED incremental form misses (1), so the
    // deviation cannot be quietly deleted.
    constexpr long long kFrames = 1000000;
    constexpr long long kEarly  = 100000;

    const double a64 = kTwoPi * 0.03 / 20.0;
    const float  a32 = static_cast<float>(a64);

    double                  b64 = 0.0;
    float                   b_literal = 0.0f, b_incremental = 0.0f;
    saans::detail::CompensatedEma b_compensated;
    bool                          seeded = false;

    Lcg    rng;
    double worst_literal = 0.0, worst_incremental = 0.0, worst_compensated = 0.0;
    double early_compensated = 0.0;
    double lo = 0.0, hi = 0.0;  // full scale of the f64 residual series

    for (long long t = 0; t < kFrames; ++t) {
        const float x = soak_amplitude(t, 1.0f, 50.0f, rng);
        if (!seeded) {
            b64 = static_cast<double>(x);
            b_literal = x;
            b_incremental = x;
            b_compensated.seed(x);
            seeded = true;
        } else {
            const double r64 = static_cast<double>(x) - b64;
            b64              = a64 * static_cast<double>(x) + (1.0 - a64) * b64;

            const float r_lit = x - b_literal;
            b_literal         = a32 * x + (1.0f - a32) * b_literal;

            const float r_inc = x - b_incremental;
            b_incremental     = b_incremental + a32 * r_inc;

            const float r_cmp = b_compensated.residual(x);
            b_compensated.advance(a32, r_cmp);

            lo = (r64 < lo) ? r64 : lo;
            hi = (r64 > hi) ? r64 : hi;

            const double e_lit = std::fabs(static_cast<double>(r_lit) - r64);
            const double e_inc = std::fabs(static_cast<double>(r_inc) - r64);
            const double e_cmp = std::fabs(static_cast<double>(r_cmp) - r64);
            worst_literal      = (e_lit > worst_literal) ? e_lit : worst_literal;
            worst_incremental  = (e_inc > worst_incremental) ? e_inc : worst_incremental;
            worst_compensated  = (e_cmp > worst_compensated) ? e_cmp : worst_compensated;
        }
        if (t + 1 == kEarly) {
            early_compensated = worst_compensated;
        }
    }

    r.note("1e6 frames = 13.9 h at 20 Hz; residual full scale %.3f. max |err| vs f64: literal "
           "%.3e, incremental %.3e, compensated %.3e",
           hi - lo, worst_literal, worst_incremental, worst_compensated);
    r.close_abs("compensated baseline stays inside the soak floor", worst_compensated, 0.0,
                kSoakAbsFloor);
    r.truth("and does not grow with uptime",
            worst_compensated <= 2.0 * early_compensated,
            "err(1e6) <= 2 * err(1e5): " + sci(worst_compensated) + " vs " +
                sci(early_compensated));
    r.truth("the deviation is load-bearing: the uncompensated form misses the floor",
            worst_incremental > kSoakAbsFloor,
            "plain incremental " + sci(worst_incremental) + " > floor " +
                sci(kSoakAbsFloor));
    r.truth("and so does the literal transcription", worst_literal > kSoakAbsFloor,
            "literal " + sci(worst_literal));
    // The claim the file header used to make — sqrt(n) growth — would predict a
    // factor of ~3.2 between 1e5 and 1e6 frames for every form. It is false for
    // all three; the literal form's merit-less growth story was replaced by the
    // measured steady-state ratio above.
}

// ---------------------------------------------------------------------------
// SOAK 2 — the variance accumulator and the weights it produces
// ---------------------------------------------------------------------------
void test_variance_soak(Report& r) {
    r.begin("StaticSuppressor: 1.5e6-frame soak against the f64 reference",
            "no vector (self-contained)");

    // `m2` is the one accumulator in this stage that grows for the whole
    // uptime of the node, and nothing resets the suppressor on a schedule.
    // In plain float32 the per-frame increment falls below half an ulp of the
    // accumulator after a few hours and the adds stagnate, so `std()` — and the
    // normalised weights built from it — walk away from the reference past the
    // 1e-4 contract. The 600-frame vector is 30 seconds and cannot see it.
    //
    // The shipped suppressor, an f64 transcription of the reference and a
    // float32 Welford with a plain `+=` are all driven from the SAME float32
    // amplitudes, and the plain accumulator is fed the shipped code's own
    // residuals, so the only difference between it and `WelfordVariance` is the
    // compensation under test.
    constexpr long long kFrames = 1500000;
    constexpr std::size_t kN    = RefSuppressor::kN;

    saans::StaticSuppressor<kN> sup(kN, 20.0f, 0.03f);
    RefSuppressor               ref(20.0, 0.03);
    PlainWelford                plain[kN];
    r.truth("soak suppressor configured", sup.valid(), "valid() == true");

    // A 2:1 spread of per-subcarrier residual std, close to the golden
    // vector's own 1.83:1, so the weight vector is not degenerate.
    const float level[kN] = {48.0f, 50.0f, 52.0f, 53.0f};
    const float gain[kN]  = {1.0f, 0.8f, 0.65f, 0.5f};

    Lcg    rng;
    float  amps[kN];
    double ref_res[kN];
    double ref_wts[kN];
    saans::CsiFrame f;
    f.amplitudes     = amps;
    f.n_subcarriers  = kN;
    f.sample_rate_hz = 20.0f;

    double worst_weight_rel = 0.0;   // strong-sample relative, the contract metric
    double worst_std_rel    = 0.0;   // shipped WelfordVariance::std() vs f64
    double worst_plain_rel  = 0.0;   // the same with an uncompensated accumulator
    std::size_t short_frames = 0;

    for (long long t = 0; t < kFrames; ++t) {
        for (std::size_t i = 0; i < kN; ++i) {
            amps[i] = soak_amplitude(t, gain[i], level[i], rng);
        }
        if (sup.process(f, g_residuals, g_weights) != kN) {
            ++short_frames;
            continue;
        }
        ref.process(amps, ref_res, ref_wts);
        for (std::size_t i = 0; i < kN; ++i) {
            plain[i].update(g_residuals[i]);
        }
        // Sampled, not every frame: the comparison is 4x an f64 sqrt and this
        // loop already dominates the binary's runtime. Stagnation is monotone,
        // so a 1-in-1000 sample cannot miss the breach, only date it.
        if ((t % 1000) == 0 || t + 1 == kFrames) {
            for (std::size_t i = 0; i < kN; ++i) {
                if (ref_wts[i] >= 0.05) {  // strong samples only
                    const double rel =
                        std::fabs(static_cast<double>(g_weights[i]) - ref_wts[i]) / ref_wts[i];
                    worst_weight_rel = (rel > worst_weight_rel) ? rel : worst_weight_rel;
                }
                const double s_ref = ref.var[i].std_();
                if (s_ref > 0.0) {
                    const double rs =
                        std::fabs(static_cast<double>(sup.variance(i).std()) - s_ref) / s_ref;
                    const double rp =
                        std::fabs(static_cast<double>(plain[i].std_()) - s_ref) / s_ref;
                    worst_std_rel   = (rs > worst_std_rel) ? rs : worst_std_rel;
                    worst_plain_rel = (rp > worst_plain_rel) ? rp : worst_plain_rel;
                }
            }
        }
    }

    r.equal("every soak frame produced a full vector", static_cast<long long>(short_frames), 0);
    r.note("1.5e6 frames = 20.8 h at 20 Hz; m2 reached %.4g on subcarrier 0 (ulp %.3g)",
           static_cast<double>(sup.variance(0).m2()),
           static_cast<double>(std::nextafterf(sup.variance(0).m2(),
                                               std::numeric_limits<float>::infinity()) -
                               sup.variance(0).m2()));
    r.close_abs("compensated std() tracks the f64 reference", worst_std_rel, 0.0, kSoakRelFloor);
    r.close_abs("and so do the normalised weights", worst_weight_rel, 0.0, kSoakRelFloor);
    r.truth("the deviation is load-bearing: a plain += breaches the 1e-4 CONTRACT",
            worst_plain_rel > kRelTol,
            "uncompensated std() rel err " + sci(worst_plain_rel) + " > 1e-4");
    r.note("uncompensated, std() is %.3e relative off at 1.5e6 samples and climbing; on an "
           "8-subcarrier stream of the same shape the WEIGHTS' strong-sample error first passes "
           "1e-4 near frame 1.87e6 (25.9 h). Compensation is a mitigation, not a cure: the "
           "compensated accumulator crosses 1e-4 itself in the third week of uptime and is "
           "5.0e-03 off by 931 h (spec_questions)",
           worst_plain_rel);
}

// ---------------------------------------------------------------------------
// The parity test: 600 frames x 8 subcarriers through the whole stage
// ---------------------------------------------------------------------------
void test_static_suppressor(Report& r) {
    const Vectors v("static_suppressor");
    r.begin("StaticSuppressor + fuse: full stream", v.name());

    const std::size_t n      = static_cast<std::size_t>(v.integer("n_subcarriers"));
    const std::size_t frames = static_cast<std::size_t>(v.integer("frames"));
    const float       fs     = v.scalar_f32("fs");
    const float       corner = v.scalar_f32("corner_hz");

    if (n == 0 || n > kMaxSub) {
        saans::golden::fatal("static_suppressor wants %zu subcarriers, scratch holds %zu", n,
                             kMaxSub);
    }

    // The node only ever sees float32, so the f64 input is narrowed first and
    // the comparison is against the f64 reference OUTPUT. Narrowing the input
    // costs 3.79e-06 absolute on amplitudes near 50 — 4.4e-07 of the residual
    // series' full scale, and the floor below which no float32 port can go.
    const std::vector<float>   amps     = v.array_f32("amplitudes");
    const std::vector<double>& exp_res  = v.array("residuals");
    const std::vector<double>& exp_wts  = v.array("weights");
    const std::vector<double>& exp_fuse = v.array("fused");

    if (amps.size() != frames * n || exp_res.size() != frames * n ||
        exp_wts.size() != frames * n || exp_fuse.size() != frames) {
        saans::golden::fatal(
            "static_suppressor shape mismatch: %zu frames x %zu subcarriers, but amplitudes=%zu "
            "residuals=%zu weights=%zu fused=%zu",
            frames, n, amps.size(), exp_res.size(), exp_wts.size(), exp_fuse.size());
    }

    saans::StaticSuppressor<kMaxSub> sup(n, fs, corner);
    r.truth("suppressor configured", sup.valid(), "valid() == true");
    r.close("alpha matches the reference", static_cast<double>(sup.alpha()), v.scalar("alpha"),
            kRelTol);
    r.close("effective corner is the one asked for",
            static_cast<double>(sup.effective_corner_hz()), static_cast<double>(corner), kRelTol);

    std::vector<float> got_res(frames * n, 0.0f);
    std::vector<float> got_wts(frames * n, 0.0f);
    std::vector<float> got_fuse(frames, 0.0f);

    std::size_t short_frames = 0;
    for (std::size_t t = 0; t < frames; ++t) {
        saans::CsiFrame f;
        f.amplitudes     = &amps[t * n];
        f.phases         = nullptr;  // amplitude-only stage, by design
        f.n_subcarriers  = n;
        f.sample_index   = static_cast<std::uint32_t>(t);
        f.sample_rate_hz = fs;

        const std::size_t m = sup.process(f, g_residuals, g_weights);
        if (m != n) {
            ++short_frames;
            continue;
        }
        for (std::size_t i = 0; i < n; ++i) {
            got_res[t * n + i] = g_residuals[i];
            got_wts[t * n + i] = g_weights[i];
        }
        got_fuse[t] = saans::fuse(g_residuals, g_weights, m);
    }
    r.equal("every frame produced a full vector", static_cast<long long>(short_frames), 0);

    r.close_array("residuals", got_res.data(), exp_res.data(), exp_res.size(), kRelTol);
    r.close_array("weights", got_wts.data(), exp_wts.data(), exp_wts.size(), kRelTol);
    r.close_array("fused", got_fuse.data(), exp_fuse.data(), exp_fuse.size(), kRelTol);

    // Frame 0 is the uniform-weight fallback: every residual is zero by
    // construction, so every std is zero and 1/n is the only answer that is not
    // a division by zero. The reference agrees, and it is worth asserting
    // separately because it is the one frame where the weights carry no data.
    double worst_uniform = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        worst_uniform = (worst_uniform > std::fabs(static_cast<double>(got_wts[i]) - exp_wts[i]))
                            ? worst_uniform
                            : std::fabs(static_cast<double>(got_wts[i]) - exp_wts[i]);
    }
    r.close_abs("frame 0 uniform-weight fallback", worst_uniform, 0.0, 0.0);
    r.note("frame 0: %zu weights of %.6f, residuals all zero (nothing to compare against yet)", n,
           static_cast<double>(got_wts[0]));
}

}  // namespace

int main(int argc, char** argv) {
    // Accept the vector directory as argv[1], the convention edge/dsp/Makefile
    // uses, so this test runs identically under either Makefile in the tree.
    saans::golden::dir_from_argv(argc, argv);

    std::printf("Saans edge DSP - static suppression parity against the Python reference\n");
    std::printf("vectors: %s\n", saans::golden::dir().c_str());
    std::printf("contract: %.0e relative, float32 against an f64 reference (ADR-003)\n", kRelTol);

    Report r;

    // test_corner first: it constructs the Vectors, so a missing vector
    // directory is fatal before the soaks spend two seconds on arithmetic.
    test_corner(r);
    test_header_coexistence(r);
    test_welford_variance(r);
    test_suppressor_guards(r);
    test_aliased_buffers(r);
    test_suppressor_edges(r);
    test_unseeded_non_finite(r);
    test_static_channel_divergence(r);
    test_fuse(r);
    test_weight_normaliser_sum(r);
    test_ema_soak(r);
    test_variance_soak(r);
    test_static_suppressor(r);

    return r.finish("preprocess");
}
