// SPDX-License-Identifier: MIT
//
// Saans edge DSP — parity and composition for `SensingPipeline`.
//
// The stages themselves are checked one at a time by test_preprocess.cpp,
// test_vitals.cpp and test_foundation.cpp. What is left — and what a port
// actually gets wrong — is the WIRING: which scalar reaches which stage, in
// what order, and under what condition a stage is allowed to answer at all.
// Four claims, each with its own section below:
//
//   1. COMPOSITION. The pipeline's own residuals, weights and fused scalar
//      are the `static_suppressor` vector's, and the presence detector is fed
//      the FUSED scalar — bit for bit, not within a tolerance.
//
//   2. THE EMPTY-ROOM GATE. "Vitals are suppressed while the room reads as
//      empty: reporting a breathing rate for an unoccupied room is the classic
//      way these systems manufacture impressive-looking numbers out of noise."
//
//   3. THE BREATHING REFERENCE. The heart stage is handed the breathing
//      frequency only when the breathing estimate is itself VALID or DEGRADED,
//      and genuinely falls back to the naive band peak otherwise — checked by
//      showing the two answers DIFFER on real data, so the assertion is not
//      vacuous.
//
//   4. THE THROTTLE IS EXACT. `push` + `reading` is the same function as
//      `process`, and skipping nineteen readings out of twenty changes the
//      twentieth by nothing at all. This is the property that lets the node
//      afford the spectral stage; if it were approximate, `features_vitals`
//      would be the wrong vector to be testing against.
//
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "golden.hpp"
#include "pipeline.hpp"

using saans::golden::Report;
using saans::golden::Vectors;

namespace {

// ~17 KB each; static storage rather than a stack frame, as on the node.
saans::SensingPipeline g_pipe;
saans::SensingPipeline g_pipe_b;

/// A frame view over one row of a flattened amplitude array. Phases are zero
/// because the reference's own vectors are amplitude-only and this stage is
/// amplitude-only — `StaticSuppressor::process` documents that it does not
/// require `phases`, but `CsiFrame::valid()` (and so the gap filler) does.
struct FrameSource {
    const float* amps  = nullptr;
    const float* zeros = nullptr;
    std::size_t  n_sub = 0;
    float        fs    = 0.0f;

    saans::CsiFrame at(std::size_t i) const {
        saans::CsiFrame f;
        f.amplitudes     = amps + i * n_sub;
        f.phases         = zeros;
        f.n_subcarriers  = n_sub;
        f.sample_index   = static_cast<std::uint32_t>(i);
        f.sample_rate_hz = fs;
        return f;
    }
};

// ---------------------------------------------------------------------------
// 1. Composition
// ---------------------------------------------------------------------------

void test_composition(Report& rep) {
    const Vectors v("static_suppressor");
    rep.begin("pipeline composition: suppress -> fuse -> presence", "static_suppressor");

    const std::size_t n_sub   = static_cast<std::size_t>(v.integer("n_subcarriers"));
    const std::size_t n_frames = static_cast<std::size_t>(v.integer("frames"));
    const float       fs      = v.scalar_f32("fs");

    const std::vector<float>   amps      = v.array_f32("amplitudes");
    const std::vector<double>& residuals = v.array("residuals");
    const std::vector<double>& weights   = v.array("weights");
    const std::vector<double>& fused_ref = v.array("fused");

    if (amps.size() != n_frames * n_sub) {
        saans::golden::fatal("static_suppressor: %zu amplitudes for %zu x %zu", amps.size(),
                             n_frames, n_sub);
    }

    g_pipe.reset();
    // calibration_s does not matter here — the 600-frame vector is 30 s and
    // the detector's warmup alone is 50 s, so it never leaves warmup. That is
    // the point of using it for claim 2 below.
    const bool ok = g_pipe.configure(n_sub, fs, 30.0f);
    rep.truth("pipeline configures for the vector", ok,
              std::to_string(n_sub) + " subcarriers at " + std::to_string(fs) + " Hz");
    if (!ok) {
        return;
    }

    const std::vector<float> zeros(n_sub, 0.0f);
    FrameSource              src{amps.data(), zeros.data(), n_sub, fs};

    std::vector<float> got_res(n_frames * n_sub);
    std::vector<float> got_wts(n_frames * n_sub);
    std::vector<float> got_fused(n_frames);
    std::vector<float> got_energy(n_frames);

    // An independent detector, fed the pipeline's OWN fused scalar. If the
    // pipeline feeds presence anything else — the vitals input, a residual, a
    // re-fused copy — these two series separate immediately, and because the
    // input is identical the comparison can be exact rather than toleranced.
    saans::BasicPresenceDetector<256> shadow;
    (void)shadow.configure(fs, 30.0f);

    std::size_t shadow_mismatch = 0;
    for (std::size_t i = 0; i < n_frames; ++i) {
        const saans::PresenceState st = g_pipe.push(src.at(i));

        const std::size_t cnt = g_pipe.last_residual_count();
        if (cnt != n_sub) {
            saans::golden::fatal("pipeline wrote %zu residuals, expected %zu", cnt, n_sub);
        }
        for (std::size_t k = 0; k < n_sub; ++k) {
            got_res[i * n_sub + k] = g_pipe.last_residuals()[k];
        }
        got_fused[i]  = g_pipe.last_fused();
        got_energy[i] = st.energy;

        const saans::PresenceState ref = shadow.process(g_pipe.last_fused());
        if (ref.energy != st.energy || ref.occupied != st.occupied ||
            ref.calibrated != st.calibrated) {
            ++shadow_mismatch;
        }
    }

    rep.close_array("residuals (pipeline's own)", got_res.data(), residuals.data(),
                    got_res.size(), 1e-4);
    rep.close_array("fused (pipeline's own)", got_fused.data(), fused_ref.data(), n_frames,
                    1e-4);

    // The weights are read back once, after the whole run, from the
    // suppressor the pipeline owns — the vector's last frame's weights.
    (void)g_pipe.suppressor().weights(got_wts.data(), n_sub);
    rep.close_array("weights (final frame)", got_wts.data(),
                    weights.data() + (n_frames - 1) * n_sub, n_sub, 1e-4);

    rep.truth("presence is fed the FUSED scalar, exactly", shadow_mismatch == 0,
              std::to_string(n_frames - shadow_mismatch) + " of " + std::to_string(n_frames) +
                  " frames bit-identical to a detector driven with last_fused()");

    // With the variance selector (the only one this port implements) the
    // vitals path and the presence path see the same scalar. Pinned because
    // it is the thing that would change if "respirfi" were ever wired in, and
    // a silent divergence there would be invisible in every other test.
    rep.truth("last_vitals_input == last_fused under variance selection",
              g_pipe.last_vitals_input() == g_pipe.last_fused(),
              "a RespirFi selector would break this on purpose; nothing else should");
    rep.truth("selection reports Variance",
              g_pipe.selection() == saans::SubcarrierSelection::Variance,
              "the C++ port implements the reference's default path only");
}

// ---------------------------------------------------------------------------
// 2. The empty-room gate
// ---------------------------------------------------------------------------

void test_empty_room_gate(Report& rep) {
    const Vectors v("static_suppressor");
    rep.begin("vitals are suppressed while the room reads empty", "static_suppressor");

    const std::size_t n_sub    = static_cast<std::size_t>(v.integer("n_subcarriers"));
    const std::size_t n_frames = static_cast<std::size_t>(v.integer("frames"));
    const float       fs       = v.scalar_f32("fs");
    const std::vector<float> amps = v.array_f32("amplitudes");
    const std::vector<float> zeros(n_sub, 0.0f);
    FrameSource              src{amps.data(), zeros.data(), n_sub, fs};

    g_pipe.reset();
    (void)g_pipe.configure(n_sub, fs, 30.0f);

    std::size_t spoke = 0;
    bool        ever_occupied = false;
    for (std::size_t i = 0; i < n_frames; ++i) {
        const saans::PresenceState st = g_pipe.push(src.at(i));
        ever_occupied = ever_occupied || st.occupied;
        const saans::VitalReading r = g_pipe.reading();
        if (r.respiratory_rate.status != saans::VitalStatus::Unavailable ||
            r.heart_rate.status != saans::VitalStatus::Unavailable ||
            r.respiratory_rate.value_bpm != 0.0f || r.heart_rate.value_bpm != 0.0f ||
            r.respiratory_rate.confidence != 0.0f || r.heart_rate.confidence != 0.0f) {
            ++spoke;
        }
    }
    rep.truth("the room never reads occupied on this vector", !ever_occupied,
              "600 frames is 30 s; the detector's warmup alone is 50 s, so it is still in "
              "warmup throughout");
    rep.truth("no vital is reported for an empty room", spoke == 0,
              std::to_string(spoke) +
                  " of 600 readings carried a rate, a confidence or a status");

    // And the suppression is a CHOICE, not an accident of the window never
    // filling: turn it off and the same stream reports whatever the
    // extractors have, which at 600 frames is still Unavailable for breathing
    // (its 60-sample window is exactly full on the last frame) but is no
    // longer gated by occupancy. The claim being pinned is that the gate is
    // the `occupied` flag and nothing else.
    g_pipe.reset();
    (void)g_pipe.configure(n_sub, fs, 30.0f, -1.0f, -1.0f, /*suppress_vitals_when_empty=*/false);
    for (std::size_t i = 0; i < n_frames; ++i) {
        (void)g_pipe.push(src.at(i));
    }
    const saans::VitalReading open = g_pipe.reading();
    rep.truth("with the gate off, the extractor is consulted instead of skipped",
              open.signal_quality > 0.0f,
              "breathing fill fraction " + std::to_string(open.signal_quality) +
                  "; the gate is occupancy, not window readiness");
}

// ---------------------------------------------------------------------------
// 3 and 4. The breathing reference, and the throttle
// ---------------------------------------------------------------------------

/// Both of these need a stream that actually contains a person, so they use
/// the end-to-end recording: 70 s of empty room (which calibrates the detector
/// at the vector's 10 s calibration window) followed by a seated breather.
void test_on_a_real_recording(Report& rep) {
    const Vectors v("features_vitals");
    rep.begin("the breathing reference, and the throttle, on a real recording",
              "features_vitals");

    const std::size_t n_sub = static_cast<std::size_t>(v.integer("n_subcarriers"));
    const float       fs    = v.scalar_f32("fs");
    const float       calib = v.scalar_f32("calibration_s");
    const std::vector<float> amps = v.array_f32("amplitudes_in");
    const std::size_t        n_frames = amps.size() / n_sub;
    const std::vector<float> zeros(n_sub, 0.0f);
    FrameSource              src{amps.data(), zeros.data(), n_sub, fs};

    rep.note("%zu frames of %zu subcarriers at %g Hz, calibration %g s", n_frames, n_sub,
             static_cast<double>(fs), static_cast<double>(calib));

    g_pipe.reset();
    g_pipe_b.reset();
    const bool ok = g_pipe.configure(n_sub, fs, calib) && g_pipe_b.configure(n_sub, fs, calib);
    rep.truth("two identical pipelines configure", ok, "one reads every frame, one every 10th");
    if (!ok) {
        return;
    }

    // Scratch for the hand-computed comparison estimates. Sized by the
    // pipeline's own published budget, so it cannot drift from what
    // `reading()` uses internally.
    static float scratch[saans::SensingPipeline::kScratchFloats];

    const std::uint32_t every = 10;  // the node's 20 Hz / 2 Hz feature rate

    std::size_t throttle_diff      = 0;
    std::size_t compared           = 0;
    std::size_t gate_checked       = 0;
    std::size_t gate_wrong         = 0;
    std::size_t fallback_witnessed = 0;
    std::size_t occupied_readings  = 0;
    std::size_t usable_breathing   = 0;

    for (std::size_t i = 0; i < n_frames; ++i) {
        const saans::CsiFrame f = src.at(i);

        // A: the reference's shape — a full reading on every frame.
        const saans::VitalReading a = g_pipe.process(f);

        // B: the node's shape — push every frame, read every tenth.
        (void)g_pipe_b.push(f);
        if ((i + 1) % every != 0) {
            continue;
        }
        const saans::VitalReading b = g_pipe_b.reading();
        ++compared;

        // 4. EXACT, not approximately equal. Bit comparison on every field a
        // FeatureFrame would carry.
        if (a.respiratory_rate.value_bpm != b.respiratory_rate.value_bpm ||
            a.respiratory_rate.confidence != b.respiratory_rate.confidence ||
            a.respiratory_rate.status != b.respiratory_rate.status ||
            a.heart_rate.value_bpm != b.heart_rate.value_bpm ||
            a.heart_rate.confidence != b.heart_rate.confidence ||
            a.heart_rate.status != b.heart_rate.status || a.occupied != b.occupied ||
            a.motion_energy != b.motion_energy) {
            ++throttle_diff;
        }

        if (b.occupied) {
            ++occupied_readings;
        }

        // 3. THE BREATHING REFERENCE. Recompute the reading's two estimates by
        // hand from the extractors the pipeline owns, applying the reference's
        // rule, and require the pipeline to have done the same thing.
        if (!b.occupied) {
            continue;  // vitals are gated off; nothing to check here
        }
        ++gate_checked;
        const saans::VitalEstimate rr = g_pipe_b.breathing().estimate(
            scratch, saans::SensingPipeline::kScratchFloats);
        const bool usable = rr.status == saans::VitalStatus::Valid ||
                            rr.status == saans::VitalStatus::Degraded;
        if (usable) {
            ++usable_breathing;
        }
        const float bh = usable ? (rr.value_bpm / 60.0f) : 0.0f;
        const saans::VitalEstimate hr =
            g_pipe_b.heart().estimate(bh, scratch, saans::SensingPipeline::kScratchFloats);

        if (rr.value_bpm != b.respiratory_rate.value_bpm ||
            rr.status != b.respiratory_rate.status || hr.value_bpm != b.heart_rate.value_bpm ||
            hr.confidence != b.heart_rate.confidence || hr.status != b.heart_rate.status) {
            ++gate_wrong;
        }

        // THE ASSERTION ABOVE IS ONLY WORTH ANYTHING IF THE TWO BRANCHES
        // DIFFER. `sideband_peak` with a modulator and a naive `band_peak` can
        // agree by luck on a clean signal, in which case "the pipeline passed
        // the breathing frequency" would be unfalsifiable. Count the readings
        // where they genuinely disagree, and require some.
        if (usable) {
            const saans::VitalEstimate naive = g_pipe_b.heart().estimate(
                0.0f, scratch, saans::SensingPipeline::kScratchFloats);
            if (naive.value_bpm != hr.value_bpm || naive.confidence != hr.confidence) {
                ++fallback_witnessed;
            }
        }
    }

    rep.truth("the recording reaches an occupied state", occupied_readings > 0,
              std::to_string(occupied_readings) + " of " + std::to_string(compared) +
                  " readings saw an occupied room");
    rep.truth("THE THROTTLE IS EXACT", throttle_diff == 0,
              std::to_string(compared - throttle_diff) + " of " + std::to_string(compared) +
                  " readings bit-identical between a pipeline read every frame and one read "
                  "every " + std::to_string(every));
    rep.truth("the heart stage's breathing reference follows the reference's rule",
              gate_wrong == 0,
              std::to_string(gate_checked - gate_wrong) + " of " + std::to_string(gate_checked) +
                  " occupied readings matched a hand-applied VALID-or-DEGRADED rule (" +
                  std::to_string(usable_breathing) + " had a usable breathing estimate)");
    rep.truth("and the rule is not vacuous: the two branches really differ",
              fallback_witnessed > 0,
              std::to_string(fallback_witnessed) +
                  " reading(s) where sideband_peak with the breathing frequency returned "
                  "something other than the naive band peak");
}

// ---------------------------------------------------------------------------
// Configuration refusals
// ---------------------------------------------------------------------------

void test_inert_configurations(Report& rep) {
    rep.begin("a configuration that cannot be honoured leaves the pipeline inert",
              "(no vector)");

    static saans::BasicSensingPipeline<8, 192, 960, 256> small;
    rep.truth("more subcarriers than the template allows is refused",
              !small.configure(9, 20.0f, 30.0f), "MaxSubcarriers is 8");
    rep.truth("  and a refused pipeline is not valid()", !small.valid(), "inert, not partial");

    saans::CsiFrame f;
    float           amp = 1.0f;
    float           ph  = 0.0f;
    f.amplitudes = &amp;
    f.phases = &ph;
    f.n_subcarriers = 1;
    f.sample_rate_hz = 20.0f;
    const saans::PresenceState st = small.push(f);
    rep.truth("  and pushing into it does nothing", !st.occupied && !st.calibrated,
              "a default PresenceState, not a decision");

    rep.truth("a RespirFi request is refused rather than silently downgraded",
              !small.configure(8, 20.0f, 30.0f, -1.0f, -1.0f, true,
                               saans::SubcarrierSelection::RespirFi),
              "this port implements the variance path only; running it under the other name "
              "would be a lie the firmware could not detect");

    rep.truth("a sample rate the biquads cannot be designed for is refused",
              !small.configure(8, 0.5f, 30.0f),
              "f_high = 2 Hz is above Nyquist at 0.5 Hz");

    rep.truth("a valid configuration still works afterwards", small.configure(8, 20.0f, 30.0f),
              "a rejection leaves no state behind");
}

}  // namespace

int main(int argc, char** argv) {
    saans::golden::dir_from_argv(argc, argv);
    Report rep;

    test_composition(rep);
    test_empty_room_gate(rep);
    test_on_a_real_recording(rep);
    test_inert_configurations(rep);

    return rep.finish("pipeline");
}
