// SPDX-License-Identifier: MIT
//
// Saans edge DSP — host throughput for the full `FeatureExtractor`.
//
// "Speed is a claim like any other" (SAANS_PLAN §4), so this reports a number
// that can be checked rather than a number that sounds good. It runs the SAME
// input as `spec/golden/bench_python.py` — the `features_vitals` recording,
// 3267 CSI frames of 32 subcarriers at 20 Hz, with its 3% sequence loss and
// its 2 s outage intact — through the same configuration, so the two figures
// are comparable rather than merely adjacent. `bench/run_bench.sh` runs both
// and prints them side by side.
//
// ---------------------------------------------------------------------------
// WHAT IS INCLUDED IN THE TIMED REGION
// ---------------------------------------------------------------------------
// Everything `bench_python.py` includes, and nothing it does not:
//
//   INCLUDED   extractor construction/configuration; gap filling; static
//              suppression; fusion; presence; both vital extractors' filters
//              and decimation; the spectral estimates at the 2 Hz feature
//              rate; the breathing-band and gross-motion RMS; the BNR scoring
//              pass; link statistics; and building each FeatureFrame.
//
//   EXCLUDED   loading the vectors from disk, and building the CsiFrame
//              views. `bench_python.py` builds its `frames` list before it
//              starts its clock, so this does too. ENCODING is excluded for
//              the same reason: the reference's loop does not encode either.
//              (It costs about 1 us per frame; `bench --encode` includes it
//              if you want to see that.)
//
// ---------------------------------------------------------------------------
// HOW IT IS MEASURED
// ---------------------------------------------------------------------------
// One untimed warm-up pass, then `--runs N` (default 5) timed passes, each a
// full reset-and-replay of the whole recording. The MEDIAN is the headline;
// min, max and the warm-up pass are printed too, because the spread is part
// of the measurement and the warm-up pass is the honest answer to "what does
// it cost the first time". No figure here is a warm-up-only timing, and none
// is a best-of.
//
// This is a HOST number on a desktop CPU. It is not an ESP32-S3 number and
// must not be quoted as one: the S3 is a 240 MHz dual-core Xtensa LX7 with a
// single-precision FPU and no SIMD, and the honest way to get its figure is
// to run this on the part. What the host number does establish is the ratio
// against the Python reference on identical input, and the cost of the
// spectral stage relative to everything else — both of which are properties
// of the algorithm rather than of the machine.
//
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "feature_extractor.hpp"
#include "golden.hpp"

namespace {

using Clock   = std::chrono::steady_clock;
using Seconds = std::chrono::duration<double>;

// 73 KB and 18 KB; static storage, as on the node.
saans::FeatureExtractor g_fx;
saans::SensingPipeline  g_pipe;

struct Recording {
    std::size_t              n_sub = 0;
    std::size_t              n     = 0;
    float                    fs    = 0.0f;
    std::vector<float>       amps;
    std::vector<float>       phases;
    std::vector<std::uint32_t> seq;
    std::vector<double>      t;
    std::vector<std::int8_t> rssi;
    std::vector<saans::CsiFrame> frames;
};

Recording load() {
    const saans::golden::Vectors v("features_vitals");
    Recording r;
    r.n_sub = static_cast<std::size_t>(v.integer("n_subcarriers"));
    r.fs    = v.scalar_f32("fs");
    r.amps  = v.array_f32("amplitudes_in");

    const std::vector<double>& seq  = v.array("seq_in");
    const std::vector<double>& t    = v.array("t_in");
    const std::vector<double>& rssi = v.array("rssi_in");
    r.n = seq.size();
    if (r.amps.size() != r.n * r.n_sub) {
        saans::golden::fatal("features_vitals: %zu amplitudes for %zu x %zu", r.amps.size(),
                             r.n, r.n_sub);
    }

    r.phases.assign(r.n_sub, 0.0f);
    r.seq.reserve(r.n);
    r.t.reserve(r.n);
    r.rssi.reserve(r.n);
    r.frames.reserve(r.n);
    for (std::size_t i = 0; i < r.n; ++i) {
        r.seq.push_back(static_cast<std::uint32_t>(seq[i]));
        r.t.push_back(t[i]);
        r.rssi.push_back(static_cast<std::int8_t>(rssi[i]));
    }
    for (std::size_t i = 0; i < r.n; ++i) {
        saans::CsiFrame f;
        f.amplitudes     = r.amps.data() + i * r.n_sub;
        f.phases         = r.phases.data();
        f.n_subcarriers  = r.n_sub;
        f.sample_index   = r.seq[i];
        f.sample_rate_hz = r.fs;
        // bench_python.py builds `CsiFrame(amps, [0.0]*n, int(seq[i]), fs)`:
        // no RSSI, no node id. Matched exactly, because `rssi_dbm` feeds a
        // running sum and a different value is a (tiny) different workload.
        f.rssi_dbm = 0;
        f.node_id  = 0;
        r.frames.push_back(f);
    }
    return r;
}

saans::FeatureExtractorConfig bench_config(const Recording& r) {
    // bench_python.py: FeatureExtractor(n_subcarriers=n, sample_rate_hz=fs,
    //                                   vitals_enabled=True, heart_enabled=True,
    //                                   calibration_s=10.0)
    // node_id and tx_id are left at their defaults there, so they are here.
    saans::FeatureExtractorConfig cfg;
    cfg.n_subcarriers  = r.n_sub;
    cfg.sample_rate_hz = r.fs;
    cfg.vitals_enabled = true;
    cfg.heart_enabled  = true;
    cfg.calibration_s  = 10.0f;
    return cfg;
}

/// One full replay. Returns the seconds taken; `emitted` and `checksum` exist
/// so the optimiser cannot decide the work is dead and delete it.
double run_extractor(const Recording& r, bool encode, std::size_t* emitted,
                     double* checksum) {
    const saans::FeatureExtractorConfig cfg = bench_config(r);
    std::size_t                         out = 0;
    double                              sum = 0.0;
    std::uint8_t                        wire[saans::kFeatureFrameSize];

    const Clock::time_point t0 = Clock::now();
    if (!g_fx.configure(cfg)) {
        saans::golden::fatal("bench: FeatureExtractor refused the benchmark configuration");
    }
    for (std::size_t i = 0; i < r.n; ++i) {
        const auto e = g_fx.push(r.frames[i], r.seq[i], r.t[i], -92);
        for (std::size_t k = 0; k < e.count; ++k) {
            ++out;
            sum += static_cast<double>(e[k].motion_energy) +
                   static_cast<double>(e[k].breathing_bpm);
            if (encode) {
                g_fx.encode(e[k], wire);
                sum += static_cast<double>(wire[20]);
            }
        }
    }
    const double elapsed = Seconds(Clock::now() - t0).count();
    *emitted             = out;
    *checksum            = sum;
    return elapsed;
}

/// The pipeline alone, pushed but never read: the per-frame stateful cost with
/// no spectral estimation at all. The lower bound on what a node must pay.
double run_push_only(const Recording& r, double* checksum) {
    double sum = 0.0;
    const Clock::time_point t0 = Clock::now();
    if (!g_pipe.configure(r.n_sub, r.fs, 10.0f)) {
        saans::golden::fatal("bench: SensingPipeline refused the benchmark configuration");
    }
    for (std::size_t i = 0; i < r.n; ++i) {
        const saans::PresenceState s = g_pipe.push(r.frames[i]);
        sum += static_cast<double>(s.energy);
    }
    const double elapsed = Seconds(Clock::now() - t0).count();
    *checksum            = sum;
    return elapsed;
}

/// The pipeline with an estimate on EVERY frame — what the node would pay
/// without the push/reading split described in pipeline.hpp. The difference
/// between this and `run_push_only` is the spectral stage; the difference
/// between this and the extractor is what the throttle buys.
double run_estimate_every_frame(const Recording& r, double* checksum) {
    double sum = 0.0;
    const Clock::time_point t0 = Clock::now();
    if (!g_pipe.configure(r.n_sub, r.fs, 10.0f)) {
        saans::golden::fatal("bench: SensingPipeline refused the benchmark configuration");
    }
    for (std::size_t i = 0; i < r.n; ++i) {
        const saans::VitalReading v = g_pipe.process(r.frames[i]);
        sum += static_cast<double>(v.respiratory_rate.value_bpm);
    }
    const double elapsed = Seconds(Clock::now() - t0).count();
    *checksum            = sum;
    return elapsed;
}

struct Stats {
    double warmup = 0.0;
    double median = 0.0;
    double best   = 0.0;
    double worst  = 0.0;
};

Stats summarise(double warmup, std::vector<double> runs) {
    Stats s;
    s.warmup = warmup;
    std::sort(runs.begin(), runs.end());
    s.best   = runs.front();
    s.worst  = runs.back();
    s.median = (runs.size() % 2 == 1)
                   ? runs[runs.size() / 2]
                   : 0.5 * (runs[runs.size() / 2 - 1] + runs[runs.size() / 2]);
    return s;
}

void report(const char* what, const Recording& r, const Stats& s) {
    const double fps  = static_cast<double>(r.n) / s.median;
    const double xrt  = fps / static_cast<double>(r.fs);
    const double us   = s.median * 1e6 / static_cast<double>(r.n);
    std::printf("  %-34s  %9.0f frames/s  %8.0fx realtime  %7.2f us/frame\n", what, fps, xrt,
                us);
    std::printf("  %-34s  median %.4f s   (warm-up %.4f s, best %.4f s, worst %.4f s)\n", "",
                s.median, s.warmup, s.best, s.worst);
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t runs   = 5;
    bool        encode = false;
    std::string dir;

    for (int i = 1; i < argc; ++i) {
        const std::string a(argv[i]);
        if (a == "--encode") {
            encode = true;
        } else if (a == "--runs" && i + 1 < argc) {
            runs = static_cast<std::size_t>(std::atoi(argv[++i]));
        } else if (a.rfind("--", 0) != 0) {
            dir = a;  // the vector directory, as every test here takes it
        }
    }
    if (runs < 1) {
        runs = 1;
    }
    if (!dir.empty()) {
        saans::golden::set_dir(dir);
    }

    const Recording r = load();

    std::printf("\n");
    std::printf("================================================================\n");
    std::printf("  Saans edge DSP host benchmark\n");
    std::printf("================================================================\n");
    std::printf("  input      %zu CSI frames, %zu subcarriers, %g Hz  [features_vitals]\n", r.n,
                r.n_sub, static_cast<double>(r.fs));
    std::printf("             = %.1f s of one link, with 3%% sequence loss and a 2 s outage\n",
                static_cast<double>(r.n) / static_cast<double>(r.fs));
    std::printf("  config     vitals + heart enabled, calibration 10 s, features at 2 Hz\n");
    std::printf("  method     1 untimed warm-up pass, then %zu timed passes; MEDIAN reported\n",
                runs);
    std::printf("  included   gap fill, suppression, fusion, presence, both vital stages,\n");
    std::printf("             the 2 Hz spectral estimates, RMS windows, BNR, link stats%s\n",
                encode ? ", ENCODING" : "");
    std::printf("  excluded   vector loading and CsiFrame construction (as bench_python.py)%s\n",
                encode ? "" : ", encoding");
    std::printf("  build      %s, %s\n",
#if defined(__clang__)
                "clang++ " __clang_version__,
#elif defined(__GNUC__)
                "g++ " __VERSION__,
#else
                "unknown compiler",
#endif
                "-O2 -fno-exceptions -fno-rtti, float32 throughout");
    std::printf("\n");

    std::size_t emitted  = 0;
    double      checksum = 0.0;

    const double warm = run_extractor(r, encode, &emitted, &checksum);
    std::vector<double> timings;
    for (std::size_t i = 0; i < runs; ++i) {
        std::size_t e = 0;
        double      c = 0.0;
        timings.push_back(run_extractor(r, encode, &e, &c));
        if (e != emitted) {
            saans::golden::fatal("bench: run %zu emitted %zu frames, warm-up emitted %zu — the "
                                 "benchmark is not deterministic and its numbers mean nothing",
                                 i, e, emitted);
        }
    }
    const Stats fx = summarise(warm, timings);

    double pw = 0.0;
    double warm_push = run_push_only(r, &pw);
    timings.clear();
    for (std::size_t i = 0; i < runs; ++i) {
        timings.push_back(run_push_only(r, &pw));
    }
    const Stats push = summarise(warm_push, timings);

    double ew = 0.0;
    double warm_every = run_estimate_every_frame(r, &ew);
    timings.clear();
    for (std::size_t i = 0; i < runs; ++i) {
        timings.push_back(run_estimate_every_frame(r, &ew));
    }
    const Stats every = summarise(warm_every, timings);

    std::printf("  FULL FeatureExtractor (the headline figure)\n");
    report("full extractor", r, fx);
    std::printf("\n");
    std::printf("  WHAT THE PARTS COST\n");
    report("pipeline push only, no estimate", r, push);
    report("pipeline, estimate EVERY frame", r, every);
    std::printf("\n");
    std::printf("  The middle row is the per-frame work a node cannot avoid. The bottom row\n");
    std::printf("  is what it would pay WITHOUT the push/reading split (pipeline.hpp): the\n");
    std::printf("  spectral estimates are %.1fx the stateful cost, and computing them 20\n",
                (every.median - push.median) / push.median);
    std::printf("  times a second instead of twice makes the pipeline %.2fx slower.\n",
                every.median / fx.median);
    std::printf("\n");
    std::printf("  emitted %zu FeatureFrames per pass (checksum %.6f, printed so the\n", emitted,
                checksum);
    std::printf("  optimiser cannot delete the work being measured)\n");
    std::printf("================================================================\n");

    // Machine-readable, for bench/run_bench.sh.
    std::printf("BENCH cpp_frames_per_s=%.1f cpp_xrt=%.2f cpp_median_s=%.6f frames=%zu fs=%g "
                "emitted=%zu\n",
                static_cast<double>(r.n) / fx.median,
                (static_cast<double>(r.n) / fx.median) / static_cast<double>(r.fs), fx.median,
                r.n, static_cast<double>(r.fs), emitted);
    return 0;
}
