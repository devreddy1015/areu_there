// SPDX-License-Identifier: MIT
//
// Saans edge DSP — what one link costs, and how many fit.
//
// "Speed is a claim like any other" (SAANS_PLAN §4) and so is a memory figure.
// This prints `sizeof` for four shapes of the same extractor, the term-by-term
// breakdown that explains the difference, and how many links each shape leaves
// room for in the node's free DIRAM.
//
//   A  stock <56>            the library defaults at the node's subcarrier
//                            count: every other capacity still sized for a
//                            generic sample rate
//   B  tightened             node_config.hpp's capacities, each derived from
//                            the node's own constants and static_asserted
//   C  tightened + shared    the spectral scratch moved out of the link and
//                            into one arena the whole node computes in
//   D  C with no BNR         `MaxBnrRows == 0`: the row store is not in the
//                            build, and `best_bnr` goes out as 0.0 = UNKNOWN
//
// ---------------------------------------------------------------------------
// THIS IS A HOST FIGURE
// ---------------------------------------------------------------------------
// `sizeof` here is measured by the compiler that built this file, on a 64-bit
// host. The ESP32-S3 figure is SMALLER, because `std::size_t` and pointers are
// 4 bytes there rather than 8; the row-pointer array alone (`MaxBnrRows`
// pointers) halves.
//
// So do not extrapolate: `run_size_report.sh --target` cross-compiles THIS
// FILE with the firmware's own toolchain in the pinned IDF container and reads
// the four objects' sizes straight out of the object file. That is why they
// are static objects at namespace scope rather than locals — they have to be
// measurable from outside. Measured that way, A..D are 67,432 / 51,192 /
// 44,792 / 15,968 bytes a link. The firmware also logs its own
// `sizeof(Extractor)` at boot, which is the only figure that is true of a
// particular build.
//
// ---------------------------------------------------------------------------
// AND A PARITY CHECK, BECAUSE THE SIZE IS ONLY HALF THE CLAIM
// ---------------------------------------------------------------------------
// The claim is not "these are smaller", it is "these are smaller AND identical".
// So the second half of this report drives the `features_vitals` golden
// recording — 3267 CSI frames, 3% sequence loss, one 2 s outage — through all
// four shapes and compares the emitted FeatureFrames FIELD BY FIELD, exactly,
// with no tolerance anywhere. Capacities and scratch ownership are not allowed
// to move a single bit:
//
//   A vs B, A vs C            every field identical, including best_bnr
//   A vs D                    every field identical EXCEPT best_bnr, which
//   A vs C(compute_bnr=false) must be exactly 0.0 — the spec's "unknown"
//
// and the one that matters most for the shared arena: two links driven
// INTERLEAVED through one arena must emit exactly what each emits alone.
//
// `make check` is still the acceptance criterion for the port. This is the
// evidence for the memory claims specifically.
//
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "feature_extractor.hpp"
#include "golden.hpp"
#include "node_config.hpp"

namespace {

using saans::golden::Vectors;

// ---------------------------------------------------------------------------
// The four shapes
// ---------------------------------------------------------------------------

using Caps = saans::node::NodeCapacities<56, 20>;

/// A — the library defaults, told only the subcarrier count.
using StockFx = saans::BasicFeatureExtractor<56>;

/// B — tightened, but each link still owning its own spectral scratch. This is
/// the shape to use if a second sensing core ever appears; see pipeline.hpp.
using TightOwnedFx =
    saans::node::NodeExtractor<56, 20, true, saans::OwnedScratch<Caps::kScratchFloats>>;

/// C — the node build.
using NodeFx = saans::node::SaansNodeExtractor;

/// D — the node build with link quality compiled out.
using NodeNoBnrFx = saans::node::SaansNodeExtractorNoBnr;

// Static storage, as on the node: these are far too large for a stack frame,
// and on the node they are members of the link object anyway.
StockFx      g_stock;
TightOwnedFx g_tight;
NodeFx       g_node;
NodeFx       g_node_b;        // the second link, for the interleaving check
NodeFx       g_node_nobnr_rt; // BNR capable, but switched off at run time
NodeNoBnrFx  g_node_nobnr;

/// Free DIRAM the node has to spend on links, per the S6 budget.
constexpr std::size_t kFreeDiramBytes = 171u * 1024u;

// ---------------------------------------------------------------------------
// Driving the recording
// ---------------------------------------------------------------------------

struct Input {
    std::size_t         n_sub = 0;
    std::size_t         n_in  = 0;
    float               fs    = 0.0f;
    int                 noise_floor = -92;
    std::vector<double> seq;
    std::vector<double> t;
    std::vector<double> rssi;
    std::vector<float>  amps;
};

Input load_input(const Vectors& v) {
    Input in;
    in.n_sub       = static_cast<std::size_t>(v.integer("n_subcarriers"));
    in.fs          = v.scalar_f32("fs");
    in.noise_floor = static_cast<int>(v.integer("noise_floor_dbm"));
    in.seq         = v.array("seq_in");
    in.t           = v.array("t_in");
    in.rssi        = v.array("rssi_in");
    in.amps        = v.array_f32("amplitudes_in");
    in.n_in        = in.seq.size();
    if (in.t.size() != in.n_in || in.rssi.size() != in.n_in ||
        in.amps.size() != in.n_in * in.n_sub) {
        saans::golden::fatal("%s: input arrays disagree", v.name().c_str());
    }
    return in;
}

saans::FeatureExtractorConfig config_from(const Vectors& v, const Input& in) {
    saans::FeatureExtractorConfig cfg;
    cfg.n_subcarriers  = in.n_sub;
    cfg.sample_rate_hz = in.fs;
    cfg.node_id        = static_cast<std::uint8_t>(v.integer("node_id"));
    cfg.tx_id          = static_cast<std::uint8_t>(v.integer("tx_id"));
    cfg.frame_rate_hz  = v.scalar_f32("frame_rate_hz");
    cfg.vitals_enabled = v.flag("vitals_enabled");
    cfg.heart_enabled  = v.flag("heart_enabled");
    cfg.calibration_s  = v.scalar_f32("calibration_s");
    cfg.max_gap_s      = v.scalar_f32("max_gap_s");
    return cfg;
}

saans::CsiFrame view_of(const Input& in, const std::vector<float>& phases, std::size_t i) {
    saans::CsiFrame f;
    f.amplitudes     = in.amps.data() + i * in.n_sub;
    f.phases         = phases.data();
    f.n_subcarriers  = in.n_sub;
    f.sample_index   = static_cast<std::uint32_t>(in.seq[i]);
    f.sample_rate_hz = in.fs;
    f.rssi_dbm       = static_cast<std::int8_t>(in.rssi[i]);
    return f;
}

template <typename FX>
std::vector<saans::FeatureFrame> drive(FX& fx, const Input& in,
                                       const saans::FeatureExtractorConfig& cfg) {
    std::vector<saans::FeatureFrame> out;
    std::vector<float>               phases(in.n_sub, 0.0f);
    fx.reset();
    if (!fx.configure(cfg)) {
        saans::golden::fatal("configure() refused the recording's parameters");
    }
    for (std::size_t i = 0; i < in.n_in; ++i) {
        const saans::CsiFrame f = view_of(in, phases, i);
        const auto emitted = fx.push(f, static_cast<std::uint32_t>(in.seq[i]), in.t[i],
                                     in.noise_floor);
        for (std::size_t k = 0; k < emitted.count; ++k) {
            out.push_back(emitted[k]);
        }
    }
    return out;
}

/// Two links, one arena, strictly alternating pushes — the thing a shared
/// scratch has to survive. Both get the same recording, so both must emit
/// exactly what one alone emits.
template <typename FX>
void drive_interleaved(FX& a, FX& b, const Input& in,
                       const saans::FeatureExtractorConfig& cfg,
                       std::vector<saans::FeatureFrame>* out_a,
                       std::vector<saans::FeatureFrame>* out_b) {
    std::vector<float> phases(in.n_sub, 0.0f);
    a.reset();
    b.reset();
    if (!a.configure(cfg) || !b.configure(cfg)) {
        saans::golden::fatal("configure() refused the recording's parameters");
    }
    for (std::size_t i = 0; i < in.n_in; ++i) {
        const saans::CsiFrame f   = view_of(in, phases, i);
        const std::uint32_t   seq = static_cast<std::uint32_t>(in.seq[i]);
        const auto            ea  = a.push(f, seq, in.t[i], in.noise_floor);
        for (std::size_t k = 0; k < ea.count; ++k) {
            out_a->push_back(ea[k]);
        }
        const auto eb = b.push(f, seq, in.t[i], in.noise_floor);
        for (std::size_t k = 0; k < eb.count; ++k) {
            out_b->push_back(eb[k]);
        }
    }
}

// ---------------------------------------------------------------------------
// Comparing
// ---------------------------------------------------------------------------

/// Every field, exactly. `best_bnr` is compared separately by the caller so
/// that "identical except the one field we turned off" can be stated as such.
bool same_except_bnr(const saans::FeatureFrame& a, const saans::FeatureFrame& b) {
    return a.version == b.version && a.node_id == b.node_id && a.tx_id == b.tx_id &&
           a.flags == b.flags && a.seq == b.seq && a.uptime_ms == b.uptime_ms &&
           a.csi_rate_chz == b.csi_rate_chz && a.loss_permille == b.loss_permille &&
           a.motion_energy == b.motion_energy &&
           a.presence_threshold == b.presence_threshold &&
           a.gross_motion == b.gross_motion && a.breath_rms == b.breath_rms &&
           a.breathing_bpm == b.breathing_bpm && a.breathing_conf == b.breathing_conf &&
           a.heart_bpm == b.heart_bpm && a.heart_conf == b.heart_conf &&
           a.calib_age_s == b.calib_age_s && a.rssi_dbm == b.rssi_dbm &&
           a.noise_floor_dbm == b.noise_floor_dbm && a.n_subcarriers == b.n_subcarriers;
}

int g_failures = 0;

void verdict(const char* what, bool ok, const std::string& detail) {
    std::printf("  %-4s %-46s %s\n", ok ? "ok" : "FAIL", what, detail.c_str());
    if (!ok) {
        ++g_failures;
    }
}

enum class Bnr { Expected, MustBeZero };

void compare(const char* what, const std::vector<saans::FeatureFrame>& ref,
             const std::vector<saans::FeatureFrame>& got, Bnr bnr) {
    if (ref.size() != got.size()) {
        verdict(what, false,
                "frame counts differ: " + std::to_string(ref.size()) + " vs " +
                    std::to_string(got.size()));
        return;
    }
    std::size_t differing = 0;
    std::size_t bnr_wrong = 0;
    std::size_t first     = 0;
    for (std::size_t i = 0; i < ref.size(); ++i) {
        bool ok = same_except_bnr(ref[i], got[i]);
        if (bnr == Bnr::Expected) {
            ok = ok && ref[i].best_bnr == got[i].best_bnr;
        } else if (got[i].best_bnr != 0.0f) {
            ++bnr_wrong;
        }
        if (!ok) {
            if (differing == 0) {
                first = i;
            }
            ++differing;
        }
    }
    std::string detail = std::to_string(ref.size()) + " frames, " +
                         std::to_string(differing) + " differing";
    if (differing != 0) {
        detail += " (first at " + std::to_string(first) + ")";
    }
    if (bnr == Bnr::MustBeZero) {
        detail += "; best_bnr non-zero in " + std::to_string(bnr_wrong);
    }
    verdict(what, differing == 0 && bnr_wrong == 0, detail);
}

// ---------------------------------------------------------------------------
// The size table
// ---------------------------------------------------------------------------

/// How many links fit, given a per-link cost and a one-off cost paid once for
/// the whole node (the shared arena, when there is one).
std::size_t links_in_budget(std::size_t per_link, std::size_t shared_once) {
    if (per_link == 0 || shared_once >= kFreeDiramBytes) {
        return 0;
    }
    return (kFreeDiramBytes - shared_once) / per_link;
}

void row(const char* name, std::size_t per_link, std::size_t shared_once,
         std::size_t baseline) {
    const std::size_t n = links_in_budget(per_link, shared_once);
    char              saved[32];
    if (baseline == 0 || per_link > baseline) {
        std::snprintf(saved, sizeof(saved), "%s", "--");
    } else {
        std::snprintf(saved, sizeof(saved), "-%zu", baseline - per_link);
    }
    std::printf("  %-34s %9zu %10s %9zu %7zu\n", name, per_link, saved, shared_once, n);
}

/// One line of the breakdown: the same term in shapes A, C and D.
void term(const char* what, std::size_t a, std::size_t c, std::size_t d, const char* note) {
    std::printf("    %-22s %7zu %7zu %7zu   %s\n", what, a, c, d, note);
}

void size_table() {
    std::printf("\n== per-link memory, and how many links fit in %zu KB of free DIRAM\n\n",
                kFreeDiramBytes / 1024u);
    std::printf("  %-34s %9s %10s %9s %7s\n", "configuration", "bytes/link", "saved",
                "once/node", "links");
    std::printf("  %-34s %9s %10s %9s %7s\n", "----------------------------------",
                "---------", "----------", "---------", "-------");

    const std::size_t stock  = sizeof(StockFx);
    const std::size_t arena  = sizeof(saans::node::NodeArena<56, 20>::storage);

    row("A  stock <56>", stock, 0, 0);
    row("B  tightened", sizeof(TightOwnedFx), 0, stock);
    row("C  tightened + shared scratch", sizeof(NodeFx), arena, stock);
    row("D  C, best_bnr compiled out", sizeof(NodeNoBnrFx), arena, stock);

    // The term-by-term account. The three columns are the A, C and D rows
    // above, so they add up to (nearly) those totals — nearly, because a few
    // hundred bytes of scalars, filter state and the emitted-frame window are
    // not worth a line each.
    std::printf("\n  where it goes (bytes, this host)        A       C       D\n");
    term("BNR row store", sizeof(float) * 128u * 56u,
         sizeof(float) * Caps::kBnrRows * 56u, 0u, "128 -> 120 rows x 56, then none");
    term("BNR row pointers", sizeof(const float*) * 128u,
         sizeof(const float*) * Caps::kBnrRows, 0u, "halves on the 32-bit target");
    term("BNR transform + acc", sizeof(saans::BnrWorkspace<128>) + 2u * 56u * sizeof(float),
         sizeof(saans::BnrWorkspace<Caps::kBnrRows>) + 2u * 56u * sizeof(float), 0u,
         "both pad the FFT to 128");
    term("spectral scratch", saans::BasicSensingPipeline<56>::kScratchFloats * sizeof(float),
         Caps::kScratchFloats * sizeof(float), Caps::kScratchFloats * sizeof(float),
         "C and D pay this ONCE for the node, not per link");
    term("gap-fill arena", sizeof(saans::SequenceGapFiller<56, 32>::Arena),
         sizeof(typename NodeFx::GapFiller::Arena),
         sizeof(typename NodeNoBnrFx::GapFiller::Arena), "MaxFill 32 -> 20");
    term("arrival window", 512u * (sizeof(double) + sizeof(std::uint32_t)),
         Caps::kArrivals * (sizeof(double) + sizeof(std::uint32_t)),
         Caps::kArrivals * (sizeof(double) + sizeof(std::uint32_t)),
         "512 -> 201 entries (fs * 10 + 1)");
    term("breathing ring", 192u * sizeof(float), Caps::kBreathHistory * sizeof(float),
         Caps::kBreathHistory * sizeof(float), "192 -> 90 samples (45 s at 2 Hz)");
    term("cardiac ring", 960u * sizeof(float), Caps::kHeartHistory * sizeof(float),
         Caps::kHeartHistory * sizeof(float), "960 -> 600 samples (60 s at 10 Hz)");
    term("presence energy ring", 256u * sizeof(float), Caps::kEnergySamples * sizeof(float),
         Caps::kEnergySamples * sizeof(float), "256 -> 160 samples (8 s at 20 Hz)");
    term("rolling RMS (x2)", 2u * 256u * sizeof(float), 2u * Caps::kRmsSamples * sizeof(float),
         2u * Caps::kRmsSamples * sizeof(float), "256 -> 40 samples (2 s at 20 Hz)");

    std::printf("\n  node capacities, derived and static_asserted in dsp/node_config.hpp:\n");
    std::printf("    MaxBnrRows %zu  MaxArrivals %zu  MaxRmsSamples %zu  MaxFill %zu\n",
                Caps::kBnrRows, Caps::kArrivals, Caps::kRmsSamples, Caps::kFill);
    std::printf("    BreathHistory %zu  HeartHistory %zu  MaxEnergySamples %zu  "
                "scratch %zu floats\n",
                Caps::kBreathHistory, Caps::kHeartHistory, Caps::kEnergySamples,
                Caps::kScratchFloats);
}

// ---------------------------------------------------------------------------
// The parity half
// ---------------------------------------------------------------------------

void parity() {
    const Vectors v("features_vitals");
    const Input   in  = load_input(v);
    const auto    cfg = config_from(v, in);

    std::printf("\n== parity: the same %zu CSI frames through every shape\n\n", in.n_in);

    const std::vector<saans::FeatureFrame> ref = drive(g_stock, in, cfg);
    std::printf("  .. reference: stock <56>, %zu frames emitted, best_bnr %g at the end\n",
                ref.size(), static_cast<double>(g_stock.best_bnr()));
    if (ref.empty()) {
        verdict("the recording produced frames", false, "nothing to compare");
        return;
    }
    verdict("the reference run computed a link quality", g_stock.best_bnr() > 0.0f,
            "best_bnr must be non-zero here or the BNR comparisons below prove nothing");

    compare("B tightened == A", ref, drive(g_tight, in, cfg), Bnr::Expected);
    compare("C tightened + shared scratch == A", ref, drive(g_node, in, cfg), Bnr::Expected);

    // (a) the RUNTIME flag, on a build that still has the row store.
    saans::FeatureExtractorConfig no_bnr = cfg;
    no_bnr.compute_bnr                   = false;
    const std::vector<saans::FeatureFrame> rt = drive(g_node_nobnr_rt, in, no_bnr);
    compare("C with compute_bnr=false == A but for best_bnr", ref, rt, Bnr::MustBeZero);
    verdict("compute_bnr=false really is off", !g_node_nobnr_rt.computes_bnr(),
            "computes_bnr() reports what the instance is doing");

    // (b) the COMPILE-TIME form, which is the one that frees the memory.
    compare("D best_bnr compiled out == A but for best_bnr", ref,
            drive(g_node_nobnr, in, no_bnr), Bnr::MustBeZero);

    // A build with no row store must REFUSE a config that asks for BNR rather
    // than quietly reporting 0.0, which on the wire is indistinguishable from
    // "the window has not filled yet".
    g_node_nobnr.reset();
    verdict("D refuses compute_bnr=true", !g_node_nobnr.configure(cfg),
            "configure() returns false; the extractor stays INERT");

    // The shared arena, under the only condition that can break it.
    std::vector<saans::FeatureFrame> a;
    std::vector<saans::FeatureFrame> b;
    drive_interleaved(g_node, g_node_b, in, cfg, &a, &b);
    compare("two links interleaved, one arena: link A", ref, a, Bnr::Expected);
    compare("two links interleaved, one arena: link B", ref, b, Bnr::Expected);
    verdict("the two links really did share one arena",
            g_node.pipeline().scratch().data() == g_node_b.pipeline().scratch().data(),
            "same float* in both pipelines");
}

}  // namespace

int main() {
    std::printf("################################################################\n");
    std::printf("##  Saans edge DSP — per-link memory report\n");
    std::printf("##  host sizeof(), %zu-bit pointers; the ESP32-S3 figure is smaller\n",
                8u * sizeof(void*));
    std::printf("################################################################\n");
    size_table();
    parity();
    std::printf("\n----------------------------------------------------------------\n");
    if (g_failures == 0) {
        std::printf("size_report: PASS — every shape emits identical frames\n");
    } else {
        std::printf("size_report: %d CHECK(S) FAILED — a capacity or the shared arena "
                    "moved a value\n",
                    g_failures);
    }
    std::printf("----------------------------------------------------------------\n");
    return g_failures == 0 ? 0 : 1;
}
