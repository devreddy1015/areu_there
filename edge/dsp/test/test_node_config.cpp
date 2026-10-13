// SPDX-License-Identifier: MIT
//
// Saans edge DSP — the NODE's shape, checked rather than asserted.
//
// `edge/dsp/node_config.hpp` cuts every capacity of `BasicFeatureExtractor`
// down to what one node's pinned constants require, and proves each cut with a
// static_assert. That is the right instinct and it is not sufficient, for a
// reason the header itself names: `constexpr` cannot call the library's own
// sizing helpers, so the derivations are MIRRORS of them, and a static_assert
// comparing a capacity against the same mirror that defined it proves
// `kX >= kX`. One of those mirrors was wrong by one — the arrival ring was
// sized for the pruned window when the code pushes before it prunes — and
// every assert passed over it while a healthy 20 Hz link evicted an arrival on
// every single frame.
//
// This file is the half that can only be done at run time. Five claims:
//
//   1. THE MIRROR AGREES WITH THE LIBRARY. Every derivation in `derive` is
//      compared against the function it mirrors — `required_history`,
//      `required_energy_samples` — across Kconfig's whole 10..200 Hz range,
//      and every rate in that range is proved to CONFIGURE rather than leave
//      an inert node. Where no helper exists, `configure()` is the oracle:
//      it re-derives everything and refuses what does not fit.
//
//   2. THE ARRIVAL RING HOLDS THE PEAK. Driven by a steady, lossless stream at
//      exactly the configured rate — the node's own nominal case, which is the
//      one that exposes an off-by-one, and the one the golden recording cannot
//      because it runs 4% slow — `arrival_overflows` must be zero. The
//      capacity one slot smaller must NOT be zero, so the check cannot pass
//      for the wrong reason.
//
//   3. THE NODE SHAPE IS BIT-IDENTICAL TO THE STOCK ONE. The golden recording
//      through `SaansNodeExtractor` and through `BasicFeatureExtractor<56>`,
//      encoded, byte for byte. A capacity is a bound on what can be stored; if
//      one of them is also changing a value, this is where it shows.
//
//   4. NO BNR MEANS NO BNR AND NOTHING ELSE. `MaxBnrRows == 0` plus
//      `compute_bnr = false` moves `best_bnr` to 0.0 and no other field, and a
//      build with no row store REFUSES a configuration that asks for link
//      quality rather than quietly reporting nothing.
//
//   5. THE SHARED ARENA IS WRITE-BEFORE-READ. Two links interleaved through
//      one `SharedScratch`, fed DIFFERENT data, must each emit what they emit
//      alone. Feeding both the same recording — which is what the size report
//      does — cannot fail for the reason the check exists: a stale float left
//      by the other link would be bit-identical to the one the solo run left.
//
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "golden.hpp"
#include "node_config.hpp"

using saans::golden::Report;
using saans::golden::Vectors;

namespace {

// ---------------------------------------------------------------------------
// Shapes under test
// ---------------------------------------------------------------------------

using Caps = saans::node::NodeCapacities<56, 20>;

/// The node's own type, and the library's defaults, for the parity run.
saans::node::SaansNodeExtractor      g_node;
saans::node::SaansNodeExtractor      g_node_b;
saans::node::SaansNodeExtractorNoBnr g_node_nobnr;
saans::BasicFeatureExtractor<56>     g_stock;

/// The arrival ring at exactly the peak occupancy, and one slot below it.
/// Everything else is the node's. These exist so claim 2 can show that the
/// invariant it asserts is reachable AND that it was not reachable before.
template <std::size_t Arrivals>
using ArrivalShape =
    saans::BasicFeatureExtractor<8, Caps::kBnrRows, Arrivals, Caps::kRmsSamples, Caps::kFill,
                                 Caps::kBreathHistory, Caps::kHeartHistory,
                                 Caps::kEnergySamples>;

ArrivalShape<202> g_arr_peak;
ArrivalShape<201> g_arr_short;

// ---------------------------------------------------------------------------
// A synthetic stream — the case the golden recording cannot provide
// ---------------------------------------------------------------------------
//
// `spec/golden/features_vitals` has a mean inter-arrival time of 52.0 ms, i.e.
// 19.2 Hz against a configured 20. Its largest closed 10 s window therefore
// holds 200 arrivals and its PRE-TRIM peak is exactly 201 — which is why the
// one recording the capacity was measured on is the one recording that cannot
// expose an off-by-one in it. A healthy link is the case that trips it, so the
// check needs a stream it can specify exactly.

struct Synth {
    std::size_t n_sub = 8;
    float       fs    = 20.0f;
    std::size_t n     = 0;
    /// Inter-arrival jitter as a fraction of the nominal period; 0 is the
    /// perfectly regular clock the reference's own `t_s=None` path assumes.
    float  jitter = 0.0f;
    /// Fraction of sequence numbers never delivered.
    float  loss   = 0.0f;
};

/// A deterministic unit-interval generator. Not `rand()`: the two compilers
/// this suite builds under must drive byte-identical streams or a parity check
/// between them means nothing.
struct Lcg {
    std::uint32_t s = 12345u;
    float next() {
        s = s * 1664525u + 1013904223u;
        return static_cast<float>(s >> 8) * (1.0f / 16777216.0f);
    }
};

/// Push `cfg.n` frames and return the extractor's overflow count. The
/// amplitudes carry a slow breathing-band modulation so the pipeline has
/// something to do; nothing here checks the vitals, only the link statistics.
template <typename FX>
std::uint32_t drive_synth(FX& fx, const Synth& cfg, std::vector<saans::FeatureFrame>* out) {
    Lcg                rng;
    std::vector<float> amp(cfg.n_sub, 0.0f);
    std::vector<float> ph(cfg.n_sub, 0.0f);
    const double       period = 1.0 / static_cast<double>(cfg.fs);
    double             t      = 0.0;
    std::uint32_t      seq    = 0;

    for (std::size_t i = 0; i < cfg.n; ++i) {
        const double step =
            period * (1.0 + static_cast<double>(cfg.jitter) *
                                (2.0 * static_cast<double>(rng.next()) - 1.0));
        t += step;
        ++seq;
        if (cfg.loss > 0.0f && rng.next() < cfg.loss) {
            continue;  // the sequence number is burned, the arrival never happens
        }
        const float ti = static_cast<float>(i);
        for (std::size_t k = 0; k < cfg.n_sub; ++k) {
            amp[k] = 20.0f + 0.4f * std::sin(0.08f * ti + 0.3f * static_cast<float>(k)) +
                     0.05f * std::sin(1.1f * ti);
        }
        saans::CsiFrame f;
        f.amplitudes     = amp.data();
        f.phases         = ph.data();
        f.n_subcarriers  = cfg.n_sub;
        f.sample_index   = seq;
        f.sample_rate_hz = cfg.fs;
        f.rssi_dbm       = -55;
        const auto emitted = fx.push(f, seq, t, -92);
        if (out != nullptr) {
            for (std::size_t k = 0; k < emitted.count; ++k) {
                out->push_back(emitted[k]);
            }
        }
    }
    return fx.arrival_overflows;
}

saans::FeatureExtractorConfig synth_config(const Synth& cfg, bool compute_bnr = true) {
    saans::FeatureExtractorConfig c;
    c.n_subcarriers  = cfg.n_sub;
    c.sample_rate_hz = cfg.fs;
    c.frame_rate_hz  = 2.0f;
    c.vitals_enabled = true;
    c.heart_enabled  = true;
    c.compute_bnr    = compute_bnr;
    return c;
}

// ---------------------------------------------------------------------------
// 1. The mirror against the library
// ---------------------------------------------------------------------------
//
// `derive::band_history` and `derive::energy_samples` restate
// `BandVitalExtractor::required_history` and
// `BasicPresenceDetector::required_energy_samples` operation for operation,
// including which step is float32 and which is f64. Those are ordinary runtime
// functions, so this is the only place the two can be compared at all.

struct Drift {
    unsigned    first_rate = 0;
    long long   got        = 0;
    long long   want       = 0;
    unsigned    n_bad      = 0;

    void feed(unsigned rate, long long g, long long w) {
        if (g == w) {
            return;
        }
        if (n_bad == 0) {
            first_rate = rate;
            got        = g;
            want       = w;
        }
        ++n_bad;
    }
    std::string detail(const char* what) const {
        if (n_bad == 0) {
            return std::string(what) + " agrees at all 191 rates in 10..200 Hz";
        }
        char buf[256];
        std::snprintf(buf, sizeof(buf),
                      "%u rate(s) disagree; first at %u Hz: mirror says %lld, the library "
                      "wants %lld",
                      n_bad, first_rate, got, want);
        return std::string(buf);
    }
};

/// Peak occupancy of the link-statistics ring, MEASURED by running the same
/// window maintenance the extractor runs — push, then prune — rather than by
/// restating the formula that is under test.
std::uint32_t measured_peak(float fs, float window_s, std::size_t n_frames) {
    std::vector<double> win;
    win.reserve(n_frames);
    std::size_t  head = 0;  // index of the oldest entry still held
    std::uint32_t peak = 0;
    for (std::size_t i = 0; i < n_frames; ++i) {
        const double t = static_cast<double>(i) / static_cast<double>(fs);
        win.push_back(t);  // THE PUSH PRECEDES THE PRUNE, as push_impl does
        const std::uint32_t held = static_cast<std::uint32_t>(win.size() - head);
        if (held > peak) {
            peak = held;
        }
        const double cutoff = t - static_cast<double>(window_s);
        while (head < win.size() && win[head] < cutoff) {
            ++head;
        }
    }
    return peak;
}

template <unsigned Rate>
void sweep_one(Drift& breath, Drift& heart, Drift& energy, Drift& arrivals) {
    using C         = saans::node::NodeCapacities<8, Rate>;
    const float fs  = static_cast<float>(Rate);

    // Big enough that `required_history` is never the binding constraint; what
    // is being compared is the COUNT it returns, not whether it fits.
    using Band  = saans::BandVitalExtractor<16384>;
    using Heart = saans::BasicHeartRateExtractor<16384>;
    using Pres  = saans::BasicPresenceDetector<16384>;

    breath.feed(Rate, static_cast<long long>(C::kBreathHistory),
                static_cast<long long>(Band::required_history(
                    saans::kBreathBandLowHz, saans::kBreathBandHighHz, fs)));
    heart.feed(Rate, static_cast<long long>(C::kHeartHistory),
               static_cast<long long>(Heart::required_history(
                   saans::node::kHeartBandLowHz, saans::node::kHeartBandHighHz, fs)));
    energy.feed(Rate, static_cast<long long>(C::kEnergySamples),
                static_cast<long long>(
                    Pres::required_energy_samples(fs, saans::node::kEnergyWindowS)));

    // The arrival ring is the one capacity with no library-side helper, so it
    // is checked against a MEASUREMENT of the data structure instead: 30 s of
    // a steady stream at this rate, with the same push-then-prune order.
    const std::uint32_t peak =
        measured_peak(fs, saans::kLinkStatsWindowS, static_cast<std::size_t>(Rate) * 30u);
    arrivals.feed(Rate,
                  static_cast<long long>(saans::node::derive::arrival_peak(
                      static_cast<double>(fs), saans::kLinkStatsWindowS)),
                  static_cast<long long>(peak));
}

template <unsigned Rate>
struct Sweep {
    static void run(Drift& b, Drift& h, Drift& e, Drift& a) {
        sweep_one<Rate>(b, h, e, a);
        Sweep<Rate + 1>::run(b, h, e, a);
    }
};
template <>
struct Sweep<201> {
    static void run(Drift&, Drift&, Drift&, Drift&) {}
};

/// `configure()` is the oracle for the capacities with no helper — it
/// re-derives the BNR row count, the gap-fill arena and the RMS windows and
/// returns false rather than analysing a shortened one. A node that cannot
/// configure boots INERT, which from the gateway is indistinguishable from an
/// empty room, so "every Kconfig-permitted rate configures" is a real claim.
template <unsigned Rate>
bool configures_at() {
    static saans::node::NodeExtractor<8, Rate> fx;
    saans::FeatureExtractorConfig              c;
    c.n_subcarriers  = 8;
    c.sample_rate_hz = static_cast<float>(Rate);
    c.frame_rate_hz  = 2.0f;
    fx.reset();
    return fx.configure(c);
}

void test_mirror(Report& rep) {
    rep.begin("the constexpr mirror against the library's own sizing", "(no vector)");

    Drift breath, heart, energy, arrivals;
    Sweep<10>::run(breath, heart, energy, arrivals);

    rep.truth("kBreathHistory == BandVitalExtractor::required_history",
              breath.n_bad == 0, breath.detail("the breathing ring"));
    rep.truth("kHeartHistory == BasicHeartRateExtractor::required_history",
              heart.n_bad == 0, heart.detail("the cardiac ring"));
    rep.truth("kEnergySamples == required_energy_samples", energy.n_bad == 0,
              energy.detail("the presence energy window"));
    rep.truth("derive::arrival_peak == the ring's measured peak occupancy",
              arrivals.n_bad == 0,
              arrivals.detail("the pre-trim peak of a steady stream"));

    // A handful of rates across the whole range rather than all 191: each one
    // is a distinct extractor object, and the claim is about the derivation,
    // which the sweep above has already covered at every rate.
    const bool all_configure = configures_at<10>() && configures_at<12>() &&
                               configures_at<15>() && configures_at<20>() &&
                               configures_at<25>() && configures_at<32>() &&
                               configures_at<50>() && configures_at<64>() &&
                               configures_at<100>() && configures_at<128>() &&
                               configures_at<200>();
    rep.truth("every sampled Kconfig rate configures rather than booting inert",
              all_configure,
              "10, 12, 15, 20, 25, 32, 50, 64, 100, 128 and 200 Hz; configure() "
              "re-derives every capacity and refuses what does not fit");

    // Non-vacuity: configure() really does refuse a capacity that is short, so
    // the line above is an assertion about the derivations and not about
    // configure() being permissive.
    static saans::BasicFeatureExtractor<8, 8, 202, 40, 20, 90, 600, 160> starved;
    saans::FeatureExtractorConfig c;
    c.n_subcarriers  = 8;
    c.sample_rate_hz = 20.0f;
    c.frame_rate_hz  = 2.0f;
    rep.truth("and configure() refuses a short one", !starved.configure(c),
              "8 BNR rows where 20 Hz over a 30 s window needs 120");

    rep.note("node capacities at 56 subcarriers / 20 Hz: bnr %zu, arrivals %zu, rms %zu, "
             "fill %zu, breath %zu, heart %zu, energy %zu, scratch %zu",
             Caps::kBnrRows, Caps::kArrivals, Caps::kRmsSamples, Caps::kFill,
             Caps::kBreathHistory, Caps::kHeartHistory, Caps::kEnergySamples,
             Caps::kScratchFloats);
    rep.note("sizeof: node %zu B/link, node without best_bnr %zu, stock <56> %zu, "
             "shared arena %zu B once per node",
             sizeof(saans::node::SaansNodeExtractor),
             sizeof(saans::node::SaansNodeExtractorNoBnr),
             sizeof(saans::BasicFeatureExtractor<56>),
             Caps::kScratchFloats * sizeof(float));
}

// ---------------------------------------------------------------------------
// 2. The arrival ring holds the peak
// ---------------------------------------------------------------------------

void test_arrival_window(Report& rep) {
    rep.begin("the 10 s link-statistics window on a healthy link", "(no vector)");

    Synth s;
    s.n  = 2400;  // 120 s at 20 Hz
    s.fs = 20.0f;

    // The capacity equal to the measured peak: zero overflows, which is what
    // `arrival_overflows` has to be able to reach for its non-zero to mean
    // anything.
    g_arr_peak.reset();
    const bool ok_peak = g_arr_peak.configure(synth_config(s));
    const std::uint32_t at_peak = ok_peak ? drive_synth(g_arr_peak, s, nullptr) : 1u;
    rep.truth("a ring sized for the PEAK never evicts", ok_peak && at_peak == 0,
              std::to_string(at_peak) + " eviction(s) over " + std::to_string(s.n) +
                  " frames of a steady, lossless 20 Hz stream (202 = 20*10 closed at "
                  "both ends, +1 because the push precedes the prune)");

    // One slot less — the figure the node shipped with — evicts constantly.
    // Without this the check above could pass for any capacity at all.
    g_arr_short.reset();
    const bool ok_short = g_arr_short.configure(synth_config(s));
    const std::uint32_t at_short = ok_short ? drive_synth(g_arr_short, s, nullptr) : 0u;
    rep.truth("and one slot less evicts on almost every frame",
              ok_short && at_short > s.n / 2u,
              std::to_string(at_short) + " eviction(s) at 201 — the window length rather "
              "than the peak occupancy, which is the off-by-one this check exists for");

    // The node's own shape, with the jitter headroom on top: zero at nominal,
    // and still zero once the arrivals are irregular. The count inside a fixed
    // span fluctuates above its mean even when the mean is exactly nominal, so
    // the peak is not the whole requirement.
    Synth n56   = s;
    n56.n_sub   = 56;
    g_node.reset();
    const bool node_ok = g_node.configure(synth_config(n56));
    const std::uint32_t node_steady = node_ok ? drive_synth(g_node, n56, nullptr) : 1u;
    rep.truth("SaansNodeExtractor never evicts at its nominal rate",
              node_ok && node_steady == 0,
              std::to_string(node_steady) + " eviction(s); the node's capacity is " +
                  std::to_string(Caps::kArrivals));

    Synth jit  = n56;
    jit.jitter = 0.2f;  // +-20%, the golden recording's own shape
    jit.loss   = 0.03f;
    jit.n      = 4000;
    g_node.reset();
    const bool jit_ok = g_node.configure(synth_config(jit));
    const std::uint32_t node_jitter = jit_ok ? drive_synth(g_node, jit, nullptr) : 1u;
    rep.truth("nor under +-20% arrival jitter with 3% loss",
              jit_ok && node_jitter == 0,
              std::to_string(node_jitter) + " eviction(s) over " + std::to_string(jit.n) +
                  " frames; the headroom in derive::arrivals is what covers this");

    // And the jittered stream's emitted link statistics are the stock
    // extractor's, which is the field-level consequence the counter stands in
    // for. 512 arrivals is far more than the window ever needs here.
    std::vector<saans::FeatureFrame> node_frames;
    g_node.reset();
    (void)g_node.configure(synth_config(jit));
    drive_synth(g_node, jit, &node_frames);

    std::vector<saans::FeatureFrame> stock_frames;
    g_stock.reset();
    const bool stock_ok = g_stock.configure(synth_config(jit));
    if (stock_ok) {
        drive_synth(g_stock, jit, &stock_frames);
    }
    std::size_t rate_diff = 0;
    std::size_t loss_diff = 0;
    for (std::size_t i = 0; i < node_frames.size() && i < stock_frames.size(); ++i) {
        if (node_frames[i].csi_rate_chz != stock_frames[i].csi_rate_chz) {
            ++rate_diff;
        }
        if (node_frames[i].loss_permille != stock_frames[i].loss_permille) {
            ++loss_diff;
        }
    }
    rep.truth("csi_rate_chz and loss_permille match the stock 512-entry window",
              stock_ok && !node_frames.empty() &&
                  node_frames.size() == stock_frames.size() && rate_diff == 0 &&
                  loss_diff == 0,
              std::to_string(node_frames.size()) + " frames, " +
                  std::to_string(rate_diff) + " rate and " + std::to_string(loss_diff) +
                  " loss figures differing");
}

// ---------------------------------------------------------------------------
// The golden recording
// ---------------------------------------------------------------------------

struct Input {
    std::size_t         n_sub = 0;
    std::size_t         n_in  = 0;
    float               fs    = 0.0f;
    std::vector<double> seq;
    std::vector<double> t;
    std::vector<double> rssi;
    std::vector<float>  amps;
    int                 noise_floor = -92;
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

/// One frame of the recording, optionally transformed. The transform is what
/// makes the interleaving check able to fail: two links carrying the same
/// numbers cannot detect that one of them read a float the other wrote.
struct Source {
    const Input* in        = nullptr;
    std::vector<float>* buf = nullptr;
    bool   transform       = false;

    saans::CsiFrame at(std::size_t i, const float* zeros) const {
        const float* row = in->amps.data() + i * in->n_sub;
        if (transform) {
            for (std::size_t k = 0; k < in->n_sub; ++k) {
                // Scaled, offset, subcarrier order reversed, plus a per-frame
                // term, so link B's residuals share no structure with link A's.
                (*buf)[k] = 1.37f * row[in->n_sub - 1u - k] + 2.5f +
                            0.25f * std::sin(0.11f * static_cast<float>(i));
            }
            row = buf->data();
        }
        saans::CsiFrame f;
        f.amplitudes     = row;
        f.phases         = zeros;
        f.n_subcarriers  = in->n_sub;
        f.sample_index   = static_cast<std::uint32_t>((*in).seq[i]);
        f.sample_rate_hz = in->fs;
        f.rssi_dbm       = static_cast<std::int8_t>((*in).rssi[i]);
        f.node_id        = 3;
        return f;
    }
};

template <typename FX>
std::vector<saans::FeatureFrame> drive_recording(FX& fx, const Source& src) {
    std::vector<saans::FeatureFrame> out;
    std::vector<float>               zeros(src.in->n_sub, 0.0f);
    for (std::size_t i = 0; i < src.in->n_in; ++i) {
        const saans::CsiFrame f = src.at(i, zeros.data());
        const auto emitted = fx.push(f, f.sample_index, src.in->t[i], src.in->noise_floor);
        for (std::size_t k = 0; k < emitted.count; ++k) {
            out.push_back(emitted[k]);
        }
    }
    return out;
}

template <typename A, typename B>
std::size_t byte_diff(A& fa, const std::vector<saans::FeatureFrame>& a, B& fb,
                      const std::vector<saans::FeatureFrame>& b) {
    std::size_t diff = (a.size() == b.size()) ? 0u : 1u;
    for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) {
        std::uint8_t wa[saans::kFeatureFrameSize];
        std::uint8_t wb[saans::kFeatureFrameSize];
        fa.encode(a[i], wa);
        fb.encode(b[i], wb);
        if (std::memcmp(wa, wb, saans::kFeatureFrameSize) != 0) {
            ++diff;
        }
    }
    return diff;
}

// ---------------------------------------------------------------------------
// 3 and 4. Parity of the node shape, and of the build with no BNR
// ---------------------------------------------------------------------------

void test_node_parity(Report& rep, const Input& in, const saans::FeatureExtractorConfig& cfg) {
    rep.begin("the node's shape against the library's defaults", "features_vitals");

    Source src;
    src.in = &in;

    g_stock.reset();
    if (!g_stock.configure(cfg)) {
        rep.truth("stock extractor configures", false, "configure() refused the vector");
        return;
    }
    const std::vector<saans::FeatureFrame> stock = drive_recording(g_stock, src);

    g_node.reset();
    const bool node_ok = g_node.configure(cfg);
    rep.truth("SaansNodeExtractor configures for the recording", node_ok,
              "56 subcarriers at 20 Hz, every capacity cut to what that needs");
    if (!node_ok) {
        return;
    }
    const std::vector<saans::FeatureFrame> node = drive_recording(g_node, src);
    rep.truth("and emits byte-identical frames", byte_diff(g_stock, stock, g_node, node) == 0,
              std::to_string(node.size()) + " frames against the stock " +
                  std::to_string(stock.size()) + "; a capacity bounds what is stored and "
                  "must appear in no arithmetic");
    rep.truth("with the window never overflowing", g_node.arrival_overflows == 0,
              std::to_string(g_node.arrival_overflows) + " eviction(s)");
    rep.truth("the shared arena is one array, not one per link",
              g_node.pipeline().scratch().data() == g_node_b.pipeline().scratch().data(),
              "two links, the same float* — this is the whole point of SharedScratch");

    // 4. best_bnr compiled out.
    g_node_nobnr.reset();
    saans::FeatureExtractorConfig no_bnr = cfg;
    no_bnr.compute_bnr                   = false;
    const bool nobnr_ok                  = g_node_nobnr.configure(no_bnr);
    rep.truth("a build with no row store configures with compute_bnr off", nobnr_ok,
              "MaxBnrRows == 0 — the row store is not in the binary at all");
    if (nobnr_ok) {
        rep.truth("and reports that it is not computing it", !g_node_nobnr.computes_bnr(),
                  "computes_bnr() false means best_bnr is 0.0 = UNKNOWN for the whole run, "
                  "by configuration and not because the link is bad");
        const std::vector<saans::FeatureFrame> nb = drive_recording(g_node_nobnr, src);
        std::size_t other_diff = (nb.size() == node.size()) ? 0u : 1u;
        std::size_t bnr_zero   = 0;
        for (std::size_t i = 0; i < nb.size() && i < node.size(); ++i) {
            saans::FeatureFrame a = node[i];
            saans::FeatureFrame b = nb[i];
            if (b.best_bnr == 0.0f) {
                ++bnr_zero;
            }
            a.best_bnr = 0.0f;  // the one field that is allowed to move
            b.best_bnr = 0.0f;
            std::uint8_t wa[saans::kFeatureFrameSize];
            std::uint8_t wb[saans::kFeatureFrameSize];
            g_node.encode(a, wa);
            g_node_nobnr.encode(b, wb);
            if (std::memcmp(wa, wb, saans::kFeatureFrameSize) != 0) {
                ++other_diff;
            }
        }
        rep.truth("best_bnr is the only field that moves", other_diff == 0,
                  std::to_string(nb.size()) + " frames identical in every other byte");
        rep.truth("and it is 0.0 on every one of them", bnr_zero == nb.size(),
                  std::to_string(bnr_zero) + " of " + std::to_string(nb.size()) +
                      " frames; spec/wire.md defines 0 as UNKNOWN, which a dead link "
                      "also reads as — the gateway cannot tell them apart and must not try");
    }

    // Asked for link quality from a build that has no room for it: refuse,
    // rather than report a 0.0 indistinguishable from "not filled yet".
    g_node_nobnr.reset();
    saans::FeatureExtractorConfig wants_bnr = cfg;
    wants_bnr.compute_bnr                   = true;
    rep.truth("and it REFUSES a configuration that asks for link quality",
              !g_node_nobnr.configure(wants_bnr),
              "silently turning the flag off would hide a misconfigured node");
}

// ---------------------------------------------------------------------------
// 5. The shared arena, with two links that are actually different
// ---------------------------------------------------------------------------

void test_shared_arena(Report& rep, const Input& in, const saans::FeatureExtractorConfig& cfg) {
    rep.begin("one spectral arena, two interleaved links", "features_vitals");

    std::vector<float> xform(in.n_sub, 0.0f);

    Source a;
    a.in = &in;
    Source b;
    b.in        = &in;
    b.buf       = &xform;
    b.transform = true;

    g_node.reset();
    g_node_b.reset();
    if (!g_node.configure(cfg) || !g_node_b.configure(cfg)) {
        rep.truth("both links configure", false, "configure() refused");
        return;
    }
    const std::vector<saans::FeatureFrame> a_solo = drive_recording(g_node, a);
    const std::vector<saans::FeatureFrame> b_solo = drive_recording(g_node_b, b);

    // The transform has to produce a genuinely different link, or interleaving
    // them proves nothing. Check that before trusting the result.
    std::size_t solo_differ = 0;
    for (std::size_t i = 0; i < a_solo.size() && i < b_solo.size(); ++i) {
        if (a_solo[i].best_bnr != b_solo[i].best_bnr ||
            a_solo[i].motion_energy != b_solo[i].motion_energy) {
            ++solo_differ;
        }
    }
    rep.truth("the second link really is a different link",
              !a_solo.empty() && a_solo.size() == b_solo.size() &&
                  solo_differ == a_solo.size(),
              std::to_string(solo_differ) + " of " + std::to_string(a_solo.size()) +
                  " frames differ between the two streams run alone; feeding both links "
                  "the SAME recording would make this check unable to fail");

    // Now strictly interleaved through the one arena: push A, push B, push A...
    g_node.reset();
    g_node_b.reset();
    if (!g_node.configure(cfg) || !g_node_b.configure(cfg)) {
        rep.truth("both links reconfigure", false, "configure() refused");
        return;
    }
    std::vector<saans::FeatureFrame> a_inter;
    std::vector<saans::FeatureFrame> b_inter;
    std::vector<float>               zeros(in.n_sub, 0.0f);
    for (std::size_t i = 0; i < in.n_in; ++i) {
        const saans::CsiFrame fa = a.at(i, zeros.data());
        const auto ea = g_node.push(fa, fa.sample_index, in.t[i], in.noise_floor);
        for (std::size_t k = 0; k < ea.count; ++k) {
            a_inter.push_back(ea[k]);
        }
        const saans::CsiFrame fb = b.at(i, zeros.data());
        const auto eb = g_node_b.push(fb, fb.sample_index, in.t[i], in.noise_floor);
        for (std::size_t k = 0; k < eb.count; ++k) {
            b_inter.push_back(eb[k]);
        }
    }

    rep.truth("link A interleaved is link A alone",
              byte_diff(g_node, a_solo, g_node, a_inter) == 0,
              std::to_string(a_inter.size()) + " frames; nothing in reading() reads a "
              "scratch float it has not written in the same call");
    rep.truth("and so is link B", byte_diff(g_node_b, b_solo, g_node_b, b_inter) == 0,
              std::to_string(b_inter.size()) + " frames");
}

// ---------------------------------------------------------------------------
// The borrowed arena cannot shrink behind a configure()
// ---------------------------------------------------------------------------

using BorrowedPipeline =
    saans::BasicSensingPipeline<56, Caps::kBreathHistory, Caps::kHeartHistory,
                                Caps::kEnergySamples, saans::BorrowedScratch>;

BorrowedPipeline g_borrowed;
float            g_arena[Caps::kScratchFloats];
float            g_too_small[64];

void test_borrowed_arena(Report& rep) {
    rep.begin("a borrowed spectral arena", "(no vector)");

    rep.truth("an unbound arena is a configure() failure", !g_borrowed.configure(56, 20.0f),
              "nullptr, zero floats; never a silent overrun");

    g_borrowed.bind_scratch(g_too_small, 64);
    rep.truth("and so is one bound too short", !g_borrowed.configure(56, 20.0f),
              "64 floats where one reading() wants " + std::to_string(Caps::kScratchFloats));

    g_borrowed.bind_scratch(g_arena, Caps::kScratchFloats);
    rep.truth("a correctly sized one configures", g_borrowed.configure(56, 20.0f),
              std::to_string(Caps::kScratchFloats) + " floats");

    // The hole this closes: bind big, configure, then rebind small. reading()
    // passes kScratchFloats rather than scratch_.size() on purpose, so nothing
    // downstream would have re-checked the span.
    g_borrowed.bind_scratch(g_too_small, 64);
    rep.truth("rebinding a SHORT arena afterwards goes inert, not out of bounds",
              !g_borrowed.valid(),
              "the span is part of the configuration, so changing it costs a configure(); "
              "it used to be writable through pipeline().scratch() with nothing re-checking it");

    g_borrowed.bind_scratch(g_arena, Caps::kScratchFloats);
    rep.truth("and comes back when it is rebound and reconfigured",
              g_borrowed.configure(56, 20.0f) && g_borrowed.valid(),
              "a rejection leaves no state behind");
}

}  // namespace

int main(int argc, char** argv) {
    saans::golden::dir_from_argv(argc, argv);
    Report rep;

    // Loaded first and unconditionally: a parity test that passes because it
    // could not find its data is the one outcome this suite refuses.
    const Vectors v("features_vitals");
    const Input   in  = load_input(v);
    const auto    cfg = config_from(v, in);

    test_mirror(rep);
    test_arrival_window(rep);
    test_node_parity(rep, in, cfg);
    test_shared_arena(rep, in, cfg);
    test_borrowed_arena(rep);

    return rep.finish("node_config");
}
