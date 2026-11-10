// SPDX-License-Identifier: MIT
//
// Saans edge DSP — what one link costs THE NODE'S OWN TYPE, frame by frame.
//
// `bench_extractor.cpp` answers "how fast is the port against the reference"
// on the golden recording (32 subcarriers, the library's generic type). This
// answers the budget question for issue #43: `node::SaansNodeExtractor`, the
// type `rx_role.cpp` instantiates, at the node's 56 subcarriers and 20 Hz,
// timed per `push` so the frames that run the 2 Hz spectral estimates show up
// as the spikes they are, and counted in retired user-space instructions so
// the figure can be carried to another CPU by stated assumptions rather than
// by a clock ratio. `tools/baselines.py node` builds and runs it.
//
// THE INPUT is synthetic and generated here, deterministically: 120 s of an
// empty room (noise on a static multipath profile), then a still breather at
// 15 BPM with a 72 BPM heartbeat a tenth its size. It is a WORKLOAD, not a
// test signal: nothing here checks what the extractor estimates, only that it
// emits at its 2 Hz rate. Only the occupied part is measured, where the vitals
// estimates run.
//
// WHAT IS AND IS NOT AN S3 FIGURE. Every number this prints is a HOST number.
// The instruction count is the stabler of the two, because it does not depend
// on this machine's clock or load; it is still an x86-64 count, built without
// auto-vectorisation (`-fno-tree-vectorize`) so that it is scalar code like
// the LX7's. `tools/baselines.py` scales it to the S3 under assumptions it
// prints, and labels the result ESTIMATE. The real figure comes from a board
// (bench session, issue #45).
//
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#if defined(__linux__)
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif

#include "node_config.hpp"

namespace {

using Clock = std::chrono::steady_clock;

constexpr std::size_t kN        = 56;
constexpr unsigned    kRateHz   = 20;
constexpr float       kFs       = static_cast<float>(kRateHz);
constexpr double      kEmptyS   = 120.0;
constexpr double      kTotalS   = 420.0;  // 5 measured minutes after the empty lead-in
constexpr float       kBreathHz = 0.25f;
constexpr float       kHeartHz  = 1.2f;
constexpr float       kPi       = 3.14159265f;

using NodeFx      = saans::node::SaansNodeExtractor;
using NodeNoBnrFx = saans::node::SaansNodeExtractorNoBnr;

// Static storage, as on the node.
NodeFx      g_fx;
NodeNoBnrFx g_fx_nobnr;

/// A small deterministic generator, so every run sees the same frames.
struct Lcg {
    std::uint64_t s = 0x5AA50043u;
    float uniform() {
        s = s * 6364136223846793005ull + 1442695040888963407ull;
        return static_cast<float>((s >> 40) & 0xFFFFFFu) / 16777216.0f;
    }
    float gauss() {  // sum of four uniforms: cheap, bounded, near enough
        return (uniform() + uniform() + uniform() + uniform() - 2.0f) * 1.7320508f;
    }
};

struct Workload {
    std::size_t        frames         = 0;
    std::size_t        first_measured = 0;
    std::vector<float> amps;
    std::vector<float> phases;
};

Workload make_workload() {
    Workload w;
    w.frames         = static_cast<std::size_t>(kTotalS * kRateHz);
    w.first_measured = static_cast<std::size_t>(kEmptyS * kRateHz);
    w.amps.resize(w.frames * kN);
    w.phases.resize(w.frames * kN);
    Lcg   rng;
    float base[kN];
    float gain[kN];
    for (std::size_t k = 0; k < kN; ++k) {
        base[k] = 30.0f + 10.0f * std::sin(0.37f * static_cast<float>(k));
        gain[k] = 0.6f + 0.4f * std::sin(0.21f * static_cast<float>(k));
    }
    for (std::size_t i = 0; i < w.frames; ++i) {
        const float t        = static_cast<float>(i) / kFs;
        const bool  occupied = i >= w.first_measured;
        const float chest    = occupied ? 0.6f * std::sin(2.0f * kPi * kBreathHz * t) +
                                           0.06f * std::sin(2.0f * kPi * kHeartHz * t)
                                        : 0.0f;
        for (std::size_t k = 0; k < kN; ++k) {
            w.amps[i * kN + k]   = base[k] + gain[k] * chest + 0.3f * rng.gauss();
            w.phases[i * kN + k] = 0.1f * rng.gauss();
        }
    }
    return w;
}

// -- retired instructions, user space only ------------------------------------

struct Counter {
    int fd = -1;
    bool open() {
#if defined(__linux__)
        perf_event_attr a;
        std::memset(&a, 0, sizeof a);
        a.type           = PERF_TYPE_HARDWARE;
        a.size           = sizeof a;
        a.config         = PERF_COUNT_HW_INSTRUCTIONS;
        a.exclude_kernel = 1;
        a.exclude_hv     = 1;
        fd = static_cast<int>(syscall(SYS_perf_event_open, &a, 0, -1, -1, 0));
        if (fd >= 0) {
            ioctl(fd, PERF_EVENT_IOC_RESET, 0);
            ioctl(fd, PERF_EVENT_IOC_ENABLE, 0);
        }
#endif
        return fd >= 0;
    }
    std::uint64_t read_count() const {
        std::uint64_t v = 0;
#if defined(__linux__)
        if (fd >= 0 && ::read(fd, &v, sizeof v) != static_cast<ssize_t>(sizeof v)) {
            v = 0;
        }
#endif
        return v;
    }
};

struct Sample {
    double        us      = 0.0;
    std::uint64_t instr   = 0;
    bool          emitted = false;
    std::uint8_t  flags   = 0;  // of the frame emitted, if any
};

saans::FeatureExtractorConfig config(bool vitals, bool heart, bool bnr) {
    saans::FeatureExtractorConfig c;
    c.n_subcarriers  = kN;
    c.sample_rate_hz = kFs;
    c.node_id        = 1;
    c.tx_id          = 0xFF;
    c.frame_rate_hz  = 2.0f;
    c.vitals_enabled = vitals;
    c.heart_enabled  = heart;
    c.calibration_s  = 30.0f;
    c.compute_bnr    = bnr;
    return c;
}

template <class Fx>
std::vector<Sample> drive(Fx& fx, const saans::FeatureExtractorConfig& cfg, const Workload& w,
                          const Counter& ctr, std::uint64_t read_overhead, double* checksum) {
    if (!fx.configure(cfg)) {
        std::fprintf(stderr, "node_budget: the node type refused its own configuration\n");
        std::exit(2);
    }
    std::vector<Sample> out;
    out.reserve(w.frames - w.first_measured);
    double sum = 0.0;
    for (std::size_t i = 0; i < w.frames; ++i) {
        saans::CsiFrame f;
        f.amplitudes     = w.amps.data() + i * kN;
        f.phases         = w.phases.data() + i * kN;
        f.n_subcarriers  = kN;
        f.sample_index   = static_cast<std::uint32_t>(i);
        f.sample_rate_hz = kFs;
        f.rssi_dbm       = -55;
        f.node_id        = 1;
        const double t   = static_cast<double>(i) / static_cast<double>(kFs);

        const std::uint64_t c0 = ctr.read_count();
        const auto          t0 = Clock::now();
        const auto          e  = fx.push(f, static_cast<std::uint32_t>(i), t, -92);
        const auto          t1 = Clock::now();
        const std::uint64_t c1 = ctr.read_count();
        std::uint8_t flags = 0;
        for (std::size_t k = 0; k < e.count; ++k) {
            sum += static_cast<double>(e[k].motion_energy) +
                   static_cast<double>(e[k].breathing_bpm);
            flags = e[k].flags;
        }
        if (i >= w.first_measured) {
            Sample s;
            s.flags   = flags;
            s.us      = std::chrono::duration<double, std::micro>(t1 - t0).count();
            s.instr   = (c1 > c0 + read_overhead) ? c1 - c0 - read_overhead : 0;
            s.emitted = e.count > 0;
            out.push_back(s);
        }
    }
    *checksum += sum;
    return out;
}

double pct(std::vector<double> v, double p) {
    std::sort(v.begin(), v.end());
    const std::size_t i = static_cast<std::size_t>(p * static_cast<double>(v.size() - 1));
    return v[i];
}

/// One machine-readable line per configuration; tools/baselines.py parses it.
void report(const char* name, const std::vector<std::vector<Sample>>& runs) {
    // Time: the median over runs of each statistic. Instructions barely move
    // between runs, so the first run's are reported.
    std::vector<double> mean_us, p50_us, emit_us, max_us;
    for (const auto& r : runs) {
        std::vector<double> all, emit;
        double              total = 0.0;
        for (const Sample& s : r) {
            all.push_back(s.us);
            total += s.us;
            if (s.emitted) {
                emit.push_back(s.us);
            }
        }
        mean_us.push_back(total / static_cast<double>(all.size()));
        p50_us.push_back(pct(all, 0.5));
        emit_us.push_back(pct(emit, 0.5));
        max_us.push_back(pct(all, 1.0));
    }
    const std::vector<Sample>& r0          = runs.front();
    double                     instr_total = 0.0;
    double                     instr_emit  = 0.0;
    std::uint64_t              instr_max   = 0;
    std::size_t                n_emit      = 0;
    std::size_t                n_occ = 0, n_breath = 0, n_heart = 0;
    for (const Sample& s : r0) {
        n_occ += (s.flags & saans::kFlagOccupied) ? 1u : 0u;
        n_breath += (s.flags & saans::kFlagBreathingValid) ? 1u : 0u;
        n_heart += (s.flags & saans::kFlagHeartValid) ? 1u : 0u;
        instr_total += static_cast<double>(s.instr);
        instr_max = std::max(instr_max, s.instr);
        if (s.emitted) {
            instr_emit += static_cast<double>(s.instr);
            ++n_emit;
        }
    }
    const double n = static_cast<double>(r0.size());
    // The workload has to reach the paths it claims to time: a breather the
    // presence detector never sees would time an empty room.
    std::printf("NODE config=%s frames=%zu emitted=%zu occupied=%zu breathing=%zu heart=%zu "
                "mean_us=%.3f p50_us=%.3f emit_us=%.3f "
                "max_us=%.3f instr_mean=%.0f instr_emit=%.0f instr_max=%llu\n",
                name, r0.size(), n_emit, n_occ, n_breath, n_heart, pct(mean_us, 0.5), pct(p50_us, 0.5),
                pct(emit_us, 0.5), pct(max_us, 0.5), instr_total / n,
                n_emit ? instr_emit / static_cast<double>(n_emit) : 0.0,
                static_cast<unsigned long long>(instr_max));
}

}  // namespace

int main(int argc, char** argv) {
    std::size_t runs = 5;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--runs") == 0 && i + 1 < argc) {
            runs = static_cast<std::size_t>(std::max(1, std::atoi(argv[++i])));
        }
    }
    const Workload w = make_workload();
    Counter        ctr;
    const bool     counting = ctr.open();
    // What two back-to-back reads cost in user space, subtracted per push.
    std::uint64_t overhead = 0;
    if (counting) {
        std::uint64_t best = ~0ull;
        for (int i = 0; i < 1000; ++i) {
            const std::uint64_t a = ctr.read_count();
            const std::uint64_t b = ctr.read_count();
            best                  = std::min(best, b - a);
        }
        overhead = best;
    }

    std::printf("NODE_SHAPE subcarriers=%zu rate_hz=%u frame_rate_hz=2 measured_s=%.0f "
                "bytes_link=%zu bytes_link_nobnr=%zu arena_bytes=%zu counting=%d "
                "read_overhead=%llu\n",
                kN, kRateHz, kTotalS - kEmptyS, sizeof(NodeFx), sizeof(NodeNoBnrFx),
                NodeFx::Pipeline::kScratchFloats * sizeof(float), counting ? 1 : 0,
                static_cast<unsigned long long>(overhead));

    double checksum = 0.0;
    struct Case {
        const char* name;
        bool        vitals, heart, bnr;
    };
    const Case cases[] = {
        {"default", false, false, true},  // Kconfig defaults: vitals and heart off, BNR on
        {"vitals", true, false, true},    // a consented space
        {"vitals+heart", true, true, true},
        {"vitals+heart,no-bnr", true, true, false},
    };
    for (const Case& c : cases) {
        std::vector<std::vector<Sample>> all;
        for (std::size_t r = 0; r < runs + 1; ++r) {  // run 0 is an untimed warm-up
            auto s = c.bnr ? drive(g_fx, config(c.vitals, c.heart, true), w, ctr, overhead,
                                   &checksum)
                           : drive(g_fx_nobnr, config(c.vitals, c.heart, false), w, ctr,
                                   overhead, &checksum);
            if (r > 0) {
                all.push_back(std::move(s));
            }
        }
        report(c.name, all);
    }
    std::printf("NODE_CHECKSUM %.6f\n", checksum);
    return 0;
}
