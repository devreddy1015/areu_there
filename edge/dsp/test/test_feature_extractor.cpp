// SPDX-License-Identifier: MIT
//
// Saans edge DSP — end-to-end parity for `FeatureExtractor`.
//
// THESE ARE THE DECISIVE TESTS. `features_vitals` and `features_private` drive
// the whole chain from raw CSI frames — with 3% of sequence numbers dropped
// and one 2 s outage, so the gap filler really engages — through static
// suppression, fusion, presence, breathing, heart, BNR and link statistics, to
// 336 emitted FeatureFrames. Everything else in this suite checks one stage;
// this checks that the stages were wired together the way the reference wires
// them, which is the thing a port actually gets wrong.
//
// ---------------------------------------------------------------------------
// WHY THIS IS NOT A BYTE COMPARISON, AND WHY ATTEMPTING ONE WOULD BE WRONG
// ---------------------------------------------------------------------------
// The vectors carry a `bytes` array: 336 x 68 bytes of encoded frame. It is
// tempting to compare it directly, and `test_feature_frame.cpp` does exactly
// that for `feature_codec` — where it is the right test, because the codec is
// given field values and asked only to lay them out.
//
// Here it would be wrong. spec/wire.md §3 is explicit: "An independent
// float32 DSP cannot reproduce the reference's f64-then-round-to-f32 bits, so
// encoded `features_*` frames are compared field by field after decoding, not
// byte by byte." Python computes every stage in f64 and rounds to f32 once, at
// the wire; this port computes in float32 throughout. The low mantissa bits
// differ BY DESIGN — that is what ADR-003 chose — so a byte comparison tests
// the one property the contract explicitly does not ask for, and the only way
// to make it pass would be to stop being a float32 port.
//
// So the vector's bytes are DECODED and the fields compared, with the
// per-field semantics of spec/wire.md §3:
//
//   exact          node_id, tx_id, seq, n_subcarriers, calib_age_s,
//                  rssi_dbm, noise_floor_dbm
//   within 1       uptime_ms, csi_rate_chz, loss_permille
//   1e-4 relative  motion_energy, presence_threshold, gross_motion,
//                  breath_rms, best_bnr, breathing_conf, heart_conf
//   0.05 BPM       breathing_bpm, heart_bpm
//   flags exact    except within one frame of a transition IN THE REFERENCE,
//                  where a threshold compare can legitimately land either side
//
// The bytes are still used, twice: once to prove the loader and the decoder
// agree with the vector's own field arrays (so a mis-parsed vector cannot look
// like a passing test), and once to prove this port's encode/decode round-trips
// its own frames exactly.
//
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "feature_extractor.hpp"
#include "golden.hpp"

using saans::golden::Report;
using saans::golden::Vectors;

namespace {

// ---------------------------------------------------------------------------
// Driving one vector set
// ---------------------------------------------------------------------------

struct Input {
    std::size_t        n_sub = 0;
    std::size_t        n_in  = 0;
    float              fs    = 0.0f;
    std::vector<double> seq;
    std::vector<double> t;
    std::vector<double> rssi;
    std::vector<float>  amps;  // flattened, n_in * n_sub
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
        saans::golden::fatal(
            "%s: input arrays disagree (seq %zu, t %zu, rssi %zu, amplitudes %zu for %zu x %zu)",
            v.name().c_str(), in.seq.size(), in.t.size(), in.rssi.size(), in.amps.size(),
            in.n_in, in.n_sub);
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

/// Feed the whole recorded stream through one extractor and collect what it
/// emits. `FX` is templated so the same drive can prove that a tightly-sized
/// instantiation and the stock one agree.
template <typename FX>
std::vector<saans::FeatureFrame> drive(FX& fx, const Input& in, std::uint32_t* overflows) {
    std::vector<saans::FeatureFrame> out;
    std::vector<float>               phases(in.n_sub, 0.0f);  // `[0.0] * n_sub`

    for (std::size_t i = 0; i < in.n_in; ++i) {
        saans::CsiFrame f;
        f.amplitudes     = in.amps.data() + i * in.n_sub;
        f.phases         = phases.data();
        f.n_subcarriers  = in.n_sub;
        f.sample_index   = static_cast<std::uint32_t>(in.seq[i]);
        f.sample_rate_hz = in.fs;
        f.rssi_dbm       = static_cast<std::int8_t>(in.rssi[i]);
        f.node_id        = 3;

        const auto emitted = fx.push(f, f.sample_index, in.t[i], in.noise_floor);
        for (std::size_t k = 0; k < emitted.count; ++k) {
            out.push_back(emitted[k]);
        }
    }
    *overflows = fx.arrival_overflows;
    return out;
}

// ---------------------------------------------------------------------------
// Field-wise comparison
// ---------------------------------------------------------------------------

using IntGetter   = long long (*)(const saans::FeatureFrame&);
using FloatGetter = float (*)(const saans::FeatureFrame&);

/// Exact integer field across every frame. Reports the first offender, which
/// is the one worth looking at: these fields do not drift, they are either
/// right or the wiring is wrong.
void check_exact(Report& rep, const std::string& what,
                 const std::vector<saans::FeatureFrame>& got, const std::vector<double>& exp,
                 IntGetter get) {
    std::size_t bad   = 0;
    std::size_t first = 0;
    long long   g0 = 0, e0 = 0;
    for (std::size_t i = 0; i < got.size() && i < exp.size(); ++i) {
        const long long g = get(got[i]);
        const long long e = static_cast<long long>(exp[i]);
        if (g != e) {
            if (bad == 0) {
                first = i;
                g0    = g;
                e0    = e;
            }
            ++bad;
        }
    }
    if (bad == 0) {
        rep.truth(what, true, "exact over " + std::to_string(got.size()) + " frames");
    } else {
        char detail[256];
        std::snprintf(detail, sizeof(detail),
                      "%zu of %zu frames differ; first at %zu: got %lld expect %lld", bad,
                      got.size(), first, g0, e0);
        rep.truth(what, false, detail);
    }
}

/// Integer field allowed to differ by at most `tol` (spec/wire.md §3: the
/// three timing-derived fields). Asserts the worst deviation, not the mean.
void check_within(Report& rep, const std::string& what,
                  const std::vector<saans::FeatureFrame>& got, const std::vector<double>& exp,
                  IntGetter get, long long tol) {
    long long   worst = 0;
    std::size_t at    = 0;
    long long   g0 = 0, e0 = 0;
    for (std::size_t i = 0; i < got.size() && i < exp.size(); ++i) {
        const long long g = get(got[i]);
        const long long e = static_cast<long long>(exp[i]);
        const long long d = (g > e) ? (g - e) : (e - g);
        if (d > worst) {
            worst = d;
            at    = i;
            g0    = g;
            e0    = e;
        }
    }
    char detail[256];
    std::snprintf(detail, sizeof(detail),
                  "worst |delta| %lld at frame %zu (got %lld expect %lld), tol %lld", worst, at,
                  g0, e0, tol);
    rep.truth(what, worst <= tol, detail);
}

/// Continuous float field, with golden.hpp's two-sided series metric:
/// worst error as a fraction of the series' own full scale, AND worst
/// per-sample relative among samples at or above half full scale. See the long
/// note at the top of golden.hpp for why a bare per-sample relative is not a
/// meaningful contract for a sequence that passes through zero.
void check_series(Report& rep, const std::string& what,
                  const std::vector<saans::FeatureFrame>& got, const std::vector<double>& exp,
                  FloatGetter get, double tol) {
    std::vector<float> g(got.size());
    for (std::size_t i = 0; i < got.size(); ++i) {
        g[i] = get(got[i]);
    }
    const std::size_t n = (g.size() < exp.size()) ? g.size() : exp.size();
    rep.close_array(what, g.data(), exp.data(), n, tol);
}

/// A rate in BPM: absolute tolerance, because 0.05 BPM is what the contract
/// says and a relative figure on a value that is legitimately 0.0 (the
/// "unavailable" encoding) means nothing.
void check_bpm(Report& rep, const std::string& what,
               const std::vector<saans::FeatureFrame>& got, const std::vector<double>& exp,
               FloatGetter get, double tol) {
    double      worst = 0.0;
    std::size_t at    = 0;
    double      g0 = 0.0, e0 = 0.0;
    for (std::size_t i = 0; i < got.size() && i < exp.size(); ++i) {
        const double g = static_cast<double>(get(got[i]));
        const double d = std::fabs(g - exp[i]);
        if (d > worst) {
            worst = d;
            at    = i;
            g0    = g;
            e0    = exp[i];
        }
    }
    char detail[256];
    std::snprintf(detail, sizeof(detail),
                  "worst %.4f BPM at frame %zu (got %.4f expect %.4f), tol %.2f BPM", worst, at,
                  g0, e0, tol);
    rep.truth(what, worst <= tol, detail);
}

/// Flags: exact, EXCEPT within one frame of a transition in the reference.
///
/// A flag is a threshold compare on a float32 quantity that itself carries
/// ~1e-6 of its own uncertainty, so on the frame where the reference's value
/// crosses, either side is defensible. The exemption is deliberately narrow:
/// it applies per BIT, and only when THE REFERENCE's own value of that bit
/// changes between this frame and a neighbour. A bit this port flips where the
/// reference never flips is a failure, which is the case that matters.
void check_flags(Report& rep, const std::vector<saans::FeatureFrame>& got,
                 const std::vector<double>& exp) {
    const std::size_t n = (got.size() < exp.size()) ? got.size() : exp.size();
    std::size_t       hard = 0;
    std::size_t       excused = 0;
    std::size_t       first_hard = 0;
    unsigned          g0 = 0, e0 = 0;

    for (std::size_t i = 0; i < n; ++i) {
        const unsigned g    = static_cast<unsigned>(got[i].flags);
        const unsigned e    = static_cast<unsigned>(static_cast<long long>(exp[i]));
        unsigned       diff = g ^ e;
        if (diff == 0u) {
            continue;
        }
        for (unsigned b = 0; b < 8u; ++b) {
            const unsigned mask = 1u << b;
            if ((diff & mask) == 0u) {
                continue;
            }
            const unsigned here = e & mask;
            const bool prev_moves =
                (i > 0) && ((static_cast<unsigned>(static_cast<long long>(exp[i - 1])) & mask) !=
                            here);
            const bool next_moves =
                (i + 1 < n) &&
                ((static_cast<unsigned>(static_cast<long long>(exp[i + 1])) & mask) != here);
            if (prev_moves || next_moves) {
                ++excused;
            } else {
                if (hard == 0) {
                    first_hard = i;
                    g0         = g;
                    e0         = e;
                }
                ++hard;
            }
        }
    }

    char detail[256];
    if (hard == 0) {
        std::snprintf(detail, sizeof(detail),
                      "%zu frames; %zu bit(s) differed and every one sat on a transition in the "
                      "reference",
                      n, excused);
    } else {
        std::snprintf(detail, sizeof(detail),
                      "%zu bit(s) differ away from any transition (%zu excused); first at frame "
                      "%zu: got 0x%02X expect 0x%02X",
                      hard, excused, first_hard, g0, e0);
    }
    rep.truth("flags", hard == 0, detail);
}

// ---------------------------------------------------------------------------
// The accessors, one per wire field
// ---------------------------------------------------------------------------

#define INT_FIELD(name) [](const saans::FeatureFrame& f) -> long long { return static_cast<long long>(f.name); }
#define FLT_FIELD(name) [](const saans::FeatureFrame& f) -> float { return f.name; }

void check_all_fields(Report& rep, const Vectors& v,
                      const std::vector<saans::FeatureFrame>& got) {
    // EXACT.
    check_exact(rep, "node_id", got, v.array("node_id"), INT_FIELD(node_id));
    check_exact(rep, "tx_id", got, v.array("tx_id"), INT_FIELD(tx_id));
    check_exact(rep, "seq", got, v.array("seq"), INT_FIELD(seq));
    check_exact(rep, "n_subcarriers", got, v.array("n_subcarriers"), INT_FIELD(n_subcarriers));
    check_exact(rep, "calib_age_s", got, v.array("calib_age_s"), INT_FIELD(calib_age_s));
    check_exact(rep, "rssi_dbm", got, v.array("rssi_dbm"), INT_FIELD(rssi_dbm));
    check_exact(rep, "noise_floor_dbm", got, v.array("noise_floor_dbm"),
                INT_FIELD(noise_floor_dbm));

    // WITHIN 1 — the timing-derived fields. See spec/wire.md §3.
    check_within(rep, "uptime_ms (within 1)", got, v.array("uptime_ms"), INT_FIELD(uptime_ms), 1);
    check_within(rep, "csi_rate_chz (within 1)", got, v.array("csi_rate_chz"),
                 INT_FIELD(csi_rate_chz), 1);
    check_within(rep, "loss_permille (within 1)", got, v.array("loss_permille"),
                 INT_FIELD(loss_permille), 1);

    // 1e-4 RELATIVE.
    check_series(rep, "motion_energy", got, v.array("motion_energy"), FLT_FIELD(motion_energy),
                 1e-4);
    check_series(rep, "presence_threshold", got, v.array("presence_threshold"),
                 FLT_FIELD(presence_threshold), 1e-4);
    check_series(rep, "gross_motion", got, v.array("gross_motion"), FLT_FIELD(gross_motion),
                 1e-4);
    check_series(rep, "breath_rms", got, v.array("breath_rms"), FLT_FIELD(breath_rms), 1e-4);
    check_series(rep, "breathing_conf", got, v.array("breathing_conf"),
                 FLT_FIELD(breathing_conf), 1e-4);
    check_series(rep, "heart_conf", got, v.array("heart_conf"), FLT_FIELD(heart_conf), 1e-4);
    check_series(rep, "best_bnr", got, v.array("best_bnr"), FLT_FIELD(best_bnr), 1e-4);

    // 0.05 BPM.
    check_bpm(rep, "breathing_bpm (0.05 BPM)", got, v.array("breathing_bpm"),
              FLT_FIELD(breathing_bpm), 0.05);
    check_bpm(rep, "heart_bpm (0.05 BPM)", got, v.array("heart_bpm"), FLT_FIELD(heart_bpm), 0.05);

    check_flags(rep, got, v.array("flags"));
}

// ---------------------------------------------------------------------------
// The vector's own bytes
// ---------------------------------------------------------------------------

/// The vector carries both decoded fields and encoded bytes. If the two
/// disagree, the vector set is half-regenerated and every check above is
/// comparing against something that never existed. Checking it here costs
/// nothing and removes a whole class of silent green.
void check_vector_self_consistency(Report& rep, const Vectors& v, std::size_t n_frames) {
    const std::vector<double>& bytes = v.array("bytes");
    if (bytes.size() != n_frames * saans::kFeatureFrameSize) {
        rep.truth("vector bytes length", false,
                  std::to_string(bytes.size()) + " bytes for " + std::to_string(n_frames) +
                      " frames");
        return;
    }
    const std::vector<double>& seq   = v.array("seq");
    const std::vector<double>& flags = v.array("flags");
    const std::vector<double>& bpm   = v.array("breathing_bpm");

    std::size_t bad = 0;
    for (std::size_t i = 0; i < n_frames; ++i) {
        std::uint8_t buf[saans::kFeatureFrameSize];
        for (std::size_t b = 0; b < saans::kFeatureFrameSize; ++b) {
            buf[b] = static_cast<std::uint8_t>(bytes[i * saans::kFeatureFrameSize + b]);
        }
        saans::FeatureFrame      f;
        const saans::DecodeError e = saans::decode_feature_frame(buf, f);
        if (e != saans::DecodeError::kOk) {
            ++bad;
            continue;
        }
        if (static_cast<double>(f.seq) != seq[i] ||
            static_cast<double>(f.flags) != flags[i] ||
            static_cast<double>(f.breathing_bpm) != bpm[i]) {
            ++bad;
        }
    }
    rep.truth("vector bytes decode to the vector's own fields", bad == 0,
              std::to_string(n_frames - bad) + " of " + std::to_string(n_frames) +
                  " frames agree (CRC, magic, version, seq, flags, breathing_bpm)");
}

/// This port's own encode -> decode must be lossless. Byte equality against
/// the reference is NOT claimed here and must not be; see the file header.
void check_round_trip(Report& rep, const saans::FeatureExtractor& fx,
                      const std::vector<saans::FeatureFrame>& got) {
    std::size_t bad = 0;
    for (const saans::FeatureFrame& f : got) {
        std::uint8_t buf[saans::kFeatureFrameSize];
        fx.encode(f, buf);
        saans::FeatureFrame back;
        if (saans::decode_feature_frame(buf, back) != saans::DecodeError::kOk) {
            ++bad;
            continue;
        }
        const bool same = back.node_id == f.node_id && back.tx_id == f.tx_id &&
                          back.flags == f.flags && back.seq == f.seq &&
                          back.uptime_ms == f.uptime_ms && back.calib_age_s == f.calib_age_s &&
                          back.rssi_dbm == f.rssi_dbm &&
                          back.noise_floor_dbm == f.noise_floor_dbm &&
                          back.n_subcarriers == f.n_subcarriers &&
                          back.motion_energy == f.motion_energy &&
                          back.presence_threshold == f.presence_threshold &&
                          back.gross_motion == f.gross_motion &&
                          back.breath_rms == f.breath_rms &&
                          back.breathing_bpm == f.breathing_bpm &&
                          back.breathing_conf == f.breathing_conf &&
                          back.heart_bpm == f.heart_bpm && back.heart_conf == f.heart_conf &&
                          back.best_bnr == f.best_bnr;
        if (!same) {
            ++bad;
        }
    }
    rep.truth("encode -> decode round-trips every emitted frame", bad == 0,
              std::to_string(got.size() - bad) + " of " + std::to_string(got.size()) +
                  " frames identical (CRC checked on the way back)");
}

// ---------------------------------------------------------------------------
// The privacy gate
// ---------------------------------------------------------------------------

/// `features_private` is the same recording with `vitals_enabled = False`.
/// What it proves is not a tolerance but a boundary: ADR-001/ADR-005 and
/// spec/wire.md's "Gating rule" require every breathing- and heart-derived
/// field to be written as 0.0 ON THE NODE, so a gateway never receives what it
/// is not allowed to hold. EXACTLY 0.0 — not "small", not "within tolerance".
void check_privacy_gate(Report& rep, const std::vector<saans::FeatureFrame>& got) {
    std::size_t leaked      = 0;
    std::size_t flag_leaked = 0;
    std::size_t first       = 0;
    for (std::size_t i = 0; i < got.size(); ++i) {
        const saans::FeatureFrame& f = got[i];
        const bool zero = f.breath_rms == 0.0f && f.breathing_bpm == 0.0f &&
                          f.breathing_conf == 0.0f && f.heart_bpm == 0.0f &&
                          f.heart_conf == 0.0f;
        if (!zero) {
            if (leaked == 0) {
                first = i;
            }
            ++leaked;
        }
        if (f.flag(saans::kFlagVitalsEnabled) || f.flag(saans::kFlagBreathingValid) ||
            f.flag(saans::kFlagHeartValid)) {
            ++flag_leaked;
        }
    }
    char detail[256];
    std::snprintf(detail, sizeof(detail),
                  "breath_rms / breathing_bpm / breathing_conf / heart_bpm / heart_conf are "
                  "exactly 0.0 in %zu of %zu frames%s",
                  got.size() - leaked, got.size(),
                  leaked ? (" — first leak at frame " + std::to_string(first)).c_str() : "");
    rep.truth("PRIVACY: vitals fields are exactly 0.0", leaked == 0, detail);
    rep.truth("PRIVACY: no vitals/breathing/heart flag is set", flag_leaked == 0,
              std::to_string(flag_leaked) + " frame(s) carried a gated flag bit");
}

/// The gate is not a property of this one recording. A frame built by hand
/// with every vitals field set, pushed through `apply_consent_gate` with
/// consent withheld, must come back zeroed — and must be unchanged when it is
/// applied twice, which is the invariant that makes the gate safe to call from
/// anywhere.
void check_gate_is_idempotent(Report& rep) {
    saans::FeatureFrame f;
    f.breath_rms     = 0.25f;
    f.breathing_bpm  = 15.5f;
    f.breathing_conf = 0.9f;
    f.heart_bpm      = 70.0f;
    f.heart_conf     = 0.8f;
    f.motion_energy  = 0.125f;
    f.gross_motion   = 0.0625f;
    f.best_bnr       = 7.0f;
    f.flags          = static_cast<std::uint8_t>(saans::kFlagVitalsEnabled |
                                        saans::kFlagBreathingValid |
                                        saans::kFlagHeartValid | saans::kFlagOccupied);

    saans::VitalsConsent none;
    saans::apply_consent_gate(f, none);
    const bool cleared = f.breath_rms == 0.0f && f.breathing_bpm == 0.0f &&
                         f.breathing_conf == 0.0f && f.heart_bpm == 0.0f &&
                         f.heart_conf == 0.0f && !f.flag(saans::kFlagVitalsEnabled) &&
                         !f.flag(saans::kFlagBreathingValid) &&
                         !f.flag(saans::kFlagHeartValid);
    rep.truth("gate clears every vitals field and flag", cleared, "consent withheld");

    const bool kept = f.motion_energy == 0.125f && f.gross_motion == 0.0625f &&
                      f.best_bnr == 7.0f && f.flag(saans::kFlagOccupied);
    rep.truth("gate keeps presence and link quality", kept,
              "motion_energy, gross_motion, best_bnr and the occupied flag are not health data");

    const saans::FeatureFrame before = f;
    saans::apply_consent_gate(f, none);
    const bool idempotent = f.breath_rms == before.breath_rms && f.flags == before.flags &&
                            f.motion_energy == before.motion_energy;
    rep.truth("gate is idempotent", idempotent, "applying it twice changes nothing");
}

// ---------------------------------------------------------------------------
// One vector set, start to finish
// ---------------------------------------------------------------------------

// 73 KB at the stock template parameters — too much for a default stack frame,
// so these live in static storage. On the node they are a member of the link
// object, which is where gapfill.hpp's Arena note says to put them.
saans::FeatureExtractor g_fx;
saans::BasicFeatureExtractor<32, 128, 256, 64, 32> g_fx_tight;

void run_set(Report& rep, const char* name, bool expect_private) {
    const Vectors v(name);
    const Input   in  = load_input(v);
    const auto    cfg = config_from(v, in);

    rep.begin(std::string("FeatureExtractor end to end: ") + name, name);
    rep.note("%zu CSI frames in (%zu subcarriers at %g Hz, 3%% sequence loss plus one 2 s "
             "outage), vitals %s, heart %s",
             in.n_in, in.n_sub, static_cast<double>(in.fs),
             cfg.vitals_enabled ? "ENABLED" : "withheld", cfg.heart_enabled ? "enabled" : "off");

    g_fx.reset();
    if (!g_fx.configure(cfg)) {
        rep.truth("extractor configures", false, "configure() refused the vector's parameters");
        return;
    }
    rep.truth("extractor configures", true,
              "emit every " + std::to_string(g_fx.frames_per_feature()) +
                  " processed frames (round(fs / frame_rate_hz))");

    std::uint32_t overflows = 0;
    const std::vector<saans::FeatureFrame> got = drive(g_fx, in, &overflows);

    rep.equal("frames emitted", static_cast<long long>(got.size()), v.integer("frames_out"));
    if (got.size() != static_cast<std::size_t>(v.integer("frames_out"))) {
        // Every field check below indexes both series; a count mismatch makes
        // them meaningless, so stop rather than print 20 misleading lines.
        rep.truth("field comparison", false,
                  "skipped: the frame counts disagree, so nothing lines up");
        return;
    }
    rep.truth("10 s link-stats window never overflowed", overflows == 0,
              std::to_string(overflows) +
                  " arrival(s) dropped for capacity (0 means the window is the reference's)");

    rep.note("gap filler: %u frames interpolated, %u discontinuities, %u duplicates",
             g_fx.gaps().filled, g_fx.gaps().discontinuities, g_fx.gaps().duplicates);
    rep.truth("the gap filler actually engaged", g_fx.gaps().filled > 0,
              "a run with no filling would not exercise the path this vector exists for");
    rep.truth("the long outage was not papered over", g_fx.gaps().discontinuities > 0,
              "the 2 s gap exceeds max_gap and must be reported, not invented");

    check_vector_self_consistency(rep, v, got.size());
    check_all_fields(rep, v, got);
    check_round_trip(rep, g_fx, got);

    if (expect_private) {
        check_privacy_gate(rep, got);
    }

    // The template parameters are a RAM budget, not a behaviour. A tightly
    // sized instantiation must produce bit-identical frames, or one of the
    // fixed capacities is silently truncating something.
    g_fx_tight.reset();
    if (!g_fx_tight.configure(cfg)) {
        rep.truth("tight instantiation configures", false, "32 subcarriers, 256 arrivals");
    } else {
        std::uint32_t tight_overflows = 0;
        const auto    tight = drive(g_fx_tight, in, &tight_overflows);
        std::size_t   diff  = 0;
        for (std::size_t i = 0; i < tight.size() && i < got.size(); ++i) {
            std::uint8_t a[saans::kFeatureFrameSize];
            std::uint8_t b[saans::kFeatureFrameSize];
            g_fx.encode(got[i], a);
            g_fx_tight.encode(tight[i], b);
            for (std::size_t k = 0; k < saans::kFeatureFrameSize; ++k) {
                if (a[k] != b[k]) {
                    ++diff;
                    break;
                }
            }
        }
        rep.truth("a tightly-sized instantiation is byte-identical",
                  tight.size() == got.size() && diff == 0,
                  std::to_string(tight.size()) + " frames, " + std::to_string(diff) +
                      " differing");
    }
}

}  // namespace

int main(int argc, char** argv) {
    saans::golden::dir_from_argv(argc, argv);
    Report rep;

    run_set(rep, "features_vitals", false);
    run_set(rep, "features_private", true);

    rep.begin("the consent gate on its own", "(no vector)");
    check_gate_is_idempotent(rep);

    return rep.finish("feature_extractor");
}
