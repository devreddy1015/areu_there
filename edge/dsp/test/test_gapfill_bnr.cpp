// SPDX-License-Identifier: MIT
//
// Saans edge DSP — gap filler, FFT and BNR parity tests.
//
// Covers gapfill.hpp, fft.hpp and bnr.hpp against the golden vectors in
// spec/golden/. Every numeric claim in here is checked against the Python
// reference; nothing is self-generated.
//
// Vectors used:
//   gapfill   — 14 sequence numbers through SequenceGapFiller(max_gap=5),
//               exercising a short gap, a duplicate, an over-long gap, a
//               backwards jump and a wrap across 2^32
//   fft64     — a 64-point complex transform of gaussian noise
//   bnr       — subcarrier_bnr over a 120 x 8 window at 4 Hz, 0.1-0.5 Hz
//
// Tolerances are the ADR-003 / SAANS_PLAN §4 contract for the float32 edge
// build: 1e-4 relative, and exactness where the reference's answer is an
// integer. They are written once, here, and never relaxed per test.
//
// A MISSING VECTOR FAILS. golden.hpp has no skip path; see its header.
//
#include <cmath>
#include <cstdio>
#include <string>
#include <vector>

#include "bnr.hpp"
#include "core.hpp"
#include "fft.hpp"
#include "gapfill.hpp"

#include "golden.hpp"

using saans::golden::Report;
using saans::golden::Vectors;

namespace {

// ---- the contract ----------------------------------------------------------
constexpr double kRelTol = 1e-4;  // continuous floats

// ---- scratch ---------------------------------------------------------------
// Static, fixed-capacity, file-scope: the same shape the firmware uses. The
// library never allocates, so the test must not either when driving it.
constexpr std::size_t kMaxRows = 256;  // >= the 120-row bnr window, pow2 pad 128
constexpr std::size_t kMaxCols = 16;
constexpr std::size_t kFftMax  = 64;

saans::BnrWorkspace<kMaxRows> g_bnr_ws;
float                        g_window[kMaxRows * kMaxCols];
float                        g_bnr[kMaxCols];
float                        g_fft_re[kFftMax];
float                        g_fft_im[kFftMax];

// ---------------------------------------------------------------------------
// gapfill
// ---------------------------------------------------------------------------
//
// The vector replays the generator's exact call sequence:
//
//   seqs = [0,1,2,5,6,6,7,40,41,0xFFFFFFFE,0xFFFFFFFF,1,2,3]
//   amplitudes = [a, 2a] with a = 1.5*i + 1.0, phases = [0,0], 20 Hz
//
// and records, for every frame the filler emitted, its first amplitude, its
// sample_index and WHICH INPUT produced it (`out_from_input`). That last array
// is what makes this a real test rather than a shape check: it pins the
// interleaving of fills and pass-throughs, so a port that emitted the right
// frames in the wrong order, or attributed a fill to the wrong gap, fails.

constexpr std::size_t kGapSub  = 2;
constexpr std::size_t kGapFill = 8;  // >= max_gap 5

using Filler = saans::SequenceGapFiller<kGapSub, kGapFill>;
Filler::Arena g_gap_arena;

void test_gapfill(Report& r) {
    const Vectors v("gapfill");
    r.begin("SequenceGapFiller", v.name());

    const std::vector<double> seq    = v.array("seq");
    const std::vector<double> in_amp = v.array("in_amp");
    const std::vector<double> ex_amp = v.array("out_amp");
    const std::vector<double> ex_seq = v.array("out_seq");
    const std::vector<double> ex_src = v.array("out_from_input");

    const std::size_t n_in  = seq.size();
    const std::size_t n_out = ex_amp.size();
    r.truth("vector shapes agree",
            in_amp.size() == n_in && ex_seq.size() == n_out && ex_src.size() == n_out,
            "in " + std::to_string(n_in) + " frames, out " + std::to_string(n_out));

    const auto max_gap = static_cast<std::uint32_t>(v.integer("max_gap"));
    Filler     g(max_gap);
    r.equal("max_gap survives the MaxFill clamp", g.max_gap(), max_gap);

    // Emitted frames, flattened the way the generator flattened them.
    std::vector<float>         got_amp;
    std::vector<std::uint32_t> got_seq;
    std::vector<std::size_t>   got_src;

    for (std::size_t i = 0; i < n_in; ++i) {
        const float    a        = static_cast<float>(in_amp[i]);
        const float    amps[2]  = {a, 2.0f * a};
        const float    phs[2]   = {0.0f, 0.0f};
        saans::CsiFrame f;
        f.amplitudes     = amps;
        f.phases         = phs;
        f.n_subcarriers  = kGapSub;
        f.sample_index   = static_cast<std::uint32_t>(seq[i]);
        f.sample_rate_hz = 20.0f;

        const Filler::Output out = g.push(f.sample_index, f, g_gap_arena);
        for (std::size_t j = 0; j < out.count; ++j) {
            got_amp.push_back(out[j].amplitudes[0]);
            got_seq.push_back(out[j].sample_index);
            got_src.push_back(i);
            // The second subcarrier is 2x the first in every input, so the
            // interpolation must preserve that exactly — it is linear, and
            // 2*(x + (y-x)w) == 2x + (2y-2x)w holds bit-for-bit in float32
            // because every operand doubles exactly. A port that interpolated
            // one subcarrier and copied the rest would be caught here.
            const bool pair_ok =
                out[j].amplitudes[1] == 2.0f * out[j].amplitudes[0] &&
                out[j].n_subcarriers == kGapSub && out[j].phases[0] == 0.0f &&
                out[j].phases[1] == 0.0f && out[j].sample_rate_hz == 20.0f;
            if (!pair_ok) {
                r.truth("frame " + std::to_string(got_amp.size() - 1) + " is self-consistent",
                        false, "amplitude pair / phase / rate mismatch");
            }
        }
    }

    r.equal("frames emitted", static_cast<long long>(got_amp.size()),
            static_cast<long long>(n_out));
    if (got_amp.size() != n_out) {
        return;  // the per-frame checks below would compare different streams
    }

    // Sample indices are integers mod 2^32 — exact, never a tolerance.
    long long seq_bad = 0;
    long long src_bad = 0;
    std::size_t first_seq_bad = 0;
    for (std::size_t i = 0; i < n_out; ++i) {
        if (static_cast<double>(got_seq[i]) != ex_seq[i]) {
            if (seq_bad == 0) {
                first_seq_bad = i;
            }
            ++seq_bad;
        }
        if (static_cast<double>(got_src[i]) != ex_src[i]) {
            ++src_bad;
        }
    }
    r.equal("out_seq exact (mod 2^32)", seq_bad, 0);
    if (seq_bad != 0) {
        r.note("first mismatch at %zu: got %u, expect %.0f", first_seq_bad,
               got_seq[first_seq_bad], ex_seq[first_seq_bad]);
    }
    r.equal("out_from_input exact (ordering)", src_bad, 0);

    r.close_array("out_amp", got_amp.data(), ex_amp.data(), n_out, kRelTol);

    r.equal("filled", g.filled, static_cast<std::uint32_t>(v.integer("filled")));
    r.equal("discontinuities", g.discontinuities,
            static_cast<std::uint32_t>(v.integer("discontinuities")));
    r.equal("duplicates", g.duplicates,
            static_cast<std::uint32_t>(v.integer("duplicates")));

    // The wrap is the reason this vector contains 0xFFFFFFFE: input 11 is
    // sequence 1 arriving after 0xFFFFFFFF, a step of 2, so the filler must
    // synthesise sequence 0 rather than see a 4-billion-frame backwards jump.
    bool wrapped = false;
    for (std::size_t i = 0; i + 1 < n_out; ++i) {
        if (got_seq[i] == 0xFFFFFFFFu && got_seq[i + 1] == 0u) {
            wrapped = true;
        }
    }
    r.truth("fills across the 2^32 wrap", wrapped,
            "0xFFFFFFFF is followed by a synthesised 0");
}

// Behaviour the vector cannot reach, pinned against the reference's text
// rather than against generated data: the guard clauses and the counters.
void test_gapfill_guards(Report& r) {
    r.begin("SequenceGapFiller guards", "(reference semantics, no vector)");

    const float amps[2] = {1.0f, 2.0f};
    const float phs[2]  = {0.25f, -0.25f};
    saans::CsiFrame f;
    f.amplitudes     = amps;
    f.phases         = phs;
    f.n_subcarriers  = kGapSub;
    f.sample_rate_hz = 20.0f;

    {   // A duplicate is dropped and does NOT advance the history, so the
        // frame after it is measured from the original, not the duplicate.
        Filler g(5);
        f.sample_index = 10;
        g.push(10, f, g_gap_arena);
        const Filler::Output dup = g.push(10, f, g_gap_arena);
        r.equal("duplicate emits nothing", static_cast<long long>(dup.count), 0);
        r.equal("duplicate counted", g.duplicates, 1u);
        r.equal("duplicate leaves last_sequence alone", g.last_sequence(), 10u);
        const Filler::Output nxt = g.push(12, f, g_gap_arena);
        r.equal("gap after a duplicate spans from the original",
                static_cast<long long>(nxt.count), 2);
        r.equal("  and counts one fill", g.filled, 1u);
    }

    {   // Phase is HELD, not interpolated — the one thing a reader is most
        // likely to "fix" into a bug.
        Filler g(5);
        const float a0[2] = {0.0f, 0.0f};
        const float p0[2] = {3.0f, -3.0f};
        saans::CsiFrame first;
        first.amplitudes     = a0;
        first.phases         = p0;
        first.n_subcarriers  = kGapSub;
        first.sample_index   = 0;
        first.sample_rate_hz = 20.0f;
        g.push(0, first, g_gap_arena);

        const float a1[2] = {4.0f, 8.0f};
        const float p1[2] = {-3.0f, 3.0f};
        saans::CsiFrame second;
        second.amplitudes     = a1;
        second.phases         = p1;
        second.n_subcarriers  = kGapSub;
        second.sample_index   = 4;
        second.sample_rate_hz = 20.0f;
        const Filler::Output o = g.push(4, second, g_gap_arena);

        r.equal("3 fills + the frame", static_cast<long long>(o.count), 4);
        bool phase_held = true;
        for (std::size_t i = 0; i + 1 < o.count; ++i) {
            phase_held = phase_held && o[i].phases[0] == 3.0f && o[i].phases[1] == -3.0f;
        }
        r.truth("phase held at the previous frame's value", phase_held,
                "wrapped phase does not interpolate: halfway between +3 and -3 "
                "rad is +-pi, the furthest point on the circle from the truth");
        // Amplitudes 0 -> 4 over a step of 4: exact quarters in float32.
        const bool amp_ok = o[0].amplitudes[0] == 1.0f && o[1].amplitudes[0] == 2.0f &&
                            o[2].amplitudes[0] == 3.0f && o[3].amplitudes[0] == 4.0f;
        r.truth("amplitudes interpolate linearly", amp_ok, "0, 4 -> 1, 2, 3, 4");
    }

    {   // A gap longer than max_gap is counted, never bridged.
        Filler g(5);
        f.sample_index = 0;
        g.push(0, f, g_gap_arena);
        const Filler::Output o = g.push(100, f, g_gap_arena);
        r.equal("over-long gap passes the frame alone", static_cast<long long>(o.count), 1);
        r.equal("  and is counted", g.discontinuities, 1u);
        r.equal("  with nothing filled", g.filled, 0u);
    }

    // -----------------------------------------------------------------------
    // THE max_gap BOUNDARY, FROM BOTH SIDES
    // -----------------------------------------------------------------------
    // `missing > max_gap` is the only tuning decision this module makes, and
    // neither the vector nor the cases above sit on it: the vector's longest
    // bridged gap is missing = 2 against max_gap = 5 and its long case jumps
    // to missing = 32, while the block above uses missing = 99. Nothing
    // distinguished `>` from `>=`, so a port that silently refused to bridge
    // the longest gap it was configured for passed parity. These two cases
    // pin it. Both numbers come from the reference, run on the same inputs.
    {   // missing == max_gap: the longest gap that MUST be bridged.
        Filler      g(5);
        const float a_lo[2] = {1.0f, 2.0f};
        const float a_hi[2] = {7.0f, 14.0f};
        const float p_z[2]  = {0.0f, 0.0f};
        saans::CsiFrame lo;
        lo.amplitudes = a_lo; lo.phases = p_z; lo.n_subcarriers = kGapSub;
        lo.sample_index = 0;  lo.sample_rate_hz = 20.0f;
        saans::CsiFrame hi = lo;
        hi.amplitudes = a_hi; hi.sample_index = 6;

        g.push(0, lo, g_gap_arena);
        const Filler::Output o = g.push(6, hi, g_gap_arena);
        // Reference: SequenceGapFiller(5).push(0, [1,2]).push(6, [7,14]) emits
        // 6 frames, amplitudes 2,3,4,5,6,7, filled = 5, discontinuities = 0.
        r.equal("missing == max_gap IS bridged", static_cast<long long>(o.count), 6);
        r.equal("  and fills exactly max_gap frames", g.filled, 5u);
        r.equal("  with no discontinuity", g.discontinuities, 0u);
        bool ramp = true;
        for (std::size_t i = 0; i < o.count; ++i) {
            ramp = ramp && o[i].amplitudes[0] == static_cast<float>(i + 2);
            ramp = ramp && o[i].sample_index == static_cast<std::uint32_t>(i + 1);
        }
        r.truth("  amplitudes 2..7 on sequences 1..6", ramp,
                "1 -> 7 over a step of 6: exact sixths are not exact in binary, "
                "but the endpoints and the integers between them are");
    }

    {   // missing == max_gap + 1: the shortest gap that must NOT be bridged.
        Filler g(5);
        f.sample_index = 0;
        g.push(0, f, g_gap_arena);
        const Filler::Output o = g.push(7, f, g_gap_arena);
        r.equal("missing == max_gap + 1 is NOT bridged", static_cast<long long>(o.count), 1);
        r.equal("  nothing filled", g.filled, 0u);
        r.equal("  counted as a discontinuity", g.discontinuities, 1u);
    }

    {   // max_gap == MaxFill: the longest gap the arena can hold, which is the
        // one case that writes arena.amplitudes[MaxFill - 1] and
        // arena.frames[MaxFill]. Nothing else in the suite reaches those rows,
        // so an off-by-one in the arena indexing would not show up anywhere
        // else. Run this file under ASan/UBSan and this is the case that makes
        // the bound real rather than merely counted.
        saans::SequenceGapFiller<kGapSub, kGapFill> g(kGapFill);
        r.equal("max_gap == MaxFill survives the clamp",
                static_cast<long long>(g.max_gap()), static_cast<long long>(kGapFill));
        const float a_lo[2] = {1.0f, 2.0f};
        const float a_hi[2] = {10.0f, 20.0f};
        const float p_z[2]  = {0.0f, 0.0f};
        saans::CsiFrame lo;
        lo.amplitudes = a_lo; lo.phases = p_z; lo.n_subcarriers = kGapSub;
        lo.sample_index = 0;  lo.sample_rate_hz = 20.0f;
        saans::CsiFrame hi = lo;
        hi.amplitudes = a_hi; hi.sample_index = 9;

        g.push(0, lo, g_gap_arena);
        const Filler::Output o = g.push(9, hi, g_gap_arena);
        // Reference: 9 frames, amplitudes 2..10, filled = 8.
        r.equal("a gap of exactly MaxFill fills every arena row",
                static_cast<long long>(o.count), static_cast<long long>(kGapFill + 1));
        r.equal("  filled == MaxFill", g.filled, static_cast<std::uint32_t>(kGapFill));
        // Guarded: a port that bridged fewer frames leaves the later arena
        // rows holding whatever the previous gap put there, and those frames'
        // amplitude pointers are long dead. Report the shortfall rather than
        // dereferencing it — the count check above has already failed by then.
        r.truth("  last arena row holds the last fill",
                o.count == kGapFill + 1 && o[kGapFill - 1].amplitudes[0] == 9.0f &&
                    o[kGapFill].amplitudes[0] == 10.0f,
                "amplitudes 2..10 across arena rows 0..7 then the pass-through");
    }

    {   // A duplicate is dropped BEFORE the history is advanced, so the next
        // gap interpolates from the ORIGINAL frame's payload. Pushing the same
        // frame object twice (as the block above does) cannot see this: the
        // duplicate carries identical samples, so both answers agree. A reused
        // sequence number on the wire can carry entirely different CSI, which
        // is the case that matters and the one this pins.
        Filler      g(5);
        const float a_first[2] = {1.0f, 2.0f};
        const float p_first[2] = {0.0f, 0.0f};
        const float a_dup[2]   = {100.0f, 200.0f};
        const float p_dup[2]   = {1.0f, 1.0f};
        const float a_next[2]  = {3.0f, 6.0f};

        saans::CsiFrame first;
        first.amplitudes = a_first; first.phases = p_first;
        first.n_subcarriers = kGapSub; first.sample_index = 10;
        first.sample_rate_hz = 20.0f;
        saans::CsiFrame dup = first;
        dup.amplitudes = a_dup; dup.phases = p_dup;
        saans::CsiFrame next = first;
        next.amplitudes = a_next; next.sample_index = 12;

        g.push(10, first, g_gap_arena);
        g.push(10, dup, g_gap_arena);          // dropped, history untouched
        const Filler::Output o = g.push(12, next, g_gap_arena);
        r.equal("duplicate with a DIFFERENT payload is still dropped", g.duplicates, 1u);
        r.equal("  the following gap emits one fill + the frame",
                static_cast<long long>(o.count), 2);
        // Reference: the fill is the midpoint of the ORIGINAL (1.0) and 3.0.
        r.truth("  the fill interpolates from the original, not the duplicate",
                o[0].amplitudes[0] == 2.0f && o[0].amplitudes[1] == 4.0f,
                "midpoint of 1 and 3 is 2, not the 51.5 a remembered duplicate gives");
        r.truth("  and holds the ORIGINAL's phase", o[0].phases[0] == 0.0f,
                "the duplicate's 1.0 rad never entered the history");
    }

    {   // sample_rate_hz, rssi_dbm and node_id on a fill frame come from the
        // NEW frame, not the old one — a deliberate fidelity decision in
        // gapfill.hpp, and the one place where taking them from the old frame
        // is the natural mistake. Every other frame in this suite leaves rssi
        // and node at 0, so old and new are indistinguishable there and the
        // choice is unobservable; here they differ.
        Filler      g(5);
        const float a_old[2] = {1.0f, 2.0f};
        const float a_new[2] = {3.0f, 6.0f};
        const float p_z[2]   = {0.0f, 0.0f};
        saans::CsiFrame older;
        older.amplitudes = a_old; older.phases = p_z; older.n_subcarriers = kGapSub;
        older.sample_index = 0;   older.sample_rate_hz = 20.0f;
        older.rssi_dbm = -30;     older.node_id = 1;
        saans::CsiFrame newer = older;
        newer.amplitudes = a_new; newer.sample_index = 2;
        newer.sample_rate_hz = 25.0f; newer.rssi_dbm = -70; newer.node_id = 9;

        g.push(0, older, g_gap_arena);
        const Filler::Output o = g.push(2, newer, g_gap_arena);
        r.equal("one fill between seq 0 and seq 2", static_cast<long long>(o.count), 2);
        r.equal("fill carries the NEW frame's rssi_dbm",
                static_cast<long long>(o[0].rssi_dbm), -70);
        r.equal("fill carries the NEW frame's node_id",
                static_cast<long long>(o[0].node_id), 9);
        r.truth("fill carries the NEW frame's sample_rate_hz",
                o[0].sample_rate_hz == 25.0f,
                "the fill reconstructs the link on the way TO `frame`; node_id picks "
                "the link and rssi_dbm is copied into the wire frame, so taking either "
                "from the old frame mis-attributes every interpolated sample");
    }

    {   // step == 2^31 exactly is the boundary of `step > _SEQ_MOD // 2`: NOT a
        // backwards jump, so it falls through to the gap-length test instead.
        // Either route counts a discontinuity here, which is the point — the
        // boundary cannot produce a 2-billion-frame fill.
        Filler g(5);
        g.push(0, f, g_gap_arena);
        const Filler::Output o = g.push(0x80000000u, f, g_gap_arena);
        r.equal("step == 2^31 does not fill", static_cast<long long>(o.count), 1);
        r.equal("  and is counted", g.discontinuities, 1u);
    }

    {   // A backwards jump (node reboot / reordering).
        Filler g(5);
        f.sample_index = 1000;
        g.push(1000, f, g_gap_arena);
        const Filler::Output o = g.push(998, f, g_gap_arena);
        r.equal("backwards jump passes the frame alone", static_cast<long long>(o.count), 1);
        r.equal("  and is counted", g.discontinuities, 1u);
        r.equal("  history advances to the new sequence", g.last_sequence(), 998u);
    }

    {   // A short gap where the subcarrier count changed: the endpoints are not
        // the same measurement, so there is nothing to interpolate along.
        Filler g(5);
        f.n_subcarriers = kGapSub;
        g.push(0, f, g_gap_arena);
        const float      one_amp[1] = {1.0f};
        const float      one_phs[1] = {0.0f};
        saans::CsiFrame  narrow;
        narrow.amplitudes     = one_amp;
        narrow.phases         = one_phs;
        narrow.n_subcarriers  = 1;
        narrow.sample_index   = 3;
        narrow.sample_rate_hz = 20.0f;
        const Filler::Output o = g.push(3, narrow, g_gap_arena);
        r.equal("subcarrier-count change is not bridged",
                static_cast<long long>(o.count), 1);
        r.equal("  and is counted", g.discontinuities, 1u);
    }

    {   // The SAME guard in the widening direction, which `Filler` cannot
        // reach: a frame wider than kGapSub hits the port-only MaxSubcarriers
        // path first and never gets as far as the count comparison, so
        // `last_n_ == n` and `last_n_ <= n` are indistinguishable above. On an
        // instantiation wide enough to hold both frames the branch is real.
        // The reference bridges nothing here and counts one discontinuity:
        // two endpoints of different widths are not the same measurement.
        using Wide = saans::SequenceGapFiller<8, 8>;
        static Wide::Arena wide_arena;
        Wide               g(5);

        const float a2[2] = {1.0f, 2.0f};
        const float p2[2] = {0.0f, 0.0f};
        const float a4[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float p4[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        saans::CsiFrame two;
        two.amplitudes = a2; two.phases = p2; two.n_subcarriers = 2;
        two.sample_index = 0; two.sample_rate_hz = 20.0f;
        saans::CsiFrame four = two;
        four.amplitudes = a4; four.phases = p4; four.n_subcarriers = 4;
        four.sample_index = 2;

        g.push(0, two, wide_arena);
        const Wide::Output o = g.push(2, four, wide_arena);
        r.equal("2 -> 4 subcarriers across a short gap is not bridged either",
                static_cast<long long>(o.count), 1);
        r.equal("  and is counted", g.discontinuities, 1u);
        r.equal("  with nothing filled", g.filled, 0u);
        r.truth("  (and the frame itself still passes through)",
                o[0].n_subcarriers == 4, "the new width is not the problem; the change is");
    }

    {   // PORT-ONLY: max_gap is clamped to MaxFill so a gap can never ask the
        // arena for more frames than it holds.
        saans::SequenceGapFiller<kGapSub, 4> small(40);
        r.equal("max_gap clamped to MaxFill", small.max_gap(), 4u);
    }

    {   // PORT-ONLY: a frame wider than MaxSubcarriers cannot be retained.
        saans::SequenceGapFiller<kGapSub, kGapFill> g(5);
        f.n_subcarriers = kGapSub;
        g.push(0, f, g_gap_arena);
        const float      wide_a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float      wide_p[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        saans::CsiFrame  wide;
        wide.amplitudes     = wide_a;
        wide.phases         = wide_p;
        wide.n_subcarriers  = 4;  // > kGapSub
        wide.sample_index   = 1;
        wide.sample_rate_hz = 20.0f;
        const Filler::Output o = g.push(1, wide, g_gap_arena);
        r.equal("over-wide frame passes through alone", static_cast<long long>(o.count), 1);
        r.equal("  and is counted", g.discontinuities, 1u);
        r.truth("  history dropped", !g.has_history(), "cannot interpolate from it");
    }

    {   // An invalid frame (no positive rate) is the reference's ValueError
        // path; with -fno-exceptions it becomes a pass-through.
        Filler g(5);
        saans::CsiFrame bad;
        bad.amplitudes     = amps;
        bad.phases         = phs;
        bad.n_subcarriers  = kGapSub;
        bad.sample_index   = 0;
        bad.sample_rate_hz = 0.0f;
        const Filler::Output o = g.push(0, bad, g_gap_arena);
        r.equal("invalid first frame passes through", static_cast<long long>(o.count), 1);
        r.equal("  but is not counted as a break", g.discontinuities, 0u);
    }

    {   // reset() clears both; forget_history() clears only the history.
        Filler g(5);
        g.push(0, f, g_gap_arena);
        g.push(0, f, g_gap_arena);  // duplicate
        g.forget_history();
        r.truth("forget_history keeps counters", g.duplicates == 1u && !g.has_history(),
                "the counters are a session total, the history is not");
        g.reset();
        r.truth("reset clears counters too",
                g.duplicates == 0u && g.filled == 0u && g.discontinuities == 0u, "zeroed");
    }
}

// ---------------------------------------------------------------------------
// fft
// ---------------------------------------------------------------------------

void test_fft(Report& r) {
    const Vectors v("fft64");
    r.begin("radix-2 complex FFT", v.name());

    const auto n = static_cast<std::size_t>(v.integer("n"));
    r.truth("n fits the test scratch", n <= kFftMax,
            std::to_string(n) + " <= " + std::to_string(kFftMax));
    if (n > kFftMax) {
        return;
    }

    const std::vector<float>  in_re  = v.array_f32("in_re");
    const std::vector<float>  in_im  = v.array_f32("in_im");
    const std::vector<double>& ex_re = v.array("out_re");
    const std::vector<double>& ex_im = v.array("out_im");
    r.truth("vector shapes agree",
            in_re.size() == n && in_im.size() == n && ex_re.size() == n && ex_im.size() == n,
            std::to_string(n) + " complex points");

    for (std::size_t i = 0; i < n; ++i) {
        g_fft_re[i] = in_re[i];
        g_fft_im[i] = in_im[i];
    }
    r.truth("fft_inplace accepts a power of two", saans::fft_inplace(g_fft_re, g_fft_im, n),
            "n = " + std::to_string(n));

    r.close_array("out_re", g_fft_re, ex_re.data(), n, kRelTol);
    r.close_array("out_im", g_fft_im, ex_im.data(), n, kRelTol);

    // NORMALISATION. The reference applies no 1/N to the forward transform, so
    // bin 0 is the plain SUM of the inputs, not their mean. These three
    // assertions read THE PORT'S OWN OUTPUT (g_fft_re / g_fft_im, just
    // overwritten by fft_inplace) against a sum of the input, which is the
    // only arrangement that can catch a scaling bug. An earlier version
    // compared the vector's expected array against a sum of the vector's input
    // array: self-consistency of the fixture, true no matter what the port
    // did. Verified by mutation — a 1/N inside the butterfly left all three
    // printing `ok` with identical deviations.
    double sum_re = 0.0;
    double sum_im = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        sum_re += static_cast<double>(in_re[i]);
        sum_im += static_cast<double>(in_im[i]);
    }
    r.close("bin 0 == sum(x), i.e. the PORT applies no 1/N scaling",
            static_cast<double>(g_fft_re[0]), sum_re, kRelTol);
    r.close("bin 0 imag == sum(imag)", static_cast<double>(g_fft_im[0]), sum_im, kRelTol);

    // Parseval with this convention: sum|X|^2 == N * sum|x|^2. Also unscaled,
    // and also over the port's spectrum rather than the vector's.
    double energy_t = 0.0;
    double energy_f = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        const double xr = static_cast<double>(in_re[i]);
        const double xi = static_cast<double>(in_im[i]);
        energy_t += xr * xr + xi * xi;
        const double Xr = static_cast<double>(g_fft_re[i]);
        const double Xi = static_cast<double>(g_fft_im[i]);
        energy_f += Xr * Xr + Xi * Xi;
    }
    r.close("Parseval on the port's spectrum: sum|X|^2 == N*sum|x|^2", energy_f,
            static_cast<double>(n) * energy_t, kRelTol);

    // The fixture's own agreement with itself is worth knowing — it says the
    // vector was generated with the convention the port implements, not just
    // that the port is self-consistent — but it is NOT a port margin, so it
    // goes through close_abs, which is excluded from the headline "worst
    // margin used". Reporting the fixture's agreement with itself as the
    // port's margin flattered the summary by ~37x in the reader's favour.
    r.close_abs("vector sanity: the fixture's own bin 0 is sum(x) too",
                saans::golden::rel_dev(ex_re[0], sum_re), 0.0, 1e-6);
}

void test_fft_guards(Report& r) {
    r.begin("FFT guards and next_pow2", "(reference semantics, no vector)");

    // `_next_pow2`: the doubling loop, including its edges.
    r.equal("next_pow2(0)", static_cast<long long>(saans::next_pow2(0)), 1);
    r.equal("next_pow2(1)", static_cast<long long>(saans::next_pow2(1)), 1);
    r.equal("next_pow2(2)", static_cast<long long>(saans::next_pow2(2)), 2);
    r.equal("next_pow2(3)", static_cast<long long>(saans::next_pow2(3)), 4);
    r.equal("next_pow2(120)", static_cast<long long>(saans::next_pow2(120)), 128);
    r.equal("next_pow2(128)", static_cast<long long>(saans::next_pow2(128)), 128);
    r.equal("next_pow2(129)", static_cast<long long>(saans::next_pow2(129)), 256);

    r.truth("is_pow2(0) is false", !saans::is_pow2(0), "0 is not a power of two");
    r.truth("is_pow2(1)", saans::is_pow2(1), "");
    r.truth("is_pow2(96) is false", !saans::is_pow2(96), "");

    // PORT-ONLY: the doubling loop terminates. CPython's ints are unbounded so
    // `_next_pow2` always does; `p <<= 1` on a size_t wraps to 0 once p is the
    // top power of two, and `0 < n` then stays true forever. On the S3 size_t
    // is 32 bits, so the threshold is n > 2^31 rather than the host's 2^63 —
    // one upstream `end - start` with start > end and the CSI task spins until
    // the watchdog resets the node. Refusing with 0 is what the rest of these
    // headers do with a precondition they cannot meet.
    constexpr std::size_t kTop = (SIZE_MAX >> 1) + 1u;
    r.equal("next_pow2(2^(bits-1)) is itself", static_cast<long long>(saans::next_pow2(kTop)),
            static_cast<long long>(kTop));
    r.truth("next_pow2(2^(bits-1) + 1) refuses with 0", saans::next_pow2(kTop + 1u) == 0,
            "no power of two fits above it");
    r.truth("next_pow2(SIZE_MAX) refuses with 0, it does not hang",
            saans::next_pow2(static_cast<std::size_t>(-1)) == 0, "");
    r.truth("  and 0 is rejected downstream", !saans::is_pow2(saans::next_pow2(kTop + 1u)),
            "fft_inplace and subcarrier_bnr both refuse it");
    // constexpr, so the guard is also a compile-time diagnostic rather than a
    // compile-time hang. This static_assert is the proof that it folds.
    static_assert(saans::next_pow2(static_cast<std::size_t>(-1)) == 0,
                  "next_pow2 must fold at compile time, not spin");

    // Non-power-of-two and null are refused rather than producing nonsense —
    // the no-exceptions form of the reference's documented precondition.
    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    r.truth("fft_inplace refuses n = 6", !saans::fft_inplace(a, b, 6), "not a power of two");
    r.truth("fft_inplace refuses n = 0", !saans::fft_inplace(a, b, 0), "");
    r.truth("fft_inplace refuses null", !saans::fft_inplace(nullptr, b, 8), "");
    const bool untouched = a[0] == 1.0f && a[7] == 8.0f;
    r.truth("a refused transform touches nothing", untouched, "buffer unchanged");

    // n == 1 is the identity, as `_fft` on a one-element list is.
    float one_re[1] = {2.5f};
    float one_im[1] = {-1.5f};
    r.truth("fft_inplace(n = 1) succeeds", saans::fft_inplace(one_re, one_im, 1), "");
    r.truth("  and is the identity", one_re[0] == 2.5f && one_im[0] == -1.5f, "");

    // A unit impulse transforms to all ones — the cheapest end-to-end check
    // that the bit-reversal and the butterflies agree with each other.
    float im_re[16];
    float im_im[16];
    for (std::size_t i = 0; i < 16; ++i) {
        im_re[i] = (i == 0) ? 1.0f : 0.0f;
        im_im[i] = 0.0f;
    }
    saans::fft_inplace(im_re, im_im, 16);
    bool flat = true;
    for (std::size_t i = 0; i < 16; ++i) {
        flat = flat && im_re[i] == 1.0f && im_im[i] == 0.0f;
    }
    r.truth("delta -> flat spectrum, exactly", flat, "16 bins of 1 + 0j");
}

// ---------------------------------------------------------------------------
// bnr
// ---------------------------------------------------------------------------

void test_bnr(Report& r) {
    const Vectors v("bnr");
    r.begin("subcarrier_bnr", v.name());

    const auto  rows = static_cast<std::size_t>(v.integer("rows"));
    const auto  cols = static_cast<std::size_t>(v.integer("cols"));
    const float rate = v.scalar_f32("rate_hz");
    const float f_lo = v.scalar_f32("f_low");
    const float f_hi = v.scalar_f32("f_high");

    const std::vector<float>   win = v.array_f32("window");
    const std::vector<double>& ex  = v.array("bnr");
    r.truth("vector shapes agree", win.size() == rows * cols && ex.size() == cols,
            std::to_string(rows) + " x " + std::to_string(cols) + " window");
    r.truth("window fits the test scratch", rows <= kMaxRows && cols <= kMaxCols, "");
    if (rows > kMaxRows || cols > kMaxCols || win.size() != rows * cols) {
        return;
    }
    for (std::size_t i = 0; i < win.size(); ++i) {
        g_window[i] = win[i];
    }

    saans::SubcarrierBnrParams p;
    p.f_low  = f_lo;
    p.f_high = f_hi;

    const saans::SubcarrierBnrResult res =
        saans::subcarrier_bnr(g_window, rows, cols, cols, rate, g_bnr, kMaxCols,
                              g_bnr_ws.re, g_bnr_ws.im, g_bnr_ws.capacity(), p);

    r.truth("call succeeded", res.ok && res.ran, "ok and past every reference guard");
    r.equal("subcarriers written", static_cast<long long>(res.n_subcarriers),
            static_cast<long long>(cols));
    r.equal("nfft = next_pow2(rows)", static_cast<long long>(res.nfft),
            static_cast<long long>(saans::next_pow2(rows)));
    // Bin accounting, asserted rather than printed. The split runs over
    // m = 1 .. nfft/2 in the reference, so in + out is exactly nfft/2 and bin
    // 0 is in neither; a scan that included DC would report nfft/2 + 1 here
    // and nothing else in the suite would object.
    r.equal("in + out == nfft/2, DC counted in neither",
            static_cast<long long>(res.n_in_band + res.n_out_band),
            static_cast<long long>(res.nfft / 2u));
    r.truth("bin_hz == rate / nfft, exactly",
            res.bin_hz == rate / static_cast<float>(res.nfft),
            "nfft is a power of two, so the division is exact whenever the rate is");
    r.note("bin_hz %.9g, %zu in-band bins, %zu out-of-band, %zu subcarriers scored",
           static_cast<double>(res.bin_hz), res.n_in_band, res.n_out_band, res.n_scored);

    // Per-subcarrier, because the interesting cases are individual: a constant
    // series must be exactly 0, a noise-only one must be small, and the
    // breathing ones must be large. close_array's full-scale metric would let a
    // small subcarrier be wrong by the large one's scale.
    for (std::size_t k = 0; k < cols; ++k) {
        const std::string what = "bnr[" + std::to_string(k) + "]";
        if (ex[k] == 0.0) {
            // The constant-7.0 column. "Left at 0" is a SKIP, not a computed
            // ratio that happens to be small, so it must be exactly zero.
            r.truth(what + " exactly 0 (series skipped)", g_bnr[k] == 0.0f,
                    "constant or non-finite series is left at zero, not scored");
        } else {
            r.close(what, static_cast<double>(g_bnr[k]), ex[k], kRelTol);
        }
    }

    // The reference's own reading of this fixture: three breathing-modulated
    // subcarriers (two of them sign-flipped against each other), one at the
    // band edge, noise, a constant, a drift and a noisy breather. Asserting the
    // RANKING as well as the values catches a port that is uniformly wrong by
    // a scale factor, which the per-value tolerance would also catch but which
    // is worth naming.
    std::size_t best = 0;
    for (std::size_t k = 1; k < cols; ++k) {
        if (g_bnr[k] > g_bnr[best]) {
            best = k;
        }
    }
    std::size_t ex_best = 0;
    for (std::size_t k = 1; k < cols; ++k) {
        if (ex[k] > ex[ex_best]) {
            ex_best = k;
        }
    }
    r.equal("argmax subcarrier (the best_bnr field)", static_cast<long long>(best),
            static_cast<long long>(ex_best));
}

void test_bnr_guards(Report& r) {
    r.begin("subcarrier_bnr guards", "(reference semantics, no vector)");

    constexpr std::size_t kRows = 64;
    constexpr std::size_t kCols = 3;
    static float          w[kRows * kCols];
    float                 out[kCols];

    for (std::size_t t = 0; t < kRows; ++t) {
        const float x = sinf(2.0f * saans::kPi * 0.25f * (static_cast<float>(t) / 4.0f));
        w[t * kCols + 0] = x;              // breathing
        w[t * kCols + 1] = 7.0f;           // constant -> skipped
        w[t * kCols + 2] = x;              // overwritten below
    }
    w[5 * kCols + 2] = NAN;                // non-finite -> skipped

    saans::SubcarrierBnrParams p;  // reference defaults: 0.1 .. 0.5 Hz

    {
        const auto res = saans::subcarrier_bnr(w, kRows, kCols, kCols, 4.0f, out, kCols,
                                               g_bnr_ws.re, g_bnr_ws.im,
                                               g_bnr_ws.capacity(), p);
        r.truth("ran on 64 rows", res.ok && res.ran, "");
        r.truth("constant series left at 0", out[1] == 0.0f, "std <= 0 -> skipped");
        r.truth("non-finite series left at 0", out[2] == 0.0f, "not all finite -> skipped");
        r.truth("breathing series scored", out[0] > 0.0f && saans::is_finite(out[0]),
                "a 0.25 Hz tone sits inside 0.1-0.5 Hz");
        r.equal("two of three skipped", static_cast<long long>(res.n_scored), 1);
    }

    {   // `if n_t < 8` — zeros, and the call still reports success, exactly as
        // the reference returns [0.0]*n_k rather than raising.
        out[0] = 9.0f;
        const auto res = saans::subcarrier_bnr(w, 7, kCols, kCols, 4.0f, out, kCols,
                                               g_bnr_ws.re, g_bnr_ws.im,
                                               g_bnr_ws.capacity(), p);
        r.truth("n_rows < 8 returns zeros, not an error",
                res.ok && !res.ran && out[0] == 0.0f && out[1] == 0.0f && out[2] == 0.0f,
                "the reference's [0.0] * n_k path");
    }

    {   // `sample_rate_hz <= 0.0`
        out[0] = 9.0f;
        const auto res = saans::subcarrier_bnr(w, kRows, kCols, kCols, 0.0f, out, kCols,
                                               g_bnr_ws.re, g_bnr_ws.im,
                                               g_bnr_ws.capacity(), p);
        r.truth("non-positive rate returns zeros", res.ok && !res.ran && out[0] == 0.0f, "");
    }

    {   // `if not in_band or not out_band` — push the band off the grid.
        saans::SubcarrierBnrParams off = p;
        off.f_low  = 100.0f;
        off.f_high = 200.0f;
        out[0]     = 9.0f;
        const auto res = saans::subcarrier_bnr(w, kRows, kCols, kCols, 4.0f, out, kCols,
                                               g_bnr_ws.re, g_bnr_ws.im,
                                               g_bnr_ws.capacity(), off);
        r.truth("empty in-band returns zeros", res.ok && !res.ran && out[0] == 0.0f,
                "no numerator -> no meaningful ratio");
        r.equal("  in-band bin count is 0", static_cast<long long>(res.n_in_band), 0);
    }

    {   // `if not out_band` — the OTHER empty-band guard, which is a different
        // branch from the +inf one below and must not be confused with it. A
        // band covering the whole spectrum leaves no out-of-band bins at all,
        // so the reference returns zeros before computing anything.
        saans::SubcarrierBnrParams wide = p;
        wide.f_low  = 0.0f;
        wide.f_high = 2.0f;  // 2 Hz == Nyquist at 4 Hz: every bin is in-band
        const auto res = saans::subcarrier_bnr(w, kRows, kCols, kCols, 4.0f, out, kCols,
                                               g_bnr_ws.re, g_bnr_ws.im,
                                               g_bnr_ws.capacity(), wide);
        r.equal("whole-spectrum band leaves no denominator",
                static_cast<long long>(res.n_out_band), 0);
        r.truth("  and is guarded, not divided by zero", res.ok && !res.ran, "");
    }

    {   // den == 0 -> +inf, ON A REAL FIXTURE.
        //
        // The spec mandates it and bnr.hpp devotes a section to why a large
        // finite sentinel would be wrong — it would win every max() it is
        // supposed to be excluded from. Nothing asserted it: the block above
        // exercises the `not out_band` guard, which is a different branch, and
        // the comment that used to sit here claimed "energy only in-band is
        // unreachable with a finite window". That is false, and this is the
        // counterexample.
        //
        // Column [1,0,-1,0,1,0,-1,0] at 8 Hz: nfft = 8, no padding, bin_hz =
        // 1 Hz, half = 4. Over 0.5 .. 2.5 Hz the in-band bins are {1, 2} and
        // the out-of-band ones are {3, 4}. The series is the 2 Hz Nyquist/2
        // component alone, so after z-scoring bins 3 and 4 come out EXACTLY
        // zero in float32 and den is exactly 0.0f.
        //
        // The Python reference returns [inf] for the same input (verified),
        // and `spectra` still gets an entry — an infinite ratio is a scored
        // subcarrier, not a skipped one.
        static float      alt[8]  = {1.0f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, -1.0f, 0.0f};
        float             io[1]   = {0.0f};
        bool              ipres[1] = {false};
        saans::SubcarrierBnrParams ip;
        ip.f_low          = 0.5f;
        ip.f_high         = 2.5f;
        ip.spectra_present = ipres;
        const auto res = saans::subcarrier_bnr(alt, 8, 1, 1, 8.0f, io, 1, g_bnr_ws.re,
                                               g_bnr_ws.im, g_bnr_ws.capacity(), ip);
        r.truth("den == 0 fixture ran", res.ok && res.ran, "8 rows, nfft 8, bin_hz 1 Hz");
        r.equal("  in-band bins {1,2}", static_cast<long long>(res.n_in_band), 2);
        r.equal("  out-of-band bins {3,4}", static_cast<long long>(res.n_out_band), 2);
        r.truth("den == 0 yields +inf, not a finite sentinel",
                std::isinf(io[0]) && io[0] > 0.0f,
                "`num / den if den > 0.0 else float(\"inf\")`; a sentinel would win "
                "every max() it is meant to be excluded from");
        r.truth("  and the subcarrier counts as scored", ipres[0] && res.n_scored == 1,
                "an infinite ratio is an answer, not a skip");
    }

    {   // THE max_gap OF THIS MODULE: the band-edge compare, named in bnr.hpp
        // as one of the two float32 classification hazards. Only the UPPER
        // edge was pinned, and only by accident — at 4 Hz / nfft 128 bin 16
        // lands exactly on f_high = 0.5, so `f <= f_high` is exercised. No
        // fixture put a bin centre on f_low, because bin_hz is 0.03125 on the
        // golden grid and 0.0625 in the guards and 0.1 is a multiple of
        // neither, so `f >= f_low` could be weakened to `f > f_low` unnoticed.
        //
        // 6.4 Hz over a 64-point transform makes bin_hz exactly 0.1f, so bin 1
        // sits exactly on f_low. The reference's `in_band` for this grid is
        // [1, 2, 3, 4, 5] (verified), i.e. the edge bin is IN.
        std::size_t nin = 0, nout = 0;
        float       bh  = 0.0f;
        saans::bnr_band_bins(64, 6.4f, 0.1f, 0.5f, &nin, &nout, &bh);
        r.truth("bin 1 lands exactly on f_low at 6.4 Hz / 64", bh == 0.1f,
                "6.4 / 64 is exact in binary; 0.1f is not, but both sides round the same");
        r.equal("lower band edge is INCLUSIVE: 5 in-band bins",
                static_cast<long long>(nin), 5);
        r.equal("  and 27 out of band", static_cast<long long>(nout), 27);
        r.equal("  in + out == nfft/2 (bin 0 is in neither)",
                static_cast<long long>(nin + nout), 32);
    }

    {   // A CONSTANT SERIES IS A CLASSIFICATION, NOT A COMPARISON.
        //
        // The reference reaches `if std <= 0.0` with an exact zero for any
        // constant column, because the input is float32-representable and
        // `n_t * c` therefore fits a f64 mantissa without rounding. In float32
        // the same round trip fails for ~6.3% of amplitudes at the pipeline's
        // own 120-row window, and the column then z-scores to a constant and
        // produces a fabricated ratio.
        //
        // c = 31.6778717f over 120 rows is one such value: the f32 mean comes
        // back one ulp high, and before the exact-equality scan in bnr.hpp the
        // port returned 1.10416269 where the reference returns 0.0 — above
        // three of the seven real columns in the golden vector, and `best_bnr`
        // is a max() over the finite values, so a room of dead subcarriers
        // reported link quality 1.10.
        //
        // The golden fixture's constant column is 7.0, and 7.0 * 120 = 840 is
        // exactly representable in float32, so the vector cannot see this. It
        // needs a hand-written value and a non-power-of-two row count.
        constexpr std::size_t kOdd = 120;  // features.py: int(30.0 * 4.0)
        static float          cw[kOdd];
        float                 co[1]    = {9.0f};
        bool                  cpres[1] = {true};
        saans::SubcarrierBnrParams cp;
        cp.spectra_present = cpres;

        for (std::size_t t = 0; t < kOdd; ++t) {
            cw[t] = 31.6778717f;
        }
        // The hazard itself, stated as an assertion so the fixture cannot
        // quietly become a safe constant later.
        saans::NeumaierSum ms;
        for (std::size_t t = 0; t < kOdd; ++t) {
            ms.add(cw[t]);
        }
        r.truth("the fixture's f32 mean does NOT round-trip",
                ms.value() / static_cast<float>(kOdd) != cw[0],
                "sum 3801.34473f / 120 == 31.6778736f, one ulp above 31.6778717f; "
                "in f64 the same sum divides back exactly, which is why the "
                "reference's `std <= 0` is exact and this one would not be");

        const auto res = saans::subcarrier_bnr(cw, kOdd, 1, 1, 4.0f, co, 1, g_bnr_ws.re,
                                               g_bnr_ws.im, g_bnr_ws.capacity(), cp);
        r.truth("constant series is skipped anyway", res.ok && res.ran && co[0] == 0.0f,
                "the reference returns 0.0 with no spectra entry; a float32 `sd > 0` "
                "would have scored it at 1.10416269");
        r.truth("  and has no spectra entry", !cpres[0], "skipped, not scored");
        r.equal("  nothing scored", static_cast<long long>(res.n_scored), 0);

        // The same column one ulp away from constant must still be SCORED —
        // the skip is an equality scan, not a tolerance, so it must not
        // swallow a real (if tiny) modulation the reference would score.
        cw[7] = std::nextafterf(cw[7], 1e30f);
        co[0] = 9.0f;
        const auto res2 = saans::subcarrier_bnr(cw, kOdd, 1, 1, 4.0f, co, 1, g_bnr_ws.re,
                                                g_bnr_ws.im, g_bnr_ws.capacity(), cp);
        r.truth("one ulp of modulation is still scored",
                res2.ok && res2.ran && res2.n_scored == 1 && cpres[0],
                "the reference's std is > 0 here, so the port must not skip it either");
    }

    {   // PORT-ONLY: an upstream size_t underflow reaches next_pow2 as a row
        // count with no power of two above it. It must come back as ok=false,
        // not as a doubling loop that never terminates. Without the guard this
        // single call hangs the binary — which is how it would present on
        // target: the CSI task spins until the watchdog resets the node, and
        // nothing in the result struct ever reports the bad length.
        float      uo[1] = {9.0f};
        const auto res   = saans::subcarrier_bnr(w, ~static_cast<std::size_t>(0) - 3u, 1, 1,
                                                 4.0f, uo, 1, g_bnr_ws.re, g_bnr_ws.im,
                                                 g_bnr_ws.capacity());
        r.truth("an impossible row count is refused, not spun on",
                !res.ok && res.nfft == 0, "next_pow2 returns 0 and subcarrier_bnr refuses");
    }

    {   // PORT-level failures must be distinguishable from a legitimate zero.
        out[0]        = 9.0f;
        const auto r1 = saans::subcarrier_bnr(w, kRows, kCols, kCols, 4.0f, out, kCols - 1,
                                              g_bnr_ws.re, g_bnr_ws.im,
                                              g_bnr_ws.capacity(), p);
        r.truth("too-small output buffer is ok=false, nothing written",
                !r1.ok && out[0] == 9.0f, "never mistakable for a computed zero");

        const auto r2 = saans::subcarrier_bnr(w, kRows, kCols, kCols, 4.0f, out, kCols,
                                              g_bnr_ws.re, g_bnr_ws.im, 4, p);
        r.truth("too-small scratch is ok=false", !r2.ok, "cannot transform");

        const auto r3 = saans::subcarrier_bnr(nullptr, kRows, kCols, kCols, 4.0f, out, kCols,
                                              g_bnr_ws.re, g_bnr_ws.im,
                                              g_bnr_ws.capacity(), p);
        r.truth("null window is ok=false", !r3.ok, "");
    }

    {   // The OPTIONAL in-band spectra — the `spectra` half of the reference's
        // (bnr, spectra) return, which respirfi_select will need. Checked
        // against an independent transform of the same z-scored column rather
        // than against itself, and the `present` flags against the reference's
        // dict semantics: a skipped subcarrier has no entry, which a dense
        // array can only express through the flag.
        std::size_t n_in = 0;
        saans::bnr_band_bins(saans::next_pow2(kRows), 4.0f, p.f_low, p.f_high, &n_in,
                             nullptr, nullptr);
        static float sp_re[kCols * 32];
        static float sp_im[kCols * 32];
        bool         present[kCols];
        r.truth("in-band bin count fits the test buffer", n_in <= 32,
                std::to_string(n_in) + " bins");

        saans::SubcarrierBnrParams sp = p;
        sp.spectra_re      = sp_re;
        sp.spectra_im      = sp_im;
        sp.spectra_stride  = 32;
        sp.spectra_present = present;
        const auto res = saans::subcarrier_bnr(w, kRows, kCols, kCols, 4.0f, out, kCols,
                                               g_bnr_ws.re, g_bnr_ws.im,
                                               g_bnr_ws.capacity(), sp);
        r.truth("spectra request succeeds", res.ok && res.ran, "");
        r.truth("present flags match the reference's dict keys",
                present[0] && !present[1] && !present[2],
                "scored subcarriers only; a skipped one has no entry");

        // Recompute subcarrier 0 independently: z-score, pad, transform.
        const std::size_t nf = res.nfft;
        static float      re2[128];
        static float      im2[128];
        saans::NeumaierSum m;
        for (std::size_t t = 0; t < kRows; ++t) {
            m.add(w[t * kCols + 0]);
        }
        const float mean = m.value() / static_cast<float>(kRows);
        saans::NeumaierSum vv;
        for (std::size_t t = 0; t < kRows; ++t) {
            const float d = w[t * kCols + 0] - mean;
            vv.add(d * d);
        }
        const float sd = sqrtf(vv.value() / static_cast<float>(kRows));
        for (std::size_t t = 0; t < nf; ++t) {
            re2[t] = (t < kRows) ? (w[t * kCols + 0] - mean) * (1.0f / sd) : 0.0f;
            im2[t] = 0.0f;
        }
        saans::fft_inplace(re2, im2, nf);

        double worst = 0.0;
        std::size_t j = 0;
        for (std::size_t mm = 1; mm <= nf / 2u; ++mm) {
            const float f = static_cast<float>(mm) * res.bin_hz;
            if (f >= p.f_low && f <= p.f_high) {
                const double dr = static_cast<double>(sp_re[j]) - static_cast<double>(re2[mm]);
                const double di = static_cast<double>(sp_im[j]) - static_cast<double>(im2[mm]);
                worst = std::fmax(worst, std::fabs(dr));
                worst = std::fmax(worst, std::fabs(di));
                ++j;
            }
        }
        r.equal("spectra holds every in-band bin", static_cast<long long>(j),
                static_cast<long long>(n_in));
        r.close_abs("spectra bins match an independent transform", worst, 0.0, 0.0);

        // Half a buffer, or too short a stride, is a caller bug and not a
        // degraded mode.
        saans::SubcarrierBnrParams half = p;
        half.spectra_re = sp_re;  // and spectra_im left null
        const auto h = saans::subcarrier_bnr(w, kRows, kCols, kCols, 4.0f, out, kCols,
                                             g_bnr_ws.re, g_bnr_ws.im,
                                             g_bnr_ws.capacity(), half);
        r.truth("half a spectra buffer is ok=false", !h.ok, "re without im");

        saans::SubcarrierBnrParams narrow = sp;
        narrow.spectra_stride = 1;
        const auto nres = saans::subcarrier_bnr(w, kRows, kCols, kCols, 4.0f, out, kCols,
                                                g_bnr_ws.re, g_bnr_ws.im,
                                                g_bnr_ws.capacity(), narrow);
        r.truth("too-short spectra stride is ok=false", !nres.ok,
                "stride must hold every in-band bin");
    }

    {   // BINS RUN FROM 1, NEVER 0 — pinned structurally, because it cannot be
        // pinned by a value.
        //
        // The obvious test (add a DC offset, check the ratio is unchanged) is
        // below and is worth keeping, but it does NOT test this: the series is
        // z-scored before the transform, so bin 0 is the residue of the mean
        // subtraction whether or not an offset was present. Measured on the
        // fixture below, |X0|^2 is 1.3e-13 without the pedestal and 8.3e-10
        // with it, against a denominator of 409.6 — a 2e-12 relative
        // contribution either way. The ratio is identical to nine significant
        // figures whether bin 0 is counted or not, so that check passed for a
        // reason unrelated to the property it named. (Both loop bounds survived
        // being changed to `m = 0`.)
        //
        // What DOES pin the bound is the bin accounting. `range(1, half + 1)`
        // splits exactly half bins between in and out; counting bin 0 makes it
        // half + 1.
        std::size_t nin = 0, nout = 0;
        saans::bnr_band_bins(128, 4.0f, p.f_low, p.f_high, &nin, &nout, nullptr);
        r.equal("bnr_band_bins splits exactly nfft/2 bins, DC in neither",
                static_cast<long long>(nin + nout), 64);
        r.equal("  in-band count at the golden 4 Hz / 128 grid",
                static_cast<long long>(nin), 13);
        r.equal("  out-of-band count", static_cast<long long>(nout), 51);

        // And in the scan itself: with f_low = 0, bin 0 would satisfy
        // `0 >= f_low && 0 <= f_high` and be written as the FIRST in-band
        // spectrum bin, shifting every later one by a slot. Comparing the
        // emitted spectra against an independent transform at exact equality
        // therefore fails the moment the loop starts at 0, which a frequency
        // band that excludes DC can never detect.
        saans::SubcarrierBnrParams dc;
        dc.f_low  = 0.0f;   // DC is "in band" if it is scanned at all
        dc.f_high = 0.5f;
        static float dc_re[16];
        static float dc_im[16];
        bool         dc_present[1] = {false};
        float        dc_out[1];
        dc.spectra_re      = dc_re;
        dc.spectra_im      = dc_im;
        dc.spectra_stride  = 16;
        dc.spectra_present = dc_present;

        static float breath[kRows];
        for (std::size_t t = 0; t < kRows; ++t) {
            const float tt = static_cast<float>(t) / 4.0f;
            breath[t] = sinf(2.0f * saans::kPi * 0.25f * tt) +
                        0.5f * sinf(2.0f * saans::kPi * 1.5f * tt + 0.7f);
        }
        const auto dres = saans::subcarrier_bnr(breath, kRows, 1, 1, 4.0f, dc_out, 1,
                                                g_bnr_ws.re, g_bnr_ws.im,
                                                g_bnr_ws.capacity(), dc);
        r.truth("f_low = 0 run succeeded", dres.ok && dres.ran && dc_present[0], "");
        r.equal("  in-band bins are 1..8, not 0..8",
                static_cast<long long>(dres.n_in_band), 8);
        // Independent transform of the same z-scored column.
        static float ref_re[64];
        static float ref_im[64];
        saans::NeumaierSum dm;
        for (std::size_t t = 0; t < kRows; ++t) {
            dm.add(breath[t]);
        }
        const float dmean = dm.value() / static_cast<float>(kRows);
        saans::NeumaierSum dv;
        for (std::size_t t = 0; t < kRows; ++t) {
            const float d = breath[t] - dmean;
            dv.add(d * d);
        }
        const float dsd = sqrtf(dv.value() / static_cast<float>(kRows));
        for (std::size_t t = 0; t < 64; ++t) {
            ref_re[t] = (t < kRows) ? (breath[t] - dmean) * (1.0f / dsd) : 0.0f;
            ref_im[t] = 0.0f;
        }
        saans::fft_inplace(ref_re, ref_im, 64);
        bool aligned = true;
        for (std::size_t j = 0; j < dres.n_in_band; ++j) {
            aligned = aligned && dc_re[j] == ref_re[j + 1] && dc_im[j] == ref_im[j + 1];
        }
        r.truth("the first emitted bin is bin 1, not bin 0", aligned,
                "a scan from m = 0 would put the DC residue in slot 0 and shift the rest");
        r.note("|X0|^2 = %.3g against the in-band bin 1 power %.3g: DC is excluded "
               "because it is the mean-subtraction residue, not because it is large",
               static_cast<double>(saans::power_at(ref_re, ref_im, 0)),
               static_cast<double>(saans::power_at(ref_re, ref_im, 1)));
    }

    {   // A large constant offset on a breathing column: after z-scoring the
        // offset is gone, so the ratio must be unchanged. This is a check on
        // the z-score, NOT on DC exclusion — see the block above for why those
        // are different properties and which one this cannot see.
        static float w2[kRows * 1];
        float        o1[1];
        float        o2[1];
        // In-band tone at 0.25 Hz plus an OUT-OF-BAND tone at 1.5 Hz. The
        // second one is not decoration: without it the denominator of a
        // noiseless, exactly-on-bin tone is pure rounding residue (~1e-12) and
        // the ratio is noise over noise, unstable at any tolerance and
        // therefore worthless as a regression check. Both land exactly on bins
        // of the 0.0625 Hz grid, so the comparison is between two real
        // energies.
        for (std::size_t t = 0; t < kRows; ++t) {
            const float tt = static_cast<float>(t) / 4.0f;
            w2[t] = sinf(2.0f * saans::kPi * 0.25f * tt) +
                    0.5f * sinf(2.0f * saans::kPi * 1.5f * tt + 0.7f);
        }
        saans::subcarrier_bnr(w2, kRows, 1, 1, 4.0f, o1, 1, g_bnr_ws.re, g_bnr_ws.im,
                              g_bnr_ws.capacity(), p);
        for (std::size_t t = 0; t < kRows; ++t) {
            w2[t] += 50.0f;
        }
        saans::subcarrier_bnr(w2, kRows, 1, 1, 4.0f, o2, 1, g_bnr_ws.re, g_bnr_ws.im,
                              g_bnr_ws.capacity(), p);
        r.close("DC offset does not change the ratio", static_cast<double>(o2[0]),
                static_cast<double>(o1[0]), kRelTol);
        r.note("ratio %.6g without the offset, %.6g with it; the z-score removes the "
               "mean, so a +50 pedestal on a ~1.5 amplitude series moves nothing but "
               "the last bits of (x - mean)",
               static_cast<double>(o1[0]), static_cast<double>(o2[0]));
    }
}

}  // namespace

int main(int argc, char** argv) {
    // Accept the vector directory as argv[1], the convention edge/dsp/Makefile
    // uses, so this test runs identically under either Makefile in the tree.
    saans::golden::dir_from_argv(argc, argv);

    std::printf("Saans edge DSP - gapfill / fft / bnr parity against the Python reference\n");
    std::printf("vectors: %s\n", saans::golden::dir().c_str());
    std::printf("contract: %.0e relative on continuous floats, exact on counters and "
                "sequence numbers (ADR-003)\n",
                kRelTol);

    Report r;

    test_gapfill(r);
    test_gapfill_guards(r);

    test_fft(r);
    test_fft_guards(r);

    test_bnr(r);
    test_bnr_guards(r);

    return r.finish("gapfill_bnr");
}
