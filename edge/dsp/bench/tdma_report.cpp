// SPDX-License-Identifier: MIT
//
// Saans edge DSP — broadcast-TDMA viability report for an N-node mesh (S6).
//
// ===========================================================================
// SYNTHETIC — MODEL, NOT MEASURED. NO HARDWARE EXISTS YET.
// ===========================================================================
// This program runs `dsp/tdma.hpp` over a grid of mesh sizes and slot lengths
// and prints what the schedule would deliver. It does not talk to a radio, it
// has never talked to a radio, and every figure it prints is a consequence of
// the assumptions listed at the top of its own output. Those assumptions are
// printed BESIDE the numbers, every run, because a table of per-link rates
// with the airtime estimate omitted is a table that will be quoted without it.
//
// WHAT IT IS FOR. Deciding how many nodes a space can carry before the
// per-link CSI rate falls below what the DSP needs — before anyone buys the
// boards. The answer turns out not to be about the radio at all; see "WHERE
// THE SCHEME STOPS WORKING" in the output.
//
//   build and run:
//       make -C edge/dsp/test tdma-report
//
//   or directly, from the repository root:
//       g++ -std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion
//           -fno-exceptions -fno-rtti -I edge/dsp edge/dsp/bench/tdma_report.cpp
//           -o /tmp/tdma_report && /tmp/tdma_report
//   (one line; it is split here only because a backslash continuation inside a
//    // comment is itself a line continuation and -Wcomment says so)
//
//   arguments (all optional, in this order):
//       bytes_per_link_bnr_on  bytes_per_link_bnr_off  per_node_ram_budget_bytes
//       shared_once_per_node_bytes
//
// They are arguments rather than constants because they are the numbers most
// likely to be stale: they come from `sizeof(BasicFeatureExtractor<...>)` and
// from whatever the firmware has left over after ESP-IDF and the WiFi stack,
// and both move. The defaults and their provenance are printed in the
// assumptions block.
//
// argv[4] IS THE ONE THAT IS EASY TO FORGET, and leaving it out is how this
// report came to contradict the one it calls the authority. `size_report`
// budgets with `links_in_budget(per_link, shared_once) = (free - shared_once)
// / per_link`: the node build pays for ONE spectral-scratch arena and then
// per-link state on top of it. Its `once/node` column is argv[4]. Drop that
// term and this report blesses 10 links where size_report fits 9, against the
// same budget and the same sizeof().
//
// `-Wdouble-promotion` note: every float handed to printf is cast to double
// explicitly. The warning is on for the library headers for a real reason (an
// emulated double on the S3 costs 10-50x) and silencing it per-file would also
// silence it for the arithmetic.
//
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "tdma.hpp"

using saans::LinkAssignment;
using saans::TdmaConfig;
using saans::TdmaSchedule;
using saans::TdmaStatus;

namespace {

// ---------------------------------------------------------------------------
// The figures this report is parameterised by
// ---------------------------------------------------------------------------

// THE THREE MEMORY FIGURES, AND WHERE THEY ARE REALLY KEPT.
//
// `bench/size_report.cpp` is the authority: it measures `sizeof` on the
// actual instantiations, proves the trimmed shapes emit byte-identical
// FeatureFrames, and says how many links fit. This file holds planning
// defaults and nothing more, because it deliberately does not include
// feature_extractor.hpp — the model has to stay usable from a planning tool
// with no DSP in it, and a `#include` that silently re-sized this report
// every time the DSP changed would be worse than a number with a date on it.
//
// Re-measure and pass the result as argv[1..4]. On 2026-10-14, on a 64-bit
// host, `size_report` read:
//
//                                        bytes/link   once/node   links fit
//     A  stock <56>                           69040           0           2
//     B  tightened                            52584           0           3
//     C  tightened + shared scratch           46184        6404           3
//     D  C, best_bnr compiled out             16880        6404           9
//
// against 171 KiB of free DIRAM. The `once/node` column is argv[4]: C and D
// moved the spectral scratch out of the link into one arena the whole node
// computes in, so it is paid once and not per link — which is also why it
// cannot simply be added to the per-link figure. The Xtensa figures are
// smaller (32-bit size_t and pointers) and are the ones to budget from.
//
// UNITS. The defaults below are the memory piece's round planning figures,
// quoted in that brief as "19.3 KB" and "46.2 KB" — decimal KB. This report
// prints KiB (/1024) because that is what the RAM budget is denominated in,
// so 46200 B shows as 45.1 KiB. The byte figure is the unambiguous one and
// is printed beside every KiB column in the assumptions block.

/// Per-link `FeatureExtractor` state with BNR computed — the default build.
/// Planning figure; `size_report` row C measured 46184 B on this host.
constexpr std::uint32_t kDefaultBytesPerLinkBnrOn = 46200;

/// Per-link state with the BNR row store compiled out. `wifisense/features.py`:
/// "best_bnr is optional (spec/wire.md): its row store is half of a node's
/// per-link memory, and no detector reads it. A memory-bound node sets
/// compute_bnr=False and reports 0.0". spec/wire.md agrees: `best_bnr` is 0
/// for "a memory-constrained node that does not compute it".
///
/// THIS IS THE LEVER THAT DECIDES MESH SIZE — see CALLOUT 3. Planning figure;
/// `size_report` row D measured 16880 B, i.e. better than this default.
constexpr std::uint32_t kDefaultBytesPerLinkBnrOff = 19300;

/// Internal DIRAM a node has left for DSP state after ESP-IDF, the WiFi stack
/// and the ESP-NOW buffers. Taken from `size_report.cpp`'s `kFreeDiramBytes`
/// so the two reports cannot disagree about the budget while disagreeing
/// about how many links fit in it. Still an estimate: the WiFi stack's
/// footprint is configuration-dependent and the firmware session owns the
/// real number.
constexpr std::uint32_t kDefaultRamBudgetBytes = 171u * 1024u;

/// What a node pays ONCE, however many links it runs: the shared spectral
/// scratch arena (`size_report`'s `once/node` column, 6404 B on a 64-bit host
/// for shapes C and D). `size_report` subtracts it from the budget before
/// dividing by the per-link figure, and so does this report — otherwise the
/// two disagree about how many links the same node holds. Pass 0 as argv[4]
/// to model a shape where each link owns its own scratch (rows A and B).
constexpr std::uint32_t kDefaultSharedOnceBytes = 6404;

struct Figures {
    std::uint32_t bnr_on      = kDefaultBytesPerLinkBnrOn;
    std::uint32_t bnr_off     = kDefaultBytesPerLinkBnrOff;
    std::uint32_t budget      = kDefaultRamBudgetBytes;
    std::uint32_t shared_once = kDefaultSharedOnceBytes;
};

double kib(std::uint32_t bytes) { return static_cast<double>(bytes) / 1024.0; }

const char* short_status(TdmaStatus s) {
    switch (s) {
        case TdmaStatus::Ok:                return "OK";
        case TdmaStatus::TooFewNodes:       return "no link";
        case TdmaStatus::TooManyNodes:      return "tx_id overflow";
        case TdmaStatus::BadTiming:         return "bad timing";
        case TdmaStatus::SlotOverrun:       return "SLOTS OVERLAP";
        case TdmaStatus::GuardTooNarrow:    return "DRIFT > GUARD";
        case TdmaStatus::RateBelowDspFloor: return "BELOW 10 Hz FLOOR";
    }
    return "?";
}

TdmaConfig make(std::uint16_t n, float slot_ms, LinkAssignment la) {
    TdmaConfig c;
    c.n_nodes         = n;
    c.slot_ms         = slot_ms;
    c.link_assignment = la;
    return c;
}

void rule(char c = '-') {
    for (int i = 0; i < 108; ++i) {
        std::putchar(c);
    }
    std::putchar('\n');
}

// ---------------------------------------------------------------------------
// ONE definition of "fits", used by every column, callout and wall below
// ---------------------------------------------------------------------------
//
// extractors * bytes_per_link + the once-per-node arena, which is what
// size_report's links_in_budget() computes from the other end. Every "does it
// fit?" in this file goes through these three functions, so a grid cell, a
// callout verdict and WALL 1 cannot answer differently about the same node —
// which is how the two reports came apart in the first place.

/// Per-node DSP RAM for a mesh of `n` under `la`. 0 for a shape the model
/// rejects (n < 2, or past the one-byte tx_id), which is "nothing to count".
///
/// The slot is a placeholder: the extractor count is a function of the shape
/// and the assignment alone, so RAM is a wall you hit at a mesh size whatever
/// the schedule does. That is also why this wall is reported separately from
/// the rate walls rather than as one verdict.
std::uint32_t node_bytes(std::uint16_t n, LinkAssignment la, std::uint32_t per_link,
                         const Figures& f) {
    return TdmaSchedule(make(n, 6.25f, la)).per_node_bytes(per_link, f.shared_once);
}

/// Note the `!= 0`: a rejected shape must not fit by virtue of costing nothing.
bool fits(std::uint16_t n, LinkAssignment la, std::uint32_t per_link, const Figures& f) {
    const std::uint32_t need = node_bytes(n, la, per_link, f);
    return need != 0u && need <= f.budget;
}

/// The largest mesh whose per-node RAM is inside the budget. Monotone in N, so
/// the scan may stop at the first miss. TWO ANSWERS NEED READING AS WORDS
/// RATHER THAN AS NODE COUNTS, and they are what this returns 0 and 255 for:
///   0   not even a 2-node mesh fits — there is no wall, the scheme is out
///   255 RAM never bound; the one-byte tx_id stopped the scan, not memory
/// `wall_text` below turns them into the sentence each deserves.
std::uint16_t last_n_that_fits(LinkAssignment la, std::uint32_t per_link, const Figures& f) {
    std::uint16_t last = 0;
    for (std::uint16_t n = 2; n <= saans::kMaxTdmaNodes; ++n) {
        if (!fits(n, la, per_link, f)) {
            break;
        }
        last = n;
    }
    return last;
}

const char* wall_text(char* buf, std::size_t cap, std::uint16_t wall) {
    if (wall == 0) {
        std::snprintf(buf, cap, "nowhere: not even N=2 fits");
    } else if (wall >= saans::kMaxTdmaNodes) {
        std::snprintf(buf, cap, "N=255 (tx_id, not RAM)");
    } else {
        std::snprintf(buf, cap, "N=%u", static_cast<unsigned>(wall));
    }
    return buf;
}

/// "1 link" / "4 links". A report that prints "1 links" invites the reader to
/// wonder what else was not read back.
const char* links_word(std::uint16_t n) { return (n == 1u) ? "link" : "links"; }

// ---------------------------------------------------------------------------
// Assumptions, printed every run
// ---------------------------------------------------------------------------
void print_assumptions(const Figures& f) {
    const TdmaConfig d;  // the documented defaults

    std::printf("\n");
    rule('=');
    std::printf("  SAANS BROADCAST-TDMA VIABILITY  —  SYNTHETIC, MODEL, NOT MEASURED\n");
    rule('=');
    std::printf(
        "  No hardware exists yet. Every number below follows from the assumptions in this\n"
        "  block; none of them has been observed on a radio. The scheme is SAANS_PLAN §5 /\n"
        "  CAMPUS_SENSING_PROMPT §5.1: N nodes, node 0 syncs, node k broadcasts in slot k,\n"
        "  every node hears every transmission, so each ordered link is sampled once per\n"
        "  superframe and per_link_rate = 1 / (N * slot).\n\n");

    std::printf("  ASSUMPTIONS AND WHERE THEY COME FROM\n");
    std::printf("    frame airtime      %6.2f ms   ESP-NOW broadcast, ~50 B payload at the\n",
                static_cast<double>(d.frame_airtime_ms));
    std::printf("                                   1 Mbps DSSS basic rate: 89 B on air =\n");
    std::printf("                                   712 us, + 192 us preamble, + 200 us\n");
    std::printf("                                   DIFS and mean backoff. At 11 Mbps it is\n");
    std::printf("                                   ~0.46 ms — the first lever to pull.\n");
    std::printf("    guard band         %6.2f ms   covers FreeRTOS scheduling jitter (the\n",
                static_cast<double>(d.guard_ms));
    std::printf("                                   dominant and least-known term), clock\n");
    std::printf("                                   skew, and airtime estimate error.\n");
    std::printf("    clock drift        %6.1f ppm  per node, one-sided; ESP32 40 MHz crystal\n",
                static_cast<double>(d.clock_drift_ppm));
    std::printf("                                   over temperature. Two nodes drift apart\n");
    std::printf("                                   at twice this.\n");
    std::printf("    resync interval    %6.2f s    how often a node re-disciplines to node 0.\n",
                static_cast<double>(d.resync_interval_s));
    std::printf("                                   Worst-case skew between syncs: %.1f us.\n",
                static_cast<double>(TdmaSchedule(d).worst_case_skew_s()) * 1e6);
    std::printf("    packet loss        %6.1f %%    DELIBERATELY ZERO: no measurement exists\n",
                static_cast<double>(d.packet_loss) * 100.0);
    std::printf("                                   for ESP-NOW broadcast loss on a campus\n");
    std::printf("                                   channel, and broadcasts are not retried.\n");
    std::printf("    DSP floor          %6.1f Hz   features.py REFUSES to construct below it\n",
                static_cast<double>(saans::kDspFloorHz));
    std::printf("                                   (the gross-motion band reaches 4 Hz).\n");
    std::printf("    rate target        %6.1f Hz   CAMPUS_SENSING_PROMPT §5.1.\n",
                static_cast<double>(saans::kTargetLinkRateHz));
    std::printf("    per-link state     %6u B    with BNR = %.1f KiB (argv[1]). The memory\n",
                static_cast<unsigned>(f.bnr_on), kib(f.bnr_on));
    std::printf("                                   piece's brief calls this '46.2 KB' —\n");
    std::printf("                                   decimal KB; this report prints KiB.\n");
    std::printf("                       %6u B    BNR row store compiled out = %.1f KiB\n",
                static_cast<unsigned>(f.bnr_off), kib(f.bnr_off));
    std::printf("                                   (argv[2]). No detector reads best_bnr and\n");
    std::printf("                                   spec/wire.md already defines 0 = unknown.\n");
    std::printf("    shared once/node   %6u B  = %.1f KiB (argv[4]): the spectral scratch\n",
                static_cast<unsigned>(f.shared_once), kib(f.shared_once));
    std::printf("                                   arena, paid ONCE however many links a\n");
    std::printf("                                   node runs. size_report's 'once/node'\n");
    std::printf("                                   column; every per-node figure below\n");
    std::printf("                                   includes it, as size_report's own\n");
    std::printf("                                   links_in_budget() does. 0 for a shape\n");
    std::printf("                                   whose links each own their scratch.\n");
    std::printf("    node RAM budget    %6u B  = %.1f KiB free DIRAM (argv[3]), the same\n",
                static_cast<unsigned>(f.budget), kib(f.budget));
    std::printf("                                   figure bench/size_report.cpp budgets\n");
    std::printf("                                   against. Still an estimate.\n");
    std::printf("    the authority      bench/size_report.cpp measures sizeof() on the real\n");
    std::printf("                                   instantiations. Re-run it and pass the\n");
    std::printf("                                   result as argv[1..4] — the fourth column\n");
    std::printf("                                   too, or this report will count more links\n");
    std::printf("                                   into the node than that one fits. The\n");
    std::printf("                                   defaults above are planning figures with\n");
    std::printf("                                   a date on them.\n");
    std::printf("\n");
    std::printf("  LINK RECIPROCITY — THE ASSUMPTION UNDER THE 'pair' COLUMNS\n");
    std::printf(
        "    a->b and b->a share the same Fresnel geometry, so they share the same blind\n"
        "    spots; the reverse direction adds receiver-noise diversity but no GEOMETRIC\n"
        "    diversity, which is what ADR-006's blind-spot strategy needs. Each unordered\n"
        "    pair is therefore extracted by one endpoint only — ceil((N-1)/2) extractors\n"
        "    instead of N-1, balanced around the ring so node 0 is not the one that fills.\n"
        "    THIS IS AN S6 ASSUMPTION TO CHECK ON HARDWARE, NOT A PROVEN FACT: CSI\n"
        "    amplitude on a->b also carries a's transmit chain and b's receive chain, and\n"
        "    those are not reciprocal. Build -DSAANS_LINK_ASSIGNMENT=0 to switch it off;\n"
        "    the 'all-ord' columns are what that costs.\n\n");
}

// ---------------------------------------------------------------------------
// The grid
// ---------------------------------------------------------------------------
void print_table(const Figures& f) {
    rule('=');
    std::printf("  THE GRID   (slot lengths chosen as 20 Hz at N=8, N=4 and N=2 respectively)\n");
    rule('=');
    std::printf(
        "    N  slot  superfr  per-link   aggregate  slot   ext/node   per-node KiB"
        "    rate       RAM\n");
    std::printf(
        "        (ms)     (s)      (Hz)  (frames/s)  util   all  pair    BNR+   BNR-"
        "   verdict    fits\n");
    rule();

    const std::uint16_t nodes[] = {2, 4, 6, 8, 12};
    const float         slots[] = {6.25f, 12.5f, 25.0f};

    for (float slot : slots) {
        for (std::uint16_t n : nodes) {
            const TdmaSchedule pair(make(n, slot, LinkAssignment::PairAssigned));
            const TdmaSchedule all(make(n, slot, LinkAssignment::AllOrderedLinks));

            const char* rate_verdict =
                !pair.is_valid()
                    ? (pair.status() == TdmaStatus::RateBelowDspFloor ? "<10Hz REJ"
                                                                      : short_status(pair.status()))
                    : (pair.meets_target() ? "20Hz+" : "<20Hz");
            // The RAM verdict is the binding one, so it gets its own column
            // rather than being left for the reader to infer from two KiB
            // figures and a budget printed forty lines earlier.
            const char* ram_verdict = fits(n, LinkAssignment::PairAssigned, f.bnr_on, f)
                                          ? "yes"
                                          : (fits(n, LinkAssignment::PairAssigned, f.bnr_off, f)
                                                 ? "BNR off"
                                                 : "NO");

            std::printf("  %3u  %5.2f  %7.4f  %8.3f  %10.1f  %4.0f%%  %4u  %4u  %6.1f %6.1f"
                        "  %-9s  %s\n",
                        static_cast<unsigned>(n), static_cast<double>(slot),
                        static_cast<double>(pair.superframe_s()),
                        static_cast<double>(pair.per_link_rate_hz()),
                        static_cast<double>(pair.aggregate_frames_per_s()),
                        static_cast<double>(pair.slot_utilisation()) * 100.0,
                        static_cast<unsigned>(all.extractors_per_node()),
                        static_cast<unsigned>(pair.extractors_per_node()),
                        kib(pair.per_node_bytes(f.bnr_on, f.shared_once)),
                        kib(pair.per_node_bytes(f.bnr_off, f.shared_once)),
                        rate_verdict, ram_verdict);
        }
        rule();
    }
    std::printf(
        "  rate verdict: 20Hz+ meets the target | <20Hz valid but below it | <10Hz REJ the\n"
        "  extractor refuses to configure | SLOTS OVERLAP, DRIFT > GUARD the two collision\n"
        "  modes. RAM fits: 'yes' with BNR on, 'BNR off' only with the row store compiled\n"
        "  out, 'NO' not at all — pair-assigned, against the %.1f KiB budget above.\n",
        kib(f.budget));

    std::printf(
        "  'aggregate' counts every ORDERED link, N*(N-1) of them, each sampled once per\n"
        "  superframe — the mesh-wide CSI frame rate. One node's own radio delivers\n"
        "  (N-1) * per-link; the channel carries 1/slot transmissions per second. Three\n"
        "  different numbers, and the published literature says 'aggregate' without saying\n"
        "  which, so all three are available from TdmaSchedule rather than only the\n"
        "  flattering one.\n");
    std::printf(
        "  'per-node KiB' is the PAIR-ASSIGNED count times the per-link state, PLUS the\n"
        "  %.1f KiB arena a node pays once (argv[4]) — the same sum size_report makes when\n"
        "  it asks how many links fit. Note that it is the WORST-CASE node: for even N half\n"
        "  the nodes carry one extractor fewer.\n\n",
        kib(f.shared_once));
}

// ---------------------------------------------------------------------------
// Callout 1 — the N=4 target
// ---------------------------------------------------------------------------
void print_n4_target(const Figures& f) {
    rule('=');
    std::printf("  CALLOUT 1 — N=4 AT >= 20 Hz PER LINK: WHICH ROWS MEET THE TARGET\n");
    rule('=');

    const TdmaConfig d;
    const float      occupied = d.frame_airtime_ms + d.guard_ms;
    const float      max_slot = saans::max_slot_ms_for_rate(4, saans::kTargetLinkRateHz);

    std::printf("  A 4-node room meets the 20 Hz target for any slot from %.2f ms (the\n",
                static_cast<double>(occupied));
    std::printf("  airtime+guard floor) up to %.2f ms (the rate ceiling, 1000/(4*20)).\n\n",
                static_cast<double>(max_slot));

    std::printf("    slot (ms)   per-link Hz   slot util   verdict\n");
    rule();
    const float probe[] = {1.50f, 2.10f, 5.00f, 6.25f, 12.50f, 12.51f, 25.00f};
    for (float s : probe) {
        const TdmaSchedule sch(make(4, s, LinkAssignment::PairAssigned));
        const char*        verdict =
            !sch.is_valid() ? short_status(sch.status())
                                   : (sch.meets_target() ? "MEETS THE 20 Hz TARGET"
                                                         : "valid, but below 20 Hz");
        std::printf("      %6.2f      %8.3f       %4.0f%%   %s\n", static_cast<double>(s),
                    static_cast<double>(sch.per_link_rate_hz()),
                    static_cast<double>(sch.slot_utilisation()) * 100.0, verdict);
    }
    rule();

    const TdmaSchedule chosen(make(4, 12.5f, LinkAssignment::PairAssigned));
    const bool         n4_on_fits  = fits(4, LinkAssignment::PairAssigned, f.bnr_on, f);
    const bool         n4_off_fits = fits(4, LinkAssignment::PairAssigned, f.bnr_off, f);
    std::printf(
        "  THE ROW TO BUILD: N=4, 12.50 ms slots. Exactly %.1f Hz per link, %.0f%% of each\n"
        "  slot used, %u extractors per node, %.1f KiB with BNR and %.1f KiB without (each\n"
        "  including the %.1f KiB node arena) against a %.0f KiB budget: %s. The whole N=4\n"
        "  band is valid, so the slot is a free choice between rate and channel occupancy.\n\n",
        static_cast<double>(chosen.per_link_rate_hz()),
        static_cast<double>(chosen.slot_utilisation()) * 100.0,
        static_cast<unsigned>(chosen.extractors_per_node()),
        kib(chosen.per_node_bytes(f.bnr_on, f.shared_once)),
        kib(chosen.per_node_bytes(f.bnr_off, f.shared_once)), kib(f.shared_once), kib(f.budget),
        n4_on_fits ? "both fit" : (n4_off_fits ? "ONLY THE BNR-OFF SHAPE FITS"
                                               : "NEITHER FITS — see CALLOUT 3"));
}

// ---------------------------------------------------------------------------
// Callout 2 — N=8 under pair assignment
// ---------------------------------------------------------------------------
void print_n8(const Figures& f) {
    rule('=');
    std::printf("  CALLOUT 2 — N=8 WITH PAIR ASSIGNMENT: THE CASE THE ASSUMPTION IS FOR\n");
    rule('=');

    const TdmaSchedule pair(make(8, 6.25f, LinkAssignment::PairAssigned));
    const TdmaSchedule all(make(8, 6.25f, LinkAssignment::AllOrderedLinks));

    std::printf("  8 nodes, 6.25 ms slots: %.1f Hz per link, %.0f%% slot utilisation, valid.\n\n",
                static_cast<double>(pair.per_link_rate_hz()),
                static_cast<double>(pair.slot_utilisation()) * 100.0);

    std::printf("    assignment      extractors/node   per-node KiB (BNR+)   (BNR-)   fits %.0f KiB?\n",
                kib(f.budget));
    rule();
    struct Row {
        const char*        name;
        const TdmaSchedule* s;
    };
    const Row rows[] = {{"all ordered", &all}, {"pair-assigned", &pair}};
    for (const Row& r : rows) {
        const std::uint32_t on  = r.s->per_node_bytes(f.bnr_on, f.shared_once);
        const std::uint32_t off = r.s->per_node_bytes(f.bnr_off, f.shared_once);
        std::printf("    %-14s %10u        %12.1f  %8.1f   %s / %s\n", r.name,
                    static_cast<unsigned>(r.s->extractors_per_node()), kib(on), kib(off),
                    on <= f.budget ? "yes" : "NO", off <= f.budget ? "yes" : "NO");
    }
    rule();

    std::printf(
        "  ceil((8-1)/2) = 4 extractors instead of 7. 28 pairs exist and 28 extractors run\n"
        "  across the mesh, four nodes carrying 4 and four carrying 3 — budget from the 4.\n\n");

    // The verdict is computed, not asserted in prose, so it cannot go stale
    // when the byte figures are re-measured and passed in as arguments.
    const bool pair_on_fits  = fits(8, LinkAssignment::PairAssigned, f.bnr_on, f);
    const bool pair_off_fits = fits(8, LinkAssignment::PairAssigned, f.bnr_off, f);
    const bool all_on_fits   = fits(8, LinkAssignment::AllOrderedLinks, f.bnr_on, f);

    std::printf("  THE VERDICT AT THE FIGURES ABOVE (%.1f KiB budget):\n", kib(f.budget));
    if (pair_on_fits) {
        std::printf(
            "    Pair assignment is enough: 8 nodes fit with BNR still on. Dropping the\n"
            "    assumption (%s) would need %.1f KiB and %s.\n\n",
            "all ordered links", kib(all.per_node_bytes(f.bnr_on, f.shared_once)),
            all_on_fits ? "would also fit" : "would NOT fit");
    } else if (pair_off_fits) {
        std::printf(
            "    PAIR ASSIGNMENT ALONE IS NOT ENOUGH. 4 links x %.1f KiB + the %.1f KiB node\n"
            "    arena = %.1f KiB is over the %.1f KiB budget. An 8-node room needs pair\n"
            "    assignment AND the BNR row store compiled out, which brings it to %.1f KiB.\n"
            "    Both levers, not one — and the BNR lever is the cheap one, since\n"
            "    spec/wire.md already defines best_bnr = 0 as 'unknown' and no detector\n"
            "    reads it.\n\n",
            kib(f.bnr_on), kib(f.shared_once), kib(pair.per_node_bytes(f.bnr_on, f.shared_once)),
            kib(f.budget), kib(pair.per_node_bytes(f.bnr_off, f.shared_once)));
    } else {
        std::printf(
            "    8 NODES DO NOT FIT AT ALL at these figures: %.1f KiB even with pair\n"
            "    assignment and BNR off, against a %.1f KiB budget.\n\n",
            kib(pair.per_node_bytes(f.bnr_off, f.shared_once)), kib(f.budget));
    }

    std::printf(
        "  WHAT PAIR ASSIGNMENT BUYS AND WHAT IT RISKS. It halves the per-node extractor\n"
        "  count; CALLOUT 3 turns that into a node count. It rests on reciprocity, which is\n"
        "  true of the CHANNEL and not of the TX and RX chains at each end; if one direction\n"
        "  of a pair turns out to see a breather the other misses, the saving was bought\n"
        "  with coverage. That is an S6 measurement (SAANS_PLAN §7: blind-spot rate with\n"
        "  1 vs 2 vs 3 links, sim then real), not something this model can settle.\n"
        "  -DSAANS_LINK_ASSIGNMENT=0 reverts it.\n\n");
}

// ---------------------------------------------------------------------------
// Callout 3 — where the scheme stops working
// ---------------------------------------------------------------------------
void print_limits(const Figures& f) {
    rule('=');
    std::printf("  CALLOUT 3 — WHERE THE SCHEME STOPS WORKING (THREE WALLS, IN ORDER)\n");
    rule('=');

    const TdmaConfig d;
    const float      occupied = d.frame_airtime_ms + d.guard_ms;

    // Wall 1: RAM. Both the per-link state and the once-per-node arena, via
    // the same `fits` every other section uses.
    const std::uint32_t per_link[2] = {f.bnr_on, f.bnr_off};
    const char*         label[2]    = {"BNR on ", "BNR off"};
    std::uint16_t       ram_wall[2] = {0, 0};
    for (int k = 0; k < 2; ++k) {
        ram_wall[k] = last_n_that_fits(LinkAssignment::PairAssigned, per_link[k], f);
    }

    std::printf("  WALL 1 — PER-NODE RAM, and it is the one that actually binds.\n");
    for (int k = 0; k < 2; ++k) {
        // THE TWO ENDS OF THE SCAN ARE NOT NODE COUNTS, and printing them as
        // if they were produces confident nonsense. A wall of 0 means nothing
        // fitted: "last N that fits is 0, N=1 needs 0 links and 0.0 KiB — over
        // budget" describes a config the model rejects as TooFewNodes, and
        // 0.0 KiB is not over any budget. A wall of 255 means RAM never bound
        // and the one-byte tx_id ended the scan, so the N=256 line underneath
        // would be a TooManyNodes config reported as a RAM overrun. Both are
        // reachable through exactly the interface argv exists for — a
        // re-measured, smaller free-DIRAM figure, or a node with PSRAM.
        if (ram_wall[k] == 0) {
            const TdmaSchedule two(make(2, 6.25f, LinkAssignment::PairAssigned));
            std::printf("    %s at %.1f KiB/link, pair-assigned: NOTHING FITS — not even a\n",
                        label[k], kib(per_link[k]));
            std::printf("            2-node mesh, which needs %u %s and %.1f KiB (%.1f + the\n",
                        static_cast<unsigned>(two.extractors_per_node()),
                        links_word(two.extractors_per_node()),
                        kib(two.per_node_bytes(per_link[k], f.shared_once)), kib(per_link[k]));
            std::printf("            %.1f KiB arena) against a %.1f KiB budget. There is no\n",
                        kib(f.shared_once), kib(f.budget));
            std::printf("            wall to report: at these figures the scheme is out.\n");
            continue;
        }
        const TdmaSchedule at(make(ram_wall[k], 6.25f, LinkAssignment::PairAssigned));
        std::printf("    %s at %.1f KiB/link, pair-assigned: last N that fits is %u "
                    "(%u %s, %.1f KiB);\n",
                    label[k], kib(per_link[k]), static_cast<unsigned>(ram_wall[k]),
                    static_cast<unsigned>(at.extractors_per_node()),
                    links_word(at.extractors_per_node()),
                    kib(at.per_node_bytes(per_link[k], f.shared_once)));
        if (ram_wall[k] >= saans::kMaxTdmaNodes) {
            std::printf("            and nothing larger was tried: RAM never binds at this\n");
            std::printf("            figure, the one-byte tx_id stops the mesh at 255 first\n");
            std::printf("            (spec/wire.md §2, 0xFF reserved).\n");
            continue;
        }
        const std::uint16_t over_n = static_cast<std::uint16_t>(ram_wall[k] + 1);
        const TdmaSchedule  over(make(over_n, 6.25f, LinkAssignment::PairAssigned));
        std::printf("            N=%u needs %u %s and %.1f KiB — over budget.\n",
                    static_cast<unsigned>(over_n),
                    static_cast<unsigned>(over.extractors_per_node()),
                    links_word(over.extractors_per_node()),
                    kib(over.per_node_bytes(per_link[k], f.shared_once)));
    }
    // Signed, because the two walls are in whatever order the arguments put
    // them: argv[1] and argv[2] swapped used to print a nine-digit node count
    // here, from a uint16 subtraction promoted to int and cast to unsigned.
    const int delta = static_cast<int>(ram_wall[1]) - static_cast<int>(ram_wall[0]);
    if (delta > 0) {
        std::printf("    Turning BNR off is worth %d extra nodes, and costs the gateway nothing\n",
                    delta);
        std::printf("    but `best_bnr`, which spec/wire.md already defines as 0 = unknown and\n");
        std::printf("    which no detector reads.\n\n");
    } else if (delta == 0) {
        if (ram_wall[0] >= saans::kMaxTdmaNodes) {
            std::printf("    Turning BNR off buys nothing here because RAM is not the wall at\n");
            std::printf("    all: both figures run into the 255-node tx_id ceiling first.\n\n");
        } else if (ram_wall[0] == 0) {
            std::printf("    Turning BNR off does not rescue this either: neither figure fits\n");
            std::printf("    a single link in the budget, so there is no lever left here —\n");
            std::printf("    the per-link state has to come down or the budget has to go up.\n\n");
        } else {
            std::printf("    Turning BNR off buys no extra nodes at these two figures — the\n");
            std::printf("    wall lands on the same N either way.\n\n");
        }
    } else {
        std::printf("    Turning BNR off would COST %d nodes here, which is not something that\n",
                    -delta);
        std::printf("    lever can do: argv[2] (%u B) is LARGER than argv[1] (%u B), so the two\n",
                    static_cast<unsigned>(f.bnr_off), static_cast<unsigned>(f.bnr_on));
        std::printf("    byte figures look swapped. Re-read the sign of this report.\n\n");
    }
    Figures no_arena     = f;
    no_arena.shared_once = 0;
    const std::uint16_t naive_wall =
        last_n_that_fits(LinkAssignment::PairAssigned, f.bnr_off, no_arena);
    char naive_buf[48];
    char real_buf[48];
    std::printf("    Every KiB figure above includes the %.1f KiB arena a node pays once, the\n",
                kib(f.shared_once));
    std::printf("    way bench/size_report.cpp budgets it: (free DIRAM - arena) / bytes per\n");
    if (naive_wall > ram_wall[1]) {
        std::printf("    link. Forget that term and the BNR-off wall reads %s where it is really\n",
                    wall_text(naive_buf, sizeof(naive_buf), naive_wall));
        std::printf("    %s — this report blessing a node that size_report, on the same\n",
                    wall_text(real_buf, sizeof(real_buf), ram_wall[1]));
        std::printf("    sizeof() and the same budget, says does not fit.\n\n");
    } else {
        // Computed from the shipped planning figures rather than written
        // down, so that "it is worth N nodes" cannot go stale when the
        // defaults are re-measured.
        const Figures def;
        Figures       def_no_arena     = def;
        def_no_arena.shared_once       = 0;
        const int     def_cost =
            static_cast<int>(last_n_that_fits(LinkAssignment::PairAssigned, def.bnr_off,
                                              def_no_arena)) -
            static_cast<int>(last_n_that_fits(LinkAssignment::PairAssigned, def.bnr_off, def));
        std::printf("    link. At these particular figures the arena does not move the wall —\n");
        std::printf("    it is still inside every KiB printed above, and at this report's own\n");
        std::printf("    defaults it is worth %d nodes.\n\n", def_cost);
    }

    // Wall 2: airtime.
    const std::uint16_t air_20 = saans::max_nodes_for_rate(saans::kTargetLinkRateHz, occupied);
    const std::uint16_t air_10 = saans::max_nodes_for_rate(saans::kDspFloorHz, occupied);
    std::printf("  WALL 2 — AIRTIME. A slot cannot be shorter than a transmission plus its\n");
    std::printf("    guard (%.2f ms at the defaults), so at 20 Hz per link the channel tops\n",
                static_cast<double>(occupied));
    std::printf("    out at N=%u, and at the 10 Hz DSP floor at N=%u. Both sit past wall 1 —\n",
                static_cast<unsigned>(air_20), static_cast<unsigned>(air_10));
    std::printf("    far past it with BNR on (%u vs %u), only just past it with BNR off\n",
                static_cast<unsigned>(air_20), static_cast<unsigned>(ram_wall[0]));
    std::printf("    (%u vs %u). AT REALISTIC MESH SIZES THE RADIO IS NOT THE CONSTRAINT.\n",
                static_cast<unsigned>(air_20), static_cast<unsigned>(ram_wall[1]));

    // And the PHY lever, computed rather than guessed at. It does NOT double
    // the node count: the guard band does not scale with the PHY rate, and at
    // 11 Mbps the guard is already the larger half of the slot floor, so most
    // of the airtime saving buys nothing.
    const float         fast_air   = 0.46f;  // 11 Mbps, from the airtime breakdown above
    const float         fast_floor = fast_air + d.guard_ms;
    const std::uint16_t fast_20 = saans::max_nodes_for_rate(saans::kTargetLinkRateHz, fast_floor);
    const std::uint16_t fast_10 = saans::max_nodes_for_rate(saans::kDspFloorHz, fast_floor);
    std::printf("    Pinning the PHY to 11 Mbps does not double that, which is the easy thing\n");
    std::printf("    to assume: the %.2f ms guard does not scale with the rate, so a %.2f ms\n",
                static_cast<double>(d.guard_ms), static_cast<double>(fast_air));
    std::printf("    airtime gives a %.2f ms slot floor and N=%u at 20 Hz (from %u) and N=%u at\n",
                static_cast<double>(fast_floor), static_cast<unsigned>(fast_20),
                static_cast<unsigned>(air_20), static_cast<unsigned>(fast_10));
    std::printf("    the 10 Hz floor (from %u) — about half as many again, not twice. And it\n",
                static_cast<unsigned>(air_10));
    std::printf("    changes nothing that matters, because wall 1 is still the one in front.\n\n");

    // Wall 3: the DSP floor at a fixed slot.
    std::printf("  WALL 3 — THE 10 Hz DSP FLOOR, for anyone who keeps the slot fixed and\n");
    std::printf("    adds nodes. The extractor does not degrade below it, it REFUSES to\n");
    std::printf("    configure (features.py raises; BasicFeatureExtractor::configure returns\n");
    std::printf("    false and the node stays inert).\n\n");
    std::printf("      slot (ms)   last N >= 20 Hz   last N >= 10 Hz   first N refused\n");
    rule();
    const float fixed[] = {6.25f, 12.5f, 25.0f};
    for (float slot : fixed) {
        std::uint16_t last20 = 0;
        std::uint16_t last10 = 0;
        std::uint16_t first_bad = 0;
        for (std::uint16_t n = 2; n <= 64; ++n) {
            const TdmaSchedule s(make(n, slot, LinkAssignment::PairAssigned));
            if (s.meets_target()) {
                last20 = n;
            }
            if (s.is_valid()) {
                last10 = n;
            } else if (first_bad == 0) {
                first_bad = n;
            }
        }
        std::printf("        %6.2f   %15u   %15u   %15u\n", static_cast<double>(slot),
                    static_cast<unsigned>(last20), static_cast<unsigned>(last10),
                    static_cast<unsigned>(first_bad));
    }
    rule();
    std::printf("\n");

    // The summary sentence this whole report exists to produce. Every figure
    // in it comes from the same `last_n_that_fits` as WALL 1 — including the
    // all-ordered pair, which used to be an open-coded `1 + budget/bytes` that
    // knew nothing about the arena, nothing about the 255-node tx_id ceiling
    // (it once printed N=435) and nothing about a budget too small for one
    // link (it printed N=1, which is not a mesh).
    char on_buf[48];
    char off_buf[48];
    char all_on_buf[48];
    char all_off_buf[48];
    const std::uint16_t all_on  = last_n_that_fits(LinkAssignment::AllOrderedLinks, f.bnr_on, f);
    const std::uint16_t all_off = last_n_that_fits(LinkAssignment::AllOrderedLinks, f.bnr_off, f);

    std::printf("  THE ANSWER THIS REPORT EXISTS TO GIVE\n");
    std::printf("    Against %.1f KiB of free DIRAM, with the %.1f KiB spectral arena paid\n",
                kib(f.budget), kib(f.shared_once));
    std::printf("    once per node, the mesh stops here:\n");
    std::printf("      pair-assigned, BNR on    %s\n",
                wall_text(on_buf, sizeof(on_buf), ram_wall[0]));
    std::printf("      pair-assigned, BNR off   %s\n",
                wall_text(off_buf, sizeof(off_buf), ram_wall[1]));
    std::printf("      all ordered links        %s with BNR on,\n",
                wall_text(all_on_buf, sizeof(all_on_buf), all_on));
    std::printf("                               %s with it off — what the mesh costs if\n",
                wall_text(all_off_buf, sizeof(all_off_buf), all_off));
    std::printf("                               reciprocity does not survive S6 and every\n");
    std::printf("                               node must run both directions of every pair.\n");
    if (ram_wall[0] != 0) {
        std::printf("    At the BNR-on wall a node holds %u links.\n",
                    static_cast<unsigned>(
                        TdmaSchedule(make(ram_wall[0], 6.25f, LinkAssignment::PairAssigned))
                            .extractors_per_node()));
    }
    std::printf("    The channel could carry %u nodes at 20 Hz and is nowhere near the\n",
                static_cast<unsigned>(air_20));
    std::printf("    limiting factor: RAM is, and `best_bnr` is the lever that moves it.\n\n");
}

// ---------------------------------------------------------------------------
// Callout 4 — the one real datapoint
// ---------------------------------------------------------------------------
void print_nine_node() {
    rule('=');
    std::printf("  CALLOUT 4 — THE ONE REAL DATAPOINT, AND THE MODEL REPRODUCING IT\n");
    rule('=');
    std::printf(
        "  RESEARCH.md §0: the one published multi-node ESP32 mesh study used backend-\n"
        "  orchestrated TDMA across 9 nodes, yielded ~1.4 Hz aggregate CSI, and at that\n"
        "  rate detected ambulatory motion ~100%% of the time but SEDENTARY PRESENCE ONLY\n"
        "  38%% OF THE TIME. A model that could not reach that regime would be too\n"
        "  optimistic to be worth running, so here it is — under all THREE readings of the\n"
        "  word 'aggregate', because the source does not define it and the three differ by\n"
        "  a factor of 72. Each row puts 1.4 in a different column; the third is the one\n"
        "  the grid above labels 'aggregate' (N*(N-1) ordered links), and it is also the\n"
        "  least flattering, which is why leaving it out would have been convenient.\n\n");

    std::printf("    1.4 Hz read as          slot (ms)   per-link Hz   mesh tx/s"
                "   aggregate   status\n");
    rule();
    struct Row {
        const char* name;
        float       slot_ms;
    };
    const Row rows[] = {{"per link", 79.4f},
                        {"transmissions/s", 714.3f},
                        {"frames/s, all 72 links", 5714.3f}};
    for (const Row& r : rows) {
        const TdmaSchedule s(make(9, r.slot_ms, LinkAssignment::PairAssigned));
        std::printf("    %-22s %9.1f   %11.4f   %9.4f   %9.4f   %s\n", r.name,
                    static_cast<double>(r.slot_ms),
                    static_cast<double>(s.per_link_rate_hz()),
                    static_cast<double>(s.mesh_tx_rate_hz()),
                    static_cast<double>(s.aggregate_frames_per_s()), short_status(s.status()));
    }
    rule();
    std::printf(
        "  All three are refused by name, which is the behaviour that would have been\n"
        "  useful to have in advance. The model also says what that mesh would have\n"
        "  NEEDED: a %.2f ms slot for 20 Hz per link, which the %.2f ms airtime+guard fits\n"
        "  inside comfortably. So nothing about 9 nodes forces 1.4 Hz — a slotted scheme at the\n"
        "  radio's own pace gets 20 Hz — and the published mesh was orchestrated from a\n"
        "  backend rather than slotted on the nodes.\n",
        static_cast<double>(saans::max_slot_ms_for_rate(9, saans::kTargetLinkRateHz)),
        static_cast<double>(TdmaConfig{}.frame_airtime_ms + TdmaConfig{}.guard_ms));
    std::printf(
        "  SAY NO MORE THAN THAT. RESEARCH.md §5 records that the verification panel\n"
        "  rejected 'sample rate is the demonstrated confound' 0-3: the 1.4 Hz and 38%%\n"
        "  numbers are real, the causal story is not established. This model shows the\n"
        "  regime is reachable and that the floor rejects it. It does not show why that\n"
        "  deployment was slow, and it is not evidence that ours would not be.\n\n");
}

void print_caveats() {
    rule('=');
    std::printf("  WHAT WOULD CHANGE THESE NUMBERS — THE S6 MEASUREMENT LIST\n");
    rule('=');
    std::printf(
        "  1. REAL AIRTIME. Measured, per PHY rate, with the CSI callback running. If it\n"
        "     is 2 ms rather than 1.1, wall 2 halves — still past wall 1, but the slot\n"
        "     floor at N=4 moves and the utilisation column is wrong by that factor.\n"
        "  2. REAL BROADCAST LOSS on a shared campus channel. The model carries loss as a\n"
        "     knob and refuses to invent a default. Loss above what SequenceGapFiller can\n"
        "     fill (max_gap, 1 s at the node's defaults) becomes a discontinuity, not a\n"
        "     lower rate, and nothing here models that transition.\n"
        "  3. REAL SCHEDULING JITTER. The guard band is sized by FreeRTOS jitter, which is\n"
        "     a guess. If it is milliseconds, every slot in the grid above is too short.\n"
        "  4. RECIPROCITY. Whether one direction of a pair sees what the other misses.\n"
        "     This is the assumption with the largest consequence and the least evidence.\n"
        "  5. REAL FREE RAM on a node running WiFi, ESP-NOW, the DSP and the uplink.\n"
        "     Wall 1 is the binding one, so this is the figure most worth measuring first.\n\n");
    std::printf(
        "  Until those exist, this is arithmetic over assumptions — useful for choosing\n"
        "  what to build and worthless as a claim about what a mesh does.\n");
    rule('=');
    std::printf("\n");
}

}  // namespace

int main(int argc, char** argv) {
    Figures f;
    if (argc > 1) {
        f.bnr_on = static_cast<std::uint32_t>(std::strtoul(argv[1], nullptr, 10));
    }
    if (argc > 2) {
        f.bnr_off = static_cast<std::uint32_t>(std::strtoul(argv[2], nullptr, 10));
    }
    if (argc > 3) {
        f.budget = static_cast<std::uint32_t>(std::strtoul(argv[3], nullptr, 10));
    }
    // argv[4] is the only figure allowed to be zero: shapes A and B give every
    // link its own scratch and a node then pays nothing once.
    if (argc > 4) {
        f.shared_once = static_cast<std::uint32_t>(std::strtoul(argv[4], nullptr, 10));
    }
    if (f.bnr_on == 0 || f.bnr_off == 0 || f.budget == 0) {
        std::fprintf(stderr,
                     "tdma_report: the per-link and budget figures must be non-zero\n"
                     "  usage: tdma_report [bnr_on_bytes] [bnr_off_bytes] [ram_budget_bytes]"
                     " [shared_once_bytes]\n"
                     "  from bench/size_report.cpp: bytes/link for row C, bytes/link for row"
                     " D,\n"
                     "  its free-DIRAM budget, and its 'once/node' column.\n");
        return 2;
    }
    if (f.shared_once >= f.budget) {
        std::fprintf(stderr,
                     "tdma_report: the once-per-node arena (%u B) is the whole budget (%u B)"
                     " or more;\n"
                     "  no link of any size fits and every wall below would read 'none'."
                     " Check argv[3] and argv[4].\n",
                     static_cast<unsigned>(f.shared_once), static_cast<unsigned>(f.budget));
        return 2;
    }

    print_assumptions(f);
    print_table(f);
    print_n4_target(f);
    print_n8(f);
    print_limits(f);
    print_nine_node();
    print_caveats();
    return 0;
}
