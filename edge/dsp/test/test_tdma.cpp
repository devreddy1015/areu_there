// SPDX-License-Identifier: MIT
//
// Saans edge DSP — property tests for the broadcast-TDMA schedule model.
//
// ---------------------------------------------------------------------------
// WHY THERE IS NO GOLDEN VECTOR HERE, AND WHY THIS FILE STILL NEEDS THE
// VECTOR DIRECTORY
// ---------------------------------------------------------------------------
// Every other test in this directory checks the float32 port against
// `wifisense/`. `tdma.hpp` has no counterpart there: it models a radio
// schedule, not a DSP stage, and there is nothing in the Python reference for
// it to disagree with. Generating a vector for it would produce a file that
// proves only that `generate.py` and this header were written by the same
// person on the same afternoon.
//
// So these are self-contained property tests: closed-form identities
// (per_link_rate == 1/(N*slot)), combinatorial invariants (every unordered
// pair owned exactly once), and the named rejections the model owes its
// caller.
//
// `make guard` nevertheless points every test in TESTS at an empty directory
// and requires a NON-ZERO exit, on the principle (golden.hpp) that a parity
// test which silently passes because it could not find its data is the worst
// possible outcome. A test that happily passed with no vectors present would
// be the single hole in that guard — and a hole that grows, because the next
// vector-free test would copy this one. This file therefore REQUIRES the
// vector directory to be the real one before it runs a single check. It reads
// nothing out of it. It simply refuses to be run against a directory the rest
// of the suite could not use either, so that a mis-pointed harness fails in
// seven places out of seven.
//
// ---------------------------------------------------------------------------
// WHAT THESE TESTS ARE EVIDENCE ABOUT
// ---------------------------------------------------------------------------
// That the arithmetic in `tdma.hpp` is the arithmetic it claims, and that its
// rejections fire for the stated reason. They are NOT evidence that an ESP32
// mesh behaves this way — see the SYNTHETIC banner at the top of the header.
// The one place reality enters is `the 9-node regime`, which pins the model
// against the single published measurement this project has (RESEARCH.md §0:
// ~1.4 Hz aggregate CSI, sedentary presence detected 38% of the time). A
// model that could not produce that regime would be too optimistic to use for
// the decision it exists to inform.
//
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <initializer_list>
#include <string>
#include <vector>

#include "golden.hpp"
#include "tdma.hpp"

using saans::golden::Report;
using saans::LinkAssignment;
using saans::TdmaConfig;
using saans::TdmaSchedule;
using saans::TdmaStatus;

namespace {

std::string fmt(const char* f, double a, double b = 0.0, double c = 0.0) {
    char buf[512];
    std::snprintf(buf, sizeof(buf), f, a, b, c);
    return std::string(buf);
}

/// See the header comment: this test owns no vectors but must still fail when
/// the vector directory is wrong, or it becomes the hole in `make guard`.
void require_vector_dir() {
    const std::string manifest = saans::golden::dir() + "/MANIFEST";
    std::FILE*        fh       = std::fopen(manifest.c_str(), "r");
    if (fh == nullptr) {
        saans::golden::fatal(
            "test_tdma reads no vectors, but refuses to run against a vector directory the rest "
            "of the suite could not use (no MANIFEST at %s)",
            manifest.c_str());
    }
    std::fclose(fh);
}

TdmaConfig base_config() {
    TdmaConfig c;  // documented defaults: 4 nodes, 12.5 ms slots -> 20 Hz
    return c;
}

// ---------------------------------------------------------------------------
// 1. per_link_rate_hz == 1 / (N * slot), and exactly so
// ---------------------------------------------------------------------------
//
// These nine cases are not spot checks of a formula, they are a check that the
// formula is evaluated the way the header says it is. Every (N, slot) below
// has N*slot_ms equal to a round number of milliseconds, so the real-arithmetic
// answer is a number float32 holds exactly, and the assertion is EXACT
// EQUALITY rather than a tolerance.
//
// That matters because the obvious implementation — seconds first, then
// reciprocal — does not agree with this one everywhere. The case in this list
// that separates them is N=2 at a 10 ms slot: the seconds-first form returns
// 49.9999962 where the answer is exactly 50, and a `>= 50.0f` check is then
// false for a reason that has nothing to do with the schedule. (NOT the N=4,
// 12.5 ms default, which an earlier version of this comment named: there the
// seconds-first error is 3e-7, inside the half-ulp at 20 Hz, and it rounds
// back to exactly 20.0f. The trap is real; the default is not where it bites,
// and a maintainer who checked the claimed example would have concluded there
// was no trap at all.) A tolerance here would hide exactly the bug the test
// exists to catch.
void check_rate_formula(Report& rep) {
    rep.begin("per-link rate is 1/(N*slot), exactly", "(no vector)");

    struct Case {
        std::uint16_t n;
        float         slot_ms;
        float         expect_hz;
    };
    const Case cases[] = {
        {2, 25.0f, 20.0f},  {4, 12.5f, 20.0f}, {5, 10.0f, 20.0f},
        {8, 6.25f, 20.0f},  {4, 6.25f, 40.0f}, {2, 10.0f, 50.0f},
        {4, 25.0f, 10.0f},  {10, 10.0f, 10.0f}, {16, 12.5f, 5.0f},
    };

    for (const Case& c : cases) {
        TdmaConfig cfg = base_config();
        cfg.n_nodes    = c.n;
        cfg.slot_ms    = c.slot_ms;
        const TdmaSchedule s(cfg);
        char               what[64];
        std::snprintf(what, sizeof(what), "N=%u, slot %g ms", static_cast<unsigned>(c.n),
                      static_cast<double>(c.slot_ms));
        rep.truth(what, s.per_link_rate_hz() == c.expect_hz,
                  fmt("%.9g Hz, want %.9g exactly", static_cast<double>(s.per_link_rate_hz()),
                      static_cast<double>(c.expect_hz)));
    }

    // superframe and rate are reciprocals of each other. Tied together here
    // because they are computed from different expressions on purpose (see the
    // header): nothing else would notice if one of them drifted.
    for (const Case& c : cases) {
        TdmaConfig cfg = base_config();
        cfg.n_nodes    = c.n;
        cfg.slot_ms    = c.slot_ms;
        const TdmaSchedule s(cfg);
        const double       product =
            static_cast<double>(s.superframe_s()) * static_cast<double>(s.per_link_rate_hz());
        char what[80];
        std::snprintf(what, sizeof(what), "N=%u/%g ms: superframe x rate == 1",
                      static_cast<unsigned>(c.n), static_cast<double>(c.slot_ms));
        rep.close(what, product, 1.0, 1e-6);
    }

    // The three other rates the report prints, on one configuration, so that
    // "aggregate" can never be quoted without knowing which of them it is.
    TdmaConfig cfg = base_config();  // N=4 at 20 Hz per link
    const TdmaSchedule s(cfg);
    rep.truth("mesh transmissions/s == 1/slot", s.mesh_tx_rate_hz() == 80.0f,
              fmt("%.9g Hz", static_cast<double>(s.mesh_tx_rate_hz())));
    rep.truth("CSI frames into one node == (N-1) * per-link",
              s.csi_frames_per_node_s() == 60.0f,
              fmt("%.9g Hz", static_cast<double>(s.csi_frames_per_node_s())));
    rep.truth("aggregate == N*(N-1) * per-link", s.aggregate_frames_per_s() == 240.0f,
              fmt("%.9g frames/s over %.0f ordered links",
                  static_cast<double>(s.aggregate_frames_per_s()),
                  static_cast<double>(s.ordered_link_count())));
}

// ---------------------------------------------------------------------------
// 2. The N=4 target: which slot lengths deliver >= 20 Hz per link
// ---------------------------------------------------------------------------
//
// The band is found by scanning rather than asserted from the closed form,
// because the question the plan asks ("what slot do I configure?") is answered
// by the predicate `meets_target()`, not by the formula — and the two could
// disagree through the validity checks without the formula noticing.
//
// Both edges are then pinned exactly, from the two different things that set
// them: the upper edge is the rate target (1000/(4*20) = 12.5 ms) and the
// lower edge is the airtime, because below airtime+guard the slots overlap.
void check_target_band_n4(Report& rep) {
    rep.begin("N=4: the slot range that delivers >= 20 Hz per link", "(no vector)");

    const TdmaConfig defaults = base_config();
    const float      occupied = defaults.frame_airtime_ms + defaults.guard_ms;

    // Scan on a 0.01 ms grid from 0.5 ms to 30 ms. `entries` counts how many
    // times the predicate turns on: a contiguous band turns on exactly once,
    // and anything else means a hole somewhere in the middle of the range —
    // which would be a far more interesting bug than a misplaced edge.
    int  lo_hundredths = -1;
    int  hi_hundredths = -1;
    int  entries       = 0;
    bool was_in        = false;
    for (int i = 50; i <= 3000; ++i) {
        TdmaConfig cfg = defaults;
        cfg.slot_ms    = static_cast<float>(i) / 100.0f;
        const bool in  = TdmaSchedule(cfg).meets_target();
        if (in && !was_in) {
            ++entries;
        }
        if (in) {
            if (lo_hundredths < 0) {
                lo_hundredths = i;
            }
            hi_hundredths = i;
        }
        was_in = in;
    }

    rep.truth("the band is a single contiguous interval", entries == 1,
              fmt("%.0f separate interval(s) over 0.50-30.00 ms",
                  static_cast<double>(entries)));

    // The expected lower edge is the first grid point that is not shorter
    // than airtime+guard, found with the same float comparison the model
    // makes. Asserting "within 0.01 ms of 2.1" instead would be a coin flip:
    // 1.1f + 1.0f lands on a tie between the two floats either side of 2.1
    // and round-half-to-even picks the lower one, so the grid point at 2.10
    // is exactly equal rather than just below. That is the FPU's business,
    // not the schedule's, and a test should not depend on it.
    int expected_lo = -1;
    for (int i = 50; i <= 3000 && expected_lo < 0; ++i) {
        if (static_cast<float>(i) / 100.0f >= occupied) {
            expected_lo = i;
        }
    }
    rep.equal("lower edge is the first slot that holds airtime+guard", lo_hundredths,
              expected_lo);
    rep.close_abs("lower edge (ms)", lo_hundredths / 100.0, static_cast<double>(occupied), 0.011);
    rep.close_abs("upper edge (ms)", hi_hundredths / 100.0, 12.5, 0.01);

    // The upper edge is the rate target and nothing else.
    TdmaConfig at_edge = defaults;
    at_edge.slot_ms    = 12.5f;
    const TdmaSchedule edge(at_edge);
    rep.truth("slot 12.50 ms meets the 20 Hz target", edge.meets_target(),
              fmt("%.9g Hz per link, status ok=%.0f",
                  static_cast<double>(edge.per_link_rate_hz()),
                  static_cast<double>(edge.is_valid() ? 1 : 0)));

    TdmaConfig past_edge = defaults;
    past_edge.slot_ms    = 12.51f;
    const TdmaSchedule past(past_edge);
    rep.truth("slot 12.51 ms does not, but is still valid",
              !past.meets_target() && past.is_valid(),
              fmt("%.9g Hz per link", static_cast<double>(past.per_link_rate_hz())));

    // The lower edge is the airtime and nothing else. At exactly
    // airtime+guard the slot is full to the millisecond and still legal.
    TdmaConfig full = defaults;
    full.slot_ms    = occupied;
    const TdmaSchedule exact(full);
    rep.truth("slot == airtime+guard is full but valid",
              exact.is_valid() && !exact.slot_overruns(),
              fmt("utilisation %.6f, status=%.0f (0=ok)",
                  static_cast<double>(exact.slot_utilisation()),
                  static_cast<double>(static_cast<int>(exact.status()))));
    rep.close("utilisation at that slot is 1.0", static_cast<double>(exact.slot_utilisation()),
              1.0, 1e-6);

    // And the closed form agrees with the scan.
    rep.truth("max_slot_ms_for_rate(4, 20) == 12.5",
              saans::max_slot_ms_for_rate(4, 20.0f) == 12.5f,
              fmt("%.9g ms", static_cast<double>(saans::max_slot_ms_for_rate(4, 20.0f))));
}

// ---------------------------------------------------------------------------
// 3. Pair assignment — the claim the whole RAM budget rests on
// ---------------------------------------------------------------------------
//
// Three properties, each of which would be a real bug if it failed:
//
//   EXACTLY ONCE. A pair owned twice wastes the RAM the scheme exists to
//   save; a pair owned ZERO times is a link nobody extracts — a silent hole
//   in the coverage the gateway is fusing over, and the kind of bug that
//   looks like a blind spot in the physics.
//
//   ceil((N-1)/2) WORST CASE. This is the number the firmware budgets from.
//
//   BALANCED TO WITHIN ONE. Without this, "the lower id owns the pair" is
//   technically correct and useless: node 0 would own all N-1 pairs and need
//   the RAM of the unassigned scheme anyway.
void check_pair_assignment(Report& rep) {
    rep.begin("pair assignment: once each, balanced, ceil((N-1)/2) worst case", "(no vector)");

    int bad_once     = 0;
    int bad_worst    = 0;
    int bad_balance  = 0;
    int bad_total    = 0;
    int bad_symmetry = 0;
    std::string first_failure;

    for (std::uint16_t n = 2; n <= 32; ++n) {
        TdmaConfig cfg       = base_config();
        cfg.n_nodes          = n;
        cfg.slot_ms          = 1.0f;  // irrelevant to assignment; keeps the object cheap
        cfg.link_assignment  = LinkAssignment::PairAssigned;
        const TdmaSchedule s(cfg);

        // Every unordered pair owned by exactly one of its endpoints.
        for (std::uint16_t a = 0; a < n; ++a) {
            for (std::uint16_t b = static_cast<std::uint16_t>(a + 1); b < n; ++b) {
                const int owners = (s.owns_pair(a, b) ? 1 : 0) + (s.owns_pair(b, a) ? 1 : 0);
                if (owners != 1) {
                    ++bad_once;
                    if (first_failure.empty()) {
                        first_failure = "N=" + std::to_string(n) + " pair {" +
                                        std::to_string(a) + "," + std::to_string(b) + "} has " +
                                        std::to_string(owners) + " owners";
                    }
                }
            }
            // A node never owns a pair with itself, and never one with a node
            // outside the mesh.
            if (s.owns_pair(a, a) || s.owns_pair(a, n)) {
                ++bad_symmetry;
            }
        }

        std::uint32_t  total = 0;
        std::uint16_t  mx    = 0;
        std::uint16_t  mn    = 0xFFFFu;
        for (std::uint16_t i = 0; i < n; ++i) {
            const std::uint16_t c = s.extractors_for_node(i);
            total += c;
            mx = (c > mx) ? c : mx;
            mn = (c < mn) ? c : mn;
        }

        const std::uint16_t want_worst = static_cast<std::uint16_t>(n / 2u);  // ceil((N-1)/2)
        if (mx != want_worst || s.extractors_per_node() != want_worst) {
            ++bad_worst;
        }
        if (static_cast<int>(mx) - static_cast<int>(mn) > 1) {
            ++bad_balance;
        }
        if (total != s.pair_count()) {
            ++bad_total;
        }
    }

    rep.truth("every unordered pair owned exactly once", bad_once == 0,
              bad_once == 0 ? std::string("N = 2..32, 5456 pairs in all") : first_failure);
    rep.truth("a node owns no self-pair and no out-of-mesh pair", bad_symmetry == 0,
              fmt("%.0f violation(s)", static_cast<double>(bad_symmetry)));
    rep.truth("extractors_per_node == ceil((N-1)/2) == max over nodes", bad_worst == 0,
              fmt("%.0f N(s) disagreed over 2..32", static_cast<double>(bad_worst)));
    rep.truth("per-node counts differ by at most one", bad_balance == 0,
              fmt("%.0f N(s) were unbalanced", static_cast<double>(bad_balance)));
    rep.truth("owned pairs sum to N(N-1)/2", bad_total == 0,
              fmt("%.0f N(s) lost or duplicated a pair", static_cast<double>(bad_total)));

    // The headline case from the brief, spelled out because it is the number
    // that decides whether an 8-node room fits in an S3.
    TdmaConfig eight       = base_config();
    eight.n_nodes          = 8;
    eight.slot_ms          = 6.25f;
    eight.link_assignment  = LinkAssignment::PairAssigned;
    const TdmaSchedule e8(eight);
    rep.equal("N=8 pair-assigned: extractors per node", e8.extractors_per_node(), 4);
    rep.equal("N=8 pair-assigned: pairs in the mesh", e8.pair_count(), 28);
    std::uint32_t e8_total = 0;
    for (std::uint16_t i = 0; i < 8; ++i) {
        e8_total += e8.extractors_for_node(i);
    }
    rep.equal("N=8 pair-assigned: extractors running anywhere", e8_total, 28);
    rep.note("N=8: four nodes own 4 pairs and four own 3 — 28 total, not 8*4=32. "
             "Budget from extractors_per_node; count work with extractors_for_node.");

    // Node 0 is not special. The naive "lower id owns" rule would give it 7.
    rep.equal("N=8: node 0 owns no more than any other", e8.extractors_for_node(0), 4);
    rep.equal("N=8: node 7 is not left idle", e8.extractors_for_node(7), 3);
}

// ---------------------------------------------------------------------------
// 4. The assumption switched off
// ---------------------------------------------------------------------------
void check_all_ordered_links(Report& rep) {
    rep.begin("SAANS_LINK_ASSIGNMENT=0: every ordered link, no assumption", "(no vector)");

    for (int n_i : {2, 4, 8, 12}) {
        const std::uint16_t n = static_cast<std::uint16_t>(n_i);
        TdmaConfig cfg      = base_config();
        cfg.n_nodes         = n;
        cfg.slot_ms         = 1.0f;
        cfg.link_assignment = LinkAssignment::AllOrderedLinks;
        const TdmaSchedule s(cfg);

        bool all_owned = true;
        for (std::uint16_t a = 0; a < n && all_owned; ++a) {
            for (std::uint16_t b = 0; b < n; ++b) {
                if (a != b && !s.owns_pair(a, b)) {
                    all_owned = false;
                    break;
                }
            }
        }
        char what[64];
        std::snprintf(what, sizeof(what), "N=%u: every node runs all N-1 links",
                      static_cast<unsigned>(n));
        rep.truth(what,
                  all_owned && s.extractors_per_node() == static_cast<std::uint16_t>(n - 1),
                  fmt("extractors_per_node %.0f, want %.0f",
                      static_cast<double>(s.extractors_per_node()),
                      static_cast<double>(n_i - 1)));
    }

    // Twice the extractors is twice the RAM: that is the price of dropping
    // the reciprocity assumption, and it is the number S6 has to weigh
    // against whatever coverage the second direction turns out to buy.
    TdmaConfig pair_cfg      = base_config();
    pair_cfg.n_nodes         = 8;
    pair_cfg.link_assignment = LinkAssignment::PairAssigned;
    TdmaConfig all_cfg       = pair_cfg;
    all_cfg.link_assignment  = LinkAssignment::AllOrderedLinks;
    const std::uint32_t kBytesPerLink = 46200;  // 46.2 KB, see bench/tdma_report.cpp
    const std::uint32_t pair_bytes    = TdmaSchedule(pair_cfg).per_node_bytes(kBytesPerLink);
    const std::uint32_t all_bytes     = TdmaSchedule(all_cfg).per_node_bytes(kBytesPerLink);
    rep.truth("N=8: dropping the assumption costs 7/4 of the RAM",
              all_bytes * 4u == pair_bytes * 7u,
              fmt("%.0f bytes pair-assigned vs %.0f all-ordered",
                  static_cast<double>(pair_bytes), static_cast<double>(all_bytes)));

    // The build's default must be what SAANS_LINK_ASSIGNMENT says it is.
    rep.truth("TdmaConfig default follows SAANS_LINK_ASSIGNMENT",
              base_config().link_assignment ==
                  ((SAANS_LINK_ASSIGNMENT != 0) ? LinkAssignment::PairAssigned
                                                : LinkAssignment::AllOrderedLinks),
              fmt("SAANS_LINK_ASSIGNMENT=%.0f", static_cast<double>(SAANS_LINK_ASSIGNMENT)));
}

// ---------------------------------------------------------------------------
// 5. A slot that cannot hold its own transmission is REJECTED, by name
// ---------------------------------------------------------------------------
//
// Silently accepting this is the dangerous failure: the model would report a
// per-link rate that the channel physically cannot deliver, because two nodes
// would be transmitting at once and neither frame would arrive.
void check_slot_overrun(Report& rep) {
    rep.begin("airtime + guard > slot is rejected as SlotOverrun", "(no vector)");

    TdmaConfig cfg       = base_config();
    cfg.n_nodes          = 4;
    cfg.slot_ms          = 1.5f;  // airtime 1.1 + guard 1.0 = 2.1 > 1.5
    const TdmaSchedule s(cfg);

    rep.truth("rejected", !s.is_valid(), s.status_text());
    rep.equal("named reason is SlotOverrun", static_cast<long long>(s.status()),
              static_cast<long long>(TdmaStatus::SlotOverrun));
    rep.truth("utilisation says so too", s.slot_utilisation() > 1.0f,
              fmt("%.4f of the slot", static_cast<double>(s.slot_utilisation())));
    rep.truth("the rate itself was fine", s.per_link_rate_hz() > saans::kDspFloorHz,
              fmt("%.4f Hz per link — fast, and uncollectable",
                  static_cast<double>(s.per_link_rate_hz())));

    // The same slot with the airtime pretended away is "valid". That is the
    // belief this parameter exists to stop: a schedule model that treats a
    // transmission as instantaneous will happily approve any slot length at
    // all, and the first thing that goes wrong on hardware is the thing it
    // declined to model.
    TdmaConfig fantasy       = cfg;
    fantasy.frame_airtime_ms = 0.0f;
    rep.truth("pretend the airtime away and the same slot passes",
              TdmaSchedule(fantasy).is_valid(),
              "which is why frame_airtime_ms is a documented parameter and not 0");

    // A zero guard band is rejected too, for the OTHER reason — drift. The
    // two collision modes are independent and both have to be modelled.
    TdmaConfig no_guard = base_config();
    no_guard.guard_ms   = 0.0f;
    rep.equal("a zero guard band is GuardTooNarrow, not SlotOverrun",
              static_cast<long long>(TdmaSchedule(no_guard).status()),
              static_cast<long long>(TdmaStatus::GuardTooNarrow));

    // Ordering: when a configuration is both overrun AND too slow, the
    // actionable complaint wins.
    TdmaConfig both   = base_config();
    both.n_nodes      = 200;
    both.slot_ms      = 1.0f;  // 200 * 1 ms = 5 Hz per link, and 2.1 > 1.0
    const TdmaSchedule b(both);
    rep.truth("overrun is reported ahead of a rate failure that also holds",
              b.status() == TdmaStatus::SlotOverrun && b.rate_below_floor(),
              fmt("status=%.0f, rate %.4f Hz", static_cast<double>(static_cast<int>(b.status())),
                  static_cast<double>(b.per_link_rate_hz())));

    // Shape failures come first of all.
    TdmaConfig one = base_config();
    one.n_nodes    = 1;
    rep.equal("1 node is TooFewNodes", static_cast<long long>(TdmaSchedule(one).status()),
              static_cast<long long>(TdmaStatus::TooFewNodes));
    TdmaConfig huge = base_config();
    huge.n_nodes    = 256;
    rep.equal("256 nodes is TooManyNodes (tx_id is one byte, 0xFF reserved)",
              static_cast<long long>(TdmaSchedule(huge).status()),
              static_cast<long long>(TdmaStatus::TooManyNodes));
    TdmaConfig nan_cfg = base_config();
    nan_cfg.slot_ms    = std::nanf("");
    rep.equal("a NaN slot is BadTiming, not a rate failure",
              static_cast<long long>(TdmaSchedule(nan_cfg).status()),
              static_cast<long long>(TdmaStatus::BadTiming));
    TdmaConfig zero_slot = base_config();
    zero_slot.slot_ms    = 0.0f;
    rep.equal("a zero slot is BadTiming", static_cast<long long>(TdmaSchedule(zero_slot).status()),
              static_cast<long long>(TdmaStatus::BadTiming));
}

// ---------------------------------------------------------------------------
// 5b. A refused schedule must not answer two ways at once
// ---------------------------------------------------------------------------
//
// `per_node_bytes` is what a planning tool calls to ask "does this mesh fit?",
// and 0 is the one answer that reads as "yes, easily". So the accessors that
// describe a node's links have to agree with each other about EVERY config,
// including the ones the model refuses — and they have to split the right way:
//
//   a TIMING failure leaves the link set perfectly well defined. Fix the slot
//   and the same mesh runs, so "this mesh needs four extractors a node" is
//   still true and still worth budgeting from.
//
//   a SHAPE failure means there is no mesh. Fewer than two nodes has no link;
//   more than 255 cannot be addressed by `tx_id` at all. Nothing to count, and
//   the count is 0 everywhere rather than 0 in one accessor and 128 in another.
void check_refused_configs_agree(Report& rep) {
    rep.begin("a refused schedule answers one way, not two", "(no vector)");

    TdmaConfig bad_time = base_config();
    bad_time.n_nodes    = 8;
    bad_time.slot_ms    = -5.0f;
    const TdmaSchedule bt(bad_time);
    rep.equal("a negative slot is BadTiming", static_cast<long long>(bt.status()),
              static_cast<long long>(TdmaStatus::BadTiming));
    std::uint16_t bt_max = 0;
    for (std::uint16_t i = 0; i < 8; ++i) {
        const std::uint16_t c = bt.extractors_for_node(i);
        bt_max                = (c > bt_max) ? c : bt_max;
    }
    rep.equal("the link set survives it: still ceil((8-1)/2) per node",
              bt.extractors_per_node(), 4);
    rep.equal("and extractors_for_node agrees with it", bt_max, 4);
    rep.truth("so per_node_bytes is a figure, not a 0 that reads as 'it fits'",
              bt.per_node_bytes(46200, 6404) == 4u * 46200u + 6404u,
              fmt("%.0f B", static_cast<double>(bt.per_node_bytes(46200, 6404))));

    // The other half: a shape the wire cannot carry.
    TdmaConfig too_many = base_config();
    too_many.n_nodes    = 256;
    const TdmaSchedule tm(too_many);
    rep.equal("256 nodes is TooManyNodes", static_cast<long long>(tm.status()),
              static_cast<long long>(TdmaStatus::TooManyNodes));
    rep.truth("and NOTHING counts links in a mesh that cannot be addressed",
              !tm.owns_pair(0, 1) && tm.extractors_for_node(0) == 0 &&
                  tm.extractors_per_node() == 0,
              fmt("owns=%.0f for_node=%.0f per_node=%.0f",
                  static_cast<double>(tm.owns_pair(0, 1) ? 1 : 0),
                  static_cast<double>(tm.extractors_for_node(0)),
                  static_cast<double>(tm.extractors_per_node())));
    rep.equal("per_node_bytes is 0 there — nothing to count, NOT headroom",
              tm.per_node_bytes(46200, 6404), 0);

    TdmaConfig lone = base_config();
    lone.n_nodes    = 1;
    const TdmaSchedule one(lone);
    rep.truth("a 1-node mesh is the same story from the other end",
              !one.owns_pair(0, 1) && one.extractors_for_node(0) == 0 &&
                  one.per_node_bytes(46200, 6404) == 0,
              one.status_text());
    rep.note("0 bytes means 'there is no mesh to count', never 'it fits'. "
             "bench/tdma_report.cpp checks the shape before it prints a wall; so should "
             "any other caller.");
}

// ---------------------------------------------------------------------------
// 6. Clock drift — the number the guard band is buying
// ---------------------------------------------------------------------------
//
// The two figures asserted here are the ones quoted in the header, so that
// the prose and the code cannot drift apart: tens of microseconds over a
// second, milliseconds over a minute, at +/-20 ppm between two nodes.
void check_drift(Report& rep) {
    rep.begin("clock drift between sync beacons", "(no vector)");

    TdmaConfig one_s      = base_config();
    one_s.clock_drift_ppm = 20.0f;
    one_s.resync_interval_s = 1.0f;
    const TdmaSchedule a(one_s);
    rep.close_abs("20 ppm over 1 s is 40 us",
                  static_cast<double>(a.worst_case_skew_s()) * 1e6, 40.0, 1e-3);
    rep.truth("and 40 us fits inside a 1 ms guard", a.is_valid() && !a.guard_too_narrow(),
              fmt("guard %.3f ms, skew %.6f ms", static_cast<double>(one_s.guard_ms),
                  static_cast<double>(a.worst_case_skew_s()) * 1e3));

    TdmaConfig minute       = one_s;
    minute.resync_interval_s = 60.0f;
    const TdmaSchedule m(minute);
    rep.close_abs("20 ppm over 60 s is 2.4 ms",
                  static_cast<double>(m.worst_case_skew_s()) * 1e3, 2.4, 1e-5);
    rep.truth("and 2.4 ms does not: rejected", !m.is_valid(), m.status_text());
    rep.equal("named reason is GuardTooNarrow", static_cast<long long>(m.status()),
              static_cast<long long>(TdmaStatus::GuardTooNarrow));
    rep.truth("nothing else was wrong with it",
              !m.slot_overruns() && !m.rate_below_floor(),
              fmt("%.4f Hz per link, utilisation %.4f",
                  static_cast<double>(m.per_link_rate_hz()),
                  static_cast<double>(m.slot_utilisation())));

    // The factor of 2 is the definition, not conservatism: one node fast by
    // ppm and the other slow by ppm.
    TdmaConfig doubled       = one_s;
    doubled.clock_drift_ppm  = 40.0f;
    rep.truth("skew is linear in ppm and in the resync interval",
              TdmaSchedule(doubled).worst_case_skew_s() == 2.0f * a.worst_case_skew_s(),
              fmt("%.9g us at 40 ppm vs %.9g us at 20 ppm",
                  static_cast<double>(TdmaSchedule(doubled).worst_case_skew_s()) * 1e6,
                  static_cast<double>(a.worst_case_skew_s()) * 1e6));

    // Where the guard band runs out at the default 1 ms: guard / (2*ppm),
    // which is 25 s at 20 ppm. The two cases straddle it rather than sitting
    // on it — at exactly 25 s the comparison is a float tie and which side it
    // lands on is a property of the rounding, not of the schedule, so pinning
    // it would be testing the FPU.
    TdmaConfig stretch        = one_s;
    stretch.resync_interval_s = 24.0f;
    rep.truth("1 ms of guard buys ~25 s between syncs at 20 ppm (24 s fits)",
              TdmaSchedule(stretch).is_valid(),
              fmt("skew %.6f ms vs guard %.3f ms",
                  static_cast<double>(TdmaSchedule(stretch).worst_case_skew_s()) * 1e3,
                  static_cast<double>(stretch.guard_ms)));
    TdmaConfig past        = one_s;
    past.resync_interval_s = 26.0f;
    rep.equal("26 s does not", static_cast<long long>(TdmaSchedule(past).status()),
              static_cast<long long>(TdmaStatus::GuardTooNarrow));
}

// ---------------------------------------------------------------------------
// 7. The 9-node regime — the one real measurement this model is held to
// ---------------------------------------------------------------------------
//
// RESEARCH.md §0: a 9-node ESP32 mesh with backend-orchestrated TDMA achieved
// ~1.4 Hz aggregate CSI and detected sedentary presence only 38% of the time.
//
// The published figure says "aggregate" without defining it, so all three
// readings are checked — and there are three, not two, because the word has a
// third meaning that is this model's own: the rate summed over every ORDERED
// link, which is what `aggregate_frames_per_s()` returns and what
// `bench/tdma_report.cpp`'s grid column of that name prints.
//
//   per link                 ~79 ms slots, 1.4 Hz on each link
//   transmissions/s        ~714 ms slots, 1.4 frames on the channel per second
//   over all N(N-1) links ~5714 ms slots, 1.4 frames/s across 72 links
//
// A factor of 72 separates the first from the last, so "1.4 Hz aggregate"
// without a definition is not one claim but three. ALL THREE land far below
// the 10 Hz floor, which is the useful conclusion: the regime is unusable for
// breathing under every interpretation, and the model says so by name rather
// than printing a small number and leaving the reader to judge it. Checking
// only the flattering reading would have been the easy mistake.
void check_nine_node_regime(Report& rep) {
    rep.begin("the 9-node, ~1.4 Hz regime (RESEARCH.md §0)", "(no vector)");

    TdmaConfig per_link = base_config();
    per_link.n_nodes    = 9;
    per_link.slot_ms    = 79.4f;
    const TdmaSchedule p(per_link);
    rep.close_abs("per-link reading: 9 nodes, 79.4 ms slots -> ~1.4 Hz per link",
                  static_cast<double>(p.per_link_rate_hz()), 1.4, 0.01);
    rep.equal("and the model refuses it, by name", static_cast<long long>(p.status()),
              static_cast<long long>(TdmaStatus::RateBelowDspFloor));
    rep.note("superframe %.3f s — a breathing cycle is 3-6 s, so a window holds "
             "a handful of samples per breath at best.",
             static_cast<double>(p.superframe_s()));

    TdmaConfig mesh_wide = base_config();
    mesh_wide.n_nodes    = 9;
    mesh_wide.slot_ms    = 714.3f;  // 1/1.4 s per transmission
    const TdmaSchedule mw(mesh_wide);
    rep.close_abs("mesh-wide reading: 1.4 transmissions/s in total",
                  static_cast<double>(mw.mesh_tx_rate_hz()), 1.4, 0.01);
    rep.close_abs("which is 0.156 Hz per link", static_cast<double>(mw.per_link_rate_hz()),
                  0.1556, 0.001);
    rep.equal("also refused", static_cast<long long>(mw.status()),
              static_cast<long long>(TdmaStatus::RateBelowDspFloor));

    // The third reading, which is the quantity this model itself calls
    // `aggregate`: 1.4 frames/s summed over all N(N-1) = 72 ordered links.
    // The least flattering of the three — 0.019 Hz per link — and therefore
    // the one a report is most tempted to leave out.
    TdmaConfig all_links = base_config();
    all_links.n_nodes    = 9;
    all_links.slot_ms    = 5714.3f;  // 1.4 / (9*8) Hz on each link
    const TdmaSchedule al(all_links);
    rep.close_abs("all-ordered-links reading: 1.4 frames/s over N*(N-1) = 72 links",
                  static_cast<double>(al.aggregate_frames_per_s()), 1.4, 0.01);
    rep.close_abs("which is 0.0194 Hz on each of them",
                  static_cast<double>(al.per_link_rate_hz()), 0.019444, 1e-4);
    rep.equal("refused as well — all three readings are", static_cast<long long>(al.status()),
              static_cast<long long>(TdmaStatus::RateBelowDspFloor));
    rep.note("72x separates the first reading from the third. bench/tdma_report.cpp "
             "CALLOUT 4 prints all three side by side for that reason: its grid column "
             "named 'aggregate' is this last one.");

    // The model must also be able to say what that mesh WOULD have needed.
    const float needed = saans::max_slot_ms_for_rate(9, saans::kTargetLinkRateHz);
    rep.close_abs("9 nodes at 20 Hz per link needs a 5.56 ms slot",
                  static_cast<double>(needed), 5.5555556, 1e-4);
    TdmaConfig would_work = base_config();
    would_work.n_nodes    = 9;
    would_work.slot_ms    = needed;
    rep.truth("which the 1.1 ms airtime and 1 ms guard do fit inside",
              TdmaSchedule(would_work).meets_target(),
              fmt("utilisation %.4f at a %.4f ms slot",
                  static_cast<double>(TdmaSchedule(would_work).slot_utilisation()),
                  static_cast<double>(needed)));
    rep.note("SYNTHETIC: the model says the radio could have carried 9 nodes at 20 Hz. "
             "It says nothing about why the published mesh did not — RESEARCH.md §5 "
             "records that the panel rejected 'sample rate is the demonstrated cause' 0-3.");
}

// ---------------------------------------------------------------------------
// 8. slot_for is a bijection over the node ids
// ---------------------------------------------------------------------------
//
// Today it is the identity (node k transmits in slot k). The test is written
// against the PROPERTY and not the identity, because the day someone permutes
// the slots to separate two nodes that interfere, what must survive is "every
// node has exactly one slot and no slot is shared".
void check_slot_bijection(Report& rep) {
    rep.begin("slot_for is a bijection over 0..N-1", "(no vector)");

    int bad_cover = 0;
    int bad_range = 0;
    int bad_time  = 0;
    for (int n_i : {2, 3, 4, 8, 9, 12, 64, 255}) {
        const std::uint16_t n = static_cast<std::uint16_t>(n_i);
        TdmaConfig cfg = base_config();
        cfg.n_nodes    = n;
        cfg.slot_ms    = 2.5f;
        const TdmaSchedule s(cfg);

        std::vector<int> seen(n, 0);
        for (std::uint16_t id = 0; id < n; ++id) {
            const std::uint16_t slot = s.slot_for(id);
            if (slot >= n) {
                ++bad_range;
                continue;
            }
            ++seen[slot];
            // The transmit instant must be that slot's start, and must land
            // inside the superframe.
            const float t    = s.transmit_offset_s(id);
            const float want = static_cast<float>(slot) * cfg.slot_ms * 0.001f;
            if (t != want || t < 0.0f || t >= s.superframe_s()) {
                ++bad_time;
            }
        }
        for (int c : seen) {
            if (c != 1) {
                ++bad_cover;
            }
        }
        // An id outside the mesh has no slot and no transmit instant.
        if (s.slot_for(n) != TdmaSchedule::kNoSlot || s.transmit_offset_s(n) >= 0.0f) {
            ++bad_range;
        }
    }

    rep.truth("every slot used exactly once", bad_cover == 0,
              fmt("%.0f slot(s) unused or shared", static_cast<double>(bad_cover)));
    rep.truth("every slot index inside the superframe, out-of-mesh ids rejected",
              bad_range == 0, fmt("%.0f violation(s)", static_cast<double>(bad_range)));
    rep.truth("transmit instant == slot * slot_ms, inside the superframe", bad_time == 0,
              fmt("%.0f violation(s)", static_cast<double>(bad_time)));

    // Node 0 starts the superframe: that is where the sync beacon is.
    const TdmaSchedule d(base_config());
    rep.truth("node 0 transmits at t=0 (the sync beacon)", d.transmit_offset_s(0) == 0.0f,
              fmt("%.9g s", static_cast<double>(d.transmit_offset_s(0))));
    rep.close_abs("node 3 of 4 transmits at 37.5 ms",
                  static_cast<double>(d.transmit_offset_s(3)) * 1e3, 37.5, 1e-4);
}

// ---------------------------------------------------------------------------
// 9. Loss, effective rate, and the permille the wire carries
// ---------------------------------------------------------------------------
void check_loss_and_memory(Report& rep) {
    rep.begin("effective rate, loss permille, per-node memory", "(no vector)");

    TdmaConfig cfg  = base_config();  // 20 Hz per link
    cfg.packet_loss = 0.05f;
    const saans::TdmaLinkRate r = TdmaSchedule(cfg).effective_rate();
    rep.truth("nominal is untouched by loss", r.nominal_hz == 20.0f,
              fmt("%.9g Hz", static_cast<double>(r.nominal_hz)));
    rep.close("5% loss leaves 19 Hz effective", static_cast<double>(r.effective_hz), 19.0, 1e-6);
    rep.equal("loss_permille is 50", r.loss_permille, 50);

    TdmaConfig lossless = base_config();
    rep.equal("no loss is 0 permille", TdmaSchedule(lossless).effective_rate().loss_permille, 0);
    TdmaConfig total  = base_config();
    total.packet_loss = 1.0f;
    const saans::TdmaLinkRate t = TdmaSchedule(total).effective_rate();
    rep.equal("total loss is 1000 permille", t.loss_permille, 1000);
    rep.truth("and 0 Hz effective", t.effective_hz == 0.0f,
              fmt("%.9g Hz", static_cast<double>(t.effective_hz)));

    // Out-of-range loss is clamped, not propagated: a negative loss must not
    // manufacture a rate higher than the schedule allocates.
    TdmaConfig silly  = base_config();
    silly.packet_loss = -0.5f;
    rep.truth("a negative packet_loss cannot inflate the rate",
              TdmaSchedule(silly).effective_rate().effective_hz == 20.0f,
              fmt("%.9g Hz", static_cast<double>(TdmaSchedule(silly).effective_rate().effective_hz)));
    silly.packet_loss = 2.0f;
    rep.equal("a loss above 1 clamps to 1000 permille",
              TdmaSchedule(silly).effective_rate().loss_permille, 1000);

    // Loss is NOT a validity failure — the gap filler owns that — but the
    // effective rate crossing the floor is still reportable.
    TdmaConfig heavy  = base_config();
    heavy.packet_loss = 0.6f;
    const TdmaSchedule h(heavy);
    rep.truth("60% loss: schedule still valid, effective rate below the floor",
              h.is_valid() && !h.effective_rate_meets_floor(),
              fmt("%.4f Hz nominal, %.4f Hz effective, floor %.1f",
                  static_cast<double>(h.per_link_rate_hz()),
                  static_cast<double>(h.effective_rate().effective_hz),
                  static_cast<double>(saans::kDspFloorHz)));

    // Memory scales with the worst-case extractor count and nothing else.
    TdmaConfig mem      = base_config();
    mem.n_nodes         = 8;
    mem.slot_ms         = 6.25f;
    mem.link_assignment = LinkAssignment::PairAssigned;
    const TdmaSchedule m(mem);
    rep.equal("N=8 pair-assigned at 46200 B/link (46.2 KB)", m.per_node_bytes(46200),
              4u * 46200u);
    rep.equal("the same mesh with BNR off at 19300 B/link (19.3 KB)", m.per_node_bytes(19300),
              4u * 19300u);
    rep.equal("zero bytes per link is zero", m.per_node_bytes(0), 0);

    // THE ONCE-PER-NODE TERM, AND WHY IT IS NOT COSMETIC.
    //
    // `bench/size_report.cpp` asks the same question from the other end:
    // links_in_budget(per_link, shared_once) = (free_diram - shared_once) /
    // per_link, because the node build moved the spectral scratch out of the
    // link into ONE arena the whole node computes in. Fold that arena into the
    // per-link figure and it is charged N times; leave it out and it is
    // charged never. Either way the two reports describe different nodes —
    // which is what happened: at row D's 16880 B/link against 171 KiB,
    // tdma_report blessed 10 links where size_report fitted 9.
    const std::uint32_t kArena  = 6404;        // size_report's `once/node`, shapes C and D
    const std::uint32_t kRowD   = 16880;       // its row D, best_bnr compiled out
    const std::uint32_t kBudget = 171u * 1024u;

    rep.equal("the arena is added ONCE, not once per link",
              m.per_node_bytes(46200, kArena), 4u * 46200u + kArena);
    rep.equal("and it defaults to zero, so a shape with no shared state pays none",
              m.per_node_bytes(46200), 4u * 46200u);

    // The cross-check that catches the disagreement, stated as the two meshes
    // either side of the wall rather than as a node count — the count moves
    // with sizeof(), the inequality does not.
    TdmaConfig nineteen      = base_config();
    nineteen.n_nodes         = 19;
    nineteen.slot_ms         = 6.25f;
    nineteen.link_assignment = LinkAssignment::PairAssigned;
    TdmaConfig twenty        = nineteen;
    twenty.n_nodes           = 20;
    rep.truth("N=19 at row D fits 171 KiB: 9 links plus the arena",
              TdmaSchedule(nineteen).per_node_bytes(kRowD, kArena) <= kBudget,
              fmt("%.0f B of %.0f", static_cast<double>(
                                        TdmaSchedule(nineteen).per_node_bytes(kRowD, kArena)),
                  static_cast<double>(kBudget)));
    rep.truth("N=20 does not: the tenth link is exactly what size_report's 9 means",
              TdmaSchedule(twenty).per_node_bytes(kRowD, kArena) > kBudget,
              fmt("%.0f B of %.0f", static_cast<double>(
                                        TdmaSchedule(twenty).per_node_bytes(kRowD, kArena)),
                  static_cast<double>(kBudget)));
    rep.equal("which is links_in_budget's own arithmetic: (171 KiB - arena) / row D",
              (kBudget - kArena) / kRowD, 9);
    rep.note("Drop the arena and the same budget divides into 10 links and N=21 — the "
             "defect this pins. bench/tdma_report.cpp takes it as argv[4].");

    // A budget comparison must never be reached through a wrapped product: a
    // wrapped figure is small, and small reads as "fits".
    rep.equal("an oversized per-link figure saturates rather than wrapping",
              m.per_node_bytes(0xFFFFFFFFu, 0xFFFFFFFFu), 0xFFFFFFFFu);
}

// ---------------------------------------------------------------------------
// 10. The inverses — "how many nodes fit?" asked directly
// ---------------------------------------------------------------------------
void check_inverses(Report& rep) {
    rep.begin("inverting the schedule: slots and node counts", "(no vector)");

    // ROUND-TRIP: the slot the inverse hands back must actually deliver the
    // rate it was asked for, or the model refuses a configuration it
    // recommended itself one line earlier.
    //
    // THE RANGE AND THE TARGETS ARE BOTH THE POINT. This loop used to stop at
    // N=40 and test one target, and the bare `1000/(N*target)` — whose
    // quotient can round UP by one ulp — first falls short at N=45 at 20 Hz:
    // five past the end of the scan. A property test that stops just short of
    // its first counterexample is indistinguishable from one that passes. So
    // the range is now everything `tx_id` can address, and the targets include
    // 1.4 Hz, the rate of the one published datapoint, where the first failure
    // is at N=3 and no range would have saved a single-target scan.
    //
    // The rate is compared directly rather than through `meets_target()`: at
    // these N the slot is far below airtime+guard and the schedule is rightly
    // rejected for collisions. What is under test is the inverse, not the
    // channel.
    struct Target {
        const char* name;
        float       hz;
    };
    const Target targets[] = {{"20 Hz (the plan's target)", saans::kTargetLinkRateHz},
                              {"15 Hz", 15.0f},
                              {"10 Hz (the DSP floor)", saans::kDspFloorHz},
                              {"1.4 Hz (the published datapoint)", 1.4f}};
    for (const Target& t : targets) {
        int bad   = 0;
        int first = 0;
        for (std::uint16_t n = 2; n <= saans::kMaxTdmaNodes; ++n) {
            TdmaConfig cfg = base_config();
            cfg.n_nodes    = n;
            cfg.slot_ms    = saans::max_slot_ms_for_rate(n, t.hz);
            if (TdmaSchedule(cfg).per_link_rate_hz() < t.hz) {
                if (bad == 0) {
                    first = n;
                }
                ++bad;
            }
        }
        char what[112];
        std::snprintf(what, sizeof(what), "max_slot_ms_for_rate round-trips, N=2..255 at %s",
                      t.name);
        rep.truth(what, bad == 0,
                  bad == 0 ? std::string("254 N(s), every slot delivers its target")
                           : ("fell short for " + std::to_string(bad) + " N(s), first at N=" +
                              std::to_string(first)));
    }

    // The named counterexample, pinned on its own so that a regression here
    // reads as the bug it is rather than as "some N in a range of 254".
    TdmaConfig n45 = base_config();
    n45.n_nodes    = 45;
    n45.slot_ms    = saans::max_slot_ms_for_rate(45, saans::kTargetLinkRateHz);
    rep.truth("N=45 at 20 Hz: the slot the inverse recommends is one the model accepts",
              TdmaSchedule(n45).per_link_rate_hz() >= saans::kTargetLinkRateHz,
              fmt("%.9g ms gives %.9g Hz, want >= 20", static_cast<double>(n45.slot_ms),
                  static_cast<double>(TdmaSchedule(n45).per_link_rate_hz())));

    const TdmaConfig d        = base_config();
    const float      min_slot = d.frame_airtime_ms + d.guard_ms;
    rep.equal("at 20 Hz and a 2.1 ms floor, 23 nodes fit on the channel",
              saans::max_nodes_for_rate(saans::kTargetLinkRateHz, min_slot), 23);
    rep.equal("at the 10 Hz DSP floor, 47 do",
              saans::max_nodes_for_rate(saans::kDspFloorHz, min_slot), 47);
    rep.note("The CHANNEL stops at 23 nodes. Per-node RAM stops an order of magnitude "
             "sooner: at ~46 KB/link and ceil((N-1)/2) extractors a node runs out of "
             "DIRAM at single-digit N. The radio is not the binding constraint. No node "
             "count is pinned here because it moves with sizeof() — "
             "bench/tdma_report.cpp CALLOUT 3 computes it from bench/size_report.cpp.");

    rep.truth("a 24-node mesh at 20 Hz needs a slot shorter than one airtime",
              saans::max_slot_ms_for_rate(24, saans::kTargetLinkRateHz) < min_slot,
              fmt("%.4f ms needed vs %.4f ms of airtime+guard",
                  static_cast<double>(saans::max_slot_ms_for_rate(24, saans::kTargetLinkRateHz)),
                  static_cast<double>(min_slot)));
    rep.equal("nonsense in, zero out (target 0 Hz)",
              static_cast<long long>(saans::max_nodes_for_rate(0.0f, min_slot)), 0);
    rep.truth("nonsense in, zero out (1 node)", saans::max_slot_ms_for_rate(1, 20.0f) == 0.0f,
              "a one-node mesh has no link to rate");
}

}  // namespace

int main(int argc, char** argv) {
    saans::golden::dir_from_argv(argc, argv);
    require_vector_dir();

    Report rep;
    std::printf("\n");
    std::printf("################################################################\n");
    std::printf("##  tdma: SYNTHETIC - MODEL, NOT MEASURED                     ##\n");
    std::printf("##  No radio has been involved. These are property tests of    ##\n");
    std::printf("##  the arithmetic in dsp/tdma.hpp, not evidence about an      ##\n");
    std::printf("##  ESP32 mesh. See the header of tdma.hpp.                    ##\n");
    std::printf("################################################################\n");

    check_rate_formula(rep);
    check_target_band_n4(rep);
    check_pair_assignment(rep);
    check_all_ordered_links(rep);
    check_slot_overrun(rep);
    check_refused_configs_agree(rep);
    check_drift(rep);
    check_nine_node_regime(rep);
    check_slot_bijection(rep);
    check_loss_and_memory(rep);
    check_inverses(rep);

    return rep.finish("tdma");
}
