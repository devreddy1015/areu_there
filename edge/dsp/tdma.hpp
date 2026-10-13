// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — broadcast-TDMA schedule model for an N-node mesh (S6 prep).
//
// ===========================================================================
// SYNTHETIC — MODEL, NOT MEASURED
// ===========================================================================
// Nothing in this file has touched a radio. It is arithmetic over a scheme
// described in `SAANS_PLAN.md` §5 and `CAMPUS_SENSING_PROMPT.md` §5.1, built
// so that the question "how many nodes can a space carry before the per-link
// CSI rate falls below what the DSP needs?" has a defensible answer BEFORE
// anyone buys twelve boards. Every number it prints is a consequence of the
// parameters handed to it; the parameters themselves are estimates, and each
// default below says where it came from and how wrong it could be.
//
// When hardware exists (S5/S6), the figures to replace are `frame_airtime_ms`
// and `packet_loss` — both are properties of a real channel that no amount of
// modelling settles — and the reciprocity assumption discussed below.
//
// ---------------------------------------------------------------------------
// THE SCHEME
// ---------------------------------------------------------------------------
//   * N nodes share one WiFi channel. Node 0 broadcasts a sync frame at the
//     start of each superframe.
//   * A superframe is N slots of `slot_ms`; node k transmits in slot k.
//   * ESP-NOW broadcast means every node hears every other transmission, so
//     one transmission yields N-1 links and each ORDERED link (tx, rx) is
//     sampled exactly once per superframe:
//
//         per_link_rate_hz = 1 / (N * slot_s)
//
//   * One FeatureFrame per link per frame interval, `tx_id` = the
//     transmitter. Fusion stays at the GATEWAY (ADR-006); nodes do not fuse
//     links on-node, so this model never counts a fusion cost.
//
// ---------------------------------------------------------------------------
// LINK RECIPROCITY — THE ASSUMPTION THIS MODEL IS BUILT ON, AND ITS RISK
// ---------------------------------------------------------------------------
// Links a->b and b->a traverse the same physical paths. Their Fresnel
// geometry — and therefore their blind spots — are identical. ADR-006 picks
// link diversity as the blind-spot strategy precisely because a second link
// from ANOTHER ANGLE is usually not blind at the same moment; a second link
// along the SAME line is not another angle. The reverse direction buys
// receiver-noise diversity, which is not what blind-spot mitigation needs.
//
// So `LinkAssignment::PairAssigned` has each UNORDERED pair extracted by one
// endpoint only. An 8-node mesh then needs ceil((8-1)/2) = 4 extractors per
// node instead of 7 — at ~46 KB of per-link state that is the difference
// between fitting in an S3's RAM and not.
//
// WHERE THIS COULD BE WRONG, stated plainly because someone will decide to
// trust it: CSI amplitude on link a->b is the product of the channel with
// a's TRANSMIT chain response and b's RECEIVE chain response. The channel is
// reciprocal; the chains are not. Two directions of the same pair therefore
// differ by a (slowly varying) gain and phase that no calibration here
// removes, and the two directions also see different AGC states, different
// noise figures and different per-packet gain decisions. If, on real
// hardware, one direction of a pair turns out to see a breather that the
// other misses — and the honest position is that nobody here has measured it
// — then the saving above is bought with real coverage and the assumption
// must be withdrawn.
//
// That is why it is a COMPILE-TIME SWITCH and not a hard-coded constant:
// build with -DSAANS_LINK_ASSIGNMENT=0 and every node runs all N-1 ordered
// links again. S6's exit gate (SAANS_PLAN §7: "blind-spot rate with 1 vs 2 vs
// 3 links measured, sim then real") is the experiment that settles it.
//
// ---------------------------------------------------------------------------
// THE ONE REAL DATAPOINT, AND WHY THIS MODEL MUST BE ABLE TO LOOK THIS BAD
// ---------------------------------------------------------------------------
// `RESEARCH.md` §0: the one published multi-node ESP32 mesh study used
// backend-orchestrated TDMA across 9 nodes, yielded ~1.4 Hz aggregate CSI,
// and at that rate detected ambulatory motion ~100% of the time but
// **sedentary presence only 38%**. (RESEARCH.md §5 also records that the
// panel rejected the causal attribution 0-3 — the 1.4 Hz and 38% numbers are
// real, "sample rate is the demonstrated cause" is not established. The
// numbers are what this model has to be able to produce; the causality is
// not what it claims.)
//
// A model that cannot reach that regime is too optimistic to be worth
// running, so `test_tdma.cpp` pins it: N=9 with `slot_ms = 79.4` gives
// 1.399 Hz per link and `TdmaStatus::RateBelowDspFloor`. The model does not
// merely reproduce the number, it REFUSES the configuration — which is the
// behaviour that would have been useful to have in advance.
//
// ---------------------------------------------------------------------------
// WHAT THIS FILE IS NOT
// ---------------------------------------------------------------------------
// No ESP-IDF, no I/O, no state. It is a pure function of `TdmaConfig`, so it
// is testable on the host and can also sit in firmware to answer "is the
// schedule I was configured with actually runnable?" at boot. It obeys the
// same ADR-003 rules as the rest of `edge/dsp`: header-only, no heap, float32
// (the S3 FPU is single precision), `-fno-exceptions` so a bad configuration
// is a NAMED STATUS rather than a throw.
//
#include <cmath>
#include <cstddef>
#include <cstdint>

// Which link set a node extracts. 1 = pair-assigned (assume reciprocity,
// ceil((N-1)/2) extractors per node); 0 = every ordered link (no assumption,
// N-1 extractors per node). See the reciprocity section above: this exists so
// the assumption can be switched off on hardware without editing code.
#ifndef SAANS_LINK_ASSIGNMENT
#define SAANS_LINK_ASSIGNMENT 1
#endif
static_assert(SAANS_LINK_ASSIGNMENT == 0 || SAANS_LINK_ASSIGNMENT == 1,
              "SAANS_LINK_ASSIGNMENT must be 0 (all ordered links) or 1 (pair-assigned)");

namespace saans {

// ---------------------------------------------------------------------------
// Fixed points the schedule is judged against
// ---------------------------------------------------------------------------

/// The DSP's hard floor. `wifisense/features.py` REFUSES to construct a
/// `FeatureExtractor` below this:
///
///     "FeatureFrame needs >= 10 Hz CSI: the gross-motion band reaches 4 Hz,
///      and breathing needs it anyway (campus plan, constraint 4)"
///
/// `BasicFeatureExtractor::configure` is the same check in C++ and returns
/// false. A schedule that lands below this does not produce a degraded node,
/// it produces an INERT one, which is why it is a validity failure here and
/// not a warning.
inline constexpr float kDspFloorHz = 10.0f;

/// What the plan asks for. `CAMPUS_SENSING_PROMPT.md` §5.1: "Target at least
/// 20 Hz per link", and `edge/README.md` ships the node at 20 Hz. Between
/// this and `kDspFloorHz` the node runs but has half the spectral resolution
/// its defaults assume, so it is reported, never silently accepted.
inline constexpr float kTargetLinkRateHz = 20.0f;

/// Node ids ride in `FeatureFrame::tx_id`, one byte, with 0xFF reserved for
/// "an access point / unknown" (spec/wire.md §2). So ids run 0..254 and a
/// mesh larger than this cannot be addressed on the wire at all — a limit
/// worth naming, even though the rate floor bites two orders of magnitude
/// sooner.
inline constexpr std::uint16_t kMaxTdmaNodes = 255;

// ---------------------------------------------------------------------------
// Configuration
// ---------------------------------------------------------------------------

/// Which links a node extracts. See the reciprocity section in the file
/// header for why `PairAssigned` is the default and what it risks.
enum class LinkAssignment : std::uint8_t {
    /// Every ordered link (tx, rx): N-1 extractors per node. Makes no
    /// reciprocity assumption and costs twice the RAM.
    AllOrderedLinks = 0,
    /// Each unordered pair extracted by exactly one of its two endpoints:
    /// ceil((N-1)/2) extractors on the busiest node.
    PairAssigned = 1,
};

/// The build's default, from `SAANS_LINK_ASSIGNMENT`.
inline constexpr LinkAssignment kDefaultLinkAssignment =
    (SAANS_LINK_ASSIGNMENT != 0) ? LinkAssignment::PairAssigned : LinkAssignment::AllOrderedLinks;

/// Everything the schedule depends on. Defaults are the N=4 room this project
/// is actually planning (4 nodes, 12.5 ms slots → exactly 20 Hz per link).
struct TdmaConfig {
    /// Nodes in the mesh, including node 0 (the sync source). Two is the
    /// minimum that has a link at all.
    std::uint16_t n_nodes = 4;

    /// One node's transmit slot. The whole trade lives here: per-link rate is
    /// 1/(N * slot), so halving the slot doubles the rate and doubles the
    /// channel's duty cycle.
    float slot_ms = 12.5f;

    /// Dead time at the end of each slot. It pays for three things, in
    /// descending order of size on a real node:
    ///   1. sync-beacon and CSI-callback scheduling jitter on a FreeRTOS task
    ///      that is also running the DSP — tens to hundreds of microseconds,
    ///      and the term this default is actually sized by;
    ///   2. accumulated clock skew between sync beacons (40 us at the
    ///      defaults — see `worst_case_skew_s`);
    ///   3. error in the airtime estimate below.
    /// 1 ms is ~25x the drift term at a 1 s resync, which is the margin that
    /// lets the resync interval stretch to ~25 s before drift alone eats the
    /// band. Item 1 is a guess until S5 measures it.
    float guard_ms = 1.0f;

    /// Per-node crystal tolerance, one-sided, in parts per million. ESP32
    /// modules carry a 40 MHz crystal typically specified at +/-10 ppm at
    /// room temperature and +/-20-30 ppm over the industrial range; 20 ppm is
    /// the conservative middle. Two nodes drift in OPPOSITE directions, so
    /// the relative error is twice this — see `worst_case_skew_s`.
    float clock_drift_ppm = 20.0f;

    /// Fraction of broadcasts lost, [0, 1]. Deliberately 0 by default: a
    /// broadcast ESP-NOW frame is not ACKed and not retried, so loss is
    /// whatever the channel does, and neither `RESEARCH.md` nor `NOTES.md`
    /// records a figure for an ESP32 mesh on a campus channel. Inventing one
    /// would be the model telling itself a story. S5 measures it; until then
    /// this knob exists so a scenario can be explored, not so a default can
    /// be quoted.
    float packet_loss = 0.0f;

    /// Time one broadcast occupies the channel, milliseconds. NOT zero, and
    /// the reason it is a parameter is that it depends on a PHY rate that the
    /// firmware chooses. For a ~50-byte payload at the 1 Mbps DSSS basic rate
    /// — which is what ESP-NOW broadcast falls back to unless the rate is
    /// pinned, and the robust choice across a building:
    ///
    ///     802.11 MAC header                      24 B
    ///     action category + OUI + ESP-NOW header ~11 B
    ///     payload                                 50 B
    ///     FCS                                      4 B
    ///                                            ----
    ///                                             89 B = 712 bits
    ///     712 bits at 1 Mbps                      712 us
    ///     long PLCP preamble + header             192 us
    ///     DIFS 50 us + mean backoff (CWmin 15,
    ///       7.5 slots x 20 us = 150 us)           200 us
    ///                                            ------
    ///                                           ~1.10 ms
    ///
    /// No SIFS+ACK term: broadcasts are unacknowledged, which is also why
    /// `packet_loss` above is pure loss with no retry to hide it. At 11 Mbps
    /// the payload term drops to ~65 us and the total to ~0.46 ms, so a mesh
    /// that needs tighter slots has that lever before it has any other.
    float frame_airtime_ms = 1.1f;

    /// How often a node re-disciplines its clock to node 0's sync beacon.
    /// Node 0 marks EVERY superframe, so this could in principle be
    /// `superframe_s`; it is a separate knob because a node that corrects on
    /// every beacon is also chasing that beacon's own jitter, and firmware
    /// will normally filter and correct on a slower cadence. Set it equal to
    /// `superframe_s` to model "correct on every beacon". The drift term
    /// scales linearly with it, which is the whole reason a sync beacon
    /// exists.
    float resync_interval_s = 1.0f;

    /// Which links this node extracts. Defaults to the build's
    /// `SAANS_LINK_ASSIGNMENT`.
    LinkAssignment link_assignment = kDefaultLinkAssignment;
};

// ---------------------------------------------------------------------------
// Validity, with reasons
// ---------------------------------------------------------------------------

/// Why a schedule is unusable. A bare bool here would be useless: "12 nodes
/// does not work" and "12 nodes does not work BECAUSE the slot cannot hold
/// the airtime" lead to different fixes (fewer nodes vs a faster PHY rate),
/// and the second one is only visible if the model says which test failed.
enum class TdmaStatus : std::uint8_t {
    Ok = 0,
    /// Fewer than two nodes: there is no link to sample.
    TooFewNodes,
    /// More nodes than `tx_id` can address (spec/wire.md: 0xFF is reserved).
    TooManyNodes,
    /// A timing parameter is negative, zero where it must be positive, or not
    /// finite. Checked before the physics so a NaN cannot masquerade as a
    /// rate failure.
    BadTiming,
    /// airtime + guard exceeds the slot: slots overlap and two nodes
    /// transmit at once. Collisions, not slow sampling.
    SlotOverrun,
    /// Accumulated clock skew between sync beacons exceeds the guard band:
    /// a node drifts out of its own slot and into its neighbour's.
    GuardTooNarrow,
    /// per_link_rate_hz is below `kDspFloorHz`. The extractor will not
    /// configure at all at this rate; the node would be inert.
    RateBelowDspFloor,
};

/// Human-readable reason, for a report or a boot log. Returns a literal, so
/// it is safe to hold and costs no storage on the node.
inline const char* tdma_status_text(TdmaStatus s) {
    switch (s) {
        case TdmaStatus::Ok:
            return "ok";
        case TdmaStatus::TooFewNodes:
            return "fewer than 2 nodes: no link exists";
        case TdmaStatus::TooManyNodes:
            return "more nodes than tx_id can address (0..254; 0xFF reserved)";
        case TdmaStatus::BadTiming:
            return "a timing parameter is non-finite, negative or zero";
        case TdmaStatus::SlotOverrun:
            return "airtime + guard exceeds the slot: slots overlap, transmissions collide";
        case TdmaStatus::GuardTooNarrow:
            return "clock drift between syncs exceeds the guard band: slots collide";
        case TdmaStatus::RateBelowDspFloor:
            return "per-link rate below the 10 Hz DSP floor: the extractor will not configure";
    }
    return "unknown";
}

/// Effective per-link rate once loss is applied, with the loss expressed the
/// way the wire expresses it.
struct TdmaLinkRate {
    /// 1/(N*slot): what the schedule allocates.
    float nominal_hz = 0.0f;
    /// nominal * (1 - packet_loss): what the extractor actually sees.
    float effective_hz = 0.0f;
    /// `round(packet_loss * 1000)`, clamped to [0, 1000]. Directly assignable
    /// to `FeatureFrame::loss_permille` (spec/wire.md byte 18).
    std::uint16_t loss_permille = 0;
};

namespace tdma_detail {

/// Python's `round()` — half to even — so that a loss fraction converted here
/// lands on the same permille as `features.py`'s
/// `int(round(loss * 1000.0))`. Named separately from `detail::round_half_even`
/// in feature_extractor.hpp rather than shared: this header deliberately
/// includes nothing from the DSP chain, so that the model stays a pure,
/// standalone thing a planning tool can include on its own.
///
/// `double` here is the documented exception, not a slip: the input is an
/// init-time scalar, not a hot-path sample, and half-even on a float32
/// product would round twice.
inline std::int32_t round_half_even(double v) {
    const double r = std::nearbyint(v);  // FE_TONEAREST == half-to-even
    return static_cast<std::int32_t>(r);
}

inline bool finite_positive(float v) { return std::isfinite(v) && v > 0.0f; }
inline bool finite_non_negative(float v) { return std::isfinite(v) && v >= 0.0f; }

}  // namespace tdma_detail

// ---------------------------------------------------------------------------
// TdmaSchedule
// ---------------------------------------------------------------------------

/// Everything derivable from a `TdmaConfig`, computed once in the
/// constructor. Trivially copyable, no allocation, no I/O.
class TdmaSchedule {
  public:
    TdmaSchedule() : TdmaSchedule(TdmaConfig{}) {}

    explicit TdmaSchedule(const TdmaConfig& cfg) : cfg_(cfg) { compute(); }

    const TdmaConfig& config() const { return cfg_; }

    // -- timing ------------------------------------------------------------

    /// N * slot. One full rotation; every ordered link is sampled once.
    float superframe_s() const { return superframe_s_; }

    /// 1 / (N * slot).
    ///
    /// COMPUTED FROM MILLISECONDS ON PURPOSE: `1000 / (N * slot_ms)` and
    /// `1 / (N * slot_ms / 1000)` are the same number in real arithmetic and
    /// NOT the same float32. The seconds-first form rounds twice — `0.001f` is
    /// not 1/1000, and the product is rounded again before the reciprocal —
    /// and the two errors do not always cancel. At N=2, slot 10 ms it returns
    /// 49.9999962 where the answer is exactly 50, so a `>= 50.0f` check fails
    /// for a reason that has nothing to do with the schedule.
    ///
    /// NOT at N=4, slot 12.5, which an earlier version of this comment claimed
    /// and which is wrong: there the seconds-first form's error is 3e-7, well
    /// inside the 9.5e-7 half-ulp at 20 Hz, so it rounds to exactly 20.0f and
    /// the default configuration survives either form. The trap is real; the
    /// project's own default is simply not where it bites, and a maintainer
    /// who checked the claimed example would have concluded the opposite.
    ///
    /// The ms-first form keeps every case where N*slot_ms is a round number of
    /// milliseconds exact, which is every case anyone configures.
    /// `test_tdma.cpp` pins nine of them by EXACT equality; swap the two forms
    /// and the N=2/10 ms case is the one that goes red.
    float per_link_rate_hz() const { return per_link_rate_hz_; }

    /// Transmissions per second across the whole mesh — one per slot,
    /// 1/slot_s. This is the channel's packet rate, and
    /// `mesh_tx_rate_hz * frame_airtime_ms / 1000` is its duty cycle.
    float mesh_tx_rate_hz() const { return mesh_tx_rate_hz_; }

    /// CSI frames one node's radio delivers per second: it hears all N-1
    /// other transmitters. This is the callback load on the node regardless
    /// of how many of those links it chooses to extract.
    float csi_frames_per_node_s() const { return csi_per_node_s_; }

    /// CSI frames produced per second across the whole mesh, counting every
    /// ORDERED link: N*(N-1) links each sampled at `per_link_rate_hz`. Named
    /// as the brief names it; the three rates above are kept separate because
    /// "aggregate" is exactly the word the one published datapoint uses
    /// without defining, and a report that prints only one of them invites
    /// the reader to assume the flattering one.
    float aggregate_frames_per_s() const { return aggregate_frames_per_s_; }

    /// (airtime + guard) / slot. At 1.0 the slot is exactly full; above 1.0
    /// it has overflowed and `status()` is `SlotOverrun`.
    float slot_utilisation() const { return slot_utilisation_; }

    /// Worst-case clock skew between two nodes between sync beacons:
    ///
    ///     2 * clock_drift_ppm * 1e-6 * resync_interval_s
    ///
    /// The factor of 2 is not conservatism, it is the definition: the spec is
    /// one-sided, so one node may run fast by `ppm` while the other runs slow
    /// by `ppm` and the error between them is their difference. At the
    /// defaults (20 ppm, 1 s) that is 40 us; stretch the resync to 60 s and
    /// it is 2.4 ms, which is past any guard band anyone would pay for. That
    /// is the entire argument for a sync beacon.
    float worst_case_skew_s() const { return skew_s_; }

    // -- the schedule itself ------------------------------------------------

    /// Returned by `slot_for` for an id that is not in this mesh.
    static constexpr std::uint16_t kNoSlot = 0xFFFFu;

    /// Which slot node `node_id` transmits in. Node k transmits in slot k
    /// (CAMPUS_SENSING_PROMPT §5.1), so today this is the identity — and the
    /// test still asserts it is a BIJECTION over 0..N-1, because the day
    /// someone permutes the slots to separate two nodes that interfere, the
    /// property that must survive is "every node gets exactly one slot and no
    /// slot is shared", not the identity.
    std::uint16_t slot_for(std::uint16_t node_id) const {
        return (node_id < cfg_.n_nodes) ? node_id : kNoSlot;
    }

    /// When node `node_id` transmits, in seconds from the start of the
    /// superframe (which is node 0's sync beacon). `-1` for an id that is not
    /// in this mesh; a negative time is unmistakable where a 0 would read as
    /// "slot 0".
    float transmit_offset_s(std::uint16_t node_id) const {
        const std::uint16_t slot = slot_for(node_id);
        if (slot == kNoSlot) {
            return -1.0f;
        }
        return static_cast<float>(slot) * cfg_.slot_ms * 0.001f;
    }

    // -- pair assignment ----------------------------------------------------

    /// Does node `a` own the unordered pair {a, b} — i.e. does `a` run an
    /// extractor for this pair?
    ///
    /// THE RULE, and why it is not simply "the lower id owns it". Lower-id
    /// ownership is correct but piles N-1 pairs on node 0 and leaves node N-1
    /// with none, so node 0 needs the RAM of the unassigned scheme anyway and
    /// nothing has been saved. The balanced form rotates ownership around the
    /// ring instead: with d = (b - a) mod N, node `a` owns the pair when `a`
    /// reaches `b` by going FORWARD around the ring in fewer than half a
    /// turn.
    ///
    ///     2*d < N    -> a owns it      (b is in a's forward half)
    ///     2*d > N    -> b owns it
    ///     2*d == N   -> min(a, b) owns it
    ///
    /// The last line only arises for even N, where the pair is a "diameter"
    /// of the ring and the two directions are the same distance apart. There
    /// is no rotational tiebreak available, so it falls back to the lower id
    /// — which is where the brief's "the lower node id owns the pair" lives,
    /// and it is why even meshes end up with half the nodes carrying one
    /// extractor more than the other half.
    ///
    /// Each node then owns the ceil((N-1)/2) or floor((N-1)/2) pairs nearest
    /// it in the forward direction. Exactly one endpoint owns each pair
    /// (`owns_pair(a,b) != owns_pair(b,a)` for a != b), and the per-node
    /// counts differ by at most one — both proved in `test_tdma.cpp` rather
    /// than asserted here.
    ///
    /// Under `AllOrderedLinks` this returns true for every valid (a, b):
    /// every node extracts every link it hears, including both directions of
    /// each pair.
    ///
    /// A mesh whose SHAPE the model rejects — fewer than two nodes, or more
    /// than `tx_id` can address — owns nothing. Answering from `cfg_` anyway
    /// would let a 256-node config report 128 pairs per node from here while
    /// `extractors_per_node()` reported 0, and two accessors over one object
    /// contradicting each other is worse than either answer alone.
    bool owns_pair(std::uint16_t a, std::uint16_t b) const {
        if (!shape_ok_ || a == b || a >= cfg_.n_nodes || b >= cfg_.n_nodes) {
            return false;
        }
        if (cfg_.link_assignment == LinkAssignment::AllOrderedLinks) {
            return true;
        }
        const std::uint32_t n = cfg_.n_nodes;
        const std::uint32_t d =
            (static_cast<std::uint32_t>(b) + n - static_cast<std::uint32_t>(a)) % n;
        const std::uint32_t two_d = 2u * d;
        if (two_d < n) {
            return true;
        }
        if (two_d > n) {
            return false;
        }
        return a < b;
    }

    /// How many extractors node `node_id` runs. O(N) and loop-free of any
    /// allocation; it is called at init, never in the CSI path. 0 for an id
    /// outside the mesh, and 0 everywhere when the mesh shape is rejected —
    /// see `owns_pair`, whose answer this counts.
    std::uint16_t extractors_for_node(std::uint16_t node_id) const {
        if (!shape_ok_ || node_id >= cfg_.n_nodes) {
            return 0;
        }
        std::uint16_t count = 0;
        for (std::uint16_t other = 0; other < cfg_.n_nodes; ++other) {
            if (owns_pair(node_id, other)) {
                ++count;
            }
        }
        return count;
    }

    /// The WORST-CASE extractor count — what a node must be built to hold,
    /// which is the number that decides whether the firmware fits in RAM.
    ///
    ///   AllOrderedLinks  N-1
    ///   PairAssigned     ceil((N-1)/2), which for integers is just N/2
    ///
    /// Note the asymmetry this hides: under `PairAssigned` with even N, half
    /// the nodes own N/2 pairs and half own N/2 - 1, so the TOTAL is
    /// N(N-1)/2 and not N * extractors_per_node. Budget from this number;
    /// count work with `extractors_for_node`.
    ///
    /// It depends on the SHAPE alone, so it survives a timing failure: a
    /// planner asking "how much RAM would this mesh need?" gets an answer for
    /// a config whose slot is negative, and the two counts above agree with it
    /// case for case. It is 0 only when there is no mesh to count — fewer than
    /// two nodes, or more than `tx_id` can address — and 0 there means
    /// "nothing to count", never "nothing to pay".
    std::uint16_t extractors_per_node() const { return extractors_per_node_; }

    /// Unordered pairs in the mesh: N(N-1)/2. Under `PairAssigned` this is
    /// also the total number of extractors running anywhere in the mesh.
    std::uint32_t pair_count() const {
        const std::uint32_t n = cfg_.n_nodes;
        return (n < 2u) ? 0u : (n * (n - 1u)) / 2u;
    }

    /// Ordered links in the mesh: N(N-1). Under `AllOrderedLinks` this is the
    /// total number of extractors running anywhere in the mesh.
    std::uint32_t ordered_link_count() const {
        const std::uint32_t n = cfg_.n_nodes;
        return (n < 2u) ? 0u : n * (n - 1u);
    }

    // -- memory -------------------------------------------------------------

    /// Per-node DSP state: one `FeatureExtractor` per link the node extracts,
    /// PLUS whatever the node pays once no matter how many links it runs.
    ///
    /// `bytes_per_link` is an INPUT rather than a constant here because it is
    /// a property of the template parameters the firmware instantiates, and
    /// this header deliberately does not include `feature_extractor.hpp` —
    /// the model must stay usable from a planning tool that has no DSP in it.
    /// Measure the real figure with `sizeof(BasicFeatureExtractor<...>)`; see
    /// `bench/tdma_report.cpp` for the figures currently in play.
    ///
    /// `shared_once_bytes` IS NOT DECORATION, and it defaults to 0 only
    /// because a shape that has no shared state must not be made to pay for
    /// one. The node build moved the spectral scratch out of the link and into
    /// one arena the whole node computes in (`pipeline.hpp`; its
    /// interleaved-links parity check in `bench/size_report.cpp` is what makes
    /// the sharing legitimate rather than merely cheaper). That arena is
    /// 6404 B on a 64-bit host and it is paid ONCE — which is exactly why it
    /// cannot be folded into `bytes_per_link`, and why `size_report.cpp`
    /// carries it as its own `once/node` column and subtracts it from the
    /// budget before dividing. Omit it here and a planner gets a different
    /// answer from the two reports about the same node: at 16880 B/link
    /// against 171 KiB, 10 links instead of the 9 that actually fit.
    ///
    /// 0 when the mesh shape is rejected (`extractors_per_node()` is 0 and
    /// there is nothing to count). THAT IS NOT "IT FITS" — check `status()`
    /// before reading a 0 as headroom.
    std::uint32_t per_node_bytes(std::uint32_t bytes_per_link,
                                 std::uint32_t shared_once_bytes = 0) const {
        if (!shape_ok_) {
            return 0u;
        }
        // 64-bit intermediate and a saturating return, because this figure is
        // compared against a budget: a wrapped product reads as "fits", which
        // is the one wrong answer that is worse than no answer. Init-time
        // arithmetic on an integer, so the float32 rule does not apply.
        const std::uint64_t total =
            static_cast<std::uint64_t>(extractors_per_node_) *
                static_cast<std::uint64_t>(bytes_per_link) +
            static_cast<std::uint64_t>(shared_once_bytes);
        const std::uint64_t kMax = static_cast<std::uint64_t>(UINT32_MAX);
        return static_cast<std::uint32_t>(total > kMax ? kMax : total);
    }

    // -- rates and loss -----------------------------------------------------

    /// Nominal and post-loss per-link rate, plus the permille the wire wants.
    ///
    /// WHY LOSS IS NOT A VALIDITY FAILURE. A dropped broadcast is a HOLE in
    /// the time axis, and `SequenceGapFiller` exists to fill holes up to
    /// `max_gap`; the extractor keeps its uniform time axis and raises
    /// `discontinuity` only when a gap is longer than that. So loss degrades
    /// the link's information content without changing what the SCHEDULE can
    /// deliver, and conflating the two would let a lossy channel look like a
    /// mis-designed superframe. `status()` therefore judges the nominal rate;
    /// this function is what the system-health view reports.
    TdmaLinkRate effective_rate() const {
        TdmaLinkRate r;
        r.nominal_hz = per_link_rate_hz_;
        float loss = cfg_.packet_loss;
        if (!std::isfinite(loss) || loss < 0.0f) {
            loss = 0.0f;
        } else if (loss > 1.0f) {
            loss = 1.0f;
        }
        r.effective_hz = per_link_rate_hz_ * (1.0f - loss);
        const std::int32_t permille =
            tdma_detail::round_half_even(static_cast<double>(loss) * 1000.0);
        r.loss_permille = static_cast<std::uint16_t>(permille < 0 ? 0 : (permille > 1000 ? 1000
                                                                                         : permille));
        return r;
    }

    /// Does the POST-LOSS rate still clear the DSP floor? Separate from
    /// `status()` for the reason above: the gap filler may well carry a link
    /// whose effective rate dips below 10 Hz, but a system-health view should
    /// still say so.
    bool effective_rate_meets_floor() const { return effective_rate().effective_hz >= kDspFloorHz; }

    // -- validity -----------------------------------------------------------

    /// The first reason this schedule is unusable, most fundamental first:
    /// shape (node count, finite timings), then the two collision modes
    /// (`SlotOverrun`, `GuardTooNarrow`), then the rate floor. The order
    /// matters: a 12-node mesh at a 1 ms slot is BOTH overrun and too slow,
    /// and "your slots overlap" is the fixable complaint.
    TdmaStatus status() const { return status_; }

    bool is_valid() const { return status_ == TdmaStatus::Ok; }

    const char* status_text() const { return tdma_status_text(status_); }

    /// Valid AND at or above the 20 Hz the plan asks for. A schedule can be
    /// valid at 12 Hz; it is then a node running with half the spectral
    /// resolution its defaults assume, which is a decision and not a default.
    bool meets_target() const { return is_valid() && per_link_rate_hz_ >= kTargetLinkRateHz; }

    // Individual predicates, so a report can show EVERY failing condition
    // rather than only the first one `status()` reaches.
    bool slot_overruns() const { return slot_overruns_; }
    bool guard_too_narrow() const { return guard_too_narrow_; }
    bool rate_below_floor() const { return rate_below_floor_; }

  private:
    void compute() {
        // Shape first. Everything below divides by n_nodes or slot_ms.
        if (cfg_.n_nodes < 2u) {
            status_ = TdmaStatus::TooFewNodes;
            return;
        }
        if (cfg_.n_nodes > kMaxTdmaNodes) {
            status_ = TdmaStatus::TooManyNodes;
            return;
        }

        // THE LINK COUNTS ARE SETTLED HERE, ahead of the timing checks, and
        // the order is the point. They are a function of the shape and of
        // nothing else — no slot, no airtime, no drift — so a config with a
        // negative slot still has a well-defined "this mesh would need four
        // extractors a node". Compute them below the early returns instead and
        // `extractors_per_node()` reads 0 while `extractors_for_node(0)` reads
        // 4 on the same object, and `per_node_bytes()` — the function a
        // planning tool calls to ask "does this mesh fit?" — answers 0, which
        // reads as "fits" rather than as "unknown".
        shape_ok_            = true;
        extractors_per_node_ = (cfg_.link_assignment == LinkAssignment::AllOrderedLinks)
                                   ? static_cast<std::uint16_t>(cfg_.n_nodes - 1u)
                                   // ceil((N-1)/2) == N/2 in integer arithmetic.
                                   : static_cast<std::uint16_t>(cfg_.n_nodes / 2u);

        if (!tdma_detail::finite_positive(cfg_.slot_ms) ||
            !tdma_detail::finite_non_negative(cfg_.guard_ms) ||
            !tdma_detail::finite_non_negative(cfg_.frame_airtime_ms) ||
            !tdma_detail::finite_non_negative(cfg_.clock_drift_ppm) ||
            !tdma_detail::finite_non_negative(cfg_.resync_interval_s)) {
            status_ = TdmaStatus::BadTiming;
            return;
        }

        const float n_f            = static_cast<float>(cfg_.n_nodes);
        const float superframe_ms  = n_f * cfg_.slot_ms;
        superframe_s_              = superframe_ms * 0.001f;
        per_link_rate_hz_          = 1000.0f / superframe_ms;
        mesh_tx_rate_hz_           = 1000.0f / cfg_.slot_ms;
        const float links_per_node = n_f - 1.0f;
        csi_per_node_s_            = links_per_node * per_link_rate_hz_;
        aggregate_frames_per_s_    = n_f * csi_per_node_s_;

        const float occupied_ms = cfg_.frame_airtime_ms + cfg_.guard_ms;
        slot_utilisation_       = occupied_ms / cfg_.slot_ms;

        // One-sided ppm on each of two nodes -> twice the one-sided figure
        // between them. 1e-6f, not 1e-6: -Wdouble-promotion catches the
        // double literal, and it would be a real emulated-double multiply on
        // the S3.
        skew_s_ = 2.0f * cfg_.clock_drift_ppm * 1.0e-6f * cfg_.resync_interval_s;

        slot_overruns_    = occupied_ms > cfg_.slot_ms;
        guard_too_narrow_ = (skew_s_ * 1000.0f) > cfg_.guard_ms;
        rate_below_floor_ = per_link_rate_hz_ < kDspFloorHz;

        if (slot_overruns_) {
            status_ = TdmaStatus::SlotOverrun;
        } else if (guard_too_narrow_) {
            status_ = TdmaStatus::GuardTooNarrow;
        } else if (rate_below_floor_) {
            status_ = TdmaStatus::RateBelowDspFloor;
        } else {
            status_ = TdmaStatus::Ok;
        }
    }

    TdmaConfig    cfg_{};
    TdmaStatus    status_                 = TdmaStatus::TooFewNodes;
    float         superframe_s_           = 0.0f;
    float         per_link_rate_hz_       = 0.0f;
    float         mesh_tx_rate_hz_        = 0.0f;
    float         csi_per_node_s_         = 0.0f;
    float         aggregate_frames_per_s_ = 0.0f;
    float         slot_utilisation_       = 0.0f;
    float         skew_s_                 = 0.0f;
    std::uint16_t extractors_per_node_    = 0;
    /// 2 <= n_nodes <= kMaxTdmaNodes: there is a mesh to answer questions
    /// about. The link accessors key off this rather than off `status_`,
    /// because a schedule can be invalid for a TIMING reason and still have a
    /// perfectly well-defined set of links.
    bool          shape_ok_               = false;
    bool          slot_overruns_          = false;
    bool          guard_too_narrow_       = false;
    bool          rate_below_floor_       = false;
};

// ---------------------------------------------------------------------------
// Inverting the schedule
// ---------------------------------------------------------------------------

/// The largest slot, in milliseconds, that still delivers `target_hz` per
/// link with `n_nodes` nodes: 1000 / (N * target).
///
/// This is the question anyone actually asks ("I have 8 nodes and I want
/// 20 Hz — how fast do slots have to be?"), and it is worth a named function
/// because the answer is frequently SMALLER THAN THE AIRTIME, which is where
/// the scheme stops working. Compare it against
/// `frame_airtime_ms + guard_ms`: if it is smaller, no slot length satisfies
/// both and the mesh is too large at this PHY rate, full stop.
///
/// IT ROUND-TRIPS BY CONSTRUCTION, which the closed form alone does not. Both
/// the product and the quotient round, and when the quotient rounds UP the
/// slot is one ulp too long — which `per_link_rate_hz()` turns back into a
/// rate one ulp too LOW, so `meets_target()` refuses the slot this function
/// just recommended. At N=45 and 20 Hz the bare form returns 1.11111116 ms and
/// gets 19.9999981 Hz back; 14 of N=2..255 do that at 20 Hz, 72 of them at
/// 15 Hz, and at 1.4 Hz — the rate of the one published datapoint this model
/// is pinned against — it starts at N=3. So the result is stepped down until
/// the rate it produces actually clears the target.
///
/// The loop is bounded by the two roundings that can be wrong, not by hope:
/// each step strictly decreases a finite float, the rate is monotone in it,
/// and the cap is there so that a pathological target cannot spin. This is the
/// same argument `per_link_rate_hz()` makes for dividing 1000 by N*slot_ms
/// rather than taking a reciprocal in seconds, applied in the other direction.
inline float max_slot_ms_for_rate(std::uint16_t n_nodes, float target_hz) {
    if (n_nodes < 2u || !(target_hz > 0.0f) || !std::isfinite(target_hz)) {
        return 0.0f;
    }
    const float n_f     = static_cast<float>(n_nodes);
    float       slot_ms = 1000.0f / (n_f * target_hz);
    for (int i = 0; i < 4 && (1000.0f / (n_f * slot_ms)) < target_hz; ++i) {
        slot_ms = std::nextafterf(slot_ms, 0.0f);
    }
    return slot_ms;
}

/// The largest mesh that can hold `target_hz` per link given a slot that must
/// be at least `min_slot_ms` long (normally airtime + guard). Returns 0 when
/// even two nodes do not fit.
///
/// N <= 1000 / (target * min_slot_ms), floored. At the defaults — 20 Hz,
/// 1.1 ms airtime, 1.0 ms guard — that is 1000/(20*2.1) = 23 nodes by AIRTIME
/// alone, which is the headline this model exists to qualify: the channel is
/// not what limits a Saans mesh at realistic sizes. PER-NODE RAM IS (see
/// `per_node_bytes`) — at the memory piece's figures the mesh runs out of
/// DIRAM at a single-digit N with the BNR row store compiled in, an order of
/// magnitude before the channel runs out of airtime. The radio is not the
/// binding constraint; the memory is. `bench/tdma_report.cpp` computes both
/// walls from whatever `bench/size_report.cpp` currently measures, which is
/// why no node count is written down here.
inline std::uint16_t max_nodes_for_rate(float target_hz, float min_slot_ms) {
    if (!(target_hz > 0.0f) || !(min_slot_ms > 0.0f) || !std::isfinite(target_hz) ||
        !std::isfinite(min_slot_ms)) {
        return 0;
    }
    const float n = 1000.0f / (target_hz * min_slot_ms);
    if (!(n >= 2.0f)) {
        return 0;
    }
    const float floored = std::floor(n);
    if (floored > static_cast<float>(kMaxTdmaNodes)) {
        return kMaxTdmaNodes;
    }
    return static_cast<std::uint16_t>(floored);
}

}  // namespace saans
