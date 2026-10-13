// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — sequence gap filling.
//
// Port of `SequenceGapFiller` in `wifisense/sources/esp32_udp.py`. Header-only,
// no allocation after init, float32 throughout.
//
// ---------------------------------------------------------------------------
// WHY THIS STAGE EXISTS (the reference's own argument, and it was measured)
// ---------------------------------------------------------------------------
//   "Every stage downstream assumes one frame per 1/rate seconds. Drop 5%
//    of packets and feed the survivors straight in, and the pipeline's
//    clock runs 5% fast: a 15 BPM breather reads ~15.8 BPM, outside the
//    0.5 BPM tolerance, with nothing to say why."
//
// That is not a hypothetical. 15 / 0.95 = 15.79 BPM, and the 0.79 BPM error is
// larger than the whole tolerance budget for the breathing rate — while every
// confidence score still reads healthy, because the spectrum is perfectly
// clean, just at the wrong frequency. The wire sequence number is the only
// thing in the stream that knows the difference between "no breath" and "no
// packet", so it is what restores the time axis.
//
// For a short gap the fill is linear in AMPLITUDE and holds PHASE:
//
//   "amplitudes interpolated; phase held, since wrapped phase does not
//    interpolate and nothing downstream uses it yet"
//
// Interpolating a wrapped phase is worse than useless: halfway between +3.1 and
// -3.1 rad is +-3.14, not 0, so the linear answer is the one point on the
// circle furthest from the truth. Holding the last value is at least a
// defensible zero-order hold until someone unwraps.
//
// What is NOT papered over: long gaps, backwards jumps (node reboot, or
// reordering on the wire) and duplicates. Those pass through or are dropped,
// and each is counted, "so the operator can see the stream was discontinuous".
// Inventing 500 frames to bridge a 25 s outage would produce a beautifully
// smooth, entirely fictional breathing trace — the one failure mode a health
// sensor must not have.
//
#include <cstddef>
#include <cstdint>

#include "core.hpp"

namespace saans {

/// The sequence number's modulus. `1 << 32` in the reference; here it is the
/// natural wrap of std::uint32_t, so the arithmetic below needs no explicit
/// mask — `sequence - last_seq_` on uint32 IS `(a - b) % _SEQ_MOD`.
inline constexpr std::uint32_t kSeqHalf = 0x80000000u;  // _SEQ_MOD // 2

/// Keep the time axis uniform across lost packets.
///
/// `MaxSubcarriers` bounds the per-frame subcarrier count this filler can
/// retain; `MaxFill` bounds how many frames one gap may synthesise.
///
/// ---------------------------------------------------------------------------
/// THE CAPACITY CONTRACT
/// ---------------------------------------------------------------------------
/// The reference appends to a Python list and cannot overflow. Here the fill
/// frames land in a caller-provided `Arena`, so the bound has to be explicit:
///
///   * `max_gap` IS CLAMPED TO `MaxFill` at construction. A gap can therefore
///     never ask for more frames than the arena holds, and the overflow branch
///     does not exist rather than being a branch that might be wrong. Ask for
///     `max_gap = 40` with `MaxFill = 32` and you get 32; `max_gap()` reports
///     what you actually got, and a longer gap then takes the discontinuity
///     path exactly as a gap longer than a correctly-sized `max_gap` would.
///     Size `MaxFill >= max_gap` and the clamp is invisible.
///
///   * A frame with more than `MaxSubcarriers` subcarriers cannot be retained
///     as history, so it cannot be interpolated from. See `push`.
///
/// The reference raises ValueError for `max_gap < 0`; an unsigned parameter
/// makes that state unrepresentable, which is the better fix given there is no
/// exception to raise.
///
/// ---------------------------------------------------------------------------
/// LIFETIME
/// ---------------------------------------------------------------------------
/// `CsiFrame` is a non-owning view (see core.hpp), so the returned frames are
/// views too:
///
///   * the interpolated frames point into `arena` and are valid until the next
///     call that uses that arena;
///   * the pass-through frame carries the caller's own pointers unchanged and
///     is valid for exactly as long as those are — on target, the duration of
///     the CSI callback.
///
/// Consume the whole `Output` before the next `push`. That is the same
/// discipline the rest of this library already requires of a CsiFrame, and it
/// is why the filler keeps its own copy of the previous frame internally
/// instead of holding the caller's pointers across calls.
template <std::size_t MaxSubcarriers = 64, std::size_t MaxFill = 32>
class SequenceGapFiller {
    static_assert(MaxSubcarriers > 0, "MaxSubcarriers must be non-zero");
    static_assert(MaxFill > 0, "MaxFill must be non-zero");

  public:
    /// Working memory for one `push`. Caller-owned; place it statically or in
    /// the owning pipeline object, not on a FreeRTOS task stack — at the
    /// defaults it is 32 * 64 floats of amplitude alone (8 KiB).
    struct Arena {
        /// Interpolated amplitudes, one row per synthesised frame.
        float amplitudes[MaxFill][MaxSubcarriers];
        /// The held phase vector, copied once and SHARED by every fill frame
        /// of this gap. The reference writes `list(self._last.phases)` per
        /// frame, i.e. MaxFill identical copies; nothing downstream mutates a
        /// frame, so one copy is observationally identical and saves the arena
        /// a second MaxFill x MaxSubcarriers block.
        float phases[MaxSubcarriers];
        /// Fill frames, then the pass-through frame.
        CsiFrame frames[MaxFill + 1];
    };

    /// "Frames to process, in order: interpolated fill, then `frame`."
    /// Empty for a dropped duplicate.
    struct Output {
        const CsiFrame* frames = nullptr;
        std::size_t     count  = 0;

        const CsiFrame* begin() const { return frames; }
        const CsiFrame* end() const { return frames + count; }
        bool            empty() const { return count == 0; }
        const CsiFrame& operator[](std::size_t i) const { return frames[i]; }
    };

    explicit SequenceGapFiller(std::uint32_t max_gap = 20)
        : max_gap_(max_gap > MaxFill ? static_cast<std::uint32_t>(MaxFill) : max_gap) {}

    /// The gap length actually in force, after the `MaxFill` clamp.
    std::uint32_t max_gap() const { return max_gap_; }

    /// Counters, public as in the reference: an operator reads them straight
    /// off the object, and `FeatureExtractor` turns `filled` into the
    /// loss_permille field of the wire frame.
    std::uint32_t filled          = 0;
    std::uint32_t discontinuities = 0;
    std::uint32_t duplicates      = 0;

    /// Forget the history without touching the counters — the counters are a
    /// session total, the history is what the next gap is measured from.
    void forget_history() { has_last_ = false; }

    /// Counters and history both back to construction.
    void reset() {
        has_last_       = false;
        filled          = 0;
        discontinuities = 0;
        duplicates      = 0;
    }

    bool          has_history() const { return has_last_; }
    std::uint32_t last_sequence() const { return last_seq_; }

    Output push(std::uint32_t sequence, const CsiFrame& frame, Arena& arena) {
        Output out;
        out.frames = arena.frames;

        // PORT-ONLY PATH, no counterpart in the reference. Python validates in
        // CsiFrame.__post_init__ and raises; with -fno-exceptions validation
        // moves to the consumer (core.hpp), and this is the consumer. A frame
        // we cannot retain is one we cannot interpolate FROM, so it passes
        // through alone and the history is dropped. It counts as a
        // discontinuity for the same reason a long gap does — the time axis
        // really is broken here and the operator should see it — except on the
        // very first frame, where the reference counts nothing either.
        if (!frame.valid() || frame.n_subcarriers > MaxSubcarriers) {
            if (has_last_) {
                ++discontinuities;
            }
            has_last_        = false;
            arena.frames[0]  = frame;
            out.count        = 1;
            return out;
        }

        if (!has_last_) {
            remember(sequence, frame);
            arena.frames[0] = frame;
            out.count       = 1;
            return out;
        }

        // `(sequence - self._last_seq) % _SEQ_MOD`, for free on uint32.
        const std::uint32_t step = sequence - last_seq_;
        if (step == 0u) {
            ++duplicates;
            out.count = 0;
            return out;  // and the history is deliberately NOT advanced
        }

        const std::uint32_t missing = step - 1u;
        std::size_t         n_fill  = 0;

        if (step > kSeqHalf || missing > max_gap_) {
            // Backwards jump (reordering, or a node that rebooted its
            // counter), or a gap too long to bridge honestly.
            ++discontinuities;
        } else if (missing > 0u && last_n_ == frame.n_subcarriers) {
            n_fill = interpolate(frame, arena, step, missing);
            filled += missing;
        } else if (missing > 0u) {
            // Short gap, but the subcarrier count changed underneath us — the
            // two endpoints are not the same measurement, so there is nothing
            // to interpolate along.
            ++discontinuities;
        }

        arena.frames[n_fill] = frame;
        out.count            = n_fill + 1u;
        remember(sequence, frame);
        return out;
    }

  private:
    /// Build the `missing` fill frames into the arena. Returns how many.
    ///
    /// FLOAT32 NOTE. The reference's weight is `w = j / step`, an f64 true
    /// division of two small ints. Here both operands convert exactly to float
    /// and the quotient is correctly rounded, so `w` differs from the f64 value
    /// by at most half a float32 ulp (~6e-8 relative). The interpolation then
    /// scales that error by the SPAN, `|y - x|`, not by the sample magnitude:
    /// the absolute error in a filled amplitude is ~6e-8 * |y - x|, which for a
    /// short gap in a slowly-varying amplitude is far below the error already
    /// present in the amplitude itself.
    ///
    /// The form is kept as the reference writes it, `x + (y - x) * w`, rather
    /// than the algebraically equal `x*(1-w) + y*w`. The reference's form is
    /// the better-conditioned one here — it is an exact perturbation OF x, so
    /// it cannot drift away from the endpoints, and at w = 0 and w = 1 it
    /// reproduces x and y exactly, which the two-term form does not guarantee
    /// in float32.
    std::size_t interpolate(const CsiFrame& frame, Arena& arena, std::uint32_t step,
                            std::uint32_t missing) {
        const std::size_t n = last_n_;

        // `phases=list(self._last.phases)` — held, not interpolated.
        for (std::size_t i = 0; i < n; ++i) {
            arena.phases[i] = last_phase_[i];
        }

        const float step_f = static_cast<float>(step);
        for (std::uint32_t jj = 1; jj <= missing; ++jj) {
            const std::size_t row = static_cast<std::size_t>(jj - 1u);
            const float       w   = static_cast<float>(jj) / step_f;
            float* const      dst = arena.amplitudes[row];
            for (std::size_t i = 0; i < n; ++i) {
                const float x = last_amp_[i];
                const float y = frame.amplitudes[i];
                dst[i]        = x + (y - x) * w;
            }

            CsiFrame& f      = arena.frames[row];
            f.amplitudes     = dst;
            f.phases         = arena.phases;
            f.n_subcarriers  = n;
            // `(self._last_seq + j) % _SEQ_MOD`, again free on uint32.
            f.sample_index   = last_seq_ + jj;
            // The reference takes these three from the NEW frame, not the old
            // one: the fill is a reconstruction of what the link was doing on
            // the way to `frame`, and the new frame is the better evidence.
            f.sample_rate_hz = frame.sample_rate_hz;
            f.rssi_dbm       = frame.rssi_dbm;
            f.node_id        = frame.node_id;
        }
        return static_cast<std::size_t>(missing);
    }

    /// `self._last_seq, self._last = sequence, frame`, by value.
    ///
    /// The reference keeps a reference to the frame object. We cannot: a
    /// CsiFrame's pointers are the ESP-IDF callback's buffer and are dangling
    /// by the next packet. Copying the two arrays is the price of holding
    /// history at all, and it is why `MaxSubcarriers` exists.
    void remember(std::uint32_t sequence, const CsiFrame& frame) {
        last_seq_ = sequence;
        last_n_   = frame.n_subcarriers;
        for (std::size_t i = 0; i < last_n_; ++i) {
            last_amp_[i]   = frame.amplitudes[i];
            last_phase_[i] = frame.phases[i];
        }
        has_last_ = true;
    }

    std::uint32_t max_gap_;
    bool          has_last_ = false;
    std::uint32_t last_seq_ = 0;
    std::size_t   last_n_   = 0;
    float         last_amp_[MaxSubcarriers]{};
    float         last_phase_[MaxSubcarriers]{};
};

}  // namespace saans
