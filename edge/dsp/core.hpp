// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — core types and the no-allocation contract.
//
// Port of the shared vocabulary in `wifisense/types.py`, plus the two
// primitives every later stage leans on: a fixed-capacity ring buffer and a
// compensated sum.
//
// ---------------------------------------------------------------------------
// THE NO-ALLOCATION CONTRACT (ADR-003, SAANS_PLAN §3.1)
// ---------------------------------------------------------------------------
// This library runs inside the ESP32-S3 CSI callback path. Nothing in any
// header here may allocate after init:
//
//   * no std::vector, std::deque, std::string, std::function
//   * no new, no malloc, no std::optional<T> where T is not trivially
//     destructible (std::optional itself is fine — it is storage-inline)
//   * every buffer is either a member array sized by a template parameter or
//     scratch handed in by the caller
//
// The reason is not purity. The CSI callback runs on a FreeRTOS task with a
// small fixed stack and hard deadlines; a heap walk at 20 Hz is both a jitter
// source and, once the heap fragments after a few days of uptime, a crash.
// Stages that need working memory take `float* scratch` and say how much.
//
// ---------------------------------------------------------------------------
// FLOAT, NOT DOUBLE (ADR-003)
// ---------------------------------------------------------------------------
// The S3 FPU is single precision. A `double` compiles to a software-emulation
// call costing 10-50x a native float op, so every literal here carries the `f`
// suffix and every libm call is the `f`-suffixed one. Build with
// -Wdouble-promotion: each hit is a real performance bug on target, not
// pedantry. Time accumulators are the documented exception (see the uptime_ms
// tolerance in spec/wire.md §3) and live in the feature stage, not here.
//
#include <cmath>
#include <cstddef>
#include <cstdint>

namespace saans {

// pi as a float. The decimal expansion is longer than float can hold on
// purpose: the compiler rounds once, correctly, instead of us rounding twice.
inline constexpr float kPi = 3.14159265358979323846f;

// Euler-Mascheroni, to the digits the Python reference writes it with
// (`wifisense/dsp.py`). Matching the literal matters: it sets the noise anchor
// of the band_peak confidence, and a different truncation would shift every
// confidence by ~1e-9 * span.
inline constexpr float kEulerMascheroni = 0.5772156649f;

/// `math.isfinite` for floats. Named rather than inlined at each call site
/// because the self-healing resets downstream all hang off it.
inline bool is_finite(float x) { return std::isfinite(x); }

/// `max(lo, min(v, hi))`, the clamp the Python reference spells out inline.
///
/// One documented divergence: for a NaN input this returns NaN, whereas
/// Python's `max(0.0, min(1.0, nan))` returns 1.0 — an artefact of how
/// CPython's two-argument min/max break ties against an unordered compare.
/// The only clamp in this library sits on a ratio of two values both already
/// proven positive and finite, so the case is unreachable; propagating NaN is
/// the better behaviour if it ever stops being unreachable.
template <typename T>
constexpr T clamp(T v, T lo, T hi) {
    return (v < lo) ? lo : ((v > hi) ? hi : v);
}

// ---------------------------------------------------------------------------
// CsiFrame
// ---------------------------------------------------------------------------

/// One CSI snapshot — the C++ form of `wifisense.types.CsiFrame`.
///
/// `amplitudes` and `phases` are per-subcarrier arrays of `n_subcarriers`
/// floats. Phase is in radians and *wrapped* to (-pi, pi]; unwrapping and
/// CFO/SFO removal are the caller's problem, and matter a great deal for
/// anything beyond amplitude-only sensing.
///
/// The pointers are NON-OWNING and this struct is a POD view. Nothing in the
/// pipeline copies subcarrier arrays — on target they point straight at the
/// ESP-IDF CSI callback's buffer, which is valid only for the duration of the
/// callback. A stage that needs history keeps the *scalar* it derived, in a
/// RingBuffer, never the frame.
///
/// Python validates lengths and a positive sample rate in __post_init__ and
/// raises. With -fno-exceptions there is nothing to raise, so validation moves
/// to the stage that consumes the frame; `valid()` is the same predicate.
struct CsiFrame {
    const float*  amplitudes     = nullptr;
    const float*  phases         = nullptr;
    std::size_t   n_subcarriers  = 0;
    std::uint32_t sample_index   = 0;
    float         sample_rate_hz = 0.0f;
    std::int8_t   rssi_dbm       = 0;
    std::uint8_t  node_id        = 0;

    bool valid() const {
        return amplitudes != nullptr && phases != nullptr && n_subcarriers > 0 &&
               sample_rate_hz > 0.0f;
    }
};

// ---------------------------------------------------------------------------
// RingBuffer
// ---------------------------------------------------------------------------

/// Fixed-capacity ring buffer; `push` overwrites the oldest element when full.
///
/// This is the C++ form of the `collections.deque(maxlen=N)` the Python stages
/// use for their analysis history. Capacity is a template parameter so the
/// storage is a member array: no heap, and the size is visible to the linker
/// (which is how `edge/firmware` will budget RAM per link).
///
/// Indexing is OLDEST-TO-NEWEST, matching `list(deque)` in Python, so a port
/// reads the same way as the reference. `operator[](0)` is the oldest sample
/// still held, `operator[](size()-1)` the newest.
template <typename T, std::size_t N>
class RingBuffer {
    static_assert(N > 0, "RingBuffer capacity must be non-zero");

  public:
    using value_type = T;

    static constexpr std::size_t capacity() { return N; }

    std::size_t size() const { return size_; }
    bool        empty() const { return size_ == 0; }
    bool        full() const { return size_ == N; }

    void clear() {
        head_ = 0;
        size_ = 0;
    }

    /// Append, dropping the oldest element once full.
    void push(const T& v) {
        buf_[head_] = v;
        head_ = (head_ + 1u) % N;
        if (size_ < N) {
            ++size_;
        }
    }

    /// `i == 0` is the oldest element held. Out-of-range is a programming
    /// error; there is no bounds check in the hot path, same as a raw array.
    const T& operator[](std::size_t i) const { return buf_[(start() + i) % N]; }
    T&       operator[](std::size_t i) { return buf_[(start() + i) % N]; }

    const T& newest() const { return (*this)[size_ - 1u]; }
    const T& oldest() const { return (*this)[0]; }

    /// CONTIGUOUS ACCESS FOR THE SPECTRAL CODE.
    ///
    /// `goertzel_power` and the band scans want a flat array, and the ring's
    /// storage wraps. Rather than hand out a pointer that is only sometimes
    /// contiguous (an invitation to a wrap-around bug that shows up as a
    /// phantom spectral line), linearise oldest-to-newest into caller-provided
    /// scratch. This is the exact analogue of `list(self._history)` in
    /// `wifisense/vitals.py`, and the only sanctioned way to get a flat view.
    ///
    /// `dst` must have room for at least `size()` elements. Returns the number
    /// of elements written, which is `size()` unless `dst_capacity` is smaller,
    /// in which case NOTHING is written and 0 is returned — a short copy would
    /// silently analyse the wrong window.
    std::size_t copy_to(T* dst, std::size_t dst_capacity) const {
        if (dst == nullptr || dst_capacity < size_) {
            return 0;
        }
        const std::size_t s = start();
        for (std::size_t i = 0; i < size_; ++i) {
            dst[i] = buf_[(s + i) % N];
        }
        return size_;
    }

  private:
    // Index of the oldest element. While not yet full the buffer has never
    // wrapped, so element 0 is at slot 0.
    std::size_t start() const { return (size_ == N) ? head_ : 0u; }

    T           buf_[N]{};
    std::size_t head_ = 0;  // next slot to write
    std::size_t size_ = 0;
};

// ---------------------------------------------------------------------------
// Compensated summation
// ---------------------------------------------------------------------------

/// Neumaier (improved Kahan-Babuska) running sum, in float32.
///
/// WHY THIS EXISTS, and why it is not optional here.
///
/// The Python reference adds floats with the builtin `sum()`. From CPython
/// 3.12 that is itself compensated (Neumaier), and the reference relies on it:
/// `_RollingRms` in `wifisense/presence.py` deliberately re-sums its whole
/// window on every read rather than keeping a running total, precisely so that
/// the three implementations cannot drift apart. A naive summation here would
/// be a *different algorithm*, not a rounding difference, and the gap grows
/// with window length.
///
/// In float32 it matters more, not less. Over a 201-point periodogram the
/// naive error is ~sqrt(n)*eps ~ 1.7e-6 relative; the band_peak confidence
/// divides the peak by the mean of exactly that sum and then rescales the
/// result across a span of ~8.5, so on the `band_peak_noise` vector (where the
/// answer is 0.0196) a 1.7e-6 error in the mean is already ~1e-4 of the
/// reported confidence — the whole tolerance budget, spent on bookkeeping.
///
/// The compensation itself costs two adds and a compare per element and no
/// memory, which is the cheapest thing in this file.
class NeumaierSum {
  public:
    void add(float x) {
        const float t = sum_ + x;
        // Whichever operand is larger in magnitude keeps its bits in `t`; the
        // other one is the one losing low bits, so recover from that side.
        if (fabsf(sum_) >= fabsf(x)) {
            c_ += (sum_ - t) + x;
        } else {
            c_ += (x - t) + sum_;
        }
        sum_ = t;
    }

    /// The sum with the accumulated correction folded in. Deliberately a
    /// read-only view: folding `c_` back into `sum_` would discard the
    /// correction's own low bits on the next add.
    float value() const { return sum_ + c_; }

    void reset() {
        sum_ = 0.0f;
        c_   = 0.0f;
    }

  private:
    float sum_ = 0.0f;
    float c_   = 0.0f;
};

/// Compensated sum of `n` floats. See NeumaierSum for why this is not a loop
/// with a `+=`.
inline float neumaier_sum(const float* v, std::size_t n) {
    if (v == nullptr || n == 0) {
        return 0.0f;
    }
    NeumaierSum acc;
    for (std::size_t i = 0; i < n; ++i) {
        acc.add(v[i]);
    }
    return acc.value();
}

/// Root-mean-square over a flat window, re-summed from scratch on every call.
///
/// Mirrors `_RollingRms.value` in `wifisense/features.py`, including the
/// decision NOT to keep a running total:
///
///     "Summed afresh on each read (twice a second) rather than kept as a
///      running total: running-sum float drift over a night of samples would
///      make the three implementations disagree for no physical reason."
///
/// That argument is stronger in float32, so the re-summing stays.
///
/// The reference pushes `x * x` into its deque and sums the squares; squaring
/// here instead is the same single rounding applied in the same order, so the
/// two agree bit for bit. Returns 0 for an empty window, as the reference does.
inline float rolling_rms(const float* v, std::size_t n) {
    if (v == nullptr || n == 0) {
        return 0.0f;
    }
    NeumaierSum acc;
    for (std::size_t i = 0; i < n; ++i) {
        acc.add(v[i] * v[i]);
    }
    const float mean_sq = acc.value() / static_cast<float>(n);
    // A sum of squares cannot be negative; the guard only stops a
    // compensation underflow from turning into a NaN. sqrtf(0) == 0 either
    // way, so this agrees with the reference's unguarded math.sqrt.
    return sqrtf(mean_sq < 0.0f ? 0.0f : mean_sq);
}

}  // namespace saans
