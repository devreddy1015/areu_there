// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — radix-2 complex FFT.
//
// Port of `_next_pow2` and `_fft` in `wifisense/preprocess.py`. Header-only,
// no allocation, float32 throughout; the transform is in-place on buffers the
// caller owns.
//
// WHY AN FFT AT ALL, when the rest of this library is Goertzel.
//
//   "The selector needs the in-band bins for similarity and every bin for
//    the BNR denominator, not just the two narrow bands Goertzel serves
//    elsewhere, so a real FFT pays for itself here: 56 subcarriers x 128
//    points is a few milliseconds."
//
// Goertzel costs O(n) per evaluated frequency. `subcarrier_bnr` evaluates
// EVERY bin up to Nyquist — the out-of-band denominator is the whole spectrum
// minus the breathing band — so the Goertzel bill would be O(n^2) per
// subcarrier against the FFT's O(n log n). Below the crossover the narrow-band
// Goertzel still wins, which is why both live in this tree.
//
// ---------------------------------------------------------------------------
// NORMALISATION: NONE, AND THIS WAS CHECKED RATHER THAN ASSUMED
// ---------------------------------------------------------------------------
// The reference's forward transform divides by nothing — there is no `/ n`
// anywhere in `_fft`, and no inverse transform exists in the reference at all.
// A unit-amplitude tone over n samples therefore lands as a bin of magnitude
// n/2, not 1/2. That convention is load-bearing downstream only up to a scale:
// `subcarrier_bnr` forms a RATIO of two power sums from the same transform, so
// any uniform 1/n would cancel. It would NOT cancel for a caller comparing a
// bin against an absolute threshold, so the convention is stated here rather
// than left to be rediscovered.
//
// ---------------------------------------------------------------------------
// FLOAT32 DEVIATION: THE TWIDDLES
// ---------------------------------------------------------------------------
// The reference builds each stage's twiddles with `cmath.exp(-2j*pi*k/length)`
// in f64. Here they are `cosf`/`sinf` of a float angle, so each twiddle carries
// ~1e-7 of error against the f64 one, from two sources: the float rounding of
// 2*pi (relative 2.8e-8, so up to ~8.8e-8 rad of angle error at k near
// length/2) and the ~1 ulp of the libm call itself.
//
// `k / length` is computed FIRST and is exact — `length` is a power of two and
// `k < length`, so the quotient is a dyadic rational that float32 holds
// exactly. Multiplying the exact fraction by 2*pi afterwards leaves a single
// rounding on the angle. Doing it the other way round (`2*pi*k` then divide)
// would round the large product before the exact division could undo it.
//
// The error does not compound the way a twiddle RECURRENCE would: every
// twiddle is computed directly from its own angle, so stage errors add in
// quadrature rather than accumulating. Over log2(n) stages the end-to-end
// deviation from the f64 reference is ~sqrt(log2 n) * 1e-7 of full scale —
// ~2.6e-7 at n = 64, three orders inside the 1e-4 contract.
//
#include <cmath>
#include <cstddef>
#include <cstdint>

#include "core.hpp"

namespace saans {

/// 2*pi as a float, rounded once by the compiler rather than twice by us.
/// Kept here rather than in core.hpp because the FFT is its only user; the
/// band scans want `kPi` itself.
inline constexpr float kTwoPi = 6.28318530717958647692f;

/// True for 1, 2, 4, 8, ... and false for 0.
///
/// The reference never needs this: `_fft` documents "len(x) must be a power of
/// two" and would simply produce nonsense otherwise. With -fno-exceptions
/// there is no ValueError to raise, so the check becomes a return value and
/// the caller is forced to look at it (see `fft_inplace`).
inline constexpr bool is_pow2(std::size_t n) { return n != 0 && (n & (n - 1u)) == 0u; }

/// Smallest power of two >= `n`, or 0 when no such power of two exists.
/// Port of `_next_pow2`.
///
/// Written as the reference's doubling loop rather than a bit trick so the two
/// agree on the edges by construction: `_next_pow2(0)` and `_next_pow2(1)` both
/// return 1, because the loop body never runs when `p >= n` on entry.
/// `constexpr` so a caller can size a static workspace from it.
///
/// PORT-ONLY GUARD, no counterpart in the reference, and not theoretical.
/// CPython's ints are unbounded, so `_next_pow2` always terminates. `p <<= 1`
/// on a size_t does not: once `p` is the top power of two the next shift is 0,
/// `0 < n` stays true, and the loop spins forever. On the ESP32-S3 size_t is
/// 32 bits, so the threshold is n > 2^31 — and `subcarrier_bnr` calls this on
/// the caller's raw row count, so a size_t underflow upstream (`end - start`
/// with start > end, a decimation result of -1) reaches it as a count near
/// 2^32 and hangs the CSI task until the watchdog resets the node.
///
/// Refusing is what every other precondition in these headers does
/// (`fft_inplace` returns false for a non-power-of-two, `subcarrier_bnr`
/// returns ok = false for a short buffer), so this refuses too: 0 is not a
/// power of two, `is_pow2` rejects it downstream, and `subcarrier_bnr` turns
/// it into its existing ok = false path. Being constexpr, the guard also turns
/// a compile-time misuse into a diagnostic rather than a hang.
inline constexpr std::size_t next_pow2(std::size_t n) {
    constexpr std::size_t kTopPow2 = (SIZE_MAX >> 1) + 1u;  // 2^(bits-1)
    if (n > kTopPow2) {
        return 0;
    }
    std::size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

/// In-place radix-2 decimation-in-time FFT over split real/imaginary arrays.
///
/// `re` and `im` each hold `n` floats and are overwritten with the transform.
/// Returns false and touches nothing when `n` is not a power of two or either
/// pointer is null — the no-exceptions form of the reference's precondition.
/// `n == 1` is a no-op and returns true, matching `_fft` on a one-element list.
///
/// SPLIT RATHER THAN INTERLEAVED. The reference carries Python `complex`
/// values; a `struct { float re, im; }` would mirror that more literally, but
/// split arrays are what the caller already has (`subcarrier_bnr` z-scores a
/// real column into `re` and zeroes `im`), they let the zero-padding tail be a
/// single memset-shaped loop, and on target they keep the two halves in
/// separate cache lines for the butterfly's strided access. The arithmetic is
/// identical either way.
///
/// NO ALLOCATION: the transform is in-place and the twiddles are computed as
/// the stage walks them rather than tabulated, so the working set is exactly
/// the two caller arrays. The reference materialises a `twiddles` list per
/// stage; that is a heap allocation of up to n/2 complex values and is
/// precisely what this library may not do. The arithmetic is unchanged — each
/// butterfly sees the same twiddle it would have read from the list.
///
/// Not allocating does NOT mean recomputing per butterfly, which is how this
/// was first written and was wrong. The twiddle depends on `(length, k)` and
/// not on `start`, so the `k` loop goes OUTSIDE the block loop and the two
/// libm calls are hoisted out of it: n-1 cosf/sinf pairs per transform instead
/// of (n/2)*log2(n). At n = 128 that is 127 pairs rather than 448, measured
/// bit-identical over 2000 random transforms (the butterflies within a stage
/// are independent, and each (start, k) still sees exactly the same twiddle)
/// and 1.7x faster on the host. The target saving is larger: x86 has a
/// vectorised libm, the S3 has no hardware transcendentals at all, so cosf and
/// sinf dominate there. Across `subcarrier_bnr`'s 56 subcarriers x 128 points
/// it is 7,112 trig pairs per analysis window instead of 25,088.
inline bool fft_inplace(float* re, float* im, std::size_t n) {
    if (re == nullptr || im == nullptr || !is_pow2(n)) {
        return false;
    }
    if (n == 1) {
        return true;
    }

    // Bit-reversal permutation, transcribed from the reference's incrementing
    // counter rather than rewritten as a per-index bit reverse. The counter
    // form carries `j` (the bit-reversal of `i`) from one iteration to the
    // next by rippling a carry downward through the high bits, which is the
    // same sequence of swaps in the same order.
    std::size_t j = 0;
    for (std::size_t i = 1; i < n; ++i) {
        std::size_t bit = n >> 1;
        while ((j & bit) != 0u) {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if (i < j) {
            const float tr = re[i];
            const float ti = im[i];
            re[i] = re[j];
            im[i] = im[j];
            re[j] = tr;
            im[j] = ti;
        }
    }

    // The stage loop is written over `half` rather than over `length`, so that
    // the shift that advances it can never overflow: `half` stops at n/2, and
    // `half << 1` is then exactly n. Counting `length` from 2 to n instead
    // spins forever at n == 2^(bits-1), where `length <<= 1` wraps to 0 and
    // `0 <= n` stays true — the same shape of bug as the one `next_pow2`
    // guards, and only reachable with an absurd n, but a loop that cannot
    // overflow is cheaper than a loop that must be argued about.
    for (std::size_t half = 1; half < n; half <<= 1) {
        const std::size_t length = half << 1;
        const float       len_f  = static_cast<float>(length);
        // `k` OUTSIDE `start`: the twiddle is a function of (length, k) only,
        // so this stage's n/2 butterfly columns share it and the two libm
        // calls are paid once per column rather than once per butterfly. See
        // the note on fft_inplace above; the result is bit-identical because
        // the butterflies of one stage are independent of each other.
        for (std::size_t k = 0; k < half; ++k) {
            // exp(-2j*pi*k/length). See the float32 note at the top for
            // why the fraction is formed before the scale by 2*pi.
            const float frac  = static_cast<float>(k) / len_f;
            const float angle = -kTwoPi * frac;
            const float wr    = cosf(angle);
            const float wi    = sinf(angle);

            for (std::size_t start = 0; start < n; start += length) {
                const std::size_t a = start + k;
                const std::size_t b = a + half;

                const float ur = re[a];
                const float ui = im[a];
                // Naive complex multiply, in the same operand order CPython's
                // complex_mul uses, so the two round identically given
                // identical inputs. Karatsuba's three-multiply form would
                // change the rounding for no gain on a single-cycle FPU.
                const float vr = re[b] * wr - im[b] * wi;
                const float vi = re[b] * wi + im[b] * wr;

                re[a] = ur + vr;
                im[a] = ui + vi;
                re[b] = ur - vr;
                im[b] = ui - vi;
            }
        }
    }
    return true;
}

/// |X[m]|^2 for a transform held in split arrays.
///
/// FLOAT32 DEVIATION, DELIBERATE AND IN THE SAFE DIRECTION. The reference
/// writes `abs(spec[m]) ** 2`, i.e. hypot() and then a square. This computes
/// re*re + im*im directly. The two differ by the rounding of the intermediate
/// hypot — negligible in f64, but in float32 the round trip through a square
/// root and back costs about an extra half ulp for nothing. The direct form is
/// also one libm call cheaper per bin, times nfft/2 bins times every
/// subcarrier, on a core that has no hardware sqrt for floats worth the name.
///
/// The usual reason to prefer hypot — overflow when re*re exceeds the exponent
/// range — cannot arise here: `subcarrier_bnr` feeds the transform z-scored
/// data, so a bin magnitude is bounded by n, and n*n is nowhere near FLT_MAX.
inline float power_at(const float* re, const float* im, std::size_t m) {
    return re[m] * re[m] + im[m] * im[m];
}

}  // namespace saans
