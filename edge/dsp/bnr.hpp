// SPDX-License-Identifier: MIT
#pragma once
//
// Saans edge DSP — per-subcarrier breathing-to-noise ratio.
//
// Port of `subcarrier_bnr` in `wifisense/preprocess.py`. Header-only, no
// allocation after init, float32 throughout.
//
//   "Each subcarrier's series is z-scored, zero-padded to a power of two and
//    transformed; BNR is in-band power over out-of-band power (DC excluded)."
//
// This is the link-quality number the node reports as `best_bnr` in
// FeatureFrame v1 (spec/wire.md), and the first stage of the RespirFi
// subcarrier selection. A WiFi link has ~56 usable subcarriers and they do not
// see the same thing: a body at the wrong distance can null one subcarrier
// completely while the one 312 kHz away is the clearest trace on the link.
// Ranking them by how much of their variance sits in the breathing band is how
// the pipeline finds the good ones without being told where the person is.
//
// ---------------------------------------------------------------------------
// POWER, NOT MAGNITUDE. DO NOT "SIMPLIFY" THIS.
// ---------------------------------------------------------------------------
// The ratio is of |X|^2 sums, not |X| sums. This was a real bug in this
// project's history, fixed upstream, and it is worth stating why the magnitude
// version looks fine and is not: the denominator runs over ~51 out-of-band bins
// against ~13 in-band ones, so with magnitudes the denominator is dominated by
// the BIN COUNT rather than by the energy, and the ratio reports the shape of
// the band split instead of the signal. In power the comparison is between two
// energies and Parseval makes it mean something.
//
// ---------------------------------------------------------------------------
// DC IS EXCLUDED, AND den == 0 IS +inf ON PURPOSE
// ---------------------------------------------------------------------------
// Bins run from 1, never 0. Bin 0 is the mean, and the series was just
// z-scored, so bin 0 carries only the residue of that subtraction — including,
// in float32, the rounding of the mean itself. Counting it would put a pure
// arithmetic artefact in the numerator or denominator depending on where zero
// frequency happens to fall relative to f_low.
//
// `den == 0` yields +inf, matching `float("inf")` in the reference. That is the
// honest answer — a series with energy only in the breathing band has an
// infinite ratio — and the reference's callers filter non-finite values before
// `max()`. Returning a large finite sentinel instead would silently win every
// comparison it is supposed to be excluded from.
//
// It is reachable, and the test suite reaches it: column [1,0,-1,0,1,0,-1,0]
// at 8 Hz over 0.5..2.5 Hz transforms to exactly zero at every out-of-band bin
// in float32, so `den` is exactly 0.0f and the reference returns `inf` for the
// same input. An earlier version of this file claimed the path was unreachable
// with a finite window; it is not.
//
// ---------------------------------------------------------------------------
// THE TWO FLOAT32 CLASSIFICATION HAZARDS
// ---------------------------------------------------------------------------
// Both are places where float32 decides a yes/no that f64 decides differently,
// so no tolerance can paper over either:
//
//   1. A CONSTANT SERIES. `sd > 0` is not a reliable way to recognise one in
//      float32, because the mean of n copies of c does not round-trip. Handled
//      by an exact equality scan before the two passes; see the long note in
//      `subcarrier_bnr`.
//
//   2. A BIN CENTRE ON A BAND EDGE. `f_low <= m*bin_hz <= f_high` moves a whole
//      bin's power between numerator and denominator. Safe on the pipeline's
//      own grid, where the nearest approach to an edge is 20% of a bin; see
//      `bnr_band_bins`.
//
#include <cmath>
#include <cstddef>

#include "core.hpp"
#include "fft.hpp"

namespace saans {

/// Scratch for one `subcarrier_bnr` call over a window of at most `MaxRows`
/// time samples. Caller-owned; `kNfft` is the zero-padded transform length.
///
/// At the pipeline's 32 s / 4 Hz analysis window (128 rows) this is 1 KiB.
template <std::size_t MaxRows>
struct BnrWorkspace {
    static constexpr std::size_t kNfft = next_pow2(MaxRows);
    float                        re[kNfft]{};
    float                        im[kNfft]{};
    static constexpr std::size_t capacity() { return kNfft; }
};

/// Optional extras. The defaults reproduce the reference's defaults exactly.
struct SubcarrierBnrParams {
    float f_low  = 0.1f;
    float f_high = 0.5f;

    /// OPTIONAL in-band spectra — the `spectra` half of the reference's
    /// `(bnr, spectra)` return, which `respirfi_select` needs for its cosine
    /// similarities. Leave null and the transform is still only done once; the
    /// bins are simply not copied out.
    ///
    /// Layout is `spectra_re[k * spectra_stride + j]`, j over the in-band bins
    /// in increasing frequency order, matching `[spec[m] for m in in_band]`.
    /// `spectra_stride` must be at least `n_in_band` (which the result reports,
    /// and which a caller can precompute with `bnr_band_bins`).
    ///
    /// The reference's `spectra` is a dict keyed only by the subcarriers it
    /// computed, so "absent" is distinguishable from "all zeros"; a dense array
    /// cannot express that, hence `spectra_present`.
    float*      spectra_re      = nullptr;
    float*      spectra_im      = nullptr;
    std::size_t spectra_stride  = 0;
    bool*       spectra_present = nullptr;  ///< n_cols flags, false = skipped
};

struct SubcarrierBnrResult {
    /// False only for a PORT-level failure — a null pointer or a buffer too
    /// small. The reference's own "insufficient data" guards (fewer than 8
    /// rows, no subcarriers, a non-positive rate, or a band split with one
    /// side empty) return an all-zeros ratio list, so they report `ok = true`
    /// with zeros written, exactly as the reference does. A too-small buffer
    /// must never be mistaken for a legitimate zero.
    bool        ok             = false;
    bool        ran            = false;  ///< false when a reference guard fired
    std::size_t n_subcarriers  = 0;      ///< bnr values written
    std::size_t nfft           = 0;
    std::size_t n_in_band      = 0;
    std::size_t n_out_band     = 0;
    std::size_t n_scored       = 0;      ///< subcarriers that produced a ratio
    float       bin_hz         = 0.0f;
};

/// How many bins land in and out of the band, for a caller sizing a spectra
/// buffer before the call. Same arithmetic as the scan below, kept in one place
/// so the two cannot disagree.
///
/// FLOAT32 NOTE — THIS IS A THRESHOLD COMPARE. `f_low <= m*bin_hz <= f_high` is
/// a classification, not a quantity: a bin whose centre sits exactly on a band
/// edge can fall to the other side in float32 than it does in f64, and that
/// moves a whole bin's power from the numerator to the denominator. There is no
/// tolerance that hides it, so it is called out rather than smoothed over.
///
/// In practice it needs an exact coincidence to bite. `bin_hz` is
/// `sample_rate_hz / nfft` with nfft a power of two, so the division is exact
/// and `m * bin_hz` is exact whenever the rate is; at the pipeline's 4 Hz / 128
/// the grid is multiples of 0.03125 and the nearest approach to the 0.1 Hz edge
/// is 0.09375 — 20% of a bin away, where a 1e-7 difference cannot matter.
/// A rate that makes a bin centre land on an edge to within 1e-7 relative would
/// be the thing to re-examine, and the edges would be the thing to nudge.
inline void bnr_band_bins(std::size_t nfft, float sample_rate_hz, float f_low, float f_high,
                          std::size_t* n_in, std::size_t* n_out, float* bin_hz_out) {
    const std::size_t half   = nfft / 2u;
    const float       bin_hz = sample_rate_hz / static_cast<float>(nfft);
    std::size_t       in_n   = 0;
    std::size_t       out_n  = 0;
    for (std::size_t m = 1; m <= half; ++m) {
        const float f = static_cast<float>(m) * bin_hz;
        if (f >= f_low && f <= f_high) {
            ++in_n;
        } else {
            ++out_n;
        }
    }
    if (n_in != nullptr) {
        *n_in = in_n;
    }
    if (n_out != nullptr) {
        *n_out = out_n;
    }
    if (bin_hz_out != nullptr) {
        *bin_hz_out = bin_hz;
    }
}

namespace detail {

/// HOW THE WINDOW IS ADDRESSED, and why there are two of these.
///
/// `subcarrier_bnr` was written against one contiguous rectangle, which is the
/// right shape for a caller that already has one. `FeatureExtractor`
/// (feature_extractor.hpp) does not: its block-averaged rows live in a ring
/// whose storage wraps, exactly as the reference's `deque(maxlen=...)` does.
/// Flattening the ring into a second buffer on every scoring pass would cost
/// another `MaxBnrRows * MaxSubcarriers` floats of SRAM — 30 KB at the node's
/// budget, to copy data that is already in RAM in the right order, just not at
/// consecutive addresses.
///
/// So the scan is parameterised on how a row is found. Both accessors are
/// empty-but-for-two-words structs with an inlinable `row()`, so the generated
/// code for the strided form is what it was before this split; nothing is
/// virtual and nothing is indirect at runtime beyond the one load that the
/// pointer form needs by definition.
struct StridedRows {
    const float* base;
    std::size_t  stride;
    const float* row(std::size_t t) const { return base + t * stride; }
};

/// `rows[t]` points at subcarrier 0 of sample `t`, oldest first.
struct PointerRows {
    const float* const* rows;
    const float* row(std::size_t t) const { return rows[t]; }
};

/// The body of `subcarrier_bnr`. Pointer validity and the row-addressing
/// precondition belong to the wrappers, which know what shape they were given.
template <typename Rows>
inline SubcarrierBnrResult subcarrier_bnr_impl(Rows rows, std::size_t n_rows,
                                               std::size_t n_cols, float sample_rate_hz,
                                               float* bnr_out, std::size_t bnr_capacity,
                                               float* scratch_re, float* scratch_im,
                                               std::size_t scratch_capacity,
                                               const SubcarrierBnrParams& params) {
    SubcarrierBnrResult r;
    if (bnr_out == nullptr || scratch_re == nullptr || scratch_im == nullptr ||
        bnr_capacity < n_cols) {
        return r;  // ok = false: nothing written, nothing implied
    }
    const bool want_spectra = params.spectra_re != nullptr || params.spectra_im != nullptr;
    if (want_spectra && (params.spectra_re == nullptr || params.spectra_im == nullptr)) {
        return r;  // half a spectra buffer is a caller bug, not a degraded mode
    }

    // `bnr = [0.0] * n_k` up front: every early return below still hands back a
    // fully written array, as the reference does.
    for (std::size_t k = 0; k < n_cols; ++k) {
        bnr_out[k] = 0.0f;
    }
    if (params.spectra_present != nullptr) {
        for (std::size_t k = 0; k < n_cols; ++k) {
            params.spectra_present[k] = false;
        }
    }
    r.ok            = true;
    r.n_subcarriers = n_cols;

    // `if n_t < 8 or n_k == 0 or sample_rate_hz <= 0.0`
    if (n_rows < 8u || n_cols == 0u || !(sample_rate_hz > 0.0f)) {
        return r;  // ran = false
    }

    const std::size_t nfft = next_pow2(n_rows);
    // `nfft == 0` is next_pow2's refusal for a row count with no power of two
    // above it — an upstream size_t underflow, not a window. Folding it into
    // the capacity check means a bad length comes back as ok = false rather
    // than as a hang inside the doubling loop; see next_pow2 in fft.hpp.
    if (nfft == 0u || scratch_capacity < nfft) {
        r.ok = false;  // too small to transform: refuse, do not half-report
        return r;
    }
    r.nfft = nfft;
    bnr_band_bins(nfft, sample_rate_hz, params.f_low, params.f_high, &r.n_in_band,
                  &r.n_out_band, &r.bin_hz);

    // `if not in_band or not out_band: return bnr, spectra`. Either way round
    // the ratio would be meaningless: no numerator, or a zero denominator for
    // every subcarrier at once.
    if (r.n_in_band == 0u || r.n_out_band == 0u) {
        return r;
    }
    if (want_spectra && params.spectra_stride < r.n_in_band) {
        r.ok = false;
        return r;
    }

    const std::size_t half  = nfft / 2u;
    const float       n_t_f = static_cast<float>(n_rows);

    for (std::size_t k = 0; k < n_cols; ++k) {
        // `col = [window[t][k] for t in range(n_t)]`, read in place.
        bool finite = true;
        for (std::size_t t = 0; t < n_rows && finite; ++t) {
            finite = is_finite(rows.row(t)[k]);
        }
        if (!finite) {
            continue;  // left at 0
        }

        // ------------------------------------------------------------------
        // A CONSTANT SERIES IS SKIPPED BY INSPECTION, NOT BY ITS std
        // ------------------------------------------------------------------
        // The reference reaches the same decision through `if std <= 0.0`, and
        // for a constant column in f64 that test is exact: the input is
        // float32-representable (it came off the wire, or through this very
        // port), so `n_t * c` needs at most 24 + log2(n_t) bits and lands in a
        // f64 mantissa without rounding, `sum(col) / n_t` returns `c` back
        // exactly, every deviation is exactly 0 and std is exactly 0.
        //
        // In float32 that round trip does NOT close. c = 31.6778717f over the
        // pipeline's own 120-row window sums to 3801.34473f, which divides to
        // 31.6778736f — one ulp high — so every deviation is a constant
        // -1.907e-06, `sd > 0` passes, the column z-scores to a constant -1,
        // and the transform of a 120-of-128 boxcar hands back a fabricated
        // ratio of 1.10416269. Measured over 200000 amplitudes in 0.5..60.5 at
        // 120 rows, 6.3% of constant columns do this; at 20 rows, 13.4%. Only
        // power-of-two row counts are safe, because only then is the division
        // a shift.
        //
        // This is a CLASSIFICATION divergence, not a tolerance one — no
        // tolerance hides "0.0 versus 1.10416269", and `best_bnr` is a max()
        // over the finite values (features.py), so a room of dead subcarriers
        // would report link quality 1.10 instead of 0. An exact equality scan
        // decides it the way the reference's f64 arithmetic decides it, for
        // every float32-representable input, and short-circuits the two passes
        // below rather than costing anything.
        //
        // Deliberately NOT a tolerance (`sd > eps * |mean|`): that would also
        // skip columns the reference scores, trading this divergence for a
        // larger one. Only an EXACTLY constant column is affected here, and
        // only an exactly constant column is skipped.
        bool constant = true;
        for (std::size_t t = 1; t < n_rows && constant; ++t) {
            constant = rows.row(t)[k] == rows.row(0)[k];
        }
        if (constant) {
            continue;  // `if std <= 0.0: continue` — left at 0, as a skip
        }

        // mean = sum(col) / n_t, with the compensated sum CPython 3.12's
        // builtin gives the reference. See core.hpp for why that is not
        // optional in float32.
        NeumaierSum acc;
        for (std::size_t t = 0; t < n_rows; ++t) {
            acc.add(rows.row(t)[k]);
        }
        const float mean = acc.value() / n_t_f;

        // std = sqrt(sum((x - mean)^2) / n_t). Two-pass, as the reference
        // writes it, and that choice matters more here than it does in f64: a
        // one-pass sum-of-squares-minus-square-of-sum would cancel
        // catastrophically for a subcarrier whose mean dwarfs its modulation,
        // which is the normal case for a raw CSI amplitude.
        NeumaierSum var;
        for (std::size_t t = 0; t < n_rows; ++t) {
            const float d = rows.row(t)[k] - mean;
            var.add(d * d);
        }
        const float mean_sq = var.value() / n_t_f;
        const float sd      = sqrtf(mean_sq < 0.0f ? 0.0f : mean_sq);
        if (!(sd > 0.0f)) {
            continue;  // `if std <= 0.0: continue` — a constant series, left at 0
        }

        // z-score into the transform buffer, then zero-pad to nfft.
        const float inv_sd = 1.0f / sd;
        for (std::size_t t = 0; t < n_rows; ++t) {
            // FLOAT32 DEVIATION, DELIBERATE. The reference divides:
            // `(x - mean) / std`. Multiplying by a reciprocal computed once
            // removes n_t divisions for one extra rounding, and it is worth
            // being precise about what that rounding costs, because the
            // tempting claim — "a uniform scale on the whole column, and BNR
            // is a ratio, so it cancels exactly" — is only half true.
            //
            // The UNIFORM part does cancel: `fl(1/sd)` is one fixed factor and
            // numerator and denominator are both quadratic in it. What does
            // not cancel is the per-element rounding of `fl(d * fl(1/sd))`
            // against `fl(d / sd)`, which is an independent half-ulp per
            // sample, not a shared factor. Measured on the golden bnr window:
            // 15 to 68 of the 120 samples differ per subcarrier and the
            // element-wise ratio spans ~1.2e-07 rather than sitting at a
            // constant, and the BNR that comes out differs from the dividing
            // form by up to 1.7e-07 relative (subcarrier 1: 5.66718864 against
            // 5.66718769).
            //
            // So: comparable to the division's own rounding, three orders
            // inside the 1e-4 contract, and the same order as the port's total
            // deviation from the f64 reference for those subcarriers — which
            // is the honest way to state it. It is not zero, and anyone
            // budgeting the error should not be told that it is.
            scratch_re[t] = (rows.row(t)[k] - mean) * inv_sd;
            scratch_im[t] = 0.0f;
        }
        for (std::size_t t = n_rows; t < nfft; ++t) {
            scratch_re[t] = 0.0f;
            scratch_im[t] = 0.0f;
        }
        if (!fft_inplace(scratch_re, scratch_im, nfft)) {
            r.ok = false;
            return r;  // unreachable: nfft is a power of two by construction
        }

        // num/den accumulated in one sweep over m = 1..half, in increasing m,
        // which is the order the reference's two comprehensions produce. The
        // sums are compensated for the same reason the mean is: ~51 terms of
        // similar size, divided into each other, against a 1e-4 contract.
        NeumaierSum num;
        NeumaierSum den;
        std::size_t written = 0;
        for (std::size_t m = 1; m <= half; ++m) {
            const float p = power_at(scratch_re, scratch_im, m);
            const float f = static_cast<float>(m) * r.bin_hz;
            if (f >= params.f_low && f <= params.f_high) {
                num.add(p);
                if (want_spectra) {
                    params.spectra_re[k * params.spectra_stride + written] = scratch_re[m];
                    params.spectra_im[k * params.spectra_stride + written] = scratch_im[m];
                    ++written;
                }
            } else {
                den.add(p);
            }
        }

        const float den_v = den.value();
        // `num / den if den > 0.0 else float("inf")`.
        bnr_out[k] = (den_v > 0.0f) ? (num.value() / den_v) : INFINITY;
        if (params.spectra_present != nullptr) {
            params.spectra_present[k] = true;
        }
        ++r.n_scored;
    }

    r.ran = true;
    return r;
}

}  // namespace detail

/// Per-subcarrier breathing-to-noise power ratio.
///
/// `window` is row-major: sample `t` of subcarrier `k` is
/// `window[t * row_stride + k]`, for `t < n_rows` and `k < n_cols`. The
/// reference takes a list of rows and uses `min(len(row))` as the subcarrier
/// count; a rectangular view makes a ragged window unrepresentable, which is
/// the same thing that `min` was defending against.
///
/// Writes `n_cols` ratios to `bnr_out`. A subcarrier whose series is
/// non-finite, or constant (`std <= 0`), is LEFT AT ZERO and skipped — "the
/// honest answer for an empty room or a whole-band blind spot" — and its
/// `spectra_present` flag stays false.
///
/// `scratch_re` / `scratch_im` must each hold `next_pow2(n_rows)` floats; see
/// `BnrWorkspace`. They are clobbered.
inline SubcarrierBnrResult subcarrier_bnr(const float* window, std::size_t n_rows,
                                          std::size_t n_cols, std::size_t row_stride,
                                          float sample_rate_hz, float* bnr_out,
                                          std::size_t bnr_capacity, float* scratch_re,
                                          float* scratch_im, std::size_t scratch_capacity,
                                          const SubcarrierBnrParams& params = {}) {
    if (window == nullptr || row_stride < n_cols) {
        return SubcarrierBnrResult{};  // ok = false: nothing written, nothing implied
    }
    return detail::subcarrier_bnr_impl(detail::StridedRows{window, row_stride}, n_rows, n_cols,
                                       sample_rate_hz, bnr_out, bnr_capacity, scratch_re,
                                       scratch_im, scratch_capacity, params);
}

/// The same scan over a window whose rows are not contiguous with each other.
///
/// `rows[t]` points at subcarrier 0 of sample `t`, OLDEST FIRST, and each row
/// must hold at least `n_cols` floats — the same contract every other buffer in
/// this library carries. Exists for `FeatureExtractor`, whose block-averaged
/// rows live in a ring; see `detail::PointerRows` for why that is worth an
/// entry point of its own rather than a flattening copy.
///
/// Identical arithmetic in identical order to the strided form: the two share
/// one body, so they cannot drift apart.
inline SubcarrierBnrResult subcarrier_bnr_rows(const float* const* rows, std::size_t n_rows,
                                               std::size_t n_cols, float sample_rate_hz,
                                               float* bnr_out, std::size_t bnr_capacity,
                                               float* scratch_re, float* scratch_im,
                                               std::size_t scratch_capacity,
                                               const SubcarrierBnrParams& params = {}) {
    if (rows == nullptr) {
        return SubcarrierBnrResult{};
    }
    return detail::subcarrier_bnr_impl(detail::PointerRows{rows}, n_rows, n_cols, sample_rate_hz,
                                       bnr_out, bnr_capacity, scratch_re, scratch_im,
                                       scratch_capacity, params);
}

}  // namespace saans
