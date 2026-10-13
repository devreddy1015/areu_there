//! Saans signal processing.
//!
//! A line-for-line port of the Python reference package `wifisense`
//! (ADR-002). Every stage reproduces the reference's golden vectors in
//! `spec/golden/` to within 1e-9 relative (ADR-003); `tests/golden.rs`
//! enforces it. Where Python semantics are subtle, this crate copies them
//! deliberately rather than "improving" on them:
//!
//! - `sum()` of floats in CPython >= 3.12 is Neumaier-compensated, so
//!   every place the reference calls `sum` uses [`psum`] here.
//! - `round()` rounds half to even ([`py_round`]).
//! - `max(iterable, key=...)` returns the *first* maximum ([`argmax_first`]).
//!
//! No dependencies, no unsafe. The hot paths allocate only when windows
//! are read for spectral analysis.

// Index loops mirror the reference line for line; keeping that
// correspondence readable matters more here than iterator style.
#![allow(clippy::needless_range_loop)]

pub mod biquad;
pub mod features;
pub mod fft;
pub mod gapfill;
pub mod pipeline;
pub mod preprocess;
pub mod presence;
pub mod respirfi;
pub mod spectral;
pub mod types;
pub mod vitals;

pub use biquad::BiquadBandpass;
pub use features::{FeatureExtractor, FeatureExtractorConfig, FeatureFrame};
pub use gapfill::SequenceGapFiller;
pub use pipeline::{SensingPipeline, SubcarrierSelection};
pub use presence::{PresenceDetector, PresenceState};
pub use types::{CsiFrame, VitalEstimate, VitalReading, VitalStatus};
pub use vitals::{BreathingExtractor, HeartRateExtractor};

/// CPython's float `sum()`: Neumaier compensated summation.
///
/// Mirrors `Python/bltinmodule.c` (3.12+): the compensation term is added
/// back only if it is non-zero and finite.
pub fn psum<I: IntoIterator<Item = f64>>(xs: I) -> f64 {
    let mut f = 0.0f64;
    let mut c = 0.0f64;
    for x in xs {
        let t = f + x;
        if f.abs() >= x.abs() {
            c += (f - t) + x;
        } else {
            c += (x - t) + f;
        }
        f = t;
    }
    if c != 0.0 && c.is_finite() {
        f += c;
    }
    f
}

/// Python's `round(x)` for floats: half to even.
pub fn py_round(x: f64) -> f64 {
    x.round_ties_even()
}

/// Index of the first maximum, as Python's `max(range(n), key=f)` picks.
pub fn argmax_first<F: Fn(usize) -> f64>(n: usize, key: F) -> Option<usize> {
    let mut best: Option<(usize, f64)> = None;
    for i in 0..n {
        let v = key(i);
        match best {
            None => best = Some((i, v)),
            Some((_, b)) if v > b => best = Some((i, v)),
            _ => {}
        }
    }
    best.map(|(i, _)| i)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn psum_compensates_like_cpython() {
        // 1e16 + 1 + -1e16: naive left-to-right gives 0, CPython gives 1.
        assert_eq!(psum([1e16, 1.0, -1e16]), 1.0);
        assert_eq!(psum(std::iter::empty()), 0.0);
    }

    #[test]
    fn round_half_even() {
        assert_eq!(py_round(2.5), 2.0);
        assert_eq!(py_round(3.5), 4.0);
        assert_eq!(py_round(-0.5), -0.0);
    }

    #[test]
    fn argmax_takes_first_of_ties() {
        let v = [1.0, 3.0, 3.0, 2.0];
        assert_eq!(argmax_first(4, |i| v[i]), Some(1));
    }
}
