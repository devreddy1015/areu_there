//! Static-environment suppression and subcarrier fusion
//! (port of `wifisense/preprocess.py`, first half).

use crate::psum;
use crate::types::{CsiFrame, RunningVariance};
use std::f64::consts::PI;

/// EMA smoothing factor giving a highpass corner at `corner_hz`.
pub fn corner_to_alpha(corner_hz: f64, sample_rate_hz: f64) -> f64 {
    assert!(sample_rate_hz > 0.0, "sample_rate_hz must be positive");
    let alpha = 2.0 * PI * corner_hz / sample_rate_hz;
    alpha.clamp(1e-6, 0.999)
}

pub fn alpha_to_corner(alpha: f64, sample_rate_hz: f64) -> f64 {
    alpha * sample_rate_hz / (2.0 * PI)
}

/// Per-subcarrier EMA baseline removal with variance-based weights.
#[derive(Debug, Clone)]
pub struct StaticSuppressor {
    pub n_subcarriers: usize,
    pub sample_rate_hz: f64,
    pub alpha: f64,
    baseline: Vec<f64>,
    seeded: Vec<bool>,
    variance: Vec<RunningVariance>,
}

impl StaticSuppressor {
    pub fn new(n_subcarriers: usize, sample_rate_hz: f64, corner_hz: f64) -> Self {
        assert!(n_subcarriers > 0, "n_subcarriers must be positive");
        Self {
            n_subcarriers,
            sample_rate_hz,
            alpha: corner_to_alpha(corner_hz, sample_rate_hz),
            baseline: vec![0.0; n_subcarriers],
            seeded: vec![false; n_subcarriers],
            variance: vec![RunningVariance::default(); n_subcarriers],
        }
    }

    /// Residuals and weights, both truncated to min(frame, configured).
    pub fn process(&mut self, frame: &CsiFrame) -> (Vec<f64>, Vec<f64>) {
        let n = frame.n_subcarriers().min(self.n_subcarriers);
        if n == 0 {
            return (Vec::new(), Vec::new());
        }
        let mut residuals = vec![0.0; n];
        for i in 0..n {
            let amp = frame.amplitudes[i];
            if !amp.is_finite() {
                residuals[i] = 0.0;
                continue; // the reference skips the variance update too
            }
            if self.seeded[i] {
                residuals[i] = amp - self.baseline[i];
                self.baseline[i] = self.alpha * amp + (1.0 - self.alpha) * self.baseline[i];
            } else {
                self.baseline[i] = amp;
                self.seeded[i] = true;
                residuals[i] = 0.0;
            }
            self.variance[i].update(residuals[i]);
        }
        let w = self.weights(n);
        (residuals, w)
    }

    pub fn weights(&self, n: usize) -> Vec<f64> {
        let count = n.min(self.n_subcarriers);
        let stds: Vec<f64> = (0..count).map(|i| self.variance[i].std()).collect();
        let total = psum(stds.iter().copied());
        if total <= 0.0 {
            return if count > 0 { vec![1.0 / count as f64; count] } else { Vec::new() };
        }
        stds.iter().map(|s| s / total).collect()
    }

    pub fn reset(&mut self) {
        self.baseline.iter_mut().for_each(|b| *b = 0.0);
        self.seeded.iter_mut().for_each(|s| *s = false);
        self.variance.iter_mut().for_each(|v| v.reset());
    }
}

/// Weighted collapse of residuals to one scalar, renormalised over the
/// subcarriers present.
pub fn fuse(residuals: &[f64], weights: &[f64]) -> f64 {
    let n = residuals.len().min(weights.len());
    if n == 0 {
        return 0.0;
    }
    let total_w = psum(weights[..n].iter().copied());
    if total_w <= 0.0 {
        return psum(residuals[..n].iter().copied()) / n as f64;
    }
    psum(residuals[..n].iter().zip(&weights[..n]).map(|(r, w)| r * w)) / total_w
}
