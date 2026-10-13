//! Goertzel band scans with noise-calibrated confidence
//! (port of `wifisense/dsp.py`).

use crate::{argmax_first, psum};
use std::f64::consts::PI;

// Truncated exactly as the Python reference writes it; std's full-precision
// constant would shift confidences by ~1e-12 and break bit-level parity.
#[allow(clippy::approx_constant)]
const EULER_GAMMA: f64 = 0.5772156649;

/// Periodic Hann window.
pub fn hann_window(n: usize) -> Vec<f64> {
    if n <= 1 {
        return vec![1.0; n];
    }
    (0..n).map(|i| 0.5 * (1.0 - (2.0 * PI * i as f64 / n as f64).cos())).collect()
}

/// |X(f)|^2 at an arbitrary frequency (generalised Goertzel).
pub fn goertzel_power(signal: &[f64], freq_hz: f64, sample_rate_hz: f64) -> f64 {
    let n = signal.len();
    if n == 0 || sample_rate_hz <= 0.0 {
        return 0.0;
    }
    let k = freq_hz * n as f64 / sample_rate_hz;
    let w = 2.0 * PI * k / n as f64;
    let cos_w = w.cos();
    let sin_w = w.sin();
    let coeff = 2.0 * cos_w;
    let (mut s1, mut s2) = (0.0f64, 0.0f64);
    for &x in signal {
        let s0 = x + coeff * s1 - s2;
        s2 = s1;
        s1 = s0;
    }
    let real = s1 - s2 * cos_w;
    let imag = s2 * sin_w;
    real * real + imag * imag
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpectralPeak {
    pub freq_hz: f64,
    pub power: f64,
    pub confidence: f64,
}

/// Dominant frequency in [f_low, f_high] with a confidence in [0, 1]
/// calibrated so band-limited noise scores near 0 and a clean tone near 1.
#[allow(clippy::too_many_arguments)]
pub fn band_peak(
    signal: &[f64],
    sample_rate_hz: f64,
    f_low: f64,
    f_high: f64,
    resolution_hz: f64,
    window: bool,
    notch_hz: &[f64],
    notch_width_hz: f64,
) -> Option<SpectralPeak> {
    let n = signal.len();
    if n == 0 || f_low <= 0.0 || f_high <= f_low || resolution_hz <= 0.0 {
        return None;
    }
    let windowed: Vec<f64>;
    let sig: &[f64] = if window {
        let w = hann_window(n);
        windowed = signal.iter().zip(&w).map(|(s, wi)| s * wi).collect();
        &windowed
    } else {
        signal
    };
    // Rounded, not truncated: see the reference (precision-stable grid).
    let steps = crate::py_round((f_high - f_low) / resolution_hz) as i64;
    if steps < 2 {
        return None;
    }
    let is_notched = |f: f64| -> bool {
        if notch_hz.is_empty() || notch_width_hz <= 0.0 {
            return false;
        }
        notch_hz.iter().any(|nf| (f - nf).abs() <= notch_width_hz)
    };
    let mut freqs = Vec::with_capacity(steps as usize + 1);
    let mut powers = Vec::with_capacity(steps as usize + 1);
    let mut candidates = Vec::new();
    for i in 0..=steps as usize {
        let f = f_low + i as f64 * resolution_hz;
        freqs.push(f);
        powers.push(goertzel_power(sig, f, sample_rate_hz));
        if !is_notched(f) {
            candidates.push(i);
        }
    }
    if candidates.is_empty() {
        return None;
    }
    let total = psum(candidates.iter().map(|&i| powers[i]));
    if total <= 0.0 {
        return None;
    }
    let peak_i = candidates[argmax_first(candidates.len(), |j| powers[candidates[j]])?];
    let mut peak_f = freqs[peak_i];
    let peak_p = powers[peak_i];

    // Never interpolate across a notched neighbour (see the reference).
    if peak_i > 0 && peak_i < powers.len() - 1 && !is_notched(freqs[peak_i - 1]) && !is_notched(freqs[peak_i + 1]) {
        let (y0, y1, y2) = (powers[peak_i - 1], powers[peak_i], powers[peak_i + 1]);
        let denom = y0 - 2.0 * y1 + y2;
        if denom != 0.0 {
            let offset = 0.5 * (y0 - y2) / denom;
            if -1.0 < offset && offset < 1.0 {
                peak_f += offset * resolution_hz;
            }
        }
    }

    let mean_power = total / candidates.len() as f64;
    if peak_p <= 0.0 || mean_power <= 0.0 {
        return Some(SpectralPeak { freq_hz: peak_f, power: peak_p, confidence: 0.0 });
    }
    let enbw = if window { 1.5 } else { 1.0 };
    let duration_s = n as f64 / sample_rate_hz;
    let n_ind = ((f_high - f_low) * duration_s).max(2.0);
    let ratio = peak_p / mean_power;
    let noise_ratio = n_ind.ln() + EULER_GAMMA;
    let tone_ratio = n_ind / enbw;
    let span = tone_ratio - noise_ratio;
    let confidence = if span <= 0.0 { 0.0 } else { (ratio - noise_ratio) / span };
    Some(SpectralPeak { freq_hz: peak_f, power: peak_p, confidence: confidence.clamp(0.0, 1.0) })
}

/// Carrier search that requires a symmetric triplet (f - m, f, f + m):
/// recovers a heart rate buried under breathing intermodulation sidebands.
pub fn sideband_peak(
    signal: &[f64],
    sample_rate_hz: f64,
    f_low: f64,
    f_high: f64,
    modulator_hz: f64,
    resolution_hz: f64,
    harmonic_tol_hz: f64,
) -> Option<SpectralPeak> {
    let n = signal.len();
    if n == 0 || modulator_hz <= 0.0 || f_high <= f_low || resolution_hz <= 0.0 {
        return None;
    }
    let w = hann_window(n);
    let windowed: Vec<f64> = signal.iter().zip(&w).map(|(s, wi)| s * wi).collect();
    let power_at = |f: f64| if f > 0.0 { goertzel_power(&windowed, f, sample_rate_hz) } else { 0.0 };
    let is_harmonic = |f: f64| {
        let k = crate::py_round(f / modulator_hz);
        k >= 1.0 && (f - k * modulator_hz).abs() <= harmonic_tol_hz
    };
    // Rounded, not truncated: see the reference (precision-stable grid).
    let steps = crate::py_round((f_high - f_low) / resolution_hz) as i64;
    if steps < 2 {
        return None;
    }
    let mut scored: Vec<(f64, f64)> = Vec::new();
    for i in 0..=steps as usize {
        let f = f_low + i as f64 * resolution_hz;
        if is_harmonic(f) {
            continue;
        }
        let lower = power_at(f - modulator_hz);
        let upper = power_at(f + modulator_hz);
        let score = power_at(f) * (lower.max(0.0) * upper.max(0.0)).sqrt();
        scored.push((f, score));
    }
    if scored.is_empty() {
        return None;
    }
    let bi = argmax_first(scored.len(), |j| scored[j].1)?;
    let (best_f, best_score) = scored[bi];
    if best_score <= 0.0 {
        return None;
    }
    let separation_hz = (modulator_hz * 0.5).max(0.05);
    let runner_up = scored
        .iter()
        .filter(|(f, _)| (f - best_f).abs() > separation_hz)
        .map(|&(_, s)| s)
        .fold(None, |acc: Option<f64>, s| Some(acc.map_or(s, |a| a.max(s))))
        .unwrap_or(0.0);
    let confidence = (1.0 - runner_up / best_score).clamp(0.0, 1.0);
    Some(SpectralPeak { freq_hz: best_f, power: power_at(best_f), confidence })
}
