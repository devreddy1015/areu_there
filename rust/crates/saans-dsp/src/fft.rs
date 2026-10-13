//! Radix-2 FFT, percentiles, a symmetric eigensolver and per-subcarrier
//! breathing-to-noise ratio. Small, exact ports of the helpers in
//! `wifisense/preprocess.py`.

use crate::psum;
use std::f64::consts::PI;

#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct C64 {
    pub re: f64,
    pub im: f64,
}

impl C64 {
    pub const ZERO: C64 = C64 { re: 0.0, im: 0.0 };
    pub fn new(re: f64, im: f64) -> Self {
        Self { re, im }
    }
    pub fn conj(self) -> C64 {
        C64::new(self.re, -self.im)
    }
    /// CPython's abs(complex) is hypot.
    pub fn abs(self) -> f64 {
        self.re.hypot(self.im)
    }
}

impl std::ops::Add for C64 {
    type Output = C64;
    fn add(self, o: C64) -> C64 {
        C64::new(self.re + o.re, self.im + o.im)
    }
}

impl std::ops::Sub for C64 {
    type Output = C64;
    fn sub(self, o: C64) -> C64 {
        C64::new(self.re - o.re, self.im - o.im)
    }
}

impl std::ops::Mul for C64 {
    type Output = C64;
    /// Same operation order as CPython's complex multiply.
    fn mul(self, o: C64) -> C64 {
        C64::new(self.re * o.re - self.im * o.im, self.re * o.im + self.im * o.re)
    }
}

pub fn next_pow2(n: usize) -> usize {
    let mut p = 1;
    while p < n {
        p <<= 1;
    }
    p
}

/// Iterative radix-2 FFT; `x.len()` must be a power of two.
pub fn fft(x: &[C64]) -> Vec<C64> {
    let n = x.len();
    let mut a = x.to_vec();
    let mut j = 0usize;
    for i in 1..n {
        let mut bit = n >> 1;
        while j & bit != 0 {
            j ^= bit;
            bit >>= 1;
        }
        j |= bit;
        if i < j {
            a.swap(i, j);
        }
    }
    let mut length = 2;
    while length <= n {
        let half = length / 2;
        // cmath.exp(-2j*pi*k/length) == (cos x, sin x) with x = -2*pi*k/length
        let twiddles: Vec<C64> = (0..half)
            .map(|k| {
                let x = -2.0 * PI * k as f64 / length as f64;
                C64::new(x.cos(), x.sin())
            })
            .collect();
        let mut start = 0;
        while start < n {
            for k in 0..half {
                let u = a[start + k];
                let v = a[start + k + half] * twiddles[k];
                a[start + k] = u + v;
                a[start + k + half] = u - v;
            }
            start += length;
        }
        length <<= 1;
    }
    a
}

/// numpy-style linear-interpolation percentile.
pub fn percentile(values: &[f64], q: f64) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    let mut s = values.to_vec();
    s.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let pos = (s.len() - 1) as f64 * q / 100.0;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(s.len() - 1);
    s[lo] + (s[hi] - s[lo]) * (pos - lo as f64)
}

/// Cyclic Jacobi eigen-decomposition of a small symmetric matrix.
/// Returns (eigenvalues, vectors) with `vectors[i][j]` = component i of
/// eigenvector j.
pub fn symmetric_eigen(matrix: &[Vec<f64>], max_sweeps: usize) -> (Vec<f64>, Vec<Vec<f64>>) {
    let n = matrix.len();
    let mut a: Vec<Vec<f64>> = matrix.to_vec();
    let mut v: Vec<Vec<f64>> = (0..n).map(|i| (0..n).map(|j| if i == j { 1.0 } else { 0.0 }).collect()).collect();
    for _ in 0..max_sweeps {
        let off = psum((0..n).flat_map(|i| ((i + 1)..n).map(move |j| (i, j))).map(|(i, j)| a[i][j] * a[i][j]));
        let diag = psum((0..n).map(|i| a[i][i] * a[i][i]));
        if off <= 1e-24 * diag.max(1e-300) {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p][q];
                if apq.abs() <= 1e-300 {
                    continue;
                }
                let theta = (a[q][q] - a[p][p]) / (2.0 * apq);
                let mut t = 1.0 / (theta.abs() + (theta * theta + 1.0).sqrt());
                if theta < 0.0 {
                    t = -t;
                }
                let c = 1.0 / (t * t + 1.0).sqrt();
                let s = t * c;
                for row in a.iter_mut() {
                    let (akp, akq) = (row[p], row[q]);
                    row[p] = c * akp - s * akq;
                    row[q] = s * akp + c * akq;
                }
                for k in 0..n {
                    let (apk, aqk) = (a[p][k], a[q][k]);
                    a[p][k] = c * apk - s * aqk;
                    a[q][k] = s * apk + c * aqk;
                }
                for row in v.iter_mut() {
                    let (vkp, vkq) = (row[p], row[q]);
                    row[p] = c * vkp - s * vkq;
                    row[q] = s * vkp + c * vkq;
                }
            }
        }
    }
    ((0..n).map(|i| a[i][i]).collect(), v)
}

/// Per-subcarrier breathing-to-noise *power* ratio over a window
/// (`window[t][k]`), plus each scored subcarrier's in-band spectrum.
/// Constant or non-finite series score 0 and have no spectrum.
pub fn subcarrier_bnr(window: &[Vec<f64>], sample_rate_hz: f64, f_low: f64, f_high: f64) -> (Vec<f64>, Vec<Option<Vec<C64>>>) {
    let n_t = window.len();
    let n_k = window.iter().map(|r| r.len()).min().unwrap_or(0);
    let mut bnr = vec![0.0; n_k];
    let mut spectra: Vec<Option<Vec<C64>>> = vec![None; n_k];
    if n_t < 8 || n_k == 0 || sample_rate_hz <= 0.0 {
        return (bnr, spectra);
    }
    let nfft = next_pow2(n_t);
    let bin_hz = sample_rate_hz / nfft as f64;
    let half = nfft / 2;
    let in_band: Vec<usize> = (1..=half).filter(|&m| f_low <= m as f64 * bin_hz && m as f64 * bin_hz <= f_high).collect();
    let out_band: Vec<usize> = (1..=half).filter(|&m| !(f_low <= m as f64 * bin_hz && m as f64 * bin_hz <= f_high)).collect();
    if in_band.is_empty() || out_band.is_empty() {
        return (bnr, spectra);
    }
    for k in 0..n_k {
        let col: Vec<f64> = window.iter().map(|row| row[k]).collect();
        if !col.iter().all(|x| x.is_finite()) {
            continue;
        }
        let mean = psum(col.iter().copied()) / n_t as f64;
        let std = (psum(col.iter().map(|x| (x - mean) * (x - mean))) / n_t as f64).sqrt();
        if std <= 0.0 {
            continue;
        }
        let mut z: Vec<C64> = col.iter().map(|x| C64::new((x - mean) / std, 0.0)).collect();
        z.resize(nfft, C64::ZERO);
        let spec = fft(&z);
        let power: Vec<f64> = (0..=half).map(|m| {
            let a = spec[m].abs();
            a * a
        }).collect();
        let num = psum(in_band.iter().map(|&m| power[m]));
        let den = psum(out_band.iter().map(|&m| power[m]));
        bnr[k] = if den > 0.0 { num / den } else { f64::INFINITY };
        spectra[k] = Some(in_band.iter().map(|&m| spec[m]).collect());
    }
    (bnr, spectra)
}
