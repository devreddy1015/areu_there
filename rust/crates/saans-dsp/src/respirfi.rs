//! RespirFi subcarrier selection and streaming signed fusion
//! (port of the second half of `wifisense/preprocess.py`; see the long
//! comment there for the paper, deviations and evidence).

use crate::fft::{percentile, subcarrier_bnr, symmetric_eigen, C64};
use crate::preprocess::fuse;
use crate::{argmax_first, psum, py_round};
use std::collections::{BTreeMap, VecDeque};

#[derive(Debug, Clone, Copy)]
pub struct SelectParams {
    pub bnr_floor: f64,
    pub bnr_percentile: f64,
    /// None derives 2/sqrt(2BT) from the window.
    pub affinity_threshold: Option<f64>,
    pub keep_fraction: f64,
}

impl Default for SelectParams {
    fn default() -> Self {
        Self { bnr_floor: 2.0, bnr_percentile: 80.0, affinity_threshold: None, keep_fraction: 0.5 }
    }
}

#[derive(Debug, Clone)]
pub struct Selection {
    /// subcarrier -> +1 / -1, retained subcarriers only, ascending.
    pub signs: BTreeMap<usize, i8>,
    pub bnr: Vec<f64>,
    pub bnr_threshold: f64,
    pub affinity_threshold: f64,
    pub candidates: Vec<usize>,
}

/// Pick and sign subcarriers from `window[t][k]` sampled at `sample_rate_hz`.
pub fn respirfi_select(window: &[Vec<f64>], sample_rate_hz: f64, f_low: f64, f_high: f64, p: SelectParams) -> Option<Selection> {
    if window.is_empty() || window[0].is_empty() {
        return None;
    }
    let n_t = window.len();
    let n_k = window.iter().map(|r| r.len()).min().unwrap_or(0);
    if n_t < 8 || sample_rate_hz <= 0.0 {
        return None;
    }
    let (bnr, spectra) = subcarrier_bnr(window, sample_rate_hz, f_low, f_high);
    if spectra.iter().all(|s| s.is_none()) {
        return None;
    }
    let eta = percentile(&bnr, p.bnr_percentile).max(p.bnr_floor);
    let candidates: Vec<usize> = (0..n_k).filter(|&k| bnr[k] >= eta && spectra[k].is_some()).collect();
    let affinity_threshold = p.affinity_threshold.unwrap_or_else(|| {
        let n_ind = 2.0 * (f_high - f_low) * n_t as f64 / sample_rate_hz;
        (2.0 / n_ind.max(1.0).sqrt()).min(0.9)
    });
    if candidates.is_empty() {
        return None;
    }
    if candidates.len() == 1 {
        let mut signs = BTreeMap::new();
        signs.insert(candidates[0], 1);
        return Some(Selection { signs, bnr, bnr_threshold: eta, affinity_threshold, candidates });
    }
    let spec = |k: usize| -> &Vec<C64> { spectra[k].as_ref().expect("candidate has spectrum") };
    let energy: Vec<f64> = candidates.iter().map(|&k| psum(spec(k).iter().map(|c| {
        let a = c.abs();
        a * a
    })).sqrt()).collect();
    let m = candidates.len();
    let mut sim = vec![vec![0.0; m]; m];
    for i in 0..m {
        sim[i][i] = 1.0;
        for j in (i + 1)..m {
            let (a, b) = (spec(candidates[i]), spec(candidates[j]));
            let dot = psum(a.iter().zip(b).map(|(x, y)| (*x * y.conj()).re));
            let denom = energy[i] * energy[j];
            let s = if denom > 0.0 { dot / denom } else { 0.0 };
            sim[i][j] = s;
            sim[j][i] = s;
        }
    }
    let tern: Vec<Vec<f64>> = sim
        .iter()
        .map(|row| row.iter().map(|&s| if s >= affinity_threshold { 1.0 } else if s <= -affinity_threshold { -1.0 } else { 0.0 }).collect())
        .collect();
    let (values, vectors) = symmetric_eigen(&tern, 60);
    let lead = argmax_first(m, |i| values[i])?;
    let mut x: Vec<i32> = (0..m)
        .map(|i| if vectors[i][lead] > 0.0 { 1 } else if vectors[i][lead] < 0.0 { -1 } else { 0 })
        .collect();
    for i in 0..m {
        if x[i] == 0 {
            let pull = psum((0..m).filter(|&j| j != i).map(|j| sim[i][j] * x[j] as f64));
            x[i] = if pull >= 0.0 { 1 } else { -1 };
        }
    }
    let scores: Vec<f64> = (0..m)
        .map(|i| psum((0..m).filter(|&j| j != i).map(|j| sim[i][j] * (x[i] * x[j]) as f64)))
        .collect();
    let best = scores.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if best <= 0.0 {
        return None;
    }
    let keep: Vec<usize> = (0..m).filter(|&i| scores[i] > 0.0 && scores[i] >= p.keep_fraction * best).collect();
    let plus = psum(keep.iter().filter(|&&i| x[i] > 0).map(|&i| scores[i]));
    let minus = psum(keep.iter().filter(|&&i| x[i] < 0).map(|&i| scores[i]));
    let flip = if minus > plus { -1 } else { 1 };
    let signs = keep.iter().map(|&i| (candidates[i], (flip * x[i]) as i8)).collect();
    Some(Selection { signs, bnr, bnr_threshold: eta, affinity_threshold, candidates })
}

/// Streaming wrapper: block-average residuals, re-select every `update_s`,
/// fuse every frame with signed weights, keep orientation continuous.
#[derive(Debug, Clone)]
pub struct RespirFiFusion {
    pub n_subcarriers: usize,
    pub sample_rate_hz: f64,
    pub f_low: f64,
    pub f_high: f64,
    pub decimation: usize,
    pub analysis_rate_hz: f64,
    pub params: SelectParams,
    window_len: usize,
    update_every: u64,
    buffer: VecDeque<Vec<f64>>,
    acc: Vec<f64>,
    acc_n: usize,
    frames: u64,
    weights: Option<Vec<f64>>,
    pub selection: Option<Selection>,
    pub passes: u64,
    pub failed_passes: u64,
}

impl RespirFiFusion {
    pub fn new(n_subcarriers: usize, sample_rate_hz: f64, params: SelectParams) -> Self {
        Self::with_options(n_subcarriers, sample_rate_hz, 0.1, 0.5, None, 5.0, 4.0, params)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        n_subcarriers: usize,
        sample_rate_hz: f64,
        f_low: f64,
        f_high: f64,
        window_s: Option<f64>,
        update_s: f64,
        analysis_rate_hz: f64,
        params: SelectParams,
    ) -> Self {
        assert!(n_subcarriers > 0 && sample_rate_hz > 0.0);
        let window_s = window_s.unwrap_or(3.0 / f_low);
        let decimation = (py_round(sample_rate_hz / analysis_rate_hz) as usize).max(1);
        let analysis = sample_rate_hz / decimation as f64;
        assert!(f_high < analysis / 2.0, "analysis rate {analysis} Hz cannot represent {f_high} Hz");
        Self {
            n_subcarriers,
            sample_rate_hz,
            f_low,
            f_high,
            decimation,
            analysis_rate_hz: analysis,
            params,
            window_len: ((window_s * analysis) as usize).max(8),
            update_every: (py_round(update_s * sample_rate_hz) as u64).max(1),
            buffer: VecDeque::new(),
            acc: vec![0.0; n_subcarriers],
            acc_n: 0,
            frames: 0,
            weights: None,
            selection: None,
            passes: 0,
            failed_passes: 0,
        }
    }

    pub fn weights(&self) -> Option<&[f64]> {
        self.weights.as_deref()
    }

    pub fn process(&mut self, residuals: &[f64], fallback_weights: &[f64]) -> f64 {
        let n = residuals.len().min(self.n_subcarriers);
        for k in 0..n {
            let r = residuals[k];
            self.acc[k] += if r.is_finite() { r } else { 0.0 };
        }
        self.acc_n += 1;
        if self.acc_n >= self.decimation {
            let row: Vec<f64> = self.acc.iter().map(|a| a / self.acc_n as f64).collect();
            if self.buffer.len() == self.window_len {
                self.buffer.pop_front();
            }
            self.buffer.push_back(row);
            self.acc.iter_mut().for_each(|a| *a = 0.0);
            self.acc_n = 0;
        }
        self.frames += 1;
        if self.frames.is_multiple_of(self.update_every) && self.buffer.len() >= self.window_len {
            self.reselect(fallback_weights);
        }
        match &self.weights {
            None => fuse(residuals, fallback_weights),
            Some(w) => psum(w[..n].iter().zip(&residuals[..n]).filter(|(_, r)| r.is_finite()).map(|(w, r)| w * r)),
        }
    }

    fn reselect(&mut self, fallback_weights: &[f64]) {
        let window: Vec<Vec<f64>> = self.buffer.iter().cloned().collect();
        self.passes += 1;
        let Some(mut new) = self.compute_weights(&window) else {
            self.failed_passes += 1;
            return;
        };
        let n = self.n_subcarriers;
        let reference: Vec<f64> = match &self.weights {
            Some(w) => w.clone(),
            None => {
                let m = fallback_weights.len().min(n);
                let total = psum(fallback_weights[..m].iter().copied());
                if total > 0.0 {
                    fallback_weights[..m].iter().map(|w| w / total).collect()
                } else {
                    vec![1.0 / n as f64; n]
                }
            }
        };
        let agreement = psum(window.iter().map(|row| {
            let a = psum(new.iter().zip(row).map(|(w, x)| w * x));
            let b = psum(reference.iter().zip(row).map(|(w, x)| w * x));
            a * b
        }));
        if agreement < 0.0 {
            new.iter_mut().for_each(|w| *w = -*w);
        }
        self.weights = Some(new);
    }

    fn compute_weights(&mut self, window: &[Vec<f64>]) -> Option<Vec<f64>> {
        let sel = respirfi_select(window, self.analysis_rate_hz, self.f_low, self.f_high, self.params)?;
        let mut sigmas = BTreeMap::new();
        for &k in sel.signs.keys() {
            let col: Vec<f64> = window.iter().map(|r| r[k]).collect();
            let mean = psum(col.iter().copied()) / col.len() as f64;
            sigmas.insert(k, (psum(col.iter().map(|x| (x - mean) * (x - mean))) / col.len() as f64).sqrt());
        }
        let live: Vec<usize> = sel.signs.keys().copied().filter(|k| sigmas[k] > 0.0).collect();
        if live.is_empty() {
            return None;
        }
        let mean_sigma = psum(live.iter().map(|k| sigmas[k])) / live.len() as f64;
        let mut new = vec![0.0; self.n_subcarriers];
        for &k in &live {
            new[k] = sel.signs[&k] as f64 * mean_sigma / (sigmas[&k] * live.len() as f64);
        }
        self.selection = Some(sel);
        Some(new)
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.acc.iter_mut().for_each(|a| *a = 0.0);
        self.acc_n = 0;
        self.frames = 0;
        self.weights = None;
        self.selection = None;
        self.passes = 0;
        self.failed_passes = 0;
    }
}
