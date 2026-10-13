//! Breathing and heart-rate extraction (port of `wifisense/vitals.py`).

use crate::biquad::BiquadBandpass;
use crate::spectral::{band_peak, sideband_peak};
use crate::types::VitalEstimate;
use std::collections::VecDeque;

/// A periodogram cannot resolve a frequency it has not seen repeat.
pub const MIN_CYCLES: f64 = 3.0;

/// Bandpass + decimate + band scan over a sliding window whose minimum
/// length is derived from the lowest frequency in the band.
#[derive(Debug, Clone)]
pub struct BandVitalExtractor {
    pub f_low: f64,
    pub f_high: f64,
    pub sample_rate_hz: f64,
    pub min_window_s: f64,
    pub window_s: f64,
    pub decimation: usize,
    pub analysis_rate_hz: f64,
    pub resolution_hz: f64,
    filter: BiquadBandpass,
    max_samples: usize,
    min_samples: usize,
    history: VecDeque<f64>,
    phase: usize,
}

impl BandVitalExtractor {
    pub fn new(f_low: f64, f_high: f64, sample_rate_hz: f64, window_s: Option<f64>, resolution_hz: Option<f64>) -> Self {
        let min_window_s = MIN_CYCLES / f_low;
        let window_s = window_s.unwrap_or(min_window_s * 1.5);
        assert!(window_s >= min_window_s, "window_s={window_s:.1}s is too short to resolve {f_low} Hz");
        let decimation = ((sample_rate_hz / (4.0 * f_high)) as usize).max(1);
        let analysis_rate_hz = sample_rate_hz / decimation as f64;
        let resolution_hz = resolution_hz.unwrap_or_else(|| (1.0 / (4.0 * window_s)).max((f_high - f_low) / 400.0));
        Self {
            f_low,
            f_high,
            sample_rate_hz,
            min_window_s,
            window_s,
            decimation,
            analysis_rate_hz,
            resolution_hz,
            filter: BiquadBandpass::new(f_low, f_high, sample_rate_hz),
            max_samples: ((analysis_rate_hz * window_s) as usize).max(1),
            min_samples: ((analysis_rate_hz * min_window_s) as usize).max(1),
            history: VecDeque::new(),
            phase: 0,
        }
    }

    pub fn push(&mut self, sample: f64) {
        let filtered = self.filter.process(sample);
        self.phase += 1;
        if self.phase >= self.decimation {
            self.phase = 0;
            if self.history.len() == self.max_samples {
                self.history.pop_front();
            }
            self.history.push_back(filtered);
        }
    }

    pub fn history(&self) -> Vec<f64> {
        self.history.iter().copied().collect()
    }

    pub fn ready(&self) -> bool {
        self.history.len() >= self.min_samples
    }

    pub fn fill_fraction(&self) -> f64 {
        (self.history.len() as f64 / self.min_samples as f64).min(1.0)
    }

    pub fn estimate(&self) -> VitalEstimate {
        if !self.ready() {
            return VitalEstimate::unavailable();
        }
        match band_peak(&self.history(), self.analysis_rate_hz, self.f_low, self.f_high, self.resolution_hz, true, &[], 0.0) {
            None => VitalEstimate::unavailable(),
            Some(p) => VitalEstimate::from_confidence(p.freq_hz * 60.0, p.confidence),
        }
    }

    pub fn reset(&mut self) {
        self.history.clear();
        self.filter.reset();
        self.phase = 0;
    }
}

/// Respiratory rate, 0.1-0.5 Hz (6-30 BPM).
#[derive(Debug, Clone)]
pub struct BreathingExtractor(pub BandVitalExtractor);

impl BreathingExtractor {
    pub fn new(sample_rate_hz: f64, window_s: Option<f64>) -> Self {
        Self(BandVitalExtractor::new(0.1, 0.5, sample_rate_hz, window_s, None))
    }
}

impl std::ops::Deref for BreathingExtractor {
    type Target = BandVitalExtractor;
    fn deref(&self) -> &BandVitalExtractor {
        &self.0
    }
}
impl std::ops::DerefMut for BreathingExtractor {
    fn deref_mut(&mut self) -> &mut BandVitalExtractor {
        &mut self.0
    }
}

/// Heart rate, 0.8-2.0 Hz. Uses the breathing frequency to reject its
/// harmonics and intermodulation sidebands when one is supplied.
#[derive(Debug, Clone)]
pub struct HeartRateExtractor(pub BandVitalExtractor);

impl HeartRateExtractor {
    pub const DEFAULT_WINDOW_S: f64 = 60.0;

    pub fn new(sample_rate_hz: f64, window_s: Option<f64>) -> Self {
        Self(BandVitalExtractor::new(0.8, 2.0, sample_rate_hz, Some(window_s.unwrap_or(Self::DEFAULT_WINDOW_S)), None))
    }

    pub fn estimate_with(&self, breathing_hz: Option<f64>) -> VitalEstimate {
        let e = &self.0;
        if !e.ready() {
            return VitalEstimate::unavailable();
        }
        match breathing_hz {
            Some(bh) if bh > 0.0 => {
                match sideband_peak(&e.history(), e.analysis_rate_hz, e.f_low, e.f_high, bh, e.resolution_hz, 0.03) {
                    None => VitalEstimate::unavailable(),
                    Some(p) => VitalEstimate::from_confidence(p.freq_hz * 60.0, p.confidence),
                }
            }
            _ => e.estimate(),
        }
    }
}

impl std::ops::Deref for HeartRateExtractor {
    type Target = BandVitalExtractor;
    fn deref(&self) -> &BandVitalExtractor {
        &self.0
    }
}
impl std::ops::DerefMut for HeartRateExtractor {
    fn deref_mut(&mut self) -> &mut BandVitalExtractor {
        &mut self.0
    }
}
