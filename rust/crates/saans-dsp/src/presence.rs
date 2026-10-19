//! Ambient-calibrated presence detection (port of `wifisense/presence.py`).

use crate::biquad::BiquadBandpass;
use crate::psum;
use crate::types::RunningVariance;
use std::collections::VecDeque;

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PresenceState {
    pub occupied: bool,
    pub energy: f64,
    pub threshold: f64,
    pub calibrated: bool,
    pub calibration_progress: f64,
}

#[derive(Debug, Clone, Copy)]
pub struct PresenceConfig {
    pub calibration_s: f64,
    pub sigma_mult: f64,
    pub energy_window_s: f64,
    pub hold_s: f64,
    pub f_low: f64,
    pub f_high: f64,
    /// None = five periods of f_low.
    pub warmup_s: Option<f64>,
}

impl Default for PresenceConfig {
    fn default() -> Self {
        Self {
            calibration_s: 30.0,
            sigma_mult: 4.0,
            energy_window_s: 8.0,
            hold_s: 10.0,
            f_low: 0.1,
            f_high: 2.0,
            warmup_s: None,
        }
    }
}

#[derive(Debug, Clone)]
pub struct PresenceDetector {
    pub sample_rate_hz: f64,
    pub sigma_mult: f64,
    pub calibration_samples: u64,
    pub hold_samples: u64,
    pub warmup_samples: u64,
    /// Derived floor: threshold >= calibrated mean * this (see the Python
    /// reference for the derivation and the bug it fixes).
    pub min_threshold_ratio: f64,
    warmed: u64,
    bandpass: BiquadBandpass,
    window: VecDeque<f64>,
    window_cap: usize,
    calib: RunningVariance,
    threshold: f64,
    calibrated: bool,
    since_trigger: u64,
    last_energy: f64,
}

impl PresenceDetector {
    pub fn new(sample_rate_hz: f64, cfg: PresenceConfig) -> Self {
        let hold_samples = ((sample_rate_hz * cfg.hold_s) as u64).max(1);
        let warmup_samples = match cfg.warmup_s {
            Some(w) => (sample_rate_hz * w).max(0.0) as u64,
            None => (sample_rate_hz * 5.0 / cfg.f_low) as u64,
        };
        let enbw = (std::f64::consts::PI / 2.0) * (cfg.f_high - cfg.f_low);
        let min_threshold_ratio =
            1.0 + cfg.sigma_mult / (2.0 * (enbw * cfg.energy_window_s).sqrt());
        Self {
            sample_rate_hz,
            sigma_mult: cfg.sigma_mult,
            min_threshold_ratio,
            calibration_samples: ((sample_rate_hz * cfg.calibration_s) as u64).max(1),
            hold_samples,
            warmup_samples,
            warmed: 0,
            bandpass: BiquadBandpass::new(cfg.f_low, cfg.f_high, sample_rate_hz),
            window: VecDeque::new(),
            window_cap: ((sample_rate_hz * cfg.energy_window_s) as usize).max(2),
            calib: RunningVariance::default(),
            threshold: f64::INFINITY,
            calibrated: false,
            since_trigger: hold_samples,
            last_energy: 0.0,
        }
    }

    pub fn process(&mut self, residual: f64) -> PresenceState {
        if self.window.len() == self.window_cap {
            self.window.pop_front();
        }
        self.window.push_back(self.bandpass.process(residual));
        let n = self.window.len();
        let energy = if n > 0 {
            (psum(self.window.iter().map(|v| v * v)) / n as f64).sqrt()
        } else {
            0.0
        };
        self.last_energy = energy;

        if self.warmed < self.warmup_samples {
            self.warmed += 1;
            return PresenceState {
                occupied: false,
                energy,
                threshold: self.threshold,
                calibrated: false,
                calibration_progress: 0.0,
            };
        }
        if !self.calibrated {
            self.calib.update(energy);
            if self.calib.count >= self.calibration_samples {
                self.threshold = (self.calib.mean + self.sigma_mult * self.calib.std())
                    .max(self.calib.mean * self.min_threshold_ratio)
                    .max(1e-6);
                self.calibrated = true;
            }
            return PresenceState {
                occupied: false,
                energy,
                threshold: self.threshold,
                calibrated: false,
                calibration_progress: self.calib.count as f64 / self.calibration_samples as f64,
            };
        }
        if energy > self.threshold {
            self.since_trigger = 0;
        } else {
            self.since_trigger += 1;
        }
        PresenceState {
            occupied: self.since_trigger < self.hold_samples,
            energy,
            threshold: self.threshold,
            calibrated: true,
            calibration_progress: 1.0,
        }
    }

    pub fn calibrated(&self) -> bool {
        self.calibrated
    }
    pub fn threshold(&self) -> f64 {
        self.threshold
    }
    pub fn warmup_s(&self) -> f64 {
        self.warmup_samples as f64 / self.sample_rate_hz
    }

    pub fn recalibrate(&mut self) {
        self.calib.reset();
        self.threshold = f64::INFINITY;
        self.calibrated = false;
        self.window.clear();
        self.bandpass.reset();
        self.warmed = 0;
        self.since_trigger = self.hold_samples;
    }
}
