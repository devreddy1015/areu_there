//! The stages wired together (port of `wifisense/pipeline.py`).
//!
//! `process` matches the reference call for call. `push` + `reading`
//! split it: the stateful updates happen every frame, and the spectral
//! estimates (the expensive part, and pure functions of state) only when
//! a caller asks. The result is identical; the FeatureExtractor uses the
//! split form so it pays for estimates twice a second instead of twenty.

use crate::preprocess::{fuse, StaticSuppressor};
use crate::presence::{PresenceConfig, PresenceDetector, PresenceState};
use crate::respirfi::{RespirFiFusion, SelectParams};
use crate::types::{CsiFrame, VitalEstimate, VitalReading};
use crate::vitals::{BreathingExtractor, HeartRateExtractor};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SubcarrierSelection {
    Variance,
    RespirFi,
}

#[derive(Debug, Clone, Copy)]
pub struct PipelineConfig {
    pub n_subcarriers: usize,
    pub sample_rate_hz: f64,
    pub calibration_s: f64,
    pub breathing_window_s: Option<f64>,
    pub heart_window_s: Option<f64>,
    pub suppress_vitals_when_empty: bool,
    pub selection: SubcarrierSelection,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            n_subcarriers: 56,
            sample_rate_hz: 20.0,
            calibration_s: 30.0,
            breathing_window_s: None,
            heart_window_s: None,
            suppress_vitals_when_empty: true,
            selection: SubcarrierSelection::Variance,
        }
    }
}

#[derive(Debug, Clone)]
pub struct SensingPipeline {
    pub config: PipelineConfig,
    pub suppressor: StaticSuppressor,
    pub presence: PresenceDetector,
    pub breathing: BreathingExtractor,
    pub heart: HeartRateExtractor,
    pub selector: Option<RespirFiFusion>,
    pub last_residuals: Vec<f64>,
    pub last_fused: f64,
    pub last_vitals_input: f64,
    pub last_presence: Option<PresenceState>,
    samples: u64,
}

impl SensingPipeline {
    pub fn new(config: PipelineConfig) -> Self {
        let fs = config.sample_rate_hz;
        Self {
            suppressor: StaticSuppressor::new(config.n_subcarriers, fs, 0.03),
            presence: PresenceDetector::new(
                fs,
                PresenceConfig {
                    calibration_s: config.calibration_s,
                    ..Default::default()
                },
            ),
            breathing: BreathingExtractor::new(fs, config.breathing_window_s),
            heart: HeartRateExtractor::new(fs, config.heart_window_s),
            selector: match config.selection {
                SubcarrierSelection::RespirFi => Some(RespirFiFusion::new(
                    config.n_subcarriers,
                    fs,
                    SelectParams::default(),
                )),
                SubcarrierSelection::Variance => None,
            },
            config,
            last_residuals: Vec::new(),
            last_fused: 0.0,
            last_vitals_input: 0.0,
            last_presence: None,
            samples: 0,
        }
    }

    /// Every stateful update for one frame; no spectral estimation.
    pub fn push(&mut self, frame: &CsiFrame) -> PresenceState {
        let (residuals, weights) = self.suppressor.process(frame);
        let fused = fuse(&residuals, &weights);
        let state = self.presence.process(fused);
        let vitals_input = match self.selector.as_mut() {
            Some(sel) => sel.process(&residuals, &weights),
            None => fused,
        };
        self.breathing.push(vitals_input);
        self.heart.push(vitals_input);
        self.last_residuals = residuals;
        self.last_fused = fused;
        self.last_vitals_input = vitals_input;
        self.last_presence = Some(state);
        self.samples += 1;
        state
    }

    /// The reading the reference would have returned for the last frame.
    pub fn reading(&self) -> VitalReading {
        let state = self.last_presence.unwrap_or(PresenceState {
            occupied: false,
            energy: 0.0,
            threshold: f64::INFINITY,
            calibrated: false,
            calibration_progress: 0.0,
        });
        let hide = self.config.suppress_vitals_when_empty && !state.occupied;
        let (rr, hr) = if hide {
            (VitalEstimate::unavailable(), VitalEstimate::unavailable())
        } else {
            let rr = self.breathing.estimate();
            let bh = if rr.status.usable() {
                Some(rr.value_bpm / 60.0)
            } else {
                None
            };
            (rr, self.heart.estimate_with(bh))
        };
        VitalReading {
            respiratory_rate: rr,
            heart_rate: hr,
            subcarrier_count: self.last_residuals.len(),
            signal_quality: self.breathing.fill_fraction().min(1.0),
            timestamp_s: self.samples as f64 / self.config.sample_rate_hz,
            occupied: state.occupied,
            motion_energy: state.energy,
        }
    }

    pub fn process(&mut self, frame: &CsiFrame) -> VitalReading {
        self.push(frame);
        self.reading()
    }

    pub fn calibrated(&self) -> bool {
        self.presence.calibrated()
    }

    pub fn samples(&self) -> u64 {
        self.samples
    }
}
