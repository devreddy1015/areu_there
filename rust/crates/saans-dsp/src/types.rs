//! Core domain types (port of `wifisense/types.py`).

/// How much to trust a vital-sign estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VitalStatus {
    Valid,
    Degraded,
    Unreliable,
    Unavailable,
}

impl VitalStatus {
    /// Golden-vector status code: 0 unavailable, 1 unreliable, 2 degraded, 3 valid.
    pub fn code(self) -> u8 {
        match self {
            VitalStatus::Unavailable => 0,
            VitalStatus::Unreliable => 1,
            VitalStatus::Degraded => 2,
            VitalStatus::Valid => 3,
        }
    }

    /// VALID or DEGRADED: good enough to report and to use as a reference.
    pub fn usable(self) -> bool {
        matches!(self, VitalStatus::Valid | VitalStatus::Degraded)
    }
}

/// A single vital-sign estimate in beats/breaths per minute.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct VitalEstimate {
    pub value_bpm: f64,
    pub confidence: f64,
    pub status: VitalStatus,
}

impl VitalEstimate {
    pub fn unavailable() -> Self {
        Self { value_bpm: 0.0, confidence: 0.0, status: VitalStatus::Unavailable }
    }

    /// Grade by spectral confidence: >= 0.70 valid, >= 0.40 degraded,
    /// otherwise reported but labelled unreliable.
    pub fn from_confidence(bpm: f64, confidence: f64) -> Self {
        let status = if confidence >= 0.70 {
            VitalStatus::Valid
        } else if confidence >= 0.40 {
            VitalStatus::Degraded
        } else {
            VitalStatus::Unreliable
        };
        Self { value_bpm: bpm, confidence, status }
    }
}

/// One CSI snapshot: per-subcarrier amplitude and (wrapped) phase.
#[derive(Debug, Clone, PartialEq)]
pub struct CsiFrame {
    pub amplitudes: Vec<f64>,
    pub phases: Vec<f64>,
    pub sample_index: u64,
    pub sample_rate_hz: f64,
    pub rssi_dbm: i32,
    pub node_id: u8,
}

impl CsiFrame {
    pub fn new(amplitudes: Vec<f64>, sample_index: u64, sample_rate_hz: f64) -> Self {
        let n = amplitudes.len();
        Self { amplitudes, phases: vec![0.0; n], sample_index, sample_rate_hz, rssi_dbm: 0, node_id: 0 }
    }

    pub fn n_subcarriers(&self) -> usize {
        self.amplitudes.len()
    }

    /// Build from ESP-IDF (imaginary, real) int pairs. Imaginary first:
    /// getting this backwards leaves amplitude right and mirrors phase.
    pub fn from_iq(iq: &[(i8, i8)], sample_index: u64, sample_rate_hz: f64, rssi_dbm: i32, node_id: u8) -> Self {
        let amplitudes = iq.iter().map(|&(im, re)| (im as f64).hypot(re as f64)).collect();
        let phases = iq.iter().map(|&(im, re)| (im as f64).atan2(re as f64)).collect();
        Self { amplitudes, phases, sample_index, sample_rate_hz, rssi_dbm, node_id }
    }
}

/// Combined output of one pipeline step.
#[derive(Debug, Clone, PartialEq)]
pub struct VitalReading {
    pub respiratory_rate: VitalEstimate,
    pub heart_rate: VitalEstimate,
    pub subcarrier_count: usize,
    pub signal_quality: f64,
    pub timestamp_s: f64,
    pub occupied: bool,
    pub motion_energy: f64,
}

/// Welford's online variance. Sample variance (n - 1), as the reference.
#[derive(Debug, Clone, Copy, Default)]
pub struct RunningVariance {
    pub count: u64,
    pub mean: f64,
    pub m2: f64,
}

impl RunningVariance {
    pub fn update(&mut self, x: f64) {
        self.count += 1;
        let delta = x - self.mean;
        self.mean += delta / self.count as f64;
        self.m2 += delta * (x - self.mean);
    }

    pub fn variance(&self) -> f64 {
        if self.count > 1 {
            self.m2 / (self.count - 1) as f64
        } else {
            0.0
        }
    }

    pub fn std(&self) -> f64 {
        self.variance().sqrt()
    }

    pub fn reset(&mut self) {
        *self = Self::default();
    }
}
