//! FeatureFrame v1: what a Saans node sends instead of raw CSI
//! (port of `wifisense/features.py`; byte layout in `spec/wire.md` §2).

use crate::biquad::BiquadBandpass;
use crate::fft::subcarrier_bnr;
use crate::gapfill::SequenceGapFiller;
use crate::pipeline::{PipelineConfig, SensingPipeline, SubcarrierSelection};
use crate::types::CsiFrame;
use crate::{psum, py_round};
use std::collections::VecDeque;

pub const FEATURE_MAGIC: u32 = 0x534E_4153; // b"SANS" on the wire
pub const FEATURE_VERSION: u8 = 1;
pub const FEATURE_FRAME_SIZE: usize = 68;

pub const FLAG_CALIBRATED: u8 = 1 << 0;
pub const FLAG_OCCUPIED: u8 = 1 << 1;
pub const FLAG_VITALS_ENABLED: u8 = 1 << 2;
pub const FLAG_RAW_MODE: u8 = 1 << 3;
pub const FLAG_BREATHING_VALID: u8 = 1 << 4;
pub const FLAG_HEART_VALID: u8 = 1 << 5;
pub const FLAG_DISCONTINUITY: u8 = 1 << 6;

pub const NOT_CALIBRATED: u32 = 0xFFFF_FFFF;
pub const NOISE_UNKNOWN: i8 = -128;

const BREATH_RMS_WINDOW_S: f64 = 2.0;
const GROSS_MOTION_WINDOW_S: f64 = 1.0;
const LINK_STATS_WINDOW_S: f64 = 10.0;
const GROSS_MOTION_BAND_HZ: (f64, f64) = (1.0, 4.0);
/// Two cascaded sections: one biquad leaks ~19% of the breathing
/// component into the gross-motion band (see the Python reference).
const GROSS_MOTION_SECTIONS: usize = 2;
const BREATH_BAND_HZ: (f64, f64) = (0.1, 0.5);

/// IEEE 802.3 CRC-32, as zlib.crc32.
pub fn crc32(data: &[u8]) -> u32 {
    static TABLE: std::sync::OnceLock<[u32; 256]> = std::sync::OnceLock::new();
    let table = TABLE.get_or_init(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { 0xEDB8_8320 ^ (c >> 1) } else { c >> 1 };
            }
            *slot = c;
        }
        t
    });
    let mut crc = 0xFFFF_FFFFu32;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FeatureFrame {
    pub version: u8,
    pub node_id: u8,
    pub tx_id: u8,
    pub flags: u8,
    pub seq: u32,
    pub uptime_ms: u32,
    pub csi_rate_chz: u16,
    pub loss_permille: u16,
    pub motion_energy: f32,
    pub presence_threshold: f32,
    pub gross_motion: f32,
    pub breath_rms: f32,
    pub breathing_bpm: f32,
    pub breathing_conf: f32,
    pub heart_bpm: f32,
    pub heart_conf: f32,
    pub best_bnr: f32,
    pub calib_age_s: u32,
    pub rssi_dbm: i8,
    pub noise_floor_dbm: i8,
    pub n_subcarriers: u8,
}

impl Default for FeatureFrame {
    fn default() -> Self {
        Self {
            version: FEATURE_VERSION,
            node_id: 0,
            tx_id: 0xFF,
            flags: 0,
            seq: 0,
            uptime_ms: 0,
            csi_rate_chz: 0,
            loss_permille: 0,
            motion_energy: 0.0,
            presence_threshold: 0.0,
            gross_motion: 0.0,
            breath_rms: 0.0,
            breathing_bpm: 0.0,
            breathing_conf: 0.0,
            heart_bpm: 0.0,
            heart_conf: 0.0,
            best_bnr: 0.0,
            calib_age_s: NOT_CALIBRATED,
            rssi_dbm: 0,
            noise_floor_dbm: NOISE_UNKNOWN,
            n_subcarriers: 0,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DecodeError {
    Size(usize),
    Crc,
    Magic(u32),
    Version(u8),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DecodeError::Size(n) => write!(f, "FeatureFrame must be {FEATURE_FRAME_SIZE} bytes, got {n}"),
            DecodeError::Crc => write!(f, "FeatureFrame CRC mismatch"),
            DecodeError::Magic(m) => write!(f, "bad magic 0x{m:08X}"),
            DecodeError::Version(v) => write!(f, "unsupported FeatureFrame version {v}"),
        }
    }
}

impl std::error::Error for DecodeError {}

impl FeatureFrame {
    pub fn flag(&self, bit: u8) -> bool {
        self.flags & bit != 0
    }

    pub fn encode(&self) -> [u8; FEATURE_FRAME_SIZE] {
        let mut b = [0u8; FEATURE_FRAME_SIZE];
        b[0..4].copy_from_slice(&FEATURE_MAGIC.to_le_bytes());
        b[4] = self.version;
        b[5] = self.node_id;
        b[6] = self.tx_id;
        b[7] = self.flags;
        b[8..12].copy_from_slice(&self.seq.to_le_bytes());
        b[12..16].copy_from_slice(&self.uptime_ms.to_le_bytes());
        b[16..18].copy_from_slice(&self.csi_rate_chz.to_le_bytes());
        b[18..20].copy_from_slice(&self.loss_permille.to_le_bytes());
        let floats = [
            self.motion_energy,
            self.presence_threshold,
            self.gross_motion,
            self.breath_rms,
            self.breathing_bpm,
            self.breathing_conf,
            self.heart_bpm,
            self.heart_conf,
            self.best_bnr,
        ];
        for (i, f) in floats.iter().enumerate() {
            b[20 + 4 * i..24 + 4 * i].copy_from_slice(&f.to_le_bytes());
        }
        b[56..60].copy_from_slice(&self.calib_age_s.to_le_bytes());
        b[60] = self.rssi_dbm as u8;
        b[61] = self.noise_floor_dbm as u8;
        b[62] = self.n_subcarriers;
        b[63] = 0;
        let crc = crc32(&b[..64]);
        b[64..68].copy_from_slice(&crc.to_le_bytes());
        b
    }

    pub fn decode(data: &[u8]) -> Result<Self, DecodeError> {
        if data.len() != FEATURE_FRAME_SIZE {
            return Err(DecodeError::Size(data.len()));
        }
        let u32_at = |o: usize| u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
        let u16_at = |o: usize| u16::from_le_bytes(data[o..o + 2].try_into().unwrap());
        let f32_at = |o: usize| f32::from_le_bytes(data[o..o + 4].try_into().unwrap());
        if crc32(&data[..64]) != u32_at(64) {
            return Err(DecodeError::Crc);
        }
        if u32_at(0) != FEATURE_MAGIC {
            return Err(DecodeError::Magic(u32_at(0)));
        }
        if data[4] != FEATURE_VERSION {
            return Err(DecodeError::Version(data[4]));
        }
        Ok(Self {
            version: data[4],
            node_id: data[5],
            tx_id: data[6],
            flags: data[7],
            seq: u32_at(8),
            uptime_ms: u32_at(12),
            csi_rate_chz: u16_at(16),
            loss_permille: u16_at(18),
            motion_energy: f32_at(20),
            presence_threshold: f32_at(24),
            gross_motion: f32_at(28),
            breath_rms: f32_at(32),
            breathing_bpm: f32_at(36),
            breathing_conf: f32_at(40),
            heart_bpm: f32_at(44),
            heart_conf: f32_at(48),
            best_bnr: f32_at(52),
            calib_age_s: u32_at(56),
            rssi_dbm: data[60] as i8,
            noise_floor_dbm: data[61] as i8,
            n_subcarriers: data[62],
        })
    }

    /// True if `data` starts with the FeatureFrame magic (cheap dispatch
    /// between FeatureFrame and raw CSI datagrams on one port).
    pub fn sniff(data: &[u8]) -> bool {
        data.len() >= 4 && data[..4] == FEATURE_MAGIC.to_le_bytes()
    }
}

/// RMS of the last `n` samples, summed afresh on each read: a running
/// total drifts over a night and would make the three implementations
/// disagree for no physical reason.
#[derive(Debug, Clone)]
struct RollingRms {
    sq: VecDeque<f64>,
    cap: usize,
}

impl RollingRms {
    fn new(n: usize) -> Self {
        Self { sq: VecDeque::with_capacity(n.max(1)), cap: n.max(1) }
    }
    fn push(&mut self, x: f64) {
        if self.sq.len() == self.cap {
            self.sq.pop_front();
        }
        self.sq.push_back(x * x);
    }
    fn value(&self) -> f64 {
        if self.sq.is_empty() {
            return 0.0;
        }
        (psum(self.sq.iter().copied()) / self.sq.len() as f64).sqrt()
    }
}

#[derive(Debug, Clone, Copy)]
pub struct FeatureExtractorConfig {
    pub n_subcarriers: usize,
    pub sample_rate_hz: f64,
    pub node_id: u8,
    pub tx_id: u8,
    pub frame_rate_hz: f64,
    pub vitals_enabled: bool,
    pub heart_enabled: bool,
    pub calibration_s: f64,
    pub max_gap_s: f64,
    pub selection: SubcarrierSelection,
    pub raw_mode: bool,
    pub bnr_window_s: f64,
    pub bnr_update_s: f64,
    pub bnr_rate_hz: f64,
    /// Optional link-quality scoring (spec/wire.md: 0 = not computed).
    pub compute_bnr: bool,
}

impl Default for FeatureExtractorConfig {
    fn default() -> Self {
        Self {
            n_subcarriers: 56,
            sample_rate_hz: 20.0,
            node_id: 0,
            tx_id: 0xFF,
            frame_rate_hz: 2.0,
            vitals_enabled: false,
            heart_enabled: false,
            calibration_s: 30.0,
            max_gap_s: 1.0,
            selection: SubcarrierSelection::Variance,
            raw_mode: false,
            bnr_window_s: 30.0,
            bnr_update_s: 5.0,
            bnr_rate_hz: 4.0,
            compute_bnr: true,
        }
    }
}

/// One link's CSI frames in, FeatureFrames out. Deterministic: frames are
/// clocked by processed CSI frames, not wall time.
#[derive(Debug, Clone)]
pub struct FeatureExtractor {
    pub config: FeatureExtractorConfig,
    pub pipeline: SensingPipeline,
    pub gaps: SequenceGapFiller,
    breath_filter: BiquadBandpass,
    motion_filters: Vec<BiquadBandpass>,
    breath_rms: RollingRms,
    motion_rms: RollingRms,
    every: u64,
    bnr_dec: usize,
    bnr_rate: f64,
    bnr_rows: VecDeque<Vec<f64>>,
    bnr_cap: usize,
    bnr_acc: Vec<f64>,
    bnr_n: usize,
    bnr_every: u64,
    best_bnr: f64,
    processed: u64,
    seq: u32,
    t0: Option<f64>,
    calibrated_at: Option<u64>,
    arrivals: VecDeque<(f64, u64)>,
    rssi_sum: i64,
    rssi_n: i64,
    noise_sum: i64,
    noise_n: i64,
    discontinuities_seen: u64,
}

/// The ADR-001/005 consent gate, in one place so it can be audited:
/// breathing- and heart-band fields leave the node only if allowed.
fn gate(value: f64, allowed: bool) -> f32 {
    if allowed {
        value as f32
    } else {
        0.0
    }
}

impl FeatureExtractor {
    pub fn new(config: FeatureExtractorConfig) -> Self {
        let fs = config.sample_rate_hz;
        assert!(fs >= 10.0, "FeatureFrame needs >= 10 Hz CSI (got {fs}): the gross-motion band reaches 4 Hz");
        assert!(config.frame_rate_hz > 0.0 && config.frame_rate_hz <= fs, "frame_rate_hz must be in (0, fs]");
        let pipeline = SensingPipeline::new(PipelineConfig {
            n_subcarriers: config.n_subcarriers,
            sample_rate_hz: fs,
            calibration_s: config.calibration_s,
            selection: config.selection,
            ..Default::default()
        });
        let bnr_dec = (py_round(fs / config.bnr_rate_hz) as usize).max(1);
        let bnr_rate = fs / bnr_dec as f64;
        Self {
            gaps: SequenceGapFiller::new((config.max_gap_s * fs) as u64),
            breath_filter: BiquadBandpass::new(BREATH_BAND_HZ.0, BREATH_BAND_HZ.1, fs),
            motion_filters: (0..GROSS_MOTION_SECTIONS).map(|_| BiquadBandpass::new(GROSS_MOTION_BAND_HZ.0, GROSS_MOTION_BAND_HZ.1, fs)).collect(),
            breath_rms: RollingRms::new(py_round(fs * BREATH_RMS_WINDOW_S) as usize),
            motion_rms: RollingRms::new(py_round(fs * GROSS_MOTION_WINDOW_S) as usize),
            every: (py_round(fs / config.frame_rate_hz) as u64).max(1),
            bnr_dec,
            bnr_rate,
            bnr_rows: VecDeque::new(),
            bnr_cap: ((config.bnr_window_s * bnr_rate) as usize).max(8),
            bnr_acc: vec![0.0; config.n_subcarriers],
            bnr_n: 0,
            bnr_every: (py_round(config.bnr_update_s * fs) as u64).max(1),
            best_bnr: 0.0,
            processed: 0,
            seq: 0,
            t0: None,
            calibrated_at: None,
            arrivals: VecDeque::new(),
            rssi_sum: 0,
            rssi_n: 0,
            noise_sum: 0,
            noise_n: 0,
            discontinuities_seen: 0,
            pipeline,
            config,
        }
    }

    /// Feed one *received* CSI frame. Returns the FeatureFrames due.
    pub fn push(&mut self, frame: CsiFrame, sequence: Option<u64>, t_s: Option<f64>, noise_floor_dbm: Option<i32>) -> Vec<FeatureFrame> {
        let seq = sequence.unwrap_or(frame.sample_index);
        let t = t_s.unwrap_or(self.processed as f64 / self.config.sample_rate_hz);
        if self.t0.is_none() {
            self.t0 = Some(t);
        }
        self.arrivals.push_back((t, seq));
        while let Some(&(t_first, _)) = self.arrivals.front() {
            if t_first < t - LINK_STATS_WINDOW_S {
                self.arrivals.pop_front();
            } else {
                break;
            }
        }
        self.rssi_sum += frame.rssi_dbm as i64;
        self.rssi_n += 1;
        if let Some(nf) = noise_floor_dbm {
            self.noise_sum += nf as i64;
            self.noise_n += 1;
        }
        let mut out = Vec::new();
        for f in self.gaps.push(seq, frame) {
            if let Some(ff) = self.process(&f, t) {
                out.push(ff);
            }
        }
        out
    }

    fn process(&mut self, frame: &CsiFrame, t: f64) -> Option<FeatureFrame> {
        self.pipeline.push(frame);
        self.processed += 1;
        if self.pipeline.calibrated() && self.calibrated_at.is_none() {
            self.calibrated_at = Some(self.processed);
        }
        let b = self.breath_filter.process(self.pipeline.last_vitals_input);
        self.breath_rms.push(b);
        let mut m = self.pipeline.last_fused;
        for section in self.motion_filters.iter_mut() {
            m = section.process(m);
        }
        self.motion_rms.push(m);

        if self.config.compute_bnr {
            self.bnr_step();
        }
        if !self.processed.is_multiple_of(self.every) {
            return None;
        }
        Some(self.emit(t))
    }

    fn bnr_step(&mut self) {
        let n = self.pipeline.last_residuals.len().min(self.config.n_subcarriers);
        for k in 0..n {
            let r = self.pipeline.last_residuals[k];
            self.bnr_acc[k] += if r.is_finite() { r } else { 0.0 };
        }
        self.bnr_n += 1;
        if self.bnr_n >= self.bnr_dec {
            let row: Vec<f64> = self.bnr_acc.iter().map(|a| a / self.bnr_n as f64).collect();
            if self.bnr_rows.len() == self.bnr_cap {
                self.bnr_rows.pop_front();
            }
            self.bnr_rows.push_back(row);
            self.bnr_acc.iter_mut().for_each(|a| *a = 0.0);
            self.bnr_n = 0;
        }
        if self.processed.is_multiple_of(self.bnr_every) && self.bnr_rows.len() == self.bnr_cap {
            let rows: Vec<Vec<f64>> = self.bnr_rows.iter().cloned().collect();
            let (bnr, _) = subcarrier_bnr(&rows, self.bnr_rate, BREATH_BAND_HZ.0, BREATH_BAND_HZ.1);
            self.best_bnr = bnr.iter().copied().filter(|b| b.is_finite()).fold(None, |a: Option<f64>, b| Some(a.map_or(b, |a| a.max(b)))).unwrap_or(0.0);
        }
    }

    fn link_stats(&self) -> (u16, u16) {
        if self.arrivals.len() < 2 {
            return (0, 0);
        }
        let (t_first, s_first) = *self.arrivals.front().unwrap();
        let (t_last, s_last) = *self.arrivals.back().unwrap();
        let span = t_last - t_first;
        let rate = if span > 0.0 { (self.arrivals.len() - 1) as f64 / span } else { 0.0 };
        let expected = ((s_last + (1u64 << 32) - s_first) % (1u64 << 32)) + 1;
        let received = self.arrivals.len() as u64;
        let loss = if expected >= received { (1.0 - received as f64 / expected as f64).max(0.0) } else { 0.0 };
        let clamp = |x: f64| x.clamp(0.0, 65535.0) as u16;
        (clamp(py_round(rate * 100.0)), clamp(py_round(loss * 1000.0)))
    }

    fn emit(&mut self, t: f64) -> FeatureFrame {
        let reading = self.pipeline.reading();
        let calibrated = self.pipeline.calibrated();
        let c = self.config;
        let mut flags = 0u8;
        if calibrated {
            flags |= FLAG_CALIBRATED;
        }
        if reading.occupied {
            flags |= FLAG_OCCUPIED;
        }
        if c.vitals_enabled {
            flags |= FLAG_VITALS_ENABLED;
        }
        if c.raw_mode {
            flags |= FLAG_RAW_MODE;
        }
        if self.gaps.discontinuities > self.discontinuities_seen {
            flags |= FLAG_DISCONTINUITY;
            self.discontinuities_seen = self.gaps.discontinuities;
        }
        let breathing_ok = c.vitals_enabled && reading.occupied && reading.respiratory_rate.status.usable();
        let heart_ok = breathing_ok && c.heart_enabled && reading.heart_rate.status.usable();
        if breathing_ok {
            flags |= FLAG_BREATHING_VALID;
        }
        if heart_ok {
            flags |= FLAG_HEART_VALID;
        }
        let (rate_chz, loss_pm) = self.link_stats();
        let vit = c.vitals_enabled;
        let heart = vit && c.heart_enabled;
        let frame = FeatureFrame {
            version: FEATURE_VERSION,
            node_id: c.node_id,
            tx_id: c.tx_id,
            flags,
            seq: self.seq,
            uptime_ms: py_round((t - self.t0.unwrap_or(0.0)) * 1000.0) as u32,
            csi_rate_chz: rate_chz,
            loss_permille: loss_pm,
            motion_energy: reading.motion_energy as f32,
            presence_threshold: if calibrated { self.pipeline.presence.threshold() as f32 } else { 0.0 },
            gross_motion: self.motion_rms.value() as f32,
            breath_rms: gate(self.breath_rms.value(), vit),
            breathing_bpm: gate(reading.respiratory_rate.value_bpm, vit),
            breathing_conf: gate(reading.respiratory_rate.confidence, vit),
            heart_bpm: gate(reading.heart_rate.value_bpm, heart),
            heart_conf: gate(reading.heart_rate.confidence, heart),
            best_bnr: self.best_bnr as f32,
            calib_age_s: match self.calibrated_at {
                Some(at) => ((self.processed - at) as f64 / c.sample_rate_hz) as u32,
                None => NOT_CALIBRATED,
            },
            rssi_dbm: if self.rssi_n > 0 { py_round(self.rssi_sum as f64 / self.rssi_n as f64).clamp(-128.0, 127.0) as i8 } else { 0 },
            noise_floor_dbm: if self.noise_n > 0 {
                py_round(self.noise_sum as f64 / self.noise_n as f64).clamp(-128.0, 127.0) as i8
            } else {
                NOISE_UNKNOWN
            },
            n_subcarriers: c.n_subcarriers.min(255) as u8,
        };
        self.seq = self.seq.wrapping_add(1);
        self.rssi_sum = 0;
        self.rssi_n = 0;
        self.noise_sum = 0;
        self.noise_n = 0;
        frame
    }

    pub fn processed_frames(&self) -> u64 {
        self.processed
    }
}
