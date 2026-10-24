//! The simulated phone (spec/locate.md §6, §7) and Locate's configuration
//! (§9).
//!
//! ADR-008 decision 16: **the phone is simulated only**. Every function
//! here that starts a phone run refuses any label but `"SIM"`. The phone
//! ranges to every node by Wi-Fi RTT (IEEE 802.11mc FTM) and reads every
//! node's RSSI at epochs (§6); a Gauss-Newton fix per method and epoch
//! (§7.1, §7.2) feeds one constant-velocity Kalman filter per method
//! (§7.3), whose estimate is published with a radius (§7.4, §8).
//!
//! Ranging a cooperative device is not the multilateration of people that
//! CAMPUS_SENSING_PROMPT.md §3 bans: people are located only by the links
//! they disturb (§4-§5).
//!
//! The radio model is parametric from geometry: the inputs are plain data
//! (nodes, walls, boxes, person cylinders, phone truth), so this module
//! depends on no scene or Room Sim crate. No `Person.id` or `Phone.id` is
//! ever an input here, so none can reach a record, log or error.

// NaN fails closed: every deciding comparison is written `!(a <= b)` /
// `!(a > b)` on purpose, exactly as the spec writes `not (...)`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]
// Index loops mirror the spec's subscripts.
#![allow(clippy::needless_range_loop)]

use std::borrow::Cow;

use saans_sim::rng::Rng;

use crate::num::{
    chi2_2_99, clamp, det2, identity, inv2, len2, len3, ln10, mat_add, mat_mul, mat_scale, mat_sub,
    mat_vec, pow10, pymax, pymin, round_half_even, sym, transpose, wavelength, Mat2, Mat4,
    Neumaier, FOUR_PI, PI,
};
use crate::radius::{publish, Outcome, Reason};

/// RTT of node `n` draws from `Rng::stream(phone_seed, 300000 + n)` (§6.3).
pub const RTT_STREAM_BASE: u64 = 300_000;
/// RSSI of node `n` draws from `Rng::stream(phone_seed, 310000 + n)` (§6.3).
pub const RSSI_STREAM_BASE: u64 = 310_000;

/// Gauss-Newton iteration limit (§7.1).
pub const GN_MAX_ITERATIONS: usize = 200;
/// Step-halving limit per iteration (§7.1).
pub const GN_MAX_HALVINGS: usize = 30;
/// Stop rule: `a * sqrt(sx*sx + sy*sy) < 1e-4` (§7.1).
pub const GN_STOP_M: f64 = 1e-4;
/// The collinear rule's relative threshold (§7): `not (det > 1e-9 *
/// ((n00 + n11)*(n00 + n11)))` on the anchors' linearised normal matrix.
pub const ANCHOR_DET_REL: f64 = 1e-9;

/// The link signals §9's `link_signal` may name.
pub const LINK_SIGNALS: [&str; 3] = ["rssi_dbm", "total_power_db", "static_suppressed_amplitude"];

// ===================================================================
// Errors
// ===================================================================

/// An error of the new Locate pieces. Messages name the parameter or field
/// and never echo a label, a `Person.id` or a `Phone.id`.
#[derive(Debug, Clone, PartialEq)]
pub enum Error {
    /// A §9 parameter failed validation; `param` names it.
    Config { param: &'static str, reason: String },
    /// A label other than `"SIM"` was given to a phone function
    /// (ADR-008 decision 16). The given label is not repeated.
    Label,
    /// Invalid input data; `field` names it.
    Input { field: &'static str, reason: String },
}

impl std::fmt::Display for Error {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Error::Config { param, reason } => write!(f, "{param}: {reason}"),
            Error::Label => f.write_str(
                "label: the phone is simulated only (ADR-008 decision 16); the only label is \"SIM\"",
            ),
            Error::Input { field, reason } => write!(f, "{field}: {reason}"),
        }
    }
}

impl std::error::Error for Error {}

fn cfg_err(param: &'static str, reason: &str) -> Error {
    Error::Config {
        param,
        reason: reason.to_string(),
    }
}

fn input_err(field: &'static str, reason: impl Into<String>) -> Error {
    Error::Input {
        field,
        reason: reason.into(),
    }
}

// ===================================================================
// Configuration (§9)
// ===================================================================

/// Every §9 parameter, with its default. Sources (§9): SIM unless noted;
/// `rtt_sigma_m` SIM->REAL; `rssi_breakpoint_m`, `rssi_far_exponent` REAL
/// (TGn models B/C); the three `*_radius_scale` uncalibrated (§8).
#[derive(Debug, Clone, PartialEq)]
pub struct Config {
    // --- RTI (the Python defaults; the ported modules check them) ---
    /// Ellipse excess path, m (0.3).
    pub lambda_m: f64,
    /// Pixel pitch, m (0.5).
    pub pitch_m: f64,
    /// Regularisation (1.0).
    pub alpha: f64,
    /// Prior sigma (1.0).
    pub sigma: f64,
    /// Prior correlation distance, m (0.5).
    pub delta_c: f64,
    /// Tracker significance threshold (5.0).
    pub z_threshold: f64,
    // --- Measurement and pipeline ---
    /// Variance window, s (1.0). `> 0`. (The measurement's
    /// `sample_rate_hz` is not a parameter: it is Room Sim's base per-link
    /// rate `rate_hz`, §4.3.)
    pub window_s: f64,
    /// Calibration windows (10). `>= 1`.
    pub min_windows: usize,
    /// Link signal name (`"rssi_dbm"` for Room Sim, decision 2); one of
    /// [`LINK_SIGNALS`].
    pub link_signal: String,
    /// Empty-room calibration run length, s (15.0). `> 0`.
    pub calib_s: f64,
    /// Tick rate, Hz (2.0). `> 0`.
    pub tick_hz: f64,
    // --- WLM (§5) ---
    /// Gate (3.0). `>= 0`.
    pub wlm_gate_z: f64,
    /// Gate floor, dB²; `None` = 0.25 for `rssi_dbm`, 0.0 otherwise
    /// ([`Config::wlm_floor_for`]). `>= 0`.
    pub wlm_floor: Option<f64>,
    // --- Radius (§8) ---
    /// 1.0, uncalibrated. `> 0` (a zero radius would publish a bare point,
    /// ADR-008 d5).
    pub wlm_radius_scale: f64,
    /// 1.0, uncalibrated. `> 0`.
    pub rti_radius_scale: f64,
    /// 1.0, uncalibrated. `> 0`.
    pub phone_radius_scale: f64,
    /// Smallest publishable radius, m (0.3). `> 0`.
    pub min_plausible_radius_m: f64,
    // --- Phone radio (§6) ---
    /// Epoch rate, Hz (5.0). `> 0`.
    pub phone_rate_hz: f64,
    /// The phone's RNG seed. `None` (the default) = the run's seed, filled
    /// by [`Config::resolved`] (the Room Sim pipeline does this); set, it
    /// is kept by [`Config::resolved`] (a plain-data phone run chooses its
    /// own) and a Room Sim run refuses it unless it equals the run's seed
    /// (§9).
    pub phone_seed: Option<u64>,
    /// RTT noise, m (1.0). `> 0`. SIM->REAL.
    pub rtt_sigma_m: f64,
    /// NLoS sigma multiplier (1.5). `>= 0`.
    pub rtt_nlos_sigma_scale: f64,
    /// NLoS bias, exponential mean, m (1.0). `>= 0`.
    pub rtt_nlos_bias_m: f64,
    /// Outlier probability (0.05). In 0..1.
    pub rtt_outlier_p: f64,
    /// Outlier magnitude low end, m (2.0). `<= rtt_outlier_hi_m`.
    pub rtt_outlier_lo_m: f64,
    /// Outlier magnitude high end, m (8.0).
    pub rtt_outlier_hi_m: f64,
    /// Per-pair offset sigma, m (0.5). `>= 0`.
    pub rtt_offset_sigma_m: f64,
    /// The estimator's persistent-error sigma for RTT, m (0.5). `>= 0`.
    pub rtt_persist_sigma_m: f64,
    /// RTT report failure probability (0.02). In 0..1.
    pub rtt_fail_p: f64,
    /// RSSI report failure probability (0.02). In 0..1.
    pub rssi_fail_p: f64,
    /// Dual-slope breakpoint, m (5.0). `> 0`. REAL (TGn).
    pub rssi_breakpoint_m: f64,
    /// Exponent beyond the breakpoint (3.5). `> 0`. REAL (TGn).
    pub rssi_far_exponent: f64,
    /// Shadowing sigma, dB (4.0). `> 0`.
    pub rssi_shadow_sigma_db: f64,
    /// The estimator's persistent-error sigma for RSSI, dB (4.0). `>= 0`.
    pub rssi_persist_sigma_db: f64,
    /// Shadowing correlation distance, m (6.0). `> 0`.
    pub rssi_shadow_corr_m: f64,
    /// NLoS loss, dB (5.0). `>= 0`.
    pub rssi_nlos_loss_db: f64,
    /// Per-device offset sigma, dB (3.0). `>= 0`.
    pub rssi_device_sigma_db: f64,
    /// Uniform jitter half-width, dB (1.0). `>= 0`.
    pub rssi_jitter_db: f64,
    // --- Phone estimators (§7) ---
    /// The estimator's path-loss exponent (2.0). `> 0`.
    pub rssi_est_exponent: f64,
    /// The estimator's 1 m level, dBm; `None` = `tx_power_dbm -
    /// 20.0*log10(FOUR_PI/LAMBDA)` ([`Config::rssi_p0`]).
    pub rssi_p0_dbm: Option<f64>,
    /// Assumed phone height, m (1.0).
    pub phone_assumed_z_m: f64,
    /// Huber constant, in sigmas (1.345). `> 0`.
    pub huber_k: f64,
    /// Kalman acceleration PSD (1.0). `> 0`.
    pub phone_accel_psd: f64,
    /// Kalman initial speed std, m/s (1.0). `> 0`.
    pub phone_init_speed_std: f64,
    /// Longest coast without an accepted update, s (3.0). `>= 0`.
    pub phone_max_coast_s: f64,
    // --- Room Sim's radio ---
    /// Room Sim's `tx_power_dbm`, dBm. `None` (the default) = Room Sim's,
    /// filled by [`Config::resolved`]; set, a Room Sim run refuses it
    /// unless it equals Room Sim's (§9). Finite.
    pub tx_power_dbm: Option<f64>,
    /// Room Sim's `carrier_hz`. `None` (the default) = Room Sim's, as
    /// `tx_power_dbm`. `> 0` (the guard of `LAMBDA = C / carrier_hz`).
    pub carrier_hz: Option<f64>,
}

impl Default for Config {
    fn default() -> Self {
        Self {
            lambda_m: 0.3,
            pitch_m: 0.5,
            alpha: 1.0,
            sigma: 1.0,
            delta_c: 0.5,
            z_threshold: 5.0,
            window_s: 1.0,
            min_windows: 10,
            link_signal: "rssi_dbm".to_string(),
            calib_s: 15.0,
            tick_hz: 2.0,
            wlm_gate_z: 3.0,
            wlm_floor: None,
            wlm_radius_scale: 1.0,
            rti_radius_scale: 1.0,
            phone_radius_scale: 1.0,
            min_plausible_radius_m: 0.3,
            phone_rate_hz: 5.0,
            phone_seed: None,
            rtt_sigma_m: 1.0,
            rtt_nlos_sigma_scale: 1.5,
            rtt_nlos_bias_m: 1.0,
            rtt_outlier_p: 0.05,
            rtt_outlier_lo_m: 2.0,
            rtt_outlier_hi_m: 8.0,
            rtt_offset_sigma_m: 0.5,
            rtt_persist_sigma_m: 0.5,
            rtt_fail_p: 0.02,
            rssi_fail_p: 0.02,
            rssi_breakpoint_m: 5.0,
            rssi_far_exponent: 3.5,
            rssi_shadow_sigma_db: 4.0,
            rssi_persist_sigma_db: 4.0,
            rssi_shadow_corr_m: 6.0,
            rssi_nlos_loss_db: 5.0,
            rssi_device_sigma_db: 3.0,
            rssi_jitter_db: 1.0,
            rssi_est_exponent: 2.0,
            rssi_p0_dbm: None,
            phone_assumed_z_m: 1.0,
            huber_k: 1.345,
            phone_accel_psd: 1.0,
            phone_init_speed_std: 1.0,
            phone_max_coast_s: 3.0,
            tx_power_dbm: None,
            carrier_hz: None,
        }
    }
}

impl Config {
    /// §9 validation; the error names the parameter (which one, when
    /// several are wrong, is not specified). Every test is NaN-safe.
    ///
    /// Every number finite (an unset `tx_power_dbm`, `carrier_hz`,
    /// `wlm_floor` or `rssi_p0_dbm` is not checked); `> 0`: `rtt_sigma_m`,
    /// `rssi_shadow_sigma_db`, `rssi_shadow_corr_m`, `rssi_breakpoint_m`,
    /// `phone_rate_hz`, `huber_k`, `rssi_far_exponent`, `rssi_est_exponent`,
    /// `phone_accel_psd`, `phone_init_speed_std`, `tick_hz`, `calib_s`,
    /// `window_s`, `carrier_hz` (when set), `min_plausible_radius_m` and the
    /// three `*_radius_scale`; `>= 0`: the NLoS, offset, persist, loss,
    /// device, jitter and coast parameters, `wlm_gate_z`, `wlm_floor`; in
    /// 0..1: `rtt_outlier_p`, `rtt_fail_p`, `rssi_fail_p`;
    /// `rtt_outlier_lo_m <= rtt_outlier_hi_m`; `min_windows >= 1`;
    /// `link_signal` one of [`LINK_SIGNALS`]. The ported modules' own checks
    /// apply to their parameters (`lambda_m` … `z_threshold`) when they are
    /// built. Room Sim's `tx_power_dbm` and `carrier_hz` are compared by
    /// [`Config::resolved`], the run's seed by the Room Sim pipeline.
    pub fn validate(&self) -> Result<(), Error> {
        let finite: [(&'static str, f64); 43] = [
            ("lambda_m", self.lambda_m),
            ("pitch_m", self.pitch_m),
            ("alpha", self.alpha),
            ("sigma", self.sigma),
            ("delta_c", self.delta_c),
            ("z_threshold", self.z_threshold),
            ("window_s", self.window_s),
            ("calib_s", self.calib_s),
            ("tick_hz", self.tick_hz),
            ("wlm_gate_z", self.wlm_gate_z),
            ("wlm_floor", self.wlm_floor.unwrap_or(0.0)),
            ("wlm_radius_scale", self.wlm_radius_scale),
            ("rti_radius_scale", self.rti_radius_scale),
            ("phone_radius_scale", self.phone_radius_scale),
            ("min_plausible_radius_m", self.min_plausible_radius_m),
            ("phone_rate_hz", self.phone_rate_hz),
            ("rtt_sigma_m", self.rtt_sigma_m),
            ("rtt_nlos_sigma_scale", self.rtt_nlos_sigma_scale),
            ("rtt_nlos_bias_m", self.rtt_nlos_bias_m),
            ("rtt_outlier_p", self.rtt_outlier_p),
            ("rtt_outlier_lo_m", self.rtt_outlier_lo_m),
            ("rtt_outlier_hi_m", self.rtt_outlier_hi_m),
            ("rtt_offset_sigma_m", self.rtt_offset_sigma_m),
            ("rtt_persist_sigma_m", self.rtt_persist_sigma_m),
            ("rtt_fail_p", self.rtt_fail_p),
            ("rssi_fail_p", self.rssi_fail_p),
            ("rssi_breakpoint_m", self.rssi_breakpoint_m),
            ("rssi_far_exponent", self.rssi_far_exponent),
            ("rssi_shadow_sigma_db", self.rssi_shadow_sigma_db),
            ("rssi_persist_sigma_db", self.rssi_persist_sigma_db),
            ("rssi_shadow_corr_m", self.rssi_shadow_corr_m),
            ("rssi_nlos_loss_db", self.rssi_nlos_loss_db),
            ("rssi_device_sigma_db", self.rssi_device_sigma_db),
            ("rssi_jitter_db", self.rssi_jitter_db),
            ("rssi_est_exponent", self.rssi_est_exponent),
            ("rssi_p0_dbm", self.rssi_p0_dbm.unwrap_or(0.0)),
            ("phone_assumed_z_m", self.phone_assumed_z_m),
            ("huber_k", self.huber_k),
            ("phone_accel_psd", self.phone_accel_psd),
            ("phone_init_speed_std", self.phone_init_speed_std),
            ("phone_max_coast_s", self.phone_max_coast_s),
            ("tx_power_dbm", self.tx_power_dbm.unwrap_or(0.0)),
            ("carrier_hz", self.carrier_hz.unwrap_or(0.0)),
        ];
        for (name, v) in finite {
            if !v.is_finite() {
                return Err(cfg_err(name, "must be a finite number"));
            }
        }
        let positive: [(&'static str, f64); 18] = [
            ("rtt_sigma_m", self.rtt_sigma_m),
            ("rssi_shadow_sigma_db", self.rssi_shadow_sigma_db),
            ("rssi_shadow_corr_m", self.rssi_shadow_corr_m),
            ("rssi_breakpoint_m", self.rssi_breakpoint_m),
            ("phone_rate_hz", self.phone_rate_hz),
            ("huber_k", self.huber_k),
            ("rssi_far_exponent", self.rssi_far_exponent),
            ("rssi_est_exponent", self.rssi_est_exponent),
            ("phone_accel_psd", self.phone_accel_psd),
            ("phone_init_speed_std", self.phone_init_speed_std),
            ("tick_hz", self.tick_hz),
            ("calib_s", self.calib_s),
            ("window_s", self.window_s),
            ("carrier_hz", self.carrier_hz.unwrap_or(1.0)),
            ("min_plausible_radius_m", self.min_plausible_radius_m),
            ("wlm_radius_scale", self.wlm_radius_scale),
            ("rti_radius_scale", self.rti_radius_scale),
            ("phone_radius_scale", self.phone_radius_scale),
        ];
        for (name, v) in positive {
            if !(v > 0.0) {
                return Err(cfg_err(name, "must be > 0"));
            }
        }
        let non_negative: [(&'static str, f64); 11] = [
            ("rtt_nlos_sigma_scale", self.rtt_nlos_sigma_scale),
            ("rtt_nlos_bias_m", self.rtt_nlos_bias_m),
            ("rtt_offset_sigma_m", self.rtt_offset_sigma_m),
            ("rtt_persist_sigma_m", self.rtt_persist_sigma_m),
            ("rssi_nlos_loss_db", self.rssi_nlos_loss_db),
            ("rssi_device_sigma_db", self.rssi_device_sigma_db),
            ("rssi_persist_sigma_db", self.rssi_persist_sigma_db),
            ("rssi_jitter_db", self.rssi_jitter_db),
            ("phone_max_coast_s", self.phone_max_coast_s),
            ("wlm_gate_z", self.wlm_gate_z),
            ("wlm_floor", self.wlm_floor.unwrap_or(0.0)),
        ];
        for (name, v) in non_negative {
            if !(v >= 0.0) {
                return Err(cfg_err(name, "must be >= 0"));
            }
        }
        let unit: [(&'static str, f64); 3] = [
            ("rtt_outlier_p", self.rtt_outlier_p),
            ("rtt_fail_p", self.rtt_fail_p),
            ("rssi_fail_p", self.rssi_fail_p),
        ];
        for (name, v) in unit {
            if !(0.0..=1.0).contains(&v) {
                return Err(cfg_err(name, "must be in 0..1 inclusive"));
            }
        }
        if !(self.rtt_outlier_lo_m <= self.rtt_outlier_hi_m) {
            return Err(cfg_err("rtt_outlier_lo_m", "must be <= rtt_outlier_hi_m"));
        }
        if self.min_windows < 1 {
            return Err(cfg_err("min_windows", "must be an integer >= 1"));
        }
        if !LINK_SIGNALS.contains(&self.link_signal.as_str()) {
            return Err(cfg_err(
                "link_signal",
                "must be rssi_dbm, total_power_db or static_suppressed_amplitude",
            ));
        }
        Ok(())
    }

    /// A validated copy with Room Sim's values filled in (§9, §10.2):
    /// `phone_seed` = `seed` when unset (a set one is kept: a plain-data
    /// phone run chooses its own; the Room Sim pipeline's
    /// `resolve_config` refuses one that is not the run's seed),
    /// `tx_power_dbm` and `carrier_hz` = Room Sim's. A `tx_power_dbm` or
    /// `carrier_hz` already set must equal Room Sim's, else the error names
    /// it. `rssi_p0_dbm` stays as given; unset, it is derived where used
    /// ([`Config::rssi_p0`]).
    pub fn resolved(&self, seed: u64, tx_power_dbm: f64, carrier_hz: f64) -> Result<Config, Error> {
        self.validate()?;
        for (param, mine, theirs) in [
            ("tx_power_dbm", self.tx_power_dbm, tx_power_dbm),
            ("carrier_hz", self.carrier_hz, carrier_hz),
        ] {
            if let Some(mine) = mine {
                if !(mine == theirs) {
                    return Err(Error::Config {
                        param,
                        reason: format!("{mine} is not Room Sim's {theirs} (§9; leave it unset)"),
                    });
                }
            }
        }
        let out = Config {
            phone_seed: Some(self.phone_seed.unwrap_or(seed)),
            tx_power_dbm: Some(tx_power_dbm),
            carrier_hz: Some(carrier_hz),
            ..self.clone()
        };
        out.validate()?;
        Ok(out)
    }

    /// `(phone_seed, tx_power_dbm, carrier_hz)`, or an error naming the
    /// first that is unset (the phone needs them: [`Config::resolved`]).
    pub fn room_sim_values(&self) -> Result<(u64, f64, f64), Error> {
        let unset = |param: &'static str| {
            cfg_err(
                param,
                "not set; the phone needs it (Config::resolved, or the Room Sim pipeline, fills it)",
            )
        };
        let seed = self.phone_seed.ok_or_else(|| unset("phone_seed"))?;
        let tx = self.tx_power_dbm.ok_or_else(|| unset("tx_power_dbm"))?;
        let fc = self.carrier_hz.ok_or_else(|| unset("carrier_hz"))?;
        Ok((seed, tx, fc))
    }

    /// [`Config::validate`], and refuse a config whose Room Sim values are
    /// unset ([`Config::room_sim_values`]).
    pub fn require_resolved(&self) -> Result<(), Error> {
        self.validate()?;
        self.room_sim_values().map(|_| ())
    }

    /// `LAMBDA = C / carrier_hz` (guard: [`Config::validate`]'s
    /// `carrier_hz > 0`); an error naming `carrier_hz` when it is unset.
    pub fn lambda(&self) -> Result<f64, Error> {
        match self.carrier_hz {
            Some(fc) => Ok(wavelength(fc)),
            None => Err(cfg_err(
                "carrier_hz",
                "not set; LAMBDA needs Room Sim's carrier (Config::resolved)",
            )),
        }
    }

    /// `20.0 * log10(FOUR_PI / LAMBDA)`: the free-space loss at 1 m (§6.5).
    pub fn pl0_db(&self) -> Result<f64, Error> {
        Ok(20.0 * (FOUR_PI / self.lambda()?).log10())
    }

    /// `rssi_p0_dbm`, defaulting to `tx_power_dbm - 20.0*log10(FOUR_PI /
    /// LAMBDA)`, the true 1 m level (§7.2); an error naming `rssi_p0_dbm`
    /// when it is unset and Room Sim's values are too.
    pub fn rssi_p0(&self) -> Result<f64, Error> {
        if let Some(p0) = self.rssi_p0_dbm {
            return Ok(p0);
        }
        match (self.tx_power_dbm, self.carrier_hz) {
            (Some(tx), Some(fc)) => Ok(tx - 20.0 * (FOUR_PI / wavelength(fc)).log10()),
            _ => Err(cfg_err(
                "rssi_p0_dbm",
                "not set, and tx_power_dbm or carrier_hz is unset to derive it (Config::resolved)",
            )),
        }
    }

    /// `wlm_floor`, defaulting to 0.25 for `rssi_dbm` and 0.0 for any other
    /// signal (§5, §9).
    pub fn wlm_floor_for(&self, signal_name: &str) -> f64 {
        match self.wlm_floor {
            Some(f) => f,
            None if signal_name == "rssi_dbm" => 0.25,
            None => 0.0,
        }
    }
}

// ===================================================================
// Geometry inputs and line of sight (§6.2)
// ===================================================================

/// A node, from `scene.nodes` (SCENE), z kept for the phone.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhoneNode {
    pub id: u8,
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// A wall of Room Sim's `W` (spec/roomsim.md §3.1, implied walls
/// included): centreline `a -> b` and height `h`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Wall {
    pub a: (f64, f64),
    pub b: (f64, f64),
    pub h: f64,
}

/// An object's box (spec/roomsim.md §3.1): centre, size `(sl, sb, sh)`,
/// `cs = cos(psi)`, `sn = sin(psi)`, and whether it is a door or window
/// (an opening, which never blocks).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BoxObj {
    pub cx: f64,
    pub cy: f64,
    pub cz: f64,
    pub sl: f64,
    pub sb: f64,
    pub sh: f64,
    pub cs: f64,
    pub sn: f64,
    /// A `door` or `window`: never blocks (§6.2).
    pub opening: bool,
}

impl BoxObj {
    /// A box with yaw in degrees: `psi = (yaw_deg * PI) / 180.0` exactly,
    /// `cs = cos(psi)`, `sn = sin(psi)` as separate libm calls
    /// (spec/roomsim.md §1, §3.1).
    #[allow(clippy::too_many_arguments)]
    pub fn from_yaw_deg(
        cx: f64,
        cy: f64,
        cz: f64,
        sl: f64,
        sb: f64,
        sh: f64,
        yaw_deg: f64,
        opening: bool,
    ) -> Self {
        let psi = (yaw_deg * PI) / 180.0;
        Self {
            cx,
            cy,
            cz,
            sl,
            sb,
            sh,
            cs: crate::num::cos(psi),
            sn: crate::num::sin(psi),
            opening,
        }
    }

    /// World to object-local (spec/roomsim.md §3.1): `dx = p.x - cx`,
    /// `dy = p.y - cy`; `lx = dx*cs + dy*sn`, `ly = -dx*sn + dy*cs`,
    /// `lz = p.z - cz`.
    pub fn to_local(&self, p: [f64; 3]) -> [f64; 3] {
        let dx = p[0] - self.cx;
        let dy = p[1] - self.cy;
        [
            dx * self.cs + dy * self.sn,
            -dx * self.sn + dy * self.cs,
            p[2] - self.cz,
        ]
    }
}

/// A person's cylinder as Room Sim §8.5 defines it: centre `(cx, cy)`,
/// radius `body_radius_m`, from the floor to `top`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    pub cx: f64,
    pub cy: f64,
    pub radius: f64,
    pub top: f64,
}

/// The static blockers of §6.2: Room Sim's walls and every object box.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Blockers {
    pub walls: Vec<Wall>,
    pub boxes: Vec<BoxObj>,
}

/// spec/roomsim.md §6.1 `hit(P, Q)` with `o = 0` (the centreline):
/// `r = (Q-P).xy`, `w = B - A`, `den = r.x*w.y - r.y*w.x`; none if
/// `den == 0`; `s = ((A-P).x*w.y - (A-P).y*w.x) / den`,
/// `v = ((A-P).x*r.y - (A-P).y*r.x) / den`; a hit iff `0 < s < 1`,
/// `0 <= v <= 1` and `0 <= X.z <= h` with `X.z = P.z + s*(Q.z-P.z)`.
pub fn wall_blocks(p: [f64; 3], q: [f64; 3], wall: &Wall) -> bool {
    let r = (q[0] - p[0], q[1] - p[1]);
    let w = (wall.b.0 - wall.a.0, wall.b.1 - wall.a.1);
    let den = r.0 * w.1 - r.1 * w.0;
    if den == 0.0 {
        return false;
    }
    let ap = (wall.a.0 - p[0], wall.a.1 - p[1]);
    let s = (ap.0 * w.1 - ap.1 * w.0) / den;
    let v = (ap.0 * r.1 - ap.1 * r.0) / den;
    if !(0.0 < s && s < 1.0 && (0.0..=1.0).contains(&v)) {
        return false;
    }
    let z = p[2] + s * (q[2] - p[2]);
    (0.0..=wall.h).contains(&z)
}

/// spec/roomsim.md §7.4: the slab test of a local segment against a box
/// of half-extents `half`, starting from `[0, 1]`.
fn slab_meets(start: &[f64; 3], end: &[f64; 3], half: &[f64; 3]) -> bool {
    let mut t_enter = 0.0;
    let mut t_exit = 1.0;
    for k in 0..3 {
        let p = start[k];
        let h = half[k];
        let d = end[k] - p;
        if d == 0.0 {
            if p.abs() > h {
                return false;
            }
        } else {
            let t1 = (-h - p) / d;
            let t2 = (h - p) / d;
            t_enter = pymax(t_enter, pymin(t1, t2));
            t_exit = pymin(t_exit, pymax(t1, t2));
        }
    }
    t_exit > t_enter
}

/// spec/roomsim.md §7.4's box test of segment `p -> q` (both ends to
/// object-local, half-extents `sl/2`, `sb/2`, `sh/2`). A door or window
/// never blocks (§6.2).
pub fn box_blocks(p: [f64; 3], q: [f64; 3], b: &BoxObj) -> bool {
    if b.opening {
        return false;
    }
    let a = b.to_local(p);
    let e = b.to_local(q);
    slab_meets(&a, &e, &[b.sl / 2.0, b.sb / 2.0, b.sh / 2.0])
}

/// spec/roomsim.md §8.5 steps 1-3's geometry for segment `p1 -> p2`:
/// `L2 == 0`: not blocked; `u* <= 0` or `u* >= 1`: blocked iff the nearer
/// end is inside the cylinder (xy distance `< radius` and `z <= top`);
/// otherwise blocked iff `d_perp < radius` and `z* <= top`.
pub fn cylinder_blocks(p1: [f64; 3], p2: [f64; 3], c: &Cylinder) -> bool {
    let r = (p2[0] - p1[0], p2[1] - p1[1]);
    let l2 = r.0 * r.0 + r.1 * r.1;
    if l2 == 0.0 {
        return false;
    }
    let u = ((c.cx - p1[0]) * r.0 + (c.cy - p1[1]) * r.1) / l2;
    if u <= 0.0 || u >= 1.0 {
        let end = if u <= 0.0 { p1 } else { p2 };
        return len2(end[0] - c.cx, end[1] - c.cy) < c.radius && end[2] <= c.top;
    }
    let z = p1[2] + u * (p2[2] - p1[2]);
    let d_perp = len2((p1[0] + u * r.0) - c.cx, (p1[1] + u * r.1) - c.cy);
    d_perp < c.radius && z <= c.top
}

/// §6.2: the phone-node segment `p -> n` is blocked if it meets any wall,
/// any object that is not a door or window, or any person's cylinder (the
/// carrier included). A fact of the geometry, not a draw.
pub fn blocked(p: [f64; 3], n: [f64; 3], blockers: &Blockers, cylinders: &[Cylinder]) -> bool {
    blockers.walls.iter().any(|w| wall_blocks(p, n, w))
        || blockers.boxes.iter().any(|b| box_blocks(p, n, b))
        || cylinders.iter().any(|c| cylinder_blocks(p, n, c))
}

// ===================================================================
// Radio simulation (§6)
// ===================================================================

/// One truth sample `j` (§6.1): its time, the phone's true `(x, y, z)` and
/// every person's cylinder (Room Sim §8.5).
#[derive(Debug, Clone, Copy)]
pub struct TruthSample<'a> {
    pub t: f64,
    pub phone: [f64; 3],
    pub cylinders: &'a [Cylinder],
}

/// One node's outcome at a fired epoch of the radio (§6.4, §6.5): the
/// reports, plus SIM diagnostics that are **not** part of the §11 record
/// (the record's entry is [`ReportRecord`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeReport {
    /// The node id.
    pub node: u8,
    /// The RTT range, m; `None` when `u_fail < rtt_fail_p`.
    pub rtt_m: Option<f64>,
    /// The RSSI, integer dBm in -100..0; `None` when `u_fail <
    /// rssi_fail_p` or `v` is not finite.
    pub rssi_dbm: Option<i32>,
    /// The pre-rounding `v` of §6.5 (computed whether or not reported).
    /// A diagnostic: never in a record.
    pub rssi_v: f64,
    /// This node's shadowing `S`, dB, at this epoch (hidden SIM state; a
    /// diagnostic, never in a record).
    pub shadow_db: f64,
    /// Whether the phone-node segment is blocked (§6.2).
    pub blocked: bool,
}

/// One fired epoch of the radio (§6.1).
#[derive(Debug, Clone, PartialEq)]
pub struct RadioEpoch {
    /// `t_j` of the sample it fired at (never `e / phone_rate_hz`).
    pub t: f64,
    /// That sample's phone `(x, y, z)`.
    pub truth: [f64; 3],
    /// One per node, ids ascending.
    pub reports: Vec<NodeReport>,
}

#[derive(Debug, Clone)]
struct NodeRadio {
    node: PhoneNode,
    rtt: Rng,
    rssi: Rng,
    off: f64,
    dev: f64,
    shadow: Option<f64>,
}

/// The phone's radio (§6): epochs, line of sight, the RNG streams and the
/// RTT and RSSI reports. SIM only.
#[derive(Debug, Clone)]
pub struct Radio {
    cfg: Config,
    tx_power_dbm: f64,
    pl0: f64,
    blockers: Blockers,
    nodes: Vec<NodeRadio>,
    e_next: u64,
    last_t: Option<f64>,
    prev_xy: Option<(f64, f64)>,
}

/// The largest `t * phone_rate_hz` the epoch counter accepts (2^53, where
/// `float(e_next)` stops being exact).
const MAX_EPOCH_INDEX: f64 = 9007199254740992.0;

/// §6.1's nodes: a sorted copy (by id), ids distinct, at least one, every
/// position finite; else an error naming `nodes`.
pub fn sorted_nodes(nodes: &[PhoneNode]) -> Result<Vec<PhoneNode>, Error> {
    for n in nodes {
        if !(n.x.is_finite() && n.y.is_finite() && n.z.is_finite()) {
            return Err(input_err(
                "nodes",
                format!("node {} has a non-finite position", n.id),
            ));
        }
    }
    let mut out = nodes.to_vec();
    out.sort_by_key(|n| n.id);
    for w in out.windows(2) {
        if w[0].id == w[1].id {
            return Err(input_err("nodes", format!("duplicate node id {}", w[1].id)));
        }
    }
    if out.is_empty() {
        return Err(input_err("nodes", "the phone needs at least one node"));
    }
    Ok(out)
}

fn check_blockers(b: &Blockers) -> Result<(), Error> {
    for w in &b.walls {
        if ![w.a.0, w.a.1, w.b.0, w.b.1, w.h]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(input_err("walls", "every number must be finite"));
        }
    }
    for o in &b.boxes {
        if ![o.cx, o.cy, o.cz, o.sl, o.sb, o.sh, o.cs, o.sn]
            .iter()
            .all(|v| v.is_finite())
        {
            return Err(input_err("objects", "every number must be finite"));
        }
    }
    Ok(())
}

/// §6.4's `d = sqrt(((px-nx)*(px-nx) + (py-ny)*(py-ny)) + (pz-nz)*(pz-nz))`.
fn node_distance(p: [f64; 3], n: &PhoneNode) -> f64 {
    let [px, py, pz] = p;
    (((px - n.x) * (px - n.x) + (py - n.y) * (py - n.y)) + (pz - n.z) * (pz - n.z)).sqrt()
}

impl Radio {
    /// Build the radio: refuses any label but `"SIM"`, validates `cfg`
    /// (§9) and needs its Room Sim values set ([`Config::require_resolved`]),
    /// takes the nodes in id order (a sorted copy; ids distinct, at least
    /// one, positions finite, else an error naming `nodes`), checks the
    /// blockers (finite, else `walls` or `objects`), and makes the creation
    /// draws (§6.3): per node in id order, RTT stream `off_n =
    /// rtt_offset_sigma_m * gauss()`, RSSI stream `dev_n =
    /// rssi_device_sigma_db * gauss()`.
    pub fn new(
        cfg: &Config,
        nodes: &[PhoneNode],
        blockers: &Blockers,
        label: &str,
    ) -> Result<Self, Error> {
        if label != crate::LABEL {
            return Err(Error::Label);
        }
        cfg.require_resolved()?;
        let (seed, tx_power_dbm, _) = cfg.room_sim_values()?;
        let pl0 = cfg.pl0_db()?;
        let nodes = sorted_nodes(nodes)?;
        check_blockers(blockers)?;
        let mut radios = Vec::with_capacity(nodes.len());
        for node in nodes {
            let n = u64::from(node.id);
            let mut rtt = Rng::stream(seed, RTT_STREAM_BASE + n);
            let mut rssi = Rng::stream(seed, RSSI_STREAM_BASE + n);
            let off = cfg.rtt_offset_sigma_m * rtt.gauss();
            let dev = cfg.rssi_device_sigma_db * rssi.gauss();
            radios.push(NodeRadio {
                node,
                rtt,
                rssi,
                off,
                dev,
                shadow: None,
            });
        }
        Ok(Self {
            cfg: cfg.clone(),
            tx_power_dbm,
            pl0,
            blockers: blockers.clone(),
            nodes: radios,
            e_next: 0,
            last_t: None,
            prev_xy: None,
        })
    }

    /// The nodes, ids ascending.
    pub fn nodes(&self) -> Vec<PhoneNode> {
        self.nodes.iter().map(|n| n.node).collect()
    }

    /// `(off_n, dev_n)` of each node, ids ascending (the creation draws).
    pub fn offsets(&self) -> Vec<(f64, f64)> {
        self.nodes.iter().map(|n| (n.off, n.dev)).collect()
    }

    /// The next epoch index `e_next` (§6.1).
    pub fn next_epoch(&self) -> u64 {
        self.e_next
    }

    /// Feed truth sample `j` (§6.1). If `t_j >= float(e_next) /
    /// phone_rate_hz` one epoch fires at this sample (time `t_j`, truth
    /// sample `j`'s), then `e_next += 1` while `t_j >= float(e_next) /
    /// phone_rate_hz` (epochs passed over are dropped: no draws, no record).
    ///
    /// The §6.1 input contract, else an [`Error::Input`]: `t` finite, not
    /// below the previous sample's, and `t * phone_rate_hz < 2^53` (`t`);
    /// the phone's coordinates finite, and its distance `d` to every node
    /// finite (`phone`); every cylinder number finite (`cylinders`). A
    /// refused sample changes nothing.
    pub fn step(&mut self, sample: &TruthSample<'_>) -> Result<Option<RadioEpoch>, Error> {
        let t = sample.t;
        if !t.is_finite() {
            return Err(input_err("t", "must be finite"));
        }
        if let Some(last) = self.last_t {
            if !(t >= last) {
                return Err(input_err("t", "truth samples must be time-ordered"));
            }
        }
        if !sample.phone.iter().all(|v| v.is_finite()) {
            return Err(input_err("phone", "must be finite"));
        }
        for nr in &self.nodes {
            if !node_distance(sample.phone, &nr.node).is_finite() {
                return Err(input_err(
                    "phone",
                    format!("its distance to node {} is not finite", nr.node.id),
                ));
            }
        }
        for c in sample.cylinders {
            if !(c.cx.is_finite() && c.cy.is_finite() && c.radius.is_finite() && c.top.is_finite())
            {
                return Err(input_err("cylinders", "every number must be finite"));
            }
        }
        let rate = self.cfg.phone_rate_hz;
        if !(t * rate < MAX_EPOCH_INDEX) {
            return Err(input_err("t", "beyond the epoch counter's range"));
        }
        self.last_t = Some(t);
        if !(t >= (self.e_next as f64) / rate) {
            return Ok(None);
        }
        // e_next += 1 while t >= float(e_next) / rate. A long gap is
        // skipped ahead first; the skip is taken only when every index it
        // passes satisfies the loop's own test (float division by a
        // positive rate is monotone), so the result is the loop's.
        let guess = (t * rate).floor();
        if guess > 2.0 {
            let cand = (guess - 2.0) as u64;
            if cand > self.e_next && t >= ((cand - 1) as f64) / rate {
                self.e_next = cand;
            }
        }
        while t >= (self.e_next as f64) / rate {
            self.e_next += 1;
        }
        Ok(Some(self.fire(sample)))
    }

    fn fire(&mut self, sample: &TruthSample<'_>) -> RadioEpoch {
        let cfg = &self.cfg;
        let [px, py, _] = sample.phone;
        let pl0 = self.pl0;
        let bp = cfg.rssi_breakpoint_m;
        let mut reports = Vec::with_capacity(self.nodes.len());
        for nr in &mut self.nodes {
            let (nx, ny, nz) = (nr.node.x, nr.node.y, nr.node.z);
            let is_blocked = blocked(sample.phone, [nx, ny, nz], &self.blockers, sample.cylinders);
            // Draws, always in this order whatever the geometry or outcome.
            let u_fail = nr.rtt.uniform();
            let g = nr.rtt.gauss();
            let u_out = nr.rtt.uniform();
            let u_mag = nr.rtt.uniform();
            let u_b = nr.rtt.uniform();
            let u_fail_s = nr.rssi.uniform();
            let g_s = nr.rssi.gauss();
            let u_j = nr.rssi.uniform();

            // §6.4 RTT (d is finite: step checked it).
            let d = node_distance(sample.phone, &nr.node);
            let sig = if is_blocked {
                cfg.rtt_sigma_m * cfg.rtt_nlos_sigma_scale
            } else {
                cfg.rtt_sigma_m
            };
            let mut r = (d + nr.off) + sig * g;
            if is_blocked {
                r += -(cfg.rtt_nlos_bias_m * (1.0 - u_b).ln());
            }
            if u_out < cfg.rtt_outlier_p {
                r += cfg.rtt_outlier_lo_m + (cfg.rtt_outlier_hi_m - cfg.rtt_outlier_lo_m) * u_mag;
            }
            r = if r > 0.0 { r } else { 0.0 };
            let rtt_m = if u_fail < cfg.rtt_fail_p {
                None
            } else {
                Some(r)
            };

            // §6.5 shadowing, before the u_fail test and whatever it decides.
            let s = match (nr.shadow, self.prev_xy) {
                (Some(s_prev), Some((xp, yp))) => {
                    let disp = ((px - xp) * (px - xp) + (py - yp) * (py - yp)).sqrt();
                    let rho = (-(disp / cfg.rssi_shadow_corr_m)).exp();
                    rho * s_prev + (1.0 - rho * rho).sqrt() * (cfg.rssi_shadow_sigma_db * g_s)
                }
                _ => cfg.rssi_shadow_sigma_db * g_s,
            };
            nr.shadow = Some(s);

            // §6.5 RSSI.
            let dd = if d > 0.1 { d } else { 0.1 };
            let pl = if dd <= bp {
                pl0 + 20.0 * dd.log10()
            } else {
                (pl0 + 20.0 * bp.log10()) + (10.0 * cfg.rssi_far_exponent) * (dd / bp).log10()
            };
            let nlos = if is_blocked {
                cfg.rssi_nlos_loss_db
            } else {
                0.0
            };
            let v = ((((self.tx_power_dbm - pl) - nlos) + s) + nr.dev)
                + (-cfg.rssi_jitter_db + (2.0 * cfg.rssi_jitter_db) * u_j);
            // No report when u_fail < rssi_fail_p or v is not finite (its
            // draws and S are kept as computed).
            let rssi_dbm = if u_fail_s < cfg.rssi_fail_p || !v.is_finite() {
                None
            } else {
                Some(clamp(round_half_even(v), -100.0, 0.0) as i32)
            };
            reports.push(NodeReport {
                node: nr.node.id,
                rtt_m,
                rssi_dbm,
                rssi_v: v,
                shadow_db: s,
                blocked: is_blocked,
            });
        }
        self.prev_xy = Some((px, py));
        RadioEpoch {
            t: sample.t,
            truth: sample.phone,
            reports,
        }
    }
}

// ===================================================================
// Fixes (§7.1, §7.2)
// ===================================================================

/// The two phone methods.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Method {
    Rtt,
    Rssi,
}

/// One report given to an estimator: the node (SCENE position) and the
/// value (`r_n`, m, for RTT; `rssi_n` as a float, dBm, for RSSI).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ranging {
    pub node: PhoneNode,
    pub value: f64,
}

/// A fix of one method at one epoch.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Fix {
    pub x: f64,
    pub y: f64,
    /// `R = (sigma*sigma) * inv(N)`, each entry (`rtt_sigma_m` or
    /// `rssi_shadow_sigma_db`).
    pub r: Mat2,
    /// `inv(N)` of the final normal matrix (§7.4's `B` uses it).
    pub inv_n: Mat2,
    /// RTT: `Σ (u*u)` over the reports used in the final pass (unweighted;
    /// about chi-squared with `m - 2` degrees of freedom under the Gaussian
    /// model alone). `None` for RSSI.
    pub chi2: Option<f64>,
    /// Reports used in the final pass.
    pub used: usize,
    /// Gauss-Newton iterations made (1..=200).
    pub iterations: usize,
    /// The start `(x0, y0)`.
    pub start: (f64, f64),
}

/// The sums at a point (§7.1, §7.2), each a Neumaier Σ, reports ascending.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Sums {
    n00: f64,
    n01: f64,
    n11: f64,
    g0: f64,
    g1: f64,
    f: f64,
    chi2: f64,
    used: usize,
}

impl Sums {
    /// `N00*N11 - N01*N01`.
    fn det(&self) -> f64 {
        self.n00 * self.n11 - self.n01 * self.n01
    }
    /// `not (det > 1e-12 * (N00*N11))`.
    fn degenerate(&self) -> bool {
        !(self.det() > 1e-12 * (self.n00 * self.n11))
    }
    fn n(&self) -> Mat2 {
        [[self.n00, self.n01], [self.n01, self.n11]]
    }
}

/// The anchor sums of §7.1's *Start* over reports nodes ascending (report
/// 1 the first; `n = 2..m`): `A_n = (2.0*(nx_n - nx_1), 2.0*(ny_n -
/// ny_1))`, `n00 = Σ A_n.x*A_n.x`, `n01 = Σ A_n.x*A_n.y`,
/// `n11 = Σ A_n.y*A_n.y`. Returns `(n00, n01, n11)`.
pub fn anchor_normal(reports: &[Ranging]) -> (f64, f64, f64) {
    let mut n00 = Neumaier::new();
    let mut n01 = Neumaier::new();
    let mut n11 = Neumaier::new();
    if let Some(first) = reports.first() {
        for rp in &reports[1..] {
            let ax = 2.0 * (rp.node.x - first.node.x);
            let ay = 2.0 * (rp.node.y - first.node.y);
            n00.add(ax * ax);
            n01.add(ax * ay);
            n11.add(ay * ay);
        }
    }
    (n00.value(), n01.value(), n11.value())
}

/// `not (det > 1e-9 * ((n00 + n11)*(n00 + n11)))` with `det = n00*n11 -
/// n01*n01` (a NaN counts as collinear). `det / (n00 + n11)^2` is the
/// normal matrix's eigenvalue product over the square of their sum, so it
/// does not depend on the anchors' line direction: anchors collinear to
/// rounding are caught at any angle (an axis-aligned line too, where `n00`
/// or `n11` is itself rounding-sized), and anchors within about 3e-5 of
/// their span from a line count as collinear.
fn start_degenerate(n00: f64, n01: f64, n11: f64) -> bool {
    let det = n00 * n11 - n01 * n01;
    !(det > ANCHOR_DET_REL * ((n00 + n11) * (n00 + n11)))
}

/// §7's collinear rule: the reports' anchors (nodes ascending) fail
/// `det > 1e-9 * ((n00 + n11)*(n00 + n11))` of [`anchor_normal`], at any
/// line direction. A range or level set from anchors on one line cannot
/// tell the two sides of the line apart, so both fixes then give no fix
/// (`degenerate`).
pub fn anchors_collinear(reports: &[Ranging]) -> bool {
    let (n00, n01, n11) = anchor_normal(reports);
    start_degenerate(n00, n01, n11)
}

/// The RTT start of §7.1 from reports nodes ascending: the linearised
/// least squares on horizontal ranges `h_n`. `Err(NoReports)` for no
/// reports; `Err(Degenerate)` when its normal matrix fails `det > 1e-9 *
/// ((n00 + n11)*(n00 + n11))` (the collinear rule, §7).
pub fn rtt_start(cfg: &Config, reports: &[Ranging]) -> Result<(f64, f64), Reason> {
    let first = reports.first().ok_or(Reason::NoReports)?;
    let zh = cfg.phone_assumed_z_m;
    let h = |rp: &Ranging| {
        let dz = zh - rp.node.z;
        let r = rp.value;
        if r * r > dz * dz {
            (r * r - dz * dz).sqrt()
        } else {
            0.0
        }
    };
    let (nx1, ny1) = (first.node.x, first.node.y);
    let h1 = h(first);
    let mut n00 = Neumaier::new();
    let mut n01 = Neumaier::new();
    let mut n11 = Neumaier::new();
    let mut c0 = Neumaier::new();
    let mut c1 = Neumaier::new();
    for rp in &reports[1..] {
        let (nx, ny) = (rp.node.x, rp.node.y);
        let hn = h(rp);
        let ax = 2.0 * (nx - nx1);
        let ay = 2.0 * (ny - ny1);
        let b = ((((nx * nx) + (ny * ny)) - (nx1 * nx1)) - (ny1 * ny1)) - ((hn * hn) - (h1 * h1));
        n00.add(ax * ax);
        n01.add(ax * ay);
        n11.add(ay * ay);
        c0.add(ax * b);
        c1.add(ay * b);
    }
    let (n00, n01, n11, c0, c1) = (
        n00.value(),
        n01.value(),
        n11.value(),
        c0.value(),
        c1.value(),
    );
    if start_degenerate(n00, n01, n11) {
        return Err(Reason::Degenerate);
    }
    let det = n00 * n11 - n01 * n01;
    Ok(((n11 * c0 - n01 * c1) / det, (n00 * c1 - n01 * c0) / det))
}

/// The RSSI start of §7.2 from reports nodes ascending: `w_n =
/// 10^(rssi_n / 10.0)`; `Err(Degenerate)` if `not (Σ w_n > 0)` or `Σ w_n`
/// is not finite; else `x0 = (Σ (w_n*nx)) / (Σ w_n)`, `y0` likewise.
pub fn rssi_start(reports: &[Ranging]) -> Result<(f64, f64), Reason> {
    let w: Vec<f64> = reports.iter().map(|rp| pow10(rp.value / 10.0)).collect();
    let sw = Neumaier::sum(w.iter().copied());
    if !(sw > 0.0 && sw.is_finite()) {
        return Err(Reason::Degenerate);
    }
    let sx = Neumaier::sum(w.iter().zip(reports).map(|(&wn, rp)| wn * rp.node.x));
    let sy = Neumaier::sum(w.iter().zip(reports).map(|(&wn, rp)| wn * rp.node.y));
    Ok((sx / sw, sy / sw))
}

fn rtt_sums(cfg: &Config, reports: &[Ranging], x: f64, y: f64) -> Sums {
    let zh = cfg.phone_assumed_z_m;
    let k = cfg.huber_k;
    let mut n00 = Neumaier::new();
    let mut n01 = Neumaier::new();
    let mut n11 = Neumaier::new();
    let mut g0 = Neumaier::new();
    let mut g1 = Neumaier::new();
    let mut f = Neumaier::new();
    let mut chi2 = Neumaier::new();
    let mut used = 0;
    for rp in reports {
        let dz = zh - rp.node.z;
        let dx = x - rp.node.x;
        let dy = y - rp.node.y;
        let rho = len3(dx, dy, dz);
        if rho < 1e-9 {
            continue;
        }
        let res = rp.value - rho;
        let u = res / cfg.rtt_sigma_m;
        let w = if u > k { k / u } else { 1.0 };
        let hx = dx / rho;
        let hy = dy / rho;
        n00.add((w * hx) * hx);
        n01.add((w * hx) * hy);
        n11.add((w * hy) * hy);
        g0.add((w * hx) * res);
        g1.add((w * hy) * res);
        f.add(if u <= k {
            0.5 * (u * u)
        } else {
            k * u - 0.5 * (k * k)
        });
        chi2.add(u * u);
        used += 1;
    }
    Sums {
        n00: n00.value(),
        n01: n01.value(),
        n11: n11.value(),
        g0: g0.value(),
        g1: g1.value(),
        f: f.value(),
        chi2: chi2.value(),
        used,
    }
}

fn rssi_sums(cfg: &Config, p0: f64, reports: &[Ranging], x: f64, y: f64) -> Sums {
    let zh = cfg.phone_assumed_z_m;
    let slope = 10.0 * cfg.rssi_est_exponent;
    let gk = slope / ln10();
    let mut n00 = Neumaier::new();
    let mut n01 = Neumaier::new();
    let mut n11 = Neumaier::new();
    let mut g0 = Neumaier::new();
    let mut g1 = Neumaier::new();
    let mut f = Neumaier::new();
    let mut used = 0;
    for rp in reports {
        let dz = zh - rp.node.z;
        let dx = x - rp.node.x;
        let dy = y - rp.node.y;
        let rho = len3(dx, dy, dz);
        if rho < 0.1 {
            continue;
        }
        let pred = p0 - slope * rho.log10();
        let res = rp.value - pred;
        let jx = ((-gk) * dx) / (rho * rho);
        let jy = ((-gk) * dy) / (rho * rho);
        n00.add(jx * jx);
        n01.add(jx * jy);
        n11.add(jy * jy);
        g0.add(jx * res);
        g1.add(jy * res);
        f.add(0.5 * (res * res));
        used += 1;
    }
    Sums {
        n00: n00.value(),
        n01: n01.value(),
        n11: n11.value(),
        g0: g0.value(),
        g1: g1.value(),
        f: f.value(),
        chi2: f64::NAN,
        used,
    }
}

/// §7.1's Gauss-Newton with step halving, at most [`GN_MAX_ITERATIONS`]
/// iterations from `(x0, y0)`.
fn gauss_newton<S: Fn(f64, f64) -> Sums>(
    x0: f64,
    y0: f64,
    sums: S,
) -> Result<(f64, f64, Sums, usize), Reason> {
    let (mut x, mut y) = (x0, y0);
    let mut iterations = 0;
    let mut converged = false;
    while iterations < GN_MAX_ITERATIONS {
        iterations += 1;
        let s = sums(x, y);
        if s.degenerate() {
            return Err(Reason::Degenerate);
        }
        let det = s.det();
        let sx = (s.n11 * s.g0 - s.n01 * s.g1) / det;
        let sy = (s.n00 * s.g1 - s.n01 * s.g0) / det;
        let mut a = 1.0;
        let mut halvings = 0;
        while !(sums(x + a * sx, y + a * sy).f <= s.f) && halvings < GN_MAX_HALVINGS {
            a *= 0.5;
            halvings += 1;
        }
        x += a * sx;
        y += a * sy;
        if a * (sx * sx + sy * sy).sqrt() < GN_STOP_M {
            converged = true;
            break;
        }
    }
    if !converged {
        return Err(Reason::NotConverged);
    }
    let s = sums(x, y);
    if s.degenerate() {
        return Err(Reason::Degenerate);
    }
    Ok((x, y, s, iterations))
}

fn finish(
    x: f64,
    y: f64,
    s: &Sums,
    iterations: usize,
    start: (f64, f64),
    sigma: f64,
    chi2: Option<f64>,
) -> Fix {
    let n = s.n();
    let inv_n = inv2(&n, det2(&n));
    Fix {
        x,
        y,
        r: mat_scale(sigma * sigma, &inv_n),
        inv_n,
        chi2,
        used: s.used,
        iterations,
        start,
    }
}

/// The reports in node order: §7 defines each method on its reports
/// "nodes ascending". A stable sort by node id, as the Python's `_gather`
/// sorts them (`sorted(reports, key=node)`); reports already in order are
/// used as given.
fn nodes_ascending(reports: &[Ranging]) -> Cow<'_, [Ranging]> {
    if reports.windows(2).all(|w| w[0].node.id <= w[1].node.id) {
        Cow::Borrowed(reports)
    } else {
        let mut sorted = reports.to_vec();
        sorted.sort_by_key(|r| r.node.id);
        Cow::Owned(sorted)
    }
}

/// §7's input contract on reports taken nodes ascending: every value
/// finite and no two reports from one node, else the fix is refused (an
/// error naming `reports`); every node position finite, else `nodes`.
pub fn check_reports(reports: &[Ranging]) -> Result<(), Error> {
    for rp in reports {
        let n = rp.node;
        if !(n.x.is_finite() && n.y.is_finite() && n.z.is_finite()) {
            return Err(input_err(
                "nodes",
                format!("node {} has a non-finite position", n.id),
            ));
        }
        if !rp.value.is_finite() {
            return Err(input_err(
                "reports",
                format!("node {}: the report value is not finite", n.id),
            ));
        }
    }
    for w in reports.windows(2) {
        if w[0].node.id == w[1].node.id {
            return Err(input_err(
                "reports",
                format!("two reports from node {}", w[1].node.id),
            ));
        }
    }
    Ok(())
}

/// The RTT fix of §7.1 from one epoch's reports `(n, r_n)`, taken nodes
/// ascending (reports in any order are sorted stably by node id first).
///
/// The outer `Result` refuses: an invalid `cfg` (§9, the error names the
/// parameter) or invalid reports ([`check_reports`], `reports`). The inner
/// one is the fix or its reason: `no_reports` when `m == 0`,
/// `too_few_ranges` when `m < 3`, `degenerate` (the collinear rule, or the
/// Gauss-Newton) or `not_converged`; the fix has
/// `R = (rtt_sigma_m*rtt_sigma_m) * inv(N)` and `chi2`.
pub fn rtt_fix(cfg: &Config, reports: &[Ranging]) -> Result<Result<Fix, Reason>, Error> {
    cfg.validate()?;
    let reports = &*nodes_ascending(reports);
    check_reports(reports)?;
    Ok(rtt_fix_checked(cfg, reports))
}

fn rtt_fix_checked(cfg: &Config, reports: &[Ranging]) -> Result<Fix, Reason> {
    let m = reports.len();
    if m == 0 {
        return Err(Reason::NoReports);
    }
    if m < 3 {
        return Err(Reason::TooFewRanges);
    }
    // The collinear rule is the start's own test (§7, §7.1).
    let (x0, y0) = rtt_start(cfg, reports)?;
    let (x, y, s, it) = gauss_newton(x0, y0, |x, y| rtt_sums(cfg, reports, x, y))?;
    Ok(finish(
        x,
        y,
        &s,
        it,
        (x0, y0),
        cfg.rtt_sigma_m,
        Some(s.chi2),
    ))
}

/// The RSSI fix of §7.2 from one epoch's reports `(n, rssi_n)` (`rssi_n`
/// as a float), taken nodes ascending as [`rtt_fix`] takes them. Refusals
/// as [`rtt_fix`], plus `rssi_p0_dbm` when neither it nor Room Sim's values
/// are set ([`Config::rssi_p0`]). Reasons: `no_reports`, `too_few_ranges`,
/// then the collinear rule and the start's weight guard (`degenerate`),
/// then the Gauss-Newton's; covariance
/// `R = (rssi_shadow_sigma_db*rssi_shadow_sigma_db) * inv(N)`.
pub fn rssi_fix(cfg: &Config, reports: &[Ranging]) -> Result<Result<Fix, Reason>, Error> {
    cfg.validate()?;
    let p0 = cfg.rssi_p0()?;
    let reports = &*nodes_ascending(reports);
    check_reports(reports)?;
    Ok(rssi_fix_checked(cfg, p0, reports))
}

fn rssi_fix_checked(cfg: &Config, p0: f64, reports: &[Ranging]) -> Result<Fix, Reason> {
    let m = reports.len();
    if m == 0 {
        return Err(Reason::NoReports);
    }
    if m < 3 {
        return Err(Reason::TooFewRanges);
    }
    if anchors_collinear(reports) {
        return Err(Reason::Degenerate);
    }
    let (x0, y0) = rssi_start(reports)?;
    let (x, y, s, it) = gauss_newton(x0, y0, |x, y| rssi_sums(cfg, p0, reports, x, y))?;
    Ok(finish(
        x,
        y,
        &s,
        it,
        (x0, y0),
        cfg.rssi_shadow_sigma_db,
        None,
    ))
}

/// [`rtt_fix`] or [`rssi_fix`].
pub fn fix(
    method: Method,
    cfg: &Config,
    reports: &[Ranging],
) -> Result<Result<Fix, Reason>, Error> {
    match method {
        Method::Rtt => rtt_fix(cfg, reports),
        Method::Rssi => rssi_fix(cfg, reports),
    }
}

// ===================================================================
// Kalman filter (§7.3) and the published estimate (§7.4)
// ===================================================================

/// A running filter's state.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct FilterState {
    /// `s = [x, y, vx, vy]`.
    pub s: [f64; 4],
    /// `P`, 4x4.
    pub p: Mat4,
    /// Time of the start or last predict.
    pub t_f: f64,
    /// Time of the start or last accepted update.
    pub t_acc: f64,
}

/// One method's constant-velocity Kalman filter (§7.3), with continuous
/// white-noise acceleration `q = phone_accel_psd`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PhoneFilter {
    state: Option<FilterState>,
}

impl PhoneFilter {
    /// No filter yet.
    pub fn new() -> Self {
        Self { state: None }
    }

    /// The running filter, if one exists.
    pub fn state(&self) -> Option<&FilterState> {
        self.state.as_ref()
    }

    /// One fired epoch at time `t` with this method's fix (or the fix's
    /// reason), in §7.3's order:
    /// 1. a filter with `not (t - t_acc <= phone_max_coast_s)` is discarded;
    /// 2. no fix: abstain with the fix's reason, nothing predicted;
    /// 3. no filter: the fix starts one and is published ungated;
    /// 4. otherwise predict to `t`, gate (`gated`: the predicted state is
    ///    kept), update (Joseph form) and publish.
    ///
    /// Publishing (§7.4): `r = phone_radius_scale * radius95(P00 + B00,
    /// P01 + B01, P11 + B11)`, `B = (beta*beta) * inv(N)` of this epoch's
    /// fix; `implausible_radius` if `r` fails §8 (the filter is still
    /// started or updated). `beta` is `rtt_persist_sigma_m` for RTT and
    /// `rssi_persist_sigma_db` for RSSI.
    pub fn epoch(&mut self, t: f64, fix: Result<&Fix, Reason>, beta: f64, cfg: &Config) -> Outcome {
        // 1. Coast limit.
        if let Some(st) = &self.state {
            if !(t - st.t_acc <= cfg.phone_max_coast_s) {
                self.state = None;
            }
        }
        // 2. No fix: abstain; nothing is predicted.
        let fx = match fix {
            Err(reason) => return Outcome::Abstain(reason),
            Ok(f) => f,
        };
        let z = [fx.x, fx.y];
        let r = fx.r;
        let st = match &mut self.state {
            // 3. Start, published ungated.
            None => {
                let v = cfg.phone_init_speed_std * cfg.phone_init_speed_std;
                let p = [
                    [r[0][0], r[0][1], 0.0, 0.0],
                    [r[1][0], r[1][1], 0.0, 0.0],
                    [0.0, 0.0, v, 0.0],
                    [0.0, 0.0, 0.0, v],
                ];
                self.state = Some(FilterState {
                    s: [z[0], z[1], 0.0, 0.0],
                    p,
                    t_f: t,
                    t_acc: t,
                });
                self.state.as_ref().expect("just set")
            }
            // 4. Predict, gate, update.
            Some(st) => {
                let q = cfg.phone_accel_psd;
                let dt = t - st.t_f;
                st.t_f = t;
                let mut f: Mat4 = identity();
                f[0][2] = dt;
                f[1][3] = dt;
                let mut qm: Mat4 = [[0.0; 4]; 4];
                for i in 0..2 {
                    qm[i][i] = q * ((dt * dt) * dt) / 3.0;
                    qm[i][i + 2] = q * (dt * dt) / 2.0;
                    qm[i + 2][i] = q * (dt * dt) / 2.0;
                    qm[i + 2][i + 2] = q * dt;
                }
                st.s = mat_vec(&f, &st.s);
                st.p = sym(&mat_add(&mat_mul(&mat_mul(&f, &st.p), &transpose(&f)), &qm));
                // Gate.
                let nu = [z[0] - st.s[0], z[1] - st.s[1]];
                let sm: Mat2 = [
                    [st.p[0][0] + r[0][0], st.p[0][1] + r[0][1]],
                    [st.p[1][0] + r[1][0], st.p[1][1] + r[1][1]],
                ];
                let det = det2(&sm);
                if !(det > 0.0) {
                    return Outcome::Abstain(Reason::Gated);
                }
                let si = inv2(&sm, det);
                let d2 = ((nu[0] * si[0][0] + nu[1] * si[1][0]) * nu[0])
                    + ((nu[0] * si[0][1] + nu[1] * si[1][1]) * nu[1]);
                if !(d2 <= chi2_2_99()) {
                    return Outcome::Abstain(Reason::Gated);
                }
                // Update (Joseph form).
                let p42: [[f64; 2]; 4] = [
                    [st.p[0][0], st.p[0][1]],
                    [st.p[1][0], st.p[1][1]],
                    [st.p[2][0], st.p[2][1]],
                    [st.p[3][0], st.p[3][1]],
                ];
                let k = mat_mul(&p42, &si);
                let kv = mat_vec(&k, &nu);
                for i in 0..4 {
                    st.s[i] += kv[i];
                }
                let h: [[f64; 4]; 2] = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]];
                let a = mat_sub(&identity::<4>(), &mat_mul(&k, &h));
                let apa = mat_mul(&mat_mul(&a, &st.p), &transpose(&a));
                let krk = mat_mul(&mat_mul(&k, &r), &transpose(&k));
                st.p = sym(&mat_add(&apa, &krk));
                st.t_acc = t;
                &*st
            }
        };
        let b = mat_scale(beta * beta, &fx.inv_n);
        publish(
            st.s[0],
            st.s[1],
            st.p[0][0] + b[0][0],
            st.p[0][1] + b[0][1],
            st.p[1][1] + b[1][1],
            cfg.phone_radius_scale,
            cfg.min_plausible_radius_m,
            None,
        )
    }
}

// ===================================================================
// The whole phone (§6 + §7 per fired epoch, §11 records)
// ===================================================================

/// One node's entry in an epoch record's `reports` (§11): `{node, rtt_m
/// or null, rssi_dbm or null, blocked}`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ReportRecord {
    pub node: u8,
    /// The RTT range, m, or `None` (`null`) when there was no report.
    pub rtt_m: Option<f64>,
    /// The RSSI, integer dBm in -100..0, or `None` (`null`).
    pub rssi_dbm: Option<i32>,
    /// Whether the phone-node segment is blocked (§6.2).
    pub blocked: bool,
}

impl From<&NodeReport> for ReportRecord {
    fn from(r: &NodeReport) -> Self {
        Self {
            node: r.node,
            rtt_m: r.rtt_m,
            rssi_dbm: r.rssi_dbm,
            blocked: r.blocked,
        }
    }
}

/// One fired phone epoch's record (§11): `{label, t, rtt, rssi, reports,
/// truth}` and nothing else. A method that abstains carries no position:
/// its rejected fix is reachable only through [`PhoneRun::last_fixes`].
#[derive(Debug, Clone, PartialEq)]
pub struct EpochRecord {
    /// Always `"SIM"`.
    pub label: &'static str,
    pub t: f64,
    pub rtt: Outcome,
    pub rssi: Outcome,
    /// One per node, ids ascending.
    pub reports: Vec<ReportRecord>,
    /// The sample's phone `(x, y, z)` (SIM truth).
    pub truth: [f64; 3],
}

/// Each method's fix (or its reason) at one fired epoch: `(rtt, rssi)`.
pub type EpochFixes = (Result<Fix, Reason>, Result<Fix, Reason>);

/// The simulated phone: the radio (§6), both fixes (§7.1, §7.2) and both
/// filters (§7.3) at every fired epoch.
#[derive(Debug, Clone)]
pub struct PhoneRun {
    cfg: Config,
    radio: Radio,
    rtt: PhoneFilter,
    rssi: PhoneFilter,
    last_epoch: Option<RadioEpoch>,
    last_fixes: Option<EpochFixes>,
}

impl PhoneRun {
    /// See [`Radio::new`]: refuses any label but `"SIM"`, validates `cfg`
    /// and needs its Room Sim values set.
    pub fn new(
        cfg: &Config,
        nodes: &[PhoneNode],
        blockers: &Blockers,
        label: &str,
    ) -> Result<Self, Error> {
        Ok(Self {
            cfg: cfg.clone(),
            radio: Radio::new(cfg, nodes, blockers, label)?,
            rtt: PhoneFilter::new(),
            rssi: PhoneFilter::new(),
            last_epoch: None,
            last_fixes: None,
        })
    }

    /// The radio.
    pub fn radio(&self) -> &Radio {
        &self.radio
    }

    /// The filter of `method`.
    pub fn filter(&self, method: Method) -> &PhoneFilter {
        match method {
            Method::Rtt => &self.rtt,
            Method::Rssi => &self.rssi,
        }
    }

    /// The radio's outcome at the last fired epoch, with its per-node
    /// diagnostics (`rssi_v`, `shadow_db`; SIM, not part of any record).
    pub fn last_epoch(&self) -> Option<&RadioEpoch> {
        self.last_epoch.as_ref()
    }

    /// Each method's fix (or reason) at the last fired epoch, published or
    /// not (diagnostics, not part of any record).
    pub fn last_fixes(&self) -> Option<EpochFixes> {
        self.last_fixes
    }

    /// Feed truth sample `j`; a record when an epoch fires (§6.1). Errors:
    /// [`Radio::step`]'s, and a fix's refusal (`reports`), which the
    /// radio's own reports never trigger unless a range overflows.
    pub fn step(&mut self, sample: &TruthSample<'_>) -> Result<Option<EpochRecord>, Error> {
        let Some(ep) = self.radio.step(sample)? else {
            return Ok(None);
        };
        let nodes = self.radio.nodes();
        let mut rtt_reports = Vec::new();
        let mut rssi_reports = Vec::new();
        for (node, rep) in nodes.iter().zip(&ep.reports) {
            if let Some(r) = rep.rtt_m {
                rtt_reports.push(Ranging {
                    node: *node,
                    value: r,
                });
            }
            if let Some(v) = rep.rssi_dbm {
                rssi_reports.push(Ranging {
                    node: *node,
                    value: f64::from(v),
                });
            }
        }
        let cfg = &self.cfg;
        let rtt_fix = rtt_fix(cfg, &rtt_reports)?;
        let rssi_fix = rssi_fix(cfg, &rssi_reports)?;
        let rtt = self.rtt.epoch(
            ep.t,
            rtt_fix.as_ref().map_err(|r| *r),
            cfg.rtt_persist_sigma_m,
            cfg,
        );
        let rssi = self.rssi.epoch(
            ep.t,
            rssi_fix.as_ref().map_err(|r| *r),
            cfg.rssi_persist_sigma_db,
            cfg,
        );
        let record = EpochRecord {
            label: crate::LABEL,
            t: ep.t,
            rtt,
            rssi,
            reports: ep.reports.iter().map(ReportRecord::from).collect(),
            truth: ep.truth,
        };
        self.last_epoch = Some(ep);
        self.last_fixes = Some((rtt_fix, rssi_fix));
        Ok(Some(record))
    }
}

/// Run the phone over a whole time-ordered truth sequence: one record per
/// fired epoch. Refuses any label but `"SIM"`.
pub fn simulate(
    cfg: &Config,
    nodes: &[PhoneNode],
    blockers: &Blockers,
    label: &str,
    samples: &[TruthSample<'_>],
) -> Result<Vec<EpochRecord>, Error> {
    let mut run = PhoneRun::new(cfg, nodes, blockers, label)?;
    let mut out = Vec::new();
    for s in samples {
        if let Some(rec) = run.step(s)? {
            out.push(rec);
        }
    }
    Ok(out)
}
