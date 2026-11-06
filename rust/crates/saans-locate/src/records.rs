//! The records file of a Room Sim run (spec/livedemo.md §6, §8.3): writer,
//! reader and the JSON encodings shared with the gateway's frames. SIM.
//!
//! The file is JSON Lines (§6.1): a manifest, then one line per tick and
//! one per fired phone epoch, ordered by `t`, the epoch first at equal `t`
//! (`run_roomsim`'s order). An outcome, a link item, a phone report and
//! the two truths are encoded as §4 and spec/locate.md §11 say, by the
//! functions the gateway's frame builder also calls ([`outcome_json`],
//! [`link_json`], [`report_json`], [`tick_truth_json`],
//! [`epoch_truth_json`]), so a frame and a file cannot disagree.
//!
//! - **Exact.** Floats are written in their shortest round-trip form and
//!   read with serde_json's `float_roundtrip`, so [`read_records`] of
//!   [`write_records`] gives back every `f64` bit for bit. JSON has no
//!   non-finite number: a link's `y` that is not finite is written `null`
//!   and read back as `f64::NAN` (§4), and [`write_records`] refuses any
//!   other non-finite value rather than write it changed.
//! - **No ids.** People are `i`, their index. Nothing here reads a
//!   `Person.id` or a `Phone.id` (ADR-008 decision 3). A refusal names the
//!   line and the key, never a value or an unknown key from the file.
//! - **Format only.** [`read_records`] makes every §6.2 refusal that needs
//!   no embedded scene. Matching [`Manifest::scene_sha256`] to an embedded
//!   scene, and allowing one file per scene, are the gateway's.
//! - **SIM only.** Records are SIM, of simulated people (spec/locate.md
//!   §11). [`write_records`] refuses a record with any other label, so
//!   nothing here writes a position from REPLAY or LIVE input (ADR-008
//!   decision 11).

// NaN fails closed: deciding comparisons are written `!(a < b)` on purpose.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::io::{self, BufRead, Write};

use saans_roomsim::{Config as RoomSimConfig, Posture};
use saans_scene::Provenance;
use serde::de::{self, Deserialize, Deserializer, MapAccess, SeqAccess, Visitor};
use serde::Serialize;
use serde_json::{Map, Number, Value};

use crate::phone::{self, Config, EpochRecord, ReportRecord};
use crate::pipeline::{
    check_roomsim_config, resolve_config, PersonTruthRecord, PipelineError, TickRecord,
};
use crate::radius::{plausible, Estimate, Outcome, Reason};
use crate::score::FalseAlarms;
use crate::stream::LinkValue;
use crate::variant::Variant;
use crate::LABEL;

/// The manifest's `spec` (§6.1).
pub const SPEC: &str = "livedemo v1";

/// The largest records file the gateway reads: 64 MiB (§6.1, §6.2).
pub const MAX_BYTES: u64 = 64 << 20;

/// The only `roomsim_config.slot_s` of Phase 5 (§3.2): the session clock
/// `T = float(2*q) * slot_s` is exact for it, and the reader refuses any
/// other (§6.2).
pub const SLOT_S: f64 = 0.0125;

/// Every [`Reason`], for decoding its spelling.
const REASONS: [Reason; 8] = [
    Reason::NoDisturbance,
    Reason::NotSignificant,
    Reason::NoReports,
    Reason::TooFewRanges,
    Reason::Degenerate,
    Reason::NotConverged,
    Reason::Gated,
    Reason::ImplausibleRadius,
];

/// The reasons each estimator may give (spec/locate.md §11).
const WLM_REASONS: &[Reason] = &[Reason::NoDisturbance, Reason::ImplausibleRadius];
const RTI_REASONS: &[Reason] = &[Reason::NotSignificant, Reason::ImplausibleRadius];
const PHONE_REASONS: &[Reason] = &[
    Reason::NoReports,
    Reason::TooFewRanges,
    Reason::Degenerate,
    Reason::NotConverged,
    Reason::Gated,
    Reason::ImplausibleRadius,
];

// ===================================================================
// The file
// ===================================================================

/// The first line of a records file (§6.1). SIM.
#[derive(Debug, Clone, PartialEq)]
pub struct Manifest {
    /// The scene's `name` (SCENE, ADR-008 decision 4).
    pub scene_name: String,
    /// The lower-case hex sha256 of the scene file's bytes. The gateway
    /// matches it to an embedded scene (§6.2); the reader checks only its
    /// form, 64 lower-case hex digits.
    pub scene_sha256: String,
    /// The scene's `provenance` (ADR-008 decision 4).
    pub provenance: Provenance,
    pub variant: Variant,
    /// Room Sim's `seed`; equal to `roomsim_config.seed`.
    pub seed: u64,
    /// The run's length, s; equal to `roomsim_config.duration_s`, with
    /// `40 * duration_s` an integer (§3.2, §6.2).
    pub duration_s: f64,
    /// The Room Sim config the run was made with, its variant applied
    /// ([`Variant::apply`]); `slot_s` is [`SLOT_S`].
    pub roomsim_config: RoomSimConfig,
    /// Locate's config (spec/locate.md §9), every field.
    pub locate_config: Config,
    /// The localize canonical link order `(tx, rx)`
    /// ([`crate::pipeline::RoomSimRun::links`]). Every tick's links are in
    /// this order.
    pub links: Vec<(i64, i64)>,
    /// Whether the scene has a phone. Without one there are no epochs.
    pub has_phone: bool,
    /// The empty run's `unoccupied` false alarms `(wlm, rti)` (§5.3);
    /// `None` exactly for variant `empty`.
    pub empty_run: Option<(FalseAlarms, FalseAlarms)>,
}

/// A whole records file (§6.1). Every record is SIM.
#[derive(Debug, Clone, PartialEq)]
pub struct Records {
    pub manifest: Manifest,
    /// One per tick, in time order.
    pub ticks: Vec<TickRecord>,
    /// One list per tick (`links[k]` is `ticks[k]`'s), in
    /// [`Manifest::links`]' order (§4 `links.items`).
    pub links: Vec<Vec<LinkValue>>,
    /// One per fired phone epoch, in time order.
    pub epochs: Vec<EpochRecord>,
}

/// Why a records file was refused (§6.2): the 1-based line, and a reason
/// that names the key, never a value from the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecordsError {
    pub line: usize,
    pub reason: String,
}

impl std::fmt::Display for RecordsError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "line {}: {}", self.line, self.reason)
    }
}

impl std::error::Error for RecordsError {}

fn refused(line: usize, reason: impl Into<String>) -> RecordsError {
    RecordsError {
        line,
        reason: reason.into(),
    }
}

// ===================================================================
// Encodings (§4, spec/locate.md §11)
// ===================================================================
//
// Each encoding is a struct whose fields are in the spec's key order: the
// file is written from them directly, and the gateway takes them as
// `Value`s. serde_json writes a non-finite `f64` as `null`.

/// spec/locate.md §11 **Estimate**.
#[derive(Serialize)]
struct EstimateOut {
    x: f64,
    y: f64,
    radius_m: f64,
    cov_xx: f64,
    cov_xy: f64,
    cov_yy: f64,
    confidence: Option<f64>,
    radius_calibrated: bool,
}

/// `{"estimate": {…}}` or `{"abstain": {"reason"}}`.
#[derive(Serialize)]
#[serde(rename_all = "lowercase")]
enum OutcomeOut {
    Estimate(EstimateOut),
    Abstain { reason: &'static str },
}

impl From<&Outcome> for OutcomeOut {
    fn from(o: &Outcome) -> Self {
        match *o {
            Outcome::Estimate(e) => {
                // Destructured, so a new field cannot be left out.
                let Estimate {
                    x,
                    y,
                    radius_m,
                    cov_xx,
                    cov_xy,
                    cov_yy,
                    confidence,
                    radius_calibrated,
                } = e;
                OutcomeOut::Estimate(EstimateOut {
                    x,
                    y,
                    radius_m,
                    cov_xx,
                    cov_xy,
                    cov_yy,
                    confidence,
                    radius_calibrated,
                })
            }
            Outcome::Abstain(r) => OutcomeOut::Abstain { reason: r.as_str() },
        }
    }
}

/// §4 `links.items[l]`.
#[derive(Serialize)]
struct LinkOut {
    tx: i64,
    rx: i64,
    y: Option<f64>,
    gate: Option<f64>,
    disturbed: bool,
}

impl LinkOut {
    fn new(tx: i64, rx: i64, v: &LinkValue) -> Self {
        let LinkValue { y, gate, disturbed } = *v;
        LinkOut {
            tx,
            rx,
            y: y.is_finite().then_some(y),
            gate,
            disturbed,
        }
    }
}

/// §4 `phone.reports[k]`: no `blocked`, which is truth.
#[derive(Serialize)]
struct ReportOut {
    node: u8,
    rtt_m: Option<f64>,
    rssi_dbm: Option<i32>,
}

impl From<&ReportRecord> for ReportOut {
    fn from(r: &ReportRecord) -> Self {
        let ReportRecord {
            node,
            rtt_m,
            rssi_dbm,
            blocked: _,
        } = *r;
        ReportOut {
            node,
            rtt_m,
            rssi_dbm,
        }
    }
}

/// §4 `truth.persons.people[i]`.
#[derive(Serialize)]
struct PersonTruthOut {
    i: usize,
    x: f64,
    y: f64,
    walking: bool,
    posture: &'static str,
}

fn tick_truth_out(truth: &[PersonTruthRecord]) -> Vec<PersonTruthOut> {
    truth
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let PersonTruthRecord {
                x,
                y,
                walking,
                posture,
            } = *p;
            PersonTruthOut {
                i,
                x,
                y,
                walking,
                posture: posture.name(),
            }
        })
        .collect()
}

/// §4 `truth.phone` without its time and scores: `{x, y, z, blocked}`.
#[derive(Serialize)]
struct EpochTruthOut {
    x: f64,
    y: f64,
    z: f64,
    blocked: Vec<u8>,
}

fn epoch_truth_out(truth: [f64; 3], reports: &[ReportRecord]) -> EpochTruthOut {
    let [x, y, z] = truth;
    EpochTruthOut {
        x,
        y,
        z,
        blocked: reports
            .iter()
            .filter(|r| r.blocked)
            .map(|r| r.node)
            .collect(),
    }
}

/// Every field of [`Config`], in its declaration order (spec/locate.md §9).
#[derive(Serialize)]
struct LocateConfigOut<'a> {
    lambda_m: &'a f64,
    pitch_m: &'a f64,
    alpha: &'a f64,
    sigma: &'a f64,
    delta_c: &'a f64,
    z_threshold: &'a f64,
    window_s: &'a f64,
    min_windows: &'a usize,
    link_signal: &'a str,
    calib_s: &'a f64,
    tick_hz: &'a f64,
    wlm_gate_z: &'a f64,
    wlm_floor: &'a Option<f64>,
    wlm_radius_scale: &'a f64,
    rti_radius_scale: &'a f64,
    phone_radius_scale: &'a f64,
    min_plausible_radius_m: &'a f64,
    phone_rate_hz: &'a f64,
    phone_seed: &'a Option<u64>,
    rtt_sigma_m: &'a f64,
    rtt_nlos_sigma_scale: &'a f64,
    rtt_nlos_bias_m: &'a f64,
    rtt_outlier_p: &'a f64,
    rtt_outlier_lo_m: &'a f64,
    rtt_outlier_hi_m: &'a f64,
    rtt_offset_sigma_m: &'a f64,
    rtt_persist_sigma_m: &'a f64,
    rtt_fail_p: &'a f64,
    rssi_fail_p: &'a f64,
    rssi_breakpoint_m: &'a f64,
    rssi_far_exponent: &'a f64,
    rssi_shadow_sigma_db: &'a f64,
    rssi_persist_sigma_db: &'a f64,
    rssi_shadow_corr_m: &'a f64,
    rssi_nlos_loss_db: &'a f64,
    rssi_device_sigma_db: &'a f64,
    rssi_jitter_db: &'a f64,
    rssi_est_exponent: &'a f64,
    rssi_p0_dbm: &'a Option<f64>,
    phone_assumed_z_m: &'a f64,
    huber_k: &'a f64,
    phone_accel_psd: &'a f64,
    phone_init_speed_std: &'a f64,
    phone_max_coast_s: &'a f64,
    tx_power_dbm: &'a Option<f64>,
    carrier_hz: &'a Option<f64>,
}

impl<'a> From<&'a Config> for LocateConfigOut<'a> {
    fn from(cfg: &'a Config) -> Self {
        // Destructured, so a new §9 parameter cannot be left out.
        let Config {
            lambda_m,
            pitch_m,
            alpha,
            sigma,
            delta_c,
            z_threshold,
            window_s,
            min_windows,
            link_signal,
            calib_s,
            tick_hz,
            wlm_gate_z,
            wlm_floor,
            wlm_radius_scale,
            rti_radius_scale,
            phone_radius_scale,
            min_plausible_radius_m,
            phone_rate_hz,
            phone_seed,
            rtt_sigma_m,
            rtt_nlos_sigma_scale,
            rtt_nlos_bias_m,
            rtt_outlier_p,
            rtt_outlier_lo_m,
            rtt_outlier_hi_m,
            rtt_offset_sigma_m,
            rtt_persist_sigma_m,
            rtt_fail_p,
            rssi_fail_p,
            rssi_breakpoint_m,
            rssi_far_exponent,
            rssi_shadow_sigma_db,
            rssi_persist_sigma_db,
            rssi_shadow_corr_m,
            rssi_nlos_loss_db,
            rssi_device_sigma_db,
            rssi_jitter_db,
            rssi_est_exponent,
            rssi_p0_dbm,
            phone_assumed_z_m,
            huber_k,
            phone_accel_psd,
            phone_init_speed_std,
            phone_max_coast_s,
            tx_power_dbm,
            carrier_hz,
        } = cfg;
        LocateConfigOut {
            lambda_m,
            pitch_m,
            alpha,
            sigma,
            delta_c,
            z_threshold,
            window_s,
            min_windows,
            link_signal,
            calib_s,
            tick_hz,
            wlm_gate_z,
            wlm_floor,
            wlm_radius_scale,
            rti_radius_scale,
            phone_radius_scale,
            min_plausible_radius_m,
            phone_rate_hz,
            phone_seed,
            rtt_sigma_m,
            rtt_nlos_sigma_scale,
            rtt_nlos_bias_m,
            rtt_outlier_p,
            rtt_outlier_lo_m,
            rtt_outlier_hi_m,
            rtt_offset_sigma_m,
            rtt_persist_sigma_m,
            rtt_fail_p,
            rssi_fail_p,
            rssi_breakpoint_m,
            rssi_far_exponent,
            rssi_shadow_sigma_db,
            rssi_persist_sigma_db,
            rssi_shadow_corr_m,
            rssi_nlos_loss_db,
            rssi_device_sigma_db,
            rssi_jitter_db,
            rssi_est_exponent,
            rssi_p0_dbm,
            phone_assumed_z_m,
            huber_k,
            phone_accel_psd,
            phone_init_speed_std,
            phone_max_coast_s,
            tx_power_dbm,
            carrier_hz,
        }
    }
}

/// `x` as a `Value`. Every encoding here (Room Sim's config included) is
/// structs, numbers, strings, sequences and maps keyed by strings or
/// integers, which serde_json always encodes.
fn value(x: impl Serialize) -> Value {
    serde_json::to_value(x).expect("a records encoding has no non-string map key")
}

/// An outcome (§4 `Outcome`, spec/locate.md §11): `{"estimate": {"x",
/// "y", "radius_m", "cov_xx", "cov_xy", "cov_yy", "confidence",
/// "radius_calibrated"}}`, `confidence` `null` when `None`; or
/// `{"abstain": {"reason"}}` with [`Reason::as_str`]. There is no encoding
/// of a position without its radius.
pub fn outcome_json(o: &Outcome) -> Value {
    value(OutcomeOut::from(o))
}

/// One link item of a tick (§4 `links.items`): `{"tx", "rx", "y", "gate",
/// "disturbed"}`, `y` `null` when it is not finite and `gate` `null` when
/// `None`.
pub fn link_json(tx: i64, rx: i64, v: &LinkValue) -> Value {
    value(LinkOut::new(tx, rx, v))
}

/// One phone report (§4 `phone.reports`): `{"node", "rtt_m", "rssi_dbm"}`,
/// each value `null` when the node gave no report. `blocked` is not here:
/// it is computed from the true positions, so it is truth
/// ([`epoch_truth_json`]).
pub fn report_json(r: &ReportRecord) -> Value {
    value(ReportOut::from(r))
}

/// A tick's truth (§4 `truth.persons.people`, §6.1): `[{"i", "x", "y",
/// "walking", "posture"}]`, people by index `i`, never by a scene id;
/// `posture` is Room Sim's [`Posture::name`].
pub fn tick_truth_json(truth: &[PersonTruthRecord]) -> Value {
    value(tick_truth_out(truth))
}

/// An epoch's truth (§4 `truth.phone`, §6.1): `{"x", "y", "z", "blocked"}`,
/// `blocked` the ids of the nodes whose report is blocked, in `reports`'
/// order.
pub fn epoch_truth_json(truth: [f64; 3], reports: &[ReportRecord]) -> Value {
    value(epoch_truth_out(truth, reports))
}

/// Every field of Locate's config (spec/locate.md §9) as a JSON object,
/// unset options `null`. The manifest's `locate_config` (§6.1).
pub fn locate_config_json(cfg: &Config) -> Value {
    value(LocateConfigOut::from(cfg))
}

/// The spelling of a scene's provenance, as the scene format writes it.
pub fn provenance_name(p: Provenance) -> &'static str {
    match p {
        Provenance::Surveyed => "surveyed",
        Provenance::RoomplanImport => "roomplan_import",
        Provenance::HandWritten => "hand_written",
        Provenance::Generated => "generated",
    }
}

// ===================================================================
// Decoding
// ===================================================================
//
// A reason is "{path}: {problem}", where `path` is built from the known
// keys and indices only (`wlm.estimate.radius_m`, `links[3]`): never a
// value, and never an unknown key, from the file.

type Obj = Map<String, Value>;

fn join(path: &str, key: &str) -> String {
    if path.is_empty() {
        key.to_string()
    } else {
        format!("{path}.{key}")
    }
}

fn at(path: &str, problem: &str) -> String {
    if path.is_empty() {
        problem.to_string()
    } else {
        format!("{path}: {problem}")
    }
}

/// `v` as an object with no key outside `known` (§6.2). The unknown key is
/// not named: it is the file's text.
fn object<'a>(v: &'a Value, path: &str, known: &[&str]) -> Result<&'a Obj, String> {
    let m = v.as_object().ok_or_else(|| at(path, "not an object"))?;
    if m.keys().any(|k| !known.contains(&k.as_str())) {
        return Err(at(path, "unknown key (spec/livedemo.md §6.2)"));
    }
    Ok(m)
}

fn get<'a>(m: &'a Obj, path: &str, key: &str) -> Result<&'a Value, String> {
    m.get(key).ok_or_else(|| at(&join(path, key), "missing"))
}

fn num(m: &Obj, path: &str, key: &str) -> Result<f64, String> {
    get(m, path, key)?
        .as_f64()
        .ok_or_else(|| at(&join(path, key), "not a number"))
}

fn num_or_null(m: &Obj, path: &str, key: &str) -> Result<Option<f64>, String> {
    match get(m, path, key)? {
        Value::Null => Ok(None),
        v => v
            .as_f64()
            .map(Some)
            .ok_or_else(|| at(&join(path, key), "not a number or null")),
    }
}

fn uint(m: &Obj, path: &str, key: &str) -> Result<u64, String> {
    get(m, path, key)?
        .as_u64()
        .ok_or_else(|| at(&join(path, key), "not an unsigned integer"))
}

fn size(m: &Obj, path: &str, key: &str) -> Result<usize, String> {
    usize::try_from(uint(m, path, key)?).map_err(|_| at(&join(path, key), "too large"))
}

fn node_id(v: &Value, path: &str) -> Result<u8, String> {
    v.as_u64()
        .and_then(|n| u8::try_from(n).ok())
        .ok_or_else(|| at(path, "not a node id (an integer in 0..=255)"))
}

fn boolean(m: &Obj, path: &str, key: &str) -> Result<bool, String> {
    get(m, path, key)?
        .as_bool()
        .ok_or_else(|| at(&join(path, key), "not true or false"))
}

fn string<'a>(m: &'a Obj, path: &str, key: &str) -> Result<&'a str, String> {
    get(m, path, key)?
        .as_str()
        .ok_or_else(|| at(&join(path, key), "not a string"))
}

fn array<'a>(m: &'a Obj, path: &str, key: &str) -> Result<&'a Vec<Value>, String> {
    get(m, path, key)?
        .as_array()
        .ok_or_else(|| at(&join(path, key), "not an array"))
}

fn reason_from_str(s: &str) -> Option<Reason> {
    REASONS.into_iter().find(|r| r.as_str() == s)
}

fn posture_from_name(s: &str) -> Option<Posture> {
    [Posture::Stand, Posture::Sit, Posture::Lie]
        .into_iter()
        .find(|p| p.name() == s)
}

fn provenance_from_name(s: &str) -> Option<Provenance> {
    [
        Provenance::Surveyed,
        Provenance::RoomplanImport,
        Provenance::HandWritten,
        Provenance::Generated,
    ]
    .into_iter()
    .find(|p| provenance_name(*p) == s)
}

fn outcome_at(v: &Value, path: &str) -> Result<Outcome, String> {
    let m = object(v, path, &["estimate", "abstain"])?;
    match (m.get("estimate"), m.get("abstain")) {
        (Some(e), None) => estimate_at(e, &join(path, "estimate")).map(Outcome::Estimate),
        (None, Some(a)) => {
            let p = join(path, "abstain");
            let a = object(a, &p, &["reason"])?;
            reason_from_str(string(a, &p, "reason")?)
                .map(Outcome::Abstain)
                .ok_or_else(|| at(&join(&p, "reason"), "not a reason of spec/locate.md §11"))
        }
        _ => Err(at(path, "not exactly one of estimate and abstain")),
    }
}

fn estimate_at(v: &Value, path: &str) -> Result<Estimate, String> {
    let m = object(
        v,
        path,
        &[
            "x",
            "y",
            "radius_m",
            "cov_xx",
            "cov_xy",
            "cov_yy",
            "confidence",
            "radius_calibrated",
        ],
    )?;
    Ok(Estimate {
        x: num(m, path, "x")?,
        y: num(m, path, "y")?,
        radius_m: num(m, path, "radius_m")?,
        cov_xx: num(m, path, "cov_xx")?,
        cov_xy: num(m, path, "cov_xy")?,
        cov_yy: num(m, path, "cov_yy")?,
        confidence: num_or_null(m, path, "confidence")?,
        radius_calibrated: boolean(m, path, "radius_calibrated")?,
    })
}

fn link_at(v: &Value, path: &str) -> Result<(i64, i64, LinkValue), String> {
    let m = object(v, path, &["tx", "rx", "y", "gate", "disturbed"])?;
    let int = |key: &str| {
        get(m, path, key)?
            .as_i64()
            .ok_or_else(|| at(&join(path, key), "not an integer"))
    };
    let tx = int("tx")?;
    let rx = int("rx")?;
    let value = LinkValue {
        y: num_or_null(m, path, "y")?.unwrap_or(f64::NAN),
        gate: num_or_null(m, path, "gate")?,
        disturbed: boolean(m, path, "disturbed")?,
    };
    Ok((tx, rx, value))
}

fn report_at(v: &Value, path: &str) -> Result<ReportRecord, String> {
    let m = object(v, path, &["node", "rtt_m", "rssi_dbm"])?;
    let rssi_dbm = match get(m, path, "rssi_dbm")? {
        Value::Null => None,
        v => Some(
            v.as_i64()
                .and_then(|n| i32::try_from(n).ok())
                .ok_or_else(|| at(&join(path, "rssi_dbm"), "not an integer dBm or null"))?,
        ),
    };
    Ok(ReportRecord {
        node: node_id(get(m, path, "node")?, &join(path, "node"))?,
        rtt_m: num_or_null(m, path, "rtt_m")?,
        rssi_dbm,
        blocked: false,
    })
}

fn tick_truth_at(v: &Value, path: &str) -> Result<Vec<PersonTruthRecord>, String> {
    let items = v.as_array().ok_or_else(|| at(path, "not an array"))?;
    let mut truth = Vec::with_capacity(items.len());
    for (k, item) in items.iter().enumerate() {
        let p = format!("{path}[{k}]");
        let m = object(item, &p, &["i", "x", "y", "walking", "posture"])?;
        if uint(m, &p, "i")? != k as u64 {
            return Err(at(
                &join(&p, "i"),
                "not its index; people are listed by index",
            ));
        }
        truth.push(PersonTruthRecord {
            x: num(m, &p, "x")?,
            y: num(m, &p, "y")?,
            walking: boolean(m, &p, "walking")?,
            posture: posture_from_name(string(m, &p, "posture")?)
                .ok_or_else(|| at(&join(&p, "posture"), "not stand, sit or lie"))?,
        });
    }
    Ok(truth)
}

fn epoch_truth_at(v: &Value, path: &str, reports: &mut [ReportRecord]) -> Result<[f64; 3], String> {
    let m = object(v, path, &["x", "y", "z", "blocked"])?;
    let truth = [num(m, path, "x")?, num(m, path, "y")?, num(m, path, "z")?];
    for r in reports.iter_mut() {
        r.blocked = false;
    }
    for (k, id) in array(m, path, "blocked")?.iter().enumerate() {
        let p = format!("{}[{k}]", join(path, "blocked"));
        let id = node_id(id, &p)?;
        let r = reports
            .iter_mut()
            .find(|r| r.node == id)
            .ok_or_else(|| at(&p, "names a node with no report"))?;
        if r.blocked {
            return Err(at(&p, "names a node twice"));
        }
        r.blocked = true;
    }
    Ok(truth)
}

fn false_alarms_at(v: &Value, path: &str) -> Result<FalseAlarms, String> {
    let m = object(
        v,
        path,
        &[
            "ticks",
            "seconds",
            "published",
            "per_h",
            "events",
            "events_per_h",
        ],
    )?;
    Ok(FalseAlarms {
        ticks: size(m, path, "ticks")?,
        seconds: num(m, path, "seconds")?,
        published: size(m, path, "published")?,
        per_h: num_or_null(m, path, "per_h")?,
        events: size(m, path, "events")?,
        events_per_h: num_or_null(m, path, "events_per_h")?,
    })
}

/// Every field of a [`Config`] from its encoding, unvalidated: the
/// writer's exactness check ([`exact`]) and [`locate_config_at`].
fn locate_config_fields(v: &Value, path: &str) -> Result<Config, String> {
    let known = locate_config_json(&Config::default());
    let known: Vec<&str> = known
        .as_object()
        .map(|m| m.keys().map(String::as_str).collect())
        .unwrap_or_default();
    let m = object(v, path, &known)?;
    let f = |key: &str| num(m, path, key);
    let opt = |key: &str| num_or_null(m, path, key);
    let phone_seed = match get(m, path, "phone_seed")? {
        Value::Null => None,
        _ => Some(uint(m, path, "phone_seed")?),
    };
    Ok(Config {
        lambda_m: f("lambda_m")?,
        pitch_m: f("pitch_m")?,
        alpha: f("alpha")?,
        sigma: f("sigma")?,
        delta_c: f("delta_c")?,
        z_threshold: f("z_threshold")?,
        window_s: f("window_s")?,
        min_windows: size(m, path, "min_windows")?,
        link_signal: string(m, path, "link_signal")?.to_string(),
        calib_s: f("calib_s")?,
        tick_hz: f("tick_hz")?,
        wlm_gate_z: f("wlm_gate_z")?,
        wlm_floor: opt("wlm_floor")?,
        wlm_radius_scale: f("wlm_radius_scale")?,
        rti_radius_scale: f("rti_radius_scale")?,
        phone_radius_scale: f("phone_radius_scale")?,
        min_plausible_radius_m: f("min_plausible_radius_m")?,
        phone_rate_hz: f("phone_rate_hz")?,
        phone_seed,
        rtt_sigma_m: f("rtt_sigma_m")?,
        rtt_nlos_sigma_scale: f("rtt_nlos_sigma_scale")?,
        rtt_nlos_bias_m: f("rtt_nlos_bias_m")?,
        rtt_outlier_p: f("rtt_outlier_p")?,
        rtt_outlier_lo_m: f("rtt_outlier_lo_m")?,
        rtt_outlier_hi_m: f("rtt_outlier_hi_m")?,
        rtt_offset_sigma_m: f("rtt_offset_sigma_m")?,
        rtt_persist_sigma_m: f("rtt_persist_sigma_m")?,
        rtt_fail_p: f("rtt_fail_p")?,
        rssi_fail_p: f("rssi_fail_p")?,
        rssi_breakpoint_m: f("rssi_breakpoint_m")?,
        rssi_far_exponent: f("rssi_far_exponent")?,
        rssi_shadow_sigma_db: f("rssi_shadow_sigma_db")?,
        rssi_persist_sigma_db: f("rssi_persist_sigma_db")?,
        rssi_shadow_corr_m: f("rssi_shadow_corr_m")?,
        rssi_nlos_loss_db: f("rssi_nlos_loss_db")?,
        rssi_device_sigma_db: f("rssi_device_sigma_db")?,
        rssi_jitter_db: f("rssi_jitter_db")?,
        rssi_est_exponent: f("rssi_est_exponent")?,
        rssi_p0_dbm: opt("rssi_p0_dbm")?,
        phone_assumed_z_m: f("phone_assumed_z_m")?,
        huber_k: f("huber_k")?,
        phone_accel_psd: f("phone_accel_psd")?,
        phone_init_speed_std: f("phone_init_speed_std")?,
        phone_max_coast_s: f("phone_max_coast_s")?,
        tx_power_dbm: opt("tx_power_dbm")?,
        carrier_hz: opt("carrier_hz")?,
    })
}

fn locate_config_at(v: &Value, path: &str) -> Result<Config, String> {
    let cfg = locate_config_fields(v, path)?;
    // Validation names only the parameter: its reason may hold the value.
    cfg.validate().map_err(|e| match e {
        phone::Error::Config { param, .. } => {
            at(&join(path, param), "fails validation (spec/locate.md §9)")
        }
        _ => at(path, "fails validation (spec/locate.md §9)"),
    })?;
    Ok(cfg)
}

fn roomsim_config_at(v: &Value, path: &str) -> Result<RoomSimConfig, String> {
    let known = value(RoomSimConfig::default());
    let known: Vec<&str> = known
        .as_object()
        .map(|m| m.keys().map(String::as_str).collect())
        .unwrap_or_default();
    let m = object(v, path, &known)?;
    // Every key, so the manifest states the whole config it was run with
    // (Room Sim's own reader would default an absent one).
    if let Some(k) = known.iter().find(|k| !m.contains_key(**k)) {
        return Err(at(&join(path, k), "missing"));
    }
    // Room Sim's reader and §3.2 validation. Its `param` is a known key
    // (checked above); its reason may hold a value, so it is dropped.
    RoomSimConfig::from_json_str(&v.to_string()).map_err(|e| match e {
        saans_roomsim::Error::Config { param, .. } if known.contains(&param.as_str()) => at(
            &join(path, &param),
            "refused by Room Sim (spec/roomsim.md §3.2)",
        ),
        _ => at(path, "refused by Room Sim (spec/roomsim.md §3.2)"),
    })
}

/// The two configs are ones `run_roomsim` takes without a scene
/// ([`check_roomsim_config`], [`resolve_config`]; spec/locate.md §9,
/// §10.2): no `roles`, `quantize` on, at most one tick per slot, and a set
/// `phone_seed`, `tx_power_dbm` or `carrier_hz` the Room Sim run's own. An
/// unset one is valid: the run resolves it.
fn run_configs_at(rs: &RoomSimConfig, cfg: &Config) -> Result<(), String> {
    const WHY: &str = "not what the Room Sim run takes (spec/locate.md §9, §10.2)";
    check_roomsim_config(rs, cfg)
        .and_then(|()| resolve_config(rs, cfg).map(drop))
        .map_err(|e| match e {
            // The parameter only: the reason may hold a value.
            PipelineError::Phone(phone::Error::Config { param, .. }) => {
                at(&join("locate_config", param), WHY)
            }
            PipelineError::Refused { param, .. } => {
                let locate = locate_config_json(&Config::default()).get(param).is_some();
                let config = if locate {
                    "locate_config"
                } else {
                    "roomsim_config"
                };
                at(&join(config, param), WHY)
            }
            _ => at("roomsim_config", WHY),
        })
}

fn links_at(v: &Value, path: &str) -> Result<Vec<(i64, i64)>, String> {
    let items = v.as_array().ok_or_else(|| at(path, "not an array"))?;
    items
        .iter()
        .enumerate()
        .map(|(k, pair)| match pair.as_array().map(Vec::as_slice) {
            Some([tx, rx]) => match (tx.as_i64(), rx.as_i64()) {
                (Some(tx), Some(rx)) => Ok((tx, rx)),
                _ => Err(at(
                    &format!("{path}[{k}]"),
                    "not a [tx, rx] pair of integers",
                )),
            },
            _ => Err(at(
                &format!("{path}[{k}]"),
                "not a [tx, rx] pair of integers",
            )),
        })
        .collect()
}

/// The inverse of [`outcome_json`]. It checks the form only: the
/// publication test and the reasons an estimator may give are the reader's
/// (spec/livedemo.md §6.2). The error names the key, never a value.
pub fn outcome_from_json(v: &Value) -> Result<Outcome, String> {
    outcome_at(v, "outcome")
}

/// The inverse of [`link_json`]: `(tx, rx, value)`, a `null` `y` read as
/// `f64::NAN`. The error names the key, never a value.
pub fn link_from_json(v: &Value) -> Result<(i64, i64, LinkValue), String> {
    link_at(v, "link")
}

/// The inverse of [`report_json`], with `blocked: false`:
/// [`epoch_truth_from_json`] sets it from the truth. The error names the
/// key, never a value.
pub fn report_from_json(v: &Value) -> Result<ReportRecord, String> {
    report_at(v, "report")
}

/// The inverse of [`tick_truth_json`]; each `i` must be its index. The
/// error names the key, never a value.
pub fn tick_truth_from_json(v: &Value) -> Result<Vec<PersonTruthRecord>, String> {
    tick_truth_at(v, "truth")
}

/// The inverse of [`epoch_truth_json`]: the phone's `(x, y, z)`, and each
/// report's `blocked` set exactly for the listed nodes. A listed node with
/// no report, or listed twice, is refused. The error names the key, never
/// a value.
pub fn epoch_truth_from_json(v: &Value, reports: &mut [ReportRecord]) -> Result<[f64; 3], String> {
    epoch_truth_at(v, "truth", reports)
}

/// The inverse of [`locate_config_json`]: every field is required, and the
/// config must pass [`Config::validate`]. The error names the parameter,
/// never a value.
pub fn locate_config_from_json(v: &Value) -> Result<Config, String> {
    locate_config_at(v, "locate_config")
}

// ===================================================================
// Writer
// ===================================================================

#[derive(Serialize)]
struct SceneOut<'a> {
    name: &'a str,
    sha256: &'a str,
    provenance: &'static str,
}

#[derive(Serialize)]
struct EmptyRunOut<'a> {
    wlm: &'a FalseAlarms,
    rti: &'a FalseAlarms,
}

/// §6.1 line 1, in its key order.
#[derive(Serialize)]
struct ManifestLine<'a> {
    #[serde(rename = "type")]
    kind: &'static str,
    label: &'static str,
    spec: &'static str,
    scene: SceneOut<'a>,
    variant: &'static str,
    seed: u64,
    duration_s: f64,
    roomsim_config: &'a RoomSimConfig,
    locate_config: LocateConfigOut<'a>,
    links: &'a [(i64, i64)],
    has_phone: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    empty_run: Option<EmptyRunOut<'a>>,
}

/// §6.1 **Tick**.
#[derive(Serialize)]
struct TickLine {
    #[serde(rename = "type")]
    kind: &'static str,
    label: &'static str,
    t: f64,
    wlm: OutcomeOut,
    rti: OutcomeOut,
    links: Vec<LinkOut>,
    truth: Vec<PersonTruthOut>,
}

/// §6.1 **Epoch**.
#[derive(Serialize)]
struct EpochLine {
    #[serde(rename = "type")]
    kind: &'static str,
    label: &'static str,
    t: f64,
    rtt: OutcomeOut,
    rssi: OutcomeOut,
    reports: Vec<ReportOut>,
    truth: EpochTruthOut,
}

fn manifest_line(m: &Manifest) -> ManifestLine<'_> {
    ManifestLine {
        kind: "manifest",
        label: LABEL,
        spec: SPEC,
        scene: SceneOut {
            name: &m.scene_name,
            sha256: &m.scene_sha256,
            provenance: provenance_name(m.provenance),
        },
        variant: m.variant.as_str(),
        seed: m.seed,
        duration_s: m.duration_s,
        roomsim_config: &m.roomsim_config,
        locate_config: LocateConfigOut::from(&m.locate_config),
        links: &m.links,
        has_phone: m.has_phone,
        empty_run: m
            .empty_run
            .as_ref()
            .map(|(wlm, rti)| EmptyRunOut { wlm, rti }),
    }
}

fn tick_line(tick: &TickRecord, pairs: &[(i64, i64)], values: &[LinkValue]) -> TickLine {
    let TickRecord {
        label: _,
        t,
        wlm,
        rti,
        truth,
    } = tick;
    TickLine {
        kind: "tick",
        label: LABEL,
        t: *t,
        wlm: OutcomeOut::from(wlm),
        rti: OutcomeOut::from(rti),
        links: pairs
            .iter()
            .zip(values)
            .map(|(&(tx, rx), v)| LinkOut::new(tx, rx, v))
            .collect(),
        truth: tick_truth_out(truth),
    }
}

fn epoch_line(epoch: &EpochRecord) -> EpochLine {
    let EpochRecord {
        label: _,
        t,
        rtt,
        rssi,
        reports,
        truth,
    } = epoch;
    EpochLine {
        kind: "epoch",
        label: LABEL,
        t: *t,
        rtt: OutcomeOut::from(rtt),
        rssi: OutcomeOut::from(rssi),
        reports: reports.iter().map(ReportOut::from).collect(),
        truth: epoch_truth_out(*truth, reports),
    }
}

fn write_line(w: &mut impl Write, line: &impl Serialize) -> io::Result<()> {
    serde_json::to_writer(&mut *w, line)?;
    w.write_all(b"\n")
}

fn invalid(reason: String) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidInput, reason)
}

/// The key of the first value that is not finite. An unset option is
/// given as `0.0`: it is written `null` and read back `None`, exactly.
fn non_finite<const N: usize>(floats: [(&'static str, f64); N]) -> Option<&'static str> {
    floats
        .into_iter()
        .find(|(_, x)| !x.is_finite())
        .map(|(key, _)| key)
}

fn estimate_non_finite(o: &Outcome) -> Option<&'static str> {
    let Outcome::Estimate(e) = o else {
        return None;
    };
    non_finite([
        ("estimate.x", e.x),
        ("estimate.y", e.y),
        ("estimate.radius_m", e.radius_m),
        ("estimate.cov_xx", e.cov_xx),
        ("estimate.cov_xy", e.cov_xy),
        ("estimate.cov_yy", e.cov_yy),
        ("estimate.confidence", e.confidence.unwrap_or(0.0)),
    ])
}

/// `records` has an exact encoding: every float is finite but a link's
/// `y`, which §4 writes `null` and the reader reads as NaN. Any other
/// non-finite value would come back changed (an option as `None`) or be
/// refused, so the error names it, by its place in `records`.
fn exact(records: &Records) -> Result<(), String> {
    const WHY: &str = "not finite; JSON cannot carry it exactly (spec/livedemo.md §6.1)";
    let m = &records.manifest;
    if !m.duration_s.is_finite() {
        return Err(at("manifest.duration_s", WHY));
    }
    if let Some((wlm, rti)) = &m.empty_run {
        for (name, fa) in [("wlm", wlm), ("rti", rti)] {
            if let Some(key) = non_finite([
                ("seconds", fa.seconds),
                ("per_h", fa.per_h.unwrap_or(0.0)),
                ("events_per_h", fa.events_per_h.unwrap_or(0.0)),
            ]) {
                return Err(at(&format!("manifest.empty_run.{name}.{key}"), WHY));
            }
        }
    }
    // Each config reads back equal (unvalidated), or one of its values is
    // not finite.
    let rs = serde_json::from_value::<RoomSimConfig>(value(&m.roomsim_config));
    if rs.ok().as_ref() != Some(&m.roomsim_config) {
        return Err(at("manifest.roomsim_config", WHY));
    }
    let cfg = locate_config_fields(&locate_config_json(&m.locate_config), "");
    if cfg.ok().as_ref() != Some(&m.locate_config) {
        return Err(at("manifest.locate_config", WHY));
    }
    for (k, tick) in records.ticks.iter().enumerate() {
        if !tick.t.is_finite() {
            return Err(at(&format!("ticks[{k}].t"), WHY));
        }
        for (name, o) in [("wlm", &tick.wlm), ("rti", &tick.rti)] {
            if let Some(key) = estimate_non_finite(o) {
                return Err(at(&format!("ticks[{k}].{name}.{key}"), WHY));
            }
        }
        for (i, p) in tick.truth.iter().enumerate() {
            if let Some(key) = non_finite([("x", p.x), ("y", p.y)]) {
                return Err(at(&format!("ticks[{k}].truth[{i}].{key}"), WHY));
            }
        }
    }
    for (k, values) in records.links.iter().enumerate() {
        if let Some(l) = values
            .iter()
            .position(|v| !v.gate.unwrap_or(0.0).is_finite())
        {
            return Err(at(&format!("links[{k}][{l}].gate"), WHY));
        }
    }
    for (j, epoch) in records.epochs.iter().enumerate() {
        if !epoch.t.is_finite() {
            return Err(at(&format!("epochs[{j}].t"), WHY));
        }
        for (name, o) in [("rtt", &epoch.rtt), ("rssi", &epoch.rssi)] {
            if let Some(key) = estimate_non_finite(o) {
                return Err(at(&format!("epochs[{j}].{name}.{key}"), WHY));
            }
        }
        if let Some(i) = epoch
            .reports
            .iter()
            .position(|r| !r.rtt_m.unwrap_or(0.0).is_finite())
        {
            return Err(at(&format!("epochs[{j}].reports[{i}].rtt_m"), WHY));
        }
        let [x, y, z] = epoch.truth;
        if let Some(key) = non_finite([("x", x), ("y", y), ("z", z)]) {
            return Err(at(&format!("epochs[{j}].truth.{key}"), WHY));
        }
    }
    Ok(())
}

/// Write `records` as a §6.1 file: the manifest
/// (`{"type":"manifest","label":"SIM","spec":"livedemo v1",…}`), then
/// every tick and epoch ordered by `t`, the epoch first at equal `t`. One
/// line each, in the spec's key order, floats in serde_json's shortest
/// round-trip form. Give it a `BufWriter` for a file.
///
/// Refused (`InvalidInput`, before anything is written): a tick or epoch
/// whose label is not `"SIM"` (ADR-008 decision 11), a `links` list count
/// other than one per tick, a tick's list not one value per
/// [`Manifest::links`] pair, and a non-finite value other than a link's
/// `y`, which JSON cannot carry exactly (the error names where). Records
/// the reader would refuse otherwise (out of order, unpublishable
/// estimates …) are written as they are.
pub fn write_records(w: &mut impl Write, records: &Records) -> io::Result<()> {
    let Records {
        manifest,
        ticks,
        links,
        epochs,
    } = records;
    if ticks.iter().any(|r| r.label != LABEL) || epochs.iter().any(|r| r.label != LABEL) {
        return Err(invalid(
            "label: only SIM records are written (ADR-008 decision 11)".into(),
        ));
    }
    if links.len() != ticks.len() {
        return Err(invalid(format!(
            "links: {} lists for {} ticks; one list per tick",
            links.len(),
            ticks.len()
        )));
    }
    if let Some(k) = links.iter().position(|l| l.len() != manifest.links.len()) {
        return Err(invalid(format!(
            "links[{k}]: not one value per manifest link"
        )));
    }
    exact(records).map_err(invalid)?;
    write_line(w, &manifest_line(manifest))?;
    let (mut i, mut j) = (0, 0);
    while i < ticks.len() || j < epochs.len() {
        let epoch_next = j < epochs.len() && (i == ticks.len() || epochs[j].t <= ticks[i].t);
        if epoch_next {
            write_line(w, &epoch_line(&epochs[j]))?;
            j += 1;
        } else {
            write_line(w, &tick_line(&ticks[i], &manifest.links, &links[i]))?;
            i += 1;
        }
    }
    Ok(())
}

// ===================================================================
// Reader
// ===================================================================

const MANIFEST_KEYS: [&str; 12] = [
    "type",
    "label",
    "spec",
    "scene",
    "variant",
    "seed",
    "duration_s",
    "roomsim_config",
    "locate_config",
    "links",
    "has_phone",
    "empty_run",
];
const TICK_KEYS: [&str; 7] = ["type", "label", "t", "wlm", "rti", "links", "truth"];
const EPOCH_KEYS: [&str; 7] = ["type", "label", "t", "rtt", "rssi", "reports", "truth"];

fn manifest_at(v: &Value) -> Result<Manifest, String> {
    let m = object(v, "", &MANIFEST_KEYS)?;
    if string(m, "", "spec")? != SPEC {
        return Err(at("spec", "not \"livedemo v1\" (spec/livedemo.md §6.2)"));
    }

    let scene = object(
        get(m, "", "scene")?,
        "scene",
        &["name", "sha256", "provenance"],
    )?;
    let scene_name = string(scene, "scene", "name")?.to_string();
    let scene_sha256 = string(scene, "scene", "sha256")?;
    if !(scene_sha256.len() == 64
        && scene_sha256
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b)))
    {
        return Err(at("scene.sha256", "not 64 lower-case hex digits"));
    }
    let provenance =
        provenance_from_name(string(scene, "scene", "provenance")?).ok_or_else(|| {
            at(
                "scene.provenance",
                "not surveyed, roomplan_import, hand_written or generated",
            )
        })?;

    let variant = Variant::parse(string(m, "", "variant")?).ok_or_else(|| {
        at(
            "variant",
            "not declared, with_walker or empty (spec/livedemo.md §3.1)",
        )
    })?;
    let seed = uint(m, "", "seed")?;
    let duration_s = num(m, "", "duration_s")?;
    // §3.2: `q_end = 40 * duration_s` quanta, exactly.
    if !(duration_s > 0.0 && (40.0 * duration_s).fract() == 0.0) {
        return Err(at(
            "duration_s",
            "not > 0 with 40 * duration_s an integer (spec/livedemo.md §3.2, §6.2)",
        ));
    }

    let rs = roomsim_config_at(get(m, "", "roomsim_config")?, "roomsim_config")?;
    if rs.slot_s != SLOT_S {
        return Err(at(
            "roomsim_config.slot_s",
            "not 0.0125 (spec/livedemo.md §3.2, §6.2)",
        ));
    }
    if rs.seed != seed {
        return Err(at("roomsim_config.seed", "not the manifest's seed"));
    }
    if rs.duration_s != duration_s {
        return Err(at(
            "roomsim_config.duration_s",
            "not the manifest's duration_s",
        ));
    }
    // The variant's people (§3.1, [`Variant::apply`]), as far as they show
    // without the scene.
    match variant {
        Variant::Empty if rs.n_people != Some(0) => {
            return Err(at(
                "roomsim_config.n_people",
                "not 0, but the variant is empty (spec/livedemo.md §3.1)",
            ));
        }
        Variant::WithWalker if !matches!(rs.n_people, Some(n) if n >= 1 && rs.phone_carrier == n - 1) =>
        {
            return Err(at(
                "roomsim_config.phone_carrier",
                "not the added last person, but the variant is with_walker (spec/livedemo.md §3.1)",
            ));
        }
        _ => {}
    }

    let locate_config = locate_config_at(get(m, "", "locate_config")?, "locate_config")?;
    run_configs_at(&rs, &locate_config)?;
    let links = links_at(get(m, "", "links")?, "links")?;
    let has_phone = boolean(m, "", "has_phone")?;
    let empty_run = match (variant, m.get("empty_run")) {
        (Variant::Empty, None) => None,
        (Variant::Empty, Some(_)) => {
            return Err(at(
                "empty_run",
                "present, but the variant is empty (spec/livedemo.md §6.1)",
            ));
        }
        (_, None) => {
            return Err(at(
                "empty_run",
                "missing; only variant empty has none (spec/livedemo.md §5.3, §6.1)",
            ));
        }
        (_, Some(v)) => {
            let e = object(v, "empty_run", &["wlm", "rti"])?;
            Some((
                false_alarms_at(get(e, "empty_run", "wlm")?, "empty_run.wlm")?,
                false_alarms_at(get(e, "empty_run", "rti")?, "empty_run.rti")?,
            ))
        }
    };

    Ok(Manifest {
        scene_name,
        scene_sha256: scene_sha256.to_string(),
        provenance,
        variant,
        seed,
        duration_s,
        roomsim_config: rs,
        locate_config,
        links,
        has_phone,
        empty_run,
    })
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Kind {
    Tick,
    Epoch,
}

/// The order of §6.1: records by `t`, the epoch first at equal `t`; ticks,
/// and epochs, strictly increasing; `0 <= t < duration_s` (§6.2).
#[derive(Default)]
struct Order {
    last: Option<(f64, Kind)>,
    tick: Option<f64>,
    epoch: Option<f64>,
}

impl Order {
    fn place(&mut self, kind: Kind, t: f64, duration_s: f64) -> Result<(), String> {
        if !(t >= 0.0) {
            return Err(at("t", "negative"));
        }
        if !(t < duration_s) {
            return Err(at("t", "not before duration_s (spec/livedemo.md §6.2)"));
        }
        let (name, prev) = match kind {
            Kind::Tick => ("tick", &mut self.tick),
            Kind::Epoch => ("epoch", &mut self.epoch),
        };
        if prev.is_some_and(|p| !(t > p)) {
            return Err(format!(
                "{name} out of order: t is not after the previous {name}'s (spec/livedemo.md §6.2)"
            ));
        }
        if let Some((p, pk)) = self.last {
            if t < p || (t == p && pk == Kind::Tick && kind == Kind::Epoch) {
                return Err("record out of order: records are ordered by t, the epoch first at equal t (spec/livedemo.md §6.1)".into());
            }
        }
        *prev = Some(t);
        self.last = Some((t, kind));
        Ok(())
    }
}

/// `o` from the file is one `allowed` reason, or an estimate that passes
/// spec/locate.md §8's publication test with `radius_calibrated: false`.
fn published_at(
    v: &Value,
    path: &str,
    allowed: &[Reason],
    min_plausible_radius_m: f64,
) -> Result<Outcome, String> {
    let o = outcome_at(v, path)?;
    match &o {
        Outcome::Abstain(r) if !allowed.contains(r) => Err(at(
            &join(path, "abstain.reason"),
            "not a reason this estimator gives (spec/locate.md §11)",
        )),
        Outcome::Estimate(e)
            if !(e.x.is_finite() && e.y.is_finite() && plausible(e.radius_m, min_plausible_radius_m)) =>
        {
            Err(at(
                &join(path, "estimate"),
                "fails the publication test: finite x, y and radius_m, radius_m > 0 and >= min_plausible_radius_m (spec/locate.md §8)",
            ))
        }
        Outcome::Estimate(e) if e.radius_calibrated => Err(at(
            &join(path, "estimate.radius_calibrated"),
            "true, but Phase 5 calibrates nothing (spec/livedemo.md §6.2)",
        )),
        _ => Ok(o),
    }
}

/// `first_people` is the first tick's number of people, once read.
fn tick_at(
    v: &Value,
    man: &Manifest,
    order: &mut Order,
    first_people: &mut Option<usize>,
) -> Result<(TickRecord, Vec<LinkValue>), String> {
    let m = object(v, "", &TICK_KEYS)?;
    let t = num(m, "", "t")?;
    order.place(Kind::Tick, t, man.duration_s)?;
    let min_r = man.locate_config.min_plausible_radius_m;
    let wlm = published_at(get(m, "", "wlm")?, "wlm", WLM_REASONS, min_r)?;
    let rti = published_at(get(m, "", "rti")?, "rti", RTI_REASONS, min_r)?;
    let items = array(m, "", "links")?;
    if items.len() != man.links.len() {
        return Err(at(
            "links",
            &format!(
                "{} items, but the manifest lists {} links (spec/livedemo.md §6.1)",
                items.len(),
                man.links.len()
            ),
        ));
    }
    let mut links = Vec::with_capacity(items.len());
    for (l, (item, &pair)) in items.iter().zip(&man.links).enumerate() {
        let p = format!("links[{l}]");
        let (tx, rx, value) = link_at(item, &p)?;
        if (tx, rx) != pair {
            return Err(at(
                &p,
                "not the manifest's link at this index; links are in the manifest's order (spec/livedemo.md §6.1)",
            ));
        }
        links.push(value);
    }
    let truth = tick_truth_at(get(m, "", "truth")?, "truth")?;
    // Room Sim lists every person in every truth (spec/roomsim.md §3.1,
    // §13): `n_people` of them when set, else the scene's, so the same
    // number at every tick. Whether a tick is occupied then follows from
    // the manifest and the scene (§5.4).
    match (man.roomsim_config.n_people, *first_people) {
        (Some(n), _) if truth.len() != n => {
            return Err(at(
                "truth",
                "not roomsim_config.n_people people (spec/livedemo.md §3.1, §5.4)",
            ));
        }
        (None, Some(n)) if truth.len() != n => {
            return Err(at(
                "truth",
                "not as many people as the first tick's; Room Sim's people are fixed for the run (spec/livedemo.md §5.4)",
            ));
        }
        _ => {}
    }
    first_people.get_or_insert(truth.len());
    Ok((
        TickRecord {
            label: LABEL,
            t,
            wlm,
            rti,
            truth,
        },
        links,
    ))
}

fn epoch_at(v: &Value, man: &Manifest, order: &mut Order) -> Result<EpochRecord, String> {
    if !man.has_phone {
        return Err("an epoch, but the manifest's has_phone is false".into());
    }
    let m = object(v, "", &EPOCH_KEYS)?;
    let t = num(m, "", "t")?;
    order.place(Kind::Epoch, t, man.duration_s)?;
    let min_r = man.locate_config.min_plausible_radius_m;
    let rtt = published_at(get(m, "", "rtt")?, "rtt", PHONE_REASONS, min_r)?;
    let rssi = published_at(get(m, "", "rssi")?, "rssi", PHONE_REASONS, min_r)?;
    let mut reports = Vec::new();
    for (k, item) in array(m, "", "reports")?.iter().enumerate() {
        let p = format!("reports[{k}]");
        let r = report_at(item, &p)?;
        // One per node, ids ascending (spec/locate.md §11).
        if reports
            .last()
            .is_some_and(|q: &ReportRecord| q.node >= r.node)
        {
            return Err(at(
                &join(&p, "node"),
                "not ascending; one report per node, in id order",
            ));
        }
        reports.push(r);
    }
    let truth = epoch_truth_at(get(m, "", "truth")?, "truth", &mut reports)?;
    Ok(EpochRecord {
        label: LABEL,
        t,
        rtt,
        rssi,
        reports,
        truth,
    })
}

/// The custom error of a repeated key. It never names the key.
const REPEATED: &str = "a repeated key in one object";

/// A JSON value in which no object repeats a key (§6.2). serde_json's
/// `Value` keeps the last of a repeated key, so a line with
/// `"label":"REAL","label":"SIM"` would read as SIM. Everything else is
/// read as `Value` reads it, numbers with `float_roundtrip`.
struct Strict(Value);

impl<'de> Deserialize<'de> for Strict {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        d.deserialize_any(StrictVisitor).map(Strict)
    }
}

struct StrictVisitor;

impl<'de> Visitor<'de> for StrictVisitor {
    type Value = Value;

    fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("a JSON value")
    }

    fn visit_bool<E: de::Error>(self, b: bool) -> Result<Value, E> {
        Ok(Value::Bool(b))
    }

    fn visit_i64<E: de::Error>(self, n: i64) -> Result<Value, E> {
        Ok(Value::from(n))
    }

    fn visit_u64<E: de::Error>(self, n: u64) -> Result<Value, E> {
        Ok(Value::from(n))
    }

    fn visit_f64<E: de::Error>(self, x: f64) -> Result<Value, E> {
        Ok(Number::from_f64(x).map_or(Value::Null, Value::Number))
    }

    fn visit_str<E: de::Error>(self, s: &str) -> Result<Value, E> {
        Ok(Value::from(s))
    }

    fn visit_string<E: de::Error>(self, s: String) -> Result<Value, E> {
        Ok(Value::String(s))
    }

    fn visit_unit<E: de::Error>(self) -> Result<Value, E> {
        Ok(Value::Null)
    }

    fn visit_seq<A: SeqAccess<'de>>(self, mut seq: A) -> Result<Value, A::Error> {
        let mut items = Vec::new();
        while let Some(Strict(v)) = seq.next_element()? {
            items.push(v);
        }
        Ok(Value::Array(items))
    }

    fn visit_map<A: MapAccess<'de>>(self, mut map: A) -> Result<Value, A::Error> {
        let mut m = Map::new();
        while let Some(key) = map.next_key::<String>()? {
            let Strict(v) = map.next_value()?;
            if m.insert(key, v).is_some() {
                return Err(de::Error::custom(REPEATED));
            }
        }
        Ok(Value::Object(m))
    }
}

/// One line as a JSON object; the problem otherwise.
fn parse_line(bytes: &[u8]) -> Result<Value, String> {
    let text = std::str::from_utf8(bytes).map_err(|_| "not UTF-8".to_string())?;
    if text.trim().is_empty() {
        return Err("an empty line".into());
    }
    let Strict(v) = serde_json::from_str(text).map_err(|e| {
        if e.is_data() && e.to_string().starts_with(REPEATED) {
            format!("{REPEATED} (column {}; spec/livedemo.md §6.2)", e.column())
        } else {
            format!("not JSON (column {})", e.column())
        }
    })?;
    if !v.is_object() {
        return Err("not a JSON object".into());
    }
    Ok(v)
}

/// Read a §6.1 file, making every §6.2 refusal that needs no embedded
/// scene, each naming its 1-based line:
/// - more than `max_bytes` bytes ([`MAX_BYTES`] for the gateway); a line
///   that is not UTF-8 JSON, or with a key repeated in one object;
/// - a first line that is not a manifest; a second manifest; a `type`
///   other than `tick` or `epoch`;
/// - any line whose `label` is not `"SIM"`;
/// - `spec` not `"livedemo v1"`, an unknown variant, a `scene.sha256`
///   that is not 64 lower-case hex digits;
/// - `duration_s` not `> 0` or `40 * duration_s` not an integer;
///   `roomsim_config` with a `slot_s` other than [`SLOT_S`], a `seed` or
///   `duration_s` other than the manifest's, people that are not the
///   variant's, a missing key, or one Room Sim refuses; `locate_config`
///   missing a field or failing [`Config::validate`]; configs that
///   `run_roomsim` refuses without a scene ([`check_roomsim_config`],
///   [`resolve_config`]: `roles`, `quantize`, `tick_hz`, or a set
///   `phone_seed`, `tx_power_dbm` or `carrier_hz` not the run's);
///   `empty_run` present for variant `empty` or absent for another;
/// - an unknown key in any object, or a missing one;
/// - ticks, or epochs, out of order, records not ordered by `t` with the
///   epoch first at equal `t`, or a record with `t < 0` or
///   `t >= duration_s`;
/// - a tick whose links are not the manifest's, in its order, or whose
///   truth does not list the run's people: `roomsim_config.n_people` of
///   them when set, else as many as the first tick's;
/// - an estimate failing spec/locate.md §8's publication test against
///   `locate_config.min_plausible_radius_m`, or with `radius_calibrated`
///   true; a reason its estimator never gives (spec/locate.md §11);
/// - phone reports not one per node in ascending id order; a `blocked`
///   node with no report;
/// - an epoch when `has_phone` is false.
///
/// Matching `scene_sha256` to an embedded scene, one file per scene, and
/// (with `n_people` unset) the scene's people count to the ticks' truth,
/// are the caller's.
pub fn read_records(r: impl BufRead, max_bytes: u64) -> Result<Records, RecordsError> {
    let mut r = r.take(max_bytes.saturating_add(1));
    let mut buf = Vec::new();
    let mut read: u64 = 0;
    let mut line = 0usize;
    let mut manifest: Option<Manifest> = None;
    let mut ticks = Vec::new();
    let mut links = Vec::new();
    let mut epochs = Vec::new();
    let mut order = Order::default();
    let mut first_people = None;
    loop {
        buf.clear();
        let n = r
            .read_until(b'\n', &mut buf)
            .map_err(|e| refused(line + 1, format!("cannot be read ({})", e.kind())))?;
        if n == 0 {
            break;
        }
        line += 1;
        read += n as u64;
        if read > max_bytes {
            return Err(refused(
                line,
                format!("the file is larger than {max_bytes} bytes (spec/livedemo.md §6.2)"),
            ));
        }
        let v = parse_line(&buf).map_err(|why| {
            refused(
                line,
                if line == 1 {
                    format!("the first line is not a manifest: {why}")
                } else {
                    why
                },
            )
        })?;
        if v.get("label").and_then(Value::as_str) != Some(LABEL) {
            return Err(refused(line, "label: not \"SIM\" (spec/livedemo.md §6.2)"));
        }
        let kind = v.get("type").and_then(Value::as_str);
        let Some(man) = manifest.as_ref() else {
            if kind != Some("manifest") {
                return Err(refused(line, "the first line is not a manifest"));
            }
            manifest = Some(manifest_at(&v).map_err(|why| refused(line, why))?);
            continue;
        };
        match kind {
            Some("tick") => {
                let (tick, values) = tick_at(&v, man, &mut order, &mut first_people)
                    .map_err(|why| refused(line, why))?;
                ticks.push(tick);
                links.push(values);
            }
            Some("epoch") => {
                epochs.push(epoch_at(&v, man, &mut order).map_err(|why| refused(line, why))?);
            }
            Some("manifest") => return Err(refused(line, "a second manifest")),
            _ => return Err(refused(line, "type: not tick or epoch")),
        }
    }
    let manifest = manifest
        .ok_or_else(|| refused(1, "the first line is not a manifest: the file is empty"))?;
    Ok(Records {
        manifest,
        ticks,
        links,
        epochs,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_reason_posture_and_provenance_spelling_reads_back() {
        for r in REASONS {
            assert_eq!(reason_from_str(r.as_str()), Some(r));
        }
        let mut names: Vec<_> = REASONS.iter().map(|r| r.as_str()).collect();
        names.sort_unstable();
        names.dedup();
        assert_eq!(names.len(), REASONS.len(), "REASONS lists each reason once");
        for p in [Posture::Stand, Posture::Sit, Posture::Lie] {
            assert_eq!(posture_from_name(p.name()), Some(p));
        }
        for p in [
            Provenance::Surveyed,
            Provenance::RoomplanImport,
            Provenance::HandWritten,
        ] {
            assert_eq!(provenance_from_name(provenance_name(p)), Some(p));
        }
        assert_eq!(reason_from_str("walker"), None);
        assert_eq!(posture_from_name("walker"), None);
        assert_eq!(provenance_from_name("walker"), None);
    }

    #[test]
    fn the_estimator_reason_sets_are_spec_locate_11s() {
        for r in WLM_REASONS.iter().chain(RTI_REASONS).chain(PHONE_REASONS) {
            assert!(REASONS.contains(r));
        }
        assert!(!WLM_REASONS.contains(&Reason::NotSignificant));
        assert!(!RTI_REASONS.contains(&Reason::NoDisturbance));
        assert!(!PHONE_REASONS.contains(&Reason::NoDisturbance));
    }

    #[test]
    fn strict_reads_as_value_reads_and_refuses_a_repeated_key() {
        let text = r#"{"a":[-0.0,0,-1,18446744073709551615,0.30000000000000004,5e-324,1.7976931348623157e308,null,true,"s",{"b":{}}],"c":{"d":[{"e":1},{"e":2}]}}"#;
        let Strict(strict) = serde_json::from_str(text).unwrap();
        let value: Value = serde_json::from_str(text).unwrap();
        assert_eq!(strict, value);
        assert_eq!(format!("{strict:?}"), format!("{value:?}"));
        for twice in [
            r#"{"label":"REAL","label":"SIM"}"#,
            r#"{"a":{"b":1,"b":1}}"#,
            r#"{"a":[{"b":1},{"c":1,"c":2}]}"#,
        ] {
            let e = parse_line(twice.as_bytes()).unwrap_err();
            assert!(e.starts_with(REPEATED), "{e}");
            assert!(!e.contains("label") && !e.contains('"'), "{e}");
        }
        assert!(parse_line(b"{\"a\":1}{")
            .unwrap_err()
            .starts_with("not JSON"));
    }

    #[test]
    fn order_takes_the_epoch_first_at_equal_t() {
        let mut o = Order::default();
        o.place(Kind::Epoch, 0.0, 10.0).unwrap();
        o.place(Kind::Epoch, 1.0, 10.0).unwrap();
        o.place(Kind::Tick, 1.0, 10.0).unwrap();
        assert!(o.place(Kind::Epoch, 1.0, 10.0).is_err());
        let mut o = Order::default();
        o.place(Kind::Tick, 1.0, 10.0).unwrap();
        assert!(o.place(Kind::Epoch, 1.0, 10.0).is_err());
        assert!(Order::default().place(Kind::Tick, 10.0, 10.0).is_err());
        assert!(Order::default().place(Kind::Tick, -0.5, 10.0).is_err());
    }
}
