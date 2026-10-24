//! The Room Sim pipeline (spec/locate.md §4.3, §10, §11), behind the
//! `roomsim` cargo feature.
//!
//! Everything here is **SIM**: the inputs are Room Sim packets and truth
//! (spec/roomsim.md), the geometry is the scene (SCENE). A Room Sim run is
//! turned into
//! - one **tick record** per §10.1 tick: the weighted-link-midpoint
//!   baseline (§5) and RTI (the ported `invert`/`track`), each an estimate
//!   with a radius or an abstention with a reason;
//! - one **epoch record** per fired phone epoch (§6, §7), from the phone
//!   module, which is simulated only (ADR-008 decision 16).
//!
//! The scene is read through `saans-scene` without `write` (ADR-008
//! decision 1). No `Person.id` or `Phone.id` is read from the scene here:
//! people are `people[i]` by index, so no id can reach a record, log or
//! error.
//!
//! Refusals name the parameter and never echo a label (§1, §2, §4, §9,
//! §10): a label other than `"SIM"` for a Room Sim run (`label`), a Room
//! Sim config with any `roles` entry, two nodes at one `(x, y)` (`nodes`),
//! `quantize` false, a `tx_period` too slow to fill a window, a
//! calibration that raises (`calib_s`), any §9 parameter that fails
//! validation, `phone_seed`, `tx_power_dbm` or `carrier_hz` set to
//! something other than the run's, and a `tick_hz` above one tick per
//! Room Sim slot or whose ticks stop advancing (`tick_hz`).

// NaN fails closed: deciding comparisons are written `!(a >= b)` on purpose.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::borrow::Borrow;
use std::sync::Arc;

use saans_roomsim::people::cylinder_top;
use saans_roomsim::{Buffer, Config as RoomSimConfig, Packet, Posture, Simulator, Truth};
use saans_scene::Scene;

use crate::baseline::{wlm, LinkGeom, WlmParams};
use crate::geometry::{Geometry, Node, PixelGrid, Room};
use crate::invert::{Inverter, InverterOptions};
use crate::measure::{
    new_csi_frame, CalibrationMode, CsiFrame, LinkMeasurement, LinkSignal, MeasurementConfig,
    RssiSignal, StaticSuppressedSignal, TotalPowerSignal,
};
use crate::phone::{
    self, Blockers, BoxObj, Config, Cylinder, EpochRecord, PhoneNode, PhoneRun, TruthSample, Wall,
};
use crate::pynum::LocateError;
use crate::radius::{publish, Outcome, Reason};
use crate::track::{Tracker, TrackerConfig};
use crate::LABEL;

// ===================================================================
// Errors
// ===================================================================

/// Why a pipeline run was refused or failed. Messages name the parameter
/// or field and never echo a label, a `Person.id` or a `Phone.id`.
#[derive(Debug, Clone, PartialEq)]
pub enum PipelineError {
    /// The run is refused (§2, §4.1, §4.2, §4.3, §9, §10.1); `param` names
    /// the parameter: `roles`, `nodes`, `quantize`, `tx_period`, `calib_s`,
    /// `tick_hz`, `label` (a person label that is not SIM, REAL or
    /// SIM->REAL; never echoed) or `people_posture` (a lying person on no
    /// object, which Room Sim never produces). A §9 parameter that fails
    /// validation or is not Room Sim's is [`PipelineError::Phone`].
    Refused { param: &'static str, reason: String },
    /// A label other than `"SIM"` for Room Sim input (the phone is
    /// simulated only, ADR-008 decision 16). The given label is not
    /// repeated.
    Label,
    /// Room Sim rejected the scene or its configuration (its error names
    /// the parameter, and people as `people[i]`).
    RoomSim(saans_roomsim::Error),
    /// A ported module raised (its Python exception's name and message).
    Locate(LocateError),
    /// A new Locate piece refused its configuration or input (§9).
    Phone(phone::Error),
}

impl PipelineError {
    /// The parameter or field the error names, when it names one.
    pub fn param(&self) -> Option<&str> {
        match self {
            PipelineError::Refused { param, .. } => Some(param),
            PipelineError::Label => Some("label"),
            PipelineError::RoomSim(saans_roomsim::Error::Config { param, .. }) => Some(param),
            PipelineError::Phone(phone::Error::Config { param, .. }) => Some(param),
            PipelineError::Phone(phone::Error::Input { field, .. }) => Some(field),
            PipelineError::Phone(phone::Error::Label) => Some("label"),
            _ => None,
        }
    }
}

impl std::fmt::Display for PipelineError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PipelineError::Refused { param, reason } => write!(f, "{param}: {reason}"),
            PipelineError::Label => f.write_str(
                "label: Room Sim input is simulated (ADR-008 decision 16); the only label is \"SIM\"",
            ),
            PipelineError::RoomSim(e) => write!(f, "Room Sim: {e}"),
            PipelineError::Locate(e) => write!(f, "{e}"),
            PipelineError::Phone(e) => write!(f, "{e}"),
        }
    }
}

impl std::error::Error for PipelineError {}

impl From<saans_roomsim::Error> for PipelineError {
    fn from(e: saans_roomsim::Error) -> Self {
        PipelineError::RoomSim(e)
    }
}

impl From<LocateError> for PipelineError {
    fn from(e: LocateError) -> Self {
        PipelineError::Locate(e)
    }
}

impl From<phone::Error> for PipelineError {
    fn from(e: phone::Error) -> Self {
        PipelineError::Phone(e)
    }
}

fn refused(param: &'static str, reason: impl Into<String>) -> PipelineError {
    PipelineError::Refused {
        param,
        reason: reason.into(),
    }
}

// ===================================================================
// Records (§11)
// ===================================================================

/// One person's truth at a tick (§11): Room Sim's `people[i]` at the last
/// transmission with `t <= t_k`. SIM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersonTruthRecord {
    pub x: f64,
    pub y: f64,
    pub walking: bool,
    pub posture: Posture,
}

/// One §10.1 tick (§11 `{label, t, wlm, rti, truth}`).
#[derive(Debug, Clone, PartialEq)]
pub struct TickRecord {
    /// The caller's label; `"SIM"` for every Room Sim run.
    pub label: &'static str,
    /// `t_k = window_s + float(k) / tick_hz`.
    pub t: f64,
    /// The weighted-link-midpoint baseline (§5): reasons
    /// `no_disturbance`, `implausible_radius`.
    pub wlm: Outcome,
    /// RTI (§10.1): reasons `not_significant`, `implausible_radius`;
    /// `confidence` is the tracker's.
    pub rti: Outcome,
    /// People in index order (`people[i]`), never by a scene id.
    pub truth: Vec<PersonTruthRecord>,
}

/// What the empty-room calibration (§4.2, §10.2 step 1) produced, in the
/// localize canonical link order.
#[derive(Debug, Clone, PartialEq)]
pub struct CalibrationSummary {
    /// Received calibration-run packets with `t >= duration_s` that went
    /// through the §4.3 glue.
    pub packets: usize,
    /// Per-link empty-room variance, dB² for `rssi_dbm`.
    pub baseline: Vec<f64>,
    /// `null_sigma = meas.baseline_std_vector()` (§4.1).
    pub null_sigma: Vec<f64>,
    /// `false` for a link with fewer than `min_windows` window estimates,
    /// which got the median over calibrated links (§4.2).
    pub calibrated_mask: Vec<bool>,
}

/// Counters of one run (diagnostics, SIM).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Diagnostics {
    /// Received occupied-run packets fed to the measurement.
    pub packets: usize,
    /// The measurement's counters (ported `LinkMeasurement`).
    pub clock_restarts: u64,
    pub stale_samples: u64,
    pub window_overflows: u64,
    /// The inverter's pixels inside no link's ellipse.
    pub blind_pixels: usize,
    /// The inverter's rows after the reciprocal-pair collapse.
    pub n_independent_links: usize,
}

/// A whole Room Sim run through Locate (§10.2). Every record is SIM.
#[derive(Debug, Clone, PartialEq)]
pub struct RoomSimRun {
    /// Always `"SIM"`.
    pub label: &'static str,
    /// Room Sim's base per-link rate `1.0 / (float(nT) * slot_s)` (§4.3),
    /// also the measurement's `sample_rate_hz` (never a parameter, §9).
    pub rate_hz: f64,
    /// The measurement's `min_samples` (§4.1).
    pub min_samples: i64,
    /// The localize canonical link order `(tx, rx)` (§2).
    pub links: Vec<(i64, i64)>,
    pub calibration: CalibrationSummary,
    /// One per §10.1 tick.
    pub ticks: Vec<TickRecord>,
    /// One per fired phone epoch (§10.2 step 3); empty when the scene has
    /// no phone.
    pub epochs: Vec<EpochRecord>,
    /// Whether the scene has a phone.
    pub has_phone: bool,
    pub diagnostics: Diagnostics,
}

// ===================================================================
// Configuration helpers
// ===================================================================

/// `cfg` resolved for the Room Sim run `rs` (§9, §10.2): Room Sim's own
/// scene-free validation first (its error names its parameter, e.g.
/// `channel`), then `cfg`'s validation, then a set `phone_seed` that is
/// not the run's `seed` is refused naming `phone_seed` (§9; a plain-data
/// phone run's [`Config::resolved`] keeps it), then [`Config::resolved`]
/// with the run's `seed` and Room Sim's `tx_power_dbm` and `carrier_hz`:
/// an unset value is taken from Room Sim; a set one must equal it, else
/// the error names it.
pub fn resolve_config(rs: &RoomSimConfig, cfg: &Config) -> Result<Config, PipelineError> {
    rs.validate()?;
    cfg.validate()?;
    if let Some(mine) = cfg.phone_seed {
        if mine != rs.seed {
            return Err(phone::Error::Config {
                param: "phone_seed",
                reason: "a set phone_seed must be the Room Sim run's seed (§9; leave it unset)"
                    .to_string(),
            }
            .into());
        }
    }
    Ok(cfg.resolved(rs.seed, rs.tx_power_dbm, rs.carrier_hz())?)
}

/// The §9 defaults resolved for a Room Sim run: [`Config::default`] with
/// `phone_seed` = the run's seed and `tx_power_dbm`, `carrier_hz` = Room
/// Sim's ([`resolve_config`]). [`run_roomsim`] resolves an unset config
/// itself; this gives the values to callers that need them.
pub fn locate_config(rs: &RoomSimConfig) -> Result<Config, PipelineError> {
    resolve_config(rs, &Config::default())
}

/// The calibration run's Room Sim config (§10.2 step 1): the same scene,
/// config and **seed**, `n_people = 0`, `people_mode`, `people_posture` and
/// `start_heading_rad` emptied, and `duration_s` = the occupied run's
/// `duration_s + calib_s`. No seed arithmetic.
pub fn calibration_config(rs: &RoomSimConfig, calib_s: f64) -> RoomSimConfig {
    let mut c = rs.clone();
    c.n_people = Some(0);
    c.people_mode.clear();
    c.people_posture.clear();
    c.start_heading_rad.clear();
    c.duration_s = rs.duration_s + calib_s;
    c
}

/// The link signal named by §9's `link_signal`: `rssi_dbm` (Room Sim's
/// default, decision 2), `total_power_db` or `static_suppressed_amplitude`
/// (corner 0.03 Hz). Anything else is refused naming `link_signal`.
pub fn link_signal(name: &str) -> Result<Box<dyn LinkSignal>, PipelineError> {
    match name {
        "rssi_dbm" => Ok(Box::new(RssiSignal)),
        "total_power_db" => Ok(Box::new(TotalPowerSignal)),
        "static_suppressed_amplitude" => Ok(Box::new(StaticSuppressedSignal::default())),
        _ => Err(refused(
            "link_signal",
            "must be rssi_dbm, total_power_db or static_suppressed_amplitude",
        )),
    }
}

/// The Room Sim configuration refusals that need no scene (§2, §4.3, §9),
/// after Room Sim's own scene-free validation: any `roles` entry,
/// `quantize` false, and `tick_hz` above `1.0 / slot_s` (at most one tick
/// per slot). Room Sim's `tx_power_dbm`, `carrier_hz` and seed are
/// compared by [`resolve_config`].
pub fn check_roomsim_config(rs: &RoomSimConfig, cfg: &Config) -> Result<(), PipelineError> {
    rs.validate()?;
    if !rs.roles.is_empty() {
        return Err(refused(
            "roles",
            "the person pipeline needs every node txrx (the localize canonical link order equals Room Sim's only then); remove every roles entry",
        ));
    }
    if !rs.quantize {
        return Err(refused(
            "quantize",
            "must be true: the glue reads the int8 buffer (§4.3), which quantize false does not produce",
        ));
    }
    let most = 1.0 / rs.slot_s;
    if !(cfg.tick_hz <= most) {
        return Err(refused(
            "tick_hz",
            format!(
                "{} Hz is more than one tick per Room Sim slot ({most} Hz, 1.0 / slot_s)",
                cfg.tick_hz
            ),
        ));
    }
    Ok(())
}

/// Room Sim's base per-link rate `rate_hz = 1.0 / (float(nT) * slot_s)`
/// (§4.3).
pub fn base_rate_hz(sim: &Simulator) -> f64 {
    1.0 / (sim.transmitters().len() as f64 * sim.config().slot_s)
}

/// §4.1: refuse (naming `tx_period`) a run where some transmitter has
/// `floor(window_s * rate_hz / float(tx_period[tx])) + 1 < min_samples`
/// (NaN refuses): its links could never fill a window.
pub fn check_tx_period(
    sim: &Simulator,
    window_s: f64,
    rate_hz: f64,
    min_samples: i64,
) -> Result<(), PipelineError> {
    for &tx in sim.transmitters() {
        let period = sim
            .config()
            .tx_period
            .get(&u32::from(tx))
            .copied()
            .unwrap_or(1);
        let most = ((window_s * rate_hz) / (period as f64)).floor() + 1.0;
        if !(most >= min_samples as f64) {
            return Err(refused(
                "tx_period",
                format!(
                    "node {tx} transmits every {period} superframes: at most {most} samples per {window_s} s window, fewer than min_samples {min_samples}"
                ),
            ));
        }
    }
    Ok(())
}

/// The ticks of §10.1: `t_k = window_s + float(k) / tick_hz`, computed so
/// and never accumulated.
pub fn tick_time(window_s: f64, tick_hz: f64, k: u64) -> f64 {
    window_s + (k as f64) / tick_hz
}

// ===================================================================
// Geometry (SCENE, §2)
// ===================================================================

/// The person pipeline's localize geometry from the scene (§2):
/// `Room(width_m = room.w_m, height_m = room.d_m)`, `Node(node_id, x, y)`
/// (z dropped), `PixelGrid(room, pitch_m)`, `lambda_m`. Two nodes at one
/// `(x, y)` are refused naming `nodes` (the localize `Geometry` is 2-D).
pub fn person_geometry(scene: &Scene, cfg: &Config) -> Result<Geometry, PipelineError> {
    let nodes = &scene.nodes;
    for (i, a) in nodes.iter().enumerate() {
        for b in &nodes[i + 1..] {
            if a.pos.0 == b.pos.0 && a.pos.1 == b.pos.1 {
                return Err(refused(
                    "nodes",
                    format!(
                        "nodes {} and {} are at one (x, y); the person pipeline is 2-D and refuses stacked nodes",
                        a.id, b.id
                    ),
                ));
            }
        }
    }
    let room = Room::new(scene.room.w_m, scene.room.d_m)?;
    let grid = PixelGrid::new(room, cfg.pitch_m)?;
    let localize_nodes = nodes
        .iter()
        .map(|n| Node::new(i64::from(n.id), n.pos.0, n.pos.1))
        .collect();
    Ok(Geometry::new(
        room,
        localize_nodes,
        Some(grid),
        cfg.lambda_m,
    )?)
}

/// §5's plain link data from the ported geometry's links, canonical order.
pub fn link_geoms(geometry: &Geometry) -> Vec<LinkGeom> {
    geometry
        .links()
        .iter()
        .map(|l| LinkGeom {
            tx: l.tx,
            rx: l.rx,
            midpoint: l.midpoint(),
            length_m: l.length_m,
        })
        .collect()
}

// ===================================================================
// Glue: Room Sim packets to frames (§4.3)
// ===================================================================

/// §4.3: the `CsiFrame` of a **received** packet, or `None` for a lost
/// one. `amplitudes[k] = sqrt(float(re*re + im*im))` over the first
/// `fw_subcarriers` `(re, im)` entries of the int8 buffer (the datagram's
/// entries), `re`, `im` widened to `i32` before multiplying; `phases[k] =
/// 0.0`; `sample_index = seq`; `sample_rate_hz = rate_hz`; `rssi_dbm =
/// rssi`; `node_id = rx`. A float buffer (`quantize` false) is refused
/// naming `quantize`.
pub fn packet_frame(
    packet: &Packet,
    fw_subcarriers: usize,
    rate_hz: f64,
) -> Result<Option<CsiFrame>, PipelineError> {
    let Some(r) = &packet.received else {
        return Ok(None);
    };
    let Buffer::Int8(entries) = &r.buffer else {
        return Err(refused(
            "quantize",
            "a float buffer has no int8 entries (§4.3)",
        ));
    };
    let n = fw_subcarriers.min(entries.len());
    let amplitudes: Vec<f64> = entries[..n]
        .iter()
        .map(|&(re, im)| {
            let re = i32::from(re);
            let im = i32::from(im);
            f64::from(re * re + im * im).sqrt()
        })
        .collect();
    let phases = vec![0.0; amplitudes.len()];
    Ok(Some(new_csi_frame(
        amplitudes,
        phases,
        u64::from(r.seq),
        rate_hz,
        r.rssi,
        packet.rx,
    )?))
}

// ===================================================================
// The person pipeline (§4, §5, §10.1)
// ===================================================================

/// The labels a person estimate may carry (spec/locate.md §1). SCENE is
/// geometry, never an estimate.
fn person_label(label: &str) -> Result<&'static str, PipelineError> {
    match label {
        "SIM" => Ok("SIM"),
        "REAL" => Ok("REAL"),
        "SIM->REAL" => Ok("SIM->REAL"),
        _ => Err(refused(
            "label",
            "a person estimate is labelled SIM, REAL or SIM->REAL",
        )),
    }
}

/// The person pipeline of §4, §5 and §10.1: one `LinkMeasurement` (the
/// calibration and then the occupied run), WLM, and RTI (`Inverter` and
/// `Tracker`). Its label is the caller's (§1): `"SIM"`, `"REAL"` or
/// `"SIM->REAL"`; anything else (SCENE labels geometry, never an estimate)
/// is refused naming `label` without echoing it.
#[derive(Debug)]
pub struct PersonLocator {
    label: &'static str,
    geometry: Arc<Geometry>,
    links: Vec<LinkGeom>,
    measurement: LinkMeasurement,
    inverter: Inverter,
    tracker: Tracker,
    wlm: WlmParams,
    rti_radius_scale: f64,
    min_plausible_radius_m: f64,
    min_windows: i64,
}

impl PersonLocator {
    /// Build the measurement (§4.1: `window_s`, `sample_rate_hz` — the
    /// feed's per-link rate, Room Sim's `rate_hz` — the signal of
    /// `cfg.link_signal`, SUBTRACT, default `min_samples` and
    /// `clamp_negative`), the inverter (`alpha`, `sigma`, `delta_c`) and
    /// the tracker (`z_threshold`, the other parameters at the ported
    /// defaults). The ported constructors check their own parameters.
    pub fn new(
        geometry: impl Into<Arc<Geometry>>,
        cfg: &Config,
        sample_rate_hz: f64,
        label: &str,
    ) -> Result<Self, PipelineError> {
        let label = person_label(label)?;
        cfg.validate()?;
        let signal = link_signal(&cfg.link_signal)?;
        let geometry = geometry.into();
        let wlm = WlmParams::from_config(cfg, signal.name());
        let measurement = LinkMeasurement::new(
            geometry.clone(),
            MeasurementConfig {
                window_s: cfg.window_s,
                sample_rate_hz,
                calibration_mode: CalibrationMode::Subtract,
                min_samples: None,
                clamp_negative: false,
            },
            signal,
        )?;
        let inverter = Inverter::new(
            geometry.clone(),
            cfg.alpha,
            cfg.sigma,
            cfg.delta_c,
            InverterOptions::default(),
        )?;
        let tracker = Tracker::new(
            geometry.clone(),
            TrackerConfig {
                z_threshold: cfg.z_threshold,
                ..TrackerConfig::default()
            },
        )?;
        Ok(Self {
            label,
            links: link_geoms(&geometry),
            geometry,
            measurement,
            inverter,
            tracker,
            wlm,
            rti_radius_scale: cfg.rti_radius_scale,
            min_plausible_radius_m: cfg.min_plausible_radius_m,
            min_windows: i64::try_from(cfg.min_windows).unwrap_or(i64::MAX),
        })
    }

    /// The caller's label.
    pub fn label(&self) -> &'static str {
        self.label
    }

    pub fn geometry(&self) -> &Geometry {
        &self.geometry
    }

    pub fn measurement(&self) -> &LinkMeasurement {
        &self.measurement
    }

    pub fn inverter(&self) -> &Inverter {
        &self.inverter
    }

    pub fn tracker(&self) -> &Tracker {
        &self.tracker
    }

    /// The measurement's `min_samples`.
    pub fn min_samples(&self) -> i64 {
        self.measurement.min_samples
    }

    /// §4.2: `meas.calibrate_from(stream, min_windows)` on the empty-room
    /// run of the same room, the stream `(t, tx, rx, frame)` in Room Sim's
    /// order. If it raises, the run is refused naming `calib_s`.
    pub fn calibrate_from<I, F>(&mut self, stream: I) -> Result<(), PipelineError>
    where
        I: IntoIterator<Item = (f64, u8, u8, F)>,
        F: Borrow<CsiFrame>,
    {
        let stream = stream
            .into_iter()
            .map(|(t, tx, rx, f)| (t, i64::from(tx), i64::from(rx), f));
        self.measurement
            .calibrate_from(stream, self.min_windows)
            .map_err(|e| refused("calib_s", format!("the empty-room calibration raised {e}")))
    }

    /// `meas.update(t, tx, rx, frame)` (§4.3).
    pub fn update(
        &mut self,
        t: f64,
        tx: u8,
        rx: u8,
        frame: &CsiFrame,
    ) -> Result<(), PipelineError> {
        Ok(self
            .measurement
            .update(t, i64::from(tx), i64::from(rx), frame)?)
    }

    /// One §10.1 tick at `t_k`: `y = meas.vector(t_k)`; WLM (§5) from `y`
    /// and `null_sigma = meas.baseline_std_vector()`; RTI: `image =
    /// inverter.solve(y)`, `est = tracker.update(image, t_k)`, abstaining
    /// `not_significant` when `est` is None and `implausible_radius` when
    /// its radius (`rti_radius_scale * radius95(cov)`) fails §8 (the
    /// tracker keeps its corrected state). Returns `(wlm, rti)`.
    ///
    /// Before any calibration `y` is the raw windowed variance and
    /// `null_sigma` is zero (the ported behaviour), so WLM abstains.
    pub fn tick(&mut self, t_k: f64) -> Result<(Outcome, Outcome), PipelineError> {
        let y = self.measurement.vector(Some(t_k));
        let null_sigma = self.measurement.baseline_std_vector();
        let wlm_out = wlm(
            &self.links,
            self.geometry.lambda_m(),
            &y,
            &null_sigma,
            &self.wlm,
        )?;
        let image = self.inverter.solve(&y)?;
        let rti_out = match self.tracker.update(&image, t_k)? {
            None => Outcome::Abstain(Reason::NotSignificant),
            Some(est) => publish(
                est.x,
                est.y,
                est.cov_xx,
                est.cov_xy,
                est.cov_yy,
                self.rti_radius_scale,
                self.min_plausible_radius_m,
                Some(est.confidence),
            ),
        };
        Ok((wlm_out, rti_out))
    }
}

// ===================================================================
// The phone's inputs from Room Sim (§6.1, §6.2)
// ===================================================================

/// The scene's nodes for the phone, z kept, ids ascending (Room Sim's
/// order).
pub fn phone_nodes(sim: &Simulator) -> Vec<PhoneNode> {
    sim.room()
        .nodes
        .iter()
        .map(|n| PhoneNode {
            id: n.id,
            x: n.p.x,
            y: n.p.y,
            z: n.p.z,
        })
        .collect()
}

/// §6.2's static blockers: Room Sim's walls `W` (implied walls included),
/// centreline and height, and every object box with its `cs`, `sn` as
/// Room Sim computed them; doors and windows are flagged as openings,
/// which never block.
pub fn blockers(sim: &Simulator) -> Blockers {
    let room = sim.room();
    Blockers {
        walls: room
            .walls
            .iter()
            .map(|w| Wall {
                a: w.a,
                b: w.b,
                h: w.h,
            })
            .collect(),
        boxes: room
            .objects
            .iter()
            .map(|o| BoxObj {
                cx: o.c.x,
                cy: o.c.y,
                cz: o.c.z,
                sl: o.sl,
                sb: o.sb,
                sh: o.sh,
                cs: o.cs,
                sn: o.sn,
                opening: o.is_opening(),
            })
            .collect(),
    }
}

/// Every person's cylinder at one transmission, Room Sim §8.5: centre the
/// truth position, radius `body_radius_m`, top `height` (stand, walk),
/// `0.72 * height` (sit) or the bed top `+ 0.25` (lie), the bed being the
/// object the person lies on. People in index order.
pub fn cylinders(sim: &Simulator, truth: &Truth) -> Result<Vec<Cylinder>, PipelineError> {
    let objects = &sim.room().objects;
    let radius = sim.config().body_radius_m;
    sim.people()
        .iter()
        .zip(&truth.people)
        .map(|(model, pt)| {
            let bed = match pt.posture {
                Posture::Lie => Some(model.on_object.and_then(|o| objects.get(o)).ok_or_else(
                    || {
                        refused(
                            "people_posture",
                            format!("people[{}] lies on no object", model.index),
                        )
                    },
                )?),
                _ => None,
            };
            Ok(Cylinder {
                cx: pt.x,
                cy: pt.y,
                radius,
                top: cylinder_top(&model.body, pt.posture, bed),
            })
        })
        .collect()
}

/// One §10.1 tick record at `t_k` with the given truth.
fn tick_record(
    person: &mut PersonLocator,
    t_k: f64,
    truth: &[PersonTruthRecord],
) -> Result<TickRecord, PipelineError> {
    let (wlm_out, rti_out) = person.tick(t_k)?;
    Ok(TickRecord {
        label: person.label(),
        t: t_k,
        wlm: wlm_out,
        rti: rti_out,
        truth: truth.to_vec(),
    })
}

/// §10.1: `t_{k+1} = tick_time(window_s, tick_hz, k + 1)`, computed from
/// `k + 1` (never accumulated), where `t_k` is tick `k`'s time. A tick
/// that does not advance (`not (t_{k+1} > t_k)`: `float(k) / tick_hz` no
/// longer moves `window_s`) refuses the run naming `tick_hz`: the ticks
/// would never reach `duration_s`.
pub fn next_tick(window_s: f64, tick_hz: f64, k: u64, t_k: f64) -> Result<f64, PipelineError> {
    let next = tick_time(window_s, tick_hz, k + 1);
    if !(next > t_k) {
        return Err(refused(
            "tick_hz",
            format!("tick {} does not advance past t = {t_k} s", k + 1),
        ));
    }
    Ok(next)
}

fn truth_records(truth: &Truth) -> Vec<PersonTruthRecord> {
    truth
        .people
        .iter()
        .map(|p| PersonTruthRecord {
            x: p.x,
            y: p.y,
            walking: p.walking,
            posture: p.posture,
        })
        .collect()
}

// ===================================================================
// The Room Sim run (§10.2)
// ===================================================================

/// Run Locate on one Room Sim run (§10.2): the calibration run, the
/// occupied run with one tick record per §10.1 tick, and the phone with
/// one epoch record per fired epoch.
///
/// `rs` is the occupied run's Room Sim config as given; `cfg` is Locate's
/// (§9): its unset `phone_seed`, `tx_power_dbm` and `carrier_hz` are taken
/// from `rs` ([`resolve_config`]), so `Config::default()` is the default.
/// `label` must be `"SIM"` (Room Sim input is simulated). Refusals name
/// the parameter: `label`, any §9 parameter (`link_signal`, `min_windows`
/// …), `phone_seed`, `tx_power_dbm` or `carrier_hz` that are not the
/// run's, `tick_hz` (above `1.0 / slot_s`, or ticks that stop advancing),
/// `roles`, `quantize`, Room Sim's own (scene and config), `nodes`
/// (stacked), `tx_period`, `calib_s`.
pub fn run_roomsim(
    scene: &Scene,
    rs: &RoomSimConfig,
    cfg: &Config,
    label: &str,
) -> Result<RoomSimRun, PipelineError> {
    if label != LABEL {
        return Err(PipelineError::Label);
    }
    cfg.validate()?;
    check_roomsim_config(rs, cfg)?;
    let cfg = &resolve_config(rs, cfg)?;
    let sim = Simulator::new(scene, rs.clone())?;
    let geometry = person_geometry(scene, cfg)?;
    let rate_hz = base_rate_hz(&sim);
    let mut person = PersonLocator::new(geometry, cfg, rate_hz, label)?;
    check_tx_period(&sim, cfg.window_s, rate_hz, person.min_samples())?;
    let fw = rs.fw_subcarriers;

    // 1. Calibration: the same scene, config and seed, nobody in the room,
    //    duration_s + calib_s; only its packets with t >= duration_s.
    let calib_sim = Simulator::new(scene, calibration_config(rs, cfg.calib_s))?;
    let mut glue_error: Option<PipelineError> = None;
    let mut calib_packets = 0usize;
    let calibrated = {
        let stream = calib_sim
            .packets()
            .filter(|p| p.t >= rs.duration_s)
            .filter_map(|p| match packet_frame(&p, fw, rate_hz) {
                Ok(Some(frame)) => {
                    calib_packets += 1;
                    Some((p.t, p.tx, p.rx, frame))
                }
                Ok(None) => None,
                Err(e) => {
                    glue_error.get_or_insert(e);
                    None
                }
            });
        person.calibrate_from(stream)
    };
    if let Some(e) = glue_error {
        return Err(e);
    }
    calibrated?;
    let calibration = CalibrationSummary {
        packets: calib_packets,
        baseline: person.measurement().baseline_vector(),
        null_sigma: person.measurement().baseline_std_vector(),
        calibrated_mask: person.measurement().calibrated_mask(),
    };

    // 3. The phone (when the scene has one), SIM only.
    let mut phone_run = match scene.phone {
        Some(_) => Some(PhoneRun::new(
            cfg,
            &phone_nodes(&sim),
            &blockers(&sim),
            label,
        )?),
        None => None,
    };

    // 2. The occupied run: packets in order; tick k once every packet with
    //    t <= t_k is in.
    let duration_s = rs.duration_s;
    let mut ticks = Vec::new();
    let mut epochs = Vec::new();
    let mut k: u64 = 0;
    let mut t_k = tick_time(cfg.window_s, cfg.tick_hz, k);
    let mut truth_now: Vec<PersonTruthRecord> = Vec::new();
    let mut packets = 0usize;
    for tr in sim.run() {
        while t_k < duration_s && t_k < tr.t {
            ticks.push(tick_record(&mut person, t_k, &truth_now)?);
            t_k = next_tick(cfg.window_s, cfg.tick_hz, k, t_k)?;
            k += 1;
        }
        truth_now = truth_records(&tr.truth);
        if let (Some(run), Some(p)) = (phone_run.as_mut(), tr.truth.phone) {
            let cyl = cylinders(&sim, &tr.truth)?;
            let sample = TruthSample {
                t: tr.t,
                phone: [p.x, p.y, p.z],
                cylinders: &cyl,
            };
            if let Some(rec) = run.step(&sample)? {
                epochs.push(rec);
            }
        }
        for p in &tr.packets {
            if let Some(frame) = packet_frame(p, fw, rate_hz)? {
                person.update(p.t, p.tx, p.rx, &frame)?;
                packets += 1;
            }
        }
    }
    while t_k < duration_s {
        ticks.push(tick_record(&mut person, t_k, &truth_now)?);
        t_k = next_tick(cfg.window_s, cfg.tick_hz, k, t_k)?;
        k += 1;
    }

    let m = person.measurement();
    let diagnostics = Diagnostics {
        packets,
        clock_restarts: m.clock_restarts,
        stale_samples: m.stale_samples,
        window_overflows: m.window_overflows,
        blind_pixels: person.inverter().blind_pixels.len(),
        n_independent_links: person.inverter().n_independent_links,
    };
    Ok(RoomSimRun {
        label: LABEL,
        rate_hz,
        min_samples: person.min_samples(),
        links: person
            .geometry()
            .links()
            .iter()
            .map(|l| (l.tx_id, l.rx_id))
            .collect(),
        calibration,
        ticks,
        epochs,
        has_phone: scene.phone.is_some(),
        diagnostics,
    })
}
