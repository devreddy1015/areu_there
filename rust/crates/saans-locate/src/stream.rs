//! The streaming driver of a Room Sim run (spec/livedemo.md §8.1). SIM.
//!
//! Room Sim's `Run<'a>` borrows its `Simulator`, so one owning type would
//! be self-referential. The driver is two types instead:
//! - a [`RoomSimSetup`] makes `run_roomsim`'s refusals, owns the occupied
//!   run's `Simulator` and simulates the calibration run once
//!   (spec/locate.md §10.2 step 1), keeping the frames it fed to
//!   `calibrate_from`;
//! - a [`RoomSimStream`] borrows a setup and runs the occupied run and the
//!   phone from `t = 0` (§10.2 steps 2 and 3), advanced to any `t_end` in
//!   any steps. A setup gives any number of streams; a restart is a new
//!   one.
//!
//! [`run_roomsim`](crate::pipeline::run_roomsim) is one setup and one
//! stream advanced to the end, so a stream gives its records, record for
//! record, whatever its steps.
//!
//! No `Person.id` or `Phone.id` is kept: a setup's scene is a copy with
//! both cleared (Room Sim never reads them), and neither type prints its
//! scene, frames or simulator (ADR-008 decision 3).

// NaN fails closed: deciding comparisons are written `!(a >= b)` on purpose.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::iter::Peekable;
use std::sync::Arc;

use saans_roomsim::{Config as RoomSimConfig, Run, Simulator};
use saans_scene::Scene;

use crate::baseline::{disturbed, WlmParams};
use crate::geometry::Geometry;
use crate::measure::CsiFrame;
use crate::phone::{Config, EpochRecord, PhoneRun, TruthSample};
use crate::pipeline::{
    base_rate_hz, blockers, calibration_config, check_roomsim_config, check_tx_period, cylinders,
    next_tick, packet_frame, person_geometry, phone_nodes, resolve_config, tick_time,
    truth_records, CalibrationSummary, Diagnostics, PersonLocator, PersonTruthRecord,
    PipelineError, TickRecord,
};
use crate::LABEL;

/// One link's values at one tick (spec/livedemo.md §4 `links.items`), in
/// the localize canonical link order: the very `y` and `null_sigma` that
/// tick's WLM used.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkValue {
    /// `meas.vector(t_k)[l]`, dB² for `rssi_dbm`; may be non-finite (sent
    /// as `null`).
    pub y: f64,
    /// `wlm_gate_z * null_sigma + wlm_floor` when `null_sigma > 0`, else
    /// `None` (such a link can never be disturbed).
    pub gate: Option<f64>,
    /// [`crate::baseline::disturbed`]`(y, null_sigma, params)`: WLM's own
    /// gate test (spec/locate.md §5 step 1).
    pub disturbed: bool,
}

/// One link's [`LinkValue`] from the `y` and `null_sigma` its tick's WLM
/// used, with that WLM's parameters.
fn link_value(y: f64, null_sigma: f64, params: &WlmParams) -> LinkValue {
    LinkValue {
        y,
        gate: (null_sigma > 0.0).then_some(params.gate_z * null_sigma + params.floor),
        disturbed: disturbed(y, null_sigma, params),
    }
}

// ===================================================================
// The setup
// ===================================================================

/// One calibration frame as fed to `calibrate_from`: `(t, tx, rx, frame)`.
type CalibrationFrame = (f64, u8, u8, CsiFrame);

/// The calibration a setup shares with its empty room: the calibration
/// run is the same for every variant (spec/locate.md §10.2 step 1).
struct Calibration {
    /// The received calibration-run packets with `t >= duration_s`, after
    /// the §4.3 glue, in Room Sim's order.
    frames: Vec<CalibrationFrame>,
    summary: CalibrationSummary,
}

/// What `run_roomsim` builds before its calibration.
struct Prepared {
    /// Resolved for the run (§9).
    cfg: Config,
    sim: Simulator,
    geometry: Arc<Geometry>,
    rate_hz: f64,
    person: PersonLocator,
}

/// `run_roomsim`'s refusals before its calibration, in its order: `label`,
/// `cfg`'s §9 validation, the scene-free Room Sim checks (Room Sim's own
/// config validation, then `roles`, `quantize`, `tick_hz`), the resolved
/// config (`phone_seed`, `tx_power_dbm`, `carrier_hz`), Room Sim's scene
/// checks, `nodes`, the locator's (`link_signal` …) and `tx_period`.
fn prepare(
    scene: &Scene,
    rs: &RoomSimConfig,
    cfg: &Config,
    label: &str,
) -> Result<Prepared, PipelineError> {
    if label != LABEL {
        return Err(PipelineError::Label);
    }
    cfg.validate()?;
    check_roomsim_config(rs, cfg)?;
    let cfg = resolve_config(rs, cfg)?;
    let sim = Simulator::new(scene, rs.clone())?;
    let geometry = Arc::new(person_geometry(scene, &cfg)?);
    let rate_hz = base_rate_hz(&sim);
    let person = PersonLocator::new(Arc::clone(&geometry), &cfg, rate_hz, label)?;
    check_tx_period(&sim, cfg.window_s, rate_hz, person.min_samples())?;
    Ok(Prepared {
        cfg,
        sim,
        geometry,
        rate_hz,
        person,
    })
}

/// The phone's run when the scene has a phone (§10.2 step 3), SIM only;
/// never stepped, so each stream starts from a copy of it.
fn phone_run(
    scene: &Scene,
    sim: &Simulator,
    cfg: &Config,
) -> Result<Option<PhoneRun>, PipelineError> {
    Ok(match scene.phone {
        Some(_) => Some(PhoneRun::new(
            cfg,
            &phone_nodes(sim),
            &blockers(sim),
            LABEL,
        )?),
        None => None,
    })
}

/// `scene` with every `Person.id` and the `Phone.id` cleared. Room Sim
/// never reads them, so the setup keeps no id (ADR-008 decision 3).
fn without_ids(scene: &Scene) -> Scene {
    let mut s = scene.clone();
    for p in &mut s.people {
        p.id = None;
    }
    if let Some(phone) = &mut s.phone {
        phone.id = None;
    }
    s
}

/// The `empty` variant of the run `rs` (spec/livedemo.md §3.1): nobody in
/// the room (`n_people = 0`, the per-person maps emptied as
/// [`calibration_config`] empties them) and `phone_carrier` at Room Sim's
/// default. With nobody in the room no one carries the phone (it stays at
/// its scene start, z = 1.0, Room Sim §9), so the carrier changes no
/// packet and no truth; resetting it gives what `Variant::Empty` gives
/// from the variants' common base. The scene, seed, duration and every
/// other parameter are `rs`'s, so the calibration run is `rs`'s.
fn empty_config(rs: &RoomSimConfig) -> RoomSimConfig {
    let mut c = rs.clone();
    c.n_people = Some(0);
    c.people_mode.clear();
    c.people_posture.clear();
    c.start_heading_rad.clear();
    c.phone_carrier = RoomSimConfig::default().phone_carrier;
    c
}

/// A Room Sim run through Locate, set up and calibrated, ready for any
/// number of [`RoomSimStream`]s (spec/livedemo.md §8.1). SIM.
///
/// It owns the occupied run's `Simulator`, the frames the calibration fed
/// to `calibrate_from` (shared with its [`RoomSimSetup::empty_room`]) and
/// the [`CalibrationSummary`]. `Debug` shows counts and parameters only.
pub struct RoomSimSetup {
    /// The caller's scene with its ids cleared ([`without_ids`]), for
    /// [`RoomSimSetup::empty_room`].
    scene: Scene,
    rs: RoomSimConfig,
    /// Resolved for `rs` (§9).
    cfg: Config,
    sim: Simulator,
    geometry: Arc<Geometry>,
    rate_hz: f64,
    min_samples: i64,
    links: Vec<(i64, i64)>,
    phone: Option<PhoneRun>,
    calibration: Arc<Calibration>,
}

impl RoomSimSetup {
    /// Make [`run_roomsim`](crate::pipeline::run_roomsim)'s refusals, in
    /// its order and naming the same parameters; build the occupied run's
    /// `Simulator`; simulate the calibration run into `calibrate_from`
    /// (spec/locate.md §10.2 step 1), keeping the frames it fed (those with
    /// `t >= duration_s`) and the [`CalibrationSummary`].
    ///
    /// `keep_going` is called before each calibration packet is drawn,
    /// whether or not its `t >= duration_s`, so a stop is noticed within
    /// one transmission (spec/livedemo.md §3.0), and once more before the
    /// run is found finished. Its first `false` abandons the calibration at
    /// once: `Ok(None)`. A refusal before the calibration is returned
    /// without calling it.
    pub fn new(
        scene: &Scene,
        rs: &RoomSimConfig,
        cfg: &Config,
        label: &str,
        keep_going: &mut dyn FnMut() -> bool,
    ) -> Result<Option<RoomSimSetup>, PipelineError> {
        let Prepared {
            cfg,
            sim,
            geometry,
            rate_hz,
            mut person,
        } = prepare(scene, rs, cfg, label)?;
        let fw = rs.fw_subcarriers;

        // The same scene, config and seed, nobody in the room,
        // duration_s + calib_s; only its packets with t >= duration_s, fed
        // lazily as run_roomsim fed them, so a glue error among the packets
        // fed wins over the calibration's own.
        let calib_sim = Simulator::new(scene, calibration_config(rs, cfg.calib_s))?;
        let mut frames: Vec<CalibrationFrame> = Vec::new();
        let mut glue_error: Option<PipelineError> = None;
        let mut abandoned = false;
        let calibrated = {
            let mut packets = calib_sim.packets();
            let stream = std::iter::from_fn(|| {
                while !abandoned {
                    if !keep_going() {
                        abandoned = true;
                        break;
                    }
                    let p = packets.next()?;
                    if p.t >= rs.duration_s {
                        match packet_frame(&p, fw, rate_hz) {
                            Ok(Some(frame)) => {
                                frames.push((p.t, p.tx, p.rx, frame.clone()));
                                return Some((p.t, p.tx, p.rx, frame));
                            }
                            Ok(None) => {}
                            Err(e) => {
                                glue_error.get_or_insert(e);
                            }
                        }
                    }
                }
                None
            });
            person.calibrate_from(stream)
        };
        if abandoned {
            return Ok(None);
        }
        if let Some(e) = glue_error {
            return Err(e);
        }
        calibrated?;
        drop(calib_sim);
        frames.shrink_to_fit();
        let m = person.measurement();
        let summary = CalibrationSummary {
            packets: frames.len(),
            baseline: m.baseline_vector(),
            null_sigma: m.baseline_std_vector(),
            calibrated_mask: m.calibrated_mask(),
        };

        let phone = phone_run(scene, &sim, &cfg)?;
        Ok(Some(RoomSimSetup {
            scene: without_ids(scene),
            rs: rs.clone(),
            min_samples: person.min_samples(),
            links: canonical_links(&person),
            cfg,
            sim,
            geometry,
            rate_hz,
            phone,
            calibration: Arc::new(Calibration { frames, summary }),
        }))
    }

    /// A fresh run from `t = 0`: a new [`PersonLocator`] calibrated from
    /// the kept frames (`calibrate_from` is deterministic, so it starts in
    /// the state `run_roomsim`'s occupied run starts in), a new phone run
    /// and a new `Run` of the setup's `Simulator`.
    pub fn stream(&self) -> Result<RoomSimStream<'_>, PipelineError> {
        let mut person =
            PersonLocator::new(Arc::clone(&self.geometry), &self.cfg, self.rate_hz, LABEL)?;
        person.calibrate_from(
            self.calibration
                .frames
                .iter()
                .map(|(t, tx, rx, frame)| (*t, *tx, *rx, frame)),
        )?;
        Ok(RoomSimStream {
            setup: self,
            run: self.sim.run().peekable(),
            person,
            phone: self.phone.clone(),
            k: 0,
            t_k: tick_time(self.cfg.window_s, self.cfg.tick_hz, 0),
            truth_now: Vec::new(),
            packets: 0,
            t: f64::NEG_INFINITY,
            exhausted: false,
            failed: None,
        })
    }

    /// The setup of the `empty` variant of the same scene, seed and
    /// duration (spec/livedemo.md §3.1, §5.3): this setup's Room Sim config
    /// with `n_people = 0`, the per-person maps emptied and `phone_carrier`
    /// at Room Sim's default, and the same Locate config. Its calibration run is this one's, so it shares
    /// this setup's calibration frames and summary and simulates no
    /// calibration. Its refusals are `run_roomsim`'s before the
    /// calibration.
    pub fn empty_room(&self) -> Result<RoomSimSetup, PipelineError> {
        let rs = empty_config(&self.rs);
        let p = prepare(&self.scene, &rs, &self.cfg, LABEL)?;
        let phone = phone_run(&self.scene, &p.sim, &p.cfg)?;
        Ok(RoomSimSetup {
            scene: self.scene.clone(),
            rs,
            min_samples: p.person.min_samples(),
            links: canonical_links(&p.person),
            cfg: p.cfg,
            sim: p.sim,
            geometry: p.geometry,
            rate_hz: p.rate_hz,
            phone,
            calibration: Arc::clone(&self.calibration),
        })
    }

    /// Always `"SIM"`.
    pub fn label(&self) -> &'static str {
        LABEL
    }

    /// Locate's config resolved for the run (§9: `phone_seed`,
    /// `tx_power_dbm` and `carrier_hz` from Room Sim).
    pub fn config(&self) -> &Config {
        &self.cfg
    }

    /// The occupied run's Room Sim config as given.
    pub fn roomsim_config(&self) -> &RoomSimConfig {
        &self.rs
    }

    /// The localize canonical link order `(tx, rx)` (§2), the order of
    /// every tick's [`LinkValue`]s.
    pub fn links(&self) -> &[(i64, i64)] {
        &self.links
    }

    /// Whether the scene has a phone (then streams emit epochs).
    pub fn has_phone(&self) -> bool {
        self.phone.is_some()
    }

    /// The run's length, SIM seconds: no tick or transmission at or after.
    pub fn duration_s(&self) -> f64 {
        self.rs.duration_s
    }

    /// Ticks per second (§10.1).
    pub fn tick_hz(&self) -> f64 {
        self.cfg.tick_hz
    }

    /// The variance window, s; the first tick is at `t = window_s`.
    pub fn window_s(&self) -> f64 {
        self.cfg.window_s
    }

    /// Room Sim's base per-link rate `1.0 / (float(nT) * slot_s)` (§4.3).
    pub fn rate_hz(&self) -> f64 {
        self.rate_hz
    }

    /// The measurement's `min_samples` (§4.1).
    pub fn min_samples(&self) -> i64 {
        self.min_samples
    }

    /// What the empty-room calibration produced (§4.2, §10.2 step 1).
    pub fn calibration(&self) -> &CalibrationSummary {
        &self.calibration.summary
    }
}

impl std::fmt::Debug for RoomSimSetup {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoomSimSetup")
            .field("label", &LABEL)
            .field("seed", &self.rs.seed)
            .field("duration_s", &self.rs.duration_s)
            .field("links", &self.links.len())
            .field("has_phone", &self.has_phone())
            .field("calibration_frames", &self.calibration.frames.len())
            .finish_non_exhaustive()
    }
}

/// The locator geometry's links `(tx, rx)`, canonical order.
fn canonical_links(person: &PersonLocator) -> Vec<(i64, i64)> {
    person
        .geometry()
        .links()
        .iter()
        .map(|l| (l.tx_id, l.rx_id))
        .collect()
}

// ===================================================================
// The stream
// ===================================================================

/// What one [`RoomSimStream::advance_to`] emitted, each list in time
/// order. Ticks and epochs interleave as in `run_roomsim`: a tick at `t_k`
/// comes after every epoch with `t <= t_k` (at equal `t` the epoch comes
/// first) and before every later one.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Advance {
    /// One per §10.1 tick (spec/locate.md §11).
    pub ticks: Vec<TickRecord>,
    /// One per entry of `ticks`, at the same index: every link's value at
    /// that tick, in [`RoomSimSetup::links`] order.
    pub links: Vec<Vec<LinkValue>>,
    /// One per fired phone epoch; empty when the scene has no phone.
    pub epochs: Vec<EpochRecord>,
}

/// One run of a [`RoomSimSetup`] from `t = 0`, advanced by
/// [`RoomSimStream::advance_to`] (spec/livedemo.md §8.1). SIM.
pub struct RoomSimStream<'s> {
    setup: &'s RoomSimSetup,
    run: Peekable<Run<'s>>,
    person: PersonLocator,
    phone: Option<PhoneRun>,
    /// The next tick's index and time (§10.1).
    k: u64,
    t_k: f64,
    /// The people of the last transmission consumed (§11 tick truth).
    truth_now: Vec<PersonTruthRecord>,
    /// Received packets fed to the measurement.
    packets: usize,
    /// The last `t_end`.
    t: f64,
    /// Every transmission is consumed.
    exhausted: bool,
    /// The error that ended the stream.
    failed: Option<PipelineError>,
}

impl<'s> RoomSimStream<'s> {
    /// Advance to `t_end`, SIM seconds: consume every transmission with
    /// `t <= t_end` (peeking one ahead, so none later is consumed), and
    /// emit every tick with `t_k <= t_end` and `t_k < duration_s` and the
    /// epochs fired, in `run_roomsim`'s order. A tick is emitted once every
    /// packet with `t <= t_k` is in (§10.1). Once the transmissions are
    /// exhausted the remaining ticks before `duration_s` are emitted too,
    /// each as `t_end` reaches it (a frame at `T` holds only the records
    /// with `t <= T`, spec/livedemo.md §4), so advancing to `duration_s` or
    /// beyond (`f64::INFINITY`) emits all that is left.
    ///
    /// The records do not depend on the steps: any sequence of `t_end`s
    /// gives `run_roomsim`'s. An equal `t_end` emits nothing; a NaN one, or
    /// one before the last, is refused naming `t_end` and changes nothing.
    /// A pipeline error ends the stream: this call and every later one
    /// return it.
    pub fn advance_to(&mut self, t_end: f64) -> Result<Advance, PipelineError> {
        if let Some(e) = &self.failed {
            return Err(e.clone());
        }
        if !(t_end >= self.t) {
            return Err(PipelineError::Refused {
                param: "t_end",
                reason: "is NaN or before the last t_end (a stream only goes forward)".to_string(),
            });
        }
        let mut out = Advance::default();
        match self.step(t_end, &mut out) {
            Ok(()) => {
                self.t = t_end;
                Ok(out)
            }
            Err(e) => {
                self.failed = Some(e.clone());
                Err(e)
            }
        }
    }

    fn step(&mut self, t_end: f64, out: &mut Advance) -> Result<(), PipelineError> {
        let setup = self.setup;
        let duration_s = setup.rs.duration_s;
        let fw = setup.rs.fw_subcarriers;
        while let Some(tr) = self.run.next_if(|tr| tr.t <= t_end) {
            while self.t_k < duration_s && self.t_k < tr.t {
                self.tick(out)?;
            }
            self.truth_now = truth_records(&tr.truth);
            if let (Some(run), Some(p)) = (self.phone.as_mut(), tr.truth.phone) {
                let cyl = cylinders(&setup.sim, &tr.truth)?;
                let sample = TruthSample {
                    t: tr.t,
                    phone: [p.x, p.y, p.z],
                    cylinders: &cyl,
                };
                if let Some(rec) = run.step(&sample)? {
                    out.epochs.push(rec);
                }
            }
            for p in &tr.packets {
                if let Some(frame) = packet_frame(p, fw, setup.rate_hz)? {
                    self.person.update(p.t, p.tx, p.rx, &frame)?;
                    self.packets += 1;
                }
            }
        }
        self.exhausted = self.run.peek().is_none();
        while self.t_k < duration_s && self.t_k <= t_end {
            self.tick(out)?;
        }
        Ok(())
    }

    /// One §10.1 tick at `t_k` with the current truth, then the next
    /// `t_k`.
    fn tick(&mut self, out: &mut Advance) -> Result<(), PipelineError> {
        let (wlm, rti, y, null_sigma) = self.person.tick_values(self.t_k)?;
        let params = self.person.wlm_params();
        out.links.push(
            y.iter()
                .zip(&null_sigma)
                .map(|(&y, &s)| link_value(y, s, params))
                .collect(),
        );
        out.ticks.push(TickRecord {
            label: self.person.label(),
            t: self.t_k,
            wlm,
            rti,
            truth: self.truth_now.clone(),
        });
        let cfg = &self.setup.cfg;
        self.t_k = next_tick(cfg.window_s, cfg.tick_hz, self.k, self.t_k)?;
        self.k += 1;
        Ok(())
    }

    /// Whether the run is over: every transmission consumed and every tick
    /// before `duration_s` emitted, so a later `advance_to` emits nothing.
    /// False for a stream an error ended.
    pub fn finished(&self) -> bool {
        self.failed.is_none() && self.exhausted && !(self.t_k < self.setup.rs.duration_s)
    }

    /// The last `t_end` advanced to, SIM seconds; `-inf` before the first.
    pub fn t(&self) -> f64 {
        self.t
    }

    /// The run's counters so far (SIM).
    pub fn diagnostics(&self) -> Diagnostics {
        let m = self.person.measurement();
        let inverter = self.person.inverter();
        Diagnostics {
            packets: self.packets,
            clock_restarts: m.clock_restarts,
            stale_samples: m.stale_samples,
            window_overflows: m.window_overflows,
            blind_pixels: inverter.blind_pixels.len(),
            n_independent_links: inverter.n_independent_links,
        }
    }

    /// The setup this stream runs.
    pub fn setup(&self) -> &'s RoomSimSetup {
        self.setup
    }
}

impl std::fmt::Debug for RoomSimStream<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RoomSimStream")
            .field("t", &self.t)
            .field("next_tick", &self.t_k)
            .field("packets", &self.packets)
            .field("finished", &self.finished())
            .finish_non_exhaustive()
    }
}
