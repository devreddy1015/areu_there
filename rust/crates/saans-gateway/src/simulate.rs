//! Simulate mode: a synthetic room driven through the real pipeline.
//!
//! ```text
//! saans-sim (physics) ─► FeatureExtractor (the node's own DSP) ─► FeatureFrame
//!   ─► Hub::ingest ─► SpaceFusion ─► SpaceMonitor ─► the same snapshots
//!      live mode publishes
//! ```
//!
//! **Everything here is SYNTHETIC.** No measurement of a real room, a
//! real body or a real ESP32 enters this module, and `shared.mode` is
//! `"simulated"` so every reader of the API is told so.
//!
//! ## What the mode is for
//!
//! It exists to show the *gap* between what is true and what a
//! single-antenna system can recover. The simulator holds the ground
//! truth — where the person is, whether they are breathing — and the
//! HTTP side can move them ([`SimState::set_person`]). The pipeline sees
//! only CSI and reports what it can. Park the person in a Fresnel null
//! and the pipeline reports an empty room while the truth still has them
//! seated and breathing in it ([`DEMO_NULL_M`], asserted by
//! `the_gap_the_explorer_exists_to_show`). That is the product argument,
//! and `docs/CAPABILITY_LIMITS.md` is why it cannot be argued away.
//!
//! The person's position is an **input**, never an inference. Nothing in
//! the gateway recovers a position from CSI, and a 1T1R link cannot.
//!
//! ## How it hangs together
//!
//! - [`SimState`] owns the simulator behind a short-held mutex. The sim
//!   task locks it to step one tick; HTTP handlers lock it to read the
//!   truth or move the person. Neither holds it across an await.
//! - Coverage maps come from a second, never-stepped simulator cloned
//!   from the first (`SimState::maps`): sensitivity is a pure function
//!   of the scenario's geometry, so the clone's answers are the live
//!   simulator's answers, and an expensive map never stalls a sim tick.
//! - Frames reach the hub through [`Hub::ingest`] as encoded
//!   `FeatureFrame` bytes — the same decode, routing, fusion and
//!   detectors a UDP datagram goes through. Nothing is written into the
//!   published state directly.
//! - Like [`crate::replay`], simulate mode never reads or writes
//!   `data_dir`: synthetic events do not belong in a real deployment's
//!   event log, and the audit log is off for the same reason.

use crate::config::GatewayConfig;
use crate::engine::TICK_S;
use crate::hub::{local_epoch_s, Hub, Wall};
use crate::shared::Shared;
use crate::timefmt::now_unix_ms;
use saans_dsp::features::{FeatureExtractor, FeatureExtractorConfig};
use saans_health::{LinkId, SpaceConfig};
use saans_sim::scenario::{Activity, Node, Point, Room, Scenario, Segment, Truth};
use saans_sim::{scenarios, Simulator, Tick};
use serde_json::{json, Value};
use std::io;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::net::TcpListener;
use tokio::sync::watch;
use tokio::task::JoinHandle;

/// The one scenario `--simulate` offers (hence no `--sim-scenario`).
/// Default cell size for `/api/sim/coverage`.
///
/// The sensitivity field oscillates with a spatial *period* of half a
/// wavelength — 6.2 cm at 2.437 GHz. The grid therefore has to be finer
/// than **half of that**, not finer than 6.2 cm: sampling a 6.2 cm
/// period at 5 cm is barely one sample per cycle, and aliases.
///
/// This was 0.05 on the reasoning that 5 cm "samples the fringes". It
/// does not. Measured against the two positions `the_gap_the_explorer
/// _exists_to_show` pins — (3.00, 3.61), where the pipeline loses the
/// person completely, and (3.00, 3.10), where it sees them at
/// confidence > 0.9:
///
/// | res_m | cells  | at the null         | at the seat      |
/// |-------|--------|---------------------|------------------|
/// | 0.25  |    480 | 0.79 below median   | 1.33 around      |
/// | 0.10  |  3,000 | 1.26 around median  | 0.29 very weak   |
/// | 0.05  | 12,000 | 0.65 below median   | 1.19 around      |
/// | 0.04  | 18,750 | 0.45 very weak      | 1.24 around      |
/// | 0.03  | 33,400 | 0.21 **blind**      | 1.26 around      |
/// | 0.02  | 75,000 | 0.09 blind          | 1.13 around      |
///
/// Only at 0.03 and below does the map call the null what the pipeline
/// calls it. At 0.05 it reads "below this room's median", which a
/// reader takes as *a bit weak* for a cell where the system cannot see
/// at all; at 0.10 it is wrong in the opposite direction at both
/// points. A map that disagrees with the pipeline it is drawn to
/// explain is worse than no map.
///
/// 0.03 m is also where Nyquist lands — two samples per 6.2 cm period —
/// so the measurement and the theory agree, which is the only reason to
/// trust either. 200x167 = 33,400 cells in the 6x5 m explorer room,
/// inside saans-sim's 100,000 cap and still sub-second to compute.
pub const DEFAULT_COVERAGE_RES_M: f64 = 0.03;

pub const SCENARIO: &str = "explorer";
/// The space the simulated room is published as.
pub const SPACE_ID: &str = "sim-room";

/// Seeds the room's furniture scatterers and every noise stream.
const SEED: u64 = 7;
/// Empty room before the person appears. Not padding: the presence
/// detector calibrates on an empty room for `calibration_s` (30 s) and
/// the breathing filters need ~50 s to settle, so a timeline that starts
/// with a person in it would learn that person as the baseline and never
/// report them. `scenarios::LEAD_IN_S` is the evaluation's own figure.
const LEAD_IN_S: f64 = scenarios::LEAD_IN_S;
/// How long the person then sits there. The explorer is driven by hand
/// after that, so this only has to outlast any demo.
const SEATED_S: f64 = 24.0 * 3600.0;
/// Resting breathing rate of the simulated person.
const BPM: f64 = 15.0;

/// Where the person appears, and sits until someone drags them: 0.6 m
/// off the line of sight, where the whole band is strong (every
/// subcarrier's sensitivity is within 2% of the band maximum), so the
/// explorer opens in the state it exists to contrast with.
const START_M: Point = (3.0, 3.1);

/// A measured whole-band null, 1.1 m off the line of sight and 2.5 m
/// from either node: park the person here and the pipeline reports an
/// empty room while the truth panel still shows them sitting in it.
/// Half a metre away in x it sees them again with full confidence.
/// `the_gap_the_explorer_exists_to_show` asserts both, so this constant
/// cannot rot into a coordinate that no longer demonstrates anything.
pub const DEMO_NULL_M: Point = (3.0, 3.61);

/// The room: 6 m x 5 m, clear line of sight, order-1 wall reflections,
/// six seeded furniture scatterers — the evaluation's "clear-LoS" room.
/// The door is drawn on the wall for the plan of the room; the person is
/// placed by the timeline rather than walked in through it.
const DOOR_M: Point = (0.5, 0.0);

/// Two nodes on opposite walls, 4.8 m apart across the long axis, and
/// the one link between them.
///
/// **Why one link, measured rather than assumed.** The explorer's whole
/// argument is a reachable blind spot, so the layout was chosen by
/// mapping where the pipeline actually goes blind, not by taste. With
/// two links at a wide angle (a third node at (3.0, 4.6), links (0,1)
/// and (0,2)) the second link fills in the first's nulls almost
/// perfectly: cells below the measured blindness threshold fall from
/// 2.9% of the room to 0.12%, and the largest connected dead patch from
/// ~20 cm across to a single 5 cm cell. Nothing a person can drag into.
/// Enlarging the room to 8 m x 6 m did not bring them back; at 10 m x 7 m
/// blind cells reappear, but because the whole room has gone marginal,
/// which demonstrates distance, not geometry.
///
/// That result is worth more than the demo: **a second link at an angle
/// is what buys you a room without blind spots.** The single link is the
/// honest minimum deployment, it is what the product being compared
/// against runs, and it is the configuration whose failure mode this
/// explorer exists to show.
const NODES_M: [Point; 2] = [(0.6, 2.5), (5.4, 2.5)];
/// (tx, rx).
const LINKS: [(u8, u8); 1] = [(0, 1)];

/// The scenario the explorer runs. Deterministic from `SEED`.
pub fn explorer_scenario() -> Scenario {
    let room = Room { door_m: DOOR_M, ..scenarios::room_clear_los() };
    let layout = scenarios::Layout {
        nodes: NODES_M.iter().enumerate().map(|(i, &pos)| Node { id: i as u8, pos }).collect(),
        links: LINKS.to_vec(),
    };
    let timeline = vec![
        Segment { duration_s: LEAD_IN_S, activity: Activity::Absent },
        Segment { duration_s: SEATED_S, activity: Activity::Seated { at: START_M, bpm: BPM, bpm_end: None } },
    ];
    scenarios::scenario(SCENARIO, room, layout, timeline, SEED)
}

/// The space the simulated room is published as.
///
/// An operator who writes `[[space]] id = "sim-room"` in the config
/// decides its settings — consent above all: `vitals = false` there and
/// the simulated room is gated exactly like a real one, at the node
/// (the extractor's own gate) and again at the gateway. Without such a
/// section the demo gets a consented room, because a breathing explorer
/// with breathing switched off explains nothing.
///
/// The links are always the scenario's: a simulated room is wired by the
/// simulator, not by the config file.
pub fn sim_space(cfg: &GatewayConfig, sc: &Scenario) -> SpaceConfig {
    let links: Vec<LinkId> = sc.links.iter().map(|&(tx, rx)| LinkId { rx, tx }).collect();
    match cfg.spaces.iter().find(|s| s.id == SPACE_ID) {
        Some(s) => SpaceConfig { links, ..s.clone() },
        None => SpaceConfig {
            id: SPACE_ID.into(),
            name: "Simulated room (synthetic)".into(),
            links,
            vitals: true,
            collapse_detection: true,
            ..SpaceConfig::default()
        },
    }
}

/// The config simulate mode runs with: exactly one space, the simulated
/// one. There are no sensors in this mode, so a configured space that is
/// not the simulated room would sit permanently unknown and invite the
/// reading that a synthetic number came from it.
pub fn simulate_config(mut cfg: GatewayConfig, sc: &Scenario) -> GatewayConfig {
    cfg.spaces = vec![sim_space(&cfg, sc)];
    cfg
}

// ------------------------------------------------------------- sim state

/// The simulator plus the time of the last tick it produced.
struct SimCore {
    sim: Simulator,
    /// Simulator time of the last tick (0 before the first).
    t_s: f64,
}

/// The simulated room, shared by the sim task and the HTTP handlers.
///
/// The mutex is held for one `step()` or one field read, never across an
/// await: an HTTP handler must not stall the tick, and the tick must not
/// stall a handler.
pub struct SimState {
    space_id: String,
    scenario: Scenario,
    /// A clone of the simulator taken before the first tick, never
    /// stepped. [`Simulator::sensitivity_at`] reads only the scenario's
    /// static channels, the torso model and the arm's rest offset — none
    /// of which a tick changes — so this answers exactly what the live
    /// simulator would, without taking its lock for the length of a map.
    maps: Simulator,
    core: Mutex<SimCore>,
}

impl SimState {
    pub fn new(scenario: Scenario, space_id: &str) -> Arc<Self> {
        let sim = Simulator::new(&scenario);
        Arc::new(Self { space_id: space_id.to_string(), maps: sim.clone(), scenario, core: Mutex::new(SimCore { sim, t_s: 0.0 }) })
    }

    fn core(&self) -> std::sync::MutexGuard<'_, SimCore> {
        self.core.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn space_id(&self) -> &str {
        &self.space_id
    }

    pub fn scenario(&self) -> &Scenario {
        &self.scenario
    }

    /// Ground truth as of the last tick, with any manual override
    /// applied — so a `POST /api/sim/person` is visible to the very next
    /// read, not only after the next 50 ms tick.
    pub fn truth(&self) -> Truth {
        let c = self.core();
        c.sim.truth_at(c.t_s)
    }

    /// `GET /api/sim`: the geometry, the space it publishes as, and the
    /// current ground truth. All SYNTHETIC.
    pub fn describe(&self) -> Value {
        let sc = &self.scenario;
        let t = self.truth();
        json!({
            "scenario": sc.name,
            "room": { "w_m": sc.room.size_m.0, "h_m": sc.room.size_m.1, "door_m": [sc.room.door_m.0, sc.room.door_m.1] },
            "nodes": sc.nodes.iter().map(|n| json!({ "id": n.id, "x_m": n.pos.0, "y_m": n.pos.1 })).collect::<Vec<_>>(),
            "links": sc.links.iter().map(|&(tx, rx)| json!({ "tx": tx, "rx": rx })).collect::<Vec<_>>(),
            "scatterers": self.maps.scatterers().iter()
                .map(|&(p, rcs)| json!({ "x_m": p.0, "y_m": p.1, "rcs_m2": rcs }))
                .collect::<Vec<_>>(),
            "space_id": self.space_id,
            "truth": {
                "t_s": t.t_s,
                "present": t.present,
                "still": t.still,
                "breathing": t.breathing,
                "bpm": t.bpm,
                "in_pause": t.in_pause,
                "x_m": t.position.0,
                "y_m": t.position.1,
            },
            "sample_rate_hz": sc.sample_rate_hz,
            "n_subcarriers": sc.n_subcarriers,
        })
    }

    /// `GET /api/sim/coverage`: per-link ground-truth breathing
    /// sensitivity over the room, plus the per-cell maximum over links
    /// (which is how ADR-006 fuses: the most confident link wins, not the
    /// average).
    ///
    /// Each link's grid is normalised by that link's own median, so
    /// `best` compares *relative* coverage between cells, not absolute
    /// signal between links. A rejected `res_m` comes back with the
    /// simulator's own message, which names the cap it breached.
    pub fn coverage(&self, res_m: f64) -> Result<Value, String> {
        let mut links = Vec::with_capacity(self.scenario.links.len());
        let mut best: Vec<f64> = Vec::new();
        let (mut w, mut h, mut origin) = (0usize, 0usize, (0.0, 0.0));
        for &(tx, rx) in &self.scenario.links {
            let c = self.maps.try_coverage((tx, rx), res_m)?;
            (w, h, origin) = (c.w, c.h, c.origin_m);
            if best.is_empty() {
                best = c.values.clone();
            } else {
                for (b, v) in best.iter_mut().zip(&c.values) {
                    *b = b.max(*v);
                }
            }
            links.push(json!({ "tx": tx, "rx": rx, "values": c.values }));
        }
        Ok(json!({
            "res_m": res_m,
            "w": w,
            "h": h,
            "origin_m": [origin.0, origin.1],
            "units": "relative breathing sensitivity; 1.0 = this room's median",
            "links": links,
            "best": best,
        }))
    }

    /// Move the person. The simulator clamps into the walkable area
    /// (0.4 m from every wall); the clamped point is what comes back and
    /// what the truth then reports. This is an input, not an inference.
    pub fn set_person(&self, p: Point) -> Point {
        let mut c = self.core();
        c.sim.set_position_override(Some(p));
        c.sim.position_override().unwrap_or(p)
    }

    /// Put a body in the room or take it out entirely. `false` removes
    /// the torso, the arm and the chest motion — an empty room, not a
    /// still one. `true` hands the body back to the timeline, which
    /// means it reappears only once the timeline has someone in it (the
    /// lead-in is deliberately empty).
    pub fn set_present(&self, present: bool) -> bool {
        let mut c = self.core();
        c.sim.set_present(present);
        c.sim.truth_at(c.t_s).present
    }
}

// ------------------------------------------------------------ the sensors

/// The per-link node DSP: the same [`FeatureExtractor`] the firmware
/// runs, one per link, fed the simulator's CSI.
struct Sensors {
    ex: Vec<((u8, u8), FeatureExtractor)>,
    /// Keeps the frame stream monotonic in `t`: per-link jitter can put
    /// one link's sample a millisecond behind another's.
    last_t: f64,
}

impl Sensors {
    /// `vitals` is the space's consent setting, applied where a real
    /// deployment applies it: in the node, which then never computes a
    /// breathing number into the frame at all.
    fn new(sc: &Scenario, vitals: bool, t0: f64) -> Self {
        let ex = sc
            .links
            .iter()
            .map(|&(tx, rx)| {
                let cfg = FeatureExtractorConfig {
                    n_subcarriers: sc.n_subcarriers,
                    sample_rate_hz: sc.sample_rate_hz,
                    node_id: rx,
                    tx_id: tx,
                    vitals_enabled: vitals,
                    ..Default::default()
                };
                ((tx, rx), FeatureExtractor::new(cfg))
            })
            .collect();
        Self { ex, last_t: t0 }
    }

    /// One simulated tick into the hub, by the path a UDP datagram takes:
    /// the node's DSP produces a `FeatureFrame`, it is encoded, and
    /// `Hub::ingest` decodes and routes it. `src` is `None` because there
    /// is no socket and no node — saying otherwise would be a lie in
    /// `/api/health`.
    fn feed(&mut self, tick: &Tick, t0: f64, hub: &mut Hub) {
        for lf in &tick.frames {
            let Some((_, ex)) = self.ex.iter_mut().find(|(k, _)| *k == (lf.tx, lf.rx)) else { continue };
            for ff in ex.push(lf.frame.clone(), Some(lf.seq as u64), Some(lf.t_s), None) {
                self.last_t = self.last_t.max(t0 + lf.t_s);
                let t = self.last_t;
                hub.ingest(&ff.encode(), t, None::<SocketAddr>);
            }
        }
    }
}

// ----------------------------------------------------------------- the run

#[derive(Debug, Clone, Copy)]
pub struct SimulateOptions {
    /// 1 = real time (the default), 10 = ten times faster, 0 = as fast as
    /// the CPU allows. The gateway clock *is* simulator time, so speed
    /// changes only how long a demo takes to watch: a 15 BPM breath is
    /// 15 BPM in the data at any speed.
    pub speed: f64,
}

impl Default for SimulateOptions {
    fn default() -> Self {
        Self { speed: 1.0 }
    }
}

/// A running simulate-mode gateway: the sim task, the HTTP server, and
/// the state both sides share. No UDP socket is bound — there are no
/// sensors in this mode.
pub struct SimGateway {
    pub http_addr: SocketAddr,
    pub shared: Arc<Shared>,
    pub sim: Arc<SimState>,
    stop: watch::Sender<bool>,
    core: JoinHandle<()>,
    http: JoinHandle<()>,
}

impl SimGateway {
    pub async fn start(cfg: GatewayConfig, opts: SimulateOptions) -> io::Result<SimGateway> {
        let scenario = explorer_scenario();
        let cfg = simulate_config(cfg, &scenario);
        let http_bind = cfg.http_socket_addr().map_err(|e| io::Error::new(io::ErrorKind::InvalidInput, e.0))?;
        let listener = TcpListener::bind(http_bind)
            .await
            .map_err(|e| io::Error::new(e.kind(), format!("cannot bind HTTP {http_bind}: {e}")))?;
        let http_addr = listener.local_addr()?;

        let shared = Shared::with_history("simulated", cfg.history);
        let sim = SimState::new(scenario, SPACE_ID);
        let (stop, stop_rx) = watch::channel(false);
        // No audit file and no event log: simulate mode leaves `data_dir`
        // alone, exactly as replay does.
        let deps = crate::http::HttpDeps::new(shared.clone(), stop_rx.clone(), Arc::new(cfg.clone()), None)
            .without_audit()
            .with_sim(sim.clone());
        let app = crate::http::router(deps);
        let mut http_stop = stop_rx.clone();
        let http = tokio::spawn(async move {
            let serve = axum::serve(listener, app).with_graceful_shutdown(async move {
                let _ = http_stop.wait_for(|s| *s).await;
            });
            if let Err(e) = serve.await {
                crate::log(&format!("HTTP server error: {e}"));
            }
        });
        let core = tokio::spawn(run(cfg, sim.clone(), shared.clone(), opts, stop_rx));
        Ok(SimGateway { http_addr, shared, sim, stop, core, http })
    }

    pub async fn shutdown(self) {
        let _ = self.stop.send(true);
        let _ = self.core.await;
        if tokio::time::timeout(Duration::from_secs(5), self.http).await.is_err() {
            crate::log("HTTP server did not stop within 5 s");
        }
    }
}

/// The sim task: step the simulator, run its CSI through the node DSP
/// into the hub, tick the hub on the same 0.5 s grid live mode uses, and
/// pace the whole thing against the wall clock.
async fn run(cfg: GatewayConfig, state: Arc<SimState>, shared: Arc<Shared>, opts: SimulateOptions, mut stop: watch::Receiver<bool>) {
    let offset = cfg.utc_offset_minutes;
    let vitals = cfg.spaces.first().is_some_and(|s| s.vitals);
    // Simulator time 0 is now: `t_s` is local-epoch seconds, like live
    // mode and replay, so persisted day boundaries mean something.
    let unix_ms_at_t0 = now_unix_ms();
    let t0 = local_epoch_s(unix_ms_at_t0, offset);
    let mut hub = Hub::new(cfg, Some(shared), false, Wall::Anchored { t0, unix_ms_at_t0 }, t0);
    for n in &hub.notes {
        crate::log(n);
    }
    hub.publish(t0);
    let mut sensors = Sensors::new(state.scenario(), vitals, t0);
    let started = Instant::now();
    let speed = opts.speed;
    let mut next_tick = t0 + TICK_S;
    let mut last_t = t0;
    loop {
        if *stop.borrow_and_update() {
            break;
        }
        let tick = {
            let mut c = state.core();
            let Some(tick) = c.sim.step() else {
                crate::log("the simulated timeline has ended; the explorer keeps serving its final state");
                break;
            };
            c.t_s = tick.truth.t_s;
            tick
        };
        let t = t0 + tick.truth.t_s;
        last_t = t;
        // Pace to the tick's own time, then run every hub tick due before
        // it, exactly as replay does.
        if speed > 0.0 {
            let target = started + Duration::from_secs_f64((tick.truth.t_s / speed).max(0.0));
            tokio::select! {
                _ = stop.changed() => break,
                _ = tokio::time::sleep_until(target.into()) => {}
            }
        } else {
            // Nothing in this loop awaits at full speed; without a yield
            // the HTTP server never runs on a single-threaded runtime.
            tokio::task::yield_now().await;
        }
        while next_tick <= t {
            hub.tick(next_tick);
            next_tick += TICK_S;
        }
        sensors.feed(&tick, t0, &mut hub);
    }
    // One more tick so the last frames are sampled.
    hub.tick(next_tick);
    hub.shutdown(last_t);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The explorer, headless, through the real hub: the simulator's
    /// CSI, the node's DSP, the gateway's fusion and detectors. The one
    /// thing the caller sets is where the person is (or whether there is
    /// one). Returns the space exactly as `/api/spaces` would show it.
    fn park(at: Option<Point>, run_s: f64) -> Value {
        let sc = explorer_scenario();
        let cfg = simulate_config(GatewayConfig { data_dir: std::path::PathBuf::new(), ..Default::default() }, &sc);
        let vitals = cfg.spaces[0].vitals;
        let t0 = 1_790_000_000.0;
        let mut hub = Hub::new(cfg, None, false, Wall::Anchored { t0, unix_ms_at_t0: 0 }, t0);
        let mut sensors = Sensors::new(&sc, vitals, t0);
        let mut sim = Simulator::new(&sc);
        match at {
            Some(p) => sim.set_position_override(Some(p)),
            None => sim.set_present(false),
        }
        let mut next_tick = t0 + TICK_S;
        while let Some(tick) = sim.step() {
            if tick.truth.t_s > run_s {
                break;
            }
            let t = t0 + tick.truth.t_s;
            while next_tick <= t {
                hub.tick(next_tick);
                next_tick += TICK_S;
            }
            sensors.feed(&tick, t0, &mut hub);
        }
        hub.tick(next_tick);
        hub.space_json(0, next_tick)
    }

    /// Long enough for the 30 s empty-room calibration and the filter
    /// warmup inside the lead-in, plus 2.5 min of the person sitting
    /// there. The outcome is the same at 200 s and at 400 s — this is
    /// geometry, not a race.
    const RUN_S: f64 = 280.0;

    fn bpm(space: &Value) -> Option<f64> {
        space["breathing"].get("bpm").and_then(Value::as_f64)
    }

    /// The reason this mode exists, as an assertion.
    ///
    /// Three positions, one simulated person, nothing else different:
    /// seen clearly; invisible; seen clearly again half a metre away.
    /// The truth panel shows the same seated person in all three.
    #[test]
    fn the_gap_the_explorer_exists_to_show() {
        let seen = park(Some(START_M), RUN_S);
        assert_eq!(seen["occupied"], true, "the person the explorer opens with must be visible: {seen}");
        assert!((bpm(&seen).unwrap_or(0.0) - BPM).abs() < 1.0, "breathing rate should be ~{BPM}: {}", seen["breathing"]);
        assert!(seen["breathing"]["confidence"].as_f64().unwrap_or(0.0) > 0.9, "{}", seen["breathing"]);

        // The same person, 0.51 m away, in a whole-band null.
        let null = park(Some(DEMO_NULL_M), RUN_S);
        assert_eq!(null["occupied"], false, "the null must actually blind the pipeline: {null}");
        assert!(null["breathing"].is_null(), "and with no person there is no breathing: {null}");

        // Half a metre out of it, the person is back.
        let beside = park(Some((DEMO_NULL_M.0 + 0.5, DEMO_NULL_M.1)), RUN_S);
        assert_eq!(beside["occupied"], true, "0.5 m from the null the person is visible again: {beside}");
        assert!((bpm(&beside).unwrap_or(0.0) - BPM).abs() < 1.0, "{}", beside["breathing"]);

        // And an empty room reads empty, so "not occupied" in the null
        // is not the pipeline simply never reporting anyone.
        let empty = park(None, RUN_S);
        assert_eq!(empty["occupied"], false, "{empty}");
        assert!(empty["breathing"].is_null(), "{empty}");
    }

    /// The coverage map is what tells a user where to drag. It has to
    /// agree with what the pipeline then does — and it only can at a
    /// resolution that resolves the structure, which is half a
    /// wavelength (~6 cm) across. At the contract's default 0.25 m the
    /// map is a point sample of a field that changes completely within
    /// 3 cm; this test pins the resolution the UI should ask for.
    #[test]
    fn the_map_marks_the_null_dark_when_it_can_resolve_it() {
        let sim = Simulator::new(&explorer_scenario());
        let c = sim.coverage(LINKS[0], 0.02);
        let at = |p: Point| {
            let ix = ((p.0 - c.origin_m.0) / c.res_m).round().clamp(0.0, (c.w - 1) as f64) as usize;
            let iy = ((p.1 - c.origin_m.1) / c.res_m).round().clamp(0.0, (c.h - 1) as f64) as usize;
            c.values[iy * c.w + ix]
        };
        assert!(at(DEMO_NULL_M) < 0.2, "the null reads dark: {}", at(DEMO_NULL_M));
        assert!(at(START_M) > 0.9, "the starting seat reads bright: {}", at(START_M));
        assert!(at((DEMO_NULL_M.0 + 0.5, DEMO_NULL_M.1)) > 0.5, "and so does the seat beside it");
    }

    /// The map served over HTTP is the simulator's own, cell for cell:
    /// the never-stepped clone answers exactly what the live simulator
    /// would, and moving the person does not change a what-if map.
    #[test]
    fn served_coverage_is_the_simulators_own_and_does_not_move_with_the_person() {
        let state = SimState::new(explorer_scenario(), SPACE_ID);
        let v = state.coverage(0.25).expect("0.25 m is a valid resolution");
        let own = Simulator::new(&explorer_scenario()).coverage(LINKS[0], 0.25);
        assert_eq!(v["w"], own.w);
        assert_eq!(v["h"], own.h);
        assert_eq!(v["origin_m"], json!([own.origin_m.0, own.origin_m.1]));
        assert_eq!(v["links"][0]["values"], json!(own.values));
        // One link, so `best` is that link; with more it is the per-cell
        // maximum, which is how ADR-006 fuses.
        assert_eq!(v["best"], json!(own.values));
        assert_eq!(v["units"], "relative breathing sensitivity; 1.0 = this room's median");

        state.set_person(DEMO_NULL_M);
        state.set_present(false);
        assert_eq!(state.coverage(0.25).unwrap(), v, "coverage is a map of the room, not of the person");

        assert!(state.coverage(0.0).is_err(), "a zero resolution is rejected, not divided by");
        assert!(state.coverage(f64::NAN).is_err());
        assert!(state.coverage(-1.0).is_err());
        assert!(state.coverage(0.0001).is_err(), "and so is a grid over the cell cap");
    }

    /// The person is an input. Setting it moves the truth immediately —
    /// before the next tick — and a point outside the room comes back
    /// clamped rather than silently ignored.
    #[test]
    fn the_person_is_set_not_inferred() {
        let state = SimState::new(explorer_scenario(), SPACE_ID);
        let p = state.set_person(DEMO_NULL_M);
        assert_eq!(p, DEMO_NULL_M);
        let t = state.truth();
        assert_eq!((t.position.0, t.position.1), DEMO_NULL_M);
        assert_eq!(state.describe()["truth"]["x_m"], DEMO_NULL_M.0);

        // The walkable area is 0.4 m from every wall (saans_sim::MARGIN_M).
        let far = state.set_person((-50.0, 500.0));
        assert_eq!(far, (0.4, 4.6), "clamped into the walkable area");
        assert_eq!(state.truth().position, far);

        // The lead-in is an empty room, so there is nobody to remove yet
        // and nobody to bring back; `present` reports what is true.
        assert!(!state.truth().present, "the explorer opens on an empty room");
        assert!(!state.set_present(false));
    }

    /// Consent is the space's, not the mode's: an operator who declares
    /// the simulated room without vitals gets a simulated room without
    /// vitals. The links are always the simulator's, whatever the config
    /// says, because a simulated room is wired by the simulator.
    #[test]
    fn the_sim_space_honours_a_configured_consent_setting() {
        let sc = explorer_scenario();
        let links: Vec<LinkId> = sc.links.iter().map(|&(tx, rx)| LinkId { rx, tx }).collect();

        let default = sim_space(&GatewayConfig::default(), &sc);
        assert_eq!(default.id, SPACE_ID);
        assert!(default.vitals, "the explorer is about breathing, so it consents by default");
        assert_eq!(default.links, links);

        let text = format!(
            "http_addr = \"127.0.0.1:0\"\n[[space]]\nid = \"{SPACE_ID}\"\nname = \"Quiet room\"\nvitals = false\n             links = [{{ rx = 9, tx = 9 }}]\n"
        );
        let cfg = GatewayConfig::from_toml(&text).expect("valid config");
        let configured = sim_space(&cfg, &sc);
        assert!(!configured.vitals, "a space that has not consented stays unconsented in simulate mode");
        assert_eq!(configured.name, "Quiet room");
        assert_eq!(configured.links, links, "the config does not get to rewire the simulator");

        // And the gateway runs that space and only that space.
        let only = simulate_config(cfg, &sc);
        assert_eq!(only.spaces.len(), 1);
        assert_eq!(only.spaces[0].id, SPACE_ID);
    }

    /// Consent reaches the node, not just the gateway: with vitals off
    /// the extractor never puts a breathing number in the frame.
    #[test]
    fn without_consent_no_breathing_number_is_even_computed() {
        let sc = explorer_scenario();
        let mut sim = Simulator::new(&sc);
        sim.set_position_override(Some(START_M));
        let mut gated = Sensors::new(&sc, false, 0.0);
        let mut frames = 0;
        while let Some(tick) = sim.step() {
            if tick.truth.t_s > 200.0 {
                break;
            }
            for lf in &tick.frames {
                let Some((_, ex)) = gated.ex.iter_mut().find(|(k, _)| *k == (lf.tx, lf.rx)) else { continue };
                for ff in ex.push(lf.frame.clone(), Some(lf.seq as u64), Some(lf.t_s), None) {
                    assert_eq!(ff.breathing_bpm, 0.0, "a node without consent reports no rate");
                    assert_eq!(ff.breathing_conf, 0.0);
                    assert_eq!(ff.breath_rms, 0.0);
                    frames += 1;
                }
            }
        }
        assert!(frames > 100, "the run produced frames to check ({frames})");
    }

    /// Geometry sanity, and that `/api/sim` says what the contract says.
    #[test]
    fn describe_matches_the_published_shape() {
        let state = SimState::new(explorer_scenario(), SPACE_ID);
        let v = state.describe();
        assert_eq!(v["scenario"], SCENARIO);
        assert_eq!(v["space_id"], SPACE_ID);
        assert_eq!(v["room"], json!({ "w_m": 6.0, "h_m": 5.0, "door_m": [DOOR_M.0, DOOR_M.1] }));
        assert_eq!(v["nodes"], json!([
            { "id": 0, "x_m": NODES_M[0].0, "y_m": NODES_M[0].1 },
            { "id": 1, "x_m": NODES_M[1].0, "y_m": NODES_M[1].1 },
        ]));
        assert_eq!(v["links"], json!([{ "tx": 0, "rx": 1 }]));
        assert_eq!(v["sample_rate_hz"], 20.0);
        assert_eq!(v["n_subcarriers"], 56);
        let sc = state.scenario();
        assert_eq!(v["scatterers"].as_array().map(Vec::len), Some(sc.room.n_scatterers));
        for s in v["scatterers"].as_array().unwrap() {
            assert!(s["x_m"].as_f64().is_some_and(|x| x > 0.0 && x < 6.0), "{s}");
            assert!(s["rcs_m2"].as_f64().is_some_and(|r| r > 0.0), "{s}");
        }
        for n in &sc.nodes {
            assert!(n.pos.0 > 0.0 && n.pos.0 < sc.room.size_m.0 && n.pos.1 > 0.0 && n.pos.1 < sc.room.size_m.1);
        }
        assert!(v["truth"]["t_s"].as_f64() == Some(0.0) && v["truth"]["present"] == false);
    }
}
