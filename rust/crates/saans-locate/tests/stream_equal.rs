//! The streaming driver (spec/livedemo.md §8.1, §11 saans-locate), record
//! for record: for every shipped scene and every variant at 20 s,
//! - `run_roomsim`, re-expressed as one setup and one stream, equals the
//!   Phase 4 batch loop bit for bit;
//! - streams from one setup, advanced in steps of 0.025, 0.1 and 0.8 s and
//!   on the session clock (`T = float(2q) * 0.0125`, every speed), each
//!   give `run_roomsim`'s ticks and epochs, and each other's links;
//! - each step emits exactly the records with `t_prev < t <= t_end`;
//! - every tick's link values are the `y` its WLM used, with the §4 gate,
//!   and WLM abstains `no_disturbance` exactly when no link is disturbed;
//! - `empty_room()` equals `run_roomsim` of the `empty` variant;
//! - ticks after the last transmission wait for `t_end` to reach them.
//!
//! Everything here is SIM.
#![cfg(feature = "roomsim")]

use std::path::PathBuf;

use saans_locate::baseline::{disturbed, wlm, WlmParams};
use saans_locate::phone::{Config, PhoneRun, TruthSample};
use saans_locate::pipeline::{
    base_rate_hz, blockers, calibration_config, check_roomsim_config, check_tx_period, cylinders,
    link_geoms, next_tick, packet_frame, person_geometry, phone_nodes, resolve_config, run_roomsim,
    tick_time, CalibrationSummary, Diagnostics, PersonLocator, PersonTruthRecord, RoomSimRun,
    TickRecord,
};
use saans_locate::radius::{Outcome, Reason};
use saans_locate::stream::{Advance, LinkValue, RoomSimSetup};
use saans_locate::variant::Variant;
use saans_roomsim::{Config as RoomSimConfig, Simulator};
use saans_scene::Scene;

const DURATION_S: f64 = 20.0;
const SHIPPED: [&str; 3] = ["bedroom", "hall-l-shaped", "office"];

fn scenes_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes")
}

fn scene(name: &str) -> Scene {
    Scene::load(scenes_dir().join(format!("{name}.json"))).expect("a shipped scene loads")
}

/// The variants' common base, as a session builds it (spec/livedemo.md
/// §3.1): Room Sim's defaults, seed 0, 20 s.
fn base() -> RoomSimConfig {
    RoomSimConfig {
        seed: 0,
        duration_s: DURATION_S,
        ..RoomSimConfig::default()
    }
}

/// Phase 4's `run_roomsim`, verbatim on the public API: the batch loop the
/// stream replaced (spec/locate.md §10.2).
fn batch_run(scene: &Scene, rs: &RoomSimConfig, cfg: &Config) -> RoomSimRun {
    cfg.validate().unwrap();
    check_roomsim_config(rs, cfg).unwrap();
    let cfg = &resolve_config(rs, cfg).unwrap();
    let sim = Simulator::new(scene, rs.clone()).unwrap();
    let geometry = person_geometry(scene, cfg).unwrap();
    let rate_hz = base_rate_hz(&sim);
    let mut person = PersonLocator::new(geometry, cfg, rate_hz, "SIM").unwrap();
    check_tx_period(&sim, cfg.window_s, rate_hz, person.min_samples()).unwrap();
    let fw = rs.fw_subcarriers;

    let calib_sim = Simulator::new(scene, calibration_config(rs, cfg.calib_s)).unwrap();
    let mut calib_packets = 0usize;
    let stream = calib_sim
        .packets()
        .filter(|p| p.t >= rs.duration_s)
        .filter_map(|p| {
            packet_frame(&p, fw, rate_hz).unwrap().map(|frame| {
                calib_packets += 1;
                (p.t, p.tx, p.rx, frame)
            })
        });
    person.calibrate_from(stream).unwrap();
    let calibration = CalibrationSummary {
        packets: calib_packets,
        baseline: person.measurement().baseline_vector(),
        null_sigma: person.measurement().baseline_std_vector(),
        calibrated_mask: person.measurement().calibrated_mask(),
    };

    let mut phone_run = scene
        .phone
        .as_ref()
        .map(|_| PhoneRun::new(cfg, &phone_nodes(&sim), &blockers(&sim), "SIM").unwrap());

    let tick = |person: &mut PersonLocator, t_k: f64, truth: &[PersonTruthRecord]| {
        let (wlm_out, rti_out) = person.tick(t_k).unwrap();
        TickRecord {
            label: person.label(),
            t: t_k,
            wlm: wlm_out,
            rti: rti_out,
            truth: truth.to_vec(),
        }
    };
    let duration_s = rs.duration_s;
    let mut ticks = Vec::new();
    let mut epochs = Vec::new();
    let mut k: u64 = 0;
    let mut t_k = tick_time(cfg.window_s, cfg.tick_hz, k);
    let mut truth_now: Vec<PersonTruthRecord> = Vec::new();
    let mut packets = 0usize;
    for tr in sim.run() {
        while t_k < duration_s && t_k < tr.t {
            ticks.push(tick(&mut person, t_k, &truth_now));
            t_k = next_tick(cfg.window_s, cfg.tick_hz, k, t_k).unwrap();
            k += 1;
        }
        truth_now = tr
            .truth
            .people
            .iter()
            .map(|p| PersonTruthRecord {
                x: p.x,
                y: p.y,
                walking: p.walking,
                posture: p.posture,
            })
            .collect();
        if let (Some(run), Some(p)) = (phone_run.as_mut(), tr.truth.phone) {
            let cyl = cylinders(&sim, &tr.truth).unwrap();
            let sample = TruthSample {
                t: tr.t,
                phone: [p.x, p.y, p.z],
                cylinders: &cyl,
            };
            if let Some(rec) = run.step(&sample).unwrap() {
                epochs.push(rec);
            }
        }
        for p in &tr.packets {
            if let Some(frame) = packet_frame(p, fw, rate_hz).unwrap() {
                person.update(p.t, p.tx, p.rx, &frame).unwrap();
                packets += 1;
            }
        }
    }
    while t_k < duration_s {
        ticks.push(tick(&mut person, t_k, &truth_now));
        t_k = next_tick(cfg.window_s, cfg.tick_hz, k, t_k).unwrap();
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
    RoomSimRun {
        label: "SIM",
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
    }
}

/// Everything one stream emitted, concatenated.
struct Driven {
    all: Advance,
    diagnostics: Diagnostics,
}

/// A fresh stream of `setup` advanced to each of `times`, then to
/// infinity. Each step must emit exactly the records with
/// `t_prev < t <= t_end` (`t_prev = -inf` at first), one link list per
/// tick, and leave `t()` at `t_end`.
fn drive(setup: &RoomSimSetup, times: &[f64]) -> Driven {
    let mut s = setup.stream().unwrap();
    assert_eq!(s.t(), f64::NEG_INFINITY);
    let mut all = Advance::default();
    let mut prev = f64::NEG_INFINITY;
    assert!(!s.finished());
    for &t_end in times.iter().chain(&[f64::INFINITY]) {
        let a = s.advance_to(t_end).unwrap();
        assert_eq!(s.t(), t_end);
        assert_eq!(a.links.len(), a.ticks.len());
        for tick in &a.ticks {
            assert!(
                prev < tick.t && tick.t <= t_end,
                "tick {} outside its step",
                tick.t
            );
            assert!(tick.t < setup.duration_s());
        }
        for e in &a.epochs {
            assert!(prev < e.t && e.t <= t_end, "epoch {} outside its step", e.t);
        }
        all.ticks.extend(a.ticks);
        all.links.extend(a.links);
        all.epochs.extend(a.epochs);
        prev = t_end;
    }
    assert!(s.finished());
    // Finished: nothing more, and the counters stay.
    let diagnostics = s.diagnostics();
    assert_eq!(s.advance_to(f64::INFINITY).unwrap(), Advance::default());
    assert_eq!(s.diagnostics(), diagnostics);
    Driven { all, diagnostics }
}

/// `k * step` for `k = 1, 2, …` up to the first at or past the end.
fn steps(step: f64) -> Vec<f64> {
    let n = (DURATION_S / step).ceil() as u64;
    (1..=n).map(|k| k as f64 * step).collect()
}

/// The session clock (spec/livedemo.md §3.2): `T = float(2q) * 0.0125`,
/// `q` from 0 by 1, 2, 4, 8, 16 and 32 quanta in turn (every speed),
/// clamped at `q_end = 40 * duration_s`, with a paused frame (the same `T`
/// again) every fifth frame.
fn session_clock() -> Vec<f64> {
    let q_end = (40.0 * DURATION_S) as u64;
    let speeds = [1u64, 2, 4, 8, 16, 32];
    let mut q = 0u64;
    let mut times = vec![0.0];
    let mut frame = 0usize;
    while q < q_end {
        q = (q + speeds[frame % speeds.len()]).min(q_end);
        let t = (2 * q) as f64 * 0.0125;
        times.push(t);
        if frame % 5 == 4 {
            times.push(t);
        }
        frame += 1;
    }
    assert_eq!(*times.last().unwrap(), DURATION_S);
    times
}

/// §4 `links` for one tick: the gate is `wlm_gate_z * null_sigma +
/// wlm_floor` when `null_sigma > 0`, `disturbed` is WLM's own test, WLM
/// recomputed from the items' `y` is the tick's WLM (so `y` is the vector
/// it used), and WLM abstains `no_disturbance` exactly when no item is
/// disturbed.
fn check_links(setup: &RoomSimSetup, scene: &Scene, tick: &TickRecord, links: &[LinkValue]) {
    let cfg = setup.config();
    let params = WlmParams::from_config(cfg, &cfg.link_signal);
    let floor = cfg.wlm_floor_for(&cfg.link_signal);
    let null_sigma = &setup.calibration().null_sigma;
    assert_eq!(links.len(), setup.links().len());
    for (v, &s) in links.iter().zip(null_sigma) {
        let gate = if s > 0.0 {
            Some(cfg.wlm_gate_z * s + floor)
        } else {
            None
        };
        assert_eq!(v.gate, gate);
        assert_eq!(v.disturbed, disturbed(v.y, s, &params));
        if let Some(g) = v.gate {
            assert_eq!(v.disturbed, v.y > g);
        } else {
            assert!(!v.disturbed);
        }
    }
    let geometry = person_geometry(scene, cfg).unwrap();
    let y: Vec<f64> = links.iter().map(|v| v.y).collect();
    let again = wlm(
        &link_geoms(&geometry),
        geometry.lambda_m(),
        &y,
        null_sigma,
        &params,
    )
    .unwrap();
    assert_eq!(again, tick.wlm, "tick {}: WLM from the link values", tick.t);
    let any = links.iter().any(|v| v.disturbed);
    assert_eq!(
        tick.wlm == Outcome::Abstain(Reason::NoDisturbance),
        !any,
        "tick {}: no_disturbance exactly when no link is disturbed",
        tick.t
    );
}

fn check_scene_variant(name: &str, variant: Variant) {
    let scene = scene(name);
    let base = base();
    let rs = variant.apply(&scene, &base);
    let cfg = Config::default();

    // run_roomsim, re-expressed, is the batch loop bit for bit.
    let run = run_roomsim(&scene, &rs, &cfg, "SIM").unwrap();
    assert_eq!(run, batch_run(&scene, &rs, &cfg), "{name} {variant:?}");
    assert!(!run.ticks.is_empty());
    assert_eq!(run.has_phone, !run.epochs.is_empty());

    let setup = RoomSimSetup::new(&scene, &rs, &cfg, "SIM", &mut || true)
        .unwrap()
        .expect("never abandoned");
    assert_eq!(setup.label(), "SIM");
    assert_eq!(setup.roomsim_config(), &rs);
    assert_eq!(
        setup.config(),
        &saans_locate::pipeline::resolve_config(&rs, &cfg).unwrap()
    );
    assert_eq!(setup.links(), &run.links[..]);
    assert_eq!(setup.calibration(), &run.calibration);
    assert_eq!(setup.has_phone(), run.has_phone);
    assert_eq!(setup.rate_hz(), run.rate_hz);
    assert_eq!(setup.min_samples(), run.min_samples);
    assert_eq!(setup.duration_s(), DURATION_S);
    assert_eq!(setup.tick_hz(), cfg.tick_hz);
    assert_eq!(setup.window_s(), cfg.window_s);

    // Every stepping gives run_roomsim's records; streams from one setup
    // give each other's links.
    let mut first: Option<Vec<Vec<LinkValue>>> = None;
    for times in [steps(0.025), steps(0.1), steps(0.8), session_clock()] {
        let d = drive(&setup, &times);
        assert_eq!(d.all.ticks, run.ticks, "{name} {variant:?}: ticks");
        assert_eq!(d.all.epochs, run.epochs, "{name} {variant:?}: epochs");
        assert_eq!(d.diagnostics, run.diagnostics);
        match &first {
            None => {
                for (tick, links) in d.all.ticks.iter().zip(&d.all.links) {
                    check_links(&setup, &scene, tick, links);
                }
                // Not vacuous: a walker disturbs links, and WLM publishes.
                if variant == Variant::WithWalker {
                    assert!(d.all.links.iter().any(|l| l.iter().any(|v| v.disturbed)));
                    assert!(d.all.ticks.iter().any(|t| t.wlm.estimate().is_some()));
                }
                first = Some(d.all.links);
            }
            Some(links) => assert_eq!(&d.all.links, links, "{name} {variant:?}: links"),
        }
    }

    // The empty room of the same scene, seed and duration, sharing this
    // calibration, is run_roomsim of the empty variant.
    let empty = setup.empty_room().unwrap();
    assert!(std::ptr::eq(empty.calibration(), setup.calibration()));
    let rs_empty = Variant::Empty.apply(&scene, &base);
    assert_eq!(empty.roomsim_config(), &rs_empty);
    let want = if variant == Variant::Empty {
        run
    } else {
        run_roomsim(&scene, &rs_empty, &cfg, "SIM").unwrap()
    };
    assert_eq!(empty.calibration(), &want.calibration);
    assert_eq!(empty.links(), &want.links[..]);
    assert_eq!(empty.has_phone(), want.has_phone);
    let d = drive(&empty, &steps(0.8));
    assert_eq!(
        d.all.ticks, want.ticks,
        "{name} {variant:?}: empty-room ticks"
    );
    assert_eq!(
        d.all.epochs, want.epochs,
        "{name} {variant:?}: empty-room epochs"
    );
    assert_eq!(d.diagnostics, want.diagnostics);
    assert!(d.all.ticks.iter().all(|t| t.truth.is_empty()));
    for (tick, links) in d.all.ticks.iter().zip(&d.all.links) {
        check_links(&empty, &scene, tick, links);
    }
}

/// The ticks after the last transmission (§8.1): with `tx_period` 2 on
/// every node, `tick_hz` 20 and 6.06 s, the bedroom's last transmission
/// comes before its last tick. Advanced to that transmission, a stream has
/// consumed every one yet emits only the ticks with `t_k <= t_end` (§4: a
/// frame at `T` holds the records with `t <= T`) and is not finished; each
/// later tick comes when `t_end` reaches it, and the whole is the batch
/// loop's.
#[test]
fn ticks_after_the_last_transmission_wait_for_t_end() {
    let scene = scene("bedroom");
    let mut base = RoomSimConfig {
        seed: 1,
        duration_s: 6.06,
        ..RoomSimConfig::default()
    };
    for n in &scene.nodes {
        base.tx_period.insert(u32::from(n.id), 2);
    }
    let rs = Variant::WithWalker.apply(&scene, &base);
    let cfg = Config {
        tick_hz: 20.0,
        ..Config::default()
    };
    let run = batch_run(&scene, &rs, &cfg);
    assert_eq!(run_roomsim(&scene, &rs, &cfg, "SIM").unwrap(), run);
    let t_last = Simulator::new(&scene, rs.clone())
        .unwrap()
        .run()
        .map(|tr| tr.t)
        .last()
        .unwrap();
    let n_before = run.ticks.iter().filter(|t| t.t <= t_last).count();
    let after: Vec<f64> = run.ticks[n_before..].iter().map(|t| t.t).collect();
    assert!(!after.is_empty(), "a tick after the last transmission");
    assert!(after.iter().all(|&t| t > t_last));

    let setup = RoomSimSetup::new(&scene, &rs, &cfg, "SIM", &mut || true)
        .unwrap()
        .unwrap();
    let mut s = setup.stream().unwrap();
    let a = s.advance_to(t_last).unwrap();
    assert_eq!(a.ticks, run.ticks[..n_before]);
    assert_eq!(a.epochs, run.epochs);
    assert_eq!(s.diagnostics().packets, run.diagnostics.packets);
    assert!(!s.finished());
    let mid = (t_last + after[0]) / 2.0;
    assert_eq!(s.advance_to(mid).unwrap(), Advance::default());
    assert!(!s.finished());
    for (i, &t) in after.iter().enumerate() {
        let b = s.advance_to(t).unwrap();
        assert_eq!(b.ticks, run.ticks[n_before + i..=n_before + i]);
        assert_eq!(b.links.len(), 1);
        assert!(b.epochs.is_empty());
        assert_eq!(s.finished(), i + 1 == after.len());
    }
    assert_eq!(s.advance_to(f64::INFINITY).unwrap(), Advance::default());
    assert_eq!(s.diagnostics(), run.diagnostics);

    // The same through the step checks.
    let d = drive(&setup, &[t_last, mid]);
    assert_eq!(d.all.ticks, run.ticks);
    assert_eq!(d.all.epochs, run.epochs);
    assert_eq!(d.diagnostics, run.diagnostics);
    for (tick, links) in d.all.ticks.iter().zip(&d.all.links) {
        check_links(&setup, &scene, tick, links);
    }
}

#[test]
fn the_shipped_scenes_are_these() {
    let mut found: Vec<String> = std::fs::read_dir(scenes_dir())
        .unwrap()
        .filter_map(|e| {
            let name = e.unwrap().file_name().into_string().unwrap();
            name.strip_suffix(".json").map(str::to_string)
        })
        .collect();
    found.sort();
    assert_eq!(found, SHIPPED);
}

macro_rules! scene_variant {
    ($($test:ident: $scene:expr, $variant:expr;)*) => {$(
        #[test]
        fn $test() {
            check_scene_variant($scene, $variant);
        }
    )*};
}

scene_variant! {
    bedroom_declared: "bedroom", Variant::Declared;
    bedroom_with_walker: "bedroom", Variant::WithWalker;
    bedroom_empty: "bedroom", Variant::Empty;
    office_declared: "office", Variant::Declared;
    office_with_walker: "office", Variant::WithWalker;
    office_empty: "office", Variant::Empty;
    hall_declared: "hall-l-shaped", Variant::Declared;
    hall_with_walker: "hall-l-shaped", Variant::WithWalker;
    hall_empty: "hall-l-shaped", Variant::Empty;
}
