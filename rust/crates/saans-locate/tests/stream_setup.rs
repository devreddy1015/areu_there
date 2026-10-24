//! The streaming driver's setup and stream (spec/livedemo.md §8.1): the
//! calibration's `keep_going` (§3.0), the refusals and which of two wins,
//! interleaved streams of one setup, `advance_to`'s own refusals, and no
//! `Person.id` or `Phone.id` in anything the driver prints. Everything
//! here is SIM.
#![cfg(feature = "roomsim")]

use std::path::PathBuf;

use saans_locate::phone::Config;
use saans_locate::pipeline::{calibration_config, run_roomsim, PipelineError};
use saans_locate::stream::{Advance, RoomSimSetup, RoomSimStream};
use saans_locate::variant::Variant;
use saans_roomsim::{Config as RoomSimConfig, Role, Simulator};
use saans_scene::Scene;

fn scene(name: &str) -> Scene {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../../spec/scenes/{name}.json"));
    Scene::load(path).expect("a shipped scene loads")
}

/// The bedroom with a walker carrying the phone, 4 s.
fn short() -> (Scene, RoomSimConfig) {
    let scene = scene("bedroom");
    let base = RoomSimConfig {
        seed: 3,
        duration_s: 4.0,
        ..RoomSimConfig::default()
    };
    let rs = Variant::WithWalker.apply(&scene, &base);
    (scene, rs)
}

/// `RoomSimSetup::new` with a `keep_going` that answers true `n` times,
/// then false; and how often it was called.
fn setup_stopping_after(
    scene: &Scene,
    rs: &RoomSimConfig,
    cfg: &Config,
    label: &str,
    n: usize,
) -> (Result<Option<RoomSimSetup>, PipelineError>, usize) {
    let mut calls = 0usize;
    let mut keep_going = || {
        calls += 1;
        calls <= n
    };
    let out = RoomSimSetup::new(scene, rs, cfg, label, &mut keep_going);
    (out, calls)
}

/// `keep_going` is called before each calibration packet is drawn, the
/// packets with `t < duration_s` included, and once more before the run is
/// found finished; its first false gives `Ok(None)` at once.
#[test]
fn keep_going_is_polled_before_every_calibration_packet() {
    let (scene, rs) = short();
    let cfg = Config::default();
    let total = Simulator::new(&scene, calibration_config(&rs, cfg.calib_s))
        .unwrap()
        .packets()
        .count();
    let before = Simulator::new(&scene, calibration_config(&rs, cfg.calib_s))
        .unwrap()
        .packets()
        .take_while(|p| p.t < rs.duration_s)
        .count();
    assert!(before > 0 && total > before);

    // Never stopped: every packet and the end are polled.
    let (out, calls) = setup_stopping_after(&scene, &rs, &cfg, "SIM", usize::MAX);
    let setup = out.unwrap().expect("not abandoned");
    assert_eq!(calls, total + 1);
    assert_eq!(
        setup.calibration(),
        &run_roomsim(&scene, &rs, &cfg, "SIM").unwrap().calibration
    );

    // Stopped at the first call, among the packets before duration_s,
    // among those after, at the last packet and at the end: Ok(None), and
    // no call after the first false.
    for n in [0, 1, before / 2, before + 5, total - 1, total] {
        let (out, calls) = setup_stopping_after(&scene, &rs, &cfg, "SIM", n);
        assert!(matches!(out, Ok(None)), "stopped after {n} calls");
        assert_eq!(calls, n + 1);
    }
}

/// The setup makes run_roomsim's refusals, naming the parameter written
/// here, and a refusal before the calibration never calls `keep_going`.
#[test]
fn refusals_are_run_roomsims() {
    let (scene, base) = short();
    let cfg = Config::default();
    let mut roles = base.clone();
    roles.roles.insert(1, Role::Rx);
    let mut tx_period = base.clone();
    tx_period.tx_period.insert(1, 3);
    let cases: Vec<(RoomSimConfig, Config, &str, &str)> = vec![
        (base.clone(), cfg.clone(), "REAL", "label"),
        (roles, cfg.clone(), "SIM", "roles"),
        (
            RoomSimConfig {
                quantize: false,
                ..base.clone()
            },
            cfg.clone(),
            "SIM",
            "quantize",
        ),
        (tx_period, cfg.clone(), "SIM", "tx_period"),
        (
            RoomSimConfig {
                channel: 0,
                ..base.clone()
            },
            cfg.clone(),
            "SIM",
            "channel",
        ),
        (
            base.clone(),
            Config {
                link_signal: "csi_magic".into(),
                ..cfg.clone()
            },
            "SIM",
            "link_signal",
        ),
        (
            base.clone(),
            Config {
                tick_hz: 1000.0,
                ..cfg.clone()
            },
            "SIM",
            "tick_hz",
        ),
    ];
    for (rs, cfg, label, param) in cases {
        let (out, calls) = setup_stopping_after(&scene, &rs, &cfg, label, usize::MAX);
        let e = out.unwrap_err();
        assert_eq!(e.param(), Some(param));
        assert_eq!(calls, 0, "{param}: refused before the calibration");
        assert_eq!(run_roomsim(&scene, &rs, &cfg, label).unwrap_err(), e);
    }

    // A calibration that raises (2 s of calibration for 10 windows) is
    // refused naming calib_s, after the calibration ran.
    let cfg = Config {
        calib_s: 2.0,
        ..Config::default()
    };
    let (out, calls) = setup_stopping_after(&scene, &base, &cfg, "SIM", usize::MAX);
    let e = out.unwrap_err();
    assert_eq!(e.param(), Some("calib_s"));
    assert!(calls > 0);
    assert_eq!(run_roomsim(&scene, &base, &cfg, "SIM").unwrap_err(), e);
}

/// With two faults, the one Phase 4's `run_roomsim` checked first wins:
/// `label`, `cfg`'s §9 validation, Room Sim's own config validation,
/// `roles`, `quantize`, `tick_hz`, `phone_seed`, … `tx_period`, then the
/// calibration. The winners are written out here, not taken from
/// `run_roomsim` (which is now the setup itself); none calls `keep_going`.
#[test]
fn refusals_win_in_run_roomsims_order() {
    let (scene, base) = short();
    let cfg = Config::default();
    let mut roles = base.clone();
    roles.roles.insert(1, Role::Rx);
    let mut tx_period = base.clone();
    tx_period.tx_period.insert(1, 3);
    let mut roles_tx_period = roles.clone();
    roles_tx_period.tx_period.insert(1, 3);
    let csi_magic = Config {
        link_signal: "csi_magic".into(),
        ..cfg.clone()
    };
    let fast = Config {
        tick_hz: 1000.0,
        ..cfg.clone()
    };
    let other_seed = Config {
        phone_seed: Some(base.seed + 1),
        ..cfg.clone()
    };
    let short_calibration = Config {
        calib_s: 2.0,
        ..cfg.clone()
    };
    let cases: Vec<(RoomSimConfig, Config, &str, &str)> = vec![
        (roles.clone(), cfg.clone(), "REAL", "label"),
        (roles.clone(), csi_magic.clone(), "SIM", "link_signal"),
        (
            RoomSimConfig {
                channel: 0,
                ..roles.clone()
            },
            cfg.clone(),
            "SIM",
            "channel",
        ),
        (
            RoomSimConfig {
                quantize: false,
                ..roles.clone()
            },
            fast.clone(),
            "SIM",
            "roles",
        ),
        (
            RoomSimConfig {
                quantize: false,
                ..base.clone()
            },
            fast.clone(),
            "SIM",
            "quantize",
        ),
        (roles.clone(), other_seed.clone(), "SIM", "roles"),
        (tx_period.clone(), other_seed, "SIM", "phone_seed"),
        (roles_tx_period, cfg.clone(), "SIM", "roles"),
        (tx_period.clone(), fast, "SIM", "tick_hz"),
        (tx_period.clone(), csi_magic, "SIM", "link_signal"),
        (tx_period, short_calibration, "SIM", "tx_period"),
    ];
    for (rs, cfg, label, param) in cases {
        let (out, calls) = setup_stopping_after(&scene, &rs, &cfg, label, usize::MAX);
        let e = out.unwrap_err();
        assert_eq!(e.param(), Some(param));
        assert_eq!(calls, 0, "{param}: refused before the calibration");
        assert_eq!(run_roomsim(&scene, &rs, &cfg, label).unwrap_err(), e);
    }
}

/// Streams of one setup share nothing mutable: two advanced in turn give
/// run_roomsim's records each, and a third started late does too.
#[test]
fn interleaved_streams_of_one_setup_are_identical() {
    let (scene, rs) = short();
    let cfg = Config::default();
    let run = run_roomsim(&scene, &rs, &cfg, "SIM").unwrap();
    let setup = RoomSimSetup::new(&scene, &rs, &cfg, "SIM", &mut || true)
        .unwrap()
        .unwrap();
    let mut a = setup.stream().unwrap();
    let mut b = setup.stream().unwrap();
    let (mut got_a, mut got_b) = (Advance::default(), Advance::default());
    let extend = |all: &mut Advance, s: &mut RoomSimStream<'_>, t: f64| {
        let x = s.advance_to(t).unwrap();
        all.ticks.extend(x.ticks);
        all.links.extend(x.links);
        all.epochs.extend(x.epochs);
    };
    for q in 1..=40 {
        let t = (2 * 4 * q) as f64 * 0.0125;
        extend(&mut got_a, &mut a, t);
        if q % 3 == 0 {
            extend(&mut got_b, &mut b, t);
        }
    }
    let mut c = setup.stream().unwrap();
    let got_c = c.advance_to(f64::INFINITY).unwrap();
    extend(&mut got_a, &mut a, f64::INFINITY);
    extend(&mut got_b, &mut b, f64::INFINITY);
    for got in [&got_a, &got_b, &got_c] {
        assert_eq!(got.ticks, run.ticks);
        assert_eq!(got.epochs, run.epochs);
        assert_eq!(got.links, got_c.links);
    }
    for s in [&a, &b, &c] {
        assert!(s.finished());
        assert_eq!(s.diagnostics(), run.diagnostics);
        assert!(std::ptr::eq(s.setup(), &setup));
    }
}

/// `advance_to` refuses a NaN `t_end` or one before the last, naming
/// `t_end` and changing nothing; an equal one emits nothing; before the
/// first transmission nothing is emitted; the phone's epoch at t = 0 comes
/// with `t_end = 0`, and the first tick with `t_end = window_s`.
#[test]
fn advance_to_takes_time_forward_only() {
    let (scene, rs) = short();
    let cfg = Config::default();
    let run = run_roomsim(&scene, &rs, &cfg, "SIM").unwrap();
    let setup = RoomSimSetup::new(&scene, &rs, &cfg, "SIM", &mut || true)
        .unwrap()
        .unwrap();
    let mut s = setup.stream().unwrap();
    assert_eq!(s.advance_to(-1.0).unwrap(), Advance::default());
    assert_eq!(s.t(), -1.0);
    let first = s.advance_to(0.0).unwrap();
    assert!(first.ticks.is_empty());
    assert_eq!(first.epochs, run.epochs[..1]);
    assert_eq!(first.epochs[0].t, 0.0);
    assert_eq!(s.advance_to(0.0).unwrap(), Advance::default());
    for bad in [f64::NAN, -0.5] {
        let e = s.advance_to(bad).unwrap_err();
        assert_eq!(e.param(), Some("t_end"));
        assert_eq!(s.t(), 0.0);
    }
    let a = s.advance_to(setup.window_s()).unwrap();
    assert_eq!(a.ticks, run.ticks[..1]);
    assert_eq!(a.ticks[0].t, setup.window_s());
    assert!(!s.finished());
    // A refused t_end did not end the stream.
    let rest = s.advance_to(setup.duration_s()).unwrap();
    assert!(s.finished());
    assert_eq!(rest.ticks, run.ticks[1..]);
    assert_eq!(
        [&first.epochs[..], &a.epochs[..], &rest.epochs[..]].concat(),
        run.epochs
    );
}

/// No `Person.id` or `Phone.id` in what the driver prints: the setup, its
/// empty room, a stream and everything it emits, for every shipped scene
/// (ADR-008 decision 3). The ids are looked for as quoted strings, since
/// "phone" is also a field name.
#[test]
fn no_scene_id_in_anything_printed() {
    for name in ["bedroom", "office", "hall-l-shaped"] {
        let scene = scene(name);
        let mut ids: Vec<String> = scene.people.iter().filter_map(|p| p.id.clone()).collect();
        ids.extend(scene.phone.as_ref().and_then(|p| p.id.clone()));
        assert!(!ids.is_empty());
        let rs = Variant::WithWalker.apply(
            &scene,
            &RoomSimConfig {
                duration_s: 2.0,
                ..RoomSimConfig::default()
            },
        );
        let setup = RoomSimSetup::new(&scene, &rs, &Config::default(), "SIM", &mut || true)
            .unwrap()
            .unwrap();
        let empty = setup.empty_room().unwrap();
        let mut stream = setup.stream().unwrap();
        let advance = stream.advance_to(f64::INFINITY).unwrap();
        let text = format!(
            "{setup:?} {empty:?} {stream:?} {advance:?} {:?}",
            setup.calibration()
        );
        for id in &ids {
            assert!(
                !text.contains(&format!("{id:?}")),
                "{name}: a scene id is printed"
            );
        }
    }
}

/// A setup can be built on one thread and streamed on another (the
/// gateway's session thread, spec/livedemo.md §3.0).
#[test]
fn setup_and_stream_are_send() {
    fn send_sync<T: Send + Sync>() {}
    fn send<T: Send>() {}
    send_sync::<RoomSimSetup>();
    send::<RoomSimStream<'static>>();
}
