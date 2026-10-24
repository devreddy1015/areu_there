//! The Room Sim pipeline (spec/locate.md §4.3, §10, §11), end to end on a
//! short bedroom run: labels, radii, reasons, tick times and truth, the
//! packet glue, and the refusals that name their parameter. Everything
//! here is SIM.
#![cfg(feature = "roomsim")]

use std::collections::BTreeMap;
use std::path::PathBuf;

use saans_locate::measure::CsiFrame;
use saans_locate::phone::Config;
use saans_locate::pipeline::{
    calibration_config, locate_config, next_tick, packet_frame, person_geometry, run_roomsim,
    tick_time, PersonLocator, PipelineError, RoomSimRun,
};
use saans_locate::radius::{radius95, Outcome, Reason};
use saans_roomsim::{Buffer, Config as RoomSimConfig, Role, Simulator};
use saans_scene::Scene;

fn bedroom() -> Scene {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes/bedroom.json");
    Scene::load(path).expect("the shipped bedroom loads")
}

/// The bedroom's sleeper plus one walker who carries the phone, 8 s.
fn short_config() -> RoomSimConfig {
    RoomSimConfig {
        seed: 7,
        duration_s: 8.0,
        n_people: Some(2),
        phone_carrier: 1,
        ..RoomSimConfig::default()
    }
}

const WLM_REASONS: [Reason; 2] = [Reason::NoDisturbance, Reason::ImplausibleRadius];
const RTI_REASONS: [Reason; 2] = [Reason::NotSignificant, Reason::ImplausibleRadius];
const PHONE_REASONS: [Reason; 6] = [
    Reason::NoReports,
    Reason::TooFewRanges,
    Reason::Degenerate,
    Reason::NotConverged,
    Reason::Gated,
    Reason::ImplausibleRadius,
];

/// Check one outcome against §8 and §11; whether it published.
fn check_outcome(out: &Outcome, allowed: &[Reason], confidence: bool, what: &str) -> bool {
    match out {
        Outcome::Estimate(e) => {
            assert!(
                !e.radius_calibrated,
                "{what}: radius_calibrated must be false"
            );
            assert!(
                e.radius_m.is_finite() && e.radius_m >= 0.3,
                "{what}: radius {} published",
                e.radius_m
            );
            assert!(e.x.is_finite() && e.y.is_finite(), "{what}: position");
            assert!(
                e.cov_xx.is_finite() && e.cov_xy.is_finite() && e.cov_yy.is_finite(),
                "{what}: covariance"
            );
            assert_eq!(e.confidence.is_some(), confidence, "{what}: confidence");
            true
        }
        Outcome::Abstain(r) => {
            assert!(allowed.contains(r), "{what}: reason {r} not allowed");
            false
        }
    }
}

fn run_short() -> (Scene, RoomSimConfig, RoomSimRun) {
    let scene = bedroom();
    let rs = short_config();
    // The default config: Room Sim's seed, tx_power_dbm and carrier_hz are
    // taken from the run (§9).
    let run = run_roomsim(&scene, &rs, &Config::default(), "SIM").expect("a valid run");
    (scene, rs, run)
}

#[test]
fn bedroom_short_run_records() {
    let (scene, rs, run) = run_short();
    let cfg = locate_config(&rs).unwrap();
    assert_eq!(
        (cfg.phone_seed, cfg.tx_power_dbm, cfg.carrier_hz),
        (Some(rs.seed), Some(rs.tx_power_dbm), Some(rs.carrier_hz()))
    );
    // The resolved defaults give the same run as the unset ones.
    assert_eq!(run_roomsim(&scene, &rs, &cfg, "SIM").unwrap(), run);

    assert_eq!(run.label, "SIM");
    assert!(run.has_phone);
    // §4.3: rate_hz = 1.0 / (float(nT) * slot_s); 4 transmitters.
    assert_eq!(run.rate_hz, 1.0 / (4.0 * rs.slot_s));
    // §4.1: max(3, int(round(0.5 * window_s * rate_hz))) = 10 in the bedroom.
    assert_eq!(run.min_samples, 10);
    assert_eq!(run.links.len(), 12);
    assert_eq!(run.links[0], (0, 1));
    assert_eq!(run.links[11], (3, 2));
    assert_eq!(run.calibration.baseline.len(), 12);
    assert!(run.calibration.calibrated_mask.iter().all(|&ok| ok));
    assert!(run.calibration.null_sigma.iter().all(|&s| s > 0.0));
    assert!(run.calibration.packets > 0);
    assert!(run.diagnostics.packets > 0);

    // §10.1: t_k = window_s + float(k) / tick_hz while t_k < duration_s.
    let want: Vec<f64> = (0..)
        .map(|k| tick_time(cfg.window_s, cfg.tick_hz, k))
        .take_while(|&t| t < rs.duration_s)
        .collect();
    assert_eq!(want.len(), 14);
    let got: Vec<f64> = run.ticks.iter().map(|t| t.t).collect();
    assert_eq!(got, want);

    let mut published = BTreeMap::<&str, usize>::new();
    for tick in &run.ticks {
        assert_eq!(tick.label, "SIM");
        assert_eq!(tick.truth.len(), 2);
        if check_outcome(&tick.wlm, &WLM_REASONS, false, "wlm") {
            *published.entry("wlm").or_default() += 1;
        }
        if check_outcome(&tick.rti, &RTI_REASONS, true, "rti") {
            *published.entry("rti").or_default() += 1;
        }
    }

    // §6.1: 5 Hz epochs over 8 s, one per fired epoch, each at its sample.
    assert_eq!(run.epochs.len(), 40);
    for ep in &run.epochs {
        assert_eq!(ep.label, "SIM");
        let ids: Vec<u8> = ep.reports.iter().map(|r| r.node).collect();
        assert_eq!(ids, vec![0, 1, 2, 3]);
        for r in &ep.reports {
            if let Some(v) = r.rssi_dbm {
                assert!((-100..=0).contains(&v));
            }
            if let Some(d) = r.rtt_m {
                assert!(d >= 0.0 && d.is_finite());
            }
        }
        if check_outcome(&ep.rtt, &PHONE_REASONS, false, "rtt") {
            *published.entry("rtt").or_default() += 1;
        }
        if check_outcome(&ep.rssi, &PHONE_REASONS, false, "rssi") {
            *published.entry("rssi").or_default() += 1;
        }
    }
    println!("SIM bedroom 8 s published per estimator: {published:?}");
    assert!(
        published.get("rtt").copied().unwrap_or(0) + published.get("rssi").copied().unwrap_or(0)
            > 0,
        "the phone published nothing in 40 epochs"
    );

    // §11: a tick's truth is Room Sim's truth of the last transmission
    // with t <= t_k; an epoch's truth is its sample's phone.
    let sim = Simulator::new(&scene, rs.clone()).unwrap();
    let truths: Vec<_> = sim.run().map(|tr| (tr.t, tr.truth)).collect();
    for tick in &run.ticks {
        let (_, truth) = truths.iter().rev().find(|(t, _)| *t <= tick.t).unwrap();
        for (rec, pt) in tick.truth.iter().zip(&truth.people) {
            assert_eq!((rec.x, rec.y), (pt.x, pt.y));
            assert_eq!(rec.walking, pt.walking);
            assert_eq!(rec.posture, pt.posture);
        }
    }
    for ep in &run.epochs {
        let (_, truth) = truths.iter().find(|(t, _)| *t == ep.t).unwrap();
        let p = truth.phone.unwrap();
        assert_eq!(ep.truth, [p.x, p.y, p.z]);
    }
}

/// §10.1 re-derived independently: the measurement holds exactly the
/// received packets with `t <= t_k` at tick k. Feed a second locator from
/// a flat packet list, tick by tick, and compare every outcome bit for bit.
#[test]
fn ticks_see_exactly_the_packets_up_to_t_k() {
    let (scene, rs, run) = run_short();
    let cfg = locate_config(&rs).unwrap();
    let sim = Simulator::new(&scene, rs.clone()).unwrap();
    let rate_hz = run.rate_hz;
    let fw = rs.fw_subcarriers;

    let mut person =
        PersonLocator::new(person_geometry(&scene, &cfg).unwrap(), &cfg, rate_hz, "SIM").unwrap();
    let calib = Simulator::new(&scene, calibration_config(&rs, cfg.calib_s)).unwrap();
    let cal: Vec<(f64, u8, u8, CsiFrame)> = calib
        .packets()
        .filter(|p| p.t >= rs.duration_s)
        .filter_map(|p| {
            packet_frame(&p, fw, rate_hz)
                .unwrap()
                .map(|f| (p.t, p.tx, p.rx, f))
        })
        .collect();
    assert_eq!(cal.len(), run.calibration.packets);
    person.calibrate_from(cal).unwrap();
    assert_eq!(
        person.measurement().baseline_vector(),
        run.calibration.baseline
    );

    let packets: Vec<(f64, u8, u8, CsiFrame)> = sim
        .packets()
        .filter_map(|p| {
            packet_frame(&p, fw, rate_hz)
                .unwrap()
                .map(|f| (p.t, p.tx, p.rx, f))
        })
        .collect();
    let mut next = 0;
    for tick in &run.ticks {
        while next < packets.len() && packets[next].0 <= tick.t {
            let (t, tx, rx, f) = &packets[next];
            person.update(*t, *tx, *rx, f).unwrap();
            next += 1;
        }
        let (wlm, rti) = person.tick(tick.t).unwrap();
        assert_eq!(wlm, tick.wlm, "wlm at t = {}", tick.t);
        assert_eq!(rti, tick.rti, "rti at t = {}", tick.t);
    }
}

#[test]
fn same_seed_same_records() {
    let (_, _, a) = run_short();
    let (_, _, b) = run_short();
    assert_eq!(a, b);
}

/// §4.3: amplitudes `sqrt(float(re*re + im*im))` over the first
/// `fw_subcarriers` int8 entries, phases 0, `sample_index = seq`,
/// `sample_rate_hz = rate_hz`, `rssi_dbm = rssi`, `node_id = rx`; a lost
/// packet has no frame.
#[test]
fn packet_glue() {
    let scene = bedroom();
    let rs = RoomSimConfig {
        duration_s: 0.5,
        loss_prob: 0.3,
        ..RoomSimConfig::default()
    };
    let sim = Simulator::new(&scene, rs.clone()).unwrap();
    let (mut lost, mut kept) = (0, 0);
    for p in sim.packets() {
        let frame = packet_frame(&p, rs.fw_subcarriers, 20.0).unwrap();
        let Some(r) = &p.received else {
            assert!(frame.is_none());
            lost += 1;
            continue;
        };
        let f = frame.unwrap();
        let Buffer::Int8(q) = &r.buffer else {
            panic!("quantize is on")
        };
        assert_eq!(f.amplitudes.len(), rs.fw_subcarriers);
        for (k, &(re, im)) in q[..rs.fw_subcarriers].iter().enumerate() {
            let want = (f64::from(re) * f64::from(re) + f64::from(im) * f64::from(im)).sqrt();
            assert_eq!(f.amplitudes[k], want);
        }
        assert!(f.phases.iter().all(|&ph| ph == 0.0));
        assert_eq!(f.phases.len(), f.amplitudes.len());
        assert_eq!(f.sample_index, u64::from(r.seq));
        assert_eq!(f.sample_rate_hz, 20.0);
        assert_eq!(f.rssi_dbm, r.rssi);
        assert_eq!(f.node_id, p.rx);
        kept += 1;
    }
    assert!(lost > 0 && kept > 0);
}

fn refusal(rs: &RoomSimConfig, scene: &Scene, label: &str) -> PipelineError {
    run_roomsim(scene, rs, &Config::default(), label).expect_err("must be refused")
}

#[test]
fn refusals_name_the_parameter() {
    let scene = bedroom();
    let base = RoomSimConfig {
        duration_s: 2.0,
        ..RoomSimConfig::default()
    };

    // Room Sim input is SIM, so the Room Sim pipeline takes only "SIM";
    // the given label is never echoed.
    for label in ["REAL", "SCENE", "sim", "SIM->REAL", "", "SIM "] {
        // The variant carries no data, so the label cannot be echoed.
        let e = refusal(&base, &scene, label);
        assert_eq!(e, PipelineError::Label);
        assert_eq!(e.param(), Some("label"));
        if !label.is_empty() {
            assert!(!e.to_string().contains(&format!("{label:?}")));
        }
    }

    // §2: any roles entry.
    let mut rs = base.clone();
    rs.roles.insert(1, Role::Txrx);
    assert_eq!(refusal(&rs, &scene, "SIM").param(), Some("roles"));
    rs.roles.insert(1, Role::Rx);
    assert_eq!(refusal(&rs, &scene, "SIM").param(), Some("roles"));

    // §4.3: no int8 buffer.
    let rs = RoomSimConfig {
        quantize: false,
        ..base.clone()
    };
    assert_eq!(refusal(&rs, &scene, "SIM").param(), Some("quantize"));

    // §4.1: node 1 every 3 superframes: floor(20/3) + 1 = 7 < 10.
    let mut rs = base.clone();
    rs.tx_period.insert(1, 3);
    assert_eq!(refusal(&rs, &scene, "SIM").param(), Some("tx_period"));
    // Every 2: floor(20/2) + 1 = 11 >= 10 runs.
    let mut rs = base.clone();
    rs.tx_period.insert(1, 2);
    rs.duration_s = 1.5;
    let cfg = locate_config(&rs).unwrap();
    assert!(run_roomsim(&scene, &rs, &cfg, "SIM").is_ok());

    // §2: two nodes at one (x, y), different z.
    let text = std::fs::read_to_string(
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes/bedroom.json"),
    )
    .unwrap();
    let stacked = text.replace(
        r#"{ "id": 3, "pos": [0.15, 3.45, 1.2] }"#,
        r#"{ "id": 3, "pos": [0.15, 3.45, 1.2] }, { "id": 4, "pos": [0.15, 3.45, 2.0] }"#,
    );
    assert_ne!(stacked, text);
    let stacked = Scene::from_json_str("stacked", &stacked).unwrap();
    // Room Sim itself accepts the scene; the person pipeline does not.
    assert!(Simulator::new(&stacked, base.clone()).is_ok());
    assert_eq!(refusal(&base, &stacked, "SIM").param(), Some("nodes"));

    // §9: phone_seed, tx_power_dbm and carrier_hz default to unset (then
    // Room Sim's); set, they must equal Room Sim's.
    let cfg = locate_config(&base).unwrap();
    for (rs, name) in [
        (
            RoomSimConfig {
                tx_power_dbm: 12.0,
                ..base.clone()
            },
            "tx_power_dbm",
        ),
        (
            RoomSimConfig {
                channel: 1,
                ..base.clone()
            },
            "carrier_hz",
        ),
        (
            RoomSimConfig {
                seed: base.seed + 1,
                ..base.clone()
            },
            "phone_seed",
        ),
    ] {
        let e = run_roomsim(&scene, &rs, &cfg, "SIM").unwrap_err();
        assert_eq!(e.param(), Some(name));
        // Unset, the same Room Sim config runs.
        assert!(run_roomsim(&scene, &rs, &Config::default(), "SIM").is_ok());
    }
    // The run-seed check is the pipeline's (as Python's run_locate):
    // Config::resolved keeps a set phone_seed for plain-data phone runs.
    let other = RoomSimConfig {
        seed: base.seed + 1,
        ..base.clone()
    };
    let e = saans_locate::pipeline::resolve_config(&other, &cfg).unwrap_err();
    assert_eq!(e.param(), Some("phone_seed"));
    let kept = cfg
        .resolved(other.seed, other.tx_power_dbm, other.carrier_hz())
        .unwrap();
    assert_eq!(kept.phone_seed, Some(base.seed));
    // A channel Room Sim refuses is Room Sim's refusal, not a panic.
    let rs = RoomSimConfig {
        channel: 0,
        ..base.clone()
    };
    assert_eq!(locate_config(&rs).unwrap_err().param(), Some("channel"));
    let e = run_roomsim(&scene, &rs, &Config::default(), "SIM").unwrap_err();
    assert_eq!(e.param(), Some("channel"));
    let e = run_roomsim(&scene, &rs, &cfg, "SIM").unwrap_err();
    assert_eq!(e.param(), Some("channel"));

    // §9: an unknown link signal, min_windows 0, a §9 parameter.
    let refused_cfg = |cfg: &Config| run_roomsim(&scene, &base, cfg, "SIM").unwrap_err();
    let e = refused_cfg(&Config {
        link_signal: "csi_magic".into(),
        ..Config::default()
    });
    assert_eq!(e.param(), Some("link_signal"));
    let e = refused_cfg(&Config {
        min_windows: 0,
        ..Config::default()
    });
    assert_eq!(e.param(), Some("min_windows"));
    for (cfg, name) in [
        (
            Config {
                tick_hz: f64::NAN,
                ..Config::default()
            },
            "tick_hz",
        ),
        (
            Config {
                min_plausible_radius_m: 0.0,
                ..Config::default()
            },
            "min_plausible_radius_m",
        ),
        (
            Config {
                rti_radius_scale: 0.0,
                ..Config::default()
            },
            "rti_radius_scale",
        ),
        (
            Config {
                carrier_hz: Some(-1.0),
                ..Config::default()
            },
            "carrier_hz",
        ),
    ] {
        assert_eq!(refused_cfg(&cfg).param(), Some(name));
    }

    // §9: at most one tick per Room Sim slot (tick_hz <= 1.0 / slot_s).
    for tick_hz in [1.0 / base.slot_s + 1.0, 1e9, 1e300] {
        let e = refused_cfg(&Config {
            tick_hz,
            ..Config::default()
        });
        assert_eq!(e.param(), Some("tick_hz"), "{tick_hz}");
    }

    // §4.2: a calibration that raises (2 s cannot give 10 windows).
    let e = refused_cfg(&Config {
        calib_s: 2.0,
        ..Config::default()
    });
    assert_eq!(e.param(), Some("calib_s"));
}

/// §10.1: a run in which some `t_{k+1}` is not greater than `t_k` is
/// refused naming `tick_hz` (with `tick_hz <= 1.0 / slot_s` a Room Sim run
/// cannot reach this, so the rule is checked directly).
#[test]
fn ticks_that_stop_advancing_are_refused() {
    assert_eq!(next_tick(1.0, 2.0, 0, 1.0).unwrap(), 1.5);
    assert_eq!(next_tick(1.0, 2.0, 4, tick_time(1.0, 2.0, 4)).unwrap(), 3.5);
    // float(k) / tick_hz below half an ulp of window_s: t_1 == t_0.
    let t0 = tick_time(1.0, 1e300, 0);
    let e = next_tick(1.0, 1e300, 0, t0).unwrap_err();
    assert_eq!(e.param(), Some("tick_hz"));
    let t0 = tick_time(1e17, 2.0, 0);
    assert_eq!(
        next_tick(1e17, 2.0, 0, t0).unwrap_err().param(),
        Some("tick_hz")
    );
}

/// §1: a person pipeline fed by a caller takes its label from the caller
/// and accepts "SIM", "REAL" or "SIM->REAL"; SCENE (geometry) or anything
/// else is refused naming `label`, without echoing it.
#[test]
fn person_labels() {
    let scene = bedroom();
    let rs = short_config();
    let cfg = locate_config(&rs).unwrap();
    let build =
        |label: &str| PersonLocator::new(person_geometry(&scene, &cfg).unwrap(), &cfg, 20.0, label);
    for ok in ["SIM", "REAL", "SIM->REAL"] {
        assert_eq!(build(ok).unwrap().label(), ok);
    }
    for bad in ["SCENE", "sim", "", "LIVE", "SIM "] {
        let e = build(bad).unwrap_err();
        assert_eq!(e.param(), Some("label"), "{bad:?}");
        let msg = e.to_string();
        assert!(msg.starts_with("label"), "{msg}");
        if bad == "SCENE" || bad == "LIVE" {
            assert!(!msg.contains(bad), "{msg}");
        }
    }
}

/// A locator fed the run's packets tick by tick (as run_roomsim feeds its
/// own): each tick's `(wlm, rti, tracker is tracking)`.
fn locator_ticks(scene: &Scene, rs: &RoomSimConfig, cfg: &Config) -> Vec<(Outcome, Outcome, bool)> {
    let cfg = &saans_locate::pipeline::resolve_config(rs, cfg).unwrap();
    let sim = Simulator::new(scene, rs.clone()).unwrap();
    let rate_hz = saans_locate::pipeline::base_rate_hz(&sim);
    let fw = rs.fw_subcarriers;
    let mut person =
        PersonLocator::new(person_geometry(scene, cfg).unwrap(), cfg, rate_hz, "SIM").unwrap();
    let calib = Simulator::new(scene, calibration_config(rs, cfg.calib_s)).unwrap();
    let cal: Vec<(f64, u8, u8, CsiFrame)> = calib
        .packets()
        .filter(|p| p.t >= rs.duration_s)
        .filter_map(|p| {
            packet_frame(&p, fw, rate_hz)
                .unwrap()
                .map(|f| (p.t, p.tx, p.rx, f))
        })
        .collect();
    person.calibrate_from(cal).unwrap();
    let packets: Vec<(f64, u8, u8, CsiFrame)> = sim
        .packets()
        .filter_map(|p| {
            packet_frame(&p, fw, rate_hz)
                .unwrap()
                .map(|f| (p.t, p.tx, p.rx, f))
        })
        .collect();
    let mut out = Vec::new();
    let mut next = 0;
    let mut k = 0;
    let mut t_k = tick_time(cfg.window_s, cfg.tick_hz, k);
    while t_k < rs.duration_s {
        while next < packets.len() && packets[next].0 <= t_k {
            let (t, tx, rx, f) = &packets[next];
            person.update(*t, *tx, *rx, f).unwrap();
            next += 1;
        }
        let (wlm, rti) = person.tick(t_k).unwrap();
        out.push((wlm, rti, person.tracker().is_tracking()));
        k += 1;
        t_k = tick_time(cfg.window_s, cfg.tick_hz, k);
    }
    out
}

/// Each `*_radius_scale` scales only its own estimator: with the three
/// distinct, every published radius is its own scale times radius95 of
/// its covariance, and the positions are those of the unscaled run.
/// (`z_threshold` 0.5 so that RTI publishes in this short run.)
#[test]
fn radius_scales_are_wired_to_their_own_estimators() {
    let scene = bedroom();
    let rs = short_config();
    let low_z = Config {
        z_threshold: 0.5,
        ..Config::default()
    };
    let scaled = Config {
        wlm_radius_scale: 1.5,
        rti_radius_scale: 2.0,
        phone_radius_scale: 0.75,
        ..low_z.clone()
    };
    let a = run_roomsim(&scene, &rs, &scaled, "SIM").unwrap();
    let b = run_roomsim(&scene, &rs, &low_z, "SIM").unwrap();
    let r95 = |e: &saans_locate::radius::Estimate| radius95(e.cov_xx, e.cov_xy, e.cov_yy);
    let mut n = [0usize; 4];
    for (ta, tb) in a.ticks.iter().zip(&b.ticks) {
        for (k, oa, ob, scale) in [(0, &ta.wlm, &tb.wlm, 1.5), (1, &ta.rti, &tb.rti, 2.0)] {
            if let (Some(ea), Some(eb)) = (oa.estimate(), ob.estimate()) {
                assert_eq!(ea.radius_m, scale * r95(ea));
                assert_eq!(eb.radius_m, r95(eb));
                assert_eq!((ea.x, ea.y, ea.cov_xx), (eb.x, eb.y, eb.cov_xx));
                n[k] += 1;
            }
        }
    }
    for (ea, eb) in a.epochs.iter().zip(&b.epochs) {
        for (oa, ob) in [(&ea.rtt, &eb.rtt), (&ea.rssi, &eb.rssi)] {
            if let (Some(x), Some(y)) = (oa.estimate(), ob.estimate()) {
                assert_eq!(x.radius_m, 0.75 * r95(x));
                assert_eq!(y.radius_m, r95(y));
                assert_eq!((x.x, x.y), (y.x, y.y));
                n[2] += 1;
            }
        }
        n[3] += 1;
    }
    println!(
        "SIM bedroom 8 s, scales 1.5/2.0/0.75: compared WLM {}, RTI {}, phone {} estimates",
        n[0], n[1], n[2]
    );
    assert!(n[0] > 0 && n[1] > 0 && n[2] > 0, "{n:?}");
}

/// §10.1: RTI abstains `implausible_radius` when its radius fails §8, and
/// the tracker keeps its corrected state (the same tracking as a run that
/// published).
#[test]
fn rti_implausible_radius_keeps_the_corrected_tracker() {
    let scene = bedroom();
    let rs = short_config();
    let low_z = Config {
        z_threshold: 0.5,
        ..Config::default()
    };
    let huge_min = Config {
        min_plausible_radius_m: 100.0,
        ..low_z.clone()
    };
    let a = locator_ticks(&scene, &rs, &low_z);
    let b = locator_ticks(&scene, &rs, &huge_min);
    let mut implausible = 0;
    for ((_, ra, ta), (_, rb, tb)) in a.iter().zip(&b) {
        assert_eq!(ta, tb, "the tracker is corrected whatever is published");
        match (ra, rb) {
            (Outcome::Estimate(_), Outcome::Abstain(Reason::ImplausibleRadius)) => {
                implausible += 1;
                assert!(*tb);
            }
            (Outcome::Abstain(x), Outcome::Abstain(y)) => assert_eq!(x, y),
            other => panic!("{other:?}"),
        }
    }
    assert!(implausible > 0);
}

#[test]
fn no_person_or_phone_id_in_errors_or_records() {
    let scene = bedroom();
    // The bedroom's ids are "sleeper" and "phone"; neither is ever echoed.
    let cfg = Config {
        calib_s: 2.0,
        ..Config::default()
    };
    let rs = RoomSimConfig {
        duration_s: 2.0,
        ..RoomSimConfig::default()
    };
    let e = run_roomsim(&scene, &rs, &cfg, "SIM").unwrap_err();
    assert!(!e.to_string().contains("sleeper"));
    let (_, _, run) = run_short();
    let text = format!("{run:?}");
    assert!(!text.contains("sleeper"));
    assert!(!text.contains("\"phone\""));
}

/// §10.2 step 1: the same config and seed, nobody in the room, the
/// people maps emptied, `duration_s + calib_s`; no seed arithmetic.
#[test]
fn calibration_run_config() {
    let mut rs = short_config();
    rs.people_mode.insert(1, saans_roomsim::Mode::Walk);
    rs.start_heading_rad.insert(0, 1.0);
    let c = calibration_config(&rs, 15.0);
    assert_eq!(c.seed, rs.seed);
    assert_eq!(c.n_people, Some(0));
    assert!(c.people_mode.is_empty());
    assert!(c.people_posture.is_empty());
    assert!(c.start_heading_rad.is_empty());
    assert_eq!(c.duration_s, rs.duration_s + 15.0);
    let back = RoomSimConfig {
        n_people: rs.n_people,
        people_mode: rs.people_mode.clone(),
        start_heading_rad: rs.start_heading_rad.clone(),
        duration_s: rs.duration_s,
        ..c
    };
    assert_eq!(back, rs);
}
