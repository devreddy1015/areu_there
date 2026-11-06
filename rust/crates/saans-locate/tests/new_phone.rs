//! spec/locate.md §6 (phone radio), §7 (fixes, Kalman filter, published
//! estimate) and §9 (configuration): validation, the §12a checks with
//! known geometry, the draw order, the epoch rule, NaN fail-closed paths.
//! Every number here is SIM.

use std::path::PathBuf;

use saans_locate::num::{chi2_2_99, k95, len3};
use saans_locate::phone::{
    anchor_normal, anchors_collinear, blocked, box_blocks, cylinder_blocks, rssi_fix, rssi_start,
    rtt_fix, rtt_start, simulate, wall_blocks, Blockers, BoxObj, Config, Cylinder, EpochRecord,
    Error, Fix, Method, PhoneFilter, PhoneNode, PhoneRun, Radio, RadioEpoch, Ranging, ReportRecord,
    TruthSample, Wall, GN_MAX_ITERATIONS, LINK_SIGNALS, RSSI_STREAM_BASE, RTT_STREAM_BASE,
};
use saans_locate::radius::{radius95, Outcome, Reason};
use saans_sim::rng::Rng;

fn scene_nodes(name: &str) -> Vec<PhoneNode> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../spec/scenes")
        .join(name);
    let scene = saans_scene::Scene::load(path).unwrap();
    let mut nodes: Vec<PhoneNode> = scene
        .nodes
        .iter()
        .map(|n| PhoneNode {
            id: n.id,
            x: n.pos.0,
            y: n.pos.1,
            z: n.pos.2,
        })
        .collect();
    nodes.sort_by_key(|n| n.id);
    nodes
}

fn exact_ranges(nodes: &[PhoneNode], p: [f64; 3]) -> Vec<Ranging> {
    nodes
        .iter()
        .map(|n| Ranging {
            node: *n,
            value: len3(p[0] - n.x, p[1] - n.y, p[2] - n.z),
        })
        .collect()
}

const RTT_NOT_CONVERGED: [f64; 4] = [0.0, 2.2, 0.65, 2.4];
const RSSI_NOT_CONVERGED: [f64; 4] = [-41.0, -70.0, -58.0, -45.0];

fn dist2(a: (f64, f64), b: (f64, f64)) -> f64 {
    ((a.0 - b.0) * (a.0 - b.0) + (a.1 - b.1) * (a.1 - b.1)).sqrt()
}

/// The §9 defaults resolved as Room Sim's defaults would resolve them
/// (seed 0, 10 dBm, channel 6): what the phone needs.
fn base() -> Config {
    Config::default().resolved(0, 10.0, 2437e6).unwrap()
}

/// The outer `Ok` of a fix (its input was valid).
fn ok_fix(got: Result<Result<Fix, Reason>, Error>) -> Result<Fix, Reason> {
    got.expect("valid reports and config")
}

fn field_of(e: &Error) -> &'static str {
    match e {
        Error::Config { param, .. } => param,
        Error::Input { field, .. } => field,
        Error::Label => "label",
    }
}

// =====================================================================
// §9 validation and the label
// =====================================================================

fn param_of(cfg: &Config) -> &'static str {
    match cfg.validate() {
        Err(Error::Config { param, .. }) => param,
        other => panic!("expected a config error, got {other:?}"),
    }
}

#[test]
fn defaults_are_the_spec_table_and_valid() {
    let c = Config::default();
    c.validate().unwrap();
    assert_eq!(
        (c.lambda_m, c.pitch_m, c.alpha, c.sigma, c.delta_c),
        (0.3, 0.5, 1.0, 1.0, 0.5)
    );
    assert_eq!((c.z_threshold, c.window_s, c.min_windows), (5.0, 1.0, 10));
    assert_eq!(c.link_signal, "rssi_dbm");
    assert_eq!((c.calib_s, c.tick_hz, c.wlm_gate_z), (15.0, 2.0, 3.0));
    assert_eq!(c.wlm_floor, None);
    assert_eq!(
        (c.wlm_radius_scale, c.rti_radius_scale, c.phone_radius_scale),
        (1.0, 1.0, 1.0)
    );
    assert_eq!((c.min_plausible_radius_m, c.phone_rate_hz), (0.3, 5.0));
    assert_eq!(
        (c.rtt_sigma_m, c.rtt_nlos_sigma_scale, c.rtt_nlos_bias_m),
        (1.0, 1.5, 1.0)
    );
    assert_eq!(
        (c.rtt_outlier_p, c.rtt_outlier_lo_m, c.rtt_outlier_hi_m),
        (0.05, 2.0, 8.0)
    );
    assert_eq!((c.rtt_offset_sigma_m, c.rtt_persist_sigma_m), (0.5, 0.5));
    assert_eq!((c.rtt_fail_p, c.rssi_fail_p), (0.02, 0.02));
    assert_eq!((c.rssi_breakpoint_m, c.rssi_far_exponent), (5.0, 3.5));
    assert_eq!(
        (c.rssi_shadow_sigma_db, c.rssi_persist_sigma_db),
        (4.0, 4.0)
    );
    assert_eq!((c.rssi_shadow_corr_m, c.rssi_nlos_loss_db), (6.0, 5.0));
    assert_eq!((c.rssi_device_sigma_db, c.rssi_jitter_db), (3.0, 1.0));
    assert_eq!((c.rssi_est_exponent, c.rssi_p0_dbm), (2.0, None));
    assert_eq!((c.phone_assumed_z_m, c.huber_k), (1.0, 1.345));
    assert_eq!(
        (
            c.phone_accel_psd,
            c.phone_init_speed_std,
            c.phone_max_coast_s
        ),
        (1.0, 1.0, 3.0)
    );
    // Room Sim's values default to unset (§9), then come from Room Sim.
    assert_eq!(
        (c.tx_power_dbm, c.carrier_hz, c.phone_seed),
        (None, None, None)
    );
    let r = base();
    assert_eq!(
        (r.tx_power_dbm, r.carrier_hz, r.phone_seed),
        (Some(10.0), Some(2437e6), Some(0))
    );
    // rssi_p0_dbm defaults to the true 1 m level: tx_power - FSPL(1 m).
    assert_eq!(r.rssi_p0().unwrap(), 10.0 - r.pl0_db().unwrap());
    let pl0 = r.pl0_db().unwrap();
    assert!((pl0 - 40.185).abs() < 1e-3, "{pl0}");
}

#[test]
fn room_sim_values_are_unset_until_resolved_and_must_match() {
    let nodes = scene_nodes("bedroom.json");
    let b = Blockers::default();
    // The phone needs them: unset, it refuses naming the first unset.
    let seed_only = Config {
        phone_seed: Some(1),
        ..Config::default()
    };
    let no_carrier = Config {
        tx_power_dbm: Some(10.0),
        ..seed_only.clone()
    };
    for (cfg, name) in [
        (Config::default(), "phone_seed"),
        (seed_only, "tx_power_dbm"),
        (no_carrier, "carrier_hz"),
    ] {
        cfg.validate().unwrap();
        assert_eq!(
            field_of(&Radio::new(&cfg, &nodes, &b, "SIM").unwrap_err()),
            name
        );
        assert_eq!(
            field_of(&PhoneRun::new(&cfg, &nodes, &b, "SIM").unwrap_err()),
            name
        );
    }
    // Unset: taken from Room Sim.
    let r = Config::default().resolved(7, 12.0, 2412e6).unwrap();
    assert_eq!(
        (r.phone_seed, r.tx_power_dbm, r.carrier_hz),
        (Some(7), Some(12.0), Some(2412e6))
    );
    // Set and equal: kept.
    let same = Config {
        phone_seed: Some(7),
        tx_power_dbm: Some(12.0),
        carrier_hz: Some(2412e6),
        ..Config::default()
    };
    assert_eq!(same.resolved(7, 12.0, 2412e6).unwrap(), r);
    // A set phone_seed is kept whatever the seed argument (a plain-data
    // phone run chooses its own; only the Room Sim pipeline refuses one
    // that is not the run's seed, pipeline_smoke).
    let own = Config {
        phone_seed: Some(8),
        ..Config::default()
    };
    let kept = own.resolved(7, 12.0, 2412e6).unwrap();
    assert_eq!(kept.phone_seed, Some(8));
    assert_eq!(
        kept,
        Config {
            phone_seed: Some(8),
            ..r.clone()
        }
    );
    // rssi_p0_dbm stays unset and is derived where used.
    assert_eq!(kept.rssi_p0_dbm, None);
    assert_eq!(kept.rssi_p0().unwrap(), 12.0 - kept.pl0_db().unwrap());
    // Set tx_power_dbm or carrier_hz different from Room Sim's: refused
    // naming it.
    let cases: [(Config, &str); 2] = [
        (
            Config {
                tx_power_dbm: Some(10.0),
                ..Config::default()
            },
            "tx_power_dbm",
        ),
        (
            Config {
                carrier_hz: Some(2437e6),
                ..Config::default()
            },
            "carrier_hz",
        ),
    ];
    for (cfg, name) in cases {
        assert_eq!(field_of(&cfg.resolved(7, 12.0, 2412e6).unwrap_err()), name);
    }
    // Room Sim's values are validated too.
    for (tx, fc, name) in [
        (f64::NAN, 2412e6, "tx_power_dbm"),
        (10.0, 0.0, "carrier_hz"),
        (10.0, f64::NAN, "carrier_hz"),
    ] {
        assert_eq!(
            field_of(&Config::default().resolved(0, tx, fc).unwrap_err()),
            name
        );
    }
    // The derived values need them.
    assert_eq!(
        field_of(&Config::default().rssi_p0().unwrap_err()),
        "rssi_p0_dbm"
    );
    assert_eq!(
        field_of(&Config::default().pl0_db().unwrap_err()),
        "carrier_hz"
    );
    let p0 = Config {
        rssi_p0_dbm: Some(-40.0),
        ..Config::default()
    };
    assert_eq!(p0.rssi_p0().unwrap(), -40.0);
    // The phone needs only phone_seed, tx_power_dbm and carrier_hz: with
    // rssi_p0_dbm unset (and no Config::resolved) it runs, deriving the
    // 1 m level (§9), and gives the same records as the resolved config.
    let plain = Config {
        phone_seed: Some(3),
        tx_power_dbm: Some(10.0),
        carrier_hz: Some(2437e6),
        ..Config::default()
    };
    plain.require_resolved().unwrap();
    assert_eq!(plain.rssi_p0_dbm, None);
    let resolved = plain.resolved(0, 10.0, 2437e6).unwrap();
    assert_eq!(resolved, plain);
    let mut a = PhoneRun::new(&plain, &nodes, &b, "SIM").unwrap();
    let mut c = PhoneRun::new(&base(), &nodes, &b, "SIM").unwrap();
    let mut d = PhoneRun::new(
        &Config {
            phone_seed: Some(3),
            ..base()
        },
        &nodes,
        &b,
        "SIM",
    )
    .unwrap();
    let (mut fired, mut differs) = (0, false);
    for k in 0..40 {
        let s = sample(f64::from(k) * 0.1, [1.0 + 0.05 * f64::from(k), 2.0, 1.2]);
        let ra = a.step(&s).unwrap();
        let rd = d.step(&s).unwrap();
        assert_eq!(ra, rd);
        if let (Some(ra), Some(rc)) = (ra, c.step(&s).unwrap()) {
            fired += 1;
            differs |= ra != rc;
        }
    }
    assert!(fired > 10);
    assert!(differs, "phone_seed 3 is not seed 0");
}

#[test]
fn validation_names_the_parameter() {
    type Set = fn(&mut Config, f64);
    let positive: [(&str, Set); 18] = [
        ("rtt_sigma_m", |c, v| c.rtt_sigma_m = v),
        ("rssi_shadow_sigma_db", |c, v| c.rssi_shadow_sigma_db = v),
        ("rssi_shadow_corr_m", |c, v| c.rssi_shadow_corr_m = v),
        ("rssi_breakpoint_m", |c, v| c.rssi_breakpoint_m = v),
        ("phone_rate_hz", |c, v| c.phone_rate_hz = v),
        ("huber_k", |c, v| c.huber_k = v),
        ("rssi_far_exponent", |c, v| c.rssi_far_exponent = v),
        ("rssi_est_exponent", |c, v| c.rssi_est_exponent = v),
        ("phone_accel_psd", |c, v| c.phone_accel_psd = v),
        ("phone_init_speed_std", |c, v| c.phone_init_speed_std = v),
        ("tick_hz", |c, v| c.tick_hz = v),
        ("calib_s", |c, v| c.calib_s = v),
        ("window_s", |c, v| c.window_s = v),
        // A zero radius would publish a bare point (ADR-008 d5).
        ("min_plausible_radius_m", |c, v| {
            c.min_plausible_radius_m = v
        }),
        ("wlm_radius_scale", |c, v| c.wlm_radius_scale = v),
        ("rti_radius_scale", |c, v| c.rti_radius_scale = v),
        ("phone_radius_scale", |c, v| c.phone_radius_scale = v),
        // The guard of LAMBDA = C / carrier_hz.
        ("carrier_hz", |c, v| c.carrier_hz = Some(v)),
    ];
    for (name, set) in positive {
        for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut c = Config::default();
            set(&mut c, bad);
            assert_eq!(param_of(&c), name, "{name} = {bad}");
        }
        let mut c = Config::default();
        set(&mut c, 1e-9);
        c.validate().unwrap();
    }
    let non_negative: [(&str, Set); 11] = [
        ("rtt_nlos_sigma_scale", |c, v| c.rtt_nlos_sigma_scale = v),
        ("rtt_nlos_bias_m", |c, v| c.rtt_nlos_bias_m = v),
        ("rtt_offset_sigma_m", |c, v| c.rtt_offset_sigma_m = v),
        ("rtt_persist_sigma_m", |c, v| c.rtt_persist_sigma_m = v),
        ("rssi_nlos_loss_db", |c, v| c.rssi_nlos_loss_db = v),
        ("rssi_device_sigma_db", |c, v| c.rssi_device_sigma_db = v),
        ("rssi_persist_sigma_db", |c, v| c.rssi_persist_sigma_db = v),
        ("rssi_jitter_db", |c, v| c.rssi_jitter_db = v),
        ("phone_max_coast_s", |c, v| c.phone_max_coast_s = v),
        ("wlm_gate_z", |c, v| c.wlm_gate_z = v),
        ("wlm_floor", |c, v| c.wlm_floor = Some(v)),
    ];
    for (name, set) in non_negative {
        for bad in [-1e-12, f64::NAN, f64::NEG_INFINITY] {
            let mut c = Config::default();
            set(&mut c, bad);
            assert_eq!(param_of(&c), name, "{name} = {bad}");
        }
        let mut c = Config::default();
        set(&mut c, 0.0);
        c.validate().unwrap();
    }
    let unit: [(&str, Set); 3] = [
        ("rtt_outlier_p", |c, v| c.rtt_outlier_p = v),
        ("rtt_fail_p", |c, v| c.rtt_fail_p = v),
        ("rssi_fail_p", |c, v| c.rssi_fail_p = v),
    ];
    for (name, set) in unit {
        for bad in [-0.1, 1.0000001, f64::NAN] {
            let mut c = Config::default();
            set(&mut c, bad);
            assert_eq!(param_of(&c), name, "{name} = {bad}");
        }
        for ok in [0.0, 1.0] {
            let mut c = Config::default();
            set(&mut c, ok);
            c.validate().unwrap();
        }
    }
    let finite_only: [(&str, Set); 11] = [
        ("lambda_m", |c, v| c.lambda_m = v),
        ("pitch_m", |c, v| c.pitch_m = v),
        ("alpha", |c, v| c.alpha = v),
        ("sigma", |c, v| c.sigma = v),
        ("delta_c", |c, v| c.delta_c = v),
        ("z_threshold", |c, v| c.z_threshold = v),
        ("rssi_p0_dbm", |c, v| c.rssi_p0_dbm = Some(v)),
        ("phone_assumed_z_m", |c, v| c.phone_assumed_z_m = v),
        ("tx_power_dbm", |c, v| c.tx_power_dbm = Some(v)),
        ("rtt_outlier_lo_m", |c, v| c.rtt_outlier_lo_m = v),
        ("rtt_outlier_hi_m", |c, v| c.rtt_outlier_hi_m = v),
    ];
    for (name, set) in finite_only {
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            let mut c = Config::default();
            set(&mut c, bad);
            assert_eq!(param_of(&c), name, "{name} = {bad}");
        }
    }
    let mut c = Config {
        rtt_outlier_lo_m: 8.5,
        ..Config::default()
    };
    assert_eq!(param_of(&c), "rtt_outlier_lo_m");
    c.rtt_outlier_lo_m = 8.0;
    c.validate().unwrap();
    // min_windows an integer >= 1.
    let mut c = Config {
        min_windows: 0,
        ..Config::default()
    };
    assert_eq!(param_of(&c), "min_windows");
    c.min_windows = 1;
    c.validate().unwrap();
    // link_signal one of the three names.
    for bad in ["csi_magic", "", "RSSI_DBM", "rssi_dbm "] {
        let c = Config {
            link_signal: bad.into(),
            ..Config::default()
        };
        assert_eq!(param_of(&c), "link_signal", "{bad:?}");
    }
    for ok in LINK_SIGNALS {
        let c = Config {
            link_signal: ok.into(),
            ..Config::default()
        };
        c.validate().unwrap();
    }
    // Room Sim's values may be unset (they are then Room Sim's) or set.
    Config::default().validate().unwrap();
    base().validate().unwrap();
    // The message names the parameter.
    let c = Config {
        huber_k: f64::NAN,
        ..Config::default()
    };
    assert!(c.validate().unwrap_err().to_string().starts_with("huber_k"));
}

#[test]
fn phone_functions_refuse_any_label_but_sim() {
    let cfg = base();
    let nodes = scene_nodes("bedroom.json");
    let b = Blockers::default();
    for label in ["REAL", "sim", "SIM->REAL", "", "SIM "] {
        assert_eq!(
            Radio::new(&cfg, &nodes, &b, label).unwrap_err(),
            Error::Label
        );
        assert_eq!(
            PhoneRun::new(&cfg, &nodes, &b, label).unwrap_err(),
            Error::Label
        );
        assert_eq!(
            simulate(&cfg, &nodes, &b, label, &[]).unwrap_err(),
            Error::Label
        );
    }
    // The error never repeats the label it was given.
    let msg = Error::Label.to_string();
    assert!(!msg.contains("REAL"));
    assert!(Radio::new(&cfg, &nodes, &b, "SIM").is_ok());
    // Config errors come before nothing else: a bad config is named.
    let mut bad = cfg.clone();
    bad.phone_rate_hz = 0.0;
    assert!(matches!(
        Radio::new(&bad, &nodes, &b, "SIM").unwrap_err(),
        Error::Config {
            param: "phone_rate_hz",
            ..
        }
    ));
}

#[test]
fn inputs_are_checked() {
    let cfg = base();
    let b = Blockers::default();
    let n = |id, x| PhoneNode {
        id,
        x,
        y: 0.0,
        z: 1.0,
    };
    // §6.1: nodes are taken in id order (a sorted copy) ...
    let radio = Radio::new(&cfg, &[n(1, 0.0), n(0, 1.0)], &b, "SIM").unwrap();
    assert_eq!(
        radio.nodes().iter().map(|x| x.id).collect::<Vec<_>>(),
        vec![0, 1]
    );
    // ... with distinct ids, at least one, and finite positions.
    for nodes in [
        vec![n(0, 0.0), n(0, 1.0)],
        vec![n(2, 0.0), n(1, 1.0), n(2, 2.0)],
        vec![],
        vec![n(0, f64::NAN)],
        vec![n(0, 0.0), n(1, f64::INFINITY)],
    ] {
        assert!(matches!(
            Radio::new(&cfg, &nodes, &b, "SIM").unwrap_err(),
            Error::Input { field: "nodes", .. }
        ));
    }
    // Unsorted nodes give the same radio as sorted ones.
    let mut a = Radio::new(&cfg, &[n(1, 0.0), n(0, 1.0)], &b, "SIM").unwrap();
    let mut c = Radio::new(&cfg, &[n(0, 1.0), n(1, 0.0)], &b, "SIM").unwrap();
    let s0 = TruthSample {
        t: 0.0,
        phone: [2.0, 1.0, 1.0],
        cylinders: &[],
    };
    assert_eq!(a.step(&s0).unwrap(), c.step(&s0).unwrap());
    let bad_box = Blockers {
        walls: vec![],
        boxes: vec![BoxObj::from_yaw_deg(
            1.0,
            1.0,
            0.5,
            f64::NAN,
            1.0,
            1.0,
            0.0,
            false,
        )],
    };
    assert!(matches!(
        Radio::new(&cfg, &[n(0, 0.0)], &bad_box, "SIM").unwrap_err(),
        Error::Input {
            field: "objects",
            ..
        }
    ));
    let bad_wall = Blockers {
        walls: vec![Wall {
            a: (0.0, 0.0),
            b: (1.0, f64::NAN),
            h: 2.0,
        }],
        boxes: vec![],
    };
    assert!(matches!(
        Radio::new(&cfg, &[n(0, 0.0)], &bad_wall, "SIM").unwrap_err(),
        Error::Input { field: "walls", .. }
    ));
    let mut radio = Radio::new(&cfg, &[n(0, 0.0)], &b, "SIM").unwrap();
    let s = |t| TruthSample {
        t,
        phone: [1.0, 1.0, 1.0],
        cylinders: &[],
    };
    assert!(radio.step(&s(0.5)).unwrap().is_some());
    for t in [f64::NAN, f64::INFINITY, 0.4] {
        assert!(matches!(
            radio.step(&s(t)).unwrap_err(),
            Error::Input { field: "t", .. }
        ));
    }
    let bad_phone = TruthSample {
        t: 0.6,
        phone: [1.0, f64::NAN, 1.0],
        cylinders: &[],
    };
    assert!(matches!(
        radio.step(&bad_phone).unwrap_err(),
        Error::Input { field: "phone", .. }
    ));
    let cyl = [Cylinder {
        cx: 1.0,
        cy: 1.0,
        radius: f64::NAN,
        top: 1.0,
    }];
    let bad_cyl = TruthSample {
        t: 0.6,
        phone: [1.0, 1.0, 1.0],
        cylinders: &cyl,
    };
    assert!(matches!(
        radio.step(&bad_cyl).unwrap_err(),
        Error::Input {
            field: "cylinders",
            ..
        }
    ));
    // A phone so far away that its distance to a node is not finite.
    let next = radio.next_epoch();
    let far = TruthSample {
        t: 0.6,
        phone: [1e200, 0.0, 1.0],
        cylinders: &[],
    };
    assert!(matches!(
        radio.step(&far).unwrap_err(),
        Error::Input { field: "phone", .. }
    ));
    // t * phone_rate_hz must stay below 2^53.
    let huge = TruthSample {
        t: 1e16,
        phone: [1.0, 1.0, 1.0],
        cylinders: &[],
    };
    assert!(matches!(
        radio.step(&huge).unwrap_err(),
        Error::Input { field: "t", .. }
    ));
    // A refused sample changes nothing: the same time then fires as usual.
    assert_eq!(radio.next_epoch(), next);
    assert!(radio.step(&s(0.6)).unwrap().is_some());
    // An equal time is in order (non-decreasing); it fires nothing here.
    assert!(radio.step(&s(0.6)).unwrap().is_none());
}

// =====================================================================
// §6.2 line of sight (§12a)
// =====================================================================

#[test]
fn line_of_sight_known_geometry() {
    // A wall centreline x = 2 from y = 0 to 4, 2.7 m high.
    let wall = Wall {
        a: (2.0, 0.0),
        b: (2.0, 4.0),
        h: 2.7,
    };
    let p = [1.0, 2.0, 1.0];
    let n = [3.0, 2.2, 1.2];
    assert!(wall_blocks(p, n, &wall), "through the centreline");
    assert!(wall_blocks(n, p, &wall), "and back");
    assert!(!wall_blocks(p, [1.8, 2.2, 1.2], &wall), "short of it");
    assert!(
        !wall_blocks([1.0, 2.0, 2.8], [3.0, 2.0, 3.0], &wall),
        "above its height"
    );
    assert!(
        !wall_blocks([1.0, 5.0, 1.0], [3.0, 5.0, 1.0], &wall),
        "past its end"
    );
    assert!(
        !wall_blocks([2.0, 1.0, 1.0], [2.0, 3.0, 1.0], &wall),
        "parallel (den == 0)"
    );

    // A door's opening: the wall is drawn as two pieces with a 1 m gap,
    // and the door object in the gap never blocks.
    let walls = vec![
        Wall {
            a: (2.0, 0.0),
            b: (2.0, 1.5),
            h: 2.7,
        },
        Wall {
            a: (2.0, 2.5),
            b: (2.0, 4.0),
            h: 2.7,
        },
    ];
    let door = BoxObj::from_yaw_deg(2.0, 2.0, 1.05, 1.0, 0.2, 2.1, 90.0, true);
    let with_door = Blockers {
        walls: walls.clone(),
        boxes: vec![door],
    };
    assert!(
        !blocked(p, n, &with_door, &[]),
        "through a door's opening (no wall)"
    );
    assert!(
        blocked([1.0, 0.5, 1.0], [3.0, 0.7, 1.2], &with_door, &[]),
        "through the wall beside it"
    );
    // The same box as furniture blocks.
    let table = BoxObj {
        opening: false,
        ..door
    };
    assert!(box_blocks(p, n, &table));
    assert!(!box_blocks(p, n, &door));
    assert!(
        !box_blocks([1.0, 2.0, 2.2], [3.0, 2.2, 2.3], &table),
        "over it"
    );
    assert!(blocked(
        p,
        n,
        &Blockers {
            walls,
            boxes: vec![table]
        },
        &[]
    ));

    // A person's cylinder at (2, 2), r 0.18, top 1.7.
    let c = Cylinder {
        cx: 2.0,
        cy: 2.1,
        radius: 0.18,
        top: 1.7,
    };
    assert!(cylinder_blocks(p, n, &c), "below its top");
    assert!(
        !cylinder_blocks([1.0, 2.0, 1.8], [3.0, 2.2, 1.9], &c),
        "above its top"
    );
    assert!(
        !cylinder_blocks([1.0, 2.4, 1.0], [3.0, 2.6, 1.2], &c),
        "beside it"
    );
    // An end inside the cylinder (u* <= 0 or >= 1): blocked iff that end
    // is below the top.
    assert!(cylinder_blocks([2.05, 2.1, 1.0], [3.0, 2.1, 1.2], &c));
    assert!(cylinder_blocks([3.0, 2.1, 1.2], [2.05, 2.1, 1.0], &c));
    assert!(!cylinder_blocks([2.05, 2.1, 1.8], [3.0, 2.1, 1.2], &c));
    // L2 == 0: not blocked, even through the cylinder.
    assert!(!cylinder_blocks([2.0, 2.1, 0.5], [2.0, 2.1, 1.5], &c));
    assert!(blocked(p, n, &Blockers::default(), &[c]));
    assert!(!blocked(p, n, &Blockers::default(), &[]));
}

// =====================================================================
// §6.1 epochs, §6.3 draw order, §6.4-§6.5 reports, shadowing
// =====================================================================

fn one_node() -> Vec<PhoneNode> {
    vec![PhoneNode {
        id: 7,
        x: 4.0,
        y: 3.0,
        z: 1.2,
    }]
}

fn sample(t: f64, p: [f64; 3]) -> TruthSample<'static> {
    TruthSample {
        t,
        phone: p,
        cylinders: &[],
    }
}

#[test]
fn epoch_rule_fires_at_samples_and_drops_passed_epochs() {
    let cfg = base(); // 5 Hz
    let nodes = one_node();
    let b = Blockers::default();
    let mut radio = Radio::new(&cfg, &nodes, &b, "SIM").unwrap();
    let ts = [0.0, 0.05, 0.2, 0.31, 0.32, 1.0, 1.05, 1.2];
    let mut fired = Vec::new();
    for &t in &ts {
        if let Some(ep) = radio.step(&sample(t, [1.0, 1.0, 1.0])).unwrap() {
            assert_eq!(ep.t, t, "its time is t_j, never e / rate");
            fired.push(t);
        }
    }
    // e_next: 0 -> fires at 0.0 (-> 1); 0.2 >= 1/5 fires (-> 2); 0.31 < 0.4;
    // 1.0 fires and passes 2, 3, 4, 5 (-> 6); 1.2 >= 6/5 fires (-> 7).
    assert_eq!(fired, vec![0.0, 0.2, 1.0, 1.2]);
    assert_eq!(radio.next_epoch(), 7);

    // Dropped epochs make no draws: the same fired samples alone give the
    // same reports.
    let run = |ts: &[f64]| -> Vec<EpochRecord> {
        let samples: Vec<TruthSample> = ts.iter().map(|&t| sample(t, [1.0, 1.0, 1.0])).collect();
        simulate(&cfg, &nodes, &b, "SIM", &samples).unwrap()
    };
    let a = run(&ts);
    let bb = run(&[0.0, 0.2, 1.0, 1.2]);
    assert_eq!(a.len(), 4);
    for (x, y) in a.iter().zip(&bb) {
        assert_eq!(x.reports, y.reports);
        assert_eq!(x.t, y.t);
    }
}

#[test]
fn epoch_rule_long_gaps_match_the_plain_loop() {
    let cfg = Config {
        phone_rate_hz: 7.3,
        ..base()
    };
    let nodes = one_node();
    let mut radio = Radio::new(&cfg, &nodes, &Blockers::default(), "SIM").unwrap();
    let mut e_next: u64 = 0;
    for &t in &[0.0, 3.1, 3.2, 1234.5678, 1234.6, 99999.9, 100000.0, 1.0e7] {
        let fires = t >= (e_next as f64) / cfg.phone_rate_hz;
        if fires {
            while t >= (e_next as f64) / cfg.phone_rate_hz {
                e_next += 1;
            }
        }
        let got = radio.step(&sample(t, [1.0, 1.0, 1.0])).unwrap();
        assert_eq!(got.is_some(), fires, "t = {t}");
        assert_eq!(radio.next_epoch(), e_next, "t = {t}");
    }
}

#[test]
fn draws_follow_the_spec_order_bit_for_bit() {
    let cfg = Config {
        phone_seed: Some(12345),
        rtt_fail_p: 0.0,
        rssi_fail_p: 0.0,
        rtt_outlier_p: 0.5, // exercise the outlier branch on some epochs
        ..base()
    };
    let nodes = one_node();
    let n = nodes[0];
    // Blocked by a wall so the NLoS terms are in play.
    let walls = Blockers {
        walls: vec![Wall {
            a: (2.5, 0.0),
            b: (2.5, 5.0),
            h: 3.0,
        }],
        boxes: vec![],
    };
    let mut radio = Radio::new(&cfg, &nodes, &walls, "SIM").unwrap();
    let mut rtt = Rng::stream(12345, RTT_STREAM_BASE + 7);
    let mut rssi = Rng::stream(12345, RSSI_STREAM_BASE + 7);
    let off = cfg.rtt_offset_sigma_m * rtt.gauss();
    let dev = cfg.rssi_device_sigma_db * rssi.gauss();
    assert_eq!(radio.offsets(), vec![(off, dev)]);

    let path = [
        [1.0, 1.0, 1.0],
        [1.3, 1.1, 1.0],
        [1.3, 1.1, 1.0],
        [2.0, 0.5, 1.0],
    ];
    let mut s_prev: Option<f64> = None;
    let mut shadows = Vec::new();
    let mut xy_prev = (0.0, 0.0);
    let mut outliers = 0;
    for (k, p) in path.iter().enumerate() {
        let ep = radio
            .step(&sample(0.2 * k as f64, *p))
            .unwrap()
            .expect("fires");
        let rep = ep.reports[0];
        assert!(rep.blocked);
        // RTT stream: u_fail, g, u_out, u_mag, u_b.
        let _u_fail = rtt.uniform();
        let g = rtt.gauss();
        let u_out = rtt.uniform();
        let u_mag = rtt.uniform();
        let u_b = rtt.uniform();
        let d = (((p[0] - n.x) * (p[0] - n.x) + (p[1] - n.y) * (p[1] - n.y))
            + (p[2] - n.z) * (p[2] - n.z))
            .sqrt();
        let sig = cfg.rtt_sigma_m * cfg.rtt_nlos_sigma_scale;
        let mut r = (d + off) + sig * g;
        r += -(cfg.rtt_nlos_bias_m * (1.0 - u_b).ln());
        if u_out < cfg.rtt_outlier_p {
            r += cfg.rtt_outlier_lo_m + (cfg.rtt_outlier_hi_m - cfg.rtt_outlier_lo_m) * u_mag;
            outliers += 1;
        }
        let r = if r > 0.0 { r } else { 0.0 };
        assert_eq!(rep.rtt_m, Some(r), "epoch {k}");
        // RSSI stream: u_fail, g_s, u_j.
        let _u_fail_s = rssi.uniform();
        let g_s = rssi.gauss();
        let u_j = rssi.uniform();
        let s = match s_prev {
            None => cfg.rssi_shadow_sigma_db * g_s,
            Some(sp) => {
                let disp = ((p[0] - xy_prev.0) * (p[0] - xy_prev.0)
                    + (p[1] - xy_prev.1) * (p[1] - xy_prev.1))
                    .sqrt();
                let rho = (-(disp / cfg.rssi_shadow_corr_m)).exp();
                rho * sp + (1.0 - rho * rho).sqrt() * (cfg.rssi_shadow_sigma_db * g_s)
            }
        };
        assert_eq!(rep.shadow_db, s, "epoch {k}");
        let dd = if d > 0.1 { d } else { 0.1 };
        let lambda = 299792458.0 / cfg.carrier_hz.unwrap();
        let pl0 = 20.0 * (12.566370614359172 / lambda).log10();
        assert!(dd <= cfg.rssi_breakpoint_m);
        let pl = pl0 + 20.0 * dd.log10();
        let v = ((((cfg.tx_power_dbm.unwrap() - pl) - cfg.rssi_nlos_loss_db) + s) + dev)
            + (-cfg.rssi_jitter_db + (2.0 * cfg.rssi_jitter_db) * u_j);
        assert_eq!(rep.rssi_v, v, "epoch {k}");
        let q = v.round_ties_even().clamp(-100.0, 0.0);
        assert_eq!(rep.rssi_dbm, Some(q as i32));
        s_prev = Some(s);
        shadows.push(s);
        xy_prev = (p[0], p[1]);
    }
    assert!(outliers > 0 && outliers < path.len(), "{outliers}");
    // A still phone keeps rho = 1: S repeats exactly (epoch 1 -> 2).
    assert_eq!(shadows[2], shadows[1]);
    assert_ne!(shadows[3], shadows[2]);
}

#[test]
fn rssi_breakpoint_and_clamp() {
    // A node 12 m away is past the 5 m breakpoint: the far slope applies.
    let cfg = Config {
        rssi_fail_p: 0.0,
        rssi_shadow_sigma_db: 1e-9,
        rssi_device_sigma_db: 0.0,
        rssi_jitter_db: 0.0,
        ..base()
    };
    let nodes = vec![PhoneNode {
        id: 0,
        x: 13.0,
        y: 1.0,
        z: 1.0,
    }];
    let mut radio = Radio::new(&cfg, &nodes, &Blockers::default(), "SIM").unwrap();
    let ep = radio.step(&sample(0.0, [1.0, 1.0, 1.0])).unwrap().unwrap();
    let pl0 = cfg.pl0_db().unwrap();
    let pl = (pl0 + 20.0 * 5.0f64.log10()) + (10.0 * 3.5) * (12.0f64 / 5.0).log10();
    let v = ep.reports[0].rssi_v;
    assert!((v - (10.0 - pl)).abs() < 1e-6, "{v}");
    assert_eq!(ep.reports[0].rssi_dbm, Some((10.0 - pl).round() as i32));
    // Very strong signals clamp at 0 dBm, very weak ones at -100 dBm
    // (clamp(round(v), -100, 0), via tx_power_dbm).
    for (tx, want) in [(80.0, 0), (-80.0, -100)] {
        let c = Config {
            tx_power_dbm: Some(tx),
            ..cfg.clone()
        };
        let mut radio = Radio::new(&c, &nodes, &Blockers::default(), "SIM").unwrap();
        let ep = radio.step(&sample(0.0, [1.0, 1.0, 1.0])).unwrap().unwrap();
        let v = ep.reports[0].rssi_v;
        assert!((v - (tx - pl)).abs() < 1e-6, "{v}");
        if want == 0 {
            assert!(v > 0.5, "{v}");
        } else {
            assert!(v < -100.5, "{v}");
        }
        assert_eq!(ep.reports[0].rssi_dbm, Some(want), "tx {tx}");
    }
}

#[test]
fn non_finite_rssi_gives_no_report_and_keeps_the_draws() {
    // §6.5: with rssi_jitter_db = 1e308, (2.0*jitter)*u_j overflows, so v
    // is never finite: no RSSI report, while the draws, S and the RTT
    // reports are exactly those of the same run with a finite jitter.
    let nodes = scene_nodes("bedroom.json");
    let b = Blockers::default();
    let ok = Config {
        rssi_fail_p: 0.0,
        ..base()
    };
    let wild = Config {
        rssi_jitter_db: 1e308,
        ..ok.clone()
    };
    wild.validate().unwrap();
    let mut a = Radio::new(&ok, &nodes, &b, "SIM").unwrap();
    let mut w = Radio::new(&wild, &nodes, &b, "SIM").unwrap();
    for (k, p) in walk(40, 0.2).iter().enumerate() {
        let s = sample(0.2 * k as f64, *p);
        let ea = a.step(&s).unwrap().unwrap();
        let ew = w.step(&s).unwrap().unwrap();
        for (ra, rw) in ea.reports.iter().zip(&ew.reports) {
            assert!(!rw.rssi_v.is_finite());
            assert_eq!(rw.rssi_dbm, None);
            assert!(ra.rssi_dbm.is_some());
            assert_eq!(rw.shadow_db, ra.shadow_db);
            assert_eq!(rw.rtt_m, ra.rtt_m);
            assert_eq!(rw.blocked, ra.blocked);
        }
    }
    // Through the whole phone: RSSI abstains no_reports, RTT is unchanged.
    let samples: Vec<TruthSample> = walk(40, 0.2)
        .iter()
        .enumerate()
        .map(|(k, p)| sample(0.2 * k as f64, *p))
        .collect();
    let ra = simulate(&ok, &nodes, &b, "SIM", &samples).unwrap();
    let rw = simulate(&wild, &nodes, &b, "SIM", &samples).unwrap();
    for (x, y) in ra.iter().zip(&rw) {
        assert_eq!(y.rssi, Outcome::Abstain(Reason::NoReports));
        assert_eq!(x.rtt, y.rtt);
    }
}

fn walk(n: usize, dt: f64) -> Vec<[f64; 3]> {
    (0..n)
        .map(|k| {
            let t = k as f64 * dt;
            let a = 0.4 * t;
            [2.1 + 1.2 * a.cos(), 1.8 + 1.0 * a.sin(), 1.0]
        })
        .collect()
}

#[test]
fn shadowing_and_draws_advance_on_failed_reports() {
    let nodes = scene_nodes("bedroom.json");
    let b = Blockers::default();
    let path = walk(400, 0.0125);
    let samples: Vec<TruthSample> = path
        .iter()
        .enumerate()
        .map(|(k, p)| sample(k as f64 * 0.0125, *p))
        .collect();
    let no_fail = Config {
        phone_seed: Some(99),
        rtt_fail_p: 0.0,
        rssi_fail_p: 0.0,
        ..base()
    };
    let all_fail = Config {
        rtt_fail_p: 1.0,
        rssi_fail_p: 1.0,
        ..no_fail.clone()
    };
    let some_fail = Config {
        rtt_fail_p: 0.5,
        rssi_fail_p: 0.5,
        ..no_fail.clone()
    };
    let a = run_with_radio(&no_fail, &nodes, &b, &samples);
    let f = run_with_radio(&all_fail, &nodes, &b, &samples);
    let h = run_with_radio(&some_fail, &nodes, &b, &samples);
    assert_eq!(a.len(), f.len());
    let mut failed = 0;
    for (((ea, ra_ep), (ef, rf_ep)), (eh, rh_ep)) in a.iter().zip(&f).zip(&h) {
        // The record's reports are the radio's, without the diagnostics.
        for (rec, ep) in [(ea, ra_ep), (ef, rf_ep), (eh, rh_ep)] {
            let want: Vec<ReportRecord> = ep.reports.iter().map(ReportRecord::from).collect();
            assert_eq!(rec.reports, want);
        }
        for ((ra, rf), rh) in ra_ep.reports.iter().zip(&rf_ep.reports).zip(&rh_ep.reports) {
            assert_eq!(rf.rtt_m, None);
            assert_eq!(rf.rssi_dbm, None);
            // S and v are computed whatever u_fail decides.
            assert_eq!(rf.shadow_db, ra.shadow_db);
            assert_eq!(rf.rssi_v, ra.rssi_v);
            assert_eq!(rh.shadow_db, ra.shadow_db);
            // Where a report survives, it is the one the no-fail run made.
            if let Some(r) = rh.rtt_m {
                assert_eq!(Some(r), ra.rtt_m);
            } else {
                failed += 1;
            }
            if let Some(r) = rh.rssi_dbm {
                assert_eq!(Some(r), ra.rssi_dbm);
            }
        }
        assert_eq!(ef.rtt, Outcome::Abstain(Reason::NoReports));
        assert_eq!(ef.rssi, Outcome::Abstain(Reason::NoReports));
    }
    assert!(failed > 0);
}

/// Each fired epoch's record with the radio's own epoch beside it (its
/// per-node diagnostics, which the record leaves out).
fn run_with_radio(
    cfg: &Config,
    nodes: &[PhoneNode],
    b: &Blockers,
    samples: &[TruthSample<'_>],
) -> Vec<(EpochRecord, RadioEpoch)> {
    let mut run = PhoneRun::new(cfg, nodes, b, "SIM").unwrap();
    let mut out = Vec::new();
    for s in samples {
        if let Some(rec) = run.step(s).unwrap() {
            out.push((rec, run.last_epoch().unwrap().clone()));
        }
    }
    out
}

// =====================================================================
// §7.1 / §7.2 fixes (§12a)
// =====================================================================

#[test]
fn rtt_noiseless_bedroom_recovers_the_truth() {
    let nodes = scene_nodes("bedroom.json");
    assert_eq!(nodes.len(), 4);
    let cfg = Config {
        phone_assumed_z_m: 1.0,
        ..Config::default()
    };
    for truth in [
        [1.1, 2.6, 1.0],
        [2.1, 1.8, 1.0],
        [0.6, 0.4, 1.0],
        [3.9, 3.3, 1.0],
        [4.05, 0.15, 1.0], // under a node: rho = |dz| > 0
    ] {
        let reps = exact_ranges(&nodes, truth);
        let fx = ok_fix(rtt_fix(&cfg, &reps)).unwrap();
        let err = dist2((fx.x, fx.y), (truth[0], truth[1]));
        assert!(err < 1e-6, "{truth:?}: {err}");
        assert!(fx.chi2.unwrap() < 1e-10);
        assert_eq!(fx.used, 4);
        // R = sigma^2 * inv(N), symmetric, positive.
        assert_eq!(fx.r[0][1], fx.r[1][0]);
        assert!(fx.r[0][0] > 0.0 && fx.r[1][1] > 0.0);
        println!(
            "SIM RTT noiseless {truth:?}: error {err:.2e} m in {} iterations",
            fx.iterations
        );
    }
    // The phone exactly at a node's xy and height: that report is skipped
    // (rho < 1e-9) and the other three still fix it.
    let at_node = [nodes[0].x, nodes[0].y, nodes[0].z];
    let cfg_n = Config {
        phone_assumed_z_m: nodes[0].z,
        ..Config::default()
    };
    let fx = ok_fix(rtt_fix(&cfg_n, &exact_ranges(&nodes, at_node))).unwrap();
    assert!(dist2((fx.x, fx.y), (at_node[0], at_node[1])) < 1e-6);
    assert_eq!(fx.used, 3);
}

/// The office's three collinear nodes `ids` (y = 0.2 or y = 4.8).
fn office_line(ids: [u8; 3]) -> Vec<PhoneNode> {
    let line: Vec<PhoneNode> = scene_nodes("office.json")
        .into_iter()
        .filter(|n| ids.contains(&n.id))
        .collect();
    assert_eq!(line.len(), 3);
    assert!(line.iter().all(|n| n.y == line[0].y));
    line
}

#[test]
fn rtt_start_is_exact_when_noiseless_and_refuses_collinear_anchors() {
    let nodes = scene_nodes("bedroom.json");
    let cfg = Config::default();
    let truth = [1.1, 2.6, 1.0];
    let (x0, y0) = rtt_start(&cfg, &exact_ranges(&nodes, truth)).unwrap();
    assert!(dist2((x0, y0), (1.1, 2.6)) < 1e-9);
    assert_eq!(rtt_start(&cfg, &[]), Err(Reason::NoReports));
    // All anchors on one line: the linearised normal matrix fails
    // det > 1e-9 * ((n00 + n11)*(n00 + n11)), so there is no start (§7's
    // collinear rule; the centroid fallback is gone).
    let line = office_line([0, 4, 1]);
    let reps = exact_ranges(&line, [3.0, 2.5, 1.0]);
    assert!(anchors_collinear(&reps));
    assert_eq!(rtt_start(&cfg, &reps), Err(Reason::Degenerate));
    assert!(!anchors_collinear(&exact_ranges(&nodes, truth)));
}

#[test]
fn collinear_office_nodes_are_degenerate_for_every_phone_position() {
    // §12a: the office's collinear nodes 0, 4, 1 (and 3, 5, 2) alone ->
    // `degenerate` for every phone position, both fixes, on a 0.1 m grid
    // over the room (6 m x 5 m). A range or level set from anchors on one
    // line cannot tell the two sides of the line apart.
    let cfg = base();
    let p0 = cfg.rssi_p0().unwrap();
    let mut checked = 0;
    for ids in [[0, 4, 1], [3, 5, 2]] {
        let line = office_line(ids);
        for i in 0..=60 {
            for j in 0..=50 {
                let truth = [f64::from(i) / 10.0, f64::from(j) / 10.0, 1.0];
                let rtt = exact_ranges(&line, truth);
                assert_eq!(
                    ok_fix(rtt_fix(&cfg, &rtt)),
                    Err(Reason::Degenerate),
                    "RTT {ids:?} {truth:?}"
                );
                // RSSI: the estimator's own model, rounded as the radio rounds.
                let rssi: Vec<Ranging> = line
                    .iter()
                    .map(|n| {
                        let rho = len3(truth[0] - n.x, truth[1] - n.y, cfg.phone_assumed_z_m - n.z);
                        let v = p0 - (10.0 * cfg.rssi_est_exponent) * rho.log10();
                        Ranging {
                            node: *n,
                            value: v.round_ties_even().clamp(-100.0, 0.0),
                        }
                    })
                    .collect();
                assert_eq!(
                    ok_fix(rssi_fix(&cfg, &rssi)),
                    Err(Reason::Degenerate),
                    "RSSI {ids:?} {truth:?}"
                );
                checked += 2;
            }
        }
    }
    println!("SIM collinear office nodes 0,4,1 and 3,5,2: {checked} fixes on a 0.1 m grid, all degenerate");
    // Collinear anchors with any reports: degenerate before the start
    // (also for values the start could not use).
    let line = office_line([0, 4, 1]);
    let flat: Vec<Ranging> = line
        .iter()
        .map(|n| Ranging {
            node: *n,
            value: -50.0,
        })
        .collect();
    assert_eq!(ok_fix(rssi_fix(&cfg, &flat)), Err(Reason::Degenerate));
    // A fourth node off the line lifts it.
    let office = scene_nodes("office.json");
    let four: Vec<PhoneNode> = office
        .iter()
        .copied()
        .filter(|n| [0, 1, 4, 5].contains(&n.id))
        .collect();
    let fx = ok_fix(rtt_fix(&cfg, &exact_ranges(&four, [2.9, 4.5, 1.0]))).unwrap();
    assert!(dist2((fx.x, fx.y), (2.9, 4.5)) < 1e-6);
}

/// Noiseless RSSI reports from the estimator's own model (unrounded).
fn model_rssi(cfg: &Config, nodes: &[PhoneNode], truth: [f64; 2]) -> Vec<Ranging> {
    let p0 = cfg.rssi_p0().unwrap();
    nodes
        .iter()
        .map(|n| {
            let rho = len3(truth[0] - n.x, truth[1] - n.y, cfg.phone_assumed_z_m - n.z);
            Ranging {
                node: *n,
                value: p0 - (10.0 * cfg.rssi_est_exponent) * rho.log10(),
            }
        })
        .collect()
}

#[test]
fn collinear_rule_does_not_depend_on_the_line_direction() {
    // §7: `not (det > 1e-9 * ((n00 + n11)*(n00 + n11)))`. Office-like
    // nodes on y = 0.3 but with two of them at 0.1 + 0.2 (one ulp above):
    // collinear to rounding on an axis-aligned line, where n11 is itself
    // rounding-sized. The old `1e-9 * (n00*n11)` test passed them (ratio
    // about 0.2) and published a mirror fix 8.4 m off.
    let y = 0.1 + 0.2;
    assert_ne!(y, 0.3);
    let nodes = [(0u8, 0.2, y), (4, 3.0, y), (1, 5.8, 0.3)].map(|(id, x, y)| PhoneNode {
        id,
        x,
        y,
        z: 1.5,
    });
    let cfg = base();
    assert_eq!(cfg.phone_assumed_z_m, 1.0);
    let rtt = exact_ranges(&nodes, [2.9, 4.5, 1.0]);
    let (n00, n01, n11) = anchor_normal(&rtt);
    let det = n00 * n11 - n01 * n01;
    assert!(det > 1e-9 * (n00 * n11), "the old rule would accept it");
    assert!(det <= 1e-9 * ((n00 + n11) * (n00 + n11)));
    assert!(anchors_collinear(&rtt));
    assert_eq!(rtt_start(&cfg, &rtt), Err(Reason::Degenerate));
    assert_eq!(ok_fix(rtt_fix(&cfg, &rtt)), Err(Reason::Degenerate));
    let rssi = model_rssi(&cfg, &nodes, [2.9, 4.5]);
    assert_eq!(ok_fix(rssi_fix(&cfg, &rssi)), Err(Reason::Degenerate));
    // The same line at any angle: rotate the triple (and the truth) about
    // the origin; still collinear to rounding, still degenerate.
    for k in 0..12 {
        let a = f64::from(k) * std::f64::consts::PI / 12.0;
        let (c, s) = (a.cos(), a.sin());
        let rot = |x: f64, y: f64| (c * x - s * y, s * x + c * y);
        let turned = nodes.map(|n| {
            let (x, y) = rot(n.x, n.y);
            PhoneNode { x, y, ..n }
        });
        let (tx, ty) = rot(2.9, 4.5);
        let rtt = exact_ranges(&turned, [tx, ty, 1.0]);
        assert_eq!(
            ok_fix(rtt_fix(&cfg, &rtt)),
            Err(Reason::Degenerate),
            "angle {k}/12 pi"
        );
        let rssi = model_rssi(&cfg, &turned, [tx, ty]);
        assert_eq!(
            ok_fix(rssi_fix(&cfg, &rssi)),
            Err(Reason::Degenerate),
            "angle {k}/12 pi"
        );
    }
    // The hall's two collinear triples (0, 5, 6 on x = 0.3; 3, 4, 6 on
    // x + y = 7) are still caught.
    let hall = scene_nodes("hall-l-shaped.json");
    let pick = |ids: [u8; 3]| -> Vec<PhoneNode> {
        hall.iter()
            .copied()
            .filter(|n| ids.contains(&n.id))
            .collect()
    };
    for ids in [[0, 5, 6], [3, 4, 6]] {
        let reps = exact_ranges(&pick(ids), [2.0, 2.0, 1.0]);
        assert!(anchors_collinear(&reps), "hall {ids:?}");
        assert_eq!(ok_fix(rtt_fix(&cfg, &reps)), Err(Reason::Degenerate));
    }
    // A thin but valid triangle (hall nodes 1, 3, 4) still gives fixes.
    let thin = pick([1, 3, 4]);
    let (n00, n01, n11) = anchor_normal(&exact_ranges(&thin, [2.0, 2.0, 1.0]));
    let ratio = (n00 * n11 - n01 * n01) / ((n00 + n11) * (n00 + n11));
    assert!(ratio > 1e-4 && ratio < 1e-3, "{ratio}");
    for truth in [[6.0, 1.5], [2.0, 5.0], [4.0, 4.0], [7.5, 2.0], [1.0, 1.0]] {
        let rtt = exact_ranges(&thin, [truth[0], truth[1], 1.0]);
        assert!(!anchors_collinear(&rtt));
        let fx = ok_fix(rtt_fix(&cfg, &rtt)).unwrap();
        let err = dist2((fx.x, fx.y), (truth[0], truth[1]));
        assert!(err < 1e-6, "RTT thin {truth:?}: {err}");
        // RSSI gives a fix too (not degenerate); with three anchors its
        // Gauss-Newton may settle on the far solution, so only that.
        let fx = ok_fix(rssi_fix(&cfg, &model_rssi(&cfg, &thin, truth))).unwrap();
        let err_rssi = dist2((fx.x, fx.y), (truth[0], truth[1]));
        println!("SIM thin hall triangle 1,3,4 at {truth:?}: RTT error {err:.1e} m, RSSI {err_rssi:.1e} m");
    }
}

#[test]
fn one_sided_huber_moves_less_than_least_squares_on_a_long_range() {
    let nodes = scene_nodes("bedroom.json");
    let truth = [1.1, 2.6, 1.0];
    let mut reps = exact_ranges(&nodes, truth);
    reps[2].value += 5.0;
    let huber = Config::default();
    let ls = Config {
        huber_k: 1e300,
        ..Config::default()
    };
    let fh = ok_fix(rtt_fix(&huber, &reps)).unwrap();
    let fl = ok_fix(rtt_fix(&ls, &reps)).unwrap();
    let eh = dist2((fh.x, fh.y), (truth[0], truth[1]));
    let el = dist2((fl.x, fl.y), (truth[0], truth[1]));
    println!(
        "SIM RTT one range +5 m: one-sided Huber error {eh:.3} m, plain least squares {el:.3} m"
    );
    assert!(eh < el, "{eh} vs {el}");
    // A short range is not down-weighted (one-sided): Huber equals LS.
    let mut short = exact_ranges(&nodes, truth);
    short[2].value -= 1.0;
    let a = ok_fix(rtt_fix(&huber, &short)).unwrap();
    let b = ok_fix(rtt_fix(&ls, &short)).unwrap();
    assert_eq!((a.x, a.y), (b.x, b.y));
}

#[test]
fn rssi_noiseless_from_its_own_model_recovers_the_truth() {
    let nodes = scene_nodes("bedroom.json");
    let cfg = base();
    let p0 = cfg.rssi_p0().unwrap();
    for truth in [[1.1, 2.6], [2.1, 1.8], [3.5, 0.7], [0.4, 3.0]] {
        let reps: Vec<Ranging> = nodes
            .iter()
            .map(|n| {
                let rho = len3(truth[0] - n.x, truth[1] - n.y, cfg.phone_assumed_z_m - n.z);
                Ranging {
                    node: *n,
                    value: p0 - (10.0 * cfg.rssi_est_exponent) * rho.log10(),
                }
            })
            .collect();
        let fx = ok_fix(rssi_fix(&cfg, &reps)).unwrap();
        let err = dist2((fx.x, fx.y), (truth[0], truth[1]));
        assert!(err < 1e-6, "{truth:?}: {err}");
        assert_eq!(fx.chi2, None);
        let (sx, sy) = rssi_start(&reps).unwrap();
        assert!(sx.is_finite() && sy.is_finite());
        assert_eq!(fx.start, (sx, sy));
        // R = shadow_sigma^2 * inv(N).
        assert_eq!(fx.r[0][0], 16.0 * fx.inv_n[0][0]);
        println!(
            "SIM RSSI noiseless {truth:?}: error {err:.2e} m in {} iterations",
            fx.iterations
        );
    }
}

#[test]
fn fix_reasons_refusals_and_guards() {
    let nodes = scene_nodes("bedroom.json");
    let cfg = base();
    let reps = exact_ranges(&nodes, [1.1, 2.6, 1.0]);
    for f in [rtt_fix, rssi_fix] {
        assert_eq!(ok_fix(f(&cfg, &[])), Err(Reason::NoReports));
        assert_eq!(ok_fix(f(&cfg, &reps[..1])), Err(Reason::TooFewRanges));
        assert_eq!(ok_fix(f(&cfg, &reps[..2])), Err(Reason::TooFewRanges));
    }
    assert_eq!(
        ok_fix(saans_locate::phone::fix(Method::Rtt, &cfg, &reps[..2])),
        Err(Reason::TooFewRanges)
    );
    // §7: reports name distinct nodes with finite values, else the fix is
    // refused naming `reports` (before any reason, whatever m is).
    for f in [rtt_fix, rssi_fix] {
        for bad_value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            for m in [1, 2, 4] {
                let mut bad = reps[..m].to_vec();
                bad[m - 1].value = bad_value;
                assert_eq!(field_of(&f(&cfg, &bad).unwrap_err()), "reports");
            }
        }
        let mut dup = reps.clone();
        dup[3].node = dup[0].node;
        assert_eq!(field_of(&f(&cfg, &dup).unwrap_err()), "reports");
        assert_eq!(
            field_of(&f(&cfg, &[reps[1], reps[1]]).unwrap_err()),
            "reports"
        );
        let mut nan_node = reps.clone();
        nan_node[2].node.x = f64::NAN;
        assert_eq!(field_of(&f(&cfg, &nan_node).unwrap_err()), "nodes");
        // An invalid config is refused naming its parameter.
        let bad_cfg = Config {
            huber_k: 0.0,
            ..cfg.clone()
        };
        assert_eq!(field_of(&f(&bad_cfg, &reps).unwrap_err()), "huber_k");
    }
    // Reports in another order are taken nodes ascending.
    let mut rev = reps.clone();
    rev.reverse();
    assert_eq!(ok_fix(rtt_fix(&cfg, &rev)), ok_fix(rtt_fix(&cfg, &reps)));
    // RSSI needs rssi_p0: set, or derived from Room Sim's values.
    let levels: Vec<Ranging> = reps
        .iter()
        .zip([-40.0, -50.0, -55.0, -45.0])
        .map(|(r, v)| Ranging { value: v, ..*r })
        .collect();
    assert_eq!(
        field_of(&rssi_fix(&Config::default(), &levels).unwrap_err()),
        "rssi_p0_dbm"
    );
    let with_p0 = Config {
        rssi_p0_dbm: Some(cfg.rssi_p0().unwrap()),
        ..Config::default()
    };
    assert_eq!(
        ok_fix(rssi_fix(&with_p0, &levels)),
        ok_fix(rssi_fix(&cfg, &levels))
    );
    // §7.2's start guard: not (Σ w_n > 0) or a non-finite Σ w_n ->
    // degenerate (10^(-4000/10) is 0; 10^(4000/10) is inf).
    for v in [-4000.0, 4000.0] {
        let lv: Vec<Ranging> = reps.iter().map(|r| Ranging { value: v, ..*r }).collect();
        assert_eq!(rssi_start(&lv), Err(Reason::Degenerate), "{v}");
        assert_eq!(ok_fix(rssi_fix(&cfg, &lv)), Err(Reason::Degenerate), "{v}");
    }
}

#[test]
fn not_converged_and_far_fixes() {
    // Inputs found by a seeded search; Gauss-Newton keeps taking steps of
    // at least 1e-4 m for all 200 iterations.
    assert_eq!(GN_MAX_ITERATIONS, 200);
    let nodes = scene_nodes("bedroom.json");
    let cfg = base();
    let with = |vals: [f64; 4]| -> Vec<Ranging> {
        nodes
            .iter()
            .zip(vals)
            .map(|(n, value)| Ranging { node: *n, value })
            .collect()
    };
    assert_eq!(
        ok_fix(rtt_fix(&cfg, &with(RTT_NOT_CONVERGED))),
        Err(Reason::NotConverged)
    );
    assert_eq!(
        ok_fix(rssi_fix(&cfg, &with(RSSI_NOT_CONVERGED))),
        Err(Reason::NotConverged)
    );
    // §7.1's slowly converging case (review): 50 iterations were not
    // enough, 200 are.
    let fx = ok_fix(rtt_fix(&cfg, &with([0.0, 2.0, 1.0, 2.0]))).unwrap();
    assert!(fx.iterations > 50, "{}", fx.iterations);
    // Ranges of 1 km from three nodes put a finite fix far away.
    let reps: Vec<Ranging> = nodes
        .iter()
        .take(3)
        .map(|n| Ranging {
            node: *n,
            value: 1e6,
        })
        .collect();
    let fx = ok_fix(rtt_fix(&cfg, &reps)).unwrap();
    assert!(fx.x.is_finite() && fx.y.is_finite());
    assert!(dist2((fx.x, fx.y), (2.1, 1.8)) > 9e5);
}

// =====================================================================
// §7.3 Kalman filter and §7.4 published estimate (§12a)
// =====================================================================

fn fix_at(x: f64, y: f64, var: f64) -> Fix {
    Fix {
        x,
        y,
        r: [[var, 0.0], [0.0, var]],
        inv_n: [[var, 0.0], [0.0, var]],
        chi2: None,
        used: 4,
        iterations: 3,
        start: (x, y),
    }
}

#[test]
fn kalman_start_is_published_ungated() {
    let cfg = Config::default();
    let mut f = PhoneFilter::new();
    assert!(f.state().is_none());
    let fx = fix_at(2.0, 1.5, 0.25);
    let out = f.epoch(1.0, Ok(&fx), 0.0, &cfg);
    let st = *f.state().unwrap();
    assert_eq!(st.s, [2.0, 1.5, 0.0, 0.0]);
    assert_eq!(
        st.p,
        [
            [0.25, 0.0, 0.0, 0.0],
            [0.0, 0.25, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ]
    );
    assert_eq!((st.t_f, st.t_acc), (1.0, 1.0));
    let e = out.estimate().unwrap();
    assert_eq!((e.x, e.y), (2.0, 1.5));
    assert_eq!(e.radius_m, radius95(0.25, 0.0, 0.25));
    assert_eq!(e.radius_m, k95() * 0.5);
    assert_eq!(e.confidence, None);
}

#[test]
fn kalman_update_matches_the_textbook_filter() {
    // Start at t = 0 with R = 0.25 I, update at t = 0.2 with z = (2.1, 2.0).
    // With no cross terms the x axis is a 2-state [pos, vel] filter:
    // P' = F P Fᵀ + Q, S = P'00 + r, K = P'[:,0] / S, P+ = (I - K H) P'.
    let cfg = Config::default();
    let (q, r, dt) = (cfg.phone_accel_psd, 0.25, 0.2);
    let mut f = PhoneFilter::new();
    f.epoch(0.0, Ok(&fix_at(2.0, 2.0, r)), 0.0, &cfg);
    let out = f.epoch(dt, Ok(&fix_at(2.1, 2.0, r)), 0.0, &cfg);
    let p00 = r + dt * dt * 1.0 + q * dt * dt * dt / 3.0;
    let p02 = dt * 1.0 + q * dt * dt / 2.0;
    let p22 = 1.0 + q * dt;
    let s = p00 + r;
    let (k0, k1) = (p00 / s, p02 / s);
    let nu = 2.1 - 2.0;
    let st = *f.state().unwrap();
    let close = |a: f64, b: f64| (a - b).abs() < 1e-12;
    assert!(close(st.s[0], 2.0 + k0 * nu), "{}", st.s[0]);
    assert!(close(st.s[2], k1 * nu), "{}", st.s[2]);
    assert!(close(st.s[1], 2.0) && close(st.s[3], 0.0));
    assert!(close(st.p[0][0], (1.0 - k0) * p00));
    assert!(close(st.p[0][2], (1.0 - k0) * p02));
    assert!(close(st.p[2][2], p22 - k1 * p02));
    assert_eq!(st.p[0][2], st.p[2][0]);
    assert_eq!(st.p[0][1], 0.0);
    let e = out.estimate().unwrap();
    assert_eq!((e.x, e.y), (st.s[0], st.s[1]));
    assert_eq!(e.radius_m, radius95(st.p[0][0], st.p[0][1], st.p[1][1]));
}

/// The predicted state from `st` to `t`, written out independently of
/// the crate's matrix helpers (CV model, dt = t - t_f).
fn predicted(st: &saans_locate::phone::FilterState, t: f64, q: f64) -> ([f64; 4], f64, f64) {
    let dt = t - st.t_f;
    let s = [
        st.s[0] + dt * st.s[2],
        st.s[1] + dt * st.s[3],
        st.s[2],
        st.s[3],
    ];
    // P00' = P00 + 2 dt P02 + dt^2 P22 + q dt^3/3 (all cross terms 0 here).
    let p00 = st.p[0][0] + dt * dt * st.p[2][2] + q * dt * dt * dt / 3.0;
    let p02 = st.p[2][2] * dt + q * dt * dt / 2.0;
    (s, p00, p02)
}

#[test]
fn kalman_gates_a_fix_ten_sigma_off_and_keeps_the_predicted_state() {
    let cfg = Config::default();
    let mut f = PhoneFilter::new();
    f.epoch(0.0, Ok(&fix_at(2.0, 2.0, 0.25)), 0.5, &cfg);
    let before = *f.state().unwrap();
    let (s_pred, p00, p02) = predicted(&before, 0.2, cfg.phone_accel_psd);
    // S00 = P00(pred) + R00; 10 sigma off in x: S is diagonal here, so
    // the gate's d2 is (10 * sqrt(S00))^2 / S00 = 100 (> CHI2_2_99 = 9.21;
    // a 3.03-sigma fix would pass).
    let s00 = p00 + 0.25;
    let off = fix_at(2.0 + 10.0 * s00.sqrt(), 2.0, 0.25);
    let d2 = (off.x - s_pred[0]) * (off.x - s_pred[0]) / s00;
    assert!((d2 - 100.0).abs() < 1e-9, "{d2}");
    let out = f.epoch(0.2, Ok(&off), 0.5, &cfg);
    assert_eq!(out, Outcome::Abstain(Reason::Gated), "nothing is published");
    let st = *f.state().unwrap();
    assert_eq!(st.s, s_pred, "the predicted state is kept");
    assert!((st.p[0][0] - p00).abs() < 1e-15);
    assert!((st.p[0][2] - p02).abs() < 1e-15);
    assert_eq!(st.p[0][2], st.p[2][0], "sym");
    assert_eq!(st.t_f, 0.2, "predicted to t");
    assert_eq!(st.t_acc, 0.0, "no accepted update");
    // A fix 2 sigma off is accepted (d2 = 4 < CHI2_2_99).
    let (_, p00b, _) = predicted(&st, 0.4, cfg.phone_accel_psd);
    let near = fix_at(2.0 + 2.0 * (p00b + 0.25).sqrt(), 2.0, 0.25);
    let out = f.epoch(0.4, Ok(&near), 0.5, &cfg);
    assert!(out.estimate().is_some(), "{out:?}");
    assert_eq!(f.state().unwrap().t_acc, 0.4);
    assert!(4.0 < chi2_2_99());
}

#[test]
fn kalman_coast_limit_is_inclusive_at_exactly_three_seconds() {
    let cfg = Config::default();
    assert_eq!(cfg.phone_max_coast_s, 3.0);
    // t - t_acc = 4.0 - 1.0 = 3.0 exactly: kept, so this is an update
    // (the velocity is no longer zero after a moved fix).
    let mut f = PhoneFilter::new();
    f.epoch(1.0, Ok(&fix_at(2.0, 2.0, 0.25)), 0.0, &cfg);
    let out = f.epoch(4.0, Ok(&fix_at(2.3, 2.0, 0.25)), 0.0, &cfg);
    assert!(out.estimate().is_some());
    let st = *f.state().unwrap();
    assert_eq!((st.t_f, st.t_acc), (4.0, 4.0));
    assert!(st.s[2] != 0.0, "updated, not restarted");
    assert!(st.s[0] != 2.3);
    // Just past 3.0 s: discarded, and the fix starts a new filter.
    let mut g = PhoneFilter::new();
    g.epoch(1.0, Ok(&fix_at(2.0, 2.0, 0.25)), 0.0, &cfg);
    let t = 4.000000000000001;
    assert!(t - 1.0 > 3.0);
    let out = g.epoch(t, Ok(&fix_at(2.3, 2.0, 0.25)), 0.0, &cfg);
    assert!(out.estimate().is_some());
    let st = *g.state().unwrap();
    assert_eq!(st.s, [2.3, 2.0, 0.0, 0.0], "restarted from the fix");
    assert_eq!((st.t_f, st.t_acc), (t, t));
}

#[test]
fn kalman_discards_after_the_coast_limit_even_without_a_fix() {
    let cfg = Config::default();
    let mut f = PhoneFilter::new();
    f.epoch(0.0, Ok(&fix_at(2.0, 2.0, 0.25)), 0.0, &cfg);
    // Gated epochs do not reset t_acc ...
    for k in 1..=15 {
        let t = 0.2 * k as f64;
        let out = f.epoch(t, Ok(&fix_at(50.0, 50.0, 0.25)), 0.0, &cfg);
        assert_eq!(out, Outcome::Abstain(Reason::Gated), "t = {t}");
        assert_eq!(f.state().unwrap().t_acc, 0.0);
    }
    // ... and a no-fix epoch past the limit discards the filter and
    // abstains with the fix's reason.
    let out = f.epoch(3.2, Err(Reason::TooFewRanges), 0.0, &cfg);
    assert_eq!(out, Outcome::Abstain(Reason::TooFewRanges));
    assert!(f.state().is_none());
    // NaN time fails closed: the filter is discarded.
    let mut g = PhoneFilter::new();
    g.epoch(0.0, Ok(&fix_at(2.0, 2.0, 0.25)), 0.0, &cfg);
    g.epoch(f64::NAN, Err(Reason::NoReports), 0.0, &cfg);
    assert!(g.state().is_none());
}

#[test]
fn kalman_never_predicts_without_a_fix() {
    let cfg = Config::default();
    let mut f = PhoneFilter::new();
    f.epoch(0.0, Ok(&fix_at(2.0, 2.0, 0.25)), 0.0, &cfg);
    let before = *f.state().unwrap();
    for (t, r) in [
        (0.2, Reason::NoReports),
        (0.4, Reason::Degenerate),
        (0.6, Reason::NotConverged),
    ] {
        assert_eq!(f.epoch(t, Err(r), 0.0, &cfg), Outcome::Abstain(r));
        assert_eq!(*f.state().unwrap(), before, "nothing predicted at {t}");
    }
    // The next fix predicts over the whole gap from t_f = 0.
    let mut g = f.clone();
    let (s_pred, p00, _) = predicted(&before, 1.0, cfg.phone_accel_psd);
    g.epoch(1.0, Ok(&fix_at(50.0, 50.0, 0.25)), 0.0, &cfg); // gated
    let st = *g.state().unwrap();
    assert_eq!(st.s, s_pred);
    assert!((st.p[0][0] - p00).abs() < 1e-15);
    assert_eq!(st.t_f, 1.0);
}

#[test]
fn kalman_gate_fails_closed_on_nan_and_on_a_non_positive_innovation_covariance() {
    let cfg = Config::default();
    let mut f = PhoneFilter::new();
    f.epoch(0.0, Ok(&fix_at(2.0, 2.0, 0.25)), 0.0, &cfg);
    let nan = fix_at(f64::NAN, 2.0, 0.25);
    assert_eq!(
        f.epoch(0.2, Ok(&nan), 0.0, &cfg),
        Outcome::Abstain(Reason::Gated)
    );
    // det(S) < 0: `not (det(S) > 0)` rejects.
    let mut indefinite = fix_at(2.0, 2.0, 0.25);
    indefinite.r = [[-10.0, 0.0], [0.0, 1.0]];
    assert_eq!(
        f.epoch(0.4, Ok(&indefinite), 0.0, &cfg),
        Outcome::Abstain(Reason::Gated)
    );
    let mut nan_r = fix_at(2.0, 2.0, 0.25);
    nan_r.r[1][1] = f64::NAN;
    assert_eq!(
        f.epoch(0.6, Ok(&nan_r), 0.0, &cfg),
        Outcome::Abstain(Reason::Gated)
    );
}

#[test]
fn persistent_error_enters_the_published_covariance_only() {
    let cfg = Config::default();
    let mut a = PhoneFilter::new();
    let mut b = PhoneFilter::new();
    let fixes = [(0.0, 2.0, 2.0), (0.2, 2.1, 2.0), (0.4, 2.15, 2.05)];
    for (t, x, y) in fixes {
        let fx = Fix {
            inv_n: [[0.3, 0.05], [0.05, 0.2]],
            ..fix_at(x, y, 0.3)
        };
        let oa = a.epoch(t, Ok(&fx), 0.0, &cfg);
        let ob = b.epoch(t, Ok(&fx), 4.0, &cfg);
        // The filter (P and the gate) never sees B.
        assert_eq!(a.state(), b.state());
        let (ea, eb) = (oa.estimate().unwrap(), ob.estimate().unwrap());
        let p = a.state().unwrap().p;
        assert_eq!(ea.cov_xx, p[0][0] + 0.0 * 0.3);
        assert_eq!(eb.cov_xx, p[0][0] + 16.0 * 0.3);
        assert_eq!(eb.cov_xy, p[0][1] + 16.0 * 0.05);
        assert_eq!(eb.cov_yy, p[1][1] + 16.0 * 0.2);
        assert_eq!(eb.radius_m, radius95(eb.cov_xx, eb.cov_xy, eb.cov_yy));
        assert!(eb.radius_m > ea.radius_m);
    }
}

#[test]
fn implausible_radius_still_starts_and_updates_the_filter() {
    let cfg = Config::default();
    let mut f = PhoneFilter::new();
    let tiny = fix_at(2.0, 2.0, 1e-4);
    let out = f.epoch(0.0, Ok(&tiny), 0.0, &cfg);
    assert_eq!(out, Outcome::Abstain(Reason::ImplausibleRadius));
    assert!(f.state().is_some());
    let out = f.epoch(0.2, Ok(&fix_at(2.0, 2.0, 1e-4)), 0.0, &cfg);
    assert_eq!(out, Outcome::Abstain(Reason::ImplausibleRadius));
    assert_eq!(f.state().unwrap().t_acc, 0.2, "updated");
    // Gated comes before implausible_radius.
    let far = fix_at(9.0, 9.0, 1e-4);
    assert_eq!(
        f.epoch(0.4, Ok(&far), 0.0, &cfg),
        Outcome::Abstain(Reason::Gated)
    );
    // With a tiny radius scale nothing is plausible (a zero scale is
    // refused by §9 validation).
    let tiny_scale = Config {
        phone_radius_scale: 1e-9,
        ..Config::default()
    };
    tiny_scale.validate().unwrap();
    let mut g = PhoneFilter::new();
    assert_eq!(
        g.epoch(0.0, Ok(&fix_at(2.0, 2.0, 1.0)), 0.5, &tiny_scale),
        Outcome::Abstain(Reason::ImplausibleRadius)
    );
    // An unvalidated config with a zero scale and a zero minimum: r = 0
    // passes `r >= 0` but not `r > 0` (§8), so no bare point is published
    // (the filter still starts).
    let zero = Config {
        phone_radius_scale: 0.0,
        min_plausible_radius_m: 0.0,
        ..Config::default()
    };
    assert!(zero.validate().is_err(), "§9 refuses both");
    let mut z = PhoneFilter::new();
    assert_eq!(
        z.epoch(0.0, Ok(&fix_at(2.0, 2.0, 1.0)), 0.5, &zero),
        Outcome::Abstain(Reason::ImplausibleRadius)
    );
    assert!(z.state().is_some());
}

#[test]
fn the_phone_radius_uses_phone_radius_scale_only() {
    // Each *_radius_scale scales only its own estimator: with the three
    // scales distinct, the phone's radius is phone_radius_scale times
    // radius95 of the published covariance, and changing the other two
    // changes nothing.
    let scaled = Config {
        wlm_radius_scale: 1.5,
        rti_radius_scale: 2.0,
        phone_radius_scale: 0.75,
        ..Config::default()
    };
    let swapped = Config {
        wlm_radius_scale: 0.75,
        rti_radius_scale: 1.5,
        phone_radius_scale: 0.75,
        ..Config::default()
    };
    let other = Config {
        phone_radius_scale: 2.0,
        ..scaled.clone()
    };
    let fixes = [(0.0, 2.0, 2.0), (0.2, 2.1, 2.0), (0.4, 2.15, 2.05)];
    let (mut a, mut b, mut c) = (PhoneFilter::new(), PhoneFilter::new(), PhoneFilter::new());
    for (t, x, y) in fixes {
        let fx = fix_at(x, y, 0.3);
        let ea = *a.epoch(t, Ok(&fx), 0.5, &scaled).estimate().unwrap();
        let eb = *b.epoch(t, Ok(&fx), 0.5, &swapped).estimate().unwrap();
        let ec = *c.epoch(t, Ok(&fx), 0.5, &other).estimate().unwrap();
        assert_eq!(
            ea.radius_m,
            0.75 * radius95(ea.cov_xx, ea.cov_xy, ea.cov_yy)
        );
        assert_eq!(ea, eb, "the WLM and RTI scales do not touch the phone");
        assert_eq!(ec.radius_m, 2.0 * radius95(ec.cov_xx, ec.cov_xy, ec.cov_yy));
        assert_eq!((ec.x, ec.y, ec.cov_xx), (ea.x, ea.y, ea.cov_xx));
    }
}

// =====================================================================
// A whole simulated walk (SIM): records, labels, radii; printed numbers
// =====================================================================

#[test]
fn simulated_walk_records_are_sim_and_radii_are_plausible() {
    let nodes = scene_nodes("bedroom.json");
    let footprint = [(0.0, 0.0), (4.2, 0.0), (4.2, 3.6), (0.0, 3.6)];
    let walls: Vec<Wall> = (0..4)
        .map(|i| Wall {
            a: footprint[i],
            b: footprint[(i + 1) % 4],
            h: 2.7,
        })
        .collect();
    let blockers = Blockers {
        walls,
        boxes: vec![BoxObj::from_yaw_deg(
            3.4, 2.4, 0.35, 1.0, 2.0, 0.7, 0.0, false,
        )],
    };
    let cfg = Config {
        phone_seed: Some(2026),
        ..base()
    };
    let dt = 0.0125;
    let path = walk(4800, dt); // 60 s at 80 Hz
    let cyls: Vec<[Cylinder; 1]> = path
        .iter()
        .map(|p| {
            [Cylinder {
                cx: p[0] - 0.25,
                cy: p[1],
                radius: 0.18,
                top: 1.7,
            }]
        })
        .collect();
    let samples: Vec<TruthSample> = path
        .iter()
        .zip(&cyls)
        .enumerate()
        .map(|(k, (p, c))| TruthSample {
            t: k as f64 * dt,
            phone: *p,
            cylinders: c,
        })
        .collect();
    let recs = simulate(&cfg, &nodes, &blockers, "SIM", &samples).unwrap();
    assert_eq!(recs.len(), 300);
    let mut blocked_n = 0;
    for (name, pick) in [
        (
            "RTT",
            (|r: &EpochRecord| r.rtt) as fn(&EpochRecord) -> Outcome,
        ),
        ("RSSI", |r: &EpochRecord| r.rssi),
    ] {
        let mut errs = Vec::new();
        let mut inside = 0;
        let mut reasons = std::collections::BTreeMap::new();
        for r in &recs {
            assert_eq!(r.label, "SIM");
            assert_eq!(r.reports.len(), 4);
            match pick(r) {
                Outcome::Estimate(e) => {
                    assert!(e.radius_m.is_finite() && e.radius_m >= cfg.min_plausible_radius_m);
                    assert!(!e.radius_calibrated);
                    let err = dist2((e.x, e.y), (r.truth[0], r.truth[1]));
                    if err <= e.radius_m {
                        inside += 1;
                    }
                    errs.push(err);
                }
                Outcome::Abstain(reason) => *reasons.entry(reason.as_str()).or_insert(0) += 1,
            }
        }
        errs.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let med = errs[errs.len() / 2];
        let p90 = errs[(errs.len() * 9) / 10];
        println!(
            "SIM bedroom walk, {name}: {} of {} epochs published, median error {med:.2} m, p90 {p90:.2} m, \
             truth within radius {:.0}%, abstentions {reasons:?}",
            errs.len(),
            recs.len(),
            100.0 * inside as f64 / errs.len() as f64
        );
        assert!(errs.len() > recs.len() / 2, "{name}");
    }
    for r in &recs {
        blocked_n += r.reports.iter().filter(|x| x.blocked).count();
    }
    assert!(blocked_n > 0, "the carrier's body blocks some segments");
}
