//! The records file (spec/livedemo.md §6, §8.3): the exact round trip of a
//! real Room Sim run and of hand-built edge cases, the shared encodings'
//! key sets (§4), and every §6.2 refusal that needs no embedded scene,
//! each at its line. Everything here is SIM.
#![cfg(feature = "roomsim")]

use std::path::PathBuf;

use saans_locate::phone::{Config, EpochRecord, ReportRecord};
use saans_locate::pipeline::{resolve_config, run_roomsim, PersonTruthRecord, TickRecord};
use saans_locate::radius::{Estimate, Outcome, Reason};
use saans_locate::records::{
    epoch_truth_from_json, epoch_truth_json, link_from_json, link_json, locate_config_from_json,
    locate_config_json, outcome_from_json, outcome_json, read_records, report_from_json,
    report_json, tick_truth_from_json, tick_truth_json, write_records, Manifest, Records,
    RecordsError, MAX_BYTES,
};
use saans_locate::score::FalseAlarms;
use saans_locate::stream::LinkValue;
use saans_locate::variant::Variant;
use saans_roomsim::{Config as RoomSimConfig, Posture};
use saans_scene::{Provenance, Scene};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

// ===================================================================
// Helpers
// ===================================================================

fn write(records: &Records) -> String {
    let mut out = Vec::new();
    write_records(&mut out, records).expect("the records encode");
    String::from_utf8(out).expect("the file is UTF-8")
}

fn read(text: &str) -> Result<Records, RecordsError> {
    read_records(text.as_bytes(), MAX_BYTES)
}

/// `a` and `b` equal with every `f64` bit for bit: `Debug` prints each
/// float in its shortest round-trip form (so `-0.0` differs from `0.0`),
/// and the one value that may be NaN, a link's `y`, is compared by bits.
fn assert_exact(a: &Records, b: &Records) {
    assert_eq!(a.manifest, b.manifest);
    assert_eq!(a.ticks, b.ticks);
    assert_eq!(a.epochs, b.epochs);
    assert_eq!(format!("{:?}", a.manifest), format!("{:?}", b.manifest));
    assert_eq!(format!("{:?}", a.ticks), format!("{:?}", b.ticks));
    assert_eq!(format!("{:?}", a.epochs), format!("{:?}", b.epochs));
    assert_eq!(a.links.len(), b.links.len());
    for (k, (la, lb)) in a.links.iter().zip(&b.links).enumerate() {
        assert_eq!(la.len(), lb.len(), "tick {k}");
        for (l, (x, y)) in la.iter().zip(lb).enumerate() {
            assert_eq!(x.y.to_bits(), y.y.to_bits(), "tick {k} link {l}: y");
            assert_eq!(
                x.gate.map(f64::to_bits),
                y.gate.map(f64::to_bits),
                "tick {k} link {l}: gate"
            );
            assert_eq!(x.disturbed, y.disturbed, "tick {k} link {l}: disturbed");
        }
    }
}

fn sorted_keys(v: &Value) -> Vec<&str> {
    let mut keys: Vec<&str> = v
        .as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect();
    keys.sort_unstable();
    keys
}

fn sorted(mut keys: Vec<&'static str>) -> Vec<&'static str> {
    keys.sort_unstable();
    keys
}

/// Every string value in `v` (keys excluded: the shipped phone's id
/// "phone" is also a frame key).
fn string_values<'a>(v: &'a Value, out: &mut Vec<&'a str>) {
    match v {
        Value::String(s) => out.push(s),
        Value::Array(items) => items.iter().for_each(|x| string_values(x, out)),
        Value::Object(m) => m.values().for_each(|x| string_values(x, out)),
        _ => {}
    }
}

// ===================================================================
// A real run
// ===================================================================

fn hall() -> (Scene, String) {
    let path =
        PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes/hall-l-shaped.json");
    let bytes = std::fs::read(&path).expect("the shipped hall is readable");
    let sha: String = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let text = String::from_utf8(bytes).expect("UTF-8");
    let scene = Scene::from_json_str("hall-l-shaped.json", &text).expect("the shipped hall loads");
    (scene, sha)
}

/// Deterministic per-tick link values: some `y` NaN or infinite, some
/// `gate` None, `disturbed` WLM-like (`y > gate`).
fn synth_links(n_ticks: usize, n_links: usize) -> Vec<Vec<LinkValue>> {
    (0..n_ticks)
        .map(|k| {
            (0..n_links)
                .map(|l| {
                    let y = match (k * 7 + l * 3) % 11 {
                        0 => f64::NAN,
                        1 => f64::INFINITY,
                        _ => ((k * 31 + l * 17) % 97) as f64 * 0.013 - 0.25,
                    };
                    let gate = (l % 5 != 0).then_some(0.25 + l as f64 * 0.1 + k as f64 * 1e-3);
                    LinkValue {
                        y,
                        gate,
                        disturbed: gate.is_some_and(|g| y > g),
                    }
                })
                .collect()
        })
        .collect()
}

#[test]
fn a_real_hall_run_round_trips_exactly() {
    let (scene, sha) = hall();
    let base = RoomSimConfig {
        duration_s: 20.0,
        ..RoomSimConfig::default()
    };
    let rs = Variant::WithWalker.apply(&scene, &base);
    let run = run_roomsim(&scene, &rs, &Config::default(), "SIM").expect("a valid run");
    let published = |o: &Outcome| o.estimate().is_some();
    assert!(run.ticks.iter().any(|t| published(&t.wlm)), "WLM publishes");
    assert!(run.ticks.iter().any(|t| !published(&t.rti)), "RTI abstains");
    assert!(
        run.epochs.iter().any(|e| published(&e.rtt)),
        "RTT publishes"
    );
    assert!(
        run.epochs
            .iter()
            .any(|e| e.reports.iter().any(|r| r.rtt_m.is_none() || r.blocked)),
        "some report is missing or blocked"
    );

    let links = synth_links(run.ticks.len(), run.links.len());
    let records = Records {
        manifest: Manifest {
            scene_name: scene.name.clone(),
            scene_sha256: sha,
            provenance: scene.provenance,
            variant: Variant::WithWalker,
            seed: rs.seed,
            duration_s: rs.duration_s,
            roomsim_config: rs.clone(),
            locate_config: resolve_config(&rs, &Config::default()).expect("resolves"),
            links: run.links.clone(),
            has_phone: run.has_phone,
            empty_run: Some((
                FalseAlarms {
                    ticks: 38,
                    seconds: 19.0,
                    published: 3,
                    per_h: Some(3.0 / 19.0 * 3600.0),
                    events: 2,
                    events_per_h: Some(2.0 / 19.0 * 3600.0),
                },
                FalseAlarms {
                    ticks: 38,
                    seconds: 19.0,
                    published: 0,
                    per_h: Some(0.0),
                    events: 0,
                    events_per_h: Some(0.0),
                },
            )),
        },
        ticks: run.ticks,
        links,
        epochs: run.epochs,
    };
    assert!(records.links.iter().flatten().any(|v| v.y.is_nan()));
    assert!(records.links.iter().flatten().any(|v| v.gate.is_none()));

    let text = write(&records);
    assert_eq!(
        text.lines().count(),
        1 + records.ticks.len() + records.epochs.len()
    );
    let back = read(&text).unwrap_or_else(|e| panic!("{e}"));
    // y = +inf is written null and read back as NaN (§4); everything else
    // is exact.
    let mut expected = records.clone();
    for v in expected.links.iter_mut().flatten() {
        if !v.y.is_finite() {
            v.y = f64::NAN;
        }
    }
    assert_exact(&back, &expected);
    // Writing what was read gives the same bytes.
    assert_eq!(write(&back), text);

    // No Person.id or Phone.id is a string value anywhere in the file.
    let mut ids: Vec<&str> = scene
        .people
        .iter()
        .filter_map(|p| p.id.as_deref())
        .collect();
    ids.extend(scene.phone.as_ref().and_then(|p| p.id.as_deref()));
    assert!(ids.contains(&"walker"));
    for (n, line) in text.lines().enumerate() {
        let v: Value = serde_json::from_str(line).expect("JSON");
        let mut strings = Vec::new();
        string_values(&v, &mut strings);
        for s in strings {
            assert!(!ids.contains(&s), "line {}: an id is a string value", n + 1);
        }
    }
}

// ===================================================================
// Hand-built edge cases
// ===================================================================

fn estimate(x: f64, y: f64, radius_m: f64, confidence: Option<f64>) -> Outcome {
    Outcome::Estimate(Estimate {
        x,
        y,
        radius_m,
        cov_xx: 5e-324,
        cov_xy: -1.7976931348623157e308,
        cov_yy: 0.1 + 0.2,
        confidence,
        radius_calibrated: false,
    })
}

fn report(node: u8, rtt_m: Option<f64>, rssi_dbm: Option<i32>, blocked: bool) -> ReportRecord {
    ReportRecord {
        node,
        rtt_m,
        rssi_dbm,
        blocked,
    }
}

fn tick(t: f64, wlm: Outcome, rti: Outcome, truth: Vec<PersonTruthRecord>) -> TickRecord {
    TickRecord {
        label: "SIM",
        t,
        wlm,
        rti,
        truth,
    }
}

fn epoch(t: f64, rtt: Outcome, rssi: Outcome, reports: Vec<ReportRecord>) -> EpochRecord {
    EpochRecord {
        label: "SIM",
        t,
        rtt,
        rssi,
        reports,
        truth: [t, -0.0, 1.0],
    }
}

/// A small declared-variant run, 2.5 s, three links, laid out as
/// - line 1 manifest
/// - line 2 epoch t=0.0 (every report null)
/// - line 3 epoch t=1.0
/// - line 4 tick t=1.0 (both abstain)
/// - line 5 tick t=1.5 (both publish)
/// - line 6 epoch t=2.0
/// - line 7 tick t=2.0
/// - line 8 epoch t=2.2
fn edge_records() -> Records {
    let people = vec![
        PersonTruthRecord {
            x: 1.25,
            y: -0.0,
            walking: true,
            posture: Posture::Stand,
        },
        PersonTruthRecord {
            x: 3.0000000000000004,
            y: 2.0,
            walking: false,
            posture: Posture::Lie,
        },
        PersonTruthRecord {
            x: 0.5,
            y: 0.5,
            walking: false,
            posture: Posture::Sit,
        },
    ];
    let lv = |y: f64, gate: Option<f64>, disturbed: bool| LinkValue { y, gate, disturbed };
    Records {
        manifest: Manifest {
            scene_name: "test room".into(),
            scene_sha256: "0123456789abcdef".repeat(4),
            provenance: Provenance::HandWritten,
            variant: Variant::Declared,
            seed: u64::MAX,
            duration_s: 2.5,
            roomsim_config: RoomSimConfig {
                seed: u64::MAX,
                duration_s: 2.5,
                ..RoomSimConfig::default()
            },
            locate_config: Config::default(),
            links: vec![(1, 2), (2, 1), (1, 3)],
            has_phone: true,
            empty_run: Some((
                FalseAlarms {
                    ticks: 0,
                    seconds: 0.0,
                    published: 0,
                    per_h: None,
                    events: 0,
                    events_per_h: None,
                },
                FalseAlarms {
                    ticks: 238,
                    seconds: 119.0,
                    published: 5,
                    per_h: Some(5.0 / 119.0 * 3600.0),
                    events: 2,
                    events_per_h: Some(2.0 / 119.0 * 3600.0),
                },
            )),
        },
        ticks: vec![
            tick(
                1.0,
                Outcome::Abstain(Reason::NoDisturbance),
                Outcome::Abstain(Reason::NotSignificant),
                people.clone(),
            ),
            tick(
                1.5,
                estimate(-0.0, 0.1 + 0.2, 4.2, None),
                estimate(1e-300, 1e300, 0.3, Some(0.987_654_321)),
                people.clone(),
            ),
            tick(
                2.0,
                Outcome::Abstain(Reason::ImplausibleRadius),
                estimate(2.0, 3.0, 0.7, Some(-0.0)),
                people,
            ),
        ],
        links: vec![
            vec![
                lv(f64::NAN, None, false),
                lv(-0.0, Some(0.25), false),
                lv(0.5, Some(0.3), true),
            ],
            vec![
                lv(1e-310, Some(1e-310), false),
                lv(f64::NAN, Some(0.25), false),
                lv(2.0, None, false),
            ],
            vec![
                lv(0.1, Some(0.2), false),
                lv(0.2, Some(0.1), true),
                lv(f64::NAN, None, false),
            ],
        ],
        epochs: vec![
            epoch(
                0.0,
                Outcome::Abstain(Reason::NoReports),
                Outcome::Abstain(Reason::NoReports),
                vec![
                    report(1, None, None, false),
                    report(2, None, None, false),
                    report(3, None, None, false),
                ],
            ),
            epoch(
                1.0,
                Outcome::Abstain(Reason::TooFewRanges),
                estimate(2.5, 1.5, 6.25, None),
                vec![
                    report(1, Some(3.25), None, true),
                    report(2, None, Some(-61), false),
                    report(3, Some(-0.5), Some(-100), true),
                ],
            ),
            epoch(
                2.0,
                estimate(2.0, 1.0, 1.5, None),
                Outcome::Abstain(Reason::Gated),
                vec![
                    report(1, Some(4.0), Some(-55), false),
                    report(2, Some(1e-9), Some(0), false),
                    report(3, Some(0.1 + 0.7), Some(-70), false),
                ],
            ),
            epoch(
                2.2,
                Outcome::Abstain(Reason::Degenerate),
                Outcome::Abstain(Reason::NotConverged),
                vec![
                    report(1, None, None, true),
                    report(3, Some(2.0), None, false),
                ],
            ),
        ],
    }
}

fn edge_text() -> String {
    write(&edge_records())
}

#[test]
fn hand_built_edge_cases_round_trip_exactly() {
    let records = edge_records();
    let text = write(&records);
    let kinds: Vec<String> = text
        .lines()
        .map(|l| {
            serde_json::from_str::<Value>(l).unwrap()["type"]
                .as_str()
                .unwrap()
                .to_string()
        })
        .collect();
    assert_eq!(
        kinds,
        ["manifest", "epoch", "epoch", "tick", "tick", "epoch", "tick", "epoch"],
        "ordered by t, the epoch first at equal t"
    );
    let back = read(&text).unwrap_or_else(|e| panic!("{e}"));
    assert_exact(&back, &records);
    assert_eq!(write(&back), text);
}

/// [`edge_records`] as variant `empty`: nobody in any truth, no
/// `empty_run`, no phone. Lines 2-4 are the ticks.
fn empty_records() -> Records {
    let mut records = edge_records();
    let m = &mut records.manifest;
    m.variant = Variant::Empty;
    m.roomsim_config.n_people = Some(0);
    m.empty_run = None;
    m.has_phone = false;
    m.provenance = Provenance::Surveyed;
    records.epochs.clear();
    for t in &mut records.ticks {
        t.truth.clear();
    }
    records
}

#[test]
fn the_empty_variant_round_trips_without_empty_run_or_epochs() {
    let records = empty_records();
    let text = write(&records);
    let manifest: Value = serde_json::from_str(text.lines().next().unwrap()).unwrap();
    assert!(manifest.get("empty_run").is_none(), "absent, not null");
    let back = read(&text).unwrap_or_else(|e| panic!("{e}"));
    assert_exact(&back, &records);
}

#[test]
fn a_manifest_alone_is_a_run_with_no_records() {
    let mut records = edge_records();
    records.ticks.clear();
    records.links.clear();
    records.epochs.clear();
    let back = read(&write(&records)).unwrap_or_else(|e| panic!("{e}"));
    assert_exact(&back, &records);
}

#[test]
fn the_writer_refuses_what_it_cannot_encode_or_must_not_write() {
    let mut out = Vec::new();
    let mut r = edge_records();
    r.ticks[1].label = "REAL";
    let e = write_records(&mut out, &r).unwrap_err();
    assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput);
    let mut r = edge_records();
    r.epochs[0].label = "LIVE";
    assert!(write_records(&mut out, &r).is_err());
    let mut r = edge_records();
    r.links.pop();
    assert!(write_records(&mut out, &r).is_err());
    let mut r = edge_records();
    r.links[2].pop();
    assert!(write_records(&mut out, &r).is_err());
    assert!(out.is_empty(), "nothing is written before a refusal");
}

fn estimate_mut(o: &mut Outcome) -> &mut Estimate {
    match o {
        Outcome::Estimate(e) => e,
        Outcome::Abstain(_) => panic!("an estimate"),
    }
}

/// JSON has no non-finite number. A link's `y` is §4's `null` (read back
/// as NaN); any other non-finite value would come back changed (`None`)
/// or be refused by the reader, so the writer refuses it, naming where.
#[test]
fn the_writer_refuses_a_value_it_cannot_write_exactly() {
    type Change = Box<dyn Fn(&mut Records)>;
    let cases: Vec<(Change, &str)> = vec![
        (
            Box::new(|r| r.links[1][0].gate = Some(f64::INFINITY)),
            "links[1][0].gate",
        ),
        (
            Box::new(|r| estimate_mut(&mut r.ticks[1].wlm).confidence = Some(f64::NAN)),
            "ticks[1].wlm.estimate.confidence",
        ),
        (
            Box::new(|r| estimate_mut(&mut r.ticks[1].rti).cov_xx = f64::INFINITY),
            "ticks[1].rti.estimate.cov_xx",
        ),
        (Box::new(|r| r.ticks[2].t = f64::NAN), "ticks[2].t"),
        (
            Box::new(|r| r.ticks[1].truth[0].x = f64::NAN),
            "ticks[1].truth[0].x",
        ),
        (
            Box::new(|r| r.epochs[1].reports[0].rtt_m = Some(f64::NAN)),
            "epochs[1].reports[0].rtt_m",
        ),
        (
            Box::new(|r| estimate_mut(&mut r.epochs[1].rssi).radius_m = f64::NAN),
            "epochs[1].rssi.estimate.radius_m",
        ),
        (
            Box::new(|r| r.epochs[2].truth[2] = f64::NEG_INFINITY),
            "epochs[2].truth.z",
        ),
        (Box::new(|r| r.epochs[3].t = f64::NAN), "epochs[3].t"),
        (
            Box::new(|r| r.manifest.duration_s = f64::NAN),
            "manifest.duration_s",
        ),
        (
            Box::new(|r| r.manifest.empty_run.as_mut().unwrap().0.seconds = f64::NAN),
            "manifest.empty_run.wlm.seconds",
        ),
        (
            Box::new(|r| r.manifest.empty_run.as_mut().unwrap().0.per_h = Some(f64::INFINITY)),
            "manifest.empty_run.wlm.per_h",
        ),
        (
            Box::new(|r| r.manifest.empty_run.as_mut().unwrap().1.events_per_h = Some(f64::NAN)),
            "manifest.empty_run.rti.events_per_h",
        ),
        (
            Box::new(|r| r.manifest.locate_config.wlm_floor = Some(f64::NAN)),
            "manifest.locate_config",
        ),
        (
            Box::new(|r| r.manifest.locate_config.lambda_m = f64::INFINITY),
            "manifest.locate_config",
        ),
        (
            Box::new(|r| r.manifest.roomsim_config.slab_thickness_m = f64::NAN),
            "manifest.roomsim_config",
        ),
    ];
    for (f, needle) in cases {
        let mut r = edge_records();
        f(&mut r);
        let mut out = Vec::new();
        let e = write_records(&mut out, &r).expect_err(needle);
        assert_eq!(e.kind(), std::io::ErrorKind::InvalidInput, "{needle}");
        assert!(
            e.to_string().starts_with(needle),
            "{e}: expected {needle:?}"
        );
        assert!(
            out.is_empty(),
            "{needle}: nothing is written before a refusal"
        );
    }
    // A link's `y` alone is written `null` when it is not finite (§4).
    let mut r = edge_records();
    r.links[0][1].y = f64::NEG_INFINITY;
    let back = read(&write(&r)).unwrap_or_else(|e| panic!("{e}"));
    assert!(back.links[0][1].y.is_nan());
}

// ===================================================================
// The shared encodings (§4, spec/locate.md §11)
// ===================================================================

#[test]
fn the_encoders_give_exactly_section_4s_keys() {
    let est = estimate(1.0, 2.0, 3.0, None);
    let v = outcome_json(&est);
    assert_eq!(sorted_keys(&v), ["estimate"]);
    assert_eq!(
        sorted_keys(&v["estimate"]),
        sorted(vec![
            "x",
            "y",
            "radius_m",
            "cov_xx",
            "cov_xy",
            "cov_yy",
            "confidence",
            "radius_calibrated"
        ])
    );
    assert_eq!(v["estimate"]["confidence"], Value::Null);
    assert_eq!(v["estimate"]["radius_calibrated"], json!(false));
    assert_eq!(v["estimate"]["radius_m"], json!(3.0));
    let v = outcome_json(&estimate(1.0, 2.0, 3.0, Some(0.5)));
    assert_eq!(v["estimate"]["confidence"], json!(0.5));
    let v = outcome_json(&Outcome::Abstain(Reason::NoDisturbance));
    assert_eq!(v, json!({"abstain": {"reason": "no_disturbance"}}));

    let v = link_json(
        1,
        2,
        &LinkValue {
            y: f64::NAN,
            gate: None,
            disturbed: false,
        },
    );
    assert_eq!(
        v,
        json!({"tx": 1, "rx": 2, "y": null, "gate": null, "disturbed": false})
    );
    let v = link_json(
        3,
        1,
        &LinkValue {
            y: f64::NEG_INFINITY,
            gate: Some(0.75),
            disturbed: true,
        },
    );
    assert_eq!(
        v,
        json!({"tx": 3, "rx": 1, "y": null, "gate": 0.75, "disturbed": true})
    );

    // No `blocked` outside truth (§4 "Phone reports").
    let v = report_json(&report(4, None, None, true));
    assert_eq!(v, json!({"node": 4, "rtt_m": null, "rssi_dbm": null}));
    let v = report_json(&report(2, Some(3.5), Some(-61), false));
    assert_eq!(v, json!({"node": 2, "rtt_m": 3.5, "rssi_dbm": -61}));

    let people = edge_records().ticks[1].truth.clone();
    let v = tick_truth_json(&people);
    assert_eq!(v.as_array().unwrap().len(), 3);
    for (i, p) in v.as_array().unwrap().iter().enumerate() {
        assert_eq!(
            sorted_keys(p),
            sorted(vec!["i", "x", "y", "walking", "posture"])
        );
        assert_eq!(p["i"], json!(i));
    }
    assert_eq!(v[0]["posture"], "stand");
    assert_eq!(v[1]["posture"], "lie");
    assert_eq!(v[2]["posture"], "sit");
    assert_eq!(tick_truth_json(&[]), json!([]));

    let reports = vec![
        report(1, Some(1.0), None, true),
        report(2, None, Some(-50), false),
        report(4, None, None, true),
    ];
    let v = epoch_truth_json([1.0, 2.0, 1.25], &reports);
    assert_eq!(v, json!({"x": 1.0, "y": 2.0, "z": 1.25, "blocked": [1, 4]}));
    assert_eq!(
        epoch_truth_json([0.0, 0.0, 1.0], &[]),
        json!({"x": 0.0, "y": 0.0, "z": 1.0, "blocked": []})
    );
}

#[test]
fn the_file_uses_the_shared_encodings_in_the_spec_key_order() {
    let records = edge_records();
    let text = write(&records);
    let lines: Vec<&str> = text.lines().collect();
    assert!(
        lines[0].starts_with(
            r#"{"type":"manifest","label":"SIM","spec":"livedemo v1","scene":{"name":"test room","sha256":"#
        ),
        "{}",
        lines[0]
    );
    assert!(lines[3].starts_with(r#"{"type":"tick","label":"SIM","t":1.0,"wlm":{"abstain""#));
    assert!(lines[4].contains(r#"{"estimate":{"x":-0.0,"y":0.30000000000000004,"radius_m":4.2,"#));
    assert!(lines[1].starts_with(r#"{"type":"epoch","label":"SIM","t":0.0,"rtt":"#));

    let manifest: Value = serde_json::from_str(lines[0]).unwrap();
    assert_eq!(
        sorted_keys(&manifest),
        sorted(vec![
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
            "empty_run"
        ])
    );
    assert_eq!(
        sorted_keys(&manifest["scene"]),
        ["name", "provenance", "sha256"]
    );
    assert_eq!(manifest["scene"]["provenance"], "hand_written");
    assert_eq!(manifest["variant"], "declared");
    assert_eq!(manifest["seed"], json!(u64::MAX));
    assert_eq!(manifest["links"], json!([[1, 2], [2, 1], [1, 3]]));
    assert_eq!(sorted_keys(&manifest["empty_run"]), ["rti", "wlm"]);
    assert_eq!(
        manifest["locate_config"],
        locate_config_json(&Config::default())
    );
    assert_eq!(
        sorted_keys(&manifest["empty_run"]["wlm"]),
        sorted(vec![
            "ticks",
            "seconds",
            "published",
            "per_h",
            "events",
            "events_per_h"
        ])
    );

    // Ticks and epochs are exactly §6.1's keys, with the shared encodings.
    let (mut k, mut e) = (0, 0);
    for line in &lines[1..] {
        let v: Value = serde_json::from_str(line).unwrap();
        if v["type"] == "tick" {
            let t = &records.ticks[k];
            assert_eq!(
                sorted_keys(&v),
                sorted(vec!["type", "label", "t", "wlm", "rti", "links", "truth"])
            );
            assert_eq!(v["wlm"], outcome_json(&t.wlm));
            assert_eq!(v["rti"], outcome_json(&t.rti));
            assert_eq!(v["truth"], tick_truth_json(&t.truth));
            let items: Vec<Value> = records
                .manifest
                .links
                .iter()
                .zip(&records.links[k])
                .map(|(&(tx, rx), lv)| link_json(tx, rx, lv))
                .collect();
            assert_eq!(v["links"], Value::Array(items));
            k += 1;
        } else {
            let ep = &records.epochs[e];
            assert_eq!(
                sorted_keys(&v),
                sorted(vec![
                    "type", "label", "t", "rtt", "rssi", "reports", "truth"
                ])
            );
            assert_eq!(v["rtt"], outcome_json(&ep.rtt));
            assert_eq!(v["rssi"], outcome_json(&ep.rssi));
            let reports: Vec<Value> = ep.reports.iter().map(report_json).collect();
            assert_eq!(v["reports"], Value::Array(reports));
            assert_eq!(v["truth"], epoch_truth_json(ep.truth, &ep.reports));
            e += 1;
        }
    }
    assert_eq!((k, e), (3, 4));
}

#[test]
fn the_decoders_invert_the_encoders() {
    let records = edge_records();
    for t in &records.ticks {
        for o in [&t.wlm, &t.rti] {
            assert_eq!(outcome_from_json(&outcome_json(o)).unwrap(), *o);
        }
        assert_eq!(
            tick_truth_from_json(&tick_truth_json(&t.truth)).unwrap(),
            t.truth
        );
    }
    for (&(tx, rx), v) in records.manifest.links.iter().zip(&records.links[1]) {
        let (tx2, rx2, back) = link_from_json(&link_json(tx, rx, v)).unwrap();
        assert_eq!((tx2, rx2), (tx, rx));
        assert_eq!(back.y.to_bits(), v.y.to_bits());
        assert_eq!(back.gate, v.gate);
        assert_eq!(back.disturbed, v.disturbed);
    }
    for ep in &records.epochs {
        let mut reports: Vec<ReportRecord> = ep
            .reports
            .iter()
            .map(|r| report_from_json(&report_json(r)).unwrap())
            .collect();
        assert!(
            reports.iter().all(|r| !r.blocked),
            "a report carries no blocked"
        );
        let truth =
            epoch_truth_from_json(&epoch_truth_json(ep.truth, &ep.reports), &mut reports).unwrap();
        assert_eq!(format!("{truth:?}"), format!("{:?}", ep.truth));
        assert_eq!(reports, ep.reports);
    }
    let cfg = resolve_config(&RoomSimConfig::default(), &Config::default()).unwrap();
    assert_eq!(
        locate_config_from_json(&locate_config_json(&cfg)).unwrap(),
        cfg
    );
    let cfg = Config {
        wlm_floor: Some(0.5),
        rssi_p0_dbm: Some(-40.25),
        ..Config::default()
    };
    assert_eq!(
        locate_config_from_json(&locate_config_json(&cfg)).unwrap(),
        cfg
    );

    // `blocked` names a report's node, once.
    let mut reports = vec![report(1, None, None, false)];
    let truth = json!({"x": 0.0, "y": 0.0, "z": 1.0, "blocked": [2]});
    assert!(epoch_truth_from_json(&truth, &mut reports).is_err());
    let truth = json!({"x": 0.0, "y": 0.0, "z": 1.0, "blocked": [1, 1]});
    assert!(epoch_truth_from_json(&truth, &mut reports).is_err());
    // A report with `blocked` is not §4's.
    assert!(report_from_json(
        &json!({"node": 1, "rtt_m": null, "rssi_dbm": null, "blocked": true})
    )
    .is_err());
}

// ===================================================================
// Refusals (§6.2), each at its line
// ===================================================================

/// An id-like value planted in a file: the hall's `Person.id` "walker",
/// marked so that it differs from the variant name `with_walker`.
const PLANTED: &str = "walker#planted";

/// `text` with line `line` (1-based) parsed, changed by `f` and written
/// back.
fn edit(text: &str, line: usize, f: impl FnOnce(&mut Value)) -> String {
    let mut lines: Vec<String> = text.lines().map(String::from).collect();
    let mut v: Value = serde_json::from_str(&lines[line - 1]).expect("JSON");
    f(&mut v);
    lines[line - 1] = serde_json::to_string(&v).expect("encodes");
    lines.join("\n") + "\n"
}

fn replace_line(text: &str, line: usize, with: &str) -> String {
    let mut lines: Vec<&str> = text.lines().collect();
    lines[line - 1] = with;
    lines.join("\n") + "\n"
}

/// `text` is refused at `line`, with `needle` in the reason, which quotes
/// no value from the file (the planted id-like value [`PLANTED`]).
#[track_caller]
fn assert_refused(text: &str, line: usize, needle: &str) {
    let e = match read(text) {
        Ok(_) => panic!("accepted; expected a refusal at line {line} ({needle})"),
        Err(e) => e,
    };
    assert_eq!(e.line, line, "{e}");
    assert!(e.reason.contains(needle), "{e}: expected {needle:?}");
    assert!(!e.reason.contains(PLANTED), "{e}: quotes the file");
    assert!(e.to_string().starts_with(&format!("line {line}: ")));
}

#[test]
fn the_edge_file_is_accepted() {
    read(&edge_text()).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn refused_a_first_line_that_is_not_a_manifest() {
    let text = edge_text();
    let without: String = text.lines().skip(1).map(|l| format!("{l}\n")).collect();
    assert_refused(&without, 1, "the first line is not a manifest");
    assert_refused("", 1, "the first line is not a manifest");
    assert_refused(
        &format!("{PLANTED}\n"),
        1,
        "the first line is not a manifest",
    );
    assert_refused("[1, 2]\n", 1, "the first line is not a manifest");
    let typo = edit(&text, 1, |v| v["type"] = json!(PLANTED));
    assert_refused(&typo, 1, "the first line is not a manifest");
}

#[test]
fn refused_a_label_other_than_sim_on_any_line() {
    let text = edge_text();
    for line in 1..=8 {
        let bad = edit(&text, line, |v| v["label"] = json!(PLANTED));
        assert_refused(&bad, line, "label");
        let real = edit(&text, line, |v| v["label"] = json!("REAL"));
        assert_refused(&real, line, "label");
    }
    let missing = edit(&text, 5, |v| {
        v.as_object_mut().unwrap().remove("label");
    });
    assert_refused(&missing, 5, "label");
}

#[test]
fn refused_a_spec_other_than_livedemo_v1() {
    let bad = edit(&edge_text(), 1, |v| v["spec"] = json!("livedemo v2"));
    assert_refused(&bad, 1, "spec");
    let bad = edit(&edge_text(), 1, |v| v["spec"] = json!(PLANTED));
    assert_refused(&bad, 1, "spec");
}

#[test]
fn refused_a_slot_s_other_than_0_0125() {
    let bad = edit(&edge_text(), 1, |v| {
        v["roomsim_config"]["slot_s"] = json!(0.025)
    });
    assert_refused(&bad, 1, "roomsim_config.slot_s");
}

#[test]
fn refused_a_duration_that_is_not_a_whole_number_of_quanta() {
    for d in [2.51, 0.0, -2.5, 1e-3] {
        let bad = edit(&edge_text(), 1, |v| {
            v["duration_s"] = json!(d);
            v["roomsim_config"]["duration_s"] = json!(d);
        });
        assert_refused(&bad, 1, "duration_s");
    }
    // 40 * 2.525 = 101 quanta: accepted.
    let ok = edit(&edge_text(), 1, |v| {
        v["duration_s"] = json!(2.525);
        v["roomsim_config"]["duration_s"] = json!(2.525);
    });
    read(&ok).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn refused_ticks_out_of_order() {
    let text = edge_text();
    let lines: Vec<&str> = text.lines().collect();
    // Swap the ticks at 1.0 (line 4) and 1.5 (line 5).
    let swapped = replace_line(&replace_line(&text, 4, lines[4]), 5, lines[3]);
    assert_refused(&swapped, 5, "tick out of order");
    // A repeated tick time.
    let repeated = edit(&text, 5, |v| v["t"] = json!(1.0));
    assert_refused(&repeated, 5, "tick out of order");
}

#[test]
fn refused_epochs_out_of_order() {
    let text = edge_text();
    let back = edit(&text, 6, |v| v["t"] = json!(0.5));
    assert_refused(&back, 6, "epoch out of order");
    let repeated = edit(&text, 3, |v| v["t"] = json!(0.0));
    assert_refused(&repeated, 3, "epoch out of order");
}

#[test]
fn refused_records_not_ordered_by_t_with_the_epoch_first() {
    let text = edge_text();
    // An epoch after a later tick (epochs alone are still in order).
    let late = edit(&text, 6, |v| v["t"] = json!(1.2));
    assert_refused(&late, 6, "record out of order");
    // At equal t the epoch comes first.
    let equal = edit(&text, 6, |v| v["t"] = json!(1.5));
    assert_refused(&equal, 6, "epoch first at equal t");
}

#[test]
fn refused_a_record_at_or_after_duration_s() {
    let text = edge_text();
    assert_refused(&edit(&text, 7, |v| v["t"] = json!(2.5)), 7, "duration_s");
    assert_refused(&edit(&text, 8, |v| v["t"] = json!(2.5)), 8, "duration_s");
    assert_refused(&edit(&text, 8, |v| v["t"] = json!(1e300)), 8, "duration_s");
    assert_refused(&edit(&text, 2, |v| v["t"] = json!(-0.1)), 2, "negative");
}

#[test]
fn refused_an_unknown_key_anywhere() {
    let text = edge_text();
    // (line, path to the object) — every object of every line type.
    let cases: &[(usize, &[&str])] = &[
        (1, &[]),
        (1, &["scene"]),
        (1, &["roomsim_config"]),
        (1, &["locate_config"]),
        (1, &["empty_run"]),
        (1, &["empty_run", "wlm"]),
        (1, &["empty_run", "rti"]),
        (4, &[]),
        (4, &["wlm"]),
        (4, &["wlm", "abstain"]),
        (5, &["rti", "estimate"]),
        (5, &["links", "0"]),
        (5, &["truth", "2"]),
        (2, &[]),
        (2, &["reports", "1"]),
        (2, &["truth"]),
        (3, &["rssi", "estimate"]),
        (8, &["rtt", "abstain"]),
    ];
    for &(line, path) in cases {
        let bad = edit(&text, line, |v| {
            let mut o = v;
            for key in path {
                o = match key.parse::<usize>() {
                    Ok(i) => &mut o[i],
                    Err(_) => &mut o[*key],
                };
            }
            o.as_object_mut()
                .unwrap_or_else(|| panic!("line {line} {path:?} is an object"))
                .insert(PLANTED.into(), json!(1));
        });
        assert_refused(&bad, line, "unknown key");
    }
    // A report's `blocked` is truth (§4), never a report key.
    let blocked = edit(&text, 3, |v| v["reports"][0]["blocked"] = json!(true));
    assert_refused(&blocked, 3, "unknown key");
}

#[test]
fn refused_links_not_in_the_manifests_order_or_count() {
    let text = edge_text();
    let swapped = edit(&text, 5, |v| {
        v["links"].as_array_mut().unwrap().swap(0, 1);
    });
    assert_refused(&swapped, 5, "manifest's order");
    let short = edit(&text, 7, |v| {
        v["links"].as_array_mut().unwrap().pop();
    });
    assert_refused(&short, 7, "links: 2 items");
    let long = edit(&text, 4, |v| {
        let first = v["links"][0].clone();
        v["links"].as_array_mut().unwrap().push(first);
    });
    assert_refused(&long, 4, "links: 4 items");
    let reordered_manifest = edit(&text, 1, |v| {
        v["links"].as_array_mut().unwrap().swap(1, 2);
    });
    assert_refused(&reordered_manifest, 4, "manifest's order");
}

#[test]
fn refused_an_estimate_failing_the_publication_test() {
    let text = edge_text();
    let set = |line: usize, key: &'static str, field: &'static str, x: Value| {
        edit(&text, line, move |v| v[key]["estimate"][field] = x)
    };
    // Below min_plausible_radius_m (0.3), zero, negative.
    for r in [0.29, 0.0, -1.0] {
        assert_refused(&set(5, "wlm", "radius_m", json!(r)), 5, "publication test");
        assert_refused(&set(6, "rtt", "radius_m", json!(r)), 6, "publication test");
    }
    // Not a number at all (null is how a non-finite value would be sent).
    assert_refused(
        &set(5, "rti", "radius_m", Value::Null),
        5,
        "rti.estimate.radius_m",
    );
    assert_refused(&set(5, "wlm", "x", Value::Null), 5, "wlm.estimate.x");
    assert_refused(&set(3, "rssi", "y", json!(PLANTED)), 3, "rssi.estimate.y");
    // The manifest's minimum is the one applied: RTI's 0.3 m at line 5
    // fails a minimum of 0.31.
    let raised = edit(&text, 1, |v| {
        v["locate_config"]["min_plausible_radius_m"] = json!(0.31)
    });
    assert_refused(&raised, 5, "rti.estimate: fails the publication test");
}

#[test]
fn refused_radius_calibrated_true() {
    let bad = edit(&edge_text(), 5, |v| {
        v["rti"]["estimate"]["radius_calibrated"] = json!(true)
    });
    assert_refused(&bad, 5, "radius_calibrated");
    let bad = edit(&edge_text(), 3, |v| {
        v["rssi"]["estimate"]["radius_calibrated"] = json!(true)
    });
    assert_refused(&bad, 3, "radius_calibrated");
}

#[test]
fn refused_a_file_larger_than_max_bytes() {
    let text = edge_text();
    let lens: Vec<u64> = text.lines().map(|l| l.len() as u64 + 1).collect();
    let total: u64 = lens.iter().sum();
    assert_eq!(total, text.len() as u64);
    read_records(text.as_bytes(), total).unwrap_or_else(|e| panic!("{e}"));
    let e = read_records(text.as_bytes(), total - 1).unwrap_err();
    assert_eq!(e.line, 8, "{e}");
    assert!(e.reason.contains("larger than"), "{e}");
    let first_three: u64 = lens[..3].iter().sum();
    let e = read_records(text.as_bytes(), first_three + 10).unwrap_err();
    assert_eq!(e.line, 4, "{e}");
    assert!(e.reason.contains("larger than"), "{e}");
    let e = read_records(text.as_bytes(), 0).unwrap_err();
    assert_eq!(e.line, 1, "{e}");
}

#[test]
fn refused_a_bad_variant_name() {
    for name in [PLANTED, "walker", "scene", "WITH_WALKER", ""] {
        let bad = edit(&edge_text(), 1, |v| v["variant"] = json!(name));
        assert_refused(&bad, 1, "variant");
    }
    let bad = edit(&edge_text(), 1, |v| v["variant"] = json!(1));
    assert_refused(&bad, 1, "variant");
}

#[test]
fn refused_epochs_when_has_phone_is_false() {
    let bad = edit(&edge_text(), 1, |v| v["has_phone"] = json!(false));
    assert_refused(&bad, 2, "has_phone");
}

// --- The reader's other refusals (spec/locate.md §11 form, manifest
// consistency) ---

#[test]
fn refused_lines_that_are_not_records() {
    let text = edge_text();
    assert_refused(&replace_line(&text, 4, PLANTED), 4, "not JSON");
    assert_refused(&replace_line(&text, 4, ""), 4, "empty line");
    assert_refused(&replace_line(&text, 4, "[]"), 4, "not a JSON object");
    let second = text.lines().next().unwrap().to_string();
    assert_refused(&replace_line(&text, 6, &second), 6, "a second manifest");
    assert_refused(&edit(&text, 6, |v| v["type"] = json!(PLANTED)), 6, "type");
    let mut bytes = text.clone().into_bytes();
    let at = text.lines().take(4).map(|l| l.len() + 1).sum::<usize>() + 2;
    bytes[at] = 0xff;
    let e = read_records(&bytes[..], MAX_BYTES).unwrap_err();
    assert_eq!((e.line, e.reason.as_str()), (5, "not UTF-8"));
}

#[test]
fn refused_a_missing_key() {
    let text = edge_text();
    for (line, key) in [(1, "seed"), (1, "has_phone"), (4, "truth"), (2, "reports")] {
        let bad = edit(&text, line, |v| {
            v.as_object_mut().unwrap().remove(key);
        });
        assert_refused(&bad, line, &format!("{key}: missing"));
    }
    let bad = edit(&text, 1, |v| {
        v["locate_config"]
            .as_object_mut()
            .unwrap()
            .remove("tick_hz");
    });
    assert_refused(&bad, 1, "locate_config.tick_hz: missing");
    let bad = edit(&text, 1, |v| {
        v["roomsim_config"].as_object_mut().unwrap().remove("seed");
    });
    assert_refused(&bad, 1, "roomsim_config.seed: missing");
}

#[test]
fn refused_a_manifest_inconsistent_with_itself() {
    let text = edge_text();
    type Change = Box<dyn Fn(&mut Value)>;
    let cases: Vec<(Change, &str)> = vec![
        (
            Box::new(|v| v["roomsim_config"]["seed"] = json!(7)),
            "roomsim_config.seed",
        ),
        (
            Box::new(|v| v["roomsim_config"]["duration_s"] = json!(5.0)),
            "roomsim_config.duration_s",
        ),
        (
            Box::new(|v| v["roomsim_config"]["channel"] = json!(99)),
            "roomsim_config.channel",
        ),
        (
            Box::new(|v| v["roomsim_config"]["floor_material"] = json!(PLANTED)),
            "roomsim_config",
        ),
        (
            Box::new(|v| v["locate_config"]["tick_hz"] = json!(-2.0)),
            "locate_config.tick_hz",
        ),
        (
            Box::new(|v| v["locate_config"]["link_signal"] = json!(PLANTED)),
            "locate_config.link_signal",
        ),
        (
            Box::new(|v| v["scene"]["sha256"] = json!(PLANTED)),
            "scene.sha256",
        ),
        (
            Box::new(|v| v["scene"]["sha256"] = json!("0123456789ABCDEF".repeat(4))),
            "scene.sha256",
        ),
        (
            Box::new(|v| v["scene"]["provenance"] = json!(PLANTED)),
            "scene.provenance",
        ),
        (
            Box::new(|v| {
                v.as_object_mut().unwrap().remove("empty_run");
            }),
            "empty_run: missing",
        ),
        (
            Box::new(|v| v["variant"] = json!("empty")),
            "roomsim_config.n_people",
        ),
        (
            Box::new(|v| {
                v["variant"] = json!("empty");
                v["roomsim_config"]["n_people"] = json!(0);
            }),
            "empty_run: present",
        ),
        (
            Box::new(|v| v["variant"] = json!("with_walker")),
            "roomsim_config.phone_carrier",
        ),
        (Box::new(|v| v["links"][1] = json!([2, 1, 0])), "links[1]"),
        (Box::new(|v| v["seed"] = json!(-1)), "seed"),
        (Box::new(|v| v["seed"] = json!(1.0)), "seed"),
    ];
    for (f, needle) in cases {
        assert_refused(&edit(&text, 1, f), 1, needle);
    }
}

#[test]
fn refused_a_reason_its_estimator_never_gives() {
    let text = edge_text();
    let bad = edit(&text, 4, |v| {
        v["wlm"]["abstain"]["reason"] = json!("not_significant")
    });
    assert_refused(&bad, 4, "wlm.abstain.reason");
    let bad = edit(&text, 4, |v| {
        v["rti"]["abstain"]["reason"] = json!("no_disturbance")
    });
    assert_refused(&bad, 4, "rti.abstain.reason");
    let bad = edit(&text, 2, |v| {
        v["rtt"]["abstain"]["reason"] = json!("no_disturbance")
    });
    assert_refused(&bad, 2, "rtt.abstain.reason");
    let bad = edit(&text, 2, |v| {
        v["rssi"]["abstain"]["reason"] = json!(PLANTED)
    });
    assert_refused(&bad, 2, "rssi.abstain.reason");
    let both = edit(&text, 4, |v| {
        v["wlm"]["estimate"] = json!({});
    });
    assert_refused(&both, 4, "not exactly one of estimate and abstain");
}

#[test]
fn refused_reports_and_truth_out_of_their_form() {
    let text = edge_text();
    let unordered = edit(&text, 3, |v| {
        v["reports"].as_array_mut().unwrap().swap(0, 1);
    });
    assert_refused(&unordered, 3, "reports[1].node");
    let blocked = edit(&text, 3, |v| v["truth"]["blocked"] = json!([1, 9]));
    assert_refused(&blocked, 3, "truth.blocked[1]");
    let node = edit(&text, 3, |v| v["reports"][0]["node"] = json!(256));
    assert_refused(&node, 3, "reports[0].node");
    let rssi = edit(&text, 3, |v| v["reports"][1]["rssi_dbm"] = json!(-61.5));
    assert_refused(&rssi, 3, "reports[1].rssi_dbm");
    let index = edit(&text, 5, |v| v["truth"][1]["i"] = json!(2));
    assert_refused(&index, 5, "truth[1].i");
    let posture = edit(&text, 5, |v| v["truth"][0]["posture"] = json!(PLANTED));
    assert_refused(&posture, 5, "truth[0].posture");
}

#[test]
fn refused_truth_that_is_not_the_runs_people() {
    // Room Sim lists every person in every truth (spec/roomsim.md §13):
    // `n_people` of them when set (here variant `empty`: nobody) ...
    let text = write(&empty_records());
    let person = json!({"i": 0, "x": 1.0, "y": 1.0, "walking": false, "posture": "stand"});
    let one = edit(&text, 3, |v| v["truth"] = json!([person]));
    assert_refused(&one, 3, "truth: not roomsim_config.n_people people");
    // ... with_walker's scene people and the walker ...
    let walker = edit(&edge_text(), 1, |v| {
        v["variant"] = json!("with_walker");
        v["roomsim_config"]["n_people"] = json!(4);
        v["roomsim_config"]["phone_carrier"] = json!(3);
    });
    assert_refused(&walker, 4, "truth: not roomsim_config.n_people people");
    // ... and otherwise the scene's, the same at every tick.
    let fewer = edit(&edge_text(), 7, |v| {
        v["truth"].as_array_mut().unwrap().pop();
    });
    assert_refused(&fewer, 7, "truth: not as many people as the first tick's");
    let more = edit(&edge_text(), 5, |v| {
        let mut extra = v["truth"][0].clone();
        extra["i"] = json!(3);
        v["truth"].as_array_mut().unwrap().push(extra);
    });
    assert_refused(&more, 5, "truth: not as many people as the first tick's");
}

#[test]
fn refused_a_key_repeated_in_one_object() {
    let text = edge_text();
    let lines: Vec<&str> = text.lines().collect();
    // serde_json's `Value` would keep the last, SIM; the line says REAL.
    for line in [1, 2, 5] {
        for first in ["REAL", PLANTED] {
            let twice = lines[line - 1].replacen(
                r#""label":"SIM""#,
                &format!(r#""label":"{first}","label":"SIM""#),
                1,
            );
            assert_ne!(twice, lines[line - 1]);
            assert_refused(&replace_line(&text, line, &twice), line, "repeated key");
        }
    }
    // Nested: an unpublishable radius hidden behind a plausible one.
    let radius = lines[4].replacen(r#""radius_m":4.2"#, r#""radius_m":0.0,"radius_m":4.2"#, 1);
    assert_ne!(radius, lines[4]);
    assert_refused(&replace_line(&text, 5, &radius), 5, "repeated key");
    let calibrated = lines[4].replacen(
        r#""radius_calibrated":false"#,
        r#""radius_calibrated":true,"radius_calibrated":false"#,
        1,
    );
    assert_refused(&replace_line(&text, 5, &calibrated), 5, "repeated key");
    // One key in two objects is not repeated.
    assert!(lines[4].matches(r#""x":"#).count() > 1);
    read(&text).unwrap_or_else(|e| panic!("{e}"));
}

#[test]
fn refused_configs_that_run_roomsim_refuses() {
    let text = edge_text();
    type Change = Box<dyn Fn(&mut Value)>;
    let cases: Vec<(Change, &str)> = vec![
        // Not the Room Sim run's (spec/locate.md §9, resolve_config).
        (
            Box::new(|v| v["locate_config"]["phone_seed"] = json!(99)),
            "locate_config.phone_seed",
        ),
        (
            Box::new(|v| v["locate_config"]["tx_power_dbm"] = json!(-3.0)),
            "locate_config.tx_power_dbm",
        ),
        (
            Box::new(|v| v["locate_config"]["carrier_hz"] = json!(5.0e9)),
            "locate_config.carrier_hz",
        ),
        // Refused without a scene (check_roomsim_config).
        (
            Box::new(|v| v["locate_config"]["tick_hz"] = json!(100.0)),
            "locate_config.tick_hz",
        ),
        (
            Box::new(|v| v["roomsim_config"]["roles"] = json!({"1": "tx"})),
            "roomsim_config.roles",
        ),
        (
            Box::new(|v| v["roomsim_config"]["quantize"] = json!(false)),
            "roomsim_config.quantize",
        ),
    ];
    for (f, needle) in cases {
        assert_refused(&edit(&text, 1, f), 1, needle);
    }
    // The run's own values, set, are accepted.
    let rs = edge_records().manifest.roomsim_config;
    let resolved = locate_config_json(&resolve_config(&rs, &Config::default()).unwrap());
    let set = edit(&text, 1, |v| v["locate_config"] = resolved);
    read(&set).unwrap_or_else(|e| panic!("{e}"));
}
