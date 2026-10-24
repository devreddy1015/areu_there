//! The live demo's score (spec/livedemo.md §5, §8.2) on hand-built
//! records: medians, nearest-rank p90, `<=` for inside, nulls at count 0,
//! unoccupied ticks and events, the nearest-person tie, the reference
//! centroid, the radius scale and §5.1's JSON keys. Every record here is
//! made up for the test; every figure is SIM.
#![cfg(feature = "roomsim")]

use std::collections::BTreeSet;
use std::path::PathBuf;

use saans_locate::phone::EpochRecord;
use saans_locate::pipeline::{PersonTruthRecord, TickRecord};
use saans_locate::radius::{Estimate, Outcome, Reason};
use saans_locate::score::{
    false_alarms, footprint_centroid, person_score, phone_score, radius_scale, score, FalseAlarms,
    PersonEstimator, RecordScore, Scores, Unscored,
};
use saans_roomsim::Posture;
use saans_scene::Scene;
use serde_json::{json, Value};

const TICK_HZ: f64 = 2.0;

fn shipped(name: &str) -> Scene {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../../spec/scenes")
        .join(name);
    Scene::load(path).expect("a shipped scene loads")
}

/// The bedroom's scene with a 4 m by 2 m rectangular footprint, so the
/// reference point is (2, 1).
fn rect_scene() -> Scene {
    let mut s = shipped("bedroom.json");
    s.room.w_m = 4.0;
    s.room.d_m = 2.0;
    s.room.polygon = None;
    s
}

fn est(x: f64, y: f64, radius_m: f64) -> Outcome {
    Outcome::Estimate(Estimate {
        x,
        y,
        radius_m,
        cov_xx: 1.0,
        cov_xy: 0.0,
        cov_yy: 1.0,
        confidence: None,
        radius_calibrated: false,
    })
}

const ABSTAIN: Outcome = Outcome::Abstain(Reason::NoDisturbance);
const RTI_ABSTAIN: Outcome = Outcome::Abstain(Reason::NotSignificant);

fn at(x: f64, y: f64) -> PersonTruthRecord {
    PersonTruthRecord {
        x,
        y,
        walking: false,
        posture: Posture::Stand,
    }
}

fn tick(k: usize, wlm: Outcome, rti: Outcome, truth: Vec<PersonTruthRecord>) -> TickRecord {
    TickRecord {
        label: "SIM",
        t: 3.0 + k as f64 / TICK_HZ,
        wlm,
        rti,
        truth,
    }
}

fn epoch(k: usize, rtt: Outcome, rssi: Outcome, truth: [f64; 3]) -> EpochRecord {
    EpochRecord {
        label: "SIM",
        t: k as f64,
        rtt,
        rssi,
        reports: Vec::new(),
        truth,
    }
}

/// Occupied ticks (one person at the origin) whose WLM estimates lie at
/// `(e, 0)`, so each error is exactly `e`; RTI abstains.
fn wlm_errors(errors: &[f64], radius_m: f64) -> Vec<TickRecord> {
    errors
        .iter()
        .enumerate()
        .map(|(k, &e)| tick(k, est(e, 0.0, radius_m), RTI_ABSTAIN, vec![at(0.0, 0.0)]))
        .collect()
}

fn wlm(ticks: &[TickRecord]) -> Scores {
    score(&rect_scene(), ticks, &[], false, TICK_HZ)
}

#[test]
fn medians_odd_and_even() {
    let s = wlm(&wlm_errors(&[10.0, 1.0, 2.0], 5.0));
    assert_eq!(s.wlm.occupied.median_m, Some(2.0));
    let s = wlm(&wlm_errors(&[4.0, 10.0, 1.0, 2.0], 5.0));
    assert_eq!(s.wlm.occupied.median_m, Some(3.0));
    // The radius median follows the same rule.
    let mut ticks = wlm_errors(&[1.0, 1.0], 5.0);
    ticks[1].wlm = est(1.0, 0.0, 2.0);
    assert_eq!(wlm(&ticks).wlm.occupied.radius_median_m, Some(3.5));
}

#[test]
fn p90_is_the_nearest_rank() {
    // m = 1: the only error; m = 9: ceil(8.1) = 9th; m = 10: ceil(9.0) =
    // 9th; m = 11: ceil(9.9) = 10th. Shuffled, so the sort is exercised.
    let cases: [(&[f64], f64); 4] = [
        (&[7.0], 7.0),
        (&[9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0], 9.0),
        (&[10.0, 1.0, 9.0, 2.0, 8.0, 3.0, 7.0, 4.0, 6.0, 5.0], 9.0),
        (
            &[11.0, 1.0, 10.0, 2.0, 9.0, 3.0, 8.0, 4.0, 7.0, 5.0, 6.0],
            10.0,
        ),
    ];
    for (errors, p90) in cases {
        let s = wlm(&wlm_errors(errors, 100.0));
        assert_eq!(s.wlm.occupied.p90_m, Some(p90), "m = {}", errors.len());
    }
}

#[test]
fn inside_radius_counts_error_equal_to_radius() {
    // (3, 4) from the origin is exactly 5.0 m.
    let ticks = vec![
        tick(0, est(3.0, 4.0, 5.0), RTI_ABSTAIN, vec![at(0.0, 0.0)]),
        tick(1, est(3.0, 4.0, 4.999), RTI_ABSTAIN, vec![at(0.0, 0.0)]),
        tick(2, est(0.0, 3.0, 3.0), RTI_ABSTAIN, vec![at(0.0, 0.0)]),
        tick(3, est(0.0, 3.0, 2.0), RTI_ABSTAIN, vec![at(0.0, 0.0)]),
    ];
    let s = wlm(&ticks);
    assert_eq!(s.wlm.occupied.inside_radius, Some(0.5));
    let rtt = score(
        &rect_scene(),
        &[],
        &[epoch(0, est(3.0, 4.0, 5.0), ABSTAIN, [0.0, 0.0, 1.0])],
        true,
        TICK_HZ,
    )
    .rtt
    .unwrap();
    assert_eq!(rtt.inside_radius, Some(1.0));
}

#[test]
fn every_figure_is_null_at_count_zero() {
    // No records at all.
    let s = score(&rect_scene(), &[], &[], true, TICK_HZ);
    for p in [&s.wlm, &s.rti] {
        assert_eq!(p.ticks, 0);
        let o = &p.occupied;
        assert_eq!((o.ticks, o.published), (0, 0));
        assert_eq!(o.coverage, None);
        assert_eq!((o.median_m, o.p90_m, o.inside_radius), (None, None, None));
        assert_eq!(o.radius_median_m, None);
        assert_eq!(
            (
                o.reference.median_m,
                o.reference.p90_m,
                o.reference.inside_radius
            ),
            (None, None, None)
        );
        assert_eq!(
            p.unoccupied,
            FalseAlarms {
                ticks: 0,
                seconds: 0.0,
                published: 0,
                per_h: None,
                events: 0,
                events_per_h: None,
            }
        );
    }
    for p in [s.rtt.unwrap(), s.rssi.unwrap()] {
        assert_eq!((p.epochs, p.published), (0, 0));
        assert_eq!(p.coverage, None);
        assert_eq!((p.median_m, p.p90_m, p.inside_radius), (None, None, None));
        assert_eq!(p.radius_median_m, None);
        assert_eq!(p.reference.median_m, None);
    }
    // In the JSON every `None` is a present key whose value is `null`, not
    // an absent key (indexing alone would read `Null` for either): the key
    // sets are §5.1's in full, and each such key holds `null`.
    let v = s.to_json();
    assert_eq!(keys(&v), set(&["wlm", "rti", "rtt", "rssi"]));
    for b in ["wlm", "rti"] {
        assert_eq!(keys(&v[b]), set(&["ticks", "occupied", "unoccupied"]));
        let o = &v[b]["occupied"];
        assert_eq!(keys(o), set(&OCCUPIED));
        assert_eq!(keys(&o["ref"]), set(&REF));
        assert_eq!(keys(&v[b]["unoccupied"]), set(&FALSE_ALARMS));
        for k in NULL_AT_ZERO {
            assert_eq!(o.get(k), Some(&Value::Null), "{b}.occupied.{k}");
        }
        for k in REF {
            assert_eq!(o["ref"].get(k), Some(&Value::Null), "{b}.occupied.ref.{k}");
        }
        for k in ["per_h", "events_per_h"] {
            assert_eq!(
                v[b]["unoccupied"].get(k),
                Some(&Value::Null),
                "{b}.unoccupied.{k}"
            );
        }
    }
    for b in ["rtt", "rssi"] {
        assert_eq!(keys(&v[b]), set(&PHONE));
        assert_eq!(keys(&v[b]["ref"]), set(&REF));
        for k in NULL_AT_ZERO {
            assert_eq!(v[b].get(k), Some(&Value::Null), "{b}.{k}");
        }
        for k in REF {
            assert_eq!(v[b]["ref"].get(k), Some(&Value::Null), "{b}.ref.{k}");
        }
    }
    // REPLAY's JSON keeps the null `coverage` and `radius_median_m` (§5.4).
    let t = s.to_json_truthless();
    for o in [
        &t["wlm"]["occupied"],
        &t["rti"]["occupied"],
        &t["rtt"],
        &t["rssi"],
    ] {
        for k in ["coverage", "radius_median_m"] {
            assert_eq!(o.get(k), Some(&Value::Null), "{k}");
        }
    }

    // Occupied ticks, all abstained: coverage 0, every error figure null.
    let ticks = vec![
        tick(0, ABSTAIN, RTI_ABSTAIN, vec![at(1.0, 1.0)]),
        tick(1, ABSTAIN, RTI_ABSTAIN, vec![at(1.0, 1.0)]),
    ];
    let s = score(
        &rect_scene(),
        &ticks,
        &[epoch(0, ABSTAIN, ABSTAIN, [1.0, 1.0, 1.0])],
        true,
        TICK_HZ,
    );
    let o = &s.wlm.occupied;
    assert_eq!((o.ticks, o.published, o.coverage), (2, 0, Some(0.0)));
    assert_eq!((o.median_m, o.p90_m, o.inside_radius), (None, None, None));
    assert_eq!(o.radius_median_m, None);
    assert_eq!(o.reference.inside_radius, None);
    let p = s.rtt.unwrap();
    assert_eq!((p.epochs, p.published, p.coverage), (1, 0, Some(0.0)));
    assert_eq!(p.median_m, None);

    // No phone: no phone blocks, whatever the epochs.
    let s = score(
        &rect_scene(),
        &ticks,
        &[epoch(0, est(1.0, 1.0, 1.0), ABSTAIN, [1.0, 1.0, 1.0])],
        false,
        TICK_HZ,
    );
    assert_eq!((s.rtt, s.rssi), (None, None));
}

#[test]
fn unoccupied_ticks_and_events() {
    // Nine unoccupied ticks, WLM published P P . P . . P P P: 6 published
    // ticks in 3 events over 9 / 2 Hz = 4.5 s.
    let pattern = [true, true, false, true, false, false, true, true, true];
    let ticks: Vec<TickRecord> = pattern
        .iter()
        .enumerate()
        .map(|(k, &p)| {
            let w = if p { est(1.0, 1.0, 1.0) } else { ABSTAIN };
            tick(k, w, RTI_ABSTAIN, Vec::new())
        })
        .collect();
    let s = wlm(&ticks);
    let fa = s.wlm.unoccupied;
    assert_eq!(fa.ticks, 9);
    assert_eq!(fa.seconds, 4.5);
    assert_eq!(fa.published, 6);
    assert_eq!(fa.events, 3);
    assert_eq!(fa.per_h, Some(6.0 / 4.5 * 3600.0));
    assert_eq!(fa.events_per_h, Some(3.0 / 4.5 * 3600.0));
    assert!((fa.per_h.unwrap() - 4800.0).abs() < 1e-9);
    assert!((fa.events_per_h.unwrap() - 2400.0).abs() < 1e-9);
    // Unoccupied ticks are not in `occupied`, and their estimates are
    // false alarms, not errors, nor radii of the occupied block.
    assert_eq!(s.wlm.ticks, 9);
    assert_eq!(s.wlm.occupied.ticks, 0);
    assert_eq!(s.wlm.occupied.published, 0);
    assert_eq!(s.wlm.occupied.median_m, None);
    assert_eq!(s.wlm.occupied.radius_median_m, None);
    assert_eq!(s.rti.unoccupied.published, 0);
    assert_eq!(s.rti.unoccupied.per_h, Some(0.0));
    assert_eq!(s.rti.unoccupied.events_per_h, Some(0.0));
    // `false_alarms` is the same figure, usable for the empty run (§5.3).
    assert_eq!(false_alarms(&ticks, PersonEstimator::Wlm, TICK_HZ), fa);
    assert_eq!(
        false_alarms(&ticks, PersonEstimator::Rti, TICK_HZ),
        s.rti.unoccupied
    );
    assert_eq!(false_alarms(&[], PersonEstimator::Wlm, TICK_HZ).per_h, None);

    // An occupied tick ends an event: two unoccupied published ticks with
    // an occupied published one between them are two events, and the
    // occupied tick counts in neither `ticks` nor `published`. The false
    // alarms' radius (9) is not in `occupied.radius_median_m`.
    let mixed = vec![
        tick(0, est(1.0, 1.0, 9.0), RTI_ABSTAIN, Vec::new()),
        tick(1, est(1.0, 1.0, 1.0), RTI_ABSTAIN, vec![at(1.0, 2.0)]),
        tick(2, est(1.0, 1.0, 9.0), RTI_ABSTAIN, Vec::new()),
    ];
    let s = wlm(&mixed);
    assert_eq!(
        (
            s.wlm.unoccupied.ticks,
            s.wlm.unoccupied.published,
            s.wlm.unoccupied.events
        ),
        (2, 2, 2)
    );
    assert_eq!(s.wlm.unoccupied.seconds, 1.0);
    assert_eq!(s.wlm.unoccupied.per_h, Some(7200.0));
    assert_eq!(s.wlm.occupied.ticks, 1);
    assert_eq!(s.wlm.occupied.published, 1);
    assert_eq!(s.wlm.occupied.median_m, Some(1.0));
    assert_eq!(s.wlm.occupied.radius_median_m, Some(1.0));
    assert_eq!(s.wlm.occupied.inside_radius, Some(1.0));
    assert_eq!(s.wlm.ticks, 3);
}

#[test]
fn record_scores_nearest_person_and_phone() {
    // Two people 3 m from the estimate: the lower index wins the tie.
    let people = [at(4.0, 4.0), at(3.0, 0.0), at(0.0, 3.0)];
    let e = est(0.0, 0.0, 1.0);
    assert_eq!(
        person_score(&e, &people),
        RecordScore::Scored {
            error_m: 3.0,
            person: Some(1)
        }
    );
    let swapped = [at(0.0, 3.0), at(3.0, 0.0)];
    assert_eq!(
        person_score(&e, &swapped),
        RecordScore::Scored {
            error_m: 3.0,
            person: Some(0)
        }
    );
    // A strictly nearer later person wins.
    let nearer = [at(0.0, 3.0), at(0.0, 2.0)];
    assert_eq!(
        person_score(&e, &nearer),
        RecordScore::Scored {
            error_m: 2.0,
            person: Some(1)
        }
    );
    assert_eq!(
        person_score(&e, &[]),
        RecordScore::Unscored(Unscored::NoPerson)
    );
    assert_eq!(
        person_score(&ABSTAIN, &people),
        RecordScore::Unscored(Unscored::Abstained)
    );
    assert_eq!(
        person_score(&ABSTAIN, &[]),
        RecordScore::Unscored(Unscored::Abstained)
    );
    // The phone's error is in (x, y); z is not used.
    assert_eq!(
        phone_score(&est(3.0, 4.0, 1.0), [0.0, 0.0, 1.0]),
        RecordScore::Scored {
            error_m: 5.0,
            person: None
        }
    );
    assert_eq!(
        phone_score(&ABSTAIN, [0.0, 0.0, 1.0]),
        RecordScore::Unscored(Unscored::Abstained)
    );
    // §4's encodings.
    assert_eq!(
        person_score(&e, &people).to_json(),
        json!({"error_m": 3.0, "person": 1})
    );
    assert_eq!(
        phone_score(&est(3.0, 4.0, 1.0), [0.0, 0.0, 1.0]).to_json(),
        json!({"error_m": 5.0})
    );
    assert_eq!(
        person_score(&e, &[]).to_json(),
        json!({"unscored": "no_person"})
    );
    assert_eq!(
        phone_score(&ABSTAIN, [0.0, 0.0, 0.0]).to_json(),
        json!({"unscored": "abstained"})
    );
}

/// `locate_measure`'s `footprint_centroid`, verbatim
/// (examples/locate_measure.rs): the score's must give the same result.
fn example_footprint_centroid(scene: &Scene) -> (f64, f64) {
    let Some(poly) = scene.room.polygon.as_ref().filter(|p| p.len() >= 3) else {
        return (0.5 * scene.room.w_m, 0.5 * scene.room.d_m);
    };
    let (mut a2, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for (i, &(x0, y0)) in poly.iter().enumerate() {
        let (x1, y1) = poly[(i + 1) % poly.len()];
        let c = x0 * y1 - x1 * y0;
        a2 += c;
        cx += (x0 + x1) * c;
        cy += (y0 + y1) * c;
    }
    (cx / (3.0 * a2), cy / (3.0 * a2))
}

fn close(a: (f64, f64), b: (f64, f64)) -> bool {
    (a.0 - b.0).abs() < 1e-12 && (a.1 - b.1).abs() < 1e-12
}

#[test]
fn reference_centroid_of_a_rectangle_and_an_l() {
    // A bare rectangle: its centre, exactly.
    let bedroom = shipped("bedroom.json");
    assert!(bedroom.room.polygon.is_none());
    assert_eq!(footprint_centroid(&bedroom), (2.1, 1.8));
    assert_eq!(footprint_centroid(&rect_scene()), (2.0, 1.0));
    // The same rectangle outlined, either way round: the shoelace centroid.
    let mut outlined = rect_scene();
    outlined.room.polygon = Some(vec![(0.0, 0.0), (4.0, 0.0), (4.0, 2.0), (0.0, 2.0)]);
    assert!(close(footprint_centroid(&outlined), (2.0, 1.0)));
    outlined.room.polygon = Some(vec![(0.0, 0.0), (0.0, 2.0), (4.0, 2.0), (4.0, 0.0)]);
    assert!(close(footprint_centroid(&outlined), (2.0, 1.0)));
    // A degenerate outline (under 3 vertices) falls back to the rectangle.
    outlined.room.polygon = Some(vec![(0.0, 0.0), (4.0, 0.0)]);
    assert_eq!(footprint_centroid(&outlined), (2.0, 1.0));

    // The shipped L-shaped hall: a 9 x 3 slab (centroid (4.5, 1.5)) and a
    // 4 x 4 wing (centroid (2, 5)), area 43; and bit for bit the example's.
    let hall = shipped("hall-l-shaped.json");
    assert!(hall.room.polygon.is_some());
    let c = footprint_centroid(&hall);
    assert!(close(c, (153.5 / 43.0, 120.5 / 43.0)), "{c:?}");
    assert_eq!(c, example_footprint_centroid(&hall));
    for name in ["bedroom.json", "office.json"] {
        let s = shipped(name);
        assert_eq!(footprint_centroid(&s), example_footprint_centroid(&s));
    }
}

#[test]
fn reference_is_scored_on_the_same_records_with_their_radii() {
    // Reference (2, 1). Tick 0: person at (2, 4), reference error 3, WLM
    // radius 3 (inside). Tick 1: people at (2, 6) and (6, 1), the nearest
    // to the reference is (6, 1) at 4 while the estimate is nearest
    // (2, 6); WLM radius 3.5 (outside). Tick 2: WLM abstains, so it is not
    // a reference record.
    let ticks = vec![
        tick(0, est(2.0, 5.0, 3.0), RTI_ABSTAIN, vec![at(2.0, 4.0)]),
        tick(
            1,
            est(2.0, 6.5, 3.5),
            RTI_ABSTAIN,
            vec![at(2.0, 6.0), at(6.0, 1.0)],
        ),
        tick(2, ABSTAIN, RTI_ABSTAIN, vec![at(2.0, 1.0)]),
    ];
    let s = wlm(&ticks);
    let o = &s.wlm.occupied;
    assert_eq!((o.ticks, o.published), (3, 2));
    assert_eq!(o.coverage, Some(2.0 / 3.0));
    assert_eq!(o.median_m, Some(0.75));
    assert_eq!(o.p90_m, Some(1.0));
    assert_eq!(o.inside_radius, Some(1.0));
    assert_eq!(o.radius_median_m, Some(3.25));
    assert_eq!(o.reference.median_m, Some(3.5));
    assert_eq!(o.reference.p90_m, Some(4.0));
    assert_eq!(o.reference.inside_radius, Some(0.5));
    // RTI published nothing: its reference figures are null.
    assert_eq!(s.rti.occupied.reference.median_m, None);

    // The phone: reference (2, 1) against each epoch's (x, y).
    let epochs = vec![
        epoch(0, est(2.0, 4.0, 1.0), ABSTAIN, [2.0, 5.0, 1.0]),
        epoch(1, est(5.0, 1.0, 3.0), ABSTAIN, [5.0, 1.0, 0.2]),
        epoch(2, ABSTAIN, ABSTAIN, [0.0, 0.0, 1.0]),
    ];
    let s = score(&rect_scene(), &[], &epochs, true, TICK_HZ);
    let p = s.rtt.unwrap();
    assert_eq!((p.epochs, p.published, p.coverage), (3, 2, Some(2.0 / 3.0)));
    assert_eq!(p.median_m, Some(0.5));
    assert_eq!(p.p90_m, Some(1.0));
    assert_eq!(p.inside_radius, Some(1.0));
    assert_eq!(p.radius_median_m, Some(2.0));
    assert_eq!(p.reference.median_m, Some(3.5));
    assert_eq!(p.reference.p90_m, Some(4.0));
    assert_eq!(p.reference.inside_radius, Some(0.5));
    let q = s.rssi.unwrap();
    assert_eq!((q.epochs, q.published, q.coverage), (3, 0, Some(0.0)));
}

#[test]
fn radius_scale_is_the_nearest_rank_95th_ratio_from_20() {
    // Errors k = 1..=n with radius 2: ratios k / 2.
    let ratios = |n: usize| -> Vec<TickRecord> {
        let errors: Vec<f64> = (1..=n).rev().map(|k| k as f64).collect();
        wlm_errors(&errors, 2.0)
    };
    let scene = rect_scene();
    assert_eq!(radius_scale(&scene, &ratios(19), &[], false).wlm, None);
    // n = 20: ceil(19.0) = 19th; n = 21: ceil(19.95) = 20th.
    assert_eq!(radius_scale(&scene, &ratios(20), &[], false).wlm, Some(9.5));
    assert_eq!(
        radius_scale(&scene, &ratios(21), &[], false).wlm,
        Some(10.0)
    );
    let r = radius_scale(&scene, &ratios(40), &[], false);
    assert_eq!(r.wlm, Some(19.0));
    assert_eq!((r.rti, r.rtt, r.rssi), (None, None, None));
    // The same estimates as RTI's (WLM abstaining): RTI's figure.
    let swapped: Vec<TickRecord> = ratios(20)
        .into_iter()
        .map(|mut t| {
            t.rti = std::mem::replace(&mut t.wlm, ABSTAIN);
            t
        })
        .collect();
    let r = radius_scale(&scene, &swapped, &[], false);
    assert_eq!((r.wlm, r.rti), (None, Some(9.5)));
    // Unoccupied estimates are not scored, so they do not count.
    let mut ticks = ratios(19);
    ticks.push(tick(19, est(1.0, 1.0, 1.0), RTI_ABSTAIN, Vec::new()));
    assert_eq!(radius_scale(&scene, &ticks, &[], false).wlm, None);
    // The phone: 20 epochs, errors 1..=20 with radius 1.
    let epochs: Vec<EpochRecord> = (1..=20)
        .map(|k| epoch(k, ABSTAIN, est(k as f64, 0.0, 1.0), [0.0, 0.0, 1.0]))
        .collect();
    let r = radius_scale(&scene, &[], &epochs, true);
    assert_eq!((r.rtt, r.rssi), (None, Some(19.0)));
    assert_eq!(radius_scale(&scene, &[], &epochs, false).rssi, None);
}

fn keys(v: &Value) -> BTreeSet<&str> {
    v.as_object()
        .expect("an object")
        .keys()
        .map(String::as_str)
        .collect()
}

fn set<'a>(ks: &[&'a str]) -> BTreeSet<&'a str> {
    ks.iter().copied().collect()
}

const OCCUPIED: [&str; 8] = [
    "ticks",
    "published",
    "coverage",
    "median_m",
    "p90_m",
    "inside_radius",
    "radius_median_m",
    "ref",
];
const PHONE: [&str; 8] = [
    "epochs",
    "published",
    "coverage",
    "median_m",
    "p90_m",
    "inside_radius",
    "radius_median_m",
    "ref",
];
const REF: [&str; 3] = ["median_m", "p90_m", "inside_radius"];
/// The `occupied` and phone keys that are `null` when nothing is counted.
const NULL_AT_ZERO: [&str; 5] = [
    "coverage",
    "median_m",
    "p90_m",
    "inside_radius",
    "radius_median_m",
];
const FALSE_ALARMS: [&str; 6] = [
    "ticks",
    "seconds",
    "published",
    "per_h",
    "events",
    "events_per_h",
];

/// A few occupied and unoccupied ticks and epochs, so every block has
/// figures.
fn mixed_records() -> (Vec<TickRecord>, Vec<EpochRecord>) {
    let ticks = vec![
        tick(
            0,
            est(1.0, 1.0, 2.0),
            est(1.5, 1.0, 1.0),
            vec![at(1.0, 2.0)],
        ),
        tick(1, est(1.0, 1.0, 2.0), RTI_ABSTAIN, vec![at(1.0, 2.0)]),
        tick(2, est(1.0, 1.0, 2.0), RTI_ABSTAIN, Vec::new()),
    ];
    let epochs = vec![epoch(
        0,
        est(1.0, 1.0, 1.0),
        est(2.0, 1.0, 2.0),
        [1.0, 1.5, 1.0],
    )];
    (ticks, epochs)
}

#[test]
fn to_json_has_exactly_section_5_1s_keys() {
    let (ticks, epochs) = mixed_records();
    let s = score(&rect_scene(), &ticks, &epochs, true, TICK_HZ);
    let v = s.to_json();
    assert_eq!(keys(&v), set(&["wlm", "rti", "rtt", "rssi"]));
    for (name, p) in [("wlm", &s.wlm), ("rti", &s.rti)] {
        let b = &v[name];
        assert_eq!(keys(b), set(&["ticks", "occupied", "unoccupied"]));
        assert_eq!(keys(&b["occupied"]), set(&OCCUPIED));
        assert_eq!(keys(&b["occupied"]["ref"]), set(&REF));
        assert_eq!(keys(&b["unoccupied"]), set(&FALSE_ALARMS));
        // `unoccupied` is FalseAlarms' own serde encoding.
        assert_eq!(b["unoccupied"], serde_json::to_value(p.unoccupied).unwrap());
        assert_eq!(b["ticks"], json!(p.ticks));
        let o = &p.occupied;
        assert_eq!(
            b["occupied"],
            json!({
                "ticks": o.ticks, "published": o.published, "coverage": o.coverage,
                "median_m": o.median_m, "p90_m": o.p90_m,
                "inside_radius": o.inside_radius, "radius_median_m": o.radius_median_m,
                "ref": {"median_m": o.reference.median_m, "p90_m": o.reference.p90_m,
                        "inside_radius": o.reference.inside_radius}
            })
        );
    }
    for (name, p) in [("rtt", s.rtt.unwrap()), ("rssi", s.rssi.unwrap())] {
        let b = &v[name];
        assert_eq!(keys(b), set(&PHONE));
        assert_eq!(keys(&b["ref"]), set(&REF));
        assert_eq!(
            *b,
            json!({
                "epochs": p.epochs, "published": p.published, "coverage": p.coverage,
                "median_m": p.median_m, "p90_m": p.p90_m,
                "inside_radius": p.inside_radius, "radius_median_m": p.radius_median_m,
                "ref": {"median_m": p.reference.median_m, "p90_m": p.reference.p90_m,
                        "inside_radius": p.reference.inside_radius}
            })
        );
    }
    // The figures themselves (RTI: one published, 0.5 m from (1, 2)).
    assert_eq!(v["rti"]["occupied"]["published"], json!(1));
    assert_eq!(v["rti"]["occupied"]["median_m"], json!(1.25_f64.sqrt()));
    assert_eq!(v["wlm"]["unoccupied"]["published"], json!(1));
    assert_eq!(v["rssi"]["median_m"], json!(1.25_f64.sqrt()));

    // No phone: no `rtt` or `rssi` key (absent, not null).
    let v = score(&rect_scene(), &ticks, &[], false, TICK_HZ).to_json();
    assert_eq!(keys(&v), set(&["wlm", "rti"]));
}

#[test]
fn to_json_truthless_deletes_the_error_figures() {
    let (ticks, epochs) = mixed_records();
    let s = score(&rect_scene(), &ticks, &epochs, true, TICK_HZ);
    let full = s.to_json();
    let v = s.to_json_truthless();
    assert_eq!(keys(&v), set(&["wlm", "rti", "rtt", "rssi"]));
    for name in ["wlm", "rti"] {
        assert_eq!(keys(&v[name]), set(&["ticks", "occupied", "unoccupied"]));
        assert_eq!(
            keys(&v[name]["occupied"]),
            set(&["ticks", "published", "coverage", "radius_median_m"])
        );
        for k in ["ticks", "published", "coverage", "radius_median_m"] {
            assert_eq!(v[name]["occupied"][k], full[name]["occupied"][k]);
        }
        assert_eq!(v[name]["ticks"], full[name]["ticks"]);
        assert_eq!(v[name]["unoccupied"], full[name]["unoccupied"]);
    }
    for name in ["rtt", "rssi"] {
        assert_eq!(
            keys(&v[name]),
            set(&["epochs", "published", "coverage", "radius_median_m"])
        );
        for k in ["epochs", "published", "coverage", "radius_median_m"] {
            assert_eq!(v[name][k], full[name][k]);
        }
    }
    // Without a phone it stays without phone keys.
    let v = score(&rect_scene(), &ticks, &[], false, TICK_HZ).to_json_truthless();
    assert_eq!(keys(&v), set(&["wlm", "rti"]));
}
