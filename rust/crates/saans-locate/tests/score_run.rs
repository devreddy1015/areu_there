//! The live demo's score (spec/livedemo.md §5, §8.2) on real Room Sim
//! runs of the shipped bedroom at 20 s: every figure equals a direct
//! recomputation written here from §5.2 (and `locate_measure`'s
//! arithmetic), not from the score module. Every figure is SIM.
#![cfg(feature = "roomsim")]

use std::path::PathBuf;

use saans_locate::phone::{Config, EpochRecord};
use saans_locate::pipeline::{run_roomsim, RoomSimRun, TickRecord};
use saans_locate::radius::Outcome;
use saans_locate::score::{
    false_alarms, person_score, phone_score, radius_scale, score, Accuracy, FalseAlarms,
    PersonEstimator, RecordScore,
};
use saans_locate::variant::Variant;
use saans_roomsim::Config as RoomSimConfig;
use saans_scene::Scene;

fn bedroom() -> Scene {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes/bedroom.json");
    Scene::load(path).expect("the shipped bedroom loads")
}

fn run(scene: &Scene, variant: Variant) -> RoomSimRun {
    let base = RoomSimConfig {
        seed: 0,
        duration_s: 20.0,
        ..RoomSimConfig::default()
    };
    run_roomsim(
        scene,
        &variant.apply(scene, &base),
        &Config::default(),
        "SIM",
    )
    .expect("the bedroom runs")
}

fn dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

/// The bedroom has no outline: the reference point is its centre.
fn centre(scene: &Scene) -> (f64, f64) {
    assert!(scene.room.polygon.is_none());
    (0.5 * scene.room.w_m, 0.5 * scene.room.d_m)
}

/// §5.2 from scratch over `(error, radius, reference error)` triples:
/// the median, the `ceil(0.9 m)`-th smallest as `locate_measure` writes
/// it, the fraction inside, the radius median and the reference's figures.
struct Direct {
    own: Accuracy,
    radius_median_m: Option<f64>,
    reference: Accuracy,
}

fn median(v: &[f64]) -> f64 {
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

fn accuracy(pairs: &[(f64, f64)]) -> Accuracy {
    if pairs.is_empty() {
        return Accuracy {
            median_m: None,
            p90_m: None,
            inside_radius: None,
        };
    }
    let mut e: Vec<f64> = pairs.iter().map(|p| p.0).collect();
    e.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = e.len();
    let rank = (0.9 * m as f64).ceil() as usize;
    let inside = pairs.iter().filter(|p| p.0 <= p.1).count();
    Accuracy {
        median_m: Some(median(&e)),
        p90_m: Some(e[rank - 1]),
        inside_radius: Some(inside as f64 / m as f64),
    }
}

fn direct(triples: &[(f64, f64, f64)]) -> Direct {
    let mut radii: Vec<f64> = triples.iter().map(|t| t.1).collect();
    radii.sort_by(|a, b| a.partial_cmp(b).unwrap());
    Direct {
        own: accuracy(&triples.iter().map(|t| (t.0, t.1)).collect::<Vec<_>>()),
        radius_median_m: (!radii.is_empty()).then(|| median(&radii)),
        reference: accuracy(&triples.iter().map(|t| (t.2, t.1)).collect::<Vec<_>>()),
    }
}

fn nearest(x: f64, y: f64, tick: &TickRecord) -> f64 {
    tick.truth
        .iter()
        .map(|p| dist(x, y, p.x, p.y))
        .fold(f64::INFINITY, f64::min)
}

fn pick(which: PersonEstimator) -> fn(&TickRecord) -> &Outcome {
    match which {
        PersonEstimator::Wlm => |t| &t.wlm,
        PersonEstimator::Rti => |t| &t.rti,
    }
}

#[test]
fn bedroom_with_walker_scores_equal_a_direct_recomputation() {
    let scene = bedroom();
    let r = run(&scene, Variant::WithWalker);
    let tick_hz = Config::default().tick_hz;
    assert!(r.has_phone);
    let s = score(&scene, &r.ticks, &r.epochs, r.has_phone, tick_hz);
    let reference = centre(&scene);

    for (which, got) in [
        (PersonEstimator::Wlm, &s.wlm),
        (PersonEstimator::Rti, &s.rti),
    ] {
        let pick = pick(which);
        let mut occupied = 0;
        let mut triples = Vec::new();
        for tick in &r.ticks {
            // The per-record score is the frame's, and agrees with the
            // nearest distance (lowest index on a tie).
            let rs = person_score(pick(tick), &tick.truth);
            if tick.truth.is_empty() {
                continue;
            }
            occupied += 1;
            if let Outcome::Estimate(e) = pick(tick) {
                let err = nearest(e.x, e.y, tick);
                let i = tick
                    .truth
                    .iter()
                    .position(|p| dist(e.x, e.y, p.x, p.y) == err)
                    .unwrap();
                assert_eq!(
                    rs,
                    RecordScore::Scored {
                        error_m: err,
                        person: Some(i)
                    }
                );
                triples.push((err, e.radius_m, nearest(reference.0, reference.1, tick)));
            }
        }
        let d = direct(&triples);
        let o = &got.occupied;
        println!(
            "SIM bedroom with_walker 20 s {}: ticks {} occupied {} published {}",
            which.as_str(),
            got.ticks,
            o.ticks,
            o.published
        );
        assert_eq!(got.ticks, r.ticks.len());
        assert_eq!(o.ticks, occupied);
        assert_eq!(o.published, triples.len());
        assert_eq!(o.coverage, Some(triples.len() as f64 / occupied as f64));
        assert_eq!(o.median_m, d.own.median_m);
        assert_eq!(o.p90_m, d.own.p90_m);
        assert_eq!(o.inside_radius, d.own.inside_radius);
        assert_eq!(o.radius_median_m, d.radius_median_m);
        assert_eq!(o.reference, d.reference);
        // Room Sim v1: every tick of an occupied variant is occupied.
        assert_eq!(occupied, r.ticks.len());
        assert_eq!(got.unoccupied.ticks, 0);
        assert_eq!(got.unoccupied.per_h, None);
    }
    // The walker crosses the room: WLM publishes on some ticks.
    assert!(s.wlm.occupied.published > 0);

    let rtt = (|e: &EpochRecord| &e.rtt) as fn(&EpochRecord) -> &Outcome;
    let rssi = (|e: &EpochRecord| &e.rssi) as fn(&EpochRecord) -> &Outcome;
    for (name, got, pick) in [
        ("rtt", s.rtt.unwrap(), rtt),
        ("rssi", s.rssi.unwrap(), rssi),
    ] {
        let mut triples = Vec::new();
        for ep in &r.epochs {
            let truth = ep.truth;
            if let Outcome::Estimate(e) = pick(ep) {
                let err = dist(e.x, e.y, truth[0], truth[1]);
                assert_eq!(phone_score(pick(ep), truth).error_m(), Some(err));
                triples.push((
                    err,
                    e.radius_m,
                    dist(reference.0, reference.1, truth[0], truth[1]),
                ));
            }
        }
        let d = direct(&triples);
        println!(
            "SIM bedroom with_walker 20 s {name}: epochs {} published {}",
            got.epochs, got.published
        );
        assert_eq!(got.epochs, r.epochs.len());
        assert_eq!(got.published, triples.len());
        assert_eq!(
            got.coverage,
            (!r.epochs.is_empty()).then(|| triples.len() as f64 / r.epochs.len() as f64)
        );
        assert_eq!(got.median_m, d.own.median_m);
        assert_eq!(got.p90_m, d.own.p90_m);
        assert_eq!(got.inside_radius, d.own.inside_radius);
        assert_eq!(got.radius_median_m, d.radius_median_m);
        assert_eq!(got.reference, d.reference);
    }

    // The radius scale: `ceil(0.95 n)`-th ratio, or None below 20.
    let scales = radius_scale(&scene, &r.ticks, &r.epochs, r.has_phone);
    let person_ratios = |ticks: &[TickRecord], which: PersonEstimator| -> Vec<f64> {
        let pick = pick(which);
        ticks
            .iter()
            .filter(|t| !t.truth.is_empty())
            .filter_map(|t| {
                let e = pick(t).estimate()?;
                Some(nearest(e.x, e.y, t) / e.radius_m)
            })
            .collect()
    };
    let expect = |mut v: Vec<f64>| {
        if v.len() < 20 {
            return None;
        }
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Some(v[(0.95 * v.len() as f64).ceil() as usize - 1])
    };
    // At 20 s each person estimator scores fewer than 20 estimates (SIM:
    // WLM 17, RTI 0), so on the run itself both person scales compare
    // `None` with `None`; the repeated ticks below give WLM a figure.
    let wlm_ratios = person_ratios(&r.ticks, PersonEstimator::Wlm);
    let rti_ratios = person_ratios(&r.ticks, PersonEstimator::Rti);
    assert_eq!(wlm_ratios.len(), s.wlm.occupied.published);
    assert_eq!(rti_ratios.len(), s.rti.occupied.published);
    assert_eq!(scales.wlm, expect(wlm_ratios));
    assert_eq!(scales.rti, expect(rti_ratios));
    // The run's ticks 20 times over hold 20 x (WLM's published, > 0)
    // scored estimates, at least 20: the person path on real ratios.
    let many: Vec<TickRecord> = (0..20).flat_map(|_| r.ticks.iter().cloned()).collect();
    let scales_many = radius_scale(&scene, &many, &r.epochs, r.has_phone);
    let wlm_many = person_ratios(&many, PersonEstimator::Wlm);
    assert!(wlm_many.len() >= 20);
    assert_eq!(scales_many.wlm, expect(wlm_many));
    assert!(scales_many.wlm.is_some());
    assert_eq!(
        scales_many.rti,
        expect(person_ratios(&many, PersonEstimator::Rti))
    );
    println!(
        "SIM bedroom with_walker 20 s radius scale: wlm {:?} rti {:?} rtt {:?} rssi {:?}; \
         wlm over the ticks 20 times {:?}",
        scales.wlm, scales.rti, scales.rtt, scales.rssi, scales_many.wlm
    );
    let phone_ratios = |pick: fn(&EpochRecord) -> &Outcome| -> Vec<f64> {
        r.epochs
            .iter()
            .filter_map(|ep| {
                let e = pick(ep).estimate()?;
                Some(dist(e.x, e.y, ep.truth[0], ep.truth[1]) / e.radius_m)
            })
            .collect()
    };
    // Both phone methods have at least 20 scored estimates here, so their
    // scales are figures, not `None`.
    assert!(phone_ratios(rtt).len() >= 20 && phone_ratios(rssi).len() >= 20);
    assert_eq!(scales.rtt, expect(phone_ratios(rtt)));
    assert_eq!(scales.rssi, expect(phone_ratios(rssi)));
    assert!(scales.rtt.is_some() && scales.rssi.is_some());

    // The JSON carries the same numbers.
    let v = s.to_json();
    assert_eq!(
        v["wlm"]["occupied"]["published"],
        serde_json::json!(s.wlm.occupied.published)
    );
    assert_eq!(
        v["wlm"]["occupied"]["median_m"],
        serde_json::json!(s.wlm.occupied.median_m)
    );
}

#[test]
fn bedroom_empty_run_false_alarms_equal_a_direct_recomputation() {
    let scene = bedroom();
    let r = run(&scene, Variant::Empty);
    let tick_hz = Config::default().tick_hz;
    let s = score(&scene, &r.ticks, &r.epochs, r.has_phone, tick_hz);
    for (which, got) in [
        (PersonEstimator::Wlm, &s.wlm),
        (PersonEstimator::Rti, &s.rti),
    ] {
        let pick = pick(which);
        // locate_measure's FalseAlarms::of: rising edges of the published
        // flags of the empty run's ticks.
        let flags: Vec<bool> = r
            .ticks
            .iter()
            .map(|t| pick(t).estimate().is_some())
            .collect();
        let mut events = 0;
        let mut prev = false;
        for &f in &flags {
            if f && !prev {
                events += 1;
            }
            prev = f;
        }
        let published = flags.iter().filter(|&&f| f).count();
        let seconds = r.ticks.len() as f64 / tick_hz;
        let expect = FalseAlarms {
            ticks: r.ticks.len(),
            seconds,
            published,
            per_h: Some(published as f64 / seconds * 3600.0),
            events,
            events_per_h: Some(events as f64 / seconds * 3600.0),
        };
        println!(
            "SIM bedroom empty 20 s {}: {} ticks, {} published, {} events",
            which.as_str(),
            expect.ticks,
            published,
            events
        );
        assert!(r.ticks.iter().all(|t| t.truth.is_empty()));
        assert_eq!(got.unoccupied, expect);
        assert_eq!(false_alarms(&r.ticks, which, tick_hz), expect);
        // Nothing is occupied, so nothing is scored.
        assert_eq!(got.occupied.ticks, 0);
        assert_eq!(got.occupied.coverage, None);
        assert_eq!(got.occupied.median_m, None);
    }
}
