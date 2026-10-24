//! `saans eval`: the S3 evaluation. Simulated timelines with ground truth
//! run through the *production* path — per-link FeatureExtractor (the
//! golden-verified DSP), SpaceFusion, SpaceMonitor — and every detector
//! is scored for what it caught and what it invented.
//!
//! Everything here is SYNTHETIC. It says whether the logic is sound and
//! where its blind spots are; it says nothing about real rooms, real
//! bodies or real ESP32 noise (that is S5).

use crate::score::{self, num, pct, BreathingScore, EventScore, Occupancy, Run};
use saans_dsp::features::{FeatureExtractor, FeatureExtractorConfig};
use saans_health::{
    EventKind, HealthEvent, LinkId, Severity, SpaceConfig, SpaceFusion, SpaceMonitor,
};
use saans_sim::scenario::{Activity, Node, Point, Room, Scenario, Segment, TruthKind};
use saans_sim::Simulator;
use std::collections::BTreeMap;
use std::sync::Mutex;

pub struct EvalOptions {
    pub quick: bool,
    pub seeds: u64,
    /// First seed index. Thresholds were tuned on seeds 0-1 (the quick
    /// default); full runs start at 2 so the reported numbers are held out.
    pub seed_offset: u64,
    pub threads: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Kind {
    Empty,
    Reader,
    Night,
    LoneWorker,
    Collapse,
    SitDown,
    Tachypnea,
}

impl Kind {
    fn name(self) -> &'static str {
        match self {
            Kind::Empty => "empty",
            Kind::Reader => "reader",
            Kind::Night => "night",
            Kind::LoneWorker => "lone_worker",
            Kind::Collapse => "collapse",
            Kind::SitDown => "sit_down",
            Kind::Tachypnea => "tachypnea",
        }
    }
}

pub fn rooms() -> Vec<(&'static str, Room)> {
    vec![
        ("clear-LoS", Room::default()),
        (
            "rich-NLoS",
            Room {
                reflection_order: 2,
                los_attenuation_db: 15.0,
                n_scatterers: 12,
                wall_reflection: 0.6,
                ..Room::default()
            },
        ),
    ]
}

/// Node layouts in the 6 x 5 m room: 1, 2 and 3 links at different angles.
/// (name, nodes, links as (tx, rx)).
pub type Layout = (&'static str, Vec<Node>, Vec<(u8, u8)>);

pub fn layouts() -> Vec<Layout> {
    let n = |id, x, y| Node { id, pos: (x, y) };
    vec![
        ("1 link", vec![n(0, 1.5, 2.5), n(1, 4.5, 2.5)], vec![(0, 1)]),
        (
            "2 links",
            vec![n(0, 1.5, 2.5), n(1, 4.5, 2.5), n(2, 3.0, 4.6)],
            vec![(0, 1), (2, 1)],
        ),
        (
            "3 links",
            vec![n(0, 1.0, 1.0), n(1, 5.0, 1.0), n(2, 3.0, 4.5)],
            vec![(0, 1), (1, 2), (2, 0)],
        ),
    ]
}

/// A tiny deterministic generator for scenario parameters (positions,
/// rates, pause times). The simulator has its own for physics noise.
pub struct Lcg(u64);
impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed)
    }
    fn next(&mut self) -> f64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.next()
    }
    pub fn seat(&mut self) -> Point {
        (self.range(0.8, 5.2), self.range(0.8, 4.2))
    }
}

pub const LEAD_IN_S: f64 = 100.0; // filter warmup 50 s + calibration 30 s + margin

fn seg(duration_s: f64, activity: Activity) -> Segment {
    Segment {
        duration_s,
        activity,
    }
}

pub fn timeline(kind: Kind, quick: bool, rng: &mut Lcg) -> Vec<Segment> {
    let mut t = vec![seg(LEAD_IN_S, Activity::Absent)];
    let seat = rng.seat();
    let bpm = rng.range(11.0, 17.0);
    let walk = |to| Activity::WalkTo { to, speed_mps: 0.9 };
    match kind {
        Kind::Empty => t.push(seg(if quick { 1200.0 } else { 3600.0 }, Activity::Absent)),
        Kind::Reader => {
            t.push(seg(15.0, walk(seat)));
            t.push(seg(
                if quick { 900.0 } else { 2400.0 },
                Activity::Seated {
                    at: seat,
                    bpm,
                    bpm_end: Some(bpm + rng.range(-2.0, 2.0)),
                },
            ));
            t.push(seg(15.0, walk((5.6, 2.5))));
            t.push(seg(120.0, Activity::Absent));
        }
        Kind::Night => {
            t.push(seg(15.0, walk(seat)));
            let hours = if quick { 0.75 } else { 2.0 };
            let mut left = hours * 3600.0;
            // Apnea-like pauses ~10/h, 10-40 s, separated by 2-8 min of
            // breathing; an occasional fidget burst.
            while left > 0.0 {
                let breathe = rng.range(120.0, 480.0).min(left);
                if rng.next() < 0.25 {
                    t.push(seg(
                        breathe,
                        Activity::Fidget {
                            at: seat,
                            bpm,
                            every_s: 90.0,
                            burst_s: 2.0,
                        },
                    ));
                } else {
                    t.push(seg(
                        breathe,
                        Activity::Seated {
                            at: seat,
                            bpm,
                            bpm_end: None,
                        },
                    ));
                }
                left -= breathe;
                if left <= 0.0 {
                    break;
                }
                let pause = rng.range(10.0, 40.0).min(left);
                t.push(seg(pause, Activity::BreathingPause { at: seat }));
                left -= pause;
            }
            t.push(seg(
                60.0,
                Activity::Seated {
                    at: seat,
                    bpm,
                    bpm_end: None,
                },
            ));
            t.push(seg(15.0, walk((5.6, 2.5))));
            t.push(seg(120.0, Activity::Absent));
        }
        Kind::LoneWorker => {
            t.push(seg(15.0, walk(seat)));
            t.push(seg(
                600.0,
                Activity::Fidget {
                    at: seat,
                    bpm,
                    every_s: 20.0,
                    burst_s: 3.0,
                },
            ));
            t.push(seg(
                2400.0,
                Activity::Seated {
                    at: seat,
                    bpm,
                    bpm_end: None,
                },
            ));
        }
        Kind::Collapse => {
            // Walk to a spot 1.2 m from where the fall lands, so the 0.8 s
            // fall covers a body length, not several metres.
            let stand = (
                if seat.0 < 3.0 {
                    seat.0 + 1.2
                } else {
                    seat.0 - 1.2
                },
                seat.1,
            );
            t.push(seg(45.0, Activity::Walk { speed_mps: 0.9 }));
            t.push(seg(20.0, walk(stand)));
            t.push(seg(300.0, Activity::Fall { to: seat, bpm }));
        }
        Kind::SitDown => {
            t.push(seg(60.0, Activity::Walk { speed_mps: 0.9 }));
            t.push(seg(600.0, Activity::SitDown { at: seat, bpm }));
        }
        Kind::Tachypnea => {
            t.push(seg(15.0, walk(seat)));
            t.push(seg(
                420.0,
                Activity::Seated {
                    at: seat,
                    bpm: rng.range(26.0, 30.0),
                    bpm_end: None,
                },
            ));
        }
    }
    t
}

pub struct Job {
    pub kind: Kind,
    pub room: &'static str,
    pub layout: &'static str,
    pub seed: u64,
    pub scenario: Scenario,
}

pub fn jobs(opts: &EvalOptions) -> Vec<Job> {
    let kinds = [
        Kind::Empty,
        Kind::Reader,
        Kind::Night,
        Kind::LoneWorker,
        Kind::Collapse,
        Kind::SitDown,
        Kind::Tachypnea,
    ];
    let mut out = Vec::new();
    for (room_name, room) in rooms() {
        for (layout_name, nodes, links) in layouts() {
            for &kind in &kinds {
                for s in opts.seed_offset..opts.seed_offset + opts.seeds {
                    let seed = 1000 + s * 7919 + kind as u64 * 104729;
                    let mut rng = Lcg(seed ^ 0x9E37_79B9_7F4A_7C15);
                    let scenario = Scenario {
                        name: format!("{}/{}/{}/{}", kind.name(), room_name, layout_name, s),
                        room: room.clone(),
                        nodes: nodes.clone(),
                        links: links.clone(),
                        timeline: timeline(kind, opts.quick, &mut rng),
                        sample_rate_hz: 20.0,
                        n_subcarriers: 56,
                        noise_std: 2.0,
                        loss_prob: 0.02,
                        body_points: 3,
                        body_width_m: 0.3,
                        seed,
                    };
                    out.push(Job {
                        kind,
                        room: room_name,
                        layout: layout_name,
                        seed,
                        scenario,
                    });
                }
            }
        }
    }
    out
}

/// One scenario through the production pipeline.
pub fn run(job: &Job) -> Run {
    let sc = &job.scenario;
    let mut sim = Simulator::new(sc);
    let links: Vec<LinkId> = sc.links.iter().map(|&(tx, rx)| LinkId { rx, tx }).collect();
    let config = SpaceConfig {
        id: sc.name.clone(),
        links: links.clone(),
        vitals: true,
        collapse_detection: true,
        ..SpaceConfig::default()
    };
    let mut extractors: BTreeMap<(u8, u8), FeatureExtractor> = sc
        .links
        .iter()
        .map(|&(tx, rx)| {
            (
                (tx, rx),
                FeatureExtractor::new(FeatureExtractorConfig {
                    n_subcarriers: sc.n_subcarriers,
                    sample_rate_hz: sc.sample_rate_hz,
                    node_id: rx,
                    tx_id: tx,
                    vitals_enabled: true,
                    ..Default::default()
                }),
            )
        })
        .collect();
    let mut fusion = SpaceFusion::new(config.clone());
    let mut monitor = SpaceMonitor::new(config);
    let mut samples = Vec::new();
    let mut events: Vec<HealthEvent> = Vec::new();
    let mut next = 0.5;
    while let Some(tick) = sim.step() {
        for lf in tick.frames {
            if let Some(ex) = extractors.get_mut(&(lf.tx, lf.rx)) {
                for ff in ex.push(lf.frame, Some(lf.seq as u64), Some(lf.t_s), None) {
                    fusion.ingest(
                        LinkId {
                            rx: lf.rx,
                            tx: lf.tx,
                        },
                        ff,
                        lf.t_s,
                    );
                }
            }
        }
        if tick.truth.t_s + 1e-9 >= next {
            let s = fusion.sample(tick.truth.t_s);
            events.extend(monitor.update(&s));
            samples.push((tick.truth, s));
            next += 0.5;
        }
    }
    Run {
        label: sc.name.clone(),
        hours: sim.duration_s() / 3600.0,
        samples,
        events,
        truth_events: sim.truth_events(),
    }
}

struct Done {
    kind: Kind,
    room: &'static str,
    layout: &'static str,
    run: Run,
}

pub fn evaluate(opts: &EvalOptions) {
    let jobs = jobs(opts);
    let n = jobs.len();
    eprintln!(
        "saans eval: {n} simulated runs on {} threads ({})",
        opts.threads,
        if opts.quick { "quick" } else { "full" }
    );
    let start = std::time::Instant::now();
    let queue = Mutex::new(jobs.into_iter().enumerate().collect::<Vec<_>>());
    let done: Mutex<Vec<Done>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..opts.threads.max(1) {
            s.spawn(|| loop {
                let Some((i, job)) = queue.lock().unwrap().pop() else {
                    break;
                };
                let run = run(&job);
                let finished = {
                    let mut d = done.lock().unwrap();
                    d.push(Done {
                        kind: job.kind,
                        room: job.room,
                        layout: job.layout,
                        run,
                    });
                    d.len()
                };
                if finished % 10 == 0 || finished == n {
                    eprintln!(
                        "  {finished}/{n} runs ({:.0} s) last: #{i}",
                        start.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    let mut done = done.into_inner().unwrap();
    done.sort_by(|a, b| (a.kind, a.room, a.layout).cmp(&(b.kind, b.room, b.layout)));
    let hours: f64 = done.iter().map(|d| d.run.hours).sum();
    report(&done, opts, hours, start.elapsed().as_secs_f64());
}

fn select<'a>(
    done: &'a [Done],
    kinds: &[Kind],
    layout: Option<&str>,
    room: Option<&str>,
) -> Vec<&'a Run> {
    done.iter()
        .filter(|d| {
            kinds.contains(&d.kind)
                && layout.is_none_or(|l| d.layout == l)
                && room.is_none_or(|r| d.room == r)
        })
        .map(|d| &d.run)
        .collect()
}

fn report(done: &[Done], opts: &EvalOptions, hours: f64, wall_s: f64) {
    let layouts: Vec<&str> = layouts().iter().map(|l| l.0).collect();
    println!("==========================================================================");
    println!(
        "saans eval -- SYNTHETIC. {} runs, {:.1} simulated hours in {:.0} s ({} seeds per cell from seed {}{})",
        done.len(), hours, wall_s, opts.seeds, opts.seed_offset,
        if opts.seed_offset >= 2 { ", HELD OUT from threshold tuning" } else { ", DEVELOPMENT seeds used for tuning" }
    );
    if opts.quick {
        println!("quick mode: shortened timelines");
    }
    println!("Rooms: clear-LoS, rich-NLoS (6 x 5 m). 56 subcarriers, 20 Hz, noise 2.0 vs |LoS| 40, 2% loss,");
    println!("3-point torso. Production path: FeatureExtractor -> SpaceFusion -> SpaceMonitor.");
    println!("==========================================================================");

    println!(
        "\nOCCUPANCY (excluding the first {:.0} s and +/-{:.0} s around enter/leave)",
        score::SETTLE_S,
        score::TRANSITION_GUARD_S
    );
    println!(
        "  {:8} {:>9} {:>17} {:>16} {:>9}",
        "layout", "accuracy", "still false-empty", "empty false-occ", "unknown"
    );
    for l in &layouts {
        let mut o = Occupancy::default();
        for r in select(
            done,
            &[
                Kind::Empty,
                Kind::Reader,
                Kind::Night,
                Kind::LoneWorker,
                Kind::SitDown,
                Kind::Tachypnea,
                Kind::Collapse,
            ],
            Some(l),
            None,
        ) {
            o.add(&score::occupancy(r));
        }
        println!(
            "  {:8} {:>9} {:>17} {:>16} {:>9}",
            l,
            pct(o.accuracy()),
            pct(o.false_empty_still()),
            pct(o.false_occupied()),
            pct(o.unknown())
        );
    }

    println!("\nBREATHING RATE (still breather, scored from 60 s after breathing steadies)");
    println!(
        "  {:8} {:>10} {:>16} {:>12}",
        "layout", "coverage", "within 1 BPM", "median err"
    );
    for l in &layouts {
        let mut b = BreathingScore::default();
        for r in select(
            done,
            &[Kind::Reader, Kind::Night, Kind::LoneWorker, Kind::Tachypnea],
            Some(l),
            None,
        ) {
            b.add(&score::breathing(r, 60.0));
        }
        println!(
            "  {:8} {:>10} {:>16} {:>12}",
            l,
            pct(b.coverage()),
            pct(b.accuracy_when_reported()),
            num(b.median_error(), 2)
        );
    }

    println!("\nBREATHING PAUSES (night timelines; truth pauses 10-40 s)");
    println!(
        "  {:8} {:>7} {:>8} {:>11} {:>11} {:>14} {:>16}",
        "layout", "truths", "caught", "sensitivity", "precision", "false / hour", "duration err s"
    );
    for l in &layouts {
        let mut e = EventScore::default();
        for r in select(done, &[Kind::Night], Some(l), None) {
            e.add(&score::pauses(r));
        }
        println!(
            "  {:8} {:>7} {:>8} {:>11} {:>11} {:>14} {:>16}",
            l,
            e.truths,
            e.caught,
            pct(e.sensitivity()),
            pct(e.precision()),
            num(e.false_per_hour(), 2),
            num(score::median(&e.duration_err_s), 1)
        );
    }
    let mut no_pause = EventScore::default();
    for r in select(
        done,
        &[
            Kind::Reader,
            Kind::LoneWorker,
            Kind::SitDown,
            Kind::Tachypnea,
        ],
        None,
        None,
    ) {
        no_pause.add(&score::pauses(r));
    }
    println!(
        "  pauses reported on timelines with none: {} in {:.1} h ({}/h)",
        no_pause.detections,
        no_pause.hours,
        num(no_pause.detections as f64 / no_pause.hours, 2)
    );

    println!("\nALERTS (per scenario type; 'runs firing' = runs with at least one)");
    println!("  (possible-collapse detection is ON here to measure it; it ships OFF, see the last column)");
    println!(
        "  {:12} {:>6} {:>22} {:>22} {:>22} {:>12} {:>18}",
        "scenario",
        "runs",
        "inactivity",
        "possible collapse",
        "rate high/low",
        "Check / h",
        "Check / h, shipped"
    );
    for kind in [
        Kind::Empty,
        Kind::Reader,
        Kind::Night,
        Kind::LoneWorker,
        Kind::Collapse,
        Kind::SitDown,
        Kind::Tachypnea,
    ] {
        let runs = select(done, &[kind], None, None);
        let firing =
            |f: &dyn Fn(&EventKind) -> bool| runs.iter().filter(|r| score::count(r, f) > 0).count();
        let inact = firing(&|k| matches!(k, EventKind::Inactivity { .. }));
        let coll = firing(&|k| matches!(k, EventKind::PossibleCollapse { .. }));
        let rate = firing(&|k| matches!(k, EventKind::RateHigh { .. } | EventKind::RateLow { .. }));
        let checks: usize = runs
            .iter()
            .map(|r| {
                r.events
                    .iter()
                    .filter(|e| e.severity == Severity::Check)
                    .count()
            })
            .sum();
        let shipped: usize = runs
            .iter()
            .map(|r| {
                r.events
                    .iter()
                    .filter(|e| {
                        e.severity == Severity::Check
                            && !matches!(e.kind, EventKind::PossibleCollapse { .. })
                    })
                    .count()
            })
            .sum();
        let h: f64 = runs.iter().map(|r| r.hours).sum();
        let frac = |k: usize| {
            format!(
                "{k}/{} ({})",
                runs.len(),
                pct(score::ratio(k as u64, runs.len() as u64))
            )
        };
        println!(
            "  {:12} {:>6} {:>22} {:>22} {:>22} {:>12} {:>18}",
            kind.name(),
            runs.len(),
            frac(inact),
            frac(coll),
            frac(rate),
            num(checks as f64 / h, 2),
            num(shipped as f64 / h, 2)
        );
    }
    let lat: Vec<f64> = select(done, &[Kind::LoneWorker], None, None)
        .iter()
        .filter_map(|r| {
            let still_start = r
                .truth_events
                .iter()
                .find(|e| e.kind == TruthKind::Burst)
                .map(|_| 0.0);
            let _ = still_start;
            r.events
                .iter()
                .find(|e| matches!(e.kind, EventKind::Inactivity { .. }))
                .map(|e| e.t_s)
        })
        .collect();
    println!("  lone_worker: inactivity fired at median t = {} s (motionless from t = {:.0} s; alert set to 30 min)", num(score::median(&lat), 0), LEAD_IN_S + 15.0 + 600.0);

    println!("\nREST PERIODS (night timelines: one long lying period with pauses and fidgets; >= 20 min of stillness starts one)");
    let nights = select(done, &[Kind::Night], None, None);
    let rests: Vec<usize> = nights
        .iter()
        .map(|r| score::count(r, |k| matches!(k, EventKind::RestEnded { .. })) as usize)
        .collect();
    println!(
        "  runs with >= 1 rest period: {}/{}; total rest periods reported: {}",
        rests.iter().filter(|&&c| c > 0).count(),
        nights.len(),
        rests.iter().sum::<usize>()
    );

    println!("\nBY ROOM (all layouts)");
    for (room, _) in rooms() {
        let mut o = Occupancy::default();
        let mut b = BreathingScore::default();
        let mut e = EventScore::default();
        for r in select(
            done,
            &[Kind::Reader, Kind::Night, Kind::LoneWorker, Kind::Tachypnea],
            None,
            Some(room),
        ) {
            o.add(&score::occupancy(r));
            b.add(&score::breathing(r, 60.0));
        }
        for r in select(done, &[Kind::Night], None, Some(room)) {
            e.add(&score::pauses(r));
        }
        println!(
            "  {:10} still false-empty {}  breathing coverage {} (within 1 BPM {})  pause sensitivity {} precision {}",
            room, pct(o.false_empty_still()), pct(b.coverage()), pct(b.accuracy_when_reported()), pct(e.sensitivity()), pct(e.precision())
        );
    }
    println!("\nSYNTHETIC. Simulated physics, bodies and noise. These numbers test the logic and");
    println!("compare configurations on identical data; real-room accuracy is measured in S5.");
}
