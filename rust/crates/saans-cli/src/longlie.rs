//! `saans longlie`: the long lie (issue #38, `docs/ELDER_CARE.md` §3.3)
//! measured. SYNTHETIC.
//!
//! The detector: a burst (motion index at least `BURST_INDEX`) that stops
//! within `COLLAPSE_SETTLE_S`, then N minutes of no gross motion while the
//! space is occupied, measured or held by the presence hold. Its misses are
//! measured on falls that end lying still; its false alarms on the ordinary
//! things that look the same to amplitude-only WiFi: dozing after sitting
//! down heavily, reading, a fidgeting night, and walking out (which is what
//! the presence hold mistakes for a fall out of sight). ADR-005: the two
//! rates are reported together.
//!
//! `saans eval`'s timelines end minutes after the event, too soon for a
//! 15- or 30-minute alert, so these timelines carry on for 40 minutes.
//! Every room is `saans eval`'s, as a 2.7 m box (the floor puts a fallen
//! chest at 0.2 m, which `saans lying` found matters). Vitals are off, as
//! they are by default; the long lie does not read them.
//!
//! The detector settings do not change the fused samples, only what the
//! monitor does with them, so each timeline is simulated once and its
//! samples replayed through every setting.

use crate::eval::{self, Job, Kind, Lcg, LEAD_IN_S};
use crate::score::Run;
use saans_health::{EventKind, LinkId, SpaceConfig, SpaceMonitor};
use saans_sim::scenario::{Activity, Point, Scenario, Segment, TruthKind, Vertical};
use std::collections::BTreeMap;
use std::sync::Mutex;

const ROOM_HEIGHT_M: f64 = 2.7;
/// How long each timeline stays still after the event: the longer N plus
/// room for it to fire.
const STILL_S: f64 = 2400.0;
/// The spec's miss rule: a lie not alerted within N + 2 minutes is missed.
const GRACE_S: f64 = 120.0;
/// (N minutes, presence hold minutes) per column. The hold has to outlast
/// N, so it is N + 5; 0 is off, as shipped.
const SETTINGS: [(f64, f64); 4] = [(15.0, 0.0), (15.0, 20.0), (30.0, 0.0), (30.0, 35.0)];
const DOOR: Point = (5.6, 2.5);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Case {
    Fall,
    Doze,
    Reader,
    Night,
    Departure,
}

impl Case {
    const ALL: [Case; 5] = [
        Case::Fall,
        Case::Doze,
        Case::Reader,
        Case::Night,
        Case::Departure,
    ];

    fn label(self) -> &'static str {
        match self {
            Case::Fall => "fall, then lying still",
            Case::Doze => "sits down heavily, dozes",
            Case::Reader => "walks in, sits, reads",
            Case::Night => "fidgeting night, 2 h",
            Case::Departure => "sits 5 min, walks out",
        }
    }

    fn short(self) -> &'static str {
        match self {
            Case::Fall => "fall",
            Case::Doze => "doze",
            Case::Reader => "reader",
            Case::Night => "night",
            Case::Departure => "departure",
        }
    }

    /// `saans eval`'s nearest kind, for the job's label only.
    fn kind(self) -> Kind {
        match self {
            Case::Fall => Kind::Collapse,
            Case::Doze => Kind::SitDown,
            Case::Reader | Case::Departure => Kind::Reader,
            Case::Night => Kind::Night,
        }
    }
}

fn seg(duration_s: f64, activity: Activity) -> Segment {
    Segment {
        duration_s,
        activity,
    }
}

fn timeline(case: Case, rng: &mut Lcg) -> Vec<Segment> {
    // The night is `saans eval`'s, drawn from the same generator; the rest
    // draw a seat and a rate first.
    let (seat, bpm) = match case {
        Case::Night => return eval::timeline(Kind::Night, false, rng),
        _ => (rng.seat(), rng.range(11.0, 17.0)),
    };
    let walk_to = |to| Activity::WalkTo { to, speed_mps: 0.9 };
    let seated = Activity::Seated {
        at: seat,
        bpm,
        bpm_end: None,
    };
    let mut t = vec![seg(LEAD_IN_S, Activity::Absent)];
    match case {
        Case::Fall => {
            // As `saans eval`: stand 1.2 m from where the fall lands, so the
            // 0.8 s fall covers a body length.
            let stand = (
                if seat.0 < 3.0 {
                    seat.0 + 1.2
                } else {
                    seat.0 - 1.2
                },
                seat.1,
            );
            t.push(seg(45.0, Activity::Walk { speed_mps: 0.9 }));
            t.push(seg(20.0, walk_to(stand)));
            t.push(seg(STILL_S, Activity::Fall { to: seat, bpm }));
        }
        Case::Doze => {
            t.push(seg(60.0, Activity::Walk { speed_mps: 0.9 }));
            t.push(seg(STILL_S, Activity::SitDown { at: seat, bpm }));
        }
        Case::Reader => {
            t.push(seg(15.0, walk_to(seat)));
            t.push(seg(STILL_S, seated));
            t.push(seg(15.0, walk_to(DOOR)));
            t.push(seg(120.0, Activity::Absent));
        }
        Case::Departure => {
            t.push(seg(15.0, walk_to(seat)));
            t.push(seg(300.0, seated));
            t.push(seg(15.0, walk_to(DOOR)));
            t.push(seg(STILL_S, Activity::Absent));
        }
        // Returned above, before the seat was drawn.
        Case::Night => {}
    }
    t
}

/// One simulated timeline, scored under every setting.
struct Done {
    /// The scenario's name: the report sorts by it, so its output does not
    /// depend on which thread finished first.
    name: String,
    case: Case,
    layout: &'static str,
    hours: f64,
    /// When the fall began, for `Case::Fall`.
    fall_t: Option<f64>,
    /// Per setting, the times the long lie alerted (not movement updates).
    alerts: [Vec<f64>; SETTINGS.len()],
}

fn replay(run: &Run, links: &[LinkId], long_lie_min: f64, fall_hold_min: f64) -> Vec<f64> {
    let mut m = SpaceMonitor::new(SpaceConfig {
        id: run.label.clone(),
        links: links.to_vec(),
        vitals: false,
        long_lie_min,
        fall_hold_min,
        ..SpaceConfig::default()
    });
    run.samples
        .iter()
        .flat_map(|(_, s)| m.update(s))
        .filter_map(|e| match e.kind {
            EventKind::LongLie {
                movement_seen: false,
                ..
            } => Some(e.t_s),
            _ => None,
        })
        .collect()
}

fn jobs(seeds: u64) -> Vec<(Case, Job)> {
    let mut out = Vec::new();
    for (ri, (room_name, mut room)) in eval::rooms().into_iter().enumerate() {
        room.vertical = Some(Vertical {
            height_m: ROOM_HEIGHT_M,
            ..Vertical::default()
        });
        for (li, (layout_name, nodes, links)) in eval::layouts().into_iter().enumerate() {
            for case in Case::ALL {
                for s in 0..seeds {
                    // Disjoint from `saans eval`'s seeds (1000 + ...), whose
                    // first two tuned the shared thresholds.
                    let seed = 90_000
                        + s * 7919
                        + case as u64 * 104_729
                        + li as u64 * 1_299_709
                        + ri as u64 * 15_485_863;
                    let mut rng = Lcg::new(seed ^ 0x2545_F491_4F6C_DD1D);
                    let scenario = Scenario {
                        name: format!("{}/{room_name}/{layout_name}/{s}", case.short()),
                        room: room.clone(),
                        nodes: nodes.clone(),
                        links: links.clone(),
                        timeline: timeline(case, &mut rng),
                        sample_rate_hz: 20.0,
                        n_subcarriers: 56,
                        noise_std: 2.0,
                        loss_prob: 0.02,
                        body_points: 3,
                        body_width_m: 0.3,
                        seed,
                    };
                    out.push((
                        case,
                        Job {
                            kind: case.kind(),
                            room: room_name,
                            layout: layout_name,
                            seed,
                            scenario,
                            fall_hold_min: 0.0,
                        },
                    ));
                }
            }
        }
    }
    out
}

fn measure(case: Case, job: &Job) -> Done {
    let run = eval::run(job);
    let links: Vec<LinkId> = job
        .scenario
        .links
        .iter()
        .map(|&(tx, rx)| LinkId { rx, tx })
        .collect();
    let fall_t = run
        .truth_events
        .iter()
        .find(|e| e.kind == TruthKind::Fall)
        .map(|e| e.t_start_s);
    Done {
        name: job.scenario.name.clone(),
        case,
        layout: job.layout,
        hours: run.hours,
        fall_t,
        alerts: SETTINGS.map(|(n, hold)| replay(&run, &links, n, hold)),
    }
}

pub fn run(seeds: u64, threads: usize) {
    let jobs = jobs(seeds);
    for (_, job) in &jobs {
        if let Err(e) = saans_sim::Simulator::try_new(&job.scenario) {
            eprintln!("saans longlie: {}: {e}", job.scenario.name);
            return;
        }
    }
    let n = jobs.len();
    eprintln!("saans longlie: {n} simulated runs on {threads} threads");
    let start = std::time::Instant::now();
    let queue = Mutex::new(jobs);
    let done: Mutex<Vec<Done>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                let Some((case, job)) = queue.lock().unwrap().pop() else {
                    break;
                };
                let d = measure(case, &job);
                let finished = {
                    let mut all = done.lock().unwrap();
                    all.push(d);
                    all.len()
                };
                if finished % 10 == 0 || finished == n {
                    eprintln!(
                        "  {finished}/{n} runs ({:.0} s)",
                        start.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    let mut done = done.into_inner().unwrap();
    done.sort_by(|a, b| a.name.cmp(&b.name));
    report(&done, seeds);
}

fn header() -> String {
    let mut h = format!("{:<28}{:>6}", "", "runs");
    for (n, hold) in SETTINGS {
        let col = if hold > 0.0 {
            format!("N={n:.0}, hold {hold:.0}")
        } else {
            format!("N={n:.0}, hold off")
        };
        h.push_str(&format!("{col:>20}"));
    }
    h
}

fn median(mut v: Vec<f64>) -> Option<f64> {
    if v.is_empty() {
        return None;
    }
    v.sort_by(f64::total_cmp);
    let m = v.len() / 2;
    Some(if v.len().is_multiple_of(2) {
        (v[m - 1] + v[m]) / 2.0
    } else {
        v[m]
    })
}

/// "caught/runs (median minutes from the fall to the alert)".
fn fall_cell(runs: &[&Done], i: usize) -> String {
    let n_s = SETTINGS[i].0 * 60.0;
    let mut minutes = Vec::new();
    for d in runs {
        let Some(f) = d.fall_t else { continue };
        if let Some(a) = d.alerts[i].iter().find(|&&a| a >= f) {
            if *a <= f + n_s + GRACE_S {
                minutes.push((a - f) / 60.0);
            }
        }
    }
    let caught = minutes.len();
    match median(minutes) {
        Some(m) => format!("{caught}/{} ({m:.1})", runs.len()),
        None => format!("{caught}/{}", runs.len()),
    }
}

/// "runs alerting/runs (alerts)".
fn false_cell(runs: &[&Done], i: usize) -> String {
    let alerting = runs.iter().filter(|d| !d.alerts[i].is_empty()).count();
    let alerts: usize = runs.iter().map(|d| d.alerts[i].len()).sum();
    format!("{alerting}/{} ({alerts})", runs.len())
}

fn report(done: &[Done], seeds: u64) {
    let layouts: Vec<&str> = eval::layouts().into_iter().map(|l| l.0).collect();
    let mut by: BTreeMap<(Case, &str), Vec<&Done>> = BTreeMap::new();
    for d in done {
        by.entry((d.case, d.layout)).or_default().push(d);
    }
    let rooms = eval::rooms().len();
    println!(
        "Long lie, SYNTHETIC: {seeds} seeds x {rooms} rooms x {} layouts per case, every room a {ROOM_HEIGHT_M} m box. \
         The long lie reads no vital signs (the monitor runs with vitals off; occupancy and motion are not \
         vitals-gated).",
        layouts.len()
    );
    println!(
        "Alert: a burst that stops within 5 s, then N min of no gross motion while present (measured, or held \
         by the presence hold for N + 5 min). Each timeline is simulated once and replayed through all four \
         settings."
    );

    println!("\nFALLS: caught within N + 2 min of the fall / runs (median minutes from the fall to the alert)");
    println!("{}", header());
    let falls_all: Vec<&Done> = done.iter().filter(|d| d.case == Case::Fall).collect();
    for layout in &layouts {
        let runs = by.get(&(Case::Fall, layout)).cloned().unwrap_or_default();
        let mut line = format!(
            "{:<28}{:>6}",
            format!("{} ({layout})", Case::Fall.short()),
            runs.len()
        );
        for i in 0..SETTINGS.len() {
            line.push_str(&format!("{:>20}", fall_cell(&runs, i)));
        }
        println!("{line}");
    }
    let mut line = format!("{:<28}{:>6}", "fall (all layouts)", falls_all.len());
    for i in 0..SETTINGS.len() {
        line.push_str(&format!("{:>20}", fall_cell(&falls_all, i)));
    }
    println!("{line}");
    // Per setting, over the runs whose fall the simulator recorded.
    let with_truth: Vec<&&Done> = falls_all.iter().filter(|d| d.fall_t.is_some()).collect();
    let mut line = format!("{:<28}{:>6}", "alerts before the fall", with_truth.len());
    for i in 0..SETTINGS.len() {
        let early: usize = with_truth
            .iter()
            .map(|d| {
                let f = d.fall_t.unwrap_or(f64::INFINITY);
                d.alerts[i].iter().filter(|&&a| a < f).count()
            })
            .sum();
        line.push_str(&format!("{early:>20}"));
    }
    println!("{line}");
    let no_truth = falls_all.len() - with_truth.len();
    if no_truth > 0 {
        println!("{no_truth} fall runs have no recorded fall and are not scored");
    }

    println!("\nFALSE ALARMS: runs alerting / runs (alerts), every layout");
    println!("{}", header());
    let mut hours = 0.0;
    for case in Case::ALL.into_iter().filter(|&c| c != Case::Fall) {
        let runs: Vec<&Done> = done.iter().filter(|d| d.case == case).collect();
        hours += runs.iter().map(|d| d.hours).sum::<f64>();
        let mut line = format!("{:<28}{:>6}", case.label(), runs.len());
        for i in 0..SETTINGS.len() {
            line.push_str(&format!("{:>20}", false_cell(&runs, i)));
        }
        println!("{line}");
    }
    println!("{hours:.1} simulated hours without a fall");

    println!("\nWALKING OUT, by layout: runs alerting / runs (alerts)");
    println!("{}", header());
    for layout in &layouts {
        let runs = by
            .get(&(Case::Departure, layout))
            .cloned()
            .unwrap_or_default();
        let mut line = format!("{:<28}{:>6}", format!("departure ({layout})"), runs.len());
        for i in 0..SETTINGS.len() {
            line.push_str(&format!("{:>20}", false_cell(&runs, i)));
        }
        println!("{line}");
    }

    println!(
        "\nNot separable here: in this simulator lying down is only reached through `Activity::Fall`, so a nap \
         that begins by lying down quickly is the fall row's timeline and would alert as often; 'sits down \
         heavily, dozes' is the nap in a chair. Not modelled: kneeling or bending to the floor (prayer), lying \
         on the side, real floors (NOTES.md §14). The home-level condition, nobody moving in another room \
         (docs/ELDER_CARE.md §3.3), needs the home view and is not applied."
    );
}
