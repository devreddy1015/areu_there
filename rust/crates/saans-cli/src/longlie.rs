//! `saans longlie`: the long lie (issue #38, docs/ELDER_CARE.md §3.3) on
//! simulated timelines. SYNTHETIC.
//!
//! Each timeline is simulated once through the production path
//! (FeatureExtractor -> SpaceFusion), and its fused samples are replayed
//! through a `SpaceMonitor` for every configuration: N (`long_lie_min`)
//! and the presence hold off or on. So every configuration sees identical
//! data, and a difference between two columns is the configuration.
//!
//! Positives are lies: a fall, then still on the floor; a slow slide to
//! the floor (10-30 s, straight down where the person stood); a fall after which the person moves a limb every few
//! minutes. A positive counts as alerted when the alert comes within N + 2
//! minutes of the lie starting (§4.6). Negatives are everything else: a
//! heavy sit-down and then still, a sit-down and up again, a sit-down and
//! fidgeting, lying down into bed quickly, an empty room, the person
//! walking out, and `saans eval`'s walk-in reader and fidgeting night (the
//! scenarios PR #59 measured). Every alert in a negative is false, and so
//! is an alert on a lie run before the lie started.
//!
//! The rooms and layouts are `saans eval`'s, as a 2.7 m box: on the floor
//! the chest is at 0.2 m and breathes vertically. A bed's height is not
//! modelled; "into bed" is a heavy sit-down at the bed that leaves the
//! chest upright. There is no second room: "seen moving in another room"
//! is the harness telling the monitor so, 10 s after the person walks out.

use crate::eval::{self, Job, Kind, Lcg, LEAD_IN_S};
use crate::score::{median, num, pct, ratio, SAMPLE_S};
use saans_health::detectors::BURST_INDEX;
use saans_health::{EventKind, SpaceConfig, SpaceMonitor, SpaceSample};
use saans_sim::scenario::{Activity, Point, Scenario, Segment, Truth, TruthKind, Vertical};
use std::sync::Mutex;

const ROOM_HEIGHT_M: f64 = 2.7;
/// The values of N swept, in minutes.
pub const NS_MIN: [f64; 6] = [3.0, 5.0, 10.0, 15.0, 20.0, 30.0];
/// The presence hold, when on: `saans eval`'s, longer than every N.
const HOLD_MIN: f64 = eval::FALL_HOLD_MIN;
/// Time on the floor in a lie: the largest N, plus the 2 minutes it has
/// to alert in, plus margin.
const LIE_S: f64 = 34.0 * 60.0;
/// A positive is alerted in time when the alert comes this long after N.
const IN_TIME_S: f64 = 120.0;
const SIT_STILL_S: f64 = 45.0 * 60.0;
const BED_S: f64 = 90.0 * 60.0;
const EMPTY_S: f64 = 60.0 * 60.0;
const AWAY_S: f64 = 40.0 * 60.0;
/// After walking out, the person is seen moving in another room from
/// this long after for `ELSEWHERE_FOR_S`.
const ELSEWHERE_AFTER_S: f64 = 10.0;
const ELSEWHERE_FOR_S: f64 = 20.0;
/// A sudden movement counts as seen if a sample reaches `BURST_INDEX`
/// within this window around the drop.
const BURST_WINDOW_S: (f64, f64) = (1.0, 3.0);
const EXIT: Point = (5.6, 2.5);

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Case {
    Fall,
    Slide,
    FallMoving,
    SitStill,
    SitActive,
    SitFidget,
    IntoBed,
    Empty,
    Departure,
    Reader,
    Night,
}

const CASES: [Case; 11] = [
    Case::Fall,
    Case::Slide,
    Case::FallMoving,
    Case::SitStill,
    Case::SitActive,
    Case::SitFidget,
    Case::IntoBed,
    Case::Empty,
    Case::Departure,
    Case::Reader,
    Case::Night,
];

impl Case {
    fn name(self) -> &'static str {
        match self {
            Case::Fall => "fall, then still (+)",
            Case::Slide => "slide to the floor (+)",
            Case::FallMoving => "fall, a limb moves (+)",
            Case::SitStill => "sit down hard, still 45m",
            Case::SitActive => "sit down, up within 5m",
            Case::SitFidget => "sit down, fidgeting 45m",
            Case::IntoBed => "into bed quickly, 90m",
            Case::Empty => "empty room, 60m",
            Case::Departure => "walks out, empty 40m",
            Case::Reader => "walks in, reads 40m",
            Case::Night => "fidgeting night, 2h",
        }
    }

    fn positive(self) -> bool {
        matches!(self, Case::Fall | Case::Slide | Case::FallMoving)
    }
}

fn seg(duration_s: f64, activity: Activity) -> Segment {
    Segment {
        duration_s,
        activity,
    }
}

fn timeline(case: Case, rng: &mut Lcg) -> Vec<Segment> {
    // `saans eval`'s own timelines, the scenarios PR #59 measured: walk
    // in and sit still reading (no heavy sit-down), and a 2-hour night
    // with breathing pauses and an occasional fidget.
    match case {
        Case::Reader => return eval::timeline(Kind::Reader, false, rng),
        Case::Night => return eval::timeline(Kind::Night, false, rng),
        _ => {}
    }
    let mut t = vec![seg(LEAD_IN_S, Activity::Absent)];
    let seat = rng.seat();
    let bpm = rng.range(11.0, 17.0);
    let walk = |to| Activity::WalkTo { to, speed_mps: 0.9 };
    let wander = || Activity::Walk { speed_mps: 0.9 };
    // Where a fall or a slide starts: a body length from where it ends.
    let stand = (
        if seat.0 < 3.0 {
            seat.0 + 1.2
        } else {
            seat.0 - 1.2
        },
        seat.1,
    );
    let walk_out = |t: &mut Vec<Segment>| {
        t.push(seg(15.0, walk(EXIT)));
        t.push(seg(120.0, Activity::Absent));
    };
    match case {
        Case::Fall | Case::Slide | Case::FallMoving => {
            t.push(seg(45.0, wander()));
            // A slide starts where it ends: down, not across.
            let from = if case == Case::Slide { seat } else { stand };
            t.push(seg(20.0, walk(from)));
            match case {
                Case::Fall => t.push(seg(0.8 + LIE_S, Activity::Fall { to: seat, bpm })),
                Case::Slide => {
                    let duration_s = rng.range(10.0, 30.0);
                    t.push(seg(
                        duration_s + LIE_S,
                        Activity::Slide {
                            to: seat,
                            bpm,
                            duration_s,
                        },
                    ));
                }
                _ => {
                    t.push(seg(0.8 + 60.0, Activity::Fall { to: seat, bpm }));
                    t.push(seg(
                        LIE_S - 60.0,
                        Activity::Fidget {
                            at: seat,
                            bpm,
                            every_s: rng.range(120.0, 360.0),
                            burst_s: 2.0,
                        },
                    ));
                }
            }
        }
        Case::SitStill | Case::SitActive | Case::SitFidget | Case::IntoBed => {
            t.push(seg(60.0, wander()));
            let sit = |s| seg(0.6 + s, Activity::SitDown { at: seat, bpm });
            match case {
                Case::SitStill => t.push(sit(SIT_STILL_S)),
                Case::SitActive => {
                    t.push(sit(rng.range(60.0, 300.0)));
                    t.push(seg(60.0, wander()));
                }
                Case::SitFidget => {
                    t.push(sit(30.0));
                    t.push(seg(
                        SIT_STILL_S,
                        Activity::Fidget {
                            at: seat,
                            bpm,
                            every_s: rng.range(60.0, 300.0),
                            burst_s: 2.0,
                        },
                    ));
                }
                _ => {
                    // Turning over in bed every 10-30 minutes on average.
                    t.push(sit(30.0));
                    t.push(seg(
                        BED_S,
                        Activity::Fidget {
                            at: seat,
                            bpm,
                            every_s: rng.range(600.0, 1800.0),
                            burst_s: 3.0,
                        },
                    ));
                }
            }
            walk_out(&mut t);
        }
        Case::Empty => t.push(seg(EMPTY_S, Activity::Absent)),
        Case::Reader | Case::Night => unreachable!("eval's timelines, above"),
        Case::Departure => {
            t.push(seg(15.0, walk(seat)));
            t.push(seg(
                300.0,
                Activity::Seated {
                    at: seat,
                    bpm,
                    bpm_end: None,
                },
            ));
            t.push(seg(15.0, walk(EXIT)));
            t.push(seg(AWAY_S, Activity::Absent));
        }
    }
    t
}

/// One configuration replayed over one run.
#[derive(Default, Clone)]
struct Outcome {
    /// (alert time, the person truly there).
    alerts: Vec<(f64, bool)>,
}

struct Done {
    case: Case,
    layout: &'static str,
    /// When the lie started (positives).
    lie_start: Option<f64>,
    /// A sample reached `BURST_INDEX` around the fall, slide or sit-down.
    burst_seen: Option<bool>,
    occupied_h: f64,
    empty_h: f64,
    /// Per (hold, N), hold-major; and the same with the person seen
    /// moving elsewhere after walking out (departures only).
    plain: Vec<Outcome>,
    elsewhere: Vec<Outcome>,
}

fn configs() -> Vec<(f64, f64)> {
    [0.0, HOLD_MIN]
        .iter()
        .flat_map(|&h| NS_MIN.iter().map(move |&n| (h, n)))
        .collect()
}

fn replay(
    samples: &[(Truth, SpaceSample)],
    hold: f64,
    n: f64,
    elsewhere: Option<(f64, f64)>,
) -> Outcome {
    let mut m = SpaceMonitor::new(SpaceConfig {
        id: "room".into(),
        long_lie_min: n,
        fall_hold_min: hold,
        ..SpaceConfig::default()
    });
    let mut out = Outcome::default();
    for (truth, s) in samples {
        if elsewhere.is_some_and(|(a, b)| (a..b).contains(&s.t_s)) {
            m.note_motion_elsewhere();
        }
        for e in m.update(s) {
            if matches!(e.kind, EventKind::LongLie { .. }) {
                out.alerts.push((e.t_s, truth.present));
            }
        }
    }
    out
}

fn measure(
    case: Case,
    room_name: &'static str,
    room: &saans_sim::scenario::Room,
    layout: &eval::Layout,
    i: u64,
) -> Done {
    let (layout_name, nodes, links) = layout;
    let seed = 90_000 + i * 7919 + case as u64 * 104_729;
    let mut rng = Lcg::new(seed ^ 0x2545_F491_4F6C_DD1D);
    let scenario = Scenario {
        name: format!("longlie/{case:?}/{room_name}/{layout_name}/{i}"),
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
    let run = eval::run(&Job {
        kind: Kind::Collapse,
        room: room_name,
        layout: layout_name,
        seed,
        scenario,
        fall_hold_min: 0.0,
    });
    let drop = run.truth_events.iter().find(|e| {
        matches!(
            e.kind,
            TruthKind::Fall | TruthKind::Slide | TruthKind::SitDown
        )
    });
    let burst_seen = drop.map(|e| {
        run.samples.iter().any(|(_, s)| {
            s.t_s >= e.t_start_s - BURST_WINDOW_S.0
                && s.t_s <= e.t_end_s + BURST_WINDOW_S.1
                && s.occupied.is_some()
                && s.motion_index >= BURST_INDEX
        })
    });
    let lie_start = drop.filter(|_| case.positive()).map(|e| e.t_end_s);
    let present_s = run.samples.iter().filter(|(t, _)| t.present).count() as f64 * SAMPLE_S;
    let scored = run
        .samples
        .iter()
        .filter(|(t, _)| t.t_s >= LEAD_IN_S)
        .count() as f64
        * SAMPLE_S;
    let leave = run
        .samples
        .windows(2)
        .find(|w| w[0].0.present && !w[1].0.present)
        .map(|w| w[1].0.t_s);
    let plain = configs()
        .iter()
        .map(|&(h, n)| replay(&run.samples, h, n, None))
        .collect();
    let elsewhere = match (case, leave) {
        (Case::Departure, Some(l)) => {
            let w = (
                l + ELSEWHERE_AFTER_S,
                l + ELSEWHERE_AFTER_S + ELSEWHERE_FOR_S,
            );
            configs()
                .iter()
                .map(|&(h, n)| replay(&run.samples, h, n, Some(w)))
                .collect()
        }
        _ => Vec::new(),
    };
    Done {
        case,
        layout: layout_name,
        lie_start,
        burst_seen,
        occupied_h: present_s / 3600.0,
        empty_h: (scored - present_s).max(0.0) / 3600.0,
        plain,
        elsewhere,
    }
}

pub fn run(seeds: u64, threads: usize) {
    let start = std::time::Instant::now();
    let rooms: Vec<_> = eval::rooms()
        .into_iter()
        .map(|(name, mut room)| {
            room.vertical = Some(Vertical {
                height_m: ROOM_HEIGHT_M,
                ..Vertical::default()
            });
            (name, room)
        })
        .collect();
    let layouts = eval::layouts();
    let mut work = Vec::new();
    for &case in &CASES {
        for ri in 0..rooms.len() {
            for li in 0..layouts.len() {
                for i in 0..seeds {
                    work.push((case, ri, li, i));
                }
            }
        }
    }
    let n = work.len();
    eprintln!("saans longlie: {n} runs on {threads} threads");
    let queue = Mutex::new(work);
    let done: Mutex<Vec<Done>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                let Some((case, ri, li, i)) = queue.lock().unwrap().pop() else {
                    break;
                };
                let (room_name, room) = &rooms[ri];
                let d = measure(case, room_name, room, &layouts[li], i);
                let mut all = done.lock().unwrap();
                all.push(d);
                if all.len().is_multiple_of(50) || all.len() == n {
                    eprintln!(
                        "  {}/{n} runs ({:.0} s)",
                        all.len(),
                        start.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    let layout_names: Vec<&str> = layouts.iter().map(|l| l.0).collect();
    report(
        &done.into_inner().unwrap(),
        seeds,
        rooms.len(),
        &layout_names,
    );
}

/// Where a positive run's alerts fall around its lie: the first within
/// N + 2 min of it starting (`in_time`, seconds after it started), alerts
/// before it started (false: nobody was lying yet), and alerts later than
/// the window (too late to count as caught). The before-the-lie count is
/// ported from PR #59.
#[derive(Debug, Clone, Copy, PartialEq)]
struct Timing {
    in_time: Option<f64>,
    before: usize,
    late: usize,
}

fn timing(lie_start: f64, alerts: &[(f64, bool)], n_min: f64) -> Timing {
    let window = n_min * 60.0 + IN_TIME_S;
    let dts = || alerts.iter().map(|&(t, _)| t - lie_start);
    Timing {
        in_time: dts().find(|dt| (0.0..=window).contains(dt)),
        before: dts().filter(|&dt| dt < 0.0).count(),
        late: dts().filter(|&dt| dt > window).count(),
    }
}

/// Alerted within N + 2 min of the lie starting.
fn in_time(d: &Done, o: &Outcome, n_min: f64) -> Option<f64> {
    timing(d.lie_start?, &o.alerts, n_min).in_time
}

fn report(all: &[Done], seeds: u64, n_rooms: usize, layouts: &[&str]) {
    let per_case = seeds as usize * n_rooms * layouts.len();
    let of = |c: Case| all.iter().filter(move |d| d.case == c);
    println!("saans longlie -- SYNTHETIC. The long lie (issue #38) on simulated timelines.");
    println!(
        "Rooms and layouts of `saans eval` as a {ROOM_HEIGHT_M} m box; {per_case} runs per scenario ({n_rooms} rooms x {} layouts x {seeds} seeds).",
        layouts.len()
    );
    println!("Production path: FeatureExtractor -> SpaceFusion, replayed through SpaceMonitor for every N.");
    println!("Vitals off, possible collapse off, as shipped. (+) = a lie, alerted if within N + 2 min of it starting;");
    println!("every alert in any other scenario is false. The presence hold, when on, is {HOLD_MIN:.0} min.");
    println!("'false / occ. h' and 'false / empty h' pool the false alerts of every scenario that is not a lie over");
    println!("its occupied or empty hours: they depend on this mix of scenarios and are not a rate for a home.");

    println!("'before lie' and 'late' count alerts on the lie runs before the lie started (false: nobody was lying)");
    println!("and after the N + 2 min window (too late to count as caught); none is dropped.");
    println!("\nSUDDEN MOVEMENT: runs where the motion index reached {BURST_INDEX:.0} at the fall, slide or sit-down");
    for c in [
        Case::Fall,
        Case::Slide,
        Case::FallMoving,
        Case::SitStill,
        Case::SitActive,
        Case::SitFidget,
        Case::IntoBed,
    ] {
        let k = of(c).filter(|d| d.burst_seen == Some(true)).count();
        println!("  {:26} {k:>4}/{per_case}", c.name());
    }

    for (hold_i, hold) in [0.0, HOLD_MIN].iter().enumerate() {
        let word = if *hold > 0.0 {
            format!("ON ({hold:.0} min)")
        } else {
            "OFF (as shipped)".into()
        };
        println!("\nRUNS ALERTED, presence hold {word}");
        print!("  {:26}", "scenario");
        for nm in NS_MIN {
            print!(" {:>8}", format!("N={nm:.0}"));
        }
        println!();
        for c in CASES {
            print!("  {:26}", c.name());
            for (ni, nm) in NS_MIN.iter().enumerate() {
                let idx = hold_i * NS_MIN.len() + ni;
                let k = of(c)
                    .filter(|d| {
                        let o = &d.plain[idx];
                        if c.positive() {
                            in_time(d, o, *nm).is_some()
                        } else {
                            !o.alerts.is_empty()
                        }
                    })
                    .count();
                print!(" {:>8}", format!("{k}/{per_case}"));
            }
            println!();
        }
        println!("\nMISS RATE AND FALSE ALARMS TOGETHER, presence hold {word}");
        println!(
            "  {:>4} {:>13} {:>13} {:>15} {:>15} {:>16} {:>16} {:>11} {:>6}",
            "N",
            "falls missed",
            "slides missed",
            "median past N",
            "false alerts",
            "false / occ. h",
            "false / empty h",
            "before lie",
            "late"
        );
        for (ni, nm) in NS_MIN.iter().enumerate() {
            let idx = hold_i * NS_MIN.len() + ni;
            let missed = |c: Case| {
                of(c)
                    .filter(|d| in_time(d, &d.plain[idx], *nm).is_none())
                    .count()
            };
            let lat: Vec<f64> = of(Case::Fall)
                .filter_map(|d| in_time(d, &d.plain[idx], *nm))
                .collect();
            let neg: Vec<&Done> = all.iter().filter(|d| !d.case.positive()).collect();
            let alerts = |there: bool| -> usize {
                neg.iter()
                    .map(|d| d.plain[idx].alerts.iter().filter(|a| a.1 == there).count())
                    .sum()
            };
            let occ_h: f64 = neg.iter().map(|d| d.occupied_h).sum();
            let empty_h: f64 = neg.iter().map(|d| d.empty_h).sum();
            let (f_occ, f_empty) = (alerts(true), alerts(false));
            let (before, late) = all
                .iter()
                .filter_map(|d| Some(timing(d.lie_start?, &d.plain[idx].alerts, *nm)))
                .fold((0, 0), |(b, l), t| (b + t.before, l + t.late));
            println!(
                "  {:>4} {:>13} {:>13} {:>15} {:>15} {:>16} {:>16} {:>11} {:>6}",
                format!("{nm:.0}"),
                pct(ratio(missed(Case::Fall) as u64, per_case as u64)),
                pct(ratio(missed(Case::Slide) as u64, per_case as u64)),
                format!("{} s", num(median(&lat) - nm * 60.0, 0)),
                f_occ + f_empty,
                num(f_occ as f64 / occ_h, 3),
                num(f_empty as f64 / empty_h, 3),
                before,
                late
            );
        }
    }

    println!("\nWALKS OUT, presence hold ON: runs with a false alert on the room once it is truly empty, left the home");
    println!("against seen moving in another room {ELSEWHERE_AFTER_S:.0} s later (`long_lie_elsewhere`; the harness says so, there");
    println!("is no second room). Alerts while they still sat there count in the tables above, not here.");
    print!("  {:26}", "");
    for nm in NS_MIN {
        print!(" {:>8}", format!("N={nm:.0}"));
    }
    println!();
    for (label, other) in [("left the home", false), ("moving in another room", true)] {
        print!("  {label:26}");
        for ni in 0..NS_MIN.len() {
            let idx = NS_MIN.len() + ni;
            let k = of(Case::Departure)
                .filter(|d| {
                    let o = if other {
                        &d.elsewhere[idx]
                    } else {
                        &d.plain[idx]
                    };
                    o.alerts.iter().any(|a| !a.1)
                })
                .count();
            print!(" {:>8}", format!("{k}/{per_case}"));
        }
        println!();
    }

    println!("\nBY LAYOUT, N = 15: falls missed, hold off / on; still sitters alerted, hold off");
    let n15 = NS_MIN.iter().position(|&n| n == 15.0).expect("15 is swept");
    for l in layouts {
        let missed = |hold_i: usize| {
            all.iter()
                .filter(|d| d.case == Case::Fall && d.layout == *l)
                .filter(|d| in_time(d, &d.plain[hold_i * NS_MIN.len() + n15], 15.0).is_none())
                .count()
        };
        let sitters = all
            .iter()
            .filter(|d| {
                d.case == Case::SitStill && d.layout == *l && !d.plain[n15].alerts.is_empty()
            })
            .count();
        let per = per_case / layouts.len();
        println!(
            "  {:8} falls missed {:>3}/{per} / {:>3}/{per}   still sitters alerted {:>3}/{per}",
            l,
            missed(0),
            missed(1),
            sitters
        );
    }
    println!("\nSYNTHETIC. Simulated physics, bodies and noise; no real lie, fall, nap or home is in these numbers.");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alerts_before_the_lie_and_too_late_are_counted_not_dropped() {
        let lie = 1000.0;
        let n = 15.0;
        let t = timing(lie, &[(400.0, true), (lie + 901.0, true)], n);
        assert_eq!(t.in_time, Some(901.0));
        assert_eq!((t.before, t.late), (1, 0));
        let t = timing(lie, &[(lie + n * 60.0 + IN_TIME_S + 1.0, true)], n);
        assert_eq!(t.in_time, None);
        assert_eq!((t.before, t.late), (0, 1));
        let t = timing(lie, &[(lie + 60.0, true), (lie + 2000.0, true)], 3.0);
        assert_eq!(
            t,
            Timing {
                in_time: Some(60.0),
                before: 0,
                late: 1
            }
        );
        assert_eq!(
            timing(lie, &[], n),
            Timing {
                in_time: None,
                before: 0,
                late: 0
            }
        );
    }
}
