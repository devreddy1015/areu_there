//! `saans octave`: the doubled breathing rate at a Fresnel null, measured
//! through the production path with two links. SYNTHETIC.
//!
//! At a stationary point of a link's response, amplitude follows the
//! *square* of chest displacement, so a breather at `f` reads as `2f`, at
//! high confidence. The simulator drives the chest with a pure sinusoid
//! (`synthetic.py:332`, `saans-sim`'s physics), and `sin^2` is a pure
//! sinusoid at `2f`: on one link, at a true null, a doubled reading is
//! exactly what a person breathing twice as fast produces. So the question
//! this measures is not whether one link can tell -- here it cannot -- but
//! what fusion does when one link is in a null and another is not.
//!
//! Cases are classified from ground truth before anything runs: a link is
//! "null" when no subcarrier's breathing sensitivity reaches 0.2 (the same
//! threshold `evaluate.py` and `scan_seats` use for a whole-band blind
//! spot) and "clear" when its best subcarrier reaches 0.5. Breathing rates
//! are 8-14 BPM so that twice the rate stays inside the 6-30 BPM band;
//! above 15 BPM a doubled reading is out of band and cannot happen, and
//! those cases would only dilute the number.
//!
//! "2x" here is the definition `evaluate.py` uses: within 1 BPM of twice
//! the truth.

use crate::eval::{self, Job, Kind, Lcg, LEAD_IN_S};
use crate::score::Run;
use saans_sim::scenario::{Activity, Point, Segment};
use saans_sim::{scenarios, Simulator};
use std::sync::Mutex;

/// A link is in a whole-band null below this best-subcarrier sensitivity.
const NULL_BELOW: f64 = 0.2;
/// ... and clearly sees breathing at or above this.
const CLEAR_FROM: f64 = 0.5;
const DURATION_S: f64 = 300.0;
/// Scored from this long after the person is still and breathing: the
/// estimator's window plus margin, as `score::breathing` uses.
const SETTLE_S: f64 = 60.0;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Class {
    /// One link null, the other clear: the case the octave rule is for.
    OneNull,
    /// Both clear: where any rule must do no harm.
    BothClear,
    /// Both null: nothing fusion can do; reported so it is not hidden.
    BothNull,
}

impl Class {
    fn name(self) -> &'static str {
        match self {
            Class::OneNull => "one link null, one clear",
            Class::BothClear => "both links clear",
            Class::BothNull => "both links null",
        }
    }
}

struct Case {
    class: Class,
    null_link: Option<(u8, u8)>,
    bpm: f64,
    job: Job,
}

#[derive(Default)]
struct Tally {
    runs: u64,
    eligible: u64,
    reported: u64,
    within_1: u64,
    doubled: u64,
    doubled_confident: u64,
    from_null_link: u64,
    errors: Vec<f64>,
}

/// At the seat, not at the body's current position: every case starts
/// with an empty room (presence must calibrate on one), so at t = 0
/// `Simulator::sensitivity` would describe wherever an absent body is
/// parked, not where the breather will sit.
fn best_sensitivity(sim: &Simulator, link: (u8, u8), seat: Point) -> f64 {
    sim.sensitivity_at(link, seat)
        .into_iter()
        .fold(0.0, f64::max)
}

fn cases(seeds: u64, per_class: usize) -> Vec<Case> {
    let mut out = Vec::new();
    let mut count = [0usize; 3];
    let rooms = [
        ("clear-LoS", scenarios::room_clear_los()),
        ("rich-NLoS", scenarios::room_rich_nlos()),
    ];
    'search: for seed in 0..seeds {
        for (room_name, room) in &rooms {
            let mut rng = Lcg::new(seed.wrapping_mul(7919) ^ 0x000c_7ae5);
            let at = rng.seat();
            let bpm = rng.range(8.0, 14.0);
            // The same shape as `eval`'s timelines: an empty lead-in so the
            // presence threshold calibrates on an empty room, a walk to the
            // seat, then the measured stretch. Seating the person at t = 0
            // would calibrate presence on the breather.
            let timeline = vec![
                Segment {
                    duration_s: LEAD_IN_S,
                    activity: Activity::Absent,
                },
                Segment {
                    duration_s: 15.0,
                    activity: Activity::WalkTo {
                        to: at,
                        speed_mps: 0.9,
                    },
                },
                Segment {
                    duration_s: DURATION_S,
                    activity: Activity::Seated {
                        at,
                        bpm,
                        bpm_end: None,
                    },
                },
            ];
            let name = format!("octave/{room_name}/{seed}");
            let sc =
                scenarios::scenario(&name, room.clone(), scenarios::two_links(), timeline, seed);
            let sim = Simulator::new(&sc);
            let s: Vec<f64> = sc
                .links
                .iter()
                .map(|&l| best_sensitivity(&sim, l, at))
                .collect();
            let (class, null_link) = match (s[0] < NULL_BELOW, s[1] < NULL_BELOW) {
                (true, false) if s[1] >= CLEAR_FROM => (Class::OneNull, Some(sc.links[0])),
                (false, true) if s[0] >= CLEAR_FROM => (Class::OneNull, Some(sc.links[1])),
                (false, false) if s[0] >= CLEAR_FROM && s[1] >= CLEAR_FROM => {
                    (Class::BothClear, None)
                }
                (true, true) => (Class::BothNull, None),
                _ => continue,
            };
            let k = class as usize;
            if count[k] >= per_class {
                if count.iter().all(|&c| c >= per_class) {
                    break 'search;
                }
                continue;
            }
            count[k] += 1;
            out.push(Case {
                class,
                null_link,
                bpm,
                job: Job {
                    kind: Kind::Reader,
                    room: room_name,
                    layout: "2 links",
                    seed,
                    scenario: sc,
                    fall_hold_min: 0.0,
                },
            });
        }
    }
    out
}

fn score(case: &Case, run: &Run, t: &mut Tally) {
    t.runs += 1;
    let mut steady_since: Option<f64> = None;
    for (truth, s) in &run.samples {
        let steady = truth.present && truth.still && truth.breathing;
        steady_since = if steady {
            steady_since.or(Some(truth.t_s))
        } else {
            None
        };
        let Some(since) = steady_since else { continue };
        if truth.t_s - since < SETTLE_S {
            continue;
        }
        t.eligible += 1;
        let Some(br) = s.breathing else { continue };
        if br.confidence < 0.4 {
            continue;
        }
        t.reported += 1;
        let err = (br.bpm - case.bpm).abs();
        t.errors.push(err);
        if err < 1.0 {
            t.within_1 += 1;
        }
        if (br.bpm - 2.0 * case.bpm).abs() < 1.0 {
            t.doubled += 1;
            if br.confidence >= 0.7 {
                t.doubled_confident += 1;
            }
        }
        if let Some((tx, rx)) = case.null_link {
            if br.link.tx == tx && br.link.rx == rx {
                t.from_null_link += 1;
            }
        }
    }
}

pub fn run(seeds: u64, per_class: usize, threads: usize) {
    let start = std::time::Instant::now();
    let all = cases(seeds, per_class);
    eprintln!("saans octave: {} cases from up to {seeds} seeds", all.len());
    let queue = Mutex::new(all.into_iter().collect::<Vec<_>>());
    let tallies: Mutex<[Tally; 3]> = Mutex::new(Default::default());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                let Some(case) = queue.lock().unwrap().pop() else {
                    break;
                };
                let r = eval::run(&case.job);
                let mut t = tallies.lock().unwrap();
                score(&case, &r, &mut t[case.class as usize]);
            });
        }
    });
    let mut tallies = tallies.into_inner().unwrap();
    println!("saans octave -- SYNTHETIC. Two links (one TX, two RX), seated breather at 8-14 BPM,");
    println!("{DURATION_S:.0} s per case, scored from {SETTLE_S:.0} s. Production path:");
    println!("FeatureExtractor -> SpaceFusion. Null: best subcarrier sensitivity < {NULL_BELOW};");
    println!("clear: >= {CLEAR_FROM}. '2x' = within 1 BPM of twice the truth.");
    println!();
    println!(
        "{:<26} {:>5} {:>9} {:>9} {:>7} {:>7} {:>9} {:>10} {:>8}",
        "case", "runs", "reported", "<1 BPM", "2x", "2x@.7", "median", "null link", "seconds"
    );
    for class in [Class::OneNull, Class::BothClear, Class::BothNull] {
        let t = &mut tallies[class as usize];
        t.errors.sort_by(f64::total_cmp);
        let pct = |n: u64, d: u64| {
            if d == 0 {
                f64::NAN
            } else {
                100.0 * n as f64 / d as f64
            }
        };
        let median = t
            .errors
            .get(t.errors.len() / 2)
            .copied()
            .unwrap_or(f64::NAN);
        let null_col = if class == Class::OneNull {
            format!("{:.1}%", pct(t.from_null_link, t.reported))
        } else {
            "-".into()
        };
        println!(
            "{:<26} {:>5} {:>8.1}% {:>8.1}% {:>6.1}% {:>6.1}% {:>9.3} {:>10} {:>8}",
            class.name(),
            t.runs,
            pct(t.reported, t.eligible),
            pct(t.within_1, t.reported),
            pct(t.doubled, t.reported),
            pct(t.doubled_confident, t.reported),
            median,
            null_col,
            t.eligible / 2,
        );
    }
    println!();
    println!("'reported' = share of eligible samples with fused breathing at confidence >= 0.4;");
    println!("'<1 BPM', '2x', '2x@.7' and 'null link' are shares of the reported samples.");
    println!("'null link' = how often fusion took its breathing from the link in the null.");
    eprintln!("({:.0} s)", start.elapsed().as_secs_f64());
}
