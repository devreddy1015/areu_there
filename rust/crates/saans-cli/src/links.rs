//! `saans links`: what each extra link buys, measured on layouts you could
//! actually build. SYNTHETIC.
//!
//! ADR-006 decided on link diversity and owes a measurement of the blind-
//! spot rate with 1, 2 and 3 links. `saans eval` reports per-layout numbers,
//! but its three layouts cannot answer the question: its "3 links" puts the
//! nodes somewhere else entirely, so link count and geometry change at once,
//! and it connects every pair of three nodes, which needs two transmitters.
//! The firmware as built has one (`rx_role.cpp` hard-codes `tx_id = 0xFF`).
//!
//! So here every layout shares one transmitter and adds receivers:
//!
//!   1 link   TX N0 -> N1
//!   2 links  TX N0 -> N1, N2          (three boards: docs/HARDWARE.md)
//!   3 links  TX N0 -> N1, N2, N3      (a fourth board)
//!
//! with N0..N2 the positions `saans-sim::scenarios` already uses and N3
//! their mirror image across the TX-N1 line. Same seats, same seeds, same
//! simulator settings as `saans eval`, so the only thing that changes
//! between rows is the thing being measured.
//!
//! Part 3 re-runs `saans eval`'s own 3-link geometry at presence quorum 2
//! and 1, because CAPABILITY_LIMITS §4 attributes eval's 3-link false-empty
//! rise to the quorum, and with the geometry changing too that was never
//! separated.
//!
//! What this cannot measure: the main *cost* of a quorum of 1. The
//! majority rule exists so that one link leaking a person from the next
//! room cannot mark this one occupied, and the simulator has no next room.
//! Its empty-room false-occupied figure is noise only, and a quorum-1 row
//! that looks free here is not free in a building.

use crate::eval::{self, Job, Kind, Lcg, LEAD_IN_S};
use crate::score::{self, BreathingScore, Occupancy, Run};
use saans_sim::scenario::{Activity, Node, Point, Room, Scenario, Segment};
use saans_sim::Simulator;
use std::sync::Mutex;

const N0: Point = (1.5, 2.5);
const N1: Point = (4.5, 2.5);
const N2: Point = (3.0, 4.6);
/// N2 mirrored across y = 2.5, the TX-N1 line.
const N3: Point = (3.0, 0.4);

/// The whole-band blind-spot threshold `evaluate.py` and `scan_seats` use.
const BLIND_BELOW: f64 = 0.2;
/// A link that sees breathing only weakly.
const WEAK_BELOW: f64 = 0.5;

const EMPTY_SCORED_S: f64 = 120.0;
const SEATED_S: f64 = 300.0;
const SETTLE_S: f64 = 60.0;

struct Config {
    name: &'static str,
    nodes: Vec<Node>,
    links: Vec<(u8, u8)>,
    /// 0 = the fusion default (1 below three usable links, else a majority).
    quorum: usize,
}

fn node(id: u8, pos: Point) -> Node {
    Node { id, pos }
}

fn configs() -> (Vec<Config>, Vec<Config>) {
    let nested = |k: usize| {
        let nodes: Vec<Node> = [N0, N1, N2, N3][..=k]
            .iter()
            .enumerate()
            .map(|(i, &p)| node(i as u8, p))
            .collect();
        let links: Vec<(u8, u8)> = (1..=k as u8).map(|rx| (0, rx)).collect();
        (nodes, links)
    };
    let (n1, l1) = nested(1);
    let (n2, l2) = nested(2);
    let (n3, l3) = nested(3);
    let buildable = vec![
        Config {
            name: "1 link",
            nodes: n1,
            links: l1,
            quorum: 0,
        },
        Config {
            name: "2 links",
            nodes: n2,
            links: l2,
            quorum: 0,
        },
        Config {
            name: "3 links, quorum 2 (default)",
            nodes: n3.clone(),
            links: l3.clone(),
            quorum: 0,
        },
        Config {
            name: "3 links, quorum 1",
            nodes: n3,
            links: l3,
            quorum: 1,
        },
    ];
    // `saans eval`'s 3-link layout, verbatim.
    let eval3 = eval::layouts()
        .into_iter()
        .find(|l| l.0 == "3 links")
        .expect("eval has a 3-link layout");
    let evals = vec![
        Config {
            name: "eval 3-link, quorum 2 (default)",
            nodes: eval3.1.clone(),
            links: eval3.2.clone(),
            quorum: 0,
        },
        Config {
            name: "eval 3-link, quorum 1",
            nodes: eval3.1,
            links: eval3.2,
            quorum: 1,
        },
    ];
    (buildable, evals)
}

/// The settings `saans eval` uses, so the two tools' numbers are comparable.
fn scenario(name: String, room: &Room, c: &Config, timeline: Vec<Segment>, seed: u64) -> Scenario {
    Scenario {
        name,
        room: room.clone(),
        nodes: c.nodes.clone(),
        links: c.links.clone(),
        timeline,
        sample_rate_hz: 20.0,
        n_subcarriers: 56,
        noise_std: 2.0,
        loss_prob: 0.02,
        body_points: 3,
        body_width_m: 0.3,
        seed,
    }
}

fn best(sim: &Simulator, link: (u8, u8), seat: Point) -> f64 {
    sim.sensitivity_at(link, seat)
        .into_iter()
        .fold(0.0, f64::max)
}

/// Part 1: ground truth only. For each seat, does every link of the layout
/// sit in a whole-band null? No estimator involved, so it can cover many
/// seats cheaply.
fn blind_spots(configs: &[Config], seats: u64) -> Vec<(f64, f64)> {
    let mut out = vec![(0u64, 0u64); configs.len()];
    let mut n = 0u64;
    for (_, room) in eval::rooms() {
        for s in 0..seats {
            let seed = 50_000 + s;
            let mut rng = Lcg::new(seed ^ 0x51ab_e7c0);
            let seat = rng.seat();
            n += 1;
            for (i, c) in configs.iter().enumerate() {
                let timeline = vec![Segment {
                    duration_s: 1.0,
                    activity: Activity::Absent,
                }];
                let sc = scenario(String::new(), &room, c, timeline, seed);
                let sim = Simulator::new(&sc);
                let b: Vec<f64> = c.links.iter().map(|&l| best(&sim, l, seat)).collect();
                let top = b.iter().cloned().fold(0.0, f64::max);
                out[i].0 += (top < BLIND_BELOW) as u64;
                out[i].1 += (top < WEAK_BELOW) as u64;
            }
        }
    }
    out.iter()
        .map(|&(b, w)| (100.0 * b as f64 / n as f64, 100.0 * w as f64 / n as f64))
        .collect()
}

#[derive(Default)]
struct Tally {
    occ: Occupancy,
    br: BreathingScore,
    /// Seats at which a still person was reported empty more than 10% of
    /// the time: the people the system mostly misses.
    seats_missed: u64,
    seats: u64,
}

fn job(c: &Config, room_name: &'static str, room: &Room, s: u64) -> Job {
    let seed = 70_000 + s * 7919;
    let mut rng = Lcg::new(seed ^ 0x9E37_79B9_7F4A_7C15);
    let seat = rng.seat();
    let bpm = rng.range(11.0, 17.0);
    let timeline = vec![
        Segment {
            duration_s: LEAD_IN_S,
            activity: Activity::Absent,
        },
        Segment {
            duration_s: EMPTY_SCORED_S,
            activity: Activity::Absent,
        },
        Segment {
            duration_s: 15.0,
            activity: Activity::WalkTo {
                to: seat,
                speed_mps: 0.9,
            },
        },
        Segment {
            duration_s: SEATED_S,
            activity: Activity::Seated {
                at: seat,
                bpm,
                bpm_end: None,
            },
        },
    ];
    Job {
        kind: Kind::Reader,
        room: room_name,
        layout: c.name,
        seed,
        scenario: scenario(
            format!("links/{room_name}/{}/{s}", c.name),
            room,
            c,
            timeline,
            seed,
        ),
    }
}

/// Part 2 and 3: the production path, every config on identical seats.
fn pipeline(configs: &[Config], seats: u64, threads: usize) -> Vec<Tally> {
    let mut work = Vec::new();
    for (i, c) in configs.iter().enumerate() {
        for (room_name, room) in eval::rooms() {
            for s in 0..seats {
                work.push((i, job(c, room_name, &room, s), c.quorum));
            }
        }
    }
    let queue = Mutex::new(work);
    let tallies: Mutex<Vec<Tally>> = Mutex::new(configs.iter().map(|_| Tally::default()).collect());
    std::thread::scope(|sc| {
        for _ in 0..threads.max(1) {
            sc.spawn(|| loop {
                let Some((i, j, q)) = queue.lock().unwrap().pop() else {
                    break;
                };
                let run: Run = eval::run_with_quorum(&j, q);
                let o = score::occupancy(&run);
                let b = score::breathing(&run, SETTLE_S);
                let mut t = tallies.lock().unwrap();
                let t = &mut t[i];
                t.seats += 1;
                if o.still_present > 0 && o.still_false_empty * 10 > o.still_present {
                    t.seats_missed += 1;
                }
                t.occ.add(&o);
                t.br.add(&b);
            });
        }
    });
    tallies.into_inner().unwrap()
}

fn pct(x: f64) -> String {
    if x.is_finite() {
        format!("{:.1}%", 100.0 * x)
    } else {
        "-".into()
    }
}

pub fn run(seats: u64, blind_seats: u64, threads: usize) {
    let start = std::time::Instant::now();
    let (buildable, evals) = configs();

    println!(
        "saans links -- SYNTHETIC. What each extra link buys, on layouts with one transmitter."
    );
    println!("TX N0 (1.5, 2.5); receivers N1 (4.5, 2.5), N2 (3.0, 4.6), N3 (3.0, 0.4), added in that order.");
    println!("Rooms, noise, loss and body as `saans eval`. Same seats and seeds in every row.");

    let blind = blind_spots(&buildable[..3], blind_seats);
    println!(
        "\nPART 1. Ground truth: share of seats where EVERY link is weak or blind ({} seats).",
        2 * blind_seats
    );
    println!(
        "{:<34} {:>22} {:>22}",
        "layout", "blind (best < 0.2)", "weak (best < 0.5)"
    );
    for (c, (b, w)) in buildable[..3].iter().zip(&blind) {
        let name = c.name.split(',').next().unwrap();
        println!("{:<34} {:>21.1}% {:>21.1}%", name, b, w);
    }

    let all: Vec<Config> = buildable.into_iter().chain(evals).collect();
    let t = pipeline(&all, seats, threads);
    println!(
        "\nPARTS 2 and 3. Production path (FeatureExtractor -> SpaceFusion), {} seats per row:",
        2 * seats
    );
    println!("{EMPTY_SCORED_S:.0} s of empty room after calibration, then a still breather for {SEATED_S:.0} s.");
    println!(
        "{:<34} {:>12} {:>14} {:>10} {:>10} {:>10} {:>12}",
        "configuration", "still false-", "seats mostly", "empty", "breathing", "within", "unknown"
    );
    println!(
        "{:<34} {:>12} {:>14} {:>10} {:>10} {:>10} {:>12}",
        "", "empty", "missed (>10%)", "false-occ", "coverage", "1 BPM", ""
    );
    for (i, (c, t)) in all.iter().zip(&t).enumerate() {
        if i == 4 {
            println!("{}", "-".repeat(108));
        }
        println!(
            "{:<34} {:>12} {:>14} {:>10} {:>10} {:>10} {:>12}",
            c.name,
            pct(t.occ.false_empty_still()),
            format!("{}/{}", t.seats_missed, t.seats),
            pct(t.occ.false_occupied()),
            pct(t.br.coverage()),
            pct(t.br.accuracy_when_reported()),
            pct(t.occ.unknown()),
        );
    }
    println!(
        "\nThe last two rows are `saans eval`'s own 3-link geometry, which moves the nodes as well"
    );
    println!("as adding a link; the quorum is the only thing that differs between them.");
    println!(
        "Empty false-occ is noise only: the simulator has no neighbouring room, so the leak the"
    );
    println!("majority quorum guards against does not exist here, and a quorum-1 row is not free.");
    eprintln!("({:.0} s)", start.elapsed().as_secs_f64());
}
