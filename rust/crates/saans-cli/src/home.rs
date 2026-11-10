//! `saans export-home`: a simulated home over whole days, as one pcap a
//! gateway can replay (issue #39). SYNTHETIC.
//!
//! Three rooms, one link each, one person moving between them on a fixed
//! daily plan, so the daily routine has known answers:
//!
//! | Local time | Where | What |
//! |---|---|---|
//! | 00:00-06:30 | bedroom | asleep (still, breathing), with a 4-minute bathroom visit at 02:30 |
//! | 06:30 | bedroom, then living room | gets up: walks for 15 minutes, then sits |
//! | 07:30-21:45 | living room | sits fidgeting, walks about each hour, a bathroom visit every 3 hours |
//! | 22:00-24:00 | bedroom | asleep, with a bathroom visit at 23:30 (the next day's night) |
//!
//! So each day after the first should read: up at about 06:30, two night
//! bathroom visits, active most hours from 06 to 21. The first day starts
//! with every room empty for calibration. Its night began before the
//! recording, so its night bathroom visits read unknown.
//!
//! Nodes: bedroom 10 -> 11, bathroom 20 -> 21, living room 30 -> 31.
//! Times are anchored at a UTC midnight, so with `utc_offset_minutes = 0`
//! the replayed days are the plan's days.

use crate::eval::LEAD_IN_S;
use crate::pcap;
use saans_dsp::features::{FeatureExtractor, FeatureExtractorConfig};
use saans_sim::scenario::{Activity, Node, Point, Room, Scenario, Segment, NODE_Z_M};
use saans_sim::Simulator;
use std::process::ExitCode;

const DAY: f64 = 86_400.0;
const ENTER_S: f64 = 15.0;
/// 2026-09-22 00:00 UTC.
const EPOCH_S: u64 = 1_790_035_200;

const BEDROOM: usize = 0;
const BATHROOM: usize = 1;
const LIVING: usize = 2;
const ROOMS: [(&str, u8, u8); 3] = [
    ("bedroom", 10, 11),
    ("bathroom", 20, 21),
    ("living", 30, 31),
];
const SEAT: Point = (3.0, 3.6);

/// One person's day as (room, duration, activity) steps; entering a room
/// is a short walk to its seat.
struct Plan {
    steps: Vec<(usize, f64, Activity)>,
    room: Option<usize>,
}

impl Plan {
    fn new() -> Self {
        Plan {
            steps: Vec::new(),
            room: None,
        }
    }

    /// `duration` in `room` doing `act`, walking in first if they were
    /// elsewhere.
    fn at(&mut self, room: usize, duration: f64, act: Activity) {
        let mut d = duration;
        if self.room != Some(room) {
            let walk = Activity::WalkTo {
                to: SEAT,
                speed_mps: 0.9,
            };
            self.steps.push((room, ENTER_S, walk));
            d -= ENTER_S;
            self.room = Some(room);
        }
        self.steps.push((room, d, act));
    }

    fn total(&self) -> f64 {
        self.steps.iter().map(|s| s.1).sum()
    }
}

fn still(bpm: f64) -> Activity {
    Activity::Seated {
        at: SEAT,
        bpm,
        bpm_end: None,
    }
}

fn fidget() -> Activity {
    Activity::Fidget {
        at: SEAT,
        bpm: 14.0,
        every_s: 60.0,
        burst_s: 3.0,
    }
}

fn walk() -> Activity {
    Activity::Walk { speed_mps: 0.9 }
}

/// The plan for `days` days, from local midnight of day 0.
fn plan(days: u32) -> Plan {
    let mut p = Plan::new();
    let h = 3600.0;
    for d in 0..days {
        let start = d as f64 * DAY;
        if d == 0 {
            // Every room empty while the links calibrate.
            p.steps.push((BEDROOM, LEAD_IN_S, Activity::Absent));
        }
        p.at(BEDROOM, start + 2.5 * h - p.total(), still(12.0));
        p.at(BATHROOM, 240.0, still(15.0));
        p.at(BEDROOM, start + 6.5 * h - p.total(), still(12.0));
        // Up: about the bedroom, then the living room.
        p.steps.push((BEDROOM, 60.0, walk()));
        p.at(LIVING, 15.0 * 60.0, walk());
        p.at(LIVING, start + 7.5 * h - p.total(), fidget());
        for hour in 7..22 {
            let next = start + (hour as f64 + 1.5) * h;
            if hour % 3 == 0 {
                p.at(BATHROOM, 240.0, still(15.0));
            }
            p.at(LIVING, 5.0 * 60.0, walk());
            let until = if hour == 21 { start + 22.0 * h } else { next };
            p.at(LIVING, until - p.total(), fidget());
        }
        p.at(BEDROOM, start + 23.5 * h - p.total(), still(12.0));
        p.at(BATHROOM, 240.0, still(15.0));
        p.at(BEDROOM, start + DAY - p.total(), still(12.0));
    }
    p
}

/// Each room's timeline: the plan where the person is there, absent
/// everywhere else.
fn timelines(p: &Plan) -> Vec<Vec<Segment>> {
    (0..ROOMS.len())
        .map(|r| {
            p.steps
                .iter()
                .map(|(room, d, act)| Segment {
                    duration_s: *d,
                    activity: if *room == r {
                        act.clone()
                    } else {
                        Activity::Absent
                    },
                })
                .collect()
        })
        .collect()
}

pub fn export_home(out: &str, days: u32, seed: u64) -> ExitCode {
    let days = days.max(1);
    let p = plan(days);
    let mut sims = Vec::new();
    let mut extractors = Vec::new();
    for (i, timeline) in timelines(&p).into_iter().enumerate() {
        let (name, tx, rx) = ROOMS[i];
        let node = |id, pos| Node {
            id,
            pos,
            z_m: NODE_Z_M,
        };
        let sc = Scenario {
            name: format!("home/{name}"),
            room: Room::default(),
            nodes: vec![node(tx, (1.5, 2.5)), node(rx, (4.5, 2.5))],
            links: vec![(tx, rx)],
            timeline,
            sample_rate_hz: 20.0,
            n_subcarriers: 56,
            noise_std: 2.0,
            loss_prob: 0.02,
            body_points: 3,
            body_width_m: 0.3,
            seed: seed.wrapping_mul(31).wrapping_add(i as u64),
        };
        if let Err(e) = Simulator::try_new(&sc) {
            eprintln!("{name}: {e}");
            return ExitCode::FAILURE;
        }
        extractors.push(FeatureExtractor::new(FeatureExtractorConfig {
            n_subcarriers: sc.n_subcarriers,
            sample_rate_hz: sc.sample_rate_hz,
            node_id: rx,
            tx_id: tx,
            vitals_enabled: false,
            ..Default::default()
        }));
        sims.push(Simulator::new(&sc));
    }
    let file = match std::fs::File::create(out) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{out}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut w = pcap::PcapWriter::new(std::io::BufWriter::new(file)).expect("pcap header");
    let epoch_ns = EPOCH_S * 1_000_000_000;
    let mut frames = 0u64;
    // Lockstep: every room's simulator ticks at the same instants, so the
    // file is in time order without buffering a day of frames.
    // Frames carry each simulator's arrival jitter, so they are held in a
    // small buffer and written only once nothing earlier can still come:
    // older than a second before the slowest simulator's clock.
    let mut pending: Vec<(u64, u8, Vec<u8>)> = Vec::new();
    let mut write = |pending: &mut Vec<(u64, u8, Vec<u8>)>, upto: u64| {
        pending.sort_by_key(|p| p.0);
        let n = pending.partition_point(|p| p.0 < upto);
        for (t_ns, rx, bytes) in pending.drain(..n) {
            w.write(
                t_ns,
                ([10, 0, 0, 100 + rx], 5566),
                ([10, 0, 0, 1], 5566),
                &bytes,
            )
            .expect("write");
            frames += 1;
        }
    };
    'run: loop {
        let mut now = f64::INFINITY;
        for (sim, ex) in sims.iter_mut().zip(extractors.iter_mut()) {
            let Some(tick) = sim.step() else {
                break 'run;
            };
            now = now.min(tick.truth.t_s);
            for lf in tick.frames {
                for ff in ex.push(lf.frame, Some(lf.seq as u64), Some(lf.t_s), None) {
                    let t_ns = epoch_ns + (lf.t_s * 1e9) as u64;
                    pending.push((t_ns, lf.rx, ff.encode().to_vec()));
                }
            }
        }
        write(&mut pending, epoch_ns + ((now - 1.0).max(0.0) * 1e9) as u64);
    }
    write(&mut pending, u64::MAX);
    w.flush().expect("flush");
    eprintln!(
        "home: {frames} FeatureFrames, {days} day(s); bedroom 10->11, bathroom 20->21, living 30->31 -> {out}"
    );
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_plan_fills_whole_days_and_starts_empty() {
        let p = plan(3);
        assert!((p.total() - 3.0 * DAY).abs() < 1e-6, "{}", p.total());
        assert!(matches!(p.steps[0].2, Activity::Absent));
        assert!(p.steps.iter().all(|s| s.1 > 0.0), "no step of zero length");
        // The same steps in every room, so every simulator has the same length.
        let t = timelines(&p);
        assert!(t.iter().all(|tl| tl.len() == p.steps.len()));
    }
}
