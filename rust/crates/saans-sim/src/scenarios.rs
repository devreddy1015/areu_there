//! Ready-made rooms, node layouts and timelines for the evaluation.
//! All SYNTHETIC.
//!
//! Every timeline starts with [`LEAD_IN_S`] of empty room: the pipeline
//! needs ~50 s of filter warmup plus 30 s of empty-room calibration, and
//! the scorer ignores the first 95 s. Positions are for the default
//! 6 m x 5 m room and lie inside its walkable area (0.4 m from the walls).
//!
//! ```
//! use saans_sim::scenarios::*;
//! let sc = scenario("reader", room_clear_los(), two_links(), reader(10.0), 7);
//! let mut sim = saans_sim::Simulator::new(&sc);
//! let first = sim.step().unwrap();
//! assert!(!first.truth.present);
//! ```

use crate::physics::dist;
use crate::rng::Rng;
use crate::scenario::{Activity, Node, Point, Room, Scenario, Segment};

/// Empty room before anyone enters (>= the scorer's 95 s settle).
pub const LEAD_IN_S: f64 = 120.0;
/// Empty room after the person leaves.
pub const LEAD_OUT_S: f64 = 60.0;
pub const WALK_SPEED_MPS: f64 = 1.0;
/// Matches `Room::default().door_m`.
pub const DOOR: Point = (0.6, 0.6);
/// Not the Python default body position (3.0, 3.5): for the 3 m
/// one-link layout in the clear-LoS room that spot sits near a Fresnel
/// boundary and is a whole-band blind spot for 36% of room seeds. Here
/// that rate is 10% (clear LoS) / 4% (rich NLoS), never blind on all three
/// links; see `examples/scan_seats.rs`.
pub const CHAIR: Point = (3.1, 3.7);
pub const BED: Point = (3.0, 1.3);
pub const DESK: Point = (4.2, 3.9);
/// Where `collapse()` falls from and to (1.2 m apart).
pub const FALL_FROM: Point = (2.2, 3.4);
pub const FALL_TO: Point = (3.3, 3.9);

/// A node layout: positions plus (tx, rx) links.
#[derive(Debug, Clone)]
pub struct Layout {
    pub nodes: Vec<Node>,
    pub links: Vec<(u8, u8)>,
}

// ---------------------------------------------------------------- rooms

/// Clear line of sight: order-1 reflections, Gamma 0.4, 6 scatterers
/// (the Python evaluation's "clear-LoS").
pub fn room_clear_los() -> Room {
    Room::default()
}

/// Order 2, LoS -6 dB, 12 scatterers ("attenuated-LoS").
pub fn room_attenuated_los() -> Room {
    Room {
        reflection_order: 2,
        los_attenuation_db: 6.0,
        n_scatterers: 12,
        ..Room::default()
    }
}

/// Order 2, LoS -15 dB, 12 scatterers, Gamma 0.6 ("rich-NLoS").
pub fn room_rich_nlos() -> Room {
    Room {
        reflection_order: 2,
        los_attenuation_db: 15.0,
        n_scatterers: 12,
        wall_reflection: 0.6,
        ..Room::default()
    }
}

// -------------------------------------------------------------- layouts

fn nodes(ps: &[Point]) -> Vec<Node> {
    ps.iter()
        .enumerate()
        .map(|(i, &pos)| Node { id: i as u8, pos })
        .collect()
}

const N0: Point = (1.5, 2.5);
const N1: Point = (4.5, 2.5);
const N2: Point = (3.0, 4.6);

/// Two nodes 3 m apart, one link (the Python reference geometry).
pub fn one_link() -> Layout {
    Layout {
        nodes: nodes(&[N0, N1]),
        links: vec![(0, 1)],
    }
}

/// Three nodes, two links at different angles (horizontal and ~54 deg).
pub fn two_links() -> Layout {
    Layout {
        nodes: nodes(&[N0, N1, N2]),
        links: vec![(0, 1), (0, 2)],
    }
}

/// The same three nodes, all three links.
pub fn three_links() -> Layout {
    Layout {
        nodes: nodes(&[N0, N1, N2]),
        links: vec![(0, 1), (0, 2), (1, 2)],
    }
}

// ------------------------------------------------------------ timelines

fn seg(duration_s: f64, activity: Activity) -> Segment {
    Segment {
        duration_s,
        activity,
    }
}

fn absent(duration_s: f64) -> Segment {
    seg(duration_s, Activity::Absent)
}

/// Walk from `from` to `to` at walking speed, plus a second standing.
fn walk(from: Point, to: Point) -> Segment {
    seg(
        dist(from, to) / WALK_SPEED_MPS + 1.0,
        Activity::WalkTo {
            to,
            speed_mps: WALK_SPEED_MPS,
        },
    )
}

fn seated(duration_s: f64, at: Point, bpm: f64, bpm_end: Option<f64>) -> Segment {
    seg(duration_s, Activity::Seated { at, bpm, bpm_end })
}

/// Nobody, for `hours`.
pub fn empty(hours: f64) -> Vec<Segment> {
    vec![absent(hours * 3600.0)]
}

/// Walk in, sit still reading for `minutes` (15 -> 14 BPM), walk out.
pub fn reader(minutes: f64) -> Vec<Segment> {
    vec![
        absent(LEAD_IN_S),
        walk(DOOR, CHAIR),
        seated(minutes * 60.0, CHAIR, 15.0, Some(14.0)),
        walk(CHAIR, DOOR),
        absent(LEAD_OUT_S),
    ]
}

/// [`night_seeded`] with seed 1.
pub fn night(hours: f64, pauses_per_hour: f64, pause_s: (f64, f64)) -> Vec<Segment> {
    night_seeded(hours, pauses_per_hour, pause_s, 1)
}

/// Fidget episodes per hour of the night (each ~20 s, a burst or two).
pub const NIGHT_FIDGETS_PER_HOUR: f64 = 1.5;
const NIGHT_FIDGET_S: f64 = 20.0;
const NIGHT_SETTLE_S: f64 = 180.0;
const NIGHT_MIN_GAP_S: f64 = 90.0;
const NIGHT_CHUNK_S: f64 = 900.0;

/// Walk in, lie still breathing 12-16 BPM (drifting), with apnea-like
/// pauses (Poisson, `pauses_per_hour`, durations uniform in `pause_s`)
/// and occasional fidget episodes, then walk out. Events are at least
/// 90 s apart, the first 3 min after lying down, so each pause follows
/// steady breathing. The night lasts `hours` in total.
pub fn night_seeded(
    hours: f64,
    pauses_per_hour: f64,
    pause_s: (f64, f64),
    seed: u64,
) -> Vec<Segment> {
    let mut rng = Rng::stream(seed, 0x004E_4947_4854);
    let walk_in = walk(DOOR, BED);
    let walk_out = walk(BED, DOOR);
    let lying =
        (hours * 3600.0 - LEAD_IN_S - walk_in.duration_s - walk_out.duration_s - LEAD_OUT_S)
            .max(NIGHT_SETTLE_S + 60.0);
    let (pmin, pmax) = (
        pause_s.0.min(pause_s.1).max(1.0),
        pause_s.0.max(pause_s.1).max(1.0),
    );
    let pph = pauses_per_hour.max(0.0);
    let rate = (pph + NIGHT_FIDGETS_PER_HOUR) / 3600.0;

    // (start, duration, is_pause) within the lying period.
    let mut events: Vec<(f64, f64, bool)> = Vec::new();
    let mut t = NIGHT_SETTLE_S;
    let mut prev_end = f64::NEG_INFINITY;
    loop {
        t += rng.exp(1.0 / rate);
        t = t.max(prev_end + NIGHT_MIN_GAP_S);
        let is_pause = rng.uniform() < pph / (pph + NIGHT_FIDGETS_PER_HOUR);
        let d = if is_pause {
            rng.range(pmin, pmax)
        } else {
            NIGHT_FIDGET_S
        };
        if t + d + 60.0 > lying {
            break;
        }
        events.push((t, d, is_pause));
        prev_end = t + d;
        t = prev_end;
    }

    let mut out = vec![absent(LEAD_IN_S), walk_in];
    let mut bpm = rng.range(12.0, 16.0);
    /// Still lying for `len` s in <= 15 min chunks, the rate drifting.
    fn fill(out: &mut Vec<Segment>, rng: &mut Rng, bpm: &mut f64, len: f64) {
        if len <= 0.0 {
            return;
        }
        let n = (len / NIGHT_CHUNK_S).ceil().max(1.0);
        for _ in 0..n as usize {
            let next = (*bpm + rng.range(-1.5, 1.5)).clamp(12.0, 16.0);
            out.push(seated(len / n, BED, *bpm, Some(next)));
            *bpm = next;
        }
    }
    let mut cursor = 0.0;
    for (start, d, is_pause) in events {
        fill(&mut out, &mut rng, &mut bpm, start - cursor);
        if is_pause {
            out.push(seg(d, Activity::BreathingPause { at: BED }));
        } else {
            out.push(seg(
                d,
                Activity::Fidget {
                    at: BED,
                    bpm,
                    every_s: 6.0,
                    burst_s: 3.0,
                },
            ));
        }
        cursor = start + d;
    }
    fill(&mut out, &mut rng, &mut bpm, lying - cursor);
    out.push(walk_out);
    out.push(absent(LEAD_OUT_S));
    out
}

/// Walk in, wander for 30 s, walk to [`FALL_FROM`], fall to [`FALL_TO`]
/// (0.8 s), lie still breathing 18 BPM for 5 min.
pub fn collapse() -> Vec<Segment> {
    vec![
        absent(LEAD_IN_S),
        walk(DOOR, CHAIR),
        seg(
            30.0,
            Activity::Walk {
                speed_mps: WALK_SPEED_MPS,
            },
        ),
        // From anywhere in the room: the longest walk is under 7 s.
        seg(
            8.0,
            Activity::WalkTo {
                to: FALL_FROM,
                speed_mps: WALK_SPEED_MPS,
            },
        ),
        seg(
            0.8 + 300.0,
            Activity::Fall {
                to: FALL_TO,
                bpm: 18.0,
            },
        ),
    ]
}

/// Walk in, sit down heavily on [`CHAIR`] (0.4 m in 0.6 s), read 10 min,
/// walk out.
pub fn sit_down() -> Vec<Segment> {
    let d = dist(DOOR, CHAIR);
    let approach = (
        CHAIR.0 + (DOOR.0 - CHAIR.0) * 0.4 / d,
        CHAIR.1 + (DOOR.1 - CHAIR.1) * 0.4 / d,
    );
    vec![
        absent(LEAD_IN_S),
        walk(DOOR, approach),
        seg(
            0.6 + 600.0,
            Activity::SitDown {
                at: CHAIR,
                bpm: 16.0,
            },
        ),
        walk(CHAIR, DOOR),
        absent(LEAD_OUT_S),
    ]
}

/// Seated: 3 min at 16 BPM, then 5 min fast breathing drifting 26 -> 30
/// BPM, then 2 min at 16 BPM; walk out.
pub fn tachypnea() -> Vec<Segment> {
    vec![
        absent(LEAD_IN_S),
        walk(DOOR, CHAIR),
        seated(180.0, CHAIR, 16.0, None),
        seated(300.0, CHAIR, 26.0, Some(30.0)),
        seated(120.0, CHAIR, 16.0, None),
        walk(CHAIR, DOOR),
        absent(LEAD_OUT_S),
    ]
}

/// Walk to [`DESK`], fidgety work for `minutes` (a 3 s burst every ~20 s),
/// then motionless (still breathing, 13 BPM) for 40 min.
pub fn lone_worker(minutes: f64) -> Vec<Segment> {
    vec![
        absent(LEAD_IN_S),
        walk(DOOR, DESK),
        seg(
            minutes * 60.0,
            Activity::Fidget {
                at: DESK,
                bpm: 15.0,
                every_s: 20.0,
                burst_s: 3.0,
            },
        ),
        seated(40.0 * 60.0, DESK, 13.0, None),
    ]
}

// ------------------------------------------------------------- assembly

/// Assemble a scenario with the evaluation defaults: 56 subcarriers at
/// 20 Hz, noise 2.0 (per component, against |LoS| = 40), 2% loss, a
/// 3-point x 0.3 m torso. (2 ms jitter and the 0.05 m^2 arm come from
/// `SimOptions::default()`.)
pub fn scenario(
    name: &str,
    room: Room,
    layout: Layout,
    timeline: Vec<Segment>,
    seed: u64,
) -> Scenario {
    Scenario {
        name: name.to_string(),
        room,
        nodes: layout.nodes,
        links: layout.links,
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

#[cfg(test)]
mod tests {
    use super::*;

    fn total(tl: &[Segment]) -> f64 {
        tl.iter().map(|s| s.duration_s).sum()
    }

    #[test]
    fn timelines_start_empty_long_enough() {
        for tl in [
            reader(5.0),
            night(1.0, 4.0, (10.0, 40.0)),
            collapse(),
            sit_down(),
            tachypnea(),
            lone_worker(5.0),
        ] {
            assert!(matches!(tl[0].activity, Activity::Absent));
            assert!(tl[0].duration_s >= 95.0);
        }
    }

    #[test]
    fn night_has_requested_length_and_pauses() {
        let tl = night_seeded(8.0, 4.0, (10.0, 40.0), 9);
        assert!((total(&tl) - 8.0 * 3600.0).abs() < 1e-6, "{}", total(&tl));
        let pauses: Vec<f64> = tl
            .iter()
            .filter(|s| matches!(s.activity, Activity::BreathingPause { .. }))
            .map(|s| s.duration_s)
            .collect();
        // ~4/h over ~7.9 h, thinned slightly by the minimum gap.
        assert!(
            pauses.len() > 15 && pauses.len() < 50,
            "{} pauses",
            pauses.len()
        );
        assert!(pauses.iter().all(|&d| (10.0..=40.0).contains(&d)));
        assert_ne!(
            format!("{:?}", night_seeded(8.0, 4.0, (10.0, 40.0), 10)),
            format!("{tl:?}")
        );
    }
}
