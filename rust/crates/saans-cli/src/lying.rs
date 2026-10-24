//! `saans lying`: does a person on the floor after a fall vanish from
//! occupancy? SYNTHETIC.
//!
//! `saans eval --room-height` found one collapse run (rich-NLoS, one
//! link) where the fallen person read as an empty room for most of five
//! minutes. Diagnosed, that run was neither a blind spot nor subcarriers
//! cancelling: every subcarrier agreed in sign, and the breathing signal
//! from the chest on the floor was simply ~4x weaker than from the same
//! person upright on the same spot, leaving the presence energy just under
//! its threshold. An upright chest breathes along the horizontal bisector,
//! strongest away from the link line; a lying one breathes vertically,
//! strongest under the link and weak to its side. This asks how general
//! that is.
//!
//! Paired: at each spot, with one seed (so the same furniture and the same
//! noise), a fall that ends lying there (chest at 0.2 m, breathing
//! vertically) and a heavy sit-down that ends upright there (chest at
//! 1.0 m). Both arrive from 0.7 m away. Rooms and layouts are
//! `saans eval`'s, as a 2.7 m box; the production path scores occupancy
//! from 15 s after the event for 300 s.
//!
//! Each spot also gets the ground truth the pipeline cannot see: per link,
//! the breathing modulation of |h| on every subcarrier, as a signal-to-
//! noise ratio against the simulator's noise. "Matched" combines the
//! subcarriers as well as any linear fusion could; "plain" averages them,
//! which is where opposite signs cancel. The occupancy rule needs one
//! link below three links and two of three at three, so the spot's ground
//! truth is the matched SNR of the link that rule depends on.

use crate::eval::{self, Job, Kind, Lcg, LEAD_IN_S};
use saans_sim::physics::{BodyPoint, LinkChannel, BODY_RCS_M2, SPEED_OF_LIGHT};
use saans_sim::scenario::{
    Activity, Scenario, Segment, SimOptions, TruthKind, Vertical, FLOOR_CHEST_Z_M,
};
use saans_sim::{Simulator, BREATHING_AMPLITUDE_M, MARGIN_M};
use std::f64::consts::PI;
use std::sync::Mutex;

const ROOM_HEIGHT_M: f64 = 2.7;
const APPROACH_M: f64 = 0.7;
const STILL_S: f64 = 300.0;
/// Scored from this long after the fall or sit-down ends: presence holds
/// for 10 s, and the energy window is 8 s.
const SCORE_FROM_S: f64 = 15.0;
/// A spot "mostly read empty".
const MOSTLY: f64 = 0.5;
/// Matched SNR below which the ground truth calls a spot weakly seen.
const WEAK_SNR: f64 = 3.0;

#[derive(Clone, Copy, PartialEq)]
enum Posture {
    Lying,
    Upright,
}

struct Spot {
    room: &'static str,
    layout: &'static str,
    posture: Posture,
    /// Horizontal distance from the spot to the nearest link line.
    off_link_m: f64,
    empty: f64,
    /// Read empty and not held after a fall-shaped motion: what the
    /// detectors take as an empty room.
    empty_unheld: f64,
    unknown: f64,
    /// Matched and plain-average breathing SNR of the link the occupancy
    /// rule depends on.
    matched: f64,
    plain: f64,
}

fn seg(duration_s: f64, activity: Activity) -> Segment {
    Segment {
        duration_s,
        activity,
    }
}

fn dist_to_segment(p: (f64, f64), a: (f64, f64), b: (f64, f64)) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    let u = if len2 > 0.0 {
        (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0)
    } else {
        0.0
    };
    (p.0 - a.0 - u * dx).hypot(p.1 - a.1 - u * dy)
}

/// Signed breathing modulation of |h| per subcarrier, in |h| units: the
/// first-order change in amplitude for the breathing amplitude, the
/// simulator's own `sensitivity` before it is normalised away.
fn modulation(ch: &LinkChannel, pts: &[BodyPoint]) -> Vec<f64> {
    (0..ch.n_subcarriers())
        .map(|k| {
            let f = ch.frequency(k);
            let (mut hr, mut hi) = (ch.static_re[k], ch.static_im[k]);
            let (mut dr, mut di) = (0.0, 0.0);
            for p in pts {
                let a = ch.scale * p.amp;
                let th = -2.0 * PI * f * p.len / SPEED_OF_LIGHT;
                let (br, bi) = (a * th.cos(), a * th.sin());
                hr += br;
                hi += bi;
                dr -= p.gain * bi;
                di += p.gain * br;
            }
            let k_wave = 2.0 * PI * f / SPEED_OF_LIGHT;
            BREATHING_AMPLITUDE_M * k_wave * (hr * dr + hi * di) / hr.hypot(hi)
        })
        .collect()
}

/// (matched, plain) breathing SNR per link at `spot` in `posture`.
fn ground_truth(sc: &Scenario, spot: (f64, f64), posture: Posture) -> Vec<(f64, f64)> {
    let sim = Simulator::new(sc);
    let placed: Vec<_> = sim
        .scatterers()
        .iter()
        .zip(sim.scatterer_z())
        .map(|(&(p, rcs), &z)| ((p.0, p.1, z), rcs))
        .collect();
    let node = |id: u8| {
        let n = sc.nodes.iter().find(|n| n.id == id).unwrap();
        (n.pos.0, n.pos.1, n.z_m)
    };
    let opts = SimOptions::default();
    let v = sc.room.vertical.expect("a box");
    let (z, lying) = match posture {
        Posture::Lying => (FLOOR_CHEST_Z_M, true),
        Posture::Upright => (v.chest_z_m, false),
    };
    let sigma = sc.noise_std;
    sc.links
        .iter()
        .map(|&(tx, rx)| {
            let ch = LinkChannel::new(
                &sc.room,
                &placed,
                node(tx),
                node(rx),
                opts.carrier_hz,
                opts.subcarrier_spacing_hz,
                sc.n_subcarriers,
            );
            let mut pts = Vec::new();
            ch.torso_posed(
                spot,
                z,
                lying,
                sc.body_points,
                sc.body_width_m,
                BODY_RCS_M2,
                &mut pts,
            );
            let m = modulation(&ch, &pts);
            let matched = m.iter().map(|x| x * x).sum::<f64>().sqrt() / sigma;
            let plain = m.iter().sum::<f64>().abs() / (sigma * (m.len() as f64).sqrt());
            (matched, plain)
        })
        .collect()
}

fn scenario(
    room: &saans_sim::scenario::Room,
    nodes: &[saans_sim::scenario::Node],
    links: &[(u8, u8)],
    posture: Posture,
    spot: (f64, f64),
    rng: &mut Lcg,
    seed: u64,
) -> Scenario {
    let (w, h) = room.size_m;
    let angle = rng.range(0.0, 2.0 * PI);
    let approach = (
        (spot.0 + APPROACH_M * angle.cos()).clamp(MARGIN_M, w - MARGIN_M),
        (spot.1 + APPROACH_M * angle.sin()).clamp(MARGIN_M, h - MARGIN_M),
    );
    let bpm = rng.range(11.0, 17.0);
    let event = match posture {
        Posture::Lying => Activity::Fall { to: spot, bpm },
        Posture::Upright => Activity::SitDown { at: spot, bpm },
    };
    Scenario {
        name: "lying".into(),
        room: room.clone(),
        nodes: nodes.to_vec(),
        links: links.to_vec(),
        timeline: vec![
            seg(LEAD_IN_S, Activity::Absent),
            seg(
                15.0,
                Activity::WalkTo {
                    to: approach,
                    speed_mps: 0.9,
                },
            ),
            seg(1.0 + STILL_S, event),
        ],
        sample_rate_hz: 20.0,
        n_subcarriers: 56,
        noise_std: 2.0,
        loss_prob: 0.02,
        body_points: 3,
        body_width_m: 0.3,
        seed,
    }
}

fn measure(
    room_name: &'static str,
    room: &saans_sim::scenario::Room,
    layout: &eval::Layout,
    posture: Posture,
    i: u64,
) -> Spot {
    let (layout_name, nodes, links) = layout;
    let seed = 70_000 + i * 7919;
    let mut rng = Lcg::new(seed ^ 0x5DEE_CE66_D1CE_4E5B);
    let spot = rng.seat();
    let sc = scenario(room, nodes, links, posture, spot, &mut rng, seed);
    let job = Job {
        kind: Kind::Collapse,
        room: room_name,
        layout: layout_name,
        seed,
        scenario: sc.clone(),
        // On: the "after the hold" table measures it (it ships off).
        fall_hold_min: eval::FALL_HOLD_MIN,
    };
    let run = eval::run(&job);
    let event = run
        .truth_events
        .iter()
        .find(|e| matches!(e.kind, TruthKind::Fall | TruthKind::SitDown))
        .expect("the fall or the sit-down");
    let scored: Vec<_> = run
        .samples
        .iter()
        .zip(&run.held)
        .filter(|((t, _), _)| t.t_s >= event.t_end_s + SCORE_FROM_S)
        .collect();
    let n = scored.len().max(1) as f64;
    let empty = scored
        .iter()
        .filter(|((_, s), _)| s.occupied == Some(false))
        .count() as f64
        / n;
    let empty_unheld = scored
        .iter()
        .filter(|((_, s), &h)| s.occupied == Some(false) && !h)
        .count() as f64
        / n;
    let unknown = scored
        .iter()
        .filter(|((_, s), _)| s.occupied.is_none())
        .count() as f64
        / n;

    let mut gt = ground_truth(&sc, spot, posture);
    gt.sort_by(|a, b| b.0.total_cmp(&a.0));
    // The fusion default: one link below three, a majority from three.
    let quorum = if links.len() < 3 {
        1
    } else {
        links.len() / 2 + 1
    };
    let (matched, plain) = gt[quorum - 1];
    let pos = |id: u8| nodes.iter().find(|n| n.id == id).unwrap().pos;
    let off_link_m = links
        .iter()
        .map(|&(tx, rx)| dist_to_segment(spot, pos(tx), pos(rx)))
        .fold(f64::INFINITY, f64::min);
    Spot {
        room: room_name,
        layout: layout_name,
        posture,
        off_link_m,
        empty,
        empty_unheld,
        unknown,
        matched,
        plain,
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

pub fn run(spots: u64, threads: usize) {
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
    for (ri, _) in rooms.iter().enumerate() {
        for (li, _) in layouts.iter().enumerate() {
            for i in 0..spots {
                for posture in [Posture::Lying, Posture::Upright] {
                    work.push((ri, li, i, posture));
                }
            }
        }
    }
    let n = work.len();
    eprintln!("saans lying: {n} runs on {threads} threads");
    let queue = Mutex::new(work);
    let done: Mutex<Vec<Spot>> = Mutex::new(Vec::new());
    std::thread::scope(|s| {
        for _ in 0..threads.max(1) {
            s.spawn(|| loop {
                let Some((ri, li, i, posture)) = queue.lock().unwrap().pop() else {
                    break;
                };
                let (room_name, room) = &rooms[ri];
                let spot = measure(room_name, room, &layouts[li], posture, i);
                let mut d = done.lock().unwrap();
                d.push(spot);
                if d.len().is_multiple_of(50) || d.len() == n {
                    eprintln!(
                        "  {}/{n} runs ({:.0} s)",
                        d.len(),
                        start.elapsed().as_secs_f64()
                    );
                }
            });
        }
    });
    report(&done.into_inner().unwrap(), spots, &rooms, &layouts);
}

fn report(
    all: &[Spot],
    spots: u64,
    rooms: &[(&'static str, saans_sim::scenario::Room)],
    layouts: &[eval::Layout],
) {
    let pick = |room: Option<&str>, layout: Option<&str>, p: Posture| -> Vec<&Spot> {
        all.iter()
            .filter(|s| {
                s.posture == p
                    && room.is_none_or(|r| s.room == r)
                    && layout.is_none_or(|l| s.layout == l)
            })
            .collect()
    };
    let mean = |v: &[&Spot], f: &dyn Fn(&Spot) -> f64| {
        v.iter().map(|s| f(s)).sum::<f64>() / v.len().max(1) as f64
    };
    let mostly = |v: &[&Spot]| v.iter().filter(|s| s.empty > MOSTLY).count();

    println!(
        "saans lying -- SYNTHETIC. Does a person on the floor after a fall vanish from occupancy?"
    );
    println!(
        "Rooms and layouts of `saans eval`, as a {ROOM_HEIGHT_M} m box (concrete floor and ceiling, nodes at 1.2 m)."
    );
    println!(
        "{spots} spots per row, uniform over the seating area. At each, with one seed: a fall that ends lying there"
    );
    println!(
        "(chest {FLOOR_CHEST_Z_M} m, on the back, breathing vertically) and a heavy sit-down that ends upright there"
    );
    println!(
        "(chest 1.0 m). Occupancy scored from {SCORE_FROM_S:.0} s after the event for {STILL_S:.0} s."
    );

    println!("\nREAD AS AN EMPTY ROOM (share of the still period; spots read empty more than half of it)");
    println!(
        "  {:<10} {:<8} {:>9} {:>9}   {:>12} {:>12}   {:>9} {:>9}",
        "room", "layout", "lying", "upright", "mostly, lie", "mostly, up", "SNR lie", "SNR up"
    );
    for (room, _) in rooms {
        for (layout, _, _) in layouts {
            let (l, u) = (
                pick(Some(room), Some(layout), Posture::Lying),
                pick(Some(room), Some(layout), Posture::Upright),
            );
            println!(
                "  {:<10} {:<8} {:>8.1}% {:>8.1}%   {:>12} {:>12}   {:>9.1} {:>9.1}",
                room,
                layout,
                100.0 * mean(&l, &|s| s.empty),
                100.0 * mean(&u, &|s| s.empty),
                format!("{}/{}", mostly(&l), l.len()),
                format!("{}/{}", mostly(&u), u.len()),
                median(l.iter().map(|s| s.matched).collect()),
                median(u.iter().map(|s| s.matched).collect()),
            );
        }
    }
    let (l, u) = (
        pick(None, None, Posture::Lying),
        pick(None, None, Posture::Upright),
    );
    println!(
        "  {:<19} {:>8.1}% {:>8.1}%   {:>12} {:>12}   {:>9.1} {:>9.1}",
        "all",
        100.0 * mean(&l, &|s| s.empty),
        100.0 * mean(&u, &|s| s.empty),
        format!("{}/{}", mostly(&l), l.len()),
        format!("{}/{}", mostly(&u), u.len()),
        median(l.iter().map(|s| s.matched).collect()),
        median(u.iter().map(|s| s.matched).collect()),
    );
    let unknown = mean(&all.iter().collect::<Vec<_>>(), &|s| s.unknown);
    println!(
        "  (occupancy unknown: {:.1}% of all scored samples)",
        100.0 * unknown
    );

    println!("\nAFTER THE PRESENCE HOLD (read empty and not held: what the detectors take as nobody there)");
    println!(
        "  {:<19} {:>9} {:>9}   {:>12} {:>12}",
        "", "lying", "upright", "mostly, lie", "mostly, up"
    );
    for (name, layout) in [
        ("1 link", Some("1 link")),
        ("2 links", Some("2 links")),
        ("3 links", Some("3 links")),
        ("all", None),
    ] {
        let (l, u) = (
            pick(None, layout, Posture::Lying),
            pick(None, layout, Posture::Upright),
        );
        let mostly_unheld = |v: &[&Spot]| v.iter().filter(|s| s.empty_unheld > MOSTLY).count();
        println!(
            "  {:<19} {:>8.1}% {:>8.1}%   {:>12} {:>12}",
            name,
            100.0 * mean(&l, &|s| s.empty_unheld),
            100.0 * mean(&u, &|s| s.empty_unheld),
            format!("{}/{}", mostly_unheld(&l), l.len()),
            format!("{}/{}", mostly_unheld(&u), u.len()),
        );
    }

    println!("\n1 LINK, BY HORIZONTAL DISTANCE FROM THE LINK LINE (both rooms)");
    println!(
        "  {:<12} {:>6} {:>9} {:>9} {:>9} {:>9}",
        "distance", "spots", "lying", "upright", "SNR lie", "SNR up"
    );
    for (lo, hi) in [(0.0, 0.5), (0.5, 1.0), (1.0, 1.5), (1.5, f64::INFINITY)] {
        let band = |p| -> Vec<&Spot> {
            pick(None, Some("1 link"), p)
                .into_iter()
                .filter(|s| s.off_link_m >= lo && s.off_link_m < hi)
                .collect()
        };
        let (l, u) = (band(Posture::Lying), band(Posture::Upright));
        let label = if hi.is_finite() {
            format!("{lo:.1}-{hi:.1} m")
        } else {
            format!(">= {lo:.1} m")
        };
        println!(
            "  {:<12} {:>6} {:>8.1}% {:>8.1}% {:>9.1} {:>9.1}",
            label,
            l.len(),
            100.0 * mean(&l, &|s| s.empty),
            100.0 * mean(&u, &|s| s.empty),
            median(l.iter().map(|s| s.matched).collect()),
            median(u.iter().map(|s| s.matched).collect()),
        );
    }

    println!("\nGROUND TRUTH AGAINST OUTCOME (all rooms and layouts)");
    for p in [Posture::Lying, Posture::Upright] {
        let v = pick(None, None, p);
        let (weak, strong): (Vec<&Spot>, Vec<&Spot>) = v.iter().partition(|s| s.matched < WEAK_SNR);
        let cancelled = v.iter().filter(|s| s.plain < 0.5 * s.matched).count();
        println!(
            "  {:<8} matched SNR < {WEAK_SNR:.0}: {:>3} spots, {:>3} mostly empty; >= {WEAK_SNR:.0}: {:>3} spots, {:>3} mostly empty; plain average under half of matched: {} spots",
            if p == Posture::Lying { "lying" } else { "upright" },
            weak.len(),
            mostly(&weak),
            strong.len(),
            mostly(&strong),
            cancelled,
        );
    }

    println!();
    println!("SNR: breathing modulation of |h| against the simulated noise (std 2.0 per component vs |LoS| 40),");
    println!("for the link the occupancy rule depends on (best link below three links; second best at three).");
    println!("Matched = the best any linear subcarrier fusion could do; the pipeline's own fusion weights are");
    println!("cumulative variances and can do worse. SYNTHETIC: a model of a body on a model of a floor; the");
    println!("floor bounce of the body's own echo is not modelled, and real floors are not concrete half-spaces.");
}

#[cfg(test)]
mod tests {
    use super::*;
    use saans_sim::scenario::Room;

    #[test]
    fn distance_to_a_link_line() {
        let (a, b) = ((1.0, 2.0), (5.0, 2.0));
        assert!((dist_to_segment((3.0, 3.5), a, b) - 1.5).abs() < 1e-12);
        assert!((dist_to_segment((0.0, 2.0), a, b) - 1.0).abs() < 1e-12);
        assert!((dist_to_segment((6.0, 5.0), a, b) - 10f64.sqrt()).abs() < 1e-12);
    }

    /// The ground truth is the simulator's own `sensitivity` before it is
    /// normalised: |m_k| = sensitivity_k * A * k_f * sum |b_i| |g_i|, for
    /// both postures, a negative gain included.
    #[test]
    fn modulation_is_the_simulators_sensitivity_unnormalised() {
        let room = Room {
            vertical: Some(Vertical::default()),
            ..Room::default()
        };
        let ch = LinkChannel::new(
            &room,
            &[((2.0, 4.0, 0.8), 0.3)],
            (1.5, 2.5, 1.2),
            (4.5, 2.5, 1.2),
            2.437e9,
            312_500.0,
            56,
        );
        for (z, lying) in [(1.0, false), (0.2, true), (2.0, true)] {
            let mut pts = Vec::new();
            ch.torso_posed((2.7, 3.6), z, lying, 3, 0.3, BODY_RCS_M2, &mut pts);
            let m = modulation(&ch, &pts);
            let s = ch.sensitivity(&pts, &[]);
            for k in 0..56 {
                let f = ch.frequency(k);
                let bound: f64 = pts
                    .iter()
                    .map(|p| ch.scale * p.amp * p.gain.abs())
                    .sum::<f64>()
                    * BREATHING_AMPLITUDE_M
                    * 2.0
                    * PI
                    * f
                    / SPEED_OF_LIGHT;
                assert!(
                    (m[k].abs() - s[k] * bound).abs() <= 1e-12 * bound,
                    "z {z} k {k}: {} vs {}",
                    m[k].abs(),
                    s[k] * bound
                );
            }
        }
    }
}
