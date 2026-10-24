//! The simulator in a box (`Room::vertical`): validation, the paired
//! furniture footprint, determinism. Parity with the Python box is in
//! `parity.rs`. SYNTHETIC.

use saans_sim::scenario::{
    Activity, Room, Scatterer, Scenario, Segment, SimOptions, TruthKind, Vertical, FLOOR_CHEST_Z_M,
};
use saans_sim::scenarios;
use saans_sim::Simulator;

fn in_box(room: Room) -> Room {
    Room {
        vertical: Some(Vertical::default()),
        ..room
    }
}

fn run(room: Room, seed: u64) -> Scenario {
    scenarios::scenario(
        "box",
        room,
        scenarios::three_links(),
        scenarios::collapse(),
        seed,
    )
}

#[test]
fn a_seed_puts_the_same_furniture_in_the_slice_and_the_box() {
    let slice = Simulator::new(&run(scenarios::room_attenuated_los(), 5));
    let boxed = Simulator::new(&run(in_box(scenarios::room_attenuated_los()), 5));
    assert_eq!(slice.scatterers(), boxed.scatterers());
    assert!(slice.scatterer_z().is_empty());
    let (lo, hi) = Vertical::default().scatterer_z_m;
    assert_eq!(boxed.scatterer_z().len(), boxed.scatterers().len());
    assert!(boxed.scatterer_z().iter().all(|z| (lo..hi).contains(z)));
}

#[test]
fn the_box_changes_the_channel_and_stays_deterministic() {
    let frames = |room: Room| {
        let mut sim = Simulator::new(&run(room, 3));
        (0..200)
            .filter_map(|_| sim.step())
            .flat_map(|t| t.frames)
            .flat_map(|f| f.frame.amplitudes)
            .collect::<Vec<f64>>()
    };
    let slice = frames(scenarios::room_clear_los());
    let a = frames(in_box(scenarios::room_clear_los()));
    let b = frames(in_box(scenarios::room_clear_los()));
    assert_eq!(a, b);
    assert_eq!(a.len(), slice.len());
    assert_ne!(a, slice);
}

#[test]
fn sensitivity_stays_in_range_in_a_box() {
    let sim = Simulator::new(&run(in_box(scenarios::room_rich_nlos()), 9));
    for link in [(0, 1), (0, 2), (1, 2)] {
        for y in [1.0, 2.0, 3.0, 4.0] {
            let s = sim.sensitivity_at(link, (2.5, y));
            assert!(s.iter().all(|v| (0.0..=1.0 + 1e-12).contains(v)), "{s:?}");
        }
    }
}

#[test]
fn heights_outside_the_box_are_refused() {
    let base = || run(in_box(scenarios::room_clear_los()), 1);
    let refused = |sc: Scenario, says: &str| {
        let err = Simulator::try_new(&sc).expect_err(says);
        assert!(err.contains(says), "{err:?} should mention {says:?}");
    };

    let mut sc = base();
    sc.nodes[1].z_m = 2.7;
    refused(sc, "node 1 at height 2.7");

    let mut sc = base();
    sc.room.vertical.as_mut().unwrap().chest_z_m = 0.0;
    refused(sc, "chest height 0");

    let mut sc = base();
    sc.room.vertical.as_mut().unwrap().height_m = -1.0;
    refused(sc, "room height -1");

    let mut sc = base();
    sc.room.vertical.as_mut().unwrap().floor_reflection = 1.5;
    refused(sc, "must be in [0, 1]");

    let mut sc = base();
    sc.room.vertical.as_mut().unwrap().scatterer_z_m = (1.8, 0.3);
    refused(sc, "furniture heights");

    // A scatterer somebody placed on purpose gets no invented height.
    let mut sc = base();
    sc.room.scatterers.push(Scatterer {
        pos: (2.0, 2.0),
        rcs_m2: 0.2,
        z_m: None,
    });
    refused(sc, "has no height");

    // The slice ignores heights altogether.
    let mut sc = run(scenarios::room_clear_los(), 1);
    sc.nodes[1].z_m = 9.0;
    sc.room.scatterers.push(Scatterer {
        pos: (2.0, 2.0),
        rcs_m2: 0.2,
        z_m: None,
    });
    assert!(Simulator::try_new(&sc).is_ok());
}

#[test]
fn nodes_stacked_vertically_are_a_link_in_a_box_only() {
    let mut sc = run(in_box(scenarios::room_clear_los()), 1);
    sc.nodes[1].pos = sc.nodes[0].pos;
    sc.nodes[1].z_m = 2.2;
    assert!(Simulator::try_new(&sc).is_ok());
    sc.room.vertical = None;
    let err = Simulator::try_new(&sc).unwrap_err();
    assert!(err.contains("nodes coincide"), "{err}");
}

/// The posture the simulator's sensitivity reports, rebuilt from the
/// channel physics: upright at the room's chest height, or lying on the
/// floor breathing vertically. No random furniture and no arm, so the
/// test's own channel is the simulator's.
fn expected(sc: &Scenario, centre: (f64, f64), lying: bool) -> Vec<f64> {
    use saans_sim::physics::{LinkChannel, BODY_RCS_M2};
    let n = |id: u8| {
        let n = sc.nodes.iter().find(|n| n.id == id).unwrap();
        (n.pos.0, n.pos.1, n.z_m)
    };
    let opts = SimOptions::default();
    let ch = LinkChannel::new(
        &sc.room,
        &[],
        n(0),
        n(1),
        opts.carrier_hz,
        opts.subcarrier_spacing_hz,
        sc.n_subcarriers,
    );
    let z = match (lying, &sc.room.vertical) {
        (true, Some(_)) => FLOOR_CHEST_Z_M,
        (_, Some(v)) => v.chest_z_m,
        (_, None) => 0.0,
    };
    let mut pts = Vec::new();
    ch.torso_posed(
        centre,
        z,
        lying,
        sc.body_points,
        sc.body_width_m,
        BODY_RCS_M2,
        &mut pts,
    );
    ch.sensitivity(&pts, &[])
}

fn falling(room: Room, after: Vec<Segment>) -> (Scenario, SimOptions) {
    let mut timeline = scenarios::collapse();
    timeline.extend(after);
    let mut sc = scenarios::scenario("fall", room, scenarios::one_link(), timeline, 4);
    sc.room.n_scatterers = 0;
    (
        sc,
        SimOptions {
            arm_rcs_m2: 0.0,
            ..SimOptions::default()
        },
    )
}

/// Step to `t_s` and return the sensitivity and where the body is.
fn at(sim: &mut Simulator, t_s: f64) -> (Vec<f64>, (f64, f64)) {
    while sim.step().is_some_and(|t| t.truth.t_s < t_s) {}
    let t = sim.ticks_done() as f64 / sim.scenario().sample_rate_hz;
    (sim.sensitivity((0, 1)), sim.position_at(t))
}

#[test]
fn a_fall_puts_the_chest_on_the_floor_in_a_box() {
    let (sc, opts) = falling(in_box(scenarios::room_clear_los()), vec![]);
    let mut sim = Simulator::with_options(&sc, opts);
    let fall = sim
        .truth_events()
        .into_iter()
        .find(|e| e.kind == TruthKind::Fall)
        .expect("a fall");

    let (before, pos) = at(&mut sim, fall.t_start_s - 1.0);
    assert_eq!(before, expected(&sc, pos, false), "upright before the fall");
    let (after, pos) = at(&mut sim, fall.t_end_s + 5.0);
    assert_eq!(after, expected(&sc, pos, true), "on the floor after it");
    assert_ne!(after, expected(&sc, pos, false));
}

#[test]
fn the_slice_has_no_floor_to_fall_to() {
    let (sc, opts) = falling(scenarios::room_clear_los(), vec![]);
    let mut sim = Simulator::with_options(&sc, opts);
    let end = sim.duration_s() - 1.0;
    let (after, pos) = at(&mut sim, end);
    assert_eq!(after, expected(&sc, pos, false));
}

#[test]
fn getting_up_to_walk_ends_lying() {
    let (sc, opts) = falling(
        in_box(scenarios::room_clear_los()),
        vec![
            Segment {
                duration_s: 20.0,
                activity: Activity::WalkTo {
                    to: (3.0, 3.5),
                    speed_mps: 0.9,
                },
            },
            Segment {
                duration_s: 30.0,
                activity: Activity::Seated {
                    at: (3.0, 3.5),
                    bpm: 15.0,
                    bpm_end: None,
                },
            },
        ],
    );
    let mut sim = Simulator::with_options(&sc, opts);
    let end = sim.duration_s() - 1.0;
    let (after, pos) = at(&mut sim, end);
    assert_eq!(after, expected(&sc, pos, false));

    // A breathing pause where they lie keeps them lying: nobody walked.
    let (sc, opts) = falling(
        in_box(scenarios::room_clear_los()),
        vec![Segment {
            duration_s: 20.0,
            activity: Activity::BreathingPause {
                at: scenarios::FALL_TO,
            },
        }],
    );
    let mut sim = Simulator::with_options(&sc, opts);
    let end = sim.duration_s() - 1.0;
    let (after, pos) = at(&mut sim, end);
    assert_eq!(after, expected(&sc, pos, true));
}

#[test]
fn a_sit_down_stays_upright() {
    let mut sc = scenarios::scenario(
        "sit",
        in_box(scenarios::room_clear_los()),
        scenarios::one_link(),
        scenarios::sit_down(),
        4,
    );
    sc.room.n_scatterers = 0;
    let opts = SimOptions {
        arm_rcs_m2: 0.0,
        ..SimOptions::default()
    };
    let mut sim = Simulator::with_options(&sc, opts);
    let sat = sim
        .truth_events()
        .into_iter()
        .find(|e| e.kind == TruthKind::SitDown)
        .expect("a sit-down");
    let (after, pos) = at(&mut sim, sat.t_end_s + 5.0);
    assert_eq!(after, expected(&sc, pos, false));
}
