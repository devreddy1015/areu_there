//! The simulator in a box (`Room::vertical`): validation, the paired
//! furniture footprint, determinism. Parity with the Python box is in
//! `parity.rs`. SYNTHETIC.

use saans_sim::scenario::{Room, Scatterer, Scenario, Vertical};
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
