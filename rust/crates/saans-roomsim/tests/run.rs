//! Runs of the whole simulator: RNG streams, scene and config rejections,
//! TDMA order, sequence numbers, datagrams, switches and draw sequences,
//! people placement and truth.

use saans_roomsim::datagram::{CSI_HEADER_SIZE, CSI_MAGIC};
use saans_roomsim::impair::{buffer_active, buffer_k};
use saans_roomsim::{Buffer, Config, Error, Mode, PathKind, Posture, Role, Simulator, LABEL};
use saans_scene::Scene;
use saans_sim::rng::Rng;
use std::collections::BTreeMap;
use std::path::PathBuf;

fn scenes_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes")
}

fn load(name: &str) -> Scene {
    Scene::load(scenes_dir().join(name)).unwrap()
}

fn scene(json: &str) -> Scene {
    Scene::from_json_str("test", json).unwrap()
}

fn rect(objects: &str, nodes: &str, people: &str) -> Scene {
    scene(&format!(
        r#"{{"schema_version": 1, "name": "t", "provenance": "hand_written",
            "room": {{"w_m": 6.0, "d_m": 4.0, "h_m": 2.7}},
            "objects": [{objects}], "nodes": [{nodes}], "people": [{people}]}}"#
    ))
}

const TWO_NODES: &str = r#"{"id": 0, "pos": [1.0, 2.0, 1.2]}, {"id": 1, "pos": [5.0, 2.0, 1.2]}"#;

/// §2: the RNG streams the Python port must reproduce. The literals are the
/// spec's, digit for digit.
#[test]
#[allow(clippy::excessive_precision)]
fn rng_stream_vectors() {
    let mut r = Rng::new(7);
    assert_eq!(
        [r.next_u64(), r.next_u64(), r.next_u64()],
        [
            12923355070828475994,
            5142052590334782674,
            15488392906492639638
        ]
    );
    let mut r = Rng::stream(42, 1000);
    assert_eq!(
        [r.next_u64(), r.next_u64(), r.next_u64()],
        [
            12345297164789059879,
            1259461603951423963,
            15861779893704225696
        ]
    );
    let mut r = Rng::stream(42, 1);
    assert_eq!(r.uniform(), 0.924814964327024258);
    assert_eq!(r.range(0.3, 5.7), 4.97446416539475589);
    assert_eq!(
        [r.gauss(), r.gauss(), r.gauss()],
        [
            -0.793888932810311143,
            -1.59718765083500513,
            -0.484773194911475802
        ]
    );
    assert_eq!(r.exp(3.0), 2.50672202106860942);
    assert_eq!(
        r.unit_vector(),
        (0.722861780515479668, 0.690992652833581578)
    );
}

#[test]
fn every_output_is_labelled_sim() {
    assert_eq!(LABEL, "SIM");
    assert_eq!(saans_roomsim::SPEC, "roomsim v1");
}

/// §3.1: the extra scene rejections, and anything `Scene::validate` rejects.
#[test]
fn scene_rejections() {
    // The bedroom as it was before the fix: node 1 inside the almirah.
    let mut s = load("bedroom.json");
    s.objects[1].pos = (3.5, 0.4, 1.05);
    match Simulator::new(&s, Config::default()) {
        Err(Error::SceneRule(r)) => {
            assert!(r.contains("node 1") && r.contains("objects[1]"), "{r}")
        }
        other => panic!("{other:?}"),
    }
    // Touching the box is not strictly inside.
    let s = rect(
        r#"{"type": "table", "pos": [2.0, 2.0, 0.6], "size": [2.0, 1.0, 1.2], "material": "wood"}"#,
        TWO_NODES,
        "",
    );
    Simulator::new(&s, Config::default()).unwrap();

    let s = rect(
        "",
        r#"{"id": 0, "pos": [1.0, 2.0, 1.2]}, {"id": 1, "pos": [1.0, 2.0, 1.2]}"#,
        "",
    );
    assert!(
        matches!(Simulator::new(&s, Config::default()), Err(Error::SceneRule(r)) if r.contains("same position"))
    );

    let people = vec![r#"{"start": [3.0, 2.0]}"#; 101].join(", ");
    let s = rect("", TWO_NODES, &people);
    assert!(matches!(
        Simulator::new(&s, Config::default()),
        Err(Error::SceneRule(_))
    ));
    let people = vec![r#"{"start": [3.0, 2.0]}"#; 100].join(", ");
    Simulator::new(
        &rect("", TWO_NODES, &people),
        Config {
            duration_s: 0.0,
            ..Config::default()
        },
    )
    .unwrap();

    let mut s = load("bedroom.json");
    s.nodes[0].pos = (10.0, 0.15, 1.2);
    assert!(matches!(
        Simulator::new(&s, Config::default()),
        Err(Error::Scene(_))
    ));

    // A node id outside 0..254 is the §3.2 rejection naming `nodes`.
    let s = rect(
        "",
        r#"{"id": 0, "pos": [1.0, 2.0, 1.2]}, {"id": 255, "pos": [5.0, 2.0, 1.2]}"#,
        "",
    );
    assert!(matches!(
        Simulator::new(&s, Config::default()),
        Err(Error::Config { param, reason }) if param == "nodes" && reason.contains("0..254")
    ));

    // Doors and windows count for "a node strictly inside an object's box".
    let s = rect(
        r#"{"type": "door", "pos": [1.0, 2.0, 1.0], "size": [0.2, 0.9, 2.0], "material": "wood"}"#,
        TWO_NODES,
        "",
    );
    assert!(matches!(
        Simulator::new(&s, Config::default()),
        Err(Error::SceneRule(r)) if r.contains("node 0")
    ));
}

/// The rejection of a node within a wall's thickness (§3.1), with the node
/// id and the wall named, never a person.
fn in_wall(r: Result<Simulator, Error>, node: u8, wall: &str) {
    match r {
        Err(Error::Config { param, reason }) => {
            assert_eq!(param, "nodes");
            assert!(
                reason.contains(&format!("node {node} ")) && reason.contains(wall),
                "{reason}"
            );
        }
        Err(e) => panic!("not the in-wall rejection: {e}"),
        Ok(_) => panic!("node {node} inside {wall} accepted"),
    }
}

/// §3.1: a node within a wall's thickness, `pdist(node.xy, A, B) < t/2 +
/// 0.01` with `node.z <= h`, is rejected naming `nodes`; implied walls
/// included, with their configured thickness.
#[test]
fn a_node_within_a_wall_is_rejected() {
    // The bedroom's brick walls are 0.23 m: their faces are 0.115 m from
    // the centrelines, and a node needs 0.125 m.
    let mut s = load("bedroom.json");
    s.nodes[0].pos = (0.15, 0.12, 1.2);
    in_wall(Simulator::new(&s, Config::default()), 0, "walls[0]");
    s.nodes[0].pos = (0.15, 0.13, 1.2);
    Simulator::new(&s, Config::default()).unwrap();
    // Wall 3 is 0.1 m drywall: 0.06 m is needed (0.05 + 0.01 rounds a
    // little above 0.06, so 0.06 itself is refused). The node is inside it
    // without being anywhere near the scene's own 1 cm footprint margin.
    s.nodes[0].pos = (0.055, 0.15, 1.2);
    in_wall(Simulator::new(&s, Config::default()), 0, "walls[3]");
    s.nodes[0].pos = (0.061, 0.15, 1.2);
    Simulator::new(&s, Config::default()).unwrap();
    // A node of the review's in-wall scene: 10 cm deep in the brick.
    let mut s = load("bedroom.json");
    s.nodes[1].pos = (4.15, 0.05, 1.2);
    in_wall(Simulator::new(&s, Config::default()), 1, "walls[");

    // Implied walls: the footprint's edges with implied_wall_thickness_m.
    let s = rect(
        "",
        r#"{"id": 0, "pos": [0.05, 2.0, 1.2]}, {"id": 1, "pos": [5.0, 2.0, 1.2]}"#,
        "",
    );
    in_wall(Simulator::new(&s, Config::default()), 0, "implied wall 3");
    Simulator::new(
        &s,
        Config {
            implied_wall_thickness_m: 0.05,
            duration_s: 0.0,
            ..Config::default()
        },
    )
    .unwrap();

    // Above a low wall (z > h) a node may stand over its centreline.
    let low_wall = |z: f64| {
        scene(&format!(
            r#"{{"schema_version": 1, "name": "t", "provenance": "hand_written",
                "room": {{"w_m": 6.0, "d_m": 4.0, "h_m": 2.7}},
                "walls": [
                  {{"from": [0.0, 0.0], "to": [6.0, 0.0], "material": "brick", "thickness_m": 0.2}},
                  {{"from": [6.0, 0.0], "to": [6.0, 4.0], "material": "brick", "thickness_m": 0.2}},
                  {{"from": [6.0, 4.0], "to": [0.0, 4.0], "material": "brick", "thickness_m": 0.2}},
                  {{"from": [0.0, 4.0], "to": [0.0, 0.0], "material": "brick", "thickness_m": 0.2}},
                  {{"from": [3.0, 0.0], "to": [3.0, 2.0], "material": "drywall", "thickness_m": 0.1, "height_m": 1.0}}
                ],
                "nodes": [{{"id": 0, "pos": [3.0, 1.0, {z}]}}, {{"id": 1, "pos": [5.0, 2.0, 1.2]}}]}}"#
        ))
    };
    Simulator::new(
        &low_wall(1.2),
        Config {
            duration_s: 0.0,
            ..Config::default()
        },
    )
    .unwrap();
    in_wall(
        Simulator::new(&low_wall(1.0), Config::default()),
        0,
        "walls[4]",
    );
}

/// §3.2: a run needs at least one link, a transmitter and a different
/// receiver; a one-node scene has none.
#[test]
fn a_run_needs_a_link() {
    let one = rect("", r#"{"id": 0, "pos": [1.0, 2.0, 1.2]}"#, "");
    assert_eq!(
        config_param(Simulator::new(&one, Config::default())),
        "roles"
    );
    for role in [Role::Txrx, Role::Tx, Role::Rx] {
        let c = Config {
            roles: [(0, role)].into(),
            ..Config::default()
        };
        assert_eq!(config_param(Simulator::new(&one, c)), "roles");
    }
    let two = rect("", TWO_NODES, "");
    for (r0, r1, ok) in [
        (Role::Tx, Role::Tx, false),
        (Role::Rx, Role::Rx, false),
        (Role::Tx, Role::Rx, true),
        (Role::Rx, Role::Tx, true),
        (Role::Txrx, Role::Tx, true),
        (Role::Rx, Role::Txrx, true),
    ] {
        let c = Config {
            roles: [(0, r0), (1, r1)].into(),
            duration_s: 0.0,
            ..Config::default()
        };
        let r = Simulator::new(&two, c);
        if ok {
            assert_eq!(r.unwrap().links().len(), 1, "{r0:?} {r1:?}");
        } else {
            assert_eq!(config_param(r), "roles", "{r0:?} {r1:?}");
        }
    }
}

/// §3.2 through the JSON a run is configured with: non-canonical integer
/// keys name their map; out-of-range decibel and scale values name the
/// parameter.
#[test]
fn json_config_rejections_name_the_parameter() {
    for (json, param) in [
        (r#"{"roles": {"01": "tx"}}"#, "roles"),
        (r#"{"roles": {"+1": "tx"}}"#, "roles"),
        (r#"{"tx_period": {"1_0": 2}}"#, "tx_period"),
        (r#"{"people_mode": {" 0": "walk"}}"#, "people_mode"),
        (r#"{"people_posture": {"00": "stand"}}"#, "people_posture"),
        (r#"{"start_heading_rad": {"+0": 0.5}}"#, "start_heading_rad"),
        (r#"{"tail_power_db": 4000}"#, "tail_power_db"),
        (r#"{"tx_power_dbm": -7000}"#, "tx_power_dbm"),
        (r#"{"agc_initial_db": 7000}"#, "agc_initial_db"),
        (r#"{"device_gain_db": [-2, 4000]}"#, "device_gain_db"),
        (r#"{"ripple_period_bins": 1e-310}"#, "ripple_period_bins"),
        (r#"{"target_counts": 0}"#, "target_counts"),
    ] {
        match Config::from_json_str(json) {
            Err(Error::Config { param: p, .. }) => assert_eq!(p, param, "{json}"),
            other => panic!("{json}: {other:?}"),
        }
    }
}

fn config_param(r: Result<Simulator, Error>) -> String {
    match r {
        Err(Error::Config { param, .. }) => param,
        Err(e) => panic!("not a config error: {e}"),
        Ok(_) => panic!("accepted"),
    }
}

/// §3.2: the validations that need the scene.
#[test]
fn scene_dependent_config_validation() {
    let s = load("bedroom.json");
    let c = Config {
        roles: [(7, Role::Tx)].into(),
        ..Config::default()
    };
    assert_eq!(config_param(Simulator::new(&s, c)), "roles");
    let c = Config {
        tx_period: [(9, 2)].into(),
        ..Config::default()
    };
    assert_eq!(config_param(Simulator::new(&s, c)), "tx_period");
    let c = Config {
        roles: (0..4).map(|i| (i, Role::Rx)).collect(),
        ..Config::default()
    };
    assert_eq!(config_param(Simulator::new(&s, c)), "roles");
    let c = Config {
        roles: (0..4).map(|i| (i, Role::Tx)).collect(),
        ..Config::default()
    };
    assert_eq!(config_param(Simulator::new(&s, c)), "roles");
    let c = Config {
        channel: 14,
        ..Config::default()
    };
    assert_eq!(config_param(Simulator::new(&s, c)), "channel");
    // Keys must name what exists: people, and "tx,rx" pairs of nodes.
    for (param, c) in [
        (
            "people_mode",
            Config {
                people_mode: [(1, Mode::Walk)].into(),
                ..Config::default()
            },
        ),
        (
            "people_posture",
            Config {
                people_posture: [(3, Posture::Sit)].into(),
                ..Config::default()
            },
        ),
        (
            "start_heading_rad",
            Config {
                start_heading_rad: [(1, 0.5)].into(),
                ..Config::default()
            },
        ),
        (
            "loss_prob_link",
            Config {
                loss_prob_link: [("0,9".to_string(), 0.5)].into(),
                ..Config::default()
            },
        ),
        (
            "loss_prob_link",
            Config {
                loss_prob_link: [("0, 1".to_string(), 0.5)].into(),
                ..Config::default()
            },
        ),
        (
            "loss_prob_link",
            Config {
                loss_prob_link: [("00,1".to_string(), 0.5)].into(),
                ..Config::default()
            },
        ),
    ] {
        assert_eq!(config_param(Simulator::new(&s, c)), param);
    }
    // With `n_people` the indices of the people it adds are valid keys.
    Simulator::new(
        &s,
        Config {
            n_people: Some(2),
            start_heading_rad: [(1, 0.5)].into(),
            loss_prob_link: [("0,1".to_string(), 0.5)].into(),
            duration_s: 0.0,
            ..Config::default()
        },
    )
    .unwrap();

    // A walking person must start somewhere walkable.
    let c = Config {
        people_mode: [(0, Mode::Walk)].into(),
        ..Config::default()
    };
    assert!(matches!(
        Simulator::new(&s, c),
        Err(Error::Person { index: 0, .. })
    ));
    // A walking person always stands.
    let c = Config {
        people_posture: [(0, Posture::Sit)].into(),
        ..Config::default()
    };
    let hall = load("hall-l-shaped.json");
    assert!(matches!(
        Simulator::new(&hall, c),
        Err(Error::Person { index: 0, .. })
    ));
    // `stand` is what a walking person does anyway: accepted, no effect.
    let c = Config {
        people_posture: [(0, Posture::Stand)].into(),
        duration_s: 0.0,
        ..Config::default()
    };
    let sim = Simulator::new(&hall, c).unwrap();
    assert_eq!(
        (sim.people()[0].mode, sim.people()[0].posture),
        (Mode::Walk, Posture::Stand)
    );
    // Lying needs a bed: nothing under the start...
    let c = Config {
        people_mode: [(0, Mode::Stationary)].into(),
        people_posture: [(0, Posture::Lie)].into(),
        ..Config::default()
    };
    assert!(matches!(
        Simulator::new(&hall, c),
        Err(Error::Person { index: 0, .. })
    ));
    // ... and an object that is not a bed (a chair) is not enough either.
    let chair = rect(
        r#"{"type": "chair", "pos": [3.0, 2.0, 0.45], "size": [0.5, 0.5, 0.9], "material": "wood"}"#,
        TWO_NODES,
        r#"{"start": [3.0, 2.0]}"#,
    );
    let c = Config {
        people_posture: [(0, Posture::Lie)].into(),
        ..Config::default()
    };
    match Simulator::new(&chair, c) {
        Err(e @ Error::Person { index: 0, .. }) => {
            assert!(e.to_string().contains("people_posture"), "{e}")
        }
        other => panic!("{other:?}"),
    }
    let e = Simulator::new(
        &s,
        Config {
            people_mode: [(0, Mode::Walk)].into(),
            ..Config::default()
        },
    )
    .unwrap_err();
    assert!(e.to_string().starts_with("people[0]"), "{e}");
}

/// §10: TDMA order, `tx_period`, roles, per-link rate and `duration_s`.
#[test]
fn tdma_schedule() {
    let s = load("bedroom.json");
    let sim = Simulator::new(
        &s,
        Config {
            duration_s: 1.0,
            ..Config::default()
        },
    )
    .unwrap();
    assert_eq!(sim.links().len(), 12);
    let mut last = None;
    let mut n_tx = 0;
    for tx in sim.run() {
        assert!(tx.t < 1.0);
        assert_eq!(tx.t, (tx.m * 4 + tx.slot as u64) as f64 * 0.0125);
        assert_eq!(tx.tx, sim.transmitters()[tx.slot]);
        if let Some(prev) = last {
            assert!((tx.m, tx.slot) > prev);
        }
        last = Some((tx.m, tx.slot));
        let rxs: Vec<u8> = tx.packets.iter().map(|p| p.rx).collect();
        let want: Vec<u8> = (0..4).filter(|&r| r != tx.tx).collect();
        assert_eq!(rxs, want, "receivers in id order, never the transmitter");
        for p in &tx.packets {
            assert_eq!(p.link_index, sim.link_index(p.tx, p.rx).unwrap());
            assert_eq!(p.t, tx.t);
        }
        n_tx += 1;
    }
    assert_eq!(n_tx, 80);
    assert_eq!(sim.per_link_rate_hz(0), 20.0);

    // One transmitter, the rest receivers: today's firmware.
    let roles: BTreeMap<u32, Role> =
        [(0, Role::Tx), (1, Role::Rx), (2, Role::Rx), (3, Role::Rx)].into();
    let sim = Simulator::new(
        &s,
        Config {
            duration_s: 0.1,
            roles,
            ..Config::default()
        },
    )
    .unwrap();
    assert_eq!(sim.transmitters(), &[0]);
    assert_eq!(
        sim.links().iter().map(|l| (l.tx, l.rx)).collect::<Vec<_>>(),
        vec![(0, 1), (0, 2), (0, 3)]
    );
    assert_eq!(sim.per_link_rate_hz(0), 80.0);
    assert_eq!(sim.run().count(), 8);

    // tx_period: node 1 transmits every other superframe.
    let sim = Simulator::new(
        &s,
        Config {
            duration_s: 1.0,
            tx_period: [(1, 2)].into(),
            ..Config::default()
        },
    )
    .unwrap();
    let txs: Vec<_> = sim.run().collect();
    assert!(txs.iter().filter(|t| t.tx == 1).all(|t| t.m % 2 == 0));
    assert_eq!(txs.iter().filter(|t| t.tx == 1).count(), 10);
    assert_eq!(txs.iter().filter(|t| t.tx == 0).count(), 20);
    assert_eq!(sim.per_link_rate_hz(sim.link_index(1, 0).unwrap()), 10.0);
}

/// §12.4: per receiving node, 0 first, +1 for every packet it receives
/// from any transmitter; a lost packet leaves no gap and carries no seq.
#[test]
fn sequence_numbers_are_per_receiver() {
    let s = load("bedroom.json");
    let cfg = Config {
        loss_prob: 0.1,
        ..Config::default()
    };
    let sim = Simulator::new(&s, cfg).unwrap();
    let mut next = [0u32; 4];
    let (mut lost, mut got) = (0, 0);
    for p in sim.packets() {
        match &p.received {
            None => {
                assert!(p.lost);
                lost += 1;
            }
            Some(r) => {
                assert!(!p.lost);
                assert_eq!(r.seq, next[p.rx as usize]);
                next[p.rx as usize] += 1;
                got += 1;
            }
        }
    }
    assert_eq!(lost + got, 2400);
    assert!(lost > 150 && lost < 330, "{lost} lost of 2400 at 10%");
}

/// §12.2: `loss_prob_link` overrides `loss_prob` per link; with impair off
/// nothing is lost.
#[test]
fn per_link_loss() {
    let s = load("bedroom.json");
    let cfg = Config {
        duration_s: 1.0,
        loss_prob_link: [("0,1".to_string(), 1.0)].into(),
        ..Config::default()
    };
    let sim = Simulator::new(&s, cfg.clone()).unwrap();
    for p in sim.packets() {
        if (p.tx, p.rx) == (0, 1) {
            assert!(p.lost);
        }
    }
    let sim = Simulator::new(
        &s,
        Config {
            impair: false,
            ..cfg
        },
    )
    .unwrap();
    assert!(sim.packets().all(|p| !p.lost));
}

/// §12.4 and §15: every datagram of a bedroom run follows the
/// `serialize_raw` layout field by field (the gateway's `raw.rs` is not
/// linked here: it pulls in the whole gateway).
#[test]
fn datagrams_follow_the_raw_layout() {
    let s = load("bedroom.json");
    let sim = Simulator::new(
        &s,
        Config {
            duration_s: 2.0,
            ..Config::default()
        },
    )
    .unwrap();
    let mut n = 0;
    for p in sim.packets() {
        let Some(r) = &p.received else { continue };
        let d = r.datagram.as_ref().expect("quantize is on");
        assert_eq!(d.len(), CSI_HEADER_SIZE + 2 * 56);
        let u32_at = |o: usize| u32::from_le_bytes(d[o..o + 4].try_into().unwrap());
        assert_eq!(u32_at(0), CSI_MAGIC);
        assert_eq!(d[4], p.rx, "node id is the receiver");
        assert_eq!(d[5], 1, "one antenna");
        assert_eq!(u16::from_le_bytes([d[6], d[7]]), 56);
        assert_eq!(u32_at(8), 2437);
        assert_eq!(u32_at(12), r.seq);
        assert_eq!(i32::from(d[16] as i8), r.rssi);
        assert_eq!(i32::from(d[17] as i8), r.noise_floor);
        assert_eq!((d[18], d[19]), (0, 0));
        let Buffer::Int8(q) = &r.buffer else {
            panic!("int8 buffer")
        };
        for (e, &(re, im)) in q.iter().take(56).enumerate() {
            assert_eq!(d[20 + 2 * e] as i8, im, "imaginary first");
            assert_eq!(d[21 + 2 * e] as i8, re);
        }
        // Simulated nulls are exactly zero; consumers mask by index.
        for (e, &(re, im)) in q.iter().enumerate() {
            if !buffer_active(e) {
                assert_eq!((re, im), (0, 0), "entry {e} (k = {})", buffer_k(e));
            }
        }
        assert!((-100..=0).contains(&r.rssi));
        assert!((-97..=-92).contains(&r.noise_floor));
        n += 1;
    }
    assert!(n > 400, "{n} datagrams");

    let sim = Simulator::new(
        &s,
        Config {
            duration_s: 0.1,
            fw_subcarriers: 64,
            ..Config::default()
        },
    )
    .unwrap();
    let p = sim.packets().find(|p| !p.lost).unwrap();
    assert_eq!(p.received.unwrap().datagram.unwrap().len(), 20 + 128);
    let sim = Simulator::new(
        &s,
        Config {
            duration_s: 0.1,
            quantize: false,
            ..Config::default()
        },
    )
    .unwrap();
    let r = sim.packets().find(|p| !p.lost).unwrap().received.unwrap();
    assert!(r.datagram.is_none());
    assert_eq!(r.buffer, Buffer::Float(r.buffer_pre.clone()));
}

/// §2: no switch changes a draw sequence. Loss, AGC, truth, clutter and
/// device draws are identical whatever is switched off.
#[test]
fn switches_never_change_draw_sequences() {
    let s = load("hall-l-shaped.json");
    let base = Config {
        duration_s: 2.0,
        agc_jump_prob: 0.3,
        loss_prob: 0.1,
        ..Config::default()
    };
    let flipped = Config {
        ripple: false,
        cpe: false,
        snr_noise: false,
        ltf_merge: false,
        smoothing: true,
        quantize: false,
        tail: false,
        shadowing: false,
        transmission: false,
        object_specular: false,
        object_scatter: false,
        object_blockage: false,
        floor_ceiling: false,
        reflection_order: 0,
        ..base.clone()
    };
    let a = Simulator::new(&s, base.clone()).unwrap();
    let b = Simulator::new(&s, flipped).unwrap();
    let c = Simulator::new(
        &s,
        Config {
            impair: false,
            ..base.clone()
        },
    )
    .unwrap();
    assert_eq!(a.clutter(), b.clutter());
    for id in 0..8 {
        let (da, db) = (a.device(id).unwrap(), b.device(id).unwrap());
        assert_eq!(
            (&da.alpha, da.g_dev_db, da.nf_dbm, da.tau_dev),
            (&db.alpha, db.g_dev_db, db.nf_dbm, db.tau_dev)
        );
    }
    assert!(b
        .paths(0)
        .iter()
        .all(|p| p.kind == PathKind::LineOfSight || p.kind == PathKind::Clutter));
    let (ra, rb, rc): (Vec<_>, Vec<_>, Vec<_>) =
        (a.run().collect(), b.run().collect(), c.run().collect());
    assert_eq!(ra.len(), rb.len());
    assert_eq!(ra.len(), rc.len());
    let mut n_lost = 0;
    for ((x, y), z) in ra.iter().zip(&rb).zip(&rc) {
        assert_eq!(x.truth, y.truth);
        assert_eq!(x.truth, z.truth);
        for ((p, q), r) in x.packets.iter().zip(&y.packets).zip(&z.packets) {
            assert_eq!((p.lost, p.agc_db), (q.lost, q.agc_db));
            assert_eq!(p.agc_db, r.agc_db, "AGC walks with impair off too");
            assert!(!r.lost);
            assert_eq!(
                p.received.as_ref().map(|r| r.seq),
                q.received.as_ref().map(|r| r.seq)
            );
            n_lost += usize::from(p.lost);
        }
    }
    assert!(n_lost > 0);
    assert!(
        ra.iter().flat_map(|t| &t.packets).any(|p| p.agc_db != 0.0),
        "the AGC moved"
    );
}

/// A run is a stream: an absurdly long duration costs nothing up front.
#[test]
fn runs_stream() {
    let s = load("hall-l-shaped.json");
    let sim = Simulator::new(
        &s,
        Config {
            duration_s: 1e9,
            ..Config::default()
        },
    )
    .unwrap();
    let first: Vec<_> = sim.run().take(3).collect();
    assert_eq!(first.len(), 3);
    assert_eq!(first[2].t, 2.0 * 0.0125);
    assert_eq!(sim.packets().take(10).count(), 10);
}

/// §8.2 and §13: the bedroom's sleeper lies on the bed and breathes; the
/// phone she carries is on the bed; truth is one record per transmission.
#[test]
fn lying_person_and_truth() {
    let s = load("bedroom.json");
    let sim = Simulator::new(
        &s,
        Config {
            duration_s: 1.0,
            ..Config::default()
        },
    )
    .unwrap();
    let p = &sim.people()[0];
    assert_eq!(
        (p.mode, p.posture, p.on_object),
        (Mode::Stationary, Posture::Lie, Some(0))
    );
    let bed = &sim.room().objects[0];
    let pose = sim.stationary_pose(p);
    let a = sim.people_at(0.0, &[pose]);
    let b = sim.people_at(1.0, &[pose]);
    assert_eq!(a.cylinders[0].top, bed.top() + 0.25);
    for (pa, pb) in a.points[0].iter().zip(&b.points[0]) {
        assert!(pa.z >= bed.top() + 0.12 - 0.13, "on the bed: {pa:?}");
        assert_eq!((pa.x, pa.y), (pb.x, pb.y), "lying: only the vertical moves");
    }
    assert_ne!(a.points[0][0].z, b.points[0][0].z, "the chest breathes");
    assert_eq!(a.points[0][1].z, b.points[0][1].z, "the head does not");
    let txs: Vec<_> = sim.run().collect();
    assert_eq!(txs.len(), 80);
    for t in &txs {
        assert_eq!(t.truth.t, t.t);
        let pt = t.truth.people[0];
        assert_eq!(
            (pt.x, pt.y, pt.walking, pt.posture),
            (1.1, 2.6, false, Posture::Lie)
        );
        let ph = t.truth.phone.unwrap();
        assert_eq!(ph.z, bed.top() + 0.05);
    }
}

/// §8.2: a stationary person on a chair sits; their own legs are not
/// blocked by the chair; the cylinder is 0.72 of their height.
#[test]
fn sitting_person() {
    let s = rect(
        r#"{"type": "chair", "pos": [3.0, 2.0, 0.45], "size": [0.5, 0.5, 0.9], "material": "metal"}"#,
        r#"{"id": 0, "pos": [1.0, 1.0, 1.2]}, {"id": 1, "pos": [5.0, 1.0, 1.2]}"#,
        r#"{"start": [3.0, 2.0]}"#,
    );
    let sim = Simulator::new(&s, Config::default()).unwrap();
    let p = &sim.people()[0];
    assert_eq!(
        (p.mode, p.posture, p.on_object, p.resting_on()),
        (Mode::Stationary, Posture::Sit, Some(0), Some(0))
    );
    let at = sim.people_at(0.0, &[sim.stationary_pose(p)]);
    assert_eq!(at.cylinders[0].top, 0.72 * p.body.height);
    let head = at.points[0][1];
    assert!((head.z - 0.72 * (0.93 * p.body.height)).abs() < 1e-15);
}

/// §9: a person with no scene start enters at the first door.
#[test]
fn people_without_a_start_enter_at_the_door() {
    let s = load("bedroom.json");
    let sim = Simulator::new(
        &s,
        Config {
            n_people: Some(2),
            ..Config::default()
        },
    )
    .unwrap();
    let p = &sim.people()[1];
    assert!(!p.scene_start);
    assert_eq!(p.mode, Mode::Walk);
    // The door at (0.6, 0.0) is thinner along local y, so people walk in along +y.
    assert_eq!(p.start.0, 0.6);
    assert!(sim.walkability().point(p.start));
    let door = &sim.room().objects[4];
    let n = (p.start.1 / 0.05).round();
    assert_eq!(p.start.1, 0.05 * n);
    // No earlier point along the axis was walkable.
    for k in 1..(n as i32) {
        assert!(!sim.walkability().point((door.c.x, 0.05 * f64::from(k))));
    }
    let none = Simulator::new(
        &s,
        Config {
            n_people: Some(0),
            ..Config::default()
        },
    )
    .unwrap();
    assert!(none.people().is_empty());
    assert!(none
        .run()
        .all(|t| t.truth.people.is_empty() && t.truth.phone.unwrap().z == 1.0));
}

/// §6 order, §6.7 clutter, §6.8 tail.
#[test]
fn static_paths_are_in_spec_order() {
    let s = load("bedroom.json");
    let cfg = Config::default();
    let sim = Simulator::new(&s, cfg.clone()).unwrap();
    assert_eq!(sim.clutter().len(), 24);
    for c in sim.clutter() {
        assert!(saans_roomsim::geometry::inside_with_margin(
            &sim.room().footprint,
            (c.p.x, c.p.y),
            0.1
        ));
        assert!((0.0..2.7).contains(&c.p.z));
        assert!((0.005..0.2).contains(&c.sigma));
    }
    for l in sim.links() {
        let ps = sim.paths(l.index);
        let codes: Vec<u8> = ps.iter().map(|p| p.kind.code()).collect();
        assert!(
            codes.windows(2).all(|w| w[0] <= w[1]),
            "kinds in order: {codes:?}"
        );
        assert_eq!(codes[0], 0);
        assert_eq!(codes.iter().filter(|&&c| c == 5).count(), 24);
        assert_eq!(codes.iter().filter(|&&c| c == 6).count(), 40);
        let l_ref = ps
            .iter()
            .filter(|p| p.kind.code() <= 3)
            .map(|p| p.length)
            .fold(0.0, f64::max);
        for p in ps.iter().filter(|p| p.kind == PathKind::Tail) {
            assert!(p.points.is_empty() && p.length > l_ref);
        }
        // Order-2 pairs in (i, j) order.
        let pairs: Vec<_> = ps
            .iter()
            .filter(|p| p.kind == PathKind::Bounce2)
            .map(|p| (p.surfaces[0], p.surfaces[1]))
            .collect();
        assert!(pairs.windows(2).all(|w| w[0] < w[1]));
    }
    // Tail power is tail_power_db below the specular power.
    let ps = sim.paths(0);
    let spec: f64 = ps
        .iter()
        .filter(|p| p.kind.code() <= 3)
        .map(|p| saans_roomsim::numerics::cabs2(p.coeff))
        .sum();
    let tail: f64 = ps
        .iter()
        .filter(|p| p.kind == PathKind::Tail)
        .map(|p| saans_roomsim::numerics::cabs2(p.coeff))
        .sum();
    assert!((10.0 * (tail / spec).log10() - cfg.tail_power_db).abs() < 1e-9);
    // Every tail term has the same amplitude (§6.8): the exponential
    // delays alone shape the power-delay profile.
    let a_tail = ((10f64.powf(cfg.tail_power_db / 10.0) * spec) / 40.0).sqrt();
    for p in ps.iter().filter(|p| p.kind == PathKind::Tail) {
        let a = saans_roomsim::numerics::cabs2(p.coeff).sqrt();
        assert!((a - a_tail).abs() <= 1e-12 * a_tail, "{a} vs {a_tail}");
    }
    // Both directions of a link share the pair's tail draws.
    let back = sim.link_index(1, 0).unwrap();
    let tails = |i: usize| -> Vec<f64> {
        sim.paths(i)
            .iter()
            .filter(|p| p.kind == PathKind::Tail)
            .map(|p| p.coeff.re.atan2(p.coeff.im))
            .collect()
    };
    for (x, y) in tails(0).iter().zip(tails(back)) {
        assert!((x - y).abs() < 1e-12, "{x} vs {y}");
    }
}

/// §13: `buffer_k` and `buffer_active` once per run.
#[test]
fn buffer_index_tables() {
    let sim = Simulator::new(&load("bedroom.json"), Config::default()).unwrap();
    let k = sim.buffer_k();
    let want: Vec<i32> = (0..32).chain(-32..0).chain(0..32).chain(-32..0).collect();
    assert_eq!(k, want);
    let a = sim.buffer_active();
    for (e, &on) in a.iter().enumerate() {
        let lim = if e < 64 { 26 } else { 28 };
        assert_eq!(on, k[e] != 0 && k[e].abs() <= lim);
    }
}

/// Seeds change the draws; the same seed reproduces a run exactly.
#[test]
fn runs_are_deterministic() {
    let s = load("fixtures/roomsim_multi.json");
    let cfg = Config {
        duration_s: 0.5,
        tx_period: [(1, 2)].into(),
        ..Config::default()
    };
    let a: Vec<_> = Simulator::new(&s, cfg.clone()).unwrap().run().collect();
    let b: Vec<_> = Simulator::new(&s, cfg.clone()).unwrap().run().collect();
    assert_eq!(a, b);
    let c: Vec<_> = Simulator::new(&s, Config { seed: 1, ..cfg })
        .unwrap()
        .run()
        .collect();
    assert_ne!(a[0].truth, c[0].truth);
}
