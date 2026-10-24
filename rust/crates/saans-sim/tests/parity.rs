//! Parity with the Python reference's geometric scene, written by
//! `spec/golden/generate.py`: the 2-D slice (`sim_geometric`) and the box
//! (`sim_geometric_3d`, floor and ceiling with their own Gammas, nodes at
//! two heights, a chest below both). Each has explicit scatterers, a
//! 3-point torso, order-2 reflections, 3 dB LoS loss, breathing 14 BPM +
//! heart 68 BPM, 200 frames at 20 Hz, noise-free. Amplitudes, phases and
//! `sensitivity()` must agree to 1e-9 relative (SAANS_PLAN §4).

use saans_sim::physics::{LinkChannel, BODY_RCS_M2};
use saans_sim::scenario::{
    Activity, Node, Room, Scatterer, Scenario, Segment, SimOptions, Vertical, NODE_Z_M,
};
use saans_sim::Simulator;
use std::collections::HashMap;
use std::path::PathBuf;

const REL: f64 = 1e-9;
const ABS: f64 = 1e-12;

struct Vectors {
    name: String,
    kv: HashMap<String, String>,
}

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/golden")
}

impl Vectors {
    fn load(name: &str) -> Self {
        let path = golden_dir().join(format!("{name}.kv"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run spec/golden/generate.py)", path.display()));
        Self {
            name: name.to_string(),
            kv: text
                .lines()
                .filter_map(|l| l.split_once(' '))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }
    fn has(&self, k: &str) -> bool {
        self.kv.contains_key(k)
    }
    fn f(&self, k: &str) -> f64 {
        self.kv[k]
            .parse()
            .unwrap_or_else(|_| panic!("{k} not a number"))
    }
    fn a(&self, k: &str) -> Vec<f64> {
        let n: usize = self.kv[&format!("array.{k}")].parse().unwrap();
        let bytes = std::fs::read(golden_dir().join(format!("{}__{k}.f64", self.name))).unwrap();
        assert_eq!(bytes.len(), n * 8, "array {k} length");
        bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect()
    }
}

/// Max relative error, and asserts every element within REL (+ABS).
fn compare(what: &str, got: &[f64], want: &[f64]) -> f64 {
    assert_eq!(got.len(), want.len(), "{what}: length");
    let mut worst = 0.0f64;
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        let err = (g - w).abs();
        let scale = g.abs().max(w.abs());
        assert!(
            err <= REL * scale + ABS,
            "{what}[{i}]: got {g:e}, want {w:e} (err {err:e})"
        );
        if scale > 0.0 {
            worst = worst.max(err / scale);
        }
    }
    worst
}

/// The box, if this vector set is one (the reference's `room_height_m`).
fn vertical(v: &Vectors) -> Option<Vertical> {
    v.has("room_z").then(|| Vertical {
        height_m: v.f("room_z"),
        floor_reflection: v.f("floor_reflection"),
        ceiling_reflection: v.f("ceiling_reflection"),
        chest_z_m: v.f("body_z"),
        // Unused: every scatterer is explicit, with its height.
        scatterer_z_m: (0.0, 0.0),
    })
}

/// Node height; ignored in the slice, so any value will do there.
fn z(v: &Vectors, k: &str) -> f64 {
    if v.has(k) {
        v.f(k)
    } else {
        NODE_Z_M
    }
}

fn python_scene(v: &Vectors) -> Scenario {
    let xs = v.a("scatterer_x");
    let ys = v.a("scatterer_y");
    let zs = vertical(v).map(|_| v.a("scatterer_z"));
    let rcs = v.a("scatterer_rcs");
    let scatterers = (0..xs.len())
        .map(|i| Scatterer {
            pos: (xs[i], ys[i]),
            rcs_m2: rcs[i],
            z_m: zs.as_ref().map(|zs| zs[i]),
        })
        .collect();
    let room = Room {
        size_m: (v.f("room_w"), v.f("room_h")),
        wall_reflection: v.f("wall_reflection"),
        reflection_order: v.f("reflection_order") as u32,
        los_attenuation_db: v.f("los_attenuation_db"),
        n_scatterers: 0,
        scatterer_rcs_m2: (0.05, 0.5),
        scatterers,
        door_m: (0.6, 0.6),
        vertical: vertical(v),
    };
    let fs = v.f("fs");
    let n_frames = v.a("amplitudes").len() / v.f("n_subcarriers") as usize;
    Scenario {
        name: "python-parity".into(),
        room,
        nodes: vec![
            Node {
                id: 0,
                pos: (v.f("tx_x"), v.f("tx_y")),
                z_m: z(v, "tx_z"),
            },
            Node {
                id: 1,
                pos: (v.f("rx_x"), v.f("rx_y")),
                z_m: z(v, "rx_z"),
            },
        ],
        links: vec![(0, 1)],
        timeline: vec![Segment {
            duration_s: n_frames as f64 / fs,
            activity: Activity::Seated {
                at: (v.f("body_x"), v.f("body_y")),
                bpm: v.f("breathing_bpm"),
                bpm_end: None,
            },
        }],
        sample_rate_hz: fs,
        n_subcarriers: v.f("n_subcarriers") as usize,
        noise_std: 0.0,
        loss_prob: 0.0,
        body_points: v.f("body_points") as usize,
        body_width_m: v.f("body_width_m"),
        seed: 0,
    }
}

/// Exact-time sampling, no arm: the reference has neither.
fn python_options(v: &Vectors) -> SimOptions {
    SimOptions {
        carrier_hz: v.f("carrier_hz"),
        subcarrier_spacing_hz: v.f("subcarrier_spacing_hz"),
        jitter_s: 0.0,
        arm_rcs_m2: 0.0,
    }
}

#[test]
fn constants_match_the_reference() {
    for name in ["sim_geometric", "sim_geometric_3d"] {
        constants_match(&Vectors::load(name));
    }
}

fn constants_match(v: &Vectors) {
    assert_eq!(
        v.f("static_magnitude"),
        saans_sim::physics::STATIC_MAGNITUDE
    );
    assert_eq!(v.f("body_rcs_m2"), saans_sim::physics::BODY_RCS_M2);
    assert_eq!(
        v.f("breathing_amplitude_m"),
        saans_sim::BREATHING_AMPLITUDE_M
    );
    assert_eq!(v.f("heart_amplitude_m"), saans_sim::HEART_AMPLITUDE_M);
    assert_eq!(v.f("heart_bpm"), saans_sim::HEART_BPM);
    let d = |a: (f64, f64, f64), b: (f64, f64, f64)| (a.0 - b.0).hypot(a.1 - b.1).hypot(a.2 - b.2);
    let (tx, rx, body) = (
        (v.f("tx_x"), v.f("tx_y"), z(v, "tx_z")),
        (v.f("rx_x"), v.f("rx_y"), z(v, "rx_z")),
        (v.f("body_x"), v.f("body_y"), z(v, "body_z")),
    );
    assert!((d(tx, body) + d(body, rx) - d(tx, rx) - v.f("excess_path_m")).abs() < 1e-12);
}

#[test]
fn geometric_scene_matches_python() {
    matches_python(&Vectors::load("sim_geometric"));
}

#[test]
fn box_scene_matches_python() {
    let v = Vectors::load("sim_geometric_3d");
    matches_python(&v);

    // The torso as the link sees it: each point's horizontal chest gain,
    // which the 2-D set cannot test (level with the link it is the
    // slice's 2 cos(beta/2) by construction).
    let sc = python_scene(&v);
    let opts = python_options(&v);
    let (tx, rx) = (&sc.nodes[0], &sc.nodes[1]);
    let placed: Vec<_> = sc
        .room
        .scatterers
        .iter()
        .map(|s| ((s.pos.0, s.pos.1, s.z_m.unwrap()), s.rcs_m2))
        .collect();
    let ch = LinkChannel::new(
        &sc.room,
        &placed,
        (tx.pos.0, tx.pos.1, tx.z_m),
        (rx.pos.0, rx.pos.1, rx.z_m),
        opts.carrier_hz,
        opts.subcarrier_spacing_hz,
        sc.n_subcarriers,
    );
    let mut pts = Vec::new();
    ch.torso(
        (v.f("body_x"), v.f("body_y")),
        sc.body_points,
        sc.body_width_m,
        BODY_RCS_M2,
        &mut pts,
    );
    let gains: Vec<f64> = pts.iter().map(|p| p.gain).collect();
    let worst = compare("body_gain", &gains, &v.a("body_gain"));
    println!("box parity: max rel err body gains {worst:.2e}");
}

fn matches_python(v: &Vectors) {
    let sc = python_scene(v);
    let n = sc.n_subcarriers;
    let mut sim = Simulator::with_options(&sc, python_options(v));
    assert_eq!(sim.n_ticks() as usize * n, v.a("amplitudes").len());

    let sens = sim.sensitivity((0, 1));
    let worst_sens = compare("sensitivity", &sens, &v.a("sensitivity"));

    let (mut amps, mut phases) = (Vec::new(), Vec::new());
    while let Some(tick) = sim.step() {
        assert_eq!(tick.frames.len(), 1);
        let f = &tick.frames[0];
        assert_eq!(f.t_s, tick.truth.t_s);
        amps.extend_from_slice(&f.frame.amplitudes);
        phases.extend_from_slice(&f.frame.phases);
    }
    let worst_amp = compare("amplitudes", &amps, &v.a("amplitudes"));
    let worst_phase = compare("phases", &phases, &v.a("phases"));
    // The sensitivity is position-only: unchanged after stepping.
    assert_eq!(sim.sensitivity((0, 1)), sens);
    println!(
        "{} parity: max rel err amplitudes {worst_amp:.2e}, phases {worst_phase:.2e}, sensitivity {worst_sens:.2e}",
        v.name
    );
}
