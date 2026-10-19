//! Parity with the Python reference's geometric scene
//! (`spec/golden/sim_geometric`, written by `spec/golden/generate.py`):
//! the same geometry, explicit scatterers, a 3-point torso, order-2
//! reflections, 3 dB LoS loss, breathing 14 BPM + heart 68 BPM, 200 frames
//! at 20 Hz, noise-free. Amplitudes, phases and `sensitivity()` must agree
//! to 1e-9 relative (SAANS_PLAN §4).

use saans_sim::scenario::{Activity, Node, Room, Scatterer, Scenario, Segment, SimOptions};
use saans_sim::Simulator;
use std::collections::HashMap;
use std::path::PathBuf;

const REL: f64 = 1e-9;
const ABS: f64 = 1e-12;

struct Vectors {
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
            kv: text
                .lines()
                .filter_map(|l| l.split_once(' '))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        }
    }
    fn f(&self, k: &str) -> f64 {
        self.kv[k]
            .parse()
            .unwrap_or_else(|_| panic!("{k} not a number"))
    }
    fn a(&self, k: &str) -> Vec<f64> {
        let n: usize = self.kv[&format!("array.{k}")].parse().unwrap();
        let bytes = std::fs::read(golden_dir().join(format!("sim_geometric__{k}.f64"))).unwrap();
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

fn python_scene(v: &Vectors) -> Scenario {
    let xs = v.a("scatterer_x");
    let ys = v.a("scatterer_y");
    let rcs = v.a("scatterer_rcs");
    let scatterers = (0..xs.len())
        .map(|i| Scatterer {
            pos: (xs[i], ys[i]),
            rcs_m2: rcs[i],
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
            },
            Node {
                id: 1,
                pos: (v.f("rx_x"), v.f("rx_y")),
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
    let v = Vectors::load("sim_geometric");
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
    let d = |a: (f64, f64), b: (f64, f64)| (a.0 - b.0).hypot(a.1 - b.1);
    let (tx, rx, body) = (
        (v.f("tx_x"), v.f("tx_y")),
        (v.f("rx_x"), v.f("rx_y")),
        (v.f("body_x"), v.f("body_y")),
    );
    assert!((d(tx, body) + d(body, rx) - d(tx, rx) - v.f("excess_path_m")).abs() < 1e-12);
}

#[test]
fn geometric_scene_matches_python() {
    let v = Vectors::load("sim_geometric");
    let sc = python_scene(&v);
    let n = sc.n_subcarriers;
    let mut sim = Simulator::with_options(&sc, python_options(&v));
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
    println!("parity: max rel err amplitudes {worst_amp:.2e}, phases {worst_phase:.2e}, sensitivity {worst_sens:.2e}");
}
