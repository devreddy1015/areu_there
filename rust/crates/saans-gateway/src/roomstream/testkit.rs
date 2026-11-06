//! Fixtures and checks shared by the room stream's unit tests: setups of
//! the shipped scenes, engines over them, a recorded run read back from a
//! written file, and JSON walks. Every number here is SIM.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use saans_locate::phone::Config;
use saans_locate::records::{write_records, Manifest, Records};
use saans_locate::score::FalseAlarms;
use saans_locate::stream::RoomSimSetup;
use saans_locate::variant::Variant;
use saans_roomsim::Config as RoomSimConfig;
use saans_scene::Scene;
use serde_json::Value;

use super::engine::{Engine, Next, Session, Source, Start};
use super::replay::{Recorded, RoomReplays};
use super::session::empty_run;
use super::Embedded;
use crate::auth::sha256_hex;

/// A calibrated setup of a shipped scene, as a SIM session builds it,
/// with its empty run.
pub(crate) struct SimFixture {
    pub(crate) e: Embedded,
    pub(crate) scene: Scene,
    /// `seed` 0 and the duration asked for; `rs` is it with the variant.
    pub(crate) base: RoomSimConfig,
    pub(crate) rs: RoomSimConfig,
    pub(crate) variant: Variant,
    pub(crate) setup: RoomSimSetup,
    pub(crate) empty_run: Option<(FalseAlarms, FalseAlarms)>,
}

pub(crate) fn sim_fixture(id: &str, variant: Variant, duration_s: f64) -> SimFixture {
    let e = Embedded::find(id).expect("an embedded scene");
    let scene = e.load().expect("it loads");
    let base = RoomSimConfig {
        seed: 0,
        duration_s,
        ..RoomSimConfig::default()
    };
    let rs = variant.apply(&scene, &base);
    let setup = RoomSimSetup::new(&scene, &rs, &Config::default(), "SIM", &mut || true)
        .expect("a valid run")
        .expect("never abandoned");
    let empty_run = match variant {
        Variant::Empty => None,
        _ => Some(
            empty_run(&setup, &mut || true)
                .expect("the empty run")
                .expect("never abandoned"),
        ),
    };
    SimFixture {
        e,
        scene,
        base,
        rs,
        variant,
        setup,
        empty_run,
    }
}

/// A SIM engine on `fx`, its first frame due at 0.
pub(crate) fn sim_engine(fx: &SimFixture, start: Start) -> Engine<'_> {
    let session = Session::sim(
        fx.e,
        fx.scene.clone(),
        &fx.setup,
        fx.variant,
        0,
        fx.empty_run,
    );
    Engine::new(
        Source::Sim {
            setup: &fx.setup,
            stream: Some(Box::new(fx.setup.stream().unwrap())),
        },
        session,
        start,
        Duration::ZERO,
    )
    .unwrap()
}

/// `fx`'s run as a records file (§6.1): every record, with its links.
pub(crate) fn records_of(fx: &SimFixture) -> Records {
    let all = fx
        .setup
        .stream()
        .unwrap()
        .advance_to(f64::INFINITY)
        .unwrap();
    Records {
        manifest: Manifest {
            scene_name: fx.scene.name.clone(),
            scene_sha256: sha256_hex(fx.e.text),
            provenance: fx.scene.provenance,
            variant: fx.variant,
            seed: fx.rs.seed,
            duration_s: fx.rs.duration_s,
            roomsim_config: fx.rs.clone(),
            locate_config: fx.setup.config().clone(),
            links: fx.setup.links().to_vec(),
            has_phone: fx.setup.has_phone(),
            empty_run: fx.empty_run,
        },
        ticks: all.ticks,
        links: all.links,
        epochs: all.epochs,
    }
}

/// A fresh path under the system temp dir.
pub(crate) fn temp_file(name: &str) -> PathBuf {
    static N: AtomicUsize = AtomicUsize::new(0);
    std::env::temp_dir().join(format!(
        "saans-roomstream-{}-{}-{name}",
        std::process::id(),
        N.fetch_add(1, Ordering::Relaxed)
    ))
}

/// Write `records` with saans-locate's writer and load the file as
/// `--room-replay` does.
pub(crate) fn load_written(records: &Records) -> Result<RoomReplays, String> {
    let path = temp_file("run.jsonl");
    let mut file = std::io::BufWriter::new(std::fs::File::create(&path).unwrap());
    write_records(&mut file, records).unwrap();
    drop(file);
    let loaded = RoomReplays::load(std::slice::from_ref(&path));
    let _ = std::fs::remove_file(&path);
    loaded
}

/// `fx`'s run, written to a file and read back as a recorded run.
pub(crate) fn recorded_from_file(fx: &SimFixture) -> Arc<Recorded> {
    let replays = load_written(&records_of(fx)).unwrap_or_else(|e| panic!("{e}"));
    replays.recorded(fx.e).expect("the file covers its scene")
}

/// The bedroom, `with_walker`, 20 s, as a recorded run (built once).
pub(crate) fn bedroom_recorded() -> Arc<Recorded> {
    static RECORDED: OnceLock<Arc<Recorded>> = OnceLock::new();
    Arc::clone(
        RECORDED
            .get_or_init(|| recorded_from_file(&sim_fixture("bedroom", Variant::WithWalker, 20.0))),
    )
}

/// A REPLAY engine on [`bedroom_recorded`], its first frame due at 0.
pub(crate) fn replay_engine(start: Start) -> Engine<'static> {
    let recorded = bedroom_recorded();
    let session = Session::replay(&recorded).unwrap();
    Engine::new(
        Source::Replay {
            recorded,
            tick: 0,
            epoch: 0,
        },
        session,
        start,
        Duration::ZERO,
    )
    .unwrap()
}

/// The next frame, moving `now` to when it is due; panics on a close.
pub(crate) fn next_frame(e: &mut Engine<'_>, now: &mut Duration) -> Value {
    loop {
        match e.next(*now) {
            Next::Frame(f) => return f,
            Next::Wait(d) => *now = d,
            Next::Close { message, code } => panic!("closed {code}: {message}"),
        }
    }
}

/// Frames from `now` up to and including the one that says `ended`, and
/// the time it was sent.
pub(crate) fn play_to_end(e: &mut Engine<'_>, mut now: Duration) -> (Vec<Value>, Duration) {
    let mut frames = Vec::new();
    loop {
        let f = next_frame(e, &mut now);
        let ended = f["ended"] == true;
        frames.push(f);
        if ended {
            return (frames, now);
        }
        assert!(frames.len() < 100_000, "the run never ended");
    }
}

/// The quanta of a session time `t` (which is `float(2*q) * 0.0125`).
pub(crate) fn q_of(t: f64) -> u64 {
    (t / 0.025).round() as u64
}

/// The JSON path of every object key named `key`.
pub(crate) fn key_paths(v: &Value, key: &str) -> Vec<String> {
    fn walk(v: &Value, path: &str, key: &str, out: &mut Vec<String>) {
        match v {
            Value::Object(m) => {
                for (k, x) in m {
                    let p = format!("{path}.{k}");
                    if k == key {
                        out.push(p.clone());
                    }
                    walk(x, &p, key, out);
                }
            }
            Value::Array(a) => {
                for (i, x) in a.iter().enumerate() {
                    walk(x, &format!("{path}[{i}]"), key, out);
                }
            }
            _ => {}
        }
    }
    let mut out = Vec::new();
    walk(v, "$", key, &mut out);
    out
}

/// Outside `truth`, every object with an `x` is an estimate with its
/// `radius_m` (§4: no position without its radius), and nothing is named
/// `blocked`. (A link item's `y` is a measurement, not a coordinate.)
pub(crate) fn assert_no_truth_outside_truth(frame: &Value) {
    fn walk(v: &Value, path: &str) {
        match v {
            Value::Object(m) => {
                if m.contains_key("x") {
                    assert!(
                        m.contains_key("radius_m"),
                        "{path}: a position without its radius"
                    );
                }
                assert!(!m.contains_key("blocked"), "{path}: blocked outside truth");
                for (k, x) in m {
                    walk(x, &format!("{path}.{k}"));
                }
            }
            Value::Array(a) => a
                .iter()
                .enumerate()
                .for_each(|(i, x)| walk(x, &format!("{path}[{i}]"))),
            _ => {}
        }
    }
    let mut outside = frame.clone();
    if let Some(m) = outside.as_object_mut() {
        m.remove("truth");
    }
    walk(&outside, "$");
}

/// Every `Person.id` and the `Phone.id` of an embedded scene's file.
pub(crate) fn scene_ids(e: Embedded) -> Vec<String> {
    let raw: Value = serde_json::from_str(e.text).unwrap();
    let mut ids: Vec<String> = raw["people"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|p| p["id"].as_str().map(String::from))
        .collect();
    ids.extend(raw["phone"]["id"].as_str().map(String::from));
    ids
}

/// No string value anywhere in `v` equals one of `ids` (keys are not
/// values: "phone" is a frame key and the shipped phone's id).
pub(crate) fn assert_no_string_value(v: &Value, ids: &[String], what: &str) {
    match v {
        Value::String(s) => assert!(!ids.contains(s), "{what}: an id was sent as a value: {v}"),
        Value::Array(a) => a.iter().for_each(|x| assert_no_string_value(x, ids, what)),
        Value::Object(m) => m
            .values()
            .for_each(|x| assert_no_string_value(x, ids, what)),
        _ => {}
    }
}
