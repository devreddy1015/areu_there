//! `saans roomsim-eval` end to end (spec/livedemo.md §10, §11 saans-cli):
//! run the binary and check its figures against the saans-locate library's
//! own run and score, its text, its refusals and the records file it
//! writes. Everything checked here is SIM.

use saans_locate::phone::Config;
use saans_locate::pipeline::{resolve_config, run_roomsim, RoomSimRun};
use saans_locate::records::{read_records, write_records, Manifest, Records, MAX_BYTES};
use saans_locate::score::{
    false_alarms, footprint_centroid, radius_scale, score, Accuracy, FalseAlarms, PersonEstimator,
    RadiusScales, Scores,
};
use saans_locate::stream::RoomSimSetup;
use saans_locate::variant::Variant;
use saans_roomsim::Config as RoomSimConfig;
use saans_scene::Scene;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::{Path, PathBuf};
use std::process::{Command, Output};

/// The run every figure test uses: short, so the dev profile stays quick.
const SECONDS: f64 = 20.0;

/// The seeds every figure test runs. For the bedroom at 20 s, seed 0's
/// empty room publishes nothing for WLM or RTI, so its false alarms are
/// all 0 and would pass in each other's place; seed 1's RTI publishes in
/// the empty room and its WLM does not. Every figure of seed 1 differs
/// from seed 0's, so `--seed` has to reach the run.
const SEEDS: [u64; 2] = [0, 1];

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn scene_path(name: &str) -> PathBuf {
    repo().join("spec/scenes").join(name)
}

fn bedroom_path() -> String {
    scene_path("bedroom.json").to_str().unwrap().to_string()
}

/// A fresh, empty directory for one test.
fn temp_dir(name: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("saans-roomsim-eval-{}-{name}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn saans(args: &[&str]) -> Output {
    Command::new(env!("CARGO_BIN_EXE_saans"))
        .args(args)
        .output()
        .expect("run saans")
}

/// `saans roomsim-eval --scene SCENE --seconds 20 --seed N --variant V`
/// plus `extra`.
fn eval(scene: &str, variant: Variant, seed: u64, extra: &[&str]) -> Output {
    let seconds = SECONDS.to_string();
    let seed = seed.to_string();
    let mut args = vec![
        "roomsim-eval",
        "--scene",
        scene,
        "--seconds",
        &seconds,
        "--seed",
        &seed,
        "--variant",
        variant.as_str(),
    ];
    args.extend_from_slice(extra);
    saans(&args)
}

fn stdout(out: &Output) -> String {
    String::from_utf8(out.stdout.clone()).expect("stdout is UTF-8")
}

fn stderr(out: &Output) -> String {
    String::from_utf8_lossy(&out.stderr).into_owned()
}

/// The single JSON object of a `--json` run: exactly one line on stdout.
fn json_of(out: &Output) -> Value {
    assert!(out.status.success(), "{}", stderr(out));
    let text = stdout(out);
    assert_eq!(text.lines().count(), 1, "one line on stdout: {text}");
    assert!(text.ends_with('\n'));
    serde_json::from_str(&text).expect("stdout is one JSON object")
}

fn bedroom() -> (Scene, String) {
    let bytes = std::fs::read(scene_path("bedroom.json")).unwrap();
    let sha = Sha256::digest(&bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let text = String::from_utf8(bytes).unwrap();
    (Scene::from_json_str("bedroom.json", &text).unwrap(), sha)
}

/// The variant's Room Sim config, as the command builds it.
fn roomsim_config(scene: &Scene, variant: Variant, seed: u64) -> RoomSimConfig {
    let base = RoomSimConfig {
        seed,
        duration_s: SECONDS,
        ..RoomSimConfig::default()
    };
    variant.apply(scene, &base)
}

fn library_run(scene: &Scene, variant: Variant, seed: u64) -> RoomSimRun {
    let rs = roomsim_config(scene, variant, seed);
    run_roomsim(scene, &rs, &Config::default(), "SIM").expect("a valid run")
}

/// A run's figures as the library gives them: `score(run_roomsim(…))`,
/// the `empty` variant's WLM and RTI false alarms (§5.3; `None` when the
/// variant is `empty`) and the radius scales (§8.2).
struct Figures {
    run: RoomSimRun,
    tick_hz: f64,
    scores: Scores,
    empty_run: Option<(FalseAlarms, FalseAlarms)>,
    scales: RadiusScales,
}

fn library(scene: &Scene, variant: Variant, seed: u64) -> Figures {
    let tick_hz = Config::default().tick_hz;
    let run = library_run(scene, variant, seed);
    let scores = score(scene, &run.ticks, &run.epochs, run.has_phone, tick_hz);
    let empty_run = (variant != Variant::Empty).then(|| {
        let empty = library_run(scene, Variant::Empty, seed);
        (
            false_alarms(&empty.ticks, PersonEstimator::Wlm, tick_hz),
            false_alarms(&empty.ticks, PersonEstimator::Rti, tick_hz),
        )
    });
    let scales = radius_scale(scene, &run.ticks, &run.epochs, run.has_phone);
    Figures {
        run,
        tick_hz,
        scores,
        empty_run,
        scales,
    }
}

/// `stats` (`Scores::to_json`, `empty_run` added to `wlm` and `rti`
/// unless the variant is `empty`) and `radius_scale`, as `--json` prints
/// them.
fn library_json(f: &Figures) -> (Value, Value) {
    let mut stats = f.scores.to_json();
    if let Some((wlm, rti)) = &f.empty_run {
        stats["wlm"]["empty_run"] = serde_json::to_value(wlm).unwrap();
        stats["rti"]["empty_run"] = serde_json::to_value(rti).unwrap();
    }
    let s = &f.scales;
    let scales = json!({"wlm": s.wlm, "rti": s.rti, "rtt": s.rtt, "rssi": s.rssi});
    (stats, scales)
}

/// `k of n<unit> (p%)`, `p` the library's fraction, or no percentage
/// when there is none.
fn of(k: usize, n: usize, unit: &str, fraction: Option<f64>) -> String {
    match fraction {
        Some(f) => format!("{k} of {n}{unit} ({:.1}%)", 100.0 * f),
        None => format!("{k} of {n}{unit}"),
    }
}

fn metres(x: Option<f64>) -> String {
    x.map_or_else(|| "null".to_string(), |v| format!("{v:.2} m"))
}

fn per_hour(x: Option<f64>) -> String {
    x.map_or_else(|| "null".to_string(), |v| format!("{v:.1} per hour"))
}

/// A `FalseAlarms` block in the command's words.
fn alarms(fa: &FalseAlarms) -> String {
    let fraction = (fa.ticks > 0).then(|| fa.published as f64 / fa.ticks as f64);
    format!(
        "{} ticks = {:.1} s; published {} = {}; {} events (runs of consecutive published ticks) = {}",
        fa.ticks,
        fa.seconds,
        of(fa.published, fa.ticks, " ticks", fraction),
        per_hour(fa.per_h),
        fa.events,
        per_hour(fa.events_per_h)
    )
}

/// The error and reference lines of `n` scored estimates; an inside count
/// is the library's fraction of `n`.
fn error_lines(
    name: &str,
    truth: &str,
    n: usize,
    own: Accuracy,
    radius_median_m: Option<f64>,
    reference: Accuracy,
) -> Vec<String> {
    if n == 0 {
        return vec![format!("{name}: no estimate scored, so no error figures")];
    }
    let inside = |a: &Accuracy| {
        let f = a.inside_radius.expect("a fraction of n > 0 scored");
        of((f * n as f64).round() as usize, n, "", Some(f))
    };
    vec![
        format!(
            "{name}: error to {truth} over {n} scored: median {}, p90 {}; truth inside the radius for {}; radius median {}",
            metres(own.median_m),
            metres(own.p90_m),
            inside(&own),
            metres(radius_median_m)
        ),
        format!(
            "{name}: reference point over the same {n}: median {}, p90 {}; truth inside the same radius for {}",
            metres(reference.median_m),
            metres(reference.p90_m),
            inside(&reference)
        ),
    ]
}

fn scale_line(name: &str, n: usize, scale: Option<f64>) -> String {
    match scale {
        Some(k) => format!(
            "{name}: suggested radius scale {k:.3} from n = {n} scored (95% of the scored errors are within scale x radius_m; SIM, reported, not applied)"
        ),
        None => format!("{name}: suggested radius scale: too few (n = {n} < 20)"),
    }
}

/// The whole text `roomsim-eval` prints for a bedroom run, line for line:
/// the command's words around the library's figures (§10), every line
/// SIM, counts first.
fn expected_text(
    scene: &Scene,
    sha: &str,
    variant: Variant,
    seed: u64,
    f: &Figures,
) -> Vec<String> {
    let (cx, cy) = footprint_centroid(scene);
    let mut want = vec![
        "saans roomsim-eval (spec/livedemo.md §10): Room Sim v1 and Locate v1 on a declared scene; simulated people and a simulated phone in simulated WiFi, a model, not a measurement".to_string(),
        format!(
            "scene {:?}: provenance hand_written, sha256 {sha}, the gateway's embedded scene bedroom",
            scene.name
        ),
        format!(
            "run: variant {}, seed {seed}, 20 s simulated, {} ticks at {} Hz, {} links; Room Sim and Locate defaults otherwise",
            variant.as_str(),
            f.run.ticks.len(),
            f.tick_hz,
            f.run.links.len()
        ),
        format!(
            "radii are uncalibrated model radii; the reference point is the footprint's area centroid ({cx:.2}, {cy:.2}) m, scored on the same records with each record's own radius: an estimator that knows nothing"
        ),
    ];
    let s = &f.scores;
    for (name, st, empty_run, scale) in [
        (
            "WLM (baseline)",
            &s.wlm,
            f.empty_run.map(|e| e.0),
            f.scales.wlm,
        ),
        ("RTI", &s.rti, f.empty_run.map(|e| e.1), f.scales.rti),
    ] {
        let (o, u) = (&st.occupied, &st.unoccupied);
        want.push(format!(
            "{name}: {} ticks: {} with a person in the truth, {} with none",
            st.ticks, o.ticks, u.ticks
        ));
        if variant == Variant::Empty {
            assert_eq!(o.ticks, 0, "an empty room has no occupied tick");
            want.push(format!(
                "{name}: false alarms, the room is empty (variant empty): {}",
                alarms(u)
            ));
        } else {
            assert_eq!(u.ticks, 0, "the bedroom's sleeper is always in the room");
            want.push(format!(
                "{name}: published {}",
                of(o.published, o.ticks, " occupied ticks", o.coverage)
            ));
            want.push(format!(
                "{name}: empty room (variant empty, same scene, seed and seconds): {}",
                alarms(&empty_run.unwrap())
            ));
            let own = Accuracy {
                median_m: o.median_m,
                p90_m: o.p90_m,
                inside_radius: o.inside_radius,
            };
            want.extend(error_lines(
                name,
                "the nearest person",
                o.published,
                own,
                o.radius_median_m,
                o.reference,
            ));
        }
        want.push(scale_line(name, o.published, scale));
    }
    for (name, p, scale) in [
        ("phone RTT", s.rtt.unwrap(), f.scales.rtt),
        ("phone RSSI", s.rssi.unwrap(), f.scales.rssi),
    ] {
        want.push(format!(
            "{name}: published {}",
            of(p.published, p.epochs, " epochs", p.coverage)
        ));
        let own = Accuracy {
            median_m: p.median_m,
            p90_m: p.p90_m,
            inside_radius: p.inside_radius,
        };
        want.extend(error_lines(
            name,
            "the phone",
            p.published,
            own,
            p.radius_median_m,
            p.reference,
        ));
        want.push(scale_line(name, p.published, scale));
    }
    want.into_iter().map(|l| format!("SIM  {l}")).collect()
}

/// Every string value in `v` (keys excluded: the shipped phone's id
/// "phone" is also a stats key).
fn string_values<'a>(v: &'a Value, out: &mut Vec<&'a str>) {
    match v {
        Value::String(s) => out.push(s),
        Value::Array(items) => items.iter().for_each(|x| string_values(x, out)),
        Value::Object(m) => m.values().for_each(|x| string_values(x, out)),
        _ => {}
    }
}

/// The scene's `Person.id`s and `Phone.id`.
fn ids(scene: &Scene) -> Vec<String> {
    let mut ids: Vec<String> = scene.people.iter().filter_map(|p| p.id.clone()).collect();
    ids.extend(scene.phone.as_ref().and_then(|p| p.id.clone()));
    ids
}

fn assert_no_id_value(v: &Value, ids: &[String]) {
    let mut strings = Vec::new();
    string_values(v, &mut strings);
    for s in strings {
        assert!(!ids.iter().any(|id| id == s), "an id is a string value");
    }
}

/// §10: `--json` prints `score(run_roomsim(…))` with the empty run's false
/// alarms, the radius scales, and the scene and run, nothing else.
#[test]
fn json_stats_are_the_library_score_plus_the_empty_run() {
    let (scene, sha) = bedroom();
    let ids = ids(&scene);
    assert_eq!(ids, ["sleeper", "phone"], "the bedroom's ids");
    for variant in [Variant::Declared, Variant::WithWalker] {
        let mut by_seed = Vec::new();
        for seed in SEEDS {
            let out = eval(&bedroom_path(), variant, seed, &["--json"]);
            assert_eq!(stderr(&out), "", "{variant:?} seed {seed}");
            let got = json_of(&out);
            let (stats, scales) = library_json(&library(&scene, variant, seed));
            assert_eq!(got["stats"], stats, "{variant:?} seed {seed}");
            assert_eq!(got["radius_scale"], scales, "{variant:?} seed {seed}");
            assert_eq!(
                got,
                json!({
                    "label": "SIM",
                    "scene": {
                        "id": "bedroom",
                        "name": scene.name,
                        "sha256": sha,
                        "provenance": "hand_written",
                    },
                    "run": {"variant": variant.as_str(), "seed": seed, "duration_s": SECONDS},
                    "stats": stats,
                    "radius_scale": scales,
                }),
                "{variant:?} seed {seed}"
            );
            // The phone is scored.
            assert!(stats["rtt"]["published"].as_u64().unwrap() > 0);
            assert_no_id_value(&got, &ids);
            by_seed.push(stats);
        }
        // Not right by default: seed 0's WLM publishes; seed 1's RTI false
        // alarms are not 0 and not WLM's; the seeds' figures differ.
        let (zero, one) = (&by_seed[0], &by_seed[1]);
        assert!(zero["wlm"]["occupied"]["published"].as_u64().unwrap() > 0);
        assert!(one["rti"]["empty_run"]["published"].as_u64().unwrap() > 0);
        assert_ne!(one["wlm"]["empty_run"], one["rti"]["empty_run"]);
        assert_ne!(zero, one, "{variant:?}: the seed changes nothing");
    }
}

/// §10: with `--variant empty` the run's own `unoccupied` false alarms
/// replace the person errors, and there is no `empty_run`.
#[test]
fn the_empty_variant_prints_false_alarms_in_place_of_errors() {
    let (scene, sha) = bedroom();
    let mut rti_published = Vec::new();
    for seed in SEEDS {
        let f = library(&scene, Variant::Empty, seed);
        let got = json_of(&eval(&bedroom_path(), Variant::Empty, seed, &["--json"]));
        let (stats, scales) = library_json(&f);
        assert_eq!(got["stats"], stats, "seed {seed}");
        assert_eq!(got["radius_scale"], scales, "seed {seed}");
        assert_eq!(
            got["run"],
            json!({"variant": "empty", "seed": seed, "duration_s": SECONDS})
        );
        for key in ["wlm", "rti"] {
            let block = got["stats"][key].as_object().unwrap();
            assert!(!block.contains_key("empty_run"), "{key}");
            assert_eq!(block["occupied"]["ticks"], 0, "{key}");
            assert_eq!(block["unoccupied"]["ticks"], block["ticks"], "{key}");
        }

        let out = eval(&bedroom_path(), Variant::Empty, seed, &[]);
        assert!(out.status.success(), "{}", stderr(&out));
        let text = stdout(&out);
        assert_eq!(
            text.lines().collect::<Vec<_>>(),
            expected_text(&scene, &sha, Variant::Empty, seed, &f),
            "seed {seed}"
        );
        for name in ["WLM (baseline)", "RTI"] {
            let alarms = format!("SIM  {name}: false alarms, the room is empty (variant empty): ");
            assert!(text.contains(&alarms), "{name}: {text}");
            assert!(
                !text.contains(&format!("{name}: error to")),
                "{name}: errors are printed for an empty room: {text}"
            );
            assert!(
                !text.contains(&format!("{name}: published")),
                "{name}: occupied coverage is printed for an empty room: {text}"
            );
        }
        assert!(!text.contains("empty room (variant empty, same scene"));
        rti_published.push(f.scores.rti.unoccupied.published);
    }
    // Seed 1's empty room has an RTI false alarm, so its line is checked on
    // figures that are not 0.
    assert!(rti_published[1] > 0, "{rti_published:?}");
}

/// §10: every line of the text is SIM, and every line is the library's
/// figures in their place: counts first, then the rates, errors and
/// scales made from them.
#[test]
fn every_text_line_is_sim_and_the_librarys_figures() {
    let (scene, sha) = bedroom();
    for seed in SEEDS {
        let out = eval(&bedroom_path(), Variant::WithWalker, seed, &[]);
        assert!(out.status.success(), "{}", stderr(&out));
        assert_eq!(stderr(&out), "");
        let text = stdout(&out);
        for line in text.lines() {
            assert!(line.starts_with("SIM  "), "a line without SIM: {line}");
        }
        let f = library(&scene, Variant::WithWalker, seed);
        assert_eq!(
            text.lines().collect::<Vec<_>>(),
            expected_text(&scene, &sha, Variant::WithWalker, seed, &f),
            "seed {seed}"
        );
        // A figure in another's place would show: the median is not the
        // p90, the reference point not the estimator, and not every truth
        // is inside the radius.
        let rtt = f.scores.rtt.unwrap();
        assert_ne!(rtt.median_m, rtt.p90_m);
        assert_ne!(rtt.median_m, rtt.reference.median_m);
        assert_ne!(rtt.inside_radius, rtt.reference.inside_radius);
        assert!(rtt.inside_radius.unwrap() < 1.0);
        // Both kinds of radius-scale line are printed.
        let s = &f.scales;
        let scales = [s.wlm, s.rti, s.rtt, s.rssi];
        assert!(scales.iter().any(Option::is_none) && scales.iter().any(Option::is_some));
    }
}

/// §10: a refusal exits 2 naming the parameter and prints nothing on
/// stdout; with `--records-out`, nothing is written.
#[test]
fn refusals_exit_2_and_name_the_parameter() {
    let dir = temp_dir("refused");
    let scene = bedroom_path();
    let scene = scene.as_str();
    let records = dir.join("r.jsonl");
    let records = records.to_str().unwrap();
    let cases: Vec<(Vec<&str>, &str, &str)> = vec![
        (vec![], "--scene", "is required"),
        (vec![scene], "--scene", "bare argument"),
        (vec!["--scene", "/nonexistent/scene.json"], "--scene", ""),
        (
            vec!["--scene", scene, "--variant", "walker"],
            "--variant",
            "",
        ),
        (vec!["--scene", scene, "--variant"], "--variant", ""),
        (
            vec!["--scene", scene, "--seconds", "4000"],
            "--seconds",
            "3600",
        ),
        (vec!["--scene", scene, "--seconds", "abc"], "--seconds", ""),
        // Room Sim's own refusals, naming its parameter too.
        (
            vec!["--scene", scene, "--seconds", "-1"],
            "--seconds",
            "duration_s",
        ),
        (
            vec!["--scene", scene, "--seconds", "nan"],
            "--seconds",
            "duration_s",
        ),
        (vec!["--scene", scene, "--seed", "-1"], "--seed", ""),
        (
            vec!["--scene", scene, "--seed", "1", "--seed", "2"],
            "--seed",
            "twice",
        ),
        (vec!["--scene", scene, "--sead", "1"], "--sead", "unknown"),
        // A records file holds whole 0.025 s quanta (§3.2, §6.2).
        (
            vec![
                "--scene",
                scene,
                "--seconds",
                "20.01",
                "--records-out",
                records,
            ],
            "--seconds",
            "--records-out",
        ),
        (
            vec!["--scene", scene, "--seconds", "0", "--records-out", records],
            "--seconds",
            "--records-out",
        ),
        (
            vec![
                "--scene",
                scene,
                "--seconds",
                "1",
                "--records-out",
                "/nonexistent/r.jsonl",
            ],
            "--records-out",
            "/nonexistent/r.jsonl",
        ),
    ];
    for (args, param, needle) in cases {
        let mut all = vec!["roomsim-eval"];
        all.extend(&args);
        let out = saans(&all);
        let err = stderr(&out);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {err}");
        assert_eq!(stdout(&out), "", "{args:?}");
        assert!(
            err.starts_with(&format!("roomsim-eval: {param}: ")),
            "{args:?} should name {param}: {err}"
        );
        assert!(err.contains(needle), "{args:?} should say {needle}: {err}");
        assert!(!Path::new(records).exists(), "{args:?} wrote records");
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `--records-out` never replaces a file, as `roomsim-export` without
/// `--overwrite`: a path that exists (another file, the scene itself, its
/// own earlier output, a dangling symlink) is refused naming
/// `--records-out` before the scene is read, and is left as it was.
#[test]
fn records_out_never_replaces_a_file() {
    let dir = temp_dir("existing");
    let existing = dir.join("existing.jsonl");
    std::fs::write(&existing, "keep\n").unwrap();
    let mine = dir.join("mine.json");
    std::fs::copy(scene_path("bedroom.json"), &mine).unwrap();
    let scene_bytes = std::fs::read(&mine).unwrap();
    let e = existing.to_str().unwrap();
    let m = mine.to_str().unwrap();
    let refused = |args: &[&str], path: &str| {
        let mut all = vec!["roomsim-eval", "--seconds", "5"];
        all.extend(args);
        let out = saans(&all);
        assert_eq!(out.status.code(), Some(2), "{args:?}: {}", stderr(&out));
        assert_eq!(stdout(&out), "", "{args:?}");
        assert_eq!(
            stderr(&out),
            format!("roomsim-eval: --records-out: {path}: already exists; roomsim-eval never replaces a file (remove it or give another path)\n"),
        );
    };
    refused(&["--scene", m, "--records-out", e], e);
    // Refused before the scene is read: a missing scene is not reported.
    refused(
        &["--scene", "/nonexistent/scene.json", "--records-out", e],
        e,
    );
    refused(&["--scene", m, "--records-out", m], m);
    refused(&["--scene", m, "--json", "--records-out", m], m);
    assert_eq!(std::fs::read(&existing).unwrap(), b"keep\n");
    assert_eq!(std::fs::read(&mine).unwrap(), scene_bytes);
    #[cfg(unix)]
    {
        let dangling = dir.join("dangling.jsonl");
        let nowhere = dir.join("nowhere.jsonl");
        std::os::unix::fs::symlink(&nowhere, &dangling).unwrap();
        let d = dangling.to_str().unwrap();
        refused(&["--scene", m, "--records-out", d], d);
        assert!(std::fs::symlink_metadata(&nowhere).is_err());
    }

    // A new path is written, and then never written again.
    let fresh = dir.join("fresh.jsonl");
    let f = fresh.to_str().unwrap();
    let out = saans(&[
        "roomsim-eval",
        "--seconds",
        "5",
        "--scene",
        m,
        "--records-out",
        f,
    ]);
    assert!(out.status.success(), "{}", stderr(&out));
    let bytes = std::fs::read(&fresh).unwrap();
    assert!(read_records(bytes.as_slice(), MAX_BYTES).is_ok());
    refused(&["--scene", m, "--seed", "1", "--records-out", f], f);
    assert_eq!(std::fs::read(&fresh).unwrap(), bytes);
    let _ = std::fs::remove_dir_all(&dir);
}

/// §10 and ADR-008 decision 3: the scene is read as `roomsim-export` reads
/// it, so an id problem is refused naming `--scene` without quoting the
/// id, and a valid id appears in no output.
#[test]
fn people_and_phone_ids_never_appear() {
    let dir = temp_dir("ids");
    let text = std::fs::read_to_string(scene_path("bedroom.json")).unwrap();
    let base: Value = serde_json::from_str(&text).unwrap();
    let with = |f: &dyn Fn(&mut Value)| {
        let mut v = base.clone();
        f(&mut v);
        serde_json::to_string_pretty(&v).unwrap()
    };
    for (i, (scene_text, want)) in [
        (
            with(&|v| v["people"][0]["id"] = json!(4242)),
            "people[0].id: expected a string",
        ),
        (
            with(&|v| v["phone"]["id"] = json!(["Asha 4242"])),
            "phone.id: expected a string",
        ),
        (
            with(&|v| v["people"][0] = json!("Asha 4242")),
            "people[0]: expected an object",
        ),
    ]
    .iter()
    .enumerate()
    {
        let scene = dir.join(format!("bad{i}.json"));
        std::fs::write(&scene, scene_text).unwrap();
        let out = saans(&["roomsim-eval", "--scene", scene.to_str().unwrap(), "--json"]);
        let err = stderr(&out);
        assert_eq!(out.status.code(), Some(2), "case {i}: {err}");
        assert!(
            err.starts_with("roomsim-eval: --scene: "),
            "case {i}: {err}"
        );
        assert!(err.trim_end().ends_with(want), "case {i}: {err}");
        assert!(
            !err.contains("4242") && !err.contains("Asha"),
            "case {i}: an id leaked: {err}"
        );
        assert_eq!(stdout(&out), "");
    }

    // Valid planted ids: in no output, and, the bytes not being an
    // embedded scene's, no `scene.id`.
    let scene = dir.join("planted.json");
    std::fs::write(
        &scene,
        with(&|v| {
            v["people"][0]["id"] = json!("Asha 4242");
            v["phone"]["id"] = json!("Ravi 9191");
        }),
    )
    .unwrap();
    let scene = scene.to_str().unwrap();
    let ids = ["Asha 4242".to_string(), "Ravi 9191".to_string()];
    for variant in Variant::ALL {
        let records = dir.join(format!("planted-{}.jsonl", variant.as_str()));
        let records = records.to_str().unwrap();
        let text = eval(scene, variant, 0, &[]);
        let json = eval(scene, variant, 0, &["--json", "--records-out", records]);
        assert!(text.status.success(), "{}", stderr(&text));
        let v = json_of(&json);
        assert!(v["scene"].get("id").is_none(), "{variant:?}");
        assert!(stdout(&text).contains("not an embedded scene"));
        assert_no_id_value(&v, &ids);
        let written = std::fs::read_to_string(records).unwrap();
        for line in written.lines() {
            assert_no_id_value(&serde_json::from_str(line).unwrap(), &ids);
        }
        for (what, s) in [
            ("text", stdout(&text)),
            ("text stderr", stderr(&text)),
            ("json", stdout(&json)),
            ("json stderr", stderr(&json)),
            ("records", written),
        ] {
            assert!(
                !s.contains("Asha") && !s.contains("Ravi"),
                "{variant:?}: an id is in the {what}"
            );
        }
    }
    let _ = std::fs::remove_dir_all(&dir);
}

/// `a` and `b` equal with every `f64` bit for bit; a link's `y`, the one
/// value that may be NaN, is compared by bits.
fn assert_exact(a: &Records, b: &Records) {
    assert_eq!(a.manifest, b.manifest);
    assert_eq!(format!("{:?}", a.manifest), format!("{:?}", b.manifest));
    assert_eq!(a.ticks, b.ticks);
    assert_eq!(format!("{:?}", a.ticks), format!("{:?}", b.ticks));
    assert_eq!(a.epochs, b.epochs);
    assert_eq!(format!("{:?}", a.epochs), format!("{:?}", b.epochs));
    assert_eq!(a.links.len(), b.links.len());
    for (k, (la, lb)) in a.links.iter().zip(&b.links).enumerate() {
        assert_eq!(la.len(), lb.len(), "tick {k}");
        for (l, (x, y)) in la.iter().zip(lb).enumerate() {
            assert_eq!(x.y.to_bits(), y.y.to_bits(), "tick {k} link {l}: y");
            assert_eq!(
                x.gate.map(f64::to_bits),
                y.gate.map(f64::to_bits),
                "tick {k} link {l}: gate"
            );
            assert_eq!(x.disturbed, y.disturbed, "tick {k} link {l}: disturbed");
        }
    }
}

/// The records the command writes, built in memory from the library: one
/// setup and stream to the end, and the empty room's false alarms.
fn library_records(scene: &Scene, sha: &str, variant: Variant, seed: u64) -> Records {
    let rs = roomsim_config(scene, variant, seed);
    let setup = RoomSimSetup::new(scene, &rs, &Config::default(), "SIM", &mut || true)
        .expect("a valid run")
        .expect("never abandoned");
    let run = setup.stream().unwrap().advance_to(f64::INFINITY).unwrap();
    let empty = library_run(scene, Variant::Empty, seed);
    let tick_hz = Config::default().tick_hz;
    let empty_run = (variant != Variant::Empty).then(|| {
        (
            false_alarms(&empty.ticks, PersonEstimator::Wlm, tick_hz),
            false_alarms(&empty.ticks, PersonEstimator::Rti, tick_hz),
        )
    });
    // A link's y that is not finite is written null and read back NaN (§4).
    let mut links = run.links;
    for v in links.iter_mut().flatten() {
        if !v.y.is_finite() {
            v.y = f64::NAN;
        }
    }
    Records {
        manifest: Manifest {
            scene_name: scene.name.clone(),
            scene_sha256: sha.to_string(),
            provenance: scene.provenance,
            variant,
            seed,
            duration_s: SECONDS,
            locate_config: resolve_config(&rs, &Config::default()).unwrap(),
            roomsim_config: rs,
            links: setup.links().to_vec(),
            has_phone: scene.phone.is_some(),
            empty_run,
        },
        ticks: run.ticks,
        links,
        epochs: run.epochs,
    }
}

/// §6, §10: `--records-out` writes the run as a records file the reader
/// accepts, equal to the run held in memory, and consistent with the
/// printed figures.
#[test]
fn records_out_round_trips_and_the_reader_accepts_it() {
    let dir = temp_dir("records");
    let (scene, sha) = bedroom();
    for variant in [Variant::WithWalker, Variant::Empty] {
        for seed in SEEDS {
            let path = dir.join(format!("{}-{seed}.jsonl", variant.as_str()));
            let p = path.to_str().unwrap();
            let out = eval(
                &bedroom_path(),
                variant,
                seed,
                &["--json", "--records-out", p],
            );
            let got = json_of(&out);
            assert!(
                stderr(&out).starts_with("SIM  records: "),
                "{}",
                stderr(&out)
            );

            let bytes = std::fs::read(&path).unwrap();
            let back = read_records(bytes.as_slice(), MAX_BYTES).unwrap_or_else(|e| panic!("{e}"));
            assert_exact(&back, &library_records(&scene, &sha, variant, seed));
            // Writing what was read gives the same bytes.
            let mut again = Vec::new();
            write_records(&mut again, &back).unwrap();
            assert_eq!(again, bytes, "{variant:?} seed {seed}");

            // The file's records give the printed figures.
            let m = &back.manifest;
            assert_eq!((m.seed, m.roomsim_config.seed), (seed, seed));
            let mut stats = score(
                &scene,
                &back.ticks,
                &back.epochs,
                m.has_phone,
                m.locate_config.tick_hz,
            )
            .to_json();
            if let Some((wlm, rti)) = &m.empty_run {
                stats["wlm"]["empty_run"] = serde_json::to_value(wlm).unwrap();
                stats["rti"]["empty_run"] = serde_json::to_value(rti).unwrap();
            }
            assert_eq!(got["stats"], stats, "{variant:?} seed {seed}");
            assert_eq!(m.scene_sha256, got["scene"]["sha256"].as_str().unwrap());
            if (variant, seed) == (Variant::WithWalker, 1) {
                // The manifest's empty run is not all 0, and WLM's is not
                // RTI's.
                let (wlm, rti) = m.empty_run.unwrap();
                assert!(rti.published > 0 && wlm != rti);
            }
        }
    }

    // Without --json the note is a SIM line on stdout.
    let path = dir.join("text.jsonl");
    let out = eval(
        &bedroom_path(),
        Variant::Declared,
        0,
        &["--records-out", path.to_str().unwrap()],
    );
    assert!(out.status.success(), "{}", stderr(&out));
    let text = stdout(&out);
    let last = text.lines().last().unwrap();
    assert!(last.starts_with("SIM  records: "), "{last}");
    assert!(read_records(std::fs::read(&path).unwrap().as_slice(), MAX_BYTES).is_ok());
    let _ = std::fs::remove_dir_all(&dir);
}
