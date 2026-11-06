//! `saans roomsim-eval`: one Room Sim run of a declared scene through
//! Locate, scored as the live demo scores it (spec/livedemo.md §10).
//!
//!     saans roomsim-eval --scene SCENE.json [--seconds S] [--seed N]
//!         [--variant declared|with_walker|empty] [--records-out FILE] [--json]
//!
//! - **The run.** The variant's Room Sim config (`seed`, `duration_s = S`,
//!   every other parameter its default) and Locate's defaults, through one
//!   [`RoomSimSetup`] and one stream advanced to the end: the records
//!   `run_roomsim` gives. Unless the variant is `empty`, the setup's
//!   [`RoomSimSetup::empty_room`] is run to the end too, and only its
//!   `unoccupied` false alarms for WLM and RTI are kept, as `empty_run`
//!   (§5.3).
//! - **The figures** are [`score`]'s §5 block, `empty_run` added, and
//!   [`radius_scale`]'s suggested scales (§8.2): reported, never applied.
//!   With `--seconds 120` (the default) and the same scene, variant and
//!   seed they are the `stats` of a SIM session's final frame.
//! - **The output.** Every line printed starts with SIM, counts before
//!   rates; with `--json`, one JSON object and nothing else on stdout.
//! - **`--records-out FILE`** writes the run as a §6 records file, the
//!   input of the gateway's `--room-replay`. It is written only after
//!   [`read_records`] has accepted the exact bytes, so the gateway's reader
//!   never refuses a file this writes for a reason it can check alone. It
//!   never replaces a file, as `roomsim-export` without `--overwrite`: the
//!   path might be the scene itself or a file a gateway replays, so one
//!   that exists is refused before the run.
//!
//! The scene is read as `roomsim-export` reads it: its people and phone
//! ids are checked before it is parsed, so no message quotes one, and no
//! output names a person or the phone by its scene id (ADR-008 decision
//! 3). A refusal names the parameter (`--scene`, `--seconds` …) and exits
//! with code 2.
//!
//! Everything printed or written is **SIM**: simulated people and a
//! simulated phone in simulated WiFi, a model, not a measurement. Every
//! radius is an uncalibrated model radius (spec/locate.md §8).

use std::fmt;
use std::fs::OpenOptions;
use std::io::{ErrorKind, Write};
use std::process::ExitCode;

use saans_locate::phone::Config;
use saans_locate::pipeline::PipelineError;
use saans_locate::records::{
    provenance_name, read_records, write_records, Manifest, Records, MAX_BYTES,
};
use saans_locate::score::{
    false_alarms, footprint_centroid, radius_scale, score, FalseAlarms, OccupiedStats,
    PersonEstimator, PersonStats, PhoneStats, RadiusScales, Scores, RADIUS_SCALE_MIN_N,
};
use saans_locate::stream::RoomSimSetup;
use saans_locate::variant::Variant;
use saans_locate::LABEL;
use saans_roomsim::Config as RoomSimConfig;
use saans_scene::Scene;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use crate::roomsim_export::scene_id_problem;

/// The run length of a live-demo session, SIM seconds (spec/livedemo.md
/// §3.1): the default `--seconds`, for which the figures equal a session's
/// final `stats` (§10).
pub const ROOMSIM_RUN_S: f64 = 120.0;

/// The longest run `--seconds` may ask for (§10).
pub const MAX_SECONDS: f64 = 3600.0;

/// The command line, as the usage text prints it.
pub const USAGE: &str = "saans roomsim-eval --scene SCENE.json [--seconds S] [--seed N] [--variant declared|with_walker|empty] [--records-out FILE] [--json]";

/// The gateway's embedded scenes `(id, file bytes)`, as saans-gateway's
/// `roomsim.rs` `SCENES` lists them. The room stream opens a scene by this
/// id, and REPLAY matches a records file to a scene by the sha256 of its
/// bytes (§3.1, §6.2), so a scene file has an id only when it is
/// byte-identical to one of these. A unit test keeps the list equal to the
/// files in `spec/scenes/`.
const EMBEDDED_SCENES: [(&str, &[u8]); 3] = [
    (
        "bedroom",
        include_bytes!("../../../../spec/scenes/bedroom.json"),
    ),
    (
        "office",
        include_bytes!("../../../../spec/scenes/office.json"),
    ),
    (
        "hall-l-shaped",
        include_bytes!("../../../../spec/scenes/hall-l-shaped.json"),
    ),
];

/// A refused invocation or run (§10: exit code 2): the parameter it names
/// and why. A reason never quotes a people or phone id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Refused {
    pub param: String,
    pub reason: String,
}

impl Refused {
    fn new(param: impl Into<String>, reason: impl Into<String>) -> Refused {
        Refused {
            param: param.into(),
            reason: reason.into(),
        }
    }
}

impl fmt::Display for Refused {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.param, self.reason)
    }
}

/// What `saans roomsim-eval` was asked to do.
#[derive(Debug, Clone, PartialEq)]
pub struct Options {
    /// The scene file's path.
    pub scene: String,
    /// The run's `duration_s`, SIM seconds.
    pub seconds: f64,
    /// Room Sim's `seed`.
    pub seed: u64,
    pub variant: Variant,
    /// Where to write the §6 records file, if anywhere.
    pub records_out: Option<String>,
    /// One JSON object on stdout instead of the text.
    pub json: bool,
}

impl Options {
    /// Parse the arguments after `roomsim-eval`. As strict as
    /// `roomsim-export`'s: a value that does not parse, an option given
    /// twice, an unknown option or a bare argument is refused rather than
    /// defaulted, because a simulated run is only worth reading if it is
    /// the run that was asked for. `--seconds` above [`MAX_SECONDS`] is
    /// refused here; any other duration Room Sim refuses is refused by the
    /// run, naming `--seconds` and `duration_s`.
    pub fn parse(args: &[String]) -> Result<Options, Refused> {
        let mut scene: Option<String> = None;
        let mut seconds: Option<f64> = None;
        let mut seed: Option<u64> = None;
        let mut variant: Option<Variant> = None;
        let mut records_out: Option<String> = None;
        let mut json = false;
        let mut it = args.iter();
        while let Some(a) = it.next() {
            if a == "--json" {
                if json {
                    return Err(Refused::new(a, "given twice"));
                }
                json = true;
                continue;
            }
            let mut value = |set: bool| -> Result<&String, Refused> {
                if set {
                    return Err(Refused::new(a, "given twice"));
                }
                it.next().ok_or_else(|| Refused::new(a, "needs a value"))
            };
            match a.as_str() {
                "--scene" => scene = Some(value(scene.is_some())?.clone()),
                "--records-out" => records_out = Some(value(records_out.is_some())?.clone()),
                "--seconds" => {
                    let v = value(seconds.is_some())?;
                    let s = v
                        .parse::<f64>()
                        .map_err(|_| Refused::new(a, "not a number of seconds"))?;
                    if s > MAX_SECONDS {
                        return Err(Refused::new(a, format!("at most {MAX_SECONDS} s per run")));
                    }
                    seconds = Some(s);
                }
                "--seed" => {
                    let v = value(seed.is_some())?;
                    seed = Some(
                        v.parse::<u64>()
                            .map_err(|_| Refused::new(a, "not an integer in 0..2^64-1"))?,
                    );
                }
                "--variant" => {
                    let v = value(variant.is_some())?;
                    variant = Some(Variant::parse(v).ok_or_else(|| {
                        Refused::new(a, "must be declared, with_walker or empty")
                    })?);
                }
                f if f.starts_with("--") => return Err(Refused::new(f, "unknown option")),
                _ => {
                    return Err(Refused::new(
                        "--scene",
                        "unexpected bare argument; give the scene as --scene SCENE.json",
                    ))
                }
            }
        }
        Ok(Options {
            scene: scene.ok_or_else(|| Refused::new("--scene", "is required"))?,
            seconds: seconds.unwrap_or(ROOMSIM_RUN_S),
            seed: seed.unwrap_or(0),
            variant: variant.unwrap_or(Variant::Declared),
            records_out,
            json,
        })
    }
}

/// A run, scored. SIM.
#[derive(Debug, Clone)]
pub struct Evaluation {
    /// The embedded scene the file is byte-identical to (§3.1), if any.
    pub scene_id: Option<&'static str>,
    /// The footprint's area centroid, the §5.2 reference point, m.
    pub reference: (f64, f64),
    /// The §5.1 block without `empty_run`, which is the manifest's.
    pub scores: Scores,
    pub radius_scale: RadiusScales,
    /// The run as a §6 records file holds it: the manifest (scene, variant,
    /// seed, configs, links, `empty_run`) and every record.
    pub records: Records,
}

/// The lower-case hex sha256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

/// A refusal of the run by Locate or Room Sim, named by the option behind
/// the parameter: `duration_s` is `--seconds`, `seed` is `--seed`, the
/// people the variant sets are `--variant`, and everything else, with
/// every other parameter at its default, is the scene's. Room Sim names a
/// person `people[i]`, never by an id.
fn run_refused(e: PipelineError) -> Refused {
    let param = match e.param() {
        Some("duration_s") => "--seconds",
        Some("seed") => "--seed",
        Some("n_people" | "phone_carrier") => "--variant",
        _ => "--scene",
    };
    Refused::new(param, e.to_string())
}

/// Read the scene (as `roomsim-export` does), run it and score it.
pub fn evaluate(opts: &Options) -> Result<Evaluation, Refused> {
    let path = &opts.scene;
    // Hash exactly the bytes that are parsed.
    let bytes = std::fs::read(path).map_err(|e| Refused::new("--scene", format!("{path}: {e}")))?;
    let text = std::str::from_utf8(&bytes)
        .map_err(|e| Refused::new("--scene", format!("{path}: not UTF-8: {e}")))?;
    if let Some(why) = scene_id_problem(text) {
        return Err(Refused::new("--scene", format!("{path}: {why}")));
    }
    let scene = Scene::from_json_str(path, text)
        .map_err(|e| Refused::new("--scene", e.to_string().trim_end().to_string()))?;
    let scene_id = EMBEDDED_SCENES
        .iter()
        .find(|(_, embedded)| *embedded == bytes.as_slice())
        .map(|(id, _)| *id);

    let base = RoomSimConfig {
        seed: opts.seed,
        duration_s: opts.seconds,
        ..RoomSimConfig::default()
    };
    let rs = opts.variant.apply(&scene, &base);
    let Some(setup) = RoomSimSetup::new(&scene, &rs, &Config::default(), LABEL, &mut || true)
        .map_err(run_refused)?
    else {
        unreachable!("a calibration whose keep_going is always true is never abandoned");
    };
    let run = setup
        .stream()
        .and_then(|mut s| s.advance_to(f64::INFINITY))
        .map_err(run_refused)?;

    // §5.3: the empty room of the same scene, seed and duration, sharing
    // the calibration; only its false alarms are kept.
    let empty_run = if opts.variant == Variant::Empty {
        None
    } else {
        let empty = setup.empty_room().map_err(run_refused)?;
        let ticks = empty
            .stream()
            .and_then(|mut s| s.advance_to(f64::INFINITY))
            .map_err(run_refused)?
            .ticks;
        let alarms = |which| false_alarms(&ticks, which, empty.tick_hz());
        Some((alarms(PersonEstimator::Wlm), alarms(PersonEstimator::Rti)))
    };

    let has_phone = setup.has_phone();
    let scores = score(&scene, &run.ticks, &run.epochs, has_phone, setup.tick_hz());
    let scales = radius_scale(&scene, &run.ticks, &run.epochs, has_phone);
    let manifest = Manifest {
        scene_name: scene.name.clone(),
        scene_sha256: sha256_hex(&bytes),
        provenance: scene.provenance,
        variant: opts.variant,
        seed: rs.seed,
        duration_s: setup.duration_s(),
        roomsim_config: setup.roomsim_config().clone(),
        locate_config: setup.config().clone(),
        links: setup.links().to_vec(),
        has_phone,
        empty_run,
    };
    Ok(Evaluation {
        scene_id,
        reference: footprint_centroid(&scene),
        scores,
        radius_scale: scales,
        records: Records {
            manifest,
            ticks: run.ticks,
            links: run.links,
            epochs: run.epochs,
        },
    })
}

/// §5.1 `FalseAlarms`, as its serde encoding writes it.
fn false_alarms_json(fa: &FalseAlarms) -> Value {
    json!({
        "ticks": fa.ticks,
        "seconds": fa.seconds,
        "published": fa.published,
        "per_h": fa.per_h,
        "events": fa.events,
        "events_per_h": fa.events_per_h,
    })
}

impl Evaluation {
    /// `{"label":"SIM","scene":{"id"?,"name","sha256","provenance"},
    /// "run":{"variant","seed","duration_s"},"stats":{…},
    /// "radius_scale":{"wlm","rti","rtt","rssi"}}`: `stats` is §5.1's
    /// block, `empty_run` added to `wlm` and `rti` unless the variant is
    /// `empty`; `scene.id` is present only for an embedded scene; a
    /// radius scale is `null` when fewer than 20 estimates were scored.
    pub fn to_json(&self) -> Value {
        let m = &self.records.manifest;
        let mut scene = Map::new();
        if let Some(id) = self.scene_id {
            scene.insert("id".into(), id.into());
        }
        scene.insert("name".into(), m.scene_name.clone().into());
        scene.insert("sha256".into(), m.scene_sha256.clone().into());
        scene.insert("provenance".into(), provenance_name(m.provenance).into());
        let mut stats = self.scores.to_json();
        if let Some((wlm, rti)) = &m.empty_run {
            for (key, fa) in [("wlm", wlm), ("rti", rti)] {
                if let Some(block) = stats.get_mut(key).and_then(Value::as_object_mut) {
                    block.insert("empty_run".into(), false_alarms_json(fa));
                }
            }
        }
        let s = &self.radius_scale;
        json!({
            "label": LABEL,
            "scene": scene,
            "run": {
                "variant": m.variant.as_str(),
                "seed": m.seed,
                "duration_s": m.duration_s,
            },
            "stats": stats,
            "radius_scale": {"wlm": s.wlm, "rti": s.rti, "rtt": s.rtt, "rssi": s.rssi},
        })
    }

    /// The text report: every line starts with SIM; counts come before the
    /// rates and fractions made from them.
    pub fn text(&self) -> Vec<String> {
        let m = &self.records.manifest;
        let cfg = &m.locate_config;
        let simulated = if m.has_phone {
            "simulated people and a simulated phone"
        } else {
            "simulated people"
        };
        let mut out = vec![
            format!("saans roomsim-eval (spec/livedemo.md §10): Room Sim v1 and Locate v1 on a declared scene; {simulated} in simulated WiFi, a model, not a measurement"),
            format!(
                "scene {:?}: provenance {}, sha256 {}, {}",
                m.scene_name,
                provenance_name(m.provenance),
                m.scene_sha256,
                match self.scene_id {
                    Some(id) => format!("the gateway's embedded scene {id}"),
                    None => "not an embedded scene (the gateway's --room-replay refuses its records)"
                        .to_string(),
                }
            ),
            format!(
                "run: variant {}, seed {}, {} s simulated, {} ticks at {} Hz, {} links; Room Sim and Locate defaults otherwise",
                m.variant.as_str(),
                m.seed,
                m.duration_s,
                self.records.ticks.len(),
                cfg.tick_hz,
                m.links.len()
            ),
            format!(
                "radii are uncalibrated model radii; the reference point is the footprint's area centroid ({:.2}, {:.2}) m, scored on the same records with each record's own radius: an estimator that knows nothing",
                self.reference.0, self.reference.1
            ),
        ];
        let empty_run = m.empty_run.as_ref();
        let s = &self.scores;
        let scales = &self.radius_scale;
        person_lines(
            &mut out,
            "WLM (baseline)",
            &s.wlm,
            empty_run.map(|e| &e.0),
            scales.wlm,
            m.variant,
        );
        person_lines(
            &mut out,
            "RTI",
            &s.rti,
            empty_run.map(|e| &e.1),
            scales.rti,
            m.variant,
        );
        match (&s.rtt, &s.rssi) {
            (Some(rtt), Some(rssi)) => {
                phone_lines(&mut out, "phone RTT", rtt, scales.rtt);
                phone_lines(&mut out, "phone RSSI", rssi, scales.rssi);
            }
            _ => out.push("phone: the scene has none, so no phone figures".to_string()),
        }
        out.into_iter().map(|l| format!("SIM  {l}")).collect()
    }
}

/// `k of n<unit> (p%)`, with `p` from `fraction` (`k / n`), or without
/// the percentage when `fraction` is `None` (n = 0).
fn of(k: usize, n: usize, unit: &str, fraction: Option<f64>) -> String {
    match fraction {
        Some(f) => format!("{k} of {n}{unit} ({:.1}%)", 100.0 * f),
        None => format!("{k} of {n}{unit}"),
    }
}

/// The count behind a fraction of `n` (exact: the fraction is `k / n`).
fn count(fraction: Option<f64>, n: usize) -> usize {
    fraction.map_or(0, |f| (f * n as f64).round() as usize)
}

/// `1.23 m`, or `null` (no figure: nothing scored).
fn metres(x: Option<f64>) -> String {
    x.map_or_else(|| "null".to_string(), |v| format!("{v:.2} m"))
}

/// `x per hour`, or `null` when there are no seconds to divide by.
fn per_hour(x: Option<f64>) -> String {
    x.map_or_else(|| "null".to_string(), |v| format!("{v:.1} per hour"))
}

/// A `FalseAlarms` block, counts first.
fn alarms(fa: &FalseAlarms) -> String {
    let published = (fa.ticks > 0).then(|| fa.published as f64 / fa.ticks as f64);
    format!(
        "{} ticks = {:.1} s; published {} = {}; {} events (runs of consecutive published ticks) = {}",
        fa.ticks,
        fa.seconds,
        of(fa.published, fa.ticks, " ticks", published),
        per_hour(fa.per_h),
        fa.events,
        per_hour(fa.events_per_h)
    )
}

/// The error figures of `n` scored estimates and their reference point.
fn accuracy_lines(
    out: &mut Vec<String>,
    name: &str,
    truth: &str,
    n: usize,
    own: (Option<f64>, Option<f64>, Option<f64>),
    radius_median_m: Option<f64>,
    reference: (Option<f64>, Option<f64>, Option<f64>),
) {
    if n == 0 {
        out.push(format!("{name}: no estimate scored, so no error figures"));
        return;
    }
    out.push(format!(
        "{name}: error to {truth} over {n} scored: median {}, p90 {}; truth inside the radius for {}; radius median {}",
        metres(own.0),
        metres(own.1),
        of(count(own.2, n), n, "", own.2),
        metres(radius_median_m)
    ));
    out.push(format!(
        "{name}: reference point over the same {n}: median {}, p90 {}; truth inside the same radius for {}",
        metres(reference.0),
        metres(reference.1),
        of(count(reference.2, n), n, "", reference.2)
    ));
}

/// The suggested radius scale of `n` scored estimates (§8.2).
fn scale_line(out: &mut Vec<String>, name: &str, n: usize, scale: Option<f64>) {
    out.push(match scale {
        Some(k) => format!(
            "{name}: suggested radius scale {k:.3} from n = {n} scored (95% of the scored errors are within scale x radius_m; SIM, reported, not applied)"
        ),
        None => format!(
            "{name}: suggested radius scale: too few (n = {n} < {RADIUS_SCALE_MIN_N})"
        ),
    });
}

/// One person estimator's lines: its ticks; its coverage of the occupied
/// ticks with the empty run's false alarms beside it (ADR-005 §5), then
/// the error figures; for variant `empty` the run's own false alarms in
/// place of both; the suggested radius scale.
fn person_lines(
    out: &mut Vec<String>,
    name: &str,
    st: &PersonStats,
    empty_run: Option<&FalseAlarms>,
    scale: Option<f64>,
    variant: Variant,
) {
    let o: &OccupiedStats = &st.occupied;
    let u = &st.unoccupied;
    out.push(format!(
        "{name}: {} ticks: {} with a person in the truth, {} with none",
        st.ticks, o.ticks, u.ticks
    ));
    let occupied = o.ticks > 0 || variant != Variant::Empty;
    if occupied {
        out.push(format!(
            "{name}: published {}",
            of(o.published, o.ticks, " occupied ticks", o.coverage)
        ));
    }
    if let Some(fa) = empty_run {
        out.push(format!(
            "{name}: empty room (variant empty, same scene, seed and seconds): {}",
            alarms(fa)
        ));
    }
    if occupied {
        accuracy_lines(
            out,
            name,
            "the nearest person",
            o.published,
            (o.median_m, o.p90_m, o.inside_radius),
            o.radius_median_m,
            (
                o.reference.median_m,
                o.reference.p90_m,
                o.reference.inside_radius,
            ),
        );
    }
    if variant == Variant::Empty {
        out.push(format!(
            "{name}: false alarms, the room is empty (variant empty): {}",
            alarms(u)
        ));
    } else if u.ticks > 0 {
        out.push(format!(
            "{name}: false alarms on the ticks with no person: {}",
            alarms(u)
        ));
    }
    scale_line(out, name, o.published, scale);
}

/// One phone method's lines (§5.1 `PhoneStats`): every published estimate
/// is scored.
fn phone_lines(out: &mut Vec<String>, name: &str, st: &PhoneStats, scale: Option<f64>) {
    out.push(format!(
        "{name}: published {}",
        of(st.published, st.epochs, " epochs", st.coverage)
    ));
    accuracy_lines(
        out,
        name,
        "the phone",
        st.published,
        (st.median_m, st.p90_m, st.inside_radius),
        st.radius_median_m,
        (
            st.reference.median_m,
            st.reference.p90_m,
            st.reference.inside_radius,
        ),
    );
    scale_line(out, name, st.published, scale);
}

/// `--records-out` naming a file that exists, which is never replaced.
fn already_exists(path: &str) -> Refused {
    Refused::new(
        "--records-out",
        format!("{path}: already exists; roomsim-eval never replaces a file (remove it or give another path)"),
    )
}

/// Write `records` to `path` as a §6 file, but only once [`read_records`]
/// has accepted the exact bytes (a run longer than the reader's 64 MiB, or
/// a duration that is not whole session quanta, would otherwise give a
/// file the gateway refuses). The file is created new, so one that
/// appeared since [`run`]'s check is refused too, and a write that fails
/// removes what it created.
fn write_records_file(path: &str, records: &Records) -> Result<(), Refused> {
    let refused = |reason: String| Refused::new("--records-out", reason);
    let mut bytes = Vec::new();
    write_records(&mut bytes, records)
        .map_err(|e| refused(format!("the run cannot be written as records: {e}")))?;
    read_records(bytes.as_slice(), MAX_BYTES).map_err(|e| {
        refused(format!(
            "the records reader would refuse this run's file ({e}); nothing was written"
        ))
    })?;
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .map_err(|e| match e.kind() {
            ErrorKind::AlreadyExists => already_exists(path),
            _ => refused(format!("{path}: {e}")),
        })?;
    if let Err(e) = file.write_all(&bytes) {
        drop(file);
        let _ = std::fs::remove_file(path);
        return Err(refused(format!(
            "{path}: {e}; the partial file was removed"
        )));
    }
    Ok(())
}

/// The command after its arguments are parsed: refuse, before running, a
/// duration no records file can hold and a records path that exists; run;
/// write the records; give the lines for stdout.
fn run(opts: &Options) -> Result<Vec<String>, Refused> {
    if let Some(path) = &opts.records_out {
        // §3.2, §6.2: a records file holds a whole number of 0.025 s
        // quanta. Refused before the run, rather than after it by the
        // reader.
        let s = opts.seconds;
        if !(s > 0.0 && (40.0 * s).fract() == 0.0) {
            return Err(Refused::new(
                "--seconds",
                "with --records-out it must be > 0 and a whole number of 0.025 s (the session clock's quanta, spec/livedemo.md §3.2)",
            ));
        }
        // A dangling symlink counts as existing, as in `roomsim-export`.
        if std::fs::symlink_metadata(path).is_ok() {
            return Err(already_exists(path));
        }
    }
    let eval = evaluate(opts)?;
    let mut lines = if opts.json {
        vec![eval.to_json().to_string()]
    } else {
        eval.text()
    };
    if let Some(path) = &opts.records_out {
        write_records_file(path, &eval.records)?;
        let r = &eval.records;
        let note = format!(
            "SIM  records: {} ticks and {} epochs, with their truth (SIM), written to {path:?}",
            r.ticks.len(),
            r.epochs.len()
        );
        if opts.json {
            // Nothing but the JSON object on stdout.
            eprintln!("{note}");
        } else {
            lines.push(note);
        }
    }
    Ok(lines)
}

/// `saans roomsim-eval …`; `args[0]` is `roomsim-eval`. Exit code 0, or 2
/// on a refusal, which names the parameter.
pub fn main(args: &[String]) -> ExitCode {
    let opts = match Options::parse(args.get(1..).unwrap_or_default()) {
        Ok(o) => o,
        Err(e) => {
            eprintln!("roomsim-eval: {e}\nusage: {USAGE}");
            return ExitCode::from(2);
        }
    };
    match run(&opts) {
        Ok(lines) => {
            let mut stdout = std::io::stdout().lock();
            let written = lines
                .iter()
                .try_for_each(|l| writeln!(stdout, "{l}"))
                .and_then(|()| stdout.flush());
            if written.is_err() {
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(e) => {
            eprintln!("roomsim-eval: {e}");
            ExitCode::from(2)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use saans_locate::score::Accuracy;

    fn parse(args: &[&str]) -> Result<Options, Refused> {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        Options::parse(&args)
    }

    #[test]
    fn the_defaults_are_the_sessions() {
        let o = parse(&["--scene", "s.json"]).unwrap();
        assert_eq!(
            o,
            Options {
                scene: "s.json".into(),
                seconds: ROOMSIM_RUN_S,
                seed: 0,
                variant: Variant::Declared,
                records_out: None,
                json: false,
            }
        );
        let o = parse(&[
            "--json",
            "--variant",
            "with_walker",
            "--seed",
            "18446744073709551615",
            "--seconds",
            "3600",
            "--records-out",
            "r.jsonl",
            "--scene",
            "s.json",
        ])
        .unwrap();
        assert_eq!(o.variant, Variant::WithWalker);
        assert_eq!(o.seed, u64::MAX);
        assert_eq!(o.seconds, MAX_SECONDS);
        assert_eq!(o.records_out.as_deref(), Some("r.jsonl"));
        assert!(o.json);
    }

    #[test]
    fn a_bad_invocation_names_its_parameter() {
        for (args, param) in [
            (vec![], "--scene"),
            (vec!["s.json"], "--scene"),
            (vec!["--scene"], "--scene"),
            (vec!["--scene", "a", "--scene", "b"], "--scene"),
            (vec!["--scene", "a", "--seconds", "3600.5"], "--seconds"),
            (vec!["--scene", "a", "--seconds", "inf"], "--seconds"),
            (vec!["--scene", "a", "--seconds", "two"], "--seconds"),
            (vec!["--scene", "a", "--seed", "-1"], "--seed"),
            (vec!["--scene", "a", "--variant", "walker"], "--variant"),
            (vec!["--scene", "a", "--variant", "scene"], "--variant"),
            (vec!["--scene", "a", "--json", "--json"], "--json"),
            (vec!["--scene", "a", "--records-out"], "--records-out"),
            (vec!["--scene", "a", "--sead", "1"], "--sead"),
        ] {
            let e = parse(&args).expect_err(&format!("{args:?}"));
            assert_eq!(e.param, param, "{args:?}: {e}");
        }
    }

    /// The embedded list is the gateway's: every file in `spec/scenes/`,
    /// by its stem, byte for byte.
    #[test]
    fn the_embedded_scenes_are_the_files_in_spec_scenes() {
        let dir = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes");
        let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
            .unwrap()
            .map(|e| e.unwrap().path())
            .filter(|p| p.extension().is_some_and(|x| x == "json"))
            .map(|p| {
                let stem = p.file_stem().unwrap().to_string_lossy().into_owned();
                (stem, std::fs::read(&p).unwrap())
            })
            .collect();
        files.sort();
        let mut embedded: Vec<(String, Vec<u8>)> = EMBEDDED_SCENES
            .iter()
            .map(|(id, b)| (id.to_string(), b.to_vec()))
            .collect();
        embedded.sort();
        assert_eq!(embedded, files);
    }

    /// Hand-built figures, every one distinct from the others, so a figure
    /// printed in another's place shows: published 3 is not events 2, and
    /// neither rate is the other.
    fn alarms_of(ticks: usize, published: usize, events: usize) -> FalseAlarms {
        let seconds = ticks as f64 / 2.0;
        FalseAlarms {
            ticks,
            seconds,
            published,
            per_h: Some(published as f64 / seconds * 3600.0),
            events,
            events_per_h: Some(events as f64 / seconds * 3600.0),
        }
    }

    fn accuracy(median_m: f64, p90_m: f64, inside_radius: f64) -> Accuracy {
        Accuracy {
            median_m: Some(median_m),
            p90_m: Some(p90_m),
            inside_radius: Some(inside_radius),
        }
    }

    #[test]
    fn false_alarms_print_counts_then_both_rates() {
        assert_eq!(
            alarms(&alarms_of(40, 3, 2)),
            "40 ticks = 20.0 s; published 3 of 40 ticks (7.5%) = 540.0 per hour; 2 events (runs of consecutive published ticks) = 360.0 per hour"
        );
        let none = FalseAlarms {
            ticks: 0,
            seconds: 0.0,
            published: 0,
            per_h: None,
            events: 0,
            events_per_h: None,
        };
        assert_eq!(
            alarms(&none),
            "0 ticks = 0.0 s; published 0 of 0 ticks = null; 0 events (runs of consecutive published ticks) = null"
        );
    }

    /// An occupied block with unoccupied ticks too, the empty run beside
    /// it: every line, each figure in its own place.
    #[test]
    fn a_person_block_prints_each_figure_in_its_place() {
        let st = PersonStats {
            ticks: 60,
            occupied: OccupiedStats {
                ticks: 50,
                published: 25,
                coverage: Some(0.5),
                median_m: Some(0.5),
                p90_m: Some(1.25),
                inside_radius: Some(0.76),
                radius_median_m: Some(2.0),
                reference: accuracy(1.5, 2.75, 0.28),
            },
            unoccupied: alarms_of(10, 4, 1),
        };
        let mut out = Vec::new();
        let empty_run = alarms_of(40, 3, 2);
        person_lines(
            &mut out,
            "WLM",
            &st,
            Some(&empty_run),
            Some(0.4567),
            Variant::WithWalker,
        );
        assert_eq!(
            out,
            [
                "WLM: 60 ticks: 50 with a person in the truth, 10 with none",
                "WLM: published 25 of 50 occupied ticks (50.0%)",
                "WLM: empty room (variant empty, same scene, seed and seconds): 40 ticks = 20.0 s; published 3 of 40 ticks (7.5%) = 540.0 per hour; 2 events (runs of consecutive published ticks) = 360.0 per hour",
                "WLM: error to the nearest person over 25 scored: median 0.50 m, p90 1.25 m; truth inside the radius for 19 of 25 (76.0%); radius median 2.00 m",
                "WLM: reference point over the same 25: median 1.50 m, p90 2.75 m; truth inside the same radius for 7 of 25 (28.0%)",
                "WLM: false alarms on the ticks with no person: 10 ticks = 5.0 s; published 4 of 10 ticks (40.0%) = 2880.0 per hour; 1 events (runs of consecutive published ticks) = 720.0 per hour",
                "WLM: suggested radius scale 0.457 from n = 25 scored (95% of the scored errors are within scale x radius_m; SIM, reported, not applied)",
            ]
        );

        // Nothing published: no error figures, and too few for a scale.
        let mut nothing = st;
        nothing.occupied = OccupiedStats {
            published: 0,
            coverage: Some(0.0),
            median_m: None,
            p90_m: None,
            inside_radius: None,
            radius_median_m: None,
            reference: Accuracy {
                median_m: None,
                p90_m: None,
                inside_radius: None,
            },
            ..st.occupied
        };
        nothing.unoccupied = alarms_of(10, 0, 0);
        let mut out = Vec::new();
        person_lines(
            &mut out,
            "RTI",
            &nothing,
            Some(&empty_run),
            None,
            Variant::Declared,
        );
        assert_eq!(out[1], "RTI: published 0 of 50 occupied ticks (0.0%)");
        assert_eq!(out[3], "RTI: no estimate scored, so no error figures");
        assert_eq!(out[5], "RTI: suggested radius scale: too few (n = 0 < 20)");
        assert_eq!(out.len(), 6);
    }

    /// `--variant empty`: the run's own false alarms replace the coverage
    /// and the errors, and there is no empty run beside them.
    #[test]
    fn an_empty_room_prints_its_false_alarms_in_place_of_errors() {
        let st = PersonStats {
            ticks: 40,
            occupied: OccupiedStats {
                ticks: 0,
                published: 0,
                coverage: None,
                median_m: None,
                p90_m: None,
                inside_radius: None,
                radius_median_m: None,
                reference: Accuracy {
                    median_m: None,
                    p90_m: None,
                    inside_radius: None,
                },
            },
            unoccupied: alarms_of(40, 3, 2),
        };
        let mut out = Vec::new();
        person_lines(&mut out, "RTI", &st, None, None, Variant::Empty);
        assert_eq!(
            out,
            [
                "RTI: 40 ticks: 0 with a person in the truth, 40 with none",
                "RTI: false alarms, the room is empty (variant empty): 40 ticks = 20.0 s; published 3 of 40 ticks (7.5%) = 540.0 per hour; 2 events (runs of consecutive published ticks) = 360.0 per hour",
                "RTI: suggested radius scale: too few (n = 0 < 20)",
            ]
        );
    }

    #[test]
    fn a_phone_block_prints_each_figure_in_its_place() {
        let st = PhoneStats {
            epochs: 50,
            published: 40,
            coverage: Some(0.8),
            median_m: Some(1.8),
            p90_m: Some(3.06),
            inside_radius: Some(0.45),
            radius_median_m: Some(1.72),
            reference: accuracy(1.51, 1.93, 0.85),
        };
        let mut out = Vec::new();
        phone_lines(&mut out, "phone RTT", &st, Some(1.745));
        assert_eq!(
            out,
            [
                "phone RTT: published 40 of 50 epochs (80.0%)",
                "phone RTT: error to the phone over 40 scored: median 1.80 m, p90 3.06 m; truth inside the radius for 18 of 40 (45.0%); radius median 1.72 m",
                "phone RTT: reference point over the same 40: median 1.51 m, p90 1.93 m; truth inside the same radius for 34 of 40 (85.0%)",
                "phone RTT: suggested radius scale 1.745 from n = 40 scored (95% of the scored errors are within scale x radius_m; SIM, reported, not applied)",
            ]
        );
    }
}
