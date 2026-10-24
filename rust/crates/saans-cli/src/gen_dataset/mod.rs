//! `saans gen-dataset`: random scenes, Room Sim and Locate on each, one
//! `.npz` shard per scene and a manifest (spec/dataset.md).
//!
//!     saans gen-dataset --out DIR [--scenes N] [--seconds S] [--seed N]
//!         [--threads N] [--no-impair] [--from DIR] [--overwrite]
//!
//! - **Phase 1** (§8) draws and checks every scene in parallel ([`generator`],
//!   §2, §9), or with `--from DIR` reads DIR's scene files, configs and
//!   splits and checks them; then prints the estimated size beside the
//!   target filesystem's free space and refuses when the estimate exceeds
//!   the free space less 5 GB.
//! - **Phase 2** runs §3-§6 per scene in parallel ([`shard`]) and writes
//!   `scenes/gen-NNNNNN.json`, `.roomsim.json`, `.locate.json` and
//!   `shards/gen-NNNNNN.npz` ([`npz`], §7.2), each to a temporary name by
//!   the thread that made it and then renamed; the manifest (§7.3) is
//!   written last, so a directory without one is an unfinished run.
//! - **Determinism** (§9): every draw comes from `saans_sim::rng::Rng`
//!   streams keyed by the scene index, results are gathered in index
//!   order, and no file holds a time or a path that varies, so `scenes/`,
//!   `shards/` and the manifest (except `totals`) are byte-identical for
//!   any `--threads`.
//! - **`--no-impair`** turns the hardware impairments off and keeps the
//!   receiver's noise ([`generator::Impairments::HardwareOff`], §8; Room
//!   Sim's master switch `impair` stays true). The scenes are drawn and
//!   checked with the impairments on (§2.6), and `--from` re-checks them
//!   so, so the scene files, the configs but those switches, the people
//!   and the labels are the impairments-on dataset's. `baseline_valid` is
//!   true when every scene's calibration has a link with `null_sigma > 0`.
//! - **Refusals** name the parameter and exit with code 2: a bad or empty
//!   option, a non-empty `--out` without `--overwrite` (which removes only
//!   `manifest.json`, `scenes/` and `shards/` there), too little free
//!   space, a `--from` that is not a directory with a manifest, is `--out`
//!   or lies in its `scenes/` or `shards/`, or disagrees with `--scenes`,
//!   `--seconds` or `--seed`. An I/O error, a generator that finds no
//!   valid draw in 100 attempts, or a run Locate or Room Sim refuses exits
//!   with code 1, naming the scene.
//!
//! Everything printed or written is **SIM**: rooms drawn at random (no
//! generated scene describes a room, provenance `generated`), simulated
//! people in simulated WiFi, a model, not a measurement. Progress goes to
//! stderr every 5% (a closed stderr is ignored, never a panic); the summary
//! to stdout; every line starts with SIM.
//! No generated person or phone has an id, and none is ever printed.

mod generator;
mod npz;
mod shard;

use std::collections::BTreeMap;
use std::fmt;
use std::fs::{self, File};
use std::io::{self, BufWriter, ErrorKind, Write};
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rayon::prelude::*;
use saans_locate::phone::Config as LocateConfig;
use saans_locate::records::locate_config_json;
use saans_roomsim::Config as RoomSimConfig;
use saans_scene::Scene;
use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};

use generator::{scene_id, wall_material_name, Impairments, Rejections};

/// The spec string every file of a dataset carries (§7.2, §7.3).
pub const SPEC: &str = "dataset v1";

/// The command line, as the usage text prints it.
pub const USAGE: &str = "saans gen-dataset --out DIR [--scenes N] [--seconds S] [--seed N] [--threads N] [--no-impair] [--from DIR] [--overwrite]";

/// §1's defaults and bounds.
pub const DEFAULT_SCENES: usize = 2000;
pub const MAX_SCENES: usize = 100_000;
pub const DEFAULT_SECONDS: u32 = 120;
pub const MAX_SECONDS: u32 = 600;

/// §8: the size estimate must fit in the free space less this (5 GB).
pub const RESERVE_BYTES: u64 = 5_000_000_000;

/// For tests only: the free bytes the size check believes, instead of
/// asking `df` (§11 "the size check (injected free space)").
pub const FREE_BYTES_ENV: &str = "SAANS_GEN_DATASET_FREE_BYTES";

/// A refused invocation (§8: exit code 2): the parameter it names and why.
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

/// Why the command stopped: a refusal (exit 2), or an I/O error or a
/// failed scene (exit 1).
#[derive(Debug)]
enum Failure {
    Refused(Refused),
    Io(String),
    Failed(String),
}

impl From<Refused> for Failure {
    fn from(r: Refused) -> Failure {
        Failure::Refused(r)
    }
}

/// An I/O error on `path`.
fn io_err(path: &Path, e: io::Error) -> Failure {
    Failure::Io(format!("{}: {e}", path.display()))
}

/// What `saans gen-dataset` was asked to do. `scenes`, `seconds` and
/// `seed` are `None` when not given (their defaults, or `--from`'s).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Options {
    pub out: String,
    pub scenes: Option<usize>,
    pub seconds: Option<u32>,
    pub seed: Option<u64>,
    pub threads: usize,
    pub no_impair: bool,
    pub from: Option<String>,
    pub overwrite: bool,
}

impl Options {
    /// Parse the arguments after `gen-dataset`, as strictly as
    /// `roomsim-eval`: a value that does not parse or is out of range, an
    /// option given twice, an unknown option or a bare argument is refused
    /// rather than defaulted.
    pub fn parse(args: &[String]) -> Result<Options, Refused> {
        let mut out: Option<String> = None;
        let mut scenes: Option<usize> = None;
        let mut seconds: Option<u32> = None;
        let mut seed: Option<u64> = None;
        let mut threads: Option<usize> = None;
        let mut from: Option<String> = None;
        let (mut no_impair, mut overwrite) = (false, false);
        let mut it = args.iter();
        while let Some(a) = it.next() {
            let flag = match a.as_str() {
                "--no-impair" => Some(&mut no_impair),
                "--overwrite" => Some(&mut overwrite),
                _ => None,
            };
            if let Some(set) = flag {
                if *set {
                    return Err(Refused::new(a, "given twice"));
                }
                *set = true;
                continue;
            }
            let mut value = |set: bool| -> Result<&String, Refused> {
                if set {
                    return Err(Refused::new(a, "given twice"));
                }
                it.next().ok_or_else(|| Refused::new(a, "needs a value"))
            };
            // A path that is empty would name the working directory's files.
            let path = |v: &String| {
                if v.is_empty() {
                    Err(Refused::new(a, "is empty; give a directory"))
                } else {
                    Ok(v.clone())
                }
            };
            match a.as_str() {
                "--out" => out = Some(path(value(out.is_some())?)?),
                "--from" => from = Some(path(value(from.is_some())?)?),
                "--scenes" => {
                    let v = value(scenes.is_some())?;
                    scenes = Some(
                        v.parse::<usize>()
                            .ok()
                            .filter(|n| (1..=MAX_SCENES).contains(n))
                            .ok_or_else(|| {
                                Refused::new(a, format!("must be an integer in 1..={MAX_SCENES}"))
                            })?,
                    );
                }
                "--seconds" => {
                    let v = value(seconds.is_some())?;
                    seconds = Some(
                        v.parse::<u32>()
                            .ok()
                            .filter(|s| *s > 0 && s % 2 == 0 && *s <= MAX_SECONDS)
                            .ok_or_else(|| {
                                Refused::new(
                                    a,
                                    format!("must be a positive multiple of 2 up to {MAX_SECONDS}"),
                                )
                            })?,
                    );
                }
                "--seed" => {
                    let v = value(seed.is_some())?;
                    seed = Some(
                        v.parse::<u64>()
                            .map_err(|_| Refused::new(a, "not an integer in 0..2^64-1"))?,
                    );
                }
                "--threads" => {
                    let v = value(threads.is_some())?;
                    threads = Some(
                        v.parse::<usize>()
                            .ok()
                            .filter(|n| (1..=4096).contains(n))
                            .ok_or_else(|| Refused::new(a, "must be an integer in 1..=4096"))?,
                    );
                }
                f if f.starts_with("--") => return Err(Refused::new(f, "unknown option")),
                _ => {
                    return Err(Refused::new(
                        "--out",
                        "unexpected bare argument; give the directory as --out DIR",
                    ))
                }
            }
        }
        Ok(Options {
            out: out.ok_or_else(|| Refused::new("--out", "is required"))?,
            scenes,
            seconds,
            seed,
            threads: threads
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get())),
            no_impair,
            from,
            overwrite,
        })
    }
}

/// One line on stderr. A closed or failing stderr is ignored: progress is
/// never worth a panic (§8).
fn note(line: &str) {
    let _ = writeln!(io::stderr(), "{line}");
}

/// The lower-case hex sha256 of `bytes`.
fn sha256_hex(bytes: &[u8]) -> String {
    Sha256::digest(bytes)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ---------------------------------------------------------------------
// Phase 1: what each scene will be
// ---------------------------------------------------------------------

/// What the manifest records about a scene's draw (§7.3), drawn here or
/// read from `--from`'s manifest.
#[derive(Debug, Clone, PartialEq)]
struct Meta {
    modes: Vec<String>,
    objects_drawn: u64,
    objects_placed: u64,
    openings_drawn: u64,
    openings_placed: u64,
    l_shape: bool,
    wall_material: String,
    attempts: u64,
}

/// A scene's split: its §9 draw, or the split `--from`'s manifest gives.
#[derive(Debug, Clone, PartialEq)]
enum Split {
    Draw(f64),
    Given(String),
}

/// One scene after phase 1: everything phase 2 needs.
#[derive(Debug, Clone)]
struct Plan {
    index: usize,
    scene: Scene,
    /// The scene file's bytes, written as they are.
    scene_bytes: Vec<u8>,
    /// The run config, its impairment profile applied.
    rs: RoomSimConfig,
    meta: Meta,
    split: Split,
}

/// Phase 1 without `--from`: draw and check every scene (§2) in parallel.
fn draw_all(
    seed: u64,
    scenes: usize,
    seconds: u32,
    impairments: Impairments,
) -> Result<(Vec<Plan>, Rejections), Failure> {
    let drawn: Vec<_> = (0..scenes)
        .into_par_iter()
        .map(|i| generator::generate(seed, i, seconds))
        .collect();
    let mut plans = Vec::with_capacity(scenes);
    let mut rejections = Rejections::default();
    for g in drawn {
        let g = g.map_err(|x| {
            let r = x.rejections;
            Failure::Failed(format!(
                "scene {} ({}): no valid draw in {} attempts (rejected: nodes {}, walk_start {}, scene_invalid {}, roomsim {}, locate {}); the last: {}. A generator bug, not data (spec/dataset.md §2.6)",
                x.index,
                scene_id(x.index),
                generator::MAX_ATTEMPTS,
                r.nodes,
                r.walk_start,
                r.scene,
                r.roomsim,
                r.locate,
                x.last
            ))
        })?;
        rejections.merge(&g.rejections);
        let mut rs = g.rs;
        impairments.apply(&mut rs);
        plans.push(Plan {
            index: g.index,
            scene_bytes: g.scene.to_json_pretty().into_bytes(),
            scene: g.scene,
            rs,
            meta: Meta {
                modes: g.modes.iter().map(|m| m.name().to_string()).collect(),
                objects_drawn: g.objects_drawn as u64,
                objects_placed: g.objects_placed as u64,
                openings_drawn: g.openings_drawn as u64,
                openings_placed: g.openings_placed as u64,
                l_shape: g.l_shape,
                wall_material: wall_material_name(g.wall_material).to_string(),
                attempts: u64::from(g.attempts),
            },
            split: Split::Draw(generator::split_draw(seed, g.index)),
        });
    }
    Ok((plans, rejections))
}

/// `--from DIR`'s manifest, checked: a finished `dataset v1`.
struct Source {
    dir: PathBuf,
    seed: u64,
    scenes: usize,
    seconds: u32,
    generator: Value,
    rejections: Rejections,
    entries: Vec<Value>,
}

/// Read `--from DIR`'s manifest; `--scenes`, `--seconds` and `--seed` are
/// taken from it and refused if given differently (§8).
fn read_source(dir: &str, opts: &Options) -> Result<Source, Failure> {
    let refused = |why: String| Failure::Refused(Refused::new("--from", why));
    if !Path::new(dir).is_dir() {
        return Err(refused(format!("{dir}: not a directory")));
    }
    let path = Path::new(dir).join("manifest.json");
    if !path.is_file() {
        return Err(refused(format!(
            "{dir}: no manifest.json, so not a finished dataset (spec/dataset.md §7.3)"
        )));
    }
    let text = fs::read_to_string(&path).map_err(|e| io_err(&path, e))?;
    let m: Value = serde_json::from_str(&text)
        .map_err(|e| refused(format!("{}: not JSON: {e}", path.display())))?;
    let field = |k: &str| {
        m.get(k)
            .ok_or_else(|| refused(format!("{}: no \"{k}\"", path.display())))
    };
    if field("label")?.as_str() != Some("SIM") || field("spec")?.as_str() != Some(SPEC) {
        return Err(refused(format!(
            "{}: not a SIM \"{SPEC}\" manifest",
            path.display()
        )));
    }
    let number = |k: &str| {
        field(k)?
            .as_u64()
            .ok_or_else(|| refused(format!("{}: \"{k}\" is not an integer", path.display())))
    };
    let seed = number("seed")?;
    let scenes = number("n_scenes")? as usize;
    let seconds = u32::try_from(number("seconds")?)
        .ok()
        .filter(|s| *s > 0 && s % 2 == 0 && *s <= MAX_SECONDS)
        .ok_or_else(|| refused(format!("{}: \"seconds\" is out of range", path.display())))?;
    let entries = field("scenes")?
        .as_array()
        .cloned()
        .ok_or_else(|| refused(format!("{}: \"scenes\" is not a list", path.display())))?;
    if !(1..=MAX_SCENES).contains(&scenes) || entries.len() != scenes {
        return Err(refused(format!(
            "{}: \"n_scenes\" says {scenes} but \"scenes\" lists {}",
            path.display(),
            entries.len()
        )));
    }
    let different = |given: Option<String>, name: &str, theirs: String| -> Result<(), Failure> {
        match given {
            Some(g) if g != theirs => Err(Failure::Refused(Refused::new(
                name,
                format!("{g} differs from {dir}'s manifest ({theirs}); with --from it is taken from there"),
            ))),
            _ => Ok(()),
        }
    };
    different(
        opts.scenes.map(|v| v.to_string()),
        "--scenes",
        scenes.to_string(),
    )?;
    different(
        opts.seconds.map(|v| v.to_string()),
        "--seconds",
        seconds.to_string(),
    )?;
    different(opts.seed.map(|v| v.to_string()), "--seed", seed.to_string())?;
    let rejections = Rejections::from_json(field("rejections")?)
        .ok_or_else(|| refused(format!("{}: \"rejections\" is malformed", path.display())))?;
    Ok(Source {
        dir: PathBuf::from(dir),
        seed,
        scenes,
        seconds,
        generator: field("generator")?.clone(),
        rejections,
        entries,
    })
}

/// Phase 1 with `--from`: read each scene file (its sha256 must be the
/// manifest's), its run config and its manifest entry; make the §2.6
/// checks again with the impairments on, as a fresh draw makes them (§8);
/// then apply the asked profile to the run config.
fn plan_from(src: &Source, i: usize, impairments: Impairments) -> Result<Plan, Failure> {
    let id = scene_id(i);
    let refused = |why: String| Failure::Refused(Refused::new("--from", format!("{id}: {why}")));
    let e = &src.entries[i];
    let get = |k: &str| {
        e.get(k)
            .ok_or_else(|| refused(format!("the manifest entry has no \"{k}\"")))
    };
    let num = |k: &str| {
        get(k)?
            .as_u64()
            .ok_or_else(|| refused(format!("\"{k}\" is not an integer")))
    };
    let text_of = |k: &str| {
        get(k)?
            .as_str()
            .map(str::to_string)
            .ok_or_else(|| refused(format!("\"{k}\" is not a string")))
    };
    if num("index")? != i as u64 || text_of("id")? != id {
        return Err(refused("the manifest lists the scenes out of order".into()));
    }
    let scene_path = src.dir.join("scenes").join(format!("{id}.json"));
    let bytes = fs::read(&scene_path).map_err(|e| io_err(&scene_path, e))?;
    if sha256_hex(&bytes) != text_of("scene_sha256")? {
        return Err(refused(
            "the scene file's sha256 is not the manifest's".into(),
        ));
    }
    let text =
        std::str::from_utf8(&bytes).map_err(|_| refused("the scene file is not UTF-8".into()))?;
    let scene = Scene::from_json_str(&id, text)
        .map_err(|e| refused(e.to_string().trim_end().to_string()))?;
    // A generated scene names no person and has no phone (§2.5); a scene
    // that does is not this command's, and no id is ever repeated.
    if scene.people.iter().any(|p| p.id.is_some()) || scene.phone.is_some() {
        return Err(refused(
            "a generated scene has no person ids and no phone".into(),
        ));
    }
    let cfg_path = src.dir.join("scenes").join(format!("{id}.roomsim.json"));
    let cfg_text = fs::read_to_string(&cfg_path).map_err(|e| io_err(&cfg_path, e))?;
    let mut rs = RoomSimConfig::from_json_str(&cfg_text)
        .map_err(|e| refused(format!("its Room Sim config: {e}")))?;
    Impairments::On.apply(&mut rs);
    if rs.duration_s != f64::from(src.seconds) + 0.5 {
        return Err(refused(format!(
            "its config's duration_s is {}, not seconds + 0.5",
            rs.duration_s
        )));
    }
    shard::tx_per_tick(&rs).map_err(refused)?;
    generator::check(&scene, &rs).map_err(|(_, why)| refused(why))?;
    impairments.apply(&mut rs);
    let openings = get("openings")?;
    let opening = |k: &str| {
        openings
            .get(k)
            .and_then(Value::as_u64)
            .ok_or_else(|| refused(format!("\"openings.{k}\" is not an integer")))
    };
    let modes = get("modes")?
        .as_array()
        .and_then(|a| {
            a.iter()
                .map(|m| m.as_str().map(str::to_string))
                .collect::<Option<Vec<_>>>()
        })
        .ok_or_else(|| refused("\"modes\" is not a list of strings".into()))?;
    Ok(Plan {
        index: i,
        scene,
        scene_bytes: bytes,
        rs,
        meta: Meta {
            modes,
            objects_drawn: num("objects_drawn")?,
            objects_placed: num("objects_placed")?,
            openings_drawn: opening("drawn")?,
            openings_placed: opening("placed")?,
            l_shape: get("l_shape")?
                .as_bool()
                .ok_or_else(|| refused("\"l_shape\" is not a boolean".into()))?,
            wall_material: text_of("wall_material")?,
            attempts: num("attempts")?,
        },
        split: Split::Given(
            text_of("split")?
                .parse::<SplitName>()
                .map_err(|()| refused("\"split\" is not train, val or test".into()))?
                .0
                .to_string(),
        ),
    })
}

/// A split's name, one of `train`, `val`, `test`.
struct SplitName(&'static str);

impl std::str::FromStr for SplitName {
    type Err = ();
    fn from_str(s: &str) -> Result<SplitName, ()> {
        SPLITS
            .iter()
            .find(|&&n| n == s)
            .map(|&n| SplitName(n))
            .ok_or(())
    }
}

/// The three splits, in the manifest's order.
const SPLITS: [&str; 3] = ["train", "val", "test"];

/// §7.3: by scene, stratified by `(n_people, l_shape)`; within a stratum
/// the scenes in the order of their split draw (ties by index), the first
/// `round(0.1 n)` to test (at least 1 when the stratum has 3 or more), the
/// next `round(0.1 n)` to val, the rest to train. `round` is half to even
/// (spec/roomsim.md's and spec/locate.md's `round`), on the double `0.1 n`.
fn assign_splits(scenes: &[(usize, bool, f64)]) -> Vec<&'static str> {
    let mut strata: BTreeMap<(usize, bool), Vec<usize>> = BTreeMap::new();
    for (i, &(people, l_shape, _)) in scenes.iter().enumerate() {
        strata.entry((people, l_shape)).or_default().push(i);
    }
    let mut out = vec!["train"; scenes.len()];
    for members in strata.values_mut() {
        members.sort_by(|&a, &b| scenes[a].2.total_cmp(&scenes[b].2).then(a.cmp(&b)));
        let n = members.len();
        // Python's round(0.1 * n) on the same double: half to even.
        let tenth = (0.1 * n as f64).round_ties_even() as usize;
        let n_test = if n >= 3 { tenth.max(1) } else { tenth };
        for (rank, &i) in members.iter().enumerate() {
            out[i] = if rank < n_test {
                "test"
            } else if rank < n_test + tenth {
                "val"
            } else {
                "train"
            };
        }
    }
    out
}

// ---------------------------------------------------------------------
// The size check (§8)
// ---------------------------------------------------------------------

/// The "Available" column of `df -Pk` (POSIX output, 1024-byte blocks),
/// in bytes: the field before the first `NN%` capacity field of the
/// second line, so a device name with spaces still parses.
fn parse_df(stdout: &str) -> Option<u64> {
    let line = stdout.lines().nth(1)?;
    let fields: Vec<&str> = line.split_whitespace().collect();
    let cap = fields.iter().position(|f| {
        f.strip_suffix('%')
            .is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()))
    })?;
    let avail: u64 = fields.get(cap.checked_sub(1)?)?.parse().ok()?;
    avail.checked_mul(1024)
}

/// The free bytes on the filesystem that will hold `out`: `df -Pk` on the
/// nearest existing ancestor (std has no safe statvfs), or the value of
/// [`FREE_BYTES_ENV`] when set (tests).
fn free_bytes(out: &Path) -> Result<u64, Failure> {
    if let Ok(v) = std::env::var(FREE_BYTES_ENV) {
        return v
            .trim()
            .parse::<u64>()
            .map_err(|_| Failure::Io(format!("{FREE_BYTES_ENV}={v:?} is not a byte count")));
    }
    let mut probe = out.to_path_buf();
    while !probe.exists() {
        match probe.parent() {
            Some(p) if !p.as_os_str().is_empty() => probe = p.to_path_buf(),
            _ => {
                probe = PathBuf::from(".");
                break;
            }
        }
    }
    let output = std::process::Command::new("df")
        .arg("-Pk")
        .arg(&probe)
        .output()
        .map_err(|e| {
            Failure::Io(format!(
                "df -Pk {}: {e} (the free-space check needs df)",
                probe.display()
            ))
        })?;
    if !output.status.success() {
        return Err(Failure::Io(format!(
            "df -Pk {}: {}",
            probe.display(),
            String::from_utf8_lossy(&output.stderr).trim()
        )));
    }
    parse_df(&String::from_utf8_lossy(&output.stdout)).ok_or_else(|| {
        Failure::Io(format!(
            "df -Pk {}: could not read the available space",
            probe.display()
        ))
    })
}

/// §8: the estimate must not exceed the free space less 5 GB.
fn size_check(estimate: u64, free: u64) -> Result<(), Refused> {
    if estimate > free.saturating_sub(RESERVE_BYTES) {
        return Err(Refused::new(
            "--out",
            format!(
                "the dataset needs about {} ({estimate} bytes) and the target filesystem has {} free; at least 5 GB must stay free (spec/dataset.md §8)",
                gb(estimate),
                gb(free)
            ),
        ));
    }
    Ok(())
}

/// `12.34 GB` (10^9 bytes).
fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

// ---------------------------------------------------------------------
// The output directory
// ---------------------------------------------------------------------

/// Refuse, before any work, an `--out` that is not a directory, or is a
/// non-empty one without `--overwrite`; and a `--from` that `--overwrite`
/// would destroy: `--out` itself or anything in its `scenes/` or `shards/`.
fn check_out(opts: &Options) -> Result<(), Failure> {
    let out = Path::new(&opts.out);
    match fs::metadata(out) {
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(()),
        Err(e) => return Err(io_err(out, e)),
        Ok(m) if !m.is_dir() => {
            return Err(Refused::new(
                "--out",
                format!("{}: exists and is not a directory", opts.out),
            )
            .into())
        }
        Ok(_) => {}
    }
    if let Some(from) = &opts.from {
        if let (Ok(out), Ok(from)) = (fs::canonicalize(out), fs::canonicalize(from)) {
            if from == out
                || from.starts_with(out.join("scenes"))
                || from.starts_with(out.join("shards"))
            {
                return Err(Refused::new(
                    "--from",
                    "is --out or lies in its scenes/ or shards/, which this run replaces; write the new dataset elsewhere",
                )
                .into());
            }
        }
    }
    let non_empty = fs::read_dir(out)
        .map_err(|e| io_err(out, e))?
        .next()
        .is_some();
    if non_empty && !opts.overwrite {
        return Err(Refused::new(
            "--out",
            format!(
                "{} is not empty; give --overwrite to replace its manifest.json, scenes/ and shards/ (nothing else there is touched)",
                opts.out
            ),
        )
        .into());
    }
    Ok(())
}

/// Make `--out`, `scenes/` and `shards/`. With `--overwrite`, first remove
/// `manifest.json` (so the directory is never a finished dataset of mixed
/// files), then `scenes/` and `shards/`, and nothing else.
fn prepare_out(out: &Path, overwrite: bool) -> Result<(), Failure> {
    fs::create_dir_all(out).map_err(|e| io_err(out, e))?;
    if overwrite {
        let manifest = out.join("manifest.json");
        match fs::remove_file(&manifest) {
            Err(e) if e.kind() != ErrorKind::NotFound => return Err(io_err(&manifest, e)),
            _ => {}
        }
        for sub in ["scenes", "shards"] {
            let d = out.join(sub);
            match fs::remove_dir_all(&d) {
                Err(e) if e.kind() != ErrorKind::NotFound => return Err(io_err(&d, e)),
                _ => {}
            }
        }
    }
    for sub in ["scenes", "shards"] {
        let d = out.join(sub);
        fs::create_dir(&d).map_err(|e| io_err(&d, e))?;
    }
    Ok(())
}

/// `path` with `.tmp` appended: where a file is written before its rename.
fn temp_path(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

/// Write `bytes` to a temporary name, then rename it to `path` (§9).
fn write_file(path: &Path, bytes: &[u8]) -> Result<(), Failure> {
    let tmp = temp_path(path);
    fs::write(&tmp, bytes).map_err(|e| io_err(&tmp, e))?;
    fs::rename(&tmp, path).map_err(|e| io_err(path, e))
}

/// `value` as pretty JSON with a trailing newline.
fn pretty(value: &impl serde::Serialize) -> Vec<u8> {
    let mut s = serde_json::to_string_pretty(value).expect("a config always serializes");
    s.push('\n');
    s.into_bytes()
}

// ---------------------------------------------------------------------
// Phase 2: one scene's files
// ---------------------------------------------------------------------

/// What phase 2 made of a scene.
#[derive(Debug, Clone)]
struct Done {
    shard_bytes: u64,
    shard_sha256: String,
    scene_sha256: String,
    /// The scene file and its two configs.
    file_bytes: u64,
    n_links: usize,
    labels: usize,
    windows: usize,
    windows_wlm: usize,
    /// Some link's calibration `null_sigma > 0` (§8 `baseline_valid`).
    calibrated_noise: bool,
}

/// Run §3-§6 on one scene and write its four files (§7.1), each to a
/// temporary name, then renamed.
fn produce(plan: &Plan, out: &Path, seconds: u32) -> Result<Done, Failure> {
    let id = scene_id(plan.index);
    let scene_sha256 = sha256_hex(&plan.scene_bytes);
    let built = shard::build(
        &plan.scene,
        &plan.rs,
        seconds,
        plan.index,
        &id,
        &scene_sha256,
    )
    .map_err(|e| Failure::Failed(format!("scene {} ({id}): {e}", plan.index)))?;
    let scenes = out.join("scenes");
    let roomsim = pretty(&plan.rs);
    let locate = pretty(&locate_config_json(&built.locate_config));
    write_file(&scenes.join(format!("{id}.json")), &plan.scene_bytes)?;
    write_file(&scenes.join(format!("{id}.roomsim.json")), &roomsim)?;
    write_file(&scenes.join(format!("{id}.locate.json")), &locate)?;
    let shard = out.join("shards").join(format!("{id}.npz"));
    let tmp = temp_path(&shard);
    let file = File::create(&tmp).map_err(|e| io_err(&tmp, e))?;
    let (shard_bytes, shard_sha256) =
        npz::write_npz(BufWriter::new(file), &built.arrays).map_err(|e| io_err(&tmp, e))?;
    fs::rename(&tmp, &shard).map_err(|e| io_err(&shard, e))?;
    Ok(Done {
        shard_bytes,
        shard_sha256,
        scene_sha256,
        file_bytes: (plan.scene_bytes.len() + roomsim.len() + locate.len()) as u64,
        n_links: built.n_links,
        labels: built.labels,
        windows: built.windows,
        windows_wlm: built.windows_wlm,
        calibrated_noise: built.calibrated_noise,
    })
}

/// Progress on stderr every 5% of the scenes (§8).
struct Progress {
    total: usize,
    done: AtomicUsize,
    printed: Mutex<usize>,
    started: Instant,
}

impl Progress {
    fn one_more(&self) {
        let done = self.done.fetch_add(1, Ordering::SeqCst) + 1;
        let step = done * 20 / self.total;
        let mut printed = self.printed.lock().unwrap_or_else(|e| e.into_inner());
        if step > *printed {
            *printed = step;
            let elapsed = self.started.elapsed().as_secs_f64();
            let eta = elapsed * (self.total - done) as f64 / done as f64;
            note(&format!(
                "SIM  gen-dataset: {done} of {} scenes ({}%), {elapsed:.0} s elapsed, ETA {eta:.0} s",
                self.total,
                100 * done / self.total
            ));
        }
    }
}

// ---------------------------------------------------------------------
// The manifest (§7.3) and the summary (§8)
// ---------------------------------------------------------------------

/// `{"value": count}` with the values in order.
fn histogram<K: Ord + ToString>(values: impl Iterator<Item = K>) -> BTreeMap<K, usize> {
    let mut h = BTreeMap::new();
    for v in values {
        *h.entry(v).or_insert(0) += 1;
    }
    h
}

fn histogram_json<K: Ord + ToString>(h: &BTreeMap<K, usize>) -> Value {
    Value::Object(h.iter().map(|(k, n)| (k.to_string(), json!(n))).collect())
}

/// `k of n (p%)`.
fn of(k: usize, n: usize) -> String {
    if n == 0 {
        format!("{k} of {n}")
    } else {
        format!("{k} of {n} ({:.1}%)", 100.0 * k as f64 / n as f64)
    }
}

/// The command after its arguments are parsed. Returns the summary's
/// lines for stdout.
fn run(opts: &Options) -> Result<Vec<String>, Failure> {
    let started = Instant::now();
    let out = PathBuf::from(&opts.out);
    check_out(opts)?;
    let impairments = if opts.no_impair {
        Impairments::HardwareOff
    } else {
        Impairments::On
    };
    let source = opts
        .from
        .as_deref()
        .map(|d| read_source(d, opts))
        .transpose()?;
    let (scenes, seconds, seed) = match &source {
        Some(s) => (s.scenes, s.seconds, s.seed),
        None => (
            opts.scenes.unwrap_or(DEFAULT_SCENES),
            opts.seconds.unwrap_or(DEFAULT_SECONDS),
            opts.seed.unwrap_or(0),
        ),
    };
    let from = source.as_ref().map(|s| {
        fs::canonicalize(&s.dir)
            .unwrap_or_else(|_| s.dir.clone())
            .display()
            .to_string()
    });
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(opts.threads)
        .build()
        .map_err(|e| Failure::Io(format!("--threads {}: {e}", opts.threads)))?;

    // Phase 1.
    let (plans, rejections) = match &source {
        Some(src) => {
            let plans = pool.install(|| {
                (0..src.scenes)
                    .into_par_iter()
                    .map(|i| plan_from(src, i, impairments))
                    .collect::<Result<Vec<_>, _>>()
            })?;
            (plans, src.rejections)
        }
        None => pool.install(|| draw_all(seed, scenes, seconds, impairments))?,
    };
    note(&format!(
        "SIM  gen-dataset: {} {scenes} scenes in {:.1} s ({} redraws)",
        if source.is_some() {
            "read and checked"
        } else {
            "drew and checked"
        },
        started.elapsed().as_secs_f64(),
        rejections.total()
    ));
    let calib_s = LocateConfig::default().calib_s;
    let estimate: u64 = plans
        .iter()
        .map(|p| shard::estimate_bytes(p.scene.nodes.len(), seconds, calib_s))
        .sum();
    let free = free_bytes(&out)?;
    note(&format!(
        "SIM  gen-dataset: estimated size {} ({estimate} bytes) for {scenes} scenes of {seconds} s; {} free on the target filesystem",
        gb(estimate),
        gb(free)
    ));
    size_check(estimate, free)?;
    prepare_out(&out, opts.overwrite)?;

    // Phase 2.
    let progress = Progress {
        total: plans.len(),
        done: AtomicUsize::new(0),
        printed: Mutex::new(0),
        started: Instant::now(),
    };
    let done: Vec<Done> = pool.install(|| {
        plans
            .par_iter()
            .map(|p| {
                let d = produce(p, &out, seconds);
                progress.one_more();
                d
            })
            .collect::<Result<Vec<_>, _>>()
    })?;

    // Splits: drawn (§7.3), or `--from`'s.
    let splits: Vec<String> = match plans.first().map(|p| &p.split) {
        Some(Split::Given(_)) => plans
            .iter()
            .map(|p| match &p.split {
                Split::Given(s) => s.clone(),
                Split::Draw(_) => unreachable!("one source per run"),
            })
            .collect(),
        _ => {
            let keys: Vec<(usize, bool, f64)> = plans
                .iter()
                .map(|p| match p.split {
                    Split::Draw(u) => (p.scene.people.len(), p.meta.l_shape, u),
                    Split::Given(_) => unreachable!("one source per run"),
                })
                .collect();
            assign_splits(&keys)
                .into_iter()
                .map(str::to_string)
                .collect()
        }
    };

    let entries: Vec<Value> = plans
        .iter()
        .zip(&done)
        .zip(&splits)
        .map(|((p, d), split)| {
            let id = scene_id(p.index);
            json!({
                "index": p.index,
                "id": id,
                "split": split,
                "shard": format!("shards/{id}.npz"),
                "shard_sha256": d.shard_sha256,
                "scene_sha256": d.scene_sha256,
                "n_nodes": p.scene.nodes.len(),
                "n_links": d.n_links,
                "n_people": p.scene.people.len(),
                "modes": p.meta.modes,
                "objects_drawn": p.meta.objects_drawn,
                "objects_placed": p.meta.objects_placed,
                "openings": {"drawn": p.meta.openings_drawn, "placed": p.meta.openings_placed},
                "l_shape": p.meta.l_shape,
                "wall_material": p.meta.wall_material,
                "attempts": p.meta.attempts,
            })
        })
        .collect();
    let mut split_lists: BTreeMap<&str, Vec<usize>> =
        SPLITS.iter().map(|&s| (s, Vec::new())).collect();
    for (p, s) in plans.iter().zip(&splits) {
        split_lists.entry(s.as_str()).or_default().push(p.index);
    }
    let split_counts: Map<String, Value> = SPLITS
        .iter()
        .map(|&s| {
            let members: Vec<&Plan> = plans.iter().zip(&splits).filter(|(_, x)| *x == s).map(|(p, _)| p).collect();
            let block = json!({
                "n_scenes": members.len(),
                "n_people": histogram_json(&histogram(members.iter().map(|p| p.scene.people.len()))),
                "n_nodes": histogram_json(&histogram(members.iter().map(|p| p.scene.nodes.len()))),
                "l_shape": histogram_json(&histogram(members.iter().map(|p| p.meta.l_shape))),
            });
            (s.to_string(), block)
        })
        .collect();
    let labels: usize = done.iter().map(|d| d.labels).sum();
    let bytes: u64 = done.iter().map(|d| d.shard_bytes + d.file_bytes).sum();
    let shard_bytes: u64 = done.iter().map(|d| d.shard_bytes).sum();
    let mut command =
        format!("saans gen-dataset --scenes {scenes} --seconds {seconds} --seed {seed}");
    if opts.no_impair {
        command.push_str(" --no-impair");
    }
    if let Some(f) = &from {
        command.push_str(&format!(" --from {f}"));
    }
    // §8: every scene's calibration has a link with null_sigma > 0, so
    // WLM can publish (RSSI jitter keeps it so with the hardware off).
    let baseline_valid = done.iter().all(|d| d.calibrated_noise);
    let wall_s = started.elapsed().as_secs_f64();
    let manifest = json!({
        "label": "SIM",
        "spec": SPEC,
        "command": command,
        "seed": seed,
        "n_scenes": scenes,
        "seconds": seconds,
        "label_hz": shard::LABEL_HZ,
        // Room Sim's master switch, true in both profiles (§8); the
        // profile is `impairments`.
        "impair": true,
        "impairments": impairments.name(),
        "impairment_switches": impairments.switches_json(),
        "baseline_valid": baseline_valid,
        "from": from,
        "roomsim_defaults": serde_json::to_value(RoomSimConfig::default()).expect("a config always serializes"),
        "locate_config": locate_config_json(&LocateConfig::default()),
        "generator": source.as_ref().map_or_else(generator::generator_json, |s| s.generator.clone()),
        "rejections": rejections.to_json(),
        "splits": split_lists,
        "split_counts": split_counts,
        "scenes": entries,
        "totals": {"labels": labels, "bytes": bytes, "wall_s": wall_s},
    });
    // Written last (§7.3).
    write_file(&out.join("manifest.json"), &pretty(&manifest))?;

    // The summary (§8), every line SIM.
    let n = plans.len();
    let with_people = plans.iter().filter(|p| !p.scene.people.is_empty()).count();
    // `value count, value count` in the histogram's order.
    fn line<K: fmt::Display>(h: &BTreeMap<K, usize>) -> String {
        h.iter()
            .map(|(k, v)| format!("{k} {v}"))
            .collect::<Vec<_>>()
            .join(", ")
    }
    let people = histogram(plans.iter().map(|p| p.scene.people.len()));
    let nodes = histogram(plans.iter().map(|p| p.scene.nodes.len()));
    let materials = histogram(plans.iter().map(|p| p.meta.wall_material.clone()));
    // WLM at the end ticks of the 2 s windows: in occupied scenes its
    // coverage, in empty ones its false alarms (§8).
    let wlm_windows = |occupied: bool| {
        plans
            .iter()
            .zip(&done)
            .filter(|(p, _)| p.scene.people.is_empty() != occupied)
            .fold((0, 0), |(w, k), (_, d)| (w + d.windows, k + d.windows_wlm))
    };
    let (occ_windows, occ_wlm) = wlm_windows(true);
    let (empty_windows, empty_wlm) = wlm_windows(false);
    let uncalibrated = done.iter().filter(|d| !d.calibrated_noise).count();
    let mut lines = vec![
        format!(
            "saans gen-dataset (spec/dataset.md): {n} generated scenes x {seconds} s, seed {seed}, impairments {}{}; rooms drawn at random, simulated people in simulated WiFi, a model, not a measurement",
            match impairments {
                Impairments::On => "on",
                Impairments::HardwareOff =>
                    "hardware_off (no loss, AGC jumps, device gain, ripple, CPE or timing offsets; the receiver's noise kept)",
            },
            from.as_ref().map_or_else(String::new, |f| format!(", scenes from {f}"))
        ),
        format!(
            "scenes: {n} (train {}, val {}, test {}); labels: {labels} ticks at {} Hz; bytes: {bytes} ({}), {shard_bytes} of them in {n} shards",
            split_lists["train"].len(),
            split_lists["val"].len(),
            split_lists["test"].len(),
            shard::LABEL_HZ,
            gb(bytes)
        ),
        format!("scenes with people: {}", of(with_people, n)),
        format!("scenes by people count: {}", line(&people)),
        format!("scenes by node count: {}", line(&nodes)),
        format!("scenes by wall material: {}", line(&materials)),
        format!(
            "WLM (baseline) published at the end tick of 2 s windows: in scenes with people (coverage) {}; in empty scenes (false alarms) {}",
            of(occ_wlm, occ_windows),
            of(empty_wlm, empty_windows)
        ),
        if baseline_valid {
            "baseline_valid true: every scene's empty-room calibration has a link with null sigma > 0".to_string()
        } else {
            format!(
                "baseline_valid false: in {} scenes the empty-room calibration has null sigma 0 on every link, so WLM can never publish there",
                of(uncalibrated, n)
            )
        },
    ];
    lines.push(format!(
        "wall time {wall_s:.1} s; manifest {}",
        out.join("manifest.json").display()
    ));
    Ok(lines.into_iter().map(|l| format!("SIM  {l}")).collect())
}

/// `saans gen-dataset …`; `args[0]` is `gen-dataset`. Exit code 0, 2 on a
/// refusal (naming the parameter), 1 on an I/O error or a failed scene.
pub fn main(args: &[String]) -> ExitCode {
    let opts = match Options::parse(args.get(1..).unwrap_or_default()) {
        Ok(o) => o,
        Err(e) => {
            note(&format!("gen-dataset: {e}\nusage: {USAGE}"));
            return ExitCode::from(2);
        }
    };
    match run(&opts) {
        Ok(lines) => {
            let mut stdout = io::stdout().lock();
            let written = lines
                .iter()
                .try_for_each(|l| writeln!(stdout, "{l}"))
                .and_then(|()| stdout.flush());
            if written.is_err() {
                return ExitCode::FAILURE;
            }
            ExitCode::SUCCESS
        }
        Err(Failure::Refused(e)) => {
            note(&format!("gen-dataset: {e}"));
            ExitCode::from(2)
        }
        Err(Failure::Io(e) | Failure::Failed(e)) => {
            note(&format!("gen-dataset: {e}"));
            ExitCode::FAILURE
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(args: &[&str]) -> Result<Options, Refused> {
        let args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        Options::parse(&args)
    }

    #[test]
    fn the_defaults_and_every_option_parse() {
        let o = parse(&["--out", "d"]).unwrap();
        assert_eq!(
            (o.scenes, o.seconds, o.seed, o.from.as_deref()),
            (None, None, None, None)
        );
        assert!(!o.no_impair && !o.overwrite && o.threads >= 1);
        let o = parse(&[
            "--overwrite",
            "--seconds",
            "600",
            "--scenes",
            "100000",
            "--seed",
            "18446744073709551615",
            "--threads",
            "3",
            "--no-impair",
            "--from",
            "src",
            "--out",
            "d",
        ])
        .unwrap();
        assert_eq!(
            o,
            Options {
                out: "d".into(),
                scenes: Some(MAX_SCENES),
                seconds: Some(600),
                seed: Some(u64::MAX),
                threads: 3,
                no_impair: true,
                from: Some("src".into()),
                overwrite: true,
            }
        );
    }

    #[test]
    fn a_bad_invocation_names_its_parameter() {
        for (args, param) in [
            (vec![], "--out"),
            (vec!["d"], "--out"),
            (vec!["--out"], "--out"),
            (vec!["--out", "a", "--out", "b"], "--out"),
            (vec!["--out", "d", "--scenes", "0"], "--scenes"),
            (vec!["--out", "d", "--scenes", "100001"], "--scenes"),
            (vec!["--out", "d", "--scenes", "ten"], "--scenes"),
            (vec!["--out", "d", "--seconds", "3"], "--seconds"),
            (vec!["--out", "d", "--seconds", "0"], "--seconds"),
            (vec!["--out", "d", "--seconds", "602"], "--seconds"),
            (vec!["--out", "d", "--seconds", "2.0"], "--seconds"),
            (vec!["--out", "d", "--seed", "-1"], "--seed"),
            (vec!["--out", "d", "--threads", "0"], "--threads"),
            (
                vec!["--out", "d", "--no-impair", "--no-impair"],
                "--no-impair",
            ),
            (
                vec!["--out", "d", "--overwrite", "--overwrite"],
                "--overwrite",
            ),
            (vec!["--out", "d", "--from"], "--from"),
            (vec!["--out", ""], "--out"),
            (vec!["--out", "d", "--from", ""], "--from"),
            (vec!["--out", "d", "--sead", "1"], "--sead"),
        ] {
            let e = parse(&args).expect_err(&format!("{args:?}"));
            assert_eq!(e.param, param, "{args:?}: {e}");
        }
    }

    #[test]
    fn df_output_parses_to_bytes() {
        let gnu = "Filesystem     1024-blocks       Used Available Capacity Mounted on\n/dev/nvme0n1p4  1927948020 1677360292 152579564      92% /\n";
        assert_eq!(parse_df(gnu), Some(152_579_564 * 1024));
        let spaced = "Filesystem 1024-blocks Used Available Capacity Mounted on\nmap auto home 10 5 5 50% /System/Volumes/Data/home\n";
        assert_eq!(parse_df(spaced), Some(5 * 1024));
        assert_eq!(parse_df("Filesystem\n"), None);
        assert_eq!(parse_df(""), None);
    }

    #[test]
    fn the_size_check_keeps_five_gb_free() {
        assert!(size_check(1_000, 5_000_001_000).is_ok());
        let e = size_check(1_001, 5_000_001_000).unwrap_err();
        assert_eq!(e.param, "--out");
        assert!(
            size_check(1, 4_000_000_000).is_err(),
            "less than the reserve"
        );
        assert!(size_check(0, 5_000_000_000).is_ok());
    }

    #[test]
    fn splits_are_stratified_and_ordered_by_the_split_draw() {
        // 25 one-person rectangles: round(2.5) = 2 (half to even) to test,
        // 2 to val, 21 to train; the smallest draws go to test, the next
        // to val.
        let mut keys: Vec<(usize, bool, f64)> =
            (0..25).map(|i| (1, false, 1.0 - i as f64 / 25.0)).collect();
        // Two empty L rooms: too few for a test scene.
        keys.push((0, true, 0.1));
        keys.push((0, true, 0.2));
        // Three empty rectangles: round(0.3) = 0, but at least 1 test.
        keys.extend([(0, false, 0.9), (0, false, 0.3), (0, false, 0.6)]);
        let s = assign_splits(&keys);
        let of_split = |name: &str| {
            (0..keys.len())
                .filter(|&i| s[i] == name)
                .collect::<Vec<_>>()
        };
        assert_eq!(of_split("test"), vec![23, 24, 28]);
        assert_eq!(of_split("val"), vec![21, 22]);
        assert_eq!(of_split("train").len(), 25);
        // Equal draws are ordered by index.
        let tied = vec![(2, false, 0.5); 10];
        let s = assign_splits(&tied);
        assert_eq!(s[0], "test");
        assert_eq!(s[1], "val");
        assert!(s[2..].iter().all(|&x| x == "train"));
        // Half to even: round(1.5) = 2, round(0.5) = 0 (then the minimum
        // of 1 test), round(3.5) = 4.
        for (n, test, val) in [(15, 2, 2), (5, 1, 0), (35, 4, 4), (45, 4, 4)] {
            let s = assign_splits(&vec![(3, true, 0.5); n]);
            let count = |x: &str| s.iter().filter(|&&y| y == x).count();
            assert_eq!((count("test"), count("val")), (test, val), "n = {n}");
        }
    }
}
