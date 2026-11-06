//! SIM: spec/locate.md §12a's measured numbers, printed for each shipped
//! scene (spec/scenes/*.json) over a few seeds. These are reported
//! numbers, not pass/fail tests; every one of them is SIM (Room Sim input,
//! scene geometry) and none is a measurement.
//!
//! ```text
//! cargo run --release -p saans-locate --features roomsim --example locate_measure \
//!     [-- --duration 120 --seeds 0,1,2,3,4 --scene bedroom]
//! ```
//!
//! Per scene and seed, three Room Sim runs (each with its own empty-room
//! calibration, §10.2), all with the default Room Sim and Locate configs:
//! - `empty`: `n_people = 0`. Every published WLM or RTI estimate is a
//!   false alarm: counted as published ticks per hour and as events (runs
//!   of consecutive published ticks) per hour. The phone lies uncarried at
//!   the scene's phone start.
//! - `scene`: the scene's people as shipped (the phone carried by
//!   `people[0]`, who may never walk).
//! - `walker`: the scene's people plus one walker (Room Sim's entry start)
//!   who carries the phone.
//!
//! On occupied runs, WLM and RTI, split by whether anyone is walking at the
//! tick (walk, still, all): report coverage (published / ticks) beside the
//! empty run's publish fraction; median and p90 error to the nearest
//! person and, on walking ticks, to the walker (the nearest walking
//! person); every published radius's median and p90; radius coverage (the
//! nearest person within `radius_m`); and the same error and coverage for a
//! **fixed reference point** (the footprint's area centroid) on the same
//! ticks with the same radius: the score of an estimator that knows
//! nothing. Then WLM and RTI on the ticks where both published, per split.
//!
//! The phone, per method: coverage (published / fired epochs), error to the
//! phone's true `(x, y)`, radius, radius coverage and the reference point's
//! figures; each line says whether the phone moved (its path length over
//! the fired epochs of each run), and runs where it did not are also given
//! one by one (a still phone's shadowing and offsets are constant for a
//! run, so its epochs are one sample, not many).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;

use saans_locate::phone::Config;
use saans_locate::pipeline::{run_roomsim, RoomSimRun};
use saans_locate::radius::Outcome;
use saans_roomsim::Config as RoomSimConfig;
use saans_scene::Scene;

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
enum Variant {
    Empty,
    Scene,
    Walker,
}

impl Variant {
    fn name(self) -> &'static str {
        match self {
            Variant::Empty => "empty",
            Variant::Scene => "scene",
            Variant::Walker => "walker",
        }
    }
}

struct Job {
    scene_name: String,
    scene: Scene,
    seed: u64,
    variant: Variant,
}

/// One published estimate, scored.
#[derive(Clone, Copy)]
struct Scored {
    /// Distance to the truth (nearest person, or the phone).
    err: f64,
    /// On walking ticks: distance to the nearest walking person.
    walker_err: Option<f64>,
    radius: f64,
    inside: bool,
    /// The fixed reference point's distance to the same truth.
    ref_err: f64,
    /// The reference point's truth within the estimate's radius.
    ref_inside: bool,
}

/// One estimator over a set of ticks or epochs.
#[derive(Default, Clone)]
struct Stats {
    total: usize,
    published: Vec<Scored>,
    reasons: BTreeMap<&'static str, usize>,
}

/// The truth of one tick or epoch, as distances from a point.
struct Truth<'a> {
    nearest: &'a dyn Fn(f64, f64) -> f64,
    walker: Option<&'a dyn Fn(f64, f64) -> f64>,
    reference: (f64, f64),
}

impl Stats {
    fn add(&mut self, out: &Outcome, truth: &Truth<'_>) {
        self.total += 1;
        match out {
            Outcome::Estimate(e) => {
                let err = (truth.nearest)(e.x, e.y);
                let ref_err = (truth.nearest)(truth.reference.0, truth.reference.1);
                self.published.push(Scored {
                    err,
                    walker_err: truth.walker.map(|w| w(e.x, e.y)),
                    radius: e.radius_m,
                    inside: err <= e.radius_m,
                    ref_err,
                    ref_inside: ref_err <= e.radius_m,
                });
            }
            Outcome::Abstain(r) => *self.reasons.entry(r.as_str()).or_default() += 1,
        }
    }

    fn merge(&mut self, other: &Stats) {
        self.total += other.total;
        self.published.extend_from_slice(&other.published);
        for (k, v) in &other.reasons {
            *self.reasons.entry(k).or_default() += v;
        }
    }

    fn n(&self) -> usize {
        self.published.len()
    }

    fn errors(&self) -> Vec<f64> {
        self.published.iter().map(|s| s.err).collect()
    }

    fn inside_pct(&self) -> f64 {
        pct(self.published.iter().filter(|s| s.inside).count(), self.n())
    }

    /// `published n/T (c%)` and, for published estimates, the error (to
    /// `truth_name`), walker error, radius, coverage and the reference
    /// point's figures.
    fn line(&self, empty_fraction: Option<f64>, truth_name: &str) -> String {
        let n = self.n();
        if self.total == 0 {
            return "no ticks".to_string();
        }
        let mut s = format!(
            "published {n:>5}/{:<5} ({:5.1}%",
            self.total,
            pct(n, self.total)
        );
        if let Some(f) = empty_fraction {
            s += &format!("; empty run {f:5.1}%");
        }
        s += ")";
        if n > 0 {
            let (em, ep) = median_p90(&self.errors());
            s += &format!("  err to {truth_name} median {em:5.2} p90 {ep:5.2} m");
            let walker: Vec<f64> = self.published.iter().filter_map(|x| x.walker_err).collect();
            if !walker.is_empty() {
                let (wm, wp) = median_p90(&walker);
                s += &format!(", to walker median {wm:5.2} p90 {wp:5.2} m");
            }
            let radii: Vec<f64> = self.published.iter().map(|x| x.radius).collect();
            let (rm, rp) = median_p90(&radii);
            s += &format!(
                "  radius median {rm:5.2} p90 {rp:5.2} m  inside radius {:5.1}%",
                self.inside_pct()
            );
            let refs: Vec<f64> = self.published.iter().map(|x| x.ref_err).collect();
            let (cm, cp) = median_p90(&refs);
            s += &format!(
                "  | reference point: err median {cm:5.2} p90 {cp:5.2} m, inside the same radius {:5.1}%",
                pct(self.published.iter().filter(|x| x.ref_inside).count(), n)
            );
        }
        if !self.reasons.is_empty() {
            let r: Vec<String> = self
                .reasons
                .iter()
                .map(|(k, v)| format!("{k} {v}"))
                .collect();
            s += &format!("  abstain: {}", r.join(", "));
        }
        s
    }
}

fn pct(k: usize, n: usize) -> f64 {
    if n == 0 {
        0.0
    } else {
        100.0 * k as f64 / n as f64
    }
}

/// Median and p90 (nearest rank) of a non-empty sample.
fn median_p90(xs: &[f64]) -> (f64, f64) {
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    let med = if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    };
    let rank = ((0.9 * n as f64).ceil() as usize).clamp(1, n);
    (med, v[rank - 1])
}

fn dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

/// The footprint's area centroid: the polygon's (shoelace) for an outlined
/// room, else the rectangle's centre.
fn footprint_centroid(scene: &Scene) -> (f64, f64) {
    let Some(poly) = scene.room.polygon.as_ref().filter(|p| p.len() >= 3) else {
        return (0.5 * scene.room.w_m, 0.5 * scene.room.d_m);
    };
    let (mut a2, mut cx, mut cy) = (0.0, 0.0, 0.0);
    for (i, &(x0, y0)) in poly.iter().enumerate() {
        let (x1, y1) = poly[(i + 1) % poly.len()];
        let c = x0 * y1 - x1 * y0;
        a2 += c;
        cx += (x0 + x1) * c;
        cy += (y0 + y1) * c;
    }
    (cx / (3.0 * a2), cy / (3.0 * a2))
}

const WALK: usize = 0;
const STILL: usize = 1;
const ALL: usize = 2;
const CLASSES: [(usize, &str); 3] = [(WALK, "walk "), (STILL, "still"), (ALL, "all  ")];

/// WLM and RTI on the ticks where both published.
#[derive(Default, Clone)]
struct Both {
    wlm: Vec<f64>,
    rti: Vec<f64>,
    wlm_walker: Vec<f64>,
    rti_walker: Vec<f64>,
}

impl Both {
    fn merge(&mut self, o: &Both) {
        self.wlm.extend_from_slice(&o.wlm);
        self.rti.extend_from_slice(&o.rti);
        self.wlm_walker.extend_from_slice(&o.wlm_walker);
        self.rti_walker.extend_from_slice(&o.rti_walker);
    }
}

#[derive(Default, Clone)]
struct PersonAgg {
    wlm: [Stats; 3],
    rti: [Stats; 3],
    both: [Both; 3],
}

/// An empty run's false alarms of one estimator.
#[derive(Default, Clone)]
struct FalseAlarms {
    published: usize,
    events: usize,
    per_run_events: Vec<usize>,
}

impl FalseAlarms {
    fn of(flags: &[bool]) -> Self {
        let mut events = 0;
        let mut prev = false;
        for &f in flags {
            if f && !prev {
                events += 1;
            }
            prev = f;
        }
        Self {
            published: flags.iter().filter(|&&f| f).count(),
            events,
            per_run_events: vec![events],
        }
    }

    fn merge(&mut self, o: &FalseAlarms) {
        self.published += o.published;
        self.events += o.events;
        self.per_run_events.extend_from_slice(&o.per_run_events);
    }
}

#[derive(Default, Clone)]
struct EmptyAgg {
    ticks: usize,
    seconds: f64,
    wlm: FalseAlarms,
    rti: FalseAlarms,
}

/// One run's phone figures.
#[derive(Default, Clone)]
struct PhoneRunScore {
    seed: u64,
    /// Σ of the phone's (x, y) steps between fired epochs, m.
    path_m: f64,
    rtt: Stats,
    rssi: Stats,
}

#[derive(Default)]
struct JobResult {
    empty: EmptyAgg,
    person: PersonAgg,
    phone: PhoneRunScore,
    calib_flagged: usize,
    n_links: usize,
}

fn score(run: &RoomSimRun, job: &Job, tick_hz: f64) -> JobResult {
    let reference = footprint_centroid(&job.scene);
    let mut r = JobResult {
        calib_flagged: run
            .calibration
            .calibrated_mask
            .iter()
            .filter(|&&ok| !ok)
            .count(),
        n_links: run.links.len(),
        ..JobResult::default()
    };
    if job.variant == Variant::Empty {
        r.empty.ticks = run.ticks.len();
        r.empty.seconds = run.ticks.len() as f64 / tick_hz;
        let flags = |pick: fn(&saans_locate::pipeline::TickRecord) -> &Outcome| {
            run.ticks
                .iter()
                .map(|t| pick(t).estimate().is_some())
                .collect::<Vec<_>>()
        };
        r.empty.wlm = FalseAlarms::of(&flags(|t| &t.wlm));
        r.empty.rti = FalseAlarms::of(&flags(|t| &t.rti));
    } else {
        for tick in &run.ticks {
            if tick.truth.is_empty() {
                continue;
            }
            let nearest = |x: f64, y: f64| {
                tick.truth
                    .iter()
                    .map(|p| dist(x, y, p.x, p.y))
                    .fold(f64::INFINITY, f64::min)
            };
            let walking = tick.truth.iter().any(|p| p.walking);
            let walker = |x: f64, y: f64| {
                tick.truth
                    .iter()
                    .filter(|p| p.walking)
                    .map(|p| dist(x, y, p.x, p.y))
                    .fold(f64::INFINITY, f64::min)
            };
            let truth = Truth {
                nearest: &nearest,
                walker: if walking { Some(&walker) } else { None },
                reference,
            };
            let class = if walking { WALK } else { STILL };
            for c in [class, ALL] {
                r.person.wlm[c].add(&tick.wlm, &truth);
                r.person.rti[c].add(&tick.rti, &truth);
                if let (Some(w), Some(q)) = (tick.wlm.estimate(), tick.rti.estimate()) {
                    let b = &mut r.person.both[c];
                    b.wlm.push(nearest(w.x, w.y));
                    b.rti.push(nearest(q.x, q.y));
                    if walking {
                        b.wlm_walker.push(walker(w.x, w.y));
                        b.rti_walker.push(walker(q.x, q.y));
                    }
                }
            }
        }
    }
    r.phone.seed = job.seed;
    let mut prev: Option<[f64; 3]> = None;
    for ep in &run.epochs {
        let phone = ep.truth;
        if let Some(p) = prev {
            r.phone.path_m += dist(phone[0], phone[1], p[0], p[1]);
        }
        prev = Some(phone);
        let at = |x: f64, y: f64| dist(x, y, phone[0], phone[1]);
        let truth = Truth {
            nearest: &at,
            walker: None,
            reference,
        };
        r.phone.rtt.add(&ep.rtt, &truth);
        r.phone.rssi.add(&ep.rssi, &truth);
    }
    r
}

fn parse_args() -> (f64, Vec<u64>, Option<String>) {
    let mut duration = 120.0;
    let mut seeds = vec![0, 1, 2, 3, 4];
    let mut only = None;
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let value = args.get(i + 1).cloned().unwrap_or_default();
        match args[i].as_str() {
            "--duration" => duration = value.parse().expect("--duration takes seconds"),
            "--seeds" => {
                seeds = value
                    .split(',')
                    .map(|s| s.trim().parse().expect("--seeds takes a,b,c"))
                    .collect()
            }
            "--scene" => only = Some(value),
            other => panic!("unknown argument {other}"),
        }
        i += 2;
    }
    (duration, seeds, only)
}

/// The phone lines of one variant: pooled per method, with whether the
/// phone moved, and the still runs one by one.
fn print_phone(scene_name: &str, variant: Variant, runs: &[PhoneRunScore]) {
    let (moved, still): (Vec<&PhoneRunScore>, Vec<&PhoneRunScore>) =
        runs.iter().partition(|r| r.path_m > 0.0);
    let paths: Vec<String> = runs.iter().map(|r| format!("{:.1}", r.path_m)).collect();
    let motion = if still.is_empty() {
        format!("phone moved in all {} runs", runs.len())
    } else if moved.is_empty() {
        format!("phone still in all {} runs", runs.len())
    } else {
        format!("phone moved in {} of {} runs", moved.len(), runs.len())
    };
    println!(
        "SIM  {scene_name} {:<6} {motion} (path per run [{}] m)",
        variant.name(),
        paths.join(", ")
    );
    for (mname, pick) in [
        (
            "RTT ",
            (|r: &PhoneRunScore| &r.rtt) as fn(&PhoneRunScore) -> &Stats,
        ),
        ("RSSI", |r: &PhoneRunScore| &r.rssi),
    ] {
        let mut pooled = Stats::default();
        for r in runs {
            pooled.merge(pick(r));
        }
        if pooled.total == 0 {
            continue;
        }
        println!(
            "SIM  {scene_name} {:<6} phone {mname} ({motion}) {}",
            variant.name(),
            pooled.line(None, "phone")
        );
        if !still.is_empty() {
            let per_run: Vec<String> = still
                .iter()
                .map(|r| {
                    let s = pick(r);
                    if s.n() == 0 {
                        format!("seed {}: n 0", r.seed)
                    } else {
                        format!(
                            "seed {}: n {}, median {:.2} m, inside {:.1}%",
                            r.seed,
                            s.n(),
                            median_p90(&s.errors()).0,
                            s.inside_pct()
                        )
                    }
                })
                .collect();
            println!(
                "SIM  {scene_name} {:<6} phone {mname} still runs one by one (each run is one sample): {}",
                variant.name(),
                per_run.join("; ")
            );
        }
    }
}

fn main() {
    let (duration, seeds, only) = parse_args();
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes");
    let mut paths: Vec<PathBuf> = std::fs::read_dir(&dir)
        .expect("spec/scenes")
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x == "json"))
        .collect();
    paths.sort();

    let mut jobs = Vec::new();
    for path in &paths {
        let name = path.file_stem().unwrap().to_string_lossy().to_string();
        if only.as_ref().is_some_and(|o| *o != name) {
            continue;
        }
        let scene = Scene::load(path).expect("a shipped scene loads");
        for &seed in &seeds {
            for variant in [Variant::Empty, Variant::Scene, Variant::Walker] {
                jobs.push(Job {
                    scene_name: name.clone(),
                    scene: scene.clone(),
                    seed,
                    variant,
                });
            }
        }
    }

    // Every run uses these defaults (§9); the header prints them.
    let cfg = Config::default();
    let rs_default = RoomSimConfig::default();
    let next = AtomicUsize::new(0);
    let results: Mutex<BTreeMap<usize, Result<JobResult, String>>> = Mutex::new(BTreeMap::new());
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(jobs.len().max(1));
    let started = std::time::Instant::now();
    std::thread::scope(|s| {
        for _ in 0..threads {
            s.spawn(|| loop {
                let i = next.fetch_add(1, Ordering::SeqCst);
                let Some(job) = jobs.get(i) else { break };
                let n_scene = job.scene.people.len();
                let mut rs = RoomSimConfig {
                    seed: job.seed,
                    duration_s: duration,
                    ..RoomSimConfig::default()
                };
                match job.variant {
                    Variant::Empty => rs.n_people = Some(0),
                    Variant::Scene => {}
                    Variant::Walker => {
                        rs.n_people = Some(n_scene + 1);
                        rs.phone_carrier = n_scene;
                    }
                }
                let out = run_roomsim(&job.scene, &rs, &cfg, "SIM")
                    .map(|run| score(&run, job, cfg.tick_hz))
                    .map_err(|e| e.to_string());
                results.lock().unwrap().insert(i, out);
            });
        }
    });
    let elapsed = started.elapsed().as_secs_f64();
    let results = results.into_inner().unwrap();

    println!("SIM  spec/locate.md §12a measured numbers (Room Sim input; reported, not pass/fail; every radius uncalibrated)");
    println!(
        "SIM  duration {duration} s per run, seeds {seeds:?}; Room Sim defaults (slot_s {} s, channel {}, tx_power_dbm {}); Locate defaults: link_signal {}, window_s {} s, tick_hz {} Hz, calib_s {} s per run, phone_rate_hz {} Hz, min_plausible_radius_m {} m, radius scales WLM {} RTI {} phone {}",
        rs_default.slot_s,
        rs_default.channel,
        rs_default.tx_power_dbm,
        cfg.link_signal,
        cfg.window_s,
        cfg.tick_hz,
        cfg.calib_s,
        cfg.phone_rate_hz,
        cfg.min_plausible_radius_m,
        cfg.wlm_radius_scale,
        cfg.rti_radius_scale,
        cfg.phone_radius_scale
    );
    println!("SIM  error is to the nearest person (people) or to the phone; 'to walker' is to the nearest walking person on walking ticks; the reference point is the footprint's area centroid, scored on the same ticks with the same radius");
    let mut by_scene: BTreeMap<&str, Vec<(usize, &Job)>> = BTreeMap::new();
    for (i, job) in jobs.iter().enumerate() {
        by_scene
            .entry(job.scene_name.as_str())
            .or_default()
            .push((i, job));
    }
    for (scene_name, list) in &by_scene {
        let mut empty = EmptyAgg::default();
        let mut person: BTreeMap<Variant, PersonAgg> = BTreeMap::new();
        let mut phone: BTreeMap<Variant, Vec<PhoneRunScore>> = BTreeMap::new();
        let mut flagged = 0usize;
        let mut links = 0usize;
        let mut runs = 0usize;
        let mut centroid = (f64::NAN, f64::NAN);
        for (i, job) in list {
            centroid = footprint_centroid(&job.scene);
            match &results[i] {
                Err(e) => println!(
                    "SIM  {scene_name} seed {} {}: refused: {e}",
                    job.seed,
                    job.variant.name()
                ),
                Ok(r) => {
                    runs += 1;
                    flagged += r.calib_flagged;
                    links = r.n_links;
                    if job.variant == Variant::Empty {
                        empty.ticks += r.empty.ticks;
                        empty.seconds += r.empty.seconds;
                        empty.wlm.merge(&r.empty.wlm);
                        empty.rti.merge(&r.empty.rti);
                    } else {
                        let p = person.entry(job.variant).or_default();
                        for c in 0..3 {
                            p.wlm[c].merge(&r.person.wlm[c]);
                            p.rti[c].merge(&r.person.rti[c]);
                            p.both[c].merge(&r.person.both[c]);
                        }
                    }
                    phone.entry(job.variant).or_default().push(r.phone.clone());
                }
            }
        }
        println!("SIM");
        println!(
            "SIM  === {scene_name}: {links} links, {runs} runs, {flagged} link calibrations imputed (median) over all runs; reference point ({:.2}, {:.2}) m ===",
            centroid.0, centroid.1
        );
        let hours = empty.seconds / 3600.0;
        let empty_frac = |fa: &FalseAlarms| pct(fa.published, empty.ticks);
        if hours > 0.0 {
            for (name, fa) in [("WLM", &empty.wlm), ("RTI", &empty.rti)] {
                println!(
                    "SIM  {scene_name} empty  {name} false alarms: {} published ticks of {} ({:.1}%) = {:.1} per hour; {} events (runs of consecutive published ticks) = {:.1} per hour; events per run {:?}; {:.0} s of empty room",
                    fa.published,
                    empty.ticks,
                    empty_frac(fa),
                    fa.published as f64 / hours,
                    fa.events,
                    fa.events as f64 / hours,
                    fa.per_run_events,
                    empty.seconds
                );
            }
        }
        let have_empty = empty.ticks > 0;
        for (variant, p) in &person {
            for (c, cname) in CLASSES {
                if p.wlm[c].total == 0 {
                    continue;
                }
                for (name, st, fa) in [
                    ("WLM", &p.wlm[c], &empty.wlm),
                    ("RTI", &p.rti[c], &empty.rti),
                ] {
                    println!(
                        "SIM  {scene_name} {:<6} {name} {cname} {}",
                        variant.name(),
                        st.line(have_empty.then(|| empty_frac(fa)), "nearest person")
                    );
                }
                let b = &p.both[c];
                if b.wlm.is_empty() {
                    println!(
                        "SIM  {scene_name} {:<6} same ticks {cname}: no tick where both published",
                        variant.name()
                    );
                } else {
                    let (wm, wp) = median_p90(&b.wlm);
                    let (rm, rp) = median_p90(&b.rti);
                    let mut s = format!(
                        "SIM  {scene_name} {:<6} same ticks {cname} (both published, {}): err to nearest WLM median {wm:5.2} p90 {wp:5.2} m, RTI median {rm:5.2} p90 {rp:5.2} m",
                        variant.name(),
                        b.wlm.len()
                    );
                    if !b.wlm_walker.is_empty() {
                        let (a, b2) = median_p90(&b.wlm_walker);
                        let (x, y) = median_p90(&b.rti_walker);
                        s += &format!("; to walker WLM median {a:5.2} p90 {b2:5.2} m, RTI median {x:5.2} p90 {y:5.2} m");
                    }
                    println!("{s}");
                }
            }
        }
        for (variant, runs) in &phone {
            print_phone(scene_name, *variant, runs);
        }
    }
    // Host timing: a wall-clock measurement of this machine, not a SIM
    // number, so it carries no SIM label.
    eprintln!(
        "host timing (wall clock, not a Locate number): {} runs in {elapsed:.1} s on {threads} threads",
        jobs.len()
    );
}
