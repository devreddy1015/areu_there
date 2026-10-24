//! Golden-vector parity with the Python reference `wifisense/localize/`
//! (ADR-002, spec/locate.md §12). Everything compared here is SIM.
//!
//! Each `localize_*` / `locate_*` set in `spec/golden/` (written by
//! `spec/golden/generate.py` on CPython 3.14; the layouts are the
//! docstrings of its `gen_localize_*` / `gen_locate_*` functions) is loaded
//! by name — `NAME.kv` scalars plus `NAME__ARRAY.f64` arrays, the
//! `saans-roomsim/tests/golden.rs` pattern — its inputs are rebuilt in
//! Rust, and **every** stored array is compared by the rule its own
//! `cmp.<ARRAY>` scalar names (the set is self-describing, so the loader
//! never guesses):
//!
//! - `input`: a recorded input, used as given;
//! - `exact`: identical (a NaN matches a NaN) — every discrete output:
//!   supports, groups, blind pixels, peak index, accept/abstain, reasons,
//!   gating, reports present, blocked flags, uniform draws;
//! - `bits`: bit-identical, sign of zero and NaN included (`py_hypot`);
//! - `rule`, `rule:N`, `rule:each`, `rule:ptr:P`: spec/roomsim.md §14's
//!   `|a - b| <= 1e-9*|b| + 1e-12*M`, `M` the largest finite `|b_j|` of the
//!   whole array, of each run of `N` values, of the value itself, or of
//!   each CSR row of lengths `P`; a non-finite `b` matches only the same
//!   non-finite `a`;
//! - `tie:V`: an integer RSSI, identical unless the pre-rounding `V[i]` is
//!   within 1e-6 of a .5 tie;
//! - `le:B`: a rounding-level diagnostic, both values `<= B`.
//!
//! Scalars named by a `cmp.X` entry follow it; other float scalars (a
//! Python `repr` with `.`, `e`, `inf` or `nan`) compare by the rule with
//! `M = |b|`, every other scalar is identical. [`Vectors::finish`] fails a
//! test that left any array or scalar of its set unread, so a field added
//! to a generator cannot go unchecked.
//!
//! A set built from a scene carries the `scene_sha256` of that file's
//! bytes; [`Vectors::load`] checks it. The pipeline sets (and the checks
//! that pin `saans_locate::pipeline`'s scene helpers to the Python's) need
//! `--features roomsim`.
//!
//! Three kinds of stored column are diagnostics the Rust API does not
//! expose, so the test derives them and ties each back to Rust's own
//! outputs: the phone radio's draws, `d` and pre-failure RTT value (drawn
//! from the §6.3 streams in §6.3's order; every range the Rust radio
//! reports must equal that value bit for bit, and its creation draws its
//! `offsets()`); the Kalman step's columns (from the filter state before
//! and after the epoch and its outcome); and the pipeline's per-tick `y`
//! and image (from an independent locator fed the same packets, whose
//! every WLM and RTI outcome must equal `run_roomsim`'s).
//!
//! Every test prints its worst error (`cargo test -p saans-locate --test
//! golden [--features roomsim] -- --nocapture`): how many compared values
//! are not bit-identical (0 on glibc), `budget` the largest
//! `|a - b| / (1e-9*|b| + 1e-12*M)` (a match needs `budget <= 1`) and
//! `rel` the largest `|a - b| / |b|` over values with `|b| >= 1e-6*M`.

// NaN fails closed: deciding comparisons are written `!(a <= b)` on
// purpose, as the spec and the crate write them.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use std::cell::RefCell;
use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use saans_locate::baseline::{link_spread, wlm, LinkGeom, WlmParams};
use saans_locate::geometry::{
    ellipse_weight, perimeter_nodes, Geometry, Link, Node, PixelGrid, Room,
};
use saans_locate::invert::{
    cholesky_factor, cholesky_inverse, cholesky_solve, cholesky_solve_factored, forward_project,
    Inverter, InverterOptions, SpatialPrior, DEFAULT_SYMMETRY_TOL,
};
use saans_locate::measure::{
    new_csi_frame, CalibrationMode, CsiFrame, LinkMeasurement, LinkSignal, MeasurementConfig,
    RssiSignal, StaticSuppressedSignal, TotalPowerSignal,
};
use saans_locate::num::{chi2_2_99, k95, ln10};
use saans_locate::phone::{
    box_blocks, cylinder_blocks, rssi_fix, rtt_fix, wall_blocks, Blockers, BoxObj, Config,
    Cylinder, FilterState, Fix, Method, PhoneFilter, PhoneNode, PhoneRun, Radio, Ranging,
    ReportRecord, TruthSample, Wall,
};
use saans_locate::pynum::{py_hypot, LocateError};
use saans_locate::radius::{plausible, radius95, Outcome, Reason};
use saans_locate::track::{extract_peak, Tracker, TrackerConfig};
use saans_locate::LABEL;
#[cfg(feature = "roomsim")]
use saans_scene::Scene;
use saans_sim::rng::Rng;
use sha2::{Digest, Sha256};

const REL: f64 = 1e-9;
const ABS: f64 = 1e-12;

fn repo() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../..")
}

fn golden_dir() -> PathBuf {
    repo().join("spec/golden")
}

// ===================================================================
// The loader and the comparison rules
// ===================================================================

/// How an array is compared (its `cmp.<name>` scalar).
#[derive(Debug, Clone, PartialEq)]
enum Cmp {
    Input,
    Exact,
    Bits,
    Rule(Grouping),
    Tie(String),
    Le(Bound),
}

#[derive(Debug, Clone, PartialEq)]
enum Grouping {
    Whole,
    Every(usize),
    Each,
    Ptr(String),
}

#[derive(Debug, Clone, PartialEq)]
enum Bound {
    Num(f64),
    Arr(String),
}

fn parse_cmp(text: &str) -> Cmp {
    match text {
        "input" => Cmp::Input,
        "exact" => Cmp::Exact,
        "bits" => Cmp::Bits,
        "rule" => Cmp::Rule(Grouping::Whole),
        "rule:each" => Cmp::Rule(Grouping::Each),
        _ => {
            if let Some(p) = text.strip_prefix("rule:ptr:") {
                Cmp::Rule(Grouping::Ptr(p.to_string()))
            } else if let Some(n) = text.strip_prefix("rule:") {
                Cmp::Rule(Grouping::Every(
                    n.parse().unwrap_or_else(|_| panic!("cmp {text}")),
                ))
            } else if let Some(v) = text.strip_prefix("tie:") {
                Cmp::Tie(v.to_string())
            } else if let Some(b) = text.strip_prefix("le:") {
                match b.parse::<f64>() {
                    Ok(x) => Cmp::Le(Bound::Num(x)),
                    Err(_) => Cmp::Le(Bound::Arr(b.to_string())),
                }
            } else {
                panic!("unknown cmp {text}")
            }
        }
    }
}

/// `a` and `b` are the same value (a NaN matches a NaN).
fn same(a: f64, b: f64) -> bool {
    a == b || (a.is_nan() && b.is_nan())
}

/// `a` and `b` are bit-identical (any NaN matches any NaN for counting).
fn bitwise(a: f64, b: f64) -> bool {
    a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan())
}

#[derive(Debug, Default)]
struct Stats {
    arrays: usize,
    values: usize,
    not_bitwise: usize,
    worst_budget: f64,
    worst_budget_at: String,
    worst_rel: f64,
    worst_rel_at: String,
}

/// One golden set: `NAME.kv` scalars and `NAME__ARRAY.f64` arrays.
struct Vectors {
    name: String,
    kv: BTreeMap<String, String>,
    cache: RefCell<BTreeMap<String, Vec<f64>>>,
    used: RefCell<BTreeSet<String>>,
    done: RefCell<BTreeSet<String>>,
    reasons: Vec<String>,
    stats: RefCell<Stats>,
    failures: RefCell<Vec<String>>,
}

impl Vectors {
    fn load(name: &str) -> Self {
        let path = golden_dir().join(format!("{name}.kv"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run spec/golden/generate.py)", path.display()));
        let kv: BTreeMap<String, String> = text
            .lines()
            .filter_map(|l| l.split_once(' '))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let reasons: Vec<String> = kv
            .get("reasons")
            .unwrap_or_else(|| panic!("{name}: no reasons"))
            .split(',')
            .map(str::to_string)
            .collect();
        let v = Self {
            name: name.to_string(),
            kv,
            cache: RefCell::new(BTreeMap::new()),
            used: RefCell::new(BTreeSet::new()),
            done: RefCell::new(BTreeSet::new()),
            reasons,
            stats: RefCell::new(Stats::default()),
            failures: RefCell::new(Vec::new()),
        };
        assert_eq!(v.s("label"), LABEL, "{name}: every golden set is SIM");
        assert_eq!(v.s("spec"), "locate_v1", "{name}: spec");
        for r in &v.reasons {
            assert!(
                reason_named(r).is_some(),
                "{name}: reason {r} has no Rust Reason"
            );
        }
        // §12: a set built from a scene records the digest of the scene
        // file's bytes, so a golden set never silently outlives a scene
        // edit.
        match v.kv.get("scene") {
            Some(rel) => {
                let path = repo().join(rel);
                let bytes =
                    std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
                let sha: String = Sha256::digest(&bytes)
                    .iter()
                    .map(|b| format!("{b:02x}"))
                    .collect();
                assert_eq!(
                    v.s("scene_sha256"),
                    sha,
                    "{name}: {rel} changed since the set was generated (rerun spec/golden/generate.py)"
                );
                v.touch("scene");
            }
            None => assert!(
                !v.kv.contains_key("scene_sha256"),
                "{name}: scene_sha256 without a scene"
            ),
        }
        // Metadata: every array has a cmp entry, every cmp entry an array
        // or a scalar.
        let arrays: Vec<String> = v.array_names();
        for a in &arrays {
            assert!(
                v.kv.contains_key(&format!("cmp.{a}")),
                "{name}: array {a} has no cmp"
            );
            parse_cmp(&v.kv[&format!("cmp.{a}")]);
        }
        let keys: Vec<String> = v.kv.keys().cloned().collect();
        for k in keys {
            if let Some(x) = k.strip_prefix("cmp.") {
                assert!(
                    v.kv.contains_key(&format!("array.{x}")) || v.kv.contains_key(x),
                    "{name}: cmp for unknown {x}"
                );
                v.touch(&k);
            } else if k.starts_with("array.") {
                v.touch(&k);
            }
        }
        v.touch("reasons");
        v
    }

    fn array_names(&self) -> Vec<String> {
        self.kv
            .keys()
            .filter_map(|k| k.strip_prefix("array.").map(str::to_string))
            .collect()
    }

    fn touch(&self, k: &str) {
        self.used.borrow_mut().insert(k.to_string());
    }

    fn has(&self, k: &str) -> bool {
        self.kv.contains_key(k)
    }

    /// A scalar (marked read).
    fn s(&self, k: &str) -> &str {
        self.touch(k);
        self.kv
            .get(k)
            .unwrap_or_else(|| panic!("{}: no scalar {k}", self.name))
    }

    fn f(&self, k: &str) -> f64 {
        let v = self.s(k);
        v.parse()
            .unwrap_or_else(|_| panic!("{}: {k}={v} not a number", self.name))
    }

    fn u(&self, k: &str) -> usize {
        let x = self.f(k);
        assert!(x >= 0.0 && x.fract() == 0.0, "{}: {k}={x}", self.name);
        x as usize
    }

    /// Mark descriptive scalars (case names, error descriptions) read.
    fn doc(&self, keys: &[&str]) {
        for k in keys {
            self.s(k);
        }
    }

    /// The error-case descriptions `err.<i>`, read and returned.
    fn err_docs(&self) -> Vec<String> {
        let n = self.u("n_err");
        (0..n)
            .map(|i| self.s(&format!("err.{i}")).to_string())
            .collect()
    }

    fn cmp_of(&self, k: &str) -> Cmp {
        parse_cmp(
            self.kv
                .get(&format!("cmp.{k}"))
                .unwrap_or_else(|| panic!("{}: no cmp.{k}", self.name)),
        )
    }

    /// The stored values of array `k` (not marked).
    fn golden(&self, k: &str) -> Vec<f64> {
        if let Some(v) = self.cache.borrow().get(k) {
            return v.clone();
        }
        let n: usize = self
            .kv
            .get(&format!("array.{k}"))
            .unwrap_or_else(|| panic!("{}: no array {k}", self.name))
            .parse()
            .unwrap();
        let path = golden_dir().join(format!("{}__{k}.f64", self.name));
        let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
        assert_eq!(bytes.len(), n * 8, "{}: array {k} length", self.name);
        let out: Vec<f64> = bytes
            .as_chunks::<8>()
            .0
            .iter()
            .map(|c| f64::from_le_bytes(*c))
            .collect();
        self.cache.borrow_mut().insert(k.to_string(), out.clone());
        out
    }

    /// A recorded input array (its cmp must be `input`).
    fn input(&self, k: &str) -> Vec<f64> {
        assert_eq!(
            self.cmp_of(k),
            Cmp::Input,
            "{}: {k} is not an input",
            self.name
        );
        self.done.borrow_mut().insert(k.to_string());
        self.golden(k)
    }

    /// A recorded input of integers.
    fn input_ints(&self, k: &str) -> Vec<i64> {
        self.input(k)
            .iter()
            .map(|&x| {
                assert!(x.fract() == 0.0, "{}: {k} holds {x}", self.name);
                x as i64
            })
            .collect()
    }

    /// A recorded input of non-negative integers.
    fn input_usize(&self, k: &str) -> Vec<usize> {
        self.input_ints(k)
            .iter()
            .map(|&x| usize::try_from(x).unwrap())
            .collect()
    }

    fn fail(&self, msg: String) {
        self.failures
            .borrow_mut()
            .push(format!("{}: {msg}", self.name));
    }

    /// An extra consistency check of this test (not a stored array).
    fn ensure(&self, ok: bool, msg: impl FnOnce() -> String) {
        if !ok {
            self.fail(msg());
        }
    }

    /// Compare Rust's `actual` with stored array `k` by its cmp rule.
    fn check(&self, k: &str, actual: &[f64]) {
        let cmp = self.cmp_of(k);
        assert!(cmp != Cmp::Input, "{}: {k} is an input", self.name);
        self.done.borrow_mut().insert(k.to_string());
        let b = self.golden(k);
        self.compare(k, &cmp, actual, &b);
    }

    /// `check` for booleans and integers.
    fn check_n<T: Copy + Into<f64>>(&self, k: &str, actual: &[T]) {
        let a: Vec<f64> = actual.iter().map(|&x| x.into()).collect();
        self.check(k, &a);
    }

    fn compare(&self, k: &str, cmp: &Cmp, a: &[f64], b: &[f64]) {
        {
            let mut st = self.stats.borrow_mut();
            st.arrays += 1;
            st.values += b.len();
        }
        if a.len() != b.len() {
            self.fail(format!(
                "{k}: Rust has {} values, Python {}",
                a.len(),
                b.len()
            ));
            return;
        }
        let mut bad = 0usize;
        let mut report = |i: usize, why: &str| {
            bad += 1;
            if bad <= 6 {
                let extra = if k == "err_code" {
                    format!(
                        " [{}]",
                        self.kv
                            .get(&format!("err.{i}"))
                            .cloned()
                            .unwrap_or_default()
                    )
                } else {
                    String::new()
                };
                self.failures.borrow_mut().push(format!(
                    "{}: {k}[{i}] Rust {:?} Python {:?} ({why}){extra}",
                    self.name, a[i], b[i]
                ));
            }
        };
        let mut not_bitwise = 0;
        for i in 0..b.len() {
            if !bitwise(a[i], b[i]) {
                not_bitwise += 1;
            }
        }
        match cmp {
            Cmp::Input => unreachable!(),
            Cmp::Exact => {
                for i in 0..b.len() {
                    if !same(a[i], b[i]) {
                        report(i, "exact");
                    }
                }
            }
            Cmp::Bits => {
                for i in 0..b.len() {
                    if a[i].to_bits() != b[i].to_bits() {
                        report(i, "bits");
                    }
                }
            }
            Cmp::Tie(vname) => {
                let v = self.golden(vname);
                assert_eq!(v.len(), b.len(), "{}: {k} vs {vname}", self.name);
                for i in 0..b.len() {
                    let tie = (v[i] - v[i].floor() - 0.5).abs() <= 1e-6;
                    let ok = if tie {
                        a[i].fract() == 0.0 && (a[i] - b[i]).abs() <= 1.0
                    } else {
                        same(a[i], b[i])
                    };
                    if !ok {
                        report(i, "tie rule");
                    }
                }
            }
            Cmp::Le(bound) => {
                let bs: Vec<f64> = match bound {
                    Bound::Num(x) => vec![*x; b.len()],
                    Bound::Arr(name) => self.golden(name),
                };
                for i in 0..b.len() {
                    if !(a[i] <= bs[i] && b[i] <= bs[i]) {
                        report(i, &format!("le {}", bs[i]));
                    }
                }
            }
            Cmp::Rule(g) => {
                let groups: Vec<(usize, usize)> = match g {
                    Grouping::Whole => vec![(0, b.len())],
                    Grouping::Each => (0..b.len()).map(|i| (i, i + 1)).collect(),
                    Grouping::Every(n) => {
                        assert_eq!(b.len() % n, 0, "{}: {k} by {n}", self.name);
                        (0..b.len() / n).map(|j| (j * n, (j + 1) * n)).collect()
                    }
                    Grouping::Ptr(p) => {
                        let lens = self.golden(p);
                        let mut out = Vec::new();
                        let mut at = 0usize;
                        for l in lens {
                            let l = l as usize;
                            out.push((at, at + l));
                            at += l;
                        }
                        assert_eq!(at, b.len(), "{}: {k} grouped by {p}", self.name);
                        out
                    }
                };
                for (lo, hi) in groups {
                    let m = b[lo..hi]
                        .iter()
                        .filter(|x| x.is_finite())
                        .fold(0.0f64, |m, x| if x.abs() > m { x.abs() } else { m });
                    for i in lo..hi {
                        let (ai, bi) = (a[i], b[i]);
                        if !bi.is_finite() {
                            if !same(ai, bi) {
                                report(i, "non-finite reference");
                            }
                            continue;
                        }
                        let tol = REL * bi.abs() + ABS * m;
                        let d = (ai - bi).abs();
                        if !(d <= tol) {
                            report(i, &format!("rule, tolerance {tol:e}"));
                        }
                        let budget = if d == 0.0 {
                            0.0
                        } else if tol > 0.0 {
                            d / tol
                        } else {
                            f64::INFINITY
                        };
                        let mut st = self.stats.borrow_mut();
                        if budget > st.worst_budget
                            || (budget.is_nan() && !st.worst_budget.is_nan())
                        {
                            st.worst_budget = budget;
                            st.worst_budget_at = format!("{k}[{i}]");
                        }
                        if bi.abs() >= 1e-6 * m && bi != 0.0 {
                            let rel = d / bi.abs();
                            if rel > st.worst_rel {
                                st.worst_rel = rel;
                                st.worst_rel_at = format!("{k}[{i}]");
                            }
                        }
                    }
                }
            }
        }
        if bad > 6 {
            self.failures
                .borrow_mut()
                .push(format!("{}: {k}: {bad} mismatches in all", self.name));
        }
        self.stats.borrow_mut().not_bitwise += not_bitwise;
    }

    /// Compare a numeric scalar: by its `cmp.<k>` if it has one, else by
    /// the rule with `M = |b|` for a float and identically for an integer.
    fn check_scalar(&self, k: &str, actual: f64) {
        let text = self.s(k).to_string();
        let b: f64 = text
            .parse()
            .unwrap_or_else(|_| panic!("{}: {k}={text} not a number", self.name));
        let cmp = if self.has(&format!("cmp.{k}")) {
            self.cmp_of(k)
        } else if text.contains(['.', 'e', 'n', 'i']) {
            Cmp::Rule(Grouping::Each)
        } else {
            Cmp::Exact
        };
        self.compare(k, &cmp, &[actual], &[b]);
    }

    fn check_str(&self, k: &str, actual: &str) {
        let b = self.s(k).to_string();
        if b != actual {
            self.fail(format!("{k}: Rust {actual:?} Python {b:?}"));
        }
    }

    /// The reason code of §11's `reasons` list (-1 published).
    fn code(&self, r: Option<Reason>) -> f64 {
        match r {
            None => -1.0,
            Some(r) => self
                .reasons
                .iter()
                .position(|n| n == r.as_str())
                .unwrap_or_else(|| panic!("{}: reason {r} not in the set's list", self.name))
                as f64,
        }
    }

    /// The Reason of a stored reason code (>= 0).
    fn reason_of(&self, code: f64) -> Reason {
        let i = code as usize;
        assert!(code >= 0.0 && code.fract() == 0.0 && i < self.reasons.len());
        reason_named(&self.reasons[i]).unwrap()
    }

    /// Fail unless every array and scalar was read; print the worst
    /// error; panic on any mismatch.
    fn finish(self) {
        let done = self.done.borrow();
        for a in self.array_names() {
            if !done.contains(&a) {
                self.fail(format!("array {a} was neither used nor compared"));
            }
        }
        let used = self.used.borrow();
        for k in self.kv.keys() {
            if !used.contains(k) && !self.kv.contains_key(&format!("array.{k}")) {
                self.fail(format!("scalar {k} was neither used nor compared"));
            }
        }
        let st = self.stats.borrow();
        println!(
            "{}: {} comparisons, {} values; {} not bit-identical; worst budget {:.3e} ({}), worst rel {:.3e} ({})",
            self.name,
            st.arrays,
            st.values,
            st.not_bitwise,
            st.worst_budget,
            if st.worst_budget_at.is_empty() { "-" } else { &st.worst_budget_at },
            st.worst_rel,
            if st.worst_rel_at.is_empty() { "-" } else { &st.worst_rel_at },
        );
        let failures = self.failures.borrow();
        if !failures.is_empty() {
            panic!(
                "{} mismatch(es) against Python:\n{}",
                failures.len(),
                failures.join("\n")
            );
        }
    }
}

const ALL_REASONS: [Reason; 8] = [
    Reason::NoDisturbance,
    Reason::ImplausibleRadius,
    Reason::NotSignificant,
    Reason::NoReports,
    Reason::TooFewRanges,
    Reason::Degenerate,
    Reason::NotConverged,
    Reason::Gated,
];

fn reason_named(name: &str) -> Option<Reason> {
    ALL_REASONS.iter().copied().find(|r| r.as_str() == name)
}

/// EXC_CODE (generate.py `_exc_code`) of a ported module's error.
fn exc_code(e: &LocateError) -> f64 {
    match e {
        LocateError::ValueError(_) => 1.0,
        LocateError::KeyError(_) => 2.0,
        LocateError::IndexError(_) => 3.0,
        LocateError::MalformedMatrixError(_) => 4.0,
        LocateError::AsymmetricMatrixError(_) => 5.0,
        LocateError::NotPositiveDefiniteError(_) => 6.0,
        LocateError::IllConditionedError(_) => 7.0,
        other => panic!("{other}: Python's _exc_code has no code for it (it would propagate)"),
    }
}

fn exc<T>(r: Result<T, LocateError>) -> f64 {
    match r {
        Ok(_) => 0.0,
        Err(e) => exc_code(&e),
    }
}

/// generate.py `_est_row`: (reason code, x, y, radius_m, cov_xx, cov_xy,
/// cov_yy, confidence), 0.0 for absent values.
fn est_row(v: &Vectors, o: &Outcome) -> [f64; 8] {
    match o {
        Outcome::Abstain(r) => [v.code(Some(*r)), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        Outcome::Estimate(e) => {
            assert!(!e.radius_calibrated, "{}: radius_calibrated", v.name);
            [
                -1.0,
                e.x,
                e.y,
                e.radius_m,
                e.cov_xx,
                e.cov_xy,
                e.cov_yy,
                e.confidence.unwrap_or(0.0),
            ]
        }
    }
}

/// Collects one column per field of [`est_row`].
#[derive(Default)]
struct EstCols {
    cols: [Vec<f64>; 8],
}

impl EstCols {
    fn push(&mut self, row: [f64; 8]) {
        for (c, x) in self.cols.iter_mut().zip(row) {
            c.push(x);
        }
    }

    /// Compare under `prefix` with generate.py's EST_FIELDS names
    /// (`y_name` for the second coordinate; `None` skips confidence).
    fn check(&self, v: &Vectors, prefix: &str, y_name: &str, confidence: bool) {
        let names = [
            "reason",
            "x",
            y_name,
            "radius",
            "cov_xx",
            "cov_xy",
            "cov_yy",
            "confidence",
        ];
        for (i, n) in names.iter().enumerate() {
            if i == 7 && !confidence {
                continue;
            }
            v.check(&format!("{prefix}{n}"), &self.cols[i]);
        }
    }
}

fn room(w: f64, h: f64) -> Room {
    Room::new(w, h).expect("a valid room")
}

fn grid(r: Room, pitch: f64) -> PixelGrid {
    PixelGrid::new(r, pitch).expect("a valid grid")
}

/// `Geometry(Room(w, h), nodes, PixelGrid(room, pitch), lambda)` from the
/// recorded nodes.
fn geometry_of(
    w: f64,
    h: f64,
    pitch: f64,
    lambda: f64,
    ids: &[i64],
    x: &[f64],
    y: &[f64],
) -> Geometry {
    let r = room(w, h);
    let nodes = ids
        .iter()
        .zip(x.iter().zip(y))
        .map(|(&i, (&x, &y))| Node::new(i, x, y))
        .collect();
    Geometry::new(r, nodes, Some(grid(r, pitch)), lambda).expect("a valid geometry")
}

/// generate.py `_min_boundary_margin`: the smallest
/// `|d_tx + d_rx - d_l - lambda|` over every link with `d_l > 0` and every
/// pixel, `d` by CPython hypot (the ported geometry's `py_hypot`).
fn min_boundary_margin(g: &Geometry) -> f64 {
    let lam = g.lambda_m();
    let pixels = g.grid().pixels();
    let dist: BTreeMap<i64, Vec<f64>> = g
        .nodes()
        .iter()
        .map(|n| {
            (
                n.node_id,
                pixels
                    .iter()
                    .map(|&(px, py)| py_hypot(n.x - px, n.y - py))
                    .collect(),
            )
        })
        .collect();
    let mut margin = f64::INFINITY;
    for lk in g.links() {
        if !(lk.length_m > 0.0) {
            continue;
        }
        let (dt, dr) = (&dist[&lk.tx_id], &dist[&lk.rx_id]);
        for p in 0..pixels.len() {
            let m = (((dt[p] + dr[p]) - lk.length_m) - lam).abs();
            if m < margin {
                margin = m;
            }
        }
    }
    margin
}

/// §5's plain link data of the ported geometry's links.
fn link_geoms(g: &Geometry) -> Vec<LinkGeom> {
    g.links()
        .iter()
        .map(|l| LinkGeom {
            tx: l.tx,
            rx: l.rx,
            midpoint: l.midpoint(),
            length_m: l.length_m,
        })
        .collect()
}

fn signal_of(code: f64) -> Box<dyn LinkSignal> {
    match code as i64 {
        0 => Box::new(RssiSignal),
        1 => Box::new(TotalPowerSignal),
        2 => Box::new(StaticSuppressedSignal::default()),
        c => panic!("signal code {c}"),
    }
}

fn signal_name(code: f64) -> &'static str {
    match code as i64 {
        0 => "rssi_dbm",
        1 => "total_power_db",
        2 => "static_suppressed_amplitude",
        c => panic!("signal code {c}"),
    }
}

fn split_csv(text: &str) -> Vec<f64> {
    text.split(',').map(|x| x.parse().unwrap()).collect()
}

/// Apply one `LocateConfig` field (a `V_cfg.<name>` / `locate.<name>`
/// scalar) to a §9 config.
fn set_cfg(cfg: &mut Config, name: &str, text: &str) {
    let f = || -> f64 { text.parse().unwrap_or_else(|_| panic!("{name}={text}")) };
    match name {
        "lambda_m" => cfg.lambda_m = f(),
        "pitch_m" => cfg.pitch_m = f(),
        "alpha" => cfg.alpha = f(),
        "sigma" => cfg.sigma = f(),
        "delta_c" => cfg.delta_c = f(),
        "z_threshold" => cfg.z_threshold = f(),
        "window_s" => cfg.window_s = f(),
        "min_windows" => cfg.min_windows = text.parse().unwrap(),
        "link_signal" => cfg.link_signal = text.to_string(),
        "calib_s" => cfg.calib_s = f(),
        "tick_hz" => cfg.tick_hz = f(),
        "wlm_gate_z" => cfg.wlm_gate_z = f(),
        "wlm_floor" => cfg.wlm_floor = Some(f()),
        "wlm_radius_scale" => cfg.wlm_radius_scale = f(),
        "rti_radius_scale" => cfg.rti_radius_scale = f(),
        "phone_radius_scale" => cfg.phone_radius_scale = f(),
        "min_plausible_radius_m" => cfg.min_plausible_radius_m = f(),
        "phone_rate_hz" => cfg.phone_rate_hz = f(),
        "phone_seed" => cfg.phone_seed = Some(text.parse().unwrap()),
        "rtt_sigma_m" => cfg.rtt_sigma_m = f(),
        "rtt_nlos_sigma_scale" => cfg.rtt_nlos_sigma_scale = f(),
        "rtt_nlos_bias_m" => cfg.rtt_nlos_bias_m = f(),
        "rtt_outlier_p" => cfg.rtt_outlier_p = f(),
        "rtt_outlier_lo_m" => cfg.rtt_outlier_lo_m = f(),
        "rtt_outlier_hi_m" => cfg.rtt_outlier_hi_m = f(),
        "rtt_offset_sigma_m" => cfg.rtt_offset_sigma_m = f(),
        "rtt_persist_sigma_m" => cfg.rtt_persist_sigma_m = f(),
        "rtt_fail_p" => cfg.rtt_fail_p = f(),
        "rssi_fail_p" => cfg.rssi_fail_p = f(),
        "rssi_breakpoint_m" => cfg.rssi_breakpoint_m = f(),
        "rssi_far_exponent" => cfg.rssi_far_exponent = f(),
        "rssi_shadow_sigma_db" => cfg.rssi_shadow_sigma_db = f(),
        "rssi_persist_sigma_db" => cfg.rssi_persist_sigma_db = f(),
        "rssi_shadow_corr_m" => cfg.rssi_shadow_corr_m = f(),
        "rssi_nlos_loss_db" => cfg.rssi_nlos_loss_db = f(),
        "rssi_device_sigma_db" => cfg.rssi_device_sigma_db = f(),
        "rssi_jitter_db" => cfg.rssi_jitter_db = f(),
        "rssi_est_exponent" => cfg.rssi_est_exponent = f(),
        "rssi_p0_dbm" => cfg.rssi_p0_dbm = Some(f()),
        "phone_assumed_z_m" => cfg.phone_assumed_z_m = f(),
        "huber_k" => cfg.huber_k = f(),
        "phone_accel_psd" => cfg.phone_accel_psd = f(),
        "phone_init_speed_std" => cfg.phone_init_speed_std = f(),
        "phone_max_coast_s" => cfg.phone_max_coast_s = f(),
        "tx_power_dbm" => cfg.tx_power_dbm = Some(f()),
        "carrier_hz" => cfg.carrier_hz = Some(f()),
        _ => panic!("unknown LocateConfig field {name}"),
    }
}

/// Every `<head><name>` scalar applied to `cfg`.
fn apply_cfg(v: &Vectors, head: &str, cfg: &mut Config) {
    let keys: Vec<String> =
        v.kv.keys()
            .filter(|k| k.starts_with(head))
            .cloned()
            .collect();
    for k in keys {
        let text = v.s(&k).to_string();
        set_cfg(cfg, &k[head.len()..], &text);
    }
}

// ===================================================================
// localize_*: the ported RTI
// ===================================================================

/// CPython 3.14's two-argument `math.hypot` against `py_hypot`, bit for
/// bit: the recorded edge pairs and 10^6 pairs drawn in Rust from
/// `Rng.stream(0, 320000)`, whose results' SHA-256 is recorded (§12).
#[test]
fn localize_py_hypot() {
    let v = Vectors::load("localize_py_hypot");
    let a = v.input("a");
    let b = v.input("b");
    assert_eq!(v.u("n_edge"), a.len());
    v.doc(&["edge_values"]);
    let h: Vec<f64> = a.iter().zip(&b).map(|(&x, &y)| py_hypot(x, y)).collect();
    v.check("h", &h);

    let n = v.u("n_random");
    let mut rng = Rng::stream(v.f("rng_seed") as u64, v.f("rng_stream") as u64);
    let (lo, hi) = (v.f("range_lo"), v.f("range_hi"));
    let mut sha = Sha256::new();
    let (mut ra, mut rb, mut rh) = (Vec::new(), Vec::new(), Vec::new());
    let mut differs = 0u64;
    for i in 0..n {
        let x = rng.range(lo, hi);
        let y = rng.range(lo, hi);
        let r = py_hypot(x, y);
        if r != (x * x + y * y).sqrt() {
            differs += 1;
        }
        sha.update(r.to_le_bytes());
        if i < 64 {
            ra.push(x);
            rb.push(y);
            rh.push(r);
        }
    }
    // The recorded first draws are inputs; the Rust draws must be them.
    let (ga, gb) = (v.input("rand_a"), v.input("rand_b"));
    v.ensure(
        ra.iter().zip(&ga).all(|(p, q)| p.to_bits() == q.to_bits())
            && rb.iter().zip(&gb).all(|(p, q)| p.to_bits() == q.to_bits()),
        || "Rng.stream(0, 320000).range(-10, 10) draws differ from the recorded ones".into(),
    );
    v.check("rand_h", &rh);
    let digest: String = sha.finalize().iter().map(|b| format!("{b:02x}")).collect();
    v.check_str("sha256_random", &digest);
    v.check_scalar("n_random_not_sqrt", differs as f64);
    v.finish();
}

/// geometry.py's Room, perimeter_nodes, links, PixelGrid, nearest_index,
/// ellipse_weight and their refusals.
#[test]
fn localize_perimeter() {
    let v = Vectors::load("localize_perimeter");
    let (w, h, n) = (
        v.input("pc_room_w"),
        v.input("pc_room_h"),
        v.input_ints("pc_n"),
    );
    let (inset, off) = (v.input("pc_inset"), v.input("pc_offset"));
    assert_eq!(v.u("n_perimeter_cases"), w.len());
    let mut ids = Vec::new();
    let (mut nx, mut ny) = (Vec::new(), Vec::new());
    let mut nlinks = Vec::new();
    let (mut ltx, mut lrx, mut llen, mut lmx, mut lmy) =
        (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
    for c in 0..w.len() {
        let r = room(w[c], h[c]);
        let nodes = perimeter_nodes(&r, n[c], inset[c], off[c]).expect("perimeter_nodes");
        for nd in &nodes {
            ids.push(nd.node_id as f64);
            nx.push(nd.x);
            ny.push(nd.y);
        }
        let g = Geometry::new(r, nodes, Some(grid(r, 0.5)), 0.3).expect("geometry");
        nlinks.push(g.n_links() as f64);
        for lk in g.links() {
            ltx.push(lk.tx_id as f64);
            lrx.push(lk.rx_id as f64);
            llen.push(lk.length_m);
            lmx.push(lk.midpoint().0);
            lmy.push(lk.midpoint().1);
        }
    }
    v.check("node_id", &ids);
    v.check("node_x", &nx);
    v.check("node_y", &ny);
    v.check("pc_nlinks", &nlinks);
    v.check("link_tx", &ltx);
    v.check("link_rx", &lrx);
    v.check("link_len", &llen);
    v.check("link_mid_x", &lmx);
    v.check("link_mid_y", &lmy);

    let (gw, gh, gp) = (v.input("g_room_w"), v.input("g_room_h"), v.input("g_pitch"));
    assert_eq!(v.u("n_grid_cases"), gw.len());
    let mut grids = Vec::new();
    let mut cols: [Vec<f64>; 8] = Default::default();
    for c in 0..gw.len() {
        let g = grid(room(gw[c], gh[c]), gp[c]);
        cols[0].push(g.n_x() as f64);
        cols[1].push(g.n_y() as f64);
        cols[2].push(g.len() as f64);
        cols[3].push(g.pitch_x_m());
        cols[4].push(g.pitch_y_m());
        cols[5].push(g.pixel_area_m2());
        for &(px, py) in g.pixels() {
            cols[6].push(px);
            cols[7].push(py);
        }
        grids.push(g);
    }
    for (i, k) in [
        "g_n_x",
        "g_n_y",
        "g_n_pixels",
        "g_pitch_x",
        "g_pitch_y",
        "g_pixel_area",
        "pixel_x",
        "pixel_y",
    ]
    .iter()
    .enumerate()
    {
        v.check(k, &cols[i]);
    }

    let (ng, nix, niy) = (v.input_usize("ni_grid"), v.input("ni_x"), v.input("ni_y"));
    assert_eq!(v.u("n_ni"), ng.len());
    let ni: Vec<f64> = (0..ng.len())
        .map(|i| {
            grids[ng[i]]
                .nearest_index(nix[i], niy[i])
                .expect("nearest_index") as f64
        })
        .collect();
    v.check("ni_index", &ni);

    let (ax, ay, bx, by) = (
        v.input("ew_ax"),
        v.input("ew_ay"),
        v.input("ew_bx"),
        v.input("ew_by"),
    );
    let (px, py, lam) = (v.input("ew_px"), v.input("ew_py"), v.input("ew_lambda"));
    assert_eq!(v.u("n_ew"), ax.len());
    let (mut ew, mut ex) = (Vec::new(), Vec::new());
    for i in 0..ax.len() {
        // Case 3 is the zero-length link, built with length 0.0.
        let length = if i == 3 {
            0.0
        } else {
            Node::new(0, ax[i], ay[i]).distance_to(bx[i], by[i])
        };
        let lk = Link {
            tx_id: 0,
            rx_id: 1,
            tx: (ax[i], ay[i]),
            rx: (bx[i], by[i]),
            length_m: length,
        };
        ew.push(ellipse_weight(&lk, px[i], py[i], lam[i]).expect("ellipse_weight"));
        ex.push(lk.excess_path_m(px[i], py[i]));
    }
    v.check("ew_weight", &ew);
    v.check("ew_excess", &ex);

    let docs = v.err_docs();
    let r86 = room(8.0, 6.0);
    let g05 = || grid(r86, 0.5);
    let n2 = || vec![Node::new(0, 1.0, 1.0), Node::new(1, 2.0, 2.0)];
    let g8 = Geometry::perimeter_n(8).unwrap();
    let codes = vec![
        exc(Room::new(0.0, 6.0)),
        exc(Room::new(8.0, f64::NAN)),
        exc(perimeter_nodes(&r86, 1, 0.1, 0.0)),
        exc(perimeter_nodes(&r86, 4, -0.1, 0.0)),
        exc(perimeter_nodes(&r86, 4, 3.0, 0.0)),
        exc(PixelGrid::new(r86, f64::NAN)),
        exc(PixelGrid::new(r86, 0.0)),
        exc(PixelGrid::new(r86, -0.5)),
        exc(g05().pixel_index(16, 0)),
        exc(g05().pixel_index(-1, 0)),
        exc(g05().nearest_index(f64::NAN, 1.0)),
        exc(Geometry::new(r86, vec![Node::new(0, 1.0, 1.0)], None, 0.3)),
        exc(Geometry::new(
            r86,
            vec![Node::new(0, 1.0, 1.0), Node::new(0, 2.0, 2.0)],
            None,
            0.3,
        )),
        exc(Geometry::new(
            r86,
            vec![Node::new(0, 1.0, 1.0), Node::new(1, 1.0, 1.0)],
            None,
            0.3,
        )),
        exc(Geometry::new(r86, n2(), None, 0.0)),
        exc(Geometry::new(r86, n2(), None, f64::NAN)),
        exc(g8.link_index(0, 0)),
        exc(g8.node(99)),
        exc(ellipse_weight(
            &Link {
                tx_id: 0,
                rx_id: 1,
                tx: (0.0, 0.0),
                rx: (4.0, 0.0),
                length_m: 4.0,
            },
            1.0,
            1.0,
            0.0,
        )),
    ];
    assert_eq!(codes.len(), docs.len(), "error cases: {docs:?}");
    v.check("err_code", &codes);
    v.finish();
}

/// The checks of one localize_weights geometry (prefix "" or "V_").
fn check_weights(v: &Vectors, p: &str) -> Geometry {
    let ids = v.input_ints(&format!("{p}node_id"));
    let (x, y) = (
        v.input(&format!("{p}node_x")),
        v.input(&format!("{p}node_y")),
    );
    let g = geometry_of(
        v.f(&format!("{p}room_w")),
        v.f(&format!("{p}room_h")),
        v.f(&format!("{p}pitch_m")),
        v.f(&format!("{p}lambda_m")),
        &ids,
        &x,
        &y,
    );
    let sup = g.supports();
    let counts = g.pixel_link_counts();
    let blind: Vec<f64> = counts
        .iter()
        .enumerate()
        .filter(|(_, &c)| c == 0)
        .map(|(i, _)| i as f64)
        .collect();
    let nnz: usize = sup.iter().map(Vec::len).sum();
    v.check_scalar(&format!("{p}n_x"), g.grid().n_x() as f64);
    v.check_scalar(&format!("{p}n_y"), g.grid().n_y() as f64);
    v.check_scalar(&format!("{p}n_nodes"), g.n_nodes() as f64);
    v.check_scalar(&format!("{p}n_links"), g.n_links() as f64);
    v.check_scalar(&format!("{p}n_pixels"), g.n_pixels() as f64);
    v.check_scalar(&format!("{p}nnz"), nnz as f64);
    v.check_scalar(&format!("{p}n_blind"), blind.len() as f64);
    v.check_scalar(&format!("{p}min_boundary_margin"), min_boundary_margin(&g));
    let links = g.links();
    v.check(
        &format!("{p}link_tx"),
        &links.iter().map(|l| l.tx_id as f64).collect::<Vec<_>>(),
    );
    v.check(
        &format!("{p}link_rx"),
        &links.iter().map(|l| l.rx_id as f64).collect::<Vec<_>>(),
    );
    v.check(
        &format!("{p}link_len"),
        &links.iter().map(|l| l.length_m).collect::<Vec<_>>(),
    );
    v.check(
        &format!("{p}link_w"),
        &links
            .iter()
            .map(|l| {
                if l.length_m > 0.0 {
                    1.0 / l.length_m.sqrt()
                } else {
                    0.0
                }
            })
            .collect::<Vec<_>>(),
    );
    v.check(
        &format!("{p}support_len"),
        &sup.iter().map(|s| s.len() as f64).collect::<Vec<_>>(),
    );
    v.check(
        &format!("{p}support_idx"),
        &sup.iter().flatten().map(|&q| q as f64).collect::<Vec<_>>(),
    );
    v.check(
        &format!("{p}pixel_link_counts"),
        &counts.iter().map(|&c| c as f64).collect::<Vec<_>>(),
    );
    v.check(&format!("{p}blind_pixels"), &blind);
    // The dense weight matrix agrees with the supports.
    for (li, s) in sup.iter().enumerate() {
        let nz: Vec<usize> = g.weight_matrix()[li]
            .iter()
            .enumerate()
            .filter(|(_, &w)| w != 0.0)
            .map(|(q, _)| q)
            .collect();
        v.ensure(nz == *s, || {
            format!("{p}: dense W row {li} disagrees with its support")
        });
    }
    g
}

/// The Locate geometry of a scene (§2) as `saans_locate::pipeline` builds
/// it equals the one Python's `locate_geometry` recorded.
#[cfg(feature = "roomsim")]
fn check_scene_geometry(v: &Vectors, g: &Geometry) {
    let scene = Scene::load(repo().join(v.s("scene"))).unwrap();
    let mine =
        saans_locate::pipeline::person_geometry(&scene, &Config::default()).expect("geometry");
    v.ensure(
        mine.room() == g.room()
            && mine.nodes() == g.nodes()
            && mine.grid() == g.grid()
            && mine.lambda_m() == g.lambda_m(),
        || "pipeline::person_geometry differs from Python's locate_geometry".into(),
    );
}

#[cfg(not(feature = "roomsim"))]
fn check_scene_geometry(v: &Vectors, _g: &Geometry) {
    v.s("scene");
}

/// geometry.py's weight model on four geometries (variants).
#[test]
fn localize_weights() {
    let v = Vectors::load("localize_weights");
    let variants: Vec<String> = v.s("variants").split(',').map(str::to_string).collect();
    assert_eq!(v.u("n_variants"), variants.len());
    for p in &variants {
        check_weights(&v, &format!("{p}_"));
    }
    v.finish();
}

fn weights_scene(name: &str) {
    let v = Vectors::load(name);
    let g = check_weights(&v, "");
    check_scene_geometry(&v, &g);
    v.finish();
}

#[test]
fn localize_weights_bedroom() {
    weights_scene("localize_weights_bedroom");
}

#[test]
fn localize_weights_office() {
    weights_scene("localize_weights_office");
}

#[test]
fn localize_weights_hall() {
    weights_scene("localize_weights_hall");
}

/// invert.py's SpatialPrior: the kernel, rows, entries and refusals.
#[test]
fn localize_prior() {
    let v = Vectors::load("localize_prior");
    assert_eq!(v.s("variants"), "aniso,default");
    v.u("n_variants");
    // The grids of the docstring: PixelGrid(Room(8, 6), 0.7) and 0.5.
    for (p, pitch) in [("aniso", 0.7), ("default", 0.5)] {
        let g = grid(room(8.0, 6.0), pitch);
        let (n_x, n_y) = (g.n_x(), g.n_y());
        v.check_scalar(&format!("{p}_n_x"), n_x as f64);
        v.check_scalar(&format!("{p}_n_y"), n_y as f64);
        v.check_scalar(&format!("{p}_pitch_x"), g.pitch_x_m());
        v.check_scalar(&format!("{p}_pitch_y"), g.pitch_y_m());
        let pr = SpatialPrior::new(&g, v.f(&format!("{p}_sigma")), v.f(&format!("{p}_delta_c")))
            .expect("prior");
        v.check_scalar(&format!("{p}_sigma2"), pr.sigma2);
        let mut kern = Vec::new();
        for dix in 0..n_x {
            for diy in 0..n_y {
                kern.push(pr.entry(0, diy * n_x + dix).unwrap());
            }
        }
        v.check(&format!("{p}_kern"), &kern);
        let rows: Vec<f64> = v
            .input_usize(&format!("{p}_row_p"))
            .iter()
            .flat_map(|&q| pr.row(q).unwrap())
            .collect();
        v.check(&format!("{p}_rows"), &rows);
        let (ep, eq) = (
            v.input_usize(&format!("{p}_entry_p")),
            v.input_usize(&format!("{p}_entry_q")),
        );
        let e: Vec<f64> = ep
            .iter()
            .zip(&eq)
            .map(|(&a, &b)| pr.entry(a, b).unwrap())
            .collect();
        v.check(&format!("{p}_entry"), &e);
    }
    let docs = v.err_docs();
    let g = grid(room(8.0, 6.0), 0.5);
    let pr = SpatialPrior::new(&g, 1.0, 0.5).unwrap();
    let codes = vec![
        exc(SpatialPrior::new(&g, 0.0, 0.5)),
        exc(SpatialPrior::new(&g, f64::NAN, 0.5)),
        exc(SpatialPrior::new(&g, f64::INFINITY, 0.5)),
        exc(SpatialPrior::new(&g, 1e200, 0.5)),
        exc(SpatialPrior::new(&g, 1.0, 0.0)),
        exc(SpatialPrior::new(&g, 1.0, -1.0)),
        exc(SpatialPrior::new(&g, 1.0, f64::INFINITY)),
        exc(pr.entry(192, 0)),
        // `entry(0, -1)`: -1 is not a usize; Python checks 0 <= q < P,
        // so the out-of-range index usize::MAX (-1 wrapped) is its case.
        exc(pr.entry(0, usize::MAX)),
        exc(pr.row(192)),
    ];
    assert_eq!(codes.len(), docs.len(), "error cases: {docs:?}");
    v.check("err_code", &codes);
    v.finish();
}

/// invert.py's cholesky_factor, cholesky_solve, cholesky_inverse.
#[test]
fn localize_cholesky() {
    let v = Vectors::load("localize_cholesky");
    let (n, nn) = (v.input_usize("m_n"), v.input_usize("m_nn"));
    let (a, b) = (v.input("m_a"), v.input("m_b"));
    assert_eq!(v.u("n_cases"), n.len());
    v.check_scalar("symmetry_tol", DEFAULT_SYMMETRY_TOL);
    let (mut code, mut pivot) = (Vec::new(), Vec::new());
    let (mut g, mut x, mut inv) = (Vec::new(), Vec::new(), Vec::new());
    let (mut ia, mut ib) = (0usize, 0usize);
    let mut g0 = None;
    for c in 0..n.len() {
        assert_eq!(nn[c], n[c] * n[c]);
        let m: Vec<Vec<f64>> = (0..n[c])
            .map(|i| a[ia + i * n[c]..ia + (i + 1) * n[c]].to_vec())
            .collect();
        let rhs = b[ib..ib + n[c]].to_vec();
        ia += nn[c];
        ib += n[c];
        match cholesky_factor(&m) {
            Ok(f) => {
                code.push(0.0);
                pivot.push(-1.0);
                g.extend(f.iter().flatten());
                x.extend(cholesky_solve(&m, &rhs).expect("solve"));
                inv.extend(cholesky_inverse(&m).expect("inverse").iter().flatten());
                if g0.is_none() {
                    g0 = Some(f);
                }
            }
            Err(e) => {
                code.push(exc_code(&e));
                // The failing pivot NotPositiveDefiniteError names.
                let pv = match &e {
                    LocateError::NotPositiveDefiniteError(msg) => msg
                        .split("pivot ")
                        .nth(1)
                        .and_then(|s| s.split(" of").next())
                        .and_then(|s| s.parse::<f64>().ok())
                        .unwrap_or(-1.0),
                    _ => -1.0,
                };
                pivot.push(pv);
                g.extend(vec![0.0; nn[c]]);
                x.extend(vec![0.0; n[c]]);
                inv.extend(vec![0.0; nn[c]]);
            }
        }
    }
    v.check("m_code", &code);
    v.check("m_pivot", &pivot);
    v.check("m_g", &g);
    v.check("m_x", &x);
    v.check("m_inv", &inv);
    let docs = v.err_docs();
    let codes = vec![exc(cholesky_solve_factored(&g0.unwrap(), &[1.0, 2.0]))];
    assert_eq!(codes.len(), docs.len());
    v.check("err_code", &codes);
    v.finish();
}

fn inverter_set(name: &str) {
    let v = Vectors::load(name);
    let ids = v.input_ints("node_id");
    let (nx, ny) = (v.input("node_x"), v.input("node_y"));
    let g = Arc::new(geometry_of(
        v.f("room_w"),
        v.f("room_h"),
        v.f("pitch_m"),
        v.f("lambda_m"),
        &ids,
        &nx,
        &ny,
    ));
    let inv = Inverter::new(
        g.clone(),
        v.f("alpha"),
        v.f("sigma"),
        v.f("delta_c"),
        InverterOptions {
            residual_tol: v.f("residual_tol"),
            ..InverterOptions::default()
        },
    )
    .expect("inverter");
    let (l, p, gg) = (g.n_links(), g.n_pixels(), inv.n_independent_links);
    v.check_scalar("n_x", g.grid().n_x() as f64);
    v.check_scalar("n_y", g.grid().n_y() as f64);
    v.check_scalar("n_links", l as f64);
    v.check_scalar("n_pixels", p as f64);
    v.check_scalar("n_independent_links", gg as f64);
    v.check_scalar("n_blind", inv.blind_pixels.len() as f64);
    v.check_scalar("asymmetry", inv.asymmetry);
    v.check_scalar("mean_diag_a", inv.mean_diag_a);
    v.check_scalar("min_diag_a", inv.min_diag_a);
    v.check_scalar(
        "inverse_residual",
        inv.inverse_residual.expect("check_inverse is on"),
    );
    v.check_scalar("projection_slots", inv.projection_slots() as f64);
    v.check_scalar("projection_bytes", inv.projection_bytes() as f64);
    v.check_scalar("min_boundary_margin", min_boundary_margin(&g));
    v.check_n(
        "blind_pixels",
        &inv.blind_pixels
            .iter()
            .map(|&q| q as f64)
            .collect::<Vec<_>>(),
    );
    v.check(
        "group_len",
        &inv.link_groups
            .iter()
            .map(|q| q.len() as f64)
            .collect::<Vec<_>>(),
    );
    v.check(
        "group_idx",
        &inv.link_groups
            .iter()
            .flatten()
            .map(|&q| q as f64)
            .collect::<Vec<_>>(),
    );
    v.check("group_scale", &inv.group_scale);
    v.check(
        "a_plus_alpha",
        &inv.a_plus_alpha
            .iter()
            .flatten()
            .copied()
            .collect::<Vec<_>>(),
    );
    v.check(
        "projection",
        &inv.projection.iter().flatten().copied().collect::<Vec<_>>(),
    );
    let ys = v.input("y");
    let n_cases = v.u("n_cases");
    v.doc(&["cases"]);
    assert_eq!(ys.len(), n_cases * l);
    let xs: Vec<f64> = ys
        .chunks(l)
        .flat_map(|y| inv.solve(y).expect("solve"))
        .collect();
    v.check("x", &xs);
    v.finish();
}

#[test]
fn localize_inverter_default() {
    inverter_set("localize_inverter_default");
}

#[test]
fn localize_inverter_aniso() {
    inverter_set("localize_inverter_aniso");
}

#[test]
fn localize_inverter_fine() {
    inverter_set("localize_inverter_fine");
}

/// The Inverter's refusals on a rank-starved geometry.
#[test]
fn localize_rankstarved() {
    let v = Vectors::load("localize_rankstarved");
    let r = room(v.f("room_w"), v.f("room_h"));
    let nodes = perimeter_nodes(&r, v.f("n_nodes") as i64, v.f("inset_m"), 0.0).unwrap();
    let g =
        Arc::new(Geometry::new(r, nodes, Some(grid(r, v.f("pitch_m"))), v.f("lambda_m")).unwrap());
    v.check_scalar("n_links", g.n_links() as f64);
    v.check_scalar("n_pixels", g.n_pixels() as f64);
    v.check_scalar("min_boundary_margin", min_boundary_margin(&g));
    let (alpha, tol, check) = (
        v.input("c_alpha"),
        v.input("c_residual_tol"),
        v.input("c_check"),
    );
    assert_eq!(v.u("n_cases"), alpha.len());
    let (sigma, delta_c) = (v.f("sigma"), v.f("delta_c"));
    let mut cols: [Vec<f64>; 6] = Default::default();
    let mut first: Option<Inverter> = None;
    for c in 0..alpha.len() {
        let made = Inverter::new(
            g.clone(),
            alpha[c],
            sigma,
            delta_c,
            InverterOptions {
                residual_tol: tol[c],
                check_inverse: check[c] != 0.0,
                ..InverterOptions::default()
            },
        );
        let (code, inv) = match made {
            Ok(i) => (0.0, Some(i)),
            Err(e) => (exc_code(&e), None),
        };
        let res = inv.as_ref().and_then(|i| i.inverse_residual);
        cols[0].push(code);
        cols[1].push(if res.is_some() { 1.0 } else { 0.0 });
        cols[2].push(res.unwrap_or(0.0));
        cols[3].push(inv.as_ref().map_or(0.0, |i| i.mean_diag_a));
        cols[4].push(inv.as_ref().map_or(0.0, |i| i.min_diag_a));
        cols[5].push(inv.as_ref().map_or(0.0, |i| i.asymmetry));
        if first.is_none() {
            first = inv;
        }
    }
    for (i, k) in [
        "c_code",
        "c_has_residual",
        "c_residual",
        "c_mean_diag_a",
        "c_min_diag_a",
        "c_asymmetry",
    ]
    .iter()
    .enumerate()
    {
        v.check(k, &cols[i]);
    }
    let first = first.expect("an accepted case");
    v.check_scalar("n_independent_links", first.n_independent_links as f64);
    v.check(
        "group_len",
        &first
            .link_groups
            .iter()
            .map(|q| q.len() as f64)
            .collect::<Vec<_>>(),
    );
    v.check(
        "group_idx",
        &first
            .link_groups
            .iter()
            .flatten()
            .map(|&q| q as f64)
            .collect::<Vec<_>>(),
    );
    v.check("group_scale", &first.group_scale);
    v.finish();
}

/// invert.py's forward_project.
#[test]
fn localize_forward() {
    let v = Vectors::load("localize_forward");
    let (nx, ny) = (v.input("node_x"), v.input("node_y"));
    let ids: Vec<i64> = (0..v.f("n_nodes") as i64).collect();
    let g = geometry_of(
        v.f("room_w"),
        v.f("room_h"),
        v.f("pitch_m"),
        v.f("lambda_m"),
        &ids,
        &nx,
        &ny,
    );
    v.check_scalar("n_links", g.n_links() as f64);
    v.check_scalar("n_pixels", g.n_pixels() as f64);
    v.check_scalar("min_boundary_margin", min_boundary_margin(&g));
    let img = v.input("image");
    assert_eq!(img.len(), v.u("n_cases") * g.n_pixels());
    let y: Vec<f64> = img
        .chunks(g.n_pixels())
        .flat_map(|im| forward_project(&g, im).expect("forward"))
        .collect();
    v.check("y", &y);
    let docs = v.err_docs();
    let codes = vec![exc(forward_project(&g, &vec![0.0; g.n_pixels() - 1]))];
    assert_eq!(codes.len(), docs.len());
    v.check("err_code", &codes);
    v.finish();
}

/// track.py's extract_peak.
#[test]
fn localize_peak() {
    let v = Vectors::load("localize_peak");
    let g = Geometry::perimeter_n(8).unwrap();
    let p = g.n_pixels();
    v.check_scalar("n_pixels", p as f64);
    v.check_scalar("n_x", g.grid().n_x() as f64);
    v.check_scalar("n_y", g.grid().n_y() as f64);
    let img = v.input("image");
    let radius = v.input("radius_m");
    assert_eq!(v.u("n_cases"), radius.len());
    let mut cols: [Vec<f64>; 9] = Default::default();
    for (c, im) in img.chunks(p).enumerate() {
        let pk = extract_peak(&g, im, radius[c]).expect("extract_peak");
        for (col, x) in cols.iter_mut().zip([
            pk.x,
            pk.y,
            pk.peak,
            pk.background,
            pk.spread,
            pk.z,
            pk.peak_index.map_or(-1.0, |i| i as f64),
            pk.n_nonfinite as f64,
            pk.tie_fraction,
        ]) {
            col.push(x);
        }
    }
    for (i, k) in [
        "x",
        "y",
        "peak",
        "background",
        "spread",
        "z",
        "peak_index",
        "n_nonfinite",
        "tie_fraction",
    ]
    .iter()
    .enumerate()
    {
        v.check(k, &cols[i]);
    }
    let docs = v.err_docs();
    let case0 = &img[..p];
    let codes = vec![
        exc(extract_peak(&g, &vec![0.0; p - 1], 1.0)),
        exc(extract_peak(&g, case0, 0.0)),
        exc(extract_peak(&g, case0, f64::NAN)),
    ];
    assert_eq!(codes.len(), docs.len());
    v.check("err_code", &codes);
    v.finish();
}

/// track.py's Tracker over a recorded image sequence, two configurations.
#[test]
fn localize_tracker() {
    let v = Vectors::load("localize_tracker");
    let g = Arc::new(Geometry::perimeter_n(8).unwrap());
    let p = g.n_pixels();
    v.check_scalar("n_pixels", p as f64);
    let t = v.input("t");
    assert_eq!(v.u("n_steps"), t.len());
    let img = v.input("image");
    let images: Vec<&[f64]> = img.chunks(p).collect();
    let z: Vec<f64> = images
        .iter()
        .map(|im| extract_peak(&g, im, 1.0).expect("extract_peak").z)
        .collect();
    v.check("z", &z);
    let cz = v.input("cz_z");
    let variants: Vec<String> = v.s("variants").split(',').map(str::to_string).collect();
    assert_eq!(v.u("n_variants"), variants.len());
    for var in &variants {
        let mut tc = TrackerConfig::default();
        let head = format!("{var}_tracker.");
        let keys: Vec<String> =
            v.kv.keys()
                .filter(|k| k.starts_with(&head))
                .cloned()
                .collect();
        for k in keys {
            let x = v.f(&k);
            match &k[head.len()..] {
                "z_threshold" => tc.z_threshold = x,
                "centroid_radius_m" => tc.centroid_radius_m = x,
                "accel_psd" => tc.accel_psd = x,
                "meas_noise_m" => tc.meas_noise_m = Some(x),
                "init_speed_std_ms" => tc.init_speed_std_ms = x,
                "max_coast_s" => tc.max_coast_s = x,
                "clamp_to_room" => tc.clamp_to_room = x != 0.0,
                other => panic!("unknown Tracker argument {other}"),
            }
        }
        let mut tr = Tracker::new(g.clone(), tc).expect("tracker");
        let mut cols: [Vec<f64>; 14] = Default::default();
        for (i, im) in images.iter().enumerate() {
            let est = tr.update(im, t[i]).expect("update");
            let e = est.map_or([0.0; 7], |e| {
                [
                    e.x,
                    e.y,
                    e.sigma_m,
                    e.confidence,
                    e.cov_xx,
                    e.cov_xy,
                    e.cov_yy,
                ]
            });
            cols[0].push(if est.is_some() { 1.0 } else { 0.0 });
            for k in 0..7 {
                cols[1 + k].push(e[k]);
            }
            cols[8].push(if tr.is_tracking() { 1.0 } else { 0.0 });
            let pos = tr.position().unwrap_or((0.0, 0.0));
            let vel = tr.velocity().unwrap_or((0.0, 0.0));
            cols[9].push(pos.0);
            cols[10].push(pos.1);
            cols[11].push(vel.0);
            cols[12].push(vel.1);
            cols[13].push(tr.sigma_m().unwrap_or(0.0));
        }
        for (i, k) in [
            "est",
            "x",
            "y",
            "sigma_m",
            "confidence",
            "cov_xx",
            "cov_xy",
            "cov_yy",
            "tracking",
            "pos_x",
            "pos_y",
            "vel_x",
            "vel_y",
            "sigma",
        ]
        .iter()
        .enumerate()
        {
            v.check(&format!("{var}_{k}"), &cols[i]);
        }
        let c: Vec<f64> = cz.iter().map(|&x| tr.confidence_from_z(x)).collect();
        v.check(&format!("{var}_cz"), &c);
    }
    v.finish();
}

/// One recorded frame stream of localize_measure.
struct Frames {
    t: Vec<f64>,
    tx: Vec<i64>,
    rx: Vec<i64>,
    frames: Vec<CsiFrame>,
}

fn measure_frames(v: &Vectors, p: &str, n_sub: usize, rate: f64) -> Frames {
    let t = v.input(&format!("{p}_t"));
    let tx = v.input_ints(&format!("{p}_tx"));
    let rx = v.input_ints(&format!("{p}_rx"));
    let seq = v.input_ints(&format!("{p}_seq"));
    let rssi = v.input_ints(&format!("{p}_rssi"));
    let amp = v.input(&format!("{p}_amp"));
    assert_eq!(amp.len(), t.len() * n_sub);
    let frames = (0..t.len())
        .map(|i| {
            new_csi_frame(
                amp[i * n_sub..(i + 1) * n_sub].to_vec(),
                vec![0.0; n_sub],
                seq[i] as u64,
                rate,
                rssi[i] as i32,
                rx[i] as u8,
            )
            .expect("frame")
        })
        .collect();
    Frames { t, tx, rx, frames }
}

/// measure.py's LinkMeasurement with each LinkSignal on recorded frames.
#[test]
fn localize_measure() {
    let v = Vectors::load("localize_measure");
    let r = room(v.f("room_w"), v.f("room_h"));
    let (nx, ny) = (split_csv(v.s("node_x")), split_csv(v.s("node_y")));
    let nodes: Vec<Node> = (0..nx.len())
        .map(|i| Node::new(i as i64, nx[i], ny[i]))
        .collect();
    let g =
        Arc::new(Geometry::new(r, nodes, Some(grid(r, v.f("pitch_m"))), v.f("lambda_m")).unwrap());
    v.check_scalar("n_links", g.n_links() as f64);
    let rate = v.f("sample_rate_hz");
    let n_sub = v.u("n_sub");
    let calib = measure_frames(&v, "c", n_sub, rate);
    let occ = measure_frames(&v, "o", n_sub, rate);
    assert_eq!(v.u("n_calib"), calib.t.len());
    assert_eq!(v.u("n_occ"), occ.t.len());
    let q_after = v.input_usize("q_after");
    let q_has_t = v.input("q_has_t");
    let q_t = v.input("q_t");
    assert_eq!(v.u("n_queries"), q_after.len());
    let set_b = v.input("input_baseline");
    let set_s = v.input("input_baseline_std");
    let min_windows = v.f("min_windows") as i64;
    let variants: Vec<String> = v.s("variants").split(',').map(str::to_string).collect();
    assert_eq!(v.u("n_variants"), variants.len());
    for var in &variants {
        let pre = |k: &str| format!("{var}_{k}");
        let signal = signal_of(v.f(&pre("signal")));
        let mode = CalibrationMode::parse(v.s(&pre("mode"))).expect("mode");
        // The runs' keyword arguments (generate.py LOCALIZE_MEASURE_RUNS):
        // only `rsm` passes min_samples (4); the others take the default.
        let min_samples = if var == "rsm" { Some(4) } else { None };
        let mut meas = LinkMeasurement::new(
            g.clone(),
            MeasurementConfig {
                window_s: v.f(&pre("window_s")),
                sample_rate_hz: rate,
                calibration_mode: mode,
                min_samples,
                clamp_negative: v.f(&pre("clamp_negative")) != 0.0,
            },
            signal,
        )
        .expect("measurement");
        if v.f(&pre("set_baseline")) != 0.0 {
            meas.set_baseline(&set_b, Some(&set_s))
                .expect("set_baseline");
        } else {
            meas.calibrate_from(
                (0..calib.t.len())
                    .map(|i| (calib.t[i], calib.tx[i], calib.rx[i], &calib.frames[i])),
                min_windows,
            )
            .expect("calibrate_from");
        }
        v.check_scalar(&pre("min_samples"), meas.min_samples as f64);
        v.check_scalar(
            &pre("is_calibrated"),
            if meas.is_calibrated() { 1.0 } else { 0.0 },
        );
        v.check(&pre("baseline"), &meas.baseline_vector());
        v.check(&pre("baseline_std"), &meas.baseline_std_vector());
        v.check(&pre("baseline_raw_std"), &meas.baseline_raw_std_vector());
        v.check_n(
            &pre("calibrated_mask"),
            &meas
                .calibrated_mask()
                .iter()
                .map(|&b| u8::from(b))
                .collect::<Vec<_>>(),
        );
        let mut fed = 0usize;
        let mut cols: [Vec<f64>; 4] = Default::default();
        for q in 0..q_after.len() {
            while fed < q_after[q] {
                meas.update(occ.t[fed], occ.tx[fed], occ.rx[fed], &occ.frames[fed])
                    .expect("update");
                fed += 1;
            }
            let t = if q_has_t[q] != 0.0 {
                Some(q_t[q])
            } else {
                None
            };
            cols[0].extend(meas.vector(t));
            cols[1].extend(meas.sample_counts(t).iter().map(|&c| c as f64));
            cols[2].extend(
                meas.ready_mask(t)
                    .iter()
                    .map(|&b| if b { 1.0 } else { 0.0 }),
            );
            cols[3].extend([
                meas.clock_restarts as f64,
                meas.stale_samples as f64,
                meas.window_overflows as f64,
            ]);
        }
        v.check(&pre("vector"), &cols[0]);
        v.check(&pre("counts"), &cols[1]);
        v.check(&pre("ready"), &cols[2]);
        v.check(&pre("counters"), &cols[3]);
    }
    let docs = v.err_docs();
    let cfg = |window_s: f64, rate: f64, min_samples: Option<i64>| MeasurementConfig {
        window_s,
        sample_rate_hz: rate,
        calibration_mode: CalibrationMode::Subtract,
        min_samples,
        clamp_negative: false,
    };
    // LinkMeasurement's default signal (signal=None): TotalPowerSignal.
    let default_signal = || -> Box<dyn LinkSignal> { Box::new(TotalPowerSignal) };
    let mut m0 = LinkMeasurement::new(g.clone(), cfg(1.0, rate, None), default_signal()).unwrap();
    let fr0 = new_csi_frame(vec![1.0; n_sub], vec![0.0; n_sub], 0, rate, -40, 1).unwrap();
    let mut codes = vec![
        exc(LinkMeasurement::new(
            g.clone(),
            cfg(0.0, rate, None),
            default_signal(),
        )),
        exc(LinkMeasurement::new(
            g.clone(),
            cfg(1.0, 0.0, None),
            default_signal(),
        )),
        exc(LinkMeasurement::new(
            g.clone(),
            cfg(1.0, rate, Some(1)),
            default_signal(),
        )),
        exc(CalibrationMode::parse("median")),
    ];
    codes.push(exc(m0.update(0.0, 0, 0, &fr0)));
    codes.push(exc(m0.update(0.0, 0, 7, &fr0)));
    codes.push(exc(m0.set_baseline(&[0.1; 5], None)));
    codes.push(exc(m0.set_baseline(&[0.1, 0.1, 0.1, 0.1, 0.1, -0.1], None)));
    codes.push(exc(
        m0.set_baseline(&[0.1, 0.1, 0.1, 0.1, 0.1, f64::NAN], None)
    ));
    codes.push(exc(m0.set_baseline(&[0.1; 6], Some(&[0.1; 5]))));
    let mut fresh =
        LinkMeasurement::new(g.clone(), cfg(1.0, rate, None), default_signal()).unwrap();
    codes.push(exc(
        fresh.calibrate_from(Vec::<(f64, i64, i64, CsiFrame)>::new(), 10)
    ));
    codes.push(exc(new_csi_frame(vec![1.0, 2.0], vec![0.0], 0, 10.0, 0, 0)));
    codes.push(exc(new_csi_frame(vec![1.0], vec![0.0], 0, 0.0, 0, 0)));
    assert_eq!(codes.len(), docs.len(), "error cases: {docs:?}");
    v.check("err_code", &codes);
    v.finish();
}

// ===================================================================
// locate_*: the new pieces
// ===================================================================

/// §8: radius95 and the publication test; the §1 constants.
#[test]
fn locate_radius() {
    let v = Vectors::load("locate_radius");
    v.check_scalar("K95", k95());
    v.check_scalar("CHI2_2_99", chi2_2_99());
    v.check_scalar("LN10", ln10());
    let minr = v.f("min_plausible_radius_m");
    let (a, b, d) = (v.input("a"), v.input("b"), v.input("d"));
    assert_eq!(v.u("n_cases"), a.len());
    let r: Vec<f64> = (0..a.len()).map(|i| radius95(a[i], b[i], d[i])).collect();
    let pubs: Vec<f64> = r
        .iter()
        .map(|&x| if plausible(x, minr) { 1.0 } else { 0.0 })
        .collect();
    v.check("r", &r);
    v.check("pub", &pubs);
    v.finish();
}

/// §5: the weighted-link-midpoint baseline.
#[test]
fn locate_wlm() {
    let v = Vectors::load("locate_wlm");
    assert_eq!(v.s("variants"), "g0,g1");
    v.u("n_variants");
    let lambda = v.f("lambda_m");
    let mut geoms = Vec::new();
    for p in ["g0", "g1"] {
        let (x, y) = (
            v.input(&format!("{p}_node_x")),
            v.input(&format!("{p}_node_y")),
        );
        let ids: Vec<i64> = (0..x.len() as i64).collect();
        // pitch 0.5 (docstring; the WLM never reads the grid).
        let g = geometry_of(
            v.f(&format!("{p}_room_w")),
            v.f(&format!("{p}_room_h")),
            0.5,
            lambda,
            &ids,
            &x,
            &y,
        );
        let links = link_geoms(&g);
        let terms: Vec<(f64, f64, f64, f64)> =
            links.iter().map(|l| link_spread(l, lambda)).collect();
        v.check(
            &format!("{p}_a_l"),
            &terms.iter().map(|t| t.0).collect::<Vec<_>>(),
        );
        v.check(
            &format!("{p}_c_l"),
            &terms.iter().map(|t| t.1).collect::<Vec<_>>(),
        );
        v.check(
            &format!("{p}_ux"),
            &terms.iter().map(|t| t.2).collect::<Vec<_>>(),
        );
        v.check(
            &format!("{p}_uy"),
            &terms.iter().map(|t| t.3).collect::<Vec<_>>(),
        );
        v.check(
            &format!("{p}_mid_x"),
            &links.iter().map(|l| l.midpoint.0).collect::<Vec<_>>(),
        );
        v.check(
            &format!("{p}_mid_y"),
            &links.iter().map(|l| l.midpoint.1).collect::<Vec<_>>(),
        );
        v.check(
            &format!("{p}_len"),
            &links.iter().map(|l| l.length_m).collect::<Vec<_>>(),
        );
        geoms.push((g, links));
    }
    #[cfg(feature = "roomsim")]
    {
        let scene = Scene::load(repo().join(v.s("scene"))).unwrap();
        let mine = saans_locate::pipeline::person_geometry(&scene, &Config::default()).unwrap();
        v.ensure(
            mine.nodes() == geoms[1].0.nodes() && mine.room() == geoms[1].0.room(),
            || "g1 is not pipeline::person_geometry(bedroom)".into(),
        );
    }
    #[cfg(not(feature = "roomsim"))]
    v.s("scene");
    v.check_scalar("n_links", geoms[0].0.n_links() as f64);
    let gi = v.input_usize("geom");
    let sig = v.input("signal");
    let gate = v.input("gate_z");
    let fdef = v.input("floor_default");
    let floor = v.input("floor");
    let scale = v.input("radius_scale");
    let minr = v.input("min_radius");
    let ys = v.input("y");
    let ss = v.input("null_sigma");
    assert_eq!(v.u("n_cases"), gi.len());
    let mut est = EstCols::default();
    let mut outs = Vec::new();
    let mut at = 0usize;
    for c in 0..gi.len() {
        let (_, links) = &geoms[gi[c]];
        let l = links.len();
        let params = WlmParams {
            gate_z: gate[c],
            floor: if fdef[c] != 0.0 {
                Config::default().wlm_floor_for(signal_name(sig[c]))
            } else {
                floor[c]
            },
            radius_scale: scale[c],
            min_plausible_radius_m: minr[c],
        };
        let out = wlm(links, lambda, &ys[at..at + l], &ss[at..at + l], &params).expect("wlm");
        at += l;
        est.push(est_row(&v, &out));
        outs.push(out);
    }
    assert_eq!(at, ys.len());
    est.check(&v, "", "y_hat", false);
    // The last three cases (the generator's docstring): the published
    // one-diagonal case (4th from the end) with radius scale 0, -0.0 and
    // -1 against minimum radius 0, 0 and -10. Each radius is >= its
    // minimum but not > 0, so only §8's r > 0 abstains them.
    let n = outs.len();
    let e = outs[n - 4]
        .estimate()
        .expect("the one-diagonal case publishes");
    let base = radius95(e.cov_xx, e.cov_xy, e.cov_yy);
    for c in n - 3..n {
        let r = scale[c] * base;
        v.ensure(
            r >= minr[c] && r <= 0.0 && outs[c] == Outcome::Abstain(Reason::ImplausibleRadius),
            || {
                format!(
                    "locate_wlm case {c}: radius {r} (minimum {}) is not abstained by r > 0",
                    minr[c]
                )
            },
        );
    }
    v.finish();
}

/// §6.2's line-of-sight primitives.
#[test]
fn locate_los() {
    let v = Vectors::load("locate_los");
    let kind = v.input_ints("kind");
    let prim = v.input("prim");
    let (p, q) = (v.input("p"), v.input("q"));
    assert_eq!(v.u("n_rows"), kind.len());
    let mut blocked = Vec::new();
    for i in 0..kind.len() {
        let pr = &prim[8 * i..8 * i + 8];
        let a = [p[3 * i], p[3 * i + 1], p[3 * i + 2]];
        let b = [q[3 * i], q[3 * i + 1], q[3 * i + 2]];
        let hit = match kind[i] {
            0 => wall_blocks(
                a,
                b,
                &Wall {
                    a: (pr[0], pr[1]),
                    b: (pr[2], pr[3]),
                    h: pr[4],
                },
            ),
            1 => box_blocks(
                a,
                b,
                &BoxObj {
                    cx: pr[0],
                    cy: pr[1],
                    cz: pr[2],
                    sl: pr[3],
                    sb: pr[4],
                    sh: pr[5],
                    cs: pr[6],
                    sn: pr[7],
                    opening: false,
                },
            ),
            2 => cylinder_blocks(
                a,
                b,
                &Cylinder {
                    cx: pr[0],
                    cy: pr[1],
                    radius: pr[2],
                    top: pr[3],
                },
            ),
            k => panic!("kind {k}"),
        };
        blocked.push(if hit { 1.0 } else { 0.0 });
    }
    v.check("blocked", &blocked);
    // The hall's walls and boxes (rows after the 21 hand rows, 40 each)
    // are Room Sim's as saans_locate::pipeline::blockers reads them.
    let (n_walls, n_boxes) = (v.u("n_walls"), v.u("n_boxes"));
    #[cfg(feature = "roomsim")]
    {
        let (sim, _) = hall_sim(&v);
        let b = saans_locate::pipeline::blockers(&sim);
        let boxes: Vec<&BoxObj> = b.boxes.iter().filter(|o| !o.opening).collect();
        v.ensure(b.walls.len() == n_walls && boxes.len() == n_boxes, || {
            format!(
                "hall: Rust has {} walls and {} boxes, Python {n_walls} and {n_boxes}",
                b.walls.len(),
                boxes.len()
            )
        });
        let hand = kind.len() - 40 * (n_walls + n_boxes + 3);
        for (i, w) in b.walls.iter().enumerate() {
            let pr = &prim[8 * (hand + 40 * i)..];
            v.ensure(
                kind[hand + 40 * i] == 0 && pr[..5] == [w.a.0, w.a.1, w.b.0, w.b.1, w.h],
                || format!("hall wall {i} differs from Python's phone_walls"),
            );
        }
        for (i, o) in boxes.iter().enumerate() {
            let row = hand + 40 * (n_walls + i);
            let pr = &prim[8 * row..8 * row + 8];
            v.ensure(
                kind[row] == 1 && pr == [o.cx, o.cy, o.cz, o.sl, o.sb, o.sh, o.cs, o.sn],
                || format!("hall box {i} differs from Python's phone_boxes"),
            );
        }
    }
    #[cfg(not(feature = "roomsim"))]
    {
        let _ = (n_walls, n_boxes);
        v.s("scene");
    }
    v.finish();
}

#[cfg(feature = "roomsim")]
fn hall_sim(v: &Vectors) -> (saans_roomsim::Simulator, Scene) {
    let scene = Scene::load(repo().join(v.s("scene"))).unwrap();
    let sim = saans_roomsim::Simulator::new(
        &scene,
        saans_roomsim::Config {
            duration_s: 0.0,
            ..saans_roomsim::Config::default()
        },
    )
    .expect("hall");
    (sim, scene)
}

/// The filter's KalmanStep (generate.py's `kf_*` columns) from the
/// filter state before and after one epoch and its outcome: the outcome
/// is published or `implausible_radius` exactly when the step accepted
/// (or started) the fix, and the step's position and `P[0:2][0:2]` are
/// then the state's.
fn kalman_step(
    v: &Vectors,
    cfg: &Config,
    t: f64,
    fix_ok: bool,
    before: Option<FilterState>,
    after: Option<&FilterState>,
    out: &Outcome,
) -> [f64; 8] {
    let discarded = before.is_some_and(|b| !(t - b.t_acc <= cfg.phone_max_coast_s));
    let reason = match out {
        Outcome::Estimate(_) | Outcome::Abstain(Reason::ImplausibleRadius) => None,
        Outcome::Abstain(r) => Some(*r),
    };
    let started = reason.is_none() && fix_ok && (before.is_none() || discarded);
    let (x, y, p00, p01, p11) = match (reason, after) {
        (None, Some(s)) => (s.s[0], s.s[1], s.p[0][0], s.p[0][1], s.p[1][1]),
        (None, None) => panic!("{}: an accepted step without a filter", v.name),
        _ => (0.0, 0.0, 0.0, 0.0, 0.0),
    };
    [
        v.code(reason),
        x,
        y,
        p00,
        p01,
        p11,
        f64::from(u8::from(started)),
        f64::from(u8::from(discarded)),
    ]
}

/// §6-§7 end to end on plain data: the simulated phone in the L-shaped
/// hall on a synthetic trajectory with recorded blockers.
///
/// The radio's draws, `d` and the pre-failure RTT value are not outputs of
/// the Rust `Radio`; the test draws them itself from the §6.3 streams in
/// §6.3's order (the generator recorded Python's) and evaluates §6.4's
/// formula on them, and every reported range of the Rust radio must equal
/// that value bit for bit.
#[test]
fn locate_phone_run() {
    let v = Vectors::load("locate_phone_run");
    let ids = v.input_ints("node_id");
    let (nx, ny, nz) = (v.input("node_x"), v.input("node_y"), v.input("node_z"));
    let nodes: Vec<PhoneNode> = (0..ids.len())
        .map(|i| PhoneNode {
            id: ids[i] as u8,
            x: nx[i],
            y: ny[i],
            z: nz[i],
        })
        .collect();
    assert_eq!(v.u("n_nodes"), nodes.len());
    let wall = v.input("wall");
    let bx = v.input("box");
    let blockers = Blockers {
        walls: wall
            .chunks(5)
            .map(|w| Wall {
                a: (w[0], w[1]),
                b: (w[2], w[3]),
                h: w[4],
            })
            .collect(),
        boxes: bx
            .chunks(8)
            .map(|o| BoxObj {
                cx: o[0],
                cy: o[1],
                cz: o[2],
                sl: o[3],
                sb: o[4],
                sh: o[5],
                cs: o[6],
                sn: o[7],
                opening: false,
            })
            .collect(),
    };
    assert_eq!(v.u("n_walls"), blockers.walls.len());
    assert_eq!(v.u("n_boxes"), blockers.boxes.len());
    #[cfg(feature = "roomsim")]
    {
        let (sim, _) = hall_sim(&v);
        let b = saans_locate::pipeline::blockers(&sim);
        let solid: Vec<BoxObj> = b.boxes.iter().filter(|o| !o.opening).copied().collect();
        v.ensure(b.walls == blockers.walls && solid == blockers.boxes, || {
            "pipeline::blockers(hall) differs from Python's phone_walls/phone_boxes".into()
        });
        v.ensure(saans_locate::pipeline::phone_nodes(&sim) == nodes, || {
            "pipeline::phone_nodes(hall) differs from Python's phone_nodes".into()
        });
    }
    #[cfg(not(feature = "roomsim"))]
    v.s("scene");
    let st = v.input("s_t");
    let (sx, sy, sz) = (v.input("s_x"), v.input("s_y"), v.input("s_z"));
    let ncyl = v.input_usize("s_ncyl");
    let cyl = v.input("cyl");
    assert_eq!(v.u("n_samples"), st.len());
    let mut cyls: Vec<Vec<Cylinder>> = Vec::new();
    let mut at = 0;
    for &n in &ncyl {
        cyls.push(
            cyl[at..at + 4 * n]
                .chunks(4)
                .map(|c| Cylinder {
                    cx: c[0],
                    cy: c[1],
                    radius: c[2],
                    top: c[3],
                })
                .collect(),
        );
        at += 4 * n;
    }
    let (txp, fc) = (v.f("tx_power_dbm"), v.f("carrier_hz"));
    let variants: Vec<String> = v.s("variants").split(',').map(str::to_string).collect();
    assert_eq!(v.u("n_variants"), variants.len());
    for var in &variants {
        let pre = |k: &str| format!("{var}_{k}");
        // LocateConfig(**kw).resolved(0, tx_power_dbm, carrier_hz): a phone
        // run outside Room Sim, so a set phone_seed is kept.
        let mut kw = Config::default();
        apply_cfg(&v, &format!("{var}_cfg."), &mut kw);
        let cfg = kw.resolved(0, txp, fc).expect("resolved");
        let seed = cfg.phone_seed.unwrap();
        assert_eq!(seed, v.f(&pre("phone_seed")) as u64);
        v.check_scalar(&pre("rssi_p0_dbm"), cfg.rssi_p0().unwrap());
        v.check_scalar(&pre("pl0"), cfg.pl0_db().unwrap());
        let mut run = PhoneRun::new(&cfg, &nodes, &blockers, "SIM").expect("phone run");
        let offs = run.radio().offsets();
        v.check(
            &pre("offset_m"),
            &offs.iter().map(|o| o.0).collect::<Vec<_>>(),
        );
        v.check(
            &pre("device_db"),
            &offs.iter().map(|o| o.1).collect::<Vec<_>>(),
        );
        // The §6.3 streams, drawn by the test.
        let mut streams: Vec<(Rng, Rng)> = nodes
            .iter()
            .zip(&offs)
            .map(|(n, &(off_n, dev_n))| {
                let mut rt = Rng::stream(seed, 300_000 + u64::from(n.id));
                let mut rs = Rng::stream(seed, 310_000 + u64::from(n.id));
                let off = cfg.rtt_offset_sigma_m * rt.gauss();
                let dev = cfg.rssi_device_sigma_db * rs.gauss();
                v.ensure(
                    off.to_bits() == off_n.to_bits() && dev.to_bits() == dev_n.to_bits(),
                    || {
                        format!(
                            "{var}: node {}: the radio's creation draws are not §6.3's",
                            n.id
                        )
                    },
                );
                (rt, rs)
            })
            .collect();
        let mut node_cols: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
        let mut meth_cols: BTreeMap<String, Vec<f64>> = BTreeMap::new();
        let (mut ep_j, mut ep_t) = (Vec::new(), Vec::new());
        for j in 0..st.len() {
            let before = (
                run.filter(Method::Rtt).state().copied(),
                run.filter(Method::Rssi).state().copied(),
            );
            let sample = TruthSample {
                t: st[j],
                phone: [sx[j], sy[j], sz[j]],
                cylinders: &cyls[j],
            };
            let Some(rec) = run.step(&sample).expect("step") else {
                continue;
            };
            // The radio's per-node diagnostics (rssi_v, shadow) and each
            // method's fix are not in the §11 record: read them from the
            // run.
            let radio_ep = run.last_epoch().expect("an epoch fired").clone();
            let (rtt_fix_k, rssi_fix_k) = run.last_fixes().expect("an epoch fired");
            ep_j.push(j as f64);
            ep_t.push(rec.t);
            for (k, ((rep, rr), node)) in radio_ep
                .reports
                .iter()
                .zip(&rec.reports)
                .zip(&nodes)
                .enumerate()
            {
                assert_eq!(rep.node, node.id);
                v.ensure(ReportRecord::from(rep) == *rr, || {
                    format!("{var}: sample {j}: the record's report is not the radio's")
                });
                let (rt, rs) = &mut streams[k];
                let draws = [
                    rt.uniform(),
                    rt.gauss(),
                    rt.uniform(),
                    rt.uniform(),
                    rt.uniform(),
                    rs.uniform(),
                    rs.gauss(),
                    rs.uniform(),
                ];
                let [u_fail, g, u_out, u_mag, u_b, u_fail_s, g_s, u_j] = draws;
                let (px, py, pz) = (sx[j], sy[j], sz[j]);
                let d = (((px - node.x) * (px - node.x) + (py - node.y) * (py - node.y))
                    + (pz - node.z) * (pz - node.z))
                    .sqrt();
                let sig = if rep.blocked {
                    cfg.rtt_sigma_m * cfg.rtt_nlos_sigma_scale
                } else {
                    cfg.rtt_sigma_m
                };
                let mut r = (d + offs[k].0) + sig * g;
                if rep.blocked {
                    r += -(cfg.rtt_nlos_bias_m * (1.0 - u_b).ln());
                }
                if u_out < cfg.rtt_outlier_p {
                    r += cfg.rtt_outlier_lo_m
                        + (cfg.rtt_outlier_hi_m - cfg.rtt_outlier_lo_m) * u_mag;
                }
                let r = if r > 0.0 { r } else { 0.0 };
                if let Some(m) = rep.rtt_m {
                    v.ensure(m.to_bits() == r.to_bits(), || {
                        format!("{var}: epoch at sample {j}, node {}: the radio's range {m} is not §6.4's {r}", node.id)
                    });
                }
                // §6.5: a failed draw or a non-finite `v` gives no RSSI
                // report (its draws and `S` are kept).
                v.ensure(
                    rep.rtt_m.is_none() == (u_fail < cfg.rtt_fail_p)
                        && rep.rssi_dbm.is_none()
                            == (u_fail_s < cfg.rssi_fail_p || !rep.rssi_v.is_finite()),
                    || {
                        format!(
                            "{var}: sample {j}, node {}: failures disagree with the draws",
                            node.id
                        )
                    },
                );
                // §6.5: a report is `clamp(round_half_even(v), -100, 0)`.
                if let Some(q) = rep.rssi_dbm {
                    let want = rep.rssi_v.round_ties_even().clamp(-100.0, 0.0);
                    v.ensure(f64::from(q) == want, || {
                        format!(
                            "{var}: sample {j}, node {}: the radio's RSSI {q} is not §6.5's {want} (v = {})",
                            node.id, rep.rssi_v
                        )
                    });
                }
                for (name, x) in [
                    ("blocked", f64::from(u8::from(rep.blocked))),
                    ("d", d),
                    ("u_fail", u_fail),
                    ("g", g),
                    ("u_out", u_out),
                    ("u_mag", u_mag),
                    ("u_b", u_b),
                    ("u_fail_s", u_fail_s),
                    ("g_s", g_s),
                    ("u_j", u_j),
                    ("rtt_value", r),
                    ("rtt_present", f64::from(u8::from(rep.rtt_m.is_some()))),
                    ("rtt_m", rep.rtt_m.unwrap_or(0.0)),
                    ("rssi_v", rep.rssi_v),
                    ("rssi_present", f64::from(u8::from(rep.rssi_dbm.is_some()))),
                    ("rssi_dbm", rep.rssi_dbm.map_or(0.0, f64::from)),
                    ("shadow", rep.shadow_db),
                ] {
                    node_cols.entry(name).or_default().push(x);
                }
            }
            for (m, meth, fix, out, pre_state) in [
                ("rtt", Method::Rtt, &rtt_fix_k, &rec.rtt, before.0),
                ("rssi", Method::Rssi, &rssi_fix_k, &rec.rssi, before.1),
            ] {
                let mut push =
                    |k: &str, x: f64| meth_cols.entry(format!("{m}_{k}")).or_default().push(x);
                match fix {
                    Ok(f) => {
                        push("fix_reason", -1.0);
                        for (k, x) in [
                            ("fix_x", f.x),
                            ("fix_y", f.y),
                            ("fix_inv00", f.inv_n[0][0]),
                            ("fix_inv01", f.inv_n[0][1]),
                            ("fix_inv11", f.inv_n[1][1]),
                            ("fix_r00", f.r[0][0]),
                            ("fix_r01", f.r[0][1]),
                            ("fix_r11", f.r[1][1]),
                            ("fix_chi2", f.chi2.unwrap_or(0.0)),
                            ("fix_iter", f.iterations as f64),
                        ] {
                            push(k, x);
                        }
                    }
                    Err(r) => {
                        push("fix_reason", v.code(Some(*r)));
                        for k in [
                            "fix_x",
                            "fix_y",
                            "fix_inv00",
                            "fix_inv01",
                            "fix_inv11",
                            "fix_r00",
                            "fix_r01",
                            "fix_r11",
                            "fix_chi2",
                            "fix_iter",
                        ] {
                            push(k, 0.0);
                        }
                    }
                }
                let ks = kalman_step(
                    &v,
                    &cfg,
                    rec.t,
                    fix.is_ok(),
                    pre_state,
                    run.filter(meth).state(),
                    out,
                );
                for (k, x) in [
                    "kf_reason",
                    "kf_x",
                    "kf_y",
                    "kf_p00",
                    "kf_p01",
                    "kf_p11",
                    "kf_started",
                    "kf_discarded",
                ]
                .iter()
                .zip(ks)
                {
                    push(k, x);
                }
                let row = est_row(&v, out);
                for (k, x) in ["reason", "x", "y", "radius", "cov_xx", "cov_xy", "cov_yy"]
                    .iter()
                    .zip(row)
                {
                    push(k, x);
                }
            }
        }
        v.check_scalar(&pre("n_epochs"), ep_j.len() as f64);
        v.check(&pre("ep_j"), &ep_j);
        v.check(&pre("ep_t"), &ep_t);
        for (k, col) in &node_cols {
            v.check(&pre(k), col);
        }
        for (k, col) in &meth_cols {
            v.check(&pre(k), col);
        }
    }
    // The epoch clock alone (EpochClock(5.0)): the Rust radio's firing
    // rule, a non-finite t firing nothing (Rust refuses it; Python's
    // clock returns False) and leaving e_next as it was.
    let ct = v.input("clk_t");
    let mut radio = Radio::new(
        &Config::default().resolved(0, txp, fc).unwrap(),
        &[PhoneNode {
            id: 0,
            x: 0.0,
            y: 0.0,
            z: 0.0,
        }],
        &Blockers::default(),
        "SIM",
    )
    .unwrap();
    let (mut fired, mut en) = (Vec::new(), Vec::new());
    for &t in &ct {
        let r = radio.step(&TruthSample {
            t,
            phone: [1.0, 1.0, 1.0],
            cylinders: &[],
        });
        v.ensure(r.is_ok() || !t.is_finite(), || {
            format!("clock: t = {t} refused")
        });
        fired.push(if matches!(r, Ok(Some(_))) { 1.0 } else { 0.0 });
        en.push(radio.next_epoch() as f64);
    }
    v.check("clk_fired", &fired);
    v.check("clk_e_next", &en);
    v.finish();
}

/// §7.1-7.2: one epoch's RTT and RSSI fixes.
#[test]
fn locate_phone_fix() {
    let v = Vectors::load("locate_phone_fix");
    let ns_n = v.input_usize("ns_n");
    let ns_id = v.input_ints("ns_id");
    let (ns_x, ns_y, ns_z) = (v.input("ns_x"), v.input("ns_y"), v.input("ns_z"));
    assert_eq!(v.u("n_node_sets"), ns_n.len());
    let mut sets: Vec<Vec<PhoneNode>> = Vec::new();
    let mut at = 0;
    for &n in &ns_n {
        sets.push(
            (at..at + n)
                .map(|i| PhoneNode {
                    id: ns_id[i] as u8,
                    x: ns_x[i],
                    y: ns_y[i],
                    z: ns_z[i],
                })
                .collect(),
        );
        at += n;
    }
    #[cfg(feature = "roomsim")]
    {
        let scene = Scene::load(repo().join(v.s("scene"))).unwrap();
        let sim = saans_roomsim::Simulator::new(
            &scene,
            saans_roomsim::Config {
                duration_s: 0.0,
                ..saans_roomsim::Config::default()
            },
        )
        .unwrap();
        v.ensure(saans_locate::pipeline::phone_nodes(&sim) == sets[0], || {
            "node set 0 is not pipeline::phone_nodes(bedroom)".into()
        });
    }
    #[cfg(not(feature = "roomsim"))]
    v.s("scene");
    let method = v.input_ints("method");
    let ns = v.input_usize("ns");
    let (zh, sigma, hk) = (v.input("zh"), v.input("sigma"), v.input("huber_k"));
    let (p0, n_exp) = (v.input("p0"), v.input("n_exp"));
    let n_rep = v.input_usize("n_rep");
    let rep_node = v.input_ints("rep_node");
    let rep_value = v.input("rep_value");
    assert_eq!(v.u("n_cases"), method.len());
    let mut cols: [Vec<f64>; 11] = Default::default();
    let mut fixes = Vec::new();
    let mut at = 0;
    for c in 0..method.len() {
        // The reports in the order recorded (case 9 is case 0 reversed):
        // both fixes take them nodes ascending themselves (§7).
        let reps: Vec<(i64, f64)> = (at..at + n_rep[c])
            .map(|i| (rep_node[i], rep_value[i]))
            .collect();
        at += n_rep[c];
        let ranging: Vec<Ranging> = reps
            .iter()
            .map(|&(id, value)| Ranging {
                node: *sets[ns[c]].iter().find(|n| i64::from(n.id) == id).unwrap(),
                value,
            })
            .collect();
        let fx = if method[c] == 0 {
            let cfg = Config {
                rtt_sigma_m: sigma[c],
                huber_k: hk[c],
                phone_assumed_z_m: zh[c],
                ..Config::default()
            };
            rtt_fix(&cfg, &ranging).expect("valid reports")
        } else {
            let cfg = Config {
                rssi_shadow_sigma_db: sigma[c],
                huber_k: hk[c],
                phone_assumed_z_m: zh[c],
                rssi_p0_dbm: Some(p0[c]),
                rssi_est_exponent: n_exp[c],
                ..Config::default()
            };
            rssi_fix(&cfg, &ranging).expect("valid reports")
        };
        let row = match &fx {
            Ok(f) => [
                -1.0,
                f.x,
                f.y,
                f.inv_n[0][0],
                f.inv_n[0][1],
                f.inv_n[1][1],
                f.r[0][0],
                f.r[0][1],
                f.r[1][1],
                f.chi2.unwrap_or(0.0),
                f.iterations as f64,
            ],
            Err(r) => {
                let mut z = [0.0; 11];
                z[0] = v.code(Some(*r));
                z
            }
        };
        for (col, x) in cols.iter_mut().zip(row) {
            col.push(x);
        }
        fixes.push(fx);
    }
    for (i, k) in [
        "reason",
        "x",
        "y",
        "inv00",
        "inv01",
        "inv11",
        "r00",
        "r01",
        "r11",
        "chi2",
        "iterations",
    ]
    .iter()
    .enumerate()
    {
        v.check(k, &cols[i]);
    }
    // Each fix's distance from the truth (1.7, 2.1): cases 2 (one-sided
    // Huber) and 3 (least squares).
    let off = |f: &Result<Fix, Reason>| {
        let f = f.as_ref().expect("a fix");
        let (dx, dy) = (f.x - 1.7, f.y - 2.1);
        (dx * dx + dy * dy).sqrt()
    };
    let (hs, ls) = (off(&fixes[2]), off(&fixes[3]));
    println!("SIM locate_phone_fix: one range +5 m: one-sided Huber moves the fix {hs:.4} m, least squares {ls:.4} m");
    v.check_scalar("huber_shift_m", hs);
    v.check_scalar("ls_shift_m", ls);
    // The last three cases (node set 5: the office's nodes 0, 4, 1 with two
    // y's written 0.1 + 0.2, a horizontal line but for one ulp; RTT, RSSI,
    // RSSI rounded): §7's collinear rule does not depend on the line's
    // direction, so all three are degenerate.
    let n = fixes.len();
    v.ensure(
        ns[n - 3..] == [5, 5, 5]
            && sets[5].iter().map(|p| p.y.to_bits()).collect::<Vec<_>>()
                == [
                    (0.1f64 + 0.2).to_bits(),
                    (0.1f64 + 0.2).to_bits(),
                    0.3f64.to_bits(),
                ]
            && fixes[n - 3..].iter().all(|f| *f == Err(Reason::Degenerate)),
        || "the node-set-5 cases are not the collinear-to-rounding triple, degenerate".into(),
    );
    v.finish();
}

/// §7.3: the phone's Kalman filter over a recorded sequence of fixes.
#[test]
fn locate_phone_kf() {
    let v = Vectors::load("locate_phone_kf");
    let cfg = Config::default();
    v.check_scalar("phone_accel_psd", cfg.phone_accel_psd);
    v.check_scalar("phone_init_speed_std", cfg.phone_init_speed_std);
    v.check_scalar("phone_max_coast_s", cfg.phone_max_coast_s);
    v.check_scalar("CHI2_2_99", chi2_2_99());
    let t = v.input("t");
    let ok = v.input("f_ok");
    let reason = v.input("f_reason");
    let (fx, fy) = (v.input("f_x"), v.input("f_y"));
    let (r00, r01, r11) = (v.input("f_r00"), v.input("f_r01"), v.input("f_r11"));
    assert_eq!(v.u("n_steps"), t.len());
    let mut kf = PhoneFilter::new();
    let mut cols: [Vec<f64>; 12] = Default::default();
    let (mut s, mut p) = (Vec::new(), Vec::new());
    for i in 0..t.len() {
        // PhoneKalman.step takes a Fix with inv(N) = 0 (no B term).
        let fix: Result<Fix, Reason> = if ok[i] != 0.0 {
            Ok(Fix {
                x: fx[i],
                y: fy[i],
                r: [[r00[i], r01[i]], [r01[i], r11[i]]],
                inv_n: [[0.0; 2]; 2],
                chi2: None,
                used: 0,
                iterations: 1,
                start: (0.0, 0.0),
            })
        } else {
            Err(v.reason_of(reason[i]))
        };
        let before = kf.state().copied();
        let out = kf.epoch(
            t[i],
            fix.as_ref().map_err(|r| *r),
            cfg.rtt_persist_sigma_m,
            &cfg,
        );
        let after = kf.state();
        let ks = kalman_step(&v, &cfg, t[i], ok[i] != 0.0, before, after, &out);
        for (col, x) in cols.iter_mut().zip(ks) {
            col.push(x);
        }
        cols[8].push(f64::from(u8::from(after.is_some())));
        cols[9].push(after.map_or(0.0, |a| a.t_f));
        cols[10].push(after.map_or(0.0, |a| a.t_acc));
        s.extend(after.map_or([0.0; 4], |a| a.s));
        p.extend(after.map_or([[0.0; 4]; 4], |a| a.p).iter().flatten());
    }
    for (i, k) in [
        "reason",
        "x",
        "y",
        "p00",
        "p01",
        "p11",
        "started",
        "discarded",
        "exists",
        "t_f",
        "t_acc",
    ]
    .iter()
    .enumerate()
    {
        v.check(k, &cols[i]);
    }
    v.check("s", &s);
    v.check("P", &p);
    v.finish();
}

// ===================================================================
// locate_pipeline_*: the Room Sim pipeline (§10, §11)
// ===================================================================

#[cfg(feature = "roomsim")]
fn pipeline_set(name: &str) {
    use saans_locate::phone::{EpochFixes, EpochRecord, RadioEpoch};
    use saans_locate::pipeline::{
        base_rate_hz, blockers, calibration_config, cylinders, link_signal, locate_config,
        packet_frame, person_geometry, phone_nodes, run_roomsim, tick_time, PersonLocator,
    };
    use saans_roomsim::{Config as RoomSimConfig, Posture, Simulator};

    let v = Vectors::load(name);
    let scene = Scene::load(repo().join(v.s("scene"))).unwrap();
    // The Room Sim config: the defaults, the seed, duration_s and the
    // non-default `cfg.*` values (here only n_people).
    let mut rs = RoomSimConfig {
        seed: v.f("seed") as u64,
        duration_s: v.f("duration_s"),
        ..RoomSimConfig::default()
    };
    let keys: Vec<String> =
        v.kv.keys()
            .filter(|k| k.starts_with("cfg."))
            .cloned()
            .collect();
    for k in keys {
        match k.as_str() {
            "cfg.n_people" => rs.n_people = Some(v.u(&k)),
            other => panic!("{name}: Room Sim config {other} not supported by this loader"),
        }
    }
    // The default LocateConfig, resolved for this Room Sim run (§9); a
    // `locate.*` scalar would be a non-default value.
    let mut cfg = locate_config(&rs).unwrap();
    apply_cfg(&v, "locate.", &mut cfg);
    v.check_str("person_label", "SIM");
    v.check_scalar("calib_s", cfg.calib_s);
    v.check_scalar("window_s", cfg.window_s);
    v.check_scalar("tick_hz", cfg.tick_hz);
    v.check_str("link_signal", &cfg.link_signal);
    v.check_scalar("phone_seed", cfg.phone_seed.unwrap() as f64);
    v.check_scalar("tx_power_dbm", cfg.tx_power_dbm.unwrap());
    v.check_scalar("carrier_hz", cfg.carrier_hz.unwrap());
    v.check_scalar("rssi_p0_dbm", cfg.rssi_p0().unwrap());
    v.check_scalar("wlm_floor", cfg.wlm_floor_for(&cfg.link_signal));

    let run = run_roomsim(&scene, &rs, &cfg, "SIM").expect("a valid run");
    assert_eq!(run.label, "SIM");

    // An independent locator fed the same packets (the pipeline_smoke
    // pattern), with a second measurement beside it, to read the y and
    // image each tick used.
    let sim = Simulator::new(&scene, rs.clone()).unwrap();
    let geometry = Arc::new(person_geometry(&scene, &cfg).unwrap());
    let rate_hz = base_rate_hz(&sim);
    let mut person = PersonLocator::new(geometry.clone(), &cfg, rate_hz, "SIM").unwrap();
    let mut meas = LinkMeasurement::new(
        geometry.clone(),
        MeasurementConfig {
            window_s: cfg.window_s,
            sample_rate_hz: rate_hz,
            calibration_mode: CalibrationMode::Subtract,
            min_samples: None,
            clamp_negative: false,
        },
        link_signal(&cfg.link_signal).unwrap(),
    )
    .unwrap();
    let fw = rs.fw_subcarriers;
    let calib_sim = Simulator::new(&scene, calibration_config(&rs, cfg.calib_s)).unwrap();
    let calib: Vec<(f64, u8, u8, CsiFrame)> = calib_sim
        .packets()
        .filter(|p| p.t >= rs.duration_s)
        .filter_map(|p| {
            packet_frame(&p, fw, rate_hz)
                .unwrap()
                .map(|f| (p.t, p.tx, p.rx, f))
        })
        .collect();
    person
        .calibrate_from(calib.iter().map(|(t, tx, rx, f)| (*t, *tx, *rx, f)))
        .unwrap();
    meas.calibrate_from(
        calib
            .iter()
            .map(|(t, tx, rx, f)| (*t, i64::from(*tx), i64::from(*rx), f)),
        cfg.min_windows as i64,
    )
    .unwrap();
    /// What one tick of the independent locator used and gave.
    struct TickTrace {
        t: f64,
        y: Vec<f64>,
        image: Vec<f64>,
        tracking: bool,
        wlm: Outcome,
        rti: Outcome,
    }
    let mut ticks: Vec<TickTrace> = Vec::new();
    let mut tick = |person: &mut PersonLocator, meas: &mut LinkMeasurement, t_k: f64| {
        let y = meas.vector(Some(t_k));
        let image = person.inverter().solve(&y).unwrap();
        let (wlm, rti) = person.tick(t_k).unwrap();
        ticks.push(TickTrace {
            t: t_k,
            y,
            image,
            tracking: person.tracker().is_tracking(),
            wlm,
            rti,
        });
    };
    // An independent phone fed the same truth, to read what the §11
    // epoch record leaves out (the radio's rssi_v, each method's fix).
    let mut phone_run = scene
        .phone
        .as_ref()
        .map(|_| PhoneRun::new(&cfg, &phone_nodes(&sim), &blockers(&sim), "SIM").unwrap());
    let mut phone_trace: Vec<(EpochRecord, RadioEpoch, EpochFixes)> = Vec::new();
    let mut k = 0u64;
    let mut t_k = tick_time(cfg.window_s, cfg.tick_hz, k);
    for tr in sim.run() {
        while t_k < rs.duration_s && t_k < tr.t {
            tick(&mut person, &mut meas, t_k);
            k += 1;
            t_k = tick_time(cfg.window_s, cfg.tick_hz, k);
        }
        if let (Some(pr), Some(p)) = (phone_run.as_mut(), tr.truth.phone) {
            let cyl = cylinders(&sim, &tr.truth).unwrap();
            let sample = TruthSample {
                t: tr.t,
                phone: [p.x, p.y, p.z],
                cylinders: &cyl,
            };
            if let Some(rec) = pr.step(&sample).unwrap() {
                phone_trace.push((
                    rec,
                    pr.last_epoch().unwrap().clone(),
                    pr.last_fixes().unwrap(),
                ));
            }
        }
        for p in &tr.packets {
            if let Some(f) = packet_frame(p, fw, rate_hz).unwrap() {
                person.update(p.t, p.tx, p.rx, &f).unwrap();
                meas.update(p.t, i64::from(p.tx), i64::from(p.rx), &f)
                    .unwrap();
            }
        }
    }
    while t_k < rs.duration_s {
        tick(&mut person, &mut meas, t_k);
        k += 1;
        t_k = tick_time(cfg.window_s, cfg.tick_hz, k);
    }
    v.ensure(ticks.len() == run.ticks.len(), || {
        "the independent locator ticked a different number of times".into()
    });
    for (mine, rec) in ticks.iter().zip(&run.ticks) {
        v.ensure(
            mine.t == rec.t && mine.wlm == rec.wlm && mine.rti == rec.rti,
            || {
                format!(
                    "tick {}: the independent locator disagrees with run_roomsim",
                    rec.t
                )
            },
        );
    }

    // Run scalars.
    v.check_scalar("rate_hz", run.rate_hz);
    v.check_scalar("min_samples", run.min_samples as f64);
    v.check_scalar("n_links", geometry.n_links() as f64);
    v.check_scalar("n_pixels", geometry.n_pixels() as f64);
    v.check_scalar("n_x", geometry.grid().n_x() as f64);
    v.check_scalar("n_y", geometry.grid().n_y() as f64);
    v.check_scalar("n_blind", run.diagnostics.blind_pixels as f64);
    v.check_scalar(
        "n_independent_links",
        run.diagnostics.n_independent_links as f64,
    );
    v.check_scalar("n_ticks", run.ticks.len() as f64);
    v.check_scalar("n_epochs", run.epochs.len() as f64);
    v.check_scalar("has_phone", f64::from(u8::from(run.has_phone)));
    v.check_scalar("n_people", sim.people().len() as f64);
    v.check_scalar("n_nodes", scene.nodes.len() as f64);

    // Calibration.
    v.check(
        "link_tx",
        &run.links.iter().map(|l| l.0 as f64).collect::<Vec<_>>(),
    );
    v.check(
        "link_rx",
        &run.links.iter().map(|l| l.1 as f64).collect::<Vec<_>>(),
    );
    v.check("baseline", &run.calibration.baseline);
    v.check("null_sigma", &run.calibration.null_sigma);
    v.check_n(
        "calibrated_mask",
        &run.calibration
            .calibrated_mask
            .iter()
            .map(|&b| u8::from(b))
            .collect::<Vec<_>>(),
    );
    v.check(
        "blind_pixels",
        &person
            .inverter()
            .blind_pixels
            .iter()
            .map(|&q| q as f64)
            .collect::<Vec<_>>(),
    );

    // Ticks.
    v.check("tk_t", &run.ticks.iter().map(|r| r.t).collect::<Vec<_>>());
    v.check(
        "tk_y",
        &ticks.iter().flat_map(|t| t.y.clone()).collect::<Vec<_>>(),
    );
    v.check(
        "tk_image",
        &ticks
            .iter()
            .flat_map(|t| t.image.clone())
            .collect::<Vec<_>>(),
    );
    v.check(
        "tk_z",
        &ticks
            .iter()
            .map(|t| extract_peak(&geometry, &t.image, 1.0).unwrap().z)
            .collect::<Vec<_>>(),
    );
    v.check(
        "tk_tracking",
        &ticks
            .iter()
            .map(|t| f64::from(u8::from(t.tracking)))
            .collect::<Vec<_>>(),
    );
    let (mut wl, mut rt) = (EstCols::default(), EstCols::default());
    let mut truth: [Vec<f64>; 4] = Default::default();
    for rec in &run.ticks {
        assert_eq!(rec.label, "SIM");
        wl.push(est_row(&v, &rec.wlm));
        rt.push(est_row(&v, &rec.rti));
        v.ensure(rec.truth.len() == sim.people().len(), || {
            "tick truth people".into()
        });
        for p in &rec.truth {
            truth[0].push(p.x);
            truth[1].push(p.y);
            truth[2].push(f64::from(u8::from(p.walking)));
            truth[3].push(match p.posture {
                Posture::Stand => 0.0,
                Posture::Sit => 1.0,
                Posture::Lie => 2.0,
            });
        }
    }
    wl.check(&v, "tk_wlm_", "y", true);
    rt.check(&v, "tk_rti_", "y", true);
    v.check("tk_p_x", &truth[0]);
    v.check("tk_p_y", &truth[1]);
    v.check("tk_p_walking", &truth[2]);
    v.check("tk_p_posture", &truth[3]);

    // Phone epochs.
    let mut ep: [Vec<f64>; 4] = Default::default();
    let mut rp: BTreeMap<&str, Vec<f64>> = BTreeMap::new();
    let mut meth: BTreeMap<String, Vec<f64>> = BTreeMap::new();
    v.ensure(phone_trace.len() == run.epochs.len(), || {
        "the independent phone fired a different number of epochs".into()
    });
    for (rec, (mine, radio_ep, (rtt_fix_k, rssi_fix_k))) in run.epochs.iter().zip(&phone_trace) {
        assert_eq!(rec.label, "SIM");
        v.ensure(rec == mine, || {
            format!(
                "epoch {}: the independent phone disagrees with run_roomsim",
                rec.t
            )
        });
        ep[0].push(rec.t);
        ep[1].push(rec.truth[0]);
        ep[2].push(rec.truth[1]);
        ep[3].push(rec.truth[2]);
        for (r, d) in rec.reports.iter().zip(&radio_ep.reports) {
            for (k, x) in [
                ("rp_node", f64::from(r.node)),
                ("rp_blocked", f64::from(u8::from(r.blocked))),
                ("rp_rtt_present", f64::from(u8::from(r.rtt_m.is_some()))),
                ("rp_rtt_m", r.rtt_m.unwrap_or(0.0)),
                ("rp_rssi_present", f64::from(u8::from(r.rssi_dbm.is_some()))),
                ("rp_rssi_dbm", r.rssi_dbm.map_or(0.0, f64::from)),
                ("rp_rssi_v", d.rssi_v),
            ] {
                rp.entry(k).or_default().push(x);
            }
        }
        for (m, fix, out) in [
            ("rtt", rtt_fix_k, &rec.rtt),
            ("rssi", rssi_fix_k, &rec.rssi),
        ] {
            let row = est_row(&v, out);
            for (k, x) in ["reason", "x", "y", "radius", "cov_xx", "cov_xy", "cov_yy"]
                .iter()
                .zip(row)
            {
                meth.entry(format!("ep_{m}_{k}")).or_default().push(x);
            }
            meth.entry(format!("ep_{m}_fix_reason"))
                .or_default()
                .push(v.code(fix.as_ref().err().copied()));
            // The filter step's reason: the outcome's, except that a step
            // that accepted the fix may still abstain implausible_radius.
            let kf = match out {
                Outcome::Abstain(Reason::ImplausibleRadius) | Outcome::Estimate(_) => None,
                Outcome::Abstain(r) => Some(*r),
            };
            meth.entry(format!("ep_{m}_kf_reason"))
                .or_default()
                .push(v.code(kf));
        }
    }
    v.check("ep_t", &ep[0]);
    v.check("ep_x", &ep[1]);
    v.check("ep_y", &ep[2]);
    v.check("ep_z", &ep[3]);
    for (k, col) in &rp {
        v.check(k, col);
    }
    for (k, col) in &meth {
        v.check(k, col);
    }
    let published = |o: &Outcome| o.estimate().is_some();
    println!(
        "SIM {name}: {} ticks, WLM published {}, RTI {}; {} epochs, RTT published {}, RSSI {}",
        run.ticks.len(),
        run.ticks.iter().filter(|r| published(&r.wlm)).count(),
        run.ticks.iter().filter(|r| published(&r.rti)).count(),
        run.epochs.len(),
        run.epochs.iter().filter(|r| published(&r.rtt)).count(),
        run.epochs.iter().filter(|r| published(&r.rssi)).count(),
    );
    v.finish();
}

#[cfg(feature = "roomsim")]
#[test]
fn locate_pipeline_bedroom() {
    pipeline_set("locate_pipeline_bedroom");
}

#[cfg(feature = "roomsim")]
#[test]
fn locate_pipeline_office() {
    pipeline_set("locate_pipeline_office");
}
