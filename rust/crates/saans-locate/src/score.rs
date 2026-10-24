//! The statistics of a Room Sim run (spec/livedemo.md §5, §8.2). SIM.
//!
//! One function, [`score`], computes every §5.1 figure from a run's tick
//! and epoch records. The gateway calls it on every frame over the run's
//! records with `t <= T` (§5.1), `roomsim-eval` once over the whole run
//! (§10), and a REPLAY session over the file's records before
//! [`Scores::to_json_truthless`] deletes the error figures (§5.4). There
//! is no separate incremental implementation.
//!
//! The per-record [`person_score`] and [`phone_score`] are the frame's
//! `truth` scores (§4), and [`score`] is built on them, so a drawn error
//! and a summarised one are the same number.
//!
//! No `Person.id` or `Phone.id` is read: people are `truth[i]` by index,
//! and the phone is its epoch's `(x, y)`.

use serde::{Deserialize, Serialize};
use serde_json::{Map, Value};

use saans_scene::Scene;

use crate::phone::EpochRecord;
use crate::pipeline::{PersonTruthRecord, TickRecord};
use crate::radius::Outcome;

/// Published estimates on ticks where nobody is in the room
/// (spec/livedemo.md §5.2 `FalseAlarms`; spec/locate.md §12a): as
/// published ticks per hour and as events (maximal runs of consecutive
/// published ticks) per hour. Rates are `None` (`null`) when `seconds` is
/// 0.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct FalseAlarms {
    pub ticks: usize,
    /// `ticks / tick_hz`.
    pub seconds: f64,
    pub published: usize,
    /// `published / seconds * 3600`.
    pub per_h: Option<f64>,
    pub events: usize,
    /// `events / seconds * 3600`.
    pub events_per_h: Option<f64>,
}

// ===================================================================
// Per-record scores (§4 `Score`)
// ===================================================================

/// Why a record has no error (§4): `{"unscored": "abstained" | "no_person"}`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Unscored {
    /// The estimator abstained; there is no position to score.
    Abstained,
    /// An estimate was published on a tick whose truth has no person: a
    /// false alarm (§5), never an error of 0, null or infinity.
    NoPerson,
}

impl Unscored {
    /// The wire spelling (§4), e.g. `"no_person"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Unscored::Abstained => "abstained",
            Unscored::NoPerson => "no_person",
        }
    }
}

/// One record's score against its own truth (§4 `Score`). SIM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RecordScore {
    /// The plane distance, m, from the estimate to the truth. `person` is
    /// the matched person's index for a person estimate (the nearest,
    /// lowest index on a tie), `None` for the phone.
    Scored {
        error_m: f64,
        person: Option<usize>,
    },
    Unscored(Unscored),
}

impl RecordScore {
    /// The error, if the record was scored.
    pub fn error_m(&self) -> Option<f64> {
        match self {
            RecordScore::Scored { error_m, .. } => Some(*error_m),
            RecordScore::Unscored(_) => None,
        }
    }

    /// §4's encoding: `{"error_m", "person"}` for a person estimate,
    /// `{"error_m"}` for the phone, `{"unscored"}` otherwise.
    pub fn to_json(&self) -> Value {
        match *self {
            RecordScore::Scored {
                error_m,
                person: Some(i),
            } => obj([
                ("error_m", Value::from(error_m)),
                ("person", Value::from(i)),
            ]),
            RecordScore::Scored {
                error_m,
                person: None,
            } => obj([("error_m", Value::from(error_m))]),
            RecordScore::Unscored(u) => obj([("unscored", Value::from(u.as_str()))]),
        }
    }
}

/// `sqrt(dx*dx + dy*dy)`, the plane distance of §4 (as `locate_measure`).
fn dist(ax: f64, ay: f64, bx: f64, by: f64) -> f64 {
    let dx = ax - bx;
    let dy = ay - by;
    (dx * dx + dy * dy).sqrt()
}

/// The person nearest `(x, y)` and its distance: the lowest index on a
/// tie (a later person must be strictly nearer); `None` when `people` is
/// empty. Inputs are finite: truth is, and so is every published estimate
/// (spec/locate.md §8).
fn nearest(x: f64, y: f64, people: &[PersonTruthRecord]) -> Option<(usize, f64)> {
    let mut best: Option<(usize, f64)> = None;
    for (i, p) in people.iter().enumerate() {
        let d = dist(x, y, p.x, p.y);
        let nearer = match best {
            None => true,
            Some((_, b)) => d < b,
        };
        if nearer {
            best = Some((i, d));
        }
    }
    best
}

/// A person estimate's score at its tick (§4): the plane distance to the
/// nearest person in the tick's own truth, lowest index on a tie;
/// `Unscored(Abstained)` for an abstention, `Unscored(NoPerson)` for an
/// estimate on a tick whose truth has no person.
pub fn person_score(outcome: &Outcome, truth: &[PersonTruthRecord]) -> RecordScore {
    match outcome {
        Outcome::Abstain(_) => RecordScore::Unscored(Unscored::Abstained),
        Outcome::Estimate(e) => match nearest(e.x, e.y, truth) {
            Some((i, error_m)) => RecordScore::Scored {
                error_m,
                person: Some(i),
            },
            None => RecordScore::Unscored(Unscored::NoPerson),
        },
    }
}

/// A phone estimate's score at its epoch (§4): the plane distance to the
/// epoch's own phone `(x, y)` (`truth` is `[x, y, z]`; `z` is not used);
/// `Unscored(Abstained)` for an abstention.
pub fn phone_score(outcome: &Outcome, truth: [f64; 3]) -> RecordScore {
    match outcome {
        Outcome::Abstain(_) => RecordScore::Unscored(Unscored::Abstained),
        Outcome::Estimate(e) => RecordScore::Scored {
            error_m: dist(e.x, e.y, truth[0], truth[1]),
            person: None,
        },
    }
}

// ===================================================================
// The reference point (§5.2 `ref`)
// ===================================================================

/// The footprint's area centroid (§5.2): the shoelace centroid of
/// `room.polygon` for an outlined room (at least 3 vertices), else the
/// centre of the `w_m` by `d_m` rectangle. Ported from `locate_measure`'s
/// `footprint_centroid`, with the same result.
pub fn footprint_centroid(scene: &Scene) -> (f64, f64) {
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

// ===================================================================
// Summaries (§5.2)
// ===================================================================

/// The median of an ascending sample (§5.2): the middle value, or the
/// mean of the two middle values for an even count; `None` when empty.
fn median(sorted: &[f64]) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        None
    } else if n % 2 == 1 {
        Some(sorted[n / 2])
    } else {
        Some(0.5 * (sorted[n / 2 - 1] + sorted[n / 2]))
    }
}

/// The `ceil(num/den * n)`-th smallest of an ascending sample of `n`
/// (nearest rank), with `0 < num < den`; `None` when empty. The rank is
/// computed in integers, so it is exactly the spec's `ceil(0.9 m)`
/// (`9/10`) or `ceil(0.95 n)` (`19/20`).
fn nearest_rank(sorted: &[f64], num: usize, den: usize) -> Option<f64> {
    let n = sorted.len();
    if n == 0 {
        return None;
    }
    // 1 <= rank <= n for n >= 1 and 0 < num < den.
    let rank = (num * n).div_ceil(den);
    Some(sorted[rank - 1])
}

/// `xs` ascending (a total order; the inputs are finite).
fn ascending(mut xs: Vec<f64>) -> Vec<f64> {
    xs.sort_by(f64::total_cmp);
    xs
}

/// `k / n`, or `None` when `n` is 0.
fn fraction(k: usize, n: usize) -> Option<f64> {
    (n > 0).then(|| k as f64 / n as f64)
}

/// The error figures of a set of scored records (§5.2), each `None` when
/// the set is empty: the median and p90 (nearest rank, `ceil(0.9 m)`-th
/// smallest) error and the fraction with `error <= radius_m`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Accuracy {
    pub median_m: Option<f64>,
    pub p90_m: Option<f64>,
    pub inside_radius: Option<f64>,
}

impl Accuracy {
    /// The figures of `(error_m, radius_m)` pairs.
    fn of(pairs: impl Iterator<Item = (f64, f64)> + Clone) -> Accuracy {
        let errors = ascending(pairs.clone().map(|(e, _)| e).collect());
        let inside = pairs.filter(|&(e, r)| e <= r).count();
        Accuracy {
            median_m: median(&errors),
            p90_m: nearest_rank(&errors, 9, 10),
            inside_radius: fraction(inside, errors.len()),
        }
    }

    fn to_json(self) -> Value {
        obj([
            ("median_m", Value::from(self.median_m)),
            ("p90_m", Value::from(self.p90_m)),
            ("inside_radius", Value::from(self.inside_radius)),
        ])
    }
}

/// One scored estimate: its error, its radius and the reference point's
/// error against the same truth.
#[derive(Debug, Clone, Copy)]
struct Sample {
    error_m: f64,
    radius_m: f64,
    ref_error_m: f64,
}

/// The §5.2 figures of a block's scored estimates (in a block every
/// published estimate is scored, so these are also its published ones):
/// the estimator's [`Accuracy`], the radius median and the reference
/// point's [`Accuracy`] with each record's own `radius_m`.
fn summarise(samples: &[Sample]) -> (Accuracy, Option<f64>, Accuracy) {
    let own = Accuracy::of(samples.iter().map(|s| (s.error_m, s.radius_m)));
    let radii = ascending(samples.iter().map(|s| s.radius_m).collect());
    let reference = Accuracy::of(samples.iter().map(|s| (s.ref_error_m, s.radius_m)));
    (own, median(&radii), reference)
}

/// A person estimator's figures over the ticks whose truth holds at least
/// one person (§5.1 `occupied`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OccupiedStats {
    pub ticks: usize,
    pub published: usize,
    /// `published / ticks`, `None` when `ticks` is 0.
    pub coverage: Option<f64>,
    pub median_m: Option<f64>,
    pub p90_m: Option<f64>,
    pub inside_radius: Option<f64>,
    /// The median `radius_m` of the published estimates.
    pub radius_median_m: Option<f64>,
    /// The footprint centroid scored on the same records (§5.2 `ref`).
    pub reference: Accuracy,
}

/// One person estimator's §5.1 block, without `empty_run` (the caller
/// adds it, §5.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PersonStats {
    /// Every tick given, occupied or not.
    pub ticks: usize,
    pub occupied: OccupiedStats,
    /// The ticks whose truth has no person.
    pub unoccupied: FalseAlarms,
}

/// One phone method's §5.1 block. Every published phone estimate is
/// scored.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PhoneStats {
    pub epochs: usize,
    pub published: usize,
    /// `published / epochs`, `None` when `epochs` is 0.
    pub coverage: Option<f64>,
    pub median_m: Option<f64>,
    pub p90_m: Option<f64>,
    pub inside_radius: Option<f64>,
    pub radius_median_m: Option<f64>,
    pub reference: Accuracy,
}

/// The §5.1 `stats` block of a run's records, without `empty_run`. SIM.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Scores {
    pub wlm: PersonStats,
    pub rti: PersonStats,
    /// `None` when the run has no phone.
    pub rtt: Option<PhoneStats>,
    pub rssi: Option<PhoneStats>,
}

/// The keys REPLAY deletes from `occupied` and from each phone block
/// (§5.4).
const TRUTH_KEYS: [&str; 4] = ["median_m", "p90_m", "inside_radius", "ref"];

impl Scores {
    /// §5.1's JSON, key for key: `ref` for [`OccupiedStats::reference`] and
    /// [`PhoneStats::reference`], `null` for every `None`, and no `rtt` or
    /// `rssi` key when the run has no phone. `empty_run` is not part of
    /// [`Scores`]; the caller adds it to `wlm` and `rti` (§5.3).
    pub fn to_json(&self) -> Value {
        let mut m = Map::new();
        m.insert("wlm".into(), self.wlm.to_json());
        m.insert("rti".into(), self.rti.to_json());
        if let Some(p) = &self.rtt {
            m.insert("rtt".into(), p.to_json());
        }
        if let Some(p) = &self.rssi {
            m.insert("rssi".into(), p.to_json());
        }
        Value::Object(m)
    }

    /// [`Scores::to_json`] for REPLAY (§5.4): `median_m`, `p90_m`,
    /// `inside_radius` and `ref` deleted (not nulled) from `occupied` and
    /// from each phone block. `ticks`, `published`, `coverage`,
    /// `radius_median_m`, `unoccupied` and the phone's `epochs` stay.
    pub fn to_json_truthless(&self) -> Value {
        let mut v = self.to_json();
        for (block, inner) in [
            ("wlm", Some("occupied")),
            ("rti", Some("occupied")),
            ("rtt", None),
            ("rssi", None),
        ] {
            let target = v
                .get_mut(block)
                .and_then(|b| match inner {
                    Some(k) => b.get_mut(k),
                    None => Some(b),
                })
                .and_then(Value::as_object_mut);
            if let Some(o) = target {
                for k in TRUTH_KEYS {
                    o.remove(k);
                }
            }
        }
        v
    }
}

impl PersonStats {
    fn to_json(self) -> Value {
        let o = &self.occupied;
        obj([
            ("ticks", Value::from(self.ticks)),
            (
                "occupied",
                obj([
                    ("ticks", Value::from(o.ticks)),
                    ("published", Value::from(o.published)),
                    ("coverage", Value::from(o.coverage)),
                    ("median_m", Value::from(o.median_m)),
                    ("p90_m", Value::from(o.p90_m)),
                    ("inside_radius", Value::from(o.inside_radius)),
                    ("radius_median_m", Value::from(o.radius_median_m)),
                    ("ref", o.reference.to_json()),
                ]),
            ),
            ("unoccupied", false_alarms_json(&self.unoccupied)),
        ])
    }
}

impl PhoneStats {
    fn to_json(self) -> Value {
        obj([
            ("epochs", Value::from(self.epochs)),
            ("published", Value::from(self.published)),
            ("coverage", Value::from(self.coverage)),
            ("median_m", Value::from(self.median_m)),
            ("p90_m", Value::from(self.p90_m)),
            ("inside_radius", Value::from(self.inside_radius)),
            ("radius_median_m", Value::from(self.radius_median_m)),
            ("ref", self.reference.to_json()),
        ])
    }
}

/// §5.1 `FalseAlarms`, the same keys and values as its serde encoding.
fn false_alarms_json(fa: &FalseAlarms) -> Value {
    obj([
        ("ticks", Value::from(fa.ticks)),
        ("seconds", Value::from(fa.seconds)),
        ("published", Value::from(fa.published)),
        ("per_h", Value::from(fa.per_h)),
        ("events", Value::from(fa.events)),
        ("events_per_h", Value::from(fa.events_per_h)),
    ])
}

/// A JSON object of these keys and values (no `unwrap` on the way).
fn obj<const N: usize>(pairs: [(&str, Value); N]) -> Value {
    Value::Object(pairs.into_iter().map(|(k, v)| (k.to_owned(), v)).collect())
}

// ===================================================================
// score (§8.2)
// ===================================================================

/// A person estimator of a tick record.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PersonEstimator {
    /// The weighted-link-midpoint baseline (spec/locate.md §5).
    Wlm,
    /// RTI (spec/locate.md §10.1).
    Rti,
}

impl PersonEstimator {
    pub const ALL: [PersonEstimator; 2] = [PersonEstimator::Wlm, PersonEstimator::Rti];

    /// The wire name, `"wlm"` or `"rti"`.
    pub fn as_str(self) -> &'static str {
        match self {
            PersonEstimator::Wlm => "wlm",
            PersonEstimator::Rti => "rti",
        }
    }

    /// This estimator's outcome at `tick`.
    pub fn outcome(self, tick: &TickRecord) -> &Outcome {
        match self {
            PersonEstimator::Wlm => &tick.wlm,
            PersonEstimator::Rti => &tick.rti,
        }
    }
}

/// The phone methods of an epoch record.
#[derive(Debug, Clone, Copy)]
enum PhoneMethod {
    Rtt,
    Rssi,
}

impl PhoneMethod {
    fn outcome(self, epoch: &EpochRecord) -> &Outcome {
        match self {
            PhoneMethod::Rtt => &epoch.rtt,
            PhoneMethod::Rssi => &epoch.rssi,
        }
    }
}

/// The false alarms of `which` over the unoccupied ticks of `ticks` (those
/// whose truth has no person; §5.2 `FalseAlarms`). `ticks` are one run's
/// records in order; an event is a maximal run of published unoccupied
/// ticks at consecutive indices of `ticks`, so an occupied tick ends one.
/// `seconds = ticks / tick_hz`, with `tick_hz` the run's (`> 0`); the
/// rates are `None` unless `seconds > 0`.
///
/// This is [`PersonStats::unoccupied`], and also the `empty_run` of §5.3
/// when given the `empty` variant's ticks, where every tick is unoccupied.
pub fn false_alarms(ticks: &[TickRecord], which: PersonEstimator, tick_hz: f64) -> FalseAlarms {
    let (mut n, mut published, mut events) = (0usize, 0usize, 0usize);
    let mut prev = false;
    for tick in ticks {
        let unoccupied = tick.truth.is_empty();
        let alarm = unoccupied && which.outcome(tick).estimate().is_some();
        n += usize::from(unoccupied);
        published += usize::from(alarm);
        events += usize::from(alarm && !prev);
        prev = alarm;
    }
    let seconds = n as f64 / tick_hz;
    let per_h = |k: usize| (seconds > 0.0).then(|| k as f64 / seconds * 3600.0);
    FalseAlarms {
        ticks: n,
        seconds,
        published,
        per_h: per_h(published),
        events,
        events_per_h: per_h(events),
    }
}

/// The scored estimates of `which` on occupied ticks, with the reference
/// point's error against the same truth.
fn person_samples(
    ticks: &[TickRecord],
    which: PersonEstimator,
    reference: (f64, f64),
) -> Vec<Sample> {
    let mut out = Vec::new();
    for tick in ticks {
        let outcome = which.outcome(tick);
        let (Some(e), Some(error_m)) = (
            outcome.estimate(),
            person_score(outcome, &tick.truth).error_m(),
        ) else {
            continue;
        };
        if let Some((_, ref_error_m)) = nearest(reference.0, reference.1, &tick.truth) {
            out.push(Sample {
                error_m,
                radius_m: e.radius_m,
                ref_error_m,
            });
        }
    }
    out
}

fn person_stats(
    ticks: &[TickRecord],
    which: PersonEstimator,
    reference: (f64, f64),
    tick_hz: f64,
) -> PersonStats {
    let occupied = ticks.iter().filter(|t| !t.truth.is_empty()).count();
    let samples = person_samples(ticks, which, reference);
    let (own, radius_median_m, reference) = summarise(&samples);
    PersonStats {
        ticks: ticks.len(),
        occupied: OccupiedStats {
            ticks: occupied,
            published: samples.len(),
            coverage: fraction(samples.len(), occupied),
            median_m: own.median_m,
            p90_m: own.p90_m,
            inside_radius: own.inside_radius,
            radius_median_m,
            reference,
        },
        unoccupied: false_alarms(ticks, which, tick_hz),
    }
}

/// The published estimates of `method` (every one is scored), each as its
/// [`phone_score`] error, its `radius_m` and its epoch's truth.
fn phone_scored(
    epochs: &[EpochRecord],
    method: PhoneMethod,
) -> impl Iterator<Item = (f64, f64, [f64; 3])> + '_ {
    epochs.iter().filter_map(move |ep| {
        let outcome = method.outcome(ep);
        let e = outcome.estimate()?;
        let error_m = phone_score(outcome, ep.truth).error_m()?;
        Some((error_m, e.radius_m, ep.truth))
    })
}

fn phone_stats(epochs: &[EpochRecord], method: PhoneMethod, reference: (f64, f64)) -> PhoneStats {
    let samples: Vec<Sample> = phone_scored(epochs, method)
        .map(|(error_m, radius_m, truth)| Sample {
            error_m,
            radius_m,
            ref_error_m: dist(reference.0, reference.1, truth[0], truth[1]),
        })
        .collect();
    let (own, radius_median_m, reference) = summarise(&samples);
    PhoneStats {
        epochs: epochs.len(),
        published: samples.len(),
        coverage: fraction(samples.len(), epochs.len()),
        median_m: own.median_m,
        p90_m: own.p90_m,
        inside_radius: own.inside_radius,
        radius_median_m,
        reference,
    }
}

/// The §5 figures of a run's records (§8.2). SIM.
///
/// `ticks` and `epochs` are one run's records in order (the gateway passes
/// those with `t <= T`); `has_phone` is whether the run has a phone (when
/// false, `rtt` and `rssi` are `None` and `epochs` is not read); `tick_hz`
/// is the run's Locate `tick_hz`. The reference point is
/// [`footprint_centroid`] of `scene`.
pub fn score(
    scene: &Scene,
    ticks: &[TickRecord],
    epochs: &[EpochRecord],
    has_phone: bool,
    tick_hz: f64,
) -> Scores {
    let reference = footprint_centroid(scene);
    let phone = |m| has_phone.then(|| phone_stats(epochs, m, reference));
    Scores {
        wlm: person_stats(ticks, PersonEstimator::Wlm, reference, tick_hz),
        rti: person_stats(ticks, PersonEstimator::Rti, reference, tick_hz),
        rtt: phone(PhoneMethod::Rtt),
        rssi: phone(PhoneMethod::Rssi),
    }
}

// ===================================================================
// radius_scale (§8.2, §10)
// ===================================================================

/// The fewest scored estimates for which [`radius_scale`] suggests a scale.
pub const RADIUS_SCALE_MIN_N: usize = 20;

/// The suggested radius scale per estimator (§8.2): reported by
/// `roomsim-eval`, never applied (§10). `None` when fewer than
/// [`RADIUS_SCALE_MIN_N`] estimates were scored, or for the phone when the
/// run has none.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RadiusScales {
    pub wlm: Option<f64>,
    pub rti: Option<f64>,
    pub rtt: Option<f64>,
    pub rssi: Option<f64>,
}

/// The `ceil(0.95 n)`-th smallest of `ratios`, or `None` when `n < 20`.
fn scale_of(ratios: Vec<f64>) -> Option<f64> {
    if ratios.len() < RADIUS_SCALE_MIN_N {
        return None;
    }
    nearest_rank(&ascending(ratios), 19, 20)
}

/// The `ceil(0.95 n)`-th smallest `error / radius_m` over each
/// estimator's `n` scored estimates (§8.2): the factor by which the radii
/// would have to grow for 95% of them to hold their truth. The arguments
/// are [`score`]'s; the scored estimates are the same (person estimates on
/// occupied ticks; every published phone estimate when `has_phone`), and
/// the `scene` itself enters no figure here.
pub fn radius_scale(
    scene: &Scene,
    ticks: &[TickRecord],
    epochs: &[EpochRecord],
    has_phone: bool,
) -> RadiusScales {
    let _ = scene;
    let person = |which: PersonEstimator| {
        let ratios = ticks
            .iter()
            .filter_map(|tick| {
                let outcome = which.outcome(tick);
                let e = outcome.estimate()?;
                let error_m = person_score(outcome, &tick.truth).error_m()?;
                Some(error_m / e.radius_m)
            })
            .collect();
        scale_of(ratios)
    };
    let phone = |method| {
        has_phone
            .then(|| {
                scale_of(
                    phone_scored(epochs, method)
                        .map(|(error_m, radius_m, _)| error_m / radius_m)
                        .collect(),
                )
            })
            .flatten()
    };
    RadiusScales {
        wlm: person(PersonEstimator::Wlm),
        rti: person(PersonEstimator::Rti),
        rtt: phone(PhoneMethod::Rtt),
        rssi: phone(PhoneMethod::Rssi),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use saans_roomsim::Posture;

    fn person(x: f64, y: f64) -> PersonTruthRecord {
        PersonTruthRecord {
            x,
            y,
            walking: false,
            posture: Posture::Stand,
        }
    }

    #[test]
    fn median_is_the_middle_or_the_mean_of_the_two_middles() {
        assert_eq!(median(&[]), None);
        assert_eq!(median(&[4.0]), Some(4.0));
        assert_eq!(median(&[1.0, 2.0, 10.0]), Some(2.0));
        assert_eq!(median(&[1.0, 2.0, 4.0, 10.0]), Some(3.0));
    }

    #[test]
    fn nearest_rank_counts_from_one() {
        let v: Vec<f64> = (1..=10).map(f64::from).collect();
        assert_eq!(nearest_rank(&v[..1], 9, 10), Some(1.0));
        assert_eq!(nearest_rank(&v[..9], 9, 10), Some(9.0));
        assert_eq!(nearest_rank(&v, 9, 10), Some(9.0));
        assert_eq!(nearest_rank(&[], 9, 10), None);
    }

    /// The integer ranks equal the float formulas `ceil(0.9 m)` and
    /// `ceil(0.95 n)` (as `locate_measure` writes them) for every count a
    /// run can have, so the two readings of §5.2 and §8.2 never differ.
    #[test]
    fn integer_ranks_equal_the_float_ceilings() {
        for m in 1..=200_000usize {
            assert_eq!(
                (9 * m).div_ceil(10),
                (0.9 * m as f64).ceil() as usize,
                "m {m}"
            );
            assert_eq!(
                (19 * m).div_ceil(20),
                (0.95 * m as f64).ceil() as usize,
                "n {m}"
            );
        }
    }

    #[test]
    fn nearest_person_takes_the_lowest_index_on_a_tie() {
        let people = [person(3.0, 0.0), person(0.0, 4.0), person(-3.0, 0.0)];
        assert_eq!(nearest(0.0, 0.0, &people), Some((0, 3.0)));
        let people = [person(0.0, 4.0), person(-3.0, 0.0), person(3.0, 0.0)];
        assert_eq!(nearest(0.0, 0.0, &people), Some((1, 3.0)));
        assert_eq!(nearest(0.0, 0.0, &[]), None);
    }
}
