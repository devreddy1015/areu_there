//! Scoring a run against ground truth. Every function here reports both
//! what was caught and what was invented (ADR-005 rule 5): a detector's
//! sensitivity is never printed without its false-alarm rate.

use saans_health::{EventKind, HealthEvent, SpaceSample};
use saans_sim::scenario::{Truth, TruthEvent, TruthKind};

/// One run's raw material.
pub struct Run {
    /// Scenario name; kept for debugging output.
    #[allow(dead_code)]
    pub label: String,
    pub hours: f64,
    pub samples: Vec<(Truth, SpaceSample)>,
    pub events: Vec<HealthEvent>,
    pub truth_events: Vec<TruthEvent>,
}

/// Seconds after start excluded from scoring: filter warmup (50 s) plus
/// presence calibration (30 s) plus margin. The empty room is required then.
pub const SETTLE_S: f64 = 95.0;
/// Seconds either side of a truth transition not scored: presence holds for
/// 10 s by design, and the 8 s energy window needs time to fill.
pub const TRANSITION_GUARD_S: f64 = 15.0;

#[derive(Debug, Default, Clone, Copy)]
pub struct Occupancy {
    pub present_occ: u64,
    pub present_empty: u64,
    pub present_unknown: u64,
    pub still_present: u64,
    pub still_false_empty: u64,
    pub absent_occ: u64,
    pub absent_empty: u64,
    pub absent_unknown: u64,
}

impl Occupancy {
    pub fn add(&mut self, o: &Occupancy) {
        self.present_occ += o.present_occ;
        self.present_empty += o.present_empty;
        self.present_unknown += o.present_unknown;
        self.still_present += o.still_present;
        self.still_false_empty += o.still_false_empty;
        self.absent_occ += o.absent_occ;
        self.absent_empty += o.absent_empty;
        self.absent_unknown += o.absent_unknown;
    }
    pub fn accuracy(&self) -> f64 {
        let ok = self.present_occ + self.absent_empty;
        let n =
            ok + self.present_empty + self.absent_occ + self.present_unknown + self.absent_unknown;
        ratio(ok, n)
    }
    pub fn false_empty_still(&self) -> f64 {
        ratio(self.still_false_empty, self.still_present)
    }
    pub fn false_occupied(&self) -> f64 {
        ratio(
            self.absent_occ,
            self.absent_occ + self.absent_empty + self.absent_unknown,
        )
    }
    pub fn unknown(&self) -> f64 {
        ratio(self.present_unknown + self.absent_unknown, self.total())
    }
    pub fn total(&self) -> u64 {
        self.present_occ
            + self.present_empty
            + self.present_unknown
            + self.absent_occ
            + self.absent_empty
            + self.absent_unknown
    }
}

pub fn ratio(a: u64, b: u64) -> f64 {
    if b == 0 {
        f64::NAN
    } else {
        a as f64 / b as f64
    }
}

fn near_transition(run: &Run, t: f64) -> bool {
    run.truth_events
        .iter()
        .filter(|e| e.kind == TruthKind::Occupied)
        .any(|e| {
            (t - e.t_start_s).abs() < TRANSITION_GUARD_S
                || (t - e.t_end_s).abs() < TRANSITION_GUARD_S
        })
}

pub fn occupancy(run: &Run) -> Occupancy {
    let mut o = Occupancy::default();
    for (truth, s) in &run.samples {
        if truth.t_s < SETTLE_S || near_transition(run, truth.t_s) {
            continue;
        }
        match (truth.present, s.occupied) {
            (true, Some(true)) => o.present_occ += 1,
            (true, Some(false)) => o.present_empty += 1,
            (true, None) => o.present_unknown += 1,
            (false, Some(true)) => o.absent_occ += 1,
            (false, Some(false)) => o.absent_empty += 1,
            (false, None) => o.absent_unknown += 1,
        }
        if truth.present && truth.still && !truth.in_burst {
            o.still_present += 1;
            if s.occupied == Some(false) {
                o.still_false_empty += 1;
            }
        }
    }
    o
}

#[derive(Debug, Default, Clone)]
pub struct BreathingScore {
    /// Samples where a still, breathing person was present (scorable).
    pub eligible: u64,
    /// ... of which the system reported a breathing rate.
    pub reported: u64,
    /// ... of which the reported rate was within 1 BPM of truth.
    pub within_1: u64,
    pub abs_errors: Vec<f64>,
}

impl BreathingScore {
    pub fn add(&mut self, o: &BreathingScore) {
        self.eligible += o.eligible;
        self.reported += o.reported;
        self.within_1 += o.within_1;
        self.abs_errors.extend_from_slice(&o.abs_errors);
    }
    pub fn coverage(&self) -> f64 {
        ratio(self.reported, self.eligible)
    }
    pub fn accuracy_when_reported(&self) -> f64 {
        ratio(self.within_1, self.reported)
    }
    pub fn median_error(&self) -> f64 {
        median(&self.abs_errors)
    }
}

/// Breathing is scored only once the person has been still and breathing
/// for the estimator's window (45 s) plus margin: earlier, the honest
/// answer is "not yet".
pub fn breathing(run: &Run, settle_s: f64) -> BreathingScore {
    let mut b = BreathingScore::default();
    let mut steady_since: Option<f64> = None;
    for (truth, s) in &run.samples {
        let steady =
            truth.present && truth.still && truth.breathing && !truth.in_pause && !truth.in_burst;
        steady_since = if steady {
            steady_since.or(Some(truth.t_s))
        } else {
            None
        };
        let Some(since) = steady_since else { continue };
        if truth.t_s - since < settle_s || truth.t_s < SETTLE_S {
            continue;
        }
        b.eligible += 1;
        if let Some(br) = s.breathing {
            if br.confidence >= 0.4 {
                b.reported += 1;
                let err = (br.bpm - truth.bpm).abs();
                b.abs_errors.push(err);
                if err < 1.0 {
                    b.within_1 += 1;
                }
            }
        }
    }
    b
}

#[derive(Debug, Default, Clone)]
pub struct EventScore {
    pub truths: u64,
    pub caught: u64,
    pub detections: u64,
    pub correct_detections: u64,
    pub hours: f64,
    pub latency_s: Vec<f64>,
    pub duration_err_s: Vec<f64>,
}

impl EventScore {
    pub fn add(&mut self, o: &EventScore) {
        self.truths += o.truths;
        self.caught += o.caught;
        self.detections += o.detections;
        self.correct_detections += o.correct_detections;
        self.hours += o.hours;
        self.latency_s.extend_from_slice(&o.latency_s);
        self.duration_err_s.extend_from_slice(&o.duration_err_s);
    }
    pub fn sensitivity(&self) -> f64 {
        ratio(self.caught, self.truths)
    }
    pub fn precision(&self) -> f64 {
        ratio(self.correct_detections, self.detections)
    }
    pub fn false_per_hour(&self) -> f64 {
        if self.hours > 0.0 {
            (self.detections - self.correct_detections) as f64 / self.hours
        } else {
            f64::NAN
        }
    }
}

/// Interval-matching: a detection is correct if it overlaps a truth
/// interval of the given kind, widened by `slack_s` at the end (detectors
/// report on completion).
pub fn interval_events<F>(
    run: &Run,
    truth_kind: TruthKind,
    min_truth_s: f64,
    slack_s: f64,
    is_detection: F,
) -> EventScore
where
    F: Fn(&HealthEvent) -> Option<(f64, f64)>,
{
    let truths: Vec<&TruthEvent> = run
        .truth_events
        .iter()
        .filter(|e| {
            e.kind == truth_kind
                && e.t_end_s - e.t_start_s >= min_truth_s
                && e.t_start_s >= SETTLE_S
        })
        .collect();
    let dets: Vec<(f64, f64)> = run.events.iter().filter_map(&is_detection).collect();
    let overlaps =
        |d: &(f64, f64), t: &TruthEvent| d.0 <= t.t_end_s + slack_s && d.1 >= t.t_start_s;
    let mut s = EventScore {
        truths: truths.len() as u64,
        detections: dets.len() as u64,
        hours: run.hours,
        ..Default::default()
    };
    for t in &truths {
        if let Some(d) = dets.iter().find(|d| overlaps(d, t)) {
            s.caught += 1;
            s.latency_s.push(d.1 - t.t_start_s);
            s.duration_err_s
                .push(((d.1 - d.0) - (t.t_end_s - t.t_start_s)).abs());
        }
    }
    s.correct_detections = dets
        .iter()
        .filter(|d| truths.iter().any(|t| overlaps(d, t)))
        .count() as u64;
    s
}

pub fn pauses(run: &Run) -> EventScore {
    interval_events(run, TruthKind::BreathingPause, 10.0, 5.0, |e| {
        match e.kind {
            EventKind::BreathingPause { ongoing: false, .. } => {
                Some((e.t_start_s.unwrap_or(e.t_s), e.t_s))
            }
            _ => None,
        }
    })
}

/// Count events of a kind (closure) per hour of the run.
pub fn count<F: Fn(&EventKind) -> bool>(run: &Run, f: F) -> u64 {
    run.events.iter().filter(|e| f(&e.kind)).count() as u64
}

pub fn median(xs: &[f64]) -> f64 {
    if xs.is_empty() {
        return f64::NAN;
    }
    let mut v = xs.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    v[v.len() / 2]
}

pub fn pct(x: f64) -> String {
    if x.is_nan() {
        "   -  ".into()
    } else {
        format!("{:5.1}%", 100.0 * x)
    }
}

pub fn num(x: f64, digits: usize) -> String {
    if x.is_nan() {
        "-".into()
    } else {
        format!("{x:.digits$}")
    }
}
