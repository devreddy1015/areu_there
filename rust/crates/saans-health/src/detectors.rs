//! Advisory health detectors over one space's fused 2 Hz `SpaceSample`
//! stream (SAANS_PLAN.md §2.1, ADR-005).
//!
//! `SpaceMonitor::update` is fed one sample at a time and returns the
//! events that started, ended or escalated on that sample. Detectors:
//!
//! | Detector | Reads | Gated by |
//! |---|---|---|
//! | Occupancy sessions | `occupied` | — |
//! | Stillness, motion bouts | `motion_index` | — |
//! | Breathing pause | `breathing.band_rms`, `confidence`, `noise_rms` | `vitals` |
//! | Rate high / low | `breathing.bpm`, `confidence` | `vitals` |
//! | Lone-worker inactivity | stillness (+ breathing when `vitals`) | `inactivity_alert_min > 0` |
//! | Possible collapse | `motion_index` burst, then stillness | `collapse_detection` |
//! | Rest period | stillness (+ breathing when `vitals`) | — |
//! | Trend deviation | daily resting breathing rate | `vitals` |
//!
//! Abstention is the default whenever evidence is thin: a pause is never
//! called on breathing that was never seen or never stood clear of the
//! link's empty-room noise, rate alarms need most of their window covered
//! by confident estimates, and blindness (`occupied == None`) is "unknown",
//! never "empty".
//!
//! ## Known limitations (measure them in S3, do not hide them)
//!
//! - **The presence detector cannot see a still person who is not
//!   breathing.** Its energy band (0.1–2 Hz) is dominated by breathing for a
//!   seated person, so about 8 s (energy window) + 10 s (hold) into a real
//!   apnea the space can read as empty. A breathing pause therefore ends
//!   when presence is lost; because a person cannot leave without gross
//!   motion, a pause that ends in a vacancy *without* motion is reported
//!   with severity `Check` and says so. Possible-collapse and inactivity
//!   need presence held and are blind to the same case.
//! - **The breathing estimate can lose validity during a pause.** Fusion
//!   sets `breathing = None` when no link has a valid estimate (45 s
//!   window), and also switches the breathing link by confidence. Without a
//!   band level from the *same* link the pause detector abstains: it does
//!   not start a pause, and it ends a confirmed one at the last sample it
//!   could see, reporting the duration as a lower bound.
//! - **Stillness is "no gross motion", not "unconscious".** Reading,
//!   sleeping and praying are still too; that is why inactivity and
//!   collapse are advisory and collapse detection is off by default.

use crate::types::{Breathing, EventKind, HealthEvent, LinkId, Severity, SpaceConfig, SpaceSample, SpaceState};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap, VecDeque};

// ---------------------------------------------------------------- timing

/// Nominal fused sample interval: fusion runs at the 2 Hz frame rate.
pub const SAMPLE_INTERVAL_S: f64 = 0.5;
/// Longest step credited to time accumulators (occupied seconds, motion
/// time). Larger gaps are missing data, not observed time.
pub const MAX_STEP_S: f64 = 2.0;
/// No samples for this long means anything could have happened unseen
/// (more than enough time to stand up and walk away), so stillness,
/// collapse candidates, rate windows and pauses are reset.
pub const SAMPLE_GAP_RESET_S: f64 = 10.0;

// ------------------------------------------------------------- occupancy

/// Blindness (`occupied == None`, no usable link) shorter than this does
/// not end an occupancy session, a rest period or a still episode: a link
/// reboot or a few lost frames must not read as the person leaving.
pub const BLIND_END_S: f64 = 60.0;
/// A still episode (and a rest period) survives a vacancy shorter than
/// this. The presence detector's own hold is 10 s; a still breather close
/// to threshold can drop out briefly, and a person cannot actually leave
/// without gross motion, which ends stillness anyway.
pub const VACANCY_GRACE_S: f64 = 30.0;
/// Confidence attached to an occupancy session closed by blindness: we
/// did not see the person leave.
pub const BLIND_END_CONFIDENCE: f64 = 0.3;

// ---------------------------------------------------------------- motion

/// Still while `motion_index` (1–4 Hz band RMS through two cascaded
/// biquads / empty-room presence threshold) stays below this, unless
/// `MOVE_CONFIRM_SAMPLES` say otherwise. An empty room sits around 0.75
/// and a still breather around 0.8, but the heartbeat (~1.1 Hz) lives in
/// this band and a still person's index wanders up to ~2.2. Chosen on the
/// development seeds of `saans eval` (seeds 0-1) as the setting that
/// keeps false stillness resets near 0.15/hour while still catching ~82%
/// of fidget bursts; 1.5 gave 60 resets/hour and broke every 30-minute
/// clock. Reported numbers come from held-out seeds.
pub const STILL_INDEX: f64 = 2.5;
/// A sample above this index is part of a motion bout (limbs, turning
/// over, walking): twice the stillness limit so noise near `STILL_INDEX`
/// does not count as restlessness.
pub const BOUT_INDEX: f64 = 3.0;
/// Consecutive samples (at 2 Hz) at or above `STILL_INDEX` before a person
/// counts as moving, unless the index reaches `BOUT_INDEX`: 1.5 s, about
/// the shortest deliberate limb movement. Chosen with `STILL_INDEX` above.
pub const MOVE_CONFIRM_SAMPLES: u32 = 3;
/// Bout samples closer than this belong to the same bout, so one
/// fidget that dips below `BOUT_INDEX` mid-way is not counted twice.
pub const BOUT_GAP_S: f64 = 5.0;
/// Restlessness is reported as bouts per rolling hour.
pub const BOUT_RATE_WINDOW_S: f64 = 3600.0;

// ------------------------------------------------------------- breathing

/// Breathing estimates at or above this confidence count as "breathing
/// detected" (pause baselines, inactivity, rest qualification, the 5-min
/// dashboard figures). Matches the reference pipeline's DEGRADED floor.
pub const CONF_DETECTED: f64 = 0.4;
/// Rate alarms, rest statistics and trends use only estimates at or above
/// this confidence (the reference pipeline's VALID floor): these are
/// numbers a person might act on.
pub const CONF_RATE: f64 = 0.7;
/// "Breathing detected" means a confident estimate within this lookback.
pub const BREATHING_RECENT_S: f64 = 60.0;

// ----------------------------------------------------------------- pause

/// Pause baseline: median band RMS over this many seconds before onset.
/// Long enough to span many breaths, short enough to follow posture.
pub const PAUSE_BASELINE_S: f64 = 60.0;
/// Minimum confident, still samples in the baseline window (10 s at
/// 2 Hz, two or more breaths). Fewer means we never really saw breathing,
/// and a pause cannot be called on breathing we never saw.
pub const PAUSE_BASELINE_MIN_SAMPLES: usize = 20;
/// Baseline samples are taken only this long after the last gross
/// motion: the 0.1–0.5 Hz filter rings after movement and would inflate
/// the baseline, which would make ordinary breathing look like a pause.
pub const PAUSE_BASELINE_SETTLE_S: f64 = 4.0;
/// Onset: band RMS below this fraction of the baseline. Slow breathing
/// (6 BPM) seen through the 2 s RMS window dips to ~0.5 of its median
/// between breaths, so 0.3 stays clear of normal breathing.
pub const PAUSE_ONSET_FRACTION: f64 = 0.3;
/// End: band RMS back above this fraction. Hysteresis against the onset
/// fraction so a pause does not flap on one shallow breath.
pub const PAUSE_END_FRACTION: f64 = 0.5;
/// Abstain when the baseline is below this multiple of the link's
/// empty-room band RMS: breathing never stood clear of the noise, so a
/// "drop" would only be noise.
pub const PAUSE_MIN_SNR: f64 = 2.0;
/// A pause still going after this long is escalated once (`ongoing`,
/// severity `Check`) instead of waiting for its end.
pub const PAUSE_ONGOING_ALERT_S: f64 = 20.0;
/// Pause confidence when the link's noise level is unknown.
pub const PAUSE_UNKNOWN_NOISE_CONFIDENCE: f64 = 0.5;
/// A pause that ended with gross motion may have been a posture change
/// that took the chest out of the link's view, not a pause.
pub const PAUSE_MOTION_END_CONF_SCALE: f64 = 0.5;

// ------------------------------------------------------------------ rate

/// Rate alarms use the median over this window: the plan's "sustained
/// (>= 2 min)" so a brief spike (a sigh, talking) never alarms.
pub const RATE_WINDOW_S: f64 = 120.0;
/// Fraction of the window's samples that must carry a confident, still,
/// present estimate; otherwise the rate is unknown and the check abstains.
pub const RATE_MIN_COVERAGE: f64 = 0.7;
/// A rate episode re-arms after this long back in range (or after nobody
/// has been seen for this long), so one episode gives one alert.
pub const RATE_REARM_S: f64 = 300.0;

// -------------------------------------------------------------- collapse

/// A sample at or above this motion index is a burst: far above walking
/// in a typical room. Starting value; the S3 evaluation must measure it
/// against `Fall` and `SitDown` timelines.
pub const BURST_INDEX: f64 = 8.0;
/// Stillness must begin within this long after the last burst sample: a
/// fall ends abruptly, whereas continued movement means someone got up.
pub const COLLAPSE_SETTLE_S: f64 = 5.0;
/// Stillness after the burst must last this long (plan §2.1).
pub const COLLAPSE_STILL_S: f64 = 30.0;
/// Blindness tolerated inside the post-burst stillness window.
pub const COLLAPSE_BLIND_TOLERANCE_S: f64 = 5.0;
/// Fixed, deliberately modest confidence: WiFi cannot tell a fall from
/// sitting down heavily, so this can never be a confident call.
pub const COLLAPSE_CONFIDENCE: f64 = 0.5;

// ------------------------------------------------------------------ rest

/// A rest period starts after this long still and present.
pub const REST_MIN_S: f64 = 1200.0;
/// With vitals on, a rest period also needs breathing detected in at
/// least this fraction of the last `REST_MIN_S` of samples, so a rest is
/// "still with steady breathing", not just an unseen chair.
pub const REST_MIN_BREATHING_FRACTION: f64 = 0.6;
/// A rest period ends after this much motion-bout time within
/// `REST_END_WINDOW_S`: turning over is part of rest, getting up is not.
pub const REST_END_ACTIVE_S: f64 = 120.0;
pub const REST_END_WINDOW_S: f64 = 300.0;
/// The rest period is dated to end where the activity that ended it
/// began: bout samples less than this apart belong to one stretch, so an
/// earlier turn-over in the window does not pull the end time back.
pub const REST_END_QUIET_S: f64 = 60.0;

// ----------------------------------------------------------------- state

/// `SpaceState` breathing statistics cover this window (plan: BRV over
/// 5 min).
pub const STATE_WINDOW_S: f64 = 300.0;
/// Fewer confident estimates than this (10 s) and the statistic is
/// unknown rather than a number from a handful of samples.
pub const STATE_MIN_SAMPLES: usize = 20;

// ----------------------------------------------------------------- trend

pub const DAY_S: f64 = 86_400.0;
/// Resting-rate samples are taken only after this long still: breathing
/// stays elevated after activity and the 45 s estimate still contains it.
pub const RESTING_SETTLE_S: f64 = 60.0;
/// A day needs this many resting samples (5 min) for a resting rate; less
/// is one sitting, not a day.
pub const TREND_MIN_RESTING_SAMPLES: u32 = 600;
/// Previous days with a value needed before a trend can be called.
pub const TREND_MIN_DAYS: usize = 3;
/// The baseline uses at most this many most recent days.
pub const TREND_BASELINE_DAYS: usize = 28;
/// Deviation threshold: `TREND_MAD_K` robust standard deviations...
pub const TREND_MAD_K: f64 = 3.0;
/// ...where one robust SD is 1.4826 × MAD (normal consistency constant).
pub const MAD_TO_SIGMA: f64 = 1.4826;
/// ...but never less than this: day-to-day resting rates of a healthy
/// adult wander by about a breath per minute, and a perfectly steady
/// baseline (MAD 0) must not turn 0.1 BPM into an alert.
pub const TREND_MIN_DEVIATION_BPM: f64 = 2.0;
/// Daily summaries kept for export (aggregates, within the plan's §8
/// retention rules).
pub const HISTORY_MAX_DAYS: usize = 90;

const HIST_BIN_BPM: f64 = 0.1;
const HIST_MAX_BPM: f64 = 80.0;

/// One day's aggregates for one space, persisted across restarts.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DailySummary {
    /// `floor(t_s / 86400)`.
    pub day: i64,
    /// Median of confident (>= `CONF_RATE`) breathing rates while still
    /// and present; None below `TREND_MIN_RESTING_SAMPLES` or vitals off.
    pub resting_bpm_median: Option<f64>,
    pub resting_samples: u32,
    pub occupied_s: f64,
    pub motion_bouts: u32,
    pub pauses: u32,
}

// ------------------------------------------------------------- internals

/// Fixed-bin histogram of breathing rates (0.1 BPM bins): exact enough
/// for medians and IQRs, constant memory for an 8-hour rest or a day.
#[derive(Debug, Clone, Default)]
struct Hist {
    bins: Vec<u32>,
    n: u64,
    sum: f64,
}

impl Hist {
    fn add_n(&mut self, v: f64, k: u32) {
        if k == 0 || !v.is_finite() {
            return;
        }
        if self.bins.is_empty() {
            self.bins = vec![0; (HIST_MAX_BPM / HIST_BIN_BPM).round() as usize];
        }
        let i = ((v / HIST_BIN_BPM).floor().max(0.0) as usize).min(self.bins.len() - 1);
        self.bins[i] += k;
        self.n += k as u64;
        self.sum += v * k as f64;
    }

    fn add(&mut self, v: f64) {
        self.add_n(v, 1);
    }

    fn mean(&self) -> Option<f64> {
        (self.n > 0).then(|| self.sum / self.n as f64)
    }

    fn quantile(&self, q: f64) -> Option<f64> {
        if self.n == 0 {
            return None;
        }
        let target = q.clamp(0.0, 1.0) * self.n as f64;
        let mut cum = 0.0;
        for (i, &c) in self.bins.iter().enumerate() {
            if c == 0 {
                continue;
            }
            let next = cum + c as f64;
            if next >= target {
                let frac = ((target - cum) / c as f64).clamp(0.0, 1.0);
                return Some((i as f64 + frac) * HIST_BIN_BPM);
            }
            cum = next;
        }
        Some(self.bins.len() as f64 * HIST_BIN_BPM)
    }

    fn iqr(&self) -> Option<f64> {
        Some(self.quantile(0.75)? - self.quantile(0.25)?)
    }
}

fn sorted(mut v: Vec<f64>) -> Vec<f64> {
    v.sort_by(|a, b| a.total_cmp(b));
    v
}

/// Linear-interpolated quantile of a sorted slice (numpy's default).
fn quantile(sorted: &[f64], q: f64) -> Option<f64> {
    if sorted.is_empty() {
        return None;
    }
    let pos = q.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let (lo, hi) = (pos.floor() as usize, pos.ceil() as usize);
    Some(sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo as f64))
}

fn median(v: Vec<f64>) -> Option<f64> {
    quantile(&sorted(v), 0.5)
}

fn event(
    space: &str,
    t_s: f64,
    t_start_s: Option<f64>,
    severity: Severity,
    confidence: f64,
    evidence: String,
    kind: EventKind,
) -> HealthEvent {
    HealthEvent { space_id: space.to_string(), t_s, t_start_s, severity, confidence: confidence.clamp(0.0, 1.0), evidence, kind }
}

/// Heuristic trust in a presence transition from how many links could
/// see the space (ADR-006: one link can sit in a blind spot).
fn link_confidence(links_usable: u8) -> f64 {
    match links_usable {
        0 => 0.0,
        1 => 0.6,
        2 => 0.8,
        _ => 0.9,
    }
}

fn breathing_phrase(vitals: bool, last: Option<&(f64, Breathing)>, t: f64) -> String {
    if !vitals {
        return "breathing is not monitored in this space (vitals off)".into();
    }
    match last {
        Some((bt, b)) if t - bt <= BREATHING_RECENT_S => {
            format!("breathing detected at {:.1} BPM (confidence {:.2}) in the last minute", b.bpm, b.confidence)
        }
        Some((bt, b)) => format!(
            "no confident breathing detected in the last minute (last seen {:.1} min ago at {:.1} BPM)",
            (t - bt) / 60.0,
            b.bpm
        ),
        None => "no confident breathing detected in the last minute".into(),
    }
}

/// Per-sample facts shared by the detectors.
#[derive(Debug, Clone, Copy)]
struct Ctx {
    t: f64,
    dt: f64,
    /// `occupied == Some(true)`.
    present: bool,
    /// Visible (some usable link) and `motion_index >= STILL_INDEX` for
    /// `MOVE_CONFIRM_SAMPLES` in a row, or `>= BOUT_INDEX` at once.
    moving: bool,
    /// Breathing, only when vitals are on and the space is occupied.
    breathing: Option<Breathing>,
}

#[derive(Debug, Clone)]
struct StillEpisode {
    since: f64,
    samples: u64,
    present_samples: u64,
    /// (t, breathing detected) over the last `REST_MIN_S`.
    breath_window: VecDeque<(f64, bool)>,
    breath_count: usize,
    /// Confident rates while still, to seed a rest period's statistics.
    bpm: Hist,
    pauses: u32,
    inactivity_alerted: bool,
    inactivity_escalated: bool,
}

impl StillEpisode {
    fn new(since: f64) -> Self {
        Self {
            since,
            samples: 0,
            present_samples: 0,
            breath_window: VecDeque::new(),
            breath_count: 0,
            bpm: Hist::default(),
            pauses: 0,
            inactivity_alerted: false,
            inactivity_escalated: false,
        }
    }

    fn present_fraction(&self) -> f64 {
        if self.samples == 0 {
            0.0
        } else {
            self.present_samples as f64 / self.samples as f64
        }
    }

    fn breath_fraction(&self) -> f64 {
        if self.breath_window.is_empty() {
            0.0
        } else {
            self.breath_count as f64 / self.breath_window.len() as f64
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum PauseEnd {
    Resumed,
    Motion,
    Vacancy,
    SignalLost,
}

#[derive(Debug, Clone)]
struct Pause {
    link: LinkId,
    t_start: f64,
    /// Last sample at which the pause was still observed.
    t_last: f64,
    baseline: f64,
    noise: Option<f64>,
    min_band: f64,
    last_band: f64,
    confirmed: bool,
    ongoing_sent: bool,
}

impl Pause {
    fn confidence(&self) -> f64 {
        match self.noise {
            Some(n) if self.baseline > 0.0 => ((self.baseline - n) / self.baseline).clamp(0.0, 1.0),
            _ => PAUSE_UNKNOWN_NOISE_CONFIDENCE,
        }
    }

    fn noise_phrase(&self) -> String {
        match self.noise {
            Some(n) => format!("empty-room noise {n:.3}"),
            None => "empty-room noise unknown".into(),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RateDir {
    High,
    Low,
}

#[derive(Debug, Clone)]
struct RateCheck {
    /// (t, Some((bpm, confidence)) when the sample is a usable estimate).
    window: VecDeque<(f64, Option<(f64, f64)>)>,
    armed_high: bool,
    armed_low: bool,
    in_range_since: Option<f64>,
    active: Option<RateDir>,
}

impl Default for RateCheck {
    fn default() -> Self {
        Self { window: VecDeque::new(), armed_high: true, armed_low: true, in_range_since: None, active: None }
    }
}

#[derive(Debug, Clone)]
struct Burst {
    peak: f64,
    last_t: f64,
    still_start: Option<f64>,
    blind_since: Option<f64>,
    fired: bool,
}

#[derive(Debug, Clone)]
struct Rest {
    t_start: f64,
    bouts: u32,
    pauses: u32,
    bpm: Hist,
}

#[derive(Debug, Clone)]
struct DayAccum {
    day: i64,
    resting: Hist,
    resting_samples: u32,
    occupied_s: f64,
    motion_bouts: u32,
    pauses: u32,
}

impl DayAccum {
    fn new(day: i64) -> Self {
        Self { day, resting: Hist::default(), resting_samples: 0, occupied_s: 0.0, motion_bouts: 0, pauses: 0 }
    }

    /// Merge a persisted summary of this same day (gateway restarted
    /// mid-day). The median is re-entered with its sample weight: an
    /// approximation of the lost histogram, exact when nothing new arrives.
    fn seed(&mut self, s: &DailySummary) {
        if let Some(m) = s.resting_bpm_median {
            self.resting.add_n(m, s.resting_samples);
        }
        self.resting_samples += s.resting_samples;
        self.occupied_s += s.occupied_s;
        self.motion_bouts += s.motion_bouts;
        self.pauses += s.pauses;
    }

    fn summary(&self) -> DailySummary {
        DailySummary {
            day: self.day,
            resting_bpm_median: if self.resting_samples >= TREND_MIN_RESTING_SAMPLES { self.resting.quantile(0.5) } else { None },
            resting_samples: self.resting_samples,
            occupied_s: self.occupied_s,
            motion_bouts: self.motion_bouts,
            pauses: self.pauses,
        }
    }
}

// ----------------------------------------------------------- the monitor

/// All advisory health detectors for one space.
#[derive(Debug, Clone)]
pub struct SpaceMonitor {
    config: SpaceConfig,
    last: Option<SpaceSample>,
    // occupancy
    session_start: Option<f64>,
    last_occupied_t: Option<f64>,
    blind_since: Option<f64>,
    vacant_since: Option<f64>,
    // motion
    last_motion_t: Option<f64>,
    bout_last_t: Option<f64>,
    bout_starts: VecDeque<f64>,
    /// (t, dt) of bout samples within `REST_END_WINDOW_S`.
    active_motion: VecDeque<(f64, f64)>,
    still: Option<StillEpisode>,
    /// Consecutive samples at or above `STILL_INDEX`.
    move_run: u32,
    // breathing
    last_breath: Option<(f64, Breathing)>,
    /// (t, bpm) of estimates >= `CONF_DETECTED` within `STATE_WINDOW_S`.
    recent: VecDeque<(f64, f64)>,
    /// Per link (the band RMS scale is per link): (t, band_rms) of
    /// confident, still samples within `PAUSE_BASELINE_S`.
    baselines: HashMap<LinkId, VecDeque<(f64, f64)>>,
    pause: Option<Pause>,
    rate: RateCheck,
    burst: Option<Burst>,
    rest: Option<Rest>,
    // trend
    day: Option<DayAccum>,
    history: BTreeMap<i64, DailySummary>,
}

impl SpaceMonitor {
    pub fn new(config: SpaceConfig) -> Self {
        Self {
            config,
            last: None,
            session_start: None,
            last_occupied_t: None,
            blind_since: None,
            vacant_since: None,
            last_motion_t: None,
            bout_last_t: None,
            bout_starts: VecDeque::new(),
            active_motion: VecDeque::new(),
            still: None,
            move_run: 0,
            last_breath: None,
            recent: VecDeque::new(),
            baselines: HashMap::new(),
            pause: None,
            rate: RateCheck::default(),
            burst: None,
            rest: None,
            day: None,
            history: BTreeMap::new(),
        }
    }

    pub fn config(&self) -> &SpaceConfig {
        &self.config
    }

    /// Feed one fused 2 Hz sample; returns events that started, ended or
    /// escalated now. `t_s` must not go backwards; a sample that does is
    /// ignored.
    pub fn update(&mut self, s: &SpaceSample) -> Vec<HealthEvent> {
        let mut ev = Vec::new();
        let t = s.t_s;
        if !t.is_finite() {
            return ev;
        }
        let prev = self.last.map(|p| p.t_s);
        if prev.is_some_and(|p| t < p) {
            return ev;
        }
        let gap = prev.map_or(0.0, |p| t - p);
        let dt = if prev.is_some() { gap.min(MAX_STEP_S) } else { SAMPLE_INTERVAL_S };
        if gap > SAMPLE_GAP_RESET_S {
            self.discontinuity(t, gap, &mut ev);
        }
        self.roll_day(t, &mut ev);

        let present = s.occupied == Some(true);
        let visible = s.occupied.is_some();
        self.move_run = if visible && s.motion_index >= STILL_INDEX { self.move_run + 1 } else { 0 };
        let c = Ctx {
            t,
            dt,
            present,
            moving: visible && (s.motion_index >= BOUT_INDEX || self.move_run >= MOVE_CONFIRM_SAMPLES),
            breathing: if self.config.vitals && present { s.breathing } else { None },
        };

        self.occupancy_step(s, &c, &mut ev);
        if self.config.vitals {
            // Before the baselines take this sample: B is the level *before* onset.
            self.pause_step(s, &c, &mut ev);
        }
        self.motion_step(s, &c);
        self.breathing_step(&c);
        if self.config.vitals {
            self.rate_step(&c, &mut ev);
        }
        self.inactivity_step(&c, &mut ev);
        if self.config.collapse_detection {
            self.collapse_step(s, &c, &mut ev);
        }
        self.rest_step(&c, &mut ev);
        self.day_step(&c);
        self.last = Some(*s);
        ev
    }

    /// Live snapshot for the API and dashboard.
    pub fn state(&self) -> SpaceState {
        let vitals = self.config.vitals;
        let Some(s) = self.last else {
            return SpaceState { space_id: self.config.id.clone(), vitals, ..Default::default() };
        };
        let t = s.t_s;
        let (bpm_median_5min, bpm_iqr_5min) = if vitals {
            let v = sorted(self.recent.iter().filter(|x| x.0 > t - STATE_WINDOW_S).map(|x| x.1).collect());
            if v.len() >= STATE_MIN_SAMPLES {
                (quantile(&v, 0.5), Some(quantile(&v, 0.75).unwrap_or(0.0) - quantile(&v, 0.25).unwrap_or(0.0)))
            } else {
                (None, None)
            }
        } else {
            (None, None)
        };
        let in_pause = self.pause.as_ref().is_some_and(|p| p.confirmed);
        let mut active_alerts = Vec::new();
        if s.occupied.is_none() {
            active_alerts.push("blind".to_string());
        }
        if in_pause {
            active_alerts.push("breathing_pause".to_string());
        }
        if let Some(ep) = &self.still {
            if ep.inactivity_escalated {
                active_alerts.push("inactivity_no_breathing".to_string());
            } else if ep.inactivity_alerted {
                active_alerts.push("inactivity".to_string());
            }
        }
        if self.burst.as_ref().is_some_and(|b| b.fired) {
            active_alerts.push("possible_collapse".to_string());
        }
        match self.rate.active {
            Some(RateDir::High) => active_alerts.push("rate_high".to_string()),
            Some(RateDir::Low) => active_alerts.push("rate_low".to_string()),
            None => {}
        }
        SpaceState {
            space_id: self.config.id.clone(),
            t_s: t,
            occupied: s.occupied,
            occupied_since_s: self.session_start,
            still_for_s: self.still.as_ref().map_or(0.0, |e| (t - e.since).max(0.0)),
            motion_index: s.motion_index,
            breathing: if vitals { s.breathing } else { None },
            bpm_median_5min,
            bpm_iqr_5min,
            motion_bouts_last_hour: self.bout_starts.iter().filter(|&&b| b > t - BOUT_RATE_WINDOW_S).count() as u32,
            resting: self.rest.is_some(),
            in_breathing_pause: in_pause,
            links_usable: s.links_usable,
            links_total: s.links_total,
            vitals,
            active_alerts,
        }
    }

    /// Completed days plus the day in progress, oldest first.
    pub fn export_history(&self) -> Vec<DailySummary> {
        let mut out: Vec<DailySummary> = self.history.values().cloned().collect();
        if let Some(d) = &self.day {
            out.retain(|s| s.day != d.day);
            out.push(d.summary());
        }
        out.sort_by_key(|s| s.day);
        out
    }

    /// Restore persisted summaries. Call once, at startup. An entry for the
    /// day in progress (a mid-day restart) is merged into today's
    /// accumulators; later entries for the same day replace earlier ones.
    pub fn import_history(&mut self, h: Vec<DailySummary>) {
        for s in h {
            match &mut self.day {
                Some(d) if d.day == s.day => d.seed(&s),
                _ => {
                    self.history.insert(s.day, s);
                }
            }
        }
        self.prune_history();
    }

    // ------------------------------------------------------------ steps

    fn discontinuity(&mut self, t: f64, gap: f64, ev: &mut Vec<HealthEvent>) {
        if let Some(p) = self.pause.take() {
            self.finish_pause(p, PauseEnd::SignalLost, t, ev);
        }
        self.still = None;
        self.burst = None;
        self.rate.window.clear();
        self.rate.in_range_since = None;
        if gap >= BLIND_END_S {
            let why = format!("no samples arrived for {gap:.0} s, so the session was closed at the last time someone was seen");
            if self.session_start.is_some() {
                let end = self.last_occupied_t.unwrap_or(t - gap);
                self.finish_session(end, t, &why, BLIND_END_CONFIDENCE, ev);
            }
            if self.rest.is_some() {
                self.finish_rest(t - gap, t, &format!("no samples arrived for {gap:.0} s"), ev);
            }
            self.blind_since = None;
            self.vacant_since = None;
        }
    }

    fn roll_day(&mut self, t: f64, ev: &mut Vec<HealthEvent>) {
        let d = (t / DAY_S).floor() as i64;
        if self.day.as_ref().is_some_and(|acc| acc.day >= d) {
            return;
        }
        if let Some(acc) = self.day.take() {
            let summary = acc.summary();
            self.history.insert(summary.day, summary.clone());
            self.trend_check(&summary, t, ev);
            self.prune_history();
        }
        let mut acc = DayAccum::new(d);
        if let Some(prev) = self.history.remove(&d) {
            acc.seed(&prev);
        }
        self.day = Some(acc);
    }

    fn prune_history(&mut self) {
        while self.history.len() > HISTORY_MAX_DAYS {
            self.history.pop_first();
        }
    }

    fn trend_check(&self, today: &DailySummary, t: f64, ev: &mut Vec<HealthEvent>) {
        if !self.config.vitals {
            return;
        }
        let Some(value) = today.resting_bpm_median else { return };
        let prev: Vec<f64> = self
            .history
            .range(..today.day)
            .rev()
            .filter_map(|(_, d)| d.resting_bpm_median)
            .take(TREND_BASELINE_DAYS)
            .collect();
        let n = prev.len();
        if n < TREND_MIN_DAYS {
            return;
        }
        let baseline = median(prev.clone()).unwrap_or(value);
        let mad = median(prev.iter().map(|v| (v - baseline).abs()).collect()).unwrap_or(0.0);
        let threshold = (TREND_MAD_K * MAD_TO_SIGMA * mad).max(TREND_MIN_DEVIATION_BPM);
        let dev = value - baseline;
        if dev.abs() <= threshold {
            return;
        }
        let evidence = format!(
            "resting breathing rate on day {} was {:.1} BPM (median of {} confident still samples), {:.1} BPM {} \
             this space's baseline of {:.1} BPM (median of the previous {} days, MAD {:.2}), beyond the {:.1} BPM \
             threshold; a trend to note, not a diagnosis",
            today.day,
            value,
            today.resting_samples,
            dev.abs(),
            if dev > 0.0 { "above" } else { "below" },
            baseline,
            n,
            mad,
            threshold
        );
        ev.push(event(
            &self.config.id,
            t,
            Some(today.day as f64 * DAY_S),
            Severity::Info,
            n as f64 / (n as f64 + 2.0),
            evidence,
            EventKind::TrendDeviation { metric: "resting_bpm".into(), value, baseline, mad },
        ));
    }

    fn finish_session(&mut self, t_end: f64, t_detect: f64, why: &str, confidence: f64, ev: &mut Vec<HealthEvent>) {
        let Some(start) = self.session_start.take() else { return };
        let dwell = (t_end - start).max(0.0);
        let evidence = format!("the space was occupied for {:.1} min; {why}", dwell / 60.0);
        ev.push(event(&self.config.id, t_detect, Some(start), Severity::Info, confidence, evidence, EventKind::OccupancyEnded {
            dwell_s: dwell,
        }));
    }

    fn occupancy_step(&mut self, s: &SpaceSample, c: &Ctx, ev: &mut Vec<HealthEvent>) {
        match s.occupied {
            Some(true) => {
                self.blind_since = None;
                self.vacant_since = None;
                if self.session_start.is_none() {
                    self.session_start = Some(c.t);
                    let evidence = format!(
                        "the fused presence vote says occupied (presence index {:.1}, {} of {} links usable)",
                        s.presence_index, s.links_usable, s.links_total
                    );
                    ev.push(event(
                        &self.config.id,
                        c.t,
                        None,
                        Severity::Info,
                        link_confidence(s.links_usable),
                        evidence,
                        EventKind::OccupancyStarted,
                    ));
                }
                self.last_occupied_t = Some(c.t);
            }
            Some(false) => {
                let was_blind = self.blind_since.take().is_some();
                self.vacant_since.get_or_insert(c.t);
                if self.session_start.is_some() {
                    let end = if was_blind { self.last_occupied_t.unwrap_or(c.t) } else { c.t };
                    let why = format!(
                        "the fused presence vote now says empty ({} of {} links usable)",
                        s.links_usable, s.links_total
                    );
                    self.finish_session(end, c.t, &why, link_confidence(s.links_usable), ev);
                }
            }
            None => {
                let since = *self.blind_since.get_or_insert(c.t);
                if self.session_start.is_some() && c.t - since >= BLIND_END_S {
                    let end = self.last_occupied_t.unwrap_or(since);
                    let why = format!(
                        "no link could see the space for {:.0} s, so the session was closed at the last time \
                         someone was seen (they may still be there)",
                        c.t - since
                    );
                    self.finish_session(end, c.t, &why, BLIND_END_CONFIDENCE, ev);
                }
            }
        }
    }

    fn baseline(&self, link: LinkId, t: f64) -> Option<f64> {
        let q = self.baselines.get(&link)?;
        let v: Vec<f64> = q.iter().filter(|x| x.0 > t - PAUSE_BASELINE_S).map(|x| x.1).collect();
        if v.len() < PAUSE_BASELINE_MIN_SAMPLES {
            return None;
        }
        median(v)
    }

    fn pause_step(&mut self, s: &SpaceSample, c: &Ctx, ev: &mut Vec<HealthEvent>) {
        let t = c.t;
        if let Some(mut p) = self.pause.take() {
            let end = if c.moving {
                Some(PauseEnd::Motion)
            } else if s.occupied == Some(false) {
                Some(PauseEnd::Vacancy)
            } else if s.occupied.is_none() {
                Some(PauseEnd::SignalLost)
            } else {
                match c.breathing {
                    Some(b) if b.link == p.link => {
                        let resumed = if p.confirmed {
                            b.band_rms > PAUSE_END_FRACTION * p.baseline
                        } else {
                            // Onset needs the band below 0.3 B *continuously*.
                            b.band_rms >= PAUSE_ONSET_FRACTION * p.baseline
                        };
                        if resumed {
                            Some(PauseEnd::Resumed)
                        } else {
                            p.t_last = t;
                            p.last_band = b.band_rms;
                            p.min_band = p.min_band.min(b.band_rms);
                            None
                        }
                    }
                    // No band level from the pause's own link: abstain.
                    _ => Some(PauseEnd::SignalLost),
                }
            };
            match end {
                None => {
                    if !p.confirmed && t - p.t_start >= self.config.breathing_pause_min_s {
                        p.confirmed = true;
                    }
                    if p.confirmed && !p.ongoing_sent && t - p.t_start > PAUSE_ONGOING_ALERT_S {
                        p.ongoing_sent = true;
                        let evidence = format!(
                            "breathing-band energy has stayed below {:.0}% of its recent baseline for {:.1} s while \
                             the space is occupied and still (now {:.3}, lowest {:.3} vs baseline {:.3}, {}); \
                             ongoing, check on the space",
                            PAUSE_ONSET_FRACTION * 100.0,
                            t - p.t_start,
                            p.last_band,
                            p.min_band,
                            p.baseline,
                            p.noise_phrase()
                        );
                        ev.push(event(
                            &self.config.id,
                            t,
                            Some(p.t_start),
                            Severity::Check,
                            p.confidence(),
                            evidence,
                            EventKind::BreathingPause { duration_s: t - p.t_start, ongoing: true },
                        ));
                    }
                    self.pause = Some(p);
                }
                Some(reason) => self.finish_pause(p, reason, t, ev),
            }
            return;
        }

        // Idle: look for an onset.
        if !c.present || c.moving {
            return;
        }
        let Some(b) = c.breathing else { return };
        // No confident breathing seen in the last minute => no baseline => abstain.
        let Some(base) = self.baseline(b.link, t) else { return };
        if b.band_rms >= PAUSE_ONSET_FRACTION * base {
            return;
        }
        if b.noise_rms.is_some_and(|n| base < PAUSE_MIN_SNR * n) {
            // Breathing never stood clear of this link's noise: abstain.
            return;
        }
        let confirmed = self.config.breathing_pause_min_s <= 0.0;
        self.pause = Some(Pause {
            link: b.link,
            t_start: t,
            t_last: t,
            baseline: base,
            noise: b.noise_rms,
            min_band: b.band_rms,
            last_band: b.band_rms,
            confirmed,
            ongoing_sent: false,
        });
    }

    fn finish_pause(&mut self, p: Pause, reason: PauseEnd, t_detect: f64, ev: &mut Vec<HealthEvent>) {
        if !p.confirmed {
            return; // shorter than the minimum, or interrupted: not a pause
        }
        let t_end = match reason {
            PauseEnd::Resumed | PauseEnd::Motion => t_detect,
            PauseEnd::Vacancy | PauseEnd::SignalLost => p.t_last,
        };
        let duration = (t_end - p.t_start).max(0.0);
        let mut confidence = p.confidence();
        let (why, severity) = match reason {
            PauseEnd::Resumed => (
                format!("breathing-band energy then came back above {:.0}% of the baseline", PAUSE_END_FRACTION * 100.0),
                Severity::Advisory,
            ),
            PauseEnd::Motion => {
                confidence *= PAUSE_MOTION_END_CONF_SCALE;
                (
                    "it ended with gross motion, which can also mean a change of posture took the chest out of this \
                     link's view"
                        .to_string(),
                    Severity::Advisory,
                )
            }
            PauseEnd::Vacancy => (
                "presence was then lost without any gross motion; a person who stops breathing can look like an \
                 empty room to WiFi, so check on the space"
                    .to_string(),
                Severity::Check,
            ),
            PauseEnd::SignalLost => (
                "the breathing signal then became unavailable, so the duration is a lower bound".to_string(),
                if duration > PAUSE_ONGOING_ALERT_S { Severity::Check } else { Severity::Advisory },
            ),
        };
        if let Some(d) = &mut self.day {
            d.pauses += 1;
        }
        if let Some(r) = &mut self.rest {
            r.pauses += 1;
        } else if let Some(ep) = &mut self.still {
            ep.pauses += 1;
        }
        let evidence = format!(
            "breathing-band energy fell below {:.0}% of its recent baseline for {:.1} s while the space was occupied \
             and still (lowest {:.3} vs baseline {:.3}, {}); {why}",
            PAUSE_ONSET_FRACTION * 100.0,
            duration,
            p.min_band,
            p.baseline,
            p.noise_phrase()
        );
        ev.push(event(
            &self.config.id,
            t_detect,
            Some(p.t_start),
            severity,
            confidence,
            evidence,
            EventKind::BreathingPause { duration_s: duration, ongoing: false },
        ));
    }

    fn motion_step(&mut self, s: &SpaceSample, c: &Ctx) {
        let t = c.t;
        if s.occupied.is_some() && s.motion_index > BOUT_INDEX {
            if !self.bout_last_t.is_some_and(|lt| t - lt < BOUT_GAP_S) {
                self.bout_starts.push_back(t);
                if let Some(d) = &mut self.day {
                    d.motion_bouts += 1;
                }
                if let Some(r) = &mut self.rest {
                    r.bouts += 1;
                }
            }
            self.bout_last_t = Some(t);
            self.active_motion.push_back((t, c.dt));
        }
        while self.bout_starts.front().is_some_and(|&b| b <= t - BOUT_RATE_WINDOW_S) {
            self.bout_starts.pop_front();
        }
        while self.active_motion.front().is_some_and(|x| x.0 <= t - REST_END_WINDOW_S) {
            self.active_motion.pop_front();
        }
        if c.moving {
            self.last_motion_t = Some(t);
        }

        match s.occupied {
            Some(_) if c.moving => self.still = None,
            Some(true) => {
                if self.still.is_none() {
                    self.still = Some(StillEpisode::new(t));
                }
            }
            Some(false) => {
                if self.vacant_since.is_some_and(|v| t - v >= VACANCY_GRACE_S) {
                    self.still = None;
                }
            }
            None => {
                if self.blind_since.is_some_and(|b| t - b >= BLIND_END_S) {
                    self.still = None;
                }
            }
        }
        if let Some(ep) = &mut self.still {
            ep.samples += 1;
            if c.present {
                ep.present_samples += 1;
            }
        }
    }

    fn breathing_step(&mut self, c: &Ctx) {
        let t = c.t;
        while self.recent.front().is_some_and(|x| x.0 <= t - STATE_WINDOW_S) {
            self.recent.pop_front();
        }
        for q in self.baselines.values_mut() {
            while q.front().is_some_and(|x| x.0 <= t - PAUSE_BASELINE_S) {
                q.pop_front();
            }
        }
        self.baselines.retain(|_, q| !q.is_empty());

        let detected = c.breathing.filter(|b| b.confidence >= CONF_DETECTED);
        if let Some(b) = detected {
            self.last_breath = Some((t, b));
            self.recent.push_back((t, b.bpm));
            let settled = !self.last_motion_t.is_some_and(|m| t - m < PAUSE_BASELINE_SETTLE_S);
            if self.pause.is_none() && !c.moving && settled {
                self.baselines.entry(b.link).or_default().push_back((t, b.band_rms));
            }
        }
        let rate_sample = c.breathing.filter(|b| b.confidence >= CONF_RATE && c.present && !c.moving).map(|b| b.bpm);
        if let Some(ep) = &mut self.still {
            ep.breath_window.push_back((t, detected.is_some()));
            if detected.is_some() {
                ep.breath_count += 1;
            }
            while ep.breath_window.front().is_some_and(|x| x.0 <= t - REST_MIN_S) {
                if ep.breath_window.pop_front().is_some_and(|x| x.1) {
                    ep.breath_count -= 1;
                }
            }
            if let Some(bpm) = rate_sample {
                ep.bpm.add(bpm);
            }
        }
        if let (Some(r), Some(bpm)) = (&mut self.rest, rate_sample) {
            r.bpm.add(bpm);
        }
    }

    fn rate_step(&mut self, c: &Ctx, ev: &mut Vec<HealthEvent>) {
        let t = c.t;
        if !self.last_occupied_t.is_some_and(|lo| t - lo < RATE_REARM_S) {
            // Nobody seen for a while: whatever comes next is a new episode.
            self.rate.armed_high = true;
            self.rate.armed_low = true;
            self.rate.active = None;
        }
        let valid = c.breathing.filter(|b| c.present && !c.moving && b.confidence >= CONF_RATE).map(|b| (b.bpm, b.confidence));
        self.rate.window.push_back((t, valid));
        while self.rate.window.front().is_some_and(|x| x.0 <= t - RATE_WINDOW_S) {
            self.rate.window.pop_front();
        }
        let Some(&(t_first, _)) = self.rate.window.front() else { return };
        let span = t - t_first;
        if span < RATE_WINDOW_S - 2.0 * SAMPLE_INTERVAL_S {
            return; // window not full yet
        }
        let n = self.rate.window.len();
        let vals: Vec<(f64, f64)> = self.rate.window.iter().filter_map(|x| x.1).collect();
        if (vals.len() as f64) < RATE_MIN_COVERAGE * n as f64 {
            return; // rate unknown: abstain
        }
        let Some(med) = median(vals.iter().map(|v| v.0).collect()) else { return };
        let mean_conf = vals.iter().map(|v| v.1).sum::<f64>() / vals.len() as f64;
        let dir = if med > self.config.rate_high_bpm {
            RateDir::High
        } else if med < self.config.rate_low_bpm {
            RateDir::Low
        } else {
            self.rate.active = None;
            let since = *self.rate.in_range_since.get_or_insert(t);
            if t - since >= RATE_REARM_S {
                self.rate.armed_high = true;
                self.rate.armed_low = true;
            }
            return;
        };
        self.rate.in_range_since = None;
        self.rate.active = Some(dir);
        let armed = match dir {
            RateDir::High => &mut self.rate.armed_high,
            RateDir::Low => &mut self.rate.armed_low,
        };
        if !*armed {
            return;
        }
        *armed = false;
        let (word, limit) = match dir {
            RateDir::High => ("above", self.config.rate_high_bpm),
            RateDir::Low => ("below", self.config.rate_low_bpm),
        };
        let evidence = format!(
            "median breathing rate {:.1} BPM over the last {:.0} s ({} of {} samples confident >= {:.1}, mean \
             confidence {:.2}) while present and still, {word} the {:.0} BPM advisory limit",
            med,
            span,
            vals.len(),
            n,
            CONF_RATE,
            mean_conf,
            limit
        );
        let kind = match dir {
            RateDir::High => EventKind::RateHigh { median_bpm: med, over_s: span },
            RateDir::Low => EventKind::RateLow { median_bpm: med, over_s: span },
        };
        ev.push(event(&self.config.id, t, Some(t_first), Severity::Advisory, mean_conf, evidence, kind));
    }

    fn inactivity_step(&mut self, c: &Ctx, ev: &mut Vec<HealthEvent>) {
        let limit_s = self.config.inactivity_alert_min * 60.0;
        if limit_s <= 0.0 || !limit_s.is_finite() || !c.present {
            return;
        }
        let vitals = self.config.vitals;
        let t = c.t;
        let detected = vitals && self.last_breath.as_ref().is_some_and(|(bt, _)| t - bt <= BREATHING_RECENT_S);
        let phrase = breathing_phrase(vitals, self.last_breath.as_ref(), t);
        let Some(ep) = &mut self.still else { return };
        let still_s = t - ep.since;
        if still_s < limit_s {
            return;
        }
        let (severity, evidence) = if !ep.inactivity_alerted {
            ep.inactivity_alerted = true;
            ep.inactivity_escalated = vitals && !detected;
            if ep.inactivity_escalated {
                (Severity::Check, format!("no gross motion for {:.1} min while present; {phrase}; check on the space", still_s / 60.0))
            } else {
                (Severity::Advisory, format!("no gross motion for {:.1} min while present; {phrase}", still_s / 60.0))
            }
        } else if vitals && !ep.inactivity_escalated && !detected {
            ep.inactivity_escalated = true;
            (
                Severity::Check,
                format!(
                    "still no gross motion after {:.1} min while present, and breathing is no longer detected: {phrase}; \
                     check on the space",
                    still_s / 60.0
                ),
            )
        } else {
            return;
        };
        let confidence = ep.present_fraction();
        let since = ep.since;
        ev.push(event(&self.config.id, t, Some(since), severity, confidence, evidence, EventKind::Inactivity {
            still_min: still_s / 60.0,
            breathing_detected: detected,
        }));
    }

    fn collapse_step(&mut self, s: &SpaceSample, c: &Ctx, ev: &mut Vec<HealthEvent>) {
        let t = c.t;
        if s.occupied.is_some() && s.motion_index >= BURST_INDEX {
            let peak = match &self.burst {
                Some(b) if b.still_start.is_none() && t - b.last_t <= COLLAPSE_SETTLE_S => b.peak.max(s.motion_index),
                _ => s.motion_index,
            };
            self.burst = Some(Burst { peak, last_t: t, still_start: None, blind_since: None, fired: false });
            return;
        }
        let Some(b) = &mut self.burst else { return };
        match s.occupied {
            None => {
                let since = *b.blind_since.get_or_insert(t);
                if t - since > COLLAPSE_BLIND_TOLERANCE_S {
                    self.burst = None;
                }
                return;
            }
            Some(false) => {
                // Presence not held: most likely they left.
                self.burst = None;
                return;
            }
            Some(true) => b.blind_since = None,
        }
        if c.moving {
            if b.still_start.is_some() || t - b.last_t > COLLAPSE_SETTLE_S {
                self.burst = None;
            }
            return;
        }
        let still_start = match b.still_start {
            Some(ss) => ss,
            None if t - b.last_t <= COLLAPSE_SETTLE_S => *b.still_start.insert(t),
            None => {
                self.burst = None;
                return;
            }
        };
        let still_s = t - still_start;
        if b.fired || still_s < COLLAPSE_STILL_S {
            return;
        }
        b.fired = true;
        let (peak, settle, burst_t) = (b.peak, still_start - b.last_t, b.last_t);
        let phrase = breathing_phrase(self.config.vitals, self.last_breath.as_ref(), t);
        let evidence = format!(
            "a motion burst (index {:.1}, {:.0}x the stillness limit) was followed within {:.1} s by {:.0} s of no \
             gross motion with presence held; {phrase}. Sitting down heavily looks the same to WiFi, so this is \
             advisory: check on the space",
            peak,
            peak / STILL_INDEX,
            settle,
            still_s
        );
        ev.push(event(&self.config.id, t, Some(burst_t), Severity::Check, COLLAPSE_CONFIDENCE, evidence, EventKind::PossibleCollapse {
            burst_index: peak,
            still_s,
        }));
    }

    fn finish_rest(&mut self, t_end: f64, t_detect: f64, why: &str, ev: &mut Vec<HealthEvent>) {
        let Some(r) = self.rest.take() else { return };
        let duration_min = (t_end - r.t_start).max(0.0) / 60.0;
        let enough = self.config.vitals && r.bpm.n as usize >= STATE_MIN_SAMPLES;
        let mean_bpm = if enough { r.bpm.mean() } else { None };
        let bpm_iqr = if enough { r.bpm.iqr() } else { None };
        let breathing = match (mean_bpm, bpm_iqr) {
            (Some(m), Some(i)) => format!("breathing {m:.1} BPM mean (IQR {i:.1})"),
            _ if self.config.vitals => "breathing rate unknown (too few confident estimates)".to_string(),
            _ => "breathing not monitored (vitals off)".to_string(),
        };
        let evidence = format!(
            "rest period of {:.1} min ended because {why}; {} motion bouts, {breathing}, {} breathing pauses. Rest \
             period, not sleep staging",
            duration_min, r.bouts, r.pauses
        );
        ev.push(event(&self.config.id, t_detect, Some(r.t_start), Severity::Info, 0.8, evidence, EventKind::RestEnded {
            duration_min,
            motion_bouts: r.bouts,
            mean_bpm,
            bpm_iqr,
            pauses: r.pauses,
        }));
    }

    /// Start of the current stretch of bout samples (see `REST_END_QUIET_S`).
    fn activity_start(&self, t: f64) -> f64 {
        let mut start = t;
        for &(ts, _) in self.active_motion.iter().rev() {
            if start - ts > REST_END_QUIET_S {
                break;
            }
            start = ts;
        }
        start
    }

    fn rest_step(&mut self, c: &Ctx, ev: &mut Vec<HealthEvent>) {
        let t = c.t;
        if self.rest.is_some() {
            let active: f64 = self.active_motion.iter().map(|x| x.1).sum();
            if active >= REST_END_ACTIVE_S {
                let t_end = self.activity_start(t);
                let why = format!("of sustained activity ({active:.0} s of motion bouts in the last {REST_END_WINDOW_S:.0} s)");
                self.finish_rest(t_end, t, &why, ev);
            } else if let Some(v) = self.vacant_since.filter(|v| t - v >= VACANCY_GRACE_S) {
                self.finish_rest(v, t, &format!("the space has read as empty for {:.0} s", t - v), ev);
            } else if let Some(b) = self.blind_since.filter(|b| t - b >= BLIND_END_S) {
                self.finish_rest(b, t, &format!("no link could see the space for {:.0} s", t - b), ev);
            }
            return;
        }
        if !c.present {
            return;
        }
        let vitals = self.config.vitals;
        let Some(ep) = &self.still else { return };
        let still_s = t - ep.since;
        if still_s < REST_MIN_S {
            return;
        }
        let frac = ep.breath_fraction();
        if vitals && frac < REST_MIN_BREATHING_FRACTION {
            return;
        }
        let (confidence, evidence) = if vitals {
            let med = ep.bpm.quantile(0.5).map_or("unknown".to_string(), |m| format!("{m:.1} BPM"));
            (
                frac,
                format!(
                    "no gross motion for {:.1} min with breathing detected in {:.0}% of samples (median {med}); \
                     rest period, not sleep staging",
                    still_s / 60.0,
                    frac * 100.0
                ),
            )
        } else {
            (
                ep.present_fraction(),
                format!(
                    "no gross motion for {:.1} min while present (breathing not monitored); rest period, not sleep \
                     staging",
                    still_s / 60.0
                ),
            )
        };
        let rest = Rest { t_start: ep.since, bouts: 0, pauses: ep.pauses, bpm: ep.bpm.clone() };
        ev.push(event(&self.config.id, t, Some(rest.t_start), Severity::Info, confidence, evidence, EventKind::RestStarted));
        self.rest = Some(rest);
    }

    fn day_step(&mut self, c: &Ctx) {
        let Some(d) = &mut self.day else { return };
        if c.present {
            d.occupied_s += c.dt;
        }
        let settled = self.still.as_ref().is_some_and(|ep| c.t - ep.since >= RESTING_SETTLE_S);
        if let Some(b) = c.breathing {
            if c.present && !c.moving && settled && b.confidence >= CONF_RATE {
                d.resting.add(b.bpm);
                d.resting_samples += 1;
            }
        }
    }
}

// ----------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    const L: LinkId = LinkId { rx: 1, tx: 9 };
    const NOISE: Option<f64> = Some(0.02);

    fn cfg(vitals: bool) -> SpaceConfig {
        SpaceConfig { id: "room".into(), vitals, ..Default::default() }
    }

    fn br(bpm: f64, conf: f64, band: f64, noise: Option<f64>) -> Option<Breathing> {
        Some(Breathing { link: L, bpm, confidence: conf, band_rms: band, noise_rms: noise })
    }

    fn sample(t: f64, occupied: Option<bool>, motion: f64, breathing: Option<Breathing>) -> SpaceSample {
        SpaceSample {
            t_s: t,
            occupied,
            presence_index: if occupied == Some(true) { 3.0 } else { 0.5 },
            motion_index: motion,
            links_usable: if occupied.is_some() { 2 } else { 0 },
            links_total: 2,
            breathing,
            heart: None,
        }
    }

    /// Present, still, breathing normally.
    fn calm(t: f64) -> SpaceSample {
        sample(t, Some(true), 0.5, br(14.0, 0.8, 0.25, NOISE))
    }

    /// Feed samples at 2 Hz for t in [t0, t1).
    fn run(m: &mut SpaceMonitor, t0: f64, t1: f64, f: impl Fn(f64) -> SpaceSample) -> Vec<HealthEvent> {
        let n = ((t1 - t0) / SAMPLE_INTERVAL_S).round() as usize;
        (0..n).flat_map(|i| m.update(&f(t0 + i as f64 * SAMPLE_INTERVAL_S))).collect()
    }

    fn pauses(ev: &[HealthEvent]) -> Vec<&HealthEvent> {
        ev.iter().filter(|e| matches!(e.kind, EventKind::BreathingPause { .. })).collect()
    }

    fn count(ev: &[HealthEvent], pred: impl Fn(&EventKind) -> bool) -> usize {
        ev.iter().filter(|e| pred(&e.kind)).count()
    }

    #[test]
    fn still_breather_with_a_15s_pause_is_detected() {
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 120.0, calm);
        ev.extend(run(&mut m, 120.0, 135.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.01, NOISE))));
        assert!(m.state().in_breathing_pause, "confirmed after 10 s");
        assert!(m.state().active_alerts.contains(&"breathing_pause".to_string()));
        ev.extend(run(&mut m, 135.0, 200.0, calm));
        let p = pauses(&ev);
        assert_eq!(p.len(), 1, "{p:?}");
        let e = p[0];
        match e.kind {
            EventKind::BreathingPause { duration_s, ongoing } => {
                assert!(!ongoing);
                assert!((duration_s - 15.0).abs() <= 2.0, "duration {duration_s}");
            }
            _ => unreachable!(),
        }
        assert_eq!(e.severity, Severity::Advisory);
        assert!((e.t_start_s.unwrap() - 120.0).abs() <= 2.0);
        assert!((e.confidence - 0.92).abs() < 1e-9, "(B - noise) / B");
        assert!(e.evidence.contains("15.0 s") && e.evidence.contains("0.250"), "{}", e.evidence);
        assert!(!m.state().in_breathing_pause);
    }

    #[test]
    fn long_pause_escalates_once_while_ongoing() {
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 120.0, calm);
        ev.extend(run(&mut m, 120.0, 150.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.01, NOISE))));
        ev.extend(run(&mut m, 150.0, 170.0, calm));
        let p = pauses(&ev);
        assert_eq!(p.len(), 2);
        assert!(matches!(p[0].kind, EventKind::BreathingPause { ongoing: true, .. }));
        assert_eq!(p[0].severity, Severity::Check);
        assert!((p[0].t_s - 120.0 - 20.0).abs() <= 1.0);
        match p[1].kind {
            EventKind::BreathingPause { duration_s, ongoing: false } => assert!((duration_s - 30.0).abs() <= 2.0),
            _ => panic!("{:?}", p[1]),
        }
    }

    #[test]
    fn short_dip_is_not_a_pause() {
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 120.0, calm);
        ev.extend(run(&mut m, 120.0, 128.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.01, NOISE))));
        ev.extend(run(&mut m, 128.0, 200.0, calm));
        assert!(pauses(&ev).is_empty());
    }

    #[test]
    fn pause_without_prior_confident_breathing_abstains() {
        for before in [br(14.0, 0.2, 0.25, NOISE), None] {
            let mut m = SpaceMonitor::new(cfg(true));
            let mut ev = run(&mut m, 0.0, 120.0, |t| sample(t, Some(true), 0.5, before));
            ev.extend(run(&mut m, 120.0, 140.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.01, NOISE))));
            ev.extend(run(&mut m, 140.0, 200.0, calm));
            assert!(pauses(&ev).is_empty(), "{before:?}: {ev:?}");
        }
    }

    #[test]
    fn pause_on_breathing_never_clear_of_noise_abstains() {
        let mut m = SpaceMonitor::new(cfg(true));
        // Baseline 0.03 < 2 x noise 0.02.
        let mut ev = run(&mut m, 0.0, 120.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.03, NOISE)));
        ev.extend(run(&mut m, 120.0, 140.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.001, NOISE))));
        ev.extend(run(&mut m, 140.0, 200.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.03, NOISE))));
        assert!(pauses(&ev).is_empty());

        // Same, but noise unknown: detected with the fallback confidence.
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 120.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.03, None)));
        ev.extend(run(&mut m, 120.0, 140.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.001, None))));
        ev.extend(run(&mut m, 140.0, 200.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.03, None))));
        let p = pauses(&ev);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].confidence, PAUSE_UNKNOWN_NOISE_CONFIDENCE);
    }

    #[test]
    fn breathing_signal_lost_mid_candidate_abstains_and_mid_pause_reports_lower_bound() {
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 120.0, calm);
        ev.extend(run(&mut m, 120.0, 125.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.01, NOISE))));
        ev.extend(run(&mut m, 125.0, 140.0, |t| sample(t, Some(true), 0.5, None)));
        assert!(pauses(&ev).is_empty(), "a 5 s candidate then no signal is not a pause");

        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 120.0, calm);
        ev.extend(run(&mut m, 120.0, 132.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.01, NOISE))));
        ev.extend(run(&mut m, 132.0, 150.0, |t| sample(t, Some(true), 0.5, None)));
        let p = pauses(&ev);
        assert_eq!(p.len(), 1);
        assert!(p[0].evidence.contains("lower bound"));
        match p[0].kind {
            EventKind::BreathingPause { duration_s, .. } => assert!((duration_s - 11.5).abs() < 1e-9),
            _ => unreachable!(),
        }
    }

    #[test]
    fn pause_ended_by_vacancy_without_motion_asks_for_a_check() {
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 120.0, calm);
        ev.extend(run(&mut m, 120.0, 136.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.01, NOISE))));
        ev.extend(run(&mut m, 136.0, 150.0, |t| sample(t, Some(false), 0.5, None)));
        let p = pauses(&ev);
        assert_eq!(p.len(), 1);
        assert_eq!(p[0].severity, Severity::Check);
        assert!(p[0].evidence.contains("without any gross motion"));
    }

    #[test]
    fn pause_ended_by_motion_has_lower_confidence() {
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 120.0, calm);
        ev.extend(run(&mut m, 120.0, 135.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.01, NOISE))));
        ev.extend(run(&mut m, 135.0, 140.0, |t| sample(t, Some(true), 5.0, br(14.0, 0.8, 0.5, NOISE))));
        let p = pauses(&ev);
        assert_eq!(p.len(), 1);
        assert!((p[0].confidence - 0.46).abs() < 1e-9);
    }

    #[test]
    fn vitals_off_never_reports_breathing_events() {
        let mut m = SpaceMonitor::new(cfg(false));
        // Inputs that would trigger pause, rate and trend detectors if vitals were on.
        let mut ev = run(&mut m, 0.0, 120.0, calm);
        ev.extend(run(&mut m, 120.0, 150.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.8, 0.01, NOISE))));
        ev.extend(run(&mut m, 150.0, 600.0, |t| sample(t, Some(true), 0.5, br(30.0, 0.95, 0.25, NOISE))));
        ev.extend(run(&mut m, 600.0, 900.0, |t| sample(t, Some(true), 0.5, br(5.0, 0.95, 0.25, NOISE))));
        for e in &ev {
            assert!(
                !matches!(
                    e.kind,
                    EventKind::BreathingPause { .. }
                        | EventKind::RateHigh { .. }
                        | EventKind::RateLow { .. }
                        | EventKind::TrendDeviation { .. }
                ),
                "{e:?}"
            );
            assert!(!e.evidence.contains("BPM"), "no breathing numbers leak: {}", e.evidence);
        }
        let st = m.state();
        assert!(st.breathing.is_none() && st.bpm_median_5min.is_none() && st.bpm_iqr_5min.is_none());
        assert!(!st.vitals);
        assert!(m.export_history().iter().all(|d| d.resting_bpm_median.is_none() && d.resting_samples == 0));
    }

    fn inactivity(ev: &[HealthEvent]) -> Vec<&HealthEvent> {
        ev.iter().filter(|e| matches!(e.kind, EventKind::Inactivity { .. })).collect()
    }

    #[test]
    fn inactivity_with_breathing_is_advisory() {
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 10.0, |t| sample(t, Some(true), 5.0, None));
        ev.extend(run(&mut m, 10.0, 10.0 + 31.0 * 60.0, |t| sample(t, Some(true), 0.5, br(13.8, 0.82, 0.25, NOISE))));
        let i = inactivity(&ev);
        assert_eq!(i.len(), 1);
        assert_eq!(i[0].severity, Severity::Advisory);
        assert!(matches!(i[0].kind, EventKind::Inactivity { breathing_detected: true, still_min } if (still_min - 30.0).abs() < 0.05));
        assert!(i[0].evidence.contains("13.8 BPM (confidence 0.82)"), "{}", i[0].evidence);
        assert!((i[0].t_s - (10.0 + 1800.0)).abs() <= 1.0);
        assert!(m.state().active_alerts.contains(&"inactivity".to_string()));

        // Breathing disappears later: one re-escalation to Check.
        ev = run(&mut m, 10.0 + 31.0 * 60.0, 10.0 + 35.0 * 60.0, |t| sample(t, Some(true), 0.5, None));
        let i = inactivity(&ev);
        assert_eq!(i.len(), 1);
        assert_eq!(i[0].severity, Severity::Check);
        assert!(matches!(i[0].kind, EventKind::Inactivity { breathing_detected: false, .. }));
        assert!(m.state().active_alerts.contains(&"inactivity_no_breathing".to_string()));

        // Moving ends the episode and its alerts.
        run(&mut m, 2200.0, 2205.0, |t| sample(t, Some(true), 5.0, None));
        assert!(m.state().active_alerts.is_empty());
    }

    #[test]
    fn inactivity_without_breathing_is_a_check_with_vitals_and_advisory_without() {
        let mut m = SpaceMonitor::new(cfg(true));
        let ev = run(&mut m, 0.0, 31.0 * 60.0, |t| sample(t, Some(true), 0.5, None));
        let i = inactivity(&ev);
        assert_eq!(i.len(), 1);
        assert_eq!(i[0].severity, Severity::Check);
        assert!(matches!(i[0].kind, EventKind::Inactivity { breathing_detected: false, .. }));
        assert!(i[0].evidence.contains("no confident breathing"));

        let mut m = SpaceMonitor::new(cfg(false));
        let ev = run(&mut m, 0.0, 31.0 * 60.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.9, 0.25, NOISE)));
        let i = inactivity(&ev);
        assert_eq!(i.len(), 1);
        assert_eq!(i[0].severity, Severity::Advisory);
        assert!(matches!(i[0].kind, EventKind::Inactivity { breathing_detected: false, .. }));
        assert!(i[0].evidence.contains("vitals off"));
    }

    #[test]
    fn sustained_high_rate_alerts_once_brief_spike_does_not() {
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 60.0, calm);
        ev.extend(run(&mut m, 60.0, 600.0, |t| sample(t, Some(true), 0.5, br(28.0, 0.9, 0.25, NOISE))));
        let hi: Vec<_> = ev.iter().filter(|e| matches!(e.kind, EventKind::RateHigh { .. })).collect();
        assert_eq!(hi.len(), 1);
        assert_eq!(hi[0].severity, Severity::Advisory);
        assert!(matches!(hi[0].kind, EventKind::RateHigh { median_bpm, over_s } if median_bpm > 24.0 && over_s > 110.0));
        assert!(m.state().active_alerts.contains(&"rate_high".to_string()));
        assert_eq!(count(&ev, |k| matches!(k, EventKind::RateLow { .. })), 0);

        // Back in range for > 5 min re-arms; a second episode alerts again.
        let mut ev = run(&mut m, 600.0, 1100.0, calm);
        assert!(!m.state().active_alerts.contains(&"rate_high".to_string()));
        ev.extend(run(&mut m, 1100.0, 1400.0, |t| sample(t, Some(true), 0.5, br(28.0, 0.9, 0.25, NOISE))));
        assert_eq!(count(&ev, |k| matches!(k, EventKind::RateHigh { .. })), 1);

        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 200.0, calm);
        ev.extend(run(&mut m, 200.0, 220.0, |t| sample(t, Some(true), 0.5, br(28.0, 0.9, 0.25, NOISE))));
        ev.extend(run(&mut m, 220.0, 600.0, calm));
        assert_eq!(count(&ev, |k| matches!(k, EventKind::RateHigh { .. } | EventKind::RateLow { .. })), 0);
    }

    #[test]
    fn low_rate_alerts_and_thin_coverage_abstains() {
        let mut m = SpaceMonitor::new(cfg(true));
        let ev = run(&mut m, 0.0, 300.0, |t| sample(t, Some(true), 0.5, br(6.0, 0.9, 0.25, NOISE)));
        assert_eq!(count(&ev, |k| matches!(k, EventKind::RateLow { .. })), 1);

        // Confident estimates in only half the samples: rate unknown.
        let mut m = SpaceMonitor::new(cfg(true));
        let ev = run(&mut m, 0.0, 600.0, |t| {
            let conf = if (t * 2.0) as i64 % 2 == 0 { 0.9 } else { 0.5 };
            sample(t, Some(true), 0.5, br(30.0, conf, 0.25, NOISE))
        });
        assert_eq!(count(&ev, |k| matches!(k, EventKind::RateHigh { .. })), 0);
    }

    fn collapse_timeline(m: &mut SpaceMonitor) -> Vec<HealthEvent> {
        let mut ev = run(m, 0.0, 20.0, |t| sample(t, Some(true), 5.0, None)); // walking
        ev.extend(run(m, 20.0, 21.0, |t| sample(t, Some(true), 12.0, None))); // burst
        ev.extend(run(m, 21.0, 22.0, |t| sample(t, Some(true), 2.0, None))); // settling
        ev.extend(run(m, 22.0, 90.0, |t| sample(t, Some(true), 0.5, br(16.0, 0.7, 0.2, NOISE))));
        ev
    }

    #[test]
    fn collapse_fires_only_when_enabled() {
        let mut m = SpaceMonitor::new(SpaceConfig { collapse_detection: true, ..cfg(true) });
        let ev = collapse_timeline(&mut m);
        let c: Vec<_> = ev.iter().filter(|e| matches!(e.kind, EventKind::PossibleCollapse { .. })).collect();
        assert_eq!(c.len(), 1);
        assert_eq!(c[0].severity, Severity::Check);
        assert!(matches!(c[0].kind, EventKind::PossibleCollapse { burst_index, still_s } if burst_index == 12.0 && still_s >= 30.0));
        assert!((c[0].t_s - 52.0).abs() <= 1.0);
        assert!(c[0].evidence.contains("Sitting down heavily looks the same"), "{}", c[0].evidence);
        assert!(m.state().active_alerts.contains(&"possible_collapse".to_string()));

        let mut m = SpaceMonitor::new(SpaceConfig { collapse_detection: false, ..cfg(true) });
        let ev = collapse_timeline(&mut m);
        assert_eq!(count(&ev, |k| matches!(k, EventKind::PossibleCollapse { .. })), 0);
    }

    #[test]
    fn burst_followed_by_more_movement_or_leaving_is_not_a_collapse() {
        let mut m = SpaceMonitor::new(SpaceConfig { collapse_detection: true, ..cfg(true) });
        let mut ev = run(&mut m, 0.0, 1.0, |t| sample(t, Some(true), 12.0, None));
        ev.extend(run(&mut m, 1.0, 11.0, |t| sample(t, Some(true), 4.0, None))); // kept moving > 5 s
        ev.extend(run(&mut m, 11.0, 80.0, |t| sample(t, Some(true), 0.5, None)));
        assert_eq!(count(&ev, |k| matches!(k, EventKind::PossibleCollapse { .. })), 0);

        let mut m = SpaceMonitor::new(SpaceConfig { collapse_detection: true, ..cfg(true) });
        let mut ev = run(&mut m, 0.0, 1.0, |t| sample(t, Some(true), 12.0, None));
        ev.extend(run(&mut m, 1.0, 15.0, |t| sample(t, Some(true), 0.5, None)));
        ev.extend(run(&mut m, 15.0, 80.0, |t| sample(t, Some(false), 0.5, None))); // presence not held
        assert_eq!(count(&ev, |k| matches!(k, EventKind::PossibleCollapse { .. })), 0);
    }

    #[test]
    fn rest_period_starts_and_ends() {
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = run(&mut m, 0.0, 5.0, |t| sample(t, Some(true), 5.0, None));
        ev.extend(run(&mut m, 5.0, 5.0 + 25.0 * 60.0, |t| {
            // A 3 s turn-over at 22 min, once resting, is part of the rest.
            let turning = (1325.0..1328.0).contains(&t);
            sample(t, Some(true), if turning { 4.0 } else { 0.5 }, br(14.0, 0.8, 0.25, NOISE))
        }));
        let started: Vec<_> = ev.iter().filter(|e| e.kind == EventKind::RestStarted).collect();
        assert_eq!(started.len(), 1);
        assert!(started[0].evidence.contains("rest period, not sleep staging"));
        assert!((started[0].t_s - (5.0 + 1200.0)).abs() <= 1.0, "{}", started[0].t_s);
        assert_eq!(started[0].t_start_s, Some(5.0));
        assert!(m.state().resting);
        assert_eq!(count(&ev, |k| matches!(k, EventKind::RestEnded { .. })), 0, "a turn-over does not end rest");

        let t_up = 5.0 + 25.0 * 60.0;
        let ev = run(&mut m, t_up, t_up + 150.0, |t| sample(t, Some(true), 5.0, None));
        let ended: Vec<_> = ev.iter().filter(|e| matches!(e.kind, EventKind::RestEnded { .. })).collect();
        assert_eq!(ended.len(), 1);
        // 117 s more of bout time on top of the 3 s turn-over.
        assert!(ended[0].t_s > t_up + 110.0 && ended[0].t_s <= t_up + 121.0, "{}", ended[0].t_s);
        match &ended[0].kind {
            EventKind::RestEnded { duration_min, motion_bouts, mean_bpm, bpm_iqr, pauses } => {
                assert!((duration_min - 25.0).abs() < 0.02, "{duration_min}");
                assert_eq!(*motion_bouts, 2, "the turn-over and getting up");
                assert!((mean_bpm.unwrap() - 14.0).abs() < 0.1);
                assert!(bpm_iqr.unwrap() < 0.2);
                assert_eq!(*pauses, 0);
            }
            _ => unreachable!(),
        }
        assert!(ended[0].evidence.contains("not sleep staging"));
        assert!(!m.state().resting);
    }

    #[test]
    fn rest_needs_breathing_with_vitals_but_not_without() {
        let mut m = SpaceMonitor::new(cfg(true));
        let ev = run(&mut m, 0.0, 25.0 * 60.0, |t| sample(t, Some(true), 0.5, br(14.0, 0.3, 0.25, NOISE)));
        assert_eq!(count(&ev, |k| *k == EventKind::RestStarted), 0);

        let mut m = SpaceMonitor::new(cfg(false));
        let mut ev = run(&mut m, 0.0, 25.0 * 60.0, |t| sample(t, Some(true), 0.5, None));
        assert_eq!(count(&ev, |k| *k == EventKind::RestStarted), 1);
        ev = run(&mut m, 1500.0, 1600.0, |t| sample(t, Some(false), 0.5, None));
        let ended: Vec<_> = ev.iter().filter(|e| matches!(e.kind, EventKind::RestEnded { .. })).collect();
        assert_eq!(ended.len(), 1);
        assert!(matches!(ended[0].kind, EventKind::RestEnded { mean_bpm: None, bpm_iqr: None, .. }));
    }

    /// 10 minutes of resting breathing at `bpm` an hour into `day`.
    fn rest_day(m: &mut SpaceMonitor, day: i64, bpm: f64) -> Vec<HealthEvent> {
        let t0 = day as f64 * DAY_S + 3600.0;
        run(m, t0, t0 + 600.0, |t| sample(t, Some(true), 0.5, br(bpm, 0.9, 0.25, NOISE)))
    }

    fn trends(ev: &[HealthEvent]) -> Vec<&HealthEvent> {
        ev.iter().filter(|e| matches!(e.kind, EventKind::TrendDeviation { .. })).collect()
    }

    #[test]
    fn trend_deviation_after_four_days() {
        let mut m = SpaceMonitor::new(cfg(true));
        let mut ev = Vec::new();
        for (d, bpm) in [(0, 14.0), (1, 14.4), (2, 13.8), (3, 19.0)] {
            ev.extend(rest_day(&mut m, d, bpm));
        }
        assert!(trends(&ev).is_empty(), "nothing before day 3 is complete");
        let ev = rest_day(&mut m, 4, 14.0);
        let tr = trends(&ev);
        assert_eq!(tr.len(), 1);
        assert_eq!(tr[0].severity, Severity::Info);
        match &tr[0].kind {
            EventKind::TrendDeviation { metric, value, baseline, mad } => {
                assert_eq!(metric, "resting_bpm");
                assert!((value - 19.0).abs() < 0.11, "{value}");
                assert!((baseline - 14.0).abs() < 0.11, "{baseline}");
                assert!(*mad < 0.5);
            }
            _ => unreachable!(),
        }
        assert_eq!(tr[0].t_start_s, Some(3.0 * DAY_S));

        // An ordinary day does not deviate.
        let ev = rest_day(&mut m, 5, 14.2);
        assert!(trends(&ev).is_empty());
        let h = m.export_history();
        assert_eq!(h.iter().map(|d| d.day).collect::<Vec<_>>(), vec![0, 1, 2, 3, 4, 5]);
        assert!(h.iter().all(|d| d.resting_samples >= TREND_MIN_RESTING_SAMPLES));
    }

    #[test]
    fn history_survives_a_restart() {
        let mut m = SpaceMonitor::new(cfg(true));
        for (d, bpm) in [(0, 14.0), (1, 14.4), (2, 13.8)] {
            rest_day(&mut m, d, bpm);
        }
        let saved = m.export_history();
        assert_eq!(saved.len(), 3);

        let mut m2 = SpaceMonitor::new(cfg(true));
        m2.import_history(saved.clone());
        assert_eq!(m2.export_history(), saved);
        let mut ev = rest_day(&mut m2, 3, 8.5);
        ev.extend(rest_day(&mut m2, 4, 14.0));
        let tr = trends(&ev);
        assert_eq!(tr.len(), 1);
        assert!(tr[0].evidence.contains("below"));
    }

    fn occ(ev: &[HealthEvent]) -> (usize, Vec<f64>) {
        let starts = count(ev, |k| *k == EventKind::OccupancyStarted);
        let ends = ev
            .iter()
            .filter_map(|e| match e.kind {
                EventKind::OccupancyEnded { dwell_s } => Some(dwell_s),
                _ => None,
            })
            .collect();
        (starts, ends)
    }

    #[test]
    fn occupancy_session_survives_a_short_blind_gap() {
        let mut m = SpaceMonitor::new(cfg(false));
        let mut ev = run(&mut m, 0.0, 60.0, |t| sample(t, Some(true), 0.5, None));
        ev.extend(run(&mut m, 60.0, 90.0, |t| sample(t, None, 0.0, None)));
        assert_eq!(m.state().occupied_since_s, Some(0.0));
        assert!(m.state().active_alerts.contains(&"blind".to_string()));
        ev.extend(run(&mut m, 90.0, 150.0, |t| sample(t, Some(true), 0.5, None)));
        ev.extend(run(&mut m, 150.0, 170.0, |t| sample(t, Some(false), 0.5, None)));
        let (starts, ends) = occ(&ev);
        assert_eq!(starts, 1);
        assert_eq!(ends.len(), 1);
        assert!((ends[0] - 150.0).abs() < 1e-9);
        assert_eq!(m.state().occupied_since_s, None);

        // A 90 s blind gap does close the session (at the last sighting).
        let mut m = SpaceMonitor::new(cfg(false));
        let mut ev = run(&mut m, 0.0, 60.0, |t| sample(t, Some(true), 0.5, None));
        ev.extend(run(&mut m, 60.0, 150.0, |t| sample(t, None, 0.0, None)));
        ev.extend(run(&mut m, 150.0, 160.0, |t| sample(t, Some(true), 0.5, None)));
        let (starts, ends) = occ(&ev);
        assert_eq!(starts, 2);
        assert_eq!(ends.len(), 1);
        assert!((ends[0] - 59.5).abs() < 1e-9);
    }

    #[test]
    fn motion_bouts_are_counted_per_hour() {
        let mut m = SpaceMonitor::new(cfg(false));
        let ev = run(&mut m, 0.0, 600.0, |t| {
            // 2 s fidgets every 60 s, each with a 1 s dip mid-way (same bout).
            let ph = t % 60.0;
            let motion = if ph < 1.0 || (2.0..3.0).contains(&ph) { 4.0 } else { 0.5 };
            sample(t, Some(true), motion, None)
        });
        assert!(ev.iter().all(|e| e.kind == EventKind::OccupancyStarted));
        let st = m.state();
        assert_eq!(st.motion_bouts_last_hour, 10);
        assert!(st.still_for_s > 50.0 && st.still_for_s < 60.0);
        assert_eq!(m.export_history()[0].motion_bouts, 10);
    }

    #[test]
    fn state_reports_five_minute_breathing_statistics() {
        let mut m = SpaceMonitor::new(cfg(true));
        run(&mut m, 0.0, 300.0, |t| {
            let bpm = if (t * 2.0) as i64 % 2 == 0 { 13.0 } else { 15.0 };
            sample(t, Some(true), 0.5, br(bpm, 0.5, 0.25, NOISE))
        });
        let st = m.state();
        assert!((st.bpm_median_5min.unwrap() - 14.0).abs() < 1e-9);
        assert!((st.bpm_iqr_5min.unwrap() - 2.0).abs() < 1e-9);
        assert_eq!(st.space_id, "room");
        assert!(st.vitals && st.breathing.is_some());
        assert!((st.still_for_s - 299.5).abs() < 1e-9);
    }
}
