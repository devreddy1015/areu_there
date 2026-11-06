//! The health layer's contract: what goes in, what comes out.
//!
//! Inputs are FeatureFrames (one stream per link). The fusion stage turns
//! the links of one space into a 2 Hz `SpaceSample` stream; detectors read
//! only `SpaceSample`s and emit `HealthEvent`s; `SpaceState` is the live
//! snapshot a dashboard shows. Every number a person might act on carries
//! a confidence, and every alert says what was observed (ADR-005).

use serde::{Deserialize, Serialize};

/// A link is one (receiving node, transmitting node) pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct LinkId {
    pub rx: u8,
    pub tx: u8,
}

impl std::fmt::Display for LinkId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}<-{}", self.rx, self.tx)
    }
}

/// Per-space configuration. Defaults are conservative: vitals off,
/// collapse detection off, alerts advisory.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default)]
pub struct SpaceConfig {
    pub id: String,
    pub name: String,
    pub links: Vec<LinkId>,
    /// ADR-005: breathing/heart features and every detector that reads
    /// them are off unless the space owner has opted in with consent.
    pub vitals: bool,
    /// Usable links that must report occupied. 0 = automatic:
    /// 1 if fewer than 3 usable links, otherwise a majority.
    pub presence_quorum: usize,
    /// Lone-worker inactivity alert after this long without gross motion.
    pub inactivity_alert_min: f64,
    /// Possible-collapse detection. Off by default: WiFi cannot tell
    /// "fell" from "sat down heavily" (see the S3 evaluation).
    pub collapse_detection: bool,
    /// Hold presence for up to this long after a fall-shaped motion: a
    /// burst that stops abruptly, after which the links read empty with
    /// nothing moving. A person on the floor can be too weak a breather
    /// for the links to see (NOTES.md §14), and without the hold the space
    /// reads empty and the inactivity alert never starts counting. Held
    /// presence is inferred, not measured: `SpaceState::occupied` stays
    /// what the links say and `presence_held` says it is being held.
    ///
    /// Off (0) by default, like collapse detection and for the same
    /// reason: it cannot tell "fell out of sight" from "walked out". In
    /// simulation about one departure in four engages it, and in a room
    /// that then stays empty each of those is an inactivity alert 30 min
    /// later. To turn it on, set it longer than `inactivity_alert_min`
    /// (35 for the default 30).
    pub fall_hold_min: f64,
    pub breathing_pause_min_s: f64,
    pub rate_high_bpm: f64,
    pub rate_low_bpm: f64,
}

impl Default for SpaceConfig {
    fn default() -> Self {
        Self {
            id: "space".into(),
            name: String::new(),
            links: Vec::new(),
            vitals: false,
            presence_quorum: 0,
            inactivity_alert_min: 30.0,
            collapse_detection: false,
            fall_hold_min: 0.0,
            breathing_pause_min_s: 10.0,
            rate_high_bpm: 24.0,
            rate_low_bpm: 8.0,
        }
    }
}

/// One fused 2 Hz sample for a space (output of `fusion::SpaceFusion`).
///
/// Motion is expressed as *indices*: a link's raw energy divided by its own
/// empty-room presence threshold, so links with different amplitude
/// scales are comparable and 1.0 means "the level an empty room stays
/// below".
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SpaceSample {
    /// Gateway time in seconds (monotonic within a run).
    pub t_s: f64,
    /// None = no usable link: the system cannot see this space right now.
    pub occupied: Option<bool>,
    /// max over usable links of motion_energy / presence_threshold.
    pub presence_index: f64,
    /// max over usable links of gross_motion / presence_threshold.
    pub motion_index: f64,
    pub links_usable: u8,
    pub links_total: u8,
    /// Breathing, from the chosen link; None when vitals are off, nobody is
    /// there, or no link has a usable estimate.
    pub breathing: Option<Breathing>,
    pub heart: Option<Rate>,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Rate {
    pub bpm: f64,
    pub confidence: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Breathing {
    pub link: LinkId,
    pub bpm: f64,
    pub confidence: f64,
    /// breath_rms of that link (0.1-0.5 Hz band RMS over 2 s).
    pub band_rms: f64,
    /// That link's empty-room breath_rms level, if learned (for abstention).
    pub noise_rms: Option<f64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Severity {
    /// Recorded, shown, never pushed.
    Info,
    /// Worth a look when convenient.
    Advisory,
    /// Someone should check on the space now. Still advisory: never a
    /// replacement for existing safety procedures.
    Check,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum EventKind {
    OccupancyStarted,
    OccupancyEnded {
        dwell_s: f64,
    },
    BreathingPause {
        duration_s: f64,
        ongoing: bool,
    },
    RateHigh {
        median_bpm: f64,
        over_s: f64,
    },
    RateLow {
        median_bpm: f64,
        over_s: f64,
    },
    Inactivity {
        still_min: f64,
        breathing_detected: bool,
    },
    PossibleCollapse {
        burst_index: f64,
        still_s: f64,
    },
    RestStarted,
    RestEnded {
        duration_min: f64,
        motion_bouts: u32,
        mean_bpm: Option<f64>,
        bpm_iqr: Option<f64>,
        pauses: u32,
    },
    TrendDeviation {
        metric: String,
        value: f64,
        baseline: f64,
        mad: f64,
    },
    // -- system health --
    LinkDown {
        link: LinkId,
        silent_s: f64,
    },
    LinkUp {
        link: LinkId,
    },
    LinkDegraded {
        link: LinkId,
        reason: String,
    },
    CalibrationStale {
        link: LinkId,
        age_h: f64,
    },
    RawModeActive {
        link: LinkId,
    },
    SpaceBlind,
    SpaceVisible,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct HealthEvent {
    pub space_id: String,
    pub t_s: f64,
    /// For events describing an interval (pauses, rest periods).
    pub t_start_s: Option<f64>,
    pub severity: Severity,
    pub confidence: f64,
    /// What was observed, in words a non-engineer can check.
    pub evidence: String,
    #[serde(flatten)]
    pub kind: EventKind,
}

/// Live snapshot of one space for the API and dashboard.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct SpaceState {
    pub space_id: String,
    pub t_s: f64,
    pub occupied: Option<bool>,
    pub occupied_since_s: Option<f64>,
    pub still_for_s: f64,
    pub motion_index: f64,
    pub breathing: Option<Breathing>,
    /// Median of confident breathing estimates over the last 5 minutes.
    pub bpm_median_5min: Option<f64>,
    pub bpm_iqr_5min: Option<f64>,
    pub motion_bouts_last_hour: u32,
    pub resting: bool,
    pub in_breathing_pause: bool,
    pub links_usable: u8,
    pub links_total: u8,
    pub vitals: bool,
    pub active_alerts: Vec<String>,
    /// The links read empty, but presence is being held after a
    /// fall-shaped motion (`SpaceConfig::fall_hold_min`). Inferred, not
    /// measured: `occupied` is what the links say.
    pub presence_held: bool,
}
