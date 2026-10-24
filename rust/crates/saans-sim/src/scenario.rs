//! Scenario description: the contract between the simulator, the
//! evaluation harness and anyone writing a new test timeline.
//!
//! Everything a [`Scenario`] produces is SYNTHETIC: a physically
//! motivated model, not a measurement.

use serde::{Deserialize, Serialize};

pub type Point = (f64, f64);
/// (x, y, z), z up from the floor: the axes `scenes/*.json` use.
pub type Point3 = (f64, f64, f64);

/// Mounting height of a node when nothing says otherwise: every scene in
/// `scenes/` puts its nodes at 1.2 m.
pub const NODE_Z_M: f64 = 1.2;

fn default_node_z() -> f64 {
    NODE_Z_M
}

/// A furniture point scatterer placed explicitly (in addition to the
/// `n_scatterers` drawn at random from the scenario seed).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Scatterer {
    pub pos: Point,
    pub rcs_m2: f64,
    /// Height above the floor. Ignored in the 2-D slice; required when the
    /// room has a [`Vertical`], because a height is not something to
    /// guess for a scatterer somebody placed on purpose.
    #[serde(default)]
    pub z_m: Option<f64>,
}

/// The room as a box rather than a horizontal slice: the Python
/// `SyntheticConfig(room_height_m=...)`. Walls, floor and ceiling all
/// reflect (image method over six faces, each image once), and nodes,
/// chest and furniture have heights. See NOTES.md §13 for what it assumes.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Vertical {
    /// Floor to ceiling.
    pub height_m: f64,
    /// |Gamma| of the floor and of the ceiling, applied with a pi phase
    /// flip like `wall_reflection`, and like it the same at every angle.
    pub floor_reflection: f64,
    pub ceiling_reflection: f64,
    /// Chest height: ~1.0 m seated, ~1.3 m standing, ~0.6 m lying in bed.
    /// One height for the whole timeline; the arm is at it too. The chest
    /// breathes horizontally, along the horizontal part of the
    /// tx-chest-rx bisector.
    pub chest_z_m: f64,
    /// Heights of the random furniture scatterers, uniform in this range,
    /// drawn after every horizontal draw so a seed puts the same furniture
    /// footprint in the slice and the box.
    pub scatterer_z_m: Point,
}

impl Default for Vertical {
    /// A 2.7 m room (`scenes/bedroom.json`) with a concrete floor and
    /// ceiling: 0.395, the ITU-R P.2040 row in `scenes/materials.json`. The
    /// same defaults as the Python reference.
    fn default() -> Self {
        Self {
            height_m: 2.7,
            floor_reflection: 0.395,
            ceiling_reflection: 0.395,
            chest_z_m: 1.0,
            scatterer_z_m: (0.3, 1.8),
        }
    }
}

/// The room's static radio environment (same physics as the Python
/// `SyntheticConfig(scene="geometric")`, extended to many links).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Room {
    pub size_m: Point,
    pub wall_reflection: f64,
    pub reflection_order: u32,
    pub los_attenuation_db: f64,
    /// Furniture scatterers at seeded random positions, uniform in the
    /// central 90% of the room (Python's rule), RCS uniform in
    /// `scatterer_rcs_m2`.
    pub n_scatterers: usize,
    pub scatterer_rcs_m2: Point,
    /// Explicit scatterers, summed before the random ones. Lets a test
    /// reproduce a Python scene exactly (`spec/golden/sim_geometric`).
    #[serde(default)]
    pub scatterers: Vec<Scatterer>,
    /// Where a person appears when they enter (and should walk to when
    /// they leave). Clamped into the walkable area.
    #[serde(default = "default_door")]
    pub door_m: Point,
    /// `None` is the 2-D slice at chest height (the original model, and
    /// every scenario in this crate); `Some` makes the room a box.
    #[serde(default)]
    pub vertical: Option<Vertical>,
}

fn default_door() -> Point {
    (0.6, 0.6)
}

impl Default for Room {
    fn default() -> Self {
        Self {
            size_m: (6.0, 5.0),
            wall_reflection: 0.4,
            reflection_order: 1,
            los_attenuation_db: 0.0,
            n_scatterers: 6,
            scatterer_rcs_m2: (0.05, 0.5),
            scatterers: Vec::new(),
            door_m: default_door(),
            vertical: None,
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Node {
    pub id: u8,
    pub pos: Point,
    /// Mounting height. Ignored in the 2-D slice.
    #[serde(default = "default_node_z")]
    pub z_m: f64,
}

/// What the person is doing for one segment of the timeline.
///
/// Positions are clamped into the walkable area (0.4 m from every wall),
/// so the body, torso points and arm always stay inside the room. A
/// still activity (`Seated`, `BreathingPause`, `Fidget`) whose `at` is not
/// where the person already is starts with a walk there at 1 m/s
/// (labelled `Walking`); if the person was absent they appear at `at`.
/// Moving activities after an absence start at the room's door.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Activity {
    /// Nobody in the room.
    Absent,
    /// Walking between random waypoints at this speed (m/s).
    Walk { speed_mps: f64 },
    /// Walk from wherever the person is to `to`, then stand still.
    WalkTo { to: Point, speed_mps: f64 },
    /// Still, breathing at `bpm` (linearly drifting to `bpm_end` if given).
    Seated {
        at: Point,
        bpm: f64,
        bpm_end: Option<f64>,
    },
    /// Still, breathing stopped (chest motion ramps to zero over ~1 s,
    /// and back over ~1 s after the segment ends). The heart keeps beating.
    BreathingPause { at: Point },
    /// Seated and breathing, plus limb-motion bursts of `burst_s` every
    /// `every_s` seconds on average (Poisson starts; a burst never
    /// straddles the segment end).
    Fidget {
        at: Point,
        bpm: f64,
        every_s: f64,
        burst_s: f64,
    },
    /// A fast fall: the body moves from its current position to `to` in
    /// 0.8 s (arm flailing), then lies still breathing at `bpm` for the
    /// rest of the segment's `duration_s`.
    Fall { to: Point, bpm: f64 },
    /// Sitting down heavily: the same kind of fast move as `Fall`, 0.4 m
    /// in 0.6 s ending at `at`, then still. If the person is further than
    /// 0.4 m from `at` they first walk to the point 0.4 m away. Exists to
    /// measure how often "possible collapse" fires on an ordinary action.
    SitDown { at: Point, bpm: f64 },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Segment {
    pub duration_s: f64,
    pub activity: Activity,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Scenario {
    pub name: String,
    pub room: Room,
    pub nodes: Vec<Node>,
    /// (tx node id, rx node id) pairs. Every link is sampled every tick.
    pub links: Vec<(u8, u8)>,
    pub timeline: Vec<Segment>,
    pub sample_rate_hz: f64,
    pub n_subcarriers: usize,
    /// Complex noise std per component against |LoS| = 40.
    pub noise_std: f64,
    /// Independent per-packet loss probability.
    pub loss_prob: f64,
    /// Torso model: scatter points across `body_width_m` (ADR/NOTES §6.3).
    pub body_points: usize,
    pub body_width_m: f64,
    pub seed: u64,
}

/// Simulator knobs outside the scenario contract, with defaults that
/// match the evaluation. Pass them with [`crate::Simulator::with_options`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct SimOptions {
    /// Channel centre frequency (2.4 GHz channel 6 by default).
    pub carrier_hz: f64,
    pub subcarrier_spacing_hz: f64,
    /// Each link's sample instant (and reported timestamp) lags the tick
    /// by a uniform random 0..`jitter_s`. 0 for exact-time sampling.
    pub jitter_s: f64,
    /// RCS of the arm scatterer (0.3 m from the body centre; swings during
    /// fidget bursts and falls, otherwise still). 0 disables it, which is
    /// what the Python parity test needs.
    pub arm_rcs_m2: f64,
}

impl Default for SimOptions {
    fn default() -> Self {
        Self {
            carrier_hz: 2.437e9,
            subcarrier_spacing_hz: 312_500.0,
            jitter_s: 0.002,
            arm_rcs_m2: 0.05,
        }
    }
}

/// Ground truth at one tick.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Truth {
    pub t_s: f64,
    pub present: bool,
    /// Present, and no walking, no fall/sit-down drop, no limb burst.
    pub still: bool,
    /// Chest moving with breathing (false during a pause or when absent).
    pub breathing: bool,
    /// Instantaneous breathing rate; 0 when `breathing` is false.
    pub bpm: f64,
    pub in_pause: bool,
    pub in_burst: bool,
    /// Body centre (when absent: where the person was last, or the door).
    pub position: Point,
}

/// Interval truths the evaluation scores detectors against.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TruthEvent {
    pub kind: TruthKind,
    pub t_start_s: f64,
    pub t_end_s: f64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum TruthKind {
    Occupied,
    BreathingPause,
    Fall,
    SitDown,
    Walking,
    Burst,
}
