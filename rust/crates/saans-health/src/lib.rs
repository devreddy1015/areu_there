//! Saans health monitoring.
//!
//! ```text
//! FeatureFrames (per link) --fusion--> SpaceSample (2 Hz, per space)
//!     --detectors--> HealthEvent stream + SpaceState snapshot
//! LinkTracks --system--> system-health events (links down, stale calibration, raw mode)
//! ```
//!
//! Everything here is advisory (ADR-005): outputs carry confidences,
//! detectors abstain when the evidence is thin, and every alert's
//! `evidence` field says what was observed.

pub mod detectors;
pub mod fusion;
pub mod system;
pub mod types;

pub use detectors::SpaceMonitor;
pub use fusion::SpaceFusion;
pub use system::SystemMonitor;
pub use types::*;
