//! Locate v1: people and phone positions with uncertainty (spec/locate.md).
//!
//! Everything this crate produces in Phase 4 is **SIM**: its inputs are
//! Room Sim packets and truth, its geometry is the scene (SCENE). An
//! estimate is always a position with a radius, or an abstention with a
//! reason; never a bare point (ADR-008 decision 5). The phone is simulated
//! only (ADR-008 decision 16).
//!
//! The ported modules (`geometry`, `invert`, `measure`, `track`) follow the
//! Python reference `wifisense/localize` line by line (ADR-002); the new
//! ones (`radius`, `baseline`, `phone`, `pipeline`) follow spec/locate.md.
#![forbid(unsafe_code)]

// Ported from wifisense/localize (spec/locate.md §1-A, §3).
pub mod geometry;
pub mod invert;
pub mod measure;
pub mod pynum;
pub mod track;

// New in Phase 4 (spec/locate.md §1-B, §5-§8).
pub mod baseline;
pub mod num;
pub mod phone;
pub mod radius;

// The Room Sim pipeline (§4.3, §10, §11), behind the `roomsim` feature.
#[cfg(feature = "roomsim")]
pub mod pipeline;

/// The only label Phase 4 produces.
pub const LABEL: &str = "SIM";
