//! # saans-roomsim: Room Sim v1. Everything it produces is SIM.
//!
//! A physically motivated model of what ESP32 nodes in a declared room
//! would report, implemented exactly as `spec/roomsim.md` writes it (the
//! Python reference `wifisense/roomsim/` implements the same spec, and the
//! two agree to ~1e-9 relative, §14). Every number this crate outputs is
//! **SIM**: a model, not a measurement. Read §16 of the spec, what it does
//! not model, before drawing a conclusion from its output.
//!
//! - **The room comes from the scene file, never from WiFi.** Walls,
//!   furniture and nodes are read from a `saans_scene::Scene` (ADR-008).
//!   This crate depends on `saans-scene` without its `write` feature, so
//!   it cannot write a scene by any route.
//! - **Static paths** per link (§6): line of sight, wall/floor/ceiling
//!   bounces to order 2 with ITU-R P.2040 slab coefficients (§5), object
//!   faces as specular reflectors or point scatterers, clutter, and a
//!   diffuse delay tail; wall transmission and object blockage on every
//!   leg ([`paths`]).
//! - **People** (§8-9): 6-16 body points with breathing and gait, a
//!   three-knife-edge cylinder shadow, walk/pause trajectories that avoid
//!   furniture ([`people`], [`motion`]).
//! - **ESP32 impairments** (§12): ripple, AGC, CPE, timing offset, noise,
//!   LTF merge, int8 quantisation and the raw datagram the firmware sends
//!   ([`impair`], [`datagram`]).
//!
//! Randomness is `saans_sim::rng::Rng` with one stream per purpose (§2);
//! no switch ever changes a draw sequence.
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let scene = saans_scene::Scene::load("spec/scenes/bedroom.json")?;
//! let sim = saans_roomsim::Simulator::new(&scene, saans_roomsim::Config::default())?;
//! for tx in sim.run() {
//!     for p in &tx.packets {
//!         if let Some(r) = &p.received {
//!             println!("SIM t={} {}->{} seq={} rssi={}", p.t, p.tx, p.rx, r.seq, r.rssi);
//!         }
//!     }
//! }
//! # Ok(()) }
//! ```

#![forbid(unsafe_code)]

pub mod config;
pub mod datagram;
pub mod error;
pub mod geometry;
pub mod impair;
pub mod materials;
pub mod motion;
pub mod numerics;
pub mod paths;
pub mod people;
pub mod room;
pub mod sim;

pub use config::{Config, Mode, Posture, Role};
pub use error::Error;
pub use geometry::{v3, P2, V3};
pub use numerics::{cx, Cx};
pub use paths::{PathKind, StaticPath};
pub use sim::{
    Buffer, Link, Packet, PeopleAt, PersonModel, PersonTrack, PersonTruth, Received, Run,
    Simulator, Transmission, Truth, LABEL, SPEC,
};
