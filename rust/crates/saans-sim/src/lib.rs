//! # saans-sim: Saans simulator. Everything it produces is SYNTHETIC.
//!
//! Physically modelled multi-link WiFi CSI for scripted human timelines
//! (walk, sit, breathe, breathing pause, fidget, fall, sit down), with
//! per-tick and interval ground truth, so the health detectors can be
//! evaluated before (and alongside) real hardware. Nothing here is a
//! measurement of a real room, a real body or a real ESP32.
//!
//! ## Physics
//!
//! A port of the Python reference's geometric scene
//! (`wifisense/sources/synthetic.py`, `scene="geometric"`), generalised to
//! many links and a moving body:
//!
//! - 2-D room; per link a static channel of line of sight, image-method
//!   wall reflections and furniture point scatterers ([`physics`]);
//! - the body is `body_points` scatterers across a torso of
//!   `body_width_m`, perpendicular to each link's tx-body-rx bisector, RCS
//!   0.3 m^2 split equally, recomputed for the current body position every
//!   sample; chest displacement (breathing 5 mm, heart 0.4 mm at 68 BPM)
//!   changes each torso path by `2 cos(beta/2)` per metre;
//! - an arm scatterer (0.05 m^2, 0.3 m from the body centre) that swings
//!   ±5 cm at 1-2 Hz during fidget bursts and flails during a fall;
//! - complex Gaussian noise, per-link timestamp jitter, independent packet
//!   loss with a per-link u32 sequence counter, RSSI from mean |h|.
//!
//! [`Scenario`] is the contract; carrier, subcarrier spacing, jitter and
//! the arm's RCS are [`SimOptions`] (defaults: 2.437 GHz, 312.5 kHz, 2 ms,
//! 0.05 m^2) passed with [`Simulator::with_options`].
//!
//! `tests/parity.rs` reproduces the Python scene's golden vectors
//! (`spec/golden/sim_geometric`) to ~1e-13.
//!
//! ## Driving it by hand
//!
//! A scripted timeline is not the only way to move the body.
//! [`Simulator::set_position_override`] pins it wherever a caller says and
//! [`Simulator::set_present`] takes it out of the room altogether, both
//! mid-run and without disturbing the sequence counters, the RNG streams
//! or the static channel — so an interactive explorer can drag a person
//! around a room and the CSI stays continuous across the change.
//!
//! Those positions are an *input*, never an inference: the truth reports
//! what the caller set. Nothing in this crate recovers a position from
//! CSI, and a single-antenna link cannot (`docs/CAPABILITY_LIMITS.md`).
//!
//! [`Simulator::sensitivity_at`] answers what breathing would do to a link
//! from an arbitrary point, and [`Simulator::coverage`] evaluates that over
//! a grid of the whole room: the ground-truth map of where a chest is
//! detectable and where it falls in a Fresnel null. It is an upper bound
//! computed from the scenario's geometry — no receiver can measure it.
//!
//! ## Simplifications (beyond the reference's own: 2-D, no AGC, no CFO)
//!
//! - The torso faces every link (spread across *that link's* bisector),
//!   as in the reference; a real torso has one orientation.
//! - No shadowing of the LoS by the body; no near-field (ranges below
//!   10 cm are clamped in the scatterer amplitude).
//! - Falls and sitting down are horizontal moves in the 2-D slice (a real
//!   one is mostly vertical); lying down does not change the RCS.
//! - Walking is constant speed between waypoints, no gait sway; still
//!   people have no postural sway.
//! - Phases are the clean channel phase (no CFO/SFO), so they are far
//!   better than what an ESP32 reports. Use amplitudes.
//!
//! [`impair`] models the receiver those simplifications are about — AGC,
//! random per-packet phase, timing slope, static per-device ripple,
//! int8 I/Q — as a separate stage applied after the clean channel. It is
//! OFF BY DEFAULT and nothing above changes when it is on: the clean
//! channel is the contract with the Python and C++ implementations
//! (`tests/parity.rs`). Its own realism is UNVERIFIED — see the module.
//!
//! [`wire`] puts the result in the raw-CSI datagram the firmware emits
//! and the gateway already parses (`spec/wire.md` §1), so a simulated
//! node and a bench node feed the gateway through one decoder. It, too,
//! does nothing unless a caller asks for it.
//!
//! ## Use
//!
//! ```
//! use saans_sim::{scenarios, Simulator};
//! let sc = scenarios::scenario("night", scenarios::room_rich_nlos(), scenarios::three_links(),
//!                              scenarios::night(0.1, 4.0, (10.0, 40.0)), 42);
//! let mut sim = Simulator::new(&sc);
//! let events = sim.truth_events();
//! while let Some(tick) = sim.step() {
//!     for lf in &tick.frames {
//!         let _ = (lf.tx, lf.rx, lf.seq, &lf.frame.amplitudes);
//!     }
//! }
//! # let _ = events;
//! ```

pub mod impair;
pub mod physics;
mod plan;
pub mod rng;
pub mod scenario;
pub mod scenarios;
mod sim;
pub mod wire;

pub use impair::{BinLayout, ImpairConfig, NodeImpairment};
pub use plan::{
    ARM_OFFSET_M, ARM_SWING_M, BREATHING_AMPLITUDE_M, DEFAULT_BPM, FALL_S, HEART_AMPLITUDE_M, HEART_BPM, MARGIN_M, PAUSE_RAMP_S, SIT_DOWN_DROP_M, SIT_DOWN_S,
};
pub use scenario::{Activity, Node, Point, Room, Scatterer, Scenario, Segment, SimOptions, Truth, TruthEvent, TruthKind};
pub use sim::{Coverage, LinkFrame, Simulator, Tick, MAX_COVERAGE_CELLS, RSSI_REF_DBM};
pub use wire::{RawNode, RawNodes};
