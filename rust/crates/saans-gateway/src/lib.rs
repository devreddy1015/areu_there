//! Saans gateway: the box between the sensors and the people who look at
//! the dashboard (SAANS_PLAN.md §3).
//!
//! ```text
//! UDP :5566 ─► engine (FeatureFrame v1 | raw CSI ─► saans-dsp) ─► per-space
//!   SpaceFusion ─► SpaceMonitor + SystemMonitor ─► hub ─┬─ events-YYYY-MM-DD.jsonl, history.json
//!                                                        └─ snapshots ─► HTTP API, WebSocket, /metrics, dashboard
//! ```
//!
//! - [`engine`]: deterministic core, no I/O, caller supplies time.
//! - [`hub`]: side effects (logs, history, retention, publishing).
//! - [`live`]: sockets, the 0.5 s tick, shutdown; [`replay`]: pcap input.
//! - [`privacy`]: the consent gate every outbound value passes through.
//! - [`history`]: a bounded in-memory time series for the trend charts.
//! - [`roles`], [`auth`], [`audit`]: who may see what, and the record of it.
//! - [`outputs`]: the fan-out MQTT ([`mqtt`]) and alerting ([`alerts`]) consume.
//! - `roomsim` (private, `roomsim` feature only): the static room demo at
//!   `/roomsim`, drawing a declared scene with vendored three.js (ADR-008).
//!   Serves no sensor data.
//! - `roomstream` (`roomsim` feature only): the room stream at
//!   `/api/roomsim`, a simulated (SIM) or recorded (REPLAY) Room Sim run
//!   through Locate, one session thread per connection
//!   (spec/livedemo.md). Never fed by, and never feeding, the sensor path.

pub mod alerts;
pub mod audit;
pub mod auth;
pub mod config;
pub mod engine;
pub mod history;
pub mod http;
pub mod hub;
pub mod live;
pub mod mqtt;
pub mod outputs;
pub mod pcap;
pub mod privacy;
pub mod raw;
pub mod replay;
pub mod roles;
#[cfg(feature = "roomsim")]
mod roomsim;
#[cfg(feature = "roomsim")]
pub mod roomstream;
pub mod shared;
pub mod store;
pub mod timefmt;

pub use config::GatewayConfig;
pub use engine::Engine;
pub use live::{ClockMode, Gateway};
pub use replay::{replay, ReplayOptions, ReplaySummary};

/// One log line on stderr, wall-clock stamped (UTC).
pub fn log(msg: &str) {
    eprintln!("{} saans-gateway: {msg}", timefmt::iso8601(timefmt::now_unix_ms(), 0));
}
