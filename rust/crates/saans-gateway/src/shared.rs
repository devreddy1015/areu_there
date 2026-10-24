//! What the core publishes for the HTTP side to read. The core task owns
//! the engine outright; after every tick it replaces the published
//! snapshot and pushes WebSocket messages, so a slow HTTP client can never
//! stall ingest and handlers never see a half-updated space.

use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
use crate::history::{History, HistoryConfig};
use crate::outputs::{OutputMsg, OUTPUT_CAPACITY};
use serde::{Deserialize, Serialize};
use tokio::sync::broadcast;

/// Events kept in memory for `/api/events` (older ones are in the logs).
pub const EVENT_RING: usize = 10_000;

#[derive(Debug, Clone)]
pub struct EventEntry {
    pub space_id: String,
    pub unix_ms: i64,
    pub value: Arc<Value>,
}

#[derive(Debug, Default)]
pub struct Published {
    pub t_s: f64,
    /// (space id, redacted state JSON), in config order.
    pub spaces: Vec<(String, Value)>,
    pub links: Value,
    pub health: Value,
    pub metrics: String,
    /// The last `{"type":"state"}` message, sent to new WebSocket clients.
    pub state_msg: Option<Arc<str>>,
}

/// Where the data a view shows comes from (ADR-008 d15, spec/livedemo.md
/// §1). Closed: every view maps it exhaustively with no default branch,
/// so a fourth mode cannot be added without each of them being updated,
/// and nothing infers the mode from anything else.
///
/// The wire spelling is lower case everywhere — `hello`, `/api/me`,
/// `/api/health` and the room stream — as it was when this was a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Mode {
    /// Simulated people and sensors (Room Sim). The only mode in which
    /// true positions may be published (d15). Never the gateway's own
    /// mode in Phase 5: the room stream sets it per session.
    Sim,
    /// A recording played back, never live sensors: a pcap through the
    /// ingest path (`--replay`; it may be a Room Sim export, which the
    /// gateway cannot tell from a capture), or a recorded Locate run on
    /// the room stream (spec/livedemo.md §6).
    Replay,
    /// Sensor input as it arrives.
    Live,
}

impl Mode {
    /// The wire spelling, identical to the serde one.
    pub fn as_str(self) -> &'static str {
        match self {
            Mode::Sim => "sim",
            Mode::Replay => "replay",
            Mode::Live => "live",
        }
    }
}

pub struct Shared {
    /// The gateway's own mode: `Live`, or `Replay` under `--replay`
    /// (spec/livedemo.md §1).
    pub mode: Mode,
    pub started: Instant,
    pub published: RwLock<Published>,
    events: Mutex<VecDeque<EventEntry>>,
    pub live_tx: broadcast::Sender<Arc<str>>,
    pub ws_clients: AtomicUsize,
    /// Bounded time series for `/api/history`, appended by the hub on
    /// publish and read by the HTTP side. In memory only.
    pub history: Mutex<History>,
    /// Fan-out to MQTT and alert routing (crate::outputs).
    pub outputs: broadcast::Sender<OutputMsg>,
    /// The room stream's handle (spec/livedemo.md §2.5): its session
    /// counter and the recorded runs `--room-replay` loaded. Opaque: no
    /// room frame, record or position is reachable through it, and none
    /// travels on `live_tx` or `outputs` or is kept in `published`.
    #[cfg(feature = "roomsim")]
    pub room: crate::roomstream::RoomStreams,
}

impl Shared {
    pub fn new(mode: Mode) -> Arc<Self> {
        Self::with_history(mode, HistoryConfig::default())
    }

    pub fn with_history(mode: Mode, history: HistoryConfig) -> Arc<Self> {
        let (live_tx, _) = broadcast::channel(512);
        let (outputs, _) = broadcast::channel(OUTPUT_CAPACITY);
        Arc::new(Self {
            mode,
            started: Instant::now(),
            published: RwLock::new(Published::default()),
            events: Mutex::new(VecDeque::new()),
            live_tx,
            ws_clients: AtomicUsize::new(0),
            history: Mutex::new(History::new(history)),
            outputs,
            #[cfg(feature = "roomsim")]
            room: crate::roomstream::RoomStreams::new(),
        })
    }

    pub fn read(&self) -> std::sync::RwLockReadGuard<'_, Published> {
        self.published.read().unwrap_or_else(|p| p.into_inner())
    }

    pub fn write(&self) -> std::sync::RwLockWriteGuard<'_, Published> {
        self.published.write().unwrap_or_else(|p| p.into_inner())
    }

    pub fn history(&self) -> std::sync::MutexGuard<'_, History> {
        self.history.lock().unwrap_or_else(|p| p.into_inner())
    }

    pub fn ws_clients(&self) -> usize {
        self.ws_clients.load(Ordering::Relaxed)
    }

    /// Record an event (already redacted, with `time` and `unix_ms`).
    pub fn push_event(&self, space_id: &str, unix_ms: i64, value: Value) -> Arc<Value> {
        let value = Arc::new(value);
        let mut ring = self.events.lock().unwrap_or_else(|p| p.into_inner());
        if ring.len() == EVENT_RING {
            ring.pop_front();
        }
        ring.push_back(EventEntry { space_id: space_id.to_string(), unix_ms, value: value.clone() });
        drop(ring);
        let _ = self.outputs.send(OutputMsg::Event { space_id: space_id.to_string(), unix_ms, value: value.clone() });
        value
    }

    /// Up to `limit` most recent events matching the filters, oldest first.
    pub fn query_events(&self, space: Option<&str>, since_ms: Option<i64>, limit: usize) -> Vec<Arc<Value>> {
        let ring = self.events.lock().unwrap_or_else(|p| p.into_inner());
        let mut out: Vec<Arc<Value>> = ring
            .iter()
            .rev()
            .filter(|e| space.is_none_or(|s| e.space_id == s) && since_ms.is_none_or(|t| e.unix_ms > t))
            .take(limit)
            .map(|e| e.value.clone())
            .collect();
        out.reverse();
        out
    }

    pub fn emit(&self, msg: OutputMsg) {
        // No receivers is fine: MQTT and alerting are optional.
        let _ = self.outputs.send(msg);
    }

    pub fn broadcast(&self, msg: Arc<str>) {
        // No receivers is fine: nobody has the dashboard open.
        let _ = self.live_tx.send(msg);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every mode, once. A fourth mode fails to compile in `index`, which
    /// is the reminder to add it here; `every_mode_is_listed_once` fails
    /// until it is, because serde's own variant list then has four.
    const ALL: [Mode; 3] = [Mode::Sim, Mode::Replay, Mode::Live];

    /// Every variant's wire name, from serde itself: deserialising an
    /// unknown name fails with "expected one of `sim`, ..." naming each.
    fn serde_variants() -> Vec<String> {
        let e = serde_json::from_value::<Mode>("?".into()).unwrap_err().to_string();
        let (_, list) = e.split_once(", expected ").unwrap_or_else(|| panic!("no variant list in {e:?}"));
        list.split('`').skip(1).step_by(2).map(String::from).collect()
    }

    fn index(m: Mode) -> usize {
        match m {
            Mode::Sim => 0,
            Mode::Replay => 1,
            Mode::Live => 2,
        }
    }

    #[test]
    fn every_mode_is_listed_once() {
        for (i, m) in ALL.into_iter().enumerate() {
            assert_eq!(index(m), i, "{m:?}");
        }
        assert_eq!(serde_variants(), ALL.map(Mode::as_str), "ALL lists every variant");
    }

    #[test]
    fn mode_serialises_and_parses_in_lower_case() {
        let wire: Vec<String> = ALL.iter().map(|m| serde_json::to_string(m).unwrap()).collect();
        assert_eq!(wire, [r#""sim""#, r#""replay""#, r#""live""#]);
        for m in ALL {
            assert_eq!(serde_json::to_value(m).unwrap(), m.as_str(), "serde and as_str agree");
            assert_eq!(serde_json::from_value::<Mode>(m.as_str().into()).unwrap(), m);
        }
        for bad in ["SIM", "Live", "REPLAY", "unknown", ""] {
            assert!(serde_json::from_value::<Mode>(bad.into()).is_err(), "{bad:?} is not a mode");
        }
    }

    #[test]
    fn shared_keeps_the_mode_it_is_given() {
        for m in ALL {
            assert_eq!(Shared::new(m).mode, m);
        }
    }
}
