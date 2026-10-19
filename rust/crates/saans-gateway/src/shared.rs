//! What the core publishes for the HTTP side to read. The core task owns
//! the engine outright; after every tick it replaces the published
//! snapshot and pushes WebSocket messages, so a slow HTTP client can never
//! stall ingest and handlers never see a half-updated space.

use crate::history::{History, HistoryConfig};
use crate::outputs::{OutputMsg, OUTPUT_CAPACITY};
use serde_json::Value;
use std::collections::VecDeque;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, RwLock};
use std::time::Instant;
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

pub struct Shared {
    pub mode: &'static str,
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
}

impl Shared {
    pub fn new(mode: &'static str) -> Arc<Self> {
        Self::with_history(mode, HistoryConfig::default())
    }

    pub fn with_history(mode: &'static str, history: HistoryConfig) -> Arc<Self> {
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
        ring.push_back(EventEntry {
            space_id: space_id.to_string(),
            unix_ms,
            value: value.clone(),
        });
        drop(ring);
        let _ = self.outputs.send(OutputMsg::Event {
            space_id: space_id.to_string(),
            unix_ms,
            value: value.clone(),
        });
        value
    }

    /// Up to `limit` most recent events matching the filters, oldest first.
    pub fn query_events(
        &self,
        space: Option<&str>,
        since_ms: Option<i64>,
        limit: usize,
    ) -> Vec<Arc<Value>> {
        let ring = self.events.lock().unwrap_or_else(|p| p.into_inner());
        let mut out: Vec<Arc<Value>> = ring
            .iter()
            .rev()
            .filter(|e| {
                space.is_none_or(|s| e.space_id == s) && since_ms.is_none_or(|t| e.unix_ms > t)
            })
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
