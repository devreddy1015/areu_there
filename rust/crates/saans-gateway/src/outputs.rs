//! The fan-out every non-HTTP output consumes (MQTT, alert routing).
//!
//! The core task sends here after the privacy gate, so a consumer can never
//! receive a value the dashboard would not show; it may only show less.
//! A broadcast channel: a slow consumer lags and loses messages (counted)
//! instead of stalling ingest.

use serde_json::Value;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub enum OutputMsg {
    /// One redacted event record, as stored in the event log.
    Event { space_id: String, unix_ms: i64, value: Arc<Value> },
    /// The redacted state of every space, published each 0.5 s tick, in
    /// config order: (space id, state JSON).
    States { unix_ms: i64, spaces: Arc<Vec<(String, Value)>> },
    /// Link and gateway health, published each tick.
    System { unix_ms: i64, links: Arc<Value>, health: Arc<Value> },
}

/// Channel capacity: about a minute of ticks with headroom for bursts.
pub const OUTPUT_CAPACITY: usize = 1024;
