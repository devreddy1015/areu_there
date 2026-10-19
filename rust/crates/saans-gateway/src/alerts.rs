//! Alert routing: events in, messages out, with acknowledgement and
//! escalation (SAANS_PLAN.md §10.3). OWNER: outputs agent.
//!
//! Contract used by the HTTP layer (do not change signatures without
//! telling the lead):
//! - [`AlertService::new`], [`AlertService::spawn`]
//! - [`AlertService::list`], [`AlertService::get`], [`AlertService::ack`],
//!   [`AlertService::resolve`]
//! - [`Alert`] (serialized as-is by `/api/alerts`)

use crate::config::GatewayConfig;
use crate::outputs::OutputMsg;
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AlertState {
    Open,
    Acknowledged,
    Resolved,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Alert {
    pub id: u64,
    pub space_id: String,
    /// Event type that raised it, e.g. "inactivity".
    pub kind: String,
    pub severity: String,
    pub route: String,
    pub state: AlertState,
    pub opened_unix_ms: i64,
    /// The redacted event that raised it (as in the event log).
    pub event: serde_json::Value,
    /// Further events of the same kind while it was open.
    pub repeats: u32,
    pub acked_by: Option<String>,
    pub acked_unix_ms: Option<i64>,
    pub resolved_by: Option<String>,
    pub resolved_unix_ms: Option<i64>,
    pub resolution_note: Option<String>,
    /// 0 = first channels; n = escalated n times.
    pub escalation_level: u32,
    /// "telegram:wardens" etc., with delivery outcome.
    pub deliveries: Vec<Delivery>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Delivery {
    pub channel: String,
    pub unix_ms: i64,
    pub ok: bool,
    pub detail: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AlertError {
    NotFound,
    /// e.g. acknowledging a resolved alert.
    InvalidTransition(&'static str),
}

pub struct AlertService {
    _private: (),
}

impl AlertService {
    /// Load persisted alert state from `data_dir/alerts.jsonl`.
    pub fn new(_cfg: &GatewayConfig, _data_dir: Option<&Path>) -> std::io::Result<Arc<Self>> {
        todo!("outputs agent")
    }

    /// Consume the output fan-out; deliver, escalate, poll Telegram for
    /// button presses. Ends when `stop` turns true.
    pub fn spawn(
        self: &Arc<Self>,
        _rx: broadcast::Receiver<OutputMsg>,
        _stop: watch::Receiver<bool>,
    ) -> JoinHandle<()> {
        todo!("outputs agent")
    }

    /// Newest first. `open_only` = Open or Acknowledged.
    pub fn list(&self, _open_only: bool, _limit: usize) -> Vec<Alert> {
        todo!("outputs agent")
    }

    pub fn get(&self, _id: u64) -> Option<Alert> {
        todo!("outputs agent")
    }

    /// `by` is the principal or Telegram user who acknowledged.
    pub fn ack(&self, _id: u64, _by: &str, _unix_ms: i64) -> Result<Alert, AlertError> {
        todo!("outputs agent")
    }

    pub fn resolve(
        &self,
        _id: u64,
        _by: &str,
        _note: &str,
        _unix_ms: i64,
    ) -> Result<Alert, AlertError> {
        todo!("outputs agent")
    }
}
