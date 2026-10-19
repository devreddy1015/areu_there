//! MQTT publisher (SAANS_PLAN.md §10.2). OWNER: outputs agent.
//!
//! Contract: [`spawn`] is called by `live.rs` when `[mqtt]` is configured.

use crate::config::GatewayConfig;
use crate::outputs::OutputMsg;
use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;

pub fn spawn(
    _cfg: &GatewayConfig,
    _rx: broadcast::Receiver<OutputMsg>,
    _stop: watch::Receiver<bool>,
) -> JoinHandle<()> {
    todo!("outputs agent")
}
