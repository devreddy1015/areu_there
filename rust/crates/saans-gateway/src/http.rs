//! HTTP API, WebSocket and the embedded dashboard.
//!
//! | Route | Returns |
//! |---|---|
//! | `GET /` | the dashboard (one self-contained HTML file, no external URLs) |
//! | `GET /api/spaces` | `[SpaceState]` (+ name, privacy badges, alerts) |
//! | `GET /api/spaces/{id}` | `{"state": SpaceState, "events": [last 50]}` |
//! | `GET /api/events?space=&since=&limit=` | events, oldest first |
//! | `GET /api/links` | `[{"space_id", "name", "links": [LinkReport]}]` |
//! | `GET /api/health` | gateway self-health |
//! | `GET /metrics` | Prometheus text format |
//! | `GET /api/live` | WebSocket: `{"type":"state"}` every 0.5 s, `{"type":"event"}` as they happen |
//!
//! Everything served here was redacted by the hub (`privacy`) before it
//! was published; handlers only copy published snapshots.

use crate::shared::Shared;
use crate::timefmt::parse_iso8601;
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::get;
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::{broadcast, watch};

pub const DASHBOARD_HTML: &str = include_str!("../assets/dashboard.html");
const CSP: &str = "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; \
                   connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

/// Everything the HTTP side needs. Built by `live.rs` / `main.rs` via
/// [`HttpDeps::new`]; the auth layer may add private fields here.
#[derive(Clone)]
pub struct HttpDeps {
    pub shared: Arc<Shared>,
    pub stop: watch::Receiver<bool>,
    pub cfg: Arc<crate::config::GatewayConfig>,
    /// `None` when no `[[alert_route]]` is configured.
    pub alerts: Option<Arc<crate::alerts::AlertService>>,
}

impl HttpDeps {
    pub fn new(
        shared: Arc<Shared>,
        stop: watch::Receiver<bool>,
        cfg: Arc<crate::config::GatewayConfig>,
        alerts: Option<Arc<crate::alerts::AlertService>>,
    ) -> Self {
        Self { shared, stop, cfg, alerts }
    }
}

#[derive(Clone)]
pub struct AppState {
    pub shared: Arc<Shared>,
    pub stop: watch::Receiver<bool>,
    pub cfg: Arc<crate::config::GatewayConfig>,
    pub alerts: Option<Arc<crate::alerts::AlertService>>,
}

pub fn router(deps: HttpDeps) -> Router {
    let HttpDeps { shared, stop, cfg, alerts } = deps;
    Router::new()
        .route("/", get(dashboard))
        .route("/api/spaces", get(spaces))
        .route("/api/spaces/{id}", get(space))
        .route("/api/events", get(events))
        .route("/api/links", get(links))
        .route("/api/health", get(health))
        .route("/metrics", get(metrics))
        .route("/api/live", get(live))
        .fallback(|| async { error(StatusCode::NOT_FOUND, "no such endpoint") })
        .with_state(AppState { shared, stop, cfg, alerts })
}

fn error(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({ "error": msg }))).into_response()
}

fn no_store(mut r: Response) -> Response {
    r.headers_mut().insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    r
}

async fn dashboard() -> Response {
    let mut r = DASHBOARD_HTML.into_response();
    let h = r.headers_mut();
    h.insert(header::CONTENT_TYPE, HeaderValue::from_static("text/html; charset=utf-8"));
    h.insert(header::CONTENT_SECURITY_POLICY, HeaderValue::from_static(CSP));
    h.insert(header::X_CONTENT_TYPE_OPTIONS, HeaderValue::from_static("nosniff"));
    h.insert(header::REFERRER_POLICY, HeaderValue::from_static("no-referrer"));
    no_store(r)
}

async fn spaces(State(app): State<AppState>) -> Response {
    let p = app.shared.read();
    let list: Vec<&Value> = p.spaces.iter().map(|(_, v)| v).collect();
    no_store(Json(list).into_response())
}

async fn space(State(app): State<AppState>, Path(id): Path<String>) -> Response {
    let state = {
        let p = app.shared.read();
        p.spaces.iter().find(|(sid, _)| *sid == id).map(|(_, v)| v.clone())
    };
    match state {
        Some(state) => {
            let events = app.shared.query_events(Some(&id), None, 50);
            let events: Vec<&Value> = events.iter().map(|e| e.as_ref()).collect();
            no_store(Json(json!({ "state": state, "events": events })).into_response())
        }
        None => error(StatusCode::NOT_FOUND, "no such space"),
    }
}

#[derive(Debug, Deserialize)]
pub struct EventsQuery {
    pub space: Option<String>,
    /// Unix seconds (may be fractional) or ISO 8601; events strictly after.
    pub since: Option<String>,
    pub limit: Option<usize>,
}

/// `since` as Unix milliseconds.
pub fn parse_since(s: &str) -> Option<i64> {
    let s = s.trim();
    if let Ok(x) = s.parse::<f64>() {
        return x.is_finite().then(|| (x * 1000.0).floor() as i64);
    }
    parse_iso8601(s)
}

async fn events(State(app): State<AppState>, Query(q): Query<EventsQuery>) -> Response {
    let since = match q.since.as_deref().filter(|s| !s.is_empty()) {
        None => None,
        Some(s) => match parse_since(s) {
            Some(ms) => Some(ms),
            None => return error(StatusCode::BAD_REQUEST, "since must be Unix seconds or ISO 8601 (2026-10-13T08:00:00+05:30)"),
        },
    };
    let limit = q.limit.unwrap_or(100).clamp(1, 1000);
    let space = q.space.as_deref().filter(|s| !s.is_empty());
    let found = app.shared.query_events(space, since, limit);
    let found: Vec<&Value> = found.iter().map(|e| e.as_ref()).collect();
    no_store(Json(found).into_response())
}

async fn links(State(app): State<AppState>) -> Response {
    no_store(Json(app.shared.read().links.clone()).into_response())
}

async fn health(State(app): State<AppState>) -> Response {
    let mut h = app.shared.read().health.clone();
    if let Value::Object(m) = &mut h {
        // Fresh values; the rest is as of the last tick.
        m.insert("uptime_s".into(), app.shared.started.elapsed().as_secs_f64().into());
        m.insert("ws_clients".into(), app.shared.ws_clients().into());
    }
    no_store(Json(h).into_response())
}

async fn metrics(State(app): State<AppState>) -> Response {
    let body = app.shared.read().metrics.clone();
    let mut r = body.into_response();
    r.headers_mut().insert(header::CONTENT_TYPE, HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"));
    no_store(r)
}

async fn live(ws: WebSocketUpgrade, State(app): State<AppState>) -> Response {
    ws.on_upgrade(move |socket| ws_session(socket, app))
}

async fn ws_session(mut socket: WebSocket, app: AppState) {
    let shared = app.shared;
    let mut rx = shared.live_tx.subscribe();
    shared.ws_clients.fetch_add(1, Ordering::Relaxed);
    let hello = json!({"type": "hello", "version": env!("CARGO_PKG_VERSION"), "mode": shared.mode}).to_string();
    let first = shared.read().state_msg.clone();
    let mut ok = socket.send(Message::Text(hello.into())).await.is_ok();
    if let (true, Some(state)) = (ok, first) {
        ok = socket.send(Message::Text(state.as_ref().into())).await.is_ok();
    }
    let mut stop = app.stop;
    while ok {
        tokio::select! {
            msg = rx.recv() => match msg {
                Ok(text) => ok = socket.send(Message::Text(text.as_ref().into())).await.is_ok(),
                // A slow client misses messages rather than slowing the
                // gateway; the next state message resynchronises it.
                Err(broadcast::error::RecvError::Lagged(_)) => {}
                Err(broadcast::error::RecvError::Closed) => break,
            },
            incoming = socket.recv() => match incoming {
                Some(Ok(Message::Close(_))) => {
                    // The library queues the close reply (RFC 6455 5.5.1)
                    // and sends it on the next read; without this the
                    // client sees an abnormal closure (1006).
                    let _ = tokio::time::timeout(std::time::Duration::from_secs(1), socket.recv()).await;
                    break;
                }
                None | Some(Err(_)) => break,
                Some(Ok(_)) => {}
            },
            _ = stop.changed() => {
                let _ = socket.send(Message::Close(None)).await;
                break;
            }
        }
    }
    shared.ws_clients.fetch_sub(1, Ordering::Relaxed);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn since_accepts_unix_seconds_and_iso() {
        assert_eq!(parse_since("1700000000.5"), Some(1_700_000_000_500));
        assert_eq!(parse_since("1970-01-01T00:00:01Z"), Some(1000));
        assert_eq!(parse_since("nope"), None);
    }

    #[test]
    fn dashboard_has_no_external_urls() {
        for bad in ["http://", "https://", "//cdn", "src=\"//", "@import", "url(http"] {
            assert!(!DASHBOARD_HTML.contains(bad), "dashboard must stay offline-capable: found {bad:?}");
        }
        assert!(DASHBOARD_HTML.contains("not a medical device"));
    }
}
