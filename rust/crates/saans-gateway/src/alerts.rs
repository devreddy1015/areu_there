//! Alert routing: events in, messages out, with acknowledgement and
//! escalation (SAANS_PLAN.md §10.3).
//!
//! Contract used by the HTTP layer:
//! - [`AlertService::new`], [`AlertService::spawn`]
//! - [`AlertService::list`], [`AlertService::get`], [`AlertService::ack`],
//!   [`AlertService::resolve`]
//! - [`Alert`] (serialized as-is by `/api/alerts`)
//!
//! # What it does, and the rules it keeps
//!
//! - **Routing.** `[[alert_route]]`s are tried in config order and the first
//!   match owns the alert, like a firewall rule list. A route matches a
//!   space when its `spaces` is empty or names it, and an event when its
//!   `kinds` names the event's type -- or, with `kinds` empty, when the
//!   severity is at least `min_severity`. An `info` event never pages
//!   anyone, whatever a route says: `Severity::Info` is "recorded, shown,
//!   never pushed".
//! - **One open alert per (space, kind).** While an alert is open or
//!   acknowledged, a repeat of the same kind in the same space updates it
//!   (`repeats`, and the latest event) and pages nobody. After it is
//!   resolved, the next event opens a new alert.
//! - **States** open -> acknowledged -> resolved; open -> resolved is
//!   allowed (someone checked and it was nothing). Every change is a line
//!   in `alerts.jsonl`, and the file is read back at startup.
//! - **Escalation**, once: an alert still *open* -- not acknowledged -- after
//!   its route's `ack_timeout_min` goes to the route's `escalate_to`. Driven
//!   by the gateway's own clock (`OutputMsg::States`), so it follows a
//!   replay's clock and can be tested with a fake one.
//! - **Leaving the campus.** A message to Telegram or a webhook carries the
//!   alert's kind, space and state and nothing else -- no `evidence`
//!   sentence, no `confidence`, none of an event's numbers -- unless the
//!   route sets `include_health_details`. That is the privacy boundary of
//!   the alert path, and `outbound` is the one function that crosses it.
//! - **Secrets.** Tokens and webhook keys come from environment variables
//!   named in the config. A Telegram request URL *contains the bot token*,
//!   and `reqwest`'s errors print their URL, so every error is passed
//!   through `without_url()` before it reaches a log or a stored delivery.
//! - **Retention.** An alert embeds its event, so a resolved alert is
//!   dropped once it is older than `event_retention_days` -- the event
//!   log's clock. Open and acknowledged alerts are never dropped.

use crate::audit::{Action, AuditLog};
use crate::config::{AlertRoute, GatewayConfig};
use crate::outputs::OutputMsg;
use crate::timefmt::now_unix_ms;
use hmac::{Hmac, Mac};
use saans_health::Severity;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::Sha256;
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::{Duration, Instant};
use tokio::sync::broadcast::error::RecvError;
use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;

/// Alert state, one JSON line per change, under `data_dir`.
pub const ALERTS_FILE: &str = "alerts.jsonl";
/// How often resolved alerts past retention are swept.
const PRUNE_EVERY_MS: i64 = 3_600_000;
const SEND_TIMEOUT: Duration = Duration::from_secs(10);
/// Telegram long-poll hold, seconds. The server answers within this.
const POLL_TIMEOUT_S: u64 = 25;
const DAY_MS: i64 = 86_400_000;

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

impl Alert {
    fn is_live(&self) -> bool {
        self.state != AlertState::Resolved
    }
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

impl std::fmt::Display for AlertError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AlertError::NotFound => write!(f, "no such alert"),
            AlertError::InvalidTransition(why) => write!(f, "{why}"),
        }
    }
}

/// One message to one channel about one alert, already redacted for that
/// route. Built only by [`AlertService::outbound`].
#[derive(Debug, Clone)]
pub struct Outbound {
    pub alert_id: u64,
    pub channel: String,
    pub text: String,
    pub payload: Value,
}

struct Telegram {
    chat_id: i64,
    api_base: String,
    token: String,
    allowed_user_ids: Vec<i64>,
}

struct Webhook {
    url: String,
    secret: Option<Vec<u8>>,
}

#[derive(Default)]
struct Inner {
    alerts: BTreeMap<u64, Alert>,
    /// (space, kind) -> the open or acknowledged alert for it.
    live: HashMap<(String, String), u64>,
    next_id: u64,
    file: Option<File>,
    write_errors: u64,
    last_error: Option<String>,
    load_skipped: u64,
    deliveries_ok: u64,
    deliveries_failed: u64,
    lagged: u64,
    last_prune_ms: Option<i64>,
}

pub struct AlertService {
    routes: Vec<AlertRoute>,
    telegram: BTreeMap<String, Telegram>,
    webhook: BTreeMap<String, Webhook>,
    space_names: BTreeMap<String, String>,
    retention_days: u32,
    path: Option<PathBuf>,
    http: reqwest::Client,
    audit: OnceLock<Arc<AuditLog>>,
    inner: Mutex<Inner>,
}

fn env_secret(what: &str, var: &str) -> io::Result<String> {
    match std::env::var(var) {
        Ok(v) if !v.is_empty() => Ok(v),
        _ => Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            format!("{what}: environment variable {var} is not set"),
        )),
    }
}

impl AlertService {
    /// Load persisted alert state from `data_dir/alerts.jsonl`.
    ///
    /// Fails if a channel's token or key is not in the environment: a
    /// gateway that would start and then fail to page anyone is the worse
    /// outcome, and the config loader cannot see the environment.
    pub fn new(cfg: &GatewayConfig, data_dir: Option<&Path>) -> io::Result<Arc<Self>> {
        let mut telegram = BTreeMap::new();
        for (name, t) in &cfg.telegram {
            telegram.insert(
                name.clone(),
                Telegram {
                    chat_id: t.chat_id,
                    api_base: t.api_base.trim_end_matches('/').to_string(),
                    token: env_secret(&format!("telegram.{name}"), &t.bot_token_env)?,
                    allowed_user_ids: t.allowed_user_ids.clone(),
                },
            );
        }
        let mut webhook = BTreeMap::new();
        for (name, w) in &cfg.webhook {
            let secret = match &w.secret_env {
                Some(var) => Some(env_secret(&format!("webhook.{name}"), var)?.into_bytes()),
                None => None,
            };
            webhook.insert(
                name.clone(),
                Webhook {
                    url: w.url.clone(),
                    secret,
                },
            );
        }
        let http = reqwest::Client::builder()
            .build()
            .map_err(|e| io::Error::other(e.without_url()))?;
        let path = match data_dir {
            Some(d) if !d.as_os_str().is_empty() => {
                fs::create_dir_all(d)?;
                Some(d.join(ALERTS_FILE))
            }
            _ => None,
        };
        let svc = Self {
            routes: cfg.alert_routes.clone(),
            telegram,
            webhook,
            space_names: cfg
                .spaces
                .iter()
                .map(|s| {
                    (
                        s.id.clone(),
                        if s.name.is_empty() {
                            s.id.clone()
                        } else {
                            s.name.clone()
                        },
                    )
                })
                .collect(),
            retention_days: cfg.event_retention_days,
            path,
            http,
            audit: OnceLock::new(),
            inner: Mutex::new(Inner {
                next_id: 1,
                ..Default::default()
            }),
        };
        svc.load(now_unix_ms())?;
        Ok(Arc::new(svc))
    }

    /// Telegram button presses are acknowledgements too, and every one
    /// belongs in the audit log. The HTTP layer owns the log; it hands it
    /// over here. Called once; later calls are ignored.
    pub fn attach_audit(&self, audit: Arc<AuditLog>) {
        let _ = self.audit.set(audit);
    }

    fn lock(&self) -> std::sync::MutexGuard<'_, Inner> {
        self.inner.lock().unwrap_or_else(|p| p.into_inner())
    }

    // ---------------------------------------------------------- persistence

    fn load(&self, now_ms: i64) -> io::Result<()> {
        let mut g = self.lock();
        let Some(path) = &self.path else {
            return Ok(());
        };
        if let Ok(f) = File::open(path) {
            for line in BufReader::new(f).lines().map_while(Result::ok) {
                if line.trim().is_empty() {
                    continue;
                }
                match serde_json::from_str::<Alert>(&line) {
                    // Later lines are later states of the same alert.
                    Ok(a) => {
                        g.alerts.insert(a.id, a);
                    }
                    Err(_) => g.load_skipped += 1,
                }
            }
        }
        g.next_id = g.alerts.keys().next_back().map_or(1, |id| id + 1);
        self.prune_locked(&mut g, now_ms);
        let live: Vec<_> = g
            .alerts
            .values()
            .filter(|a| a.is_live())
            .map(|a| ((a.space_id.clone(), a.kind.clone()), a.id))
            .collect();
        g.live = live.into_iter().collect();
        self.compact_locked(&mut g)
    }

    /// Rewrite the file as one line per alert, atomically, and reopen it
    /// for appending.
    fn compact_locked(&self, g: &mut Inner) -> io::Result<()> {
        let Some(path) = &self.path else {
            return Ok(());
        };
        g.file = None;
        let tmp = path.with_extension("jsonl.tmp");
        {
            let mut f = File::create(&tmp)?;
            for a in g.alerts.values() {
                let line = serde_json::to_string(a).map_err(io::Error::other)?;
                f.write_all(line.as_bytes())?;
                f.write_all(b"\n")?;
            }
            f.sync_all()?;
        }
        fs::rename(&tmp, path)?;
        g.file = Some(OpenOptions::new().append(true).open(path)?);
        Ok(())
    }

    fn persist(g: &mut Inner, id: u64) {
        let Some(a) = g.alerts.get(&id) else { return };
        let Ok(line) = serde_json::to_string(a) else {
            return;
        };
        let Some(f) = g.file.as_mut() else { return };
        // Alerts are rare and an acknowledgement lost to a crash is a page
        // that will be repeated to the wrong person, so each line is
        // synced, not merely written.
        let r = f
            .write_all(line.as_bytes())
            .and_then(|_| f.write_all(b"\n"))
            .and_then(|_| f.sync_data());
        if let Err(e) = r {
            g.write_errors += 1;
            g.last_error = Some(e.to_string());
        }
    }

    /// Drop resolved alerts whose resolution is older than the event
    /// retention. Returns how many went.
    fn prune_locked(&self, g: &mut Inner, now_ms: i64) -> usize {
        let cutoff = now_ms - self.retention_days as i64 * DAY_MS;
        let before = g.alerts.len();
        g.alerts
            .retain(|_, a| a.is_live() || a.resolved_unix_ms.unwrap_or(a.opened_unix_ms) >= cutoff);
        g.last_prune_ms = Some(now_ms);
        before - g.alerts.len()
    }

    // ------------------------------------------------------------- routing

    fn route_for(&self, space: &str, kind: &str, sev: Severity) -> Option<&AlertRoute> {
        if sev == Severity::Info {
            return None;
        }
        self.routes.iter().find(|r| {
            (r.spaces.is_empty() || r.spaces.iter().any(|s| s == space))
                && if r.kinds.is_empty() {
                    sev >= r.min_severity
                } else {
                    r.kinds.iter().any(|k| k == kind)
                }
        })
    }

    fn route_named(&self, name: &str) -> Option<&AlertRoute> {
        self.routes.iter().find(|r| r.name == name)
    }

    /// One redacted event in; the messages it causes out. Pure apart from
    /// the state change and its line in the file: delivery is the caller's.
    pub fn on_event(&self, event: &Value, unix_ms: i64) -> Vec<Outbound> {
        let (Some(space), Some(kind)) = (
            event.get("space_id").and_then(Value::as_str),
            event.get("type").and_then(Value::as_str),
        ) else {
            return Vec::new();
        };
        let Ok(sev) = serde_json::from_value::<Severity>(
            event.get("severity").cloned().unwrap_or(Value::Null),
        ) else {
            return Vec::new();
        };
        let Some(route) = self.route_for(space, kind, sev) else {
            return Vec::new();
        };
        let mut g = self.lock();
        let key = (space.to_string(), kind.to_string());
        if let Some(&id) = g.live.get(&key) {
            if let Some(a) = g.alerts.get_mut(&id) {
                a.repeats += 1;
                a.event = event.clone();
                Self::persist(&mut g, id);
                return Vec::new();
            }
        }
        let id = g.next_id;
        g.next_id += 1;
        let alert = Alert {
            id,
            space_id: space.to_string(),
            kind: kind.to_string(),
            severity: event
                .get("severity")
                .and_then(Value::as_str)
                .unwrap_or("check")
                .to_string(),
            route: route.name.clone(),
            state: AlertState::Open,
            opened_unix_ms: unix_ms,
            event: event.clone(),
            repeats: 0,
            acked_by: None,
            acked_unix_ms: None,
            resolved_by: None,
            resolved_unix_ms: None,
            resolution_note: None,
            escalation_level: 0,
            deliveries: Vec::new(),
        };
        let out = route
            .channels
            .iter()
            .map(|c| self.outbound(&alert, route, c, "opened"))
            .collect();
        g.alerts.insert(id, alert);
        g.live.insert(key, id);
        Self::persist(&mut g, id);
        out
    }

    /// The gateway clock moved to `now_ms`: escalate what has waited too
    /// long, and sweep what has outlived retention.
    pub fn tick(&self, now_ms: i64) -> Vec<Outbound> {
        let mut out = Vec::new();
        let mut g = self.lock();
        let due: Vec<u64> = g
            .alerts
            .values()
            .filter(|a| a.state == AlertState::Open && a.escalation_level == 0)
            .filter(|a| {
                self.route_named(&a.route).is_some_and(|r| {
                    r.ack_timeout_min > 0.0
                        && !r.escalate_to.is_empty()
                        && (now_ms - a.opened_unix_ms) as f64 >= r.ack_timeout_min * 60_000.0
                })
            })
            .map(|a| a.id)
            .collect();
        for id in due {
            let Some(a) = g.alerts.get_mut(&id) else {
                continue;
            };
            a.escalation_level = 1;
            let a = a.clone();
            if let Some(r) = self.route_named(&a.route) {
                let why = format!(
                    "not acknowledged within {} min, escalated",
                    r.ack_timeout_min
                );
                out.extend(r.escalate_to.iter().map(|c| self.outbound(&a, r, c, &why)));
            }
            Self::persist(&mut g, id);
        }
        if g.last_prune_ms.is_none_or(|t| now_ms - t >= PRUNE_EVERY_MS)
            && self.prune_locked(&mut g, now_ms) > 0
        {
            if let Err(e) = self.compact_locked(&mut g) {
                g.write_errors += 1;
                g.last_error = Some(e.to_string());
            }
        }
        out
    }

    /// THE privacy boundary of the alert path: what leaves the building.
    /// Kind, space and state always; evidence, confidence and the event's
    /// numbers only when the route sets `include_health_details`.
    fn outbound(&self, a: &Alert, route: &AlertRoute, channel: &str, why: &str) -> Outbound {
        let space_name = self
            .space_names
            .get(&a.space_id)
            .cloned()
            .unwrap_or_else(|| a.space_id.clone());
        let words = a.kind.replace('_', " ");
        let mut text = format!(
            "Saans alert #{}: {words} in {space_name} ({}), {why}. \
             Advisory only -- check the space and follow your usual procedure.",
            a.id, a.severity
        );
        let mut payload = json!({
            "alert_id": a.id,
            "space_id": a.space_id,
            "space_name": space_name,
            "kind": a.kind,
            "severity": a.severity,
            "state": a.state,
            "opened_unix_ms": a.opened_unix_ms,
            "escalation_level": a.escalation_level,
            "reason": why,
            "advisory": true,
        });
        if route.include_health_details {
            if let Some(ev) = a.event.get("evidence").and_then(Value::as_str) {
                text.push('\n');
                text.push_str(ev);
            }
            payload["evidence"] = a.event.get("evidence").cloned().unwrap_or(Value::Null);
            payload["confidence"] = a.event.get("confidence").cloned().unwrap_or(Value::Null);
            payload["event"] = a.event.clone();
        }
        Outbound {
            alert_id: a.id,
            channel: channel.to_string(),
            text,
            payload,
        }
    }

    // ---------------------------------------------------------- the contract

    /// Newest first. `open_only` = Open or Acknowledged.
    pub fn list(&self, open_only: bool, limit: usize) -> Vec<Alert> {
        self.lock()
            .alerts
            .values()
            .rev()
            .filter(|a| !open_only || a.is_live())
            .take(limit)
            .cloned()
            .collect()
    }

    pub fn get(&self, id: u64) -> Option<Alert> {
        self.lock().alerts.get(&id).cloned()
    }

    /// `by` is the principal or Telegram user who acknowledged.
    pub fn ack(&self, id: u64, by: &str, unix_ms: i64) -> Result<Alert, AlertError> {
        let mut g = self.lock();
        let a = g.alerts.get_mut(&id).ok_or(AlertError::NotFound)?;
        match a.state {
            AlertState::Open => {}
            AlertState::Acknowledged => {
                return Err(AlertError::InvalidTransition("already acknowledged"))
            }
            AlertState::Resolved => return Err(AlertError::InvalidTransition("already resolved")),
        }
        a.state = AlertState::Acknowledged;
        a.acked_by = Some(by.to_string());
        a.acked_unix_ms = Some(unix_ms);
        let a = a.clone();
        Self::persist(&mut g, id);
        Ok(a)
    }

    pub fn resolve(
        &self,
        id: u64,
        by: &str,
        note: &str,
        unix_ms: i64,
    ) -> Result<Alert, AlertError> {
        let mut g = self.lock();
        let a = g.alerts.get_mut(&id).ok_or(AlertError::NotFound)?;
        if a.state == AlertState::Resolved {
            return Err(AlertError::InvalidTransition("already resolved"));
        }
        a.state = AlertState::Resolved;
        a.resolved_by = Some(by.to_string());
        a.resolved_unix_ms = Some(unix_ms);
        a.resolution_note = (!note.trim().is_empty()).then(|| note.trim().to_string());
        let a = a.clone();
        g.live.remove(&(a.space_id.clone(), a.kind.clone()));
        Self::persist(&mut g, id);
        Ok(a)
    }

    /// For `/api/health`.
    pub fn status(&self) -> Value {
        let g = self.lock();
        let count = |s: AlertState| g.alerts.values().filter(|a| a.state == s).count();
        json!({
            "routes": self.routes.len(),
            "open": count(AlertState::Open),
            "acknowledged": count(AlertState::Acknowledged),
            "resolved_kept": count(AlertState::Resolved),
            "retention_days": self.retention_days,
            "persisted": self.path.is_some(),
            "deliveries_ok": g.deliveries_ok,
            "deliveries_failed": g.deliveries_failed,
            "lagged_messages": g.lagged,
            "write_errors": g.write_errors,
            "last_error": g.last_error,
            "unreadable_lines_skipped": g.load_skipped,
        })
    }

    // ------------------------------------------------------------- delivery

    fn dispatch(self: &Arc<Self>, out: Vec<Outbound>) {
        for o in out {
            let me = self.clone();
            tokio::spawn(async move { me.deliver(o).await });
        }
    }

    async fn deliver(&self, out: Outbound) {
        let (ok, detail) = match out.channel.split_once(':') {
            Some(("webhook", n)) => self.send_webhook(n, &out).await,
            Some(("telegram", n)) => self.send_telegram(n, &out).await,
            _ => (false, "unknown channel".to_string()),
        };
        if !ok {
            crate::log(&format!(
                "alert #{}: delivery to {} failed: {detail}",
                out.alert_id, out.channel
            ));
        }
        let mut g = self.lock();
        if ok {
            g.deliveries_ok += 1;
        } else {
            g.deliveries_failed += 1;
        }
        if let Some(a) = g.alerts.get_mut(&out.alert_id) {
            a.deliveries.push(Delivery {
                channel: out.channel.clone(),
                unix_ms: now_unix_ms(),
                ok,
                detail,
            });
            Self::persist(&mut g, out.alert_id);
        }
    }

    async fn send_webhook(&self, name: &str, out: &Outbound) -> (bool, String) {
        let Some(w) = self.webhook.get(name) else {
            return (false, "no such webhook".into());
        };
        let body = match serde_json::to_vec(&out.payload) {
            Ok(b) => b,
            Err(e) => return (false, e.to_string()),
        };
        let mut req = self
            .http
            .post(&w.url)
            .timeout(SEND_TIMEOUT)
            .header(reqwest::header::CONTENT_TYPE, "application/json");
        if let Some(key) = &w.secret {
            req = req.header(
                "X-Saans-Signature",
                format!("sha256={}", hmac_hex(key, &body)),
            );
        }
        match req.body(body).send().await {
            Ok(r) if r.status().is_success() => (true, format!("HTTP {}", r.status().as_u16())),
            Ok(r) => (false, format!("HTTP {}", r.status().as_u16())),
            Err(e) => (false, e.without_url().to_string()),
        }
    }

    async fn send_telegram(&self, name: &str, out: &Outbound) -> (bool, String) {
        let Some(t) = self.telegram.get(name) else {
            return (false, "no such telegram channel".into());
        };
        let body = json!({
            "chat_id": t.chat_id,
            "text": out.text,
            "reply_markup": { "inline_keyboard": [[
                { "text": "Acknowledge", "callback_data": format!("saans:ack:{}", out.alert_id) },
                { "text": "Resolve", "callback_data": format!("saans:res:{}", out.alert_id) },
            ]]},
        });
        let url = format!("{}/bot{}/sendMessage", t.api_base, t.token);
        match self
            .http
            .post(url)
            .timeout(SEND_TIMEOUT)
            .json(&body)
            .send()
            .await
        {
            Ok(r) => {
                let code = r.status().as_u16();
                let accepted = r
                    .json::<Value>()
                    .await
                    .ok()
                    .and_then(|v| v.get("ok").and_then(Value::as_bool))
                    .unwrap_or(false);
                (
                    accepted,
                    format!(
                        "HTTP {code}{}",
                        if accepted { "" } else { ", not accepted" }
                    ),
                )
            }
            Err(e) => (false, e.without_url().to_string()),
        }
    }

    // ------------------------------------------------- Telegram button presses

    /// Long-poll one Telegram bot for Acknowledge / Resolve presses. No
    /// inbound port: the gateway asks Telegram, Telegram never calls in.
    async fn poll_telegram(self: Arc<Self>, name: String, mut stop: watch::Receiver<bool>) {
        let Some(t) = self.telegram.get(&name) else {
            return;
        };
        let url = format!("{}/bot{}/getUpdates", t.api_base, t.token);
        let mut offset: i64 = 0;
        let mut backoff = Duration::from_secs(1);
        loop {
            let started = Instant::now();
            let req = self
                .http
                .post(&url)
                .timeout(Duration::from_secs(POLL_TIMEOUT_S + 10))
                .json(&json!({ "offset": offset, "timeout": POLL_TIMEOUT_S, "allowed_updates": ["callback_query"] }));
            let res = tokio::select! {
                _ = stop.wait_for(|s| *s) => return,
                r = req.send() => r,
            };
            let updates = match res {
                Ok(r) => r
                    .json::<Value>()
                    .await
                    .ok()
                    .filter(|v| v.get("ok").and_then(Value::as_bool) == Some(true))
                    .and_then(|v| v.get("result").and_then(Value::as_array).cloned()),
                Err(_) => None,
            };
            let Some(updates) = updates else {
                // Network down or Telegram unhappy: back off, never spin.
                tokio::select! {
                    _ = stop.wait_for(|s| *s) => return,
                    _ = tokio::time::sleep(backoff) => {},
                }
                backoff = (backoff * 2).min(Duration::from_secs(60));
                continue;
            };
            backoff = Duration::from_secs(1);
            for u in &updates {
                if let Some(id) = u.get("update_id").and_then(Value::as_i64) {
                    offset = offset.max(id + 1);
                }
                if let Some(cq) = u.get("callback_query") {
                    self.on_callback(&name, cq).await;
                }
            }
            // A server that answers at once with nothing would otherwise
            // turn this into a busy loop.
            if updates.is_empty() && started.elapsed() < Duration::from_secs(1) {
                tokio::select! {
                    _ = stop.wait_for(|s| *s) => return,
                    _ = tokio::time::sleep(Duration::from_secs(1)) => {},
                }
            }
        }
    }

    /// One button press. The presser must be in the configured chat, on
    /// the allowed list if there is one, and pressing a button on an alert
    /// that was actually sent to this chat.
    async fn on_callback(&self, name: &str, cq: &Value) {
        let Some(t) = self.telegram.get(name) else {
            return;
        };
        let cq_id = cq
            .get("id")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        let user = cq.pointer("/from/id").and_then(Value::as_i64).unwrap_or(0);
        let chat = cq.pointer("/message/chat/id").and_then(Value::as_i64);
        let data = cq.get("data").and_then(Value::as_str).unwrap_or("");
        let who = format!("telegram:{user}");
        let reply = if chat != Some(t.chat_id) {
            "not from this alert chat".to_string()
        } else if !t.allowed_user_ids.is_empty() && !t.allowed_user_ids.contains(&user) {
            self.audit(
                Action::Denied,
                &who,
                json!({ "via": "telegram", "channel": name, "data": data }),
            );
            "you are not allowed to act on alerts".to_string()
        } else {
            self.act_from_telegram(name, &who, data)
        };
        let url = format!("{}/bot{}/answerCallbackQuery", t.api_base, t.token);
        let _ = self
            .http
            .post(url)
            .timeout(SEND_TIMEOUT)
            .json(&json!({ "callback_query_id": cq_id, "text": reply }))
            .send()
            .await;
    }

    fn act_from_telegram(&self, channel: &str, who: &str, data: &str) -> String {
        let mut parts = data.splitn(3, ':');
        let (Some("saans"), Some(verb), Some(id)) = (parts.next(), parts.next(), parts.next())
        else {
            return "not a Saans button".into();
        };
        let Ok(id) = id.parse::<u64>() else {
            return "not a Saans button".into();
        };
        // Only alerts that were sent to this chat can be acted on from it.
        let this = format!("telegram:{channel}");
        let reached = self.get(id).is_some_and(|a| {
            self.route_named(&a.route)
                .is_some_and(|r| r.channels.contains(&this) || r.escalate_to.contains(&this))
        });
        if !reached {
            return format!("alert #{id} was not sent to this chat");
        }
        let now = now_unix_ms();
        let (action, result) = match verb {
            "ack" => (Action::AlertAck, self.ack(id, who, now)),
            "res" => (
                Action::AlertResolve,
                self.resolve(id, who, "resolved from Telegram", now),
            ),
            _ => return "not a Saans button".into(),
        };
        match result {
            Ok(a) => {
                self.audit(
                    action,
                    who,
                    json!({ "via": "telegram", "alert_id": a.id, "space_id": a.space_id, "kind": a.kind }),
                );
                format!(
                    "alert #{id} {}",
                    if verb == "ack" {
                        "acknowledged"
                    } else {
                        "resolved"
                    }
                )
            }
            Err(e) => format!("alert #{id}: {e}"),
        }
    }

    fn audit(&self, action: Action, who: &str, detail: Value) {
        if let Some(a) = self.audit.get() {
            a.record(now_unix_ms(), action, who, None, detail);
        }
    }

    // ------------------------------------------------------------ the task

    /// Consume the output fan-out; deliver, escalate, poll Telegram for
    /// button presses. Ends when `stop` turns true.
    pub fn spawn(
        self: &Arc<Self>,
        mut rx: broadcast::Receiver<OutputMsg>,
        mut stop: watch::Receiver<bool>,
    ) -> JoinHandle<()> {
        let me = self.clone();
        tokio::spawn(async move {
            let pollers: Vec<JoinHandle<()>> = me
                .telegram
                .keys()
                .map(|n| tokio::spawn(me.clone().poll_telegram(n.clone(), stop.clone())))
                .collect();
            loop {
                let msg = tokio::select! {
                    _ = stop.wait_for(|s| *s) => break,
                    m = rx.recv() => m,
                };
                match msg {
                    Ok(OutputMsg::Event { unix_ms, value, .. }) => {
                        let out = me.on_event(&value, unix_ms);
                        me.dispatch(out);
                    }
                    Ok(OutputMsg::States { unix_ms, .. }) => {
                        let out = me.tick(unix_ms);
                        me.dispatch(out);
                    }
                    Ok(_) => {}
                    // A consumer that lags loses messages rather than
                    // stalling ingest (outputs.rs). Counted, because a lost
                    // event here may be a lost page.
                    Err(RecvError::Lagged(n)) => me.lock().lagged += n,
                    Err(RecvError::Closed) => break,
                }
            }
            for p in pollers {
                p.abort();
            }
        })
    }
}

fn hmac_hex(key: &[u8], body: &[u8]) -> String {
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(key).expect("HMAC takes any key length");
    mac.update(body);
    mac.finalize()
        .into_bytes()
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::WebhookChannel;
    use saans_health::SpaceConfig;

    const T0: i64 = 1_760_000_000_000;

    fn route(name: &str) -> AlertRoute {
        AlertRoute {
            name: name.into(),
            spaces: vec![],
            kinds: vec![],
            min_severity: Severity::Check,
            channels: vec!["webhook:pager".into()],
            ack_timeout_min: 0.0,
            escalate_to: vec![],
            include_health_details: false,
        }
    }

    fn cfg(routes: Vec<AlertRoute>) -> GatewayConfig {
        let mut c = GatewayConfig {
            alert_routes: routes,
            ..GatewayConfig::default()
        };
        c.spaces = vec![SpaceConfig {
            id: "lab".into(),
            name: "Wet lab".into(),
            ..Default::default()
        }];
        for n in ["pager", "boss"] {
            c.webhook.insert(
                n.into(),
                WebhookChannel {
                    url: "http://127.0.0.1:9/never".into(),
                    secret_env: None,
                },
            );
        }
        c
    }

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("saans-alerts-{name}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        d
    }

    fn event(kind: &str, sev: &str) -> Value {
        json!({
            "space_id": "lab", "type": kind, "severity": sev, "confidence": 0.83,
            "evidence": "no gross motion for 22 min, breathing detected at 13 BPM",
            "median_bpm": 31.5, "duration_s": 27.0,
        })
    }

    #[test]
    fn open_ack_resolve_and_the_transitions_that_are_refused() {
        let s = AlertService::new(&cfg(vec![route("r")]), None).unwrap();
        assert_eq!(s.on_event(&event("inactivity", "check"), T0).len(), 1);
        let id = s.list(true, 10)[0].id;
        assert_eq!(s.get(id).unwrap().state, AlertState::Open);
        let a = s.ack(id, "asha", T0 + 1000).unwrap();
        assert_eq!(
            (a.state, a.acked_by.as_deref()),
            (AlertState::Acknowledged, Some("asha"))
        );
        assert_eq!(
            s.ack(id, "ben", T0 + 2000),
            Err(AlertError::InvalidTransition("already acknowledged"))
        );
        let a = s.resolve(id, "asha", "  was asleep  ", T0 + 3000).unwrap();
        assert_eq!(a.state, AlertState::Resolved);
        assert_eq!(a.resolution_note.as_deref(), Some("was asleep"));
        assert_eq!(
            s.resolve(id, "ben", "", T0),
            Err(AlertError::InvalidTransition("already resolved"))
        );
        assert_eq!(
            s.ack(id, "ben", T0),
            Err(AlertError::InvalidTransition("already resolved"))
        );
        assert_eq!(s.ack(999, "ben", T0), Err(AlertError::NotFound));
        // Open straight to resolved is allowed: someone checked, it was nothing.
        s.on_event(&event("inactivity", "check"), T0 + 4000);
        let id2 = s.list(true, 10)[0].id;
        assert_ne!(id2, id);
        assert_eq!(
            s.resolve(id2, "asha", "", T0 + 5000).unwrap().acked_by,
            None
        );
    }

    #[test]
    fn a_repeat_updates_the_open_alert_and_pages_nobody() {
        let s = AlertService::new(&cfg(vec![route("r")]), None).unwrap();
        assert_eq!(s.on_event(&event("breathing_pause", "check"), T0).len(), 1);
        assert!(s
            .on_event(&event("breathing_pause", "check"), T0 + 10)
            .is_empty());
        let id = s.list(true, 10)[0].id;
        assert_eq!(s.get(id).unwrap().repeats, 1);
        // Acknowledged is still live: a repeat still does not page.
        s.ack(id, "asha", T0 + 20).unwrap();
        assert!(s
            .on_event(&event("breathing_pause", "check"), T0 + 30)
            .is_empty());
        assert_eq!(s.get(id).unwrap().repeats, 2);
        // A different kind in the same space is a different alert.
        assert_eq!(s.on_event(&event("inactivity", "check"), T0 + 40).len(), 1);
        // Once resolved, the next one is new and pages again.
        s.resolve(id, "asha", "", T0 + 50).unwrap();
        assert_eq!(
            s.on_event(&event("breathing_pause", "check"), T0 + 60)
                .len(),
            1
        );
        assert_eq!(s.list(false, 10).len(), 3);
    }

    #[test]
    fn routing_first_match_kinds_severity_and_info_never_pages() {
        let mut narrow = route("narrow");
        narrow.kinds = vec!["space_blind".into()];
        narrow.channels = vec!["webhook:boss".into()];
        let mut other_space = route("elsewhere");
        other_space.spaces = vec!["hall".into()];
        let s =
            AlertService::new(&cfg(vec![other_space, narrow, route("catch-all")]), None).unwrap();
        // Named kind: matches even below min_severity, and the first match owns it.
        let out = s.on_event(&event("space_blind", "advisory"), T0);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].channel, "webhook:boss");
        assert_eq!(s.list(true, 1)[0].route, "narrow");
        // Unnamed kind: min_severity applies (check), so advisory is ignored.
        assert!(s.on_event(&event("rate_high", "advisory"), T0).is_empty());
        assert_eq!(
            s.on_event(&event("rate_high", "check"), T0)[0].channel,
            "webhook:pager"
        );
        // Info never pages, even when a route names it.
        let mut loud = route("loud");
        loud.kinds = vec!["occupancy_started".into()];
        let s = AlertService::new(&cfg(vec![loud]), None).unwrap();
        assert!(s
            .on_event(&event("occupancy_started", "info"), T0)
            .is_empty());
    }

    #[test]
    fn what_leaves_the_building_without_health_details() {
        let s = AlertService::new(&cfg(vec![route("r")]), None).unwrap();
        let out = s.on_event(&event("inactivity", "check"), T0);
        let wire = format!("{} {}", out[0].text, out[0].payload);
        for leak in [
            "13 BPM",
            "13",
            "31.5",
            "27",
            "0.83",
            "evidence",
            "confidence",
            "median_bpm",
            "no gross motion",
        ] {
            assert!(!wire.contains(leak), "{leak:?} left the building: {wire}");
        }
        assert!(
            wire.contains("inactivity") && wire.contains("Wet lab"),
            "{wire}"
        );
        assert_eq!(out[0].payload["advisory"], json!(true));
    }

    #[test]
    fn health_details_go_out_only_when_the_route_says_so() {
        let mut r = route("r");
        r.include_health_details = true;
        let s = AlertService::new(&cfg(vec![r]), None).unwrap();
        let out = s.on_event(&event("inactivity", "check"), T0);
        assert!(out[0].text.contains("breathing detected at 13 BPM"));
        assert_eq!(out[0].payload["confidence"], json!(0.83));
        assert_eq!(out[0].payload["event"]["median_bpm"], json!(31.5));
    }

    #[test]
    fn escalation_on_a_fake_clock_once_and_only_while_unacknowledged() {
        let mut r = route("r");
        r.ack_timeout_min = 5.0;
        r.escalate_to = vec!["webhook:boss".into()];
        let s = AlertService::new(&cfg(vec![r]), None).unwrap();
        s.on_event(&event("inactivity", "check"), T0);
        assert!(s.tick(T0 + 4 * 60_000).is_empty(), "too early");
        let out = s.tick(T0 + 5 * 60_000);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].channel, "webhook:boss");
        assert!(out[0].text.contains("not acknowledged within 5 min"));
        assert_eq!(s.list(true, 1)[0].escalation_level, 1);
        assert!(s.tick(T0 + 60 * 60_000).is_empty(), "escalates once");
        // An acknowledged alert never escalates.
        s.on_event(&event("breathing_pause", "check"), T0);
        let id = s.list(true, 1)[0].id;
        s.ack(id, "asha", T0 + 60_000).unwrap();
        assert!(s.tick(T0 + 10 * 60_000).is_empty());
    }

    #[test]
    fn state_survives_a_restart_and_a_repeat_after_it_still_pages_nobody() {
        let dir = tmp("restart");
        let c = cfg(vec![route("r")]);
        // Today, not the shared T0: T0 is a year old, and on reload the
        // resolved alert would (correctly) be pruned as past retention.
        let t0 = now_unix_ms();
        {
            let s = AlertService::new(&c, Some(&dir)).unwrap();
            s.on_event(&event("inactivity", "check"), t0);
            s.on_event(&event("breathing_pause", "check"), t0);
            let ids: Vec<u64> = s.list(true, 10).iter().map(|a| a.id).collect();
            s.ack(ids[0], "asha", t0 + 1).unwrap();
            s.resolve(ids[1], "ben", "fine", t0 + 2).unwrap();
        }
        // A torn last line, as a crash mid-write would leave.
        fs::OpenOptions::new()
            .append(true)
            .open(dir.join(ALERTS_FILE))
            .unwrap()
            .write_all(b"{\"id\": 7, \"spa")
            .unwrap();
        let s = AlertService::new(&c, Some(&dir)).unwrap();
        let all = s.list(false, 10);
        assert_eq!(all.len(), 2);
        assert_eq!(all[0].state, AlertState::Acknowledged);
        assert_eq!(all[1].resolution_note.as_deref(), Some("fine"));
        assert_eq!(s.status()["unreadable_lines_skipped"], json!(1));
        // The acknowledged alert is still the live one for its kind.
        assert!(s
            .on_event(&event("breathing_pause", "check"), t0 + 3)
            .is_empty());
        // New ids continue after the old ones.
        s.on_event(&event("inactivity", "check"), t0 + 4);
        assert!(s.list(false, 10).iter().all(|a| a.id <= 3));
        // Compacted: one line per alert, the torn one gone.
        let lines = fs::read_to_string(dir.join(ALERTS_FILE)).unwrap();
        assert!(lines
            .lines()
            .all(|l| serde_json::from_str::<Alert>(l).is_ok()));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn resolved_alerts_age_out_with_the_events_and_open_ones_never_do() {
        let dir = tmp("retention");
        let mut c = cfg(vec![route("r")]);
        c.event_retention_days = 30;
        let long_ago = now_unix_ms() - 40 * DAY_MS;
        {
            let s = AlertService::new(&c, Some(&dir)).unwrap();
            s.on_event(&event("inactivity", "check"), long_ago);
            s.on_event(&event("breathing_pause", "check"), long_ago);
            let id = s
                .list(true, 10)
                .iter()
                .find(|a| a.kind == "inactivity")
                .unwrap()
                .id;
            s.resolve(id, "asha", "", long_ago + 1000).unwrap();
        }
        let s = AlertService::new(&c, Some(&dir)).unwrap();
        let kinds: Vec<String> = s.list(false, 10).into_iter().map(|a| a.kind).collect();
        assert_eq!(
            kinds,
            vec!["breathing_pause"],
            "the open one stays however old"
        );
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn webhook_signature_matches_rfc_4231() {
        // RFC 4231, test case 2.
        assert_eq!(
            hmac_hex(b"Jefe", b"what do ya want for nothing?"),
            "5bdcc146bf60754e6a042426089575c75a003f089d2739839dec58b964ec3843"
        );
    }

    #[test]
    fn a_missing_token_stops_the_gateway_rather_than_starting_it_mute() {
        let mut c = cfg(vec![route("r")]);
        c.telegram.insert(
            "wardens".into(),
            crate::config::TelegramChannel {
                bot_token_env: "SAANS_TEST_TOKEN_THAT_IS_NOT_SET_7f3a".into(),
                chat_id: -1,
                api_base: "http://127.0.0.1:9".into(),
                allowed_user_ids: vec![],
            },
        );
        let e = AlertService::new(&c, None).err().expect("must refuse");
        assert!(
            e.to_string()
                .contains("SAANS_TEST_TOKEN_THAT_IS_NOT_SET_7f3a"),
            "{e}"
        );
    }
}
