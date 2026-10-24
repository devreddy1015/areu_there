//! HTTP API, WebSocket and the embedded dashboard.
//!
//! | Route | Returns | Who |
//! |---|---|---|
//! | `GET /app` | the dashboard (one self-contained HTML file, no external URLs) | anyone |
//! | `GET /api/me` | who you are, the mode, the version, what you may see | anyone |
//! | `POST /api/login` | `{user, password}` → session cookie | anyone |
//! | `POST /api/logout` | ends the session | anyone |
//! | `GET /api/spaces` | `[SpaceState]` (+ name, privacy badges, alerts) | in scope |
//! | `GET /api/spaces/{id}` | `{"state": SpaceState, "events": [last 50]}` | in scope |
//! | `GET /api/events?space=&since=&limit=` | events, oldest first | in scope |
//! | `GET /api/history?space=&since=&max_points=` | bounded time series | in scope |
//! | `GET /api/links` | `[{"space_id", "name", "links": [LinkReport]}]` | ops |
//! | `GET /api/health` | gateway self-health | ops |
//! | `GET /metrics` | Prometheus text format | ops |
//! | `GET /api/live` | WebSocket: `{"type":"state"}` every 0.5 s, `{"type":"event"}` as they happen | in scope |
//! | `GET /api/alerts?open=&limit=` | alerts, newest first, each as its event would be shown to you | as the event |
//! | `GET /api/alerts/{id}` | one alert | as the event |
//! | `POST /api/alerts/{id}/ack` | acknowledge; audited | admin, safety; in scope |
//! | `POST /api/alerts/{id}/resolve` | `{"note"}` optional; audited | admin, safety; in scope |
//!
//! Simulate mode (`--simulate`) adds four more. They exist only in that
//! mode, and everything they return is SYNTHETIC ([`crate::simulate`]).
//!
//! | Route | Returns | Who |
//! |---|---|---|
//! | `GET /api/sim` | the simulated room's geometry and ground truth | anyone |
//! | `GET /api/sim/coverage?res_m=` | ground-truth breathing sensitivity per cell, per link | anyone |
//! | `POST /api/sim/person` | `{x_m, y_m}` → moves the simulated person | in scope |
//! | `POST /api/sim/present` | `{present}` → puts a body in the room or takes it out | in scope |
//!
//! **Two locks on every byte of data.** The hub redacts by *consent*
//! before publishing, so a space with `vitals = false` has no breathing
//! data in the snapshot at all ([`crate::privacy`]). Handlers here apply
//! the second lock, by *role and scope* ([`crate::roles`]), to whatever
//! consent allowed. Neither substitutes for the other: consent decides
//! what may exist, the principal decides who may see it.
//!
//! The static surface is deliberately public: the seven site pages
//! ([`crate::site`]) and the dashboard at `/app` are files with no data
//! in them, and the dashboard has to load before anyone can log in. Every
//! route that carries data is behind [`principal`] without exception.
//!
//! The dashboard used to answer `/`. The public site took that path when
//! the two servers became one, and `assets/site/site.js` forwards the old
//! `/#system`-style fragments to `/app` so saved links still arrive.

use crate::audit::{Action, AuditLog};
use crate::auth::{self, Auth, Denied};
use crate::privacy;
use crate::roles::Principal;
use crate::shared::Shared;
use crate::timefmt::{now_unix_ms, parse_iso8601};
use axum::extract::ws::{Message, WebSocket, WebSocketUpgrade};
use axum::extract::{Path, Query, State};
use axum::http::{header, HeaderMap, HeaderValue, StatusCode};
use axum::response::{IntoResponse, Response};
use axum::routing::{get, post};
use axum::{Json, Router};
use serde::Deserialize;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::Arc;
use tokio::sync::{broadcast, watch};

pub const DASHBOARD_HTML: &str = include_str!("../assets/dashboard.html");
const CSP: &str =
    "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; img-src data:; \
                   connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

/// Everything the HTTP side needs. Built by `live.rs` / `main.rs` via
/// [`HttpDeps::new`].
#[derive(Clone)]
pub struct HttpDeps {
    pub shared: Arc<Shared>,
    pub stop: watch::Receiver<bool>,
    pub cfg: Arc<crate::config::GatewayConfig>,
    /// `None` when no `[[alert_route]]` is configured.
    pub alerts: Option<Arc<crate::alerts::AlertService>>,
    pub auth: Arc<Auth>,
    pub audit: Arc<AuditLog>,
    /// `Some` only in simulate mode, which is the only mode where the
    /// `/api/sim` routes exist at all.
    pub sim: Option<Arc<crate::simulate::SimState>>,
}

impl HttpDeps {
    pub fn new(
        shared: Arc<Shared>,
        stop: watch::Receiver<bool>,
        cfg: Arc<crate::config::GatewayConfig>,
        alerts: Option<Arc<crate::alerts::AlertService>>,
    ) -> Self {
        let auth = Arc::new(Auth::new(&cfg));
        let dir = (!cfg.data_dir.as_os_str().is_empty()).then(|| cfg.data_dir.clone());
        let audit = Arc::new(AuditLog::new(dir.as_deref(), cfg.utc_offset_minutes));
        // A Telegram button press is an acknowledgement like any other and
        // belongs in the same audit log, which this layer owns.
        if let Some(a) = &alerts {
            a.attach_audit(audit.clone());
        }
        Self {
            shared,
            stop,
            cfg,
            alerts,
            auth,
            audit,
            sim: None,
        }
    }

    /// For tests and the replay server, which want no audit file.
    pub fn without_audit(mut self) -> Self {
        self.audit = Arc::new(AuditLog::new(None, self.cfg.utc_offset_minutes));
        self
    }

    /// Turn on simulate mode's own routes. Nothing else registers them,
    /// so a live or replay gateway answers 404 for every `/api/sim` path.
    pub fn with_sim(mut self, sim: Arc<crate::simulate::SimState>) -> Self {
        self.sim = Some(sim);
        self
    }
}

#[derive(Clone)]
pub struct AppState {
    pub shared: Arc<Shared>,
    pub stop: watch::Receiver<bool>,
    pub cfg: Arc<crate::config::GatewayConfig>,
    pub alerts: Option<Arc<crate::alerts::AlertService>>,
    pub auth: Arc<Auth>,
    pub audit: Arc<AuditLog>,
    pub sim: Option<Arc<crate::simulate::SimState>>,
}

pub fn router(deps: HttpDeps) -> Router {
    let HttpDeps {
        shared,
        stop,
        cfg,
        alerts,
        auth,
        audit,
        sim,
    } = deps;
    let mut r = Router::new()
        .route("/app", get(dashboard))
        .route("/api/me", get(me))
        .route("/api/login", post(login))
        .route("/api/logout", post(logout))
        .route("/api/spaces", get(spaces))
        .route("/api/spaces/{id}", get(space))
        .route("/api/events", get(events))
        .route("/api/history", get(history))
        .route("/api/links", get(links))
        .route("/api/health", get(health))
        .route("/api/alerts", get(alerts_list))
        .route("/api/alerts/{id}", get(alert_get))
        .route("/api/alerts/{id}/ack", post(alert_ack))
        .route("/api/alerts/{id}/resolve", post(alert_resolve))
        .route("/metrics", get(metrics))
        .route("/api/live", get(live));
    // Registered only in simulate mode: in any other mode these paths
    // fall through to the 404, because there is no simulated room to
    // describe and nothing should suggest there is.
    if sim.is_some() {
        r = r
            .route("/api/sim", get(sim_describe))
            .route("/api/sim/coverage", get(sim_coverage))
            .route("/api/sim/person", post(sim_person))
            .route("/api/sim/present", post(sim_present));
    }
    // The public site. Registered from the one table in `site.rs` rather
    // than listed again here, so a page cannot exist as a file and be
    // unreachable, or be routed and not exist.
    for p in crate::site::PAGES {
        r = r.route(
            p.path,
            get(move || async move { crate::site::page_response(p.html) }),
        );
    }
    for a in crate::site::ASSETS {
        r = r.route(
            a.path,
            get(move || async move { crate::site::asset_response(a) }),
        );
    }
    r.fallback(|| async { error(StatusCode::NOT_FOUND, "no such endpoint") })
        .with_state(AppState {
            shared,
            stop,
            cfg,
            alerts,
            auth,
            audit,
            sim,
        })
}

fn error(status: StatusCode, msg: &str) -> Response {
    (status, Json(json!({ "error": msg }))).into_response()
}

fn no_store(mut r: Response) -> Response {
    r.headers_mut()
        .insert(header::CACHE_CONTROL, HeaderValue::from_static("no-store"));
    r
}

/// 401 with the challenge, so a machine client knows what to present.
fn unauthorized() -> Response {
    let mut r = error(StatusCode::UNAUTHORIZED, "authentication required");
    r.headers_mut().insert(
        header::WWW_AUTHENTICATE,
        HeaderValue::from_static("Bearer realm=\"saans\""),
    );
    r
}

fn header_str(h: &HeaderMap, name: header::HeaderName) -> Option<&str> {
    h.get(name).and_then(|v| v.to_str().ok())
}

/// Set `Secure` on the session cookie only when the request actually
/// arrived over TLS. The gateway speaks plain HTTP itself, so this is
/// the reverse proxy's word for it; without the header the cookie would
/// be dropped by the browser on a plain-HTTP LAN deployment.
fn is_https(h: &HeaderMap) -> bool {
    header_str(h, header::HeaderName::from_static("x-forwarded-proto"))
        .is_some_and(|v| v.eq_ignore_ascii_case("https"))
}

/// The principal for a request, or a 401. Denials are audited: a stream
/// of them is how you notice someone probing the port.
/// What the dashboard may offer this principal. The one source for both
/// `/api/me` and the WebSocket `hello` frame: they used to be built
/// separately, `hello` carried only `system`, and the dashboard replaces its
/// copy with whichever arrived last -- so once the socket connected it
/// forgot `audit`, and would have forgotten `alerts`. The gateway checks
/// every request regardless; this only decides what is shown.
fn capabilities(p: &Principal) -> Value {
    json!({
        "system": p.can_see_system(),
        "audit": p.can_see_audit(),
        // Acknowledge / Resolve. Still checked per request, per space.
        "alerts": matches!(p.role, crate::roles::Role::Admin | crate::roles::Role::Safety),
    })
}

fn principal(app: &AppState, h: &HeaderMap, path: &str) -> Result<Principal, Response> {
    let authz = header_str(h, header::AUTHORIZATION);
    let cookie = header_str(h, header::COOKIE);
    app.auth.principal(authz, cookie).map_err(|d| {
        let why = match d {
            Denied::Missing => "no credential",
            Denied::Invalid => "credential rejected",
        };
        app.audit.record(
            now_unix_ms(),
            Action::Denied,
            "-",
            None,
            json!({ "path": path, "why": why }),
        );
        unauthorized()
    })
}

/// Routes that exist to run the building, not to watch the people in it.
fn system_only(app: &AppState, h: &HeaderMap, path: &str) -> Result<Principal, Response> {
    let p = principal(app, h, path)?;
    if !p.can_see_system() {
        return Err(error(
            StatusCode::FORBIDDEN,
            "this endpoint needs the admin or ops role",
        ));
    }
    Ok(p)
}

async fn dashboard() -> Response {
    let mut r = DASHBOARD_HTML.into_response();
    let h = r.headers_mut();
    h.insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/html; charset=utf-8"),
    );
    h.insert(
        header::CONTENT_SECURITY_POLICY,
        HeaderValue::from_static(CSP),
    );
    h.insert(
        header::X_CONTENT_TYPE_OPTIONS,
        HeaderValue::from_static("nosniff"),
    );
    h.insert(
        header::REFERRER_POLICY,
        HeaderValue::from_static("no-referrer"),
    );
    no_store(r)
}

/// What the shell needs before it can render anything: is a login
/// required, am I logged in, what may I see, and what is this gateway.
/// Answers for anonymous callers too — otherwise the login form could
/// never find out that it is needed.
async fn me(State(app): State<AppState>, h: HeaderMap) -> Response {
    let who = app.auth.principal(
        header_str(&h, header::AUTHORIZATION),
        header_str(&h, header::COOKIE),
    );
    let base = json!({
        "auth_required": app.auth.required(),
        "version": env!("CARGO_PKG_VERSION"),
        "mode": app.shared.mode,
    });
    let mut v = base;
    match who {
        Ok(p) => {
            v["authenticated"] = true.into();
            v["name"] = p.name.as_str().into();
            v["role"] = p.role.as_str().into();
            v["spaces"] = match &p.spaces {
                Some(s) => s.iter().cloned().collect::<Vec<_>>().into(),
                None => Value::Null,
            };
            v["can"] = capabilities(&p);
        }
        Err(_) => {
            v["authenticated"] = false.into();
            v["can"] = json!({ "system": false, "audit": false, "alerts": false });
        }
    }
    no_store(Json(v).into_response())
}

#[derive(Debug, Deserialize)]
pub struct LoginBody {
    pub user: String,
    pub password: String,
}

/// Logins in flight. Argon2 is deliberately expensive and the failure
/// path deliberately sleeps, so an unauthenticated flood of them is a
/// denial-of-service vector against a safety system. Four at a time is
/// more than any real operator needs and bounds the blocking pool.
const MAX_LOGINS_IN_FLIGHT: usize = 4;
static LOGINS_IN_FLIGHT: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);

/// Decrements on drop, so an early return or a panic cannot leak a slot.
struct LoginSlot;

impl LoginSlot {
    fn take() -> Option<Self> {
        let prev = LOGINS_IN_FLIGHT.fetch_add(1, Ordering::AcqRel);
        if prev >= MAX_LOGINS_IN_FLIGHT {
            LOGINS_IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
            return None;
        }
        Some(Self)
    }
}

impl Drop for LoginSlot {
    fn drop(&mut self) {
        LOGINS_IN_FLIGHT.fetch_sub(1, Ordering::AcqRel);
    }
}

async fn login(State(app): State<AppState>, h: HeaderMap, Json(body): Json<LoginBody>) -> Response {
    if !app.auth.required() {
        return error(
            StatusCode::BAD_REQUEST,
            "this gateway does not use authentication (loopback http_addr)",
        );
    }
    let Some(slot) = LoginSlot::take() else {
        return error(
            StatusCode::TOO_MANY_REQUESTS,
            "too many sign-ins at once; try again in a moment",
        );
    };
    // Argon2 verification and the fixed failure delay are both blocking,
    // and this runtime has two worker threads. Running them inline would
    // let a handful of sign-in attempts stall ingest.
    let auth = app.auth.clone();
    let user = body.user.clone();
    let password = body.password;
    let attempt = tokio::task::spawn_blocking(move || {
        let r = auth.login(&user, &password);
        drop(slot);
        r
    })
    .await;
    let attempt = match attempt {
        Ok(r) => r,
        Err(_) => {
            return error(
                StatusCode::INTERNAL_SERVER_ERROR,
                "sign-in could not be completed",
            )
        }
    };
    let Some((id, p)) = attempt else {
        app.audit.record(
            now_unix_ms(),
            Action::LoginFailed,
            &body.user,
            None,
            json!({}),
        );
        // Deliberately the same answer for a wrong password and an
        // unknown user.
        return error(StatusCode::UNAUTHORIZED, "wrong user or password");
    };
    app.audit.record(
        now_unix_ms(),
        Action::Login,
        &p.name,
        Some(p.role.as_str()),
        json!({}),
    );
    let max_age = (app.cfg.auth.session_hours * 3600.0) as u64;
    let cookie = auth::session_cookie(&id, max_age, is_https(&h));
    let mut r = Json(
        json!({ "name": p.name, "role": p.role.as_str(), "can": { "system": p.can_see_system() } }),
    )
    .into_response();
    if let Ok(v) = HeaderValue::from_str(&cookie) {
        r.headers_mut().insert(header::SET_COOKIE, v);
    }
    no_store(r)
}

async fn logout(State(app): State<AppState>, h: HeaderMap) -> Response {
    let who = app.auth.principal(
        header_str(&h, header::AUTHORIZATION),
        header_str(&h, header::COOKIE),
    );
    app.auth.logout(header_str(&h, header::COOKIE));
    if let Ok(p) = who {
        app.audit.record(
            now_unix_ms(),
            Action::Logout,
            &p.name,
            Some(p.role.as_str()),
            json!({}),
        );
    }
    let mut r = Json(json!({ "ok": true })).into_response();
    if let Ok(v) = HeaderValue::from_str(&auth::clear_cookie(is_https(&h))) {
        r.headers_mut().insert(header::SET_COOKIE, v);
    }
    no_store(r)
}

/// Record that this principal was shown a space's consented vitals.
/// Throttled to once an hour per (principal, space) inside the log.
fn note_vitals(app: &AppState, p: &Principal, v: &Value) {
    if !app.audit.enabled() {
        return;
    }
    let id = v.get("space_id").and_then(Value::as_str).unwrap_or("");
    let consented = v.get("vitals").and_then(Value::as_bool).unwrap_or(false);
    if p.can_see_vitals(id, consented) {
        app.audit.vitals_read(now_unix_ms(), p, id);
    }
}

async fn spaces(State(app): State<AppState>, h: HeaderMap) -> Response {
    let p = match principal(&app, &h, "/api/spaces") {
        Ok(p) => p,
        Err(r) => return r,
    };
    let list: Vec<Value> = {
        let pub_ = app.shared.read();
        pub_.spaces
            .iter()
            .filter_map(|(_, v)| privacy::space_for(v, &p))
            .collect()
    };
    for v in &list {
        note_vitals(&app, &p, v);
    }
    no_store(Json(list).into_response())
}

async fn space(State(app): State<AppState>, h: HeaderMap, Path(id): Path<String>) -> Response {
    let p = match principal(&app, &h, "/api/spaces/{id}") {
        Ok(p) => p,
        Err(r) => return r,
    };
    let state = {
        let pub_ = app.shared.read();
        pub_.spaces
            .iter()
            .find(|(sid, _)| *sid == id)
            .and_then(|(_, v)| privacy::space_for(v, &p))
    };
    // A space out of scope is indistinguishable from one that does not
    // exist: the scope list is not a directory of the building.
    let Some(state) = state else {
        return error(StatusCode::NOT_FOUND, "no such space");
    };
    note_vitals(&app, &p, &state);
    let events: Vec<Value> = app
        .shared
        .query_events(Some(&id), None, 50)
        .iter()
        .filter_map(|e| privacy::event_for(e, &p))
        .collect();
    no_store(Json(json!({ "state": state, "events": events })).into_response())
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

async fn events(
    State(app): State<AppState>,
    h: HeaderMap,
    Query(q): Query<EventsQuery>,
) -> Response {
    let p = match principal(&app, &h, "/api/events") {
        Ok(p) => p,
        Err(r) => return r,
    };
    let since = match q.since.as_deref().filter(|s| !s.is_empty()) {
        None => None,
        Some(s) => match parse_since(s) {
            Some(ms) => Some(ms),
            None => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "since must be Unix seconds or ISO 8601 (2026-10-13T08:00:00+05:30)",
                )
            }
        },
    };
    let limit = q.limit.unwrap_or(100).clamp(1, 1000);
    let space = q.space.as_deref().filter(|s| !s.is_empty());
    // Filter after the ring query, so `limit` is a limit on what the
    // caller receives rather than on what they were eligible for.
    let found: Vec<Value> = app
        .shared
        .query_events(space, since, limit)
        .iter()
        .filter_map(|e| privacy::event_for(e, &p))
        .collect();
    no_store(Json(found).into_response())
}

#[derive(Debug, Deserialize)]
pub struct HistoryQuery {
    pub space: Option<String>,
    /// Gateway seconds (`t_s`), not wall clock: points strictly after.
    pub since: Option<f64>,
    /// Stride the series down to at most this many points. 0 = no limit.
    pub max_points: Option<usize>,
}

async fn history(
    State(app): State<AppState>,
    h: HeaderMap,
    Query(q): Query<HistoryQuery>,
) -> Response {
    let p = match principal(&app, &h, "/api/history") {
        Ok(p) => p,
        Err(r) => return r,
    };
    let space = q.space.as_deref().filter(|s| !s.is_empty());
    // Capped: a client cannot ask for more points than the ring holds,
    // and cannot ask for an unbounded response by omitting the limit.
    let cap = app.cfg.history.capacity();
    let max_points = q
        .max_points
        .map_or(cap, |n| if n == 0 { cap } else { n.min(cap) });
    let consented = |id: &str| app.cfg.spaces.iter().any(|s| s.id == id && s.vitals);
    let named = |id: &str| {
        app.cfg.spaces.iter().find(|s| s.id == id).map_or_else(
            || id.to_string(),
            |s| crate::config::GatewayConfig::space_name(s).to_string(),
        )
    };
    let since = q.since.filter(|x| x.is_finite());
    let v = app
        .shared
        .history()
        .to_json(&p, &consented, &named, space, since, max_points);
    if app.audit.enabled() {
        for s in v["spaces"].as_array().into_iter().flatten() {
            if s.get("breathing_bpm").is_some() {
                app.audit
                    .vitals_read(now_unix_ms(), &p, s["space_id"].as_str().unwrap_or(""));
            }
        }
    }
    no_store(Json(v).into_response())
}

async fn links(State(app): State<AppState>, h: HeaderMap) -> Response {
    let p = match system_only(&app, &h, "/api/links") {
        Ok(p) => p,
        Err(r) => return r,
    };
    let all = app.shared.read().links.clone();
    let scoped: Vec<Value> = all
        .as_array()
        .into_iter()
        .flatten()
        .filter(|sp| p.can_see_space(sp.get("space_id").and_then(Value::as_str).unwrap_or("")))
        .cloned()
        .collect();
    no_store(Json(scoped).into_response())
}

async fn health(State(app): State<AppState>, h: HeaderMap) -> Response {
    if let Err(r) = system_only(&app, &h, "/api/health") {
        return r;
    }
    let mut v = app.shared.read().health.clone();
    if let Value::Object(m) = &mut v {
        // Fresh values; the rest is as of the last tick.
        m.insert(
            "uptime_s".into(),
            app.shared.started.elapsed().as_secs_f64().into(),
        );
        m.insert("ws_clients".into(), app.shared.ws_clients().into());
        m.insert(
            "auth".into(),
            json!({
                "required": app.auth.required(),
                "sessions_open": app.auth.sessions_open(),
                "users": app.cfg.users.len(),
                "tokens": app.cfg.tokens.len(),
            }),
        );
        let mut audit = app.audit.status();
        // What will be kept, beside what was written: an operator who can
        // see the second but not the first cannot check ADR-010 is in force.
        audit["retention_days"] = app.cfg.audit_retention_days.into();
        m.insert("audit".into(), audit);
        if let Some(alerts) = &app.alerts {
            m.insert("alerts".into(), alerts.status());
        }
        if let Some(c) = &app.cfg.mqtt {
            let mut mqtt = app.shared.mqtt.json();
            mqtt["broker"] = format!("{}:{}", c.host, c.port).into();
            m.insert("mqtt".into(), mqtt);
        }
    }
    no_store(Json(v).into_response())
}

async fn metrics(State(app): State<AppState>, h: HeaderMap) -> Response {
    if let Err(r) = system_only(&app, &h, "/metrics") {
        return r;
    }
    let body = app.shared.read().metrics.clone();
    let mut r = body.into_response();
    r.headers_mut().insert(
        header::CONTENT_TYPE,
        HeaderValue::from_static("text/plain; version=0.0.4; charset=utf-8"),
    );
    no_store(r)
}

// ------------------------------------------------------------- simulate
//
// Every value these four return is SYNTHETIC: it comes from `saans-sim`
// (geometry, ground truth, coverage) or it is a position the caller
// themselves set. None of it is an inference from CSI, and nothing in
// the gateway can produce a position from CSI (`docs/CAPABILITY_LIMITS.md`).

/// The simulated room, or the 404 any other mode gives this path.
fn sim_state(app: &AppState) -> Result<Arc<crate::simulate::SimState>, Response> {
    app.sim
        .clone()
        .ok_or_else(|| error(StatusCode::NOT_FOUND, "no such endpoint"))
}

/// Moving the simulated person writes to the simulated space, so the
/// principal has to be allowed to see that space — the same scope check
/// every other space answers to. On a loopback gateway with no users
/// configured everyone is the local admin, so the demo just works.
fn sim_writer(
    app: &AppState,
    h: &HeaderMap,
    path: &str,
    sim: &crate::simulate::SimState,
) -> Result<Principal, Response> {
    let p = principal(app, h, path)?;
    if !p.can_see_space(sim.space_id()) {
        return Err(error(
            StatusCode::FORBIDDEN,
            "the simulated space is not in your scope",
        ));
    }
    Ok(p)
}

/// Reading the simulated room. 404 rather than 403, matching
/// `/api/spaces/{id}`: a space outside a principal's scope is
/// indistinguishable from one that does not exist, because the scope
/// list is not a directory of the building.
fn sim_reader(
    app: &AppState,
    h: &HeaderMap,
    path: &str,
    sim: &crate::simulate::SimState,
) -> Result<Principal, Response> {
    let p = principal(app, h, path)?;
    if !p.can_see_space(sim.space_id()) {
        return Err(error(StatusCode::NOT_FOUND, "no such space"));
    }
    Ok(p)
}

/// The request body as a JSON object. Parsed here rather than with the
/// `Json` extractor so that a missing or non-numeric field is the 400
/// the API promises, in the same `{"error": ...}` shape as every other
/// failure, instead of an extractor rejection in a different format.
fn sim_body(body: &str) -> Result<Value, Response> {
    match serde_json::from_str::<Value>(body) {
        Ok(v) if v.is_object() => Ok(v),
        _ => Err(error(StatusCode::BAD_REQUEST, "body must be a JSON object")),
    }
}

fn sim_f64(v: &Value, key: &str) -> Result<f64, Response> {
    match v.get(key).and_then(Value::as_f64) {
        Some(x) if x.is_finite() => Ok(x),
        _ => Err(error(
            StatusCode::BAD_REQUEST,
            &format!("{key} must be a finite number"),
        )),
    }
}

async fn sim_describe(State(app): State<AppState>, h: HeaderMap) -> Response {
    let sim = match sim_state(&app) {
        Ok(s) => s,
        Err(r) => return r,
    };
    // The same scope check the POSTs beside this one make. /api/sim
    // carries the ground-truth position of the simulated occupant, so it
    // answers to the space's scope like any other space-level read — a
    // principal scoped away from sim-room gets 404 here, exactly as
    // /api/spaces/{id} would give them.
    if let Err(r) = sim_reader(&app, &h, "/api/sim", &sim) {
        return r;
    }
    no_store(Json(sim.describe()).into_response())
}

#[derive(Debug, Deserialize)]
pub struct CoverageQuery {
    /// Cell size in metres. Defaults to
    /// [`crate::simulate::DEFAULT_COVERAGE_RES_M`], which is small
    /// enough to sample the half-wavelength structure the map is of
    /// rather than alias it.
    ///
    /// Taken as a string and parsed here rather than as an `f64`, so that
    /// `res_m=abc` is rejected by this route in the same
    /// `{"error": ...}` shape as every other bad `res_m`. Typed straight
    /// as `f64`, the extractor rejects it first, with a plain-text axum
    /// message a client parsing `.error` cannot read.
    pub res_m: Option<String>,
}

async fn sim_coverage(
    State(app): State<AppState>,
    h: HeaderMap,
    Query(q): Query<CoverageQuery>,
) -> Response {
    let sim = match sim_state(&app) {
        Ok(s) => s,
        Err(r) => return r,
    };
    if let Err(r) = sim_reader(&app, &h, "/api/sim/coverage", &sim) {
        return r;
    }
    // A map may be up to 100k cells x 56 subcarriers x 3 torso points.
    // That is sub-second, but it is sub-second of solid arithmetic on a
    // two-worker runtime, so it runs where blocking work belongs —
    // the same reason `login` does.
    let res_m = match q.res_m.as_deref() {
        // DEFAULT_COVERAGE_RES_M, not a round number. The field this map
        // draws oscillates on a half wavelength, 6.2 cm at 2.437 GHz, so
        // a 25 cm grid is a point sample of something that changes
        // completely between cells: it can paint a null as ordinary and
        // an ordinary cell as a null, which is the one error that makes
        // the map worse than no map. 5 cm samples the fringes rather
        // than aliasing them, and in a 6x5 m room costs 120x100 cells
        // -- well inside saans-sim's 100k cap, and still sub-second.
        None => crate::simulate::DEFAULT_COVERAGE_RES_M,
        Some(raw) => match raw.trim().parse::<f64>() {
            Ok(v) => v,
            Err(_) => {
                return error(
                    StatusCode::BAD_REQUEST,
                    &format!("res_m {raw} must be finite and positive"),
                )
            }
        },
    };
    match tokio::task::spawn_blocking(move || sim.coverage(res_m)).await {
        Ok(Ok(v)) => no_store(Json(v).into_response()),
        Ok(Err(e)) => error(StatusCode::BAD_REQUEST, &e),
        Err(_) => error(
            StatusCode::INTERNAL_SERVER_ERROR,
            "the coverage map could not be computed",
        ),
    }
}

async fn sim_person(State(app): State<AppState>, h: HeaderMap, body: String) -> Response {
    let sim = match sim_state(&app) {
        Ok(s) => s,
        Err(r) => return r,
    };
    if let Err(r) = sim_writer(&app, &h, "/api/sim/person", &sim) {
        return r;
    }
    let v = match sim_body(&body) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let (x, y) = match (sim_f64(&v, "x_m"), sim_f64(&v, "y_m")) {
        (Ok(x), Ok(y)) => (x, y),
        (Err(r), _) | (_, Err(r)) => return r,
    };
    let (x_m, y_m) = sim.set_person((x, y));
    no_store(Json(json!({ "ok": true, "x_m": x_m, "y_m": y_m })).into_response())
}

async fn sim_present(State(app): State<AppState>, h: HeaderMap, body: String) -> Response {
    let sim = match sim_state(&app) {
        Ok(s) => s,
        Err(r) => return r,
    };
    if let Err(r) = sim_writer(&app, &h, "/api/sim/present", &sim) {
        return r;
    }
    let v = match sim_body(&body) {
        Ok(v) => v,
        Err(r) => return r,
    };
    let Some(want) = v.get("present").and_then(Value::as_bool) else {
        return error(StatusCode::BAD_REQUEST, "present must be true or false");
    };
    no_store(Json(json!({ "ok": true, "present": sim.set_present(want) })).into_response())
}

async fn live(ws: WebSocketUpgrade, State(app): State<AppState>, h: HeaderMap) -> Response {
    // The handshake is a GET: authenticate it like any other, before the
    // upgrade, so an anonymous client never reaches the stream.
    let p = match principal(&app, &h, "/api/live") {
        Ok(p) => p,
        Err(r) => return r,
    };
    ws.on_upgrade(move |socket| ws_session(socket, app, p))
}

/// One broadcast message as `p` may see it, or `None` to withhold it.
///
/// Re-filtering per client costs a parse per tick per connection. The
/// alternative — one pre-serialized message for everybody — is how a
/// space ends up on a screen that should not have it, so the cost is the
/// point. An unrestricted admin skips it entirely (see `ws_session`).
fn filter_live(text: &str, p: &Principal) -> Option<String> {
    let v: Value = serde_json::from_str(text).ok()?;
    match v.get("type").and_then(Value::as_str)? {
        "state" => {
            let spaces: Vec<Value> = v
                .get("spaces")?
                .as_array()?
                .iter()
                .filter_map(|s| privacy::space_for(s, p))
                .collect();
            Some(json!({"type": "state", "t_s": v.get("t_s").cloned().unwrap_or(Value::Null), "spaces": spaces}).to_string())
        }
        "event" => privacy::event_for(v.get("event")?, p)
            .map(|e| json!({"type": "event", "event": e}).to_string()),
        _ => Some(text.to_string()),
    }
}

async fn ws_session(mut socket: WebSocket, app: AppState, who: Principal) {
    let shared = app.shared;
    let mut rx = shared.live_tx.subscribe();
    shared.ws_clients.fetch_add(1, Ordering::Relaxed);
    // An admin with no space restriction sees exactly the published
    // message, so the filter is a no-op and is skipped.
    let unrestricted = who.role == crate::roles::Role::Admin && who.spaces.is_none();
    let send = |text: &str| -> Option<String> {
        if unrestricted {
            Some(text.to_string())
        } else {
            filter_live(text, &who)
        }
    };
    let hello = json!({
        "type": "hello",
        "version": env!("CARGO_PKG_VERSION"),
        "mode": shared.mode,
        "name": who.name,
        "role": who.role.as_str(),
        "can": capabilities(&who),
    })
    .to_string();
    let first = shared.read().state_msg.clone();
    let mut ok = socket.send(Message::Text(hello.into())).await.is_ok();
    if let (true, Some(state)) = (ok, first) {
        if let Some(text) = send(&state) {
            ok = socket.send(Message::Text(text.into())).await.is_ok();
        }
    }
    let mut stop = app.stop;
    while ok {
        tokio::select! {
            msg = rx.recv() => match msg {
                Ok(text) => {
                    if let Some(out) = send(&text) {
                        ok = socket.send(Message::Text(out.into())).await.is_ok();
                    }
                }
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
    use crate::roles::Role;

    #[test]
    fn since_accepts_unix_seconds_and_iso() {
        assert_eq!(parse_since("1700000000.5"), Some(1_700_000_000_500));
        assert_eq!(parse_since("1970-01-01T00:00:01Z"), Some(1000));
        assert_eq!(parse_since("nope"), None);
    }

    /// The dashboard must fetch nothing from the internet: it is served
    /// from a box in a building that may have no route out, and a
    /// stylesheet that fails to load is a dashboard nobody can read.
    ///
    /// XML namespace URIs are the one exception. `http://www.w3.org/...`
    /// in `xmlns` or `createElementNS` is an identifier, not an address
    /// — nothing dereferences it — and SVG cannot be built without it.
    /// Allowed by name here rather than by assembling the string at
    /// runtime to slip past the check, which would leave this test
    /// guarding nothing.
    #[test]
    fn dashboard_has_no_external_urls() {
        const XML_NS: [&str; 3] = [
            "http://www.w3.org/2000/svg",
            "http://www.w3.org/1999/xhtml",
            "http://www.w3.org/1999/xlink",
        ];
        let mut stripped = DASHBOARD_HTML.to_string();
        for ns in XML_NS {
            stripped = stripped.replace(ns, "");
        }
        for bad in [
            "http://", "https://", "//cdn", "src=\"//", "@import", "url(http",
        ] {
            assert!(
                !stripped.contains(bad),
                "dashboard must stay offline-capable: found {bad:?}"
            );
        }
        // The namespaces that *are* present may only appear as a
        // namespace argument, never as something to fetch. (A bare
        // `https:` — the scheme test the WebSocket URL does — is not a
        // URL and is not matched here.)
        for scheme in ["http://", "https://"] {
            for (i, _) in DASHBOARD_HTML.match_indices(scheme) {
                let rest = &DASHBOARD_HTML[i..];
                let before = &DASHBOARD_HTML[i.saturating_sub(40)..i];
                assert!(
                    XML_NS.iter().any(|ns| rest.starts_with(ns))
                        && (before.contains("xmlns") || before.contains("createElementNS")),
                    "unexpected URL at byte {i}: ...{before}{}",
                    &rest[..rest.len().min(40)]
                );
            }
        }
        assert!(DASHBOARD_HTML.contains("not a medical device"));
    }

    fn p(role: Role, spaces: Option<&[&str]>) -> Principal {
        Principal {
            name: "x".into(),
            role,
            spaces: spaces.map(|s| s.iter().map(|x| x.to_string()).collect()),
        }
    }

    const STATE: &str = r#"{"type":"state","t_s":12.5,"spaces":[
        {"space_id":"lab","vitals":true,"occupied":true,"breathing":{"bpm":14.0},"alerts":[{"name":"inactivity"}]},
        {"space_id":"hall","vitals":false,"occupied":false,"alerts":[]}]}"#;

    #[test]
    fn live_state_is_filtered_per_client() {
        let full: Value =
            serde_json::from_str(&filter_live(STATE, &p(Role::Admin, None)).unwrap()).unwrap();
        assert_eq!(full["spaces"].as_array().map(Vec::len), Some(2));
        assert_eq!(full["t_s"], json!(12.5));
        assert_eq!(full["spaces"][0]["breathing"]["bpm"], json!(14.0));

        let ops: Value =
            serde_json::from_str(&filter_live(STATE, &p(Role::Ops, None)).unwrap()).unwrap();
        assert!(
            ops["spaces"][0].get("breathing").is_none(),
            "ops sees the room, not the person"
        );
        assert_eq!(
            ops["spaces"][0]["alerts"],
            json!([]),
            "and no health alerts"
        );

        let scoped: Value =
            serde_json::from_str(&filter_live(STATE, &p(Role::Safety, Some(&["hall"]))).unwrap())
                .unwrap();
        assert_eq!(scoped["spaces"].as_array().map(Vec::len), Some(1));
        assert_eq!(scoped["spaces"][0]["space_id"], "hall");
    }

    #[test]
    fn live_events_can_be_withheld_entirely() {
        let ev = r#"{"type":"event","event":{"space_id":"lab","type":"breathing_pause","duration_s":11.0}}"#;
        assert!(filter_live(ev, &p(Role::Safety, None)).is_some());
        assert!(
            filter_live(ev, &p(Role::Ops, None)).is_none(),
            "nothing is sent, not an empty event"
        );
        assert!(filter_live(ev, &p(Role::Safety, Some(&["hall"]))).is_none());
    }

    /// The only thing that touches this counter is `login`, which no
    /// unit test calls, so taking every slot here is self-contained.
    #[test]
    fn concurrent_logins_are_capped_and_slots_are_returned() {
        let held: Vec<LoginSlot> = (0..MAX_LOGINS_IN_FLIGHT)
            .map(|_| LoginSlot::take().expect("under the cap"))
            .collect();
        assert!(
            LoginSlot::take().is_none(),
            "the {MAX_LOGINS_IN_FLIGHT}th + 1 attempt is refused"
        );
        drop(held);
        assert!(LoginSlot::take().is_some(), "slots are returned on drop");
    }

    #[test]
    fn a_message_that_is_not_json_is_dropped_not_forwarded() {
        assert!(filter_live("{oops", &p(Role::Admin, None)).is_none());
        assert!(filter_live(r#"{"no_type":1}"#, &p(Role::Admin, None)).is_none());
        // An unknown message type passes through unchanged: it carries
        // no space data by construction (`hello`).
        assert_eq!(
            filter_live(r#"{"type":"hello","mode":"live"}"#, &p(Role::Ops, None)).as_deref(),
            Some(r#"{"type":"hello","mode":"live"}"#)
        );
    }
}

// ---------------------------------------------------------------- alerts
//
// Who sees an alert is exactly who would see the event that raised it:
// `privacy::event_for` decides, so the rules cannot drift between the
// event list and the alert list. An alert out of a principal's scope, or of
// a kind their role may not see, answers 404 -- indistinguishable from one
// that does not exist. Seeing an alert and acting on it are different:
// `can_act_on_alerts` is admin or safety, in scope, so an ops user who can
// see a link_down alert gets 403 if they try to acknowledge it.

#[derive(Debug, Deserialize)]
pub struct AlertsQuery {
    #[serde(default)]
    pub open: bool,
    pub limit: Option<usize>,
}

/// The alert as `p` may see it, or `None`.
fn alert_for(a: &crate::alerts::Alert, p: &Principal) -> Option<Value> {
    let event = privacy::event_for(&a.event, p)?;
    let mut v = serde_json::to_value(a).ok()?;
    v["event"] = event;
    Some(v)
}

fn alerts_service(app: &AppState) -> Result<&Arc<crate::alerts::AlertService>, Response> {
    app.alerts.as_ref().ok_or_else(|| {
        error(
            StatusCode::NOT_FOUND,
            "alert routing is not configured on this gateway",
        )
    })
}

async fn alerts_list(
    State(app): State<AppState>,
    h: HeaderMap,
    Query(q): Query<AlertsQuery>,
) -> Response {
    let p = match principal(&app, &h, "/api/alerts") {
        Ok(p) => p,
        Err(r) => return r,
    };
    let svc = match alerts_service(&app) {
        Ok(s) => s,
        Err(r) => return r,
    };
    let limit = q.limit.unwrap_or(50).min(500);
    // Filter first, then take: a principal scoped to one space must get
    // `limit` of *their* alerts, not `limit` minus everyone else's.
    let out: Vec<Value> = svc
        .list(q.open, usize::MAX)
        .iter()
        .filter_map(|a| alert_for(a, &p))
        .take(limit)
        .collect();
    no_store(Json(json!({ "alerts": out })).into_response())
}

async fn alert_get(State(app): State<AppState>, h: HeaderMap, Path(id): Path<u64>) -> Response {
    let p = match principal(&app, &h, "/api/alerts/{id}") {
        Ok(p) => p,
        Err(r) => return r,
    };
    let svc = match alerts_service(&app) {
        Ok(s) => s,
        Err(r) => return r,
    };
    match svc.get(id).as_ref().and_then(|a| alert_for(a, &p)) {
        Some(v) => no_store(Json(v).into_response()),
        None => error(StatusCode::NOT_FOUND, "no such alert"),
    }
}

#[derive(Debug, Default, Deserialize)]
struct ResolveBody {
    #[serde(default)]
    note: String,
}

/// Longer than this is a paste, not a note, and it goes in the audit log.
const MAX_NOTE_CHARS: usize = 500;

async fn alert_act(app: AppState, h: HeaderMap, id: u64, resolve: bool, note: String) -> Response {
    let path = if resolve {
        "/api/alerts/{id}/resolve"
    } else {
        "/api/alerts/{id}/ack"
    };
    let p = match principal(&app, &h, path) {
        Ok(p) => p,
        Err(r) => return r,
    };
    let svc = match alerts_service(&app) {
        Ok(s) => s,
        Err(r) => return r,
    };
    let Some(alert) = svc.get(id).filter(|a| alert_for(a, &p).is_some()) else {
        return error(StatusCode::NOT_FOUND, "no such alert");
    };
    if !p.can_act_on_alerts(&alert.space_id) {
        return error(
            StatusCode::FORBIDDEN,
            "your role can see this alert but not act on it",
        );
    }
    let now = now_unix_ms();
    let note: String = note.chars().take(MAX_NOTE_CHARS).collect();
    let result = if resolve {
        svc.resolve(id, &p.name, &note, now)
    } else {
        svc.ack(id, &p.name, now)
    };
    match result {
        Ok(a) => {
            app.audit.record(
                now,
                if resolve {
                    Action::AlertResolve
                } else {
                    Action::AlertAck
                },
                &p.name,
                Some(p.role.as_str()),
                json!({ "via": "api", "alert_id": a.id, "space_id": a.space_id, "kind": a.kind }),
            );
            match alert_for(&a, &p) {
                Some(v) => no_store(Json(v).into_response()),
                None => error(StatusCode::NOT_FOUND, "no such alert"),
            }
        }
        Err(crate::alerts::AlertError::NotFound) => error(StatusCode::NOT_FOUND, "no such alert"),
        Err(e) => error(StatusCode::CONFLICT, &e.to_string()),
    }
}

async fn alert_ack(State(app): State<AppState>, h: HeaderMap, Path(id): Path<u64>) -> Response {
    alert_act(app, h, id, false, String::new()).await
}

async fn alert_resolve(
    State(app): State<AppState>,
    h: HeaderMap,
    Path(id): Path<u64>,
    body: String,
) -> Response {
    let note = if body.trim().is_empty() {
        String::new()
    } else {
        match serde_json::from_str::<ResolveBody>(&body) {
            Ok(b) => b.note,
            Err(_) => {
                return error(
                    StatusCode::BAD_REQUEST,
                    "body must be {\"note\": \"...\"} or empty",
                )
            }
        }
    };
    alert_act(app, h, id, true, note).await
}
