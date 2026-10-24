//! The alert service end to end: a real gateway on loopback with
//! authentication forced on, frames over UDP, a manual clock, and real HTTP
//! servers standing in for the webhook host and for Telegram.
//!
//! The first test is the exit gate in the build brief: a scripted breathing
//! pause produces one alert -> the webhook receives it -> it is acknowledged
//! through the API the dashboard calls -> the audit log has the line -> it
//! survives a gateway restart.

mod support;

use axum::body::Bytes;
use axum::extract::State;
use axum::http::{HeaderMap, Uri};
use axum::routing::any;
use axum::{Json, Router};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::VecDeque;
use std::net::SocketAddr;
use std::path::Path;
use std::sync::{Arc, Mutex, Once};
use std::time::Duration;
use support::*;
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::{TcpListener, TcpStream};

const HOOK_KEY_ENV: &str = "SAANS_ALERTS_TEST_HOOK_KEY";
const HOOK_KEY: &str = "hook-key-for-tests";
const TG_TOKEN_ENV: &str = "SAANS_ALERTS_TEST_TG_TOKEN";
const TG_TOKEN: &str = "12345:TEST-TOKEN-must-never-be-logged";

/// Set once per process, before any gateway reads them: setting the
/// environment while another thread reads it is not safe.
fn env() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        std::env::set_var(HOOK_KEY_ENV, HOOK_KEY);
        std::env::set_var(TG_TOKEN_ENV, TG_TOKEN);
    });
}

fn sha256_hex(s: &str) -> String {
    Sha256::digest(s.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect()
}

// ----------------------------------------------------------- fake servers

#[derive(Debug, Clone)]
struct Hit {
    path: String,
    headers: Vec<(String, String)>,
    body: Value,
    raw: Vec<u8>,
}

/// Records every request; answers Telegram-shaped JSON. `getUpdates`
/// returns whatever has been queued.
#[derive(Clone, Default)]
struct Sink {
    hits: Arc<Mutex<Vec<Hit>>>,
    updates: Arc<Mutex<VecDeque<Value>>>,
}

impl Sink {
    async fn start() -> (SocketAddr, Sink) {
        let sink = Sink::default();
        let app = Router::new().fallback(any(record)).with_state(sink.clone());
        let l = TcpListener::bind("127.0.0.1:0").await.unwrap();
        let addr = l.local_addr().unwrap();
        tokio::spawn(async move { axum::serve(l, app).await.unwrap() });
        (addr, sink)
    }

    fn hits(&self, path_contains: &str) -> Vec<Hit> {
        self.hits
            .lock()
            .unwrap()
            .iter()
            .filter(|h| h.path.contains(path_contains))
            .cloned()
            .collect()
    }

    /// Wait (real time) until `n` requests to `path_contains` have arrived:
    /// delivery is asynchronous by design, so a test has to wait for it.
    async fn wait_for(&self, path_contains: &str, n: usize) -> Vec<Hit> {
        for _ in 0..100 {
            let h = self.hits(path_contains);
            if h.len() >= n {
                return h;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
        panic!(
            "expected {n} request(s) to {path_contains}, got {:?}",
            self.hits(path_contains)
                .iter()
                .map(|h| &h.path)
                .collect::<Vec<_>>()
        );
    }
}

async fn record(State(s): State<Sink>, uri: Uri, headers: HeaderMap, body: Bytes) -> Json<Value> {
    s.hits.lock().unwrap().push(Hit {
        path: uri.path().to_string(),
        headers: headers
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_str().unwrap_or("").to_string()))
            .collect(),
        body: serde_json::from_slice(&body).unwrap_or(Value::Null),
        raw: body.to_vec(),
    });
    if uri.path().ends_with("/getUpdates") {
        let queued: Vec<Value> = s.updates.lock().unwrap().drain(..).collect();
        if queued.is_empty() {
            tokio::time::sleep(Duration::from_millis(100)).await;
        }
        return Json(json!({ "ok": true, "result": queued }));
    }
    Json(json!({ "ok": true, "result": {} }))
}

// ------------------------------------------------------- authenticated HTTP

async fn call(
    addr: SocketAddr,
    method: &str,
    path: &str,
    token: Option<&str>,
    body: Option<&str>,
) -> (u16, Value) {
    let mut s = TcpStream::connect(addr).await.unwrap();
    let auth = token
        .map(|t| format!("Authorization: Bearer {t}\r\n"))
        .unwrap_or_default();
    let body = body.unwrap_or("");
    let req = format!(
        "{method} {path} HTTP/1.1\r\nHost: {addr}\r\n{auth}Content-Type: application/json\r\n\
         Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
        body.len()
    );
    s.write_all(req.as_bytes()).await.unwrap();
    let mut raw = Vec::new();
    tokio::time::timeout(Duration::from_secs(10), s.read_to_end(&mut raw))
        .await
        .expect("response")
        .unwrap();
    let text = String::from_utf8_lossy(&raw).to_string();
    let status: u16 = text.split_whitespace().nth(1).unwrap().parse().unwrap();
    let body = text.split("\r\n\r\n").nth(1).unwrap_or("");
    // Non-chunked JSON is all these endpoints return; tolerate chunking anyway.
    let json = serde_json::from_str(body)
        .or_else(|_| serde_json::from_str(body.lines().nth(1).unwrap_or("")))
        .unwrap_or(Value::Null);
    (status, json)
}

// ---------------------------------------------------------------- set-up

const ADMIN: &str = "tok-admin";
const WARDEN: &str = "tok-warden-lab";
const WARDEN_HALL: &str = "tok-warden-hall";
const OPS: &str = "tok-ops";
const DOOR: &str = "tok-occupancy";

fn tokens_toml() -> String {
    [
        ("admin-bot", "admin", ADMIN, ""),
        ("warden", "safety", WARDEN, "spaces = [\"lab\"]\n"),
        (
            "hall-warden",
            "safety",
            WARDEN_HALL,
            "spaces = [\"hall\"]\n",
        ),
        ("it", "ops", OPS, ""),
        ("door", "occupancy", DOOR, ""),
    ]
    .iter()
    .map(|(n, r, t, sp)| {
        format!(
            "[[token]]\nname = \"{n}\"\nrole = \"{r}\"\nsha256 = \"{}\"\n{sp}",
            sha256_hex(t)
        )
    })
    .collect()
}

const SPACES: &str =
    "[[space]]\nid = \"lab\"\nname = \"Wet lab\"\nvitals = true\nlinks = [{ rx = 1, tx = 2 }]\n\
                      [[space]]\nid = \"hall\"\nname = \"Hall\"\nlinks = [{ rx = 3, tx = 2 }]\n";

fn gateway_config(dir: &Path, extra: &str) -> saans_gateway::GatewayConfig {
    let text = format!(
        "{}[auth]\nrequired = true\n{}{extra}",
        config_toml(dir, SPACES),
        tokens_toml()
    );
    saans_gateway::GatewayConfig::from_toml(&text).unwrap_or_else(|e| panic!("{e}\n{text}"))
}

/// The lab's link: an empty room long enough to calibrate, a still
/// breather long enough for a 60 s breathing baseline, 25 s with the
/// breathing band at a tenth of it -- past the 20 s at which an ongoing
/// pause is reported -- and breathing again. Sequence numbers run on,
/// because a gap would read as packet loss and take vitals away.
async fn scripted_pause(g: &mut TestGw) {
    let breathing = |seq: u32| occupied_frame(1, 2, seq, Some((14.0, 0.8)));
    g.run(30.0, |i| vec![empty_frame(1, 2, i)]).await;
    g.run(90.0, |i| vec![breathing(60 + i)]).await;
    g.run(25.0, |i| {
        let mut f = breathing(240 + i);
        f.breath_rms = 0.01;
        vec![f]
    })
    .await;
    g.run(20.0, |i| vec![breathing(290 + i)]).await;
}

fn audit_lines(dir: &Path) -> Vec<Value> {
    let mut out = Vec::new();
    for e in std::fs::read_dir(dir).unwrap().flatten() {
        let n = e.file_name().to_string_lossy().to_string();
        if n.starts_with("audit-") {
            for l in std::fs::read_to_string(e.path()).unwrap().lines() {
                out.push(serde_json::from_str(l).unwrap());
            }
        }
    }
    out
}

// ------------------------------------------------------------------ tests

#[tokio::test]
async fn exit_gate_a_pause_pages_once_is_acked_audited_and_survives_a_restart() {
    env();
    let (sink, hooks) = Sink::start().await;
    let dir = temp_dir("alerts-gate");
    let routes = format!(
        "[webhook.pager]\nurl = \"http://{sink}/hook\"\nsecret_env = \"{HOOK_KEY_ENV}\"\n\
         [[alert_route]]\nname = \"night\"\nkinds = [\"breathing_pause\"]\nchannels = [\"webhook:pager\"]\n"
    );
    let cfg = gateway_config(&dir, &routes);
    let mut g = TestGw::start(cfg.clone()).await;
    scripted_pause(&mut g).await;

    // One page, though the pause produced two events (ongoing, then ended).
    let got = hooks.wait_for("/hook", 1).await;
    tokio::time::sleep(Duration::from_millis(300)).await;
    assert_eq!(hooks.hits("/hook").len(), 1, "one alert, one page");
    let page = &got[0];
    assert_eq!(page.body["kind"], "breathing_pause");
    assert_eq!(page.body["space_name"], "Wet lab");
    // Signed with the key from the environment.
    let sig = page
        .headers
        .iter()
        .find(|(k, _)| k == "x-saans-signature")
        .map(|(_, v)| v.clone());
    let want = {
        use hmac::{Hmac, Mac};
        let mut m = <Hmac<Sha256> as Mac>::new_from_slice(HOOK_KEY.as_bytes()).unwrap();
        m.update(&page.raw);
        let hex: String = m
            .finalize()
            .into_bytes()
            .iter()
            .map(|b| format!("{b:02x}"))
            .collect();
        format!("sha256={hex}")
    };
    assert_eq!(sig.as_deref(), Some(want.as_str()));
    // No health details left the building: this route did not ask for them.
    let wire = String::from_utf8_lossy(&page.raw);
    for leak in ["evidence", "confidence", "baseline", "breathing-band"] {
        assert!(!wire.contains(leak), "{leak} in the page: {wire}");
    }

    let (st, list) = call(g.http(), "GET", "/api/alerts?open=true", Some(WARDEN), None).await;
    assert_eq!(st, 200, "{list}");
    let alerts = list["alerts"].as_array().unwrap();
    assert_eq!(alerts.len(), 1);
    let a = &alerts[0];
    assert_eq!(a["state"], "open");
    assert!(
        a["repeats"].as_u64().unwrap() >= 1,
        "the ended event updated it: {a}"
    );
    let id = a["id"].as_u64().unwrap();

    // Acknowledged through the endpoint the dashboard's button calls.
    let (st, acked) = call(
        g.http(),
        "POST",
        &format!("/api/alerts/{id}/ack"),
        Some(WARDEN),
        None,
    )
    .await;
    assert_eq!(st, 200, "{acked}");
    assert_eq!(acked["state"], "acknowledged");
    assert_eq!(acked["acked_by"], "warden");
    let audit = audit_lines(&dir);
    assert!(
        audit.iter().any(|l| l["action"] == "alert_ack"
            && l["who"] == "warden"
            && l["detail"]["alert_id"] == id),
        "no audit line: {audit:?}"
    );

    // Restart on the same data_dir.
    g.stop().await;
    let g = TestGw::start(cfg).await;
    let (st, after) = call(
        g.http(),
        "GET",
        &format!("/api/alerts/{id}"),
        Some(ADMIN),
        None,
    )
    .await;
    assert_eq!(st, 200, "{after}");
    assert_eq!(after["state"], "acknowledged");
    assert_eq!(after["acked_by"], "warden");
    assert_eq!(after["deliveries"][0]["ok"], true, "{after}");
    g.stop().await;
}

#[tokio::test]
async fn who_may_see_an_alert_and_who_may_act_on_it() {
    env();
    let (sink, _hooks) = Sink::start().await;
    let dir = temp_dir("alerts-roles");
    let routes = format!(
        "[webhook.pager]\nurl = \"http://{sink}/hook\"\n\
         [[alert_route]]\nname = \"all\"\nkinds = [\"breathing_pause\", \"link_down\"]\nchannels = [\"webhook:pager\"]\n"
    );
    let mut g = TestGw::start(gateway_config(&dir, &routes)).await;
    scripted_pause(&mut g).await;
    g.run(10.0, |_| vec![]).await; // the lab's link goes silent: link_down
    let (_, all) = call(g.http(), "GET", "/api/alerts", Some(ADMIN), None).await;
    let id_of = |kind: &str| {
        all["alerts"]
            .as_array()
            .unwrap()
            .iter()
            .find(|a| a["kind"] == kind)
            .unwrap_or_else(|| panic!("{kind}: {all}"))["id"]
            .as_u64()
            .unwrap()
    };
    let (pause, down) = (id_of("breathing_pause"), id_of("link_down"));
    let ack = |id: u64| format!("/api/alerts/{id}/ack");
    let get = |id: u64| format!("/api/alerts/{id}");

    // No credential: 401.
    assert_eq!(
        call(g.http(), "GET", "/api/alerts", None, None).await.0,
        401
    );
    // Out of scope: 404, the same answer as an alert that does not exist.
    assert_eq!(
        call(g.http(), "GET", &get(pause), Some(WARDEN_HALL), None)
            .await
            .0,
        404
    );
    assert_eq!(
        call(g.http(), "POST", &ack(pause), Some(WARDEN_HALL), None)
            .await
            .0,
        404
    );
    assert_eq!(
        call(g.http(), "GET", &get(999), Some(ADMIN), None).await.0,
        404
    );
    // A role that may not see health alerts: 404 for the pause, absent from its list.
    for t in [OPS, DOOR] {
        assert_eq!(
            call(g.http(), "GET", &get(pause), Some(t), None).await.0,
            404,
            "{t}"
        );
        assert_eq!(
            call(g.http(), "POST", &ack(pause), Some(t), None).await.0,
            404,
            "{t}"
        );
        let (_, l) = call(g.http(), "GET", "/api/alerts", Some(t), None).await;
        assert!(
            l["alerts"]
                .as_array()
                .unwrap()
                .iter()
                .all(|a| a["kind"] != "breathing_pause"),
            "{t}: {l}"
        );
    }
    // Ops sees the sensor fault, and may not act on it: 403, not 404.
    assert_eq!(
        call(g.http(), "GET", &get(down), Some(OPS), None).await.0,
        200
    );
    let (st, why) = call(g.http(), "POST", &ack(down), Some(OPS), None).await;
    assert_eq!(st, 403, "{why}");
    // The warden in scope may, once; the second time is a conflict.
    assert_eq!(
        call(g.http(), "POST", &ack(pause), Some(WARDEN), None)
            .await
            .0,
        200
    );
    assert_eq!(
        call(g.http(), "POST", &ack(pause), Some(WARDEN), None)
            .await
            .0,
        409
    );
    let (st, r) = call(
        g.http(),
        "POST",
        &format!("/api/alerts/{pause}/resolve"),
        Some(WARDEN),
        Some(r#"{"note": "checked, was asleep"}"#),
    )
    .await;
    assert_eq!(
        (st, r["state"].as_str(), r["resolution_note"].as_str()),
        (200, Some("resolved"), Some("checked, was asleep"))
    );
    let audit = audit_lines(&dir);
    assert!(audit
        .iter()
        .any(|l| l["action"] == "alert_resolve" && l["who"] == "warden"));
    assert!(
        !audit
            .iter()
            .any(|l| l["action"] == "alert_ack" && l["who"] == "it"),
        "a refused act is not an act"
    );
    // /api/health reports the service.
    let (_, h) = call(g.http(), "GET", "/api/health", Some(ADMIN), None).await;
    assert_eq!(h["alerts"]["routes"], 1, "{h}");
    g.stop().await;
}

#[tokio::test]
async fn an_unacknowledged_alert_escalates_on_the_gateway_clock() {
    env();
    let (sink, hooks) = Sink::start().await;
    let dir = temp_dir("alerts-escalate");
    let routes = format!(
        "[webhook.pager]\nurl = \"http://{sink}/hook\"\n[webhook.boss]\nurl = \"http://{sink}/boss\"\n\
         [[alert_route]]\nname = \"night\"\nkinds = [\"breathing_pause\"]\nchannels = [\"webhook:pager\"]\n\
         ack_timeout_min = 1\nescalate_to = [\"webhook:boss\"]\n"
    );
    let mut g = TestGw::start(gateway_config(&dir, &routes)).await;
    scripted_pause(&mut g).await;
    hooks.wait_for("/hook", 1).await;
    assert!(hooks.hits("/boss").is_empty(), "not yet");
    // Nobody acknowledges; a minute passes on the gateway's clock.
    g.run(60.0, |i| {
        vec![occupied_frame(1, 2, 330 + i, Some((14.0, 0.8)))]
    })
    .await;
    let boss = hooks.wait_for("/boss", 1).await;
    assert!(
        boss[0].body["reason"]
            .as_str()
            .unwrap()
            .contains("not acknowledged within 1 min"),
        "{:?}",
        boss[0].body
    );
    assert_eq!(boss[0].body["escalation_level"], 1);
    g.stop().await;
}

#[tokio::test]
async fn telegram_buttons_acknowledge_for_allowed_users_only_and_the_token_stays_secret() {
    env();
    let (tg, telegram) = Sink::start().await;
    let dir = temp_dir("alerts-telegram");
    let routes = format!(
        "[telegram.wardens]\nbot_token_env = \"{TG_TOKEN_ENV}\"\nchat_id = -100\napi_base = \"http://{tg}\"\n\
         allowed_user_ids = [42]\n\
         [[alert_route]]\nname = \"night\"\nkinds = [\"breathing_pause\"]\nchannels = [\"telegram:wardens\"]\n"
    );
    let mut g = TestGw::start(gateway_config(&dir, &routes)).await;
    scripted_pause(&mut g).await;
    let sent = telegram.wait_for("/sendMessage", 1).await;
    let msg = &sent[0].body;
    assert_eq!(msg["chat_id"], -100);
    let buttons = &msg["reply_markup"]["inline_keyboard"][0];
    let ack_data = buttons[0]["callback_data"].as_str().unwrap().to_string();
    assert!(ack_data.starts_with("saans:ack:"), "{msg}");
    assert!(
        !msg["text"].as_str().unwrap().contains("baseline"),
        "no evidence by default: {msg}"
    );
    let id: u64 = ack_data.rsplit(':').next().unwrap().parse().unwrap();

    let press = |user: i64, chat: i64, n: i64| {
        json!({ "update_id": n, "callback_query": {
            "id": format!("cq{n}"), "from": { "id": user }, "message": { "chat": { "id": chat } }, "data": ack_data } })
    };
    // Someone not on the list, then someone in another chat: both refused.
    telegram
        .updates
        .lock()
        .unwrap()
        .push_back(press(7, -100, 1));
    telegram
        .updates
        .lock()
        .unwrap()
        .push_back(press(42, -999, 2));
    let answers = telegram.wait_for("/answerCallbackQuery", 2).await;
    assert!(
        answers
            .iter()
            .all(|a| !a.body["text"].as_str().unwrap().contains("acknowledged")),
        "{answers:?}"
    );
    let (_, a) = call(
        g.http(),
        "GET",
        &format!("/api/alerts/{id}"),
        Some(ADMIN),
        None,
    )
    .await;
    assert_eq!(a["state"], "open");
    // The allowed user, in the alert chat.
    telegram
        .updates
        .lock()
        .unwrap()
        .push_back(press(42, -100, 3));
    let answers = telegram.wait_for("/answerCallbackQuery", 3).await;
    assert!(
        answers[2].body["text"]
            .as_str()
            .unwrap()
            .contains("acknowledged"),
        "{:?}",
        answers[2].body
    );
    let (_, a) = call(
        g.http(),
        "GET",
        &format!("/api/alerts/{id}"),
        Some(ADMIN),
        None,
    )
    .await;
    assert_eq!(
        (a["state"].as_str(), a["acked_by"].as_str()),
        (Some("acknowledged"), Some("telegram:42"))
    );
    let audit = audit_lines(&dir);
    assert!(audit.iter().any(|l| l["action"] == "alert_ack"
        && l["who"] == "telegram:42"
        && l["detail"]["via"] == "telegram"));
    assert!(audit
        .iter()
        .any(|l| l["action"] == "denied" && l["who"] == "telegram:7"));

    // The bot token is in Telegram's URLs, as it must be, and nowhere else.
    assert!(telegram.hits("/sendMessage")[0].path.contains(TG_TOKEN));
    let (_, h) = call(g.http(), "GET", "/api/health", Some(ADMIN), None).await;
    for text in [
        a.to_string(),
        h.to_string(),
        std::fs::read_to_string(dir.join("alerts.jsonl")).unwrap(),
    ] {
        assert!(
            !text.contains(TG_TOKEN) && !text.contains("TEST-TOKEN"),
            "token leaked: {text}"
        );
    }
    g.stop().await;
}
