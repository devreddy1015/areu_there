//! MQTT publisher (SAANS_PLAN.md §10.2): occupancy for facilities systems
//! and Home Assistant, from the same privacy-filtered stream every other
//! output reads (`outputs.rs`).
//!
//! | topic | retained | QoS | what |
//! |---|---|---|---|
//! | `{base}/gateway/status` | yes | 1 | `online`, or `offline` (also the last will) |
//! | `{base}/{space}/occupancy` | yes | 1 | `occupied` true / false / null, `since`, links usable of total; on change |
//! | `{base}/{space}/event` | no | 1 | occupancy and coverage events |
//! | `{base}/{space}/vitals` | yes | 0 | only with `vitals_topics`, only for `vitals = true` spaces; every 10 s |
//! | `{base}/system/links` | yes | 0 | link health, no vitals keys; every 10 s |
//! | `homeassistant/binary_sensor/{client}/{space}/config` | yes | 1 | discovery, if enabled |
//!
//! The audience is facilities (SAANS_PLAN §10: "occupied / empty /
//! unknown. Never vitals, never health alerts"), so every space topic but
//! `vitals` is filtered exactly as the `occupancy` role sees the API, by
//! the same functions ([`privacy::space_for`], [`privacy::event_for`]).
//! Health alerts and their events never reach the broker. Vitals have a
//! topic of their own so broker ACLs can keep them from occupancy clients.
//!
//! §10.2 lists a `confidence` for occupancy. The fusion decides occupancy
//! by a vote of links and computes none; the only such number is the one
//! occupancy events carry, `link_confidence` in saans-health, a lookup on
//! links usable (1 → 0.6, 2 → 0.8, 3+ → 0.9) that its own comment calls a
//! heuristic. NOTES.md §12 shows what an unmeasured number labelled
//! confidence invites, so the payload carries its input instead: links
//! usable of links configured.
//!
//! A dead broker must never stall ingest. The core never waits on this
//! task (a broadcast channel: a consumer that falls behind loses messages,
//! counted). This task never waits on the broker: it hands messages to
//! the client with `try_publish`, and only while connected. Retained state
//! is cached here and republished on every (re)connect, so an outage loses
//! the events it spanned (counted) and nothing else.
//!
//! The connection is plain TCP (rumqttc is built without TLS), so config
//! validation lets a password or vitals go only to a broker on this
//! machine; reaching a campus broker means bridging from a local one.

use crate::config::{GatewayConfig, MqttConfig};
use crate::outputs::OutputMsg;
use crate::privacy;
use crate::roles::{Principal, Role};
use crate::timefmt::iso8601;
use rumqttc::{AsyncClient, Event, EventLoop, LastWill, MqttOptions, Outgoing, Packet, QoS};
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::io;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use tokio::sync::{broadcast, watch};
use tokio::task::JoinHandle;

/// Home Assistant's default discovery prefix.
pub const HA_PREFIX: &str = "homeassistant";
/// Vitals and link health change every tick; nobody needs them at 2 Hz.
pub const VITALS_EVERY_MS: i64 = 10_000;
pub const LINKS_EVERY_MS: i64 = 10_000;
/// Larger payloads are not sent (counted): rumqttc treats an oversize
/// outgoing packet as a connection error, so one would drop the connection.
pub const MAX_PAYLOAD: usize = 128 * 1024;
/// A broker that loses the connection uncleanly fires the last will after
/// 1.5 keep-alives.
const KEEP_ALIVE: Duration = Duration::from_secs(30);
const BACKOFF_MIN: Duration = Duration::from_secs(1);
const BACKOFF_MAX: Duration = Duration::from_secs(30);
/// Requests waiting for the event loop. Steady state is a few per space
/// per minute; a full queue means the broker is not keeping up.
const QUEUE: usize = 256;
/// How long shutdown waits for `offline` and DISCONNECT to go out.
const FLUSH: Duration = Duration::from_secs(2);

/// Renders `occupied: null` as HA's "unknown" (a payload of `None`).
const HA_VALUE_TEMPLATE: &str =
    "{{ 'None' if value_json.occupied is none else ('ON' if value_json.occupied else 'OFF') }}";

/// One message for the broker.
#[derive(Debug, Clone, PartialEq)]
pub struct Msg {
    pub topic: String,
    pub payload: String,
    pub qos: QoS,
    pub retain: bool,
}

#[derive(Debug, Clone)]
struct SpaceInfo {
    id: String,
    name: String,
    vitals: bool,
}

/// What to publish, given the output stream. No I/O, so the topic set,
/// the retain flags and the privacy filtering are tested without a broker.
pub struct Publisher {
    base: String,
    client_id: String,
    discovery: bool,
    vitals_topics: bool,
    utc_offset_min: i32,
    spaces: Vec<SpaceInfo>,
    /// Who the occupancy and event topics are for.
    audience: Principal,
    /// Per space: the occupancy value last seen, and since when (Unix ms).
    since: BTreeMap<String, (Value, i64)>,
    vitals_sent_ms: BTreeMap<String, i64>,
    links_sent_ms: Option<i64>,
    /// The latest message for every retained topic, for the next connect.
    retained: BTreeMap<String, Msg>,
}

impl Publisher {
    /// `None` when `[mqtt]` is not configured.
    pub fn new(cfg: &GatewayConfig) -> Option<Self> {
        let m = cfg.mqtt.as_ref()?;
        Some(Self {
            base: m.base_topic.clone(),
            client_id: m.client_id.clone(),
            discovery: m.homeassistant_discovery,
            vitals_topics: m.vitals_topics,
            utc_offset_min: cfg.utc_offset_minutes,
            spaces: cfg
                .spaces
                .iter()
                .map(|s| SpaceInfo {
                    id: s.id.clone(),
                    name: GatewayConfig::space_name(s).to_string(),
                    vitals: s.vitals,
                })
                .collect(),
            audience: Principal {
                name: "mqtt".into(),
                role: Role::Occupancy,
                spaces: None,
            },
            since: BTreeMap::new(),
            vitals_sent_ms: BTreeMap::new(),
            links_sent_ms: None,
            retained: BTreeMap::new(),
        })
    }

    pub fn status_topic(&self) -> String {
        format!("{}/gateway/status", self.base)
    }

    fn topic(&self, space: &str, leaf: &str) -> String {
        format!("{}/{space}/{leaf}", self.base)
    }

    /// What a fresh connection needs, in order: `online`, discovery, and
    /// the latest value of every retained topic.
    pub fn on_connect(&self) -> Vec<Msg> {
        let mut out = vec![Msg {
            topic: self.status_topic(),
            payload: "online".into(),
            qos: QoS::AtLeastOnce,
            retain: true,
        }];
        if self.discovery {
            out.extend(self.spaces.iter().map(|s| self.discovery_msg(s)));
        }
        out.extend(self.retained.values().cloned());
        out
    }

    /// Messages to send now for one output message. Retained topics are
    /// cached whether or not anything is sent.
    pub fn on_output(&mut self, msg: &OutputMsg) -> Vec<Msg> {
        let mut out = Vec::new();
        match msg {
            OutputMsg::States { unix_ms, spaces } => {
                for (id, v) in spaces.iter() {
                    let Some(info) = self.spaces.iter().find(|s| &s.id == id).cloned() else {
                        continue;
                    };
                    out.extend(self.occupancy(&info, v, *unix_ms));
                    out.extend(self.vitals(&info, v, *unix_ms));
                }
            }
            OutputMsg::System { unix_ms, links, .. } => {
                let mut links = (**links).clone();
                privacy::redact_value(&mut links, false);
                let payload = json!({
                    "time": iso8601(*unix_ms, self.utc_offset_min),
                    "unix_ms": unix_ms,
                    "spaces": links,
                });
                let (m, _) = self.cache(
                    format!("{}/system/links", self.base),
                    payload,
                    QoS::AtMostOnce,
                );
                if self
                    .links_sent_ms
                    .is_none_or(|t| unix_ms - t >= LINKS_EVERY_MS)
                {
                    self.links_sent_ms = Some(*unix_ms);
                    out.push(m);
                }
            }
            OutputMsg::Event {
                space_id, value, ..
            } => {
                if !self.spaces.iter().any(|s| &s.id == space_id) {
                    return out;
                }
                if let Some(e) = privacy::event_for(value, &self.audience) {
                    out.push(Msg {
                        topic: self.topic(space_id, "event"),
                        payload: e.to_string(),
                        qos: QoS::AtLeastOnce,
                        retain: false,
                    });
                }
            }
        }
        out
    }

    fn occupancy(&mut self, info: &SpaceInfo, v: &Value, unix_ms: i64) -> Option<Msg> {
        let seen = privacy::space_for(v, &self.audience)?;
        let occupied = seen.get("occupied").cloned().unwrap_or(Value::Null);
        let since = match self.since.get(&info.id) {
            Some((was, t)) if *was == occupied => *t,
            _ => unix_ms,
        };
        self.since
            .insert(info.id.clone(), (occupied.clone(), since));
        let payload = json!({
            "occupied": occupied,
            "since": iso8601(since, self.utc_offset_min),
            "since_unix_ms": since,
            "links_usable": seen.get("links_usable"),
            "links_total": seen.get("links_total"),
        });
        let (m, changed) = self.cache(self.topic(&info.id, "occupancy"), payload, QoS::AtLeastOnce);
        changed.then_some(m)
    }

    fn vitals(&mut self, info: &SpaceInfo, v: &Value, unix_ms: i64) -> Option<Msg> {
        // Both: the space's consent as configured, and as the state that
        // passed the consent gate says.
        let consented = info.vitals && v.get("vitals").and_then(Value::as_bool) == Some(true);
        if !(self.vitals_topics && consented) {
            return None;
        }
        let mut payload = privacy::vitals_only(v);
        payload.insert("time".into(), iso8601(unix_ms, self.utc_offset_min).into());
        payload.insert("unix_ms".into(), unix_ms.into());
        payload.insert("notice".into(), crate::hub::ADVISORY.into());
        let (m, _) = self.cache(
            self.topic(&info.id, "vitals"),
            Value::Object(payload),
            QoS::AtMostOnce,
        );
        let due = self
            .vitals_sent_ms
            .get(&info.id)
            .is_none_or(|t| unix_ms - t >= VITALS_EVERY_MS);
        if due {
            self.vitals_sent_ms.insert(info.id.clone(), unix_ms);
        }
        due.then_some(m)
    }

    /// Store the latest value of a retained topic; also says whether it
    /// differs from the one before.
    fn cache(&mut self, topic: String, payload: Value, qos: QoS) -> (Msg, bool) {
        let m = Msg {
            topic: topic.clone(),
            payload: payload.to_string(),
            qos,
            retain: true,
        };
        let changed = self
            .retained
            .insert(topic, m.clone())
            .is_none_or(|old| old.payload != m.payload);
        (m, changed)
    }

    fn discovery_msg(&self, s: &SpaceInfo) -> Msg {
        let node = ha_id(&self.client_id);
        let obj = ha_id(&s.id);
        let state = self.topic(&s.id, "occupancy");
        let payload = json!({
            "name": "Occupancy",
            "unique_id": format!("{node}_{obj}_occupancy"),
            "device_class": "occupancy",
            "state_topic": state,
            "value_template": HA_VALUE_TEMPLATE,
            "json_attributes_topic": state,
            "availability_topic": self.status_topic(),
            "payload_available": "online",
            "payload_not_available": "offline",
            "qos": 1,
            "device": {
                "identifiers": [format!("{node}_{obj}")],
                "name": s.name,
                "manufacturer": "Saans",
                "model": "WiFi sensing space",
                "sw_version": env!("CARGO_PKG_VERSION"),
            },
        });
        Msg {
            topic: format!("{HA_PREFIX}/binary_sensor/{node}/{obj}/config"),
            payload: payload.to_string(),
            qos: QoS::AtLeastOnce,
            retain: true,
        }
    }
}

/// Home Assistant ids allow `[A-Za-z0-9_-]`; space ids may also hold `.`.
fn ha_id(s: &str) -> String {
    s.chars()
        .map(|c| {
            if c.is_ascii_alphanumeric() || c == '-' || c == '_' {
                c
            } else {
                '_'
            }
        })
        .collect()
}

/// Connection state and counters, for `/api/health` and `/metrics`.
#[derive(Debug, Default)]
pub struct Status {
    connected: AtomicBool,
    connects: AtomicU64,
    /// Handed to the client. Delivery at QoS 1 is the broker's business.
    published: AtomicU64,
    /// Events that arrived while disconnected, and anything a full queue
    /// refused. Retained topics skipped while disconnected are not counted:
    /// they are republished on connect.
    dropped: AtomicU64,
    too_large: AtomicU64,
    /// Output messages this consumer fell too far behind to see.
    lagged: AtomicU64,
    last_error: Mutex<Option<String>>,
}

impl Status {
    pub fn connected(&self) -> bool {
        self.connected.load(Ordering::SeqCst)
    }

    pub fn published(&self) -> u64 {
        self.published.load(Ordering::Relaxed)
    }

    pub fn dropped(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn json(&self) -> Value {
        json!({
            "connected": self.connected(),
            "connects": self.connects.load(Ordering::Relaxed),
            "published": self.published(),
            "dropped": self.dropped(),
            "too_large": self.too_large.load(Ordering::Relaxed),
            "lagged": self.lagged.load(Ordering::Relaxed),
            "last_error": *self.last_error.lock().unwrap_or_else(|p| p.into_inner()),
        })
    }
}

/// Startup check, like the alert channels': the password variable named
/// in the config must be set. Its value is read once, by [`spawn`].
pub fn require_env(m: &MqttConfig) -> io::Result<()> {
    match m.password_env.as_deref() {
        Some(var) if std::env::var_os(var).is_none_or(|v| v.is_empty()) => Err(io::Error::new(
            io::ErrorKind::NotFound,
            format!("mqtt.password_env names {var}, which is not set"),
        )),
        _ => Ok(()),
    }
}

/// Start publishing; the counters are lost. See [`spawn_with_status`].
pub fn spawn(
    cfg: &GatewayConfig,
    rx: broadcast::Receiver<OutputMsg>,
    stop: watch::Receiver<bool>,
) -> JoinHandle<()> {
    spawn_with_status(cfg, rx, stop, Arc::default())
}

/// Start publishing, counting into `status`. Called by `live.rs` when
/// `[mqtt]` is configured. One task: aborting it drops the connection
/// without DISCONNECT, which is what makes the broker send the last will.
pub fn spawn_with_status(
    cfg: &GatewayConfig,
    rx: broadcast::Receiver<OutputMsg>,
    stop: watch::Receiver<bool>,
    status: Arc<Status>,
) -> JoinHandle<()> {
    let publisher = Publisher::new(cfg).expect("mqtt::spawn is only called with [mqtt] configured");
    let m = cfg.mqtt.as_ref().expect("checked above");
    let (client, eventloop) = AsyncClient::new(options(m, &publisher.status_topic()), QUEUE);
    let (conn_tx, conn_rx) = watch::channel(0u64);
    let broker = format!("{}:{}", m.host, m.port);
    tokio::spawn(async move {
        tokio::join!(
            drive(eventloop, broker, conn_tx, status.clone(), stop.clone()),
            feed(publisher, client, rx, conn_rx, status, stop),
        );
    })
}

fn options(m: &MqttConfig, status_topic: &str) -> MqttOptions {
    let mut o = MqttOptions::new(&m.client_id, &m.host, m.port);
    o.set_keep_alive(KEEP_ALIVE)
        .set_clean_session(true)
        .set_max_packet_size(2 * MAX_PAYLOAD, 2 * MAX_PAYLOAD)
        .set_last_will(LastWill::new(
            status_topic,
            "offline",
            QoS::AtLeastOnce,
            true,
        ));
    if let Some(user) = &m.username {
        // The client keeps the credentials for reconnects, so this is the
        // one place a secret is held. Never format `MqttOptions` with
        // `{:?}`: its Debug prints them.
        let password = m
            .password_env
            .as_deref()
            .and_then(|var| std::env::var(var).ok())
            .unwrap_or_default();
        o.set_credentials(user, password);
    }
    o
}

/// Own the connection: poll, reconnect with backoff, and tell `feed` about
/// each new connection (a generation number on `conn`).
async fn drive(
    mut el: EventLoop,
    broker: String,
    conn: watch::Sender<u64>,
    status: Arc<Status>,
    mut stop: watch::Receiver<bool>,
) {
    let mut backoff = BACKOFF_MIN;
    let mut generation = 0u64;
    loop {
        let ev = tokio::select! {
            _ = stop.wait_for(|s| *s) => break,
            ev = el.poll() => ev,
        };
        match ev {
            Ok(Event::Incoming(Packet::ConnAck(_))) => {
                // Requests left from the last connection are stale; `feed`
                // republishes every retained topic from its cache.
                el.pending.clear();
                backoff = BACKOFF_MIN;
                status.connected.store(true, Ordering::SeqCst);
                status.connects.fetch_add(1, Ordering::Relaxed);
                generation += 1;
                conn.send_replace(generation);
                crate::log(&format!("MQTT connected to {broker}"));
            }
            Ok(_) => {}
            Err(e) => {
                let msg = e.to_string();
                let was_connected = status.connected.swap(false, Ordering::SeqCst);
                {
                    let mut last = status.last_error.lock().unwrap_or_else(|p| p.into_inner());
                    // One line per outage and per new kind of failure, not
                    // one per retry.
                    if was_connected || last.as_deref() != Some(msg.as_str()) {
                        crate::log(&format!("MQTT {broker}: {msg}; retrying with backoff"));
                    }
                    *last = Some(msg);
                }
                tokio::select! {
                    _ = stop.wait_for(|s| *s) => break,
                    _ = tokio::time::sleep(backoff) => {}
                }
                backoff = (backoff * 2).min(BACKOFF_MAX);
            }
        }
    }
    // On stop, `feed` queues `offline` and DISCONNECT if connected; drive
    // the loop until they are out.
    if status.connected() {
        let _ = tokio::time::timeout(FLUSH, async {
            loop {
                match el.poll().await {
                    Ok(Event::Outgoing(Outgoing::Disconnect)) | Err(_) => break,
                    Ok(_) => {}
                }
            }
        })
        .await;
        status.connected.store(false, Ordering::SeqCst);
    }
}

/// Turn the output stream into messages and hand them to the client.
async fn feed(
    mut p: Publisher,
    client: AsyncClient,
    mut rx: broadcast::Receiver<OutputMsg>,
    mut conn: watch::Receiver<u64>,
    status: Arc<Status>,
    mut stop: watch::Receiver<bool>,
) {
    loop {
        tokio::select! {
            _ = stop.wait_for(|s| *s) => break,
            r = conn.changed() => {
                if r.is_err() {
                    break;
                }
                for m in p.on_connect() {
                    send(&client, &status, &m);
                }
            }
            r = rx.recv() => match r {
                Ok(msg) => {
                    for m in p.on_output(&msg) {
                        if status.connected() {
                            send(&client, &status, &m);
                        } else if !m.retain {
                            status.dropped.fetch_add(1, Ordering::Relaxed);
                        }
                    }
                }
                Err(broadcast::error::RecvError::Lagged(n)) => {
                    status.lagged.fetch_add(n, Ordering::Relaxed);
                }
                Err(broadcast::error::RecvError::Closed) => break,
            },
        }
    }
    if status.connected() {
        let _ = client.try_publish(p.status_topic(), QoS::AtLeastOnce, true, "offline");
        let _ = client.try_disconnect();
    }
}

fn send(client: &AsyncClient, status: &Status, m: &Msg) {
    if m.payload.len() > MAX_PAYLOAD {
        status.too_large.fetch_add(1, Ordering::Relaxed);
        return;
    }
    match client.try_publish(m.topic.as_str(), m.qos, m.retain, m.payload.as_bytes()) {
        Ok(()) => status.published.fetch_add(1, Ordering::Relaxed),
        Err(_) => status.dropped.fetch_add(1, Ordering::Relaxed),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    const T: i64 = 1_790_000_000_000;

    fn cfg(mqtt: &str) -> GatewayConfig {
        GatewayConfig::from_toml(&format!(
            "[[space]]\nid = \"lab\"\nname = \"Wet lab\"\nvitals = true\nlinks = [{{ rx = 1, tx = 2 }}]\n\
             [[space]]\nid = \"hall.b\"\nlinks = [{{ rx = 3, tx = 2 }}]\n\
             [mqtt]\n{mqtt}\n"
        ))
        .unwrap()
    }

    /// Two spaces as `space_json` publishes them: the lab consented and
    /// has a breathing estimate and a health alert; the hall did not.
    fn states(unix_ms: i64, lab_occupied: Value) -> OutputMsg {
        let lab = json!({
            "space_id": "lab", "name": "Wet lab", "vitals": true, "t_s": 1.0,
            "occupied": lab_occupied, "links_usable": 1, "links_total": 1,
            "breathing": {"bpm": 14.0, "confidence": 0.9}, "bpm_median_5min": 14.0,
            "in_breathing_pause": false, "motion_index": 0.1,
            "active_alerts": ["breathing_pause", "blind"],
            "alerts": [{"name": "breathing_pause", "evidence": "x"}, {"name": "blind"}],
        });
        let hall = json!({
            "space_id": "hall.b", "name": "hall.b", "vitals": false, "t_s": 1.0,
            "occupied": null, "links_usable": 0, "links_total": 1,
            "active_alerts": [], "alerts": [],
        });
        OutputMsg::States {
            unix_ms,
            spaces: Arc::new(vec![("lab".into(), lab), ("hall.b".into(), hall)]),
        }
    }

    fn topics(ms: &[Msg]) -> Vec<&str> {
        ms.iter().map(|m| m.topic.as_str()).collect()
    }

    fn payload(ms: &[Msg], topic: &str) -> Value {
        let m = ms
            .iter()
            .find(|m| m.topic == topic)
            .unwrap_or_else(|| panic!("no {topic}"));
        serde_json::from_str(&m.payload).unwrap()
    }

    #[test]
    fn occupancy_is_retained_and_sent_only_on_change() {
        let mut p = Publisher::new(&cfg("")).unwrap();
        let first = p.on_output(&states(T, json!(true)));
        assert_eq!(
            topics(&first),
            ["saans/v1/lab/occupancy", "saans/v1/hall.b/occupancy"]
        );
        assert!(first.iter().all(|m| m.retain && m.qos == QoS::AtLeastOnce));
        let lab = payload(&first, "saans/v1/lab/occupancy");
        assert_eq!(lab["occupied"], true);
        assert_eq!(lab["since_unix_ms"], T);
        assert_eq!(
            (lab["links_usable"].as_u64(), lab["links_total"].as_u64()),
            (Some(1), Some(1))
        );
        assert_eq!(
            payload(&first, "saans/v1/hall.b/occupancy")["occupied"],
            Value::Null
        );

        assert!(
            p.on_output(&states(T + 500, json!(true))).is_empty(),
            "nothing changed"
        );
        let flip = p.on_output(&states(T + 1000, json!(false)));
        assert_eq!(topics(&flip), ["saans/v1/lab/occupancy"]);
        assert_eq!(
            payload(&flip, "saans/v1/lab/occupancy")["since_unix_ms"],
            T + 1000
        );
    }

    #[test]
    fn occupancy_topics_carry_no_vitals_and_no_health_alerts() {
        let mut p = Publisher::new(&cfg("vitals_topics = true")).unwrap();
        let ms = p.on_output(&states(T, json!(true)));
        for m in ms.iter().filter(|m| !m.topic.ends_with("/vitals")) {
            for word in ["breath", "bpm", "heart", "pause", "evidence"] {
                assert!(
                    !m.payload.contains(word),
                    "{word} on {}: {}",
                    m.topic,
                    m.payload
                );
            }
        }
    }

    #[test]
    fn vitals_only_for_a_consented_space_and_only_when_asked() {
        let mut off = Publisher::new(&cfg("")).unwrap();
        let ms = off.on_output(&states(T, json!(true)));
        assert!(
            !topics(&ms).iter().any(|t| t.ends_with("/vitals")),
            "vitals_topics is off"
        );

        let mut on = Publisher::new(&cfg("vitals_topics = true")).unwrap();
        let ms = on.on_output(&states(T, json!(true)));
        let vitals: Vec<&str> = topics(&ms)
            .into_iter()
            .filter(|t| t.ends_with("/vitals"))
            .collect();
        assert_eq!(vitals, ["saans/v1/lab/vitals"], "never the hall's");
        let v = payload(&ms, "saans/v1/lab/vitals");
        assert_eq!(v["breathing"]["bpm"], 14.0);
        assert_eq!(v["notice"], crate::hub::ADVISORY);
        assert!(
            v.get("occupied").is_none() && v.get("alerts").is_none(),
            "{v}"
        );
        let m = ms.iter().find(|m| m.topic.ends_with("/vitals")).unwrap();
        assert!(m.retain);

        // A config that says vitals but a state that does not (the consent
        // gate's view wins), and the reverse, both publish nothing.
        let mut lab = states(T, json!(true));
        if let OutputMsg::States { spaces, .. } = &mut lab {
            let mut v = (**spaces).clone();
            v[0].1["vitals"] = json!(false);
            v[1].1["vitals"] = json!(true);
            *spaces = Arc::new(v);
        }
        let mut p = Publisher::new(&cfg("vitals_topics = true")).unwrap();
        assert!(!topics(&p.on_output(&lab))
            .iter()
            .any(|t| t.ends_with("/vitals")));
    }

    #[test]
    fn vitals_and_links_are_rate_limited_but_cached() {
        let mut p = Publisher::new(&cfg("vitals_topics = true")).unwrap();
        let sys = |ms| OutputMsg::System {
            unix_ms: ms,
            links: Arc::new(
                json!([{"space_id": "lab", "links": [{"rx": 1, "breathing_conf": 0.4}]}]),
            ),
            health: Arc::new(json!({})),
        };
        let count = |ms: &[Msg], leaf: &str| ms.iter().filter(|m| m.topic.ends_with(leaf)).count();
        let mut sent = (0, 0);
        for i in 0..40 {
            let t = T + i * 500;
            sent.0 += count(&p.on_output(&states(t, json!(true))), "/vitals");
            let l = p.on_output(&sys(t));
            sent.1 += count(&l, "/system/links");
            if let Some(m) = l.first() {
                assert!(
                    !m.payload.contains("breath"),
                    "links carry no vitals: {}",
                    m.payload
                );
            }
        }
        // 20 s at 2 Hz: at t = 0 and t = 10 s.
        assert_eq!(sent, (2, 2));
        // The cache holds the latest of each, for the next connect.
        let c = p.on_connect();
        assert_eq!(payload(&c, "saans/v1/lab/vitals")["unix_ms"], T + 39 * 500);
        assert_eq!(
            payload(&c, "saans/v1/system/links")["unix_ms"],
            T + 39 * 500
        );
    }

    #[test]
    fn events_are_what_the_occupancy_role_sees() {
        let mut p = Publisher::new(&cfg("")).unwrap();
        let ev = |ty: &str| OutputMsg::Event {
            space_id: "lab".into(),
            unix_ms: T,
            value: Arc::new(
                json!({"type": ty, "space_id": "lab", "severity": "check", "bpm_at_start": 14.0}),
            ),
        };
        let started = p.on_output(&ev("occupancy_started"));
        assert_eq!(topics(&started), ["saans/v1/lab/event"]);
        assert!(!started[0].retain && started[0].qos == QoS::AtLeastOnce);
        assert!(
            !started[0].payload.contains("bpm"),
            "{}",
            started[0].payload
        );
        for ty in [
            "breathing_pause",
            "inactivity",
            "fall_suspected",
            "link_down",
        ] {
            assert!(
                p.on_output(&ev(ty)).is_empty(),
                "{ty} must not reach the broker"
            );
        }
    }

    #[test]
    fn connect_sends_online_discovery_then_the_cached_state() {
        let mut p =
            Publisher::new(&cfg("homeassistant_discovery = true\nclient_id = \"gw 1\"")).unwrap();
        p.on_output(&states(T, json!(true)));
        let c = p.on_connect();
        assert_eq!(c[0].topic, "saans/v1/gateway/status");
        assert_eq!((c[0].payload.as_str(), c[0].retain), ("online", true));
        assert_eq!(
            topics(&c[1..3]),
            [
                "homeassistant/binary_sensor/gw_1/lab/config",
                "homeassistant/binary_sensor/gw_1/hall_b/config"
            ]
        );
        let d = payload(&c, "homeassistant/binary_sensor/gw_1/lab/config");
        assert_eq!(d["device_class"], "occupancy");
        assert_eq!(d["state_topic"], "saans/v1/lab/occupancy");
        assert_eq!(d["availability_topic"], "saans/v1/gateway/status");
        assert_eq!(d["unique_id"], "gw_1_lab_occupancy");
        assert_eq!(d["device"]["name"], "Wet lab");
        assert!(c.iter().all(|m| m.retain));
        assert!(topics(&c).contains(&"saans/v1/lab/occupancy"));
    }

    #[test]
    fn the_last_will_is_the_status_topic() {
        let c = cfg("base_topic = \"campus/saans\"");
        let p = Publisher::new(&c).unwrap();
        let o = options(c.mqtt.as_ref().unwrap(), &p.status_topic());
        let will = o.last_will().unwrap();
        assert_eq!(will.topic, "campus/saans/gateway/status");
        assert_eq!(&will.message[..], b"offline");
        assert!(will.retain && will.qos == QoS::AtLeastOnce);
    }

    #[test]
    fn the_ha_template_maps_null_to_unknown() {
        // HA's binary_sensor reads a rendered `None` as unknown; anything
        // else that is not ON/OFF is an error in its log.
        assert!(HA_VALUE_TEMPLATE.contains("'None' if value_json.occupied is none"));
        assert_eq!(ha_id("lab.3-b_x"), "lab_3-b_x");
    }
}
