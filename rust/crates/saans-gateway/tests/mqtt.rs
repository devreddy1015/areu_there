//! The MQTT publisher against a real broker (rumqttd, embedded): what
//! reaches the wire, with which retain flags; the last will; reconnection;
//! and that a broker which is down or never answers costs ingest nothing.

mod support;

use rumqttc::{AsyncClient, Event, MqttOptions, Packet, QoS};
use saans_gateway::outputs::OutputMsg;
use serde_json::{json, Value};
use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};
use support::*;
use tokio::sync::{broadcast, mpsc, watch};

/// A consented two-link rest room and a hall without consent whose node
/// is misconfigured and sends breathing anyway.
const SPACES: &str = "[[space]]\nid = \"rest\"\nname = \"Rest room\"\nvitals = true\n\
                      links = [{ rx = 3, tx = 4 }, { rx = 5, tx = 4 }]\n\
                      [[space]]\nid = \"hall\"\nvitals = false\nlinks = [{ rx = 1, tx = 255 }]\n";

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .unwrap()
        .local_addr()
        .unwrap()
        .port()
}

/// An embedded broker on `port`, for the rest of the process.
async fn start_broker(port: u16) {
    let addr: SocketAddr = format!("127.0.0.1:{port}").parse().unwrap();
    let server = rumqttd::ServerSettings {
        name: "v4".into(),
        listen: addr,
        tls: None,
        next_connection_delay_ms: 1,
        connections: rumqttd::ConnectionSettings {
            connection_timeout_ms: 60_000,
            max_payload_size: 1 << 20,
            max_inflight_count: 100,
            auth: None,
            external_auth: None,
            dynamic_filters: true,
        },
    };
    let config = rumqttd::Config {
        router: rumqttd::RouterConfig {
            max_connections: 100,
            max_outgoing_packet_count: 200,
            max_segment_size: 1 << 20,
            max_segment_count: 10,
            ..Default::default()
        },
        v4: Some(HashMap::from([("1".to_string(), server)])),
        ..Default::default()
    };
    std::thread::spawn(move || rumqttd::Broker::new(config).start().unwrap());
    let deadline = Instant::now() + Duration::from_secs(10);
    while tokio::net::TcpStream::connect(addr).await.is_err() {
        assert!(Instant::now() < deadline, "broker did not start");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

#[derive(Debug, Clone)]
struct Got {
    topic: String,
    payload: String,
    retain: bool,
}

impl Got {
    fn json(&self) -> Value {
        serde_json::from_str(&self.payload)
            .unwrap_or_else(|e| panic!("{}: {e}: {}", self.topic, self.payload))
    }
}

/// A client subscribed to everything; returns once the broker has acked
/// the subscription, so nothing published after this is missed.
async fn subscribe(port: u16) -> mpsc::UnboundedReceiver<Got> {
    static N: AtomicUsize = AtomicUsize::new(0);
    let mut o = MqttOptions::new(
        format!("sub-{}", N.fetch_add(1, Ordering::Relaxed)),
        "127.0.0.1",
        port,
    );
    o.set_max_packet_size(1 << 20, 1 << 20);
    let (client, mut el) = AsyncClient::new(o, 16);
    client.subscribe("#", QoS::AtLeastOnce).await.unwrap();
    let (tx, rx) = mpsc::unbounded_channel();
    let (acked_tx, acked_rx) = tokio::sync::oneshot::channel();
    tokio::spawn(async move {
        let _client = client;
        let mut acked = Some(acked_tx);
        loop {
            match el.poll().await {
                Ok(Event::Incoming(Packet::SubAck(_))) => {
                    if let Some(a) = acked.take() {
                        let _ = a.send(());
                    }
                }
                Ok(Event::Incoming(Packet::Publish(p))) => {
                    let got = Got {
                        topic: p.topic,
                        payload: String::from_utf8_lossy(&p.payload).into_owned(),
                        retain: p.retain,
                    };
                    if tx.send(got).is_err() {
                        return;
                    }
                }
                Ok(_) => {}
                Err(_) => tokio::time::sleep(Duration::from_millis(100)).await,
            }
        }
    });
    tokio::time::timeout(Duration::from_secs(10), acked_rx)
        .await
        .expect("subscription acked")
        .unwrap();
    rx
}

/// Everything received until `quiet` passes with nothing new.
async fn drain(rx: &mut mpsc::UnboundedReceiver<Got>, quiet: Duration) -> Vec<Got> {
    let mut out = Vec::new();
    while let Ok(Some(g)) = tokio::time::timeout(quiet, rx.recv()).await {
        out.push(g);
    }
    out
}

/// The first message matching `pred`, within `within`.
async fn wait_for(
    rx: &mut mpsc::UnboundedReceiver<Got>,
    within: Duration,
    pred: impl Fn(&Got) -> bool,
) -> Got {
    let deadline = tokio::time::Instant::now() + within;
    loop {
        match tokio::time::timeout_at(deadline, rx.recv()).await {
            Ok(Some(g)) if pred(&g) => return g,
            Ok(Some(_)) => {}
            _ => panic!("no matching message within {within:?}"),
        }
    }
}

async fn until(within: Duration, what: &str, f: impl Fn() -> bool) {
    let deadline = Instant::now() + within;
    while !f() {
        assert!(Instant::now() < deadline, "timed out waiting for {what}");
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
}

fn mqtt_cfg(name: &str, port: u16, extra: &str) -> saans_gateway::GatewayConfig {
    config(
        name,
        &format!("{SPACES}[mqtt]\nhost = \"127.0.0.1\"\nport = {port}\n{extra}\n"),
    )
}

fn night(i: u32, occupied: bool) -> Vec<saans_dsp::features::FeatureFrame> {
    if occupied {
        vec![
            occupied_frame(3, 4, i, Some((14.0, 0.9))),
            empty_frame(5, 4, i),
            occupied_frame(1, 255, i, Some((17.25, 0.9))),
        ]
    } else {
        vec![
            empty_frame(3, 4, i),
            empty_frame(5, 4, i),
            empty_frame(1, 255, i),
        ]
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn the_topic_set_retain_flags_and_what_never_reaches_the_broker() {
    let port = free_port();
    start_broker(port).await;
    let mut live = subscribe(port).await;
    let mut g = TestGw::start(mqtt_cfg(
        "mqtt-topics",
        port,
        "vitals_topics = true\nhomeassistant_discovery = true",
    ))
    .await;
    let status = g.gw.shared.mqtt.clone();
    until(Duration::from_secs(10), "connect", || status.connected()).await;
    g.run(5.0, |i| night(i, false)).await;
    g.run(15.0, |i| night(10 + i, true)).await;
    let mut seen = drain(&mut live, Duration::from_millis(500)).await;

    // A subscriber arriving now gets exactly the retained state.
    let mut late = subscribe(port).await;
    let retained = drain(&mut late, Duration::from_millis(500)).await;
    let mut topics: Vec<&str> = retained.iter().map(|g| g.topic.as_str()).collect();
    topics.sort();
    assert_eq!(
        topics,
        [
            "homeassistant/binary_sensor/saans-gateway/hall/config",
            "homeassistant/binary_sensor/saans-gateway/rest/config",
            "saans/v1/gateway/status",
            "saans/v1/hall/occupancy",
            "saans/v1/rest/occupancy",
            "saans/v1/rest/vitals",
            "saans/v1/system/links",
        ],
        "no hall/vitals, no events"
    );
    assert!(retained.iter().all(|g| g.retain), "{retained:?}");
    let by = |t: &str| retained.iter().find(|g| g.topic == t).unwrap();
    assert_eq!(by("saans/v1/gateway/status").payload, "online");
    let rest = by("saans/v1/rest/occupancy").json();
    assert_eq!(rest["occupied"], true);
    assert_eq!(
        (rest["links_usable"].as_u64(), rest["links_total"].as_u64()),
        (Some(2), Some(2))
    );
    assert!(
        rest["since"].is_string() && rest["since_unix_ms"].is_i64(),
        "{rest}"
    );
    assert_eq!(by("saans/v1/rest/vitals").json()["breathing"]["bpm"], 14.0);
    let ha = by("homeassistant/binary_sensor/saans-gateway/rest/config").json();
    assert_eq!(ha["state_topic"], "saans/v1/rest/occupancy");
    assert_eq!(ha["device"]["name"], "Rest room");

    // Events are live only, and only occupancy and coverage.
    seen.extend(retained);
    let events: Vec<&Got> = seen
        .iter()
        .filter(|g| g.topic.ends_with("/event"))
        .collect();
    assert!(events.iter().all(|g| !g.retain));
    let types: Vec<String> = events
        .iter()
        .map(|g| g.json()["type"].as_str().unwrap().to_string())
        .collect();
    assert!(
        types.contains(&"occupancy_started".to_string()),
        "{types:?}"
    );
    for t in &types {
        assert!(
            [
                "occupancy_started",
                "occupancy_ended",
                "space_blind",
                "space_visible"
            ]
            .contains(&t.as_str()),
            "{t} reached the broker"
        );
    }
    // No vital sign anywhere but the consented space's vitals topic.
    for m in seen.iter().filter(|g| g.topic != "saans/v1/rest/vitals") {
        if let Ok(v) = serde_json::from_str::<Value>(&m.payload) {
            assert_eq!(
                vital_keys(&v),
                Vec::<String>::new(),
                "{}: {}",
                m.topic,
                m.payload
            );
            assert!(
                !m.payload.contains("17.25"),
                "the hall's leaked rate on {}",
                m.topic
            );
        }
    }
    assert!(!seen.iter().any(|g| g.topic == "saans/v1/hall/vitals"));

    // The flip to empty, as a facilities system would see it.
    g.run(120.0, |i| night(100 + i, false)).await;
    let off = wait_for(&mut late, Duration::from_secs(5), |g| {
        g.topic == "saans/v1/rest/occupancy"
    })
    .await;
    assert_eq!(off.json()["occupied"], false, "{}", off.payload);
    let health = g.json("/api/health").await;
    assert_eq!(health["mqtt"]["connected"], true, "{}", health["mqtt"]);
    assert_eq!(health["mqtt"]["broker"], format!("127.0.0.1:{port}"));
    g.stop().await;
    let bye = wait_for(&mut late, Duration::from_secs(5), |g| {
        g.topic == "saans/v1/gateway/status"
    })
    .await;
    assert_eq!(bye.payload, "offline", "a clean stop says so");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_killed_publisher_leaves_its_last_will() {
    let port = free_port();
    start_broker(port).await;
    let cfg = mqtt_cfg("mqtt-will", port, "");
    let (tx, rx) = broadcast::channel(16);
    let (_stop_tx, stop) = watch::channel(false);
    let status = Arc::new(saans_gateway::mqtt::Status::default());
    let task = saans_gateway::mqtt::spawn_with_status(&cfg, rx, stop, status.clone());
    until(Duration::from_secs(10), "connect", || status.connected()).await;
    let mut sub = subscribe(port).await;
    let online = wait_for(&mut sub, Duration::from_secs(5), |g| {
        g.topic == "saans/v1/gateway/status"
    })
    .await;
    assert_eq!(online.payload, "online");
    // Something to publish, so the kill lands on a connection in use.
    tx.send(OutputMsg::Event {
        space_id: "rest".into(),
        unix_ms: 0,
        value: Arc::new(json!({"type": "occupancy_started", "space_id": "rest"})),
    })
    .unwrap();
    wait_for(&mut sub, Duration::from_secs(5), |g| {
        g.topic == "saans/v1/rest/event"
    })
    .await;

    task.abort();
    let will = wait_for(&mut sub, Duration::from_secs(10), |g| {
        g.topic == "saans/v1/gateway/status"
    })
    .await;
    assert_eq!(will.payload, "offline");
    // And it stays: a client that arrives later reads "offline".
    let mut later = subscribe(port).await;
    let kept = wait_for(&mut later, Duration::from_secs(5), |g| {
        g.topic == "saans/v1/gateway/status"
    })
    .await;
    assert_eq!((kept.payload.as_str(), kept.retain), ("offline", true));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_broker_that_appears_late_gets_the_current_state() {
    let port = free_port(); // nothing listens yet
    let mut g = TestGw::start(mqtt_cfg("mqtt-late", port, "")).await;
    let status = g.gw.shared.mqtt.clone();
    g.run(5.0, |i| night(i, false)).await;
    g.run(15.0, |i| night(10 + i, true)).await;
    until(Duration::from_secs(5), "a connection error", || {
        status.json()["last_error"].is_string()
    })
    .await;
    assert!(!status.connected());
    assert!(
        status.dropped() >= 1,
        "occupancy_started fell in the outage: {}",
        status.json()
    );
    let before = status.published();

    start_broker(port).await;
    // Backoff is at most a few seconds this early.
    until(Duration::from_secs(20), "reconnect", || status.connected()).await;
    let mut sub = subscribe(port).await;
    let occ = wait_for(&mut sub, Duration::from_secs(5), |g| {
        g.topic == "saans/v1/rest/occupancy"
    })
    .await;
    assert!(occ.retain);
    assert_eq!(
        occ.json()["occupied"],
        true,
        "the state from before the broker existed"
    );
    assert!(status.published() > before);
    g.stop().await;
}

/// Wall time to push `steps` ticks of three links through a gateway, and
/// whether every datagram was ingested.
async fn ingest(cfg: saans_gateway::GatewayConfig, steps: u32) -> (Duration, u64, u64) {
    let mut g = TestGw::start(cfg).await;
    let t = Instant::now();
    let mut received = 0;
    for i in 0..steps {
        g.send_frames(&night(i, i % 40 >= 20)).await;
        received = g.advance(0.5).await;
    }
    let took = t.elapsed();
    g.stop().await;
    (took, received, 3 * steps as u64)
}

/// On one thread, so anything in the publisher that blocked instead of
/// yielding would hold up the core directly. The control is the same
/// config with the broker up: publishing has a cost of its own (the core
/// only builds the output stream when something consumes it), and that is
/// not what this test is about.
#[tokio::test(flavor = "current_thread")]
async fn a_dead_or_silent_broker_never_slows_ingest() {
    const STEPS: u32 = 1200; // ten minutes of 2 Hz ticks
    let (none, ..) = ingest(config("mqtt-none", SPACES), STEPS).await;
    let up_port = free_port();
    start_broker(up_port).await;
    let (up, got, sent) = ingest(mqtt_cfg("mqtt-up", up_port, ""), STEPS).await;
    assert_eq!(got, sent);

    // Refused: nothing listens on the port.
    let refused = mqtt_cfg("mqtt-refused", free_port(), "");
    // Silent: accepts the TCP connection and never answers CONNECT, the
    // worst case (rumqttc waits out its connect timeout).
    let hole = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let hole_port = hole.local_addr().unwrap().port();
    tokio::spawn(async move {
        let mut held = Vec::new();
        while let Ok((s, _)) = hole.accept().await {
            held.push(s);
        }
    });
    let silent = mqtt_cfg("mqtt-silent", hole_port, "");

    eprintln!("ingest of {STEPS} ticks: no MQTT {none:?}, broker up {up:?}");
    // Chosen before measuring: a publisher that waited on the broker would
    // cost at least one 1 s backoff or 5 s connect timeout. A timeout as
    // well as the assert, so a stall fails rather than crawls.
    let bound = up * 2 + Duration::from_secs(1);
    for (name, cfg) in [("refused", refused), ("silent", silent)] {
        let (took, got, sent) = tokio::time::timeout(bound, ingest(cfg, STEPS))
            .await
            .unwrap_or_else(|_| panic!("{name}: not done within {bound:?}; broker up took {up:?}"));
        eprintln!("ingest of {STEPS} ticks: broker {name} {took:?}");
        assert_eq!(got, sent, "{name}: every datagram ingested");
        assert!(took < bound, "{name}: {took:?} vs broker up {up:?}");
    }
}
