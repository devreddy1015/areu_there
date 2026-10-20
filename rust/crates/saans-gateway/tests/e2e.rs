//! End to end: a gateway in this process on ephemeral ports, fed real UDP
//! datagrams and read over real HTTP/1.1 and WebSocket connections, with
//! its manual clock standing in for the 0.5 s timer.

mod support;

use saans_dsp::features::{FeatureFrame, FLAG_HEART_VALID};
use saans_gateway::pcap::PcapWriter;
use saans_gateway::shared::Shared;
use saans_gateway::{replay, GatewayConfig, ReplayOptions};
use serde_json::{json, Value};
use std::net::{Ipv4Addr, SocketAddrV4};
use std::path::Path;
use std::time::Duration;
use support::*;

/// A consented rest room with two links.
const REST: &str = "[[space]]\nid = \"rest\"\nname = \"Rest room\"\nvitals = true\n\
                    links = [{ rx = 3, tx = 4 }, { rx = 5, tx = 4 }]\n";
/// A single-link space without vitals.
const LAB: &str = "[[space]]\nid = \"lab\"\nlinks = [{ rx = 1, tx = 2 }]\n";

fn has_type(events: &Value, ty: &str) -> bool {
    events.as_array().unwrap().iter().any(|e| e["type"] == ty)
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn occupied_space_with_consent_shows_breathing() {
    let mut g = TestGw::start(config("occupied", REST)).await;
    // 5 s empty, then 15 s with a still, breathing person whom link 3<-4
    // sees while 5<-4 sits in a blind spot (it still votes "empty").
    g.run(5.0, |i| vec![empty_frame(3, 4, i), empty_frame(5, 4, i)]).await;
    g.run(15.0, |i| vec![occupied_frame(3, 4, 10 + i, Some((14.0, 0.9))), empty_frame(5, 4, 10 + i)]).await;

    let spaces = g.json("/api/spaces").await;
    assert_eq!(spaces.as_array().unwrap().len(), 1);
    let s = &spaces[0];
    assert_eq!(s["space_id"], "rest");
    assert_eq!(s["name"], "Rest room");
    assert_eq!(s["vitals"], true);
    assert_eq!(s["occupied"], true);
    assert_eq!((s["links_usable"].as_u64(), s["links_total"].as_u64()), (Some(2), Some(2)));
    let b = &s["breathing"];
    assert_eq!(b["bpm"], 14.0, "{s}");
    assert!((b["confidence"].as_f64().unwrap() - 0.9).abs() < 1e-6);
    assert_eq!(b["link"], json!({"rx": 3, "tx": 4}));
    assert_eq!(s["bpm_median_5min"], 14.0);
    assert_eq!(s["raw_mode"], false);
    assert_eq!(s["in_breathing_pause"], false);
    assert!(s["occupied_since_s"].as_f64().unwrap() >= T0 + 5.0);
    assert_eq!(s["t_s"], g.t);

    let one = g.json("/api/spaces/rest").await;
    assert_eq!(one["state"]["occupied"], true);
    assert_eq!(one["state"]["breathing"]["bpm"], 14.0);
    assert!(has_type(&one["events"], "occupancy_started"), "{}", one["events"]);

    let started = g.events_of(Some("rest"), "occupancy_started").await;
    assert_eq!(started.len(), 1);
    let e = &started[0];
    assert_eq!(e["space_id"], "rest");
    assert_eq!(e["severity"], "info");
    assert!(e["evidence"].as_str().is_some_and(|s| !s.is_empty()));
    assert!(e["time"].as_str().unwrap().ends_with('Z'), "utc_offset_minutes = 0: {e}");
    assert!(e["unix_ms"].as_i64().unwrap() > 1_700_000_000_000);

    let links = g.json("/api/links").await;
    assert_eq!(links[0]["space_id"], "rest");
    assert_eq!(links[0]["links_usable"], 2);
    for r in links[0]["links"].as_array().unwrap() {
        assert_eq!(r["up"], true, "{r}");
        assert_eq!(r["frames"], 40);
        assert_eq!(r["rate_hz"], 20.0);
        assert_eq!(r["calib_age_h"], 1.0);
    }

    let r = g.get("/api/spaces/nope").await;
    assert_eq!(r.status, 404);
    assert!(r.json()["error"].is_string());
    assert_eq!(g.get("/api/nope").await.status, 404);
    assert_eq!(g.get("/api/spaces").await.header("cache-control"), Some("no-store"));
    g.stop().await;
}

/// No breathing or heart value, and no key naming one, anywhere a
/// vitals-off space's data can be read: the API, the WebSocket, /metrics
/// and the event log on disk.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn vitals_off_space_never_exposes_vital_signs() {
    const READING: &str = "[[space]]\nid = \"reading\"\nvitals = false\ninactivity_alert_min = 1.0\n\
                           links = [{ rx = 1, tx = 255 }]\n";
    // A misconfigured node sends what the space never consented to.
    // Exact in f32 and unlike any time or count the gateway reports.
    const BPM: f32 = 17.296875;
    const HEART: f32 = 71.53125;
    const BAND: f32 = 0.123;
    fn leaky(seq: u32) -> FeatureFrame {
        let mut f = occupied_frame(1, 255, seq, Some((BPM, 0.9)));
        f.flags |= FLAG_HEART_VALID;
        f.heart_bpm = HEART;
        f.heart_conf = 0.8;
        f.breath_rms = BAND;
        f
    }
    let leaked = [BPM as f64, HEART as f64, BAND as f64];
    fn leaked_numbers(v: &Value, leaked: &[f64], path: &str, out: &mut Vec<String>) {
        match v {
            Value::Number(n) if n.as_f64().is_some_and(|x| leaked.contains(&x)) => out.push(format!("{path} = {n}")),
            Value::Object(m) => m.iter().for_each(|(k, x)| leaked_numbers(x, leaked, &format!("{path}.{k}"), out)),
            Value::Array(a) => a.iter().enumerate().for_each(|(i, x)| leaked_numbers(x, leaked, &format!("{path}[{i}]"), out)),
            _ => {}
        }
    }
    let check = |what: &str, v: &Value| {
        let keys = vital_keys(v);
        assert!(keys.is_empty(), "{what} exposes vital-sign keys {keys:?}");
        let mut nums = Vec::new();
        leaked_numbers(v, &leaked, "$", &mut nums);
        assert!(nums.is_empty(), "{what} exposes vital-sign values {nums:?}");
    };

    let cfg = config("novitals", READING);
    let data_dir = cfg.data_dir.clone();
    let mut g = TestGw::start(cfg).await;
    let mut ws = Ws::connect(g.http(), "/api/live").await;
    g.run(5.0, |i| vec![empty_frame(1, 255, i)]).await;
    // 90 s present and still: occupancy, then inactivity after 1 min.
    g.run(90.0, |i| vec![leaky(10 + i)]).await;
    // The link goes silent: link down, then the space is blind.
    g.run(20.0, |_| vec![]).await;

    let inactivity = g.events_of(Some("reading"), "inactivity").await;
    assert_eq!(inactivity.len(), 1, "the detectors ran");
    assert!(inactivity[0].get("still_min").is_some());
    assert!(inactivity[0].get("breathing_detected").is_none(), "{}", inactivity[0]);
    for ty in ["occupancy_started", "link_down", "space_blind"] {
        assert_eq!(g.events_of(Some("reading"), ty).await.len(), 1, "{ty}");
    }

    for path in [
        "/api/spaces",
        "/api/spaces/reading",
        "/api/events?limit=1000",
        "/api/events?space=reading&limit=1000",
        "/api/links",
        "/api/health",
    ] {
        check(path, &g.json(path).await);
    }
    let s = &g.json("/api/spaces").await[0];
    assert_eq!(s["vitals"], false);
    assert_eq!(s["vitals_gated_links"], json!(["1<-255"]), "the misconfigured node is named");
    let h = g.json("/api/health").await;
    assert_eq!(h["ingest"]["vitals_gated"], 180);
    assert_eq!(h["vitals_gated_links"], json!([{"space_id": "reading", "rx": 1, "tx": 255}]));

    let metrics = parse_prometheus(&g.get("/metrics").await.body).unwrap();
    for m in &metrics {
        let text = format!("{} {:?}", m.name, m.labels).to_ascii_lowercase();
        assert!(!["breath", "bpm", "heart", "pause"].iter().any(|w| text.contains(w)), "{text}");
        assert!(!leaked.contains(&m.value), "{m:?}");
    }
    assert_eq!(metric(&metrics, "saans_vitals_gated_frames_total", &[]), Some(180.0));

    // Everything the WebSocket pushed since before the first frame.
    let mut n = 0;
    loop {
        let v = ws.next_json().await;
        check("WebSocket message", &v);
        n += 1;
        if v["type"] == "event" && v["event"]["type"] == "space_blind" {
            break;
        }
    }
    assert!(n > 100, "{n} WebSocket messages");
    ws.close().await;

    g.stop().await;
    let mut lines = 0;
    for entry in std::fs::read_dir(&data_dir).unwrap() {
        let path = entry.unwrap().path();
        let name = path.file_name().unwrap().to_string_lossy().into_owned();
        if name.starts_with("events-") {
            for line in std::fs::read_to_string(&path).unwrap().lines() {
                check(&name, &serde_json::from_str(line).unwrap());
                lines += 1;
            }
        }
    }
    assert_eq!(lines, 4, "occupancy_started, inactivity, link_down, space_blind");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn silent_link_raises_link_down() {
    let mut g = TestGw::start(config("linkdown", LAB)).await;
    g.run(10.0, |i| vec![empty_frame(1, 2, i)]).await;
    assert!(g.events_of(None, "link_down").await.is_empty());
    assert_eq!(g.json("/api/links").await[0]["links"][0]["up"], true);

    g.run(10.0, |_| vec![]).await;
    let downs = g.events_of(Some("lab"), "link_down").await;
    assert_eq!(downs.len(), 1, "once per outage");
    let d = &downs[0];
    assert_eq!(d["space_id"], "lab");
    assert_eq!(d["link"], json!({"rx": 1, "tx": 2}));
    assert_eq!(d["severity"], "advisory");
    let silent = d["silent_s"].as_f64().unwrap();
    assert!(silent > 5.0 && silent <= 6.0, "{silent}");
    assert!(d["evidence"].as_str().unwrap().contains("1<-2"), "{d}");
    assert!((d["t_start_s"].as_f64().unwrap() - (T0 + 9.5)).abs() < 1e-9, "last frame at T0 + 9.5 s");

    let r = &g.json("/api/links").await[0]["links"][0];
    assert_eq!(r["up"], false);
    assert!(r["silent_s"].as_f64().unwrap() > 9.0);
    let s = &g.json("/api/spaces").await[0];
    assert_eq!(s["occupied"], Value::Null, "no usable link: unknown, not empty");
    assert!(s["active_alerts"].as_array().unwrap().contains(&json!("blind")));
    let m = parse_prometheus(&g.get("/metrics").await.body).unwrap();
    assert_eq!(metric(&m, "saans_link_up", &[("space", "lab"), ("rx", "1"), ("tx", "2")]), Some(0.0));
    assert_eq!(metric(&m, "saans_space_occupied", &[("space", "lab")]), Some(-1.0));

    g.run(3.0, |i| vec![empty_frame(1, 2, 100 + i)]).await;
    assert_eq!(g.events_of(Some("lab"), "link_up").await.len(), 1);
    assert_eq!(g.json("/api/links").await[0]["links"][0]["up"], true);

    // `since` (ISO 8601 or Unix seconds) returns strictly later events.
    let later = g.json(&format!("/api/events?space=lab&since={}", d["time"].as_str().unwrap())).await;
    assert!(!has_type(&later, "link_down") && has_type(&later, "link_up"), "{later}");
    let all = g.json("/api/events?since=0").await;
    assert!(has_type(&all, "link_down"));
    assert_eq!(g.json("/api/events?limit=1").await.as_array().unwrap().len(), 1);
    assert_eq!(g.json("/api/events?space=elsewhere").await, json!([]));
    assert_eq!(g.get("/api/events?since=yesterday").await.status, 400);
    g.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn unknown_links_and_bad_datagrams_are_counted() {
    let mut g = TestGw::start(config("unknown", LAB)).await;
    g.run(5.0, |i| vec![empty_frame(1, 2, i), empty_frame(9, 9, i)]).await;
    g.send(&raw_datagram(6, 0, 0.0, 56)).await;
    let mut bad_crc = empty_frame(1, 2, 99).encode();
    bad_crc[30] ^= 1;
    g.send(&bad_crc).await;
    g.send(&empty_frame(1, 2, 99).encode()[..60]).await;
    g.send(b"not a saans datagram").await;
    g.advance(0.5).await;

    let h = g.json("/api/health").await;
    assert_eq!(h["datagrams_total"], 24);
    assert_eq!(h["ingest"]["feature_frames"], 20);
    assert_eq!(h["ingest"]["unassigned"], 11, "10 FeatureFrames from 9<-9, 1 raw datagram from node 6");
    let unknown = h["unknown_links"].as_array().unwrap();
    assert_eq!(unknown.len(), 2);
    let u = unknown.iter().find(|u| u["rx"] == 9).unwrap();
    assert_eq!((u["tx"].as_u64(), u["kind"].as_str(), u["frames"].as_u64()), (Some(9), Some("feature"), Some(10)));
    assert!(u["last_src"].as_str().unwrap().starts_with("127.0.0.1:"), "{u}");
    assert_eq!(u["first_seen_s"], T0);
    let r = unknown.iter().find(|u| u["rx"] == 6).unwrap();
    assert_eq!((r["tx"].as_u64(), r["kind"].as_str()), (Some(255), Some("raw")));
    assert_eq!(h["decode_errors_total"], 3);
    assert_eq!(h["ingest"]["errors"]["crc"], 1);
    assert_eq!(h["ingest"]["errors"]["size"], 1);
    assert_eq!(h["ingest"]["errors"]["unknown_type"], 1);
    assert_eq!(h["links_configured"], 1, "an unknown link is not adopted by any space");
    assert_eq!(h["mode"], "live");

    let m = parse_prometheus(&g.get("/metrics").await.body).unwrap();
    assert_eq!(metric(&m, "saans_unknown_link_frames_total", &[("rx", "9"), ("tx", "9"), ("kind", "feature")]), Some(10.0));
    assert_eq!(metric(&m, "saans_unknown_link_frames_total", &[("rx", "6"), ("tx", "255"), ("kind", "raw")]), Some(1.0));
    assert_eq!(metric(&m, "saans_decode_errors_total", &[("reason", "crc")]), Some(1.0));
    assert_eq!(metric(&m, "saans_datagrams_total", &[("type", "invalid")]), Some(3.0));
    g.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn raw_csi_becomes_raw_mode_feature_frames() {
    const RAW: &str = "[raw]\ncalibration_s = 2.0\n[[space]]\nid = \"lab\"\nlinks = [{ rx = 5, tx = 255 }]\n";
    let mut g = TestGw::start(config("raw", RAW)).await;
    // 10 s of raw CSI at 20 Hz, delivered 0.5 s at a time.
    let mut seq = 0u32;
    for _ in 0..20 {
        for _ in 0..10 {
            g.send(&raw_datagram(5, seq, seq as f64 / 20.0, 56)).await;
            seq += 1;
        }
        g.advance(0.5).await;
    }

    let h = g.json("/api/health").await;
    assert_eq!(h["ingest"]["raw_datagrams"], 200);
    assert_eq!(h["ingest"]["raw_feature_frames"], 20, "one FeatureFrame per 10 CSI frames (2 Hz)");
    assert_eq!(h["ingest"]["feature_frames"], 0);
    assert_eq!(h["raw_mode_links_active"], 1);
    assert_eq!(h["raw_mode_links"], json!([{"space_id": "lab", "rx": 5, "tx": 255}]));

    let s = &g.json("/api/spaces").await[0];
    assert_eq!(s["raw_mode"], true, "{s}");
    assert_eq!(s["raw_mode_links"], json!(["5<-255"]));
    let r = &g.json("/api/links").await[0]["links"][0];
    assert_eq!(r["raw_mode"], true);
    assert_eq!(r["frames"], 20);
    assert_eq!(r["up"], true);
    assert!(r["rate_hz"].as_f64().unwrap() > 15.0, "{r}");

    let ev = g.events_of(Some("lab"), "raw_mode_active").await;
    assert_eq!(ev.len(), 1, "a privacy notice, once per hour");
    assert_eq!(ev[0]["link"], json!({"rx": 5, "tx": 255}));
    assert_eq!(ev[0]["severity"], "advisory");

    let m = parse_prometheus(&g.get("/metrics").await.body).unwrap();
    assert_eq!(metric(&m, "saans_link_raw_mode", &[("space", "lab"), ("rx", "5"), ("tx", "255")]), Some(1.0));
    assert_eq!(metric(&m, "saans_raw_feature_frames_total", &[]), Some(20.0));
    assert_eq!(metric(&m, "saans_datagrams_total", &[("type", "raw")]), Some(200.0));
    g.stop().await;
}

/// Two spaces, an unknown link, a garbage datagram and one to another
/// port; 2 minutes at 2 Hz starting at Unix time 1_790_000_000.
fn write_recording(path: &Path) {
    let file = std::fs::File::create(path).unwrap();
    let mut w = PcapWriter::new(std::io::BufWriter::new(file)).unwrap();
    let gw = SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 1), 5566);
    let node = |n: u8| SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 100 + n), 5566);
    let base_ns: i64 = 1_790_000_000 * 1_000_000_000;
    for i in 0..240u32 {
        let t_ns = base_ns + i as i64 * 500_000_000;
        // Person present from 20 s to 80 s; link 5<-4 dies at 60 s.
        let present = (40..160).contains(&i);
        let a = if present { occupied_frame(3, 4, i, Some((14.0, 0.9))) } else { empty_frame(3, 4, i) };
        w.write(t_ns + 1_000_000, node(3), gw, &a.encode()).unwrap();
        if i < 120 {
            w.write(t_ns + 2_000_000, node(5), gw, &empty_frame(5, 4, i).encode()).unwrap();
        }
        w.write(t_ns + 3_000_000, node(1), gw, &occupied_frame(1, 255, i, Some((17.25, 0.9))).encode()).unwrap();
        if i % 20 == 0 {
            w.write(t_ns + 4_000_000, node(9), gw, &empty_frame(9, 9, i).encode()).unwrap();
        }
    }
    w.write(base_ns + 10_000_000_000, node(7), gw, b"garbage").unwrap();
    let other = SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 1), 9999);
    w.write(base_ns + 11_000_000_000, node(7), other, &empty_frame(3, 4, 0).encode()).unwrap();
    w.into_inner().into_inner().unwrap();
}

const REPLAY_SPACES: &str = "[[space]]\nid = \"rest\"\nvitals = true\nlinks = [{ rx = 3, tx = 4 }, { rx = 5, tx = 4 }]\n\
                             [[space]]\nid = \"hall\"\nvitals = false\nlinks = [{ rx = 1, tx = 255 }]\n";

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn replay_is_deterministic() {
    let dir = temp_dir("replay");
    let pcap = dir.join("rec.pcap");
    write_recording(&pcap);
    let cfg_text = config_toml(&dir.join("data"), REPLAY_SPACES);
    let cfg = GatewayConfig::from_toml(&cfg_text).unwrap();
    let opts = |out: &str| ReplayOptions {
        pcap: pcap.clone(),
        speed: 0.0,
        events_out: Some(dir.join(out)),
        port: Some(5566),
    };

    let s1 = replay(cfg.clone(), &opts("a.jsonl"), None).await.unwrap();
    let s2 = replay(cfg.clone(), &opts("b.jsonl"), None).await.unwrap();
    let a = std::fs::read_to_string(dir.join("a.jsonl")).unwrap();
    let b = std::fs::read_to_string(dir.join("b.jsonl")).unwrap();
    assert!(!a.is_empty());
    assert_eq!(a, b, "same recording and config, same events, byte for byte");
    assert_eq!(s1.to_string(), s2.to_string());
    assert!(!dir.join("data").exists(), "replay never touches data_dir");

    let events: Vec<Value> = a.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(events.len() as u64, s1.events);
    let of = |space: &str, ty: &str| events.iter().filter(|e| e["space_id"] == space && e["type"] == ty).count();
    assert_eq!(of("rest", "occupancy_started"), 1);
    assert_eq!(of("rest", "link_down"), 1);
    assert_eq!(of("hall", "occupancy_started"), 1);
    // Event time comes from the packet timestamps, not the replay's clock.
    let first_ms = events[0]["unix_ms"].as_i64().unwrap();
    assert!((1_790_000_000_000..1_790_000_120_000).contains(&first_ms), "{first_ms}");
    assert!(a.lines().all(|l| l.starts_with("{\"time\":\"2026-")), "{a}");
    for e in events.iter().filter(|e| e["space_id"] == "hall") {
        assert!(vital_keys(e).is_empty(), "{e}");
    }

    assert_eq!(s1.read.packets, 240 * 3 - 120 + 12 + 1);
    assert_eq!(s1.read.other_port, 1);
    assert_eq!(s1.ingest.errors.unknown_type, 1);
    assert_eq!(s1.ingest.unassigned, 12);
    assert_eq!(s1.ingest.vitals_gated, 240, "hall's node sends vitals without consent");
    // First packet at +1 ms, last at 119.5 s + 3 ms; the garbage datagram
    // was written last with an earlier timestamp and is clamped.
    assert!((s1.duration_s - 119.502).abs() < 1e-6, "{}", s1.duration_s);
    assert_eq!(s1.clock_steps_back, 1);

    // Replay with a published state (as `--replay --serve` does).
    let shared = Shared::new("replay");
    let s3 = replay(cfg.clone(), &ReplayOptions { events_out: None, ..opts("") }, Some(shared.clone())).await.unwrap();
    assert_eq!(s3.events, s1.events);
    assert_eq!(shared.query_events(None, None, 10_000).len() as u64, s1.events);
    {
        let p = shared.read();
        assert_eq!(p.spaces.iter().map(|(id, _)| id.as_str()).collect::<Vec<_>>(), ["rest", "hall"]);
        assert_eq!(p.spaces[1].1["occupied"], true);
        assert!(vital_keys(&p.spaces[1].1).is_empty());
        assert_eq!(p.health["mode"], "replay");
    }

    // And the binary: same events, the same summary on stdout.
    let cfg_path = dir.join("saans.toml");
    std::fs::write(&cfg_path, &cfg_text).unwrap();
    let out = std::process::Command::new(env!("CARGO_BIN_EXE_saans-gateway"))
        .arg("--config")
        .arg(&cfg_path)
        .arg("--replay")
        .arg(&pcap)
        .args(["--port", "5566", "--events-out"])
        .arg(dir.join("c.jsonl"))
        .output()
        .unwrap();
    assert!(out.status.success(), "{}", String::from_utf8_lossy(&out.stderr));
    assert_eq!(String::from_utf8(out.stdout).unwrap(), s1.to_string());
    assert_eq!(std::fs::read_to_string(dir.join("c.jsonl")).unwrap(), a);

    let check = std::process::Command::new(env!("CARGO_BIN_EXE_saans-gateway"))
        .arg("--check")
        .arg("--config")
        .arg(concat!(env!("CARGO_MANIFEST_DIR"), "/saans.example.toml"))
        .output()
        .unwrap();
    assert!(check.status.success());
    let typo = dir.join("typo.toml");
    std::fs::write(&typo, "[[space]]\nid = \"a\"\nvital = true\n").unwrap();
    let bad = std::process::Command::new(env!("CARGO_BIN_EXE_saans-gateway")).arg("--check").arg("-c").arg(&typo).output().unwrap();
    assert_eq!(bad.status.code(), Some(2));
    assert!(String::from_utf8_lossy(&bad.stderr).contains("unknown key `vital`"));
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn metrics_are_valid_prometheus_text() {
    let spaces = format!("{REST}{LAB}");
    let mut g = TestGw::start(config("metrics", &spaces)).await;
    g.run(5.0, |i| vec![empty_frame(3, 4, i), empty_frame(5, 4, i), empty_frame(1, 2, i)]).await;
    g.run(10.0, |i| vec![occupied_frame(3, 4, 10 + i, Some((14.0, 0.9))), empty_frame(1, 2, 10 + i), empty_frame(9, 9, i)])
        .await;

    let r = g.get("/metrics").await;
    assert_eq!(r.status, 200);
    assert!(r.header("content-type").unwrap().starts_with("text/plain; version=0.0.4"), "{:?}", r.headers);
    let m = parse_prometheus(&r.body).unwrap_or_else(|e| panic!("{e}\n{}", r.body));
    assert!(m.len() > 30, "{}", m.len());
    assert_eq!(metric(&m, "saans_space_occupied", &[("space", "rest")]), Some(1.0));
    assert_eq!(metric(&m, "saans_space_occupied", &[("space", "lab")]), Some(0.0));
    assert_eq!(metric(&m, "saans_space_links_usable", &[("space", "rest")]), Some(1.0), "5<-4 went silent");
    assert_eq!(metric(&m, "saans_space_links_total", &[("space", "rest")]), Some(2.0));
    assert_eq!(metric(&m, "saans_link_frames_total", &[("space", "rest"), ("rx", "3"), ("tx", "4")]), Some(30.0));
    assert_eq!(metric(&m, "saans_link_csi_rate_hz", &[("space", "lab"), ("rx", "1"), ("tx", "2")]), Some(20.0));
    assert_eq!(metric(&m, "saans_link_calibration_age_hours", &[("space", "lab"), ("rx", "1"), ("tx", "2")]), Some(1.0));
    assert_eq!(metric(&m, "saans_link_rssi_dbm", &[("space", "lab"), ("rx", "1"), ("tx", "2")]), Some(-55.0));
    assert_eq!(
        metric(&m, "saans_events_total", &[("space", "rest"), ("type", "occupancy_started"), ("severity", "info")]),
        Some(1.0)
    );
    assert_eq!(metric(&m, "saans_events_total", &[("space", "rest"), ("type", "link_down"), ("severity", "advisory")]), Some(1.0));
    assert_eq!(metric(&m, "saans_datagrams_total", &[("type", "feature")]), Some(90.0));
    assert!(metric(&m, "saans_gateway_uptime_seconds", &[]).unwrap() >= 0.0);
    // ADR-001: a time-series database is no place for vital signs, even
    // for a consented space.
    assert!(m.iter().all(|s| !s.name.contains("breath") && !s.name.contains("bpm") && !s.name.contains("heart")));

    // The parser is strict enough to catch what it is for.
    assert!(parse_prometheus("# TYPE a gauge\na 1\n# TYPE b gauge\nb 1\na{x=\"1\"} 2\n").is_err(), "split group");
    assert!(parse_prometheus("a 1\n# TYPE a gauge\n").is_err(), "TYPE after samples");
    assert!(parse_prometheus("a{x=\"1\\q\"} 1\n").is_err(), "bad escape");
    assert!(parse_prometheus("a one\n").is_err(), "bad value");
    assert!(parse_prometheus("a 1\na 2\n").is_err(), "duplicate series");
    g.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn websocket_streams_state_and_events() {
    let mut g = TestGw::start(config("ws", LAB)).await;
    let mut ws = Ws::connect(g.http(), "/api/live").await;
    let hello = ws.next_json().await;
    assert_eq!(hello["type"], "hello");
    assert_eq!(hello["mode"], "live");
    // The latest state straight away, before the next tick.
    let first = ws.next_json().await;
    assert_eq!(first["type"], "state");
    assert_eq!(first["t_s"], T0);
    assert_eq!(first["spaces"][0]["space_id"], "lab");
    assert_eq!(g.json("/api/health").await["ws_clients"], 1);

    g.send_frames(&[empty_frame(1, 2, 0)]).await;
    g.advance(0.5).await;
    let st = ws.until(10, |v| v["type"] == "state" && v["t_s"] == T0 + 0.5).await;
    assert_eq!(st["spaces"][0]["occupied"], false);
    assert_eq!(st["spaces"][0]["links_usable"], 1);

    g.run(8.0, |_| vec![]).await;
    let ev = ws.until(100, |v| v["type"] == "event").await;
    assert_eq!(ev["event"]["type"], "link_down");
    assert!(ev["event"]["time"].is_string() && ev["event"]["unix_ms"].is_i64());

    // A second client gets the latest state on connect.
    let mut ws2 = Ws::connect(g.http(), "/api/live").await;
    assert_eq!(ws2.next_json().await["type"], "hello");
    assert_eq!(ws2.next_json().await["t_s"], g.t);
    ws2.close().await;
    ws.close().await;
    let mut clients = Value::Null;
    for _ in 0..100 {
        clients = g.json("/api/health").await["ws_clients"].clone();
        if clients == 0 {
            break;
        }
        tokio::time::sleep(Duration::from_millis(20)).await;
    }
    assert_eq!(clients, 0);
    g.stop().await;
}

/// ADR-008 decision 12: the room demo is its own page, served from the
/// binary, under a CSP stricter than the dashboard's, and does not ship
/// without a self-containment check. This is that check, against a real
/// gateway rather than the constants.
#[cfg(feature = "roomsim")]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn room_demo_is_served_self_contained_and_draws_no_one() {
    use sha2::{Digest, Sha256};
    let g = TestGw::start(config("roomsim", LAB)).await;

    let page = g.get("/roomsim").await;
    assert_eq!(page.status, 200);
    assert!(page.header("content-type").unwrap().starts_with("text/html"));
    let csp = page.header("content-security-policy").expect("CSP header");
    assert!(csp.contains("default-src 'none'") && csp.contains("script-src 'self'"), "{csp}");
    assert!(!csp.contains("unsafe-inline") && !csp.contains("unsafe-eval"), "demo CSP must allow nothing inline: {csp}");
    assert!(csp.contains("require-trusted-types-for 'script'") && csp.contains("trusted-types 'none'"), "{csp}");
    assert_eq!(page.header("x-content-type-options"), Some("nosniff"));
    // The dashboard's own policy is untouched by the demo's existence.
    let dash = g.get("/").await;
    assert!(dash.header("content-security-policy").unwrap().contains("script-src 'unsafe-inline'"));

    // No inline style or script survives the CSP, so none may be written:
    // every <script> has a src, and every link is same-origin.
    let lower = page.body.to_ascii_lowercase();
    assert!(!lower.contains("<style") && !lower.contains(" style="), "inline style would be refused by the CSP");
    for (i, _) in lower.match_indices("<script") {
        let tag = &lower[i..i + lower[i..].find('>').unwrap()];
        assert!(tag.contains(" src="), "inline <script> at byte {i} would be refused by the CSP");
    }
    for v in link_targets(&page.body) {
        assert!((v.starts_with('/') && !v.starts_with("//")) || v.starts_with("data:"), "{v} is not same-origin");
    }
    assert_nothing_foreign("roomsim.html", &page.body);

    for (path, ctype) in [("/roomsim/roomsim.js", "text/javascript"), ("/roomsim/roomsim.css", "text/css")] {
        let r = g.get(path).await;
        assert_eq!(r.status, 200, "{path}");
        assert!(r.header("content-type").unwrap().starts_with(ctype), "{path}: {:?}", r.header("content-type"));
        assert_eq!(r.header("x-content-type-options"), Some("nosniff"), "{path}");
        assert_nothing_foreign(path, &r.body);
    }

    // Follow the page's module graph from the real served script, so a
    // version bump that misses roomsim.js fails here without this test
    // naming a version.
    let js = g.get("/roomsim/roomsim.js").await.body;
    let mut seen = 0;
    for spec in import_specifiers(&js) {
        let url = format!("/roomsim/{}", spec.trim_start_matches("./"));
        let r = g.get(&url).await;
        assert_eq!(r.status, 200, "roomsim.js imports {spec}, which is not served");
        assert!(r.header("content-type").unwrap().starts_with("text/javascript"), "{url}");
        assert!(r.header("cache-control").unwrap().contains("immutable"), "{url}: vendored files are cacheable for good");
        let dir = &url[..url.rfind('/').unwrap() + 1];
        for inner in import_specifiers(&r.body) {
            let inner_url = format!("{dir}{}", inner.trim_start_matches("./"));
            assert_eq!(g.get(&inner_url).await.status, 200, "{url} imports {inner}, which is not served");
            seen += 1;
        }
        seen += 1;
    }
    assert!(seen >= 3, "the page's module graph was not found ({seen} modules)");

    // The vendor route answers from its own allowlist (nosniff proves it
    // was this handler, not the router's fallback), whatever the path.
    let rev = import_specifiers(&js)[0].split('/').nth(2).unwrap().to_string();
    for bad in [
        format!("/roomsim/vendor/{rev}/..%2F..%2Fetc%2Fpasswd"),
        format!("/roomsim/vendor/{rev}/LICENSE"),
        format!("/roomsim/vendor/{rev}/VENDOR.md"),
        "/roomsim/vendor/0.185.0/three.module.js".to_string(),
    ] {
        let r = g.get(&bad).await;
        assert_eq!(r.status, 404, "{bad}");
        assert_eq!(r.header("x-content-type-options"), Some("nosniff"), "{bad} was not answered by the vendor route");
        assert_eq!(r.header("cache-control"), Some("no-store"), "{bad}");
    }
    assert_eq!(g.get("/roomsim/vendor/0.186.1/../../../etc/passwd").await.status, 404);

    // The scene list names each scene, its provenance and the hash of its
    // file — checked here against a digest of the file on disk.
    let list = g.get("/roomsim/scenes").await.json();
    assert_eq!(list["label"], "SCENE");
    let scenes = list["scenes"].as_array().expect("scenes");
    assert_eq!(scenes.len(), 3);
    let spec_dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes");
    const SERVED_KEYS: [&str; 7] = ["schema_version", "name", "provenance", "room", "walls", "objects", "nodes"];
    for s in scenes {
        let id = s["id"].as_str().unwrap();
        let file = std::fs::read_to_string(spec_dir.join(format!("{id}.json"))).unwrap();
        let digest: String = Sha256::digest(file.as_bytes()).iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(s["sha256"], digest, "{id}: the listed hash is not the file's");
        assert_eq!(s["provenance"], "hand_written");

        let sc = g.get(&format!("/roomsim/scenes/{id}")).await;
        assert_eq!(sc.status, 200);
        let v = sc.json();
        for k in v.as_object().unwrap().keys() {
            assert!(SERVED_KEYS.contains(&k.as_str()), "{id}: {k} is outside the allowlist");
        }
        assert!(v["nodes"].as_array().is_some_and(|n| !n.is_empty()), "{id} has nodes");
        // ADR-008 decision 3, checked by value as well as by key: no id
        // of a person or the phone appears anywhere in what is served.
        let raw: Value = serde_json::from_str(&file).unwrap();
        let mut ids: Vec<&str> = raw["people"].as_array().unwrap().iter().filter_map(|p| p["id"].as_str()).collect();
        ids.extend(raw["phone"]["id"].as_str());
        assert!(!ids.is_empty(), "{id}: fixture has no person ids, so this check would be vacuous");
        for pid in ids {
            assert!(!sc.body.contains(&format!("\"{pid}\"")), "{id}: person/phone id {pid:?} was published");
        }
    }
    assert_eq!(g.get("/roomsim/scenes/nope").await.status, 404);
    g.stop().await;
}

/// ADR-008 decision 13: a gateway built without the `roomsim` feature
/// has no room view at all.
#[cfg(not(feature = "roomsim"))]
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn room_demo_is_absent_from_a_default_build() {
    let g = TestGw::start(config("no-roomsim", LAB)).await;
    for path in ["/roomsim", "/roomsim/scenes", "/roomsim/roomsim.js"] {
        assert_eq!(g.get(path).await.status, 404, "{path} is served without the roomsim feature");
    }
    g.stop().await;
}

/// Module specifiers a served script names: `from '…'` and `import('…')`,
/// skipping comment lines.
#[cfg(feature = "roomsim")]
fn import_specifiers(js: &str) -> Vec<String> {
    let mut out = Vec::new();
    for line in js.lines() {
        let t = line.trim_start();
        if t.starts_with('*') || t.starts_with("//") || t.starts_with("/*") {
            continue;
        }
        for pat in [" from '", " from \"", "import('", "import(\""] {
            let mut rest = line;
            while let Some(i) = rest.find(pat) {
                let after = &rest[i + pat.len()..];
                let q = pat.chars().last().unwrap();
                if let Some(end) = after.find(q) {
                    out.push(after[..end].to_string());
                }
                rest = after;
            }
        }
    }
    out
}


#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn dashboard_is_served_and_self_contained() {
    let g = TestGw::start(config("dashboard", LAB)).await;
    let r = g.get("/").await;
    assert_eq!(r.status, 200);
    assert!(r.header("content-type").unwrap().starts_with("text/html"));
    let csp = r.header("content-security-policy").expect("CSP header");
    assert!(csp.contains("default-src 'none'") && csp.contains("connect-src 'self'"), "{csp}");
    let body = r.body.to_ascii_lowercase();
    // XML namespace URIs are identifiers, not addresses: nothing
    // dereferences `createElementNS("http://www.w3.org/2000/svg", ...)`,
    // and SVG cannot be built without it. The src=/href= check below is
    // what actually guarantees nothing is fetched.
    let mut stripped = body.clone();
    for ns in ["http://www.w3.org/2000/svg", "http://www.w3.org/1999/xhtml", "http://www.w3.org/1999/xlink"] {
        stripped = stripped.replace(ns, "");
    }
    for bad in ["http://", "https://", "//cdn", "@import", "url(http"] {
        assert!(!stripped.contains(bad), "external reference {bad:?} in the dashboard");
    }
    // Every src= / href= is relative or a fragment, never protocol-relative.
    for v in link_targets(&body) {
        assert!(!v.starts_with("//") && !v.contains("://"), "dashboard links to {v}");
    }
    assert!(r.body.contains("not a medical device"));
    assert!(r.body.contains("/api/live"), "the dashboard uses the live feed");
    g.stop().await;
}

/// Datagrams a buggy or hostile sender could produce must be counted and
/// dropped, never stop ingest: NaN fields (saans-health compares
/// confidences with `partial_cmp().unwrap()`) and a raw header with no
/// subcarriers (saans-dsp asserts a positive count).
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn hostile_datagrams_do_not_stop_the_gateway() {
    const SPACE: &str = "[[space]]\nid = \"rest\"\nvitals = true\n\
                         links = [{ rx = 3, tx = 4 }, { rx = 5, tx = 4 }, { rx = 7, tx = 255 }]\n";
    let mut g = TestGw::start(config("hostile", SPACE)).await;
    let good = |i: u32| vec![occupied_frame(3, 4, i, Some((14.0, 0.9))), occupied_frame(5, 4, i, Some((14.0, 0.8)))];
    g.run(3.0, good).await;

    let mut nan_conf = occupied_frame(3, 4, 100, Some((14.0, f32::NAN)));
    nan_conf.seq = 100;
    let mut nan_all = occupied_frame(5, 4, 100, Some((f32::NAN, f32::NAN)));
    nan_all.motion_energy = f32::NAN;
    nan_all.presence_threshold = f32::INFINITY;
    g.send_frames(&[nan_conf, nan_all]).await;
    g.send(&saans_gateway::raw::build(&[], 7, 0, -50, -95)).await;
    let mut no_antennas = raw_datagram(7, 1, 0.0, 56);
    no_antennas[5] = 0;
    g.send(&no_antennas).await;
    g.advance(0.5).await;
    g.advance(0.5).await;

    let before = g.json("/api/health").await;
    assert_eq!(before["t_s"], g.t, "still ticking");
    assert_eq!(before["datagrams_total"], 16);
    assert_eq!(before["decode_errors_total"], 4, "{}", before["ingest"]);
    g.run(2.0, |i| good(200 + i)).await;
    let h = g.json("/api/health").await;
    assert_eq!(h["t_s"], g.t);
    assert_eq!(h["datagrams_total"], 24);
    let s = &g.json("/api/spaces").await[0];
    assert_eq!(s["occupied"], true);
    assert_eq!(s["breathing"]["bpm"], 14.0);
    g.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn events_and_history_survive_a_restart() {
    let cfg = config("restart", LAB);
    let dir = cfg.data_dir.clone();
    let mut g = TestGw::start(cfg.clone()).await;
    g.run(5.0, |i| vec![empty_frame(1, 2, i)]).await;
    g.run(8.0, |_| vec![]).await;
    assert_eq!(g.events_of(None, "link_down").await.len(), 1);
    g.stop().await;

    let logs: Vec<_> = std::fs::read_dir(&dir)
        .unwrap()
        .map(|e| e.unwrap().path())
        .filter(|p| p.file_name().unwrap().to_string_lossy().starts_with("events-"))
        .collect();
    assert_eq!(logs.len(), 1, "{logs:?}");
    let log = std::fs::read_to_string(&logs[0]).unwrap();
    assert_eq!(log.lines().count(), 1);
    assert!(log.starts_with("{\"time\":") && log.contains("\"type\":\"link_down\""), "{log}");
    let history: Value = serde_json::from_str(&std::fs::read_to_string(dir.join("history.json")).unwrap()).unwrap();
    assert_eq!(history["version"], 1);
    assert!(history["spaces"]["lab"].is_array(), "{history}");

    let g2 = TestGw::start(cfg).await;
    let downs = g2.events_of(Some("lab"), "link_down").await;
    assert_eq!(downs.len(), 1, "yesterday's events are still listed after a restart");
    assert_eq!(g2.json("/api/health").await["notes"], json!([]));
    g2.stop().await;
}
