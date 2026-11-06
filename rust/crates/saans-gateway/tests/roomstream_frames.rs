//! A SIM session on the room stream end to end (spec/livedemo.md §3,
//! §4, §7): `calibrating`, then frames on the quantum clock, the
//! commands, and a live gateway that still serves only `sim` frames.
//! Every number on the stream is SIM; the one wall-clock check is the
//! frame rate.
#![cfg(feature = "roomsim")]

mod support;

use saans_gateway::shared::Mode;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use std::time::{Duration, Instant};
use support::*;

const LAB: &str = "[[space]]\nid = \"lab\"\nlinks = [{ rx = 1, tx = 2 }]\n";

/// The quanta of a session time `t`, checked to be `float(2*q) * 0.0125`.
fn q_of(t: &Value) -> u64 {
    let t = t.as_f64().expect("t is a number");
    let q = (t / 0.025).round() as u64;
    assert_eq!((2 * q) as f64 * 0.0125, t, "t = float(2q) * slot_s");
    q
}

/// The next frame; an `error` message on the way is returned instead.
async fn next(ws: &mut Ws) -> Value {
    let v = ws.next_json().await;
    assert!(v["type"] == "frame" || v["type"] == "error", "{v}");
    v
}

async fn next_frame(ws: &mut Ws) -> Value {
    let v = next(ws).await;
    assert_eq!(v["type"], "frame", "{v}");
    v
}

/// Every frame says what it is, and only SIM ever.
fn check(f: &Value, seq: u64) {
    assert_eq!(f["seq"], seq, "seq has no gaps");
    assert_eq!(
        f["mode"], "sim",
        "a live gateway's room frames are SIM (§7)"
    );
    assert_eq!(f["label"], "SIM");
    assert!(f.get("truth").is_some(), "SIM frames carry truth");
    assert!(f.get("phone").is_some(), "the bedroom has a phone");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_sim_session_calibrates_then_streams_on_the_quantum_clock() {
    let g = TestGw::start(config("room-frames", LAB)).await;
    assert_eq!(g.gw.shared.mode, Mode::Live, "the gateway itself is live");
    let origin = same_origin(g.http());
    let mut ws = Ws::connect_with(
        g.http(),
        "/api/roomsim?scene=bedroom&variant=with_walker",
        &[("Origin", &origin)],
    )
    .await;

    let state = ws.next_json().await;
    let file = std::fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes/bedroom.json"),
    )
    .unwrap();
    let sha: String = Sha256::digest(&file)
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    assert_eq!(state["type"], "state");
    assert_eq!(state["state"], "calibrating");
    assert_eq!(
        (state["mode"].as_str(), state["label"].as_str()),
        (Some("sim"), Some("SIM"))
    );
    assert_eq!(state["scene"]["id"], "bedroom");
    assert_eq!(state["scene"]["sha256"], sha);
    assert_eq!(state["scene"]["provenance"], "hand_written");
    assert_eq!(
        state["run"],
        json!({ "index": 0, "variant": "with_walker", "seed": 0, "duration_s": 120.0 })
    );

    // The run's first frame is at q = 0, then q advances 4 per frame (1x).
    let first = next_frame(&mut ws).await;
    let t_first = Instant::now();
    check(&first, 0);
    assert_eq!(q_of(&first["t"]), 0);
    assert_eq!(first["state"], "warming_up");
    assert_eq!(first["persons"], Value::Null);
    assert_eq!(first["scene"], state["scene"]);
    assert_eq!(first["run"], state["run"]);
    assert_eq!(
        (
            first["speed"].as_f64(),
            first["paused"].as_bool(),
            first["ended"].as_bool()
        ),
        (Some(1.0), Some(false), Some(false))
    );
    let mut frames = vec![first];
    let mut fifth_after = None;
    while frames.len() < 15 {
        let f = next_frame(&mut ws).await;
        check(&f, frames.len() as u64);
        assert_eq!(
            q_of(&f["t"]),
            q_of(&frames.last().unwrap()["t"]) + 4,
            "q advances by speed * 4"
        );
        frames.push(f);
        if frames.len() == 6 {
            fifth_after = Some(t_first.elapsed());
        }
    }
    // The only wall-clock check (§11): 5 frames within 2 s of the first.
    let fifth_after = fifth_after.unwrap();
    assert!(
        fifth_after <= Duration::from_secs(2),
        "5 frames took {fifth_after:?}"
    );
    // The first Locate tick is at t = window_s = 1.0: running from then.
    for f in &frames {
        let running = f["t"].as_f64().unwrap() >= 1.0;
        assert_eq!(
            f["state"],
            if running { "running" } else { "warming_up" },
            "{}",
            f["t"]
        );
        assert_eq!(f["persons"].is_null(), !running);
    }
    let mut seq = frames.len() as u64;

    // An unknown command is answered, naming it, and the session goes on.
    ws.send_text(r#"{"cmd":"stop"}"#).await;
    ws.send_text(r#"{"cmd":"speed","value":3}"#).await;
    let mut errors = Vec::new();
    while errors.len() < 2 {
        let v = next(&mut ws).await;
        if v["type"] == "error" {
            assert_eq!(
                (v["mode"].as_str(), v["label"].as_str()),
                (Some("sim"), Some("SIM"))
            );
            errors.push(v["param"].as_str().unwrap().to_string());
        } else {
            check(&v, seq);
            seq += 1;
        }
    }
    assert_eq!(errors, ["cmd", "value"]);

    // Pause: q stops; the frame that says so repeats the last t, then a
    // heartbeat a second later with the same t and nothing new.
    ws.send_text(r#"{"cmd":"pause"}"#).await;
    let mut last = next_frame(&mut ws).await;
    check(&last, seq);
    seq += 1;
    let paused = loop {
        let f = next_frame(&mut ws).await;
        check(&f, seq);
        seq += 1;
        if f["paused"] == true {
            break f;
        }
        last = f;
    };
    assert_eq!(paused["t"], last["t"], "pausing does not move t");
    assert_eq!(paused["effective_speed"], Value::Null);
    let beat = next_frame(&mut ws).await;
    check(&beat, seq);
    seq += 1;
    assert_eq!(
        (beat["t"].clone(), beat["paused"].clone()),
        (paused["t"].clone(), json!(true))
    );
    assert_eq!(
        beat["new"],
        json!({ "persons": [], "phone": [] }),
        "a heartbeat has nothing new"
    );

    // Speed 2 and play: q advances by 8 per frame.
    ws.send_text(r#"{"cmd":"speed","value":2}"#).await;
    ws.send_text(r#"{"cmd":"play"}"#).await;
    let mut prev = loop {
        let f = next_frame(&mut ws).await;
        check(&f, seq);
        seq += 1;
        if f["paused"] == false {
            assert_eq!(q_of(&f["t"]), q_of(&paused["t"]) + 8);
            break f;
        }
    };
    for _ in 0..5 {
        let f = next_frame(&mut ws).await;
        check(&f, seq);
        seq += 1;
        assert_eq!(f["speed"], 2.0);
        assert_eq!(
            q_of(&f["t"]),
            q_of(&prev["t"]) + 8,
            "q advances by speed * 4"
        );
        prev = f;
    }

    // Restart: a new run from t = 0, seq still counting, no new
    // calibration.
    ws.send_text(r#"{"cmd":"restart"}"#).await;
    let restarted = loop {
        let f = next_frame(&mut ws).await;
        check(&f, seq);
        seq += 1;
        if f["run"]["index"] == 1 {
            break f;
        }
        assert_eq!(f["run"]["index"], 0);
    };
    assert_eq!(restarted["t"], 0.0);
    assert_eq!(restarted["speed"], 2.0, "restart keeps the speed");
    assert_eq!(restarted["state"], "warming_up");
    let after = next_frame(&mut ws).await;
    check(&after, seq);
    assert_eq!(q_of(&after["t"]), 8);
    ws.close().await;
    g.stop().await;
}

/// Gateway stop ends a session at once, mid-calibration too (the hall's
/// takes seconds): its socket is closed with 1001 (going away), with no
/// frame first.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn stopping_the_gateway_closes_the_room_stream() {
    let g = TestGw::start(config("room-stop", LAB)).await;
    let origin = same_origin(g.http());
    let mut ws = Ws::connect_with(
        g.http(),
        "/api/roomsim?scene=hall-l-shaped&variant=with_walker",
        &[("Origin", &origin)],
    )
    .await;
    assert_eq!(ws.next_json().await["state"], "calibrating");
    let asked = Instant::now();
    let stopping = tokio::spawn(g.stop());
    match ws.next_message().await {
        WsMsg::Close { code, .. } => assert_eq!(code, Some(1001)),
        WsMsg::Text(t) => panic!("still calibrating, yet: {t}"),
    }
    let took = asked.elapsed();
    assert!(
        took < Duration::from_secs(1),
        "closed {took:?} after the stop"
    );
    stopping.await.unwrap();
}

/// §3.2: a client that accepts no frame for 10 s is closed with 1008.
/// The gateway learns what a client took from its pongs (the kernel takes
/// megabytes of frames for a client that stopped reading), so a client
/// that reads but answers no ping is, to it, one that stopped reading.
#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_client_that_takes_nothing_for_10_s_is_closed_with_1008() {
    let g = TestGw::start(config("room-stall", LAB)).await;
    let origin = same_origin(g.http());
    let mut ws = Ws::connect_with(
        g.http(),
        "/api/roomsim?scene=bedroom",
        &[("Origin", &origin)],
    )
    .await;
    assert_eq!(ws.next_json().await["state"], "calibrating");
    // Read as a browser does, answering pings: kept.
    let mut seq = 0;
    while seq < 15 {
        check(&next_frame(&mut ws).await, seq);
        seq += 1;
    }
    let stopped = Instant::now();
    let (mut pings, mut last_t) = (0, 0.0);
    let code = loop {
        match ws.next_raw().await {
            (0x1, text) => {
                let f: Value = serde_json::from_slice(&text).unwrap();
                check(&f, seq);
                seq += 1;
                last_t = f["t"].as_f64().unwrap();
            }
            (0x9, payload) => {
                assert_eq!(payload.len(), 8, "a numbered ping");
                pings += 1;
            }
            (0x8, payload) => break u16::from_be_bytes([payload[0], payload[1]]),
            (op, _) => panic!("unexpected opcode {op}"),
        }
    };
    let waited = stopped.elapsed();
    assert_eq!(code, 1008);
    assert!(pings >= 9, "{pings} pings left unanswered");
    // Not before 10 s without a pong (a lower bound cannot flake), and
    // long before the run would end: at 1x, SIM time runs no faster than
    // wall time.
    assert!(waited >= Duration::from_secs(9), "closed after {waited:?}");
    assert!(last_t < 30.0, "closed only at t = {last_t}");
    g.stop().await;
}
