//! Signed FeatureFrames (ADR-012): byte-exact against the Python
//! reference's golden vectors, then through a running gateway. What is
//! accepted, what is refused and counted, and that a restart does not
//! reopen frames already accepted.

mod support;

use saans_dsp::features::{FeatureFrame, SIGNED_FRAME_SIZE};
use saans_gateway::frameauth::{self, NodeKey, SiteKey};
use saans_gateway::{ClockMode, Gateway};
use serde_json::Value;
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Once;
use support::*;

// ------------------------------------------------------------ golden bytes

fn golden_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/golden")
}

/// `NAME.kv` scalars and `NAME__X.f64` arrays (`spec/wire.md` §3).
struct Vectors {
    name: &'static str,
    kv: HashMap<String, String>,
}

impl Vectors {
    fn load(name: &'static str) -> Self {
        let path = golden_dir().join(format!("{name}.kv"));
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|e| panic!("{}: {e} (run spec/golden/generate.py)", path.display()));
        let kv = text
            .lines()
            .filter_map(|l| l.split_once(' '))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Self { name, kv }
    }

    fn a(&self, k: &str) -> Vec<f64> {
        let n: usize = self.kv[&format!("array.{k}")].parse().unwrap();
        let bytes = std::fs::read(golden_dir().join(format!("{}__{k}.f64", self.name))).unwrap();
        assert_eq!(bytes.len(), n * 8, "{}: array {k}", self.name);
        bytes
            .chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    fn bytes(&self, k: &str) -> Vec<u8> {
        self.a(k).iter().map(|&b| b as u8).collect()
    }
}

fn frame_from(v: &Vectors, i: usize) -> FeatureFrame {
    let g = |k: &str| v.a(k)[i];
    FeatureFrame {
        node_id: g("node_id") as u8,
        tx_id: g("tx_id") as u8,
        flags: g("flags") as u8,
        seq: g("seq") as u32,
        uptime_ms: g("uptime_ms") as u32,
        csi_rate_chz: g("csi_rate_chz") as u16,
        loss_permille: g("loss_permille") as u16,
        motion_energy: g("motion_energy") as f32,
        presence_threshold: g("presence_threshold") as f32,
        gross_motion: g("gross_motion") as f32,
        breath_rms: g("breath_rms") as f32,
        breathing_bpm: g("breathing_bpm") as f32,
        breathing_conf: g("breathing_conf") as f32,
        heart_bpm: g("heart_bpm") as f32,
        heart_conf: g("heart_conf") as f32,
        best_bnr: g("best_bnr") as f32,
        calib_age_s: g("calib_age_s") as u32,
        rssi_dbm: g("rssi_dbm") as i8,
        noise_floor_dbm: g("noise_floor_dbm") as i8,
        n_subcarriers: g("n_subcarriers") as u8,
        ..Default::default()
    }
}

fn hex(b: &[u8]) -> String {
    b.iter().map(|x| format!("{x:02x}")).collect()
}

#[test]
fn signed_frames_match_the_reference_byte_for_byte() {
    let v = Vectors::load("feature_codec_signed");
    let site = SiteKey::from_bytes(v.bytes("site_key").try_into().unwrap());
    let node_keys = v.bytes("node_keys");
    let (key_ids, boots, bytes) = (v.a("key_id"), v.a("boot"), v.bytes("bytes"));
    for (i, want) in bytes.chunks(SIGNED_FRAME_SIZE).enumerate() {
        let f = frame_from(&v, i);
        let key = site.node_key(f.node_id);
        assert_eq!(
            key.to_hex(),
            hex(&node_keys[32 * i..32 * (i + 1)]),
            "node key {i}"
        );
        let (key_id, boot) = (key_ids[i] as u8, boots[i] as u32);
        assert_eq!(
            frameauth::sign(&f, &key, key_id, boot)[..],
            *want,
            "frame {i}"
        );

        let keys = BTreeMap::from([(key_id, site.clone())]);
        let got = frameauth::verify(want, &keys).expect("the reference's frame verifies");
        assert_eq!((got.key_id, got.boot), (key_id, boot));
        assert_eq!(
            got.frame,
            FeatureFrame { version: 2, ..f },
            "frame {i} fields"
        );
        for j in 0..SIGNED_FRAME_SIZE {
            let mut bad = want.to_vec();
            bad[j] ^= 0x80;
            assert!(
                frameauth::verify(&bad, &keys).is_err(),
                "frame {i}, byte {j} flipped"
            );
        }
    }
}

// ----------------------------------------------------------------- gateway

const KEY_ENV: &str = "SAANS_FRAME_AUTH_TEST_KEY";
/// Hex of the site key the tests' gateways read from `KEY_ENV`.
const SITE_HEX: &str = "000102030405060708090a0b0c0d0e0f101112131415161718191a1b1c1d1e1f";

/// Set once: changing the environment while another test thread reads
/// it is not safe.
fn env() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| std::env::set_var(KEY_ENV, SITE_HEX));
}

fn site() -> SiteKey {
    SiteKey::from_hex(SITE_HEX).unwrap()
}

const LAB: &str = "[[space]]\nid = \"lab\"\nvitals = true\nlinks = [{ rx = 1, tx = 2 }]\n";

fn signed_cfg(name: &str) -> saans_gateway::GatewayConfig {
    env();
    config(
        name,
        &format!("{LAB}[[frame_key]]\nid = 4\nenv = \"{KEY_ENV}\"\n"),
    )
}

fn node1() -> NodeKey {
    site().node_key(1)
}

fn signed(seq: u32, boot: u32, occupied: bool) -> Vec<u8> {
    let f = if occupied {
        occupied_frame(1, 2, seq, Some((14.0, 0.9)))
    } else {
        empty_frame(1, 2, seq)
    };
    frameauth::sign(&f, &node1(), 4, boot).to_vec()
}

async fn auth_stats(g: &TestGw) -> Value {
    g.json("/api/health").await["ingest"]["frame_auth"].clone()
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn signed_frames_drive_the_space_and_nothing_else_does() {
    let mut g = TestGw::start(signed_cfg("fa-on")).await;
    let mut seq = 0;
    for occupied in [false, true] {
        for _ in 0..30 {
            g.send(&signed(seq, 1, occupied)).await;
            seq += 1;
            g.advance(0.5).await;
        }
    }
    assert_eq!(g.json("/api/spaces/lab").await["state"]["occupied"], true);

    // Each way in that authentication closes, once.
    g.send(&signed(seq - 1, 1, true)).await; // the last frame again
    g.send(&signed(3, 1, false)).await; // an old frame
    g.send(&empty_frame(1, 2, seq + 100).encode()).await; // v1
    g.send(&raw_datagram(1, 0, 0.0, 56)).await; // raw CSI
    let mut forged = signed(seq + 1, 1, false);
    forged[30] ^= 1;
    g.send(&forged).await;
    let other_node = frameauth::sign(&empty_frame(1, 2, seq + 2), &site().node_key(9), 4, 1);
    g.send(&other_node).await; // signed with another node's key
    let mut unknown = signed(seq + 3, 1, false);
    unknown[63] = 5;
    g.send(&unknown).await;
    g.advance(0.5).await;

    let a = auth_stats(&g).await;
    assert_eq!(a["accepted"], 60, "{a}");
    assert_eq!(a["replayed"], 2, "{a}");
    assert_eq!(a["unsigned_refused"], 1, "{a}");
    assert_eq!(a["raw_refused"], 1, "{a}");
    assert_eq!(a["bad_tag"], 2, "{a}");
    assert_eq!(a["unknown_key"], 1, "{a}");
    assert_eq!(a["no_key"], 0, "{a}");
    let health = g.json("/api/health").await;
    assert_eq!(health["frame_auth"]["required"], true);
    assert_eq!(health["frame_auth"]["key_ids"], serde_json::json!([4]));
    // None of the refused frames reached the space: it is still occupied,
    // though the v1 and forged frames said empty.
    assert_eq!(g.json("/api/spaces/lab").await["state"]["occupied"], true);
    let metrics = g.get("/metrics").await.body;
    assert!(
        metrics.contains("saans_frame_auth_total{result=\"replayed\"} 2"),
        "{metrics}"
    );
    g.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_restart_does_not_reopen_frames_already_accepted() {
    let cfg = signed_cfg("fa-restart");
    let mut g = TestGw::start(cfg.clone()).await;
    for seq in 0..10 {
        g.send(&signed(seq, 3, false)).await;
        g.advance(0.5).await;
    }
    assert_eq!(auth_stats(&g).await["accepted"], 10);
    g.stop().await;
    let saved = std::fs::read_to_string(cfg.data_dir.join("frame_counters.json")).unwrap();
    assert!(
        saved.contains("\"boot\":3") && saved.contains("\"seq\":9"),
        "{saved}"
    );

    let mut g = TestGw::start(cfg).await;
    g.send(&signed(9, 3, false)).await; // accepted before the restart
    g.send(&signed(5, 3, false)).await;
    g.send(&signed(10, 3, false)).await; // new
    g.send(&signed(0, 4, false)).await; // the node rebooted
    g.send(&signed(11, 3, false)).await; // the old boot is over
    g.advance(0.5).await;
    let a = auth_stats(&g).await;
    assert_eq!(
        (a["accepted"].clone(), a["replayed"].clone()),
        (2.into(), 3.into()),
        "{a}"
    );
    g.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn without_a_key_v1_works_and_v2_is_refused_not_trusted() {
    let mut g = TestGw::start(config("fa-off", LAB)).await;
    for seq in 0..4 {
        g.send(&empty_frame(1, 2, seq).encode()).await;
        g.send(&signed(seq, 1, true)).await;
        g.advance(0.5).await;
    }
    let health = g.json("/api/health").await;
    let a = &health["ingest"]["frame_auth"];
    assert_eq!(
        (a["no_key"].clone(), a["accepted"].clone()),
        (4.into(), 0.into()),
        "{a}"
    );
    assert_eq!(health["ingest"]["feature_frames"], 4, "the v1 frames");
    assert_eq!(health["frame_auth"]["required"], false);
    g.stop().await;
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn a_missing_key_variable_stops_startup() {
    let cfg = config(
        "fa-missing",
        &format!("{LAB}[[frame_key]]\nid = 0\nenv = \"SAANS_FRAME_AUTH_TEST_UNSET\"\n"),
    );
    let err = Gateway::start(cfg, ClockMode::Manual { t0: T0 })
        .await
        .err()
        .expect("startup refuses");
    let msg = err.to_string();
    assert!(
        msg.contains("SAANS_FRAME_AUTH_TEST_UNSET") && msg.contains("not set"),
        "{msg}"
    );
}
