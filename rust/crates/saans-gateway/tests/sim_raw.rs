//! The simulator's raw-CSI datagrams through the gateway's own parser and
//! its own raw-mode ingest path — the point of the raw wire format being
//! that one decoder serves a simulated node and a bench node.
//!
//! The CSI here is SYNTHETIC and the receiver model that impaired it is
//! UNVERIFIED (`saans-sim/src/impair.rs`). What these tests check is the
//! plumbing: that the bytes a simulated node writes are the bytes this
//! gateway already knows how to read, and that they drive the same code
//! path a real node's datagrams will.

mod support;

use saans_dsp::CsiFrame;
use saans_gateway::raw;
use saans_sim::scenario::{Activity, Scenario, Segment};
use saans_sim::scenarios::{self, CHAIR};
use saans_sim::wire::{RawNodes, RAW_HEADER_SIZE};
use saans_sim::{ImpairConfig, Simulator};
use support::*;

/// 10 s of empty room, then a seated person breathing at 14 BPM. One
/// link, so one receiving node (node 1) and one raw-mode link `1<-255`.
///
/// That lead-in is far too short for the presence detector, which
/// ignores its first 50 s before it starts calibrating — deliberately,
/// because these tests are about the wire and the ingest path and assert
/// nothing about presence or breathing. What the receiver model costs
/// those detectors is measured, on timelines long enough to calibrate
/// honestly, by `saans-sim/examples/impairment_cost.rs`.
fn scenario(seconds: f64) -> Scenario {
    let timeline = vec![
        Segment { duration_s: 10.0, activity: Activity::Absent },
        Segment { duration_s: seconds - 10.0, activity: Activity::Seated { at: CHAIR, bpm: 14.0, bpm_end: None } },
    ];
    let mut sc = scenarios::scenario("raw-wire", scenarios::room_clear_los(), scenarios::one_link(), timeline, 7);
    // No air loss: this is a test of the wire and the ingest path, and a
    // dropped packet would make the datagram counts depend on the RNG.
    sc.loss_prob = 0.0;
    sc
}

#[test]
fn a_simulated_datagram_is_what_the_gateways_parser_expects() {
    let sc = scenario(30.0);
    let mut sim = Simulator::new(&sc);
    let mut nodes = RawNodes::new(&sc, sim.options(), ImpairConfig::esp32_s3(), 0);
    let tick = sim.step().expect("one tick");
    let lf = &tick.frames[0];
    let datagrams = nodes.datagrams(&tick);
    assert_eq!(datagrams.len(), 1);
    let (rx, d) = &datagrams[0];
    let iq = nodes.node(*rx).unwrap().iq().to_vec();

    assert!(raw::sniff(d), "the gateway must recognise this as raw CSI");
    let f = raw::parse(d).expect("the gateway's own parser accepts it");
    assert_eq!(f.node_id, lf.rx);
    assert_eq!(f.n_antennas, 1);
    assert_eq!(f.n_subcarriers as usize, sc.n_subcarriers);
    assert_eq!(f.freq_mhz, 2437, "the default 2.437 GHz carrier");
    assert_eq!(f.sequence, 0, "first datagram of this node's boot");
    assert_eq!(f.rssi_dbm as i32, lf.frame.rssi_dbm);
    assert_eq!(f.noise_floor_dbm, i8::MIN, "not modelled, and the format says so");
    assert_eq!(d.len(), RAW_HEADER_SIZE + 2 * sc.n_subcarriers);

    // The values that went in are the values that came out, pair by pair,
    // imaginary first.
    assert_eq!(f.iq.len(), sc.n_subcarriers);
    for k in 0..sc.n_subcarriers {
        assert_eq!(f.iq[k], (iq[2 * k], iq[2 * k + 1]), "subcarrier {k}");
    }

    // And the bytes are the ones the gateway's own encoder would write
    // for the same frame. `raw::build` is the port of the Python
    // reference's `build_esp32_frame`; it fixes one antenna and 2437 MHz,
    // which is what this scenario has, so the two encoders must agree
    // byte for byte. If they ever stop agreeing, one of them has drifted
    // from `spec/wire.md`.
    assert_eq!(d, &raw::build(&f.iq, f.node_id, f.sequence, f.rssi_dbm, f.noise_floor_dbm));
}

#[test]
fn flipping_the_pair_order_changes_what_the_gateway_reads() {
    // What test 1 is actually worth: with the pairs swapped, every
    // amplitude is identical and every phase is mirrored, so an
    // amplitude-only check would pass a datagram whose phase is wrong.
    let sc = scenario(30.0);
    let mut sim = Simulator::new(&sc);
    let mut nodes = RawNodes::new(&sc, sim.options(), ImpairConfig::esp32_s3(), 0);
    let tick = sim.step().unwrap();
    let (_, d) = nodes.datagrams(&tick).remove(0);

    let mut flipped = d.clone();
    for k in 0..sc.n_subcarriers {
        flipped.swap(RAW_HEADER_SIZE + 2 * k, RAW_HEADER_SIZE + 2 * k + 1);
    }
    let good = raw::parse(&d).unwrap();
    let bad = raw::parse(&flipped).unwrap();
    assert_ne!(good.iq, bad.iq, "a swapped payload must not parse to the same channel");

    let a = CsiFrame::from_iq(&good.iq, 0, 20.0, -50, 1);
    let b = CsiFrame::from_iq(&bad.iq, 0, 20.0, -50, 1);
    assert_eq!(a.amplitudes, b.amplitudes, "amplitude cannot tell the two orders apart");
    for k in 0..sc.n_subcarriers {
        // atan2(re, im) = pi/2 - atan2(im, re), wrapped. Equivalently the
        // phases differ unless the pair happens to be symmetric.
        if good.iq[k].0 != good.iq[k].1 {
            assert_ne!(a.phases[k], b.phases[k], "subcarrier {k} phase is unchanged by the swap");
        }
    }
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn simulated_raw_csi_drives_the_gateways_raw_mode_path() {
    // The same space configuration a bench node in raw mode needs: raw
    // CSI carries no transmitter id, so the gateway routes every raw
    // datagram to `(rx, 0xFF)`.
    const RAW: &str = "[raw]\ncalibration_s = 2.0\n[[space]]\nid = \"lab\"\nlinks = [{ rx = 1, tx = 255 }]\n";
    let seconds = 45.0;
    let sc = scenario(seconds);
    let mut sim = Simulator::new(&sc);
    let mut nodes = RawNodes::new(&sc, sim.options(), ImpairConfig::esp32_s3(), 0);

    let mut g = TestGw::start(config("sim-raw", RAW)).await;
    let mut sent = 0u64;
    // 20 Hz CSI delivered 0.5 s at a time, as a node streams it.
    while let Some(tick) = sim.step() {
        for (_, d) in nodes.datagrams(&tick) {
            g.send(&d).await;
            sent += 1;
        }
        if sim.ticks_done().is_multiple_of(10) {
            g.advance(0.5).await;
        }
    }
    g.advance(0.5).await;
    assert_eq!(sent, (seconds * sc.sample_rate_hz) as u64);

    let h = g.json("/api/health").await;
    assert_eq!(h["ingest"]["raw_datagrams"], sent, "{h}");
    assert_eq!(h["ingest"]["feature_frames"], 0, "nothing here is a FeatureFrame datagram");
    assert_eq!(h["ingest"]["errors"]["raw_malformed"], 0);
    assert_eq!(h["ingest"]["errors"]["raw_subcarrier_mismatch"], 0);
    assert_eq!(h["ingest"]["errors"]["unknown_type"], 0);
    assert_eq!(h["ingest"]["unassigned"], 0);
    // One FeatureFrame per 10 CSI frames (2 Hz), computed by the gateway
    // from the int8 pairs it parsed.
    assert_eq!(h["ingest"]["raw_feature_frames"], sent / 10);
    assert_eq!(h["raw_mode_links_active"], 1);

    let s = &g.json("/api/spaces").await[0];
    assert_eq!(s["raw_mode"], true, "{s}");
    assert_eq!(s["raw_mode_links"], serde_json::json!(["1<-255"]));
    let l = &g.json("/api/links").await[0]["links"][0];
    assert_eq!(l["link"], serde_json::json!({"rx": 1, "tx": 255}), "{l}");
    assert_eq!(l["up"], true);
    assert_eq!(l["raw_mode"], true);
    assert_eq!(l["frames"], sent / 10);
    assert!(l["rate_hz"].as_f64().unwrap() > 15.0, "{l}");
    // The RSSI the node reported, which the AGC does not touch; and no
    // noise floor, because the simulator has no measurement to report.
    assert!((-90.0..-20.0).contains(&l["rssi_dbm"].as_f64().unwrap()), "{l}");
    assert_eq!(l["noise_floor_dbm"], -128);
    // The link-quality score the DSP computed from the int8 pairs. Only
    // that it exists: a 30 s BNR window filled and produced a finite
    // number from impaired, quantized CSI. What value it should have is
    // the question `examples/impairment_cost.rs` measures, not one this
    // test is entitled to assert.
    assert!(l["best_bnr"].as_f64().unwrap() > 0.0, "{l}");
    g.stop().await;
}
