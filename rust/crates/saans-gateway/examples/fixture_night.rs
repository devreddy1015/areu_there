//! A SYNTHETIC night as a pcap: the replay fixture for the browser smoke
//! tests (`tests/browser/`). Built from hand-written FeatureFrames, not
//! the simulator and not a recording, so it measures nothing; it exists
//! so every view of the dashboard has data to draw.
//!
//! ```text
//! cargo run -p saans-gateway --example fixture_night -- night.pcap
//! ```
//!
//! Thirty minutes at 2 Hz, for the spaces `tests/browser/setup.mjs`
//! configures:
//!
//! - `rest` (consented): links 3<-4 and 5<-4. Empty, a still breather at
//!   14 BPM from 5 to 12 min, empty, back from 20 min to the end, so the
//!   final state shows breathing. Link 5<-4 sits in a blind spot and
//!   always votes empty, as in `tests/e2e.rs`.
//! - `hall` (not consented): link 1<-255, occupied from 10 min to the end,
//!   so a breathing estimate would exist in the final state if consent were
//!   wrongly allowed. Its node is misconfigured and reports breathing
//!   (17.25 BPM) throughout, which the gateway must withhold.

use saans_dsp::features::{
    FeatureFrame, FLAG_BREATHING_VALID, FLAG_CALIBRATED, FLAG_OCCUPIED, FLAG_VITALS_ENABLED,
};
use saans_gateway::pcap::PcapWriter;
use std::net::{Ipv4Addr, SocketAddrV4};

const TICKS: u32 = 30 * 60 * 2;

fn frame(rx: u8, tx: u8, seq: u32, occupied: bool, breathing: Option<f32>) -> FeatureFrame {
    let mut f = FeatureFrame {
        node_id: rx,
        tx_id: tx,
        flags: FLAG_CALIBRATED,
        seq,
        uptime_ms: seq.wrapping_mul(500),
        csi_rate_chz: 2000,
        loss_permille: 10,
        motion_energy: 0.02,
        presence_threshold: 0.05,
        gross_motion: 0.01,
        best_bnr: 3.0,
        calib_age_s: 3600,
        rssi_dbm: -55,
        noise_floor_dbm: -92,
        n_subcarriers: 56,
        ..Default::default()
    };
    if occupied {
        f.flags |= FLAG_OCCUPIED;
        f.motion_energy = 0.2;
        f.gross_motion = 0.03;
    }
    if let Some(bpm) = breathing {
        f.flags |= FLAG_VITALS_ENABLED | FLAG_BREATHING_VALID;
        f.breath_rms = 0.1;
        f.breathing_bpm = bpm;
        f.breathing_conf = 0.85;
    }
    f
}

fn main() -> std::io::Result<()> {
    let out = std::env::args()
        .nth(1)
        .expect("usage: fixture_night <out.pcap>");
    let file = std::fs::File::create(&out)?;
    let mut w = PcapWriter::new(std::io::BufWriter::new(file))?;
    let gw = SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 1), 5566);
    let node = |n: u8| SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 100 + n), 5566);
    // 2026-09-21 22:00 UTC: an arbitrary night.
    let base_ns: i64 = 1_790_028_000 * 1_000_000_000;
    for i in 0..TICKS {
        let min = i as f64 / 120.0;
        let t_ns = base_ns + i as i64 * 500_000_000;
        let rest = (5.0..12.0).contains(&min) || min >= 20.0;
        let hall = min >= 10.0;
        let seen = frame(3, 4, i, rest, rest.then_some(14.0));
        w.write(t_ns + 1_000_000, node(3), gw, &seen.encode())?;
        w.write(
            t_ns + 2_000_000,
            node(5),
            gw,
            &frame(5, 4, i, false, None).encode(),
        )?;
        let leaky = frame(1, 255, i, hall, Some(17.25));
        w.write(t_ns + 3_000_000, node(1), gw, &leaky.encode())?;
    }
    w.into_inner().into_inner()?;
    eprintln!("fixture_night: {TICKS} ticks, SYNTHETIC, written to {out}");
    Ok(())
}
