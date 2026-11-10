//! A SYNTHETIC load for the gateway: N links of simulated nights as a pcap,
//! with the config that assigns them, so `saans-gateway --replay` can be
//! timed on its real ingest path (issue #43, `tools/baselines.py gateway`).
//!
//! ```text
//! cargo run --release -p saans-gateway --example link_load -- \
//!     --links 12 --minutes 15 --mode features --out DIR
//! ```
//!
//! writes `DIR/load.pcap` and `DIR/saans.toml`. Nothing here is timed: the
//! simulator and, in `features` mode, the node's own DSP run while the file
//! is written, so the replay that reads it pays only for what a gateway
//! pays for.
//!
//! - `features` (the product): every link sends its node's FeatureFrames,
//!   2 Hz, computed here by the same `FeatureExtractor` the firmware runs.
//! - `raw` (bench and debug mode, ADR-001): every link sends raw CSI at
//!   20 Hz and the gateway runs the extractor itself.
//!
//! Links come in rooms of up to three, one transmitter and one receiver per
//! link, the shape the firmware builds (one TX, more RX), so a raw link is
//! one receiving node. Room r has tx `4r` and rx `4r+1..`, its own seed, and
//! its own night (`scenarios::night_seeded`: walk in, lie breathing with
//! pauses and fidgets, walk out). Every space is consented, so the vitals
//! path runs: the heaviest case.

use saans_dsp::features::{FeatureExtractor, FeatureExtractorConfig};
use saans_gateway::pcap::PcapWriter;
use saans_sim::scenario::{Node, Scenario, NODE_Z_M};
use saans_sim::scenarios::{self, Layout};
use saans_sim::{ImpairConfig, RawNodes, Simulator};
use std::fmt::Write as _;
use std::net::{Ipv4Addr, SocketAddrV4};
use std::path::PathBuf;

const LINKS_PER_ROOM: usize = 3;
const TX_M: (f64, f64) = (1.5, 2.5);
const RX_M: [(f64, f64); LINKS_PER_ROOM] = [(4.5, 2.5), (3.0, 4.6), (3.0, 0.4)];
const PAUSES_PER_HOUR: f64 = 4.0;
const PAUSE_S: (f64, f64) = (10.0, 40.0);
const UDP_PORT: u16 = 5566;
/// 2026-09-21 22:00 UTC: an arbitrary night, as in `fixture_night`.
const BASE_NS: i64 = 1_790_028_000 * 1_000_000_000;

#[derive(Clone, Copy, PartialEq)]
enum Mode {
    Features,
    Raw,
}

struct Args {
    links: usize,
    minutes: f64,
    mode: Mode,
    out: PathBuf,
}

fn parse_args() -> Args {
    let usage = "usage: link_load --links N --minutes M --mode features|raw --out DIR";
    let mut a = std::env::args().skip(1);
    let (mut links, mut minutes, mut mode, mut out) = (None, None, None, None);
    while let Some(k) = a.next() {
        let v = a
            .next()
            .unwrap_or_else(|| panic!("{k} needs a value; {usage}"));
        match k.as_str() {
            "--links" => links = Some(v.parse().expect("--links: an integer")),
            "--minutes" => minutes = Some(v.parse().expect("--minutes: a number")),
            "--mode" => {
                mode = Some(match v.as_str() {
                    "features" => Mode::Features,
                    "raw" => Mode::Raw,
                    _ => panic!("--mode is features or raw; {usage}"),
                })
            }
            "--out" => out = Some(PathBuf::from(v)),
            _ => panic!("unknown argument {k}; {usage}"),
        }
    }
    let args = Args {
        links: links.expect(usage),
        minutes: minutes.expect(usage),
        mode: mode.expect(usage),
        out: out.expect(usage),
    };
    assert!((1..=60).contains(&args.links), "--links is 1..=60");
    assert!(
        args.minutes >= 10.0,
        "--minutes is at least 10 (a night needs room to lie down)"
    );
    args
}

/// Room `r` with `m` links: tx `4r`, rx `4r+1 ..= 4r+m`.
fn room(r: usize, m: usize, minutes: f64) -> Scenario {
    let tx = (4 * r) as u8;
    let mut nodes = vec![Node {
        id: tx,
        pos: TX_M,
        z_m: NODE_Z_M,
    }];
    let mut links = Vec::new();
    for (i, &pos) in RX_M.iter().take(m).enumerate() {
        let rx = tx + 1 + i as u8;
        nodes.push(Node {
            id: rx,
            pos,
            z_m: NODE_Z_M,
        });
        links.push((tx, rx));
    }
    let seed = 1000 + r as u64;
    let night = scenarios::night_seeded(minutes / 60.0, PAUSES_PER_HOUR, PAUSE_S, seed);
    scenarios::scenario(
        &format!("link-load-{r}"),
        scenarios::room_clear_los(),
        Layout { nodes, links },
        night,
        seed,
    )
}

fn config(rooms: &[Scenario], mode: Mode, out: &std::path::Path) -> String {
    let mut s = String::new();
    let _ = writeln!(
        s,
        "# SYNTHETIC: written by examples/link_load.rs for a timed replay."
    );
    let _ = writeln!(s, "utc_offset_minutes = 0");
    // Any free loopback port: `--serve` is timed, never browsed.
    let _ = writeln!(s, "http_addr = \"127.0.0.1:0\"");
    let _ = writeln!(s, "data_dir = {:?}", out.join("data").display().to_string());
    let _ = writeln!(
        s,
        "\n[raw]\nsample_rate_hz = 20.0\nsubcarriers = 0\ncalibration_s = 30.0"
    );
    for (r, sc) in rooms.iter().enumerate() {
        let links: Vec<String> = sc
            .links
            .iter()
            .map(|&(tx, rx)| {
                let tx = if mode == Mode::Raw { 255 } else { tx };
                format!("{{ rx = {rx}, tx = {tx} }}")
            })
            .collect();
        let _ = writeln!(
            s,
            "\n[[space]]\nid = \"room{r}\"\nname = \"Room {r}\"\nvitals = true\n\
             collapse_detection = true\nlinks = [{}]",
            links.join(", ")
        );
    }
    s
}

/// The node DSP per link, as `simulate.rs`'s `Sensors` builds it.
fn extractors(sc: &Scenario) -> Vec<((u8, u8), FeatureExtractor)> {
    sc.links
        .iter()
        .map(|&(tx, rx)| {
            let cfg = FeatureExtractorConfig {
                n_subcarriers: sc.n_subcarriers,
                sample_rate_hz: sc.sample_rate_hz,
                node_id: rx,
                tx_id: tx,
                vitals_enabled: true,
                ..Default::default()
            };
            ((tx, rx), FeatureExtractor::new(cfg))
        })
        .collect()
}

fn main() -> std::io::Result<()> {
    let args = parse_args();
    std::fs::create_dir_all(&args.out)?;
    let rooms: Vec<Scenario> = (0..args.links.div_ceil(LINKS_PER_ROOM))
        .map(|r| {
            let m = LINKS_PER_ROOM.min(args.links - r * LINKS_PER_ROOM);
            room(r, m, args.minutes)
        })
        .collect();
    std::fs::write(
        args.out.join("saans.toml"),
        config(&rooms, args.mode, &args.out),
    )?;

    let mut sims: Vec<Simulator> = rooms.iter().map(Simulator::new).collect();
    let mut raw: Vec<RawNodes> = rooms
        .iter()
        .zip(&sims)
        .map(|(sc, sim)| RawNodes::new(sc, sim.options(), ImpairConfig::esp32_s3(), 0))
        .collect();
    let mut fx: Vec<_> = rooms.iter().map(extractors).collect();

    let file = std::fs::File::create(args.out.join("load.pcap"))?;
    let mut w = PcapWriter::new(std::io::BufWriter::new(file))?;
    let gw = SocketAddrV4::new(Ipv4Addr::new(10, 0, 0, 1), UDP_PORT);
    let node = |n: u8| SocketAddrV4::new(Ipv4Addr::new(10, 0, 1, n), UDP_PORT);
    let mut datagrams = 0u64;
    let mut batch: Vec<(f64, u8, Vec<u8>)> = Vec::new();
    loop {
        batch.clear();
        let mut ended = 0;
        for r in 0..sims.len() {
            let Some(tick) = sims[r].step() else {
                ended += 1;
                continue;
            };
            match args.mode {
                Mode::Raw => {
                    let ts: Vec<f64> = tick.frames.iter().map(|lf| lf.t_s).collect();
                    for ((rx, d), t) in raw[r].datagrams(&tick).into_iter().zip(ts) {
                        batch.push((t, rx, d));
                    }
                }
                Mode::Features => {
                    for lf in &tick.frames {
                        let Some((_, ex)) = fx[r].iter_mut().find(|(k, _)| *k == (lf.tx, lf.rx))
                        else {
                            continue;
                        };
                        for ff in ex.push(lf.frame.clone(), Some(lf.seq as u64), Some(lf.t_s), None)
                        {
                            batch.push((lf.t_s, lf.rx, ff.encode().to_vec()));
                        }
                    }
                }
            }
        }
        if ended == sims.len() {
            break;
        }
        // Jitter is 2 ms against a 50 ms tick, so sorting within a tick
        // keeps the whole file in time order.
        batch.sort_by(|a, b| a.0.total_cmp(&b.0));
        for (t, rx, d) in &batch {
            w.write(BASE_NS + (t * 1e9) as i64, node(*rx), gw, d)?;
            datagrams += 1;
        }
    }
    w.into_inner().into_inner()?;
    let secs = sims.iter().map(Simulator::duration_s).fold(0.0, f64::max);
    eprintln!(
        "link_load: {} links in {} rooms, {:.0} s, {} datagrams ({}), SYNTHETIC, in {}",
        args.links,
        rooms.len(),
        secs,
        datagrams,
        if args.mode == Mode::Raw {
            "raw CSI"
        } else {
            "FeatureFrames"
        },
        args.out.display()
    );
    Ok(())
}
