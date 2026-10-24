//! `saans`: simulate, evaluate and benchmark the Saans pipeline.
//!
//!     saans eval [--quick] [--seeds N] [--threads N] [--room-height M]
//!     saans lying [--spots N] [--threads N]
//!     saans export SCENARIO --out FILE.pcap [--seed N] [--vitals]
//!
//! `export` runs a simulated scenario through the node-side FeatureExtractor
//! and writes the FeatureFrames as a pcap, so `saans-gateway --replay`
//! and its dashboard can play back a synthetic night end to end.

mod eval;
mod links;
mod lying;
mod octave;
mod pcap;
mod score;

use saans_dsp::features::{FeatureExtractor, FeatureExtractorConfig};
use saans_sim::Simulator;
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  saans eval [--quick] [--seeds N] [--seed-offset N] [--threads N] [--room-height M]\n  saans octave [--seeds N] [--per-class N] [--threads N]\n  saans links [--seats N] [--blind-seats N] [--threads N]\n  saans lying [--spots N] [--threads N]\n  saans export {{empty|reader|night|lone_worker|collapse|sit_down|tachypnea}} --out FILE.pcap [--seed N] [--layout 1|2|3] [--no-vitals]"
    );
    ExitCode::from(2)
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("eval") => {
            let quick = args.iter().any(|a| a == "--quick");
            let threads = arg(&args, "--threads")
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
            let seeds = arg(&args, "--seeds").unwrap_or(if quick { 2 } else { 6 });
            let seed_offset = arg(&args, "--seed-offset").unwrap_or(if quick { 0 } else { 2 });
            // Strict: a typo here must not quietly evaluate the slice.
            let room_height_m = match args.iter().position(|a| a == "--room-height") {
                None => None,
                Some(i) => match args.get(i + 1).and_then(|v| v.parse::<f64>().ok()) {
                    Some(h) => Some(h),
                    None => {
                        eprintln!("--room-height needs a height in metres");
                        return ExitCode::FAILURE;
                    }
                },
            };
            match eval::evaluate(&eval::EvalOptions {
                quick,
                seeds,
                seed_offset,
                threads,
                room_height_m,
            }) {
                Ok(()) => ExitCode::SUCCESS,
                Err(e) => {
                    eprintln!("saans eval: {e}");
                    ExitCode::FAILURE
                }
            }
        }
        Some("links") => {
            let threads = arg(&args, "--threads")
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
            links::run(
                arg(&args, "--seats").unwrap_or(150),
                arg(&args, "--blind-seats").unwrap_or(2000),
                threads,
            );
            ExitCode::SUCCESS
        }
        Some("octave") => {
            let threads = arg(&args, "--threads")
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
            octave::run(
                arg(&args, "--seeds").unwrap_or(4000),
                arg(&args, "--per-class").unwrap_or(40),
                threads,
            );
            ExitCode::SUCCESS
        }
        Some("lying") => {
            let threads = arg(&args, "--threads")
                .unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
            lying::run(arg(&args, "--spots").unwrap_or(60), threads);
            ExitCode::SUCCESS
        }
        Some("export") => export(&args),
        Some("trace") => trace(&args),
        _ => usage(),
    }
}

/// Per-sample CSV of truth and fused state for one evaluation job, plus
/// its events: the raw material for diagnosing a detector.
fn trace(args: &[String]) -> ExitCode {
    let Some(name) = args.get(1) else {
        return usage();
    };
    let Some(out) = arg::<String>(args, "--out") else {
        return usage();
    };
    let seeds: u64 = arg(args, "--seeds").unwrap_or(2);
    let jobs = eval::jobs(&eval::EvalOptions {
        quick: true,
        seeds,
        seed_offset: 0,
        threads: 1,
        room_height_m: None,
    });
    let Some(job) = jobs.into_iter().find(|j| &j.scenario.name == name) else {
        eprintln!("no job named {name}; names look like reader/clear-LoS/1 link/0");
        return ExitCode::FAILURE;
    };
    let run = eval::run(&job);
    let mut csv = String::from("t,present,still,breathing,truth_bpm,in_pause,in_burst,occupied,presence_index,motion_index,links_usable,br_link,br_bpm,br_conf,band_rms,noise_rms\n");
    for (t, s) in &run.samples {
        let occ = match s.occupied {
            Some(true) => "1",
            Some(false) => "0",
            None => "",
        };
        let (bl, bb, bc, br, bn) = match s.breathing {
            Some(b) => (
                format!("{}", b.link),
                format!("{:.3}", b.bpm),
                format!("{:.3}", b.confidence),
                format!("{:.5}", b.band_rms),
                b.noise_rms.map_or(String::new(), |n| format!("{n:.5}")),
            ),
            None => Default::default(),
        };
        csv += &format!(
            "{:.1},{},{},{},{:.2},{},{},{},{:.3},{:.3},{},{},{},{},{},{}\n",
            t.t_s,
            t.present as u8,
            t.still as u8,
            t.breathing as u8,
            t.bpm,
            t.in_pause as u8,
            t.in_burst as u8,
            occ,
            s.presence_index,
            s.motion_index,
            s.links_usable,
            bl,
            bb,
            bc,
            br,
            bn
        );
    }
    std::fs::write(&out, csv).expect("write csv");
    for e in &run.events {
        println!(
            "{:8.1}  {:?}  conf {:.2}  {:?}  -- {}",
            e.t_s, e.severity, e.confidence, e.kind, e.evidence
        );
    }
    for t in &run.truth_events {
        eprintln!("truth {:?} {:.1}-{:.1}", t.kind, t.t_start_s, t.t_end_s);
    }
    ExitCode::SUCCESS
}

fn export(args: &[String]) -> ExitCode {
    use eval::Kind;
    let kind = match args.get(1).map(String::as_str) {
        Some("empty") => Kind::Empty,
        Some("reader") => Kind::Reader,
        Some("night") => Kind::Night,
        Some("lone_worker") => Kind::LoneWorker,
        Some("collapse") => Kind::Collapse,
        Some("sit_down") => Kind::SitDown,
        Some("tachypnea") => Kind::Tachypnea,
        _ => return usage(),
    };
    let Some(out) = arg::<String>(args, "--out") else {
        return usage();
    };
    let seed: u64 = arg(args, "--seed").unwrap_or(1);
    let layout: usize = arg(args, "--layout").unwrap_or(2);
    let vitals = !args.iter().any(|a| a == "--no-vitals");
    let jobs = eval::jobs(&eval::EvalOptions {
        quick: true,
        seeds: seed + 1,
        seed_offset: 0,
        threads: 1,
        room_height_m: None,
    });
    let layout_name = ["1 link", "2 links", "3 links"][layout.clamp(1, 3) - 1];
    let Some(job) = jobs.into_iter().find(|j| {
        j.kind == kind
            && j.room == "clear-LoS"
            && j.layout == layout_name
            && j.seed == 1000 + seed * 7919 + kind as u64 * 104729
    }) else {
        eprintln!("no such scenario");
        return ExitCode::FAILURE;
    };
    let sc = &job.scenario;
    let mut sim = Simulator::new(sc);
    let mut ex: Vec<((u8, u8), FeatureExtractor)> = sc
        .links
        .iter()
        .map(|&(tx, rx)| {
            (
                (tx, rx),
                FeatureExtractor::new(FeatureExtractorConfig {
                    n_subcarriers: sc.n_subcarriers,
                    sample_rate_hz: sc.sample_rate_hz,
                    node_id: rx,
                    tx_id: tx,
                    vitals_enabled: vitals,
                    ..Default::default()
                }),
            )
        })
        .collect();
    let file = match std::fs::File::create(&out) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("{out}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let mut w = pcap::PcapWriter::new(std::io::BufWriter::new(file)).expect("pcap header");
    // Anchor simulated time at a fixed epoch so exports are reproducible.
    let epoch_ns: u64 = 1_790_000_000 * 1_000_000_000;
    let mut frames = 0u64;
    while let Some(tick) = sim.step() {
        for lf in tick.frames {
            if let Some((_, e)) = ex.iter_mut().find(|(k, _)| *k == (lf.tx, lf.rx)) {
                for ff in e.push(lf.frame, Some(lf.seq as u64), Some(lf.t_s), None) {
                    let t_ns = epoch_ns + (lf.t_s * 1e9) as u64;
                    w.write(
                        t_ns,
                        ([10, 0, 0, 100 + lf.rx], 5566),
                        ([10, 0, 0, 1], 5566),
                        &ff.encode(),
                    )
                    .expect("write");
                    frames += 1;
                }
            }
        }
    }
    w.flush().expect("flush");
    eprintln!(
        "{}: {frames} FeatureFrames, {:.0} simulated seconds, {} link(s), vitals {} -> {out}",
        sc.name,
        sim.duration_s(),
        sc.links.len(),
        if vitals { "on" } else { "off" }
    );
    ExitCode::SUCCESS
}
