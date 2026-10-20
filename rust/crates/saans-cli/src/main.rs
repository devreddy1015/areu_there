//! `saans`: simulate, evaluate and benchmark the Saans pipeline.
//!
//!     saans eval [--quick] [--seeds N] [--threads N]
//!     saans export SCENARIO --out FILE.pcap [--seed N] [--vitals]
//!     saans import-roomplan FILE.json --out SCENE.json [--name NAME]
//!
//! `export` runs a simulated scenario through the node-side FeatureExtractor
//! and writes the FeatureFrames as a pcap, so `saans-gateway --replay`
//! and its dashboard can play back a synthetic night end to end.
//!
//! `import-roomplan` converts an Apple RoomPlan scan into a scene.json.
//! Room geometry is DECLARED, never inferred from WiFi, and a scan is one
//! of the two ways to declare it (the other is measuring the room by
//! hand).

mod eval;
mod pcap;
mod score;

use saans_dsp::features::{FeatureExtractor, FeatureExtractorConfig};
use saans_sim::Simulator;
use std::process::ExitCode;

fn usage() -> ExitCode {
    eprintln!(
        "usage:\n  saans eval [--quick] [--seeds N] [--seed-offset N] [--threads N]\n  saans export {{empty|reader|night|lone_worker|collapse|sit_down|tachypnea}} --out FILE.pcap [--seed N] [--layout 1|2|3] [--no-vitals]\n  saans import-roomplan FILE.json --out SCENE.json [--name NAME]"
    );
    ExitCode::from(2)
}

fn arg<T: std::str::FromStr>(args: &[String], name: &str) -> Option<T> {
    args.iter().position(|a| a == name).and_then(|i| args.get(i + 1)).and_then(|v| v.parse().ok())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    match args.first().map(String::as_str) {
        Some("eval") => {
            let quick = args.iter().any(|a| a == "--quick");
            let threads = arg(&args, "--threads").unwrap_or_else(|| std::thread::available_parallelism().map_or(4, |n| n.get()));
            let seeds = arg(&args, "--seeds").unwrap_or(if quick { 2 } else { 6 });
            let seed_offset = arg(&args, "--seed-offset").unwrap_or(if quick { 0 } else { 2 });
            eval::evaluate(&eval::EvalOptions { quick, seeds, seed_offset, threads });
            ExitCode::SUCCESS
        }
        Some("export") => export(&args),
        Some("trace") => trace(&args),
        Some("import-roomplan") => import_roomplan(&args),
        _ => usage(),
    }
}

/// Per-sample CSV of truth and fused state for one evaluation job, plus
/// its events: the raw material for diagnosing a detector.
fn trace(args: &[String]) -> ExitCode {
    let Some(name) = args.get(1) else { return usage() };
    let Some(out) = arg::<String>(args, "--out") else { return usage() };
    let seeds: u64 = arg(args, "--seeds").unwrap_or(2);
    let jobs = eval::jobs(&eval::EvalOptions { quick: true, seeds, seed_offset: 0, threads: 1 });
    let Some(job) = jobs.into_iter().find(|j| &j.scenario.name == name) else {
        eprintln!("no job named {name}; names look like reader/clear-LoS/1 link/0");
        return ExitCode::FAILURE;
    };
    let run = eval::run(&job);
    let mut csv = String::from("t,present,still,breathing,truth_bpm,in_pause,in_burst,occupied,presence_index,motion_index,links_usable,br_link,br_bpm,br_conf,band_rms,noise_rms\n");
    for (t, s) in &run.samples {
        let occ = match s.occupied { Some(true) => "1", Some(false) => "0", None => "" };
        let (bl, bb, bc, br, bn) = match s.breathing {
            Some(b) => (format!("{}", b.link), format!("{:.3}", b.bpm), format!("{:.3}", b.confidence), format!("{:.5}", b.band_rms), b.noise_rms.map_or(String::new(), |n| format!("{n:.5}"))),
            None => Default::default(),
        };
        csv += &format!(
            "{:.1},{},{},{},{:.2},{},{},{},{:.3},{:.3},{},{},{},{},{},{}\n",
            t.t_s, t.present as u8, t.still as u8, t.breathing as u8, t.bpm, t.in_pause as u8, t.in_burst as u8, occ,
            s.presence_index, s.motion_index, s.links_usable, bl, bb, bc, br, bn
        );
    }
    std::fs::write(&out, csv).expect("write csv");
    for e in &run.events {
        println!("{:8.1}  {:?}  conf {:.2}  {:?}  -- {}", e.t_s, e.severity, e.confidence, e.kind, e.evidence);
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
    let Some(out) = arg::<String>(args, "--out") else { return usage() };
    let seed: u64 = arg(args, "--seed").unwrap_or(1);
    let layout: usize = arg(args, "--layout").unwrap_or(2);
    let vitals = !args.iter().any(|a| a == "--no-vitals");
    let jobs = eval::jobs(&eval::EvalOptions { quick: true, seeds: seed + 1, seed_offset: 0, threads: 1 });
    let layout_name = ["1 link", "2 links", "3 links"][layout.clamp(1, 3) - 1];
    let Some(job) = jobs.into_iter().find(|j| j.kind == kind && j.room == "clear-LoS" && j.layout == layout_name && j.seed == 1000 + seed * 7919 + kind as u64 * 104729) else {
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
                    w.write(t_ns, ([10, 0, 0, 100 + lf.rx], 5566), ([10, 0, 0, 1], 5566), &ff.encode()).expect("write");
                    frames += 1;
                }
            }
        }
    }
    w.flush().expect("flush");
    eprintln!("{}: {frames} FeatureFrames, {:.0} simulated seconds, {} link(s), vitals {} -> {out}", sc.name, sim.duration_s(), sc.links.len(), if vitals { "on" } else { "off" });
    ExitCode::SUCCESS
}

/// The scene's name: `--name` if given, else the input file's stem, so
/// `bedroom-scan.json` becomes the scene "bedroom-scan" rather than
/// something anonymous.
fn scene_name(input: &str, explicit: Option<String>) -> String {
    match explicit {
        Some(n) if !n.trim().is_empty() => n,
        _ => std::path::Path::new(input)
            .file_stem()
            .map(|s| s.to_string_lossy().into_owned())
            .filter(|s| !s.trim().is_empty())
            .unwrap_or_else(|| "imported-room".to_string()),
    }
}

/// Convert an Apple RoomPlan export into a scene.json.
///
/// RoomPlan measures geometry, not materials, not wall thickness, and of
/// course not where anyone put an ESP32. The scene this writes is
/// therefore deliberately incomplete: every assumption is printed, and so
/// is the list of what still has to be filled in before the file will
/// load. Writing a file that does not yet validate is the expected
/// outcome here, so it is not an error — but it is said out loud, because
/// a scene that quietly looked finished would be the worse failure.
fn import_roomplan(args: &[String]) -> ExitCode {
    let Some(input) = args.get(1) else { return usage() };
    if input.starts_with("--") {
        return usage();
    }
    let Some(out) = arg::<String>(args, "--out") else { return usage() };
    let name = scene_name(input, arg::<String>(args, "--name"));

    let text = match std::fs::read_to_string(input) {
        Ok(t) => t,
        Err(e) => {
            eprintln!("{input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let imported = match saans_scene::import_roomplan_str(&name, &text) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("{input}: {e}");
            return ExitCode::FAILURE;
        }
    };
    let scene = imported.scene;

    if let Err(e) = scene.save(&out) {
        eprintln!("{out}: {e}");
        return ExitCode::FAILURE;
    }

    let r = &scene.room;
    eprintln!(
        "{name}: {:.2} x {:.2} x {:.2} m, {} wall(s), {} object(s), {} node(s) -> {out}",
        r.w_m,
        r.d_m,
        r.h_m,
        scene.walls.len(),
        scene.objects.len(),
        scene.nodes.len()
    );
    // SCENE, not a measurement: say what had to be assumed.
    for w in &imported.warnings {
        eprintln!("  assumed: {w}");
    }
    let errors = scene.validate();
    if !errors.is_empty() {
        eprintln!("  this scene does not load yet ({} problem(s) to fix by hand):", errors.len());
        for e in &errors {
            eprintln!("    - {e}");
        }
    }
    ExitCode::SUCCESS
}

#[cfg(test)]
mod tests {
    use super::scene_name;

    #[test]
    fn an_explicit_name_wins() {
        assert_eq!(scene_name("/tmp/scan.json", Some("living room".into())), "living room");
    }

    #[test]
    fn the_file_stem_is_the_default_name() {
        assert_eq!(scene_name("/tmp/bedroom-scan.json", None), "bedroom-scan");
        assert_eq!(scene_name("office.json", None), "office");
    }

    #[test]
    fn a_blank_name_falls_back_to_the_stem_rather_than_an_empty_scene_name() {
        // An empty name fails validation, so it must never be what we write.
        assert_eq!(scene_name("/tmp/hall.json", Some("   ".into())), "hall");
    }

    #[test]
    fn a_path_with_no_stem_still_produces_a_name() {
        assert_eq!(scene_name("..", None), "imported-room");
        assert_eq!(scene_name("", None), "imported-room");
    }
}
