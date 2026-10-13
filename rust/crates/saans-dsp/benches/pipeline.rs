//! Throughput of the full FeatureExtractor on the golden end-to-end input.
//!
//!     cargo bench -p saans-dsp
//!
//! Prints frames/s and the multiple of real time for one link. Compare with
//! `python3 spec/golden/bench_python.py` on the same input: speed is a
//! claim like any other (SAANS_PLAN.md §4).

use saans_dsp::features::{FeatureExtractor, FeatureExtractorConfig};
use saans_dsp::types::CsiFrame;
use std::path::PathBuf;
use std::time::Instant;

fn read_f64(name: &str) -> Vec<f64> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/golden").join(name);
    std::fs::read(&p)
        .unwrap_or_else(|e| panic!("{}: {e}", p.display()))
        .chunks(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect()
}

fn main() {
    let (n, fs) = (32usize, 20.0);
    let seq = read_f64("features_vitals__seq_in.f64");
    let t = read_f64("features_vitals__t_in.f64");
    let amps = read_f64("features_vitals__amplitudes_in.f64");
    let frames: Vec<CsiFrame> = amps.chunks(n).enumerate().map(|(i, a)| CsiFrame::new(a.to_vec(), seq[i] as u64, fs)).collect();
    let recorded_s = t.last().unwrap() - t[0];

    let reps = 20;
    let start = Instant::now();
    let mut emitted = 0usize;
    for _ in 0..reps {
        let mut fx = FeatureExtractor::new(FeatureExtractorConfig {
            n_subcarriers: n,
            sample_rate_hz: fs,
            vitals_enabled: true,
            heart_enabled: true,
            calibration_s: 10.0,
            ..Default::default()
        });
        for (i, f) in frames.iter().enumerate() {
            emitted += fx.push(f.clone(), Some(seq[i] as u64), Some(t[i]), Some(-92)).len();
        }
    }
    let el = start.elapsed().as_secs_f64();
    let fps = (frames.len() * reps) as f64 / el;
    println!(
        "saans-dsp FeatureExtractor: {} CSI frames x {} reps ({} subcarriers, {} Hz, vitals+heart on) in {:.3} s",
        frames.len(), reps, n, fs, el
    );
    println!("  {:.0} frames/s = {:.0}x real time for one link; {} FeatureFrames", fps, fps / fs, emitted / reps);
    println!("  recording length {:.0} s", recorded_s);

    // Same work as the Python reference does: a full spectral estimate on
    // every frame (SensingPipeline.process), to separate the language
    // speedup from the lazy-estimate one.
    use saans_dsp::pipeline::{PipelineConfig, SensingPipeline};
    let start = Instant::now();
    let reps_eager = 3;
    for _ in 0..reps_eager {
        let mut p = SensingPipeline::new(PipelineConfig { n_subcarriers: n, sample_rate_hz: fs, calibration_s: 10.0, ..Default::default() });
        for f in &frames {
            std::hint::black_box(p.process(f));
        }
    }
    let el = start.elapsed().as_secs_f64();
    let fps = (frames.len() * reps_eager) as f64 / el;
    println!("saans-dsp SensingPipeline.process (estimate every frame, like Python): {:.0} frames/s = {:.0}x real time", fps, fps / fs);
}
