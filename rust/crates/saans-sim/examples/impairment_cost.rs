//! What the receiver model costs the signal: the same scenario and seed
//! with impairments off, with the int8 quantizer alone, and with the
//! whole ESP32-S3 model. SYNTHETIC, and the model is UNVERIFIED
//! (`src/impair.rs`) — these are statements about the model, not about
//! any hardware.
//! `cargo run --release -p saans-sim --example impairment_cost`
//!
//! Three arms, because "impairments on or off" does not say whether it is
//! the quantizer or the AGC that matters:
//!
//! - `clean`: the simulator's own amplitudes, what every number in this
//!   repository is computed from today;
//! - `int8`: the identity receiver, quantized to signed int8 — the wire
//!   format alone, no AGC, no phase, no ripple;
//! - `no-agc`: the full model with the gain control switched off, which
//!   is what separates the AGC from everything else in it;
//! - `esp32`: [`ImpairConfig::esp32_s3`] in full.
//!
//! The `int8` arm is only as meaningful as the clean channel's arbitrary
//! scale (the physics puts an unattenuated line of sight at 40), which
//! happens to sit near the AGC's 32-count target, so the two arms are
//! quantized at comparable levels. That is a coincidence of the units,
//! not a calibration.
//!
//! Metrics are computed over several room seeds and reported as the
//! median across them, so a single Fresnel null cannot be mistaken for a
//! property of the receiver.

use saans_dsp::fft::{percentile, subcarrier_bnr};
use saans_dsp::{CsiFrame, FeatureExtractor, FeatureExtractorConfig};
use saans_sim::scenario::{Activity, Scenario, Segment};
use saans_sim::scenarios::{self, CHAIR};
use saans_sim::wire::RawNodes;
use saans_sim::{ImpairConfig, Simulator};

/// Room seeds every arm is run on.
const SEEDS: [u64; 5] = [1, 2, 3, 4, 5];
/// Packets dropped from the statistics at the start of each run: the AGC
/// boots at nominal gain and needs a few packets to find the signal. A
/// real node has the same transient; it is just not what is being
/// measured here.
const SETTLE_PACKETS: usize = 40;
/// The breathing band and BNR rate `saans-dsp`'s own feature extractor
/// uses (`features.rs`: BREATH_BAND_HZ, bnr_rate_hz 4 Hz over 30 s).
const BAND_HZ: (f64, f64) = (0.1, 0.5);
const BNR_RATE_HZ: f64 = 4.0;
const BNR_WINDOW_S: f64 = 30.0;

#[derive(Clone, Copy, PartialEq)]
enum Arm {
    Clean,
    Int8,
    NoAgc,
    Esp32,
}

impl Arm {
    fn name(self) -> &'static str {
        match self {
            Arm::Clean => "clean",
            Arm::Int8 => "int8",
            Arm::NoAgc => "no-agc",
            Arm::Esp32 => "esp32",
        }
    }

    fn config(self) -> ImpairConfig {
        match self {
            // Never used: the clean arm does not go through a receiver.
            Arm::Clean => ImpairConfig::identity(),
            Arm::Int8 => ImpairConfig { quantize: true, ..ImpairConfig::identity() },
            // Everything the model has except the gain control: the
            // static ripple, the per-packet phase, the receiver noise and
            // the quantizer. Not a different calibration — one term
            // switched off, to attribute what the others cannot explain.
            Arm::NoAgc => ImpairConfig { agc_step_db: 0.0, ..ImpairConfig::esp32_s3() },
            Arm::Esp32 => ImpairConfig::esp32_s3(),
        }
    }
}

const ARMS: [Arm; 4] = [Arm::Clean, Arm::Int8, Arm::NoAgc, Arm::Esp32];

/// One run of one arm: the frames a consumer sees, the clean amplitudes
/// of the same packets, and how often the AGC moved.
struct Run {
    frames: Vec<CsiFrame>,
    clean: Vec<Vec<f64>>,
    agc_moves: usize,
    agc_steps_db: Vec<f64>,
}

impl Run {
    /// `rows[packet][subcarrier]` of reported amplitude.
    fn rows(&self) -> Vec<Vec<f64>> {
        self.frames.iter().map(|f| f.amplitudes.clone()).collect()
    }
}

fn run(sc: &Scenario, arm: Arm) -> Run {
    run_arm(sc, arm, arm.config())
}

/// [`run`] of the full receiver model with one field overridden.
fn run_cfg(sc: &Scenario, cfg: ImpairConfig) -> Run {
    run_arm(sc, Arm::Esp32, cfg)
}

fn run_arm(sc: &Scenario, arm: Arm, cfg: ImpairConfig) -> Run {
    let mut sim = Simulator::new(sc);
    let mut nodes = RawNodes::new(sc, sim.options(), cfg, 0);
    let rx = sc.links[0].1;
    let (mut frames, mut clean, mut steps) = (Vec::new(), Vec::new(), Vec::new());
    let mut last_gain = f64::NAN;
    let mut buf = Vec::new();
    while let Some(tick) = sim.step() {
        let Some(lf) = tick.frames.iter().find(|f| (f.tx, f.rx) == sc.links[0]) else { continue };
        clean.push(lf.frame.amplitudes.clone());
        if arm == Arm::Clean {
            frames.push(lf.frame.clone());
            continue;
        }
        let node = nodes.node_mut(rx).unwrap();
        node.encode(&lf.frame, &mut buf);
        // Through the wire pairs, by the same call the gateway makes on a
        // parsed datagram (`engine.rs`): amplitude is hypot(imaginary,
        // real) and phase is atan2 of the pair, nothing else.
        let iq: Vec<(i8, i8)> = (0..sc.n_subcarriers).map(|k| (node.iq()[2 * k], node.iq()[2 * k + 1])).collect();
        frames.push(CsiFrame::from_iq(&iq, lf.seq as u64, sc.sample_rate_hz, lf.frame.rssi_dbm, lf.rx));
        let g = node.impairment().gain_db();
        if last_gain.is_finite() && g != last_gain {
            steps.push(g - last_gain);
        }
        last_gain = g;
    }
    Run { frames, clean, agc_moves: steps.len(), agc_steps_db: steps }
}

/// One emitted FeatureFrame, reduced to what this comparison needs.
struct Feat {
    t: f64,
    best_bnr: f64,
    bpm: f64,
    conf: f64,
    occupied: bool,
    motion: f64,
    threshold: f64,
}

/// What the node's own DSP makes of a run: presence, the link-quality
/// score and the breathing it recovers. The same [`FeatureExtractor`] the
/// firmware runs and the gateway runs on raw CSI, configured as
/// `engine.rs` configures it for a raw link. Every emitted frame, with
/// the time it was emitted at, so the empty lead-in and the seated part
/// can be compared.
///
/// Presence is reported alongside breathing because the pipeline
/// suppresses vitals in a room it believes is empty: without it, a lost
/// BPM cannot be told apart from a lost person.
fn features(sc: &Scenario, frames: &[CsiFrame]) -> Vec<Feat> {
    let mut ex = FeatureExtractor::new(FeatureExtractorConfig {
        n_subcarriers: sc.n_subcarriers,
        sample_rate_hz: sc.sample_rate_hz,
        node_id: sc.links[0].1,
        tx_id: 0xFF,
        vitals_enabled: true,
        heart_enabled: false,
        raw_mode: true,
        ..Default::default()
    });
    let mut out = Vec::new();
    for (i, f) in frames.iter().enumerate() {
        let t = i as f64 / sc.sample_rate_hz;
        for ff in ex.push(f.clone(), Some(i as u64), Some(t), None) {
            {
                out.push(Feat {
                    t,
                    best_bnr: ff.best_bnr as f64,
                    bpm: ff.breathing_bpm as f64,
                    conf: ff.breathing_conf as f64,
                    occupied: ff.flag(saans_dsp::features::FLAG_OCCUPIED),
                    motion: ff.motion_energy as f64,
                    threshold: ff.presence_threshold as f64,
                });
            }
        }
    }
    out
}

/// Per-subcarrier coefficient of variation of the reported amplitude.
fn cv(rows: &[Vec<f64>]) -> Vec<f64> {
    let n_k = rows[0].len();
    (0..n_k)
        .map(|k| {
            let col: Vec<f64> = rows.iter().map(|r| r[k]).collect();
            let mean = col.iter().sum::<f64>() / col.len() as f64;
            if mean <= 0.0 {
                return 0.0;
            }
            let var = col.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / col.len() as f64;
            var.sqrt() / mean
        })
        .collect()
}

/// Packet-to-packet change in band-mean power, dB. The AGC's signature:
/// a gain step moves every subcarrier of a packet by the same factor.
fn power_jumps_db(rows: &[Vec<f64>]) -> Vec<f64> {
    let p: Vec<f64> = rows.iter().map(|r| r.iter().map(|a| a * a).sum::<f64>() / r.len() as f64).collect();
    p.windows(2).filter(|w| w[0] > 0.0 && w[1] > 0.0).map(|w| 10.0 * (w[1] / w[0]).log10()).collect()
}

fn mean(xs: &[f64]) -> f64 {
    xs.iter().sum::<f64>() / xs.len().max(1) as f64
}

/// Rows decimated to [`BNR_RATE_HZ`] by averaging, the way the feature
/// extractor builds its BNR rows.
fn decimate(rows: &[Vec<f64>], fs: f64) -> Vec<Vec<f64>> {
    let dec = (fs / BNR_RATE_HZ).round().max(1.0) as usize;
    rows.chunks_exact(dec)
        .map(|c| (0..c[0].len()).map(|k| c.iter().map(|r| r[k]).sum::<f64>() / c.len() as f64).collect())
        .collect()
}

/// Breathing-band signal-to-noise per subcarrier over consecutive 30 s
/// windows: `saans-dsp`'s own `subcarrier_bnr`, the in-band to
/// out-of-band power ratio the link-quality score is made of. Returns
/// (best subcarrier, median subcarrier), dB, median across windows.
fn bnr_db(rows: &[Vec<f64>], fs: f64) -> (f64, f64) {
    let d = decimate(rows, fs);
    let w = (BNR_WINDOW_S * BNR_RATE_HZ) as usize;
    let (mut best, mut mid) = (Vec::new(), Vec::new());
    for chunk in d.chunks_exact(w) {
        let (bnr, _) = subcarrier_bnr(chunk, BNR_RATE_HZ, BAND_HZ.0, BAND_HZ.1);
        let finite: Vec<f64> = bnr.into_iter().filter(|b| b.is_finite()).collect();
        if finite.is_empty() {
            continue;
        }
        best.push(percentile(&finite, 100.0));
        mid.push(percentile(&finite, 50.0));
    }
    let db = |v: &[f64]| 10.0 * percentile(v, 50.0).max(1e-12).log10();
    (db(&best), db(&mid))
}

/// In-band RMS of one subcarrier's amplitude series, in the units of
/// `rows` — counts for the two wire arms. The breathing signal itself,
/// against which an AGC step and an LSB can be compared.
fn band_rms(rows: &[Vec<f64>], k: usize, fs: f64) -> f64 {
    let mut f = saans_dsp::BiquadBandpass::new(BAND_HZ.0, BAND_HZ.1, fs);
    let y: Vec<f64> = rows.iter().map(|r| f.process(r[k])).collect();
    // Drop the filter's start-up transient with the settling packets.
    let tail = &y[y.len() / 4..];
    (tail.iter().map(|v| v * v).sum::<f64>() / tail.len() as f64).sqrt()
}

/// The subcarrier with the highest BNR in the clean arm — the one the
/// pipeline would pick — so every arm is compared on the same bin.
fn best_subcarrier(rows: &[Vec<f64>], fs: f64) -> usize {
    let d = decimate(rows, fs);
    let w = (BNR_WINDOW_S * BNR_RATE_HZ) as usize;
    let chunk = if d.len() >= w { &d[d.len() - w..] } else { &d[..] };
    let (bnr, _) = subcarrier_bnr(chunk, BNR_RATE_HZ, BAND_HZ.0, BAND_HZ.1);
    bnr.iter().enumerate().filter(|(_, b)| b.is_finite()).max_by(|a, b| a.1.total_cmp(b.1)).map(|(i, _)| i).unwrap_or(0)
}

fn empty_scenario(seed: u64, seconds: f64) -> Scenario {
    let tl = vec![Segment { duration_s: seconds, activity: Activity::Absent }];
    let mut sc = scenarios::scenario("empty", scenarios::room_clear_los(), scenarios::one_link(), tl, seed);
    // No packet loss: a missing packet is a hole in a time series, and
    // what is being measured here is the receiver, not the air.
    sc.loss_prob = 0.0;
    sc
}

/// `LEAD_IN_S` of empty room, then a seated person breathing at 14 BPM.
/// The lead-in is the scenarios module's own and is not negotiable: the
/// presence detector ignores its first 50 s (five periods of 0.1 Hz) and
/// only then calibrates, so a shorter one learns its "empty room"
/// threshold with the person already in the room.
fn seated_scenario(seed: u64, seated_s: f64) -> Scenario {
    let tl = vec![
        Segment { duration_s: scenarios::LEAD_IN_S, activity: Activity::Absent },
        Segment { duration_s: seated_s, activity: Activity::Seated { at: CHAIR, bpm: 14.0, bpm_end: None } },
    ];
    let mut sc = scenarios::scenario("seated", scenarios::room_clear_los(), scenarios::one_link(), tl, seed);
    sc.loss_prob = 0.0;
    sc
}

fn row(name: &str, values: &[f64], fmt: fn(f64) -> String) {
    print!("{name:<42}");
    for v in values {
        print!("{:>11}", fmt(*v));
    }
    println!();
}

fn f3(v: f64) -> String {
    format!("{v:.3}")
}

fn f2(v: f64) -> String {
    format!("{v:.2}")
}

fn med(xs: &[f64]) -> f64 {
    percentile(xs, 50.0)
}

fn main() {
    let fs = 20.0;
    println!("saans-sim impairment cost. SYNTHETIC; the receiver model is UNVERIFIED (src/impair.rs).");
    println!("{} room seeds, 1 link x 56 subcarriers at {fs} Hz, no packet loss, first {SETTLE_PACKETS} packets dropped.\n", SEEDS.len());

    // ---------------------------------------------------------- empty room
    let secs = 120.0;
    let mut cv_mean = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut cv_max = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut jump_med = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut jump_p99 = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut jump_max = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut jump_big = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut level = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut agc_rate = Vec::new();
    let mut agc_step = Vec::new();
    let mut clean_identical = true;

    for &seed in &SEEDS {
        let sc = empty_scenario(seed, secs);
        let mut reference: Option<Vec<Vec<f64>>> = None;
        for (i, &arm) in ARMS.iter().enumerate() {
            let r = run(&sc, arm);
            // The clean channel must be bit-identical across arms: the
            // receiver model draws from its own RNG streams and must not
            // have perturbed the physics.
            match &reference {
                None => reference = Some(r.clean.clone()),
                Some(c) => clean_identical &= *c == r.clean,
            }
            let all = r.rows();
            let rows = &all[SETTLE_PACKETS..];
            let c = cv(rows);
            cv_mean[i].push(mean(&c));
            cv_max[i].push(percentile(&c, 100.0));
            let j: Vec<f64> = power_jumps_db(rows).iter().map(|x| x.abs()).collect();
            jump_med[i].push(percentile(&j, 50.0));
            jump_p99[i].push(percentile(&j, 99.0));
            jump_max[i].push(percentile(&j, 100.0));
            jump_big[i].push(100.0 * j.iter().filter(|&&x| x > 0.3).count() as f64 / j.len() as f64);
            level[i].push(mean(&rows.iter().map(|r| mean(r)).collect::<Vec<_>>()));
            if arm == Arm::Esp32 {
                agc_rate.push(r.agc_moves as f64 / r.frames.len() as f64);
                if !r.agc_steps_db.is_empty() {
                    agc_step.push(mean(&r.agc_steps_db.iter().map(|s| s.abs()).collect::<Vec<_>>()));
                }
            }
        }
    }

    println!("EMPTY ROOM, {secs} s, median over seeds");
    print!("{:<42}", "");
    for a in ARMS {
        print!("{:>11}", a.name());
    }
    println!();
    row("per-subcarrier amplitude CV (mean over k)", &cv_mean.iter().map(|v| med(v)).collect::<Vec<_>>(), f3);
    row("per-subcarrier amplitude CV (worst k)", &cv_max.iter().map(|v| med(v)).collect::<Vec<_>>(), f3);
    row("|packet power jump| median, dB", &jump_med.iter().map(|v| med(v)).collect::<Vec<_>>(), f3);
    row("|packet power jump| p99, dB", &jump_p99.iter().map(|v| med(v)).collect::<Vec<_>>(), f3);
    row("|packet power jump| max, dB", &jump_max.iter().map(|v| med(v)).collect::<Vec<_>>(), f3);
    row("packets jumping more than 0.3 dB, %", &jump_big.iter().map(|v| med(v)).collect::<Vec<_>>(), f2);
    row("mean |H| (clean units / int8 counts)", &level.iter().map(|v| med(v)).collect::<Vec<_>>(), f2);
    println!("\nAGC moved on {:.1}% of packets, mean move {:.2} dB.", 100.0 * med(&agc_rate), med(&agc_step));
    println!("clean channel bit-identical across arms: {}", if clean_identical { "yes" } else { "NO" });

    // ------------------------------------------------------------- seated
    let seated_s = 180.0;
    let mut bnr_best = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut bnr_mid = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut breath_rms = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut breath_frac = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut pipe_bnr = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut pipe_bpm = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut pipe_conf = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut pipe_hit = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut pipe_occ = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut pipe_empty = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut pipe_motion = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut pipe_thresh = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    let mut sens = Vec::new();

    for &seed in &SEEDS {
        let sc = seated_scenario(seed, seated_s);
        // Ground truth: is this seat visible to this link at all? A blind
        // spot would make every arm look equally bad for a reason that
        // has nothing to do with the receiver.
        let sim = Simulator::new(&sc);
        sens.push(sim.sensitivity(sc.links[0]).into_iter().fold(0.0, f64::max));

        // The bin the pipeline would choose, chosen once on the clean arm
        // so all three arms are compared on the same subcarrier.
        let clean = run(&sc, Arm::Clean);
        let seated_from = (scenarios::LEAD_IN_S * fs) as usize;
        let k = best_subcarrier(&clean.rows()[seated_from..], fs);
        for (i, &arm) in ARMS.iter().enumerate() {
            let r = run(&sc, arm);
            let all = r.rows();
            let rows = &all[seated_from..];
            let (b, m) = bnr_db(rows, fs);
            bnr_best[i].push(b);
            bnr_mid[i].push(m);
            let rms = band_rms(rows, k, fs);
            breath_rms[i].push(rms);
            let lvl = mean(&rows.iter().map(|r| r[k]).collect::<Vec<_>>());
            breath_frac[i].push(if lvl > 0.0 { rms / lvl } else { 0.0 });

            // And what the node's own DSP does with the same stream.
            // Features are read from a minute after the person sits, by
            // when the breathing extractor's 60 s window holds nothing
            // but seated person.
            let all_ff = features(&sc, &r.frames);
            // The empty room as the presence detector saw it, after its
            // 50 s warm-up and before the person arrives: the other half
            // of the contrast a threshold is supposed to separate.
            let empty_motion: Vec<f64> = all_ff.iter().filter(|f| (60.0..scenarios::LEAD_IN_S - 5.0).contains(&f.t)).map(|f| f.motion).collect();
            pipe_empty[i].push(percentile(&empty_motion, 50.0));
            let ff: Vec<&Feat> = all_ff.iter().filter(|f| f.t >= scenarios::LEAD_IN_S + 65.0).collect();
            let valid: Vec<f64> = ff.iter().filter(|f| f.conf > 0.0).map(|f| f.bpm).collect();
            pipe_bnr[i].push(10.0 * percentile(&ff.iter().map(|f| f.best_bnr).collect::<Vec<_>>(), 50.0).max(1e-12).log10());
            pipe_conf[i].push(percentile(&ff.iter().map(|f| f.conf).collect::<Vec<_>>(), 50.0));
            pipe_bpm[i].push(if valid.is_empty() { f64::NAN } else { percentile(&valid, 50.0) });
            // Frames whose BPM is within 2 of the truth, the tolerance
            // `evaluate.py` scores breathing with.
            pipe_hit[i].push(100.0 * valid.iter().filter(|b| (*b - 14.0).abs() <= 2.0).count() as f64 / ff.len().max(1) as f64);
            pipe_occ[i].push(100.0 * ff.iter().filter(|f| f.occupied).count() as f64 / ff.len().max(1) as f64);
            pipe_motion[i].push(percentile(&ff.iter().map(|f| f.motion).collect::<Vec<_>>(), 50.0));
            pipe_thresh[i].push(percentile(&ff.iter().map(|f| f.threshold).collect::<Vec<_>>(), 50.0));
        }
    }

    println!(
        "\nSEATED, 14 BPM, {seated_s} s after a {} s empty lead-in, median over seeds (ground-truth sensitivity {:.2}, min {:.2})",
        scenarios::LEAD_IN_S,
        med(&sens),
        percentile(&sens, 0.0)
    );
    print!("{:<42}", "");
    for a in ARMS {
        print!("{:>11}", a.name());
    }
    println!();
    row("breathing BNR, best subcarrier, dB", &bnr_best.iter().map(|v| med(v)).collect::<Vec<_>>(), f2);
    row("breathing BNR, median subcarrier, dB", &bnr_mid.iter().map(|v| med(v)).collect::<Vec<_>>(), f2);
    row("breathing-band RMS of that bin (units)", &breath_rms.iter().map(|v| med(v)).collect::<Vec<_>>(), f3);
    row("as a fraction of that bin's mean |H|", &breath_frac.iter().map(|v| med(v)).collect::<Vec<_>>(), f3);
    println!("\nthrough the node's own FeatureExtractor (the gateway's raw path), frames from t = {} s on:", scenarios::LEAD_IN_S + 65.0);
    row("presence: frames reporting occupied, %", &pipe_occ.iter().map(|v| med(v)).collect::<Vec<_>>(), f2);
    row("presence: motion energy, empty room", &pipe_empty.iter().map(|v| med(v)).collect::<Vec<_>>(), f3);
    row("presence: motion energy, person seated", &pipe_motion.iter().map(|v| med(v)).collect::<Vec<_>>(), f3);
    row("presence: calibrated threshold", &pipe_thresh.iter().map(|v| med(v)).collect::<Vec<_>>(), f3);
    row("best_bnr reported, dB", &pipe_bnr.iter().map(|v| med(v)).collect::<Vec<_>>(), f2);
    row("breathing BPM (truth 14.0)", &pipe_bpm.iter().map(|v| med(v)).collect::<Vec<_>>(), f2);
    row("breathing confidence", &pipe_conf.iter().map(|v| med(v)).collect::<Vec<_>>(), f2);
    row("frames within 2 BPM of truth, %", &pipe_hit.iter().map(|v| med(v)).collect::<Vec<_>>(), f2);
    println!("\nOne int8 count is 1.0 in the three wire columns; one 0.5 dB AGC step at 32 counts is 1.9 counts.");

    // ------------------------------------------------- what it rests on
    //
    // The presence result above is dominated by how often the gain moves,
    // and in a still room almost every move is the dither term — a pure
    // guess at "a real AGC also moves for reasons invisible in the CSI".
    // Sweeping it is not tuning: the committed default does not change.
    // It says what would have to be true of a real node for the
    // conclusion to hold, and it is the first thing an empty-room
    // recording would settle.
    println!("\nSENSITIVITY TO agc_dither_prob (every other constant as committed; default {:.3})", ImpairConfig::esp32_s3().agc_dither_prob);
    println!("{:<12}{:>11}{:>11}{:>11}{:>11}{:>11}", "dither", "moves %", "empty", "seated", "threshold", "occupied %");
    for dither in [0.0, 0.002, 0.005, 0.02] {
        let cfg = ImpairConfig { agc_dither_prob: dither, ..ImpairConfig::esp32_s3() };
        let (mut moves, mut empty, mut seated, mut thr, mut occ) = (Vec::new(), Vec::new(), Vec::new(), Vec::new(), Vec::new());
        for &seed in &SEEDS {
            let sc = seated_scenario(seed, seated_s);
            let r = run_cfg(&sc, cfg);
            moves.push(100.0 * r.agc_moves as f64 / r.frames.len() as f64);
            let ff = features(&sc, &r.frames);
            let e: Vec<f64> = ff.iter().filter(|f| (60.0..scenarios::LEAD_IN_S - 5.0).contains(&f.t)).map(|f| f.motion).collect();
            let o: Vec<&Feat> = ff.iter().filter(|f| f.t >= scenarios::LEAD_IN_S + 65.0).collect();
            empty.push(percentile(&e, 50.0));
            seated.push(percentile(&o.iter().map(|f| f.motion).collect::<Vec<_>>(), 50.0));
            thr.push(percentile(&o.iter().map(|f| f.threshold).collect::<Vec<_>>(), 50.0));
            occ.push(100.0 * o.iter().filter(|f| f.occupied).count() as f64 / o.len().max(1) as f64);
        }
        println!(
            "{dither:<12.3}{:>11.2}{:>11.3}{:>11.3}{:>11.3}{:>11.1}",
            med(&moves),
            med(&empty),
            med(&seated),
            med(&thr),
            med(&occ)
        );
    }
}
