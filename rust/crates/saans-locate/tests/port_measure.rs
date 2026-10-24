//! Port of the key cases of tests/test_localize_measure.py, plus the new
//! `RssiSignal` (spec/locate.md §3).
//!
//! The Python file drives a synthetic CSI channel; these tests use a
//! simpler generator (noise around a level, optionally modulated) because
//! what they pin is the windowing, clock and calibration logic. Value
//! parity with the Python is the golden stage's job.

use std::sync::Arc;

use saans_locate::geometry::*;
use saans_locate::measure::*;
use saans_locate::pynum::LocateError;
use saans_sim::rng::Rng;

const RATE_HZ: f64 = 12.5;

fn two_node_geometry() -> Arc<Geometry> {
    Arc::new(
        Geometry::new(
            Room::default(),
            vec![Node::new(0, 0.1, 3.0), Node::new(1, 7.9, 3.0)],
            None,
            0.3,
        )
        .unwrap(),
    )
}

fn four_node_geometry() -> Arc<Geometry> {
    Arc::new(Geometry::perimeter_n(4).unwrap())
}

/// A frame of `n_sub` amplitudes around `level` with relative noise
/// `sigma`, scaled by `gain` (a body on the link moves it).
fn frame(rng: &mut Rng, level: f64, gain: f64, sigma: f64, n_sub: usize, i: u64) -> CsiFrame {
    let amps: Vec<f64> = (0..n_sub)
        .map(|k| (gain * level * (1.0 + 0.05 * k as f64) + sigma * rng.gauss()).abs())
        .collect();
    new_csi_frame(amps, vec![0.0; n_sub], i, RATE_HZ, -50 - (i % 3) as i32, 1).unwrap()
}

fn walking_gain(t: f64) -> f64 {
    1.0 + 0.3 * (2.0 * std::f64::consts::PI * t / 1.3).sin()
}

fn measurement(g: &Arc<Geometry>, window_s: f64, rate: f64) -> LinkMeasurement {
    let config = MeasurementConfig {
        window_s,
        sample_rate_hz: rate,
        ..Default::default()
    };
    LinkMeasurement::new(g.clone(), config, Box::new(TotalPowerSignal)).unwrap()
}

/// Every link of `g`, `duration_s` of empty room (or a walker on every
/// link when `walking`), in TDMA order with a shared timestamp per frame.
fn corpus(
    g: &Geometry,
    seed: u64,
    duration_s: f64,
    walking: bool,
    slot_s: f64,
) -> Vec<(f64, i64, i64, CsiFrame)> {
    let mut rng = Rng::new(seed);
    let mut out = Vec::new();
    let n = (duration_s * RATE_HZ) as u64;
    for i in 0..n {
        for l in g.links() {
            let t = i as f64 / RATE_HZ + slot_s * l.tx_id as f64;
            let gain = if walking { walking_gain(t) } else { 1.0 };
            out.push((
                t,
                l.tx_id,
                l.rx_id,
                frame(&mut rng, 10.0 + l.tx_id as f64, gain, 0.2, 16, i),
            ));
        }
    }
    out
}

fn feed(m: &mut LinkMeasurement, frames: &[(f64, i64, i64, CsiFrame)]) {
    for (t, tx, rx, f) in frames {
        m.update(*t, *tx, *rx, f).unwrap();
    }
}

fn mean_std(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean) * (v - mean)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

// ------------------------------------------------------------ construction

#[test]
fn constructor_checks_and_derived_sizes() {
    let g = two_node_geometry();
    let m = measurement(&g, 1.0, RATE_HZ);
    assert_eq!(
        m.min_samples, 6,
        "max(3, round(0.5 * 12.5)) = round(6.25) = 6"
    );
    assert_eq!(
        measurement(&g, 1.0, 6.25).min_samples,
        3,
        "round(3.125) = 3"
    );
    assert_eq!(measurement(&g, 1.0, 20.0).min_samples, 10);
    assert_eq!(
        measurement(&g, 1.0, 13.0).min_samples,
        6,
        "round(6.5) is 6, half to even"
    );
    assert_eq!(
        measurement(&g, 1.0, 15.0).min_samples,
        8,
        "round(7.5) is 8, half to even"
    );
    for (window_s, rate) in [(0.1, 12.5), (1.0, 2.0), (0.5, 4.0), (0.05, 1.0)] {
        assert!(measurement(&g, window_s, rate).min_samples >= 3);
    }
    let bad = |window_s: f64, rate: f64, min_samples: Option<i64>| {
        let config = MeasurementConfig {
            window_s,
            sample_rate_hz: rate,
            min_samples,
            ..Default::default()
        };
        LinkMeasurement::new(g.clone(), config, Box::new(TotalPowerSignal)).unwrap_err()
    };
    assert!(matches!(bad(0.0, 12.5, None), LocateError::ValueError(_)));
    assert!(matches!(bad(1.0, -1.0, None), LocateError::ValueError(_)));
    assert!(matches!(
        bad(1.0, 12.5, Some(1)),
        LocateError::ValueError(_)
    ));
    // `nan <= 0` passes the check; round(nan) then raises ValueError.
    assert!(matches!(
        bad(f64::NAN, 12.5, None),
        LocateError::ValueError(_)
    ));
    // round(inf) and ceil(inf) raise OverflowError.
    assert!(matches!(
        bad(1.0, f64::INFINITY, None),
        LocateError::OverflowError(_)
    ));
    assert!(matches!(
        bad(1.0, f64::INFINITY, Some(4)),
        LocateError::OverflowError(_)
    ));
    assert_eq!(
        CalibrationMode::parse("normalize").unwrap(),
        CalibrationMode::Normalize
    );
    assert!(CalibrationMode::parse("divide").is_err());
}

#[test]
fn csi_frame_post_init_checks() {
    assert!(matches!(
        new_csi_frame(vec![1.0; 3], vec![0.0; 2], 0, 10.0, 0, 0),
        Err(LocateError::ValueError(_))
    ));
    assert!(matches!(
        new_csi_frame(vec![1.0; 3], vec![0.0; 3], 0, 0.0, 0, 0),
        Err(LocateError::ValueError(_))
    ));
    // `nan <= 0.0` is False: Python accepts a NaN rate.
    assert!(new_csi_frame(vec![1.0; 3], vec![0.0; 3], 0, f64::NAN, 0, 0).is_ok());
    // A frame built around the checks is refused at `update`.
    let g = two_node_geometry();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let mut f = new_csi_frame(vec![1.0; 3], vec![0.0; 3], 0, 10.0, 0, 0).unwrap();
    f.phases.pop();
    assert!(matches!(
        m.update(0.0, 0, 1, &f),
        Err(LocateError::ValueError(_))
    ));
}

// ------------------------------------------------------------ shape

#[test]
fn vector_length_and_ready_mask() {
    let g = four_node_geometry();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    assert_eq!(m.vector(None), vec![0.0; 12]);

    let mut rng = Rng::new(3);
    let fed = [(0i64, 1i64), (2, 3)];
    for i in 0..60u64 {
        for &(tx, rx) in &fed {
            m.update(
                i as f64 / RATE_HZ,
                tx,
                rx,
                &frame(&mut rng, 10.0, 1.0, 0.2, 16, i),
            )
            .unwrap();
        }
    }
    let v = m.vector(None);
    let ready = m.ready_mask(None);
    for (index, l) in g.links().iter().enumerate() {
        if fed.contains(&(l.tx_id, l.rx_id)) {
            assert!(ready[index] && v[index] > 0.0);
        } else {
            assert!(!ready[index]);
            assert_eq!(v[index], 0.0);
        }
    }
    let f = frame(&mut rng, 10.0, 1.0, 0.2, 16, 0);
    assert!(matches!(
        m.update(0.0, 0, 99, &f),
        Err(LocateError::KeyError(_))
    ));
}

// ------------------------------------------------------------ window numerics

#[test]
fn windowed_variance_matches_independent_two_pass() {
    let g = two_node_geometry();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let index = g.link_index(0, 1).unwrap();
    let mut rng = Rng::new(11);
    let mut pushed: Vec<(f64, f64)> = Vec::new();
    for i in 0..80u64 {
        let t = i as f64 / RATE_HZ;
        let f = frame(&mut rng, 10.0, walking_gain(t), 0.2, 56, i);
        m.update(t, 0, 1, &f).unwrap();
        pushed.push((
            t,
            10.0 * f.amplitudes.iter().map(|a| a * a).sum::<f64>().log10(),
        ));
        let in_window: Vec<f64> = pushed
            .iter()
            .filter(|(ts, _)| *ts >= t - 1.0)
            .map(|&(_, v)| v)
            .collect();
        if (in_window.len() as i64) < m.min_samples {
            continue;
        }
        let (_, std) = mean_std(&in_window);
        let got = m.vector(None)[index];
        assert!(
            (got - std * std).abs() < 1e-12,
            "sample {i}: {got} vs {}",
            std * std
        );
    }
}

#[test]
fn sample_count_tracks_the_window_without_flicker() {
    let g = two_node_geometry();
    let index = g.link_index(0, 1).unwrap();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let mut rng = Rng::new(12);
    for i in 0..60u64 {
        m.update(
            i as f64 / RATE_HZ,
            0,
            1,
            &frame(&mut rng, 10.0, 1.0, 0.2, 8, i),
        )
        .unwrap();
    }
    assert_eq!(
        m.sample_counts(None)[index],
        13,
        "1.0 s at 12.5 Hz, boundary sample kept"
    );

    for (rate, window_s, expected) in [(10.0, 1.0, 11usize), (20.0, 1.0, 21), (12.5, 0.8, 11)] {
        let mut m = measurement(&g, window_s, rate);
        let mut seen = std::collections::BTreeSet::new();
        for i in 0..80u64 {
            m.update(
                i as f64 / rate,
                0,
                1,
                &frame(&mut rng, 10.0, 1.0, 0.2, 8, i),
            )
            .unwrap();
            if i > 40 {
                seen.insert(m.sample_counts(None)[index]);
            }
        }
        assert_eq!(
            seen.into_iter().collect::<Vec<_>>(),
            vec![expected],
            "{rate} Hz / {window_s} s"
        );
    }
}

#[test]
fn an_under_sampled_link_abstains_at_exactly_min_samples_minus_one() {
    let g = two_node_geometry();
    let index = g.link_index(0, 1).unwrap();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let mut rng = Rng::new(17);
    for i in 0..(m.min_samples - 1) as u64 {
        m.update(
            i as f64 / RATE_HZ,
            0,
            1,
            &frame(&mut rng, 10.0, 1.0, 0.2, 16, i),
        )
        .unwrap();
    }
    assert_eq!(m.sample_counts(None)[index] as i64, m.min_samples - 1);
    assert!(!m.ready_mask(None)[index]);
    assert_eq!(m.vector(None)[index], 0.0);
    let i = (m.min_samples - 1) as u64;
    m.update(
        i as f64 / RATE_HZ,
        0,
        1,
        &frame(&mut rng, 10.0, 1.0, 0.2, 16, i),
    )
    .unwrap();
    assert!(m.ready_mask(None)[index]);
    assert!(m.vector(None)[index] > 0.0);
}

#[test]
fn the_memory_cap_is_counted_rather_than_silent() {
    let g = two_node_geometry();
    let index = g.link_index(0, 1).unwrap();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let mut rng = Rng::new(19);
    for i in 0..300u64 {
        m.update(
            i as f64 / 100.0,
            0,
            1,
            &frame(&mut rng, 10.0, 1.0, 0.2, 8, i),
        )
        .unwrap();
    }
    assert!(m.window_overflows > 0);
    let count = m.sample_counts(None)[index];
    assert_eq!(count, 50, "max(8, ceil(4 * 12.5)) = 50");
    assert!((count as f64) / 100.0 < 0.75 * m.window_s);
}

#[test]
fn a_link_that_stops_reporting_decays_on_the_progress_clock() {
    let g = two_node_geometry();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let mut rng = Rng::new(14);
    let mut t = 0.0;
    for i in 0..40u64 {
        m.update(t, 0, 1, &frame(&mut rng, 10.0, walking_gain(t), 0.2, 16, i))
            .unwrap();
        m.update(t, 1, 0, &frame(&mut rng, 10.0, walking_gain(t), 0.2, 16, i))
            .unwrap();
        t += 1.0 / RATE_HZ;
    }
    let index = g.link_index(0, 1).unwrap();
    assert!(m.vector(None)[index] > 0.0);
    for i in 0..40u64 {
        m.update(t, 1, 0, &frame(&mut rng, 10.0, 1.0, 0.2, 16, i))
            .unwrap();
        t += 1.0 / RATE_HZ;
    }
    assert_eq!(m.vector(None)[index], 0.0);
    assert_eq!(m.sample_counts(None)[index], 0);
}

#[test]
fn a_clock_restart_recovers() {
    let g = two_node_geometry();
    let index = g.link_index(0, 1).unwrap();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let mut rng = Rng::new(15);
    for i in 0..60u64 {
        m.update(
            100.0 + i as f64 / RATE_HZ,
            0,
            1,
            &frame(&mut rng, 10.0, 1.0, 0.2, 16, i),
        )
        .unwrap();
    }
    assert!(m.ready_mask(None)[index]);
    for i in 0..60u64 {
        m.update(
            i as f64 / RATE_HZ,
            0,
            1,
            &frame(&mut rng, 10.0, 1.0, 0.2, 16, i),
        )
        .unwrap();
    }
    assert!(
        m.ready_mask(None)[index],
        "clock restart left the link starved"
    );
    assert_eq!(m.clock_restarts, 1);
    assert_eq!(
        m.stale_samples, 1,
        "the first stale sample is dropped, the second confirms"
    );
}

// ------------------------------------------------------------ clock domains

#[test]
fn one_stale_packet_perturbs_nothing() {
    let g = four_node_geometry();
    let frames = corpus(&g, 43, 6.0, true, 0.0);
    let mut reference = measurement(&g, 1.0, RATE_HZ);
    feed(&mut reference, &frames);
    let counts = reference.sample_counts(None);
    let expected = reference.vector(None);
    assert!(counts.iter().all(|&c| c == 13), "{counts:?}");

    let mut m = measurement(&g, 1.0, RATE_HZ);
    feed(&mut m, &frames);
    let t_latest = frames.last().unwrap().0;
    m.update(t_latest - 1.5, 0, 1, &frames[0].3).unwrap();
    assert_eq!(m.sample_counts(None), counts);
    assert!(m.ready_mask(None).iter().all(|&r| r));
    assert_eq!(m.vector(None), expected);
    assert_eq!((m.stale_samples, m.clock_restarts), (1, 0));
}

#[test]
fn sustained_stale_packets_never_blind_the_mesh() {
    let g = four_node_geometry();
    let frames = corpus(&g, 43, 6.0, true, 0.0);
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let mut previous_t = None;
    for (t, tx, rx, f) in &frames {
        m.update(*t, *tx, *rx, f).unwrap();
        if previous_t.is_some_and(|p| p != *t) {
            m.update(t - 1.7, 0, 1, f).unwrap();
        }
        previous_t = Some(*t);
    }
    assert!(m.ready_mask(None).iter().all(|&r| r));
    assert!(m.sample_counts(None).iter().all(|&c| c == 13));
    assert!(m.stale_samples > 50);
    assert_eq!(m.clock_restarts, 0);
}

#[test]
fn a_constant_node_clock_offset_costs_nothing() {
    let g = two_node_geometry();
    let mut synced = measurement(&g, 1.0, RATE_HZ);
    let mut skewed = measurement(&g, 1.0, RATE_HZ);
    let mut rng = Rng::new(16);
    for i in 0..120u64 {
        let t = i as f64 / RATE_HZ;
        let one = frame(&mut rng, 10.0, walking_gain(t), 0.2, 16, i);
        let two = frame(&mut rng, 10.0, walking_gain(t), 0.2, 16, i);
        synced.update(t, 0, 1, &one).unwrap();
        synced.update(t, 1, 0, &two).unwrap();
        skewed.update(t - 1.5, 0, 1, &one).unwrap();
        skewed.update(t, 1, 0, &two).unwrap();
    }
    assert!(skewed.ready_mask(None).iter().all(|&r| r));
    assert_eq!(skewed.sample_counts(None), synced.sample_counts(None));
    assert_eq!((skewed.clock_restarts, skewed.stale_samples), (0, 0));
    for (a, b) in skewed.vector(None).iter().zip(synced.vector(None)) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn restarts_stay_on_their_link_and_a_whole_replay_converges() {
    let g = four_node_geometry();
    let frames = corpus(&g, 43, 6.0, false, 0.0);
    let mut m = measurement(&g, 1.0, RATE_HZ);
    feed(&mut m, &frames);
    let rebooted = g.link_index(0, 1).unwrap();
    let others: Vec<usize> = (0..g.n_links()).filter(|&i| i != rebooted).collect();
    let mut rng = Rng::new(20);
    let mut mesh_t = frames.last().unwrap().0 + 1.0 / RATE_HZ;
    for i in 0..30u64 {
        for &index in &others {
            let l = g.links()[index];
            m.update(
                mesh_t,
                l.tx_id,
                l.rx_id,
                &frame(&mut rng, 10.0, 1.0, 0.2, 16, i),
            )
            .unwrap();
        }
        m.update(
            i as f64 / RATE_HZ,
            0,
            1,
            &frame(&mut rng, 10.0, 1.0, 0.2, 16, i),
        )
        .unwrap();
        mesh_t += 1.0 / RATE_HZ;
        let counts = m.sample_counts(None);
        for &index in &others {
            assert_eq!(counts[index], 13, "link {index} lost samples");
        }
    }
    assert_eq!(m.clock_restarts, 1);
    assert!(m.ready_mask(None)[rebooted]);

    let mut replayed = measurement(&g, 1.0, RATE_HZ);
    feed(&mut replayed, &frames);
    feed(&mut replayed, &frames);
    let mut fresh = measurement(&g, 1.0, RATE_HZ);
    feed(&mut fresh, &frames);
    assert_eq!(replayed.clock_restarts as usize, g.n_links());
    assert_eq!(replayed.sample_counts(None), fresh.sample_counts(None));
    for (a, b) in replayed.vector(None).iter().zip(fresh.vector(None)) {
        assert!((a - b).abs() < 1e-12);
    }
}

#[test]
fn a_late_sample_never_outlives_its_window() {
    let g = two_node_geometry();
    let index = g.link_index(0, 1).unwrap();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let mut rng = Rng::new(21);
    let step = 1.0 / RATE_HZ;
    let mut t = 0.0;
    for i in 0..30u64 {
        m.update(t, 0, 1, &frame(&mut rng, 10.0, 1.0, 0.2, 16, i))
            .unwrap();
        t += step;
    }
    m.update(
        t - step - 0.95,
        0,
        1,
        &frame(&mut rng, 10.0, 1.0, 0.2, 16, 99),
    )
    .unwrap();
    let window = m.window_samples(index);
    assert!(
        window.windows(2).all(|w| w[0].0 <= w[1].0),
        "the late sample was inserted in order"
    );
    let mut worst = 0.0f64;
    for i in 0..30u64 {
        m.update(t, 0, 1, &frame(&mut rng, 10.0, 1.0, 0.2, 16, i))
            .unwrap();
        let oldest = m
            .window_samples(index)
            .iter()
            .map(|s| s.0)
            .fold(f64::INFINITY, f64::min);
        worst = worst.max(t - oldest);
        t += step;
    }
    assert!(worst <= m.window_s + 1e-9, "a sample survived to {worst} s");
}

#[test]
fn a_stopped_stream_reads_as_stopped_once_told_the_time() {
    let g = two_node_geometry();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let mut rng = Rng::new(22);
    let mut t = 0.0;
    for i in 0..40u64 {
        for l in g.links() {
            m.update(
                t,
                l.tx_id,
                l.rx_id,
                &frame(&mut rng, 10.0, walking_gain(t), 0.2, 16, i),
            )
            .unwrap();
        }
        t += 1.0 / RATE_HZ;
    }
    assert!(m.ready_mask(None).iter().all(|&r| r));
    let held = m.vector(None);
    assert!(held.iter().any(|&v| v > 0.0));
    assert_eq!(m.vector(None), held);
    let later = t + 3600.0;
    assert_eq!(m.vector(Some(later)), vec![0.0; g.n_links()]);
    assert!(!m.ready_mask(Some(later)).iter().any(|&r| r));
    assert_eq!(m.sample_counts(Some(later)), vec![0; g.n_links()]);
}

#[test]
fn tdma_slot_timestamps_give_every_link_the_same_window() {
    let g = four_node_geometry();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    feed(&mut m, &corpus(&g, 23, 5.0, false, 1.0 / (RATE_HZ * 4.0)));
    let counts = m.sample_counts(None);
    assert!(counts.iter().all(|&c| c == counts[0]), "{counts:?}");
    assert!(m.ready_mask(None).iter().all(|&r| r));
}

// ------------------------------------------------------------ calibration

#[test]
fn calibration_counts_independent_windows() {
    let g = four_node_geometry();
    for duration_s in [1.3, 2.0, 5.0] {
        let mut m = measurement(&g, 1.0, RATE_HZ);
        let err = m
            .calibrate_from(corpus(&g, 31, duration_s, false, 0.0), 10)
            .unwrap_err();
        assert!(matches!(err, LocateError::ValueError(_)));
        assert!(!m.is_calibrated());
    }
    let mut m = measurement(&g, 1.0, RATE_HZ);
    m.calibrate_from(corpus(&g, 31, 20.0, false, 0.0), 10)
        .unwrap();
    assert!(m.calibrated_mask().iter().all(|&ok| ok));
    assert!(m.is_calibrated());
    assert!(
        m.sample_counts(None).iter().all(|&c| c == 0),
        "windows are cleared afterwards"
    );
    let mut m = measurement(&g, 1.0, RATE_HZ);
    assert!(m
        .calibrate_from(corpus(&g, 31, 20.0, false, 0.0), 200)
        .is_err());
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let empty: Vec<(f64, i64, i64, CsiFrame)> = Vec::new();
    assert!(matches!(
        m.calibrate_from(empty, 10),
        Err(LocateError::ValueError(_))
    ));
}

#[test]
fn calibration_removes_the_baseline_and_predicts_the_null() {
    let g = four_node_geometry();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    m.calibrate_from(corpus(&g, 31, 30.0, false, 0.0), 10)
        .unwrap();
    let baseline = m.baseline_vector();
    let null = m.baseline_std_vector();
    assert!(baseline.iter().all(|&b| b > 0.0) && null.iter().all(|&s| s > 0.0));
    assert_eq!(m.baseline_raw_std_vector(), null);
    // Fresh empty data reads near zero against the null.
    feed(&mut m, &corpus(&g, 99, 4.0, false, 0.0));
    for (y, s) in m.vector(None).iter().zip(&null) {
        assert!(y.abs() < 6.0 * s, "y {y} null sigma {s}");
    }
}

#[test]
fn uncalibrated_links_are_imputed_with_the_median_and_flagged() {
    let g = four_node_geometry();
    let missing = [(0i64, 1i64), (1, 0)];
    let partial: Vec<_> = corpus(&g, 31, 20.0, false, 0.0)
        .into_iter()
        .filter(|(_, tx, rx, _)| !missing.contains(&(*tx, *rx)))
        .collect();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    m.calibrate_from(partial.iter().map(|(t, tx, rx, f)| (*t, *tx, *rx, f)), 10)
        .unwrap();
    let mask = m.calibrated_mask();
    let baseline = m.baseline_vector();
    let mut measured: Vec<f64> = baseline
        .iter()
        .zip(&mask)
        .filter(|(_, &ok)| ok)
        .map(|(&b, _)| b)
        .collect();
    measured.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = measured.len();
    let median = if n % 2 == 1 {
        measured[n / 2]
    } else {
        0.5 * (measured[n / 2 - 1] + measured[n / 2])
    };
    for (tx, rx) in missing {
        let index = g.link_index(tx, rx).unwrap();
        assert!(!mask[index]);
        assert_eq!(baseline[index], median);
    }
    assert_eq!(mask.iter().filter(|&&ok| ok).count(), g.n_links() - 2);
}

#[test]
fn set_baseline_round_trips_and_refuses_corruption() {
    let g = four_node_geometry();
    let n = g.n_links();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let baseline: Vec<f64> = (0..n).map(|i| 0.1 * (i + 1) as f64).collect();
    let sigmas: Vec<f64> = (0..n).map(|i| 0.01 * (i + 1) as f64).collect();
    m.set_baseline(&baseline, Some(&sigmas)).unwrap();
    assert!(m.is_calibrated());
    assert_eq!(m.baseline_vector(), baseline);
    assert_eq!(m.baseline_std_vector(), sigmas);
    assert!(m.calibrated_mask().iter().all(|&ok| ok));
    assert!(matches!(
        m.set_baseline(&[0.0, 1.0], None),
        Err(LocateError::ValueError(_))
    ));

    let mut fresh = measurement(&g, 1.0, RATE_HZ);
    for bad in [f64::NAN, f64::INFINITY, -1.0] {
        let mut values = vec![1.0; n];
        values[2] = bad;
        assert!(fresh.set_baseline(&values, Some(&vec![0.1; n])).is_err());
        let mut stds = vec![0.1; n];
        stds[0] = bad;
        assert!(fresh.set_baseline(&vec![1.0; n], Some(&stds)).is_err());
    }
    assert!(!fresh.is_calibrated());

    fresh.set_baseline(&vec![1.0; n], None).unwrap();
    assert!(fresh.is_calibrated());
    assert!(!fresh.calibrated_mask().iter().any(|&ok| ok));
    assert_eq!(fresh.baseline_std_vector(), vec![0.0; n]);
}

#[test]
fn normalize_mode_and_its_unusable_links() {
    let g = four_node_geometry();
    let n = g.n_links();
    let config = MeasurementConfig {
        calibration_mode: CalibrationMode::Normalize,
        ..Default::default()
    };
    let mut m = LinkMeasurement::new(g.clone(), config, Box::new(TotalPowerSignal)).unwrap();
    let mut baseline = vec![1.0; n];
    baseline[0] = 0.0;
    m.set_baseline(&baseline, Some(&vec![0.1; n])).unwrap();
    feed(&mut m, &corpus(&g, 43, 4.0, true, 0.0));
    assert!(!m.calibrated_mask()[0]);
    assert!(!m.ready_mask(None)[0]);
    assert_eq!(m.vector(None)[0], 0.0);
    assert!(m.ready_mask(None)[1..].iter().all(|&r| r));
    let null = m.baseline_std_vector();
    assert_eq!(null[0], 0.0);
    assert_eq!(null[1], 0.1, "std / baseline");

    // y = var / baseline - 1 against the raw variance.
    let mut raw = measurement(&g, 1.0, RATE_HZ);
    feed(&mut raw, &corpus(&g, 43, 4.0, true, 0.0));
    let (var, y) = (raw.vector(None), m.vector(None));
    for i in 1..n {
        assert_eq!(y[i], var[i] / 1.0 - 1.0);
    }
}

#[test]
fn clamp_negative_and_reset() {
    let g = four_node_geometry();
    let n = g.n_links();
    let config = MeasurementConfig {
        clamp_negative: true,
        ..Default::default()
    };
    let mut m = LinkMeasurement::new(g.clone(), config, Box::new(TotalPowerSignal)).unwrap();
    m.set_baseline(&vec![1e6; n], Some(&vec![1.0; n])).unwrap();
    feed(&mut m, &corpus(&g, 31, 4.0, false, 0.0));
    assert!(
        m.vector(None).iter().all(|&y| y == 0.0),
        "every y is negative and clamped"
    );

    m.reset(true);
    assert!(m.is_calibrated());
    assert!(m.sample_counts(None).iter().all(|&c| c == 0));
    m.reset(false);
    assert!(!m.is_calibrated());
}

// ------------------------------------------------------------ dropped samples and signals

#[test]
fn bad_frames_are_dropped_not_substituted() {
    let g = two_node_geometry();
    let index = g.link_index(0, 1).unwrap();
    let mut m = measurement(&g, 1.0, RATE_HZ);
    let mut rng = Rng::new(7);
    let mut t = 0.0;
    for i in 0..20u64 {
        m.update(t, 0, 1, &frame(&mut rng, 10.0, 1.0, 0.2, 16, i))
            .unwrap();
        t += 1.0 / RATE_HZ;
    }
    let before = m.sample_counts(None)[index];
    let bad_frames = [
        new_csi_frame(vec![f64::NAN; 16], vec![0.0; 16], 0, RATE_HZ, -50, 1).unwrap(),
        new_csi_frame(vec![0.0; 16], vec![0.0; 16], 0, RATE_HZ, -50, 1).unwrap(),
        new_csi_frame(vec![1e200; 16], vec![0.0; 16], 0, RATE_HZ, -50, 1).unwrap(),
        new_csi_frame(Vec::new(), Vec::new(), 0, RATE_HZ, -50, 1).unwrap(),
    ];
    for f in &bad_frames {
        assert_eq!(frame_power(f), None);
        m.update(t - 0.01, 0, 1, f).unwrap();
    }
    assert_eq!(m.sample_counts(None)[index], before);
    // Mean, not total: 2 subcarriers of 3.0 give 9.0.
    let f = new_csi_frame(vec![3.0, 3.0], vec![0.0, 0.0], 0, RATE_HZ, -50, 1).unwrap();
    assert_eq!(frame_power(&f), Some(9.0));
}

#[test]
fn the_three_signals() {
    let f = new_csi_frame(vec![3.0, 4.0], vec![0.0, 0.0], 5, RATE_HZ, -61, 2).unwrap();
    let bad = new_csi_frame(vec![0.0, 0.0], vec![0.0, 0.0], 5, RATE_HZ, -61, 2).unwrap();

    assert_eq!(TotalPowerSignal.name(), "total_power_db");
    let mut tp = TotalPowerSignal.new_state(RATE_HZ);
    assert_eq!(tp.push(&f).unwrap(), Some(10.0 * 12.5f64.log10()));
    assert_eq!(tp.push(&bad).unwrap(), None);

    assert_eq!(RssiSignal.name(), "rssi_dbm");
    let mut rs = RssiSignal.new_state(RATE_HZ);
    assert_eq!(rs.push(&f).unwrap(), Some(-61.0));
    assert_eq!(
        rs.push(&bad).unwrap(),
        None,
        "the shared validity gate applies"
    );

    let ss_signal = StaticSuppressedSignal::default();
    assert_eq!(ss_signal.corner_hz, 0.03);
    assert_eq!(ss_signal.name(), "static_suppressed_amplitude");
    let mut ss = ss_signal.new_state(RATE_HZ);
    assert_eq!(ss.push(&bad).unwrap(), None, "a bad frame does not seed");
    assert_eq!(
        ss.push(&f).unwrap(),
        None,
        "the first good frame only seeds"
    );
    let g = new_csi_frame(vec![3.5, 4.0], vec![0.0, 0.0], 6, RATE_HZ, -61, 2).unwrap();
    let v = ss.push(&g).unwrap().unwrap();
    // Residuals (0.5, 0.0); the second subcarrier's std is 0, so all the
    // weight is on the first: CPython 3.14 gives exactly 0.5.
    assert_eq!(v, 0.5);
    ss.reset();
    assert_eq!(ss.push(&g).unwrap(), None, "reset forgets the suppressor");

    // A non-positive per-link rate is refused when the suppressor is built.
    let mut zero_rate = StaticSuppressedSignal::default().new_state(0.0);
    assert!(matches!(
        zero_rate.push(&f),
        Err(LocateError::ValueError(_))
    ));
}

#[test]
fn rssi_measurement_is_the_variance_of_the_integer_readings() {
    let g = two_node_geometry();
    let index = g.link_index(0, 1).unwrap();
    let config = MeasurementConfig {
        window_s: 1.0,
        sample_rate_hz: 10.0,
        ..Default::default()
    };
    let mut m = LinkMeasurement::new(g.clone(), config, Box::new(RssiSignal)).unwrap();
    let readings = [-50, -51, -50, -52, -50, -51, -50, -50, -53, -50, -51];
    for (i, &r) in readings.iter().enumerate() {
        let f = new_csi_frame(vec![1.0; 4], vec![0.0; 4], i as u64, 10.0, r, 1).unwrap();
        m.update(i as f64 / 10.0, 0, 1, &f).unwrap();
    }
    assert_eq!(m.sample_counts(None)[index], 11);
    let values: Vec<f64> = readings.iter().map(|&r| r as f64).collect();
    let (_, std) = mean_std(&values);
    assert!((m.vector(None)[index] - std * std).abs() < 1e-12);
}
