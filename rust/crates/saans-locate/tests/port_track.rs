//! Port of the key cases of tests/test_localize_track.py, plus the
//! `cov_xx`/`cov_xy`/`cov_yy` fields of spec/locate.md §3.
#![allow(clippy::needless_range_loop, clippy::type_complexity)]

use std::sync::Arc;

use saans_locate::geometry::*;
use saans_locate::pynum::LocateError;
use saans_locate::track::*;
use saans_sim::rng::Rng;

fn g8(pitch: f64) -> Arc<Geometry> {
    Arc::new(Geometry::perimeter(8, None, pitch, 0.3, 0.1, 0.0).unwrap())
}

fn blob(g: &Geometry, cx: f64, cy: f64, sigma_m: f64, amplitude: f64) -> Vec<f64> {
    let two_s2 = 2.0 * sigma_m * sigma_m;
    g.grid()
        .pixels()
        .iter()
        .map(|&(x, y)| amplitude * (-((x - cx).powi(2) + (y - cy).powi(2)) / two_s2).exp())
        .collect()
}

fn noise(g: &Geometry, rng: &mut Rng) -> Vec<f64> {
    (0..g.n_pixels()).map(|_| rng.gauss()).collect()
}

fn tracker(g: &Arc<Geometry>, meas_noise_m: f64, max_coast_s: f64) -> Tracker {
    Tracker::new(
        g.clone(),
        TrackerConfig {
            meas_noise_m: Some(meas_noise_m),
            max_coast_s,
            ..Default::default()
        },
    )
    .unwrap()
}

// ------------------------------------------------------------ extract_peak

#[test]
fn flat_images_abstain_at_every_magnitude() {
    let g = g8(0.5);
    for level in [0.0, 1.0, -3.5, 1e9, 1e308, -1e308, 8.99e307] {
        let image = vec![level; g.n_pixels()];
        let mut t = Tracker::with_defaults(g.clone()).unwrap();
        assert_eq!(t.update(&image, 0.0).unwrap(), None);
        assert!(!t.is_tracking());
        let reading = extract_peak(&g, &image, 1.0).unwrap();
        assert!(reading.background.is_finite());
        assert_eq!(reading.z, 0.0);
        assert_eq!(reading.tie_fraction, 1.0);
        assert_eq!(reading.peak_index, Some(0), "ties go to the first index");
    }
}

#[test]
fn non_finite_images_abstain_with_no_index() {
    let g = g8(0.5);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut image = blob(&g, 4.0, 3.0, 0.5, 1.0);
        image[17] = bad;
        let reading = extract_peak(&g, &image, 1.0).unwrap();
        assert_eq!(
            (reading.z, reading.n_nonfinite, reading.peak_index),
            (0.0, 1, None)
        );
        assert!(reading.x.is_nan() && reading.y.is_nan() && reading.background.is_nan());
        assert_eq!(reading.spread, 0.0);
        assert_eq!(
            Tracker::with_defaults(g.clone())
                .unwrap()
                .update(&image, 0.0)
                .unwrap(),
            None
        );
    }
}

#[test]
fn a_lone_spike_on_a_flat_image_abstains() {
    for (pitch, n) in [(0.5, 192), (0.25, 768)] {
        let g = g8(pitch);
        assert_eq!(g.n_pixels(), n);
        for spike in [1e-300, 1e-12, 1.0, 1e9] {
            let mut image = vec![0.0; n];
            image[100] = spike;
            let reading = extract_peak(&g, &image, 1.0).unwrap();
            assert_eq!(reading.z, 0.0);
            assert!(reading.tie_fraction > 0.9);
            assert_eq!(reading.peak_index, Some(100));
            assert_eq!(
                Tracker::with_defaults(g.clone())
                    .unwrap()
                    .update(&image, 0.0)
                    .unwrap(),
                None
            );
        }
    }
}

#[test]
fn pinned_blocks_are_the_background() {
    let g = g8(0.5);
    // Rectified noise: about half the pixels pinned at exactly zero.
    let mut rng = Rng::new(90000);
    let mut fixes = 0;
    for _ in 0..100 {
        let image: Vec<f64> = (0..g.n_pixels()).map(|_| rng.gauss().max(0.0)).collect();
        let reading = extract_peak(&g, &image, 1.0).unwrap();
        assert_eq!(reading.background, 0.0);
        assert!(reading.tie_fraction > 0.3 && reading.tie_fraction < 0.7);
        if let Some(e) = Tracker::with_defaults(g.clone())
            .unwrap()
            .update(&image, 0.0)
            .unwrap()
        {
            fixes += 1;
            assert!(e.confidence < 0.40);
        }
    }
    assert!(fixes <= 2, "{fixes} rectified-noise fixes");

    // The deterministic witness: 96 zeros, one 1e-9, 94 ones and a 2.
    let mut image = vec![0.0; 96];
    image.push(1e-9);
    image.extend(vec![1.0; 94]);
    image.push(2.0);
    let reading = extract_peak(&g, &image, 1.0).unwrap();
    assert!(reading.spread > 0.5);
    assert!(reading.z < DEFAULT_Z_THRESHOLD);
    assert_eq!(
        reading.background, 0.0,
        "the 96-zero run is the dominant pinned block"
    );
    assert_eq!(
        Tracker::with_defaults(g.clone())
            .unwrap()
            .update(&image, 0.0)
            .unwrap(),
        None
    );
}

#[test]
fn z_responds_to_spike_height_when_there_is_a_floor() {
    let g = g8(0.5);
    let mut rng = Rng::new(7);
    let floor = noise(&g, &mut rng);
    let mut previous = 0.0;
    for height in [3.0, 6.0, 12.0, 24.0] {
        let mut image = floor.clone();
        image[100] = height;
        let reading = extract_peak(&g, &image, 1.0).unwrap();
        assert!(reading.z > previous);
        previous = reading.z;
        assert_eq!(reading.tie_fraction, 0.0);
    }
    assert!(previous > DEFAULT_Z_THRESHOLD);
}

#[test]
fn extreme_values_neither_overflow_the_median_nor_emit_a_position() {
    let g = g8(0.5);
    let image: Vec<f64> = (0..g.n_pixels())
        .map(|i| 1.0e308 + i as f64 * 3.0e305)
        .collect();
    let reading = extract_peak(&g, &image, 1.0).unwrap();
    assert!(reading.background.is_finite());
    assert_eq!(reading.background, 0.5 * image[95] + 0.5 * image[96]);
    assert!(reading.spread.is_finite());
    assert!(reading.z < DEFAULT_Z_THRESHOLD);

    // A finite image whose weighted centroid overflows.
    let mut rng = Rng::new(2);
    let mut image = vec![0.0; 100];
    image.extend((0..92).map(|_| rng.gauss().abs()));
    for index in [191, 190, 189, 175, 159] {
        image[index] = 1.7e308;
    }
    let reading = extract_peak(&g, &image, 1.0).unwrap();
    assert!(!reading.x.is_finite());
    assert!(reading.z > DEFAULT_Z_THRESHOLD);
    let mut t = Tracker::with_defaults(g.clone()).unwrap();
    assert_eq!(t.update(&image, 0.0).unwrap(), None);
    assert!(!t.is_tracking());

    // Half at -1e308, half at +1e308; then immediate recovery.
    let mut t = Tracker::with_defaults(g.clone()).unwrap();
    let mut image = vec![-1e308; 97];
    image.extend(vec![1e308; 95]);
    assert_eq!(t.update(&image, 0.0).unwrap(), None);
    assert!(!t.is_tracking());
    let mut last = None;
    for i in 1..6 {
        last = t
            .update(&blob(&g, 4.0, 3.0, 0.5, 1.0), i as f64 * 0.4)
            .unwrap();
        let e = last.expect("recovered");
        assert!(e.x.is_finite() && e.y.is_finite());
    }
    let e = last.unwrap();
    assert!(((e.x - 4.0).powi(2) + (e.y - 3.0).powi(2)).sqrt() < 0.35);
}

#[test]
fn ties_go_to_the_first_maximum() {
    let g = g8(0.5);
    let mut rng = Rng::new(3);
    let mut image = noise(&g, &mut rng);
    image[60] = 9.0;
    image[50] = 9.0;
    assert_eq!(extract_peak(&g, &image, 1.0).unwrap().peak_index, Some(50));
}

#[test]
fn centroid_window_is_symmetric_at_walls_and_sub_pixel_inside() {
    for pitch in [0.5, 0.25] {
        let g = g8(pitch);
        let n_x = g.grid().n_x();
        for corner in [0, n_x - 1, g.n_pixels() - 1] {
            let (px, py) = g.grid().pixels()[corner];
            let reading = extract_peak(&g, &blob(&g, px, py, 0.5, 1.0), 1.0).unwrap();
            assert!(
                (reading.x - px).abs() < 1e-9 && (reading.y - py).abs() < 1e-9,
                "corner {corner}"
            );
        }
    }
    let g = g8(0.5);
    let grid = g.grid();
    let left = grid.pixels()[grid.pixel_index(7, 6).unwrap()];
    let right = grid.pixels()[grid.pixel_index(8, 6).unwrap()];
    let cx = 0.5 * (left.0 + right.0);
    let reading = extract_peak(&g, &blob(&g, cx, left.1, 0.5, 1.0), 1.0).unwrap();
    assert!((reading.x - cx).abs() < 0.05 * grid.pitch_x_m());
    assert!((reading.y - left.1).abs() < 0.05 * grid.pitch_y_m());
}

#[test]
fn extract_peak_validation() {
    let g = g8(0.5);
    let image = vec![0.0; g.n_pixels()];
    assert!(matches!(
        extract_peak(&g, &[0.0; 3], 1.0),
        Err(LocateError::ValueError(_))
    ));
    for radius in [0.0, -1.0, f64::NAN] {
        assert!(matches!(
            extract_peak(&g, &image, radius),
            Err(LocateError::ValueError(_))
        ));
    }
    // `inf > 0` passes; int(inf / pitch) then raises OverflowError.
    assert!(matches!(
        extract_peak(&g, &image, f64::INFINITY),
        Err(LocateError::OverflowError(_))
    ));
    let reading = extract_peak(&g, &blob(&g, 4.0, 3.0, 0.5, 1.0), 1.0).unwrap();
    assert!(reading.peak_index.is_some());
    assert_eq!(reading.tie_fraction, 0.0);
}

// ------------------------------------------------------------ tracker

#[test]
fn tracker_constructor_rejects_nonsense_including_nan() {
    let g = g8(0.5);
    let nan = f64::NAN;
    let base = TrackerConfig::default();
    let cases = [
        TrackerConfig {
            z_threshold: 0.0,
            ..base
        },
        TrackerConfig {
            z_threshold: -1.0,
            ..base
        },
        TrackerConfig {
            z_threshold: nan,
            ..base
        },
        TrackerConfig {
            centroid_radius_m: 0.0,
            ..base
        },
        TrackerConfig {
            centroid_radius_m: nan,
            ..base
        },
        TrackerConfig {
            accel_psd: 0.0,
            ..base
        },
        TrackerConfig {
            accel_psd: nan,
            ..base
        },
        TrackerConfig {
            meas_noise_m: Some(-0.5),
            ..base
        },
        TrackerConfig {
            meas_noise_m: Some(nan),
            ..base
        },
        TrackerConfig {
            init_speed_std_ms: 0.0,
            ..base
        },
        TrackerConfig {
            init_speed_std_ms: nan,
            ..base
        },
        TrackerConfig {
            max_coast_s: -1.0,
            ..base
        },
        TrackerConfig {
            max_coast_s: nan,
            ..base
        },
    ];
    for config in cases {
        assert!(
            matches!(
                Tracker::new(g.clone(), config),
                Err(LocateError::ValueError(_))
            ),
            "{config:?}"
        );
    }
    for pitch in [0.5, 0.25] {
        assert!((Tracker::with_defaults(g8(pitch)).unwrap().meas_noise_m - pitch).abs() < 1e-12);
    }
}

#[test]
fn confidence_is_monotonic_bounded_and_nan_safe() {
    let t = Tracker::with_defaults(g8(0.5)).unwrap();
    assert_eq!(t.confidence_from_z(0.0), 0.0);
    assert_eq!(t.confidence_from_z(-5.0), 0.0);
    assert_eq!(t.confidence_from_z(DEFAULT_Z_THRESHOLD - 0.01), 0.0);
    assert_eq!(t.confidence_from_z(DEFAULT_Z_THRESHOLD), 0.0);
    assert_eq!(t.confidence_from_z(2.0 * DEFAULT_Z_THRESHOLD), 0.5);
    let mut previous = -1.0;
    for z in [5.0, 6.0, 8.0, 10.0, 25.0, 100.0, 1e6] {
        let c = t.confidence_from_z(z);
        assert!(c >= previous && c <= 1.0);
        previous = c;
    }
    for z in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(t.confidence_from_z(z), 0.0);
    }
}

#[test]
fn estimates_carry_the_covariance_they_report() {
    let g = g8(0.5);
    let mut t = tracker(&g, 0.3, 3.0);
    let mut rng = Rng::new(11);
    for i in 0..20 {
        let cx = 1.0 + 0.2 * i as f64 + 0.1 * rng.gauss();
        let est = t
            .update(&blob(&g, cx, 2.0, 0.5, 1.0), i as f64 * 0.4)
            .unwrap()
            .expect("fix");
        let p = t.covariance().unwrap();
        assert_eq!(
            (est.cov_xx, est.cov_xy, est.cov_yy),
            (p[0][0], p[0][1], p[1][1])
        );
        assert_eq!(est.sigma_m, (p[0][0] + p[1][1]).max(0.0).sqrt());
        assert_eq!(Some(est.sigma_m), t.sigma_m());
        assert_eq!(est.t, i as f64 * 0.4);
        assert!(est.cov_xx > 0.0 && est.cov_yy > 0.0);
    }
    // The first estimate's covariance is the initial r = (meas_noise * (2 - c))^2.
    let mut t = tracker(&g, 0.3, 3.0);
    let est = t
        .update(&blob(&g, 4.0, 3.0, 0.5, 1.0), 1.25)
        .unwrap()
        .unwrap();
    let r = (0.3 * (2.0 - est.confidence)) * (0.3 * (2.0 - est.confidence));
    assert_eq!((est.cov_xx, est.cov_xy, est.cov_yy), (r, 0.0, r));
}

#[test]
fn prediction_matches_the_cwna_integral() {
    let g = g8(0.5);
    let (dt, q) = (0.7, 0.1);
    let mut t = Tracker::new(
        g.clone(),
        TrackerConfig {
            meas_noise_m: Some(0.3),
            accel_psd: q,
            ..Default::default()
        },
    )
    .unwrap();
    let est = t
        .update(&blob(&g, 4.0, 3.0, 0.5, 1.0), 0.0)
        .unwrap()
        .unwrap();
    let r = est.cov_xx;
    // An abstaining frame predicts and returns nothing.
    assert_eq!(t.update(&vec![0.0; g.n_pixels()], dt).unwrap(), None);
    let p = t.covariance().unwrap();
    let pp = r + dt * dt * 1.0 + q * dt.powi(3) / 3.0;
    let pv = dt * 1.0 + q * dt * dt / 2.0;
    let vv = 1.0 + q * dt;
    for (axis, vel) in [(0, 2), (1, 3)] {
        assert!((p[axis][axis] - pp).abs() < 1e-12);
        assert!((p[axis][vel] - pv).abs() < 1e-12);
        assert!((p[vel][vel] - vv).abs() < 1e-12);
    }
    assert_eq!((p[0][1], p[0][3], p[1][2], p[2][3]), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn joseph_form_survives_a_vanishing_r() {
    let g = g8(0.5);
    let mut t = tracker(&g, 1e-9, 3.0);
    let image = blob(&g, 4.0, 3.0, 0.5, 1.0);
    for i in 0..300 {
        t.update(&image, i as f64 * 0.4).unwrap().expect("fix");
        let p = t.covariance().unwrap();
        for a in 0..4 {
            assert!(p[a][a] > 0.0, "diagonal collapsed at step {i}");
        }
    }
}

#[test]
fn the_room_clamp_binds_on_an_emitted_frame() {
    let g = g8(0.5);
    let mut t = tracker(&g, 1.0, 3.0);
    let mut time = 0.0;
    for _ in 0..14 {
        let cx = 1.0 + 1.2 * time;
        if cx > 7.6 {
            break;
        }
        t.update(&blob(&g, cx, 3.0, 0.5, 1.0), time).unwrap();
        time += 0.4;
    }
    for _ in 0..2 {
        assert_eq!(t.update(&vec![0.0; g.n_pixels()], time).unwrap(), None);
        time += 0.4;
    }
    let est = t
        .update(&blob(&g, 7.75, 3.0, 0.5, 1.0), time)
        .unwrap()
        .expect("fix");
    let raw = t.position().unwrap();
    assert!(raw.0 > g.room().width_m, "the state is outside the room");
    assert_eq!(est.x, g.room().width_m);
    assert!(g.room().contains(est.x, est.y, 0.0));
}

/// Walk a line at dt = 0.4 s, replacing frames `gap` with noise.
fn run_with_gap(
    gap: std::ops::Range<usize>,
    max_coast_s: f64,
) -> Vec<(f64, Option<PositionEstimate>, Option<f64>, bool, (f64, f64))> {
    let g = g8(0.5);
    let mut rng = Rng::new(3);
    let mut t = tracker(&g, 0.2, max_coast_s);
    let mut log = Vec::new();
    for i in 0..40 {
        let time = i as f64 * 0.4;
        let (cx, cy) = (1.0 + 0.5 * time, 1.0 + 0.25 * time);
        if cx > 7.0 || cy > 5.0 {
            break;
        }
        let image = if gap.contains(&i) {
            noise(&g, &mut rng)
        } else {
            blob(&g, cx, cy, 0.5, 1.0)
        };
        let est = t.update(&image, time).unwrap();
        log.push((time, est, t.sigma_m(), t.is_tracking(), (cx, cy)));
    }
    log
}

#[test]
fn coasting_grows_sigma_and_a_long_gap_resets() {
    let log = run_with_gap(12..16, 3.0);
    for row in &log[12..16] {
        assert!(row.1.is_none(), "a fix during the gap");
    }
    let before = log[11].2.unwrap();
    let gap: Vec<f64> = log[12..16].iter().map(|r| r.2.unwrap()).collect();
    assert!(gap[0] > before);
    assert!(gap.windows(2).all(|w| w[1] > w[0]));
    let after: Vec<&PositionEstimate> = log[16..].iter().filter_map(|r| r.1.as_ref()).collect();
    assert!(after.len() > 2);
    assert!(after[0].sigma_m < gap[3]);
    assert!(after[2].sigma_m < 1.5 * before);
    let (truth_x, truth_y) = log[16].4;
    assert!(((after[0].x - truth_x).powi(2) + (after[0].y - truth_y).powi(2)).sqrt() < 0.5);

    let log = run_with_gap(8..22, 3.0);
    assert!(log[8].3, "coasts first");
    assert!(!log[21].3, "reset after the gap");
    assert!(log[21].2.is_none());
    let last_fix_t = log[..8]
        .iter()
        .filter(|r| r.1.is_some())
        .map(|r| r.0)
        .fold(f64::MIN, f64::max);
    let first_dropped = log[8..22].iter().find(|r| !r.3).unwrap().0;
    assert!(first_dropped - last_fix_t > 3.0);
    assert!(first_dropped - last_fix_t <= 3.0 + 0.4 + 1e-9);
    let reacquired = log[22..]
        .iter()
        .find_map(|r| r.1.map(|e| (e, r.4)))
        .unwrap();
    let (e, (cx, cy)) = reacquired;
    assert!(((e.x - cx).powi(2) + (e.y - cy).powi(2)).sqrt() < 0.4);
}

fn seed_track(t: &mut Tracker, g: &Geometry) -> f64 {
    let mut time = 0.0;
    for i in 0..5 {
        t.update(&blob(g, 1.0 + 0.2 * i as f64, 1.0, 0.5, 1.0), time)
            .unwrap();
        time += 0.4;
    }
    time
}

#[test]
fn a_non_finite_timestamp_is_rejected_and_changes_nothing() {
    let g = g8(0.5);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut t = tracker(&g, 0.3, 3.0);
        seed_track(&mut t, &g);
        let before = (t.position(), t.covariance());
        assert!(matches!(
            t.update(&blob(&g, 2.0, 1.0, 0.5, 1.0), bad),
            Err(LocateError::ValueError(_))
        ));
        assert_eq!((t.position(), t.covariance()), before);
    }
    let mut t = tracker(&g, 0.3, 3.0);
    let time = seed_track(&mut t, &g);
    assert!(t.update(&blob(&g, 2.0, 1.0, 0.5, 1.0), f64::NAN).is_err());
    for i in 0..20 {
        assert_eq!(
            t.update(&vec![0.0; g.n_pixels()], time + i as f64 * 0.4)
                .unwrap(),
            None
        );
    }
    assert!(!t.is_tracking(), "coast expiry never fired");
}

#[test]
fn a_rewound_clock_neither_sustains_a_track_nor_inflates_the_next_interval() {
    let g = g8(0.5);
    let mut t = tracker(&g, 0.3, 3.0);
    seed_track(&mut t, &g);
    let mut rng = Rng::new(7);
    for i in 0..60 {
        assert_eq!(
            t.update(&noise(&g, &mut rng), -0.4 * i as f64).unwrap(),
            None
        );
    }
    assert!(!t.is_tracking());

    let image = blob(&g, 4.0, 3.0, 0.5, 1.0);
    let run = |times: &[f64]| {
        let mut t = tracker(&g, 0.3, 3.0);
        for &time in times {
            t.update(&image, time).unwrap();
        }
        t.sigma_m().unwrap()
    };
    let rewound = run(&[0.0, 0.4, 0.1, 0.8, 1.2]);
    let stalled = run(&[0.0, 0.4, 0.4, 0.8, 1.2]);
    assert_eq!(rewound.to_bits(), stalled.to_bits());
    assert_ne!(rewound, run(&[0.0, 0.4, 0.8, 1.2]));

    let mut t = tracker(&g, 0.3, 3.0);
    t.update(&image, 10.0).unwrap();
    t.update(&image, 10.4).unwrap();
    let before = t.sigma_m().unwrap();
    t.update(&image, 9.0).unwrap();
    let after = t.sigma_m().unwrap();
    assert!(after.is_finite() && after <= before + 1e-9 && after > 0.0);
}

#[test]
fn clock_need_not_start_at_zero_and_reset_clears() {
    let g = g8(0.5);
    let mut t = tracker(&g, 0.3, 3.0);
    for i in 0..6 {
        let est = t
            .update(
                &blob(&g, 4.0 + 0.1 * i as f64, 3.0, 0.5, 1.0),
                1000.0 + i as f64 * 0.4,
            )
            .unwrap();
        assert!(est.is_some());
    }
    assert!(t.is_tracking());
    t.reset();
    assert!(!t.is_tracking());
    assert_eq!(
        (t.position(), t.velocity(), t.sigma_m(), t.covariance()),
        (None, None, None, None)
    );
    assert!(matches!(
        t.update(&vec![0.0; g.n_pixels() + 1], 0.0),
        Err(LocateError::ValueError(_))
    ));
}

#[test]
fn works_at_other_node_counts_and_pitches() {
    for n in [4, 6, 12] {
        for pitch in [0.5, 0.25] {
            let g = Arc::new(Geometry::perimeter(n, None, pitch, 0.3, 0.1, 0.0).unwrap());
            let mut t = Tracker::with_defaults(g.clone()).unwrap();
            let est = t
                .update(&blob(&g, 4.1, 3.1, 0.5, 1.0), 0.0)
                .unwrap()
                .expect("fix");
            assert!(((est.x - 4.1).powi(2) + (est.y - 3.1).powi(2)).sqrt() < 0.35);
            assert_eq!(t.update(&vec![0.0; g.n_pixels()], 0.4).unwrap(), None);
        }
    }
}

#[test]
fn repeated_runs_are_bit_identical() {
    let g = g8(0.5);
    let image = blob(&g, 3.3, 2.7, 0.5, 1.0);
    let run = || {
        let mut t = tracker(&g, 0.3, 3.0);
        (0..20)
            .map(|i| t.update(&image, i as f64 * 0.4).unwrap())
            .collect::<Vec<_>>()
    };
    assert_eq!(run(), run());
}
