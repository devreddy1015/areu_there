//! Port of the key cases of tests/test_localize_invert.py.
#![allow(clippy::needless_range_loop)]

use saans_locate::geometry::*;
use saans_locate::invert::*;
use saans_locate::pynum::{py_hypot, LocateError};
use saans_sim::rng::Rng;

const HAND_A: [[f64; 3]; 3] = [[4.0, 2.0, 0.0], [2.0, 5.0, 2.0], [0.0, 2.0, 5.0]];
const HAND_A_INV: [[f64; 3]; 3] = [
    [21.0 / 64.0, -10.0 / 64.0, 4.0 / 64.0],
    [-10.0 / 64.0, 20.0 / 64.0, -8.0 / 64.0],
    [4.0 / 64.0, -8.0 / 64.0, 16.0 / 64.0],
];

fn m(rows: &[&[f64]]) -> Matrix {
    rows.iter().map(|r| r.to_vec()).collect()
}

fn hand_a() -> Matrix {
    HAND_A.iter().map(|r| r.to_vec()).collect()
}

/// Gauss-Jordan with partial pivoting, independent of the module.
fn gauss_solve(a: &[Vec<f64>], b: &[f64]) -> Vec<f64> {
    let n = a.len();
    let mut m: Vec<Vec<f64>> = a
        .iter()
        .zip(b)
        .map(|(row, &bi)| row.iter().copied().chain([bi]).collect())
        .collect();
    for col in 0..n {
        let pivot = (col..n)
            .max_by(|&r, &s| m[r][col].abs().partial_cmp(&m[s][col].abs()).unwrap())
            .unwrap();
        m.swap(col, pivot);
        let pv = m[col][col];
        for r in 0..n {
            if r == col {
                continue;
            }
            let f = m[r][col] / pv;
            if f != 0.0 {
                for c in col..=n {
                    m[r][c] -= f * m[col][c];
                }
            }
        }
    }
    (0..n).map(|i| m[i][n] / m[i][i]).collect()
}

fn gauss_inverse(a: &[Vec<f64>]) -> Matrix {
    let n = a.len();
    let cols: Vec<Vec<f64>> = (0..n)
        .map(|j| {
            gauss_solve(
                a,
                &(0..n)
                    .map(|i| if i == j { 1.0 } else { 0.0 })
                    .collect::<Vec<_>>(),
            )
        })
        .collect();
    (0..n)
        .map(|i| (0..n).map(|j| cols[j][i]).collect())
        .collect()
}

fn random_spd(n: usize, rng: &mut Rng) -> Matrix {
    let m: Vec<Vec<f64>> = (0..n)
        .map(|_| (0..n).map(|_| rng.range(-1.0, 1.0)).collect())
        .collect();
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    (0..n).map(|k| m[i][k] * m[j][k]).sum::<f64>()
                        + if i == j { n as f64 } else { 0.0 }
                })
                .collect()
        })
        .collect()
}

fn blob(geometry: &Geometry, cx: f64, cy: f64, width_m: f64) -> Vec<f64> {
    let two_w2 = 2.0 * width_m * width_m;
    geometry
        .grid()
        .pixels()
        .iter()
        .map(|&(px, py)| (-((px - cx).powi(2) + (py - cy).powi(2)) / two_w2).exp())
        .collect()
}

fn peak_index(image: &[f64]) -> usize {
    saans_locate::pynum::argmax_first(image.len(), |i| image[i]).unwrap()
}

fn inverter(g: &Geometry, alpha: f64, sigma: f64, delta_c: f64) -> Inverter {
    Inverter::new(g.clone(), alpha, sigma, delta_c, InverterOptions::default()).unwrap()
}

fn max_rel_diff(a: &[f64], b: &[f64]) -> f64 {
    let scale = a.iter().fold(0.0f64, |s, v| s.max(v.abs()));
    a.iter()
        .zip(b)
        .fold(0.0f64, |s, (x, y)| s.max((x - y).abs()))
        / scale
}

// ----------------------------------------------------------------- Cholesky

#[test]
fn cholesky_on_the_hand_computed_matrix() {
    let g = cholesky_factor(&hand_a()).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            let got: f64 = (0..3).map(|k| g[i][k] * g[j][k]).sum();
            assert!((got - HAND_A[i][j]).abs() < 1e-13);
            if j > i {
                assert_eq!(g[i][j], 0.0);
            }
        }
    }
    let x = cholesky_solve(&hand_a(), &[1.0, 2.0, 3.0]).unwrap();
    for (got, want) in x.iter().zip([13.0 / 64.0, 6.0 / 64.0, 36.0 / 64.0]) {
        assert!((got - want).abs() < 5e-15);
    }
    let inv = cholesky_inverse(&hand_a()).unwrap();
    for i in 0..3 {
        for j in 0..3 {
            assert!((inv[i][j] - HAND_A_INV[i][j]).abs() < 5e-15);
            assert_eq!(inv[i][j], inv[j][i], "symmetrized exactly");
        }
    }
}

#[test]
fn cholesky_agrees_with_independent_gaussian_elimination() {
    let mut rng = Rng::new(20261013);
    let mut worst = 0.0f64;
    for _ in 0..40 {
        let n = 2 + (rng.uniform() * 11.0) as usize;
        let a = random_spd(n, &mut rng);
        let b: Vec<f64> = (0..n).map(|_| rng.gauss()).collect();
        let x_chol = cholesky_solve(&a, &b).unwrap();
        let x_gauss = gauss_solve(&a, &b);
        let scale = x_gauss.iter().fold(1e-30f64, |s, v| s.max(v.abs()));
        worst = worst.max(
            x_chol
                .iter()
                .zip(&x_gauss)
                .fold(0.0f64, |s, (p, q)| s.max((p - q).abs()))
                / scale,
        );
    }
    assert!(worst < 1e-10, "worst relative disagreement {worst:e}");
}

#[test]
fn cholesky_refusals_have_the_python_classes() {
    let nan = f64::NAN;
    let cases: Vec<(Matrix, &str)> = vec![
        (m(&[&[4.0, 2.0], &[3.0, 5.0]]), "AsymmetricMatrixError"),
        (m(&[&[1.0, 0.0]]), "MalformedMatrixError"),
        (Vec::new(), "MalformedMatrixError"),
        (m(&[&[nan, 0.0], &[0.0, 1.0]]), "MalformedMatrixError"),
        (m(&[&[1.0, 2.0], &[2.0, 1.0]]), "NotPositiveDefiniteError"),
        (m(&[&[1.0, 1.0], &[1.0, 1.0]]), "NotPositiveDefiniteError"),
        (m(&[&[-1.0, 0.0], &[0.0, -1.0]]), "NotPositiveDefiniteError"),
        (m(&[&[0.0, 0.0], &[0.0, 0.0]]), "NotPositiveDefiniteError"),
    ];
    for (a, class) in cases {
        let err = cholesky_factor(&a).unwrap_err();
        assert_eq!(err.python_name(), class, "{a:?}");
        assert!(err.is_linear_algebra_error());
    }
    for a in [
        m(&[&[1.0, 0.0]]),
        Vec::new(),
        m(&[&[f64::INFINITY, 0.0], &[0.0, 1.0]]),
    ] {
        assert!(
            cholesky_factor(&a).unwrap_err().is_value_error(),
            "shape and finiteness stay ValueErrors"
        );
    }
    let err = cholesky_factor(&m(&[&[4.0, 2.0], &[3.0, 5.0]])).unwrap_err();
    assert!(err.message().contains("symmetric"));
    let err =
        cholesky_factor(&m(&[&[2.0, 0.0, 0.0], &[0.0, 3.0, 0.0], &[0.0, 0.0, -1.0]])).unwrap_err();
    assert!(err.message().contains("pivot 2"), "{err}");
}

#[test]
fn cholesky_rejects_a_nan_wherever_it_sits() {
    let mirrored = m(&[
        &[4.0, f64::NAN, 0.0],
        &[2.0, 5.0, f64::NAN],
        &[0.0, 2.0, 5.0],
    ]);
    let err = cholesky_factor(&mirrored).unwrap_err();
    assert!(matches!(err, LocateError::MalformedMatrixError(_)));
    assert!(
        err.message().contains("non-finite") && err.message().contains("a[0][1]"),
        "{err}"
    );
    for i in 0..4 {
        for j in 0..4 {
            for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
                let mut a: Matrix = (0..4)
                    .map(|r| (0..4).map(|c| if r == c { 4.0 } else { 1.0 }).collect())
                    .collect();
                a[i][j] = bad;
                assert!(matches!(
                    cholesky_factor(&a),
                    Err(LocateError::MalformedMatrixError(_))
                ));
            }
        }
    }
}

#[test]
fn symmetry_tolerance_is_per_pair_with_an_absolute_floor() {
    assert!(matches!(
        cholesky_factor(&m(&[&[1e9, 1.0], &[2.0, 1e9]])),
        Err(LocateError::AsymmetricMatrixError(_))
    ));
    let (base, tol) = (1.0, 1e-9);
    for (factor, should_raise) in [(0.5, false), (0.9, false), (10.0, true)] {
        let off = base * (1.0 + factor * tol);
        let a = m(&[&[4.0, base, 0.0], &[off, 5.0, 0.0], &[0.0, 0.0, 6.0]]);
        assert_eq!(
            cholesky_factor_tol(&a, tol).is_err(),
            should_raise,
            "factor {factor}"
        );
    }
    let eps = 2.220446049250313e-16;
    cholesky_factor(&m(&[
        &[1e6, 0.0, 0.0],
        &[3.0 * eps * 1e6, 1e6, 0.0],
        &[0.0, 0.0, 1e6],
    ]))
    .unwrap();
    // A NaN tolerance refuses (the comparison is written negated).
    assert!(matches!(
        cholesky_factor_tol(&hand_a(), f64::NAN),
        Err(LocateError::AsymmetricMatrixError(_))
    ));
}

#[test]
fn solve_factored_checks_the_right_hand_side() {
    let g = cholesky_factor(&hand_a()).unwrap();
    assert!(matches!(
        cholesky_solve_factored(&g, &[1.0, 2.0]),
        Err(LocateError::ValueError(_))
    ));
    // A caller-supplied factor with a zero pivot divides by zero, as Python.
    let singular = m(&[&[0.0, 0.0], &[1.0, 1.0]]);
    assert!(matches!(
        cholesky_solve_factored(&singular, &[1.0, 1.0]),
        Err(LocateError::ZeroDivisionError(_))
    ));
}

// ----------------------------------------------------------------- prior

#[test]
fn prior_matches_the_direct_formula_on_an_anisotropic_grid() {
    for (room, pitch) in [(Room::new(4.0, 3.0).unwrap(), 0.5), (Room::default(), 0.7)] {
        let grid = PixelGrid::new(room, pitch).unwrap();
        let (sigma, delta_c) = (1.7, 0.65);
        let prior = SpatialPrior::new(&grid, sigma, delta_c).unwrap();
        let mut worst = 0.0f64;
        for (p, &(px, py)) in grid.pixels().iter().enumerate() {
            for (q, &(qx, qy)) in grid.pixels().iter().enumerate() {
                let want = sigma * sigma * (-py_hypot(px - qx, py - qy) / delta_c).exp();
                worst = worst.max((prior.entry(p, q).unwrap() - want).abs());
            }
        }
        assert!(worst < 1e-12, "worst entry error {worst:e}");
    }
    let grid = PixelGrid::new(Room::default(), 0.7).unwrap();
    assert_eq!((grid.n_x(), grid.n_y()), (11, 9));
    let prior = SpatialPrior::new(&grid, 1.0, 0.5).unwrap();
    let p = grid.pixel_index(3, 3).unwrap();
    assert!((prior.entry(p, p + 1).unwrap() - (-grid.pitch_x_m() / 0.5).exp()).abs() < 1e-14);
    assert!(
        (prior.entry(p, p + grid.n_x()).unwrap() - (-grid.pitch_y_m() / 0.5).exp()).abs() < 1e-14
    );
}

#[test]
fn prior_rows_entries_dense_symmetry_and_definiteness() {
    let grid = PixelGrid::new(Room::new(4.0, 3.0).unwrap(), 0.5).unwrap();
    let prior = SpatialPrior::new(&grid, 2.5, 0.4).unwrap();
    let dense = prior.dense();
    for p in 0..grid.len() {
        let row = prior.row(p).unwrap();
        assert_eq!(row, dense[p]);
        assert!((prior.entry(p, p).unwrap() - 6.25).abs() < 1e-14);
        for q in 0..grid.len() {
            assert_eq!(row[q], prior.entry(p, q).unwrap());
            assert_eq!(prior.entry(p, q).unwrap(), prior.entry(q, p).unwrap());
        }
    }
    let small = PixelGrid::new(Room::new(3.0, 2.0).unwrap(), 0.5).unwrap();
    for delta_c in [0.2, 0.5, 1.0, 3.0] {
        cholesky_factor(&SpatialPrior::new(&small, 1.0, delta_c).unwrap().dense()).unwrap();
    }
    let n = grid.len();
    assert!(matches!(prior.entry(n, 0), Err(LocateError::IndexError(_))));
    assert!(matches!(prior.entry(0, n), Err(LocateError::IndexError(_))));
    assert!(matches!(prior.row(n), Err(LocateError::IndexError(_))));
}

#[test]
fn prior_rejects_bad_parameters_by_name() {
    let grid = PixelGrid::new(Room::new(2.0, 2.0).unwrap(), 1.0).unwrap();
    for (sigma, delta_c) in [(0.0, 0.5), (-1.0, 0.5), (1.0, 0.0), (1.0, -0.5)] {
        assert!(matches!(
            SpatialPrior::new(&grid, sigma, delta_c),
            Err(LocateError::ValueError(_))
        ));
    }
    for (name, sigma, delta_c) in [
        ("sigma", f64::NAN, 0.5),
        ("sigma", f64::INFINITY, 0.5),
        ("sigma", 1e200, 0.5),
        ("delta_c", 1.0, f64::NAN),
        ("delta_c", 1.0, f64::INFINITY),
    ] {
        let err = SpatialPrior::new(&grid, sigma, delta_c).unwrap_err();
        assert!(err.message().contains(name), "{err}");
    }
}

// ----------------------------------------------------------------- forward / dual = primal

#[test]
fn forward_project_matches_the_dense_weight_matrix() {
    let g = Geometry::perimeter(6, None, 0.5, 0.5, 0.1, 0.0).unwrap();
    let image = blob(&g, 3.0, 2.0, 0.4);
    let y = forward_project(&g, &image).unwrap();
    for (yl, row) in y.iter().zip(g.weight_matrix()) {
        let want: f64 = row.iter().zip(&image).map(|(w, x)| w * x).sum();
        assert!((yl - want).abs() < 1e-12 * want.abs().max(1.0));
    }
    assert!(matches!(
        forward_project(&g, &image[1..]),
        Err(LocateError::ValueError(_))
    ));
}

#[test]
fn dual_equals_primal() {
    let room = Room::new(3.0, 2.0).unwrap();
    let g = Geometry::new(
        room,
        perimeter_nodes(&room, 4, 0.1, 0.0).unwrap(),
        Some(PixelGrid::new(room, 0.5).unwrap()),
        0.8,
    )
    .unwrap();
    assert_eq!((g.n_pixels(), g.n_links()), (24, 12));
    assert!(g.pixel_link_counts().iter().all(|&c| c > 0));
    let w = g.weight_matrix();
    let (n_p, n_l) = (g.n_pixels(), g.n_links());
    let mut rng = Rng::new(4242);
    for (alpha, sigma, delta_c) in [
        (1.0, 1.0, 0.5),
        (0.05, 2.0, 1.5),
        (10.0, 0.5, 0.25),
        (0.5, 1.0, 1.0),
    ] {
        let inv = inverter(&g, alpha, sigma, delta_c);
        let cov_inv = gauss_inverse(&SpatialPrior::new(g.grid(), sigma, delta_c).unwrap().dense());
        let primal: Matrix = (0..n_p)
            .map(|i| {
                (0..n_p)
                    .map(|j| {
                        (0..n_l).map(|l| w[l][i] * w[l][j]).sum::<f64>() + alpha * cov_inv[i][j]
                    })
                    .collect()
            })
            .collect();
        let y: Vec<f64> = (0..n_l).map(|_| rng.gauss()).collect();
        let wty: Vec<f64> = (0..n_p)
            .map(|i| (0..n_l).map(|l| w[l][i] * y[l]).sum())
            .collect();
        let x_primal = gauss_solve(&primal, &wty);
        let x_dual = inv.solve(&y).unwrap();
        let worst = max_rel_diff(&x_primal, &x_dual);
        assert!(
            worst < 1e-9,
            "dual and primal disagree by {worst:e} at {alpha} {sigma} {delta_c}"
        );
    }
}

// ----------------------------------------------------------------- blob recovery

#[test]
fn every_pixel_of_the_standard_deployment_is_recovered() {
    let g = Geometry::perimeter_n(8).unwrap();
    let inv = inverter(&g, 1.0, 1.0, 0.5);
    let n_x = g.grid().n_x();
    let mut exact = 0;
    for (p, &(cx, cy)) in g.grid().pixels().iter().enumerate() {
        let image = inv
            .solve(&forward_project(&g, &blob(&g, cx, cy, 0.4)).unwrap())
            .unwrap();
        let k = peak_index(&image);
        let (ix, iy, kx, ky) = (
            (p % n_x) as i64,
            (p / n_x) as i64,
            (k % n_x) as i64,
            (k / n_x) as i64,
        );
        assert!(
            (kx - ix).abs().max((ky - iy).abs()) <= 1,
            "blob at {p} came back at {k}"
        );
        if k == p {
            exact += 1;
        } else {
            let interior =
                0 < ix && ix < n_x as i64 - 1 && 0 < iy && iy < g.grid().n_y() as i64 - 1;
            assert!(!interior, "interior miss at {p}");
        }
    }
    assert_eq!(exact, 184, "CPython 3.14: 184 of 192 exact");
}

// ----------------------------------------------------------------- contract

#[test]
fn inverter_contract() {
    let g = Geometry::perimeter(6, None, 0.5, 0.5, 0.1, 0.0).unwrap();
    let inv = inverter(&g, 1.0, 1.0, 0.5);
    assert_eq!(
        inv.solve(&vec![0.0; g.n_links()]).unwrap(),
        vec![0.0; g.n_pixels()]
    );
    assert_eq!(inv.projection.len(), g.n_pixels());
    assert!(inv
        .projection
        .iter()
        .all(|r| r.len() == inv.n_independent_links));
    assert_eq!(
        (inv.n_links, inv.n_independent_links),
        (g.n_links(), g.n_links() / 2)
    );

    let mut rng = Rng::new(5);
    let y1: Vec<f64> = (0..g.n_links()).map(|_| rng.gauss()).collect();
    let y2: Vec<f64> = (0..g.n_links()).map(|_| rng.gauss()).collect();
    let (a, b) = (2.5, -1.75);
    let combined = inv
        .solve(
            &y1.iter()
                .zip(&y2)
                .map(|(u, v)| a * u + b * v)
                .collect::<Vec<_>>(),
        )
        .unwrap();
    let (x1, x2) = (inv.solve(&y1).unwrap(), inv.solve(&y2).unwrap());
    let separate: Vec<f64> = x1.iter().zip(&x2).map(|(u, v)| a * u + b * v).collect();
    assert!(max_rel_diff(&combined, &separate) < 1e-12);

    let y = forward_project(&g, &blob(&g, 3.0, 2.0, 0.4)).unwrap();
    assert_eq!(inv.solve(&y).unwrap(), inv.solve(&y).unwrap());
    assert_eq!(inv.forward(&blob(&g, 3.0, 2.0, 0.4)).unwrap(), y);

    for n in [g.n_links() - 1, g.n_links() + 1, 0] {
        assert!(matches!(
            inv.solve(&vec![0.0; n]),
            Err(LocateError::ValueError(_))
        ));
    }
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut y = vec![0.0; g.n_links()];
        y[3] = bad;
        let err = inv.solve(&y).unwrap_err();
        assert!(err.message().contains("y[3]"), "{err}");
    }
    let opts = InverterOptions::default();
    for alpha in [0.0, -1.0] {
        assert!(matches!(
            Inverter::new(g.clone(), alpha, 1.0, 0.5, opts),
            Err(LocateError::ValueError(_))
        ));
    }
    for (name, alpha, sigma, delta_c) in [
        ("alpha", f64::NAN, 1.0, 0.5),
        ("alpha", f64::INFINITY, 1.0, 0.5),
        ("sigma", 1.0, f64::NAN, 0.5),
        ("sigma", 1.0, 1e200, 0.5),
        ("delta_c", 1.0, 1.0, f64::INFINITY),
    ] {
        let err = Inverter::new(g.clone(), alpha, sigma, delta_c, opts).unwrap_err();
        assert!(
            matches!(err, LocateError::ValueError(_)) && err.message().contains(name),
            "{err}"
        );
    }
}

#[test]
fn sigma_is_redundant_with_alpha_and_scale_of_y_cancels() {
    let g = Geometry::perimeter_n(8).unwrap();
    let y = forward_project(&g, &blob(&g, 2.5, 4.0, 0.4)).unwrap();
    let reference = inverter(&g, 1.0, 1.0, 0.5).solve(&y).unwrap();
    for sigma in [0.25, 0.5, 2.0, 3.0, 10.0] {
        let image = inverter(&g, sigma * sigma, sigma, 0.5).solve(&y).unwrap();
        assert!(max_rel_diff(&reference, &image) < 1e-12, "sigma {sigma}");
    }
    let off = inverter(&g, 1.0, 3.0, 0.5).solve(&y).unwrap();
    assert!(max_rel_diff(&reference, &off) > 1e-3);

    let inv = inverter(&g, 1.0, 1.0, 0.5);
    let base = inv.solve(&y).unwrap();
    for k in [1e-6, 1e-3, 1e3, 1e6] {
        let image = inv
            .solve(&y.iter().map(|v| k * v).collect::<Vec<_>>())
            .unwrap();
        assert_eq!(peak_index(&image), peak_index(&base));
        let scaled: Vec<f64> = base.iter().map(|v| k * v).collect();
        assert!(max_rel_diff(&scaled, &image) < 1e-12);
    }
}

#[test]
fn blind_pixel_warning_is_a_field() {
    let g = Geometry::perimeter_n(8).unwrap();
    let inv = inverter(&g, 1.0, 1.0, 0.5);
    assert_eq!(inv.blind_pixels, vec![15, 176]);
    let warning = inv
        .blind_pixel_warning
        .as_deref()
        .expect("BlindPixelWarning");
    assert!(warning.starts_with("2 of 192 pixels"), "{warning}");
    let quiet = Inverter::new(
        g.clone(),
        1.0,
        1.0,
        0.5,
        InverterOptions {
            warn_blind_pixels: false,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(quiet.blind_pixels.len(), 2);
    assert!(quiet.blind_pixel_warning.is_none());
    let covered = Geometry::perimeter(8, None, 0.5, 0.75, 0.1, 0.0).unwrap();
    let inv = inverter(&covered, 1.0, 1.0, 0.5);
    assert!(inv.blind_pixels.is_empty() && inv.blind_pixel_warning.is_none());
}

#[test]
fn dual_system_diagnostics_and_ridge() {
    for (n, pitch) in [(8, 0.5), (8, 0.25), (12, 0.5)] {
        let g = Geometry::perimeter(n, None, pitch, 0.3, 0.1, 0.0).unwrap();
        let inv = inverter(&g, 1.0, 1.0, 0.5);
        assert!(inv.asymmetry < 1e-14);
        assert!(inv.inverse_residual.unwrap() < 1e-9);
        assert!(inv.mean_diag_a > 0.0 && inv.min_diag_a > 0.0);
    }
    let g = Geometry::perimeter_n(8).unwrap();
    let inv = inverter(&g, 1.0, 1.0, 0.5);
    assert_eq!(inv.mean_diag_a, 39.85025302193726, "CPython 3.14");
    assert_eq!(inv.min_diag_a, 14.624010016843268, "CPython 3.14");

    let g = Geometry::perimeter(4, None, 1.0, 1.5, 0.1, 0.0).unwrap();
    let (a1, a2) = (
        inverter(&g, 1.0, 1.0, 0.5).a_plus_alpha,
        inverter(&g, 5.0, 1.0, 0.5).a_plus_alpha,
    );
    for i in 0..a1.len() {
        for j in 0..a1.len() {
            assert!((a2[i][j] - (a1[i][j] + if i == j { 4.0 } else { 0.0 })).abs() < 1e-9);
        }
    }
}

// ----------------------------------------------------------------- deduplication

#[test]
fn reciprocal_links_collapse_into_pairs() {
    for (n, pitch) in [(6, 0.5), (8, 0.5), (8, 0.25), (12, 0.5)] {
        let g = Geometry::perimeter(n, None, pitch, 0.3, 0.1, 0.0).unwrap();
        let inv = inverter(&g, 1.0, 1.0, 0.5);
        assert_eq!(inv.n_independent_links, g.n_links() / 2);
        let mut seen: Vec<usize> = inv.link_groups.iter().flatten().copied().collect();
        seen.sort();
        assert_eq!(seen, (0..g.n_links()).collect::<Vec<_>>());
        for group in &inv.link_groups {
            assert_eq!(group.len(), 2);
            let (a, b) = (g.links()[group[0]], g.links()[group[1]]);
            assert_eq!((a.tx_id, a.rx_id), (b.rx_id, b.tx_id));
            assert!(group[0] < group[1]);
        }
    }
    let g = Geometry::perimeter_n(8).unwrap();
    let inv = inverter(&g, 1.0, 1.0, 0.5);
    let firsts: Vec<Vec<usize>> = inv.link_groups[..3].to_vec();
    assert_eq!(firsts, vec![vec![0, 7], vec![1, 14], vec![2, 21]]);
    assert_eq!(inv.link_groups.last().unwrap(), &vec![48, 55]);
}

#[test]
fn collapsed_system_is_positive_definite_and_survives_a_vanishing_ridge() {
    for (n, pitch) in [(6, 0.5), (8, 0.5), (12, 0.5)] {
        let g = Geometry::perimeter(n, None, pitch, 0.3, 0.1, 0.0).unwrap();
        let mut ridgeless = inverter(&g, 1.0, 1.0, 0.5).a_plus_alpha;
        for (i, row) in ridgeless.iter_mut().enumerate() {
            row[i] -= 1.0;
        }
        cholesky_factor(&ridgeless).unwrap();
    }
    let g = Geometry::perimeter_n(8).unwrap();
    for alpha in [1e-15, 1e-100, 1e-300] {
        Inverter::new(
            g.clone(),
            alpha,
            1.0,
            0.5,
            InverterOptions {
                check_inverse: false,
                ..Default::default()
            },
        )
        .unwrap();
    }
}

#[test]
fn only_the_sum_of_the_two_directions_reaches_the_image() {
    let g = Geometry::perimeter(6, None, 0.5, 0.5, 0.1, 0.0).unwrap();
    let inv = inverter(&g, 1.0, 1.0, 0.5);
    let mut rng = Rng::new(31337);
    let y: Vec<f64> = (0..g.n_links()).map(|_| rng.gauss()).collect();
    let mut swapped = y.clone();
    for (i, l) in g.links().iter().enumerate() {
        swapped[g.link_index(l.rx_id, l.tx_id).unwrap()] = y[i];
    }
    let averaged: Vec<f64> = y.iter().zip(&swapped).map(|(a, b)| 0.5 * (a + b)).collect();
    let base = inv.solve(&y).unwrap();
    assert!(max_rel_diff(&base, &inv.solve(&swapped).unwrap()) < 1e-12);
    assert!(max_rel_diff(&base, &inv.solve(&averaged).unwrap()) < 1e-12);
    let mut bumped = y.clone();
    bumped[0] += 1.0;
    let l0 = g.links()[0];
    bumped[g.link_index(l0.rx_id, l0.tx_id).unwrap()] += 1.0;
    assert!(max_rel_diff(&base, &inv.solve(&bumped).unwrap()) > 1e-3);
}

// ----------------------------------------------------------------- refusals

fn rank_starved() -> Geometry {
    let room = Room::default();
    Geometry::new(
        room,
        perimeter_nodes(&room, 12, 0.1, 0.0).unwrap(),
        Some(PixelGrid::new(room, 2.0).unwrap()),
        0.3,
    )
    .unwrap()
}

#[test]
fn the_back_check_refuses_and_degrades_smoothly() {
    let g = rank_starved();
    assert_eq!((g.n_links(), g.n_pixels()), (132, 12));
    let opts = InverterOptions::default();
    let err = Inverter::new(g.clone(), 1e-10, 1.0, 0.5, opts).unwrap_err();
    assert!(matches!(err, LocateError::IllConditionedError(_)));
    assert!(err.message().contains("B B^-1"));
    assert!(err.message().contains("alpha/sigma^2") && err.message().contains("mean diagonal"));

    let residuals: Vec<f64> = [1e-2, 1e-4, 1e-6, 1e-8]
        .iter()
        .map(|&a| {
            Inverter::new(g.clone(), a, 1.0, 0.5, opts)
                .unwrap()
                .inverse_residual
                .unwrap()
        })
        .collect();
    for w in residuals.windows(2) {
        assert!(w[1] > w[0], "{residuals:?}");
    }
    assert!(residuals[0] < 1e-12);
    let starved = Inverter::new(g.clone(), 1e-2, 1.0, 0.5, opts).unwrap();
    assert_eq!(
        starved.n_independent_links, 51,
        "132 links collapse to 51 rows"
    );

    let loose = Inverter::new(
        g.clone(),
        1e-10,
        1.0,
        0.5,
        InverterOptions {
            residual_tol: 1e-3,
            ..opts
        },
    )
    .unwrap();
    assert!(loose.inverse_residual.unwrap() > 1e-6);

    let err = Inverter::new(
        g,
        5e-324,
        1.0,
        0.5,
        InverterOptions {
            check_inverse: false,
            ..opts
        },
    )
    .unwrap_err();
    assert!(matches!(err, LocateError::IllConditionedError(_)) && err.is_linear_algebra_error());
    assert!(err.message().contains("alpha"));
}

#[test]
fn check_inverse_false_skips_without_lying() {
    let g = Geometry::perimeter_n(8).unwrap();
    let skipped = Inverter::new(
        g.clone(),
        1.0,
        1.0,
        0.5,
        InverterOptions {
            check_inverse: false,
            ..Default::default()
        },
    )
    .unwrap();
    let checked = inverter(&g, 1.0, 1.0, 0.5);
    assert!(skipped.inverse_residual.is_none());
    assert!(checked.inverse_residual.is_some());
    assert_eq!(skipped.projection, checked.projection);
    let entries = checked.n_pixels * checked.n_independent_links;
    assert_eq!(checked.projection_slots(), 8 * entries);
    assert_eq!(checked.projection_bytes(), 32 * entries);
}
