//! Regularized inversion, per-link disturbance vector -> spatial image: a
//! port of `wifisense/localize/invert.py`, line by line (spec/locate.md §3).
//!
//! ```text
//! C_pq  = sigma^2 * exp(-dist(p, q) / delta_c)
//! x_hat = C W^T (W C W^T + alpha I)^-1 y          [dual, implemented]
//! ```
//!
//! Duplicate rows of W (reciprocal links always; any two links with the
//! same support and weight) are collapsed into one row scaled by
//! `sqrt(m_g)`, and `y` is reduced as `r_g = (sum of the group) /
//! sqrt(m_g)`: the same estimator on a genuinely positive definite matrix.
//! The `P x L` projection `Pi = C V^T (V C V^T + alpha I)^-1` is built once;
//! [`Inverter::solve`] is one reduction plus one matrix-vector product.
//!
//! Every Python `sum()` (including `_dot = sum(map(mul, a, b))`) is
//! [`psum`]; `A`'s `list(map(add, ...))` accumulation and the Cholesky back
//! substitution are plain loops in the code's order. Python's exceptions
//! are [`LocateError`] variants (`MalformedMatrixError`,
//! `AsymmetricMatrixError`, `NotPositiveDefiniteError`,
//! `IllConditionedError`, plain `ValueError`); `BlindPixelWarning` is the
//! [`Inverter::blind_pixel_warning`] field.

// The Python is the specification line by line: index loops mirror it, and
// `!(a > b)` is how a comparison is written so that a NaN fails closed.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

use std::collections::HashMap;
use std::sync::Arc;

use crate::geometry::{Geometry, PixelGrid};
use crate::pynum::{fmt_e, fmt_py, psum, py_div, py_hypot, pymax, pymax_iter, pymin_iter, Result};

/// The Python exceptions (`LinearAlgebraError` and its four subclasses
/// among them) as one enum; defined in [`crate::pynum`].
pub use crate::pynum::LocateError;

/// A row-major matrix as a list of rows (Python's `list[list[float]]`).
pub type Matrix = Vec<Vec<f64>>;

/// `cholesky_factor`'s default `symmetry_tol`.
pub const DEFAULT_SYMMETRY_TOL: f64 = 1e-9;

/// `_require_positive_finite`: `not (value > 0) or not isfinite(value)`
/// raises `ValueError` naming the parameter.
fn require_positive_finite(name: &str, value: f64) -> Result<f64> {
    if !(value > 0.0) || !value.is_finite() {
        return Err(LocateError::ValueError(format!(
            "{name} must be positive and finite, got {}",
            fmt_py(value)
        )));
    }
    Ok(value)
}

/// `_dot(a, b) = sum(map(mul, a, b))`: Neumaier over the shorter length.
fn dot(a: &[f64], b: &[f64]) -> f64 {
    psum(a.iter().zip(b).map(|(x, y)| x * y))
}

/// Lower-triangular G with G G^T = A, `symmetry_tol = 1e-9`.
pub fn cholesky_factor(a: &[Vec<f64>]) -> Result<Matrix> {
    cholesky_factor_tol(a, DEFAULT_SYMMETRY_TOL)
}

/// `cholesky_factor(a, symmetry_tol)`.
///
/// Refuses, in this order: an empty or non-square matrix
/// (`MalformedMatrixError`); a non-finite entry, named `a[i][j]`, scanned
/// row by row (`MalformedMatrixError`); an all-zero matrix
/// (`NotPositiveDefiniteError`); a pair with `not (|a_ij - a_ji| <=
/// symmetry_tol * max(|a_ij|, |a_ji|) + 8 * eps * scale)`
/// (`AsymmetricMatrixError`); a pivot that is not finite or `<= 0`
/// (`NotPositiveDefiniteError` naming the pivot).
pub fn cholesky_factor_tol(a: &[Vec<f64>], symmetry_tol: f64) -> Result<Matrix> {
    let n = a.len();
    if n == 0 {
        return Err(LocateError::MalformedMatrixError(
            "cannot factor an empty matrix".to_string(),
        ));
    }
    for (i, row) in a.iter().enumerate() {
        if row.len() != n {
            return Err(LocateError::MalformedMatrixError(format!(
                "matrix is not square: row {i} has {} entries, need {n}",
                row.len()
            )));
        }
    }

    let mut scale = 0.0;
    for (i, row) in a.iter().enumerate() {
        if let Some(j) = row.iter().position(|v| !v.is_finite()) {
            return Err(LocateError::MalformedMatrixError(format!(
                "a[{i}][{j}] is {}; the matrix contains a non-finite entry. Cholesky would \
                 otherwise return a plausible-looking factor of a different matrix",
                fmt_py(row[j])
            )));
        }
        // The row is finite and non-empty here.
        let largest = pymax_iter(row.iter().map(|v| v.abs())).unwrap_or(0.0);
        if largest > scale {
            scale = largest;
        }
    }
    if scale <= 0.0 {
        return Err(LocateError::NotPositiveDefiniteError(
            "matrix is entirely zero".to_string(),
        ));
    }

    let floor = 8.0 * f64::EPSILON * scale;
    for i in 0..n {
        let ai = &a[i];
        for j in 0..i {
            let upper = a[j][i];
            let d = (ai[j] - upper).abs();
            if !(d <= symmetry_tol * pymax(ai[j].abs(), upper.abs()) + floor) {
                return Err(LocateError::AsymmetricMatrixError(format!(
                    "matrix is not symmetric: a[{i}][{j}] = {} but a[{j}][{i}] = {}, differing by {} \
                     against a tolerance of {} relative to the pair plus {}; this usually means a \
                     transposed index upstream",
                    fmt_e(ai[j], 6),
                    fmt_e(upper, 6),
                    fmt_e(d, 3),
                    fmt_e(symmetry_tol, 1),
                    fmt_e(floor, 3)
                )));
            }
        }
    }

    let mut g: Matrix = vec![vec![0.0; n]; n];
    for i in 0..n {
        let ai = &a[i];
        let mut gi = vec![0.0; n];
        for j in 0..i {
            let gj = &g[j];
            gi[j] = (ai[j] - dot(&gi[..j], &gj[..j])) / gj[j];
        }
        let diag = ai[i] - dot(&gi[..i], &gi[..i]);
        if !diag.is_finite() || diag <= 0.0 {
            return Err(LocateError::NotPositiveDefiniteError(format!(
                "pivot {i} of {n} is {}, which is not positive; the matrix is not positive \
                 definite (original diagonal entry was {})",
                fmt_e(diag, 6),
                fmt_e(ai[i], 6)
            )));
        }
        gi[i] = diag.sqrt();
        g[i] = gi;
    }
    Ok(g)
}

/// Solve G G^T x = b given the factor G (`ValueError` on a length
/// mismatch). Forward substitution uses `_dot` (Neumaier); the back
/// substitution `s -= g[k][i] * x[k]` is a plain ascending loop. A row
/// shorter than its own index raises `IndexError` and a zero diagonal
/// `ZeroDivisionError`, as Python would.
pub fn cholesky_solve_factored(g: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    let n = g.len();
    if b.len() != n {
        return Err(LocateError::ValueError(format!(
            "right-hand side has length {}, need {n}",
            b.len()
        )));
    }

    let mut z = vec![0.0; n];
    for i in 0..n {
        let gi = &g[i];
        if gi.len() <= i {
            return Err(LocateError::IndexError(
                "list index out of range".to_string(),
            ));
        }
        z[i] = py_div(b[i] - dot(&gi[..i], &z[..i]), gi[i])?;
    }

    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut s = z[i];
        for k in (i + 1)..n {
            s -= g[k][i] * x[k];
        }
        x[i] = py_div(s, g[i][i])?;
    }
    Ok(x)
}

/// Solve A x = b for symmetric positive definite A.
pub fn cholesky_solve(a: &[Vec<f64>], b: &[f64]) -> Result<Vec<f64>> {
    cholesky_solve_factored(&cholesky_factor(a)?, b)
}

/// A^-1 for symmetric positive definite A, symmetrized:
/// `inv[i][j] = 0.5 * (cols[j][i] + cols[i][j])`.
pub fn cholesky_inverse(a: &[Vec<f64>]) -> Result<Matrix> {
    let g = cholesky_factor(a)?;
    let n = g.len();
    let mut cols: Matrix = Vec::with_capacity(n);
    for j in 0..n {
        let mut e = vec![0.0; n];
        e[j] = 1.0;
        cols.push(cholesky_solve_factored(&g, &e)?);
    }
    Ok((0..n)
        .map(|i| (0..n).map(|j| 0.5 * (cols[j][i] + cols[i][j])).collect())
        .collect())
}

/// `C_pq = sigma^2 * exp(-hypot(dix * pitch_x, diy * pitch_y) / delta_c)`
/// over a [`PixelGrid`], stored as two-level Toeplitz strips (never P x P).
/// `dix` pairs with `pitch_x` and `diy` with `pitch_y`.
#[derive(Debug, Clone)]
pub struct SpatialPrior {
    pub grid: PixelGrid,
    pub sigma: f64,
    pub delta_c: f64,
    pub sigma2: f64,
    n_x: usize,
    n_y: usize,
    strips: Vec<Vec<Vec<f64>>>,
}

impl SpatialPrior {
    /// `SpatialPrior(grid, sigma, delta_c)`: `ValueError` unless both are
    /// positive and finite and `sigma * sigma` is finite.
    pub fn new(grid: &PixelGrid, sigma: f64, delta_c: f64) -> Result<SpatialPrior> {
        require_positive_finite("sigma", sigma)?;
        require_positive_finite("delta_c", delta_c)?;
        let sigma2 = sigma * sigma;
        if !sigma2.is_finite() {
            return Err(LocateError::ValueError(format!(
                "sigma = {} overflows to infinity when squared; sigma is redundant with alpha \
                 anyway (see Inverter), so fix sigma = 1 and scale alpha instead",
                fmt_py(sigma)
            )));
        }
        let n_x = grid.n_x();
        let n_y = grid.n_y();
        let px = grid.pitch_x_m();
        let py = grid.pitch_y_m();
        let kern: Vec<Vec<f64>> = (0..n_x)
            .map(|dix| {
                (0..n_y)
                    .map(|diy| {
                        sigma2 * ((-py_hypot(dix as f64 * px, diy as f64 * py)) / delta_c).exp()
                    })
                    .collect()
            })
            .collect();
        let strips = (0..n_y)
            .map(|diy| {
                (0..n_x)
                    .map(|ixp| (0..n_x).map(|ix| kern[ixp.abs_diff(ix)][diy]).collect())
                    .collect()
            })
            .collect();
        Ok(SpatialPrior {
            grid: grid.clone(),
            sigma,
            delta_c,
            sigma2,
            n_x,
            n_y,
            strips,
        })
    }

    /// C_pq for two canonical pixel indices; `IndexError` outside the grid.
    pub fn entry(&self, p: usize, q: usize) -> Result<f64> {
        let n_x = self.n_x;
        let n = n_x * self.n_y;
        if !(p < n && q < n) {
            return Err(LocateError::IndexError(format!(
                "pixel pair ({p}, {q}) outside grid of {n}"
            )));
        }
        Ok(self.strips[(p / n_x).abs_diff(q / n_x)][p % n_x][q % n_x])
    }

    /// Row p of C in canonical pixel order; `IndexError` outside the grid.
    pub fn row(&self, p: usize) -> Result<Vec<f64>> {
        let n_x = self.n_x;
        if p >= n_x * self.n_y {
            return Err(LocateError::IndexError(format!(
                "pixel {p} outside grid of {}",
                n_x * self.n_y
            )));
        }
        let ixp = p % n_x;
        let iyp = p / n_x;
        let mut out = Vec::with_capacity(n_x * self.n_y);
        for iy in 0..self.n_y {
            out.extend_from_slice(&self.strips[iyp.abs_diff(iy)][ixp]);
        }
        Ok(out)
    }

    /// The full P x P matrix. For tests and small primal solves only.
    pub fn dense(&self) -> Matrix {
        (0..self.n_x * self.n_y)
            .map(|p| self.row(p).expect("in range"))
            .collect()
    }
}

/// `y = W x` in canonical link order, over all `geometry.n_links()`:
/// `y_l = (1/sqrt(d_l) or 0) * sum(image[p] for p in supports[l])`.
/// `ValueError` if the image is not `n_pixels` long.
pub fn forward_project(geometry: &Geometry, image: &[f64]) -> Result<Vec<f64>> {
    if image.len() != geometry.n_pixels() {
        return Err(LocateError::ValueError(format!(
            "image has {} pixels, geometry has {}",
            image.len(),
            geometry.n_pixels()
        )));
    }
    Ok(geometry
        .links()
        .iter()
        .zip(geometry.supports())
        .map(|(link, support)| {
            let w = if link.length_m > 0.0 {
                1.0 / link.length_m.sqrt()
            } else {
                0.0
            };
            w * psum(support.iter().map(|&p| image[p]))
        })
        .collect())
}

/// `Inverter`'s keyword-only arguments.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverterOptions {
    /// Refuse when `max |B B^-1 - I| > residual_tol` (default 1e-6).
    pub residual_tol: f64,
    /// Run the back-check (default true).
    pub check_inverse: bool,
    /// Record [`Inverter::blind_pixel_warning`] (default true).
    pub warn_blind_pixels: bool,
}

impl Default for InverterOptions {
    fn default() -> Self {
        InverterOptions {
            residual_tol: 1e-6,
            check_inverse: true,
            warn_blind_pixels: true,
        }
    }
}

/// Dual-form regularized least squares, precomputed per geometry.
///
/// Treat as immutable; the fields mirror the Python attributes
/// (`_a_plus_alpha` is `a_plus_alpha`, `_group_scale` is `group_scale`).
#[derive(Debug, Clone)]
pub struct Inverter {
    pub geometry: Arc<Geometry>,
    pub alpha: f64,
    pub prior: SpatialPrior,
    pub sigma: f64,
    pub delta_c: f64,
    pub n_links: usize,
    pub n_pixels: usize,
    /// Pixels inside no link's ellipse.
    pub blind_pixels: Vec<usize>,
    /// The `BlindPixelWarning` Python would emit: `Some(message)` iff there
    /// are blind pixels and `warn_blind_pixels` is set.
    pub blind_pixel_warning: Option<String>,
    /// Link indices merged into each row, rows in first-occurrence order,
    /// members ascending.
    pub link_groups: Vec<Vec<usize>>,
    pub n_independent_links: usize,
    /// Relative asymmetry of V C V^T before symmetrization.
    pub asymmetry: f64,
    pub mean_diag_a: f64,
    pub min_diag_a: f64,
    /// `max |B B^-1 - I|`, or `None` when `check_inverse` was false.
    pub inverse_residual: Option<f64>,
    /// Pi, `n_pixels` rows of `n_independent_links`.
    pub projection: Matrix,
    /// V C V^T symmetrized, plus alpha on the diagonal.
    pub a_plus_alpha: Matrix,
    /// `1 / sqrt(len(group))` per row, applied to y in `solve`.
    pub group_scale: Vec<f64>,
}

impl Inverter {
    /// `Inverter(geometry, alpha, sigma, delta_c, residual_tol=...,
    /// check_inverse=..., warn_blind_pixels=...)`.
    ///
    /// `ValueError` for a non-positive or non-finite `alpha`, `sigma` or
    /// `delta_c`; `IllConditionedError` for an asymmetric V C V^T, an
    /// indefinite factorisation, or a back-check residual above
    /// `residual_tol`.
    pub fn new(
        geometry: impl Into<Arc<Geometry>>,
        alpha: f64,
        sigma: f64,
        delta_c: f64,
        options: InverterOptions,
    ) -> Result<Inverter> {
        let geometry = geometry.into();
        require_positive_finite("alpha", alpha)?;
        let prior = SpatialPrior::new(geometry.grid(), sigma, delta_c)?;
        let row_of = |p: usize| prior.row(p).expect("pixel in range");
        Inverter::build(geometry.clone(), alpha, &prior, row_of, options)
            .map(|parts| parts.into_inverter(geometry, alpha, prior))
    }

    /// `Inverter(geometry)` with alpha 1, sigma 1, delta_c 0.5 and the
    /// default options.
    pub fn with_defaults(geometry: impl Into<Arc<Geometry>>) -> Result<Inverter> {
        Inverter::new(geometry, 1.0, 1.0, 0.5, InverterOptions::default())
    }

    /// The construction, with the prior's rows supplied by `row_of` so the
    /// asymmetry guard can be exercised by a test.
    fn build(
        geometry: Arc<Geometry>,
        alpha: f64,
        prior: &SpatialPrior,
        row_of: impl Fn(usize) -> Vec<f64>,
        options: InverterOptions,
    ) -> Result<Parts> {
        let n_pixels = geometry.n_pixels();
        let counts = geometry.pixel_link_counts();
        let blind_pixels: Vec<usize> = counts
            .iter()
            .enumerate()
            .filter(|(_, &c)| c == 0)
            .map(|(p, _)| p)
            .collect();
        let blind_pixel_warning = if !blind_pixels.is_empty() && options.warn_blind_pixels {
            Some(format!(
                "{} of {} pixels are inside no link's ellipse and cannot be reconstructed; their \
                 value comes entirely from the spatial prior. Raise lambda_m or add nodes. First \
                 few: {:?}",
                blind_pixels.len(),
                n_pixels,
                &blind_pixels[..blind_pixels.len().min(5)]
            ))
        } else {
            None
        };

        let supports = geometry.supports();
        let link_w: Vec<f64> = geometry
            .links()
            .iter()
            .map(|l| {
                if l.length_m > 0.0 {
                    1.0 / l.length_m.sqrt()
                } else {
                    0.0
                }
            })
            .collect();

        // --- collapse duplicate rows of W ---
        // key = (tuple(support), w) if support else ((), 0.0); w is never a
        // NaN or -0.0, so its bits are exact float equality.
        let mut seen: HashMap<(Vec<usize>, u64), usize> = HashMap::new();
        let mut link_groups: Vec<Vec<usize>> = Vec::new();
        for (i, (&w, support)) in link_w.iter().zip(supports).enumerate() {
            let key = if support.is_empty() {
                (Vec::new(), 0.0f64.to_bits())
            } else {
                (support.clone(), w.to_bits())
            };
            let at = *seen.entry(key).or_insert_with(|| {
                link_groups.push(Vec::new());
                link_groups.len() - 1
            });
            link_groups[at].push(i);
        }

        let n_rows = link_groups.len();
        let reps: Vec<usize> = link_groups.iter().map(|g| g[0]).collect();
        let row_support: Vec<&Vec<usize>> = reps.iter().map(|&r| &supports[r]).collect();
        let row_w: Vec<f64> = reps
            .iter()
            .zip(&link_groups)
            .map(|(&r, group)| link_w[r] * (group.len() as f64).sqrt())
            .collect();
        let group_scale: Vec<f64> = link_groups
            .iter()
            .map(|g| 1.0 / (g.len() as f64).sqrt())
            .collect();

        // --- M = C V^T, P x L ---
        let mut m: Matrix = Vec::with_capacity(n_pixels);
        for p in 0..n_pixels {
            let row = row_of(p);
            m.push(
                row_w
                    .iter()
                    .zip(&row_support)
                    .map(|(&w, s)| w * psum(s.iter().map(|&q| row[q])))
                    .collect(),
            );
        }

        // --- A = V M, L x L ---
        let mut a: Matrix = Vec::with_capacity(n_rows);
        for (&w, support) in row_w.iter().zip(&row_support) {
            let mut acc = vec![0.0; n_rows];
            for &p in support.iter() {
                for (acc_k, m_pk) in acc.iter_mut().zip(&m[p]) {
                    *acc_k += m_pk;
                }
            }
            a.push(acc.iter().map(|v| w * v).collect());
        }

        // --- symmetry check, then ridge ---
        let scale = pymax_iter(a.iter().flat_map(|row| row.iter().map(|v| v.abs()))).unwrap_or(0.0);
        let mut worst = 0.0;
        for i in 0..n_rows {
            for j in 0..i {
                worst = pymax(worst, (a[i][j] - a[j][i]).abs());
            }
        }
        let asymmetry = if scale > 0.0 { worst / scale } else { 0.0 };
        if asymmetry > 1e-7 {
            return Err(LocateError::IllConditionedError(format!(
                "V C V^T is asymmetric by a relative {}, far beyond rounding. The link/pixel index \
                 order is wrong.",
                fmt_e(asymmetry, 3)
            )));
        }

        let diag: Vec<f64> = (0..n_rows).map(|i| a[i][i]).collect();
        let mean_diag_a = if diag.is_empty() {
            0.0
        } else {
            psum(diag.iter().copied()) / diag.len() as f64
        };
        let min_diag_a = pymin_iter(diag.iter().copied()).unwrap_or(0.0);

        for i in 0..n_rows {
            for j in 0..i {
                let mean = 0.5 * (a[i][j] + a[j][i]);
                a[i][j] = mean;
                a[j][i] = mean;
            }
            a[i][i] += alpha;
        }

        // --- Pi = M (A + alpha I)^-1 ---
        let b_inv = match cholesky_inverse(&a) {
            Ok(b) => b,
            Err(LocateError::NotPositiveDefiniteError(exc)) => {
                let ratio = if mean_diag_a > 0.0 {
                    alpha / mean_diag_a
                } else {
                    f64::INFINITY
                };
                return Err(LocateError::IllConditionedError(format!(
                    "(V C V^T + alpha I) came out numerically indefinite: {exc}. alpha = {} is {} of \
                     the mean diagonal {}, so the ridge is at or below the rounding level of the \
                     matrix it is regularising. Raise alpha, or lower sigma (only alpha/sigma^2 \
                     matters).",
                    fmt_py(alpha),
                    fmt_e(ratio, 2),
                    fmt_py(mean_diag_a)
                )));
            }
            Err(other) => return Err(other),
        };

        let inverse_residual = if options.check_inverse {
            Some(max_inverse_residual(&a, &b_inv))
        } else {
            None
        };
        if let Some(residual) = inverse_residual {
            if residual > options.residual_tol {
                let effective = py_div(alpha, prior.sigma2)?;
                return Err(LocateError::IllConditionedError(format!(
                    "(V C V^T + alpha I) is too ill-conditioned to invert reliably: max |B B^-1 - I| \
                     = {} exceeds {}. The knob is the EFFECTIVE ridge alpha/sigma^2 = {} against a \
                     mean diagonal of {} (sigma and alpha act only through that ratio, so raising \
                     alpha at a large sigma buys nothing - lower sigma or raise alpha until the \
                     ratio moves). delta_c = {} m is a second, weaker axis: a correlation length \
                     above the room scale makes C nearly rank one, though the residual is not \
                     monotone in it past ~8 m.",
                    fmt_e(residual, 3),
                    fmt_e(options.residual_tol, 1),
                    fmt_e(effective, 3),
                    fmt_py(mean_diag_a),
                    fmt_py(prior.delta_c)
                )));
            }
        }

        // B^-1 is symmetric, so its column k is its row k.
        let projection: Matrix = m
            .iter()
            .map(|m_p| (0..n_rows).map(|k| dot(m_p, &b_inv[k])).collect())
            .collect();

        Ok(Parts {
            blind_pixels,
            blind_pixel_warning,
            link_groups,
            asymmetry,
            mean_diag_a,
            min_diag_a,
            inverse_residual,
            projection,
            a_plus_alpha: a,
            group_scale,
        })
    }

    /// `x_hat = Pi r(y)`, the image in canonical pixel order. `y` is in
    /// canonical link order, all `n_links`; `ValueError` on a wrong length
    /// or a non-finite entry (named by index).
    pub fn solve(&self, y: &[f64]) -> Result<Vec<f64>> {
        if y.len() != self.n_links {
            return Err(LocateError::ValueError(format!(
                "y has {} entries, geometry has {} links",
                y.len(),
                self.n_links
            )));
        }
        for (i, v) in y.iter().enumerate() {
            if !v.is_finite() {
                return Err(LocateError::ValueError(format!(
                    "y[{i}] is {}; link vector must be finite",
                    fmt_py(*v)
                )));
            }
        }
        let r: Vec<f64> = self
            .group_scale
            .iter()
            .zip(&self.link_groups)
            .map(|(&s, group)| s * psum(group.iter().map(|&i| y[i])))
            .collect();
        Ok(self.projection.iter().map(|row| dot(row, &r)).collect())
    }

    /// `y = W x`, the forward model of this inverter's geometry.
    pub fn forward(&self, image: &[f64]) -> Result<Vec<f64>> {
        forward_project(&self.geometry, image)
    }

    /// `8 * n_pixels * n_independent_links` (pointer slots in CPython).
    pub fn projection_slots(&self) -> usize {
        8 * self.n_pixels * self.n_independent_links
    }

    /// `32 * n_pixels * n_independent_links` (CPython resident size).
    pub fn projection_bytes(&self) -> usize {
        32 * self.n_pixels * self.n_independent_links
    }
}

/// What `build` computes beyond the arguments.
struct Parts {
    blind_pixels: Vec<usize>,
    blind_pixel_warning: Option<String>,
    link_groups: Vec<Vec<usize>>,
    asymmetry: f64,
    mean_diag_a: f64,
    min_diag_a: f64,
    inverse_residual: Option<f64>,
    projection: Matrix,
    a_plus_alpha: Matrix,
    group_scale: Vec<f64>,
}

impl Parts {
    fn into_inverter(self, geometry: Arc<Geometry>, alpha: f64, prior: SpatialPrior) -> Inverter {
        let n_links = geometry.n_links();
        let n_pixels = geometry.n_pixels();
        let n_independent_links = self.link_groups.len();
        Inverter {
            geometry,
            alpha,
            sigma: prior.sigma,
            delta_c: prior.delta_c,
            prior,
            n_links,
            n_pixels,
            blind_pixels: self.blind_pixels,
            blind_pixel_warning: self.blind_pixel_warning,
            link_groups: self.link_groups,
            n_independent_links,
            asymmetry: self.asymmetry,
            mean_diag_a: self.mean_diag_a,
            min_diag_a: self.min_diag_a,
            inverse_residual: self.inverse_residual,
            projection: self.projection,
            a_plus_alpha: self.a_plus_alpha,
            group_scale: self.group_scale,
        }
    }
}

/// `max |A A^-1 - I|` with `max(worst, ...)` two-argument semantics.
fn max_inverse_residual(a: &[Vec<f64>], a_inv: &[Vec<f64>]) -> f64 {
    let n = a.len();
    let mut worst = 0.0;
    for i in 0..n {
        let ai = &a[i];
        for j in 0..n {
            let v = dot(ai, &a_inv[j]);
            worst = pymax(worst, (v - if i == j { 1.0 } else { 0.0 }).abs());
        }
    }
    worst
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_asymmetry_guard_fires_when_the_prior_is_not_symmetric() {
        // test_the_asymmetry_guard_fires_when_the_prior_is_not_symmetric:
        // a prior whose row reads `v * (1 + 0.3 * (q < p))`.
        let geometry = Arc::new(Geometry::perimeter(6, None, 0.5, 0.5, 0.1, 0.0).unwrap());
        let prior = SpatialPrior::new(geometry.grid(), 1.0, 0.5).unwrap();
        let skewed = |p: usize| {
            prior
                .row(p)
                .unwrap()
                .iter()
                .enumerate()
                .map(|(q, v)| v * (1.0 + 0.3 * if q < p { 1.0 } else { 0.0 }))
                .collect::<Vec<f64>>()
        };
        let err = Inverter::build(geometry, 1.0, &prior, skewed, InverterOptions::default())
            .err()
            .unwrap();
        assert!(matches!(err, LocateError::IllConditionedError(_)));
        assert!(err.message().contains("asymmetric"));
    }
}
