//! Turning an RTI image into a position, or into an honest refusal: a port
//! of `wifisense/localize/track.py`, line by line (spec/locate.md §3), with
//! `PositionEstimate`'s new `cov_xx`, `cov_xy`, `cov_yy` fields.
//!
//! [`extract_peak`] scores the image maximum against the image's own robust
//! floor (median background, MAD spread, with a pinned block of exactly
//! equal pixels taken as the background and excluded) and finds a
//! sub-pixel centroid in a window shrunk symmetrically at the walls.
//! [`Tracker`] is a constant-velocity Kalman filter (CWNA process noise,
//! Joseph-form update) that returns `None` whenever the peak fails
//! `z >= z_threshold`; coasted states are never emitted, and a coast
//! longer than `max_coast_s` discards the track. Every comparison that can
//! see a NaN fails closed. Estimates are SIM in Phase 4.

// The Python is the specification line by line: index loops mirror it, and
// `!(a > b)` is how a comparison is written so that a NaN fails closed.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

use std::sync::Arc;

use crate::geometry::Geometry;
use crate::pynum::{
    argmax_first, fmt_py, py_div, py_int, py_sort, pymax, pymin, LocateError, Result,
};

/// Significance threshold on the robust z-score (measured on
/// `localize/sim.py`, 8 x 6 m, 8 nodes, 192 pixels).
pub const DEFAULT_Z_THRESHOLD: f64 = 5.0;
/// Radius of the centroid neighbourhood, metres.
pub const DEFAULT_CENTROID_RADIUS_M: f64 = 1.0;
/// `_MAD_TO_SIGMA`: median absolute deviation to sigma for Gaussian data.
pub const MAD_TO_SIGMA: f64 = 1.4826;
/// `_TIE_MIN_FRACTION`: a run of exactly equal pixels at least this share
/// of the image is a pinned block.
pub const TIE_MIN_FRACTION: f64 = 0.125;
/// `_MIN_FREE_FRACTION`: below this share of free pixels the floor is not
/// measurable and `z = 0`.
pub const MIN_FREE_FRACTION: f64 = 0.25;

/// One accepted fix.
///
/// `sigma_m = sqrt(max(P00 + P11, 0))` is the filter's 1-sigma RMS radial
/// uncertainty; `cov_xx`, `cov_xy`, `cov_yy` are `P[0][0]`, `P[0][1]`,
/// `P[1][1]` of the same matrix (spec/locate.md §3; an estimate built
/// without them has zeros, which §8 never publishes). `confidence` scores
/// the image, not the filter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PositionEstimate {
    pub x: f64,
    pub y: f64,
    pub sigma_m: f64,
    pub confidence: f64,
    pub t: f64,
    pub cov_xx: f64,
    pub cov_xy: f64,
    pub cov_yy: f64,
}

/// Raw image statistics, before any significance decision.
///
/// `peak_index` is `None` (and `x`, `y`, `peak`, `background` NaN) when the
/// image held non-finite values; `tie_fraction` is the pinned block's
/// share of the pixels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PeakReading {
    pub x: f64,
    pub y: f64,
    pub peak: f64,
    pub background: f64,
    pub spread: f64,
    pub z: f64,
    pub peak_index: Option<usize>,
    pub n_nonfinite: usize,
    pub tie_fraction: f64,
}

/// `track._median` of an already-sorted, non-empty list:
/// `0.5 * a + 0.5 * b` for an even count (cannot overflow).
fn median_sorted(sorted_values: &[f64]) -> f64 {
    let n = sorted_values.len();
    let mid = n / 2;
    if n % 2 == 1 {
        return sorted_values[mid];
    }
    0.5 * sorted_values[mid - 1] + 0.5 * sorted_values[mid]
}

/// `_dominant_run`: value and length of the longest run of exactly equal
/// values in a sorted, non-empty list; the earlier run wins a tie.
fn dominant_run(sorted_values: &[f64]) -> (f64, usize) {
    let mut best_value = sorted_values[0];
    let mut best_count = 1;
    let mut run_value = sorted_values[0];
    let mut run_count = 1;
    for &value in &sorted_values[1..] {
        if value == run_value {
            run_count += 1;
        } else {
            run_value = value;
            run_count = 1;
        }
        if run_count > best_count {
            best_count = run_count;
            best_value = run_value;
        }
    }
    (best_value, best_count)
}

/// Locate the image maximum to sub-pixel accuracy and score it.
///
/// `ValueError` if the image is not `n_pixels` long or `radius_m` is not
/// `> 0`; `int(radius_m / pitch)` raises `OverflowError` for an infinite
/// radius, as in Python.
pub fn extract_peak(geometry: &Geometry, image: &[f64], radius_m: f64) -> Result<PeakReading> {
    let n_pixels = geometry.n_pixels();
    if image.len() != n_pixels {
        return Err(LocateError::ValueError(format!(
            "image has {} pixels, geometry has {n_pixels}; images must be in geometry.grid.pixels order",
            image.len()
        )));
    }
    if n_pixels == 0 {
        return Err(LocateError::ValueError(
            "geometry has no pixels".to_string(),
        ));
    }
    if !(radius_m > 0.0) {
        return Err(LocateError::ValueError(format!(
            "radius_m must be positive, got {}",
            fmt_py(radius_m)
        )));
    }

    let n_nonfinite = image.iter().filter(|v| !v.is_finite()).count();
    if n_nonfinite > 0 {
        return Ok(PeakReading {
            x: f64::NAN,
            y: f64::NAN,
            peak: f64::NAN,
            background: f64::NAN,
            spread: 0.0,
            z: 0.0,
            peak_index: None,
            n_nonfinite,
            tie_fraction: 0.0,
        });
    }

    let grid = geometry.grid();
    let peak_index = argmax_first(n_pixels, |i| image[i]).expect("non-empty image");
    let peak = image[peak_index];

    let mut ordered = image.to_vec();
    py_sort(&mut ordered);
    let (pinned_value, mut pinned_count) = dominant_run(&ordered);
    let background;
    let mut free: Vec<f64>;
    if pinned_count as f64 >= TIE_MIN_FRACTION * n_pixels as f64 {
        background = pinned_value;
        free = image
            .iter()
            .filter(|&&v| v != pinned_value)
            .map(|&v| (v - background).abs())
            .collect();
    } else {
        background = median_sorted(&ordered);
        free = image.iter().map(|&v| (v - background).abs()).collect();
        pinned_count = 0;
    }
    py_sort(&mut free);
    let tie_fraction = pinned_count as f64 / n_pixels as f64;

    let mut spread = if (free.len() as f64) < MIN_FREE_FRACTION * n_pixels as f64 {
        0.0
    } else {
        MAD_TO_SIGMA * median_sorted(&free)
    };

    let z = if spread > 0.0 && spread.is_finite() {
        let z = (peak - background) / spread;
        if !z.is_finite() || z < 0.0 {
            0.0
        } else {
            z
        }
    } else {
        if !spread.is_finite() {
            spread = 0.0;
        }
        0.0
    };

    let n_x = grid.n_x() as i64;
    let n_y = grid.n_y() as i64;
    let mut rx = py_int(py_div(radius_m, grid.pitch_x_m())?)?.max(1);
    let mut ry = py_int(py_div(radius_m, grid.pitch_y_m())?)?.max(1);

    let ix0 = peak_index as i64 % n_x;
    let iy0 = peak_index as i64 / n_x;

    rx = rx.min(ix0).min(n_x - 1 - ix0);
    ry = ry.min(iy0).min(n_y - 1 - iy0);

    let pixels = grid.pixels();
    let mut sum_w = 0.0;
    let mut sum_wx = 0.0;
    let mut sum_wy = 0.0;
    for dy in -ry..=ry {
        let row = (iy0 + dy) * n_x;
        for dx in -rx..=rx {
            let index = (row + ix0 + dx) as usize;
            let w = image[index] - background;
            if w <= 0.0 {
                continue;
            }
            let (px, py) = pixels[index];
            sum_w += w;
            sum_wx += w * px;
            sum_wy += w * py;
        }
    }

    let (cx, cy) = if sum_w > 0.0 {
        (sum_wx / sum_w, sum_wy / sum_w)
    } else {
        pixels[peak_index]
    };

    Ok(PeakReading {
        x: cx,
        y: cy,
        peak,
        background,
        spread,
        z,
        peak_index: Some(peak_index),
        n_nonfinite: 0,
        tie_fraction,
    })
}

type M4 = [[f64; 4]; 4];

fn identity4() -> M4 {
    let mut m = [[0.0; 4]; 4];
    for (i, row) in m.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    m
}

/// `_mat_mul`: plain accumulation, ascending, skipping `a[i][p] == 0.0`.
fn mat_mul(a: &M4, b: &M4) -> M4 {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for p in 0..4 {
            let aip = a[i][p];
            if aip == 0.0 {
                continue;
            }
            for j in 0..4 {
                out[i][j] += aip * b[p][j];
            }
        }
    }
    out
}

fn mat_t(a: &M4) -> M4 {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[j][i] = a[i][j];
        }
    }
    out
}

fn mat_add(a: &M4, b: &M4) -> M4 {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][j] + b[i][j];
        }
    }
    out
}

fn mat_sub(a: &M4, b: &M4) -> M4 {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = a[i][j] - b[i][j];
        }
    }
    out
}

/// `_symmetrize`: `0.5 * (a[i][j] + a[j][i])` for every entry.
fn symmetrize(a: &M4) -> M4 {
    let mut out = [[0.0; 4]; 4];
    for i in 0..4 {
        for j in 0..4 {
            out[i][j] = 0.5 * (a[i][j] + a[j][i]);
        }
    }
    out
}

/// `Tracker`'s keyword arguments other than `geometry`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrackerConfig {
    /// Default [`DEFAULT_Z_THRESHOLD`].
    pub z_threshold: f64,
    /// Default [`DEFAULT_CENTROID_RADIUS_M`].
    pub centroid_radius_m: f64,
    /// CWNA acceleration PSD, m²/s³ (default 0.1).
    pub accel_psd: f64,
    /// `None`: `0.5 * (pitch_x + pitch_y)`.
    pub meas_noise_m: Option<f64>,
    /// Default 1.0 m/s.
    pub init_speed_std_ms: f64,
    /// Default 3.0 s.
    pub max_coast_s: f64,
    /// Clamp the REPORTED position to the room (default true).
    pub clamp_to_room: bool,
}

impl Default for TrackerConfig {
    fn default() -> Self {
        TrackerConfig {
            z_threshold: DEFAULT_Z_THRESHOLD,
            centroid_radius_m: DEFAULT_CENTROID_RADIUS_M,
            accel_psd: 0.1,
            meas_noise_m: None,
            init_speed_std_ms: 1.0,
            max_coast_s: 3.0,
            clamp_to_room: true,
        }
    }
}

/// Constant-velocity Kalman filter over `(x, y, vx, vy)`, with abstention.
#[derive(Debug, Clone)]
pub struct Tracker {
    pub geometry: Arc<Geometry>,
    pub z_threshold: f64,
    pub centroid_radius_m: f64,
    pub accel_psd: f64,
    pub meas_noise_m: f64,
    pub init_speed_std_ms: f64,
    pub max_coast_s: f64,
    pub clamp_to_room: bool,
    state: Option<[f64; 4]>,
    cov: Option<M4>,
    t: Option<f64>,
    last_fix_t: Option<f64>,
}

impl Tracker {
    /// `Tracker(geometry, ...)`. `ValueError` (NaN-safe) unless
    /// `z_threshold`, `centroid_radius_m`, `accel_psd`, `init_speed_std_ms`
    /// and `meas_noise_m` are `> 0` and `max_coast_s >= 0`.
    pub fn new(geometry: impl Into<Arc<Geometry>>, config: TrackerConfig) -> Result<Tracker> {
        let geometry = geometry.into();
        let TrackerConfig {
            z_threshold,
            centroid_radius_m,
            accel_psd,
            meas_noise_m,
            init_speed_std_ms,
            max_coast_s,
            clamp_to_room,
        } = config;
        let refuse = |what: String| Err(LocateError::ValueError(what));
        if !(z_threshold > 0.0) {
            return refuse(format!(
                "z_threshold must be positive, got {}",
                fmt_py(z_threshold)
            ));
        }
        if !(centroid_radius_m > 0.0) {
            return refuse(format!(
                "centroid_radius_m must be positive, got {}",
                fmt_py(centroid_radius_m)
            ));
        }
        if !(accel_psd > 0.0) {
            return refuse(format!(
                "accel_psd must be positive, got {}",
                fmt_py(accel_psd)
            ));
        }
        if !(init_speed_std_ms > 0.0) {
            return refuse(format!(
                "init_speed_std_ms must be positive, got {}",
                fmt_py(init_speed_std_ms)
            ));
        }
        if !(max_coast_s >= 0.0) {
            return refuse(format!(
                "max_coast_s must be non-negative, got {}",
                fmt_py(max_coast_s)
            ));
        }
        let grid = geometry.grid();
        let meas_noise_m = meas_noise_m.unwrap_or(0.5 * (grid.pitch_x_m() + grid.pitch_y_m()));
        if !(meas_noise_m > 0.0) {
            return refuse(format!(
                "meas_noise_m must be positive, got {}",
                fmt_py(meas_noise_m)
            ));
        }
        Ok(Tracker {
            geometry,
            z_threshold,
            centroid_radius_m,
            accel_psd,
            meas_noise_m,
            init_speed_std_ms,
            max_coast_s,
            clamp_to_room,
            state: None,
            cov: None,
            t: None,
            last_fix_t: None,
        })
    }

    /// `Tracker(geometry)` with every default.
    pub fn with_defaults(geometry: impl Into<Arc<Geometry>>) -> Result<Tracker> {
        Tracker::new(geometry, TrackerConfig::default())
    }

    // -- introspection (diagnostics, never an output channel) ----------

    pub fn is_tracking(&self) -> bool {
        self.state.is_some()
    }

    /// The (possibly coasted) state position.
    pub fn position(&self) -> Option<(f64, f64)> {
        self.state.map(|s| (s[0], s[1]))
    }

    pub fn velocity(&self) -> Option<(f64, f64)> {
        self.state.map(|s| (s[2], s[3]))
    }

    /// `sqrt(max(P00 + P11, 0.0))`, including while coasting.
    pub fn sigma_m(&self) -> Option<f64> {
        self.cov.map(|p| pymax(p[0][0] + p[1][1], 0.0).sqrt())
    }

    /// The filter covariance `P` (4 x 4), including while coasting.
    pub fn covariance(&self) -> Option<[[f64; 4]; 4]> {
        self.cov
    }

    /// Forget the track; the next accepted measurement starts fresh.
    pub fn reset(&mut self) {
        self.state = None;
        self.cov = None;
    }

    // -- filter internals ----------------------------------------------

    /// `_process_noise`: per axis `q * [[dt^3/3, dt^2/2], [dt^2/2, dt]]`.
    fn process_noise(&self, dt: f64) -> M4 {
        let q = self.accel_psd;
        let q_pp = q * dt * dt * dt / 3.0;
        let q_pv = q * dt * dt / 2.0;
        let q_vv = q * dt;
        [
            [q_pp, 0.0, q_pv, 0.0],
            [0.0, q_pp, 0.0, q_pv],
            [q_pv, 0.0, q_vv, 0.0],
            [0.0, q_pv, 0.0, q_vv],
        ]
    }

    fn predict(&mut self, dt: f64) {
        let [x, y, vx, vy] = self.state.expect("tracking");
        let p = self.cov.expect("tracking");
        self.state = Some([x + vx * dt, y + vy * dt, vx, vy]);
        let f = [
            [1.0, 0.0, dt, 0.0],
            [0.0, 1.0, 0.0, dt],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ];
        let cov = mat_mul(&mat_mul(&f, &p), &mat_t(&f));
        self.cov = Some(symmetrize(&mat_add(&cov, &self.process_noise(dt))));
    }

    /// `_correct`: Kalman correction with `H = [I2 0]`, `R = r I`, Joseph
    /// form; silently skipped if `det(S)` is not finite or is zero.
    fn correct(&mut self, zx: f64, zy: f64, r: f64) {
        let state = self.state.expect("tracking");
        let p = self.cov.expect("tracking");

        let s00 = p[0][0] + r;
        let s01 = p[0][1];
        let s10 = p[1][0];
        let s11 = p[1][1] + r;
        let det = s00 * s11 - s01 * s10;
        if !det.is_finite() || det.abs() <= 0.0 {
            return;
        }
        let inv00 = s11 / det;
        let inv01 = -s01 / det;
        let inv10 = -s10 / det;
        let inv11 = s00 / det;

        let k: [[f64; 2]; 4] = std::array::from_fn(|i| {
            [
                p[i][0] * inv00 + p[i][1] * inv10,
                p[i][0] * inv01 + p[i][1] * inv11,
            ]
        });

        let innov_x = zx - state[0];
        let innov_y = zy - state[1];
        self.state = Some(std::array::from_fn(|i| {
            state[i] + k[i][0] * innov_x + k[i][1] * innov_y
        }));

        let kh: M4 = std::array::from_fn(|i| [k[i][0], k[i][1], 0.0, 0.0]);
        let i_kh = mat_sub(&identity4(), &kh);
        let term1 = mat_mul(&mat_mul(&i_kh, &p), &mat_t(&i_kh));
        let krk: M4 = std::array::from_fn(|i| {
            std::array::from_fn(|j| r * (k[i][0] * k[j][0] + k[i][1] * k[j][1]))
        });
        self.cov = Some(symmetrize(&mat_add(&term1, &krk)));
    }

    /// `_initialise`: state `[zx, zy, 0, 0]`, `P = diag(r, r, v^2, v^2)`.
    fn initialise(&mut self, zx: f64, zy: f64, r: f64) {
        self.state = Some([zx, zy, 0.0, 0.0]);
        let v2 = self.init_speed_std_ms * self.init_speed_std_ms;
        self.cov = Some([
            [r, 0.0, 0.0, 0.0],
            [0.0, r, 0.0, 0.0],
            [0.0, 0.0, v2, 0.0],
            [0.0, 0.0, 0.0, v2],
        ]);
    }

    // -- public API ----------------------------------------------------

    /// `1 - threshold / z` clamped to `[0, 1]`; 0 for a non-finite,
    /// non-positive or sub-threshold z.
    pub fn confidence_from_z(&self, z: f64) -> f64 {
        if !z.is_finite() || z <= 0.0 || z < self.z_threshold {
            return 0.0;
        }
        pymax(0.0, pymin(1.0, 1.0 - self.z_threshold / z))
    }

    /// Feed one image (canonical pixel order) at time `t`; a fix, or
    /// `None` to abstain. `ValueError` for a non-finite `t` (the state is
    /// untouched) or a wrong-length image.
    pub fn update(&mut self, image: &[f64], t: f64) -> Result<Option<PositionEstimate>> {
        if !t.is_finite() {
            return Err(LocateError::ValueError(format!(
                "t must be finite, got {}",
                fmt_py(t)
            )));
        }

        let reading = extract_peak(&self.geometry, image, self.centroid_radius_m)?;

        if self.t.is_none() {
            self.t = Some(t);
            self.last_fix_t = Some(t);
        }
        let self_t = self.t.expect("set above");
        let last_fix_t = self.last_fix_t.expect("set with t");

        let dt = t - self_t;
        if self.state.is_some() {
            if dt > 0.0 {
                self.predict(dt);
            }
            if !((t - last_fix_t).abs() <= pymax(self.max_coast_s, dt)) {
                self.reset();
            }
        }
        if t > self_t {
            self.t = Some(t);
        }

        if !(reading.z >= self.z_threshold) {
            return Ok(None);
        }
        if !(reading.x.is_finite() && reading.y.is_finite()) {
            return Ok(None);
        }

        let confidence = self.confidence_from_z(reading.z);
        let sigma = self.meas_noise_m * (2.0 - confidence);
        let r = sigma * sigma;

        if self.state.is_none() {
            self.initialise(reading.x, reading.y, r);
        } else {
            self.correct(reading.x, reading.y, r);
        }
        self.last_fix_t = self.t;

        let state = self.state.expect("tracking");
        let (mut x, mut y) = (state[0], state[1]);
        if self.clamp_to_room {
            let room = self.geometry.room();
            x = pymin(pymax(x, 0.0), room.width_m);
            y = pymin(pymax(y, 0.0), room.height_m);
        }

        let p = self.cov.expect("tracking");
        let sigma_m = pymax(p[0][0] + p[1][1], 0.0).sqrt();
        Ok(Some(PositionEstimate {
            x,
            y,
            sigma_m,
            confidence,
            t,
            cov_xx: p[0][0],
            cov_xy: p[0][1],
            cov_yy: p[1][1],
        }))
    }
}
