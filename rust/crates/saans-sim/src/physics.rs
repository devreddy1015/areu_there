//! The geometric radio scene: a port of the Python reference
//! `SyntheticConfig(scene="geometric")` (`wifisense/sources/synthetic.py`),
//! generalised from one fixed link and one fixed body to many links and a
//! moving body.
//!
//! Every path contributes `gain * exp(-j 2 pi f_k L / c)`:
//!
//! - line of sight: `10^(-los_db/20) / d_los`;
//! - wall reflections by the image method up to `reflection_order`, each
//!   bounce multiplying by `-Gamma` (never mirroring straight back across
//!   the wall just used; in a rectangular room every image is valid);
//! - point scatterers (furniture, torso points, the arm):
//!   `sqrt(rcs / 4 pi) / (d1 d2)`.
//!
//! Each link is scaled so its *unattenuated* LoS has |h| = 40. Chest motion
//! along the tx-body-rx bisector changes a torso point's path by
//! `2 cos(beta/2)` per metre (`path_change`).
//!
//! Where the reference relies on CPython semantics that change the last
//! bits (Neumaier-compensated `sum()` of floats and, since 3.14, of
//! complex numbers), the construction-time code here copies them so the
//! static channel matches the golden vectors as closely as possible.

use crate::scenario::{Point, Room};
use std::f64::consts::PI;

pub const SPEED_OF_LIGHT: f64 = 299_792_458.0;
/// |h| of the unattenuated line of sight, every link.
pub const STATIC_MAGNITUDE: f64 = 40.0;
/// Bistatic RCS of a seated torso at 2.4 GHz, split over the torso points.
pub const BODY_RCS_M2: f64 = 0.3;
/// Ranges below this are clamped in the point-scatterer amplitude: the
/// far-field formula diverges at an antenna, a real body does not.
pub const MIN_RANGE_M: f64 = 0.1;

pub fn dist(a: Point, b: Point) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1)
}

/// Unit vector bisecting the tx-p-rx angle (the chest normal).
pub fn bisector(tx: Point, rx: Point, p: Point) -> Point {
    let d1 = dist(tx, p).max(1e-12);
    let d2 = dist(p, rx).max(1e-12);
    let bx = (tx.0 - p.0) / d1 + (rx.0 - p.0) / d2;
    let by = (tx.1 - p.1) / d1 + (rx.1 - p.1) / d2;
    let norm = bx.hypot(by);
    if norm < 1e-12 {
        // p on the LoS segment: bistatic angle is pi.
        let d = dist(tx, rx);
        let (ux, uy) = ((rx.0 - tx.0) / d, (rx.1 - tx.1) / d);
        return (-uy, ux);
    }
    (bx / norm, by / norm)
}

/// Path-length change per metre of chest motion along the bisector:
/// `2 cos(beta/2)` for bistatic angle beta.
pub fn path_change(tx: Point, rx: Point, p: Point) -> f64 {
    let d1 = dist(tx, p).max(1e-12);
    let d2 = dist(p, rx).max(1e-12);
    let u1 = ((tx.0 - p.0) / d1, (tx.1 - p.1) / d1);
    let u2 = ((rx.0 - p.0) / d2, (rx.1 - p.1) / d2);
    let cos_beta = (u1.0 * u2.0 + u1.1 * u2.1).clamp(-1.0, 1.0);
    2.0 * (cos_beta.acos() / 2.0).cos()
}

/// (amplitude, path length) of a bistatic point scatterer, unscaled.
pub fn point_path(tx: Point, rx: Point, p: Point, rcs_m2: f64) -> (f64, f64) {
    let d1 = dist(tx, p);
    let d2 = dist(p, rx);
    let amp = (rcs_m2 / (4.0 * PI)).sqrt() / (d1.max(MIN_RANGE_M) * d2.max(MIN_RANGE_M));
    (amp, d1 + d2)
}

/// CPython's compensated summation (`cs_add` / `cs_to_double`).
#[derive(Debug, Clone, Copy, Default)]
struct Neumaier {
    hi: f64,
    lo: f64,
}

impl Neumaier {
    fn add(&mut self, x: f64) {
        let t = self.hi + x;
        if self.hi.abs() >= x.abs() {
            self.lo += (self.hi - t) + x;
        } else {
            self.lo += (x - t) + self.hi;
        }
        self.hi = t;
    }
    fn value(self) -> f64 {
        if self.lo != 0.0 && self.lo.is_finite() {
            self.hi + self.lo
        } else {
            self.hi
        }
    }
}

/// One torso scatter point as seen by one link.
#[derive(Debug, Clone, Copy)]
pub struct BodyPoint {
    pub pos: Point,
    /// Unscaled amplitude `sqrt(rcs / n / 4 pi) / (e1 e2)`.
    pub amp: f64,
    /// Path length tx -> point -> rx at rest.
    pub len: f64,
    /// Path change per metre of chest displacement.
    pub gain: f64,
}

/// One tx -> rx link: its static channel per subcarrier and the means to
/// add moving paths to it.
#[derive(Debug, Clone)]
pub struct LinkChannel {
    pub tx: Point,
    pub rx: Point,
    pub d_los: f64,
    /// `STATIC_MAGNITUDE * d_los`: makes the unattenuated LoS |h| = 40.
    pub scale: f64,
    carrier_hz: f64,
    spacing_hz: f64,
    n: usize,
    pub static_re: Vec<f64>,
    pub static_im: Vec<f64>,
}

impl LinkChannel {
    /// Static channel: LoS + image-method reflections + scatterers, in the
    /// reference's path order. `scatterers` are (position, RCS).
    pub fn new(room: &Room, scatterers: &[(Point, f64)], tx: Point, rx: Point, carrier_hz: f64, spacing_hz: f64, n: usize) -> Self {
        let d_los = dist(tx, rx);
        assert!(d_los > 0.0, "tx and rx must not coincide");
        let scale = STATIC_MAGNITUDE * d_los;

        let los_gain = 10f64.powf(-room.los_attenuation_db / 20.0);
        let mut paths: Vec<(f64, f64)> = vec![(los_gain / d_los, d_los)];

        let (w, h) = room.size_m;
        let mirrors = |p: Point| -> [(Point, i32); 4] {
            [((-p.0, p.1), 0), ((2.0 * w - p.0, p.1), 1), ((p.0, -p.1), 2), ((p.0, 2.0 * h - p.1), 3)]
        };
        let mut frontier: Vec<(Point, i32)> = vec![(tx, -1)];
        for order in 1..=room.reflection_order {
            let mut next = Vec::with_capacity(frontier.len() * 3);
            for &(src, last_wall) in &frontier {
                for (image, wall) in mirrors(src) {
                    if wall == last_wall {
                        continue; // mirroring straight back is the source itself
                    }
                    let length = dist(image, rx);
                    paths.push(((-room.wall_reflection).powf(order as f64) / length, length));
                    next.push((image, wall));
                }
            }
            frontier = next;
        }

        for &(p, rcs) in scatterers {
            paths.push(point_path(tx, rx, p, rcs));
        }

        let mut ch = Self { tx, rx, d_los, scale, carrier_hz, spacing_hz, n, static_re: vec![0.0; n], static_im: vec![0.0; n] };
        for k in 0..n {
            let f = ch.frequency(k);
            let (mut re, mut im) = (Neumaier::default(), Neumaier::default());
            for &(amp, length) in &paths {
                let theta = -2.0 * PI * f * length / SPEED_OF_LIGHT;
                re.add(amp * theta.cos());
                im.add(amp * theta.sin());
            }
            ch.static_re[k] = scale * re.value();
            ch.static_im[k] = scale * im.value();
        }
        ch
    }

    pub fn n_subcarriers(&self) -> usize {
        self.n
    }

    /// Subcarrier frequency, `carrier + (k - K/2) * spacing`.
    pub fn frequency(&self, k: usize) -> f64 {
        self.carrier_hz + (k as f64 - self.n as f64 / 2.0) * self.spacing_hz
    }

    /// Add `scale * amp * exp(-j 2 pi f_k len / c)` to every subcarrier.
    /// Uses the phasor recurrence across the (uniformly spaced) band: two
    /// sin/cos per path instead of one per subcarrier, ~1e-14 relative
    /// from the direct evaluation.
    pub fn add_path(&self, re: &mut [f64], im: &mut [f64], amp: f64, len: f64) {
        let a = self.scale * amp;
        let th0 = -2.0 * PI * self.frequency(0) * len / SPEED_OF_LIGHT;
        let dth = -2.0 * PI * self.spacing_hz * len / SPEED_OF_LIGHT;
        let (s0, c0) = th0.sin_cos();
        let (ds, dc) = dth.sin_cos();
        let (mut zr, mut zi) = (a * c0, a * s0);
        for (r, i) in re.iter_mut().zip(im.iter_mut()) {
            *r += zr;
            *i += zi;
            let nr = zr * dc - zi * ds;
            zi = zr * ds + zi * dc;
            zr = nr;
        }
    }

    /// The torso as this link sees it: `n_points` scatterers spread across
    /// `width_m` perpendicular to the tx-centre-rx bisector, RCS split
    /// equally (the reference's extended body).
    pub fn torso(&self, centre: Point, n_points: usize, width_m: f64, rcs_m2: f64, out: &mut Vec<BodyPoint>) {
        out.clear();
        let b = bisector(self.tx, self.rx, centre);
        let across = (-b.1, b.0);
        let n = n_points.max(1);
        let k = (rcs_m2 / n as f64 / (4.0 * PI)).sqrt();
        for i in 0..n {
            let off = if n == 1 { 0.0 } else { width_m * (i as f64 / (n - 1) as f64 - 0.5) };
            let p = (centre.0 + off * across.0, centre.1 + off * across.1);
            let e1 = dist(self.tx, p);
            let e2 = dist(p, self.rx);
            out.push(BodyPoint {
                pos: p,
                amp: k / (e1.max(MIN_RANGE_M) * e2.max(MIN_RANGE_M)),
                len: e1 + e2,
                gain: path_change(self.tx, self.rx, p),
            });
        }
    }

    /// Per-subcarrier first-order breathing sensitivity in [0, 1], the
    /// reference's `sensitivity()`: |d|h|/dx| at rest normalised by its
    /// largest possible value. `extra` are further static-at-rest paths
    /// (amp, len), e.g. the arm, included in h0.
    pub fn sensitivity(&self, pts: &[BodyPoint], extra: &[(f64, f64)]) -> Vec<f64> {
        let mut out = Vec::with_capacity(self.n);
        let mut bp: Vec<(f64, f64)> = Vec::with_capacity(pts.len());
        for k in 0..self.n {
            let f = self.frequency(k);
            bp.clear();
            for p in pts {
                let a = self.scale * p.amp;
                let theta = -2.0 * PI * f * p.len / SPEED_OF_LIGHT;
                bp.push((a * theta.cos(), a * theta.sin()));
            }
            let (mut br, mut bi) = (Neumaier::default(), Neumaier::default());
            let (mut dr, mut di) = (Neumaier::default(), Neumaier::default());
            let mut mag = Neumaier::default();
            for (b, p) in bp.iter().zip(pts) {
                br.add(b.0);
                bi.add(b.1);
                // 1j * g * bp
                dr.add(-(p.gain * b.1));
                di.add(p.gain * b.0);
                mag.add(b.0.hypot(b.1) * p.gain);
            }
            let mut h0r = self.static_re[k] + br.value();
            let mut h0i = self.static_im[k] + bi.value();
            for &(amp, len) in extra {
                let a = self.scale * amp;
                let theta = -2.0 * PI * f * len / SPEED_OF_LIGHT;
                h0r += a * theta.cos();
                h0i += a * theta.sin();
            }
            let scale = h0r.hypot(h0i) * mag.value();
            let proj = h0r * dr.value() + h0i * di.value();
            out.push(if scale == 0.0 { 0.0 } else { proj.abs() / scale });
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn path_change_limits() {
        // Monostatic-like geometry far off the link: close to 2.
        let g = path_change((0.0, 0.0), (0.01, 0.0), (0.0, 100.0));
        assert!((g - 2.0).abs() < 1e-6);
        // On the LoS segment: zero.
        let g = path_change((0.0, 0.0), (2.0, 0.0), (1.0, 0.0));
        assert!(g.abs() < 1e-9);
    }

    #[test]
    fn recurrence_matches_direct() {
        let room = Room::default();
        let ch = LinkChannel::new(&room, &[], (1.5, 2.5), (4.5, 2.5), 2.437e9, 312_500.0, 56);
        let (mut re, mut im) = (vec![0.0; 56], vec![0.0; 56]);
        ch.add_path(&mut re, &mut im, 0.05, 3.7);
        for k in 0..56 {
            let th = -2.0 * PI * ch.frequency(k) * 3.7 / SPEED_OF_LIGHT;
            let a = ch.scale * 0.05;
            assert!((re[k] - a * th.cos()).abs() < 1e-12 * a);
            assert!((im[k] - a * th.sin()).abs() < 1e-12 * a);
        }
    }

    #[test]
    fn unattenuated_los_is_forty() {
        let room = Room { reflection_order: 0, n_scatterers: 0, ..Room::default() };
        let ch = LinkChannel::new(&room, &[], (1.0, 1.0), (4.0, 3.0), 2.437e9, 312_500.0, 8);
        for k in 0..8 {
            assert!((ch.static_re[k].hypot(ch.static_im[k]) - 40.0).abs() < 1e-9);
        }
    }
}
