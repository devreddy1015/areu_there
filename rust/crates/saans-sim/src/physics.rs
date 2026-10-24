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
//!   the wall just used; in a rectangular room every image is valid). Like
//!   the reference, this reaches each order-2 corner image twice and adds
//!   both, which is a known bug kept for parity (NOTES.md §13);
//! - point scatterers (furniture, torso points, the arm):
//!   `sqrt(rcs / 4 pi) / (d1 d2)`.
//!
//! Each link is scaled so its *unattenuated* LoS has |h| = 40. Chest motion
//! along the tx-body-rx bisector changes a torso point's path by
//! `2 cos(beta/2)` per metre (`path_change`).
//!
//! With [`Room::vertical`] the room is a box, the reference's
//! `room_height_m`: every distance is 3-D, the image method runs over
//! walls, floor and ceiling with each lattice image once ([`box_images`]),
//! and the chest moves horizontally ([`chest_motion`]). Without it, the
//! slice above is computed exactly as it always was, z ignored.
//!
//! Where the reference relies on CPython semantics that change the last
//! bits (Neumaier-compensated `sum()` of floats and, since 3.14, of
//! complex numbers), the construction-time code here copies them so the
//! static channel matches the golden vectors as closely as possible.

use crate::scenario::{Point, Point3, Room, Vertical};
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

pub fn dist3(a: Point3, b: Point3) -> f64 {
    (a.0 - b.0).hypot(a.1 - b.1).hypot(a.2 - b.2)
}

fn xy(p: Point3) -> Point {
    (p.0, p.1)
}

/// Image `n` of `x` across the walls at 0 and `length`, and its bounces
/// off each: (position, at 0, at length). Image n is at `n*length + x` for
/// even n and `n*length + length - x` for odd n, |n| bounces alternating
/// between the walls, the first off the wall at `length` when n > 0.
fn axis_image(x: f64, length: f64, n: i32) -> (f64, u32, u32) {
    let pos = n as f64 * length + if n % 2 == 0 { x } else { length - x };
    let m = n.unsigned_abs();
    if n >= 0 {
        (pos, m / 2, m.div_ceil(2))
    } else {
        (pos, m.div_ceil(2), m / 2)
    }
}

/// Every image of `p` in a box with 1 to `order` bounces, each exactly once,
/// with the product of -Gamma over its bounces: the lattice (nx, ny, nz)
/// with |nx| + |ny| + |nz| <= order, by bounces and then nx, ny, nz
/// ascending, the reference's `_box_images` and its summation order.
///
/// The slice's breadth-first search reaches each order-2 corner image by
/// two routes and adds both (NOTES.md §13); the lattice cannot.
pub fn box_images(
    p: Point3,
    size: Point3,
    order: u32,
    wall: f64,
    floor: f64,
    ceiling: f64,
) -> Vec<(Point3, f64)> {
    let mut out = Vec::new();
    for total in 1..=order as i32 {
        for nx in -total..=total {
            let rest = total - nx.abs();
            for ny in -rest..=rest {
                let rz = rest - ny.abs();
                let zs = [-rz, rz];
                for &nz in &zs[..if rz == 0 { 1 } else { 2 }] {
                    let (x, _, _) = axis_image(p.0, size.0, nx);
                    let (y, _, _) = axis_image(p.1, size.1, ny);
                    let (z, on_floor, on_ceiling) = axis_image(p.2, size.2, nz);
                    let gain = (-wall).powf((nx.abs() + ny.abs()) as f64)
                        * (-floor).powf(on_floor as f64)
                        * (-ceiling).powf(on_ceiling as f64);
                    out.push(((x, y, z), gain));
                }
            }
        }
    }
    out
}

/// Which way a chest at `p` breathes, and the path change per metre along
/// it. The chest moves horizontally; along a horizontal unit vector n the
/// path tx-p-rx shortens by (u_tx + u_rx) . n per metre, largest along the
/// horizontal part of u_tx + u_rx and equal to its length. Level with
/// both ends that is `2 cos(beta/2)`; straight below the link's midpoint
/// it is 0, and the direction is then taken across the link.
pub fn chest_motion(tx: Point3, rx: Point3, p: Point3) -> (Point, f64) {
    let d1 = dist3(tx, p).max(1e-12);
    let d2 = dist3(p, rx).max(1e-12);
    let sx = (tx.0 - p.0) / d1 + (rx.0 - p.0) / d2;
    let sy = (tx.1 - p.1) / d1 + (rx.1 - p.1) / d2;
    let norm = sx.hypot(sy);
    if norm < 1e-12 {
        let (hx, hy) = (rx.0 - tx.0, rx.1 - tx.1);
        let hn = hx.hypot(hy);
        let across = if hn > 0.0 {
            (-hy / hn, hx / hn)
        } else {
            (1.0, 0.0)
        };
        return (across, 0.0);
    }
    ((sx / norm, sy / norm), norm)
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

/// [`point_path`] in a box.
pub fn point_path3(tx: Point3, rx: Point3, p: Point3, rcs_m2: f64) -> (f64, f64) {
    let d1 = dist3(tx, p);
    let d2 = dist3(p, rx);
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
    /// Node positions; z is ignored unless `vertical` is set.
    pub tx: Point3,
    pub rx: Point3,
    /// The box, when the room is one.
    pub vertical: Option<Vertical>,
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
    /// reference's path order. `scatterers` are (position, RCS). In the
    /// 2-D slice every z is ignored.
    pub fn new(
        room: &Room,
        scatterers: &[(Point3, f64)],
        tx: Point3,
        rx: Point3,
        carrier_hz: f64,
        spacing_hz: f64,
        n: usize,
    ) -> Self {
        let (d_los, paths) = match &room.vertical {
            None => Self::slice_paths(room, scatterers, xy(tx), xy(rx)),
            Some(v) => Self::box_paths(room, v, scatterers, tx, rx),
        };
        let scale = STATIC_MAGNITUDE * d_los;

        let mut ch = Self {
            tx,
            rx,
            vertical: room.vertical,
            d_los,
            scale,
            carrier_hz,
            spacing_hz,
            n,
            static_re: vec![0.0; n],
            static_im: vec![0.0; n],
        };
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

    /// (d_los, unscaled static paths) in the box: the reference's
    /// `_init_box`.
    fn box_paths(
        room: &Room,
        v: &Vertical,
        scatterers: &[(Point3, f64)],
        tx: Point3,
        rx: Point3,
    ) -> (f64, Vec<(f64, f64)>) {
        let d_los = dist3(tx, rx);
        assert!(d_los > 0.0, "tx and rx must not coincide");
        let los_gain = 10f64.powf(-room.los_attenuation_db / 20.0);
        let mut paths: Vec<(f64, f64)> = vec![(los_gain / d_los, d_los)];
        let size = (room.size_m.0, room.size_m.1, v.height_m);
        for (image, gain) in box_images(
            tx,
            size,
            room.reflection_order,
            room.wall_reflection,
            v.floor_reflection,
            v.ceiling_reflection,
        ) {
            let length = dist3(image, rx);
            paths.push((gain / length, length));
        }
        for &(p, rcs) in scatterers {
            paths.push(point_path3(tx, rx, p, rcs));
        }
        (d_los, paths)
    }

    /// (d_los, unscaled static paths) in the 2-D slice, unchanged from the
    /// first port.
    fn slice_paths(
        room: &Room,
        scatterers: &[(Point3, f64)],
        tx: Point,
        rx: Point,
    ) -> (f64, Vec<(f64, f64)>) {
        let d_los = dist(tx, rx);
        assert!(d_los > 0.0, "tx and rx must not coincide");

        let los_gain = 10f64.powf(-room.los_attenuation_db / 20.0);
        let mut paths: Vec<(f64, f64)> = vec![(los_gain / d_los, d_los)];

        let (w, h) = room.size_m;
        let mirrors = |p: Point| -> [(Point, i32); 4] {
            [
                ((-p.0, p.1), 0),
                ((2.0 * w - p.0, p.1), 1),
                ((p.0, -p.1), 2),
                ((p.0, 2.0 * h - p.1), 3),
            ]
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
            paths.push(point_path(tx, rx, xy(p), rcs));
        }
        (d_los, paths)
    }

    /// A point scatterer on the body at `p` (the arm), as this link sees
    /// it: in the box, at chest height.
    pub fn body_point_path(&self, p: Point, rcs_m2: f64) -> (f64, f64) {
        match &self.vertical {
            None => point_path(xy(self.tx), xy(self.rx), p, rcs_m2),
            Some(v) => point_path3(self.tx, self.rx, (p.0, p.1, v.chest_z_m), rcs_m2),
        }
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
    pub fn torso(
        &self,
        centre: Point,
        n_points: usize,
        width_m: f64,
        rcs_m2: f64,
        out: &mut Vec<BodyPoint>,
    ) {
        out.clear();
        let n = n_points.max(1);
        let k = (rcs_m2 / n as f64 / (4.0 * PI)).sqrt();
        let offset = |i: usize| {
            if n == 1 {
                0.0
            } else {
                width_m * (i as f64 / (n - 1) as f64 - 0.5)
            }
        };
        if let Some(v) = &self.vertical {
            // The reference's `_init_box`: points across the horizontal
            // bisector at chest height, each with its own horizontal gain.
            let z = v.chest_z_m;
            let (b, _) = chest_motion(self.tx, self.rx, (centre.0, centre.1, z));
            let across = (-b.1, b.0);
            for i in 0..n {
                let off = offset(i);
                let p = (centre.0 + off * across.0, centre.1 + off * across.1, z);
                let e1 = dist3(self.tx, p);
                let e2 = dist3(p, self.rx);
                out.push(BodyPoint {
                    pos: xy(p),
                    amp: k / (e1.max(MIN_RANGE_M) * e2.max(MIN_RANGE_M)),
                    len: e1 + e2,
                    gain: chest_motion(self.tx, self.rx, p).1,
                });
            }
            return;
        }
        let (tx, rx) = (xy(self.tx), xy(self.rx));
        let b = bisector(tx, rx, centre);
        let across = (-b.1, b.0);
        for i in 0..n {
            let off = offset(i);
            let p = (centre.0 + off * across.0, centre.1 + off * across.1);
            let e1 = dist(tx, p);
            let e2 = dist(p, rx);
            out.push(BodyPoint {
                pos: p,
                amp: k / (e1.max(MIN_RANGE_M) * e2.max(MIN_RANGE_M)),
                len: e1 + e2,
                gain: path_change(tx, rx, p),
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
            out.push(if scale == 0.0 {
                0.0
            } else {
                proj.abs() / scale
            });
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
        let ch = LinkChannel::new(
            &room,
            &[],
            (1.5, 2.5, 1.2),
            (4.5, 2.5, 1.2),
            2.437e9,
            312_500.0,
            56,
        );
        let (mut re, mut im) = (vec![0.0; 56], vec![0.0; 56]);
        ch.add_path(&mut re, &mut im, 0.05, 3.7);
        for k in 0..56 {
            let th = -2.0 * PI * ch.frequency(k) * 3.7 / SPEED_OF_LIGHT;
            let a = ch.scale * 0.05;
            assert!((re[k] - a * th.cos()).abs() < 1e-12 * a);
            assert!((im[k] - a * th.sin()).abs() < 1e-12 * a);
        }
    }

    /// A box that is the slice in disguise: floor and ceiling absorbing,
    /// and everything at the chest's height.
    fn flat(room: &Room) -> Room {
        Room {
            vertical: Some(Vertical {
                height_m: 2.7,
                floor_reflection: 0.0,
                ceiling_reflection: 0.0,
                chest_z_m: 1.2,
                scatterer_z_m: (1.2, 1.2),
            }),
            ..room.clone()
        }
    }

    const TX: Point3 = (1.5, 2.5, 1.2);
    const RX: Point3 = (4.5, 2.5, 1.2);

    #[test]
    fn box_images_each_once() {
        // Lattice points with 1 <= |nx|+|ny|+|nz| <= N: 6, 24, 62.
        for (order, count) in [(0, 0), (1, 6), (2, 24), (3, 62)] {
            let images = box_images((1.1, 2.3, 0.7), (6.0, 5.0, 2.7), order, 0.4, 0.3, 0.2);
            assert_eq!(images.len(), count);
            let mut keys: Vec<_> = images
                .iter()
                .map(|(p, _)| {
                    let r = |x: f64| (x * 1e9).round() as i64;
                    (r(p.0), r(p.1), r(p.2))
                })
                .collect();
            keys.sort();
            keys.dedup();
            assert_eq!(keys.len(), count, "order {order}: an image repeats");
        }
        // Floor and ceiling carry their own Gamma, with the phase flip.
        let first = box_images((1.0, 2.0, 1.2), (6.0, 5.0, 2.7), 1, 0.4, 0.3, 0.2);
        assert_eq!(first[2], ((1.0, 2.0, -1.2), -0.3));
        assert!((first[3].0 .2 - 4.2).abs() < 1e-12 && first[3].1 == -0.2);
    }

    #[test]
    fn a_flat_box_is_the_slice_at_order_one() {
        let room = Room::default();
        let scat = [((2.0, 4.0, 1.2), 0.3), ((5.0, 1.0, 1.2), 0.1)];
        let a = LinkChannel::new(&room, &scat, TX, RX, 2.437e9, 312_500.0, 56);
        let b = LinkChannel::new(&flat(&room), &scat, TX, RX, 2.437e9, 312_500.0, 56);
        for k in 0..56 {
            let scale = a.static_re[k].hypot(a.static_im[k]);
            assert!((a.static_re[k] - b.static_re[k]).abs() < 1e-13 * scale);
            assert!((a.static_im[k] - b.static_im[k]).abs() < 1e-13 * scale);
        }
        let (mut pa, mut pb) = (Vec::new(), Vec::new());
        a.torso((2.6, 3.4), 3, 0.3, BODY_RCS_M2, &mut pa);
        b.torso((2.6, 3.4), 3, 0.3, BODY_RCS_M2, &mut pb);
        for (x, y) in pa.iter().zip(&pb) {
            assert!((x.gain - y.gain).abs() < 1e-12);
            assert!((x.len - y.len).abs() < 1e-12);
            assert!((x.amp - y.amp).abs() < 1e-12 * x.amp);
        }
    }

    #[test]
    fn at_order_two_the_slice_counts_each_corner_twice() {
        // The known bug the slice keeps for parity (NOTES.md §13): it
        // differs from the flat box by exactly one more copy of each of
        // the four corner paths.
        let room = Room {
            reflection_order: 2,
            wall_reflection: 0.45,
            ..Room::default()
        };
        let a = LinkChannel::new(&room, &[], TX, RX, 2.437e9, 312_500.0, 56);
        let b = LinkChannel::new(&flat(&room), &[], TX, RX, 2.437e9, 312_500.0, 56);
        let (w, h) = room.size_m;
        let (mut re, mut im) = (b.static_re.clone(), b.static_im.clone());
        for cx in [-TX.0, 2.0 * w - TX.0] {
            for cy in [-TX.1, 2.0 * h - TX.1] {
                let len = dist((cx, cy), (RX.0, RX.1));
                b.add_path(&mut re, &mut im, 0.45 * 0.45 / len, len);
            }
        }
        for k in 0..56 {
            let scale = a.static_re[k].hypot(a.static_im[k]);
            assert!((a.static_re[k] - re[k]).abs() < 1e-12 * scale);
            assert!((a.static_im[k] - im[k]).abs() < 1e-12 * scale);
        }
    }

    #[test]
    fn the_chest_moves_horizontally() {
        let (tx, rx) = ((1.0, 2.5, 2.0), (5.0, 2.5, 2.0));
        // Level with the link: the slice's 2 cos(beta/2).
        let (_, level) = chest_motion(tx, rx, (3.0, 4.0, 2.0));
        assert!((level - path_change((1.0, 2.5), (5.0, 2.5), (3.0, 4.0))).abs() < 1e-12);
        // A metre below it: only the horizontal part of u_tx + u_rx.
        let (_, below) = chest_motion(tx, rx, (3.0, 4.0, 1.0));
        assert!((below - 2.0 * 1.5 / (4.0f64 + 2.25 + 1.0).sqrt()).abs() < 1e-12);
        // Straight below the midpoint: none at first order, and the
        // direction falls back to across the link.
        assert_eq!(chest_motion(tx, rx, (3.0, 2.5, 1.0)), ((-0.0, 1.0), 0.0));
    }

    #[test]
    fn the_arm_sits_at_chest_height_in_a_box() {
        let room = Room {
            vertical: Some(Vertical::default()),
            ..Room::default()
        };
        let ch = LinkChannel::new(&room, &[], TX, RX, 2.437e9, 312_500.0, 8);
        assert_eq!(
            ch.body_point_path((3.0, 3.5), 0.05),
            point_path3(TX, RX, (3.0, 3.5, Vertical::default().chest_z_m), 0.05)
        );
        let slice = LinkChannel::new(&Room::default(), &[], TX, RX, 2.437e9, 312_500.0, 8);
        assert_eq!(
            slice.body_point_path((3.0, 3.5), 0.05),
            point_path((1.5, 2.5), (4.5, 2.5), (3.0, 3.5), 0.05)
        );
    }

    #[test]
    fn unattenuated_los_is_forty() {
        let room = Room {
            reflection_order: 0,
            n_scatterers: 0,
            ..Room::default()
        };
        let ch = LinkChannel::new(
            &room,
            &[],
            (1.0, 1.0, 1.2),
            (4.0, 3.0, 1.2),
            2.437e9,
            312_500.0,
            8,
        );
        for k in 0..8 {
            assert!((ch.static_re[k].hypot(ch.static_im[k]) - 40.0).abs() < 1e-9);
        }
    }
}
