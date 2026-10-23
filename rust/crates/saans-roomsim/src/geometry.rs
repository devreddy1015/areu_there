//! Geometry primitives of spec/roomsim.md §7.
//!
//! 2-D points are `(x, y)` tuples (the `saans_scene::Point` convention),
//! 3-D points are [`V3`]. Every length is `sqrt(dx*dx + dy*dy)` or
//! `sqrt((dx*dx + dy*dy) + dz*dz)`: never `hypot`.

use crate::numerics::{clamp, pymax, pymin};

/// A point on the floor plane, metres.
pub type P2 = (f64, f64);

/// A point or vector in the scene frame, metres.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct V3 {
    pub x: f64,
    pub y: f64,
    pub z: f64,
}

/// `V3 { x, y, z }`.
pub const fn v3(x: f64, y: f64, z: f64) -> V3 {
    V3 { x, y, z }
}

impl V3 {
    pub fn xy(self) -> P2 {
        (self.x, self.y)
    }
}

/// `|P - Q| = sqrt((dx*dx + dy*dy) + dz*dz)` (§7.1).
pub fn dist3(p: V3, q: V3) -> f64 {
    let dx = q.x - p.x;
    let dy = q.y - p.y;
    let dz = q.z - p.z;
    ((dx * dx + dy * dy) + dz * dz).sqrt()
}

/// xy length `sqrt(dx*dx + dy*dy)` (§7.1).
pub fn dist2(p: P2, q: P2) -> f64 {
    let dx = q.0 - p.0;
    let dy = q.1 - p.1;
    (dx * dx + dy * dy).sqrt()
}

/// `a.x*b.x + a.y*b.y`.
pub fn dot2(a: P2, b: P2) -> f64 {
    a.0 * b.0 + a.1 * b.1
}

/// `(a.x*b.x + a.y*b.y) + a.z*b.z`.
pub fn dot3(a: V3, b: V3) -> f64 {
    (a.x * b.x + a.y * b.y) + a.z * b.z
}

/// `b - a`, componentwise.
pub fn sub3(b: V3, a: V3) -> V3 {
    v3(b.x - a.x, b.y - a.y, b.z - a.z)
}

/// Unit direction from `p` to `q`: `(dx/D, dy/D, dz/D)`, `D = |q - p|` (§7.1).
pub fn unit_dir(p: V3, q: V3) -> V3 {
    let dx = q.x - p.x;
    let dy = q.y - p.y;
    let dz = q.z - p.z;
    let d = ((dx * dx + dy * dy) + dz * dz).sqrt();
    v3(dx / d, dy / d, dz / d)
}

/// `pdist(p, a, b)`: distance from `p` to segment `a`-`b` (§7.2).
pub fn pdist(p: P2, a: P2, b: P2) -> f64 {
    let d = (b.0 - a.0, b.1 - a.1);
    let dd = dot2(d, d);
    if dd == 0.0 {
        return dist2(p, a);
    }
    let t = clamp(dot2((p.0 - a.0, p.1 - a.1), d) / dd, 0.0, 1.0);
    dist2(p, (a.0 + t * d.0, a.1 + t * d.1))
}

/// The edges `(F[i], F[(i+1) % n])` of a polygon, in order.
pub fn edges(f: &[P2]) -> impl Iterator<Item = (P2, P2)> + '_ {
    let n = f.len();
    (0..n).map(move |i| (f[i], f[(i + 1) % n]))
}

/// `inside(F, p)` (§7.2): on an edge (within 1e-9), or by crossing number.
pub fn inside(f: &[P2], p: P2) -> bool {
    if edges(f).any(|(a, b)| pdist(p, a, b) <= 1e-9) {
        return true;
    }
    let mut c = false;
    for (a, b) in edges(f) {
        if (a.1 > p.1) != (b.1 > p.1) && p.0 < a.0 + (p.1 - a.1) * (b.0 - a.0) / (b.1 - a.1) {
            c = !c;
        }
    }
    c
}

/// `inside_with_margin(F, p, m)` (§7.2): inside, and `pdist >= m` for every edge.
pub fn inside_with_margin(f: &[P2], p: P2, m: f64) -> bool {
    inside(f, p) && edges(f).all(|(a, b)| pdist(p, a, b) >= m)
}

/// The §6.1 wall test for segment `p`→`q` against segment `a`→`b`, 2-D:
/// `(s, v)` or `None` when `den == 0`.
pub fn seg_params(p: P2, q: P2, a: P2, b: P2) -> Option<(f64, f64)> {
    let r = (q.0 - p.0, q.1 - p.1);
    let w = (b.0 - a.0, b.1 - a.1);
    let den = r.0 * w.1 - r.1 * w.0;
    if den == 0.0 {
        return None;
    }
    let ap = (a.0 - p.0, a.1 - p.1);
    let s = (ap.0 * w.1 - ap.1 * w.0) / den;
    let v = (ap.0 * r.1 - ap.1 * r.0) / den;
    Some((s, v))
}

/// Segment–segment distance, 2-D (§7.3): 0 if they cross (`0 <= s <= 1`
/// and `0 <= v <= 1`), else the least of the four point–segment distances.
pub fn seg_seg_dist(p: P2, q: P2, a: P2, b: P2) -> f64 {
    if let Some((s, v)) = seg_params(p, q, a, b) {
        if (0.0..=1.0).contains(&s) && (0.0..=1.0).contains(&v) {
            return 0.0;
        }
    }
    let d = pdist(p, a, b);
    let d = pymin(d, pdist(q, a, b));
    let d = pymin(d, pdist(a, p, q));
    pymin(d, pdist(b, p, q))
}

/// Slab test of §7.4 on object-local coordinates: does the segment from
/// `start` to `end` meet the box with these half-extents? Two axes for the
/// 2-D version, three for the 3-D one.
pub fn slab_meets(start: &[f64], end: &[f64], half: &[f64]) -> bool {
    let mut t_enter = 0.0;
    let mut t_exit = 1.0;
    for ((&p, &e), &h) in start.iter().zip(end).zip(half) {
        let d = e - p;
        if d == 0.0 {
            if p.abs() > h {
                return false;
            }
        } else {
            let t1 = (-h - p) / d;
            let t2 = (h - p) / d;
            t_enter = pymax(t_enter, pymin(t1, t2));
            t_exit = pymin(t_exit, pymax(t1, t2));
        }
    }
    t_exit > t_enter
}

/// Bounding box `(x0, y0, x1, y1)` of a polygon.
pub fn bbox(f: &[P2]) -> (f64, f64, f64, f64) {
    let mut b = (f[0].0, f[0].1, f[0].0, f[0].1);
    for &(x, y) in &f[1..] {
        b.0 = pymin(b.0, x);
        b.1 = pymin(b.1, y);
        b.2 = pymax(b.2, x);
        b.3 = pymax(b.3, y);
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;

    const SQUARE: [P2; 4] = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)];
    const ELL: [P2; 6] = [
        (0.0, 0.0),
        (9.0, 0.0),
        (9.0, 3.0),
        (4.0, 3.0),
        (4.0, 7.0),
        (0.0, 7.0),
    ];

    #[test]
    fn inside_and_margins() {
        assert!(inside(&SQUARE, (1.0, 1.0)));
        assert!(inside(&SQUARE, (0.0, 1.0)), "on an edge counts as inside");
        assert!(!inside(&SQUARE, (5.0, 1.0)));
        assert!(inside(&ELL, (2.0, 6.0)));
        assert!(!inside(&ELL, (6.0, 6.0)));
        assert!(inside_with_margin(&SQUARE, (1.0, 1.0), 0.5));
        assert!(!inside_with_margin(&SQUARE, (0.2, 1.0), 0.5));
        assert!(
            !inside_with_margin(&ELL, (4.2, 3.2), 0.4),
            "near the reflex corner"
        );
    }

    #[test]
    fn distances() {
        assert_eq!(pdist((2.0, 2.0), (0.0, 0.0), (4.0, 0.0)), 2.0);
        assert_eq!(pdist((-3.0, 4.0), (0.0, 0.0), (4.0, 0.0)), 5.0);
        assert_eq!(pdist((1.0, 1.0), (0.0, 0.0), (0.0, 0.0)), 2f64.sqrt());
        assert_eq!(
            seg_seg_dist((0.0, -1.0), (0.0, 1.0), (-1.0, 0.0), (1.0, 0.0)),
            0.0
        );
        assert_eq!(
            seg_seg_dist((0.0, 1.0), (0.0, 2.0), (-1.0, 0.0), (1.0, 0.0)),
            1.0
        );
        assert_eq!(
            seg_seg_dist((0.0, 1.0), (1.0, 1.0), (0.0, 0.0), (1.0, 0.0)),
            1.0,
            "parallel"
        );
        assert_eq!(dist3(v3(0.0, 0.0, 0.0), v3(2.0, 3.0, 6.0)), 7.0);
    }

    #[test]
    fn slab_test() {
        let h = [1.0, 1.0, 1.0];
        assert!(slab_meets(&[-2.0, 0.0, 0.0], &[2.0, 0.0, 0.0], &h));
        assert!(!slab_meets(&[-2.0, 2.0, 0.0], &[2.0, 2.0, 0.0], &h));
        assert!(
            !slab_meets(&[-3.0, 0.0, 0.0], &[-2.0, 0.0, 0.0], &h),
            "stops short"
        );
        assert!(
            slab_meets(&[0.0, 0.0, 0.0], &[0.5, 0.0, 0.0], &h),
            "starts inside"
        );
        assert!(
            slab_meets(&[-2.0, 1.0, 0.0], &[2.0, 1.0, 0.0], &h),
            "the box is closed: a leg along a face meets it"
        );
        assert!(slab_meets(&[-2.0, 0.0], &[2.0, 0.0], &[1.0, 1.0]), "2-D");
    }
}
