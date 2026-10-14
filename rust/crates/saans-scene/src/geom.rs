//! Plane geometry a scene needs: is this polygon simple, is this point
//! inside it, do these two segments cross.
//!
//! Written out rather than pulled from a crate because the whole repo is
//! dependency-shy (saans-dsp has none at all) and these are twenty lines
//! each. Everything is f64 metres.

use crate::scene::Point;

/// Tolerance for "on the boundary" tests, metres. A millimetre is far
/// below anything a room survey or a RoomPlan scan resolves.
pub const EPS_M: f64 = 1e-9;

/// Twice the signed area of a polygon (the shoelace sum). Positive when
/// the vertices run counter-clockwise.
pub fn signed_area2(poly: &[Point]) -> f64 {
    let n = poly.len();
    let mut acc = 0.0;
    for i in 0..n {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % n];
        acc += x0 * y1 - x1 * y0;
    }
    acc
}

/// Area of a polygon, metres squared.
pub fn area(poly: &[Point]) -> f64 {
    signed_area2(poly).abs() / 2.0
}

/// Axis-aligned bounding box as ((x0, y0), (x1, y1)).
pub fn bbox(poly: &[Point]) -> (Point, Point) {
    let mut x0 = f64::INFINITY;
    let mut y0 = f64::INFINITY;
    let mut x1 = f64::NEG_INFINITY;
    let mut y1 = f64::NEG_INFINITY;
    for &(x, y) in poly {
        x0 = x0.min(x);
        y0 = y0.min(y);
        x1 = x1.max(x);
        y1 = y1.max(y);
    }
    ((x0, y0), (x1, y1))
}

/// Cross product of (b - a) and (c - a). Sign gives the turn direction.
fn cross(a: Point, b: Point, c: Point) -> f64 {
    (b.0 - a.0) * (c.1 - a.1) - (b.1 - a.1) * (c.0 - a.0)
}

/// Is `p` within `EPS_M` of the segment `a`-`b` (endpoints included)?
pub fn on_segment(a: Point, b: Point, p: Point) -> bool {
    let len = (b.0 - a.0).hypot(b.1 - a.1);
    if len <= EPS_M {
        return (p.0 - a.0).hypot(p.1 - a.1) <= EPS_M;
    }
    // Distance from the line, then the clamp onto the span.
    if (cross(a, b, p) / len).abs() > EPS_M {
        return false;
    }
    let t = ((p.0 - a.0) * (b.0 - a.0) + (p.1 - a.1) * (b.1 - a.1)) / (len * len);
    (-EPS_M..=1.0 + EPS_M).contains(&t)
}

/// Do segments `a`-`b` and `c`-`d` intersect, touching included?
pub fn segments_intersect(a: Point, b: Point, c: Point, d: Point) -> bool {
    let (d1, d2) = (cross(c, d, a), cross(c, d, b));
    let (d3, d4) = (cross(a, b, c), cross(a, b, d));
    // Proper crossing: each segment straddles the other's line.
    if ((d1 > 0.0 && d2 < 0.0) || (d1 < 0.0 && d2 > 0.0)) && ((d3 > 0.0 && d4 < 0.0) || (d3 < 0.0 && d4 > 0.0)) {
        return true;
    }
    // Collinear or touching.
    on_segment(c, d, a) || on_segment(c, d, b) || on_segment(a, b, c) || on_segment(a, b, d)
}

/// Is this polygon simple: at least 3 vertices, no zero-length edge, no
/// repeated vertex, and no two non-adjacent edges touching?
///
/// Returns the indices of the first offending edge pair, so the caller can
/// say *which* edges cross rather than just "invalid".
pub fn self_intersection(poly: &[Point]) -> Option<(usize, usize)> {
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        for j in (i + 1)..n {
            // Skip the shared-vertex neighbours; they always "touch".
            let adjacent = j == i + 1 || (i == 0 && j == n - 1);
            if adjacent {
                continue;
            }
            let (c, d) = (poly[j], poly[(j + 1) % n]);
            if segments_intersect(a, b, c, d) {
                return Some((i, j));
            }
        }
    }
    None
}

/// Is `p` inside the polygon? Points on the boundary count as inside,
/// which is what a wall-mounted node needs.
///
/// Crossing-number test, with the boundary handled first so a node sitting
/// exactly on a wall is never called "outside" by a rounding accident.
pub fn point_in_polygon(poly: &[Point], p: Point) -> bool {
    let n = poly.len();
    for i in 0..n {
        if on_segment(poly[i], poly[(i + 1) % n], p) {
            return true;
        }
    }
    let mut inside = false;
    for i in 0..n {
        let (x0, y0) = poly[i];
        let (x1, y1) = poly[(i + 1) % n];
        // Half-open rule on y so a vertex is counted exactly once.
        if (y0 > p.1) != (y1 > p.1) {
            let t = (p.1 - y0) / (y1 - y0);
            if p.0 < x0 + t * (x1 - x0) {
                inside = !inside;
            }
        }
    }
    inside
}

/// Is `p` strictly inside: inside, and at least `margin_m` from every edge?
pub fn point_inside_with_margin(poly: &[Point], p: Point, margin_m: f64) -> bool {
    if !point_in_polygon(poly, p) {
        return false;
    }
    let n = poly.len();
    for i in 0..n {
        if distance_to_segment(poly[i], poly[(i + 1) % n], p) < margin_m {
            return false;
        }
    }
    true
}

/// Shortest distance from `p` to the segment `a`-`b`, metres.
pub fn distance_to_segment(a: Point, b: Point, p: Point) -> f64 {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len2 = dx * dx + dy * dy;
    if len2 <= EPS_M * EPS_M {
        return (p.0 - a.0).hypot(p.1 - a.1);
    }
    let t = (((p.0 - a.0) * dx + (p.1 - a.1) * dy) / len2).clamp(0.0, 1.0);
    (p.0 - (a.0 + t * dx)).hypot(p.1 - (a.1 + t * dy))
}
