//! People of spec/roomsim.md §8: body parameters, the body-point table,
//! posture reshapes and the three-edge cylinder shadowing.
//!
//! A body is 6-16 point scatterers and a cylinder: a SIM model, not a
//! measured human (§16).

use crate::config::{Config, Posture};
use crate::geometry::{dist2, dist3, dot2, v3, P2, V3};
use crate::numerics::{clamp, pow10, sin, Neumaier, PI, TWO_PI};
use crate::room::{body_to_world, ObjGeom};
use saans_sim::rng::Rng;

/// One row of the §8.3 table.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyPoint {
    pub name: &'static str,
    /// Body frame: +x forward.
    pub x: f64,
    /// Body frame: +y left.
    pub y: f64,
    /// Height as a fraction of the person's height.
    pub zfrac: f64,
    pub weight: f64,
    pub swing_amp: f64,
    pub swing_phase: f64,
    pub breath: f64,
}

#[allow(clippy::too_many_arguments)]
const fn bp(
    name: &'static str,
    x: f64,
    y: f64,
    zfrac: f64,
    weight: f64,
    swing_amp: f64,
    swing_phase: f64,
    breath: f64,
) -> BodyPoint {
    BodyPoint {
        name,
        x,
        y,
        zfrac,
        weight,
        swing_amp,
        swing_phase,
        breath,
    }
}

/// The §8.3 table; the first `body_points` rows are used, in order.
pub const BODY_POINTS: [BodyPoint; 16] = [
    bp("chest", 0.12, 0.0, 0.72, 0.20, 0.0, 0.0, 1.0),
    bp("head", 0.0, 0.0, 0.93, 0.08, 0.0, 0.0, 0.0),
    bp("pelvis", 0.0, 0.0, 0.53, 0.15, 0.0, 0.0, 0.0),
    bp("back", -0.12, 0.0, 0.70, 0.15, 0.0, 0.0, 0.0),
    bp("left hand", 0.0, 0.22, 0.45, 0.04, 0.25, PI, 0.0),
    bp("right hand", 0.0, -0.22, 0.45, 0.04, 0.25, 0.0, 0.0),
    bp("left foot", 0.0, 0.10, 0.05, 0.04, 0.30, 0.0, 0.0),
    bp("right foot", 0.0, -0.10, 0.05, 0.04, 0.30, PI, 0.0),
    bp("left shoulder", 0.0, 0.20, 0.80, 0.05, 0.0, 0.0, 0.0),
    bp("right shoulder", 0.0, -0.20, 0.80, 0.05, 0.0, 0.0, 0.0),
    bp("left knee", 0.0, 0.10, 0.28, 0.04, 0.15, 0.0, 0.0),
    bp("right knee", 0.0, -0.10, 0.28, 0.04, 0.15, PI, 0.0),
    bp("left elbow", 0.0, 0.22, 0.62, 0.03, 0.12, PI, 0.0),
    bp("right elbow", 0.0, -0.22, 0.62, 0.03, 0.12, 0.0, 0.0),
    bp("abdomen", 0.11, 0.0, 0.60, 0.08, 0.0, 0.0, 0.5),
    bp("neck", 0.0, 0.0, 0.85, 0.03, 0.0, 0.0, 0.0),
];

/// Point RCS `body_rcs_m2 * w_j / W` of the first `n` rows, `W` the
/// Neumaier sum of their weights (§8.3).
pub fn body_point_rcs(body_rcs_m2: f64, n: usize) -> Vec<f64> {
    let w = Neumaier::sum(BODY_POINTS[..n].iter().map(|p| p.weight));
    BODY_POINTS[..n]
        .iter()
        .map(|p| body_rcs_m2 * p.weight / w)
        .collect()
}

/// Person `i`'s body parameters, drawn from stream `100 + i` (§8.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BodyParams {
    pub height: f64,
    pub f_b: f64,
    pub p2p: f64,
    pub phi_b: f64,
    pub phi_g: f64,
}

impl BodyParams {
    pub fn draw(cfg: &Config, i: usize) -> Self {
        let mut r = Rng::stream(cfg.seed, 100 + i as u64);
        let height = r.range(cfg.height_m.0, cfg.height_m.1);
        let f_b = r.range(cfg.breathing_hz.0, cfg.breathing_hz.1);
        let p2p = r.range(cfg.breathing_p2p_m.0, cfg.breathing_p2p_m.1);
        let phi_b = r.range(0.0, TWO_PI);
        let phi_g = r.range(0.0, TWO_PI);
        Self {
            height,
            f_b,
            p2p,
            phi_b,
            phi_g,
        }
    }
}

/// Where a person is and what they are doing at one instant.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Pose {
    pub x: f64,
    pub y: f64,
    /// Radians. A lying person's body points ignore it (§8.3); it still
    /// orients the phone they carry (§9) and is reported in truth.
    pub heading: f64,
    /// `s = 1`: on a walk segment.
    pub walking: bool,
    /// Leg speed `v` of the walk, m/s (0 when not walking).
    pub speed: f64,
    pub posture: Posture,
}

/// World position of body row `row` at time `t` (§8.3). `bed` is the
/// object a lying person lies on.
pub fn body_point_world(
    row: &BodyPoint,
    body: &BodyParams,
    pose: &Pose,
    bed: Option<&ObjGeom>,
    t: f64,
) -> V3 {
    let s = if pose.walking { 1.0 } else { 0.0 };
    let f_s = if pose.walking {
        clamp(0.95 * (pose.speed / 1.1).sqrt(), 0.5, 1.2)
    } else {
        0.0
    };
    let sw = (row.swing_amp * s) * sin((TWO_PI * f_s) * t + body.phi_g + row.swing_phase);
    let br = (row.breath * (body.p2p / 2.0)) * sin((TWO_PI * body.f_b) * t + body.phi_b);
    let bx = row.x + sw + br;
    let by = row.y;
    let bz = row.zfrac * body.height;
    match pose.posture {
        Posture::Stand => body_to_world(pose.x, pose.y, pose.heading, bx, by, bz),
        Posture::Sit => body_to_world(pose.x, pose.y, pose.heading, bx, by, 0.72 * bz),
        Posture::Lie => {
            let o = bed.expect("a lying person lies on an object");
            let a = if o.sl >= o.sb {
                (o.cs, o.sn)
            } else {
                (-o.sn, o.cs)
            };
            let c = (-a.1, a.0);
            let along = bz - 0.5 * body.height;
            v3(
                pose.x + (along * a.0 + by * c.0),
                pose.y + (along * a.1 + by * c.1),
                o.top() + 0.12 + bx,
            )
        }
    }
}

/// Top of a person's shadowing cylinder (§8.5).
pub fn cylinder_top(body: &BodyParams, posture: Posture, bed: Option<&ObjGeom>) -> f64 {
    match posture {
        Posture::Stand => body.height,
        Posture::Sit => 0.72 * body.height,
        Posture::Lie => bed.expect("a lying person lies on an object").top() + 0.25,
    }
}

/// ITU-R P.526 single knife edge `J1(nu)`, dB (§8.5).
pub fn j1(nu: f64) -> f64 {
    if nu <= -0.78 {
        return 0.0;
    }
    let w = nu - 0.1;
    6.9 + 20.0 * (((w * w + 1.0).sqrt() + nu) - 0.1).log10()
}

/// Three-edge loss `J` of §8.5 step 3, dB: the ITU-R P.526
/// finite-width-screen average over the near side edge `nu_1`, the far
/// side edge `nu_2` and the top `nu_t`,
/// `clamp(-10.0 * log10((10^(-J1(nu_1)/10) + 10^(-J1(nu_2)/10)) + 10^(-J1(nu_t)/10)), 0, cap)`.
pub fn three_edge_j(nu_1: f64, nu_2: f64, nu_t: f64, shadow_cap_db: f64) -> f64 {
    clamp(
        -10.0
            * ((pow10(-j1(nu_1) / 10.0) + pow10(-j1(nu_2) / 10.0)) + pow10(-j1(nu_t) / 10.0))
                .log10(),
        0.0,
        shadow_cap_db,
    )
}

/// A person as a shadowing cylinder (§8.5).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Cylinder {
    pub c: P2,
    pub radius: f64,
    pub top: f64,
}

/// The loss `J`, dB, of leg `p1 -> p2` past cylinder `cy` (§8.5 steps
/// 1-3), or `None` for no loss.
pub fn leg_shadow_db(
    p1: V3,
    p2: V3,
    cy: &Cylinder,
    lambda: f64,
    shadow_cap_db: f64,
) -> Option<f64> {
    let r = (p2.x - p1.x, p2.y - p1.y);
    let l2 = dot2(r, r);
    if l2 == 0.0 {
        return None;
    }
    let u = dot2((cy.c.0 - p1.x, cy.c.1 - p1.y), r) / l2;
    if u <= 0.0 || u >= 1.0 {
        let end = if u <= 0.0 { p1 } else { p2 };
        if dist2(cy.c, (end.x, end.y)) < cy.radius && end.z <= cy.top {
            return Some(shadow_cap_db);
        }
        return None;
    }
    let d = dist3(p1, p2);
    let z = p1.z + u * (p2.z - p1.z);
    let d_perp = dist2(cy.c, (p1.x + u * r.0, p1.y + u * r.1));
    let d1 = u * d;
    let d2 = (1.0 - u) * d;
    let k = (2.0 * (d1 + d2) / ((lambda * d1) * d2)).sqrt();
    let nu_1 = (cy.radius - d_perp) * k;
    let nu_2 = (cy.radius + d_perp) * k;
    let nu_t = (cy.top - z) * k;
    Some(three_edge_j(nu_1, nu_2, nu_t, shadow_cap_db))
}

/// Shadow factor `S` of a polyline past `people` (§8.5): the product, legs
/// in order and people in order within each leg, of `10^(-J/20)`.
/// `skip` is a person whose own body this polyline belongs to.
pub fn shadow_factor(
    points: &[V3],
    people: &[Cylinder],
    skip: Option<usize>,
    lambda: f64,
    cap: f64,
) -> f64 {
    let mut s = 1.0;
    for leg in points.windows(2) {
        for (q, cy) in people.iter().enumerate() {
            if skip == Some(q) {
                continue;
            }
            if let Some(j) = leg_shadow_db(leg[0], leg[1], cy, lambda, cap) {
                s *= pow10(-j / 20.0);
            }
        }
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn j1_at_zero_is_the_closed_form() {
        let want = 6.9 + 20.0 * ((1.01f64).sqrt() - 0.1).log10();
        assert_eq!(j1(0.0), want);
        assert!((j1(0.0) - 6.03).abs() < 0.01, "{}", j1(0.0));
        assert_eq!(j1(-0.78), 0.0);
        assert!(j1(-0.77) > -0.1);
        assert!(j1(3.0) > j1(1.0));
    }

    #[test]
    fn three_edge_j_is_clamped() {
        assert_eq!(three_edge_j(-5.0, -5.0, -5.0, 20.0), 0.0);
        assert_eq!(three_edge_j(50.0, 50.0, 50.0, 20.0), 20.0);
        // Three equal edges: J1 less 10*log10(3).
        let j = three_edge_j(1.0, 1.0, 1.0, 40.0);
        assert!((j - (j1(1.0) - 10.0 * 3f64.log10())).abs() < 1e-12);
        // One clear edge (J1 = 0) lets everything through: J = 0.
        assert_eq!(three_edge_j(-0.78, 3.0, 3.0, 20.0), 0.0);
        assert_eq!(three_edge_j(3.0, 3.0, -1.0, 20.0), 0.0);
    }

    /// §8.5: a leg through the axis below the head loses about 8.7 dB on
    /// a 4 m link (the bedroom's 3.9 m wall-side link, a 1.7 m person
    /// half-way), and J falls off smoothly and symmetrically to either side.
    #[test]
    fn a_leg_through_the_axis_loses_about_8_7_db() {
        let (p1, p2) = (v3(0.15, 0.15, 1.2), v3(4.05, 0.15, 1.2));
        let at = |dy: f64| Cylinder {
            c: (2.1, 0.15 + dy),
            radius: 0.18,
            top: 1.7,
        };
        let lam = crate::numerics::C / crate::numerics::carrier_hz(6);
        let j = |dy: f64| leg_shadow_db(p1, p2, &at(dy), lam, 20.0).unwrap();
        assert!((j(0.0) - 8.7).abs() < 0.05, "{}", j(0.0));
        assert!((j(0.05) - j(-0.05)).abs() < 1e-9, "either side of the axis");
        assert!(j(0.0) > j(0.02) && j(0.02) > j(0.05) && j(0.05) > j(0.1));
        assert!(j(0.0) - j(0.02) < 0.1, "no cusp at the axis");
        assert_eq!(j(1.0), 0.0, "a leg well beside the body");
    }

    #[test]
    fn rcs_sums_to_body_rcs() {
        for n in 6..=16 {
            let r = body_point_rcs(0.5, n);
            assert_eq!(r.len(), n);
            assert!((Neumaier::sum(r.iter().copied()) - 0.5).abs() < 1e-15);
        }
    }

    #[test]
    fn leg_ends_inside_the_cylinder_take_the_cap() {
        let cy = Cylinder {
            c: (0.0, 0.0),
            radius: 0.18,
            top: 1.7,
        };
        let lam = 0.123;
        assert_eq!(
            leg_shadow_db(v3(0.1, 0.0, 1.0), v3(3.0, 0.0, 1.0), &cy, lam, 20.0),
            Some(20.0)
        );
        assert_eq!(
            leg_shadow_db(v3(0.1, 0.0, 1.8), v3(3.0, 0.0, 1.8), &cy, lam, 20.0),
            None,
            "above the top"
        );
        assert_eq!(
            leg_shadow_db(v3(1.0, 0.0, 1.0), v3(1.0, 0.0, 2.0), &cy, lam, 20.0),
            None,
            "vertical leg"
        );
        let mid = leg_shadow_db(v3(-2.0, 0.0, 1.0), v3(2.0, 0.0, 1.0), &cy, lam, 20.0).unwrap();
        assert!(mid > 8.0 && mid < 10.0, "{mid}");
    }
}
