//! Materials of spec/roomsim.md §5: ITU-R P.2040 slabs for walls, floor
//! and ceiling, single interfaces and fixed losses for objects.
//!
//! The electrical constants are ITU-R P.2040-3 Table 3 (REAL, published);
//! the object `rho` and `L_db` values are SIM assumptions for Phase 10.

use crate::numerics::{cadd, cdiv, cexp, cmul, cscale, csqrt, csub, cx, Cx, EPS0, TWO_PI};
use saans_scene::{ObjectMaterial, WallMaterial};
use serde::{Deserialize, Serialize};

/// An ITU-R P.2040-3 material (§5.1 table).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ItuMaterial {
    Concrete,
    Brick,
    Plasterboard,
    Glass,
    Wood,
    Metal,
}

impl ItuMaterial {
    /// `(a, b, c, d)`: `er = a * f_ghz^b`, `sigma = c * f_ghz^d` S/m.
    pub fn params(self) -> (f64, f64, f64, f64) {
        match self {
            Self::Concrete => (5.24, 0.0, 0.0462, 0.7822),
            Self::Brick => (3.91, 0.0, 0.0238, 0.16),
            Self::Plasterboard => (2.73, 0.0, 0.0085, 0.9395),
            Self::Glass => (6.31, 0.0, 0.0036, 1.3394),
            Self::Wood => (1.99, 0.0, 0.0047, 1.0718),
            Self::Metal => (1.0, 0.0, 1e7, 0.0),
        }
    }
}

/// A material a wall, the floor or the ceiling can be made of (the scene's
/// wall materials; also the values of `floor_material`, `ceiling_material`
/// and `implied_wall_material` in the configuration).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SurfaceMaterial {
    Concrete,
    Brick,
    Drywall,
    Glass,
    Wood,
}

impl SurfaceMaterial {
    pub fn itu(self) -> ItuMaterial {
        match self {
            Self::Concrete => ItuMaterial::Concrete,
            Self::Brick => ItuMaterial::Brick,
            Self::Drywall => ItuMaterial::Plasterboard,
            Self::Glass => ItuMaterial::Glass,
            Self::Wood => ItuMaterial::Wood,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Concrete => "concrete",
            Self::Brick => "brick",
            Self::Drywall => "drywall",
            Self::Glass => "glass",
            Self::Wood => "wood",
        }
    }
}

impl From<WallMaterial> for SurfaceMaterial {
    fn from(m: WallMaterial) -> Self {
        match m {
            WallMaterial::Concrete => Self::Concrete,
            WallMaterial::Brick => Self::Brick,
            WallMaterial::Drywall => Self::Drywall,
            WallMaterial::Glass => Self::Glass,
            WallMaterial::Wood => Self::Wood,
        }
    }
}

/// The ITU material behind an object's surface; `None` for fabric (§5.2).
pub fn object_itu(m: ObjectMaterial) -> Option<ItuMaterial> {
    match m {
        ObjectMaterial::Metal => Some(ItuMaterial::Metal),
        ObjectMaterial::Wood => Some(ItuMaterial::Wood),
        ObjectMaterial::Glass => Some(ItuMaterial::Glass),
        ObjectMaterial::Fabric => None,
    }
}

/// Scatter `rho` of an object material (§5.2, SIM).
pub fn object_rho(m: ObjectMaterial) -> f64 {
    match m {
        ObjectMaterial::Metal => 1.0,
        ObjectMaterial::Wood => 0.3,
        ObjectMaterial::Glass => 0.2,
        ObjectMaterial::Fabric => 0.1,
    }
}

/// Blockage `L_db` per crossing of an object material (§5.2, SIM).
pub fn object_loss_db(m: ObjectMaterial) -> f64 {
    match m {
        ObjectMaterial::Metal => 20.0,
        ObjectMaterial::Wood => 4.0,
        ObjectMaterial::Glass => 2.0,
        ObjectMaterial::Fabric => 3.0,
    }
}

/// Complex relative permittivity
/// `eta = (er, -(sigma / ((TWO_PI * carrier_hz) * EPS0)))` (§5.1).
pub fn eta(m: ItuMaterial, carrier_hz: f64) -> Cx {
    let (a, b, c, d) = m.params();
    let f_ghz = carrier_hz / 1e9;
    let er = a * f_ghz.powf(b);
    let sigma = c * f_ghz.powf(d);
    cx(er, -(sigma / ((TWO_PI * carrier_hz) * EPS0)))
}

/// Polarisation of a reflection: TE for walls and vertical object faces,
/// TM for the floor, the ceiling and an object's top (§5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Pol {
    Te,
    Tm,
}

/// `s = csqrt(csub(eta, (sin2, 0)))`, `sin2 = 1.0 - cos_t*cos_t`.
fn s_of(eta: Cx, cos_t: f64) -> Cx {
    let sin2 = 1.0 - cos_t * cos_t;
    csqrt(csub(eta, cx(sin2, 0.0)))
}

fn g_of(eta: Cx, cos_t: f64, s: Cx, pol: Pol) -> Cx {
    match pol {
        Pol::Te => cdiv(csub(cx(cos_t, 0.0), s), cadd(cx(cos_t, 0.0), s)),
        Pol::Tm => cdiv(csub(cscale(eta, cos_t), s), cadd(cscale(eta, cos_t), s)),
    }
}

/// Single-interface Fresnel coefficient `G_TE` or `G_TM` (§5.1).
pub fn interface_g(eta: Cx, cos_t: f64, pol: Pol) -> Cx {
    g_of(eta, cos_t, s_of(eta, cos_t), pol)
}

/// Every coefficient of a slab at one angle (§5.1).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Slab {
    /// Single-interface coefficient `G` of the slab's face.
    pub g: Cx,
    /// Slab reflection, referenced to the incident face: a bounce is
    /// placed on that face (§5.1, §6.1), so walls, floor and ceiling all
    /// use it unchanged.
    pub r: Cx,
    /// Slab transmission, including `e^{+jq0}` (path lengths count the
    /// slab as free space; a crossing is tested on the centreline).
    pub t: Cx,
}

/// A slab of thickness `t_m` at `cos_t`, for `lambda = C / carrier_hz` (§5.1).
pub fn slab(eta: Cx, cos_t: f64, t_m: f64, lambda: f64, pol: Pol) -> Slab {
    let s = s_of(eta, cos_t);
    let g = g_of(eta, cos_t, s, pol);
    let k0t = TWO_PI * t_m / lambda;
    let q = cscale(s, k0t);
    let q0 = k0t * cos_t;
    let one = cx(1.0, 0.0);
    let e2 = cexp(cmul(cx(0.0, -2.0), q));
    let den = csub(one, cmul(cmul(g, g), e2));
    let r = cdiv(cmul(g, csub(one, e2)), den);
    let t = cdiv(
        cmul(
            csub(one, cmul(g, g)),
            cexp(cmul(cx(0.0, -1.0), csub(q, cx(q0, 0.0)))),
        ),
        den,
    );
    Slab { g, r, t }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::numerics::{cabs2, carrier_hz, C, PI};

    fn close(a: Cx, b: Cx, tol: f64) -> bool {
        (a.re - b.re).abs() <= tol && (a.im - b.im).abs() <= tol
    }

    const ALL: [ItuMaterial; 6] = [
        ItuMaterial::Concrete,
        ItuMaterial::Brick,
        ItuMaterial::Plasterboard,
        ItuMaterial::Glass,
        ItuMaterial::Wood,
        ItuMaterial::Metal,
    ];

    /// §15: at normal incidence G_TE = (1 - sqrt(eta)) / (1 + sqrt(eta)), G_TM = -G_TE.
    #[test]
    fn fresnel_at_normal_incidence() {
        let f = carrier_hz(6);
        for m in ALL {
            let e = eta(m, f);
            let se = csqrt(e);
            let want = cdiv(csub(cx(1.0, 0.0), se), cadd(cx(1.0, 0.0), se));
            let te = interface_g(e, 1.0, Pol::Te);
            let tm = interface_g(e, 1.0, Pol::Tm);
            assert!(close(te, want, 1e-12), "{m:?}: {te:?} vs {want:?}");
            assert!(close(tm, cx(-want.re, -want.im), 1e-12), "{m:?}: {tm:?}");
        }
    }

    /// §15: a lossless slab conserves energy, |R|^2 + |T|^2 = 1.
    #[test]
    fn lossless_slab_conserves_energy() {
        let lambda = C / carrier_hz(6);
        for er in [2.73, 5.24, 6.31] {
            for t in [0.02, 0.1, 0.23, 0.5] {
                for deg in [0.0, 10.0, 30.0, 60.0, 85.0] {
                    let cos_t = ((deg * PI) / 180.0f64).cos();
                    for pol in [Pol::Te, Pol::Tm] {
                        let s = slab(cx(er, 0.0), cos_t, t, lambda, pol);
                        let e = cabs2(s.r) + cabs2(s.t);
                        assert!(
                            (e - 1.0).abs() < 1e-12,
                            "er {er} t {t} {deg} deg {pol:?}: {e}"
                        );
                    }
                }
            }
        }
    }

    /// §15: a lossless half-wave slab at normal incidence has R = 0.
    #[test]
    fn half_wave_slab_is_transparent() {
        let lambda = C / carrier_hz(6);
        for er in [2.73, 3.91, 5.24] {
            let t = lambda / (2.0 * f64::sqrt(er));
            for pol in [Pol::Te, Pol::Tm] {
                let s = slab(cx(er, 0.0), 1.0, t, lambda, pol);
                assert!(
                    cabs2(s.r).sqrt() < 1e-12,
                    "er {er}: |R| = {}",
                    cabs2(s.r).sqrt()
                );
                assert!((cabs2(s.t) - 1.0).abs() < 1e-12);
            }
        }
    }

    /// §15: eta = (1, 0) is free space: R = 0 and T = 1 at every angle.
    #[test]
    fn free_space_slab_is_invisible() {
        let lambda = C / carrier_hz(1);
        for deg in 0..90 {
            let cos_t = ((f64::from(deg) * PI) / 180.0).cos();
            for t in [0.1, 0.23] {
                for pol in [Pol::Te, Pol::Tm] {
                    let s = slab(cx(1.0, 0.0), cos_t, t, lambda, pol);
                    assert!(cabs2(s.r).sqrt() < 1e-12, "{deg} deg: R = {:?}", s.r);
                    assert!(close(s.t, cx(1.0, 0.0), 1e-12), "{deg} deg: T = {:?}", s.t);
                }
            }
        }
    }

    #[test]
    fn concrete_transmits_less_than_drywall() {
        let f = carrier_hz(6);
        let lambda = C / f;
        let t_conc = slab(eta(ItuMaterial::Concrete, f), 1.0, 0.2, lambda, Pol::Te).t;
        let t_dry = slab(eta(ItuMaterial::Plasterboard, f), 1.0, 0.2, lambda, Pol::Te).t;
        assert!(cabs2(t_conc) < cabs2(t_dry));
        let metal = interface_g(eta(ItuMaterial::Metal, f), 0.7, Pol::Te);
        assert!(
            cabs2(metal) > 0.999,
            "metal is a near-perfect mirror: {metal:?}"
        );
    }

    #[test]
    fn eta_is_the_table_formula() {
        let f = 2437e6;
        let e = eta(ItuMaterial::Concrete, f);
        assert_eq!(e.re, 5.24);
        let sigma = 0.0462 * (f / 1e9).powf(0.7822);
        assert_eq!(e.im, -(sigma / ((TWO_PI * f) * EPS0)));
    }
}
