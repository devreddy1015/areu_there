//! Numerics of the new code (spec/locate.md §1-B).
//!
//! Every formula of §4.3-§8 is evaluated exactly as the spec writes it,
//! with spec/roomsim.md §1's rules: IEEE-754 binary64, left-associative
//! binary operators, parentheses first, one rounding per operator, no fused
//! multiply-add (Rust never contracts `a * b + c`), `x^2` as `x * x`,
//! `10^y` as `10f64.powf(y)`, a product `a*b*c` as `(a*b)*c`, and **never
//! `hypot`**: a length is [`len2`] or [`len3`]. Every `Σ` is a
//! [`Neumaier`] accumulation in the stated order. Every comparison that
//! decides something is written so that NaN fails closed.
//!
//! The ported modules (`geometry`, `invert`, `measure`, `track`) follow
//! §1-A instead (`pynum`); nothing here is used to reproduce CPython.

// Index loops mirror the spec's subscripts (`c_ij`, ascending `k`).
#![allow(clippy::needless_range_loop)]

/// The language's pi.
pub const PI: f64 = std::f64::consts::PI;
/// `4.0 * PI` (spec/roomsim.md §1).
pub const FOUR_PI: f64 = 4.0 * PI;
/// Speed of light, m/s (spec/roomsim.md §1).
pub const C: f64 = 299792458.0;

/// `K95 = sqrt(-2.0 * ln(0.05))` (= 2.4477468306808166), computed, never
/// typed (§1): the radius of the circle holding 95% of a unit circular
/// Gaussian.
pub fn k95() -> f64 {
    (-2.0 * 0.05f64.ln()).sqrt()
}

/// `CHI2_2_99 = -2.0 * ln(0.01)` (= 9.210340371976182), computed (§1): the
/// 99% quantile of chi-squared with two degrees of freedom.
pub fn chi2_2_99() -> f64 {
    -2.0 * 0.01f64.ln()
}

/// `LN10 = ln(10.0)` (§1).
pub fn ln10() -> f64 {
    10.0f64.ln()
}

/// `LAMBDA = C / carrier_hz` (spec/roomsim.md §1). The caller has checked
/// `carrier_hz > 0` (the guard; [`crate::phone::Config::validate`]).
pub fn wavelength(carrier_hz: f64) -> f64 {
    C / carrier_hz
}

/// Neumaier compensated summation, exactly as spec/roomsim.md §1 defines
/// it (`hi = lo = 0`; `add(x)`: `t = hi + x`; if `|hi| >= |x|` then
/// `lo += (hi - t) + x` else `lo += (x - t) + hi`; `hi = t`; `value()`:
/// `hi + lo` if `lo != 0` and `lo` is finite, else `hi`). A copy of
/// `saans_roomsim::numerics::Neumaier`'s logic, so this crate does not
/// need `saans-roomsim` outside its `roomsim` feature.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Neumaier {
    hi: f64,
    lo: f64,
}

impl Neumaier {
    /// An empty sum.
    pub fn new() -> Self {
        Self { hi: 0.0, lo: 0.0 }
    }

    /// Add one term.
    pub fn add(&mut self, x: f64) {
        let t = self.hi + x;
        if self.hi.abs() >= x.abs() {
            self.lo += (self.hi - t) + x;
        } else {
            self.lo += (x - t) + self.hi;
        }
        self.hi = t;
    }

    /// `hi + lo` if `lo != 0` and `lo` is finite, else `hi`.
    pub fn value(&self) -> f64 {
        if self.lo != 0.0 && self.lo.is_finite() {
            self.hi + self.lo
        } else {
            self.hi
        }
    }

    /// The Neumaier sum of `xs`, in order.
    pub fn sum<I: IntoIterator<Item = f64>>(xs: I) -> f64 {
        let mut n = Self::new();
        for x in xs {
            n.add(x);
        }
        n.value()
    }
}

/// Python's two-argument `max(a, b)`: `b if b > a else a` (so a NaN `a`
/// is returned as is, and a NaN `b` never wins). Never `f64::max`.
pub fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's two-argument `min(a, b)`: `b if b < a else a`. Never `f64::min`.
pub fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `clamp(x, lo, hi) = min(max(x, lo), hi)` (spec/roomsim.md §1).
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    pymin(pymax(x, lo), hi)
}

/// Round half to even (Python `round`, `f64::round_ties_even`).
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// `10^y` as `pow(10.0, y)`.
pub fn pow10(y: f64) -> f64 {
    10f64.powf(y)
}

/// libm `sin(x)`, always its own call: the argument goes through
/// `black_box` so an optimising build cannot merge a `sin` and a `cos` of
/// the same argument into one `sincos` (spec/roomsim.md §1).
pub fn sin(x: f64) -> f64 {
    std::hint::black_box(x).sin()
}

/// libm `cos(x)`, always its own call (see [`sin`]).
pub fn cos(x: f64) -> f64 {
    std::hint::black_box(x).cos()
}

/// A 2-D length `sqrt(dx*dx + dy*dy)` (never `hypot`).
pub fn len2(dx: f64, dy: f64) -> f64 {
    (dx * dx + dy * dy).sqrt()
}

/// A 3-D length `sqrt((dx*dx + dy*dy) + dz*dz)` (never `hypot`).
pub fn len3(dx: f64, dy: f64, dz: f64) -> f64 {
    ((dx * dx + dy * dy) + dz * dz).sqrt()
}

/// `x.is_finite() && x >= min`: the publication test of §8, false for NaN.
pub fn finite_at_least(x: f64, min: f64) -> bool {
    x.is_finite() && x >= min
}

// --- Small linear algebra (§1 "Small linear algebra (B)") ---------------

/// A row-major 2x2 matrix.
pub type Mat2 = [[f64; 2]; 2];
/// A row-major 4x4 matrix.
pub type Mat4 = [[f64; 4]; 4];

/// `C = A·B`, row-major: `c_ij = (((a_i0*b_0j) + a_i1*b_1j) + a_i2*b_2j) + …`,
/// ascending `k`, no term skipped (the accumulation starts from the first
/// product, not from `0.0`). `K` must be at least 1.
pub fn mat_mul<const N: usize, const K: usize, const M: usize>(
    a: &[[f64; K]; N],
    b: &[[f64; M]; K],
) -> [[f64; M]; N] {
    let mut c = [[0.0; M]; N];
    for i in 0..N {
        for j in 0..M {
            let mut acc = a[i][0] * b[0][j];
            for k in 1..K {
                acc += a[i][k] * b[k][j];
            }
            c[i][j] = acc;
        }
    }
    c
}

/// `A·v` with the same accumulation as [`mat_mul`] (`v` a column).
pub fn mat_vec<const N: usize, const K: usize>(a: &[[f64; K]; N], v: &[f64; K]) -> [f64; N] {
    let mut c = [0.0; N];
    for i in 0..N {
        let mut acc = a[i][0] * v[0];
        for k in 1..K {
            acc += a[i][k] * v[k];
        }
        c[i] = acc;
    }
    c
}

/// `Aᵀ`.
pub fn transpose<const N: usize, const M: usize>(a: &[[f64; M]; N]) -> [[f64; N]; M] {
    let mut t = [[0.0; N]; M];
    for i in 0..N {
        for j in 0..M {
            t[j][i] = a[i][j];
        }
    }
    t
}

/// Element-wise `A + B`.
pub fn mat_add<const N: usize, const M: usize>(
    a: &[[f64; M]; N],
    b: &[[f64; M]; N],
) -> [[f64; M]; N] {
    let mut c = [[0.0; M]; N];
    for i in 0..N {
        for j in 0..M {
            c[i][j] = a[i][j] + b[i][j];
        }
    }
    c
}

/// Element-wise `A - B`.
pub fn mat_sub<const N: usize, const M: usize>(
    a: &[[f64; M]; N],
    b: &[[f64; M]; N],
) -> [[f64; M]; N] {
    let mut c = [[0.0; M]; N];
    for i in 0..N {
        for j in 0..M {
            c[i][j] = a[i][j] - b[i][j];
        }
    }
    c
}

/// Each entry `s * a_ij`.
pub fn mat_scale<const N: usize, const M: usize>(s: f64, a: &[[f64; M]; N]) -> [[f64; M]; N] {
    let mut c = [[0.0; M]; N];
    for i in 0..N {
        for j in 0..M {
            c[i][j] = s * a[i][j];
        }
    }
    c
}

/// The `N`x`N` identity.
pub fn identity<const N: usize>() -> [[f64; N]; N] {
    let mut c = [[0.0; N]; N];
    for (i, row) in c.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    c
}

/// `sym(P)`: `p_ij` and `p_ji` replaced by `0.5*(p_ij + p_ji)` for `j < i`.
pub fn sym<const N: usize>(p: &[[f64; N]; N]) -> [[f64; N]; N] {
    let mut s = *p;
    for i in 0..N {
        for j in 0..i {
            let v = 0.5 * (p[i][j] + p[j][i]);
            s[i][j] = v;
            s[j][i] = v;
        }
    }
    s
}

/// `det = a*d - b*c` of `[[a, b], [c, d]]`.
pub fn det2(m: &Mat2) -> f64 {
    m[0][0] * m[1][1] - m[0][1] * m[1][0]
}

/// The 2x2 inverse `[[d/det, (-b)/det], [(-c)/det, a/det]]` with the
/// caller's `det` (normally [`det2`]). Used only after the caller's guard
/// on `det` (§1): this function itself guards nothing.
pub fn inv2(m: &Mat2, det: f64) -> Mat2 {
    [
        [m[1][1] / det, (-m[0][1]) / det],
        [(-m[1][0]) / det, m[0][0] / det],
    ]
}
