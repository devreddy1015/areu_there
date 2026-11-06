//! Numerical conventions of spec/roomsim.md §1.
//!
//! Every formula in this crate is evaluated exactly as the spec writes it:
//! left-associative binary operators, one rounding per operator, no fused
//! multiply-add (Rust never contracts `a * b + c`), `x^2` as `x * x`,
//! `10^y` as `10f64.powf(y)`, and never `hypot`. The complex helpers below
//! are plain functions rather than operator overloads so that the order of
//! every operation is visible at the call site.

/// The language's pi.
pub const PI: f64 = std::f64::consts::PI;
/// `2.0 * PI`.
pub const TWO_PI: f64 = 2.0 * PI;
/// `4.0 * PI`.
pub const FOUR_PI: f64 = 4.0 * PI;
/// `FOUR_PI * sqrt(FOUR_PI)`. `sqrt` is not a `const fn`, so the value is
/// written out; `four_pi_1_5_is_computed_as_written` checks it bit for bit.
pub const FOUR_PI_1_5: f64 = 44.546623974653656;
/// Speed of light, m/s.
pub const C: f64 = 299792458.0;
/// Vacuum permittivity, F/m.
pub const EPS0: f64 = 8.8541878128e-12;
/// Range clamp for point scatterers, metres (§6.6).
pub const MIN_RANGE_M: f64 = 0.1;

/// A complex number as an explicit `(re, im)` pair (§1).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct Cx {
    pub re: f64,
    pub im: f64,
}

/// `(re, im)`.
pub const fn cx(re: f64, im: f64) -> Cx {
    Cx { re, im }
}

/// `(a.re + b.re, a.im + b.im)`.
pub fn cadd(a: Cx, b: Cx) -> Cx {
    cx(a.re + b.re, a.im + b.im)
}

/// `(a.re - b.re, a.im - b.im)`.
pub fn csub(a: Cx, b: Cx) -> Cx {
    cx(a.re - b.re, a.im - b.im)
}

/// `(a.re*b.re - a.im*b.im, a.re*b.im + a.im*b.re)`.
pub fn cmul(a: Cx, b: Cx) -> Cx {
    cx(a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re)
}

/// `(a.re*s, a.im*s)`.
pub fn cscale(a: Cx, s: f64) -> Cx {
    cx(a.re * s, a.im * s)
}

/// `d = b.re*b.re + b.im*b.im`; `((a.re*b.re + a.im*b.im) / d, (a.im*b.re - a.re*b.im) / d)`.
pub fn cdiv(a: Cx, b: Cx) -> Cx {
    let d = b.re * b.re + b.im * b.im;
    cx(
        (a.re * b.re + a.im * b.im) / d,
        (a.im * b.re - a.re * b.im) / d,
    )
}

/// Principal square root, written out as §1 gives it.
pub fn csqrt(z: Cx) -> Cx {
    if z.re == 0.0 && z.im == 0.0 {
        return cx(0.0, 0.0);
    }
    let m = (z.re * z.re + z.im * z.im).sqrt();
    if z.re >= 0.0 {
        let t = ((m + z.re) / 2.0).sqrt();
        cx(t, z.im / (2.0 * t))
    } else {
        let t = ((m - z.re) / 2.0).sqrt();
        cx(z.im.abs() / (2.0 * t), t.copysign(z.im))
    }
}

/// libm `sin(x)`, always its own call (§1).
///
/// An optimising build lets LLVM merge a `sin` and a `cos` of the same
/// argument into one `sincos` call, and glibc's `sincos` differs from its
/// `sin` and `cos` in the last bit for about one argument in 1500. That
/// made release builds disagree with debug builds and with the Python
/// reference (`math.sin`, `math.cos`). Passing the argument through
/// `black_box` gives each call its own operand, so no merge can happen.
pub fn sin(x: f64) -> f64 {
    std::hint::black_box(x).sin()
}

/// libm `cos(x)`, always its own call (see [`sin`]).
pub fn cos(x: f64) -> f64 {
    std::hint::black_box(x).cos()
}

/// `(cos(th), sin(th))`.
pub fn cexpj(th: f64) -> Cx {
    cx(cos(th), sin(th))
}

/// `cscale(cexpj(z.im), exp(z.re))`.
pub fn cexp(z: Cx) -> Cx {
    cscale(cexpj(z.im), z.re.exp())
}

/// `re*re + im*im`.
pub fn cabs2(a: Cx) -> f64 {
    a.re * a.re + a.im * a.im
}

/// Neumaier compensated summation, exactly as §1 defines it.
#[derive(Debug, Clone, Copy, Default)]
pub struct Neumaier {
    hi: f64,
    lo: f64,
}

impl Neumaier {
    pub fn new() -> Self {
        Self { hi: 0.0, lo: 0.0 }
    }

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

/// Python's two-argument `max(a, b)`: `b` only if `b > a`.
pub fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's two-argument `min(a, b)`: `b` only if `b < a`.
pub fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// `clamp(x, lo, hi) = min(max(x, lo), hi)` (§1).
pub fn clamp(x: f64, lo: f64, hi: f64) -> f64 {
    pymin(pymax(x, lo), hi)
}

/// `1` if `d > 0`, `-1` if `d < 0`, else `0` (`-0.0` is 0; not `f64::signum`).
pub fn sign(d: f64) -> i32 {
    if d > 0.0 {
        1
    } else if d < 0.0 {
        -1
    } else {
        0
    }
}

/// Round half to even (Python `round`).
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

/// `10^y` as `pow(10.0, y)`.
pub fn pow10(y: f64) -> f64 {
    10f64.powf(y)
}

/// Phase of a path of length `l` at frequency `f`: `((-TWO_PI) * f) * L / C`.
pub fn theta(f: f64, l: f64) -> f64 {
    ((-TWO_PI) * f) * l / C
}

/// `carrier_hz = (2412 + 5*(channel-1)) * 1e6` (§3.2).
pub fn carrier_hz(channel: u32) -> f64 {
    f64::from(2412 + 5 * (channel - 1)) * 1e6
}

/// Bin `k` of `[-32, 31]` to its index in an ascending-k array of 64.
pub fn bin_index(k: i32) -> usize {
    (k + 32) as usize
}

/// The 64 bins in ascending order.
pub fn bins() -> impl Iterator<Item = i32> {
    -32..32
}

/// `f_k = carrier_hz + float(k) * spacing_hz` (§4).
pub fn bin_freq(carrier_hz: f64, spacing_hz: f64, k: i32) -> f64 {
    carrier_hz + f64::from(k) * spacing_hz
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn four_pi_1_5_is_computed_as_written() {
        assert_eq!(FOUR_PI_1_5.to_bits(), (FOUR_PI * FOUR_PI.sqrt()).to_bits());
        // The §1 table, as printed (repr of the double).
        assert_eq!(format!("{TWO_PI:?}"), "6.283185307179586");
        assert_eq!(format!("{FOUR_PI:?}"), "12.566370614359172");
        assert_eq!(format!("{FOUR_PI_1_5:?}"), "44.546623974653656");
        assert_eq!(C / carrier_hz(6), 0.1230170118998769);
    }

    #[test]
    fn neumaier_recovers_what_a_plain_sum_loses() {
        let xs = [1.0, 1e100, 1.0, -1e100];
        assert_eq!(xs.iter().sum::<f64>(), 0.0);
        assert_eq!(Neumaier::sum(xs), 2.0);
        assert_eq!(Neumaier::sum([]), 0.0);
        assert_eq!(Neumaier::sum([0.1]), 0.1);
    }

    #[test]
    fn csqrt_branches() {
        let z = csqrt(cx(-4.0, 0.0));
        assert_eq!(z, cx(0.0, 2.0));
        let z = csqrt(cx(-4.0, -0.0));
        assert_eq!(z.re, 0.0);
        assert_eq!(z.im, -2.0);
        let z = csqrt(cx(3.0, 4.0));
        assert!((z.re - 2.0).abs() < 1e-15 && (z.im - 1.0).abs() < 1e-15);
        assert_eq!(csqrt(cx(0.0, 0.0)), cx(0.0, 0.0));
        let w = csqrt(cx(-3.0, -4.0));
        let back = cmul(w, w);
        assert!((back.re + 3.0).abs() < 1e-14 && (back.im + 4.0).abs() < 1e-14);
    }

    #[test]
    fn rounding_and_clamp_and_sign() {
        assert_eq!(round_half_even(2.5), 2.0);
        assert_eq!(round_half_even(-2.5), -2.0);
        assert_eq!(round_half_even(3.5), 4.0);
        assert_eq!(clamp(5.0, 0.0, 1.0), 1.0);
        assert_eq!(clamp(-5.0, 0.0, 1.0), 0.0);
        assert_eq!(sign(-0.0), 0);
        assert_eq!(sign(0.0), 0);
        assert_eq!(sign(1e-300), 1);
        assert_eq!(sign(-1e-300), -1);
        assert_eq!(pymax(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
    }

    #[test]
    fn complex_ops_follow_the_formulas() {
        let a = cx(1.5, -2.0);
        let b = cx(0.25, 3.0);
        assert_eq!(
            cmul(a, b),
            cx(1.5 * 0.25 - (-2.0) * 3.0, 1.5 * 3.0 + (-2.0) * 0.25)
        );
        let q = cdiv(cmul(a, b), b);
        assert!((q.re - a.re).abs() < 1e-15 && (q.im - a.im).abs() < 1e-15);
        let e = cexp(cx(0.0, PI));
        assert!((e.re + 1.0).abs() < 1e-15 && e.im.abs() < 1e-15);
    }
}
