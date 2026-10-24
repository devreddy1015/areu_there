//! CPython 3.14 numerics the ported modules rely on (spec/locate.md §1-A).
//!
//! The Python modules in `wifisense/localize` are the specification line
//! by line, so wherever CPython's behaviour differs from the obvious Rust
//! spelling, the Rust spelling lives here:
//!
//! - builtin `sum()` over floats is Neumaier-compensated: [`psum`] (from
//!   `saans-dsp`, golden-matched). Only the builtin is; every Python loop,
//!   `+=` or `map(add, ...)` stays a plain left-to-right accumulation at its
//!   call site;
//! - `math.hypot(a, b)` is CPython's `vector_norm`, not libm: [`py_hypot`];
//! - two-argument `max`/`min` keep the first argument unless the second
//!   compares strictly greater/smaller: [`pymax`], [`pymin`]; over an
//!   iterable they fold the same way from the first element
//!   ([`pymax_iter`], [`pymin_iter`]); `max(..., key=)` takes the first
//!   maximum ([`argmax_first`]);
//! - float `%` takes the sign of the divisor: [`py_mod`];
//! - `int(x)`, `round(x)` and `math.ceil(x)` raise on NaN and ±inf:
//!   [`py_int`], [`py_round_int`], [`py_ceil`]; `round` is half to even;
//! - float `/` raises `ZeroDivisionError` on a zero divisor: [`py_div`],
//!   used wherever the Python divides without a guard of its own;
//! - `sorted` over floats is a stable sort by `<`: [`py_sort`].
//!
//! Python's exceptions map to the one error enum [`LocateError`], whose
//! variants name them.

// The Python is the specification line by line: index loops mirror it, and
// `!(a > b)` is how a comparison is written so that a NaN fails closed.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

use std::cmp::Ordering;
use std::fmt;

pub use saans_dsp::{argmax_first, psum};

/// The Python exception a ported function raises, with its message.
///
/// `LinearAlgebraError` (the base of the four matrix refusals in
/// `invert.py`) is [`LocateError::is_linear_algebra_error`];
/// `MalformedMatrixError` and `AsymmetricMatrixError` also derive from
/// `ValueError` in Python, which is [`LocateError::is_value_error`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LocateError {
    /// `ValueError`: a bad argument or an input the code refuses.
    ValueError(String),
    /// `KeyError`: an unknown node id or link.
    KeyError(String),
    /// `IndexError`: a pixel index outside the grid.
    IndexError(String),
    /// `OverflowError`: `int()`/`round()`/`ceil()` of an infinity.
    OverflowError(String),
    /// `ZeroDivisionError`: float `/` or `%` by zero.
    ZeroDivisionError(String),
    /// `MemoryError`: a grid too large to allocate (see `PixelGrid::new`).
    MemoryError(String),
    /// `invert.MalformedMatrixError` (a `LinearAlgebraError` and a `ValueError`).
    MalformedMatrixError(String),
    /// `invert.AsymmetricMatrixError` (a `LinearAlgebraError` and a `ValueError`).
    AsymmetricMatrixError(String),
    /// `invert.NotPositiveDefiniteError` (a `LinearAlgebraError`).
    NotPositiveDefiniteError(String),
    /// `invert.IllConditionedError` (a `LinearAlgebraError`).
    IllConditionedError(String),
}

impl LocateError {
    /// The Python exception class name.
    pub fn python_name(&self) -> &'static str {
        match self {
            LocateError::ValueError(_) => "ValueError",
            LocateError::KeyError(_) => "KeyError",
            LocateError::IndexError(_) => "IndexError",
            LocateError::OverflowError(_) => "OverflowError",
            LocateError::ZeroDivisionError(_) => "ZeroDivisionError",
            LocateError::MemoryError(_) => "MemoryError",
            LocateError::MalformedMatrixError(_) => "MalformedMatrixError",
            LocateError::AsymmetricMatrixError(_) => "AsymmetricMatrixError",
            LocateError::NotPositiveDefiniteError(_) => "NotPositiveDefiniteError",
            LocateError::IllConditionedError(_) => "IllConditionedError",
        }
    }

    /// The message.
    pub fn message(&self) -> &str {
        match self {
            LocateError::ValueError(m)
            | LocateError::KeyError(m)
            | LocateError::IndexError(m)
            | LocateError::OverflowError(m)
            | LocateError::ZeroDivisionError(m)
            | LocateError::MemoryError(m)
            | LocateError::MalformedMatrixError(m)
            | LocateError::AsymmetricMatrixError(m)
            | LocateError::NotPositiveDefiniteError(m)
            | LocateError::IllConditionedError(m) => m,
        }
    }

    /// `isinstance(e, LinearAlgebraError)`.
    pub fn is_linear_algebra_error(&self) -> bool {
        matches!(
            self,
            LocateError::MalformedMatrixError(_)
                | LocateError::AsymmetricMatrixError(_)
                | LocateError::NotPositiveDefiniteError(_)
                | LocateError::IllConditionedError(_)
        )
    }

    /// `isinstance(e, ValueError)`.
    pub fn is_value_error(&self) -> bool {
        matches!(
            self,
            LocateError::ValueError(_)
                | LocateError::MalformedMatrixError(_)
                | LocateError::AsymmetricMatrixError(_)
        )
    }
}

impl fmt::Display for LocateError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.python_name(), self.message())
    }
}

impl std::error::Error for LocateError {}

/// Shorthand for the ported modules' results.
pub type Result<T> = std::result::Result<T, LocateError>;

// ---------------------------------------------------------------------------
// math.hypot (CPython 3.14 Modules/mathmodule.c: math_hypot_impl, vector_norm)
// ---------------------------------------------------------------------------

/// `DoubleLength dl_fast_sum(a, b)`: Algorithm 1.1, `|a| >= |b|` assumed.
fn dl_fast_sum(a: f64, b: f64) -> (f64, f64) {
    let x = a + b;
    let y = (a - x) + b;
    (x, y)
}

/// `DoubleLength dl_mul(x, y)`, the default (fma) build: Algorithm 3.5.
fn dl_mul(x: f64, y: f64) -> (f64, f64) {
    let z = x * y;
    let zz = x.mul_add(y, -z);
    (z, zz)
}

/// The exponent `frexp(x, &e)` stores, for a finite, positive, nonzero `x`.
fn frexp_exponent(x: f64) -> i32 {
    let bits = x.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: x = m * 2^-1074 with m the mantissa field, and frexp's
        // exponent is floor(log2 x) + 1.
        let m = bits & ((1u64 << 52) - 1);
        (63 - m.leading_zeros() as i32) - 1074 + 1
    } else {
        biased - 1022
    }
}

/// `ldexp(1.0, k)` for `-1074 <= k <= 1023` (exact: a power of two).
fn ldexp_one(k: i32) -> f64 {
    if k >= -1022 {
        f64::from_bits(((k + 1023) as u64) << 52)
    } else {
        f64::from_bits(1u64 << (k + 1074))
    }
}

/// `vector_norm(2, vec, max, found_nan)`, line for line.
fn vector_norm2(mut vec: [f64; 2], max: f64, found_nan: bool) -> f64 {
    if max.is_infinite() {
        return max;
    }
    if found_nan {
        return f64::NAN;
    }
    // `max == 0.0 || n <= 1`; n is 2 here.
    if max == 0.0 {
        return max;
    }
    let max_e = frexp_exponent(max);
    if max_e < -1023 {
        // When max_e < -1023, ldexp(1.0, -max_e) would overflow.
        for v in vec.iter_mut() {
            *v /= f64::MIN_POSITIVE; // convert subnormals to normals
        }
        return f64::MIN_POSITIVE * vector_norm2(vec, max / f64::MIN_POSITIVE, found_nan);
    }
    let scale = ldexp_one(-max_e);
    let mut csum = 1.0;
    let mut frac1 = 0.0;
    let mut frac2 = 0.0;
    for &v in vec.iter() {
        let x = v * scale; // lossless scaling
        let (pr_hi, pr_lo) = dl_mul(x, x); // lossless squaring
        let (sm_hi, sm_lo) = dl_fast_sum(csum, pr_hi); // lossless addition
        csum = sm_hi;
        frac1 += pr_lo; // lossy addition
        frac2 += sm_lo; // lossy addition
    }
    let mut h = (csum - 1.0 + (frac1 + frac2)).sqrt();
    let (pr_hi, pr_lo) = dl_mul(-h, h);
    let (sm_hi, sm_lo) = dl_fast_sum(csum, pr_hi);
    csum = sm_hi;
    frac1 += pr_lo;
    frac2 += sm_lo;
    let x = csum - 1.0 + (frac1 + frac2);
    h += x / (2.0 * h); // differential correction
    h / scale
}

/// CPython 3.14's `math.hypot(a, b)`, bit for bit.
///
/// `math_hypot_impl` takes `fabs` of each argument, notes any NaN, keeps
/// the running maximum by `x > max`, and calls `vector_norm`: an infinity
/// wins over a NaN, a NaN gives NaN, zeros give `0.0`, a subnormal maximum
/// is rescaled by `DBL_MIN`, and otherwise the sum of squares is formed
/// after power-of-two scaling with an fma-based exact square, compensated
/// summation and a differential correction of the square root. Not libm
/// `hypot`, which differs in the last bit on a fraction of a percent of
/// pairs.
pub fn py_hypot(a: f64, b: f64) -> f64 {
    let coordinates = [a.abs(), b.abs()];
    let mut max = 0.0;
    let mut found_nan = false;
    for &x in coordinates.iter() {
        found_nan |= x.is_nan();
        if x > max {
            max = x;
        }
    }
    vector_norm2(coordinates, max, found_nan)
}

// ---------------------------------------------------------------------------
// max / min / sorted
// ---------------------------------------------------------------------------

/// Python's two-argument `max(a, b)`: `b if b > a else a`.
pub fn pymax(a: f64, b: f64) -> f64 {
    if b > a {
        b
    } else {
        a
    }
}

/// Python's two-argument `min(a, b)`: `b if b < a else a`.
pub fn pymin(a: f64, b: f64) -> f64 {
    if b < a {
        b
    } else {
        a
    }
}

/// Python's `max(iterable)`: folds from the first element, replacing only
/// when an element compares strictly greater (so a NaN anywhere but first
/// is skipped). `None` for an empty iterable (Python raises, or returns its
/// `default=`).
pub fn pymax_iter<I: IntoIterator<Item = f64>>(xs: I) -> Option<f64> {
    let mut it = xs.into_iter();
    let mut best = it.next()?;
    for x in it {
        if x > best {
            best = x;
        }
    }
    Some(best)
}

/// Python's `min(iterable)`, the mirror of [`pymax_iter`].
pub fn pymin_iter<I: IntoIterator<Item = f64>>(xs: I) -> Option<f64> {
    let mut it = xs.into_iter();
    let mut best = it.next()?;
    for x in it {
        if x < best {
            best = x;
        }
    }
    Some(best)
}

/// Python's `list.sort()` over floats: stable, ordered by `<`.
///
/// For NaN-free input this is exactly CPython's result (equal values,
/// including `0.0` and `-0.0`, keep their order). Every sort in the ported
/// modules sorts finite values (or infinities); a NaN, which CPython would
/// place according to its merge order, is put last here so the sort stays
/// a total order.
pub fn py_sort(v: &mut [f64]) {
    v.sort_by(|a, b| match (a.is_nan(), b.is_nan()) {
        (false, false) => a.partial_cmp(b).unwrap_or(Ordering::Equal),
        (true, true) => Ordering::Equal,
        (true, false) => Ordering::Greater,
        (false, true) => Ordering::Less,
    });
}

/// `sorted(values)`.
pub fn py_sorted(values: &[f64]) -> Vec<f64> {
    let mut out = values.to_vec();
    py_sort(&mut out);
    out
}

// ---------------------------------------------------------------------------
// / % int() round() ceil()
// ---------------------------------------------------------------------------

/// Float `a / b`, raising `ZeroDivisionError` on `b == 0` as CPython does.
pub fn py_div(a: f64, b: f64) -> Result<f64> {
    if b == 0.0 {
        return Err(LocateError::ZeroDivisionError(
            "float division by zero".to_string(),
        ));
    }
    Ok(a / b)
}

/// CPython's float `a % b` (`float_rem`): `fmod`, then the sign of the
/// divisor; `ZeroDivisionError` on `b == 0`.
pub fn py_mod(a: f64, b: f64) -> Result<f64> {
    if b == 0.0 {
        return Err(LocateError::ZeroDivisionError(
            "division by zero".to_string(),
        ));
    }
    let mut r = a % b; // fmod
    if r != 0.0 {
        // `if (mod)`; a NaN is truthy in C and `NaN != 0.0` here too.
        if (b < 0.0) != (r < 0.0) {
            r += b;
        }
    } else {
        r = 0.0f64.copysign(b);
    }
    Ok(r)
}

/// `int(x)` for a float: truncation toward zero; `ValueError` on NaN,
/// `OverflowError` on ±inf.
///
/// Python's result is an unbounded integer; this saturates at the `i64`
/// range beyond ±2^63. Every caller in the ported modules only compares
/// or clamps the result (a pixel index, a window size, a sample count), so
/// saturation gives the same decisions.
pub fn py_int(x: f64) -> Result<i64> {
    if x.is_nan() {
        return Err(LocateError::ValueError(
            "cannot convert float NaN to integer".to_string(),
        ));
    }
    if x.is_infinite() {
        return Err(LocateError::OverflowError(
            "cannot convert float infinity to integer".to_string(),
        ));
    }
    Ok(x.trunc() as i64)
}

/// `round(x)` with no `ndigits` (an `int`): half to even, with
/// [`py_int`]'s errors.
pub fn py_round_int(x: f64) -> Result<i64> {
    py_int(x.round_ties_even())
}

/// `math.ceil(x)` (an `int`), with [`py_int`]'s errors.
pub fn py_ceil(x: f64) -> Result<i64> {
    py_int(x.ceil())
}

/// `round(x)` as a float: half to even (`f64::round_ties_even`).
pub fn round_half_even(x: f64) -> f64 {
    x.round_ties_even()
}

// ---------------------------------------------------------------------------
// Message formatting (informational only; not part of any contract)
// ---------------------------------------------------------------------------

/// A float as Python's `repr` would roughly print it (`nan`, `inf`, `1.0`).
pub(crate) fn fmt_py(x: f64) -> String {
    if x.is_nan() {
        "nan".to_string()
    } else if x.is_infinite() {
        if x > 0.0 { "inf" } else { "-inf" }.to_string()
    } else {
        format!("{x:?}")
    }
}

/// A float as Python's `f"{x:.{prec}e}"` prints it (`1.500000e+00`).
pub(crate) fn fmt_e(x: f64, prec: usize) -> String {
    if !x.is_finite() {
        return fmt_py(x);
    }
    let s = format!("{x:.prec$e}");
    match s.split_once('e') {
        Some((mantissa, exp)) => {
            let (sign, digits) = match exp.strip_prefix('-') {
                Some(d) => ('-', d),
                None => ('+', exp),
            };
            format!("{mantissa}e{sign}{digits:0>2}")
        }
        None => s,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn frexp_and_ldexp_helpers() {
        assert_eq!(frexp_exponent(1.0), 1);
        assert_eq!(frexp_exponent(0.75), 0);
        assert_eq!(frexp_exponent(f64::MIN_POSITIVE), -1021);
        assert_eq!(frexp_exponent(f64::from_bits(1)), -1073);
        assert_eq!(frexp_exponent(f64::MAX), 1024);
        assert_eq!(ldexp_one(0), 1.0);
        assert_eq!(ldexp_one(-1024), f64::from_bits(1u64 << 50));
        assert_eq!(ldexp_one(1023), 2f64.powi(1023));
    }

    #[test]
    fn fmt_e_matches_python_style() {
        assert_eq!(fmt_e(1.5, 6), "1.500000e+00");
        assert_eq!(fmt_e(-2.5e-10, 3), "-2.500e-10");
        assert_eq!(fmt_e(f64::NAN, 3), "nan");
    }
}
