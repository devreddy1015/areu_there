//! CPython semantics of `saans_locate::pynum` (spec/locate.md §1-A).
//!
//! Every expected bit pattern below was printed by CPython 3.14.8.
//! `hypot_matches_a_cpython_dump` re-checks `py_hypot` against a file of
//! `(a, b, math.hypot(a, b))` triples written by CPython 3.14 (little-endian
//! f64); it is ignored unless `SAANS_HYPOT_DUMP` names such a file.

use saans_locate::pynum::*;
use saans_sim::rng::Rng;
use sha2::{Digest, Sha256};

fn h(bits: u64) -> f64 {
    f64::from_bits(bits)
}

#[test]
fn hypot_matches_cpython_on_recorded_pairs() {
    let cases: &[(u64, u64, u64)] = &[
        // CPython differs from glibc hypot on these (glibc is one ulp off).
        (0x3ff99103dfe02c80, 0xbfec07518a7fbc20, 0x3ffd27d06df34f25),
        (0xc0165cee7efa7a4b, 0xc015da328d4c9864, 0x401f445de4cd4742),
        (0xbffe7f29e86c2780, 0xc0086443405cc3b8, 0x400cc404ba192167),
        (0xc0226b0bdafe4061, 0x40169579c741ab24, 0x40259aa4f4a7b976),
        // CPython differs from sqrt(a*a + b*b) on these.
        (0xc00c2fcf590aedec, 0xc01bee9be20d407a, 0x401f49419bccdfe0),
        (0x40230cda2f1afc18, 0xc02222fe4c37780d, 0x402a4d9405290c4c),
        (0xc00ea495a0cb7b98, 0x40194a4490d1ed48, 0x401d91b5f1f2b311),
        (0xc01478e7492e7382, 0x3ff7d0ca54028d08, 0x4015520b55efafc3),
        // Subnormals (the DBL_MIN rescaling branch and its neighbour).
        (0x0000000000000001, 0x0000000000000001, 0x0000000000000001),
        (0x000012688b70e62b, 0x00003739a252b281, 0x00003a365ff2ea11),
        (0x0008000000000000, 0x0004000000000000, 0x0008f1bbcdcbfa54),
        // Huge, an overflow to inf, a 2^-27 ratio, and 3-4-5.
        (0x7fefffffffffffff, 0x7fefffffffffffff, 0x7ff0000000000000),
        (0x7fe1ccf385ebc8a0, 0x7fe1ccf385ebc8a0, 0x7fe92c80954c51f5),
        (0x3ff0000000000000, 0x3e40000000000000, 0x3ff0000000000000),
        (0x4008000000000000, 0x4010000000000000, 0x4014000000000000),
    ];
    for &(a, b, want) in cases {
        let got = py_hypot(h(a), h(b));
        assert_eq!(
            got.to_bits(),
            want,
            "hypot({:e}, {:e}) = {got:e}",
            h(a),
            h(b)
        );
        // math.hypot is symmetric and sign-blind.
        assert_eq!(py_hypot(-h(b), h(a)).to_bits(), want);
    }
}

#[test]
fn hypot_special_values() {
    assert_eq!(py_hypot(f64::INFINITY, f64::NAN), f64::INFINITY);
    assert_eq!(py_hypot(f64::NAN, f64::NEG_INFINITY), f64::INFINITY);
    assert!(py_hypot(f64::NAN, 1.0).is_nan());
    assert!(py_hypot(0.0, f64::NAN).is_nan());
    let z = py_hypot(-0.0, -0.0);
    assert_eq!(z.to_bits(), 0.0f64.to_bits(), "hypot(-0, -0) is +0");
    assert_eq!(py_hypot(-2.5, 0.0), 2.5);
    assert_eq!(py_hypot(0.0, -7.0), 7.0);
}

#[test]
fn hypot_over_the_spec_rng_stream_has_the_recorded_sha256() {
    // spec/locate.md §12: 10^6 pairs from Rng.stream(0, 320000), each
    // `range(-10.0, 10.0)` twice; SHA-256 of the little-endian results.
    // The digest was computed by CPython 3.14.8 with wifisense.roomsim.rng.
    let mut rng = Rng::stream(0, 320000);
    let mut sha = Sha256::new();
    for _ in 0..1_000_000 {
        let a = rng.range(-10.0, 10.0);
        let b = rng.range(-10.0, 10.0);
        sha.update(py_hypot(a, b).to_le_bytes());
    }
    let hex: String = sha.finalize().iter().map(|b| format!("{b:02x}")).collect();
    assert_eq!(
        hex,
        "faec91f3f3c26e33eed9cc8a9e09a2242df32313f00fb4373623b9d162d72046"
    );
}

#[test]
#[ignore = "needs SAANS_HYPOT_DUMP: (a, b, hypot) f64 triples written by CPython 3.14"]
fn hypot_matches_a_cpython_dump() {
    let path = std::env::var("SAANS_HYPOT_DUMP").expect("SAANS_HYPOT_DUMP");
    let bytes = std::fs::read(&path).expect("read dump");
    assert_eq!(bytes.len() % 24, 0);
    let word = |i: usize| f64::from_le_bytes(bytes[i * 8..i * 8 + 8].try_into().unwrap());
    let n = bytes.len() / 24;
    let mut mismatches = 0usize;
    for k in 0..n {
        let (a, b, want) = (word(3 * k), word(3 * k + 1), word(3 * k + 2));
        let got = py_hypot(a, b);
        let same = got.to_bits() == want.to_bits() || (got.is_nan() && want.is_nan());
        if !same {
            mismatches += 1;
            if mismatches <= 10 {
                eprintln!("hypot({a:e}, {b:e}) = {got:e}, CPython {want:e}");
            }
        }
    }
    println!("py_hypot: {n} pairs checked, {mismatches} mismatches");
    assert_eq!(mismatches, 0);
}

#[test]
fn two_argument_max_and_min_keep_the_first_unless_strictly_beaten() {
    assert!(pymax(f64::NAN, 1.0).is_nan()); // max(nan, 1.0) is nan
    assert_eq!(pymax(1.0, f64::NAN), 1.0); // max(1.0, nan) is 1.0
    assert!(pymin(f64::NAN, 1.0).is_nan());
    assert_eq!(pymin(1.0, f64::NAN), 1.0);
    assert_eq!(pymax(0.0, -0.0).to_bits(), 0.0f64.to_bits());
    assert_eq!(pymax(-0.0, 0.0).to_bits(), (-0.0f64).to_bits());
    assert_eq!(pymax_iter([1.0, f64::NAN, 3.0, 3.0]), Some(3.0));
    assert!(pymax_iter([f64::NAN, 5.0]).unwrap().is_nan());
    assert_eq!(pymin_iter([2.0, f64::NAN, -1.0]), Some(-1.0));
    assert_eq!(pymax_iter(std::iter::empty()), None);
    assert_eq!(argmax_first(4, |i| [1.0, 3.0, 3.0, 2.0][i]), Some(1));
}

#[test]
fn float_modulo_takes_the_sign_of_the_divisor() {
    assert_eq!(py_mod(5.5, 2.0).unwrap(), 1.5);
    assert_eq!(py_mod(-5.5, 2.0).unwrap(), 0.5);
    assert_eq!(py_mod(5.5, -2.0).unwrap(), -0.5);
    assert_eq!(py_mod(-5.5, -2.0).unwrap(), -1.5);
    assert_eq!(py_mod(-4.0, 2.0).unwrap().to_bits(), 0.0f64.to_bits());
    assert_eq!(py_mod(4.0, -2.0).unwrap().to_bits(), (-0.0f64).to_bits());
    // A tiny negative dividend rounds up to the divisor itself in CPython.
    assert_eq!(py_mod(-1e-20, 27.2).unwrap(), 27.2);
    assert!(py_mod(f64::INFINITY, 2.0).unwrap().is_nan());
    assert!(matches!(
        py_mod(1.0, 0.0),
        Err(LocateError::ZeroDivisionError(_))
    ));
    assert!(matches!(
        py_div(1.0, -0.0),
        Err(LocateError::ZeroDivisionError(_))
    ));
    assert_eq!(py_div(1.0, 4.0).unwrap(), 0.25);
}

#[test]
fn int_round_and_ceil_raise_like_cpython() {
    assert_eq!(py_int(2.9).unwrap(), 2);
    assert_eq!(py_int(-2.9).unwrap(), -2);
    assert!(matches!(py_int(f64::NAN), Err(LocateError::ValueError(_))));
    assert!(matches!(
        py_int(f64::INFINITY),
        Err(LocateError::OverflowError(_))
    ));
    assert!(matches!(
        py_int(f64::NEG_INFINITY),
        Err(LocateError::OverflowError(_))
    ));
    assert_eq!(py_round_int(2.5).unwrap(), 2);
    assert_eq!(py_round_int(3.5).unwrap(), 4);
    assert_eq!(py_round_int(-2.5).unwrap(), -2);
    assert_eq!(py_round_int(0.49999999999999994).unwrap(), 0);
    assert!(matches!(
        py_round_int(f64::NAN),
        Err(LocateError::ValueError(_))
    ));
    assert!(matches!(
        py_round_int(f64::INFINITY),
        Err(LocateError::OverflowError(_))
    ));
    assert_eq!(py_ceil(3.2).unwrap(), 4);
    assert_eq!(py_ceil(-3.2).unwrap(), -3);
    assert!(matches!(py_ceil(f64::NAN), Err(LocateError::ValueError(_))));
    assert_eq!(round_half_even(0.5), 0.0);
    assert_eq!(round_half_even(1.5), 2.0);
}

#[test]
fn builtin_sum_is_neumaier() {
    assert_eq!(psum([1e16, 1.0, -1e16]), 1.0);
    assert_eq!(psum([-0.0]).to_bits(), 0.0f64.to_bits()); // sum([-0.0]) is 0.0
    assert_eq!(psum(std::iter::empty()), 0.0);
}

#[test]
fn sorting_is_stable_and_by_less_than() {
    let mut v = vec![3.0, -0.0, 1.0, 0.0, -2.0];
    py_sort(&mut v);
    let bits: Vec<u64> = v.iter().map(|x| x.to_bits()).collect();
    let want: Vec<u64> = [-2.0f64, -0.0, 0.0, 1.0, 3.0]
        .iter()
        .map(|x| x.to_bits())
        .collect();
    assert_eq!(bits, want, "-0.0 and 0.0 keep their input order");
    let mut w = vec![1.0, f64::NAN, 0.5];
    py_sort(&mut w);
    assert_eq!(&w[..2], &[0.5, 1.0]);
}

#[test]
fn error_classes_mirror_the_python_hierarchy() {
    let malformed = LocateError::MalformedMatrixError("x".into());
    assert!(malformed.is_linear_algebra_error() && malformed.is_value_error());
    let asym = LocateError::AsymmetricMatrixError("x".into());
    assert!(asym.is_linear_algebra_error() && asym.is_value_error());
    let npd = LocateError::NotPositiveDefiniteError("x".into());
    assert!(npd.is_linear_algebra_error() && !npd.is_value_error());
    let ill = LocateError::IllConditionedError("x".into());
    assert!(ill.is_linear_algebra_error() && !ill.is_value_error());
    let value = LocateError::ValueError("bad".into());
    assert!(!value.is_linear_algebra_error() && value.is_value_error());
    assert_eq!(value.to_string(), "ValueError: bad");
    assert_eq!(LocateError::KeyError("k".into()).python_name(), "KeyError");
}
