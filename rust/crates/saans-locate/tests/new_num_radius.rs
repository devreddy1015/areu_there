//! spec/locate.md §1-B numerics and §8 radius, on the new code
//! (`saans_locate::num`, `saans_locate::radius`). Known values only; no
//! golden file.

use saans_locate::num::{
    chi2_2_99, clamp, det2, inv2, k95, len2, len3, ln10, mat_mul, mat_vec, pymax, pymin,
    round_half_even, sym, transpose, wavelength, Neumaier, C, FOUR_PI,
};
use saans_locate::radius::{plausible, publish, radius95, Outcome, Reason};

#[test]
fn constants_are_computed_to_the_spec_values() {
    // §1: K95 = sqrt(-2.0 * ln(0.05)) (= 2.4477468306808166);
    // CHI2_2_99 = -2.0 * ln(0.01) (= 9.210340371976182).
    assert_eq!(k95(), 2.4477468306808166);
    assert_eq!(chi2_2_99(), 9.210340371976182);
    assert_eq!(ln10(), std::f64::consts::LN_10);
    assert_eq!(FOUR_PI, 12.566370614359172);
    // spec/roomsim.md §1: LAMBDA at channel 6.
    assert_eq!(C, 299792458.0);
    assert_eq!(wavelength(2437e6), 0.1230170118998769);
}

#[test]
fn neumaier_is_roomsim_section_1() {
    assert_eq!(Neumaier::sum([1e100, 1.0, -1e100]), 1.0);
    assert_eq!(Neumaier::sum([0.1, 0.2, 0.3]), 0.6);
    // Plain summation would give 0.6000000000000001 here.
    assert_eq!(0.1 + 0.2 + 0.3, 0.6000000000000001);
    assert_eq!(Neumaier::sum(std::iter::empty()), 0.0);
    // `value()` is `hi` when `lo` is not finite.
    let s = Neumaier::sum([f64::INFINITY, 1.0]);
    assert_eq!(s, f64::INFINITY);
    assert!(Neumaier::sum([f64::NAN, 1.0]).is_nan());
    // Order is part of the definition: the same terms, as given.
    let mut n = Neumaier::new();
    for x in [3.0, 1e-16, -3.0] {
        n.add(x);
    }
    assert_eq!(n.value(), 1e-16);
}

#[test]
fn python_max_min_and_clamp_keep_their_nan_behaviour() {
    // max(a, b) = b if b > a else a.
    assert!(pymax(f64::NAN, 0.0).is_nan());
    assert_eq!(pymax(0.0, f64::NAN), 0.0);
    assert_eq!(pymin(0.0, f64::NAN), 0.0);
    assert!(pymin(f64::NAN, 0.0).is_nan());
    // max(-0.0, 0.0) keeps -0.0 (b > a is false).
    assert!(pymax(-0.0, 0.0).is_sign_negative());
    assert_eq!(clamp(-120.4, -100.0, 0.0), -100.0);
    assert_eq!(clamp(3.0, -100.0, 0.0), 0.0);
    assert_eq!(clamp(-42.0, -100.0, 0.0), -42.0);
}

#[test]
fn round_half_even_and_lengths() {
    assert_eq!(round_half_even(-42.5), -42.0);
    assert_eq!(round_half_even(-43.5), -44.0);
    assert_eq!(round_half_even(0.5), 0.0);
    assert_eq!(round_half_even(1.5), 2.0);
    assert_eq!(len2(3.0, 4.0), 5.0);
    assert_eq!(len3(2.0, 3.0, 6.0), 7.0);
    // Never hypot: the plain formula overflows where hypot would not.
    assert_eq!(len2(1e200, 1e200), f64::INFINITY);
}

#[test]
fn matrix_product_accumulates_in_ascending_k_from_the_first_product() {
    let a = [[1.0, 2.0], [3.0, 4.0]];
    let b = [[5.0, 6.0], [7.0, 8.0]];
    assert_eq!(mat_mul(&a, &b), [[19.0, 22.0], [43.0, 50.0]]);
    // ((a0*b0 + a1*b1) + a2*b2) = 0 here; a0*b0 + (a1*b1 + a2*b2) = 1.
    let row = [[1.0, 1e16, -1e16]];
    let col = [[1.0], [1.0], [1.0]];
    assert_eq!(mat_mul(&row, &col)[0][0], 0.0);
    assert_eq!(mat_vec(&row, &[1.0, 1.0, 1.0])[0], 0.0);
    // No `0.0 +` start: a single negative-zero product stays -0.0.
    let z = mat_mul(&[[-0.0]], &[[1.0]]);
    assert!(z[0][0] == 0.0 && z[0][0].is_sign_negative());
    // A 4x2 by 2x4 product and a transpose.
    let k = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]];
    let kt = transpose(&k);
    assert_eq!(kt, [[1.0, 3.0, 5.0, 7.0], [2.0, 4.0, 6.0, 8.0]]);
    let kk = mat_mul(&k, &kt);
    assert_eq!(kk[3][2], 7.0 * 5.0 + 8.0 * 6.0);
}

#[test]
fn sym_and_the_2x2_inverse() {
    let p = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]];
    let s = sym(&p);
    assert_eq!(s, [[1.0, 3.0, 5.0], [3.0, 5.0, 7.0], [5.0, 7.0, 9.0]]);
    let m = [[4.0, 1.0], [2.0, 3.0]];
    let det = det2(&m);
    assert_eq!(det, 10.0);
    let inv = inv2(&m, det);
    assert_eq!(inv, [[0.3, -0.1], [-0.2, 0.4]]);
}

// --- §8 / §12a: radius95 ------------------------------------------------

#[test]
fn radius95_known_geometry() {
    // §12a: identity -> K95; [[4,0],[0,1]] -> 2*K95; [[2,1],[1,2]] ->
    // K95*sqrt(3); zero matrix -> abstains implausible_radius.
    assert_eq!(radius95(1.0, 0.0, 1.0), k95());
    assert_eq!(radius95(4.0, 0.0, 1.0), 2.0 * k95());
    assert_eq!(radius95(1.0, 0.0, 4.0), 2.0 * k95());
    assert_eq!(radius95(2.0, 1.0, 2.0), k95() * 3.0f64.sqrt());
    assert_eq!(radius95(0.0, 0.0, 0.0), 0.0);
    assert_eq!(
        publish(1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.3, None),
        Outcome::Abstain(Reason::ImplausibleRadius)
    );
}

#[test]
fn radius95_nan_and_negative_fail_closed() {
    // lmax > 0.0 ? lmax : 0.0 -- a NaN gives 0.0.
    assert_eq!(radius95(f64::NAN, 0.0, 1.0), 0.0);
    assert_eq!(radius95(1.0, f64::NAN, 1.0), 0.0);
    assert_eq!(radius95(-1.0, 0.0, -2.0), 0.0);
    assert_eq!(radius95(f64::INFINITY, 0.0, 1.0), f64::INFINITY);
    for cov in [
        (f64::NAN, 0.0, 1.0),
        (f64::INFINITY, 0.0, 1.0),
        (-1.0, 0.0, -1.0),
        (1e-3, 0.0, 1e-3),
    ] {
        let o = publish(2.0, 2.0, cov.0, cov.1, cov.2, 1.0, 0.3, None);
        assert_eq!(o, Outcome::Abstain(Reason::ImplausibleRadius), "{cov:?}");
    }
    assert!(!plausible(f64::NAN, 0.3));
    assert!(!plausible(f64::INFINITY, 0.3));
    assert!(!plausible(0.29, 0.3));
    assert!(plausible(0.3, 0.3));
    // r > 0 whatever the minimum (§8): a raw call with minimum 0 and a
    // zero radius abstains implausible_radius (ADR-008 d5).
    assert!(!plausible(0.0, 0.0));
    assert!(!plausible(-0.0, -1.0));
    assert!(!plausible(-1.0, -10.0));
    assert!(plausible(f64::MIN_POSITIVE, 0.0));
    for (cov, scale) in [
        ((1.0, 0.0, 1.0), 0.0),
        ((0.0, 0.0, 0.0), 1.0),
        ((-1.0, 0.0, -1.0), 1.0),
    ] {
        assert_eq!(
            publish(2.0, 2.0, cov.0, cov.1, cov.2, scale, 0.0, None),
            Outcome::Abstain(Reason::ImplausibleRadius),
            "{cov:?} x {scale}"
        );
    }
    assert_eq!(
        publish(2.0, 2.0, 1.0, 0.0, 1.0, -1.0, -10.0, None),
        Outcome::Abstain(Reason::ImplausibleRadius)
    );
    assert!(publish(2.0, 2.0, 1e-6, 0.0, 1e-6, 1.0, 0.0, None)
        .estimate()
        .is_some());
    // A NaN scale fails closed too.
    assert_eq!(
        publish(2.0, 2.0, 1.0, 0.0, 1.0, f64::NAN, 0.3, None),
        Outcome::Abstain(Reason::ImplausibleRadius)
    );
}

#[test]
fn publish_carries_the_covariance_and_is_never_calibrated() {
    let o = publish(1.5, 2.5, 4.0, 0.0, 1.0, 0.5, 0.3, Some(0.7));
    let e = o.estimate().expect("published");
    assert_eq!((e.x, e.y), (1.5, 2.5));
    assert_eq!(e.radius_m, 0.5 * (2.0 * k95()));
    assert_eq!((e.cov_xx, e.cov_xy, e.cov_yy), (4.0, 0.0, 1.0));
    assert_eq!(e.confidence, Some(0.7));
    assert!(!e.radius_calibrated);
    assert_eq!(o.reason(), None);
    // A position that is not finite is never published, even with
    // min_plausible_radius_m = 0 (radius 0 from a NaN covariance).
    assert_eq!(
        publish(f64::NAN, 1.0, f64::NAN, 0.0, 0.0, 1.0, 0.0, None),
        Outcome::Abstain(Reason::ImplausibleRadius)
    );
}

#[test]
fn reason_spellings_are_the_record_spellings() {
    let all = [
        (Reason::NoDisturbance, "no_disturbance"),
        (Reason::NotSignificant, "not_significant"),
        (Reason::NoReports, "no_reports"),
        (Reason::TooFewRanges, "too_few_ranges"),
        (Reason::Degenerate, "degenerate"),
        (Reason::NotConverged, "not_converged"),
        (Reason::Gated, "gated"),
        (Reason::ImplausibleRadius, "implausible_radius"),
    ];
    for (r, s) in all {
        assert_eq!(r.as_str(), s);
        assert_eq!(r.to_string(), s);
    }
}
