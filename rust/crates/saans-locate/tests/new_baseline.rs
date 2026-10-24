//! spec/locate.md §5 (weighted-link-midpoint baseline) and its §12a
//! checks with known geometry, on plain link data.

use saans_locate::baseline::{disturbed, link_spread, wlm, LinkGeom, WlmParams};
use saans_locate::num::k95;
use saans_locate::phone::{Config, Error};
use saans_locate::radius::{radius95, Outcome, Reason};

const LAMBDA: f64 = 0.3;

fn params() -> WlmParams {
    WlmParams::from_config(&Config::default(), "rssi_dbm")
}

/// The localize canonical order: every ordered pair `tx != rx`, sorted by
/// id, tx outer (§2).
fn links(nodes: &[(f64, f64)]) -> Vec<LinkGeom> {
    let mut out = Vec::new();
    for (i, &a) in nodes.iter().enumerate() {
        for (j, &b) in nodes.iter().enumerate() {
            if i == j {
                continue;
            }
            let (dx, dy) = (b.0 - a.0, b.1 - a.1);
            out.push(LinkGeom {
                tx: a,
                rx: b,
                midpoint: (0.5 * (a.0 + b.0), 0.5 * (a.1 + b.1)),
                length_m: (dx * dx + dy * dy).sqrt(),
            });
        }
    }
    out
}

const SQUARE: [(f64, f64); 4] = [(0.0, 0.0), (4.0, 0.0), (4.0, 4.0), (0.0, 4.0)];

/// Index of link (tx, rx) in the canonical order of 4 nodes.
fn li(tx: usize, rx: usize) -> usize {
    tx * 3 + if rx > tx { rx - 1 } else { rx }
}

fn close(a: f64, b: f64, rel: f64) -> bool {
    (a - b).abs() <= rel * a.abs().max(b.abs()) + 1e-300
}

#[test]
fn floor_and_params_from_config() {
    let mut cfg = Config::default();
    assert_eq!(cfg.wlm_floor_for("rssi_dbm"), 0.25);
    assert_eq!(cfg.wlm_floor_for("total_power"), 0.0);
    let p = WlmParams::from_config(&cfg, "rssi_dbm");
    assert_eq!(
        p,
        WlmParams {
            gate_z: 3.0,
            floor: 0.25,
            radius_scale: 1.0,
            min_plausible_radius_m: 0.3
        }
    );
    cfg.wlm_floor = Some(0.1);
    assert_eq!(cfg.wlm_floor_for("rssi_dbm"), 0.1);
    assert_eq!(cfg.wlm_floor_for("total_power"), 0.1);
    // Each *_radius_scale scales only its estimator: the WLM takes
    // wlm_radius_scale, never the RTI's or the phone's (distinct values).
    let scaled = Config {
        wlm_radius_scale: 1.5,
        rti_radius_scale: 2.0,
        phone_radius_scale: 0.75,
        min_plausible_radius_m: 0.4,
        ..Config::default()
    };
    let p = WlmParams::from_config(&scaled, "rssi_dbm");
    assert_eq!((p.radius_scale, p.min_plausible_radius_m), (1.5, 0.4));
    // ... and so the published radius is 1.5 * radius95 of its covariance.
    let sq = links(&SQUARE);
    let mut y = vec![0.0; sq.len()];
    y[li(0, 2)] = 2.0;
    let ns = vec![0.1; sq.len()];
    let e = *wlm(&sq, LAMBDA, &y, &ns, &p).unwrap().estimate().unwrap();
    assert_eq!(e.radius_m, 1.5 * radius95(e.cov_xx, e.cov_xy, e.cov_yy));
}

#[test]
fn one_disturbed_link_gives_its_midpoint_and_segment_ellipse_covariance() {
    // §12a: one link with y > 0 -> that link's midpoint, covariance
    // a_l·uuᵀ + c_l·vvᵀ.
    let ls = links(&SQUARE);
    let mut y = vec![0.0; ls.len()];
    let k = li(0, 2);
    y[k] = 2.0;
    let null_sigma = vec![0.1; ls.len()];
    let out = wlm(&ls, LAMBDA, &y, &null_sigma, &params()).unwrap();
    let e = *out.estimate().expect("published");
    assert_eq!((e.x, e.y), (2.0, 2.0));
    let d = ls[k].length_m;
    let (ux, uy) = (4.0 / d, 4.0 / d);
    let (vx, vy) = (-uy, ux);
    let a = (d * d) / 12.0;
    // b_l^2 / 3 = ((d+λ)^2 - d^2) / 12: the across-link variance of a
    // uniform density over the first Fresnel-like ellipse's width.
    let c = ((d + LAMBDA) * (d + LAMBDA) - d * d) / 12.0;
    let want = (
        a * ux * ux + c * vx * vx,
        a * ux * uy + c * vx * vy,
        a * uy * uy + c * vy * vy,
    );
    assert!(close(e.cov_xx, want.0, 1e-14), "{} {}", e.cov_xx, want.0);
    assert!(close(e.cov_xy, want.1, 1e-14), "{} {}", e.cov_xy, want.1);
    assert!(close(e.cov_yy, want.2, 1e-14), "{} {}", e.cov_yy, want.2);
    assert_eq!(e.radius_m, radius95(e.cov_xx, e.cov_xy, e.cov_yy));
    // Its largest eigenvalue is the along-link a_l.
    assert!(close(e.radius_m, k95() * a.sqrt(), 1e-12));
    assert_eq!(e.confidence, None);
    assert!(!e.radius_calibrated);
    let (al, cl, lux, luy) = link_spread(&ls[k], LAMBDA);
    assert_eq!((al, lux, luy), (a, ux, uy));
    assert!(close(cl, c, 1e-12));
}

#[test]
fn a_squares_two_diagonals_equally_disturbed_give_the_centre() {
    let ls = links(&SQUARE);
    let mut y = vec![0.0; ls.len()];
    for (tx, rx) in [(0, 2), (2, 0), (1, 3), (3, 1)] {
        y[li(tx, rx)] = 1.0;
    }
    let null_sigma = vec![0.1; ls.len()];
    let e = *wlm(&ls, LAMBDA, &y, &null_sigma, &params())
        .unwrap()
        .estimate()
        .expect("published");
    assert_eq!((e.x, e.y), (2.0, 2.0));
    // The two crossed segment ellipses make a round covariance.
    let d = 32.0f64.sqrt();
    let a = (d * d) / 12.0;
    let c = ((d + LAMBDA) * (d + LAMBDA) - d * d) / 12.0;
    assert!(close(e.cov_xx, 0.5 * (a + c), 1e-12));
    assert!(close(e.cov_yy, 0.5 * (a + c), 1e-12));
    assert!(e.cov_xy.abs() < 1e-12);
    println!(
        "SIM WLM square diagonals: centre ({}, {}), radius {:.4} m",
        e.x, e.y, e.radius_m
    );
}

#[test]
fn nothing_over_the_gate_abstains_no_disturbance() {
    let ls = links(&SQUARE);
    let null_sigma = vec![0.1; ls.len()];
    // Gate: y > 3.0 * 0.1 + 0.25 = 0.55 (strict).
    let y = vec![0.55; ls.len()];
    assert_eq!(
        wlm(&ls, LAMBDA, &y, &null_sigma, &params()).unwrap(),
        Outcome::Abstain(Reason::NoDisturbance)
    );
    let mut y2 = y.clone();
    y2[3] = 0.5500001;
    assert!(wlm(&ls, LAMBDA, &y2, &null_sigma, &params())
        .unwrap()
        .estimate()
        .is_some());
    // null_sigma <= 0 never passes the gate, whatever y.
    let y3 = vec![100.0; ls.len()];
    let zero = vec![0.0; ls.len()];
    assert_eq!(
        wlm(&ls, LAMBDA, &y3, &zero, &params()).unwrap(),
        Outcome::Abstain(Reason::NoDisturbance)
    );
    // All empty.
    assert_eq!(
        wlm(&[], LAMBDA, &[], &[], &params()).unwrap(),
        Outcome::Abstain(Reason::NoDisturbance)
    );
}

#[test]
fn nan_fails_closed_in_the_gate_and_the_weights() {
    let p = params();
    assert!(!disturbed(f64::NAN, 0.1, &p));
    assert!(!disturbed(5.0, f64::NAN, &p));
    assert!(!disturbed(5.0, 0.0, &p));
    assert!(disturbed(5.0, 0.1, &p));
    let nan_gate = WlmParams {
        gate_z: f64::NAN,
        ..p
    };
    assert!(!disturbed(5.0, 0.1, &nan_gate));

    let ls = links(&SQUARE);
    let null_sigma = vec![0.1; ls.len()];
    // A NaN y elsewhere makes W NaN: `not (W > 0)` abstains.
    let mut y = vec![0.0; ls.len()];
    y[li(0, 2)] = 2.0;
    y[li(1, 0)] = f64::NAN;
    assert_eq!(
        wlm(&ls, LAMBDA, &y, &null_sigma, &p).unwrap(),
        Outcome::Abstain(Reason::NoDisturbance)
    );
    // A disturbed link with weight 0 (only with a negative floor, which
    // Config refuses): W = 0 abstains.
    let neg = WlmParams {
        gate_z: 0.0,
        floor: -1.0,
        ..p
    };
    let y = vec![-0.5; ls.len()];
    assert_eq!(
        wlm(&ls, LAMBDA, &y, &null_sigma, &neg).unwrap(),
        Outcome::Abstain(Reason::NoDisturbance)
    );
    // An infinite y gives a NaN estimate: never published.
    let mut y = vec![0.0; ls.len()];
    y[li(0, 2)] = f64::INFINITY;
    let zero_min = WlmParams {
        min_plausible_radius_m: 0.0,
        ..p
    };
    for q in [p, zero_min] {
        assert_eq!(
            wlm(&ls, LAMBDA, &y, &null_sigma, &q).unwrap(),
            Outcome::Abstain(Reason::ImplausibleRadius)
        );
    }
}

#[test]
fn small_spread_abstains_implausible_radius_and_scale_multiplies() {
    // Two nodes 0.2 m apart with a tiny λ: the radius is below 0.3 m.
    let ls = links(&[(1.0, 1.0), (1.2, 1.0)]);
    let y = vec![2.0, 2.0];
    let ns = vec![0.1, 0.1];
    assert_eq!(
        wlm(&ls, 0.001, &y, &ns, &params()).unwrap(),
        Outcome::Abstain(Reason::ImplausibleRadius)
    );
    // radius_scale multiplies radius95 exactly.
    let sq = links(&SQUARE);
    let mut y = vec![0.0; sq.len()];
    y[li(0, 2)] = 2.0;
    let ns = vec![0.1; sq.len()];
    let r1 = wlm(&sq, LAMBDA, &y, &ns, &params())
        .unwrap()
        .estimate()
        .unwrap()
        .radius_m;
    let doubled = WlmParams {
        radius_scale: 2.0,
        ..params()
    };
    let r2 = wlm(&sq, LAMBDA, &y, &ns, &doubled)
        .unwrap()
        .estimate()
        .unwrap()
        .radius_m;
    assert_eq!(r2, 2.0 * r1);
}

#[test]
fn zero_length_links_add_only_their_midpoint_spread() {
    let l = LinkGeom {
        tx: (1.0, 1.0),
        rx: (1.0, 1.0),
        midpoint: (1.0, 1.0),
        length_m: 0.0,
    };
    assert_eq!(link_spread(&l, LAMBDA), (0.0, 0.0, 0.0, 0.0));
    let nan = LinkGeom {
        length_m: f64::NAN,
        ..l
    };
    assert_eq!(link_spread(&nan, LAMBDA), (0.0, 0.0, 0.0, 0.0));
}

#[test]
fn length_mismatch_is_an_input_error() {
    let ls = links(&SQUARE);
    let e = wlm(&ls, LAMBDA, &[0.0; 3], &vec![0.1; ls.len()], &params()).unwrap_err();
    assert!(matches!(e, Error::Input { field: "y", .. }), "{e}");
    let e = wlm(&ls, LAMBDA, &vec![0.0; ls.len()], &[0.1; 2], &params()).unwrap_err();
    assert!(
        matches!(
            e,
            Error::Input {
                field: "null_sigma",
                ..
            }
        ),
        "{e}"
    );
}

#[test]
fn no_published_wlm_radius_is_below_the_floor() {
    // A deterministic sweep of y patterns over the square.
    let ls = links(&SQUARE);
    let ns = vec![0.1; ls.len()];
    let mut rng = saans_sim::rng::Rng::new(4);
    let mut published = 0;
    for _ in 0..2000 {
        let y: Vec<f64> = ls.iter().map(|_| rng.range(-1.0, 3.0)).collect();
        if let Outcome::Estimate(e) = wlm(&ls, LAMBDA, &y, &ns, &params()).unwrap() {
            assert!(e.radius_m.is_finite() && e.radius_m >= 0.3);
            assert!(e.x.is_finite() && e.y.is_finite());
            published += 1;
        }
    }
    assert!(published > 1000);
}

#[test]
fn a_zero_radius_is_never_published_even_with_a_zero_minimum() {
    // §8: every publication test also requires r > 0, so a raw library
    // call with min_plausible_radius_m 0 (which Config refuses) still
    // never publishes a bare point (ADR-008 d5).
    let sq = links(&SQUARE);
    let mut y = vec![0.0; sq.len()];
    y[li(0, 2)] = 2.0;
    let ns = vec![0.1; sq.len()];
    // The estimate this data gives with the defaults: a positive radius.
    let e = *wlm(&sq, LAMBDA, &y, &ns, &params())
        .unwrap()
        .estimate()
        .unwrap();
    assert!(e.radius_m > 0.3);
    // Scale 0, minimum 0: r = 0 >= 0, but not > 0.
    let zero = WlmParams {
        radius_scale: 0.0,
        min_plausible_radius_m: 0.0,
        ..params()
    };
    // Scale -1, minimum -10: a negative radius.
    let negative = WlmParams {
        radius_scale: -1.0,
        min_plausible_radius_m: -10.0,
        ..params()
    };
    for q in [zero, negative] {
        assert_eq!(
            wlm(&sq, LAMBDA, &y, &ns, &q).unwrap(),
            Outcome::Abstain(Reason::ImplausibleRadius),
            "{q:?}"
        );
    }
    // A zero covariance (one zero-length link, its midpoint only) gives
    // r = 0 with scale 1: with minimum 0 it abstains too.
    let point = links(&[(1.0, 1.0), (1.0, 1.0)]);
    let zero_min = WlmParams {
        min_plausible_radius_m: 0.0,
        ..params()
    };
    assert_eq!(
        wlm(&point, LAMBDA, &[2.0, 2.0], &[0.1, 0.1], &zero_min).unwrap(),
        Outcome::Abstain(Reason::ImplausibleRadius)
    );
}
