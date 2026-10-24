//! Weighted-link-midpoint baseline, WLM (spec/locate.md §5).
//!
//! Works on plain link data ([`LinkGeom`]: the ported `Link`'s `tx`, `rx`,
//! `midpoint` and `length_m`, the last computed by the ported code with
//! `py_hypot`) and a per-link `y` and `null_sigma` in the localize
//! canonical order. It abstains instead of falling back to the room
//! centre, and every estimate carries a radius (§8).

// NaN fails closed: `!(W > 0)` is the spec's `not (W > 0)`.
#![allow(clippy::neg_cmp_op_on_partial_ord)]

use crate::num::{pymax, Neumaier};
use crate::phone::{Config, Error};
use crate::radius::{publish, Outcome, Reason};

/// One link's geometry, as the ported `geometry.Link` gives it.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinkGeom {
    /// Transmitter node position `(x, y)`.
    pub tx: (f64, f64),
    /// Receiver node position `(x, y)`.
    pub rx: (f64, f64),
    /// `Link.midpoint`.
    pub midpoint: (f64, f64),
    /// `Link.length_m` (ported code: CPython `math.hypot`, Rust `py_hypot`).
    pub length_m: f64,
}

/// The WLM parameters of §5 (§9 defaults: `wlm_gate_z` 3.0, `wlm_floor`
/// 0.25 for `rssi_dbm` and 0.0 otherwise, `wlm_radius_scale` 1.0,
/// `min_plausible_radius_m` 0.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WlmParams {
    pub gate_z: f64,
    pub floor: f64,
    pub radius_scale: f64,
    pub min_plausible_radius_m: f64,
}

impl WlmParams {
    /// The parameters of `cfg` for the link signal named `signal_name`
    /// (the floor's default depends on it: [`Config::wlm_floor_for`]).
    pub fn from_config(cfg: &Config, signal_name: &str) -> Self {
        Self {
            gate_z: cfg.wlm_gate_z,
            floor: cfg.wlm_floor_for(signal_name),
            radius_scale: cfg.wlm_radius_scale,
            min_plausible_radius_m: cfg.min_plausible_radius_m,
        }
    }
}

/// The WLM covariance terms of one link (§5 step 4): `(a_l, c_l, ux, uy)`.
/// For `d_l > 0`: `ux = (rx.x - tx.x) / d_l`, `uy = (rx.y - tx.y) / d_l`,
/// `a_l = (d_l*d_l) / 12.0`, `b_l = 0.5 * sqrt((d_l + λ)*(d_l + λ) - d_l*d_l)`,
/// `c_l = (b_l*b_l) / 3.0`; otherwise (the guard, NaN included) all zero.
pub fn link_spread(link: &LinkGeom, lambda_m: f64) -> (f64, f64, f64, f64) {
    let d = link.length_m;
    if d > 0.0 {
        let ux = (link.rx.0 - link.tx.0) / d;
        let uy = (link.rx.1 - link.tx.1) / d;
        let a = (d * d) / 12.0;
        let b = 0.5 * ((d + lambda_m) * (d + lambda_m) - d * d).sqrt();
        let c = (b * b) / 3.0;
        (a, c, ux, uy)
    } else {
        (0.0, 0.0, 0.0, 0.0)
    }
}

/// Is link `l` disturbed (§5 step 1)? `null_sigma_l > 0` and
/// `y_l > wlm_gate_z * null_sigma_l + wlm_floor`; NaN fails.
pub fn disturbed(y: f64, null_sigma: f64, params: &WlmParams) -> bool {
    null_sigma > 0.0 && y > params.gate_z * null_sigma + params.floor
}

/// The weighted-link-midpoint estimate of §5.
///
/// `links`, `y` and `null_sigma` are in the localize canonical order and
/// of one length `L` (else an [`Error::Input`] naming `y` or
/// `null_sigma`); `lambda_m` is `geometry.lambda_m`. Returns
/// `Abstain(NoDisturbance)` when no link passes the gate or
/// `not (W > 0)`, `Abstain(ImplausibleRadius)` when the radius fails §8,
/// else the estimate with `cov_* = s_*` and `confidence: None`.
pub fn wlm(
    links: &[LinkGeom],
    lambda_m: f64,
    y: &[f64],
    null_sigma: &[f64],
    params: &WlmParams,
) -> Result<Outcome, Error> {
    if y.len() != links.len() {
        return Err(Error::Input {
            field: "y",
            reason: format!("has {} entries for {} links", y.len(), links.len()),
        });
    }
    if null_sigma.len() != links.len() {
        return Err(Error::Input {
            field: "null_sigma",
            reason: format!("has {} entries for {} links", null_sigma.len(), links.len()),
        });
    }
    // 1. Gate.
    let any = y
        .iter()
        .zip(null_sigma)
        .any(|(&yl, &sl)| disturbed(yl, sl, params));
    if !any {
        return Ok(Outcome::Abstain(Reason::NoDisturbance));
    }
    // 2. Weights, every link: w_l = max(y_l, 0.0) (Python two-argument max).
    let w: Vec<f64> = y.iter().map(|&yl| pymax(yl, 0.0)).collect();
    let big_w = Neumaier::sum(w.iter().copied());
    if !(big_w > 0.0) {
        return Ok(Outcome::Abstain(Reason::NoDisturbance));
    }
    // 3. Estimate.
    let xh = Neumaier::sum(w.iter().zip(links).map(|(&wl, l)| wl * l.midpoint.0)) / big_w;
    let yh = Neumaier::sum(w.iter().zip(links).map(|(&wl, l)| wl * l.midpoint.1)) / big_w;
    // 4. Covariance.
    let mut sxx = Neumaier::new();
    let mut sxy = Neumaier::new();
    let mut syy = Neumaier::new();
    for (&wl, l) in w.iter().zip(links) {
        let (a, c, ux, uy) = link_spread(l, lambda_m);
        let ex = l.midpoint.0 - xh;
        let ey = l.midpoint.1 - yh;
        sxx.add(wl * ((((a * ux) * ux) + ((c * uy) * uy)) + ex * ex));
        sxy.add(wl * ((((a * ux) * uy) - ((c * uy) * ux)) + ex * ey));
        syy.add(wl * ((((a * uy) * uy) + ((c * ux) * ux)) + ey * ey));
    }
    let s_xx = sxx.value() / big_w;
    let s_xy = sxy.value() / big_w;
    let s_yy = syy.value() / big_w;
    // 5. Radius: r = wlm_radius_scale * radius95(s_xx, s_xy, s_yy).
    Ok(publish(
        xh,
        yh,
        s_xx,
        s_xy,
        s_yy,
        params.radius_scale,
        params.min_plausible_radius_m,
        None,
    ))
}
