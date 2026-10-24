//! The radius (spec/locate.md §8) and the published estimate (§11).
//!
//! An estimate is always a position **with a radius**, or an abstention
//! with a reason; never a bare point (ADR-008 decision 5). The radius is a
//! model statement, uncalibrated in Phase 4: every [`Estimate`] carries
//! `radius_calibrated: false`.

use crate::num::{finite_at_least, k95};

/// Why an estimator abstained (§11). One enum for every estimator; each
/// estimator uses only its own reasons:
/// WLM `no_disturbance`, `implausible_radius`; RTI `not_significant`,
/// `implausible_radius`; phone (§7.4, first that applies wins)
/// `no_reports`, `too_few_ranges`, `degenerate`, `not_converged`, `gated`,
/// `implausible_radius`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Reason {
    /// WLM: no link over the gate, or no positive weight (§5 steps 1-2).
    NoDisturbance,
    /// RTI: the tracker returned no estimate (§10.1).
    NotSignificant,
    /// Phone: no report from any node at this epoch (§7.1, §7.2).
    NoReports,
    /// Phone: fewer than 3 reports (§7.1, §7.2).
    TooFewRanges,
    /// Phone: the reports' anchors are collinear (§7), the RSSI start's
    /// weight sum fails its guard (§7.2), or a normal matrix is degenerate
    /// (§7.1).
    Degenerate,
    /// Phone: Gauss-Newton did not meet its stop rule in 200 iterations
    /// (§7.1).
    NotConverged,
    /// Phone: the Kalman gate rejected the fix (§7.3).
    Gated,
    /// Any: the position or radius is not finite, or the radius is not
    /// `> 0` or is below `min_plausible_radius_m` (§8).
    ImplausibleRadius,
}

impl Reason {
    /// The record spelling (§11), e.g. `"no_disturbance"`.
    pub fn as_str(self) -> &'static str {
        match self {
            Reason::NoDisturbance => "no_disturbance",
            Reason::NotSignificant => "not_significant",
            Reason::NoReports => "no_reports",
            Reason::TooFewRanges => "too_few_ranges",
            Reason::Degenerate => "degenerate",
            Reason::NotConverged => "not_converged",
            Reason::Gated => "gated",
            Reason::ImplausibleRadius => "implausible_radius",
        }
    }
}

impl std::fmt::Display for Reason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// A published estimate (§11 **Estimate**): a position with a radius and
/// the covariance the radius was taken from.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Estimate {
    pub x: f64,
    pub y: f64,
    /// `scale * radius95(cov_xx, cov_xy, cov_yy)`; finite, `> 0` and at
    /// least `min_plausible_radius_m`, by construction ([`publish`]).
    pub radius_m: f64,
    pub cov_xx: f64,
    pub cov_xy: f64,
    pub cov_yy: f64,
    /// The tracker's confidence for RTI; `None` (`null`) otherwise.
    pub confidence: Option<f64>,
    /// Always `false` in Phase 4 (§8).
    pub radius_calibrated: bool,
}

/// An estimator's output at one tick or epoch: an [`Estimate`] or an
/// abstention with its [`Reason`] (§11 **Abstention** `{reason}`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Outcome {
    Estimate(Estimate),
    Abstain(Reason),
}

impl Outcome {
    /// The estimate, if one was published.
    pub fn estimate(&self) -> Option<&Estimate> {
        match self {
            Outcome::Estimate(e) => Some(e),
            Outcome::Abstain(_) => None,
        }
    }

    /// The reason, if the estimator abstained.
    pub fn reason(&self) -> Option<Reason> {
        match self {
            Outcome::Estimate(_) => None,
            Outcome::Abstain(r) => Some(*r),
        }
    }
}

/// `radius95(a, b, d)` for the covariance `[[a, b], [b, d]]` (§8):
/// `h = 0.5*(a - d)`; `lmax = 0.5*(a + d) + sqrt(h*h + b*b)`;
/// `mm = lmax > 0.0 ? lmax : 0.0` (a NaN gives 0.0); `K95 * sqrt(mm)`.
/// The circle holds at least 95% of a Gaussian with that covariance,
/// whatever its shape.
pub fn radius95(a: f64, b: f64, d: f64) -> f64 {
    let h = 0.5 * (a - d);
    let lmax = 0.5 * (a + d) + (h * h + b * b).sqrt();
    let mm = if lmax > 0.0 { lmax } else { 0.0 };
    k95() * mm.sqrt()
}

/// The radius half of §8's publication test: `isfinite(r) and r > 0.0
/// and r >= min_plausible_radius_m` (NaN fails). `r > 0` holds even for a
/// caller that passes a zero (or negative) minimum, so a bare point is
/// never published (ADR-008 d5).
pub fn plausible(r: f64, min_plausible_radius_m: f64) -> bool {
    finite_at_least(r, min_plausible_radius_m) && r > 0.0
}

/// Publish `(x, y)` with `r = scale * radius95(cov_xx, cov_xy, cov_yy)`
/// (§5 step 5, §7.4, §8): an [`Estimate`] only if `x`, `y` and `r` are
/// finite, `r > 0` and `r >= min_plausible_radius_m` (NaN fails), else
/// `Abstain(ImplausibleRadius)`. The `r > 0` term holds whatever the
/// caller's minimum or scale (the library entry points take them raw), so
/// no radius-0 point is ever published.
#[allow(clippy::too_many_arguments)]
pub fn publish(
    x: f64,
    y: f64,
    cov_xx: f64,
    cov_xy: f64,
    cov_yy: f64,
    scale: f64,
    min_plausible_radius_m: f64,
    confidence: Option<f64>,
) -> Outcome {
    let r = scale * radius95(cov_xx, cov_xy, cov_yy);
    if !(plausible(r, min_plausible_radius_m) && x.is_finite() && y.is_finite()) {
        return Outcome::Abstain(Reason::ImplausibleRadius);
    }
    Outcome::Estimate(Estimate {
        x,
        y,
        radius_m: r,
        cov_xx,
        cov_xy,
        cov_yy,
        confidence,
        radius_calibrated: false,
    })
}
