//! Per-link disturbance measurement, the `y` vector RTI inverts: a port of
//! `wifisense/localize/measure.py`, line by line (spec/locate.md §3, §4.1),
//! plus the new [`RssiSignal`] of spec/locate.md §3.
//!
//! `y_l` is the sample variance (Welford, n-1) over a sliding **time**
//! window of a per-packet scalar, minus the link's empty-room baseline
//! (SUBTRACT) or divided by it, minus one (NORMALIZE). A link with fewer
//! than `min_samples` samples in its window reads `0.0`; [`LinkMeasurement::ready_mask`]
//! tells "no data" from "quiet". Each link prunes against its own newest
//! timestamp plus a mesh progress clock, so one link's timing trouble
//! stays on that link (stale samples dropped once, a restart confirmed by
//! a second sample, a hard cap on window length, all counted).
//!
//! The per-packet scalar is a [`LinkSignal`] strategy:
//! [`TotalPowerSignal`] (`10*log10` of the mean power per subcarrier),
//! [`StaticSuppressedSignal`] (`saans_dsp`'s `StaticSuppressor` + `fuse`)
//! and [`RssiSignal`] (`float(frame.rssi_dbm)`). All three share the
//! validity gate [`frame_power`]. Inputs are SIM in Phase 4.

// The Python is the specification line by line: index loops mirror it, and
// `!(a > b)` is how a comparison is written so that a NaN fails closed.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

use std::borrow::Borrow;
use std::collections::VecDeque;
use std::fmt;
use std::sync::Arc;

use saans_dsp::preprocess::{fuse, StaticSuppressor};
pub use saans_dsp::types::{CsiFrame, RunningVariance};

use crate::geometry::Geometry;
use crate::pynum::{fmt_py, py_ceil, py_round_int, py_sort, pymax, pymin, LocateError, Result};

/// What to do with the empty-room baseline variance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum CalibrationMode {
    /// `y = var - baseline` (dB² for the dB and RSSI signals). Default.
    #[default]
    Subtract,
    /// `y = var / baseline - 1` (`0.0` when the baseline is not positive).
    Normalize,
}

impl CalibrationMode {
    /// The enum's string value, `"subtract"` or `"normalize"`.
    pub fn value(self) -> &'static str {
        match self {
            CalibrationMode::Subtract => "subtract",
            CalibrationMode::Normalize => "normalize",
        }
    }

    /// `CalibrationMode(value)`: `ValueError` for anything else.
    pub fn parse(value: &str) -> Result<CalibrationMode> {
        match value {
            "subtract" => Ok(CalibrationMode::Subtract),
            "normalize" => Ok(CalibrationMode::Normalize),
            other => Err(LocateError::ValueError(format!(
                "'{other}' is not a valid CalibrationMode"
            ))),
        }
    }
}

/// `CsiFrame(...)` with `__post_init__`'s two checks: `ValueError` if the
/// amplitude and phase lists differ in length or `sample_rate_hz <= 0`
/// (a NaN rate passes, as in Python).
pub fn new_csi_frame(
    amplitudes: Vec<f64>,
    phases: Vec<f64>,
    sample_index: u64,
    sample_rate_hz: f64,
    rssi_dbm: i32,
    node_id: u8,
) -> Result<CsiFrame> {
    let frame = CsiFrame {
        amplitudes,
        phases,
        sample_index,
        sample_rate_hz,
        rssi_dbm,
        node_id,
    };
    check_csi_frame(&frame)?;
    Ok(frame)
}

/// `CsiFrame.__post_init__`'s two checks on an existing frame.
pub fn check_csi_frame(frame: &CsiFrame) -> Result<()> {
    if frame.amplitudes.len() != frame.phases.len() {
        return Err(LocateError::ValueError(format!(
            "amplitude/phase length mismatch: {} != {}",
            frame.amplitudes.len(),
            frame.phases.len()
        )));
    }
    if frame.sample_rate_hz <= 0.0 {
        return Err(LocateError::ValueError(format!(
            "sample_rate_hz must be positive, got {}",
            fmt_py(frame.sample_rate_hz)
        )));
    }
    Ok(())
}

/// `_frame_power`: the MEAN received power per subcarrier, `sum(amp^2) /
/// n` with a plain sequential sum, or `None` (drop the frame) if it has no
/// subcarriers, any non-finite amplitude, or a total that is not `> 0` and
/// finite.
pub fn frame_power(frame: &CsiFrame) -> Option<f64> {
    let n = frame.n_subcarriers();
    if n == 0 {
        return None;
    }
    let mut total = 0.0;
    for &amp in &frame.amplitudes {
        if !amp.is_finite() {
            return None;
        }
        total += amp * amp;
    }
    if !(total > 0.0 && total.is_finite()) {
        return None;
    }
    Some(total / n as f64)
}

/// Per-link running state for one [`LinkSignal`]: one per link, never
/// shared.
pub trait LinkSignalState: Send + fmt::Debug {
    /// Reduce one frame to a scalar, `Ok(None)` to drop the sample.
    fn push(&mut self, frame: &CsiFrame) -> Result<Option<f64>>;

    /// Forget all state (a no-op for stateless signals).
    fn reset(&mut self) {}
}

/// Strategy: how a link's frames become a scalar time series.
pub trait LinkSignal: Send + Sync + fmt::Debug {
    /// Short label naming the signal in reports.
    fn name(&self) -> &str;

    /// Fresh per-link state; `sample_rate_hz` is the PER-LINK rate.
    fn new_state(&self, sample_rate_hz: f64) -> Box<dyn LinkSignalState>;
}

/// (a) Mean received power per subcarrier in dB (`"total_power_db"`).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct TotalPowerSignal;

#[derive(Debug)]
struct TotalPowerState;

impl LinkSignalState for TotalPowerState {
    fn push(&mut self, frame: &CsiFrame) -> Result<Option<f64>> {
        let Some(mean_power) = frame_power(frame) else {
            return Ok(None);
        };
        let value = 10.0 * mean_power.log10();
        Ok(if value.is_finite() { Some(value) } else { None })
    }
}

impl LinkSignal for TotalPowerSignal {
    fn name(&self) -> &str {
        "total_power_db"
    }

    fn new_state(&self, _sample_rate_hz: f64) -> Box<dyn LinkSignalState> {
        Box::new(TotalPowerState)
    }
}

/// (b) Static-suppressed amplitude residual fused across subcarriers
/// (`"static_suppressed_amplitude"`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StaticSuppressedSignal {
    pub corner_hz: f64,
}

impl Default for StaticSuppressedSignal {
    fn default() -> Self {
        StaticSuppressedSignal { corner_hz: 0.03 }
    }
}

/// `preprocess.corner_to_alpha` as Python computes it:
/// `min(max(2*pi*corner/rate, 1e-6), 0.999)`, `ValueError` if `rate <= 0`.
fn corner_to_alpha_py(corner_hz: f64, sample_rate_hz: f64) -> Result<f64> {
    if sample_rate_hz <= 0.0 {
        return Err(LocateError::ValueError(format!(
            "sample_rate_hz must be positive, got {}",
            fmt_py(sample_rate_hz)
        )));
    }
    let alpha = 2.0 * std::f64::consts::PI * corner_hz / sample_rate_hz;
    Ok(pymin(pymax(alpha, 1e-6), 0.999))
}

#[derive(Debug)]
struct StaticSuppressedState {
    sample_rate_hz: f64,
    corner_hz: f64,
    suppressor: Option<StaticSuppressor>,
}

impl LinkSignalState for StaticSuppressedState {
    fn push(&mut self, frame: &CsiFrame) -> Result<Option<f64>> {
        if frame.n_subcarriers() == 0 {
            return Ok(None);
        }
        // Validate BEFORE the suppressor sees the frame.
        if frame_power(frame).is_none() {
            return Ok(None);
        }
        let Some(suppressor) = self.suppressor.as_mut() else {
            // Built lazily from the first frame's subcarrier count; the
            // first frame only seeds the baselines and is discarded. The
            // alpha is Python's (saans-dsp's constructor would panic on a
            // NaN rate where Python carries a NaN alpha), so it is set here.
            let alpha = corner_to_alpha_py(self.corner_hz, self.sample_rate_hz)?;
            let mut suppressor = StaticSuppressor::new(frame.n_subcarriers(), 1.0, 0.0);
            suppressor.alpha = alpha;
            suppressor.sample_rate_hz = self.sample_rate_hz;
            suppressor.process(frame);
            self.suppressor = Some(suppressor);
            return Ok(None);
        };
        let (residuals, weights) = suppressor.process(frame);
        let value = fuse(&residuals, &weights);
        Ok(if value.is_finite() { Some(value) } else { None })
    }

    fn reset(&mut self) {
        self.suppressor = None;
    }
}

impl LinkSignal for StaticSuppressedSignal {
    fn name(&self) -> &str {
        "static_suppressed_amplitude"
    }

    fn new_state(&self, sample_rate_hz: f64) -> Box<dyn LinkSignalState> {
        Box::new(StaticSuppressedState {
            sample_rate_hz,
            corner_hz: self.corner_hz,
            suppressor: None,
        })
    }
}

/// The datagram's RSSI (`"rssi_dbm"`, spec/locate.md §3): stateless,
/// `float(frame.rssi_dbm)` for every frame [`frame_power`] accepts. Room
/// Sim's AGC normalises the CSI buffer but not the RSSI, so this is the
/// disturbance signal for Room Sim input (decision 2).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RssiSignal;

#[derive(Debug)]
struct RssiState;

impl LinkSignalState for RssiState {
    fn push(&mut self, frame: &CsiFrame) -> Result<Option<f64>> {
        if frame_power(frame).is_none() {
            return Ok(None);
        }
        Ok(Some(f64::from(frame.rssi_dbm)))
    }
}

impl LinkSignal for RssiSignal {
    fn name(&self) -> &str {
        "rssi_dbm"
    }

    fn new_state(&self, _sample_rate_hz: f64) -> Box<dyn LinkSignalState> {
        Box::new(RssiState)
    }
}

/// `measure._median`: median of a non-empty list, `0.5 * (a + b)` for an
/// even count.
fn median(values: &[f64]) -> Result<f64> {
    if values.is_empty() {
        return Err(LocateError::ValueError(
            "median of an empty list".to_string(),
        ));
    }
    let mut ordered = values.to_vec();
    py_sort(&mut ordered);
    let n = ordered.len();
    let mid = n / 2;
    if n % 2 == 1 {
        return Ok(ordered[mid]);
    }
    Ok(0.5 * (ordered[mid - 1] + ordered[mid]))
}

/// `LinkMeasurement`'s keyword arguments other than `geometry` and `signal`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeasurementConfig {
    /// The VRTI window, seconds (default 1.0).
    pub window_s: f64,
    /// The PER-LINK rate, Hz (default 12.5): sizes `min_samples` and the
    /// window cap.
    pub sample_rate_hz: f64,
    /// Default SUBTRACT.
    pub calibration_mode: CalibrationMode,
    /// `None`: `max(3, int(round(0.5 * window_s * sample_rate_hz)))`.
    pub min_samples: Option<i64>,
    /// Clamp negative `y` to 0 (default false).
    pub clamp_negative: bool,
}

impl Default for MeasurementConfig {
    fn default() -> Self {
        MeasurementConfig {
            window_s: 1.0,
            sample_rate_hz: 12.5,
            calibration_mode: CalibrationMode::Subtract,
            min_samples: None,
            clamp_negative: false,
        }
    }
}

/// Sliding-window per-link variance change, in canonical link order.
#[derive(Debug)]
pub struct LinkMeasurement {
    pub geometry: Arc<Geometry>,
    pub window_s: f64,
    pub sample_rate_hz: f64,
    pub signal: Box<dyn LinkSignal>,
    pub calibration_mode: CalibrationMode,
    pub clamp_negative: bool,
    pub min_samples: i64,
    /// Links whose stream restarted (window cleared).
    pub clock_restarts: u64,
    /// Samples dropped as more than one window behind their link's clock.
    pub stale_samples: u64,
    /// Samples evicted by the window-length cap.
    pub window_overflows: u64,
    windows: Vec<VecDeque<(f64, f64)>>,
    states: Vec<Box<dyn LinkSignalState>>,
    cached_var: Vec<f64>,
    dirty: Vec<bool>,
    baseline: Vec<f64>,
    baseline_std: Vec<f64>,
    baseline_ok: Vec<bool>,
    calibrated: bool,
    t_last: Vec<Option<f64>>,
    progress_at: Vec<f64>,
    pending_restart: Vec<Option<f64>>,
    progress: f64,
    t_stream: Option<f64>,
    max_window: i64,
}

impl LinkMeasurement {
    /// `LinkMeasurement(geometry, window_s, sample_rate_hz, signal,
    /// calibration_mode, min_samples, clamp_negative)`.
    ///
    /// `ValueError` if `window_s <= 0` or `sample_rate_hz <= 0`, or the
    /// resulting `min_samples < 2`; `round`/`ceil` of a NaN window size is
    /// a `ValueError` and of an infinite one an `OverflowError`, as in
    /// Python.
    pub fn new(
        geometry: impl Into<Arc<Geometry>>,
        config: MeasurementConfig,
        signal: Box<dyn LinkSignal>,
    ) -> Result<LinkMeasurement> {
        let geometry = geometry.into();
        let MeasurementConfig {
            window_s,
            sample_rate_hz,
            calibration_mode,
            min_samples,
            clamp_negative,
        } = config;
        if window_s <= 0.0 {
            return Err(LocateError::ValueError(format!(
                "window_s must be positive, got {}",
                fmt_py(window_s)
            )));
        }
        if sample_rate_hz <= 0.0 {
            return Err(LocateError::ValueError(format!(
                "sample_rate_hz must be positive, got {}",
                fmt_py(sample_rate_hz)
            )));
        }

        let expected = window_s * sample_rate_hz;
        let min_samples = match min_samples {
            None => py_round_int(0.5 * expected)?.max(3),
            Some(m) => m,
        };
        if min_samples < 2 {
            return Err(LocateError::ValueError(format!(
                "min_samples must be at least 2, got {min_samples}"
            )));
        }

        let n_links = geometry.n_links();
        let states = (0..n_links)
            .map(|_| signal.new_state(sample_rate_hz))
            .collect();
        let max_window = py_ceil(4.0 * expected)?.max(8);

        Ok(LinkMeasurement {
            geometry,
            window_s,
            sample_rate_hz,
            signal,
            calibration_mode,
            clamp_negative,
            min_samples,
            clock_restarts: 0,
            stale_samples: 0,
            window_overflows: 0,
            windows: (0..n_links).map(|_| VecDeque::new()).collect(),
            states,
            cached_var: vec![0.0; n_links],
            dirty: vec![true; n_links],
            baseline: vec![0.0; n_links],
            baseline_std: vec![0.0; n_links],
            baseline_ok: vec![false; n_links],
            calibrated: false,
            t_last: vec![None; n_links],
            progress_at: vec![0.0; n_links],
            pending_restart: vec![None; n_links],
            progress: 0.0,
            t_stream: None,
            max_window,
        })
    }

    // ---- ingest -------------------------------------------------------

    /// Absorb one packet. `KeyError` for a link not in the geometry;
    /// `ValueError` for a frame `CsiFrame.__post_init__` would refuse.
    pub fn update(&mut self, t: f64, tx_id: i64, rx_id: i64, frame: &CsiFrame) -> Result<()> {
        check_csi_frame(frame)?;
        let index = self.geometry.link_index(tx_id, rx_id)?;
        let Some(value) = self.states[index].push(frame)? else {
            return Ok(());
        };
        self.ingest(index, t, value);
        Ok(())
    }

    /// `_ingest`: place one scalar into its link's window; whether it
    /// entered.
    fn ingest(&mut self, index: usize, t: f64, value: f64) -> bool {
        let mut last = self.t_last[index];
        let mut restarted = false;

        if let Some(l) = last {
            if t < l - self.window_s {
                let pending = self.pending_restart[index];
                if pending.is_none() || pending.is_some_and(|p| t < p) {
                    self.pending_restart[index] = Some(t);
                    self.stale_samples += 1;
                    return false;
                }
                self.restart_link(index);
                last = None;
                restarted = true;
            }
        }

        self.pending_restart[index] = None;

        let mut step = match last {
            None => 0.0,
            Some(l) => t - l,
        };
        if step < 0.0 {
            step = 0.0;
        } else if step > self.window_s {
            step = self.window_s;
        }
        let reached = self.progress_at[index] + step;
        if reached > self.progress {
            self.progress = reached;
        }
        self.progress_at[index] = self.progress;

        match last {
            None => self.t_last[index] = Some(t),
            Some(l) if t > l => self.t_last[index] = Some(t),
            _ => {}
        }
        if self.t_stream.is_none() || self.t_stream.is_some_and(|s| t > s) || restarted {
            self.t_stream = Some(t);
        }

        self.insert(index, t, value);
        self.dirty[index] = true;
        self.prune(index);
        true
    }

    /// `_insert`: append in time order, scanning back from the tail.
    fn insert(&mut self, index: usize, t: f64, value: f64) {
        let window = &mut self.windows[index];
        if window.back().is_some_and(|&(last_t, _)| t < last_t) {
            let mut position = window.len();
            while position > 0 && window[position - 1].0 > t {
                position -= 1;
            }
            window.insert(position, (t, value));
        } else {
            window.push_back((t, value));
        }
    }

    /// `_restart_link`: forget one link's window (its signal state stays).
    fn restart_link(&mut self, index: usize) {
        self.windows[index].clear();
        self.cached_var[index] = 0.0;
        self.dirty[index] = true;
        self.t_last[index] = None;
        self.progress_at[index] = self.progress;
        self.clock_restarts += 1;
    }

    /// `_clear_windows`: drop all samples and clock state, keep signals.
    fn clear_windows(&mut self) {
        let n_links = self.geometry.n_links();
        for window in self.windows.iter_mut() {
            window.clear();
        }
        self.dirty = vec![true; n_links];
        self.cached_var = vec![0.0; n_links];
        self.t_last = vec![None; n_links];
        self.progress_at = vec![0.0; n_links];
        self.pending_restart = vec![None; n_links];
        self.progress = 0.0;
        self.t_stream = None;
    }

    /// `_advance_to`: move the progress clock to `t` without a sample
    /// (uncapped, monotone).
    fn advance_to(&mut self, t: Option<f64>) {
        let (Some(t), Some(t_stream)) = (t, self.t_stream) else {
            return;
        };
        let step = t - t_stream;
        if step <= 0.0 {
            return;
        }
        self.t_stream = Some(t);
        self.progress += step;
    }

    /// `_prune`: drop samples older than one window on this link's own
    /// clock (a sample exactly `window_s` old is kept, with a 1e-9
    /// relative tolerance), then enforce the length cap.
    fn prune(&mut self, index: usize) {
        let Some(last) = self.t_last[index] else {
            return;
        };
        let cutoff = last - self.window_s + (self.progress - self.progress_at[index]);
        let tolerance = 1e-9 * pymax(last.abs(), self.window_s);
        let window = &mut self.windows[index];
        while window
            .front()
            .is_some_and(|&(ts, _)| ts < cutoff - tolerance)
        {
            window.pop_front();
            self.dirty[index] = true;
        }
        while window.len() as i64 > self.max_window {
            window.pop_front();
            self.dirty[index] = true;
            self.window_overflows += 1;
        }
    }

    /// `_variance`: Welford over the window, cached until dirtied.
    fn variance(&mut self, index: usize) -> f64 {
        if self.dirty[index] {
            let mut accumulator = RunningVariance::default();
            for &(_, value) in &self.windows[index] {
                accumulator.update(value);
            }
            self.cached_var[index] = accumulator.variance();
            self.dirty[index] = false;
        }
        self.cached_var[index]
    }

    // ---- output -------------------------------------------------------

    /// The per-link disturbance vector `y`, canonical link order, always
    /// `n_links` long; `0.0` for links without `min_samples` samples.
    /// `t`: "the time is now this" (ages the windows of a stopped stream).
    /// Uncalibrated, it is the raw windowed variance.
    pub fn vector(&mut self, t: Option<f64>) -> Vec<f64> {
        self.advance_to(t);
        let n = self.geometry.n_links();
        let mut out = vec![0.0; n];
        let subtract = self.calibration_mode == CalibrationMode::Subtract;
        for index in 0..n {
            self.prune(index);
            if (self.windows[index].len() as i64) < self.min_samples {
                continue;
            }
            let variance = self.variance(index);
            let mut y = if !self.calibrated {
                variance
            } else if subtract {
                variance - self.baseline[index]
            } else {
                let baseline = self.baseline[index];
                if baseline > 0.0 {
                    variance / baseline - 1.0
                } else {
                    0.0
                }
            };
            if self.clamp_negative && y < 0.0 {
                y = 0.0;
            }
            out[index] = y;
        }
        out
    }

    /// Samples currently in each link's window, canonical order.
    pub fn sample_counts(&mut self, t: Option<f64>) -> Vec<usize> {
        self.advance_to(t);
        for index in 0..self.geometry.n_links() {
            self.prune(index);
        }
        self.windows.iter().map(|w| w.len()).collect()
    }

    /// Per link: enough samples for `vector()` to be real (and, calibrated
    /// in NORMALIZE mode, a positive baseline).
    pub fn ready_mask(&mut self, t: Option<f64>) -> Vec<bool> {
        let counts = self.sample_counts(t);
        let normalize = self.calibrated && self.calibration_mode == CalibrationMode::Normalize;
        counts
            .iter()
            .enumerate()
            .map(|(index, &count)| {
                count as i64 >= self.min_samples && !(normalize && !(self.baseline[index] > 0.0))
            })
            .collect()
    }

    /// The `(t, value)` samples in one link's window, oldest first (the
    /// Python's private `_windows[index]`, for diagnostics and tests).
    pub fn window_samples(&self, index: usize) -> Vec<(f64, f64)> {
        self.windows[index].iter().copied().collect()
    }

    // ---- calibration --------------------------------------------------

    pub fn is_calibrated(&self) -> bool {
        self.calibrated
    }

    /// Learn the empty-room baseline from a `(t, tx_id, rx_id, frame)`
    /// stream: one non-overlapping window estimate per `window_s` of each
    /// link's own clock; baseline = their mean, null sigma = their sample
    /// std. Links with fewer than `min_windows` estimates get the median
    /// over the links that have them and are flagged in
    /// [`LinkMeasurement::calibrated_mask`]. `ValueError` if no link
    /// reaches `min_windows`; `KeyError` for an unknown link. Windows are
    /// cleared before and after; signal state stays warm.
    pub fn calibrate_from<I, F>(&mut self, stream: I, min_windows: i64) -> Result<()>
    where
        I: IntoIterator<Item = (f64, i64, i64, F)>,
        F: Borrow<CsiFrame>,
    {
        let n_links = self.geometry.n_links();
        self.clear_windows();

        let mut accumulators = vec![RunningVariance::default(); n_links];
        let mut due: Vec<Option<f64>> = vec![None; n_links];

        for (t, tx_id, rx_id, frame) in stream {
            let frame = frame.borrow();
            check_csi_frame(frame)?;
            let index = self.geometry.link_index(tx_id, rx_id)?;
            let Some(value) = self.states[index].push(frame)? else {
                continue;
            };
            if !self.ingest(index, t, value) {
                continue;
            }

            let when = due[index];
            match when {
                None => {
                    due[index] = Some(t + self.window_s);
                    continue;
                }
                Some(w) if t < w - self.window_s => {
                    due[index] = Some(t + self.window_s);
                    continue;
                }
                Some(w) if t < w => continue,
                Some(_) => {}
            }
            if (self.windows[index].len() as i64) < self.min_samples {
                continue;
            }
            let variance = self.variance(index);
            accumulators[index].update(variance);
            due[index] = Some(t + self.window_s);
        }

        let ok: Vec<bool> = accumulators
            .iter()
            .map(|acc| i128::from(acc.count) >= i128::from(min_windows))
            .collect();
        if !ok.iter().any(|&b| b) {
            let best = accumulators.iter().map(|acc| acc.count).max().unwrap_or(0);
            return Err(LocateError::ValueError(format!(
                "calibration saw no link reach {min_windows} independent windows (best was {best}); \
                 the stream covers less than {} s per link, or the geometry does not match it",
                fmt_py(min_windows as f64 * self.window_s)
            )));
        }

        let means: Vec<f64> = (0..n_links)
            .filter(|&i| ok[i])
            .map(|i| accumulators[i].mean)
            .collect();
        let stds: Vec<f64> = (0..n_links)
            .filter(|&i| ok[i])
            .map(|i| accumulators[i].std())
            .collect();
        let fallback_mean = median(&means)?;
        let fallback_std = median(&stds)?;

        for index in 0..n_links {
            if ok[index] {
                self.baseline[index] = accumulators[index].mean;
                self.baseline_std[index] = accumulators[index].std();
            } else {
                self.baseline[index] = fallback_mean;
                self.baseline_std[index] = fallback_std;
            }
        }
        self.baseline_ok = ok;
        self.calibrated = true;
        self.clear_windows();
        Ok(())
    }

    /// Install a saved baseline (and optionally its raw null std).
    /// `ValueError` on a length mismatch or a non-finite or negative entry.
    /// Without a std every link is flagged; with one, `ok = baseline > 0`.
    pub fn set_baseline(&mut self, baseline: &[f64], baseline_std: Option<&[f64]>) -> Result<()> {
        let n_links = self.geometry.n_links();
        if baseline.len() != n_links {
            return Err(LocateError::ValueError(format!(
                "baseline has {} entries, geometry has {n_links} links",
                baseline.len()
            )));
        }
        if let Some(std) = baseline_std {
            if std.len() != n_links {
                return Err(LocateError::ValueError(format!(
                    "baseline_std has {} entries, geometry has {n_links} links",
                    std.len()
                )));
            }
        }
        for (index, &value) in baseline.iter().enumerate() {
            if !value.is_finite() {
                return Err(LocateError::ValueError(format!(
                    "baseline[{index}] is {}, not a finite number",
                    fmt_py(value)
                )));
            }
            if value < 0.0 {
                return Err(LocateError::ValueError(format!(
                    "baseline[{index}] is {}; a variance is not negative",
                    fmt_py(value)
                )));
            }
        }
        if let Some(std) = baseline_std {
            for (index, &value) in std.iter().enumerate() {
                if !value.is_finite() {
                    return Err(LocateError::ValueError(format!(
                        "baseline_std[{index}] is {}, not a finite number",
                        fmt_py(value)
                    )));
                }
                if value < 0.0 {
                    return Err(LocateError::ValueError(format!(
                        "baseline_std[{index}] is {}; a deviation is not negative",
                        fmt_py(value)
                    )));
                }
            }
        }

        self.baseline = baseline.to_vec();
        match baseline_std {
            None => {
                self.baseline_std = vec![0.0; n_links];
                self.baseline_ok = vec![false; n_links];
            }
            Some(std) => {
                self.baseline_std = std.to_vec();
                self.baseline_ok = baseline.iter().map(|&v| v > 0.0).collect();
            }
        }
        self.calibrated = true;
        Ok(())
    }

    /// Per-link empty-room variance, canonical order.
    pub fn baseline_vector(&self) -> Vec<f64> {
        self.baseline.clone()
    }

    /// Per-link null sigma of `y`, in `vector()`'s units: the raw spread in
    /// SUBTRACT, `std / baseline` (0 if the baseline is not positive) in
    /// NORMALIZE.
    pub fn baseline_std_vector(&self) -> Vec<f64> {
        if self.calibration_mode == CalibrationMode::Subtract {
            return self.baseline_std.clone();
        }
        self.baseline_std
            .iter()
            .zip(&self.baseline)
            .map(|(&std, &base)| if base > 0.0 { std / base } else { 0.0 })
            .collect()
    }

    /// Per-link spread of the empty-room windowed variance, unscaled.
    pub fn baseline_raw_std_vector(&self) -> Vec<f64> {
        self.baseline_std.clone()
    }

    /// Per link: is its calibration real (not imputed, not unusable)?
    pub fn calibrated_mask(&self) -> Vec<bool> {
        self.baseline_ok.clone()
    }

    // ---- lifecycle ----------------------------------------------------

    /// Drop all windowed data and signal state; calibration is kept unless
    /// `keep_calibration` is false. The counters are kept.
    pub fn reset(&mut self, keep_calibration: bool) {
        self.clear_windows();
        for state in self.states.iter_mut() {
            state.reset();
        }
        if !keep_calibration {
            let n_links = self.geometry.n_links();
            self.baseline = vec![0.0; n_links];
            self.baseline_std = vec![0.0; n_links];
            self.baseline_ok = vec![false; n_links];
            self.calibrated = false;
        }
    }
}
