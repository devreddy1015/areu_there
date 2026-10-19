//! RBJ constant-peak-gain bandpass biquad, transposed direct form II.

use std::f64::consts::PI;

#[derive(Debug, Clone)]
pub struct BiquadBandpass {
    pub f_low: f64,
    pub f_high: f64,
    pub sample_rate_hz: f64,
    b0: f64,
    b1: f64,
    b2: f64,
    a1: f64,
    a2: f64,
    s1: f64,
    s2: f64,
}

impl BiquadBandpass {
    /// Panics on an invalid band, like the reference raises ValueError:
    /// a misconfigured filter is a programming error, not a runtime state.
    pub fn new(f_low: f64, f_high: f64, sample_rate_hz: f64) -> Self {
        assert!(
            0.0 < f_low && f_low < f_high,
            "need 0 < f_low < f_high, got {f_low}, {f_high}"
        );
        let nyquist = sample_rate_hz / 2.0;
        assert!(
            f_high < nyquist,
            "f_high ({f_high} Hz) must be below Nyquist ({nyquist} Hz)"
        );
        let f0 = (f_low * f_high).sqrt();
        let q = f0 / (f_high - f_low);
        let w0 = 2.0 * PI * f0 / sample_rate_hz;
        let alpha = w0.sin() / (2.0 * q);
        let cos_w0 = w0.cos();
        let a0 = 1.0 + alpha;
        Self {
            f_low,
            f_high,
            sample_rate_hz,
            b0: alpha / a0,
            b1: 0.0,
            b2: -alpha / a0,
            a1: (-2.0 * cos_w0) / a0,
            a2: (1.0 - alpha) / a0,
            s1: 0.0,
            s2: 0.0,
        }
    }

    pub fn center_hz(&self) -> f64 {
        (self.f_low * self.f_high).sqrt()
    }

    /// Filter one sample; a non-finite input or output resets instead of
    /// poisoning the recurrence forever.
    pub fn process(&mut self, x: f64) -> f64 {
        if !x.is_finite() {
            self.reset();
            return 0.0;
        }
        let y = self.b0 * x + self.s1;
        self.s1 = self.b1 * x - self.a1 * y + self.s2;
        self.s2 = self.b2 * x - self.a2 * y;
        if !y.is_finite() {
            self.reset();
            return 0.0;
        }
        y
    }

    pub fn reset(&mut self) {
        self.s1 = 0.0;
        self.s2 = 0.0;
    }
}
