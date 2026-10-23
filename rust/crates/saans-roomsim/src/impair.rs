//! ESP32 impairments of spec/roomsim.md §12 and the ESP-IDF buffer of §4.
//!
//! Ripple, AGC, noise and LTF merging are behavioural models with guessed
//! (SIM) magnitudes; the buffer layout is an assumption not yet verified on
//! hardware (§4). Phase 10 calibrates them.

use crate::config::Config;
use crate::numerics::{
    bin_index, cadd, cexpj, clamp, cmul, cos, cscale, cx, pow10, round_half_even, sin, Cx,
    Neumaier, PI, TWO_PI,
};
use saans_sim::rng::Rng;

/// Entries in the ESP-IDF buffer of an HT20 packet: LLTF then HT-LTF (§4).
pub const BUFFER_LEN: usize = 128;

/// Bin `k` of buffer entry `e`: `0..31, -32..-1` per LTF (FFT order).
pub fn buffer_k(e: usize) -> i32 {
    let i = (e % 64) as i32;
    if i < 32 {
        i
    } else {
        i - 64
    }
}

/// LLTF (legacy, HT20) active bins: `[-26, -1] U [1, 26]`.
pub fn lltf_active(k: i32) -> bool {
    k != 0 && (-26..=26).contains(&k)
}

/// HT-LTF (HT20) active bins: `[-28, -1] U [1, 28]`.
pub fn htltf_active(k: i32) -> bool {
    k != 0 && (-28..=28).contains(&k)
}

/// Whether buffer entry `e` carries a value in its LTF.
pub fn buffer_active(e: usize) -> bool {
    let k = buffer_k(e);
    if e < 64 {
        lltf_active(k)
    } else {
        htltf_active(k)
    }
}

/// All 128 `buffer_k` values (§13).
pub fn buffer_k_all() -> Vec<i32> {
    (0..BUFFER_LEN).map(buffer_k).collect()
}

/// All 128 `buffer_active` flags (§13).
pub fn buffer_active_all() -> Vec<bool> {
    (0..BUFFER_LEN).map(buffer_active).collect()
}

/// One node's radio, drawn from stream `500 + node_id` (§12.1).
#[derive(Debug, Clone, PartialEq)]
pub struct Device {
    pub alpha: Vec<f64>,
    pub beta: Vec<f64>,
    pub gamma: Vec<f64>,
    pub delta: Vec<f64>,
    pub g_dev_db: f64,
    /// `round(range(noise_floor_dbm))`, an integer.
    pub nf_dbm: f64,
    pub tau_dev: f64,
    /// `Rdev(k)` for the 64 bins, ascending `k`; `(1, 0)` with `ripple` off.
    pub rdev: Vec<Cx>,
}

impl Device {
    pub fn draw(cfg: &Config, node_id: u8) -> Self {
        let mut r = Rng::stream(cfg.seed, 500 + u64::from(node_id));
        let n = cfg.ripple_harmonics as usize;
        let (mut alpha, mut beta, mut gamma, mut delta) = (
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
            Vec::with_capacity(n),
        );
        for _ in 0..n {
            alpha.push(r.range(-cfg.ripple_amp, cfg.ripple_amp));
            beta.push(r.range(0.0, TWO_PI));
            gamma.push(r.range(-cfg.ripple_phase_rad, cfg.ripple_phase_rad));
            delta.push(r.range(0.0, TWO_PI));
        }
        let g_dev_db = r.range(cfg.device_gain_db.0, cfg.device_gain_db.1);
        let nf_dbm = round_half_even(r.range(cfg.noise_floor_dbm.0, cfg.noise_floor_dbm.1));
        let tau_dev = r.range(cfg.timing_offset_dev_s.0, cfg.timing_offset_dev_s.1);
        let mut d = Self {
            alpha,
            beta,
            gamma,
            delta,
            g_dev_db,
            nf_dbm,
            tau_dev,
            rdev: Vec::new(),
        };
        d.rdev = crate::numerics::bins()
            .map(|k| {
                if cfg.ripple {
                    d.ripple(k, cfg.ripple_period_bins)
                } else {
                    cx(1.0, 0.0)
                }
            })
            .collect();
        d
    }

    /// `Rdev(k) = cscale(cexpj(ph(k)), mag(k))` with ripple on (§12.1).
    pub fn ripple(&self, k: i32, period_bins: f64) -> Cx {
        let mut mag = Neumaier::new();
        let mut ph = Neumaier::new();
        for (i, (((a, b), g), d)) in self
            .alpha
            .iter()
            .zip(&self.beta)
            .zip(&self.gamma)
            .zip(&self.delta)
            .enumerate()
        {
            let m = (i + 1) as f64;
            let x = (TWO_PI * m) * f64::from(k) / period_bins;
            mag.add(a * cos(x + b));
            ph.add(g * sin(x + d));
        }
        cscale(cexpj(ph.value()), 1.0 + mag.value())
    }
}

/// The per-packet draws of a link, from stream `1000 + link_index` (§12.2).
#[derive(Debug, Clone, PartialEq)]
pub struct PacketDraws {
    pub u_loss: f64,
    pub u_jump: f64,
    pub u_dir: f64,
    pub phi0: f64,
    pub tau: f64,
    pub rj: f64,
    /// 256 `gauss()`: per buffer entry, `n_re` then `n_im`.
    pub noise: Vec<f64>,
}

impl PacketDraws {
    pub fn draw(rng: &mut Rng, cfg: &Config) -> Self {
        let u_loss = rng.uniform();
        let u_jump = rng.uniform();
        let u_dir = rng.uniform();
        let phi0 = rng.range(-PI, PI);
        let tau = rng.range(-cfg.timing_offset_s, cfg.timing_offset_s);
        let rj = rng.range(-cfg.rssi_jitter_db, cfg.rssi_jitter_db);
        let noise = (0..2 * BUFFER_LEN).map(|_| rng.gauss()).collect();
        Self {
            u_loss,
            u_jump,
            u_dir,
            phi0,
            tau,
            rj,
            noise,
        }
    }
}

/// The AGC random walk of §12.2, applied after the packet's draws.
pub fn agc_step(agc: f64, d: &PacketDraws, cfg: &Config) -> f64 {
    if d.u_jump < cfg.agc_jump_prob {
        let step = if d.u_dir < cfg.agc_up_prob {
            cfg.agc_step_db
        } else {
            -cfg.agc_step_db
        };
        clamp(agc + step, -cfg.agc_max_db, cfg.agc_max_db)
    } else {
        agc
    }
}

/// What §12.3 makes of one received packet.
#[derive(Debug, Clone, PartialEq)]
pub struct Processed {
    pub rssi: i32,
    pub rssi_true: f64,
    /// The 128 entries before step 9 (quantisation).
    pub buffer_pre: Vec<Cx>,
}

/// `Neumaier over the 56 HT-LTF bins of re*re + im*im, value() / 56.0`.
fn htltf_power(x: &[Cx]) -> f64 {
    let mut n = Neumaier::new();
    for k in crate::numerics::bins().filter(|&k| htltf_active(k)) {
        let v = x[bin_index(k)];
        n.add(v.re * v.re + v.im * v.im);
    }
    n.value() / 56.0
}

/// §12.3 steps 1-8 (and §12.5 with `impair` off) for the channel `h` (64
/// bins ascending) of link `tx -> rx` with AGC state `agc`.
pub fn process(
    h: &[Cx],
    rx: &Device,
    tx: &Device,
    d: &PacketDraws,
    agc: f64,
    cfg: &Config,
) -> Processed {
    let on = cfg.impair;
    // 1.
    let x: Vec<Cx> = if on {
        h.iter()
            .enumerate()
            .map(|(i, &hk)| cmul(cmul(hk, rx.rdev[i]), tx.rdev[i]))
            .collect()
    } else {
        h.to_vec()
    };
    // 2.
    let (rlo, rhi) = cfg.rssi_range_dbm;
    let p_h = htltf_power(h);
    let rssi_true = if p_h == 0.0 {
        rlo
    } else {
        cfg.tx_power_dbm + 10.0 * p_h.log10()
    };
    let jittered = if on { rssi_true + d.rj } else { rssi_true };
    let rssi = clamp(round_half_even(jittered), rlo, rhi) as i32;
    // 3.
    let p_x = htltf_power(&x);
    let c = if p_x == 0.0 {
        0.0
    } else {
        cfg.target_counts / p_x.sqrt()
    };
    let g = if on {
        pow10((rx.g_dev_db + agc) / 20.0)
    } else {
        1.0
    };
    // 4.
    let phi0 = if on && cfg.cpe { d.phi0 } else { 0.0 };
    let (tau, tau_dev) = if on { (d.tau, rx.tau_dev) } else { (0.0, 0.0) };
    let y: Vec<Cx> = crate::numerics::bins()
        .map(|k| {
            let ph = phi0 - ((TWO_PI * f64::from(k)) * cfg.spacing_hz) * (tau + tau_dev);
            cmul(cscale(x[bin_index(k)], c * g), cexpj(ph))
        })
        .collect();
    // 5.
    let sigma = if on {
        let a = (cfg.target_counts * g) * pow10(-(rssi_true - rx.nf_dbm) / 20.0);
        let thermal = if cfg.snr_noise { (a * a) / 2.0 } else { 0.0 };
        (cfg.chip_noise_counts * cfg.chip_noise_counts + thermal).sqrt()
    } else {
        0.0
    };
    // 6.
    let mut buf: Vec<Cx> = (0..BUFFER_LEN)
        .map(|e| {
            if buffer_active(e) {
                let n = cx(d.noise[2 * e], d.noise[2 * e + 1]);
                cadd(y[bin_index(buffer_k(e))], cscale(n, sigma))
            } else {
                cx(0.0, 0.0)
            }
        })
        .collect();
    // 7.
    if on && cfg.ltf_merge {
        let w = cfg.ltf_merge_weight;
        for e in 64..BUFFER_LEN {
            if lltf_active(buffer_k(e)) {
                buf[e] = cadd(cscale(buf[e - 64], w), cscale(buf[e], 1.0 - w));
            }
        }
    }
    // 8.
    if on && cfg.smoothing {
        smooth(&mut buf, 0, lltf_active, cfg);
        smooth(&mut buf, 64, htltf_active, cfg);
    }
    Processed {
        rssi,
        rssi_true,
        buffer_pre: buf,
    }
}

/// §12.3 step 8 within one LTF starting at buffer entry `base`.
fn smooth(buf: &mut [Cx], base: usize, active: fn(i32) -> bool, cfg: &Config) {
    let idx: Vec<usize> = crate::numerics::bins()
        .filter(|&k| active(k))
        .map(|k| {
            base + if k >= 0 {
                k as usize
            } else {
                (k + 64) as usize
            }
        })
        .collect();
    let (k0, k1, k2) = cfg.smoothing_kernel;
    let (e0, e1) = cfg.smoothing_edge;
    let n = idx.len();
    let part = |buf: &[Cx], f: fn(Cx) -> f64| -> Vec<f64> {
        let xs: Vec<f64> = idx.iter().map(|&e| f(buf[e])).collect();
        (0..n)
            .map(|i| {
                if i == 0 {
                    e0 * xs[0] + e1 * xs[1]
                } else if i == n - 1 {
                    e0 * xs[n - 1] + e1 * xs[n - 2]
                } else {
                    (k0 * xs[i - 1] + k1 * xs[i]) + k2 * xs[i + 1]
                }
            })
            .collect()
    };
    let re = part(buf, |c| c.re);
    let im = part(buf, |c| c.im);
    for (i, &e) in idx.iter().enumerate() {
        buf[e] = cx(re[i], im[i]);
    }
}

/// §12.3 step 9: `clamp(round(v), -128, 127)` as int8, per component.
pub fn quantize(v: f64) -> i8 {
    clamp(round_half_even(v), -128.0, 127.0) as i8
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn buffer_layout() {
        let k = buffer_k_all();
        assert_eq!(&k[..3], &[0, 1, 2]);
        assert_eq!(k[31], 31);
        assert_eq!(k[32], -32);
        assert_eq!(k[63], -1);
        assert_eq!(k[64], 0);
        assert_eq!(k[127], -1);
        let a = buffer_active_all();
        assert_eq!(a[..64].iter().filter(|&&x| x).count(), 52);
        assert_eq!(a[64..].iter().filter(|&&x| x).count(), 56);
        assert!(!a[0] && !a[64], "DC is null in both LTFs");
        assert!(!a[27] && a[64 + 27] && a[64 + 28] && !a[64 + 29]);
        // The firmware's first 56 entries: 44 carry a value, 12 are nulls (§4).
        assert_eq!(a[..56].iter().filter(|&&x| x).count(), 44);
    }

    #[test]
    fn quantize_rounds_half_to_even_and_saturates() {
        assert_eq!(quantize(2.5), 2);
        assert_eq!(quantize(3.5), 4);
        assert_eq!(quantize(-0.5), 0);
        assert_eq!(quantize(300.0), 127);
        assert_eq!(quantize(-300.0), -128);
    }

    #[test]
    fn agc_walk_is_clamped() {
        let cfg = Config::default();
        let up = PacketDraws {
            u_loss: 0.5,
            u_jump: 0.0,
            u_dir: 0.0,
            phi0: 0.0,
            tau: 0.0,
            rj: 0.0,
            noise: vec![],
        };
        let down = PacketDraws {
            u_dir: 0.9,
            ..up.clone()
        };
        let still = PacketDraws {
            u_jump: 0.5,
            ..up.clone()
        };
        assert_eq!(agc_step(0.0, &up, &cfg), 3.0);
        assert_eq!(agc_step(6.0, &up, &cfg), 6.0);
        assert_eq!(agc_step(-6.0, &down, &cfg), -6.0);
        assert_eq!(agc_step(1.0, &still, &cfg), 1.0);
    }
}
