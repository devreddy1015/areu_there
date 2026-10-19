//! Small deterministic PRNG: xoshiro256** seeded through SplitMix64.
//!
//! No dependencies, same stream on every platform. Independent streams
//! (room layout, timeline, each link) are derived from the scenario seed
//! by hashing, so e.g. changing the loss probability never changes the
//! noise a link sees, and adding a link never changes another link.

use crate::scenario::Point;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

#[derive(Debug, Clone)]
pub struct Rng {
    s: [u64; 4],
    spare: Option<f64>,
}

impl Rng {
    pub fn new(seed: u64) -> Self {
        let mut sm = seed;
        let s = [
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
            splitmix64(&mut sm),
        ];
        Self { s, spare: None }
    }

    /// An independent stream for (`seed`, `stream`).
    pub fn stream(seed: u64, stream: u64) -> Self {
        let mut a = stream;
        let mut b = seed ^ splitmix64(&mut a);
        Self::new(splitmix64(&mut b))
    }

    pub fn next_u64(&mut self) -> u64 {
        let s = &mut self.s;
        let result = s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = s[1] << 17;
        s[2] ^= s[0];
        s[3] ^= s[1];
        s[1] ^= s[2];
        s[0] ^= s[3];
        s[2] ^= t;
        s[3] = s[3].rotate_left(45);
        result
    }

    /// Uniform in [0, 1), 53 bits.
    pub fn uniform(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 * (1.0 / (1u64 << 53) as f64)
    }

    /// Uniform in [lo, hi).
    pub fn range(&mut self, lo: f64, hi: f64) -> f64 {
        lo + (hi - lo) * self.uniform()
    }

    /// Standard normal (Marsaglia polar method, spare cached).
    pub fn gauss(&mut self) -> f64 {
        if let Some(z) = self.spare.take() {
            return z;
        }
        loop {
            let u = 2.0 * self.uniform() - 1.0;
            let v = 2.0 * self.uniform() - 1.0;
            let s = u * u + v * v;
            if s > 0.0 && s < 1.0 {
                let m = (-2.0 * s.ln() / s).sqrt();
                self.spare = Some(v * m);
                return u * m;
            }
        }
    }

    /// Exponential with the given mean.
    pub fn exp(&mut self, mean: f64) -> f64 {
        -mean * (1.0 - self.uniform()).ln()
    }

    /// A random direction.
    pub fn unit_vector(&mut self) -> Point {
        let a = self.range(0.0, std::f64::consts::TAU);
        (a.cos(), a.sin())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_streams_differ() {
        let a: Vec<u64> = {
            let mut r = Rng::new(7);
            (0..4).map(|_| r.next_u64()).collect()
        };
        let b: Vec<u64> = {
            let mut r = Rng::new(7);
            (0..4).map(|_| r.next_u64()).collect()
        };
        assert_eq!(a, b);
        let mut s0 = Rng::stream(7, 0);
        let mut s1 = Rng::stream(7, 1);
        assert_ne!(s0.next_u64(), s1.next_u64());
    }

    #[test]
    fn moments() {
        let mut r = Rng::new(1);
        let n = 200_000;
        let (mut su, mut sg, mut sg2, mut se) = (0.0, 0.0, 0.0, 0.0);
        for _ in 0..n {
            let u = r.uniform();
            assert!((0.0..1.0).contains(&u));
            su += u;
            let g = r.gauss();
            sg += g;
            sg2 += g * g;
            se += r.exp(3.0);
        }
        let n = n as f64;
        assert!((su / n - 0.5).abs() < 0.005);
        assert!((sg / n).abs() < 0.01);
        assert!((sg2 / n - 1.0).abs() < 0.02);
        assert!((se / n - 3.0).abs() < 0.05);
    }
}
