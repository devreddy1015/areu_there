"""Exact port of `saans_sim::rng::Rng` (spec/roomsim.md §2).

xoshiro256** seeded through SplitMix64, with the same `new`, `stream`,
`next_u64`, `uniform`, `range`, `gauss` (Marsaglia polar, spare cached),
`exp` and `unit_vector` as rust/crates/saans-sim/src/rng.rs. Every value
is bit-identical to the Rust generator; the §2 test vectors check it.
"""

from __future__ import annotations

import math

__all__ = ["Rng", "splitmix64"]

_M64 = 0xFFFFFFFFFFFFFFFF
_INV_2_53 = 1.0 / float(1 << 53)


def splitmix64(state: int) -> tuple[int, int]:
    """One SplitMix64 step: returns `(new_state, output)`."""
    state = (state + 0x9E3779B97F4A7C15) & _M64
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & _M64
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & _M64
    return state, z ^ (z >> 31)


class Rng:
    """xoshiro256** with a cached Marsaglia spare, as saans_sim::rng::Rng."""

    __slots__ = ("s0", "s1", "s2", "s3", "spare")

    def __init__(self, seed: int) -> None:
        sm = seed & _M64
        sm, a = splitmix64(sm)
        sm, b = splitmix64(sm)
        sm, c = splitmix64(sm)
        sm, d = splitmix64(sm)
        self.s0, self.s1, self.s2, self.s3 = a, b, c, d
        self.spare: float | None = None

    @classmethod
    def stream(cls, seed: int, stream: int) -> "Rng":
        """An independent stream for (`seed`, `stream`)."""
        a = stream & _M64
        a, x = splitmix64(a)
        b = (seed & _M64) ^ x
        b, y = splitmix64(b)
        return cls(y)

    def next_u64(self) -> int:
        s0, s1, s2, s3 = self.s0, self.s1, self.s2, self.s3
        x = (s1 * 5) & _M64
        result = ((((x << 7) | (x >> 57)) & _M64) * 9) & _M64
        t = (s1 << 17) & _M64
        s2 ^= s0
        s3 ^= s1
        s1 ^= s2
        s0 ^= s3
        s2 ^= t
        self.s3 = ((s3 << 45) | (s3 >> 19)) & _M64
        self.s0, self.s1, self.s2 = s0, s1, s2
        return result

    def uniform(self) -> float:
        """Uniform in [0, 1), 53 bits."""
        return float(self.next_u64() >> 11) * _INV_2_53

    def range(self, lo: float, hi: float) -> float:
        """Uniform in [lo, hi): `lo + (hi - lo) * uniform()`."""
        return lo + (hi - lo) * self.uniform()

    def gauss(self) -> float:
        """Standard normal (Marsaglia polar method, spare cached)."""
        if self.spare is not None:
            z = self.spare
            self.spare = None
            return z
        while True:
            u = 2.0 * self.uniform() - 1.0
            v = 2.0 * self.uniform() - 1.0
            s = u * u + v * v
            if s > 0.0 and s < 1.0:
                m = math.sqrt(-2.0 * math.log(s) / s)
                self.spare = v * m
                return u * m

    def gauss_list(self, n: int) -> list:
        """`[gauss() for _ in range(n)]`, bit-identical, with the
        generator inlined (the per-packet noise draws are the hot path)."""
        out = []
        spare = self.spare
        s0, s1, s2, s3 = self.s0, self.s1, self.s2, self.s3
        M = _M64
        inv = _INV_2_53
        log = math.log
        sqrt = math.sqrt
        while len(out) < n:
            if spare is not None:
                out.append(spare)
                spare = None
                continue
            while True:
                # uniform() twice, inlined
                x = (s1 * 5) & M
                r = ((((x << 7) | (x >> 57)) & M) * 9) & M
                t = (s1 << 17) & M
                s2 ^= s0
                s3 ^= s1
                s1 ^= s2
                s0 ^= s3
                s2 ^= t
                s3 = ((s3 << 45) | (s3 >> 19)) & M
                u = 2.0 * (float(r >> 11) * inv) - 1.0
                x = (s1 * 5) & M
                r = ((((x << 7) | (x >> 57)) & M) * 9) & M
                t = (s1 << 17) & M
                s2 ^= s0
                s3 ^= s1
                s1 ^= s2
                s0 ^= s3
                s2 ^= t
                s3 = ((s3 << 45) | (s3 >> 19)) & M
                v = 2.0 * (float(r >> 11) * inv) - 1.0
                s = u * u + v * v
                if s > 0.0 and s < 1.0:
                    m = sqrt(-2.0 * log(s) / s)
                    spare = v * m
                    out.append(u * m)
                    break
        self.s0, self.s1, self.s2, self.s3 = s0, s1, s2, s3
        self.spare = spare
        return out

    def exp(self, mean: float) -> float:
        """Exponential with the given mean: `-mean * ln(1 - uniform())`."""
        return -mean * math.log(1.0 - self.uniform())

    def unit_vector(self) -> tuple[float, float]:
        """A random direction: `(cos a, sin a)`, `a = range(0, TAU)`."""
        a = self.range(0.0, 2.0 * math.pi)
        return (math.cos(a), math.sin(a))
