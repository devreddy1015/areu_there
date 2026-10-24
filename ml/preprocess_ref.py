"""Stdlib-only reference of Preprocessing v1 (spec/preprocess.md §3–§7).

The steps of `ml.preprocess.preprocess` written out with explicit loops,
`math` for every operation on data and nothing else (`fractions` checks
§7's bin count, an integer question), so that `spec/golden/generate.py` on
CPython 3.14, which has no numpy, can import it to make golden vectors, and
a later port can be held to them. It imports no numpy. The numpy version's
float64 values must match it (§10); this module is slow (about a second per
SIM window) and is not meant for training.

The arithmetic is §4's and is normative, operation by operation:

- `|h| = sqrt(re*re + im*im)` in float64, never `abs` or `hypot` (for int8
  sources exactly Locate's amplitude, spec/locate.md §4.3);
- every real sum is `math.fsum`, never builtin `sum` (whose rounding changed
  in CPython 3.12); a complex sum is its real and imaginary parts summed
  that way, each in ascending `k` (fsum's result does not depend on order);
- a complex product is spelled out: `(ar + j ai)(br + j bi)` is
  `(ar*br - ai*bi) + j (ar*bi + ai*br)`, `x * conj(y)` is
  `(xr*yr + xi*yi) + j (xi*yr - xr*yi)`, and `x * exp(-j th)` is
  `(xr*cos th + xi*sin th) + j (xi*cos th - xr*sin th)`, left to right,
  no fused multiply-add;
- an angle is `math.atan2(im, re)` with a result of `-π` replaced by `π`.

Input is a window as plain Python data, the canonical record of §2 without
its provenance (`ml.preprocess.reference_input` converts a `CsiWindow`):

    {"csi":       L lists of P lists of 52 (re, im) float pairs,
     "received":  L lists of P bools,
     "carried":   52 bools,
     "t_rel_ns":  L lists of P ints (packet time minus the window's end),
     "rssi":      L lists of P floats (dBm, 0 where absent),
     "rssi_mask": L lists of P bools,
     "length_ns": int}

The output holds §9's computed fields as nested lists of Python floats,
ints and bools, before any float32 cast. The data's label (SIM or REAL)
is the caller's to carry; nothing here reads or changes it.
"""

from __future__ import annotations

import math
from fractions import Fraction

__all__ = [
    "GRID_K", "SHARED44", "REF_BINS", "PHASE_MIN_AMP", "angle", "magnitude", "reference_bins",
    "consecutive_pairs", "bin_layout", "normalise", "sanitise", "mean", "zscore", "preprocess_ref",
]

#: §3: the 52 LLTF bins of HT20 in ascending k.
GRID_K = tuple(range(-26, 0)) + tuple(range(1, 27))
#: §3 / decision 3: the reference bins every source carries (k = -26..-9, 1..26).
SHARED44 = tuple(range(-26, -8)) + tuple(range(1, 27))
REF_BINS = ("shared44", "carried")
#: §5: a bin whose normalised amplitude is below this has no phase (20 dB below the packet's mean).
PHASE_MIN_AMP = 0.1
_NS_PER_S = 1_000_000_000


def angle(im: float, re: float) -> float:
    """§4's angle: `math.atan2(im, re)` in (-π, π], a result of -π replaced
    by π (so the sign of a zero imaginary part never changes an angle)."""
    a = math.atan2(im, re)
    if a == -math.pi:
        return math.pi
    return a


def magnitude(re: float, im: float) -> float:
    """§4's `|h| = sqrt(re*re + im*im)`, never `hypot`."""
    return math.sqrt(re * re + im * im)


def reference_bins(ref_bins: str, carried: list) -> list:
    """§3: the grid indices of the reference bins R, ascending. "shared44"
    is k = -26..-9, 1..26 and refuses a source that does not carry them
    all; "carried" is every carried bin (at least one)."""
    if ref_bins == "shared44":
        want = [GRID_K.index(k) for k in SHARED44]
        missing = [GRID_K[b] for b in want if not carried[b]]
        if missing:
            raise ValueError(f"ref_bins 'shared44': the source does not carry k = {missing} (§3)")
        return want
    if ref_bins == "carried":
        ref = [b for b in range(52) if carried[b]]
        if not ref:
            raise ValueError("ref_bins 'carried': the source carries no grid bin (§3)")
        return ref
    raise ValueError(f"ref_bins {ref_bins!r} is not one of {REF_BINS}")


def consecutive_pairs(ref: list) -> list:
    """§5: the pairs (b, b + 1) of reference bins whose k are consecutive,
    ascending: never across the DC hole (-1 -> 1) or a gap in R (the
    firmware's -9 -> 1)."""
    inside = set(ref)
    return [(b, b + 1) for b in ref if b + 1 in inside and GRID_K[b + 1] == GRID_K[b] + 1]


def bin_layout(bin_hz, length_ns: int) -> tuple:
    """§7: `(nb, w_ns)`. `nb = bin_hz * length_s` must be an exact integer
    (the rate is read as the decimal its float prints as, so 0.1 is 1/10)
    and `length_ns` a multiple of `nb`; anything else is refused."""
    if isinstance(bin_hz, bool) or not isinstance(bin_hz, (int, float)) or not math.isfinite(bin_hz) or bin_hz <= 0:
        raise ValueError(f"bin_hz {bin_hz!r} is not a positive finite number (§7)")
    nb = Fraction(repr(float(bin_hz))) * Fraction(length_ns, _NS_PER_S)
    if nb.denominator != 1 or nb < 1:
        raise ValueError(f"bin_hz {bin_hz!r} x length {length_ns} ns = {nb} bins, not a positive integer (§7)")
    nb = int(nb)
    if length_ns % nb:
        raise ValueError(f"length {length_ns} ns is not a multiple of {nb} bins (§7)")
    return nb, length_ns // nb


def normalise(h: list, ref: list, carried_idx: list) -> tuple:
    """§4 for one received packet: `(g, a1, hp)`. `g = fsum(|h_b| over R) /
    |R|`; when `g == 0` (every reference bin exactly zero) the packet is
    dropped and `a1`, `hp` are None. Otherwise, for every carried bin,
    `a1[b] = |h_b| / g` and `hp[b] = (re / g, im / g)` (0 elsewhere)."""
    mag = [0.0] * 52
    for b in carried_idx:
        re, im = h[b]
        mag[b] = magnitude(re, im)
    g = math.fsum([mag[b] for b in ref]) / len(ref)
    if g == 0.0:
        return g, None, None
    a1 = [0.0] * 52
    hp = [(0.0, 0.0)] * 52
    for b in carried_idx:
        re, im = h[b]
        a1[b] = mag[b] / g
        hp[b] = (re / g, im / g)
    return g, a1, hp


def sanitise(hp: list, ref: list, pairs: list, carried_idx: list):
    """§5 steps 1-3 for one packet: `{b: r_b}` for every carried bin, `r_b`
    the residual `h'_b * exp(-j (a k_b + c))` as a (re, im) pair, or None
    when the packet has no phase (`S == 0` or `C == 0`). `a` and `c` stay
    here: they are never output."""
    s_re = []
    s_im = []
    for b0, b1 in pairs:                       # S = sum h'_{k+1} * conj(h'_k)
        x0r, x0i = hp[b0]
        x1r, x1i = hp[b1]
        s_re.append(x1r * x0r + x1i * x0i)
        s_im.append(x1i * x0r - x1r * x0i)
    sr = math.fsum(s_re)
    si = math.fsum(s_im)
    if sr == 0.0 and si == 0.0:                # guard: no slope, no phase
        return None
    a = angle(si, sr)
    c_re = []
    c_im = []
    for b in ref:                              # C = sum h'_b * exp(-j a k_b)
        xr, xi = hp[b]
        th = a * float(GRID_K[b])
        cs = math.cos(th)
        sn = math.sin(th)
        c_re.append(xr * cs + xi * sn)
        c_im.append(xi * cs - xr * sn)
    cr = math.fsum(c_re)
    ci = math.fsum(c_im)
    if cr == 0.0 and ci == 0.0:                # guard: no offset, no phase
        return None
    c = angle(ci, cr)
    out = {}
    for b in carried_idx:                      # r_b = h'_b * exp(-j (a k_b + c))
        xr, xi = hp[b]
        th = a * float(GRID_K[b]) + c
        cs = math.cos(th)
        sn = math.sin(th)
        out[b] = (xr * cs + xi * sn, xi * cs - xr * sn)
    return out


def mean(values: list) -> float:
    """A bin's mean (§7): `fsum(values) / n`, except that values all equal
    (compared exactly) give that value. (`fsum([x] * 3) / 3` is not always
    `x`: without this, a bin of identical packets would differ from its
    neighbours in the last bit and defeat §6's equal-values guard.)"""
    first = values[0]
    if all(v == first for v in values):
        return first
    return math.fsum(values) / len(values)


def _unit(re: float, im: float):
    """`(re, im) / |(re, im)|`, or None for a zero vector (the guard)."""
    m = magnitude(re, im)
    if m == 0.0:
        return None
    return (re / m, im / m)


def zscore(xs: list) -> tuple:
    """§6 on one (link, bin)'s `n >= 2` values: `(z list, mu, sigma)`.
    Values all equal (compared exactly, before any arithmetic) give that
    value, sigma 0 and z 0; otherwise `mu = fsum / n`, `sigma = sqrt(fsum((x
    - mu)^2) / (n - 1))`, `z = (x - mu) / sigma`, and a sigma that still
    comes out 0 (only by underflow of the squares) gives z 0."""
    n = len(xs)
    first = xs[0]
    if all(v == first for v in xs):
        return [0.0] * n, first, 0.0
    mu = math.fsum(xs) / n
    sigma = math.sqrt(math.fsum([(v - mu) * (v - mu) for v in xs]) / (n - 1))
    if sigma == 0.0:
        return [0.0] * n, mu, 0.0
    return [(v - mu) / sigma for v in xs], mu, sigma


def preprocess_ref(win: dict, phase: bool = False, bin_hz=None, ref_bins: str = "shared44",
                   min_packets: int = 3) -> dict:
    """§3–§7 on one window (the module docstring gives the input), in §1's
    order: subcarrier mask, gain normalisation, phase sanitisation (when
    `phase`), binning (when `bin_hz`), per-link z-score. Returns float64
    values, before the float32 cast: `k_grid`, `carried`, `ref`, `amp`
    [L][52][T], `mu` and `sigma` [L][52], `n_packets` [L], `sample_mask`,
    `gain_db`, `rssi`, `rssi_mask`, `t_rel` [L][T], `link_mask` [L], with
    `bin_hz` `bin_count` [L][T], with `phase` `phase` [L][52][T][2] and
    `phase_mask` [L][52][T], and the counts `zero_gain_packets`,
    `links_dropped`. Every masked or uncarried entry is 0 (False); no value
    is NaN."""
    if isinstance(min_packets, bool) or not isinstance(min_packets, int) or min_packets < 2:
        raise ValueError(f"min_packets {min_packets!r} is not an integer >= 2 (§6)")
    csi = win["csi"]
    received = win["received"]
    carried = [bool(c) for c in win["carried"]]
    t_rel_ns = win["t_rel_ns"]
    rssi_in = win["rssi"]
    rssi_mask_in = win["rssi_mask"]
    length_ns = win["length_ns"]
    n_links = len(csi)
    n_pk = len(csi[0]) if n_links else 0

    # §3: the grid, the carried bins and the reference bins R.
    carried_idx = [b for b in range(52) if carried[b]]
    ref = reference_bins(ref_bins, carried)
    ref_mask = [b in ref for b in range(52)]
    pairs = consecutive_pairs(ref)
    if bin_hz is not None:
        nb, w_ns = bin_layout(bin_hz, length_ns)

    # §4 (and §5): per received packet.
    zero_gain = 0
    valid = [[False] * n_pk for _ in range(n_links)]
    a1 = [[None] * n_pk for _ in range(n_links)]
    gdb = [[0.0] * n_pk for _ in range(n_links)]
    resid = [[None] * n_pk for _ in range(n_links)]      # {b: r_b} per packet with phase
    for l in range(n_links):
        for p in range(n_pk):
            if not received[l][p]:
                continue
            g, a, hp = normalise(csi[l][p], ref, carried_idx)
            if a is None:                                # guard: g == 0, dropped
                zero_gain += 1
                continue
            valid[l][p] = True
            a1[l][p] = a
            gdb[l][p] = 20.0 * math.log10(g)
            if phase:
                r = sanitise(hp, ref, pairs, carried_idx)
                if r is not None:
                    keep = {}
                    for b in carried_idx:
                        hr, hi = hp[b]
                        if (hr != 0.0 or hi != 0.0) and a[b] >= PHASE_MIN_AMP:
                            keep[b] = r[b]
                    resid[l][p] = keep

    # §7: samples are packets, or bins of packets when bin_hz is set.
    if bin_hz is None:
        n_t = n_pk
        present = valid
        x = [[a1[l][p] if valid[l][p] else [0.0] * 52 for p in range(n_pk)] for l in range(n_links)]
        gain = [[gdb[l][p] if valid[l][p] else 0.0 for p in range(n_pk)] for l in range(n_links)]
        rssi = [[rssi_in[l][p] if valid[l][p] and rssi_mask_in[l][p] else 0.0 for p in range(n_pk)]
                for l in range(n_links)]
        rmask = [[bool(valid[l][p] and rssi_mask_in[l][p]) for p in range(n_pk)] for l in range(n_links)]
        t_rel = [[t_rel_ns[l][p] / 1e9 if valid[l][p] else 0.0 for p in range(n_pk)] for l in range(n_links)]
        ph = None
        if phase:
            ph = [[{} for _ in range(n_pk)] for _ in range(n_links)]
            for l in range(n_links):
                for p in range(n_pk):
                    for b, (rr, ri) in (resid[l][p] or {}).items():
                        u = _unit(rr, ri)
                        if u is not None:
                            ph[l][p][b] = u
        count = None
    else:
        n_t = nb
        present = [[False] * nb for _ in range(n_links)]
        x = [[[0.0] * 52 for _ in range(nb)] for _ in range(n_links)]
        gain = [[0.0] * nb for _ in range(n_links)]
        rssi = [[0.0] * nb for _ in range(n_links)]
        rmask = [[False] * nb for _ in range(n_links)]
        # The bin centre, -length + (i + 1/2) w, as (2 (i w - length) + w) / 2e9: one rounding.
        t_rel = [[(2 * (i * w_ns - length_ns) + w_ns) / 2e9 for i in range(nb)] for _ in range(n_links)]
        count = [[0] * nb for _ in range(n_links)]
        ph = [[{} for _ in range(nb)] for _ in range(n_links)] if phase else None
        for l in range(n_links):
            members = [[] for _ in range(nb)]
            for p in range(n_pk):
                if valid[l][p]:
                    members[(t_rel_ns[l][p] + length_ns) // w_ns].append(p)   # integer bins
            for i in range(nb):
                ps = members[i]
                if not ps:                                # an empty bin is masked
                    continue
                n = len(ps)
                present[l][i] = True
                count[l][i] = n
                for b in carried_idx:                     # the mean of the amplitudes, never |complex mean|
                    x[l][i][b] = mean([a1[l][p][b] for p in ps])
                gain[l][i] = mean([gdb[l][p] for p in ps])
                with_rssi = [rssi_in[l][p] for p in ps if rssi_mask_in[l][p]]
                if with_rssi:
                    rssi[l][i] = mean(with_rssi)
                    rmask[l][i] = True
                if phase:
                    for b in carried_idx:
                        rs = [resid[l][p][b] for p in ps if resid[l][p] is not None and b in resid[l][p]]
                        u = _unit(math.fsum([r[0] for r in rs]), math.fsum([r[1] for r in rs]))
                        if u is not None:                 # masked when the sum is 0
                            ph[l][i][b] = u

    # §6: per-link z-score over the link's samples.
    amp = [[[0.0] * n_t for _ in range(52)] for _ in range(n_links)]
    mu = [[0.0] * 52 for _ in range(n_links)]
    sigma = [[0.0] * 52 for _ in range(n_links)]
    n_packets = [0] * n_links
    link_mask = [False] * n_links
    sample_mask = [[False] * n_t for _ in range(n_links)]
    dropped = 0
    for l in range(n_links):
        ts = [t for t in range(n_t) if present[l][t]]
        n_packets[l] = len(ts)
        if len(ts) < min_packets:                         # guard: the link is not used
            dropped += 1
            continue
        link_mask[l] = True
        for t in ts:
            sample_mask[l][t] = True
        for b in carried_idx:
            z, m, s = zscore([x[l][t][b] for t in ts])
            mu[l][b] = m
            sigma[l][b] = s
            for t, v in zip(ts, z):
                amp[l][b][t] = v

    # Per-sample outputs are 0 (False) wherever sample_mask is false.
    def masked(rows, zero):
        return [[rows[l][t] if sample_mask[l][t] else zero for t in range(n_t)] for l in range(n_links)]

    out = {
        "k_grid": list(GRID_K), "carried": carried, "ref": ref_mask,
        "amp": amp, "mu": mu, "sigma": sigma, "n_packets": n_packets,
        "sample_mask": sample_mask, "link_mask": link_mask,
        "gain_db": masked(gain, 0.0), "rssi": masked(rssi, 0.0), "rssi_mask": masked(rmask, False),
        "t_rel": t_rel if bin_hz is not None else masked(t_rel, 0.0),
        "zero_gain_packets": zero_gain, "links_dropped": dropped,
    }
    if bin_hz is not None:
        out["bin_count"] = masked(count, 0)
    if phase:
        pv = [[[[0.0, 0.0] for _ in range(n_t)] for _ in range(52)] for _ in range(n_links)]
        pm = [[[False] * n_t for _ in range(52)] for _ in range(n_links)]
        for l in range(n_links):
            for t in range(n_t):
                if not sample_mask[l][t]:
                    continue
                for b, (c, s) in ph[l][t].items():
                    pv[l][b][t] = [c, s]
                    pm[l][b][t] = True
        out["phase"] = pv
        out["phase_mask"] = pm
    return out
