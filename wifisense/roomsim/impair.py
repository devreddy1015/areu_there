"""ESP32 impairments (spec/roomsim.md §4, §12): the 128-entry buffer, the
per-node device draws, the per-packet draws and AGC walk, and the buffer
of one received packet.

These are behavioural models with guessed magnitudes (SIM) for Phase 10
to calibrate; the buffer layout itself is an unverified assumption (§4).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

from .numerics import PI, TWO_PI, Neumaier, cadd, cexpj, clamp, cmul, cscale

__all__ = [
    "BUFFER_K", "BUFFER_ACTIVE", "BUFFER_LEN", "LLTF_BINS", "HTLTF_BINS",
    "lltf_active", "htltf_active", "Device", "draw_device", "PacketDraws", "LinkImpairment",
    "received_buffer", "CSI_MAGIC", "datagram",
]

BUFFER_LEN = 128


def lltf_active(k: int) -> bool:
    """LLTF (legacy, HT20): k in [-26, -1] U [1, 26]."""
    return 1 <= abs(k) <= 26


def htltf_active(k: int) -> bool:
    """HT-LTF (HT20): k in [-28, -1] U [1, 28]."""
    return 1 <= abs(k) <= 28


def _fft_order():
    return [i if i < 32 else i - 64 for i in range(64)]


# §13: buffer_k = 0..31, -32..-1, 0..31, -32..-1 ; buffer_active per LTF.
BUFFER_K = tuple(_fft_order() + _fft_order())
BUFFER_ACTIVE = tuple([lltf_active(k) for k in _fft_order()] + [htltf_active(k) for k in _fft_order()])
LLTF_BINS = tuple(k for k in range(-32, 32) if lltf_active(k))
HTLTF_BINS = tuple(k for k in range(-32, 32) if htltf_active(k))

CSI_MAGIC = 0xC5110001
_HEADER = struct.Struct("<IBBHIIbbBB")


@dataclass
class Device:
    """§12.1 draws of one node and its ripple `Rdev(k)` (64 bins, ascending k)."""

    node_id: int
    alpha: tuple
    beta: tuple
    gamma: tuple
    delta: tuple
    g_dev_db: float
    nf_dbm: int
    tau_dev: float
    mag: tuple
    ph: tuple
    rdev: tuple


def draw_device(cfg, node_id: int, rng) -> Device:
    """Node `node_id`'s device, from stream `500 + node_id` (§12.1)."""
    alpha, beta, gamma, delta = [], [], [], []
    for _m in range(1, cfg.ripple_harmonics + 1):
        alpha.append(rng.range(-cfg.ripple_amp, cfg.ripple_amp))
        beta.append(rng.range(0.0, TWO_PI))
        gamma.append(rng.range(-cfg.ripple_phase_rad, cfg.ripple_phase_rad))
        delta.append(rng.range(0.0, TWO_PI))
    g_dev_db = rng.range(cfg.device_gain_db[0], cfg.device_gain_db[1])
    nf_dbm = round(rng.range(cfg.noise_floor_dbm[0], cfg.noise_floor_dbm[1]))
    tau_dev = rng.range(cfg.timing_offset_dev_s[0], cfg.timing_offset_dev_s[1])
    mags, phs, rdev = [], [], []
    for k in range(-32, 32):
        sm = Neumaier()
        sp = Neumaier()
        for i in range(cfg.ripple_harmonics):
            m = i + 1
            x = (TWO_PI * float(m)) * float(k) / cfg.ripple_period_bins
            sm.add(alpha[i] * math.cos(x + beta[i]))
            sp.add(gamma[i] * math.sin(x + delta[i]))
        mag = 1.0 + sm.value()
        ph = sp.value()
        mags.append(mag)
        phs.append(ph)
        rdev.append(cscale(cexpj(ph), mag) if cfg.ripple else (1.0, 0.0))
    return Device(node_id, tuple(alpha), tuple(beta), tuple(gamma), tuple(delta),
                  g_dev_db, nf_dbm, tau_dev, tuple(mags), tuple(phs), tuple(rdev))


@dataclass
class PacketDraws:
    """§12.2 draws of one transmitted packet on one link."""

    u_loss: float
    u_jump: float
    u_dir: float
    phi0: float
    tau: float
    rj: float
    noise: list   # 256 gauss: n_re, n_im per buffer entry, buffer order


class LinkImpairment:
    """Per-link stream `1000 + link_index` and AGC state (§12.2)."""

    def __init__(self, cfg, rng, p_loss: float) -> None:
        self.cfg = cfg
        self.rng = rng
        self.p_loss = p_loss
        self.agc = cfg.agc_initial_db

    def next_packet(self):
        """Draw one packet's values and step the AGC. Returns
        `(draws, lost)`; `lost` already honours `impair` (§12.5)."""
        cfg = self.cfg
        rng = self.rng
        u_loss = rng.uniform()
        u_jump = rng.uniform()
        u_dir = rng.uniform()
        phi0 = rng.range(-PI, PI)
        tau = rng.range(-cfg.timing_offset_s, cfg.timing_offset_s)
        rj = rng.range(-cfg.rssi_jitter_db, cfg.rssi_jitter_db)
        noise = rng.gauss_list(256)
        if u_jump < cfg.agc_jump_prob:
            step = cfg.agc_step_db if u_dir < cfg.agc_up_prob else -cfg.agc_step_db
            self.agc = clamp(self.agc + step, -cfg.agc_max_db, cfg.agc_max_db)
        lost = cfg.impair and u_loss < self.p_loss
        return PacketDraws(u_loss, u_jump, u_dir, phi0, tau, rj, noise), lost


def _p_mean(H, bins) -> tuple:
    acc = Neumaier()
    for k in bins:
        z = H[k + 32]
        acc.add(z[0] * z[0] + z[1] * z[1])
    return acc.value() / 56.0


def received_buffer(cfg, H, dev_tx: Device, dev_rx: Device, draws: PacketDraws, agc: float):
    """§12.3 (and §12.5 when `impair` is off): returns
    `(rssi_true, rssi, buffer_pre, buffer)`; `buffer` is int8 pairs when
    `quantize`, else the same floats as `buffer_pre`."""
    imp = cfg.impair
    if imp:
        X = [cmul(cmul(H[i], dev_rx.rdev[i]), dev_tx.rdev[i]) for i in range(64)]
    else:
        X = H
    P_H = _p_mean(H, HTLTF_BINS)
    lo, hi = cfg.rssi_range_dbm
    rssi_true = cfg.tx_power_dbm + 10.0 * math.log10(P_H) if P_H != 0.0 else float(lo)
    rj = draws.rj if imp else 0.0
    rssi = int(clamp(round(rssi_true + rj) if imp else round(rssi_true), lo, hi))
    P_X = _p_mean(X, HTLTF_BINS)
    c = cfg.target_counts / math.sqrt(P_X) if P_X != 0.0 else 0.0
    if imp:
        g = 10.0 ** ((dev_rx.g_dev_db + agc) / 20.0)
        phi0 = draws.phi0 if cfg.cpe else 0.0
        tau = draws.tau
        tau_dev = dev_rx.tau_dev
    else:
        g = 1.0
        phi0 = tau = tau_dev = 0.0
    cg = c * g
    tt = tau + tau_dev
    Y = []
    for i in range(64):
        k = i - 32
        Y.append(cmul(cscale(X[i], cg), cexpj(phi0 - ((TWO_PI * float(k)) * cfg.spacing_hz) * tt)))
    if imp:
        a = (cfg.target_counts * g) * 10.0 ** (-(rssi_true - float(dev_rx.nf_dbm)) / 20.0)
        sigma = math.sqrt(cfg.chip_noise_counts * cfg.chip_noise_counts + ((a * a) / 2.0 if cfg.snr_noise else 0.0))
    else:
        sigma = 0.0
    n = draws.noise
    buf = []
    for e in range(BUFFER_LEN):
        k = BUFFER_K[e]
        if BUFFER_ACTIVE[e]:
            buf.append(cadd(Y[k + 32], cscale((n[2 * e], n[2 * e + 1]), sigma)))
        else:
            buf.append((0.0, 0.0))
    if imp and cfg.ltf_merge:
        w = cfg.ltf_merge_weight
        for e in range(64, BUFFER_LEN):
            if lltf_active(BUFFER_K[e]):
                buf[e] = cadd(cscale(buf[e - 64], w), cscale(buf[e], 1.0 - w))
    if imp and cfg.smoothing:
        for base, bins in ((0, LLTF_BINS), (64, HTLTF_BINS)):
            idx = [base + (k if k >= 0 else k + 64) for k in bins]
            buf = _smooth(buf, idx, cfg.smoothing_kernel, cfg.smoothing_edge)
    pre = buf
    if cfg.quantize:
        out = [(int(clamp(round(re), -128, 127)), int(clamp(round(im), -128, 127))) for re, im in pre]
    else:
        out = list(pre)
    return rssi_true, rssi, pre, out


def _smooth(buf, idx, kern, edge):
    """§12.3 step 8 over the entries `idx` (active bins, ascending k)."""
    k0, k1, k2 = kern
    e0, e1 = edge
    x = [buf[i] for i in idx]
    n = len(x)
    new = list(buf)
    for c in range(2):
        for i in range(n):
            if i == 0:
                y = e0 * x[0][c] + e1 * x[1][c]
            elif i == n - 1:
                y = e0 * x[n - 1][c] + e1 * x[n - 2][c]
            else:
                y = (k0 * x[i - 1][c] + k1 * x[i][c]) + k2 * x[i + 1][c]
            old = new[idx[i]]
            new[idx[i]] = (y, old[1]) if c == 0 else (old[0], y)
    return new


def datagram(cfg, rx: int, seq: int, rssi: int, nf_dbm: int, buffer) -> bytes:
    """§12.4: exactly `csi_capture.cpp::serialize_raw`: a 20-byte LE header
    and the first `fw_subcarriers` entries as (i8 imag, i8 real)."""
    n = cfg.fw_subcarriers
    freq_mhz = 2412 + 5 * (cfg.channel - 1)
    head = _HEADER.pack(CSI_MAGIC, rx, 1, n, freq_mhz, seq & 0xFFFFFFFF, rssi, nf_dbm, 0, 0)
    body = bytearray()
    for e in range(n):
        re, im = buffer[e]
        body += struct.pack("<bb", im, re)
    return head + bytes(body)
