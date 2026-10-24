"""The simulated phone (spec/locate.md §6-§7): RTT and RSSI to every node, fixes, filters.

**SIM only** (ADR-008 decision 16). Every public entry point refuses any
label but `"SIM"`. A phone ranging to nodes is a cooperative device
measuring its own distances; it is not the multilateration of people that
CAMPUS_SENSING_PROMPT.md §3 bans (people are located only by the links they
disturb, `baseline.py` and the RTI).

Everything here works on plain data, so the golden generators and the Rust
port can feed it recorded inputs:

  * nodes      `(node_id, x, y, z)`, any order (used in id order);
  * walls      `(ax, ay, bx, by, h)` — Room Sim's `W`, implied walls included;
  * boxes      `(cx, cy, cz, sl, sb, sh, cs, sn)` — objects that are not a
               door or window, with `cs`, `sn` the cosine and sine of the yaw;
  * cylinders  `(cx, cy, radius, top)` — every person, the carrier included;
  * a truth sample `(t, (px, py, pz), cylinders)`.

§6 simulates the reports (line of sight, the RNG streams and their draw
order, the RTT and RSSI models with Gudmundson shadowing); §7 turns one
epoch's reports into a fix by step-halving Gauss-Newton, filters each
method with its own constant-velocity Kalman filter, and publishes a
position with a radius (§7.4) or an abstention with a reason.

New code (spec/locate.md §1 B): binary64, left to right, `x^2` is `x*x`,
no `hypot`, every Σ an explicit Neumaier sum in the stated order, every
decision NaN-fail-closed, every division after its guard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from ..roomsim.numerics import C, FOUR_PI, Neumaier
from ..roomsim.rng import Rng
from .config import LocateConfig, LocateError, check_sim_label
from .radius import CHI2_2_99, LN10, abstention, estimate_record, publishable_estimate, radius95

__all__ = [
    "RTT_STREAM_BASE",
    "RSSI_STREAM_BASE",
    "METHODS",
    "wall_blocks",
    "box_blocks",
    "cylinder_blocks",
    "segment_blocked",
    "EpochClock",
    "NodeReport",
    "PhoneRadio",
    "Fix",
    "rtt_fix",
    "rssi_fix",
    "KalmanStep",
    "PhoneKalman",
    "PhoneMethod",
    "PhoneRun",
    "run_phone",
]

RTT_STREAM_BASE = 300000
RSSI_STREAM_BASE = 310000
METHODS = ("rtt", "rssi")

# §6.1: `t * phone_rate_hz` stays below 2^53, where `float(e_next)` stops
# being exact.
_MAX_EPOCH_INDEX = 9007199254740992.0

# §7.1 numeric thresholds.
_START_DET_REL = 1e-9
_DEGENERATE_REL = 1e-12
_RTT_MIN_RHO = 1e-9
_RSSI_MIN_RHO = 0.1
_MAX_ITER = 200
_MAX_HALVINGS = 30
_STOP_STEP_M = 1e-4


def _sorted_nodes(nodes) -> list[tuple[int, float, float, float]]:
    """Nodes in id order, checked: unique integer ids, finite positions."""
    out = []
    for n in nodes:
        nid, x, y, z = n
        if not (isinstance(nid, int) and not isinstance(nid, bool)):
            raise LocateError("nodes", f"node id {nid!r} is not an integer")
        x, y, z = float(x), float(y), float(z)
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(z)):
            raise LocateError("nodes", f"node {nid}: position is not finite")
        out.append((nid, x, y, z))
    out.sort(key=lambda r: r[0])
    for i in range(1, len(out)):
        if out[i][0] == out[i - 1][0]:
            raise LocateError("nodes", f"duplicate node id {out[i][0]}")
    return out


# ---------------------------------------------------------------------------
# §6.2 line of sight
# ---------------------------------------------------------------------------


def wall_blocks(p, q, wall) -> bool:
    """Roomsim §6.1 `hit(P, Q)` on the centreline (`o = 0`) of the wall
    `(ax, ay, bx, by, h)`: `0 < s < 1`, `0 <= v <= 1` and `0 <= X.z <= h`."""
    ax, ay, bx, by, h = wall
    rx = q[0] - p[0]
    ry = q[1] - p[1]
    wx = bx - ax
    wy = by - ay
    den = rx * wy - ry * wx
    if den == 0.0:
        return False
    apx = ax - p[0]
    apy = ay - p[1]
    s = (apx * wy - apy * wx) / den
    v = (apx * ry - apy * rx) / den
    if not (0.0 < s < 1.0 and 0.0 <= v <= 1.0):
        return False
    xz = p[2] + s * (q[2] - p[2])
    return 0.0 <= xz <= h


def _to_local(p, box):
    cx, cy, cz, _sl, _sb, _sh, cs, sn = box
    dx = p[0] - cx
    dy = p[1] - cy
    lx = dx * cs + dy * sn
    ly = -dx * sn + dy * cs
    lz = p[2] - cz
    return (lx, ly, lz)


def box_blocks(p, q, box) -> bool:
    """Roomsim §7.4's box test of the leg `p -> q` against the box
    `(cx, cy, cz, sl, sb, sh, cs, sn)` (3-D, both ends to object-local)."""
    a = _to_local(p, box)
    b = _to_local(q, box)
    half = (box[3] / 2.0, box[4] / 2.0, box[5] / 2.0)
    t_enter = 0.0
    t_exit = 1.0
    for ax in range(3):
        h = half[ax]
        pa = a[ax]
        d = b[ax] - a[ax]
        if d == 0.0:
            if abs(pa) > h:
                return False
            continue
        t1 = (-h - pa) / d
        t2 = (h - pa) / d
        lo = t1 if t1 < t2 else t2
        hi = t2 if t1 < t2 else t1
        t_enter = lo if lo > t_enter else t_enter      # Python max(t_enter, lo)
        t_exit = hi if hi < t_exit else t_exit         # Python min(t_exit, hi)
    return t_exit > t_enter


def cylinder_blocks(p, q, cyl) -> bool:
    """Roomsim §8.5 steps 1-3's geometry for the cylinder
    `(cx, cy, radius, top)`: `L2 == 0` is not blocked; `u* <= 0` or
    `u* >= 1`: blocked iff the nearer end is inside the cylinder; otherwise
    blocked iff `d_perp < radius` and `z* <= top`."""
    cx, cy, radius, top = cyl
    rx = q[0] - p[0]
    ry = q[1] - p[1]
    L2 = rx * rx + ry * ry
    if L2 == 0.0:
        return False
    u = ((cx - p[0]) * rx + (cy - p[1]) * ry) / L2
    if u <= 0.0 or u >= 1.0:
        end = p if u <= 0.0 else q
        ex = end[0] - cx
        ey = end[1] - cy
        return math.sqrt(ex * ex + ey * ey) < radius and end[2] <= top
    zs = p[2] + u * (q[2] - p[2])
    fx = cx - (p[0] + u * rx)
    fy = cy - (p[1] + u * ry)
    d_perp = math.sqrt(fx * fx + fy * fy)
    return d_perp < radius and zs <= top


def segment_blocked(p, q, walls, boxes, cylinders) -> bool:
    """§6.2: the segment `p -> q` is blocked by any wall, box or cylinder.
    A fact of the geometry, not a draw."""
    for w in walls:
        if wall_blocks(p, q, w):
            return True
    for b in boxes:
        if box_blocks(p, q, b):
            return True
    for c in cylinders:
        if cylinder_blocks(p, q, c):
            return True
    return False


# ---------------------------------------------------------------------------
# §6.1 epochs
# ---------------------------------------------------------------------------


class EpochClock:
    """§6.1's firing rule. `fires(t_j)` is called once per truth sample in
    time order: an epoch fires at sample `j` iff `t_j >= float(e_next) /
    phone_rate_hz`; then `e_next` advances past every epoch time `<= t_j`
    (epochs passed over are dropped). At most one epoch fires per sample.
    A non-finite `t_j` fires nothing (a NaN fails the comparison; an
    infinite one would never stop advancing); a finite one with `t_j *
    phone_rate_hz >= 2^53` is refused naming `t` (`PhoneRun` refuses it
    first). A long gap is skipped ahead before the advancing loop, as the
    Rust port does: the skip is taken only when every index it passes
    satisfies the loop's own test, so `e_next` ends where the loop alone
    would put it, without `t_j * phone_rate_hz` iterations."""

    __slots__ = ("rate_hz", "e_next")

    def __init__(self, phone_rate_hz: float) -> None:
        if not (phone_rate_hz > 0.0 and math.isfinite(phone_rate_hz)):
            raise LocateError("phone_rate_hz", f"{phone_rate_hz!r} must be finite and > 0")
        self.rate_hz = phone_rate_hz
        self.e_next = 0

    def fires(self, t: float) -> bool:
        if not math.isfinite(t):
            return False
        rate = self.rate_hz
        if not (t * rate < _MAX_EPOCH_INDEX):
            raise LocateError("t", "beyond the epoch counter's range (t * phone_rate_hz >= 2^53)")
        if not (t >= float(self.e_next) / rate):
            return False
        # t >= 0 and t * rate < 2^53 here: the floor is exact as an int.
        guess = math.floor(t * rate)
        if guess > 2:
            cand = guess - 2
            if cand > self.e_next and t >= float(cand - 1) / rate:
                self.e_next = cand
        while t >= float(self.e_next) / rate:
            self.e_next += 1
        return True


# ---------------------------------------------------------------------------
# §6.3-6.5 the radio
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NodeReport:
    """One node's outcome at one fired epoch.

    `rtt_m` / `rssi_dbm` are None when that report failed (`u_fail <
    *_fail_p`; for RSSI also a non-finite `v`). The rest are diagnostics
    for the golden vectors: `d` the
    true 3-D distance, `rtt_value` and `rssi_v` the values before the
    failure test (`rssi_v` before rounding), `shadow_db` the node's `S`
    after this epoch, and the draws in their §6.3 order.
    """

    node: int
    blocked: bool
    rtt_m: float | None
    rssi_dbm: int | None
    d: float
    rtt_value: float
    rssi_v: float
    shadow_db: float
    rtt_draws: tuple      # (u_fail, g, u_out, u_mag, u_b)
    rssi_draws: tuple     # (u_fail, g_s, u_j)

    def to_record(self) -> dict:
        """§11 `reports` entry: `{node, rtt_m, rssi_dbm, blocked}`."""
        return {"node": self.node, "rtt_m": self.rtt_m, "rssi_dbm": self.rssi_dbm, "blocked": self.blocked}


class PhoneRadio:
    """§6.3-6.5: the per-run streams, offsets and shadowing state.

    At creation, per node in id order: RTT stream (`Rng.stream(phone_seed,
    300000 + n)`) `off_n = rtt_offset_sigma_m * gauss()`; RSSI stream
    (`310000 + n`) `dev_n = rssi_device_sigma_db * gauss()`. `epoch()` then
    makes every fired epoch's draws in §6.3's fixed order, whatever the
    geometry or outcome.
    """

    def __init__(self, nodes, cfg: LocateConfig, label: str = "SIM") -> None:
        check_sim_label(label)
        cfg.require_resolved()
        self.cfg = cfg
        self.nodes = _sorted_nodes(nodes)
        if not self.nodes:
            raise LocateError("nodes", "the phone needs at least one node")
        seed = cfg.phone_seed
        self._rtt_rng = {}
        self._rssi_rng = {}
        self.offset_m = {}
        self.device_db = {}
        for nid, _x, _y, _z in self.nodes:
            rt = Rng.stream(seed, RTT_STREAM_BASE + nid)
            rs = Rng.stream(seed, RSSI_STREAM_BASE + nid)
            self._rtt_rng[nid] = rt
            self._rssi_rng[nid] = rs
            self.offset_m[nid] = cfg.rtt_offset_sigma_m * rt.gauss()
            self.device_db[nid] = cfg.rssi_device_sigma_db * rs.gauss()
        lam = C / cfg.carrier_hz       # carrier_hz > 0: validated
        self.pl0 = 20.0 * math.log10(FOUR_PI / lam)
        self._shadow: dict = {}        # node -> S at the previous fired epoch
        self._prev_xy = None           # phone (x, y) at the previous fired epoch

    def epoch(self, phone, blocked: list[bool]) -> list[NodeReport]:
        """One fired epoch with the phone at `phone = (px, py, pz)` and the
        §6.2 `blocked` flag of each node (id order). Returns one
        `NodeReport` per node in id order."""
        cfg = self.cfg
        if len(blocked) != len(self.nodes):
            raise ValueError(f"blocked has {len(blocked)} entries for {len(self.nodes)} nodes")
        px, py, pz = float(phone[0]), float(phone[1]), float(phone[2])
        if not (math.isfinite(px) and math.isfinite(py) and math.isfinite(pz)):
            raise LocateError("phone", "the phone's position is not finite")
        # §6.1: a node whose distance d is not finite refuses the run, before
        # any draw.
        for nid, nx, ny, nz in self.nodes:
            ex = px - nx
            ey = py - ny
            ez = pz - nz
            if not math.isfinite(math.sqrt((ex * ex + ey * ey) + ez * ez)):
                raise LocateError("phone", f"the distance to node {nid} is not finite")
        first = self._prev_xy is None
        if first:
            disp = 0.0
        else:
            qx, qy = self._prev_xy
            ddx = px - qx
            ddy = py - qy
            disp = math.sqrt(ddx * ddx + ddy * ddy)
        out = []
        for (nid, nx, ny, nz), blk in zip(self.nodes, blocked):
            blk = bool(blk)
            rt = self._rtt_rng[nid]
            rs = self._rssi_rng[nid]
            # §6.3: RTT stream, then RSSI stream, always in this order.
            u_fail = rt.uniform()
            g = rt.gauss()
            u_out = rt.uniform()
            u_mag = rt.uniform()
            u_b = rt.uniform()
            s_fail = rs.uniform()
            g_s = rs.gauss()
            u_j = rs.uniform()

            # §6.4 RTT.
            ex = px - nx
            ey = py - ny
            ez = pz - nz
            d = math.sqrt((ex * ex + ey * ey) + ez * ez)
            sig = cfg.rtt_sigma_m * cfg.rtt_nlos_sigma_scale if blk else cfg.rtt_sigma_m
            r = (d + self.offset_m[nid]) + sig * g
            if blk:
                # 1.0 - u_b is in (0, 1]: u_b is in [0, 1).
                r = r + (-(cfg.rtt_nlos_bias_m * math.log(1.0 - u_b)))
            if u_out < cfg.rtt_outlier_p:
                r = r + (cfg.rtt_outlier_lo_m + (cfg.rtt_outlier_hi_m - cfg.rtt_outlier_lo_m) * u_mag)
            r = r if r > 0.0 else 0.0
            rtt_m = None if u_fail < cfg.rtt_fail_p else r

            # §6.5 shadowing: every node, every fired epoch, before u_fail.
            sg = cfg.rssi_shadow_sigma_db * g_s
            if first:
                S = sg
            else:
                rho = math.exp(-(disp / cfg.rssi_shadow_corr_m))     # corr > 0: validated
                S = rho * self._shadow[nid] + math.sqrt(1.0 - rho * rho) * sg
            self._shadow[nid] = S

            # §6.5 RSSI.
            dd = d if d > 0.1 else 0.1
            bp = cfg.rssi_breakpoint_m
            if dd <= bp:
                pl = self.pl0 + 20.0 * math.log10(dd)
            else:
                pl = (self.pl0 + 20.0 * math.log10(bp)) + (10.0 * cfg.rssi_far_exponent) * math.log10(dd / bp)
            nlos = cfg.rssi_nlos_loss_db if blk else 0.0
            jit = cfg.rssi_jitter_db
            v = ((((cfg.tx_power_dbm - pl) - nlos) + S) + self.device_db[nid]) + (-jit + (2.0 * jit) * u_j)
            # No report if u_fail < rssi_fail_p or v is not finite (an absurd
            # but valid config can overflow it); the draws and S are kept.
            if s_fail < cfg.rssi_fail_p or not math.isfinite(v):
                rssi = None
            else:
                q = round(v)                       # half-to-even
                q = -100 if -100 > q else q        # clamp(round(v), -100, 0)
                rssi = 0 if 0 < q else q
            out.append(NodeReport(nid, blk, rtt_m, rssi, d, r, v, S,
                                  (u_fail, g, u_out, u_mag, u_b), (s_fail, g_s, u_j)))
        self._prev_xy = (px, py)
        return out


# ---------------------------------------------------------------------------
# §7.1-7.2 fixes
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Fix:
    """One epoch's fix for one method, or the reason there is none.

    When `reason` is None: `(x, y)`, `inv(N)` as `inv00, inv01, inv11`
    (the final normal matrix's inverse), `R = sigma*sigma * inv(N)` as
    `r00, r01, r11`, the Gauss-Newton `iterations` and, for RTT, `chi2`
    (unweighted Σ u*u of the final pass; None for RSSI). When `reason` is
    set, everything else is None.
    """

    reason: str | None
    x: float | None = None
    y: float | None = None
    inv00: float | None = None
    inv01: float | None = None
    inv11: float | None = None
    r00: float | None = None
    r01: float | None = None
    r11: float | None = None
    chi2: float | None = None
    iterations: int | None = None

    @property
    def ok(self) -> bool:
        return self.reason is None


class _Sums:
    __slots__ = ("N00", "N01", "N11", "G0", "G1", "F", "chi2", "used")

    def __init__(self, N00, N01, N11, G0, G1, F, chi2, used):
        self.N00, self.N01, self.N11 = N00, N01, N11
        self.G0, self.G1 = G0, G1
        self.F = F
        self.chi2 = chi2
        self.used = used

    def det(self) -> float:
        return self.N00 * self.N11 - self.N01 * self.N01

    def degenerate(self, det: float) -> bool:
        return not (det > _DEGENERATE_REL * (self.N00 * self.N11))


def _rtt_sums(x, y, rep, zh, sigma, k) -> _Sums:
    n00, n01, n11, g0, g1, f, c2 = (Neumaier() for _ in range(7))
    used = 0
    for _nid, nx, ny, nz, r in rep:
        dz = zh - nz
        dx = x - nx
        dy = y - ny
        rho = math.sqrt((dx * dx + dy * dy) + dz * dz)
        if rho < _RTT_MIN_RHO:
            continue
        res = r - rho
        u = res / sigma                      # sigma > 0: validated
        w = k / u if u > k else 1.0          # u > k > 0 here
        hx = dx / rho                        # rho >= 1e-9 (or NaN)
        hy = dy / rho
        n00.add((w * hx) * hx)
        n01.add((w * hx) * hy)
        n11.add((w * hy) * hy)
        g0.add((w * hx) * res)
        g1.add((w * hy) * res)
        f.add(0.5 * (u * u) if u <= k else k * u - 0.5 * (k * k))
        c2.add(u * u)
        used += 1
    return _Sums(n00.value(), n01.value(), n11.value(), g0.value(), g1.value(), f.value(), c2.value(), used)


def _rssi_sums(x, y, rep, zh, p0, n_exp) -> _Sums:
    n00, n01, n11, g0, g1, f = (Neumaier() for _ in range(6))
    a10 = 10.0 * n_exp
    gk = a10 / LN10
    used = 0
    for _nid, nx, ny, nz, rs in rep:
        dz = zh - nz
        dx = x - nx
        dy = y - ny
        rho = math.sqrt((dx * dx + dy * dy) + dz * dz)
        if rho < _RSSI_MIN_RHO:
            continue
        pred = p0 - a10 * math.log10(rho)    # rho >= 0.1 (or NaN)
        res = rs - pred
        rr = rho * rho
        jx = ((-gk) * dx) / rr
        jy = ((-gk) * dy) / rr
        n00.add(jx * jx)
        n01.add(jx * jy)
        n11.add(jy * jy)
        g0.add(jx * res)
        g1.add(jy * res)
        f.add(0.5 * (res * res))
        used += 1
    return _Sums(n00.value(), n01.value(), n11.value(), g0.value(), g1.value(), f.value(), None, used)


def _gauss_newton(sums_at, x0: float, y0: float, sigma: float, with_chi2: bool) -> Fix:
    """§7.1's loop: at most 200 iterations, step halving (at most 30), stop
    when `a * sqrt(sx*sx + sy*sy) < 1e-4`, then one final pass."""
    x, y = x0, y0
    converged = False
    it = 0
    while it < _MAX_ITER:
        it += 1
        s = sums_at(x, y)
        det = s.det()
        if s.degenerate(det):
            return Fix("degenerate")
        sx = (s.N11 * s.G0 - s.N01 * s.G1) / det
        sy = (s.N00 * s.G1 - s.N01 * s.G0) / det
        a = 1.0
        halvings = 0
        f0 = s.F
        while not (sums_at(x + a * sx, y + a * sy).F <= f0) and halvings < _MAX_HALVINGS:
            a = 0.5 * a
            halvings += 1
        x = x + a * sx
        y = y + a * sy
        if a * math.sqrt(sx * sx + sy * sy) < _STOP_STEP_M:
            converged = True
            break
    if not converged:
        return Fix("not_converged")
    s = sums_at(x, y)
    det = s.det()
    if s.degenerate(det):
        return Fix("degenerate")
    inv00 = s.N11 / det
    inv01 = (-s.N01) / det
    inv11 = s.N00 / det
    s2 = sigma * sigma
    return Fix(None, x, y, inv00, inv01, inv11, s2 * inv00, s2 * inv01, s2 * inv11,
               s.chi2 if with_chi2 else None, it)


def _gather(reports, nodes) -> list:
    """`[(node, nx, ny, nz, value)]`, nodes ascending (a stable sort), from
    `(node, value)` pairs. §7: reports name distinct nodes the geometry has,
    with finite values, else `LocateError('reports')`."""
    pos = {n[0]: n for n in nodes}
    rep = []
    for nid, val in sorted(reports, key=lambda r: r[0]):
        if nid not in pos:
            raise LocateError("reports", f"a report names node {nid}, which is not a node")
        val = float(val)
        if not math.isfinite(val):
            raise LocateError("reports", f"node {nid}: the report's value is not finite")
        _, nx, ny, nz = pos[nid]
        rep.append((nid, nx, ny, nz, val))
    for i in range(1, len(rep)):
        if rep[i][0] == rep[i - 1][0]:
            raise LocateError("reports", f"two reports from node {rep[i][0]}")
    return rep


def _anchor_normal(rep) -> tuple[float, float, float, float]:
    """§7.1's start normal matrix from the reports' anchors alone: with
    report 1 the first, `A_n = (2.0*(nx_n - nx_1), 2.0*(ny_n - ny_1))` for
    `n = 2..m`; `(n00, n01, n11, det)`, each Σ a Neumaier sum in node order,
    `det = n00*n11 - n01*n01`."""
    _, nx1, ny1, _nz1, _v1 = rep[0]
    n00, n01, n11 = Neumaier(), Neumaier(), Neumaier()
    for i in range(1, len(rep)):
        _, nxn, nyn, _nzn, _vn = rep[i]
        ax = 2.0 * (nxn - nx1)
        ay = 2.0 * (nyn - ny1)
        n00.add(ax * ax)
        n01.add(ax * ay)
        n11.add(ay * ay)
    a00, a01, a11 = n00.value(), n01.value(), n11.value()
    return a00, a01, a11, a00 * a11 - a01 * a01


def _collinear(a00: float, a11: float, det: float) -> bool:
    """§7's collinear rule: `not (det > 1e-9 * ((n00 + n11)*(n00 + n11)))`
    (a NaN counts as collinear). A range or level set from anchors on one
    line cannot tell the two sides of the line apart, so there is no fix
    (`degenerate`). `det / (n00 + n11)^2` is the product of the normal
    matrix's eigenvalues over the square of their sum, which does not
    depend on the line's direction: anchors collinear to rounding are
    caught at any angle (an axis-aligned line too, where `n00` or `n11` is
    itself rounding-sized), and anchors within about 3e-5 of their span
    from a line count as collinear."""
    return not (det > _START_DET_REL * ((a00 + a11) * (a00 + a11)))


def _pow10(e: float) -> float:
    """`10^e` as libm `pow(10.0, e)` (§1 B), overflow included: CPython's
    `10.0 ** e` raises where libm returns +inf, so that one case is mapped
    back to the IEEE value. Nothing is decided by the exception: the
    caller's NaN-safe test (§7.2's start guard) decides."""
    try:
        return 10.0 ** e
    except OverflowError:
        return math.inf


def rtt_fix(reports, nodes, cfg: LocateConfig, label: str = "SIM") -> Fix:
    """§7.1: the RTT fix of one epoch.

    `reports` are `(node, range_m)` pairs (any order; used ascending),
    `nodes` the `(node_id, x, y, z)` anchors. Reports name distinct known
    nodes with finite values, else `LocateError('reports')`. Reasons:
    `no_reports`, `too_few_ranges`, `degenerate` (§7's collinear rule, or
    a singular normal matrix in Gauss-Newton), `not_converged`.
    """
    check_sim_label(label)
    cfg.validate()
    rep = _gather(reports, _sorted_nodes(nodes))
    m = len(rep)
    if m == 0:
        return Fix("no_reports")
    if m < 3:
        return Fix("too_few_ranges")
    # Collinear anchors: no fix (§7). The start's normal matrix is the same.
    a00, a01, a11, det = _anchor_normal(rep)
    if _collinear(a00, a11, det):
        return Fix("degenerate")
    zh = cfg.phone_assumed_z_m
    # Start: linearised least squares on the horizontal ranges.
    hs = []
    for _nid, _nx, _ny, nz, r in rep:
        dz = zh - nz
        rr = r * r
        dd = dz * dz
        hs.append(math.sqrt(rr - dd) if rr > dd else 0.0)
    _, nx1, ny1, _nz1, _r1 = rep[0]
    h1 = hs[0]
    c0, c1 = Neumaier(), Neumaier()
    for i in range(1, m):
        _, nxn, nyn, _nzn, _rn = rep[i]
        hn = hs[i]
        ax = 2.0 * (nxn - nx1)
        ay = 2.0 * (nyn - ny1)
        b = ((((nxn * nxn) + (nyn * nyn)) - (nx1 * nx1)) - (ny1 * ny1)) - ((hn * hn) - (h1 * h1))
        c0.add(ax * b)
        c1.add(ay * b)
    b0, b1 = c0.value(), c1.value()
    # det > 1e-9 * ((n00 + n11)*(n00 + n11)) >= 0 here: the division is
    # guarded.
    x0 = (a11 * b0 - a01 * b1) / det
    y0 = (a00 * b1 - a01 * b0) / det
    sigma = cfg.rtt_sigma_m
    k = cfg.huber_k
    return _gauss_newton(lambda x, y: _rtt_sums(x, y, rep, zh, sigma, k), x0, y0, sigma, True)


def rssi_fix(reports, nodes, cfg: LocateConfig, label: str = "SIM") -> Fix:
    """§7.2: the RSSI fix of one epoch.

    `reports` are `(node, rssi_dbm)` pairs (taken as floats), `nodes` the
    `(node_id, x, y, z)` anchors. `rssi_p0_dbm` unset is derived from
    `tx_power_dbm` and `carrier_hz` (`LocateConfig.rssi_p0_value`);
    `LocateError('rssi_p0_dbm')` when those are unset too. Reports as
    `rtt_fix`. Reasons as `rtt_fix`:
    `no_reports`, `too_few_ranges`, then §7's collinear rule
    (`degenerate`), then a start whose weight sum is not finite and > 0
    (unreachable for dBm in -100..0) is `degenerate`.
    """
    check_sim_label(label)
    cfg.validate()
    p0 = cfg.rssi_p0_value()
    rep = _gather(reports, _sorted_nodes(nodes))
    m = len(rep)
    if m == 0:
        return Fix("no_reports")
    if m < 3:
        return Fix("too_few_ranges")
    a00, _a01, a11, det = _anchor_normal(rep)
    if _collinear(a00, a11, det):
        return Fix("degenerate")
    sw = Neumaier()
    sx = Neumaier()
    sy = Neumaier()
    for _nid, nx, ny, _nz, rs in rep:
        w = _pow10(rs / 10.0)
        sw.add(w)
        sx.add(w * nx)
        sy.add(w * ny)
    wsum = sw.value()
    if not (wsum > 0.0 and math.isfinite(wsum)):
        return Fix("degenerate")
    x0 = sx.value() / wsum
    y0 = sy.value() / wsum
    zh = cfg.phone_assumed_z_m
    n_exp = cfg.rssi_est_exponent
    return _gauss_newton(lambda x, y: _rssi_sums(x, y, rep, zh, p0, n_exp), x0, y0,
                         cfg.rssi_shadow_sigma_db, False)


# ---------------------------------------------------------------------------
# §7.3 Kalman filter
# ---------------------------------------------------------------------------


def _mm(a, b):
    """`C = A·B`: `c_ij = (((a_i0*b_0j) + a_i1*b_1j) + …)`, ascending k, no term skipped."""
    nk = len(b)
    out = []
    for row in a:
        r = []
        for j in range(len(b[0])):
            acc = row[0] * b[0][j]
            for k in range(1, nk):
                acc = acc + row[k] * b[k][j]
            r.append(acc)
        out.append(r)
    return out


def _tr(a):
    return [list(col) for col in zip(*a)]


def _add(a, b):
    return [[a[i][j] + b[i][j] for j in range(len(a[0]))] for i in range(len(a))]


def _sym(p):
    """`sym(P)`: `p_ij` and `p_ji` become `0.5*(p_ij + p_ji)` for `j < i`."""
    out = [list(r) for r in p]
    n = len(out)
    for i in range(n):
        for j in range(i):
            v = 0.5 * (out[i][j] + out[j][i])
            out[i][j] = v
            out[j][i] = v
    return out


def _identity(n):
    return [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]


@dataclass(frozen=True)
class KalmanStep:
    """The filter's outcome at one fired epoch: `reason` None and the state
    position `(x, y)` with `P[0:2][0:2]` (`p00, p01, p11`), or an
    abstention `reason` (the fix's, or `gated`). `started` says the filter
    was (re)started by this fix; `discarded` that step 1 discarded it."""

    reason: str | None
    x: float | None = None
    y: float | None = None
    p00: float | None = None
    p01: float | None = None
    p11: float | None = None
    started: bool = False
    discarded: bool = False


class PhoneKalman:
    """§7.3: constant velocity, continuous white-noise acceleration
    `q = phone_accel_psd`; state `[x, y, vx, vy]`; `t_f` the time of its
    start or last predict, `t_acc` of its start or last accepted update."""

    def __init__(self, cfg: LocateConfig, label: str = "SIM") -> None:
        check_sim_label(label)
        cfg.validate()
        self.q = cfg.phone_accel_psd
        self.v0 = cfg.phone_init_speed_std
        self.max_coast_s = cfg.phone_max_coast_s
        self.s = None
        self.P = None
        self.t_f = None
        self.t_acc = None

    @property
    def exists(self) -> bool:
        return self.s is not None

    def discard(self) -> None:
        self.s = None
        self.P = None
        self.t_f = None
        self.t_acc = None

    def step(self, t: float, fix: Fix) -> KalmanStep:
        """One fired epoch at time `t`, in §7.3's order."""
        # 1. Coast limit (a NaN discards).
        discarded = False
        if self.exists and not (t - self.t_acc <= self.max_coast_s):
            self.discard()
            discarded = True
        # 2. No fix: abstain with its reason; nothing is predicted.
        if not fix.ok:
            return KalmanStep(fix.reason, discarded=discarded)
        zx, zy = fix.x, fix.y
        R = [[fix.r00, fix.r01], [fix.r01, fix.r11]]
        # 3. Start.
        if not self.exists:
            v2 = self.v0 * self.v0
            self.s = [zx, zy, 0.0, 0.0]
            self.P = [
                [R[0][0], R[0][1], 0.0, 0.0],
                [R[1][0], R[1][1], 0.0, 0.0],
                [0.0, 0.0, v2, 0.0],
                [0.0, 0.0, 0.0, v2],
            ]
            self.t_f = t
            self.t_acc = t
            return KalmanStep(None, zx, zy, R[0][0], R[0][1], R[1][1], started=True, discarded=discarded)
        # 4. Predict from t_f to t.
        dt = t - self.t_f
        self.t_f = t
        q = self.q
        F = _identity(4)
        F[0][2] = dt
        F[1][3] = dt
        Q = [[0.0] * 4 for _ in range(4)]
        for i in (0, 1):
            Q[i][i] = q * ((dt * dt) * dt) / 3.0
            Q[i][i + 2] = q * (dt * dt) / 2.0
            Q[i + 2][i] = q * (dt * dt) / 2.0
            Q[i + 2][i + 2] = q * dt
        s = [row[0] for row in _mm(F, [[v] for v in self.s])]
        P = _sym(_add(_mm(_mm(F, self.P), _tr(F)), Q))
        self.s = s
        self.P = P
        # Gate.
        nu0 = zx - s[0]
        nu1 = zy - s[1]
        S00 = P[0][0] + R[0][0]
        S01 = P[0][1] + R[0][1]
        S10 = P[1][0] + R[1][0]
        S11 = P[1][1] + R[1][1]
        det = S00 * S11 - S01 * S10
        if not (det > 0.0):
            return KalmanStep("gated", discarded=discarded)
        Si = [[S11 / det, (-S01) / det], [(-S10) / det, S00 / det]]
        d2 = ((nu0 * Si[0][0] + nu1 * Si[1][0]) * nu0) + ((nu0 * Si[0][1] + nu1 * Si[1][1]) * nu1)
        if not (d2 <= CHI2_2_99):
            return KalmanStep("gated", discarded=discarded)
        # Update (Joseph form).
        K = _mm([[P[i][0], P[i][1]] for i in range(4)], Si)
        Kn = _mm(K, [[nu0], [nu1]])
        s = [s[i] + Kn[i][0] for i in range(4)]
        H = [[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]]
        KH = _mm(K, H)
        I4 = _identity(4)
        A = [[I4[i][j] - KH[i][j] for j in range(4)] for i in range(4)]
        P = _sym(_add(_mm(_mm(A, P), _tr(A)), _mm(_mm(K, R), _tr(K))))
        self.s = s
        self.P = P
        self.t_acc = t
        return KalmanStep(None, s[0], s[1], P[0][0], P[0][1], P[1][1], discarded=discarded)


# ---------------------------------------------------------------------------
# §7.4 one method end to end, and the run
# ---------------------------------------------------------------------------


class PhoneMethod:
    """One method (`"rtt"` or `"rssi"`): fix, filter, published estimate.

    `step(t, reports)` takes one fired epoch's `NodeReport`s (or
    `(node, value)` pairs for this method, value None for a failed report)
    and returns the §11 record: an Estimate (`confidence` None) with
    `cov = P[0:2][0:2] + B`, `B = beta*beta * inv(N)`, or an Abstention.
    Reasons, first that applies: `no_reports`, `too_few_ranges`,
    `degenerate`, `not_converged`, `gated`, `implausible_radius`.
    """

    def __init__(self, method: str, nodes, cfg: LocateConfig, label: str = "SIM") -> None:
        check_sim_label(label)
        if method not in METHODS:
            raise ValueError(f"method: {method!r} is not one of {', '.join(METHODS)}")
        cfg.require_resolved()
        self.method = method
        self.cfg = cfg
        self.nodes = _sorted_nodes(nodes)
        self.kf = PhoneKalman(cfg, label)
        self.beta = cfg.rtt_persist_sigma_m if method == "rtt" else cfg.rssi_persist_sigma_db
        self.last_fix: Fix | None = None
        self.last_step: KalmanStep | None = None

    def _pairs(self, reports) -> list:
        out = []
        for r in reports:
            if isinstance(r, NodeReport):
                v = r.rtt_m if self.method == "rtt" else r.rssi_dbm
                nid = r.node
            else:
                nid, v = r
            if v is not None:
                out.append((nid, v))
        return out

    def step(self, t: float, reports) -> dict:
        pairs = self._pairs(reports)
        if self.method == "rtt":
            fix = rtt_fix(pairs, self.nodes, self.cfg)
        else:
            fix = rssi_fix(pairs, self.nodes, self.cfg)
        st = self.kf.step(t, fix)
        self.last_fix = fix
        self.last_step = st
        if st.reason is not None:
            return abstention(st.reason)
        bb = self.beta * self.beta
        cxx = st.p00 + bb * fix.inv00
        cxy = st.p01 + bb * fix.inv01
        cyy = st.p11 + bb * fix.inv11
        r = self.cfg.phone_radius_scale * radius95(cxx, cxy, cyy)
        if not publishable_estimate(st.x, st.y, r, self.cfg.min_plausible_radius_m):
            return abstention("implausible_radius")
        return estimate_record(st.x, st.y, r, cxx, cxy, cyy, None)


class PhoneRun:
    """§6 + §7 streamed over truth samples: `sample(t, phone, cylinders)`
    returns the §11 epoch record when an epoch fires at that sample,
    else None. `walls` and `boxes` are fixed for the run.

    §6.1's input contract, each refusal a `LocateError` naming the input:
    nodes in id order (a sorted copy), distinct, at least one (`nodes`);
    every wall and box number finite (`walls`, `objects`); per sample, `t`
    finite, not below the previous sample's and `t * phone_rate_hz < 2^53`
    (`t`), the phone's coordinates finite (`phone`), every cylinder number
    finite (`cylinders`), and every node's distance finite (`phone`).
    Every one is checked at every sample, whether or not an epoch fires
    there, before any state changes: a refused sample changes nothing
    (`last_t`, the clock's `e_next`, the radio and the filters)."""

    def __init__(self, nodes, walls, boxes, cfg: LocateConfig, label: str = "SIM") -> None:
        check_sim_label(label)
        cfg.require_resolved()
        self.cfg = cfg
        self.label = label
        self.nodes = _sorted_nodes(nodes)
        self.walls = [tuple(float(v) for v in w) for w in walls]
        self.boxes = [tuple(float(v) for v in b) for b in boxes]
        for w in self.walls:
            if len(w) != 5:
                raise LocateError("walls", "a wall is (ax, ay, bx, by, h)")
            if not all(math.isfinite(v) for v in w):
                raise LocateError("walls", "every number must be finite")
        for b in self.boxes:
            if len(b) != 8:
                raise LocateError("objects", "a box is (cx, cy, cz, sl, sb, sh, cs, sn)")
            if not all(math.isfinite(v) for v in b):
                raise LocateError("objects", "every number must be finite")
        self.clock = EpochClock(cfg.phone_rate_hz)
        self.radio = PhoneRadio(self.nodes, cfg, label)
        self.rtt = PhoneMethod("rtt", self.nodes, cfg, label)
        self.rssi = PhoneMethod("rssi", self.nodes, cfg, label)
        self.last_reports: list[NodeReport] = []
        self.last_t: float | None = None

    def sample(self, t: float, phone, cylinders) -> dict | None:
        t = float(t)
        if not math.isfinite(t):
            raise LocateError("t", "a truth sample's time must be finite")
        if self.last_t is not None and not (t >= self.last_t):
            raise LocateError("t", "truth samples must be time-ordered")
        p = (float(phone[0]), float(phone[1]), float(phone[2]))
        if not (math.isfinite(p[0]) and math.isfinite(p[1]) and math.isfinite(p[2])):
            raise LocateError("phone", "the phone's position must be finite")
        # §6.1: every node's distance d finite, at every sample whether or not
        # an epoch fires, before any state changes (PhoneRadio.epoch keeps
        # the same check for its direct callers).
        for nid, nx, ny, nz in self.nodes:
            ex = p[0] - nx
            ey = p[1] - ny
            ez = p[2] - nz
            if not math.isfinite(math.sqrt((ex * ex + ey * ey) + ez * ez)):
                raise LocateError("phone", f"the distance to node {nid} is not finite")
        cyl = []
        for c in cylinders:
            c = tuple(float(v) for v in c)
            if len(c) != 4:
                raise LocateError("cylinders", "a cylinder is (cx, cy, radius, top)")
            if not all(math.isfinite(v) for v in c):
                raise LocateError("cylinders", "every number must be finite")
            cyl.append(c)
        if not (t * self.cfg.phone_rate_hz < _MAX_EPOCH_INDEX):
            raise LocateError("t", "beyond the epoch counter's range (t * phone_rate_hz >= 2^53)")
        self.last_t = t
        if not self.clock.fires(t):
            return None
        blocked = [segment_blocked(p, (nx, ny, nz), self.walls, self.boxes, cyl)
                   for _nid, nx, ny, nz in self.nodes]
        reports = self.radio.epoch(p, blocked)
        self.last_reports = reports
        return {
            "label": self.label,
            "t": t,
            "rtt": self.rtt.step(t, reports),
            "rssi": self.rssi.step(t, reports),
            "reports": [r.to_record() for r in reports],
            "truth": {"x": p[0], "y": p[1], "z": p[2]},
        }


def run_phone(samples, nodes, walls, boxes, cfg: LocateConfig, label: str = "SIM") -> list[dict]:
    """§6-§7 over a time-ordered sequence of truth samples
    `(t, (px, py, pz), cylinders)`: one §11 epoch record per fired epoch
    (`PhoneRun`'s input contract applies)."""
    run = PhoneRun(nodes, walls, boxes, cfg, label)
    out = []
    for t, phone, cylinders in samples:
        rec = run.sample(t, phone, cylinders)
        if rec is not None:
            out.append(rec)
    return out
