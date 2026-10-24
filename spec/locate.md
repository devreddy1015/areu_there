# Locate v1 — people and phone positions with uncertainty

**Status:** spec for Phase 4, 2026-10-24, revised after a four-lens review
of the spec (50 findings confirmed or partly confirmed) and two reviews of
both implementations (44 and 28 more). Implemented twice — the
Python reference in `wifisense/localize/` (ADR-002: normative) and the
Rust crate `saans-locate` — and the two must agree (§12).

Everything this produces in Phase 4 is **SIM**: its inputs are Room Sim
packets and truth (spec/roomsim.md), its geometry is the scene (SCENE).
An estimate is always a position **with a radius**, or an abstention with
a reason; never a bare point (ADR-008 decision 5).

**Decisions recorded for the owner** (each reversible; none changes an
existing crate's public API):

1. **Python first.** The existing RTI (`wifisense/localize/geometry.py`,
   `invert.py`, `measure.py`, `track.py`) is normative and is ported to
   Rust as it is. Everything new — the baseline as a library function,
   the radius, an RSSI link signal, the phone — is added to
   `wifisense/localize` first, then ported, with golden vectors.
2. **Link disturbance from RSSI on Room Sim data.** Room Sim normalises
   each packet's CSI to a fixed level with a random-step AGC (spec/roomsim.md
   §12.3), as the ESP32 does, so the existing per-packet total-power signal
   sees AGC steps, not bodies. A new `RssiSignal` (the datagram's RSSI,
   which the AGC does not touch) is the default for Room Sim input; the
   existing signals stay available. A FeatureFrame-based disturbance
   (ADR-008 decision 9's runtime input) is Phase 5's; the estimators here
   take any per-link `y`.
3. **The radius is a model, flagged uncalibrated** (§8): a conservative
   95% circle of a covariance, times `*_radius_scale` (1.0). Every
   estimate carries `radius_calibrated: false` until Phase 5 sets the
   scales from measured coverage; Phase 4's checks print the coverage they
   see. A non-finite radius, or one below `min_plausible_radius_m`
   (0.3 m), is never published (ADR-008 d5).
4. **The baseline abstains** instead of falling back to the room centre,
   behind a significance gate whose false-alarm rate is measured, and
   gains a radius (§5). `evaluate_localize.py` is unchanged.
5. **The phone is SIM only** (ADR-008 d16) and its radio model lives here,
   not in Room Sim (§6). Ranging a cooperative device is not the
   multilateration of people that CAMPUS_SENSING_PROMPT.md §3 bans:
   people are located only by the links they disturb (§4–§5).
6. **One person estimate per estimator**; the existing tracker is
   single-target. Multi-person tracking and zones (ADR-008 d2, d8: no zone
   type exists) are not in Phase 4. Scenes with `roles` (some nodes
   transmit-only or receive-only) or with two nodes at one `(x, y)` are
   refused by the person pipeline (§2).
7. **RTI is expected to abstain in small rooms.** On `rssi_dbm` with
   4 nodes the image carries almost no position: in the bedroom RTI
   publishes on about 0.5% of ticks whether anyone is there or not (§13,
   SIM). That is reported, not tuned away.

**Rules.** `saans-locate` starts with `#![forbid(unsafe_code)]`. It
depends on `saans-sim` (the `Rng`) and `saans-dsp` (`psum`,
`argmax_first`, `RunningVariance`, `StaticSuppressor`, `fuse`), and —
only under its `roomsim` cargo feature, off by default, and in its tests —
on `saans-roomsim` and on `saans-scene` **without** `write`; it is added to
READERS in `saans-scene/tests/boundary.rs`. `wifisense/localize` imports
only the standard library and `wifisense`. `.github/workflows/tests.yml`
gains `cargo check -p saans-locate --locked`, `cargo test -p saans-locate
--locked` and `cargo test -p saans-locate --features roomsim --locked`
(standalone: they also prove the crate builds without `saans-scene/write`).
Existing files this phase edits: `saans-scene/tests/boundary.rs`,
`tests.yml`, `spec/golden/generate.py` (and the regenerated
`spec/golden/MANIFEST`), `rust/Cargo.lock`, the docs (`README.md`,
`spec/wire.md`, `ROOMSIM.md`, `RUN.md`), and additive changes to
`wifisense/localize/__init__.py` (exports, docstring), `measure.py`
(`RssiSignal`) and `track.py` (`PositionEstimate` fields) — a package
SAANS_PLAN.md §3.1 calls "Unchanged API", so flagged. No Rust crate's
public API changes.

---

## 1. Conventions

Units and frames as spec/roomsim.md §1. Two numeric regimes.

**A. Ported code** — `geometry.py`, `invert.py`, `measure.py`, `track.py`,
and what `measure.py` imports (`wifisense/types.py`: `CsiFrame` with its
two `__post_init__` checks, `RunningVariance` (n−1);
`wifisense/preprocess.py`: `StaticSuppressor`, `fuse`). The Python is the
specification line by line, as it is at the commit that adds this spec.
`saans-locate` takes `RunningVariance`, `StaticSuppressor`, `fuse`, `psum`
and `argmax_first` from `saans-dsp` (golden-matched to those modules) and
checks `CsiFrame`'s two conditions itself. Rust reproduces CPython 3.14:
- builtin `sum()` over floats is Neumaier-compensated (`psum`). Only the
  builtin `sum()` is: every loop, `+=`, `-=` or `map(add, …)` accumulation
  is plain, left to right in the code's order, with the code's skips kept
  (e.g. `_mat_mul`'s `if aip == 0.0: continue`);
- `math.hypot(a, b)` is CPython's `vector_norm` (Modules/mathmodule.c,
  3.14), not libm: Rust ports it for two arguments as `py_hypot`, with
  its inf/NaN, zero and subnormal branches and its `frexp`/`ldexp`
  scaling;
- `round(x)` is half-to-even (`f64::round_ties_even`); `int(x)` truncates
  toward zero and raises on NaN and ±inf (Rust checks `is_finite` before
  `as`);
- float `a % b` is CPython's: `r = fmod(a, b)` (Rust `a % b`); then
  `r += b` if `r != 0` and `(r < 0) != (b < 0)`, and `r = copysign(0.0, b)`
  if `r == 0`; never `a - b*floor(a/b)`;
- two-argument `max(a, b)` is `b if b > a else a`, `min(a, b)` is
  `b if b < a else a` (never `f64::max`/`min`); over an iterable they fold
  the same way from the first element; `max(..., key=)` takes the first
  maximum (`argmax_first`); `sorted` is stable; exact float equality where
  the code uses it;
- `math.exp`, `sqrt`, `log10`, `log` are libm (`f64::exp`, `sqrt`,
  `log10`, `ln`).

**B. New code** (every formula of §4.3–§8): evaluated exactly as written,
spec/roomsim.md §1 rules — binary64, left to right, parentheses first, no
FMA, `x^2` is `x*x`, `10^y` is `pow(10.0, y)`; **no `hypot`**: a length is
`sqrt(dx*dx + dy*dy)` or `sqrt((dx*dx + dy*dy) + dz*dz)` (the one exception:
§5 reads `Link.length_m`, which is ported code); every `Σ` is the explicit
**Neumaier** accumulation of spec/roomsim.md §1 in the order stated (the
Python uses an explicit helper, never builtin `sum`); a product `a*b*c` is
`(a*b)*c`; round half-to-even; libm `ln`, `log10`, `exp`, `sqrt`. Every
comparison that decides something is written so that NaN fails closed
(it abstains, rejects or refuses); every division has its guard stated
before it, and the Python never relies on `ZeroDivisionError`, a math
domain error or `OverflowError` to decide anything.

**Small linear algebra (B).** Matrices are row-major. `C = A·B` is
`c_ij = (((a_i0*b_0j) + a_i1*b_1j) + a_i2*b_2j) + …`, ascending `k`, no
term skipped. A 2×2 inverse: `det = a*d - b*c`;
`inv = [[d/det, (-b)/det], [(-c)/det, a/det]]`, used only after its
guard. `sym(P)` replaces `p_ij` and `p_ji` by `0.5*(p_ij + p_ji)` for
`j < i`.

**Constants** (computed, never typed): `K95 = sqrt(-2.0 * ln(0.05))`
(= 2.4477468306808166); `CHI2_2_99 = -2.0 * ln(0.01)`
(= 9.210340371976182); `LN10 = ln(10.0)`; `C`, `FOUR_PI` as roomsim.

**Labels.** Source labels are SCENE, SIM, REAL (firmware or a published
standard, as spec/roomsim.md §3.2) and SIM->REAL (a SIM value used where a
REAL one is known to differ, the gap stated). Every record carries
`label`; in Phase 4 always `"SIM"`. Room Sim input is SIM, so the Room
Sim pipeline (§10) refuses any label but `"SIM"`. A person pipeline fed by
a caller takes its label from the caller and accepts `"SIM"`, `"REAL"` or
`"SIM->REAL"` (SCENE labels geometry, never an estimate); the phone
functions refuse any label but `"SIM"` (ADR-008 d16). Every refusal names
`label` without echoing the value.

## 2. Inputs

- **Geometry (SCENE).** `Room(width_m = room.w_m, height_m = room.d_m)`:
  the localize `Room` is the footprint's bounding rectangle with origin at
  (0, 0), and its `height_m` is the scene's second horizontal axis, not
  `h_m`. Nodes `Node(node_id, x, y)` from `scene.nodes` (z dropped for
  people, kept for the phone). An L-shaped footprint is tiled whole (§13).
  A scene with two nodes at one `(x, y)` and different `z` is refused by
  the person pipeline (error naming `nodes`; the localize `Geometry` is 2-D
  and would raise).
- **Links.** The localize canonical order: every ordered pair `tx != rx`
  sorted by id, tx outer. This equals Room Sim's link order only when every
  node is `txrx`, so the person pipeline refuses a Room Sim config with any
  `roles` entry (error naming `roles`).
- **Packets (SIM).** Room Sim packets (§4.3). **Truth (SIM).** Room Sim
  truth records: used only to simulate the phone's radio (§6) and to score.

## 3. Ported RTI (normative: the Python modules)

Ported with every public function, default, check and exception:
`geometry.py`, `invert.py` (Cholesky routines, SpatialPrior,
forward_project, Inverter with duplicate-row collapse, guards,
diagnostics), `measure.py` (LinkSignal strategies, LinkMeasurement:
windows, progress clock, calibration, masks, counters), `track.py`
(extract_peak, PeakReading, Tracker, PositionEstimate). Python exceptions
map to one Rust error enum whose variants name them; warnings
(`BlindPixelWarning`) become a diagnostics field. `perimeter_nodes` is
ported as the layout helper it is; Locate itself takes nodes only from the
scene.

**Additions to the Python modules** (every existing test passes unchanged):
- `measure.py`: `RssiSignal` (`name = "rssi_dbm"`), a stateless
  `LinkSignal` whose per-frame value is `float(frame.rssi_dbm)` for every
  frame the shared validity gate `_frame_power` accepts (an empty or
  non-finite frame is dropped as for the other signals).
- `track.py`: `PositionEstimate` gains trailing fields `cov_xx: float =
  0.0`, `cov_xy: float = 0.0`, `cov_yy: float = 0.0`. `Tracker.update`
  always sets them to `self._cov[0][0]`, `[0][1]`, `[1][1]` of the same
  matrix `sigma_m` is read from, after `_initialise`/`_correct`. An
  estimate built without them has radius 0, which §8 never publishes
  (`min_plausible_radius_m > 0`).

## 4. Link disturbance

### 4.1 The measurement
`meas = LinkMeasurement(geometry, window_s=window_s, sample_rate_hz=rate_hz,
signal=RssiSignal())` — SUBTRACT, `min_samples` and `clamp_negative` at
their defaults, so `min_samples = max(3, int(round(0.5*window_s*rate_hz)))`
(10 in the bedroom, 7 in the office, 5 in the hall). One object serves the
calibration and then the occupied run. `y = meas.vector(t)` per link, in
dB²: the sample variance over the last `window_s` of the per-packet
signal, minus the link's empty-room baseline. `null_sigma =
meas.baseline_std_vector()`. A run where some transmitter has
`floor(window_s * rate_hz / float(tx_period[tx])) + 1 < min_samples` is
refused (error naming `tx_period`): its links could never fill a window.

### 4.2 Calibration
`meas.calibrate_from(stream, min_windows=10)` on the empty-room run of the
**same room** (§10.2 step 1). If it raises, the run is refused with an
error naming `calib_s`. A link with fewer than 10 window estimates gets
the median over calibrated links (existing behaviour) and is flagged.

### 4.3 Room Sim packets to frames (glue)
`rate_hz = 1.0 / (float(nT) * slot_s)`, Room Sim's base per-link rate.
For each **received** packet, in Room Sim's order, routed by `(tx, rx)`
(never by Room Sim's `link_index`): `frame = CsiFrame(amplitudes, phases,
sample_index, sample_rate_hz, rssi_dbm, node_id)` with
- `amplitudes[k] = sqrt(float(re*re + im*im))` over the first
  `fw_subcarriers` entries `(re, im)` of the int8 `buffer` (the datagram's
  entries), `re`, `im` widened to 32-bit integers before multiplying
  (Rust `i32::from`); never `hypot` or `CsiFrame::from_iq`;
- `phases[k] = 0.0`; `sample_index = seq`; `sample_rate_hz = rate_hz`;
  `rssi_dbm = rssi`; `node_id = rx`;
- `meas.update(t, tx, rx, frame)` with the packet's `t`.

Lost packets are skipped. With `quantize` false there is no int8 buffer:
the glue refuses that configuration (error naming `quantize`).

## 5. Weighted-link-midpoint baseline (new: `wifisense/localize/baseline.py`)

Inputs: `geometry`, `y` and `null_sigma` (canonical order, length L), the
signal's name; config `wlm_gate_z` (3.0), `wlm_floor` (0.25 for
`rssi_dbm`, 0.0 otherwise), `wlm_radius_scale` (1.0),
`min_plausible_radius_m` (0.3). For link `l`: `m_l = Link.midpoint`,
`d_l = Link.length_m` (ported code: CPython `math.hypot`, Rust
`py_hypot`), `tx`, `rx` = `Link.tx`, `Link.rx`.

1. **Gate.** Link `l` is *disturbed* if `null_sigma_l > 0` and
   `y_l > wlm_gate_z * null_sigma_l + wlm_floor` (NaN fails). If no link is
   disturbed: abstain, `no_disturbance`. (`wlm_gate_z` is not a Gaussian
   z: `y` is a skewed sample variance of integer dBm and `null_sigma` the
   spread of about 14 calibration windows; and with ±1 dB jitter the
   empty-room variance of an integer reading is `0.5 - d*d`, `d` the true
   level's distance from the nearest integer, so a static change of level
   moves it by up to 0.25 dB². The floor covers both; §12a measures the
   result.)
2. **Weights** `w_l = max(y_l, 0.0)` (Python two-argument `max`), every
   link; `W = Σ w_l` (links in order). If `not (W > 0)`: abstain,
   `no_disturbance`.
3. **Estimate** `x̂ = (Σ (w_l * m_l.x)) / W`, `ŷ = (Σ (w_l * m_l.y)) / W`.
4. **Covariance** — a mixture in which each link contributes a uniform
   density along its segment and across its ellipse (`λ = geometry.lambda_m`):
   for a link with `d_l > 0`: `ux = (rx.x - tx.x) / d_l`,
   `uy = (rx.y - tx.y) / d_l`, `a_l = (d_l*d_l) / 12.0`,
   `b_l = 0.5 * sqrt((d_l + λ)*(d_l + λ) - d_l*d_l)`, `c_l = (b_l*b_l) / 3.0`;
   for `d_l <= 0`, `a_l = c_l = ux = uy = 0.0`. With `ex = m_l.x - x̂`,
   `ey = m_l.y - ŷ`:
   ```
   s_xx = (Σ (w_l * ((((a_l*ux)*ux) + ((c_l*uy)*uy)) + ex*ex))) / W
   s_xy = (Σ (w_l * ((((a_l*ux)*uy) - ((c_l*uy)*ux)) + ex*ey))) / W
   s_yy = (Σ (w_l * ((((a_l*uy)*uy) + ((c_l*ux)*ux)) + ey*ey))) / W
   ```
5. **Radius** `r = wlm_radius_scale * radius95(s_xx, s_xy, s_yy)` (§8);
   published only if `x̂`, `ŷ` and `r` are finite and
   `r >= min_plausible_radius_m`, else abstain, `implausible_radius`.

## 6. Phone radio simulation (new: `wifisense/localize/phone.py`, SIM)

ADR-008 d16: simulated only. The phone ranges to every node by Wi-Fi RTT
(IEEE 802.11mc FTM) and reads every node's RSSI, at epochs.

### 6.1 Epochs and truth
Input: a time-ordered sequence of truth samples `j`: `(t_j, phone xyz,
cylinders)` — every person's `(cx, cy, radius, top)` as Room Sim §8.5
defines them — and the blockers (§6.2). Every `t_j`, phone coordinate and
blocker or cylinder number is finite, `t_j >= t_{j-1}` and
`t_j * phone_rate_hz < 2^53`, else the run is refused naming `t`,
`phone`, `walls`, `objects` or `cylinders`; nodes are taken in id order (a
sorted copy), with distinct ids and at least one, else refused naming
`nodes`. A node whose distance `d` (§6.4) is not finite refuses the run
naming `phone`; every one of these checks is made at every truth sample,
before any state changes, whether or not an epoch fires there. Keep `e_next = 0`. At each sample
`j` in order: if `t_j >= float(e_next) / phone_rate_hz`, **one epoch
fires at sample `j`**: its time is `t_j` (never `e/phone_rate_hz`) and its
truth is sample `j`'s; then `e_next += 1` while
`t_j >= float(e_next) / phone_rate_hz` (epochs passed over are dropped:
no draws, no record). At most one epoch fires per sample. Nodes in id
order; node `n` at `(nx, ny, nz)`.

### 6.2 Line of sight
The phone–node segment `P → N` is **blocked** if it meets any wall of Room
Sim's `W` (§3.1, implied walls included) by roomsim §6.1's `hit` with
`o = 0` (`0 <= X.z <= h`); or any object that is not a door or window by
roomsim §7.4's box test; or, for any person (the carrier included), by
roomsim §8.5 steps 1–3's geometry exactly: `L2 == 0`: not blocked;
`u* <= 0` or `u* >= 1`: blocked iff the nearer end is inside the cylinder;
otherwise blocked iff `d_perp < radius` and `z* <= top`. Blocked is a fact
of the geometry, not a draw.

### 6.3 Random numbers
`Rng.stream(phone_seed, id)` (saans_sim's generator, roomsim §2): RTT of
node `n` uses `300000 + n`, RSSI of node `n` uses `310000 + n`. At
creation, per node in id order: RTT stream `off_n = rtt_offset_sigma_m *
gauss()`; RSSI stream `dev_n = rssi_device_sigma_db * gauss()`. Then at
every fired epoch, per node in id order, **always in this order whatever
the geometry or outcome**: RTT stream `u_fail = uniform()`, `g = gauss()`,
`u_out = uniform()`, `u_mag = uniform()`, `u_b = uniform()`; RSSI stream
`u_fail = uniform()`, `g_s = gauss()`, `u_j = uniform()`.

### 6.4 RTT report
```
d   = sqrt(((px-nx)*(px-nx) + (py-ny)*(py-ny)) + (pz-nz)*(pz-nz))
sig = blocked ? rtt_sigma_m * rtt_nlos_sigma_scale : rtt_sigma_m
r   = (d + off_n) + sig * g
if blocked:               r = r + (-(rtt_nlos_bias_m * ln(1.0 - u_b)))
if u_out < rtt_outlier_p: r = r + (rtt_outlier_lo_m + (rtt_outlier_hi_m - rtt_outlier_lo_m) * u_mag)
r   = r > 0.0 ? r : 0.0
```
No report if `u_fail < rtt_fail_p`. (After the clamp `r` is non-finite
only when it is `+inf`, reachable only with absurd configurations; it is
reported and then refused by the fix, naming `reports`. A `-inf` or NaN
`r` before the clamp becomes 0.0, as any negative `r` does.)

### 6.5 RSSI report
```
dd  = d > 0.1 ? d : 0.1
pl0 = 20.0 * log10(FOUR_PI / LAMBDA)                  # free-space loss at 1 m
pl  = dd <= rssi_breakpoint_m
        ? pl0 + 20.0 * log10(dd)
        : (pl0 + 20.0 * log10(rssi_breakpoint_m)) + (10.0 * rssi_far_exponent) * log10(dd / rssi_breakpoint_m)
v   = ((((tx_power_dbm - pl) - (blocked ? rssi_nlos_loss_db : 0.0)) + S) + dev_n) + (-rssi_jitter_db + (2.0*rssi_jitter_db)*u_j)
rssi = clamp(round(v), -100, 0)                       # integer dBm, half-to-even
```
`LAMBDA = C / carrier_hz` with Room Sim's `carrier_hz`. No report if
`u_fail < rssi_fail_p` or if `v` is not finite (its draws and `S` are kept
as computed). **Shadowing** `S` (Gudmundson) is per node and is
computed at every fired epoch for every node, before the `u_fail` test and
whatever it decides: at the first fired epoch `S = rssi_shadow_sigma_db *
g_s`; later `S = rho * S_prev + sqrt(1.0 - rho*rho) * (rssi_shadow_sigma_db
* g_s)` with `rho = exp(-(disp / rssi_shadow_corr_m))`,
`disp = sqrt((px-px_prev)*(px-px_prev) + (py-py_prev)*(py-py_prev))`, and
`S_prev`, `px_prev`, `py_prev` that node's `S` and the phone's `x, y` at the
previous fired epoch.

## 7. Phone estimators (new)

Each method (RTT, RSSI) works on one fired epoch's reports `(n, value)`,
nodes ascending, `m` of them: a fix given its reports in another order
sorts them by node id first (stable), so "report 1" and every `Σ` below
follow node order whatever the caller's. Reports name distinct nodes
with finite values, else the fix is refused (error naming `reports`).
Assumed phone height `zh = phone_assumed_z_m`; `dz_n = zh - nz`. The RSSI
estimator takes `rssi_n` as a float. **Collinear anchors:** both fixes
first form `n00`, `n01`, `n11` and `det` of the *Start* below from their
reports' anchors; if `not (det > 1e-9 * ((n00 + n11)*(n00 + n11)))` there is no fix
(`degenerate`) — a range or level set from anchors on one line cannot tell
the two sides of the line apart. The test does not depend on the line's
direction: it catches anchors collinear to rounding at any angle, and
treats anchors within about 3e-5 of their span from a line as collinear;
it does not catch merely nearly collinear ones. The hall has two
such triples (nodes 0, 5, 6 on x = 0.3; 3, 4, 6 on x + y = 7): an epoch at
which only one triple reports is `degenerate`.

### 7.1 RTT fix
`m == 0`: no fix (`no_reports`); `m < 3`: no fix (`too_few_ranges`).

*Start.* `h_n = sqrt(r_n*r_n - dz_n*dz_n)` if `r_n*r_n > dz_n*dz_n`, else
`0.0`. With report `1` the first: for reports `n = 2..m`:
`A_n = (2.0*(nx_n - nx_1), 2.0*(ny_n - ny_1))`,
`b_n = ((((nx_n*nx_n) + (ny_n*ny_n)) - (nx_1*nx_1)) - (ny_1*ny_1)) - ((h_n*h_n) - (h_1*h_1))`.
`n00 = Σ A_n.x*A_n.x`, `n01 = Σ A_n.x*A_n.y`, `n11 = Σ A_n.y*A_n.y`,
`c0 = Σ A_n.x*b_n`, `c1 = Σ A_n.y*b_n`, `det = n00*n11 - n01*n01`. If
`not (det > 1e-9 * ((n00 + n11)*(n00 + n11)))`: no fix (`degenerate`, the
collinear rule above); else `x0 = (n11*c0 - n01*c1) / det`, `y0 = (n00*c1 - n01*c0) / det`.

*Sums at a point* `(x, y)` (reports ascending, each a Σ): for each report
`dx = x - nx`, `dy = y - ny`, `rho = sqrt((dx*dx + dy*dy) + dz_n*dz_n)`;
skip it if `rho < 1e-9`; `res = r_n - rho`; `u = res / rtt_sigma_m`;
`w = u > huber_k ? huber_k / u : 1.0` (one-sided: only ranges measured too
long are down-weighted); `hx = dx / rho`, `hy = dy / rho`;
`N00 = Σ (w*hx)*hx`, `N01 = Σ (w*hx)*hy`, `N11 = Σ (w*hy)*hy`,
`G0 = Σ (w*hx)*res`, `G1 = Σ (w*hy)*res`; and the objective
`F = Σ f(u)` with `f(u) = u <= huber_k ? 0.5*(u*u) : huber_k*u - 0.5*(huber_k*huber_k)`.
*Degenerate* means `not (det > 1e-12 * (N00*N11))` with
`det = N00*N11 - N01*N01`.

*Gauss–Newton with step halving*, at most 200 iterations from `(x0, y0)`:
form the sums at `(x, y)`; if degenerate: no fix (`degenerate`);
`sx = (N11*G0 - N01*G1) / det`, `sy = (N00*G1 - N01*G0) / det`; `a = 1.0`;
while `not (F(x + a*sx, y + a*sy) <= F(x, y))` and fewer than 30 halvings
have been made, `a = 0.5 * a`; `x = x + a*sx`, `y = y + a*sy`; stop if
`a * sqrt(sx*sx + sy*sy) < 1e-4`. If the stop rule is not met within 200
iterations: no fix (`not_converged`). After the stop the sums are formed
once more at the final `(x, y)`; if degenerate: no fix (`degenerate`);
otherwise the fix is `(x, y)` with covariance `R = (rtt_sigma_m*rtt_sigma_m)
* inv(N)` (each entry) and the diagnostic `chi2 = Σ (u*u)` over the reports
used in that pass (unweighted; approximately χ² with `m - 2` degrees of
freedom under the Gaussian model alone).

### 7.2 RSSI fix
`m == 0`: `no_reports`; `m < 3`: `too_few_ranges`; then the collinear
rule (§7). *Start:* `w_n = 10^(rssi_n / 10.0)`; if `not (Σ w_n > 0)`
(and finite): no fix (`degenerate`); else `x0 = (Σ (w_n*nx)) / (Σ w_n)`,
`y0` likewise.
*Sums at a point:* for each report `dx`, `dy`, `rho` as §7.1; skip it if
`rho < 0.1`; `pred = rssi_p0_dbm - (10.0*rssi_est_exponent) * log10(rho)`;
`res = rssi_n - pred`; `gk = (10.0*rssi_est_exponent) / LN10`;
`jx = ((-gk) * dx) / (rho*rho)`, `jy = ((-gk) * dy) / (rho*rho)`;
`N00 = Σ jx*jx`, `N01 = Σ jx*jy`, `N11 = Σ jy*jy`, `G0 = Σ jx*res`,
`G1 = Σ jy*res`, `F = Σ 0.5*(res*res)`. Then exactly §7.1's loop
(degeneracy, step halving, iteration limit, stop rule, `not_converged`,
final pass), with covariance `R = (rssi_shadow_sigma_db*rssi_shadow_sigma_db)
* inv(N)`. `rssi_p0_dbm` defaults to `tx_power_dbm - 20.0*log10(FOUR_PI /
LAMBDA)` (the true 1 m level; the estimator knows neither the shadowing,
the breakpoint nor the device offsets).

### 7.3 Kalman filter (one per method)
State `s = [x, y, vx, vy]`, covariance `P` (4×4), constant velocity with
continuous white-noise acceleration `q = phone_accel_psd`. The filter keeps
`t_f` (time of its start or last predict) and `t_acc` (time of its start
or last accepted update). At each fired epoch at time `t`, in this order:

1. If a filter exists and `not (t - t_acc <= phone_max_coast_s)`, it is
   discarded.
2. If the epoch has no fix, the epoch abstains with the fix's reason;
   nothing is predicted.
3. If no filter exists, the fix `(z, R)` starts one: `s = [z.x, z.y, 0, 0]`;
   `P` = `R` in the position block, `phone_init_speed_std*phone_init_speed_std`
   on the two velocity diagonal entries, zeros elsewhere; `t_f = t_acc = t`;
   the fix is **published** ungated (§7.4).
4. Otherwise **predict** from `t_f` to `t`: `dt = t - t_f`; `t_f = t`; `F`
   is the identity with `F[0][2] = F[1][3] = dt`; `Q` is zero except, for
   `i` = 0, 1, `Q[i][i] = q*((dt*dt)*dt)/3.0`, `Q[i][i+2] = Q[i+2][i] =
   q*(dt*dt)/2.0`, `Q[i+2][i+2] = q*dt`; `s = F·s`, `P = sym((F·P)·Fᵀ + Q)`.
   Then **gate**: `ν = z - [s0, s1]`, `S = P[0:2][0:2] + R`; if
   `not (det(S) > 0)`: rejected; else `Si = inv(S)`,
   `d2 = ((ν0*Si00 + ν1*Si10)*ν0) + ((ν0*Si01 + ν1*Si11)*ν1)`; rejected if
   `not (d2 <= CHI2_2_99)`. A rejected epoch abstains `gated`; the
   predicted state is kept. Otherwise **update**: `K = P[:, 0:2]·Si` (4×2),
   `s = s + K·ν`, `A = I - K·H` with `H = [I2 0]`,
   `P = sym((A·P)·Aᵀ + (K·R)·Kᵀ)` (Joseph form), `t_acc = t`, and the
   estimate `(s0, s1)` with `P[0:2][0:2]` is **published** (§7.4).

Nothing is published at an epoch without an accepted fix (ADR-008 d6).

### 7.4 Published phone estimate
`(x, y)` and `r = phone_radius_scale * radius95(P00 + B00, P01 + B01,
P11 + B11)` (§8), where `B = (beta*beta) * inv(N)` uses the final normal
matrix `N` of the fix at this epoch (the same `inv(N)` as its `R`), with
`beta = rtt_persist_sigma_m` for RTT (§6.4's per-pair offset, constant for
a run) or `rssi_persist_sigma_db` for RSSI (§6.5's shadowing, constant
while the phone is still). The filter treats fixes as independent, so
errors that persist across epochs are carried only by `B`, which enters
the published covariance and the record's `cov_*`, never `P` or the gate.
Published only if `x`, `y` and `r` are finite and `r >=
min_plausible_radius_m`; else the epoch abstains `implausible_radius` (the
filter is still started or
updated). Reasons, first that applies wins: `no_reports`,
`too_few_ranges`, `degenerate`, `not_converged`, `gated`,
`implausible_radius`.

## 8. The radius

`radius95(a, b, d)` for the covariance `[[a, b], [b, d]]`:
`h = 0.5*(a - d)`; `lmax = 0.5*(a + d) + sqrt(h*h + b*b)`;
`mm = lmax > 0.0 ? lmax : 0.0` (a NaN gives 0.0 in both languages);
result `K95 * sqrt(mm)`. Every publication test below also requires
`r > 0`, so a zero radius is never published even by a caller that passes
a zero minimum to a library function. The circle holds at least 95% of a Gaussian with
that covariance, whatever its shape.

It is a **model** statement and every estimate says so
(`radius_calibrated: false`): forward-model bias (RTI's ellipse), RTT NLoS
bias and outliers, and the RSSI breakpoint and device offsets are in no
covariance; errors that persist across epochs are carried only by §7.4's
`B`. Known: on `localize/sim.py` (SIM) the tracker's χ² circle covered the
truth in 53% of 534 fixes. Phase 4's checks print the coverage they
measure (§12a); Phase 5 sets each `*_radius_scale` from coverage on
held-out runs and only then may a radius be presented as 95%.

- WLM: §5. RTI: the tracker's `cov_xx, cov_xy, cov_yy` at the estimate,
  times `rti_radius_scale`. Phone: §7.4. Every one is published only if
  `x`, `y` and `r` are finite, `r > 0` and `r >= min_plausible_radius_m`
  (written so that NaN fails); otherwise it abstains `implausible_radius`.

## 9. Configuration

| Parameter | Default | Source |
|---|---|---|
| `lambda_m`, `pitch_m`, `alpha`, `sigma`, `delta_c` | 0.3, 0.5, 1.0, 1.0, 0.5 | SIM (the Python defaults; λ is 0.003–0.03 m in the papers, §13) |
| `z_threshold` | 5.0 | SIM (track.py: measured on localize/sim.py, 8×6 m, 8 nodes, 192 pixels, total power; not re-measured for Room Sim or `rssi_dbm`) |
| `window_s` | 1.0 | SIM |
| measurement `sample_rate_hz` | derived: `rate_hz` (§4.3), not a parameter | SIM |
| `min_windows` | 10 | SIM |
| `link_signal` | `rssi_dbm` for Room Sim | SIM (decision 2) |
| `calib_s` | 15.0 | SIM (≥ 10 windows plus margin) |
| `tick_hz` | 2.0 | SIM (the FeatureFrame cadence) |
| `wlm_gate_z`, `wlm_floor` | 3.0; 0.25 for `rssi_dbm`, 0.0 otherwise | SIM |
| `wlm_radius_scale`, `rti_radius_scale`, `phone_radius_scale` | 1.0 | SIM, uncalibrated (§8) |
| `min_plausible_radius_m` | 0.3 | SIM (below a 20 MHz single-antenna system's resolution) |
| `phone_rate_hz` | 5.0 | SIM (Android ranging ~5–10 Hz, ≤ 10 peers) |
| `phone_seed` | the run's seed | SIM |
| `rtt_sigma_m` | 1.0 | SIM->REAL (owner's plan; ESP32-S2 HT20 indoors: 40% of ranges within 2 m) |
| `rtt_nlos_sigma_scale`, `rtt_nlos_bias_m` | 1.5, 1.0 (exponential mean) | SIM (magnitudes from published body/wall NLoS errors) |
| `rtt_outlier_p`, `rtt_outlier_lo_m`, `rtt_outlier_hi_m` | 0.05, 2.0, 8.0 | SIM |
| `rtt_offset_sigma_m`, `rtt_persist_sigma_m` | 0.5, 0.5 | SIM (residual after per-pair calibration) |
| `rtt_fail_p`, `rssi_fail_p` | 0.02, 0.02 | SIM |
| `rssi_breakpoint_m`, `rssi_far_exponent` | 5.0, 3.5 | REAL (IEEE 802.11 TGn channel models B/C) |
| `rssi_shadow_sigma_db`, `rssi_persist_sigma_db` | 4.0, 4.0 | SIM (TGn: 3 dB below 5 m, 4–5 dB beyond) |
| `rssi_shadow_corr_m` | 6.0 | SIM (3GPP TR 38.901 InH-Office NLOS; LOS is 10 m) |
| `rssi_nlos_loss_db` | 5.0 | SIM |
| `rssi_device_sigma_db` | 3.0 | SIM ("several dB" per board) |
| `rssi_jitter_db` | 1.0 | SIM (as Room Sim's) |
| `rssi_est_exponent` | 2.0 | SIM |
| `rssi_p0_dbm` | `tx_power_dbm - 20.0*log10(FOUR_PI/LAMBDA)` | SIM |
| `phone_assumed_z_m` | 1.0 | SIM |
| `huber_k` | 1.345 | SIM (Huber's constant for 95% efficiency of the two-sided estimator; one-sided here: 96.9% efficiency, each range fitted about 0.046·σ short) |
| `phone_accel_psd`, `phone_init_speed_std`, `phone_max_coast_s` | 1.0, 1.0, 3.0 | SIM (on smooth 1 m/s walks 0.1 lags and the gate rejects 20–25% of RTT epochs, against 12–14% at 1.0) |
| `tx_power_dbm`, `carrier_hz` | Room Sim's | SIM |

**Validation** names the parameter (when several are wrong, which is named
is not specified), every test NaN-safe. Every number finite;
`phone_seed` an integer in 0..2^64−1. `> 0`: `rtt_sigma_m`,
`rssi_shadow_sigma_db`, `rssi_shadow_corr_m`, `rssi_breakpoint_m`,
`phone_rate_hz`, `huber_k`, `rssi_far_exponent`, `rssi_est_exponent`,
`phone_accel_psd`, `phone_init_speed_std`, `tick_hz`, `calib_s`,
`window_s`, `carrier_hz`, `min_plausible_radius_m` and the three
`*_radius_scale` (a zero radius would publish a bare point, ADR-008 d5).
`>= 0`: `rtt_nlos_sigma_scale`, `rtt_nlos_bias_m`, `rtt_offset_sigma_m`,
`rtt_persist_sigma_m`, `rssi_nlos_loss_db`, `rssi_device_sigma_db`,
`rssi_persist_sigma_db`, `rssi_jitter_db`, `phone_max_coast_s`,
`wlm_gate_z`, `wlm_floor`. In 0..1 inclusive: `rtt_outlier_p`,
`rtt_fail_p`, `rssi_fail_p`. `rtt_outlier_lo_m <= rtt_outlier_hi_m`.
`min_windows` an integer `>= 1`. `link_signal` one of `rssi_dbm`,
`total_power_db`, `static_suppressed_amplitude`. `phone_seed`,
`tx_power_dbm` and `carrier_hz` default to "unset" and are then taken from
Room Sim (§10.2); set, they must equal Room Sim's or a Room Sim run is
refused naming them (a plain-data phone run keeps a set `phone_seed`, and
`rssi_p0_dbm` unset is always derived as the table says). The measurement's `sample_rate_hz` is derived (§4.3), never a
parameter. For a Room Sim run, `tick_hz <= 1.0 / slot_s` (at most one tick
per slot), else refused naming `tick_hz`. The ported modules' own checks
apply to their parameters.

## 10. Pipelines

### 10.1 One tick
Ticks are at `t_k = window_s + float(k) / tick_hz`, computed so and never
accumulated, for every `k = 0, 1, …` with `t_k < duration_s`; a run in
which some `t_{k+1}` is not greater than `t_k` is refused naming
`tick_hz`. At tick
`k` the measurement has received exactly the received packets with
`t <= t_k`; then `y = meas.vector(t_k)`, WLM (§5) from `y` and
`null_sigma`, and RTI: `image = inverter.solve(y)`, `est =
tracker.update(image, t_k)`. RTI abstains `not_significant` when `est` is
None (both of `update`'s None paths; `extract_peak` is not called again),
and `implausible_radius` when its radius fails §8 (the tracker has then
already been corrected and its state is kept).

### 10.2 Room Sim run (`roomsim` glue)
1. **Calibration:** a Room Sim run with the same scene, config and
   **seed**, but `n_people = 0`, `people_mode`, `people_posture` and
   `start_heading_rad` emptied (Room Sim refuses their indices with no
   people), and `duration_s` = the occupied run's `duration_s + calib_s`.
   Only its packets with `t >= duration_s` go through §4.3 into
   `calibrate_from`, which clears its windows before and after. The same
   seed keeps the occupied run's clutter, tail and devices (streams 1,
   100000 + pair, 500 + n): the same room. Packets at `t >= duration_s`
   use per-link draws (stream 1000 + link) the occupied run never
   consumed, so the two runs share no noise. No seed arithmetic is done.
2. **Occupied run** (the config as given): packets in order into the
   measurement; one tick record per §10.1.
3. **Phone:** truth samples from every transmission, with each person's
   cylinder from Room Sim (§8.5: radius `body_radius_m`, top by posture and
   height), into §6; §7 at each fired epoch; one epoch record each.

## 11. Records

Per tick: `{label, t, wlm, rti, truth}`; per fired phone epoch:
`{label, t, rtt, rssi, reports, truth}`. `wlm`, `rti`, `rtt`, `rssi` are
each an **Estimate** `{x, y, radius_m, cov_xx, cov_xy, cov_yy, confidence,
radius_calibrated}` (`confidence`: the tracker's for RTI, `null`
otherwise; `radius_calibrated`: false in Phase 4) or an **Abstention**
`{reason}`. Reasons: WLM `no_disturbance`, `implausible_radius`; RTI
`not_significant`, `implausible_radius`; phone §7.4. A tick's `truth` is
Room Sim's truth of the last transmission with `t <= t_k`, per person
`people[i]: {x, y, walking, posture}`; an epoch's `truth` is its sample's
phone `(x, y, z)`. `reports`: one entry per node in id order,
`{node, rtt_m or null, rssi_dbm or null, blocked}`. No `Person.id` or
`Phone.id` appears in any record, log or error. Records are SIM, of
simulated people, and may be written to golden files as Room Sim's truth
is; nothing here writes records from REPLAY or LIVE input (ADR-008 d11).

## 12. Agreement

Rust matches Python by spec/roomsim.md §14's rule (1e-9 relative + 1e-12
of the array's scale), and every discrete output — supports, groups,
blind pixels, peak index, accept/abstain, reasons, gating, reports
present, blocked flags — identically. Phone RSSI integers are identical
unless the recorded pre-rounding `v` is within 1e-6 of a .5 tie. On glibc
both implementations agree bit for bit; the tolerance is for other libms.
Golden sets from `spec/golden/generate.py` on CPython 3.14 (`localize_*`
for the ported modules, `locate_*` for the new pieces and the Room Sim
pipeline); each records its inputs (recorded arrays for any random input,
never MT19937 draws), `scene_sha256` where a scene is used, and for weight
supports the smallest boundary margin `|d_tx + d_rx - d_l - λ|`. Array
layouts are the docstrings of the generators. `py_hypot` is checked bit for
bit against CPython 3.14 on 1,403 recorded edge pairs (zeros,
subnormals, equal magnitudes, ratios near 2^±27) and on 10^6 pairs drawn
in both languages from `Rng.stream(0, 320000)` (`range(-10.0, 10.0)` twice
per pair) whose results' SHA-256 (little-endian f64) is recorded.

## 12a. Checks with known geometry (both implementations)

- `radius95`: identity → `K95`; `[[4,0],[0,1]]` → `2*K95`; `[[2,1],[1,2]]`
  → `K95*sqrt(3)`; zero matrix → abstains `implausible_radius`.
- WLM: one link with `y > 0` → that link's midpoint, covariance
  `a_l·uuᵀ + c_l·vvᵀ`; a square's two diagonals equally disturbed → the
  centre; nothing over the gate → `no_disturbance`.
- RTT: noiseless reports (zero offsets, outliers and failures,
  `phone_assumed_z_m` = the true z) from the bedroom's 4 nodes recover the
  truth within 1e-6 m; the office's collinear nodes 0, 4, 1 alone →
  `degenerate` for **every** phone position (both fixes; checked on a
  0.1 m grid over the room); one range +5 m long moves the one-sided Huber
  fix less than plain least squares (both printed, SIM).
- RSSI: noiseless reports from the estimator's own model recover the truth
  within 1e-6 m.
- Line of sight: a segment through a wall centreline is blocked, through a
  door's opening (no wall) not; through a cylinder below its top blocked,
  above it not.
- Kalman: a fix 10σ off is `gated` and nothing is published; the filter is
  discarded after `phone_max_coast_s` without an accepted update.
- No published estimate anywhere has a non-finite radius or one below
  `min_plausible_radius_m`.
- **Measured and printed** (SIM, each shipped scene, Rust example
  `locate_measure`), each line labelled SIM: WLM's and RTI's empty-run
  false alarms, as published ticks per hour and as events (runs of
  consecutive published ticks) per hour; on occupied runs their report
  coverage beside the empty run's publish fraction, and median/p90 error
  to the nearest person (and, on walking ticks, to the walker), split by
  walking, with WLM and RTI also compared on the ticks both published;
  every published radius's median and p90, the fraction of estimates
  whose truth is within `radius_m`, and the same figures for a **fixed
  reference point** (the footprint's centroid) on the same ticks, with the
  same radius — the score of an estimator that knows nothing; the phone's
  per-method coverage, error, radius and radius coverage, each line saying
  whether the phone moved, with per-run figures where it did not (a still
  phone's shadowing and offsets are constant for a run, so its epochs are
  one sample). These are reported numbers, not pass/fail tests.

## 13. Limits

- RTI's ellipse model and the defaults are not published values (λ
  0.3 m against 0.003–0.03 m in the papers); NOTES-localize.md's numbers
  come from another simulator and are not comparable.
- RSSI variance is a narrow signal (integer dBm, ±1 dB jitter); bodies
  away from a link's line move it little, and RTI on it in a 4-node room
  carries almost no position (decision 7). Lowering `z_threshold` does not
  help: noise reaches the same z.
- Rectangular grid over an L-shaped footprint: pixels outside it exist.
- Single target; zones absent; FeatureFrame input is Phase 5's; scenes
  with `roles` or stacked nodes are refused by the person pipeline.
- The phone model is parametric from geometry: no multipath, no
  frequency-selective ranging error, a constant per-pair offset, no
  receiver AGC; the estimator knows the noise scales it assumes and no
  more.
- Every radius is uncalibrated until Phase 5 measures coverage.

**What the §12a measurements show** (SIM; `locate_measure`, 120 s × 5
seeds per variant, defaults; read with the fixed reference point beside
each figure):
- **WLM's radius is room-sized** (median 3.4–3.6 m bedroom, 4.4–4.6 m
  office, 5.4–6.3 m hall), so its "inside 100%" says nothing: the
  footprint centroid with the same radius is inside as often. Its
  empty-room false alarms: 0 per hour (bedroom), 30 events per hour
  (office), 127 (hall). It publishes on 95–100% of walking ticks, with
  median error to the nearest person 0.71 m (bedroom), 1.06 m (office),
  1.57–1.73 m (hall), where the reference point scores 0.89, 1.55 and
  1.59–2.51 m. On still ticks it can do worse than the reference point:
  bedroom scene run 1.43 against 1.28 m; hall walker run 1.97 against
  1.84 m (1.68 against 1.61 m over all ticks).
- **RTI on `rssi_dbm` does not localise in the bedroom or the office**: it
  publishes about as often in occupied runs as in the empty room, not
  distinguishably more
  (bedroom 0.2–0.5% against 0.5%; office 1.4–5.5% against 4.6%). In the
  hall it publishes on 76–80% of walking ticks and beats WLM on the ticks
  both publish (median 1.44 vs 1.65 m), but it also publishes on 17.6% of
  empty-room ticks (944 events per hour), and its radius covers the truth
  in only 34–52% of its estimates.
- **Phone RTT** is the useful phone method in the hall and the office:
  median error 0.48–0.98 m when still and 0.70–0.84 m when carried,
  against the reference point's 1.59–3.22 m (the office's empty runs
  excepted, below). In the occupied bedroom it is not: 1.25 m (scene,
  still) and 1.19 m (walker) against the reference point's 1.10 and
  1.19 m, whose circle with RTT's own radius covers the phone in 100% and
  84.8% of epochs against RTT's 68.9% and 69.7%. Its radius covers the
  truth in 69–98% of estimates. Gating rejects 14–37% of fixes in the
  bedroom (blocked paths).
- **Phone RSSI** publishes on 95–100% of epochs but is no better than the
  reference point in the bedroom (median 1.5–2.3 m against 1.1–1.3 m) or
  in the occupied office with the phone still (2.34 m against 1.59 m), and
  its radius (median 2.7–3.9 m) is near room-sized. A still phone's
  figures are per-run outcomes (5 runs): office seed 1's radius covered the
  truth in 4.8% of epochs.
- In the office's empty runs the still phone sits exactly at the footprint
  centroid, so the reference point scores 0 there.
- These runs ran 45 at once (about 12 s of wall time on this 96-core
  host; about 5 CPU-minutes in all).

None of this is a measurement. Every number from it is SIM.
