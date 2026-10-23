# Room Sim v1 — physics, people and ESP32 impairments

**Status:** spec for Phase 3, 2026-10-21, revised after a four-lens review
(53 confirmed findings). Implemented twice — the Python reference
`wifisense/roomsim/` and the Rust crate `saans-roomsim` — and the two must
agree (§14).

Everything this simulator produces is **SIM**: a physically motivated
model, not a measurement. It reads a room from a scene file (SCENE,
ADR-008) and computes what ESP32 nodes in that room would report. Read §16,
what it does not model, before drawing a conclusion from its output.

**Where it lives — a decision recorded for the owner.** The Phase 3 brief
says "Rust saans-sim + Python reference". This is implemented as a **new**
crate, `saans-roomsim`, and a **new** package, `wifisense/roomsim/`,
because that changes no existing public API (the owner's rule) and leaves
`saans-sim`, `wifisense/sources/synthetic.py` and the `sim_geometric`
golden vectors — which the S3 detector evaluation depends on — exactly as
they are. The alternative, extending `saans-sim`, changes its public types
(`Scenario`, `Room`, `SimOptions`, `LinkFrame`, `Tick`) and so needs the
owner's approval first. `wifisense/localize/sim.py` stays the simulator
behind NOTES-localize.md; its numbers are not comparable with Room Sim's,
whose shadowing law differs.

**Rules.** `wifisense/roomsim` imports only the standard library and
`wifisense`. `saans-roomsim` starts with `#![forbid(unsafe_code)]` and
depends on `saans-scene` **without** its `write` feature (ADR-008
decision 1; `saans-scene/tests/boundary.rs` lists it as a reader). Where
this spec and the Python reference disagree, the Python reference and its
golden vectors are normative (ADR-002) and this spec is corrected in the
same change. The gateway's `roomsim` cargo feature and `/roomsim` page are
the Phase 2 demo and unrelated to this crate.

---

## 1. Conventions

**Units.** Metres, seconds, hertz, radians, decibels where named `_db`.
Scene frame (ADR-008): right-handed, +z up, origin at a room corner,
floor at z = 0.

**Evaluation.** IEEE-754 binary64. Every formula is evaluated exactly as
written: binary operators left-associative, parentheses first, one
rounding per operator. No fused multiply-add, no FP contraction, no
fast-math, no reassociation. `x^2` means `x*x`. `10^y` means `pow(10.0, y)`
(Python `10.0 ** y`, Rust `10f64.powf(y)`). `sqrt`, `sin`, `cos`, `exp`,
`ln`, `log10`, `atan2` are the platform's (libm). **`sin` and `cos` are
separate calls, never one `sincos`**: glibc's `sincos` differs from its
`sin` and `cos` in the last bit for about one argument in 1500, and an
optimising Rust build merges a `sin` and a `cos` of the same argument into
it, so `saans-roomsim` passes each argument through `std::hint::black_box`
(its `numerics::sin` and `numerics::cos`). **`hypot` is never
used** (CPython's `math.hypot` and libm's differ in the last bit); a 2-D
length is `sqrt(dx*dx + dy*dy)`. Numbers read from a scene or config file
are converted with correct rounding (Python `float`; Rust `serde_json`
with `float_roundtrip`).

**Constants**, computed as written:

| Name | Definition | Value |
|---|---|---|
| `PI` | the language's π | 3.141592653589793 |
| `TWO_PI` | `2.0 * PI` | 6.283185307179586 |
| `FOUR_PI` | `4.0 * PI` | 12.566370614359172 |
| `FOUR_PI_1_5` | `FOUR_PI * sqrt(FOUR_PI)` | 44.546623974653656 |
| `C` | speed of light | 299792458.0 |
| `EPS0` | vacuum permittivity | 8.8541878128e-12 |
| `MIN_RANGE_M` | range clamp for point scatterers | 0.1 |
| `LAMBDA` | `C / carrier_hz` | 0.1230170118998769 at channel 6 |

**Complex numbers** are pairs `(re, im)`:

```
cadd(a, b)   = (a.re + b.re, a.im + b.im)           csub likewise
cmul(a, b)   = (a.re*b.re - a.im*b.im,  a.re*b.im + a.im*b.re)
cscale(a, s) = (a.re*s, a.im*s)
cdiv(a, b):  d = b.re*b.re + b.im*b.im
             ((a.re*b.re + a.im*b.im) / d,  (a.im*b.re - a.re*b.im) / d)
csqrt(z):    if z.re == 0 and z.im == 0: (0, 0)
             m = sqrt(z.re*z.re + z.im*z.im)
             if z.re >= 0: t = sqrt((m + z.re) / 2);  (t, z.im / (2*t))
             else:         t = sqrt((m - z.re) / 2);  (|z.im| / (2*t), copysign(t, z.im))
cexpj(th)    = (cos(th), sin(th))
cexp(z)      = cscale(cexpj(z.im), exp(z.re))
```

**Neumaier** (`hi = lo = 0`; `add(x)`: `t = hi + x`; if `|hi| >= |x|`
then `lo += (hi - t) + x` else `lo += (x - t) + hi`; `hi = t`;
`value()`: `hi + lo` if `lo != 0` and `lo` is finite, else `hi`) is used for
**exactly** these sums, terms in the order given: `H_k` (§11, re and im
separately); the two HT-LTF means of §12.3 (`value() / 56.0`); `W` in
§8.3; a path length `L` over its legs (§6); `P_spec` of §6.8;
the two harmonic sums of §12.1.
Every other sum (dot and cross products, rotations, smoothing taps) is
plain left to right. A product of factors is accumulated as `S = 1.0`,
then `S = S * f` in the stated order.

**Rounding to an integer**: round-half-to-even (Python `round`, Rust
`f64::round_ties_even`). **clamp**`(x, lo, hi) = min(max(x, lo), hi)`.
**sign**`(d)` is `1` if `d > 0`, `-1` if `d < 0`, else `0` (`-0.0` is 0;
never Rust's `f64::signum`).

**Phase of a path** of length `L` at frequency `f`:
`theta = ((-TWO_PI) * f) * L / C`. A path term is `cmul(A, cexpj(theta))`;
a longer path lags.

## 2. Random numbers

One generator: `saans_sim::rng::Rng` (xoshiro256** seeded through
SplitMix64), with its exact `new`, `stream`, `next_u64`, `uniform`,
`range`, `gauss` (Marsaglia polar with a cached spare), `exp` and
`unit_vector`. The Python reference ports it and must reproduce:

```
Rng(7).next_u64() x3                = 12923355070828475994, 5142052590334782674, 15488392906492639638
Rng.stream(42, 1000).next_u64() x3  = 12345297164789059879, 1259461603951423963, 15861779893704225696
Rng.stream(42, 1): uniform           = 0.924814964327024258
                   range(0.3, 5.7)   = 4.97446416539475589
                   gauss x3          = -0.793888932810311143, -1.59718765083500513, -0.484773194911475802
                   exp(3.0)          = 2.50672202106860942
                   unit_vector       = (0.722861780515479668, 0.690992652833581578)
```

Streams, all `Rng.stream(cfg.seed, id)`:

| id | Draws | Section |
|---|---|---|
| `1` | clutter points | §6.7 |
| `100 + i` | person `i`'s body parameters | §8.1 |
| `200 + i` | person `i`'s start (when drawn) and motion | §9 |
| `500 + node_id` | node `node_id`'s device | §12.1 |
| `1000 + link_index` | link `link_index`'s per-packet impairments | §12.2 |
| `100000 + pair_index` | the diffuse tail of an unordered node pair | §6.8 |

**Draw sequences never depend on switches.** Every listed draw happens in
full whether or not its feature is enabled; a switch changes only how a
drawn value is used. People are limited to 100 so stream ids cannot
collide.

## 3. Inputs

### 3.1 The scene

A `saans-scene` scene (`spec/scene.schema.json`), read only. Both
implementations reject any scene `saans_scene::Scene::validate` rejects
(the Python reference ports `validate.rs`'s checks with stdlib `json`), and
also reject: two nodes at identical `(x, y, z)`; a node strictly inside any
object's box (§7.4), doors and windows included; a node within a wall's
thickness, `pdist(node.xy, A, B) < t/2 + 0.01` (§7.2) for any wall of `W`
(implied walls included) with `node.z <= h` (the 1 cm is saans-scene's
`NODE_WALL_CLEARANCE_M`, measured here from the face rather than the
centreline); more than 100 people.
A node id outside 0..254 is the §3.2 error naming `nodes`. (The port
computes 2-D lengths as `sqrt(dx*dx + dy*dy)` where `validate.rs` uses
`hypot`, so a scene within an ulp of one of its tolerances may be judged
differently by the two; such a scene is not a valid test input.)

- **Footprint** `F`: `room.polygon`, or `[(0,0), (w,0), (w,d), (0,d)]`.
  `H` = `room.h_m`. Bounding box `(x0, y0)–(x1, y1)`.
- **Walls** `W[0..nW)` in file order: centreline `A → B`, height
  `h = height_m` or `H`, thickness `t`, material. With **no** walls listed,
  the footprint's edges, in footprint order, become walls with
  `implied_wall_material` and `implied_wall_thickness_m`; the run manifest
  records them as SIM assumptions. Per wall:
  `len = sqrt((B-A).x^2 + (B-A).y^2)`, `e = (B-A)/len`,
  `n = (-e.y, e.x, 0)`.
- **Objects** `O[0..nO)` in file order: type, centre `(cx, cy, cz)`, size
  `(sl, sb, sh)`, yaw `psi = (yaw_deg * PI) / 180.0` (exactly so; not
  `radians()`/`to_radians()`), material, `cs = cos(psi)`, `sn = sin(psi)`.
- **Nodes**, sorted by id ascending, each at `(x, y, z)`.
- **People** `scene.people` in file order (`n_people`, when set, takes the
  first `n_people`, or adds people with no start). A person is named
  `people[i]` in every message; **no `Person.id` or `Phone.id` appears in
  any output, log or manifest.**
- **Phone**: optional.

**Local ↔ world.** Object-local `(lx, ly, lz)` to world:
`x = cx + (lx*cs - ly*sn)`, `y = cy + (lx*sn + ly*cs)`, `z = cz + lz`.
World to object-local: `dx = p.x - cx`, `dy = p.y - cy`;
`lx = dx*cs + dy*sn`, `ly = -dx*sn + dy*cs`, `lz = p.z - cz`. Body or phone
local `(bx, by, bz)` with heading `hd` (`ch = cos(hd)`, `sh = sin(hd)`):
`x = px + (bx*ch - by*sh)`, `y = py + (bx*sh + by*ch)`, `z = bz`. `+y` is to
the left of the heading.

### 3.2 Configuration

`Source`: **REAL** = from the firmware or a published standard; **SIM** =
a modelling assumption, to be calibrated in Phases 9–10.

| Group | Parameter | Default | Source / note |
|---|---|---|---|
| run | `seed` | 0 | |
| | `duration_s` | 10.0 | packets with `t < duration_s` |
| radio | `channel` | 6 | REAL. 1..13; `carrier_hz = (2412 + 5*(channel-1)) * 1e6` |
| | `spacing_hz` | 312500.0 | REAL (802.11) |
| | `tx_power_dbm` | 10.0 | SIM. Effective EIRP, isotropic antennas (ESP-IDF's default 20 dBm conducted, less PCB-antenna and enclosure losses at both ends) |
| TDMA | `slot_s` | 0.0125 | SIM: the planned broadcast TDMA (`edge/dsp/tdma.hpp`, itself a model: 4 nodes → 20 Hz per link); today's firmware has no TDMA (§10) |
| | `roles` | `{}` | node id → `txrx` (default), `tx` or `rx` |
| | `tx_period` | `{}` | node id → integer ≥ 1 (default 1) |
| surfaces | `reflection_order` | 2 | 0, 1 or 2 |
| | `floor_ceiling` | true | |
| | `floor_material`, `ceiling_material` | `concrete` | SIM; a wall material (validation) |
| | `slab_thickness_m` | 0.2 | SIM |
| | `transmission` | true | |
| | `implied_wall_material` | `drywall` | SIM; a wall material (validation) |
| | `implied_wall_thickness_m` | 0.1 | SIM |
| objects | `object_specular` | true | |
| | `object_scatter` | true | |
| | `object_scatter_scale` | 1.0 | SIM |
| | `object_blockage` | true | |
| clutter | `clutter_count` | 24 | SIM |
| | `clutter_rcs_m2` | (0.005, 0.2) | SIM, log-uniform |
| | `clutter_margin_m` | 0.1 | |
| tail | `tail` | true | |
| | `tail_count` | 40 | SIM |
| | `tail_power_db` | -15.0 | SIM, relative to specular power |
| | `tail_decay_s` | 20e-9 | SIM; the tail's power-delay profile decays as `exp(-τ/tail_decay_s)` |
| people | `n_people` | unset | scene count; ≤ 100 |
| | `people_mode` | `{}` | index → `walk` or `stationary` (default §8.2) |
| | `people_posture` | `{}` | index → `stand`, `sit` or `lie` (default §8.2) |
| | `start_heading_rad` | `{}` | index → heading of a stationary person (default 0.0) |
| | `body_points` | 10 | 6..16 |
| | `body_rcs_m2` | 0.5 | SIM |
| | `body_radius_m` | 0.18 | SIM |
| | `height_m` | (1.55, 1.85) | SIM |
| | `breathing_hz` | (0.1, 0.3) | brief |
| | `breathing_p2p_m` | (0.004, 0.012) | brief |
| | `speed_mps` | (0.3, 1.2) | brief; matches localize/sim.py |
| | `pause_s` | (0.0, 20.0) | matches localize/sim.py |
| | `walk_margin_m` | 0.4 | |
| | `object_clearance_m` | 0.05 | |
| | `shadowing` | true | |
| | `shadow_cap_db` | 20.0 | SIM |
| phone | `phone_carrier` | 0 | person index |
| | `phone_offset_m` | (0.05, -0.25, 1.0) | body frame |
| impairments | `impair` | true | master switch (§12.5) |
| | `loss_prob` | 0.02 | SIM, no measured figure |
| | `loss_prob_link` | `{}` | `"tx,rx"` → probability |
| | `agc_jump_prob` | 0.01 | SIM |
| | `agc_step_db` | 3.0 | SIM |
| | `agc_up_prob` | 0.5 | SIM |
| | `agc_initial_db` | 0.0 | |
| | `agc_max_db` | 6.0 | SIM |
| | `device_gain_db` | (-2.0, 2.0) | SIM |
| | `cpe` | true | false: `phi0 = 0` (§12.3 step 4); the draw still happens |
| | `timing_offset_s` | 50e-9 | SIM, ± per packet |
| | `timing_offset_dev_s` | (0.0, 0.0) | SIM, per node |
| | `ripple` | true | |
| | `ripple_amp` | 0.05 | SIM |
| | `ripple_phase_rad` | 0.1 | SIM |
| | `ripple_harmonics` | 3 | |
| | `ripple_period_bins` | 64.0 | |
| | `snr_noise` | true | |
| | `chip_noise_counts` | 0.5 | SIM, per component |
| | `noise_floor_dbm` | (-97.0, -92.0) | SIM |
| | `rssi_jitter_db` | 1.0 | SIM |
| | `rssi_range_dbm` | (-100, 0) | |
| | `target_counts` | 24.0 | SIM |
| | `ltf_merge` | true | REAL (`ltf_merge_en = true`) |
| | `ltf_merge_weight` | 0.5 | SIM |
| | `smoothing` | **false** | REAL (`channel_filter_en = false`, csi_capture.cpp:112) |
| | `smoothing_kernel` | (0.25, 0.5, 0.25) | |
| | `smoothing_edge` | (2/3, 1/3) | |
| output | `quantize` | true | int8; false keeps floats, emits no datagram |
| | `fw_subcarriers` | 56 | REAL (Kconfig default); 8..64 |

**Validation.** A run is rejected with an error naming the parameter
(when several are wrong, which one is named is not specified). An unknown
parameter is an error; every number is finite, every count and id an
integer (`6.0` is not), every switch a boolean. Then:

- `seed` in 0..2^64−1; `duration_s ≥ 0`; `channel` in 1..13;
  `spacing_hz > 0`; `slot_s > 0`.
- Node ids 0..254 (named `nodes`); `roles` keys are node ids of the scene
  and values `txrx`, `tx` or `rx`; `tx_period` keys are node ids of the
  scene and values integers ≥ 1; at least one link, i.e. a transmitter and
  a different receiver (named `roles`; a one-node scene has none).
- Every integer map key (`roles`, `tx_period`, `people_mode`,
  `people_posture`, `start_heading_rad`) is canonical decimal: ASCII
  digits only, no sign, spaces, underscores or leading zeros (`"0"` is
  the only key starting with `0`); otherwise the error names the map.
- `reflection_order` 0, 1 or 2; `floor_material`, `ceiling_material` and
  `implied_wall_material` one of `concrete`, `brick`, `drywall`, `glass`,
  `wood` (`metal` is for objects only, §5.1); `slab_thickness_m > 0`;
  `implied_wall_thickness_m > 0`.
- `object_scatter_scale ≥ 0`; `clutter_count ≥ 0`; `clutter_margin_m ≥ 0`;
  `tail_count ≥ 0`; `tail_decay_s > 0`.
- Decibel and scale parameters stay where every formula is finite:
  `tx_power_dbm` in −100..60; `tail_power_db` in −100..30;
  `agc_step_db`, `agc_max_db` in 0..60; `agc_initial_db` in −60..60;
  `device_gain_db` lo and hi in −60..60; `rssi_jitter_db` in 0..60;
  `ripple_amp` in 0..1; `ripple_phase_rad ≥ 0`; `timing_offset_s` in
  0..1e-3 and `timing_offset_dev_s` lo and hi in −1e-3..1e-3;
  `chip_noise_counts` in 0..1000; `target_counts` in 0..1000 and `> 0`;
  `breathing_p2p_m` lo and hi in 0..1; `height_m` lo and hi in 0..3;
  `|ripple_period_bins| ≥ 1`.
- Every `(lo, hi)` pair has `lo ≤ hi`; `clutter_rcs_m2` lo `> 0` (it is
  log-uniform); `speed_mps` lo `> 0`; `pause_s` lo `≥ 0`;
  `noise_floor_dbm` within −127..−1; `rssi_range_dbm` integers within
  −128..127 (the datagram's RSSI is an int8).
- `n_people` in 0..100; `people_mode`, `people_posture` and
  `start_heading_rad` keys are indices `0..n−1` of simulated people (`n`
  the scene's count, or `n_people`); `body_points` in 6..16;
  `body_rcs_m2`, `body_radius_m`, `walk_margin_m`, `shadow_cap_db` ≥ 0;
  `phone_carrier ≥ 0` (one with no such person leaves the phone at its
  start, §9).
- `loss_prob`, `agc_jump_prob`, `agc_up_prob`, `ltf_merge_weight` and every
  `loss_prob_link` value in 0..1; a `loss_prob_link` key is exactly the
  key §12.2 looks up, `"tx,rx"` with two node ids of the scene in decimal
  (no spaces, signs or leading zeros); `ripple_harmonics ≥ 0`;
  `fw_subcarriers` in 8..64.

A number outside the binary64 range (`1e400`) is a JSON error, which an
implementation may report by its position in the text rather than by
parameter.

A person who cannot be placed as configured (§8.2, §9) is an error naming
`people[i]` and, when one is at fault, the parameter (`people_mode` or
`people_posture`).

## 4. Frequency grid and the ESP32 buffer

Bins `k ∈ [-32, 31]`, `f_k = carrier_hz + float(k) * spacing_hz`. The
clean channel is reported for all 64 bins in ascending `k`. Active bins:

- **LLTF** (legacy, HT20): `k ∈ [-26, -1] ∪ [1, 26]` — 52 bins.
- **HT-LTF** (HT20): `k ∈ [-28, -1] ∪ [1, 28]` — 56 bins.

Every other bin of an LTF is exactly `0 + 0j`. An LTF is laid out in
**FFT order** `k = 0, 1, …, 31, -32, …, -1`. The ESP-IDF buffer for an HT20
packet is LLTF (64 entries) then HT-LTF (64 entries): 128 entries, each
**imaginary first** as two int8.

> **Assumptions, not verified on hardware.** (1) This layout is what
> ESP-IDF documents for ESP32 packets with `sig_mode = 1` (HT),
> `cwb = 0` (20 MHz), `stbc = 0`, no secondary channel and
> `first_word_invalid = false`; the firmware guarantees none of these.
> (2) The firmware forwards the first `CONFIG_SAANS_N_SUBCARRIERS` = 56
> entries unchanged (`csi_capture.cpp`). If the layout holds, those are
> LLTF bins `0..31` and `-32..-9`: 44 carry a value and 12 are DC and guard
> nulls — not the "56 usable subcarriers (LLTF+HT-LTF)" the Kconfig help
> says, nor the HT-LTF subcarriers `csi_capture.cpp`'s comment expects.
> (3) The TX's ESP-NOW frames may not be HT at all: `tx_role.cpp` sets no
> ESP-NOW PHY rate, and `tdma.hpp` notes broadcast falls back to 1 Mbps
> DSSS, whose frames carry no LTF. For a legacy 11g frame the buffer is
> LLTF only and a datagram with `fw_subcarriers ≤ 64` is byte-identical,
> because its first entries are LLTF either way. Checking all three on an
> ESP32-S3 is an open item in ROOMSIM.md (log `info->len`,
> `rx_ctrl.sig_mode`, `rx_ctrl.rate`, `cwb`, `first_word_invalid`).

The simulator copies the firmware exactly (§12.4), so it reproduces
whatever is true of this layout.

## 5. Materials

### 5.1 Walls, floor, ceiling: ITU-R P.2040 slabs

`er = a * pow(f_ghz, b)`, `sigma = c * pow(f_ghz, d)` (S/m),
`f_ghz = carrier_hz / 1e9`, from ITU-R P.2040-3 Table 3 (checked against
Sionna RT 2.2.0, `radio_materials/itu.py`):

| Scene material | ITU material | a | b | c | d |
|---|---|---|---|---|---|
| `concrete` | concrete | 5.24 | 0 | 0.0462 | 0.7822 |
| `brick` | brick | 3.91 | 0 | 0.0238 | 0.16 |
| `drywall` | plasterboard | 2.73 | 0 | 0.0085 | 0.9395 |
| `glass` | glass | 6.31 | 0 | 0.0036 | 1.3394 |
| `wood` | wood | 1.99 | 0 | 0.0047 | 1.0718 |
| `metal` (objects) | metal | 1.0 | 0 | 1e7 | 0 |

(`floorboard` is defined only from 50 GHz and is not used.)
`eta = (er, -(sigma / ((TWO_PI * carrier_hz) * EPS0)))`.

For unit propagation direction `u` and unit surface normal `n`:
`cos_t = |u·n|`, `sin2 = 1.0 - cos_t*cos_t`,

```
s    = csqrt(csub(eta, (sin2, 0)))
G_TE = cdiv(csub((cos_t, 0), s), cadd((cos_t, 0), s))
G_TM = cdiv(csub(cscale(eta, cos_t), s), cadd(cscale(eta, cos_t), s))
```

A surface of thickness `t` is a slab (ITU-R P.2040 §3.4), with
`k0t = TWO_PI * t / LAMBDA`, `q = cscale(s, k0t)`, `q0 = k0t * cos_t`:

```
E2  = cexp(cmul((0, -2), q))
den = csub((1, 0), cmul(cmul(G, G), E2))
R   = cdiv(cmul(G, csub((1, 0), E2)), den)
T   = cdiv(cmul(csub((1, 0), cmul(G, G)), cexp(cmul((0, -1), csub(q, (q0, 0))))), den)
```

`T` includes `e^{+jq0}` because every path length `L` already counts the
slab's thickness as free space. The printed P.2040 form and Sionna RT omit
it, so `arg T` differs from them by exactly `q0` while `R` and `|T|` agree;
do not "fix" this. Walls use `G = G_TE`, floor and ceiling `G = G_TM`
(vertical antennas: the field is pure TE at a wall only for a horizontal
leg; a tilted leg's TM part is ignored, §16).

**Reflections are referenced to the incident face.** `R` is the slab's
reflection seen from the face the wave arrives at, so §6 places every
bounce on that face: a wall bounce on the face `t/2` from the centreline
on the bouncing wave's side (§6.1), the floor and ceiling at `z = 0` and
`z = H`. Transmission is unchanged: a leg crossing a wall is tested
against the centreline and multiplied by `T`, whose `e^{+jq0}` accounts
for the thickness counted as free space.

### 5.2 Objects

Objects are not slabs. A face reflects specularly when it is large enough
(§6.5) with the **single-interface** `G` of its material (`G_TE` for
vertical faces, `G_TM` for the top); `fabric` never reflects specularly.
Otherwise a face is a point scatterer. Object blockage is a per-crossing
loss. The `rho` and `L_db` values are SIM assumptions for Phase 10:

| Object material | ITU | scatter `rho` | blockage `L_db` per crossing |
|---|---|---|---|
| `metal` | metal | 1.0 | 20.0 |
| `wood` | wood | 0.3 | 4.0 |
| `glass` | glass | 0.2 | 2.0 |
| `fabric` | — | 0.1 | 3.0 |

## 6. Static paths

For each link `(tx, rx)`, nodes at `Tp`, `Rp`, the static paths are, **in
this order**: LoS; order-1 surface bounces (surface index ascending);
order-2 bounces (`i`, then `j`); object faces (objects in order, faces in
order); clutter (draw order); tail (§6.8). A path is a polyline
`[Tp, P_1, …, Rp]` with a kind, a complex coefficient and its legs; its
length `L` is the Neumaier sum of its legs' 3-D lengths (§7.1), in order.
Kind codes: 0 LoS, 1 order-1 bounce, 2 order-2 bounce, 3 object specular,
4 object point, 5 clutter, 6 tail.

### 6.1 Surfaces
`S` = walls in order, then (if `floor_ceiling`) floor, then ceiling. A
bounce is taken on a **face** of a surface, chosen by a side `f ∈ {+1, -1}`
(§6.3, §6.4). For a wall, with `dc(P) = (P.x-A.x)*n.x + (P.y-A.y)*n.y`
(signed distance from the centreline) and `o = f * (t / 2.0)`:

| | wall `w`, face `f` | floor | ceiling |
|---|---|---|---|
| `side(P)` | `sign(dc(P))` | `sign(P.z)` | `sign(H - P.z)` |
| `front_f(P)` | `sign(dc(P) - o) == f` | `side(P) == f` | `side(P) == f` |
| `image_f(P)` | `dd = 2.0 * (dc(P) - o)`; `(P.x - dd*n.x, P.y - dd*n.y, P.z)` | `(P.x, P.y, -P.z)` | `(P.x, P.y, 2*H - P.z)` |
| normal | `n` | `(0, 0, 1)` | `(0, 0, 1)` |
| `R(cos_t)` | slab `R` (TE, its material and `t`) | slab `R` (TM, `floor_material`, `slab_thickness_m`) | likewise for the ceiling |

`hit_f(P, Q)` — where segment `P → Q` meets the face, or none:

- **wall**: the face line is the centreline moved by `o` along `n`,
  `Af = (A.x + o*n.x, A.y + o*n.y)`; `r = (Q-P).xy`, `w = (B-A)` (the
  centreline's own direction), `den = r.x*w.y - r.y*w.x`; none if
  `den == 0`; `s = ((Af-P).x*w.y - (Af-P).y*w.x) / den`,
  `v = ((Af-P).x*r.y - (Af-P).y*r.x) / den`; it hits iff `0 < s < 1` and
  `0 <= v <= 1`, at `X = (P.x + s*r.x, P.y + s*r.y, P.z + s*(Q.z-P.z))`,
  and only if `0 <= X.z <= h`. (Near a corner a face line runs up to
  `t/2` into the adjacent wall; no trimming.)
- **floor**: `dz = Q.z - P.z`; none if `dz == 0`; `s = (0.0 - P.z) / dz`;
  hits iff `0 < s < 1`, `X = P + s*(Q-P)` with `X.z = 0.0`, and only if
  `inside(F, X.xy)`.
- **ceiling**: `s = (H - P.z) / dz`, `X.z = H`, otherwise as the floor.

`hit(P, Q)` without a face — used for transmission (§6.9) — is the wall
test above with `o = 0` (the centreline).

### 6.2 Line of sight
`[Tp, Rp]`, coefficient `(LAMBDA / (FOUR_PI * L), 0)`, kind 0.

### 6.3 Order 1 (if `reflection_order >= 1`)
For each surface `i`: `f = side_i(Tp)`; the path exists iff `f != 0`,
`front_f(Tp)`, `front_f(Rp)` and `X = hit_{i,f}(image_{i,f}(Tp), Rp)`
exists. Points `[Tp, X, Rp]`; `cos_t` from the incoming leg `Tp → X`;
coefficient `cscale(R_i(cos_t), LAMBDA / (FOUR_PI * L))`; kind 1.

### 6.4 Order 2 (if `reflection_order >= 2`)
For each ordered pair `(i, j)`, `i != j`, `i` outer: `fi = side_i(Tp)`,
`fj = side_j(Rp)`, both `!= 0`; `T1 = image_{i,fi}(Tp)`,
`T2 = image_{j,fj}(T1)`; `Xj = hit_{j,fj}(T2, Rp)`, then
`Xi = hit_{i,fi}(T1, Xj)`; both must exist, and `front_fi(Tp)`,
`front_fi(Xj)` (surface `i`), `front_fj(Xi)`, `front_fj(Rp)` (surface `j`).
Points `[Tp, Xi, Xj, Rp]`; `cos_t1` from
`Tp → Xi` with surface `i`, `cos_t2` from `Xi → Xj` with surface `j`;
coefficient `cscale(cmul(R_i(cos_t1), R_j(cos_t2)), LAMBDA / (FOUR_PI * L))`;
kind 2. (This construction is exact for planar mirrors, so heights along
the unfolded path need no separate formula.) A path that meets the edge
where two surfaces join is admitted or not by the last bit of `hit`
(bedroom link (0, 1) has two such order-2 paths, `[floor, wall 2]` and
`[ceiling, wall 2]`, whose bounce points coincide on the edge); the two
implementations agree on it because they evaluate `hit` exactly as
written, but the reverse link may decide it the other way (§15).

### 6.5 Object faces
`door` and `window` objects are openings drawn in a wall, not furniture:
they take no part in §6.5 or in §6.9 blockage (the wall they sit in is
modelled whole, §16). Every other object has five faces, in order `+x, -x, +y, -y, top` (object-local).
Face centre `P_f`: the local point `(±sl/2, 0, 0)`, `(0, ±sb/2, 0)` or
`(0, 0, sh/2)` taken to the world by the §3.1 formula,
`(cx + (lx*cs - ly*sn), cy + (lx*sn + ly*cs), cz + lz)`; outward normal
`n_f`: the local `(±1,0,0)`, `(0,±1,0)` or `(0,0,1)` rotated by the same
formula without the centre, `(lx*cs - ly*sn, lx*sn + ly*cs, lz)`;
in-plane extents `(e1, e2)`: `(sb, sh)`, `(sb, sh)`, `(sl, sh)`,
`(sl, sh)`, `(sl, sb)`; area `e1*e2`.

A face takes part only if **both** nodes are in front of it:
`(Tp - P_f)·n_f > 0` and `(Rp - P_f)·n_f > 0` (an object's back faces are
not lit through it). The test is exact: in the bedroom the table's −y face
centre is `y = 0.45 + (-0.3)` = `0.15000000000000002` and nodes 0 and 1 sit
at `y = 0.15`, in its plane but for that last bit, so `P_f` must be
computed exactly as written. Then:

1. **Specular** (if `object_specular` and the material is not `fabric`):
   `T' = Tp - 2*((Tp - P_f)·n_f)*n_f`; `den = (Rp - T')·n_f`; if `den != 0`,
   `s = ((P_f - T')·n_f) / den`, and if `0 < s < 1`,
   `X = T' + s*(Rp - T')`. `X` must lie on the face: with `X` in
   object-local `(lx, ly, lz)` (§3.1), `|ly| <= sb/2` and `|lz| <= sh/2` for
   the `±x` faces, `|lx| <= sl/2` and `|lz| <= sh/2` for `±y`,
   `|lx| <= sl/2` and `|ly| <= sb/2` for the top. With
   `d1 = |Tp - X|`, `d2 = |X - Rp|`, `rF = sqrt(LAMBDA*d1*d2 / (d1 + d2))`:
   if `e1 >= 2*rF` and `e2 >= 2*rF` the face is specular — points
   `[Tp, X, Rp]`, coefficient `cscale(G(cos_t), LAMBDA / (FOUR_PI * L))`
   with the single-interface `G` of §5.2 and `cos_t` from `Tp → X`, kind 3.
2. **Otherwise** (if `object_scatter`): a point scatterer at `P_f` (§6.6)
   with `sigma = object_scatter_scale * rho * area`, kind 4.

### 6.6 Point-scatter coefficient
For a point `P` with RCS `sigma`: `d1 = max(|Tp - P|, MIN_RANGE_M)`,
`d2 = max(|P - Rp|, MIN_RANGE_M)`,
`a = (LAMBDA * sqrt(sigma)) / ((FOUR_PI_1_5 * d1) * d2)`, coefficient
`(a, 0)`, points `[Tp, P, Rp]`. The clamp affects `a` only; `L` uses true
distances.

### 6.7 Clutter
`clutter_count` points, drawn once per run from stream 1 before anything
else uses it. For each point: up to 1000 tries of `x = range(x0, x1)` then
`y = range(y0, y1)`, accepting the first `(x, y)` with
`inside_with_margin(F, (x, y), clutter_margin_m)`; then
`z = range(0.0, H)`, `sigma = exp(range(ln(lo), ln(hi)))` (`exp` and `ln`
are libm's, not `Rng::exp`). A point with no
accepted position consumes its 2000 draws and no more, and is dropped.
Each kept point is a point scatterer (§6.6), kind 5, shared by every link.

### 6.8 Diffuse tail (if `tail`)
A delay-domain tail, so the channel has more than a handful of resolvable
paths. For the unordered pair `{a, b}` (`a < b`; `pair_index` its position in
the list of all node-id pairs `(a, b)`, `a < b`, sorted by `(a, b)` —
every node, whatever its role), from stream `100000 + pair_index`,
`tail_count` terms; for each: `x_n = -((C * tail_decay_s) * ln(1.0 - uniform()))`,
`phi_n = range(0.0, TWO_PI)`. With `L_ref` the length of the link's longest
kind-0/1/2/3 path and `P_spec` the Neumaier sum of `|A_p|^2` over those
paths (§6.9 coefficients, path order), every term has the same amplitude
`a_tail = sqrt((10^(tail_power_db/10) * P_spec) / float(tail_count))`:
coefficient `cscale(cexpj(phi_n), a_tail)`, length `L_n = L_ref + x_n`,
kind 6. The delays are exponential with mean `C * tail_decay_s` and the
weights equal, so the expected power-delay profile after `L_ref` is
`exp(-τ / tail_decay_s)` and its expected total is
`10^(tail_power_db/10) * P_spec`. Tail terms have no legs: no §6.9
losses and no shadowing. Both directions of a link share the pair's draws,
so the tail is reciprocal.

### 6.9 Static losses on legs
For every leg of every kind-0..5 path, in leg order, the coefficient is
multiplied by:

1. **Wall transmission** (if `transmission`): for each wall `k` in order,
   except the walls this leg touches as bounce points (order 1: surface
   `i` on both legs; order 2: `i` on leg 1, `i` and `j` on leg 2, `j` on
   leg 3), if the leg hits wall `k` (§6.1 `hit`, applied to the leg's own
   endpoints) multiply by `T_k(cos_t)` (`cmul`), `cos_t` from the leg.
2. **Object blockage** (if `object_blockage`): for each object in order
   that is not a door or window, except the object a face point or
   specular point of this leg belongs to, and except the object a
   stationary sitting or lying person is on (§8.2) for that person's body
   legs, if the leg meets the object's box (§7.4) multiply by
   `10^(-L_db/20)` (`cscale`).

The result is the path's **static coefficient** `A_p`.

## 7. Geometry primitives

- **7.1** `|P - Q| = sqrt((dx*dx + dy*dy) + dz*dz)`; xy length
  `sqrt(dx*dx + dy*dy)`; `dot2`, `dot3` left to right; unit direction
  `(dx/D, dy/D, dz/D)`.
- **7.2** `pdist(p, a, b)`: `d = b - a`; if `dot2(d, d) == 0` the distance
  to `a`; else `t = clamp(dot2(p - a, d) / dot2(d, d), 0, 1)` and the
  distance to `a + t*d`. `inside(F, p)`: `pdist <= 1e-9` to any edge, else
  crossing number over edges `(a, b)`: toggle when
  `(a.y > p.y) != (b.y > p.y)` and
  `p.x < a.x + (p.y - a.y) * (b.x - a.x) / (b.y - a.y)`.
  `inside_with_margin(F, p, m)`: `inside` and `pdist >= m` for every edge.
- **7.3** Segment–segment (2-D): they cross iff the §6.1 wall test with
  `0 <= s <= 1` and `0 <= v <= 1`; distance 0 if they cross, else the
  least of the four point–segment distances.
- **7.4** Leg against an object's box (3-D): both ends to object-local;
  for each axis with half-extent `h` (`sl/2`, `sb/2`, `sh/2`), position
  `p`, direction `d` (local end minus local start): if `d == 0`, reject
  when `|p| > h`; else `t1 = (-h - p) / d`, `t2 = (h - p) / d`, ordered,
  `t_enter = max(t_enter, min)`, `t_exit = min(t_exit, max)` starting from
  `[0, 1]`. They meet iff `t_exit > t_enter`. "Strictly inside" (node
  validation): `|lx| < sl/2` and `|ly| < sb/2` and `|lz| < sh/2`. The 2-D
  version drops the z axis.

## 8. People

### 8.1 Body parameters
Person `i`, stream `100 + i`, in order: `height = range(height_m)`,
`f_b = range(breathing_hz)`, `p2p = range(breathing_p2p_m)`,
`phi_b = range(0.0, TWO_PI)`, `phi_g = range(0.0, TWO_PI)`.

### 8.2 Mode and posture
A person with a scene start that is **not walkable** (§9) is
**stationary** — lying on a bed, sitting on a chair; otherwise **walk**.
`people_mode` overrides (a `walk` person whose start is not walkable is an
error naming `people_mode` and `people[i]`). Posture: a walking person
stands. A stationary person's posture is from the first object, in file
order, doors and windows included, whose box footprint (not grown, closed:
`|lx| <= sl/2` and `|ly| <= sb/2` with the start in object-local
coordinates, §3.1) contains the start: `bed` → `lie`, `chair`/`sofa` →
`sit`, otherwise (or with no such object) `stand`. That object is what a
stationary person is **on** (§6.9) and, lying, their bed (§8.3).
`people_posture` overrides: `stand` for a walking person changes nothing;
`sit` or `lie` for a walking person is an error; `lie` is an error unless
that first object is a `bed` (both name `people_posture` and
`people[i]`). A stationary person never moves, `s = 0`, and breathes.

### 8.3 Body points
Body frame: `+x` forward, `+y` left, `+z` up. The first `body_points`
rows are used, in order; `z` is the fraction times `height`; `π` is `PI`,
`–` is 0.0:

| # | Point | x | y | z/height | weight | swing amp | swing phase | breath |
|---|---|---|---|---|---|---|---|---|
| 1 | chest | 0.12 | 0 | 0.72 | 0.20 | 0 | – | 1.0 |
| 2 | head | 0 | 0 | 0.93 | 0.08 | 0 | – | 0 |
| 3 | pelvis | 0 | 0 | 0.53 | 0.15 | 0 | – | 0 |
| 4 | back | -0.12 | 0 | 0.70 | 0.15 | 0 | – | 0 |
| 5 | left hand | 0 | 0.22 | 0.45 | 0.04 | 0.25 | π | 0 |
| 6 | right hand | 0 | -0.22 | 0.45 | 0.04 | 0.25 | 0 | 0 |
| 7 | left foot | 0 | 0.10 | 0.05 | 0.04 | 0.30 | 0 | 0 |
| 8 | right foot | 0 | -0.10 | 0.05 | 0.04 | 0.30 | π | 0 |
| 9 | left shoulder | 0 | 0.20 | 0.80 | 0.05 | 0 | – | 0 |
| 10 | right shoulder | 0 | -0.20 | 0.80 | 0.05 | 0 | – | 0 |
| 11 | left knee | 0 | 0.10 | 0.28 | 0.04 | 0.15 | 0 | 0 |
| 12 | right knee | 0 | -0.10 | 0.28 | 0.04 | 0.15 | π | 0 |
| 13 | left elbow | 0 | 0.22 | 0.62 | 0.03 | 0.12 | π | 0 |
| 14 | right elbow | 0 | -0.22 | 0.62 | 0.03 | 0.12 | 0 | 0 |
| 15 | abdomen | 0.11 | 0 | 0.60 | 0.08 | 0 | – | 0.5 |
| 16 | neck | 0 | 0 | 0.85 | 0.03 | 0 | – | 0 |

Point RCS `body_rcs_m2 * w_j / W`, `W` the Neumaier sum of the selected
weights. At time `t`, with walk flag `s` and leg speed `v`:
`f_s = clamp(0.95 * sqrt(v / 1.1), 0.5, 1.2)` if `s = 1`, else 0;
`sw = (amp * s) * sin((TWO_PI * f_s) * t + phi_g + phase)`;
`br = (breath * (p2p / 2.0)) * sin((TWO_PI * f_b) * t + phi_b)`;
standing local point `(bx, by, bz) = (x + sw + br, y, zfrac * height)`.

- **stand**: world by §3.1 with the person's position and heading.
- **sit**: `bz` replaced by `0.72 * bz`, then as stand.
- **lie**, on bed object `o`: long axis `a = (cs, sn)` if `sl >= sb`, else
  `(-sn, cs)`; across `c = (-a.y, a.x)`; with `along = bz - 0.5 * height`,
  world `x = px + (along*a.x + by*c.x)`, `y = py + (along*a.y + by*c.y)`,
  `z = (o.cz + o.sh/2.0) + 0.12 + bx` (on the bed, chest motion vertical).

### 8.4 Body paths
Each body point is a point scatterer (§6.6) with its two legs, which get
§6.9 losses (recomputed every packet) and shadowing by every **other**
person (§8.5).

### 8.5 Shadowing (if `shadowing`)
A person is a vertical cylinder of radius `body_radius_m` at their
position `c`, from the floor to `top` = `height` (stand, walk), `0.72 * height`
(sit), or the bed top `+ 0.25` (lie). For a leg `P1 → P2` (3-D length
`D`), `r = (P2 - P1).xy`, `L2 = dot2(r, r)`:

1. If `L2 == 0`, no loss.
2. `u* = dot2(c - P1.xy, r) / L2` (not clamped). If `u* <= 0` or `u* >= 1`:
   no loss, unless the nearer end (`P1` if `u* <= 0`, `P2` if `u* >= 1`)
   is inside the cylinder (xy distance from `c` `< body_radius_m` and
   `z <= top`), in which case `J = shadow_cap_db`.
3. Otherwise `z* = P1.z + u* * (P2.z - P1.z)`, `d_perp` = xy distance from
   `c` to `P1.xy + u* * r`, `d1 = u* * D`, `d2 = (1.0 - u*) * D`,
   `k = sqrt(2.0 * (d1 + d2) / ((LAMBDA * d1) * d2))`,
   the near and far side edges `nu_1 = (body_radius_m - d_perp) * k` and
   `nu_2 = (body_radius_m + d_perp) * k`, the top `nu_t = (top - z*) * k`,
   and with the ITU-R P.526 edge `J1(nu)` = 0 if `nu <= -0.78`, else
   `w = nu - 0.1; 6.9 + 20.0 * log10((sqrt(w*w + 1.0) + nu) - 0.1)`, the
   P.526 finite-width-screen average over the three edges:
   `J = clamp(-10.0 * log10((10^(-J1(nu_1)/10) + 10^(-J1(nu_2)/10)) + 10^(-J1(nu_t)/10)), 0, shadow_cap_db)`.
   (`J1` is not 0 until `nu <= -0.78`, so a leg loses nothing only when
   it passes more than `0.78/k` outside the cylinder or above its top —
   about 0.19 m at the middle of a 4 m link; one grazing the side loses
   about 5 dB, and one through the axis below the head about 8.7 dB.)
4. The factor is `10^(-J/20)`.

A path's shadow factor is `S = 1.0`, then for each leg in order, for each
person in index order: `S = S * 10^(-J/20)`, a leg and person with no loss
contributing `1.0` (§1). A body path (§8.4) skips the person it belongs
to. Static paths are shadowed by every person; tail terms are not.

## 9. Motion

**Walkable point** `p`: `inside_with_margin(F, p, walk_margin_m)`, and
outside the footprint of every object that is not a door or window, grown
by `rg = body_radius_m + object_clearance_m` (`|lx| > sl/2 + rg` or
`|ly| > sb/2 + rg`). **Walkable leg** `a → b`: both ends walkable;
segment–segment distance to every footprint edge `>= walk_margin_m`; and
the segment misses (§7.4, 2-D) every grown rectangle of an object that is
not a door or window.

**Start** of person `i`:
- With a scene start: used as given; §8.2 decides the mode.
- Without one: they enter at the first `door` object in file order — its
  normal axis is its thinner horizontal extent (local x if `sl < sb`,
  else local y); taking `+` then `-` along that axis, the first direction
  whose point `0.5` m from the door centre is inside `F` is "into the
  room"; the start is the first walkable point at distances
  `0.05 * n`, `n = 1..40`, from the door centre in that direction.
- With no door, or no walkable point found: up to 1000 tries, from stream
  `200 + i`, of `x = range(x0 + m, x1 - m)` then `y = range(y0 + m, y1 - m)`
  (`m = walk_margin_m`), taking the first walkable; if none, the run is
  rejected naming `people[i]`.

**Trajectory** (walking people), a list of contiguous segments from
stream `200 + i`: the first segment's `t0` is `0.0` and its start is the
person's start; each new segment's `t0` is the previous `t1` (the same
double) and its start the previous end. Before evaluating time `t`, extend
while there is no segment or `last.t1 <= t`. Times are evaluated in
nondecreasing order (packet order, §10), so extension is shared by all
receivers of a transmission. To extend: up to 64 tries of `x`, `y` as above, accepting
the first walkable `(x, y)` whose distance `d` from the current position
is `>= 0.5` and whose leg is walkable; then `v = range(speed_mps)`,
`pause = range(pause_s)`; append a walk with `t1 = t0 + d / v`, then a
pause with `t1 = t0 + pause` only if that `t1 > t0`. If no try is
accepted: `pause = range(pause_s)` (one draw) and append a pause with
`t1 = t0 + max(pause, 0.5)`. Use the segment with `t0 <= t < t1`.

On a walk, position `from + (to - from) * ((t - t0) / (t1 - t0))`, `s = 1`,
`v` the walk's speed; on a pause, its position, `s = 0`. Heading:
`atan2(to.y - from.y, to.x - from.x)` of the walk containing `t`; during a
pause, of the last walk with `t1 <= t`; `0.0` if none. A stationary person
has `start_heading_rad` (default 0.0); a lying person's heading does not
move their body points (§8.3), but it orients the phone they carry (below)
and is reported in truth (§13).

**Phone** (if the scene has one): carried by person `phone_carrier` at
`phone_offset_m` in the body frame (§3.1) with the person's position and
heading (a stationary person's `start_heading_rad`), with `z` scaled by
0.72 when sitting; when lying, `x` and `y` are as when standing and
`z = (o.cz + o.sh/2.0) + 0.05` on bed `o`. With no such person
(`phone_carrier` ≥ the number of people) it stays at `phone.start`,
z = 1.0. Truth only; it does not radiate in Phase 3.

## 10. TDMA schedule and links

Transmitters `T[0..nT)` are the nodes with role `tx` or `txrx`, in id
order; receivers are `rx` or `txrx`. Links are the ordered pairs
`(tx, rx)`, `tx` a transmitter, `rx` a receiver, `tx != rx`, sorted by
`(tx, rx)`; `link_index` is the position. Superframe `m = 0, 1, …`,
slot `i`: `T[i]` transmits at `t = float(m * nT + i) * slot_s` if
`m % tx_period[T[i]] == 0`, received by every receiver except itself, in
id order. Output order: `m`, then `i`, then receiver id; lost packets
included with `lost = true`. Per-link rate
`1 / ((nT * slot_s) * tx_period[tx])`.

The default all-`txrx` mesh is the **planned** firmware
(CAMPUS_SENSING_PROMPT.md §5.1), but with today's datagram, which has no
transmitter id. Today's firmware is TX xor RX with no TDMA: one node
beacons every `1 / CONFIG_SAANS_TX_RATE_HZ` (20 Hz by default,
`tx_role.cpp`). Model it with `roles` one node `tx` and the rest `rx`,
and `slot_s = 0.05`, which gives 20 Hz per link, the gateway's default
`raw.sample_rate_hz`; with the default `slot_s` a lone transmitter runs
at 80 Hz, four times the firmware.

## 11. Channel of one packet

At the packet's `t`, people at their §9 positions, for each bin `k`
ascending:

```
H_k = Σ static kind 0..5 paths p:  cmul(cscale(A_p, S_p(t)), cexpj(theta(f_k, L_p)))
    + Σ tail terms n:              cmul(A_n, cexpj(theta(f_k, L_n)))
    + Σ people q, points j:        cmul(cscale(A_qj(t), S_qj(t)), cexpj(theta(f_k, L_qj(t))))
```

one Neumaier sum per component, in exactly that order. `S` is §8.5's shadow
factor. Before impairments the channel is reciprocal to rounding.

## 12. ESP32 impairments

### 12.0 Firmware configuration this mirrors
`csi_capture.cpp:106-118`: `lltf_en = htltf_en = true` → the 128-entry
buffer (§4); `stbc_htltf2_en = false` → no STBC; `ltf_merge_en = true` →
`ltf_merge`; `channel_filter_en = false` → `smoothing` off;
`manu_scale = false`, `shift = 0` → the per-packet AGC scale of §12.3.

### 12.1 Devices
Node `n`, stream `500 + n`, in order: for `m = 1..ripple_harmonics`:
`alpha_m = range(-ripple_amp, ripple_amp)`, `beta_m = range(0, TWO_PI)`,
`gamma_m = range(-ripple_phase_rad, ripple_phase_rad)`,
`delta_m = range(0, TWO_PI)`; then `g_dev_db = range(device_gain_db)`;
then `nf_dbm = round(range(noise_floor_dbm))`; then
`tau_dev = range(timing_offset_dev_s)`. Ripple at bin `k`, with
`x_m = (TWO_PI * float(m)) * float(k) / ripple_period_bins`:
`mag(k) = 1.0 + A.value()`, `A` the Neumaier sum over `m` (ascending) of
`alpha_m*cos(x_m + beta_m)` (the `1.0` is not a term of the sum);
`ph(k) = P.value()`, `P` the Neumaier sum over `m` of
`gamma_m*sin(x_m + delta_m)`;
`Rdev(k) = cscale(cexpj(ph(k)), mag(k))`; `(1, 0)` with `ripple` off.

### 12.2 Per packet
From stream `1000 + link_index`, for **every** transmitted packet, lost or
not, in order: `u_loss = uniform()`; `u_jump = uniform()`;
`u_dir = uniform()`; `phi0 = range(-PI, PI)`;
`tau = range(-timing_offset_s, timing_offset_s)`;
`rj = range(-rssi_jitter_db, rssi_jitter_db)`; then 256 `gauss()`: for each
of the 128 entries in buffer order (LLTF then HT-LTF, FFT order),
`n_re` then `n_im`. Then the link's AGC state (starting at
`agc_initial_db`): if `u_jump < agc_jump_prob`,
`agc = clamp(agc + (u_dir < agc_up_prob ? agc_step_db : -agc_step_db), -agc_max_db, agc_max_db)`.
The packet is lost iff `u_loss < p`, `p` = `loss_prob_link["tx,rx"]` or
`loss_prob`. A received packet uses the updated `agc`.

### 12.3 One received packet
`Rr = Rdev(rx)`, `Rt = Rdev(tx)`:

1. `X_k = cmul(cmul(H_k, Rr(k)), Rt(k))`, every bin.
2. `P_H` = Neumaier over the 56 HT-LTF bins (ascending k) of
   `re*re + im*im` of `H_k`, `value() / 56.0`;
   `rssi_true = tx_power_dbm + 10.0 * log10(P_H)` (`P_H == 0` →
   `rssi_true = rssi_range_dbm.lo`);
   `rssi = clamp(round(rssi_true + rj), lo, hi)` of `rssi_range_dbm`.
3. `P_X` likewise from `X_k`; `c = target_counts / sqrt(P_X)` (`P_X == 0`
   → `c = 0`); `g = 10^((g_dev_db(rx) + agc) / 20.0)`.
4. `Y_k = cmul(cscale(X_k, c * g), cexpj(phi0 - ((TWO_PI * float(k)) * spacing_hz) * (tau + tau_dev(rx))))`,
   with `phi0` the drawn value when `cpe` is true and `0.0` when it is
   false (the draw always happens).
5. `a = (target_counts * g) * 10^(-(rssi_true - nf_dbm(rx)) / 20.0)`;
   `sigma = sqrt(chip_noise_counts*chip_noise_counts + (snr_noise ? (a*a) / 2.0 : 0.0))`
   (thermal noise passes through the same gain as the signal).
6. LLTF entry at an LLTF bin: `cadd(Y_k, cscale((n_re, n_im), sigma))`;
   HT-LTF likewise with its own draws; inactive entries `(0, 0)`.
7. `ltf_merge`: an HT-LTF entry whose bin is an LLTF bin becomes
   `cadd(cscale(LLTF, w), cscale(HTLTF, 1.0 - w))`, `w = ltf_merge_weight`.
8. `smoothing` (off by default): within each LTF, over its active bins in
   ascending `k` (`-1` and `+1` neighbours; 52 or 56 values), on the values
   after step 7, re and im separately, from the unsmoothed values:
   `y_i = (k0*x_{i-1} + k1*x_i) + k2*x_{i+1}`; ends
   `y_0 = e0*x_0 + e1*x_1`, `y_{n-1} = e0*x_{n-1} + e1*x_{n-2}`.
9. `quantize`: each component `clamp(round(v), -128, 127)` as int8.

### 12.4 The datagram
If `quantize`: the first `fw_subcarriers` of the 128 entries in buffer
order, each `(imag, real)`, in a datagram exactly as
`csi_capture.cpp::serialize_raw` writes it — 20-byte little-endian header:
u32 magic `0xC5110001`; u8 node id (rx); u8 `n_antennas = 1`; u16
`n_subcarriers = fw_subcarriers`; u32 `freq_mhz = 2412 + 5*(channel-1)`;
u32 sequence; i8 rssi; i8 `nf_dbm(rx)`; u8 0; u8 0; then
`fw_subcarriers` pairs of (i8 imag, i8 real).
`wifisense.sources.esp32_udp.parse_esp32_frame` and the gateway's
`raw.rs` must parse every datagram unchanged.

**Sequence numbers** are per receiving node: 0 first, `(seq + 1) mod 2^32`
for every packet that node receives from any transmitter — the firmware's
CSI-callback counter. Consequences: a lost packet leaves no gap, so
`seq`-based loss and gap filling see none of `loss_prob` (use the record's
`lost`); per link, `seq` strides by about the number of transmitters, so a
`SequenceGapFiller` must never run on a per-link split. A receiver's
datagram stream interleaves every transmitter it hears: about `nT - 1`
times the per-link rate for a `txrx` receiver (it never hears its own
slot), `nT` times for an `rx` one (less with `tx_period`). The datagram
carries no transmitter id (the gateway keys raw CSI by rx alone), so no
consumer can split such a stream per link. A run with **one** transmitter
is gateway-faithful, provided the gateway's `raw.sample_rate_hz` equals
its per-link rate (the gateway's default 20 Hz needs `slot_s = 0.05`,
§10). The exports record this (§13); the CLI warns about interleaved
transmitters and notes a lone transmitter's rate when it is not 20 Hz.

### 12.5 With `impair = false`
All §12.1 and §12.2 draws still happen, the AGC state still steps (it is
reported as `agc_db` but not applied: `g = 1`), and `seq` is assigned as
above. No packet is lost; `X_k = H_k`; `g = 1`; `phi0 = tau = tau_dev = 0`;
`sigma = 0`; no merge, no smoothing; `quantize` and datagrams follow their
own switch; `rssi = clamp(round(rssi_true), lo, hi)`; `nf_dbm` is still
reported.

## 13. Output

**Per packet**: `t`, `tx`, `rx`, `link_index`, `lost`, `agc_db` (the
link's AGC state after this packet's step); when received only (a lost
packet has none of these, `H` included):
`seq`, `rssi`, `rssi_true`, `noise_floor`, `H` (64 bins ascending `k`),
`buffer` (128 entries, int8 or float) with `buffer_pre` (the float values
before step 9), and the datagram. Also, once per run: `buffer_k` (128
ints: `0..31, -32..-1, 0..31, -32..-1`) and `buffer_active` (128 bools).
**Consumers mask by index, never by value**: simulated nulls are exactly
0, hardware nulls need not be.

**Truth**: one record per transmission `(m, i)` at its `t`, lost or not:
per person in index order `(x, y, heading, walking, posture)`, `walking`
being the walk flag `s` at `t` (false in a pause and for a stationary
person); the phone `(x, y, z)` or none.

**Files**: `saans roomsim-export SCENE.json --out PREFIX` and the
reference's `wifisense.roomsim.export_run(sim, PREFIX)` write the same
three files, streamed in §10 order; their pcaps are byte-identical and
their JSON files are equal as parsed values (a number may be spelled
differently, `1e-05` against `1e-5`). Both refuse to start when any of
the three exists, unless told to overwrite (`--overwrite`,
`overwrite=True`), and then first delete the old manifest; the manifest is
written last, so a run that fails part-way never leaves a manifest that
describes other files. Both also refuse `quantize` false, which emits no
datagram, naming `quantize` in the error.

- `PREFIX.sim.pcap`: the datagram of every received packet (a lost packet
  has none), written exactly as `wifisense.sources.recorder.PcapWriter`
  writes (ns pcap, LINKTYPE_RAW, IPv4+UDP with the IPv4 identification
  counting from 0 per file, don't-fragment set and UDP checksum 0; source
  `10.1.<rx>.1:5566`, destination `10.0.0.1:5566` — addresses and source
  port are a convention, never the gateway's own or a network address (a
  real node sends from an ephemeral port); timestamp `1_790_000_000 s`
  plus `round(t * 1e9)` ns, round half to even). It holds the raw-CSI
  datagrams only, not the FeatureFrames a raw-mode node also sends (§16).
- `PREFIX.truth.jsonl`: one JSON object per transmission `(m, i)`, lost or
  not: `{"label": "SIM", "m", "slot", "t", "tx", "people": [{"name":
  "people[i]", "x", "y", "heading", "walking", "posture"}], "phone":
  [x, y, z] or null, "packets": [{"rx", "link_index", "lost", "seq",
  "rssi", "rssi_true"}]}`, the last three `null` for a lost packet. Truth
  goes only here, never into the pcap.
- `PREFIX.manifest.json`: `{"label": "SIM", "spec": "roomsim v1",
  "scene_name", "scene_sha256" (of the scene file's bytes), "provenance",
  "config" (every §3.2 value, defaults included), "per_link_rate_hz"
  (`"tx,rx"` → Hz, link order), "links" (per link, in order:
  `link_index`, `tx`, `rx`, `per_link_rate_hz`, `packets`, `lost`,
  `datagrams`), "counts" (`transmissions`, `packets`, `lost`,
  `datagrams`), "receiver_rate_hz" (`"rx"` → the mean rate of transmissions
  that receiver hears, `Σ 1/((nT*slot_s)*tx_period[tx])` over its links,
  receivers ascending), "gateway_faithful" (`nT == 1`, §12.4: the
  gateway's `raw.sample_rate_hz` must then equal `per_link_rate_hz`)}`, plus,
  only when the walls are implied (§3.1), `"assumptions"`: a list of
  strings starting `"SIM: "` that say so (numbers in them spelled as
  Python's `repr` of a float, `0.1`, `1.0`).

## 14. Agreement between Python and Rust

A value `a` matches the reference `b` iff
`|a - b| <= 1e-9*|b| + 1e-12*M`, `M` the largest `|b_j|` of the array `b`
belongs to (one packet's 64 bins or 128 buffer entries, one link's
coefficients, lengths or points, one trajectory's coordinates; for a
complex array, the largest magnitude). RNG output is bit-identical. Lost flags,
sequence numbers, AGC states, kinds and path counts are identical. RSSI
and int8 values are identical unless the pre-rounding value (recorded) is
within 1e-6 of a .5 tie in either implementation. Datagrams are identical
bytes under the same rule.

Golden sets, from `spec/golden/generate.py` on **CPython 3.14**, encoded as
the existing sets are (`_re`/`_im` arrays, packet-major, bins ascending
`k`, datagram bytes as f64 0..255 concatenated with a per-packet length
array):

| Set | Content |
|---|---|
| `roomsim_materials` | every material × thickness {0.1, 0.23} × {0°, 30°, 60°, 85°} × {TE, TM}: `G`, `R`, `T`; object single-interface `G` |
| `roomsim_shadow` | `J1(ν)` for ν = −1.0..3.0 step 0.25; three-edge `J` on a small grid |
| `roomsim_paths` | bedroom, links (0,1) and (0,3): every static path's kind, surface ids, points, `L`, coefficient |
| `roomsim_paths_hall` | hall-l-shaped, links (1,6) and (0,7) |
| `roomsim_paths_specular` | `spec/scenes/fixtures/roomsim_specular.json` (implied walls), links (0,3), (3,0) and (1,2): object-specular faces (kind 3), which no other set has — metal at yaw 0 and 80° and wood at −5° (TE), a glass table top at 10° (TM) |
| `roomsim_motion` | hall-l-shaped, `people[0]` walking, `t_n = float(n) / 10.0`, n = 0..299 |
| `roomsim_channel` | bedroom, seed 0, `people[0]` stationary (lying on the bed), 0.5 s, `impair` false (so no packet is lost and every packet has `H`): every packet's `H`, rssi, truth and body points |
| `roomsim_impair` | the same packets impaired (`smoothing` off): `buffer_pre`, `buffer`, rssi, `rssi_true`, seq, lost, datagrams |
| `roomsim_multi` | `spec/scenes/fixtures/roomsim_multi.json` (polygon footprint, implied walls, a 30°-yawed metal object, 3 walking people, one entering at the door), `tx_period {1: 2}`, 2 s, `impair` false: `H` for links (0,1), (1,2), (2,0); truth; those links' static paths |
| `roomsim_switches` | hall-l-shaped with its seated `people[1]` carrying the phone, impaired, the non-default impairment, scheduling and output branches: `smoothing`, `cpe`/`ripple`/`ltf_merge`/`snr_noise` off, `timing_offset_dev_s`, `roles` with a `tx`-only and an `rx`-only node, `tx_period`, `loss_prob_link`, `fw_subcarriers` 64, `channel` 11, `body_points` 16, a `start_heading_rad`: `buffer_pre`, `buffer`, rssi, seq, lost, AGC, datagrams, truth, body points; plus the same configuration for 0.05 s with `quantize` false: its packets and `buffer_pre`, `buffer` (the floats, equal to `buffer_pre`), and no datagram (its truth is the main run's first 0.05 s, not stored again) |
| `roomsim_physics_switches` | the non-default geometry, object, tail, shadowing and people branches, `impair` false: `reflection_order` 1 and 0, `floor_ceiling`, `transmission`, `object_specular`, `object_scatter`, `object_blockage` off, `object_scatter_scale`, `clutter_count`, `tail` and `shadowing` off, `n_people` adding a person at the door, `people_mode` and `people_posture` overrides: static paths, `H` and truth (layout in the generator's docstring) |
| `roomsim_export` | `export_run` on the bedroom, 0.25 s, `loss_prob` 0.25: the pcap's bytes and the truth file's values, which `saans-cli/tests/roomsim_export.rs` compares with `saans roomsim-export` |

The array layouts are the docstrings of the `gen_roomsim_*` functions.
Every set built from a scene (all but `roomsim_materials` and
`roomsim_shadow`) records `scene_sha256`, the digest of the scene file's
bytes, and every parity test that loads the set
(`saans-roomsim/tests/golden.rs`, `saans-cli/tests/roomsim_export.rs`,
`tests/test_roomsim_golden.py`) checks it against the file; the gateway's
`roomsim_datagrams.rs` parses only the datagrams. `tests/test_roomsim_golden.py` re-runs every
generator and compares every array it writes with the committed one, and
`saans-roomsim/tests/golden.rs` checks Rust: each Rust test loads a set by
name (the `saans-dsp/tests/golden.rs` pattern), rebuilds the inputs from
the set's scalars (`cfg.*`: every configuration value that differs from
§3.2) and compares by the rule above, printing its worst relative error.
On glibc both implementations agree bit for bit on every set; the
tolerance is for other libms. CI also checks `saans-roomsim` **by itself**
(`cargo check -p saans-roomsim` and `cargo test -p saans-roomsim`): in a
workspace build cargo unifies features, so only the standalone build
proves the crate compiles without `saans-scene`'s `write`.

## 15. Checks against independent ground truth

- At normal incidence `G_TE = (1 - sqrt(eta)) / (1 + sqrt(eta))` and
  `G_TM = -G_TE`.
- A lossless slab (`sigma = 0`) conserves energy: `|R|^2 + |T|^2 = 1`.
- A lossless half-wave slab (`t = LAMBDA / (2*sqrt(er))`, normal) has `R = 0`.
- `eta = (1, 0)`: `|R| < 1e-12` and `|T - 1| < 1e-12` at every angle up to
  89°. (Within about 0.1° of grazing `sin2 = 1.0 - cos_t*cos_t` cancels
  and the error passes 1e-12: 1.9e-12 at 89.99°, 1.9e-11 at 89.999°.)
- `J1(0) = 6.9 + 20*log10(sqrt(1.01) - 0.1) ≈ 6.03` dB.
- First-order wall images in a rectangle have closed-form lengths; every
  admitted bounce obeys the specular law.
- A person on a link's LoS lowers the LoS by exactly `10^(-J/20)`; a
  path whose legs all pass more than `0.78/k` (§8.5) outside their
  cylinder in plan, or that far above its top, is unchanged; nearer legs,
  including legs through the cylinder such as a floor bounce at their
  feet, are shadowed.
- A link through a concrete wall is weaker than through drywall.
- A metal box on the LoS weakens the link (no face radiates through it).
- Waypoints and legs avoid furniture; speeds and pauses are in range.
- With impairments off, `|buffer|` is proportional to `|H|` per bin.
- Every datagram parses with the Python parser and the gateway's `raw.rs`
  (`saans-gateway/tests/roomsim_datagrams.rs` parses every
  `roomsim_impair` datagram).
- `H(a→b)` and `H(b→a)` agree before impairments, except where a path meets
  the edge between two surfaces exactly or a leg grazes a box edge: such a
  path is admitted or blocked by the last bit, possibly differently in the
  two directions (§6.4). The shipped scenes' round coordinates make
  several such links (measured with the face-referenced bounces of §6.1):
  bedroom (0, 1); hall (1, 4), (1, 5), (3, 4), (3, 6), (4, 5), whose legs
  pass exactly through the re-entrant corner or graze the sofa's top
  edge, and (4, 6), whose order-2 path bounces exactly in the inner
  corner of walls 4 and 5 (both nodes lie on the line through it); office
  (2, 4), (3, 4); the multi fixture's (0, 2) — up to about 14% of
  `max |H|` (hall (4, 6)), so the check moves the nodes a few centimetres.
- `python3 -m wifisense stats run.sim.pcap` parses an export. It is a
  framing check: it prints a SIM banner for a `.sim.pcap`, or for a file
  whose `PREFIX.manifest.json` says SIM and whose `counts.datagrams` equals
  the file's record count (so a real capture beside an export is not
  mislabelled); its `seq`-based loss column cannot see the simulated loss
  (§12.4; the manifest's `counts.lost` has it), and the banner says so,
  and says when the stream interleaves several transmitters.
  `saans-cli/tests/roomsim_export.rs` checks an export's framing, headers,
  manifest and truth against a library run and against `roomsim_export`.

## 16. What this simulator does not model

- Walls are vertical slabs: a bounce is specular on the incident face
  with the plane-wave slab coefficient (internal multiple reflections
  folded into `R`), and a crossing is tested on the centreline. Floor and
  ceiling are flat. Surfaces reflect to second order; edges do not
  diffract; door and window openings are not cut out of walls. Face lines
  are not trimmed at corners, and a wall split into collinear pieces
  counts a bounce or crossing exactly at the joint twice (do not split
  walls along one line).
- Wall coefficients use TE, exact only for horizontal legs: typically
  < 0.5 dB error, up to about 7 dB on order-2 wall-and-floor/ceiling paths
  in the shipped scenes, more near the TM Brewster angle (about 63–66° for
  brick and concrete).
- Objects are boxes: large faces reflect once with a single-interface
  coefficient, small faces are point scatterers with made-up constants
  (§5.2), and crossing an object costs a fixed loss. "Large" uses the
  transverse Fresnel radius and ignores its `1/cos θ` stretch on the
  face, so a near-grazing face is treated as an infinite plane (about
  3 dB optimistic for the specular fixture's glass top).
- No scattering between objects, people and surfaces beyond what is
  listed; a body point is not mirrored in walls; the diffuse tail is a
  statistical stand-in for everything else.
- A body is 6–16 points and a cylinder with three knife edges (two sides
  and the top); it does not shadow its own points, so a person on the LoS
  partly refills their own shadow (the link drops less than `J` ≈ 8.7 dB
  on a 4 m link).
  Sitting and lying are crude reshapes. Gait is a sinusoid whose phase
  jumps when the speed changes. People do not avoid each other.
- Coefficients are evaluated at the carrier and applied across the
  17.5 MHz band. Antennas are isotropic and vertically polarised.
- RSSI is perfectly calibrated: no per-board RSSI or antenna-gain offset
  (on real ESP32 boards, several dB), so reciprocal links agree and only
  `rssi_jitter_db` varies it. `device_gain_db` scales the CSI counts only.
- The ESP-IDF buffer layout and the ESP-NOW PHY are assumptions (§4);
  `ltf_merge`, the AGC (a channel-independent random walk), noise and
  ripple are behavioural models with guessed magnitudes for Phase 10 to
  calibrate against a static-room capture.
- Real nodes also count ambient frames (beacons, broadcasts) in their
  sequence counter, lose frames to queue overflow after numbering them,
  and reboot; none of that is modelled.
- An export holds a raw-mode node's raw-CSI datagrams only, not the
  FeatureFrames that node also sends, so a gateway replay of it runs only
  the gateway-side DSP (nothing native to shadow it).
- A replayed export carries no SIM marker inside the pcap (the wire format
  has no room for one); its name, `.sim.pcap`, and its manifest are the
  label. `wifisense stats`/`replay` print a SIM banner for it; the
  gateway's replay labels it only once Phase 5's `Mode::Sim` exists.

None of this is a measurement. Every number from it is SIM.
