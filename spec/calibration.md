# Calibration of Room Sim against recordings (Phase 10b)

**Status:** spec for Phase 10b, 2026-10-25. Written under the owner's
blanket permission of that day and flagged in ROOMSIM.md. It answers the
plan's "differentiable calibration of 5–10 global parameters with
Wasserstein".

**Labels:**
- The parameters it fits are Room Sim's own SIM assumptions
  (spec/roomsim.md §3.2), so their values stay **SIM**.
- A distance between a SIM distribution and a REAL one is **SIM->REAL**.

**What it does not change:** no physics and no default. Room Sim, its
Python reference and the goldens are untouched. A calibrated parameter set
is a configuration override, nothing more. Adopting one as Room Sim's
default would be a physics-default change; it would need the owner, and
Rust, Python and the goldens together.

## 1. Why a population calibration

The only REAL data are OpenCSI's three HT20 sessions (spec/phase9.md §1).
They declare neither node nor subject positions, so nothing can be
calibrated link by link against geometry: there is no geometry to
ray-trace.

What can be matched are **distributions over links and windows**:
- noise level;
- frequency selectivity;
- the size of a person's effect;
- RSSI level and spread.

On the SIM side, these come from random dataset rooms with roughly
OpenCSI's node counts (6–10 nodes).

## 2. Features (`ml/calibrate/features.py`)

Every feature is per used link per 2 s window. They are computed through
the shared preprocessing with `bin_hz = None` (a packet-level spread,
independent of how packets fall into bins, which differs between SIM's
TDMA and OpenCSI's bursts) and `ref_bins = "shared40"`.

| Group | Value | States |
|---|---|---|
| `log_sigma` | `log10` mean over the 40 reference bins of `sigma`, the window's per-subcarrier spread of the gain-normalised amplitude | empty, still, moving |
| `fs_db` | std over the 40 bins of `20 log10 mu`: frequency selectivity | empty |
| `rssi_std` | std of the window's RSSI values (dBm) | empty, moving |
| `rssi_mean` | mean RSSI (dBm) | empty |

**REAL:** OpenCSI windows every 2 s wholly inside each segment, through
Phase 9's adapter. `still` is the "seated at desk" segment and `moving`
the walking one.

**SIM:** 30 s runs per room and state:
- `empty`: `n_people = 0`;
- `still`: one person, `people_mode {0: stationary}`;
- `moving`: one person, `people_mode {0: walk}`.

Windows end every 2 s from 4 s. Runs use the dataset's stored configs, so
seeds and the clutter, noise and device draws are common random numbers
across parameter vectors.

## 3. Parameters (`ml/calibrate/simulate.py`)

Nine global parameters, each an existing config field:

| Parameter | Field | Default | Range | Scale |
|---|---|---|---|---|
| `tx_power_dbm` | `tx_power_dbm` | 10 | −5..20 | linear |
| `noise_shift_db` | added to both ends of `noise_floor_dbm` | 0 | −10..20 | linear |
| `chip_noise_counts` | `chip_noise_counts` | 0.5 | 0.05..4 | log |
| `tail_power_db` | `tail_power_db` | −15 | −30..−3 | linear |
| `clutter_scale` | multiplies `clutter_rcs_m2` | 1 | 0.1..10 | log |
| `body_rcs_m2` | `body_rcs_m2` | 0.5 | 0.05..10 | log |
| `breathing_scale` | multiplies `breathing_p2p_m` | 1 | 0.5..4 | log |
| `rssi_jitter_db` | `rssi_jitter_db` | 1 | 0.2..4 | log |
| `ripple_amp` | `ripple_amp` | 0.05 | 0..0.3 | linear |

Each parameter's place in the feature table:
- **Noise level:** `tx_power_dbm`, `noise_shift_db` and `chip_noise_counts`
  (the SNR, through §12.3 step 5).
- **Frequency selectivity:** `tail_power_db`, `clutter_scale` and
  `ripple_amp`.
- **The person's effect:** `body_rcs_m2` and `breathing_scale`.
- **RSSI spread:** `rssi_jitter_db`.

## 4. Method (`ml/calibrate/calibrate.py`)

1. **Design.** A Latin hypercube of 128 points in the unit cube, plus the
   defaults. Each point is run on the same 24 fit rooms (from the
   dataset's train split) in the three states. The result is each
   group's quantiles at `J = 20` levels `τ_j = (j + 0.5) / 20`.
2. **Emulator.** A torch MLP (9 → 256 → 256 → 140) from the unit vector to
   the quantiles, trained on standardised outputs. Its accuracy is
   reported on 16 held-out design points.
3. **Fit**, per Phase 9 fold, against the fold's **training sessions
   only**, so the fold's test room calibrates nothing:
   - objective: `Σ_g mean_j |q̂_g(τ_j; u) − q_g^REAL(τ_j)| / IQR_g^REAL`,
     the 1-D Wasserstein-1 distance from quantiles with each group scaled
     by its REAL interquartile range;
   - minimised by Adam on `u = sigmoid(z)` from 256 starts;
   - the spread of the ten best starts is reported, as an
     identifiability check;
   - any parameter within 2% of a bound is listed.
4. **Check**, by simulation and not by the emulator. The defaults and each
   fold's fit run on 24 *other* rooms. For both, the exact
   `scipy.stats.wasserstein_distance` per group is reported against:
   - the fold's training sessions;
   - its unseen **test room**, which shows whether the calibration
     transfers to a room it never saw.

## 5. Outputs

- `ml/results/phase10/calibration.json`: the parameters per fold, the
  emulator's accuracy, every distance before and after, the REAL and SIM
  quantiles. Every number is labelled.
- `runs/phase10/calibration/`: the design and the emulator, outside git.

## 6. Limits

- **Population matching.** A calibrated parameter set makes SIM's
  distributions look like these three rooms' distributions, not like any
  room in particular.
- **Mechanisms the model lacks.** A parameter driven to a bound, or far
  from physical (a human RCS of several m²), means the model lacks a
  mechanism rather than that the value is true. The report says so when
  it happens.
- **Thin "still" data.** "Still" in OpenCSI is a seated person at a desk,
  with small movements. SIM's still person only breathes, so
  `breathing_scale` can absorb more than breathing.
- **One subject, three rooms, one hardware family (ESP32-S3/C3).**
