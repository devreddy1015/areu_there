# Phase 9 — public ESP32 data: zero-shot, fine-tune, real-only

**Status:** spec for Phase 9, 2026-10-25, written before any REAL figure
was computed. Decided under the owner's blanket permission of 2026-10-25
and flagged in ROOMSIM.md. Runs offline under ADR-009. Its figures are
**REAL** (methods trained and tested on recordings), or **SIM->REAL** (a
model trained on SIM, with or without REAL fine-tuning, tested on
recordings). The SIM pre-training figures stay **SIM**. Nothing here is a
product claim (ADR-009 d3): one subject, three rooms and a scripted
protocol are far from a deployment.

## 1. Data

**OpenCSI LCN 2026** (CC BY 4.0, so ADR-009 d6 allows it; cite *OpenCSI:
Self-Calibrating WiFi CSI Meshes Across 802.11 Generations*, IEEE LCN
2026). It has four sessions in three rooms. Each session uses eight ESP32
nodes on the room's perimeter (56 directed links) and the same anonymous
subject. Each runs 12 minutes:

| Phase | Duration | Annotation note |
|---|---|---|
| Empty | 300 s | — |
| Occupied, static | 180 s | "seated at desk" |
| Occupied, moving | 120 s | "walking naturally through room" |
| Empty again | 120 s | "leave room — breaks temporal confound" |

About 10 s between phases is left unlabelled.

| Session | Room (L x W, m) | Chip, PHY | Records kept (of read) |
|---|---|---|---|
| envA_S3 | A (6.06 x 3.28) | ESP32-S3, HT20 | 879,354 (1 cut-off line, 1,842 exact duplicates) |
| envB_S3prime | B (7.02 x 4.44) | ESP32-S3 (another batch), HT20 | 958,586 (1 cut-off line) |
| envC_C3 | C (4.43 x 2.91) | ESP32-C3, HT20 | 852,931 |
| envA_C6 | A, re-instrumented | ESP32-C6, HE20 | 311,115 (1 cut-off line, 149 HT20 records) |

**What it gives:** presence, a count of 0 or 1, and still vs moving; room
length and width. **What it does not give:** node positions, the subject's
position, and so zones and xy. Phase 9 therefore scores presence and count
(at most one person) only; the zone and xy heads are not evaluable on it.

**LoFi and WiCount** carry no licence. Under ADR-009 d6 they are not used
until their authors agree, so no loader is written for them. This is
flagged for the owner (ROOMSIM.md open items).

## 2. Adapter (`ml/opencsi.py`)

1. **Records.** Packet time is the backend's `server_ts`, parsed exactly
   from its decimal text to integer ns after `t0_s`, the first record's
   whole second. Labels come from the annotations on the same clock.
   The adapter skips and counts three kinds of record:
   - malformed lines;
   - exact duplicates (same link, firmware timestamp, CSI and RSSI);
   - records of another layout (refused if they exceed 1%).

   Records are put in time order per link. Node ids are the ranks of the
   MAC addresses; MACs never appear in an output.
2. **Layouts** (spec/preprocess.md §8, Phase 9 rows):
   - *HT20 sessions:* the data are the first 52 entries of the ESP32's
     64-entry LLTF buffer in FFT order (`k = 0..31, -32..-13`). Entry 0
     and entries 27..37 are zero in 99.8% of records, so they carry
     `k = -26..-13, 1..26` (40 bins).
   - *C6 session:* 256 HE-LTF tones in centred order (`t = -128..127`;
     its zero pattern fits no other order). Tone `t = 4k` sits at LLTF
     subcarrier `k`'s frequency, which gives all 52 bins. The layout is
     inferred from data, so the C6 results are exploratory.
   - *Pair order:* undocumented. ESP-IDF's `(imag, real)` is assumed,
     which affects phase only; Phase 9 uses amplitude only.
3. **Labels.** Windows lie wholly inside one labelled segment, never
   across a transition. Labels per segment:

   | Segment | presence | count | moving |
   |---|---|---|---|
   | `empty_room` | 0 | 0 | 0 |
   | `occupied_static` | 1 | 1 | 0 |
   | `occupied_moving` | 1 | 1 | 1 |

   `geometry_known` is false for every window (positions unknown). The
   room's L x W is kept as data, but no input uses it: with unknown
   geometry, the model replaces the room token by its unknown token.
4. **Rate cap** (spec/model.md decision 2). The links average about
   22 Hz, but packets come in bursts of about ten within a few ms every
   ~0.25 s (firmware timestamps). A uniform 20 Hz decimation would
   therefore keep one packet per burst. Instead the adapter keeps at most
   4 packets per link per 200 ms cell, on a grid anchored at `t0_s` and
   evenly spaced through the cell's packets. That matches SIM's densest
   bin (4 nodes, 20 Hz per link).

   After the cap, 33–76% of each session's records remain. Once binned at
   5 Hz, 37–71% of a link's bins hold a packet (SIM: almost all), at 3.2–3.8
   packets each. This sparsity is a known SIM/REAL difference that
   fine-tuning has to absorb.

## 3. Preprocessing and windows

- `PreprocessConfig(phase=False, bin_hz=5, ref_bins="shared40",
  min_packets=3)`. `shared40` (`k = -26..-13, 1..26`) is the set of bins
  every Phase 9 source carries (spec/preprocess.md §3).
- **Window length.** `ℓ` is Phase 8's chosen one, unless that is 1 s; then
  Phase 9 uses 2 s with Phase 8's other choices. The reason is coverage,
  measured before any label was read. At 1 s the bursts leave most links
  with fewer than `min_packets` filled bins:

  | ℓ | Links used |
  |---|---|
  | 1 s | 12–81% |
  | 2 s | 86–97% |
  | 4 s | 96–99.5% |
- **Training windows:** every `t_end = start + ℓ + j · 0.5 s` inside a
  segment of a training session.
- **Evaluation windows:** the same with a 1 s step. Two exclusions apply:
  - the session's first 90 s, which the baselines spend calibrating
    (§5);
  - any window ending before `start + ℓ`.

  Every method is scored on exactly these windows.

## 4. Models and protocol

Splits are **leave-one-room-out** over the three rooms, and a test room's
windows never train, select or tune anything:

| Fold | Test sessions | Training sessions |
|---|---|---|
| A | envA_S3 (and envA_C6, separately) | envB_S3prime, envC_C3 |
| B | envB_S3prime | envA_S3, envC_C3 |
| C | envC_C3 | envA_S3, envB_S3prime |

The C6 session is never used for training.

1. **P9-SIM (pre-training).** Phase 8's chosen configuration (lr, width,
   `ℓ`, recipe), retrained on `roomsim-v1` with `ref_bins = "shared40"`,
   3 seeds. Its SIM test figures are reported beside Phase 8's 44-bin
   ones (label SIM).
2. **Zero-shot (SIM->REAL).** P9-SIM's seeds on the evaluation windows,
   unchanged.
3. **Fine-tuned (SIM->REAL, fraction `f`).** The recipe below is fixed
   here, before any REAL test figure exists. There is no early stopping:
   no REAL validation room is independent of the test room.
   - Start from P9-SIM seed `s` and train every parameter.
   - Optimiser: AdamW, lr 1e-4, weight decay 0.01, batch 64, 600 steps
     (50 warm-up, cosine), fp16 autocast with a gradient scaler, gradient
     clipping 1.0.
   - Augmentation: Phase 8's link-subset rule; geometry is always
     unknown.
   - Loss: presence BCE + count CE. The zone and xy losses are off (no
     labels).
4. **Real-only (REAL, `f`).** The same architecture from a random
   initialisation (seed `s`), the same sets and steps, at Phase 8's chosen
   lr.
5. **Fractions** `f` ∈ {1%, 10%, 100%} of each training session's
   training windows:
   - From every labelled segment, take a contiguous block of
     `max(1, ceil(f · n_segment))` windows, its offset drawn by seed `s`.
   - Every method trained on REAL data at `f` uses the same blocks.
   - At 1%, that is a few seconds of each phase.

**Seeds:** 0, 1, 2. Seed `s` fixes P9-SIM, the blocks, the
initialisation, the sampling and the augmentation.

## 5. Baselines (the same evaluation windows)

1. **DSP presence (REAL)** — the S0–S5 `wifisense.pipeline.SensingPipeline`
   with its defaults (50 s warm-up, then 30 s of empty-room calibration,
   10 s hold). It runs per link on the 40 shared amplitudes of the capped
   packets, held onto a 20 Hz grid (zero-order hold: each tick takes the
   link's latest packet). It uses the session's first 80 s, which lie
   inside the leading empty segment and outside every scored window.
   - *Decision:* `occupied` at the window's end.
   - *Score for AUC:* energy / threshold.
   - *Reported as:* the median link (each link a one-link deployment, as
     today's firmware is), and the majority vote of the links (AUC: the
     median ratio).
2. **WLM (REAL)** — the significance gate of spec/locate.md §5 (step 1)
   on the per-link RSSI signal, through the Python reference's
   measurement:
   - `window_s = ℓ`;
   - null sigma calibrated on the session's first 80 s;
   - placeholder node positions, because positions are unknown, the
     estimate and its radius are not applicable, and the gate never reads
     them.

   *Presence* = the gate fires. *Score* = the largest `y_l / null_sigma_l`.
   RTI needs positions and is not applicable.
3. **Variance features + forest and MLP.** The geometry-free part of
   spec/model.md §3.3: the statistics of the per-link spread `s_l` (mean,
   std, min, max, quartiles, 90th percentile) and the used-link count.
   - *REAL-trained (label REAL):* on the fraction-`f` blocks of the
     training sessions, per seed.
   - *SIM-trained (label SIM->REAL):* on P9-SIM's SIM training windows,
     beside zero-shot.
4. **Prior** — the training blocks' majority class (AUC 0.5).

DSP and WLM calibrate on the test room's first 80 s, as a deployment
calibrates at install. Every table says so. The learned methods get no
REAL test-room data at all.

## 6. Metrics and uncertainty

**Per test session:**
- *Presence:* accuracy, balanced accuracy, AUC, false-alarm rate (on
  empty windows; leading and trailing empty also separately) and miss rate
  (still and moving separately).
- *Count:* accuracy and MAE (true counts are 0 or 1; a predicted 2 or 3
  is an error).
- *Decision rules:* spec/model.md §4's. No method abstains on presence,
  so coverage is 1 and the tables say so.

**Uncertainty:**
- Learned methods: mean ± std over the 3 seeds.
- Every method: a 95% interval from a moving-block bootstrap of the
  session's evaluation windows. Blocks are 30 windows (30 s), drawn within
  each segment so the label mix is kept, with 2,000 resamples. Each
  resample recomputes the figure per seed and averages over seeds.
- Model-versus-baseline differences: paired, on the same resampled
  blocks.

**Across folds:** the mean of the three per-session figures and their
range. With three rooms, that range is the honest spread; no pooled
interval pretends otherwise.

## 7. Outputs and tests

- **Code:**
  - `ml/opencsi.py`;
  - `ml/phase9/` (REAL cache, fine-tuning, baselines, metrics, runner);
  - tests in `ml/tests/`.
- **Results:** runs in `/home/Aman/are_u_theone/runs/phase9/`; result
  JSONs in `ml/results/phase9/`, every figure labelled; the summary table
  in ROOMSIM.md.
- **Tests:**
  - the adapter on a hand-built trial (layouts, exact times, duplicates,
    cap, windows, labels);
  - a REAL cache window equals `preprocess` on the same window;
  - the fraction blocks;
  - the moving-block bootstrap on a hand-built case;
  - the DSP zero-order hold;
  - a fine-tuning smoke run.

## 8. Limits

- One subject and a scripted protocol, with "static" meaning seated at a
  desk. Three rooms, so three folds. The folds' spread is the uncertainty
  that matters, and it is wide by construction.
- Unknown geometry: the model's geometry path is untested on REAL data,
  and so are zones and xy.
- The time base is the backend's arrival clock, with batches of ~10 packets
  per link (§2.4).
- Labels mark the protocol's phases, not verified presence frame by
  frame.
