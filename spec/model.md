# Model v1 — SIM pre-training (Phase 8)

**Status:** spec for Phase 8, 2026-10-25. Revised after review (12
findings, all adopted, each settled by a measurement). Trains and
evaluates a neural model offline on the SIM dataset (spec/dataset.md)
through the shared preprocessing (spec/preprocess.md), under ADR-009
(learned models offline, for research only). Every number it produces is
**SIM**: it says nothing about real rooms (ADR-008 d9).

**The brief (owner, Phase 8):** a per-link encoder (1D conv or small
transformer over time x subcarriers) + link geometry embedding (tx/rx
positions) + attention pooling across links + heads for presence, zone,
xy, people count; under 10M parameters; bf16, torch.compile if stable;
baselines in the same harness (weighted-link-midpoint; MLP / random forest
on per-link variance features); scene split 80/10/10; a parallel sweep
(lr, width, window length) on the 48 GB GPU; required ablation:
impairment layer ON vs OFF. Hard rules: splits by scene; the WLM baseline
on the same split; 3 seeds, mean ± std; coverage beside any accuracy that
uses abstention; estimates with uncertainty.

**Decisions taken here** (under the owner's blanket permission of
2026-10-25; flagged in the report):
1. **fp16, not bf16.** This GPU (Quadro RTX 8000, Turing) has no bf16
   tensor cores (RUN.md §1: bf16 about 12x slower). fp16 autocast with a
   gradient scaler; `torch.compile` (measured working on torch 2.5.1 and
   this GPU: 49 ms per step compiled against 86 ms eager).
2. **Input: the Phase 7 preprocessing binned at 5 Hz** (`bin_hz = 5`,
   `ref_bins = "shared44"`, `min_packets = 3`, phase off; phase on is an
   ablation). Every SIM bin then holds a packet unless one was lost (empty
   bins 0%, 0.04%, 1.4% at 4, 8, 12 nodes). **Only the 44 reference
   subcarriers `R` reach the model** (SIM never carries the other 8, so
   weights on them would never train and would meet real data untrained
   in Phase 9), and **`bin_count` and `gain_db` are not inputs**:
   `gain_db` follows Room Sim's normalisation to the HT-LTF power and
   separates occupied from empty windows by itself (AUC 0.80, walking
   0.96, measured SIM), an artefact nothing says the ESP32 shares; it is
   an ablation. A bin's noise falls with its packet count (empty-room
   `sigma` binned at 5 Hz is 0.50, 0.71, 0.91 of unbinned at 4, 8, 12
   nodes), so Phase 9 decimates a source faster than 20 Hz per link to
   20 Hz before binning.
3. **xy as up to three slots with uncertainty.** Three query slots each
   predict an existence probability and an isotropic 2-D Gaussian; the
   reported radius is its 95% radius (`2.448 σ`), floored so that it is at
   least `min_plausible_radius_m` (0.3 m, spec/locate.md §9); its coverage
   is measured, never assumed.
4. **What the model may use:** the preprocessed CSI and the declared scene
   geometry (room size, node positions: SCENE). Not the SIM calibration
   recording (`calib_*`, idealised), not the scene's people count, not the
   scene id. The WLM and RTI baselines do use the idealised calibration;
   every comparison says so.
5. **Evaluation windows** are fixed per split: every second label tick with
   `t_k >= 4 s` (so every window length of the sweep fits and every length
   is scored on the same windows), on the manifest's split (train 1,598 /
   val 201 / test 201 scenes).
6. **Intervals resample scenes.** Presence and count are constant per
   scene, so windows are not independent: every interval is a scene-level
   bootstrap (spec/dataset.md §7.3), never a window-level one (measured:
   window-level intervals are 2.6–5.8 times too narrow).

---

## 1. Data

**Cache.** For each dataset (`roomsim-v1`, `roomsim-v1-hwoff`) and window
length `ℓ`, every second label tick's window is preprocessed once, in
parallel, and stored as float16 in a memory-mapped cache (`amp`, `mu`,
`sigma` at `R`; `rssi`; the masks; the labels; the stored baselines), in
`/dev/shm` when it fits and otherwise under
`/home/Aman/are_u_theone/runs/phase8/cache/`. Preprocessing in the training
workers would cost about 280 core-hours over the sweep (measured: 5, 9, 16
ms per window at `ℓ` = 1, 2, 4 s); the cache is computed once. Workers
gather, apply link subsets and pad.

**Samples.** A sample is (scene, label tick `k`, `ℓ`) with `t_k >= ℓ`.
Training draws, each epoch, 64 cached ticks per train scene; evaluation
uses decision 5's windows.

**Batch** (padded to the batch's largest link count `L`; `R` = the 44
reference subcarriers):

| Key | Shape | Content |
|---|---|---|
| `amp` | [B, L, 44, T] f16 | z-scores at `R` (spec/preprocess.md §6), 0 where masked |
| `mu`, `sigma` | [B, L, 44] f16 | the removed mean and spread at `R` |
| `sample_mask` | [B, L, T] bool | |
| `link_mask` | [B, L] bool | links present, used and kept by the subset rule |
| `rssi` | [B, L, T] f16 | dBm, with `rssi_mask` |
| `link_pos` | [B, L, 2, 3] f32 | tx and rx positions, m (SCENE) |
| `room` | [B, 3] f32 | `w, d, h` (SCENE) |
| `geometry_known` | [B] bool | false when §2's unknown-geometry rule applies |
| `presence` | [B] f32 | target |
| `count` | [B] i64 | 0-3 |
| `zone_occupied` | [B, 9] f32 | target (multi-label) |
| `zone_reachable` | [B, 9] bool | |
| `xy` | [B, 3, 2] f32 | people's positions, 0 where absent |
| `person_mask` | [B, 3] bool | |
| `walked` | [B, 3] bool | walking at some label tick in `(t_k - ℓ, t_k]` |
| `wlm_*`, `rti_*` | | the stored baseline at tick `k` (published, xy, radius) |
| `meta` | list | scene index, `k`, `n_nodes`, `l_shape`, split |

## 2. Model

Coordinates are normalised by the room: `x / w`, `y / d`, `z / h`.

1. **Per-link encoder.** Input per link: `amp` (44 channels over `T`,
   masked bins 0), `rssi` minus its per-link window mean (dB, masked bins
   0; no division) and the `sample_mask` → a 1-D conv stack over time (3
   layers, kernel 3, width `D`, GELU, masked mean and max pooling over
   time) → `D`. The side features `mu` and `log(sigma + 1e-4)` (2 x 44)
   and the per-link RSSI mean and `log(std + 0.1)` (dB; a constant link has
   std 0) → an MLP → `D`, added.
2. **Link geometry embedding.** tx, rx (normalised), their midpoint, length
   and direction (`cos`, `sin`) → MLP → `D`, added to the link token; the
   room (`w, d, h`, m) → MLP → `D`, a global token. When `geometry_known`
   is false, both are replaced by learned "unknown" tokens.
3. **Across links.** Two transformer encoder layers over the link tokens
   and the room token (`D`, 4 heads, padding masked), then attention
   pooling with learned queries: one pooled vector for presence, count and
   zone, and three slot queries for xy.
4. **Heads.** Presence: 1 logit. Count: 4 logits. Zone: 9 logits
   (unreachable cells masked out of the loss and the metrics). xy: per slot
   an existence logit, a centre (sigmoid, room-normalised) and
   `σ = 0.1225 + softplus(raw)` m, at most 7.4 m (so every 95% radius is
   at least 0.3 m).

**Size:** under 10M parameters at every width (measured on a sketch: 0.18M,
0.68M, 2.65M at `D` = 64, 128, 256).

**Losses:** presence BCE + count CE + zone BCE (reachable cells) + xy:
Hungarian matching of slots to present people on the Euclidean distance
in metres, then the Gaussian negative log-likelihood of each matched person
and BCE on existence (matched slots 1, others 0); xy and zone losses are
skipped when `geometry_known` is false. Equal weights except xy NLL (0.5);
listed in the run's config.

**Training:** AdamW (weight decay 0.01), cosine schedule with 500 warm-up
steps, batch 128, 30 epochs, gradient clipping at 1.0, fp16 autocast,
early stopping on the validation loss (patience 5). **Link subsets** per
sample: with probability 0.2 keep only one uniformly drawn transmitter's
links, with probability 0.1 one uniformly drawn link, otherwise drop each
link with probability 0.1 (at least one kept). **Unknown geometry:** with
probability 0.1. Seeds fix the initialisation, sampling, subsets and
dropout.

## 3. Baselines (same split, same evaluation windows)

1. **WLM** (spec/locate.md §5), two forms, each labelled: *stored*
   (Locate's outcome at the end tick: 1 s of RSSI, defaults; spec/dataset.md
   §6) and *recomputed* (the Python reference `wifisense/localize` on the
   stored RSSI streams and calibration, `window_s` = the compared model's
   `ℓ`, the same ticks, and on every link subset of §5; spec/dataset.md
   §6, §12). Presence = published; xy = its centre with its radius
   (uncalibrated model) against the nearest present person; zone = its
   centre's cell against `zone_occupied`; count not applicable. Both use
   the idealised calibration the model is denied (decision 4). Coverage
   (the published share) is reported beside every figure.
2. **RTI** likewise, stored and recomputed.
3. **Variance features + random forest and MLP.** Per window, from the same
   preprocessing: each link's spread `s_l` = mean of `sigma` over `R`;
   features = summary statistics of `s_l` over the links (mean, std, min,
   max, quartiles, 90th percentile), the node count, the room size, and a
   **4 x 4 link-variance map** in room-normalised coordinates: cell
   `(i, j)`, centre `((i + 0.5)/4, (j + 0.5)/4)`, holds the mean `s_l` over
   links whose normalised 2-D segment passes within 1/8 of that centre,
   plus a 4 x 4 map of those links' count (both 0 where none). A random
   forest (scikit-learn, 300 trees) and an MLP (two hidden layers of 128)
   are trained on decision 5's windows of the train split for presence,
   count, zone (per cell) and, on one-person windows, xy; the xy MLP
   predicts a Gaussian like the model; the forest's radius is the 95th
   percentile of its trees' distances to the forest's mean.
4. **Prior:** the training split's majority class; for xy the footprint
   centroid (spec/locate.md §13's reference point), its radius the 95th
   percentile of training errors.

## 4. Metrics (SIM, test split)

**Decision rules** (fixed before training): presence at probability >=
0.5; count the argmax; a reachable cell occupied at >= 0.5; a slot is an
estimate at existence >= 0.5. Matching (loss and metrics) is Hungarian on
Euclidean distance in metres; a matched pair beyond 2 m counts as a miss
and a false estimate (also reported at 1 m).

- **Presence:** accuracy, balanced accuracy, false-alarm rate (empty
  windows called occupied), miss rate, AUC.
- **Count:** accuracy, mean absolute error, the 4 x 4 confusion.
- **Zone:** per-cell F1 averaged over reachable cells, each reachable
  cell's confusion counts (ADR-008 d8), and the exact-set accuracy over
  the reachable cells.
- **xy (multi-person):** median and p90 error of matched people (m),
  matching recall and precision, the median predicted 95% radius and its
  coverage (the fraction of matched people inside it).
- **Against one-estimate methods** (WLM, RTI, prior, forest and MLP xy),
  on one-person windows only: each method gives one estimate (the model:
  its slot with the highest existence probability, whatever the others
  say) and is scored on median and p90 error, single-cell zone accuracy
  (the model's most probable reachable cell) and radius coverage: (a) on
  all one-person windows, the abstaining method's coverage beside; (b)
  paired, on exactly the windows the abstaining method published, with the
  paired bootstrap interval of the difference and the window count.
- **Breakdowns:** every figure by people count and by motion within the
  window: *walking* (some present person `walked`), *still* (none did),
  and empty; xy per person likewise.

**Uncertainty:** on test, the mean over seeds ± the std over seeds (model,
forest, MLP: 3 seeds), and a 95% interval from a scene-level bootstrap
(2,000 resamples of the test scenes with replacement, stratified by people
count, every window of a drawn scene kept, the figure recomputed per seed
and averaged over seeds). WLM, RTI and the prior are deterministic and get
the bootstrap interval only, never a bare figure. Every model-versus-
baseline difference is a paired bootstrap interval (the same resampled
scenes). The sweep table on val carries the same interval.

## 5. Experiments

1. **Sweep** (one seed each): lr {3e-4, 1e-3, 3e-3} x width `D` {64, 128,
   256} x window `ℓ` {1, 2, 4} s = 27 runs, several at once on the GPU.
   Selection by the validation loss; the whole table is reported (val
   only; test is touched once).
2. **Main result:** the chosen configuration, 3 seeds, on test, against
   every baseline.
3. **Ablation, impairments ON vs OFF (required):** the same configuration
   and seeds trained and tested on `roomsim-v1-hwoff` (same scenes, splits
   and labels; spec/dataset.md §8), and the cross cases (ON model on OFF,
   OFF model on ON). Every figure sits beside the evaluated dataset's own
   WLM and RTI on the same split (hwoff's `baseline_valid` is true). The
   report states that with phase off the per-packet gain normalisation
   removes AGC and device gain exactly, ripple moves into `mu`, and CPE and
   timing offsets reach only phase — so ON ≈ OFF means the preprocessing
   removes them.
4. **Ablation, phase:** the main configuration with `phase` on, as 2 x 44
   more channels (cos, sin at `R`, 0 where `phase_mask` is false); SIM's
   phase impairment is exactly what spec/preprocess.md §5 removes, so a gain
   is an upper bound.
5. **Ablation, gain:** the main configuration with `gain_db` minus its
   per-link window mean as one more channel; any gain is reported as a SIM
   artefact (decision 2).
6. **Link subsets:** the main model on (a) the links whose tx is node 0
   (today's firmware has one transmitter), (b) the single link (0, 1), (c)
   (b) with geometry unknown — each against WLM and RTI recomputed on the
   same links; presence and count for all, xy and zone for (a).

## 6. Outputs

- Code in `ml/train/` (cache, data, model, train, evaluate, baselines,
  metrics, sweep) with tests in `ml/tests/`.
- Runs (configs, logs, checkpoints) in `/home/Aman/are_u_theone/runs/phase8/`,
  outside git.
- Results as JSON in `ml/results/phase8/` (committed: small, SIM-labelled)
  and a summary table in ROOMSIM.md; the full write-up is Phase 12's
  ML_RESULTS.md.

## 7. Tests

- Cache and batch: shapes, masks and dtypes; no NaN; a cached window equals
  `preprocess` on the same window (float16 rounding); padding never reaches
  a loss or a metric; link-subset and unknown-geometry rules.
- The model under 10M parameters at every width; a deterministic forward
  pass for a fixed seed; it overfits a tiny batch; every radius >= 0.3 m.
- Hungarian matching, every metric and the scene bootstrap against
  hand-built cases.
- Baseline features (the link-variance and count maps) by hand-built cases;
  the recomputed WLM equals the stored one when `window_s` = 1 s and every
  link is kept.
- A smoke run: a few scenes, one epoch, end to end.

## 8. Limits

- SIM only: the scenes are random, the people simulated, the impairments
  uncalibrated; nothing here is a claim about real rooms (ADR-008 d9,
  ADR-009).
- The dataset assumes the planned TDMA mesh (spec/dataset.md §12); §5.6's
  link-subset evaluations are the closest SIM proxies for today's hardware
  and for Phase 9's single link.
- Presence is constant per scene in SIM; still people are hard (spec/
  preprocess.md §11), so every figure is broken down by walking and still.
- The radius is the model's own; its coverage is measured on test, not
  calibrated.
