# Model v1 — SIM pre-training (Phase 8)

**Status:** spec for Phase 8, 2026-10-25, draft. Trains and evaluates a
neural model offline on the SIM dataset (spec/dataset.md) through the
shared preprocessing (spec/preprocess.md), under ADR-009 (learned models
offline, for research only). Every number it produces is **SIM**: it
says nothing about real rooms (ADR-008 d9).

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
   tensor cores; RUN.md §1 measured bf16 about 12x slower than fp16.
   Training uses fp16 autocast with a gradient scaler. `torch.compile` is
   used when it compiles and runs on this torch (2.5.1) and GPU; the
   report says whether it did.
2. **The input is the Phase 7 preprocessing, binned at 5 Hz**
   (`bin_hz = 5`, `ref_bins = "shared44"`, `min_packets = 3`, phase off;
   phase on is an ablation). Binning gives every room (4-12 nodes, 6.7-20
   Hz per link) and every later source one time grid, `T = 5 ℓ`; at 5 Hz
   every bin of a SIM link holds at least one packet unless a packet is
   lost.
3. **xy as up to three slots with uncertainty.** Three query slots each
   predict an existence probability and a 2-D Gaussian (centre and an
   isotropic `log σ`), matched to the present people by Hungarian
   assignment; the reported radius is the 95% radius of that Gaussian
   (`2.448 σ`), and its coverage is measured, never assumed.
4. **What the model may use:** the preprocessed CSI and the declared
   scene geometry (room size, node positions: SCENE). Not the SIM
   calibration recording (`calib_*`, idealised), not the people count of
   the scene, not the scene id.
5. **Evaluation windows** are fixed per split: every second label tick
   with `t_k >= 4 s` (so every window length of the sweep fits), on the
   manifest's split (train 1,598 / val 201 / test 201 scenes).

---

## 1. Data

**Samples.** A sample is (scene, label tick `k`, window length `ℓ`):
`preprocess(from_roomsim_shard(dir, scene, k, ℓ), cfg)` with decision 2's
config, plus the labels at tick `k` (spec/dataset.md §5) and the stored
baseline at tick `k` (§6). Training draws, each epoch, `R = 64` ticks per
train scene uniformly from those with `t_k >= ℓ`; evaluation uses the
fixed windows of decision 5.

**Batch** (padded to the batch's largest link count `L`):

| Key | Shape | Content |
|---|---|---|
| `amp` | [B, L, 52, T] f32 | z-scores (spec/preprocess.md §6), 0 where masked |
| `mu`, `sigma` | [B, L, 52] f32 | the removed mean and spread |
| `carried` | [B, 52] bool | |
| `sample_mask` | [B, L, T] bool | |
| `link_mask` | [B, L] bool | links present and used |
| `gain_db`, `rssi` | [B, L, T] f32 | with `rssi_mask` [B, L, T] |
| `bin_count` | [B, L, T] f32 | packets per bin |
| `link_pos` | [B, L, 2, 3] f32 | tx and rx positions, m (SCENE) |
| `room` | [B, 3] f32 | `w, d, h` (SCENE) |
| `presence` | [B] f32 | target |
| `count` | [B] i64 | 0-3 |
| `zone_occupied` | [B, 9] f32 | target (multi-label) |
| `zone_reachable` | [B, 9] bool | cells a person can be in |
| `xy` | [B, 3, 2] f32 | people's positions, 0 where absent |
| `person_mask` | [B, 3] bool | |
| `walking` | [B, 3] bool | at the window's end |
| `wlm_published`, `rti_published` | [B] bool | stored baseline at tick `k` |
| `wlm_xy`, `rti_xy` | [B, 2] f32 | 0 when not published |
| `wlm_radius`, `rti_radius` | [B] f32 | 0 when not published |
| `meta` | list | scene index, `k`, `n_nodes`, `l_shape`, split |

Shards are loaded once into memory (the dataset is 17 GB; this host has
503 GB) before the data-loading workers start, so workers share them;
preprocessing runs in the workers.

## 2. Model

Coordinates are normalised by the room: `x / w`, `y / d`, `z / h`.

1. **Per-link encoder.** Input per link: `amp` as 52 channels over `T`
   (masked bins 0), concatenated with `gain_db - mean`, `rssi` (z-scored
   per link over the window, masked bins 0), `bin_count` and the
   `sample_mask` as 4 more channels → a 1-D conv stack over time (3
   layers, kernel 3, width `D`, GELU, masked mean and max pooling over
   time) → `D`. The side features `mu`, `log(sigma + 1e-4)` (2 x 52)
   and the per-link mean RSSI → an MLP → `D`, added.
2. **Link geometry embedding.** tx, rx (normalised), their midpoint,
   length and direction (`cos`, `sin`) → MLP → `D`, added to the link
   token. The room (`w, d, h`, m) → MLP → `D`, a global token.
3. **Across links.** Two transformer encoder layers over the link tokens
   and the room token (`D`, 4 heads, link padding masked), then
   **attention pooling** with learned queries: one pooled vector for
   presence, count and zone, and three slot queries for xy.
4. **Heads.** Presence: 1 logit. Count: 4 logits. Zone: 9 logits
   (unreachable cells masked out of the loss and the metric). xy: per slot
   an existence logit, a centre (sigmoid, in room-normalised units) and
   `log σ` (m, clamped to [-4, 2]).

**Size:** under 10M parameters at every width of the sweep (counted and
reported).

**Losses:** presence BCE + count CE + zone BCE (reachable cells) + xy:
Hungarian matching of slots to present people on the centre distance,
then the Gaussian negative log-likelihood of each matched person (in
metres) and BCE on existence (matched slots 1, others 0). Equal weights
except xy NLL (0.5); listed in the run's config.

**Training:** AdamW (weight decay 0.01), cosine schedule with 500 warm-up
steps, batch 128, 30 epochs, gradient clipping at 1.0, fp16 autocast
(decision 1), early stopping on the validation loss (patience 5). Link
dropout 0.1 during training (each link dropped with that probability) so
the model also works on link subsets (§5's one-transmitter evaluation).
Seeds fix the initialisation, the sampling and dropout.

## 3. Baselines (same split, same evaluation windows)

1. **WLM** (spec/locate.md §5): the stored outcome at the window's end
   tick — presence = published; xy = its centre with its radius
   (uncalibrated model) against the nearest present person; zone = the
   cell of its centre against `zone_occupied`; count not applicable.
   Coverage (the published share) is reported beside every figure.
2. **RTI** likewise (reported for completeness; spec/locate.md §13).
3. **Variance features + random forest and MLP.** Per window, from the
   same preprocessing: each link's spread `s_l` = mean of `sigma` over the
   carried bins; features = summary statistics of `s_l` over the links
   (mean, std, min, max, quartiles, 90th percentile), the node count, the
   room size, and a **4 x 4 link-variance map** over the room (each cell
   the mean `s_l` of the links whose segment passes within half a cell
   of the cell's centre, 0 where none). A random forest (scikit-learn,
   300 trees) and an MLP (two hidden layers of 128) are trained on the
   train windows for presence, count, zone (per cell) and, on windows with
   one person, xy. The xy MLP predicts a Gaussian like the model; the
   forest's radius is the spread of its trees' predictions (the 95th
   percentile of their distance to the forest's mean).
4. **Prior:** always the training split's majority class, the room centre
   for xy (its radius the 95th percentile of the training errors).

## 4. Metrics (SIM, test split)

Per method, mean ± std over 3 seeds (the forest and MLP with 3 seeds too;
WLM, RTI and the prior are deterministic and reported once):
- **Presence:** accuracy, balanced accuracy, false-alarm rate (empty
  windows called occupied), miss rate, AUC.
- **Count:** accuracy, mean absolute error, the 4 x 4 confusion.
- **Zone:** per-cell F1 averaged over reachable cells, and the exact-set
  accuracy (all 9 cells right).
- **xy:** median and p90 error of matched people (metres), matching recall
  and precision, the median predicted 95% radius and its **coverage**
  (fraction of matched people inside it); also on one-person windows
  only, where WLM and RTI are comparable.
- Every figure is broken down by people count and by walking/still at the
  window's end. Abstaining methods report coverage beside each figure.

## 5. Experiments

1. **Sweep** (one seed each): lr {3e-4, 1e-3, 3e-3} x width `D` {64, 128,
   256} x window `ℓ` {1, 2, 4} s = 27 runs, several at once on the GPU.
   Selection by the validation loss; the chosen configuration is reported
   with the whole table (validation only; test is touched once).
2. **Main result:** the chosen configuration, 3 seeds, on test, against
   every baseline.
3. **Ablation, impairments ON vs OFF (required):** the same configuration
   and seeds trained and tested on the hardware-impairments-off dataset
   (`roomsim-v1-hwoff`, spec/dataset.md §8; same scenes, splits and labels);
   also the ON model tested on OFF, and the OFF model tested on ON.
4. **Ablation, phase:** the main configuration with `phase` on.
5. **One transmitter:** the main model evaluated with only the links whose
   tx is node 0 (today's firmware has one transmitter, spec/dataset.md
   §12), against WLM on the same subset.

## 6. Outputs

- Code in `ml/train/` (data, model, train, evaluate, baselines, metrics,
  sweep) with tests in `ml/tests/`.
- Runs (configs, logs, checkpoints) in `/home/Aman/are_u_theone/runs/phase8/`,
  outside git.
- Results as JSON in `ml/results/phase8/` (committed: small, SIM-labelled)
  and a summary table in ROOMSIM.md; the full write-up is Phase 12's
  ML_RESULTS.md.

## 7. Tests

- Batch shapes, masks and dtypes; no NaN; padding never reaches a loss or
  a metric.
- The model under 10M parameters at every width; a deterministic forward
  pass for a fixed seed; it overfits a tiny batch.
- Hungarian matching and every metric against hand-built cases; the
  radius coverage; the per-count breakdown.
- Baseline features by hand-built cases (the link-variance map).
- A smoke run: a few scenes, one epoch, end to end, on CPU.

## 8. Limits

- SIM only: the scenes are random, the people simulated, the impairments
  uncalibrated; nothing here is a claim about real rooms (ADR-008 d9,
  ADR-009).
- The dataset assumes the planned TDMA mesh (spec/dataset.md §12); §5's
  one-transmitter evaluation is the closest SIM proxy for today's
  hardware.
- Presence is constant per scene in SIM; still people are hard (spec/
  preprocess.md §11), so presence accuracy is reported per walking/still.
- The radius is the model's own; its coverage is measured on test, not
  calibrated.
