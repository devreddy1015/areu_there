# Dataset v1 — `saans gen-dataset`

**Status:** spec for Phase 6, 2026-10-24. Revised after a two-lens review
(rules and ML fitness; feasibility against the code, with measurements):
23 findings, all adopted. Builds on spec/roomsim.md (Room Sim v1, the
simulator), spec/locate.md (Locate v1, the weighted-link-midpoint
baseline) and spec/scene.schema.json (the scene format).

**Owner decisions, 2026-10-24** (asked and answered):
1. **Provenance.** `saans-scene` gains `Provenance::Generated`, serialised
   `"generated"` (a public-API change to `saans-scene`, approved; ADR-008
   amended). Every scene this command writes says so; no generated scene
   claims to be a room.
2. **What is generated now.** The impairment layer ON only, 2,000 scenes
   x 2 min, outside git under `/home/Aman/are_u_theone/datasets/`. The
   impairment-OFF copy for Phase 8's ablation is generated then, from the
   same scene files, with `--no-impair --from DIR` (§8).

**Decisions taken here** (flagged in the report):
1. **Format:** `.npz`, one file per scene, written by a small hand-written
   zip writer (§7.2); no new heavy dependency (Parquet would pull the
   arrow crates). `rayon` is the only new crate (the brief names it).
2. **What is stored is what the ESP32 sends:** the int8 `(imag, real)`
   pairs of the first 56 buffer entries — the same two bytes per entry as
   a float16 amplitude, with phase kept for Phase 7. The brief's amplitude
   tensor `[links, subcarriers, time]` is what the loader returns (§10);
   amplitude is computed there exactly as Locate computes it.
3. **One continuous stream per scene, labels every 0.5 s.** The brief's 2 s
   windows are a view the loader cuts (§4.3), so Phase 8 can sweep the
   window length without regenerating anything. Nothing is resampled.
4. **The baseline is stored and recomputable:** Locate v1's WLM and RTI
   outcomes with its defaults at every label tick, plus the empty-room
   calibration recording Locate used (§6).
5. **Splits are fixed in the manifest**, by scene, stratified (§7.3).
6. **ML tooling lives in a new `ml/` directory** (Python with numpy, run in
   `.venv`), not in `wifisense/` or `tests/`, which stay stdlib-only.
7. **Size.** About 8 MB of occupied-room stream and 1 MB of calibration per
   scene on average: about 18 GB (17 GiB) for 2,000 scenes (the owner was
   told "about 15 GB" before the calibration recording was added).

**Rules.** No unsafe Rust. Everything written is **SIM** (scene geometry
drawn at random, people and CSI simulated) and says so in every file.
No physics changes and no public-API change beyond owner decision 1:
Room Sim and Locate run as specified, through their existing public APIs,
so no Python reference change and no golden regeneration (the provenance
value is the only shared-format change, and Python's scene loader accepts
it too). Splits are by scene, never by window or tick.

---

## 1. What it is

`saans gen-dataset --out DIR [--scenes N] [--seconds S] [--seed N]
[--threads N] [--no-impair] [--from DIR] [--overwrite]` draws `N` random
scenes (§2), runs Room Sim and Locate on each (§3, §6), and writes per
scene the scene file, its exact run configs and one `.npz` shard, plus one
manifest for the whole dataset (§7). Scenes are independent and run in
parallel with `rayon`; the output is byte-identical for any thread count
(§9).

Defaults: `--scenes 2000`, `--seconds 120`, `--seed 0`, impairments on,
threads = available cores. `S` is a positive multiple of 2 up to 600; `N`
is 1..=100000.

## 2. The scene generator

Scene `i` is drawn from its own random stream (§9) and named `gen-{i:06}`
(the `name` field). Lengths in metres, angles in degrees. Every
distribution below is a modelling choice (SIM), recorded in the manifest's
`generator` block so a later change is visible.

**2.1 Room.** `w_m`, `d_m` ~ U[3, 10], `h_m` ~ U[2.5, 3.5] (each rounded to
0.01). With probability 0.25 the footprint is an L: one corner, uniform
among the four, is cut away, with cut extents ~ U[0.3, 0.5] of `w_m` and of
`d_m` (rounded to 0.01), so each arm is at least half the room. The
polygon's bounding box is exactly (0,0)..(w,d) (scene format rule).

**2.2 Walls.** One explicit wall per footprint edge, centreline on the
edge, height the room's. One material for the room, drawn from `concrete`
0.35, `brick` 0.30, `drywall` 0.20, `wood` 0.10, `glass` 0.05; thickness
`t` ~ U by material: concrete/brick [0.15, 0.25], drywall [0.08, 0.12],
wood [0.04, 0.08], glass [0.008, 0.012] (rounded to 0.001). Room Sim's
`floor_material` from `concrete` 0.8, `wood` 0.2 and `ceiling_material`
from `concrete` 0.9, `drywall` 0.1 go into the run config (§3). The wall's
**inner face** is `t/2` inside the edge; every clearance below is measured
from it.

**2.3 Furniture.** `n` ~ U{0..6} objects, each kind from `bed` 0.2,
`table` 0.2, `chair` 0.25, `almirah` 0.15, `sofa` 0.2:

| Kind | size x | size y | size z | material |
|---|---|---|---|---|
| bed | 1.8–2.1 | 0.9–1.9 | 0.4–0.6 | fabric |
| table | 0.8–1.6 | 0.6–0.9 | 0.72–0.78 | wood 0.7, metal 0.3 |
| chair | 0.45–0.55 | 0.45–0.55 | 0.8–1.0 | wood 0.6, metal 0.2, fabric 0.2 |
| almirah | 0.9–1.4 | 0.5–0.65 | 1.8–2.1 | wood 0.5, metal 0.5 |
| sofa | 1.6–2.2 | 0.8–0.95 | 0.75–0.9 | fabric |

Sizes uniform in their ranges (rounded to 0.01); `pos.z = size.z / 2` (it
stands on the floor); yaw ∈ {0, 90, 180, 270} uniform. The centre is
uniform in the bounding box; accepted when the rotated footprint lies
inside the room footprint at least `t/2 + 0.1` from every edge and at
least 0.1 m from every placed object's footprint (axis-aligned boxes);
up to 50 tries, else the object is dropped (drawn and placed counts go to
the manifest).

Openings, which Room Sim v1 does not use as signal obstacles (spec/roomsim.md
§6.5) but which constrain node placement: with probability 0.7 one `door`
(wood, size 0.9 x 0.1 x 2.0, `pos.z = 1.0`) and `n_w` ~ U{0..2} `window`s
(glass, 1.0 x 0.1 x 1.0, `pos.z = 1.5`), each centred on the centreline
of an edge at least 1.6 m long (edge chosen in proportion to length), at
least 0.6 m from its ends, `yaw_deg` the edge's direction (`atan2(dy, dx)`
in degrees); openings on one edge at least 0.1 m apart (50 tries, else
dropped).

**2.4 Nodes.** `N` ~ U{4..12}, ids 0..N-1 in placement order. Each is
placed `max(t/2, 0.05) + 0.05` inside a random footprint edge (edge chosen
in proportion to its length; Room Sim rejects a node closer than
`t/2 + 0.01` to a wall centreline, and an opening reaches 0.05 m in), at a
uniform position along the edge at least 0.3 m from its ends, z ~
U[0.8, 2.2] (rounded to 0.01); accepted when at least 0.5 m from every
other node **in (x, y)** (Locate is 2-D, spec/locate.md §2) and not inside
any object's box; up to 200 tries per node, else the scene is redrawn
(§2.6).

**2.5 People.** `P` ~ U{0..3}. Person `j` gets a mode from walk 0.5,
stand 0.15, sit 0.2, lie 0.15:
- `lie`: on a bed no one is on yet; start at the bed's centre (Room Sim
  makes them stationary, lying, spec/roomsim.md §8.2);
- `sit`: on a free chair or sofa; start at its centre (stationary,
  sitting);
- `walk`: start drawn uniform in the bounding box, up to 1000 tries, kept
  when `saans_roomsim::motion::Walkability::point` (built from
  `RoomModel::from_scene` and the run config) accepts it; `people_mode` =
  `walk` is set explicitly, so a start Room Sim does not find walkable is
  an error, never a silent change of mode;
- `stand`: likewise, with `people_mode` = `stationary` and
  `start_heading_rad` ~ U[0, 2π).

A `lie` or `sit` with no free object falls back to `walk`. No walkable
point in 1000 tries redraws the scene (§2.6). `people[j]` has `start` only
(no `id`); `phone` is absent.

**2.6 Validity.** A drawn scene is kept when all of these accept it, with
the run config of §3 **with `impair = true`** (so `--no-impair` never
changes which scene is kept): `Scene::validate()` returns no error;
`Simulator::new`; and Locate's scene-free and geometric checks
(`check_roomsim_config`, `resolve_config`, `person_geometry`,
`check_tx_period`, all public in `saans_locate::pipeline`). Otherwise the
scene is redrawn from attempt `a + 1` (§9), at most 100 attempts (then the
command fails naming the scene index: a generator bug, not data). The
manifest records each scene's attempts and the rejection counts by reason.
The calibration (§6) runs later; its failure aborts the command naming the
scene (it has not been seen to fail).

## 3. The run

Room Sim's defaults (spec/roomsim.md §3.2) with, per scene: `seed` = the
scene's Room Sim seed (§9); `duration_s = S + 0.5` (Locate then has a tick
at `t = S`, §6; no CSI transmission at or after `t = S` is stored);
`floor_material`, `ceiling_material` (§2.2); `n_people = P`, `people_mode`
and `start_heading_rad` (§2.5); `impair = false` with `--no-impair`.
Everything else default: every node `txrx`, `slot_s` 0.0125,
`fw_subcarriers` 56, `quantize` on. The exact config is written per scene
(§7.1).

Three Room Sim runs per scene, all deterministic: the occupied run that
fills §4 and §5; Locate's own `RoomSimSetup` (its calibration run and, via
`stream()`, the same occupied run again for §6's ticks, so Locate is
exactly `run_roomsim`); and the calibration run once more to store its
packets (§6). Measured (release, 10 x 10 m, 3 people): 4 nodes 4.0 + 4.8 s,
12 nodes 15.2 + 18.3 s; 2,000 scenes take about 10 minutes on 96 cores.

## 4. The CSI stream

### 4.1 Indexing (integers only)
Room Sim transmits on a TDMA grid: transmission `n = m*N + slot` (its
`Packet.m`, `Packet.slot`) at `t = float(n) * slot_s`, `N` transmitters,
80 transmissions per second (`1 / slot_s`; the command refuses a config
where `2 / slot_s` is not an integer). Every directed link `l = (tx, rx)`
(Room Sim's `link_index` order, `L = N(N-1)`, equal to Locate's canonical
order) is received at most once per superframe `m`, in its transmitter's
slot `tx_slot[l]`. Superframe `m` is column `m` of the stream; `M =
ceil(80 S / N)` columns; transmissions with `n >= 80 S` are not stored (the
last column may be partly masked). No floating-point time decides a
column.

### 4.2 Arrays

| Array | Shape | dtype | Content |
|---|---|---|---|
| `csi_iq` | [L, 56, M, 2] | int8 | the first 56 entries of the ESP32 buffer, buffer order, each `(imag, real)` exactly as the datagram carries them (spec/roomsim.md §12.4); 0 where masked |
| `csi_mask` | [L, M] | uint8 | 1 where the packet was received (a lost packet, or a column past `n = 80S`, is 0) |
| `rssi` | [L, M] | int8 | the packet's RSSI, dBm (−100..0); 0 where masked |
| `tx_slot` | [L] | uint8 | the transmitter's slot: packet `(l, m)` is at `t = float(m*N + tx_slot[l]) * slot_s` |
| `subcarrier_k` | [56] | int8 | Room Sim's `buffer_k[0..56]` (LLTF bins 0..31, −32..−9) |
| `subcarrier_active` | [56] | uint8 | `buffer_active[0..56]`: 44 ones; the 12 DC and guard nulls are found by this index, never by value (spec/roomsim.md §13) |

Amplitude is `sqrt(re*re + im*im)` with `re`, `im` widened to int32 (as
Locate computes it, spec/locate.md §4.3).

### 4.3 Windows
A window of length ℓ (a multiple of 0.5 s) **ending at label tick `k`**
(§5, time `t_k`) holds, per link, the columns `m` whose transmission `n =
m*N + tx_slot[l]` lies in `[80 (t_k − ℓ), 80 t_k)`: `ceil(80ℓ/N)` or one
fewer columns per link (when `80ℓ/N` is not an integer some links get one
fewer, masked). The brief's 2 s windows are ℓ = 2 ending at `t_k = 2(w+1)`,
`w = 0..S/2−1`; their label ticks are `window_end_tick` [S/2] int32
(`k = 4w + 3`). The loader's `window(scene, k, ℓ)` returns the brief's
`[L, 56, T]` amplitude with `T = ceil(80ℓ/N)` (40 at 4 nodes, 20 at 8, 14
at 12 for ℓ = 2) and its `[L, T]` mask, and the per-window `link_mask`
(links with at least one packet).

## 5. Labels (SIM truth)

At label ticks `t_k = 0.5 (k + 1)`, `k = 0..2S−1`, from Room Sim's truth
of **transmission `n = 40(k+1) − 1`**, the last before `t_k` (truth exists
for lost transmissions too). People are by index, at most 3, padded; index
order carries no meaning.

| Array | Shape | dtype | Content |
|---|---|---|---|
| `count` | [K] | uint8 | people in the room (constant per scene in Room Sim v1: nobody enters or leaves) |
| `presence` | [K] | uint8 | `count > 0` |
| `xy` | [K, 3, 2] | float32 | person `i`'s `(x, y)`, m; **NaN** where absent ((0,0) is a real corner) |
| `person_mask` | [K, 3] | uint8 | 1 for `i < count` |
| `walking` | [K, 3] | uint8 | on a walk segment |
| `posture` | [K, 3] | uint8 | 0 stand, 1 sit, 2 lie; 255 absent |
| `zone` | [K, 3] | int8 | person `i`'s cell of the 3 x 3 grid over the bounding box (0,0)..(w,d): `3*row + col`, `col = min(floor(3x/w), 2)`, `row = min(floor(3y/d), 2)`; −1 absent |
| `zone_occupied` | [K, 9] | uint8 | 1 for each cell holding a person: **the zone target** (multi-label) |
| `zone_single` | [K] | int8 | the cell when `count == 1`, else −1 |
| `t_label` | [K] | float64 | `t_k` |

Per scene: `zone_floor_frac` [9] float32, the fraction of each cell's area
that is floor (1 except where an L's cut corner removes part of up to four
cells, and the whole corner cell when both cut extents exceed a third),
and `zone_reachable` [9] uint8 = `zone_floor_frac > 0`. Phase 8 masks
unreachable cells out of the zone head and its accuracy.

**Scoring notes for Phase 8** (recorded here so the labels are read as
meant): an xy head is scored by optimal (Hungarian, Euclidean) assignment
over present people; a one-estimate baseline (WLM, RTI) is scored for
presence as `published`, for xy against the nearest present person
(optimistic when `count >= 2`; reported per count), for zone as a hit when
its cell is set in `zone_occupied`; coverage (the published share) is
reported beside every accuracy (ADR-008 d7).

## 6. The baseline (Locate v1, SIM)

Locate with its defaults (`window_s` 1, `tick_hz` 2, `link_signal`
`rssi_dbm`, spec/locate.md §9) through `RoomSimSetup::new` and `stream()`
on the same scene and config — exactly `run_roomsim`. Its ticks `1 + j/2`
are label ticks `k = j + 1`; the arrays below are on the label grid
(`K = 2S`), with `k = 0` (`t = 0.5`, before Locate's first tick) marked
`warming_up`.

| Array | Shape | dtype | Content |
|---|---|---|---|
| `wlm_xy`, `rti_xy` | [K, 2] | float32 | estimate centre; NaN when not published |
| `wlm_radius`, `rti_radius` | [K] | float32 | `radius_m` (uncalibrated model); NaN when not published |
| `wlm_reason`, `rti_reason` | [K] | uint8 | 0 estimate, 1 `no_disturbance`, 2 `not_significant`, 3 `implausible_radius`, 4 `warming_up` |
| `calib_baseline`, `calib_null_sigma` | [L] | float64 | Locate's empty-room calibration (spec/locate.md §4.2) |
| `calib_mask` | [L] | uint8 | calibrated from enough windows |
| `calib_iq`, `calib_mask_m`, `calib_rssi` | [L, 56, Mc, 2], [L, Mc], [L, Mc] | int8, uint8, int8 | the calibration run's packets Locate calibrated from (same scene, config and seed, nobody in the room, `S + 0.5 <= t < S + 0.5 + calib_s`), on the §4.1 grid from superframe `calib_m0` (in `meta_json`) |

Locate's tick at `t_k` uses the received packets with `t_k − 1 <= t <=
t_k`: it also sees the transmission at exactly `t_k`, which a window ending
at `t_k` does not hold (and, at `t_k = S`, is not stored); its truth is one
12.5 ms slot after the label's. It sees 1 s of RSSI where a 2 s window
holds 2 s of CSI. Phase 8 recomputes the WLM in Python (`wifisense/localize`,
which agrees with Rust by spec/locate.md §12, bit for bit on glibc) from
the stored streams and calibration when it compares at another window
length or signal, and says which baseline each number uses.

## 7. Files

### 7.1 Layout
```
DIR/
  manifest.json
  scenes/gen-000000.json           the scene (provenance "generated")
  scenes/gen-000000.roomsim.json   the exact Room Sim config of its run
  scenes/gen-000000.locate.json    Locate's resolved config
  shards/gen-000000.npz            one per scene
```
`Simulator::new` on the scene and its `.roomsim.json` reproduces the
shard's packets (tested).

### 7.2 The shard
A zip of `.npy` members, in the order of §4.2, §5, §6, then: `links` [L,
2] uint8 (tx, rx), `node_ids` [N] uint8, `node_pos` [N, 3] float32 (SCENE
geometry, generated), `room_size` [3] float32 (`w_m, d_m, h_m`),
`footprint` [V, 2] float32 (4 or 6 vertices), `window_end_tick` [S/2]
int32, `meta_json` [B] uint8 (UTF-8 JSON: `{"label":"SIM", "spec":"dataset
v1", "scene_index", "scene_id", "scene_sha256", "roomsim_seed", "n_nodes",
"n_people", "slot_s", "n_tx", "columns", "calib_m0", "impair"}`).

Exact bytes, so byte-identity (§9) holds:
- each member `NAME.npy`: magic `\x93NUMPY`, version 1.0, u16 LE header
  length, header `{'descr': '<f2', 'fortran_order': False, 'shape': (60,), }`
  (descr one of `<f4 <f8 |u1 |i1 <i4`; a 1-tuple keeps its comma),
  space-padded and ended by `\n` so the data starts at a multiple of 64;
  little-endian C-order data;
- the zip: members in the order above; each local header version 20,
  flags 0, method 0 (STORED), DOS time 0, date 0x0021, IEEE CRC-32, sizes in
  the local header (no data descriptor), no extra field, no comment; a
  central directory and end record likewise; no Zip64 (a shard of 4 GiB or
  more is refused).

### 7.3 The manifest
`{"label":"SIM", "spec":"dataset v1", "command", "seed", "n_scenes",
"seconds", "label_hz":2, "impair", "baseline_valid"` (false with
`--no-impair`, §8), `"from"` (the source dataset of `--from`, or null),
`"roomsim_defaults", "locate_config", "generator"` (every §2 distribution),
`"rejections"` (counts by reason), `"splits": {"train":[…], "val":[…],
"test":[…]}`, `"split_counts"` (by `n_people`, `n_nodes`, `l_shape`),
`"scenes": [{"index", "id", "split", "shard", "shard_sha256",
"scene_sha256", "n_nodes", "n_links", "n_people", "modes", "objects_drawn",
"objects_placed", "openings", "l_shape", "wall_material", "attempts"}]`,
`"totals"` (labels, bytes, wall time)`}`. Written last: a directory without
it is an unfinished run.

**Splits** are by scene, stratified by (`n_people`, `l_shape`): within each
stratum the scenes are ordered by their split draw (§9); the first
`round(0.1 n)` go to test (at least 1 when the stratum has 3 or more), the
next `round(0.1 n)` to val, the rest to train. The scene is the unit of
independence: a scene's labels are strongly correlated (presence and count
are constant), so every interval on these data resamples scenes, never
windows.

## 8. The command

- Phase 1 draws and checks every scene (§2) in parallel; phase 2 prints the
  estimated size from the accepted node counts and the target filesystem's
  free space, refuses when the estimate exceeds the free space less 5 GB,
  then runs §3–§6 in parallel.
- Refuses a non-empty `--out` unless `--overwrite` (which removes only
  `manifest.json`, `scenes/` and `shards/` there).
- Progress on stderr every 5% (scenes done, elapsed, ETA); at the end a
  summary, every line labelled SIM: scenes, labels, bytes, the share of
  scenes with people, people-count, node-count and wall-material
  histograms, and the share of 2 s windows whose end tick has a published
  WLM estimate (its coverage).
- Exit 0 on success, 2 on a refused argument (naming it), 1 on an I/O
  error.
- **`--no-impair`** changes only the radio: the scene files, configs (but
  `impair`), people and every §5 label are byte-identical to the
  impairments-ON dataset of the same seed (tested); §4's and §6's arrays
  change (no packet is lost). With `impair = false` the empty room's RSSI
  is constant, every `calib_null_sigma` is 0 and WLM never publishes, so
  the manifest says `"baseline_valid": false`: Phase 8's ablation reports
  the WLM baseline from the impairments-ON dataset on the same split.
- **`--from DIR`** reads DIR's scene files, configs and splits instead of
  drawing them (`--scenes`, `--seconds`, `--seed` are then taken from DIR's
  manifest and refused if given differently).

## 9. Determinism

All randomness comes from the repo's one generator, `saans_sim::rng::Rng`
(spec/roomsim.md §2): scene `i`, attempt `a` (0..99) draws in §2's order
from `Rng::stream(seed, (i << 8) | a)`; the scene's Room Sim seed is that
stream's next `next_u64()` after the scene is accepted; the split draw is
one `uniform()` of `Rng::stream(seed, (1 << 40) | i)`. Shards are written to
a temporary name by the thread that made them, then renamed. The same
arguments give byte-identical `scenes/`, `shards/` and manifest (except
`totals`) for any `--threads`; a test checks 1 vs 4 threads.

## 10. Python side (`ml/`)

ML tooling, Python 3.11 in `.venv` with numpy and matplotlib:
- `ml/roomsim_dataset.py`: `load_manifest(dir)`, `load_scene(dir, i)` →
  dict of arrays (checks `label == "SIM"`, the spec string and the shard's
  sha256 against the manifest), `amplitude(iq)`, and `window(scene, k, ℓ)`
  → the brief's `[L, 56, T]` amplitude, its mask and `link_mask` (§4.3).
- `ml/plot_sample.py DIR --scene i --window w --out FILE.png`: the sample
  plot for the owner — the room plan (footprint, nodes, links, the 3 x 3
  grid, people at the window's end, the WLM estimate with its radius or its
  abstention) and the window's CSI amplitude for a few links (subcarrier x
  time, masked slots shown as gaps), every panel labelled SIM.
- Tests in `ml/tests/`, run with `.venv/bin/python -m unittest discover -s
  ml/tests`.

## 11. Tests

- `saans-scene`: `generated` round-trips; the schema accepts it; Python's
  scene loader accepts it.
- Generator, seeds 0..3 x 200 indices: every accepted scene valid (§2.6);
  at least 95% accepted at attempt 0; the accepted shares of wall material,
  L-shape, node count and people count inside a 99% binomial interval of
  §2's; objects never overlap and keep their clearances; nodes 0.5 m apart
  in (x, y); openings along their edges; people modes as drawn.
- Stream: for a small scene, every received packet lands in exactly one
  `(l, m)`, with `csi_iq` equal to the buffer and `rssi` equal; no
  collision for N = 4..12 over 120 s; masks consistent.
- Labels: equal to Room Sim's truth at transmission `40(k+1) − 1`; zones and
  `zone_floor_frac` by hand-built cases (rectangle, each L corner).
- Baseline: the §6 arrays equal `run_roomsim`'s tick records for the same
  scene and config; the calibration packets are the ones Locate counted.
- Reproduction: `Simulator::new` on a written scene and `.roomsim.json`
  gives the shard's packets.
- Format: a Rust reader of the zip/npy bytes round-trips every member; numpy
  (in `ml/tests`) loads every member with the right shape and dtype; the
  loader's windows match §4.3 by hand-built cases.
- Determinism: 1 vs 4 threads byte-identical; `--no-impair` and `--from`
  as §8 says.
- CLI: refusals, `--overwrite`, the size check (injected free space).

## 12. Limits

- Every scene is a random box or L with random furniture: not a room
  anyone surveyed. §2's distributions are guesses (SIM).
- Room Sim v1's limits apply (spec/roomsim.md §16); the impairment
  parameters are uncalibrated (Phase 10).
- People never enter or leave during a run; presence is per scene.
- No phone; Phase 8's heads do not use it.
- **The tensor assumes the planned all-`txrx` TDMA mesh**, links separated
  by transmitter, which today's firmware (one transmitter, no TDMA) and
  datagram (no transmitter id) cannot provide (spec/roomsim.md §10,
  §12.4). A recording from today's hardware has one transmitter's N−1
  links at 20 Hz; a model trained here transfers to REAL only once the
  firmware carries a transmitter id (a SIM->REAL item). Phase 8 also
  reports a one-transmitter evaluation (links with `tx = 0` only).
- The stored baseline sees 1 s of RSSI per tick with an uncalibrated
  radius (§6); comparisons at other windows or signals use the recomputed
  WLM and say so.
