# Room Sim — status and handoff

The working record for the Room Sim / ML / simulator-realism plan (twelve
phases: scene format → web demo → simulator upgrade → localisation →
live demo → dataset → model → real data → Sionna and calibration → export).
Read it with [RUN.md](RUN.md) (how to run everything on the GPU box) and
[ADR-008](docs/adr/008-declared-geometry-and-drawing.md) (what may be
drawn). Every number below comes from a run on this machine; simulator and
scene numbers are labelled as such.

**Branch:** `saans/scene-format` · **Last updated:** 2026-10-24

## Where it stands

| Phase | | Status |
|---|---|---|
| 0 | Environment | **Done.** Rust 1.99, venv with CUDA torch, sionna-rt; RUN.md. |
| 1 | Scene format | **Done.** `saans-scene` crate, JSON Schema, 3 example scenes, `saans import-roomplan`. |
| 2 | Static web demo | **Done.** |
| 3 | Simulator upgrade | **Done.** Room Sim v1: `spec/roomsim.md`, the Python reference `wifisense/roomsim/` and the Rust crate `saans-roomsim`, bit-identical; `saans roomsim-export`. Three decisions need the owner (*Open items* 7). |
| 4 | Localisation and phone tracking | **Done.** Locate v1: `spec/locate.md`, Python additions in `wifisense/localize/` and the Rust crate `saans-locate` (the existing RTI ported, a baseline that abstains, a simulated phone tracked by RTT and RSSI), bit-identical on 24 golden sets; measured on Room Sim. Decisions for the owner: *Open items* 12. |
| 5 | Live demo | **Done.** `spec/livedemo.md`: the `/roomsim` page's SIM view over a 10 Hz room stream (`/api/roomsim`, admin and safety only), Room Sim and Locate run per viewer in the gateway, REPLAY of recorded runs, `saans roomsim-eval`; `Mode { Sim, Replay, Live }`. No LIVE room view exists. Decisions for the owner: *Open items* 13. |
| 6 | Dataset | **Done.** `spec/dataset.md`: `saans gen-dataset` draws random rooms (`provenance: generated`, an owner-approved change to `saans-scene`) and writes per room one `.npz` shard (the ESP32's int8 I/Q per link on the TDMA grid, labels every 0.5 s, Locate's baseline and the empty-room calibration), the scene and its run configs, and a manifest with stratified by-scene splits; `ml/` reads it. 2,000 rooms x 2 min generated in `/home/Aman/are_u_theone/datasets/roomsim-v1` (outside git). Decisions for the owner: *Open items* 14. |
| 7 | Preprocessing | **Done — waiting for the owner's review.** `spec/preprocess.md`: one function (`ml/preprocess.py`, numpy) for every CSI source — the SIM dataset, Room Sim's `.sim.pcap` exports, later the public ESP32 datasets and the owner's recordings — through one record (received packets only, integer-ns times, the 52 LLTF subcarriers with a carried mask): per-packet gain normalisation over the 44 shared subcarriers, optional unwrap-free phase sanitisation, optional binning, and a per-link, per-subcarrier z-score over the window that keeps the removed mean and spread. A stdlib reference (`ml/preprocess_ref.py`) pins the arithmetic. A dataset room read as a shard and as a pcap gives identical windows. Decisions for the owner: *Open items* 15. |
| 8–12 | Model, real data, Sionna/calibration, recordings, export | Not started. **Blocked on an ADR** for learned models (see *Open items*). |

The working rule is to stop after each phase for review. That is why
nothing past Phase 7 has been started, not lack of time.

## What exists

| Path | What |
|---|---|
| `rust/crates/saans-scene/` | Scene types, validation (every error, not the first), plane geometry, Apple RoomPlan importer. |
| `spec/scene.schema.json`, `spec/scenes/*.json` | The contract and three worked examples (bedroom 4.2×3.6, office 6×5, L-shaped hall 9×7), all `provenance: hand_written`. |
| `rust/crates/saans-cli` → `saans import-roomplan` | RoomPlan scan → scene.json, printing every assumption it had to make. |
| `spec/roomsim.md` | **Room Sim v1**, the normative spec: ITU-R P.2040 slab walls/floor/ceiling to second order, furniture as specular faces or point scatterers plus blockage, clutter and a diffuse tail, 6–16-point people with gait, breathing, three-edge knife shadowing, stand/sit/lie, random-waypoint walks around furniture, the ESP32 impairment layer and the exact datagram. Every number it produces is SIM. §16 lists what it does not model. |
| `wifisense/roomsim/` | The Python reference (stdlib only), normative under ADR-002; `export_run` writes the same files as the CLI. |
| `rust/crates/saans-roomsim/` | The Rust implementation: streaming, constant memory, about 1 s of compute per simulated bedroom-minute. Reads scenes without `saans-scene`'s `write`. |
| `rust/crates/saans-cli` → `saans roomsim-export` | Scene → `PREFIX.sim.pcap` (raw-CSI datagrams in `wifisense record`'s format), `PREFIX.truth.jsonl`, `PREFIX.manifest.json` (label SIM). RUN.md §4c. |
| `spec/golden/roomsim_*`, `spec/scenes/fixtures/` | Twelve golden sets that pin both implementations to each other, and two fixture scenes (a polygon room with implied walls and a yawed metal cabinet; a specular-face test room). |
| `spec/locate.md` | **Locate v1**, the normative spec: the existing RTI ported as is; a link disturbance from RSSI variance; the weighted-link-midpoint baseline with a significance gate, abstention and a radius; a simulated phone (Wi-Fi RTT and RSSI to every node, line of sight from the scene) with Gauss–Newton fixes and a Kalman filter; every estimate a position with a radius or an abstention with a reason. §13 states what the measurements show. |
| `wifisense/localize/{radius,config,baseline,phone,pipeline}.py` | The Python reference for the new pieces (stdlib only), plus `RssiSignal` in `measure.py` and covariance fields on `track.py`'s `PositionEstimate`. |
| `rust/crates/saans-locate/` | The Rust crate: the RTI port (`pynum`, `geometry`, `invert`, `measure`, `track`, with a bit-exact port of CPython's `math.hypot`), the new pieces, the Room Sim pipeline behind the `roomsim` feature, and `examples/locate_measure.rs`, which prints the measured numbers. |
| `spec/golden/localize_*`, `locate_*` | 24 golden sets (2.1 MB) pinning the ported RTI, the new estimators and the Room Sim pipeline. |
| `rust/crates/saans-gateway/src/roomsim.rs` + `assets/roomsim.*` | The `/roomsim` demo page: routes, its own stricter CSP, embedded scenes served without people or phone. |
| `spec/livedemo.md` | **Live Demo v1**, the normative spec of Phase 5: the room stream's access, sessions, clock and frames; statistics; the REPLAY records file; the page's drawing rules; `roomsim-eval`. |
| `rust/crates/saans-locate/src/{variant,stream,score,records}.rs` | The run variants; `RoomSimSetup`/`RoomSimStream`, a streaming driver equal to `run_roomsim` record for record (which now runs on it); the §5 statistics; the records file and the JSON encodings the gateway's frames share. |
| `rust/crates/saans-gateway/src/roomstream/` | The room stream: handshake checks, one session thread per viewer, a socket-free engine with an injected clock, frame building and redaction, REPLAY loading (`--room-replay`). |
| `rust/crates/saans-gateway/assets/roomsim-sim.js` | The page's SIM layer: the one function that draws an estimate, links, trails, truth, controls and the statistics panel. Served under a path named by its own digest (`SIM_REV` in `roomsim.rs`). |
| `rust/crates/saans-cli` → `saans roomsim-eval` | The page's statistics without a browser, every line SIM, or one JSON object; `--records-out` writes a REPLAY file. RUN.md §4e. |
| `spec/dataset.md` | **Dataset v1**, the normative spec of Phase 6: the random-room generator, the run, the CSI stream, labels, the stored baseline, the shard bytes, splits, the command and its determinism. |
| `rust/crates/saans-cli` → `saans gen-dataset` (`src/gen_dataset/`) | Random rooms → Room Sim and Locate → `.npz` shards, scene files, configs and a manifest; parallel (rayon), byte-identical for any thread count; `--no-impair --from` makes the paired hardware-impairments-off copy. RUN.md §4f. |
| `ml/` | ML tooling (Python, numpy, `.venv`): `roomsim_dataset.py` (loader; windows of any length cut by integer index), `plot_sample.py` (the sample plot), tests. |
| `/home/Aman/are_u_theone/datasets/roomsim-v1` | The generated dataset, **outside git**: 2,000 rooms x 2 min, 17.3 GB, SIM. |
| `spec/preprocess.md` | **Preprocessing v1**, the normative spec of Phase 7: the canonical record, the readers, the subcarrier grid, gain normalisation, phase sanitisation, binning, the z-score, the exact arithmetic and the tests. |
| `ml/preprocess.py`, `ml/preprocess_ref.py` | The shared preprocessing (numpy) and its stdlib-only reference, which pins the arithmetic for a later port; `ml/roomsim_dataset.py` gains an I/Q window. RUN.md §4g. |
| `phase6-samples/` | The sample plot the brief asked for, from that dataset: an L-shaped room with two people, and an empty room. |
| `rust/crates/saans-gateway/assets/vendor/three-0.186.1/` | Vendored three.js, integrity-checked, pinned by SHA-256, one documented one-line change (VENDOR.md). |
| `rust/crates/saans-gateway/roomsim-demo.toml` | Loopback config to run the demo. |
| `rust/crates/saans-gateway/tests/browser/roomsim_check.py` | **The repo's first browser end-to-end test.** Drives Chrome: render, CSP, clicks, toggles, phone width. |
| `docs/adr/008-declared-geometry-and-drawing.md` | Accepted. What the room view may draw and who may see it. |
| `.github/workflows/tests.yml` | First CI job that runs tests at all. **Never run** — no push access from here. |
| `RUN.md` | Environment facts, how to run, monitor and resume, known failures with causes. |

## Decisions taken

| When | Decision | By |
|---|---|---|
| 2026-10-21 | ADR-008 accepted: geometry is declared, never inferred; the room may be drawn, a person only as an estimate with an uncertainty radius, gated by role and consent; LIVE gated by a mechanism. | Owner |
| 2026-10-21 | `Shared::mode` becomes `Mode { Sim, Replay, Live }` (a public-API change to saans-gateway). Done in Phase 5; the wire spelling stays `live`/`replay`. | Owner |
| 2026-10-21 | The phone is simulated only; a LIVE phone needs its own ADR. | Owner |
| 2026-10-21 | Static demo now; mode enum and scene-to-space binding left for Phase 5. | Owner |
| 2026-10-21 | `provenance` is a required scene field; the examples say `hand_written`. | Implements ADR-008 decision 4 |
| 2026-10-21 | three.js 0.186.1 vendored, served from the binary; demo CSP allows nothing inline. | Implements ADR-008 decision 12 |
| 2026-10-21 | Room Sim is a **new** crate `saans-roomsim` and package `wifisense/roomsim/`, not an extension of `saans-sim` as the brief says. Extending `saans-sim` changes its public types (`Scenario`, `Room`, `SimOptions`, `LinkFrame`, `Tick`) and moves the `sim_geometric` golden vectors the S3 detector evaluation depends on. | **Proposed — owner to confirm** (open item 7) |
| 2026-10-21 | `saans-scene` gains a `write` cargo feature: `Serialize`, `Scene::save`, `to_json_pretty` and the RoomPlan importer exist only with it. Only `saans-cli` enables it; `tests/boundary.rs` holds two allowlists (writers; readers such as `saans-roomsim`). A public-API change to `saans-scene`. | Implements ADR-008 decision 1; **owner to confirm the API change** |
| 2026-10-21 | `saans-cli` is both the scene writer and a SIM CSI producer (`export` already was; `roomsim-export` now too). No command writes a scene from CSI; nothing structural stops a future one. | **Proposed — owner to confirm**, or split `roomsim-export` into its own binary |
| 2026-10-21 | Scene fixes in files this work wrote: bedroom almirah centre x 3.5 → 3.3 (node 1 was inside it); hall "sitter" (7.0, 1.8) → (7.0, 0.6), onto the sofa; office occupants y 2.5 → 2.1, onto their chairs (they were frozen standing behind them). | Fixes found by the Phase 3 reviews |
| 2026-10-24 | Locate v1 (spec/locate.md header): Python first, then Rust (ADR-002); the person estimators read RSSI variance on Room Sim data because Room Sim's AGC normalises CSI power away; the radius is a model flagged `radius_calibrated: false` until Phase 5 measures coverage; the baseline abstains rather than guessing the room centre; the phone is SIM only and its radio model lives in `saans-locate`, not Room Sim; one person estimate per estimator; scenes with `roles` or stacked nodes refused by the person pipeline. | **Proposed — owner to confirm** (open item 12) |
| 2026-10-24 | Phase 5 (spec/livedemo.md header): the additive APIs (`can_see_position`, a room handle on `Shared`, the streaming driver, score and records in `saans-locate`); the frame adapted to ADR-008 (per-estimator estimates with radii or abstentions, truth by index in SIM only, phone RTT and RSSI each with a radius, a timestamp per block); SIM plus REPLAY of recorded runs, no LIVE; RTI never drawn without the baseline, the baseline may be drawn alone. | Owner |
| 2026-10-24 | Phase 5 choices made here (spec/livedemo.md *Decisions taken here*): `/api/roomsim` checks `Origin` and, on an auth-off gateway, a loopback `Host`; one thread per session, at most 4; frames never dropped; radii shown as uncalibrated models; truth drawn from each record's own truth; an empty-room run per session so coverage always sits beside the false-alarm rate; variants named `declared`/`with_walker`/`empty`, the page defaulting to `with_walker`; `can_see_position(Some(_))` false until a per-space opt-in exists; the simulation crates at `opt-level = 2` in the dev profile; the pcap `--replay` left unlabelled. | **Proposed — owner to confirm** (open item 13) |
| 2026-10-24 | Phase 6 (spec/dataset.md header): `Provenance::Generated` in `saans-scene` (ADR-008 amended); the impairments-ON dataset now, the OFF copy in Phase 8 from the same scenes. | Owner |
| 2026-10-24 | Phase 6 choices made here (spec/dataset.md *Decisions taken here*): `.npz` from a hand-written zip writer; the ESP32's int8 I/Q stored rather than amplitude (same bytes, phase kept; the loader returns the brief's amplitude); one continuous stream per room with labels every 0.5 s, the brief's 2 s windows cut by the loader; Locate's outcomes and the empty-room calibration stored; stratified by-scene splits in the manifest; `--no-impair` = the hardware impairments off with the receiver's noise kept (Room Sim's master switch removes the noise too and makes presence trivial); people start 0.5 m apart; `ml/` for ML tooling; about 17 GB rather than the 15 GB first quoted. | **Proposed — owner to confirm** (open item 14) |
| 2026-10-24 | Phase 7 (spec/preprocess.md header): the z-score is per link and subcarrier over the window, with the removed mean and spread kept as side features; the output grid is the 52 LLTF subcarriers with a per-source carried mask. | Owner |
| 2026-10-25 | Phase 7 choices made here (spec/preprocess.md *Decisions taken here*): numpy in `ml/` with a stdlib reference and no Rust port; one canonical record and one reader per source; gain and the phase fit computed over the 44 subcarriers every source carries (so 44- and 52-bin sources are normalised identically), the other carried bins output with the same gain; phase off by default, output as (cos, sin), the slope never output; no resampling by default, optional integer-ns binning that averages amplitudes; unbiased sigma with a 3-packet minimum; labels derived, never overridable; a recording with several transmitters refused rather than pooled. | **Proposed — owner to confirm** (open item 15) |
| 2026-10-21 | ESP32 defaults follow the firmware where it says: `ltf_merge` on, smoothing **off** (`channel_filter_en = false`), 56 forwarded entries. TDMA (`slot_s` 12.5 ms) is the *planned* schedule, labelled SIM; today's one-transmitter firmware is a documented config (RUN.md §4c). | Spec §3.2, §10 |

## How it was verified

All on the GPU box. The rows above **Phase 3** are the Phase 2 record
(2026-10-21, commit `ce115fa`); the Phase 3 rows are from 2026-10-23 and
the Phase 4, 5 and 6 rows are from 2026-10-24.

| Suite | Command | Result |
|---|---|---|
| Rust workspace | `cargo test --workspace --locked` | all pass (the gateway's 56 unit + 12 e2e include a test that `/roomsim` is **absent** from a default build) |
| Gateway with the demo | `cargo test -p saans-gateway --features roomsim --locked` | 65 unit + 12 e2e pass |
| saans-scene | `cargo test -p saans-scene` | 31 + 13 + 2 + 1 doc-test pass |
| Browser, real Chrome | `tests/browser/roomsim_check.py` | **119/119**, six consecutive runs |
| Python reference, CPython 3.11.5 (box default) | `python3 -m unittest discover -s tests` | 459 run; 2 fail + 1 error, all pre-existing (RUN.md §3) |
| Python reference, CPython 3.12.15 | same, under 3.12 | 459 run; **1** error, the UDP buffer test |
| C++ edge DSP | `make -C edge/dsp/test check` | PASS with **g++ only** — clang++ is not installed here, and the Makefile's banner wrongly says both (RUN.md §3) |
| clippy | `--workspace --all-targets`, with and without `roomsim` | 4 warnings, all in code this work did not touch |
| Scene files | JSON Schema draft 2020-12 | all three validate |
| **Phase 3** | | |
| Room Sim, Rust | `cargo test -p saans-roomsim --locked` (standalone, as CI) | 68 pass: 28 unit, 12 golden, 8 independent-truth, 19 run, 1 doc; `cargo tree` shows `saans-scene` with its default features only (no `write`) |
| saans-scene | `cargo test -p saans-scene` | 2 + 13 + 32 + 1 doc-test pass (Phase 3 added a check of the fixture scenes) |
| Room Sim parity | `saans-roomsim/tests/golden.rs` (11 sets) and `saans-cli/tests/roomsim_export.rs` (`roomsim_export`) against `spec/golden/roomsim_*` | all 12 sets agree **bit for bit** (worst relative error 0, no rounding-tie exemptions); a single changed value in any golden file fails its test (checked by mutation) |
| Room Sim, Python | `python3 -m unittest discover -s tests -p 'test_roomsim*'` on 3.11.5 and 3.12.15 | 100 pass on both (also on 3.10.12); the golden test re-runs every generator and compares every stored array |
| `roomsim-export` | `cargo test -p saans-cli`; Python vs Rust export on four configurations | 9 unit + 10 integration pass; Python `export_run` and the CLI give byte-identical pcaps and equal truth and manifests on the bedroom (default, 2 s), the hall (one transmitter, `slot_s` 0.05), the office and the multi fixture |
| Golden regeneration | CPython 3.14.8, `generate.py` | 0 files changed by a full regeneration; 35 sets, of which 12 are Room Sim (3.0 MB); every file committed before Phase 3 is byte-identical to HEAD |
| **Phase 4** | | |
| Locate, Rust | `cargo test -p saans-locate --locked`, and with `--features roomsim` (standalone, as CI) | 167 pass and 1 ignored (the optional 1.2 M-pair `py_hypot` dump) standalone; 180 with `--features roomsim`; `cargo tree` shows `saans-scene` without `write` |
| Locate parity | `saans-locate/tests/golden.rs` against `spec/golden/localize_*`, `locate_*` | all 24 sets agree **bit for bit** (0 values differ); `py_hypot` bit-exact against CPython 3.14 on 1,403 edge pairs and 10^6 drawn pairs (SHA-256); a mutation in each kind of check fails it |
| Locate, Python | `tests/test_locate_*.py` and `tests/test_localize_*.py` | 136 locate tests pass on 3.10.12, 3.11.5, 3.12.15 and 3.14.8 (the golden test re-runs every generator and compares every stored array); the full suite runs **695** (559 + 136): on 3.12.15 only the known UDP-buffer error, on 3.11.5 and 3.10.12 that plus the two known pre-3.12 failures — the pre-Phase 4 baseline |
| Measured (SIM) | `cargo run --release -p saans-locate --features roomsim --example locate_measure` | 45 runs of 120 s in about 12 s; figures in spec/locate.md §13 |
| **Phase 5** | | |
| Stream, score, records | `cargo test -p saans-locate --features roomsim --locked` | 254 pass (1 ignored); `RoomSimStream` equals `run_roomsim` record for record for every shipped scene, variant and step size, `run_roomsim` now runs on it and every golden set is still bit-identical; standalone 167 |
| Room stream | `cargo test -p saans-gateway --features roomsim --locked` | 132 pass: Origin/Host and role refusals, audit, sessions and the cap, the clock to the quantum, pause/speed/restart, frames never dropped, REPLAY without truth, no `Person.id`/`Phone.id` anywhere (JSON walk), nothing on `live_tx`; a default build serves no `/api/roomsim` |
| `roomsim-eval` | `cargo test -p saans-cli --locked` | 33 pass; its `--json` statistics equal `score(run_roomsim(..))` plus the empty-room run |
| Browser, real Chrome | `tests/browser/roomsim_check.py` against release builds, two gateways and a recorded hall run | **233/233** (119 static + 114 SIM view) in about 83 s, in three runs |
| Measured (SIM) | `saans roomsim-eval` on each scene, `declared` and `with_walker`, 120 s, seed 0 | 3-12 s per run in release; figures in spec/livedemo.md §13, every line SIM |
| **Phase 6** | | |
| `gen-dataset` | `cargo test -p saans-cli --locked` | 70 pass: the generator's distributions against §2's numbers over 800 scenes, the integer stream indexing for N = 4..12, labels against Room Sim's truth, the baseline against `run_roomsim`, the zip/npy bytes read back by a separate Rust reader, splits rebuilt from their draws, `--no-impair` and `--from` pairing, an exact `zone_reachable` (property-tested against every person position of 800 runs) |
| Determinism | 1, 4, 8 and 96 threads; two separate full runs | byte-identical shards and scene files; the second full run, after the `zone_reachable` fix, changed exactly the 16 shards the fix named and nothing else |
| `ml/` loader and plot | `.venv/bin/python -m unittest discover -s ml/tests`, with `SAANS_DATASET_DIR` on the generated dataset | 42 pass; every one of the 2,000 shards checked (members, dtypes, shapes, meta, null subcarriers, labels inside the floor and in reachable zones, packet counts per window) in 197 s |
| Generated (SIM) | `saans gen-dataset --out /home/Aman/are_u_theone/datasets/roomsim-v1` (release, 96 cores) | 2,000 rooms x 120 s in 10 min (6.1 GB peak memory), 17.34 GB, 0 redraws; 1,492 rooms with people; splits 1,598 / 201 / 201, val and test each 51 / 54 / 48 / 48 rooms with 0 / 1 / 2 / 3 people and 50 L-shaped; packet loss 2.0%; at the end of 2 s windows the WLM baseline publishes in 57.5% of occupied windows (median error 1.31 m, median radius 5.07 m, uncalibrated) and in 5.3% of empty ones (false alarms); RTI publishes in 31.8% of occupied and 23.8% of empty windows |
| **Phase 7** | | |
| Preprocessing | `.venv/bin/python -m unittest discover -s ml/tests` (with `SAANS_DATASET_DIR`) | 105 pass for the preprocessing and the dataset loader: numpy against the stdlib reference within per-field tolerances on random and SIM windows; gain and phase invariance; layouts and the REAL layout check; guards; the pcap reader's label, address, duplicate, clock-step and malformed-datagram rules; and a dataset room exported with `saans roomsim-export` read back from the pcap giving windows identical to the shard's (also across all 60 windows of a full-length 12-node run) |
| Mutation | 86 mutants of `ml/preprocess.py` and its tests, then 18 targeted ones | the first round killed 70; the survivors exposed seven untested behaviours, each now pinned by its own test, and every targeted mutant is caught |
| Throughput | one core, SIM 2 s windows | about 390 windows/s (2.3 µs per packet); 110 with phase; 230 binned at 10 Hz |
| Whole suites (Phase 7) | workspace `cargo test --locked`; full Python suite on 3.10.12 and 3.12.15 | Rust 598 pass (1 ignored); Python 695 at the baseline (3.12.15: only the UDP-buffer error; 3.10.12: that plus the two known pre-3.12 failures). No Rust or `wifisense/` code changed in Phase 7 |
| Whole suites (Phase 6, from a clean build) | workspace `cargo test --locked`; gateway with `roomsim`; locate, roomsim, scene standalone; clippy with and without `roomsim`; full Python suite on 3.10.12, 3.11.5, 3.12.15; `make -C edge/dsp/test check`; golden regeneration | Rust workspace 593 pass (1 ignored), gateway with `roomsim` 132, saans-locate 167 / 254, saans-roomsim 68, saans-scene 49, saans-cli 70 after the last fix; clippy shows only the 4 earlier warnings; Python 695 at the pre-Phase 6 baseline (3.12.15: only the UDP-buffer error; 3.11.5 and 3.10.12: that plus the two known pre-3.12 failures); edge PASS; a full regeneration on CPython 3.14.8 changes 0 of 59 sets' files |
| Whole suites (Phase 5, from a clean build) | workspace `cargo test --locked`; gateway with `roomsim`; clippy with and without `roomsim`; full Python suite on 3.10.12, 3.11.5, 3.12.15; `make -C edge/dsp/test check`; golden regeneration | Rust workspace 560 pass (1 ignored), gateway with `roomsim` 132; clippy shows only the 4 earlier warnings; Python 695 at the pre-Phase 5 baseline (3.12.15: only the UDP-buffer error; 3.11.5 and 3.10.12: that plus the two known pre-3.12 failures); edge PASS; a full regeneration on CPython 3.14.8 changes 0 of 59 sets' files |
| Whole suites (Phase 4, from a clean build) | workspace `cargo test --locked`; gateway with `roomsim`; full Python suite on 3.10.12, 3.11.5, 3.12.15; `make -C edge/dsp/test check`; golden regeneration | Rust workspace 446 pass (1 ignored, the optional hypot dump), gateway with `roomsim` 79; Python 695 at the pre-Phase 4 baseline (row above); edge PASS (g++); a full regeneration on CPython 3.14.8 changes 0 of 59 sets' files |
| Whole suites (Phase 3 record) | workspace `cargo test --locked`; full Python suite on 3.11.5 and 3.12.15; `make -C edge/dsp/test check` | Rust workspace 279 pass, gateway with `roomsim` 79 pass; Python **559** run (459 + 100 Room Sim): on 3.12.15 only the known UDP-buffer error, on 3.11.5 and 3.10.12 that plus the two known pre-3.12 failures — exactly the pre-Phase 3 baseline; edge PASS (g++) |

**Review.** Phase 2 went through two rounds of multi-agent review, each
finding checked by a second agent who tried to refute it by reproducing
it. Round one — ADR-008 compliance, security, frontend correctness, repo
conventions — confirmed 44 findings and refuted none; 43 are fixed and
the 44th is open item 2. Its two serious ones: the page was compiled into
every gateway build, contrary to the accepted ADR-008 decision 13, and
clicking a label selected whatever was behind it. Round two audited that
the fixes had landed, hunted for regressions they introduced, and
fact-checked these documents: 35 more confirmed, mostly in the docs (a
clang++ pass that never happened, a CI trigger that would not have fired
on this branch, a "rule" with no source in the tree). Those are fixed too.

The browser test checks what a browser can drive: label clicks,
Back/Forward, keyboard (and leaving Ctrl+0 to the browser), plan-view
panning, failed and unknown scenes (picker and URL stay honest), implied
walls, floor clicks, a tiny room, a landscape phone, and the no-WebGL
fallback (whose notice must survive later loads). Fixed but **not** under
any check: an OS theme change while open, WebGL context loss and restore,
a pixel-ratio change, out-of-order scene loads, a hash change during the
first load, the inset focus ring, and selection links when object names
repeat.

**Phase 3 review.** Two independent implementations were written from
the spec by separate agents, then reconciled set by set until they
agreed bit for bit (the one numeric difference: an optimised Rust build
fuses `sin` and `cos` into glibc `sincos`, which differs in the last bit
for about one argument in 1500; Rust now keeps them apart). A five-lens
review (physics, ESP32 and wire format, Rust robustness, Python and
tests, spec and rules), each finding checked by a second agent who tried
to refute it by reproducing it, confirmed 24 findings, partly confirmed
20 and refuted 3. The serious ones, all fixed in the spec and both
implementations: wall bounces were taken on the wall's centreline
instead of its face, which for wall-mounted nodes moved the simulated
RSSI (SIM) by up to 10.6 dB; the diffuse tail decayed twice as fast as
its parameter said; body shadowing missed the far side edge (up to 2.7
dB too much simulated loss); a node inside a wall's thickness was
accepted; `slot_s` was labelled REAL though it comes from a planning
model; the export overwrote earlier runs and could leave a manifest
describing other files; a malformed `phone.id` was echoed in an error
message; several golden sets compared only part of what they store, and
no set covered the non-default impairment switches. A second round (fix
audit, regression hunt, docs fact-check, again with a refuting verifier
per finding) confirmed 15 more and partly confirmed 9, refuted none, and
all are fixed: the scene-id check could still echo a phone number when
the scene also had a later JSON error; `stats` and `replay` labelled a
real capture SIM when an export with the same prefix sat beside it; a
few unbounded parameters still let Rust write all-zero CSI where Python
raised; no golden set covered the geometry, object, tail, shadowing or
people switches (a twelfth set now does); and spec and doc statements
the first fixes had made stale (the non-reciprocal links now reach about
14% of `max |H|`, SIM).

**Phase 4 review.** As in Phase 3: two implementations from one spec,
reconciled to bit-identity set by set (one real divergence: Rust took a
fix's reports in the caller's order, the spec and Python in node order).
Before any code, a four-lens review of the spec confirmed or partly
confirmed 50 findings — among them that calibrating on another seed
calibrates another room, that a per-link 3σ gate over 12–56 links fires in
an empty room, that a NaN radius could be published, and that undamped
Gauss–Newton diverged. After the build, a five-lens review with
differential fuzzing of Python against Rust confirmed or partly confirmed
44 more: a 50-iteration limit discarded fixes that had converged to a
millimetre; collinear anchors published a confident mirror-image fix up to
9.4 m off; a zero radius scale could publish a bare point; Python could label
Room Sim output REAL; and the measurement printout lacked the reference
point, the radius sizes and the empty-room rates needed to read it
honestly. All are fixed except the regeneration hints in `edge/dsp/bench/` (open item 12). A second round (fix audit, regression hunt with differential checks, docs fact-check; each finding again checked by a refuting agent) confirmed 19 and partly confirmed 9 more, all fixed: the collinear test still let through anchors collinear to rounding on an axis-aligned line (a node at `0.1+0.2` published a mirror fix 8.4 m off), now replaced by a test that does not depend on the line's direction; a library caller could still publish a zero radius; Python checked a far-away phone only at epochs; the two implementations resolved phone defaults differently outside Room Sim runs; and several §13 sentences overstated what the measurements show.

**Phase 5 review.** Before any code, a four-lens review of
`spec/livedemo.md` (rules and ADR-008, gateway, protocol, page), with a
refuting verifier per finding, confirmed or partly confirmed 46 of 48
findings. Among them:
- a browser cannot see why a WebSocket handshake was refused, so the
  page's sign-in message must come from `/api/me`;
- coverage shown without the empty-room false-alarm rate breaks ADR-005
  §5;
- truth drawn at the frame's time but scored at the tick's made the drawn
  gap disagree with the printed error by up to 0.6 m;
- the phone's `blocked` flags were ground truth outside the truth key,
  so REPLAY leaked them;
- `Origin == Host` alone does not stop DNS rebinding;
- the streaming driver as first specified could not be written in safe
  Rust.

Each part was then built and attacked by one or two reviewers who ran
experiments and mutation tests, and the confirmed defects were fixed:
- the stream: 5, including a refusal test that compared the code with
  itself;
- the score: 5, including nulls that were never told apart from absent
  keys;
- the records reader: 4, including repeated JSON keys accepted and tick
  truth not checked against the run's people;
- the gateway groundwork: 5;
- `roomsim-eval`: 4;
- the room stream: 7, including a restart late in calibration that was
  still applied, a restart flood starving frames, a 10 s stall rule that
  fired after 41 s, and two frames in flight;
- the page: 18, including truth crosses and error segments drawn without
  their labels, estimate labels under the view's corner boxes, a plan
  view opened from the URL that did not fit room-sized rings, REPLAY
  controls showing the requested run instead of the recorded one, and a
  LIVE `state` message accepted.

**Phase 6 review.** Before any code, two reviewers (rules and ML
fitness; feasibility against the code, with measurements) found 23
defects in the first draft of `spec/dataset.md`, among them that nodes
placed 0.05 m inside an edge sit inside any wall thicker than 0.08 m, so
Room Sim would have rejected nearly every concrete, brick or drywall room
and the dataset would have been silently wood and glass; that a
floating-point slot formula collides for 7 of the 9 node counts; that
storing amplitude discards phase for no saving; and that fixed 2 s windows
block Phase 8's window-length sweep. After the build, a code review found
7 more (an empty `--out ""` could overwrite a dataset; a floating-point
residue marked a fully cut zone as floor) and a data review found 13: Room
Sim's `impair = false` also removes the receiver's noise, so an empty room
has exactly zero variance and presence is trivial (AUC 0.999 from one
number) — `--no-impair` now turns off only the hardware impairments; the
stored calibration is an exact, drift-free empty room, so a model given it
separates even still people perfectly (AUC 1.00 against 0.60 without it)
— documented, and such a model is reported separately; AGC steps (in
about 17% of 2 s link-windows) dominate raw amplitude variance; and the
sample plot clipped the WLM's room-sized circle and ranked links by gain
steps. All are fixed.

**Phase 7 review.** Two reviewers read the draft of `spec/preprocess.md`
(fitness for Phases 8, 9, 11 and honesty; exactness against the code and
data, with experiments) and found 24 defects. Among them:
- the optional binning averaged complex values that carry a random
  per-packet phase, keeping 40–60% of the amplitude and erasing the
  presence contrast — bins now average amplitudes;
- a pcap reader pooled every transmitter a receiver hears into one
  "link" (an empty room looked busier than an occupied one);
- normalising 52-bin and 44-bin sources over different bins shifted the
  features by about 3% — normalisation now uses the 44 shared bins;
- float time edges put SIM packets in different bins in the two
  readers — times are integer nanoseconds;
- numpy and the stdlib disagree in the last bits wherever the arithmetic
  was left open — the spec now pins it, with per-field tolerances.

A code review of the build then found 9 defects in the pcap reader and the
record, all fixed. A renamed SIM export could be read as REAL. Two boards
sharing a node id, duplicate datagrams and clock steps were accepted
silently. Extreme finite CSI could overflow into NaN.

**GPU.** Phases 0–7 used none. The only GPU work so far was the fp16/bf16
benchmark in RUN.md §1.

## Open items for the owner

1. **Nothing is pushed.** This box has no SSH key, no credential helper
   and no token. Every commit is on `saans/scene-format` locally, and in a
   standalone bundle at the repo root, `saans-scene-format.bundle`
   (git-ignored locally), with the series as `patches/*.patch`. From a
   machine that can push:
   `git fetch saans-scene-format.bundle saans/scene-format:saans/scene-format && git push origin saans/scene-format`.
   Or give this box a key, add the printed public key at
   github.com/settings/keys, and point the remote at SSH:
   `ssh-keygen -t ed25519 -f ~/.ssh/id_ed25519 -N "" && cat ~/.ssh/id_ed25519.pub`, then
   `git remote set-url origin git@github.com:devreddy1015/areu_there.git`.
2. **Security, pre-existing: DNS rebinding on a loopback gateway.** On the
   default loopback deployment authentication is off by design, and the
   gateway checks neither `Host` nor `Origin`. A web page that rebinds its
   own hostname to 127.0.0.1 can therefore talk to the gateway as the
   local admin. This is gateway-wide and predates this work; it matters
   more once Phase 5 serves real rooms. Suggested fix, for the owner to
   approve because it touches the auth layer: a router middleware that
   answers `421` to any `Host` other than localhost, 127.0.0.1, [::1], the
   configured `http_addr`, or an explicitly configured public host name
   (needed behind a reverse proxy, or every proxied request would be
   refused), and `403` to a foreign `Origin` on the WebSocket and on every
   POST, with e2e cases for both. Phase 5's `/api/roomsim` already
   checks `Origin` and, on an auth-off gateway, a loopback `Host`, so
   rebinding is closed for that endpoint only (spec/livedemo.md §2.2).
3. **An ADR for learned models, before Phase 8.** ADR-007 keeps neural
   networks and trained classifiers out of the S0–S5 path until the DSP
   is measured on real rooms. Offline training and evaluation can sit
   outside that path, but the carve-out should be written down as an ADR
   before Phase 8 starts, the way every other decision here is recorded,
   rather than left implicit.
4. **CI has never run.** `.github/workflows/tests.yml` exists because
   ADR-008 decision 14 requires it, and it now triggers on a push to any
   branch, so pushing `saans/scene-format` will be its first run. Every
   command in it was run locally first except `sudo sysctl`, which was
   not run on this shared machine; that it makes the UDP test pass is
   expected, not observed.
5. **ADR-008 items that arrive with Phase 5:** the `Mode { Sim, Replay,
   Live }` enum (approved); refusing at config load to serve the room
   view LIVE without the feature (decision 13's second half); binding a
   surveyed scene to a configured space, behind the authenticated `/api`
   routes; and zones (decision 2), which have no type in the runtime yet.
   (Decision 1's `write` feature now exists, Phase 3; see item 7.)
   Also before a real scene is served: add `#[serde(deny_unknown_fields)]`
   to the scene types, so the Rust reader rejects what the schema's
   `additionalProperties: false` rejects, and validate a scene with
   `saans-scene` before the gateway serves it.
6. **Small things, each the owner's call.** The four clippy warnings in
   `http.rs` and `history.rs` are trivial but in code this work did not
   touch. The workspace declares MSRV 1.80 while `saans-sim`'s tests need
   a much newer toolchain. `test_replay_send_over_udp` would pass on any
   kernel if it read while sending. The box's Chrome is 123 (2024); the
   browser test passes on it, and should be run on a current Chrome too.
   `edge/dsp/test/Makefile` prints "(g++ and clang++)… under both
   toolchains" even when it skipped clang++ — a misleading banner in
   `edge/`, which this work does not touch. Three early commit messages
   on this branch mention the gateway demo's tests before the commit that
   adds them; the history was left as it is rather than rewritten.
7. **Three Phase 3 decisions for the owner** (*Decisions taken*): Room Sim
   as a new crate rather than inside `saans-sim`; the `write` feature as a
   public-API change to `saans-scene`; `saans-cli` as both scene writer and
   SIM CSI producer. Each is reversible; the second is the one the
   standing rule requires asking about.
8. **Check the ESP32 CSI buffer on hardware.** The simulator copies the
   firmware byte for byte, but the layout it copies is an assumption
   (spec §4): ESP-IDF's documented HT20 order (LLTF then HT-LTF, FFT
   order, imaginary first), with the firmware forwarding the first 56
   entries — which, if the layout holds, are 44 LLTF values and 12 nulls,
   not the "56 usable subcarriers (LLTF+HT-LTF)" the Kconfig help says. Worse,
   `tx_role.cpp` never pins the ESP-NOW PHY rate, and broadcast ESP-NOW
   defaults to 1 Mbps DSSS, which carries no OFDM training field at all.
   On an ESP32-S3 receiving `tx_role.cpp`'s frames, log `info->len`,
   `rx_ctrl.sig_mode`, `rx_ctrl.rate`, `cwb`, `secondary_channel` and
   `first_word_invalid`; then decide whether `tx_role.cpp` should call
   `esp_wifi_config_espnow_rate` (a firmware change, the owner's call).
9. **Links that are not reciprocal by the last bit.** The shipped scenes
   use round coordinates, so some paths pass exactly through a wall
   corner or graze a box edge and are admitted or blocked differently in
   the two directions (spec §15: up to about 14% of `max |H|` (SIM), hall
   link (4, 6), a double bounce exactly in a wall corner). Both implementations agree; the asymmetry is physical-model
   fragility, not a bug. Nudging node positions by a centimetre would
   remove it (and change the scene hashes and golden vectors).
10. **SIM labelling of replays.** A `.sim.pcap` carries no label inside
    (the wire format must match the firmware). `wifisense stats` and
    `replay` now print a SIM banner for one. The gateway's pcap replay is
    labelled SIM only when `hello`, `/api/me` and `/api/health` gain a
    `label` (later work, spec/livedemo.md §1): it cannot tell an export
    from a capture, so Phase 5 keeps it plain `replay`, and the
    dashboard's tooltip says a `.sim.pcap` is simulated.
11. **What Phase 10 must calibrate.** Every SIM-labelled parameter in spec
    §3.2 is a guess: transmit power, the tail's power and decay, object
    scatter and blockage constants, AGC, ripple, chip noise and the noise
    floor. Known simplifications (spec §16) that calibration cannot fix:
    TE-only wall coefficients (up to about 7 dB on steep paths), no
    per-board RSSI offset, the Fresnel test for furniture faces ignoring
    obliquity, and exports without the FeatureFrames a raw-mode node also
    sends.

12. **Phase 4 decisions and findings for the owner.**
    - *Decisions* (spec/locate.md header; *Decisions taken*): Python first;
      RSSI variance as the Room Sim link disturbance; model radii flagged
      uncalibrated; the baseline abstains; the phone's radio model in
      `saans-locate`; one person estimate per estimator; `saans-locate` a
      READER of scenes (a `boundary.rs` edit); additive changes to
      `wifisense/localize/measure.py` and `track.py`, a package SAANS_PLAN
      §3.1 calls "Unchanged API".
    - *RTI on RSSI does not localise in the bedroom or the office* and
      false-alarms on 17.6% of the hall's empty-room ticks (spec §13, SIM).
      ADR-008 d9's "both or neither" would then hide the baseline whenever
      RTI abstains; Phase 5 must decide how the demo treats that.
      (Decided 2026-10-24: RTI is never drawn without the baseline; the
      baseline may be drawn alone, with RTI's abstention shown.)
    - *The baseline's radius is room-sized* (3.4–6.3 m, SIM) and very
      conservative: no estimate's error exceeds 0.43–0.69 of its radius,
      so a Phase 5 calibration to 95% coverage would set
      `wlm_radius_scale` near 0.4–0.6 (median radius about 1.3–3.4 m).
      Its error (median 0.7–1.7 m on walking ticks) is its real limit.
    - *Zones* (ADR-008 d2, d8) still have no type; the measurements are in
      metres. Adding them is a public-API change to `saans-scene`.
    - *FeatureFrame input* (ADR-008 d9's runtime) is Phase 5's: the
      estimators take any per-link `y`.
    - *The classic ESP32 cannot be an FTM responder* (ESP32-S2/S3/C3/C6
      can), so phone RTT to classic nodes is SIM-only in principle, not
      only by ADR-008 d16.
    - `edge/dsp/bench/run_bench.sh` and `run_size_report.sh` still print
      `python3 spec/golden/generate.py`, which `generate.py` now refuses
      (it runs only on CPython 3.14); `edge/` was left untouched.

13. **Phase 5 decisions and findings for the owner.**
    - *Decisions* (spec/livedemo.md *Decisions taken here*):
      - the room stream checks `Origin`, and on an auth-off gateway a
        loopback `Host`;
      - one thread per session, at most 4;
      - frames are never dropped, and a slow client slows only its own
        session;
      - an empty-room run per SIM session, so coverage always sits beside
        the false-alarm rate;
      - the variants are named `declared`/`with_walker`/`empty`, and the
        page defaults to `with_walker`, because the shipped bedroom and
        office people are still;
      - `can_see_position(Some(_))` is false until a per-space opt-in
        exists;
      - the pcap `--replay` stays unlabelled;
      - `opt-level = 2` in the dev profile for the five simulation
        crates.
    - *The session cap of 4 is not measured on the target gateway.* On
      this Xeon a hall session at 8x costs about 59 ms of every 100 ms in
      release, so 4 sessions cost about 2.4 cores. A Raspberry Pi 5 needs
      measuring before the cap is relied on; a slower host shows up as
      `effective_speed` below the requested speed.
    - *Memory per session.* Each setup keeps its calibration frames so a
      restart and the empty-room run need no second calibration: 3.5 MB
      (bedroom), 5.8 MB (office), 8.1 MB (hall). Keeping only the int8
      pairs would cut the hall to about 1.2 MB.
    - *`SIM_REV`.* The page's SIM module is served under its own digest
      (`roomsim.rs`) so it can be cached immutably. Any edit to
      `roomsim-sim.js` must update `SIM_REV` and the import line in
      `roomsim.js`; the tests fail until both match.
    - *The radius scales are still uncalibrated.* `roomsim-eval` prints a
      suggested scale per estimator (the 95th percentile of error over
      radius, SIM) without applying it. Choosing and applying them is
      later work.
    - *What the demo shows honestly:* in `declared` most bedroom and
      office ticks are abstentions, and WLM's circle usually covers the
      whole room. The page says so in words ("covers the whole room"),
      not by shrinking it.

14. **Phase 6 decisions and findings for the owner.**
    - *Decisions* (spec/dataset.md *Decisions taken here*): I/Q stored
      instead of amplitude; a continuous stream with labels every 0.5 s
      instead of stored 2 s windows; the baseline and the empty-room
      calibration stored; stratified splits; `--no-impair` meaning the
      hardware impairments off with the noise kept; about 17 GB.
    - *The dataset assumes the planned all-`txrx` TDMA mesh*, links
      separated by transmitter. Today's firmware has one transmitter and
      no TDMA, and its datagram carries no transmitter id, so a model
      trained on this transfers to REAL only once the firmware carries one
      (a SIM->REAL item for Phases 9-11).
    - *The stored calibration is idealised* (same seed, same clutter and
      devices, no drift): any Phase 8 model that uses it is reported
      separately.
    - *Walkers pause.* Room Sim's default `pause_s` U[0, 20] s keeps a
      walking person moving only about a third of the time; the per-tick
      `walking` label says who moves. Shortening the pauses would change
      the data's distribution: an owner call if wanted.
    - *Not Phase 6, found on the way:* the workspace declares
      `rust-version = "1.80"`, but `saans-dsp` (`is_multiple_of`),
      `saans-cli/src/eval.rs` (`is_none_or`) and
      `saans-cli/tests/roomsim_export.rs` (`as_chunks`) already use newer
      APIs, so the declared MSRV is not true (CI builds on stable); and
      `roomsim_export::an_id_problem_is_reported_even_when_the_text_fails_later`
      fails whenever `TMPDIR` contains the user name ("Aman"), because it
      checks that stderr never contains it.

15. **Phase 7 decisions and findings for the owner.**
    - *Decisions* (spec/preprocess.md *Decisions taken here*): numpy in
      `ml/` with a stdlib reference and no Rust port; one record and one
      reader per source; gain and the phase fit over the 44 shared
      subcarriers (the other carried bins output with the same gain);
      phase off by default and output as (cos, sin), the slope never
      output; no resampling by default; unbiased spread with a 3-packet
      minimum; labels derived, never overridable.
    - *A recording from today's firmware is read as one link per receiver
      only with `single_tx=True`*: the datagram carries no transmitter id,
      so a multi-transmitter recording cannot be split, and the reader
      cannot tell.
    - *SIM-only regularities survive preprocessing*: reciprocal links are
      near-identical, `gain_db` is an invented AGC walk, SIM carries 44
      bins, and SIM packet times sit on an exact TDMA grid
      (spec/preprocess.md §11). Phase 8 keeps them out of the model's
      inputs where it can.

## Changing the simulator

Under ADR-002 a physics change starts in the spec and the Python
reference, then `spec/golden/` is regenerated, then Rust is made to match;
for Room Sim, `spec/roomsim.md` is the contract and both implementations
must agree bit for bit (RUN.md §4c). Two traps:

1. **Regenerate golden vectors on CPython 3.14, nothing older.**
   `generate.py` first deletes every vector, so the interpreter matters:
   on 3.14.8 a full regeneration reproduces every committed vector byte
   for byte (checked: 0 files changed). On 3.12 three
   `sim_geometric` files change by up to 1e-15 relative (3.14 also
   compensates complex `sum()`), and on this box's default 3.11.5
   24 files change (both measured the same way). Use
   `$(.venv/bin/uv python find 3.14) -I spec/golden/generate.py`, then
   confirm `git status spec/golden` shows only the sets you meant to add.
   (Running the *tests* on 3.12 is fine; it is regeneration that must be
   3.14.)
2. **The parity tests pin path order and summation order.** saans-sim
   reproduces CPython's compensated `sum()` bit-for-bit with a hand-written
   Neumaier sum, and sums explicit scatterers before random ones. Adding
   walls or objects as new path terms must keep the existing terms in the
   existing order, or `tests/parity.rs` moves at 1e-9.

## Environment gotchas

All in RUN.md §1, briefly: the GPU is **shared** (another user's job holds
~15 GB at 100% for weeks); **bf16 is 12× slower than fp16** on this Turing
card, so train in fp16; egress to `pypi.org` and `static.crates.io` resets
about one connection in three, so every fetch needs retries.
