# Room Sim — status and handoff

The working record for the Room Sim / ML / simulator-realism plan (twelve
phases: scene format → web demo → simulator upgrade → localisation →
live demo → dataset → model → real data → Sionna and calibration → export).
Read it with [RUN.md](RUN.md) (how to run everything on the GPU box) and
[ADR-008](docs/adr/008-declared-geometry-and-drawing.md) (what may be
drawn). Every number below comes from a run on this machine; simulator and
scene numbers are labelled as such.

**Branch:** `saans/scene-format` · **Last updated:** 2026-10-21

## Where it stands

| Phase | | Status |
|---|---|---|
| 0 | Environment | **Done.** Rust 1.99, venv with CUDA torch, sionna-rt; RUN.md. |
| 1 | Scene format | **Done.** `saans-scene` crate, JSON Schema, 3 example scenes, `saans import-roomplan`. |
| 2 | Static web demo | **Done.** |
| 3 | Simulator upgrade | **Done — waiting for the owner's review.** Room Sim v1: `spec/roomsim.md`, the Python reference `wifisense/roomsim/` and the Rust crate `saans-roomsim`, bit-identical; `saans roomsim-export`. Three decisions need the owner (*Open items* 7). |
| 4–7 | Localisation, live demo, dataset, preprocessing | Not started. |
| 8–12 | Model, real data, Sionna/calibration, recordings, export | Not started. **Blocked on an ADR** for learned models (see *Open items*). |

The working rule is to stop after each phase for review. That is why
nothing past Phase 3 has been started, not lack of time.

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
| `rust/crates/saans-gateway/src/roomsim.rs` + `assets/roomsim.*` | The `/roomsim` demo page: routes, its own stricter CSP, embedded scenes served without people or phone. |
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
| 2026-10-21 | `Shared::mode` becomes `Mode { Sim, Replay, Live }` (a public-API change to saans-gateway) — **for Phase 5, not yet done.** | Owner |
| 2026-10-21 | The phone is simulated only; a LIVE phone needs its own ADR. | Owner |
| 2026-10-21 | Static demo now; mode enum and scene-to-space binding left for Phase 5. | Owner |
| 2026-10-21 | `provenance` is a required scene field; the examples say `hand_written`. | Implements ADR-008 decision 4 |
| 2026-10-21 | three.js 0.186.1 vendored, served from the binary; demo CSP allows nothing inline. | Implements ADR-008 decision 12 |
| 2026-10-21 | Room Sim is a **new** crate `saans-roomsim` and package `wifisense/roomsim/`, not an extension of `saans-sim` as the brief says. Extending `saans-sim` changes its public types (`Scenario`, `Room`, `SimOptions`, `LinkFrame`, `Tick`) and moves the `sim_geometric` golden vectors the S3 detector evaluation depends on. | **Proposed — owner to confirm** (open item 7) |
| 2026-10-21 | `saans-scene` gains a `write` cargo feature: `Serialize`, `Scene::save`, `to_json_pretty` and the RoomPlan importer exist only with it. Only `saans-cli` enables it; `tests/boundary.rs` holds two allowlists (writers; readers such as `saans-roomsim`). A public-API change to `saans-scene`. | Implements ADR-008 decision 1; **owner to confirm the API change** |
| 2026-10-21 | `saans-cli` is both the scene writer and a SIM CSI producer (`export` already was; `roomsim-export` now too). No command writes a scene from CSI; nothing structural stops a future one. | **Proposed — owner to confirm**, or split `roomsim-export` into its own binary |
| 2026-10-21 | Scene fixes in files this work wrote: bedroom almirah centre x 3.5 → 3.3 (node 1 was inside it); hall "sitter" (7.0, 1.8) → (7.0, 0.6), onto the sofa; office occupants y 2.5 → 2.1, onto their chairs (they were frozen standing behind them). | Fixes found by the Phase 3 reviews |
| 2026-10-21 | ESP32 defaults follow the firmware where it says: `ltf_merge` on, smoothing **off** (`channel_filter_en = false`), 56 forwarded entries. TDMA (`slot_s` 12.5 ms) is the *planned* schedule, labelled SIM; today's one-transmitter firmware is a documented config (RUN.md §4c). | Spec §3.2, §10 |

## How it was verified

All on the GPU box. The rows above **Phase 3** are the Phase 2 record
(2026-10-21, commit `ce115fa`); the Phase 3 rows are current (2026-10-23).

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
| Whole suites | workspace `cargo test --locked`; full Python suite on 3.11.5 and 3.12.15; `make -C edge/dsp/test check` | Rust workspace 279 pass, gateway with `roomsim` 79 pass; Python **559** run (459 + 100 Room Sim): on 3.12.15 only the known UDP-buffer error, on 3.11.5 and 3.10.12 that plus the two known pre-3.12 failures — exactly the pre-Phase 3 baseline; edge PASS (g++) |

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

**GPU.** Phases 0–3 used none. The only GPU work so far was the fp16/bf16
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
   POST, with e2e cases for both.
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
    `replay` now print a SIM banner for one; the gateway's replay labels
    it only once Phase 5's `Mode::Sim` lands.
11. **What Phase 10 must calibrate.** Every SIM-labelled parameter in spec
    §3.2 is a guess: transmit power, the tail's power and decay, object
    scatter and blockage constants, AGC, ripple, chip noise and the noise
    floor. Known simplifications (spec §16) that calibration cannot fix:
    TE-only wall coefficients (up to about 7 dB on steep paths), no
    per-board RSSI offset, the Fresnel test for furniture faces ignoring
    obliquity, and exports without the FeatureFrames a raw-mode node also
    sends.

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
