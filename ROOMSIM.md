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
| 2 | Static web demo | **Done — waiting for the owner's look** ("show me before continuing"). |
| 3 | Simulator upgrade | Not started. Needs the owner's go-ahead; see *Before Phase 3*. |
| 4–7 | Localisation, live demo, dataset, preprocessing | Not started. |
| 8–12 | Model, real data, Sionna/calibration, recordings, export | Not started. **Blocked on an ADR** for learned models (see *Open items*). |

The working rule is to stop after each phase for review. That is why
nothing past Phase 2 has been started, not lack of time.

## What exists

| Path | What |
|---|---|
| `rust/crates/saans-scene/` | Scene types, validation (every error, not the first), plane geometry, Apple RoomPlan importer. |
| `spec/scene.schema.json`, `spec/scenes/*.json` | The contract and three worked examples (bedroom 4.2×3.6, office 6×5, L-shaped hall 9×7), all `provenance: hand_written`. |
| `rust/crates/saans-cli` → `saans import-roomplan` | RoomPlan scan → scene.json, printing every assumption it had to make. |
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

## How it was verified

All on the GPU box, 2026-10-21, at the commits on this branch.

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

**GPU.** Phases 0–2 used none. The only GPU work so far was the fp16/bf16
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
   routes; zones (decision 2), which have no type in the runtime yet; and
   the `write` feature of decision 1, which today is replaced by the
   stronger test that no CSI-carrying crate depends on `saans-scene` at
   all — that test fails, with instructions, the day it stops being true.
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

## Before Phase 3

Phase 3 changes the simulator's physics (walls with materials, objects,
second-order images, people with gait and shadowing, the ESP32 impairment
layer). Under ADR-002 a physics change starts in the Python reference,
then `spec/golden/` is regenerated, then Rust is ported to match. Two
traps found while getting here:

1. **Regenerate golden vectors on CPython ≥ 3.12, never on this box's
   default `python3` (3.11.5).** `generate.py` first deletes every vector,
   and on 3.11 float `sum()` rounds differently, so a regeneration would
   rewrite committed binaries with nonzero diffs even where no code
   changed. Use `$(.venv/bin/uv python find 3.12) spec/golden/generate.py`.
2. **The parity test pins path order and summation order.** saans-sim
   reproduces CPython's compensated `sum()` bit-for-bit with a hand-written
   Neumaier sum, and sums explicit scatterers before random ones. Adding
   walls or objects as new path terms must keep the existing terms in the
   existing order, or `tests/parity.rs` moves at 1e-9.

## Environment gotchas

All in RUN.md §1, briefly: the GPU is **shared** (another user's job holds
~15 GB at 100% for weeks); **bf16 is 12× slower than fp16** on this Turing
card, so train in fp16; egress to `pypi.org` and `static.crates.io` resets
about one connection in three, so every fetch needs retries.
