# Cleanup audit: dead code (#33), parking (#34), doc drift (#35)

## Outcome (branch `saans/cleanup`)

| Commit | Issue | What landed |
|---|---|---|
| `61e3562` | #33 | Rows 1-5 removed (78 lines, including the `appended` field that only the removed accessor read) |
| `b38a5b5` | #34 | `/room-sim` is behind `roomsim` (D1 = park). README has a park table and research labels. No heart switch, because the gateway never exposes heart rate. No CLI gate (D2 = no) |
| `3fd5983` | #35 | SAANS_PLAN retention and the NEXT.md status header. The page count was fixed in `b38a5b5`. Test count, S3-only and the ROOMSIM.md phases were fixed on `saans/consolidate` |

The rest of this file is the audit as it was written before any edit,
including the first draft's mistakes and the review that caught them.

Audited 2026-11-06, read-only, against the code as it will look after the
research merge:

- `origin/main` @ `d5194b0`
- `origin/saans/scene-format` @ `d385c13`
- the consolidation merge commit `7ce4c9f` (local, not yet pushed) plus that
  session's later commits `4506efa`, `a02f256`, `82c5ed3` and its uncommitted
  edits to README.md, ROOMSIM.md and docs/EVIDENCE.md

Everything below gets re-verified against `origin/saans/consolidate` once it
is pushed, before any edit.

## How each claim was checked

| Check | Command or method | Result |
|---|---|---|
| Rust dead code and lints | `cargo clippy --workspace --all-targets [--all-features] --locked -- -W dead_code -W unused -W unused-crate-dependencies` on main, scene-format and `7ce4c9f` | **No dead_code or unused warnings in any tree.** `-D warnings` passes on `7ce4c9f`. |
| Rust unused dependencies | `-W unused-crate-dependencies` per target (cargo-machete and cargo-udeps are not installed) | None real. The only hits are false positives: dev-deps (`sha2`) seen from lib-test targets, `saans-scene`'s self dev-dep for its `write` feature, and the gateway bin vs lib split. |
| Rust `pub` items with no non-test caller | Text scan of `crates/*/src` (excluding `#[cfg(test)]` items and comments) against every src, test, example and bench | 58 flagged (~430 lines). 4 are dead; the rest are cross-language mirror accessors, port-parity accessors used by `tests/port_*.rs`, or `#[cfg(test)]` support. |
| C++ | `make -C edge/dsp/test test CXX=clang++ WARN="… -Wunused -Wunused-macros -Wunused-member-function -Wunused-template -Wunused-const-variable -Wunused-function"` | 10/10 binaries pass, **zero warnings.** Firmware (`edge/firmware/main`) needs ESP-IDF and was read by hand; nothing unused found. |
| Python | Stdlib AST scan: modules with no non-test importer, and top-level defs and methods referenced only by tests or not at all | 1 dead def. Every other flag is a mirror of the C++ or Rust API, or an entry-point script. |
| File-level dead weight | `ecc:refactor-cleaner` (read-only): grep of CI, Makefiles, docs, JS and HTML for every root file, example, script and asset directory | See tables. |

**Overall:** there is very little unreachable code. Each language's compiler
and linter is clean. The weight is duplication left by two lines of work
that grew apart: two scene formats, two 3-D viewers and two three.js copies.
There is also research code compiled into the default `saans` CLI.

Where the line count actually comes from (consolidated tree):

| Area | Lines |
|---|---|
| Gateway assets | 92,797, of which **82,289 are vendored, unminified three.js 0.186.1** (`assets/vendor/three-0.186.1/`, research `/roomsim` only) |
| Rust `src` | 55,303 |
| Rust `tests` | 29,431 |
| `edge/` | 19,251 |
| `ml/` | 17,959 |
| `wifisense/` | 16,609 |
| `tests/` | 15,030 |
| `docs/` | 6,663 |

Any "lines of code" figure for the pitch should leave the vendored
three.js out.

## #33 Dead code: classified candidates

| # | Candidate | Lang | Lines | Evidence | Class | Reason |
|---|---|---|---|---|---|---|
| 1 | `wifisense/dsp.py:277 _noise_calibrated_confidence` | Py | 19 | Only hit in the tree is its `def`. `band_peak` inlines the same maths. No C++ or Rust mirror. | **REMOVE** | Dead private helper left by a refactor. |
| 2 | `saans-locate/src/pynum.rs:305 py_sorted` | Rust | 5 | Zero callers. `py_sort` (:295) is the one in use. | **REMOVE** | Unused wrapper. |
| 3 | `saans-roomsim/src/motion.rs:181 segments_appended` | Rust | 3 | Zero hits. No Python counterpart. | **REMOVE** | Unused accessor, no mirror. |
| 4 | `saans-dsp/src/features.rs:643 processed_frames` | Rust | 3 | Zero hits. No Python or C++ counterpart. | **REMOVE** | Unused accessor, no mirror. |
| 5 | `saans-sim/examples/speed.rs` | Rust | 37 | Named only in its own header. Not in CI, a Makefile or any doc. | **REMOVE** | Ad-hoc timing loop. `saans-dsp/benches` and `edge/dsp/bench` do the real benchmarking. |
| 6 | `evaluate_localize.py` | Py | 314 | Cited only by `NOTES-localize.md:26`, `P0B_TASK.md:24,132`, `spec/locate.md:38` and docstrings (`localize/baseline.py:5`, `tests/test_localize_track.py:1852`). No test, no CI. Its usage text hard-codes the root path. | **MOVE-TO-RESEARCH: labelled, not moved** | The P0b sweep harness. Nothing on the MVP path imports it. Add it to README's Product vs research table. That table (from consolidation) is the boundary, and directories are not moved to match it. |
| 7 | `P0B_TASK.md` | Doc | 177 | Cited by `wifisense/localize/sim.py:143` and four test docstrings, as history. | **MOVE-TO-RESEARCH: labelled, not moved** | Brief for a finished phase. Moving it means editing the frozen oracle's comments for no gain. |
| 8 | `phase6-samples/*.png` (2) | Asset | 717 KB | `ROOMSIM.md:58`, `RUN.md:532` (both being edited by consolidation) | **MOVE-TO-RESEARCH: labelled, not moved** | Research sample plots. Revisit after consolidation lands. |
| 9 | `ml/` (all of it, including `ml/phase9/run.py`) | Py | ~9.5K | `RUN.md` runs it from the repo root on the GPU box. CI never runs `ml/tests`. | **KEEP in place** | Already a research-only top-level directory. Moving it breaks the GPU-box commands in RUN.md for no product gain. README's new Product vs research table (consolidation) names it research. |
| 10 | Main's scene stack: `saans-sim/src/scene.rs` (2,290), `wifisense/scene.py` (1,019), `wifisense/roomplan_import.py` (576), `tests/test_scene.py` (601), `tests/test_roomplan_import.py` (310), `scenes/scene.schema.json` (197) | Rust/Py | 4,993 | No crate imports `saans_sim::Scene` (only the `lib.rs:115` re-export). `wifisense.scene` is imported only by `roomplan_import.py` and the two tests. `generate.py` uses `wifisense.roomsim`, not this. `roomplan_import.py:554` says it "has never been run against a real scan". The `scenes/*.json` files are served raw by the site (`site.rs:228-255`) and parsed in the browser by `site/room-sim.js`. | **DECIDE (see D1)** | Unreachable as code, but it is the schema check for files the site serves. It goes when the site page goes or is ported. |
| 11 | Site `/room-sim` page: `assets/site/room-sim.{html,js}` (1,454) and `assets/site/vendor/three*.min.js` (0.185.1, 751 KB) | HTML/JS | 1,454 + 751 KB | Default build (`site.rs:76-77,158,196-214`). Linked from every site page's nav. Exported by `pages.yml`. | **PARK or REMOVE (see D1)** | On the #34 park list. It duplicates the research `/roomsim` demo, which has its own three.js 0.186.1 (2.16 MB, `assets/vendor/three-0.186.1/`, already behind `roomsim`). |
| 12 | Mirror and parity methods the scans flagged: `center_hz`, `effective_corner_hz`, `alpha_to_corner`, `ReplayGuard.{accept,to_json,from_json}`, `FeatureFrame.as_dict`, `Room.perimeter_m`, the `MultiLinkSimulator` accessors, 43 `saans-locate`/`saans-sim` items used only by `tests/port_*.rs`, examples or `saans-cli` tests | Py/Rust | ~400 | Each mirrors the same API in another port (`biquad.hpp:110`, `preprocess.hpp:423`, `frameauth.rs`, `geometry.rs:80`), or a test of record exercises it. | **KEEP** | Three-language spec: the Python oracle and the parity accessors are reachable by definition. |
| 13 | `roomstream/testkit.rs` (5 items, ~100 lines) | Rust | ~100 | `#[cfg(test)] mod testkit;` (`roomstream/mod.rs:42-43`) | **KEEP** | Unit-test support (false positive of the text scan). |
| 14 | `saans-sim/src/impair.rs:538 phase_ripple` | Rust | 3 | Used by the `impair.rs:897-915` unit tests. | **KEEP** | Tested helper. |
| 15 | `tools/results.py`, `spec/golden/{generate,check_drift,bench_python}.py`, `edge/dsp/bench/*`, `bench/`, `deploy/`, examples `fixture_night`, `impairment_cost`, `locate_measure`, `scan_seats`, `mqtt_watch` | mixed | — | CI (`ci.yml:184,221`), `Makefile:153,174`, `README.md:183,477`, `tools/results.py:134`, `RUN.md:379`, `docs/NEXT.md:402` | **KEEP** | CI or a documented command uses each. |
| 16 | Root docs: `NOTES.md`, `NOTES-localize.md`, `RESEARCH.md`, `CAMPUS_SENSING_PROMPT.md`, `RUN.md`, `ROOMSIM.md` | Doc | — | All cited live (`CONTRIBUTING.md:15`, `SAANS_PLAN.md:8`, `phone.py:6`, `.gitignore:32`). | **KEEP (location: optional move to `docs/research/`, owner call)** | Moving them is churn across many links. Ask before doing it. |
| 17 | `.github/workflows/tests.yml` duplicating `ci.yml` | YAML | ~85 | Both ran cargo test, unittest and `make check`. | **Done by consolidation** (`a02f256`) | Verify at Phase 2. |

**Estimated lines removed by this PR:** 67 for certain (rows 1-5). With D1
taken as "remove", add about 6,450 lines (rows 10-11) and 751 KB of
minified three.js. Rows 6-8 are labelled research in README, not moved.

## #34 Park non-MVP features: what the code does today

| Feature | Where | Default today | Gap | Phase 2 action |
|---|---|---|---|---|
| Heart rate | Edge `SAANS_HEART_ENABLED` (`Kconfig.projbuild:156`, `default n`, `depends on SAANS_VITALS_ENABLED`); `static_assert` in `saans_config.h:87`; raw-CSI path `heart_enabled: false` (`engine.rs:357`) | **Off** at the node. Even with it on, nothing leaves the gateway: `fusion.rs:235` fills `SpaceSample.heart`, but `SpaceRt` (`engine.rs:118`) is not serialized. `history.rs:155` records only breathing, and the API's `SpaceState` has no heart field. The dashboard never renders it. | None (an earlier draft assumed a leak; review disproved it). | README row only. No code change. |
| Fall/collapse detector | `collapse_detection` per space (`saans.example.toml:85`) | **Off.** `config.rs` test `defaults_are_conservative` asserts it. The dashboard badge shows only when enabled (`dashboard.html:781`). | None. | README row only. PR #31 (hold presence after a fall-shaped motion, opt-in, off) is separate and open against main. It needs a rebase onto the merged main. |
| RTI positioning | `wifisense/localize` (Python oracle for `saans-locate` golden vectors), `saans-locate`, `evaluate_localize.py` | Gateway: **off** (only under the `roomsim` feature). CLI: `import-roomplan`, `roomsim-export`, `roomsim-eval` and `gen-dataset` are always compiled (`saans-cli/Cargo.toml:16-22`). | The CLI links research crates, but the GPU box depends on that. `ml/calibrate/simulate.py:38,138` shells out to `saans roomsim-export`, and RUN.md uses all four subcommands after a plain release build. | README row: positioning is research, not in the gateway's default build. **No CLI gate in this PR** (see D2). `wifisense/localize` stays as the oracle for the golden vectors. |
| Headcount | No product code. Site copy only (`hardware.html:110` "Classroom headcount", tagged "T1 · research"). `gen-dataset` labels counts. | Nothing to switch. | None in code. | README row: research, no implementation. Elder-care repositioning of the site copy is the #37 spec session's call. |
| 3-D room page | (a) `/roomsim` demo (three.js 0.186.1) behind `roomsim`. (b) Site `/room-sim` page (three.js 0.185.1) in the **default** build. The nav link is hard-coded in all 8 `assets/site/*.html` pages. | (a) **off**, (b) **on** | (b) | Per D1. Gating (b) means: remove the nav link from the 8 pages; split `site::PAGES` and `ASSETS` into a core part and a `#[cfg(feature = "roomsim")]` part; keep `internal_links_resolve` (`site.rs:644`) green in both builds; fix the counts at `site.rs:738`. |

MVP signals that stay on, confirmed in code:
- occupancy and still person (fusion and `SpaceMonitor`, always on);
- breathing rate (`vitals = false` default, opt-in per space, `saans.example.toml:82`);
- inactivity (`inactivity_alert_min = 30`, on by default).

The long-lie hold (PR #31) is opt-in and not on main yet.

## #35 Doc drift: one value each, with its source

| Drift | Says | Truth and source | Who fixes |
|---|---|---|---|
| Python test count | `README.md:466` 550 (242 non-localize); `docs/EVIDENCE.md:146` 547 (308 localize); `COMPLETION_REPORT.md:186` and `RUN.md:123` 459 | 829 test cases after the merge (**Ran 829 tests, OK** in 568 s on `7ce4c9f`; `unittest.defaultTestLoader.discover('tests').countTestCases()` on `7ce4c9f`; consolidation's README edit says the same). `origin/main` alone has 593, so both 550 and 547 were already stale before the merge. | **Consolidation**: README and EVIDENCE.md edits in progress. Phase 2: verify; `RUN.md:123` is dated "as of a9c5abb", so leave it. |
| ESP32-S3 vs C6 | `README.md:431` "S3 or C6"; `NOTES.md:182` "S3 or C6" | S3 only: `edge/firmware/sdkconfig.defaults:4`, `edge/build.sh:30`, `docs/HARDWARE.md:126` | README: **consolidation** (done in their working tree). `NOTES.md:182` is a dated research note; add a one-line pointer to HARDWARE.md §2 in Phase 2. RESEARCH.md's S3/C6 lines are literature and stay. |
| Retention | `SAANS_PLAN.md:202-203` "events 30 days raw, then 15-minute aggregates" | Events 30 d, daily summaries in `history.json` 90 d, audit 365 d (never less than events), alerts until resolved: `config.rs:234-236`, `README.md:284-287`, ADR-010. Nothing makes 15-minute aggregates. | **Phase 2** (#35), `SAANS_PLAN.md`. |
| NEXT.md status | `docs/NEXT.md:3-4` "A plan, not a record: nothing below is done" | Every §1 item already says **Done** in its own body: §1.1 → #6, §1.2 → #7, §1.3 → #8 (part a), §1.4 → #9 and #24 (done as a measurement, nothing shipped), §1.5 → #14, §1.6 → #12, §1.7 → #11, §1.8 → #13 (0 `todo!()` left in `alerts.rs`), §1.9 → #15 and #16, §1.10 → #17, §1.11 → #19-#21. Only §2 (boards) is open. Only the header is stale. | **Phase 2**: rewrite the header to say so, with the PR map. The body is left as written. |
| Site page count | `site.rs:33`, `http.rs:46` and ADR-008-one-origin body say "seven"; `README.md:468` says "eight" | `site::PAGES` has 8 today. After #34: 7 by default, 8 with `roomsim` (or 7 if removed). | **Phase 2**: code comments stop naming a number ("every page in `PAGES`"); README states the default and the feature. The ADR body stays (its addendum records the change). |
| ROOMSIM.md phases | "8-12 Not started. Blocked on an ADR"; "Branch: saans/scene-format" | ADR-014 (was 009, `207df45`); Phase 8 spec and trainer `62979fe`, `d385c13`; Phase 9 tooling `b8f29d6`; Phase 10b calibration run `e9b01d1`, `d385c13` (parameters at bounds, `ml/results/phase10/calibration.json`); Phase 11/12 prep `b8f29d6`. Trainer integrity fixes are on unpushed `claude/sweep-guards-aea624`. | **Consolidation** (phase table edit in progress). Phase 2: verify against this list. |

Found along the way (not on the issue):
- `site/room-sim.html:177` says the scene files are "compiled into the
  simulator". Only `#[cfg(test)]` code reads them
  (`saans-sim/src/scene.rs:1754`). Fix with D1.
- `scenes/bedroom.json` and `spec/scenes/bedroom.json` are different files
  for the same room (different schemas).
- Duplicate ADR numbers 008 and 009 in `docs/adr/`: consolidation is
  renaming them to 013 and 014.
- EVIDENCE §3.3 and NOTES §8.2 say 99.9% (2-link, within 1 BPM), where RESULTS says 100.0%. The #37 session found this,
  and it can ride with #35 if wanted.
- `hardware.html` use cases are campus-first. This is the elder-care
  positioning, so it belongs to #37, not here.

## Risks

- **Open PR #31** (`saans/hold-after-fall`) touches `saans-gateway/src/{config,privacy}.rs`, `saans.example.toml`, `assets/dashboard.html` and `saans-health`. This plan no longer edits any of them (the heart switch is dropped), so there is no conflict. The README park table should name #31 as the long-lie path.
- **The consolidation branch is still moving.** Every row is re-checked
  against pushed `origin/saans/consolidate` before editing, and the drift
  rows marked "consolidation" are verified rather than redone.
- **Parking `/room-sim` changes the public site.** `pages.yml` exports the
  default build, so the Pages site drops to 7 pages unless the workflow
  builds with the feature. The site tests check that every nav link
  resolves, so the nav link must be removed in the same commit.
- **PR #52 (elder-care spec, #37)** lists follow-up edits to README, SAANS_PLAN, NEXT.md and others in its body.
  #35 touches SAANS_PLAN retention (§8 area) and the NEXT.md header. Keep those edits to the lines named here,
  so whichever lands second rebases a small hunk.
- **Line counts in docs.** Python counts change again if D1 removes `tests/test_scene.py` and `tests/test_roomplan_import.py`. Count last, on the final tree.
- **Files consolidation is editing.** README.md, ROOMSIM.md and docs/EVIDENCE.md are still open in consolidation. #35 edits there wait for its push and are kept to the rows it did not fix.
- **No ML/torch work runs locally.** Nothing in this plan touches `ml/`
  beyond reading it.

## Decisions for the owner

- **D1. The site's `/room-sim` page and main's scene stack.**
  - **(a) PARK (recommended; it matches #34's "keep the code, default
    off").** Put the page and its assets behind the existing `roomsim`
    feature and remove its nav link from the 8 site pages. The default
    build and the public Pages site become 7 pages, and nothing is
    deleted. This is about 8 HTML one-line edits plus a `site.rs` split.
    A follow-up issue unifies the two scene formats and the two viewers.
  - **(b) REMOVE.** Delete the page, main's scene stack and the 0.185.1
    vendor copy now: about 6,450 lines and 751 KB. The public site loses
    the 3-D page, and the `/roomsim` demo (feature build) remains the
    only 3-D view.
  - Either way, should `pages.yml` export the site with `--features roomsim`
    to keep the 3-D page public?
- **D2.** Gate the `saans-cli` research subcommands behind a `research`
  feature? Recommended: **not in this PR.** The GPU box's calibration
  script and about ten RUN.md commands call these subcommands on a plain
  release build. The gate would also need clippy and test runs in `ci.yml`
  and a doc sweep, and it saves compile time only. Make it a follow-up
  issue if wanted.
- **D3.** Move root research docs (`NOTES*.md`, `RESEARCH.md`,
  `CAMPUS_SENSING_PROMPT.md`, `ROOMSIM.md`, `RUN.md`) to `docs/research/`?
  Recommended: not in this PR. It is link churn with no product effect.

## Review of this plan

The `ecc:code-reviewer` agent reviewed the first draft. Its verdict was "do not
approve as written". Its three HIGH findings were checked in the code, all
three held, and the plan above now reflects them:

1. **The heart switch was based on a leak that doesn't exist.**
   `SpaceRt.sample` is never serialized, so the switch is dropped.
2. **A CLI `research` gate would break the GPU box.**
   `ml/calibrate/simulate.py:138` and RUN.md depend on those subcommands,
   so the gate is deferred (D2).
3. **Gating `/room-sim` is a bigger change than the draft said.** It
   touches 8 HTML files, the `site.rs` consts and tests, and `pages.yml`;
   the cost is now spelled out under D1.

The MEDIUM finding was also adopted: moves without a destination churn
files that other sessions are editing. Rows 6-8 are now labelled rather
than moved.

## Phase 2: commits on `saans/cleanup` (base `saans/consolidate`)

1. `refactor: remove dead code found by the #33 sweep`. Rows 1-5, 67
   lines. Golden vectors regenerated only if `generate.py` changes, and it
   doesn't.
2. `docs: park non-MVP features in README (#34)`. A park table (heart,
   collapse, RTI positioning, headcount, 3-D views: what each is, why it is
   off, the switch and its default), the MVP signals that stay on, and
   research labels for rows 6-8. Then, per D1(a), `feat(gateway): serve
   the site's /room-sim page only with --features roomsim` as its own
   commit.
3. `docs: one stated value for each drift in #35`. `SAANS_PLAN.md`
   retention, the `docs/NEXT.md` header, and the "seven pages" comments
   in `site.rs` and `http.rs`. Any row consolidation left undone is
   re-checked.
4. This file, as `docs/CLEANUP_AUDIT.md`.

Every suite runs before the push:
- `python3 -m unittest discover -s tests`;
- `cargo test --workspace --locked`, plus `-p saans-gateway --features roomsim` and `-p saans-locate --features roomsim`;
- `cargo clippy --workspace --all-targets --locked -- -D warnings`, with and without `roomsim`;
- `make -C edge/dsp/test check`;
- `git grep` for third-party names, which must come back empty.

## Owner decisions (2026-11-06)

- **D1: park.** The site's `/room-sim` page goes behind the `roomsim`
  feature: the code stays, and the feature is off by default.
  - The public site builds without the feature, so it has 7 pages.
  - Update the nav in every page, plus `site.rs` and its tests.
  - Fix every "seven pages" or "eight pages" reference.
  - Main's scene-format stack stays for now, labelled research in the
    README table.
- **D2: no.** The CLI research subcommands stay ungated in this PR.
- The REMOVE list (67 lines) and the #35 drift fixes go ahead as planned.
