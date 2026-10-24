# Contributing

Read this before the code. It covers the rules every change to Saans is
held to, why each one exists, and the commands that check them. It
applies to people and to coding agents alike. Most of it was learned
in this repository. Where a rule has a story, the commit or PR is named so
you can read it.

## Before you start

Read, in this order: [README.md](README.md),
[SAANS_PLAN.md](SAANS_PLAN.md) §1, §4 and §7,
[docs/CAPABILITY_LIMITS.md](docs/CAPABILITY_LIMITS.md),
[docs/COMPLETION_REPORT.md](docs/COMPLETION_REPORT.md) §7–9,
[NOTES.md](NOTES.md) §8–9, [NOTES-localize.md](NOTES-localize.md) §3 and
§7, and every decision record in [docs/adr/](docs/adr/README.md).
[docs/NEXT.md](docs/NEXT.md) is the plan: what is done, what is next, and
what waits for hardware.

## The rules

These six rules come from the project brief (2026-10-14) and bind every
change.

1. **Measured or silent.** Every new capability ships with an evaluation
   that prints a table, and every output carries a confidence and may
   abstain. Label synthetic results SYNTHETIC. Nothing in this repository
   has yet been measured on real hardware, so today that is every result.
2. **One spec, three implementations** ([ADR-002](docs/adr/002-one-spec-three-implementations.md)).
   Python (`wifisense/`) is the reference. Any DSP change updates Python
   first and regenerates `spec/golden/`. Then Rust must match it to 1e-9
   and the C++ float32 port to 1e-4. **Never loosen a tolerance to make a
   test pass.**
3. **Privacy and safety.** [ADR-001](docs/adr/001-edge-private-features.md):
   raw CSI never leaves the node except in flagged debug mode.
   [ADR-005](docs/adr/005-advisory-consented-abstaining.md): health
   outputs are advisory, consent-gated per space (off by default), and
   carry no identities. [ADR-007](docs/adr/007-no-learned-models-yet.md):
   no learned models in the product path until the DSP ceiling is measured
   on real data.
4. **Dependencies.** No new dependency, runtime or dev, without saying
   why in the PR description. The site and the dashboard fetch nothing
   remote. `site.rs` tests `pages_fetch_nothing_remote` and
   `assets_fetch_nothing_remote` enforce this.
5. **Config that validates must run.** If a feature is unfinished, the
   config loader rejects it with a message. It never panics at runtime,
   so no `todo!()` may be reachable. (`[mqtt]` once passed `--check` and
   then panicked on startup: COMPLETION_REPORT §1.2.)
6. **Capability claims follow [CAPABILITY_LIMITS](docs/CAPABILITY_LIMITS.md).**
   Do not build pose, through-wall imaging, RTI or metre-level tracking.
   Do not draw a person in any view unless their position comes from a
   labelled source (SIM).

## What the work since has added

- **A test must be able to fail.** Before trusting a new test, break the
  thing it guards once and watch it go red. Then restore the code and say
  in the PR what you broke. One leak check in the browser suite passed
  with consent wrongly switched on: the fixture's room was empty, so
  there was nothing to leak (#16).
- **A number in prose comes from a table that CI regenerates.** If README
  or NOTES quotes a result, register it in `tools/results.py`.
  `results.py --check` then fails when the prose and the table disagree.
  Every stale figure found in review had been typed from memory (#7).
- **A tolerance that fails on one platform is a finding.** On Ubuntu
  26.04, contraction into fused multiply-add broke bit-exact C++ tests,
  and a libm that does not round correctly moved golden vectors. The fixes
  were `-ffp-contract=off` and an array-scaled comparison. Neither was a
  wider tolerance (#10).
- **Secrets.** A secret never goes in the config file. The config names
  an environment variable instead, and startup checks that it is set. The
  value is read at the point of use. Never keep one in a struct that can
  reach a log, and never format such a type with `{:?}`. CodeQL's
  `rust/cleartext-logging` found exactly that once (#13). The one place a
  secret has to be held, the MQTT client's reconnect credentials, says
  so where it lives.
- **Every outbound path goes through `privacy.rs`.** Apply the consent
  gate first, then the role. A new output is filtered as an existing role
  sees the API, using the same functions.
- **Withheld is not absent.** A value hidden from your role is shown as
  "not shown to your role", never as "not detected". The browser suite
  fails if that changes.
- **Ask before changing an existing crate's public API.** Additive changes
  are fine; list them in the PR.
- **NOTICE stays.** The MIT licence of code the Python core was adapted
  from requires it. Keep NOTICE as the only place that attribution
  lives.
- **People.** No recording of a person who is not you before ethics
  committee approval
  ([HARDWARE.md, "Consent and ethics"](docs/HARDWARE.md)).

## Working method

Use small PRs, one task each. For each task:

1. Plan.
2. Write the tests first.
3. Implement.
4. Run every affected suite.
5. Update the relevant doc (NOTES.md, README.md, an ADR, docs/NEXT.md).
6. In the PR description, say what was run and what was not.

Stacked PRs are fine: CI runs on every pull request, whatever its base.
Commit subjects start with the area, in lower case:
`gateway: ...`, `eval: ...`, `ci, edge: ...`.

## Commands

| What | Command | CI job |
|---|---|---|
| Rust format, lints, tests | `cd rust && cargo fmt --check && cargo clippy --workspace --all-targets -- -D warnings && cargo test --workspace` | rust |
| Python reference | `python3 -m unittest discover -s tests` | python |
| C++ edge DSP, both compilers | `make -C edge/dsp/test check` | edge |
| Golden-vector drift | `python3 spec/golden/check_drift.py` | golden |
| Results tables vs the prose that quotes them | `python3 tools/results.py --check` | results |
| Dashboard in a browser | `cd rust/crates/saans-gateway/tests/browser && npm ci && npx playwright install chromium && npx playwright test` | browser |

The slow sweeps (`evaluate.py --suite nulls`, `saans eval`,
`saans links`) are measurements, not checks. Run them when you change
what they measure, and regenerate their tables with
`tools/results.py --slow`.

## Reporting a problem or a result

The issue templates ask first whether the data is **SYNTHETIC** (the
simulator, a fixture, `--simulate`) or **REAL** (a board in a room).
A result without that label cannot be read correctly, so the templates
require it.
