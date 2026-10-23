# Completion report — observatory pass

Branch `saans/observatory`. Everything below was run; where something
was not run, it says so.

## 1. What was missing

Four bugs, all the same shape: **configuration the validator accepts
and the runtime cannot honour.**

### 1.1 The API had no authentication (critical)

`auth.rs` and `audit.rs` were two lines of doc comment each. `roles.rs`
held a complete, tested permission model — `Principal::can_see_vitals`,
`can_see_health_alerts`, `can_see_system`, `can_see_space` — with **no
caller anywhere in the tree**. `http.rs::router` applied no auth layer.

Meanwhile `config.rs::validate_outputs` validated the auth
configuration in full: it rejects `auth.required = false` on a
non-loopback `http_addr`, rejects a non-loopback address with no
`[[user]]` or `[[token]]`, requires Argon2 PHC strings, and requires
64 hex digits for a token hash. The config file was the only thing
saying the API was protected.

Reproduced before the fix, with `http_addr = "0.0.0.0:8080"`,
`auth.required = true` and a token configured — `--check` printed `OK`,
the gateway started, and with **no credential at all**:

```
/api/spaces   200      (including a vitals = true space)
/api/health   200
/api/links    200
/metrics      200
/             200
```

### 1.2 `[mqtt]` panicked at startup

`mqtt::spawn` is `todo!()`; `live.rs:115` calls it whenever `[mqtt]` is
configured. `--check` said `mqtt.toml: OK (1 spaces)`, then:

```
thread 'main' panicked at crates/saans-gateway/src/mqtt.rs:11:5:
not yet implemented: outputs agent
```

### 1.3 `[[alert_route]]` panicked at startup

`AlertService::new` is `todo!()`; `live.rs:90` calls it whenever a route
exists. Same shape.

### 1.4 `--hash-password` did not exist

`config.rs` rejects a plaintext password with a message telling the
operator to run `saans-gateway --hash-password`. There was no such flag.

### 1.5 UI gaps

No per-space detail (`GET /api/spaces/{id}` existed and nothing called
it); no time-series anywhere; no event search or time filter; no sensor
visualisation; mode conflated with connection state; light-first theme.

## 2. What was implemented

### Backend — no new dependencies

`argon2`, `sha2`, `hmac` and `getrandom` were already declared in
`Cargo.toml` for the auth layer that was never written.

- **`src/auth.rs`** (new, 323 lines). Bearer tokens for machines
  (SHA-256, constant-time compare); Argon2id password login with an
  in-memory session cookie for people (`HttpOnly`, `SameSite=Strict`,
  `Secure` behind TLS). A wrong password and an unknown user cost the
  same and answer the same. Sessions end on restart.
- **`src/audit.rs`** (new, 213 lines). `audit-YYYY-MM-DD.jsonl`:
  logins, failures, logouts, denials, and reads of consented vitals
  throttled to once an hour per principal per space. Write failures are
  counted and surfaced, never fatal.
- **`src/history.rs`** (new, 383 lines). A bounded in-memory time
  series. Sized at startup from `window_s / cadence_s`; never
  persisted; columnar JSON; stride-decimated on read and never
  interpolated; `occupied` stays three-valued.
- **`src/privacy.rs`** — `space_for` / `event_for`: the second lock.
  Consent decides what may exist, the principal decides who may see it.
- **`src/http.rs`** — rewritten. Every data route resolves a principal
  first. `/api/links`, `/api/health` and `/metrics` need admin or ops.
  An out-of-scope space answers 404, not 403. The WebSocket handshake
  is authenticated before the upgrade and each client's stream is
  filtered for them, with the pre-serialized fast path kept for an
  unrestricted admin.
- **`src/config.rs`** — `[mqtt]` and `[[alert_route]]` are now rejected
  at config load, so `--check` catches them.
- **`src/main.rs`** — `--hash-password`, reading from stdin so the
  password never reaches shell history or the process list.

### Dashboard

`assets/dashboard.html` replaced (453 → 1,312 lines). Five views:
Overview, Signals, Topology, Events, System, plus a per-space detail
modal and a sign-in screen. Dark-first. Still one self-contained file
with no external fetch and no build step; charts are hand-drawn SVG.

Two defects found by driving it rather than reading it:

- The tab strip was rebuilt on every state message and every 2 s tick,
  throwing keyboard focus out of the nav twice a second and dropping
  clicks that landed mid-rebuild. Built once now; only the selected
  attributes change. The topology SVG had the same problem and the same
  fix, keyed on link state.
- Topology was drawn once on tab entry and never refreshed.

One honesty defect, found with an occupancy-role login: the card read
"Breathing: not confidently detected" for a value that had been
*withheld from that role*. The consent gate deletes vitals keys rather
than nulling them, so the two cases are distinguishable and are now
distinguished.

## 3. Files

**Created:** `src/history.rs`, `docs/UI_GAP_ANALYSIS.md`,
`docs/UI_IMPLEMENTATION_PLAN.md`, `docs/COMPLETION_REPORT.md`.

**Rewritten:** `src/auth.rs`, `src/audit.rs`, `src/http.rs`,
`assets/dashboard.html`.

**Modified:** `src/privacy.rs`, `src/config.rs`, `src/shared.rs`,
`src/hub.rs`, `src/live.rs`, `src/main.rs`, `src/lib.rs`,
`tests/e2e.rs`, `saans.example.toml`, `README.md`.

**Also in this branch** (earlier edge work): `edge/firmware/main/rx_role.cpp`
halts loudly instead of running inert on a misconfigured extractor;
`.gitignore` gained `edge/dsp/bench/build/` and two compiled binaries
were untracked.

## 4. API changes

**New:** `GET /api/me`, `POST /api/login`, `POST /api/logout`,
`GET /api/history`.

**Unchanged in shape:** `/api/spaces`, `/api/spaces/{id}`,
`/api/events`, `/api/links`, `/api/health`, `/metrics`, `/api/live`.
No field was renamed, removed or reinterpreted. `/api/health` gained
`auth`, `audit` and `series` sub-objects; the existing `history` key
still means the persisted daily summaries, and the new in-memory series
was named `series` precisely so it could not collide.

**Behaviour change:** these routes now return 401 without a credential
and 403 for an insufficient role, where before they returned 200 to
anyone. A deployment on a loopback `http_addr` is unaffected — auth is
off there and every request is a local admin.

**New config:** `[history] window_s, cadence_s`. `[mqtt]` and
`[[alert_route]]` are now rejected.

## 5. Commands

```bash
cd rust
cargo build --release -p saans-gateway -p saans-cli
cargo test --workspace

./target/release/saans-gateway --config saans.toml --check
./target/release/saans-gateway --config saans.toml
./target/release/saans-gateway --hash-password

# no hardware needed
./target/release/saans export night --out night.pcap --layout 2
./target/release/saans-gateway --config saans.toml --replay night.pcap --speed 25 --serve

cd ..
python3 -m unittest discover -s tests
make -C edge/dsp/test check
```

## 6. Tests actually run

| Suite | Result |
|---|---|
| `cargo test --workspace` | **141 passed, 0 failed** (113 before this pass) |
| — `saans-gateway` lib | 55 passed (27 before): +7 auth, +5 privacy roles, +9 history, +3 audit, +4 http (live filtering, login cap) |
| — `saans-gateway` e2e | 11 passed, unchanged |
| — `saans-dsp` golden vectors | 12 passed, unchanged |
| — `saans-health` | 36 passed, unchanged |
| — `saans-sim` | 21 passed, unchanged |
| `make -C edge/dsp/test check` | **9 binaries, 845 checks, g++ and clang++, PASS**; missing-vector guard passes for each |
| `python3 -m unittest discover -s tests` | **459 passed, 0 failed** (259 s), unchanged by this work |
| `saans-gateway --config saans.example.toml --check` | OK (2 spaces) |

There is **no CI in this repository** — no `.github/workflows` — so a
pull request shows no checks. That is an absence, not a pass.

> *Amended 2026-10-24:* true when written. `ci.yml` landed in `101639b`
> and runs fmt, clippy, the Rust and Python suites, the C++ suite under
> both compilers, and a numeric golden-drift check, on every push to
> `main` and every pull request.

**Manual verification** (no browser-test framework exists in this tree,
so this is weaker than a suite and is recorded as such). Against a
replayed synthetic night:

- All five views rendered with real data: occupancy band 89% / 3% / 8%,
  motion and presence, breathing ~12 BPM with its confidence, CSI rate
  19.8 Hz, loss 1%, BNR, RSSI.
- Dark and light.
- 375 px and desktop; **no horizontal overflow on any view** at 375 px,
  checked by comparing `scrollWidth` to `clientWidth`.
- Anonymous → sign-in screen; every data route 401 including the
  WebSocket upgrade.
- Occupancy role → System tab absent, breathing shown as "not shown to
  your role", `/api/health` 403, out-of-scope space 404.
- Safety role → breathing visible where consented, `/api/health` 403.
- Audit log contained the logins, the denials and the safety role's
  `vitals_read`, and not the occupancy role's.
- **Login flood:** 20 concurrent wrong-password attempts → 4 processed,
  16 refused with 429, and `/api/me` answered in under a millisecond
  throughout. Argon2 verification and the fixed failure delay are both
  blocking and this runtime has two worker threads, so running them
  inline let a handful of sign-in attempts stall ingest; they now run
  on the blocking pool behind a four-at-a-time cap.

## 7. Remaining gaps

See `UI_GAP_ANALYSIS.md` §3 for the full list. The ones that matter:

- **Alert acknowledge / resolve / escalate** — `AlertService` is a stub.
  No controls were added, because a button that does nothing is worse
  than no button.
- **MQTT** — still a stub, now refused at config load.
- **Browser end-to-end tests** — none exist; the above was manual.
- **Audit log retention** — not swept. Audit records usually outlive
  the events they describe, but "forever" is not a policy either.
- **Room geometry, floor plan, 3D** — blocked on data that does not
  exist, not on effort.

## 8. Hardware-dependent limits

Unchanged by this work and still true: one link per RX node (multi-link
is phase S6); Fresnel blind spots are real and the mitigation is more
links, not more cleverness; calibration needs an empty room for ~80 s
including filter warm-up; heart rate is the weak output and is off by
default; the system is advisory and not a medical device.

## 9. The next three most valuable tasks

1. **Browser smoke tests.** Everything in §6's manual list should be a
   test. Without them the next change to the dashboard is unverified,
   and this pass found three defects that only appeared when driven.
2. **Finish or delete the two stubs.** `AlertService` and `mqtt::spawn`
   are now refused at config load, which is honest but not finished.
   Alert acknowledgement is the one that unlocks UI: it needs backend
   state, authorisation, persistence and audit before a button appears.
3. **Space geometry, if a floor plan is actually wanted.** It needs a
   config schema for coordinates, a survey of each room, and an honest
   way to draw what is *not* covered — which is the hard part, and the
   reason not to start with the drawing.
