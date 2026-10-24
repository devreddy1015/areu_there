# UI gap analysis

What the dashboard could show, what the gateway actually publishes, and
where those two disagree. Written against the working tree on branch
`saans/observatory`, with every claim traceable to a file.

Status words mean exactly one thing each:

- **Implemented** — code exists and a test or a run exercises it.
- **Partial** — works for some cases, named below.
- **Missing** — nothing in the tree does it.
- **Blocked** — cannot be done honestly with the data that exists.

Dates and numbers come from runs recorded in `COMPLETION_REPORT.md`.

---

## 0. Method, and one deliberate omission

I inspected this repository: every Rust crate under `rust/crates`, the
embedded dashboard, the gateway's HTTP and WebSocket handlers, the
Python reference under `wifisense/`, the firmware under `edge/`, the
ADRs, and the test suites. Everything in the tables below was read, and
in most cases run.

**I did not re-read the reference project's source for this pass.** The
DSP and firmware in this tree are a clean-room implementation whose only
inherited parts are declared in [`/NOTICE`](../NOTICE); re-reading the
reference while writing new code in the same repository puts that at
risk for no gain, because every UI decision here has to follow from
Saans's own data contract anyway. The comparison in §2 is therefore at
the level of *product surface* — what a reference observatory shows an
operator — and not a source comparison. Where the brief asked for a
judgement about the reference's internals, the honest answer is that I
did not look, and the matrix says so.

---

## 1. Capability audit

### 1.1 Shell and data provenance

| Capability | Status | Evidence | Gap | Priority |
|---|---|---|---|---|
| Dashboard served, no external fetch | **Implemented** | `src/http.rs::dashboard`, CSP header, `http::tests::dashboard_has_no_external_urls`, `e2e::dashboard_is_served_and_self_contained` | — | — |
| LIVE vs REPLAY distinguishable | **Implemented** | `shared.mode` set in `live.rs` / `main.rs::run_replay`; surfaced in `hello`, `/api/me`, `/api/health` | — | — |
| Mode separated from connection state | **Implemented** (was Missing) | `dashboard.html::renderShell` — two independent pills | Previously one "live" indicator conflated socket health with data validity | P0 |
| Data freshness visible | **Implemented** (was Partial) | `renderShell` freshness pill; `S.lastState` | The old page detected staleness but only by overwriting the connection label | P0 |
| Version shown | **Implemented** | `/api/me`, `hello` | — | — |
| SYNTHETIC label | **N/A** | No synthetic *live* path exists: `saans-sim` output is exported to a pcap and consumed as `replay` (`saans-cli/src/main.rs::export`) | Nothing to label; REPLAY is the truthful word | — |

**Acceptance:** start a replay with `--serve`, confirm the REPLAY pill;
kill the gateway, confirm the connection pill goes red while the mode
pill does not change. Done.

### 1.2 Spaces

| Capability | Status | Evidence | Gap | Priority |
|---|---|---|---|---|
| Space grid with state, motion, links | **Implemented** | `hub.rs::space_json` → `dashboard.html::spaceCard` | — | — |
| "Cannot see" distinct from "empty" | **Implemented** | `SpaceSample.occupied: Option<bool>` (`saans-health/src/types.rs`); card `data-s="blind"`, separate KPI tile | — | P0 |
| Privacy badges preserved | **Implemented** | `space_json` emits `vitals`, `raw_mode`, `raw_mode_links`, `vitals_gated_links` | — | — |
| Per-space detail view | **Implemented** (was **Missing**) | `GET /api/spaces/{id}` existed in `http.rs` and **nothing called it**; now `dashboard.html::openDetail` | — | P1 |
| Withheld vs unmeasured distinguished | **Implemented** (was **Missing**) | `privacy::redact_value` deletes keys rather than nulling, so `'breathing' in s` separates the two cases | The card used to say "not confidently detected" for a value the role was not allowed — a false statement about the sensing | P0 |
| State duration | **Implemented** | `occupied_since_s`, `still_for_s` | — | — |
| Calibration freshness per space | **Partial** | `calib_age_h` is per **link** (`saans-health/src/system.rs::LinkReport`), not per space | A space-level "oldest calibration" roll-up would need a new field; the links table and detail view show it per link instead | P3 |

### 1.3 History and trends

| Capability | Status | Evidence | Gap | Priority |
|---|---|---|---|---|
| Time-series storage | **Implemented** (was **Missing**) | `src/history.rs`; before this there was only the 10,000-entry event ring (`shared.rs::EVENT_RING`) and the daily summaries in `history.json` | — | P1 |
| `/api/history` | **Implemented** (new) | `http.rs::history`, 9 unit tests in `history.rs` | — | P1 |
| Bounded and privacy-filtered | **Implemented** | Ring sized at startup from `window_s / cadence_s`, capped by `MAX_WINDOW_S`; breathing recorded only where the space consented and returned only to a permitted role | — | P0 |
| Charts from real timestamps | **Implemented** | `dashboard.html::chart`, `occupancyChart` | Nothing is interpolated: decimation strides, it never averages, so every plotted point was measured | P1 |
| Persisted history | **Deliberately not done** | — | A minute-by-minute motion trace is the kind of record DPDP retention limits exist for. In memory, the window *is* the retention policy and a restart is a delete | — |

### 1.4 Room and sensor visualisation

| Capability | Status | Evidence | Gap | Priority |
|---|---|---|---|---|
| Floor plan with room geometry | **Blocked** | `SpaceConfig` is `{id, name, links, vitals, presence_quorum, …}` (`saans-health/src/types.rs`) — **no coordinates anywhere in the gateway config** | Would require a new config schema, a survey of each room, and a way to show what is *not* covered. Until then any drawing is invented | P3 |
| Person position in a room | **Blocked** | `NOTES-localize.md`: on synthetic scenarios, radio-tomographic imaging did **not** beat a weighted-link-midpoint baseline at 4–12 nodes. No validated real-world position tracking exists | Not a UI gap. Do not render a position | — |
| Sensor/link topology | **Implemented** (was Missing) | `dashboard.html::renderTopology` from `/api/links` | Explicitly labelled not-to-scale and not a floor plan | P1 |
| Link quality on the diagram | **Implemented** | Edge colour and dash from `up`, `loss`, `calib_age_h`; `<title>` carries the numbers | — | — |
| 3D observatory | **Blocked** | Same reason as the floor plan: no coordinates, no validated localisation | A 3D room drawn from invented geometry would imply spatial capability this system does not have | P3 |

### 1.5 Events

| Capability | Status | Evidence | Gap | Priority |
|---|---|---|---|---|
| Event list with severity and evidence | **Implemented** | `shared.rs::query_events`, `HealthEvent.evidence` | — | — |
| Filter by space and severity | **Implemented** | was already there | — | — |
| Full-text search | **Implemented** (was Missing) | `dashboard.html::renderEvents` | — | P2 |
| Time-range filter | **Implemented** (was Missing) | client-side over `unix_ms`; `/api/events?since=` also exists and is used on reconnect | — | P2 |
| De-duplication and retention | **Implemented** | `evKey` in the client; `store.rs` deletes a day's file once every event in it is older than `event_retention_days` | — | — |
| Acknowledge / resolve / escalate | **Missing, and correctly absent from the UI** | `alerts.rs::AlertService::{ack,resolve}` are `todo!()` | Controls would be fake. Not added | P2 |

### 1.6 System health

| Capability | Status | Evidence | Gap | Priority |
|---|---|---|---|---|
| Gateway uptime, throughput, decode errors | **Implemented** | `hub.rs::health_json` | — | — |
| Per-link diagnostics | **Implemented** | `LinkReport`: `up`, `silent_s`, `rate_hz`, `loss`, `calib_age_h`, `best_bnr`, `rssi_dbm`, `noise_floor_dbm`, `raw_mode`, `frames` | — | — |
| Actionable warnings | **Implemented** (was Partial) | `renderSystem` — each note now says what happened, what shows it, and what to do | The old page listed the condition only | P1 |
| Historical trends for health | **Implemented** | per-link series in `/api/history`, charted under Signals | — | P2 |
| Prometheus metrics | **Implemented** | `hub.rs::metrics_text`; deliberately carries no vital signs (ADR-001) | — | — |
| Unknown/unassigned links surfaced | **Implemented** | `engine.rs::note_unknown` → `health_json.unknown_links` | — | — |

### 1.7 Access control — the serious one

| Capability | Status (before) | Status (now) | Evidence |
|---|---|---|---|
| Authentication | **Missing** | **Implemented** | `auth.rs` was two lines of doc comment. `config.rs::validate_outputs` validated `[auth]`, `[[user]]` and `[[token]]` in full and even refused `auth.required = false` on a non-loopback `http_addr` — but `http.rs::router` had no auth layer. A gateway on `0.0.0.0` with `auth.required = true` and a token configured passed `--check`, started, and served every space, every consented breathing value and the whole event ring to an unauthenticated caller. Reproduced: HTTP 200 on `/api/spaces`, `/api/health`, `/metrics`, `/`. |
| Role-based filtering | **Missing** | **Implemented** | `roles.rs` had the full `Principal` model with its own passing tests, and **no caller anywhere**. Now applied in `privacy::space_for` / `privacy::event_for` and on every route. |
| Per-space scope | **Missing** | **Implemented** | `Principal::can_see_space`; an out-of-scope space answers 404, not 403 |
| WebSocket authorisation | **Missing** | **Implemented** | handshake authenticated before upgrade; per-client filtering in `http.rs::filter_live` |
| Audit log | **Missing** | **Implemented** | `audit.rs` was two lines. Now `audit-YYYY-MM-DD.jsonl`: logins, failures, logouts, denials, and vitals reads throttled per principal per space per hour |
| `--hash-password` | **Missing** | **Implemented** | `config.rs` told operators to run a flag that did not exist |

### 1.8 Two more crashes of the same shape

| Issue | Evidence |
|---|---|
| `[mqtt]` panics at startup | `mqtt.rs::spawn` is `todo!()`, called unconditionally from `live.rs:115`. `--check` printed `OK`, then the process panicked. Now rejected at config load. |
| `[[alert_route]]` panics at startup | `alerts.rs::AlertService::new` is `todo!()`, called from `live.rs:90`. Same shape, same fix. |

Both were "configuration the validator accepts and the runtime cannot
honour". That is the pattern worth watching for in the two remaining
stubs.

---

## 2. Product-surface comparison

Read §0 first: this compares what an operator sees, not how either
project is built. "Reference" is the project named in the brief, as
described there and by its public demo.

| Area | Reference observatory | Saans now | Why the difference |
|---|---|---|---|
| Visual identity | Dark, immersive, demo-forward | Dark-first, engineering-forward, information-dense | Saans is read on a wall screen by someone deciding whether to walk to a room |
| Live vitals display | Prominent heart and breathing | Breathing with a confidence and an abstention; **heart rate off by default** | CSI amplitude is nonlinear in chest displacement, so breathing intermodulation sidebands can exceed the true cardiac peak (`NOTES.md`). Breathing is solid; heart rate is not |
| Subcarrier / CSI heatmap | Shown | **Not shown** | A node emits 68-byte FeatureFrames; raw CSI exists only in an explicit debug mode (ADR-001) and is flagged in every frame. FeatureFrame data must not be drawn as if it were subcarrier data |
| 3D room / person | Central | **Not built** | No coordinates in the config; localisation is a published negative result. See §1.4 |
| Signal charts | Live amplitude traces | Motion, presence, breathing, confidence, CSI rate, loss, BNR, RSSI — all from a bounded server-side history | Charts are only as honest as their source; the browser invents nothing |
| Room/sensor view | Room-shaped | Topology, labelled not-to-scale | §1.4 |
| Multi-tenancy / roles | Not a stated feature | admin / ops / safety / occupancy, scoped per space | Health data under DPDP; ADR-005 |
| Consent | Not a stated feature | Per space, opt-in, off by default, enforced on the node *and* at the gateway | ADR-001, ADR-005 |
| Offline deployment | Unknown | Required and tested | Runs on a box in a building that may have no route out |

**Adopted from the reference as a product idea:** a single operator view
that answers "what is happening right now" in a few seconds; treating
signal quality as something an operator should see rather than a
developer detail; and the value of an immersive view as a goal worth
building *toward*.

**Not adopted:** anything that would imply spatial or cardiac capability
this system has not demonstrated.

---

## 3. What is still missing

| Gap | Priority | Why it is not done |
|---|---|---|
| Alert acknowledge / resolve / escalate | P2 | `AlertService` is a stub. Needs backend state, authorisation, persistence and audit before any button appears *Closed 2026-10-24: built; Acknowledge / Resolve in the Events view.* |
| MQTT publishing | P2 | `mqtt::spawn` is a stub. Now refused at config load instead of panicking *Closed 2026-10-24: built; see README "MQTT and Home Assistant".* |
| Room geometry and coverage | P3 | Needs a config schema, a room survey, and an honest way to draw what is *not* covered |
| 3D observatory | P3 | Blocked on the above plus a localisation result that does not yet exist |
| Space-level calibration roll-up | P3 | Per-link today; a roll-up needs a new field in `space_json` |
| Browser end-to-end tests | P2 | No browser-test framework in the tree. The dashboard was driven manually across five views, two themes, three roles and two viewport widths; that is weaker than a suite and is recorded as such |
| Audit log retention | P3 | `events-*.jsonl` is swept on the DPDP clock; `audit-*.jsonl` is not swept at all yet, deliberately — audit records usually outlive the events they describe, but "forever" is not a policy either *Closed 2026-10-24: swept on `audit_retention_days`, ADR-010.* |

---

## 4. Hardware-dependent limits

None of the following can be closed by the dashboard, and the interface
should keep saying so:

- **One link per RX node** today; multi-link spaces are phase S6 (`edge/README.md`).
- **Fresnel blind spots are real.** A subject can sit where breathing is
  nearly invisible, and at 20 MHz the whole band sits at essentially the
  same Fresnel phase, so subcarrier diversity cannot rescue it
  (`RESEARCH.md` §5a). The mitigation is more links.
- **Calibration needs an empty room** for about 80 s including filter
  warm-up. Calibrate with someone present and the room reads as
  permanently empty — which looks exactly like a tuning problem and is
  not one.
- **Not a medical device.** Advisory, abstaining, never a replacement
  for existing safety procedures (ADR-005).
