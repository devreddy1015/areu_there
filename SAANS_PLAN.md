# Saans — camera-free health and safety sensing

**Saans** (साँस, "breath") turns the WiFi signal already in a room into
health and safety signals, without cameras, wearables or microphones. It
uses an ESP32 sensor running C++, a Rust gateway, and the existing Python
`wifisense` package as the reference implementation.

This plan supersedes nothing in `CAMPUS_SENSING_PROMPT.md`; it is the build
plan for the system that prompt describes, re-centred on health monitoring.
The physics constraints there and in `RESEARCH.md` still bind.

---

## 1. Four design bets

| Bet | What Saans does |
|---|---|
| Where the radio data goes | **Stays on the sensor.** The ESP32 runs the DSP and sends 68-byte feature frames twice a second. Raw CSI leaves only in an explicit debug mode. |
| How correctness is kept | **One spec, three implementations.** Python is the reference; Rust and C++ must match it on golden vectors. |
| What it claims | **Measured or silent.** Each capability has an evaluation that prints a table; each output carries a confidence and may abstain. |
| How it beats blind spots | **Link diversity.** P0a showed subcarrier selection cannot escape a whole-band Fresnel null at 20 MHz; a second link from another angle usually isn't blind at the same time. |

## 2. What it monitors

All human-health outputs are **advisory**, carry a confidence, and are
**off unless the space has opted in** (`vitals = true` in its config).
None of this is a medical device (see §8).

### 2.1 People (per space)

| Signal | How | Status |
|---|---|---|
| Occupied / empty | Band-limited motion energy vs empty-room calibration, hysteresis | Built (Python), synthetic-validated |
| Still person present | Breathing band catches a seated reader PIR misses | Built, synthetic |
| Breathing rate | Goertzel band scan + noise-calibrated confidence | Built, synthetic |
| Breathing-rate variability | IQR of valid per-window rates over 5 min | New (S3) |
| **Breathing pause** | Presence held, no gross motion, breathing-band energy falls below a fraction of its own recent baseline for ≥ 10 s, after confident breathing | New (S3) |
| Fast / slow breathing | Sustained (≥ 2 min) rate outside 8–24 BPM at rest | New (S3) |
| Restlessness | Motion bouts per hour; fraction of time still | New (S3) |
| Rest period | Long still period with steady breathing (start, end, restlessness) — not sleep staging | New (S3) |
| **Lone-worker inactivity** | Present + no gross motion > N min; escalates if breathing is *not* detected | New (S3) |
| **Possible collapse** | Motion burst ≫ baseline, then stillness ≥ 30 s with presence held | New (S3), advisory only |
| Trend deviation | Resting breathing rate vs that space's robust baseline (median ± 3 MAD over ≥ 3 days) | New (S3) |
| Heart rate | Sideband search; **off by default**, unreliable without a reference | Built, not trusted |

### 2.2 The system itself

The plan's other meaning of "health": the sensing system must say when it
is not able to see.

| Signal | How |
|---|---|
| Node alive | Last frame time, uptime, reboot count |
| Link quality | Achieved rate, sequence loss, jitter, per-link breathing-band SNR |
| Calibration age / staleness | Age since last empty-room calibration; drift of the empty-room energy |
| Interference | Noise-floor and out-of-band energy jumps |
| Blind link | Link whose breathing confidence stays low while a neighbour's is high |
| Coverage | Per space: how many links are usable right now |

## 3. Architecture

```
 ESP32-S3 node (C++, ESP-IDF)                          one per link end
 ┌────────────────────────────────────────────────┐
 │ CSI callback ─► gap fill ─► static suppression  │
 │   ─► presence │ breathing │ heart (opt) │ BNR   │   raw CSI never leaves
 │   ─► FeatureFrame v1 @ 2 Hz ──────────────────┐ │   (debug: raw-CSI raw)
 └────────────────────────────────────────────────┘ │
                                         UDP, sensor VLAN
 Gateway (Rust, Raspberry Pi 5 / mini-PC)            ▼
 ┌──────────────────────────────────────────────────────────────┐
 │ ingest: FeatureFrame v1 | raw CSI ─► saans-dsp (same DSP) │
 │ per space: link-diversity fusion ─► saans-health detectors    │
 │ system-health monitor │ event log (JSONL) │ pcap recorder     │
 │ HTTP API + WebSocket + /metrics + static dashboard            │
 └──────────────────────────────────────────────────────────────┘
```

### 3.1 Repository layout (monorepo)

| Path | Language | Owner | What |
|---|---|---|---|
| `wifisense/` | Python | lead | **Reference implementation.** Unchanged API; generates golden vectors. |
| `spec/` | docs + Python | lead | Wire formats, event schema, golden-vector generator and vectors. |
| `rust/saans-dsp` | Rust | lead | Port of the wifisense core: biquad, Goertzel, band peak, static suppression, presence, breathing, heart, gap filling, RespirFi. No dependencies. |
| `rust/saans-health` | Rust | lead | Health detectors (§2.1) over feature streams. |
| `rust/saans-sim` | Rust | lead | Geometric scene simulator (port of `scene="geometric"`) plus scripted timelines: enter, sit, breathe, pause, burst, leave. Ground truth for S3. |
| `rust/saans-gateway` | Rust | lead | The gateway binary. |
| `rust/saans-cli` | Rust | lead | `saans sim | replay | eval | bench`. |
| `edge/dsp` | C++17 | edge | Header-only DSP for the ESP32: no heap after init, `float` (the S3 FPU is single precision). Host tests on golden vectors. |
| `edge/firmware` | C++ / ESP-IDF | edge | CSI capture, on-node pipeline, FeatureFrame sender, ESP-NOW TX role, telemetry, optional raw mode. |
| `dashboard/` | HTML/JS | lead | Static, served by the gateway: space grid, live state, system-health page. |
| `docs/adr/` | — | both | Decision records (§6). |

## 4. One spec, three implementations

`spec/golden/generate.py` drives the Python reference on fixed inputs and
writes JSON vectors: input sequences and expected outputs per stage
(biquad, Goertzel, band peak, static suppression, presence, breathing
estimate, gap filler, RespirFi selection, FeatureFrame encoding).

- **Rust (f64)** must match within 1e-9 relative on filter outputs and
  0.001 BPM on rates.
- **C++ (float32)** must match within 1e-4 relative on filter outputs and
  0.05 BPM on rates; drift beyond that is a bug, not rounding.
- **Confidences** (bounded [0, 1] scores compared against absolute
  thresholds 0.40 / 0.70) are compared with an **absolute** tolerance:
  1e-9 for Rust, 5e-3 for C++. A relative tolerance is meaningless near 0,
  where float32 Goertzel cancellation makes tiny absolute noise look huge.
- A disagreement is resolved by fixing the implementation that disagrees
  with the reference, or by changing the reference *and* regenerating
  vectors, never by loosening a tolerance silently.

### Port rules (where precision or language semantics bite)

Found by randomised testing beyond the vectors (C++ port, 2026-10-14):

1. **Scan grids are counted by rounding:** `steps = round((f_high -
   f_low) / resolution)`, giving `steps + 1` points. Truncation sat on an
   integer boundary for every configuration in use and lost a point in
   float32. Pinned by `band_peak_grid_rule`.
2. **Interpolation never crosses a notched neighbour.** If either
   neighbour of the peak is notched, the grid point is returned. Pinned by
   `band_peak_notch_neighbour`.
3. **Python `round` is half-to-even** (C: `nearbyint`/`rint` in the
   default rounding mode, not `round`/`roundf`). Applies to the sideband
   harmonic test, window and rate rounding.
4. **Keep float32 divisions in float32** on the edge: promoting one
   operand to double for "accuracy" changes results that the reference
   computes from float32-representable configs differently.
5. **Invalid filter configuration is fatal at start-up**, never a silent
   degraded mode. Python raises; the `-fno-exceptions` C++ build may
   return `false` from `configure()`, but the firmware must then refuse to
   start that extractor and report it in telemetry. An inert filter would
   make presence read "empty" forever, the worst possible failure. No
   clamping to Nyquist.

Speed is a claim like any other: `saans bench` and an `edge/dsp` host
benchmark report frames/s against the Python pipeline on the same input.

## 5. Wire formats

See `spec/wire.md`. Two datagram types share the sensor port:

- **raw CSI** (existing, magic `0xC5110001`): debug and data
  collection only.
- **FeatureFrame v1** (new, magic `0x534E4153`, ASCII "SANS" on the wire): ~64 bytes at 2 Hz
  per link: motion energy, breathing-band RMS, breathing rate + confidence,
  heart rate + confidence (zeroed unless consented), presence state,
  calibration age, achieved CSI rate, loss, RSSI, noise floor, BNR of the
  best subcarrier. Enough for every detector in §2 and nothing that
  reconstructs the raw channel.

## 6. Decisions (ADRs)

| ADR | Decision |
|---|---|
| 001 | Edge-private: features leave the node, raw CSI does not (except debug mode, which is visible in every frame's flags). |
| 002 | One spec, three implementations, golden vectors; Python is the reference. |
| 003 | Edge DSP is `float32`; gateway DSP is `f64`. Tolerances in §4. |
| 004 | FeatureFrame v1 wire format. |
| 005 | Health outputs are advisory, consent-gated per space, and must abstain below a confidence. No identities, ever. |
| 006 | Link diversity (k-of-n presence, best-confidence breathing) over subcarrier selection as the blind-spot strategy. RespirFi stays available per link. |
| 007 | No learned models until the DSP ceiling is measured on real data. |
| 008 | The site and the dashboard are one server; the dashboard moved to `/app`. |
| 009 | A static copy of the site is published, and says on every page that it has no gateway behind it. |
| 010 | The audit log is kept for a year, then deleted. *Decided, not built.* |
| 011 | One gateway per site. A hosted control plane is ruled out; multi-site, if built, is read-only federation. *Decided, not built.* |

## 7. Roadmap with exit gates

**Status (2026-10-14):** S0 done. S1 done (Rust, 672/672 frames
byte-identical, 15x language / 144x total vs Python). S2 done (C++ float32:
845 golden checks, firmware builds for esp32s3, 26k frames/s host). S3
done in simulation (NOTES.md §8: held-out results, three bugs found and
fixed). S4 done (gateway: API, WebSocket, /metrics, replay, dashboard;
36 tests). S6 prep in progress (node memory, TDMA, reciprocal links).
S5 needs hardware.


| Phase | Scope | Exit gate (measured) |
|---|---|---|
| **S0** | Plan, ADRs, wire spec, golden vectors | Vectors generated; spec reviewed by both sessions |
| **S1** | `saans-dsp` parity | All golden vectors pass in Rust; bench vs Python reported |
| **S2** | `edge/dsp` parity + firmware compiles | All golden vectors pass in C++ (float32); firmware builds for esp32s3; host bench reported |
| **S3** | `saans-sim` timelines + `saans-health` | Event-level precision/recall per detector on scripted synthetic timelines (labelled SYNTHETIC), including false alarms per hour on empty and still-only timelines |
| **S4** | Gateway + dashboard | Replays a recording and a simulated 8-hour night end to end; API, WebSocket, /metrics, system-health alarms demonstrated |
| **S5** | Hardware bring-up (= P1) | Presence F1, still-person false-empty, breathing vs belt — the P1 gates |
| **S6** | Multi-link spaces (= P2) | Blind-spot rate with 1 vs 2 vs 3 links measured, sim then real |
| **S7** | Pilot | 2-week run, uptime ≥ 99%, alarm rates per space per day reported |

## 8. Safety, privacy, governance

- Not a medical device, not a safety-of-life system. Alerts are advisory
  and never replace existing procedures. Every alert says what it saw and
  how confident it is.
- Breathing and heart data are health data under India's DPDP Act 2023:
  opt-in per space, written consent for vitals, retention limits
  (events 30 days raw, then 15-minute aggregates), no identities.
- Ethics committee approval before any human-subject data collection (S5).
- A false "breathing pause" alarm costs trust; a missed one costs more.
  Both rates are reported per detector, never one without the other.

## 9. Work split between the two Claude sessions

- **Lead session:** `spec/`, `rust/`, `dashboard/`, `docs/adr/`, this file.
- **Edge session:** `edge/` (C++ DSP library, host tests, ESP-IDF
  firmware, Docker build script). Reads `spec/`, never edits it; requests
  spec changes by message.
- Shared files (`README.md`, `NOTES.md`) are edited by the lead only.

## 10. Output interfaces

One core, many thin outputs. Every output reads the same privacy-filtered
stream (`saans-gateway/src/outputs.rs`), so nothing can show more than the
consent gate allows; roles then show *less*.

| Who | Interface | Sees |
|---|---|---|
| Facilities / security | MQTT (`{base}/{space}/occupancy`, Home Assistant discovery), `occupancy` role on the API | Occupied / empty / unknown. Never vitals, never health alerts |
| Lab safety, wardens | Telegram / webhook alerts with acknowledge, resolve and escalation; `safety` role on the dashboard | Their spaces; vitals only where consented |
| IT / operations | `/metrics` (Grafana), System tab; `ops` role | Sensor and gateway health, occupancy |
| Governance | Audit log; `admin` role | Who read health data, who acted on which alert |
| Occupant (later) | Personal page: own data, consent switch, erase | Their own space |

### 10.1 Authentication, roles, audit
- Roles: `admin`, `ops`, `safety`, `occupancy` (`roles.rs`). A principal
  may be limited to a list of spaces.
- People log in (Argon2id password hashes in the config, HttpOnly
  SameSite=Strict session cookie). Machines use bearer tokens (only the
  SHA-256 is stored). Secrets never live in the config file.
- Auth may be off only while the API listens on loopback; the config
  refuses anything else.
- Every response that contains vital signs or health-alert evidence, and
  every alert acknowledgement or resolution, is appended to the audit log.

### 10.2 MQTT
- Retained `{base}/{space}/occupancy` (`occupied` true/false/null,
  confidence, since); `{base}/{space}/event` (QoS 1); optional retained
  `{base}/{space}/vitals` only for consented spaces, on its own topic so
  broker ACLs can keep it from occupancy clients; `{base}/system/links`;
  last-will `{base}/gateway/status` = `offline`.
- Optional Home Assistant discovery: one occupancy `binary_sensor` per
  space, so lights and AC can follow occupancy with no custom code.

### 10.3 Alerts
- `[[alert_route]]`: which event kinds and severities in which spaces go
  to which channels (`telegram:<name>`, `webhook:<name>`), and where they
  escalate if nobody acknowledges within `ack_timeout_min`.
- One open alert per (space, kind); repeats update it instead of paging
  again. States: open → acknowledged → resolved, persisted to
  `alerts.jsonl`, survive restarts.
- Telegram: inline Acknowledge / Resolve buttons, read back by long
  polling (no inbound port needed), limited to allowed user ids.
- **Leaving the campus:** messages to third-party services carry no
  vital-sign numbers or evidence text unless the route sets
  `include_health_details = true`; by default they say only that an alert
  of a given kind is open in a given space and link to the dashboard.
