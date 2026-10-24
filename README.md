# wifisense

A minimal, honest WiFi CSI sensing pipeline: detect room occupancy and
measure breathing from ordinary WiFi radio signals, with no camera and no
wearable.

Every stage is derived from first principles and every claim is measured
against ground truth (see [NOTES.md](NOTES.md)).

**Pure Python standard library.** No numpy, no scipy, no install step.
Every stage is simple float arithmetic that ports directly to C or Rust
for on-device use.

## Quick start

No hardware needed — the synthetic source models the radio physics:

```bash
python3 -m wifisense demo
```

```
ground truth: breathing 15.0 BPM, heart 68.0 BPM
phase 1: empty room, 90s (50s filter warmup + 30s calibration)
  calibrated: True  threshold=0.0527
phase 2: subject present, 150s
  t=  130.1s  OCCUPIED  energy=0.1513  breathing=15.00 BPM [ok] c=0.92  heart=77.00 BPM [ ?] c=0.21
  t=  150.1s  OCCUPIED  energy=0.1450  breathing=15.00 BPM [ok] c=0.93  heart=68.00 BPM [ok] c=0.83
phase 3: subject leaves, 80s
  t=  270.1s  empty     energy=0.0421  breathing= 0.00 BPM [ -] c=0.00  heart= 0.00 BPM [ -] c=0.00
```

Run the tests and the ground-truth evaluation:

```bash
python3 -m unittest discover -s tests
python3 evaluate.py               # both suites, ~4 min on 20 cores
python3 evaluate.py --suite legacy                # the original 64-scenario sweep, ~15 s
python3 evaluate.py --suite nulls --quick         # Fresnel blind-spot study, small
```

## Saans: the system built on this

`wifisense` is now the **reference implementation** of Saans (साँस,
"breath"), a camera-free health and safety sensing system for campus
spaces: ESP32 sensors run the DSP in C++ and send only 68-byte
FeatureFrames (raw CSI never leaves the room; breathing data only where a
space has opted in), and a Rust gateway fuses links, runs advisory health
detectors and serves a dashboard. Plan: [SAANS_PLAN.md](SAANS_PLAN.md);
decisions: [docs/adr/](docs/adr/); wire format: [spec/wire.md](spec/wire.md).

**Read the site without running anything:**
<https://devreddy1015.github.io/areu_there/>. It is a static copy, so
the live demo and the dashboard need a gateway on your own machine, and
the copy says so on those pages.

| Part | Where | Checked by |
|---|---|---|
| Reference DSP + FeatureFrame | `wifisense/` (Python) | `python3 -m unittest discover -s tests` |
| Golden vectors | `spec/golden/` | `python3 spec/golden/generate.py` |
| Rust DSP (byte-identical frames) | `rust/crates/saans-dsp` | `cargo test -p saans-dsp` |
| Health detectors, link fusion | `rust/crates/saans-health` | `cargo test -p saans-health` |
| Simulator (matches Python physics to 1e-14) | `rust/crates/saans-sim` | `cargo test -p saans-sim` |
| Evaluation | `rust/crates/saans-cli` | `cargo run --release -p saans-cli -- eval` |
| Gateway, API, dashboard | `rust/crates/saans-gateway` | `cargo test -p saans-gateway` |
| ESP32 DSP + firmware (C++) | `edge/` | `make -C edge/dsp/test check`, `edge/build.sh` |

Rust lives in `rust/` (`~/.cargo/bin/cargo` if cargo is not on PATH).
Measured results and every bug the evaluation found are in
[NOTES.md](NOTES.md) §7–8.

## What works, measured

Across 64 synthetic scenarios (breathing 8–24 BPM x heart 50–110 BPM),
`evaluate.py`:

| Capability | Result |
|---|---|
| **Presence / occupancy** | Reliable. ~4x energy separation between occupied and empty after calibration; stable, no flicker. |
| **Breathing rate** | **Mean error 0.009 BPM, max 0.017 BPM, 64/64 within 0.5 BPM.** |
| **Heart rate, ungated** | 80% within 3 BPM — not trustworthy on its own. |
| **Heart rate, confidence ≥ 0.4** | **97% correct (38/39), mean error 0.38 BPM**, accepting 61% of scenarios. |

The breathing and heart-rate rows are **checked**: they quote the
`legacy` section of [docs/RESULTS.md](docs/RESULTS.md), which CI
regenerates on every push, and `tools/results.py --check` fails the build
if a figure in those rows stops matching what the suite prints. The
presence row is **not** checked: it is the early measurement in
`NOTES.md` §4 (0.034 empty against 0.15 occupied, after bandpassing to
0.1–2.0 Hz), and no current command reproduces it. All of it is
synthetic.

The heart-rate row is the important one. The estimator is wrong roughly a
fifth of the time, but it *knows* when: gating on reported confidence
turns an unreliable number into a reliable one that sometimes abstains.
Confidence below 0.4 means "I don't know", not "here is a worse guess".

That sweep uses the original synthetic scene, whose per-subcarrier phases
are random and independent, and a single seed (one room). A physically
coherent room behaves differently: at ESP32 bandwidth a Fresnel-zone
blind spot can cover every subcarrier at once. In 480 uniformly placed
geometric scenes (`evaluate.py --suite nulls`), breathing is within
0.5 BPM in **95.1%** of readings by default and **95.7%** with RespirFi
selection; at whole-band blind spots it drops to 64.5% / 68.1%, typically
by locking onto twice the true rate.

These are synthetic-data numbers. They establish that the DSP is sound.
They say nothing about real hardware, real rooms, or real bodies — see
[NOTES.md](NOTES.md).

## How it works

```
CSI frame (per-subcarrier amplitude + phase)
   |
   v  StaticSuppressor    walls and furniture dominate the signal; subtract
   |                      a slowly-varying per-subcarrier baseline, and weight
   |                      each subcarrier by how much it actually moves
   v  fuse()              weighted collapse to one scalar
   |
   +--> PresenceDetector  bandpass 0.1-2.0 Hz, RMS energy, threshold learned
   |                      from an empty room, hysteresis to stop flicker
   |
   +--> BreathingExtractor  bandpass 0.1-0.5 Hz -> decimate -> Goertzel band
   |                        scan -> peak + noise-calibrated confidence
   |
   +--> HeartRateExtractor  bandpass 0.8-2.0 Hz -> symmetric-sideband search
                            using the breathing frequency to reject its
                            harmonics and intermodulation products
```

Why Goertzel rather than an FFT: only two narrow bands matter, so
evaluating the DFT at just those frequencies is both cheaper and finer
than binning the whole spectrum — and it ports to an ESP32 in a few lines.

## Using it

```python
from wifisense import SensingPipeline
from wifisense.sources import SyntheticCsiSource

source = SyntheticCsiSource()
pipeline = SensingPipeline(sample_rate_hz=source.config.sample_rate_hz)

for frame in source.seconds(300):
    reading = pipeline.process(frame)

print(reading.occupied, reading.respiratory_rate.value_bpm)
```

With real hardware, feed `CsiFrame`s from any source:

```bash
python3 -m wifisense listen --port 5566 --rate 20
```

`wifisense.sources.esp32_udp` parses the raw CSI debug format
(`spec/wire.md` §1), which the Saans node emits in raw mode.
Lost packets are interpolated across by sequence number (up to
`--max-gap-s`, default 1 s). Without that, 15% loss compresses the time
axis and a 15 BPM breather reads 17.8 BPM at confidence 0.98.

### The gateway, the site and the dashboard

The gateway ingests FeatureFrames (or raw CSI in debug mode), fuses the
links of each space, runs the health detectors, and serves both the
public site and the operator dashboard from the same binary and the same
origin.

```bash
cd rust
cargo build --release -p saans-gateway -p saans-cli
cp crates/saans-gateway/saans.example.toml saans.toml   # then edit it
./target/release/saans-gateway --config saans.toml --check
./target/release/saans-gateway --config saans.toml
```

Everything is under the `http_addr` in the config, `http://127.0.0.1:8080`
by default:

| Path | What it is |
| --- | --- |
| `/` | The site: what this is, and who it is not for |
| `/how-it-works` | The eight stages between a radio packet and a decision |
| `/demo` | The simulator and the real pipeline, side by side |
| `/evidence` | Measured numbers, comparisons, and what is not built |
| `/privacy` | The 68-byte frame, encoded live, with the consent switch |
| `/hardware` | Boards, cost, and campus deployments |
| `/start` | Running it, and the API surfaces |
| `/app` | The operator dashboard |
| `/api/…`, `/metrics` | The API. Authenticated; see below |

The site and the dashboard are embedded in the binary with
`include_str!`. There is no build step and no bundler, and nothing is
fetched from the internet: no CDN, no web font, no remote script. Links
to the repository are ordinary links a reader may follow — the tests in
`src/site.rs` draw that distinction explicitly and ban remote
subresources while permitting anchors.

`/demo` drives the gateway's own simulate mode through `/api/sim`, so it
shows the real DSP reading a synthetic channel. It needs the gateway to
be running with `--simulate`; without it the page says so rather than
drawing a room.

The same site is published as static files on GitHub Pages by
`.github/workflows/pages.yml` on every push to `main` that touches it.
`saans-gateway --export-site DIR` writes the copy from the table the
router walks, so the two cannot drift; that is also why every in-site URL
is relative (`site.css`, not `/site.css`): Pages serves the project under
`/areu_there/`. On the copy, `/demo` finds no gateway and says so, and
`/app` is a page that says the dashboard is not there.
[ADR-009](docs/adr/009-static-copy.md) has the reasoning.

The dashboard answered `/` before the site existed. Old `/#system`-style
links are forwarded to `/app`.

**See it work without hardware.** Export a simulated night and replay
it; every view is labelled REPLAY throughout, and the recording is the
same wire format a real node sends:

```bash
./target/release/saans export night --out night.pcap --layout 2
./target/release/saans-gateway --config saans.toml --replay night.pcap --speed 25 --serve
```

The export writes links `1<-0` and `1<-2`, so the space in `saans.toml`
has to list those; `/api/health` lists anything that arrived from a link
no space claims, which is the quickest way to find a mismatch.

| View | Shows |
|---|---|
| Overview | Per-space state, motion, breathing where consented, usable links, open alerts |
| Signals | Real trends from `/api/history`: occupancy band, motion, breathing and its confidence, and per-link CSI rate, loss, BNR and RSSI |
| Topology | Which nodes are linked to which, and the quality of each link. Not a floor plan — the configuration has no room coordinates, so node positions are arbitrary and labelled as such |
| Events | Searchable and filterable event log with the evidence for each. Above it, when alert routing is configured, the open alerts, with Acknowledge and Resolve for admin and safety |
| System | Gateway health, per-link diagnostics, unknown links, auth and audit status (admin or ops only) |

**Data-source modes are never mixed.** `LIVE` and `REPLAY` are shown as
their own indicator, separate from the connection state, because a
connected WebSocket does not mean sensors are reporting. A feed that
stops delivering shows as stale rather than live.

### Who may see what

Authentication is **required automatically whenever `http_addr` is not a
loopback address**, and the config refuses to start otherwise. On
loopback it is off and every request is a local admin.

```bash
saans-gateway --hash-password          # reads the password from stdin
```

Put the printed Argon2id string in a `[[user]]`. Machines use
`Authorization: Bearer <token>`, with only the token's SHA-256 in the
config.

| Role | Sees |
|---|---|
| `admin` | Everything, including the audit log |
| `ops` | Gateway and sensor health, occupancy. **Never** vital signs or health alerts |
| `safety` | Full state and alerts for their spaces; vital signs only where the space has consented |
| `occupancy` | Occupied / empty / unknown per space. No vital signs, no health alerts |

Either role can be limited to named spaces. A space outside a
principal's scope answers 404, not 403 — the scope list is not a
directory of the building. Reads of consented vital signs are recorded
in `audit-YYYY-MM-DD.jsonl` alongside logins and denials.

### How long it is kept

Four files in `data_dir`, three clocks, and they are deliberately not
the same length. `/api/health` reports each one's `retention_days` beside
what it has written, so an operator can check the policy is in force.

| File | What it holds | Kept |
|---|---|---|
| `events-YYYY-MM-DD.jsonl` | What the building did, per event | `event_retention_days`, default 30 |
| `history.json` | One summary per space per day: occupied seconds, motion bouts, pauses, and for consented spaces a resting breathing median | `history_retention_days`, default 90 |
| `audit-YYYY-MM-DD.jsonl` | Who read health data and who acted on an alert | `audit_retention_days`, default 365, and never less than `event_retention_days` — [ADR-010](docs/adr/010-audit-retention.md) |
| `alerts.jsonl` | Each alert and every change to it, including the event that raised it | open and acknowledged: until resolved; resolved: `event_retention_days`, because it carries the event |

The daily summaries are also capped at 90 entries in memory
(`HISTORY_MAX_DAYS`). That is a memory bound, not a retention policy: a
gateway that runs one day a month would hold ninety summaries spanning
years and never exceed the cap. `history_retention_days` is the one that
bounds age, and it is applied where the data becomes a file — when
history is loaded and when it is written.

### Alerts

With `[[alert_route]]`s configured, events become advisory alerts on
Telegram or a webhook (`saans.example.toml` has a commented example).
Routes are tried in order and the first match owns the alert; `info`
events never page. One alert per space and kind is open at a time —
repeats update it rather than paging again — and an alert still open
after `ack_timeout_min` escalates once to `escalate_to`. Acknowledge and
Resolve work from the dashboard, from `POST /api/alerts/{id}/ack|resolve`
(admin and safety, in their spaces), and from Telegram's buttons (users
in `allowed_user_ids`), and every one is in the audit log.

**What leaves the building** is, by default, only that an alert of a given
kind is open in a given space. The evidence sentence, the confidence and
every number stay home unless a route sets `include_health_details`.
Tokens and webhook keys come from environment variables; the gateway
refuses to start if one is missing, rather than starting unable to page.

### MQTT and Home Assistant

With `[mqtt]` configured (`saans.example.toml` has a commented example)
the gateway publishes occupancy for building systems:

| Topic, under `base_topic` | Retained | What |
|---|---|---|
| `gateway/status` | yes | `online`, or `offline` — also the last will, so a crash says so |
| `<space>/occupancy` | yes | `{"occupied": true \| false \| null, "since", "links_usable", "links_total"}`, on change |
| `<space>/event` | no | Occupancy and coverage events only |
| `<space>/vitals` | yes | Only with `vitals_topics = true`, only for `vitals = true` spaces, every 10 s |
| `system/links` | yes | Link health, no vital signs, every 10 s |

`null` is "unknown": no link is usable. With
`homeassistant_discovery = true` each space appears in Home Assistant as
an occupancy `binary_sensor` (unknown for `null`, unavailable while the
gateway is offline), so lights and air conditioning can follow it with no
custom code. Everything except the vitals topic is exactly what the
`occupancy` role sees: health alerts and their events never reach the
broker. There is no occupancy *confidence* field: the fusion decides by a
vote of links, and the 0.6 / 0.8 / 0.9 on occupancy events is a lookup on
how many links are usable, not a measured probability, so the payload
carries the link count itself.

The connection is plain TCP, so a password (`password_env`) or
`vitals_topics` are refused unless the broker is on the same machine; to
reach a campus broker, bridge a local one to it over TLS. A broker that is
down or never answers costs ingest nothing (tested), events during an
outage are dropped and counted, and the retained topics are republished on
reconnect. `/api/health` and `/metrics` show the connection.

### Recording and replay

    python3 -m wifisense record --port 5566 --out run.pcap [--duration-s 600]
    python3 -m wifisense stats run.pcap
    python3 -m wifisense replay run.pcap [--node 3] [--rate auto|20] [--calibration-s 30]
    python3 -m wifisense replay run.pcap --send 127.0.0.1:5566 [--speed 1|0]

`record` writes every raw datagram, malformed ones included, to a
nanosecond pcap that opens in Wireshark; captures from `tcpdump -w` work
too. `stats` shows the per-node rate, sequence-gap loss, resets (reboots)
and inter-arrival jitter the gateway actually received. Check it before
trusting vitals on a link (at least 10 Hz sustained). `replay` runs the
pipeline offline over one node's frames and gives the same readings as
the live run, or re-sends the datagrams over UDP to exercise `listen`.
Timestamps are gateway wall-clock time, so keep NTP in slew mode.

### Subcarrier selection (RespirFi)

`SensingPipeline(subcarrier_selection="respirfi")`, or
`--subcarrier-selection respirfi` on `listen` / `replay`, replaces the
variance-weighted average with RespirFi's BNR selection and sign-flip
fusion (Wang et al., IEEE IoT-J 2026) for the vitals path. On 960
synthetic rooms with Fresnel blind spots it was never worse than the
default and slightly better (95.1% -> 95.7% within 0.5 BPM; blind spots
64.5% -> 68.1%), mostly by removing confidently-wrong double-rate
readings. It stays off by default until it has been compared on real
recordings (`replay --subcarrier-selection respirfi`). Details in
[NOTES.md](NOTES.md) §6; rerun with `python3 evaluate.py --suite nulls`.

### Operating requirements

Both matter more than they look, and getting either wrong looks like a
tuning problem when it is not:

- **The room must be empty during calibration.** Default: 50 s filter
  warmup, then 30 s of measurement. Calibrating with someone present
  learns them as the baseline and the room then reads as permanently
  empty.
- **Breathing needs ~30 s of history** before the first estimate (three
  cycles at the slow end of the band); heart rate needs ~60 s.

## Hardware

You have none yet, so: the cheapest real CSI path is an **ESP32-S3 or
ESP32-C6 devkit, about $6–10**. The ESP-IDF API (`esp_wifi_set_csi_config`,
`esp_wifi_set_csi_rx_cb`) exposes per-subcarrier CSI directly, and the
Saans node firmware in `edge/firmware` is ready to flash. One link gives presence and breathing for one space; more links per space
buy blind-spot coverage, not new capabilities.

Multi-person separation and through-wall sensing are **not** on this
roadmap and are not gated on node count. No through-wall respiration
result exists on single-antenna hardware anywhere in the literature, and
a link with no range or bearing resolution cannot tell which side of a
wall the person is on — so through-wall reach is something to measure
and suppress, not a feature to pursue. See
[docs/CAPABILITY_LIMITS.md](docs/CAPABILITY_LIMITS.md).

Consumer WiFi laptops give RSSI only — enough for crude motion, not for
breathing. Research NICs (Intel 5300, Atheros via nexmon on a Raspberry Pi
CYW43455) are the other real option and cost more hassle than an ESP32.

## Layout

| Path | What |
|---|---|
| `wifisense/dsp.py` | Biquad bandpass, Goertzel, band scan, sideband search |
| `wifisense/preprocess.py` | Static suppression, subcarrier weighting, RespirFi selection |
| `wifisense/vitals.py` | Breathing and heart-rate extractors |
| `wifisense/presence.py` | Ambient-calibrated occupancy |
| `wifisense/pipeline.py` | The stages wired together |
| `wifisense/sources/synthetic.py` | Physics-based CSI generator; `scene="geometric"` models Fresnel-zone blind spots |
| `wifisense/sources/esp32_udp.py` | ESP32 wire format, UDP listener, sequence-gap filling |
| `wifisense/sources/recorder.py` | Raw-datagram pcap recorder, replayer, per-node link stats |
| `scenes/` | Scene files: the room as DEFINED. Schema, the ITU-R P.2040 material table, three example rooms |
| `wifisense/scene.py` | Scene loader and validator (stdlib only), and the conversion to `localize.geometry` |
| `wifisense/roomplan_import.py` | Apple RoomPlan scan to scene file. A stub: the mapping table is real, the format assumptions are untested |
| `rust/crates/saans-sim/src/scene.rs` | The same format in Rust, with its own reader |
| `evaluate.py` | Ground-truth accuracy sweeps (`--suite legacy`, `--suite nulls`) |
| `tests/` | 550 tests, 242 of them outside `test_localize_*` (which belong to the P0b localization work) |
| `rust/crates/saans-gateway/assets/dashboard.html` | The whole dashboard, one file, served from the binary |
| `rust/crates/saans-gateway/assets/site/` | The eight public pages, the stylesheet, the scripts and the fonts |
| `rust/crates/saans-gateway/src/site.rs` | The route table for those pages, and the offline and link tests |
| `docs/HARDWARE.md` | What to buy for the first bench run, and how to bring it up |
| `docs/PROTOCOL_S5.md` | The recording protocol, to be followed before any human-subject data exists |
| `docs/EVIDENCE.md` | What this project has actually demonstrated, and what it has not |
| `docs/UI_GAP_ANALYSIS.md` | What the interface can show, what the gateway publishes, and where they disagree |
| `docs/UI_IMPLEMENTATION_PLAN.md` | The design the dashboard was built to, and the rules it keeps |

## Licence

MIT. `NOTICE` holds a third-party copyright notice that must be kept in
any redistribution.

## Not a medical device

Breathing and heart rate here are engineering measurements on synthetic
data. Nothing in this project is validated for clinical, diagnostic, or
safety-of-life use, and sensing people has legal and consent implications
that vary by jurisdiction.
