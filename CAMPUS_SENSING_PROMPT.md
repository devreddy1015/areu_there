# Prompt: camera-free campus sensing with WiFi

> **How to use:** fill in the `[brackets]` in §1, then paste everything
> below the line into Claude Code (or any coding agent) from the root of
> this repo. Don't ask for the whole system at once. Run it one phase at a
> time with the kickoff prompts in §11.

---

## 0. Role

You are the lead engineer on a camera-free sensing system for a university
campus. Picture IIT Bhilai: classrooms, labs, a library, hostels and
corridors, with no cameras by policy. The system works out where people
are and what each space is doing using only WiFi radio signals. You write
the firmware, signal processing, ML, backend and evaluation code. You
check every claim against ground truth before you make it.

## 1. Deployment parameters (fill in)

- Pilot spaces: `[e.g. one 10x8 m classroom, one lab, one library reading room]`
- Outputs needed (tick): `[ ] occupancy  [ ] headcount  [ ] position in room  [ ] still-person detection  [ ] no-motion / fall alert  [ ] per-building footfall`
- Pilot hardware budget: `[₹ ...]`
- Campus IT: `[can we get a dedicated 2.4 GHz channel? a VLAN for sensor nodes? access to AP analytics?]`
- Governance: `[ethics committee contact; data-protection owner]`

Use the use cases below to decide which tier (§4) each space needs:

| Use case | Tier | Note |
|---|---|---|
| Lights/AC off in empty rooms | T0 | Highest value, lowest risk |
| Classroom headcount | T1 | Counts only. Device-free sensing **cannot** take attendance because it can't tell who anyone is |
| Library / study-room free seats | T2 | Zone-level position |
| Lab lone-worker safety (no motion for N min) | T0 | Advisory alerts only |
| Hostel corridor movement at night | T0 | Motion only, no identity |
| Evacuation: who's still in the building | T4 + T0 | Building counts plus per-room occupancy |

## 2. Read first: existing work in this repo

- `wifisense/` is a working CSI pipeline that uses only the Python standard library: static suppression → fusion → presence (calibrated on an empty room, with hysteresis) → breathing (Goertzel band scan with noise-calibrated confidence) → heart rate (sideband search). It has 66 tests and `evaluate.py`.
- `wifisense/sources/esp32_udp.py` parses the raw-CSI ESP32 wire format byte for byte.
- `NOTES.md` explains each design decision and the bug it prevents. **Do not reintroduce these:** a fixed EMA alpha, calibrating while the filter is still warming up, measuring presence energy across the full band, and estimating rate from zero crossings.
- `RESEARCH.md` holds the verified literature constraints. They are binding (see §3).
- `edge/firmware` is the ESP-IDF node firmware (CSI capture, on-node DSP, FeatureFrames).

Extend this code rather than rewriting what works. Keep the `wifisense/`
core on the standard library only, so it can be ported to C. The new
layers (localization, ML, server) go in separate packages and may use
numpy, scikit-learn and so on.

## 3. Hard constraints (physics and evidence, non-negotiable)

1. **ESP32 has one RF chain (1T1R).** That rules out AoA, MUSIC, SpotFi and cross-antenna CSI ratio. Two separate ESP32s do not form a coherent array, because their oscillators drift independently.
2. **Never sanitize phase with a linear fit and then range from it.** The fit removes the time-of-flight slope that ranging needs.
3. **Single-antenna nodes locate people by which links a body disturbs** (link-based / tomographic). Never use multilateration from absolute range.
4. **Breathing needs at least 10 Hz per link, sustained.** Every multi-node schedule must report the per-link rate and packet loss it actually achieves. If a room can't sustain the rate, turn vitals off there.
5. **Accuracy figures from papers are in-domain ceilings.** Report every learned model three ways: same-session split, leave-one-session-out and leave-one-room-out. The leave-one-room-out figure is the real one.
6. **Every output has a confidence, and the system abstains below a threshold.** "Unknown" is a valid answer.
7. **Calibration is per room and it drifts:** furniture moves, doors open and close, APs change channel, temperature changes. Re-calibrate automatically during known-empty periods and detect when a calibration has gone stale.
8. **WiFi goes through walls.** A person next door can trigger a room's sensor. Measure leakage between adjacent rooms.

## 4. What "location" can honestly mean: the tier model

| Tier | Question | Method | Hardware per space | Expected quality | Maturity |
|---|---|---|---|---|---|
| **T0 Room state** | Is anyone here? Moving or still? | Existing `wifisense` pipeline on one TX→RX link. The breathing band catches people sitting still, which PIR misses | 2× ESP32-S3 (~$6–10 each) | Presence including still people. Must be confirmed on real hardware | Built and tested on synthetic data |
| **T1 Count** | Roughly how many people? | Motion features from many links → classifier over bins 0 / 1 / 2 / 3–5 / 6+ | T0 or T2 hardware | Coarse, trained per room | Research |
| **T2 Position in room** | Where in the room? | Radio tomographic imaging across a mesh of links (RTI/VRTI) on CSI motion energy, plus a tracker. Fallback: zone classifier | 6–12 ESP32s around the walls | Target is zone level (about 1–2 m). **Must be measured** | Supported by RTI literature (RSS, dozens of nodes) |
| **T3 Precise angle or position** | Exact bearing, sub-metre | Phase-coherent arrays: ESPARGOS (ESP32-based) or Intel AX210 + PicoScenes (2 Rx chains, 160 MHz) | Specialist | Literature reports under 2° angle error *on arrays* | Research pilot only |
| **T4 Campus footfall** | How many people per building or floor right now? | Existing enterprise AP analytics (association counts). Device-based and **aggregate only** | Nothing new (campus IT) | Counts per building or floor | Production technology |

**Default plan:** T0 wherever it's useful. T2 only in rooms where position
matters. T4 for campus-wide crowd counts. T3 only as a research
side-project.

T4 counts devices, not people: one person may carry several devices,
phones get switched off, and MACs are randomized. Never store per-device
tracks.

## 5. Architecture

```
[ESP32 nodes in a room] --ESP-NOW broadcast TDMA--> each other (CSI on every link)
        |  UDP (raw CSI + link header) on a dedicated sensor VLAN
        v
[Building gateway: Raspberry Pi 5 / mini-PC]
   ingest -> per-link wifisense pipeline -> room fusion          (T0)
          -> RTI solver + tracker                                (T2)
          -> count model                                         (T1)
          -> events over MQTT
        v
[Campus server] MQTT broker -> TimescaleDB -> FastAPI -> dashboard (floor maps)
                + T4 AP-analytics importer (aggregate counts only)
```

### 5.1 Firmware (ESP-IDF, C)

- Capture CSI with `esp_wifi_set_csi_config` / `esp_wifi_set_csi_rx_cb`. Start from Espressif's `esp-csi` examples. The `esp-csi` examples already send ESP-NOW frames and capture CSI from them.
- **Broadcast TDMA:** node *k* broadcasts an ESP-NOW frame in slot *k* of each superframe, and every other node captures CSI from it. One transmission gives N−1 links, and a superframe lasts N × slot. Node 0 sends sync and the others align by their slot offset. Target at least 20 Hz per link.
- **Per-packet header:** rx node_id, tx_id, seq, local timestamp (µs), channel, RSSI, noise floor, and a subcarrier-layout ID. The LLTF/HT-LTF layout varies by chip and config, so record it.
- Fix the channel, bandwidth and TX power. Agree the channel with campus IT.
- Time-sync to the gateway, plus OTA updates, a watchdog and health telemetry (rate, loss, free heap, temperature).
- Mount nodes at about 1–1.2 m (chest height), rigidly. Keep them off doors and away from metal.

### 5.2 Gateway

- **Multi-link source.** `CsiFrame` already has `node_id`. Add `tx_id` (link id).
- **Recorder and replayer.** Write raw packets to disk with timestamps so every real-data bug can be reproduced from a recording.
- **Per-link pipelines.** Run one `SensingPipeline` per link. Fuse at room level: the room is occupied if k of n links agree, and breathing comes from the link with the best confidence.
- **RTI:**
  1. Divide the room into 0.25–0.5 m pixels.
  2. Weight each link with an ellipse (Fresnel) model.
  3. Measure motion energy per link, as in VRTI (the variance of the static-suppressed amplitude).
  4. Invert with Tikhonov-regularized least squares.
  5. Pick peaks and track them with a Kalman or particle filter.
  6. Output (x, y, covariance) with a confidence.
- **Count.** Take features from the distribution of link energies and train a small model with calibrated probabilities. It must be able to abstain.

### 5.3 Server and dashboard

Event schema (versioned):

```json
{ "v": 1, "ts": "2026-10-13T10:15:02Z", "space_id": "LHC-101",
  "occupied": true, "occupied_conf": 0.97,
  "motion": "still",
  "count": {"bin": "3-5", "conf": 0.71},
  "positions": [{"x": 3.2, "y": 4.8, "sigma_m": 1.1, "conf": 0.64}],
  "calibration": {"age_h": 6.5, "state": "fresh"},
  "health": {"links_ok": 11, "links_total": 12, "rate_hz_min": 22.4} }
```

- No identities anywhere in the system.
- Retention: keep raw CSI for at most `[7]` days (debugging only). After `[30]` days, aggregate events into 15-minute buckets.
- Dashboard: a floor plan showing occupancy per room, live position dots in T2 rooms, and a health page for dead nodes, stale calibrations and rate drops.

## 6. Ground truth and evaluation (build these before the models)

- **Presence:** labelled sessions plus a door contact sensor, then a 48-hour run in a real room.
- **Still-person presence:** a subject sits reading or typing for 20 minutes. PIR fails at this, so this test is where WiFi has to earn its place.
- **Breathing:** a respiration belt, or breathing paced to a metronome and counted.
- **Position:** a UWB tag (Qorvo DWM1000/DWM3000 class) worn by the test subject during data collection only. Alternatively, a floor-tape grid and scripted walks.
- **Count:** scripted sessions with a known headcount, plus timetable data.
- **Baselines:** put a PIR sensor and a cheap 24 GHz mmWave presence module (such as the HLK-LD2410) in the same room. If WiFi neither beats nor complements them, report that plainly.
- **Metrics:**
  - presence precision, recall and F1, plus the false-empty rate with a still person
  - position error (median and p90, in metres) and zone accuracy
  - count accuracy (exact and ±1)
  - latency, achieved rate per link, and uptime
- Write one evaluation script per capability, in the style of `evaluate.py`. Each must be reproducible from recordings and print a results table.

## 7. Roadmap with exit gates

| Phase | Scope | Exit gate (measured) |
|---|---|---|
| **P0** | No hardware. Add RespirFi subcarrier selection to `preprocess.py`, Fresnel nulls to the synthetic source, a multi-link synthetic source with a walking body, and the recorder/replayer | Tests pass. Synthetic RTI median error reported for 4/6/8/12 nodes |
| **P1** | One room, 2 nodes (TX + RX), real CSI through `wifisense listen` | Presence F1 ≥ 0.95 over 48 h. Still-person false-empty < 5%. Breathing within ±1 BPM of the belt in ≥ 80% of 60 s windows (seated) |
| **P2** | Same room, 8-node TDMA mesh | ≥ 20 Hz per link sustained with < 5% loss. RTI median and p90 error measured against UWB |
| **P3** | Count model, leakage study between adjacent rooms, automatic re-calibration | Count ±1 accuracy reported leave-one-room-out. Leakage quantified |
| **P4** | Building pilot of 10–20 rooms with gateway and dashboard, 2-week run | ≥ 99% node uptime. Drift alarms work. IT signs off on airtime use |
| **P5** | Campus rollout plus T4 aggregate footfall | Governance approved. Ops runbook written |

The gate numbers are starting targets. Revise them after the P1
measurements and write down why.

## 8. Privacy, safety and governance (requirements, not afterthoughts)

The no-camera policy exists for privacy, so keep to its intent.

- **Aggregate, anonymous outputs only.** Store no identities, and keep no per-person trajectory beyond a single room session.
- **Vital signs count as health data.** Keep them off by default across the campus. Turn them on only for explicit opt-in uses, such as lone-worker lab safety, with written consent.
- **India's DPDP Act 2023 and its Rules** require notice, purpose limitation and retention limits. T4 AP data is linked to people: hash it with a rotating salt at ingestion and delete it after aggregation.
- **Ethics approval** from the institute ethics committee comes before any human-subject data collection (P1 onward). Put up signs in instrumented spaces.
- **This is not a safety-of-life system.** No-motion and fall alerts are advisory and never replace existing safety procedures.

## 9. References (verify before relying on any number)

Wilson & Patwari, *Radio Tomographic Imaging with Wireless Networks*
(IEEE TMC 2010) and *See-Through Walls: Motion Tracking Using
Variance-Based Radio Tomography Networks* (IEEE TMC 2011) · Espressif
`esp-csi` · ESPARGOS phase-coherent ESP32 array · PicoScenes (Intel AX210)
· RespirFi · Widar3.0 · everything listed in `RESEARCH.md`. Also check the
current IEEE 802.11bf (WLAN sensing) status and whether any hardware
supports it.

## 10. Working rules for you (the agent)

- **Measure before you claim.** Every capability ships with an evaluation script and a results table. Label synthetic results as synthetic.
- Every real-data bug gets a recording and a regression test.
- Prefer simple, explainable DSP over deep models until you have measured the ceiling of the simple version.
- Keep the core portable. Heavy dependencies belong only in the gateway, server and ML packages.
- Log each finding in `NOTES.md`: what you expected, what you measured and what you changed.
- **Ask me first** before buying hardware, picking a radio channel, or touching the campus network or IT systems.
- When the evidence says something won't work (ESP32 AoA, for example), say so and propose the honest alternative. Don't build demos that only look like they work.

## 11. Phase kickoff prompts

**P0a: signal processing upgrades (no hardware)**
> Read RESEARCH.md, NOTES.md and the wifisense package. Implement RespirFi
> subcarrier selection (RESEARCH.md §3) in preprocess.py behind a flag. Add
> Fresnel-null modelling to the synthetic source, and use evaluate.py to
> show whether the selection improves breathing accuracy in rooms with
> nulls. Then build a recorder/replayer for raw UDP packets.

**P0b: localization on synthetic data**
> Build a multi-link synthetic source: N nodes around the room perimeter
> and one walking person who perturbs the links near their position.
> Implement VRTI-style localization in a new `wifisense/localize/` package
> (numpy allowed). It needs link geometry, the ellipse weight model,
> Tikhonov-regularized inversion, peak extraction and a Kalman tracker.
> Report median and p90 error against node count (4, 6, 8, 12) and pixel
> size. Report it honestly.

**P1: first real hardware**
> I have 2 ESP32-S3 devkits. Write or adapt ESP-IDF firmware so that one
> node broadcasts ESP-NOW at 50 Hz on channel [X] and the other captures CSI
> and streams raw CSI over UDP. Give me the flash and run steps. Then give
> me a data-collection protocol (empty / seated still / walking / breathing
> with a belt) and the evaluation scripts for presence and breathing.

**P2: the mesh**
> Extend the firmware to N-node broadcast TDMA: node 0 sends sync, each node
> gets a slot, and every node captures every other node. Measure the
> achieved per-link rate and loss at N = 4, 8 and 12. Then run the
> localization stack on real data with UWB ground truth.

**P3/P4: system**
> Build the gateway service: ingest → per-link pipelines → room fusion →
> RTI → MQTT events following the §5.3 schema. Add automatic
> re-calibration during detected-empty periods. Then build a FastAPI +
> TimescaleDB backend with a floor-plan dashboard and a health page.
