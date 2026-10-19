# S5 recording protocol (Track B3)

The procedure that must exist before any human-subject data is recorded.
It is written to be followed without asking a question, and to produce a
dataset whose headline number survives contact with a room that was not
chosen to flatter it.

---

## 0. Status, and what this document is for

**No board in this project has ever run this firmware.** Every number in
`NOTES.md`, `README.md` and `docs/` is simulated, and
`docs/CAPABILITY_LIMITS.md:338-339` says so in those words. This protocol
produces the first measurements that are not.

That is the whole reason for the `Source` column in every results table
in §15. A row may read `REAL` only when the number in it was computed
from a pcap that came off a radio, and the pcap's filename is in the
row's provenance cell. Everything else reads `SYNTHETIC` and carries the
`file:line` it was copied from. There is no third value. A blank cell
means not measured, and a blank cell is an acceptable thing to publish.

Track B3 does **not** close the S5 exit gate on its own. The gate in
`CAMPUS_SENSING_PROMPT.md:157` reads "Presence F1 ≥ 0.95 over 48 h.
Still-person false-empty < 5%. Breathing within ±1 BPM of the belt in
≥ 80% of 60 s windows (seated)". The 48-hour part needs the separate
unattended run in §14. This protocol closes the geometry-and-posture
part: how the system behaves across postures, distances, link geometry
and breathing rate, measured against a reference, with its failures
counted rather than averaged away.

---

## 1. Before anyone sits down

Five questions have to be answered on an empty room. None of them is
human-subject work and none of them needs ethics approval. Four are
cheap; §1.1 needs a firmware change first, and says so. Answering them first is not bureaucracy: if the AGC turns out
to move often, the presence detector is broken and every subject hour
recorded against it is wasted.

### 1.1 Count the AGC gain moves

`NOTES.md:585-588` states the task and why it ranks above everything
else: "log `agc_gain` on an idle node in an empty room for ten minutes
and count the moves. It is the single most valuable number in the first
recording, and until someone takes it the table above ranks nothing."

The table it refers to (`NOTES.md:528-535`, SYNTHETIC) has the modelled
ESP32 arm reporting **0%** of frames occupied where the clean arm reports
100%, with the empty room showing *more* motion energy (1.018) than the
seated person (0.882). The sweep at `NOTES.md:563-568` (SYNTHETIC) brackets
the breakpoint between a gain that moves on 0.02% of packets (presence
survives) and 0.18% (presence fails completely), and
`NOTES.md:580-583` says plainly that narrowing that bracket in simulation
"still answers nothing a bench recording would not answer better".

**The firmware cannot do this today, and that is the first task.**
Nothing in `edge/firmware/main/` reads or logs an AGC gain; a grep for
`agc` over the firmware source returns one unrelated comment
(`edge/dsp/tdma.hpp:57`) and nothing else. The raw-CSI header has no room
reserved for it either: bytes 18 and 19 are written as literal zeroes,
"PPDU type: unused here" and "flags"
(`edge/firmware/main/csi_capture.cpp:237-238`). So this step is not a
recording, it is a firmware change followed by a recording, and it is not
cheap in the way the other four questions in this section are.

**Do, in order:**

1. Establish whether the gain is readable at all. `wifi_pkt_rx_ctrl_t` is
   what the CSI callback already receives, and `rssi` and `noise_floor`
   are taken from it (`csi_capture.cpp:235-236`). [UNVERIFIED: whether
   that struct exposes an AGC gain field on the ESP32-S3 — the ESP-IDF
   headers live inside the `espressif/idf:v5.4.1` container and have not
   been read.] Read the header first; the answer decides the next step.
2. If a gain field exists, log it per frame — byte 18 or 19 of the raw
   header is free — and count the moves directly.
3. If it does not, use the proxy: RSSI is already in the header at byte
   16 and an AGC step appears in it as a staircase. **Establish the
   proxy before quoting a number from it.** A staircase in RSSI can also
   be the channel, and this protocol does not accept a figure whose
   quantity is undefined. The cheapest way to establish it is a
   controlled gain change — move the TX, or attenuate it — and show the
   step appears in RSSI where it was induced.

Then: one RX node, empty room, idle, 10 minutes. Report moves per minute
and the fraction of packets on which the gain moved, and state in the
cell which of the three routes produced it. Put it in Table E.

**If** the measured fraction lands above ~0.1% of packets, stop. Presence
calibration is the thing at risk, not breathing, and §12's presence gates
cannot be measured until it is understood. Record the number, write up
what it implies, and do not start recording people against a detector you
already know is inverted.

**If route 1 says the gain is not exposed and route 3's proxy cannot be
established**, that is also a result, and it goes in Table E as a blank
cell with the reason. It does not block the rest of this protocol; it
means the presence gates in §12 are reported with the caveat that the
mechanism `NOTES.md` predicts would invert them was never measured.
Breathing is unaffected either way — `NOTES.md:528-535` has the AGC arm
breaking presence while breathing survives.

### 1.2 Settle the bin layout

`NOTES.md:600-604`: "Which 56 bins the firmware actually hands over is
unresolved. The simulator's 56 are the idealised occupied set; by
ESP-IDF's documentation the firmware's first 56 pairs may be a different
56 entirely... A subcarrier index may not mean the same thing on both
sides."

**Do:** capture 60 s of raw CSI from an empty room and plot mean amplitude
against reported bin index. Null bins (DC, guards) are identifiable by
near-zero amplitude. Record which indices are dead and which carry
signal. This decides whether `best_bnr` and the RespirFi comparison are
comparing like with like, and it is a 10-minute job.

### 1.3 Record the `first_word_invalid` flag

`NOTES.md:597-599` notes the flag is not modelled and that its real
effect is "a two-bin index shift on affected frames, not garbage".
Log the flag per frame during 1.2 and report what fraction of frames
carry it. If it is common, every per-bin result in this protocol needs
those frames excluded or corrected, and that decision has to be made
before the dataset exists, not after.

### 1.4 Measure the amplitude noise floor

`NOTES.md:362-363`: "Real ESP32 amplitude noise is unmeasured; P1 sets
the noise level the table should be read at." The synthetic breathing
results in `NOTES.md:294-301` were produced at noise levels std 0.6–8
against |LoS| = 40 (`NOTES.md:290-291`), and until the real figure exists
nobody knows which row of that table applies.

**Do:** 10 minutes of empty-room raw CSI, no movement in or adjacent to
the room. Report per-subcarrier amplitude mean and standard deviation,
and the coefficient of variation. This is the number that says where in
the synthetic table the real system sits.

### 1.5 Prove the link can carry vitals

`README.md:271` sets the bar: check `wifisense stats` "before trusting
vitals on a link (at least 10 Hz sustained)". `Kconfig.projbuild:55`
gives the reason — "The DSP needs >= 10 Hz: the gross-motion band reaches
4 Hz."

**Do:**

```bash
python3 -m wifisense record --port 5566 --out bench/link-check.pcap --duration-s 600
python3 -m wifisense stats bench/link-check.pcap
```

Report rate, loss %, resets, `dt_med_ms`, `dt_p95_ms`, `dt_max_ms` into
Table D. Two things in that output need interpreting, and both will
confuse someone who has not been told:

- **FeatureFrames count as malformed.** Both datagram types share port
  5566 (`spec/wire.md:3`), `link_stats` parses every packet as raw CSI
  and counts the failures file-wide (`wifisense/sources/recorder.py:840-851`),
  and `cli.py:223` prints them as "malformed (not raw CSI)". At 2 Hz
  (`spec/wire.md:19-20`) a 600 s capture should show ~1200 of them. If
  the count is far from that, something else is on the port.
- **Kernel socket drops are the gateway's loss, not the radio's.**
  `cli.py:208-211` prints them with that warning attached. Record them.

---

## 2. Ethics and consent

### 2.1 Self-only until the committee says otherwise

`SAANS_PLAN.md:196`: "Ethics committee approval before any human-subject
data collection (S5)." `CAMPUS_SENSING_PROMPT.md:173` repeats it and adds
signage.

Until that approval exists in writing, **the author is the only subject.**
Not a flatmate who offered. Not a labmate who is curious. Not a family
member. A willing person is not an approved protocol, and the fact that
someone said yes is the thing the committee exists to examine rather than
a substitute for it. A recording of anyone who is not the author, made
before approval, is not a dataset with a paperwork problem — it is data
that has to be deleted.

[UNVERIFIED: the institution's committee name, its forms and its
turnaround time are not recorded anywhere in this repository. Find them,
and write them into this section.]

### 2.2 Signs on the door

Put a sign on every instrumented space saying what is running, that it is
a WiFi sensing experiment, that no camera or microphone is in use, and
whom to contact. `CAMPUS_SENSING_PROMPT.md:173` requires it. Self-only
recording does not exempt the room: other people walk past doors.

### 2.3 What the consent must say

Written, before recording, and written for the self-only runs too. Writing
it for yourself is the rehearsal for writing it for someone else, and it
is where you find out which sentences you cannot actually write.

It must state:

1. **That raw CSI is being recorded, and what raw CSI is.** ADR-001:6-10:
   "Raw CSI from a room is enough to re-derive breathing, heartbeat and,
   with several links, movement through walls." The subject is consenting
   to the full channel measurement, not to a breathing number.
2. **That this is health data under India's DPDP Act 2023**, which carries
   notice, purpose and retention obligations (ADR-001:8-10,
   `SAANS_PLAN.md:193-195`).
3. **The purpose, narrowly.** Validating a breathing and presence
   estimator against a reference. Not "research", not "product
   development".
4. **Retention and deletion.** When the pcaps are destroyed and how.
   [UNVERIFIED: this repository sets no retention period for S5 research
   recordings. `CAMPUS_SENSING_PROMPT.md:134` gives at most 7 days for
   operational raw CSI and 30 days before events are aggregated into
   15-minute buckets. Seven days is almost certainly too short for a
   dataset you intend to re-analyse. Pick a number, write it here, and
   say why.]
5. **That withdrawal deletes the recordings**, including derived feature
   streams, and that the subject does not have to give a reason.
6. **Who can open the files**, by name, and where they are stored.
7. **That this is not a medical device** and the numbers have no clinical
   meaning (`SAANS_PLAN.md:190-191`, `README.md:350-355`).
8. **That "no identities are stored" is not anonymity.** ADR-005:15 says
   outputs carry no identities, and that is true of the system. It is not
   true of this dataset: a single named person, in a known room, at a
   recorded time, with a take sheet. Say so. A consent form that implies
   otherwise is the one sentence in this document most likely to be
   wrong.

### 2.4 Raw mode is deliberately high-friction, and must be visible

ADR-001:23-24: "New analyses that need raw data require a debug
collection, recorded with `wifisense record`, under ethics approval. That
friction is the point."

This protocol runs with `SAANS_RAW_MODE=y`, which is the off-by-default
debug path (`Kconfig.projbuild:164-171`). The firmware sets the
`raw_mode` flag in every FeatureFrame while it is on, because "raw mode
must never be silent" (`Kconfig.projbuild:170-171`, `spec/wire.md:56`).

**Per-session check:** decode one FeatureFrame from the session's pcap and
confirm flag bit 3 is set. If raw CSI is on the wire and that bit is
clear, the session stops: that is a privacy defect in the firmware, not a
logging inconvenience, and it has to be fixed before more raw data is
collected.

### 2.5 Vitals on, heart off

`SAANS_VITALS_ENABLED` must be `y` for these recordings, or the node
writes 0.0 into every breathing field and there is nothing to score
(`Kconfig.projbuild:146-154`, `spec/wire.md:62-67`). Enabling it is the
consent decision of ADR-005:6-8, and for a self-only run the author is
both space owner and occupant. The written consent in §2.3 still gets
written.

`SAANS_HEART_ENABLED` stays `n`. `NOTES.md:187-188`: "Do not trust heart
rate without a pulse-oximeter comparison across subjects and postures."
There is no pulse oximeter in this protocol, so **no heart-rate number
comes out of it**, and none may appear in any table in §15.

### 2.6 Storage

Recordings live on an encrypted volume outside the repository. Not in a
cloud-synced folder, not in `git`, not in the scratch directory of a
shared machine. The take sheet may live in the repository; the pcaps may
not.

At 20 Hz with 56 subcarriers the raw stream is a 20-byte header
(`wifisense/sources/esp32_udp.py:41`) plus 56 int8 I/Q pairs
(`esp32_udp.py:18`) = 132 bytes of payload, ~160 bytes on the wire
with IPv4+UDP, ~176 bytes per pcap record: about **12.7 MB per hour**
(arithmetic, not a measurement). The full matrix in §6 is around 7.5
hours of room time, so roughly 100 MB. Disk is not a constraint here;
exposure is.

---

## 3. Equipment and configuration

### 3.1 Nodes

Two ESP32-S3 devkits. One built `SAANS_ROLE_TX`, one `SAANS_ROLE_RX`
(`Kconfig.projbuild:3-16`). The TX does no sensing; it exists to put
packets in the air at a steady rate.

| Setting | Value | Why |
|---|---|---|
| `SAANS_NODE_ID` | TX 1, RX 2 | Distinct; 255 is reserved (`Kconfig:23-24`) |
| `SAANS_WIFI_CHANNEL` | one value, same on both | ESP-NOW is connectionless; this is the only thing tying them together (`Kconfig:32-34`) |
| `SAANS_CSI_RATE_HZ` (RX) | 20 | `Kconfig:70-73` |
| `SAANS_TX_RATE_HZ` (TX) | 20 | Must equal the RX's rate. Beaconing faster scales the RX's sample clock, its vitals bands and its gap filler wrongly (`Kconfig:45-53`) |
| `SAANS_N_SUBCARRIERS` | 56 | `Kconfig:61-68` |
| `SAANS_FRAME_RATE_HZ` | 2 | `Kconfig:92-95`, `spec/wire.md:19-20` |
| `SAANS_CALIBRATION_S` | 30 | `Kconfig:99-106` |
| `SAANS_COMPUTE_BNR` | y | §5's null hunt reads `best_bnr`. Costs 26.9 KB per link (`Kconfig:116-119`); two links is fine |
| `SAANS_VITALS_ENABLED` | y | §2.5 |
| `SAANS_HEART_ENABLED` | n | §2.5 |
| `SAANS_RAW_MODE` | y | §2.4 |
| `SAANS_GATEWAY_IP` / `PORT` | the recording host, 5566 | `Kconfig:177-184` |
| `SAANS_TELEMETRY_S` | 60 | Free heap, uptime, reboot count (`Kconfig:205-212`) |

Record the firmware git commit hash in the take sheet. A protocol whose
results cannot be attributed to a build is a protocol that will be rerun.

### 3.2 Gateway

The recording host runs `wifisense record` on port 5566. Nothing else may
bind that port — `cli.py:188-194` catches `EADDRINUSE` and says so,
and `NOTES.md:340-342` records why `SO_REUSEADDR` was removed: with it, a
second listener silently took half the datagrams.

Run `chrony` or `ntpd` **in slew mode**, not step mode. pcap records are
stamped with `time.time_ns()` when the receive call returns
(`wifisense/sources/recorder.py:34-35`), so a clock step during a
recording shows up as one bogus inter-arrival gap
(`recorder.py:36-38`). Record `chronyc tracking` output at the start and
end of each session into the take sheet.

### 3.3 Everything else

- **A laser distance meter.** `docs/CAPABILITY_LIMITS.md:227` prices one
  at about ₹2,000 and the section's title is "ask, do not infer". Every
  distance in this protocol is measured, never estimated.
- **A tape measure** for heights.
- **A chair with a flat back** and a **mat** for the lying posture.
- **A metronome** with an audible click, able to run at 8, 12, 16 and 20
  beats per minute. One click per *inhale onset*, not one per half-cycle.
  Confirm by counting against a watch before the first session: at 8 BPM
  the clicks are 7.5 s apart.
- **A phone with a triaxial accelerometer**, logging at ≥ 50 Hz (§8).
- **A strap** to hold the phone flat against the sternum without the
  subject holding it — a hand on the chest is an arm in the link.

[UNVERIFIED: no accelerometer logging application is named anywhere in
this repository. Choose one that writes raw triaxial samples with
per-sample timestamps and applies no smoothing, filtering or step
detection, and name it in the take sheet. A logger that silently
low-passes its output will quietly become the thing you are measuring.]

---

## 4. Room and link geometry

One room, used for the whole matrix. Changing rooms mid-matrix makes
every comparison in §15 ambiguous.

Both node antennas at the same height `h`, measured with the tape and
written down. Choose `h` to match the sternum height of the subject when
seated, so the seated posture sits closest to the plane of the link.
[UNVERIFIED: no node height is specified anywhere in this repository.
Measure yours, record it, and record the subject's sternum height in each
of the three postures, because they differ by more than a Fresnel zone.]

Measure and record with the laser meter:

- the link length `L` (antenna to antenna),
- the position of each node relative to two walls,
- the room's extent,
- the clearance from each node to the nearest wall.

[UNVERIFIED: no minimum wall clearance is specified in this repository.
Keep the nodes and the subject at least a metre from walls if the room
allows, record the actual clearances, and treat any result from a tighter
room as a different room.]

**These coordinates go in the log and are not used to predict anything.**
`docs/CAPABILITY_LIMITS.md:237-240` measured (SYNTHETIC, against this
repo) that with node positions off by 5 mm the Fresnel coverage map's
correlation with truth falls to 0.87; at 1 cm, 0.65; at 2 cm, 0.27 — and
"a human with a tape measure does not place a node to 5 mm". The map is
honest in the Explorer only because the simulator knows its positions
exactly. In a real room it predicts nothing. The null in §5 is **found by
measurement**, not computed from the survey.

The survey is recorded anyway, for three reasons: so a result can be
attributed to a geometry, so a node that moves between sessions is
detectable, and so the §12 numbers can be read as applying to a room of
known size rather than to rooms in general.

---

## 5. The three positions, and why one is a null

### 5.1 Why a null is in the protocol rather than avoided

A Fresnel-zone blind spot is the measured worst case of this system, and
it is the one place where the confidence gate — the feature the whole
project rests on — does not save you.

What the repository has measured, all SYNTHETIC:

- At whole-band nulls, breathing is within 0.5 BPM in **64.5%** of
  readings by default and **68.1%** with RespirFi selection, against
  **95.1% / 95.7%** at uniformly placed positions (`NOTES.md:295-296`).
- "Whole-band blind spots stay at ~2/3 accuracy for every real method, as
  eq. 6 predicts. Moving the node, or a second link, is the fix, not
  subcarrier selection." (`NOTES.md:316-318`)
- The failure is typically a **doubled rate** (`README.md:95-96`), and it
  is confident: readings at confidence ≥ 0.7 that are ≥ 1 BPM off run at
  **4.2%** of whole-band-null readings by default, **2.7%** with RespirFi
  (`NOTES.md:300-301`). `docs/CAPABILITY_LIMITS.md:326-331` records the
  Explorer review watching it report a doubled rate at **confidence
  1.00** — "unanimous about a doubled rate".
- The physics, asserted from first principles in a test:
  `tests/test_synthetic.py:76-78` — "On a Fresnel-zone boundary the
  first-order response vanishes and the amplitude oscillates at twice the
  breathing rate; one zone over, it doesn't."
- An extended torso does not escape it. `NOTES.md:268-271`: a torso
  facing the link "lies along the Fresnel ellipse tangent, where excess
  path changes only to second order, and it moves as one block, so it has
  one net reflection vector: the null moves, it does not disappear".
- Subcarrier diversity cannot escape it at this bandwidth. RespirFi's own
  eq. 6 needs **4.4 m** of excess path at the span an ESP32 reports
  (`NOTES.md:214-218`).

Now the sampling argument. The headroom study (`NOTES.md:253-258`,
SYNTHETIC) found whole-band nulls at **2.0% to 6.0%** of 300 uniformly
sampled positions per room type. A protocol that places a chair wherever
it looks sensible and records 108 takes would, at that rate, land on a
null a handful of times, at unrecorded positions, scattered across
unrelated strata, and with no label. The resulting pooled number would be
a good number with a few points of unexplained tail, and every one of
those points would be the known failure mode being quietly diluted.

So: **one of the three positions at every posture and distance is a
deliberately located null.** Three consequences follow, and all three are
the point.

1. The null stratum has enough takes to produce a rate rather than an
   anecdote: 36 takes of the 108.
2. Every result in §15 is reported **stratified by position class**, so
   the clear-geometry number and the blind-spot number are both visible
   and neither is hidden inside the other.
3. The pooled number is explicitly a weighted average of strata whose
   weights this protocol chose. It is not an estimate of "accuracy in a
   room", because the fraction of real occupied positions that fall at a
   null in a real room is not known. Say that wherever the pooled number
   is quoted.

A protocol that only samples good geometry produces a number that will
not survive a real room, and it produces it in the most damaging possible
way: it looks excellent, it is reproducible, and the first deployment
where someone's favourite chair happens to sit in a null reports their
breathing at twice its rate, confidently.

### 5.2 How the three positions are found

They are the output of a measurement, not a calculation. Run this once
per (posture, distance) pair — nine times for the full matrix.

The subject is in the posture under test, at the measured perpendicular
distance from the link, phone on chest, breathing to a **12 BPM**
metronome. 12 BPM is chosen deliberately: it is the rate at which the
doubling signature is observable in band (§13.2).

**Coarse pass.** Starting at a marked origin, move the chair (or mat)
along a line parallel to the line of sight, in **20 cm** steps, for nine
stops spanning 1.6 m. Hold each stop for **40 s** — `best_bnr` is the
highest per-subcarrier breathing-to-noise ratio over the last 30 s
(`Kconfig.projbuild:112-113`), so 40 s gives one full window plus
settling. Read `best_bnr` and the reported `breathing_bpm` from the live
FeatureFrame stream at each stop and write both down.

**Fine pass.** Around the stop with the lowest `best_bnr`, or the stop
reporting ~24 BPM against a 12 BPM pace, step **5 cm** over ±20 cm: nine
more stops, 40 s each.

**The three positions are then:**

| Label | Definition |
|---|---|
| `C1` | the stop with the **highest** `best_bnr` found in the hunt |
| `C2` | a stop roughly midway in `best_bnr` between `C1` and `N` |
| `N` | the stop with the **lowest** `best_bnr`, or the stop reporting a doubled rate |

Mark all three on the floor with tape and record their offsets from the
origin, to the centimetre, in the take sheet. The hunt itself is a
result: the `best_bnr`-versus-offset curve is the first measurement of
real Fresnel structure anywhere in this project and goes into Table F.

Two cautions that will otherwise cost a session:

- **`best_bnr = 0.0` is ambiguous and must not be read as "null".**
  `spec/wire.md:41` and `Kconfig.projbuild:125-136` define 0.0 as
  *unknown*: before the first window has filled, on a node that does not
  compute it, and on a channel that does not vary at all. There is no
  flag bit separating them. Discard the first 30 s after any stop and
  treat a persistent 0.0 as "the field is not working", not as a reading.
- **Redo the hunt for each posture.** The null depends on the excess path
  from TX to chest to RX, and the chest of a lying subject is tens of
  centimetres lower than that of a seated one. A position that is a null
  seated is not a null lying, and labelling it as one would put the wrong
  takes in the null stratum. The hunt is about 20 minutes; doing it nine
  times is about three hours and is not optional.

[UNVERIFIED: the position step corresponding to one Fresnel zone has
never been computed or measured for a real room in this repository. The
20 cm / 5 cm steps and the 1.6 m span above are a **search range**
chosen so the fine pass is finer than λ = 12.3 cm
(`docs/CAPABILITY_LIMITS.md:178`) and the coarse span is several times
larger. They are not a prediction of where the null is. If the coarse
pass shows no variation in `best_bnr` at all, widen the span before
concluding there is no null.]

---

## 6. The recording matrix

Three postures × three distances × three positions × four rates = **108
takes of 2 minutes each**.

| Axis | Values | Code |
|---|---|---|
| Posture | seated (upright, back against the chair, hands on thighs) | `SE` |
| | lying (supine on a mat, arms at sides) | `LY` |
| | standing still (feet shoulder-width, arms at sides, eyes open) | `ST` |
| Distance | 1 m, 2 m, 3 m | `1`, `2`, `3` |
| Position | clear-best, clear-mid, null | `C1`, `C2`, `N` |
| Paced rate | 8, 12, 16, 20 BPM | `08`, `12`, `16`, `20` |

**Distance** is the perpendicular distance from the subject's sternum to
the line of sight between the two node antennas, in the horizontal plane,
measured at the link midpoint with the laser meter. Not the distance to a
node.

**Take id:** `T<nnn>-<posture><distance>-<position><rate>`, so
`T041-SE2-N16` is take 41: seated, 2 m, null position, 16 BPM.

**Files:** `s5/<YYYYMMDD>/<block_id>.pcap`,
`s5/<YYYYMMDD>/<block_id>.accel.csv`, `s5/<YYYYMMDD>/takes.csv`.
A block is one position's four rates recorded continuously (§7).

### 6.1 Rate order

Rate order is rotated per position so that drift in the room, fatigue in
the subject and filter state do not correlate with rate. No random number
generator; the cycle is deterministic and is written in the take sheet:

| Position within its pair | Order |
|---|---|
| 1st (`C1`) | 8, 12, 16, 20 |
| 2nd (`C2`) | 12, 16, 20, 8 |
| 3rd (`N`) | 16, 20, 8, 12 |

### 6.2 Time, honestly

Per (posture, distance) pair: one null hunt (~20 min) plus three
positions × (60 s empty segment + 9 min recording block) = ~50 minutes.
Nine pairs is about **7.5 hours of room time**, plus per-session setup
and calibration.

A pair is **never split across sessions**. Its three positions must share
one presence calibration and one null hunt, or the position classes are
not comparable to each other and the stratification in §15 is meaningless.
Two pairs per session is about two hours and is a reasonable day.

### 6.3 The minimum publishable set

Nothing from this protocol is published until at least one complete pair
exists, **including its null position and all four rates**. A pair with
`C1` and `C2` recorded and `N` missing is a selected sample, and
publishing it would reproduce exactly the failure §5.1 exists to prevent.

The first complete row to aim for is **seated at all three distances** —
three pairs, 36 takes, about 2.5 hours — because seated is the posture
the P1 gate is written against (`CAMPUS_SENSING_PROMPT.md:157`) and the
one a real deployment sees most.

---

## 7. The per-block procedure

A **block** is one position, four rates, recorded as a single continuous
pcap. Recording continuously matters: the breathing bandpass sits at
0.1 Hz and takes tens of seconds to settle (`NOTES.md:161-163`), so
stopping and starting between rates would pay that cost four times.

### 7.1 Once per session

1. Nodes powered, in their surveyed positions, untouched since the survey.
2. Room **empty**. Start `wifisense record`. Wait for the node to
   calibrate: 50 s of filter warmup then 30 s of measurement
   (`README.md:294-296`, `Kconfig.projbuild:99-106`). Confirm the
   `calibrated` flag (bit 0, `spec/wire.md:53`) is set and that
   `calib_age_s` is not `0xFFFFFFFF` (`spec/wire.md:42`).
   **Calibrating with anyone in the room learns them as the baseline and
   the room then reads as permanently empty** (`README.md:294-297`).
3. Confirm `raw_mode` (bit 3) is set — §2.4.
4. Record `chronyc tracking`, the firmware commit, the node ids, the
   channel, and the room temperature if you have a thermometer.
5. **Empty-room segment, 5 minutes.** This is the session's
   false-occupancy sample and its noise reference.

### 7.2 Once per position

6. Move the chair or mat to the taped position. Leave the room.
7. **Empty segment, 60 s, with the furniture in its new place.** Three
   purposes: a per-position false-occupancy sample; a check that moving
   furniture did not break the calibration; and a clean settling window
   for the static suppressor.
   [UNVERIFIED: whether moving a chair between positions invalidates the
   empty-room presence calibration has not been measured anywhere in this
   repository. This segment is how you find out. If `motion_energy`
   during it has moved substantially against the session's 5-minute empty
   segment, recalibrate and record that you did.]
8. Enter, take the posture, fit the phone on the sternum, start the
   accelerometer logger **first**.

### 7.3 The block itself

9. **Sync marker, in.** Three deliberate arm raises — both arms from the
   sides to overhead and back down, about one second each, one second
   apart. Lying: raise both arms vertically off the mat and lower them.
   This is the only large movement in the block and it is what §9 aligns
   on.
10. **Lead-in, 60 s.** Metronome at the block's first rate. Subject
    settles, breathes to the pace, does not move otherwise. This covers
    the filter settling.
11. **Four paced segments, 120 s each, back to back.** Change the
    metronome at each boundary; do not pause, do not move. The subject
    breathes **normally in depth** — only the timing is paced.
12. **Sync marker, out.** Three arm raises, as in step 9.
13. Stop the pcap. Stop the accelerometer logger **last**, so the CSI
    recording lies strictly inside the accelerometer recording.
14. Leave the room before the next position.

Block length: 60 + 4 × 120 = **540 s** plus markers, about 9.5 minutes.

### 7.4 Abandoned takes

**Every abandoned block is logged, with its reason, and its files are
kept.** A cough, a phone notification, someone opening the door, the
subject losing the pace, lightheadedness — write it down, mark the block
`abandoned`, record the reason, and re-record.

This is not paperwork. A protocol that lets a bad block disappear
silently is a protocol that selects its own dataset, and the selection
runs in exactly the direction that flatters the result. The abandoned
count and its reasons go in Table G.

### 7.5 Safety

Paced breathing at 8 BPM is one breath every 7.5 seconds. If the subject
compensates by breathing more deeply, lightheadedness follows. Instruct:
normal depth, only the timing is paced. If the subject feels lightheaded,
dizzy or tingly, **the block ends immediately**, is marked abandoned with
that reason, and is re-recorded later or dropped. Standing-still blocks
end immediately on any unsteadiness; standing still for ten minutes is
harder than it sounds.

---

## 8. The reference signal

### 8.1 What it is

A phone, flat against the sternum, screen outward, long axis along the
body head-to-toe, held by a strap. It logs triaxial accelerometer samples
at **≥ 50 Hz** with per-sample timestamps.

Why ≥ 50 Hz: the CSI stream is 20 Hz nominal (`Kconfig:70-73`) and the
estimator's decimated analysis rate is 2 Hz (`wifisense/vitals.py:59-60`:
`decimation = int(20 / (4 × 0.5)) = 10`). At 8–20 BPM the signal of
interest is 0.133–0.333 Hz. 50 Hz gives at least 150 samples per breath
at the fastest paced rate, so the reference is nowhere near the limiting
bandwidth and no result can be blamed on it.

Record the phone's orientation in the take sheet.

### 8.2 How the reference rate is derived

Band-pass each of the three axes at **0.1–0.5 Hz** using the repository's
own `BiquadBandpass` (`wifisense/dsp.py`), take the axis with the largest
band RMS, then run the repository's own `band_peak` over it
(`wifisense/vitals.py:90-96`) with the same band and the same resolution
rule.

Using the project's own estimator on the reference is deliberate. It
means the two rates are produced by identical arithmetic, so a difference
between them is a difference in the *signal*, not in two estimators that
were never compared. It also means no new estimator has to be validated
before this protocol can produce a number.

A phone on the chest measures chest-wall **tilt** relative to gravity, so
a single axis carrying the gravity projection is the signal; the vector
magnitude is not, because it is nearly constant under pure rotation.
Picking the axis by band energy avoids depending on how the strap sat.

### 8.3 What the reference is not

It is a motion reference, not a respiration reference. It cannot tell a
breath from a chest movement of any other cause. It will track a cough, a
swallow or a shift as a large transient. It says nothing about tidal
volume or airflow.

The P1 gate (`CAMPUS_SENSING_PROMPT.md:157`) says "within ±1 BPM of the
belt". A respiratory inductance belt is the better instrument and is what
that sentence means. [UNVERIFIED: no belt is specified, sourced or priced
anywhere in this repository.] The phone is the substitution this protocol
actually makes, and the gate's wording should be amended to say what the
reference was, rather than leaving a reader to assume a belt.

### 8.4 The reference has its own error, and this protocol measures it

The metronome is a second, independent reference: it is what the subject
was *told* to do. The accelerometer is what the chest *did*.

For every take, report `accel_bpm − paced_bpm`. The distribution of its
absolute value across all 108 takes is the reference's own self-consistency
and it is a **required output** (Table B, bottom rows), because it bounds
how much of the CSI-versus-reference error belongs to the reference.

Where the two disagree by more than **0.5 BPM** on a take the subject
reports having followed, flag the take. The accelerometer wins — it is
what the chest did — but the flag goes in the take sheet, because a run
of such takes means either the pacing is not being followed or the
derivation in §8.2 is wrong, and those need telling apart.

### 8.5 Does the phone perturb the measurement?

A phone is a few hundred grams of metal and glass sitting exactly where
the measurement is being made. It adds a scatterer co-located with the
chest, and it moves with the chest, so it is not something the static
suppressor removes.

[UNVERIFIED: whether a phone on the chest changes the CSI breathing
signal has not been measured anywhere in this repository.]

**Control:** at each (posture, distance) pair, at position `C1`, at
12 BPM, record one extra 120 s take with the phone **removed** and the
subject breathing to the same metronome. Nine control takes, 18 minutes
total. Compare `best_bnr` and the reported `breathing_bpm` against the
matching phone-present take. If they differ materially, the reference is
perturbing the thing it is referencing, and that has to be reported
beside every breathing number in §15 rather than discovered later.

---

## 9. Clock alignment

There are **three clocks** and none of them is shared.

1. **The node has no clock on the wire.** The raw-CSI header is magic,
   node id, antenna count, subcarrier count, centre frequency, sequence
   number, RSSI, noise floor, PPDU type and flags
   (`wifisense/sources/esp32_udp.py:6-19`). There is no timestamp field.
   Ordering comes from the 32-bit sequence number at bytes 12..15.
2. **The gateway stamps on receive.** pcap records carry
   `time.time_ns()` taken when the receive call returns
   (`wifisense/sources/recorder.py:34-35`), in user space, including
   scheduling latency "typically well under 1 ms" (`recorder.py:39-40`).
3. **The replay time axis is not wall clock at all.**
   `wifisense/pipeline.py:115` sets `timestamp_s = self._samples /
   self.sample_rate_hz` — frame count over the *assumed* rate, counting
   gap-filled frames.
4. **The phone has its own clock**, with its own offset and its own drift.

Two separate corrections are needed, and they fix different things.

### 9.1 Gateway wall clock ↔ pipeline time axis

This is a **scale** error, and it biases every rate in the dataset in the
same direction.

If the radio actually delivers 19.8 Hz and the pipeline is told 20 Hz,
every reported BPM is multiplied by 20/19.8 = 1.0101. A 20 BPM breather
reads 20.2 BPM. That is inside a ±1 BPM criterion and therefore invisible
in the headline, while being a systematic bias across all 108 takes —
which is worse than noise, because noise averages out and this does not.

The repository already has the failure on record in a sharper form:
feeding the survivors of 15% packet loss in directly made a 15 BPM
breather read **17.80 BPM at confidence 0.98** (`NOTES.md:331-335`,
SYNTHETIC), fixed to 14.97 by the gap filler (`NOTES.md:339-340`).

**Do, per block:**

```bash
python3 -m wifisense stats s5/<date>/<block>.pcap
python3 -m wifisense replay s5/<date>/<block>.pcap --node 2 --rate auto --max-gap-s 1.0
```

`--rate auto` takes `1000 / dt_median_ms` (`wifisense/cli.py:288-298`).
The **median** inter-arrival is the right statistic and is why `auto`
uses it: it is unaffected by loss as long as most frames arrive, whereas
`rate_hz` (packets over span) is depressed by every lost packet. Do not
use `rate_hz` as the replay rate.

Record the measured rate in the take sheet. Flag — do not discard — any
block whose measured rate deviates from 20 Hz by more than 1%, and use
the measured rate.

### 9.2 Phone clock ↔ gateway wall clock

This is an **offset plus skew**, and it is fixed by the sync markers.

**Coarse bound, per session.** Photograph the phone's clock beside the
gateway's clock at session start. This bounds the offset to about a
second. It is a sanity check, not the alignment.

**Fine alignment, per block.** Extract the marker onsets from both
streams:

- *Accelerometer side:* band-pass 1–4 Hz, take the envelope, and take each
  marker's onset as the 50% point of its rise. At ≥ 50 Hz the sample
  spacing is ≤ 20 ms.
- *CSI side:* replay the raw CSI, band-pass the fused residual 1–4 Hz —
  the gross-motion band (`spec/wire.md:35`) — take the envelope, and take
  the same 50% rise point. At 20 Hz the sample spacing is 50 ms.

  Do **not** use the `gross_motion` field from the FeatureFrames for
  this: it is an RMS over the last 1.0 s sampled at 2 Hz
  (`spec/wire.md:35`, `spec/wire.md:73-75`), which localises a marker to
  roughly ±0.5 s. The 20 Hz raw stream localises it ten times better.

Fit a straight line — offset and skew — through the six marker pairs
(three in, three out). Record the fitted offset, the fitted skew, and the
fit's residual.

**Resolution of the fit.** Marker onsets are good to about 100 ms,
dominated by the CSI side. The lever arm between the in and out markers
is about 560 s. So the skew is resolved to roughly
2 × 0.1 / 560 ≈ **0.04%** (arithmetic).

**Rejection threshold.** A skew ε scales a measured rate by (1 + ε). To
keep the induced error below 0.1 BPM at the fastest paced rate, 20 BPM,
needs ε < 0.005, i.e. **0.5%** — which over a 120 s segment is 0.6 s of
relative drift (arithmetic). Reject a block whose fitted skew exceeds
0.5%, or whose fit residual exceeds 100 ms, and re-record it. The
resolution above is more than an order of magnitude finer than the
threshold, so the fit is capable of deciding this.

### 9.3 How much of this actually matters

Say it plainly so nobody over-engineers it: **for comparing rates, a
constant offset does not matter at all.** A 300 ms offset between two
streams changes neither stream's frequency content. The offset matters
for *windowing* — deciding which 60 s of accelerometer corresponds to
which 60 s of CSI — where being a second out at a rate boundary puts the
wrong breathing rate in the window. The skew matters because it is a
scale error on the rate itself.

That is why the rejection threshold is on skew and the tolerance on
offset is loose. It is also why §9.1 is the more important of the two
corrections: it is a scale error that the markers cannot see, because both
marker times are read off the same pcap clock.

---

## 10. What gets logged

One row per take in `takes.csv`, one row per block in `blocks.csv`, one
row per session in `sessions.csv`. Every field is filled; a blank is a
defect.

**`sessions.csv`:** session id, date, operator, room, firmware commit,
node ids and roles, WiFi channel, node height `h`, link length `L`, node
positions, room extent, wall clearances, `chronyc tracking` at start and
end, calibration completion time, `motion_energy` over the 5-minute empty
segment, `raw_mode` flag confirmed, notes.

**`blocks.csv`:** block id, session id, posture, distance, position label
and class, taped offset from the origin, subject sternum height in that
posture, phone present, phone orientation, logger application and its
configured rate, pcap filename, accel filename, measured CSI rate from
`stats`, loss %, resets, `dt_p95_ms`, kernel drops, malformed count,
fitted clock offset, fitted skew, fit residual, marker count found,
abandoned (y/n) and reason.

**`takes.csv`:** take id, block id, paced BPM, rate order index, segment
start and end offsets within the block, accel-derived BPM, |accel −
paced|, pace-followed flag, anything the subject noticed.

**On subject identity.** Use a subject code, not a name, even for
self-only recording. ADR-005:15 keeps identities out of system outputs;
this is a research dataset and the code anonymises nothing — §2.3 item 8
says so. The code is a habit being established now, before the protocol
covers anyone else.

---

## 11. Scoring

### 11.1 What is scored

The scored span of each take is the **last 75 s of its 120 s segment**.

The reason is the estimator's own window. `wifisense/vitals.py:40-41`
sets `min_window_s = MIN_CYCLES / f_low = 3.0 / 0.1 = 30 s` and
`window_s = min_window_s × 1.5 = 45 s`. The analysis history is 45 s
long, so for 45 s after a metronome change the window still contains the
previous rate. Discard those 45 s uniformly — including for the first
segment of a block, whose window is already warm from the lead-in — so
that every take has the same scoring rule and the same N.

Two scoring cadences, both reported:

- **Frame level**, at the FeatureFrame rate of 2 Hz (`spec/wire.md:19-20`).
  75 s gives 150 scored frames per take, 16,200 across the matrix. This
  is where median and p90 come from.
- **Window level**, non-overlapping **60 s** windows, which is the
  criterion the P1 gate is written in ("≥ 80% of 60 s windows",
  `CAMPUS_SENSING_PROMPT.md:157`). A 75 s scored span yields exactly one
  clean 60 s window per take, so the window-level N equals the take
  count: 108. That is a thin N and it is stated rather than inflated with
  overlapping windows.

### 11.2 Ground truth

- **Occupancy** is true from the entry marker to the exit marker of each
  block, and false during the session's 5-minute and each position's 60 s
  empty segments. It comes from the take sheet, not from any detector.
- **Breathing rate** is the accelerometer-derived rate (§8.2) over the
  same window, with the paced rate as the cross-check of §8.4.

### 11.3 Grid resolution

With `window_s = 45 s`, `resolution_hz = max(1/(4 × 45), 0.4/400) =
0.00556 Hz = 0.333 BPM` (`wifisense/vitals.py:65-68`), refined by
parabolic interpolation. A ±1 BPM criterion is therefore three grid steps
wide — comfortably above the grid, so a near-miss is a real estimation
error and not a quantisation artefact.

### 11.4 The tool

No script in this repository scores a recording against an external
reference. `evaluate.py` scores synthetic scenes against the simulator's
own ground truth, which is a different job. A new analysis script has to
be written for §12, and it needs one thing the CLI does not currently
expose — see §16.

---

## 12. The gates to report

ADR-005:16-17 is binding and is the reason the list below is shaped the
way it is: "Every detector reports both its miss rate and its false-alarm
rate per hour; publishing one without the other is not allowed."

Every one of these is reported **per stratum** — posture × distance ×
position class — and then pooled, with the pooled number carrying the
sentence from §5.1 about what it weights.

### 12.1 Presence

| Gate | Definition |
|---|---|
| **Presence F1** | Over all scored frames plus all empty segments, with occupancy from the take sheet. Report precision and recall separately beside it; F1 alone hides which side failed. |
| **Still-person false-empty rate** | Fraction of scored frames in occupied takes where the `occupied` flag (`spec/wire.md:54`) is clear. This is the miss rate. |
| **Empty-room false-occupied rate** | Fraction of frames in empty segments where `occupied` is set. This is the false-alarm rate, and ADR-005 does not permit reporting the previous row without it. |

Comparators, all SYNTHETIC and from the S3 held-out simulation at one
link: occupancy accuracy 98.2%, still person read as empty 2.1%, empty
room read as occupied 1.0% (`NOTES.md:468-470`).

### 12.2 Breathing

Reported **twice**: at confidence ≥ 0.4 and ungated. 0.40 is the
`DEGRADED` boundary in `wifisense/types.py:47`; 0.70 is `VALID` at
`types.py:45`. Report the ≥ 0.7 slice as well where the N supports it.

| Gate | Definition |
|---|---|
| **Median absolute error** | BPM, against the accelerometer rate. |
| **p90 absolute error** | BPM. |
| **% within 1 BPM** | Frame level and window level, both. |
| **Abstention rate, low confidence** | Fraction of scored frames where `breathing_conf` < 0.4. |
| **Abstention rate, presence-suppressed** | Fraction of scored frames where no breathing estimate exists because presence said empty — see below. |

**The two abstentions are different failures and must not be summed.**
`wifisense/pipeline.py:92-95`, with `suppress_vitals_when_empty` defaulting
to `True` at `pipeline.py:52`, returns `VitalEstimate.unavailable()` for
breathing whenever presence says the room is empty. So a still person
misread as absent does not merely cost a presence point — it deletes the
breathing estimate that would have proved they were there.
`NOTES.md:553-556` names the consequence: "A lost BPM here is a lost
person, not a lost signal — which is only visible because presence is
reported beside it." A single pooled "no answer" rate would hide which of
the two happened, and they have opposite fixes.

**Why abstention is a gate and not a footnote.** Every gated accuracy
figure can be made to look perfect by abstaining more. A gated number
published without its abstention rate is not a measurement, it is a
choice of threshold. Report them adjacent, always.

Comparators, SYNTHETIC, S3 held-out at one link: breathing reported for a
still breather 93.6%, within 1 BPM 93.9% (`NOTES.md:471-472`). Note those
are coverage and accuracy over simulated timelines, not the same strata
as here. At whole-band nulls, the comparator is 64.5% / 68.1% within
**0.5** BPM (`NOTES.md:296`) — a different criterion, and it must not be
read against a 1 BPM column.

### 12.3 Double-rate failures, counted separately

See §13 for why.

| Gate | Definition |
|---|---|
| **Double-rate failures** | Scored frames where \|f_est − 2 × f_ref\| ≤ 1 BPM. Count and rate, with the denominator stated. |
| **…of which confident** | The subset at `breathing_conf` ≥ 0.7. |
| **Half-rate failures** | \|f_est − 0.5 × f_ref\| ≤ 1 BPM, own column. |
| **Observable denominator** | Double-rate frames are counted over the **8 and 12 BPM takes only** — §13.2. |

Comparator, SYNTHETIC: confident-and-wrong readings at whole-band nulls,
4.2% by default and 2.7% with RespirFi (`NOTES.md:300-301`).

### 12.4 Reference self-consistency

Median and p90 of |accel rate − paced rate| across all takes (§8.4), and
the count of takes flagged as pace-not-followed. Without this the
breathing error above cannot be attributed between the system and its
reference.

### 12.5 Link health

Per block, from `wifisense stats`: achieved rate, loss %, resets,
`dt_med_ms`, `dt_p95_ms`, `dt_max_ms`, kernel drops. A block below 10 Hz
sustained does not support vitals (`README.md:271`) and its breathing rows
are reported separately rather than pooled.

---

## 13. Why double-rate failures are counted separately

### 13.1 It is a different quantity, not a large error

At a Fresnel-zone boundary the first-order response of amplitude to chest
displacement vanishes, the second-order term dominates, and the amplitude
genuinely oscillates at twice the breathing rate. That is asserted from
first principles in `tests/test_synthetic.py:76-78` and the test
demonstrates both halves: at the null the dominant frequency is 2 × f_b,
one zone over it is f_b.

So when the estimator reports a doubled rate it is **not malfunctioning**.
It is correctly reporting the dominant frequency of the signal it was
handed. The DSP is right; the mapping from chest motion to amplitude is
the thing that broke. Pooling that into an error distribution labels a
named, reproducible, geometric effect as estimator noise, which is both
false and useless — noise gets fixed by averaging longer, and this does
not.

Four further reasons, each of which independently justifies its own
column:

**It is confident, so the gate does not catch it.** Readings at
confidence ≥ 0.7 that are ≥ 1 BPM off run at 4.2% of whole-band-null
readings, 2.7% with RespirFi (`NOTES.md:300-301`, SYNTHETIC), and
`docs/CAPABILITY_LIMITS.md:326-331` records it happening at **confidence
1.00**. The abstention mechanism that makes every other number in this
project defensible is blind to this one.

**It hides in the statistics.** At a few percent of readings it does not
move a median at all, and whether p90 catches it depends entirely on how
many null takes are in the pool — which is an arbitrary choice of this
protocol. A failure whose visibility depends on the sampling design is a
failure that must be counted, not estimated.

**Its operational consequence is categorical, not proportional.** The
rate alarm fires on a sustained rate outside 8–24 BPM
(`SAANS_PLAN.md:38`). Doubling maps the paced rates as: 8 → 16 (inside
the band, silent), 12 → 24 (at the edge), 16 → 32 (alarm), 20 → 40
(alarm). The protocol's rate span deliberately brackets the point where a
doubled reading starts to page somebody.

**It is the known worst case of the whole system**, and §5.1 built the
geometry of this protocol around provoking it. Having gone to the trouble
of provoking it, averaging it away would be perverse.

### 13.2 It can only be observed at 8 and 12 BPM

This follows from the band and has to be built into the counting.

`BreathingExtractor` scans 0.1–0.5 Hz, which is 6–30 BPM
(`wifisense/vitals.py:122`, `vitals.py:132-133`). Doubling the four paced
rates (arithmetic):

| Paced | Doubled | In the 6–30 BPM scan? |
|---|---|---|
| 8 BPM (0.133 Hz) | 16 BPM (0.267 Hz) | yes |
| 12 BPM (0.200 Hz) | 24 BPM (0.400 Hz) | yes |
| 16 BPM (0.267 Hz) | 32 BPM (0.533 Hz) | **no** |
| 20 BPM (0.333 Hz) | 40 BPM (0.667 Hz) | **no** |

At 16 and 20 BPM the doubled component lands outside the scan grid, so
the estimator **cannot report it** no matter how strongly the channel
produces it. The bandpass is a single biquad at 6 dB/octave
(`NOTES.md:73-74`), so the component is attenuated rather than removed —
it is still in the filtered signal, just not in the search range.

Two consequences:

1. The double-rate rate is computed over the **8 and 12 BPM takes only**,
   with that denominator printed next to it. Computing it over all 108
   takes would halve it by construction.
2. This is a **falsifiable prediction** of the protocol. If the real data
   shows double-rate failures at 8 and 12 BPM and none at 16 and 20, the
   mechanism in `tests/test_synthetic.py:76-78` is confirmed on hardware. If it
   shows them at 16 and 20 as well, something other than the modelled
   mechanism is operating and that is a finding worth more than the rest
   of the table.

What the estimator does instead at a 16 or 20 BPM null is an open
question. It has to report something in band, and whether that is the
true fundamental (now weaker than a component it cannot see) or a
spurious in-band peak is exactly what these takes measure.

### 13.3 Two other predictions worth recording in advance

Writing predictions down before the data exists is what makes the data
able to contradict them.

- **Coverage should fall with distance.** The body's bistatic path
  contributes amplitude proportional to 1/(d₁d₂) — the simulator's
  bistatic body path is `sqrt(sigma/4pi)/(d1 d2)`
  (`NOTES.md:240-241`) — so at 3 m from the link the breathing signal is
  weaker and the abstention rate of §12.2 should rise. If it does not,
  either the distances are not doing what the geometry says or something
  else dominates.
- **8 BPM should be the hardest rate.** Its fundamental at 0.1333 Hz sits
  near the 0.1 Hz band edge, where the biquad's −3 dB point lands exactly
  (`NOTES.md:69-71`), and 30 s of history covers only four cycles against
  the `MIN_CYCLES = 3.0` floor (`wifisense/vitals.py:15`, `vitals.py:40`).
  Expect more abstention at 8 BPM than at 12–20 across every stratum.

---

## 14. The 48-hour presence run

The presence half of the S5 gate is written "over 48 h"
(`CAMPUS_SENSING_PROMPT.md:157`) and the matrix in §6 is about 7.5 hours
of room time, almost all of it occupied. It cannot produce that number.

**Separate run, same room, after at least one complete pair from §6 has
been scored:**

- Nodes as configured in §3.1, but **`SAANS_RAW_MODE=n`**. FeatureFrames
  only. 68 bytes at 2 Hz is about 0.7 MB per hour (arithmetic), so 48 h is
  roughly 33 MB — and more to the point, it keeps 48 hours of raw channel
  measurement of someone's living space from existing at all. ADR-001's
  friction applies most strongly to the longest recordings.
- Continuous `wifisense record` for 48 h, with the author living normally.
- A written occupancy log: entry and exit times to the nearest minute,
  kept by hand, including the gaps where the author was asleep in the
  room (occupied, still) and away (empty).
- Score presence F1, still-person false-empty and empty-room
  false-occupied against that log, and report false alarms per hour as
  ADR-005:16-17 requires.

Two things this run will surface that the matrix cannot: calibration
drift over a day (`calib_age_s`, `spec/wire.md:42`), and the behaviour of
the presence detector across a real diurnal cycle of temperature and
ambient WiFi traffic. Both are listed as system-health signals in
`SAANS_PLAN.md:52-58` and neither has ever been measured.

A hand-kept log to the nearest minute is a coarse ground truth and will
dominate the error at transitions. Score transitions separately from
steady-state, and say which is which.

---

## 15. Results tables

Every table carries a `Source` column. `REAL` requires a pcap filename in
`Provenance`. `SYNTHETIC` requires a `file:line`. `—` means not measured
and is a publishable value.

### Table A — Presence

| Stratum | Takes | Frames | Precision | Recall | F1 | Still false-empty % | Empty false-occupied % | False alarms /h | Source | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|
| Seated 1 m clear (C1+C2) | — | — | — | — | — | — | — | — | — | — |
| Seated 1 m null (N) | — | — | — | — | — | — | — | — | — | — |
| Seated 2 m clear | — | — | — | — | — | — | — | — | — | — |
| Seated 2 m null | — | — | — | — | — | — | — | — | — | — |
| Seated 3 m clear | — | — | — | — | — | — | — | — | — | — |
| Seated 3 m null | — | — | — | — | — | — | — | — | — | — |
| Lying 1 m clear | — | — | — | — | — | — | — | — | — | — |
| Lying 1 m null | — | — | — | — | — | — | — | — | — | — |
| Lying 2 m clear | — | — | — | — | — | — | — | — | — | — |
| Lying 2 m null | — | — | — | — | — | — | — | — | — | — |
| Lying 3 m clear | — | — | — | — | — | — | — | — | — | — |
| Lying 3 m null | — | — | — | — | — | — | — | — | — | — |
| Standing 1 m clear | — | — | — | — | — | — | — | — | — | — |
| Standing 1 m null | — | — | — | — | — | — | — | — | — | — |
| Standing 2 m clear | — | — | — | — | — | — | — | — | — | — |
| Standing 2 m null | — | — | — | — | — | — | — | — | — | — |
| Standing 3 m clear | — | — | — | — | — | — | — | — | — | — |
| Standing 3 m null | — | — | — | — | — | — | — | — | — | — |
| **Pooled (weights = this protocol's, not a room's)** | — | — | — | — | — | — | — | — | — | — |
| 48 h run (§14) | — | — | — | — | — | — | — | — | — | — |
| *Comparator: S3 sim, 1 link* | — | — | — | — | 98.2% accuracy | 2.1% | 1.0% | — | SYNTHETIC | `NOTES.md:468-470` |

### Table B — Breathing accuracy

Reference: phone accelerometer (§8), **not** a belt.

| Stratum | Takes | Frames | Median err BPM (c ≥ 0.4) | p90 err BPM (c ≥ 0.4) | % within 1 BPM (c ≥ 0.4) | Median err BPM (ungated) | p90 err BPM (ungated) | % within 1 BPM (ungated) | % of 60 s windows within 1 BPM (c ≥ 0.4) | Source | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Seated 1 m clear | — | — | — | — | — | — | — | — | — | — | — |
| Seated 1 m null | — | — | — | — | — | — | — | — | — | — | — |
| Seated 2 m clear | — | — | — | — | — | — | — | — | — | — | — |
| Seated 2 m null | — | — | — | — | — | — | — | — | — | — | — |
| Seated 3 m clear | — | — | — | — | — | — | — | — | — | — | — |
| Seated 3 m null | — | — | — | — | — | — | — | — | — | — | — |
| Lying, all distances, clear | — | — | — | — | — | — | — | — | — | — | — |
| Lying, all distances, null | — | — | — | — | — | — | — | — | — | — | — |
| Standing, all distances, clear | — | — | — | — | — | — | — | — | — | — | — |
| Standing, all distances, null | — | — | — | — | — | — | — | — | — | — | — |
| By rate: 8 BPM | — | — | — | — | — | — | — | — | — | — | — |
| By rate: 12 BPM | — | — | — | — | — | — | — | — | — | — | — |
| By rate: 16 BPM | — | — | — | — | — | — | — | — | — | — | — |
| By rate: 20 BPM | — | — | — | — | — | — | — | — | — | — | — |
| **Pooled** | — | — | — | — | — | — | — | — | — | — | — |
| Phone-absent controls (§8.5) | — | — | — | — | — | — | — | — | — | — | — |
| **Reference self-consistency** \|accel − paced\|: median / p90 / takes flagged | — | — | — | — | — | — | — | — | — | — | — |
| *Comparator: S3 sim, 1 link, still breather* | — | — | — | — | 93.9% | — | — | — | — | SYNTHETIC | `NOTES.md:472` |
| *Comparator: sim, whole-band null, within **0.5** BPM* | — | — | — | — | 64.5% default / 68.1% RespirFi | — | — | — | — | SYNTHETIC | `NOTES.md:296` |
| *Comparator: sim, uniform placement, within **0.5** BPM* | — | — | — | — | 95.1% / 95.7% | — | — | — | — | SYNTHETIC | `NOTES.md:295-296` |

### Table C — Non-answers and failures

Double-rate denominators are the 8 and 12 BPM takes only (§13.2).

| Stratum | Frames | Abstained, conf < 0.4 % | Suppressed by presence % | Double-rate count | Double-rate % (of 8+12 BPM frames) | …of which conf ≥ 0.7 | Half-rate count | Source | Provenance |
|---|---|---|---|---|---|---|---|---|---|
| Seated clear | — | — | — | — | — | — | — | — | — |
| Seated null | — | — | — | — | — | — | — | — | — |
| Lying clear | — | — | — | — | — | — | — | — | — |
| Lying null | — | — | — | — | — | — | — | — | — |
| Standing clear | — | — | — | — | — | — | — | — | — |
| Standing null | — | — | — | — | — | — | — | — | — |
| 16 and 20 BPM takes (doubling not observable in band) | — | — | — | n/a | n/a | n/a | — | — | — |
| **Pooled** | — | — | — | — | — | — | — | — | — |
| *Comparator: sim, whole-band null, conf ≥ 0.7 and ≥ 1 BPM off* | — | — | — | — | 4.2% default / 2.7% RespirFi | — | — | SYNTHETIC | `NOTES.md:300-301` |

### Table D — Link health per block

| Block | Rate Hz (median) | Deviation from 20 Hz % | Loss % | Resets | dt_med ms | dt_p95 ms | dt_max ms | Kernel drops | Malformed (= FeatureFrames) | ≥ 10 Hz sustained? | Source | Provenance |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — | — | — | — | — | — |

### Table E — Bench preconditions (§1)

| Measurement | Value | Source | Provenance |
|---|---|---|---|
| AGC gain moves per minute, idle empty room, 10 min | — | — | — |
| AGC gain moves as % of packets | — | — | — |
| Bins reported dead (indices) | — | — | — |
| Bins carrying signal (count) | — | — | — |
| `first_word_invalid` frames, % | — | — | — |
| Per-subcarrier amplitude mean | — | — | — |
| Per-subcarrier amplitude std | — | — | — |
| Amplitude coefficient of variation | — | — | — |
| Empty-room `motion_energy`, mean | — | — | — |
| Calibrated presence threshold | — | — | — |
| *Comparator: modelled ESP32 arm, empty-room motion energy* | 1.018 | SYNTHETIC | `NOTES.md:531` |
| *Comparator: modelled ESP32 arm, seated motion energy* | 0.882 | SYNTHETIC | `NOTES.md:532` |
| *Comparator: modelled ESP32 arm, calibrated threshold* | 2.251 | SYNTHETIC | `NOTES.md:533` |
| *Comparator: modelled ESP32 arm, frames reporting occupied* | 0% | SYNTHETIC | `NOTES.md:534` |
| *Comparator: clean arm, frames reporting occupied* | 100% | SYNTHETIC | `NOTES.md:534` |
| *Comparator: AGC dither sweep, presence survives at* | 0.02% of packets | SYNTHETIC | `NOTES.md:565`, `NOTES.md:573-576` |
| *Comparator: AGC dither sweep, presence fails at* | 0.18% of packets | SYNTHETIC | `NOTES.md:566`, `NOTES.md:573-576` |

### Table F — Null hunt curves (§5.2)

One block of rows per (posture, distance) pair.

| Pair | Offset cm | best_bnr | Reported BPM @ 12 BPM pace | breathing_conf | Class assigned | Source | Provenance |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

### Table G — Session accounting

| Session | Date | Pairs recorded | Blocks recorded | Blocks abandoned | Reasons | Clock offset / skew range | Source |
|---|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — | — |

---

## 16. What this protocol does not measure, and what it needs

**Does not measure:**

- **Heart rate.** §2.5. No pulse oximeter, no number.
- **Anything spatial.** No position, no pose, no room geometry, no
  through-wall reach. `docs/CAPABILITY_LIMITS.md` settles all of those and
  none of them is on this protocol's axes.
- **Posture classification.** `NOTES.md:483-486` records (SYNTHETIC) that
  the collapse detector caught 100% of falls and also fired on 100% of
  sit-downs, 92% of readers walking in and 97% of fidgeting nights,
  "because amplitude-only 2-D WiFi cannot see height". The three postures
  here are a covariate, not a label to be predicted. Nothing in this
  protocol licenses a posture classifier.
- **Multiple people.** One subject, one link.
- **Link diversity.** ADR-006's two-link claim needs S6 and its own
  protocol; `NOTES.md:489-490` has the simulated version and
  `docs/CAPABILITY_LIMITS.md:303-306` warns that jumping to three links
  silently changes the presence quorum.
- **Anything learned.** ADR-007: no learned models in the S0–S5 path. This
  protocol produces the real-data measurement of the DSP ceiling that
  ADR-007 names as the precondition for even discussing one.

**Needs, before the scoring in §11 can run.** These are changes to
existing files and are deliberately not made here:

- `wifisense replay` has no way to disable `suppress_vitals_when_empty`
  (`wifisense/pipeline.py:52`). Without it, the two abstention causes in
  §12.2 cannot be separated from the CLI, because a presence-suppressed
  frame and a low-confidence frame both arrive as `UNAVAILABLE`. A
  `--no-suppress-when-empty` flag on `replay` would make the second
  scoring pass a one-liner.
- The P1 gate wording in `CAMPUS_SENSING_PROMPT.md:157` says "the belt".
  This protocol uses a phone accelerometer (§8.3). Amend the gate to name
  the reference actually used, or source a belt.
- `NOTES.md:216-217` gives the measured span as 56 × 312.5 kHz = 17.5 MHz;
  `docs/CAPABILITY_LIMITS.md:189-190` gives "56 subcarriers × 312.5 kHz is
  a 17.19 MHz measured span". These differ by one subcarrier spacing. One
  of them is wrong and the Fresnel arguments in both files depend on it.
- No scoring script exists (§11.4). One has to be written that reads a
  pcap and an accelerometer CSV, applies §9's two corrections, and emits
  Tables A, B and C. It belongs beside `evaluate.py` and must print
  `REAL` in its own output.

**Last line, and it governs everything above.** Until a block from §6
exists on disk, every number in this document's tables is a placeholder
and every comparator in them is simulated. `docs/CAPABILITY_LIMITS.md:338-340`:
"no board has ever run this firmware. Every number in `NOTES.md` is
simulated. Until a radio has been attached, the ceiling in this file is
the *least* uncertain thing about the system."
