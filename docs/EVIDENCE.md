# Evidence

What Saans has actually demonstrated, and what it has not, for a reader
deciding whether to fund it, pilot it, or work on it.

This file is written to be checked rather than believed. Every figure in
it names the file and line it was read from, in the form `NOTES.md:468`.
Nothing here is quoted from the field, from memory, or from the
literature as if it were a result of this project.

Marking works by section, not by line. Every section that reports a
performance figure of this system says **SYNTHETIC** once, at its head,
and that blanket covers every number under it: §2.2, §3, §5 and §6. The
blanket is the claim — if you find a performance figure in those sections
that did not come out of a simulator in this repository, that is a defect
worth reporting.

Three kinds of number are not covered by it and are not simulator output:
figures from the literature, which always name their paper; configuration
constants and code facts, which name a `file:line` in the tree; and
counts of the tree itself, whose method is given in §9. None of the three
is a measurement of how well this system works, which is the only thing
SYNTHETIC is about.

Where this document disagrees with another file in the repository, §9
says so and says which one I believe. A reader who finds an inflated
number here that §9 does not already admit should discard the rest of
the file.

---

## 1. The fact that frames everything else

**No part of this system has ever run on real hardware.** No ESP32 has
ever been flashed with this firmware. No CSI from a real radio has ever
been through this pipeline. Every performance number in this repository
— presence accuracy, breathing error, detector precision, throughput,
the cost of the receiver front-end — was produced by a simulator in this
same repository.

The repository states this in its own voice in three places:

- `docs/CAPABILITY_LIMITS.md:338` — "no board has ever run this
  firmware. Every number in `NOTES.md` is simulated."
- `edge/README.md:142` — the same claim, attached to the telemetry line
  a bench session is supposed to read first.
- `NOTES.md:517`, of the ESP32 receiver model — "Nothing here has
  touched a radio."

What follows from that, stated plainly:

- There is no presence precision, recall or F1 from a real room.
- There is no breathing error against a respiration belt, a metronome,
  or a counted breath.
- There is no uptime figure, no false-alarm rate per occupied night, no
  measured packet loss, no measured CSI rate, no measured RSSI.
- There is no evidence that the firmware's assumption about how many
  subcarriers the ESP-IDF driver hands over is correct. The code says so
  itself (`edge/README.md:140-144`): it logs what arrived and warns if
  the count or the buffer length is not what the DSP was compiled for.

What does exist is a signal-processing pipeline that is internally
consistent across three languages, held to one reference by generated
vectors, and measured against a physics simulator that was written to
model the radio rather than to flatter the detector. That is worth
something and it is not worth the thing it is often mistaken for. A
measurement of a model is evidence about the model.

There is a second-order version of the same caveat, and the repository
makes it about its own localization work rather than waiting for a
reviewer to: "test on a second room before believing anything. Every
number here came from one simulator written by the same process that
wrote the estimator" (`NOTES-localize.md:236-237`). That sentence
applies to every table below.

---

## 2. What is built, and what holds it together

### 2.1 One spec, three implementations

The design bet is recorded as ADR-002: the Python package `wifisense` is
the reference implementation; `spec/golden/generate.py` records its
output on fixed inputs for every stage; the Rust gateway DSP and the C++
edge DSP must reproduce those vectors within stated tolerances, and a
disagreement is resolved by fixing the implementation that disagrees —
never by loosening a tolerance
(`docs/adr/002-one-spec-three-implementations.md`).

The tolerances are a deliberate consequence of ADR-003: the ESP32-S3 has
a single-precision FPU, so the edge computes in `float` and the gateway
in `f64`. Rust must match the reference to 1e-9 relative on filter
outputs and 0.001 BPM on rates; C++ to 1e-4 relative and 0.05 BPM;
confidences carry an absolute tolerance because a relative one is
meaningless near zero (`SAANS_PLAN.md` §4,
`docs/adr/003-float32-edge-f64-gateway.md`).

This is the part of the project with the strongest evidence behind it,
and it is worth being precise about what it proves. It proves that three
independent implementations of the same arithmetic agree. It does not
prove the arithmetic is the right arithmetic for a real channel.

### 2.2 Parity, measured

Everything in this section is **SYNTHETIC**: these are three
implementations agreeing with each other on synthetic input, which is a
statement about the ports and not about a radio.

| Claim | Value | Source |
|---|---|---|
| Rust golden tests passing | 12, at 1e-9 relative | `NOTES.md:391` |
| End-to-end FeatureFrames, Rust vs Python | **672 / 672 byte-identical**, across both consent settings | `NOTES.md:392` |
| Rust parity after the receiver-impairment layer was added | max relative error 9.3e-15 against a 1e-9 tolerance | `NOTES.md:515` |
| Rust simulator vs Python geometric physics | agree to 1e-14 | `NOTES.md:428` |
| C++ float32 checks | **845** per compiler, nine binaries — re-counted 2026-10-24 from `make -C edge/dsp/test test` (eight suites print `PASS n/n`, totalling 711; `test_feature_frame` prints `14 cases, 134 checks`) | The site said 669 until 2026-10-24; see §9 |

Byte-identical FeatureFrames across two languages is the single
strongest verification result in the repository. It is stronger than a
tolerance: it means the Rust port reproduces the reference's floating-
point behaviour exactly, including the places where that behaviour is
idiosyncratic. `NOTES.md:394-400` records what that cost — CPython's
Neumaier-compensated `sum()`, half-to-even `round()`, `max(key=...)`
returning the first maximum, and a deliberately truncated
Euler–Mascheroni constant kept truncated on purpose so the confidences
match to 1e-12.

The three ports are not equally verified. Rust is byte-identical. C++ is
field-wise within tolerance, which `edge/README.md` explains is by
design: Python computes in f64 and rounds to f32 at the wire, the edge
computes in float32 throughout, so the low bits differ. The one place
C++ is held to byte equality is the `feature_codec` vector — the 68-byte
encoder must produce exactly the reference's bytes.

### 2.3 The counts, recounted

The counts below were produced for this document by reading the tree,
not by running anything: this pass was instructed not to run the test
suites, and did not. They are therefore counts of **tests defined**, not
tests passed. `rust/` was being edited by another process while I read
it, so the Rust and golden-vector counts were taken from the committed
`HEAD` (`e0d01ee`) rather than from the working tree.

| What | Counted here | Last count the repository records as *run* |
|---|---|---|
| Golden vector sets in `spec/golden/` | **23** (`git ls-files spec/golden \| grep '\.json$'`) | 21 (`NOTES.md:384`, `edge/README.md`) |
| Python test methods under `tests/` | **547** across 17 files, of which **308** are `test_localize_*` | 459 passed, 0 failed (`docs/COMPLETION_REPORT.md:186`) |
| Rust test functions at HEAD | **217** = 199 `#[test]` + 18 `#[tokio::test]` | 141 passed, 0 failed (`docs/COMPLETION_REPORT.md:179`) |
| — `saans-dsp` | 15 | 12 golden |
| — `saans-health` | 36 | 36 |
| — `saans-sim` | 72 | 21 |
| — `saans-gateway` | 93 (75 + 18 async) | 55 lib + 11 e2e |
| — `saans-cli` | 1 | — |
| C++ host test binaries | 9 files in `edge/dsp/test/` | 9 binaries, g++ and clang++, PASS (`docs/COMPLETION_REPORT.md:185`) |

Two of those gaps have an arithmetic explanation and one does not.

The Python gap is 547 − 459 = 88, which is exactly the size of
`tests/test_scene.py` (61) plus `tests/test_roomplan_import.py` (27).
Both files were added by commit `170c963`, after the run recorded in
`COMPLETION_REPORT.md`. The 459 figure is therefore not wrong, it is
superseded.

The Rust gap is dominated by `saans-sim`, which grew from 21 recorded
passing tests to 72 defined ones when the ESP32 receiver model (§3.5)
was added in commit `629634a`.

`README.md` used to describe `tests/` as "141 core tests", which was
stale in both directions: the non-localization Python files alone define
239 test methods, and 141 is the number of *Rust* tests recorded in
`COMPLETION_REPORT.md:179`. It now says 547, 239 of them outside
`test_localize_*`.

### 2.4 Continuous integration

`git ls-files .github/workflows` returns two files: `codeql.yml` and
`pages.yml`. `pages.yml` runs one narrow test selector —
`cargo test --locked -p saans-gateway --lib site::` — as a gate on
publishing the static site, and then exports and deploys it.

A third file, `.github/workflows/ci.yml`, was committed in `101639b` on
2026-10-19. It runs four jobs: `cargo fmt --check` then
`cargo clippy -D warnings` then `cargo test --workspace --locked`;
`python3 -m unittest discover -s tests`; `make -C edge/dsp/test check`;
and regenerate-and-diff on `spec/golden/`.

Two things about it are worth a sceptic's attention. Each command was run
locally before being written into the workflow, rather than written from
what the commands are supposed to do -- which is how the tree turned out
to be unformatted on 62 of 64 files and to carry nine clippy warnings,
both fixed in the two commits before it. And the golden-drift job was
checked by breaking it on purpose: perturbing the Goertzel coefficient by
one part in 1e9 drifted 26 vector files and the check exited 1.

What it does not yet have is a green run on GitHub's own runners. At the
time of writing it has been verified locally and not yet on a push, so
"CI passes" here means "the four jobs pass on one Linux machine", and the
first real push may still find something environment-specific.

`docs/COMPLETION_REPORT.md:189` says "There is **no CI in this
repository** — no `.github/workflows`". That was true when written and is
now stale in both directions. Its following sentence was the one that
mattered -- "That is an absence, not a pass" -- and the absence is closed.

### 2.5 What runs end to end

The S4 gate — a gateway that ingests, fuses, detects, serves and replays
— is recorded as met in simulation (`SAANS_PLAN.md:168-174`). Concretely,
what exists is: a Rust gateway that ingests FeatureFrames or raw CSI,
fuses the links of a space, runs the health detectors, keeps a bounded
in-memory time series, writes an event log, serves an HTTP API, a
WebSocket, `/metrics`, a seven-page public site and an operator
dashboard from one binary and one origin (ADR-008), and can replay a
recorded or simulated night at speed. Authentication is required
automatically whenever the listen address is not loopback and the config
refuses to start otherwise; roles are `admin`, `ops`, `safety`,
`occupancy`, a space outside a principal's scope answers 404 rather than
403, and reads of consented vital signs are appended to a daily audit
log (`docs/COMPLETION_REPORT.md` §2, §4).

Two defects found by driving that dashboard rather than reading it are
recorded in `docs/COMPLETION_REPORT.md` §2, and a third is worth
repeating because of what it is about: the per-space card read
"Breathing: not confidently detected" for a value that had been withheld
from the viewer's role. Those are different statements and the interface
was making the wrong one. The consent gate deletes vitals keys rather
than nulling them, so the two cases are distinguishable, and they are
now distinguished.

### 2.6 What is a stub

- *No longer a stub, since 2026-10-24:* the MQTT publisher
  (`mqtt.rs`). Tested against an embedded broker: the topic set and
  retain flags, no vitals topic for a space without consent, no vital
  sign on any other topic, the last will when the task is killed,
  republishing to a broker that appears late, and ingest time with the
  broker up, refused and silent. Not yet run against Home Assistant
  itself.
- *No longer a stub, since 2026-10-24:* the alert service. Routing,
  acknowledgement, escalation, persistence, Telegram and webhooks are
  built and tested, including the brief's exit gate run live: a scripted
  breathing pause paged a webhook once, survived a gateway restart, and
  was acknowledged and resolved from the dashboard, with both actions in
  the audit log. None of that has met a real Telegram bot or paging
  system -- the tests stand in local servers for both.
- *No longer missing, since 2026-10-24:* browser end-to-end tests. The
  manual checklist of `COMPLETION_REPORT.md` §6 is a Playwright suite
  (`saans-gateway/tests/browser/`) run in CI, against a SYNTHETIC replay.
- `wifisense/roomplan_import.py` is a stub: the mapping table is real,
  the format assumptions are untested (`README.md:335`).

### 2.7 The privacy boundary is one function

ADR-001 decides that raw CSI stays on the node: the ESP32 runs the DSP
and transmits only a 68-byte FeatureFrame at 2 Hz (`spec/wire.md:17`),
and the breathing and heart fields are zeroed **on the node** unless the
space has opted in. Raw CSI is a debug mode, off by default, and every
frame carries a flag saying it is on.

That guarantee is auditable by reading one function —
`apply_consent_gate` in `edge/dsp/feature_frame.hpp:124` — and checking
its callers. `edge/README.md` §Privacy names three layers behind it: with
`CONFIG_SAANS_VITALS_ENABLED` off the consent value is a `constexpr
false`, so a node built without consent has no code path that emits a
breathing rate; the gate runs immediately before encoding, so the
gateway receives zeros rather than being trusted to discard something;
and the consent state is a flag in every frame and is logged at boot.

ADR-005 adds the rules that make the outputs quotable: outputs are
advisory, carry a confidence and may abstain; there are no identities
anywhere, only spaces; and **every detector reports both its miss rate
and its false-alarm rate — publishing one without the other is not
allowed** (`docs/adr/005-advisory-consented-abstaining.md:16`). That last
clause is why §3.3 below reads the way it does, and its stated cost is
accepted in the ADR itself: "Coverage will look worse than systems that
always answer. That is the price of being believable."

---

## 3. What is measured in simulation

Everything in this section is **SYNTHETIC**. I read these numbers out of
the files named; I did not re-run the scripts that produced them.

### 3.1 The original 64-scenario sweep — the weakest provenance here

| Capability | Result | Source |
|---|---|---|
| Presence / occupancy | ~4× energy separation after calibration, no flicker | `README.md:79` |
| Breathing rate | mean error 0.009 BPM, max 0.017 BPM, 64/64 within 0.5 BPM | `README.md:80` |
| Heart rate, ungated | 80% within 3 BPM | `README.md:81`, `NOTES.md:144` |
| Heart rate, confidence ≥ 0.4 | 97% correct (38/39), mean error 0.38 BPM, accepting 61% of scenarios | `README.md:82`, `NOTES.md:146` |

Read these with the caveat the repository attaches to them two sections
later: that scene draws each subcarrier's static phase independently, so
the band is a random mix of rising and falling subcarriers; plain
averaging keeps only 8% of the breathing signal there and still reads
100% correct because the noise is low; and **every `evaluate.py`
scenario uses seed 1234, so the original 64-scenario table is one room**
(`NOTES.md:231-236`).

These four figures also have the thinnest provenance in this document.
The heart-rate pair is corroborated in `NOTES.md:144-146`. The breathing
and presence figures appear only in `README.md` and on the site's
landing page; there is no notes entry, no committed results file and no
re-run here. Treat them as the README's claim about a script, not as a
checked result.

The presence margin itself is better sourced: broadband RMS gave 0.197
empty against 0.21–0.26 occupied and the flag chattered constantly;
bandpassing to 0.1–2.0 Hz first gives 0.034 empty against 0.15 occupied
(`NOTES.md:155-158`).

### 3.2 Fresnel blind spots — 960 scenes, geometric physics

From `python3 evaluate.py --suite nulls`. Seeds start at 10000 and
nothing was tuned on the reported scenes (`NOTES.md:202-203`). Breathing
within 0.5 BPM, four readings per scene:

| Scenes | legacy | respirfi | respirfi, no BNR floor | PCA | oracle |
|---|---|---|---|---|---|
| Uniform placement (1920) | 95.1% | **95.7%** | 92.3% | 94.9% | 98.0% |
| Whole-band null (1068) | 64.5% | **68.1%** | 59.6% | 63.1% | 81.1% |
| Partial null (1348) | 89.2% | **91.5%** | 81.9% | 90.0% | 97.6% |
| Mixed-sign (236) | 61.0% | 67.8% | 69.9% | **80.5%** | 94.5% |

`NOTES.md:295-298`.

Confidently wrong — confidence ≥ 0.7 and yet ≥ 1 BPM off — falls from
4.2% to 2.7% at whole-band nulls, 6.8% to 1.7% on mixed-sign scenes and
1.5% to 0.3% on partial nulls with RespirFi selection (`NOTES.md:300-302`).
Paired against the default, RespirFi is right where the default is wrong
in 13 / 30 / 38 / 16 readings and **wrong where the default is right in
zero**; it engages on 40% of selection passes and falls back otherwise
(`NOTES.md:302-305`).

The conclusion the repository draws is the interesting one, and it is a
negative: subcarrier selection is "safe and slightly useful at ESP32
bandwidth, not transformative", capturing 20–27% of the headroom the
oracle shows, and whole-band blind spots stay at about two-thirds
accuracy for every real method, as the RespirFi paper's own equation 6
predicts at a 17.5 MHz span. The fix is a second link from another
angle, not more cleverness in the subcarrier domain (`NOTES.md:312-322`).
That is ADR-006, and it is the reason a space where breathing matters
costs two nodes instead of one.

One expectation that failed, recorded as such: a 0.3 m torso was
expected to average over several Fresnel zones and remove the nulls. It
does not. A torso facing the link lies along the Fresnel ellipse
tangent, where excess path changes only to second order, and it moves as
one block, so the null moves rather than disappearing
(`NOTES.md:265-272`).

### 3.3 The health detectors, on held-out seeds

`saans eval` runs scripted timelines from `saans-sim` through the
production path — the golden-verified per-link feature extractor, space
fusion, the space monitor. Thresholds were tuned on seeds 0–1; the table
is from held-out seeds 2–7: 252 runs, 187 simulated hours
(`NOTES.md:428-434`).

| | 1 link | 2 links | 3 links |
|---|---|---|---|
| Occupancy accuracy | 98.2% | 99.5% | 98.9% |
| Still person read as empty | 2.1% | 0.4% | 1.3% |
| Empty room read as occupied | 1.0% | 1.2% | 0.6% |
| Breathing reported (still breather) | 93.6% | 100% | 100% |
| ... within 1 BPM | 93.9% | 99.9% | 100% |
| Breathing pauses caught (248 true, 10–40 s) | 81.5% | 89.9% | 88.7% |
| ... precision | 98.5% | 98.7% | 98.7% |

`NOTES.md:468-474`. Note that both halves of every pair are present, per
ADR-005: the miss rate and the false-alarm rate, never one alone.

Beside the table (`NOTES.md:476-488`):

- No breathing pause was reported in 70.6 h of timelines that contained
  none. Detected pause duration is late by 9–10 s at the median, from
  the band filter plus a 2 s RMS lag.
- Lone-worker inactivity (30 min) fired in 35 of 36 runs, at a median of
  2500 s for stillness starting around 700 s. It also fires for every
  40-minute reader, which is correct by its definition and is why it is
  advisory rather than an alarm.
- Fast breathing (26–30 BPM) raised `RateHigh` in 31 of 36 runs, and
  also fired in 2 of 36 reader runs and 3 of 36 sit-down runs.
- With the collapse detector off, Check-severity alerts are 0 per hour
  everywhere except nights, at 2.9/h, which are breathing pauses — what
  the detector is for.
- Link diversity is confirmed in simulation: a second link takes
  breathing coverage from 94% to 100% and halves missed pauses.

Two limits are recorded immediately after
(`NOTES.md:491-499`): the simulator's body never blocks the line of
sight, never sways, and its CSI phase is clean; and a still person who
stops breathing can drop out of presence after about 18 s on a single
link, because the presence energy is mostly breathing. Pauses that end
that way are reported with that caveat attached.

### 3.4 Throughput

32 subcarriers, 20 Hz, 3267 frames, one core (`NOTES.md:410-417`):

| | frames/s | × real time, one link |
|---|---|---|
| Python `SensingPipeline.process` | 191 | 9.6 |
| Rust, same algorithm, estimate every frame | 2,851 | 143 |
| Python `FeatureExtractor` | 175 | 8.8 |
| Rust `FeatureExtractor`, estimate only when a frame is due | 25,283 | 1,264 |

The language is worth about 15×; computing the spectral estimates twice
a second instead of twenty times is worth another 9×, with identical
output because the estimates are pure functions of the extractor state.
One gateway core keeps up with about 1,200 links (`NOTES.md:419-423`).

This is the one table in the document that is not a claim about radio
physics, so it is also the one least affected by the absence of
hardware — though the frames it processed were still synthetic.

### 3.5 The ESP32 receiver model, and the single constant it rests on

`saans-sim` produces a clean complex channel. A real node does not: it
applies automatic gain control, a random per-packet phase, a timing
offset that tilts phase across subcarriers, a static per-device ripple,
and quantizes to int8. `src/impair.rs` models that, after the clean
channel and off by default, because the clean channel is the contract
with the other two implementations (`NOTES.md:505-516`).

**Its constants are guesses** (`NOTES.md:517`). They come from published
descriptions of ESP32 CSI, not from a measurement taken in this project.

Four arms, same scenario and seed, 5 room seeds, one link, 56
subcarriers at 20 Hz, seated subject at 14 BPM (`NOTES.md:528-536`):

| | clean | int8 | no-AGC | esp32 |
|---|---|---|---|---|
| breathing BNR, best subcarrier, dB | −0.49 | −0.45 | −0.72 | **−2.11** |
| motion energy, empty room | 0.129 | 0.129 | 0.152 | **1.018** |
| motion energy, person seated | 0.351 | 0.352 | 0.356 | 0.882 |
| calibrated presence threshold | 0.185 | 0.185 | 0.213 | **2.251** |
| frames reporting occupied, % | 100 | 100 | 100 | **0** |
| breathing BPM (truth 14.0) | 14.00 | 14.00 | 14.00 | **none** |

Three findings, in the order the file ranks them by confidence
(`NOTES.md:538-558`):

1. **int8 quantization costs nothing measurable.** Every metric is
   within 0.04 dB of the clean arm. This was the term expected to hurt.
2. Static ripple, per-packet phase and receiver noise cost little:
   breathing BNR 0.23 dB worse, breathing still recovered at 14.00 BPM
   on 100% of frames.
3. **The AGC is the whole effect, and it breaks the detector nobody was
   worried about.** Breathing survives — 1.6 dB of BNR against clean.
   What breaks is presence: gain steps are impulses in the 0.1–2 Hz
   presence band, the empty-room threshold inflates twelvefold, and the
   empty room ends up reporting *more* motion energy than the seated
   person. The contrast does not shrink, it inverts, and vitals are then
   suppressed because the pipeline believes the room is empty.

The part of this result that matters most to a reader is what the file
does next: it tests its own conclusion to destruction. The whole effect
rests on one uncalibrated constant, `agc_dither_prob` — how often a
converged gain loop moves anyway (`NOTES.md:560-569`):

| `agc_dither_prob` | gain moves, % | empty | seated | threshold | occupied, % |
|---|---|---|---|---|---|
| 0.000 | 0.02 | 0.148 | 0.245 | 0.210 | 100.0 |
| 0.002 | 0.18 | 0.185 | 0.329 | 0.959 | 0.0 |
| 0.005 | 0.57 | 0.465 | 0.433 | 1.203 | 0.0 |
| 0.020 (committed) | 2.43 | 1.018 | 0.882 | 2.251 | 0.0 |

So the honest claim is not "an ESP32 breaks presence". It is that
presence survives a gain that moves about once every four minutes and
fails completely at once every twenty-eight seconds, and the sweep
brackets the breakpoint between those two without locating it any more
finely (`NOTES.md:572-578`). An earlier draft of that section put the
breakpoint at "once every nine seconds"; the file corrects itself and
explains that the 0.005 row is already inside the failing region, not at
its edge (`NOTES.md:580-585`).

And the resolution is explicitly not more simulation: "log `agc_gain` on
an idle node in an empty room for ten minutes and count the moves. It is
the single most valuable number in the first recording, and until
someone takes it the table above ranks nothing" (`NOTES.md:585-590`).

Two gaps in the layer itself are recorded (`NOTES.md:597-606`):
`first_word_invalid` is not modelled, so a consumer of simulated raw CSI
never sees the two-bin index shift it causes; and **which 56 bins the
firmware actually hands over is unresolved** — the simulator's 56 are the
idealised occupied set, and by ESP-IDF's documentation the firmware's
first 56 pairs may be a different 56 entirely. A subcarrier index may
not mean the same thing on both sides of the wire.

---

## 4. What the evaluation broke

Six defects are recorded as found by running the evaluations rather than
by unit tests. They are listed here because they are the clearest
available evidence about how the project treats its own results — in
each case the expectation is stated first, the measurement contradicts
it, and the fix goes into the Python reference, regenerates the golden
vectors, and is then caught by those vectors in the Rust and C++ ports
before either is edited by hand (`NOTES.md:460-465`).

1. **Presence threshold too low.** Expected: mean + 4σ of 30 s of empty-
   room energy is safe. Measured: an empty room read occupied 36% of the
   time. The energy is an 8 s sliding RMS, so 30 s holds only a few
   independent windows and the sample σ is badly underestimated. Fixed
   with a derived floor; empty-room false occupancy fell to 0.6–1.2%
   (`NOTES.md:437-444`).
2. **The gross-motion band leaked breathing.** Expected: a 1–4 Hz
   bandpass sees limbs, not breathing. Measured: a still breather's
   motion index was 1.33 against 0.88 for an empty room, because one
   biquad's 6 dB/octave skirt passes about 19% of a 0.25 Hz component.
   Fixed with two cascaded sections: 0.79 against 0.75
   (`NOTES.md:446-451`).
3. **One noisy sample reset every stillness clock.** The heartbeat lives
   in the gross-motion band, so a still person's index wanders. The
   30-minute inactivity alert fired in 2 of 12 lone-worker runs. Fixed
   with a higher threshold and three confirming samples: false stillness
   resets fell from 60/h to 0.15/h while still catching about 82% of
   fidget bursts (`NOTES.md:453-458`).
4. **Packet loss silently biased the breathing rate.** Expected: loss
   only reduces SNR. Measured: the pipeline counted frames, not time, so
   5% loss made a 15 BPM breather read 15.76 BPM and 15% loss read
   **17.80 BPM at confidence 0.98** — confidently wrong. Fixed with a
   sequence gap filler; now 14.98 and 14.97 BPM (`NOTES.md:330-341`).
5. **BNR computed by magnitude was wrong, and had already invalidated a
   full evaluation run.** A clean 0.25 Hz tone with 30% noise scored
   0.66, because one strong peak loses to 51 out-of-band noise bins. The
   selector almost never engaged. Fixed by using spectral power, and the
   evaluation run that had used the broken metric was discarded
   (`NOTES.md:276-286`).
6. **Calibrating through the filter transient is fatal.** A biquad near
   0.1 Hz takes tens of seconds to settle; calibrating during that window
   measures the filter waking up, sets the threshold far too high, and the
   detector then reports "empty" forever. Fixed with a warmup period of
   five periods of `f_low`, 50 s by default (`NOTES.md:160-166`).

Items 1–3 were found by the S3 evaluation after the unit tests passed.
That is the argument for the evaluations existing at all, and it is also
the reason to be sceptical of any capability in this repository that
does not have one.

---

## 5. Localization: the result that went the other way

Phase P0b built link-based localization of one walking person on
simulated WiFi links, and the result is a negative one that the project
kept.

**At 4–12 single-antenna nodes, radio tomographic imaging did not beat a
trivial weighted-link-midpoint estimator.** RTI reached 0.72 m median
error at 13–18% coverage; the baseline reached 0.709 m at 100% coverage,
and beat or tied RTI in every one of 15 (λ, α) configurations tested
(`NOTES-localize.md:6-10`). All of it is SYNTHETIC.

The (λ, α) grid at N = 8, 177 moving samples (`NOTES-localize.md:104-112`):

| λ | best RTI centroid | weighted baseline | blind pixels |
|---|---|---|---|
| 0.02 (published) | 0.754 | 0.709 | 26/192 |
| 0.1 | 0.747 | 0.709 | 6/192 |
| 0.3 | 0.715 | 0.709 | 2/192 |
| 0.6 | 0.717 | 0.709 | 0/192 |
| 1.0 | 0.716 | 0.709 | 0/192 |

**The caveat, and the one configuration where the baseline did not
win.** The "15 configurations" above are the (λ, α) grid at N = 8. The
separate node-count sweep (`NOTES-localize.md:88-93`) has one row that
goes the other way — N = 4, where RTI's median moving error is 0.44 m
against the baseline's 0.72 m:

| Variant | links | RTI median (moving) | coverage | weighted baseline |
|---|---|---|---|---|
| N=4 | 12 | **0.44** | 15% | 0.72 |
| N=6 | 30 | 0.97 | 17% | **0.65** |
| N=8 | 56 | 0.82 | 18% | **0.76** |
| N=12 | 132 | 0.71 | 13% | **0.69** |

The file does not let that row stand as a win, and the reason is
general: "Accuracy improved as coverage fell, in every sweep… any
headline accuracy number is selected by the abstention threshold and is
meaningless without its coverage figure. The same effect makes N=4 look
best (0.44 m at 15%); it is selection, not capability"
(`NOTES-localize.md:188-193`). The baseline answered 100% of the time;
RTI at N = 4 answered 15% of the time, and N = 4 also leaves a third of
the room structurally unobservable at tight λ — pixels inside no
ellipse, which no inversion can reconstruct (`NOTES-localize.md:116-123`,
`133`).

The mechanism behind the negative result is stated and is physical, not
an implementation complaint: correlation between the published ellipse
weight model and the measured link disturbance was **+0.13**, with 29%
of timesteps actively negative and an inside/outside ratio of 1.8×.
Raising the simulated shadowing to 30 dB lifted that to +0.50 and 19×,
which confirms the mechanism — RTI's premise is shadowing, valid for RSS
on 802.15.4 where a body blocks the path, while CSI amplitude is
dominated by coherent multipath, which is not monotonic in distance from
the line of sight (`NOTES-localize.md:173-181`). It is the same Fresnel
physics that gates the breathing work, seen from the other side.

Two further things from that file are worth a funder's attention. First,
its literature table (`NOTES-localize.md:34-68`) checked each cited RTI
result against a primary source and found that the number everyone
quotes for Wilson & Patwari 2010 does not exist in that paper, that the
inventors failed to reproduce their own VRTI result with identical
hardware, and that there is no verifiable published localization error
for RTI on WiFi CSI at all. Second, the author lists three things they
got wrong during the phase, each caught by measurement — including
scoring the still-person case by whether the tracker answered rather
than whether it was right, which "mistakes a capability for a bug" — and
attaches the correct caveat to the correction: the still-case result
depends entirely on the simulator giving a still person meaningful
breathing modulation, and "this result is simulator-optimistic"
(`NOTES-localize.md:150-160`).

The operational conclusion: do not expect metre-level tracking from 8
ESP32 nodes; plan for motion detection and zone-level localization; ship
the six-line baseline first and treat RTI as the thing that must earn its
place (`NOTES-localize.md:126-131`).

---

## 6. What is known not to work

Where this section gives a performance figure of this system it is
**SYNTHETIC**, on the same terms as §3. The physical quantities here --
wavelengths, apertures, bandwidths, antenna counts -- are not: they are
arithmetic from the hardware's own specification, and they are what make
these limits permanent rather than a question of effort. Where a figure
comes from a published paper it names the paper.

### 6.1 Pose and skeletons — not with this hardware, at any effort

The ESP32-S3 is 1T1R: one transmitter, one receiver, one antenna
(`docs/CAPABILITY_LIMITS.md:20`). With one receive antenna the array
steering vector is the scalar `[1]` for every angle, so the angular
spectrum is constant in angle and a 1×1 covariance matrix has no noise
subspace. The angular axis carries zero bits, not few — a discontinuity,
not a gradient.

The file does not stop there, because angle-of-arrival is only one route
to pose. It argues the other three axes separately: delay resolution is
one range cell of 3.75–7.5 m, so every limb, the torso and the head sit
in the same cell (`docs/CAPABILITY_LIMITS.md:27`); Doppler gives the
aggregate radial velocity with nothing attributing a velocity to the
limb that produced it; so the forward map from pose to measurement is
many-to-one by construction, and a skeleton recovered from it is being
supplied by a learned prior conditioned weakly on the radio.

It then states the test that would falsify its own argument: a 1×1 pose
result that holds up against held-out subjects and held-out rooms, not a
random split. And it records, marked `[sourced]`, that cross-environment
per-joint error for published systems on correct (≥ 3 RX) hardware is
254–296 mm for the 2026 state of the art and 369.5 mm for the MM-Fi WiFi
baseline (`docs/CAPABILITY_LIMITS.md:86-88`) — a body-sized blob, produced
by models trained against cameras.

The weaker fallback fails too, and this project measured that itself:
the collapse detector caught 100% of simulated falls and also fired on
100% of sit-downs, 92% of readers walking in and 97% of fidgeting
nights, because amplitude-only WiFi cannot see height. It ships off and
should stay off (`NOTES.md:483-487`,
`docs/CAPABILITY_LIMITS.md:95-100`).

### 6.2 Through-wall — three different things wearing one name

- **Imaging or pose through a wall: permanently out of scope.** Ten-
  centimetre depth bins need about 1.5 GHz of swept bandwidth; HT20
  gives 7.5 m and HT40 gives 3.75 m (`docs/CAPABILITY_LIMITS.md:115-118`).
  The famous results are FMCW radars sweeping ~1.7 GHz on USRPs with
  3–10 directional antennas; "WiFi frequencies" there means the band,
  not the protocol.
- **Presence and gross motion through a partition: plausible,
  unmeasured.** A wall costs SNR, not resolution, and nothing Saans
  emits depends on range or bearing. Status: measure it yourself.
- **Breathing through a wall: not demonstrated on this hardware class.**
  Every through-wall respiration result in the cited literature needs at
  least two coherent receive chains.
- **The product fact that outranks all three:** a link with no range and
  no bearing resolution cannot tell which side of the wall the person is
  on, so through-wall reach converts corridor traffic into occupancy
  false positives for the instrumented room — a regression in the one
  feature that works (`docs/CAPABILITY_LIMITS.md:150-166`).

### 6.3 Room size, wall layout, and a warning about one of our own pictures

The deciding quantity here is coherent aperture extent, not bandwidth.
Resolving a wall at 3.5 m to 30 cm needs an aperture of about 4.5 m —
larger than the room — and Saans has an aperture of zero: nodes bolted
to walls, positions typed in by a human, each PLL free-running with no
inter-node phase reference (`docs/CAPABILITY_LIMITS.md:177-186`). The
backup limit is the measured span: 56 subcarriers × 312.5 kHz is
17.19 MHz, giving a bistatic path-length cell of 17.45 m, so in a
3.5 × 3.5 × 3 m room the line of sight and all six first-order
reflections land in one cell (`docs/CAPABILITY_LIMITS.md:188-199`).

The reason to point a funder at this specific section is what it does to
this project's own work. The Explorer's Fresnel coverage map is honest
only because the simulator knows its node positions exactly. With node
positions off by 5 mm the map's correlation with truth falls to 0.87; at
1 cm, 0.65; at 2 cm, 0.27 (`docs/CAPABILITY_LIMITS.md:236-240`). A human
with a tape measure does not place a node to 5 mm, so a per-cell blind-
spot heatmap over a real room is not something this system may draw —
and neither is an install-time CLI printing a median blind-area
percentage. The conclusion is "ask, do not infer": a laser distance
meter is exact, and effort spent inferring what an installer could type
in is waste.

### 6.4 Heart rate

Heart rate is built and is not trusted. CSI amplitude is the magnitude of
a static multipath vector plus a body-reflected vector, so it is a
nonlinear function of chest displacement, which fills the cardiac band
with breathing harmonics and intermodulation sidebands. On synthetic
data with true `f_h` = 1.133 Hz and `f_b` = 0.25 Hz, the `f_h + f_b`
sideband measured 147.4 power against the true carrier's 123.2 — the
artefact is stronger than the signal, and plain peak picking returns
83 BPM for a 68 BPM heart (`NOTES.md:126-136`). The symmetric-triplet
search recovers 68.04 BPM against 68.0 truth, and the estimator is still
unreliable: 80% within 3 BPM ungated, 97% correct over 61% of scenarios
when gated at confidence ≥ 0.4 (`NOTES.md:140-147`). It is off by default
and the honest position is that breathing is solid and heart rate is not.

### 6.5 The confidence is a gate, not a calibrated probability

`docs/CAPABILITY_LIMITS.md:324-333` corrects that file's own earlier
wording. Calling the breathing confidence "calibrated" is stronger than
the evidence: the estimator locks onto twice the true rate at whole-band
blind spots, and the Explorer review observed it doing so at **confidence
1.00** — unanimous about a doubled rate. The gate turns an unreliable
number into a mostly-reliable one that abstains; it does not make the
confidence a probability, and the double-rate failure is the known case
where it does not save you.

---

## 7. `docs/CAPABILITY_LIMITS.md` is part of the work, not a disclaimer

A reader evaluating this project should read that file before this one.
It is unusual in four specific ways, and each is worth more than a
capability claim would be:

1. **It names the deciding physical quantity for each refused
   capability** — receive-chain count for pose, swept bandwidth for
   through-wall imaging, coherent aperture extent for room geometry —
   and resolves the question against that quantity rather than against
   an opinion. It also warns that the bandwidth argument does *not*
   transfer to the geometry question, so a reader who assumes it does
   will go shopping for the wrong hardware.
2. **It marks its own provenance.** `[verified here]` means checked
   against source, datasheet or code in this repository; `[sourced]`
   means it comes from the feasibility study's literature search.
   Numbers without a source are not in the file.
3. **It states what would falsify it.** The pose section says outright
   that if a 1×1 pose result appears that holds against held-out
   subjects and held-out rooms, the section is wrong and should be
   rewritten — and that the same held-out-room test should be applied to
   this project's work as readily as to anyone else's.
4. **It lists the traps that look like progress**
   (`docs/CAPABILITY_LIMITS.md:295-315`): revisiting RTI, jumping from
   one link to three without setting the presence quorum explicitly,
   chasing CSI phase on the strength of a stale note, and buying mmWave
   modules to retire the localization result. One of those entries flags
   a contradiction between two other files in the repository and says
   which one is stale.

Its closing sentence is the right frame for everything above: until a
radio has been attached, the ceiling in that file is the least uncertain
thing about the system (`docs/CAPABILITY_LIMITS.md:338-340`).

---

## 8. What would have to happen for any of this to count as evidence about the real world

In the order the repository supports, and with the cheapest first.

**Attach a radio.** Two ESP32-S3 devkits, one TX and one RX, both on the
same channel; the firmware is written and builds for esp32s3, and the
build is Docker-pinned so nothing needs installing locally
(`edge/README.md`). This is the step that converts everything above from
a model into a measurement.

**Read the capture telemetry line before trusting any number.** The RX
prints what actually arrived — buffer length, inferred subcarrier count,
expected count, and counters for `2nd_ch`, `first_word_invalid`, short
and changed layouts. Two assumptions are waiting on it: that the TX is
sending HT frames at all, and that the driver hands over the 56 bins the
DSP was compiled for (`edge/README.md:120-144`, `NOTES.md:597-604`).

**Take the one measurement that ranks the receiver model.** Log
`agc_gain` on an idle node in an empty room for ten minutes and count
the gain moves. Until that number exists, the AGC table in §3.5 ranks
nothing (`NOTES.md:585-590`).

**Then run the P1 gates, which are already written down**
(`CAMPUS_SENSING_PROMPT.md:157`): presence F1 ≥ 0.95 over 48 h;
still-person false-empty < 5%; breathing within ±1 BPM of a respiration
belt in ≥ 80% of 60 s windows, seated. The measurement protocol is
specified beside them — labelled sessions with a door contact sensor for
presence, a subject reading or typing for 20 minutes for the still-person
case (the case PIR fails, and therefore where WiFi has to earn its
place), a belt or metronome-paced counted breathing for the rate, and a
PIR and a cheap 24 GHz module in the same room as baselines, with the
instruction to report plainly if WiFi neither beats nor complements them
(`CAMPUS_SENSING_PROMPT.md:138-150`).

**Get ethics approval first.** Human-subject data collection is gated on
institute ethics committee approval, and breathing and heart data are
health data under India's DPDP Act 2023: opt-in per space, written
consent, retention limits, no identities (`SAANS_PLAN.md` §8,
`CAMPUS_SENSING_PROMPT.md:173`).

**Then a second room, before believing anything.** The localization
notes end with the instruction that applies to every table in this
document: "test on a second room before believing anything. Every number
here came from one simulator written by the same process that wrote the
estimator" (`NOTES-localize.md:236-237`). The cross-environment collapse
described in `CAPABILITY_LIMITS.md` §1 is the specific failure mode a
single-room result hides.

**And only then, if at all, a learned model.** ADR-007 forbids neural
networks and trained classifiers in the S0–S5 path, and allows one
afterwards only if it beats the measured DSP version leave-one-room-out
(`docs/adr/007-no-learned-models-yet.md`).

The realistic summary of what a successful P1 would buy: evidence for
presence and for breathing-with-abstention on one link in one room.
It would say nothing about multi-link blind-spot coverage (S6), nothing
about alarm rates across a building (S7), and nothing about the
capabilities refused in §6, which are refused on grounds that hardware
cannot change.

---

## 9. Where this repository's own numbers are stale

Listed because a reader who finds an unlisted inconsistency should
discount everything else here.

Most of this table has since been fixed. It is kept, with a status per
row, because the list of what was wrong is the evidence that checking
found it; a table of only the current problems would hide how many there
were.

| Claim | Where | Status |
|---|---|---|
| "21 vector sets" | `NOTES.md:384`, `edge/README.md` | Fixed: 23, the count `generate.py` prints. |
| C++ golden checks: 845 | `SAANS_PLAN.md`, `docs/COMPLETION_REPORT.md:185` | **Correct.** Re-counted 2026-10-24: 845 per compiler. "golden" dropped from SAANS_PLAN, since `tdma` and `node_config` check configuration, not vectors. |
| C++ golden checks: 669 | `rust/.../assets/site/evidence.html:134` and `:168` | Was the stale one: written in `187201e`, before later suites existed. Fixed to 845. |
| "141 core tests" for `tests/` | `README.md` | Fixed: 547, 239 outside `test_localize_*`. |
| "459 passed" (Python) | `docs/COMPLETION_REPORT.md:186` | Superseded by 88 tests added in `170c963`; 459 + 88 = 547. |
| "141 passed" (Rust) | `docs/COMPLETION_REPORT.md:179` | Superseded; 217 test functions defined at HEAD. |
| "208 tests" / "351 tests" | `NOTES-localize.md:214`, `:218` | Stale; 308 localization methods and 547 total are defined now. |
| "no CI in this repository" | `docs/COMPLETION_REPORT.md:189` | Amended in place with a dated note: `ci.yml` since `101639b`. |
| Site evidence page: "21 vector sets" (S0 row) | `rust/.../assets/site/evidence.html` | Fixed: 23. |
| "CSI phase is the highest-value next piece of work" | `NOTES.md:192` (§5 item 4) | Marked superseded in place, with the reason: `docs/CAPABILITY_LIMITS.md` §4, RespirFi's footnote that single-antenna phase is unreliable under CFO/SFO. |
| TX beacon default "50 Hz", "bounds the achievable CSI rate" | `edge/README.md:39`, `:91` | Wrong on both counts. Fixed: the TX rate tracks the CSI rate, 20 Hz, because one beacon is one CSI frame and a faster TX scales the RX's time axis (`saans_config.h:27-31`). |
| "56 subcarriers × 312.5 kHz is a 17.19 MHz span" | `docs/CAPABILITY_LIMITS.md:189` | Arithmetic: that product is 17.5 MHz; 17.19 is 55 spacings, edge to edge. Fixed, with a note on the other convention. |
| FeatureFrame "~64 bytes" | `SAANS_PLAN.md` §5 | Fixed: 68, fixed size. |
| Breathing confidence described as "calibrated" | earlier draft of `CAPABILITY_LIMITS.md` | Corrected in that file, `:324-333`. See §6.5. |
| AGC breakpoint "once every nine seconds" | earlier draft of `NOTES.md` §9 | Corrected in place, `:580-585`. See §3.5. |

Two notes on the provenance of this document's own counts. They were
produced by reading the tree — `git ls-files`, a grep for `#[test]` and
`#[tokio::test]`, and an AST walk over `tests/*.py` counting methods on
`unittest.TestCase` subclasses — and not by running anything, because
this pass was instructed not to run the suites. A count of tests defined
is weaker evidence than a count of tests passed. And `rust/` and
`spec/golden/` were being edited by another process while I read them,
so the Rust and vector-set counts were taken from the committed `HEAD`
(`e0d01ee`), not the working tree.

---

## 10. How to check this document

Nothing below needs hardware.

```bash
python3 -m unittest discover -s tests          # the reference's own tests
python3 spec/golden/generate.py                # regenerate the vectors
cd rust && cargo test --workspace              # Rust parity + gateway + sim
make -C edge/dsp/test check                    # C++ float32 parity
python3 evaluate.py --suite legacy             # the 64-scenario sweep, ~15 s
python3 evaluate.py --suite nulls              # the Fresnel study
cargo run --release -p saans-cli -- eval       # the S3 detector evaluation
cargo run --release -p saans-sim --example impairment_cost   # the AGC table
```

The four files a sceptical reader should read in full, in order:
`docs/CAPABILITY_LIMITS.md` (what the hardware cannot do, and the
deciding quantity for each), `NOTES.md` §6–§9 (every measured result and
every bug the evaluations found), `NOTES-localize.md` (the negative
result, and its literature audit), and `docs/adr/` (the decision
records, each with its consequences spelled out).

And one function, for the privacy claim specifically:
`apply_consent_gate` in `edge/dsp/feature_frame.hpp:124`, plus its
callers.
