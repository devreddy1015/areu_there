# What to change next, and why each one makes it more accurate

Written 2026-10-24 against `main` @ `ed924f8`. A plan, not a record:
nothing below is done. Every figure cites where it was read from and is
SYNTHETIC unless it says otherwise, which in this repository means all of
them.

"More accurate" means three different things here and the plan keeps
them apart, because a change that helps one can hurt another:

1. **Accuracy of the estimate.** Breathing BPM error, presence F1,
   false-empty rate. The numbers a user would care about.
2. **Robustness.** The same numbers after the radio does what a real
   radio does: gain steps, dropped frames, a subcarrier map that is not
   the one the simulator assumed.
3. **Accuracy of the record.** That what the documents say is what the
   code does and what was measured. This project's credibility is this
   and nothing else, and the last pull request found it wrong in nine
   places.

---

## 0. The fact that ranks everything else

No board has ever run this firmware (`docs/CAPABILITY_LIMITS.md:338`).
Every number in this plan is from a simulator written by the same
process that wrote the detector. Nothing in sections 1–2 is worth as
much as the first ten minutes of real CSI, and three items below are
written to be *ready* for that recording rather than to replace it.

The three questions that recording answers, and what each unblocks:

| Question (NOTES.md §9 "Known gaps") | Unblocks |
|---|---|
| How often does the AGC gain move? | Whether §1.4 ships on or off |
| Which 56 bins does the firmware hand over? | `BinLayout`; every per-bin result |
| How often does `first_word_invalid` fire? | Whether the 2-bin shift needs modelling |

`docs/HARDWARE.md` is the shopping list. `docs/PROTOCOL_S5.md` is the
procedure. Both exist now; neither has been followed.

---

## 1. Start now — nothing here needs a board

In the order they should be done. Each says what it touches, how it is
measured, and what would make it wrong.

### 1.1 Fix the record: nine stale claims, half a day

**Done**, on branch `saans/record`. Two corrections to the list below
came out of doing it rather than reading it. The 845-vs-669 question is
answered: 845 per compiler is right, re-counted; the site's 669 dates
from the first landing-page commit. And a row this plan carried was
wrong: an audit flagged `CAPABILITY_LIMITS.md` §2e as a mis-levelled
heading, and it is not -- it is about room size, not through-wall, its
own subtitle calls it a "*third* reason" alongside §1 and §2, and the
"2e" label only avoids renumbering the §3–5 that other files cite.
Demoting it would have filed room geometry under "Through-wall". The row
is removed. README's legacy figures were re-run and reproduce exactly,
so they gained a source rather than a warning.

The adversarial audit in PR #5 found these and I fixed the ones in files
I was already touching. These are the rest. Each is a line that says
something the tree no longer does. None is a judgement call.

| Where | Says | Reality |
|---|---|---|
| `SAANS_PLAN.md:147` | FeatureFrame "~64 bytes" | 68, fixed size (`spec/wire.md:17`); `crc32` at 64..67 |
| `NOTES.md:192` (§5 item 4) | CSI phase is "the highest-value next piece of work" | Retired by `CAPABILITY_LIMITS.md:308-313`: single-antenna phase is unreliable under CFO/SFO. The line is still there, unmarked, and an agent reading NOTES.md alone will act on it |
| `edge/README.md:39, :91` | TX beacon default 50 Hz | Kconfig default is `SAANS_CSI_RATE_HZ` (`Kconfig.projbuild:40`), which defaults to 20 (`:73`). One of them is wrong and the firmware is the one that runs |
| `docs/COMPLETION_REPORT.md:189` | "no CI in this repository" | `ci.yml` since `101639b`, four jobs, green on main |
| `CAMPUS_SENSING_PROMPT.md:157` | S5 gate: "within ±1 BPM of **the belt**" | The same file's line 141 allows "a respiration belt, or breathing paced to a metronome and counted", and PROTOCOL_S5 uses a phone accelerometer. The gate should say "the reference" and match its own §141; as written it names a device nobody has |
| `CAPABILITY_LIMITS.md:189` vs `NOTES.md:217` | 17.19 MHz vs 17.5 MHz for "the ESP32 span" | Different quantities: 55 spacings edge-to-edge vs 56 × 312.5 kHz. Both right, never said side by side, so a reader thinks one is a typo |
| `SAANS_PLAN.md:170`, `COMPLETION_REPORT.md:185` vs `site/evidence.html:134, :168` | 845 vs 669 C++ golden checks | Still unresolved. `make check` prints per-binary totals, not "golden checks"; whoever wrote each number should say what they counted |
| `docs/adr/` | no index | Nine ADRs and a reader finds them by `ls`. The numbering gaps that would signal a missing decision go unnoticed |

Also: README.md:79-82 carries four figures — "Mean error 0.009 BPM, max
0.017 BPM, 64/64 within 0.5 BPM", the heart-rate 80% / 97% / 61% — with
no source anywhere but README and the site. NOTES.md §6.2 says every
legacy scenario uses seed 1234 and that scene "exaggerates cancellation".
Those numbers are either regenerated and dated by §1.2, or marked in
README as typed and unreproduced. Not left as they are.

**Measured by:** the audit runs again and finds nothing in this list.

### 1.2 Make the numbers self-checking

**Done**, on branch `saans/results`. Three places it differs from the
plan below, each for a measured reason:

- README keeps its figures rather than replacing them with links. Each
  one is registered in `tools/results.py` and checked against the table
  on every push, which is stronger than a link (a link does not fail when
  the prose is wrong) and keeps README readable.
- The S3 detector table is in the slow tier, not CI. It takes ~160 s on
  20 cores, so most of an hour on a 4-core runner, and the obvious
  shortcut, `saans eval --quick`, uses seeds 0-1 -- the seeds the
  thresholds were tuned on. A table on tuning data is not a result.
- Integers are compared exactly. The first version allowed one unit in
  the last place everywhere, and a hand edit of `51/64` to `52/64`
  passed. Counts are not rounded floats.

Before any of it was built, every suite was run twice and the nulls
suite at 1, 4 and 20 workers: only wall-clock text and the printed worker
count moved. README's figures, including the four from the full Fresnel
sweep, all reproduce.

The pattern in §1.1 is the pattern in every stale number PR #5 found:
a figure typed into a document from memory of a result. The fix is
structural, not another round of proofreading.

`python3 evaluate.py --suite legacy` runs in ~15 s and prints a table.
`cargo run --release -p saans-cli -- eval` prints the S3 detector table.
`cargo run --release -p saans-sim --example impairment_cost` prints the
AGC table that NOTES.md §9 quotes.

- A script, `tools/results.py`, runs the fast suites and writes
  `docs/RESULTS.md`: one table per suite, each row labelled SYNTHETIC,
  with the command, the commit and the date in the header.
- A CI job regenerates it and fails on a diff, using the same numeric
  comparison `spec/golden/check_drift.py` already uses -- the lesson
  from the golden job applies here too; these are floats and two
  machines' libm differ in the sixteenth figure.
- README, NOTES and the site link to `RESULTS.md` for any performance
  figure and stop carrying their own copies. A number that appears in
  prose is either a link to a generated table or carries the words
  "typed, unreproduced".

The slow sweeps (`--suite nulls` without `--quick`, `--suite headroom`)
stay out of CI, as `ci.yml` already says; they are a measurement someone
reads, not a check. They can be regenerated by hand and committed.

**Measured by:** `git grep -E '[0-9]+\.[0-9]+ BPM' README.md NOTES.md`
returns only lines that are links or carry the marker.

**Risk:** the results table becomes a thing that goes red on an
unrelated change and gets `--no-verify`'d. Mitigation is the tolerance,
which `check_drift.py` has already shown to be settable with a measured
margin on both sides.

### 1.3 The double-rate breathing failure, which the gate does not catch

**Done** (part a), on branch `saans/octave`: the cross-link octave rule
in fusion. Doubled readings 6.9% → 3.4% with one link in a null, 25.2% →
16.5% with both, 1.1% → 0.2% with both clear; no class made worse,
coverage unchanged. Full table and its caveats in `NOTES.md` §10. Part
(b), a single-link signal, stays unbuilt: the simulator cannot represent
what it would need, so it waits for real recordings.

**Redesigned before building, 2026-10-24.** The single-link guard
described below cannot work in this simulator, and finding that out by
reading the code first is the reason to read the code first.
`synthetic.py:332` drives the chest with a pure sinusoid. At a true
stationary point the amplitude responds to the *square* of
displacement, and `sin²(2πft)` is a pure sinusoid at `2f` with no energy
at `f`. So at a whole-band null a doubled reading is bit-for-bit what a
person breathing twice as fast produces, and a guard that looks for
energy at `f/2` finds none to look at. Any guard tuned to pass here is
tuned to nothing. (Real breathing is asymmetric, so a real null may
leave energy at `f`; that is a question for the bench, not a parameter
to fit to a simulator that cannot represent it.)

What the physics does hand us is an asymmetry between links: a null
turns `f` into `2f` by a known mechanism, and nothing turns `f` into
`f/2`. **When two links disagree by 2:1, the lower one is right.** That
is a fusion rule, it lives in `saans-health/src/fusion.rs`, and ADR-006
already requires the second link it needs. Today fusion picks the
highest-confidence link -- which at a null is often the doubled one,
since it arrives at confidence near 1.0.

Measured baseline, the full `evaluate.py --suite nulls` in
`docs/RESULTS.md`, method `legacy`: at whole-band nulls **16.9% of
readings are at twice the true rate and 4.2% are wrong at confidence
>= 0.7**; partial nulls 5.3%; uniform placement 1.6%. (The quick sweep
gives 20.5% on a quarter of the scenes; the full one is the figure to
quote.)

Revised order: (a) the cross-link octave rule in fusion, measured on
two-link `saans-sim` runs with one link forced into a null; (b) only
then, and only if a measurement shows separation, any single-link
signal. The original text is kept below because the metrics in it are
still the right ones.

This is the one estimate-accuracy change that can be made and measured
now, because the failure reproduces in the simulator.

**What happens.** At a whole-band Fresnel null the static path cancels,
and amplitude becomes `|A_body · sin(φ(t))|` rather than
`|A_static + A_body · e^{jφ(t)}|`. The rectification doubles the
fundamental. The estimator locks onto twice the true rate -- and
`CAPABILITY_LIMITS.md:324-333` records it doing so at **confidence
1.00**, "unanimous about a doubled rate". The gate exists to turn an
unreliable number into one that abstains; this is the known case where
it does not.

`wifisense/features.py` has no handling for it. `grep -n 'double\|
harmonic\|half' features.py` finds the gross-motion band comment and
nothing else.

**What to build.** Not "pick f/2 when it exists": 24 BPM with nothing at
12 is a real breathing rate and the guard must not halve it. Instead,
when the band peak at `f` has a candidate at `f/2` that is inside the
breathing band and carries more than some fraction of `P(f)`, the
situation is ambiguous and the *confidence* comes down, not the rate.
The fraction is derived the way `STILL_INDEX` was (NOTES.md §8.1): from
the distribution on the nulls suite, not from taste. The second candidate
is reported alongside, the way `sideband_peak` already reports a triplet
score, so a consumer can see the ambiguity rather than a wrong number
with a clean confidence.

**Order.** Python first; `spec/golden/generate.py` gains a vector set for
the ambiguous case; Rust to 1e-9; C++ to 1e-4. ADR-002.

**Measured by.** `evaluate.py --suite nulls` already classifies scenes as
uniform / whole-band null / partial / mixed-sign (NOTES.md §6.5). Report,
per class: breathing error median and p90, abstention rate, and
**double-rate failures counted separately** -- the brief's own metric.
The win is double-rate failures at whole-band nulls moving from "reported
at confidence 1.00" to "abstained or flagged", with the uniform class
unchanged. If the uniform class gets worse, the fraction is wrong.

**Risk.** Breathing at exactly twice a sub-band artefact. Low, and the
guard lowers confidence rather than changing the answer, so the cost of a
false trigger is an abstention, not a wrong number.

### 1.4 AGC gain steps: build the detector, ship it off

**Done as a measurement; nothing shipped, and nothing should be.** Both
designs were built in the `impairment_cost` harness (`--corrections`)
and neither works. Common-mode removal takes the *clean* arm to 0%
occupied. The step detector does no harm but never rescues presence, at
any threshold from 2 to 5 -- even catching 98% of moves -- because each
correction is off by ~0.1 dB and the residuals are transients too. What
does work is an oracle that divides out the model's *true* gain:
presence and breathing back to 100%. So the fix, if one is needed, is
reading the AGC gain from the radio, and `PROTOCOL_S5.md` §1.1 route 1
now says so. Table in `NOTES.md` §9. The plan below is kept because it
was the right thing to test.

The largest simulated failure in the tree. NOTES.md §9's table:

| seated, 14 BPM | clean | esp32 |
|---|---|---|
| motion energy, empty room | 0.129 | **1.018** |
| motion energy, person seated | 0.351 | 0.882 |
| frames reporting occupied, % | 100 | **0** |

The contrast *inverts*: the empty room reads busier than the seated
person, so presence reports 0%. Breathing survives (it is a narrowband
feature; a gain step is broadband) and presence does not.

NOTES.md §9 then says, deliberately: "a per-packet common-mode gain is
something the DSP could normalise away, and the static suppressor
cannot. Changing the detectors on the strength of an uncalibrated model
is the opposite of what this layer is for." That stands. The whole
result rests on `agc_dither_prob`, bracketed between one gain move per
~4 minutes (presence survives) and per ~28 seconds (fails) -- and
nobody has measured which a real board does.

So: build it, measure what it costs on the clean arm, and **ship it
behind a switch, off**. The B1 recording decides the default.

**Two designs, both built, compared.**

1. *Common-mode removal.* Subtract the per-frame median log-amplitude
   across subcarriers before the static suppressor. Kills any gain step
   outright. Also kills a genuine whole-band attenuation -- a person in
   the line of sight dims every subcarrier -- so it trades a real signal
   for immunity. The clean-arm cost is the measurement that matters.
2. *Step detection.* A gain step is a jump that is the same dB on all
   56 subcarriers in one 50 ms frame. A person is not: Fresnel geometry
   gives a body a signature across the band. So the discriminator is the
   *spread* of the per-subcarrier jump. On a detected step, re-seed the
   static baseline by the step rather than letting it through as motion.
   Keeps the whole-band signal; costs a false step when a person moves
   very fast and very uniformly, which is a measurable thing.

**Measured by.** `cargo run --release -p saans-sim --example
impairment_cost` with each design on, all four arms. The table in NOTES.md
§9 gains two columns. The win is the esp32 arm's presence row moving from
0% toward the no-agc arm's 100% *while the clean arm's rows do not move*.
If clean moves, that is the cost, stated in the same table.

**Risk.** Exactly the one NOTES.md names. Mitigated by the switch, by
measuring the clean-arm cost, and by not calling either design the
default until a board has been listened to.

### 1.5 Measure how far the confidence is from a probability

**Done**, on branch `saans/reliability` (NOTES §12). Expected calibration
error 0.064 in uniform placement, 0.154 at whole-band nulls --
underconfident in one, overconfident at the top in the other, which means
no calibration map could fix it: the same 0.98 is right 100% of the time
in one population and 94% in the other. Calibration stays where this plan
put it, behind real recordings.

`CAPABILITY_LIMITS.md:324-333`: the confidence "is a gate, not a
calibrated probability". Nobody has measured how far. A reliability
diagram -- bin by reported confidence, plot fraction actually within
1 BPM per bin -- on each `evaluate.py` suite says whether 0.7 means 70%
or something else. On synthetic data this is a statement about the
simulator; it is still the measurement that has to exist before anyone
calibrates anything, and it goes into `RESULTS.md` under §1.2.

This is a measurement, not a fix. Calibration needs real data (B3).

### 1.6 Link diversity, in simulation, before the boards arrive

**Done**, on branch `saans/links` (`saans links`, NOTES §11). The fourth
board does not earn its place as a third receiver for detection: two
links take seats where a still person is mostly missed from 44 to 0 of
1,200, and a third adds nothing there, only less noise-driven false
occupancy. Along the way: eval's own 3-link rows mixed geometry with
link count and assumed two transmitters, and a first run at 300 seats
put the single-link miss rate at 0.5% when it is 2.8%.

ADR-006's decision rests on one simulated number: a second link takes
breathing coverage "from 94% to 100% and halves missed pauses"
(`NOTES.md:489-490`). Its own consequences clause says S6 "must measure
the blind-spot rate with 1, 2 and 3 links, in simulation and then on
hardware, before this is relied on."

`saans-sim` already has multi-link layouts (`scenarios.rs:40-44`,
`two_links()`). The simulation half of C1 can be done now:

- Same subject positions, 1 vs 2 vs 3 links at different angles.
- Report: blind-spot rate (scenes where every link is at a whole-band
  null), still-person false-empty, breathing coverage, per link count.
- `presence_quorum` set explicitly, because `fusion.rs` changes its
  default at 3 links (CAPABILITY_LIMITS §4).

**Measured by.** The table. If three links do not beat two by much, that
is the number that decides whether to buy a fourth board.

### 1.7 Build ADR-010, which I wrote and did not build

**Done**, on branch `saans/audit-retention`. One deviation from the ADR,
recorded in it: an absent `audit_retention_days` defaults to 365 or
`event_retention_days`, whichever is longer, so no existing config stops
booting on upgrade. `/api/health` now reports `retention_days` for the
history and the audit log as well as the events.

`docs/adr/010-audit-retention.md` is *decided, not built*. The decision
is one config key, one prefix parameter on `prune_logs`, a second call
in `Hub::housekeeping`, and `/api/health` reporting it. Its `changes_needed`
list in the PR #5 audit is the implementation plan. Half a day, with a
fake-clock test matching the one `history_retention_days` has. Leaving
an ADR "accepted" with nothing behind it for long is how ADRs stop
meaning anything.

While there: `/api/health` should report `history_retention_days` too.
ADR-010's own consequence -- "an operator who can see what has been
written but not what will be kept cannot check this decision is in
force" -- applies to the key that *was* built.

### 1.8 AlertService: the six `todo!()`s

**Done**, on branch `saans/alerts`, and the exit gate was run live as
well as in tests: a real gateway, frames over UDP, a local webhook, and
the dashboard clicked in a browser. Clicking found two bugs no Rust test
could: the WebSocket `hello` frame replaced the dashboard's capabilities
with a smaller set, so a connected dashboard forgot `audit` and would
have hidden the buttons; and the panel rebuilt every 2 s, wiping a
half-typed note. Both are fixed, and the first has a test.

The brief's A2. Not an accuracy change; the thing that makes the system
deployable, and the one the integrations (webhook, Telegram) hang off.
`reqwest` and `rumqttc` are already in `Cargo.toml` so no new
dependency. The spec is SAANS_PLAN.md §10.3 in full and the exit gate is
the brief's: a replayed night with a scripted breathing pause produces one
alert, a local webhook receives it, it is acked from the dashboard, the
audit line is there, and it survives a restart.

One honesty point, which the brief already makes: outbound messages carry
no vital numbers unless `include_health_details = true`. That is
"leaving the campus" in §10.3 and it is the privacy boundary of the
alert path. It gets a test that asserts the redaction on the wire.

### 1.9 MQTT and browser tests

A3 and A4, in the brief's order after A2. A4 converts the manual
checklist in `COMPLETION_REPORT.md` §6 into Playwright, and the brief's
exit gate is specific: breaking the "not shown to your role" label makes
it fail. Both are deployability, not accuracy.

**A3 (MQTT) done**, on branch `saans/mqtt`. `mqtt.rs` publishes the
§10.2 topic set from the privacy-filtered stream, with everything but
the vitals topic filtered as the `occupancy` role. Two deviations from
§10.2, both now written there: no occupancy `confidence` (the fusion
computes none, and the events' 0.6 / 0.8 / 0.9 is a heuristic lookup on
links usable; the payload carries that count instead), and a password or
vitals only to a broker on the same machine (rumqttc is built without
TLS). Tests run against an embedded rumqttd; each of five deliberate
breakages (a blocking reconnect sleep, vitals without the consent check,
no last will, no republish on reconnect, events filtered as admin) fails
one. "Ingest unchanged" is measured against the same config with the
broker up (~0.44 s for 1200 ticks either way), not against no MQTT at
all: publishing has a cost of its own (~0.15 s per 1200 ticks, debug
build), because the core only builds the output stream when something
consumes it. Not yet run against Home Assistant itself; the exit gate
used `examples/mqtt_watch.rs` as the subscriber.

**A4 (browser tests) done**, on branch `saans/browser-tests`. The
§6 checklist is seven Playwright tests in
`saans-gateway/tests/browser/`, against the gateway replaying a SYNTHETIC
night (`examples/fixture_night.rs`) with `--serve`, and a `browser` job
in CI. The brief's exit gate holds: making the dashboard show withheld
breathing as "not confidently detected" fails the occupancy-role test.
So do a 400 px element (the 375 px test), no cap on logins in flight
(the flood test) and the hall wrongly consented (the leak check), each
tried. The last one first passed: the fixture's hall was empty at the
end of the night, so there was nothing to leak. The hall now stays
occupied to the end, and the check looks for any breathing on its card
and its Signals panel rather than for one number.

### 1.10 The basic things

Each is small. Each has cost someone time in this project already.

- `CONTRIBUTING.md`: the master prompt's rules, so the next
  contributor -- or the next agent -- reads them before the code. Today
  they are in a chat message.
- `docs/adr/README.md`: an index. See §1.1.
- `.editorconfig`: the tree is rustfmt-clean as of `13acc2d`; this keeps
  the non-Rust files from drifting the same way.
- `rust/rust-toolchain.toml` to an exact pin the day a second contributor
  appears. The file says why it is `stable` today.
- `/api/health` reports every retention key. §1.7.
- Issue templates that ask for the SYNTHETIC / REAL label up front.

**Done**, on branch `saans/basics`, except the toolchain pin, whose
trigger (a second contributor) has not happened:

- `CONTRIBUTING.md` holds the brief's six rules (there were six, not ten:
  "ten" above came from a ten-bullet summary of them), the working
  method and every suite's command. It also has what the work since
  added, each with the PR it came from: a test must be able to fail;
  quoted numbers go through `results.py`; a platform-dependent failure
  is a finding, not a tolerance; how secrets are handled; withheld is
  not absent.
- `.editorconfig` states what the tree already does, from a file-by-file
  survey: LF, no trailing whitespace, a final newline, 4 / 2 / tab
  indents. Golden vectors and the vendored font licence are exempt,
  because those must stay byte-exact.
- `/api/health` already reported all three retention periods. What was
  missing was a test that would notice a fourth. It now walks the config
  for every `*_retention_days` key and fails until health reports it.
  Checked by removing one from health, and by adding a key to the
  config.
- Issue templates for a problem and for a result. The first field of
  each is SYNTHETIC or REAL, and it is required. Blank issues are off.

### 1.11 Frame authentication: a decision owed

Found triaging CodeQL's two `py/bind-socket-all-network-interfaces`
alerts on 2026-10-24. The bind is not the problem: the nodes are
remote, so something must listen on a network interface, and `--host` /
`udp_bind` narrow it. The problem is what the listener believes.
FeatureFrame v1 carries a CRC-32 (ADR-004), which catches corruption,
not forgery. Anything that can reach the UDP port can send frames for
any link: fake occupancy, hide a person, or for a consented space fake
breathing and so raise or suppress a health alert. For an advisory
health system that is the gap that matters.

Mitigation today: a sensor VLAN, a narrowed bind, a firewall. README
"Operating requirements" now says so; it said nothing before.

The fix would be a per-node key and a truncated HMAC-SHA256 tag in each
frame, which means FeatureFrame v2. That is a new ADR, the Python
reference, regenerated golden vectors, the Rust and C++ ports, key
provisioning in the firmware, and replay protection, because `seq`
alone does not stop a recorded frame being sent again. Worth deciding
before B2 puts frames on a real network, not after. Not started: the
wire format is a decision, not a task.

**Decided 2026-10-24: [ADR-012](adr/012-authenticated-frames.md).** It
uses HMAC-SHA256, not an AEAD: the Python reference stays standard-library
only, and a lost boot counter fails closed instead of repeating a nonce.
Off by default until nodes send v2. In ADR-002's order:

1. **Done:** the Python reference (`encode_signed`, `decode_signed`,
   `derive_node_key`, `ReplayGuard` in `wifisense/features.py`) and
   golden vectors (`spec/golden/feature_codec_signed`).
2. **Done:** the gateway. `saans-dsp` gained only the v2 byte layout and
   stays dependency-free; keys, signing, checking and the replay floor are
   in `saans-gateway/src/frameauth.rs`, which already had `hmac` and
   `sha2`. It has `[[frame_key]]` (the key in an environment variable),
   floors persisted in `data_dir/frame_counters.json`, refusals counted
   in `/api/health` and `/metrics`, and `--new-site-key` and `--node-key N`
   for provisioning. It is held byte for byte to the Python reference's
   golden frames.
3. The C++ codec with its own SHA-256 (RFC 4231 vectors), held to the
   golden bytes, and the firmware's key and boot counter in NVS.

---

## 2. Start when the boards land

In B1 → B2 → B3 order, as the brief says, and `docs/PROTOCOL_S5.md`
§1 is the first hour. What each one changes in this plan:

- **B1 §1.1 (AGC move rate)** flips §1.4's switch, or leaves it. It
  cannot be done with the firmware as it is -- nothing in
  `edge/firmware/main/` reads a gain, and PROTOCOL_S5 §1.1 now says so
  and gives the three routes.
- **B1 (bin layout)** sets `BinLayout`, after which every per-bin
  number in NOTES.md is re-run. Some will move.
- **B1 (noise floor)** replaces the simulator's noise constant with a
  measured one, and `evaluate.py --suite nulls` is re-run at it. The
  old table stays, next to the new one, labelled.
- **B3** produces the first REAL column in any table in the tree.
  Until then §1.2's `RESULTS.md` is entirely SYNTHETIC and says so in
  every row.

---

## 3. What this plan does not do, and why

Listed because each is the obvious next thing, and each has been
measured or decided against.

- **Phase.** `CAPABILITY_LIMITS.md:308-313` retired it: single-antenna
  phase under CFO/SFO. The memory of this project's own heart-rate work
  calls phase "the real fix" -- and it may be -- but it comes back only
  with real data showing the amplitude ceiling, as a research item, not
  a product change. See §1.1 for the stale NOTES.md line that still
  recommends it.
- **Learned models.** ADR-007. Nothing in the S0–S5 path. `research/`
  is allowed and empty.
- **RTI.** Measured: a six-line weighted-midpoint baseline beat it in
  every one of 15 configurations at N ≤ 12 (`NOTES-localize.md`). Zone
  labels, not metres; C2, after C1.
- **Continuous auto-recalibration.** Tempting, because the 30 s
  calibration goes stale over hours. But a detector that recalibrates
  during a stretch it *believes* is empty will calibrate to a person who
  is there and then never see them -- port rule 5's "worst possible
  failure" (`SAANS_PLAN.md` §4), built on purpose. `calib_age_s` is
  already in the frame (`spec/wire.md:42`); the right change is an
  advisory when it is old, which is an alert kind for §1.8.
- **Pose, through-wall, room geometry from WiFi.** `CAPABILITY_LIMITS.md`
  §1, §2, §2e. Not at any effort.

---

## 4. The order

1. §1.1 stale claims — half a day, no risk, pure record
2. §1.2 generated results — makes every later number self-checking
3. §1.3 double-rate guard — the one estimate-accuracy fix measurable now
4. §1.4 AGC detector behind a switch — robustness, ready for B1
5. §1.7 build ADR-010 — debt
6. §1.6 C1 in simulation — decides the fourth board
7. §1.8 AlertService — A2
8. §1.5 reliability diagram — folded into §1.2's table
9. §1.9 MQTT, browser tests — A3, A4
10. §1.10 as they come up
11. §1.11 frame authentication — a decision first, before B2

Boards arrive → §2, which pre-empts everything after item 4.

Each item lands as its own pull request, with the suites run and the
result stated, and with a line in `NOTES.md` saying what was measured
and what was not. The audit that found nine stale claims runs on each.
