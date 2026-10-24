# What this hardware cannot do, and why

Two capabilities get asked for repeatedly — **body-pose tracking** and
**through-wall sensing** — and both answers are decided by a single
physical quantity each. This file records those quantities so the
question does not have to be re-litigated, and so nobody buys hardware
on the strength of a demo video.

Claims below are marked **[verified here]** where I checked them against
the source, datasheet or code in this repository, and **[sourced]** where
they come from the feasibility study's literature search. Numbers without
a source are not in this file.

---

## 1. Body pose / skeletons — no, and not by trying harder

**The deciding quantity is the receive-chain count.**

The ESP32-S3 is 1T1R: one transmitter, one receiver, one antenna.
Every pose-from-WiFi system exploits three axes — antenna, delay, and
time. Saans has:

| Axis | What it gives | Saans |
|---|---|---|
| Antenna (spatial) | Where the reflection came from | **extent 1** |
| Delay (bandwidth) | How far away it is | 7.5 m at 20 MHz, 3.75 m at 40 MHz |
| Time (Doppler) | How fast it is moving | works — this is why breathing works |

With one receive antenna the array steering vector is the scalar `[1]`
for every angle, so the angular spectrum is **constant in angle** and a
1×1 covariance matrix has no noise subspace for MUSIC to find. The
angular axis carries *zero* bits, not few. That is a discontinuity, not
a gradient: no firmware change, no amount of training data, and no model
architecture recovers an axis whose measurement has one sample.

A standing human is roughly one tenth of a single range bin at this
bandwidth. Doppler yields velocity, not limb position.

### Zero angular bits is not, by itself, the whole argument

A reviewer pushed back on an earlier draft of this section, correctly:
"the angular axis carries zero bits" closes **angle-of-arrival**, and
AoA is only one route to pose. You could in principle try to learn pose
from temporal and Doppler structure without measuring angle at all. So
that half has to be argued separately, and here it is.

A 1T1R link measures `h(f, t)` — one complex number per subcarrier per
sample. Everything it knows about a body arrives on three axes, and all
three are exhausted:

- **Angular: zero bits.** As above.
- **Delay:** one range cell covers 3.75–7.5 m. Every limb, the torso
  and the head sit inside the same cell. No depth separation exists to
  attribute anything to a body part.
- **Doppler:** the *aggregate* radial velocity spectrum, summed over
  every scatterer on the body, with nothing that attributes a velocity
  to the limb that produced it.

So the forward map from pose to measurement is **many-to-one by
construction**: an unbounded set of joint configurations produces the
same aggregate Doppler at the same unresolved range. A skeleton
recovered from that is not being measured; it is being supplied by a
learned prior over human bodies, conditioned weakly on the radio.

That is not a theoretical worry — it is the exact failure mode already
on record. It predicts a model that looks excellent in its training room
and collapses elsewhere, because the prior is doing the work and the
prior was fit to that room. Which is what the cross-environment numbers
above show: 197 mm in-domain against ~370 mm cross-environment.

**This is falsifiable, and stating it that way is the point.** If a 1×1
pose result appears that holds up against held-out *subjects* and
held-out *rooms* — not a random split of one dataset — then this section
is wrong and should be rewritten. The claim is not "no one has managed
it yet". The claim is that the measurement does not contain the answer,
so anything that appears to recover it is reporting its prior. Check the
cross-domain split before believing any result that says otherwise.

**[sourced]** Every published WiFi pose system uses **≥ 3 receive
antennas** — MM-Fi and MetaFi at 1 TX × 3 RX; Person-in-WiFi and
DensePose-from-WiFi at 3×3; WiPose and Person-in-WiFi-3D at 3 receivers
× 3 antennas. No 1T1R pose result exists, peer-reviewed or otherwise.

**And the capability is weaker than the demos suggest even on correct
hardware.** **[sourced]** Cross-environment per-joint error is 254–296 mm
for the 2026 state of the art and 369.5 mm for the MM-Fi WiFi baseline,
in rooms around 8.5 × 7.8 m. At 27 cm per joint the output is a
body-sized blob — and it is produced by a model trained against cameras
in the training rooms, which is an awkward foundation for a system whose
entire argument is that there is no camera.

### The weaker version fails too, and this project already measured it

"Just tell standing from sitting from lying" is not a safe fallback.
`NOTES.md` §8.2 records that the collapse detector caught **100% of
falls** and also fired on **100% of sit-downs, 92% of readers walking in,
and 97% of fidgeting nights**, because amplitude-only 2-D WiFi cannot
see height. It ships off by default and should stay off.

**Do not inherit skeleton code or keypoint-accuracy claims from any
project that has not published a held-out-room evaluation.** A
demonstration that reports only a random split cannot distinguish a
model that measured something from one that is drawing its prior, and
those look identical in a video. That is the test to apply, to anyone's
work including ours — it needs no argument about who built what.

---

## 2. Through-wall — three different things wearing one name

**The deciding quantity is swept bandwidth, for one of the three.**

### 2a. Through-wall imaging or pose — permanently out of scope

Range resolution is `Δr = c / 2B`. Ten-centimetre depth bins need about
**1.5 GHz** of swept bandwidth. HT20 gives 7.5 m; HT40 gives 3.75 m.

**[sourced]** Every famous through-wall result — WiTrack, WiTrack2,
Vital-Radio, RF-Pose, RF-Capture — is an **FMCW radar sweeping ~1.7 GHz**
on a USRP with 3–10 directional antennas. "WiFi frequencies" in those
papers means the *band*, not the protocol. They are not 802.11 CSI
results and no amount of work on an 802.11 radio approaches them.

Reaching that means abandoning 802.11 for a radar front-end — and
**[sourced]** WiTrack's front-end was *built*, not bought; nothing
equivalent is sold.

### 2b. Through-wall presence and gross motion — plausible, unmeasured

A wall costs these outputs **SNR, not resolution**. Presence rests on
temporal variance, and nothing Saans emits depends on range or bearing,
so a partition attenuates the signal rather than invalidating the method.

**[sourced]** The nearest hardware evidence is the only peer-reviewed
ESP32-S3 NLOS result: three coarse classes over 18 m of hallway linking
brick-separated rooms, with directional antennas at both ends and CSI at
100 Hz — 5× Saans's rate. That is a long-range NLOS link result, not a
per-wall penetration measurement. Treat it as demonstrated-adjacent and
measure it yourself.

### 2c. Through-wall breathing — not demonstrated on this hardware class

**[sourced]** Every through-wall respiration result in the literature
needs **at least two coherent receive chains**: FarSense's entire
contribution is the *ratio* between two RX antennas; PhaseBeat is a
two-antenna phase difference on an Intel 5300. The one single-antenna
ESP32 respiration paper contains **zero** through-wall experiments and
needs 40 MHz at 120 Hz.

Status: plausible, undemonstrated. Measure before promising.

### 2d. The product fact that outranks all three

**A link with no range and no bearing resolution cannot tell which side
of the wall the person is on.**

Making links through-wall-capable converts corridor traffic and the next
classroom into **occupancy false positives** for the instrumented room —
a correctness regression in the one feature that works today. For most
campus rooms, through-wall reach is a liability to be measured and
suppressed, not a feature to pursue.

And do not relax the confidence gating to "see through" a wall. A wall
lowers BNR, which lowers `breathing_conf`, which abstains. **That is
correct behaviour.** The system saying "I don't know" is the product.

---

## 2e. Room size and internal wall layout — no, and for a *third* reason

**The deciding quantity is coherent aperture extent.** Not antenna
count, not bandwidth. Say that out loud, because the bandwidth argument
in §2a does **not** carry over, and a reader who assumes it does will go
shopping for UWB — which would not fix this either.

In the near field, depth comes from wavefront curvature, not pulse
compression. Depth of focus is `4λR²/L²`. At λ = 12.3 cm, resolving a
wall at R = 3.5 m to 30 cm needs an aperture of **L ≈ 4.5 m — larger
than the room**. Cross-range at that aperture would be fine
(λ/L = 2.0°, 14 cm at 4 m), which is precisely why bandwidth is not the
binding term here.

**Saans has L = 0.** Two or three nodes bolted to walls form no
aperture, their positions are typed in by a human, and each PLL is
free-running at 10–40 ppm with no inter-node phase reference.

The backup limit, for the fixed configuration you actually have: 56
subcarriers span 55 × 312.5 kHz = **17.19 MHz** edge to edge, so the
bistatic path-length cell is `c/B` = **17.45 m** (with c = 3 × 10⁸ m/s).
`NOTES.md` §6 uses the other convention, 56 × 312.5 kHz = 17.5 MHz of
occupied bandwidth, which gives 17.14 m; neither is a typo for the other,
and nothing below depends on the 2% between them. In a 3.5 × 3.5 × 3 m
room the six first-order reflections sit 0.70–2.22 m of excess path from
the line of sight. **All of them, plus the LoS, land in one cell** — at
20 MHz, at 40 MHz, and at a perfectly stitched 83.5 MHz whole-ISM-band
span (3.6 m cell). Roughly 900 image sources inside the first cell
against at most ~19–37 degrees of freedom in a 56-subcarrier snapshot.
That is rank deficiency, not noise, and no amount of integration fixes
it.

**Long integration reduces variance. It does not improve delay
resolution.** That conflation is the single most common way to get this
question wrong.

### Three answers that stop this being re-asked

- **802.11mc FTM ranging** is the first thing anyone suggests, and the
  ESP32-S3 supports it in shipping firmware. **[sourced]** Measured
  error is ~1.8 m on S3, with an S2 study showing only ~75% of readings
  under 5 m and excursions to 20 m; Espressif's own driver guide says
  RTT distance measurement "is not accurate". A monitored room is 4–8 m
  across, so the error is the size of the thing being measured. It
  cannot even reliably detect that a node has been moved.
- **Wiffract (MobiCom 2022)** is the strongest WiFi geometry result that
  exists — and it needs a ground robot scanning 140 × 42 positions over
  a 2 m × 1.2 m aperture with 9 antennas. **[sourced]** The paper states
  outright that WiFi imaging "cannot capture the depth information".
  Room size *is* depth.
- **Acoustics**, for the humility: **[sourced]** five ordinary
  microphones in arbitrary positions, spacing measured with a tape
  measure, reconstructed all six walls of a lecture room (PNAS 2013),
  recovering a 90.0° corner as 90.1°. Sound is ~875,000× slower, so a
  20 kHz sweep gives a **0.86 cm** range bin against 7.5 m for WiFi
  HT20. A few hundred rupees of microphones beats every radio system in
  this literature.

### So: ask, do not infer

A laser distance meter costs about ₹2,000 and is exact. Adding
`geometry` to `SpaceConfig` — node coordinates, room extent, survey
provenance — is roughly a day's work and backward compatible. **Effort
spent inferring what an installer could simply type in is waste**, and
that sentence is here so the question is not re-litigated.

### A warning about the coverage map in the Explorer

The Explorer's Fresnel coverage map is honest **because the simulator
knows its node positions exactly**. Do not assume it transfers to a real
room. **[sourced, measured by the study against this repo]** With node
positions off by 5 mm the map's correlation with truth falls to 0.87; at
1 cm, 0.65; at **2 cm, 0.27**. A human with a tape measure does not
place a node to 5 mm.

So: a to-scale room outline with node markers and a door is fine from
surveyed config. **A per-cell blind-spot heatmap over a real room is
not**, and neither is an install-time CLI printing "median 2.7% blind
area" from a surveyed layout. Those numbers would be precision the
survey cannot support.

### The weaker claim that might actually be useful

"Is this room *different* from when we calibrated?" — a node was moved,
a door is propped open, a cupboard now blocks a sensor. That is change
detection in the static channel, not mapping, and it is the one place
inference earns its keep here.

Treat it as a **measurement task, not a capability**. The nearest
single-antenna amplitude-only result **[sourced]** reports 99.6% TPR at
0% FPR — but the event it detects is *rotating a directional antenna*,
which perturbs the channel far more than furniture does, and the same
paper notes that two people walking leave the shape of |H| "largely the
same". It also uses a deep autoencoder, which ADR-007 forbids until the
DSP ceiling is measured on real data. Build the cheap version from
existing telemetry, measure its false-alarm rate on replay, and promise
nothing until that number exists.

---

## 3. If you want these capabilities anyway

The honest path does not run through Saans nodes.

**Step 0, free, and it will probably end the conversation:** download
MM-Fi — the public NeurIPS 2023 benchmark — and train its published WiFi
baseline yourself. **[sourced]** You will reproduce ~197 mm mean
per-joint error on a random split and ~370 mm on the cross-environment
split, which is the deployment setting for a campus room. Look at a
37 cm-per-joint skeleton overlaid on a person and decide whether you
still want it.

**If yes**, you need *coherent* receive hardware, which means a shared
clock **and** a shared phase reference — independently crystal-clocked
nodes scattered around a room cannot be made coherent after the fact.
**[sourced]** An ESPARGOS-class array (eight ESP32s on one PCB with a
shared 40 MHz reference *plus* a wired RF phase reference) is the most
platform-native option, but there is no published pose result on it, and
its advertised price does not appear in the paper — build a real BOM
including Indian import duty before committing.

**For through-wall, there is no cheap path at all.** **[sourced]** A USRP
B210 was listed at ₹305,250 at an Indian retailer; HackRF gives 20 MHz,
which is useless for ranging. India's 2018 UWB rules are also a far
worse regulatory position than the delicensed 2.4 GHz band this project
already sits in.

---

## 4. Traps that look like progress

- **Revisiting RTI**, including as a through-wall feature. It already
  lost to a six-line weighted-link-midpoint baseline in all 15
  configurations at 4–12 nodes (`NOTES-localize.md`). The root cause —
  RTI's shadowing model mismatching coherent-multipath-dominated CSI —
  does not improve with more nodes.
- **Jumping from one link straight to three.** **[sourced]**
  `fusion.rs:123-130` silently changes the default presence quorum from
  1 to 2 at three links, and still-person false-empty goes back up from
  0.4% to 1.3%. Set `presence_quorum` explicitly and measure.
- **Chasing CSI phase** as the next big thing. `NOTES.md` §5 item 4 says
  it is "the highest-value next piece of work"; `RESEARCH.md` §5a quotes
  RespirFi's own footnote that single-antenna CSI phase is unreliable
  under CFO/SFO, which is *why* that paper is amplitude-only. These two
  contradict each other and §5 item 4 is the stale one.
- **Buying mmWave modules to "retire the localisation result".** The
  cheap 24 GHz modules are moving-target XY trackers with a 6 m, 120°
  field — a different product, not an upgrade to this one.

---

## 5. What this does not mean

None of the above says the system is limited in the ways that matter for
what it is for. Presence, breathing with a confidence and an abstention, per-link signal
quality and consent-gated vitals are all working, measured (on synthetic
data) and useful.

One correction to this file's own earlier wording: it called the
breathing confidence "calibrated". That is stronger than the evidence.
`NOTES.md` records the estimator locking onto **twice** the true rate at
whole-band blind spots, and the Explorer review observed it doing so at
**confidence 1.00** — i.e. unanimous about a doubled rate. The gate
turns an unreliable number into a mostly-reliable one that abstains; it
does not make the confidence a calibrated probability, and the double-
rate failure is the known case where it does not save you. The ceiling described
here is on *spatial* capability specifically.

The next three things worth building are in
[`COMPLETION_REPORT.md`](COMPLETION_REPORT.md) §9 and
[`SAANS_PLAN.md`](../SAANS_PLAN.md) §7. The first of them is the one
that matters: **no board has ever run this firmware.** Every number in
`NOTES.md` is simulated. Until a radio has been attached, the ceiling in
this file is the *least* uncertain thing about the system.
