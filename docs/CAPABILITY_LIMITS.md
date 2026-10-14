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

**[verified here]** Do not inherit skeleton code or keypoint-accuracy
claims from the upstream project this one started from. **[sourced]** Its
maintainer states in issue #299 that the skeleton keypoints are
"synthesized from CSI signal features", are "not detected by a neural
network", and that the 3D skeleton in its UI is a visualisation.

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
what it is for. Presence, breathing with a calibrated confidence and an
abstention, per-link signal quality and consent-gated vitals are all
working, measured (on synthetic data) and useful. The ceiling described
here is on *spatial* capability specifically.

The next three things worth building are in
[`COMPLETION_REPORT.md`](COMPLETION_REPORT.md) §9 and
[`SAANS_PLAN.md`](../SAANS_PLAN.md) §7. The first of them is the one
that matters: **no board has ever run this firmware.** Every number in
`NOTES.md` is simulated. Until a radio has been attached, the ceiling in
this file is the *least* uncertain thing about the system.
