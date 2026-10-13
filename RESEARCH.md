# WiFi CSI sensing: verified research findings

Deep research pass, Oct 2026. 110 agents, 27 sources fetched, 134 claims
extracted, 25 adversarially verified (3-vote panel, 2/3 refutes kills a
claim), **17 confirmed / 8 killed**, merged to 14 findings.

Every result below is **commodity WiFi with a passive human target**. No
FMCW radar, mmWave or UWB result was admitted — that conflation check was
the primary guard on this pass and it passed.

---

## 0. What this changes about the plan

Three findings invalidate the multi-ESP32 multilateration design:

1. **ESP32 is 1T1R — one RF chain.** It cannot run MUSIC, ESPRIT, SpotFi
   AoA, or FarSense-style two-antenna CSI ratio. Those need ≥2 receive
   chains on *one radio sharing one local oscillator*. **Two separate
   ESP32 nodes do not substitute** — their PLLs free-run independently.
2. **Linear-fit phase calibration destroys absolute ranging.** Physical
   ToF is a linear slope across subcarriers; so is the random SFO/PDD
   offset. Least-squares removal of the latter strips the former. Any
   pipeline that sanitizes phase this way and then multilaterates from
   absolute ranges **is broken by construction**.
3. **There is no published median localization error** for a passive human
   using N spatially separated single-antenna nodes with no coherent array
   and no absolute ranging. Every localization result found used an
   antenna array, or required manually supplied person location, or used
   Wi-Fi 6E/7 multi-band hopping.

A fourth finding makes it worse: **adding nodes costs sample rate.** The
one published multi-node ESP32 mesh study used backend-orchestrated TDMA
across 9 nodes, yielding ~1.4 Hz aggregate CSI. At that rate it detected
ambulatory motion ~100% of the time but **sedentary presence only 38%**.
Breathing extraction needs ≥10–20 Hz. Round-robin polling across many
nodes trades away exactly the temporal resolution our pipeline depends on.

---

## 1. Phase sanitization and CSI ratio

**CSI Ratio (two antennas, one receiver)** — `R = H_j1,k / H_j2,k` cancels
all common-mode multiplicative distortion (gain variation, random phase
noise) because both chains share one RF oscillator. FarSense states the
precondition verbatim: *"the time-varying phase offset is the same across
different antennas on a Wi-Fi card as they share the same RF oscillator."*

Cost: you lose absolute signal strength and absolute phase reference.

**Trap:** the ratio's phase is **not linear in target displacement** — it
is a Möbius transform of the target signal. Anyone assuming CSI ratio
hands them clean linear phase is wrong. *(confidence: high, 2-1)*

**ESP32 caveat:** ESP-IDF multi-antenna support is GPIO-controlled RF-switch
antenna *selection* (one active path per packet, e.g. WROOM-DA), **not**
simultaneous dual-chain reception. `H_1` and `H_2` for the same packet are
unobtainable.

**SA-WiSense cross-subcarrier CSI ratio (CSCR)** is the only verified
single-antenna substitute: divide CSI across two *subcarriers* on one
chain, exploiting frequency diversity instead of spatial diversity.

- Cancels the subcarrier-independent CFO term **exactly**
- Packet boundary delay removed by a **separate pre-averaging stage**
- **SFO is NOT cancelled** — survives as a constant offset the paper
  argues is stable over minutes

*(confidence: high, 3-0, merged from three claims)*

**Open question:** does CSCR deliver usable SNR at ESP32 bandwidths? The
physics argument against is in the evidence: adjacent subcarriers across
40 MHz at 2.4 GHz see near-identical dynamic path phase, making the
cross-subcarrier differential intrinsically weaker than the cross-antenna
one. SA-WiSense needed a selection stage over **218 subcarriers** (HT40,
120 Hz). An ESP32-C6 gives ~52–64 and possibly only 20 MHz.

## 2. Localization and tracking

**2025 SOTA passive AoA** (IMWUT, Dec 2025) reattributes the long-standing
±45° limit to **noise-induced phase wrapping** rather than linear-array
geometry, extending usable range to **±85°** via iterative phase
adjustment. Preprocessing is temporal signal differencing with a
*dynamic* (adaptively chosen) differencing interval to cancel static
multipath. Reported **average angle error below 2°** vs ~7° for prior SOTA.

**This is an angular metric on an antenna array — never report it as
metre-level positioning on ESP32 nodes.** *(confidence: high, 3-0)*

The paper could not be read in full (dl.acm.org 403s automated access —
bot-blocking, not a paywall; it is bronze OA), so its node count,
bandwidth, room size, antenna count and NIC are **unverified**.

**Angle estimation under respiration-only motion** is claimed to remain
robust — i.e. a motionless occupant could in principle be angularly
localized from chest displacement alone. Strategically this is the most
interesting result in the whole pass for combining our breathing extractor
with bearing estimation. But it needs a coherent array and **carries no
published error figure**. *(confidence: medium, 3-0)*

**Widar3.0 BVP** (body-coordinate velocity profile) is the best-specified
multi-node passive tracking algorithm available, published as fully
implementable math. Prerequisites are severe:

- Known Tx/Rx coordinates
- Deployment used **1 transmitter and 6 receivers**
- Derivation assumes only single-reflection paths are prominent
- **The published formulation requires the person's global location and
  orientation to be supplied manually**

That last point is circular if your goal *is* localization. *(high, 3-0)*

**Frequency-diversity discrete channel sampling (UWB-Fi / CCS-Fi)** is the
published route past metre-scale range resolution: hop irregular
2.4/5/6 GHz slices to synthesise ~4.7 GHz effective bandwidth for ~20 cm
resolution with 20 channels. Requires Wi-Fi 6E/7 NICs plus driver-kernel
hop control, **categorically out of reach for ESP32**. Conventional IFFT
and MUSIC **cannot** be applied to the resulting irregular samples. The
20 cm figure is a compressive-sensing *theory* result from companion work,
not a measured passive-human deployment number. *(high, 3-0)*

## 3. Vital signs

**RespirFi** is the best-verified commodity-WiFi passive respiration
result in this evidence base:

| System | RR MAE | Waveform PCC |
|---|---|---|
| **RespirFi** | **0.05 BPM** | **>85%** |
| M2Fi | 0.07 BPM | 75.10% |
| WiResP | 0.14 BPM | 74.29% |
| FarSense | 0.22 BPM | 69.01% |

Conditions: **Intel AX210, 802.11ax, 6025 MHz centre, 160 MHz bandwidth,
100 packets/s, 1 Tx × 2 Rx laptop-to-laptop**, seated stationary subjects,
3 environments, 30 participants, smart respiratory belt as ground truth.
*(confidence: high, 2-1 — dissent was about scoping, not facts)*

**RespirFi's location-robustness algorithm is directly portable to our
Python pipeline and needs no hardware.** This is the single most
actionable finding in the pass *(high, 3-0)*:

1. Per-subcarrier **BNR** (breathing-to-noise ratio) scoring
2. Hybrid adaptive threshold `eta = max(80th percentile, 2)`
3. **Spectral clustering** of surviving subcarriers into two monotonic
   trend groups via a **ternary cosine-similarity affinity matrix**
4. **Sign-flip the second group**
5. Average across the subcarrier dimension

It is a Fresnel-null / blind-spot countermeasure implemented in the
**subcarrier domain** rather than by moving nodes — which is exactly the
degree of freedom we have and the physical-placement one we don't.

**Reproducibility warning** *(high, 3-0)*: none of the headline numbers
are reproducible on ESP32-S3/C6. RespirFi used 6 GHz / 160 MHz / 100 pps
on an AX210 with PicoScenes-class tooling. SA-WiSense's ESP32 testbed used
HT40 with 218 subcarriers at 120 Hz and never names the chip variant.
ESP32-S3/C6 give 2.4 GHz, 20/40 MHz, ~52–64 subcarriers, and the CSI
payload layout **varies by chip family** and by which LLTF/HT-LTF/
STBC-HT-LTF fields are enabled.

## 4. Activity recognition and generalization

**Published 90%+ CSI activity/gesture accuracies are in-domain ceilings
and carry no information about an unseen room.** The leading 2026 survey
(accepted IEEE COMST) states outright that its large dataset table reports
in-domain results copied from original papers, and **nowhere in the entire
survey is there a single paired in-domain vs cross-domain degradation
percentage** — only the qualitative "performance degrades under large
domain gaps". *(high, 3-0)*

**The canonical open recipe for cross-environment generalization** is
adversarial domain-invariant training where "domain" is defined purely as
human **location**: the network subtracts a weighted domain-prediction
loss from the gesture-label loss. Published as runnable Keras code **with
no accuracy evaluation whatsoever**. *(high, 3-0)*

**ESP32 is a validated CSI receiver for this class of work:** the
WiGesture/CrossFi dataset used a six-antenna router as transmitter and a
**single-antenna ESP32-S3 as receiver** — 8 subjects, 6 gestures, 2.4 GHz,
100 Hz, 52 subcarriers, 1.5 m Tx-Rx, bistatic 1Tx/1Rx.

---

## 5. Claims the verification panel KILLED

Do not carry these forward as fact. Listed because several are repeated
widely and sound authoritative:

| Killed claim | Vote |
|---|---|
| The "dilution effect" disproves multi-node sensing | 0-3 |
| Sample rate is the *demonstrated* confound behind dilution (mechanism plausible, 1.4 Hz/38% numbers real, but causal attribution unproven) | 0-3 |
| SA-WiSense's GASS genetic-algorithm subcarrier selector is a validated recipe | 0-3 |
| 20 MHz → 50 ns / ~15 m resolution; Wi-Fi 7's 320 MHz caps multi-person separation at ~1 m | 0-3 |
| Conjugate multiplication and division are the *only* two ways to remove SFO/PDD | 0-3 |
| ESP32 phase coherence requires shared crystal + 2.4–2.5 GHz reference over matched-length coax | 1-2 (plausible, corroborated in passing, not established) |

### 5a. CORRECTION (post-pass, verified against the paper itself)

Two of the killed claims above were killed **wrongly**. The RespirFi full
text was obtained afterwards and settles both directly. Credit to the peer
session that pulled the paper.

**RespirFi IS amplitude-only.** Footnote 1, verbatim:

> "for single-antenna systems, CSI phase is highly susceptible to
> hardware-induced impairments such as carrier frequency offset and
> sampling frequency offset, which makes it unreliable for practical use.
> Therefore, RespirFi focuses exclusively on CSI amplitude for robust and
> scalable respiration sensing."

The panel killed this on a 1-2 split. It was correct as originally stated.
This is good news for ESP32-class hardware: the algorithm needs no phase.

**But equation 6 is the sting, and it is worse than the headline.** RespirFi
bounds when subcarrier diversity can escape a Fresnel blind spot:

```
B * (d_bre - d_LoS)  >=  0.255 * c          [eq. 6]
```

with `0.255c = 7.645e7 m*Hz`. Working it both ways:

| Bandwidth | Required excess path d_bre - d_LoS |
|---|---|
| WiFi 7 @ 160 MHz | **0.478 m** (paper states 0.47 m - checks out) |
| ESP32 HT-LTF, 17.5 MHz (56 x 312.5 kHz) | **4.37 m** |
| ESP32 LLTF only, 16.25 MHz (52 subcarriers) | **4.70 m** |

Use the span of subcarriers ACTUALLY REPORTED, not the nominal 20 MHz channel.
The paper's argument is "shift to a different subcarrier", so the usable
delta-f is between measured subcarriers; the guard bands carry nothing.
Taking the full 20 MHz would give an optimistic 3.82 m. **Pin 4.4 m.**

**On ESP32 at 20 MHz, subcarrier selection cannot rescue a blind spot
unless the chest-reflected path exceeds the line-of-sight path by ~4.4 m.**
In a normal room, for anyone near the LoS line, the excess path is well
under 1 metre. The entire band sits at essentially the same Fresnel phase.

This does not make the algorithm wrong - it makes it bandwidth-gated. It
pays off on a 160 MHz AX210 and largely does not on a 20 MHz ESP32.

**An extended torso does NOT average the nulls away** (measured by the peer
session, and it corrects an error of mine). The tempting argument - a
wavelength of 12.3 cm means a Fresnel crossing every ~6.2 cm of excess
path, so a 0.3 m torso spans ~5 crossings and averages them out - is
geometrically wrong. A torso FACES the link, so its width lies along the
Fresnel ellipse TANGENT, where excess path changes only to second order.
The torso does not span 0.3 m of excess path; it spans almost none.

More fundamentally: a rigid reflector moving as one block presents a single
net body vector, so the null exists regardless of extent. Extent changes
WHERE the null falls and HOW DEEP the modulation is, not WHETHER there is
one. Measured first-order sensitivity at a null (body at y=3.03, tx (1.5,2.5),
rx (4.5,2.5)): point body 0.02/0.04/0.14 min/med/max across 56 subcarriers,
versus 0.00/0.03/0.12 for 5 points spread over 0.3 m. Still a null.

**But depth-wise motion genuinely does average.** A person WALKING moves
along the bisector, through successive Fresnel zones, so the link's variance
integrates over many crossings. This is why VRTI (which measures variance
under motion) is robust where static-subject respiration sensing is not -
and it is the physical reason the P0b localization approach can work on
hardware where breathing-in-a-blind-spot does not.

---

## 6. Coverage gaps — treat as "not researched", not "no literature"

Angles 1–3 got substantive verified coverage. **Angles 4 and 5 did not.**
Nothing in this evidence base verifies:

- IEEE 802.11bf ratification status or shipping hardware support
- Beamforming feedback (BFI) as a firmware-free CSI alternative
- Nexmon / Raspberry Pi, ath9k/ath11k CSI export, AX210 tooling, PicoScenes
- Open dataset sizes, modalities, licensing (Widar3.0, SignFi, UT-HAR,
  WiAR were *named in a survey table only*, not characterised)
- **VMD, CEEMDAN/EMD, wavelet decomposition** for respiration/heartbeat
- **The intermodulation problem** (f_heart ± k·f_breath) — our own finding
- Random body movement cancellation
- **Heartbeat extraction or any ECG/pulse-oximeter comparison — every
  vital-sign result here is RESPIRATION ONLY**
- Multi-person vital-sign separation; blind source separation
- Fall detection; people counting
- DensePose-from-WiFi replication status
- Quantitative domain-adaptation / few-shot results beyond one CrossFi point
- **Radio tomographic imaging (RTI/VRTI)** — a significant omission. It is
  the one localization family that single-antenna nodes genuinely support,
  and this pass missed it entirely. Being addressed in phase P0b.

Several verifiers exhausted the shared search budget before running
adversarial contradiction sweeps, so "no contradicting evidence found"
is in those cases **absence of evidence, not evidence of absence** —
applies to UWB-Fi/CCS-Fi, the IMWUT angle result, and the Widar3 tutorial.

**Source quality ladder.** Peer-reviewed: IMWUT 2025 angle estimation,
RespirFi (IEEE IoT-J 2026), generalizability survey (IEEE COMST 2026),
FarSense (IMWUT 2019), UWB-Fi (MobiSys 2024), CCS-Fi (INFOCOM 2025).
Unreviewed preprints: ISAC primer, SA-WiSense, dilution study. The
Tsinghua tutorial is June 2022 — dated, but its BVP derivation is
mathematically stable and valuable as reference math.

---

## 7. Revised plan

**Do now, no hardware required:**

1. **Port RespirFi's subcarrier selection** (§3) — BNR scoring, adaptive
   threshold, spectral clustering, sign-flip, average. **Revised
   expectation, see §5a:** eq. 6 gates this on bandwidth, and at ESP32's
   20 MHz it should deliver little or no gain in a blind spot. Build it
   because it is correct, cheap, and pays off at 160 MHz — but a large
   measured gain on 20 MHz synthetic data indicates a bug in the scene
   model, not a win. Decide the falsifying result before measuring.
2. **Add Fresnel-null modelling to the synthetic generator** so there is
   something for that algorithm to actually defeat. Our current generator
   has per-subcarrier gain variation but no true nulls.

**Hardware decision, revised:**

- **One ESP32-S3/C6 ($6–10)** remains the right first buy — validated as a
  CSI receiver, sufficient for presence + breathing + activity, and the
  cheapest way to replace synthetic data with real radio.
- **Do not buy 4 nodes expecting multilateration.** There is no published
  accuracy figure for that configuration, and TDMA across nodes costs the
  sample rate breathing needs.
- **If localization is the real goal**, the honest paths are: an
  **Intel AX210 + PicoScenes** setup (2 Rx chains, 160 MHz — the hardware
  every good result in this pass actually used), or an **ESPARGOS-style
  phase-coherent ESP32 array**, which is a hardware clock-distribution
  project, not a software one.

**Still unanswered and worth a second research pass:** 802.11bf status,
open datasets (we could develop against real CSI with *zero* hardware),
heartbeat algorithms, and the intermodulation problem we hit ourselves.

---

## Sources

Primary: [generalizability survey (COMST 2026)](https://arxiv.org/abs/2503.08008) ·
[CrossFi](https://arxiv.org/abs/2408.10919) ·
[SA-WiSense](https://arxiv.org/abs/2507.17623) ·
[ISAC primer](https://arxiv.org/abs/2601.12980) ·
[FarSense (IMWUT 2019)](https://doi.org/10.1145/3351279) ·
[IMWUT 2025 AoA](https://doi.org/10.1145/3770662) ·
[Widar3.0](https://ct.cswu.me/papers/mobisys19_widar3_paper.pdf) ·
[Tsinghua CSI tutorial](https://arxiv.org/abs/2206.09532) ·
[SpotFi](https://web.stanford.edu/~skatti/pubs/sigcomm15-spotfi.pdf) ·
[dilution study](https://arxiv.org/abs/2602.10823) ·
[SpaceBeat multi-person vitals](https://par.nsf.gov/biblio/10554426-spacebeat-identity-aware-multi-person-vital-signs-monitoring-using-commodity-wifi) ·
[CSI dataset/reproducibility survey](https://www.researchgate.net/publication/401244480_A_Survey_on_CSI-based_Wi-Fi_Sensing_Datasets_and_Models_with_a_Focus_on_Reproducibility) ·
[IEEE 802.11 WG](https://www.ieee802.org/11/)

Full 27-source list and per-agent journal:
`~/.claude/projects/.../workflows/wf_c96fc471-6c0/journal.jsonl`
