# Engineering notes

Why the code is the way it is: each design decision, what was expected,
what was measured, and what changed. Every number here comes from a
script in this repository; synthetic results are labelled SYNTHETIC.

---

## 1. What this repository is

`wifisense/` is a small, dependency-free Python pipeline for WiFi channel
state information (CSI): static suppression, subcarrier fusion, presence,
breathing and heart rate, each with a calibrated confidence. It is the
**reference implementation** of Saans (`SAANS_PLAN.md`): the Rust gateway
(`rust/`) and the C++ ESP32 node (`edge/`) must reproduce it on golden
vectors (`spec/golden/`).

## 2. Working principles

- **Measure before claiming.** Every capability ships with an evaluation
  script that prints a table, and synthetic results say SYNTHETIC.
- **Confidence and abstention.** Every estimate carries a confidence
  calibrated against what noise alone would produce; "unknown" is a valid
  answer and is preferred to a plausible guess.
- **Explainable DSP first.** Simple, inspectable signal processing whose
  failure modes can be named (blind spots, intermodulation, packet loss)
  comes before any learned model (ADR-007).
- **Portable core.** Plain float arithmetic, no dependencies, so the same
  algorithm runs in Python, Rust and on a microcontroller.

## 3. The hardware path

An ESP32-S3 (one RF chain, 2.4 GHz, HT20: 56 subcarriers at 312.5 kHz)
exposes per-subcarrier CSI through ESP-IDF (`esp_wifi_set_csi_config`,
`esp_wifi_set_csi_rx_cb`). One node transmits (ESP-NOW broadcast), others
capture CSI from those frames. The raw debug format (magic `0xC5110001`,
`spec/wire.md` §1) carries I/Q as int8 pairs, imaginary first; the normal
output is the 68-byte FeatureFrame (§7). One RF chain rules out
angle-of-arrival methods entirely (`RESEARCH.md` §0).

---

## 4. DSP design decisions, and why

### 4.1 Static suppression: alpha must track the sample rate

A tempting shortcut is a fixed smoothing factor such as `alpha = 0.05` at any sample rate. It is wrong.

Subtracting an EMA is a first-order highpass:

```
p[n] = a*x[n] + (1-a)*p[n-1]
r[n] = x[n] - p[n-1]    =>    H(z) = (1 - z^-1) / (1 - (1-a)z^-1)
```

whose −3 dB corner sits near `a*fs/(2*pi)`. At 100 Hz, `a = 0.05` puts
the corner at **~0.8 Hz — above the
entire 0.1–0.5 Hz breathing band it is meant to preserve**, attenuating
the target signal by roughly 12 dB at 0.2 Hz.

`corner_to_alpha(corner_hz, fs)` inverts the relation so the corner is
specified directly and defaults to 0.03 Hz, below the band.

### 4.2 Bandpass: a standard biquad instead of an ad-hoc resonator

An ad-hoc resonator with pole radius `r = 1 - bw/2` needs clamping to
stay inside the unit circle and its centre drifts with the sample rate.
An RBJ constant-peak-gain biquad in transposed direct form II costs the
same per sample, is unconditionally stable for Q > 0,
and the −3 dB points land exactly on the requested edges (verified in
`test_minus_3db_at_band_edges`).

Caveat kept honest: one biquad is 6 dB/octave, giving ~8x rejection at
3 Hz. Cascade sections if you need more.

### 4.3 Frequency estimation: spectral peak instead of zero crossings

Counting zero crossings of the filtered signal is cheap but biased by any
residual DC, degrades once a second periodicity rides on the first, and
**yields no confidence signal at all** — it always returns a number.

Instead: a Goertzel band scan plus parabolic interpolation. Goertzel
rather than FFT because only two narrow bands matter, so evaluating the
DFT at exactly those frequencies is cheaper *and* finer than binning the
whole spectrum — and it ports to an ESP32 in a few lines.

### 4.4 Confidence calibrated against noise

A natural concentration measure, `1 - mean/peak`, scores **band-limited
noise at about 0.75** — the largest of M exponentially-distributed
periodogram values sits several times above their mean by chance alone.

Fixed by comparing the observed peak-to-mean ratio against what noise
would produce. With `n_ind = bandwidth * T` independent bins, noise gives
`E[peak/mean] = ln(n_ind) + gamma` (Gumbel); a pure tone approaches
`n_ind / enbw`, where `enbw` is the window's equivalent noise bandwidth
(1.5 bins for Hann). Rescaling between those anchors gives:

| Input | Confidence |
|---|---|
| Clean sinusoid | 1.000 |
| Sinusoid + moderate noise | 0.990 |
| Sinusoid + heavy noise | 0.479 |
| Pure noise | 0.040 |

This is what makes the system able to say "nobody is breathing here".

### 4.5 Analysis windows derived, not chosen

A breathing estimate after **10 seconds** would cover under two cycles at
the 0.1 Hz bottom of the band, not long enough for the frequency to mean
anything. So the minimum is derived:
`MIN_CYCLES / f_low` = 30 s for breathing. Heart rate defaults to 60 s,
because separating the cardiac carrier from sidebands one breathing period
away needs far more than the 5.6 s the cycle rule alone would permit.

### 4.6 Heart rate: the intermodulation problem

**The most substantive finding.** CSI amplitude is the magnitude of a
static multipath vector plus a body-reflected vector, so it is a
*nonlinear* function of chest displacement. That fills the heart-rate band
with two kinds of artefact:

1. harmonics of the breathing rate, `k * f_b`
2. intermodulation sidebands, `f_h ± k * f_b`

On synthetic data with true `f_h = 1.133 Hz` and `f_b = 0.25 Hz`, the
`f_h + f_b` sideband at 1.383 Hz measured **147.4 power against the true
carrier's 123.2** — the artefact is *stronger than the signal*. Plain peak
picking returns 83 BPM for a 68 BPM heart, and a zero-crossing counter
has the same failure with no way to notice.

Notching the harmonics alone does not help, because 1.383 Hz is not a
harmonic. What does work: a true cardiac component appears as a
**symmetric triplet** (`f-f_b`, `f`, `f+f_b`), while an artefact lacks a
matching partner on both sides. Scoring candidates by
`P(f) * sqrt(P(f-f_b) * P(f+f_b))` — a geometric mean, so both sidebands
must be present — recovers 68.04 BPM against 68.0 truth.

Implemented as `dsp.sideband_peak()`. It needs the breathing frequency, so
the pipeline feeds the breathing stage's output into the heart-rate stage,
and only when that estimate is itself sound.

**It is still not reliable**, and the evaluation says so: 80% within 3 BPM
ungated, with errors that are erratic rather than systematic. Gated at
confidence ≥ 0.4 it is 97% correct over 61% of scenarios. Treat abstention
as the feature.

### 4.7 Presence: measure energy in the band, and let filters settle

Two bugs found by running it, both of which look like tuning problems:

**Broadband energy barely separates occupied from empty.** A still
person's entire signature is a narrowband 0.1–0.5 Hz modulation; sensor
and multipath noise is broadband and larger. Broadband RMS gave 0.197
empty against 0.21–0.26 occupied and the occupancy flag chattered
constantly. Bandpassing to 0.1–2.0 Hz before measuring gives **0.034 empty
against 0.15 occupied, a ~4x margin**, and the flag stops flickering.

**Calibrating through the filter transient is fatal.** A biquad near
0.1 Hz takes tens of seconds to settle. Calibrating during that window
measures the filter waking up, sets a threshold far too high, and the
detector then reports "empty" forever. `PresenceDetector` now discards a
warmup period (default five periods of `f_low` = 50 s) before accumulating
calibration statistics.

### 4.8 Decimation and throttling

Resolving a 0.5 Hz band from 20 Hz samples is 10x more arithmetic than the
information warrants. Each extractor decimates after its bandpass to 4x
the top of its band. Separately, the pipeline recomputed a full spectral
scan on *every* frame — 20 times a second over a 60 s window, where
consecutive frames differ by one sample in hundreds. Now throttled to once
per second with caching.

Together: **12x realtime, 8.5% of one core**, pure Python at 20 Hz.

---

## 5. Where to go next

1. **Buy an ESP32-S3 or C6 devkit ($6–10)** and get real CSI through
   `wifisense listen`. Everything above is synthetic; nothing is confirmed
   until real radio data runs through it.
2. **Validate breathing against a reference** — a chest strap, or counted
   breaths. This is the capability most likely to hold up.
3. **Do not trust heart rate** without a pulse-oximeter comparison across
   subjects and postures.
4. **Phase, not just amplitude.** Everything here uses amplitude, which is
   why the nonlinearity in §4.6 bites. Unwrapped phase is linear in
   displacement and has no intermodulation problem — but needs CFO/SFO
   correction first. That is the highest-value next piece of work.
5. **Port the DSP to C** for the ESP32. It was written to make this easy:
   Goertzel and a biquad are a few dozen lines each.

---

## 6. P0a: Fresnel blind spots, RespirFi selection, recording (Oct 2026)

Phase P0a of `CAMPUS_SENSING_PROMPT.md`: no hardware. Every number below
is **SYNTHETIC** and comes from `python3 evaluate.py --suite nulls`
(960 scenes, ~3 min on 20 cores) or the snippets named. Nothing was tuned
on the reported scenes: their seeds start at 10000.

### 6.1 What the RespirFi paper actually says

Read in full (arXiv:2604.20397, accepted IEEE IoT-J 2026). Three things
differ from what we expected:

- **It is amplitude-only.** Footnote 1: single-antenna phase is too
  corrupted by CFO/SFO, "therefore, RespirFi focuses exclusively on CSI
  amplitude". RESEARCH.md had this claim killed 1-2; it is now corrected
  there (§5a).
- **Its own bound limits it on ESP32.** Eq. 6: subcarrier diversity
  guarantees an escape from a blind spot only when
  `B * (d_body - d_LoS) >= 0.255 c`. With the span ESP32 actually reports
  (HT-LTF, 56 x 312.5 kHz = 17.5 MHz) that needs **4.4 m** of excess path
  (4.7 m LLTF-only). The paper's own 0.47 m is for 160 MHz.
- **Key parameters are unstated.** The similarity ternarisation threshold
  and the eq. 12 retention rule are not given. We derive the first from
  significance (2/sqrt(2BT), ~0.39 for a 32 s window) and keep scores
  >= 0.5 x best for the second. The partition uses the leading
  eigenvector of the ternary affinity (the standard relaxation of eq. 10),
  not the signed power-mean Laplacian of their ref. [46].

### 6.2 The original synthetic scene could not test this

**Expected:** the random scene has per-subcarrier gain variation, so it
exercises blind spots somewhat.

**Measured:** it draws each subcarrier's static phase independently, so
the band is a random mix of rising and falling subcarriers. Plain
averaging keeps only **8%** of the breathing signal there (|sum of signed
sensitivities| / sum of magnitudes = 0.08, printed by `--suite headroom`), yet it still reads 100%
correct because the noise is low. And every evaluate.py scenario uses
seed 1234, so the original 64-scenario table is **one room**.

**Changed:** `SyntheticConfig(scene="geometric")` builds the channel from
geometry: line of sight, image-method wall reflections (order 1 or 2,
Gamma = 0.4 with a pi flip), seeded furniture scatterers, a bistatic body
path sqrt(sigma/4pi)/(d1 d2) with sigma = 0.3 m^2, and chest motion along
the tx-body-rx bisector (path change 2 cos(beta/2) per metre). Knobs for
LoS attenuation and an extended torso (`body_points`, `body_width_m`).
The default `scene="random"` is bit-identical to before (SHA-256 pinned
in `tests/test_synthetic.py`). The 2x-rate signature of a blind spot is
asserted from first principles in `test_blind_spot_doubles_the_apparent_rate`.

### 6.3 How much room is there for any fusion method?

300 uniform positions per room type, ground-truth signed sensitivity per
subcarrier (`python3 evaluate.py --suite headroom`, under a second):

| Room | Mixed-sign scenes | Whole-band null | Averaging efficiency p10 / p50 |
|---|---|---|---|
| Clear LoS (order 1, 6 scatterers) | 3.3% | 4.0% | 0.97 / 1.00 |
| Order 2, LoS -6 dB, 12 scatterers | 2.3% | 6.0% | 0.98 / 1.00 |
| Order 2, LoS -15 dB, 12 scatterers | 4.0% | 5.3% | 0.92 / 1.00 |
| Same, Gamma = 0.6 (richest) | 10.3% | 2.0% | 0.68 / 1.00 |

At 17.5 MHz the static phase barely rotates across the band, so nearly
every subcarrier sits at the same point of the Fresnel cycle. Sign
cancellation (the case sign-flip fusion fixes) is rare. Whole-band
blind spots (which nothing in the subcarrier domain fixes) are a few
percent of positions.

**Torso vs point body.** Expected: a 0.3 m torso averages over several
Fresnel zones and removes the nulls. Measured: it does not. A torso facing
the link lies along the Fresnel ellipse tangent, where excess path changes
only to second order, and it moves as one block, so it has one net
reflection vector: the null moves, it does not disappear (minimum
sensitivity 0.00-0.12 at y = 3.03 m either way). In the suite, point and
torso bodies differ by < 1 point everywhere.

### 6.4 Bug found by the tests: BNR by magnitude

**Expected:** eq. 7 (sum of |H(F)| in band over out of band) with the
paper's floor of 2 transfers if we low-pass to ~4 Hz first.

**Measured:** a clean 0.25 Hz tone with 30% noise scores **0.66**: one
strong peak loses to 51 out-of-band noise bins. The paper's SG + LOWESS
smoothing empties those bins; block averaging does not. The first full
evaluation ran with this and the selector almost never engaged.

**Changed:** BNR uses spectral power. The floor of 2 then means "at least
two thirds of the variance is in the breathing band"; white noise scores
0.25. That evaluation run was discarded.

### 6.5 Result: fusion methods on identical data

Breathing within 0.5 BPM, 4 readings per scene (t = 75/90/105/120 s),
3 rooms x 2 bodies x 4 noise levels (std 0.6-8 against |LoS| = 40):

| Scenes | legacy | **respirfi** | respirfi no floor | PCA | oracle |
|---|---|---|---|---|---|
| Uniform placement (1920) | 95.1% | **95.7%** | 92.3% | 94.9% | 98.0% |
| Whole-band null (1068) | 64.5% | **68.1%** | 59.6% | 63.1% | 81.1% |
| Partial null (1348) | 89.2% | **91.5%** | 81.9% | 90.0% | 97.6% |
| Mixed-sign (236) | 61.0% | 67.8% | 69.9% | **80.5%** | 94.5% |

Confidently wrong (confidence >= 0.7 yet >= 1 BPM off): whole-band null
4.2% -> **2.7%**, mixed-sign 6.8% -> **1.7%**, partial 1.5% -> **0.3%**.
Paired against legacy, RespirFi is right where legacy is wrong in
13 / 30 / 38 / 16 readings (uniform / partial / null / mixed) and **wrong
where legacy is right in 0** -- it engages on 40% of selection passes and
falls back to `fuse` otherwise. On the original random scene
(`--suite legacy --fusion respirfi`) breathing error falls 0.009 -> 0.0014
BPM and gated heart rate goes from 97% correct on 61% of scenarios to
100% on 92%, but that scene exaggerates cancellation (6.2), so treat it
as an upper bound.

What it means:

- **RespirFi is safe and slightly useful at ESP32 bandwidth, not
  transformative.** It captures ~20-27% of the headroom the oracle shows.
  Its main contribution is fewer confident double-rate errors.
- **Whole-band blind spots stay at ~2/3 accuracy** for every real method,
  as eq. 6 predicts. Moving the node, or a second link, is the fix, not
  subcarrier selection. (The oracle's 81% there is not reachable: it uses
  the simulator's true signs.)
- **Without the BNR floor** the selector keeps 20% of subcarriers even
  when they are noise, which costs more SNR than the signs gain (92.3%).
- **PCA** wins where signs genuinely split but loses as often as it wins
  elsewhere (+33 / -35 paired). Not a drop-in improvement.
- **Kept behind the flag**, default off, until compared on real
  recordings. `replay --subcarrier-selection respirfi` makes that a
  one-line experiment in P1.

### 6.6 Packet loss silently biased breathing rate

**Expected:** loss only reduces SNR.

**Measured:** the pipeline counts frames, not time. Feeding survivors of
5% loss in directly made a 15 BPM breather read **15.76 BPM**, and 15%
loss read **17.80 BPM at confidence 0.98** -- confidently wrong. The
recorder review flagged it; `tests/test_gapfill.py` pins it.

**Changed:** `SequenceGapFiller` (sources/esp32_udp.py) interpolates
amplitudes across raw-CSI sequence gaps up to `--max-gap-s` (1 s) and
counts longer gaps, reboots and duplicates as discontinuities. Now 14.98
and 14.97 BPM. Used by `listen` and `replay`. Also removed SO_REUSEADDR
from `listen_udp`: with it, a second listener on the port silently took
half the datagrams.

### 6.7 Recorder / replayer

`wifisense record | stats | replay` (sources/recorder.py). Raw datagrams,
malformed included, in nanosecond pcap (LINKTYPE_RAW, synthesized
IPv4/UDP), readable by Wireshark; reads `tcpdump -w` captures too
(Ethernet+VLAN, SLL, SLL2, null). Flushes every second (a SIGKILL lost
< 1 s in testing). `stats` reports per-node rate, sequence loss, reboots,
duplicates, reordering and inter-arrival jitter: the per-link numbers
constraint 4 of the campus plan requires. Replay through the pipeline is
deterministic (tested). Not verified against real Wireshark/tcpdump
(neither installed here); foreign framings are tested with hand-built
captures.

### 6.8 Not done / open

- RespirFi's breathing-phase identification (paper IV-D, inhale vs
  exhale) is not ported; rate estimation does not need it.
- The geometric scene is 2-D, has no LoS shadowing by the body, no AGC,
  no CFO, and a rigid chest. Real ESP32 amplitude noise is unmeasured;
  P1 sets the noise level the table should be read at.
- `listen` feeds every node_id into one pipeline. Fine for P1's single
  RX, wrong for a mesh.

---

## 7. Saans S0–S1: spec, golden vectors, Rust port (Oct 2026)

Plan: `SAANS_PLAN.md`. Decisions: `docs/adr/`. Wire format: `spec/wire.md`.

### 7.1 The feature frame is an executable spec

`wifisense/features.py` is the reference FeatureFrame v1 extractor (68
bytes, 2 Hz per link). It wraps the reference `SensingPipeline` and adds
gap filling, a 1–4 Hz gross-motion RMS, a 0.1–0.5 Hz breathing-band RMS,
link statistics, BNR link quality and the consent gate. To make it
possible without re-running stages, `SensingPipeline` now exposes
`last_residuals`, `last_fused` and `last_vitals_input` (additive), and
`preprocess.subcarrier_bnr` was split out of `respirfi_select` (no
behaviour change; RespirFi tests unchanged).

`spec/golden/generate.py` writes 23 vector sets (3.2 MB) covering every
stage, the codec, two end-to-end consent settings and the noise-free
geometric physics for the Rust simulator.

### 7.2 Rust parity (S1 gate)

`rust/crates/saans-dsp`: a dependency-free port of the whole reference.
All 12 golden tests pass at 1e-9 relative. The end-to-end FeatureFrames
are **byte-identical to Python's: 672/672 frames** across both consent
settings. What it took, beyond a straight port:

- CPython ≥ 3.12 `sum()` over floats is Neumaier-compensated; the port
  uses the same compensation everywhere the reference calls `sum`.
- Python `round()` is half-to-even; `max(key=...)` takes the first
  maximum; `abs(complex)` is `hypot`. Each is reproduced explicitly.
- The reference's Euler–Mascheroni constant is truncated (0.5772156649);
  the port keeps the truncation on purpose (a lint is allowed with a
  comment), because the full constant moves confidences by ~1e-12.

The one failure on first compile was in the test, not the port: it cast
the u32 `calib_age_s` (0xFFFFFFFF = not calibrated) through f32.

### 7.3 Speed, measured, on the same input

32 subcarriers, 20 Hz, 3267 frames (`cargo bench -p saans-dsp`,
`python3 spec/golden/bench_python.py`), one core:

| | frames/s | × real time (one link) |
|---|---|---|
| Python `SensingPipeline.process` | 191 | 9.6 |
| Rust, same algorithm (estimate every frame) | 2,851 | 143 |
| Python `FeatureExtractor` | 175 | 8.8 |
| Rust `FeatureExtractor` (estimate only when a frame is due) | 25,283 | 1,264 |

So the language is worth ~15×; computing the spectral estimates only when
a FeatureFrame is due (twice a second instead of twenty) is worth another
~9×, with identical output because the estimates are pure functions of
the extractor state. One gateway core keeps up with ~1,200 links.

---

## 8. Saans S3: health detectors meet a simulator (Oct 2026)

`saans eval` (rust/crates/saans-cli) runs scripted timelines from
`saans-sim` (whose physics matches the Python geometric scene to 1e-14)
through the production path: per-link `FeatureExtractor` (the golden-
verified DSP), `SpaceFusion`, `SpaceMonitor`. Every number is SYNTHETIC.
Thresholds were tuned on seeds 0–1 (`saans eval --quick`); the table
below is from held-out seeds 2–7 (`saans eval`, 252 runs, 187 simulated
hours, 171 s on 20 cores).

### 8.1 Three bugs the evaluation found (none visible to unit tests)

**Presence threshold too low.** *Expected:* mean + 4σ of 30 s of
empty-room energy is a safe threshold. *Measured:* an empty room read
occupied 36% of the time; the presence index sat at a median of 0.89 of
the threshold. The energy is an 8 s sliding RMS, so 30 s holds only a
few independent windows and the sample σ is badly underestimated.
*Changed (reference first, ADR-002):* a derived floor, threshold ≥ mean ×
(1 + k/(2√(B·T))), B = π/2 × bandwidth (biquad ENBW), T = 8 s → 1.41×.
Empty-room false occupancy fell to 0.6–1.2%.

**Gross-motion band leaked breathing.** *Expected:* a 1–4 Hz bandpass
sees limbs, not breathing. *Measured:* a still breather's motion index
was 1.33 vs 0.88 in the empty room: one biquad's 6 dB/octave skirt passes
~19% of the 0.25 Hz breathing component. *Changed:* two cascaded
sections (FeatureFrame `gross_motion`, spec/wire.md). Still ≈ empty now
(0.79 vs 0.75).

**One noisy sample reset every stillness clock.** The heartbeat (~1.1 Hz)
lives in the gross-motion band, so a still person's index wanders to
~2.2. With STILL_INDEX 1.5 the 30-minute inactivity alert fired in 2 of
12 lone-worker runs. A sweep on the development traces (`saans trace`)
chose STILL_INDEX 2.5 with 3 confirming samples: false stillness resets
60/h → 0.15/h while still catching ~82% of fidget bursts.

All three changes went into the Python reference, regenerated the golden
vectors, and were caught by them in both the Rust and the C++ ports
before either was edited by hand.

### 8.2 Results (held out, SYNTHETIC)

| | 1 link | 2 links | 3 links |
|---|---|---|---|
| Occupancy accuracy | 98.2% | 99.5% | 98.9% |
| Still person read as empty | 2.1% | 0.4% | 1.3% |
| Empty room read as occupied | 1.0% | 1.2% | 0.6% |
| Breathing reported (still breather) | 93.6% | 100% | 100% |
| ... within 1 BPM | 93.9% | 99.9% | 100% |
| Breathing pauses caught (248 true, 10–40 s) | 81.5% | 89.9% | 88.7% |
| ... precision | 98.5% | 98.7% | 98.7% |

- No breathing pause was reported in 70.6 h of timelines without one.
  Pause duration is late by ~9–10 s (median): band filter + 2 s RMS lag.
- Lone-worker inactivity (30 min) fired in 35/36 runs at a median of
  2500 s for stillness starting at ~700 s. It also fires for every
  40-minute reader: correct by its definition, which is why it is advisory.
- Fast breathing (26–30 BPM) raised RateHigh in 31/36 runs; rate alarms
  also fired in 2/36 reader and 3/36 sit-down runs.
- **Possible collapse is not usable**: 100% of falls, but also 100% of
  sit-downs, 92% of readers walking in and 97% of fidgeting nights.
  Amplitude-only 2-D WiFi cannot see height, so "fell" and "sat down"
  look the same. It ships OFF (ADR-005 config) and should stay off.
  With it off, Check-severity alerts are 0/h everywhere except nights
  (2.9/h: breathing pauses, which is what they are for).
- Link diversity (ADR-006) is confirmed in simulation: a second link
  takes breathing coverage from 94% to 100% and halves missed pauses.

### 8.3 Known limits

- The simulator's body never blocks the line of sight, never sways, and
  its CSI phase is clean; real ESP32 noise is unmeasured (S5).
- A still person who stops breathing can still drop out of presence after
  ~18 s on a single link (the presence energy is mostly breathing).
  Pauses that end that way are reported as Check with that caveat.
- `best_bnr` is now optional (0 = not computed): its row store is half a
  node's per-link memory and nothing reads it but system health. With it
  off, every other field is bit-identical.

## 9. The ESP32 receiver layer, and what it costs (Oct 2026)

`saans-sim` produced a clean complex channel. A real node does not: it
applies automatic gain control, a random per-packet phase, a timing
offset that tilts phase across subcarriers, a static per-device ripple,
and quantizes to int8. `src/impair.rs` models that; `src/wire.rs` puts
the result in the raw-CSI datagram the firmware already emits, so a
simulated node and a bench node reach the gateway through one decoder.

It is a stage **after** the clean channel and **off by default**. The
clean channel is the contract with the Python and C++ implementations;
`tests/parity.rs` still reproduces the golden vectors at a max relative
error of 9.3e-15 against a 1e-9 tolerance.

**Its constants are guesses.** Nothing here has touched a radio. The
numbers come from published descriptions of ESP32 CSI, not from a
measurement taken in this project.

### What it costs, measured

`cargo run --release -p saans-sim --example impairment_cost`. Four arms,
same scenario and seed, 5 room seeds, 1 link x 56 subcarriers at 20 Hz.
`no-agc` is the full model with `agc_step_db = 0` — one term off, no
constant changed. The clean channel is bit-identical across arms.

| seated, 14 BPM | clean | int8 | no-agc | esp32 |
|---|---|---|---|---|
| breathing BNR, best subcarrier, dB | -0.49 | -0.45 | -0.72 | **-2.11** |
| motion energy, empty room | 0.129 | 0.129 | 0.152 | **1.018** |
| motion energy, person seated | 0.351 | 0.352 | 0.356 | 0.882 |
| calibrated presence threshold | 0.185 | 0.185 | 0.213 | **2.251** |
| frames reporting occupied, % | 100 | 100 | 100 | **0** |
| breathing BPM (truth 14.0) | 14.00 | 14.00 | 14.00 | **none** |

Three findings, in order of how much they are believed.

1. **int8 quantization costs nothing measurable.** Every metric in the
   table is within 0.04 dB of the clean arm, and the presence figures
   match it to three decimals. At a mean of ~38 counts the
   quantizer's error sits far below the scenario's own channel noise.
   Going in, this was the term expected to hurt. It does not.
2. **Static ripple, per-packet phase and receiver noise cost little.**
   Amplitude CV 0.054 to 0.061; breathing BNR 0.23 dB worse; breathing
   still recovered at 14.00 BPM with 100% of frames.
3. **The AGC is the whole effect, and it does not break what was
   expected.** Breathing BNR loses 1.4 dB against the no-AGC arm, 1.6 dB
   against clean — the breathing signal is still there. What breaks is the *presence* detector's ambient
   calibration: gain steps are impulses in the 0.1–2 Hz presence band,
   so the empty-room threshold inflates 12x, and the empty room ends up
   reporting *more* motion energy than the seated person. The contrast
   does not shrink, it inverts. Vitals are then suppressed because the
   pipeline believes the room is empty. **A lost BPM here is a lost
   person, not a lost signal** — which is only visible because presence
   is reported beside it.

### What that result actually rests on

One uncalibrated constant, `agc_dither_prob` — how often a converged
loop moves anyway. Sweeping it, nothing else changed:

| `agc_dither_prob` | gain moves, % | empty | seated | threshold | occupied, % |
|---|---|---|---|---|---|
| 0.000 | 0.02 | 0.148 | 0.245 | 0.210 | 100.0 |
| 0.002 | 0.18 | 0.185 | 0.329 | 0.959 | 0.0 |
| 0.005 | 0.57 | 0.465 | 0.433 | 1.203 | 0.0 |
| 0.020 (committed) | 2.43 | 1.018 | 0.882 | 2.251 | 0.0 |

With no dither the model costs the detectors essentially nothing. So the
honest claim is **not** "an ESP32 breaks presence". It is:

> Presence survives a gain that moves about once every four minutes
> (the 0.000 row: 0.02% of packets at 20 Hz). It fails completely at
> once every twenty-eight seconds (0.002: 0.18%). The sweep brackets the
> breakpoint between those two and does not locate it any more finely.

An earlier draft of this section put the breakpoint at "once every nine
seconds". That is the 0.005 row, which already reports 0% occupied — it
is inside the failing region, not at its edge. Narrowing the bracket is
a matter of adding rows between 0.000 and 0.002 to the sweep, which is
a one-line change to the example and still answers nothing a bench
recording would not answer better.

That is a bench question with a cheap answer: log `agc_gain` on an idle
node in an empty room for ten minutes and count the moves. It is the
single most valuable number in the first recording, and until someone
takes it the table above ranks nothing.

Deliberately **not** acted on: a per-packet common-mode gain is
something the DSP could normalise away, and the static suppressor
cannot. Changing the detectors on the strength of an uncalibrated model
is the opposite of what this layer is for.

### Known gaps in the layer itself

- `first_word_invalid` is not modelled. Its real effect is a two-bin
  index shift on affected frames, not garbage, so a consumer of
  simulated raw CSI never sees the one thing the flag does.
- Which 56 bins the firmware actually hands over is unresolved. The
  simulator's 56 are the idealised occupied set; by ESP-IDF's
  documentation the firmware's first 56 pairs may be a different 56
  entirely. `BinLayout` can produce either once a bench session says
  which. A subcarrier index may not mean the same thing on both sides.
- No cross-subcarrier smoothing or LTF merge, deliberately: our capture
  path disables both.
