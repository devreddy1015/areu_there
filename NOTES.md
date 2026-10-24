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

   *Superseded, 2026-10-24.* Left as written because it was the right
   reading at the time, but do not act on it. `docs/CAPABILITY_LIMITS.md`
   §4 retires it: RespirFi's own footnote is that single-antenna CSI
   phase is unreliable under CFO/SFO, which is why that paper is
   amplitude-only, and this hardware is single-antenna. Phase comes back
   only as a research item, once real recordings show where the
   amplitude ceiling actually is.
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
- The geometric scene is 2-D (a box with floor, ceiling and heights is
  now an option, §13), has no LoS shadowing by the body, no AGC,
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

*2026-10-24:* the firmware now logs it: the RX's `gain:` telemetry line,
and raw header bytes 18-19 for `wifisense stats`. It reads the offset
Espressif's own gain-control component uses, undocumented, so the bench
confirms it first by moving the AGC on purpose (`docs/PROTOCOL_S5.md`
§1.1).

Deliberately **not** acted on: a per-packet common-mode gain is
something the DSP could normalise away, and the static suppressor
cannot. Changing the detectors on the strength of an uncalibrated model
is the opposite of what this layer is for.

### Correcting it: the only fix that works needs the radio's own gain

Measured 2026-10-24, `cargo run --release -p saans-sim --example
impairment_cost -- --corrections`, the seated scenario above, 5 seeds.
**SYNTHETIC.** Each candidate is applied to the frames a consumer sees,
before the unchanged production `FeatureExtractor`; none is in the DSP.

| correction | esp32: occupied | esp32: within 2 BPM | clean arm: occupied |
|---|---|---|---|
| none | 0% | 0% | 100% |
| common-mode (per-frame median removed) | 0% | 0% | **0%** |
| step detector, t ≥ 5 | 0% | 0% | 100% |
| **oracle: the model's true gain undone** | **100%** | **100%** | 100% |

What each row says:

- **Common-mode removal is ruled out.** It kills the clean arm too. At
  ESP32 bandwidth the static phase barely rotates across the band (§6.3),
  so nearly all of a breather's signal is common to every subcarrier;
  removing what is common removes the person.
- **The oracle is the answer to "is the AGC the whole problem".** Undo
  the true gain exactly and presence and breathing recover completely,
  at both `agc_dither_prob` 0.020 and 0.002, with empty-room motion
  energy back to 0.145 against no-agc's 0.152. In this model the gain
  steps are the entire failure.
- **No estimate of the steps from the CSI gets there.** The detector
  (a common frame-to-frame jump, many standard errors clear of the
  per-subcarrier scatter) measures each step fairly well -- +0.39 to
  +0.53 dB against a true +0.50 -- but its standard error at a move is
  0.10–0.15 dB, not the ~0.03 I estimated from quantization alone, since
  many bins sit at low counts in frequency-selective fades. Sweeping the
  threshold instead of refitting it:

  | t ≥ | moves caught | detections not at a move | on the clean stream | esp32 occupied |
  |---|---|---|---|---|
  | 2 | 689 / 702 | 1298 | 942 | 0% |
  | 3 | 631 / 702 | 147 | 171 | 0% |
  | 4 | 509 / 702 | 0 | 29 | 0% |
  | 5 | 355 / 702 | 0 | 7 | 0% |

  Even catching 98% of moves leaves presence at 0%. Each correction uses
  the *measured* step, wrong by ~0.1 dB, and seven hundred residual
  0.1 dB steps are seven hundred small transients. A higher threshold
  trades them for missed moves and arrives at the same place.

**So the fix is not a DSP algorithm; it is reading the gain register.**
The oracle row is exactly what a node would get if the firmware logged
the true AGC gain per packet and divided it out. That moves
`docs/PROTOCOL_S5.md` §1.1 route 1 -- read the gain from
`wifi_pkt_rx_ctrl_t`, if the S3 exposes it -- from "a way to count the
moves" to "the fix, if the moves turn out to matter". *2026-10-24:* the
headers were read. There is no documented field, but the gain sits at
an offset Espressif's own tooling reads, and the firmware now logs it
per frame. Whether it is the true gain is confirmed on the bench; if it
is, dividing it out is the oracle row above, built.

One caveat about the oracle's clean result: in this model the gain step
is the only impairment that varies packet to packet in amplitude, so
undoing it returns to the no-agc arm by construction. A real radio may
have others.

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

---

## 10. The octave rule: a doubled breathing rate, fixed in fusion (Oct 2026)

Everything here is **SYNTHETIC**.

### Why the obvious fix could not work

At a Fresnel null a link's first-order breathing sensitivity is zero, so
its amplitude follows the *square* of chest displacement. A breather at
`f` reads as `2f`, and because `sin²` is a clean tone it reads that way
at high confidence. The full nulls sweep puts it at 16.9% of readings at
whole-band nulls, 4.2% of them at confidence ≥ 0.7 (`docs/RESULTS.md`).

The plan was a single-link guard: lower the confidence when there is a
rival peak at `f/2`. Reading the simulator first killed it. The chest is
driven by a pure sinusoid (`synthetic.py:332`), and `sin²(2πft)` has no
energy at `f` at all. On one link, at a true null, a doubled reading is
exactly what a person breathing twice as fast produces. A guard tuned to
pass here would have been tuned to nothing. Real breathing is asymmetric
-- inhale and exhale differ -- so a real null may leave energy at `f`;
that is a question for the bench, not a parameter to fit.

### The rule

The physics is asymmetric between links even though one link cannot see
it: a null turns `f` into `2f`, and nothing turns `f` into `f/2`. So in
`SpaceFusion::sample`, a link whose rate is within 1 BPM of twice
another link's is set aside before the most confident link is chosen,
and hysteresis does not keep it. The lowest rate in a chain of octaves
always survives, so the candidates never empty. One link alone is left
alone. The tolerance is the definition `evaluate.py` already used for
"2x rate"; nothing was tuned.

### Measured

`saans octave`: two links (one TX, two RX, the layout `docs/HARDWARE.md`
says to buy), a seated breather at 8–14 BPM so that twice the rate stays
inside the band, 300 s per case after an empty lead-in so presence
calibrates on an empty room, through the production path. Cases are
classed from ground-truth sensitivity *at the seat*. 200 per class,
identical scenarios before and after:

| case | within 1 BPM | 2x rate | 2x at conf ≥ 0.7 | breathing from the null link |
|---|---|---|---|---|
| one link null, one clear | 93.1% → **96.6%** | 6.9% → **3.4%** | 4.4% → **1.8%** | 10.7% → 8.1% |
| both links clear | 98.9% → **99.8%** | 1.1% → **0.2%** | 1.1% → **0.2%** | — |
| both links null | 74.8% → **83.5%** | 25.2% → **16.5%** | 19.1% → **10.9%** | — |

Coverage is 100% / 100% / 96.7% before and after: the rule never
removes the last candidate. In every class the gain in readings within
1 BPM is exactly the fall in doubled ones (+3.5 / −3.5, +0.9 / −0.9,
+8.7 / −8.7), so no correct reading was broken to fix a doubled one.

### Three things worth knowing about that number

1. **A first run at 40 cases per class said the one-null class was only
   2.5% doubled.** At 200 it is 6.9%. The small sample was one or two
   unlucky runs, and a decision taken on it would have concluded the
   rule was barely worth having.
2. **Two mistakes in the measurement were caught before it ran.** The
   first draft seated the person at t = 0, which would have calibrated
   presence on the breather; and it classified cases with
   `Simulator::sensitivity`, which uses the body's position *now* -- an
   absent body parked somewhere -- rather than at the seat. Either would
   have produced a confident, wrong table.
3. **It does not touch the case where both links double together.**
   Both nulls reading `2f` agree with each other, and agreement is not
   evidence of anything. That residual 16.5% is what a third link, or a
   moved node, is for.

### What it did to the held-out S3 evaluation

`saans eval`, six held-out seeds per cell, rerun with the rule. Exactly
three lines of the table moved, and nothing got worse:

| row | before | after |
|---|---|---|
| breathing within 1 BPM, 2 links | 99.9% | 100.0% |
| reader: rate high/low alerts fired | 2/36 runs | **1/36** |
| clear-LoS breathing within 1 BPM | 96.1% | 96.2% |

The middle row is the one that matters most. A doubled breathing rate in
someone sitting still *is* a tachypnea false alarm, so this rule removes
false alerts, not only BPM error. True tachypnea detection is unchanged
at 31/36, as are occupancy, pause detection and every other alert row:
the rule did not suppress genuinely fast breathing.

---

## 11. One, two or three links, on layouts the firmware can build (Oct 2026)

Everything here is **SYNTHETIC**. `saans links` (`docs/RESULTS.md`).

### Why `saans eval`'s per-layout rows could not answer this

ADR-006 owes a measurement of the blind-spot rate with 1, 2 and 3 links.
`saans eval` already prints rows for each, and they were the source of
CAPABILITY_LIMITS §4's "still-person false-empty goes back up from 0.4%
to 1.3%" at three links. Two things stop those rows answering the
question:

- **Its 3-link layout moves the nodes.** One and two links use nodes at
  (1.5, 2.5), (4.5, 2.5), (3.0, 4.6); three links uses (1.0, 1.0),
  (5.0, 1.0), (3.0, 4.5). Link count and geometry change together.
- **It connects every pair of three nodes**, which needs two
  transmitters. The firmware has one: `rx_role.cpp:76` hard-codes
  `tx_id = 0xFF`. On hardware, a third link means a fourth board.

So `saans links` keeps one transmitter and adds receivers -- N1, then
N2, then N3 at (3.0, 0.4), N2's mirror image across the TX–N1 line --
and runs every row on the same seats and seeds, with `saans eval`'s
rooms, noise, loss and body.

### Measured

Ground truth only, 10,000 seats: the share where *every* link is weak or
blind.

| layout | all blind (best < 0.2) | all weak (best < 0.5) |
|---|---|---|
| 1 link | 4.5% | 23.4% |
| 2 links | **0.2%** | 4.6% |
| 3 links | **0.1%** | 1.6% |

Production path, 1,200 seats per row: 120 s of empty room after
calibration, then a still breather for 300 s.

| configuration | still false-empty | seats mostly missed | empty false-occupied | breathing coverage |
|---|---|---|---|---|
| 1 link | 2.8% | **44**/1200 | 0.2% | 95.6% |
| 2 links | 0.0% | **0**/1200 | 0.5% | 99.8% |
| 3 links, quorum 2 (default) | 0.0% | 2/1200 | **0.0%** | 99.9% |
| 3 links, quorum 1 | 0.0% | 0/1200 | 0.7% | 100.0% |
| eval's 3-link geometry, quorum 2 | 0.2% | 6/1200 | 0.0% | 99.8% |
| eval's 3-link geometry, quorum 1 | 0.0% | 0/1200 | 0.5% | 100.0% |

"Mostly missed" is a seat at which a still person was reported empty
more than 10% of the time: the person, not the sample.

### What it says

1. **The second link is the one that matters.** Seats where a still
   person is mostly missed fall from 44 to 0, and the share of seats
   blind to every link from 4.5% to 0.2%. This is ADR-006's decision,
   measured on the topology it will be built in.
2. **The third link finds no extra people.** At the default quorum it
   misses 2 seats where two links missed none -- within noise -- and
   what it buys instead is false occupancy from noise falling from 0.5%
   to 0.0%, because a majority of three outvotes one noisy link. So a
   fourth board, as a third receiver, is a false-alarm measure in a
   room this size, not a detection one. `docs/HARDWARE.md`'s advice to
   buy it as a spare stands, now on evidence.
3. **The quorum alone can cause misses, but only with the geometry.** In
   eval's 3-link geometry, quorum 2 misses 6 of 1,200 seats and quorum 1
   none: same seats, so the quorum is the cause there. In the buildable
   geometry quorum 2 costs nothing measurable. CAPABILITY_LIMITS' 0.4% →
   1.3% comes from eval's whole scenario mix (nights, lying down) and
   also changes the geometry; this separates part of it, not all of it.

### Two things that make these numbers easy to misread

- **A first run at 300 seats per row said the single link missed 0.5%
  of the time, with 3 seats mostly missed.** At 1,200 it is 2.8% and
  44. The first 150 seeds per room happened to be kind. Rare failures
  live in a few seats, so a few hundred seats is not enough to see them;
  this is the second time in two days a small sample would have put a
  wrong number in a document (§10 has the first).
- **"Empty false-occupied" here is noise only.** The majority quorum
  exists so that one link picking up a person in the *next* room cannot
  mark this one occupied, and the simulator has no next room. The
  quorum-1 row looks nearly free; in a building it is not, and its real
  cost is a bench measurement.

---

## 12. How far the breathing confidence is from a probability (Oct 2026)

Everything here is **SYNTHETIC**. `evaluate.py --suite nulls`, the
reliability tables in `docs/RESULTS.md`.

`CAPABILITY_LIMITS.md` §5 already says the confidence "is a gate, not a
calibrated probability". Nobody had measured how far. The breathing
confidence is the share of in-band spectral energy at the peak; nothing
about that construction makes 0.7 mean "right 70% of the time". A
reliability table bins readings by stated confidence and compares each
bin's mean confidence with the share of its readings that were right --
here, within 1 BPM, the S5 gate's tolerance. The expected calibration
error (ECE) is the reading-weighted mean gap.

Uniform placement is the honest population. The suite over-samples
nulls on purpose, so those are shown apart rather than mixed in.

| | uniform placement | whole-band nulls |
|---|---|---|
| readings | 1920 | 1068 |
| expected calibration error | **0.064** | **0.154** |
| at confidence 0.9–1.0 | 1534 readings, stated 0.988, right 100% | 290 readings, stated 0.980, right 94.1% |
| at confidence 0.3–0.5 | stated ~0.40, right 88–96% | stated ~0.40, right 56–60% |

### What it says

1. **It is wrong in opposite directions in different rooms.** In
   ordinary placement it is *under*confident below 0.9: a reading at
   0.4 is right about nine times in ten. At whole-band nulls the top
   bins are *over*confident, and those are the doubled readings of §10
   sitting at high confidence.
2. **So no recalibration can fix it.** Any monotone map from confidence
   to probability sees 0.98 in both populations -- right 100% of the time
   in one and 94% in the other -- and cannot separate them. The missing
   information is *whether the link is at a null*, which is what the
   cross-link octave rule (§10) supplies and a single link's confidence
   does not. Fitting such a map to simulator output would in any case be
   fitting the simulator; calibration waits for the S5 recordings.
3. **The 0.4 gate is conservative in ordinary rooms.** It discards
   readings that are right most of the time. For an advisory system that
   abstains rather than guesses, that is a defensible trade; it is now a
   measured one rather than an assumed one.

Measurement only: nothing in the DSP changed.

---

## 13. Floor, ceiling and heights: the simulator as a box (Oct 2026)

Everything here is **SYNTHETIC**. `evaluate.py --suite height` and
`evaluate.py --suite nulls --room-height 2.7`, both in `docs/RESULTS.md`.

The geometric scene was a horizontal slice at chest height: four walls,
and the nodes, the chest and the furniture all at one height. A real room
has a floor and a ceiling at least as reflective as its walls, nodes on a
shelf and a chest somewhere else. `SyntheticConfig(room_height_m=...)`
makes the room a box:

- the image method runs over all six faces, each image once (below);
- tx, rx, chest and furniture get heights (`tx_z_m`, `rx_z_m`,
  `body_z_m`, `scatterer_z_m`), and every distance is 3-D;
- the chest breathes horizontally, along the horizontal part of the
  tx-chest-rx bisector. Level with the link this is the old
  `2 cos(beta/2)`; a chest directly below the link's midpoint has no
  first-order path change at all;
- the floor and ceiling default to concrete, 0.395 from the ITU-R P.2040
  row in `scenes/materials.json`.

Without `room_height_m` the 2-D model is unchanged to the last bit: the
regenerated golden vectors did not move at all. The box has its own
golden set, `sim_geometric_3d`.

### What the box assumes

- **A rectangle.** `hall-l`'s L-shaped polygon is not modelled, as it
  was not in 2-D.
- **One |Gamma| per surface at every angle**, as the walls always had.
  For the floor this is the weakest assumption in the model. On a 3 m
  link at 1.2 m the floor bounce arrives 51° from normal, where concrete
  reflects 0.55 of the field polarised along the floor and 0.21 of the
  other; on a 5 m link, 0.66 and 0.05 (half-space Fresnel equations,
  same permittivity). The 0.395 sits between, so the box is right on
  average and wrong for any one antenna orientation.
- **The torso** is still points spread horizontally, now at chest height,
  each moving along its own bisector as in 2-D. It has no vertical
  extent and casts no shadow.
- **Furniture heights**, uniform 0.3-1.8 m, are a guess. They are drawn
  after every 2-D draw, so one seed gives the same furniture footprint
  in both models and the comparisons below are paired.

### A bug the box found in the 2-D model

The 2-D image search mirrors breadth-first, refusing only to mirror
straight back. At order 2 it reaches each of the four corner images
(one bounce off an x wall, one off a y wall) twice, x-then-y and
y-then-x, and adds both. That gives 12 order-2 paths where there are 8
images. A corner reflection is one specular path, so in 2-D every corner
path is twice as strong as it should be. The box enumerates the image
lattice instead, and `test_order_two_counts_each_corner_once` pins the
difference to exactly one copy of each corner path.

Every scene with `reflection_order >= 2` has the bug:

- the attenuated-LoS and rich-NLoS rooms in `nulls`;
- three of the four `headroom` rooms;
- the `sim_geometric` golden set;
- in Rust, which copies the search, the order-2 rooms in
  `saans-sim/src/scenarios.rs` and `saans eval`.

**It is not fixed here.** Fixing it moves every published number those
rooms produce, and that is a separate decision. The `height` table's
second row isolates what it does. In the order-2 room:

- mixed-sign positions go from 2.6% to 4.8%;
- the 10th-percentile plain-average efficiency goes from 1.00 to 0.91.

In the order-1 room nothing changes, as it should not.

### Measured: ground truth

Paired: 1000 positions, the same furniture in each row
(`--suite height`).

| clear-LoS room (order 1) | mixed-sign | whole-band null | avg-eff p10 | K, median |
|---|---|---|---|---|
| 2-D slice | 4.4% | 4.4% | 0.94 | 5.3 dB |
| box, heights, floor and ceiling absorbing | 4.5% | 3.6% | 0.96 | 5.8 dB |
| box, concrete floor and ceiling | **13.0%** | **1.8%** | **0.67** | **3.6 dB** |

Of the 2-D slice's 44 whole-band blind spots, **none** is still one in
the box, which has 18 of its own. Chest height matters less than the
floor does. In the full box, a chest at 0.6 m (lying down) gives 16.6%
mixed-sign and 1.4% whole-band nulls; at 1.0 m and at 1.3 m, 13.0% and
1.8%.

### Measured: breathing, end to end

`--suite nulls`, the same 960 case seeds. The box run kept 953: seven
cells found no whole-band null in 5000 tries.

| within 0.5 BPM | 2-D slice | box, 2.7 m |
|---|---|---|
| uniform placement, default fusion | 95.1% | **95.8%** |
| uniform placement, RespirFi | 95.7% | **96.7%** |
| uniform placement, PCA | 94.9% | **96.9%** |
| whole-band nulls, default / RespirFi | 64.5% / 68.1% | **60.2% / 62.0%** |
| mixed-sign scenes, count | 59 | **98** |
| mixed-sign scenes, default / RespirFi / PCA | 61.0% / 67.8% / 80.5% | 65.3% / 73.0% / **92.9%** |
| clear-LoS room, default / PCA | 97.8% / 97.7% | 95.3% / **98.6%** |

### What it says

1. **The floor and ceiling are not a detail.** In the clear-LoS room
   they triple the positions where subcarriers disagree in sign, take
   1.7 dB off the K-factor, and move every blind spot. A whole-band blind
   spot is rarer in the box, but it still exists.
2. **The README's headline survives.** Default fusion reads 95.8%
   against 95.1%. RespirFi is still never worse than the default on a
   reading: paired +16/-0 at uniform placement and +18/-0 at nulls. At
   whole-band nulls every method except the oracle reads lower.
3. **The ranking of fusion methods does not survive.** In 2-D, PCA was
   level with the default (94.9%, paired +33/-35). In the box it is the
   best real method (96.9%, +54/-33), because the box makes the
   mixed-sign case common, and there PCA reads 92.9% to RespirFi's
   73.0%. Which fusion wins depends on how much multipath the room has,
   and here that is a property of the model. The choice stays with the
   S5 recordings, as RespirFi's off-by-default status already says.
   Nothing in the pipeline changes on the strength of this.

### The Rust port

`saans-sim` has the box as `Room::vertical`, with heights on `Node`
and on explicit `Scatterer`s.

- **Parity.** It reproduces `sim_geometric_3d` to 4e-14 relative in
  amplitude and 9e-15 in phase and sensitivity, and the per-point chest
  gains to 2e-16 (`tests/parity.rs`).
- **The slice is untouched.** A digest of every bit of five scripted
  runs (all three rooms, one to three links, the arm, sensitivity and
  coverage maps; 257k frames) is identical before and after the port.
- **Scenes.** `Scene::to_sim_box_room` runs a scene in a box at its own
  height, and `to_sim_nodes` now keeps each node's.

What the port adds beyond the reference, because the Rust body moves
and has an arm:

- **The arm** sits at chest height.
- **A fall puts the chest on the floor.** Upright, standing or seated
  alike, the chest is at `Vertical::chest_z_m`. A `Fall` lowers it to
  0.2 m (`FLOOR_CHEST_Z_M`) on the drop's own smoothstep. There the
  person lies on their back and breathes vertically, the reference's
  `chest_axis="vertical"` (golden set `sim_geometric_3d_lying`, matched
  to 1.5e-14). They stay down until they next walk, and getting up is
  instant.
  - Nothing else changes posture. A sit-down stays at one height:
    standing and seated heights are not separated.
  - The slice ignores all of it.
  - Digests on `main` and after the change are identical for every run
    without a fall, and for every collapse run up to the fall.

Every built-in scenario, `links`, `octave` and the gateway's
`--simulate` are still the slice; `saans eval --room-height` is the one
way in, below.

### Measured: the S3 detectors in a box

`saans eval --room-height 2.7`: the same 252 held-out runs and seeds as
`saans eval`, every room a box, with a fall putting the chest on the
floor. Both tables are in `docs/RESULTS.md`. The slice's reproduced
exactly on this code, so the two columns differ only by the box.

| held out, 252 runs | slice | box, 2.7 m |
|---|---|---|
| occupancy accuracy, 1 / 2 / 3 links | 98.2 / 99.5 / 98.9% | 98.6 / 99.7 / 98.7% |
| breathing within 1 BPM, 1 / 2 / 3 links | 93.9 / 100.0 / 100.0% | 91.7 / **96.5** / 100.0% |
| breathing within 1 BPM, clear-LoS room | 96.2% | **92.3%** |
| pause sensitivity, 1 / 2 / 3 links | 81.5 / 89.9 / 88.7% | 87.9 / 87.9 / 92.7% |
| pauses reported on timelines with none | 0 in 70.6 h | **2** in 70.6 h |
| rate alerts in runs whose rate never changes: reader / night / sit-down | 1 / 0 / 3 | 0 / **3** / 0 |
| Check alerts per hour on a night, as shipped | 2.88 | **3.34** |
| falls flagged as a possible collapse (detection on to measure it; it ships off) | 36/36 | **33/36** |

What it says:

1. **Occupancy holds.** It is within half a point either way on every
   layout.
2. **Breathing rate loses accuracy where the slice was easiest.** In
   the clear-LoS room it drops from 96.2% to 92.3% within 1 BPM, and the
   2-link layout from 100% to 96.5%. This is consistent with the
   ground truth above: in that room the box triples the positions where
   subcarriers disagree in sign.
3. **False alarms move to the night.**
   - Runs with a rate alert and no rate change go from 4 to 3, but all
     three are now normal nights, where a false alert wakes somebody.
   - Pauses on pause-free timelines go from 0 to 2 in 70.6 h.
   - The Check rate on a night, as shipped, rises from 2.88 to 3.34 an
     hour.
4. **Pause detection improves.** It improves most in the rich-NLoS room
   (82.8% to 93.0%).
5. **The collapse detector misses three falls in a box, all on one
   seed, for two different reasons.** It needs a motion burst (index
   >= 8) followed within 5 s by 30 s of stillness with presence held.
   With the chest staying at 1.0 m the box missed none; dropping it to
   the floor costs those three.
   - **A fall too weak to register.** In the clear-LoS room on one and
     on two links, the fall's peak motion index is 7.1, under the burst
     threshold; the same fall reads 13.9 in the slice. The three-link
     layout sees 16.3 and fires.
   - **A fallen person read as nobody.** In the rich-NLoS room on one
     link, occupancy reads empty for 92.7% of the five minutes after
     the fall. The detector takes that as the person having left and
     drops the burst. Occupancy saying nobody is there while someone
     lies on the floor is the worse half of this finding. It is one
     collapse run of 36, on one link, in a simulator; no other collapse
     run in either model reads empty after the fall. But it is the
     failure the whole system exists to avoid.

Nothing is retuned on the strength of this. The thresholds were tuned
on the slice's development seeds, and retuning them on the box would
fit a different model rather than a room. What the two columns give is
a spread: how far the simulator's own answer moves when the room
model does. The S5 recordings decide.

### Not done

- **Postures beyond a fall.** Standing versus seated heights, getting
  up, and lying in bed at night are not modelled. Night timelines are
  upright at 1.0 m.
- **Why the drop weakens the fall's burst on one and two links.** The
  geometry of a chest descending through the link's Fresnel zones has
  not been worked through; the number is measured, not explained.
- **The 2-D corner bug**, above.
- **Angle- and polarisation-dependent reflection.**
