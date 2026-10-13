# Saans edge — ESP32-S3 sensor node

The on-node half of Saans: a header-only C++17 DSP library and the ESP-IDF
firmware that runs it. The node captures WiFi CSI, runs the whole sensing
pipeline locally, and sends 68-byte **FeatureFrames** to the gateway at 2 Hz.

**Raw CSI never leaves the node** except in an explicit debug mode, and that
mode is visible in every frame's flags so a dashboard can show it (ADR-001).

```
edge/
  dsp/            header-only C++17 DSP — no heap after init, float32, no deps
    test/         host tests against spec/golden vectors
    bench/        host benchmark vs the Python reference
  firmware/       ESP-IDF v5.4 app, esp32s3, TX and RX roles
  build.sh        Docker-pinned build (no local IDF install needed)
```

## Quick start

```bash
./build.sh tests          # host DSP tests against the golden vectors
./build.sh                # build the firmware for esp32s3
./build.sh menuconfig     # configure role, channel, consent, gateway
PORT=/dev/ttyUSB0 ./build.sh flash-monitor
```

Nothing needs installing beyond Docker: the toolchain lives in
`espressif/idf:v5.4.1`.

## The two roles

A deployment needs **at least two boards**. Pick the role in `menuconfig`
under *Saans node → Node role*; it is a compile-time choice, so a TX binary
contains no sensing code at all.

| Role | What it does |
|---|---|
| **TX** | Broadcasts a 12-byte ESP-NOW frame at a fixed rate (default 50 Hz). Does no sensing and holds no state about people. It exists so the RX has a channel to measure. |
| **RX** | Captures CSI, runs the DSP, emits FeatureFrames over UDP to the gateway. Optionally also streams the raw CSI debug format. |

Both must be on the **same WiFi channel** — ESP-NOW is connectionless and the
channel is the only thing tying them together.

### Why a dedicated TX

You cannot measure a channel nobody is using. Relying on ambient traffic means
a sample rate you do not control and that varies with whatever else is on the
network. A TX node makes the rate a parameter instead of a hope, which the
whole DSP depends on: the pipeline assumes a uniform time axis, and the gap
filler can only repair gaps it can see.

## Privacy, and where to audit it

Breathing and heart data are health data under India's DPDP Act 2023. Consent
is **per space, opt-in, and off by default** (ADR-005).

**The privacy boundary is one function:**
[`apply_consent_gate`](dsp/feature_frame.hpp) in `edge/dsp/feature_frame.hpp`.

It zeroes `breath_rms`, `breathing_bpm`, `breathing_conf`, `heart_bpm` and
`heart_conf` unless the space has consented, with `heart_*` additionally
requiring heart consent. It is idempotent, it runs on the node before the bytes
exist, and `encode_feature_frame_gated` is the only encode path the firmware
uses. A reviewer should be able to satisfy themselves about the privacy
guarantee by reading that one function and checking its callers.

Three layers reinforce it:

1. **Compile-time.** With `CONFIG_SAANS_VITALS_ENABLED` off, the consent value
   is a `constexpr false`. A node built without consent has no code path that
   emits a breathing rate — it cannot be talked into one by a malformed packet.
2. **Wire-level.** The gate runs immediately before encoding, so the gateway
   receives zeros, not something it is trusted to discard.
3. **Visible.** `vitals_enabled` and `raw_mode` are flags in every frame, and
   the consent state is logged at every boot.

`motion_energy`, `gross_motion`, `occupied` and `best_bnr` are deliberately
**not** gated: they are presence and link quality, not health data, and the
system is useless without them.

## Configuration that matters

Under `menuconfig → Saans node`:

| Option | Default | Why |
|---|---|---|
| Node role | RX | TX/RX, compile-time |
| Node id | 0 | 255 is reserved for "unknown/AP" in `tx_id` |
| WiFi channel | 6 | TX and RX must agree |
| TX beacon rate | 50 Hz | Bounds the achievable CSI rate |
| Subcarriers | 56 | HT20 usable count; sizes the DSP at compile time |
| CSI rate | 20 Hz | Must be ≥ 10 Hz — the gross-motion band reaches 4 Hz |
| FeatureFrame rate | 2 Hz | What the gateway receives |
| Calibration | 30 s | **The space must be EMPTY**, after filter warmup |
| Vitals / heart | off | Consent, see above |
| Raw mode | off | Debug only; visible in every frame |

### Calibration is not optional

The presence detector learns an empty-room threshold. Two things must hold or
it misbehaves in ways that look like tuning problems and are not:

- **The space must actually be empty during calibration.** Calibrate with
  someone present and it learns them as the baseline; the room then reads as
  permanently empty.
- **The filters must settle first.** A biquad near 0.1 Hz takes tens of seconds
  to settle, so the DSP discards a warmup period (~50 s) before accumulating
  calibration statistics. Total quiet time needed is warmup + calibration,
  about 80 s at defaults.

`calib_age_s` in every FeatureFrame reports how long ago calibration completed,
and `0xFFFFFFFF` means not calibrated.

## Verification

The DSP is validated against **golden vectors** generated from the Python
reference (`spec/golden/`, 21 sets). The same headers compile for the host
tests and for the firmware, so the tests are evidence about the shipped code.

```bash
./build.sh tests
```

Match semantics differ by vector, deliberately (ADR-003):

- **`feature_codec` → byte equality.** The encoder must produce the exact 68
  bytes the reference does.
- **`features_*` → field-wise.** Python computes in f64 and rounds to f32 only
  at the wire; the edge computes in float32 throughout, so the low bits differ
  by design. Continuous fields match within 1e-4 relative and 0.05 BPM; integer
  fields exactly; `uptime_ms`/`csi_rate_chz`/`loss_permille` within ±1.

A tolerance is never loosened to make a test pass. If the port disagrees with
the reference, one of them is wrong.

## Wire formats

Two datagram types share the sensor port (default 5566), distinguished by the
first four bytes. Full spec in [`spec/wire.md`](../spec/wire.md).

| Magic | Type |
|---|---|
| `0x534E4153` ("SANS") | FeatureFrame v1, 68 bytes — normal operation |
| `0xC5110001` | raw CSI — debug only |

The raw format is defined in [`spec/wire.md`](../spec/wire.md) §1 and the
Python tooling parses it unchanged. I/Q pairs are **imaginary first** — reversing them
leaves amplitude correct and mirrors every phase, which survives amplitude-only
testing and then quietly breaks anything phase-based.

## Hardware

- **2× ESP32-S3 DevKit** (~$6–10 each). One TX, one RX.
- A 2.4 GHz network for the RX's uplink to the gateway.

The S3's FPU is **single precision**, which is why the DSP is `float`
throughout and the host build uses `-Wdouble-promotion`: every double in the
hot path is a software-emulation trap costing 10–50×.

## Known limits

- **One link per RX node.** Multi-link spaces are phase S6.
- **Heart rate is unreliable.** CSI amplitude is nonlinear in chest
  displacement, so breathing creates intermodulation sidebands that can exceed
  the true cardiac peak. The DSP uses a symmetric-triplet search to suppress
  this and reports a confidence, but the honest position is that breathing is
  solid and heart rate is not. See `../NOTES.md`.
- **Blind spots are real.** A subject can sit at a Fresnel null where breathing
  is nearly invisible, and at ESP32's 20 MHz the whole band sits at essentially
  the same Fresnel phase, so subcarrier diversity cannot rescue it. See
  `../RESEARCH.md` §5a. The mitigation is more links, not more cleverness.
- **Not a medical device.** Outputs are advisory and must abstain below a
  confidence threshold. They never replace existing procedures.

## Attribution

Portions of `firmware/main/csi_capture.*` are adapted from MIT-licensed code.
The copyright and permission notice is in [`/NOTICE`](../NOTICE).
