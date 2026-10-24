# Hardware: bill of materials and bench bring-up

The first real hardware run. The brief calls it Track B1; inside this
repository it is phase **S5**, whose exit gate is "Presence F1,
still-person false-empty, breathing vs belt — the P1 gates"
(`SAANS_PLAN.md:184`). Those gates are stated in full in
`CAMPUS_SENSING_PROMPT.md:157`: presence F1 ≥ 0.95 over 48 h,
still-person false-empty < 5%, breathing within ±1 BPM of the belt in
≥ 80% of 60 s windows, seated.

**No board has ever run this firmware.** Every number in `NOTES.md`,
every table in this repository's documentation, every accuracy figure on
the site: all of it is simulated, and `docs/CAPABILITY_LIMITS.md:338`
says so in those words. Nothing in this file is a measurement either. It
is a shopping list and a procedure, built by reading the firmware that
will run on the parts.

What this session is *for* is in §9. Two specific questions are recorded
as open in `NOTES.md` §9, and a bench run is the only thing that closes
them.

---

## 1. What to buy, and how many

| Item | Count | Why that count |
|---|---|---|
| ESP32-S3 devkit | **3** (buy 4) | See below |
| USB data cable, one per board | 3–4 | See §3 |
| A 2.4 GHz access point you control | 1 | `edge/README.md:191`: the RX needs an uplink to reach the gateway |
| A host machine with Docker | 1 | §4 |
| A respiration belt or counted-breath protocol | 1 | The P1 breathing gate is *against a reference*; without one the run cannot be scored |
| Tape measure | 1 | `docs/CAPABILITY_LIMITS.md:225-231`: node positions are typed in by a human, never inferred from the radio |

**Two boards is the floor.** `edge/README.md:33`: "A deployment needs at
least two boards." One runs the TX role and does nothing but put frames
in the air at a fixed rate; the other runs the RX role, captures CSI,
runs the DSP and sends FeatureFrames. The roles are a compile-time
choice (`edge/firmware/main/main.cpp:71-75`), so they are two different
binaries, not one binary with a switch.

**Three is the number to actually buy**, because of ADR-006. Its
decision clause reads: "Spaces where breathing matters get at least two
links at different angles" (`docs/adr/006-link-diversity.md:12`). The
reason is measured, in simulation: at ESP32's HT20 bandwidth a
Fresnel-zone blind spot usually covers the whole band at once, and
subcarrier selection recovered whole-band-null breathing accuracy only
from 64.5% to 68.1% (`docs/adr/006-link-diversity.md:7-9`, SYNTHETIC).
`NOTES.md:489-490` records the other half of the same result: a second link
took breathing coverage from 94% to 100% and halved missed pauses, also
SYNTHETIC. One link is enough to prove the pipeline works at all. Two
links are what the system is specified to need.

**Two links means one TX and two RX, not two TX and two RX.** The TX
broadcasts over ESP-NOW to `FF:FF:FF:FF:FF:FF`, so every listener on the
channel sees the same frame (`edge/firmware/main/tx_role.cpp:31-32`).
Each RX independently measures its own path from that one transmitter,
and two RX nodes placed at different angles to the subject give two
links with different Fresnel geometry. Going the other way does not work
today: the RX hard-codes `tx_id = 0xFF` — "one TX beacon source, not
individually identified yet" (`edge/firmware/main/rx_role.cpp:76`) — so
a single RX hearing two transmitters would fold both into one link and
the DSP would be averaging two different channels. `edge/README.md:198`
states the same limit from the other side: "One link per RX node.
Multi-link spaces are phase S6."

The gateway already expects exactly this shape. The example config's
reading room is `links = [{ rx = 1, tx = 255 }, { rx = 2, tx = 255 }]`
(`rust/crates/saans-gateway/saans.example.toml:64`): two receivers,
`tx = 255` meaning "an access point / unknown", which is what the
firmware sends.

**Buy a fourth as a spare.** That began as a judgement -- a dead or
flaky board stops a bench session that took a week to schedule, and a
spare costs less than a second trip -- and a measurement now agrees
with it. Wired as a third receiver, a fourth board finds no extra
people in a room this size: in simulation, seats where a still person
is mostly missed go 44 → 0 of 1,200 from one link to two, and stay
there at three (`NOTES.md` §11, SYNTHETIC). What the third link does buy
is fewer false "occupied" readings from noise. Worth knowing before
buying for detection; not a reason to skip the spare.

Buy them all the same variant. The node id is compile-time
(`CONFIG_SAANS_NODE_ID`, `edge/firmware/main/Kconfig.projbuild:18-24`),
so each board gets its own build anyway, but mixed board revisions add a
variable to a session whose whole purpose is to remove variables.

### What it costs

A devkit is roughly **₹600–900** each. [UNVERIFIED: price not checked,
varies by seller.] That figure came from the build prompt this file was
written against — a chat message, not a file in the tree. It is not in
the repository, and `CAMPUS_SENSING_PROMPT.md:23` leaves the hardware
budget as an unfilled placeholder (`- Pilot hardware budget: [₹ ...]`),
so do not go looking for it there. What the repository says is
**$6–10** each, in five places that all trace to the same estimate:
`edge/README.md:190`, `README.md:304`, `NOTES.md:182`,
`RESEARCH.md:322` and `CAMPUS_SENSING_PROMPT.md:66`. [UNVERIFIED: that
figure is an estimate recorded in the docs, not a quote from a seller,
and no price in this repository has been checked against a current
listing.]

So three boards is three times the brief's per-board figure, and that
is as precise as this file is willing to be. No vendor names, no links,
no current prices: `docs/CAPABILITY_LIMITS.md` makes the point for the
expensive hardware and it holds here too — "build a real BOM including
Indian import duty before committing" (`docs/CAPABILITY_LIMITS.md:287`).
Price the cables, the AP and the respiration belt separately; the belt
is likely to cost more than the radios.

### What *not* to buy

`RESEARCH.md:325-327`: "Do not buy 4 nodes expecting multilateration.
There is no published accuracy figure for that configuration, and TDMA
across nodes costs the sample rate breathing needs." More nodes buy
blind-spot coverage, not new capabilities (`README.md:306-307`). Pose,
through-wall imaging, RTI, metre-level tracking and inferring room
geometry from the radio are all ruled out by physics the hardware cannot
change; `docs/CAPABILITY_LIMITS.md` records which quantity decides each
one so the question does not get re-asked after the boards arrive.

---

## 2. The exact chip

**ESP32-S3.** Not a guess, and not interchangeable with the C6 that
`README.md:304` mentions as an alternative CSI receiver. The build
script defaults `TARGET` to `esp32s3` (`edge/build.sh:23`) and
`edge/firmware/sdkconfig.defaults:4` pins `CONFIG_IDF_TARGET="esp32s3"`.
`edge/README.md:15` names the same target.

Requirements the firmware actually places on the board:

- **At least 4 MB of flash.** `edge/firmware/sdkconfig.defaults:15-16`
  sets `CONFIG_ESPTOOLPY_FLASHSIZE_4MB`, and the comment above it
  (lines 12-14) explains why: the dual-OTA partition table does not fit
  in IDF's 2 MB default. `edge/firmware/partitions.csv` lays out two
  1.75 MB app slots (`0x1C0000` each) at `0x20000` and `0x1E0000`, so
  the table itself needs 3.75 MB. Most S3 devkits ship with 8 MB or
  more; `sdkconfig.defaults:14` calls 4 MB "a safe floor".
- **No PSRAM needed.** Nothing in `edge/firmware/sdkconfig.defaults`
  enables it. The DSP allocates nothing after init
  (`edge/README.md:12`) and the per-link state is sized against internal
  DIRAM: 45,408 bytes a link on the S3 with `best_bnr` on, 16,584 with
  it off (`edge/dsp/node_config.hpp:458-461`), against a budget of
  171 KB of free DIRAM (`edge/dsp/node_config.hpp:11`). A bench node
  runs one link, so memory is not a constraint here — it becomes one at
  S6.
- **240 MHz.** `edge/firmware/sdkconfig.defaults:32`.
- **Single-precision FPU, which is why the DSP is `float` throughout.**
  `edge/README.md:193-195`: "every double in the hot path is a
  software-emulation trap costing 10–50×."

A board with a PCB or external antenna is preferable to one with a
chip-only trace antenna, but nothing in this repository measures the
difference, so treat that as a preference rather than a requirement.

---

## 3. Cables, and the one that wastes an afternoon

One USB cable per board, and **it must be a data cable, not a
charge-only cable.** This is the single most common first-hour failure
on any microcontroller bring-up: a charge-only cable has power and
ground wired and the data pair absent or not connected, so the board
powers up, its LED lights, and no serial device ever appears on the
host. Nothing is wrong with the board, the firmware or the toolchain,
and every diagnostic points somewhere else. If `/dev/ttyUSB*` or
`/dev/ttyACM*` does not appear within a second of plugging a board in,
change the cable before changing anything else.

The second cable trap is the connector on the board. Many ESP32-S3
devkits expose **two** USB sockets — one wired to a USB-to-UART bridge
chip and one wired to the chip's native USB peripheral — and they do not
enumerate the same way on the host. Everything in this repository is
written for the bridge: `edge/build.sh:18` documents `PORT` as "serial
port for flash/monitor, e.g. `/dev/ttyUSB0`", and `edge/README.md:25`
uses the same path. [UNVERIFIED: which socket on a given devkit
enumerates as which device node, and whether the native-USB port works
with this build's flash and monitor settings — not checked against a
board, because there is no board.] If two device nodes appear, try the
one named `ttyUSB*` first.

Power: a devkit on a host USB port is the normal bring-up arrangement.
A TX node left running as a channel metronome for a 48-hour presence
gate wants a mains USB supply rather than a laptop that sleeps.

---

## 4. Host toolchain

Read off `edge/build.sh`, not assumed.

**Docker, and nothing else.** `edge/build.sh:4-7`: "Nothing is installed
on the host: the toolchain, Python environment and target support all
live in `espressif/idf:v5.4.1`. That is the whole reason this script
exists rather than a README line telling you to install IDF."
`edge/README.md:28-29` repeats it: "Nothing needs installing beyond
Docker."

The pinned image is **`espressif/idf:v5.4.1`** (`edge/build.sh:22`),
overridable with the `IDF_IMAGE` environment variable. The first build
pulls it; budget time for that on a campus connection. [UNVERIFIED:
image download size not checked.]

Three things the script does that matter when something goes wrong:

- It mounts the **repository root**, not `edge/firmware`, because the
  firmware includes `../../dsp` and a narrower mount breaks the include
  path "in a way that is tedious to diagnose" (`edge/build.sh:27-30`).
- It runs the container as the **invoking user**, not root
  (`edge/build.sh:33-36`), so `build/` and `sdkconfig` stay cleanable
  from the host. It also means the invoking user needs read/write access
  to the serial device: `--device "${PORT}"` is passed through to Docker
  (`edge/build.sh:37-39`), and a permission error there surfaces as a
  flash failure, not as a permissions message.
- `./build.sh tests` needs **no container at all** — it is `make -C
  edge/dsp/test check` on the host with `g++` (`edge/build.sh:72-75`).
  That is the one command to run before touching hardware, because it
  compiles the same headers the firmware compiles and checks them
  against the golden vectors (`edge/README.md:152-159`).

On the gateway side: `cargo build --release -p saans-gateway -p
saans-cli` (`README.md:163`), or the pure-stdlib Python path,
`python3 -m wifisense listen --port 5566 --rate 20` (`README.md:145`),
which needs no install step at all (`README.md:10-12`).

---

## 5. Build, configure, flash

The quick-start sequence is `edge/README.md:21-26`:

```bash
./build.sh tests          # host DSP tests against the golden vectors
./build.sh                # build the firmware for esp32s3
./build.sh menuconfig     # configure role, channel, consent, gateway
PORT=/dev/ttyUSB0 ./build.sh flash-monitor
```

For a three-board bench that sequence runs **three times**, because role
and node id are both compile-time. There is one `build/` directory and
one `sdkconfig` (`edge/build.sh:58`), so the boards are done in turn:
configure, build, flash that board, reconfigure, build, flash the next.

| Board | `menuconfig → Saans node` | Then |
|---|---|---|
| TX | Node role = **TX**, Node id = **0**, WiFi channel = **6** | `PORT=… ./build.sh flash` |
| RX A | Node role = **RX**, Node id = **1**, same channel, gateway IP + SSID | `PORT=… ./build.sh flash-monitor` |
| RX B | Node role = **RX**, Node id = **2**, same channel, gateway IP + SSID | `PORT=… ./build.sh flash-monitor` |

Node id 255 is not available: it is reserved for "unknown / an access
point" in `tx_id` (`edge/firmware/main/Kconfig.projbuild:22-24`, and
asserted at compile time in `saans_config.h:108`).

Settings worth setting deliberately, with their defaults:

| Option | Default | Source | Note for the bench |
|---|---|---|---|
| WiFi channel | 6 | `Kconfig.projbuild:28-34` | TX and RX **must** agree; ESP-NOW is connectionless and the channel is the only thing tying them together |
| TX beacon rate | tracks the CSI rate | `Kconfig.projbuild:36-40` | Set it to the same number as the RX's CSI rate. Faster is actively harmful; see below |
| CSI rate | 20 Hz | `Kconfig.projbuild:70-74` | Minimum 10 Hz, because the gross-motion band reaches 4 Hz (`saans_config.h:114`) |
| Subcarriers | 56 | `Kconfig.projbuild:61-65` | An assumption, not a measurement — this is question 2 in §9 |
| FeatureFrame rate | 2 Hz | `Kconfig.projbuild:92-97` | |
| Calibration | 30 s | `Kconfig.projbuild:99-102` | Plus ~50 s of filter warmup; the space must be empty for both |
| Vitals / heart | off | `Kconfig.projbuild:146-162` | Leave off unless the space has written consent (ADR-005) |
| Raw mode | off | `Kconfig.projbuild:164-171` | **Turn this on for the bring-up session**; see §6 |
| Telemetry interval | 60 s | `Kconfig.projbuild:205-208` | How often the diagnostic lines in §7 print |
| Watchdog | 15 s | `Kconfig.projbuild:200-203` | |

Two traps in this table.

**The TX beacon rate must equal the RX's CSI rate.** The Kconfig help
spells out why at `Kconfig.projbuild:45-53`: nothing between the radio
and the DSP decimates, so beaconing faster makes the RX "process more
frames than it believes it is processing, so its sample clock, its
vitals bands and its gap filler are all scaled wrong". Beaconing slower
is fine — that is what the gap filler exists for. Note that
`edge/README.md:39` and `edge/README.md:91` both say the TX default is
50 Hz; the Kconfig default is `SAANS_CSI_RATE_HZ`, which is 20. Read the
value in `menuconfig`, not the README.

**`./build.sh clean` deletes `sdkconfig`** (`edge/build.sh:58`), which
means it deletes your `menuconfig` answers along with the build output.

### Signing the frames (ADR-012)

**Not for the bring-up session.** Raw CSI is never signed, and a gateway
with a `[[frame_key]]` refuses it, so §6 runs with no key anywhere. Before
frames go onto a network anyone else can reach (B2 onward), give each RX
board its key:

```bash
saans-gateway --new-site-key                         # once per site; keep it secret
export SAANS_FRAME_KEY=<that key>                    # where the gateway runs
export SAANS_NODE_KEY=$(saans-gateway -c saans.toml --node-key 1)
KEY_ID=0 PORT=/dev/ttyUSB0 ./build.sh provision      # RX A; then --node-key 2 for RX B
```

`provision` writes the key, its `key_id` and a boot counter into the
`saanskey` partition (`edge/firmware/partitions.csv`). The counter starts
at the Unix time, so re-provisioning a board never trips the gateway's
replay check. `./build.sh flash` does not touch that partition, so
reflashing the app keeps the key. `DRY_RUN=1` builds the image and shows
what is in it, key redacted, without a board. At boot an RX logs
`frames are SIGNED: key_id 0, boot …`. A board without a key logs
`frames are UNSIGNED` at every telemetry report, and a gateway with a
`[[frame_key]]` refuses its frames (`/api/health` → `ingest.frame_auth`).

### The uplink channel

`run_rx_role` associates to the AP, waits up to 30 s for an IP, opens
the UDP socket and *then* forces the radio to `CONFIG_SAANS_WIFI_CHANNEL`
(`edge/firmware/main/rx_role.cpp:117-125`, calling
`esp_wifi_set_channel(channel, WIFI_SECOND_CHAN_NONE)` at
`net.cpp:124-126`). If the uplink AP is on a different channel from the
sensing channel, those two requirements are in direct conflict. Put the
RX's uplink AP on the same channel you configure for sensing. The
firmware does not currently compare the two or warn about it.

### Consent and ethics, before a person is in the room

The empty-room and idle-node parts of this session involve no human
subject and need nothing. The moment someone sits in the room to be
breathed at, it is human-subject data collection: `SAANS_PLAN.md:196`
requires ethics committee approval before any of it, and ADR-001's
consequences clause says a raw debug collection requires it explicitly —
"That friction is the point" (`docs/adr/001-edge-private-features.md:23`).
Leave `CONFIG_SAANS_VITALS_ENABLED` off until the space has opted in in
writing (ADR-005 clause 1).

---

## 6. First capture

Raw mode is off by default and should normally stay off — ADR-001 is the
reason the node runs the DSP locally at all. But the two questions this
session exists to answer (§9) are both about what the radio hands the
firmware, and that is only visible in the raw I/Q. So for the bring-up
session, build the RX with `CONFIG_SAANS_RAW_MODE=y`. The `raw_mode`
flag is then set in every FeatureFrame (`spec/wire.md:56`) so nothing
about it is silent, and the RX logs it as a warning at boot
(`main.cpp:43-45`, `rx_role.cpp:168`).

Both datagram types go to the same destination,
`CONFIG_SAANS_GATEWAY_IP:CONFIG_SAANS_GATEWAY_PORT` (default
`192.168.1.10:5566`, `Kconfig.projbuild:177-184`), and are told apart by
their first four bytes: `0xC5110001` for raw CSI, `0x534E4153` ("SANS")
for the 68-byte FeatureFrame (`spec/wire.md:6-9`). Point the gateway IP
at the laptop.

On the host, record first and analyse afterwards (`README.md:262-265`):

```bash
python3 -m wifisense record --port 5566 --out bench01.pcap --duration-s 600
python3 -m wifisense stats bench01.pcap
python3 -m wifisense replay bench01.pcap --node 1 --rate auto
```

`record` writes every raw datagram, malformed ones included, to a
nanosecond pcap that opens in Wireshark. `stats` reports the per-node
rate, sequence-gap loss, reboots, duplicates, reordering and
inter-arrival jitter actually received — `README.md:271` says to check
it before trusting vitals on a link, at least 10 Hz sustained. `replay`
runs the pipeline offline and is deterministic, so the same recording
can be re-analysed after the DSP changes, which is the whole reason to
record rather than only to stream.

A live view instead of a recording is
`python3 -m wifisense listen --port 5566 --rate 20` (`README.md:145`).

**Do not commit the recordings.** `.gitignore` excludes `*.pcap`,
`*.pcapng`, `saans-data/`, the event logs and the audit log, with the
reason written beside them: "they can contain health data (ADR-001,
ADR-005)". A raw-CSI capture of a room with a person in it is exactly
the artefact ADR-001 exists to keep off a server.

Record both an empty room and an occupied one in the same session.
ADR-005 clause 5 (`docs/adr/005-advisory-consented-abstaining.md:16-17`):
"Every detector reports both its miss rate and its false-alarm rate per
hour; publishing one without the other is not allowed." A presence
result from occupied segments alone cannot be reported.

---

## 7. What "working" looks like on the serial monitor

`PORT=/dev/ttyUSB0 ./build.sh monitor`. These are the lines the firmware
actually prints; each is cited so you can find the code that emits it.

**At boot, both roles** (`main.cpp:23-47`, `telemetry.cpp:36`):

```
I saans: Saans node starting
I saans:   role        : RX (sensing)
I saans:   node id     : 1
I saans:   channel     : 6
I saans:   subcarriers : 56
I saans:   csi rate    : 20 Hz
I saans:   frame rate  : 2 Hz
I saans:   vitals      : disabled
I saans:   heart       : disabled
I saans.health: boot #1
```

The consent state is printed at every boot on purpose: "Someone reading
a serial log should never have to guess whether this node is emitting
health data" (`main.cpp:36-38`). A rising `boot #N` across a session is
a node that is rebooting.

**The TX**, within a second of start (`tx_role.cpp:50, 86, 116`):

```
I saans.tx: TX role: node 0, channel 6, 20 Hz
I saans.tx: ESP-NOW rate: HT20 MCS0 LGI
...and every 60 s:
I saans.tx: sent=1200 failed=0 rate=20.0 Hz
```

`ESP-NOW rate: HT20 MCS0 LGI` is the line that matters. Without it the
RX is measuring L-LTF rather than the HT-LTF subcarriers the DSP is
sized for (`tx_role.cpp:63-70`). Its failure form is an error, not a
silence: `HT20 rate config FAILED (…)` (`tx_role.cpp:82-84`).

**The RX**, in order (`rx_role.cpp:113-154`, `net.cpp:47,145`,
`csi_capture.cpp:192`):

```
I saans.rx: RX role: node 1, 56 subcarriers @ 20 Hz -> FeatureFrames @ 2 Hz
I saans.net: got ip 192.168.1.42
I saans.net: uplink -> 192.168.1.10:5566
I saans.rx: extractor ready (45408 bytes/link + 6404 bytes of shared spectral arena, best_bnr on)
I saans.csi: CSI capture started (queue 32 frames, N bytes each)
I saans.rx: keep the space EMPTY for ~80 s (50 s filter warmup + 30 s calibration)
I saans.rx: calibrating with someone present learns them as the baseline, and the room then reads as permanently empty
```

The 50 s and 30 s are computed, not typed: the warmup is five periods of
the presence band's low edge (`rx_role.cpp:63`) and the calibration is
`CONFIG_SAANS_CALIBRATION_S`. The byte figures are the extractor's own
`sizeof` on the part, which `rx_role.cpp:98-101` notes is "the only one
that is actually true of the node"; 45,408 and 6,404 are the values
`node_config.hpp:458-459` predicts for this configuration with
`best_bnr` on.

**Then every 60 s**, three lines (`rx_role.cpp:230-250`,
`telemetry.cpp:60`):

```
I saans.rx: csi=20.0 Hz frames=120 dropped=0 send_fail=0
I saans.rx: capture: len=... (first ...) -> ... subcarriers, expecting 56; 2nd_ch=0 first_word_invalid=0 short=0 layout_changed=0 too_short=0
I saans.health: up=60s heap=... min=... reboots=1 rssi=-48
```

Healthy means: `csi=` within a few percent of the configured rate,
`dropped=0`, `send_fail=0`, the subcarrier count in the `capture:` line
equal to what the DSP expects, `layout_changed=0`, `too_short=0`, and
`min=` (the heap low-water mark) flat across reports rather than
falling. `telemetry.cpp:56-59` explains why `min` is the number to watch
and not `heap`: the low-water mark only ever falls, so a downward trend
is a leak, and since the DSP allocates nothing after init any trend is a
bug somewhere else.

`edge/README.md:137` shows an illustrative `capture:` line reading
`len=128 (first 128) -> 64 subcarriers, expecting 56`. That is an
example of the format, not an observation. No board has produced one.

**And the FeatureFrames arrive.** Check this gateway-side, not with
`python3 -m wifisense stats`. The recorder decodes raw CSI only: it
parses every datagram with `parse_esp32_frame`, which rejects anything
whose magic is not `CSI_MAGIC` = `0xC5110001`
(`wifisense/sources/esp32_udp.py:96-97`), so a 68-byte FeatureFrame
(`0x534E4153`) is counted file-wide as malformed and never per node.
A capture full of working FeatureFrames and a capture full of garbage
look the same through `stats`.

Use `/api/health` on a running gateway instead, which lists anything
that arrived from a link no space claims (`README.md:219`) — the
quickest way to find a node id that does not match the config.

---

## 8. When it is not working: what to check, in order

The firmware was written to answer this itself. Work down its own
warnings.

**1. `no CSI for 1 s (TX down, or channel mismatch?)`**
(`rx_role.cpp:178`). The RX has an uplink but no CSI. Check that the TX
is powered and printing `sent=` lines, and compare the `channel :` line
in the two boot banners. This warning exists because "silence from a
sensing node reads downstream as 'nothing happening in that room', which
is the dangerous failure" (`rx_role.cpp:175-177`).

**2. `HT20 rate config FAILED`** on the TX (`tx_role.cpp:82`). Frames
are going out non-HT, the RX is slicing L-LTF, and every claim about
subcarrier count is void.

**3. `THE RADIO IS DELIVERING FEWER SUBCARRIERS THAN THE DSP EXPECTS`**
(`rx_role.cpp:252-255`). The tail of each frame is zero-padded, "which
the BNR stage reads as a quiet channel". Either the TX is not sending
HT20 (check 2) or `CONFIG_SAANS_N_SUBCARRIERS` does not match what
arrives.

**4. `CSI buffer length changed mid-run`** (`rx_role.cpp:258-261`). The
subcarrier indices do not mean the same thing across frames and the
static suppressor is averaging two different channels together. Nothing
downstream is interpretable until this is zero.

**5. `CSI arriving at X Hz but the DSP is configured for Y Hz`**
(`rx_role.cpp:280-287`). Fires above 1.25× the configured rate. Set the
TX beacon rate equal to the RX's CSI rate. The related, quieter warning
is `link-stats window overflowed N time(s) this interval`
(`rx_role.cpp:298-300`): the rate is right on average but bursty, so
`csi_rate_chz` and `loss_permille` cover less than the spec's 10 s.

**6. `HALTED: extractor misconfigured, this node is NOT sensing`**
(`rx_role.cpp:142-146`). A compile-time capacity mismatch. The node
refuses to run and keeps saying so on the telemetry interval rather than
logging once at boot and scrolling away — because "an inert link looks
exactly like an empty room from the gateway" (`rx_role.cpp:90-91`).
Rebuild with matching Kconfig.

**7. No `got ip`, or `disconnected, retrying`** (`net.cpp:34`,
`rx_role.cpp:122`). The RX keeps capturing CSI without an uplink by
design, so this does not stop sensing — it stops delivery. Check SSID,
password, and the channel conflict described in §5.

**8. Nothing on the wire at all, but the RX looks healthy.** Check
`uplink -> IP:PORT` against where the gateway is actually listening, and
the node ids against the `links` in the space config.

**9. No serial device at all.** The cable (§3).

---

## 9. The two open questions this session exists to answer

Both are quoted verbatim from `NOTES.md` §9, "Known gaps in the layer
itself" (`NOTES.md:595-606`). They are the reason to do a bench session
at all rather than another simulation run.

### Question 1

> `first_word_invalid` is not modelled. Its real effect is a two-bin
> index shift on affected frames, not garbage, so a consumer of
> simulated raw CSI never sees the one thing the flag does.

What to do with it: the firmware already counts the flag and already
skips the two affected pairs rather than reading them as signal
(`csi_capture.cpp:76-91`), and reports the count in the `capture:`
telemetry line as `first_word_invalid=N`. So the measurement is just:
run an RX for a known number of frames and read that counter against
`frames=`. If it is zero, the gap is closed by observation and the
simulator does not need the term. If it is not zero, the number to
report is the *fraction* of frames affected, because that is what sets
how often the two-bin shift happens.

### Question 2

> Which 56 bins the firmware actually hands over is unresolved. The
> simulator's 56 are the idealised occupied set; by ESP-IDF's
> documentation the firmware's first 56 pairs may be a different 56
> entirely. `BinLayout` can produce either once a bench session says
> which. A subcarrier index may not mean the same thing on both sides.

This is the one that decides whether anything else from the session can
be compared against the simulator. The evidence is already plumbed:
`capture:` prints `len=` and the first `len` seen, and
`csi_capture.cpp:54-58` explains why `len` is the right proxy — on the
S3 the PHY mode cannot be read back, since `sig_mode`, `cwb` and `rate`
are fields of `wifi_pkt_rx_ctrl_t` on the original ESP32 only, and the
LTF layout determines the buffer length, so "a change in `len` IS a
change in what the subcarrier indices mean".

`edge/README.md:140-144` is explicit about the order of operations:
"**Read that line on the first bench session before trusting any
measurement**". Record raw CSI alongside, so the per-bin amplitude
profile can be inspected offline — the guard bands and the DC bin are
the landmarks that say which 56 of the FFT's bins the driver handed
over.

### A third number, from the same section

Not in the "Known gaps" list, but `NOTES.md:585-588` calls it "the
single most valuable number in the first recording":

> That is a bench question with a cheap answer: log `agc_gain` on an
> idle node in an empty room for ten minutes and count the moves.

It matters because the whole impairment result in `NOTES.md` §9 rests on
one uncalibrated constant, `agc_dither_prob`. The honest claim from the
sweep (`NOTES.md:574-576`, SYNTHETIC) is:

> Presence survives a gain that moves about once every four minutes
> (the 0.000 row: 0.02% of packets at 20 Hz). It fails completely at
> once every twenty-eight seconds (0.002: 0.18%). The sweep brackets the
> breakpoint between those two and does not locate it any more finely.

**The firmware cannot do this today.** Nothing in `edge/firmware/main/`
logs an AGC gain; the per-frame radio fields it records are `rssi` and
`noise_floor` (`csi_capture.cpp:107-108`), and both are carried in the
raw CSI header at bytes 16 and 17 (`csi_capture.cpp:234-235`). [UNVERIFIED:
whether the S3's `wifi_pkt_rx_ctrl_t` exposes an AGC gain field at all —
not checked against the ESP-IDF headers, which are inside the Docker
image.] The usable approximation with today's firmware is to record ten
minutes of raw CSI from an idle node in an empty room and count the
step changes in per-frame RSSI and in whole-frame mean amplitude; a
converged AGC loop that moves shows up as a staircase in both. Whether
that is the same quantity as `agc_gain` is the thing to establish before
the number is quoted anywhere.

---

## 10. What this session does not settle

It is one room. `docs/CAPABILITY_LIMITS.md:338` is the sentence to keep
in view while reading the first results: "no board has ever run this
firmware. Every number in `NOTES.md` is simulated." A single successful
bench run replaces that with "one board has run this firmware in one
room", which is a large improvement and still not a measurement of the
system.

The P1 gates (`CAMPUS_SENSING_PROMPT.md:157`) need 48 hours of presence
data, a still-person protocol and a breathing reference. Blind-spot rate
with 1 vs 2 vs 3 links is S6, not this (`SAANS_PLAN.md:185`). And
nothing in a bench session licenses any of the capabilities
`docs/CAPABILITY_LIMITS.md` rules out — a working link measures that a
body is there and how fast its chest moves, with a confidence and the
right to abstain, and that is the whole product.
