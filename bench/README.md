# B1 bench session: the first real CSI

One ESP32-S3 TX and one RX on a desk, raw CSI recorded to this laptop, and
the two questions in `docs/HARDWARE.md` §9 answered. This page is the
order of work for the day; the reasons are in `docs/HARDWARE.md` and
`docs/PROTOCOL_S5.md` §1, which it points into.

**Exit gate** (the brief, task B1): `wifisense stats` shows at least 10 Hz
sustained, with the loss reported, and NOTES.md §10 "First real CSI" is
written: what matched the simulator and what did not.

| File | What |
|---|---|
| `edge/firmware/profiles/bench-tx.defaults` | The TX board: node 0, channel 6, 20 Hz |
| `edge/firmware/profiles/bench-rx.defaults` | The RX board: node 1, channel 6, 20 Hz, 56 subcarriers, **raw mode on**, vitals off |
| `edge/firmware/profiles/bench-rx2.defaults` | An optional second RX, node 2 |
| `bench/saans.bench.toml` | The gateway, for looking at recordings afterwards |
| `bench/session.sh` | Records the three segments and writes the session's README |

## Before the day

- **Two boards and the right cables:** `docs/HARDWARE.md` §1-3.
- **Docker** with `espressif/idf:v5.4.1` pulled, and your user able to open
  the serial device (on Arch the `uucp` group, on Debian and Ubuntu
  `dialout`). The board appears as `/dev/ttyACM0` on the S3's native USB
  port, or `/dev/ttyUSB0` on a UART-bridge port.
- **Your written consent for yourself** (`docs/PROTOCOL_S5.md` §2.3), and
  a sign on the door (§2.2). Until an ethics committee says otherwise you
  are the only person who may be recorded (§2.1). The seated segment is you.
- **A clock that slews, not steps:** chrony or ntpd (`PROTOCOL_S5` §3.2).
  `session.sh` saves `chronyc tracking` at the start and end if it can.
- **The uplink AP on channel 6**, the same channel the boards sense on, or
  change `CONFIG_SAANS_WIFI_CHANNEL` in both profiles (`HARDWARE.md`, "The
  uplink channel").
- **A tape measure.** Every distance in the session README is measured.

Not covered today: counting AGC gain moves (`PROTOCOL_S5` §1.1). The
firmware does not read a gain yet, so that needs a firmware change first.

## 1. Build and flash the boards

The site's values come from the environment, so no file holds the WiFi
password:

```bash
cd edge
export SAANS_GATEWAY_IP=<this laptop's address on the WiFi the RX joins>
export SAANS_WIFI_SSID=<that network>
read -rs SAANS_WIFI_PASSWORD && export SAANS_WIFI_PASSWORD
```

```bash
PROFILE=bench-tx ./build.sh build
PROFILE=bench-tx PORT=/dev/ttyACM0 ./build.sh flash
```

```bash
PROFILE=bench-rx ./build.sh build
PROFILE=bench-rx PORT=/dev/ttyACM0 ./build.sh flash-monitor
```

Each profile builds into its own `edge/firmware/build-<profile>/`, from
its profile file, every time. Building one board cannot change another,
and `menuconfig` is refused for a profile: change the profile file.

**On the RX's serial monitor** (`HARDWARE.md` §7), in this order:

- `role : RX (sensing)`, `node id : 1`, and the warning `raw mode : ON`;
- `got ip …`, then `uplink -> <SAANS_GATEWAY_IP>:5566`;
- `CSI capture started`, then `keep the space EMPTY for ~80 s`.

**On the TX**, `ESP-NOW rate: HT20 MCS0 LGI`. Without that line the RX
measures the wrong subcarriers (`HARDWARE.md` §7).

**Write down from the RX's `capture:` line** after a minute: `len=`, the
subcarrier count it implies, and `first_word_invalid=` against `frames=`.
That is question 1, and the first half of question 2 (`HARDWARE.md` §9).

## 2. Record

Power-cycle the RX with the room empty, give it the ~80 s it asks for,
then:

```bash
bench/session.sh desk-01
```

It asks for the room, the measured TX-RX distance, the heights and the
orientation. It then records `empty`, `seated` and `walking` for 10 minutes
each (`SEGMENT_S` changes that), runs `wifisense stats` on each, and writes
`data/bench/<date>-desk-01/README.md`. Before each occupied segment it asks
you to confirm the person is you. It refuses to start while anything else
holds UDP 5566, because a second listener takes half the datagrams.

## 3. Look at it

```bash
python3 -m wifisense stats data/bench/<session>/empty.pcap          # also in empty.stats.txt
python3 -m wifisense replay data/bench/<session>/seated.pcap --node 1 --rate auto
cd rust && cargo run --release -p saans-gateway -- -c ../bench/saans.bench.toml \
    --replay ../data/bench/<session>/seated.pcap --serve            # http://127.0.0.1:8080/app
```

In `stats`, about 1,200 "malformed (not raw CSI)" in a 10-minute segment
are the RX's own FeatureFrames, which share the port (`PROTOCOL_S5` §1.5).
Kernel socket drops are this laptop's loss, not the radio's.

## 4. Write it up

The session README ends with the list. Its results go in NOTES.md §10,
and every number there is labelled **REAL** with the recording it came
from:

- rate and loss per segment;
- which bins are dead: mean amplitude per bin index on `empty.pcap`
  (question 2, `PROTOCOL_S5` §1.2);
- the `first_word_invalid` fraction (question 1, §1.3);
- amplitude noise per subcarrier on `empty.pcap`, as mean, std and CV.
  This sets the simulator's noise, and then `evaluate.py --suite nulls`
  is re-run at it, beside the old table (§1.4).

**What gets committed:** the session README and the stats files. **The
recordings do not.** `.gitignore` excludes `*.pcap`, a recording with a
person in it is health data (ADR-001, ADR-005), and this repository is
public. The brief asked for the pcaps to be committed under `data/bench/`;
that conflicts with the rule above, so it is your decision. An empty-room
recording holds no person, and could be added with `git add -f` if you
want it public.

If something does not work, `docs/HARDWARE.md` §8 lists what to check, in
order.
