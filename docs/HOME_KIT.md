# Set up a home kit

One home, one older person living alone, a family member who gets the
alert (docs/ELDER_CARE.md). This page takes a box of boards to a running
system in three commands, one for the gateway, one per board and one to
check them. It is written so that someone who does not program can follow it.

**Nothing in this kit has run on real boards yet.** What has been checked,
and what waits for hardware day, is at the end.

## What you need

| Thing | How many | Notes |
|---|---|---|
| ESP32-S3 devkit | 4: one TX, three receivers | The S3, not another ESP32: the tool refuses any other chip (docs/HARDWARE.md §2) |
| USB **data** cable | one per board | A charge-only cable lights the board and does nothing else (§3) |
| The gateway machine | 1 | A Raspberry Pi 4 or 5 with Raspberry Pi OS (64-bit), or any Debian 12 machine or VM, always on, on the home WiFi |
| A laptop with Docker | 1 | Builds and flashes the boards. It can be the gateway machine |
| The home WiFi's name, password and channel | | 2.4 GHz. The boards sense on the WiFi's channel, so it must not change: fix it in the router (§5, "The uplink channel") |

Four boards make **one TX and three receivers**: two receivers in the room
that matters most (bedroom or bathroom; two links, because one can miss a
person lying still, ELDER_CARE.md §2) and one in a second room. The TX
is node 0. The receivers are nodes 1, 2, 3. Write the number on each
board with a marker.

## 1. The gateway

On the gateway machine:

```bash
sudo apt install -y git python3 docker.io
git clone https://github.com/devreddy1015/areu_there.git saans && cd saans
sudo python3 tools/homekit gateway
```

It asks:

- **the rooms**, and which receivers watch each: for example `bedroom`
  with `1,2` and `living` with `3`;
- **breathing, per room.** Breathing is health data. Say yes only for a
  room whose person has agreed **in writing** (ADR-005). You can turn it
  on later;
- **a password** for the dashboard's `admin`.

Then it builds the gateway, installs it as a service that starts at boot
(deploy/README.md), checks the config and prints the dashboard's address:

```
The dashboard:
  http://192.168.1.20:8080/app   (from a phone or laptop on this WiFi)
```

Keep that address: the receivers send to it. The first build on a Pi takes
a long time; later runs are quick. With Rust installed (`cargo`) the
build uses it; without, it builds in Docker.

The same, without questions:

```bash
sudo python3 tools/homekit gateway --room bedroom=1,2 --room living=3 \
    --vitals bedroom --admin-password-file pw.txt
```

Running it again is safe. It changes only what differs, keeps your config
and `/etc/saans/saans.env`, and restarts the service only if its binary,
config or secrets changed. `--check` shows what it would do and changes
nothing.

## 2. The boards

On the laptop with Docker, one board at a time. Plug it in using the USB
socket marked **UART** (or COM), then:

```bash
python3 tools/homekit node --role tx --id 0 --channel 6
python3 tools/homekit node --role rx --id 1 --channel 6 --gateway 192.168.1.20 --ssid HomeWiFi
python3 tools/homekit node --role rx --id 2 --channel 6 --gateway 192.168.1.20 --ssid HomeWiFi
python3 tools/homekit node --role rx --id 3 --channel 6 --gateway 192.168.1.20 --ssid HomeWiFi
```

`--channel` is your WiFi's channel and the same for every board. The
WiFi password is asked for; it is never typed on the command line, and
it is kept only in the git-ignored build folder. For each board the tool:

1. finds the board and warns about the cable and the socket (below);
2. asks the board which chip it is and stops if it is not an ESP32-S3;
3. builds the firmware for that board in Docker (edge/build.sh) and flashes it;
4. reads the board's log for about a minute and says whether it works.

Add `--vitals` for a receiver in a room with written consent for
breathing; the gateway's room must say so too. `--dry-run` checks
everything and builds nothing.

Put the boards in place: the TX where it can be seen from every room,
on mains power; each receiver in its room. **Leave each room empty for
about 80 seconds after its receiver powers on:** it is learning the room
empty, and a person there is learned as part of the room.

## 3. Check

On the gateway machine:

```bash
sudo python3 tools/homekit selftest
```

```
  frames arriving at 2.0 Hz from node 1, bedroom (CSI at 20.0 Hz on the node; loss 0.2%)
  frames arriving at 2.0 Hz from node 2, bedroom (CSI at 20.0 Hz on the node; loss 0.4%)
  frames arriving at 2.0 Hz from node 3, living (CSI at 19.9 Hz on the node; loss 0.1%)
PASS
```

(An illustration of the format: no board has produced these numbers.)

Healthy is 2 frames a second from every receiver and a CSI rate of at
least 10 Hz on each, near 20. A receiver that sends but is in no room is
named, with where it sends from. Then open the dashboard: every room
should say empty.

`selftest --serial /dev/ttyUSB0` reads one board's own log (what the node
command does after flashing). `selftest --udp` listens on the sensor port
itself, for when no gateway is running, on this machine's network address
(the one the boards send to; `--bind` to choose another). The log is never saved, because
the WiFi driver prints the network's name in it.

## When something is wrong

| You see | Do this |
|---|---|
| "No serial device", board lit | Change the cable: it is charge-only (docs/HARDWARE.md §3) |
| "the chip's native USB socket" | Move the cable to the board's other socket, marked UART |
| "you cannot open /dev/ttyUSB0" | Run the command it prints (`usermod -aG dialout`), log out and in |
| "could not talk to a chip" | Hold BOOT, tap RESET, release BOOT, run it again |
| "is an ESP32 ..., not an ESP32-S3" | Wrong board. This firmware is S3 only |
| "your WiFi is on channel 11" | Give `--channel 11` to every board |
| "no CSI", in the board's log | The TX is off or on another channel |
| "cannot join the WiFi" | Name, password, or a 5 GHz-only network |
| A node "in no room" | `sudo python3 tools/homekit gateway --reconfigure` and add it |
| Gateway does not start | `journalctl -u saans-gateway -n 50` names the setting (deploy/README.md) |

docs/HARDWARE.md §8 is the full list, in the order to check it.

## Changing things later

- Rooms, receivers or consent: `sudo python3 tools/homekit gateway --reconfigure`.
  The old config is kept beside the new one, and the password stays.
- The password: add `--new-password`.
- Alerts to the family, the long lie and signed frames are not set up by
  this tool yet: `rust/crates/saans-gateway/saans.example.toml` explains
  each, and their secrets go in `/etc/saans/saans.env`.
- Removing it: deploy/README.md, "Remove". Also delete
  `/etc/saans/selftest.token`.

## Hardware day checklist

The first time boards exist. The first session is the bench (#45,
bench/README.md): one TX and one receiver on a desk, raw CSI recorded.
The home kit comes after it.

**Before**

- [ ] 4 boards labelled 0, 1, 2, 3, and four data cables. Cable test: each board appears as `/dev/ttyUSB*` within a second
- [ ] Docker works without sudo (`docker run --rm hello-world`), and the IDF image is pulled: `docker pull espressif/idf:v5.4.1`
- [ ] You can open the serial port (`dialout` on Debian and Ubuntu, `uucp` on Arch)
- [ ] `python3 -m unittest tests.test_homekit` and `./edge/build.sh tests` pass
- [ ] The WiFi's channel is fixed in the router, and noted: ______
- [ ] The gateway machine's address on that WiFi, noted: ______
- [ ] `sudo python3 tools/homekit gateway --check`, then without `--check`; the dashboard opens

**Bench session (#45)**

- [ ] TX: `python3 tools/homekit node --role tx --id 0 --bench --channel N`
- [ ] Receiver: `python3 tools/homekit node --role rx --id 1 --bench --channel N --gateway <laptop> --ssid <wifi>`
- [ ] The node command printed PASS for both, and the TX's log showed `ESP-NOW rate: HT20 MCS0 LGI`
- [ ] Recording with `bench/session.sh` needs UDP 5566 to itself: `sudo systemctl stop saans-gateway` on the laptop first
- [ ] `python3 tools/homekit selftest --udp`: frames from node 1 at about 2 Hz, raw CSI at about 20 Hz
- [ ] Then bench/README.md from "Write down from the RX's `capture:` line"

**Home kit**

- [ ] Rebuild each board without `--bench` (raw CSI off): TX 0, receivers 1, 2, 3
- [ ] `sudo python3 tools/homekit gateway --reconfigure` if the rooms differ from what you set
- [ ] Boards in place; each room empty for 80 s after power-on
- [ ] `sudo python3 tools/homekit selftest`: PASS, every receiver named, none "in no room"
- [ ] The dashboard shows every room empty, then occupied when you walk in
- [ ] Write down each receiver's rate and loss from the self-test, labelled **REAL**

**First nights and the floor test (ADR-018, #46, #62)**

- [ ] Bedroom: one link across the bed. The TX on one side and a receiver on the other, both a hand's width above the mattress. Breathing switched on for that room in `tools/homekit gateway`. Positions noted: ______
- [ ] Three nights. Each morning write down, labelled **REAL**: the hours that gave a breathing rate, and the night's median
- [ ] A reference for each night: breaths counted over 60 s, five times, while lying still and awake; and a watch's sleeping breathing rate if you have one
- [ ] Floor test, for #62: two receivers on one wall, one 15 cm above the floor and one at 1.2 m, set up as two rooms named `low` and `high` so that each shows its own breathing. The TX on the opposite wall at knee height. Three minutes each: lying on the floor, sitting on a chair, lying on the bed
- [ ] For each posture write down which receiver reported a breathing rate, and its confidence. The question is whether `low` carries the breathing on the floor and `high` on the chair

## What has been checked, and what waits for hardware

Checked without boards (`tests/test_homekit.py`, 2026-11-07):

- The config the tool writes is valid TOML, has every room, has consent only
  where named, and holds no secret: the password is stored as its Argon2 hash,
  and the self-test token only as its SHA-256.
- The installer, run into a temporary root with a stand-in binary and
  systemd: the files deploy/README.md lists, with their modes; a second
  run writes nothing and restarts nothing; a new binary restarts; `--check`
  writes nothing; `--reconfigure` keeps the password and the old file;
  `/etc/saans/saans.env` is never overwritten; a config `--check` refuses
  stops the install before the service starts.
- edge/build.sh writes a board's node id, channel and WiFi values into its
  git-ignored build directory, mode 0600, and refuses an id or channel
  outside what the firmware accepts. The password is in no output and
  in no docker argument.
- Serial devices: a bridge is told from the native socket, a missing
  device is reported as a cable problem, and two boards make it ask which.
- The self-test, on a log in the firmware's own words: passes a healthy
  receiver, fails a slow or warning one, and never prints the network's
  name. It counts per-node rates from the gateway's `/api/links` and from
  UDP.

Waits for real boards:

- Flashing, and esptool's answer to "which chip", on a real S3 and on a
  non-S3 board.
- Which device node each socket of the devkits you bought appears as.
- Whether opening the serial port resets the board (the log reader works
  either way).
- The node's log lines as a board prints them, and every rate above.
- The installer on a fresh Raspberry Pi: the build time there, and the
  service under real systemd as root (deploy/README.md says the unit has
  only been run as a user service).
