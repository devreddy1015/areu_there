"""`homekit node`: one board, from the box to flashed and checked.

It is edge/build.sh with the questions asked and the traps checked:

    python3 tools/homekit node --role tx --id 0
    python3 tools/homekit node --role rx --id 1 --gateway 192.168.1.20 --ssid HomeWiFi
    python3 tools/homekit node --role rx --id 1 ... --dry-run    # no board touched

The WiFi password is asked for (or read from SAANS_WIFI_PASSWORD), never
taken on the command line, and reaches build.sh only through its
environment. build.sh writes it to the git-ignored build-<profile>/,
readable by you alone; nothing here prints it, or the network's name.
"""

from __future__ import annotations

import getpass
import ipaddress
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from homekit import ports, selftest

REPO = Path(__file__).resolve().parents[2]
BUILD_SH = REPO / "edge/build.sh"
TARGET = "esp32s3"
CHIP = re.compile(r"(?:Chip is|Detecting chip type\.\.\.|Chip type:)\s*(ESP32[-\w]*)")
PLUG_WAIT_S = 15.0


class NodeError(ValueError):
    pass


@dataclass(frozen=True)
class Plan:
    role: str
    node: int
    profile: str
    channel: int
    gateway: str | None
    vitals: bool
    bench: bool


def plan(role: str, node: int, channel: int, gateway: str | None,
         vitals: bool = False, bench: bool = False) -> Plan:
    """Check the answers and choose the profile in edge/firmware/profiles/."""
    target = os.environ.get("TARGET", TARGET)
    if target != TARGET:
        raise NodeError(f"TARGET={target}: this firmware is for the ESP32-S3 only (docs/HARDWARE.md §2)")
    if role not in ("tx", "rx"):
        raise NodeError("--role is tx or rx")
    lo = 0 if role == "tx" else 1
    if not lo <= node <= 254:
        raise NodeError(f"--id for a{'n' if role == 'rx' else ''} {role.upper()} is {lo}..254 "
                        "(0 is the TX by convention, 255 is reserved)")
    if not 1 <= channel <= 13:
        raise NodeError("--channel is 1..13, the 2.4 GHz channel of your home WiFi")
    if role == "tx":
        if vitals:
            raise NodeError("--vitals is for a receiver; a TX senses nothing")
        return Plan(role, node, "bench-tx" if bench else "home-tx", channel, None, False, bench)
    if not gateway:
        raise NodeError("a receiver needs --gateway, the address of the machine running the gateway")
    try:
        if ipaddress.ip_address(gateway).version != 4:
            raise ValueError
    except ValueError:
        raise NodeError(f"--gateway {gateway!r}: an IPv4 address such as 192.168.1.20") from None
    if bench and vitals:
        raise NodeError("the bench profile keeps vitals off (docs/PROTOCOL_S5.md §2.1)")
    profile = "bench-rx" if bench else ("home-rx-vitals" if vitals else "home-rx")
    return Plan(role, node, profile, channel, gateway, vitals, bench)


def build_env(p: Plan, ssid: str | None, password: str | None) -> dict[str, str]:
    """The environment build.sh reads. Site values only for a receiver:
    a TX joins no network."""
    env = {k: v for k, v in os.environ.items()
           if k not in ("SAANS_WIFI_SSID", "SAANS_WIFI_PASSWORD", "SAANS_GATEWAY_IP")}
    env.update(PROFILE=p.profile, TARGET=TARGET, SAANS_NODE_ID=str(p.node),
               SAANS_WIFI_CHANNEL=str(p.channel))
    if p.role == "rx":
        env.update(SAANS_GATEWAY_IP=p.gateway or "", SAANS_WIFI_SSID=ssid or "",
                   SAANS_WIFI_PASSWORD=password or "")
    return env


def chip_of(output: str) -> str | None:
    m = CHIP.search(output)
    return m.group(1).upper() if m else None


def check_chip(dev: str) -> None:
    r = subprocess.run([str(BUILD_SH), "chip"], env={**os.environ, "PORT": dev},
                       capture_output=True, text=True, stdin=subprocess.DEVNULL)
    chip = chip_of(r.stdout + r.stderr)
    if chip is None:
        raise NodeError(
            f"could not talk to a chip on {dev}. Hold the board's BOOT button, tap RESET,\n"
            "  release BOOT, and run this again; if that fails, try another data cable."
        )
    if chip != "ESP32-S3":
        raise NodeError(f"the board on {dev} is an {chip}, not an ESP32-S3: this firmware "
                        "is built for the S3 only (docs/HARDWARE.md §2)")
    print(f"  [  ok  ] {dev}: an ESP32-S3")


def wifi_channel_of(ssid: str) -> int | None:
    """The home network's channel as this machine sees it, if it can."""
    if not shutil.which("nmcli"):
        return None
    try:
        r = subprocess.run(["nmcli", "-t", "-f", "SSID,CHAN", "dev", "wifi", "list"],
                           capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired):
        return None
    for line in r.stdout.splitlines():
        name, _, chan = line.rpartition(":")
        if name.replace("\\:", ":") == ssid and chan.isdigit():
            return int(chan)
    return None


def find_port(wanted: str | None, interactive: bool) -> str:
    found = ports.list_ports()
    if not found and not wanted and interactive:
        input("  Plug the board in now (UART socket, data cable), then press Enter ")
        found = ports.wait_for_new([], PLUG_WAIT_S)
    for p in found:
        print(f"  [ port ] {p.describe()}")
    for line in ports.advice(found):
        print(f"  [ note ] {line}")
    chosen = ports.choose(found, wanted)
    if chosen is None:
        raise NodeError("which board? give --port, e.g. --port /dev/ttyUSB0" if found
                        else "no board found (see the note above)")
    problem = ports.access_problem(chosen.dev)
    if problem:
        raise NodeError(problem)
    return chosen.dev


def ask_site(args, p: Plan, interactive: bool) -> tuple[str | None, str | None]:
    if p.role == "tx":
        return None, None
    ssid = args.ssid or os.environ.get("SAANS_WIFI_SSID")
    if not ssid and interactive:
        ssid = input("  Home WiFi name (2.4 GHz network the boards join): ").strip()
    if not ssid:
        raise NodeError("a receiver needs the WiFi name: --ssid NAME")
    password = os.environ.get("SAANS_WIFI_PASSWORD")
    if password is None:
        if not interactive:
            raise NodeError("set SAANS_WIFI_PASSWORD, or run in a terminal to be asked")
        password = getpass.getpass("  WiFi password (not shown, not saved in the repository): ")
    chan = wifi_channel_of(ssid)
    if chan is not None and chan != p.channel:
        raise NodeError(f"your WiFi is on channel {chan}, the board would sense on {p.channel}: "
                        f"give --channel {chan} to every board (docs/HARDWARE.md, 'The uplink channel')")
    return ssid, password


def confirm_vitals(p: Plan, assume_yes: bool) -> None:
    if not p.vitals or assume_yes:
        return
    print("  This receiver will measure breathing. That is health data (ADR-005): only")
    print("  with the WRITTEN consent of the person in its room, and the gateway's room")
    print("  must have vitals on too.")
    if input("  Written consent is in hand? type yes: ").strip().lower() != "yes":
        raise NodeError("stopped: no consent, no vitals. Build without --vitals.")


def run(args) -> int:
    interactive = sys.stdin.isatty() and not args.yes
    try:
        p = plan(args.role, args.id, args.channel, args.gateway, args.vitals, args.bench)
        print(f"Saans node {p.node}, {p.role.upper()}, profile {p.profile}, channel {p.channel}"
              + (f", sending to {p.gateway}:5566" if p.gateway else "")
              + (" (dry run: nothing is built or flashed)" if args.dry_run else ""))
        if not shutil.which("docker"):
            raise NodeError("Docker is needed to build the firmware (edge/README.md); install it first")
        confirm_vitals(p, args.yes)
        ssid, password = ask_site(args, p, interactive)
        dev = None
        if not args.no_flash:
            try:
                dev = find_port(args.port, interactive and not args.dry_run)
            except NodeError as e:
                if not args.dry_run:
                    raise
                print(f"  [ note ] {e}; a real run stops here")
            if dev and not args.dry_run:
                check_chip(dev)
    except NodeError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    if p.role == "rx":
        print("  [ wifi ] network name and password set (neither is printed)")
    if args.dry_run:
        print(f"  [would ] PROFILE={p.profile} edge/build.sh build, with node id {p.node}")
        if dev:
            print(f"  [would ] PROFILE={p.profile} PORT={dev} edge/build.sh flash")
        return 0
    env = build_env(p, ssid, password)
    if subprocess.run([str(BUILD_SH), "build"], env=env).returncode != 0:
        print("error: the firmware build failed; the output above says why", file=sys.stderr)
        return 1
    if args.no_flash:
        return 0
    if subprocess.run([str(BUILD_SH), "flash"], env={**env, "PORT": dev},
                      stdin=subprocess.DEVNULL).returncode != 0:
        print("error: flashing failed. Hold BOOT, tap RESET, release BOOT, then run this again.",
              file=sys.stderr)
        return 1
    print(f"  [flashed] node {p.node} ({p.role.upper()}) on {dev}")
    if p.role == "rx":
        print("  Keep the room EMPTY for the next ~80 s: the receiver is learning it.")
    if args.no_selftest:
        return 0
    return selftest.from_serial(dev, 75)
