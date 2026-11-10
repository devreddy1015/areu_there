"""Serial ports: which board is plugged in, through which socket, and
reading its log. Linux only, stdlib only (sysfs and termios), so there
is nothing to install beside Docker.

The two traps it exists for are docs/HARDWARE.md §3: a charge-only cable
powers the board and no device appears; and many ESP32-S3 devkits have
two USB sockets, of which everything here is written for the one wired
to the USB-to-UART bridge (/dev/ttyUSB*).
"""

from __future__ import annotations

import glob
import grp
import os
import select
import termios
import time
import tty
from dataclasses import dataclass
from pathlib import Path

# USB vendor ids of the bridges ESP32-S3 devkits carry, and Espressif's own.
BRIDGES = {
    "10c4": "Silicon Labs CP210x USB-to-UART bridge",
    "1a86": "WCH CH34x USB-to-UART bridge",
    "0403": "FTDI USB-to-UART bridge",
}
ESPRESSIF = "303a"  # the S3's native USB (USB-Serial/JTAG): the other socket
BAUD = termios.B115200  # the ESP-IDF console's default


@dataclass(frozen=True)
class Port:
    dev: str
    vid: str = ""
    pid: str = ""
    product: str = ""

    @property
    def kind(self) -> str:
        if self.vid == ESPRESSIF:
            return "native"
        if self.vid in BRIDGES:
            return "bridge"
        return "unknown"

    def describe(self) -> str:
        what = {
            "native": "the chip's native USB socket",
            "bridge": BRIDGES.get(self.vid, ""),
        }.get(self.kind, self.product or "an unrecognised USB serial device")
        ids = f" ({self.vid}:{self.pid})" if self.vid else ""
        return f"{self.dev}: {what}{ids}"


def _usb_attr(dev_name: str, attr: str, sys_root: str) -> str:
    """Walk up from the tty's device to the USB device that has `attr`."""
    node = Path(sys_root, "class/tty", dev_name, "device")
    try:
        node = node.resolve()
    except OSError:
        return ""
    for _ in range(4):
        f = node / attr
        if f.exists():
            try:
                return f.read_text().strip()
            except OSError:
                return ""
        node = node.parent
    return ""


def list_ports(dev_root: str = "/dev", sys_root: str = "/sys") -> list[Port]:
    out = []
    for dev in sorted(glob.glob(f"{dev_root}/ttyUSB*") + glob.glob(f"{dev_root}/ttyACM*")):
        name = os.path.basename(dev)
        out.append(Port(dev, _usb_attr(name, "idVendor", sys_root).lower(),
                        _usb_attr(name, "idProduct", sys_root).lower(),
                        _usb_attr(name, "product", sys_root)))
    return out


def advice(ports: list[Port]) -> list[str]:
    """What to tell the person about the ports they have, before flashing."""
    if not ports:
        return [
            "No serial device. If the board's light is on, the cable is almost certainly",
            "charge-only: it carries power and no data (docs/HARDWARE.md §3). Change the",
            "cable before changing anything else.",
        ]
    notes = []
    if all(p.kind == "native" for p in ports):
        notes += [
            "Only the chip's native USB socket is connected. Many S3 devkits have two",
            "sockets; this kit is written for the one marked UART (or COM), which shows",
            "up as /dev/ttyUSB*. The native socket has not been tried with this build",
            "(docs/HARDWARE.md §3): move the cable to the other socket if the board has one.",
        ]
    if len(ports) > 1:
        notes.append("More than one serial device: unplug every board but the one to flash, or give --port.")
    return notes


def choose(ports: list[Port], wanted: str | None) -> Port | None:
    if wanted:
        return next((p for p in ports if p.dev == wanted), Port(wanted))
    bridges = [p for p in ports if p.kind == "bridge"]
    if len(ports) == 1:
        return ports[0]
    if len(bridges) == 1:
        return bridges[0]
    return None


def wait_for_new(before: list[Port], seconds: float, lister=list_ports) -> list[Port]:
    """Ports that appear within `seconds` of asking for the board. A data
    cable enumerates in well under a second."""
    seen = {p.dev for p in before}
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        new = [p for p in lister() if p.dev not in seen]
        if new:
            time.sleep(0.5)  # let a second interface of the same board appear
            return [p for p in lister() if p.dev not in seen]
        time.sleep(0.2)
    return []


def access_problem(dev: str) -> str | None:
    if not os.path.exists(dev):
        return f"{dev} does not exist"
    if os.access(dev, os.R_OK | os.W_OK):
        return None
    try:
        group = grp.getgrgid(os.stat(dev).st_gid).gr_name
    except (KeyError, OSError):
        group = "dialout"
    return (f"you cannot open {dev}: add yourself to its group and log in again:\n"
            f"  sudo usermod -aG {group} $USER")


def read_lines(dev: str, seconds: float, on_line) -> None:
    """Feed each line the board prints for `seconds` to `on_line`. Raw
    mode at 115200, no echo. Opening the port asserts DTR and RTS
    together, which on the usual devkit auto-reset circuit does not reset
    the chip; on a board where it does, the boot banner simply appears."""
    fd = os.open(dev, os.O_RDONLY | os.O_NOCTTY | os.O_NONBLOCK)
    try:
        tty.setraw(fd)
        attrs = termios.tcgetattr(fd)
        attrs[4] = attrs[5] = BAUD
        termios.tcsetattr(fd, termios.TCSANOW, attrs)
        buf = b""
        deadline = time.monotonic() + seconds
        while (left := deadline - time.monotonic()) > 0:
            ready, _, _ = select.select([fd], [], [], min(left, 0.5))
            if not ready:
                continue
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                continue
            if not chunk:
                break
            buf += chunk
            *lines, buf = buf.split(b"\n")
            for raw in lines:
                if on_line(raw.decode("utf-8", "replace").rstrip("\r")) is False:
                    return
    finally:
        os.close(fd)
