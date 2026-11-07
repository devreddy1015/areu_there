"""The gateway config for one home: rooms, consent per room, the admin.

Everything here is pure: it turns answers into the text of
/etc/saans/saans.toml and reads back what an earlier run wrote. Writing
the file, hashing the password and asking the questions are gateway.py's.

What the file never holds is a secret (CONTRIBUTING, "Secrets"). The
admin's password is stored as the Argon2 hash `saans-gateway
--hash-password` prints, and the self-test token as its SHA-256; the
token itself lives in a root-only file beside the config.
"""

from __future__ import annotations

import ipaddress
import json
import re
from dataclasses import dataclass

# The firmware writes tx_id 255 on every frame: one TX beacon, not
# individually identified (edge/firmware/main/rx_role.cpp, HARDWARE.md §1).
TX_ANY = 255
# Node ids the firmware accepts (Kconfig.projbuild: range 0 254). 0 is the
# TX by convention (HARDWARE.md §5), so a room's receivers are 1..254.
RX_MIN, RX_MAX = 1, 254
ROOM_ID = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
USER_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,31}$")
SELFTEST_TOKEN_NAME = "homekit-selftest"
HTTP_PORT = 8080
UDP_PORT = 5566
DATA_DIR = "/var/lib/saans"


class ConfigError(ValueError):
    """An answer the gateway would refuse, said before it gets there."""


@dataclass(frozen=True)
class Room:
    id: str
    name: str
    rx: tuple[int, ...]
    vitals: bool = False


@dataclass(frozen=True)
class Site:
    rooms: tuple[Room, ...]
    admin: str
    admin_hash: str
    token_sha256: str
    lan: bool = True
    utc_offset_minutes: int = 0


def room_name(room_id: str) -> str:
    return room_id.replace("-", " ").capitalize()


def parse_room(spec: str) -> Room:
    """`ID[:Name]=RX[,RX...]`, e.g. `bedroom=1,2` or `living:Living room=3`."""
    head, sep, nodes = spec.partition("=")
    if not sep:
        raise ConfigError(f"room {spec!r}: expected ID=RX[,RX...], e.g. bedroom=1,2")
    room_id, _, name = head.partition(":")
    room_id = room_id.strip()
    if not ROOM_ID.match(room_id):
        raise ConfigError(
            f"room id {room_id!r}: lower case letters, digits and -, up to 32 (e.g. bedroom)"
        )
    return Room(room_id, name.strip() or room_name(room_id), parse_nodes(nodes))


def parse_nodes(text: str) -> tuple[int, ...]:
    out = []
    for part in text.replace(" ", "").split(","):
        if not part.isdigit() or not RX_MIN <= int(part) <= RX_MAX:
            raise ConfigError(
                f"receiver node {part!r}: a number from {RX_MIN} to {RX_MAX} "
                "(0 is the TX, 255 is reserved)"
            )
        out.append(int(part))
    if len(set(out)) != len(out):
        raise ConfigError(f"receiver nodes {text!r}: a node is listed twice")
    return tuple(out)


def with_consent(rooms: list[Room], consented: list[str]) -> tuple[Room, ...]:
    """The rooms, with vitals on exactly where `consented` names them (ADR-005)."""
    unknown = sorted(set(consented) - {r.id for r in rooms})
    if unknown:
        raise ConfigError(f"--vitals names no room: {', '.join(unknown)}")
    return tuple(Room(r.id, r.name, r.rx, r.id in consented) for r in rooms)


def check_rooms(rooms: tuple[Room, ...]) -> None:
    """What `saans-gateway --check` would refuse, said in this tool's words."""
    if not rooms:
        raise ConfigError("a home needs at least one room")
    seen: set[str] = set()
    owner: dict[int, str] = {}
    for r in rooms:
        if r.id in seen:
            raise ConfigError(f"room {r.id!r} appears twice")
        seen.add(r.id)
        for n in r.rx:
            if n in owner:
                raise ConfigError(
                    f"receiver node {n} is in both {owner[n]!r} and {r.id!r}: "
                    "one board senses one room"
                )
            owner[n] = r.id


def check_admin(name: str) -> None:
    if not USER_NAME.match(name):
        raise ConfigError(f"admin name {name!r}: letters, digits, . _ -, up to 32")


def _s(value: str) -> str:
    """A TOML basic string. JSON's escapes are a subset of TOML's."""
    return json.dumps(value)


def render(site: Site) -> str:
    """The whole saans.toml. Deterministic: the same answers give the same
    bytes, so a second run can tell that nothing changed."""
    check_rooms(site.rooms)
    check_admin(site.admin)
    if not site.admin_hash.startswith("$argon2"):
        raise ConfigError("the admin password hash must be an Argon2 PHC string")
    if not re.fullmatch(r"[0-9a-f]{64}", site.token_sha256):
        raise ConfigError("the self-test token hash must be 64 lowercase hex digits")
    host = "0.0.0.0" if site.lan else "127.0.0.1"
    lines = [
        "# Saans gateway for one home. Written by tools/homekit (docs/HOME_KIT.md);",
        "# change the rooms with: sudo python3 tools/homekit gateway --reconfigure",
        "# Every key is explained in rust/crates/saans-gateway/saans.example.toml.",
        "# No secret goes in this file: a setting names an environment variable,",
        "# and its value goes in /etc/saans/saans.env (deploy/README.md).",
        "",
        f"udp_port = {UDP_PORT}",
        "# The boards reach the gateway over the home WiFi.",
        'udp_bind = "0.0.0.0"',
    ]
    if site.lan:
        lines += [
            "# Reachable from the home network, so sign-in is required (README,",
            '# "Who may see what"). For this machine only, use "127.0.0.1:8080".',
        ]
    lines += [
        f"http_addr = {_s(f'{host}:{HTTP_PORT}')}",
        "# The one directory the service may write (deploy/systemd).",
        f"data_dir = {_s(DATA_DIR)}",
        "event_retention_days = 30",
        "history_retention_days = 90",
        "# This machine's offset from UTC when the kit was set up.",
        f"utc_offset_minutes = {site.utc_offset_minutes}",
        "",
        "[[user]]",
        f"name = {_s(site.admin)}",
        'role = "admin"',
        f"password_hash = {_s(site.admin_hash)}",
        "",
        "# For `tools/homekit selftest`: it reads /api/links with this token,",
        "# kept in /etc/saans/selftest.token (root only). Delete both to retire it.",
        "[[token]]",
        f"name = {_s(SELFTEST_TOKEN_NAME)}",
        'role = "ops"',
        f"sha256 = {_s(site.token_sha256)}",
    ]
    for r in site.rooms:
        links = ", ".join(f"{{ rx = {n}, tx = {TX_ANY} }}" for n in r.rx)
        lines += ["", "[[space]]", f"id = {_s(r.id)}", f"name = {_s(r.name)}", f"links = [{links}]"]
        if r.vitals:
            lines.append("# ADR-005: on because the person sensed here consented in writing.")
        else:
            lines.append("# ADR-005: breathing only with the written consent of who is sensed here.")
        lines.append(f"vitals = {'true' if r.vitals else 'false'}")
    lines += [
        "",
        "# Alerts to the family (Telegram, webhook), the long lie (long_lie_min)",
        "# and signed frames are off until set up: see saans.example.toml and",
        "# docs/ELDER_CARE.md. Their secrets go in /etc/saans/saans.env.",
        "",
    ]
    return "\n".join(lines)


def read_existing(text: str) -> dict:
    """What an earlier run left that this one should keep: each user's
    hash by name, the rooms, and whether it listened on the network.
    A file this cannot parse keeps nothing, and the questions are asked."""
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        return {}
    try:
        doc = tomllib.loads(text)
    except tomllib.TOMLDecodeError:
        return {}
    users = {
        u.get("name"): u.get("password_hash", "") for u in doc.get("user", []) if isinstance(u, dict)
    }
    rooms = []
    for sp in doc.get("space", []):
        try:
            rx = tuple(int(link["rx"]) for link in sp.get("links", []))
            rooms.append(Room(sp["id"], sp.get("name", sp["id"]), rx, bool(sp.get("vitals"))))
        except (KeyError, TypeError, ValueError):
            continue
    http = str(doc.get("http_addr", ""))
    return {"users": users, "rooms": rooms, "lan": bool(http) and not _loopback(http)}


def _loopback(addr: str) -> bool:
    host = addr.rsplit(":", 1)[0].strip("[]")
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return host == "localhost"
