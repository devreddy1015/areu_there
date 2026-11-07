"""`homekit gateway`: a fresh Debian or Raspberry Pi OS machine to a
running gateway for one home. It is deploy/README.md's install, done by
a program: the same files, the same paths, the same unit.

Every step looks before it acts, so a second run changes nothing and
restarts nothing. `--check` prints what would change and changes nothing.

    sudo python3 tools/homekit gateway                       # asks
    sudo python3 tools/homekit gateway --room bedroom=1,2 --room living=3 \\
        --vitals bedroom --admin-password-file pw.txt         # does not ask
    sudo python3 tools/homekit gateway --check               # dry run
"""

from __future__ import annotations

import getpass
import hashlib
import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

from homekit import siteconfig
from homekit.siteconfig import ConfigError, Room, Site

REPO = Path(__file__).resolve().parents[2]
BIN = "/usr/local/bin/saans-gateway"
ETC = "/etc/saans"
CONFIG = f"{ETC}/saans.toml"
ENV_FILE = f"{ETC}/saans.env"
TOKEN_FILE = f"{ETC}/selftest.token"
UNIT = "/etc/systemd/system/saans-gateway.service"
UNIT_SRC = REPO / "deploy/systemd/saans-gateway.service"
ENV_SRC = REPO / "deploy/systemd/saans-gateway.env.example"
# The Rust build when this machine has no cargo: the toolchain in a pinned
# container, as edge/build.sh does for the firmware. Debian bookworm, so the
# binary runs on Raspberry Pi OS and Debian 12 or later. rust-toolchain.toml
# says "stable", which would make rustup download another toolchain inside
# the container; RUSTUP_TOOLCHAIN keeps the image's own.
RUST_IMAGE = "rust:1.88.0-bookworm"
RUST_IMAGE_TOOLCHAIN = "1.88.0"
ADMIN_MIN_CHARS = 10


def say(tag: str, msg: str) -> None:
    print(f"  [{tag:^6}] {msg}", flush=True)


@dataclass
class Host:
    """The machine being set up. `root` prefixes every system path, which
    is how the tests install into a temporary directory."""

    root: Path = Path("/")
    dry: bool = False
    changed: list[str] = field(default_factory=list)
    last_status: int = 0

    def p(self, path: str) -> Path:
        return self.root / path.lstrip("/")

    def read(self, path: str) -> bytes | None:
        try:
            return self.p(path).read_bytes()
        except FileNotFoundError:
            return None

    def put(self, path: str, data: bytes, mode: int, what: str) -> bool:
        """Write `data` unless it is already there, atomically. True if
        it changed (or, dry, would have)."""
        target = self.p(path)
        if self.read(path) == data:
            if target.stat().st_mode & 0o777 != mode and not self.dry:
                os.chmod(target, mode)
                say("fixed", f"{what}: mode {oct(mode)}")
            else:
                say("same", f"{what}: {path}")
            return False
        say("would" if self.dry else "write", f"{what}: {path}")
        if not self.dry:
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp = target.with_name(f".{target.name}.homekit")
            fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, mode)
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.chmod(tmp, mode)
            os.replace(tmp, target)
        self.changed.append(path)
        return True

    def run(self, argv: list[str], **kw) -> subprocess.CompletedProcess:
        return subprocess.run(argv, **kw)

    def systemctl(self, *args: str) -> subprocess.CompletedProcess:
        return self.run(["systemctl", *args], capture_output=True, text=True)

    def is_systemd(self) -> bool:
        return Path("/run/systemd/system").is_dir() and shutil.which("systemctl") is not None

    def http_json(self, url: str, token: str | None) -> dict | list | None:
        """The JSON at `url`, or None. `last_status` keeps the HTTP status
        of a refusal (401: sign-in needed), 0 when nothing answered."""
        req = urllib.request.Request(url)
        if token:
            req.add_header("Authorization", f"Bearer {token}")
        self.last_status = 0
        try:
            with urllib.request.urlopen(req, timeout=3) as r:
                self.last_status = r.status
                return json.load(r)
        except urllib.error.HTTPError as e:
            self.last_status = e.code
            return None
        except (urllib.error.URLError, OSError, ValueError):
            return None


# ------------------------------------------------------------------ steps


def preflight(host: Host) -> None:
    if sys.platform != "linux":
        raise SystemExit("error: the gateway service needs Linux with systemd")
    if not host.dry and host.root == Path("/") and os.geteuid() != 0:
        raise SystemExit("error: run with sudo (or add --check to see what it would do)")
    if host.root == Path("/") and not host.is_systemd():
        raise SystemExit("error: systemd is not running here; deploy/README.md needs it")
    if not Path("/etc/debian_version").exists():
        say("note", "not Debian or Raspberry Pi OS: the steps are the same, untried here")


def build(host: Host, binary: str | None) -> Path:
    """The binary to install: --binary, or built from this checkout."""
    if binary:
        path = Path(binary).resolve()
        if not os.access(path, os.X_OK):
            raise SystemExit(f"error: --binary {binary}: not an executable file")
        return path
    out = REPO / "rust/target/release/saans-gateway"
    user = os.environ.get("SUDO_USER") if os.geteuid() == 0 else None
    as_user = ["sudo", "-u", user, "-H"] if user and user != "root" else []
    cargo = _cargo(user)
    if cargo:
        argv = [*as_user, cargo, "build", "--release", "--locked", "-p", "saans-gateway"]
        how = "cargo on this machine"
    elif shutil.which("docker"):
        uid, gid = _owner(REPO)
        argv = [
            "docker", "run", "--rm", "--user", f"{uid}:{gid}",
            "-v", f"{REPO}:/src", "-w", "/src/rust",
            "-e", "CARGO_HOME=/src/rust/target/cargo-home",
            "-e", f"RUSTUP_TOOLCHAIN={RUST_IMAGE_TOOLCHAIN}",
            RUST_IMAGE, "cargo", "build", "--release", "--locked", "-p", "saans-gateway",
        ]
        how = f"cargo in {RUST_IMAGE}"
    else:
        raise SystemExit(
            "error: no cargo and no docker to build the gateway with. Either install\n"
            "  Rust (https://rustup.rs, as your own user, not root), or Docker, or\n"
            "  build it elsewhere for this machine and pass --binary PATH."
        )
    if host.dry:
        say("would", f"build saans-gateway with {how}")
        return out
    say("build", f"saans-gateway with {how}; the first build on a Pi takes a while")
    if host.run(argv, cwd=REPO / "rust").returncode != 0 or not out.exists():
        raise SystemExit("error: the gateway build failed; the output above says why")
    return out


def _cargo(user: str | None) -> str | None:
    if user:
        home = Path(os.path.expanduser(f"~{user}"))
        if (home / ".cargo/bin/cargo").exists():
            return str(home / ".cargo/bin/cargo")
    return shutil.which("cargo")


def _owner(path: Path) -> tuple[int, int]:
    st = path.stat()
    return st.st_uid, st.st_gid


def install_files(host: Host, built: Path) -> None:
    data = built.read_bytes() if built.exists() else b""
    if host.dry and not data:
        say("would", f"install the new build as {BIN}")
        host.changed.append(BIN)
    else:
        host.put(BIN, data, 0o755, "gateway binary")
    if not host.dry:
        host.p(ETC).mkdir(parents=True, exist_ok=True)
        os.chmod(host.p(ETC), 0o755)
    if host.read(ENV_FILE) is None:
        host.put(ENV_FILE, ENV_SRC.read_bytes(), 0o600, "secrets file (empty until you need one)")
    else:
        say("kept", f"secrets file: {ENV_FILE} (yours; never overwritten)")
        if not host.dry:
            os.chmod(host.p(ENV_FILE), 0o600)
    host.put(UNIT, UNIT_SRC.read_bytes(), 0o644, "systemd unit")


@dataclass
class Answers:
    rooms: tuple[Room, ...]
    consent: tuple[str, ...]
    admin: str
    password: str | None
    lan: bool


def gather(args, existing: dict, interactive: bool) -> Answers:
    rooms = [siteconfig.parse_room(r) for r in args.room]
    consent = list(args.vitals)
    if not rooms and interactive:
        rooms, consent = _ask_rooms(existing.get("rooms") or [])
    if not rooms:
        raise ConfigError("no rooms: give --room ID=RX[,RX...] (e.g. --room bedroom=1,2)")
    rooms_c = siteconfig.with_consent(rooms, consent)
    siteconfig.check_rooms(rooms_c)
    siteconfig.check_admin(args.admin)
    password = None
    if args.admin_password_file:
        lines = Path(args.admin_password_file).read_text().splitlines()
        password = lines[0] if lines else ""
    elif not existing.get("users", {}).get(args.admin) or args.new_password:
        if not interactive:
            raise ConfigError(f"a password for {args.admin!r}: give --admin-password-file FILE")
        password = _ask_password(args.admin)
    if password is not None and len(password) < ADMIN_MIN_CHARS:
        raise ConfigError(f"the admin password must be at least {ADMIN_MIN_CHARS} characters")
    return Answers(rooms_c, tuple(consent), args.admin, password, not args.local_only)


def _ask(q: str, default: str = "") -> str:
    a = input(f"  {q}{f' [{default}]' if default else ''}: ").strip()
    return a or default


def _ask_rooms(before: list[Room]) -> tuple[list[Room], list[str]]:
    print("\nThe rooms. Each room has the receiver boards (RX) that watch it, by node")
    print("number: 1, 2, 3 ... The TX board is node 0 and is not listed.")
    print("docs/ELDER_CARE.md §2: bathroom and bedroom first, two receivers each.")
    default_n = str(len(before) or 2)
    n = int(_ask("How many rooms", default_n) or default_n)
    rooms, consent, next_node = [], [], 1
    for i in range(n):
        old = before[i] if i < len(before) else None
        rid = _ask(f"Room {i + 1} id (lower case, e.g. bedroom)", old.id if old else "")
        default_rx = ",".join(map(str, old.rx)) if old else f"{next_node},{next_node + 1}"
        rx = _ask(f"  receiver nodes in {rid}", default_rx)
        room = siteconfig.parse_room(f"{rid}={rx}")
        next_node = max(room.rx) + 1
        print("  Breathing (vitals) is health data. Only turn it on if the person who")
        print("  is sensed in this room has agreed IN WRITING (ADR-005).")
        if _ask(f"  {room.name}: written consent for breathing? y/N", "y" if old and old.vitals else "n").lower() == "y":
            consent.append(room.id)
        rooms.append(room)
    return rooms, consent


def _ask_password(admin: str) -> str:
    # Says how long, never echoes what was typed (getpass does not echo).
    print(f"\nThe dashboard's admin, {admin!r}: choose a password (at least {ADMIN_MIN_CHARS} characters).")
    while True:
        a = getpass.getpass("  password: ")
        if a == getpass.getpass("  again:    "):
            return a
        print("  they differ; again")


def hash_password(host: Host, binary: Path, password: str) -> str:
    r = host.run([str(binary), "--hash-password"], input=password + "\n",
                 capture_output=True, text=True)
    out = r.stdout.strip()
    if r.returncode != 0 or not out.startswith("$argon2"):
        raise SystemExit("error: saans-gateway --hash-password failed")
    return out


def write_config(host: Host, args, binary: Path) -> None:
    current = host.read(CONFIG)
    existing = siteconfig.read_existing(current.decode()) if current else {}
    if current is not None and not args.reconfigure:
        say("kept", f"config: {CONFIG} (change the rooms with --reconfigure)")
        return
    interactive = sys.stdin.isatty() and not args.room
    answers = gather(args, existing, interactive)
    if answers.password is not None:
        pw_hash = "$argon2id$(would be hashed)" if host.dry else hash_password(host, binary, answers.password)
    else:
        pw_hash = existing["users"][answers.admin]
        say("kept", f"admin {answers.admin!r}: the password set before (--new-password to change it)")
    token = host.read(TOKEN_FILE)
    if token is None or len(token.strip()) < 32:
        token = secrets.token_hex(32).encode()
        host.put(TOKEN_FILE, token + b"\n", 0o600, "self-test token")
    site = Site(
        rooms=answers.rooms,
        admin=answers.admin,
        admin_hash=pw_hash,
        token_sha256=hashlib.sha256(token.strip()).hexdigest(),
        lan=answers.lan,
        utc_offset_minutes=args.utc_offset_minutes
        if args.utc_offset_minutes is not None
        else time.localtime().tm_gmtoff // 60,
    )
    text = siteconfig.render(site).encode()
    if current is not None and current != text and not host.dry:
        backup = f"{CONFIG}.{time.strftime('%Y%m%d-%H%M%S')}"
        host.put(backup, current, 0o644, "the config before this run")
    host.put(CONFIG, text, 0o644, "config")
    for r in answers.rooms:
        say("room", f"{r.id}: receivers {', '.join(map(str, r.rx))}; "
                    f"breathing {'ON (written consent)' if r.vitals else 'off'}")


def env_from_file(host: Host) -> dict[str, str]:
    """saans.env as the unit's EnvironmentFile= reads it, for `--check`."""
    env = dict(os.environ)
    for line in (host.read(ENV_FILE) or b"").decode().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip().strip('"')
    return env


def validate(host: Host) -> None:
    if host.dry and host.read(CONFIG) is None:
        say("would", "check the config with saans-gateway --check")
        return
    if host.dry and BIN in host.changed:
        say("would", "check the config with the new binary")
        return
    r = host.run([str(host.p(BIN)), "--config", str(host.p(CONFIG)), "--check"],
                 env=env_from_file(host), capture_output=True, text=True)
    if r.returncode != 0:
        sys.stderr.write(r.stdout + r.stderr)
        raise SystemExit(f"error: saans-gateway --check refused {CONFIG}; it says why above")
    say("ok", "saans-gateway --check")


def service(host: Host) -> None:
    unit_changed = UNIT in host.changed
    restart = any(p in host.changed for p in (BIN, CONFIG, ENV_FILE))
    if host.dry:
        if unit_changed:
            say("would", "systemctl daemon-reload")
        say("would", "systemctl enable --now saans-gateway" + (" and restart it" if restart else ""))
        return
    if unit_changed:
        host.systemctl("daemon-reload")
    if host.systemctl("is-enabled", "--quiet", "saans-gateway").returncode != 0:
        host.systemctl("enable", "saans-gateway")
        say("enable", "saans-gateway starts at boot")
    active = host.systemctl("is-active", "--quiet", "saans-gateway").returncode == 0
    if not active:
        host.systemctl("reset-failed", "saans-gateway")
        r = host.systemctl("start", "saans-gateway")
        say("start" if r.returncode == 0 else "FAILED", "saans-gateway")
    elif restart or unit_changed:
        host.systemctl("restart", "saans-gateway")
        say("restart", "saans-gateway: its binary, config or secrets changed")
    else:
        say("same", "saans-gateway is running; nothing changed, not restarted")


def wait_healthy(host: Host, seconds: float = 20.0) -> bool:
    token = (host.read(TOKEN_FILE) or b"").decode().strip() or None
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        h = host.http_json(f"http://127.0.0.1:{siteconfig.HTTP_PORT}/api/health", token)
        if h is not None:
            say("ok", f"gateway {h.get('version', '?')} answering, {h.get('spaces', '?')} rooms")
            return True
        time.sleep(1)
    say("FAILED", "no answer on /api/health; see: journalctl -u saans-gateway -n 50")
    return False


def addresses() -> list[str]:
    """This machine's address on the home network, found by asking the
    routing table (a UDP connect sends nothing)."""
    out = []
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))
            out.append(s.getsockname()[0])
    except OSError:
        pass
    return out


def run(args) -> int:
    host = Host(root=Path(args.root), dry=args.check)
    print(f"Saans home gateway{' (dry run: --check changes nothing)' if host.dry else ''}")
    preflight(host)
    try:
        built = build(host, args.binary)
        install_files(host, built)
        write_config(host, args, built if built.exists() else host.p(BIN))
    except ConfigError as e:
        raise SystemExit(f"error: {e}")
    validate(host)
    if args.no_start:
        say("skip", "service (--no-start)")
        return 0
    service(host)
    if host.dry:
        return 0
    ok = wait_healthy(host)
    lan = (siteconfig.read_existing((host.read(CONFIG) or b"").decode()) or {}).get("lan", True)
    print("\nThe dashboard:")
    if lan:
        for ip in addresses():
            print(f"  http://{ip}:{siteconfig.HTTP_PORT}/app   (from a phone or laptop on this WiFi)")
    print(f"  http://127.0.0.1:{siteconfig.HTTP_PORT}/app   (on this machine)")
    print("\nThe boards send to this machine at UDP 5566; give them this address:")
    print(f"  python3 tools/homekit node --role rx --id 1 --gateway {(addresses() or ['<this address>'])[0]}")
    print("Then, with boards running:  sudo python3 tools/homekit selftest")
    return 0 if ok else 1
