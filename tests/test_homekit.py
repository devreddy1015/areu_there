"""tools/homekit (docs/HOME_KIT.md): the parts that run without a board.

The gateway installer runs here into a temporary root, with a stand-in
binary and a stand-in systemctl, twice, to show the second run changes
nothing. edge/build.sh runs with a stand-in docker, to show what it writes
for a board's site values. What needs real boards is listed in
docs/HOME_KIT.md, "What has been checked".
"""

from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
import types
import unittest
import unittest.mock
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "tools"))
sys.path.insert(0, str(REPO))

from homekit import gateway, node, ports, selftest, siteconfig  # noqa: E402
from homekit.siteconfig import ConfigError, Room, Site  # noqa: E402

FAKE_HASH = "$argon2id$v=19$m=19456,t=2,p=1$ZmFrZXNhbHQ$ZmFrZWhhc2g"
# Obviously not anyone's: an answer file for the installer's prompt.
FAKE_ADMIN_ENTRY = "fake-entry-for-tests-only"

# A stand-in for saans-gateway: --hash-password prints a fixed PHC
# string; --check exits 0 when the config has a [[space]].
FAKE_BINARY = f"""#!/bin/sh
case "$*" in
  *--hash-password*) read -r pw; echo '{FAKE_HASH}' ;;
  *--check*) grep -q '^\\[\\[space\\]\\]' "$2" ;;
  *) exit 1 ;;
esac
"""


def site(**kw) -> Site:
    base = dict(
        rooms=(Room("bedroom", "Bedroom", (1, 2), True), Room("living", "Living", (3,))),
        admin="admin", admin_hash=FAKE_HASH, token_sha256="a" * 64, utc_offset_minutes=330,
    )
    base.update(kw)
    return Site(**base)


class SiteConfigTest(unittest.TestCase):
    def test_parses_a_room_with_and_without_a_name(self):
        self.assertEqual(siteconfig.parse_room("bedroom=1,2"), Room("bedroom", "Bedroom", (1, 2)))
        self.assertEqual(siteconfig.parse_room("liv:Living room=3"), Room("liv", "Living room", (3,)))

    def test_refuses_rooms_the_gateway_would_refuse(self):
        for bad in ["bedroom", "Bed Room=1", "bedroom=0", "bedroom=255", "bedroom=1,1", "bedroom=x"]:
            with self.subTest(bad), self.assertRaises(ConfigError):
                siteconfig.parse_room(bad)
        with self.assertRaisesRegex(ConfigError, "both"):
            siteconfig.check_rooms((Room("a", "A", (1,)), Room("b", "B", (1,))))
        with self.assertRaisesRegex(ConfigError, "twice"):
            siteconfig.check_rooms((Room("a", "A", (1,)), Room("a", "A", (2,))))

    def test_consent_is_exactly_where_named(self):
        rooms = [Room("bedroom", "B", (1,)), Room("bath", "Ba", (2,))]
        out = siteconfig.with_consent(rooms, ["bath"])
        self.assertEqual([r.vitals for r in out], [False, True])
        with self.assertRaisesRegex(ConfigError, "kitchen"):
            siteconfig.with_consent(rooms, ["kitchen"])

    def test_renders_valid_toml_with_every_room(self):
        doc = tomllib.loads(siteconfig.render(site()))
        self.assertEqual(doc["http_addr"], "0.0.0.0:8080")
        self.assertEqual(doc["data_dir"], "/var/lib/saans")
        self.assertEqual([s["id"] for s in doc["space"]], ["bedroom", "living"])
        self.assertEqual(doc["space"][0]["links"], [{"rx": 1, "tx": 255}, {"rx": 2, "tx": 255}])
        self.assertEqual([s["vitals"] for s in doc["space"]], [True, False])
        self.assertEqual(doc["user"], [{"name": "admin", "role": "admin", "password_hash": FAKE_HASH}])
        self.assertEqual(doc["token"][0]["role"], "ops")
        local = tomllib.loads(siteconfig.render(site(lan=False)))
        self.assertEqual(local["http_addr"], "127.0.0.1:8080")

    def test_render_is_deterministic_and_round_trips(self):
        text = siteconfig.render(site())
        self.assertEqual(text, siteconfig.render(site()))
        back = siteconfig.read_existing(text)
        self.assertEqual(back["users"], {"admin": FAKE_HASH})
        self.assertEqual(tuple(back["rooms"]), site().rooms)
        self.assertTrue(back["lan"])

    def test_names_that_need_escaping_stay_one_toml_string(self):
        r = Room("kitchen", 'Nani\'s "kitchen"\n[[user]]', (4,))
        doc = tomllib.loads(siteconfig.render(site(rooms=(r,))))
        self.assertEqual(doc["space"][0]["name"], r.name)
        self.assertEqual(len(doc["user"]), 1)

    def test_refuses_a_plain_password_where_the_hash_goes(self):
        with self.assertRaisesRegex(ConfigError, "Argon2"):
            siteconfig.render(site(admin_hash="hunter2"))


class FakeHost(gateway.Host):
    """systemd as a dict: enabled and active, and the calls recorded."""

    def __init__(self, root, dry=False):
        super().__init__(root=Path(root), dry=dry)
        self.calls: list[tuple[str, ...]] = []
        self.state = {"enabled": False, "active": False}

    def systemctl(self, *args):
        self.calls.append(args)
        rc = 0
        if args[0] == "is-enabled":
            rc = 0 if self.state["enabled"] else 1
        elif args[0] == "is-active":
            rc = 0 if self.state["active"] else 3
        elif args[0] == "enable":
            self.state["enabled"] = True
        elif args[0] in ("start", "restart"):
            self.state["active"] = True
        return subprocess.CompletedProcess(args, rc, "", "")


def install(host: FakeHost, binary: Path, **kw) -> str:
    """gateway.run's steps, without preflight and the health wait."""
    args = types.SimpleNamespace(
        room=["bedroom=1,2", "living=3"], vitals=["bedroom"], admin="admin",
        admin_password_file=kw.pop("answers_file"), new_password=False, reconfigure=False,
        local_only=False, utc_offset_minutes=330, binary=str(binary),
    )
    vars(args).update(kw)
    out = StringIO()
    host.changed = []
    with redirect_stdout(out):
        built = gateway.build(host, args.binary)
        gateway.install_files(host, built)
        gateway.write_config(host, args, built)
        gateway.validate(host)
        gateway.service(host)
    return out.getvalue()


def snapshot(root: Path) -> dict[str, tuple[bytes, int]]:
    return {
        str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mode & 0o777)
        for p in root.rglob("*") if p.is_file()
    }


class GatewayInstallTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        self.root = self.tmp / "root"
        self.binary = self.tmp / "saans-gateway"
        self.binary.write_text(FAKE_BINARY)
        self.binary.chmod(0o755)
        self.answers = self.tmp / "pw"
        self.answers.write_text(FAKE_ADMIN_ENTRY + "\n")

    def test_installs_the_deploy_files_and_starts_the_service(self):
        host = FakeHost(self.root)
        install(host, self.binary, answers_file=str(self.answers))
        files = snapshot(self.root)
        self.assertEqual(files["usr/local/bin/saans-gateway"][1], 0o755)
        self.assertEqual(files["etc/saans/saans.env"], (gateway.ENV_SRC.read_bytes(), 0o600))
        self.assertEqual(files["etc/saans/selftest.token"][1], 0o600)
        self.assertEqual(files["etc/systemd/system/saans-gateway.service"][0], gateway.UNIT_SRC.read_bytes())
        self.assertIn(("daemon-reload",), host.calls)
        self.assertIn(("start", "saans-gateway"), host.calls)
        self.assertTrue(host.state["enabled"])

    def test_no_secret_lands_in_the_config(self):
        host = FakeHost(self.root)
        install(host, self.binary, answers_file=str(self.answers))
        config = (self.root / "etc/saans/saans.toml").read_text()
        token = (self.root / "etc/saans/selftest.token").read_text().strip()
        self.assertNotIn(FAKE_ADMIN_ENTRY, config)
        self.assertNotIn(token, config)
        self.assertIn(FAKE_HASH, config)
        for data, _ in snapshot(self.root).values():
            self.assertNotIn(FAKE_ADMIN_ENTRY.encode(), data)

    def test_running_twice_changes_nothing_and_restarts_nothing(self):
        host = FakeHost(self.root)
        install(host, self.binary, answers_file=str(self.answers))
        before = snapshot(self.root)
        host.calls.clear()
        out = install(host, self.binary, answers_file=str(self.answers))
        self.assertEqual(snapshot(self.root), before)
        self.assertEqual(host.changed, [])
        self.assertNotIn(("restart", "saans-gateway"), host.calls)
        self.assertNotIn(("daemon-reload",), host.calls)
        self.assertIn("not restarted", out)

    def test_a_new_binary_restarts_the_service(self):
        host = FakeHost(self.root)
        install(host, self.binary, answers_file=str(self.answers))
        self.binary.write_text(FAKE_BINARY + "# rebuilt\n")
        host.calls.clear()
        install(host, self.binary, answers_file=str(self.answers))
        self.assertIn(("restart", "saans-gateway"), host.calls)

    def test_check_mode_writes_nothing(self):
        dry = FakeHost(self.root, dry=True)
        out = install(dry, self.binary, answers_file=str(self.answers))
        self.assertFalse(self.root.exists())
        self.assertEqual(dry.calls, [])
        self.assertIn("would", out)
        install(FakeHost(self.root), self.binary, answers_file=str(self.answers))
        before = snapshot(self.root)
        install(FakeHost(self.root, dry=True), self.binary, answers_file=str(self.answers))
        self.assertEqual(snapshot(self.root), before)

    def test_reconfigure_keeps_the_password_and_the_old_file(self):
        host = FakeHost(self.root)
        install(host, self.binary, answers_file=str(self.answers))
        first = (self.root / "etc/saans/saans.toml").read_text()
        install(host, self.binary, answers_file=None, reconfigure=True,
                room=["bedroom=1,2", "living=3", "kitchen=4"], vitals=[])
        doc = tomllib.loads((self.root / "etc/saans/saans.toml").read_text())
        self.assertEqual([s["id"] for s in doc["space"]], ["bedroom", "living", "kitchen"])
        self.assertFalse(any(s["vitals"] for s in doc["space"]))
        self.assertEqual(doc["user"][0]["password_hash"], FAKE_HASH)
        backups = list((self.root / "etc/saans").glob("saans.toml.*"))
        self.assertEqual([b.read_text() for b in backups], [first])

    def test_an_existing_config_is_kept_without_reconfigure(self):
        (self.root / "etc/saans").mkdir(parents=True)
        mine = '[[space]]\nid = "mine"\nlinks = []\n'
        (self.root / "etc/saans/saans.toml").write_text(mine)
        install(FakeHost(self.root), self.binary, answers_file=str(self.answers))
        self.assertEqual((self.root / "etc/saans/saans.toml").read_text(), mine)

    def test_the_secrets_file_is_never_overwritten(self):
        (self.root / "etc/saans").mkdir(parents=True)
        env = self.root / "etc/saans/saans.env"
        env.write_text("SAANS_TELEGRAM_TOKEN=x\n")
        env.chmod(0o644)
        install(FakeHost(self.root), self.binary, answers_file=str(self.answers))
        self.assertEqual(env.read_text(), "SAANS_TELEGRAM_TOKEN=x\n")
        self.assertEqual(env.stat().st_mode & 0o777, 0o600)

    def test_a_short_password_is_refused_before_anything_is_hashed(self):
        self.answers.write_text("short\n")
        with self.assertRaisesRegex(ConfigError, "at least"):
            install(FakeHost(self.root), self.binary, answers_file=str(self.answers))

    def test_a_config_the_gateway_refuses_stops_the_install(self):
        self.binary.write_text(FAKE_BINARY.replace("grep -q", "! grep -q"))
        host = FakeHost(self.root)
        with self.assertRaises(SystemExit), redirect_stdout(StringIO()):
            install(host, self.binary, answers_file=str(self.answers))
        self.assertNotIn(("start", "saans-gateway"), host.calls)


@unittest.skipUnless(os.environ.get("SAANS_GATEWAY_BIN"), "set SAANS_GATEWAY_BIN to a built saans-gateway")
class RealGatewayCheckTest(unittest.TestCase):
    """The config this writes, through the real binary's --hash-password
    and --check. Run with SAANS_GATEWAY_BIN=rust/target/release/saans-gateway."""

    def test_the_gateway_accepts_what_homekit_writes(self):
        binary = Path(os.environ["SAANS_GATEWAY_BIN"])
        h = gateway.hash_password(gateway.Host(), binary, FAKE_ADMIN_ENTRY)
        for lan in (True, False):
            with tempfile.NamedTemporaryFile("w", suffix=".toml") as f:
                f.write(siteconfig.render(site(admin_hash=h, lan=lan)))
                f.flush()
                r = subprocess.run([str(binary), "--config", f.name, "--check"],
                                   capture_output=True, text=True)
                self.assertEqual(r.returncode, 0, r.stdout + r.stderr)


class NodePlanTest(unittest.TestCase):
    def test_chooses_the_profile_from_role_consent_and_bench(self):
        self.assertEqual(node.plan("tx", 0, 6, None).profile, "home-tx")
        self.assertEqual(node.plan("rx", 1, 6, "192.168.1.20").profile, "home-rx")
        self.assertEqual(node.plan("rx", 1, 6, "192.168.1.20", vitals=True).profile, "home-rx-vitals")
        self.assertEqual(node.plan("rx", 1, 6, "192.168.1.20", bench=True).profile, "bench-rx")
        self.assertEqual(node.plan("tx", 0, 6, None, bench=True).profile, "bench-tx")
        for p in ["home-tx", "home-rx", "home-rx-vitals", "bench-rx", "bench-tx"]:
            self.assertTrue((REPO / f"edge/firmware/profiles/{p}.defaults").exists(), p)

    def test_the_vitals_profile_differs_from_home_rx_only_in_consent(self):
        def settings(name):
            text = (REPO / f"edge/firmware/profiles/{name}.defaults").read_text()
            return {l for l in text.splitlines() if l.startswith("CONFIG_")}
        self.assertEqual(settings("home-rx") ^ settings("home-rx-vitals"),
                         {"CONFIG_SAANS_VITALS_ENABLED=n", "CONFIG_SAANS_VITALS_ENABLED=y"})
        self.assertIn("CONFIG_SAANS_RAW_MODE=n", settings("home-rx"))

    def test_refuses_what_would_build_the_wrong_board(self):
        bad = [
            dict(role="rx", node=0, channel=6, gateway="10.0.0.2"),
            dict(role="rx", node=255, channel=6, gateway="10.0.0.2"),
            dict(role="rx", node=1, channel=14, gateway="10.0.0.2"),
            dict(role="rx", node=1, channel=6, gateway=None),
            dict(role="rx", node=1, channel=6, gateway="fe80::1"),
            dict(role="rx", node=1, channel=6, gateway="gateway.local"),
            dict(role="tx", node=0, channel=6, gateway=None, vitals=True),
            dict(role="rx", node=1, channel=6, gateway="10.0.0.2", vitals=True, bench=True),
        ]
        for kw in bad:
            with self.subTest(kw), self.assertRaises(node.NodeError):
                node.plan(**kw)

    def test_refuses_a_target_other_than_the_s3(self):
        old = os.environ.get("TARGET")
        os.environ["TARGET"] = "esp32c6"
        try:
            with self.assertRaisesRegex(node.NodeError, "ESP32-S3"):
                node.plan("tx", 0, 6, None)
        finally:
            if old is None:
                del os.environ["TARGET"]
            else:
                os.environ["TARGET"] = old

    def test_a_tx_build_gets_no_wifi_secrets(self):
        os.environ["SAANS_WIFI_PASSWORD"] = "from-the-shell"
        self.addCleanup(os.environ.pop, "SAANS_WIFI_PASSWORD")
        env = node.build_env(node.plan("tx", 0, 11, None), "Home", "pw")
        self.assertNotIn("SAANS_WIFI_PASSWORD", env)
        self.assertNotIn("SAANS_WIFI_SSID", env)
        self.assertEqual((env["SAANS_NODE_ID"], env["SAANS_WIFI_CHANNEL"]), ("0", "11"))
        rx = node.build_env(node.plan("rx", 2, 6, "10.0.0.2"), "Home", "pw")
        self.assertEqual((rx["PROFILE"], rx["SAANS_WIFI_PASSWORD"]), ("home-rx", "pw"))

    def test_reads_the_chip_from_esptool(self):
        self.assertEqual(node.chip_of("Detecting chip type... ESP32-S3\nChip is ESP32-S3 (QFN56) (revision v0.2)"),
                         "ESP32-S3")
        self.assertEqual(node.chip_of("Detecting chip type... ESP32\nChip is ESP32-D0WD-V3 (revision v3.1)"), "ESP32")
        self.assertEqual(node.chip_of("Detecting chip type... ESP32-C6"), "ESP32-C6")
        self.assertIsNone(node.chip_of("A fatal error occurred: Failed to connect to Espressif device"))


class BuildShSiteValuesTest(unittest.TestCase):
    """edge/build.sh with docker replaced: what it writes for one board."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.tmp)
        edge = self.tmp / "repo/edge"
        (edge / "firmware").mkdir(parents=True)
        shutil.copy(REPO / "edge/build.sh", edge / "build.sh")
        shutil.copytree(REPO / "edge/firmware/profiles", edge / "firmware/profiles")
        bindir = self.tmp / "bin"
        bindir.mkdir()
        (bindir / "docker").write_text('#!/bin/sh\necho "docker $*" >> "$DOCKER_LOG"\n')
        (bindir / "docker").chmod(0o755)
        self.edge = edge
        self.env = {"PATH": f"{bindir}:{os.environ['PATH']}", "HOME": str(self.tmp),
                    "DOCKER_LOG": str(self.tmp / "docker.log")}

    def build(self, **env):
        return subprocess.run(["bash", str(self.edge / "build.sh"), "build"], env={**self.env, **env},
                              capture_output=True, text=True, stdin=subprocess.DEVNULL)

    def test_writes_the_boards_values_where_only_you_can_read_them(self):
        r = self.build(PROFILE="home-rx", SAANS_NODE_ID="3", SAANS_WIFI_CHANNEL="11",
                       SAANS_GATEWAY_IP="192.168.1.20", SAANS_WIFI_SSID="Home",
                       SAANS_WIFI_PASSWORD='p"w\\x')
        self.assertEqual(r.returncode, 0, r.stderr)
        local = self.edge / "firmware/build-home-rx/local.defaults"
        self.assertEqual(local.stat().st_mode & 0o777, 0o600)
        self.assertEqual(local.read_text().splitlines(), [
            'CONFIG_SAANS_GATEWAY_IP="192.168.1.20"',
            'CONFIG_SAANS_WIFI_SSID="Home"',
            'CONFIG_SAANS_WIFI_PASSWORD="p\\"w\\\\x"',
            "CONFIG_SAANS_NODE_ID=3",
            "CONFIG_SAANS_WIFI_CHANNEL=11",
        ])
        self.assertNotIn('p"w', r.stdout + r.stderr)
        log = (self.tmp / "docker.log").read_text()
        self.assertNotIn('p"w', log)
        self.assertIn("profiles/home-rx.defaults;build-home-rx/local.defaults", log)

    def test_refuses_a_node_id_or_channel_out_of_range(self):
        for env in [dict(SAANS_NODE_ID="255"), dict(SAANS_NODE_ID="1x"), dict(SAANS_WIFI_CHANNEL="14"),
                    dict(SAANS_WIFI_CHANNEL="0")]:
            with self.subTest(env):
                r = self.build(PROFILE="home-tx", **env)
                self.assertEqual(r.returncode, 2)
                self.assertIn("must be", r.stderr)

    def test_chip_needs_a_port(self):
        r = subprocess.run(["bash", str(self.edge / "build.sh"), "chip"], env=self.env,
                           capture_output=True, text=True, stdin=subprocess.DEVNULL)
        self.assertEqual(r.returncode, 1)
        self.assertIn("set PORT", r.stderr)


class PortsTest(unittest.TestCase):
    def make(self, devices: dict[str, str]) -> tuple[str, str]:
        """A fake /dev and /sys with each tty under a USB device vid:pid."""
        tmp = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp)
        dev, sys_root = tmp / "dev", tmp / "sys"
        dev.mkdir()
        for name, ids in devices.items():
            (dev / name).touch()
            usb = sys_root / "devices/usb1" / name
            (usb / "iface").mkdir(parents=True)
            vid, pid = ids.split(":")
            (usb / "idVendor").write_text(vid + "\n")
            (usb / "idProduct").write_text(pid + "\n")
            (sys_root / "class/tty" / name).mkdir(parents=True)
            (sys_root / "class/tty" / name / "device").symlink_to(usb / "iface")
        return str(dev), str(sys_root)

    def test_tells_the_bridge_from_the_native_socket(self):
        found = ports.list_ports(*self.make({"ttyUSB0": "10c4:ea60", "ttyACM0": "303a:1001"}))
        self.assertEqual({p.dev.rsplit("/", 1)[1]: p.kind for p in found},
                         {"ttyUSB0": "bridge", "ttyACM0": "native"})
        self.assertTrue(ports.choose(found, None).dev.endswith("ttyUSB0"))

    def test_warns_about_the_cable_and_the_socket(self):
        self.assertIn("charge-only", " ".join(ports.advice([])))
        native = ports.list_ports(*self.make({"ttyACM0": "303a:1001"}))
        self.assertIn("UART", " ".join(ports.advice(native)))
        bridge = ports.list_ports(*self.make({"ttyUSB0": "1a86:55d4"}))
        self.assertEqual(ports.advice(bridge), [])

    def test_two_boards_need_a_choice(self):
        two = ports.list_ports(*self.make({"ttyUSB0": "10c4:ea60", "ttyUSB1": "10c4:ea60"}))
        self.assertIsNone(ports.choose(two, None))
        self.assertTrue(any("More than one" in a for a in ports.advice(two)))

    def test_waits_for_a_board_to_appear(self):
        board = [ports.Port("/dev/ttyUSB0", "10c4")]
        calls = iter([[], board, board])
        new = ports.wait_for_new([], 2.0, lister=lambda: next(calls))
        self.assertEqual([p.dev for p in new], ["/dev/ttyUSB0"])
        self.assertEqual(ports.wait_for_new([], 0.3, lister=lambda: []), [])


RX_LOG = """\x1b[0;32mI (312) saans:   role        : RX (sensing)\x1b[0m
I (313) saans:   node id     : 2
I (2101) wifi:connected with MyHomeNetwork, aid = 3, channel 6, BW20
I (3001) saans.net: got ip 192.168.1.42
I (3002) saans.net: uplink -> 192.168.1.20:5566
I (63001) saans.rx: csi=19.8 Hz frames=120 dropped=0 send_fail=0
I (63002) saans.rx: capture: len=128 (first 128) -> 64 subcarriers, expecting 56; not_ht=0
"""


class SelfTestTest(unittest.TestCase):
    def report(self, text: str) -> selftest.SerialReport:
        rep = selftest.SerialReport()
        for line in text.splitlines():
            rep.feed(line)
        return rep

    def test_a_healthy_receiver_passes_and_its_network_name_is_not_shown(self):
        ok, lines = self.report(RX_LOG).verdict()
        self.assertTrue(ok)
        text = "\n".join(lines)
        self.assertIn("node 2 (RX): CSI arriving at 19.8 Hz", text)
        self.assertNotIn("MyHomeNetwork", text)

    def test_firmware_warnings_fail_it_in_their_own_words(self):
        rep = self.report(RX_LOG + "W (9) saans.rx: no CSI for 1 s (TX down, or channel mismatch?)\n")
        ok, lines = rep.verdict()
        self.assertFalse(ok)
        self.assertIn("same channel", " ".join(lines))
        self.assertFalse(self.report(RX_LOG.replace("csi=19.8", "csi=6.0")).verdict()[0])
        self.assertFalse(self.report(RX_LOG.replace("send_fail=0", "send_fail=7")).verdict()[0])

    def test_a_tx_reports_its_beacon_rate(self):
        rep = self.report("I saans:   role        : TX (beacon)\nI saans.tx: ESP-NOW rate: HT20 MCS0 LGI\n"
                          "I saans.tx: sent=1200 failed=0 rate=20.0 Hz\n")
        ok, lines = rep.verdict()
        self.assertTrue(ok)
        self.assertIn("beacons at 20.0 Hz", lines[-1])

    def test_silence_is_not_a_pass(self):
        self.assertFalse(self.report("").verdict()[0])

    def test_frame_rates_from_two_snapshots(self):
        def snap(frames):
            return [{"space_id": "bedroom", "links": [
                {"link": {"rx": 1, "tx": 255}, "up": True, "rate_hz": 20.0, "loss": 0.01,
                 "raw_mode": False, "frames": frames}]}]
        (r,) = selftest.rates_from_links(snap(100), snap(140), 20.0)
        self.assertEqual((r.node, r.frame_hz, r.csi_hz), (1, 2.0, 20.0))
        self.assertTrue(r.ok())
        self.assertIn("frames arriving at 2.0 Hz from node 1", r.line())
        (stalled,) = selftest.rates_from_links(snap(100), snap(100), 20.0)
        self.assertFalse(stalled.ok())

    def test_classifies_the_two_datagram_types(self):
        from wifisense.features import FeatureFrame
        self.assertEqual(selftest.classify(FeatureFrame(node_id=7).encode()), ("feature", 7))
        raw = (0xC5110001).to_bytes(4, "little") + bytes([3]) + bytes(40)
        self.assertEqual(selftest.classify(raw), ("raw", 3))
        self.assertIsNone(selftest.classify(b"hello world, not a frame"))

    def test_listens_and_counts_per_node(self):
        from wifisense.features import FeatureFrame
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]

        def send():
            time.sleep(0.2)
            with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
                for i in range(4):
                    s.sendto(FeatureFrame(node_id=5, seq=i).encode(), ("127.0.0.1", port))

        threading.Thread(target=send, daemon=True).start()
        out = StringIO()
        with redirect_stdout(out):
            rc = selftest.from_udp(port, 1.0, "127.0.0.1")
        self.assertEqual(rc, 0, out.getvalue())
        self.assertIn("frames arriving at 4.0 Hz from node 5", out.getvalue())

    def test_listens_on_one_address_and_says_when_it_is_not_this_machines(self):
        err = StringIO()
        with redirect_stdout(StringIO()), unittest.mock.patch("sys.stderr", err):
            rc = selftest.from_udp(0, 0.1, "192.0.2.1")  # TEST-NET-1: never a local address
        self.assertEqual(rc, 2)
        self.assertIn("--bind", err.getvalue())


class CliTest(unittest.TestCase):
    def test_check_and_dry_run_are_offered_and_no_password_flag(self):
        cli = [sys.executable, str(REPO / "tools/homekit")]
        r = subprocess.run([*cli, "gateway", "--help"], capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--check", r.stdout)
        r = subprocess.run([*cli, "node", "--help"], capture_output=True, text=True)
        self.assertIn("--dry-run", r.stdout)
        self.assertNotIn("--password", r.stdout)


if __name__ == "__main__":
    unittest.main()
