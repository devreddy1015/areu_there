"""Set up a Saans home kit: docs/HOME_KIT.md.

    sudo python3 tools/homekit gateway [--check]     the gateway, as a service
    python3 tools/homekit node --role rx --id 1 ...  one board, built and flashed
    python3 tools/homekit selftest [--serial DEV]    frames arriving, per node

Python 3.11 or later, standard library only. Building the gateway needs
cargo or Docker; building a board needs Docker (edge/build.sh).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# `python3 tools/homekit` puts tools/homekit on the path; the modules
# import each other as `homekit.*`, so put tools/ there instead.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from homekit import gateway, node, selftest  # noqa: E402


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="homekit", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    g = sub.add_parser("gateway", help="install and start the gateway for one home")
    g.add_argument("--check", action="store_true", help="say what would change; change nothing")
    g.add_argument("--room", action="append", default=[], metavar="ID=RX[,RX]",
                   help="a room and its receiver node ids, e.g. bedroom=1,2 (repeat per room)")
    g.add_argument("--vitals", action="append", default=[], metavar="ID",
                   help="breathing on in this room: only with written consent (ADR-005)")
    g.add_argument("--admin", default="admin", help="the dashboard admin's user name")
    g.add_argument("--admin-password-file", metavar="FILE",
                   help="read the admin's password from the first line of FILE")
    g.add_argument("--new-password", action="store_true", help="replace the admin's password")
    g.add_argument("--reconfigure", action="store_true",
                   help="rewrite the config (the old one is kept beside it)")
    g.add_argument("--local-only", action="store_true",
                   help="dashboard on this machine only (127.0.0.1), no sign-in")
    g.add_argument("--utc-offset-minutes", type=int, help="default: this machine's")
    g.add_argument("--binary", metavar="PATH", help="install this saans-gateway instead of building")
    g.add_argument("--no-start", action="store_true", help="install, but do not start the service")
    g.add_argument("--root", default="/", help=argparse.SUPPRESS)  # tests install into a tmpdir

    n = sub.add_parser("node", help="build and flash one board")
    n.add_argument("--role", required=True, choices=["tx", "rx"])
    n.add_argument("--id", type=int, required=True, help="node id: TX 0, receivers 1..254")
    n.add_argument("--gateway", help="receiver: the gateway machine's IPv4 address")
    n.add_argument("--ssid", help="receiver: the home WiFi's name (password is asked for)")
    n.add_argument("--channel", type=int, default=6, help="the home WiFi's channel (default 6)")
    n.add_argument("--vitals", action="store_true", help="receiver: breathing, with written consent")
    n.add_argument("--bench", action="store_true",
                   help="the first bench session's profile: raw CSI on (bench/README.md)")
    n.add_argument("--port", help="serial device; found automatically when one board is plugged in")
    n.add_argument("--dry-run", action="store_true", help="check everything; build and flash nothing")
    n.add_argument("--no-flash", action="store_true", help="build only")
    n.add_argument("--no-selftest", action="store_true", help="skip reading the board's log after")
    n.add_argument("--yes", action="store_true", help="ask nothing (consent for --vitals included)")

    s = sub.add_parser("selftest", help="are frames arriving, and how fast")
    where = s.add_mutually_exclusive_group()
    where.add_argument("--serial", metavar="DEV", help="read a board's own log instead")
    where.add_argument("--udp", action="store_true", help="listen on the UDP port; no gateway running")
    s.add_argument("--url", default="http://127.0.0.1:8080", help="the gateway (default: this machine)")
    s.add_argument("--token-file", help=f"default {gateway.TOKEN_FILE}")
    s.add_argument("--port", type=int, default=5566, help="with --udp")
    s.add_argument("--seconds", type=float, help="how long to watch")
    return ap


def main(argv: list[str] | None = None) -> int:
    if sys.version_info < (3, 11):
        sys.exit("homekit needs Python 3.11 or later (Debian 12 and Raspberry Pi OS bookworm have it)")
    args = parser().parse_args(argv)
    try:
        return {"gateway": gateway.run, "node": node.run, "selftest": selftest.run}[args.cmd](args)
    except KeyboardInterrupt:
        print("\nstopped", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
