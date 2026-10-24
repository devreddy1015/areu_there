#!/usr/bin/env bash
# One B1 bench session (bench/README.md): three segments recorded with
# `wifisense record`, each checked with `wifisense stats`, and a README that
# records what the brief asks for (date, room, distance, orientation,
# firmware commit). Nothing here talks to a board; the RX must already be
# streaming raw CSI to this laptop's UDP port.
#
#   bench/session.sh desk-01
#
# Env:
#   SEGMENT_S   seconds per segment (default 600, the brief's 10 minutes)
#   SEGMENTS    which, in order (default "empty seated walking")
#   UDP_PORT    where the RX sends (default 5566)
#
# Writes data/bench/<YYYY-MM-DD>-<name>/: <segment>.pcap, <segment>.stats.txt,
# README.md. The pcaps are git-ignored and stay that way: a recording with a
# person in it is health data (ADR-001, ADR-005), and this repository is
# public. The README holds no recording and can be committed.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEGMENT_S="${SEGMENT_S:-600}"
SEGMENTS="${SEGMENTS:-empty seated walking}"
UDP_PORT="${UDP_PORT:-5566}"

name="${1:-}"
if ! [[ "${name}" =~ ^[a-z0-9][a-z0-9-]*$ ]]; then
    echo "usage: $0 <session-name>   (lower case, digits and -; e.g. desk-01)" >&2
    exit 2
fi
dir="${REPO}/data/bench/$(date +%F)-${name}"
if [ -e "${dir}" ]; then
    echo "error: ${dir} exists; pick another name" >&2
    exit 1
fi

# Nothing else may hold the port: a second listener takes half the
# datagrams and it reads as packet loss (PROTOCOL_S5 §3.2).
if ! python3 -c "import socket; s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM); s.bind(('0.0.0.0', ${UDP_PORT}))" 2>/dev/null; then
    echo "error: UDP ${UDP_PORT} is already in use (is the gateway or a 'wifisense listen' running?)" >&2
    exit 1
fi

ask() {  # ask <variable> <question>
    local answer
    read -r -p "  $2: " answer
    printf -v "$1" '%s' "${answer}"
}

echo "B1 bench session '${name}': ${SEGMENTS} at ${SEGMENT_S} s each, on UDP ${UDP_PORT}"
echo "Measure, do not estimate (PROTOCOL_S5 §3.3). Answers go into the session README."
ask room        "Room (where, which wall, what is in it)"
ask distance    "TX-RX distance, measured (m)"
ask heights     "TX and RX heights above the floor (m)"
ask orientation "Orientation (which way each board's antenna faces)"
ask notes       "Anything else (people nearby, other WiFi, doors open)"

commit="$(git -C "${REPO}" rev-parse --short HEAD)"
if [ -n "$(git -C "${REPO}" status --porcelain -- edge)" ]; then
    commit="${commit} (edge/ has uncommitted changes)"
fi
mkdir -p "${dir}"
{ command -v chronyc >/dev/null && chronyc tracking; } > "${dir}/chrony-start.txt" 2>&1 || true

instructions() {
    case "$1" in
        empty)   echo "Leave the room. Nobody within a few metres of the link, nothing moving." ;;
        seated)  echo "Sit still in the chair beside the link and breathe normally. You only (PROTOCOL_S5 §2.1)." ;;
        walking) echo "Walk slowly back and forth across the link, through the whole segment." ;;
        *)       echo "Segment '$1': whatever you planned for it." ;;
    esac
}

rows=""
for seg in ${SEGMENTS}; do
    echo
    echo "== ${seg} (${SEGMENT_S} s)"
    instructions "${seg}"
    if [ "${seg}" != "empty" ]; then
        read -r -p "  Only you, the author, will be recorded, with your written consent (PROTOCOL_S5 §2.3)? [y/N] " ok
        if [ "${ok}" != "y" ]; then
            echo "  stopped before '${seg}': nothing recorded for it" >&2
            break
        fi
    fi
    read -r -p "  Enter to start recording ${seg} " _
    python3 -m wifisense record --port "${UDP_PORT}" --out "${dir}/${seg}.pcap" \
        --duration-s "${SEGMENT_S}"
    python3 -m wifisense stats "${dir}/${seg}.pcap" | tee "${dir}/${seg}.stats.txt"
    rows="${rows}| ${seg} | \`${seg}.pcap\` (not committed) | \`${seg}.stats.txt\` |"$'\n'
done

{ command -v chronyc >/dev/null && chronyc tracking; } > "${dir}/chrony-end.txt" 2>&1 || true

cat > "${dir}/README.md" <<EOF
# B1 bench session: ${name}

SOURCE: **REAL** (recorded from ESP32-S3 boards). Not yet analysed.

| | |
|---|---|
| Date | $(date '+%F %H:%M %Z') |
| Room | ${room} |
| TX-RX distance (measured) | ${distance} m |
| Heights | ${heights} |
| Orientation | ${orientation} |
| Notes | ${notes} |
| Firmware commit | ${commit} |
| Board profiles | \`edge/firmware/profiles/bench-tx.defaults\`, \`bench-rx.defaults\` at that commit |
| Segment length | ${SEGMENT_S} s |

| Segment | Recording | Link check |
|---|---|---|
${rows}
The recordings stay on this machine. They are git-ignored, and a recording
with a person in it is health data (ADR-001, ADR-005). This README, the
stats files and the clock logs hold none.

Next (docs/HARDWARE.md §9, PROTOCOL_S5 §1):

- [ ] \`wifisense stats\`: at least 10 Hz sustained, and the loss, for every segment
- [ ] bin layout: mean amplitude per bin on \`empty.pcap\` (which bins are dead)
- [ ] \`first_word_invalid\` fraction, from the RX's \`capture:\` telemetry line
- [ ] amplitude noise per subcarrier on \`empty.pcap\` (mean, std, CV)
- [ ] NOTES.md §10, "First real CSI": what matched the simulator and what did not
EOF

echo
echo "==> ${dir}"
ls -la "${dir}"
