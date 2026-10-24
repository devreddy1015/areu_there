#!/usr/bin/env bash
# Build the Saans node firmware in the pinned ESP-IDF container.
#
# Nothing is installed on the host: the toolchain, Python environment and
# target support all live in espressif/idf:v5.4.1. That is the whole reason
# this script exists rather than a README line telling you to install IDF.
#
#   ./build.sh                 build for esp32s3 (default)
#   ./build.sh menuconfig      open the config UI
#   ./build.sh clean           remove build output
#   ./build.sh size            report binary size
#   ./build.sh flash           flash over USB (needs a board and --device)
#   ./build.sh tests           run the host DSP tests instead (no Docker)
#
#   PROFILE=bench-rx ./build.sh build        one board, from firmware/profiles/
#   PROFILE=bench-rx PORT=… ./build.sh flash-monitor
#
# Env:
#   IDF_IMAGE   override the container image
#   TARGET      override the chip target
#   PORT        serial port for flash/monitor, e.g. /dev/ttyUSB0
#   PROFILE     build from firmware/profiles/$PROFILE.defaults, in its own
#               build-$PROFILE/ with its own sdkconfig (see below)
#   SAANS_GATEWAY_IP, SAANS_WIFI_SSID, SAANS_WIFI_PASSWORD
#               with PROFILE: the site's values, kept out of committed files
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDF_IMAGE="${IDF_IMAGE:-espressif/idf:v5.4.1}"
TARGET="${TARGET:-esp32s3}"
CMD="${1:-build}"

# PROFILES. Without one, the build uses firmware/sdkconfig, which
# `menuconfig` edits and which is shared by every board you build. With
# PROFILE=<name>, a board is built from sdkconfig.defaults, then
# profiles/<name>.defaults, then the site's values from the environment, into
# build-<name>/ with its own sdkconfig, which is regenerated on every build.
# The profile file is then the whole truth about that board: building the TX
# cannot change the RX, and `clean` cannot delete anyone's answers.
IDF_OPTS=()
BUILD_REL="build"
if [ -n "${PROFILE:-}" ]; then
    if [ ! -f "${HERE}/firmware/profiles/${PROFILE}.defaults" ]; then
        echo "error: no firmware/profiles/${PROFILE}.defaults; have:" \
             "$(cd "${HERE}/firmware/profiles" && ls ./*.defaults | sed 's|^\./||; s|\.defaults$||' | tr '\n' ' ')" >&2
        exit 2
    fi
    BUILD_REL="build-${PROFILE}"
    IDF_OPTS=(-B "${BUILD_REL}" -D "SDKCONFIG=${BUILD_REL}/sdkconfig"
              -D "SDKCONFIG_DEFAULTS=sdkconfig.defaults;profiles/${PROFILE}.defaults;${BUILD_REL}/local.defaults")
fi

run_idf() {
    # The repo root is mounted, not just edge/firmware, because the firmware
    # includes ../../dsp — a narrower mount would break the include path in a
    # way that is tedious to diagnose.
    local repo_root
    repo_root="$(cd "${HERE}/.." && pwd)"
    # Run as the invoking user, not root. Without this the container writes
    # build/ and sdkconfig as root and the host cannot clean them afterwards -
    # a small thing that wastes real time the first time you hit it.
    local docker_args=(--rm -v "${repo_root}":/project -w /project/edge/firmware
                       --user "$(id -u):$(id -g)" -e HOME=/tmp)
    # Variables passed through by NAME (`-e VAR`), so a value such as a key
    # never appears on a command line or in `ps`.
    local v
    for v in ${IDF_ENV:-}; do
        docker_args+=(-e "${v}")
    done
    if [ -n "${PORT:-}" ]; then
        docker_args+=(--device "${PORT}")
    fi
    if [ -t 0 ]; then
        docker_args+=(-it)
    fi
    docker run "${docker_args[@]}" "${IDF_IMAGE}" "$@"
}

case "${CMD}" in
    build)
        echo "==> building ${PROFILE:-firmware/sdkconfig} for ${TARGET} in ${IDF_IMAGE}"
        if [ -n "${PROFILE:-}" ]; then
            mkdir -p "${HERE}/firmware/${BUILD_REL}"
            # The site's values, quoted for Kconfig, written here and only
            # here (a later `flash` must not blank them). In the git-ignored
            # build dir, readable by you alone: the WiFi password is in it.
            local_defaults="${HERE}/firmware/${BUILD_REL}/local.defaults"
            ( umask 077; : > "${local_defaults}" )
            for var in SAANS_GATEWAY_IP SAANS_WIFI_SSID SAANS_WIFI_PASSWORD; do
                if [ -n "${!var+set}" ]; then
                    value="${!var//\\/\\\\}"
                    printf 'CONFIG_%s="%s"\n' "${var}" "${value//\"/\\\"}" >> "${local_defaults}"
                fi
            done
            # Regenerated from the profile every time; see PROFILES above.
            rm -f "${HERE}/firmware/${BUILD_REL}/sdkconfig" "${HERE}/firmware/${BUILD_REL}/sdkconfig.old"
        fi
        # The target as a define, not `idf.py set-target`: set-target replaces
        # sdkconfig with the defaults (the old one becomes sdkconfig.old), so
        # running it before every build threw away every `menuconfig` answer
        # and built each board as the defaults. -D IDF_TARGET sets it on a
        # fresh checkout and keeps an existing sdkconfig as it is.
        run_idf idf.py "${IDF_OPTS[@]}" -D "IDF_TARGET=${TARGET}" build
        echo
        echo "==> artefacts in edge/firmware/${BUILD_REL}/"
        ;;
    menuconfig)
        if [ -n "${PROFILE:-}" ]; then
            echo "error: a profile is configured by editing firmware/profiles/${PROFILE}.defaults;" \
                 "menuconfig answers would be regenerated away on the next build" >&2
            exit 2
        fi
        run_idf idf.py -D "IDF_TARGET=${TARGET}" menuconfig
        ;;
    size|size-components|reconfigure)
        run_idf idf.py "${IDF_OPTS[@]}" -D "IDF_TARGET=${TARGET}" "${CMD}"
        ;;
    clean)
        if [ -n "${PROFILE:-}" ]; then
            echo "==> removing edge/firmware/${BUILD_REL}"
            rm -rf "${HERE:?}/firmware/${BUILD_REL}"
        else
            echo "==> removing build output"
            rm -rf "${HERE}/firmware/build" "${HERE}/firmware/sdkconfig" "${HERE}/firmware/sdkconfig.old"
        fi
        echo "done"
        ;;
    flash|monitor|flash-monitor)
        if [ -z "${PORT:-}" ]; then
            echo "error: set PORT, e.g. PORT=/dev/ttyUSB0 ./build.sh flash" >&2
            exit 1
        fi
        case "${CMD}" in
            flash)         run_idf idf.py "${IDF_OPTS[@]}" -p "${PORT}" flash ;;
            monitor)       run_idf idf.py "${IDF_OPTS[@]}" -p "${PORT}" monitor ;;
            flash-monitor) run_idf idf.py "${IDF_OPTS[@]}" -p "${PORT}" flash monitor ;;
        esac
        ;;
    tests)
        # Host tests need no container; they are the same headers the firmware
        # compiles, built with g++ against the golden vectors.
        make -C "${HERE}/dsp/test" check
        ;;
    provision)
        # Write a node's frame key (ADR-012) into its `saanskey` partition:
        #
        #   export SAANS_NODE_KEY=$(saans-gateway -c saans.toml --node-key 3)
        #   KEY_ID=0 PORT=/dev/ttyUSB0 ./build.sh provision
        #
        # DRY_RUN=1 builds the image and prints what is in it, key redacted,
        # without a board. The image is built in the container's own /tmp,
        # which is not the mounted repository, so the key never lands on this
        # disk; it reaches the container by variable name, never by value.
        #
        # The boot counter starts at the Unix time now. Each boot adds one, so
        # a node provisioned again later starts above every frame the gateway
        # has accepted from it, and no replay floor needs clearing.
        if ! [[ "${SAANS_NODE_KEY:-}" =~ ^[0-9a-fA-F]{64}$ ]]; then
            echo "error: set SAANS_NODE_KEY to the 64 hex digits saans-gateway --node-key prints" >&2
            exit 1
        fi
        KEY_ID="${KEY_ID:-0}"
        if ! [[ "${KEY_ID}" =~ ^[0-9]+$ ]] || [ "${KEY_ID}" -gt 255 ]; then
            echo "error: KEY_ID must be 0..255, the id of the gateway's [[frame_key]]" >&2
            exit 1
        fi
        if [ -z "${DRY_RUN:-}" ] && [ -z "${PORT:-}" ]; then
            echo "error: set PORT, e.g. PORT=/dev/ttyUSB0 (or DRY_RUN=1 to only build the image)" >&2
            exit 1
        fi
        export KEY_ID DRY_RUN="${DRY_RUN:-}" BOOT_START="$(date +%s)" ESPPORT="${PORT:-}"
        IDF_ENV="SAANS_NODE_KEY KEY_ID DRY_RUN BOOT_START ESPPORT" run_idf sh -c '
            set -eu
            d=$(mktemp -d)
            trap "rm -rf $d" EXIT
            {
                echo "key,type,encoding,value"
                echo "frameauth,namespace,,"
                echo "key,data,hex2bin,${SAANS_NODE_KEY}"
                echo "key_id,data,u8,${KEY_ID}"
                echo "boot,data,u32,${BOOT_START}"
            } > "$d/saanskey.csv"
            python "$IDF_PATH/components/nvs_flash/nvs_partition_generator/nvs_partition_gen.py" \
                generate "$d/saanskey.csv" "$d/saanskey.bin" 0x3000 >/dev/null
            if [ -n "${DRY_RUN}" ]; then
                python "$IDF_PATH/components/nvs_flash/nvs_partition_tool/nvs_tool.py" \
                    -f json "$d/saanskey.bin" > "$d/dump.json"
                python - "$d/dump.json" <<"PY"
import json, sys
# What a node will read, without printing the key itself. The "raw" fields
# of the dump hold the key bytes, so only these three fields are printed.
# (No apostrophes anywhere in here: this is inside a single-quoted sh -c.)
for page in json.load(open(sys.argv[1]))["pages"]:
    for e in page.get("entries", []):
        name, kind = e.get("key"), e.get("metadata", {}).get("type")
        if not name or e.get("state") != "Written" or kind == "blob_data":
            continue
        value = e.get("data", {}).get("value")
        if name == "key":
            value = "<%s bytes, redacted>" % e["data"].get("size")
        print(f"  {name:10} {kind:10} {value}")
PY
                echo "dry run: image built, nothing flashed"
            else
                python -m esptool --chip esp32s3 -p "${ESPPORT}" write_flash 0x12000 "$d/saanskey.bin"
                echo "provisioned: key_id ${KEY_ID}, boot counter starts at ${BOOT_START}"
            fi
        '
        ;;
    *)
        echo "usage: $0 {build|menuconfig|clean|size|flash|monitor|flash-monitor|tests|provision}" >&2
        exit 2
        ;;
esac
