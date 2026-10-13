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
# Env:
#   IDF_IMAGE   override the container image
#   TARGET      override the chip target
#   PORT        serial port for flash/monitor, e.g. /dev/ttyUSB0
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IDF_IMAGE="${IDF_IMAGE:-espressif/idf:v5.4.1}"
TARGET="${TARGET:-esp32s3}"
CMD="${1:-build}"

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
        echo "==> building for ${TARGET} in ${IDF_IMAGE}"
        run_idf sh -c "idf.py set-target ${TARGET} >/dev/null && idf.py build"
        echo
        echo "==> artefacts in edge/firmware/build/"
        ;;
    menuconfig|size|size-components|reconfigure)
        run_idf idf.py "${CMD}"
        ;;
    clean)
        echo "==> removing build output"
        rm -rf "${HERE}/firmware/build" "${HERE}/firmware/sdkconfig" "${HERE}/firmware/sdkconfig.old"
        echo "done"
        ;;
    flash|monitor|flash-monitor)
        if [ -z "${PORT:-}" ]; then
            echo "error: set PORT, e.g. PORT=/dev/ttyUSB0 ./build.sh flash" >&2
            exit 1
        fi
        case "${CMD}" in
            flash)         run_idf idf.py -p "${PORT}" flash ;;
            monitor)       run_idf idf.py -p "${PORT}" monitor ;;
            flash-monitor) run_idf idf.py -p "${PORT}" flash monitor ;;
        esac
        ;;
    tests)
        # Host tests need no container; they are the same headers the firmware
        # compiles, built with g++ against the golden vectors.
        make -C "${HERE}/dsp/test" check
        ;;
    *)
        echo "usage: $0 {build|menuconfig|clean|size|flash|monitor|flash-monitor|tests}" >&2
        exit 2
        ;;
esac
