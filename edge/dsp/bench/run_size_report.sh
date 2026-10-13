#!/usr/bin/env bash
# Saans edge DSP — build and run the per-link memory report.
#
#   ./run_size_report.sh              g++ (or $CXX)
#   ./run_size_report.sh --both       g++ AND clang++, because a size claim
#                                     that only holds under one compiler's
#                                     layout is not a size claim
#   ./run_size_report.sh --target     the REAL figures: cross-compile the same
#                                     objects with the ESP32-S3 toolchain in
#                                     the pinned IDF container and read their
#                                     sizes out of the object file
#   CXX=clang++ ./run_size_report.sh  the other toolchain
#
# It prints `sizeof` for four shapes of the extractor, the term-by-term
# breakdown behind the difference, how many links each leaves room for in the
# node's 171 KB of free DIRAM -- and then drives the `features_vitals` golden
# recording through every shape and compares the emitted FeatureFrames field by
# field, exactly. Exit status is 0 only if they all agree: the memory claims
# are worth nothing if the smaller shapes compute something different.
#
# The sizes are HOST sizes. The ESP32-S3 figure is smaller (4-byte pointers and
# size_t); the firmware logs its own sizeof(Extractor) at boot, which is the
# only figure that is true of the node. See the header of size_report.cpp.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSP="$(cd "${HERE}/.." && pwd)"
ROOT="$(cd "${DSP}/../.." && pwd)"
GOLDEN="${ROOT}/spec/golden"
BUILD="${HERE}/build"

CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion -fno-exceptions -fno-rtti"

IDF_IMAGE="${IDF_IMAGE:-espressif/idf:v5.4.1}"

# The four static extractor objects in size_report.cpp, in table order. The
# names are the ones nm prints for a file-static object in an anonymous
# namespace, which is where they live precisely so they are measurable.
TARGET_SYMBOLS=(
    "g_stock:A  stock <56>"
    "g_tight:B  tightened"
    "g_node:C  tightened + shared scratch"
    "g_node_nobnr:D  C, best_bnr compiled out"
)

# The ESP32-S3 figures, measured rather than extrapolated from the host's.
#
# It cross-compiles the SAME translation unit with the SAME flags and reads the
# .bss size of each object out of the unlinked object file. Nothing is linked
# and nothing is flashed; the toolchain is the one the firmware is built with,
# in the pinned container, so this is the layout the node actually gets.
target_report() {
    if ! command -v docker >/dev/null 2>&1; then
        echo "ERROR: --target needs docker and ${IDF_IMAGE}; the host figures above" >&2
        echo "are NOT the node's and must not be quoted as them." >&2
        return 2
    fi
    local repo
    repo="$(cd "${ROOT}" && pwd)"
    echo ""
    echo "== ESP32-S3 (xtensa-esp32s3-elf-g++ in ${IDF_IMAGE}), .bss per object"
    echo ""
    local nm_out
    nm_out="$(docker run --rm -v "${repo}":/project -w /project \
        --user "$(id -u):$(id -g)" -e HOME=/tmp "${IDF_IMAGE}" sh -c \
        "xtensa-esp32s3-elf-g++ ${CXXFLAGS/-std=c++17/-std=gnu++17} \
             -I edge/dsp -I edge/dsp/test -DSAANS_GOLDEN_DIR='\"/nonexistent\"' \
             -c edge/dsp/bench/size_report.cpp -o /tmp/size_report.o \
         && xtensa-esp32s3-elf-nm -S -t d /tmp/size_report.o" 2>/dev/null)"
    if [ -z "${nm_out}" ]; then
        echo "ERROR: the cross build produced nothing; is ${IDF_IMAGE} present?" >&2
        return 2
    fi
    printf '  %-34s %9s %9s %7s\n' "configuration" "bytes/link" "once/node" "links"
    printf '  %-34s %9s %9s %7s\n' "----------------------------------" \
        "---------" "---------" "-------"
    local arena
    arena="$(echo "${nm_out}" | awk '/SharedScratch/ && /storage/ {print $2 + 0; exit}')"
    arena="${arena:-0}"
    local entry sym label bytes
    for entry in "${TARGET_SYMBOLS[@]}"; do
        sym="${entry%%:*}"
        label="${entry#*:}"
        bytes="$(echo "${nm_out}" | awk -v s="${sym}E\$" \
            '$4 ~ s {print $2 + 0; exit}')"
        if [ -z "${bytes}" ]; then
            printf '  %-34s %9s %9s %7s\n' "${label}" "?" "?" "?"
            continue
        fi
        local once=0
        case "${sym}" in g_node|g_node_nobnr) once="${arena}" ;; esac
        awk -v l="${label}" -v b="${bytes}" -v o="${once}" -v budget=$((171 * 1024)) \
            'BEGIN { printf "  %-34s %9d %9d %7d\n", l, b, o, int((budget - o) / b) }'
    done
    echo ""
    echo "  (171 KB of free DIRAM; the arena is counted once for the node, the"
    echo "   rest once per link. Host figures above are larger: 8-byte pointers.)"
}

COMPILERS=("${CXX:-g++}")
WANT_TARGET=0
if [ "${1:-}" = "--both" ]; then
    COMPILERS=(g++ clang++)
elif [ "${1:-}" = "--target" ]; then
    WANT_TARGET=1
elif [ $# -gt 0 ]; then
    echo "usage: $0 [--both | --target]" >&2
    exit 2
fi

if [ ! -f "${GOLDEN}/features_vitals.kv" ]; then
    echo "ERROR: ${GOLDEN}/features_vitals.kv is missing." >&2
    echo "The parity half of this report drives the golden recording and will" >&2
    echo "not invent one. Regenerate with: python3 spec/golden/generate.py" >&2
    exit 2
fi

mkdir -p "${BUILD}"
rc=0
for cxx in "${COMPILERS[@]}"; do
    if ! command -v "${cxx}" >/dev/null 2>&1; then
        echo ">>> ${cxx} not installed, skipping COMPILER (never a measurement)"
        continue
    fi
    echo "  CXX  size_report  (${cxx##*/})"
    # shellcheck disable=SC2086
    "${cxx}" ${CXXFLAGS} -I"${DSP}" -I"${DSP}/test" \
        -DSAANS_GOLDEN_DIR="\"${GOLDEN}\"" \
        "${HERE}/size_report.cpp" -o "${BUILD}/size_report"
    "${BUILD}/size_report" || rc=1
done

if [ "${WANT_TARGET}" -eq 1 ]; then
    target_report || rc=$?
fi
exit "${rc}"
