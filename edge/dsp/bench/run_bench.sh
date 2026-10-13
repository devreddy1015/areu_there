#!/usr/bin/env bash
# Saans edge DSP — build and run the host benchmark, next to the Python
# reference on the SAME input, and print the comparison.
#
#   ./run_bench.sh                      5 C++ passes, 3 Python passes
#   ./run_bench.sh --runs 9             more C++ passes
#   ./run_bench.sh --python-runs 1      one Python pass (it costs ~17 s each)
#   ./run_bench.sh --no-python          skip the reference entirely
#   CXX=clang++ ./run_bench.sh          the other toolchain
#
# The input is `spec/golden/features_vitals` for both sides: 3267 CSI frames
# of 32 subcarriers at 20 Hz, with its 3% sequence loss and its 2 s outage.
# `spec/golden/bench_python.py` drives the reference `FeatureExtractor` over
# exactly those arrays, so the two figures are comparable rather than merely
# adjacent -- both sides emit 336 FeatureFrames, and the script checks that
# they did before printing a ratio.
#
# WHAT THE NUMBER IS AND IS NOT. This is a host figure on a desktop CPU. It is
# NOT an ESP32-S3 figure and must not be quoted as one; the S3 is a 240 MHz
# Xtensa LX7 with a single-precision FPU and no SIMD. What the comparison does
# establish is the ratio against the reference on identical input, which is a
# property of the two implementations rather than of this machine.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DSP="$(cd "${HERE}/.." && pwd)"
ROOT="$(cd "${DSP}/../.." && pwd)"
GOLDEN="${ROOT}/spec/golden"
BUILD="${HERE}/build"

CXX="${CXX:-g++}"
CXXFLAGS="-std=c++17 -O2 -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion -fno-exceptions -fno-rtti"

RUNS=5
PY_RUNS=3
WANT_PYTHON=1
EXTRA=()

while [ $# -gt 0 ]; do
    case "$1" in
        --runs)        RUNS="$2"; shift 2 ;;
        --python-runs) PY_RUNS="$2"; shift 2 ;;
        --no-python)   WANT_PYTHON=0; shift ;;
        --encode)      EXTRA+=(--encode); shift ;;
        -h|--help)     sed -n '2,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)             echo "unknown option: $1" >&2; exit 2 ;;
    esac
done

if [ ! -f "${GOLDEN}/features_vitals.kv" ]; then
    echo "ERROR: ${GOLDEN}/features_vitals.kv is missing." >&2
    echo "The benchmark drives the golden recording and will not invent one." >&2
    echo "Regenerate with: python3 spec/golden/generate.py" >&2
    exit 2
fi

mkdir -p "${BUILD}"
echo "  CXX  bench_extractor  (${CXX##*/})"
# shellcheck disable=SC2086
"${CXX}" ${CXXFLAGS} -I"${DSP}" -I"${DSP}/test" \
    -DSAANS_GOLDEN_DIR="\"${GOLDEN}\"" \
    "${HERE}/bench_extractor.cpp" -o "${BUILD}/bench_extractor"

CPP_OUT="$("${BUILD}/bench_extractor" --runs "${RUNS}" "${EXTRA[@]+"${EXTRA[@]}"}" "${GOLDEN}")"
echo "${CPP_OUT}" | grep -v '^BENCH '

CPP_FPS="$(echo "${CPP_OUT}"  | sed -n 's/.*cpp_frames_per_s=\([0-9.]*\).*/\1/p')"
CPP_XRT="$(echo "${CPP_OUT}"  | sed -n 's/.*cpp_xrt=\([0-9.]*\).*/\1/p')"
CPP_EMIT="$(echo "${CPP_OUT}" | sed -n 's/.*emitted=\([0-9]*\).*/\1/p')"
FS="$(echo "${CPP_OUT}"       | sed -n 's/.*fs=\([0-9.]*\).*/\1/p')"

if [ "${WANT_PYTHON}" -eq 0 ]; then
    echo
    echo "  (reference skipped: --no-python)"
    exit 0
fi

echo
echo "  PYTHON REFERENCE on the same input (spec/golden/bench_python.py)"
PY_FPS_LIST=()
PY_EMIT=""
for _ in $(seq 1 "${PY_RUNS}"); do
    line="$(cd "${ROOT}" && python3 "${GOLDEN}/bench_python.py" | tail -1)"
    echo "    ${line}"
    PY_FPS_LIST+=("$(echo "${line}" | sed -n 's/^ *\([0-9.]*\) frames\/s.*/\1/p')")
    PY_EMIT="$(echo "${line}" | sed -n 's/.*; \([0-9]*\) FeatureFrames.*/\1/p')"
done

PY_FPS="$(printf '%s\n' "${PY_FPS_LIST[@]}" | sort -n | awk '{a[NR]=$1} END {print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}')"

echo
echo "================================================================"
echo "  COMPARISON — same 3267-frame recording, same configuration"
echo "================================================================"
if [ "${CPP_EMIT}" != "${PY_EMIT}" ]; then
    echo "  !! C++ emitted ${CPP_EMIT} FeatureFrames, Python ${PY_EMIT}."
    echo "  !! The two are NOT doing the same work; the ratio below is meaningless."
    echo "  !! Fix the parity failure before quoting any speed number."
else
    echo "  both emitted ${CPP_EMIT} FeatureFrames — the same work, measured twice"
fi
awk -v c="${CPP_FPS}" -v p="${PY_FPS}" -v fs="${FS}" 'BEGIN {
    printf "\n  %-24s %12s %14s\n", "", "frames/s", "x realtime";
    printf "  %-24s %12.0f %14.0f\n", "C++ (float32, edge/dsp)", c, c/fs;
    printf "  %-24s %12.0f %14.1f\n", "Python (f64, reference)", p, p/fs;
    printf "\n  the C++ port is %.0fx the reference on identical input.\n", c/p;
    printf "  One link at %g Hz needs 1x realtime, so the host has %.0fx of headroom;\n", fs, c/fs;
    printf "  the ESP32-S3 figure is NOT this number and is only measurable on the part\n";
    printf "  (see the header of this script and of bench_extractor.cpp).\n";
    printf "\n  Both sides are affected by what else this machine is doing. Run it on an\n";
    printf "  idle box before quoting either figure; the RATIO is the stabler claim.\n";
}'
echo "================================================================"
