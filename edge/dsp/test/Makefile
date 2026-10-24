# Saans edge DSP — host test and benchmark build. THE single entry point.
#
# Plain make, no cmake (SAANS_PLAN §3.1). Builds each test_<name>.cpp into
# build/test_<name> and runs it against the golden vectors in spec/golden.
#
#   make check          EVERYTHING: both compilers, the missing-vector guard,
#                       and the benchmark against the Python reference.
#                       This is the one command. CI runs this.
#   make                build and run every test once, with $(CXX)
#   make all-compilers  the same with g++ and clang++ in turn
#   make guard          prove a missing vector FAILS rather than skips
#   make bench          host throughput vs spec/golden/bench_python.py
#   make clean
#
#   make check RUN_BENCH=0       skip the benchmark (it costs ~30 s)
#   make bench BENCH_RUNS=9 PY_RUNS=3     more repeats, tighter median
#   make CXX=clang++             one compiler, your choice
#
# Adding a later module is one word: append its name to TESTS below and drop
# test_<name>.cpp next to this file. Nothing else changes.
#
# THE SECOND HARNESS IS GONE. There used to be a second Makefile at
# edge/dsp/Makefile, written while two sessions were working in this tree at
# once; it globbed test/*.cpp and passed the vector directory as argv[1] where
# this one names its tests and passes it as -DSAANS_GOLDEN_DIR. Two harnesses
# meant two ways for a test to be "run", and a test that only one of them built
# was a test nobody noticed was missing. edge/dsp/Makefile now forwards here.
# Every test still accepts BOTH conventions (golden.hpp's dir() resolution
# order), so nothing depends on which one invoked it.

TESTS := foundation vitals preprocess gapfill_bnr feature_frame pipeline feature_extractor \
         tdma node_config

# ---------------------------------------------------------------------------

HERE       := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))
DSP_DIR    := $(abspath $(HERE)/..)
BENCH_DIR  := $(abspath $(HERE)/../bench)
GOLDEN_DIR := $(abspath $(HERE)/../../../spec/golden)
# Overridable so two runs can coexist. That is not hypothetical: this tree has
# had two sessions working in it at once, and `make check` begins by cleaning,
# so a second run deleted the first run's binaries between building and
# executing them -- which surfaces as "No such file or directory" and reads
# exactly like a test that refused to run. `make check BUILD=/tmp/mine` keeps
# them apart.
BUILD      ?= $(HERE)/build

CXX ?= g++

# How many repeats `make check` gives the benchmark. Deliberately modest: the
# Python reference takes ~17 s per pass, and `check` is run often. `make bench`
# on its own defaults to 5 and 3, which is the figure worth quoting.
BENCH_RUNS ?= 3
PY_RUNS    ?= 1
RUN_BENCH  ?= 1

# -Wdouble-promotion is not pedantry: on the ESP32-S3 every float that gets
# promoted to double is a software-emulation call costing 10-50x. Each warning
# is a real performance bug on target, so the test build treats the library
# headers exactly as the firmware build will.
STD   := -std=c++17
OPT   := -O2
WARN  := -Wall -Wextra -Wpedantic -Wconversion -Wdouble-promotion
EMBED := -fno-exceptions -fno-rtti
# No floating-point contraction: a*b + c is two roundings, never one fused
# multiply-add. The Python reference is two roundings by construction, so a
# build that fuses is not implementing the same spec -- and the compiler is
# free to fuse at one call site and not another, which breaks every
# bit-for-bit check in this suite. Plain x86-64 has no FMA, so this never
# showed until CI's runner moved to Ubuntu 26.04, whose build targets a CPU
# that has one: test_vitals' compensated-sum check then failed under g++
# (reproduced locally with -march=x86-64-v3, and gone with this flag). The
# ESP32-S3 FPU has a fused multiply-add too. Kept apart from OPT so that a
# run overriding OPT cannot drop it; CI builds once with -march=x86-64-v3
# precisely to catch anyone who does.
FPFLAGS := -ffp-contract=off

CXXFLAGS := $(STD) $(OPT) $(FPFLAGS) $(WARN) $(EMBED) -I$(DSP_DIR) -I$(HERE) \
            -DSAANS_GOLDEN_DIR='"$(GOLDEN_DIR)"'

HEADERS := $(wildcard $(DSP_DIR)/*.hpp) $(HERE)/golden.hpp
BINS    := $(addprefix $(BUILD)/test_,$(TESTS))

.PHONY: all build test check all-compilers guard bench tdma-report clean

all: test

build: $(BINS)

$(BUILD):
	@mkdir -p $(BUILD)

$(BUILD)/test_%: $(HERE)/test_%.cpp $(HEADERS) | $(BUILD)
	@printf '  CXX  %s  (%s)\n' '$(notdir $@)' '$(notdir $(CXX))'
	@$(CXX) $(CXXFLAGS) $< -o $@

test: build
	@fail=0; total=0; failed=""; \
	for t in $(TESTS); do \
	  total=$$((total+1)); \
	  if $(BUILD)/test_$$t; then :; else fail=$$((fail+1)); failed="$$failed $$t"; fi; \
	done; \
	echo ""; \
	echo "================================================================"; \
	if [ $$fail -eq 0 ]; then \
	  echo "  edge/dsp: ALL $$total TEST BINARIES PASSED  ($(notdir $(CXX)))"; \
	else \
	  echo "  edge/dsp: $$fail OF $$total TEST BINARIES FAILED  ($(notdir $(CXX)))"; \
	  echo "  failed:$$failed"; \
	fi; \
	echo "================================================================"; \
	exit $$fail

# Both toolchains, because a parity bug that only one compiler's float
# scheduling exposes is still a parity bug.
all-compilers:
	@rc=0; found=0; \
	for c in g++ clang++; do \
	  if command -v $$c >/dev/null 2>&1; then \
	    found=$$((found+1)); \
	    echo ""; echo ">>> $$c"; \
	    $(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) clean >/dev/null; \
	    $(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) test CXX=$$c || rc=1; \
	  else \
	    echo ">>> $$c not installed, skipping COMPILER (never a vector)"; \
	  fi; \
	done; \
	if [ $$found -eq 0 ]; then \
	  echo "  no C++ compiler found — nothing was built, and that is a FAILURE,"; \
	  echo "  not a skip: an unbuilt suite proves nothing."; rc=1; \
	fi; \
	exit $$rc

# A parity test that silently passes because it could not find its data is the
# worst possible outcome here, so prove the opposite: point the suite at an
# empty directory and require a non-zero exit.
guard: build
	@mkdir -p $(BUILD)/no-vectors-here
	@echo ""
	@echo ">>> guard: a missing vector must FAIL, never skip"
	@rc=0; \
	for t in $(TESTS); do \
	  if SAANS_GOLDEN_DIR=$(BUILD)/no-vectors-here $(BUILD)/test_$$t >/dev/null 2>&1; then \
	    echo "  GUARD FAILED: test_$$t exited 0 with no vectors present"; rc=1; \
	  else \
	    echo "  ok  test_$$t refuses to run without its vectors"; \
	  fi; \
	done; \
	exit $$rc

# "Speed is a claim like any other" (SAANS_PLAN §4). The runner builds the
# benchmark, drives the same golden recording through it and through
# spec/golden/bench_python.py, and prints both with the ratio.
bench:
	@echo ""
	@echo ">>> bench: host throughput against the Python reference"
	@$(BENCH_DIR)/run_bench.sh --runs $(BENCH_RUNS) --python-runs $(PY_RUNS)

# The S6 broadcast-TDMA viability table. DELIBERATELY NOT part of `check`:
# it measures nothing and proves nothing, it prints a model's opinion about
# hardware that does not exist yet, and a report like that belongs where a
# human asks for it rather than in the output people read to decide whether
# the port is green. `test_tdma` is the part that gets checked.
#
#   make tdma-report                      the defaults, printed
#   make tdma-report TDMA_ARGS='46184 16880 175104 6404'
#                                         size_report's four columns: bytes per
#                                         link with BNR, without, the DIRAM
#                                         budget, and the 'once/node' arena.
#                                         Pass the fourth or the two reports
#                                         will disagree about the same node.
tdma-report: | $(BUILD)
	@printf '  CXX  %s  (%s)\n' 'tdma_report' '$(notdir $(CXX))'
	@$(CXX) $(CXXFLAGS) $(BENCH_DIR)/tdma_report.cpp -o $(BUILD)/tdma_report
	@$(BUILD)/tdma_report $(TDMA_ARGS)

# ---------------------------------------------------------------------------
# THE ONE COMMAND
# ---------------------------------------------------------------------------
# Runs every test under both toolchains, proves a missing vector fails rather
# than skips, and measures the port against the reference — then says in one
# line whether the whole thing is green. Each stage's own output is left
# intact above the summary; the summary is there so that "did it pass?" does
# not require reading 500 lines of margins.
check:
	@rc=0; tests=FAIL; guard=FAIL; bench=skipped; \
	if $(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) all-compilers; then tests=PASS; else rc=1; fi; \
	if $(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) guard; then guard=PASS; else rc=1; fi; \
	if [ "$(RUN_BENCH)" != "0" ]; then \
	  if $(MAKE) --no-print-directory -f $(firstword $(MAKEFILE_LIST)) bench; then bench=done; else bench=FAIL; rc=1; fi; \
	fi; \
	echo ""; \
	echo "################################################################"; \
	echo "##  edge/dsp  make check"; \
	echo "##"; \
	printf '##    golden-vector parity (g++ and clang++)   %s\n' "$$tests"; \
	printf '##    missing-vector guard                     %s\n' "$$guard"; \
	printf '##    benchmark vs the Python reference        %s\n' "$$bench"; \
	echo "##"; \
	if [ $$rc -eq 0 ]; then \
	  echo "##  RESULT: PASS — the float32 port matches wifisense/ on every"; \
	  echo "##          vector in spec/golden, under both toolchains."; \
	else \
	  echo "##  RESULT: FAIL — see above. Do NOT loosen a tolerance to fix"; \
	  echo "##          this; SAANS_PLAN §4 is explicit that a deviation"; \
	  echo "##          beyond the contract is a bug, not rounding."; \
	fi; \
	echo "################################################################"; \
	exit $$rc

# `clean` removes what THIS run owns. The bench tree is the exception and it
# has to be treated as one: run_bench.sh hardcodes bench/build, so a run with
# BUILD pointed elsewhere does not own that directory -- and `all-compilers`
# cleans before each compiler, so deleting it unconditionally is how one
# session removed another's bench binary between its mkdir and its link step.
# That surfaces as "cannot open output file ... No such file or directory" and
# reads exactly like a failing benchmark, which is the symptom the BUILD
# override at the top of this file exists to prevent. So the bench tree is
# cleaned only by a run at the default BUILD.
#
# This does not make two concurrent `make check` runs safe -- they still link
# the same bench binary path, because run_bench.sh does not take a build
# directory. It makes `make check BUILD=...` stop DESTROYING the other run,
# which is the part this Makefile controls.
clean:
	@rm -rf $(BUILD)
	@if [ "$(BUILD)" = "$(HERE)/build" ]; then rm -rf $(BENCH_DIR)/build; fi
