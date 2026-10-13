# Saans edge DSP — forwarder. The real harness is test/Makefile.
#
#   make check      everything: both compilers, the missing-vector guard,
#                   and the benchmark against the Python reference
#   make            build and run every test once
#   make bench      host throughput vs spec/golden/bench_python.py
#   make clean
#
# WHY THIS FILE IS A FORWARDER AND NOT A SECOND HARNESS.
#
# It used to be one: it globbed test/*.cpp and passed the golden directory as
# argv[1], while test/Makefile named its tests and passed it as
# -DSAANS_GOLDEN_DIR. Two sessions were working in this tree at the time and
# each grew the harness it needed. Two harnesses is one too many -- they built
# into different directories, had different notions of which tests existed,
# and only one of them had the two-compiler run and the missing-vector guard.
# A test that only one harness built is a test nobody notices is missing,
# which is the same failure mode as a parity test that skips.
#
# So there is one harness now, and this is the door into it from the directory
# people's shells are most often in. Every target forwards unchanged, so
# `make -C edge/dsp check` and `make -C edge/dsp/test check` are the same run.

HERE := $(patsubst %/,%,$(dir $(abspath $(lastword $(MAKEFILE_LIST)))))

.PHONY: all check test build all-compilers guard bench clean

all check test build all-compilers guard bench clean:
	@$(MAKE) --no-print-directory -C $(HERE)/test $@
