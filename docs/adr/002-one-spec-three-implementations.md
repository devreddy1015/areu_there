# ADR-002: One spec, three implementations, golden vectors

**Status:** accepted, 2026-10-13

## Context
The DSP must exist in C++ (on the node), Rust (gateway, for raw-mode
links and replay) and Python (where research happens). Three
implementations drift apart silently unless something forces agreement.

## Decision
The Python package `wifisense` is the reference. `spec/golden/generate.py`
records its outputs on fixed inputs for every stage. Rust and C++ test
suites must reproduce them within the tolerances in `SAANS_PLAN.md` §4.
A behaviour change starts in Python, regenerates the vectors, and then
the ports follow.

## Consequences
- Porting bugs surface as a named vector failing, not as a field anomaly.
- The reference cannot be "improved" in one port only.
- Python stays slow; that is acceptable for a reference.
