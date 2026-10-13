# ADR-003: float32 on the edge, f64 on the gateway

**Status:** accepted, 2026-10-13

## Context
The ESP32-S3 has a single-precision FPU; double arithmetic is emulated in
software and an order of magnitude slower. The gateway has no such limit.

## Decision
`edge/dsp` computes in `float`. `saans-dsp` computes in `f64`, matching
the Python reference. Golden-vector tolerances: Rust 1e-9 relative on
filter outputs and 0.001 BPM on rates; C++ 1e-4 relative and 0.05 BPM.

## Consequences
- The near-0.1 Hz biquad and the Goertzel recurrence are the parts most
  sensitive to precision; the C++ port must check them against the
  vectors over long inputs, not just short ones.
- A frame computed on the node and the same frame computed by the gateway
  from raw mode differ in the last few significant digits. Detectors must
  not depend on exact equality.
