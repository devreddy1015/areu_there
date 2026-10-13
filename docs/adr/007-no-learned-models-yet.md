# ADR-007: No learned models until the DSP ceiling is measured on real data

**Status:** accepted, 2026-10-13

## Context
Published CSI accuracies above 90% are in-domain ceilings (RESEARCH.md §4).
Every Saans capability so far is explainable DSP whose failure modes we
can name: blind spots, intermodulation, packet loss.

## Decision
No neural networks or trained classifiers in the S0-S5 path. A learned
component may be added only after S5 measures the DSP version on real
rooms, and then only if it beats that measurement leave-one-room-out.
