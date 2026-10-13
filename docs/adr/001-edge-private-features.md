# ADR-001: Raw CSI stays on the sensor

**Status:** accepted, 2026-10-13

## Context
Raw CSI from a room is enough to re-derive breathing, heartbeat and, with
several links, movement through walls. A server that receives it becomes a
store of health data whether or not anyone intended that, and under
India's DPDP Act 2023 that store carries consent, purpose and retention
obligations. Streaming raw CSI to a server, the common design, creates exactly that store.

## Decision
The ESP32 node runs the DSP and transmits only FeatureFrame v1 (68 bytes,
2 Hz, `spec/wire.md`). Breathing- and heart-band fields are zeroed **on
the node** unless the space has opted in. Raw raw CSI is a debug mode,
off by default; when on, every FeatureFrame carries the `raw_mode` flag so
the dashboard can show that it is on.

## Consequences
- A gateway compromise leaks motion levels and, for consented spaces,
  breathing rates, not radio fingerprints of the room.
- The DSP must run on an ESP32-S3: float32, no heap after init (ADR-003).
- New analyses that need raw data require a debug collection, recorded
  with `wifisense record`, under ethics approval. That friction is the point.
