# ADR-004: FeatureFrame v1 wire format

**Status:** accepted, 2026-10-13

## Decision
A fixed 68-byte little-endian UDP datagram, magic bytes "SANS", version 1,
CRC-32 trailer; fields and gating in `spec/wire.md` §2. One frame per link
every 0.5 s, clocked by processed CSI frames so it is deterministic.

## Why fixed and binary
The node is a microcontroller; fixed offsets need no allocator and no
parser. 68 bytes x 2 Hz x 12 links is 1.6 kB/s per room, negligible next
to the CSI traffic itself. A version byte leaves room to change.

## Why these fields
Each health and system-health detector in `SAANS_PLAN.md` §2 reads only
these; nothing in them reconstructs the channel (ADR-001).
