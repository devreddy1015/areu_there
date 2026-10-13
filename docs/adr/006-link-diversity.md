# ADR-006: Link diversity over subcarrier selection for blind spots

**Status:** accepted, 2026-10-13

## Context
P0a (NOTES.md §6) measured that on an ESP32 HT20 channel (17.5 MHz) a
Fresnel-zone blind spot usually covers the whole band at once: RespirFi's
own bound needs ~4.4 m of excess path to guarantee an escape. Subcarrier
selection recovered whole-band-null accuracy only from 64.5% to 68.1%.

## Decision
Spaces where breathing matters get at least two links at different angles.
The gateway fuses per space: occupied if k of n usable links agree;
breathing from the link with the highest confidence, with hysteresis so
the reported source does not flap. RespirFi stays available per link.

## Consequences
- Twice the nodes per monitored space.
- S6 must measure the blind-spot rate with 1, 2 and 3 links, in simulation
  (`saans-sim`) and then on hardware, before this is relied on.
