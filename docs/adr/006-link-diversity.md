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

## Amendment, 2026-10-24: the octave rule
"Breathing from the link with the highest confidence" is no longer the
whole rule. Before choosing, any link whose rate is within 1 BPM of twice
another link's is set aside: a Fresnel null turns a breather at `f` into
`2f` at high confidence, and nothing turns `f` into `f/2`, so the higher
of an octave pair is the null. Hysteresis does not keep a link the rule
has set aside.

This is a consequence of the decision above, not a change to it -- it is
the second link earning its place in a case the original text did not
name. Measured on identical simulated scenarios (`NOTES.md` §10), doubled
readings fall from 6.9% to 3.4% with one link in a null and from 25.2% to
16.5% with both, with no class made worse. The 1/2/3-link blind-spot
measurement this ADR's consequences require is still owed.
