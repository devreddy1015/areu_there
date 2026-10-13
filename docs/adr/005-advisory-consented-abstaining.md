# ADR-005: Health outputs are advisory, consented and allowed to abstain

**Status:** accepted, 2026-10-13

## Decision
1. Vitals (breathing, heart, pauses, rate alarms, trends) are computed and
   transmitted only for spaces with `vitals = true`, set by the space owner
   with written consent of occupants.
2. Every output carries a confidence. Below its threshold the system says
   "unknown"; it never fills in a plausible number.
3. Alerts (inactivity, possible collapse, breathing pause) are advisory.
   They state what was observed ("no gross motion for 22 min, breathing
   detected at 13 BPM, confidence 0.8") and never replace existing safety
   procedures.
4. No identities anywhere: outputs are per space, not per person.
5. Every detector reports both its miss rate and its false-alarm rate per
   hour; publishing one without the other is not allowed.

## Consequences
Coverage will look worse than systems that always answer. That is the
price of being believable.
