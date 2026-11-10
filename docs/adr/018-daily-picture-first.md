# ADR-018: Lead with the daily picture

**Status:** accepted, 2026-11-10. **Decided, not built.** It changes the
order of the product more than the code: the night's breathing trend,
the rest period and the restlessness count already exist
(`saans-health`), and the routine is PR #61. The morning message and
the "day did not start" alert are new work. It amends
[ADR-015](015-elder-care-first.md) points 2 and 5.

## Context

ADR-015 made the long lie the lead safety signal and kept breathing
behind an opt-in, shown as a line that is never pushed
(`docs/ELDER_CARE.md` §3.4, §7.3). Four things have become clear since.

- **The long lie was measured, and it cannot lead.** On simulated
  timelines it catches most falls. It also alerts on nearly every
  person who sits down hard and stays still, on most readers and on
  most people who get into bed quickly (`docs/RESULTS.md`, "Long lie:
  miss rate and false alarms together, by N"; issue #62). A sudden
  movement followed by stillness looks the same on the floor, in a
  chair and in bed.
- **The same evidence says what this hardware is best at:** telling an
  occupied room from an empty one, and the breathing rate of a still
  person (`docs/RESULTS.md`, "S3 health detectors, held-out seeds").
  The product led with its weakest signal and deferred its strongest,
  the one it is named after.
- **A sleeping person is the best case for that signal.** They are
  still for hours, in a place known in advance, and a link can be put
  across the bed. The night's figure is a median that needs about five
  minutes of confident samples out of a whole night
  (`DailySummary::resting_bpm_median`), so it survives the windows that
  a blind spot or a turn in bed spoils.
- **An alarm is the wrong thing to lead with for a second reason.** It
  is silent for months and is then judged on one event. A family needs
  a reason to open the product on the days nothing is wrong, and the
  monitored person needs it to be about their wellbeing, not only their
  worst day.

None of this is measured on hardware. No board has run the firmware
(`docs/EVIDENCE.md` §1). The breathing trend itself has never been
evaluated either, because no multi-day timeline exists yet
(`docs/ELDER_CARE.md` §3.4); PR #61 adds simulated days.

## Decision

1. **The product leads with the daily picture.** Every morning the
   family gets one short message built from the night (time in bed,
   times up, restlessness, the night's breathing rate against the
   person's own baseline) and from the day before (up, active hours,
   rooms). When all is as usual, it says so in words.
2. **The bedroom comes first.** The first link in a home goes across
   the bed. Its place is declared at setup, not inferred (ADR-013).
3. **The night's breathing trend is part of the product.** The monitored
   person agrees to it once, at setup, with the rest of the kit, and can
   pause it. This amends ADR-015 point 5 for the nightly trend only.
   Breathing pauses and rate alarms stay out of family alerts, on their
   simulated false rates (`docs/ELDER_CARE.md` §3.5), and heart rate
   stays off.
4. **Alerts come in the order their evidence allows:** the day that did
   not start (not up within a margin of the person's own usual time);
   no movement anywhere while the person is home; then the long lie.
   The long lie stays off as a family alert until issue #62 gives it a
   second cue, or a check-in asks the person before the family is
   paged. This amends ADR-015 point 2: the long lie is one alert, not
   the lead signal.
5. **The first homes use the channel that exists.** The morning message
   and the alerts go out through the gateway's Telegram channel, which
   is tested against a local stand-in only (`docs/EVIDENCE.md` §2.6).
   WhatsApp is added only if a pilot family cannot use Telegram. The
   relay and the native app (PRs #54, #55, #57) wait until a pilot
   shows that is not enough.
6. **Real nights before more simulation.** No new simulation study and
   no model training starts before the first real nights are recorded.
   The offline learned-model line (ADR-014) resumes when there are real
   recordings to train and test on.

## Consequences

- **`docs/ELDER_CARE.md` §3.3-§3.5, §4 and §7 are revised to this
  order** in a follow-up. Until then, where the spec and this ADR
  differ, this ADR wins, and the spec says so at its top.
- **Little new code.** New: composing the morning message from
  `DailySummary` and the routine, and sending it through the existing
  alert channels; the "day did not start" alert on the routine's
  baseline; a bed link in the home-kit questions. Each gets an issue.
- **The first real recording (#46) becomes a bedroom**, and its gate is
  nights, not one session: how many nights give a trend, and how the
  night's median compares with a reference.
- **The demo (#47) changes** to a night, its morning message and a day
  that does not start. A staged long lie is in it only if #62 lands.
- **Issue #62 gets its first candidate from the hardware, not from the
  simulator:** two links on one wall at different heights, to see
  whether the link that carries the breathing tells the floor from a
  chair (`docs/HOME_KIT.md`, "First nights and the floor test").
- **Known risks to the night,** all from simulation: the receiver's
  gain steps, which may hide a still person (`NOTES.md` §9; the bench
  session #45 settles it); a doubled rate when a link sits in a blind
  spot (`docs/CAPABILITY_LIMITS.md` §5), which placement and the median
  have to absorb; and a second person or a pet in the bedroom, which
  the first homes do not have.
- **The relay and app decisions are not undone.** PRs #54, #55 and #57
  stay open as designs for the stage that needs them (`SAANS_PLAN.md`,
  "Where this is going").
