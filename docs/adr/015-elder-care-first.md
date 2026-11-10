# ADR-015: Lead with elder care

**Status:** accepted, 2026-11-06. **Decided, not built.** The signals,
the relay and the phone app this decision calls for are issues #38-#42;
nothing in the code changes with this ADR. The spec it points to is
[`docs/ELDER_CARE.md`](../ELDER_CARE.md).

## Context

Saans has been framed for campus spaces. `CAMPUS_SENSING_PROMPT.md`
sets the goal, and `SAANS_PLAN.md` §10 names the users: facilities,
wardens and lab safety, IT, governance. ADR-011 decides one gateway per
site, and the MQTT output exists so building management can follow
occupancy. None of that names a first customer who would pay for it and
act on its alerts.

What the evidence supports today is narrow, and every number behind it
is **SYNTHETIC** (`docs/EVIDENCE.md` §1: no board has run this firmware):

- **It works:** telling an occupied room from an empty one, including a
  person sitting still, and a breathing rate that abstains when unsure.
  On held-out simulated runs, a still person reads as empty 2.1% of the
  time on one link and 0.4% on two (`docs/RESULTS.md`, "S3 health
  detectors, held-out seeds").
- **It does not work:** telling a fall from a sit-down. The
  possible-collapse detector fired on 36 of 36 simulated falls and on 36
  of 36 sit-downs (same section, 2-D slice; 33 and 35 of 36 in the 2.7 m
  box), because amplitude-only WiFi cannot see height
  (`docs/CAPABILITY_LIMITS.md` §1). It ships off.
- **It cannot be made to work on this hardware:** pose, counting people,
  telling people apart (`docs/CAPABILITY_LIMITS.md`).

On 2026-11-06 the founder chose elder care as the first customer: an
older person who lives alone, with alerts to family. This ADR records
that choice and what it changes. The reasons are about fit with the
evidence above. They are not market data; the competitor and
alternatives scan is issue #48 and has not been done.

### Why elder care fits what the system can do

1. **The rooms that matter are the ones where cameras are refused.**
   The bathroom and bedroom are where many falls at home happen
   (`docs/ELDER_CARE.md` §2 cites the studies) and where a camera is not
   acceptable. A sensor that sees presence and stillness without an
   image is most useful exactly there.
2. **One occupant removes the problem the system cannot solve.** Saans
   has no identities and cannot count people (ADR-005). In a shared lab,
   "someone is still" says nothing about who. In a home where one person
   lives alone, a room's state is that person's state most of the time.
   Visitors break this, and the spec says how.
3. **The safety signal can rest on time, which the system measures,
   instead of posture, which it cannot.** A long lie, "present and not
   moving for N minutes after a sudden movement", needs only presence,
   stillness and a motion burst, all of which exist. The bet is that a
   person who sat down heavily moves again within minutes and a person who
   fell and cannot get up does not. That is a hypothesis, not a finding:
   a nap after lying down quickly looks the same, and issue #38 measures
   how often.
4. **The alert has a recipient who wants it.** A family member far away
   wants to know. In a campus space the occupant often does not want to be
   watched, and the person paged has no relationship with them.
5. **Consent is one person's, about their own home.** A shared space has
   many occupants, and some of them never agreed.

### Other first customers considered

- **Campus lone-worker and lab safety** (the original framing). The
  inactivity alert fired in 36 of 36 simulated 40-minute reading
  sessions (`docs/RESULTS.md`, same section, 2-D slice; 35 of 36 in the
  box): correct by its definition,
  and a page every time someone reads. The buyer is an institution,
  consent involves every occupant, and nobody in particular wants the
  alert.
- **Care homes.** Many rooms, staff on site, institutional buyer. Closer
  to the campus model, and the code's roles and routes already fit it.
  It is the second customer in the spec, not the first, because a pilot
  there needs an institution's approval before a demo can be shown.
- **Hostels and student wellbeing.** Shared rooms break the
  one-occupant assumption, and the alerts would concern mental health,
  which this system cannot see.

## Decision

1. **The first customer is an older person living alone at home, with
   alerts to family.** `docs/ELDER_CARE.md` is the product spec.
   Care homes come second. The code stays general: spaces, roles and
   routes do not become elder-care-only.
2. **The lead safety signal is a long lie, not fall detection.** Possible
   collapse stays off by default, as `SpaceConfig` already has it
   (`collapse_detection: false`, `saans-health/src/types.rs`). Nothing
   Saans ships or says claims to detect falls.
3. **The home gateway stays in the home.** ADR-001 is unchanged: raw CSI
   stays on the node and FeatureFrames stay on the home network. A cloud
   relay carries to family phones only what the spec's privacy section
   (§5.2) lists: alerts and their acknowledgements; sensing status,
   including a heartbeat that says the gateway is alive and nothing
   about the rooms; answers to a family member's "home now" request;
   and, if the monitored person allows it, a daily summary and the
   breathing trend. It carries no stream of room state.
4. **The monitored person is the one who consents.** The family member
   who buys the kit does not consent for them. The monitored person can
   pause sensing and can see what their family sees.
5. **Vitals stay off unless that person opts in for a room, and heart
   rate stays off.** Breathing pauses and rate alarms are not family
   alerts in the first version (spec §3.5).
6. **Wellness and safety wording only.** No disease names, no "fall
   detection", no "patient", no clinical users (spec §6).

## Consequences

- **A cloud component exists for the first time.** The home gateway sits
  behind home NAT and has no public address, so a relay has to carry
  alerts out (issue #40). The gateway connects out to the relay and
  never listens for it, the same pattern the Telegram channel already
  uses with long polling (`SAANS_PLAN.md` §10.3).
- **The relay must notice silence.** A gateway that has lost power or
  internet cannot report its own failure, so the relay, not the gateway,
  tells the family "no contact from the home since 14:05". Without this,
  a dead kit reads as a quiet day.
- **ADR-011 stands, and the relay must not erode it.** ADR-011 rules out
  a hosted control plane because aggregated space state across sites is
  its own privacy object. A relay that stores daily summaries for many
  homes is a version of that store. Issue #40 has to decide whether the
  relay can read what it carries; end-to-end encryption from the home
  gateway to the family's phones would keep ADR-011's argument intact,
  and is the option to beat.
- **Saans, the company, becomes a Data Fiduciary** under India's DPDP
  Act 2023 for what leaves the home, with the duties listed in spec §5.4.
  Until now no data left a site.
- **A phone app or channel has to be chosen** (issue #41). Telegram
  exists and is tested against a local stand-in only (`docs/EVIDENCE.md`
  §2.6). Whether it is acceptable even as an interim channel for the
  demo is an open question in the spec.
- **New detectors:** long lie (#38) and the daily routine summary (#39).
  The gateway also needs a home-level view across its spaces: "home",
  "away", "last seen in", and ending a presence hold when the person
  shows up in another room. A home is one gateway, so this is site
  level, not a new tier.
- **A severity rule changes.** Today the inactivity alert can only reach
  `Check` when vitals are on and breathing is not detected
  (`saans-health/src/detectors.rs`, `inactivity_step`). A long lie has
  to reach `Check` with vitals off, because vitals are off by default.
- **ADR-005's rule is restated in home units.** Every elder-care alert
  reports its false alerts per home per week next to its miss rate, from
  simulated homes first and staged real ones later. The simulator needs
  multi-room, multi-day home timelines it does not have.
- **The medical-device line is closer than it was for campus spaces.**
  India's definition of a medical device covers monitoring of "any injury
  or disability", and CDSCO's software guidance asks an intended-use
  statement to name its patient population, with "geriatric" as an
  example (spec §6). The spec
  lists the wording that stays on the wellness side, and a written
  regulatory opinion is needed before the first paid pilot.
- **Docs written for campus need rewording:** README and
  `SAANS_PLAN.md`. That is listed in the PR that adds this ADR and is
  done after the consolidation (#32) merges.

### What would reverse this

- Pilot families find the false-alarm rate unacceptable at any N that
  still catches a long lie in useful time, and nothing available tells a
  nap from a fallen person.
- A regulatory opinion says the product is a medical device under any
  honest wording, and the cost of that route is out of reach.

Either result is a reason to write a new ADR, not to edit this one.

## Amendment, 2026-11-10
[ADR-018](018-daily-picture-first.md) changes points 2 and 5. The long
lie was measured in simulation (PR #60) and alerts on most ordinary
stillness, so it is one alert that waits for issue #62, not the lead
signal. The night's breathing trend becomes part of the product, with
the monitored person's agreement at setup. The rest of this decision
stands, and its body is left as it was written.
