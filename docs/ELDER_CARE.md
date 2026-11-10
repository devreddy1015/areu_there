# Elder care: the product spec

**Status:** draft for issue #37, 2026-11-06, revised 2026-11-10 for
issue #68. It follows [ADR-015](adr/015-elder-care-first.md) and
[ADR-018](adr/018-daily-picture-first.md). Issues #39, #47, #62, #66
and #67 build on this file.

Saans for elder care is a few small sensors in the home of an older
person who lives alone, and a gateway on the same home network. Every
morning their family gets one short message about the night (time in
bed, times up, restlessness, the breathing rate against the person's
own usual) and about the day before. When nothing is wrong, it says so
in words. When the day does not start, or nothing moves while the
person is home, the family gets an alert. There is no camera, no
microphone and nothing to wear. The raw radio data stays on the sensor
that measured it, except in a debug mode that is off by default and
flagged in every frame when on (ADR-001).

**Every performance number in this file is SIMULATED.** No ESP32 has
run this firmware and no real radio has been through this pipeline
(`docs/EVIDENCE.md` §1). Markings:

| Mark | Means |
|---|---|
| **SIM** | Produced by a simulator in this repository. Evidence about the model, not about real rooms, radios or bodies. |
| **REAL** | Measured on hardware. There are none yet. |
| **[sourced]** | From outside this project: a paper, a law, a regulator. Named in §9. |
| **[proposal]** | A number or rule this spec suggests. Not measured; the founder or the named issue decides. |
| **[UNVERIFIED]** | Something I could not check against a primary source. |

Code facts name a file; repository numbers name the section of
`docs/RESULTS.md` (generated; do not edit by hand) or `NOTES.md` they
come from.

---

## 1. Who

### 1.1 The monitored person

An older person who lives alone in their own home. They agree to the
sensors being installed, they can pause sensing, and they can see
everything their family is shown about them (§5.5). They are not a
patient and Saans never calls them one.

What they want from it:

- To keep living alone at home without being watched.
- For it to be about their wellbeing, not only their worst day.
- If they fall and cannot get up, for someone to find out within the
  hour, without having to press a button or wear anything.
- Not to have their children worrying, or calling, over nothing.
- To switch it off when they want privacy, and to know when it is on.

The alert for the third of these, the long lie, stays off for now
(§3.3), so Saans does not yet stand in for a pendant alarm.

### 1.2 Family caregivers

Adult children or other relatives, often in another city or country.
They receive the morning message and the alerts, and decide what to do:
call, ask a neighbour to look in, travel.

What they want:

- To know their parent is up and about without phoning every day.
- To be alerted quickly when something may be wrong, and only then.
- To know whether an alert needs them now or can wait.
- To hand an alert to someone nearer: a sibling, a neighbour.

### 1.3 Later: care-home staff

A care home has many rooms, staff on site and an institutional buyer.
The existing campus design (spaces, roles, alert routes, escalation)
fits it more closely than it fits a family. It is the second customer,
not the first (ADR-015), and nothing in this spec is designed for it.

### 1.4 Who is not a user

Doctors, nurses and other clinicians. Saans sends nothing to clinicians
and makes no claim to inform clinical decisions. That keeps the intended
users on the non-clinical side of the line in §6.

---

## 2. Where: the home

A home kit senses **2-4 rooms**, each a space in `saans.toml`. The
bedroom comes first (ADR-018): the first link in a home goes across the
bed, and its place is declared at setup, not inferred (ADR-013).

| Room | Why it is sensed | Links [proposal] |
|---|---|---|
| Bedroom | The night (§3.4). Falls happen here, often at night; a camera is not acceptable here | 2, the first across the bed |
| Bathroom | Falls happen here; a camera is not acceptable here | 2 |
| Living room | Most waking hours; usually next to the front door | 1-2 |
| Kitchen | Daily routine (meals) | 1, optional |

**Why the bathroom and the bedroom.** Studies of falls by older people
put most falls indoors and at home, and the bedroom and bathroom among
the most common rooms; which of the two ranks first varies by study and
setting [sourced: Moreland et al. 2021; AIHW]. These are also the two
rooms where cameras are not acceptable, which is where a sensor that
produces no image has most to offer.

**Why two links in those rooms.** ADR-006 asks for two links wherever
breathing matters, and the measurement behind it applies to a person on
the floor too. On one link a still person reads as empty 2.8% of the
time and 44 of 1,200 simulated seats are mostly missed; on two links,
0.0% and 0 of 1,200 (**SIM**, `docs/RESULTS.md`, "What each extra link
buys"). For someone lying on the floor, one link misses spots to the
side of the link line (3.8% of the still period at 1.0-1.5 m from it);
two links read 1 of 400 lying spots mostly empty (**SIM**, "A fallen
person read as an empty room"). How many nodes that takes per room is
for the hardware issue (#44).

**What a home has that the simulator does not** (none of these is
modelled or measured, and the repository does not mention fans or pets):

- **Walls shared with neighbours.** A link cannot tell which side of a
  wall a person is on (`docs/CAPABILITY_LIMITS.md` §2d). In a flat, a
  neighbour moving next door can read as presence. The simulator has no
  neighbouring room (`docs/RESULTS.md`, "What each extra link buys").
- **Ceiling fans** move constantly while they are on.
- **Pets** move, and Saans cannot tell a cat from a person.
- **Bathrooms:** water, tiles, mirrors and metal fittings.
- **Visitors.** Saans cannot count people (§5.3).

---

## 3. Signals

The daily picture leads (ADR-018): the night (§3.4), then the day
(§3.1, §3.2), and the morning message that carries both to the family
(§4.1). Alerts follow, each when its own evidence allows (§4.1). The
subsections keep their old numbers, because other files cite them.

Each signal: what it is, what it builds on, what is measured (all
**SIM**), and what is not yet built.

### 3.1 Room presence, and home or away

**Definition.** Each room is occupied, empty or unknown, twice a
second. That exists. From the rooms, the home is [proposal]:

- **home:** at least one room is occupied;
- **away:** every room has read empty for 20 minutes, and the last room
  occupied was the one marked as the way out (usually the living room by
  the front door);
- **not seen:** every room reads empty, but the last room occupied was
  not the way out. The person may be in a room with no sensor, or lying
  where the links see them weakly (§3.3);
- **unknown:** any room is blind (`occupied == None`). Unknown is never
  shown as away.

Without a door sensor, "left the home" and "went to the balcony" look
the same. That is open question 2.

**Builds on.** Presence per link, fused per space with a k-of-n quorum
(ADR-006, `saans-health/src/fusion.rs`); occupancy sessions
(`OccupancyStarted`, `OccupancyEnded`).

**Measured (SIM**, `docs/RESULTS.md`, "S3 health detectors, held-out
seeds", 252 runs, 187 simulated hours):

| | 1 link | 2 links | 3 links |
|---|---|---|---|
| Occupancy accuracy | 98.2% | 99.5% | 98.9% |
| Still person read as empty | 2.1% | 0.4% | 1.3% |
| Empty room read as occupied | 1.0% | 1.2% | 0.6% |

In the 2.7 m box model, a still person reads as empty 1.6%, 0.1% and
1.5% of the time (same runs, "in a 2.7 m box").

**Not built.** The home-level view: home, away, not seen, "last seen in
the kitchen at 10:40". A home is one gateway, so this is a view across
that gateway's spaces.

### 3.2 Activity and daily routine (issue #39)

**Definitions** [proposal; #39 settles them]. Per home, per day, using
the gateway's local day (`utc_offset_minutes`, 330 for India):

- **Got up:** when the night's rest period in the bedroom ends and is
  followed by at least 10 minutes with motion somewhere in the home.
- **Active hours:** hours of the day with at least one motion bout in
  any room.
- **Night bathroom visits:** bathroom occupancy sessions that start in
  the night window (default 22:00-06:00, set per home).
- **The day did not start** (issue #67, not built): the person is not
  up within a margin of their own usual getting-up time, while the home
  reads as occupied. It needs a baseline of several days, and it never
  fires on a day the bedroom was not observed.
- **No movement while home** (unusual quiet): a period in the day
  window, while the person is home, with no motion bout anywhere,
  longer than the person's own 95th-percentile quiet period for that
  time of day over the last 28 days. Needs at least 14 days of the
  person's data; until then it is not computed.

Each value is compared with the person's own baseline, never with other
people. The robust baseline the breathing trend already uses (median of
up to 28 previous days, flagged beyond the larger of 3 × 1.4826 × MAD
and a fixed floor, `saans-health/src/detectors.rs`) is the starting
point.

**Builds on.** Motion bouts (`BOUT_INDEX` 3.0, `BOUT_GAP_S` 5 s),
stillness (`STILL_INDEX` 2.5, 3 confirming samples), occupancy
sessions, rest periods (`REST_MIN_S` 1200 s), and the per-space
`DailySummary` (`occupied_s`, `motion_bouts`, `pauses`) kept in
`history.json` for `history_retention_days` (90 by default).

**Measured (SIM).** Only rest periods: at least one was reported in 31
of 36 night runs ("S3 health detectors, held-out seeds") and 36 of 36
in the box model. Nothing at the level of a routine: the simulator has
no multi-room or multi-day timelines.

**Not built.** All of it (#39). It is advisory: it describes a day and
diagnoses nothing.

### 3.3 Long lie: one alert, off until issue #62 (issue #38)

The long lie is one alert, not the lead signal (ADR-018). It was
measured in simulation (PR #60) and alerts on most ordinary stillness:
a sudden movement followed by stillness looks the same on the floor, in
a chair and in bed. It stays off as a family alert until issue #62
gives it a second cue, or a check-in asks the person before the family
is paged (open question 9).

**Definition** (`saans-health/src/longlie.rs`; off by default,
`long_lie_min`). In one room, an urgent (`Check`) alert when a sudden
movement (motion index at least 8, `BURST_INDEX`) stops within 5 s
(`COLLAPSE_SETTLE_S`), the room then stays present without gross motion
for N minutes, and nobody is seen moving in another room in that time
(`long_lie_elsewhere`). It needs no vitals. Gross motion in the room
after the alert updates it ("movement seen at 06:31"). The alert stays
open until a person resolves it, because moving does not mean being
fine.

**N.** The clinical term "long lie" usually means an hour or more on
the floor, although there is no agreed definition [sourced: Physiopedia;
Kubitza et al. 2023], and about one fall in five among people over 65
becomes a long lie [sourced: Sheffield Long Lies Study]. N has to be
well under an hour. There is no default N.

**Why not fall detection.** The possible-collapse detector fired on 36
of 36 simulated falls and on 36 of 36 sit-downs (**SIM**, "S3 health
detectors, held-out seeds"; 33 and 35 of 36 in the 2.7 m box).
Amplitude-only WiFi cannot see height, so a fall and a heavy sit-down
look the same (`NOTES.md` §8.2, `docs/CAPABILITY_LIMITS.md` §1). It
ships off and stays off.

**The person who vanishes.** If the room reads empty, the long lie
never starts. A lying person is read as empty about as often as an
upright one, but the misses move to the side of the link line (§2;
**SIM**, "A fallen person read as an empty room"). A presence hold
after a fall-shaped motion exists, off by default (`fall_hold_min`):
about one departure in four engages it (**SIM**, `NOTES.md` §14, which
also lists what the model leaves out).

**Measured (SIM**, `docs/RESULTS.md`, "Long lie: miss rate and false
alarms together, by N"; read in `NOTES.md` §15). 120 simulated runs per
scenario in the 2.7 m box, presence hold off:

- falls that end still: 6.7% missed at every N from 3 to 30 minutes,
  and 2/40 on two links at N = 15 (the staged target is 1 in 20);
- slow slides to the floor: 46.7% missed at N = 15, because most never
  reach `BURST_INDEX`;
- a heavy sit-down and then 45 minutes still: alerted in 118 of 120 at
  N = 30, and no N fixes it;
- lying down into bed quickly and turning over now and then: alerted in
  94 of 120 at N = 15;
- walking in and reading for 40 minutes: alerted in 101 of 120 at every
  N, because walking in reaches `BURST_INDEX` too; a two-hour fidgeting
  night: 78 of 120 at N = 15;
- an empty room: never; a walk-out with the presence hold on: 33 of 120
  alert on the empty room, and none when the person is seen moving in
  another room (`long_lie_elsewhere`).

### 3.4 The night: breathing trend, time in bed, times up, restlessness

Breathing is the signal Saans is named after, and a sleeping person
should be its best case: still for hours, in a known place, with a link
across the bed (§2). The night's breathing trend is part of the product
(ADR-018): the monitored person agrees to it once, at setup, with the
rest of the kit, and can pause it (§5.5). In the config that is
`vitals = true` for the bedroom (§5.1).

- **Breathing trend.** The median resting breathing rate in the bedroom
  (`DailySummary::resting_bpm_median`, which needs 600 confident still
  samples, about 5 minutes). Today it is one number per calendar day,
  so a night falls into two; one number per night is part of #66. It is
  compared with the person's own baseline: the median of up to 28
  previous days, after at least 3, flagged when it differs by more than
  the larger of 3 × 1.4826 × MAD and 2 BPM (`TrendDeviation`,
  `saans-health/src/detectors.rs`).
- **Time in bed and restlessness.** The night's rest periods in the
  bedroom (`RestEnded`): how long they lasted, and the motion bouts in
  them. A rest period is a long still period, not sleep staging.
- **Times up.** Not defined yet; #66 settles it.

Each goes into the morning message against the person's own usual
(§4.1). A night that differs is a line in that message, never an alert
(`TrendDeviation` is `Info`).

**Measured (SIM).** Breathing rate on a still breather (**SIM**, "S3
health detectors, held-out seeds" and "in a 2.7 m box"):

| | 1 link | 2 links | 3 links |
|---|---|---|---|
| reported (coverage) | 93.6% | 100.0% | 100.0% |
| within 1 BPM | 93.9% | 100.0% | 100.0% |
| reported, box | 98.3% | 100.0% | 99.7% |
| within 1 BPM, box | 91.7% | 96.5% | 100.0% |

The confidence is a gate, not a probability. Its calibration error is
0.064 in ordinary placements and 0.154 at whole-band nulls, where the
estimator can report twice the true rate (`docs/CAPABILITY_LIMITS.md`
§5, from `NOTES.md` §12). With both of two links in a null, 16.5% of
reported samples are at twice the true rate (**SIM**, "Doubled
breathing rate at a null, two links").

No simulation here has a bed (`NOTES.md` §15), and the trend has not
been evaluated on a multi-day timeline. The first real nights (#46)
test both: how many nights give a trend, and how the night's median
compares with a reference (ADR-018).

### 3.5 What is not a family alert in the first version

| Detector | Exists | Why it is not a family alert |
|---|---|---|
| Breathing pause | yes, `Check` when ongoing over 20 s or ending in an empty reading | An alert on a physiological value is the clearest "medical purpose" in §6. Its false rate on simulated nights is 0.12 per hour on every layout in the 2-D slice ("S3 health detectors, held-out seeds"), which is about one false pause per 8-hour night by arithmetic. |
| Fast or slow breathing | yes, `Advisory` | An alert on a physiological value (§6). |
| Heart rate | built, off | Not trusted: 80% within 3 BPM ungated; gated, 97% correct but only in 61% of cases (**SIM**, `docs/RESULTS.md`, "Original 64-scenario sweep"; `docs/EVIDENCE.md` §6.4). Stays off. |
| Possible collapse | yes, off | Fires on sit-downs as often as falls (§3.3). |
| Inactivity, as it is | yes, `Advisory`/`Check` | Fired for every simulated 40-minute reader (36 of 36 in the 2-D slice, 35 of 36 in the box). A person who reads or naps still for 40 minutes once a day would cause about 7 alerts a week, by arithmetic. No movement while home (§3.2) is the family alert in its place. |

They all stay in the gateway and on the local dashboard. The family's
alert route has to name the kinds it pages (`kinds`,
`saans.example.toml`): a route that names none pages every `Check`, a
breathing pause included.

### 3.6 Sensing status

A safety product must never let silence look like a quiet day.

- **A room cannot see:** `SpaceBlind`, `LinkDown`, `CalibrationStale`
  exist (`saans-health/src/types.rs`, `system.rs`). The family sees
  "Saans cannot see the bathroom since 14:05" as a heads-up after 15
  minutes [proposal].
- **The home is silent:** a gateway that has lost power or internet
  cannot say so. The relay (#40) would notice missing heartbeats and
  tell the family "No contact with the home since 14:05" after 10
  minutes [proposal], but the relay waits (§7.3). Until then this is a
  gap: the only sign is a morning with no message.
- **Raw radio mode is on:** `RawModeActive` exists; the person and the
  family both see it.

---

## 4. The morning message and alerts

### 4.1 The morning message, and the alert levels

**The morning message is not an alert.** It comes every morning,
whatever happened, with the night (§3.4) and the day before (§3.2),
each against the person's own usual. When all is as usual, it says so
in words. A value that was not observed is called unknown, never shown
as zero. Not built (#66).

**Alerts** come only when something may be wrong, in the order their
evidence allows (ADR-018):

1. **The day did not start** (§3.2). Not built (#67). Its level is not
   decided.
2. **No movement while home** (§3.2). Not built; it needs 14 days of
   the person's data.
3. **The long lie** (§3.3). Built, and off until issue #62.

| Level | Saans severity | Examples | On the phone | Quiet hours |
|---|---|---|---|---|
| **Urgent** | `Check` | The long lie | Notification with sound, repeated until someone acknowledges it [proposal] | Never held |
| **Heads-up** | `Advisory` | No movement while home; not seen for long; a room cannot see | Ordinary notification | Held until quiet hours end |

### 4.2 What a family member sees

In the first homes these are Telegram messages (ADR-018). The wording
is a [proposal]; #66 and #67 settle theirs. The morning message, on a
day when nothing is wrong:

> **Saans · Amma's morning**
> All as usual.
> Night: in bed 22:40 to 06:10, up once, restful. Breathing as usual.
> Yesterday: up at 06:20, active through the day, in every room.

When a value differs from the person's own usual, the message names
it. No movement while home:

> **Saans · Home**
> Quieter than usual: no movement seen since 11:20. Usually active by now.

With the app, which waits (§7.3), comes "Home now": home, away, not
seen or unknown, and the last room with movement, fetched when a family
member opens it, not streamed. Every look is written to the audit log
the monitored person can read [proposal].

An alert carries no breathing numbers and no evidence sentences, unless
the monitored person has turned vitals on for that room and turned on
sharing details. The gateway already keeps details at home by default
(`include_health_details = false`, `saans.example.toml`). The morning
message says more than that default allows, so #66 has to give it its
own rule.

### 4.3 Acknowledgement

The gateway has open, acknowledged and resolved states, one open alert
per space and kind, and repeats that update it instead of paging again
(`SAANS_PLAN.md` §10.3). For elder care [proposal]:

- **I'm checking** acknowledges, and every other contact sees who.
- **All fine** resolves (the button uses the person's name).
- **False alarm** resolves and records it, and those records are the
  false-alarm count in §4.6.
- Movement after an urgent alert updates it and does not resolve it.

### 4.4 Escalation

Today an unacknowledged alert escalates once, after `ack_timeout_min`,
to `escalate_to`. For elder care [proposal]:

1. At 0 min: the primary contact or contacts.
2. After 5 min without acknowledgement: every family contact.
3. After 10 min: a nearby contact chosen by the family, such as a
   neighbour, by phone call.

Saans never calls emergency services itself. It is not an emergency
service (§6).

### 4.5 Quiet hours and pause

- **Quiet hours** are set by each family member. Heads-ups wait until
  they end; urgent alerts never wait.
- **Pause** belongs to the monitored person (§5.5). While paused, no
  alerts are raised, and the family sees "Paused by Amma until 18:00".
  A pause ends at the time chosen, at most 12 hours later [proposal], and
  the family is told when sensing resumes.

### 4.6 The false-alarm budget and the miss rate, together

ADR-005 requires every detector to report its miss rate and its false
alarm rate, never one alone. In a home the units are:

- **false urgent alerts per home per week;**
- **false heads-ups per home per week;**
- **miss rate:** for the long lie, the share of long lies that are not
  alerted within N + 2 minutes, with the median time to alert; for the
  day that did not start, #67 defines it;

in one table per configuration, marked SIM or REAL.

**Budget** [proposal, not yet tested with families (open question 1)]:

- urgent: at most **1 false alert per home per month** (0.25 per week);
- heads-up: at most **2 per home per week**;
- staged long lies missed: at most **1 in 20**.

---

## 5. Privacy and consent

### 5.1 What stays in the home

| Data | Where it stays | Rule |
|---|---|---|
| Raw radio data (CSI) | On the sensor | Never leaves it (ADR-001). A debug raw mode exists, is off by default, and every frame says when it is on. |
| FeatureFrames (68 bytes, 2 Hz) | Home network, to the gateway | Never leave the home. |
| Breathing and heart fields | Zeroed on the sensor | Unless the person turned vitals on for that room (`apply_consent_gate`, `edge/dsp/feature_frame.hpp`). |
| Time series behind the charts (a point every 2 s) | Gateway memory | Never written to disk; one hour by default (`[history]`). |
| Event logs | Gateway disk | 30 days (`event_retention_days`). |
| Daily summaries | Gateway disk | 90 days (`history_retention_days`). |
| Audit log | Gateway disk | 365 days (ADR-010). |

### 5.2 What leaves the home

Only this, through Telegram in the first homes (ADR-018) and through
the relay (#40) later [proposal; #40 decides the format]:

- **alerts:** level, kind, room name as the family labelled it, start
  time, acknowledgement state;
- **sensing status:** which rooms cannot see and, with the relay, the
  gateway's heartbeat;
- **"home now"** answers, with the app, when a family member asks
  (§4.2);
- **the morning message** (§4.1), as the monitored person agreed at
  setup: the night (time in bed, times up, night bathroom visits,
  restlessness, the breathing rate against the person's own usual) and
  the day before (got-up time, active hours, rooms, unusual quiet
  periods).

Never: raw CSI, FeatureFrames, the time series behind the charts, heart
rate, breathing numbers in alerts. The heartbeat says the gateway is
alive and nothing about the rooms.

Telegram can read what it carries. Whether the relay can is the main
privacy decision in #40. End-to-end encryption from the home gateway to
the family's phones would leave the relay holding only routing data
(ADR-015).

### 5.3 No identities, and what that does not mean

Saans outputs are per room, never per person (ADR-005), and the system
has no way to tell people apart. But in a home where one person lives
alone, a room's data is that person's data. "No identities" does not
make it anonymous. Everything that leaves the home is treated as the
monitored person's personal data.

When there is a visitor, presence and routine describe the visitor too,
and Saans cannot tell. The monitored person can mark "I have guests",
which pauses the routine summary and unusual quiet and does not pause
the long lie [proposal].

### 5.4 DPDP Act 2023

Not legal advice; the founder should get a lawyer's reading (§8).
[sourced: the Act's text as published on dpdpa.com; the Rules via the
secondary sources in §9.] It is written for the relay and the app
(§7.3); the roles while the channel is Telegram are part of that
reading.

**Who is who.**

- The monitored person is the **Data Principal**.
- Saans, the company running the relay and the app, is a **Data
  Fiduciary** for what they process. Until now no data left a site.
- Cloud hosting and push-notification services are **Data
  Processors**, engaged only under a contract (s. 8(2)).
- Family members are Data Principals for their own phone numbers and
  actions. Their own use of alerts about a parent is probably "personal
  or domestic purpose", which the Act does not apply to (s. 3(c)(i)).
  [UNVERIFIED: not checked with a lawyer.]

The Act has no separate category for health data. The repository
treats breathing as health data anyway (`SAANS_PLAN.md` §8), which is
stricter than the Act, and that stays.

**What the Act asks, and how Saans meets it** [proposal]:

| Section | Duty | In Saans |
|---|---|---|
| 5 | Notice before consent, itemised | A plain-language notice in the person's language, shown at setup, listing §5.2 |
| 6(1) | Consent free, specific, informed, unconditional, unambiguous, by clear affirmative action | The monitored person consents in their own step at setup, which covers the night's breathing trend (§3.4). The buyer's consent does not count. Vitals in any other room are a separate consent. |
| 6(4) | Withdrawing consent as easy as giving it | One action, on the gateway or in the app |
| 7(f) | Processing allowed to respond to a medical emergency | Not relied on. Routine sensing runs on consent. |
| 8(5) | Reasonable security safeguards | Signed frames (ADR-012), authentication, audit log; the relay's to be designed in #40 |
| 8(6) | Tell the Board and each affected person about a breach | A breach procedure before the first pilot. Secondary sources report a detailed report to the Board within 72 hours under the Rules. [UNVERIFIED against the Gazette text.] |
| 8(7) | Erase on withdrawal, or when the purpose is served | Withdrawal erases the relay's copy and the gateway's logs |
| 8(9), 8(10) | A published contact, and a way to raise grievances | Before the first pilot |
| 9 | A person with a disability who has a lawful guardian consents through the guardian | Open question 4: a person with dementia |
| 11-13 | Access to a summary of their data; correction and erasure; grievance | The person's view in the app (§5.5) |
| 14 | Nominate someone to exercise these rights on death or incapacity | The app asks the person to name a nominee at setup |

**When.** The DPDP Rules 2025 were notified on 13 November 2025
(G.S.R. 846(E)). Most fiduciary duties apply 18 months later, from 13
May 2027. A proposal reported in January 2026 to shorten that to 12
months had not been notified, according to the latest sources found.
[sourced: secondary; UNVERIFIED against the Gazette.] A first pilot is
likely to run before May 2027. Build to the Act anyway.

### 5.5 The monitored person's controls

- **Pause**, for up to 12 hours, and resume at any time [proposal]. How
  they pause, whether with a button on the gateway, a phone or a voice
  command, is open question 3.
- **See what the family sees:** every alert, every summary, and every
  time someone looked at "home now".
- **Choose what is shared:** the daily summary, the breathing trend.
- **Turn vitals on or off** per room.
- **Choose who gets alerts** and in what order.
- **Withdraw consent and erase.**
- **A sensing light** on each sensor or on the gateway, so they know it
  is on [proposal; hardware #44].

---

## 6. What Saans does not claim

### 6.1 The claims

- Saans is **not a medical device**. Nothing it says claims to
  diagnose, treat, prevent or monitor any disease, disorder or
  condition. It does compute a breathing rate, with the person's
  agreement (§3.4), and §6.4 says why that is the part nearest the line.
- Its alerts are **advisory** (ADR-005): they say what was observed and
  how confident the system is. They do not replace calling for help, a
  doctor, or a personal alarm.
- It is **not an emergency service**. It does not call an ambulance.
- It **can miss** and **can raise false alarms**, and it publishes both
  rates (§4.6).
- It **says when it cannot see** (§3.6).

### 6.2 What India's rules say

**The device definition.** Since 1 April 2020, every device intended
for the listed medical purposes is a medical device under the Drugs and
Cosmetics Act, 1940 and the Medical Devices Rules, 2017; the
notification is S.O. 648(E) of 11 February 2020 [sourced: Khaitan & Co;
Mondaq]. The listed purposes, as CDSCO's software guidance quotes them,
include "diagnosis, prevention, monitoring, treatment or alleviation of
any disease or disorder" and "diagnosis, monitoring, treatment,
alleviation or assistance for, any injury or disability" [sourced: CDSCO
guidance, §4.9]. The definition covers instruments and apparatus as
well as software, so the same test applies to the sensors.

**CDSCO's software guidance** (Doc No. CDSCO/MD/GD/MDSW/01/2026,
reported as issued on 21 July 2026) [sourced: CDSCO PDF, read;
date from secondary sources]:

- It does not apply to software "merely intended for general
  wellness, promoting a healthy lifestyle" or that measures body
  parameters only for regular tracking and fitness (§2).
- Wellness software may claim general improvement to a state of health
  but "should not be making any reference to diseases or disorders or
  pathological conditions" (§5.2, note i).
- Software that reports physiologic values for "screening, diagnosis,
  monitoring, alerting, or management" of a disease, disorder or
  condition is **not** wellness software (§5.2, note ii).
- "Real-time patient monitoring" is given as an added function that
  may make a hospital or laboratory information system a medical
  device (§5.2, in the note on those systems).
- Intended use is read from labelling, instructions and promotional
  material (§4.7). The guidance asks an intended-use statement to name
  the patient population, with "geriatric" as an example, and the use
  environment, with "home use" as an example (§6).
- The guidance says it is for "creating public awareness" only and is
  not a legal instrument; the Act and Rules govern (opening notice).

### 6.3 The wording that follows

| Use | Avoid |
|---|---|
| home safety check-in | fall detection, detects falls |
| no movement for a long time after a sudden movement | fall, collapse, injury, unconscious |
| lets your family know | monitors, monitoring (of the person) |
| unusually quiet day; daily routine | decline, deterioration, symptoms |
| breathing rate trend (wellness) | apnoea, sleep disorder, respiratory, vital signs monitoring |
| older person living alone; your parent | patient, elderly patient |
| advisory; may miss; may raise false alarms | clinical-grade, medical-grade, accurate, guaranteed, saves lives |
| | any named disease or condition (dementia, Parkinson's, COPD, heart disease) |
| | anything addressed to doctors or care professionals |

"Fall detection" is avoided for two reasons: the system cannot tell a
fall from a sit-down (§3.3), and detecting an injury is the claim
closest to "monitoring ... any injury". The jobs in §1.1 use the word
"fall" because they describe what the person fears, not what the
product claims; product copy does not.

A starting intended-use statement, for the regulatory opinion to
correct [proposal]:

> Saans is a home safety and wellness product. It senses movement,
> presence and resting breathing rate in rooms from WiFi signals. It
> sends people the user chooses a short daily summary, and an advisory
> notification when the day's activity differs from the user's usual
> pattern. It is not intended to diagnose, treat or prevent any disease
> or condition, and it is not an emergency service.

### 6.4 What is uncertain

1. **Wording may not be enough.** The definition turns on intended use,
   and the guidance asks the intended-use statement to name the
   population and the environment as well as the purpose. A product for older people living alone that alerts on
   a possible fall could be read as "monitoring ... any injury" or
   "assistance for ... disability" however it is phrased.
2. **The breathing trend** is a physiologic value, and it now goes to
   the family every morning (ADR-018). Showing a trend without an alert
   or a named condition is the most defensible form, and it is still
   the signal closest to the line.
3. **The hardware** is judged by the same definition as the software.
4. **The guidance is not law** (its own notice), and a law firm's read
   of the October 2025 draft found the point where monitoring becomes
   medical "often unclear" [sourced: Cyril Amarchand Mangaldas].

**Before the first paid pilot:** a written opinion from an Indian
medical-device regulatory lawyer or consultant on the intended-use
statement, the app text and the website. This spec found no source that
settles points 1-3.

---

## 7. The MVP cut

### 7.1 The demo (issue #47)

One real bedroom, real hardware, in two uncut parts: a night and its
morning message, then a morning that does not start (ADR-018). An
on-screen clock runs the whole time; where a stretch is sped up, as the
night is, the video says so on screen. The configuration (vitals on,
number of links), the sensor positions and the link across the bed are
shown at the start. The raw recording and the config are archived with
the video.

| # | What happens | Dashboard | Family phone | Needs |
|---|---|---|---|---|
| 1 | The bedroom is empty. | Bedroom: empty. Sensing on, 2 links. | Nothing | #42 setup, #44 hardware, #45 first CSI |
| 2 | The person goes to bed and lies still. | Occupied, still. A breathing rate with its confidence, because vitals are on for this room, which the screen says. | Nothing | Presence and breathing on real data; written consent from the person on screen |
| 3 | The night passes, and they get up. | The night's rest periods, and its median breathing rate against the baseline. | The morning message (§4.2) | #66; earlier nights for the baseline |
| 4 | Another morning, they stay in bed past their usual time. Staged. | Not up yet. | The day did not start (§4.2) | #67; a baseline of the person's own mornings |
| 5 | The family member acknowledges it. | Acknowledged by [name] | Acknowledged | Telegram's buttons, with a real bot |

A staged long lie is in the demo only if #62 lands.

Risks to the demo, in the order they could stop it:

- **AGC.** In the ESP32 receiver model, which is unverified, the radio's
  automatic gain steps make a seated person read as empty in every frame
  at the model's default setting, and the fix the repository expects is
  reading the gain register rather than a DSP change (**SIM**,
  `docs/RESULTS.md`, "ESP32 receiver model"; `NOTES.md` §9). The first
  bench session (#45) settles it.
- **A doubled rate.** A link in a blind spot can report twice the true
  breathing rate (§3.4). The placement across the bed and the night's
  median have to absorb it.
- **Someone else in the bedroom.** A second person or a pet. The first
  homes have neither.
- **The channel.** Telegram works in tests against a local stand-in
  only (`docs/EVIDENCE.md` §2.6).

The video must not say or imply fall detection, medical monitoring,
accuracy figures, or that it works in any room.

### 7.2 In the MVP software

- The night in the bedroom (§3.4), the morning message (#66) and the
  "day did not start" alert (#67).
- Room presence and the home-level view (§3.1).
- The daily routine summary without unusual quiet (#39); unusual quiet
  follows once there are 14 days of data.
- Sensing status for each room (§3.6).
- Telegram as the channel, with its acknowledge and resolve buttons
  and the gateway's one escalation.
- Pause and consent for the monitored person (§5.5), at least a pause
  button.
- One-command home setup (#42), and a bed link in its questions (no
  issue yet).

### 7.3 Deferred

- The long lie as a family alert, until issue #62 (§3.3); breathing
  pause and rate alarms as family alerts (§3.5).
- Heart rate, in any form.
- The presence hold on by default (it ships off).
- Possible collapse, instantaneous fall detection.
- The relay (#40) and the native app (#41), designed in PRs #54, #55
  and #57, until a pilot shows Telegram is not enough. With them wait
  "no contact with the home" (§3.6), "home now" (§4.2), and what §4.3
  to §4.5 propose beyond Telegram's two buttons and the gateway's one
  escalation. WhatsApp only if a pilot family cannot use Telegram.
- Care homes, homes with more than one resident, pets.
- Two-way voice, a local "are you OK?" prompt (open question 9).
- Clinician access, integrations with health records.
- Learned models (ADR-007); new simulation studies and model training,
  until the first real nights are recorded (ADR-018).

---

## 8. Open questions for the founder

1. **The budget.** Are the targets in §4.6 acceptable to families? N
   waits with the long lie (§3.3).
2. **A door sensor?** A cheap contact sensor on the front door would
   tell "away" from "in an unsensed room". It adds a non-WiFi sensor;
   is that acceptable?
3. **How does the monitored person pause?** A physical button, the
   phone, or voice. Which can every intended user manage?
4. **Who consents for someone with dementia?** Under s. 9 a lawful
   guardian consents for a person with a disability who has one. Who
   consents when there is no guardian, and should Saans be installed at
   all then?
5. **Vitals in the first version?** Closed by ADR-018: the night's
   breathing trend is in (§3.4); pauses, rate alarms and heart rate
   stay out (§3.5).
6. **Escalation beyond family.** Neighbour, building security, a paid
   responder service? Who calls them, and how?
7. **The interim channel.** Closed by ADR-018: Telegram, for the demo
   and the first homes. The relay and the app wait (§7.3).
8. **A regulatory opinion before the first pilot.** Who, and on what
   budget?
9. **A local check before paging the family?** A chime or light in the
   home, "Are you OK? Press to cancel", could cut false alarms. It needs
   a device in the home that speaks or lights up and a button.
10. **Ethics and pilot consent.** `SAANS_PLAN.md` §8 requires ethics
    committee approval before human-subject data collection. Which
    committee for a company's home pilot, and consent forms in which
    languages?
11. **Flats.** Measure a neighbour through the wall before the first
    pilot in an apartment?
12. **Guests mode.** Once the long lie is on, should it stay on when
    the person has guests (§5.3)?

---

## 9. Sources

Repository (all performance numbers **SIM**): `docs/RESULTS.md`
sections named in the text; `NOTES.md` §8.2, §9, §12, §14, §15;
`docs/EVIDENCE.md`; `docs/CAPABILITY_LIMITS.md`; `SAANS_PLAN.md` §2,
§8, §10; `rust/crates/saans-health/src/detectors.rs`, `longlie.rs`,
`types.rs`, `system.rs`; `rust/crates/saans-gateway/saans.example.toml`;
ADR-001, 005, 006, 010, 011, 012, 013, 015, 018.

Outside (read 2026-11-06):

- CDSCO, Guidance Document on Medical Device Software, Doc No.
  CDSCO/MD/GD/MDSW/01/2026 (read in full text):
  https://www.cdsco.gov.in/opencms/export/sites/CDSCO_WEB/Pdf-documents/Guidance-document-on-Medical-Device-Software-under-MDR-2017.pdf
- Issue date of that guidance (secondary): Nishith Desai Associates,
  https://nishithdesai.com/research-and-articles/hotline/pharma-healthcare-update/from-code-to-compliance-decoding-indias-medical-device-software-guidance-15623 ;
  The National Law Review,
  https://natlawreview.com/article/code-compliance-decoding-indias-medical-device-software-guidance
- On the October 2025 draft: Cyril Amarchand Mangaldas, "Medical Device
  As Software: Has CDSCO Guidance Changed The Rules?", 29 January 2026,
  https://corporate.cyrilamarchandblogs.com/2026/01/medical-device-as-software-has-cdsco-guidance-changed-the-rules/
- S.O. 648(E), all medical devices notified (secondary): Khaitan & Co,
  https://khaitanco.com/thought-leaderships/All-medical-devices-notified-to-be-regulated ;
  Mondaq,
  https://www.mondaq.com/india/life-sciences-biotechnology-nanotechnology/899470/all-medical-devices-notified-to-be-regulated
- Digital Personal Data Protection Act, 2023, sections 3, 7 and 8 as
  published at https://dpdpa.com/dpdpa2023/chapter-1/section3.html ,
  https://dpdpa.com/dpdpa2023/chapter-2/section7.html and
  https://dpdpa.com/dpdpa2023/chapter-2/section8.html (a mirror, not the
  Gazette). Sections 5, 6, 9 and 11-14 are summarised from secondary
  sources and were not re-read in full.
- DPDP Rules 2025, notification and phasing (secondary): YourStory,
  https://yourstory.com/2025/11/digital-personal-data-protection-dpdp-rules-2025-rollout-over-18-months ;
  Mondaq,
  https://www.mondaq.com/india/data-protection/1709066/key-highlights-of-the-digital-personal-data-protection-rules-2025
- The 12-month proposal (secondary): Business Standard,
  https://www.business-standard.com/technology/tech-news/meity-may-cut-compliance-timeline-for-key-dpdp-rules-to-12-months-126012201293_1.html ;
  https://dpdprules.org/blog/dpdp-18-months-to-12-months-proposal
- Where falls happen: Moreland B. L., Kakara R., Haddad Y. K., Shakya I.,
  Bergen G., "A Descriptive Analysis of Location of Older Adult Falls
  That Resulted in Emergency Department Visits in the United States,
  2015", https://pmc.ncbi.nlm.nih.gov/articles/PMC8669898/
  [UNVERIFIED: full text not read; the room ranking is from secondary
  summaries]; Australian Institute of Health and Welfare, "Falls in
  older Australians 2019-20: hospitalisations",
  https://aihw.gov.au/reports/injury/falls-in-older-australians-2019-20-hospitalisation/contents/where-did-falls-occur
- Long lie: Physiopedia, "Long Lie", https://www.physio-pedia.com/Long_Lie ;
  Kubitza et al., scoping review, European Review of Aging and
  Physical Activity, 2023,
  https://link.springer.com/10.1186/s11556-023-00326-3 ; Sheffield Long
  Lies Study (the one-in-five figure and the over-one-hour threshold,
  read on the page),
  https://www.sheffield.ac.uk/cure/blank-basic-2-use-first
