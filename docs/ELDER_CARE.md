# Elder care: the product spec

**Status:** draft for issue #37, 2026-11-06. The decision to lead with
elder care is [ADR-015](adr/015-elder-care-first.md). Issues #38-#42
and #47 build on this file.

Saans for elder care is a few small sensors in the home of an older
person who lives alone, and a gateway on the same home network. When
the person has been still for a long time after a sudden movement, or
the day looks unusually quiet, their family gets a message on their
phone. There is no camera, no microphone and nothing to wear. The raw
radio data stays on the sensor that measured it, except in a debug mode
that is off by default and flagged in every frame when on (ADR-001).

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
- If they fall and cannot get up, for someone to find out within the
  hour, without having to press a button or wear anything.
- Not to have their children worrying, or calling, over nothing.
- To switch it off when they want privacy, and to know when it is on.

A pendant alarm needs the person to be wearing it and able to press it.
Saans needs neither. That is the case for it over a button, and it only
holds if the false alarms stay rare (§4.6).

### 1.2 Family caregivers

Adult children or other relatives, often in another city or country.
They receive the alerts and decide what to do: call, ask a neighbour to
look in, travel.

What they want:

- To know their parent is up and about without phoning every day.
- To be told quickly when something may be wrong, and only then.
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

A home kit senses **2-4 rooms**, each a space in `saans.toml`:

| Room | Why it is sensed | Links [proposal] |
|---|---|---|
| Bathroom | Falls happen here; a camera is not acceptable here | 2 |
| Bedroom | Falls happen here, often at night; a camera is not acceptable here | 2 |
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
- **Unusual quiet:** a period in the day window, while the person is
  home, with no motion bout anywhere, longer than the person's own
  95th-percentile quiet period for that time of day over the last 28
  days. Needs at least 14 days of the person's data; until then it is
  not computed.

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
in the box model. Nothing at the level of a routine has been *measured*.

`saans export-home` now writes one fixed multi-room, multi-day plan to
replay. That shows the routine working on that plan; it does not say how
accurate the routine is across homes.

**Built (#39), except unusual quiet.**

- **Where:** `saans-health/src/routine.rs`.
- **What:** got up, active hours and night bathroom visits, each compared
  with up to 28 previous complete days, once there are at least 7.
- **How to use it:**
  - set `room = "bedroom"` or `room = "bathroom"` on a space;
  - set the night window under `[home]`;
  - read it at `/api/routine` or on the dashboard's overview. It goes
    to admin, or to safety with every space of the home in scope
    (`privacy::routine_for`): every room's motion feeds it.
- **Getting up**, as built: a bedroom rest period of at least two hours
  that ends between 03:00 and 12:00, followed by a motion bout 10 to 30
  minutes later. The latest such rise wins, so a night-time trip and a nap
  do not count.
- **Active hours**, as built: hours in which some room's detector saw
  the person moving, which means measured present with the debounced
  motion the long lie listens for. A raw motion sample in a room measured
  empty does not count, and the same motion confirms a getting-up.
- **A room that was not seen is unknown, not zero.** A value is known
  only if its room was visible in every hour of its window: the bathroom
  through the night, the bedroom from two hours before the morning window
  to its end. Otherwise the night's bathroom visits are unknown (`null`,
  never 0) and a day without a getting-up says "unknown", not "did not
  get up". An unknown value is not compared and is no part of the
  baseline. So a home's first night reads unknown: it began before the
  gateway did.
- **Not covered yet:** active hours have no such check per room. A room
  that is neither the bedroom nor the bathroom and is down all day lowers
  them, and can hide the motion that confirms a getting-up. Only the
  whole-day rule (some room seen in 20 hours) applies there.
- **Restarts:** a gateway restart loses the day in progress. That day
  closes partial, and if fewer than 20 hours were observed it is not used.
- **Starting values:** every threshold is a starting value for pilots
  [proposal].
- **Simulated home:** `saans export-home` writes a simulated home over whole
  days for a replay (SIM).

Unusual quiet is not built: it needs 14 days and per-time-of-day
baselines. It is advisory: it describes a day and
diagnoses nothing.

### 3.3 Long lie: the lead safety signal (issue #38)

**Definition** [proposal; #38 settles it]. In one room, an urgent alert
when all three hold:

1. **A sudden movement:** motion index at least 8 (`BURST_INDEX`),
   followed within 5 s (`COLLAPSE_SETTLE_S`) by stillness. This is the
   existing possible-collapse arming, unchanged.
2. **Then present and not moving for N minutes:** no gross motion
   (`STILL_INDEX`), with presence measured, or held if that room has the
   presence hold switched on (below).
3. **Nobody seen moving in another room** in that time (needs the
   home-level view, §3.1).

Any gross motion in the room after the alert updates it ("movement seen
at 06:31"). The alert stays open until a person resolves it, because
moving does not mean being fine [proposal].

**N.** The clinical term "long lie" usually means an hour or more on
the floor, although there is no agreed definition [sourced: Physiopedia;
Kubitza et al. 2023], and about one fall in five among people over 65
becomes a long lie [sourced: Sheffield Long Lies Study]. N has to be
well under an hour so help can arrive before the lie becomes long; how
far under is set by the false-alarm budget (§4.6). Starting values for
issue #38 to test, per room [proposal]:

- bathroom: 15 minutes;
- living room and kitchen: 30 minutes (the existing inactivity
  default);
- bedroom during the night window: the hardest case. Getting out of bed,
  falling and lying still looks like getting back into bed quickly and
  sleeping. #38 measures it before a value is chosen.

**Severity.** `Check`, whether or not vitals are on. Today the
inactivity alert reaches `Check` only when vitals are on and breathing
is not detected (`inactivity_step`); the long lie cannot depend on
vitals, which are off by default.

**Why not fall detection.** The possible-collapse detector, which looks
for a motion burst followed by 30 s of stillness, fired in (**SIM**,
"S3 health detectors, held-out seeds"):

| scenario | 2-D slice | 2.7 m box |
|---|---|---|
| falls (collapse runs) | 36/36 | 33/36 |
| sit-downs | 36/36 | 35/36 |
| readers walking in and sitting | 33/36 | 32/36 |
| fidgeting nights | 35/36 | 30/36 |

Amplitude-only WiFi cannot see height, so a fall and a heavy sit-down
look the same (`NOTES.md` §8.2, `docs/CAPABILITY_LIMITS.md` §1). It
ships off and stays off.

**What the time condition buys, and what it does not.** In the same
runs, the 30-minute inactivity alert fired in 0 of 36 sit-downs and in
36 of 36 readers (2-D slice; 0 and 35 of 36 in the box). So a sit-down followed by activity is filtered out by
time, and a person who sits down quickly and then stays still for 30
minutes is not. Reading those two columns together suggests the long
lie as defined above would fire on most simulated 40-minute readers.
That is an inference from two columns of one table, not a measurement
of the joint event. Naps, television, reading and prayer are the false
alarms #38 has to measure, and they are why N is set per room.

**The person who vanishes.** If the room reads empty, the long lie
never starts. A lying person is not read as empty more often than an
upright one: 1.0% against 0.9% of the still period, and 10 against 12
of 1,200 spots mostly empty (**SIM**, "A fallen person read as an empty
room"). But the misses move to the side of the link line, and on one
link reach 3.8% of the still period at 1.0-1.5 m from it.

PR #31 (open, not merged) adds a presence hold after a fall-shaped
motion, off by default. On its branch (**SIM**, `NOTES.md` §14 there):

- what it buys: lying read as empty goes from 1.0% to 0.3% of the
  still period, and spots read mostly empty from 10 to 2 of 1,200;
- what it costs: 18 of 72 departures (2-D slice) and 21 of 72 (box)
  engage the hold, and in a room left empty each would raise an alert
  30 minutes later. About one departure in four.

In a home there is a remedy the campus did not have: if the person
appears in another sensed room, the hold ends. That removes the cost for
moves inside the home, not for leaving it. It is not built or measured.

**Not modelled** (`NOTES.md` §14): the floor's reflection of the body's
own echo, lying on the side, nodes at other heights, real floors. Real
recordings of someone lying on a floor (#46) settle it.

**Measured for the long lie itself (SIM**, `docs/RESULTS.md`, "Long
lie"; read in `NOTES.md` §15). Built in #38, off by default
(`long_lie_min`). 120 simulated runs per scenario in the 2.7 m box,
presence hold off:

- falls that end still: 6.7% missed at every N from 3 to 30 minutes,
  and 2/40 on two links at N = 15 (the staged target is 1 in 20);
- slow slides to the floor: 46.7% missed at N = 15, because most never
  reach `BURST_INDEX`;
- a heavy sit-down and then 45 minutes still: alerted in 118 of 120 at
  N = 30, so the inference above holds, and no N fixes it;
- lying down into bed quickly and turning over now and then: alerted in
  94 of 120 at N = 15;
- walking in and reading for 40 minutes: alerted in 101 of 120 at every
  N, because walking in reaches `BURST_INDEX` too; a two-hour fidgeting
  night: 78 of 120 at N = 15;
- an empty room: never; a walk-out with the presence hold on: 33 of 120
  alert on the empty room, and none when the person is seen moving in
  another room (`long_lie_elsewhere`).

### 3.4 Sleep breathing rate trend (only where opted in)

**Definition.** One number per night: the median resting breathing rate
in the bedroom (`DailySummary::resting_bpm_median`, which needs 600
confident still samples, about 5 minutes). It is compared with the
person's own baseline: the median of up to 28 previous nights, after at
least 3, flagged when it differs by more than the larger of
3 × 1.4826 × MAD and 2 BPM (`TrendDeviation`,
`saans-health/src/detectors.rs`). It
exists only for a room whose `vitals` the monitored person has switched
on.

It is severity `Info`, so it is never pushed (alert routes never page
`info`, `saans.example.toml`). It appears as a line in the app, to the
person, and to the family only if the person shares it.

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

The trend itself has not been evaluated: no multi-day timeline exists.

**Why it is the signal nearest the medical-device line:** see §6.

### 3.5 What is not a family alert in the first version

| Detector | Exists | Why it is not a family alert |
|---|---|---|
| Breathing pause | yes, `Check` when ongoing over 20 s or ending in an empty reading | An alert on a physiological value is the clearest "medical purpose" in §6. Its false rate on simulated nights is 0.12 per hour on every layout in the 2-D slice ("S3 health detectors, held-out seeds"), which is about one false pause per 8-hour night by arithmetic. |
| Fast or slow breathing | yes, `Advisory` | An alert on a physiological value (§6). |
| Heart rate | built, off | Not trusted: 80% within 3 BPM ungated; gated, 97% correct but only in 61% of cases (**SIM**, `docs/RESULTS.md`, "Original 64-scenario sweep"; `docs/EVIDENCE.md` §6.4). Stays off. |
| Possible collapse | yes, off | Fires on sit-downs as often as falls (§3.3). |
| Inactivity, as it is | yes, `Advisory`/`Check` | Fired for every simulated 40-minute reader (36 of 36 in the 2-D slice, 35 of 36 in the box). A person who reads or naps still for 40 minutes once a day would cause about 7 alerts a week, by arithmetic. The long lie replaces it as the family alert. |

They all stay in the gateway and on the local dashboard for a room
where the person has opted in.

### 3.6 Sensing status

A safety product must never let silence look like a quiet day.

- **A room cannot see:** `SpaceBlind`, `LinkDown`, `CalibrationStale`
  exist (`saans-health/src/types.rs`, `system.rs`). The family sees
  "Saans cannot see the bathroom since 14:05" as a heads-up after 15
  minutes [proposal].
- **The home is silent:** a gateway that has lost power or internet
  cannot say so. The relay (#40) notices missing heartbeats and tells
  the family "No contact with the home since 14:05" after 10 minutes
  [proposal].
- **Raw radio mode is on:** `RawModeActive` exists; the person and the
  family both see it.

---

## 4. Alerts

### 4.1 Levels

| Level | Saans severity | Examples | On the phone | Quiet hours |
|---|---|---|---|---|
| **Urgent** | `Check` | Long lie | Notification with sound, repeated until someone acknowledges it [proposal] | Never held |
| **Heads-up** | `Advisory` | Unusual quiet; not seen for long; a room cannot see; no contact with the home | Ordinary notification | Held until quiet hours end |
| **Summary** | `Info` | Daily routine; breathing trend, if shared | In the app, once a day | Not a notification |

### 4.2 What a family member sees

Lock screen, urgent [proposal]:

> **Saans · Bathroom**
> No movement for 15 min after a sudden movement at 06:12.
> Please call or check on Amma.
> [I'm checking] [Call Amma] [False alarm]

Lock screen, heads-up [proposal]:

> **Saans · Home**
> Quieter than usual: no movement seen since 11:20. Usually active by now.

Inside the app:

- the alert's timeline for the last hour: which rooms were occupied and
  when, and whether there has been movement since;
- who else has seen it and who is acting ("Ravi is checking, 06:29");
- the sensing status of each room;
- "Home now": home, away, not seen or unknown, and the last room with
  movement. This is fetched when the family member opens the app, not
  streamed, and every look is written to the audit log the monitored
  person can read [proposal]. It is the one place the family sees the
  present rather than an event.

No breathing numbers and no evidence sentences, unless the monitored
person has turned vitals on for that room and turned on sharing details.
The gateway already keeps details at home by default
(`include_health_details = false`, `saans.example.toml`).

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
  they end; urgent alerts never wait. An urgent alert sounding through a
  phone's Do Not Disturb depends on the platform: iOS has a "critical
  alerts" permission that Apple grants per app, and Android has its own
  rules. That is part of the app decision (#41). [UNVERIFIED: whether
  Apple would grant it to this product.]
- **Pause** belongs to the monitored person (§5.5). While paused, no
  alerts are raised, and the family sees "Paused by Amma until 18:00".
  A pause ends at the time chosen, at most 12 hours later [proposal], and
  the family is told when sensing resumes.

### 4.6 The false-alarm budget and the miss rate, together

ADR-005 requires every detector to report its miss rate and its false
alarm rate, never one alone. In a home the units are:

- **false urgent alerts per home per week;**
- **false heads-ups per home per week;**
- **miss rate:** the share of long lies that are not alerted within
  N + 2 minutes, with the median time to alert;

in one table per configuration, from simulated homes first (#38), then
staged real lies (#46), then pilot weeks.

**Budget** [proposal, to be tested with families before #38 measures,
so the measurement cannot move the target]:

- urgent: at most **1 false alert per home per month** (0.25 per week);
- heads-up: at most **2 per home per week**;
- staged long lies missed: at most **1 in 20**.

A simulated estimate for a home adds up, room by room, the false alerts
per occupied hour from #38's scenarios (sit-downs, naps, readers,
departures) times the hours the room is occupied in a week. #38 needs
home timelines for that, which `saans-sim` does not have.

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

### 5.2 What leaves the home, through the relay (#40)

Only this [proposal; #40 decides the format]:

- **alerts:** level, kind, room name as the family labelled it, start
  time, acknowledgement state;
- **sensing status:** the gateway's heartbeat, and which rooms cannot
  see;
- **"home now"** answers, when a family member asks (§4.2);
- **the daily summary,** if the monitored person allows it: got-up time,
  active hours, night bathroom visits, unusual quiet periods;
- **the breathing trend,** only if vitals are on for the bedroom and the
  person chooses to share it.

Never: raw CSI, FeatureFrames, the time series behind the charts, heart
rate, breathing numbers in alerts. The heartbeat says the gateway is
alive and nothing about the rooms.

Whether the relay can read what it carries is the main privacy decision
in #40. End-to-end encryption from the home gateway to the family's
phones would leave the relay holding only routing data (ADR-015).

### 5.3 No identities, and what that does not mean

Saans outputs are per room, never per person (ADR-005), and the system
has no way to tell people apart. But in a home where one person lives
alone, a room's data is that person's data. "No identities" does not
make it anonymous. Everything that leaves the home is treated as the
monitored person's personal data.

When there is a visitor, presence and routine describe the visitor too,
and Saans cannot tell. The monitored person can mark "I have guests",
which pauses the routine summary and unusual quiet and leaves the long
lie on [proposal].

### 5.4 DPDP Act 2023

Not legal advice; the founder should get a lawyer's reading (§8).
[sourced: the Act's text as published on dpdpa.com; the Rules via the
secondary sources in §9.]

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
| 6(1) | Consent free, specific, informed, unconditional, unambiguous, by clear affirmative action | The monitored person consents in their own step at setup. The buyer's consent does not count. Vitals are a separate consent per room. |
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
  condition. It does compute a breathing rate where the person opts in
  (§3.4), and §6.4 says why that is the part nearest the line.
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

> Saans is a home safety and wellness product. It senses movement and
> presence in rooms from WiFi signals and sends advisory notifications
> to people the user chooses when there has been no movement for a set
> time after a sudden movement, or when the day's activity differs from
> the user's usual pattern. It is not intended to diagnose, treat or
> prevent any disease or condition, and it is not an emergency service.

### 6.4 What is uncertain

1. **Wording may not be enough.** The definition turns on intended use,
   and the guidance asks the intended-use statement to name the
   population and the environment as well as the purpose. A product for older people living alone that alerts on
   a possible fall could be read as "monitoring ... any injury" or
   "assistance for ... disability" however it is phrased.
2. **The breathing trend** is a physiologic value. Showing a trend
   without an alert or a named condition is the most defensible form,
   and it is still the signal closest to the line.
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

One real room, real hardware, uncut. An on-screen clock runs the whole
time; if a stretch is sped up, the video says so on screen. The
configuration (N, vitals on, number of links) is shown at the start.
The room's sensor positions are shown, so a viewer can see where the
person lies relative to the links. The raw recording and the config are
archived with the video.

| # | What happens | Dashboard | Family phone | Needs |
|---|---|---|---|---|
| 1 | The room is empty. | Bedroom: empty. Sensing on, 2 links. | Home: not seen | #42 setup, #44 hardware, #45 first CSI |
| 2 | A person walks in. | Occupied | Home: home (when opened) | Presence on real data |
| 3 | They sit still, breathing. | Occupied, still. A breathing rate with its confidence, because vitals are on for this room, which the screen says. | Nothing | Breathing on real data; written consent from the person on screen |
| 4 | They lower themselves quickly onto a floor mat and lie still. Staged; nobody falls. | Still for 0:00, 0:30, ... "Long lie: 2:40 of 3:00" | Nothing yet | #38 long lie; the movement must reach `BURST_INDEX` |
| 5 | N minutes pass (N = 3 for the demo, shown on screen). | Long lie: alert sent | Urgent alert (§4.2) | #40 relay, #41 app or interim channel |
| 6 | The family member taps "I'm checking". | Acknowledged by [name] | Acknowledged | Acknowledgement through the relay |
| 7 | The person gets up. | Movement seen; alert still open | Movement seen at [time] | — |

Risks to the demo, in the order they could stop it:

- **AGC.** In the ESP32 receiver model, which is unverified, the radio's
  automatic gain steps make a seated person read as empty in every frame
  at the model's default setting, and the fix the repository expects is
  reading the gain register rather than a DSP change (**SIM**,
  `docs/RESULTS.md`, "ESP32 receiver model"; `NOTES.md` §9). The first
  bench session (#45) settles it.
- **The burst.** Whether a staged, careful lowering onto a mat reaches
  motion index 8 is unknown until there are real recordings (#46). If it
  does not, the threshold is not changed for the video without saying
  so on screen.
- **Lying out of sight.** On one link a lying person to the side of the
  link line is weakly seen (§3.3). Two links, and the sensor positions
  shown on screen.
- **The channel.** Telegram works in tests against a local stand-in
  only (`docs/EVIDENCE.md` §2.6). If #41 is not ready, open question 7
  decides what the phone shows.

The video must not say or imply fall detection, medical monitoring,
accuracy figures, or that it works in any room.

### 7.2 In the MVP software

- Room presence and the home-level view (§3.1).
- Long lie, per room (#38).
- Sensing status, including "no contact with the home" (§3.6).
- The relay with acknowledgement and escalation (#40).
- The phone app, or an interim channel (#41).
- The daily routine summary without unusual quiet (#39); unusual quiet
  follows once there are 14 days of data.
- Pause and consent for the monitored person (§5.5), at least a pause
  button.
- One-command home setup (#42).

### 7.3 Deferred

- The breathing trend beyond a demo display; breathing pause and rate
  alarms as family alerts.
- Heart rate, in any form.
- The presence hold on by default (PR #31 ships it off).
- Possible collapse, instantaneous fall detection.
- Care homes, homes with more than one resident, pets.
- Two-way voice, a local "are you OK?" prompt (open question 9).
- Clinician access, integrations with health records.
- Learned models (ADR-007).

---

## 8. Open questions for the founder

1. **The budget and N.** Are the targets in §4.6 and the starting N in
   §3.3 acceptable to families? Ask before #38 measures.
2. **A door sensor?** A cheap contact sensor on the front door would
   tell "away" from "in an unsensed room". It adds a non-WiFi sensor;
   is that acceptable?
3. **How does the monitored person pause?** A physical button, the
   phone, or voice. Which can every intended user manage?
4. **Who consents for someone with dementia?** Under s. 9 a lawful
   guardian consents for a person with a disability who has one. Who
   consents when there is no guardian, and should Saans be installed at
   all then?
5. **Vitals in the first version?** Offer the breathing trend at all,
   given §6.4 point 2?
6. **Escalation beyond family.** Neighbour, building security, a paid
   responder service? Who calls them, and how?
7. **The interim channel for the demo.** Telegram (built, tested only
   against a stand-in), WhatsApp (business API), or wait for the app
   (#41)?
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
12. **Guests mode.** Should the long lie stay on when the person has
    guests (§5.3)?

---

## 9. Sources

Repository (all performance numbers **SIM**): `docs/RESULTS.md`
sections named in the text; `NOTES.md` §8.2, §9, §12, §14 (and §14 on
branch `saans/hold-after-fall` for the presence hold, PR #31);
`docs/EVIDENCE.md`; `docs/CAPABILITY_LIMITS.md`; `SAANS_PLAN.md` §2,
§8, §10; `rust/crates/saans-health/src/detectors.rs`, `types.rs`,
`system.rs`; `rust/crates/saans-gateway/saans.example.toml`; ADR-001,
005, 006, 010, 011, 012.

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
