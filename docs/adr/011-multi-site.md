# ADR-011: One gateway per site, and no hosted control plane

**Status:** accepted, 2026-10-19. **Decided, not built** — nothing in this
ADR exists in the code, and nothing in it is scheduled.

## Context

A gateway serves one site. It holds that site's `saans.toml` — its spaces,
their `vitals` consent flags, its users and tokens — writes that site's
`events-*.jsonl`, `audit-*.jsonl` and `history.json`, and is deployed by
copying one binary onto a box (ADR-008). Nothing in the system knows that
a second gateway exists.

Two different things are called history here and only one of them is
transient. The minute-by-minute series behind `/api/history` is bounded
in memory and never written down — `saans.example.toml:28-33`, "the
window IS the retention policy and a restart is a delete". But
`history.json` is a different file: one `DailySummary` per space per day,
holding `resting_bpm_median`, `occupied_s`, `motion_bouts` and `pauses`
(`saans-health/src/detectors.rs:234-243`), written atomically every ten
minutes and on shutdown (`store.rs:211-231`, `hub.rs:32`), read back at
startup, and kept even for spaces that have since been removed from the
config (`hub.rs:173-175`). `resting_bpm_median` is derived from consented
vitals. There is no retention key for any of it and `prune_logs` does not
touch it.

That is worth stating plainly in an ADR about not creating stores of
health data: one already exists, on every gateway, with no deletion
policy. It is small and it is per-space rather than per-person, but it is
the same kind of object as the thing this ADR goes on to rule out, and it
is not covered by ADR-010 either.

Someone running six buildings wants one page showing all six. That is a
real need, not a vanity one: six sites is six logins, and the person who
has to check them at 02:00 will otherwise write a scraper with an admin
token. It is also the point where this architecture is easiest to lose,
which is why it is worth deciding the direction before anyone has two
sites. That is now: **no board has ever run this firmware, every number in
this repository is simulated, and there are zero sites.** Deciding early
costs a page of writing; undoing a control plane later costs the argument
the project is built on.

Capacity is not what forces the question. One gateway core kept up with
~1,200 links in a bench on synthetic frames with no radio involved
(`NOTES.md` §7.3, line 421 — **SYNTHETIC**), and a room's twelve links are
1.6 kB/s (ADR-004). A building fits in one gateway comfortably. What
forces the question is organisation: separate buildings have separate IT,
separate consent owners, and often separate networks.

### Aggregation is its own privacy object

ADR-001 says a server that receives raw CSI "becomes a store of health
data whether or not anyone intended that". The same sentence holds one
layer up with a different object. One building's per-space occupancy,
held in that building, is a facilities signal. Forty spaces across a
campus, in one place, over months, is a record of who works late in which
lab and which hostel wing empties on which nights. ADR-005 keeps
identities out of every output and the gateway has none — but a per-room
occupancy series read alongside a timetable and a door-access log is
re-identifying at room granularity, and we control neither of those other
two datasets.

Centralising FeatureFrame-derived state therefore does not breach
ADR-001's letter, and that is precisely why it needs a decision of its
own. [UNVERIFIED: whether operating such a plane would make its operator
a separate data fiduciary under the DPDP Act 2023 — not checked against
the Act.]

## The options

**A hosted control plane that gateways report up to.** Each gateway dials
out and pushes state; one server renders every site. It is the shape
everyone expects, it crosses NAT without an inbound port, and it gives one
place to ship updates from. It also creates the store described above, in
a place the site operator cannot unplug. Consent under ADR-005 is given
per space, by the space owner, with the written consent of occupants, to a
gateway in their building; it is not consent to a third party's server,
and there is no mechanism to collect or revoke a second one per space. It
re-opens retention: the gateway deletes an event log once every event in
it is older than `event_retention_days` (30 in the example config), keeps
the `/api/history` series only in memory, and keeps `history.json`
indefinitely — three different policies, one of which is "none" — and a
plane that copies any of them upward owes an answer for each somewhere
else. And it moves the answer
to "where is our data" from "that box, and here is its config" to "ours
and theirs", which is the one credibility asset this project has. Today
the only thing we publish off-site is a copy of the site with no data in
it, which says so when asked for data (ADR-009).

**Read-only federation.** One gateway is configured with the addresses of
others and a scoped token at each, and renders a combined view by reading
them. No new process, no new organisation, no new store: each site keeps
its config, its users, its consent flags, its logs and its retention
clock, and the aggregate exists only while someone is looking at it. The
properties that make this work already exist. Authorisation stays at the
site that owns the space — a space outside a principal's scope answers
404, not 403, so the scope list is not a directory of the building
(`README.md:256`) — and a read of consented vitals is written to the audit
log by the gateway that served it, which is the right file and the right
box. Revocation is a one-line edit on the site's own machine, with no
one's cooperation needed. It needs an inter-site network path and TLS,
which campus IT must agree to; and under ADR-008 the browser must keep
talking to one origin, so the reading gateway proxies rather than the page
fetching across origins. Its real residual risk is that whoever holds
tokens for forty sites can assemble the campus record in a loop anyway.
The difference is that the record is not created by default, not retained
by default, and any site can withdraw unilaterally.

**Nothing: each site stands alone.** Costs the operator real work, and
will be worked around badly. Costs nothing to keep, and there is no
evidence anyone has a second site.

## Decision

1. Today, each site stands alone. No cross-site feature ships.
2. **Gateways reporting up to a hosted control plane is ruled out**, by us
   or by anyone else, not deferred. The claim that the data is in your
   building and you can unplug it is the product.
3. When multi-site is built, it is read-only federation: a gateway holding
   scoped, revocable tokens at other gateways, proxying rather than
   copying, retaining nothing it does not already retain, with every
   authorisation decision made by the site that owns the space.
4. No API and no schema is designed here. `/api/*` and the four roles are
   already the interface a federation would read; stability and role
   filtering there are now load-bearing for a second reason.

## What this does not change

Nothing on the node, in the DSP, in the FeatureFrame, in the consent gate,
in link fusion or in the detectors. S0–S5 and their exit gates are
untouched, and so is the one thing that actually blocks this project:
hardware bring-up at S5. This is a decision about the gateway's edges,
written down so it does not get made by accident during a demo.

## Consequences

- The honest answer to "we have six buildings" is six gateways and, for
  now, six dashboards. Nobody should be told otherwise.
- A federated overview must label which site each value came from and when
  that site was last heard from, and a site that has gone quiet must read
  as stale, never as empty — the same distinction the dashboard already
  keeps between connection state and data freshness (`README.md:229-233`),
  and the same failure class as an inert filter making a room read empty
  forever (`SAANS_PLAN.md` §4, port rule 5).
- A federated vitals read will be audited against the federation token,
  not the human behind it. That is a real gap in the audit story and it
  belongs in the ADR that builds this, not in a footnote afterwards. The
  entry itself lands at the site that served the read and ages out on
  that site's clock (ADR-010), which is where it should be.
- Documenting the `occupancy` role and per-space scoping as the supported
  way to read another site is cheaper than trying to stop people from
  scripting it with an admin token.

## What would have to be true to revisit

Reopen the hosted plane only if **all** of these hold, and by rewriting
this ADR rather than extending it quietly:

- An operator with more than one running site asks for it, after S7. The
  need is observed, not imagined.
- Their sites cannot reach each other at all — isolated networks with only
  outbound internet. This is the one fact read-only federation cannot
  route around, and the only strong technical argument for a plane.
- A per-space consent mechanism for export to a third party exists, is
  revocable at the space, and the spaces concerned have used it.
- Retention and audit on the plane are written down and are at least as
  strong as the gateway's.

Separately: the first thing that would push federation past read-only is
cross-site alert acknowledgement. Acknowledgement is not finished even for
one site (`docs/COMPLETION_REPORT.md` §9, item 2), so that is a reason to look
again later, not a reason to build anything now.
