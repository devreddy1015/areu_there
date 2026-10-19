# ADR-010: The audit log is kept for a year, then deleted

**Status:** accepted, 2026-10-19. **Decided, not built** — `audit_retention_days` does not exist in the code, and the sweep described below is what this ADR asks for, not what runs today.

## Context
`audit-YYYY-MM-DD.jsonl` is the file that answers "who read this space's
breathing data". `AuditLog::record` appends one JSON line per login,
failed login, logout, denial and `vitals_read`, each carrying the
principal's name, their role, the space id and a local timestamp
(`rust/crates/saans-gateway/src/audit.rs`). It lives in `data_dir`,
beside the event logs.

It has a writer and no end. The hourly sweep in `Hub::housekeeping`
calls `store::prune_logs`, which finds files through `log_file_day`, and
that function matches `events-` and nothing else (`store.rs:107`). So an
event log is deleted once every line in it is older than
`event_retention_days` — 30 by default (`config.rs:208`,
`saans.example.toml:17`) — and audit files accumulate for as long as the
gateway has a disk.

That asymmetry was decided, and this ADR reverses half of it. `audit.rs`
says so at the top of the file: "Separate from the event log on purpose.
`events-*.jsonl` records what the *building* did and is deleted on the
DPDP retention clock; `audit-*.jsonl` records what *people* did with the
system, which is the record you need when someone asks who looked at a
space's breathing data. Different question, different file, different
lifetime." `docs/UI_GAP_ANALYSIS.md:173` records the same thing as an
open item: "not swept at all yet, deliberately — audit records usually
outlive the events they describe, but 'forever' is not a policy either."

So the reasoning being superseded is not an oversight, and the rule this
ADR adopts — that audit retention must be at least event retention —
comes directly from it. What changes is only the second half of that
sentence: "outlive" was right, "forever" was the part nobody chose.

Both sides of this are real. The audit log exists so that reads of health
data can be accounted for: a space that consented to breathing monitoring
is owed an answer when it asks who looked. That argues for keeping the
file. But the file is itself a record of who looked at whom — named
people, timestamped, per space — and under India's DPDP Act 2023 a store
of that kind carries consent, purpose and retention obligations
(ADR-001). Kept forever, the accountability mechanism becomes a permanent
register of staff behaviour, which has the shape of the thing it was
built to prevent. ADR-005 says no identities anywhere; that rule is about
the people being sensed, and the audit log is the one file that names the
people doing the sensing. A retention limit is what stops that exception
growing into a second surveillance system.

## Decision
`audit_retention_days`, default 365. It is validated like its sibling —
at least 1 — and additionally required to be at least
`event_retention_days`: while an event is still on disk, the record of
who read it should be too.

It is swept by the mechanism already there, not a second one. Both
families live in one directory, so one sweep over that directory covers
both. `prune_logs` and `log_file_day` currently hard-code the `events-`
prefix (`store.rs:107`, `store.rs:128`); they gain it as a parameter, and
`Hub::housekeeping` calls the sweep twice on the same hourly timer
(`RETENTION_CHECK_S`, `hub.rs:33`), once for `events-` with
`event_retention_days` and once for `audit-` with `audit_retention_days`.
The day-boundary rule is unchanged: a file is removed only when
`today - (day + 1) >= retention_days` (`store.rs:133`), so it survives
until every line it could possibly hold is past the deadline.

365 is a choice, not a derivation. It is long enough to answer a
complaint about something that happened in a previous term, and short
enough that "we keep an audit log" does not quietly mean "forever".
[UNVERIFIED: this repository cites no statutory retention period for
access logs under the DPDP Act 2023, and this ADR does not claim one.]
No figure here is a measurement; they are configuration constants read
out of the files named above.

## Consequences
- After a year, "who read rest-room-1 last March" has no answer. That is
  the cost, accepted deliberately: the alternative is a file that never
  stops growing and never stops naming people.
- Volume is bounded by the throttle, not by traffic. `vitals_read` is
  written at most once per (principal, space) per hour
  (`VITALS_READ_EVERY_MS`, `audit.rs:27`), so a year of lines scales with
  principals × spaces × hours, not with requests; a dashboard polling at
  2 Hz costs one line an hour.
- Deletion happens within an hour of the gateway running past a deadline,
  not at midnight. A gateway that was off for a month deletes everything
  then due at its first sweep after it starts.
- Only the live, persisting path sweeps. Replay never touches `data_dir`,
  and the replay and test servers build the audit log with no directory
  at all (`HttpDeps::without_audit`), so neither writes nor deletes.
- The writer and the sweep cannot collide: `AuditLog` holds an open
  handle for the current local day only, and a file does not become
  eligible until a full retention period after its last possible line.
- `/api/health` must report the audit retention alongside the audit
  counters, as the event log already reports `retention_days`. An
  operator who can see what has been written but not what will be kept
  cannot check this decision is in force.
