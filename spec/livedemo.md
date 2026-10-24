# Live Demo v1: the room stream, its page and `roomsim-eval`

**Status:** spec for Phase 5, 2026-10-24. Revised after a four-lens
adversarial review: 48 findings, of which 46 were confirmed or partly
confirmed and adopted, and 2 refuted. It builds on spec/roomsim.md (Room
Sim, SIM data), spec/locate.md (Locate, estimates with radii) and ADR-008
(what may be drawn, and to whom).

**Owner decisions, 2026-10-24** (asked and answered):
1. **API.** Additive public-API changes are approved:
   - `saans-gateway` gains `Principal::can_see_position` (ADR-008 d10)
     and a room-stream handle on `Shared` (§2.5).
   - `saans-locate` gains a streaming driver (§8.1: `RoomSimSetup` and
     `RoomSimStream`), a score function (§8.2) and a records file format
     (§8.3). The gateway and `saans roomsim-eval` share all three.
   - `Shared::mode` becomes the closed enum `Mode { Sim, Replay, Live }`
     (approved 2026-10-21).
   
   No other existing signature changes.
2. **Frame type adapted to ADR-008.** The brief's sketch predates the ADR.
   - Person estimates are keyed by estimator (`wlm`, `rti`), each an
     estimate with its radius or an abstention, never a bare point.
   - Truth people are named by index and sent in SIM only.
   - The phone has `rtt` and `rssi`, each with its own radius.
   - Every block carries its own timestamp.
3. **Modes.** SIM runs Room Sim and Locate in the gateway. REPLAY plays
   back a recorded Locate run written by `roomsim-eval`; it is labelled
   SIM and its truth is stripped. LIVE room frames are not produced at all
   (§7).
4. **Baseline (ADR-008 d9).** RTI is never drawn without the baseline
   beside it. The baseline may be drawn alone, with RTI's abstention shown.

**Decisions taken here.** Each is flagged in the phase report.
1. The wire spelling of modes stays lower case (ADR-008 d15, "as
   before"). The page shows upper case. A `label` field sits beside `mode`.
2. The stream is `/api/roomsim`.
   - Its handshake checks `Origin`.
   - On an auth-off gateway it also checks that `Host` names loopback
     (§2.2). This closes DNS rebinding for this one endpoint. The
     gateway-wide fix in ROOMSIM.md item 2 remains the owner's call.
3. Each session gets its own OS thread and one simulation, with at most
   `ROOMSIM_MAX_SESSIONS = 4` sessions at once.
4. Frames are never dropped. A slow client slows only its own session.
5. Radii are shown as **uncalibrated model radii**, never as "95%"
   (spec/locate.md §8).
6. Truth is drawn from each record's own truth, so the drawn gap between
   an estimate and the truth equals the printed error.
7. Every SIM session also runs the empty room of the same scene and seed.
   Coverage is therefore always shown beside the empty-room false-alarm
   rate (ADR-005 §5, spec/locate.md §12a).
8. REPLAY shows no truth and no error figures. It shows coverage, the
   radius median and the recorded empty-room false alarms.
9. The variants are named `declared`, `with_walker` and `empty`, not
   `scene`/`walker`. The shipped hall has a `Person.id` of `walker`, so
   that name would be an id-like value in every frame.
10. The page's default variant is `with_walker`. The shipped bedroom and
    office people are still, so `declared` mostly shows abstentions. The
    variant is always shown.
11. `can_see_position(Some(_))` is false in Phase 5, because d10's
    per-space opt-in does not exist yet (§2.3).
12. The simulation crates build at `opt-level = 2` in the dev profile
    (Rules).
13. The gateway's pcap `--replay` stays unlabelled `replay`, and
    spec/roomsim.md §16 and ROOMSIM.md item 10 are amended to match (§1).
14. No page text, control or hash key calls this stream "live".
15. Nothing in this spec writes a position from REPLAY or LIVE input to
    disk (ADR-008 d11).

**Rules.**
- No unsafe Rust.
- New gateway code lives behind the existing `roomsim` cargo feature,
  which is off by default (ADR-008 d13). That feature now also pulls in
  `saans-roomsim`, `saans-locate` (with its `roomsim` feature) and
  `saans-scene` (without `write`).
- `saans-gateway` joins READERS in `saans-scene/tests/boundary.rs`.
- The page keeps its CSP and Trusted Types policy unchanged.
- Every number shown is labelled.
- `rust/Cargo.toml` gains `[profile.dev.package.<name>] opt-level = 2` for
  `saans-roomsim`, `saans-locate`, `saans-scene`, `saans-sim` and
  `saans-dsp`.
  - Float results do not change: Rust contracts no FMA, and the existing
    `black_box` guards stay. Overflow checks and debug assertions stay on.
  - Why: CI runs every test in the dev profile. There, workspace crates
    otherwise build at opt-level 0, and a 120 s calibration takes 7-17 s,
    which is beyond the e2e harness's 10 s per-message timeout.

---

## 1. Modes and labels

`mode` says where frames come from. `label` says what kind of numbers they
carry (SCENE, SIM, REAL, SIM->REAL, LIVE).

| `mode` (wire) | Source | `label` | Truth and errors | Phone |
|---|---|---|---|---|
| `sim` | Room Sim + Locate in the gateway | `SIM` | yes | yes (simulated) |
| `replay` | a recorded Locate run (§6) | `SIM` | **no** (keys deleted) | yes (simulated, ADR-008 d16) |
| `live` | none in Phase 5 (§7) | — | — | — |

**Serialisation.** `Mode` serialises as `"sim"`, `"replay"` and `"live"`
everywhere: `hello`, `/api/me`, `/api/health` and the room stream.

**Badges.** Every view renders its badge by an exhaustive mapping over
these three values, with no default branch. An unknown value shows
"unknown mode" as an error, never as a mode.
- The dashboard takes the mode from `hello` and `/api/me`, which report
  the gateway's own mode.
- The room page takes it only from room-stream messages (§3.4).

**The gateway's own mode.** `Shared::mode` stays `live`, or `replay` with
`--replay`. The room stream's mode is set per session.

**Pcap replay.** `saans-gateway --replay F` of a pcap stays `mode: replay`
with no `label` in Phase 5, because the gateway cannot tell a Room Sim
export from a capture.
- The last sentence of spec/roomsim.md §16, and ROOMSIM.md item 10, are
  amended to: "the gateway's pcap replay is labelled SIM only when
  `hello`, `/api/me` and `/api/health` gain a `label` (later work)".
- The dashboard's `replay` tooltip reads "Replaying a capture file — not
  live sensors; a .sim.pcap is simulated (SIM)".

## 2. Access

### 2.1 Endpoint
`GET /api/roomsim` (WebSocket) is compiled only with the `roomsim`
feature. Without it the path returns 404, like any unknown endpoint.

The handshake checks run in this order:
1. Origin and Host (§2.2).
2. The principal.
3. `can_see_position` (§2.3).
4. A session slot (§2.4).

### 2.2 Origin and Host
These are checked first, before authentication. A refusal is answered 403
and is not audited. The upgrade is refused unless all of these hold:
- An `Origin` header is present and is not `null`.
- With its `scheme://` removed, `Origin` equals the `Host` header as sent:
  - compared ASCII case-insensitively;
  - no default ports filled in;
  - IPv6 hosts stay bracketed.
- On a gateway with authentication off, where every caller is the local
  admin, the host part of `Host` is `localhost`, `127.0.0.1` or `[::1]`.

What this covers:
- It refuses cross-site WebSocket hijacking, including from another port
  on the same host.
- On an auth-off gateway it also refuses DNS rebinding. There, Origin and
  Host both carry the attacker's name and would otherwise match.
- This holds for this endpoint only. ROOMSIM.md item 2 stays open for the
  others.

**Reverse proxies.** A proxy must forward the browser's `Host` unchanged
(nginx: `proxy_set_header Host $host;`). RUN.md says so.

**Non-browser clients**, including the e2e harness, send a same-origin
`Origin`.

### 2.3 Principal and position scope
**Principal.** It is resolved as for `/api/live`: a bearer token or a
session cookie, or the local admin on an auth-off loopback gateway. A
missing or invalid credential gets 401, audited by `principal()` as today.

**Scope.** Next, `Principal::can_see_position(None)` must be true.
Otherwise the request gets 403 and an audit `Denied` record:
- `who` and `role` are the principal's;
- `detail` is `{path:"/api/roomsim", why:"position"}`;
- the doc comment of `Action::Denied` is widened to cover role refusals.

`can_see_position(space: Option<&str>) -> bool`:
- **`None`** (a simulated scene bound to no configured space): true for
  role `admin` or `safety`, false for `ops` and `occupancy`.
- **`Some(_)`**: false in Phase 5, because d10's per-space opt-in does not
  exist yet.
- The change that adds the opt-in will make `Some(id)` true for "role
  `admin` or `safety`, `can_see_space(id)`, and the space's opt-in".

`ops` and `occupancy` get no room stream at all, links included.
`/api/me` gains `can.position` (the answer for `None`) for every caller,
and it is `false` for anonymous callers.

### 2.4 Sessions
**One simulation per connection.** At most `ROOMSIM_MAX_SESSIONS = 4`
session threads (§3.0) exist at once. REPLAY sessions count, and so do
sessions that are still calibrating.

**Taking a slot.** After the checks of §2.2 and §2.3, the handler takes a
slot with `fetch_add` on a per-gateway counter held in the room handle
(§2.5), never in a `static`.
- If the previous value was already the maximum, it undoes the increment
  at once.
- The slot is an RAII guard. It moves into the `on_upgrade` closure and
  then into the session thread.
- It is released when that thread exits, not when the socket closes. A
  failed upgrade drops the closure, and the slot with it.

**When no slot is free**, the upgrade still completes. The server then:
1. sends `{type:"error", mode, label, param:"session", message:"4 room
   simulations are running; try again later"}`;
2. closes with code 1013;
3. builds no simulator.

**When a session ends.** A session ends when:
- its socket closes;
- the gateway stops;
- it has been paused or ended for 10 minutes. It then sends
  `{type:"error", …, param:"session", message:"idle for 10 minutes"}` and
  closes with 1000.

Its thread exits within 0.5 s of any of these.

**Input limits.** The upgrade sets `max_message_size(1024)`. A binary
message closes the socket with 1008.

### 2.5 The room handle and persistence
**The handle** is `#[cfg(feature = "roomsim")] pub room: RoomStreams` on
`Shared`. It is an opaque type with private fields:
- the per-gateway session counter (§2.4);
- the REPLAY records loaded by `--room-replay` (§6), in a `OnceLock`.

It has no public method that yields a frame, a record or a position.

**Where frames go.** A room frame is written only to the socket of the
session that computed it. It never travels on `Shared::live_tx` or
`Shared::outputs`, and it is never stored in `Published`. A unit test
asserts that a running session puts nothing on either channel.

**Persistence.** Nothing from the room stream is written to disk, the
event log or the history. The only exception is the audit `Denied`
entries of §2.3, which carry no position (ADR-008 d11).

## 3. Session

### 3.0 Where a session runs
**The thread.** Each session, SIM or REPLAY, runs on its own `std::thread`
named `roomsim-session`.
- It never runs on the gateway's two runtime workers, which carry UDP
  ingest and the 0.5 s tick.
- It never runs in `spawn_blocking`, because runtime shutdown waits for
  every started blocking task.

**What it owns.** The thread owns the setup and the stream (§8.1, held on
its stack) and the session clock: an `Instant`, with `recv_timeout` until
the next 100 ms deadline.

**The connection task.** The connection's async task only parses commands
and forwards pre-serialised frames.
- Commands reach the thread over a `std::sync::mpsc` channel.
- Frames come back over a `tokio::sync::mpsc` channel of capacity 1, sent
  with `blocking_send`.

**Stopping.** The thread exits within 0.5 s of wall time after either:
- the command channel disconnects (the socket closed);
- the gateway's stop signal is set.

This includes calibration, which polls a stop flag before every
calibration transmission (§8.1).

**Cost.** One hall session at 8x costs about 65% of one 3.0 GHz Xeon core
in release, so four sessions cost about 2.6 cores. A slower host shows up
as `effective_speed` below `speed`. Nothing has been measured on the
target gateway yet (§12).

### 3.1 Opening
**Query parameters:**
- `scene`: an embedded scene id. Required.
- `variant`: `declared` (the default), `with_walker` or `empty`.
- `seed`: a u64, default 0.
- `replay=1`: REPLAY (§6).

**The variants**, built the way `locate_measure` builds its own:
- `empty` sets `n_people = 0`. The phone then stays at its scene start, at
  z = 1.0 (Room Sim §9).
- `with_walker` sets `n_people` to the scene's count + 1, and sets
  `phone_carrier` to the added person, who enters at the door and walks.
- `declared` takes the scene's people as they are.

Room Sim and Locate otherwise take their defaults, with
`duration_s = ROOMSIM_RUN_S = 120`.

**Refusals after the upgrade.** These are sent as an `error` message
(§3.4) naming `param`, followed by close code 1008:
- a bad or unknown parameter (an unknown key is named as `query`);
- a scene that is not embedded;
- a refusal from `RoomSimSetup::new`.

The message never echoes the given value.

### 3.2 Clock
**Quanta.** Simulated time advances in integer quanta of 0.025 s. `q`
counts them, and the session time is `T = float(2*q) * slot_s`. This is
Room Sim's own formula, and it is exact for `slot_s = 0.0125`. That is
the only `slot_s` in Phase 5: sessions use Room Sim's defaults, and the
§6 reader refuses any other value.

**Stepping.** A run's first frame is at `q = 0` (so a restart shows
`t = 0`). After it, every 100 ms of wall time, a playing session:
1. sets `q = min(q + 4*speed, q_end)`, with `q_end = 40 * duration_s`
   (exact; the §6 reader refuses a duration for which it is not an
   integer);
2. advances the stream to `T`;
3. sends one frame.

The speeds are `{0.25, 0.5, 1, 2, 4, 8}`, adding 1, 2, 4, 8, 16 or 32
quanta per frame. The default is 1, and a session starts playing.

**Deadlines.** Each deadline is the previous one plus 100 ms. A deadline
missed because computing or sending a frame took too long is skipped, not
caught up.

**`effective_speed`** is the simulated seconds advanced per wall second
over the frames sent in the last wall second. It is `null` until one wall
second has passed in the current run, or since the last resume.

**Frames are never dropped.**
- The frame channel holds one frame. The thread computes the next frame
  only after the connection task has taken the previous one.
- While the thread waits, the clock does not advance.
- So `seq` has no gaps, and every record appears in exactly one frame's
  `new` block.
- A connection that accepts no frame for 10 s is closed with 1008. The
  connection detects this with numbered WebSocket pings, sent at most once
  a second and only after a message went out: a ping left unanswered for
  10 s closes the socket. Browsers answer pings on their own; a slow
  client whose pongs keep arriving is kept.

**Pause.** When paused, `q` stops. One frame with `paused: true` is sent,
then one heartbeat per second. A heartbeat is an ordinary frame with the
same `t`, `seq + 1` and empty `new` lists.

**End.** When `q` reaches `q_end`, that frame has `ended: true` and
`t = duration_s`. No frame follows until a `restart` or the close (§2.4
idle rule).

### 3.3 Commands (client to server, JSON text)
- `{"cmd":"play"}`
- `{"cmd":"pause"}`
- `{"cmd":"speed","value":v}`, with `v` one of the six speeds
- `{"cmd":"restart"}`

**Unknown commands.** Anything else gets an `error` message naming `cmd`,
or `value` for a bad speed. The session then continues.

**Order.** Commands are applied in arrival order, before the next frame is
computed. During `calibrating`, `play`, `pause` and `speed` set the state
the run starts in, and `restart` does nothing.

**Restart.**
- `restart` ends the run and starts a new one at `q = 0`, with the same
  scene, variant and seed. It keeps the play/pause state and the speed.
- In SIM it is a new stream from the same setup (§8.1), so it needs no
  new calibration and sends no new `calibrating` state. In REPLAY it
  rewinds to the first record.
- The new run's first frame goes at the next 100 ms deadline, so a burst
  of restarts costs no extra frames. A restart before the session's
  first frame does nothing.
- `seq` keeps counting, and `run.index` goes up by one.

**Fixed for the session.** A client cannot change scene, variant or seed
within a session. It reconnects instead.

### 3.4 Messages and states
**Every message carries `mode` and `label`.** The page renders its badge
from them only.

- **`state`:** `{type:"state", state, mode, label, scene, run}`. It is
  always the first message of a session.
  - In SIM the state is `calibrating`. This covers the calibration
    simulation and the empty-room run (§5.3), together about 2-5 s in
    release and in the dev profile of the Rules.
  - In REPLAY the state is `ready`.
- **`frame`:** §4. Its `state` is `warming_up` until the first Locate
  tick (`t = window_s`), with `persons`, `links` and `truth.persons` set
  to `null`. After that its `state` is `running`.
- **`error`:** `{type:"error", mode, label, param, message}`. Its `mode`
  is the requested one: `replay` when `replay=1` was given, else `sim`.

## 4. The frame (server to client, JSON text, 10 Hz)

```
{
  "type": "frame",
  "seq": n,                         // per connection, from 0, +1 per frame
  "run": {"index": r, "variant": "declared"|"with_walker"|"empty",
          "seed": s, "duration_s": D},
  "t": T,                           // the session clock, SIM seconds
  "mode": "sim" | "replay",
  "label": "SIM",
  "scene": {"id", "name", "sha256", "provenance"},          // ADR-008 d4
  "speed": v, "effective_speed": e | null, "paused": bool, "ended": bool,
  "state": "warming_up" | "running",
  "persons": {"t": t_k, "wlm": Outcome, "rti": Outcome} | null,
                                    // the latest tick with t_k <= T
  "links": {"t": t_k, "items": [{"tx", "rx", "y", "gate", "disturbed"}]} | null,
                                    // the same tick, canonical link order
  "phone": {"t": t_e, "rtt": Outcome, "rssi": Outcome,
            "reports": [{"node", "rtt_m", "rssi_dbm"}]} | null,
                                    // the latest epoch with t_e <= T;
                                    // the key is absent when the scene has no phone
  "new": {                          // every record with T_prev < t <= T
    "persons": [{"t", "wlm", "rti"}],
    "phone":   [{"t", "rtt", "rssi"}]
  },
  "truth": { ... },                 // mode sim only; the key is absent otherwise
  "stats": { ... }                  // §5
}
```

**`T_prev`** is the previous frame's `t` in the same run. On a run's first
frame, including after a restart, it is −∞, so the phone epoch at
`t = 0` is included. Trails are built from `new` only.

**`Outcome`** is spec/locate.md §11's encoding:
- an estimate, `{"estimate": {"x", "y", "radius_m", "cov_xx", "cov_xy",
  "cov_yy", "confidence", "radius_calibrated"}}`;
- or an abstention, `{"abstain": {"reason"}}`.

There is no way to send a position without its radius.

**Links.** The fields of a link item:
- `y` is the measurement Locate used at that tick, in dB², or `null` when
  it is not finite.
- `gate` is `wlm_gate_z * null_sigma + wlm_floor` when `null_sigma > 0`,
  else `null` (such a link can never be disturbed).
- `disturbed` is `baseline::disturbed(y, null_sigma)`, WLM's own test
  (spec/locate.md §5 step 1). WLM abstains `no_disturbance` exactly when
  no item is `disturbed`.

**Phone reports.** In a report, `rtt_m` and `rssi_dbm` are `null` when
the node gave no report. `blocked` is not a measurement: it is computed
from the true positions (spec/locate.md §6.2), so it lives in `truth`.

**Truth** (mode `sim` only). Each part is its record's own truth (spec/locate.md §11), never a
later transmission:

```
"truth": {
  "persons": {"t": t_k,
              "people": [{"i", "x", "y", "walking", "posture"}],
              "wlm": Score, "rti": Score} | null,
  "phone": {"t": t_e, "x", "y", "z", "blocked": [node ids],
            "rtt": Score, "rssi": Score} | null
}
Score = {"error_m": e, "person": i}       // persons: the nearest person, lowest index on a tie
      | {"error_m": e}                    // phone
      | {"unscored": "abstained" | "no_person"}
```

**Errors** are in metres and in the plane:
- a person estimate is measured against the nearest person in its tick's
  truth;
- a phone estimate is measured against its epoch's `(x, y)`.

An estimate published on a tick whose truth has no person is
`{"unscored": "no_person"}`. It is a false alarm (§5), never an error of
0, null or infinity.

**Rules for every frame.**
- A 10 Hz frame repeats the latest records unchanged, with their own
  timestamps. Nothing is interpolated, extrapolated or predicted (ADR-008
  d6).
- People appear only as `i`, the index, never by their scene file's
  `Person.id`. The phone is never named by its `Phone.id` (d3).
- `scene.id` and the node ids are declared geometry, and they are sent
  (d4).
- Redaction deletes keys, never nulls them.

A frame is about 4-9 KB at 1x (bedroom 4.4 KB, office 6.1 KB, hall 8.7
KB with its 56 link items, measured), and up to about 12 KB at 8x, where
`new` holds up to 2 ticks and 4 epochs.

## 5. Statistics

### 5.1 The block
`stats` is computed by the shared score function (§8.2). It is called on
every frame over all of the run's records with `t <= T`; there is no
separate incremental implementation.

```
"stats": {
  "wlm": PersonStats, "rti": PersonStats,
  "rtt": PhoneStats, "rssi": PhoneStats           // absent when the run has no phone
}
PersonStats = {
  "ticks": n,
  "occupied": {"ticks", "published", "coverage",
               "median_m", "p90_m", "inside_radius", "radius_median_m",
               "ref": {"median_m", "p90_m", "inside_radius"}},
  "unoccupied": FalseAlarms,                       // ticks whose truth has no person
  "empty_run": FalseAlarms                         // §5.3; absent in variant `empty`
}
PhoneStats = {"epochs", "published", "coverage",
              "median_m", "p90_m", "inside_radius", "radius_median_m",
              "ref": {"median_m", "p90_m", "inside_radius"}}
FalseAlarms = {"ticks", "seconds", "published", "per_h", "events", "events_per_h"}
```

### 5.2 Definitions
- **`occupied`** covers the ticks whose truth holds at least one person.
  **`unoccupied`** covers the rest. In Room Sim v1 that is every tick of
  an `empty` variant and none otherwise.
- **`coverage`** is `published / ticks`, or `published / epochs` for the
  phone. It is a fraction in [0, 1], and `null` when its denominator is 0.
- **Errors** are as §4. Every published estimate in `occupied` and every
  published phone estimate is scored.
- **Medians** are the middle value, or the mean of the two middle values
  for an even count.
- **`p90_m`** is the `ceil(0.9 m)`-th smallest of the `m` errors (nearest
  rank).
- **`inside_radius`** is the fraction, in [0, 1], of scored estimates with
  `error <= radius_m`.
- **`radius_median_m`** is the median `radius_m` of the published
  estimates in the block.
- **`ref`** gives the same figures for a fixed **reference point**: the
  footprint's area centroid (shoelace for an outlined polygon, as
  `locate_measure`). It is scored against the same truth for the same
  records, with each record's own `radius_m`. It is the score of an
  estimator that knows nothing (spec/locate.md §12a).
- Every figure is `null` when its count is 0.
- **`FalseAlarms`:**
  - `seconds = ticks / tick_hz`;
  - `per_h = published / seconds * 3600` (published ticks per hour);
  - `events` are maximal runs of consecutive published ticks;
  - `events_per_h = events / seconds * 3600`;
  - rates are `null` when `seconds` is 0.

### 5.3 The empty run (SIM)
When the variant is not `empty`, the session's `calibrating` state also
runs the `empty` variant of the same scene and seed for the whole
`duration_s`, through the setup's `empty_room()` (§8.1). It keeps only
that run's `unoccupied` figures for `wlm` and `rti`, as `empty_run`.
- Its records are never sent.
- It shares the session's calibration, because Locate's calibration run
  is the same for every variant (spec/locate.md §10.2 step 1).
- It costs one empty Room Sim run, about 0.9-2.3 s in release.

### 5.4 REPLAY
The REPLAY session runs the same score function on the file's records,
which include truth. It then deletes, from `occupied` and from each
phone block, `median_m`, `p90_m`, `inside_radius` and `ref`. `empty_run`
is the manifest's (§6).

It keeps `ticks`, `published`, `coverage`, `radius_median_m` and
`unoccupied`. Whether ticks were occupied follows from the variant,
which the manifest states, so it reveals no position.

## 6. REPLAY: recorded Locate runs

### 6.1 File
The file is JSON Lines, UTF-8, at most 64 MiB. Floats are written in
Rust's shortest round-trip form and read with serde_json's
`float_roundtrip`.

**Line 1, the manifest:**

```
{"type":"manifest","label":"SIM","spec":"livedemo v1",
 "scene":{"name","sha256","provenance"},"variant","seed","duration_s",
 "roomsim_config":{…},"locate_config":{…},"links":[[tx,rx],…],"has_phone",
 "empty_run":{"wlm":FalseAlarms,"rti":FalseAlarms}}
```

`empty_run` is absent for variant `empty`.

**Then the records, ordered by `t`.** At equal `t` the epoch comes first,
which is `run_roomsim`'s order.

- **Tick:** `{"type":"tick","label":"SIM","t","wlm","rti","links":[{"tx","rx","y","gate","disturbed"}…],"truth":[{"i","x","y","walking","posture"}…]}`.
  Its links are in the manifest's order.
- **Epoch:** `{"type":"epoch","label":"SIM","t","rtt","rssi","reports":[{"node","rtt_m","rssi_dbm"}…],"truth":{"x","y","z","blocked":[node ids]}}`.

The encodings are §4's. Truth stays in the file: records are SIM, of
simulated people (spec/locate.md §11). The file is written by
`saans roomsim-eval … --records-out FILE` (§10).

### 6.2 Reader
**The flag.** `saans-gateway --room-replay FILE` is a command-line flag.
- It exists only in a `roomsim` build, and it may be repeated, with at
  most one file per scene.
- It is read and validated after the config and before `--check` returns,
  so `--check --room-replay bad.jsonl` exits 2 naming the problem.
- The binary sets the room handle's records after `Gateway::start`, or on
  the `Shared` that `--replay --serve` builds.

**Refusals.** A file is refused, naming the line, when:
- the first line is not a manifest;
- any line's `label` is not `SIM`;
- `spec` is not `livedemo v1`;
- the scene's sha256 matches no embedded scene, or another file already
  covers that scene;
- `roomsim_config.slot_s` is not 0.0125, or `40 * duration_s` is not an
  integer;
- ticks, or epochs, are out of order, or a record has `t >= duration_s`;
- a key is unknown, or links are not in the manifest's order;
- an estimate fails spec/locate.md §8's publication test (finite `x`, `y`
  and radius; `radius_m > 0` and `>= min_plausible_radius_m` from
  `locate_config`), or has `radius_calibrated: true`, since Phase 5
  calibrates nothing;
- the file is larger than 64 MiB;
- a key is repeated within one object;
- `locate_config` is one `run_roomsim` would refuse with
  `roomsim_config` (`check_roomsim_config`, then `resolve_config`);
- a tick's truth does not have `roomsim_config.n_people` people, or, when
  that is unset, not as many as the first tick has. The gateway also
  refuses a file whose first tick does not have the embedded scene's
  people count when `n_people` is unset.

The writer refuses, before writing anything, any non-finite value except
a link's `y`, so every file it writes is one the reader accepts.

### 6.3 A REPLAY session
**Opening.** `replay=1` requires that:
- `scene` is the embedded scene of a loaded file;
- `variant` and `seed` are absent, because the file fixes them.

Otherwise an `error` names the parameter. A scene with no loaded file gets
`{param:"replay", message:"no recorded run for this scene"}`.

**Playback.** The session:
- takes `run.variant`, `run.seed` and `duration_s` from the manifest;
- sends `ready`;
- plays the records on the §3.2 clock, with play, pause, speed and
  restart;
- uses mode `replay` and label `SIM`.

**Redaction.** The session deletes every `truth` key and redacts `stats`
as §5.4 says.

## 7. LIVE

No code path produces a LIVE room frame or a LIVE phone. The room
stream's sources are Room Sim (SIM) and record files (REPLAY), and
`Mode::Live` is never a session mode.

ADR-008 d13's other gates are unmet: a surveyed scene bound to a space,
and a view of what is not covered. Adding a LIVE source is therefore a
later decision, and it must arrive with d13's config-load refusal.

A test asserts that a gateway running live (`Shared::mode == Live`) still
serves only `sim` and `replay` room frames.

## 8. saans-locate additions (feature `roomsim`)

### 8.1 `RoomSimSetup` and `RoomSimStream`
Room Sim's `Run<'a>` borrows its `Simulator`, so a single owning type
would be self-referential. saans-roomsim's API does not change, so the
driver is two types:

**`RoomSimSetup::new(scene, &RoomSimConfig, &Config, label, keep_going: &mut dyn FnMut() -> bool) -> Result<Option<RoomSimSetup>, PipelineError>`**
- It makes `run_roomsim`'s refusals and builds the occupied `Simulator`.
- It simulates the calibration run (spec/locate.md §10.2 step 1), calling
  `keep_going` before each calibration transmission. `Ok(None)` means it
  was abandoned.
- It keeps the calibration frames (`t >= duration_s`) that it fed to
  `calibrate_from`. The phase report measures their memory.

**`setup.stream(&self) -> Result<RoomSimStream<'_>, PipelineError>`**
- It starts a fresh run from `t = 0`.
- It builds a new `PersonLocator`, calibrated from the kept frames.
  `calibrate_from` is deterministic, so every stream starts in the state
  `run_roomsim`'s occupied run starts in.
- It also builds a new phone run and a new `Run` of the setup's
  `Simulator`.
- A setup can give any number of streams. A restart is a new stream.

**`setup.empty_room(&self) -> Result<RoomSimSetup, PipelineError>`**
- It gives the setup of the `empty` variant of the same scene, seed and
  duration (§3.1).
- That setup shares this one's calibration frames, because its
  calibration run is identical, so no calibration is simulated again.

**`stream.advance_to(t_end) -> Result<Advance, PipelineError>`**
- It consumes every transmission with `t <= t_end`, peeking one ahead.
- It emits every tick with `t_k <= t_end` and `t_k < duration_s`, plus the
  epochs fired, in `run_roomsim`'s order. Once the transmissions are
  exhausted it also emits the remaining ticks before `duration_s`, each
  as `t_end` reaches it, so a step still holds exactly the records with
  `t_prev < t <= t_end` (§4).
- It refuses a NaN or decreasing `t_end`, naming `t_end`.
- It returns `Advance { ticks, links, epochs }`:
  - `links` holds one `Vec<LinkValue { y, gate, disturbed }>` per returned
    tick, in canonical link order;
  - `y` and `null_sigma` are the very values that tick's WLM used, taken
    from a crate-internal tick function. No public `PersonLocator`
    signature changes.

**`run_roomsim` is re-expressed** as one setup and one stream advanced to
the end. Every existing golden and parity test must still pass,
bit-identical.

### 8.2 Score
`score(scene, ticks, epochs, has_phone, tick_hz) -> Scores`
- It computes the §5 figures over the records given.
- `Scores` serialises to §5's JSON; the caller adds `empty_run`.

`radius_scale(scene, ticks, epochs, has_phone) -> RadiusScales { wlm, rti, rtt, rssi: Option<f64> }`
- It gives the `ceil(0.95 n)`-th smallest `error / radius_m` over the
  scored estimates.
- It is `None` when `n < 20`.
- It is used only by `roomsim-eval` (§10).

### 8.3 Records
A writer and a reader of the §6 file, with the §6.2 refusals. The round
trip is exact. These add `serde` and `serde_json` (with
`float_roundtrip`) as optional dependencies under the `roomsim` feature.

## 9. The page (`/roomsim`)

### 9.1 Opening
**Before the simulation view is asked for.** The static view stays as it
is until the user presses "Run simulation (SIM)" or the URL has `#sim=1`.
Only then does the page open the WebSocket.
- `#variant=declared|with_walker|empty` defaults to `with_walker` on the
  page (decision 10).
- `#seed=N` defaults to 0.
- `#replay=1` selects REPLAY.
- Each of these has a matching control, survives the page's own hash
  rewrites, and is shown in the panel.

**Checking access first.** Before opening the socket the page fetches
`/api/me`:
- if `auth_required` and not `authenticated`, it shows "Sign in at / with
  an admin or safety account to run the simulation (SIM)" and opens no
  socket;
- otherwise, if `can.position` is false, it shows "Your role (<role>)
  cannot see simulated positions; an admin or safety account can" and
  opens no socket;
- only otherwise does it open the socket.

**When the socket fails.** A browser cannot see why a handshake was
refused.
- A socket that closes before its first message shows "The room stream
  could not be opened (signed out, not permitted, or not built into this
  gateway)".
- An `error` message followed by close 1008 or 1013 shows that message.
- The page never reconnects on its own.

**Wording.** No control, hash key or text calls this stream "live".

### 9.2 Lifecycle
**A layer of its own.** The simulation layer is its own scene group.

**Clearing.** The layer is cleared, holding nothing (d7):
- when the socket closes, with "Disconnected — nothing is shown";
- when `run.index` changes, which clears trails, circles, truth and the
  panel figures;
- on every scene change. A scene change closes the socket, and a new one
  opens only if the simulation view is still on.

**Wrong scene.** A frame whose `scene.sha256` is not the drawn scene's is
not drawn, and the page shows "Stream is for another scene".

**A gap in `seq`** cannot happen (§3.2). If one does, it is a protocol
error: the page clears the layer, closes the socket and shows "frames
lost — stopped".

**After `ended: true`** the last frame stays drawn, and the panel says
"Ended at t = 120.0 s (SIM) — Restart to run again".

**Rendering** happens on frame arrival only. Nothing moves between
frames.

### 9.3 Drawing
**Links.**
- Each declared node pair is one segment between the nodes' SCENE
  positions.
- **Its value `g`** is the larger, over its two directions, of
  `y / gate`. It is 0 when `y` is `null` or `<= 0`, or when `gate` is
  `null`.
- **Disturbed:** a pair is disturbed when either direction is.
- **Colour** comes from one fixed sequential scale over `g`, clamped to
  [0, 2].
- **The legend** reads "link disturbance: y ÷ WLM gate (SIM) — 0 at or
  below the empty room, 1 at the gate, 2 or more twice the gate".
- **Shape, so the gate reads without colour:** disturbed pairs are a
  solid band mesh 0.03 m wide, and the others are 1 px dashed lines.
- **The panel** states "links over the WLM gate: k of K pairs".

**Estimates** (persons and phone).
- **One function draws every estimate.** It draws nothing, and shows
  "estimate not drawn: radius", when `radius_m` is absent, non-finite,
  `<= 0` or `< 0.3` m (`min_plausible_radius_m`; ADR-008 d5).
- **What is drawn:** a ring of `radius_m` around `(x, y)`, with a centre
  mark.
  - The ring is never clipped to the footprint.
  - Any fill is at most 10% opacity.
  - Rings are drawn over the room, with depth testing off and a high
    render order, so no wall or furniture hides them.
- **Names and line styles:**

  | Estimate | Line style |
  |---|---|
  | "WLM (baseline)" | solid |
  | "RTI" | dashed |
  | "phone RTT" | dotted |
  | "phone RSSI" | dash-dot |

  Truth is drawn as a cross.
- **The label** reads "<name> ± r m · model radius, uncalibrated (SIM)".
  It adds "covers the whole room" when the circle contains every
  footprint vertex, and "centre outside the room" when the centre lies
  outside the footprint.
- **Labels are never hidden** by the label layer's decluttering. They
  outrank dimension, node and object labels; only the selection outranks
  them.
- **A live key**, independent of the Dimensions toggle, shows each style.
- **Framing.** While the simulation view is on, the plan view and Reset
  view fit the footprint together with every drawn ring.
- **The live card** lists each estimate as "(x, y) ± r m" or "abstained —
  <reason>". The radius can therefore be read even where the ring leaves
  the view.
- **The baseline rule.** RTI is never drawn unless WLM's outcome from the
  same `persons` block is drawn too, as an estimate or as its abstention
  (d9, owner decision 4).
- **Abstention.** An abstention removes that estimator's previous circle
  and shows its reason as text (d7).
- **Variant `empty`.** Every drawn person estimate is labelled "false
  alarm — the room is empty (SIM)". This comes from `run.variant` and
  needs no truth.

**Phone trails.**
- Trails are built from `new.phone` only.
- A segment joins consecutive published estimates of the same method.
- A new segment starts at every abstention.
- Trails are cleared when the run changes.
- The run bounds a trail: at most 600 epochs.

**Truth** (mode `sim` only). The toggle is off by default. It is not
offered in REPLAY, which shows "no truth in REPLAY". With truth on:
- people are drawn as crosses labelled "P<i+1> · truth (SIM)" at the
  `truth.persons` time;
- the phone is a cross labelled "phone · truth (SIM)" at the
  `truth.phone` time;
- each scored estimate gets a segment from its centre to its matched
  person or phone, labelled e.g. "WLM error to P2: 1.23 m (SIM)";
- the panel shows "Simulated people: N".

Drawn truth is the record's own truth, so each segment's length is the
printed error.

**Controls.** Play/pause, the six speeds and restart. `effective_speed` is
shown beside `speed` whenever it is lower.

### 9.4 Side panel
**Header facts:**
- the mode badge, from the latest room-stream message only, by
  exhaustive mapping: `sim` → "SIM", `replay` → "REPLAY · recorded SIM
  run", `live` → "LIVE". Any other value shows the "unknown mode" error;
- the label;
- the scene's name, hash and provenance;
- the variant and seed;
- `t` and the state.

**One block per estimator.** WLM is always shown beside RTI. Each block
shows:
1. "published k of n (coverage %)".
2. The median and p90 error. The median is shown only when `k >= 5` and
   p90 only when `k >= 10`; otherwise "too few to summarise (k)".
3. The radius median and the fraction with truth inside the radius.
4. Beside these, the same figures for "reference: the floor's centroid
   with the same radius — an estimator that knows nothing".
5. For WLM and RTI, always beside coverage: "empty room: x published
   ticks/h, y events/h (SIM)".

**Variant `empty`.** The person blocks are headed "False alarms — empty
room (SIM)" and show the `unoccupied` figures.

**REPLAY.** The panel shows coverage, the radius median and the
empty-room false alarms, with "no truth in REPLAY" in place of the error
figures.

Every figure is labelled SIM. Blocks stack at phone width.

### 9.5 Text, accessibility, fallback
**Text that changes while the simulation view is on.**
- The header gains a badge "SIM · simulated people, estimates from
  simulated WiFi".
- The scene badge's title, `#scene-help`, the canvas `aria-label`, the
  People row and the footer are reworded. None of them may say that
  nothing is inferred from WiFi, or that no people or estimates are
  drawn. They say instead that the room is declared, and that people and
  estimates appear only in the SIM view.
- RUN.md §4b, the `roomsim.rs` module doc and the e2e test
  `room_demo_is_served_self_contained_and_draws_no_one` change in the
  same commit.

**Accessibility.** The live card's figures are not in an `aria-live`
region. A separate polite region announces only state changes:
calibrating, running, paused, ended and disconnected.

**Without WebGL** the stream still runs, and the live card lists every
outcome as text.

**Where the code lives.** The simulation layer is a second ES module,
served from the binary like `roomsim.js`: the same CSP, immutable caching
and Trusted Types `'none'` (no `innerHTML`; styles only via
`element.style`). The import test of `roomsim.rs` and the self-containment
tests are extended to it deliberately. Phone-width layout and the static
page's rules still hold.

## 10. `saans roomsim-eval`

`saans roomsim-eval --scene SCENE.json [--seconds S] [--seed N] [--variant declared|with_walker|empty] [--records-out FILE] [--json]`

**Arguments.**
- `S` defaults to `ROOMSIM_RUN_S` (120). A value above 3600 is refused,
  naming `--seconds`.
- The scene is a file path, read the way `roomsim-export` reads it:
  `scene_id_problem` runs before parsing, so no error quotes an id. For
  REPLAY the file must be byte-identical to an embedded scene, since §6
  matches by sha256.
- Exit code 0 on success; exit code 2 on a refusal, which names the
  parameter.

**What it prints.** Every line is labelled SIM. With `--json` it prints
instead one JSON object `{"label":"SIM","scene":{…},"run":{…},"stats":{…},"radius_scale":{…}}`, whose
`stats` is exactly §5's block, `empty_run` included.
- The scene's name, sha256 and provenance; the variant, seed and seconds.
- Per estimator, the §5 figures, counts first.
- For WLM and RTI, the empty run's false alarms (§5.3), as published
  ticks per hour and events per hour, with their counts and seconds. With
  `--variant empty`, the run's own `unoccupied` figures replace the person
  errors.
- Per estimator, the suggested radius scale `radius_scale` (§8.2): "SIM,
  reported, not applied", or "too few" when `n < 20`.

**Equality with the session.** With `--seconds 120` and the same scene,
variant and seed, its §5 figures equal the `stats` of the session's final
(`ended: true`) frame. With any other `--seconds` they do not, because the
calibration window moves with `duration_s` (spec/locate.md §10.2).

## 11. Tests

### saans-locate
- `RoomSimStream` equals `run_roomsim` record for record:
  - at `duration_s = 20`, for every embedded scene, every variant and
    step sizes 0.025, 0.1 and 0.8 s;
  - WLM is `abstain no_disturbance` exactly when no link item is
    `disturbed`.
- Two streams from one setup are identical.
- `empty_room()` equals `run_roomsim` of the `empty` variant.
- `keep_going` returning false gives `Ok(None)`.
- After `run_roomsim` is re-expressed, every golden and parity test
  passes.
- `score` is checked against hand-built records:
  - odd and even medians;
  - nearest-rank p90;
  - `<=` for inside;
  - nulls at count 0;
  - unoccupied ticks and events;
  - the reference centroid.
- Records round-trip exactly. Every §6.2 refusal is tested.

### Gateway unit
- `Mode` serialises and parses in lower case.
- The `can_see_position` table holds: admin and safety with `None` →
  true; ops and occupancy with `None` → false; every role with `Some` →
  false.
- The frame builder:
  - deletes truth outside SIM;
  - never puts `blocked` outside `truth`;
  - sends no position without its radius.
- For every shipped scene and variant, no `Person.id` or `Phone.id`
  appears as a JSON string value in any frame, state or error. The check
  walks the parsed JSON, not the text, because the shipped phone's id
  "phone" is also a frame key.
- `new` holds exactly the records with `T_prev < t <= T`, with
  `T_prev = −∞` on a run's first frame.
- The clock: steps, the clamp at `q_end`, and `t = float(2q)*0.0125`.
- Command parsing.
- The session engine with an injected clock and no sockets:
  - for the bedroom at 120 s, the final `stats` equal
    `score(run_roomsim(…))` plus the `empty` variant's false alarms;
  - a REPLAY engine fed a file written by §8.3 gives the same truthless
    stats;
  - a running session puts nothing on `live_tx` or `outputs`.

### Gateway e2e (`--features roomsim`)
**Harness.**
- Every room request sends `Origin: http://{addr}`.
- Roles use `[auth] required = true` with one `[[token]]` per role
  (sha256, so no Argon2) and `Authorization: Bearer`.
- The harness gains a text send, and can observe a refusal.

**Cases.**
- **Origin and Host:** a missing, foreign or `null` Origin gets 403. On an
  auth-off gateway, `Host: evil.example:PORT` with a matching Origin gets
  403.
- **Roles:**
  - unauthenticated gets 401;
  - ops and occupancy get 403 and an audit `Denied`;
  - safety and admin get 101, then `state`;
  - `/api/me` reports `can.position` per role, and false when anonymous.
- **Frames:** a session sends `calibrating` and then frames.
  - `seq` is consecutive.
  - `t = float(2q)*0.0125`, with `q` advancing by exactly `speed*4` per
    frame.
  - After `pause`, two consecutive frames have equal `t`.
  - After `speed 2`, `q` advances by 8 per frame.
  - `restart` raises `run.index` and returns `t` to 0.
- **Refusals:**
  - a bad parameter gets an error naming it, then close 1008;
  - a fifth session gets an error, then close 1013.
- **REPLAY**, from a gateway started with `--room-replay` on a file
  written in the test: there is no `truth` or `blocked` key, and `stats`
  have no error keys.
- A live gateway's room frames are never `live`.
- The dashboard and `/api/me` carry the enum.
- **The only wall-clock check:** at least 5 frames within 2 s after the
  first frame.

### saans-cli
- `roomsim-eval` prints `score(run_roomsim(…))`. The test parses its
  output for the bedroom at 20 s.
- Its refusals are tested.
- `--records-out` writes a file the reader accepts.

### Browser
The script is `rust/crates/saans-gateway/tests/browser/roomsim_check.py`.
**Setup.**
- It runs against a release build started as RUN.md §4e says, with
  `--room-replay` pointing at a file written by
  `saans roomsim-eval --scene spec/scenes/hall-l-shaped.json --variant with_walker --seed 0 --records-out FILE`.
- It uses the hall, `with_walker`, seed 0, where WLM, RTI and RTT all
  publish early.
- It waits up to 30 s for `running`.

**Assertions:**
- circles, labels and abstention text are drawn;
- `t` is unchanged while paused;
- `t[seq+1] − t[seq] = speed × 0.1 s` at each speed, never measured in
  wall time;
- truth toggles in SIM, and is absent, with no error figures, in REPLAY;
- after a scene change there is one open session and a cleared layer;
- nothing overflows at 375 px;
- with WebGL disabled, the outcome text is shown;
- given an auth-required gateway (`--base-auth`), a signed-out view shows
  the sign-in text without opening a socket.

The console-error allowlist gains only the handshake errors from the
script's own refusal checks, matched by URL.

## 12. Limits

- **Single target.** Locate is single-target: one baseline and one RTI
  estimate per tick for the whole room, scored against the nearest
  person.
- **Still people.** In the shipped bedroom and office the people are
  still, so in `declared` most ticks are abstentions (spec/locate.md
  §13). `with_walker` is what shows movement.
- **Radii.** Radii are uncalibrated models, and the baseline's is
  room-sized (3.4-6.3 m).
- **REPLAY** plays recorded SIM runs only. Replaying a pcap through
  Locate, and LIVE, are later work.
- **Session cost** was measured on a 3.0 GHz Xeon, not on the target
  gateway (a Raspberry Pi 5 or a mini-PC). The cap of 4 is to be measured
  there before it is relied on.
- **Memory.** The calibration frames each setup keeps (§8.1) cost memory
  per session; the phase report measures it.
- **DNS rebinding** is closed for this endpoint only (§2.2).
