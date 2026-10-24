# ADR-008: Room geometry is declared, and what the demo may draw

**Status:** accepted, 2026-10-21

## Context
UI_IMPLEMENTATION_PLAN §9 lists "Floor plan, person position, 3D room"
as deliberately not built, for two stated reasons: "No coordinates exist,
and localisation is a published negative result." UI_GAP_ANALYSIS §1.4
marks all three **Blocked** and gives the floor-plan gap as three things:
"a new config schema, a survey of each room, and a way to show what is
*not* covered. Until then any drawing is invented." COMPLETION_REPORT §9
item 3 says the same in order, ending "which is the hard part, and the
reason not to start with the drawing."

Those are two different claims, and only one of them has moved.

**The first reason is half-retired.** `spec/scene.schema.json` and the
`saans-scene` crate give coordinates a schema: room shell, walls with
materials and thickness, furniture boxes and node positions in metres,
written by hand or imported from an Apple RoomPlan scan. That retires the
first of §1.4's three objections — a room drawn from such a file is
declared, not invented. The other two stand: no room has been surveyed
into a scene file, and nothing can yet draw what is not covered. Nor is
any scene reachable at runtime: `SpaceConfig` still has no coordinates,
`saans-gateway` does not depend on `saans-scene`, and the string "scene"
does not appear in the gateway source.

**The second reason has not moved at all.** NOTES-localize.md: RTI
reached 0.72 m median error at 13–18% coverage, the weighted-link-midpoint
baseline reached 0.709 m at 100% coverage and "beat or tied RTI in every
one of 15 (λ, α) configurations tested", and the standing guidance is
"Do not expect metre-level tracking from 8 ESP32 nodes. Plan for motion
detection and zone-level localization". All SYNTHETIC.

So the room became describable. The person did not.

## Decision

### What geometry is
1. **Declared, never inferred.** Room shape, walls, materials, furniture,
   node positions and zones come from a scene file and are labelled
   SCENE. An ESP32 at 20 MHz with one antenna cannot measure a room or
   recognise a sofa, so nothing may write a scene from CSI. Enforced
   structurally, not by good intentions: scene-writing APIs sit behind a
   `write` cargo feature that `saans-cli` enables and no crate carrying
   CSI may, and a test asserts that.
2. **Zones are declared too.** A zone is a named polygon (or a declared
   grid over the footprint) in the scene file, validated like any other
   geometry. There is no zone type in the runtime today; defining one is
   part of implementing this ADR, and a zone inferred from radio would
   violate decision 1 like any other invented geometry.
3. **Scene identifiers never leave the file.** `Person.id` and `Phone.id`
   are simulator bookkeeping. They are never published, drawn or logged
   (ADR-005 §4, "no identities anywhere").

### What may be drawn
4. **The room, its nodes and its links may be drawn**, because all of it
   is declared or directly measured. The view shows the scene's name, the
   hash of the file as loaded, and its `provenance` — `surveyed`,
   `roomplan_import` or `hand_written` — so a viewer can tell declared
   geometry from sensing. A scene imported from RoomPlan with unresolved
   importer warnings renders with those assumptions visible.
5. **A person is an estimate, never a point.** The only type the renderer
   accepts is `{ xy_m, radius_m }`, where `radius_m` is the 95% credible
   radius in metres; it has no function that takes a bare position. An
   estimate whose radius is absent, non-finite or implausibly small is
   not rendered. A build where the radius can be omitted, disabled by
   config or hidden by CSS violates this decision.
6. **Nothing is synthesised between estimates.** The marker steps between
   measured estimates and is never interpolated, smoothed or animated
   between them; a trail is a sequence of measured estimates, broken
   wherever the system abstained. This is the chart rule the project
   already wrote down ("Decimation strides, never averages",
   UI_IMPLEMENTATION_PLAN), extended from a series to a position.
7. **Abstention is drawn, not hidden.** When no estimate is available the
   view says so rather than holding the last one. (ADR-005 §2 for never
   filling in a plausible number; UI_IMPLEMENTATION_PLAN §9 for "if the
   series is empty the view says so".) Report coverage — the fraction of
   windows an estimate was produced, the sense NOTES-localize.md uses —
   is shown next to any accuracy figure. Spatial coverage, the floor area
   within range of two usable links, is a different thing that does not
   exist yet and is one of decision 13's gates.
8. **The claim is zone-level, not metre-level.** The zone readout is the
   primary spatial output and is a detector under ADR-005 §5: its
   evaluation reports per-zone confusion, the abstention rate and the
   error rate among non-abstentions, and no zone accuracy is published
   without its abstention rate beside it.
9. **The baseline is always present.** At runtime, the demo computes the
   weighted-link-midpoint estimate from the same FeatureFrames as any
   other estimator on screen, in the same tick, and draws both or
   neither. In reporting, any accuracy figure for a non-baseline
   estimator is published next to the baseline's on the same split. If
   the baseline is ported to Rust it gets golden vectors like every other
   number this project computes twice (ADR-002), with
   `wifisense/localize` as the reference. ADR-007 is unchanged by this:
   a learned estimator is not admissible here at all, and beating the
   baseline in simulation earns nothing.

### Who may see it
10. **Position passes both locks, like everything else.** Consent decides
    what may exist; the principal decides who may see it. A person
    coordinate is a new data class — it is derived from `motion_energy`
    and `gross_motion`, which are *not* consent-gated on the node, so the
    existing vitals gate does not cover it. It therefore needs its own
    per-space opt-in, and a `can_see_position(space_id)` scope held by
    `admin` and `safety` only. `ops` and `occupancy` get the room shell
    and the occupancy state, never a person marker. Redaction deletes the
    key rather than nulling it, as everywhere else.
11. **Position is not persisted.** Estimates live in the bounded
    in-memory window and are never written to the event log or to disk; a
    track longer than that window does not exist. A sequence of positions
    is a movement record, and changing this needs a DPDP retention answer
    first, not a code change.

### How it is built and gated
12. **The demo is its own page, served from the binary.** `dashboard.html`
    keeps its rules unchanged. three.js **must** be vendored into the
    gateway's assets — it is not in the tree today — which requires a
    static-asset route and widening the CSP from `script-src
    'unsafe-inline'` to include `'self'`, since the served policy is
    `default-src 'none'` and would refuse a vendored file as it stands.
    The two self-containment tests cover `DASHBOARD_HTML` and `GET /`
    only; they are parameterised over every served HTML asset, and the
    e2e test gains a sibling that fetches the demo route. The demo page
    does not ship before those tests cover it.
13. **LIVE is gated by a mechanism, not an intention.** The demo page is
    compiled behind a cargo feature that is off by default, and the
    gateway refuses at config load to serve it live without that feature
    — the project's named bug class is "configuration the validator
    accepts and the runtime cannot honour", and a gate in prose only is
    exactly that. Turning it on requires both remaining §1.4 objections
    to be met: a real room surveyed into a scene file and bound to a
    space, and a view that shows what is not covered.
14. **Enforcement is by review until CI runs the tests.** Every rule above
    rests on `cargo test --workspace`, and nothing runs it: the only
    workflow is CodeQL with `build-mode: none`, which compiles nothing. A
    CI job that runs the Rust and Python suites is a prerequisite of this
    ADR. Until it exists, these decisions are enforced by review only, and
    this document says so rather than implying otherwise.
15. **Mode becomes a closed enum.** `Shared::mode` is a `&'static str`
    today, only ever "live" or "replay", and UI_GAP_ANALYSIS §1.1 records
    the deliberate choice that "REPLAY is the truthful word" for
    simulator output. Drawing ground truth needs a state REPLAY cannot
    express, so `Shared::mode` becomes `Mode { Sim, Replay, Live }`,
    serialised in `hello`, `/api/me` and `/api/health` as before. Views
    render the badge from that enum by exhaustive match with no default
    branch, so a fourth mode cannot be added without every view being
    updated, and the browser never infers the mode. True positions are
    published and drawn only in `Sim`. This widens `saans-gateway`'s
    public API, agreed with the owner on 2026-10-21.
16. **The phone is simulated only.** A phone tracked by Wi-Fi RTT or RSSI
    is device-identified localisation, a different consent basis from
    anything in ADR-001 or ADR-005. It may be simulated, estimated and
    drawn in `Sim` and `Replay` as a simulated device. There is no LIVE
    phone: the gateway publishes no phone estimate outside those modes,
    and adding one needs its own ADR and its own consent basis.

## Amendments made on acceptance
- UI_IMPLEMENTATION_PLAN §9's floor-plan bullet now reads *superseded by
  ADR-008* and states what may be drawn.
- UI_GAP_ANALYSIS §1.4's three **Blocked** rows are re-stated: the floor
  plan is "Unblocked, not built", the 3D observatory is "Unblocked for
  SIM/REPLAY", and "Person position in a room" is "Blocked for LIVE" with
  "Do not render a position" replaced by decision 5.
- UI_GAP_ANALYSIS §1.1's "SYNTHETIC label — N/A" row records that this
  ADR adds the third mode.
- SAANS_PLAN §6 gains an 008 row; §3.1 gains `rust/crates/saans-scene`.

One edit deliberately waits. The topology caption in `dashboard.html`
says it is "not a floor plan: the gateway configuration has no room
coordinates", which is still true — no scene reaches the runtime yet. It
is reworded in the change that makes it false, not before.

## Consequences
- A drawn room is persuasive in a way a table is not. The mode badge and
  the uncertainty radius are the only things between a demo and a claim,
  which is why decisions 5 and 12 are written as type and test rules
  rather than as intentions.
- Every scene is someone's measurement, and a wrong scene is a wrong
  answer everywhere downstream with no sensor to contradict it. That is
  why `saans-scene` rejects rather than clamps.
- The RoomPlan importer guesses materials and wall thickness because the
  scan does not report them. Those guesses surface as warnings and stay
  wrong until a human edits them.
- An immersive view was listed as "a goal worth building *toward*"
  (UI_GAP_ANALYSIS §2). This is that step, taken without also claiming
  the localisation accuracy that would make it mean more than it does.

## Amendments after acceptance
- **2026-10-24, owner-approved (Phase 6).** Decision 4's provenance list
  gains `generated`: a scene drawn at random by `saans gen-dataset`
  (spec/dataset.md) to make simulated training data. It describes no real
  room, every number derived from it is SIM, and the demo serves only its
  embedded hand-written scenes, so a generated scene is never drawn as a
  room. The bullet above ("Every scene is someone's measurement") holds
  for every other provenance.
