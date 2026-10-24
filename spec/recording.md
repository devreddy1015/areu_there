# Recording protocol — the owner's rooms (Phase 11)

**Status:** spec for Phase 11, 2026-10-25, prepared under the owner's
blanket permission of that day. **Waiting for hardware and recordings:**
there are none yet (README "Hardware"). Everything here is ready to run on
the first capture. Figures from these recordings are **REAL**; a model
pre-trained on SIM and scored on them is **SIM->REAL**.

## 1. What a session is

A session is one capture with **today's firmware** (`edge/`, unchanged).
It has one transmitter and two or more receivers, so links are
`(-1, rx)`, read with `from_pcap(..., single_tx=True)`. A session takes
place in one room on one day and follows a script like OpenCSI's
(spec/phase9.md §1), so the two can be compared:

| Segment | Duration | `fine_label` |
|---|---|---|
| Empty, nobody inside, door as usual | 5 min | `empty_room` |
| One person seated and still | 3 min | `occupied_static` |
| The same person walking normally | 2 min | `occupied_moving` |
| Empty again | 2 min | `empty_room` |

Leave about 10 s unlabelled at each change; nobody is timed to the
second. Optional extra segments use the same labels, for example a second
still period lying down (`note: "lying"`), or two people
(`occupied_static` with `count: 2`).

**Day-to-day variation** (the point of Phase 11):
- at least **three sessions on three different days** in the same room,
  with the nodes never moved;
- then one session after a deliberate change: furniture moved, a door
  left open, or a node moved by 20–50 cm, noted in the label file.

## 2. Files per session

```
saans-data/recordings/<room>/<YYYY-MM-DD>_<n>/
  capture.pcap        # python3 -m wifisense record --port 5566 --out capture.pcap
  labels.jsonl        # one line per segment, §3
  scene.json          # optional: the room as declared (spec/scene.schema.json)
```

`saans-data/` and `*.pcap` are git-ignored (RUN.md §5). Recordings are
the owner's data and never leave this machine.

## 3. `labels.jsonl`

One JSON object per labelled segment, in time order, times as integer
epoch nanoseconds on the gateway's clock (the pcap's clock):

```json
{"start_ns": 1790000000000000000, "end_ns": 1790000300000000000, "fine_label": "empty_room", "count": 0, "note": ""}
```

- `fine_label`: `empty_room`, `occupied_static` or `occupied_moving`.
- `count`: optional; 0 for empty and 1 otherwise when absent.
- `note`: free text, never a name.

Segments may not overlap. A window is scored only if it lies wholly inside
one segment.

**Scene.** When the room is declared, node positions are SCENE and the
windows carry `geometry_known` true; otherwise false, as for OpenCSI. A
person's position is never recorded (ADR-008); zones and xy stay
unscored unless a later protocol adds marked positions with the owner's
agreement.

## 4. What runs on them (`ml/phase11/`)

1. **Read.**
   - `from_pcap(capture, ℓ, ends, single_tx=True, label="REAL", scene=…)`,
     with every check of spec/preprocess.md §2 and §8. The layout check is
     the first hardware confirmation of `esp32-fw56`, an open item.
   - `wifisense stats` must show at least 10 Hz per receiver, else the
     session is reported as too sparse and not scored.
2. **Windows.** The same rules as spec/phase9.md §3:
   - preprocessing `bin_hz = 5`, `shared40`;
   - training windows every 0.5 s, evaluation windows every 1 s;
   - the first 90 s are excluded from scoring, because the DSP baseline
     calibrates there.
3. **Methods.**
   - Phase 9's models: zero-shot SIM->REAL, the OpenCSI fine-tuned ones,
     and real-only.
   - The S0–S5 DSP presence detector (the product's own) and the WLM gate.
   - **Fine-tuning on the owner's data is leave-one-day-out:** the test
     day never trains anything.
4. **Day-to-day variation.** For every figure: its value per test day,
   the spread over days, and the change after the deliberate change of
   §1. Plus the drift of the empty-room statistics (per-link `mu` and
   `sigma`) between days, against the size of a person's effect.
5. **Reporting.** spec/phase9.md §6's figures and intervals. Every number
   is labelled REAL or SIM->REAL, and every comparison sits beside the DSP
   baseline on the same windows.

## 5. Limits

- One transmitter means one link per receiver, which is far fewer links
  than OpenCSI's 56.
- One subject; scripted segments.
- The gateway's clock is the time base, as for OpenCSI.
