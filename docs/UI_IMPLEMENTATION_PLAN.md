# UI implementation plan

The design the dashboard in `rust/crates/saans-gateway/assets/dashboard.html`
was built to, and the rules it has to keep. Read
[`UI_GAP_ANALYSIS.md`](UI_GAP_ANALYSIS.md) first: this says what to do
about what that found.

## 1. Constraints that decided the architecture

1. **One file, no build step.** The dashboard is `include_str!`-ed into
   the binary (`http.rs::DASHBOARD_HTML`). A bundler would add a build
   dependency to a project whose point is that it runs on a cheap box in
   a building. The file is ~1,300 lines; the moment it needs splitting,
   it needs a reason better than tidiness.
2. **Nothing is fetched from the internet.** Asserted twice, in
   `http::tests::dashboard_has_no_external_urls` and
   `e2e::dashboard_is_served_and_self_contained`. No CDN, no web fonts,
   no chart library — which is why the charts are hand-drawn SVG.
3. **No framework.** The state is one object and five render functions.
   A 15 KB page that a Raspberry Pi serves instantly is worth more here
   than component ergonomics.
4. **Every number maps to a published field.** If a view wants something
   the gateway does not publish, the answer is either a backend change
   with tests or the view does not get it.
5. **The gateway stays the source of truth.** The browser derives
   nothing a server could get wrong, and invents nothing at all.

## 2. Navigation

```
Overview   KPIs + space grid            (any role)
  └─ detail modal per space             (any role, filtered)
Signals    charts from /api/history     (any role, filtered)
Topology   sensor/link diagram          (any role, filtered)
Events     search + filters             (any role, filtered)
System     gateway and link health      (admin, ops)
```

A tab the principal cannot use is not rendered at all, rather than
rendered and refused — `visibleTabs()` is driven by `/api/me`'s `can`
object, so the menu never offers a dead end.

## 3. Data contract

Everything below already existed except `/api/history` and `/api/me`.

| View | Endpoint | Fields used |
|---|---|---|
| Shell | `GET /api/me`, `hello` frame | `auth_required`, `authenticated`, `name`, `role`, `can`, `mode`, `version` |
| Overview | `GET /api/live` (WS), fallback `GET /api/spaces` | `occupied`, `occupied_since_s`, `still_for_s`, `motion_index`, `motion_bouts_last_hour`, `breathing`, `links_usable`, `links_total`, `vitals`, `raw_mode`, `vitals_gated_links`, `alerts[]` |
| Detail | `GET /api/spaces/{id}` + `GET /api/history?space=` | above, plus `bpm_median_5min`, `bpm_iqr_5min`, `in_breathing_pause`, `events[]` |
| Signals | `GET /api/history` | `t_s`, `occupied`, `motion_index`, `presence_index`, `breathing_bpm`, `breathing_conf`, and per link `rate_hz`, `loss`, `best_bnr`, `rssi_dbm` |
| Topology | `GET /api/links` | `link.rx`, `link.tx`, `up`, `loss`, `calib_age_h`, `silent_s`, `rate_hz` |
| Events | `GET /api/events`, WS `event` frames | `time`, `unix_ms`, `space_id`, `type`, `severity`, `confidence`, `evidence` |
| System | `GET /api/health`, `GET /api/links` | the whole health object, plus the new `auth`, `audit` and `series` sub-objects |

### `/api/history` — the one new read endpoint

`GET /api/history?space=&since=&max_points=`

Columnar, because an hour of six series as an array of objects is about
ten times larger and a Pi has to serialise it:

```json
{ "cadence_s": 2.0, "window_s": 3600.0, "capacity": 1800,
  "spaces": [ { "space_id": "lab", "name": "Lab", "points": 1800,
                "t_s": [...], "unix_ms": [...], "occupied": [true, null, false, ...],
                "motion_index": [...], "presence_index": [...],
                "links_usable": [...], "links_total": [...],
                "breathing_bpm": [...], "breathing_conf": [...], "breath_rms": [...] } ],
  "links":  [ { "space_id": "lab", "rx": 1, "tx": 255, "points": 1800,
                "t_s": [...], "up": [...], "rate_hz": [...], "loss": [...],
                "best_bnr": [...], "rssi_dbm": [...] } ] }
```

Rules it keeps, each with a test in `history.rs`:

- **Bounded at startup.** `window_s / cadence_s` points per series,
  clamped by `MAX_WINDOW_S` and `MIN_CADENCE_S`. Memory does not grow
  with uptime.
- **Never persisted.** The window is the retention policy.
- **`occupied` is three-valued.** `null` means no usable link, which is
  not `false`.
- **`breathing_*` keys are absent, not null**, when consent or role
  forbids them — that is what lets the UI say "not shown to your role"
  instead of "not detected".
- **Decimation strides, never averages.** Every point returned was
  measured, and the newest is always last.
- **Time going backwards clears the series** (a replay restarting).

`max_points` is clamped to the ring capacity, so a client cannot ask for
an unbounded response.

## 4. Visual system

Dark is the default (`:root`), light is the media-query override — the
inverse of the old page. Tokens only; no literal colours in components.

| Token | Means |
|---|---|
| `--accent` teal | signal and measurement |
| `--ok` green | healthy, occupied |
| `--warn` amber | degraded, uncertain, **cannot see** |
| `--bad` red | a sensor is broken, or an alert needs someone |
| `--none` grey | inactive, unavailable, empty |

Amber for "cannot see" is deliberate: an unobservable room is a
degraded state, not a neutral one.

**Status never relies on colour alone.** Every coloured element carries
a word: the state is spelled out on the card, link rows say "up"/"down",
and chart series carry a text summary of their latest values.

## 5. Charts

Hand-drawn SVG, built with `createElementNS` and `textContent` — no
`innerHTML` anywhere, so a server-provided string cannot become markup.

- A line **breaks at a gap** rather than drawing through it. A straight
  line across an abstention is a claim nobody measured.
- Axes do **not** pad below an explicit zero.
- Occupancy is a three-state band, not a line, because it is not a
  quantity.
- Each chart names its source, unit and window in its subtitle, and
  exposes `role="img"` with an `aria-label` summarising the latest
  values.

## 6. Re-render discipline

The shell re-renders on every state message and on a 2 s timer, so
anything rebuilt there must be cheap *and* must not destroy user state:

- The **tab strip** is built once and only its `aria-selected` /
  `tabindex` change. Rebuilding it was throwing keyboard focus out of
  the nav twice a second and dropping clicks that landed mid-rebuild.
- The **topology SVG** is rebuilt only when a signature of link state
  changes, because its nodes are focusable.
- The **space grid** is rebuilt wholesale; it contains no focusable
  element that survives a tick, and a diff would cost more than it saves.

## 7. Accessibility

Tablist with `aria-controls` and Arrow/Home/End; labelled `tabpanel`s;
`:focus-visible` rings; `<dialog>` + `showModal()` for the detail panel
so focus trapping and Escape come from the platform; `aria-live="polite"`
for connection and mode changes only; `prefers-reduced-motion` honoured;
no horizontal overflow at 375 px on any view.

## 8. Order of work, and what it cost

1. Auth, authorisation, audit — nothing else matters if the API is open.
2. `/api/history` — charts need a real source before they are worth drawing.
3. The dashboard itself.
4. Docs.

Phases 1 and 2 are backend and carry unit tests; phase 3 was verified by
driving it. See `COMPLETION_REPORT.md` for exactly what was run.

## 9. Deliberately not built

- **Alert acknowledge / resolve.** `AlertService` is a stub; a button
  that does nothing is worse than no button.
- **Raw CSI heatmap.** FeatureFrames are not subcarrier data and must
  not be drawn as if they were.
- **Floor plan, person position, 3D room.** No coordinates exist, and
  localisation is a published negative result.
- **Client-side synthesis of any kind.** If the series is empty the view
  says so.
