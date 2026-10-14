// Saans /demo. Loaded after site.js, and only this page has the elements
// it looks for.
//
// Two columns, and the gap between them is the point.
//
// Left is the simulator's *ground truth*: a room we configured, nodes at
// known positions, and a person whose position we set by dragging. That
// position is an input. Under it is the coverage map the gateway computes
// from the simulator's own Fresnel model — where a breathing chest would
// be detectable for this fixed geometry.
//
// Right is what the DSP recovered from the simulated channel, through the
// ordinary ingest path, with nothing positional in it: this hardware is
// 1T1R and has no angular resolution at all.
//
// The browser models nothing. Every number drawn here is a field of
// /api/sim, /api/sim/coverage, or the published state. The only arithmetic
// below is turning metres into pixels, and comparing two published rates.
//
// When a fetch fails, the panel says which one failed and draws nothing.
// Nothing is substituted for it.
(() => {
  "use strict";

  const $ = (s, el = document) => el.querySelector(s);
  const PLAN = $("#demo-plan");
  if (!PLAN) return;

  // Every SVG element below is created in the namespace of the plan that
  // is already in the page. The namespace is a URI, and this file is
  // scanned for remote references by site.rs, so it is read off the
  // document rather than spelled out. Nothing is fetched either way.
  const NS = PLAN.namespaceURI;

  /**
   * Cell size the coverage map is fetched at, in metres.
   *
   * This is not a cosmetic choice and it must not be raised for speed.
   * The field being drawn is Fresnel structure: it goes from a maximum to
   * a null in a quarter wavelength, about 3 cm at 2.4 GHz. Sampled
   * coarser than that, the grid is an alias of a field it cannot
   * represent, and the picture is not merely blurry — it is wrong in both
   * directions. The page carries the measured table; 0.25 m calls the
   * demo null "below median", 0.10 m calls it brighter than median, and
   * only 0.03 m calls it blind, which is what the pipeline does.
   */
  const COV_RES_M = 0.03;
  /**
   * How far the recovered rate may sit from the simulator's input before
   * the verdict stops calling it a success, in BPM. The pipeline's own
   * test (simulate.rs, the_gap_the_explorer_exists_to_show) asserts
   * agreement to within 1 BPM, so this is that test's tolerance and not a
   * looser one chosen to make the demo look better.
   */
  const BPM_TOL = 1.0;
  /** At most one POST per interval while dragging. */
  const PERSON_POST_MS = 120;
  const POLL_MS = 1000;
  const EX_PAD = 28;

  const COV_BANDS = [
    { lo: 0, range: "below 0.25", short: "effectively blind" },
    { lo: 0.25, range: "0.25 to 0.5", short: "very weak" },
    { lo: 0.5, range: "0.5 to 1.0", short: "below this room’s median" },
    { lo: 1.0, range: "1.0 to 2.0", short: "around this room’s median" },
    { lo: 2.0, range: "2.0 to 4.0", short: "strong" },
    { lo: 4.0, range: "4.0 and above", short: "strongest" },
  ];

  const SIM = {
    cfg: null, cov: null, covFailed: false, person: null, present: true,
    err: "", errKind: "", space: null, spaceErr: "", postErr: "",
    built: "", els: null, drag: null, said: "",
  };

  // ---------------------------------------------------------- formatting
  const fx = (n) => Math.round(n * 100) / 100;
  const num = (x, d) => (x == null || !isFinite(x) ? "—" : Number(x).toFixed(d == null ? 1 : d));
  const linkName = (l) => (l ? l.rx + "←" + l.tx : "");
  const linkKey = (l) => l.rx + "<" + l.tx;
  const stateOf = (s) => (s.occupied === true ? "occ" : s.occupied === false ? "empty" : "blind");
  const stateWord = (k) => (k === "occ" ? "Occupied" : k === "empty" ? "Empty" : "Unknown");
  function dur(s) {
    if (s == null || !isFinite(s)) return "—";
    s = Math.max(0, s);
    if (s < 90) return Math.round(s) + " s";
    if (s < 5400) return Math.round(s / 60) + " min";
    return (s / 3600).toFixed(1) + " h";
  }

  function el(tag, attrs, text) {
    const e = document.createElement(tag);
    for (const k in attrs || {}) if (attrs[k] != null) e.setAttribute(k, attrs[k]);
    if (text != null) e.textContent = text;
    return e;
  }
  function svgEl(tag, attrs) {
    const e = document.createElementNS(NS, tag);
    for (const k in attrs || {}) if (attrs[k] != null) e.setAttribute(k, String(attrs[k]));
    return e;
  }
  function add(parent, ...kids) {
    for (const k of kids) if (k != null) parent.append(k);
    return parent;
  }
  const announce = (msg) => { $("#ex-live").textContent = msg; };

  // ----------------------------------------------------------- transport
  class HttpError extends Error {
    constructor(status, path) { super(path + ": " + status); this.status = status; this.path = path; }
  }
  async function getJSON(path) {
    const r = await fetch(path, { cache: "no-store", credentials: "same-origin" });
    if (!r.ok) throw new HttpError(r.status, path);
    return r.json();
  }
  async function postJSON(path, body) {
    const r = await fetch(path, {
      method: "POST", cache: "no-store", credentials: "same-origin",
      headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
    });
    const data = await r.json().catch(() => ({}));
    if (!r.ok) { const e = new HttpError(r.status, path); e.data = data; throw e; }
    return data;
  }

  // ------------------------------------------------------ coverage lookup
  function covValues() {
    const c = SIM.cov;
    if (!c) return null;
    return Array.isArray(c.best) ? c.best : null;
  }
  function covBandOf(v) {
    if (v == null || !isFinite(v)) return -1;
    let i = 0;
    for (let b = 0; b < COV_BANDS.length; b++) if (v >= COV_BANDS[b].lo) i = b;
    return i;
  }
  /** Nearest cell of the published grid. A lookup, not an interpolation. */
  function covAt(x_m, y_m) {
    const c = SIM.cov, vals = covValues();
    if (!c || !vals || !Array.isArray(c.origin_m)) return null;
    const ix = Math.round((x_m - c.origin_m[0]) / c.res_m);
    const iy = Math.round((y_m - c.origin_m[1]) / c.res_m);
    if (!(ix >= 0 && iy >= 0 && ix < c.w && iy < c.h)) return null;
    const v = vals[iy * c.w + ix];
    return v == null || !isFinite(v) ? null : v;
  }
  const stepM = () => (SIM.cov && SIM.cov.res_m > 0 ? SIM.cov.res_m : 0.25);

  // -------------------------------------------------------------- polling
  let personPending = null, personTimer = null, personBusy = false;

  async function refreshSim() {
    try {
      const cfg = await getJSON("/api/sim");
      SIM.cfg = cfg;
      SIM.err = ""; SIM.errKind = "";
      const t = cfg.truth || {};
      SIM.present = t.present !== false;
      // Do not pull the marker out from under a drag or an unacknowledged
      // move: the pending POST is the newer fact.
      if (!SIM.drag && !personPending && !personBusy && isFinite(t.x_m) && isFinite(t.y_m)) {
        SIM.person = { x_m: t.x_m, y_m: t.y_m };
      }
      if (!SIM.cov && !SIM.covFailed) {
        // Computed for the configured geometry, which does not change, so
        // it is fetched once.
        SIM.cov = await getJSON("/api/sim/coverage?res_m=" + COV_RES_M)
          .catch(() => { SIM.covFailed = true; return null; });
      }
    } catch (e) {
      SIM.cfg = null;
      SIM.errKind = e instanceof HttpError ? String(e.status) : "fetch";
      SIM.err = e.status === 404
        ? "This gateway is not running in simulate mode, so there is no simulated room to show."
        : e.status === 401
          ? "This gateway requires sign-in before it will describe its simulated room."
          : "The gateway did not answer /api/sim" + (e.status ? " (" + e.status + ")" : "") + ".";
    }
  }

  async function refreshSpaces() {
    const id = SIM.cfg && SIM.cfg.space_id;
    if (!id) { SIM.space = null; return; }
    try {
      const list = await getJSON("/api/spaces");
      SIM.spaceErr = "";
      SIM.space = (Array.isArray(list) ? list : []).find((x) => x.space_id === id) || null;
      if (!SIM.space) {
        SIM.spaceErr = "The gateway has not published a state for “" + id
          + "” yet, or your account cannot see that space. Nothing is substituted for it.";
      }
    } catch (e) {
      SIM.space = null;
      SIM.spaceErr = e.status === 401
        ? "This gateway requires sign-in before it will publish a space. Nothing is substituted for it."
        : "The gateway did not answer /api/spaces" + (e.status ? " (" + e.status + ")" : "")
          + ". Nothing is substituted for it.";
    }
  }

  async function refresh() {
    if (document.hidden) return;
    await refreshSim();
    await refreshSpaces();
    render();
  }

  let timer = null;
  function startPolling() {
    if (timer || document.hidden) return;
    refresh();
    timer = setInterval(refresh, POLL_MS);
  }
  function stopPolling() { if (timer) { clearInterval(timer); timer = null; } }
  document.addEventListener("visibilitychange", () => (document.hidden ? stopPolling() : startPolling()));

  // --------------------------------------------------------- moving them
  function queuePerson(x_m, y_m) {
    personPending = { x_m, y_m };
    if (personTimer || personBusy) return;
    sendPerson();
  }
  async function sendPerson() {
    if (!personPending) return;
    const body = personPending;
    personPending = null;
    personBusy = true;
    personTimer = setTimeout(() => { personTimer = null; if (personPending) sendPerson(); }, PERSON_POST_MS);
    try {
      const r = await postJSON("/api/sim/person", body);
      SIM.postErr = "";
      // The gateway clamps into the room; show where the person actually
      // ended up, unless a newer move is already on its way.
      if (!SIM.drag && !personPending && isFinite(r.x_m) && isFinite(r.y_m)) {
        SIM.person = { x_m: r.x_m, y_m: r.y_m };
      }
    } catch (e) {
      SIM.postErr = "The gateway refused that move (" + (e.status || "no answer") + "). "
        + "The marker shows where you dragged it, not where the simulator put the person.";
    } finally {
      personBusy = false;
      updateRoom();
      if (!personTimer && personPending) sendPerson();
    }
  }

  async function togglePresent() {
    const want = !SIM.present;
    try {
      const r = await postJSON("/api/sim/present", { present: want });
      SIM.present = !!r.present;
      SIM.postErr = "";
      announce(SIM.present ? "Person put back into the simulated room."
        : "Person removed. The simulated room is empty.");
    } catch (e) {
      SIM.postErr = "The gateway refused that change (" + (e.status || "no answer") + ").";
    }
    updateRoom();
    renderRight();
  }

  /** Pointer, keyboard and the preset buttons all end in moveTo. */
  function moveTo(x_m, y_m, src) {
    const e = SIM.els;
    if (!e || !SIM.present) return;
    const x = Math.min(Math.max(x_m, 0), e.w), y = Math.min(Math.max(y_m, 0), e.h);
    SIM.person = { x_m: x, y_m: y };
    updateRoom();
    queuePerson(x, y);
    if (src !== "pointer") announcePerson();
  }
  function announcePerson() {
    const p = SIM.person;
    if (!p) return;
    const cv = covAt(p.x_m, p.y_m);
    announce("Person at " + num(p.x_m, 2) + " metres across, " + num(p.y_m, 2) + " metres up"
      + (cv == null ? "." : ". Coverage here " + num(cv, 2) + ", " + COV_BANDS[covBandOf(cv)].short + "."));
  }

  // ------------------------------------------------------------- the room
  function render() {
    const note = $("#plan-note");
    if (SIM.err || !SIM.cfg) {
      SIM.built = ""; SIM.els = null;
      PLAN.replaceChildren();
      PLAN.hidden = true;
      note.replaceChildren(el("b", null, SIM.err ? "No simulated room is being drawn." : "Loading the simulated room…"));
      if (SIM.err) {
        add(note, document.createTextNode(" " + SIM.err));
        if (SIM.errKind === "404") {
          add(note, document.createTextNode(" Start one with "),
            el("code", null, "saans-gateway --config saans.toml --simulate --speed 10"),
            document.createTextNode(" — the same command is under “run this yourself” below."));
        } else if (SIM.errKind === "401") {
          const a = el("a", { class: "link", href: "/app" }, "sign in at the dashboard");
          add(note, document.createTextNode(" You can "), a, document.createTextNode("."));
        } else {
          add(note, document.createTextNode(" Nothing is substituted for it."));
        }
      }
      $("#t-pos").textContent = "—";
      $("#t-cov").textContent = "—";
      $("#t-breath").textContent = "—";
      renderRight();
      return;
    }
    PLAN.hidden = false;
    note.textContent = "";
    buildRoom();
    updateRoom();
    renderRight();
  }

  /**
   * Built once per geometry, then only moved. The person marker is
   * focusable and the user may be part-way through dragging it:
   * rebuilding this panel on a 1 s timer would throw that away.
   */
  function buildRoom() {
    const cfg = SIM.cfg, room = cfg.room || {};
    const w = Number(room.w_m) > 0 ? Number(room.w_m) : 1;
    const h = Number(room.h_m) > 0 ? Number(room.h_m) : 1;
    const sig = JSON.stringify([w, h, room.door_m, cfg.nodes, cfg.links, cfg.space_id, cfg.scenario,
      SIM.cov ? [SIM.cov.res_m, SIM.cov.w, SIM.cov.h] : null]);
    if (sig === SIM.built && SIM.els) return;
    SIM.built = sig;

    const SC = Math.min(600 / w, 420 / h);
    const VW = w * SC + EX_PAD * 2, VH = h * SC + EX_PAD * 2 + 24;
    const X = (m) => EX_PAD + m * SC;
    const Y = (m) => EX_PAD + (h - m) * SC;

    PLAN.replaceChildren();
    PLAN.setAttribute("viewBox", "0 0 " + fx(VW) + " " + fx(VH));
    PLAN.setAttribute("aria-label", "Plan of the simulated room, " + num(w, 1) + " by " + num(h, 1)
      + " metres, with " + ((cfg.nodes || []).length)
      + " node(s) and a draggable ground-truth person marker.");

    // A clip so a coverage grid whose cell count rounds up does not paint
    // past the wall.
    const clipId = "demo-room-clip";
    const defs = svgEl("defs");
    const clip = svgEl("clipPath", { id: clipId });
    add(clip, svgEl("rect", { x: fx(X(0)), y: fx(Y(h)), width: fx(w * SC), height: fx(h * SC) }));
    add(defs, clip);
    add(PLAN, defs);

    add(PLAN, svgEl("rect", { class: "wall", x: fx(X(0)), y: fx(Y(h)), width: fx(w * SC), height: fx(h * SC), rx: 3 }));

    const heat = svgEl("g", { "clip-path": "url(#" + clipId + ")", "aria-hidden": "true" });
    add(PLAN, heat);

    const grid = svgEl("g", { "aria-hidden": "true" });
    for (let m = 1; m < w; m++) add(grid, svgEl("line", { class: "grid-m", x1: fx(X(m)), x2: fx(X(m)), y1: fx(Y(0)), y2: fx(Y(h)) }));
    for (let m = 1; m < h; m++) add(grid, svgEl("line", { class: "grid-m", x1: fx(X(0)), x2: fx(X(w)), y1: fx(Y(m)), y2: fx(Y(m)) }));
    add(PLAN, grid);

    // Door, on whichever wall it is nearest.
    const d = room.door_m;
    if (Array.isArray(d) && isFinite(d[0]) && isFinite(d[1])) {
      const dx = Math.min(Math.max(d[0], 0), w), dy = Math.min(Math.max(d[1], 0), h);
      const near = Math.min(dy, h - dy, dx, w - dx), half = 0.45;
      const lo = (v, max) => Math.max(0, Math.min(max, v) - half), hi = (v, max) => Math.min(max, Math.max(0, v) + half);
      let a, b, lx, ly;
      if (near === dy) { a = [X(lo(dx, w)), Y(0)]; b = [X(hi(dx, w)), Y(0)]; lx = X(dx); ly = Y(0) - 7; }
      else if (near === h - dy) { a = [X(lo(dx, w)), Y(h)]; b = [X(hi(dx, w)), Y(h)]; lx = X(dx); ly = Y(h) + 13; }
      else if (near === dx) { a = [X(0), Y(lo(dy, h))]; b = [X(0), Y(hi(dy, h))]; lx = X(0) + 17; ly = Y(dy); }
      else { a = [X(w), Y(lo(dy, h))]; b = [X(w), Y(hi(dy, h))]; lx = X(w) - 17; ly = Y(dy); }
      add(PLAN, svgEl("line", { class: "door", x1: fx(a[0]), y1: fx(a[1]), x2: fx(b[0]), y2: fx(b[1]) }));
      const dt = svgEl("text", { class: "ax", x: fx(lx), y: fx(ly), "text-anchor": "middle" });
      dt.textContent = "door";
      add(PLAN, dt);
    }

    const byId = new Map((cfg.nodes || []).map((n) => [n.id, n]));
    const unresolved = [];
    for (const l of cfg.links || []) {
      const a = byId.get(l.tx), b = byId.get(l.rx);
      // A link whose endpoints are not nodes we were given has no position
      // we know. Say that, rather than guess a line across the room.
      if (!a || !b) { unresolved.push(linkName(l)); continue; }
      const ln = svgEl("line", { class: "lnk", x1: fx(X(a.x_m)), y1: fx(Y(a.y_m)), x2: fx(X(b.x_m)), y2: fx(Y(b.y_m)) });
      const lt = svgEl("title");
      lt.textContent = "Link " + linkName(l) + ": node " + l.tx + " transmits, node " + l.rx + " receives.";
      add(ln, lt);
      add(PLAN, ln);
    }
    for (const n of cfg.nodes || []) {
      if (!isFinite(n.x_m) || !isFinite(n.y_m)) continue;
      const grp = svgEl("g", { class: "nd" });
      add(grp, svgEl("circle", { cx: fx(X(n.x_m)), cy: fx(Y(n.y_m)), r: 12 }));
      const t = svgEl("text", { x: fx(X(n.x_m)), y: fx(Y(n.y_m)) });
      t.textContent = n.id === 255 ? "AP" : String(n.id);
      add(grp, t);
      const nt = svgEl("title");
      nt.textContent = "Node " + n.id + " at " + num(n.x_m, 2) + " m, " + num(n.y_m, 2)
        + " m (a position we configured).";
      add(grp, nt);
      add(PLAN, grp);
    }

    const person = svgEl("g", {
      class: "person", tabindex: "0", role: "application",
      "aria-roledescription": "draggable ground-truth position handle",
    });
    add(person, svgEl("circle", { class: "halo", cx: 0, cy: 0, r: 17 }));
    add(person, svgEl("circle", { class: "dot", cx: 0, cy: 0, r: 8 }));
    const tag = svgEl("text", { class: "tag", x: 0, y: 32 });
    tag.textContent = "GROUND TRUTH (we set this) — drag me";
    add(person, tag);
    add(PLAN, person);

    // Axes and a scale bar, so the geometry is readable as metres.
    const ax = svgEl("g", { "aria-hidden": "true" });
    for (let m = 0; m <= Math.floor(w); m++) {
      const t = svgEl("text", { class: "ax", x: fx(X(m)), y: fx(Y(0) + 14), "text-anchor": "middle" });
      t.textContent = String(m); add(ax, t);
    }
    for (let m = 0; m <= Math.floor(h); m++) {
      const t = svgEl("text", { class: "ax", x: fx(X(0) - 7), y: fx(Y(m) + 3), "text-anchor": "end" });
      t.textContent = String(m); add(ax, t);
    }
    const sy = Y(0) + 30;
    add(ax, svgEl("line", { class: "barm", x1: fx(X(0)), y1: fx(sy), x2: fx(X(1)), y2: fx(sy) }));
    const sb = svgEl("text", { class: "ax", x: fx(X(1) + 7), y: fx(sy + 3) });
    sb.textContent = "1 metre"; add(ax, sb);
    add(PLAN, ax);

    SIM.els = {
      heat, person, tag, w, h, X, Y, SC,
      pos: $("#t-pos"), cov: $("#t-cov"), truth: $("#t-breath"),
      present: $("#ex-present"), key: $("#covkey"), legendNote: $("#covnote"), err: $("#ex-err"),
      unresolved,
    };
    attachRoomInput();
    paintHeat();
  }

  /**
   * Painted once: the map is a prediction for a fixed geometry and does
   * not move with the person.
   *
   * Cells of one row that fall in the same band are merged into a single
   * rect. The painted result is identical cell for cell — same colour,
   * same edges — and nothing is averaged, smoothed or interpolated; it is
   * only fewer nodes for the same picture.
   */
  function paintHeat() {
    const e = SIM.els;
    if (!e) return;
    const c = SIM.cov, vals = covValues();
    e.heat.replaceChildren();
    e.key.replaceChildren();
    if (!c || !vals) {
      e.legendNote.textContent = SIM.covFailed
        ? "The gateway did not return a coverage map, so none is drawn. Nothing is substituted for it."
        : "Loading the coverage map…";
      return;
    }
    const r = c.res_m, half = r / 2, side = fx(r * e.SC) + 0.5;
    let lo = Infinity, hi = -Infinity, drawn = 0;
    const counts = COV_BANDS.map(() => 0);
    const frag = document.createDocumentFragment();
    for (let iy = 0; iy < c.h; iy++) {
      let band = -1, start = 0;
      const flush = (endIx) => {
        if (band < 0) return;
        const x0 = c.origin_m[0] + start * r - half;
        const y = c.origin_m[1] + iy * r;
        frag.append(svgEl("rect", {
          class: "cell", x: fx(e.X(x0)), y: fx(e.Y(y + half)),
          width: fx((endIx - start) * r * e.SC) + 0.5, height: side,
          fill: "var(--cov" + band + ")",
        }));
        band = -1;
      };
      for (let ix = 0; ix < c.w; ix++) {
        const v = vals[iy * c.w + ix];
        const b = covBandOf(v);
        if (b !== band) { flush(ix); start = ix; band = b; }
        if (b < 0) continue;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
        counts[b]++; drawn++;
      }
      flush(c.w);
    }
    e.heat.append(frag);

    COV_BANDS.forEach((b, i) => {
      const item = el("span");
      const sw = el("i");
      sw.style.background = "var(--cov" + i + ")";
      add(item, sw, document.createTextNode(b.range + " — " + b.short
        + (counts[i] ? " (" + Math.round((100 * counts[i]) / drawn) + "% of the room)" : "")));
      e.key.append(item);
    });

    const units = c.units || "relative breathing sensitivity";
    const nLinks = ((c.links || []).length) || 1;
    e.legendNote.textContent = "SYNTHETIC. This is not a measurement of a room. It is the simulator’s own "
      + "Fresnel model, asked at every cell: if a simulated torso stood here and breathed, how strongly "
      + "would this link’s best subcarrier move? Scale: " + units + ". Values run " + num(lo, 2) + " to "
      + num(hi, 2) + " at " + num(r, 2) + " m per cell, "
      + (nLinks > 1 ? "taking the best link at each point. " : "for the one link this scenario runs. ")
      + "Nothing here was sensed; the map is a prediction for this fixed geometry and does not change as "
      + "the person moves. It is drawn at " + num(r, 2) + " m because the pattern goes from peak to null "
      + "in about 3 cm — a coarser grid aliases it and would disagree with the panel on the right.";
  }

  function updateRoom() {
    const e = SIM.els;
    if (!e) return;
    const p = SIM.person || { x_m: 0, y_m: 0 };
    e.person.setAttribute("transform", "translate(" + fx(e.X(p.x_m)) + " " + fx(e.Y(p.y_m)) + ")");
    e.person.style.display = SIM.present ? "" : "none";
    // Against the bottom wall the label would land on the axis numbers.
    e.tag.setAttribute("y", p.y_m < 0.8 ? -24 : 32);
    const where = num(p.x_m, 2) + " m across, " + num(p.y_m, 2) + " m up";
    e.person.setAttribute("aria-label", "Ground-truth person position, an input: " + where
      + ". Arrow keys move by " + num(stepM(), 2) + " metres, Shift and arrow by 1 metre.");
    e.pos.textContent = SIM.present ? where : "removed — the simulated room is empty";

    const cv = SIM.present ? covAt(p.x_m, p.y_m) : null;
    e.cov.textContent = !SIM.present ? "—"
      : cv == null ? "no coverage value published for this cell"
        : num(cv, 2) + " — " + COV_BANDS[covBandOf(cv)].short;

    const t = (SIM.cfg && SIM.cfg.truth) || {};
    e.truth.textContent = !SIM.present ? "nobody in the room"
      : t.breathing ? num(t.bpm, 1) + " BPM" + (t.in_pause ? ", currently in a simulated pause" : "")
        + (t.still ? ", still" : ", moving")
        : "none — we did not give this simulated body breathing motion";

    e.present.textContent = SIM.present ? "empty the room" : "put the person back";
    let err = SIM.postErr;
    if (e.unresolved.length) {
      err += (err ? " " : "") + "Not drawn: link(s) " + e.unresolved.join(", ")
        + " — their endpoints do not match any node /api/sim listed, so this view does not know where "
        + "they run.";
    }
    e.err.textContent = err;
  }

  function attachRoomInput() {
    const e = SIM.els, g = PLAN;
    const roomPoint = (ev) => {
      const m = g.getScreenCTM && g.getScreenCTM();
      if (!m) return null;
      let q;
      if (g.createSVGPoint) {
        const pt = g.createSVGPoint(); pt.x = ev.clientX; pt.y = ev.clientY;
        q = pt.matrixTransform(m.inverse());
      } else {
        q = new DOMPoint(ev.clientX, ev.clientY).matrixTransform(m.inverse());
      }
      return { x_m: (q.x - EX_PAD) / e.SC, y_m: e.h - (q.y - EX_PAD) / e.SC };
    };
    g.addEventListener("pointerdown", (ev) => {
      if (!SIM.present) return;
      const p = roomPoint(ev);
      const onHandle = e.person.contains(ev.target);
      // A press in the margin is not a command to teleport the person.
      if (!p || (!onHandle && !(p.x_m >= -0.15 && p.y_m >= -0.15 && p.x_m <= e.w + 0.15 && p.y_m <= e.h + 0.15))) return;
      ev.preventDefault();
      SIM.drag = ev.pointerId;
      try { g.setPointerCapture(ev.pointerId); } catch (err) { /* capture is an optimisation */ }
      if (!onHandle) moveTo(p.x_m, p.y_m, "pointer");
      e.person.focus({ preventScroll: true });
    });
    const end = (ev) => {
      if (SIM.drag !== ev.pointerId) return;
      SIM.drag = null;
      try { g.releasePointerCapture(ev.pointerId); } catch (err) { /* already released */ }
      // The position the pointer stopped on is always sent, whatever the
      // throttle was doing when the button came up.
      if (SIM.person) queuePerson(SIM.person.x_m, SIM.person.y_m);
      announcePerson();
    };
    g.addEventListener("pointermove", (ev) => {
      if (SIM.drag !== ev.pointerId) return;
      // A pointerup can be lost — to a tab switch, a dialog, a devtools
      // pause. Without this the marker would follow the mouse forever.
      if (ev.buttons === 0) { end(ev); return; }
      const p = roomPoint(ev);
      if (p) moveTo(p.x_m, p.y_m, "pointer");
    });
    g.addEventListener("pointerup", end);
    g.addEventListener("pointercancel", end);
    e.person.addEventListener("keydown", (ev) => {
      const d = { ArrowLeft: [-1, 0], ArrowRight: [1, 0], ArrowUp: [0, 1], ArrowDown: [0, -1] }[ev.key];
      if (!d || !SIM.person) return;
      ev.preventDefault();
      const step = ev.shiftKey ? 1 : stepM();
      moveTo(SIM.person.x_m + d[0] * step, SIM.person.y_m + d[1] * step, "key");
    });
  }

  // ------------------------------------------- what the pipeline recovers
  function renderRight() {
    const host = $("#sys-dl"), box = $("#verdict"), badges = $("#sys-badges"),
      alerts = $("#sys-alerts"), note = $("#sys-note");
    host.replaceChildren(); box.replaceChildren(); badges.replaceChildren(); alerts.replaceChildren();

    if (!SIM.cfg) {
      note.textContent = SIM.err
        ? "No room is being simulated, so the gateway publishes no state for one."
        : "Waiting for the simulator.";
      return;
    }
    const s = SIM.space;
    if (!s) {
      note.textContent = SIM.spaceErr || "Waiting for the gateway to publish a state.";
      return;
    }
    note.textContent = "";

    box.append(verdictBox(s));

    badges.append(s.vitals
      ? el("span", { class: "chip st-ok" }, "vitals: consented")
      : el("span", { class: "chip" }, "vitals: off for this space"));
    badges.append(el("span", { class: "chip st-sim" }, "synthetic input"));

    const kind = stateOf(s);
    const row = (k, v) => add(host, el("dt", null, k), el("dd", null, v));
    row("Occupancy", stateWord(kind) + (kind === "blind" ? " — no usable link, which is not “empty”" : ""));
    if (kind === "occ" && s.occupied_since_s != null && s.t_s != null) row("Occupied for", dur(s.t_s - s.occupied_since_s));
    row("Motion index", num(s.motion_index, 2) + " × the empty-room level");
    if (kind === "occ") row("Still for", dur(s.still_for_s));
    if (!s.vitals) {
      row("Breathing", "not consented for this space — nothing is measured, simulated or not");
    } else if (!("breathing" in s)) {
      row("Breathing", "this space has consented, but your role is not permitted to see it");
    } else if (s.breathing) {
      row("Breathing", num(s.breathing.bpm, 1) + " BPM from link " + linkName(s.breathing.link));
      row("Confidence", num(s.breathing.confidence, 2));
    } else {
      row("Breathing", "no confident estimate — the estimator abstains rather than guess");
    }
    row("Links usable", (s.links_usable ?? 0) + " of " + (s.links_total ?? 0));

    for (const a of s.alerts || []) alerts.append(alertBox(a, s));
  }

  const ALERT_LABEL = {
    blind: "Cannot see this space", breathing_pause: "Breathing pause", inactivity: "No movement",
    inactivity_no_breathing: "No movement and no breathing detected", possible_collapse: "Possible collapse",
    rate_high: "Fast breathing", rate_low: "Slow breathing",
  };
  function alertBox(a, s) {
    const sev = a.severity || "advisory";
    const box = el("div", { class: "demo-alert", "data-sev": sev });
    add(box, el("strong", null, (ALERT_LABEL[a.name] || a.name) + " · " + sev));
    if (a.evidence) add(box, el("span", null, a.evidence));
    if (a.since_s != null && s && s.t_s != null) add(box, el("span", { class: "note" }, "since " + dur(s.t_s - a.since_s) + " ago"));
    add(box, el("span", { class: "note" }, a.notice || "advisory — not a medical device"));
    return box;
  }

  /**
   * The comparison, in words. This is the point of the page: when the
   * right-hand column loses a person the left-hand column is still
   * holding, it has to be unmissable.
   */
  function verdictBox(s) {
    const t = (SIM.cfg && SIM.cfg.truth) || {};
    const here = SIM.present && t.present !== false;
    const sees = stateOf(s);
    const vitals = !!s.vitals, shown = ("breathing" in s), b = s.breathing;
    let kind, head;
    if (!here) {
      if (sees === "occ") { kind = "bad"; head = "The system reports someone in an empty room."; }
      else if (sees === "blind") { kind = "warn"; head = "The room is empty — and the system cannot see it either way."; }
      else { kind = "ok"; head = "Empty room, and the system agrees."; }
    } else if (sees === "blind") {
      kind = "bad"; head = "The system has lost the person — it cannot see this space at all.";
    } else if (sees === "empty") {
      kind = "bad"; head = "The system has lost the person.";
    } else if (vitals && shown && t.breathing && !b) {
      kind = "warn"; head = "The system sees someone, but has lost their breathing.";
    } else if (vitals && shown && b && t.breathing && isFinite(t.bpm) && Math.abs(b.bpm - t.bpm) > BPM_TOL) {
      // A confident number that is not the right number is the worst of
      // the three outcomes. The comparison is two published rates side by
      // side — the simulator's input and the DSP's output — not anything
      // this page computed.
      kind = "bad"; head = "The system sees the person, but reports the wrong breathing rate.";
    } else {
      kind = "ok"; head = "The system sees the person.";
    }

    const box = el("div", { class: "verdict-box", "data-k": kind });
    add(box, el("strong", null, head));
    add(box, el("p", null, "Ground truth: " + (here
      ? "we placed them at " + num(t.x_m ?? (SIM.person || {}).x_m, 2) + " m, " + num(t.y_m ?? (SIM.person || {}).y_m, 2) + " m"
        + (t.breathing ? ", and told the simulator to breathe them at " + num(t.bpm, 1) + " BPM"
          + (t.in_pause ? " (in a pause right now)" : "")
          : ", with no breathing motion")
        + (t.still ? ", still" : ", moving")
      : "nobody in the room") + "."));
    add(box, el("p", null, "The system sees: "
      + (sees === "occ" ? "occupied" : sees === "empty" ? "empty" : "cannot see — no usable link")
      + (!vitals ? ", vital signs not consented for this space"
        : !shown ? ", breathing not shown to your role"
          : b ? ", breathing " + num(b.bpm, 1) + " BPM at confidence " + num(b.confidence, 2)
            : ", no confident breathing estimate") + "."));

    if (here && (kind === "bad" || kind === "warn")) {
      const p = SIM.person || {};
      const cv = covAt(p.x_m, p.y_m);
      const lost = sees !== "occ" || !b;
      let why;
      if (!lost) {
        why = "Ground truth is " + num(t.bpm, 1) + " BPM; the estimator returned " + num(b.bpm, 1)
          + " BPM at confidence " + num(b.confidence, 2) + ". A high confidence is a statement about the "
          + "signal, not a guarantee of the rate — compare the two columns, never the confidence alone.";
      } else if (cv == null) {
        why = "The simulator publishes no predicted sensitivity for this cell, so this view will not offer "
          + "a reason. The system is reporting that it cannot see, rather than guessing — that abstention "
          + "is the behaviour this view exists to show.";
      } else if (covBandOf(cv) <= 1) {
        why = "The simulator predicts " + num(cv, 2) + " (" + COV_BANDS[covBandOf(cv)].short
          + ") for a chest at this spot, and the pipeline did indeed lose them. The prediction and the "
          + "result agree here. The system is reporting that it cannot see, rather than guessing — that "
          + "abstention is the behaviour this view exists to show.";
      } else {
        // The map is a model. When it and the pipeline disagree, say so
        // rather than quoting the map as if it were the explanation.
        why = "Note the disagreement: the simulator predicted " + num(cv, 2) + " ("
          + COV_BANDS[covBandOf(cv)].short + ") for a chest at this spot, yet the pipeline lost them "
          + "anyway. The map is the simulator’s model of the channel; the right-hand column is what the "
          + "DSP actually recovered. Where they differ, the right-hand column is the one that is about "
          + "this system’s behaviour.";
      }
      add(box, el("p", { class: "why" }, why));
    }
    add(box, el("p", { class: "why" },
      "Both columns describe the same instant. The left is what the simulator was told; the right is what "
      + "the signal processing got back from it."));

    if (head !== SIM.said) { SIM.said = head; announce(head); }
    return box;
  }

  // -------------------------------------------------------------- controls
  for (const btn of document.querySelectorAll(".demo-ctl [data-x]")) {
    btn.addEventListener("click", () => {
      if (!SIM.els) return;
      if (!SIM.present) { togglePresent().then(() => moveTo(+btn.dataset.x, +btn.dataset.y, "preset")); return; }
      moveTo(+btn.dataset.x, +btn.dataset.y, "preset");
    });
  }
  $("#ex-present").addEventListener("click", togglePresent);

  const copy = $("#cmd-copy");
  copy.addEventListener("click", async () => {
    const pre = $("#cmd-text");
    let ok = false;
    try { await navigator.clipboard.writeText(pre.textContent); ok = true; } catch {
      const r = document.createRange();
      r.selectNodeContents(pre);
      const sel = getSelection();
      sel.removeAllRanges(); sel.addRange(r);
      try { ok = document.execCommand("copy"); } catch { /* leave it selected */ }
      sel.removeAllRanges();
    }
    const label = $("span", copy);
    label.textContent = ok ? "copied" : "select and copy";
    setTimeout(() => (label.textContent = "copy"), 1600);
  });

  startPolling();
})();
