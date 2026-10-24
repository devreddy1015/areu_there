// Saans room demo, simulation layer (spec/livedemo.md §9; ADR-008).
//
// Draws a room stream from GET /api/roomsim over the declared room that
// roomsim.js draws. Every number on that stream is SIM: simulated people
// and a simulated phone in simulated WiFi (mode `sim`), or a recorded
// run of the same (mode `replay`). Nothing here is sensed, and nothing
// here is LIVE: the page never calls this stream live.
//
// What this layer may draw, and how (ADR-008 decisions 3-16):
//   * a person or the phone only as an estimate WITH its radius, through
//     one function, drawEstimate, which refuses a radius that is absent,
//     non-finite, <= 0 or below min_plausible_radius_m (d5);
//   * the radius as an uncalibrated model radius, never as "95%";
//   * each frame as it arrives, with nothing between frames: no tween, no
//     interpolation, no animation loop (d6); trails join measured
//     estimates only and break at every abstention;
//   * an abstention as its reason, never the last estimate held (d7); a
//     closed socket clears the layer;
//   * RTI only beside the WLM baseline of the same tick (d9);
//   * truth and errors only in mode `sim`; truth in any other mode is a
//     protocol error (d15); people by index, the phone by no name (d3);
//     a truth cross or error segment never without its label, so neither
//     reads as a point estimate (d5);
//   * nothing written anywhere: positions stay in this module's memory,
//     bounded by the run, never in the URL, storage or the console (d11).
//
// Rules shared with roomsim.js: no innerHTML (Trusted Types would throw),
// styles through CSSOM only, colours from CSS tokens in roomsim.css,
// render on demand. This module imports nothing: roomsim.js hands it
// three.js and the view (createSim's `host`), so the page's module graph
// stays exactly what the gateway's tests pin.

/** The variants of a SIM session (§3.1); the page's default is with_walker. */
export const VARIANTS = ['declared', 'with_walker', 'empty'];
const DEFAULT_VARIANT = 'with_walker';

/** spec/locate.md §8: the smallest radius an estimate may be drawn with. */
const MIN_RADIUS_M = 0.3;
/** A run bounds a phone trail (§9.3): 120 s at 5 Hz. */
const TRAIL_MAX = 600;
/** How many frames the test hook keeps, as `seq:t:speed:paused:run`. */
const LOG_MAX = 120;

/** Heights the flat marks are drawn at, metres above the floor. */
const Z_FILL = 0.015;
const Z_MARK = 0.03;
/** Band widths, metres: rings, error segments and disturbed links. */
const RING_W = 0.025;
const ERR_W = 0.035;
const LINK_BAND_M = 0.03;
/** Half the extent of a truth cross (two 0.5 m arms at 45°), metres. */
const CROSS_HALF = 0.2;
/** Where along an error segment its label may sit, in order of preference. */
const ALONG = [0.5, 0.35, 0.65, 0.25, 0.75];

/** Render order: everything of this layer is drawn over the room. */
const ORDER = { link: 20, fill: 30, trail: 35, ring: 40, err: 45, truth: 50 };

/**
 * The four estimates and how each is drawn (§9.3). A pattern is metres
 * on, off, on, off … along the ring, scaled so a whole number of periods
 * fits; null is solid.
 */
const STYLES = {
  wlm: { name: 'WLM (baseline)', short: 'WLM', token: '--est-wlm', cls: 'wlm', pattern: null, line: 'solid' },
  rti: { name: 'RTI', short: 'RTI', token: '--est-rti', cls: 'rti', pattern: [0.30, 0.15], line: 'dashed' },
  rtt: { name: 'phone RTT', short: 'phone RTT', token: '--est-rtt', cls: 'rtt', pattern: [0.05, 0.10], line: 'dotted' },
  rssi: { name: 'phone RSSI', short: 'phone RSSI', token: '--est-rssi', cls: 'rssi', pattern: [0.30, 0.10, 0.05, 0.10], line: 'dash-dot' },
};

/** Exact texts the spec fixes (§9.1-§9.4). */
const TEXT = {
  signIn: 'Sign in at / with an admin or safety account to run the simulation (SIM)',
  unopened: 'The room stream could not be opened (signed out, not permitted, or not built into this gateway)',
  disconnected: 'Disconnected — nothing is shown',
  otherScene: 'Stream is for another scene',
  framesLost: 'frames lost — stopped',
  unknownMode: 'unknown mode',
  notDrawn: 'estimate not drawn: radius',
  noTruthReplay: 'no truth in REPLAY',
  emptyFalseAlarm: 'false alarm — the room is empty (SIM)',
  modelRadius: 'model radius, uncalibrated (SIM)',
  linkLegend: 'link disturbance: y ÷ WLM gate (SIM) — 0 at or below the empty room, 1 at the gate, 2 or more twice the gate',
  refDesc: "reference: the floor's centroid with the same radius — an estimator that knows nothing",
};

const f2 = (x) => Number(x).toFixed(2);
const finite = (...xs) => xs.every((x) => typeof x === 'number' && Number.isFinite(x));
const pct = (f) => (finite(f) ? `${(100 * f).toFixed(1)}%` : '—');
const metres = (x) => (finite(x) ? `${f2(x)} m` : '—');
const rate = (x) => (finite(x) ? x.toFixed(1) : '—');

/**
 * The mode badge, by an exhaustive mapping over the three modes of the
 * wire (§1, §9.4). Any other value is an error, never a mode.
 */
function modeBadge(mode) {
  switch (mode) {
    case 'sim': return { text: 'SIM', cls: 'mode-sim', known: true };
    case 'replay': return { text: 'REPLAY · recorded SIM run', cls: 'mode-replay', known: true };
    case 'live': return { text: 'LIVE', cls: 'mode-live', known: true };
  }
  return { text: TEXT.unknownMode, cls: 'mode-unknown', known: false };
}

/** Is (x, y) inside the polygon (even-odd rule)? */
function inside(x, y, poly) {
  let inPoly = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inPoly = !inPoly;
  }
  return inPoly;
}

/**
 * The simulation layer. `host` is the view roomsim.js owns: three.js,
 * the world, the label layer, the drawn scene and a redraw. Returns the
 * few calls roomsim.js makes back.
 */
export function createSim(host) {
  const { THREE, el } = host;
  const $ = (id) => document.getElementById(id);
  const ui = {
    toggle: $('sim-toggle'),
    badge: $('sim-badge'),
    overlay: $('sim-overlay'),
    card: $('sim-card'),
    statsCard: $('sim-stats-card'),
    mode: $('mode-badge'),
    variant: $('sim-variant'),
    seed: $('sim-seed'),
    replay: $('sim-replay'),
    play: $('sim-play'),
    speed: $('sim-speed'),
    restart: $('sim-restart'),
    truth: $('sim-truth'),
    noTruth: $('sim-notruth'),
    msg: $('sim-msg'),
    facts: $('sim-facts'),
    now: $('sim-now'),
    key: $('sim-key'),
    stats: $('sim-stats'),
    announce: $('sim-announce'),
  };
  const body = document.body;

  const S = {
    on: false,
    variant: DEFAULT_VARIANT, // as the URL gave it; the server judges it
    seed: '0',
    replay: false,
    truth: false,
    ws: null,
    token: 0,          // bumped by every open and close: stale events are dropped
    key: null,         // what the current (or last tried) socket is for
    got: false,        // the socket has delivered a message
    lastError: null,   // the latest `error` message's text
    expectSeq: 0,
    runIndex: null,
    mode: null,        // the latest message's mode
    recorded: null,    // REPLAY: {variant, seed} of the recorded run, as the stream says
    last: null,        // the frame drawn last
    msg: '',
    msgCls: '',
    announced: '',
    pendingSpeed: null,
    pendingPaused: null,
    trails: { rtt: [], rssi: [] }, // per method: segments of [x, y]
    trailOpen: { rtt: false, rssi: false },
    trailCount: { rtt: 0, rssi: 0 },
    log: [],
  };

  const group = new THREE.Group();
  group.name = 'sim';
  host.world.add(group);
  /** Labels this layer owns; roomsim.js places them with its own. */
  const simLabels = [];
  /** Rings drawn for the current frame: framing and counts. */
  let rings = [];
  let truthMarks = 0;
  let errSegs = 0;
  /** The length of each error segment as drawn, metres, for checks. */
  let errLens = [];
  let trailLines = 0;
  /** What the view's corner says about this frame beside the state. */
  let overlayLines = [];

  // ---------------------------------------------------------- palette

  let pal = null;
  function palette() {
    pal = {
      link: [host.colour('--link-0'), host.colour('--link-1'), host.colour('--link-2')],
      wlm: host.colour('--est-wlm'), rti: host.colour('--est-rti'),
      rtt: host.colour('--est-rtt'), rssi: host.colour('--est-rssi'),
      truth: host.colour('--truth'), err: host.colour('--truth-err'),
    };
  }
  palette();

  /** One sequential scale over g in [0, 2] (§9.3): empty room, gate, twice the gate. */
  function linkColour(g) {
    const c = Math.min(Math.max(g, 0), 2);
    return c <= 1
      ? new THREE.Color().lerpColors(pal.link[0], pal.link[1], c)
      : new THREE.Color().lerpColors(pal.link[1], pal.link[2], c - 1);
  }

  // ---------------------------------------------------------- materials

  /**
   * Materials outlive frames: a frame replaces geometry only, so no
   * shader program is relinked ten times a second (a phone's GPU pays
   * for that). They are dropped when the colours change and when the
   * view is turned off.
   */
  const mats = new Map();
  function material(key, make) {
    let m = mats.get(key);
    if (!m) mats.set(key, (m = make()));
    return m;
  }
  function dropMaterials() {
    for (const m of mats.values()) m.dispose();
    mats.clear();
  }

  /** A flat mark over the room in a palette colour: never depth-tested. */
  const flat = (key, opacity = 1) => material(`flat:${key}:${opacity}`, () => new THREE.MeshBasicMaterial({
    color: pal[key], transparent: true, opacity, depthTest: false, depthWrite: false, side: THREE.DoubleSide,
  }));

  /** Colour every vertex of a geometry: links share one material each. */
  function painted(geometry, colour) {
    const n = geometry.attributes.position.count;
    const c = new Float32Array(n * 3);
    for (let i = 0; i < n; i++) colour.toArray(c, i * 3);
    geometry.setAttribute('color', new THREE.BufferAttribute(c, 3));
    return geometry;
  }

  // ---------------------------------------------------------- geometry

  /** Quads of an annulus between r0 and r1, over the given arcs (radians). */
  function arcBands(r0, r1, arcs, z) {
    const pos = [];
    for (const [a0, a1] of arcs) {
      const n = Math.max(2, Math.ceil(((a1 - a0) / (2 * Math.PI)) * 128));
      for (let k = 0; k < n; k++) {
        const t0 = a0 + ((a1 - a0) * k) / n;
        const t1 = a0 + ((a1 - a0) * (k + 1)) / n;
        const p = (r, t) => [r * Math.cos(t), r * Math.sin(t), z];
        const [a, b, c, d] = [p(r0, t0), p(r1, t0), p(r1, t1), p(r0, t1)];
        pos.push(...a, ...b, ...c, ...a, ...c, ...d);
      }
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    return g;
  }

  /** The arcs of a line pattern along a circle of radius r. */
  function patternArcs(r, pattern) {
    if (!pattern) return [[0, 2 * Math.PI]];
    const period = pattern.reduce((a, b) => a + b, 0);
    const n = Math.max(4, Math.round((2 * Math.PI * r) / period));
    const scale = (2 * Math.PI) / (n * period); // metres of pattern -> radians
    const arcs = [];
    for (let k = 0; k < n; k++) {
      let a = k * period * scale;
      pattern.forEach((len, i) => {
        if (i % 2 === 0) arcs.push([a, a + len * scale]);
        a += len * scale;
      });
    }
    return arcs;
  }

  /** A flat band from a to b ([x, y]), width w, at height z. */
  function band(a, b, w, material, z) {
    const len = Math.hypot(b[0] - a[0], b[1] - a[1]);
    const m = new THREE.Mesh(new THREE.PlaneGeometry(Math.max(len, 1e-6), w), material);
    m.position.set((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, z);
    m.rotation.z = Math.atan2(b[1] - a[1], b[0] - a[0]);
    return m;
  }

  /** A truth cross (×): exact, so a point is honest here, and never a ring. */
  function cross(x, y) {
    const g = new THREE.Group();
    for (const ang of [Math.PI / 4, -Math.PI / 4]) {
      const m = new THREE.Mesh(new THREE.PlaneGeometry(0.5, 0.06), flat('truth'));
      m.rotation.z = ang;
      m.renderOrder = ORDER.truth;
      g.add(m);
    }
    g.position.set(x, y, Z_MARK + 0.005);
    return g;
  }

  /**
   * A label of this layer. roomsim.js places it with its own and never
   * hides it (`keep`): an estimate's carries its radius (d5), and a truth
   * cross or an error segment without its label would read as a point
   * estimate. `at` is one anchor or candidate anchors, tried in order;
   * `extra` may say the label is clamped into the view even when its
   * anchor is off screen (`clamp`, an estimate's: its ring may still
   * show) or what mark it names (`mark`, a truth cross, which the other
   * labels keep off).
   */
  function addLabel(at, cls, text, kind, extra = {}) {
    // Hidden until placed: a frame's labels are made as it arrives and
    // placed on the next animation frame, never shown at the corner between.
    const e = el('div', { class: `label ${cls} hidden`, text });
    host.labelLayer.append(e);
    simLabels.push({ el: e, at: [].concat(at), dim: false, text: () => text, kind, obj: null, keep: true, ...extra });
  }

  // ---------------------------------------------------------- the layer

  /** Remove this frame's meshes and labels; the materials stay. */
  function disposeLayer() {
    for (const c of [...group.children]) {
      group.remove(c);
      c.traverse((o) => { if (o.geometry) o.geometry.dispose(); });
    }
    for (const L of simLabels) L.el.remove();
    simLabels.length = 0;
    rings = [];
    truthMarks = 0;
    errSegs = 0;
    errLens = [];
    trailLines = 0;
  }

  function hook() {
    body.dataset.simLayer = String(group.children.length + simLabels.length);
    body.dataset.simRings = String(rings.length);
    body.dataset.simTruth = String(truthMarks);
    body.dataset.simErr = String(errSegs);
    body.dataset.simErrLen = errLens.map((l) => l.toFixed(3)).join(' ');
    body.dataset.simTrails = String(trailLines);
  }

  /**
   * Clear the layer, holding nothing (d7): meshes, labels, trails, truth,
   * the view corner's lines and the panel's figures. `why` is kept for
   * automated checks.
   */
  function clearLayer(why) {
    disposeLayer();
    if (why === 'off') dropMaterials();
    S.last = null;
    S.runIndex = null;
    S.trails = { rtt: [], rssi: [] };
    S.trailOpen = { rtt: false, rssi: false };
    S.trailCount = { rtt: 0, rssi: 0 };
    overlayLines = [];
    ui.now.replaceChildren();
    ui.stats.replaceChildren();
    ui.facts.replaceChildren();
    body.dataset.simCleared = why;
    hook();
    paintOverlay();
    host.render();
  }

  // ---------------------------------------------------------- estimates

  /**
   * The ONE function that draws an estimate (ADR-008 d5, §9.3). It takes
   * an Outcome's `estimate`, which carries its radius, never a bare
   * position, and refuses to draw one whose radius is absent, not
   * finite, <= 0 or below min_plausible_radius_m. What it draws: a ring
   * of the radius around the centre, never clipped to the room, with a
   * centre mark and a fill of at most 10%, over the room (no depth test,
   * a high render order), and its label, which the declutter never hides.
   * Returns what the panel says about it.
   */
  function drawEstimate(est, style, { person = false } = {}) {
    const r = est ? est.radius_m : undefined;
    if (!finite(r) || r <= 0 || r < MIN_RADIUS_M) return { drawn: false, why: TEXT.notDrawn };
    if (!finite(est.x, est.y)) return { drawn: false, why: 'estimate not drawn: position' };
    const sc = host.scene().sc;
    const fp = host.footprint(sc);
    const notes = [];
    if (fp.every(([px, py]) => Math.hypot(px - est.x, py - est.y) <= r)) notes.push('covers the whole room');
    if (!inside(est.x, est.y, fp)) notes.push('centre outside the room');
    if (person && S.last && S.last.run && S.last.run.variant === 'empty') notes.push(TEXT.emptyFalseAlarm);
    // Always "uncalibrated": a stream claiming otherwise was stopped
    // before drawing (frame), since Phase 5 calibrates nothing.
    const label = [`${style.name} ± ${f2(r)} m · ${TEXT.modelRadius}`, ...notes].join(' · ');
    if (host.gl()) {
      rings.push({ key: style.cls, x: est.x, y: est.y, r });
      const g = new THREE.Group();
      const ring = new THREE.Mesh(arcBands(r - RING_W / 2, r + RING_W / 2, patternArcs(r, style.pattern), Z_MARK), flat(style.cls));
      ring.renderOrder = ORDER.ring;
      const fill = new THREE.Mesh(new THREE.CircleGeometry(r, 96), flat(style.cls, 0.06));
      fill.position.z = Z_FILL;
      fill.renderOrder = ORDER.fill;
      const mark = new THREE.Mesh(new THREE.CircleGeometry(0.06, 20), flat(style.cls));
      mark.position.z = Z_MARK;
      mark.renderOrder = ORDER.ring;
      g.add(fill, ring, mark);
      g.position.set(est.x, est.y, 0);
      group.add(g);
      // On the ring itself first (its near side, then the others), so the
      // room inside it stays free for truth; the centre last.
      const on = (dx, dy) => new THREE.Vector3(est.x + dx * r, est.y + dy * r, Z_MARK);
      addLabel([on(0, -1), on(0, 1), on(-1, 0), on(1, 0), on(0, 0)], `est est-${style.cls}`, label, 'est', { clamp: true });
    }
    return { drawn: true, text: `(${f2(est.x)}, ${f2(est.y)}) ± ${f2(r)} m · ${TEXT.modelRadius}${notes.length ? ` · ${notes.join(' · ')}` : ''}` };
  }

  /**
   * One estimator's outcome: an estimate through drawEstimate, or an
   * abstention, which removes its circle (nothing of a previous frame is
   * kept) and is said as text (d7).
   */
  function outcome(o, style, opts) {
    if (o && o.estimate) {
      const d = drawEstimate(o.estimate, style, opts);
      return d.drawn
        ? { shown: true, est: o.estimate, text: d.text }
        : { shown: false, refused: true, text: `${d.why} (SIM)` };
    }
    if (o && o.abstain) return { shown: true, abstained: true, text: `abstained — ${o.abstain.reason} (SIM)` };
    return { shown: false, refused: true, text: 'no outcome in this frame (SIM)' };
  }

  // ---------------------------------------------------------- links

  /**
   * Links (§9.3): one segment per declared node pair between the nodes'
   * SCENE positions. Its value g is the larger over its two directions
   * of y / gate (0 for a null or non-positive y, or a null gate); a pair
   * over the WLM gate in either direction is a solid band, the others a
   * 1 px dashed line, so the gate reads without colour.
   */
  function drawLinks(links, sc) {
    const pairs = new Map();
    for (const it of links.items || []) {
      const a = Math.min(it.tx, it.rx);
      const b = Math.max(it.tx, it.rx);
      const key = `${a}-${b}`;
      const g = finite(it.y) && it.y > 0 && finite(it.gate) && it.gate > 0 ? it.y / it.gate : 0;
      const p = pairs.get(key) || { a, b, g: 0, disturbed: false };
      p.g = Math.max(p.g, g);
      p.disturbed = p.disturbed || it.disturbed === true;
      pairs.set(key, p);
    }
    const pos = new Map((sc.nodes || []).map((n) => [n.id, n.pos]));
    let over = 0;
    for (const p of pairs.values()) {
      if (p.disturbed) over++;
      const pa = pos.get(p.a);
      const pb = pos.get(p.b);
      if (!host.gl() || !pa || !pb) continue;
      const va = new THREE.Vector3(...pa);
      const vb = new THREE.Vector3(...pb);
      const colour = linkColour(p.g);
      if (p.disturbed) {
        const len = va.distanceTo(vb);
        const m = new THREE.Mesh(painted(new THREE.BoxGeometry(len, LINK_BAND_M, LINK_BAND_M), colour),
          material('link:band', () => new THREE.MeshBasicMaterial({ vertexColors: true })));
        m.position.copy(va).add(vb).multiplyScalar(0.5);
        m.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), vb.clone().sub(va).normalize());
        m.renderOrder = ORDER.link;
        group.add(m);
      } else {
        const line = new THREE.Line(painted(new THREE.BufferGeometry().setFromPoints([va, vb]), colour),
          material('link:dash', () => new THREE.LineDashedMaterial({ vertexColors: true, dashSize: 0.12, gapSize: 0.09 })));
        line.computeLineDistances();
        line.renderOrder = ORDER.link;
        group.add(line);
      }
    }
    return { over, pairs: pairs.size };
  }

  // ---------------------------------------------------------- trails

  /**
   * Phone trails (§9.3), from `new.phone` only: every epoch arrives in
   * exactly one frame's `new`. A segment joins consecutive published
   * estimates of one method; an abstention (or a refused radius) ends it.
   */
  function addEpochs(epochs) {
    for (const e of epochs || []) {
      for (const k of ['rtt', 'rssi']) {
        const est = e[k] && e[k].estimate;
        const ok = est && finite(est.x, est.y, est.radius_m) && est.radius_m >= MIN_RADIUS_M;
        if (!ok) {
          S.trailOpen[k] = false;
          continue;
        }
        if (!S.trailOpen[k]) S.trails[k].push([]);
        S.trailOpen[k] = true;
        S.trails[k][S.trails[k].length - 1].push([est.x, est.y]);
        S.trailCount[k]++;
        // The run bounds a trail; drop the oldest beyond it.
        while (S.trailCount[k] > TRAIL_MAX) {
          const first = S.trails[k][0];
          first.shift();
          S.trailCount[k]--;
          if (!first.length) S.trails[k].shift();
        }
      }
    }
  }

  function drawTrails() {
    if (!host.gl()) return;
    for (const k of ['rtt', 'rssi']) {
      for (const seg of S.trails[k]) {
        if (seg.length < 2) continue;
        const line = new THREE.Line(
          new THREE.BufferGeometry().setFromPoints(seg.map(([x, y]) => new THREE.Vector3(x, y, Z_MARK - 0.005))),
          material(`trail:${k}`, () => new THREE.LineBasicMaterial({
            color: pal[k], transparent: true, opacity: 0.6, depthTest: false, depthWrite: false,
          })),
        );
        line.renderOrder = ORDER.trail;
        group.add(line);
        trailLines++;
      }
    }
  }

  // ---------------------------------------------------------- truth

  /**
   * Truth (mode `sim` only, toggle on): crosses at each record's own
   * truth, and a segment from each scored estimate to what it was scored
   * against, so a segment's length is the printed error (§9.3).
   */
  function drawTruth(f, drawn) {
    const lines = [];
    const tp = f.truth && f.truth.persons;
    if (tp) {
      lines.push(`Simulated people: ${tp.people.length}`);
      for (const p of tp.people) {
        if (!finite(p.x, p.y)) continue;
        if (host.gl()) truthCross(p.x, p.y, `P${p.i + 1} · truth (SIM)`);
        truthMarks++;
      }
      for (const k of ['wlm', 'rti']) {
        const s = tp[k];
        const est = drawn[k];
        if (!s) continue;
        if (s.unscored) {
          lines.push(`${STYLES[k].short}: unscored — ${s.unscored} (SIM)`);
          continue;
        }
        const p = tp.people.find((q) => q.i === s.person);
        const text = `${STYLES[k].short} error to P${s.person + 1}: ${f2(s.error_m)} m (SIM)`;
        lines.push(text);
        if (est && p) segment(est, [p.x, p.y], text);
      }
    }
    const ph = f.truth && f.truth.phone;
    if (ph && finite(ph.x, ph.y)) {
      if (host.gl()) truthCross(ph.x, ph.y, 'phone · truth (SIM)');
      truthMarks++;
      for (const k of ['rtt', 'rssi']) {
        const s = ph[k];
        if (!s) continue;
        if (s.unscored) {
          lines.push(`${STYLES[k].short}: unscored — ${s.unscored} (SIM)`);
          continue;
        }
        const text = `${STYLES[k].short} error to the phone: ${f2(s.error_m)} m (SIM)`;
        lines.push(text);
        if (drawn[k]) segment(drawn[k], [ph.x, ph.y], text);
      }
    }
    return lines;
  }

  /** A cross and the label that names it; other labels keep off the cross. */
  function truthCross(x, y, text) {
    group.add(cross(x, y));
    const at = new THREE.Vector3(x, y, Z_MARK);
    addLabel(at, 'truth', text, 'truth', { mark: { at, half: CROSS_HALF } });
  }

  /** An error segment, its label anywhere along it. */
  function segment(est, to, text) {
    errSegs++;
    if (!host.gl()) return;
    const m = band([est.x, est.y], to, ERR_W, flat('err'), Z_MARK + 0.002);
    m.renderOrder = ORDER.err;
    group.add(m);
    errLens.push(m.geometry.parameters.width);
    const a = new THREE.Vector3(est.x, est.y, Z_MARK);
    const b = new THREE.Vector3(to[0], to[1], Z_MARK);
    addLabel(ALONG.map((t) => a.clone().lerp(b, t)), 'err', text, 'err');
  }

  // ---------------------------------------------------------- panel

  function kv(pairs) {
    return pairs.flatMap(([k, v, cls, title]) => [el('dt', { text: k }), el('dd', { text: v, class: cls, title })]);
  }

  /** "too few to summarise (k)" below 5 for a median and 10 for p90 (§9.4). */
  function errorFigures(k, median, p90) {
    if (k < 5) return `too few to summarise (${k})`;
    const m = `median ${metres(median)}`;
    return k < 10 ? `${m}; p90: too few to summarise (${k})` : `${m}, p90 ${metres(p90)}`;
  }

  /** One estimator's block (§9.4), from §5's statistics of this frame. */
  function block(key, st, f) {
    const style = STYLES[key];
    const items = [];
    const li = (text, cls) => items.push(el('li', { text, class: cls }));
    const replay = f.mode === 'replay';
    const person = key === 'wlm' || key === 'rti';
    let heading = `${style.name} · SIM`;
    if (person && f.run.variant === 'empty') {
      heading = `${style.name} · False alarms — empty room (SIM)`;
      const u = st.unoccupied || {};
      li(`published ${u.published ?? 0} of ${u.ticks ?? 0} ticks (${pct(u.ticks ? u.published / u.ticks : null)}) over ${rate(u.seconds)} s (SIM)`);
      li(`${rate(u.per_h)} published ticks/h, ${rate(u.events_per_h)} events/h (${u.events ?? 0} events) (SIM)`);
      return el('div', { class: 'est-block' }, el('h4', { text: heading }), el('ul', {}, ...items));
    }
    const o = person ? st.occupied || {} : st;
    const n = person ? o.ticks : o.epochs;
    const k = o.published ?? 0;
    li(`published ${k} of ${n ?? 0} (${pct(o.coverage)}) (SIM)`);
    if (person) {
      const e = st.empty_run;
      li(e ? `empty room: ${rate(e.per_h)} published ticks/h, ${rate(e.events_per_h)} events/h (SIM)`
        : 'empty room: not run (SIM)');
    }
    if (replay) {
      li(`radius median ${metres(o.radius_median_m)} (SIM)`);
      li(TEXT.noTruthReplay, 'none');
    } else {
      li(`error to ${person ? 'the nearest person' : 'the phone'}: ${errorFigures(k, o.median_m, o.p90_m)} (SIM)`);
      li(`radius median ${metres(o.radius_median_m)}; truth inside the radius ${pct(o.inside_radius)} (SIM)`);
      const r = o.ref || {};
      li(`${TEXT.refDesc}: ${errorFigures(k, r.median_m, r.p90_m)}; inside ${pct(r.inside_radius)} (SIM)`, 'ref');
    }
    return el('div', { class: 'est-block' }, el('h4', { text: heading }), el('ul', {}, ...items));
  }

  function fillStats(f) {
    const st = f.stats || {};
    const blocks = [];
    // WLM is always shown beside RTI (d9).
    if (st.wlm && st.rti) blocks.push(block('wlm', st.wlm, f), block('rti', st.rti, f));
    for (const k of ['rtt', 'rssi']) if (st[k]) blocks.push(block(k, st[k], f));
    ui.stats.replaceChildren(...blocks);
  }

  /** A frame's state, as checks see it: warming_up, running, paused or ended. */
  function stateOf(f) {
    if (f.ended) return 'ended';
    if (f.paused) return 'paused';
    return f.state === 'warming_up' ? 'warming_up' : 'running';
  }
  const WORDS = { warming_up: 'Warming up', running: 'Running', paused: 'Paused', ended: 'Ended' };

  function speedText(f) {
    const eff = f.effective_speed;
    // Shown whenever it is lower, as printed: a measured 0.9999x at 1x is 1.00x.
    const lower = finite(eff) && Number(eff.toFixed(2)) < f.speed;
    return `${f.speed}×${lower ? ` (effective ${eff.toFixed(2)}×)` : ''} (SIM)`;
  }

  function fillFacts(m) {
    const sc = m.scene || {};
    const run = m.run || {};
    const hash = String(sc.sha256 || '');
    const fixed = m.mode === 'replay' ? ' (fixed by the recorded run)' : '';
    if (m.mode === 'replay') {
      // The controls show what is running: the recording's, never what was asked.
      S.recorded = { variant: String(run.variant), seed: String(run.seed) };
      paintRunControls();
    }
    const rows = [
      ['Label', String(m.label)],
      ['Scene', String(sc.name)],
      ['Scene file sha256', hash ? `${hash.slice(0, 16)}…` : '—', 'mono', hash],
      ['Provenance', host.provenanceText(sc.provenance), `prov-${sc.provenance}`],
      ['Variant · seed', `${run.variant} · seed ${run.seed}${fixed}`],
    ];
    if (m.type === 'frame') {
      rows.push(['Time', `t = ${m.t.toFixed(1)} s of ${run.duration_s} s (SIM)`]);
      rows.push(['State', `${WORDS[stateOf(m)].toLowerCase()} · run ${run.index}`]);
      rows.push(['Speed', speedText(m)]);
    } else {
      rows.push(['State', m.state]);
    }
    ui.facts.replaceChildren(...kv(rows));
  }

  // ---------------------------------------------------------- status

  /**
   * What the polite region says on entering a state (§9.5): only state
   * changes, never figures. Every state in which nothing is shown says
   * so, with its reason, so no one is left believing the run goes on.
   */
  const ANNOUNCE = {
    calibrating: () => 'Simulation calibrating (SIM)',
    running: () => 'Simulation running (SIM)',
    paused: () => 'Simulation paused (SIM)',
    ended: () => 'Simulation ended (SIM)',
    disconnected: () => 'Simulation disconnected — nothing is shown',
    // The socket was closed: by the page on a protocol error, or by the
    // gateway on a refusal or a failed handshake.
    stopped: (msg) => `Simulation stopped — nothing is shown: ${msg}`,
    refused: (msg) => `Simulation stopped — nothing is shown: ${msg}`,
    unopened: (msg) => `Simulation stopped — nothing is shown: ${msg}`,
    // No socket was opened.
    'signed-out': (msg) => `Simulation not started: ${msg}`,
    'not-permitted': (msg) => `Simulation not started: ${msg}`,
    error: (msg) => `Simulation not started: ${msg}`,
    'other-scene': (msg) => `Nothing is shown: ${msg}`,
  };

  /** The state, for the panel, the view's corner, checks and announcements. */
  function setState(state, msg = '', cls = '') {
    S.msg = msg;
    S.msgCls = cls;
    body.dataset.simState = state;
    // Pause and speed act on an open session only.
    ui.play.disabled = ui.speed.disabled = !(S.ws && S.ws.readyState <= 1);
    ui.msg.textContent = msg;
    ui.msg.className = `sim-msg${cls ? ` ${cls}` : ''}`;
    const word = state === 'warming_up' ? 'running' : state;
    const say = ANNOUNCE[word] ? ANNOUNCE[word](msg) : null;
    if (state === 'off') S.announced = ''; // the viewer's own act; the next run is news
    if (say && say !== S.announced) {
      S.announced = say;
      ui.announce.textContent = say;
    }
    paintOverlay();
  }

  /** Is this view REPLAY: by the stream's word once it has spoken, else as asked. */
  const isReplay = () => (S.mode ? S.mode === 'replay' : S.replay);
  function paintOverlay() {
    const lines = [];
    if (S.msg) lines.push([S.msg, S.msgCls]);
    for (const l of overlayLines) lines.push(l);
    if (S.on && isReplay()) lines.push([TEXT.noTruthReplay, '']);
    ui.overlay.replaceChildren(...lines.map(([t, c]) => el('div', { text: t, class: c || null })));
    ui.overlay.hidden = !S.on || !lines.length;
  }

  function paintBadge(b) {
    ui.mode.hidden = !b;
    if (!b) {
      body.dataset.simMode = '';
      return;
    }
    ui.mode.textContent = b.text;
    ui.mode.className = `badge mode ${b.cls}`;
    body.dataset.simMode = b.known ? S.mode : 'unknown';
  }

  // ---------------------------------------------------------- frames

  /** Every Outcome a frame carries, the latest records' and `new`'s. */
  function outcomes(f) {
    const out = [];
    if (f.persons) out.push(f.persons.wlm, f.persons.rti);
    if (f.phone) out.push(f.phone.rtt, f.phone.rssi);
    for (const r of (f.new && f.new.persons) || []) out.push(r.wlm, r.rti);
    for (const e of (f.new && f.new.phone) || []) out.push(e.rtt, e.rssi);
    return out;
  }

  function frame(f) {
    if (!Number.isInteger(f.seq) || f.seq !== S.expectSeq) return protocolError(TEXT.framesLost);
    S.expectSeq = f.seq + 1;
    if ('truth' in f && f.mode !== 'sim') {
      return protocolError('the stream sent ground truth outside SIM — stopped, nothing is drawn');
    }
    // Phase 5 calibrates nothing (§6.2): a radius said to be calibrated
    // is a broken promise, never a label to print.
    if (outcomes(f).some((o) => o && o.estimate && o.estimate.radius_calibrated === true)) {
      return protocolError('the stream says a radius is calibrated, and none is — stopped, nothing is drawn');
    }
    const { entry } = host.scene();
    if (!entry || !f.scene || f.scene.sha256 !== entry.sha256) {
      clearLayer('other-scene');
      setState('other-scene', TEXT.otherScene, 'warn');
      return;
    }
    if (S.runIndex !== null && f.run.index !== S.runIndex) clearLayer('run');
    S.runIndex = f.run.index;
    S.last = f;
    addEpochs(f.new && f.new.phone);
    S.log.push(`${f.seq}:${f.t}:${f.speed}:${f.paused ? 1 : 0}:${f.run.index}`);
    if (S.log.length > LOG_MAX) S.log.shift();
    body.dataset.simLog = S.log.join(' ');
    body.dataset.simSeq = String(f.seq);
    body.dataset.simT = String(f.t);
    body.dataset.simRun = String(f.run.index);
    draw(f);
    host.drawn();
    host.render();
  }

  /** Draw a frame and fill the panel from it. Called on arrival only. */
  function draw(f) {
    disposeLayer();
    const sc = host.scene().sc;
    const now = [];
    const drawn = {};
    overlayLines = [];
    const say = (k, text, why) => {
      now.push(el('li', {}, `${STYLES[k].name}: `, el('span', { text, class: why ? 'why' : null })));
      if (why) overlayLines.push([`${STYLES[k].name}: ${text}`, 'warn']);
    };
    if (f.links) {
      const l = drawLinks(f.links, sc);
      now.push(el('li', { text: `links over the WLM gate: ${l.over} of ${l.pairs} pairs (SIM, tick t = ${f.links.t.toFixed(1)} s)` }));
      body.dataset.simLinks = `${l.over}/${l.pairs}`;
    }
    if (f.persons) {
      now.push(el('li', { text: `people estimates, tick t = ${f.persons.t.toFixed(1)} s (SIM):` }));
      const w = outcome(f.persons.wlm, STYLES.wlm, { person: true });
      say('wlm', w.text, !w.est);
      if (w.est) drawn.wlm = w.est;
      // The baseline rule (d9): RTI only beside WLM's outcome of the same tick.
      if (w.shown) {
        const r = outcome(f.persons.rti, STYLES.rti, { person: true });
        say('rti', r.text, !r.est);
        if (r.est) drawn.rti = r.est;
      } else {
        say('rti', 'not drawn — the WLM baseline of this tick is not drawn (SIM)', true);
      }
    } else {
      now.push(el('li', { text: 'people estimates: none yet — Locate is warming up (SIM)' }));
    }
    if ('phone' in f) {
      if (f.phone) {
        now.push(el('li', { text: `phone estimates, epoch t = ${f.phone.t.toFixed(1)} s (SIM):` }));
        for (const k of ['rtt', 'rssi']) {
          const o = outcome(f.phone[k], STYLES[k]);
          say(k, o.text, !o.est);
          if (o.est) drawn[k] = o.est;
        }
      } else {
        now.push(el('li', { text: 'phone estimates: none yet (SIM)' }));
      }
      drawTrails();
    }
    if (f.mode === 'sim' && S.truth) {
      for (const t of drawTruth(f, drawn)) now.push(el('li', { text: t }));
    }
    ui.now.replaceChildren(...now);
    fillFacts(f);
    fillStats(f);
    syncControls(f);
    const st = stateOf(f);
    if (st === 'ended') setState(st, `Ended at t = ${f.t.toFixed(1)} s (SIM) — Restart to run again`);
    else setState(st, `${WORDS[st]} · t = ${f.t.toFixed(1)} s · ${speedText(f)}`);
    hook();
  }

  function syncControls(f) {
    if (S.pendingSpeed === null || f.speed === S.pendingSpeed) {
      S.pendingSpeed = null;
      ui.speed.value = String(f.speed);
    }
    if (S.pendingPaused === null || f.paused === S.pendingPaused) {
      S.pendingPaused = null;
      ui.play.textContent = f.paused ? 'Play' : 'Pause';
    }
    showTruthControl();
  }

  /** Truth is offered in SIM only; REPLAY says it has none (§9.3). */
  function showTruthControl() {
    const replay = isReplay();
    ui.truth.hidden = replay;
    ui.noTruth.hidden = !replay;
  }

  // ---------------------------------------------------------- the socket

  function send(cmd) {
    if (S.ws && S.ws.readyState === WebSocket.OPEN) S.ws.send(JSON.stringify(cmd));
  }

  /**
   * Close our socket, if any; its events no longer reach the page. One
   * still connecting is closed once it opens: closing it mid-handshake
   * would only make the browser log an error.
   */
  function closeSocket() {
    S.token++;
    const ws = S.ws;
    S.ws = null;
    if (!ws) return;
    if (ws.readyState === WebSocket.CONNECTING) ws.addEventListener('open', () => ws.close(1000));
    else if (ws.readyState === WebSocket.OPEN) ws.close(1000);
  }

  /** A broken promise of the stream (§9.2): clear, close, say so. */
  function protocolError(text) {
    closeSocket();
    clearLayer('protocol');
    setState('stopped', text, 'bad');
  }

  function message(text) {
    let m;
    try {
      m = JSON.parse(text);
    } catch (err) {
      return protocolError('the stream sent a message that is not JSON — stopped');
    }
    S.got = true;
    if (!m || typeof m !== 'object') return protocolError('the stream sent a message that is not an object — stopped');
    // The badge comes from the stream's own messages, never inferred.
    S.mode = m.mode;
    const b = modeBadge(S.mode);
    paintBadge(b);
    if (!b.known) return protocolError(TEXT.unknownMode);
    // No LIVE source exists (§7), and this page draws none (d13, d16): a
    // LIVE message of any type stops the stream, its badge an error.
    if (m.mode === 'live') return protocolError('the stream said LIVE; this page shows SIM and REPLAY only — stopped');
    if (m.label !== 'SIM') return protocolError('the stream sent numbers not labelled SIM — stopped, nothing is drawn');
    switch (m.type) {
      case 'state': {
        const { entry } = host.scene();
        if (!entry || !m.scene || m.scene.sha256 !== entry.sha256) {
          setState('other-scene', TEXT.otherScene, 'warn');
          return;
        }
        fillFacts(m);
        showTruthControl();
        if (m.state === 'calibrating') setState('calibrating', 'Calibrating the simulation (SIM): a few seconds');
        else setState(m.state, m.state === 'ready' ? 'Ready: a recorded SIM run' : String(m.state));
        return;
      }
      case 'frame':
        return frame(m);
      case 'error':
        // An error naming a command leaves the session running; one
        // followed by a close is said when the close comes.
        S.lastError = String(m.message);
        ui.msg.textContent = `The gateway: ${S.lastError}`;
        ui.msg.className = 'sim-msg warn';
        return;
    }
    return protocolError('the stream sent a message of an unknown type — stopped');
  }

  function closed(code) {
    S.ws = null;
    clearLayer('close');
    if (!S.got) setState('unopened', TEXT.unopened, 'bad');
    else if (S.lastError && (code === 1008 || code === 1013)) setState('refused', S.lastError, 'bad');
    else setState('disconnected', S.lastError ? `${TEXT.disconnected} (${S.lastError})` : TEXT.disconnected, 'warn');
    // Never reconnect on its own (§9.1).
  }

  /** What a socket for the current view and parameters is for. */
  function wantKey() {
    const { entry } = host.scene();
    if (!S.on || !entry) return null;
    return S.replay ? `${entry.id}|replay` : `${entry.id}|${S.variant}|${S.seed}`;
  }

  /**
   * Open the stream, after asking /api/me whether this viewer may see
   * simulated positions (§9.1); a refusal there opens no socket.
   */
  async function open() {
    closeSocket();
    const token = S.token;
    const { entry } = host.scene();
    S.key = wantKey();
    if (!S.key) return;
    S.log = [];
    body.dataset.simLog = '';
    S.got = false;
    S.lastError = null;
    S.expectSeq = 0;
    S.mode = null;
    S.recorded = null;
    S.pendingSpeed = null;
    S.pendingPaused = null;
    paintBadge(null);
    paintRunControls();
    if (S.last || group.children.length || simLabels.length) clearLayer('open');
    setState('checking', 'Checking access…');
    let me;
    try {
      const r = await fetch('/api/me', { cache: 'no-store', credentials: 'same-origin' });
      if (!r.ok) throw new Error(`HTTP ${r.status}`);
      me = await r.json();
    } catch (err) {
      if (token === S.token) setState('error', `Could not check access (/api/me): ${err.message}`, 'bad');
      return;
    }
    if (token !== S.token) return;
    if (me.auth_required && !me.authenticated) return setState('signed-out', TEXT.signIn, 'warn');
    if (!(me.can && me.can.position === true)) {
      return setState('not-permitted', `Your role (${me.role}) cannot see simulated positions; an admin or safety account can`, 'warn');
    }
    const q = new URLSearchParams({ scene: entry.id });
    if (S.replay) q.set('replay', '1');
    else {
      q.set('variant', S.variant);
      q.set('seed', S.seed);
    }
    setState('connecting', 'Opening the room stream…');
    const ws = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/api/roomsim?${q}`);
    S.ws = ws;
    ws.addEventListener('message', (e) => { if (token === S.token) message(e.data); });
    ws.addEventListener('close', (e) => { if (token === S.token) closed(e.code); });
  }

  /** Make the socket match the view: open, reopen or close. */
  function sync() {
    const want = wantKey();
    if (want === S.key) return;
    if (want === null) {
      closeSocket();
      S.key = null;
      clearLayer('off');
      setState(S.on ? 'no-scene' : 'off');
      return;
    }
    open();
  }

  // ---------------------------------------------------------- controls

  /**
   * The variant and seed controls show what runs (§9.1): what the viewer
   * asked for in SIM; in REPLAY, the recorded run's once the stream has
   * said it, and nothing before, since a recording fixes both.
   */
  function paintRunControls() {
    const rec = S.replay ? S.recorded : null;
    ui.variant.value = S.replay ? (rec ? rec.variant : '') : S.variant;
    ui.seed.value = S.replay ? (rec ? rec.seed : '') : S.seed;
  }

  function showOn() {
    ui.toggle.setAttribute('aria-pressed', String(S.on));
    for (const e of [ui.badge, ui.card, ui.statsCard]) e.hidden = !S.on;
    paintRunControls();
    ui.replay.checked = S.replay;
    ui.variant.disabled = ui.seed.disabled = S.replay;
    ui.truth.setAttribute('aria-pressed', String(S.truth));
    showTruthControl();
    paintOverlay();
    host.textChanged();
  }

  ui.toggle.addEventListener('click', () => {
    S.on = !S.on;
    if (!S.on) {
      closeSocket();
      S.key = null;
      clearLayer('off');
      paintBadge(null);
      setState('off');
    } else {
      host.simStarted();
    }
    showOn();
    host.writeHash();
    sync();
  });
  ui.variant.addEventListener('change', () => {
    S.variant = ui.variant.value;
    host.writeHash();
    sync();
  });
  ui.seed.addEventListener('change', () => {
    S.seed = ui.seed.value.trim() || '0';
    ui.seed.value = S.seed;
    host.writeHash();
    sync();
  });
  ui.replay.addEventListener('change', () => {
    S.replay = ui.replay.checked;
    showOn();
    host.writeHash();
    sync();
  });
  ui.play.addEventListener('click', () => {
    const pause = ui.play.textContent === 'Pause';
    S.pendingPaused = pause;
    ui.play.textContent = pause ? 'Play' : 'Pause';
    send({ cmd: pause ? 'pause' : 'play' });
  });
  ui.speed.addEventListener('change', () => {
    const v = Number(ui.speed.value);
    S.pendingSpeed = v;
    send({ cmd: 'speed', value: v });
  });
  ui.restart.addEventListener('click', () => {
    // A restart of an open session; with no session, the viewer asks for
    // a new one (the page never reconnects by itself).
    if (S.ws && S.ws.readyState === WebSocket.OPEN) send({ cmd: 'restart' });
    else if (S.on) open();
  });
  ui.truth.addEventListener('click', () => {
    S.truth = !S.truth;
    ui.truth.setAttribute('aria-pressed', String(S.truth));
    if (S.last) {
      draw(S.last);
      host.drawn();
    }
    host.render();
  });

  /** The key (§9.3): every line style, independent of Dimensions. */
  function buildKey() {
    const svg = (tag, attrs) => {
      const e = document.createElementNS('http://www.w3.org/2000/svg', tag);
      for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, String(v));
      return e;
    };
    const DASH = { solid: null, dashed: '6 3', dotted: '1.5 3', 'dash-dot': '6 3 1.5 3' };
    const items = Object.values(STYLES).map((s) => {
      const pic = svg('svg', { viewBox: '0 0 34 12', 'aria-hidden': 'true', class: `k-${s.cls}` });
      const line = svg('line', { x1: 1, y1: 6, x2: 33, y2: 6, stroke: 'currentColor', 'stroke-width': 2 });
      if (DASH[s.line]) line.setAttribute('stroke-dasharray', DASH[s.line]);
      pic.append(line);
      return el('li', {}, pic, `${s.name}: ${s.line} ring of its model radius, uncalibrated (SIM)`);
    });
    const x = svg('svg', { viewBox: '0 0 34 12', 'aria-hidden': 'true', class: 'k-truth' });
    x.append(svg('path', { d: 'M12 1 L22 11 M22 1 L12 11', stroke: 'currentColor', 'stroke-width': 2 }));
    items.push(el('li', {}, x, 'truth: a cross, in SIM only, with Truth on'));
    const e = svg('svg', { viewBox: '0 0 34 12', 'aria-hidden': 'true', class: 'k-err' });
    e.append(svg('line', { x1: 1, y1: 6, x2: 33, y2: 6, stroke: 'currentColor', 'stroke-width': 3 }));
    items.push(el('li', {}, e, 'error: estimate centre to its truth, labelled in metres (SIM)'));
    items.push(el('li', {}, el('span', { class: 'ramp', 'aria-hidden': 'true' }), TEXT.linkLegend));
    items.push(el('li', {}, 'links: a solid band over the WLM gate, a dashed line under it; trails: thin lines through published phone estimates'));
    ui.key.replaceChildren(...items);
  }
  buildKey();
  hook();
  body.dataset.simState = 'off';

  return {
    /** The hash keys of the simulation view (§9.1). */
    readParams(p) {
      const was = S.on;
      S.on = p.get('sim') === '1';
      S.replay = p.get('replay') === '1';
      // A recording fixes the variant and seed; the URL keeps none in REPLAY.
      S.variant = (!S.replay && p.get('variant')) || DEFAULT_VARIANT;
      S.seed = (!S.replay && p.get('seed')) || '0';
      if (!S.on && (S.ws || S.key)) {
        closeSocket();
        S.key = null;
        clearLayer('off');
        paintBadge(null);
        setState('off');
      }
      if (S.on && !was) host.simStarted();
      showOn();
    },
    writeParams(p) {
      if (S.on) p.set('sim', '1');
      if (S.replay) p.set('replay', '1');
      else {
        if (S.variant !== DEFAULT_VARIANT) p.set('variant', S.variant);
        if (S.seed !== '0') p.set('seed', S.seed);
      }
    },
    /** The drawn scene changed (or is gone): close, clear, reopen if on (§9.2). */
    sceneChanged() {
      closeSocket();
      S.key = null;
      clearLayer('scene');
      paintBadge(null);
      setState(S.on ? 'scene' : 'off');
      sync();
    },
    sync,
    isOn: () => S.on,
    labels: () => simLabels,
    /** The drawn rings' extent, for framing while the view is on (§9.3). */
    bounds() {
      if (!S.on || !rings.length) return null;
      return rings.reduce((b, { x, y, r }) => [
        Math.min(b[0], x - r), Math.min(b[1], y - r), Math.max(b[2], x + r), Math.max(b[3], y + r),
      ], [Infinity, Infinity, -Infinity, -Infinity]);
    },
    /** New colour tokens: new materials, and the last frame redrawn with them. */
    retheme() {
      disposeLayer();
      dropMaterials();
      palette();
      if (S.last) draw(S.last);
      hook();
    },
  };
}
