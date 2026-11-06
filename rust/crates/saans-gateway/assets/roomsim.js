// Saans room demo (ADR-008): draws a DECLARED scene with three.js.
//
// The room comes from a scene file and is labelled SCENE: it is declared,
// never measured by WiFi. The gateway serves a scene through an allowlist
// that leaves out the simulator's people and phone, so the static view
// draws no one. A value the scene does not declare is shown as "not
// declared", never as a number.
//
// People and estimates appear only in the simulation view (SIM), which
// roomsim-sim.js draws over this room from the room stream when the
// viewer asks for it ("Run simulation (SIM)" or #sim=1): simulated
// people, estimated from simulated WiFi. This file hands that layer the
// view, places its labels with the room's, and frames its rings.
//
// Rules this file keeps (the dashboard's, plus the demo page's CSP):
//   * no innerHTML anywhere — the page's CSP turns on Trusted Types with
//     no policy, so the browser throws if anything tries;
//   * no inline style or script: styles are set through CSSOM properties
//     only, which the page's `style-src 'self'` permits;
//   * colours come from CSS tokens in roomsim.css, read here, so a list
//     swatch and its box in the view cannot disagree;
//   * render on demand, not in a loop: a static room has nothing to
//     animate, and a phone should not burn battery redrawing it.
//
// Frame: the scene's own. Metres, right-handed, +z up, origin at a room
// corner, so a scene coordinate is used as-is with no axis swap.

import * as THREE from './vendor/0.186.1-saans1/three.module.js';
import { OrbitControls } from './vendor/0.186.1-saans1/OrbitControls.js';
// Named by the digest of its bytes, because it is cached as immutable;
// src/roomsim.rs pins the two together.
import { createSim } from './sim/32c8ff29371a/roomsim-sim.js';

// ---------------------------------------------------------------- helpers

const $ = (id) => document.getElementById(id);

/** Build an element. Text goes through textContent only. */
function el(tag, props = {}, ...kids) {
  const e = document.createElement(tag);
  for (const [k, v] of Object.entries(props)) {
    if (v == null) continue;
    if (k === 'class') e.className = v;
    else if (k === 'text') e.textContent = v;
    else e.setAttribute(k, String(v));
  }
  for (const c of kids) if (c != null) e.append(c);
  return e;
}

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const colour = (name) => new THREE.Color(css(name));

const fm = (x) => `${Number(x).toFixed(2)} m`;
const fsize = (s) => `${s[0].toFixed(2)} × ${s[1].toFixed(2)} × ${s[2].toFixed(2)} m`;
/** The same, compact, for a label in the 3-D view where width is scarce. */
const fsizeTight = (s) => `${s[0].toFixed(2)}×${s[1].toFixed(2)}×${s[2].toFixed(2)} m`;
const fpt = (p) => `(${p.map((v) => v.toFixed(2)).join(', ')}) m`;
const finite = (...xs) => xs.every((x) => typeof x === 'number' && Number.isFinite(x));

const PROVENANCE = {
  surveyed: 'surveyed — measured in the real room',
  roomplan_import: 'RoomPlan import — geometry from a scan; materials and wall thickness are guesses',
  hand_written: 'hand-written — an illustration, not a measurement of a real room',
  generated: 'generated — drawn at random for training data (SIM), not a real room',
};

/**
 * Thickness an implied wall is DRAWN with, so it is visible. It is never
 * shown as a measurement: a wall the scene does not list has no declared
 * thickness, and the panel says so. Mirrors saans-scene's
 * DEFAULT_WALL_THICKNESS_M, the value an import assumes.
 */
const IMPLIED_WALL_DRAW_THICKNESS_M = 0.1;
const NOT_DECLARED = 'not declared';

function footprint(sc) {
  const r = sc.room;
  return r.polygon && r.polygon.length >= 3
    ? r.polygon
    : [[0, 0], [r.w_m, 0], [r.w_m, r.d_m], [0, r.d_m]];
}

/** Shoelace area, m². */
function area(poly) {
  let a = 0;
  for (let i = 0; i < poly.length; i++) {
    const [x0, y0] = poly[i];
    const [x1, y1] = poly[(i + 1) % poly.length];
    a += x0 * y1 - x1 * y0;
  }
  return Math.abs(a) / 2;
}

// ---------------------------------------------------------------- state

const viewport = $('viewport');
const labelLayer = $('labels');
const notice = $('notice');
const simOverlay = $('sim-overlay');
const ui = {
  picker: $('scene-picker'),
  view: $('view-toggle'),
  dims: $('dims-toggle'),
  reset: $('reset-view'),
  key: $('overlay-key'),
  info: $('scene-info'),
  selEmpty: $('sel-empty'),
  selInfo: $('sel-info'),
  list: $('object-list'),
};

const state = {
  scenes: [],       // [{id, name, provenance, sha256}] from /roomsim/scenes
  current: null,    // the scene JSON being drawn
  entry: null,      // its list entry
  requested: null,  // the scene id the URL asks for, even if not served
  topDown: false,
  dims: true,
  selected: null,   // a pickable Object3D
};

/**
 * One overlay, several reasons to show it. Every active reason is kept and
 * the most recent is shown; each is cleared only by the event that
 * resolves it. So a later good load clears a failed one, and nothing but
 * WebGL itself can clear "WebGL is not available".
 */
const notices = new Map(); // reason -> text, in the order raised
function paintNotice() {
  const last = [...notices].pop();
  notice.classList.toggle('hidden', !last);
  if (last) {
    notice.dataset.reason = last[0];
    notice.textContent = last[1];
  } else {
    delete notice.dataset.reason;
  }
}
function showNotice(text, reason) {
  notices.delete(reason);
  notices.set(reason, text);
  paintNotice();
}
function clearNotice(...reasons) {
  for (const r of reasons) notices.delete(r);
  paintNotice();
}

// ---------------------------------------------------------------- three.js

let renderer = null;
try {
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setClearColor(0x000000, 0);
  viewport.prepend(renderer.domElement);
  const c = renderer.domElement;
  c.setAttribute('role', 'img');
  c.setAttribute('aria-describedby', 'view-help');
  c.tabIndex = 0; // keyboard: arrows pan, Shift+arrows rotate, +/- zoom, 0 resets
  // A lost context is restored by the browser; draw again when it is.
  c.addEventListener('webglcontextlost', (e) => e.preventDefault());
  c.addEventListener('webglcontextrestored', () => render());
} catch (err) {
  console.warn('WebGL unavailable:', err);
  renderer = null;
  showNotice('WebGL is not available in this browser, so the 3-D view cannot be drawn. ' +
    'The scene details and the list beside this message still describe the room, ' +
    'and the simulation view (SIM) lists every estimate there as text.', 'webgl');
}

const world = new THREE.Scene();
const hemi = new THREE.HemisphereLight(0xffffff, 0x404040, 1.6);
hemi.position.set(0, 0, 1); // +z is up in this frame
world.add(hemi);
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
sun.position.set(-4, -6, 10);
world.add(sun);

const persp = new THREE.PerspectiveCamera(45, 1, 0.05, 500);
persp.up.set(0, 0, 1);
const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.05, 500);
ortho.up.set(0, 1, 0); // plan view: +y points up the screen

let orbit = null;
let plan = null;
if (renderer) {
  // Each camera gets its own controls; only the active one listens.
  orbit = new OrbitControls(persp, renderer.domElement);
  orbit.enableDamping = false; // render on demand; nothing to ease
  orbit.screenSpacePanning = true;
  orbit.maxPolarAngle = Math.PI * 0.49; // stay above the floor
  orbit.listenToKeyEvents(renderer.domElement);
  orbit.addEventListener('change', render);
  plan = new OrbitControls(ortho, renderer.domElement);
  plan.enableRotate = false; // a plan is a plan: pan and zoom only
  plan.screenSpacePanning = true;
  // With rotation off, the primary drag must pan, or it does nothing.
  plan.mouseButtons = { LEFT: THREE.MOUSE.PAN, MIDDLE: THREE.MOUSE.DOLLY, RIGHT: THREE.MOUSE.PAN };
  plan.touches = { ONE: THREE.TOUCH.PAN, TWO: THREE.TOUCH.DOLLY_PAN };
  plan.listenToKeyEvents(renderer.domElement);
  plan.enabled = false;
  plan.addEventListener('change', render);
}

const camera = () => (state.topDown ? ortho : persp);
const controls = () => (state.topDown ? plan : orbit);

let roomGroup = null;
let dimGroup = null;
let floorMesh = null;
const pickables = [];
// {el, at: Vector3[], dim: bool, text(): string, kind: 'dim'|'node'|'object', obj: Object3D|null}
const labels = [];

function disposeTree(obj) {
  obj.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) [].concat(o.material).forEach((m) => m.dispose());
  });
}

function clearRoom() {
  for (const g of [roomGroup, dimGroup]) {
    if (g) {
      world.remove(g);
      disposeTree(g);
    }
  }
  roomGroup = dimGroup = floorMesh = null;
  pickables.length = 0;
  // Only the room's labels: the simulation layer clears its own.
  for (const L of labels) L.el.remove();
  labels.length = 0;
  state.selected = null;
}

/**
 * Label priority when two would overlap: higher wins, then nearer. A
 * simulated estimate's label (`est`) outranks everything but the
 * selection; truth and error labels come next. None of the three is ever
 * hidden: a ring without its radius, or a cross or segment without its
 * name, would read as a point estimate (ADR-008 d5).
 */
const PRIORITY = { est: 8, truth: 7, err: 7, dim: 3, node: 2, object: 1 };

/**
 * `at` is one anchor, or a list of candidate anchors tried in order when
 * the first would collide. A dimension's label may sit anywhere along its
 * own dimension line without becoming ambiguous, so dimensions get
 * several; an object's or a node's label belongs over that thing only.
 */
function addLabel(at, cls, text, { dim = false, kind = 'object', obj = null } = {}) {
  const e = el('div', { class: `label ${cls}` });
  labelLayer.append(e);
  const L = { el: e, at: Array.isArray(at) ? at : [at], dim, text, kind, obj };
  e.textContent = text();
  labels.push(L);
  return L;
}

function edges(mesh, colourName, opacity = 0.9) {
  const line = new THREE.LineSegments(
    new THREE.EdgesGeometry(mesh.geometry),
    new THREE.LineBasicMaterial({ color: colour(colourName), transparent: opacity < 1, opacity }),
  );
  mesh.add(line);
  return line;
}

/**
 * The walls to draw. A scene may omit `walls`, in which case its footprint
 * implies them; those carry no thickness and no material, because the
 * scene declares neither.
 */
function wallsOf(sc) {
  if (sc.walls && sc.walls.length) return sc.walls.map((w) => ({ ...w, implied: false }));
  const fp = footprint(sc);
  return fp.map((p, i) => ({ from: p, to: fp[(i + 1) % fp.length], material: null, thickness_m: null, implied: true }));
}

function buildRoom(sc) {
  clearRoom();
  const r = sc.room;
  roomGroup = new THREE.Group();
  dimGroup = new THREE.Group();
  const fp = footprint(sc);

  // Floor: the footprint polygon at z = 0, so an L-shaped hall is an L.
  const shape = new THREE.Shape(fp.map(([x, y]) => new THREE.Vector2(x, y)));
  floorMesh = new THREE.Mesh(
    new THREE.ShapeGeometry(shape),
    new THREE.MeshStandardMaterial({ color: colour('--k-floor'), roughness: 1, metalness: 0 }),
  );
  floorMesh.userData = { kind: 'floor' };
  roomGroup.add(floorMesh);
  const outline = new THREE.LineLoop(
    new THREE.BufferGeometry().setFromPoints(fp.map(([x, y]) => new THREE.Vector3(x, y, 0.002))),
    new THREE.LineBasicMaterial({ color: colour('--k-wall') }),
  );
  roomGroup.add(outline);

  // Walls: semi-transparent boxes centred on their segment.
  wallsOf(sc).forEach((w, i) => {
    const [x1, y1] = w.from;
    const [x2, y2] = w.to;
    const len = Math.hypot(x2 - x1, y2 - y1);
    const h = w.height_m ?? r.h_m;
    const t = w.thickness_m ?? IMPLIED_WALL_DRAW_THICKNESS_M;
    if (!finite(len, h, t) || len <= 0) return;
    const glass = w.material === 'glass';
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(len, t, h),
      new THREE.MeshStandardMaterial({
        color: colour('--k-wall'), transparent: true, opacity: glass ? 0.08 : 0.16,
        depthWrite: false, side: THREE.DoubleSide, roughness: 0.9,
      }),
    );
    mesh.position.set((x1 + x2) / 2, (y1 + y2) / 2, h / 2);
    mesh.rotation.z = Math.atan2(y2 - y1, x2 - x1);
    mesh.userData = { kind: 'wall', index: i, data: { ...w, length_m: len, height_m: h } };
    mesh.userData.edges = edges(mesh, '--k-wall', 0.45);
    roomGroup.add(mesh);
    pickables.push(mesh);
  });

  // Objects: upright boxes, centre and size exactly as declared.
  (sc.objects || []).forEach((o, i) => {
    if (!finite(...o.pos, ...o.size, o.yaw_deg ?? 0)) return;
    const see = o.type === 'window' || o.type === 'door';
    const mesh = new THREE.Mesh(
      new THREE.BoxGeometry(o.size[0], o.size[1], o.size[2]),
      new THREE.MeshStandardMaterial({
        color: colour(`--k-${o.type}`), roughness: 0.75, metalness: o.material === 'metal' ? 0.5 : 0,
        transparent: see, opacity: o.type === 'window' ? 0.45 : see ? 0.85 : 1,
      }),
    );
    mesh.position.set(o.pos[0], o.pos[1], o.pos[2]);
    mesh.rotation.z = THREE.MathUtils.degToRad(o.yaw_deg ?? 0);
    mesh.userData = { kind: 'object', index: i, data: o };
    mesh.userData.edges = edges(mesh, `--k-${o.type}`);
    roomGroup.add(mesh);
    pickables.push(mesh);
    const name = o.id || o.type;
    addLabel(new THREE.Vector3(o.pos[0], o.pos[1], o.pos[2] + o.size[2] / 2 + 0.04), '',
      () => (state.dims ? `${name}\n${fsizeTight(o.size)}` : name), { kind: 'object', obj: mesh });
  });

  // Nodes: a marker at the antenna and a stalk to the floor, because a
  // node's height matters to every link it is on.
  (sc.nodes || []).forEach((n, i) => {
    if (!finite(...n.pos)) return;
    const [x, y, z] = n.pos;
    const mesh = new THREE.Mesh(
      new THREE.OctahedronGeometry(0.09),
      new THREE.MeshStandardMaterial({ color: colour('--k-node'), emissive: colour('--k-node'), emissiveIntensity: 0.35 }),
    );
    mesh.position.set(x, y, z);
    mesh.userData = { kind: 'node', index: i, data: n };
    roomGroup.add(mesh);
    pickables.push(mesh);
    const stalk = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(x, y, 0), new THREE.Vector3(x, y, z)]),
      new THREE.LineDashedMaterial({ color: colour('--k-node'), dashSize: 0.06, gapSize: 0.05 }),
    );
    stalk.computeLineDistances();
    roomGroup.add(stalk);
    addLabel(new THREE.Vector3(x, y, z + 0.12), 'node',
      () => (state.dims ? `N${n.id}\nh ${fm(z)}` : `N${n.id}`), { kind: 'node', obj: mesh });
  });

  buildDimensions(sc, fp);
  world.add(roomGroup);
  world.add(dimGroup);
  dimGroup.visible = state.dims;
}

/** Where along a dimension line its label may sit, in order of preference. */
const ALONG = [0.5, 0.35, 0.65, 0.2, 0.8];

/** How far dimension lines sit outside the room, metres. Shared with the
 *  camera framing, so the overlay is never cut off by the view. */
const dimOffset = (r) => Math.max(r.w_m, r.d_m) * 0.05 + 0.15;

/** The dimension overlay: room L/B (or every edge of an L-shape) and H. */
function buildDimensions(sc, fp) {
  const r = sc.room;
  const mat = new THREE.LineBasicMaterial({ color: colour('--accent') });
  const seg = (a, b) => {
    dimGroup.add(new THREE.Line(new THREE.BufferGeometry().setFromPoints([a, b]), mat));
  };
  const off = dimOffset(r);
  const z = 0.01;

  // Outward normal of an edge, for a CCW polygon. Signed area tells us
  // the winding, so a clockwise outline still gets outward offsets.
  let s = 0;
  for (let i = 0; i < fp.length; i++) {
    const [x0, y0] = fp[i];
    const [x1, y1] = fp[(i + 1) % fp.length];
    s += x0 * y1 - x1 * y0;
  }
  const ccw = s > 0;

  const edgeDims = r.polygon && r.polygon.length >= 3
    ? fp.map((p, i) => [p, fp[(i + 1) % fp.length], null])
    : [[[0, 0], [r.w_m, 0], 'L'], [[0, r.d_m], [0, 0], 'B']];

  for (const [[x0, y0], [x1, y1], tag] of edgeDims) {
    const len = Math.hypot(x1 - x0, y1 - y0);
    if (len <= 1e-6) continue;
    let nx = (y1 - y0) / len;
    let ny = -(x1 - x0) / len;
    if (!ccw) { nx = -nx; ny = -ny; }
    const a = new THREE.Vector3(x0 + nx * off, y0 + ny * off, z);
    const b = new THREE.Vector3(x1 + nx * off, y1 + ny * off, z);
    seg(a, b);
    // Extension lines back to the corners, so the measure reads as a measure.
    seg(new THREE.Vector3(x0, y0, z), new THREE.Vector3(x0 + nx * off * 1.25, y0 + ny * off * 1.25, z));
    seg(new THREE.Vector3(x1, y1, z), new THREE.Vector3(x1 + nx * off * 1.25, y1 + ny * off * 1.25, z));
    const text = tag ? `${tag} ${fm(len)}` : fm(len);
    addLabel(ALONG.map((t) => a.clone().lerp(b, t)), 'dim', () => text, { dim: true, kind: 'dim' });
  }

  // Height at the corner farthest along +x and -y: away from the corner
  // where L and B meet, which is where the default view puts its eye.
  const [hx, hy] = fp.reduce((best, p) => (p[0] - p[1] > best[0] - best[1] ? p : best), fp[0]);
  seg(new THREE.Vector3(hx, hy, 0), new THREE.Vector3(hx, hy, r.h_m));
  addLabel(ALONG.map((t) => new THREE.Vector3(hx, hy, r.h_m * t)), 'dim', () => `H ${fm(r.h_m)}`, { dim: true, kind: 'dim' });
}

// ---------------------------------------------------------------- cameras

/**
 * What the plan view was last fitted to ({w, d, cx, cy}), and the plan
 * camera as that fit left it ({x, y, zoom}): while the camera is still
 * there, the viewer has not moved the view, and the fit may grow to take
 * in a new ring (simDrawn).
 */
let fitted = null;
let planAuto = null;

/** Fit the canvas and both projections to the viewport. */
function frame() {
  if (!renderer) return;
  const w = viewport.clientWidth || 1;
  const h = viewport.clientHeight || 1;
  // Re-read the pixel ratio: browser zoom and a move to another monitor
  // change it after load.
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(w, h, true);
  const aspect = w / h;
  persp.aspect = aspect;
  persp.updateProjectionMatrix();
  const sc = state.current;
  if (!sc) return;
  const r = sc.room;
  const box = fitted || fitBox(sc);
  // Plan view: fit the footprint plus the dimension lines and their
  // labels. A proportional margin alone clips a small room's overlay.
  const pad = 2 * (dimOffset(r) * 1.25 + 0.2);
  let vw = Math.max(box.w * 1.35, box.w + pad);
  let vh = Math.max(box.d * 1.35, box.d + pad);
  if (vw / vh < aspect) vw = vh * aspect; else vh = vw / aspect;
  ortho.left = -vw / 2; ortho.right = vw / 2; ortho.top = vh / 2; ortho.bottom = -vh / 2;
  ortho.updateProjectionMatrix();
}

/**
 * What the views fit: the room's box, and while the simulation view is
 * on, every ring it has drawn too (spec/livedemo.md §9.3), which may
 * reach far outside the room.
 */
function fitBox(sc) {
  const r = sc.room;
  let [x0, y0, x1, y1] = [0, 0, r.w_m, r.d_m];
  const b = sim.bounds();
  if (b) {
    x0 = Math.min(x0, b[0]); y0 = Math.min(y0, b[1]);
    x1 = Math.max(x1, b[2]); y1 = Math.max(y1, b[3]);
  }
  return { w: x1 - x0, d: y1 - y0, cx: (x0 + x1) / 2, cy: (y0 + y1) / 2 };
}

/** Centre the plan view on a box, at zoom 1, and size it to fit. */
function fitPlan(box) {
  fitted = box;
  frame();
  ortho.position.set(box.cx, box.cy, state.current.room.h_m + 20);
  ortho.zoom = 1;
  ortho.updateProjectionMatrix();
  plan.target.set(box.cx, box.cy, 0);
  ortho.lookAt(box.cx, box.cy, 0);
  plan.update();
  planAuto = { x: ortho.position.x, y: ortho.position.y, zoom: ortho.zoom };
}

/** Let the plan view follow the rings again, from wherever it is now. */
function armPlanFit() {
  if (renderer) planAuto = { x: ortho.position.x, y: ortho.position.y, zoom: ortho.zoom };
}

/**
 * After every simulation frame. While the simulation view is on, the
 * plan view fits the footprint and every drawn ring (spec/livedemo.md
 * §9.3), and rings arrive after the room is framed: so it grows to take
 * in a ring that reaches past it. Grow-only, so the zoom never pumps from
 * frame to frame, and only while the viewer has not panned or zoomed it;
 * Reset view and entering the plan view fit afresh.
 */
function simDrawn() {
  const b = sim.bounds();
  if (renderer && state.current && state.topDown && b && fitted && planAuto
      && ortho.zoom === planAuto.zoom && ortho.position.x === planAuto.x && ortho.position.y === planAuto.y) {
    const f = [fitted.cx - fitted.w / 2, fitted.cy - fitted.d / 2, fitted.cx + fitted.w / 2, fitted.cy + fitted.d / 2];
    if (b[0] < f[0] || b[1] < f[1] || b[2] > f[2] || b[3] > f[3]) {
      const u = [Math.min(f[0], b[0]), Math.min(f[1], b[1]), Math.max(f[2], b[2]), Math.max(f[3], b[3])];
      fitPlan({ w: u[2] - u[0], d: u[3] - u[1], cx: (u[0] + u[2]) / 2, cy: (u[1] + u[3]) / 2 });
    }
  }
  // For automated checks: in the plan view, does every drawn ring fit in it?
  let fit = '';
  if (renderer && state.topDown && b) {
    const z = ortho.zoom;
    const inView = b[0] >= ortho.position.x + ortho.left / z && b[2] <= ortho.position.x + ortho.right / z
      && b[1] >= ortho.position.y + ortho.bottom / z && b[3] <= ortho.position.y + ortho.top / z;
    fit = inView ? 'all' : 'clipped';
  }
  document.body.dataset.simPlanFit = fit;
}

function resetView() {
  const sc = state.current;
  if (!sc || !renderer) return;
  const r = sc.room;
  const box = fitBox(sc);
  fitPlan(box); // sizes both projections first
  const cx = box.cx;
  const cy = box.cy;
  // A three-quarter view from the south-west, far enough back to fit the
  // room and its dimension lines in the narrower of the two fields of
  // view. The target sits a quarter of the way up the room, so the top
  // corners are 0.75 h above it.
  const m = 2 * dimOffset(r) * 1.3;
  const radius = Math.hypot((box.w + m) / 2, (box.d + m) / 2, r.h_m * 0.75);
  const vfov = THREE.MathUtils.degToRad(persp.fov);
  const hfov = 2 * Math.atan(Math.tan(vfov / 2) * persp.aspect);
  const dist = (radius / Math.sin(Math.min(vfov, hfov) / 2)) * 1.08;
  const dir = new THREE.Vector3(-0.55, -0.85, 0.75).normalize();
  orbit.target.set(cx, cy, r.h_m * 0.25);
  persp.position.copy(orbit.target).addScaledVector(dir, dist);
  orbit.update();
}

/** Zoom by a factor from the keyboard: > 1 zooms in. */
function zoomBy(f) {
  if (!renderer || !state.current) return;
  if (state.topDown) {
    ortho.zoom = Math.min(Math.max(ortho.zoom * f, 0.2), 20);
    ortho.updateProjectionMatrix();
    plan.update();
  } else {
    const off = persp.position.clone().sub(orbit.target).multiplyScalar(1 / f);
    persp.position.copy(orbit.target).add(off);
    orbit.update();
  }
  render();
}

// ---------------------------------------------------------------- render

const _v = new THREE.Vector3();

/** How much two rectangles ({l, t, r, b}) overlap, px²; touching is 0. */
const overlap = (a, b) => Math.max(0, Math.min(a.r, b.r) - Math.max(a.l, b.l))
  * Math.max(0, Math.min(a.b, b.b) - Math.max(a.t, b.t));

/**
 * The boxes over the view's corners (the dimension key, the simulation's
 * status), in label-layer pixels. They paint over the labels, so while
 * the simulation view is on no label is placed under them: one there
 * would be hidden by CSS, which an estimate's radius never may be (d5).
 */
function cornerBoxes() {
  if (!sim.isOn()) return [];
  const v = viewport.getBoundingClientRect();
  return [ui.key, simOverlay].filter((e) => e.getClientRects().length).map((e) => {
    const r = e.getBoundingClientRect();
    return { l: r.left - v.left, t: r.top - v.top, r: r.right - v.left, b: r.bottom - v.top };
  });
}

/** The screen box of a simulation mark (a truth cross), or null if behind the camera. */
function markBox(m, cam, w, h) {
  const box = { l: Infinity, t: Infinity, r: -Infinity, b: -Infinity };
  for (const [dx, dy] of [[-1, -1], [1, -1], [1, 1], [-1, 1]]) {
    _v.set(m.at.x + dx * m.half, m.at.y + dy * m.half, m.at.z).project(cam);
    if (_v.z < -1 || _v.z > 1) return null;
    const x = (_v.x * 0.5 + 0.5) * w;
    const y = (-_v.y * 0.5 + 0.5) * h;
    box.l = Math.min(box.l, x); box.r = Math.max(box.r, x);
    box.t = Math.min(box.t, y); box.b = Math.max(box.b, y);
  }
  return box;
}

/**
 * Position every label over its 3-D anchor, then declutter: labels are
 * placed in priority order (the selection, then the simulation's
 * estimates, truth and errors, then room dimensions, nodes and objects;
 * nearer first within a rank) and any label that would overlap one
 * already placed is hidden (a dimension label first tries other spots
 * along its line, above and below it). Nothing hidden is lost — every
 * object, node and wall is in the list beside the view, and selecting
 * one puts its label on top. Labels are clamped inside the view so a node
 * near the edge is never cut in half.
 *
 * A label that must stay (`keep`: the selection's, and every simulation
 * label) is never hidden: it takes the least-covered spot it can find,
 * and never one under a corner box while any other exists. Labels keep
 * off a truth cross (`mark`) before they keep off each other, so a cross
 * is seen beside its name.
 */
function placeLabels() {
  const cam = camera();
  const w = viewport.clientWidth;
  const h = viewport.clientHeight;
  const boxes = cornerBoxes();
  const marks = [];
  const cands = [];
  for (const L of [...labels, ...sim.labels()]) {
    const sel = state.selected != null && L.obj === state.selected;
    L.el.classList.toggle('selected', sel);
    if (L.dim && !state.dims) {
      L.el.classList.add('hidden');
      L.el.dataset.why = 'dims-off';
      continue;
    }
    const pts = [];
    for (const a of L.at) {
      _v.copy(a).project(cam);
      const on = _v.z >= -1 && _v.z <= 1 && _v.x >= -1.1 && _v.x <= 1.1 && _v.y >= -1.1 && _v.y <= 1.1;
      // An estimate's label is clamped into the view even when its anchor
      // is not on screen (its ring may be); its on-screen anchors first.
      if (on || L.clamp) pts.push({ x: (_v.x * 0.5 + 0.5) * w, y: (-_v.y * 0.5 + 0.5) * h, z: _v.z, on });
    }
    if (L.clamp) pts.sort((a, b) => b.on - a.on);
    if (!pts.length) {
      L.el.classList.add('hidden');
      L.el.dataset.why = 'off-screen';
      continue;
    }
    // A mark is kept clear of every label; its own sits just outside it.
    let gap = 4;
    const mb = L.mark && markBox(L.mark, cam, w, h);
    if (mb) {
      marks.push(mb);
      const p = pts[0];
      gap = Math.max(p.x - mb.l, mb.r - p.x, p.y - mb.t, mb.b - p.y) + 3;
    }
    cands.push({ L, pts, gap, keep: !!L.keep || sel, z: pts[0].z, pri: sel ? 9 : PRIORITY[L.kind] });
  }
  cands.sort((a, b) => b.pri - a.pri || a.z - b.z);
  const placed = [];
  const pad = 2;
  for (const c of cands) {
    const e = c.L.el;
    e.classList.remove('hidden');
    // Fractional sizes: a rounded one lets a label reach a pixel under a box.
    const { width: bw, height: bh } = e.getBoundingClientRect();
    const g = c.gap;
    // An object's or node's label sits above it. A dimension's may also
    // sit just below its line: it still reads as that line's measure. A
    // simulation label tries above, below, right and left of its anchor,
    // then a row further out, up to three. An estimate's then tries just
    // clear of each corner box (its ring is large and still names it); a
    // truth or error label stays by its mark, or it would name nothing.
    const spots = c.L.keep
      ? (p) => [0, 1, 2, 3].flatMap((k) => [
        [p.x - bw / 2, p.y - g - bh - k * (bh + 4)], [p.x - bw / 2, p.y + g + k * (bh + 4)],
        [p.x + g, p.y - bh / 2 + k * (bh + 4)], [p.x - g - bw, p.y - bh / 2 + k * (bh + 4)]])
      : c.L.dim ? (p) => [[p.x - bw / 2, p.y - bh - 2], [p.x - bw / 2, p.y + 2]] : (p) => [[p.x - bw / 2, p.y - bh - 2]];
    const tries = c.pts.flatMap(spots);
    if (c.L.clamp) for (const q of boxes) tries.push([c.pts[0].x - bw / 2, q.b + 2], [c.pts[0].x - bw / 2, q.t - bh - 2]);
    let spot = null;
    let best = null;
    let bestCost = Infinity;
    for (const [x, y] of tries) {
      const left = Math.min(Math.max(x, pad), Math.max(pad, w - bw - pad));
      const top = Math.min(Math.max(y, pad), Math.max(pad, h - bh - pad));
      const r = { l: left, t: top, r: left + bw, b: top + bh };
      const under = boxes.reduce((s, q) => s + overlap(r, q), 0);
      const onMark = marks.reduce((s, q) => s + overlap(r, q), 0);
      const over = placed.reduce((s, q) => s + overlap(r, q), 0);
      if (!under && !onMark && !over) {
        spot = r;
        break;
      }
      // Never under a corner box if any spot is not; then rather over
      // another label's edge than over a small cross, which it would hide.
      const cost = under * 1e6 + onMark * 10 + over;
      if (cost < bestCost) {
        best = r;
        bestCost = cost;
      }
    }
    if (!spot && c.keep) spot = best;
    if (!spot) {
      e.classList.add('hidden');
      e.dataset.why = 'overlap';
      continue;
    }
    e.dataset.why = 'shown';
    placed.push(spot);
    e.style.transform = `translate(${spot.l.toFixed(1)}px, ${spot.t.toFixed(1)}px)`;
  }
}

let queued = false;
function render() {
  if (!renderer || queued) return;
  queued = true;
  requestAnimationFrame(() => {
    queued = false;
    renderer.render(world, camera());
    placeLabels();
  });
}

new ResizeObserver(() => {
  frame();
  render();
}).observe(viewport);

// A pixel-ratio change without a resize (a window dragged to a HiDPI
// monitor) only shows up as a resolution media query flipping.
function watchPixelRatio() {
  const mq = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`);
  mq.addEventListener('change', () => { frame(); render(); watchPixelRatio(); }, { once: true });
}
watchPixelRatio();

// ---------------------------------------------------------------- simulation

/** The view the simulation layer draws into; it imports nothing itself. */
const sim = createSim({
  THREE,
  el,
  colour,
  world,
  labelLayer,
  render: () => render(),
  gl: () => renderer !== null,
  scene: () => ({ sc: state.current, entry: state.entry }),
  footprint,
  provenanceText: (p) => PROVENANCE[p] || String(p),
  writeHash: () => writeHash(),
  textChanged: () => simTextChanged(),
  drawn: () => simDrawn(),
  simStarted: () => armPlanFit(),
});

// ---------------------------------------------------------------- selection

function describe(obj) {
  const { kind, data } = obj.userData;
  if (kind === 'object') {
    return [
      ['Type', data.type],
      ['Name', data.id || '—'],
      ['Size (L × B × H)', fsize(data.size)],
      ['Material', data.material],
      ['Centre', fpt(data.pos)],
      ['Rotation', `${(data.yaw_deg ?? 0).toFixed(1)}° about +z`],
    ];
  }
  if (kind === 'node') {
    return [
      ['Type', 'ESP32 node'],
      ['Node id', `N${data.id}`],
      ['Position', fpt(data.pos)],
      ['Height above floor', fm(data.pos[2])],
    ];
  }
  return [
    ['Type', data.implied ? 'Wall (implied by the footprint)' : 'Wall'],
    ['Material', data.material ?? NOT_DECLARED],
    ['Length', fm(data.length_m)],
    ['Thickness', data.thickness_m == null
      ? `${NOT_DECLARED} (drawn ${fm(IMPLIED_WALL_DRAW_THICKNESS_M)} thick so it is visible)`
      : fm(data.thickness_m)],
    ['Height', fm(data.height_m)],
    ['From → to', `${fpt(data.from)} → ${fpt(data.to)}`],
  ];
}

function highlight(obj, on) {
  if (!obj) return;
  const m = obj.material;
  if (m && 'emissive' in m) {
    if (on) {
      obj.userData.prevEmissive = m.emissive.clone();
      obj.userData.prevIntensity = m.emissiveIntensity;
      m.emissive.copy(colour('--k-pick'));
      m.emissiveIntensity = 0.55;
    } else if (obj.userData.prevEmissive) {
      m.emissive.copy(obj.userData.prevEmissive);
      m.emissiveIntensity = obj.userData.prevIntensity;
    }
  }
  const e = obj.userData.edges;
  if (e) e.material.color.copy(colour(on ? '--k-pick' : obj.userData.kind === 'wall' ? '--k-wall' : `--k-${obj.userData.data.type}`));
}

const keyOf = (obj) => `${obj.userData.kind}:${obj.userData.index}`;
/**
 * How a selection is named in the URL: an object's id when exactly one
 * object has it (readable links like select=almirah), otherwise the
 * positional key, because ids are optional and need not be unique.
 */
function urlKey(obj) {
  if (!obj) return null;
  const id = obj.userData.kind === 'object' && obj.userData.data.id;
  const unique = id && pickables.filter((o) => o.userData.kind === 'object' && o.userData.data.id === id).length === 1;
  return unique ? id : keyOf(obj);
}
function fromUrlKey(key) {
  if (!key) return null;
  return pickables.find((o) => keyOf(o) === key)
    || (pickables.filter((o) => o.userData.kind === 'object' && o.userData.data.id === key).length === 1
      ? pickables.find((o) => o.userData.kind === 'object' && o.userData.data.id === key) : null);
}

/** `write: false` while a load is in progress: the URL is written once,
 *  by whoever finishes the navigation, not halfway through it. */
function select(obj, { write = true } = {}) {
  highlight(state.selected, false);
  state.selected = obj;
  highlight(obj, true);
  for (const b of ui.list.querySelectorAll('button')) {
    b.setAttribute('aria-current', String(obj != null && b.dataset.key === keyOf(obj)));
  }
  if (!obj) {
    ui.selInfo.hidden = true;
    ui.selEmpty.hidden = false;
  } else {
    ui.selInfo.replaceChildren(...describe(obj).flatMap(([k, v]) => [el('dt', { text: k }), el('dd', { text: v })]));
    ui.selInfo.hidden = false;
    ui.selEmpty.hidden = true;
  }
  if (write) writeHash();
  render();
}

const raycaster = new THREE.Raycaster();
const ndc = new THREE.Vector2();
let downAt = null;

if (renderer) {
  const canvas = renderer.domElement;
  canvas.addEventListener('pointerdown', (e) => { downAt = { x: e.clientX, y: e.clientY }; });
  canvas.addEventListener('pointerup', (e) => {
    if (!downAt) return;
    const moved = Math.hypot(e.clientX - downAt.x, e.clientY - downAt.y);
    downAt = null;
    if (moved > 6) return; // that was a drag, not a click
    // A label names one thing; clicking it selects that thing, not
    // whatever happens to be behind the label. Labels never overlap after
    // the declutter pass, so at most one can contain the point.
    for (const L of labels) {
      if (!L.obj || L.el.dataset.why !== 'shown') continue;
      const r = L.el.getBoundingClientRect();
      if (e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom) {
        select(L.obj);
        return;
      }
    }
    const rect = canvas.getBoundingClientRect();
    ndc.set(((e.clientX - rect.left) / rect.width) * 2 - 1, -((e.clientY - rect.top) / rect.height) * 2 + 1);
    raycaster.setFromCamera(ndc, camera());
    // In the 3-D view the floor is a target too, so clicking bare floor
    // clears the selection instead of picking the transparent near wall.
    const targets = state.topDown || !floorMesh ? pickables : [...pickables, floorMesh];
    const hits = raycaster.intersectObjects(targets, false);
    // Walls enclose everything, so a click on a sofa from outside the
    // room hits a wall first. Prefer anything that is not a wall.
    const hit = hits.find((h) => h.object.userData.kind !== 'wall') || hits[0];
    select(!hit || hit.object.userData.kind === 'floor' ? null : hit.object);
  });
  canvas.addEventListener('keydown', (e) => {
    // Ctrl/Cmd with + - 0 is the browser's page zoom; never take it over.
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    if (e.key === '+' || e.key === '=') zoomBy(1.2);
    else if (e.key === '-' || e.key === '_') zoomBy(1 / 1.2);
    else if (e.key === '0') { resetView(); render(); }
    else return;
    e.preventDefault();
  });
}
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') select(null); });

// ---------------------------------------------------------------- panel

/**
 * The texts that say what is drawn. The room is declared; people and
 * estimates appear only in the SIM view (spec/livedemo.md §9.5), so
 * none of these may say that nothing is inferred or no one is drawn.
 */
let peopleDd = null;
function peopleText() {
  return sim.isOn()
    ? 'simulated (SIM): estimates of simulated people are drawn, and with Truth on the simulated people too'
    : 'appear only in the SIM view (Run simulation): simulated, never sensed';
}
function viewLabel(sc) {
  const r = sc.room;
  return `${sc.name}: a ${r.w_m.toFixed(2)} by ${r.d_m.toFixed(2)} by ${r.h_m.toFixed(2)} metre room with ` +
    `${(sc.objects || []).length} objects and ${(sc.nodes || []).length} ESP32 nodes. The room is declared in a scene file. ` +
    (sim.isOn()
      ? 'Simulation view on: estimates from simulated WiFi (SIM) are drawn over it, and listed as text beside the view.'
      : 'People and estimates appear only in the SIM view.');
}
/** The simulation view was turned on or off. */
function simTextChanged() {
  if (peopleDd) peopleDd.textContent = peopleText();
  if (renderer && state.current) renderer.domElement.setAttribute('aria-label', viewLabel(state.current));
}

function kv(pairs) {
  return pairs.flatMap(([k, v, cls]) => [el('dt', { text: k }), el('dd', { text: v, class: cls })]);
}

function fillPanel(sc, entry) {
  const r = sc.room;
  const fp = footprint(sc);
  const ws = wallsOf(sc);
  const mats = {};
  for (const w of ws) mats[w.material ?? NOT_DECLARED] = (mats[w.material ?? NOT_DECLARED] || 0) + 1;
  const matText = Object.entries(mats).map(([m, n]) => `${m} ×${n}`).join(', ');
  const hash = entry ? entry.sha256 : '';
  ui.info.replaceChildren(...kv([
    ['Name', sc.name],
    ['Provenance', PROVENANCE[sc.provenance] || String(sc.provenance), `prov-${sc.provenance}`],
    ['L × B × H', `${r.w_m.toFixed(2)} × ${r.d_m.toFixed(2)} × ${r.h_m.toFixed(2)} m${r.polygon ? ' (bounding box)' : ''}`],
    ['Floor area', `${area(fp).toFixed(2)} m²`],
    ['Walls', `${ws.length}${ws.some((w) => w.implied) ? ' (implied by the footprint)' : ''} — ${matText}`],
    ['Objects', String((sc.objects || []).length)],
    ['ESP32 nodes', String((sc.nodes || []).length)],
  ]));
  peopleDd = el('dd', { text: peopleText() });
  ui.info.append(el('dt', { text: 'People' }), peopleDd);
  const shaDd = el('dd', { class: 'mono', title: hash, text: hash ? `${hash.slice(0, 16)}…` : '—' });
  ui.info.append(el('dt', { text: 'Scene file sha256' }), shaDd);

  if (renderer) {
    renderer.domElement.setAttribute('aria-label', viewLabel(sc));
    // The list is the keyboard and screen-reader way to select anything.
    const items = [];
    for (const obj of pickables) {
      const { kind, data } = obj.userData;
      let label;
      let meta;
      let sw;
      if (kind === 'object') {
        label = data.id || data.type;
        meta = label === data.type ? fsize(data.size) : `${data.type} · ${fsize(data.size)}`;
        sw = `sw-${data.type}`;
      } else if (kind === 'node') {
        label = `N${data.id}`; meta = `ESP32 · h ${fm(data.pos[2])}`; sw = 'sw-node';
      } else {
        label = `Wall ${obj.userData.index + 1}`; meta = `${data.material ?? NOT_DECLARED} · ${fm(data.length_m)}`; sw = 'sw-wall';
      }
      const b = el('button', { type: 'button', 'aria-current': 'false' },
        el('span', {}, el('span', { class: `swatch ${sw}`, 'aria-hidden': 'true' }), label),
        el('span', { class: 'meta', text: meta }));
      b.dataset.key = keyOf(obj);
      b.addEventListener('click', () => select(obj));
      items.push(el('li', {}, b));
    }
    ui.list.replaceChildren(...items);
    return;
  }
  // Without WebGL nothing can be selected, so the list describes every
  // wall, object and node of the room as plain text instead.
  const lines = [
    ...ws.map((w, i) => {
      const len = Math.hypot(w.to[0] - w.from[0], w.to[1] - w.from[1]);
      const t = w.thickness_m == null ? `thickness ${NOT_DECLARED}` : `${fm(w.thickness_m)} thick`;
      return `Wall ${i + 1}: ${w.material ?? `material ${NOT_DECLARED}`}, ${fm(len)} long, ${t}, ${fpt(w.from)} → ${fpt(w.to)}`;
    }),
    ...(sc.objects || []).map((o) => `${o.id || o.type}: ${o.type}, ${fsize(o.size)}, ${o.material}, centre ${fpt(o.pos)}`),
    ...(sc.nodes || []).map((n) => `N${n.id}: ESP32 at ${fpt(n.pos)}, h ${fm(n.pos[2])}`),
  ];
  ui.list.replaceChildren(...lines.map((t) => el('li', { text: t })));
}

/** The view when the URL asks for a scene this gateway does not serve. */
function showMissing(id) {
  loadSeq++; // a load still in flight must not paint over this
  clearRoom();
  state.current = null;
  state.entry = null;
  ui.picker.value = '';
  ui.info.replaceChildren(...kv([['Scene', `none — "${id}" is not served by this gateway`]]));
  ui.list.replaceChildren();
  select(null, { write: false });
  showNotice(`No scene named "${id}" is served by this gateway. Choose one from the list above.`, 'scene');
  document.body.dataset.scene = '';
  sim.sceneChanged();
  render();
}

// ---------------------------------------------------------------- loading

function readHash() {
  const p = new URLSearchParams(location.hash.slice(1));
  return { scene: p.get('scene'), view: p.get('view'), dims: p.get('dims'), select: p.get('select') };
}

function hashFor({ scene, topDown, dims, select: sel }) {
  const p = new URLSearchParams();
  if (scene) p.set('scene', scene);
  if (topDown) p.set('view', 'top');
  if (dims === false) p.set('dims', 'off');
  if (sel) p.set('select', sel);
  // The simulation view's keys (sim, replay, and outside REPLAY variant
  // and seed) survive every rewrite. Never a position or a time: history
  // keeps URLs (d11).
  sim.writeParams(p);
  return `#${p}`;
}

/** Record view state in the URL without adding history entries. */
function writeHash() {
  const h = hashFor({
    scene: state.entry ? state.entry.id : state.requested,
    topDown: state.topDown, dims: state.dims, select: urlKey(state.selected),
  });
  if (h !== location.hash) history.replaceState(null, '', h);
}

async function getJSON(url) {
  const r = await fetch(url, { cache: 'no-store', credentials: 'same-origin' });
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return r.json();
}

let loadSeq = 0;

/**
 * Load and draw a served scene. Returns false if a newer request
 * overtook this one (its response is discarded, so a slow earlier pick
 * can never overwrite a later one) or if the load failed. Never writes
 * the URL: the caller does, once the navigation has succeeded.
 */
async function show(id) {
  const entry = state.scenes.find((s) => s.id === id);
  if (!entry) return false;
  const seq = ++loadSeq;
  let sc;
  try {
    sc = await getJSON(`/roomsim/scenes/${encodeURIComponent(entry.id)}`);
  } catch (err) {
    if (seq === loadSeq) {
      console.error(err);
      showNotice(`Could not load ${entry.name}: ${err.message}`, 'load');
      // Keep the picker naming what is actually drawn.
      ui.picker.value = state.entry ? state.entry.id : '';
    }
    return false;
  }
  if (seq !== loadSeq) return false;
  state.current = sc;
  state.entry = entry;
  state.requested = entry.id;
  ui.picker.value = entry.id;
  // A new room: the simulation layer closes its stream, clears, and
  // opens one for this scene if its view is on (spec/livedemo.md §9.2).
  // First, so the new room is framed without the old room's rings.
  sim.sceneChanged();
  if (renderer) {
    buildRoom(sc);
    resetView();
  }
  fillPanel(sc, entry);
  select(null, { write: false });
  clearNotice('load', 'scene');
  document.body.dataset.scene = entry.id; // which scene is drawn, for automated checks
  return true;
}

function setTopDown(on) {
  const entering = on && !state.topDown;
  state.topDown = on;
  ui.view.setAttribute('aria-pressed', String(on));
  if (orbit) {
    orbit.enabled = !on;
    plan.enabled = on;
  }
  // With the simulation view on, the plan view fits the room and every
  // drawn ring as it opens (spec/livedemo.md §9.3).
  if (entering && renderer && sim.isOn()) resetView();
  writeHash();
  render();
}

function setDims(on) {
  state.dims = on;
  ui.dims.setAttribute('aria-pressed', String(on));
  ui.key.classList.toggle('hidden', !on);
  if (dimGroup) dimGroup.visible = on;
  for (const L of labels) {
    L.el.textContent = L.text();
    // Hide or reveal now; the next frame only repositions.
    if (L.dim) L.el.classList.toggle('hidden', !on);
  }
  writeHash();
  render();
}

/**
 * Make the view match the URL. Runs at load and on every hashchange, so
 * Back, Forward and a hand-edited link all work. Picking a scene pushes a
 * history entry; everything else replaces the current one. replaceState
 * and pushState do not fire hashchange, so this cannot loop.
 */
async function applyHash() {
  const h = readHash();
  sim.readParams(new URLSearchParams(location.hash.slice(1)));
  state.dims = h.dims !== 'off';
  state.topDown = h.view === 'top';
  const want = h.scene || (state.entry ? state.entry.id : state.scenes[0].id);
  state.requested = want;
  if (!state.scenes.some((s) => s.id === want)) {
    showMissing(want);
    setDims(state.dims);
    setTopDown(state.topDown);
    return;
  }
  if (!state.entry || state.entry.id !== want) {
    if (!(await show(want))) {
      sim.sync();
      return;
    }
  } else {
    // Already drawn: any load still in flight is for a scene the viewer
    // has navigated away from, and any load error is about it too.
    loadSeq++;
    clearNotice('load', 'scene');
    sim.sync();
  }
  setDims(state.dims);
  setTopDown(state.topDown);
  select(fromUrlKey(h.select));
}

function fail(err) {
  console.error(err);
  showNotice(`Could not load the scene: ${err.message}`, 'load');
}

// A pick adds a history entry only once the scene has loaded, so a failed
// pick leaves the URL, the picker, the panel and the drawing agreeing.
ui.picker.addEventListener('change', () => {
  const id = ui.picker.value;
  show(id).then((ok) => {
    if (!ok) return;
    history.pushState(null, '', hashFor({ scene: id, topDown: state.topDown, dims: state.dims }));
    setDims(state.dims);
    setTopDown(state.topDown);
  }).catch(fail);
});
ui.view.addEventListener('click', () => setTopDown(!state.topDown));
ui.dims.addEventListener('click', () => setDims(!state.dims));
ui.reset.addEventListener('click', () => { resetView(); render(); });

// An OS theme change swaps every colour token. Rebuild the room from the
// new tokens, keeping the camera and the selection, so the 3-D view and
// the swatches in the list stay in the same theme.
matchMedia('(prefers-color-scheme: light)').addEventListener('change', () => {
  if (!renderer || !state.current) return;
  const key = state.selected && keyOf(state.selected);
  buildRoom(state.current);
  fillPanel(state.current, state.entry);
  sim.retheme();
  select(key ? pickables.find((o) => keyOf(o) === key) || null : null);
});

async function main() {
  const list = await getJSON('/roomsim/scenes');
  // This page draws SCENE data and nothing else. Any other label means
  // the server and the page disagree about what the numbers are.
  if (list.label !== 'SCENE') {
    showNotice(`Unexpected data label ${JSON.stringify(list.label)}; nothing is drawn.`, 'label');
    return;
  }
  state.scenes = list.scenes || [];
  if (!state.scenes.length) {
    showNotice('The gateway serves no scenes.', 'scene');
    return;
  }
  ui.picker.replaceChildren(
    el('option', { value: '', disabled: '', text: '— choose a scene —' }),
    ...state.scenes.map((s) => el('option', { value: s.id, text: s.name })),
  );
  if (!renderer) {
    ui.selEmpty.textContent = 'The 3-D view is unavailable in this browser, so nothing can be selected; '
      + 'every wall, object and node is listed below.';
    for (const b of [ui.view, ui.dims, ui.reset]) b.disabled = true;
  }
  // Listen first: a hash change during the first load must not be lost.
  window.addEventListener('hashchange', () => applyHash().catch(fail));
  await applyHash();
  document.body.dataset.ready = 'true'; // for automated checks of the page
}

main().catch(fail);
