// Saans /room-sim. A scene file, drawn.
//
// THE WHOLE POINT OF THIS FILE: everything it draws is an INPUT. The
// floor outline, the wall thicknesses, the furniture sizes and angles and
// the node heights are typed into `scenes/*.json` by a person. No radio
// measured any of them and no radio could: an ESP32 at 20 MHz on one
// antenna forms no coherent aperture, so there is no imaging here to
// recover a wall from and nothing to recognise a bed in. The page labels
// every number SCENE for that reason, and it shows no SIM badge and no
// LIVE badge, because at this point there is no simulator and no hardware
// behind the view.
//
// It also draws no people and no phone, although two of the files carry
// them. Nothing in the repository can produce a person's position yet,
// and a figure standing in the room would read as a measurement.
//
// This is an ES module (so it is strict, and it has its own scope). It
// imports three.js from this origin. Nothing is fetched from anywhere
// else, by design: the gateway may be running in a building with no route
// out, and a page that renders only when a CDN answers is a page that
// does not render.
//
// The camera is forty-odd lines at the bottom rather than three.js's
// OrbitControls, which is an addon and is not vendored. Drag to rotate,
// wheel or pinch to zoom, arrows and +/- from the keyboard. No pan in 3D;
// in the top-down view a drag pans, because there it is the only way to
// look at the far end of an 8 m hall up close.

import * as THREE from "./vendor/three.module.min.js";

const $ = (s, el = document) => el.querySelector(s);

// --------------------------------------------------------------- scenes

/** The three example rooms, in the order the picker shows them. */
const SCENES = [
  { key: "bedroom", nav: "bedroom", file: "bedroom.json", path: "scenes/bedroom.json" },
  { key: "office", nav: "office", file: "office.json", path: "scenes/office.json" },
  { key: "hall-l", nav: "L-shaped hall", file: "hall-l.json", path: "scenes/hall-l.json" },
];

/** The only scene format this page reads, same as the Rust loader. */
const SCHEMA_VERSION = 1;

/**
 * Colour per declared material. These say what the file declares, not
 * what the radio sees — nothing here is a propagation result. Metal is
 * the one that also happens to be true at this frequency: an almirah is
 * an opaque screen in the 2.4 GHz band, not a lossy dielectric.
 */
const MAT_COLOUR = {
  wood: 0x8d6f4e,
  metal: 0x9ba2a8,
  glass: 0x5d93ad,
  fabric: 0x7f8d77,
  concrete: 0x8d8d8a,
  brick: 0x9c6a56,
  drywall: 0xa8a49c,
};
const FALLBACK_COLOUR = 0x6f6b64;

const INK = 0xefebe2;
const FAINT = 0x8a857e;
const AMBER = 0xffb547;

// ------------------------------------------------------------ page state

const EL = {
  stage: $("#rs-stage"),
  pick: $("#rs-pick"),
  top: $("#rs-top"),
  dim: $("#rs-dim"),
  reset: $("#rs-reset"),
  name: $("#rs-name"),
  room: $("#rs-room"),
  list: $("#rs-list"),
  sel: $("#rs-sel"),
  err: $("#rs-err"),
  live: $("#rs-live"),
};

const STATE = {
  entry: SCENES[0],
  scene: null,
  /** Rising counter, so a slow fetch cannot overwrite a newer one. */
  token: 0,
  materials: null,
  materialsErr: "",
  topDown: false,
  dims: false,
  /** `{ kind: "object" | "node" | "wall", index }`, or null. */
  sel: null,
};

/** Everything three.js owns. Null when WebGL could not be started. */
let G = null;

// ----------------------------------------------------------- formatting

/**
 * Two decimals, or as many as the file actually used.
 *
 * Centimetres is the right precision for a room and it keeps a column
 * of numbers aligned, but a 0.115 m brick partition printed as "0.12 m"
 * is a number this page invented. So two decimals only while they are
 * lossless, and the exact value otherwise.
 */
function n2(v) {
  const fixed = Number(v).toFixed(2);
  return Number(fixed) === Number(v) ? fixed : String(v);
}

/**
 * Two decimals, always. For the handful of numbers this page works out
 * rather than reads: a floor area, a wall's length, the spread of the
 * node heights. They are arithmetic on SCENE values and carry binary
 * floating-point tails (4.2 x 3.6 is 15.120000000000001), so printing
 * them to their last digit would be precision this page does not have.
 */
const nc = (v) => Number(v).toFixed(2);
const size3 = (s) => `${n2(s[0])} × ${n2(s[1])} × ${n2(s[2])} m`;
const pos3 = (p) => `${n2(p[0])}, ${n2(p[1])}, ${n2(p[2])} m`;

/**
 * Writes only when the string changes.
 *
 * `#rs-err` is a role="status" and `#rs-live` is an aria-live region. A
 * reselection that lands on the same object, or a resize that re-runs the
 * same code path, would otherwise reassign an identical sentence and have
 * a screen reader announce it again. An unchanged fact is not news.
 */
function setText(node, s) {
  if (node && node.textContent !== s) node.textContent = s;
}

function el(tag, cls, text) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text != null) e.textContent = text;
  return e;
}

/**
 * One row of the panel's definition lists: term, value, and the tag that
 * says where the value came from. Every number this page prints goes
 * through here, so none of them can appear without a source beside it.
 */
function row(dl, term, value, src = "SCENE") {
  dl.append(el("dt", null, term));
  const dd = el("dd", null, value);
  const tag = el("span", "rs-src", src);
  dd.append(" ", tag);
  dl.append(dd);
}

// ------------------------------------------------- reading a scene file

/**
 * Parse and check a scene, or throw with the field that is wrong.
 *
 * Strict in the same places the Rust loader is strict, and for the same
 * reason: a room 4.2 m wide and NaN deep would otherwise load and then
 * poison every length drawn from it. What this does not do is re-derive
 * the Rust checks (furniture inside the room, nodes not coincident) —
 * those are the loader's business, and duplicating them here would give
 * the page a second opinion about what a valid scene is.
 */
function readScene(text) {
  let v;
  try {
    v = JSON.parse(text);
  } catch (e) {
    throw new Error("it is not valid JSON (" + e.message + ")");
  }
  if (!v || typeof v !== "object") throw new Error("the top level is not an object");
  if (v.schema_version !== SCHEMA_VERSION) {
    throw new Error("schema_version is " + JSON.stringify(v.schema_version) + ", and this page reads " + SCHEMA_VERSION);
  }
  const r = v.room;
  if (!r || typeof r !== "object") throw new Error("it has no room");
  for (const k of ["width_m", "depth_m", "height_m"]) {
    if (!(Number.isFinite(r[k]) && r[k] > 0)) throw new Error("room." + k + " is not a positive number");
  }
  const poly = Array.isArray(r.polygon_m) ? r.polygon_m : [];
  for (const [i, p] of poly.entries()) {
    if (!Array.isArray(p) || p.length !== 2 || !p.every(Number.isFinite)) {
      throw new Error("room.polygon_m[" + i + "] is not a pair of numbers");
    }
  }
  if (poly.length !== 0 && poly.length < 3) throw new Error("room.polygon_m has " + poly.length + " points");

  const arr = (x) => (Array.isArray(x) ? x : []);
  const walls = arr(v.walls);
  for (const [i, w] of walls.entries()) {
    const at = "walls[" + i + "]";
    for (const k of ["from_m", "to_m"]) {
      if (!Array.isArray(w[k]) || w[k].length !== 2 || !w[k].every(Number.isFinite)) {
        throw new Error(at + "." + k + " is not a pair of numbers");
      }
    }
    if (!(Number.isFinite(w.thickness_m) && w.thickness_m > 0)) throw new Error(at + ".thickness_m is not a positive number");
    if (typeof w.material !== "string") throw new Error(at + ".material is not a string");
  }
  const objects = arr(v.objects);
  for (const [i, o] of objects.entries()) {
    const at = "objects[" + i + "]";
    for (const k of ["centre_m", "size_m"]) {
      if (!Array.isArray(o[k]) || o[k].length !== 3 || !o[k].every(Number.isFinite)) {
        throw new Error(at + "." + k + " is not three numbers");
      }
    }
    if (!o.size_m.every((s) => s > 0)) throw new Error(at + ".size_m has a zero or negative side");
    if (o.rotation_deg != null && !Number.isFinite(o.rotation_deg)) throw new Error(at + ".rotation_deg is not a number");
    if (typeof o.type !== "string") throw new Error(at + ".type is not a string");
    if (typeof o.material !== "string") throw new Error(at + ".material is not a string");
  }
  const nodes = arr(v.nodes);
  for (const [i, nd] of nodes.entries()) {
    if (!Array.isArray(nd.pos_m) || nd.pos_m.length !== 3 || !nd.pos_m.every(Number.isFinite)) {
      throw new Error("nodes[" + i + "].pos_m is not three numbers");
    }
  }
  return {
    name: typeof v.name === "string" ? v.name : "",
    room: { width_m: r.width_m, depth_m: r.depth_m, height_m: r.height_m, polygon_m: poly },
    walls,
    objects,
    nodes,
  };
}

/** The floor outline: the polygon when there is one, else the box. */
function footprint(room) {
  if (room.polygon_m.length) return room.polygon_m;
  return [[0, 0], [room.width_m, 0], [room.width_m, room.depth_m], [0, room.depth_m]];
}

/** Shoelace. For an L-shape this is less than width x depth, which is
 *  the entire reason the L-shaped hall is one of the examples. */
function area(poly) {
  let t = 0;
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    t += a[0] * b[1] - b[0] * a[1];
  }
  return Math.abs(t / 2);
}

/** Ray casting. Used to decide which side of a wall is outside. */
function contains(poly, x, y) {
  let inside = false;
  for (let i = 0, j = poly.length - 1; i < poly.length; j = i++) {
    const [xi, yi] = poly[i];
    const [xj, yj] = poly[j];
    if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

// -------------------------------------------------------------- the view

/**
 * Scene coordinates are (x across, y deep, z up). three.js is y-up, so
 * the map is x -> x, y -> z, z -> y. It is written out once, here, and
 * every other line goes through it.
 */
const toWorld = (p) => new THREE.Vector3(p[0], p[2] == null ? 0 : p[2], p[1]);

/**
 * A plan-view rotation counter-clockwise in (x, y) is a rotation about
 * world Y by the NEGATIVE of the same angle, because world z runs the
 * way scene y does and the handedness flips. Getting this backwards
 * leaves a bed across the room instead of along it.
 */
const yawOf = (deg) => -THREE.MathUtils.degToRad(deg || 0);

function startThree() {
  let renderer;
  try {
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  } catch (e) {
    return null;
  }
  if (!renderer || !renderer.getContext()) return null;
  renderer.setPixelRatio(Math.min(devicePixelRatio || 1, 2));

  const root = new THREE.Scene();
  const content = new THREE.Group();
  root.add(content);
  root.add(new THREE.HemisphereLight(0xd8e2ea, 0x14140f, 2.1));
  const sun = new THREE.DirectionalLight(0xfff4e2, 1.9);
  sun.position.set(3, 8, 2);
  root.add(sun);

  const persp = new THREE.PerspectiveCamera(45, 1, 0.05, 300);
  const ortho = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.1, 200);
  ortho.up.set(0, 0, -1); // top-down: x to the right, scene y down the screen

  const labels = el("div", "rs-labels");
  labels.setAttribute("aria-hidden", "true");
  EL.stage.append(renderer.domElement, labels);

  return {
    renderer,
    root,
    content,
    persp,
    ortho,
    labels,
    /** Per-frame label bookkeeping: {el, at: Vector3, dimOnly}. */
    tags: [],
    pick: { objects: [], nodes: [], walls: [] },
    ray: new THREE.Raycaster(),
    /** Orbit state, in metres and radians. */
    cam: { target: new THREE.Vector3(), r: 8, theta: -0.95, phi: 0.95, half: 4, fit: 8, fitHalf: 4 },
    dirty: true,
    w: 1,
    h: 1,
  };
}

function camera() {
  return STATE.topDown ? G.ortho : G.persp;
}

function invalidate() {
  if (G) G.dirty = true;
}

function resize() {
  if (!G) return;
  const w = Math.max(1, EL.stage.clientWidth);
  const h = Math.max(1, EL.stage.clientHeight);
  G.w = w;
  G.h = h;
  G.renderer.setSize(w, h, false);
  G.persp.aspect = w / h;
  G.persp.updateProjectionMatrix();
  applyOrtho();
  invalidate();
}

function applyOrtho() {
  const aspect = G.w / G.h;
  const hh = G.cam.half;
  G.ortho.top = hh;
  G.ortho.bottom = -hh;
  G.ortho.right = hh * aspect;
  G.ortho.left = -hh * aspect;
  G.ortho.updateProjectionMatrix();
}

/**
 * Frame the whole room, in both cameras.
 *
 * The distance is measured rather than derived from the bounding
 * sphere: a box seen from one corner uses far less of the frame than
 * the sphere around it does, and fitting the sphere leaves a small room
 * floating in the middle of a large panel. So put the camera somewhere
 * plausible, project the eight corners, and divide by how much of the
 * frame they actually used. Three passes is enough to converge.
 */
function fitCamera() {
  if (!G || !STATE.scene) return;
  const room = STATE.scene.room;
  const poly = footprint(room);
  const box = poly.reduce(
    (acc, [x, y]) => [Math.min(acc[0], x), Math.min(acc[1], y), Math.max(acc[2], x), Math.max(acc[3], y)],
    [Infinity, Infinity, -Infinity, -Infinity],
  );
  const w = box[2] - box[0];
  const d = box[3] - box[1];
  const h = room.height_m;
  const c = G.cam;
  c.target.set((box[0] + box[2]) / 2, h / 2, (box[1] + box[3]) / 2);
  c.theta = -0.95;
  c.phi = 0.95;

  const corners = [];
  for (const x of [box[0], box[2]]) {
    for (const y of [0, h]) {
      for (const z of [box[1], box[3]]) corners.push(new THREE.Vector3(x, y, z));
    }
  }
  const vfov = THREE.MathUtils.degToRad(G.persp.fov);
  c.r = 0.5 * Math.hypot(w, d, h) / Math.sin(vfov / 2);
  for (let pass = 0; pass < 3; pass++) {
    placeCamera();
    let used = 0;
    for (const p of corners) {
      PROJ.copy(p).project(G.persp);
      used = Math.max(used, Math.abs(PROJ.x), Math.abs(PROJ.y));
    }
    if (!(used > 0)) break;
    c.r *= Math.min(Math.max(used, 0.3), 3);
  }
  c.r *= 1.06; // a little air around the walls
  c.fit = c.r;

  const aspect = Math.max(0.2, G.w / G.h);
  c.fitHalf = 1.08 * Math.max(d / 2, w / (2 * aspect));
  c.half = c.fitHalf;
  applyCamera();
}

/** Put both cameras where the orbit state says, with no clamping. */
function placeCamera() {
  const c = G.cam;
  const s = Math.sin(c.phi);
  G.persp.position.set(
    c.target.x + c.r * s * Math.sin(c.theta),
    c.target.y + c.r * Math.cos(c.phi),
    c.target.z + c.r * s * Math.cos(c.theta),
  );
  G.persp.lookAt(c.target);
  G.persp.updateMatrixWorld(true);

  G.ortho.position.set(c.target.x, 60, c.target.z);
  G.ortho.lookAt(c.target.x, 0, c.target.z);
  G.ortho.updateMatrixWorld(true);
  applyOrtho();
}

function applyCamera() {
  const c = G.cam;
  c.phi = Math.min(Math.max(c.phi, 0.08), Math.PI / 2 - 0.03);
  c.r = Math.min(Math.max(c.r, c.fit * 0.2), c.fit * 2.4);
  c.half = Math.min(Math.max(c.half, c.fitHalf * 0.15), c.fitHalf * 1.8);
  placeCamera();
  invalidate();
}

// -------------------------------------------------------------- building

function clearContent() {
  if (!G) return;
  G.content.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    const m = o.material;
    if (Array.isArray(m)) m.forEach((x) => x.dispose());
    else if (m) m.dispose();
  });
  G.content.clear();
  G.pick = { objects: [], nodes: [], walls: [] };
  for (const t of G.tags) t.el.remove();
  G.tags = [];
  G.dimGroup = null;
  invalidate();
}

/**
 * An HTML label over the canvas, rather than a texture in the scene.
 * Text that stays crisp at any zoom, in the site's own font, and legible
 * at 375 px — none of which a canvas sprite manages without work. `tag`
 * is the source marker; it is present whenever the label carries a
 * number, and absent when the label is only a word.
 */
function label(at, kind, head, detail, ref) {
  const e = el("div", "rs-tag rs-tag-" + kind);
  e.append(el("b", null, head));
  if (detail) {
    e.append(el("span", null, detail));
    e.append(el("i", null, "SCENE"));
  }
  G.labels.append(e);
  const t = { el: e, at, kind, ref: ref || null };
  G.tags.push(t);
  return t;
}

function lineMat(colour, opacity) {
  return new THREE.LineBasicMaterial({ color: colour, transparent: opacity < 1, opacity });
}

function segments(points, material) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  return new THREE.LineSegments(g, material);
}

function build() {
  clearContent();
  const sc = STATE.scene;
  if (!G || !sc) return;
  const room = sc.room;
  const poly = footprint(room);

  // ---- floor: the actual polygon, triangulated. Drawing the bounding
  // box here instead would claim 48 m2 of floor in the L-shaped hall
  // where there are 35.75.
  const shape = new THREE.Shape(poly.map(([x, y]) => new THREE.Vector2(x, y)));
  const floorGeo = new THREE.ShapeGeometry(shape);
  floorGeo.rotateX(Math.PI / 2); // (x, y, 0) -> (x, 0, y): scene plan onto the ground
  const floor = new THREE.Mesh(
    floorGeo,
    new THREE.MeshLambertMaterial({ color: 0x24241f, side: THREE.DoubleSide }),
  );
  floor.position.y = -0.002;
  G.content.add(floor);

  // The outline, bright, closed. The polygon is the thing to see.
  const edge = [];
  for (let i = 0; i < poly.length; i++) {
    const a = poly[i];
    const b = poly[(i + 1) % poly.length];
    edge.push(a[0], 0.004, a[1], b[0], 0.004, b[1]);
  }
  G.content.add(segments(edge, lineMat(INK, 0.85)));

  // A 1 m grid, clipped to the floor polygon, so a length can be read
  // off the picture and not only out of the panel.
  const grid = [];
  const box = poly.reduce(
    (acc, [x, y]) => [Math.min(acc[0], x), Math.min(acc[1], y), Math.max(acc[2], x), Math.max(acc[3], y)],
    [Infinity, Infinity, -Infinity, -Infinity],
  );
  for (let x = Math.ceil(box[0]); x <= box[2] - 1e-9; x += 1) {
    for (let y = box[1]; y < box[3] - 1e-9; y += 0.25) {
      if (contains(poly, x, y + 0.125)) grid.push(x, 0.001, y, x, 0.001, Math.min(y + 0.25, box[3]));
    }
  }
  for (let y = Math.ceil(box[1]); y <= box[3] - 1e-9; y += 1) {
    for (let x = box[0]; x < box[2] - 1e-9; x += 0.25) {
      if (contains(poly, x + 0.125, y)) grid.push(x, 0.001, y, Math.min(x + 0.25, box[2]), 0.001, y);
    }
  }
  G.content.add(segments(grid, lineMat(FAINT, 0.45)));

  // ---- walls. `from_m`/`to_m` is the CENTRELINE of the clear interior,
  // and thickness goes outwards from it: a wall drawn centred on that
  // line would eat half its thickness out of the floor. Which side is
  // "outwards" is decided by testing a point against the polygon, which
  // gets the L-shape's two inner walls right as well.
  for (const [i, w] of sc.walls.entries()) {
    const dx = w.to_m[0] - w.from_m[0];
    const dy = w.to_m[1] - w.from_m[1];
    const len = Math.hypot(dx, dy);
    if (!(len > 0)) continue;
    const t = w.thickness_m;
    const h = Number.isFinite(w.height_m) ? w.height_m : room.height_m;
    const mx = (w.from_m[0] + w.to_m[0]) / 2;
    const my = (w.from_m[1] + w.to_m[1]) / 2;
    let nx = -dy / len;
    let ny = dx / len;
    if (contains(poly, mx + nx * (t / 2 + 1e-4), my + ny * (t / 2 + 1e-4))) {
      nx = -nx;
      ny = -ny;
    }
    const geo = new THREE.BoxGeometry(len, h, t);
    const mat = new THREE.MeshLambertMaterial({
      color: MAT_COLOUR[w.material] != null ? MAT_COLOUR[w.material] : FALLBACK_COLOUR,
      transparent: true,
      opacity: 0.17,
      depthWrite: false,
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.set(mx + nx * t / 2, h / 2, my + ny * t / 2);
    mesh.rotation.y = -Math.atan2(dy, dx);
    mesh.renderOrder = 2;
    mesh.userData = { kind: "wall", index: i };
    G.content.add(mesh);
    G.pick.walls.push(mesh);

    const frame = new THREE.LineSegments(new THREE.EdgesGeometry(geo), lineMat(FAINT, 0.45));
    mesh.add(frame);
  }

  // ---- objects: real size, real rotation, one box each.
  for (const [i, o] of sc.objects.entries()) {
    const geo = new THREE.BoxGeometry(o.size_m[0], o.size_m[2], o.size_m[1]);
    const glassy = o.material === "glass";
    const mat = new THREE.MeshLambertMaterial({
      color: MAT_COLOUR[o.material] != null ? MAT_COLOUR[o.material] : FALLBACK_COLOUR,
      transparent: glassy,
      opacity: glassy ? 0.42 : 1,
      emissive: 0x000000,
    });
    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.copy(toWorld(o.centre_m));
    mesh.rotation.y = yawOf(o.rotation_deg);
    mesh.userData = { kind: "object", index: i };
    G.content.add(mesh);
    G.pick.objects.push(mesh);

    const frame = new THREE.LineSegments(new THREE.EdgesGeometry(geo), lineMat(INK, 0.3));
    mesh.add(frame);

    const top = new THREE.Vector3(o.centre_m[0], o.centre_m[2] + o.size_m[2] / 2 + 0.07, o.centre_m[1]);
    label(top, "object", o.type, size3(o.size_m), { kind: "object", index: i });
  }

  // ---- nodes: an ESP32 at the height someone mounted it, with a drop
  // line to the floor so the height is legible and not just asserted.
  for (const [i, nd] of sc.nodes.entries()) {
    const p = toWorld(nd.pos_m);
    const mesh = new THREE.Mesh(
      new THREE.SphereGeometry(0.075, 18, 12),
      new THREE.MeshLambertMaterial({ color: AMBER, emissive: 0x000000 }),
    );
    mesh.position.copy(p);
    mesh.userData = { kind: "node", index: i };
    G.content.add(mesh);
    G.pick.nodes.push(mesh);
    G.content.add(segments([p.x, 0.002, p.z, p.x, p.y, p.z], lineMat(AMBER, 0.45)));
    label(new THREE.Vector3(p.x, p.y + 0.16, p.z), "node", "n" + nd.id, n2(nd.pos_m[2]) + " m", { kind: "node", index: i });
  }

  buildDimensions(room, poly, box);
  applySelection();
  invalidate();
}

/**
 * The dimension overlay. Toggleable, drawn outside the room so it never
 * sits on top of the furniture, and labelled in metres because that is
 * what the file says and what a tape measure would say.
 */
function buildDimensions(room, poly, box) {
  const off = 0.45;
  const y = 0.01;
  const pts = [];
  const tick = (x, z, dx, dz) => pts.push(x - dx, y, z - dz, x + dx, y, z + dz);

  // width, along the near edge
  const zW = box[1] - off;
  pts.push(box[0], y, zW, box[2], y, zW);
  tick(box[0], zW, 0, 0.09);
  tick(box[2], zW, 0, 0.09);
  // depth, along the left edge
  const xD = box[0] - off;
  pts.push(xD, y, box[1], xD, y, box[3]);
  tick(xD, box[1], 0.09, 0);
  tick(xD, box[3], 0.09, 0);
  // height, vertical at the near-left corner
  pts.push(xD, 0, zW, xD, room.height_m, zW);
  pts.push(xD - 0.09, 0, zW, xD + 0.09, 0, zW);
  pts.push(xD - 0.09, room.height_m, zW, xD + 0.09, room.height_m, zW);

  const group = new THREE.Group();
  group.add(segments(pts, lineMat(AMBER, 0.75)));

  // The bounding box, dashed in as a plain rectangle, only when the
  // floor is a polygon: it is the number width_m and depth_m actually
  // describe, and the gap between it and the floor area is the point of
  // the L-shaped example.
  if (room.polygon_m.length) {
    const r = [
      [box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]],
    ];
    const bb = [];
    for (let i = 0; i < 4; i++) {
      const a = r[i];
      const b = r[(i + 1) % 4];
      // a dashed look without a dashed-line material: eight short pieces
      for (let k = 0; k < 8; k += 2) {
        const t0 = k / 8;
        const t1 = (k + 1) / 8;
        bb.push(
          a[0] + (b[0] - a[0]) * t0, 0.006, a[1] + (b[1] - a[1]) * t0,
          a[0] + (b[0] - a[0]) * t1, 0.006, a[1] + (b[1] - a[1]) * t1,
        );
      }
    }
    group.add(segments(bb, lineMat(FAINT, 0.6)));
    label(
      new THREE.Vector3((box[0] + box[2]) / 2, 0.02, box[3] + off),
      "dim",
      "bounding box",
      `${n2(room.width_m)} × ${n2(room.depth_m)} m, floor ${nc(area(poly))} m²`,
    );
  }

  G.content.add(group);
  label(new THREE.Vector3((box[0] + box[2]) / 2, y, zW), "dim", "width", n2(room.width_m) + " m");
  label(new THREE.Vector3(xD, y, (box[1] + box[3]) / 2), "dim", "depth", n2(room.depth_m) + " m");
  label(new THREE.Vector3(xD, room.height_m / 2, zW), "dim", "height", n2(room.height_m) + " m");
  label(new THREE.Vector3(box[0], 0.02, box[1]), "dim", "origin", "0.00, 0.00 m");
  group.userData.dims = true;
  group.visible = STATE.dims;
  G.dimGroup = group;
}

// ------------------------------------------------------------ selection

function applySelection() {
  if (!G) return;
  const all = [...G.pick.objects, ...G.pick.nodes];
  for (const m of all) {
    const on = STATE.sel && m.userData.kind === STATE.sel.kind && m.userData.index === STATE.sel.index;
    m.material.emissive.setHex(on ? 0x55431a : 0x000000);
  }
  for (const m of G.pick.walls) {
    const on = STATE.sel && STATE.sel.kind === "wall" && m.userData.index === STATE.sel.index;
    m.material.opacity = on ? 0.4 : 0.17;
  }
  for (const t of G.tags) {
    const on = Boolean(STATE.sel && t.ref && t.ref.kind === STATE.sel.kind && t.ref.index === STATE.sel.index);
    t.el.classList.toggle("is-sel", on);
  }
  invalidate();
}

function select(sel, announce) {
  STATE.sel = sel;
  applySelection();
  renderSelection();
  for (const b of EL.list.querySelectorAll("button")) {
    const on = sel != null && b.dataset.kind === sel.kind && Number(b.dataset.index) === sel.index;
    b.setAttribute("aria-pressed", on ? "true" : "false");
  }
  if (announce) setText(EL.live, selectionSentence());
}

function selectionSentence() {
  const sc = STATE.scene;
  if (!sc || !STATE.sel) return "Nothing selected.";
  const { kind, index } = STATE.sel;
  if (kind === "object") {
    const o = sc.objects[index];
    return `Selected: ${o.type}, ${size3(o.size_m)}, material ${o.material}. Defined in the scene file.`;
  }
  if (kind === "node") {
    const nd = sc.nodes[index];
    return `Selected: node ${nd.id} at ${pos3(nd.pos_m)}. A surveyed position, typed in.`;
  }
  const w = sc.walls[index];
  return `Selected: wall ${w.id || index}, ${w.material}, ${n2(w.thickness_m)} m thick. Defined in the scene file.`;
}

function renderSelection() {
  EL.sel.replaceChildren();
  const sc = STATE.scene;
  if (!sc) return;
  if (!STATE.sel) {
    EL.sel.append(el("p", "fine rs-none", "Nothing selected. Click a box in the view, or a line above."));
    return;
  }
  const { kind, index } = STATE.sel;
  const card = el("div", "rs-card");

  if (kind === "object") {
    const o = sc.objects[index];
    card.append(el("p", "rs-card-h", o.type));
    if (o.id) card.append(el("p", "fine", "id " + o.id));
    const dl = el("dl", "rs-dl");
    row(dl, "size (w × d × h)", size3(o.size_m));
    row(dl, "centre (x, y, z)", pos3(o.centre_m));
    row(dl, "rotation", n2(o.rotation_deg || 0) + "° counter-clockwise");
    row(dl, "material", String(o.material));
    card.append(dl);
    card.append(materialBlock(o.material));
  } else if (kind === "node") {
    const nd = sc.nodes[index];
    card.append(el("p", "rs-card-h", "node " + nd.id));
    if (nd.label) card.append(el("p", "fine", nd.label));
    const dl = el("dl", "rs-dl");
    row(dl, "position (x, y, z)", pos3(nd.pos_m));
    row(dl, "height", n2(nd.pos_m[2]) + " m");
    card.append(dl);
    card.append(el("p", "fine", "A surveyed position. Nothing in this project can work out where a node is: one RF chain gives no angle of arrival, and phase sanitization destroys absolute ranging. Any error in this number lands in whatever is computed from it."));
  } else {
    const w = sc.walls[index];
    card.append(el("p", "rs-card-h", "wall " + (w.id || index)));
    const dl = el("dl", "rs-dl");
    row(dl, "from → to", `${n2(w.from_m[0])}, ${n2(w.from_m[1])} → ${n2(w.to_m[0])}, ${n2(w.to_m[1])} m`);
    row(dl, "length", nc(Math.hypot(w.to_m[0] - w.from_m[0], w.to_m[1] - w.from_m[1])) + " m");
    row(dl, "thickness", n2(w.thickness_m) + " m");
    row(dl, "material", String(w.material));
    card.append(dl);
    card.append(materialBlock(w.material));
  }
  EL.sel.append(card);
}

/**
 * The material, from `scenes/materials.json`.
 *
 * `fabric` has a null row there on purpose: ITU-R P.2040 Table 3 has no
 * fabric, and no sourced permittivity was available, so none was
 * invented. That case prints the reason rather than a blank or a guess.
 * If the table itself did not load, this says so and prints no constants
 * at all — a number this page cannot source is a number it does not show.
 */
function materialBlock(name) {
  const wrap = el("div", "rs-mat");
  if (!STATE.materials) {
    wrap.append(el("p", "fine rs-warn", "materials.json did not load: " + STATE.materialsErr + ". No electrical constants are shown, and none are guessed."));
    return wrap;
  }
  const m = STATE.materials.materials ? STATE.materials.materials[name] : null;
  if (!m) {
    wrap.append(el("p", "fine rs-warn", "materials.json has no row named " + JSON.stringify(name) + ", so there are no constants to show."));
    return wrap;
  }
  const at = m.at_2437_mhz;
  wrap.append(el("p", "rs-mat-h", name));
  if (!at) {
    wrap.append(el("p", "fine rs-warn", "no sourced permittivity"));
    if (m.note) wrap.append(el("p", "fine", m.note));
    return wrap;
  }
  const ghz = STATE.materials.evaluated_at_ghz;
  const dl = el("dl", "rs-dl");
  row(dl, "εᵣ′", n2(at.epsilon_r));
  row(dl, "εᵣ″", Number(at.epsilon_imag).toPrecision(4));
  row(dl, "σ", Number(at.sigma_s_per_m).toPrecision(4) + " S/m");
  row(dl, "|Γ| at normal incidence", Number(at.reflection_normal).toPrecision(4));
  wrap.append(dl);
  wrap.append(el("p", "fine", `ITU-R P.2040 Table 3, evaluated at ${ghz} GHz. Confidence in this row: ${m.confidence}. |Γ| is one flat air-to-material interface, not a wall: a real wall is a slab with two of them. A sourced default, not a calibration — nothing in that table has been checked against a measurement made with this hardware.`));
  if (m.note) wrap.append(el("p", "fine", m.note));
  return wrap;
}

// ---------------------------------------------------------- the panel

function renderPanel() {
  const sc = STATE.scene;
  EL.room.replaceChildren();
  EL.list.replaceChildren();
  if (!sc) {
    setText(EL.name, "");
    renderSelection();
    return;
  }
  setText(EL.name, sc.name);

  const poly = footprint(sc.room);
  const a = area(poly);
  row(EL.room, "width (x)", n2(sc.room.width_m) + " m");
  row(EL.room, "depth (y)", n2(sc.room.depth_m) + " m");
  row(EL.room, "height (z)", n2(sc.room.height_m) + " m");
  if (sc.room.polygon_m.length) {
    row(EL.room, "floor", nc(a) + " m², a " + poly.length + "-sided polygon");
    row(EL.room, "bounding box", nc(sc.room.width_m * sc.room.depth_m) + " m²");
  } else {
    row(EL.room, "floor", nc(a) + " m²");
  }
  row(EL.room, "walls", sc.walls.length + " · " + countBy(sc.walls.map((w) => w.material)));
  row(EL.room, "objects", String(sc.objects.length));
  const zs = sc.nodes.map((n) => n.pos_m[2]);
  const zSpread = zs.length ? Math.max(...zs) - Math.min(...zs) : 0;
  row(
    EL.room,
    "nodes",
    sc.nodes.length + (zs.length ? (zSpread < 1e-9 ? ", all at " + n2(zs[0]) + " m" : ", heights spread " + nc(zSpread) + " m") : ""),
  );

  for (const [i, o] of sc.objects.entries()) EL.list.append(listRow("object", i, o.type, size3(o.size_m), o.material));
  for (const [i, nd] of sc.nodes.entries()) EL.list.append(listRow("node", i, "node " + nd.id, pos3(nd.pos_m), "esp32"));
  renderSelection();
}

function listRow(kind, index, head, detail, tail) {
  const li = el("li");
  const b = el("button", "rs-item");
  b.type = "button";
  b.dataset.kind = kind;
  b.dataset.index = String(index);
  b.setAttribute("aria-pressed", "false");
  b.append(el("b", null, head));
  const d = el("span", null, detail);
  d.append(el("i", null, "SCENE"));
  b.append(d);
  b.append(el("em", "rs-mtag", tail));
  b.addEventListener("click", () => select({ kind, index }, true));
  li.append(b);
  return li;
}

function countBy(list) {
  const seen = new Map();
  for (const k of list) seen.set(k, (seen.get(k) || 0) + 1);
  return [...seen].map(([k, n]) => k + " × " + n).join(", ");
}

// ----------------------------------------------------------- the frame

function tick() {
  requestAnimationFrame(tick);
  if (!G || !G.dirty) return;
  G.dirty = false;
  const cam = camera();
  G.renderer.render(G.root, cam);
  placeLabels(cam);
}

/**
 * Project each label to the canvas, nearest first, and drop the ones
 * that would land on a label already placed.
 *
 * Twenty-three labels in a room this size overlap into a pile, and a
 * pile is not a dimension overlay — it is the thing the overlay exists
 * to avoid. So the room's own dimensions are placed first and always
 * win, then the selected object, then whatever is nearest the camera.
 * A label behind the camera or off the edge is hidden rather than
 * clamped to the border, where it would point at nothing.
 */
const PROJ = new THREE.Vector3();
function placeLabels(cam) {
  const hide = (t) => {
    t.el.style.display = "none";
  };
  const live = [];
  for (const t of G.tags) {
    // The dimension labels belong to the overlay and come and go with
    // it. An object's own label stays, and loses its numbers: with the
    // overlay off it is the type and nothing else.
    if (t.kind === "dim" && !STATE.dims) {
      hide(t);
      continue;
    }
    PROJ.copy(t.at).project(cam);
    if (PROJ.z > 1 || PROJ.x < -1.15 || PROJ.x > 1.15 || PROJ.y < -1.15 || PROJ.y > 1.15) {
      hide(t);
      continue;
    }
    t.px = (PROJ.x * 0.5 + 0.5) * G.w;
    t.py = (-PROJ.y * 0.5 + 0.5) * G.h;
    t.depth = PROJ.z;
    t.rank = t.kind === "dim" ? 0 : t.el.classList.contains("is-sel") ? 1 : t.kind === "node" ? 2 : 3;
    live.push(t);
  }
  live.sort((a, b) => a.rank - b.rank || a.depth - b.depth);

  const taken = [];
  for (const t of live) {
    const w = (t.w || 56) / 2 + 2;
    const h = (t.h || 18) / 2 + 2;
    const b = [t.px - w, t.py - h, t.px + w, t.py + h];
    if (taken.some((p) => b[0] < p[2] && b[2] > p[0] && b[1] < p[3] && b[3] > p[1])) {
      hide(t);
      continue;
    }
    taken.push(b);
    t.el.style.display = "";
    t.el.style.transform = `translate(-50%,-50%) translate(${t.px.toFixed(1)}px, ${t.py.toFixed(1)}px)`;
  }
}

/**
 * Cache each label's size, which the overlap test needs and which only
 * changes when the labels do: on a new scene, and when the dimension
 * overlay adds or removes the numbers. Measuring here rather than in
 * the frame loop keeps the per-frame work to arithmetic.
 */
function measureTags() {
  if (!G) return;
  for (const t of G.tags) {
    const was = t.el.style.display;
    t.el.style.visibility = "hidden";
    t.el.style.display = "";
    t.w = t.el.offsetWidth;
    t.h = t.el.offsetHeight;
    t.el.style.display = was;
    t.el.style.visibility = "";
  }
  invalidate();
}

function applyDims() {
  if (!G) return;
  if (G.dimGroup) G.dimGroup.visible = STATE.dims;
  EL.stage.classList.toggle("is-dims", STATE.dims);
  measureTags();
}

// ------------------------------------------------------- camera control
//
// OrbitControls is a three.js addon and is not vendored, so this is the
// part of it this page needs: drag to rotate, wheel and pinch to zoom,
// and in the top-down view a drag pans instead of rotating. No damping,
// no auto-rotate, nothing that keeps running after the finger is lifted.

function wireCamera() {
  const dom = G.renderer.domElement;
  const pointers = new Map();
  let last = null;
  let pinch = 0;
  let moved = 0;

  dom.addEventListener("pointerdown", (e) => {
    dom.setPointerCapture(e.pointerId);
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    last = { x: e.clientX, y: e.clientY };
    if (pointers.size === 2) pinch = spread(pointers);
    moved = 0;
  });

  dom.addEventListener("pointermove", (e) => {
    if (!pointers.has(e.pointerId)) return;
    pointers.set(e.pointerId, { x: e.clientX, y: e.clientY });
    const c = G.cam;
    if (pointers.size >= 2) {
      const now = spread(pointers);
      if (pinch > 0 && now > 0) {
        const k = pinch / now;
        if (STATE.topDown) c.half *= k;
        else c.r *= k;
        applyCamera();
      }
      pinch = now;
      moved = 99;
      return;
    }
    const dx = e.clientX - last.x;
    const dy = e.clientY - last.y;
    last = { x: e.clientX, y: e.clientY };
    moved += Math.abs(dx) + Math.abs(dy);
    if (STATE.topDown) pan(dx, dy);
    else {
      c.theta -= dx * 0.006;
      c.phi -= dy * 0.006;
      applyCamera();
    }
  });

  const end = (e) => {
    pointers.delete(e.pointerId);
    if (pointers.size < 2) pinch = 0;
    if (pointers.size === 0 && moved < 6) pickAt(e);
  };
  dom.addEventListener("pointerup", end);
  dom.addEventListener("pointercancel", (e) => {
    pointers.delete(e.pointerId);
    pinch = 0;
  });

  dom.addEventListener(
    "wheel",
    (e) => {
      e.preventDefault();
      const k = Math.exp(e.deltaY * 0.0012);
      if (STATE.topDown) G.cam.half *= k;
      else G.cam.r *= k;
      applyCamera();
    },
    { passive: false },
  );

  // Keyboard, on the stage rather than the canvas, because the stage is
  // the element in the tab order.
  EL.stage.addEventListener("keydown", (e) => {
    const c = G.cam;
    const step = e.shiftKey ? 0.25 : 0.08;
    const top = STATE.topDown;
    let used = true;
    switch (e.key) {
      case "ArrowLeft":
        if (top) pan(40, 0);
        else c.theta -= step;
        break;
      case "ArrowRight":
        if (top) pan(-40, 0);
        else c.theta += step;
        break;
      case "ArrowUp":
        if (top) pan(0, 40);
        else c.phi -= step;
        break;
      case "ArrowDown":
        if (top) pan(0, -40);
        else c.phi += step;
        break;
      case "+":
      case "=":
        if (top) c.half *= 0.88;
        else c.r *= 0.88;
        break;
      case "-":
      case "_":
        if (top) c.half /= 0.88;
        else c.r /= 0.88;
        break;
      default:
        used = false;
    }
    if (used) {
      e.preventDefault();
      applyCamera();
    }
  });
}

function spread(pointers) {
  const [a, b] = [...pointers.values()];
  return Math.hypot(a.x - b.x, a.y - b.y);
}

/** Top-down only: move the look-at point so the floor follows the drag. */
function pan(dx, dy) {
  const c = G.cam;
  const aspect = G.w / G.h;
  c.target.x -= (dx * 2 * c.half * aspect) / G.w;
  c.target.z -= (dy * 2 * c.half) / G.h;
  applyCamera();
}

/**
 * Click to select. Furniture and nodes first, walls only if nothing else
 * was hit: a wall is a large translucent sheet between the camera and
 * half the room, and having it swallow every click would make the
 * furniture unselectable.
 */
function pickAt(e) {
  const r = G.renderer.domElement.getBoundingClientRect();
  const ndc = new THREE.Vector2(
    ((e.clientX - r.left) / r.width) * 2 - 1,
    -((e.clientY - r.top) / r.height) * 2 + 1,
  );
  G.ray.setFromCamera(ndc, camera());
  const near = G.ray.intersectObjects([...G.pick.objects, ...G.pick.nodes], false);
  const hit = near.length ? near[0] : (G.ray.intersectObjects(G.pick.walls, false)[0] || null);
  select(hit ? { kind: hit.object.userData.kind, index: hit.object.userData.index } : null, true);
}

// ------------------------------------------------------------- loading

async function loadMaterials() {
  try {
    const r = await fetch("scenes/materials.json", { cache: "no-cache" });
    if (!r.ok) throw new Error("the server answered " + r.status);
    STATE.materials = JSON.parse(await r.text());
    STATE.materialsErr = "";
  } catch (e) {
    STATE.materials = null;
    STATE.materialsErr = e.message;
  }
}

async function loadScene(entry) {
  const token = ++STATE.token;
  STATE.entry = entry;
  for (const b of EL.pick.querySelectorAll("button")) {
    b.setAttribute("aria-pressed", b.dataset.key === entry.key ? "true" : "false");
  }
  setText(EL.err, "");
  try {
    const r = await fetch(entry.path, { cache: "no-cache" });
    if (!r.ok) throw new Error("the server answered " + r.status);
    const scene = readScene(await r.text());
    if (token !== STATE.token) return;
    STATE.scene = scene;
    STATE.sel = null;
    renderPanel();
    build();
    fitCamera();
    applyDims();
    if (G) {
      EL.stage.setAttribute(
        "aria-label",
        `A 3D view of ${scene.name || entry.nav}, drawn from ${entry.file}. Drag to rotate, scroll to zoom. With this view focused, the arrow keys rotate and the plus and minus keys zoom. Every object in it is listed in the panel.`,
      );
    }
  } catch (e) {
    if (token !== STATE.token) return;
    STATE.scene = null;
    STATE.sel = null;
    clearContent();
    renderPanel();
    setText(EL.err, entry.file + " did not load: " + e.message + ". Nothing is drawn: this page does not substitute a room it could not read.");
  }
}

// ---------------------------------------------------------------- boot

function boot() {
  for (const s of SCENES) {
    const b = el("button", null, s.nav);
    b.type = "button";
    b.dataset.key = s.key;
    b.setAttribute("aria-pressed", "false");
    b.addEventListener("click", () => loadScene(s));
    EL.pick.append(b);
  }

  const toggle = (btn, read, write) => {
    btn.addEventListener("click", () => {
      write(!read());
      btn.setAttribute("aria-pressed", read() ? "true" : "false");
    });
  };
  toggle(EL.top, () => STATE.topDown, (v) => {
    STATE.topDown = v;
    EL.stage.classList.toggle("is-top", v);
    if (G) applyCamera();
  });
  toggle(EL.dim, () => STATE.dims, (v) => {
    STATE.dims = v;
    applyDims();
  });
  // Zooming into a corner is easy and finding the way out by hand is
  // not, and a reload would be a heavy way to ask for the first view
  // back.
  EL.reset.addEventListener("click", () => {
    fitCamera();
    setText(EL.live, "The view is back where it started.");
  });

  G = startThree();
  if (!G) {
    setText(EL.err, "This browser did not give the page a WebGL context, so the room is not drawn. The dimensions, the objects and their materials are still listed in the panel and in the text below.");
  } else {
    wireCamera();
    resize();
    if ("ResizeObserver" in window) new ResizeObserver(resize).observe(EL.stage);
    else addEventListener("resize", resize);
    requestAnimationFrame(tick);
  }

  loadMaterials().then(() => {
    if (STATE.sel) renderSelection();
  });
  loadScene(SCENES[0]);
}

// One page has these elements. The module is served to that page only,
// but a missing stage would otherwise be a thrown error in the console
// rather than a page that simply does nothing.
if (EL.stage) boot();


