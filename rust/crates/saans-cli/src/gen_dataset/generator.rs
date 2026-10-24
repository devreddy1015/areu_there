//! The scene generator of spec/dataset.md §2, with the randomness of §9.
//!
//! Scene `i`, attempt `a` (0..99) draws, in §2's order, from
//! `Rng::stream(seed, (i << 8) | a)`: the room (§2.1), its walls (§2.2),
//! furniture and openings (§2.3), nodes (§2.4) and people (§2.5); then the
//! scene's Room Sim seed is that stream's next `next_u64()`. A draw that
//! fails a §2.4/§2.5 rule, or that a §2.6 check rejects, is redrawn from
//! attempt `a + 1`; the rejections are counted by reason.
//!
//! Where §2 leaves an order open, the draws are, per object: kind, size x,
//! y, z, material (only for a kind with more than one); then per try yaw
//! and the centre (x, then y). Per opening try: the edge, then the
//! position along it. Per node try: the edge, the position along it, then
//! z. A `lie` or `sit` person takes a free object uniformly among the free
//! ones (one draw); a `walk` or `stand` start is `x` then `y` per try, and
//! a `stand` heading is drawn after the start. Every distribution is a
//! modelling choice (SIM), listed by [`generator_json`] for the manifest.

use std::f64::consts::TAU;

use saans_locate::phone::Config as LocateConfig;
use saans_locate::pipeline::{
    base_rate_hz, check_roomsim_config, check_tx_period, person_geometry, resolve_config,
    PersonLocator,
};
use saans_roomsim::materials::SurfaceMaterial;
use saans_roomsim::motion::Walkability;
use saans_roomsim::room::RoomModel;
use saans_roomsim::{Config as RoomSimConfig, Mode, Simulator};
use saans_scene::geom::point_in_polygon;
use saans_scene::{
    Node, Object, ObjectKind, ObjectMaterial, Person, Point, Provenance, Room, Scene, Wall,
    WallMaterial, SCHEMA_VERSION,
};
use saans_sim::rng::Rng;
use serde_json::{json, Value};

// ---------------------------------------------------------------------
// §2's distributions (SIM modelling choices). Lengths in metres.
// ---------------------------------------------------------------------

/// §2.1: `w_m`, `d_m` ~ U[3, 10], `h_m` ~ U[2.5, 3.5], rounded to 0.01.
pub const ROOM_W_M: (f64, f64) = (3.0, 10.0);
pub const ROOM_D_M: (f64, f64) = (3.0, 10.0);
pub const ROOM_H_M: (f64, f64) = (2.5, 3.5);
/// §2.1: the footprint is an L with this probability.
pub const L_SHAPE_P: f64 = 0.25;
/// §2.1: an L's cut extents, as fractions of `w_m` and of `d_m`.
pub const L_CUT_FRAC: (f64, f64) = (0.3, 0.5);

/// §2.2: the room's wall material, its probability and its thickness
/// range (rounded to 0.001).
pub const WALL_MATERIALS: [(WallMaterial, f64, (f64, f64)); 5] = [
    (WallMaterial::Concrete, 0.35, (0.15, 0.25)),
    (WallMaterial::Brick, 0.30, (0.15, 0.25)),
    (WallMaterial::Drywall, 0.20, (0.08, 0.12)),
    (WallMaterial::Wood, 0.10, (0.04, 0.08)),
    (WallMaterial::Glass, 0.05, (0.008, 0.012)),
];
/// §2.2: Room Sim's `floor_material` and `ceiling_material`.
pub const FLOOR_MATERIALS: [(SurfaceMaterial, f64); 2] = [
    (SurfaceMaterial::Concrete, 0.8),
    (SurfaceMaterial::Wood, 0.2),
];
pub const CEILING_MATERIALS: [(SurfaceMaterial, f64); 2] = [
    (SurfaceMaterial::Concrete, 0.9),
    (SurfaceMaterial::Drywall, 0.1),
];

/// One furniture kind of §2.3's table.
#[derive(Debug, Clone, Copy)]
pub struct Furniture {
    pub kind: ObjectKind,
    pub p: f64,
    /// Size x, y, z ranges (uniform, rounded to 0.01).
    pub size: [(f64, f64); 3],
    pub materials: &'static [(ObjectMaterial, f64)],
}

/// §2.3: `n` ~ U{0..6} objects of these kinds.
pub const FURNITURE_COUNT: (usize, usize) = (0, 6);
pub const FURNITURE: [Furniture; 5] = [
    Furniture {
        kind: ObjectKind::Bed,
        p: 0.2,
        size: [(1.8, 2.1), (0.9, 1.9), (0.4, 0.6)],
        materials: &[(ObjectMaterial::Fabric, 1.0)],
    },
    Furniture {
        kind: ObjectKind::Table,
        p: 0.2,
        size: [(0.8, 1.6), (0.6, 0.9), (0.72, 0.78)],
        materials: &[(ObjectMaterial::Wood, 0.7), (ObjectMaterial::Metal, 0.3)],
    },
    Furniture {
        kind: ObjectKind::Chair,
        p: 0.25,
        size: [(0.45, 0.55), (0.45, 0.55), (0.8, 1.0)],
        materials: &[
            (ObjectMaterial::Wood, 0.6),
            (ObjectMaterial::Metal, 0.2),
            (ObjectMaterial::Fabric, 0.2),
        ],
    },
    Furniture {
        kind: ObjectKind::Almirah,
        p: 0.15,
        size: [(0.9, 1.4), (0.5, 0.65), (1.8, 2.1)],
        materials: &[(ObjectMaterial::Wood, 0.5), (ObjectMaterial::Metal, 0.5)],
    },
    Furniture {
        kind: ObjectKind::Sofa,
        p: 0.2,
        size: [(1.6, 2.2), (0.8, 0.95), (0.75, 0.9)],
        materials: &[(ObjectMaterial::Fabric, 1.0)],
    },
];
/// §2.3: an object's footprint keeps this much beyond the inner face
/// (`t/2` inside the edge) and from every placed object's footprint.
pub const FURNITURE_FACE_GAP_M: f64 = 0.1;
pub const FURNITURE_GAP_M: f64 = 0.1;
/// §2.3: placement tries per object and per opening.
pub const PLACE_TRIES: usize = 50;

/// One opening kind of §2.3.
#[derive(Debug, Clone, Copy)]
pub struct Opening {
    pub kind: ObjectKind,
    pub size: (f64, f64, f64),
    pub z: f64,
    pub material: ObjectMaterial,
}

pub const DOOR_P: f64 = 0.7;
pub const DOOR: Opening = Opening {
    kind: ObjectKind::Door,
    size: (0.9, 0.1, 2.0),
    z: 1.0,
    material: ObjectMaterial::Wood,
};
/// `n_w` ~ U{0..2} windows.
pub const WINDOW_COUNT: (usize, usize) = (0, 2);
pub const WINDOW: Opening = Opening {
    kind: ObjectKind::Window,
    size: (1.0, 0.1, 1.0),
    z: 1.5,
    material: ObjectMaterial::Glass,
};
/// §2.3: openings go on edges at least this long, centred at least
/// `OPENING_END_M` from the edge's ends, `OPENING_GAP_M` apart on one edge.
pub const OPENING_MIN_EDGE_M: f64 = 1.6;
pub const OPENING_END_M: f64 = 0.6;
pub const OPENING_GAP_M: f64 = 0.1;

/// §2.4: `N` ~ U{4..12} nodes, inset `max(t/2, 0.05) + 0.05` from an
/// edge, at least 0.3 m from its ends, z ~ U[0.8, 2.2] (rounded to 0.01),
/// at least 0.5 m apart in (x, y); 200 tries per node.
pub const NODE_COUNT: (usize, usize) = (4, 12);
pub const NODE_INSET_MIN_M: f64 = 0.05;
pub const NODE_INSET_EXTRA_M: f64 = 0.05;
pub const NODE_END_M: f64 = 0.3;
pub const NODE_Z_M: (f64, f64) = (0.8, 2.2);
pub const NODE_GAP_M: f64 = 0.5;
pub const NODE_TRIES: usize = 200;

/// A generated person's mode (§2.5).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PersonMode {
    Walk,
    Stand,
    Sit,
    Lie,
}

impl PersonMode {
    pub fn name(self) -> &'static str {
        match self {
            PersonMode::Walk => "walk",
            PersonMode::Stand => "stand",
            PersonMode::Sit => "sit",
            PersonMode::Lie => "lie",
        }
    }
}

/// §2.5: `P` ~ U{0..3} people, each with a mode; 1000 tries for a
/// walkable start.
pub const PEOPLE_COUNT: (usize, usize) = (0, 3);
pub const MODES: [(PersonMode, f64); 4] = [
    (PersonMode::Walk, 0.5),
    (PersonMode::Stand, 0.15),
    (PersonMode::Sit, 0.2),
    (PersonMode::Lie, 0.15),
];
pub const START_TRIES: usize = 1000;

/// §2.6: attempts per scene before the command fails.
pub const MAX_ATTEMPTS: u32 = 100;

// ---------------------------------------------------------------------
// What a scene is drawn into
// ---------------------------------------------------------------------

/// Why a draw was redrawn (§2.6), for the manifest's `rejections`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Reason {
    /// A node found no place in 200 tries (§2.4).
    Nodes,
    /// A walking or standing person found no walkable start in 1000 tries
    /// (§2.5).
    WalkStart,
    /// `Scene::validate()` returned an error.
    Scene,
    /// Room Sim refused the scene or the run config (`Simulator::new`, or
    /// the `RoomModel` the walkability rules are built from).
    RoomSim,
    /// One of Locate's refusals before its calibration.
    Locate,
}

/// Rejected draws by reason (§7.3 `rejections`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Rejections {
    pub nodes: u64,
    pub walk_start: u64,
    pub scene: u64,
    pub roomsim: u64,
    pub locate: u64,
}

impl Rejections {
    pub fn add(&mut self, r: Reason) {
        match r {
            Reason::Nodes => self.nodes += 1,
            Reason::WalkStart => self.walk_start += 1,
            Reason::Scene => self.scene += 1,
            Reason::RoomSim => self.roomsim += 1,
            Reason::Locate => self.locate += 1,
        }
    }

    pub fn merge(&mut self, o: &Rejections) {
        self.nodes += o.nodes;
        self.walk_start += o.walk_start;
        self.scene += o.scene;
        self.roomsim += o.roomsim;
        self.locate += o.locate;
    }

    pub fn total(&self) -> u64 {
        self.nodes + self.walk_start + self.scene + self.roomsim + self.locate
    }

    pub fn to_json(self) -> Value {
        json!({
            "nodes": self.nodes,
            "walk_start": self.walk_start,
            "scene_invalid": self.scene,
            "roomsim": self.roomsim,
            "locate": self.locate,
        })
    }

    pub fn from_json(v: &Value) -> Option<Rejections> {
        let n = |k: &str| v.get(k).and_then(Value::as_u64);
        Some(Rejections {
            nodes: n("nodes")?,
            walk_start: n("walk_start")?,
            scene: n("scene_invalid")?,
            roomsim: n("roomsim")?,
            locate: n("locate")?,
        })
    }
}

/// An accepted scene (§2) with its §3 run config and what the manifest
/// records about it.
#[derive(Debug, Clone)]
pub struct Generated {
    pub index: usize,
    pub scene: Scene,
    /// The §3 run config, `impair = true` (§2.6); the caller sets `impair`.
    pub rs: RoomSimConfig,
    pub attempts: u32,
    pub l_shape: bool,
    pub wall_material: WallMaterial,
    pub objects_drawn: usize,
    pub objects_placed: usize,
    pub openings_drawn: usize,
    pub openings_placed: usize,
    /// Per person, in index order, after a `lie`/`sit` fell back to `walk`.
    pub modes: Vec<PersonMode>,
    /// This scene's rejected attempts.
    pub rejections: Rejections,
}

/// No valid draw in [`MAX_ATTEMPTS`] attempts: a generator bug, not data.
#[derive(Debug, Clone)]
pub struct Exhausted {
    pub index: usize,
    pub rejections: Rejections,
    /// The last attempt's rejection, as Room Sim or Locate worded it.
    pub last: String,
}

/// The scene's name, `gen-{i:06}` (§2).
pub fn scene_id(i: usize) -> String {
    format!("gen-{i:06}")
}

pub fn wall_material_name(m: WallMaterial) -> &'static str {
    SurfaceMaterial::from(m).name()
}

// ---------------------------------------------------------------------
// Draw helpers. Each is one draw unless it says otherwise.
// ---------------------------------------------------------------------

/// Rounded to 0.01.
fn r2(x: f64) -> f64 {
    (x * 100.0).round() / 100.0
}

/// Rounded to 0.001.
fn r3(x: f64) -> f64 {
    (x * 1000.0).round() / 1000.0
}

/// Uniform on the integers `lo..=hi`.
fn int(rng: &mut Rng, (lo, hi): (usize, usize)) -> usize {
    let n = hi - lo + 1;
    lo + ((rng.uniform() * n as f64) as usize).min(n - 1)
}

/// One of `items` with its probability (the last on rounding).
fn pick<T: Copy>(rng: &mut Rng, items: &[(T, f64)]) -> T {
    let u = rng.uniform();
    let mut acc = 0.0;
    for &(v, p) in items {
        acc += p;
        if u < acc {
            return v;
        }
    }
    items[items.len() - 1].0
}

/// The polygon's edges `(a, b)`, `b` the next vertex.
pub fn edges(poly: &[Point]) -> Vec<(Point, Point)> {
    (0..poly.len())
        .map(|i| (poly[i], poly[(i + 1) % poly.len()]))
        .collect()
}

fn length((a, b): (Point, Point)) -> f64 {
    (b.0 - a.0).hypot(b.1 - a.1)
}

/// An edge chosen in proportion to its length.
fn by_length(rng: &mut Rng, es: &[(Point, Point)]) -> usize {
    let total: f64 = es.iter().map(|&e| length(e)).sum();
    let u = rng.uniform() * total;
    let mut acc = 0.0;
    for (i, &e) in es.iter().enumerate() {
        acc += length(e);
        if u < acc {
            return i;
        }
    }
    es.len() - 1
}

/// The point `s` along edge `e` and `inset` towards the inside (the left
/// of a counter-clockwise edge).
fn along((a, b): (Point, Point), s: f64, inset: f64) -> Point {
    let len = length((a, b));
    let (ex, ey) = ((b.0 - a.0) / len, (b.1 - a.1) / len);
    (a.0 + ex * s - ey * inset, a.1 + ey * s + ex * inset)
}

/// An axis-aligned box on the floor plane.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rect {
    pub x0: f64,
    pub y0: f64,
    pub x1: f64,
    pub y1: f64,
}

impl Rect {
    /// A segment as a degenerate box (the footprint's edges are axis-aligned).
    pub fn of_segment((a, b): (Point, Point)) -> Rect {
        Rect {
            x0: a.0.min(b.0),
            y0: a.1.min(b.1),
            x1: a.0.max(b.0),
            y1: a.1.max(b.1),
        }
    }

    /// The footprint of a box at yaw 0, 90, 180 or 270 degrees.
    pub fn of_object(o: &Object) -> Rect {
        let quarter = (o.yaw_deg / 90.0).round() as i64;
        let (hx, hy) = if quarter.rem_euclid(2) == 1 {
            (o.size.1 / 2.0, o.size.0 / 2.0)
        } else {
            (o.size.0 / 2.0, o.size.1 / 2.0)
        };
        Rect {
            x0: o.pos.0 - hx,
            y0: o.pos.1 - hy,
            x1: o.pos.0 + hx,
            y1: o.pos.1 + hy,
        }
    }

    /// The distance between two boxes, 0 when they touch or overlap.
    pub fn distance(&self, o: &Rect) -> f64 {
        let dx = (self.x0 - o.x1).max(o.x0 - self.x1).max(0.0);
        let dy = (self.y0 - o.y1).max(o.y0 - self.y1).max(0.0);
        dx.hypot(dy)
    }
}

/// `p` at height `z` is inside the object's closed box.
fn in_box(o: &Object, p: Point, z: f64) -> bool {
    let psi = o.yaw_deg.to_radians();
    let (c, s) = (psi.cos(), psi.sin());
    let (dx, dy) = (p.0 - o.pos.0, p.1 - o.pos.1);
    let (lx, ly) = (dx * c + dy * s, -dx * s + dy * c);
    lx.abs() <= o.size.0 / 2.0
        && ly.abs() <= o.size.1 / 2.0
        && (z - o.pos.2).abs() <= o.size.2 / 2.0
}

/// §2.1: the rectangle `(0,0)..(w,d)` with corner `corner` (0 `(0,0)`, 1
/// `(w,0)`, 2 `(w,d)`, 3 `(0,d)`) cut away by `cx` x `cy`,
/// counter-clockwise. The bounding box stays `(0,0)..(w,d)`.
pub fn l_polygon(w: f64, d: f64, corner: usize, cx: f64, cy: f64) -> Vec<Point> {
    let (xa, ya) = (r2(w - cx), r2(d - cy));
    match corner {
        0 => vec![(cx, 0.0), (w, 0.0), (w, d), (0.0, d), (0.0, cy), (cx, cy)],
        1 => vec![(0.0, 0.0), (xa, 0.0), (xa, cy), (w, cy), (w, d), (0.0, d)],
        2 => vec![(0.0, 0.0), (w, 0.0), (w, ya), (xa, ya), (xa, d), (0.0, d)],
        _ => vec![(0.0, 0.0), (w, 0.0), (w, d), (cx, d), (cx, ya), (0.0, ya)],
    }
}

/// The §3 run config of a scene: Room Sim's defaults with `seed`,
/// `duration_s = S + 0.5`, the floor and ceiling materials and the people
/// (`n_people`, `people_mode` for walkers and standers,
/// `start_heading_rad` for standers); `impair = true` (§2.6).
pub fn run_config(
    seed: u64,
    seconds: u32,
    floor: SurfaceMaterial,
    ceiling: SurfaceMaterial,
    people: &[(PersonMode, Option<f64>)],
) -> RoomSimConfig {
    let mut rs = RoomSimConfig {
        seed,
        duration_s: f64::from(seconds) + 0.5,
        floor_material: floor,
        ceiling_material: ceiling,
        n_people: Some(people.len()),
        impair: true,
        ..RoomSimConfig::default()
    };
    for (j, &(mode, heading)) in people.iter().enumerate() {
        match mode {
            PersonMode::Walk => {
                rs.people_mode.insert(j, Mode::Walk);
            }
            PersonMode::Stand => {
                rs.people_mode.insert(j, Mode::Stationary);
            }
            PersonMode::Sit | PersonMode::Lie => {}
        }
        if let Some(h) = heading {
            rs.start_heading_rad.insert(j, h);
        }
    }
    rs
}

/// A drawn scene before the §2.6 checks.
struct Draw {
    scene: Scene,
    rs: RoomSimConfig,
    l_shape: bool,
    wall_material: WallMaterial,
    objects_drawn: usize,
    objects_placed: usize,
    openings_drawn: usize,
    openings_placed: usize,
    modes: Vec<PersonMode>,
}

/// Attempt `a` of scene `i` (§2.1-§2.5, §9).
fn draw(seed: u64, i: usize, a: u32, seconds: u32) -> Result<Draw, (Reason, String)> {
    let mut rng = Rng::stream(seed, ((i as u64) << 8) | u64::from(a));

    // §2.1 the room.
    let w = r2(rng.range(ROOM_W_M.0, ROOM_W_M.1));
    let d = r2(rng.range(ROOM_D_M.0, ROOM_D_M.1));
    let h = r2(rng.range(ROOM_H_M.0, ROOM_H_M.1));
    let l_shape = rng.uniform() < L_SHAPE_P;
    let polygon = if l_shape {
        let corner = int(&mut rng, (0, 3));
        let cx = r2(rng.range(L_CUT_FRAC.0, L_CUT_FRAC.1) * w);
        let cy = r2(rng.range(L_CUT_FRAC.0, L_CUT_FRAC.1) * d);
        Some(l_polygon(w, d, corner, cx, cy))
    } else {
        None
    };
    let footprint = polygon
        .clone()
        .unwrap_or_else(|| vec![(0.0, 0.0), (w, 0.0), (w, d), (0.0, d)]);
    let all_edges = edges(&footprint);

    // §2.2 walls: one per edge, centreline on it, the room's height.
    let wall_material = pick(&mut rng, &WALL_MATERIALS.map(|(m, p, _)| (m, p)));
    let (t_lo, t_hi) = WALL_MATERIALS
        .iter()
        .find(|(m, _, _)| *m == wall_material)
        .map(|&(_, _, r)| r)
        .expect("a listed material");
    let t = r3(rng.range(t_lo, t_hi));
    let floor = pick(&mut rng, &FLOOR_MATERIALS);
    let ceiling = pick(&mut rng, &CEILING_MATERIALS);
    let walls: Vec<Wall> = all_edges
        .iter()
        .map(|&(from, to)| Wall {
            from,
            to,
            material: wall_material,
            thickness_m: t,
            height_m: None,
        })
        .collect();

    // §2.3 furniture.
    let kinds: Vec<(usize, f64)> = FURNITURE
        .iter()
        .enumerate()
        .map(|(k, f)| (k, f.p))
        .collect();
    let objects_drawn = int(&mut rng, FURNITURE_COUNT);
    let mut objects: Vec<Object> = Vec::new();
    let mut rects: Vec<Rect> = Vec::new();
    for _ in 0..objects_drawn {
        let f = &FURNITURE[pick(&mut rng, &kinds)];
        let sx = r2(rng.range(f.size[0].0, f.size[0].1));
        let sy = r2(rng.range(f.size[1].0, f.size[1].1));
        let sz = r2(rng.range(f.size[2].0, f.size[2].1));
        let material = if f.materials.len() > 1 {
            pick(&mut rng, f.materials)
        } else {
            f.materials[0].0
        };
        for _ in 0..PLACE_TRIES {
            let yaw = 90.0 * int(&mut rng, (0, 3)) as f64;
            let cx = rng.range(0.0, w);
            let cy = rng.range(0.0, d);
            let o = Object {
                kind: f.kind,
                id: None,
                pos: (cx, cy, sz / 2.0),
                size: (sx, sy, sz),
                yaw_deg: yaw,
                material,
            };
            let r = Rect::of_object(&o);
            let inside = point_in_polygon(&footprint, (cx, cy))
                && all_edges
                    .iter()
                    .all(|&e| r.distance(&Rect::of_segment(e)) >= t / 2.0 + FURNITURE_FACE_GAP_M);
            if inside && rects.iter().all(|q| r.distance(q) >= FURNITURE_GAP_M) {
                rects.push(r);
                objects.push(o);
                break;
            }
        }
    }
    let objects_placed = objects.len();

    // §2.3 openings, on the centreline of an edge at least 1.6 m long.
    let long: Vec<(Point, Point)> = all_edges
        .iter()
        .copied()
        .filter(|&e| length(e) >= OPENING_MIN_EDGE_M)
        .collect();
    let door = rng.uniform() < DOOR_P;
    let n_windows = int(&mut rng, WINDOW_COUNT);
    // `repeat().take()`, not `repeat_n`: the workspace's rust-version is 1.80.
    #[allow(clippy::manual_repeat_n)]
    let wanted: Vec<Opening> = door
        .then_some(DOOR)
        .into_iter()
        .chain(std::iter::repeat(WINDOW).take(n_windows))
        .collect();
    // (edge, position along it, half width) of each placed opening.
    let mut placed: Vec<(usize, f64, f64)> = Vec::new();
    for op in &wanted {
        let half = op.size.0 / 2.0;
        for _ in 0..PLACE_TRIES {
            let e = by_length(&mut rng, &long);
            let len = length(long[e]);
            let s = rng.range(OPENING_END_M, len - OPENING_END_M);
            let clear = placed
                .iter()
                .filter(|(pe, _, _)| *pe == e)
                .all(|&(_, ps, ph)| (s - ps).abs() >= half + ph + OPENING_GAP_M);
            if clear {
                let (a, b) = long[e];
                let (x, y) = along(long[e], s, 0.0);
                objects.push(Object {
                    kind: op.kind,
                    id: None,
                    pos: (x, y, op.z),
                    size: op.size,
                    yaw_deg: (b.1 - a.1).atan2(b.0 - a.0).to_degrees(),
                    material: op.material,
                });
                placed.push((e, s, half));
                break;
            }
        }
    }
    let openings_drawn = wanted.len();
    let openings_placed = placed.len();

    // §2.4 nodes, inset from an edge so Room Sim's t/2 + 0.01 holds.
    let n_nodes = int(&mut rng, NODE_COUNT);
    let inset = (t / 2.0).max(NODE_INSET_MIN_M) + NODE_INSET_EXTRA_M;
    let mut nodes: Vec<Node> = Vec::with_capacity(n_nodes);
    for id in 0..n_nodes {
        let mut found = None;
        for _ in 0..NODE_TRIES {
            let e = by_length(&mut rng, &all_edges);
            let len = length(all_edges[e]);
            let s = rng.range(NODE_END_M, len - NODE_END_M);
            let z = r2(rng.range(NODE_Z_M.0, NODE_Z_M.1));
            let p = along(all_edges[e], s, inset);
            let apart = nodes
                .iter()
                .all(|n| (n.pos.0 - p.0).hypot(n.pos.1 - p.1) >= NODE_GAP_M);
            if apart && !objects.iter().any(|o| in_box(o, p, z)) {
                found = Some((p.0, p.1, z));
                break;
            }
        }
        let Some(pos) = found else {
            return Err((
                Reason::Nodes,
                format!("node {id} found no place in {NODE_TRIES} tries"),
            ));
        };
        nodes.push(Node { id: id as u8, pos });
    }

    let mut scene = Scene {
        schema_version: SCHEMA_VERSION,
        name: scene_id(i),
        provenance: Provenance::Generated,
        room: Room {
            w_m: w,
            d_m: d,
            h_m: h,
            polygon,
        },
        walls,
        objects,
        nodes,
        people: Vec::new(),
        phone: None,
    };

    // §2.5 people. Walkability is Room Sim's own, from the run config
    // (its walk margin, body radius and object clearance are defaults).
    let base = run_config(0, seconds, floor, ceiling, &[]);
    let model = RoomModel::from_scene(&scene, &base).map_err(|e| {
        let reason = match e {
            saans_roomsim::Error::Scene(_) => Reason::Scene,
            _ => Reason::RoomSim,
        };
        (reason, e.to_string())
    })?;
    let walk = Walkability::new(&model, &base);
    let n_people = int(&mut rng, PEOPLE_COUNT);
    let mut taken = vec![false; scene.objects.len()];
    let mut people: Vec<(PersonMode, Option<f64>)> = Vec::with_capacity(n_people);
    let mut starts: Vec<Point> = Vec::with_capacity(n_people);
    for j in 0..n_people {
        let mut mode = pick(&mut rng, &MODES);
        let mut start = None;
        if matches!(mode, PersonMode::Sit | PersonMode::Lie) {
            let fits = |k: ObjectKind| match mode {
                PersonMode::Lie => k == ObjectKind::Bed,
                _ => matches!(k, ObjectKind::Chair | ObjectKind::Sofa),
            };
            let free: Vec<usize> = (0..objects_placed)
                .filter(|&k| !taken[k] && fits(scene.objects[k].kind))
                .collect();
            if free.is_empty() {
                mode = PersonMode::Walk;
            } else {
                let k = free[int(&mut rng, (0, free.len() - 1))];
                taken[k] = true;
                let o = &scene.objects[k];
                start = Some((o.pos.0, o.pos.1));
            }
        }
        let mut heading = None;
        if matches!(mode, PersonMode::Walk | PersonMode::Stand) {
            for _ in 0..START_TRIES {
                let p = (rng.range(0.0, w), rng.range(0.0, d));
                if walk.point(p) {
                    start = Some(p);
                    break;
                }
            }
            if start.is_none() {
                return Err((
                    Reason::WalkStart,
                    format!("people[{j}] found no walkable start in {START_TRIES} tries"),
                ));
            }
            if mode == PersonMode::Stand {
                heading = Some(rng.range(0.0, TAU));
            }
        }
        people.push((mode, heading));
        starts.push(start.expect("a start for every person"));
    }
    scene.people = starts
        .iter()
        .map(|&p| Person {
            id: None,
            start: Some(p),
        })
        .collect();

    // §9: the Room Sim seed is the stream's next draw.
    let roomsim_seed = rng.next_u64();
    let rs = run_config(roomsim_seed, seconds, floor, ceiling, &people);
    Ok(Draw {
        scene,
        rs,
        l_shape,
        wall_material,
        objects_drawn,
        objects_placed,
        openings_drawn,
        openings_placed,
        modes: people.iter().map(|&(m, _)| m).collect(),
    })
}

/// §2.6: `Scene::validate()`, `Simulator::new`, and Locate's refusals
/// before its calibration, in `run_roomsim`'s order (its scene-free
/// checks, the resolved config, the 2-D geometry, the locator and
/// `tx_period`), with Locate's defaults.
pub fn check(scene: &Scene, rs: &RoomSimConfig) -> Result<(), (Reason, String)> {
    let errors = scene.validate();
    if let Some(e) = errors.first() {
        return Err((Reason::Scene, e.to_string()));
    }
    let sim = Simulator::new(scene, rs.clone()).map_err(|e| (Reason::RoomSim, e.to_string()))?;
    let locate = |e: saans_locate::pipeline::PipelineError| (Reason::Locate, e.to_string());
    let cfg = LocateConfig::default();
    check_roomsim_config(rs, &cfg).map_err(locate)?;
    let cfg = resolve_config(rs, &cfg).map_err(locate)?;
    let geometry = person_geometry(scene, &cfg).map_err(locate)?;
    let rate_hz = base_rate_hz(&sim);
    let person =
        PersonLocator::new(geometry, &cfg, rate_hz, saans_locate::LABEL).map_err(locate)?;
    check_tx_period(&sim, cfg.window_s, rate_hz, person.min_samples()).map_err(locate)?;
    Ok(())
}

/// Scene `i` (§2): attempts `0..100` until one is drawn and passes §2.6.
pub fn generate(seed: u64, i: usize, seconds: u32) -> Result<Generated, Exhausted> {
    let mut rejections = Rejections::default();
    let mut last = String::new();
    for a in 0..MAX_ATTEMPTS {
        let drawn = draw(seed, i, a, seconds).and_then(|d| check(&d.scene, &d.rs).map(|()| d));
        match drawn {
            Ok(d) => {
                return Ok(Generated {
                    index: i,
                    scene: d.scene,
                    rs: d.rs,
                    attempts: a + 1,
                    l_shape: d.l_shape,
                    wall_material: d.wall_material,
                    objects_drawn: d.objects_drawn,
                    objects_placed: d.objects_placed,
                    openings_drawn: d.openings_drawn,
                    openings_placed: d.openings_placed,
                    modes: d.modes,
                    rejections,
                })
            }
            Err((reason, why)) => {
                rejections.add(reason);
                last = why;
            }
        }
    }
    Err(Exhausted {
        index: i,
        rejections,
        last,
    })
}

/// §9: scene `i`'s split draw, one `uniform()` of
/// `Rng::stream(seed, (1 << 40) | i)`.
pub fn split_draw(seed: u64, i: usize) -> f64 {
    Rng::stream(seed, (1u64 << 40) | i as u64).uniform()
}

/// Every §2 distribution, for the manifest's `generator` block.
pub fn generator_json() -> Value {
    let wall: Vec<Value> = WALL_MATERIALS
        .iter()
        .map(|&(m, p, (lo, hi))| json!({"material": wall_material_name(m), "p": p, "thickness_m": [lo, hi]}))
        .collect();
    let surface = |xs: &[(SurfaceMaterial, f64)]| -> Vec<Value> {
        xs.iter()
            .map(|&(m, p)| json!({"material": m.name(), "p": p}))
            .collect()
    };
    let furniture: Vec<Value> = FURNITURE
        .iter()
        .map(|f| {
            json!({
                "kind": kind_name(f.kind),
                "p": f.p,
                "size_m": [[f.size[0].0, f.size[0].1], [f.size[1].0, f.size[1].1], [f.size[2].0, f.size[2].1]],
                "material": f.materials.iter().map(|&(m, p)| json!({"material": object_material_name(m), "p": p})).collect::<Vec<_>>(),
            })
        })
        .collect();
    let opening = |o: &Opening| {
        json!({
            "kind": kind_name(o.kind),
            "size_m": [o.size.0, o.size.1, o.size.2],
            "pos_z_m": o.z,
            "material": object_material_name(o.material),
        })
    };
    let modes: Vec<Value> = MODES
        .iter()
        .map(|&(m, p)| json!({"mode": m.name(), "p": p}))
        .collect();
    json!({
        "label": "SIM",
        "rng": "saans_sim::rng::Rng: scene i, attempt a draws from Rng::stream(seed, (i << 8) | a) in spec/dataset.md §2's order; the Room Sim seed is that stream's next next_u64() after acceptance; the split draw is one uniform() of Rng::stream(seed, (1 << 40) | i)",
        "max_attempts": MAX_ATTEMPTS,
        "room": {
            "w_m": [ROOM_W_M.0, ROOM_W_M.1],
            "d_m": [ROOM_D_M.0, ROOM_D_M.1],
            "h_m": [ROOM_H_M.0, ROOM_H_M.1],
            "round_m": 0.01,
            "l_shape_p": L_SHAPE_P,
            "l_corner": "uniform among the four",
            "l_cut_frac": [L_CUT_FRAC.0, L_CUT_FRAC.1],
        },
        "walls": {
            "per_edge": "one wall per footprint edge, centreline on the edge, the room's height",
            "material": wall,
            "thickness_round_m": 0.001,
            "floor_material": surface(&FLOOR_MATERIALS),
            "ceiling_material": surface(&CEILING_MATERIALS),
            "inner_face": "t/2 inside the edge",
        },
        "furniture": {
            "count": [FURNITURE_COUNT.0, FURNITURE_COUNT.1],
            "kinds": furniture,
            "size_round_m": 0.01,
            "pos_z": "size.z / 2",
            "yaw_deg": [0, 90, 180, 270],
            "centre": "uniform in the bounding box",
            "face_gap_m": FURNITURE_FACE_GAP_M,
            "object_gap_m": FURNITURE_GAP_M,
            "tries": PLACE_TRIES,
            "draw_order": "kind, size x, y, z, material (when more than one); per try yaw, centre x, centre y",
        },
        "openings": {
            "door_p": DOOR_P,
            "door": opening(&DOOR),
            "windows": [WINDOW_COUNT.0, WINDOW_COUNT.1],
            "window": opening(&WINDOW),
            "min_edge_m": OPENING_MIN_EDGE_M,
            "end_gap_m": OPENING_END_M,
            "gap_m": OPENING_GAP_M,
            "yaw": "the edge's direction, atan2(dy, dx) in degrees",
            "edge": "in proportion to length",
            "tries": PLACE_TRIES,
        },
        "nodes": {
            "count": [NODE_COUNT.0, NODE_COUNT.1],
            "inset_m": "max(t/2, 0.05) + 0.05",
            "edge": "in proportion to length",
            "end_gap_m": NODE_END_M,
            "z_m": [NODE_Z_M.0, NODE_Z_M.1],
            "z_round_m": 0.01,
            "xy_gap_m": NODE_GAP_M,
            "tries": NODE_TRIES,
        },
        "people": {
            "count": [PEOPLE_COUNT.0, PEOPLE_COUNT.1],
            "mode": modes,
            "lie": "at the centre of a free bed (uniform among the free ones), else walk",
            "sit": "at the centre of a free chair or sofa (uniform among the free ones), else walk",
            "walk": "start uniform in the bounding box, kept when Room Sim's Walkability::point accepts it; people_mode walk",
            "stand": "as walk, people_mode stationary, start_heading_rad ~ U[0, 2 pi)",
            "start_tries": START_TRIES,
        },
    })
}

pub fn kind_name(k: ObjectKind) -> &'static str {
    match k {
        ObjectKind::Bed => "bed",
        ObjectKind::Table => "table",
        ObjectKind::Chair => "chair",
        ObjectKind::Almirah => "almirah",
        ObjectKind::Sofa => "sofa",
        ObjectKind::Door => "door",
        ObjectKind::Window => "window",
    }
}

fn object_material_name(m: ObjectMaterial) -> &'static str {
    match m {
        ObjectMaterial::Wood => "wood",
        ObjectMaterial::Metal => "metal",
        ObjectMaterial::Fabric => "fabric",
        ObjectMaterial::Glass => "glass",
    }
}

#[cfg(test)]
mod tests;
