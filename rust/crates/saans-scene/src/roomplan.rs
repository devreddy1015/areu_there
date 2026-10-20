//! Import an Apple RoomPlan export into a [`Scene`].
//!
//! RoomPlan (iOS 16+) scans a room with the LiDAR sensor and exports a
//! `CapturedRoom` as JSON: walls, doors, windows, openings and a list of
//! recognised objects, each with `dimensions` and a 4x4 `transform`.
//!
//! # What RoomPlan does not tell us
//!
//! Three things this crate needs are simply absent from the export, and
//! the importer guesses them and says so in [`RoomPlanWarning`] rather
//! than inventing them silently:
//!
//! * **Materials.** RoomPlan reports geometry, not what anything is made
//!   of. Walls become [`WallMaterial::Drywall`] and objects get a default
//!   by category. For a concrete or brick room this is wrong in the
//!   direction that matters (concrete reflects far more), so edit the
//!   file before trusting a simulation built on it.
//! * **Wall thickness.** Surfaces are planar; `dimensions.z` is ~0.
//!   Every wall gets [`DEFAULT_WALL_THICKNESS_M`].
//! * **ESP32 nodes.** There are none in a scan. The imported scene has an
//!   empty `nodes` list and will NOT pass [`Scene::validate`] until you
//!   add them, which is deliberate: a scene without radios cannot drive
//!   anything.
//!
//! # Frames
//!
//! ARKit is right-handed with +y up, so the floor is the x-z plane. This
//! crate is right-handed with +z up. The map is
//! `(x, y, z)_arkit -> (x, -z, y)_scene`, which preserves handedness, and
//! then everything is translated so the room's minimum corner is the
//! origin and the floor is z = 0.

use crate::scene::{
    Node, Object, ObjectKind, ObjectMaterial, Point, Room, Scene, Wall, WallMaterial, DEFAULT_WALL_THICKNESS_M,
    SCHEMA_VERSION,
};
use serde_json::Value;
use std::fmt;

/// A RoomPlan surface or object the importer could not take at face value.
#[derive(Debug, Clone, PartialEq)]
pub enum RoomPlanWarning {
    /// Every wall material is a guess; RoomPlan does not measure them.
    MaterialsAssumed { walls: usize, objects: usize },
    /// Wall thickness is not in the export.
    ThicknessAssumed { walls: usize, thickness_m: f64 },
    /// A scan has no radios in it.
    NoNodes,
    /// A category this scene format has no box for (a toilet, a stove).
    UnsupportedCategory { category: String, count: usize },
    /// An `openings` entry: a hole in a wall, not an object. Dropped,
    /// because a gap is the absence of a wall and the scene format has no
    /// way to subtract one yet.
    OpeningDropped { count: usize },
    /// An element whose transform or dimensions could not be read.
    Unreadable { what: String, index: usize },
    /// The floor height had to be inferred from the walls.
    FloorInferred { z_arkit: f64 },
}

impl fmt::Display for RoomPlanWarning {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::MaterialsAssumed { walls, objects } => write!(
                f,
                "materials are assumed, not measured: {walls} wall(s) set to drywall and {objects} object(s) to a default for their category - edit them before simulating a concrete or brick room"
            ),
            Self::ThicknessAssumed { walls, thickness_m } => {
                write!(f, "RoomPlan surfaces are planar, so all {walls} wall(s) were given thickness {thickness_m} m")
            }
            Self::NoNodes => write!(f, "a scan contains no ESP32 nodes: add them to `nodes` before this scene will load"),
            Self::UnsupportedCategory { category, count } => {
                write!(f, "dropped {count} object(s) of category \"{category}\": the scene format has no box for it")
            }
            Self::OpeningDropped { count } => {
                write!(f, "dropped {count} opening(s): the scene format cannot cut a hole in a wall yet")
            }
            Self::Unreadable { what, index } => write!(f, "{what}[{index}] has no readable transform or dimensions; skipped"),
            Self::FloorInferred { z_arkit } => {
                write!(f, "no floor in the export; took the floor from the lowest wall edge (ARKit y = {z_arkit:.3} m)")
            }
        }
    }
}

/// What an import produced.
#[derive(Debug, Clone)]
pub struct RoomPlanImport {
    /// The scene. Its `nodes` list is empty: add the ESP32s yourself.
    pub scene: Scene,
    /// Everything the importer had to assume or drop, in reading order.
    pub warnings: Vec<RoomPlanWarning>,
}

/// Why a RoomPlan file could not be imported at all.
#[derive(Debug)]
pub enum RoomPlanError {
    Json(serde_json::Error),
    /// Valid JSON, but not a `CapturedRoom`.
    NotACapturedRoom,
    /// A `CapturedRoom` with no walls: nothing to build a room from.
    NoWalls,
}

impl fmt::Display for RoomPlanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Json(e) => write!(f, "not JSON: {e}"),
            Self::NotACapturedRoom => {
                write!(f, "no \"walls\" array: this does not look like a RoomPlan CapturedRoom export")
            }
            Self::NoWalls => write!(f, "the export has no walls, so there is no room to import"),
        }
    }
}

impl std::error::Error for RoomPlanError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Json(e) => Some(e),
            _ => None,
        }
    }
}

/// A 4x4 transform, column-major, as Apple writes `simd_float4x4`.
#[derive(Debug, Clone, Copy)]
struct Transform([f64; 16]);

impl Transform {
    /// ARKit translation: the fourth column.
    fn translation(&self) -> (f64, f64, f64) {
        (self.0[12], self.0[13], self.0[14])
    }

    /// The element's local +x in ARKit world coordinates: the first column.
    fn local_x(&self) -> (f64, f64, f64) {
        (self.0[0], self.0[1], self.0[2])
    }
}

/// Read Apple's `simd_float4x4`, which appears either as 16 flat numbers
/// (column-major) or as 4 columns of 4.
fn parse_transform(v: &Value) -> Option<Transform> {
    let mut out = [0.0f64; 16];
    if let Some(arr) = v.as_array() {
        if arr.len() == 16 {
            for (i, x) in arr.iter().enumerate() {
                out[i] = x.as_f64()?;
            }
            return Some(Transform(out));
        }
        if arr.len() == 4 {
            for (c, col) in arr.iter().enumerate() {
                let col = col.as_array()?;
                if col.len() != 4 {
                    return None;
                }
                for (r, x) in col.iter().enumerate() {
                    out[c * 4 + r] = x.as_f64()?;
                }
            }
            return Some(Transform(out));
        }
        return None;
    }
    // Some encoders wrap it: {"columns": [[..],[..],[..],[..]]}
    v.get("columns").and_then(parse_transform)
}

/// Read a `simd_float3`: `[x, y, z]`, or `{"x":_, "y":_, "z":_}`.
fn parse_vec3(v: &Value) -> Option<(f64, f64, f64)> {
    if let Some(a) = v.as_array() {
        if a.len() == 3 {
            return Some((a[0].as_f64()?, a[1].as_f64()?, a[2].as_f64()?));
        }
        return None;
    }
    Some((v.get("x")?.as_f64()?, v.get("y")?.as_f64()?, v.get("z")?.as_f64()?))
}

/// Read a RoomPlan category, written either as `"table"` or as the
/// single-key object `{"table": {}}` that Swift enums encode to.
fn parse_category(v: Option<&Value>) -> Option<String> {
    let v = v?;
    if let Some(s) = v.as_str() {
        return Some(s.to_string());
    }
    v.as_object()?.keys().next().cloned()
}

/// ARKit (x, y, z) -> scene (x, y, z) before the origin shift.
fn to_scene(p: (f64, f64, f64)) -> (f64, f64, f64) {
    (p.0, -p.2, p.1)
}

/// Round to the nearest micrometre, and turn -0.0 into 0.0.
///
/// Positions come out of a 4x4 transform and a `cos`/`atan2` pair, so an
/// axis-aligned wall arrives at 9.2e-17 rather than 0, a 3 m room
/// measures 3.0000000000000004 m, and a rotation of zero prints as -0.
/// A scene file is read and edited by people and diffed in review; a
/// micrometre is orders of magnitude below what a LiDAR scan resolves,
/// so the noise is dropped rather than published.
fn round_m(x: f64) -> f64 {
    let r = (x * 1e6).round() / 1e6;
    // -0.0 == 0.0, so this normalises the sign as well as the value.
    if r == 0.0 {
        0.0
    } else {
        r
    }
}

fn round_point(p: Point) -> Point {
    (round_m(p.0), round_m(p.1))
}

fn default_material(kind: ObjectKind) -> ObjectMaterial {
    match kind {
        ObjectKind::Bed | ObjectKind::Sofa => ObjectMaterial::Fabric,
        ObjectKind::Table | ObjectKind::Chair | ObjectKind::Almirah | ObjectKind::Door => ObjectMaterial::Wood,
        ObjectKind::Window => ObjectMaterial::Glass,
    }
}

/// RoomPlan's object categories that this scene format has a box for.
fn map_category(category: &str) -> Option<ObjectKind> {
    match category {
        "bed" => Some(ObjectKind::Bed),
        "table" => Some(ObjectKind::Table),
        "chair" => Some(ObjectKind::Chair),
        "sofa" => Some(ObjectKind::Sofa),
        // RoomPlan's "storage" is a wardrobe/cupboard: an almirah.
        "storage" => Some(ObjectKind::Almirah),
        _ => None,
    }
}

/// One element of the export, already in scene axes (pre-shift).
struct Element {
    /// Centre, scene axes, metres.
    pos: (f64, f64, f64),
    /// (along local x, along local y, up), metres.
    size: (f64, f64, f64),
    yaw_deg: f64,
}

fn read_element(v: &Value) -> Option<Element> {
    let t = parse_transform(v.get("transform")?)?;
    let d = parse_vec3(v.get("dimensions")?)?;
    let pos = to_scene(t.translation());
    // ARKit dimensions are (local x, up, local z); scene wants (x, y, up).
    let size = (d.0, d.2, d.1);
    let lx = t.local_x();
    // Local +x projected onto the scene floor plane.
    let yaw_deg = round_m((-lx.2).atan2(lx.0).to_degrees());
    if ![pos.0, pos.1, pos.2, size.0, size.1, size.2, yaw_deg].iter().all(|x| x.is_finite()) {
        return None;
    }
    Some(Element { pos, size, yaw_deg })
}

/// Convert a RoomPlan `CapturedRoom` JSON export into a scene.
///
/// `name` becomes the scene's name. The returned scene has no nodes; add
/// them before loading it.
pub fn import_roomplan_str(name: &str, text: &str) -> Result<RoomPlanImport, RoomPlanError> {
    let root: Value = serde_json::from_str(text).map_err(RoomPlanError::Json)?;
    let walls_json = root.get("walls").and_then(Value::as_array).ok_or(RoomPlanError::NotACapturedRoom)?;
    let mut warnings = Vec::new();

    // --- walls -------------------------------------------------------
    struct RawWall {
        a: (f64, f64),
        b: (f64, f64),
        top: f64,
        bottom: f64,
    }
    let mut raw_walls = Vec::new();
    for (i, w) in walls_json.iter().enumerate() {
        let Some(e) = read_element(w) else {
            warnings.push(RoomPlanWarning::Unreadable { what: "walls".into(), index: i });
            continue;
        };
        // A wall's local x runs along its length.
        let half = e.size.0 / 2.0;
        let (c, s) = (e.yaw_deg.to_radians().cos(), e.yaw_deg.to_radians().sin());
        let a = (e.pos.0 - half * c, e.pos.1 - half * s);
        let b = (e.pos.0 + half * c, e.pos.1 + half * s);
        raw_walls.push(RawWall { a, b, top: e.pos.2 + e.size.2 / 2.0, bottom: e.pos.2 - e.size.2 / 2.0 });
    }
    if raw_walls.is_empty() {
        return Err(RoomPlanError::NoWalls);
    }

    // --- floor level and origin --------------------------------------
    // Prefer an explicit floor; otherwise the lowest wall edge is it.
    let floor_z = match root.get("floors").and_then(Value::as_array).and_then(|fs| {
        fs.iter().filter_map(read_element).map(|e| e.pos.2).fold(None::<f64>, |acc, z| Some(acc.map_or(z, |a: f64| a.min(z))))
    }) {
        Some(z) => z,
        None => {
            let z = raw_walls.iter().map(|w| w.bottom).fold(f64::INFINITY, f64::min);
            warnings.push(RoomPlanWarning::FloorInferred { z_arkit: z });
            z
        }
    };
    let min_x = raw_walls.iter().flat_map(|w| [w.a.0, w.b.0]).fold(f64::INFINITY, f64::min);
    let min_y = raw_walls.iter().flat_map(|w| [w.a.1, w.b.1]).fold(f64::INFINITY, f64::min);
    let max_x = raw_walls.iter().flat_map(|w| [w.a.0, w.b.0]).fold(f64::NEG_INFINITY, f64::max);
    let max_y = raw_walls.iter().flat_map(|w| [w.a.1, w.b.1]).fold(f64::NEG_INFINITY, f64::max);
    let shift = |p: (f64, f64)| -> Point { (p.0 - min_x, p.1 - min_y) };
    let shift3 = |p: (f64, f64, f64)| (p.0 - min_x, p.1 - min_y, p.2 - floor_z);

    let room_h = round_m(raw_walls.iter().map(|w| w.top - floor_z).fold(0.0, f64::max));
    let room = Room { w_m: round_m(max_x - min_x), d_m: round_m(max_y - min_y), h_m: room_h, polygon: None };

    let walls: Vec<Wall> = raw_walls
        .iter()
        .map(|w| Wall {
            from: round_point(shift(w.a)),
            to: round_point(shift(w.b)),
            material: WallMaterial::Drywall,
            thickness_m: DEFAULT_WALL_THICKNESS_M,
            height_m: Some(round_m((w.top - floor_z).min(room_h))),
        })
        .collect();
    warnings.push(RoomPlanWarning::ThicknessAssumed { walls: walls.len(), thickness_m: DEFAULT_WALL_THICKNESS_M });

    // --- doors, windows, objects -------------------------------------
    let mut objects: Vec<Object> = Vec::new();
    let mut unsupported: Vec<(String, usize)> = Vec::new();

    for (key, kind) in [("doors", ObjectKind::Door), ("windows", ObjectKind::Window)] {
        let Some(arr) = root.get(key).and_then(Value::as_array) else { continue };
        for (i, v) in arr.iter().enumerate() {
            let Some(e) = read_element(v) else {
                warnings.push(RoomPlanWarning::Unreadable { what: key.into(), index: i });
                continue;
            };
            let p = shift3(e.pos);
            // A door or window is planar in the export; give it enough
            // depth to be a box, matching the wall it sits in.
            let depth = if e.size.1 > 1e-3 { e.size.1 } else { DEFAULT_WALL_THICKNESS_M };
            objects.push(Object {
                kind,
                id: None,
                pos: (round_m(p.0), round_m(p.1), round_m(p.2)),
                size: (round_m(e.size.0), round_m(depth), round_m(e.size.2)),
                yaw_deg: e.yaw_deg,
                material: default_material(kind),
            });
        }
    }

    if let Some(arr) = root.get("openings").and_then(Value::as_array) {
        if !arr.is_empty() {
            warnings.push(RoomPlanWarning::OpeningDropped { count: arr.len() });
        }
    }

    if let Some(arr) = root.get("objects").and_then(Value::as_array) {
        for (i, v) in arr.iter().enumerate() {
            let category = parse_category(v.get("category")).unwrap_or_default();
            let Some(kind) = map_category(&category) else {
                match unsupported.iter_mut().find(|(c, _)| *c == category) {
                    Some((_, n)) => *n += 1,
                    None => unsupported.push((category, 1)),
                }
                continue;
            };
            let Some(e) = read_element(v) else {
                warnings.push(RoomPlanWarning::Unreadable { what: "objects".into(), index: i });
                continue;
            };
            let p = shift3(e.pos);
            objects.push(Object {
                kind,
                id: None,
                pos: (round_m(p.0), round_m(p.1), round_m(p.2)),
                size: (round_m(e.size.0), round_m(e.size.1), round_m(e.size.2)),
                yaw_deg: e.yaw_deg,
                material: default_material(kind),
            });
        }
    }
    for (category, count) in unsupported {
        warnings.push(RoomPlanWarning::UnsupportedCategory { category, count });
    }

    warnings.push(RoomPlanWarning::MaterialsAssumed { walls: walls.len(), objects: objects.len() });
    warnings.push(RoomPlanWarning::NoNodes);

    let scene = Scene {
        schema_version: SCHEMA_VERSION,
        name: name.to_string(),
        room,
        walls,
        objects,
        nodes: Vec::<Node>::new(),
        people: Vec::new(),
        phone: None,
    };
    Ok(RoomPlanImport { scene, warnings })
}
