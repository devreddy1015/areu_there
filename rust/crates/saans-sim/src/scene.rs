//! Scene files: the room as DEFINED, parsed and checked.
//!
//! A scene says where the walls, the furniture and the sensor nodes are.
//! It is typed in, measured with a tape, or imported from a LiDAR scan.
//! None of it comes from WiFi — an ESP32 at 20 MHz on one antenna cannot
//! measure a room's dimensions and cannot recognise a bed — so anything a
//! viewer draws from this file is SCENE data, labelled separately from a
//! SIM estimate and from a LIVE measurement.
//!
//! The format is `scenes/scene.schema.json` (JSON Schema draft 2020-12),
//! the examples are `scenes/*.json`, and `wifisense/scene.py` is the same
//! loader in Python. This module is additive: it changes nothing that was
//! already public in this crate, and the simulator does not read scenes.
//!
//! # Axes, and the one collision that matters
//!
//! ```text
//! scene x -> width_m    floor plan, metres, origin at a floor corner
//! scene y -> depth_m    the SECOND HORIZONTAL axis of the floor plan
//! scene z -> height_m   ELEVATION, up
//! ```
//!
//! `wifisense/localize/geometry.py`'s `Room` is a 2-D plan-view type
//! whose fields are `width_m` and `height_m`, and whose `height_m` is the
//! second horizontal axis, not elevation. This crate's
//! [`crate::scenario::Room`] has the same convention in `size_m`: a pair
//! of HORIZONTAL extents. So:
//!
//! ```text
//! scene room.depth_m   -> Room.height_m (Python), size_m.1 (here)
//! scene room.height_m  -> dropped by both 2-D types; the box model's
//!                         Vertical::height_m (Scene::to_sim_box_room)
//! ```
//!
//! [`Scene::sim_room_size_m`] is the only place that mapping is made in
//! Rust. Using the elevation instead turns a 4.2 x 3.6 m bedroom into a
//! 4.2 x 2.7 m one that still loads and still runs, which is why the test
//! `depth_not_elevation_maps_to_the_simulator_room` gives every example
//! scene three different numbers.
//!
//! # Why the JSON parser is written out here
//!
//! `saans-sim` depends on `serde` and not on `serde_json`, and this work
//! was not allowed to add a dependency. The types below derive
//! `Serialize` and `Deserialize` as usual, so any crate that does have a
//! serde data format (the gateway has `serde_json`) can use them
//! directly; the loader in this crate reads the file with the small
//! reader in [`json`] instead. The two paths are kept in step by
//! construction: every field name, default and rename lives on the struct
//! and is read by [`Scene::from_json`] from the same name.
//!
//! Validation is in [`Scene::from_json`], not in the derive, because the
//! useful checks are not type checks: a bed inside the room, two nodes
//! that are not the same point, a polygon whose bounding box matches the
//! width and depth it was declared with. Nothing here panics; everything
//! returns [`SceneError`] with the path of the offending field.

use crate::scenario::{Node as SimNode, Point, Room as SimRoom, Vertical};
use serde::{Deserialize, Serialize};
use std::fmt;

/// The only format version this loader reads.
///
/// A loader that does not know a version refuses the file. The version is
/// for the change a schema cannot catch: a field that keeps its name and
/// changes its meaning.
pub const SCHEMA_VERSION: u32 = 1;

/// Containment slack, metres. A door leaf sits exactly in the plane of
/// its wall and a window's outer face is exactly on the boundary, so
/// "inside" has to admit the boundary itself. Floating-point slack, not a
/// licence to place furniture outside the room.
pub const CONTAIN_TOL_M: f64 = 1e-6;

/// Two nodes closer than this are one point: the link between them has
/// zero length, and the weight model's `1/sqrt(d)` is not a number there.
pub const NODE_MIN_SEPARATION_M: f64 = 1e-6;

/// 802.11 channel 6: the carrier the rest of this crate already uses.
pub const MATERIAL_REFERENCE_GHZ: f64 = 2.437;

/// Vacuum permittivity (CODATA 2018), for `eps'' = sigma / (2 pi f eps0)`.
pub const EPSILON_0_F_PER_M: f64 = 8.854_187_812_8e-12;

/// Nesting limit for the JSON reader. A scene is four levels deep; the
/// limit is here so a hostile file cannot recurse the parser into the
/// stack guard.
const MAX_JSON_DEPTH: u32 = 32;

// ---------------------------------------------------------------------------
// Errors
// ---------------------------------------------------------------------------

/// A scene that cannot be loaded, and where the problem is.
///
/// `path` is the JSON path of the offending value (`scene.nodes[2].pos_m`),
/// which is the part a person needs to fix the file.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SceneError {
    pub path: String,
    pub message: String,
}

impl SceneError {
    fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}: {}", self.path, self.message)
    }
}

impl std::error::Error for SceneError {}

// ---------------------------------------------------------------------------
// Materials: ITU-R P.2040 Table 3
// ---------------------------------------------------------------------------

/// ITU-R P.2040 Table 3 coefficients for one material.
///
/// ```text
/// eps_r' = a * f_GHz^b          sigma = c * f_GHz^d   [S/m]
/// ```
///
/// Stored as coefficients rather than as values at one frequency, so the
/// table stays checkable and a 5 GHz radio gets the right numbers.
/// `scenes/materials.json` carries the same coefficients and their values
/// at 2.437 GHz; `materials_json_matches_the_table` recomputes those from
/// here.
/// (`Serialize` only: the table is code, not input. There is nothing a
/// file could say about ITU-R P.2040 that this crate should believe.)
#[derive(Debug, Clone, Copy, PartialEq, Serialize)]
pub struct MaterialConstants {
    pub name: &'static str,
    pub a: f64,
    pub b: f64,
    pub c: f64,
    pub d: f64,
    /// The Recommendation's stated frequency range for this row, GHz.
    /// Outside it the fit is an extrapolation.
    pub valid_ghz: (f64, f64),
}

impl MaterialConstants {
    pub fn epsilon_r(&self, f_ghz: f64) -> f64 {
        self.a * f_ghz.powf(self.b)
    }

    pub fn sigma_s_per_m(&self, f_ghz: f64) -> f64 {
        self.c * f_ghz.powf(self.d)
    }

    /// `sigma / (2 pi f eps0)`.
    pub fn epsilon_imag(&self, f_ghz: f64) -> f64 {
        self.sigma_s_per_m(f_ghz) / (2.0 * std::f64::consts::PI * f_ghz * 1e9 * EPSILON_0_F_PER_M)
    }

    /// `|Gamma|` at normal incidence, air onto a half-space of this material.
    ///
    /// `Gamma = (1 - sqrt(eta)) / (1 + sqrt(eta))`, `eta = eps' - j eps''`.
    ///
    /// ONE INTERFACE, NOT A WALL. A real wall is a slab with two
    /// interfaces and internal reflections (ITU-R P.2040 section 5.2),
    /// and for a sheet as thin as drywall the slab response differs
    /// substantially and swings with thickness. The number is here
    /// because it is the quantity [`crate::scenario::Room::wall_reflection`]
    /// stands for: concrete comes out at 0.395 against that field's
    /// long-standing default of 0.4, which says the simulator's scalar is
    /// a concrete-like wall. It is not the reflectivity of a finished wall.
    pub fn reflection_normal(&self, f_ghz: f64) -> f64 {
        let eta = (self.epsilon_r(f_ghz), -self.epsilon_imag(f_ghz));
        let root = complex_sqrt(eta);
        let num = (1.0 - root.0, -root.1);
        let den = (1.0 + root.0, root.1);
        let den_sq = den.0 * den.0 + den.1 * den.1;
        let re = (num.0 * den.0 + num.1 * den.1) / den_sq;
        let im = (num.1 * den.0 - num.0 * den.1) / den_sq;
        (re * re + im * im).sqrt()
    }

    pub fn in_range(&self, f_ghz: f64) -> bool {
        f_ghz >= self.valid_ghz.0 && f_ghz <= self.valid_ghz.1
    }
}

/// Principal square root of `(re, im)`, without a complex-number crate.
fn complex_sqrt(z: (f64, f64)) -> (f64, f64) {
    let r = (z.0 * z.0 + z.1 * z.1).sqrt();
    if r == 0.0 {
        return (0.0, 0.0);
    }
    let re = ((r + z.0) / 2.0).sqrt();
    let im = ((r - z.0) / 2.0).sqrt().copysign(z.1);
    (re, im)
}

pub const CONCRETE: MaterialConstants = MaterialConstants {
    name: "concrete",
    a: 5.24,
    b: 0.0,
    c: 0.0462,
    d: 0.7822,
    valid_ghz: (1.0, 100.0),
};
pub const BRICK: MaterialConstants = MaterialConstants {
    name: "brick",
    a: 3.91,
    b: 0.0,
    c: 0.0238,
    d: 0.16,
    valid_ghz: (1.0, 10.0),
};
pub const DRYWALL: MaterialConstants = MaterialConstants {
    name: "drywall",
    a: 2.73,
    b: 0.0,
    c: 0.0085,
    d: 0.9395,
    valid_ghz: (1.0, 100.0),
};
pub const GLASS: MaterialConstants = MaterialConstants {
    name: "glass",
    a: 6.31,
    b: 0.0,
    c: 0.0036,
    d: 1.3394,
    valid_ghz: (0.1, 100.0),
};
pub const WOOD: MaterialConstants = MaterialConstants {
    name: "wood",
    a: 1.99,
    b: 0.0,
    c: 0.0047,
    d: 1.0718,
    valid_ghz: (0.001, 100.0),
};
pub const METAL: MaterialConstants = MaterialConstants {
    name: "metal",
    a: 1.0,
    b: 0.0,
    c: 1.0e7,
    d: 0.0,
    valid_ghz: (1.0, 100.0),
};

/// Every material with an ITU-R P.2040 row. `fabric` is not here: see
/// [`ObjectMaterial::constants`].
pub const MATERIALS: [MaterialConstants; 6] = [CONCRETE, BRICK, DRYWALL, GLASS, WOOD, METAL];

/// What a wall is made of. Every wall material has a P.2040 row.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum WallMaterial {
    Concrete,
    Brick,
    Drywall,
    Glass,
    Wood,
}

impl WallMaterial {
    pub fn constants(&self) -> MaterialConstants {
        match self {
            WallMaterial::Concrete => CONCRETE,
            WallMaterial::Brick => BRICK,
            WallMaterial::Drywall => DRYWALL,
            WallMaterial::Glass => GLASS,
            WallMaterial::Wood => WOOD,
        }
    }

    pub fn as_str(&self) -> &'static str {
        self.constants().name
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "concrete" => WallMaterial::Concrete,
            "brick" => WallMaterial::Brick,
            "drywall" => WallMaterial::Drywall,
            "glass" => WallMaterial::Glass,
            "wood" => WallMaterial::Wood,
            _ => return None,
        })
    }
}

/// What a piece of furniture is made of.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectMaterial {
    Wood,
    Metal,
    Fabric,
    Glass,
}

impl ObjectMaterial {
    /// Electrical constants, or `None` for fabric.
    ///
    /// ITU-R P.2040 Table 3 has no row for fabric and no sourced value
    /// was available, so none is given rather than invented. A sofa
    /// carries `fabric` as a LABEL for the viewer, and a physics layer
    /// has to refuse it until someone supplies a value with a citation.
    pub fn constants(&self) -> Option<MaterialConstants> {
        Some(match self {
            ObjectMaterial::Wood => WOOD,
            ObjectMaterial::Metal => METAL,
            ObjectMaterial::Glass => GLASS,
            ObjectMaterial::Fabric => return None,
        })
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            ObjectMaterial::Wood => "wood",
            ObjectMaterial::Metal => "metal",
            ObjectMaterial::Fabric => "fabric",
            ObjectMaterial::Glass => "glass",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "wood" => ObjectMaterial::Wood,
            "metal" => ObjectMaterial::Metal,
            "fabric" => ObjectMaterial::Fabric,
            "glass" => ObjectMaterial::Glass,
            _ => return None,
        })
    }
}

/// The seven things a scene can hold. `Almirah` is a free-standing
/// wardrobe; `Door` and `Window` are openings drawn inside their wall.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ObjectType {
    Bed,
    Table,
    Chair,
    Almirah,
    Sofa,
    Door,
    Window,
}

impl ObjectType {
    pub fn as_str(&self) -> &'static str {
        match self {
            ObjectType::Bed => "bed",
            ObjectType::Table => "table",
            ObjectType::Chair => "chair",
            ObjectType::Almirah => "almirah",
            ObjectType::Sofa => "sofa",
            ObjectType::Door => "door",
            ObjectType::Window => "window",
        }
    }

    fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "bed" => ObjectType::Bed,
            "table" => ObjectType::Table,
            "chair" => ObjectType::Chair,
            "almirah" => ObjectType::Almirah,
            "sofa" => ObjectType::Sofa,
            "door" => ObjectType::Door,
            "window" => ObjectType::Window,
            _ => return None,
        })
    }
}

// ---------------------------------------------------------------------------
// The scene types
// ---------------------------------------------------------------------------

/// The floor plan's bounds.
///
/// `width_m` x `depth_m` is the clear interior floor; `height_m` is floor
/// to ceiling. When `polygon_m` is present its bounding box is the same
/// `width_m` x `depth_m`, so a reader that cannot draw a polygon still
/// gets the extent right.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneRoom {
    pub width_m: f64,
    /// The second HORIZONTAL axis. Maps to `Room.height_m` in the Python
    /// localization types and to `size_m.1` in [`crate::scenario::Room`].
    pub depth_m: f64,
    /// ELEVATION. Neither 2-D type has a field for it; the box model's
    /// [`Vertical::height_m`] is it.
    pub height_m: f64,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub polygon_m: Vec<[f64; 2]>,
}

impl SceneRoom {
    /// The floor outline, counter-clockwise: the polygon, or the box.
    pub fn footprint(&self) -> Vec<[f64; 2]> {
        if self.polygon_m.is_empty() {
            vec![
                [0.0, 0.0],
                [self.width_m, 0.0],
                [self.width_m, self.depth_m],
                [0.0, self.depth_m],
            ]
        } else {
            self.polygon_m.clone()
        }
    }

    /// Floor area. For an L-shape this is less than width x depth.
    pub fn area_m2(&self) -> f64 {
        signed_area(&self.footprint()).abs()
    }

    pub fn contains(&self, x: f64, y: f64, tol: f64) -> bool {
        contains_point(&self.footprint(), x, y, tol)
    }
}

/// One wall segment in plan view, with the material the radio sees.
///
/// `from_m` and `to_m` are the centreline and `thickness_m` is the full
/// thickness about it. A perimeter wall's centreline runs along the
/// boundary of the clear interior, so a viewer draws its thickness
/// outwards rather than into the floor.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Wall {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub from_m: [f64; 2],
    pub to_m: [f64; 2],
    pub material: WallMaterial,
    pub thickness_m: f64,
    /// Defaults to the room's height when absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height_m: Option<f64>,
}

impl Wall {
    pub fn length_m(&self) -> f64 {
        (self.to_m[0] - self.from_m[0]).hypot(self.to_m[1] - self.from_m[1])
    }

    pub fn effective_height_m(&self, room: &SceneRoom) -> f64 {
        self.height_m.unwrap_or(room.height_m)
    }
}

/// One upright box of furniture, or a door or a window.
///
/// `size_m` is (width, depth, height) in the object's own frame, before
/// `rotation_deg` turns it counter-clockwise about the vertical axis.
/// `centre_m` includes z: a 2.05 m door has its centre at z = 1.025.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneObject {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    #[serde(rename = "type")]
    pub kind: ObjectType,
    pub centre_m: [f64; 3],
    pub size_m: [f64; 3],
    #[serde(default)]
    pub rotation_deg: f64,
    pub material: ObjectMaterial,
}

impl SceneObject {
    /// The four plan-view corners after rotation, counter-clockwise.
    pub fn footprint(&self) -> [[f64; 2]; 4] {
        rotated_footprint(
            self.centre_m[0],
            self.centre_m[1],
            self.size_m[0],
            self.size_m[1],
            self.rotation_deg,
        )
    }

    pub fn z_extent_m(&self) -> (f64, f64) {
        let half = 0.5 * self.size_m[2];
        (self.centre_m[2] - half, self.centre_m[2] + half)
    }
}

/// One ESP32 at a surveyed position.
///
/// The position is an input, never an output. Nothing in this project can
/// work out where a node is: one RF chain gives no angle of arrival and
/// phase sanitization destroys absolute ranging, so the coordinates are
/// surveyed by hand and any error in them lands in whatever is computed
/// from them.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SceneNode {
    /// 0-255, so it fits the `u8` node id the wire format and
    /// [`crate::scenario::Node`] already use.
    pub id: u8,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub label: Option<String>,
    pub pos_m: [f64; 3],
}

/// A simulated person's starting position, in plan view only.
///
/// No z: the sensing stack is 2-D and assumes chests at a common height.
/// `height_m` is for drawing.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Person {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub pos_m: [f64; 2],
    #[serde(default = "default_person_height")]
    pub height_m: f64,
}

fn default_person_height() -> f64 {
    1.7
}

/// The one tracked handset's starting position.
///
/// It has a z because a phone in a pocket and a phone on a table are at
/// different heights. The 2-D estimators ignore it.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Phone {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    pub pos_m: [f64; 3],
}

/// A whole scene file, parsed and checked.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Scene {
    pub schema_version: u32,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub description: Option<String>,
    pub room: SceneRoom,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub walls: Vec<Wall>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub objects: Vec<SceneObject>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub nodes: Vec<SceneNode>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub people: Vec<Person>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phone: Option<Phone>,
}

impl Scene {
    /// Parse and check a scene from JSON text.
    pub fn from_json(text: &str) -> Result<Scene, SceneError> {
        let value = json::parse(text)?;
        Scene::from_value(&value)
    }

    /// THE ONE PLACE THE TWO MEANINGS OF "HEIGHT" MEET IN RUST.
    ///
    /// ```text
    /// scene room.width_m  -> size_m.0     both are x
    /// scene room.depth_m  -> size_m.1     both are the second HORIZONTAL axis
    /// scene room.height_m -> dropped      elevation; the 2-D room has no field
    ///                        (Vertical::height_m in Scene::to_sim_box_room)
    /// ```
    ///
    /// [`crate::scenario::Room::size_m`] is a pair of horizontal extents,
    /// the same convention as the Python `Room(width_m, height_m)`. Pass
    /// the elevation instead and a 4.2 x 3.6 m bedroom becomes a
    /// 4.2 x 2.7 m one that still runs and is wrong everywhere.
    ///
    /// An L-shaped room flattens to its bounding box, because the
    /// simulator's room is a rectangle. Use [`SceneRoom::contains`] to
    /// test a point against the real floor.
    pub fn sim_room_size_m(&self) -> Point {
        (self.room.width_m, self.room.depth_m)
    }

    /// The scene as a simulator room, with the simulator's own defaults
    /// for everything the scene does not describe.
    ///
    /// Wall materials are NOT applied. [`crate::scenario::Room`] carries
    /// one scalar `wall_reflection` for all four walls, and deciding
    /// which of a room's materials that scalar should stand for is a
    /// modelling decision nobody has made yet;
    /// [`WallMaterial::constants`] and
    /// [`MaterialConstants::reflection_normal`] are what that decision
    /// would be built from. Until then this is the default channel in a
    /// scene-shaped room, and saying otherwise would be inventing physics.
    pub fn to_sim_room(&self) -> SimRoom {
        SimRoom {
            size_m: self.sim_room_size_m(),
            ..SimRoom::default()
        }
    }

    /// [`Scene::to_sim_room`] as a box: the scene's own floor-to-ceiling
    /// height, and the [`Vertical`] defaults for everything the scene does
    /// not describe. The floor and ceiling are concrete because the scene
    /// format has no field for either material, for the same reason the
    /// walls are not applied.
    pub fn to_sim_box_room(&self) -> SimRoom {
        SimRoom {
            vertical: Some(Vertical {
                height_m: self.room.height_m,
                ..Vertical::default()
            }),
            ..self.to_sim_room()
        }
    }

    /// The scene's nodes as simulator nodes, elevation included.
    ///
    /// In the 2-D slice the simulator ignores `z_m`: nodes and chests are
    /// assumed to be at roughly one height, and [`Scene::node_z_spread_m`]
    /// says whether a given scene honours that. In a box
    /// ([`Scene::to_sim_box_room`]) the heights are used.
    pub fn to_sim_nodes(&self) -> Vec<SimNode> {
        self.nodes
            .iter()
            .map(|n| SimNode {
                id: n.id,
                pos: (n.pos_m[0], n.pos_m[1]),
                z_m: n.pos_m[2],
            })
            .collect()
    }

    /// Largest difference in node elevation; 0.0 with fewer than two nodes.
    pub fn node_z_spread_m(&self) -> f64 {
        if self.nodes.len() < 2 {
            return 0.0;
        }
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for n in &self.nodes {
            lo = lo.min(n.pos_m[2]);
            hi = hi.max(n.pos_m[2]);
        }
        hi - lo
    }

    fn from_value(value: &json::Json) -> Result<Scene, SceneError> {
        let root = value.as_object("scene")?;
        json::check_keys(
            root,
            &["schema_version", "room"],
            &[
                "name",
                "description",
                "walls",
                "objects",
                "nodes",
                "people",
                "phone",
            ],
            "scene",
        )?;

        let version = json::integer(
            json::get(root, "schema_version").unwrap(),
            0,
            u32::MAX as i64,
            "scene.schema_version",
        )? as u32;
        if version != SCHEMA_VERSION {
            return Err(SceneError::new(
                "scene.schema_version",
                format!(
                    "is {version}; this loader reads version {SCHEMA_VERSION} only. \
                     Refusing rather than guessing at a format it has not been told about."
                ),
            ));
        }

        let room = parse_room(json::get(root, "room").unwrap())?;
        let footprint = room.footprint();

        let mut walls = Vec::new();
        for (i, item) in json::array_or_empty(root, "walls", "scene.walls")?
            .iter()
            .enumerate()
        {
            walls.push(parse_wall(item, &room, &format!("scene.walls[{i}]"))?);
        }

        let mut objects = Vec::new();
        for (i, item) in json::array_or_empty(root, "objects", "scene.objects")?
            .iter()
            .enumerate()
        {
            objects.push(parse_object(
                item,
                &room,
                &footprint,
                &format!("scene.objects[{i}]"),
            )?);
        }

        let mut nodes = Vec::new();
        for (i, item) in json::array_or_empty(root, "nodes", "scene.nodes")?
            .iter()
            .enumerate()
        {
            nodes.push(parse_node(
                item,
                &room,
                &footprint,
                &format!("scene.nodes[{i}]"),
            )?);
        }
        check_nodes_distinct(&nodes)?;

        let mut people = Vec::new();
        for (i, item) in json::array_or_empty(root, "people", "scene.people")?
            .iter()
            .enumerate()
        {
            people.push(parse_person(
                item,
                &footprint,
                &format!("scene.people[{i}]"),
            )?);
        }

        let phone = match json::get(root, "phone") {
            None | Some(json::Json::Null) => None,
            Some(item) => Some(parse_phone(item, &room, &footprint, "scene.phone")?),
        };

        Ok(Scene {
            schema_version: version,
            name: json::opt_string(root, "name", 120, "scene.name")?,
            description: json::opt_string(root, "description", 4000, "scene.description")?,
            room,
            walls,
            objects,
            nodes,
            people,
            phone,
        })
    }
}

fn parse_room(value: &json::Json) -> Result<SceneRoom, SceneError> {
    let obj = value.as_object("scene.room")?;
    json::check_keys(
        obj,
        &["width_m", "depth_m", "height_m"],
        &["polygon_m"],
        "scene.room",
    )?;
    let width = json::positive(json::get(obj, "width_m").unwrap(), "scene.room.width_m")?;
    let depth = json::positive(json::get(obj, "depth_m").unwrap(), "scene.room.depth_m")?;
    let height = json::positive(json::get(obj, "height_m").unwrap(), "scene.room.height_m")?;

    let mut polygon = Vec::new();
    if let Some(raw) = json::get(obj, "polygon_m") {
        let items = raw.as_array("scene.room.polygon_m")?;
        if items.len() < 3 {
            return Err(SceneError::new(
                "scene.room.polygon_m",
                format!("a floor needs at least 3 vertices, got {}", items.len()),
            ));
        }
        for (i, item) in items.iter().enumerate() {
            polygon.push(json::point2(item, &format!("scene.room.polygon_m[{i}]"))?);
        }
        check_polygon(&polygon, width, depth)?;
    }

    Ok(SceneRoom {
        width_m: width,
        depth_m: depth,
        height_m: height,
        polygon_m: polygon,
    })
}

/// A polygon has to agree with the width and depth it is declared with.
///
/// `width_m` and `depth_m` stay the authority on the room's extent, so a
/// viewer that cannot draw a polygon still gets the bounds right. If the
/// two disagree there is nothing in the file that says which is wrong.
///
/// Self-intersection is NOT checked here or in the Python loader. A
/// figure-of-eight floor passes and then behaves strangely in the
/// point-in-polygon test; catching it costs an O(n^2) segment sweep and
/// no example needs it yet.
fn check_polygon(polygon: &[[f64; 2]], width: f64, depth: f64) -> Result<(), SceneError> {
    let n = polygon.len();
    for i in 0..n {
        let (a, b) = (polygon[i], polygon[(i + 1) % n]);
        if (b[0] - a[0]).hypot(b[1] - a[1]) <= CONTAIN_TOL_M {
            return Err(SceneError::new(
                "scene.room.polygon_m",
                format!("vertices [{i}] and [{}] are the same point", (i + 1) % n),
            ));
        }
    }

    let (mut min_x, mut max_x) = (f64::INFINITY, f64::NEG_INFINITY);
    let (mut min_y, mut max_y) = (f64::INFINITY, f64::NEG_INFINITY);
    for p in polygon {
        min_x = min_x.min(p[0]);
        max_x = max_x.max(p[0]);
        min_y = min_y.min(p[1]);
        max_y = max_y.max(p[1]);
    }
    for (got, want, what, want_name) in [
        (min_x, 0.0, "min x", "0"),
        (max_x, width, "max x", "width_m"),
        (min_y, 0.0, "min y", "0"),
        (max_y, depth, "max y", "depth_m"),
    ] {
        if (got - want).abs() > 1e-6 {
            return Err(SceneError::new(
                "scene.room.polygon_m",
                format!(
                    "{what} is {got}, but {want_name} is {want}. The polygon's \
                     bounding box must be the stated width_m x depth_m."
                ),
            ));
        }
    }

    let area = signed_area(polygon);
    if area <= 0.0 {
        return Err(SceneError::new(
            "scene.room.polygon_m",
            format!(
                "vertices must run counter-clockwise and enclose an area; signed area is {area} m2"
            ),
        ));
    }
    Ok(())
}

fn parse_wall(value: &json::Json, room: &SceneRoom, path: &str) -> Result<Wall, SceneError> {
    let obj = value.as_object(path)?;
    json::check_keys(
        obj,
        &["from_m", "to_m", "material", "thickness_m"],
        &["id", "height_m"],
        path,
    )?;
    let from = json::point2(json::get(obj, "from_m").unwrap(), &format!("{path}.from_m"))?;
    let to = json::point2(json::get(obj, "to_m").unwrap(), &format!("{path}.to_m"))?;
    if (to[0] - from[0]).hypot(to[1] - from[1]) <= CONTAIN_TOL_M {
        return Err(SceneError::new(path, "from_m and to_m are the same point"));
    }
    // Checked against the bounding box, not the polygon: a wall runs
    // ALONG the floor's edge, which is exactly the case point-in-polygon
    // leaves undefined, and a partition runs inside it.
    check_in_box(from[0], from[1], room, &format!("{path}.from_m"))?;
    check_in_box(to[0], to[1], room, &format!("{path}.to_m"))?;

    let thickness = json::positive(
        json::get(obj, "thickness_m").unwrap(),
        &format!("{path}.thickness_m"),
    )?;
    if thickness > 2.0 {
        return Err(SceneError::new(
            format!("{path}.thickness_m"),
            format!("{thickness} m is not a wall"),
        ));
    }

    let height = match json::get(obj, "height_m") {
        None => None,
        Some(raw) => {
            let h = json::positive(raw, &format!("{path}.height_m"))?;
            if h > room.height_m + CONTAIN_TOL_M {
                return Err(SceneError::new(
                    format!("{path}.height_m"),
                    format!("{h} m is taller than the room's {} m", room.height_m),
                ));
            }
            Some(h)
        }
    };

    let material_name = json::get(obj, "material")
        .unwrap()
        .as_string(&format!("{path}.material"))?;
    let material = WallMaterial::parse(material_name).ok_or_else(|| {
        SceneError::new(
            format!("{path}.material"),
            format!("{material_name:?} is not one of brick, concrete, drywall, glass, wood"),
        )
    })?;

    Ok(Wall {
        id: json::opt_string(obj, "id", 64, &format!("{path}.id"))?,
        from_m: from,
        to_m: to,
        material,
        thickness_m: thickness,
        height_m: height,
    })
}

fn parse_object(
    value: &json::Json,
    room: &SceneRoom,
    footprint: &[[f64; 2]],
    path: &str,
) -> Result<SceneObject, SceneError> {
    let obj = value.as_object(path)?;
    json::check_keys(
        obj,
        &["type", "centre_m", "size_m", "material"],
        &["id", "rotation_deg"],
        path,
    )?;

    let type_name = json::get(obj, "type")
        .unwrap()
        .as_string(&format!("{path}.type"))?;
    let kind = ObjectType::parse(type_name).ok_or_else(|| {
        SceneError::new(
            format!("{path}.type"),
            format!("{type_name:?} is not one of almirah, bed, chair, door, sofa, table, window"),
        )
    })?;

    let centre = json::point3(
        json::get(obj, "centre_m").unwrap(),
        &format!("{path}.centre_m"),
    )?;
    let size_raw = json::get(obj, "size_m")
        .unwrap()
        .as_array(&format!("{path}.size_m"))?;
    if size_raw.len() != 3 {
        return Err(SceneError::new(
            format!("{path}.size_m"),
            format!("expected 3 numbers, got {}", size_raw.len()),
        ));
    }
    let mut size = [0.0f64; 3];
    for (i, item) in size_raw.iter().enumerate() {
        size[i] = json::positive(item, &format!("{path}.size_m[{i}]"))?;
    }

    let rotation = match json::get(obj, "rotation_deg") {
        None => 0.0,
        Some(raw) => {
            let r = raw.as_number(&format!("{path}.rotation_deg"))?;
            if !(-360.0..=360.0).contains(&r) {
                return Err(SceneError::new(
                    format!("{path}.rotation_deg"),
                    format!("{r} is outside -360..360"),
                ));
            }
            r
        }
    };

    let material_name = json::get(obj, "material")
        .unwrap()
        .as_string(&format!("{path}.material"))?;
    let material = ObjectMaterial::parse(material_name).ok_or_else(|| {
        SceneError::new(
            format!("{path}.material"),
            format!("{material_name:?} is not one of fabric, glass, metal, wood"),
        )
    })?;

    let out = SceneObject {
        id: json::opt_string(obj, "id", 64, &format!("{path}.id"))?,
        kind,
        centre_m: centre,
        size_m: size,
        rotation_deg: rotation,
        material,
    };

    for (i, corner) in out.footprint().iter().enumerate() {
        if !contains_point(footprint, corner[0], corner[1], CONTAIN_TOL_M) {
            return Err(SceneError::new(
                path,
                format!(
                    "corner {i} of the {} is at ({:.3}, {:.3}) m, outside the room",
                    out.kind.as_str(),
                    corner[0],
                    corner[1]
                ),
            ));
        }
    }
    let (low, high) = out.z_extent_m();
    if low < -CONTAIN_TOL_M || high > room.height_m + CONTAIN_TOL_M {
        return Err(SceneError::new(
            path,
            format!(
                "the {} spans {low:.3} to {high:.3} m in z, outside the room's 0 to {} m",
                out.kind.as_str(),
                room.height_m
            ),
        ));
    }
    Ok(out)
}

fn parse_node(
    value: &json::Json,
    room: &SceneRoom,
    footprint: &[[f64; 2]],
    path: &str,
) -> Result<SceneNode, SceneError> {
    let obj = value.as_object(path)?;
    json::check_keys(obj, &["id", "pos_m"], &["label"], path)?;
    // 0..=255 because the wire format and the simulator both carry a node
    // id in one byte.
    let id = json::integer(json::get(obj, "id").unwrap(), 0, 255, &format!("{path}.id"))? as u8;
    let pos = json::point3(json::get(obj, "pos_m").unwrap(), &format!("{path}.pos_m"))?;
    if !contains_point(footprint, pos[0], pos[1], CONTAIN_TOL_M) {
        return Err(SceneError::new(
            path,
            format!(
                "node {id} is at ({:.3}, {:.3}) m, outside the room",
                pos[0], pos[1]
            ),
        ));
    }
    if pos[2] < -CONTAIN_TOL_M || pos[2] > room.height_m + CONTAIN_TOL_M {
        return Err(SceneError::new(
            path,
            format!(
                "node {id} is at z = {:.3} m, outside the room's 0 to {} m",
                pos[2], room.height_m
            ),
        ));
    }
    Ok(SceneNode {
        id,
        label: json::opt_string(obj, "label", 64, &format!("{path}.label"))?,
        pos_m: pos,
    })
}

fn check_nodes_distinct(nodes: &[SceneNode]) -> Result<(), SceneError> {
    for i in 0..nodes.len() {
        for j in 0..i {
            if nodes[i].id == nodes[j].id {
                return Err(SceneError::new(
                    format!("scene.nodes[{i}]"),
                    format!(
                        "node id {} is already used by scene.nodes[{j}]",
                        nodes[i].id
                    ),
                ));
            }
        }
    }
    for i in 0..nodes.len() {
        for j in (i + 1)..nodes.len() {
            let (a, b) = (nodes[i].pos_m, nodes[j].pos_m);
            let gap =
                ((a[0] - b[0]).powi(2) + (a[1] - b[1]).powi(2) + (a[2] - b[2]).powi(2)).sqrt();
            if gap <= NODE_MIN_SEPARATION_M {
                return Err(SceneError::new(
                    format!("scene.nodes[{i}]"),
                    format!(
                        "nodes {} and {} are {gap} m apart, which is the same point; \
                         the link between them would be degenerate",
                        nodes[i].id, nodes[j].id
                    ),
                ));
            }
        }
    }
    Ok(())
}

fn parse_person(
    value: &json::Json,
    footprint: &[[f64; 2]],
    path: &str,
) -> Result<Person, SceneError> {
    let obj = value.as_object(path)?;
    json::check_keys(obj, &["pos_m"], &["id", "height_m"], path)?;
    let pos = json::point2(json::get(obj, "pos_m").unwrap(), &format!("{path}.pos_m"))?;
    if !contains_point(footprint, pos[0], pos[1], CONTAIN_TOL_M) {
        return Err(SceneError::new(
            path,
            format!(
                "person is at ({:.3}, {:.3}) m, outside the room",
                pos[0], pos[1]
            ),
        ));
    }
    let height = match json::get(obj, "height_m") {
        None => default_person_height(),
        Some(raw) => {
            let h = json::positive(raw, &format!("{path}.height_m"))?;
            if h > 2.5 {
                return Err(SceneError::new(
                    format!("{path}.height_m"),
                    format!("{h} m is not a person"),
                ));
            }
            h
        }
    };
    Ok(Person {
        id: json::opt_string(obj, "id", 64, &format!("{path}.id"))?,
        pos_m: pos,
        height_m: height,
    })
}

fn parse_phone(
    value: &json::Json,
    room: &SceneRoom,
    footprint: &[[f64; 2]],
    path: &str,
) -> Result<Phone, SceneError> {
    let obj = value.as_object(path)?;
    json::check_keys(obj, &["pos_m"], &["id"], path)?;
    let pos = json::point3(json::get(obj, "pos_m").unwrap(), &format!("{path}.pos_m"))?;
    if !contains_point(footprint, pos[0], pos[1], CONTAIN_TOL_M) {
        return Err(SceneError::new(
            path,
            format!(
                "phone is at ({:.3}, {:.3}) m, outside the room",
                pos[0], pos[1]
            ),
        ));
    }
    if pos[2] < -CONTAIN_TOL_M || pos[2] > room.height_m + CONTAIN_TOL_M {
        return Err(SceneError::new(
            path,
            format!(
                "phone is at z = {:.3} m, outside the room's 0 to {} m",
                pos[2], room.height_m
            ),
        ));
    }
    Ok(Phone {
        id: json::opt_string(obj, "id", 64, &format!("{path}.id"))?,
        pos_m: pos,
    })
}

fn check_in_box(x: f64, y: f64, room: &SceneRoom, path: &str) -> Result<(), SceneError> {
    let inside = x >= -CONTAIN_TOL_M
        && x <= room.width_m + CONTAIN_TOL_M
        && y >= -CONTAIN_TOL_M
        && y <= room.depth_m + CONTAIN_TOL_M;
    if inside {
        Ok(())
    } else {
        Err(SceneError::new(
            path,
            format!(
                "({x:.3}, {y:.3}) m is outside the room's {} x {} m bounds",
                room.width_m, room.depth_m
            ),
        ))
    }
}

// ---------------------------------------------------------------------------
// Plan-view geometry
// ---------------------------------------------------------------------------

/// Shoelace area; positive when the vertices run counter-clockwise.
pub fn signed_area(poly: &[[f64; 2]]) -> f64 {
    let n = poly.len();
    let mut total = 0.0;
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        total += a[0] * b[1] - b[0] * a[1];
    }
    0.5 * total
}

/// Ray casting, with the boundary left undefined on purpose.
///
/// A point exactly on an edge may come out either way, which is why
/// [`contains_point`] tests the boundary separately. Doors and windows sit
/// exactly in the plane of their wall, so boundary points are the normal
/// case here, not an edge case.
pub fn point_in_polygon(poly: &[[f64; 2]], x: f64, y: f64) -> bool {
    let mut inside = false;
    let n = poly.len();
    for i in 0..n {
        let (a, b) = (poly[i], poly[(i + 1) % n]);
        if (a[1] > y) != (b[1] > y) {
            let t = (y - a[1]) / (b[1] - a[1]);
            if x < a[0] + t * (b[0] - a[0]) {
                inside = !inside;
            }
        }
    }
    inside
}

pub fn distance_to_segment(p: [f64; 2], a: [f64; 2], b: [f64; 2]) -> f64 {
    let (dx, dy) = (b[0] - a[0], b[1] - a[1]);
    let len_sq = dx * dx + dy * dy;
    if len_sq <= 0.0 {
        return (p[0] - a[0]).hypot(p[1] - a[1]);
    }
    let t = (((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / len_sq).clamp(0.0, 1.0);
    (p[0] - (a[0] + t * dx)).hypot(p[1] - (a[1] + t * dy))
}

/// Inside the polygon, or within `tol` metres of its boundary.
pub fn contains_point(poly: &[[f64; 2]], x: f64, y: f64, tol: f64) -> bool {
    if point_in_polygon(poly, x, y) {
        return true;
    }
    let n = poly.len();
    (0..n).any(|i| distance_to_segment([x, y], poly[i], poly[(i + 1) % n]) <= tol)
}

/// The four plan corners of a `w` x `d` box centred at (cx, cy).
pub fn rotated_footprint(cx: f64, cy: f64, w: f64, d: f64, rotation_deg: f64) -> [[f64; 2]; 4] {
    let theta = rotation_deg.to_radians();
    let (sin_t, cos_t) = theta.sin_cos();
    let (hw, hd) = (0.5 * w, 0.5 * d);
    let offsets = [[-hw, -hd], [hw, -hd], [hw, hd], [-hw, hd]];
    let mut out = [[0.0f64; 2]; 4];
    for (i, o) in offsets.iter().enumerate() {
        out[i] = [
            cx + o[0] * cos_t - o[1] * sin_t,
            cy + o[0] * sin_t + o[1] * cos_t,
        ];
    }
    out
}

// ---------------------------------------------------------------------------
// A small JSON reader
// ---------------------------------------------------------------------------

/// Just enough JSON to read a scene file, because this crate has `serde`
/// and not `serde_json` and was not allowed a new dependency.
///
/// Strict on purpose: no trailing commas, no comments, no `NaN` or
/// `Infinity` (CPython's `json` accepts those two as an extension, and a
/// room 4.2 m wide and NaN deep would otherwise load and then poison
/// every distance computed from it), no duplicate keys, and a nesting
/// limit. It returns [`SceneError`] with a line and column; it never
/// panics and it never recurses without bound.
pub mod json {
    use super::{SceneError, MAX_JSON_DEPTH};

    #[derive(Debug, Clone, PartialEq)]
    pub enum Json {
        Null,
        Bool(bool),
        Number(f64),
        String(String),
        Array(Vec<Json>),
        /// Key order is kept, so error messages can quote the file.
        Object(Vec<(String, Json)>),
    }

    impl Json {
        pub fn kind(&self) -> &'static str {
            match self {
                Json::Null => "null",
                Json::Bool(_) => "a boolean",
                Json::Number(_) => "a number",
                Json::String(_) => "a string",
                Json::Array(_) => "an array",
                Json::Object(_) => "an object",
            }
        }

        pub fn as_object(&self, path: &str) -> Result<&[(String, Json)], SceneError> {
            match self {
                Json::Object(fields) => Ok(fields),
                other => Err(SceneError::new(
                    path,
                    format!("expected an object, got {}", other.kind()),
                )),
            }
        }

        pub fn as_array(&self, path: &str) -> Result<&[Json], SceneError> {
            match self {
                Json::Array(items) => Ok(items),
                other => Err(SceneError::new(
                    path,
                    format!("expected an array, got {}", other.kind()),
                )),
            }
        }

        pub fn as_number(&self, path: &str) -> Result<f64, SceneError> {
            match self {
                Json::Number(v) => Ok(*v),
                other => Err(SceneError::new(
                    path,
                    format!("expected a number, got {}", other.kind()),
                )),
            }
        }

        pub fn as_string(&self, path: &str) -> Result<&str, SceneError> {
            match self {
                Json::String(s) => Ok(s),
                other => Err(SceneError::new(
                    path,
                    format!("expected a string, got {}", other.kind()),
                )),
            }
        }
    }

    pub fn get<'a>(fields: &'a [(String, Json)], key: &str) -> Option<&'a Json> {
        fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
    }

    pub fn check_keys(
        fields: &[(String, Json)],
        required: &[&str],
        optional: &[&str],
        path: &str,
    ) -> Result<(), SceneError> {
        let missing: Vec<&str> = required
            .iter()
            .copied()
            .filter(|k| get(fields, k).is_none())
            .collect();
        if !missing.is_empty() {
            return Err(SceneError::new(
                path,
                format!("missing {}", missing.join(", ")),
            ));
        }
        let mut extra: Vec<&str> = fields
            .iter()
            .map(|(k, _)| k.as_str())
            .filter(|k| !required.contains(k) && !optional.contains(k))
            .collect();
        if !extra.is_empty() {
            extra.sort_unstable();
            let mut allowed: Vec<&str> = required.iter().chain(optional.iter()).copied().collect();
            allowed.sort_unstable();
            return Err(SceneError::new(
                path,
                format!(
                    "unknown key(s) {} (allowed: {})",
                    extra.join(", "),
                    allowed.join(", ")
                ),
            ));
        }
        Ok(())
    }

    pub fn array_or_empty<'a>(
        fields: &'a [(String, Json)],
        key: &str,
        path: &str,
    ) -> Result<&'a [Json], SceneError> {
        match get(fields, key) {
            None => Ok(&[]),
            Some(value) => value.as_array(path),
        }
    }

    pub fn opt_string(
        fields: &[(String, Json)],
        key: &str,
        max_length: usize,
        path: &str,
    ) -> Result<Option<String>, SceneError> {
        match get(fields, key) {
            None | Some(Json::Null) => Ok(None),
            Some(value) => {
                let s = value.as_string(path)?;
                if s.chars().count() > max_length {
                    return Err(SceneError::new(
                        path,
                        format!("longer than {max_length} characters"),
                    ));
                }
                Ok(Some(s.to_string()))
            }
        }
    }

    pub fn positive(value: &Json, path: &str) -> Result<f64, SceneError> {
        let v = value.as_number(path)?;
        if v <= 0.0 {
            return Err(SceneError::new(path, format!("must be positive, got {v}")));
        }
        Ok(v)
    }

    pub fn integer(value: &Json, lo: i64, hi: i64, path: &str) -> Result<i64, SceneError> {
        let v = value.as_number(path)?;
        if v.fract() != 0.0 {
            return Err(SceneError::new(
                path,
                format!("expected an integer, got {v}"),
            ));
        }
        let as_int = v as i64;
        if as_int < lo || as_int > hi {
            return Err(SceneError::new(
                path,
                format!("{as_int} is outside {lo}..{hi}"),
            ));
        }
        Ok(as_int)
    }

    pub fn point2(value: &Json, path: &str) -> Result<[f64; 2], SceneError> {
        let items = value.as_array(path)?;
        if items.len() != 2 {
            return Err(SceneError::new(
                path,
                format!("expected 2 numbers, got {}", items.len()),
            ));
        }
        Ok([
            items[0].as_number(&format!("{path}[0]"))?,
            items[1].as_number(&format!("{path}[1]"))?,
        ])
    }

    pub fn point3(value: &Json, path: &str) -> Result<[f64; 3], SceneError> {
        let items = value.as_array(path)?;
        if items.len() != 3 {
            return Err(SceneError::new(
                path,
                format!("expected 3 numbers, got {}", items.len()),
            ));
        }
        Ok([
            items[0].as_number(&format!("{path}[0]"))?,
            items[1].as_number(&format!("{path}[1]"))?,
            items[2].as_number(&format!("{path}[2]"))?,
        ])
    }

    pub fn parse(text: &str) -> Result<Json, SceneError> {
        let mut reader = Reader {
            bytes: text.as_bytes(),
            pos: 0,
        };
        reader.skip_whitespace();
        let value = reader.value(0)?;
        reader.skip_whitespace();
        if reader.pos != reader.bytes.len() {
            return Err(reader.error("trailing content after the top-level value"));
        }
        Ok(value)
    }

    struct Reader<'a> {
        bytes: &'a [u8],
        pos: usize,
    }

    impl<'a> Reader<'a> {
        fn error(&self, message: impl Into<String>) -> SceneError {
            let mut line = 1usize;
            let mut column = 1usize;
            for &b in &self.bytes[..self.pos.min(self.bytes.len())] {
                if b == b'\n' {
                    line += 1;
                    column = 1;
                } else {
                    column += 1;
                }
            }
            SceneError::new(
                "<json>",
                format!("line {line}, column {column}: {}", message.into()),
            )
        }

        fn peek(&self) -> Option<u8> {
            self.bytes.get(self.pos).copied()
        }

        fn skip_whitespace(&mut self) {
            while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
                self.pos += 1;
            }
        }

        fn expect(&mut self, byte: u8) -> Result<(), SceneError> {
            if self.peek() == Some(byte) {
                self.pos += 1;
                Ok(())
            } else {
                Err(self.error(format!("expected {:?}", byte as char)))
            }
        }

        fn literal(&mut self, word: &str, value: Json) -> Result<Json, SceneError> {
            if self.bytes[self.pos..].starts_with(word.as_bytes()) {
                self.pos += word.len();
                Ok(value)
            } else {
                Err(self.error("not a valid JSON value"))
            }
        }

        fn value(&mut self, depth: u32) -> Result<Json, SceneError> {
            if depth > MAX_JSON_DEPTH {
                return Err(self.error(format!("nested deeper than {MAX_JSON_DEPTH} levels")));
            }
            match self.peek() {
                None => Err(self.error("the file ends where a value was expected")),
                Some(b'{') => self.object(depth),
                Some(b'[') => self.array(depth),
                Some(b'"') => Ok(Json::String(self.string()?)),
                Some(b't') => self.literal("true", Json::Bool(true)),
                Some(b'f') => self.literal("false", Json::Bool(false)),
                Some(b'n') => self.literal("null", Json::Null),
                Some(b'-' | b'0'..=b'9') => self.number(),
                Some(b'N') | Some(b'I') => Err(self.error(
                    "NaN and Infinity are not JSON, and not a position: every number in a scene must be finite",
                )),
                Some(c) => Err(self.error(format!("unexpected {:?}", c as char))),
            }
        }

        fn object(&mut self, depth: u32) -> Result<Json, SceneError> {
            self.expect(b'{')?;
            let mut fields: Vec<(String, Json)> = Vec::new();
            self.skip_whitespace();
            if self.peek() == Some(b'}') {
                self.pos += 1;
                return Ok(Json::Object(fields));
            }
            loop {
                self.skip_whitespace();
                let key_at = self.pos;
                let key = self.string()?;
                if fields.iter().any(|(k, _)| *k == key) {
                    self.pos = key_at;
                    return Err(self.error(format!("duplicate key {key:?}")));
                }
                self.skip_whitespace();
                self.expect(b':')?;
                self.skip_whitespace();
                let value = self.value(depth + 1)?;
                fields.push((key, value));
                self.skip_whitespace();
                match self.peek() {
                    Some(b',') => {
                        self.pos += 1;
                    }
                    Some(b'}') => {
                        self.pos += 1;
                        return Ok(Json::Object(fields));
                    }
                    _ => return Err(self.error("expected ',' or '}'")),
                }
            }
        }

        fn array(&mut self, depth: u32) -> Result<Json, SceneError> {
            self.expect(b'[')?;
            let mut items = Vec::new();
            self.skip_whitespace();
            if self.peek() == Some(b']') {
                self.pos += 1;
                return Ok(Json::Array(items));
            }
            loop {
                self.skip_whitespace();
                items.push(self.value(depth + 1)?);
                self.skip_whitespace();
                match self.peek() {
                    Some(b',') => {
                        self.pos += 1;
                    }
                    Some(b']') => {
                        self.pos += 1;
                        return Ok(Json::Array(items));
                    }
                    _ => return Err(self.error("expected ',' or ']'")),
                }
            }
        }

        fn string(&mut self) -> Result<String, SceneError> {
            self.expect(b'"')?;
            let mut out = String::new();
            loop {
                let byte = match self.peek() {
                    None => return Err(self.error("the file ends inside a string")),
                    Some(b) => b,
                };
                match byte {
                    b'"' => {
                        self.pos += 1;
                        return Ok(out);
                    }
                    b'\\' => {
                        self.pos += 1;
                        let escape = self
                            .peek()
                            .ok_or_else(|| self.error("the file ends inside an escape"))?;
                        self.pos += 1;
                        match escape {
                            b'"' => out.push('"'),
                            b'\\' => out.push('\\'),
                            b'/' => out.push('/'),
                            b'b' => out.push('\u{8}'),
                            b'f' => out.push('\u{c}'),
                            b'n' => out.push('\n'),
                            b'r' => out.push('\r'),
                            b't' => out.push('\t'),
                            b'u' => out.push(self.unicode_escape()?),
                            other => {
                                return Err(
                                    self.error(format!("unknown escape \\{}", other as char))
                                )
                            }
                        }
                    }
                    0x00..=0x1f => {
                        return Err(self.error("a control character must be escaped in a string"))
                    }
                    _ => {
                        // Copy one whole UTF-8 sequence. The input is a
                        // &str, so it is already valid UTF-8.
                        let start = self.pos;
                        self.pos += 1;
                        while matches!(self.peek(), Some(b) if b & 0xc0 == 0x80) {
                            self.pos += 1;
                        }
                        match std::str::from_utf8(&self.bytes[start..self.pos]) {
                            Ok(s) => out.push_str(s),
                            Err(_) => return Err(self.error("not valid UTF-8")),
                        }
                    }
                }
            }
        }

        fn hex4(&mut self) -> Result<u16, SceneError> {
            if self.pos + 4 > self.bytes.len() {
                return Err(self.error("a \\u escape needs four hex digits"));
            }
            let mut value: u16 = 0;
            for _ in 0..4 {
                let digit = match self.bytes[self.pos] {
                    c @ b'0'..=b'9' => c - b'0',
                    c @ b'a'..=b'f' => c - b'a' + 10,
                    c @ b'A'..=b'F' => c - b'A' + 10,
                    _ => return Err(self.error("a \\u escape needs four hex digits")),
                };
                value = value * 16 + digit as u16;
                self.pos += 1;
            }
            Ok(value)
        }

        fn unicode_escape(&mut self) -> Result<char, SceneError> {
            let first = self.hex4()?;
            if (0xdc00..=0xdfff).contains(&first) {
                return Err(self.error("a low surrogate with no high surrogate before it"));
            }
            if !(0xd800..=0xdbff).contains(&first) {
                return char::from_u32(first as u32).ok_or_else(|| self.error("not a character"));
            }
            if !(self.peek() == Some(b'\\') && self.bytes.get(self.pos + 1) == Some(&b'u')) {
                return Err(self.error("a high surrogate with no \\u escape after it"));
            }
            self.pos += 2;
            let second = self.hex4()?;
            if !(0xdc00..=0xdfff).contains(&second) {
                return Err(self
                    .error("a high surrogate followed by something that is not a low surrogate"));
            }
            let combined = 0x10000 + (((first as u32) - 0xd800) << 10) + ((second as u32) - 0xdc00);
            char::from_u32(combined).ok_or_else(|| self.error("not a character"))
        }

        fn number(&mut self) -> Result<Json, SceneError> {
            let start = self.pos;
            if self.peek() == Some(b'-') {
                self.pos += 1;
            }
            match self.peek() {
                Some(b'0') => self.pos += 1,
                Some(b'1'..=b'9') => {
                    while matches!(self.peek(), Some(b'0'..=b'9')) {
                        self.pos += 1;
                    }
                }
                _ => return Err(self.error("a number needs a digit here")),
            }
            if self.peek() == Some(b'.') {
                self.pos += 1;
                if !matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error("a decimal point needs a digit after it"));
                }
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            if matches!(self.peek(), Some(b'e' | b'E')) {
                self.pos += 1;
                if matches!(self.peek(), Some(b'+' | b'-')) {
                    self.pos += 1;
                }
                if !matches!(self.peek(), Some(b'0'..=b'9')) {
                    return Err(self.error("an exponent needs a digit"));
                }
                while matches!(self.peek(), Some(b'0'..=b'9')) {
                    self.pos += 1;
                }
            }
            let text = std::str::from_utf8(&self.bytes[start..self.pos])
                .map_err(|_| self.error("not valid UTF-8"))?;
            let value: f64 = text
                .parse()
                .map_err(|_| self.error(format!("{text} is not a number")))?;
            if !value.is_finite() {
                // 1e400 parses without complaint and comes out infinite.
                self.pos = start;
                return Err(self.error(format!("{text} is too large for a 64-bit float")));
            }
            Ok(Json::Number(value))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // include_str! rather than a path read at run time, so a missing or
    // renamed scene is a compile error instead of a test that quietly
    // passes because it checked nothing.
    const BEDROOM: &str = include_str!("../../../../scenes/bedroom.json");
    const OFFICE: &str = include_str!("../../../../scenes/office.json");
    const HALL_L: &str = include_str!("../../../../scenes/hall-l.json");
    const MATERIALS_JSON: &str = include_str!("../../../../scenes/materials.json");
    const SCHEMA_JSON: &str = include_str!("../../../../scenes/scene.schema.json");

    fn examples() -> Vec<(&'static str, Scene)> {
        vec![
            ("bedroom", Scene::from_json(BEDROOM).expect("bedroom")),
            ("office", Scene::from_json(OFFICE).expect("office")),
            ("hall-l", Scene::from_json(HALL_L).expect("hall-l")),
        ]
    }

    #[test]
    fn every_example_scene_loads() {
        for (name, scene) in examples() {
            assert_eq!(scene.schema_version, SCHEMA_VERSION, "{name}");
            assert!(!scene.walls.is_empty(), "{name}: no walls");
            assert!(!scene.objects.is_empty(), "{name}: no objects");
            assert!(
                scene.nodes.len() >= 4,
                "{name}: {} nodes",
                scene.nodes.len()
            );
            assert!(scene.phone.is_some(), "{name}: no phone");
        }
    }

    #[test]
    fn the_example_scenes_are_the_advertised_rooms() {
        let by_name: Vec<(&str, f64, f64, f64)> = examples()
            .iter()
            .map(|(n, s)| (*n, s.room.width_m, s.room.depth_m, s.room.height_m))
            .collect();
        assert_eq!(by_name[0], ("bedroom", 4.2, 3.6, 2.7));
        assert_eq!(by_name[1], ("office", 6.0, 5.0, 2.9));
        assert_eq!(by_name[2], ("hall-l", 8.0, 6.0, 3.0));
    }

    #[test]
    fn the_l_shaped_hall_is_smaller_than_its_bounding_box() {
        let scene = Scene::from_json(HALL_L).unwrap();
        assert!(
            (scene.room.area_m2() - 35.75).abs() < 1e-9,
            "{}",
            scene.room.area_m2()
        );
        // The notch is genuinely outside the floor.
        assert!(!scene.room.contains(7.0, 5.0, 0.0));
        assert!(scene.room.contains(7.0, 1.0, 0.0));
    }

    #[test]
    fn depth_not_elevation_maps_to_the_simulator_room() {
        for (name, scene) in examples() {
            let size = scene.sim_room_size_m();
            assert_eq!(size.0, scene.room.width_m, "{name}");
            assert_eq!(size.1, scene.room.depth_m, "{name}");
            // The trap this test exists for: every example gives width,
            // depth and elevation three different values, so swapping the
            // last two cannot pass.
            assert_ne!(
                size.1, scene.room.height_m,
                "{name}: depth and elevation are equal, so this test proves nothing"
            );
            assert_eq!(scene.to_sim_room().size_m, size, "{name}");
        }
    }

    #[test]
    fn nodes_convert_to_simulator_nodes_at_one_height() {
        for (name, scene) in examples() {
            let sim = scene.to_sim_nodes();
            assert_eq!(sim.len(), scene.nodes.len(), "{name}");
            for (a, b) in sim.iter().zip(&scene.nodes) {
                assert_eq!(a.id, b.id, "{name}");
                assert_eq!(a.pos, (b.pos_m[0], b.pos_m[1]), "{name}");
                assert_eq!(a.z_m, b.pos_m[2], "{name}");
            }
            // The slice ignoring z is only honest while the nodes share a
            // height.
            assert_eq!(scene.node_z_spread_m(), 0.0, "{name}");
        }
    }

    #[test]
    fn a_scene_runs_as_a_box_at_its_own_height() {
        for (name, scene) in examples() {
            let room = scene.to_sim_box_room();
            assert_eq!(room.size_m, scene.sim_room_size_m(), "{name}");
            let v = room.vertical.expect("a box");
            assert_eq!(v.height_m, scene.room.height_m, "{name}");
            assert!(scene.to_sim_room().vertical.is_none(), "{name}");

            // And the simulator takes it, nodes at their scene heights.
            let nodes = scene.to_sim_nodes();
            let sc = crate::scenarios::scenario(
                name,
                room,
                crate::scenarios::Layout {
                    links: vec![(nodes[0].id, nodes[1].id)],
                    nodes,
                },
                crate::scenarios::empty(0.01),
                1,
            );
            assert!(crate::Simulator::try_new(&sc).is_ok(), "{name}");
        }
    }

    #[test]
    fn furniture_is_a_plausible_size() {
        // Not a schema's business, and the kind of thing a scene gets
        // wrong by a factor of ten without anyone noticing.
        let bedroom = Scene::from_json(BEDROOM).unwrap();
        let bed = bedroom
            .objects
            .iter()
            .find(|o| o.kind == ObjectType::Bed)
            .expect("a bed");
        assert!(
            (bed.size_m[0] - 1.9).abs() < 0.2,
            "bed length {}",
            bed.size_m[0]
        );
        assert!(
            (bed.size_m[1] - 1.4).abs() < 0.2,
            "double bed width {}",
            bed.size_m[1]
        );
        let door = bedroom
            .objects
            .iter()
            .find(|o| o.kind == ObjectType::Door)
            .expect("a door");
        assert!(
            (door.size_m[0] - 0.8).abs() < 0.15,
            "door width {}",
            door.size_m[0]
        );
        assert!(
            door.size_m[2] > 1.9 && door.size_m[2] < 2.2,
            "door height {}",
            door.size_m[2]
        );
    }

    #[test]
    fn a_rotated_bed_is_still_inside_the_room() {
        // The bedroom's bed is at 90 degrees; its footprint is 1.4 x 1.9
        // and not 1.9 x 1.4. A loader that ignored rotation_deg would
        // place a corner at x = 0.75 - 0.95 = -0.2.
        let bedroom = Scene::from_json(BEDROOM).unwrap();
        let bed = bedroom
            .objects
            .iter()
            .find(|o| o.kind == ObjectType::Bed)
            .unwrap();
        assert_eq!(bed.rotation_deg, 90.0);
        let xs: Vec<f64> = bed.footprint().iter().map(|c| c[0]).collect();
        let span = xs.iter().cloned().fold(f64::MIN, f64::max)
            - xs.iter().cloned().fold(f64::MAX, f64::min);
        assert!((span - 1.4).abs() < 1e-9, "rotated x span {span}");
    }

    #[test]
    fn materials_json_matches_the_table() {
        let value = json::parse(MATERIALS_JSON).expect("materials.json parses");
        let root = value.as_object("materials").unwrap();
        assert_eq!(
            json::get(root, "evaluated_at_ghz")
                .unwrap()
                .as_number("f")
                .unwrap(),
            MATERIAL_REFERENCE_GHZ
        );
        let table = json::get(root, "materials")
            .unwrap()
            .as_object("materials.materials")
            .unwrap();

        let mut checked = 0;
        for (name, entry) in table {
            let fields = entry.as_object(name).unwrap();
            let itu = json::get(fields, "itu_p2040").unwrap();
            if *itu == json::Json::Null {
                // fabric: no row in the Recommendation, so no constants
                // here either. The type system says the same thing.
                assert_eq!(name, "fabric");
                assert_eq!(ObjectMaterial::Fabric.constants(), None);
                continue;
            }
            let constants = MATERIALS
                .iter()
                .find(|m| m.name == name)
                .unwrap_or_else(|| panic!("{name} is in materials.json and not in MATERIALS"));
            let itu = itu.as_object("itu").unwrap();
            for (key, got) in [
                ("a", constants.a),
                ("b", constants.b),
                ("c", constants.c),
                ("d", constants.d),
            ] {
                let want = json::get(itu, key).unwrap().as_number(key).unwrap();
                assert_eq!(want, got, "{name}.{key}");
            }

            let at = json::get(fields, "at_2437_mhz")
                .unwrap()
                .as_object("at")
                .unwrap();
            let f = MATERIAL_REFERENCE_GHZ;
            // The stored values are rounded to 9 significant digits, so
            // 1e-8 relative is the tightest honest comparison. It is a
            // check that the table was evaluated from these coefficients,
            // not a physics tolerance; nothing is loosened to make it pass.
            for (key, got) in [
                ("epsilon_r", constants.epsilon_r(f)),
                ("sigma_s_per_m", constants.sigma_s_per_m(f)),
                ("epsilon_imag", constants.epsilon_imag(f)),
                ("reflection_normal", constants.reflection_normal(f)),
            ] {
                let want = json::get(at, key).unwrap().as_number(key).unwrap();
                assert!(
                    (want - got).abs() <= 1e-8 * want.abs().max(got.abs()),
                    "{name}.{key}: materials.json says {want}, the table computes {got}"
                );
            }
            assert!(
                constants.in_range(f),
                "{name} is outside its stated frequency range at {f} GHz"
            );
            checked += 1;
        }
        assert_eq!(
            checked,
            MATERIALS.len(),
            "every material in the table is in the file"
        );
    }

    #[test]
    fn concrete_is_near_the_simulators_own_wall_reflection() {
        // Not a coincidence worth hiding: scenario::Room's default
        // wall_reflection is 0.4, and concrete at normal incidence comes
        // out at 0.395. The simulator's scalar is a concrete-like wall.
        let gamma = CONCRETE.reflection_normal(MATERIAL_REFERENCE_GHZ);
        assert!((gamma - 0.395).abs() < 0.001, "{gamma}");
        assert!(
            (gamma - SimRoom::default().wall_reflection).abs() < 0.01,
            "{gamma}"
        );
        // Metal reflects essentially everything; wood the least.
        assert!(METAL.reflection_normal(MATERIAL_REFERENCE_GHZ) > 0.99);
        assert!(WOOD.reflection_normal(MATERIAL_REFERENCE_GHZ) < 0.2);
    }

    #[test]
    fn the_schema_file_and_this_loader_agree() {
        // The schema ships for external tooling and this loader checks by
        // hand; the two can drift. This is what stops them.
        let value = json::parse(SCHEMA_JSON).expect("scene.schema.json parses");
        let root = value.as_object("schema").unwrap();
        let defs = json::get(root, "$defs")
            .unwrap()
            .as_object("$defs")
            .unwrap();

        let enum_of = |name: &str| -> Vec<String> {
            let def = json::get(defs, name).unwrap().as_object(name).unwrap();
            json::get(def, "enum")
                .unwrap()
                .as_array("enum")
                .unwrap()
                .iter()
                .map(|v| v.as_string("enum item").unwrap().to_string())
                .collect()
        };

        let mut wall: Vec<String> = enum_of("wall_material");
        wall.sort();
        let mut ours: Vec<String> = [
            WallMaterial::Concrete,
            WallMaterial::Brick,
            WallMaterial::Drywall,
            WallMaterial::Glass,
            WallMaterial::Wood,
        ]
        .iter()
        .map(|m| m.as_str().to_string())
        .collect();
        ours.sort();
        assert_eq!(wall, ours, "wall materials");

        let mut object: Vec<String> = enum_of("object_material");
        object.sort();
        let mut ours: Vec<String> = [
            ObjectMaterial::Wood,
            ObjectMaterial::Metal,
            ObjectMaterial::Fabric,
            ObjectMaterial::Glass,
        ]
        .iter()
        .map(|m| m.as_str().to_string())
        .collect();
        ours.sort();
        assert_eq!(object, ours, "object materials");

        let object_def = json::get(defs, "object")
            .unwrap()
            .as_object("object")
            .unwrap();
        let props = json::get(object_def, "properties")
            .unwrap()
            .as_object("properties")
            .unwrap();
        let type_def = json::get(props, "type").unwrap().as_object("type").unwrap();
        let mut types: Vec<String> = json::get(type_def, "enum")
            .unwrap()
            .as_array("enum")
            .unwrap()
            .iter()
            .map(|v| v.as_string("enum item").unwrap().to_string())
            .collect();
        types.sort();
        let mut ours: Vec<String> = [
            ObjectType::Bed,
            ObjectType::Table,
            ObjectType::Chair,
            ObjectType::Almirah,
            ObjectType::Sofa,
            ObjectType::Door,
            ObjectType::Window,
        ]
        .iter()
        .map(|t| t.as_str().to_string())
        .collect();
        ours.sort();
        assert_eq!(types, ours, "object types");

        let props = json::get(root, "properties")
            .unwrap()
            .as_object("properties")
            .unwrap();
        let version = json::get(props, "schema_version")
            .unwrap()
            .as_object("schema_version")
            .unwrap();
        assert_eq!(
            json::get(version, "const")
                .unwrap()
                .as_number("const")
                .unwrap(),
            SCHEMA_VERSION as f64
        );
    }

    // ---- what a schema cannot check -------------------------------------

    fn bedroom_with(find: &str, replace: &str) -> Result<Scene, SceneError> {
        assert!(
            BEDROOM.contains(find),
            "fixture text {find:?} is not in bedroom.json"
        );
        Scene::from_json(&BEDROOM.replace(find, replace))
    }

    #[test]
    fn a_negative_dimension_is_refused() {
        let err = bedroom_with("\"width_m\": 4.2", "\"width_m\": -4.2").unwrap_err();
        assert_eq!(err.path, "scene.room.width_m");
        assert!(err.message.contains("positive"), "{err}");
    }

    #[test]
    fn furniture_outside_the_room_is_refused() {
        // Push the almirah through the east wall.
        let err = bedroom_with(
            "\"centre_m\": [3.88, 1.2, 1.0]",
            "\"centre_m\": [4.5, 1.2, 1.0]",
        )
        .unwrap_err();
        assert!(err.path.starts_with("scene.objects"), "{err}");
        assert!(err.message.contains("outside the room"), "{err}");
    }

    #[test]
    fn furniture_through_the_ceiling_is_refused() {
        let err = bedroom_with(
            "\"centre_m\": [3.88, 1.2, 1.0]",
            "\"centre_m\": [3.88, 1.2, 2.0]",
        )
        .unwrap_err();
        assert!(err.message.contains("in z"), "{err}");
    }

    #[test]
    fn a_node_outside_the_room_is_refused() {
        let err =
            bedroom_with("\"pos_m\": [4.1, 2.4, 1.2]", "\"pos_m\": [4.3, 2.4, 1.2]").unwrap_err();
        assert!(err.message.contains("outside the room"), "{err}");
    }

    #[test]
    fn two_nodes_at_one_point_are_refused() {
        let err =
            bedroom_with("\"pos_m\": [4.1, 2.4, 1.2]", "\"pos_m\": [0.1, 0.9, 1.2]").unwrap_err();
        assert!(err.message.contains("same point"), "{err}");
    }

    #[test]
    fn a_duplicate_node_id_is_refused() {
        let err = bedroom_with("{ \"id\": 1, \"label\"", "{ \"id\": 0, \"label\"").unwrap_err();
        assert!(err.message.contains("already used"), "{err}");
    }

    #[test]
    fn a_node_outside_the_l_is_refused() {
        // (7.0, 5.0) is inside the 8 x 6 bounding box and outside the
        // floor. A loader that checked the box instead of the polygon
        // would accept it.
        let text = HALL_L.replace("\"pos_m\": [7.88, 0.6, 1.2]", "\"pos_m\": [7.0, 5.0, 1.2]");
        let err = Scene::from_json(&text).unwrap_err();
        assert!(err.message.contains("outside the room"), "{err}");
    }

    #[test]
    fn a_polygon_that_disagrees_with_the_room_is_refused() {
        let err =
            Scene::from_json(&HALL_L.replace("\"width_m\": 8.0", "\"width_m\": 9.0")).unwrap_err();
        assert_eq!(err.path, "scene.room.polygon_m");
        assert!(err.message.contains("bounding box"), "{err}");
    }

    #[test]
    fn a_clockwise_polygon_is_refused() {
        // Same six vertices, walked the other way round. The bounding box
        // still matches; only the winding is wrong, and the winding is
        // what tells an inside from an outside.
        let forward = "      [0.0, 0.0],\n      [8.0, 0.0],\n      [8.0, 2.5],\n      \
                       [4.5, 2.5],\n      [4.5, 6.0],\n      [0.0, 6.0]";
        let backward = "      [0.0, 0.0],\n      [0.0, 6.0],\n      [4.5, 6.0],\n      \
                        [4.5, 2.5],\n      [8.0, 2.5],\n      [8.0, 0.0]";
        assert!(HALL_L.contains(forward), "polygon fixture text has moved");
        let err = Scene::from_json(&HALL_L.replace(forward, backward)).unwrap_err();
        assert!(err.message.contains("counter-clockwise"), "{err}");
    }

    #[test]
    fn an_unknown_key_is_refused() {
        let err = bedroom_with(
            "\"schema_version\": 1,",
            "\"schema_version\": 1, \"colour\": \"blue\",",
        )
        .unwrap_err();
        assert!(err.message.contains("unknown key"), "{err}");
    }

    #[test]
    fn a_future_schema_version_is_refused_not_guessed_at() {
        let err = bedroom_with("\"schema_version\": 1,", "\"schema_version\": 2,").unwrap_err();
        assert_eq!(err.path, "scene.schema_version");
        assert!(err.message.contains("Refusing"), "{err}");
    }

    #[test]
    fn a_bad_material_names_the_alternatives() {
        let err = bedroom_with("\"material\": \"brick\"", "\"material\": \"cheese\"").unwrap_err();
        assert!(err.message.contains("cheese"), "{err}");
        assert!(err.message.contains("concrete"), "{err}");
    }

    // ---- the JSON reader -------------------------------------------------

    #[test]
    fn the_json_reader_refuses_what_is_not_json() {
        for bad in [
            "",
            "{",
            "{}{}",
            "{\"a\": 1,}",
            "[1, 2,]",
            "{'a': 1}",
            "{\"a\": 01}",
            "{\"a\": .5}",
            "{\"a\": 1e}",
            "{\"a\": NaN}",
            "{\"a\": Infinity}",
            "{\"a\": 1e400}",
            "{\"a\": 1, \"a\": 2}",
            "{\"a\": \"unterminated",
            "// a comment\n{}",
        ] {
            assert!(
                json::parse(bad).is_err(),
                "{bad:?} parsed and should not have"
            );
        }
    }

    #[test]
    fn the_json_reader_reads_what_is_json() {
        assert_eq!(json::parse(" 1.5e-3 ").unwrap(), json::Json::Number(1.5e-3));
        assert_eq!(
            json::parse("\"a\\u0041\\n\"").unwrap(),
            json::Json::String("aA\n".to_string())
        );
        // A surrogate pair: U+1F600, which a naive reader turns into two
        // broken characters or a panic.
        assert_eq!(
            json::parse("\"\\ud83d\\ude00\"").unwrap(),
            json::Json::String("\u{1f600}".to_string())
        );
        assert_eq!(
            json::parse("\"ü\"").unwrap(),
            json::Json::String("ü".to_string())
        );
        assert_eq!(json::parse("[]").unwrap(), json::Json::Array(vec![]));
        assert_eq!(json::parse("{}").unwrap(), json::Json::Object(vec![]));
        assert!(json::parse("\"\\ud83d\"").is_err(), "a lone high surrogate");
        assert!(json::parse("\"\\udc00\"").is_err(), "a lone low surrogate");
    }

    #[test]
    fn deep_nesting_is_an_error_and_not_a_stack_overflow() {
        let deep = format!("{}{}", "[".repeat(200), "]".repeat(200));
        let err = json::parse(&deep).unwrap_err();
        assert!(err.message.contains("nested deeper"), "{err}");
    }

    #[test]
    fn an_error_says_where_it_is() {
        let err = json::parse("{\n  \"a\": 1,\n  \"b\": x\n}").unwrap_err();
        assert!(err.message.contains("line 3"), "{err}");
    }
}
