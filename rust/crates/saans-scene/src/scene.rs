//! The `scene.json` contract: what a room *is*, declared by a human or by
//! a RoomPlan scan.
//!
//! Every value here is DECLARED geometry, labelled SCENE. Nothing in this
//! module is measured by WiFi and nothing here may ever be inferred from
//! CSI: an ESP32 at 20 MHz with one antenna cannot measure a room's
//! dimensions or recognise a sofa. The simulator reads a scene to decide
//! what the radio should do; the radio never writes one back.
//!
//! Coordinates are METRES in a right-handed frame with +z up:
//!   * the floor is the z = 0 plane,
//!   * x runs 0..`Room::w_m`, y runs 0..`Room::d_m`, z runs 0..`Room::h_m`,
//!   * the origin is a room corner, matching `saans_sim::scenario::Point`
//!     (which is this frame's (x, y) with z dropped).
//!
//! Angles are DEGREES counter-clockwise about +z, measured from +x.

use serde::Deserialize;
#[cfg(feature = "write")]
use serde::Serialize;

/// A point on the floor plane, metres. Same convention as `saans_sim::Point`.
pub type Point = (f64, f64);

/// A point in the room, metres, z measured up from the floor.
pub type Point3 = (f64, f64, f64);

/// The scene format version this crate reads and writes.
pub const SCHEMA_VERSION: u32 = 1;

/// Wall thickness assumed when a source (e.g. RoomPlan) does not measure one.
pub const DEFAULT_WALL_THICKNESS_M: f64 = 0.1;

/// One room, everything in it, and where the radios are.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
pub struct Scene {
    /// Format version. Readers reject a version they do not know.
    pub schema_version: u32,
    pub name: String,
    /// Where this geometry came from. Shown next to every drawing of the
    /// scene (ADR-008 decision 4), so a viewer can tell a surveyed room
    /// from a scan with guessed materials from an illustrative example.
    pub provenance: Provenance,
    pub room: Room,
    /// Wall segments. May be empty: a bare `Room` already implies four
    /// walls around its footprint, and the simulator's image method uses
    /// that footprint. Listing walls explicitly is how you give them
    /// materials, thicknesses, or a shape the footprint does not have.
    #[serde(default)]
    pub walls: Vec<Wall>,
    #[serde(default)]
    pub objects: Vec<Object>,
    pub nodes: Vec<Node>,
    /// Where people start. Optional because most scenes describe a room,
    /// not a rehearsal; the timeline decides what anyone does.
    #[serde(default)]
    pub people: Vec<Person>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub phone: Option<Phone>,
}

/// How a scene's geometry was obtained.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
#[serde(rename_all = "snake_case")]
pub enum Provenance {
    /// Measured in the real room by a person with a tape or a laser.
    Surveyed,
    /// Produced by `saans import-roomplan` from an Apple RoomPlan scan:
    /// the geometry is measured, but materials and wall thickness are
    /// guesses until someone edits them.
    RoomplanImport,
    /// Written by hand without measuring a particular room: an example,
    /// a test fixture, or a plan drawing.
    HandWritten,
    /// Drawn at random by a program (`saans gen-dataset`, spec/dataset.md
    /// §2): a simulated room for training data, describing no real room.
    Generated,
}

/// The room shell: a box, or a box plus a floor outline for an L-shape.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
pub struct Room {
    /// Extent along x, metres.
    pub w_m: f64,
    /// Extent along y ("depth"), metres.
    pub d_m: f64,
    /// Floor-to-ceiling height, metres.
    pub h_m: f64,
    /// Floor outline for non-rectangular rooms (L-shaped halls and the
    /// like), as a simple polygon in metres. When present its bounding
    /// box must be exactly (0, 0)..(`w_m`, `d_m`), so `w_m`/`d_m` stay
    /// honest as the room's extent. When absent the footprint is that
    /// whole rectangle.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub polygon: Option<Vec<Point>>,
}

/// A wall segment on the floor plane, extruded upwards.
///
/// `from` and `to` are the ends of the wall's CENTRELINE: half of
/// `thickness_m` lies on each side. A wall along the footprint's edge is
/// therefore half outside the room, and a door or window in it is centred
/// on the same line (which is why validation only asks an opening's
/// centre to be inside the footprint).
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
pub struct Wall {
    pub from: Point,
    pub to: Point,
    pub material: WallMaterial,
    pub thickness_m: f64,
    /// Height of this wall, metres. Defaults to the room height; a lower
    /// value describes a half-wall or a parapet.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub height_m: Option<f64>,
}

/// Bulk wall materials. The set is deliberately small: these are the
/// materials a 2.4 GHz link actually behaves differently through, and
/// each one has published ITU-R P.2040 constants.
///
/// NOTE: this enum names a material, it does not carry its electrical
/// constants. Permittivity and conductivity belong to the propagation
/// model, and under ADR-002 any physics change starts in the Python
/// reference and is then ported, so those numbers arrive with that work
/// rather than here.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
#[serde(rename_all = "snake_case")]
pub enum WallMaterial {
    Concrete,
    Brick,
    Drywall,
    Glass,
    Wood,
}

/// A piece of furniture or an opening, as an upright box.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
pub struct Object {
    /// Serialized as `"type"`, which reads better in a hand-written scene.
    #[serde(rename = "type")]
    pub kind: ObjectKind,
    /// Optional stable name, useful for clicking an object in the demo.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Centre of the box, metres. `pos.2` is the centre's height above
    /// the floor, so a 0.75 m table sits at z = 0.375.
    pub pos: Point3,
    /// Extent along the object's own x, y and z before rotation, metres.
    pub size: Point3,
    /// Rotation about +z, degrees counter-clockwise from +x.
    #[serde(default)]
    pub yaw_deg: f64,
    pub material: ObjectMaterial,
}

/// What a box is. Doors and windows are objects rather than wall
/// attributes because the demo has to draw them and the simulator has to
/// treat them as a different kind of obstacle from the wall they sit in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
#[serde(rename_all = "snake_case")]
pub enum ObjectKind {
    Bed,
    Table,
    Chair,
    /// Free-standing wardrobe (Indian English, kept because that is what
    /// the rooms this is built for contain).
    Almirah,
    Sofa,
    Door,
    Window,
}

/// Dominant surface material of an object.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
#[serde(rename_all = "snake_case")]
pub enum ObjectMaterial {
    Wood,
    Metal,
    Fabric,
    Glass,
}

/// An ESP32 node.
#[derive(Debug, Clone, Copy, PartialEq, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
pub struct Node {
    /// Node id, matching `saans_sim::scenario::Node::id` so a scene can
    /// drive the simulator without a lookup table.
    pub id: u8,
    /// Position, metres, z measured up from the floor.
    pub pos: Point3,
}

/// A person the scene expects to be in the room.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
pub struct Person {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Where they start, metres on the floor plane. Absent means "the
    /// timeline decides", which for the simulator means the door.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub start: Option<Point>,
}

/// The one tracked phone.
#[derive(Debug, Clone, PartialEq, Deserialize)]
#[cfg_attr(feature = "write", derive(Serialize))]
pub struct Phone {
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub id: Option<String>,
    /// Where it starts, metres on the floor plane.
    pub start: Point,
}

impl Default for Room {
    /// The simulator's default room (`saans_sim::Room::default()` is
    /// 6.0 x 5.0 m) at a typical Indian ceiling height, so a scene and a
    /// bare simulator scenario describe the same box.
    fn default() -> Self {
        Self { w_m: 6.0, d_m: 5.0, h_m: 2.7, polygon: None }
    }
}

impl Wall {
    /// Height this wall is drawn and simulated at, given its room.
    pub fn height_or(&self, room_h_m: f64) -> f64 {
        self.height_m.unwrap_or(room_h_m)
    }

    /// Length of the segment on the floor plane, metres.
    pub fn length_m(&self) -> f64 {
        let (dx, dy) = (self.to.0 - self.from.0, self.to.1 - self.from.1);
        dx.hypot(dy)
    }
}

impl Object {
    /// Axis-aligned bounding box of the rotated box on the floor plane,
    /// as ((x0, y0), (x1, y1)) in metres.
    ///
    /// A rotated rectangle's half-extent along x is
    /// `(w|cos t| + d|sin t|) / 2`, and along y `(w|sin t| + d|cos t|) / 2`.
    pub fn footprint_bbox(&self) -> (Point, Point) {
        let t = self.yaw_deg.to_radians();
        let (c, s) = (t.cos().abs(), t.sin().abs());
        let hx = (self.size.0 * c + self.size.1 * s) / 2.0;
        let hy = (self.size.0 * s + self.size.1 * c) / 2.0;
        ((self.pos.0 - hx, self.pos.1 - hy), (self.pos.0 + hx, self.pos.1 + hy))
    }

    /// Vertical extent of the box, metres above the floor.
    pub fn z_extent(&self) -> (f64, f64) {
        (self.pos.2 - self.size.2 / 2.0, self.pos.2 + self.size.2 / 2.0)
    }
}

impl Scene {
    /// The room's floor outline, as a polygon in metres. A rectangular
    /// room becomes its four corners, so callers have one code path.
    pub fn footprint(&self) -> Vec<Point> {
        match &self.room.polygon {
            Some(p) => p.clone(),
            None => vec![(0.0, 0.0), (self.room.w_m, 0.0), (self.room.w_m, self.room.d_m), (0.0, self.room.d_m)],
        }
    }

    /// Node with this id, if the scene has one.
    pub fn node(&self, id: u8) -> Option<&Node> {
        self.nodes.iter().find(|n| n.id == id)
    }
}
