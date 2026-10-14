//! Declared room geometry for Saans.
//!
//! A *scene* says what a room is: its shell, its walls and their
//! materials, the furniture in it, where the ESP32 nodes are, and
//! optionally where people and one tracked phone start. It is the input
//! to the simulator and to the 3-D demo.
//!
//! # Geometry is declared, never inferred
//!
//! Everything in a scene is labelled SCENE: a human measured it, or a
//! RoomPlan scan produced it. None of it is measured by WiFi and none of
//! it may ever be written back from CSI. An ESP32 at 20 MHz with one
//! antenna cannot measure a room's dimensions or recognise a sofa, and a
//! system that drew a floor plan from radio would be inventing it. This
//! crate exists so that the demo can draw a room honestly: the walls come
//! from the scene file, and only the person's position is estimated.
//!
//! This is also the piece COMPLETION_REPORT §9.3 named as the first of
//! three things a floor-plan view needs ("a config schema for
//! coordinates"); the other two, a survey of each room and an honest way
//! to draw what is not covered, are not this crate's job.
//!
//! # Frame and units
//!
//! Metres, right-handed, +z up, origin at a room corner. The floor plane
//! (x, y) is exactly `saans_sim::scenario::Point`, so a scene maps onto a
//! simulator scenario without a coordinate change; z is new here and the
//! 2-D simulator ignores it today.
//!
//! # Example
//!
//! ```no_run
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let scene = saans_scene::Scene::load("spec/scenes/bedroom.json")?;
//! println!("{} is {} x {} m", scene.name, scene.room.w_m, scene.room.d_m);
//! # Ok(()) }
//! ```

pub mod geom;
pub mod roomplan;
pub mod scene;
pub mod validate;

pub use roomplan::{import_roomplan_str, RoomPlanImport, RoomPlanWarning};
pub use scene::{
    Node, Object, ObjectKind, ObjectMaterial, Person, Phone, Point, Point3, Room, Scene, Wall, WallMaterial,
    DEFAULT_WALL_THICKNESS_M, SCHEMA_VERSION,
};
pub use validate::SceneError;

use std::fmt;
use std::path::Path;

/// Why a scene could not be loaded.
#[derive(Debug)]
pub enum LoadError {
    Io { path: String, source: std::io::Error },
    /// The bytes are not the JSON this crate expects.
    Json { path: String, source: serde_json::Error },
    /// The JSON parsed but the room it describes is not a room.
    Invalid { path: String, errors: Vec<SceneError> },
}

impl fmt::Display for LoadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io { path, source } => write!(f, "{path}: {source}"),
            Self::Json { path, source } => write!(f, "{path}: {source}"),
            Self::Invalid { path, errors } => {
                writeln!(f, "{path}: {} problem(s) with this scene:", errors.len())?;
                for e in errors {
                    writeln!(f, "  - {e}")?;
                }
                Ok(())
            }
        }
    }
}

impl std::error::Error for LoadError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io { source, .. } => Some(source),
            Self::Json { source, .. } => Some(source),
            Self::Invalid { .. } => None,
        }
    }
}

impl Scene {
    /// Parse and validate a scene from JSON text.
    ///
    /// `label` names the source in error messages (a path, or something
    /// like `"<stdin>"`).
    pub fn from_json_str(label: &str, text: &str) -> Result<Self, LoadError> {
        let scene: Scene =
            serde_json::from_str(text).map_err(|e| LoadError::Json { path: label.to_string(), source: e })?;
        let errors = scene.validate();
        if errors.is_empty() {
            Ok(scene)
        } else {
            Err(LoadError::Invalid { path: label.to_string(), errors })
        }
    }

    /// Read, parse and validate a scene file.
    pub fn load(path: impl AsRef<Path>) -> Result<Self, LoadError> {
        let path = path.as_ref();
        let label = path.display().to_string();
        let text = std::fs::read_to_string(path).map_err(|e| LoadError::Io { path: label.clone(), source: e })?;
        Self::from_json_str(&label, &text)
    }

    /// Serialize as the pretty JSON a human will edit, with a trailing
    /// newline so the file ends the way every other text file does.
    pub fn to_json_pretty(&self) -> String {
        let mut s = serde_json::to_string_pretty(self).expect("a Scene always serializes");
        s.push('\n');
        s
    }

    /// Write the scene to a file.
    pub fn save(&self, path: impl AsRef<Path>) -> std::io::Result<()> {
        std::fs::write(path, self.to_json_pretty())
    }
}
