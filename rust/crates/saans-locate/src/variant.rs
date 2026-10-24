//! The three run variants of the live demo (spec/livedemo.md §3.1), built
//! as `locate_measure` builds its own. SIM.
//!
//! The names are `declared`, `with_walker` and `empty`, never `walker`:
//! the shipped hall has a `Person.id` "walker", and a variant name travels
//! in every frame (ADR-008 decision 3).

use saans_roomsim::Config as RoomSimConfig;
use saans_scene::Scene;

/// Which people a Room Sim run of a scene has.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Variant {
    /// The scene's people as declared.
    Declared,
    /// The scene's people and one more, who enters at the door, walks and
    /// carries the phone.
    WithWalker,
    /// Nobody (`n_people = 0`); the phone stays at its scene start, z = 1.0.
    Empty,
}

impl Variant {
    pub const ALL: [Variant; 3] = [Variant::Declared, Variant::WithWalker, Variant::Empty];

    /// The wire name.
    pub fn as_str(self) -> &'static str {
        match self {
            Variant::Declared => "declared",
            Variant::WithWalker => "with_walker",
            Variant::Empty => "empty",
        }
    }

    /// The variant with this wire name, or `None`.
    pub fn parse(name: &str) -> Option<Variant> {
        Variant::ALL.into_iter().find(|v| v.as_str() == name)
    }

    /// `base` with this variant's people: `Empty` sets `n_people = 0`;
    /// `WithWalker` sets `n_people` to the scene's count + 1 and
    /// `phone_carrier` to the added person; `Declared` changes nothing.
    pub fn apply(self, scene: &Scene, base: &RoomSimConfig) -> RoomSimConfig {
        let mut rs = base.clone();
        match self {
            Variant::Declared => {}
            Variant::WithWalker => {
                let n_scene = scene.people.len();
                rs.n_people = Some(n_scene + 1);
                rs.phone_carrier = n_scene;
            }
            Variant::Empty => rs.n_people = Some(0),
        }
        rs
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_round_trip_and_unknown_names_are_none() {
        for v in Variant::ALL {
            assert_eq!(Variant::parse(v.as_str()), Some(v));
        }
        assert_eq!(Variant::parse("walker"), None);
        assert_eq!(Variant::parse("scene"), None);
        assert_eq!(Variant::parse(""), None);
    }
}
