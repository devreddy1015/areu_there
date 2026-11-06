//! Scene loading, validation and geometry.
//!
//! Numbers here are derived independently of the code under test: the
//! L-shaped hall's area is checked against 9x3 + 4x4 worked out by hand,
//! not against another call into `geom`.

use saans_scene::geom;
use saans_scene::scene::{Node, Object, ObjectKind, ObjectMaterial, Person, Phone, Provenance, Room, Scene, Wall, WallMaterial};
use saans_scene::validate::SceneError;
use saans_scene::SCHEMA_VERSION;

fn scenes_dir() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes")
}

fn load(name: &str) -> Scene {
    let path = scenes_dir().join(name);
    Scene::load(&path).unwrap_or_else(|e| panic!("{name} should load and validate:\n{e}"))
}

/// A minimal scene every mutation test starts from: a 4 x 3 x 2.5 m room
/// with two nodes, nothing else.
fn minimal() -> Scene {
    Scene {
        schema_version: SCHEMA_VERSION,
        name: "test".into(),
        provenance: Provenance::HandWritten,
        room: Room { w_m: 4.0, d_m: 3.0, h_m: 2.5, polygon: None },
        walls: Vec::new(),
        objects: Vec::new(),
        nodes: vec![Node { id: 0, pos: (0.5, 0.5, 1.0) }, Node { id: 1, pos: (3.5, 2.5, 1.0) }],
        people: Vec::new(),
        phone: None,
    }
}

#[test]
fn the_three_example_scenes_load_and_validate() {
    for name in ["bedroom.json", "office.json", "hall-l-shaped.json"] {
        let s = load(name);
        assert!(!s.name.is_empty());
        assert_eq!(s.schema_version, SCHEMA_VERSION);
        assert!(s.validate().is_empty(), "{name} reported {:?}", s.validate());
    }
}

#[test]
fn bedroom_is_the_size_it_claims() {
    let s = load("bedroom.json");
    assert_eq!((s.room.w_m, s.room.d_m), (4.2, 3.6));
    assert_eq!(s.nodes.len(), 4);
    // Four walls, and every one of them encloses the stated footprint:
    // the perimeter of a 4.2 x 3.6 m room is 2 * (4.2 + 3.6) = 15.6 m.
    let perimeter: f64 = s.walls.iter().map(|w| w.length_m()).sum();
    assert!((perimeter - 15.6).abs() < 1e-9, "perimeter was {perimeter}");
}

#[test]
fn l_shaped_hall_has_the_area_worked_out_by_hand() {
    let s = load("hall-l-shaped.json");
    let poly = s.footprint();
    assert_eq!(poly.len(), 6);
    // The L is a 9 x 3 strip plus a 4 x 4 block above its left end.
    let expected = 9.0 * 3.0 + 4.0 * 4.0;
    assert!((geom::area(&poly) - expected).abs() < 1e-9, "area was {}", geom::area(&poly));
}

#[test]
fn the_notch_of_an_l_shaped_room_is_outside_it() {
    let s = load("hall-l-shaped.json");
    let poly = s.footprint();
    // (7, 5) is in the bounding box but in the missing corner.
    assert!(!geom::point_in_polygon(&poly, (7.0, 5.0)));
    // ... while these are in the two legs.
    assert!(geom::point_in_polygon(&poly, (7.0, 1.5)));
    assert!(geom::point_in_polygon(&poly, (2.0, 5.0)));
}

#[test]
fn every_node_of_every_example_is_inside_its_room() {
    for name in ["bedroom.json", "office.json", "hall-l-shaped.json"] {
        let s = load(name);
        let poly = s.footprint();
        for n in &s.nodes {
            assert!(geom::point_in_polygon(&poly, (n.pos.0, n.pos.1)), "{name}: node {} is outside", n.id);
            assert!(n.pos.2 >= 0.0 && n.pos.2 <= s.room.h_m, "{name}: node {} is outside the room height", n.id);
        }
    }
}

#[test]
fn node_ids_are_unique_in_every_example() {
    for name in ["bedroom.json", "office.json", "hall-l-shaped.json"] {
        let s = load(name);
        let mut ids: Vec<u8> = s.nodes.iter().map(|n| n.id).collect();
        let before = ids.len();
        ids.sort_unstable();
        ids.dedup();
        assert_eq!(ids.len(), before, "{name} has a duplicate node id");
    }
}

#[test]
fn a_scene_round_trips_through_json() {
    for name in ["bedroom.json", "office.json", "hall-l-shaped.json"] {
        let s = load(name);
        let back = Scene::from_json_str(name, &s.to_json_pretty()).expect("re-reading our own output");
        assert_eq!(s, back, "{name} changed on a JSON round trip");
    }
}

/// A generated scene (spec/dataset.md) says so, and the word survives a
/// round trip: it is the only thing telling a training scene from a room.
#[test]
fn a_generated_scene_says_so_on_a_round_trip() {
    let mut s = minimal();
    s.provenance = Provenance::Generated;
    let text = s.to_json_pretty();
    assert!(text.contains("\"provenance\": \"generated\""), "{text}");
    let back = Scene::from_json_str("generated", &text).expect("re-reading our own output");
    assert_eq!(back.provenance, Provenance::Generated);
    assert_eq!(s, back);
}

/// The written file and the Rust types must not drift apart: anything
/// serde renames or defaults away shows up here as a difference between
/// the file on disk and the file we would write.
#[test]
fn the_example_files_are_what_this_crate_would_write() {
    for name in ["bedroom.json", "office.json", "hall-l-shaped.json"] {
        let s = load(name);
        let on_disk: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(scenes_dir().join(name)).unwrap()).unwrap();
        let ours: serde_json::Value = serde_json::from_str(&s.to_json_pretty()).unwrap();
        assert_eq!(on_disk, ours, "{name} is not what saans-scene would write for the same scene");
    }
}

#[test]
fn a_good_scene_has_no_errors() {
    assert_eq!(minimal().validate(), Vec::new());
}

#[test]
fn an_unknown_schema_version_is_rejected() {
    let mut s = minimal();
    s.schema_version = 99;
    assert!(s
        .validate()
        .contains(&SceneError::UnknownSchemaVersion { found: 99, supported: SCHEMA_VERSION }));
}

#[test]
fn a_room_with_no_size_is_rejected() {
    for (field, set) in [
        ("w_m", (|s: &mut Scene| s.room.w_m = 0.0) as fn(&mut Scene)),
        ("d_m", |s: &mut Scene| s.room.d_m = -1.0),
        ("h_m", |s: &mut Scene| s.room.h_m = f64::NAN),
    ] {
        let mut s = minimal();
        set(&mut s);
        let errs = s.validate();
        assert!(
            errs.iter().any(|e| matches!(e, SceneError::BadRoomDimension { field: f, .. } if *f == field)),
            "{field} was accepted: {errs:?}"
        );
    }
}

#[test]
fn a_node_outside_the_room_is_rejected() {
    let mut s = minimal();
    s.nodes[1].pos = (4.5, 1.0, 1.0);
    assert!(s.validate().iter().any(|e| matches!(e, SceneError::NodeOutsideRoom { id: 1, .. })));
}

#[test]
fn a_node_flat_against_a_wall_is_rejected_because_the_simulator_rejects_it() {
    // saans-sim requires 0 < x < w strictly, so x = 0 cannot be simulated.
    let mut s = minimal();
    s.nodes[0].pos = (0.0, 1.5, 1.0);
    assert!(s.validate().iter().any(|e| matches!(e, SceneError::NodeOutsideRoom { id: 0, .. })));
}

#[test]
fn duplicate_node_ids_are_rejected() {
    let mut s = minimal();
    s.nodes[1].id = 0;
    assert!(s.validate().contains(&SceneError::DuplicateNodeId { id: 0 }));
}

#[test]
fn a_scene_with_no_nodes_is_rejected() {
    let mut s = minimal();
    s.nodes.clear();
    assert!(s.validate().contains(&SceneError::NoNodes));
}

#[test]
fn an_object_that_leaves_the_room_is_rejected() {
    let mut s = minimal();
    s.objects.push(Object {
        kind: ObjectKind::Table,
        id: None,
        pos: (3.9, 1.5, 0.375),
        size: (1.2, 0.6, 0.75),
        yaw_deg: 0.0,
        material: ObjectMaterial::Wood,
    });
    assert!(s.validate().iter().any(|e| matches!(e, SceneError::ObjectOutsideRoom { index: 0, .. })));
}

#[test]
fn an_object_taller_than_the_room_is_rejected() {
    let mut s = minimal();
    s.objects.push(Object {
        kind: ObjectKind::Almirah,
        id: None,
        pos: (2.0, 1.5, 1.5),
        size: (1.0, 0.6, 3.0),
        yaw_deg: 0.0,
        material: ObjectMaterial::Wood,
    });
    let errs = s.validate();
    assert!(errs.iter().any(|e| matches!(e, SceneError::ObjectAboveCeiling { .. })));
}

#[test]
fn rotating_an_object_can_push_it_out_of_the_room() {
    // A 2.0 x 0.4 m table fits across a 3 m depth at 0 degrees but not at
    // 90 degrees when its centre is 0.5 m from the wall.
    let make = |yaw: f64| {
        let mut s = minimal();
        s.objects.push(Object {
            kind: ObjectKind::Table,
            id: None,
            pos: (2.0, 0.5, 0.375),
            size: (2.0, 0.4, 0.75),
            yaw_deg: yaw,
            material: ObjectMaterial::Wood,
        });
        s
    };
    assert!(make(0.0).validate().is_empty(), "the unrotated table should fit");
    assert!(make(90.0).validate().iter().any(|e| matches!(e, SceneError::ObjectOutsideRoom { .. })));
}

#[test]
fn a_wall_thicker_than_a_metre_is_rejected_as_a_unit_mistake() {
    let mut s = minimal();
    s.walls.push(Wall {
        from: (0.0, 0.0),
        to: (4.0, 0.0),
        material: WallMaterial::Brick,
        thickness_m: 23.0,
        height_m: None,
    });
    assert!(s.validate().iter().any(|e| matches!(e, SceneError::WallBadThickness { index: 0, .. })));
}

#[test]
fn a_wall_taller_than_the_room_is_rejected() {
    let mut s = minimal();
    s.walls.push(Wall {
        from: (0.0, 0.0),
        to: (4.0, 0.0),
        material: WallMaterial::Brick,
        thickness_m: 0.23,
        height_m: Some(9.0),
    });
    assert!(s.validate().iter().any(|e| matches!(e, SceneError::WallBadHeight { index: 0, .. })));
}

#[test]
fn a_self_intersecting_floor_outline_is_rejected() {
    let mut s = minimal();
    // A bow tie across the 4 x 3 box.
    s.room.polygon = Some(vec![(0.0, 0.0), (4.0, 3.0), (4.0, 0.0), (0.0, 3.0)]);
    assert!(s.validate().iter().any(|e| matches!(e, SceneError::PolygonSelfIntersects { .. })));
}

#[test]
fn a_floor_outline_that_does_not_fill_its_declared_box_is_rejected() {
    let mut s = minimal();
    // Only 3 m of the declared 4 m width.
    s.room.polygon = Some(vec![(0.0, 0.0), (3.0, 0.0), (3.0, 3.0), (0.0, 3.0)]);
    assert!(s.validate().iter().any(|e| matches!(e, SceneError::PolygonBboxMismatch { .. })));
}

#[test]
fn a_person_or_phone_starting_outside_the_room_is_rejected() {
    let mut s = minimal();
    s.people.push(Person { id: None, start: Some((9.0, 9.0)) });
    s.phone = Some(Phone { id: None, start: (-1.0, 1.0) });
    let errs = s.validate();
    assert!(errs.iter().any(|e| matches!(e, SceneError::PersonOutsideRoom { .. })));
    assert!(errs.iter().any(|e| matches!(e, SceneError::PhoneOutsideRoom { .. })));
}

#[test]
fn validation_reports_every_problem_not_just_the_first() {
    let mut s = minimal();
    s.name = String::new();
    s.nodes[1].id = 0;
    s.nodes[1].pos = (99.0, 99.0, 1.0);
    assert!(s.validate().len() >= 3, "expected several errors, got {:?}", s.validate());
}

#[test]
fn malformed_json_is_an_error_not_a_panic() {
    assert!(Scene::from_json_str("<test>", "{ not json").is_err());
    assert!(Scene::from_json_str("<test>", r#"{"schema_version": 1}"#).is_err());
}

#[test]
fn a_rejected_scene_names_every_problem_in_its_message() {
    let mut s = minimal();
    s.nodes.clear();
    let text = serde_json::to_string(&s).unwrap();
    let err = Scene::from_json_str("<test>", &text).unwrap_err();
    assert!(err.to_string().contains("nodes is empty"), "message was: {err}");
}

#[test]
fn distance_to_a_segment_is_measured_from_its_ends_not_its_line() {
    // Point beyond the end of a segment: the nearest point is the end,
    // 3-4-5 away, not the perpendicular foot.
    let d = geom::distance_to_segment((0.0, 0.0), (0.0, 4.0), (3.0, 8.0));
    assert!((d - 5.0).abs() < 1e-12, "got {d}");
}

/// Regression: `phone` used to serialize as `"phone": null` when absent,
/// which this crate read back happily but `spec/scene.schema.json`
/// rejects (it has no null for that key). A file the Rust side accepts
/// and the schema refuses is exactly the drift these two are supposed to
/// prevent, so the key is omitted instead.
#[test]
fn a_scene_without_a_phone_writes_no_null() {
    let text = minimal().to_json_pretty();
    assert!(!text.contains("null"), "a null reached the file:\n{text}");
    assert!(!text.contains("\"phone\""), "an absent phone was written:\n{text}");
    // ... and it still reads back as absent.
    assert!(Scene::from_json_str("<test>", &text).unwrap().phone.is_none());
}

/// The examples are illustrative rooms, not surveys of real ones, and the
/// demo prints this field next to the drawing (ADR-008 decision 4). An
/// example that claimed to be `surveyed` would be a false statement on
/// screen.
#[test]
fn the_examples_admit_they_are_hand_written() {
    for name in ["bedroom.json", "office.json", "hall-l-shaped.json"] {
        assert_eq!(load(name).provenance, Provenance::HandWritten, "{name}");
    }
}

#[test]
fn a_scene_that_does_not_say_where_it_came_from_is_refused() {
    let mut v: serde_json::Value = serde_json::from_str(&minimal().to_json_pretty()).unwrap();
    v.as_object_mut().unwrap().remove("provenance");
    assert!(Scene::from_json_str("<test>", &v.to_string()).is_err());
}

/// Every file in spec/scenes/, not a fixed list: the gateway serves
/// whatever is there (its own test keeps the two sets equal), so a scene
/// added without being listed here must still be validated.
#[test]
fn every_scene_file_in_spec_scenes_loads_and_validates() {
    let mut n = 0;
    for entry in std::fs::read_dir(scenes_dir()).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "json") {
            Scene::load(&path).unwrap_or_else(|e| panic!("{}:\n{e}", path.display()));
            n += 1;
        }
    }
    assert!(n >= 3, "found only {n} scene files");
}

/// Test fixtures (spec/scenes/fixtures/, used by the Room Sim golden
/// vectors) are not served by the gateway but must be valid scenes too.
#[test]
fn every_fixture_scene_loads_and_validates() {
    let mut n = 0;
    for entry in std::fs::read_dir(scenes_dir().join("fixtures")).unwrap() {
        let path = entry.unwrap().path();
        if path.extension().is_some_and(|e| e == "json") {
            Scene::load(&path).unwrap_or_else(|e| panic!("{}:\n{e}", path.display()));
            n += 1;
        }
    }
    assert!(n >= 1, "found no fixture scenes");
}
