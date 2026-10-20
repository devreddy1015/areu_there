//! The Apple RoomPlan importer.
//!
//! The fixture is built by hand in ARKit coordinates (+y up, floor in the
//! x-z plane) from a room whose scene-space answer is known in advance: a
//! 4 x 3 m room, 2.5 m high, with a table, a bed, a door, a window, one
//! opening and one object category the scene format has no box for. The
//! expected numbers below are read off that construction, not off the
//! importer.

use saans_scene::roomplan::{import_roomplan_str, RoomPlanWarning};
use saans_scene::scene::{Node, ObjectKind, WallMaterial};

/// A 4x4 transform as Apple writes it: 16 numbers, column-major, with
/// `col0` the element's local +x and `col3` its translation.
fn xform(col0: [f64; 3], col1: [f64; 3], col2: [f64; 3], t: [f64; 3]) -> String {
    format!(
        "[{},{},{},0, {},{},{},0, {},{},{},0, {},{},{},1]",
        col0[0], col0[1], col0[2], col1[0], col1[1], col1[2], col2[0], col2[1], col2[2], t[0], t[1], t[2]
    )
}

/// The four walls of a 4 (x) by 3 (scene y) metre room, 2.5 m tall, with
/// the floor at ARKit y = 0. Scene y maps to -z, so the far wall is at
/// z = -3.
fn fixture() -> String {
    let up = [0.0, 1.0, 0.0];
    // Each wall's local +x runs along its length.
    let south = xform([1.0, 0.0, 0.0], up, [0.0, 0.0, 1.0], [2.0, 1.25, 0.0]);
    let east = xform([0.0, 0.0, -1.0], up, [1.0, 0.0, 0.0], [4.0, 1.25, -1.5]);
    let north = xform([-1.0, 0.0, 0.0], up, [0.0, 0.0, -1.0], [2.0, 1.25, -3.0]);
    let west = xform([0.0, 0.0, 1.0], up, [-1.0, 0.0, 0.0], [0.0, 1.25, -1.5]);
    let ident = xform([1.0, 0.0, 0.0], up, [0.0, 0.0, 1.0], [1.5, 0.375, -1.0]);
    let bed = xform([1.0, 0.0, 0.0], up, [0.0, 0.0, 1.0], [3.0, 0.25, -2.0]);
    let door = xform([1.0, 0.0, 0.0], up, [0.0, 0.0, 1.0], [0.6, 1.0, 0.0]);
    let window = xform([1.0, 0.0, 0.0], up, [0.0, 0.0, 1.0], [2.5, 1.5, 0.0]);
    let toilet = xform([1.0, 0.0, 0.0], up, [0.0, 0.0, 1.0], [3.5, 0.4, -0.5]);
    let opening = xform([1.0, 0.0, 0.0], up, [0.0, 0.0, 1.0], [3.0, 1.0, -3.0]);
    format!(
        r#"{{
          "identifier": "fixture",
          "version": 2,
          "walls": [
            {{ "dimensions": [4.0, 2.5, 0.0], "transform": {south}, "category": {{"wall": {{}}}} }},
            {{ "dimensions": [3.0, 2.5, 0.0], "transform": {east},  "category": {{"wall": {{}}}} }},
            {{ "dimensions": [4.0, 2.5, 0.0], "transform": {north}, "category": {{"wall": {{}}}} }},
            {{ "dimensions": [3.0, 2.5, 0.0], "transform": {west},  "category": {{"wall": {{}}}} }}
          ],
          "doors":   [ {{ "dimensions": [0.9, 2.0, 0.0], "transform": {door},   "category": {{"door": {{}}}} }} ],
          "windows": [ {{ "dimensions": [1.2, 1.0, 0.0], "transform": {window}, "category": {{"window": {{}}}} }} ],
          "openings":[ {{ "dimensions": [1.0, 2.0, 0.0], "transform": {opening},"category": {{"opening": {{}}}} }} ],
          "objects": [
            {{ "dimensions": [1.2, 0.75, 0.8], "transform": {ident},  "category": {{"table": {{}}}} }},
            {{ "dimensions": [2.0, 0.5, 1.5],  "transform": {bed},    "category": {{"bed": {{}}}} }},
            {{ "dimensions": [0.4, 0.8, 0.4],  "transform": {toilet}, "category": {{"toilet": {{}}}} }}
          ]
        }}"#
    )
}

#[test]
fn a_scan_becomes_a_room_of_the_right_size() {
    let out = import_roomplan_str("fixture", &fixture()).expect("fixture should import");
    let r = &out.scene.room;
    assert!((r.w_m - 4.0).abs() < 1e-9, "width was {}", r.w_m);
    assert!((r.d_m - 3.0).abs() < 1e-9, "depth was {}", r.d_m);
    assert!((r.h_m - 2.5).abs() < 1e-9, "height was {}", r.h_m);
    assert_eq!(out.scene.walls.len(), 4);
}

#[test]
fn walls_come_back_as_the_segments_they_were_drawn_as() {
    let out = import_roomplan_str("fixture", &fixture()).unwrap();
    // Every wall endpoint must be a corner of the 4 x 3 room.
    let corners = [(0.0, 0.0), (4.0, 0.0), (4.0, 3.0), (0.0, 3.0)];
    let is_corner = |p: (f64, f64)| corners.iter().any(|c| (c.0 - p.0).abs() < 1e-9 && (c.1 - p.1).abs() < 1e-9);
    for (i, w) in out.scene.walls.iter().enumerate() {
        assert!(is_corner(w.from), "wall {i} starts at {:?}", w.from);
        assert!(is_corner(w.to), "wall {i} ends at {:?}", w.to);
    }
    // Their lengths are the room's sides: two of 4 m and two of 3 m.
    let mut lengths: Vec<f64> = out.scene.walls.iter().map(|w| w.length_m()).collect();
    lengths.sort_by(|a, b| a.partial_cmp(b).unwrap());
    assert!((lengths[0] - 3.0).abs() < 1e-9 && (lengths[3] - 4.0).abs() < 1e-9, "{lengths:?}");
}

#[test]
fn the_arkit_frame_is_mapped_not_copied() {
    let out = import_roomplan_str("fixture", &fixture()).unwrap();
    // The table sits at ARKit (1.5, 0.375, -1.0), which is scene
    // (1.5, 1.0, 0.375): z becomes -y, and up becomes z.
    let table = out.scene.objects.iter().find(|o| o.kind == ObjectKind::Table).expect("a table");
    assert!((table.pos.0 - 1.5).abs() < 1e-9, "x was {}", table.pos.0);
    assert!((table.pos.1 - 1.0).abs() < 1e-9, "y was {}", table.pos.1);
    assert!((table.pos.2 - 0.375).abs() < 1e-9, "z was {}", table.pos.2);
    // ARKit dimensions are (x, up, z); the scene wants (x, y, up).
    assert!((table.size.0 - 1.2).abs() < 1e-9);
    assert!((table.size.1 - 0.8).abs() < 1e-9, "depth was {}", table.size.1);
    assert!((table.size.2 - 0.75).abs() < 1e-9, "height was {}", table.size.2);
}

#[test]
fn recognised_categories_map_and_the_rest_are_dropped_loudly() {
    let out = import_roomplan_str("fixture", &fixture()).unwrap();
    let kinds: Vec<ObjectKind> = out.scene.objects.iter().map(|o| o.kind).collect();
    assert!(kinds.contains(&ObjectKind::Table));
    assert!(kinds.contains(&ObjectKind::Bed));
    assert!(kinds.contains(&ObjectKind::Door));
    assert!(kinds.contains(&ObjectKind::Window));
    // A toilet has no box in this format, and the scene must not pretend.
    assert!(out
        .warnings
        .iter()
        .any(|w| matches!(w, RoomPlanWarning::UnsupportedCategory { category, count: 1 } if category == "toilet")));
    assert_eq!(out.scene.objects.len(), 4);
}

#[test]
fn an_opening_is_dropped_and_said_so() {
    let out = import_roomplan_str("fixture", &fixture()).unwrap();
    assert!(out.warnings.contains(&RoomPlanWarning::OpeningDropped { count: 1 }));
}

#[test]
fn everything_roomplan_does_not_measure_is_warned_about() {
    let out = import_roomplan_str("fixture", &fixture()).unwrap();
    assert!(out.warnings.iter().any(|w| matches!(w, RoomPlanWarning::MaterialsAssumed { .. })));
    assert!(out.warnings.iter().any(|w| matches!(w, RoomPlanWarning::ThicknessAssumed { .. })));
    assert!(out.warnings.contains(&RoomPlanWarning::NoNodes));
    // ... and the guesses are the documented ones.
    assert!(out.scene.walls.iter().all(|w| w.material == WallMaterial::Drywall));
    assert!(out.scene.walls.iter().all(|w| (w.thickness_m - 0.1).abs() < 1e-12));
}

#[test]
fn an_imported_scene_is_valid_once_nodes_are_added() {
    let mut out = import_roomplan_str("fixture", &fixture()).unwrap();
    // A scan has no radios, so the scene is deliberately incomplete.
    assert!(!out.scene.validate().is_empty());
    out.scene.nodes = vec![Node { id: 0, pos: (0.5, 0.5, 1.2) }, Node { id: 1, pos: (3.5, 2.5, 1.2) }];
    assert_eq!(out.scene.validate(), Vec::new(), "{:?}", out.scene.validate());
}

/// Apple's `simd_float4x4` is written either as 16 flat numbers or as
/// four columns of four. Both must give the same wall.
#[test]
fn a_transform_written_as_four_columns_reads_the_same() {
    let nested = r#"{
      "walls": [ { "dimensions": [4.0, 2.5, 0.0],
        "transform": [[1,0,0,0],[0,1,0,0],[0,0,1,0],[2.0,1.25,0.0,1]],
        "category": "wall" } ]
    }"#;
    let flat = r#"{
      "walls": [ { "dimensions": [4.0, 2.5, 0.0],
        "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 2.0,1.25,0.0,1],
        "category": "wall" } ]
    }"#;
    let a = import_roomplan_str("nested", nested).expect("nested columns should parse");
    let b = import_roomplan_str("flat", flat).expect("flat columns should parse");
    assert_eq!(a.scene.walls, b.scene.walls);
    assert!((a.scene.walls[0].length_m() - 4.0).abs() < 1e-9);
}

#[test]
fn a_category_written_as_a_bare_string_reads_the_same() {
    let text = r#"{
      "walls": [ { "dimensions": [4.0, 2.5, 0.0],
        "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 2.0,1.25,0.0,1], "category": "wall" } ],
      "objects": [ { "dimensions": [1.2, 0.75, 0.8],
        "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 1.5,0.375,-1.0,1], "category": "table" } ]
    }"#;
    let out = import_roomplan_str("strings", text).unwrap();
    assert_eq!(out.scene.objects.len(), 1);
    assert_eq!(out.scene.objects[0].kind, ObjectKind::Table);
}

#[test]
fn something_that_is_not_a_captured_room_is_refused() {
    assert!(import_roomplan_str("x", "{}").is_err());
    assert!(import_roomplan_str("x", "not json").is_err());
    assert!(import_roomplan_str("x", r#"{"walls": []}"#).is_err());
}

#[test]
fn an_unreadable_element_is_skipped_rather_than_failing_the_import() {
    let text = r#"{
      "walls": [
        { "dimensions": [4.0, 2.5, 0.0], "transform": [1,0,0,0, 0,1,0,0, 0,0,1,0, 2.0,1.25,0.0,1], "category": "wall" },
        { "dimensions": "nonsense", "category": "wall" }
      ]
    }"#;
    let out = import_roomplan_str("partial", text).unwrap();
    assert_eq!(out.scene.walls.len(), 1);
    assert!(out.warnings.iter().any(|w| matches!(w, RoomPlanWarning::Unreadable { index: 1, .. })));
}

/// Regression: positions come out of a 4x4 transform and an `atan2`, so
/// before rounding an axis-aligned wall landed at 9.184850993605148e-17,
/// a 3 m room measured 3.0000000000000004 m, and every `yaw_deg` printed
/// as `-0.0`. Scene files are read, edited and diffed by people.
#[test]
fn imported_geometry_carries_no_floating_point_noise() {
    let out = import_roomplan_str("fixture", &fixture()).unwrap();
    assert_eq!(out.scene.room.w_m, 4.0);
    assert_eq!(out.scene.room.d_m, 3.0);
    assert_eq!(out.scene.room.h_m, 2.5);

    let tidy = |v: f64, what: &str| {
        assert_eq!(v, (v * 1e6).round() / 1e6, "{what} = {v} is not rounded to a micrometre");
        assert!(!(v == 0.0 && v.is_sign_negative()), "{what} is negative zero");
    };
    for (i, w) in out.scene.walls.iter().enumerate() {
        for p in [w.from, w.to] {
            tidy(p.0, &format!("walls[{i}] x"));
            tidy(p.1, &format!("walls[{i}] y"));
        }
    }
    for (i, o) in out.scene.objects.iter().enumerate() {
        tidy(o.pos.0, &format!("objects[{i}].pos.x"));
        tidy(o.pos.1, &format!("objects[{i}].pos.y"));
        tidy(o.pos.2, &format!("objects[{i}].pos.z"));
        tidy(o.yaw_deg, &format!("objects[{i}].yaw_deg"));
    }
    // The fixture is axis aligned, so every rotation should be exactly 0.
    assert!(out.scene.objects.iter().all(|o| o.yaw_deg == 0.0));
}
