//! Scene validation.
//!
//! A scene is the one place room geometry enters the system, so a wrong
//! scene is a wrong answer everywhere downstream and nowhere else will
//! catch it. The recurring defect in this repo is "configuration the
//! validator accepts and the runtime cannot honour" (COMPLETION_REPORT
//! §1), so this module rejects loudly and says exactly which element is
//! wrong, rather than clamping a value the way the simulator quietly
//! does with out-of-bounds activity positions.
//!
//! [`Scene::validate`] returns EVERY problem it finds, not just the
//! first: fixing a hand-written scene one error per run is miserable.

use crate::geom::{self, EPS_M};
use crate::scene::{Object, ObjectKind, Point, Scene};
use std::fmt;

/// Largest wall thickness a scene may declare, metres. A metre-thick wall
/// in a 3 m room is a unit mistake (centimetres typed as metres), and
/// that is the mistake worth catching.
pub const MAX_WALL_THICKNESS_M: f64 = 1.0;

/// How far a node must sit from any wall, metres.
///
/// `saans_sim::Simulator::try_new` rejects a node that is not strictly
/// inside the room (`0 < x < w`), so a scene with a node flat against a
/// wall cannot drive the simulator. Nodes are physical boxes anyway: the
/// antenna is centimetres off the plaster, not in it.
pub const NODE_WALL_CLEARANCE_M: f64 = 0.01;

/// One thing wrong with a scene.
#[derive(Debug, Clone, PartialEq)]
pub enum SceneError {
    /// The file declares a format version this build does not know.
    UnknownSchemaVersion { found: u32, supported: u32 },
    EmptyName,
    /// A room dimension is zero, negative, or not a number.
    BadRoomDimension { field: &'static str, value: f64 },
    PolygonTooShort { vertices: usize },
    PolygonNotFinite { index: usize },
    /// Two edges of the floor outline cross or touch.
    PolygonSelfIntersects { edge_a: usize, edge_b: usize },
    PolygonZeroArea,
    /// The outline's bounding box is not the declared `w_m` x `d_m`.
    PolygonBboxMismatch { got: (Point, Point), want: (Point, Point) },
    WallNotFinite { index: usize },
    WallZeroLength { index: usize },
    WallBadThickness { index: usize, value: f64 },
    WallBadHeight { index: usize, value: f64, room_h_m: f64 },
    WallOutsideRoom { index: usize, point: Point },
    ObjectNotFinite { index: usize },
    ObjectBadSize { index: usize, size: (f64, f64, f64) },
    ObjectOutsideRoom { index: usize, bbox: (Point, Point) },
    ObjectAboveCeiling { index: usize, z_top: f64, room_h_m: f64 },
    ObjectBelowFloor { index: usize, z_bottom: f64 },
    NoNodes,
    DuplicateNodeId { id: u8 },
    NodeNotFinite { index: usize, id: u8 },
    /// Outside the footprint, or closer to a wall than a node can be.
    NodeOutsideRoom { id: u8, pos: (f64, f64, f64) },
    NodeBadHeight { id: u8, z_m: f64, room_h_m: f64 },
    PersonOutsideRoom { index: usize, start: Point },
    PhoneOutsideRoom { start: Point },
}

impl fmt::Display for SceneError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::UnknownSchemaVersion { found, supported } => {
                write!(f, "schema_version {found} is not supported (this build reads version {supported})")
            }
            Self::EmptyName => write!(f, "name is empty: a scene is referred to by name in reports and the demo"),
            Self::BadRoomDimension { field, value } => {
                write!(f, "room.{field} = {value} m: must be finite and greater than zero")
            }
            Self::PolygonTooShort { vertices } => {
                write!(f, "room.polygon has {vertices} vertices: a floor outline needs at least 3")
            }
            Self::PolygonNotFinite { index } => write!(f, "room.polygon[{index}] is not a finite coordinate"),
            Self::PolygonSelfIntersects { edge_a, edge_b } => {
                write!(f, "room.polygon edges {edge_a} and {edge_b} cross or touch: the floor outline must be a simple polygon")
            }
            Self::PolygonZeroArea => write!(f, "room.polygon encloses no area"),
            Self::PolygonBboxMismatch { got, want } => write!(
                f,
                "room.polygon spans ({:.3}, {:.3})..({:.3}, {:.3}) but room is {:.3} x {:.3} m: the outline's bounding box must be (0, 0)..(w_m, d_m) so w_m and d_m stay truthful",
                got.0 .0, got.0 .1, got.1 .0, got.1 .1, want.1 .0, want.1 .1
            ),
            Self::WallNotFinite { index } => write!(f, "walls[{index}] has a non-finite endpoint"),
            Self::WallZeroLength { index } => write!(f, "walls[{index}] has zero length"),
            Self::WallBadThickness { index, value } => write!(
                f,
                "walls[{index}].thickness_m = {value}: must be finite, greater than zero and at most {MAX_WALL_THICKNESS_M} m (a thicker wall is usually centimetres typed as metres)"
            ),
            Self::WallBadHeight { index, value, room_h_m } => {
                write!(f, "walls[{index}].height_m = {value} m: must be finite, greater than zero and at most the room height {room_h_m} m")
            }
            Self::WallOutsideRoom { index, point } => {
                write!(f, "walls[{index}] endpoint ({:.3}, {:.3}) is outside the room footprint", point.0, point.1)
            }
            Self::ObjectNotFinite { index } => write!(f, "objects[{index}] has a non-finite position, size or rotation"),
            Self::ObjectBadSize { index, size } => write!(
                f,
                "objects[{index}].size = ({}, {}, {}) m: every extent must be finite and greater than zero",
                size.0, size.1, size.2
            ),
            Self::ObjectOutsideRoom { index, bbox } => write!(
                f,
                "objects[{index}] covers ({:.3}, {:.3})..({:.3}, {:.3}) m, which leaves the room footprint",
                bbox.0 .0, bbox.0 .1, bbox.1 .0, bbox.1 .1
            ),
            Self::ObjectAboveCeiling { index, z_top, room_h_m } => {
                write!(f, "objects[{index}] reaches {z_top:.3} m, above the {room_h_m} m ceiling")
            }
            Self::ObjectBelowFloor { index, z_bottom } => {
                write!(f, "objects[{index}] reaches {z_bottom:.3} m, below the floor")
            }
            Self::NoNodes => write!(f, "nodes is empty: a scene with no ESP32 has nothing to simulate"),
            Self::DuplicateNodeId { id } => {
                write!(f, "two nodes share id {id}: ids address links as (tx, rx) and must be unique")
            }
            Self::NodeNotFinite { index, id } => write!(f, "nodes[{index}] (id {id}) has a non-finite position"),
            Self::NodeOutsideRoom { id, pos } => write!(
                f,
                "node {id} at ({:.3}, {:.3}) m is outside the room footprint or within {NODE_WALL_CLEARANCE_M} m of a wall (saans-sim requires 0 < x < w and 0 < y < d, strictly)",
                pos.0, pos.1
            ),
            Self::NodeBadHeight { id, z_m, room_h_m } => {
                write!(f, "node {id} is at z = {z_m} m, outside the room's 0..{room_h_m} m height")
            }
            Self::PersonOutsideRoom { index, start } => {
                write!(f, "people[{index}].start ({:.3}, {:.3}) m is outside the room footprint", start.0, start.1)
            }
            Self::PhoneOutsideRoom { start } => {
                write!(f, "phone.start ({:.3}, {:.3}) m is outside the room footprint", start.0, start.1)
            }
        }
    }
}

impl std::error::Error for SceneError {}

fn finite2(p: Point) -> bool {
    p.0.is_finite() && p.1.is_finite()
}

fn finite3(p: (f64, f64, f64)) -> bool {
    p.0.is_finite() && p.1.is_finite() && p.2.is_finite()
}

/// Is the object's footprint inside the room outline?
///
/// Checked at the four bounding-box corners plus the box's own corners:
/// a convex footprint inside a convex room is settled by its corners, and
/// for an L-shaped room the corner test still catches every box that
/// pokes out of a leg, because a box can only leave through a corner
/// unless it is larger than the notch (which the wall test then catches).
///
/// Doors and windows are exempt and only have to have their CENTRE
/// inside: an opening lives in the wall, so half its depth is outside the
/// footprint by construction. Demanding that a door be wholly inside the
/// room would force every scene to fake it by pushing the door half a
/// thickness inwards, which is a lie about where the door is and would
/// make a RoomPlan import (which places openings exactly in the wall
/// plane) fail for being correct.
fn object_inside(poly: &[Point], o: &Object) -> bool {
    if matches!(o.kind, ObjectKind::Door | ObjectKind::Window) {
        return geom::point_in_polygon(poly, (o.pos.0, o.pos.1));
    }
    let (lo, hi) = o.footprint_bbox();
    let t = o.yaw_deg.to_radians();
    let (c, s) = (t.cos(), t.sin());
    let (hx, hy) = (o.size.0 / 2.0, o.size.1 / 2.0);
    let corners = [
        (o.pos.0 + hx * c - hy * s, o.pos.1 + hx * s + hy * c),
        (o.pos.0 - hx * c - hy * s, o.pos.1 - hx * s + hy * c),
        (o.pos.0 + hx * c + hy * s, o.pos.1 + hx * s - hy * c),
        (o.pos.0 - hx * c + hy * s, o.pos.1 - hx * s - hy * c),
        (lo.0, lo.1),
        (hi.0, hi.1),
    ];
    // The bbox corners are only required for an axis-aligned box; a
    // rotated box's bbox legitimately sticks out past its own corners.
    let n = if o.yaw_deg.abs() < EPS_M { corners.len() } else { 4 };
    corners[..n].iter().all(|&p| geom::point_in_polygon(poly, p))
}

impl Scene {
    /// Every problem with this scene, in reading order. Empty means valid.
    pub fn validate(&self) -> Vec<SceneError> {
        let mut errs = Vec::new();

        if self.schema_version != crate::scene::SCHEMA_VERSION {
            errs.push(SceneError::UnknownSchemaVersion {
                found: self.schema_version,
                supported: crate::scene::SCHEMA_VERSION,
            });
        }
        if self.name.trim().is_empty() {
            errs.push(SceneError::EmptyName);
        }

        let r = &self.room;
        for (field, v) in [("w_m", r.w_m), ("d_m", r.d_m), ("h_m", r.h_m)] {
            if !v.is_finite() || v <= 0.0 {
                errs.push(SceneError::BadRoomDimension { field, value: v });
            }
        }
        // Without a usable shell there is nothing to place anything in.
        if !errs.is_empty() && errs.iter().any(|e| matches!(e, SceneError::BadRoomDimension { .. })) {
            return errs;
        }

        if let Some(poly) = &r.polygon {
            if poly.len() < 3 {
                errs.push(SceneError::PolygonTooShort { vertices: poly.len() });
                return errs;
            }
            if let Some(i) = poly.iter().position(|&p| !finite2(p)) {
                errs.push(SceneError::PolygonNotFinite { index: i });
                return errs;
            }
            if let Some((a, b)) = geom::self_intersection(poly) {
                errs.push(SceneError::PolygonSelfIntersects { edge_a: a, edge_b: b });
                return errs;
            }
            if geom::area(poly) <= EPS_M {
                errs.push(SceneError::PolygonZeroArea);
                return errs;
            }
            let got = geom::bbox(poly);
            let want = ((0.0, 0.0), (r.w_m, r.d_m));
            let tol = 1e-6;
            let off = (got.0 .0 - want.0 .0).abs().max((got.0 .1 - want.0 .1).abs()).max((got.1 .0 - want.1 .0).abs()).max((got.1 .1 - want.1 .1).abs());
            if off > tol {
                errs.push(SceneError::PolygonBboxMismatch { got, want });
            }
        }

        let poly = self.footprint();

        for (i, w) in self.walls.iter().enumerate() {
            if !finite2(w.from) || !finite2(w.to) {
                errs.push(SceneError::WallNotFinite { index: i });
                continue;
            }
            if w.length_m() <= EPS_M {
                errs.push(SceneError::WallZeroLength { index: i });
            }
            if !w.thickness_m.is_finite() || w.thickness_m <= 0.0 || w.thickness_m > MAX_WALL_THICKNESS_M {
                errs.push(SceneError::WallBadThickness { index: i, value: w.thickness_m });
            }
            if let Some(h) = w.height_m {
                if !h.is_finite() || h <= 0.0 || h > r.h_m + EPS_M {
                    errs.push(SceneError::WallBadHeight { index: i, value: h, room_h_m: r.h_m });
                }
            }
            for p in [w.from, w.to] {
                if !geom::point_in_polygon(&poly, p) {
                    errs.push(SceneError::WallOutsideRoom { index: i, point: p });
                }
            }
        }

        for (i, o) in self.objects.iter().enumerate() {
            if !finite3(o.pos) || !finite3(o.size) || !o.yaw_deg.is_finite() {
                errs.push(SceneError::ObjectNotFinite { index: i });
                continue;
            }
            if o.size.0 <= 0.0 || o.size.1 <= 0.0 || o.size.2 <= 0.0 {
                errs.push(SceneError::ObjectBadSize { index: i, size: o.size });
                continue;
            }
            if !object_inside(&poly, o) {
                errs.push(SceneError::ObjectOutsideRoom { index: i, bbox: o.footprint_bbox() });
            }
            let (z0, z1) = o.z_extent();
            if z0 < -EPS_M {
                errs.push(SceneError::ObjectBelowFloor { index: i, z_bottom: z0 });
            }
            if z1 > r.h_m + EPS_M {
                errs.push(SceneError::ObjectAboveCeiling { index: i, z_top: z1, room_h_m: r.h_m });
            }
        }

        if self.nodes.is_empty() {
            errs.push(SceneError::NoNodes);
        }
        let mut seen: Vec<u8> = Vec::with_capacity(self.nodes.len());
        for (i, n) in self.nodes.iter().enumerate() {
            if seen.contains(&n.id) {
                errs.push(SceneError::DuplicateNodeId { id: n.id });
            } else {
                seen.push(n.id);
            }
            if !finite3(n.pos) {
                errs.push(SceneError::NodeNotFinite { index: i, id: n.id });
                continue;
            }
            if !geom::point_inside_with_margin(&poly, (n.pos.0, n.pos.1), NODE_WALL_CLEARANCE_M) {
                errs.push(SceneError::NodeOutsideRoom { id: n.id, pos: n.pos });
            }
            if n.pos.2 < 0.0 || n.pos.2 > r.h_m + EPS_M {
                errs.push(SceneError::NodeBadHeight { id: n.id, z_m: n.pos.2, room_h_m: r.h_m });
            }
        }

        for (i, p) in self.people.iter().enumerate() {
            if let Some(s) = p.start {
                if !finite2(s) || !geom::point_in_polygon(&poly, s) {
                    errs.push(SceneError::PersonOutsideRoom { index: i, start: s });
                }
            }
        }
        if let Some(ph) = &self.phone {
            if !finite2(ph.start) || !geom::point_in_polygon(&poly, ph.start) {
                errs.push(SceneError::PhoneOutsideRoom { start: ph.start });
            }
        }

        errs
    }
}
