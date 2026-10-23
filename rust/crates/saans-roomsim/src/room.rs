//! The scene adapter of spec/roomsim.md §3.1: a `saans_scene::Scene`, read
//! only, as the geometry the simulator works with.
//!
//! The room comes from the scene file and nothing else (ADR-008): no
//! number here is ever derived from CSI.

use crate::config::Config;
use crate::error::Error;
use crate::geometry::{pdist, v3, P2, V3};
use crate::materials::SurfaceMaterial;
use crate::numerics::{cos, sin, PI};
use saans_scene::validate::NODE_WALL_CLEARANCE_M;
use saans_scene::{ObjectKind, ObjectMaterial, Scene};

/// At most this many people (§2: so stream ids cannot collide).
pub const MAX_PEOPLE: usize = 100;

/// One wall: centreline `a -> b` (§3.1).
#[derive(Debug, Clone, PartialEq)]
pub struct WallGeom {
    pub a: P2,
    pub b: P2,
    /// Height, metres: `height_m` or the room's `H`.
    pub h: f64,
    /// Thickness, metres.
    pub t: f64,
    pub material: SurfaceMaterial,
    /// `sqrt((B-A).x^2 + (B-A).y^2)`.
    pub len: f64,
    /// `(B-A) / len`.
    pub e: P2,
    /// `(-e.y, e.x)` (z = 0).
    pub n: P2,
}

impl WallGeom {
    pub fn new(a: P2, b: P2, h: f64, t: f64, material: SurfaceMaterial) -> Self {
        let dx = b.0 - a.0;
        let dy = b.1 - a.1;
        let len = (dx * dx + dy * dy).sqrt();
        let e = (dx / len, dy / len);
        let n = (-e.1, e.0);
        Self {
            a,
            b,
            h,
            t,
            material,
            len,
            e,
            n,
        }
    }
}

/// One object box (§3.1): centre, size `(sl, sb, sh)`, yaw.
#[derive(Debug, Clone, PartialEq)]
pub struct ObjGeom {
    pub kind: ObjectKind,
    pub c: V3,
    pub sl: f64,
    pub sb: f64,
    pub sh: f64,
    /// `(yaw_deg * PI) / 180.0`.
    pub psi: f64,
    pub cs: f64,
    pub sn: f64,
    pub material: ObjectMaterial,
}

impl ObjGeom {
    /// Doors and windows are openings drawn in a wall, not furniture (§6.5).
    pub fn is_opening(&self) -> bool {
        matches!(self.kind, ObjectKind::Door | ObjectKind::Window)
    }

    /// Object-local to world (§3.1).
    pub fn to_world(&self, lx: f64, ly: f64, lz: f64) -> V3 {
        v3(
            self.c.x + (lx * self.cs - ly * self.sn),
            self.c.y + (lx * self.sn + ly * self.cs),
            self.c.z + lz,
        )
    }

    /// A local direction rotated into the world (§3.1 without the translation).
    pub fn rotate(&self, lx: f64, ly: f64, lz: f64) -> V3 {
        v3(lx * self.cs - ly * self.sn, lx * self.sn + ly * self.cs, lz)
    }

    /// World to object-local (§3.1).
    pub fn to_local(&self, p: V3) -> V3 {
        let dx = p.x - self.c.x;
        let dy = p.y - self.c.y;
        v3(
            dx * self.cs + dy * self.sn,
            -dx * self.sn + dy * self.cs,
            p.z - self.c.z,
        )
    }

    /// World to object-local on the floor plane.
    pub fn to_local_xy(&self, p: P2) -> P2 {
        let dx = p.0 - self.c.x;
        let dy = p.1 - self.c.y;
        (dx * self.cs + dy * self.sn, -dx * self.sn + dy * self.cs)
    }

    /// "Strictly inside" of §7.4: `|lx| < sl/2` and `|ly| < sb/2` and `|lz| < sh/2`.
    pub fn strictly_contains(&self, p: V3) -> bool {
        let l = self.to_local(p);
        l.x.abs() < self.sl / 2.0 && l.y.abs() < self.sb / 2.0 && l.z.abs() < self.sh / 2.0
    }

    /// The box footprint, not grown, contains `p` (closed): `|lx| <= sl/2`
    /// and `|ly| <= sb/2` (§8.2).
    pub fn footprint_contains(&self, p: P2) -> bool {
        let (lx, ly) = self.to_local_xy(p);
        lx.abs() <= self.sl / 2.0 && ly.abs() <= self.sb / 2.0
    }

    /// The top of the box, `cz + sh/2.0`.
    pub fn top(&self) -> f64 {
        self.c.z + self.sh / 2.0
    }
}

/// An ESP32 node.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NodeGeom {
    pub id: u8,
    pub p: V3,
}

/// Everything §3.1 reads from a scene.
#[derive(Debug, Clone, PartialEq)]
pub struct RoomModel {
    /// Footprint `F`: `room.polygon`, or the `w x d` rectangle.
    pub footprint: Vec<P2>,
    /// Room height `H`.
    pub h: f64,
    /// Bounding box `(x0, y0, x1, y1)` of `F`.
    pub bbox: (f64, f64, f64, f64),
    pub walls: Vec<WallGeom>,
    /// True when the scene lists no walls and the footprint's edges became
    /// walls of `implied_wall_material` (a SIM assumption for the manifest).
    pub walls_implied: bool,
    pub objects: Vec<ObjGeom>,
    /// Sorted by id ascending.
    pub nodes: Vec<NodeGeom>,
    /// Scene start of each person, after `n_people`.
    pub people: Vec<Option<P2>>,
    /// The phone's scene start, if the scene has a phone.
    pub phone: Option<P2>,
}

impl RoomModel {
    /// Read a scene (§3.1), rejecting anything `Scene::validate` rejects and
    /// also: two nodes at the same position, a node strictly inside an
    /// object's box (doors and windows included), more than 100 people; a
    /// node id outside 0..254, and a node within a wall's thickness
    /// (`pdist(node.xy, A, B) < t/2 + 0.01` with `node.z <= h`, implied
    /// walls included), are §3.2 errors naming `nodes`.
    pub fn from_scene(scene: &Scene, cfg: &Config) -> Result<Self, Error> {
        let errors = scene.validate();
        if !errors.is_empty() {
            return Err(Error::Scene(errors));
        }
        if scene.people.len() > MAX_PEOPLE {
            return Err(Error::SceneRule(format!(
                "{} people; at most {MAX_PEOPLE}",
                scene.people.len()
            )));
        }

        let footprint = scene.footprint();
        let h = scene.room.h_m;
        let bbox = crate::geometry::bbox(&footprint);

        let walls_implied = scene.walls.is_empty();
        let walls: Vec<WallGeom> = if walls_implied {
            crate::geometry::edges(&footprint)
                .map(|(a, b)| {
                    WallGeom::new(
                        a,
                        b,
                        h,
                        cfg.implied_wall_thickness_m,
                        cfg.implied_wall_material,
                    )
                })
                .collect()
        } else {
            scene
                .walls
                .iter()
                .map(|w| {
                    WallGeom::new(
                        w.from,
                        w.to,
                        w.height_or(h),
                        w.thickness_m,
                        w.material.into(),
                    )
                })
                .collect()
        };

        let objects: Vec<ObjGeom> = scene
            .objects
            .iter()
            .map(|o| {
                let psi = (o.yaw_deg * PI) / 180.0;
                ObjGeom {
                    kind: o.kind,
                    c: v3(o.pos.0, o.pos.1, o.pos.2),
                    sl: o.size.0,
                    sb: o.size.1,
                    sh: o.size.2,
                    psi,
                    cs: cos(psi),
                    sn: sin(psi),
                    material: o.material,
                }
            })
            .collect();

        let mut nodes: Vec<NodeGeom> = scene
            .nodes
            .iter()
            .map(|n| NodeGeom {
                id: n.id,
                p: v3(n.pos.0, n.pos.1, n.pos.2),
            })
            .collect();
        nodes.sort_by_key(|n| n.id);
        for n in &nodes {
            if n.id == 255 {
                return Err(Error::Config {
                    param: "nodes".into(),
                    reason: "node id 255 is not in 0..254".into(),
                });
            }
        }
        for (i, a) in nodes.iter().enumerate() {
            for b in &nodes[i + 1..] {
                if a.p == b.p {
                    return Err(Error::SceneRule(format!(
                        "nodes {} and {} are at the same position",
                        a.id, b.id
                    )));
                }
            }
            for (oi, o) in objects.iter().enumerate() {
                if o.strictly_contains(a.p) {
                    return Err(Error::SceneRule(format!(
                        "node {} at ({}, {}, {}) is strictly inside objects[{oi}]'s box",
                        a.id, a.p.x, a.p.y, a.p.z
                    )));
                }
            }
        }
        // A node within a wall's thickness (implied walls included): its
        // legs would run through the wall as free space (§3.1).
        for n in &nodes {
            for (k, w) in walls.iter().enumerate() {
                let d = pdist(n.p.xy(), w.a, w.b);
                let min = w.t / 2.0 + NODE_WALL_CLEARANCE_M;
                if d < min && n.p.z <= w.h {
                    let wall = if walls_implied {
                        format!("implied wall {k} (footprint edge {k})")
                    } else {
                        format!("walls[{k}]")
                    };
                    return Err(Error::Config {
                        param: "nodes".into(),
                        reason: format!(
                            "node {} at ({}, {}, {}) is within {wall}'s thickness: {d} m from its centreline, less than t/2 + {NODE_WALL_CLEARANCE_M} = {min} m",
                            n.id, n.p.x, n.p.y, n.p.z
                        ),
                    });
                }
            }
        }

        let mut people: Vec<Option<P2>> = scene.people.iter().map(|p| p.start).collect();
        if let Some(n) = cfg.n_people {
            people.resize(n, None);
        }

        Ok(Self {
            footprint,
            h,
            bbox,
            walls,
            walls_implied,
            objects,
            nodes,
            people,
            phone: scene.phone.as_ref().map(|p| p.start),
        })
    }

    /// Index of node `id` in [`RoomModel::nodes`].
    pub fn node_index(&self, id: u8) -> Option<usize> {
        self.nodes.iter().position(|n| n.id == id)
    }
}

/// Body or phone local `(bx, by, bz)` with heading `hd` to world (§3.1):
/// `x = px + (bx*ch - by*sh)`, `y = py + (bx*sh + by*ch)`, `z = bz`.
pub fn body_to_world(px: f64, py: f64, hd: f64, bx: f64, by: f64, bz: f64) -> V3 {
    let ch = cos(hd);
    let sh = sin(hd);
    v3(px + (bx * ch - by * sh), py + (bx * sh + by * ch), bz)
}
