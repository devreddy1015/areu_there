//! Static paths of spec/roomsim.md §6: line of sight, surface bounces to
//! order 2, object faces, clutter, the diffuse tail, and the static losses
//! of §6.9.

use crate::config::Config;
use crate::geometry::{dist3, dot3, inside, slab_meets, unit_dir, v3, V3};
use crate::materials::{eta, interface_g, object_itu, object_loss_db, object_rho, slab, Pol};
use crate::numerics::{
    cabs2, cexpj, cmul, cscale, cx, pow10, pymax, sign, Cx, Neumaier, C, FOUR_PI, FOUR_PI_1_5,
    MIN_RANGE_M, TWO_PI,
};
use crate::room::{ObjGeom, RoomModel};
use saans_scene::ObjectMaterial;
use saans_sim::rng::Rng;

/// Kind of a path (§6), with the spec's code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PathKind {
    LineOfSight,
    Bounce1,
    Bounce2,
    ObjectSpecular,
    ObjectPoint,
    Clutter,
    Tail,
}

impl PathKind {
    /// 0 LoS, 1 order-1 bounce, 2 order-2 bounce, 3 object specular,
    /// 4 object point, 5 clutter, 6 tail.
    pub fn code(self) -> u8 {
        match self {
            Self::LineOfSight => 0,
            Self::Bounce1 => 1,
            Self::Bounce2 => 2,
            Self::ObjectSpecular => 3,
            Self::ObjectPoint => 4,
            Self::Clutter => 5,
            Self::Tail => 6,
        }
    }
}

/// A surface of §6.1: walls in order, then the floor, then the ceiling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Surface {
    Wall(usize),
    Floor,
    Ceiling,
}

/// One static path of a link (§6).
#[derive(Debug, Clone, PartialEq)]
pub struct StaticPath {
    pub kind: PathKind,
    /// Surface indices (into [`World::surfaces`]) of the bounces: one for
    /// kind 1, `[i, j]` for kind 2, empty otherwise.
    pub surfaces: Vec<usize>,
    /// The object (index into the scene's objects) and face (0..5:
    /// `+x, -x, +y, -y, top`) of a kind-3 or kind-4 path.
    pub object: Option<(usize, usize)>,
    /// Index of a kind-5 path's point among the kept clutter points.
    pub clutter: Option<usize>,
    /// Index of a kind-6 term among the link's tail terms.
    pub tail: Option<usize>,
    /// Polyline `[Tp, P_1, ..., Rp]`; empty for a tail term (no legs).
    pub points: Vec<V3>,
    /// `L`: the Neumaier sum of the legs' 3-D lengths; `L_ref + x_n` for a tail term.
    pub length: f64,
    /// Coefficient before the §6.9 losses.
    pub coeff_free: Cx,
    /// Static coefficient `A_p` (after §6.9; for a tail term, its coefficient).
    pub coeff: Cx,
}

/// A clutter point (§6.7): a point scatterer shared by every link.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ClutterPoint {
    pub p: V3,
    pub sigma: f64,
}

/// Draw the clutter points from stream 1 (§6.7).
pub fn draw_clutter(cfg: &Config, room: &RoomModel) -> Vec<ClutterPoint> {
    let mut rng = Rng::stream(cfg.seed, 1);
    let (x0, y0, x1, y1) = room.bbox;
    let (lo, hi) = cfg.clutter_rcs_m2;
    let mut out = Vec::new();
    for _ in 0..cfg.clutter_count {
        let mut found = None;
        for _ in 0..1000 {
            let x = rng.range(x0, x1);
            let y = rng.range(y0, y1);
            if crate::geometry::inside_with_margin(&room.footprint, (x, y), cfg.clutter_margin_m) {
                found = Some((x, y));
                break;
            }
        }
        if let Some((x, y)) = found {
            let z = rng.range(0.0, room.h);
            let sigma = rng.range(lo.ln(), hi.ln()).exp();
            out.push(ClutterPoint {
                p: v3(x, y, z),
                sigma,
            });
        }
    }
    out
}

/// The tail draws of an unordered node pair (§6.8): `(x_n, phi_n)` from
/// stream `100000 + pair_index`.
pub fn draw_tail(cfg: &Config, pair_index: usize) -> Vec<(f64, f64)> {
    let mut rng = Rng::stream(cfg.seed, 100_000 + pair_index as u64);
    (0..cfg.tail_count)
        .map(|_| {
            let x = -((C * cfg.tail_decay_s) * (1.0 - rng.uniform()).ln());
            let phi = rng.range(0.0, TWO_PI);
            (x, phi)
        })
        .collect()
}

/// The room, its materials and the static-path switches: everything the
/// path and loss formulas need.
#[derive(Debug, Clone)]
pub struct World {
    pub room: RoomModel,
    pub carrier_hz: f64,
    /// `LAMBDA = C / carrier_hz`.
    pub lambda: f64,
    /// `S`: walls in order, then (if `floor_ceiling`) floor, then ceiling.
    pub surfaces: Vec<Surface>,
    wall_eta: Vec<Cx>,
    floor_eta: Cx,
    ceiling_eta: Cx,
    slab_thickness_m: f64,
    obj_eta: Vec<Option<Cx>>,
    transmission: bool,
    object_blockage: bool,
}

impl World {
    pub fn new(room: RoomModel, cfg: &Config) -> Self {
        let carrier_hz = cfg.carrier_hz();
        let lambda = C / carrier_hz;
        let mut surfaces: Vec<Surface> = (0..room.walls.len()).map(Surface::Wall).collect();
        if cfg.floor_ceiling {
            surfaces.push(Surface::Floor);
            surfaces.push(Surface::Ceiling);
        }
        let wall_eta = room
            .walls
            .iter()
            .map(|w| eta(w.material.itu(), carrier_hz))
            .collect();
        let obj_eta = room
            .objects
            .iter()
            .map(|o| object_itu(o.material).map(|m| eta(m, carrier_hz)))
            .collect();
        Self {
            carrier_hz,
            lambda,
            surfaces,
            wall_eta,
            floor_eta: eta(cfg.floor_material.itu(), carrier_hz),
            ceiling_eta: eta(cfg.ceiling_material.itu(), carrier_hz),
            slab_thickness_m: cfg.slab_thickness_m,
            obj_eta,
            transmission: cfg.transmission,
            object_blockage: cfg.object_blockage,
            room,
        }
    }

    /// The face offset `o = f * (t / 2.0)` of wall `k`'s face `f` (§6.1):
    /// the signed distance of that face from the centreline along `n`.
    pub fn face_offset(&self, k: usize, f: i32) -> f64 {
        f64::from(f) * (self.room.walls[k].t / 2.0)
    }

    /// `dc(P) = (P.x-A.x)*n.x + (P.y-A.y)*n.y` of wall `k`: the signed
    /// distance of `P` from the centreline (§6.1).
    pub fn wall_dc(&self, k: usize, p: V3) -> f64 {
        let w = &self.room.walls[k];
        (p.x - w.a.0) * w.n.0 + (p.y - w.a.1) * w.n.1
    }

    /// `side(P)` of §6.1: which side of the surface `P` is on (`0` on it).
    pub fn side(&self, s: Surface, p: V3) -> i32 {
        match s {
            Surface::Wall(k) => sign(self.wall_dc(k, p)),
            Surface::Floor => sign(p.z),
            Surface::Ceiling => sign(self.room.h - p.z),
        }
    }

    /// `front_f(P)` of §6.1: `P` is in front of face `f` of the surface
    /// (for a wall, `sign(dc(P) - o) == f`; for the floor and ceiling,
    /// `side(P) == f`).
    pub fn front(&self, s: Surface, f: i32, p: V3) -> bool {
        match s {
            Surface::Wall(k) => sign(self.wall_dc(k, p) - self.face_offset(k, f)) == f,
            Surface::Floor | Surface::Ceiling => self.side(s, p) == f,
        }
    }

    /// `image_f(P)` of §6.1: `P` mirrored in face `f` of the surface (a
    /// wall's face `t/2` from its centreline; the floor at `z = 0`, the
    /// ceiling at `z = H`, whatever `f`).
    pub fn image(&self, s: Surface, f: i32, p: V3) -> V3 {
        match s {
            Surface::Wall(k) => {
                let w = &self.room.walls[k];
                let dd = 2.0 * (self.wall_dc(k, p) - self.face_offset(k, f));
                v3(p.x - dd * w.n.0, p.y - dd * w.n.1, p.z)
            }
            Surface::Floor => v3(p.x, p.y, -p.z),
            Surface::Ceiling => v3(p.x, p.y, 2.0 * self.room.h - p.z),
        }
    }

    /// Unit normal of a surface.
    pub fn normal(&self, s: Surface) -> V3 {
        match s {
            Surface::Wall(k) => {
                let w = &self.room.walls[k];
                v3(w.n.0, w.n.1, 0.0)
            }
            Surface::Floor | Surface::Ceiling => v3(0.0, 0.0, 1.0),
        }
    }

    /// `hit_f(P, Q)` of §6.1: where segment `P -> Q` meets face `f` of the
    /// surface (a bounce point).
    pub fn hit(&self, s: Surface, f: i32, p: V3, q: V3) -> Option<V3> {
        match s {
            Surface::Wall(k) => self.hit_wall(k, self.face_offset(k, f), p, q),
            Surface::Floor => self.hit_plane(0.0, p, q),
            Surface::Ceiling => self.hit_plane(self.room.h, p, q),
        }
    }

    /// `hit(P, Q)` of §6.1 without a face: where segment `P -> Q` crosses
    /// wall `k`'s centreline (`o = 0`), the test of §6.9 transmission.
    pub fn hit_centreline(&self, k: usize, p: V3, q: V3) -> Option<V3> {
        self.hit_wall(k, 0.0, p, q)
    }

    /// The §6.1 wall test against the centreline of wall `k` moved by `o`
    /// along `n`: `Af = (A.x + o*n.x, A.y + o*n.y)`, direction `w = B - A`.
    fn hit_wall(&self, k: usize, o: f64, p: V3, q: V3) -> Option<V3> {
        let w = &self.room.walls[k];
        let af = (w.a.0 + o * w.n.0, w.a.1 + o * w.n.1);
        let r = (q.x - p.x, q.y - p.y);
        let ww = (w.b.0 - w.a.0, w.b.1 - w.a.1);
        let den = r.0 * ww.1 - r.1 * ww.0;
        if den == 0.0 {
            return None;
        }
        let ap = (af.0 - p.x, af.1 - p.y);
        let s = (ap.0 * ww.1 - ap.1 * ww.0) / den;
        let v = (ap.0 * r.1 - ap.1 * r.0) / den;
        if !(0.0 < s && s < 1.0 && (0.0..=1.0).contains(&v)) {
            return None;
        }
        let x = v3(p.x + s * r.0, p.y + s * r.1, p.z + s * (q.z - p.z));
        if !(0.0..=w.h).contains(&x.z) {
            return None;
        }
        Some(x)
    }

    fn hit_plane(&self, z: f64, p: V3, q: V3) -> Option<V3> {
        let dz = q.z - p.z;
        if dz == 0.0 {
            return None;
        }
        let s = (z - p.z) / dz;
        if !(0.0 < s && s < 1.0) {
            return None;
        }
        let x = v3(p.x + s * (q.x - p.x), p.y + s * (q.y - p.y), z);
        if !inside(&self.room.footprint, (x.x, x.y)) {
            return None;
        }
        Some(x)
    }

    /// `R(cos_t)` of a surface (§6.1): the slab `R`, referenced to the
    /// incident face (§5.1) — TE with the wall's material and thickness
    /// for a wall, TM with `floor_material` / `ceiling_material` and
    /// `slab_thickness_m` for the floor and ceiling.
    pub fn reflection(&self, s: Surface, cos_t: f64) -> Cx {
        match s {
            Surface::Wall(k) => {
                let w = &self.room.walls[k];
                slab(self.wall_eta[k], cos_t, w.t, self.lambda, Pol::Te).r
            }
            Surface::Floor => {
                slab(
                    self.floor_eta,
                    cos_t,
                    self.slab_thickness_m,
                    self.lambda,
                    Pol::Tm,
                )
                .r
            }
            Surface::Ceiling => {
                slab(
                    self.ceiling_eta,
                    cos_t,
                    self.slab_thickness_m,
                    self.lambda,
                    Pol::Tm,
                )
                .r
            }
        }
    }

    /// `T_k(cos_t)` of wall `k` (§6.9).
    pub fn wall_transmission(&self, k: usize, cos_t: f64) -> Cx {
        let w = &self.room.walls[k];
        slab(self.wall_eta[k], cos_t, w.t, self.lambda, Pol::Te).t
    }

    /// Does the leg `p -> q` meet object `o`'s box (§7.4, 3-D)?
    pub fn leg_meets_object(o: &ObjGeom, p: V3, q: V3) -> bool {
        let a = o.to_local(p);
        let b = o.to_local(q);
        slab_meets(
            &[a.x, a.y, a.z],
            &[b.x, b.y, b.z],
            &[o.sl / 2.0, o.sb / 2.0, o.sh / 2.0],
        )
    }

    /// §6.9 losses of one leg `p -> q`, applied to `a`: wall transmission
    /// (walls in order, skipping the surfaces `skip_surfaces` this leg
    /// bounces on), then object blockage (objects in order, skipping doors,
    /// windows and `skip_object`).
    pub fn leg_losses(
        &self,
        a: Cx,
        p: V3,
        q: V3,
        skip_surfaces: &[usize],
        skip_object: Option<usize>,
    ) -> Cx {
        let mut a = a;
        if self.transmission {
            for (k, w) in self.room.walls.iter().enumerate() {
                if skip_surfaces.contains(&k) {
                    continue;
                }
                if self.hit_centreline(k, p, q).is_some() {
                    let cos_t = dot3(unit_dir(p, q), v3(w.n.0, w.n.1, 0.0)).abs();
                    a = cmul(a, self.wall_transmission(k, cos_t));
                }
            }
        }
        if self.object_blockage {
            for (oi, o) in self.room.objects.iter().enumerate() {
                if o.is_opening() || skip_object == Some(oi) {
                    continue;
                }
                if Self::leg_meets_object(o, p, q) {
                    a = cscale(a, pow10(-object_loss_db(o.material) / 20.0));
                }
            }
        }
        a
    }

    /// Point-scatter coefficient of §6.6 for a point `p` with RCS `sigma`,
    /// and the path length `L` (true distances; the clamp affects `a` only).
    pub fn point_scatter(&self, tp: V3, p: V3, rp: V3, sigma: f64) -> (Cx, f64) {
        let l1 = dist3(tp, p);
        let l2 = dist3(p, rp);
        let d1 = pymax(l1, MIN_RANGE_M);
        let d2 = pymax(l2, MIN_RANGE_M);
        let a = (self.lambda * sigma.sqrt()) / ((FOUR_PI_1_5 * d1) * d2);
        (cx(a, 0.0), Neumaier::sum([l1, l2]))
    }

    /// Free-space amplitude `LAMBDA / (FOUR_PI * L)`.
    fn spread(&self, l: f64) -> f64 {
        self.lambda / (FOUR_PI * l)
    }

    /// Every static path of the link `tp -> rp`, in the §6 order, with
    /// §6.9 losses applied. `tail` holds the pair's `(x_n, phi_n)` draws, or
    /// `None` when `tail` is off.
    pub fn static_paths(
        &self,
        cfg: &Config,
        tp: V3,
        rp: V3,
        clutter: &[ClutterPoint],
        tail: Option<&[(f64, f64)]>,
    ) -> Vec<StaticPath> {
        // Each path with the surfaces to skip per leg and the object to skip.
        let mut out: Vec<(StaticPath, Vec<Vec<usize>>, Option<usize>)> = Vec::new();
        let path = |kind, points: Vec<V3>, length, coeff: Cx| StaticPath {
            kind,
            surfaces: Vec::new(),
            object: None,
            clutter: None,
            tail: None,
            points,
            length,
            coeff_free: coeff,
            coeff,
        };

        // §6.2 line of sight.
        let l = Neumaier::sum([dist3(tp, rp)]);
        out.push((
            path(
                PathKind::LineOfSight,
                vec![tp, rp],
                l,
                cx(self.spread(l), 0.0),
            ),
            vec![vec![]],
            None,
        ));

        // §6.3 order 1.
        if cfg.reflection_order >= 1 {
            for (i, &s) in self.surfaces.iter().enumerate() {
                // The face on the transmitter's side; both nodes in front of it.
                let f = self.side(s, tp);
                if !(f != 0 && self.front(s, f, tp) && self.front(s, f, rp)) {
                    continue;
                }
                let Some(x) = self.hit(s, f, self.image(s, f, tp), rp) else {
                    continue;
                };
                let cos_t = dot3(unit_dir(tp, x), self.normal(s)).abs();
                let l = Neumaier::sum([dist3(tp, x), dist3(x, rp)]);
                let coeff = cscale(self.reflection(s, cos_t), self.spread(l));
                let mut p = path(PathKind::Bounce1, vec![tp, x, rp], l, coeff);
                p.surfaces = vec![i];
                out.push((p, vec![vec![i], vec![i]], None));
            }
        }

        // §6.4 order 2.
        if cfg.reflection_order >= 2 {
            for (i, &si) in self.surfaces.iter().enumerate() {
                for (j, &sj) in self.surfaces.iter().enumerate() {
                    if i == j {
                        continue;
                    }
                    // Face fi of i on the transmitter's side, face fj of j
                    // on the receiver's.
                    let fi = self.side(si, tp);
                    let fj = self.side(sj, rp);
                    if fi == 0 || fj == 0 {
                        continue;
                    }
                    let t1 = self.image(si, fi, tp);
                    let t2 = self.image(sj, fj, t1);
                    let Some(xj) = self.hit(sj, fj, t2, rp) else {
                        continue;
                    };
                    let Some(xi) = self.hit(si, fi, t1, xj) else {
                        continue;
                    };
                    if !(self.front(si, fi, tp)
                        && self.front(si, fi, xj)
                        && self.front(sj, fj, xi)
                        && self.front(sj, fj, rp))
                    {
                        continue;
                    }
                    let cos1 = dot3(unit_dir(tp, xi), self.normal(si)).abs();
                    let cos2 = dot3(unit_dir(xi, xj), self.normal(sj)).abs();
                    let l = Neumaier::sum([dist3(tp, xi), dist3(xi, xj), dist3(xj, rp)]);
                    let coeff = cscale(
                        cmul(self.reflection(si, cos1), self.reflection(sj, cos2)),
                        self.spread(l),
                    );
                    let mut p = path(PathKind::Bounce2, vec![tp, xi, xj, rp], l, coeff);
                    p.surfaces = vec![i, j];
                    out.push((p, vec![vec![i], vec![i, j], vec![j]], None));
                }
            }
        }

        // §6.5 object faces.
        for (oi, o) in self.room.objects.iter().enumerate() {
            if o.is_opening() {
                continue;
            }
            for f in 0..5 {
                let face = Face::of(o, f);
                let in_front_t = dot3(sub(tp, face.centre), face.normal) > 0.0;
                let in_front_r = dot3(sub(rp, face.centre), face.normal) > 0.0;
                if !(in_front_t && in_front_r) {
                    continue;
                }
                let specular = if cfg.object_specular && o.material != ObjectMaterial::Fabric {
                    self.face_specular(o, oi, &face, tp, rp)
                } else {
                    None
                };
                if let Some(mut p) = specular {
                    p.object = Some((oi, f));
                    out.push((p, vec![vec![], vec![]], Some(oi)));
                } else if cfg.object_scatter {
                    let sigma =
                        cfg.object_scatter_scale * object_rho(o.material) * (face.e1 * face.e2);
                    let (coeff, l) = self.point_scatter(tp, face.centre, rp, sigma);
                    let mut p = path(PathKind::ObjectPoint, vec![tp, face.centre, rp], l, coeff);
                    p.object = Some((oi, f));
                    out.push((p, vec![vec![], vec![]], Some(oi)));
                }
            }
        }

        // §6.7 clutter.
        for (ci, c) in clutter.iter().enumerate() {
            let (coeff, l) = self.point_scatter(tp, c.p, rp, c.sigma);
            let mut p = path(PathKind::Clutter, vec![tp, c.p, rp], l, coeff);
            p.clutter = Some(ci);
            out.push((p, vec![vec![], vec![]], None));
        }

        // §6.9 static losses, leg by leg.
        let mut paths: Vec<StaticPath> = out
            .into_iter()
            .map(|(mut p, skips, skip_obj)| {
                let mut a = p.coeff_free;
                for (leg, skip) in p.points.windows(2).zip(&skips) {
                    a = self.leg_losses(a, leg[0], leg[1], skip, skip_obj);
                }
                p.coeff = a;
                p
            })
            .collect();

        // §6.8 diffuse tail.
        if let Some(draws) = tail {
            let specular = paths.iter().filter(|p| p.kind.code() <= 3);
            let mut l_ref = f64::NEG_INFINITY;
            let mut p_spec = Neumaier::new();
            for p in specular {
                l_ref = if l_ref == f64::NEG_INFINITY {
                    p.length
                } else {
                    pymax(l_ref, p.length)
                };
                p_spec.add(cabs2(p.coeff));
            }
            let p_spec = p_spec.value();
            // Equal amplitudes: the exponential delays alone shape the
            // power-delay profile, exp(-tau / tail_decay_s).
            let a_tail =
                ((pow10(cfg.tail_power_db / 10.0) * p_spec) / cfg.tail_count as f64).sqrt();
            for (n, &(x, phi)) in draws.iter().enumerate() {
                let coeff = cscale(cexpj(phi), a_tail);
                let mut p = path(PathKind::Tail, Vec::new(), l_ref + x, coeff);
                p.tail = Some(n);
                paths.push(p);
            }
        }
        paths
    }

    /// §6.5 step 1: the specular path off a face, if the face is specular.
    fn face_specular(
        &self,
        o: &ObjGeom,
        oi: usize,
        face: &Face,
        tp: V3,
        rp: V3,
    ) -> Option<StaticPath> {
        let n = face.normal;
        let dd = dot3(sub(tp, face.centre), n);
        let tq = v3(
            tp.x - 2.0 * dd * n.x,
            tp.y - 2.0 * dd * n.y,
            tp.z - 2.0 * dd * n.z,
        );
        let den = dot3(sub(rp, tq), n);
        if den == 0.0 {
            return None;
        }
        let s = dot3(sub(face.centre, tq), n) / den;
        if !(0.0 < s && s < 1.0) {
            return None;
        }
        let x = v3(
            tq.x + s * (rp.x - tq.x),
            tq.y + s * (rp.y - tq.y),
            tq.z + s * (rp.z - tq.z),
        );
        let l = o.to_local(x);
        let on_face = match face.index {
            0 | 1 => l.y.abs() <= o.sb / 2.0 && l.z.abs() <= o.sh / 2.0,
            2 | 3 => l.x.abs() <= o.sl / 2.0 && l.z.abs() <= o.sh / 2.0,
            _ => l.x.abs() <= o.sl / 2.0 && l.y.abs() <= o.sb / 2.0,
        };
        if !on_face {
            return None;
        }
        let d1 = dist3(tp, x);
        let d2 = dist3(x, rp);
        let r_f = (self.lambda * d1 * d2 / (d1 + d2)).sqrt();
        if !(face.e1 >= 2.0 * r_f && face.e2 >= 2.0 * r_f) {
            return None;
        }
        let eta = self.obj_eta[oi]?;
        let cos_t = dot3(unit_dir(tp, x), n).abs();
        let l = Neumaier::sum([d1, d2]);
        let coeff = cscale(interface_g(eta, cos_t, face.pol), self.spread(l));
        Some(StaticPath {
            kind: PathKind::ObjectSpecular,
            surfaces: Vec::new(),
            object: None,
            clutter: None,
            tail: None,
            points: vec![tp, x, rp],
            length: l,
            coeff_free: coeff,
            coeff,
        })
    }
}

fn sub(b: V3, a: V3) -> V3 {
    crate::geometry::sub3(b, a)
}

/// One of an object's five faces (§6.5), in world coordinates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Face {
    /// 0..5: `+x, -x, +y, -y, top` (object-local).
    pub index: usize,
    /// Face centre `P_f`.
    pub centre: V3,
    /// Outward unit normal `n_f`.
    pub normal: V3,
    /// In-plane extents `(e1, e2)`.
    pub e1: f64,
    pub e2: f64,
    /// TE for vertical faces, TM for the top.
    pub pol: Pol,
}

impl Face {
    /// Face `f` of object `o`.
    pub fn of(o: &ObjGeom, f: usize) -> Self {
        let (hl, hb, hh) = (o.sl / 2.0, o.sb / 2.0, o.sh / 2.0);
        let (centre, normal, e1, e2) = match f {
            0 => ((hl, 0.0, 0.0), (1.0, 0.0, 0.0), o.sb, o.sh),
            1 => ((-hl, 0.0, 0.0), (-1.0, 0.0, 0.0), o.sb, o.sh),
            2 => ((0.0, hb, 0.0), (0.0, 1.0, 0.0), o.sl, o.sh),
            3 => ((0.0, -hb, 0.0), (0.0, -1.0, 0.0), o.sl, o.sh),
            4 => ((0.0, 0.0, hh), (0.0, 0.0, 1.0), o.sl, o.sb),
            _ => panic!("an object has five faces, not {}", f + 1),
        };
        Self {
            index: f,
            centre: o.to_world(centre.0, centre.1, centre.2),
            normal: o.rotate(normal.0, normal.1, normal.2),
            e1,
            e2,
            pol: if f == 4 { Pol::Tm } else { Pol::Te },
        }
    }
}
