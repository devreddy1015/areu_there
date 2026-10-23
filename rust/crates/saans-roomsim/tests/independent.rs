//! spec/roomsim.md §15: checks against independent ground truth (closed
//! forms and physical laws), not against the Python reference. The golden
//! parity test is a separate stage.

use saans_roomsim::geometry::{dist3, dot3, sub3, unit_dir, v3, V3};
use saans_roomsim::impair::{buffer_active, buffer_k};
use saans_roomsim::numerics::{bin_index, cabs2, pow10, Cx, C};
use saans_roomsim::paths::Surface;
use saans_roomsim::people::shadow_factor;
use saans_roomsim::{Config, Mode, PathKind, Simulator};
use saans_scene::Scene;
use std::path::PathBuf;

fn scenes_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../../spec/scenes")
}

fn load(name: &str) -> Scene {
    Scene::load(scenes_dir().join(name)).unwrap()
}

fn scene(json: &str) -> Scene {
    Scene::from_json_str("test", json).unwrap()
}

/// A rectangular room, no walls listed (implied), with the given objects,
/// nodes and people JSON fragments.
fn rect(w: f64, d: f64, h: f64, walls: &str, objects: &str, nodes: &str, people: &str) -> Scene {
    scene(&format!(
        r#"{{"schema_version": 1, "name": "t", "provenance": "hand_written",
            "room": {{"w_m": {w}, "d_m": {d}, "h_m": {h}}},
            "walls": [{walls}], "objects": [{objects}], "nodes": [{nodes}], "people": [{people}]}}"#
    ))
}

fn power(h: &[Cx]) -> f64 {
    h.iter().map(|&c| cabs2(c)).sum()
}

/// §15: first-order wall images in a rectangle have closed-form lengths, and
/// every admitted bounce (orders 1 and 2) obeys the specular law. A wall
/// bounce is on the wall's face, `t/2` from its centreline on the side the
/// wave arrives from (§5.1, §6.1), so the images are mirror images in the
/// faces: the implied walls' (0.1 m drywall), and listed walls of four
/// different thicknesses, one listed backwards so that its normal points
/// out of the room and the nodes are on its `f = -1` side.
#[test]
fn rectangle_images_have_closed_form_lengths_and_obey_the_specular_law() {
    let (w, d, h) = (5.0, 4.0, 2.7);
    let nodes = r#"{"id": 0, "pos": [1.0, 1.0, 1.2]}, {"id": 1, "pos": [4.0, 2.5, 1.5]}"#;
    let implied = rect(w, d, h, "", "", nodes, "");
    let sim = Simulator::new(&implied, Config::default()).unwrap();
    assert!(sim.room().walls_implied);
    let hf = sim.config().implied_wall_thickness_m / 2.0;
    check_rectangle_images(&sim, [hf; 4], (w, d, h));

    // Wall 2 runs from (0, d) to (w, d): against the footprint's order.
    let listed = rect(
        w,
        d,
        h,
        r#"{"from": [0.0, 0.0], "to": [5.0, 0.0], "material": "brick", "thickness_m": 0.2},
           {"from": [5.0, 0.0], "to": [5.0, 4.0], "material": "concrete", "thickness_m": 0.3},
           {"from": [0.0, 4.0], "to": [5.0, 4.0], "material": "glass", "thickness_m": 0.1},
           {"from": [0.0, 4.0], "to": [0.0, 0.0], "material": "wood", "thickness_m": 0.25}"#,
        "",
        nodes,
        "",
    );
    let sim = Simulator::new(&listed, Config::default()).unwrap();
    assert!(!sim.room().walls_implied);
    assert_eq!(sim.world().side(Surface::Wall(2), v3(1.0, 1.0, 1.2)), -1);
    check_rectangle_images(&sim, [0.1, 0.15, 0.05, 0.125], (w, d, h));
}

/// The rectangle check for nodes (1, 1, 1.2) and (4, 2.5, 1.5); `half[k]`
/// is wall `k`'s half thickness (walls y = 0, x = w, y = d, x = 0).
fn check_rectangle_images(sim: &Simulator, half: [f64; 4], (w, d, h): (f64, f64, f64)) {
    let (tp, rp) = (v3(1.0, 1.0, 1.2), v3(4.0, 2.5, 1.5));
    let images = [
        v3(1.0, 2.0 * half[0] - 1.0, 1.2),       // wall 0: face y = t/2
        v3(2.0 * (w - half[1]) - 1.0, 1.0, 1.2), // wall 1: face x = w - t/2
        v3(1.0, 2.0 * (d - half[2]) - 1.0, 1.2), // wall 2: face y = d - t/2
        v3(2.0 * half[3] - 1.0, 1.0, 1.2),       // wall 3: face x = t/2
        v3(1.0, 1.0, -1.2),                      // floor
        v3(1.0, 1.0, 2.0 * h - 1.2),             // ceiling
    ];
    let li = sim.link_index(0, 1).unwrap();
    let paths = sim.paths(li);
    let first: Vec<_> = paths
        .iter()
        .filter(|p| p.kind == PathKind::Bounce1)
        .collect();
    assert_eq!(
        first.len(),
        6,
        "every surface gives a first-order path in a convex room"
    );
    for p in &first {
        let i = p.surfaces[0];
        let want = dist3(images[i], rp);
        assert!(
            (p.length - want).abs() <= 1e-12 * want,
            "surface {i}: {} vs {want}",
            p.length
        );
    }
    let los = &paths[0];
    assert_eq!(los.kind, PathKind::LineOfSight);
    assert!((los.length - dist3(tp, rp)).abs() < 1e-15);

    let world = sim.world();
    let mut checked = 0;
    for p in paths
        .iter()
        .filter(|p| matches!(p.kind, PathKind::Bounce1 | PathKind::Bounce2))
    {
        for (b, &si) in p.surfaces.iter().enumerate() {
            let surf = world.surfaces[si];
            let n = world.normal(surf);
            let x = p.points[b + 1];
            let u_in = unit_dir(p.points[b], x);
            let u_out = unit_dir(x, p.points[b + 2]);
            let dn = 2.0 * dot3(u_in, n);
            let mirrored = v3(u_in.x - dn * n.x, u_in.y - dn * n.y, u_in.z - dn * n.z);
            let err = dist3(mirrored, u_out);
            assert!(
                err < 1e-12,
                "{:?} bounce {b} on {surf:?}: specular error {err}",
                p.surfaces
            );
            // The bounce point lies on the surface: for a wall, on the face
            // t/2 from the centreline on the incoming wave's side, which
            // the wave leaves from too.
            let on = match surf {
                Surface::Wall(k) => {
                    let wall = &sim.room().walls[k];
                    let dc = |q: V3| (q.x - wall.a.0) * wall.n.0 + (q.y - wall.a.1) * wall.n.1;
                    let (before, after) = (dc(p.points[b]), dc(p.points[b + 2]));
                    assert!(
                        before.abs() > wall.t / 2.0 && after.abs() > wall.t / 2.0,
                        "both ends in front of the face"
                    );
                    assert_eq!(before.signum(), after.signum(), "one side of the wall");
                    (dc(x) - before.signum() * (wall.t / 2.0)).abs()
                }
                Surface::Floor => x.z.abs(),
                Surface::Ceiling => (x.z - h).abs(),
            };
            assert!(on < 1e-12, "bounce point off its face by {on}");
            checked += 1;
        }
    }
    assert!(checked > 6 + 10, "only {checked} bounces checked");
}

/// ITU-R P.526 knife edge `J1(nu)` (§8.5), written here from the spec.
fn knife_edge_db(nu: f64) -> f64 {
    if nu <= -0.78 {
        return 0.0;
    }
    let w = nu - 0.1;
    6.9 + 20.0 * (((w * w + 1.0).sqrt() + nu) - 0.1).log10()
}

/// The xy distance from `c` to segment `a`-`b`, the parameter of the
/// closest point, and the leg's height there.
fn closest_in_plan(a: V3, b: V3, c: (f64, f64)) -> (f64, f64, f64) {
    let (rx, ry) = (b.x - a.x, b.y - a.y);
    let l2 = rx * rx + ry * ry;
    let u = if l2 == 0.0 {
        0.0
    } else {
        (((c.0 - a.x) * rx + (c.1 - a.y) * ry) / l2).clamp(0.0, 1.0)
    };
    let (px, py) = (a.x + u * rx, a.y + u * ry);
    let dist = ((px - c.0) * (px - c.0) + (py - c.1) * (py - c.1)).sqrt();
    (dist, u, a.z + u * (b.z - a.z))
}

/// §15: a person on a link's LoS lowers the LoS by exactly `10^(-J/20)`,
/// `J` the three-edge loss of §8.5 computed here from the geometry; a path
/// whose legs all pass clear of them in plan, or above their top, is
/// unchanged; and paths whose legs pass through the cylinder, such as the
/// floor bounce at their feet, are shadowed too.
///
/// "Clear" has a margin: a leg passing just outside the cylinder is still
/// in the near edge's diffraction zone (`J1(nu) > 0` for `-0.78 < nu < 0`,
/// up to about 0.2 m beyond `body_radius_m` here) and loses a few dB, so
/// the unchanged paths checked are those whose legs miss the cylinder by
/// more than 0.5 m in plan, or pass more than 0.5 m above its top, or
/// end short of it (§8.5 step 2).
#[test]
fn a_person_on_the_line_of_sight_lowers_it_by_exactly_j() {
    let s = rect(
        6.0,
        4.0,
        2.7,
        "",
        "",
        r#"{"id": 0, "pos": [1.0, 2.0, 1.0]}, {"id": 1, "pos": [5.0, 2.0, 1.0]}"#,
        r#"{"start": [3.0, 2.0]}"#,
    );
    let cfg = Config {
        people_mode: [(0, Mode::Stationary)].into(),
        ..Config::default()
    };
    let sim = Simulator::new(&s, cfg.clone()).unwrap();
    let person = &sim.people()[0];
    let height = person.body.height;
    let pose = sim.stationary_pose(person);
    let at = sim.people_at(0.0, &[pose]);
    let (c, radius, top) = ((3.0, 2.0), cfg.body_radius_m, height);
    assert_eq!(at.cylinders[0].top, top, "a standing person's top");

    let lambda = C / cfg.carrier_hz();
    let cap = cfg.shadow_cap_db;
    let li = sim.link_index(0, 1).unwrap();
    let paths = sim.paths(li);
    // The LoS passes through the axis half-way: u* = 0.5, d1 = d2 = 2,
    // d_perp = 0, z* = 1.0.
    let (d1, d2, d_perp, z) = (2.0, 2.0, 0.0, 1.0);
    let k = (2.0 * (d1 + d2) / ((lambda * d1) * d2)).sqrt();
    let (nu_1, nu_2, nu_t) = ((radius - d_perp) * k, (radius + d_perp) * k, (top - z) * k);
    let e = |nu: f64| 10f64.powf(-knife_edge_db(nu) / 10.0);
    let j = (-10.0 * ((e(nu_1) + e(nu_2)) + e(nu_t)).log10()).clamp(0.0, cap);
    assert!(j > 7.0 && j < 10.0, "J = {j}: about 8.7 dB on a 4 m link");
    let factor =
        |p: &saans_roomsim::StaticPath| shadow_factor(&p.points, &at.cylinders, None, lambda, cap);
    assert_eq!(paths[0].kind, PathKind::LineOfSight);
    assert_eq!(
        factor(&paths[0]),
        10f64.powf(-j / 20.0),
        "the LoS is lowered by exactly 10^(-J/20)"
    );

    let (mut unchanged, mut through) = (0, 0);
    for p in paths.iter().filter(|p| p.kind != PathKind::Tail) {
        let clear = p.points.windows(2).all(|leg| {
            let (dist, u, z) = closest_in_plan(leg[0], leg[1], c);
            let end_outside = |q: V3| {
                let (dx, dy) = (q.x - c.0, q.y - c.1);
                (dx * dx + dy * dy).sqrt() >= radius || q.z > top
            };
            dist > radius + 0.5
                || (u > 0.0 && u < 1.0 && z > top + 0.5)
                || (u == 0.0 && end_outside(leg[0]))
                || (u == 1.0 && end_outside(leg[1]))
        });
        let crosses = p.points.windows(2).any(|leg| {
            let (dist, _, z) = closest_in_plan(leg[0], leg[1], c);
            dist < radius && z < top
        });
        if clear {
            assert_eq!(factor(p), 1.0, "{:?} {:?} is clear", p.kind, p.surfaces);
            unchanged += 1;
        } else if crosses {
            assert!(
                factor(p) < 1.0,
                "{:?} {:?} passes through",
                p.kind,
                p.surfaces
            );
            through += 1;
        }
    }
    // The y-wall bounces, the ceiling bounce and most clutter paths are
    // clear; the LoS, the x-wall bounces and the floor bounce are not.
    assert!(unchanged >= 3, "{unchanged} unchanged paths");
    assert!(through >= 4, "{through} paths through the person");
    // The floor bounce lands at their feet: both legs end inside the
    // cylinder, each taking the cap (§8.5 step 2).
    let floor = paths
        .iter()
        .find(|p| p.kind == PathKind::Bounce1 && p.surfaces == [4])
        .expect("a floor bounce");
    assert_eq!(floor.points[1], v3(3.0, 2.0, 0.0));
    let capped = 10f64.powf(-cap / 20.0);
    assert_eq!(factor(floor), (1.0 * capped) * capped);

    // In the channel: switching shadowing off raises the link's power
    // (the person's own body terms are not shadowed by themselves).
    let off = Simulator::new(
        &s,
        Config {
            shadowing: false,
            ..cfg.clone()
        },
    )
    .unwrap();
    let h_on = sim.channel(li, &at);
    let h_off = off.channel(li, &off.people_at(0.0, &[pose]));
    assert!(power(&h_on) < power(&h_off));
}

fn wall_link_power(material: &str) -> (f64, Cx) {
    let walls = format!(
        r#"{{"from": [0.0, 0.0], "to": [6.0, 0.0], "material": "brick", "thickness_m": 0.2}},
           {{"from": [6.0, 0.0], "to": [6.0, 4.0], "material": "brick", "thickness_m": 0.2}},
           {{"from": [6.0, 4.0], "to": [0.0, 4.0], "material": "brick", "thickness_m": 0.2}},
           {{"from": [0.0, 4.0], "to": [0.0, 0.0], "material": "brick", "thickness_m": 0.2}},
           {{"from": [3.0, 0.0], "to": [3.0, 4.0], "material": "{material}", "thickness_m": 0.15}}"#
    );
    let s = rect(
        6.0,
        4.0,
        2.7,
        &walls,
        "",
        r#"{"id": 0, "pos": [1.0, 2.0, 1.2]}, {"id": 1, "pos": [5.0, 2.0, 1.2]}"#,
        "",
    );
    let sim = Simulator::new(&s, Config::default()).unwrap();
    let li = sim.link_index(0, 1).unwrap();
    let h = sim.channel(li, &sim.people_at(0.0, &[]));
    (power(&h), sim.paths(li)[0].coeff)
}

/// §15: a link through a concrete wall is weaker than through drywall.
#[test]
fn concrete_wall_is_weaker_than_drywall() {
    let (p_conc, los_conc) = wall_link_power("concrete");
    let (p_dry, los_dry) = wall_link_power("drywall");
    assert!(
        cabs2(los_conc) < cabs2(los_dry),
        "LoS through concrete {los_conc:?} vs drywall {los_dry:?}"
    );
    assert!(p_conc < p_dry, "link power {p_conc} vs {p_dry}");
}

/// §15: a metal box on the LoS weakens the link, and no face of it radiates
/// through it.
#[test]
fn metal_box_on_the_line_of_sight() {
    let nodes = r#"{"id": 0, "pos": [1.0, 2.0, 1.2]}, {"id": 1, "pos": [5.0, 2.0, 1.2]}"#;
    let boxed = rect(
        6.0,
        4.0,
        2.7,
        "",
        r#"{"type": "almirah", "pos": [3.0, 2.0, 1.0], "size": [0.6, 1.0, 2.0], "material": "metal"}"#,
        nodes,
        "",
    );
    let open = rect(6.0, 4.0, 2.7, "", "", nodes, "");
    let (a, b) = (
        Simulator::new(&boxed, Config::default()).unwrap(),
        Simulator::new(&open, Config::default()).unwrap(),
    );
    let li = a.link_index(0, 1).unwrap();
    assert!(
        a.paths(li).iter().all(|p| p.object.is_none()),
        "a face lit from one side only must not radiate"
    );
    let los = a.paths(li)[0].coeff;
    let los_free = b.paths(li)[0].coeff;
    assert!(
        (cabs2(los) / cabs2(los_free) - pow10(-2.0)).abs() < 1e-12,
        "20 dB blockage"
    );
    let pa = power(&a.channel(li, &a.people_at(0.0, &[])));
    let pb = power(&b.channel(li, &b.people_at(0.0, &[])));
    assert!(pa < pb, "{pa} vs {pb}");
}

/// §15: waypoints and legs avoid furniture; speeds and pauses are in range.
#[test]
fn walks_avoid_furniture_and_stay_in_range() {
    let hall = Simulator::new(&load("hall-l-shaped.json"), Config::default()).unwrap();
    check_walks(&hall, 0);
    // A polygon room with a 30-degree yawed cabinet, and a person who
    // entered at the door.
    let multi = Simulator::new(&load("fixtures/roomsim_multi.json"), Config::default()).unwrap();
    for i in 0..3 {
        check_walks(&multi, i);
    }
    assert!(!multi.people()[2].scene_start);
}

fn check_walks(sim: &Simulator, person: usize) {
    let cfg = sim.config().clone();
    assert_eq!(sim.people()[person].mode, Mode::Walk);
    let w = sim.walkability();
    let room = sim.room();
    let mut tr = sim.track(person).unwrap();
    let (mut t, mut walks, mut pauses) = (0.0, 0, 0);
    while t < 3000.0 {
        let seg = tr.segment(t);
        assert_eq!(seg.t0, t, "segments are contiguous");
        assert!(seg.t1 > seg.t0);
        match seg.kind {
            saans_roomsim::motion::SegmentKind::Walk { from, to, speed } => {
                walks += 1;
                assert!(w.point(from) && w.point(to), "waypoints walkable");
                assert!(w.leg(from, to), "leg walkable");
                assert!(
                    (cfg.speed_mps.0..cfg.speed_mps.1).contains(&speed),
                    "speed {speed}"
                );
                let dx = to.0 - from.0;
                let dy = to.1 - from.1;
                let dd = (dx * dx + dy * dy).sqrt();
                assert!(dd >= 0.5);
                assert!(((seg.t1 - seg.t0) - dd / speed).abs() < 1e-9);
                // Every sampled point of the leg is clear of every grown box.
                for i in 0..=20 {
                    let f = f64::from(i) / 20.0;
                    let p = (from.0 + dx * f, from.1 + dy * f);
                    for o in room.objects.iter().filter(|o| !o.is_opening()) {
                        let (lx, ly) = o.to_local_xy(p);
                        let rg = cfg.body_radius_m + cfg.object_clearance_m;
                        assert!(
                            lx.abs() >= o.sl / 2.0 + rg || ly.abs() >= o.sb / 2.0 + rg,
                            "inside {:?}",
                            o.kind
                        );
                    }
                }
            }
            saans_roomsim::motion::SegmentKind::Pause { .. } => {
                pauses += 1;
                let dur = seg.t1 - seg.t0;
                assert!(dur > 0.0 && dur < cfg.pause_s.1 + 1e-9, "pause {dur}");
                assert!(w.point(seg.end()));
            }
        }
        t = seg.t1;
    }
    assert!(walks > 50 && pauses > 50, "{walks} walks, {pauses} pauses");
}

/// §15: with impairments off, |buffer| is proportional to |H| per bin.
#[test]
fn without_impairments_the_buffer_is_proportional_to_h() {
    let s = load("bedroom.json");
    let cfg = Config {
        impair: false,
        quantize: false,
        duration_s: 0.3,
        ..Config::default()
    };
    let sim = Simulator::new(&s, cfg).unwrap();
    let mut n = 0;
    for p in sim.packets() {
        assert!(!p.lost, "nothing is lost with impair off");
        let r = p.received.as_ref().unwrap();
        let mut ratio = None;
        for e in 0..128 {
            let b = r.buffer_pre[e];
            if !buffer_active(e) {
                assert_eq!(b, Cx::default(), "inactive entries are exactly zero");
                continue;
            }
            let hk = r.h[bin_index(buffer_k(e))];
            let q = (cabs2(b) / cabs2(hk)).sqrt();
            let r0 = *ratio.get_or_insert(q);
            assert!((q - r0).abs() <= 1e-12 * r0, "entry {e}: {q} vs {r0}");
        }
        n += 1;
    }
    assert!(n > 50);
}

fn assert_reciprocal(sim: &Simulator, poses: &[saans_roomsim::people::Pose]) -> usize {
    let at = sim.people_at(0.25, poses);
    let mut n = 0;
    for l in sim.links() {
        let back = sim.link_index(l.rx, l.tx).unwrap();
        let (a, b) = (sim.channel(l.index, &at), sim.channel(back, &at));
        let m = a.iter().map(|c| cabs2(*c).sqrt()).fold(0.0, f64::max);
        for (x, y) in a.iter().zip(&b) {
            let diff = cabs2(Cx {
                re: x.re - y.re,
                im: x.im - y.im,
            })
            .sqrt();
            assert!(
                diff <= 1e-9 * m,
                "{}->{}: |dH| = {diff}, max |H| = {m}",
                l.tx,
                l.rx
            );
        }
        // Same path count by kind both ways.
        let kinds = |i: usize| {
            let mut k = [0usize; 7];
            sim.paths(i)
                .iter()
                .for_each(|p| k[p.kind.code() as usize] += 1);
            k
        };
        assert_eq!(kinds(l.index), kinds(back));
        n += 1;
    }
    n
}

/// Move every node by a few centimetres so that no leg passes exactly
/// through a box edge and no bounce lands exactly on a wall-ceiling edge.
fn jitter_nodes(s: &mut Scene) {
    const D: [f64; 8] = [
        0.0131, -0.0173, 0.0219, -0.0097, 0.0157, -0.0211, 0.0083, -0.0149,
    ];
    for (i, n) in s.nodes.iter_mut().enumerate() {
        n.pos.0 += D[i % 8];
        n.pos.1 += D[(i + 3) % 8];
        n.pos.2 += D[(i + 5) % 8] * 3.0;
    }
}

/// §15: H(a->b) and H(b->a) agree before impairments.
///
/// The nodes are moved a few centimetres first. The scenes' round numbers
/// put some paths exactly on a boundary: in the bedroom, nodes 0 and 1 are
/// mirror images at one height, so a few order-2 paths bounce on the edge
/// where a wall meets the floor or ceiling and `hit()` admits them (`X.z <=
/// h`) by the last bit, differently in the two directions; in the hall the
/// leg from node 1 to a floor bounce passes exactly through the sofa's top
/// edge (`t_exit == t_enter` to rounding). §15 states this exception (and
/// §6.4 why the two implementations still agree on such paths). Away from
/// such coincidences the channel is reciprocal to rounding.
#[test]
fn channel_is_reciprocal_before_impairments() {
    let mut s = load("bedroom.json");
    jitter_nodes(&mut s);
    let bedroom = Simulator::new(&s, Config::default()).unwrap();
    let poses: Vec<_> = bedroom
        .people()
        .iter()
        .map(|p| bedroom.stationary_pose(p))
        .collect();
    assert_eq!(assert_reciprocal(&bedroom, &poses), 12);

    let mut s = load("hall-l-shaped.json");
    jitter_nodes(&mut s);
    let hall = Simulator::new(
        &s,
        Config {
            n_people: Some(0),
            ..Config::default()
        },
    )
    .unwrap();
    assert_eq!(assert_reciprocal(&hall, &[]), 56);

    let mut s = load("fixtures/roomsim_multi.json");
    jitter_nodes(&mut s);
    let multi = Simulator::new(&s, Config::default()).unwrap();
    let poses: Vec<_> = (0..3)
        .map(|i| {
            let s = multi.track(i).unwrap().at(0.25);
            saans_roomsim::people::Pose {
                x: s.x,
                y: s.y,
                heading: s.heading,
                walking: s.walking,
                speed: s.speed,
                posture: saans_roomsim::Posture::Stand,
            }
        })
        .collect();
    assert_eq!(assert_reciprocal(&multi, &poses), 6);
}

/// §6.5: an object face large enough for its Fresnel zone reflects
/// specularly, off the face, with the single-interface G of its material.
#[test]
fn object_faces_reflect_specularly() {
    let s = rect(
        6.0,
        5.0,
        2.7,
        "",
        r#"{"type": "almirah", "pos": [3.0, 4.4, 1.0], "size": [3.0, 0.6, 2.0], "material": "wood"}"#,
        r#"{"id": 0, "pos": [1.5, 1.0, 1.2]}, {"id": 1, "pos": [4.5, 1.0, 1.2]}"#,
        "",
    );
    let sim = Simulator::new(&s, Config::default()).unwrap();
    let li = sim.link_index(0, 1).unwrap();
    let spec: Vec<_> = sim
        .paths(li)
        .iter()
        .filter(|p| p.kind == PathKind::ObjectSpecular)
        .collect();
    assert_eq!(spec.len(), 1);
    let p = spec[0];
    assert_eq!(p.object, Some((0, 3)), "the -y face");
    let x = p.points[1];
    assert!(
        (x.x - 3.0).abs() < 1e-12 && (x.y - 4.1).abs() < 1e-12 && (x.z - 1.2).abs() < 1e-12,
        "{x:?}"
    );
    let lambda = sim.world().lambda;
    let cos_t = 3.1 / (1.5f64 * 1.5 + 3.1 * 3.1).sqrt();
    let g = saans_roomsim::materials::interface_g(
        saans_roomsim::materials::eta(
            saans_roomsim::materials::ItuMaterial::Wood,
            sim.world().carrier_hz,
        ),
        cos_t,
        saans_roomsim::materials::Pol::Te,
    );
    let want = cabs2(g).sqrt() * lambda / (saans_roomsim::numerics::FOUR_PI * p.length);
    assert!((cabs2(p.coeff_free).sqrt() - want).abs() < 1e-12 * want);

    // Every kind-3 path of the shipped scenes lies on its face and obeys
    // the specular law about the face normal, yawed objects included: the
    // specular fixture's metal locker (80 degrees), glass table (10) and
    // wardrobe (-5) are the yawed faces that reflect specularly.
    let mut yawed = 0;
    for name in [
        "hall-l-shaped.json",
        "office.json",
        "fixtures/roomsim_multi.json",
        "fixtures/roomsim_specular.json",
    ] {
        let sim = Simulator::new(&load(name), Config::default()).unwrap();
        for l in sim.links() {
            for p in sim
                .paths(l.index)
                .iter()
                .filter(|p| p.kind == PathKind::ObjectSpecular)
            {
                let (oi, f) = p.object.unwrap();
                let object = &sim.room().objects[oi];
                if object.psi != 0.0 {
                    yawed += 1;
                }
                let face = saans_roomsim::paths::Face::of(object, f);
                let n = face.normal;
                let x = p.points[1];
                assert!(
                    dot3(sub3(x, face.centre), n).abs() < 1e-12,
                    "{name}: off the face plane"
                );
                let u_in = unit_dir(p.points[0], x);
                let u_out = unit_dir(x, p.points[2]);
                let dn = 2.0 * dot3(u_in, n);
                let err = dist3(
                    v3(u_in.x - dn * n.x, u_in.y - dn * n.y, u_in.z - dn * n.z),
                    u_out,
                );
                assert!(err < 1e-12, "{name}: specular error {err}");
                // The point is on the face itself, not just its plane.
                let lx = object.to_local(x);
                let (hl, hb, hh) = (object.sl / 2.0, object.sb / 2.0, object.sh / 2.0);
                let inside = |a: f64, h: f64| a.abs() <= h + 1e-12;
                let on_face = match f {
                    0 | 1 => inside(lx.y, hb) && inside(lx.z, hh),
                    2 | 3 => inside(lx.x, hl) && inside(lx.z, hh),
                    _ => inside(lx.x, hl) && inside(lx.y, hb),
                };
                assert!(on_face, "{name}: {x:?} is off face {f} of objects[{oi}]");
            }
        }
    }
    assert!(yawed > 0, "no yawed object-specular path was checked");
}
