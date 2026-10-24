//! §11 "Generator": seeds 0..3 x 200 indices, drawn and checked (§2.6)
//! only; nothing is simulated past `Simulator::new`. Everything is SIM.

use super::*;
use rayon::prelude::*;
use saans_roomsim::Posture;
use saans_scene::geom::{distance_to_segment, signed_area2};

const SEEDS: [u64; 4] = [0, 1, 2, 3];
const INDICES: usize = 200;
const SECONDS: u32 = 120;
/// The two-sided 99% normal quantile.
const Z99: f64 = 2.5758;
const EPS: f64 = 1e-9;

fn all() -> Vec<Generated> {
    let jobs: Vec<(u64, usize)> = SEEDS
        .iter()
        .flat_map(|&s| (0..INDICES).map(move |i| (s, i)))
        .collect();
    jobs.par_iter()
        .map(|&(s, i)| {
            generate(s, i, SECONDS).unwrap_or_else(|x| panic!("seed {s} scene {i}: {x:?}"))
        })
        .collect()
}

/// `count` of `n` is inside the 99% binomial interval of `p`.
fn binomial_ok(what: &str, count: usize, n: usize, p: f64) {
    let (n, k) = (n as f64, count as f64);
    let half = Z99 * (n * p * (1.0 - p)).sqrt();
    assert!(
        (k - n * p).abs() <= half,
        "{what}: {k} of {n}, expected {} +/- {half:.1} (p = {p})",
        n * p
    );
}

/// The one test that draws all 800 scenes (they take a while in the dev
/// profile), with every §11 generator check on them.
#[test]
fn the_generator_draws_valid_scenes_with_its_distributions() {
    let scenes = all();
    let n = scenes.len();
    assert_eq!(n, SEEDS.len() * INDICES);

    // At least 95% accepted at attempt 0.
    let first = scenes.iter().filter(|g| g.attempts == 1).count();
    assert!(
        first * 100 >= 95 * n,
        "{first} of {n} accepted at attempt 0"
    );
    for g in &scenes {
        assert_eq!(u64::from(g.attempts) - 1, g.rejections.total());
    }

    // The accepted shares against §2's.
    for (m, p, _) in WALL_MATERIALS {
        let k = scenes.iter().filter(|g| g.wall_material == m).count();
        binomial_ok(wall_material_name(m), k, n, p);
    }
    binomial_ok(
        "L-shape",
        scenes.iter().filter(|g| g.l_shape).count(),
        n,
        L_SHAPE_P,
    );
    let nodes = NODE_COUNT.1 - NODE_COUNT.0 + 1;
    for count in NODE_COUNT.0..=NODE_COUNT.1 {
        let k = scenes
            .iter()
            .filter(|g| g.scene.nodes.len() == count)
            .count();
        binomial_ok(&format!("{count} nodes"), k, n, 1.0 / nodes as f64);
    }
    let people = PEOPLE_COUNT.1 - PEOPLE_COUNT.0 + 1;
    for count in PEOPLE_COUNT.0..=PEOPLE_COUNT.1 {
        let k = scenes
            .iter()
            .filter(|g| g.scene.people.len() == count)
            .count();
        binomial_ok(&format!("{count} people"), k, n, 1.0 / people as f64);
    }

    scenes.par_iter().for_each(check_one);
}

/// Every geometric rule of §2 on one accepted scene, and §2.6 again.
fn check_one(g: &Generated) {
    let s = &g.scene;
    let name = &s.name;
    assert_eq!(*name, scene_id(g.index));
    assert_eq!(s.provenance, Provenance::Generated);
    assert!(s.validate().is_empty(), "{name}: {:?}", s.validate());
    check(s, &g.rs).unwrap_or_else(|e| panic!("{name}: {e:?}"));

    // §2.1: the footprint is counter-clockwise, its bounding box the room.
    let fp = s.footprint();
    assert!(signed_area2(&fp) > 0.0, "{name}: clockwise footprint");
    assert_eq!(fp.len(), if g.l_shape { 6 } else { 4 });
    assert_eq!(s.room.polygon.is_some(), g.l_shape);
    let (w, d) = (s.room.w_m, s.room.d_m);
    assert!(fp
        .iter()
        .all(|p| (0.0..=w).contains(&p.0) && (0.0..=d).contains(&p.1)));
    for (lo, hi, v) in [
        (ROOM_W_M.0, ROOM_W_M.1, w),
        (ROOM_D_M.0, ROOM_D_M.1, d),
        (ROOM_H_M.0, ROOM_H_M.1, s.room.h_m),
    ] {
        assert!(
            v >= lo && v <= hi && (v * 100.0 - (v * 100.0).round()).abs() < 1e-6,
            "{name}: {v}"
        );
    }

    // §2.2: one wall per edge, one material and thickness.
    let es = edges(&fp);
    assert_eq!(s.walls.len(), es.len());
    let t = s.walls[0].thickness_m;
    for (wall, &(a, b)) in s.walls.iter().zip(&es) {
        assert_eq!((wall.from, wall.to), (a, b));
        assert_eq!(
            (wall.material, wall.thickness_m, wall.height_m),
            (g.wall_material, t, None)
        );
    }
    let (_, _, (t_lo, t_hi)) = WALL_MATERIALS
        .iter()
        .find(|m| m.0 == g.wall_material)
        .copied()
        .unwrap();
    assert!(t >= t_lo && t <= t_hi, "{name}: t = {t}");

    // §2.3: furniture keeps its clearances and never overlaps.
    let furniture: Vec<&Object> = s
        .objects
        .iter()
        .filter(|o| !matches!(o.kind, ObjectKind::Door | ObjectKind::Window))
        .collect();
    assert_eq!(furniture.len(), g.objects_placed);
    assert!(g.objects_placed <= g.objects_drawn && g.objects_drawn <= FURNITURE_COUNT.1);
    for (k, o) in furniture.iter().enumerate() {
        assert!(o.yaw_deg == 0.0 || o.yaw_deg == 90.0 || o.yaw_deg == 180.0 || o.yaw_deg == 270.0);
        assert_eq!(o.pos.2, o.size.2 / 2.0, "{name}: stands on the floor");
        let r = Rect::of_object(o);
        assert!(point_in_polygon(&fp, (o.pos.0, o.pos.1)));
        for &e in &es {
            assert!(
                r.distance(&Rect::of_segment(e)) >= t / 2.0 + FURNITURE_FACE_GAP_M - EPS,
                "{name}: object {k} near a wall"
            );
        }
        for (j, q) in furniture.iter().enumerate().skip(k + 1) {
            assert!(
                r.distance(&Rect::of_object(q)) >= FURNITURE_GAP_M - EPS,
                "{name}: objects {k} and {j}"
            );
        }
    }

    // §2.3: openings centred on an edge's centreline, along it, apart.
    let openings: Vec<&Object> = s
        .objects
        .iter()
        .filter(|o| matches!(o.kind, ObjectKind::Door | ObjectKind::Window))
        .collect();
    assert_eq!(openings.len(), g.openings_placed);
    assert!(g.openings_placed <= g.openings_drawn);
    assert!(s.objects[..g.objects_placed]
        .iter()
        .all(|o| !matches!(o.kind, ObjectKind::Door | ObjectKind::Window)));
    let mut on_edge: Vec<(usize, f64, f64)> = Vec::new();
    for o in &openings {
        let p = (o.pos.0, o.pos.1);
        let e = es
            .iter()
            .position(|&(a, b)| distance_to_segment(a, b, p) < EPS)
            .unwrap_or_else(|| panic!("{name}: an opening off every edge"));
        let (a, b) = es[e];
        let len = length((a, b));
        assert!(len >= OPENING_MIN_EDGE_M);
        let s_along = (p.0 - a.0).hypot(p.1 - a.1);
        assert!(s_along >= OPENING_END_M - EPS && s_along <= len - OPENING_END_M + EPS);
        let dir = (b.1 - a.1).atan2(b.0 - a.0).to_degrees();
        assert_eq!(o.yaw_deg, dir, "{name}: yaw is the edge's direction");
        let want = if o.kind == ObjectKind::Door {
            DOOR
        } else {
            WINDOW
        };
        assert_eq!(
            (o.size, o.pos.2, o.material),
            (want.size, want.z, want.material)
        );
        for &(_, ps, ph) in on_edge.iter().filter(|x| x.0 == e) {
            assert!(
                (s_along - ps).abs() >= o.size.0 / 2.0 + ph + OPENING_GAP_M - EPS,
                "{name}: openings overlap"
            );
        }
        on_edge.push((e, s_along, o.size.0 / 2.0));
    }

    // §2.4: nodes inset from an edge, along it, 0.5 m apart in (x, y).
    let inset = (t / 2.0).max(NODE_INSET_MIN_M) + NODE_INSET_EXTRA_M;
    assert!(s.nodes.len() >= NODE_COUNT.0 && s.nodes.len() <= NODE_COUNT.1);
    for (k, node) in s.nodes.iter().enumerate() {
        assert_eq!(usize::from(node.id), k, "{name}: ids in placement order");
        let p = (node.pos.0, node.pos.1);
        assert!(
            es.iter()
                .any(|&(a, b)| (distance_to_segment(a, b, p) - inset).abs() < 1e-9),
            "{name}: node {k} is not inset {inset} from an edge"
        );
        assert!(es
            .iter()
            .all(|&(a, b)| distance_to_segment(a, b, p) >= inset - 1e-9));
        assert!(node.pos.2 >= NODE_Z_M.0 && node.pos.2 <= NODE_Z_M.1);
        for q in &s.nodes[k + 1..] {
            assert!(
                (q.pos.0 - p.0).hypot(q.pos.1 - p.1) >= NODE_GAP_M,
                "{name}: nodes too close"
            );
        }
    }

    // §2.5: people by index, no ids, no phone; Room Sim runs them in the
    // modes drawn.
    assert!(s.phone.is_none());
    assert!(s.people.iter().all(|p| p.id.is_none() && p.start.is_some()));
    assert_eq!(s.people.len(), g.modes.len());
    assert_eq!(g.rs.n_people, Some(s.people.len()));
    let sim = Simulator::new(s, g.rs.clone()).unwrap();
    for (j, (model, mode)) in sim.people().iter().zip(&g.modes).enumerate() {
        let want = match mode {
            PersonMode::Walk => (Mode::Walk, Posture::Stand),
            PersonMode::Stand => (Mode::Stationary, Posture::Stand),
            PersonMode::Sit => (Mode::Stationary, Posture::Sit),
            PersonMode::Lie => (Mode::Stationary, Posture::Lie),
        };
        assert_eq!((model.mode, model.posture), want, "{name}: people[{j}]");
        match mode {
            PersonMode::Sit | PersonMode::Lie => {
                let o = &s.objects[model.on_object.expect("on an object")];
                assert_eq!(
                    s.people[j].start,
                    Some((o.pos.0, o.pos.1)),
                    "{name}: at the object's centre"
                );
                let kinds: &[ObjectKind] = if *mode == PersonMode::Lie {
                    &[ObjectKind::Bed]
                } else {
                    &[ObjectKind::Chair, ObjectKind::Sofa]
                };
                assert!(kinds.contains(&o.kind));
            }
            PersonMode::Stand => assert!(g.rs.start_heading_rad.contains_key(&j)),
            PersonMode::Walk => {}
        }
    }
    let on: Vec<usize> = sim.people().iter().filter_map(|p| p.resting_on()).collect();
    let mut unique = on.clone();
    unique.sort_unstable();
    unique.dedup();
    assert_eq!(unique.len(), on.len(), "{name}: two people on one object");

    // §3: the run config is Room Sim's defaults but these.
    let mut base = g.rs.clone();
    base.seed = 0;
    base.duration_s = 10.0;
    base.floor_material = SurfaceMaterial::Concrete;
    base.ceiling_material = SurfaceMaterial::Concrete;
    base.n_people = None;
    base.people_mode.clear();
    base.start_heading_rad.clear();
    assert_eq!(base, RoomSimConfig::default());
    assert_eq!(g.rs.duration_s, f64::from(SECONDS) + 0.5);
    assert!(g.rs.impair);
}

#[test]
fn a_scene_is_its_seed_and_index_only() {
    let a = generate(5, 17, 4).unwrap();
    let b = generate(5, 17, 4).unwrap();
    assert_eq!(a.scene, b.scene);
    assert_eq!(a.rs, b.rs);
    let c = generate(5, 18, 4).unwrap();
    assert_ne!(a.scene, c.scene);
    let d = generate(6, 17, 4).unwrap();
    assert_ne!(a.scene, d.scene);
    // The seconds change the run's length only.
    let e = generate(5, 17, 8).unwrap();
    assert_eq!(a.scene, e.scene);
    assert_eq!(e.rs.duration_s, 8.5);
    assert_eq!(a.rs.seed, e.rs.seed);
    // The split draw is its own stream.
    assert_eq!(split_draw(5, 17), split_draw(5, 17));
    assert_ne!(split_draw(5, 17), split_draw(5, 18));
}

#[test]
fn l_polygons_cut_one_corner_and_keep_the_bounding_box() {
    for corner in 0..4 {
        let p = l_polygon(7.31, 4.2, corner, 2.5, 1.7);
        assert_eq!(p.len(), 6);
        assert!(signed_area2(&p) > 0.0);
        let area = signed_area2(&p) / 2.0;
        assert!(
            (area - (7.31 * 4.2 - 2.5 * 1.7)).abs() < 1e-9,
            "corner {corner}: {area}"
        );
        let xs: Vec<f64> = p.iter().map(|q| q.0).collect();
        let ys: Vec<f64> = p.iter().map(|q| q.1).collect();
        assert_eq!(xs.iter().cloned().fold(f64::INFINITY, f64::min), 0.0);
        assert_eq!(xs.iter().cloned().fold(f64::NEG_INFINITY, f64::max), 7.31);
        assert_eq!(ys.iter().cloned().fold(f64::INFINITY, f64::min), 0.0);
        assert_eq!(ys.iter().cloned().fold(f64::NEG_INFINITY, f64::max), 4.2);
    }
}

#[test]
fn the_manifest_lists_every_distribution() {
    let g = generator_json();
    assert_eq!(g["label"], "SIM");
    assert_eq!(g["walls"]["material"].as_array().unwrap().len(), 5);
    assert_eq!(g["furniture"]["kinds"].as_array().unwrap().len(), 5);
    assert_eq!(g["people"]["mode"].as_array().unwrap().len(), 4);
    assert_eq!(g["nodes"]["count"], json!([4, 12]));
    let r = Rejections {
        nodes: 1,
        walk_start: 2,
        scene: 3,
        roomsim: 4,
        locate: 5,
    };
    assert_eq!(Rejections::from_json(&r.to_json()), Some(r));
    assert_eq!(r.total(), 15);
}
