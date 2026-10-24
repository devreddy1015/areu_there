//! Port of the key cases of tests/test_localize_geometry.py.
//!
//! Exact numbers marked "CPython 3.14" were printed by the Python reference.

use saans_locate::geometry::*;
use saans_locate::pynum::LocateError;

const INSET: f64 = 0.1;

/// Independent inverse of the perimeter walk (the test file's own).
fn arc_length_on_inset_rect(room: &Room, inset: f64, x: f64, y: f64) -> f64 {
    let (x0, y0) = (inset, inset);
    let x1 = room.width_m - inset;
    let y1 = room.height_m - inset;
    let w = x1 - x0;
    let h = y1 - y0;
    let tol = 1e-9;
    if (y - y0).abs() < tol && x0 - tol <= x && x <= x1 + tol {
        return x - x0;
    }
    if (x - x1).abs() < tol && y0 - tol <= y && y <= y1 + tol {
        return w + (y - y0);
    }
    if (y - y1).abs() < tol && x0 - tol <= x && x <= x1 + tol {
        return w + h + (x1 - x);
    }
    if (x - x0).abs() < tol && y0 - tol <= y && y <= y1 + tol {
        return 2.0 * w + h + (y1 - y);
    }
    panic!("({x}, {y}) is not on the inset rectangle");
}

fn rem(a: f64, b: f64) -> f64 {
    saans_locate::pynum::py_mod(a, b).unwrap()
}

#[test]
fn room_defaults_validation_and_contains() {
    let room = Room::default();
    assert_eq!((room.width_m, room.height_m), (8.0, 6.0));
    assert_eq!(room.center(), (4.0, 3.0));
    assert_eq!(room.perimeter_m(), 28.0);
    assert_eq!(room.area_m2(), 48.0);
    for (w, h) in [(0.0, 6.0), (8.0, -1.0)] {
        assert!(matches!(Room::new(w, h), Err(LocateError::ValueError(_))));
    }
    // `nan <= 0.0` is False: Python accepts a NaN side.
    assert!(Room::new(f64::NAN, 6.0).is_ok());
    assert!(room.contains(0.0, 0.0, 0.0) && room.contains(8.0, 6.0, 0.0));
    assert!(!room.contains(8.01, 3.0, 0.0) && room.contains(8.01, 3.0, 0.02));
    assert!(!room.contains(f64::NAN, 3.0, 0.0));
}

#[test]
fn spacing_is_equal_arc_length_for_many_n() {
    let room = Room::default();
    let p = 2.0 * ((8.0 - 2.0 * INSET) + (6.0 - 2.0 * INSET));
    for n in [2i64, 3, 4, 5, 6, 7, 8, 11, 12, 13, 17] {
        let nodes = perimeter_nodes(&room, n, INSET, 0.0).unwrap();
        assert_eq!(nodes.len() as i64, n);
        let s: Vec<f64> = nodes
            .iter()
            .map(|nd| arc_length_on_inset_rect(&room, INSET, nd.x, nd.y))
            .collect();
        for i in 0..n as usize {
            let gap = rem(s[(i + 1) % n as usize] - s[i], p);
            assert!((gap - p / n as f64).abs() < 1e-9, "n={n} gap {gap}");
        }
    }
}

#[test]
fn eight_nodes_land_three_one_three_one_with_cpython_artefacts() {
    let nodes = perimeter_nodes(&Room::default(), 8, INSET, 0.0).unwrap();
    // CPython 3.14 positions, bit for bit.
    let want = [
        (0.1, 0.1),
        (3.5000000000000004, 0.1),
        (6.9, 0.1),
        (7.9, 2.5000000000000004),
        (7.9, 5.9),
        (4.500000000000002, 5.9),
        (1.0999999999999996, 5.9),
        (0.1, 3.4999999999999982),
    ];
    for (i, (nd, &(x, y))) in nodes.iter().zip(&want).enumerate() {
        assert_eq!(nd.node_id, i as i64);
        assert_eq!((nd.x, nd.y), (x, y), "node {i}");
    }
}

#[test]
fn start_offset_rotates_and_tall_room_keeps_spacing() {
    let room = Room::default();
    let p = 2.0 * ((8.0 - 2.0 * INSET) + (6.0 - 2.0 * INSET));
    let nodes = perimeter_nodes(&room, 8, INSET, p / 16.0).unwrap();
    let s: Vec<f64> = nodes
        .iter()
        .map(|nd| arc_length_on_inset_rect(&room, INSET, nd.x, nd.y))
        .collect();
    assert!((s[0] - p / 16.0).abs() < 1e-9);

    let tall = Room::new(2.0, 10.0).unwrap();
    let pt = 2.0 * ((2.0 - 2.0 * INSET) + (10.0 - 2.0 * INSET));
    let nodes = perimeter_nodes(&tall, 6, INSET, 0.0).unwrap();
    let s: Vec<f64> = nodes
        .iter()
        .map(|nd| arc_length_on_inset_rect(&tall, INSET, nd.x, nd.y))
        .collect();
    for i in 0..6 {
        assert!((rem(s[(i + 1) % 6] - s[i], pt) - pt / 6.0).abs() < 1e-9);
    }
}

#[test]
fn perimeter_rejects_bad_arguments() {
    let room = Room::default();
    assert!(matches!(
        perimeter_nodes(&room, 1, INSET, 0.0),
        Err(LocateError::ValueError(_))
    ));
    assert!(matches!(
        perimeter_nodes(&room, 4, -0.1, 0.0),
        Err(LocateError::ValueError(_))
    ));
    assert!(matches!(
        perimeter_nodes(&room, 4, 3.5, 0.0),
        Err(LocateError::ValueError(_))
    ));
}

#[test]
fn links_are_every_ordered_pair_sorted_tx_then_rx() {
    for n in [2i64, 3, 4, 6, 8, 12] {
        let g = Geometry::perimeter_n(n).unwrap();
        assert_eq!(g.n_links() as i64, n * (n - 1));
        let pairs: Vec<(i64, i64)> = g.links().iter().map(|l| (l.tx_id, l.rx_id)).collect();
        let mut sorted = pairs.clone();
        sorted.sort();
        assert_eq!(pairs, sorted);
        for (i, l) in g.links().iter().enumerate() {
            assert_ne!(l.tx_id, l.rx_id);
            assert_eq!(g.link_index(l.tx_id, l.rx_id).unwrap(), i);
            // For ids 0..N-1: index(i, j) = i*(N-1) + (j if j < i else j-1).
            let (t, r) = (l.tx_id, l.rx_id);
            assert_eq!(i as i64, t * (n - 1) + if r < t { r } else { r - 1 });
        }
    }
    let g = Geometry::perimeter_n(4).unwrap();
    assert!(matches!(g.link_index(0, 0), Err(LocateError::KeyError(_))));
    assert!(matches!(g.link_index(0, 99), Err(LocateError::KeyError(_))));
}

#[test]
fn hand_placed_ids_are_ordered_by_id() {
    let nodes = vec![
        Node::new(7, 0.5, 0.5),
        Node::new(3, 7.5, 0.5),
        Node::new(11, 4.0, 5.5),
    ];
    let g = Geometry::new(Room::default(), nodes, None, 0.3).unwrap();
    assert_eq!(g.n_links(), 6);
    let pairs: Vec<(i64, i64)> = g.links().iter().map(|l| (l.tx_id, l.rx_id)).collect();
    assert_eq!(
        pairs,
        vec![(3, 7), (3, 11), (7, 3), (7, 11), (11, 3), (11, 7)]
    );
    assert_eq!(g.link_index(11, 7).unwrap(), 5);
    assert_eq!(g.node(3).unwrap().x, 7.5);
    assert!(matches!(g.node(99), Err(LocateError::KeyError(_))));
}

#[test]
fn excess_path_is_zero_on_the_segment() {
    let g = Geometry::perimeter_n(8).unwrap();
    for l in g.links() {
        for t in [0.0, 0.25, 0.5, 1.0] {
            let x = l.tx.0 + t * (l.rx.0 - l.tx.0);
            let y = l.tx.1 + t * (l.rx.1 - l.tx.1);
            assert!(l.excess_path_m(x, y).abs() < 1e-12);
            for lam in [0.001, 0.05, 0.3, 2.0] {
                assert!(ellipse_weight(l, x, y, lam).unwrap() > 0.0);
            }
        }
        for (x, y) in [(-50.0, -50.0), (100.0, 0.0), (4.0, 80.0)] {
            assert_eq!(ellipse_weight(l, x, y, 3.0).unwrap(), 0.0);
        }
    }
}

#[test]
fn ellipse_boundary_is_exactly_excluded() {
    let link = Link {
        tx_id: 0,
        rx_id: 1,
        tx: (0.0, 0.0),
        rx: (4.0, 0.0),
        length_m: 4.0,
    };
    assert_eq!(link.excess_path_m(-0.5, 0.0), 1.0);
    assert_eq!(ellipse_weight(&link, -0.5, 0.0, 1.0).unwrap(), 0.0);
    let just_over = f64::from_bits(1.0f64.to_bits() + 1);
    let just_under = f64::from_bits(1.0f64.to_bits() - 1);
    assert!(ellipse_weight(&link, -0.5, 0.0, just_over).unwrap() > 0.0);
    assert_eq!(ellipse_weight(&link, -0.5, 0.0, just_under).unwrap(), 0.0);
    assert_eq!(ellipse_weight(&link, 2.0, 0.0, 0.5).unwrap(), 0.5);
    // Minor half-axis b = 1.5 for d = 4, lambda = 1.
    assert!(ellipse_weight(&link, 2.0, 1.5 - 1e-6, 1.0).unwrap() > 0.0);
    assert_eq!(ellipse_weight(&link, 2.0, 1.5 + 1e-6, 1.0).unwrap(), 0.0);
    for lam in [0.0, -0.5] {
        assert!(matches!(
            ellipse_weight(&link, 2.0, 0.0, lam),
            Err(LocateError::ValueError(_))
        ));
    }
    let degenerate = Link {
        tx_id: 0,
        rx_id: 1,
        tx: (1.0, 1.0),
        rx: (1.0, 1.0),
        length_m: 0.0,
    };
    assert_eq!(ellipse_weight(&degenerate, 1.0, 1.0, 1.0).unwrap(), 0.0);
    // Walking outward along the major axis crosses once.
    for x in [-0.49, -0.3, 0.0, 2.0, 4.0, 4.3, 4.49] {
        assert!(ellipse_weight(&link, x, 0.0, 1.0).unwrap() > 0.0);
    }
    for x in [-0.51, -1.0, -5.0, 4.51, 5.0, 20.0] {
        assert_eq!(ellipse_weight(&link, x, 0.0, 1.0).unwrap(), 0.0);
    }
}

#[test]
fn reciprocal_links_have_identical_rows() {
    let g = Geometry::perimeter_n(8).unwrap();
    let w = g.weight_matrix();
    for (i, l) in g.links().iter().enumerate() {
        let j = g.link_index(l.rx_id, l.tx_id).unwrap();
        assert_eq!(w[i], w[j]);
        assert_eq!(l.length_m.to_bits(), g.links()[j].length_m.to_bits());
    }
}

#[test]
fn pixel_grid_counts_order_and_half_even_rounding() {
    let room = Room::default();
    for (pitch, expected) in [(2.0, 12), (1.0, 48), (0.5, 192), (0.25, 768), (0.2, 1200)] {
        assert_eq!(PixelGrid::new(room, pitch).unwrap().len(), expected);
    }
    let g = PixelGrid::new(room, 0.7).unwrap();
    assert_eq!((g.n_x(), g.n_y()), (11, 9));
    assert_eq!(g.pitch_x_m(), 8.0 / 11.0);
    // Half to even: 5/2 = 2.5 rounds to 2, 7/2 = 3.5 rounds to 4.
    let five = PixelGrid::new(Room::new(5.0, 5.0).unwrap(), 2.0).unwrap();
    assert_eq!((five.n_x(), five.n_y()), (2, 2));
    let seven = PixelGrid::new(Room::new(7.0, 7.0).unwrap(), 2.0).unwrap();
    assert_eq!((seven.n_x(), seven.n_y()), (4, 4));

    let g = PixelGrid::new(room, 0.5).unwrap();
    assert_eq!(g.pixels()[0], (0.25, 0.25));
    assert_eq!(g.pixels()[1], (0.75, 0.25));
    assert_eq!(g.pixels()[g.n_x()], (0.25, 0.75));
    for iy in 0..g.n_y() as i64 {
        for ix in 0..g.n_x() as i64 {
            assert_eq!(
                g.pixel_index(ix, iy).unwrap() as i64,
                iy * g.n_x() as i64 + ix
            );
        }
    }
    assert!(matches!(
        g.pixel_index(16, 0),
        Err(LocateError::IndexError(_))
    ));
    assert!(matches!(
        g.pixel_index(0, -1),
        Err(LocateError::IndexError(_))
    ));
    assert_eq!(g.nearest_index(-3.0, -3.0).unwrap(), 0);
    assert_eq!(g.nearest_index(99.0, 99.0).unwrap(), g.len() - 1);
    assert!(matches!(
        g.nearest_index(f64::NAN, 1.0),
        Err(LocateError::ValueError(_))
    ));
    assert!(matches!(
        g.nearest_index(1.0, f64::INFINITY),
        Err(LocateError::OverflowError(_))
    ));
    let fine = PixelGrid::new(room, 0.25).unwrap();
    for (i, &(x, y)) in fine.pixels().iter().enumerate() {
        assert_eq!(fine.nearest_index(x, y).unwrap(), i);
    }
    for pitch in [0.0, -0.5] {
        assert!(matches!(
            PixelGrid::new(room, pitch),
            Err(LocateError::ValueError(_))
        ));
    }
    // round(nan) raises ValueError; round(inf) raises OverflowError.
    assert!(matches!(
        PixelGrid::new(room, f64::NAN),
        Err(LocateError::ValueError(_))
    ));
    assert!(matches!(
        PixelGrid::new(
            Room {
                width_m: f64::INFINITY,
                height_m: 6.0
            },
            0.5
        ),
        Err(LocateError::OverflowError(_))
    ));
    // An infinite pitch gives a one-pixel grid, as in Python.
    let one = PixelGrid::new(room, f64::INFINITY).unwrap();
    assert_eq!((one.n_x(), one.n_y()), (1, 1));
}

#[test]
fn weight_matrix_agrees_with_ellipse_weight_and_supports() {
    let g = Geometry::perimeter_n(8).unwrap();
    let w = g.weight_matrix();
    assert_eq!((w.len(), w[0].len()), (56, 192));
    for (l, (row, support)) in g.links().iter().zip(w.iter().zip(g.supports())) {
        let nz: Vec<usize> = row
            .iter()
            .enumerate()
            .filter(|(_, &v)| v != 0.0)
            .map(|(p, _)| p)
            .collect();
        assert_eq!(&nz, support);
        for (p, &(x, y)) in g.grid().pixels().iter().enumerate() {
            assert_eq!(row[p], ellipse_weight(l, x, y, g.lambda_m()).unwrap());
        }
    }
    let nnz: usize = g.supports().iter().map(|s| s.len()).sum();
    assert_eq!(
        nnz, 1856,
        "CPython 3.14: 1856 nonzeros at 8 nodes, 0.5 m, 0.3 m"
    );
}

#[test]
fn larger_lambda_nests_supports() {
    let g = Geometry::perimeter_n(8).unwrap();
    let wide = g.with_lambda(0.75).unwrap();
    assert_eq!(wide.grid(), g.grid());
    assert_eq!(wide.nodes(), g.nodes());
    for (a, b) in g.supports().iter().zip(wide.supports()) {
        assert!(a.iter().all(|p| b.contains(p)));
        assert!(b.len() > a.len());
    }
}

#[test]
fn blind_pixels_are_the_documented_ones() {
    let g = Geometry::perimeter(8, None, 0.5, 0.3, INSET, 0.0).unwrap();
    let holes: Vec<(f64, f64)> = g
        .pixel_link_counts()
        .iter()
        .enumerate()
        .filter(|(_, &c)| c == 0)
        .map(|(p, _)| g.grid().pixels()[p])
        .collect();
    assert_eq!(holes, vec![(7.75, 0.25), (0.25, 5.75)]);
    assert_eq!(
        Geometry::perimeter(8, None, 0.5, 0.75, INSET, 0.0)
            .unwrap()
            .pixel_link_counts()
            .iter()
            .filter(|&&c| c == 0)
            .count(),
        0
    );
    assert_eq!(
        Geometry::perimeter(12, None, 0.5, 0.3, INSET, 0.0)
            .unwrap()
            .pixel_link_counts()
            .iter()
            .filter(|&&c| c == 0)
            .count(),
        2
    );
    let four = Geometry::perimeter(4, None, 0.5, 0.3, INSET, 0.0).unwrap();
    let holes: Vec<(f64, f64)> = four
        .pixel_link_counts()
        .iter()
        .enumerate()
        .filter(|(_, &c)| c == 0)
        .map(|(p, _)| four.grid().pixels()[p])
        .collect();
    assert_eq!(holes.len(), 16);
    assert!(holes
        .iter()
        .any(|&(x, y)| 1.0 < x && x < 7.0 && 1.0 < y && y < 5.0));
    let six = Geometry::perimeter(6, None, 0.5, 0.5, INSET, 0.0).unwrap();
    assert_eq!(
        six.pixel_link_counts().iter().sum::<usize>(),
        six.supports().iter().map(|s| s.len()).sum::<usize>()
    );
}

#[test]
fn geometry_validation() {
    let room = Room::default();
    assert!(matches!(
        Geometry::new(room, vec![Node::new(0, 1.0, 1.0)], None, 0.3),
        Err(LocateError::ValueError(_))
    ));
    assert!(matches!(
        Geometry::new(
            room,
            vec![Node::new(0, 1.0, 1.0), Node::new(0, 2.0, 2.0)],
            None,
            0.3
        ),
        Err(LocateError::ValueError(_))
    ));
    assert!(matches!(
        Geometry::new(
            room,
            vec![Node::new(0, 1.0, 1.0), Node::new(1, 1.0, 1.0)],
            None,
            0.3
        ),
        Err(LocateError::ValueError(_))
    ));
    // 0.0 and -0.0 are one position, as in a Python set.
    assert!(Geometry::new(
        room,
        vec![Node::new(0, 0.0, 1.0), Node::new(1, -0.0, 1.0)],
        None,
        0.3
    )
    .is_err());
    assert!(matches!(
        Geometry::perimeter(4, None, 0.5, 0.0, INSET, 0.0),
        Err(LocateError::ValueError(_))
    ));
    let g = Geometry::perimeter(8, None, 0.25, 0.3, INSET, 0.0).unwrap();
    assert_eq!((g.n_nodes(), g.n_pixels(), g.n_links()), (8, 768, 56));
}
