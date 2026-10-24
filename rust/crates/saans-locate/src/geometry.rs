//! Room geometry, link topology and the RTI ellipse weight model: a port
//! of `wifisense/localize/geometry.py`, line by line (spec/locate.md §3).
//!
//! The weight model is the Wilson & Patwari ellipse:
//!
//! ```text
//! w(l, p) = 1/sqrt(d_l)   if   d(tx, p) + d(p, rx) < d_l + lambda
//!           0             otherwise
//! ```
//!
//! The comparison is strict: a pixel exactly on the boundary is OUTSIDE.
//! `lambda_m` is a tuning knob in metres, not the RF wavelength. Every
//! distance is CPython's `math.hypot` ([`py_hypot`]), so supports match the
//! reference pixel for pixel.
//!
//! Two canonical orders every other module indexes against:
//! - **links**: every ordered pair `tx != rx`, sorted by node id, tx
//!   outer, rx inner (`N*(N-1)` links);
//! - **pixels**: row-major with y outermost, `pixels[iy * n_x + ix]`.
//!
//! Geometry is SCENE-labelled input; nothing here touches radio data.

// The Python is the specification line by line: index loops mirror it, and
// `!(a > b)` is how a comparison is written so that a NaN fails closed.
#![allow(clippy::needless_range_loop, clippy::neg_cmp_op_on_partial_ord)]

use std::collections::{HashMap, HashSet};
use std::sync::OnceLock;

use crate::pynum::{
    fmt_py, py_div, py_hypot, py_int, py_mod, py_round_int, pymin, LocateError, Result,
};

/// `Geometry.perimeter`'s and `PixelGrid`'s default pitch, metres.
pub const DEFAULT_PITCH_M: f64 = 0.5;
/// `Geometry`'s default ellipse excess, metres.
pub const DEFAULT_LAMBDA_M: f64 = 0.3;
/// `perimeter_nodes`' default inset from the walls, metres.
pub const DEFAULT_INSET_M: f64 = 0.1;
/// The largest grid `PixelGrid::new` builds; beyond it CPython would raise
/// `MemoryError` or exhaust the machine building the pixel list.
pub const MAX_PIXELS: u128 = 1 << 28;

/// A 2-D rectangular floor plan with its origin at one corner.
///
/// `height_m` is the second horizontal axis of the plan, not elevation
/// (for a scene: `Room(width_m = room.w_m, height_m = room.d_m)`).
/// Build it with [`Room::new`], which applies `__post_init__`'s check.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Room {
    pub width_m: f64,
    pub height_m: f64,
}

impl Default for Room {
    /// `Room()`: 8 x 6 m.
    fn default() -> Self {
        Room {
            width_m: 8.0,
            height_m: 6.0,
        }
    }
}

impl Room {
    /// `Room(width_m, height_m)`: `ValueError` if either side is `<= 0`
    /// (a NaN side passes, as in Python).
    pub fn new(width_m: f64, height_m: f64) -> Result<Room> {
        if width_m <= 0.0 || height_m <= 0.0 {
            return Err(LocateError::ValueError(format!(
                "room dimensions must be positive, got {} x {}",
                fmt_py(width_m),
                fmt_py(height_m)
            )));
        }
        Ok(Room { width_m, height_m })
    }

    /// `2.0 * (width_m + height_m)`.
    pub fn perimeter_m(&self) -> f64 {
        2.0 * (self.width_m + self.height_m)
    }

    /// `width_m * height_m`.
    pub fn area_m2(&self) -> f64 {
        self.width_m * self.height_m
    }

    /// `(width_m / 2.0, height_m / 2.0)`.
    pub fn center(&self) -> (f64, f64) {
        (self.width_m / 2.0, self.height_m / 2.0)
    }

    /// Is `(x, y)` inside the room, within `tol` metres of slack? Inclusive.
    pub fn contains(&self, x: f64, y: f64, tol: f64) -> bool {
        -tol <= x && x <= self.width_m + tol && -tol <= y && y <= self.height_m + tol
    }
}

/// One radio at a fixed, known position (SCENE).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Node {
    pub node_id: i64,
    pub x: f64,
    pub y: f64,
}

impl Node {
    pub fn new(node_id: i64, x: f64, y: f64) -> Node {
        Node { node_id, x, y }
    }

    /// `(x, y)`.
    pub fn position(&self) -> (f64, f64) {
        (self.x, self.y)
    }

    /// `math.hypot(self.x - x, self.y - y)`.
    pub fn distance_to(&self, x: f64, y: f64) -> f64 {
        py_hypot(self.x - x, self.y - y)
    }
}

/// One ordered `(tx, rx)` pair and its line-of-sight length `d_l`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Link {
    pub tx_id: i64,
    pub rx_id: i64,
    pub tx: (f64, f64),
    pub rx: (f64, f64),
    pub length_m: f64,
}

impl Link {
    /// Alias matching the paper's notation.
    pub fn d_l(&self) -> f64 {
        self.length_m
    }

    /// `(0.5 * (tx.x + rx.x), 0.5 * (tx.y + rx.y))`.
    pub fn midpoint(&self) -> (f64, f64) {
        (0.5 * (self.tx.0 + self.rx.0), 0.5 * (self.tx.1 + self.rx.1))
    }

    /// `d(tx, p) + d(p, rx) - d_l`.
    pub fn excess_path_m(&self, x: f64, y: f64) -> f64 {
        let d1 = py_hypot(self.tx.0 - x, self.tx.1 - y);
        let d2 = py_hypot(x - self.rx.0, y - self.rx.1);
        d1 + d2 - self.length_m
    }
}

/// `w(l, p)` for a single link and pixel centre; strict `excess < lambda`.
pub fn ellipse_weight(link: &Link, x: f64, y: f64, lambda_m: f64) -> Result<f64> {
    if lambda_m <= 0.0 {
        return Err(LocateError::ValueError(format!(
            "lambda_m must be positive, got {}",
            fmt_py(lambda_m)
        )));
    }
    if link.length_m <= 0.0 {
        return Ok(0.0);
    }
    if link.excess_path_m(x, y) < lambda_m {
        return Ok(1.0 / link.length_m.sqrt());
    }
    Ok(0.0)
}

/// Place `n_nodes` at equal ARC LENGTH around the room perimeter, inset by
/// `inset_m`, walking counter-clockwise from the inset bottom-left corner
/// (+x first); `start_offset_m` rotates the pattern. Ids are `0..n`.
pub fn perimeter_nodes(
    room: &Room,
    n_nodes: i64,
    inset_m: f64,
    start_offset_m: f64,
) -> Result<Vec<Node>> {
    if n_nodes < 2 {
        return Err(LocateError::ValueError(format!(
            "need at least 2 nodes to form a link, got {n_nodes}"
        )));
    }
    if inset_m < 0.0 {
        return Err(LocateError::ValueError(format!(
            "inset_m must be non-negative, got {}",
            fmt_py(inset_m)
        )));
    }
    if 2.0 * inset_m >= pymin(room.width_m, room.height_m) {
        return Err(LocateError::ValueError(format!(
            "inset_m={} collapses a {} x {} room",
            fmt_py(inset_m),
            fmt_py(room.width_m),
            fmt_py(room.height_m)
        )));
    }

    let x0 = inset_m;
    let y0 = inset_m;
    let x1 = room.width_m - inset_m;
    let y1 = room.height_m - inset_m;
    let w = x1 - x0;
    let h = y1 - y0;
    let perimeter = 2.0 * (w + h);
    let step = perimeter / n_nodes as f64;

    let mut nodes = Vec::new();
    for i in 0..n_nodes {
        let s = py_mod(start_offset_m + i as f64 * step, perimeter)?;
        let pos = if s < w {
            // bottom edge, left to right
            (x0 + s, y0)
        } else if s < w + h {
            // right edge, bottom to top
            (x1, y0 + (s - w))
        } else if s < 2.0 * w + h {
            // top edge, right to left
            (x1 - (s - w - h), y1)
        } else {
            // left edge, top to bottom
            (x0, y1 - (s - 2.0 * w - h))
        };
        nodes.push(Node::new(i, pos.0, pos.1));
    }
    Ok(nodes)
}

/// A regular grid of pixels tiling the room exactly.
///
/// `pitch_m` is a target: `n_x = max(1, round(width / pitch))` (half to
/// even) and the realised pitch is `width / n_x`, so the two axes' pitches
/// generally differ. `pixels()` is the canonical order, `pixels[iy*n_x+ix]`
/// at `((ix + 0.5) * pitch_x, (iy + 0.5) * pitch_y)`.
#[derive(Debug, Clone, PartialEq)]
pub struct PixelGrid {
    room: Room,
    pitch_m: f64,
    n_x: usize,
    n_y: usize,
    pitch_x_m: f64,
    pitch_y_m: f64,
    pixels: Vec<(f64, f64)>,
}

impl PixelGrid {
    /// `PixelGrid(room, pitch_m)`: `ValueError` if `pitch_m <= 0`; `round`
    /// raises `ValueError` on a NaN ratio and `OverflowError` on an infinite
    /// one. A grid above [`MAX_PIXELS`] is refused with `MemoryError`.
    pub fn new(room: Room, pitch_m: f64) -> Result<PixelGrid> {
        if pitch_m <= 0.0 {
            return Err(LocateError::ValueError(format!(
                "pitch_m must be positive, got {}",
                fmt_py(pitch_m)
            )));
        }
        let n_x = py_round_int(room.width_m / pitch_m)?.max(1);
        let n_y = py_round_int(room.height_m / pitch_m)?.max(1);
        if (n_x as u128) * (n_y as u128) > MAX_PIXELS {
            return Err(LocateError::MemoryError(format!(
                "a {n_x} x {n_y} grid is too large to build"
            )));
        }
        let pitch_x_m = room.width_m / n_x as f64;
        let pitch_y_m = room.height_m / n_y as f64;
        let (n_x, n_y) = (n_x as usize, n_y as usize);
        let mut pixels = Vec::with_capacity(n_x * n_y);
        for iy in 0..n_y {
            for ix in 0..n_x {
                pixels.push(((ix as f64 + 0.5) * pitch_x_m, (iy as f64 + 0.5) * pitch_y_m));
            }
        }
        Ok(PixelGrid {
            room,
            pitch_m,
            n_x,
            n_y,
            pitch_x_m,
            pitch_y_m,
            pixels,
        })
    }

    pub fn room(&self) -> &Room {
        &self.room
    }

    /// The requested (target) pitch.
    pub fn pitch_m(&self) -> f64 {
        self.pitch_m
    }

    pub fn n_x(&self) -> usize {
        self.n_x
    }

    pub fn n_y(&self) -> usize {
        self.n_y
    }

    /// Realised pitch along x, `width_m / n_x`.
    pub fn pitch_x_m(&self) -> f64 {
        self.pitch_x_m
    }

    /// Realised pitch along y, `height_m / n_y`.
    pub fn pitch_y_m(&self) -> f64 {
        self.pitch_y_m
    }

    /// Pixel centres in canonical order.
    pub fn pixels(&self) -> &[(f64, f64)] {
        &self.pixels
    }

    /// `len(grid)`.
    pub fn len(&self) -> usize {
        self.pixels.len()
    }

    /// Never true for a grid built by [`PixelGrid::new`].
    pub fn is_empty(&self) -> bool {
        self.pixels.is_empty()
    }

    /// `pitch_x_m * pitch_y_m`.
    pub fn pixel_area_m2(&self) -> f64 {
        self.pitch_x_m * self.pitch_y_m
    }

    /// `iy * n_x + ix`; `IndexError` outside the grid.
    pub fn pixel_index(&self, ix: i64, iy: i64) -> Result<usize> {
        let (nx, ny) = (self.n_x as i64, self.n_y as i64);
        if !(0 <= ix && ix < nx && 0 <= iy && iy < ny) {
            return Err(LocateError::IndexError(format!(
                "({ix}, {iy}) outside {} x {} grid",
                self.n_x, self.n_y
            )));
        }
        Ok((iy * nx + ix) as usize)
    }

    /// Index of the pixel containing `(x, y)`, clamped to the grid:
    /// `min(max(int(x / pitch_x), 0), n_x - 1)`, likewise y. `int` raises
    /// on a NaN (`ValueError`) or infinite (`OverflowError`) coordinate.
    pub fn nearest_index(&self, x: f64, y: f64) -> Result<usize> {
        let ix = py_int(py_div(x, self.pitch_x_m)?)?
            .max(0)
            .min(self.n_x as i64 - 1);
        let iy = py_int(py_div(y, self.pitch_y_m)?)?
            .max(0)
            .min(self.n_y as i64 - 1);
        Ok((iy * self.n_x as i64 + ix) as usize)
    }
}

/// The weights and supports, computed once on first use (`_build`).
#[derive(Debug, Clone)]
struct Built {
    weights: Vec<Vec<f64>>,
    supports: Vec<Vec<usize>>,
}

/// The complete deployment: room, nodes, links (canonical order), grid,
/// lambda. Immutable once built; [`Geometry::with_lambda`] makes a copy
/// with another ellipse width.
#[derive(Debug, Clone)]
pub struct Geometry {
    room: Room,
    nodes: Vec<Node>,
    grid: PixelGrid,
    lambda_m: f64,
    links: Vec<Link>,
    node_by_id: HashMap<i64, usize>,
    link_index: HashMap<(i64, i64), usize>,
    built: OnceLock<Built>,
}

impl Geometry {
    /// `Geometry(room, nodes, grid=None, lambda_m)`. Refuses (`ValueError`)
    /// fewer than 2 nodes, `lambda_m <= 0` (a NaN passes, as in Python),
    /// duplicate ids and two nodes at one position. A `None` grid is
    /// `PixelGrid(room, 0.5)`.
    ///
    /// Positions are compared with `==` (so `0.0` equals `-0.0` and a NaN
    /// coordinate never matches); CPython's `set` would also call two
    /// nodes sharing one NaN *object* equal, which has no Rust analogue.
    pub fn new(
        room: Room,
        nodes: Vec<Node>,
        grid: Option<PixelGrid>,
        lambda_m: f64,
    ) -> Result<Geometry> {
        if nodes.len() < 2 {
            return Err(LocateError::ValueError(format!(
                "need at least 2 nodes, got {}",
                nodes.len()
            )));
        }
        if lambda_m <= 0.0 {
            return Err(LocateError::ValueError(format!(
                "lambda_m must be positive, got {}",
                fmt_py(lambda_m)
            )));
        }

        let mut seen = HashSet::new();
        if !nodes.iter().all(|n| seen.insert(n.node_id)) {
            let mut ids: Vec<i64> = nodes.iter().map(|n| n.node_id).collect();
            ids.sort_unstable();
            return Err(LocateError::ValueError(format!(
                "duplicate node ids: {ids:?}"
            )));
        }

        for (i, a) in nodes.iter().enumerate() {
            for b in &nodes[..i] {
                if a.x == b.x && a.y == b.y {
                    return Err(LocateError::ValueError(
                        "two nodes share a position; links would be degenerate".to_string(),
                    ));
                }
            }
        }

        let grid = match grid {
            Some(g) => g,
            None => PixelGrid::new(room, DEFAULT_PITCH_M)?,
        };

        let node_by_id = nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (n.node_id, i))
            .collect();

        let mut ordered: Vec<&Node> = nodes.iter().collect();
        ordered.sort_by_key(|n| n.node_id);
        let mut links = Vec::new();
        for tx in &ordered {
            for rx in &ordered {
                if tx.node_id == rx.node_id {
                    continue;
                }
                links.push(Link {
                    tx_id: tx.node_id,
                    rx_id: rx.node_id,
                    tx: tx.position(),
                    rx: rx.position(),
                    length_m: tx.distance_to(rx.x, rx.y),
                });
            }
        }
        let link_index = links
            .iter()
            .enumerate()
            .map(|(i, l)| ((l.tx_id, l.rx_id), i))
            .collect();

        Ok(Geometry {
            room,
            nodes,
            grid,
            lambda_m,
            links,
            node_by_id,
            link_index,
            built: OnceLock::new(),
        })
    }

    /// `Geometry.perimeter(n_nodes, room, pitch_m, lambda_m, inset_m,
    /// start_offset_m)`: `perimeter_nodes` first, then the grid, then the
    /// geometry's own checks (Python's evaluation order). `None` room is
    /// `Room()` (8 x 6 m).
    pub fn perimeter(
        n_nodes: i64,
        room: Option<Room>,
        pitch_m: f64,
        lambda_m: f64,
        inset_m: f64,
        start_offset_m: f64,
    ) -> Result<Geometry> {
        let room = room.unwrap_or_default();
        let nodes = perimeter_nodes(&room, n_nodes, inset_m, start_offset_m)?;
        let grid = PixelGrid::new(room, pitch_m)?;
        Geometry::new(room, nodes, Some(grid), lambda_m)
    }

    /// `Geometry.perimeter(n_nodes)` with every other argument at its
    /// default: 8 x 6 m, pitch 0.5, lambda 0.3, inset 0.1, offset 0.
    pub fn perimeter_n(n_nodes: i64) -> Result<Geometry> {
        Geometry::perimeter(
            n_nodes,
            None,
            DEFAULT_PITCH_M,
            DEFAULT_LAMBDA_M,
            DEFAULT_INSET_M,
            0.0,
        )
    }

    /// A copy with a different ellipse width, sharing room, nodes and grid.
    pub fn with_lambda(&self, lambda_m: f64) -> Result<Geometry> {
        Geometry::new(
            self.room,
            self.nodes.clone(),
            Some(self.grid.clone()),
            lambda_m,
        )
    }

    pub fn room(&self) -> &Room {
        &self.room
    }

    /// The nodes in the order given.
    pub fn nodes(&self) -> &[Node] {
        &self.nodes
    }

    pub fn grid(&self) -> &PixelGrid {
        &self.grid
    }

    pub fn lambda_m(&self) -> f64 {
        self.lambda_m
    }

    /// Links in canonical order.
    pub fn links(&self) -> &[Link] {
        &self.links
    }

    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }

    pub fn n_links(&self) -> usize {
        self.links.len()
    }

    pub fn n_pixels(&self) -> usize {
        self.grid.pixels.len()
    }

    /// The node with this id; `KeyError` if none.
    pub fn node(&self, node_id: i64) -> Result<&Node> {
        self.node_by_id
            .get(&node_id)
            .map(|&i| &self.nodes[i])
            .ok_or_else(|| LocateError::KeyError(format!("no node with id {node_id}")))
    }

    /// Position of link `(tx, rx)` in the canonical order; `KeyError` if none.
    pub fn link_index(&self, tx_id: i64, rx_id: i64) -> Result<usize> {
        self.link_index
            .get(&(tx_id, rx_id))
            .copied()
            .ok_or_else(|| LocateError::KeyError(format!("no link ({tx_id} -> {rx_id})")))
    }

    fn build(&self) -> &Built {
        self.built.get_or_init(|| {
            let pixels = &self.grid.pixels;
            let mut dist: HashMap<i64, Vec<f64>> = HashMap::new();
            for node in &self.nodes {
                let (nx, ny) = (node.x, node.y);
                dist.insert(
                    node.node_id,
                    pixels
                        .iter()
                        .map(|&(px, py)| py_hypot(nx - px, ny - py))
                        .collect(),
                );
            }
            let lam = self.lambda_m;
            let mut weights = Vec::with_capacity(self.links.len());
            let mut supports = Vec::with_capacity(self.links.len());
            for link in &self.links {
                let d_tx = &dist[&link.tx_id];
                let d_rx = &dist[&link.rx_id];
                let d_l = link.length_m;
                let w_value = if d_l > 0.0 { 1.0 / d_l.sqrt() } else { 0.0 };
                let mut row = vec![0.0; pixels.len()];
                let mut support = Vec::new();
                for p in 0..pixels.len() {
                    if d_tx[p] + d_rx[p] - d_l < lam {
                        row[p] = w_value;
                        support.push(p);
                    }
                }
                weights.push(row);
                supports.push(support);
            }
            Built { weights, supports }
        })
    }

    /// W as `n_links` rows of `n_pixels` floats, canonical order (cached).
    pub fn weight_matrix(&self) -> &[Vec<f64>] {
        &self.build().weights
    }

    /// Per link, the ascending pixel indices inside its ellipse (cached).
    pub fn supports(&self) -> &[Vec<usize>] {
        &self.build().supports
    }

    /// How many links cover each pixel, canonical pixel order.
    pub fn pixel_link_counts(&self) -> Vec<usize> {
        let mut counts = vec![0usize; self.n_pixels()];
        for support in self.supports() {
            for &p in support {
                counts[p] += 1;
            }
        }
        counts
    }
}
