"""Room geometry, link topology, and the RTI ellipse weight model.

Radio tomographic imaging turns "which links were disturbed" into "where
the person is". That inversion is only ever as good as the forward model
saying which pixels a link can see, and this module is that forward model
— plus the bookkeeping every other `localize` module indexes against: the
canonical link order and the canonical pixel order.

Nothing here touches CSI or time. Keeping the geometry separable matters
because it is the one layer we can check against ground truth without a
radio: distances are distances, and a bug in the perimeter walk or the
link ordering would otherwise show up as a mysterious bias three modules
downstream.

The weight model is the Wilson & Patwari ellipse (IEEE TMC 2010):

    w(l, p) = 1/sqrt(d_l)   if   d(tx, p) + d(p, rx) < d_l + lambda
              0             otherwise

A link is attenuated by any pixel inside the ellipse with foci at its two
nodes, uniformly, and not at all by anything outside it.

Be clear-eyed about what that is: a crude geometric stand-in for Fresnel
zone sensitivity, not a propagation model.

  * The real first Fresnel zone *is* an ellipsoid with foci at the two
    antennas, which is where the shape comes from — but its influence is
    neither uniform inside nor zero outside. It oscillates in sign
    through the higher-order zones.
  * `lambda_m` is a free tuning parameter in METRES. It is not the RF
    wavelength (0.123 m at 2.437 GHz). The resemblance of the symbol is
    an accident of the paper's notation and has caused confusion before.
  * A body standing a metre from the transmitter attenuates a link far
    more than the same body at the midpoint. Uniform weighting inside
    the ellipse throws that away.

The model survives anyway because the inversion is heavily regularized
and because any one pixel is covered by many overlapping ellipses, so the
per-link modelling error partly averages out. Treat `lambda_m` as a knob
to sweep, not a physical constant, and do not read the resulting image as
a map of attenuation.

The 1/sqrt(d_l) factor is the paper's normalisation. A long link's
ellipse contains many more pixels than a short link's, so without it the
long links — which are also the noisiest, having the weakest received
power — would dominate the least-squares fit purely by virtue of being
long.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

__all__ = [
    "Room",
    "Node",
    "Link",
    "PixelGrid",
    "Geometry",
    "perimeter_nodes",
    "ellipse_weight",
]


@dataclass(frozen=True)
class Room:
    """A 2-D rectangular floor plan with its origin at one corner.

    `height_m` is the second horizontal axis of the floor plan, not
    elevation: this is a plan view, and the whole localization stack is
    2-D. Nodes and people are assumed to live at roughly the same height,
    which is the usual RTI simplification and is roughly true for a
    standing adult and nodes mounted at chest height.
    """

    width_m: float = 8.0
    height_m: float = 6.0

    def __post_init__(self) -> None:
        if self.width_m <= 0.0 or self.height_m <= 0.0:
            raise ValueError(
                f"room dimensions must be positive, got "
                f"{self.width_m} x {self.height_m}"
            )

    @property
    def perimeter_m(self) -> float:
        return 2.0 * (self.width_m + self.height_m)

    @property
    def area_m2(self) -> float:
        return self.width_m * self.height_m

    @property
    def center(self) -> tuple[float, float]:
        return (self.width_m / 2.0, self.height_m / 2.0)

    def contains(self, x: float, y: float, tol: float = 0.0) -> bool:
        """Is (x, y) inside the room, within `tol` metres of slack?"""
        return (
            -tol <= x <= self.width_m + tol and -tol <= y <= self.height_m + tol
        )


@dataclass(frozen=True)
class Node:
    """One radio at a fixed, known position.

    Known positions are a hard prerequisite for RTI — see RESEARCH.md §0:
    single-chain ESP32 nodes cannot do angle of arrival and phase
    sanitization destroys absolute ranging, so there is nothing to
    self-localize the mesh with. The coordinates have to be surveyed by
    hand, and any error in them lands directly in the image.
    """

    node_id: int
    x: float
    y: float

    @property
    def position(self) -> tuple[float, float]:
        return (self.x, self.y)

    def distance_to(self, x: float, y: float) -> float:
        return math.hypot(self.x - x, self.y - y)


@dataclass(frozen=True)
class Link:
    """One ordered (tx, rx) pair and its line-of-sight length d_l.

    Ordered, not unordered. The *geometry* of a link is symmetric — see
    `Geometry.weight_matrix` — but the radio is not: tx and rx have
    different noise floors, different antenna orientations and, under
    TDMA, different slots. Keeping both directions as separate links lets
    the measurement layer treat them as the two independent observations
    they actually are, and costs only a factor of two in an L x L solve.
    """

    tx_id: int
    rx_id: int
    tx: tuple[float, float]
    rx: tuple[float, float]
    length_m: float

    @property
    def d_l(self) -> float:
        """Alias matching the paper's notation."""
        return self.length_m

    @property
    def midpoint(self) -> tuple[float, float]:
        return (
            0.5 * (self.tx[0] + self.rx[0]),
            0.5 * (self.tx[1] + self.rx[1]),
        )

    def excess_path_m(self, x: float, y: float) -> float:
        """d(tx, p) + d(p, rx) - d_l: how far a detour via p costs.

        Zero on the line-of-sight segment, positive everywhere else. This
        is the quantity the ellipse test thresholds, and it is also the
        shadowing coordinate the simulator needs, so it lives here rather
        than being recomputed in two places.
        """
        d1 = math.hypot(self.tx[0] - x, self.tx[1] - y)
        d2 = math.hypot(x - self.rx[0], y - self.rx[1])
        return d1 + d2 - self.length_m


def ellipse_weight(link: Link, x: float, y: float, lambda_m: float) -> float:
    """w(l, p) for a single link and pixel centre.

    The comparison is strict (`excess < lambda`), so a pixel exactly on
    the ellipse boundary is OUTSIDE. That choice is arbitrary but it has
    to be made and documented, because a sweep over `lambda_m` otherwise
    produces counts that jitter with the floating-point wind.

    Two floating-point notes:

      * On the line-of-sight segment the excess is exactly zero up to
        about 1e-16 m of cancellation, so any sane `lambda_m` (0.01 m and
        up) always admits it. `lambda_m <= 0` would make that guarantee
        fail and is rejected.
      * The cancellation in `excess_path_m` is at the 1e-16 m level
        against a threshold of order 0.1-1 m. It cannot move a pixel
        across the boundary unless the pixel was already on it to within
        a femtometre.
    """
    if lambda_m <= 0.0:
        raise ValueError(f"lambda_m must be positive, got {lambda_m}")
    if link.length_m <= 0.0:
        return 0.0
    if link.excess_path_m(x, y) < lambda_m:
        return 1.0 / math.sqrt(link.length_m)
    return 0.0


def perimeter_nodes(
    room: Room,
    n_nodes: int,
    inset_m: float = 0.1,
    start_offset_m: float = 0.0,
) -> list[Node]:
    """Place `n_nodes` at equal ARC LENGTH around the room perimeter.

    Equal arc length, not equal x and y. The obvious wrong implementation
    distributes nodes per edge (n/4 each) or interpolates the corners,
    and both give visibly uneven spacing on a non-square room: on 8 x 6
    the long edges are 1.33x the short ones, so per-edge placement
    stretches the spacing by that factor on two sides out of four. Uneven
    spacing biases the image, because link density is what sets the local
    resolution of the reconstruction.

    Nodes sit on an inset rectangle `inset_m` inside the walls, which is
    where they physically are: a radio on a bracket is a few centimetres
    off the plaster, and placing a node exactly on the boundary makes
    every wall-adjacent pixel sit at zero distance from a focus.

    The walk starts at the inset bottom-left corner and goes
    counter-clockwise (+x along the bottom edge first). `start_offset_m`
    rotates the whole pattern along the perimeter; with the default 0.0,
    node 0 lands exactly on that corner.
    """
    if n_nodes < 2:
        raise ValueError(f"need at least 2 nodes to form a link, got {n_nodes}")
    if inset_m < 0.0:
        raise ValueError(f"inset_m must be non-negative, got {inset_m}")
    if 2.0 * inset_m >= min(room.width_m, room.height_m):
        raise ValueError(
            f"inset_m={inset_m} collapses a "
            f"{room.width_m} x {room.height_m} room"
        )

    x0 = y0 = inset_m
    x1 = room.width_m - inset_m
    y1 = room.height_m - inset_m
    w = x1 - x0
    h = y1 - y0
    perimeter = 2.0 * (w + h)
    step = perimeter / n_nodes

    nodes: list[Node] = []
    for i in range(n_nodes):
        s = (start_offset_m + i * step) % perimeter
        if s < w:  # bottom edge, left to right
            pos = (x0 + s, y0)
        elif s < w + h:  # right edge, bottom to top
            pos = (x1, y0 + (s - w))
        elif s < 2.0 * w + h:  # top edge, right to left
            pos = (x1 - (s - w - h), y1)
        else:  # left edge, top to bottom
            pos = (x0, y1 - (s - 2.0 * w - h))
        nodes.append(Node(i, pos[0], pos[1]))
    return nodes


class PixelGrid:
    """A regular grid of square-ish pixels tiling the room.

    `pixels` is the canonical ordered list of (x, y) centres and every
    image in this package is a flat list indexed in that order. Ordering
    is row-major with y outermost, so `pixels[iy * n_x + ix]`.

    `pitch_m` is a TARGET, not a guarantee. The grid is forced to tile the
    room exactly — `n_x = round(width / pitch)` and the realised pitch is
    `width / n_x` — so that pixel area is uniform and every pixel centre
    is inside the room. A grid that merely stepped by `pitch_m` from the
    origin would leave a ragged strip at the far wall whose pixels are
    partly outside, and the inversion has no way to know they are smaller.
    """

    __slots__ = ("room", "pitch_m", "n_x", "n_y", "pitch_x_m", "pitch_y_m", "pixels")

    def __init__(self, room: Room, pitch_m: float = 0.5) -> None:
        if pitch_m <= 0.0:
            raise ValueError(f"pitch_m must be positive, got {pitch_m}")

        self.room = room
        self.pitch_m = pitch_m
        self.n_x = max(1, int(round(room.width_m / pitch_m)))
        self.n_y = max(1, int(round(room.height_m / pitch_m)))
        self.pitch_x_m = room.width_m / self.n_x
        self.pitch_y_m = room.height_m / self.n_y

        self.pixels: list[tuple[float, float]] = [
            ((ix + 0.5) * self.pitch_x_m, (iy + 0.5) * self.pitch_y_m)
            for iy in range(self.n_y)
            for ix in range(self.n_x)
        ]

    def __len__(self) -> int:
        return len(self.pixels)

    @property
    def pixel_area_m2(self) -> float:
        return self.pitch_x_m * self.pitch_y_m

    def pixel_index(self, ix: int, iy: int) -> int:
        if not (0 <= ix < self.n_x and 0 <= iy < self.n_y):
            raise IndexError(f"({ix}, {iy}) outside {self.n_x} x {self.n_y} grid")
        return iy * self.n_x + ix

    def nearest_index(self, x: float, y: float) -> int:
        """Index of the pixel whose centre is closest to (x, y).

        Clamped rather than raising, because callers scoring a tracker
        against ground truth would otherwise have to special-case a person
        standing a centimetre outside the nominal wall.
        """
        ix = min(max(int(x / self.pitch_x_m), 0), self.n_x - 1)
        iy = min(max(int(y / self.pitch_y_m), 0), self.n_y - 1)
        return iy * self.n_x + ix


class Geometry:
    """The complete deployment description: room, nodes, links, grid, lambda.

    One object rather than four arguments everywhere, because every other
    module in this package (`measure`, `invert`, `track`) needs some slice
    of it and the shared interface contract passes `geometry` alone. It
    also makes the two canonical orderings — `links` and `grid.pixels` —
    impossible to get out of step: they are properties of a single object
    that the whole pipeline shares.

    Links are every ordered pair with tx != rx, sorted by (tx_id, rx_id),
    giving N*(N-1) of them. Treat the instance as immutable; to sweep
    `lambda_m`, call `with_lambda`, which shares the room, nodes and grid
    and only recomputes the weights.
    """

    def __init__(
        self,
        room: Room,
        nodes: list[Node],
        grid: PixelGrid | None = None,
        lambda_m: float = 0.3,
    ) -> None:
        if len(nodes) < 2:
            raise ValueError(f"need at least 2 nodes, got {len(nodes)}")
        if lambda_m <= 0.0:
            raise ValueError(f"lambda_m must be positive, got {lambda_m}")

        ids = [n.node_id for n in nodes]
        if len(set(ids)) != len(ids):
            raise ValueError(f"duplicate node ids: {sorted(ids)}")

        positions = [n.position for n in nodes]
        if len(set(positions)) != len(positions):
            # Two nodes at one point give a zero-length link, and 1/sqrt(0)
            # is not a weight. Catch it here rather than as an inf in W.
            raise ValueError("two nodes share a position; links would be degenerate")

        self.room = room
        self.nodes = list(nodes)
        self.grid = PixelGrid(room) if grid is None else grid
        self.lambda_m = lambda_m

        self._node_by_id = {n.node_id: n for n in self.nodes}

        ordered = sorted(self.nodes, key=lambda n: n.node_id)
        self.links: list[Link] = []
        for tx in ordered:
            for rx in ordered:
                if tx.node_id == rx.node_id:
                    continue
                self.links.append(
                    Link(
                        tx_id=tx.node_id,
                        rx_id=rx.node_id,
                        tx=tx.position,
                        rx=rx.position,
                        length_m=tx.distance_to(rx.x, rx.y),
                    )
                )
        self._link_index = {(l.tx_id, l.rx_id): i for i, l in enumerate(self.links)}

        self._weights: list[list[float]] | None = None
        self._supports: list[list[int]] | None = None

    @classmethod
    def perimeter(
        cls,
        n_nodes: int = 8,
        room: Room | None = None,
        pitch_m: float = 0.5,
        lambda_m: float = 0.3,
        inset_m: float = 0.1,
        start_offset_m: float = 0.0,
    ) -> "Geometry":
        """The standard deployment: N nodes evenly around the perimeter."""
        room = Room() if room is None else room
        return cls(
            room=room,
            nodes=perimeter_nodes(room, n_nodes, inset_m, start_offset_m),
            grid=PixelGrid(room, pitch_m),
            lambda_m=lambda_m,
        )

    def with_lambda(self, lambda_m: float) -> "Geometry":
        """A copy with a different ellipse width, for sweeps."""
        return Geometry(self.room, self.nodes, self.grid, lambda_m)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_links(self) -> int:
        return len(self.links)

    @property
    def n_pixels(self) -> int:
        return len(self.grid.pixels)

    def node(self, node_id: int) -> Node:
        try:
            return self._node_by_id[node_id]
        except KeyError:
            raise KeyError(f"no node with id {node_id}") from None

    def link_index(self, tx_id: int, rx_id: int) -> int:
        """Position of link (tx, rx) in the canonical `links` order."""
        try:
            return self._link_index[(tx_id, rx_id)]
        except KeyError:
            raise KeyError(f"no link ({tx_id} -> {rx_id})") from None

    def _build(self) -> None:
        """Compute W and the sparse support lists in one pass.

        Node-to-pixel distances are precomputed because the inner loop
        would otherwise call hypot L*P times (132 * 768 = 101k at 12 nodes
        and a 0.25 m pitch) when N*P = 9.2k distinct distances exist. The
        inner loop then costs one add, one subtract and one compare.

        Every nonzero entry in a row is the same object, 1/sqrt(d_l), and
        every zero entry is the same `0.0` constant, so a dense row costs
        8 bytes per pixel for the list slot rather than 32 for a distinct
        float. See the memory note on `weight_matrix`.
        """
        pixels = self.grid.pixels
        n_pixels = len(pixels)

        dist: dict[int, list[float]] = {}
        for node in self.nodes:
            nx, ny = node.x, node.y
            dist[node.node_id] = [math.hypot(nx - px, ny - py) for px, py in pixels]

        weights: list[list[float]] = []
        supports: list[list[int]] = []
        lam = self.lambda_m
        for link in self.links:
            d_tx = dist[link.tx_id]
            d_rx = dist[link.rx_id]
            d_l = link.length_m
            w_value = 1.0 / math.sqrt(d_l) if d_l > 0.0 else 0.0
            row = [0.0] * n_pixels
            support: list[int] = []
            for p in range(n_pixels):
                if d_tx[p] + d_rx[p] - d_l < lam:
                    row[p] = w_value
                    support.append(p)
            weights.append(row)
            supports.append(support)

        self._weights = weights
        self._supports = supports

    def weight_matrix(self) -> list[list[float]]:
        """W as `n_links` rows of `n_pixels` floats, in canonical order.

        Computed once and cached; the returned list is the live cache, so
        do not mutate it.

        MEMORY, measured on CPython 3.14.7 (`sys.getsizeof` over the outer
        list plus every row), 8 x 6 m room, lambda = 0.3 m:

            nodes  links  pitch   pixels    W      supports   build
              8      56   0.50 m     192    88 KB     20 KB   0.5 ms
              8      56   0.25 m     768   340 KB     65 KB   1.8 ms
             12     132   0.25 m     768   800 KB    150 KB   4.1 ms

        Those are the list slots, which is nearly all of it: every nonzero
        entry in a row is the *same* 1/sqrt(d_l) object and every zero is
        the same `0.0` constant, so a row of 768 pixels costs 8 bytes per
        pixel rather than 32, and the distinct float objects across the
        whole matrix add up to about 3 KB. So the honest figure for the
        0.25 m / 12-node case is ~0.8 MB.

        That is comfortable on a laptop, but W is dense and grows as
        N^2 / pitch^2: a 0.1 m pitch at 12 nodes would be ~5 MB, and the
        ESP32 this code is eventually meant to port to could not hold it
        at all. Only 16-21% of the entries are nonzero at these settings
        — which is what `supports` is for. It carries the same information
        as a per-link list of pixel indices at a quarter of the memory,
        and it is what the dual-form inversion should iterate over.
        """
        if self._weights is None:
            self._build()
        assert self._weights is not None
        return self._weights

    def supports(self) -> list[list[int]]:
        """Sparse form of W: per link, the pixel indices inside its ellipse.

        Weights are uniform within a row, so this plus `1/sqrt(d_l)`
        reconstructs W exactly.
        """
        if self._supports is None:
            self._build()
        assert self._supports is not None
        return self._supports

    def pixel_link_counts(self) -> list[int]:
        """How many links cover each pixel, in canonical pixel order.

        The diagnostic that catches a `lambda_m` chosen too small for the
        pixel pitch. A pixel covered by zero links contributes nothing to
        any measurement and so can never light up — the inversion will
        happily return a smooth, confident, meaningless value there from
        the spatial prior alone. Check `min(...) > 0` before trusting an
        image.

        MEASURED blind spots, 8 x 6 m room, nodes inset 0.1 m, default
        start offset (so node 0 sits on the bottom-left corner):

            nodes  pitch    lambda  pixels with zero coverage
              8    0.50 m   0.30 m   2 of 192  — the two unoccupied corners
              8    0.50 m   0.75 m   0
              8    0.25 m   0.30 m  12 of 768  — same two corners, finer
              8    0.25 m   0.75 m   2 of 768
              8    0.25 m   1.00 m   0
              4    0.50 m   0.30 m  16 of 192  — corners AND interior
              6    0.50 m   0.30 m   6 of 192
             12    0.50 m   0.30 m   2 of 192
             16    0.50 m   0.30 m   0

        Two things to take from that. First, the corner holes are
        structural, not a bad default: nodes are inset, so a room corner
        with no node on it lies outside the convex hull of the nodes and
        no ellipse between two adjacent nodes reaches around it. Adding
        nodes does not fix the corners — N=12 has the same two holes as
        N=8 — only a larger lambda or a node placed there does.

        Second, N=4 is qualitatively different: its 16 blind pixels
        include interior ones, because 12 links simply do not cover an
        8 x 6 room. Expect the N=4 row of any sweep to fail for this
        reason and not for a reason the measurement layer can fix.
        """
        counts = [0] * self.n_pixels
        for support in self.supports():
            for p in support:
                counts[p] += 1
        return counts

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"Geometry(room={self.room.width_m:g}x{self.room.height_m:g} m, "
            f"nodes={self.n_nodes}, links={self.n_links}, "
            f"pixels={self.grid.n_x}x{self.grid.n_y}, "
            f"pitch={self.grid.pitch_x_m:g}x{self.grid.pitch_y_m:g} m, "
            f"lambda={self.lambda_m:g} m)"
        )
