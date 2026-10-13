"""Device-free localization by radio tomographic imaging.

RESEARCH.md §0 rules out every bearing- or range-based route on this
hardware: an ESP32 has one RF chain, separate nodes do not form a
coherent array, and linear-fit phase sanitization destroys the absolute
delay that multilateration would need. What N single-antenna nodes *can*
do is notice which of the N*(N-1) links between them a body is standing
in, and invert that into an image. This package is that inversion, built
up in layers:

    geometry  — room, nodes, links, pixel grid, ellipse weight model
    sim       — multi-link synthetic CSI source  (not yet written)
    measure   — per-link variance change vector  (not yet written)
    invert    — regularized least squares        (not yet written)
    track     — peak extraction and Kalman filter (not yet written)

Only `geometry` exists so far; the rest land beside it and are not
imported here until they do.
"""

from __future__ import annotations

from .geometry import (
    Geometry,
    Link,
    Node,
    PixelGrid,
    Room,
    ellipse_weight,
    perimeter_nodes,
)

__all__ = [
    "Geometry",
    "Link",
    "Node",
    "PixelGrid",
    "Room",
    "ellipse_weight",
    "perimeter_nodes",
]
