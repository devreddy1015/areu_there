"""Device-free localization of people, and a simulated phone that ranges itself.

RESEARCH.md §0 rules out every bearing- or range-based route to *people*
on this hardware: an ESP32 has one RF chain, separate nodes do not form a
coherent array, and linear-fit phase sanitization destroys the absolute
delay that multilateration would need. What N single-antenna nodes *can*
do is notice which of the N*(N-1) links between them a body is standing
in. People are located only that way: by radio tomographic imaging (the
ported RTI) and by the weighted-link-midpoint baseline beside it.

The RTI, in layers (normative; geometry, measure, invert and track are
ported to Rust as they are; sim is not ported):

    geometry  — room, nodes, links, pixel grid, ellipse weight model
    sim       — multi-link synthetic CSI source (localize/sim.py; SIM;
                Python only, the simulator behind NOTES-localize.md)
    measure   — per-link variance change vector, with the link signals
                (total power, static-suppressed amplitude, `RssiSignal`)
    invert    — regularized least squares (Cholesky, spatial prior)
    track     — peak extraction and Kalman filter (`PositionEstimate`
                carries the filter's covariance)

Locate v1 (spec/locate.md), added beside them:

    radius    — the published radius (a model, `radius_calibrated` False)
                and the estimate/abstention records (§8, §11)
    config    — every §9 parameter and its validation (`LocateConfig`)
    baseline  — the weighted-link-midpoint baseline with a gate and a
                radius (§5)
    phone     — a SIM-only phone (ADR-008 d16) ranging itself to the nodes
                by Wi-Fi RTT and RSSI, its fixes and Kalman filters (§6-§7).
                A cooperative device measuring its own distances; this is
                not the multilateration of people that RESEARCH.md §0 and
                CAMPUS_SENSING_PROMPT.md §3 rule out.
    pipeline  — the glue from Room Sim packets and truth to tick and epoch
                records (§4.3, §10)

Every estimate is a position with a radius, or an abstention with a
reason; never a bare point (ADR-008 d5). Everything Locate produces in
Phase 4 is SIM.
"""

from __future__ import annotations

from .baseline import weighted_link_midpoint
from .config import LocateConfig, LocateError
from .geometry import (
    Geometry,
    Link,
    Node,
    PixelGrid,
    Room,
    ellipse_weight,
    perimeter_nodes,
)
from .phone import PhoneRun, rssi_fix, rtt_fix, run_phone
from .pipeline import LocateRun, PersonPipeline, run_locate
from .radius import radius95

__all__ = [
    # RTI geometry
    "Geometry",
    "Link",
    "Node",
    "PixelGrid",
    "Room",
    "ellipse_weight",
    "perimeter_nodes",
    # Locate v1
    "LocateConfig",
    "LocateError",
    "LocateRun",
    "PersonPipeline",
    "PhoneRun",
    "radius95",
    "rssi_fix",
    "rtt_fix",
    "run_locate",
    "run_phone",
    "weighted_link_midpoint",
]
