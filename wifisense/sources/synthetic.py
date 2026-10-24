"""Physically-motivated synthetic CSI.

Lets you exercise and validate the whole pipeline before any hardware
arrives, and — more usefully — lets you write tests with a known ground
truth, which is what any headline accuracy number needs and what
real-world recordings rarely have.

The model: each subcarrier sees a large static multipath sum plus a
smaller body-reflected component whose path length is modulated by chest
wall displacement. Phase of the body component advances by 4*pi*d/lambda
(round trip), and the measured amplitude is the magnitude of the vector
sum. That interference between static and body paths is the real
mechanism behind WiFi vital-sign sensing, and it reproduces the
awkward parts too: fading nulls, per-subcarrier sensitivity differences,
and a heartbeat an order of magnitude weaker than the breath.

Two scenes are available:

``scene="random"`` (the default, unchanged since the first version) draws
each subcarrier's static phase independently. That gives per-subcarrier
sensitivity differences, but they are i.i.d.: no physical position puts
the *whole band* in a blind spot, and every seed is effectively the same
kind of room.

``scene="geometric"`` builds the channel from actual geometry: a
line-of-sight path, first-order wall reflections (image method), optional
furniture scatterers, and a body reflection at a given position. The
phase between the static vector and the body path,

    theta_k = 2*pi*f_k*(d_body - d_static)/c + const,

is then *coherent across subcarriers*. Where theta sits near 0 or pi the
amplitude is a stationary point of the breathing displacement: the
first-order response vanishes, and what survives is the second-order
term at **twice** the breathing rate. Those are the Fresnel-zone
boundaries ("blind spots") that RespirFi (Wang et al., IEEE IoT-J 2026)
is designed to defeat by picking subcarriers. Over the ~17.5 MHz an
ESP32 HT20 channel spans, theta only moves by 2*pi*B*excess/c across the
band, so without rich static multipath a blind spot covers nearly every
subcarrier at once. That is the physics the evaluation has to face.
"""

from __future__ import annotations

import cmath
import math
import random
from dataclasses import dataclass

from ..types import CsiFrame

__all__ = ["SyntheticConfig", "SyntheticCsiSource", "SPEED_OF_LIGHT"]

SPEED_OF_LIGHT = 299_792_458.0

Point = tuple[float, float]
Point3 = tuple[float, float, float]


@dataclass
class SyntheticConfig:
    """Scene parameters. Defaults describe one still adult at rest."""

    n_subcarriers: int = 56
    sample_rate_hz: float = 20.0
    carrier_hz: float = 2.437e9           # 2.4 GHz channel 6
    subcarrier_spacing_hz: float = 312_500.0

    breathing_bpm: float = 15.0
    breathing_amplitude_m: float = 0.005  # 5 mm chest wall displacement
    heart_bpm: float = 68.0
    heart_amplitude_m: float = 0.0004     # 0.4 mm, ~12x weaker than breathing

    static_magnitude: float = 40.0
    body_magnitude: float = 6.0           # body path is ~15% of static
    noise_std: float = 0.6

    occupied: bool = True
    seed: int | None = 1234

    # --- Geometric scene ------------------------------------------------
    # Ignored unless scene == "geometric". Coordinates are metres in a 2-D
    # room [0, room_m[0]] x [0, room_m[1]] (a horizontal slice at chest
    # height); tx, rx and body must lie strictly inside it.
    scene: str = "random"
    room_m: Point = (6.0, 5.0)
    tx_xy_m: Point = (1.5, 2.5)
    rx_xy_m: Point = (4.5, 2.5)
    body_xy_m: Point = (3.0, 3.5)
    # Bistatic radar cross-section of a seated torso at 2.4 GHz. 0.3 m^2
    # makes the body path ~14% of line of sight at the default geometry,
    # matching the random scene's body_magnitude / static_magnitude.
    body_rcs_m2: float = 0.3
    # A torso is not a point. With body_points > 1 the RCS is split over
    # that many scatterers spread across body_width_m, perpendicular to the
    # tx-body-rx bisector. Fresnel zones are only ~6 cm apart in excess
    # path, so a point body flips between blind and sensitive every few
    # centimetres; an extended one averages over that and is more honest.
    body_points: int = 1
    body_width_m: float = 0.0
    # Amplitude reflection coefficient of the walls (drywall/brick at
    # 2.4 GHz is roughly 0.3-0.5). Applied with a pi phase flip.
    wall_reflection: float = 0.4
    # Image-method order. 1 = single bounces; 2 adds double bounces. Real
    # rooms sit between a clear-LoS link (Rician K ~ 8 dB, what order 1
    # with Gamma=0.4 gives here) and NLoS (K -> -inf).
    reflection_order: int = 1
    # Extra loss on the direct path (a partition, a cabinet, people). The
    # richer the multipath relative to LoS, the more the static phase
    # rotates across the band, and only then can subcarriers disagree in
    # sign -- the case subcarrier selection exists for.
    los_attenuation_db: float = 0.0
    # Furniture: point scatterers at seeded random positions, each with an
    # RCS drawn from scatterer_rcs_m2. They are what give a real room
    # per-subcarrier diversity in the static phase.
    n_scatterers: int = 0
    scatterer_rcs_m2: Point = (0.05, 0.5)

    # --- Height: the 3-D box ----------------------------------------------
    # None keeps the 2-D slice above. A height makes the room the box
    # [0, room_m[0]] x [0, room_m[1]] x [0, room_height_m], z up, the axes
    # scenes/*.json use: everything below then applies, and tx, rx, body
    # and every torso point must lie strictly inside the box.
    room_height_m: float | None = None
    # Every scene in scenes/ mounts its nodes at 1.2 m.
    tx_z_m: float = 1.2
    rx_z_m: float = 1.2
    # Chest height: ~1.0 m seated, ~1.3 m standing, ~0.6 m lying in bed.
    # The chest breathes horizontally, along the horizontal part of the
    # tx-chest-rx bisector; a torso spreads horizontally at this height.
    body_z_m: float = 1.0
    # |Gamma| of the floor and ceiling, applied with a pi phase flip like
    # the walls'. 0.395 is concrete at 2.437 GHz and normal incidence
    # (ITU-R P.2040, scenes/materials.json); a plasterboard ceiling is
    # 0.247, a wooden floor 0.171. Like wall_reflection it ignores the
    # angle of incidence, which for a floor bounce is far from normal.
    floor_reflection: float = 0.395
    ceiling_reflection: float = 0.395
    # Furniture heights, drawn after every 2-D draw so one seed puts the
    # same furniture footprint in both models. A guess, not a survey: a
    # seat to the top of a shelf.
    scatterer_z_m: Point = (0.3, 1.8)

    @property
    def breathing_hz(self) -> float:
        return self.breathing_bpm / 60.0

    @property
    def heart_hz(self) -> float:
        return self.heart_bpm / 60.0


def _dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _inside(p: Point, room: Point) -> bool:
    return 0.0 < p[0] < room[0] and 0.0 < p[1] < room[1]


def _dist3(a: Point3, b: Point3) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _inside3(p: Point3, box: Point3) -> bool:
    return all(0.0 < c < size for c, size in zip(p, box))


def _axis_images(x: float, length: float, n: int) -> tuple[float, int, int]:
    """Image n of x across the walls at 0 and `length`, and how many
    bounces it takes off each: (position, at 0, at length).

    Image n sits at n*length + x for even n and n*length + length - x for
    odd n; it is |n| bounces, alternating between the two walls and
    starting with the wall at `length` for n > 0.
    """
    pos = n * length + (x if n % 2 == 0 else length - x)
    m = abs(n)
    return (pos, m // 2, (m + 1) // 2) if n >= 0 else (pos, (m + 1) // 2, m // 2)


def _box_images(
    p: Point3, box: Point3, order: int, wall: float, floor: float, ceiling: float
) -> list[tuple[Point3, float]]:
    """Every image of p in a rectangular box with 1 to `order` bounces, each
    exactly once, with the product of -Gamma over its bounces.

    The images of a box are the lattice (nx, ny, nz) with one axis's images
    per index, and |nx| + |ny| + |nz| bounces. Enumerating the lattice
    rather than mirroring breadth-first is what makes each path appear
    once: a corner reflection off two walls is one path whichever wall the
    search mirrors in first. Order: by bounces, then nx, ny, nz ascending.
    A port must sum in this order to reproduce the golden vectors to the
    last bits rather than to rounding.
    """
    out: list[tuple[Point3, float]] = []
    for total in range(1, order + 1):
        for nx in range(-total, total + 1):
            rest = total - abs(nx)
            for ny in range(-rest, rest + 1):
                rz = rest - abs(ny)
                for nz in (0,) if rz == 0 else (-rz, rz):
                    x, _, _ = _axis_images(p[0], box[0], nx)
                    y, _, _ = _axis_images(p[1], box[1], ny)
                    z, on_floor, on_ceiling = _axis_images(p[2], box[2], nz)
                    gain = (-wall) ** (abs(nx) + abs(ny)) * (-floor) ** on_floor * (-ceiling) ** on_ceiling
                    out.append(((x, y, z), gain))
    return out


class SyntheticCsiSource:
    """Generates `CsiFrame`s on demand."""

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self._rng = random.Random(self.config.seed)
        self._index = 0

        cfg = self.config
        # Per-subcarrier static vector and body-path vector *at rest*.
        self._static: list[complex] = []
        self._body: list[complex] = []
        # Path-length change per metre of chest displacement. A round trip
        # in the random scene (the original model); 2*cos(beta/2) for a
        # bistatic geometry with bistatic angle beta.
        self._path_gain = 2.0
        self._body_points: list[list[complex]] = []
        self._body_gains: list[float] = []
        self.excess_path_m: float | None = None
        # (x, y), or (x, y, z) in the box.
        self.scatterers: list[tuple[float, ...]] = []
        self.scatterer_rcs: list[float] = []
        # Geometric scenes: (amplitude, length) of every static path, line of
        # sight first, before scaling; and where each torso point sits.
        self.static_paths: list[tuple[float, float]] = []
        self.body_point_positions: list[tuple[float, ...]] = []

        if cfg.scene == "random":
            self._init_random()
        elif cfg.scene == "geometric":
            self._init_geometric()
        else:
            raise ValueError(f"unknown scene {cfg.scene!r}; use 'random' or 'geometric'")

    # -- scene construction ----------------------------------------------

    def _init_random(self) -> None:
        """The original model, preserved draw-for-draw.

        Per-subcarrier static multipath: random complex, plus a slow
        frequency-dependent ripple so adjacent subcarriers correlate the
        way real ones do rather than being independent noise.
        """
        cfg = self.config
        for k in range(cfg.n_subcarriers):
            ripple = 1.0 + 0.3 * math.sin(2.0 * math.pi * k / 18.0)
            phase = self._rng.uniform(-math.pi, math.pi)
            mag = cfg.static_magnitude * ripple * self._rng.uniform(0.75, 1.25)
            self._static.append(cmath.rect(mag, phase))
            self._body.append(complex(cfg.body_magnitude * self._rng.uniform(0.6, 1.4), 0.0))

    def _init_geometric(self) -> None:
        """Ray model: LoS + 4 first-order wall images + scatterers + body.

        Each path contributes `gain * exp(-j*2*pi*f_k*length/c)`. Point
        scatterers (furniture, the body) use the bistatic amplitude
        sqrt(rcs / 4pi) / (d1 * d2), relative to 1/d for line of sight.
        Everything is scaled so line of sight has `static_magnitude`,
        which keeps amplitudes in the int8 range the ESP32 reports.
        """
        cfg = self.config
        if cfg.room_height_m is not None:
            self._init_box()
            return
        room, tx, rx, body = cfg.room_m, cfg.tx_xy_m, cfg.rx_xy_m, cfg.body_xy_m
        for name, p in (("tx_xy_m", tx), ("rx_xy_m", rx), ("body_xy_m", body)):
            if not _inside(p, room):
                raise ValueError(f"{name}={p} is not inside room {room}")

        d_los = _dist(tx, rx)
        if d_los <= 0.0:
            raise ValueError("tx and rx must not coincide")
        # Scale so the *unattenuated* LoS would have static_magnitude.
        scale = cfg.static_magnitude * d_los

        # (amplitude, path length) for every static path.
        los_gain = 10.0 ** (-cfg.los_attenuation_db / 20.0)
        paths: list[tuple[float, float]] = [(los_gain / d_los, d_los)]

        # Image method: mirror tx in the walls. In a rectangular room every
        # image of every order is a valid specular path, so no visibility
        # test is needed. Each bounce costs Gamma and flips the phase.
        w, h = room

        def mirrors(p: Point) -> list[tuple[Point, int]]:
            return [((-p[0], p[1]), 0), ((2.0 * w - p[0], p[1]), 1), ((p[0], -p[1]), 2), ((p[0], 2.0 * h - p[1]), 3)]

        frontier: list[tuple[Point, int]] = [(tx, -1)]
        for order in range(1, max(0, cfg.reflection_order) + 1):
            nxt = []
            for src_pt, last_wall in frontier:
                for image, wall in mirrors(src_pt):
                    if wall == last_wall:
                        continue  # mirroring straight back is the source itself
                    length = _dist(image, rx)
                    paths.append(((-cfg.wall_reflection) ** order / length, length))
                    nxt.append((image, wall))
            frontier = nxt

        lo, hi = cfg.scatterer_rcs_m2
        for _ in range(cfg.n_scatterers):
            p = (self._rng.uniform(0.05 * w, 0.95 * w), self._rng.uniform(0.05 * h, 0.95 * h))
            rcs = self._rng.uniform(lo, hi)
            d1, d2 = _dist(tx, p), _dist(p, rx)
            self.scatterers.append(p)
            self.scatterer_rcs.append(rcs)
            paths.append((math.sqrt(rcs / (4.0 * math.pi)) / (d1 * d2), d1 + d2))

        d1, d2 = _dist(tx, body), _dist(body, rx)
        self.excess_path_m = d1 + d2 - d_los
        bisector = self._bisector(tx, rx, body)
        across = (-bisector[1], bisector[0])
        n_pts = max(1, cfg.body_points)
        # (amplitude, path length, path change per metre of displacement)
        body_paths: list[tuple[float, float, float]] = []
        for i in range(n_pts):
            off = 0.0 if n_pts == 1 else cfg.body_width_m * (i / (n_pts - 1) - 0.5)
            p = (body[0] + off * across[0], body[1] + off * across[1])
            if not _inside(p, room):
                raise ValueError(f"body point {p} falls outside room {room}")
            e1, e2 = _dist(tx, p), _dist(p, rx)
            amp = math.sqrt(cfg.body_rcs_m2 / n_pts / (4.0 * math.pi)) / (e1 * e2)
            body_paths.append((amp, e1 + e2, self._path_change(tx, rx, p)))
            self.body_point_positions.append(p)
        self._assemble(scale, paths, body_paths)

    def _init_box(self) -> None:
        """The geometric scene in a box: the 2-D model plus height.

        Line of sight, image-method reflections off walls, floor and
        ceiling (`_box_images`), furniture and the torso, all with 3-D
        distances and the same amplitude laws as the 2-D model. The chest
        moves horizontally along the horizontal part of the tx-chest-rx
        bisector (`_chest_motion`). With every height equal and the floor
        and ceiling absorbing, this is the 2-D model, except that a corner
        reflection is counted once (the 2-D search reaches each of the four
        order-2 corner images twice and adds both).
        """
        cfg = self.config
        height = cfg.room_height_m
        if height is None or not height > 0.0:
            raise ValueError(f"room_height_m={height} must be a positive height")
        size: Point3 = (cfg.room_m[0], cfg.room_m[1], height)
        tx: Point3 = (cfg.tx_xy_m[0], cfg.tx_xy_m[1], cfg.tx_z_m)
        rx: Point3 = (cfg.rx_xy_m[0], cfg.rx_xy_m[1], cfg.rx_z_m)
        body: Point3 = (cfg.body_xy_m[0], cfg.body_xy_m[1], cfg.body_z_m)
        for name, p in (("tx", tx), ("rx", rx), ("body", body)):
            if not _inside3(p, size):
                raise ValueError(f"{name} {p} is not inside the room {size}")

        d_los = _dist3(tx, rx)
        if d_los <= 0.0:
            raise ValueError("tx and rx must not coincide")
        scale = cfg.static_magnitude * d_los

        los_gain = 10.0 ** (-cfg.los_attenuation_db / 20.0)
        paths: list[tuple[float, float]] = [(los_gain / d_los, d_los)]
        for image, gain in _box_images(
            tx, size, max(0, cfg.reflection_order), cfg.wall_reflection, cfg.floor_reflection, cfg.ceiling_reflection
        ):
            length = _dist3(image, rx)
            paths.append((gain / length, length))

        w, h = cfg.room_m
        lo, hi = cfg.scatterer_rcs_m2
        footprint = []
        for _ in range(cfg.n_scatterers):
            xy = (self._rng.uniform(0.05 * w, 0.95 * w), self._rng.uniform(0.05 * h, 0.95 * h))
            footprint.append((xy, self._rng.uniform(lo, hi)))
        zlo, zhi = cfg.scatterer_z_m
        for xy, rcs in footprint:
            p = (xy[0], xy[1], self._rng.uniform(zlo, zhi))
            d1, d2 = _dist3(tx, p), _dist3(p, rx)
            self.scatterers.append(p)
            self.scatterer_rcs.append(rcs)
            paths.append((math.sqrt(rcs / (4.0 * math.pi)) / (d1 * d2), d1 + d2))

        self.excess_path_m = _dist3(tx, body) + _dist3(body, rx) - d_los
        direction, _ = self._chest_motion(tx, rx, body)
        across = (-direction[1], direction[0])
        n_pts = max(1, cfg.body_points)
        body_paths: list[tuple[float, float, float]] = []
        for i in range(n_pts):
            off = 0.0 if n_pts == 1 else cfg.body_width_m * (i / (n_pts - 1) - 0.5)
            p = (body[0] + off * across[0], body[1] + off * across[1], body[2])
            if not _inside3(p, size):
                raise ValueError(f"body point {p} falls outside the room {size}")
            e1, e2 = _dist3(tx, p), _dist3(p, rx)
            amp = math.sqrt(cfg.body_rcs_m2 / n_pts / (4.0 * math.pi)) / (e1 * e2)
            body_paths.append((amp, e1 + e2, self._chest_motion(tx, rx, p)[1]))
            self.body_point_positions.append(p)
        self._assemble(scale, paths, body_paths)

    def _assemble(
        self, scale: float, paths: list[tuple[float, float]], body_paths: list[tuple[float, float, float]]
    ) -> None:
        """Per-subcarrier static and body vectors from the path lists."""
        cfg = self.config
        self.static_paths = paths
        self._body_gains = [g for _, _, g in body_paths]
        # Kept for the single-point fast path and for sensitivity().
        self._path_gain = body_paths[0][2]

        for k in range(cfg.n_subcarriers):
            f = self._frequency(k)
            static = sum(
                (amp * cmath.exp(-2j * math.pi * f * length / SPEED_OF_LIGHT) for amp, length in paths),
                0j,
            )
            self._static.append(scale * static)
            self._body_points.append(
                [scale * amp * cmath.exp(-2j * math.pi * f * length / SPEED_OF_LIGHT) for amp, length, _ in body_paths]
            )
            self._body.append(sum(self._body_points[-1], 0j))

    @staticmethod
    def _bisector(tx: Point, rx: Point, p: Point) -> Point:
        """Unit vector bisecting the tx-p-rx angle (the chest normal)."""
        d1, d2 = _dist(tx, p), _dist(p, rx)
        bx = (tx[0] - p[0]) / d1 + (rx[0] - p[0]) / d2
        by = (tx[1] - p[1]) / d1 + (rx[1] - p[1]) / d2
        norm = math.hypot(bx, by)
        if norm == 0.0:  # p on the LoS segment: bistatic angle is pi
            ux, uy = (rx[0] - tx[0]) / _dist(tx, rx), (rx[1] - tx[1]) / _dist(tx, rx)
            return (-uy, ux)
        return (bx / norm, by / norm)

    @staticmethod
    def _path_change(tx: Point, rx: Point, p: Point) -> float:
        """Path-length change per metre of chest motion along the bisector.

        2*cos(beta/2) for bistatic angle beta: the Fresnel-zone
        generalisation of the monostatic round-trip factor 2. It falls to
        zero as the body approaches the LoS line, where moving the chest
        barely changes the reflected path at all.
        """
        d1, d2 = _dist(tx, p), _dist(p, rx)
        u1 = ((tx[0] - p[0]) / d1, (tx[1] - p[1]) / d1)
        u2 = ((rx[0] - p[0]) / d2, (rx[1] - p[1]) / d2)
        cos_beta = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
        return 2.0 * math.cos(math.acos(cos_beta) / 2.0)

    @staticmethod
    def _chest_motion(tx: Point3, rx: Point3, p: Point3) -> tuple[Point, float]:
        """Which way a chest at p breathes, and the path change per metre.

        A chest moves horizontally whatever the link's height. Moving it
        along a horizontal unit vector n shortens tx-p-rx by
        (u_tx + u_rx) . n per metre, with u the unit vectors from p to
        each end; that is largest along the horizontal part of u_tx + u_rx,
        and equals its length there. Level with both ends this is the 2-D
        model's 2*cos(beta/2). A chest straight below the link's midpoint
        has no horizontal part at all and changes the path only to second
        order; its direction is then taken across the link.
        """
        d1, d2 = _dist3(tx, p), _dist3(p, rx)
        sx = (tx[0] - p[0]) / d1 + (rx[0] - p[0]) / d2
        sy = (tx[1] - p[1]) / d1 + (rx[1] - p[1]) / d2
        norm = math.hypot(sx, sy)
        if norm < 1e-12:
            hx, hy = rx[0] - tx[0], rx[1] - tx[1]
            hn = math.hypot(hx, hy)
            return ((-hy / hn, hx / hn) if hn > 0.0 else (1.0, 0.0)), 0.0
        return (sx / norm, sy / norm), norm

    # -- physics helpers --------------------------------------------------

    def _frequency(self, k: int) -> float:
        cfg = self.config
        return cfg.carrier_hz + (k - cfg.n_subcarriers / 2.0) * cfg.subcarrier_spacing_hz

    def _wavelength(self, k: int) -> float:
        return SPEED_OF_LIGHT / self._frequency(k)

    def sensitivity(self) -> list[float]:
        """Per-subcarrier first-order breathing sensitivity, in [0, 1].

        d|h|/dx at rest, normalised by its largest possible value: 1 means
        the chest's path change lands fully on the amplitude, 0 means the
        subcarrier sits on a Fresnel-zone boundary and only the
        second-order term survives, at *twice* the breathing rate. For a
        single body point this is |sin| of the angle between h and the
        body vector. Ground truth for the evaluation; a receiver cannot
        measure it.
        """
        out = []
        for k, (s, b) in enumerate(zip(self._static, self._body)):
            h0 = s + b
            pts = self._body_points[k] if self._body_points else [b]
            gains = self._body_gains if self._body_gains else [self._path_gain]
            # dh/dx up to the common factor 2*pi/lambda.
            dh = sum((1j * g * bp for bp, g in zip(pts, gains)), 0j)
            scale = abs(h0) * sum(abs(bp) * g for bp, g in zip(pts, gains))
            out.append(0.0 if scale == 0.0 else abs((h0.conjugate() * dh).real) / scale)
        return out

    # -- generation -------------------------------------------------------

    def next_frame(self) -> CsiFrame:
        """Produce the next frame and advance time by one sample."""
        cfg = self.config
        t = self._index / cfg.sample_rate_hz

        if cfg.occupied:
            displacement = cfg.breathing_amplitude_m * math.sin(
                2.0 * math.pi * cfg.breathing_hz * t
            ) + cfg.heart_amplitude_m * math.sin(2.0 * math.pi * cfg.heart_hz * t)
        else:
            displacement = 0.0

        amplitudes: list[float] = []
        phases: list[float] = []
        for k in range(cfg.n_subcarriers):
            if cfg.occupied and self._body_points:
                # Extended body: every scatter point moves with the chest.
                k_wave = 2.0 * math.pi / self._wavelength(k)
                h = self._static[k] + sum(
                    (bp * cmath.rect(1.0, k_wave * g * displacement) for bp, g in zip(self._body_points[k], self._body_gains)),
                    0j,
                )
            elif cfg.occupied:
                # Path change -> 2*pi*(path_gain*d)/lambda of phase; the
                # random scene uses path_gain = 2 (round trip).
                body_phase = 2.0 * math.pi * self._path_gain * displacement / self._wavelength(k)
                h = self._static[k] + self._body[k] * cmath.rect(1.0, body_phase)
            else:
                h = self._static[k]

            h += complex(
                self._rng.gauss(0.0, cfg.noise_std),
                self._rng.gauss(0.0, cfg.noise_std),
            )
            amplitudes.append(abs(h))
            phases.append(cmath.phase(h))

        frame = CsiFrame(
            amplitudes=amplitudes,
            phases=phases,
            sample_index=self._index,
            sample_rate_hz=cfg.sample_rate_hz,
            rssi_dbm=-45,
            node_id=0,
        )
        self._index += 1
        return frame

    def frames(self, count: int) -> list[CsiFrame]:
        return [self.next_frame() for _ in range(count)]

    def seconds(self, duration_s: float) -> list[CsiFrame]:
        return self.frames(int(duration_s * self.config.sample_rate_hz))

    def reset(self) -> None:
        self._index = 0
