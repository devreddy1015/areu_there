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
        self.scatterers: list[Point] = []
        self.scatterer_rcs: list[float] = []

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
