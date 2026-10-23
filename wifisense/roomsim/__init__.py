"""Room Sim v1 — the Python reference (spec/roomsim.md).

Reads a room from a scene file (SCENE, ADR-008) and computes what ESP32
nodes in that room would report: ITU-R P.2040 wall, floor and ceiling
slabs to second order, furniture as boxes, clutter and a diffuse tail,
people as moving point clouds with cylinder shadowing, and the ESP32's
buffer layout, AGC, noise and quantisation.

Everything it produces is **SIM** — a physically motivated model, not a
measurement; read §16 of the spec before drawing a conclusion from it.
Room geometry and furniture come from the scene file, never from WiFi.
Standard library only; the Rust crate `saans-roomsim` must agree with it
to ~1e-9 (§14), so every formula is evaluated exactly as the spec writes
it.
"""

from .config import Config, ConfigError
from .export import export_run, manifest
from .impair import BUFFER_ACTIVE, BUFFER_K
from .rng import Rng
from .scene import Scene, SceneError, load_scene, parse_scene
from .sim import LABEL, Packet, PersonTruth, RoomSim, Transmission, Truth

__all__ = [
    "Config", "ConfigError", "Rng", "Scene", "SceneError", "load_scene", "parse_scene",
    "RoomSim", "Packet", "Truth", "PersonTruth", "Transmission", "LABEL",
    "BUFFER_K", "BUFFER_ACTIVE", "export_run", "manifest",
]
