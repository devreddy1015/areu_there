"""Phase 10a: the Sionna RT audit of Room Sim (spec/sionna_audit.md).

Room Sim (spec/roomsim.md, the stdlib reference `wifisense/roomsim/`) and
Sionna RT 2.2.0 are both **simulators**: every number this package produces
is **SIM**, a comparison of two models of the same declared room, never a
property of a real room. Agreement here says the two models agree, nothing
more.

- `scene_to_sionna`: a Room Sim scene -> a Sionna RT scene (Mitsuba XML and
  PLY meshes), the backend selection, and the extraction of Sionna's paths.
- `roomsim_side`: Room Sim's deterministic per-link paths and channel, read
  through the reference's own functions (read-only; nothing under
  `wifisense/` is modified).
- `metrics`: CFR, delay spread, frequency selectivity, path matching and the
  distribution summaries.
- `audit`: the 10-scene audit and the targeted slab check (command line).
"""

LABEL = "SIM"
