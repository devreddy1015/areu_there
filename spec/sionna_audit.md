# Sionna RT audit of Room Sim — Phase 10a

**Status:** method and results of Phase 10a, 2026-10-25. Built and run
under the owner's blanket permission of 2026-10-25. The Phase 10 plan is
"Sionna RT audit on 10 randomized scenes; differentiable calibration of
5–10 global parameters with Wasserstein". This is the audit only; the
calibration (10b) is designed from it.

**Every number here is SIM.** Room Sim v1 (spec/roomsim.md) and Sionna RT
2.2.0 are both simulators. An agreement below is an agreement between two
models of the same declared room. A disagreement says the two models
differ. Neither is a property of a real room, and neither simulator is
"the truth" for the other. The rooms are dataset v1's generated scenes
(spec/dataset.md), which describe no real room.

Code: `ml/sionna_audit/` (how to run: its README). Results:
`ml/results/phase10/sionna_audit.json`. Tests: `ml/tests/test_sionna_audit.py`.
Nothing under `wifisense/`, `rust/`, `spec/golden/` or `edge/` was
changed; Room Sim's Python reference is only called.

## 1. What is compared

For each of 10 rooms, every directed link `(tx, rx)` of distinct nodes, the
**static, empty-room channel**: no people, no ESP32 impairments (Room
Sim's §11 with `people = []`, before §12). Room Sim side: its Python
reference, `RoomSim(scene, cfg).link_model(tx, rx)`, with the scene's
own dataset run config (`scenes/<id>.roomsim.json`: seed, floor and
ceiling materials, defaults otherwise). Sionna side: the same scene
converted by `ml/sionna_audit/scene_to_sionna.py` (§3) and ray traced by
Sionna's `PathSolver`.

Four comparisons:

| Name | Room Sim | Sionna | Metrics |
|---|---|---|---|
| `shell` | walls, floor, ceiling: every object effect off (`object_specular`, `object_scatter`, `object_blockage` false) | the same room without furniture | (a)–(e), per-path |
| `furnished` | the full static model's specular part, kinds 0–3 (LoS, order 1, order 2, object faces) with its §6.9 transmission and blockage losses | the room with furniture as half-space boxes (§3.2) | (a)–(e), per-path |
| `furnished_panel` | as `furnished` | furniture as thin panels (§3.2): how much the choice matters | (a)–(e), per-path |
| `full` | the whole static model, kinds 0–6: also object point scatterers, the 24 clutter points and the diffuse tail, realized with the scene's seed; the tail also by expectation | `furnished` with diffuse reflection on every surface, scattering coefficient `S` = 0.3 (and 0.7 as a sensitivity) | delay spread, selectivity: qualitative |

Plus a targeted check of the slab coefficients (§4.4).

## 2. Scenes and statistics

Ten rooms drawn from dataset v1's test split (201 scenes) with
`numpy.random.default_rng(10).choice(test, 10, replace=False)`. If none of
the ten were L-shaped, the last would be replaced by an L-shaped test
scene; the draw already holds one (gen-000288), so nothing was replaced.
The ten are gen-000224, -000288, -000393, -001018, -001023, -001515,
-001600, -001691, -001722 and -001865. Between them they have 5–12
nodes, 906 directed links, brick (4), drywall (4) and wood (2) walls,
0–6 pieces of furniture, and concrete or wood floors. The draw happens
to contain no concrete or glass walls (35% and 5% of the dataset).

Both simulators are reciprocal before impairments, so `(a, b)` and
`(b, a)` agree up to edge cases. Each unordered pair is therefore counted
twice. Every distribution is over links, reported as median, p10 and p90,
per scene and over all 906 links. The JSON also has n, mean, min, max and
the count of non-finite values (a link with no direct path in Sionna, for
instance).

## 3. The conversion and every approximation

### 3.1 Geometry

The converter reads Room Sim's own resolved world
(`wifisense.roomsim.world.World`, read-only). Implied walls, wall heights,
the footprint, object frames and node positions are therefore exactly
Room Sim's.

- **Walls.** Each wall is one Sionna surface: a quad carrying an ITU slab
  material of the wall's thickness. This is Sionna's own wall model ("walls
  should be modeled as single flat surfaces having an attached radio material
  that accounts for their thickness", `PathSolver` docstring). The quad lies
  on the face Room Sim reflects from (§6.1: `t/2` from the centreline on the
  transmitter's side) and spans Room Sim's face line exactly, untrimmed at
  corners as §6.1 says, from `z = 0` to the wall's height.
  - When every node is on one side of the wall's line (every wall of a
    convex room), Room Sim uses that face for every bounce and the
    placement is exact.
  - An L's two re-entrant walls have nodes on both sides of their line.
    Room Sim reflects off whichever face the wave meets; a single surface
    cannot. The surface goes on the room-facing face: bounces from inside
    are exact, and Room Sim's bounces off the outer face (reached through
    the other re-entrant wall) differ by up to `t` in length (A1, §3.5).
  - A wall off the footprint boundary with nodes on both sides would go on
    its centreline. No audited scene has one.
- **Floor and ceiling.** The footprint polygon (the walls' centreline
  polygon in the dataset) at `z = 0` and `z = H`, ear-clipped into
  triangles. Material `floor_material` / `ceiling_material`, thickness
  `slab_thickness_m` (0.2 m), from the run config.
- **Furniture.** Every object but doors and windows becomes a closed box
  (12 triangles, corners from Room Sim's own `Box.to_world`). A box standing
  on the floor drops its bottom face, which would be coplanar with the
  floor.
- **Doors and windows** are not converted. Room Sim ignores them as
  obstacles (§6.5: the wall is modelled whole), so both simulators see
  whole walls.
- **Nodes.** Every node is both a Sionna `Transmitter` and a `Receiver` at
  its position.

### 3.2 Materials

Room Sim's §5.1 table is ITU-R P.2040-3 Table 3. Sionna's
`ITURadioMaterial` uses the same rows: the (a, b, c, d) of every mapped
material are identical at 2.437 GHz (checked in the tests).

| Room Sim | Used for | Sionna material | Thickness |
|---|---|---|---|
| `concrete` | walls, floor, ceiling | ITU `concrete` | the wall's `t`; 0.2 m for floor and ceiling |
| `brick` | walls | ITU `brick` | `t` |
| `drywall` | walls, ceiling | ITU `plasterboard` | `t`; 0.2 m |
| `glass` | walls | ITU `glass` | `t` |
| `wood` | walls, floor | ITU `wood` | `t`; 0.2 m |
| `metal` (object) | furniture | ITU `metal` | 0.01 m (half-space at any thickness above ~1 µm); `panel`: 0.002 m |
| `wood` (object) | furniture | ITU `wood` | 8.48 m (`halfspace`); `panel`: 0.02 m |
| `glass` (object) | furniture | ITU `glass` | 15.52 m (`halfspace`); `panel`: 0.01 m |
| `fabric` (object) | furniture | custom: `eps_r` 1, `sigma` 0.00920 S/m | 0.1 m |

- **Constants.** Room Sim's `EPS0` is 8.8541878128e-12 (CODATA 2018).
  Sionna takes `epsilon_0` from scipy 1.17.1, 8.8541878188e-12 (CODATA
  2022). The imaginary permittivity therefore differs by 6.8e-10 relative,
  which is negligible. Sionna computes in float32.
- **Furniture has no exact Sionna counterpart** (A4). Room Sim gives a
  large face the single-interface reflection `G` of its material (§5.2)
  and charges a fixed `L_db` per box crossing (metal 20, wood 4, glass 2,
  fabric 3 dB). A Sionna face is a slab whose `R` and `T` follow from one
  thickness, so no single thickness gives both.
  - `halfspace` (the `furnished` default) chooses the reflection: a thickness
    at which the slab's internal round trip is 1e-12, so `R` equals Room
    Sim's `G` to 1e-9 at every angle. The boxes are then opaque (per face
    `|T|^2 < 1e-12`).
  - `panel` chooses physical furniture panels: transmissive, but reflecting
    with thin-slab interference instead of `G`.
- **`fabric`** has no ITU material (Room Sim's `OBJECT_ITU` has none
  either). The stand-in is non-reflecting (`eps_r` = 1, reflection below
  −30 dB). Its conductivity is solved so that two faces lose Room Sim's
  3 dB at normal incidence. The loss grows at oblique incidence, where Room
  Sim's per-crossing loss does not.

### 3.3 Antennas, polarisation, frequencies

- **Antennas.** Every node uses Sionna's `iso` pattern with `V`
  polarisation (`C_theta` = 1, `C_phi` = 0, antenna z = world z). This is
  Room Sim's "isotropic and vertically polarised" (§16): gain 1 in every
  direction, field along theta-hat. The LoS coefficient is then
  `LAMBDA / (FOUR_PI * L)` in both (measured: identical to 1e-6 dB on every
  link).
- **Polarisation.** Sionna transports the full polarised field (Jones
  matrices). Room Sim multiplies by a scalar: `G_TE` at walls, `G_TM` at
  floor and ceiling. The two agree exactly where the field is pure TE (a
  horizontal leg at a vertical wall) or pure TM (a floor or ceiling bounce
  in a vertical plane), as the ray-traced check shows (Table 5). On tilted
  legs at walls they differ: Room Sim's TE-only approximation (§16), A3.
- **Frequencies.** Both use channel 6 (2.437 GHz). The 52 LLTF subcarriers
  are `carrier + k * 312.5 kHz`, `k = -26..-1, 1..26` (§4), bit for bit
  Room Sim's grid (tested). Both evaluate the materials at the carrier and
  apply them across the 16.25 MHz band (Room Sim §16; Sionna's scene
  frequency is the carrier).
- **CFR.** Both CFRs are evaluated as `H(f) = sum a * exp(-j 2 pi f tau)`
  in float64 from each simulator's path coefficients `a` (no propagation
  phase) and delays `tau = L / C`. This equals Room Sim's own §11 channel
  to 1e-12 (checked on every link) and Sionna's `Paths.cfr` with baseband
  offsets to 1e-3 (tested).

### 3.4 Sionna solver settings, and three Sionna 2.2.0 behaviours

**Settings.** `PathSolver` with LoS, specular reflection and refraction on;
diffraction and (except in `full`) diffuse reflection off;
`max_depth` 4; 1,000,000 rays per source; `max_num_paths_per_src`
2,000,000; seed 42; synthetic arrays. The run stops if a solve stores as
many candidates as the cap. `max_depth` counts reflections and refractions
together: 4 holds Room Sim's order-2 paths plus two crossings (an L's
re-entrant walls, a fabric box) and also bounds Sionna's "all orders"
reference.

**Backend.** Mitsuba `llvm_ad_mono_polarized` on the CPU, with libLLVM 23
from the Rust toolchain (README). The CUDA variant aborts the process
inside OptiX's compiler on this machine (driver 545, CUDA 12.3: "missing
implementation for PTX intrinsic optix.ptx.copysign.f32"). The audit uses
no GPU memory beyond a ~160 MB CUDA context.

**Three behaviours, and how the audit handles them:**

1. **Several receivers in one solve lose candidates.** With all nodes as
   receivers in one solve, the candidate generator never stored some
   specular chains for whole (source, target) blocks. In gen-000000
   (9 nodes, shell), the floor → ceiling double bounce was missing on 37 of
   72 links (sources {1, 3, 4} × targets {0, 1, 3} in a 5-node subset). It
   was missing at candidate generation, not rejected later. The same chain
   was found on every link when each solve had one receiver, and in an
   isolated floor/ceiling scene. A replica of Sionna's plane, chain and
   target hashing (FNV-1a) shows no collision for these chains, so the
   cause is not pinned down; it is recorded as observed. **The audit solves
   once per receiver**, with every transmitter present. That costs one ray
   launch per receiver.
2. **Refraction-only paths through opaque boxes.** Sionna hashes a
   refraction by its type alone, not its surface, and its image method
   accepts any surface for a refraction. A candidate of k refractions
   anywhere therefore becomes the straight line crossing whatever k surfaces
   it meets, including a half-space box. Such a path is about 240 dB per box
   below free space. Paths more than 150 dB below free space at their own
   length are dropped as blocked, and the count is reported (Table 1 notes).
   Real paths in these rooms stay well within 100 dB of free space.
3. **Depth 6 loses candidates at random.** Chains that leave the room fill
   the candidate de-duplication hash table. Real order-2 corner paths
   (−70 to −88 dB) were then present in one run and absent in another. At
   depth 4 the path sets were identical across seeds and across 10^6 vs
   4·10^6 rays in gen-000000.

**Reproducibility** at the audit's settings: the `furnished` comparison of
the smallest scene is rerun with seed 43. The result is in §5.

### 3.5 What does not map, both ways (A1–A11)

- **A1. Wall faces.** Room Sim reflects off the face on the wave's side and
  tests a crossing on the centreline (§5.1, §6.1). Sionna's wall reflects
  and transmits in one plane. This is exact for walls with every node on
  one side. For an L's re-entrant walls (§3.1), Room Sim's outer-face
  bounces and crossings near the re-entrant corner differ slightly.
- **A2. Transmission phase.** Room Sim's `T` carries `e^{+jq0}` because
  every path length counts the slab as free space (§5.1). Sionna's `T` does
  not. Each wall crossing therefore differs by exactly
  `q0 = k0 * t * cos(theta)` in phase, with equal magnitude (Table 5). This
  is a convention of each model and neither is changed. It shows in the
  phases of "+T" paths (Table 3).
- **A3. Polarisation.** Room Sim uses scalar TE (walls) and TM (floor,
  ceiling) coefficients; Sionna is fully polarimetric (§3.3).
- **A4. Furniture.** In Room Sim, a large face reflects once with `G`; a
  face smaller than two Fresnel radii is a point scatterer
  (`rho * area`); a box crossing costs `L_db`; and objects never take part
  in order-2 paths. In Sionna, every face reflects specularly wherever the
  specular point falls on it (no size test); reflections combine with walls
  and other objects to any order; crossing a face is its slab `T`; and
  there are no point scatterers. Section 3.2 gives the two thickness
  choices. In `halfspace`, furniture blocks completely where Room Sim
  charges 2–20 dB.
- **A5. Fabric** has no ITU material (§3.2).
- **A6. Point scatterers, clutter, tail** (kinds 4–6) have no Sionna
  counterpart. In `full`, Sionna's diffuse reflection (scattering
  coefficient `S`, Lambertian, on every surface) is the physical analogue.
  Its `S` is a choice, not a fit.
- **A7. Openings.** Neither simulator models doors or windows.
- **A8. Diffraction.** Off in Sionna; Room Sim has none.
- **A9. Depth.** Sionna counts refractions in `max_depth` (4). Room Sim's
  transmissions and blockages are free, so an order-2 path with more than
  two crossings has no Sionna counterpart. Such paths cross an L's
  re-entrant walls, or cross a transmissive box (fabric, `panel`) twice.
  They are weak.
- **A10. Precision and edges.** Room Sim is binary64 and admits a path that
  meets the edge between two surfaces by the last bit (§6.4). Sionna is
  float32 and drops any segment shorter than 1 cm. Order-2 paths whose two
  bounce points lie within ~1 cm of each other, at a wall–floor or
  wall–ceiling junction, are therefore Room Sim only. These are the
  residual unmatched paths of `shell`, 25–60 dB below their link.
- **A11. Frequency-flat materials** in both: no difference.

## 4. How

### 4.1 The specular comparison (metrics a–e)

Per link, the two path sets:

- **Room Sim:** kinds 0–2 (`shell`) or 0–3 (`furnished*`), with their
  static coefficients `A_p`, including §6.9 wall transmissions and object
  blockage.
- **Sionna:** every path made only of specular reflections and refractions
  with at most 2 reflections (Room Sim's `reflection_order`). Any number of
  refractions is allowed within depth 4; blocked paths are dropped (§3.4).

The metrics:

- **(a) LoS power.** Room Sim's kind-0 `|A|^2` against Sionna's direct path
  (no reflection; refractions allowed, e.g. through an L's walls or a fabric
  box), as `dP_LoS = 10 log10(P_RS / P_S)`. A link where Sionna has no
  direct path (furniture blocks it) is counted, not put in the distribution;
  for those links Table 1b gives Room Sim's LoS against free space.
- **(b) Total specular power.** `dP_spec = 10 log10(sum |A|^2 / sum |a|^2)`.
  Also reported: Sionna's power with 3–4 reflections added (what order-2
  truncation leaves out, per Sionna), and the power fractions of paths only
  one side has.
- **(c) CFR magnitude** over the 52 LLTF subcarriers:
  - the Pearson correlation of `20 log10 |H|`;
  - the same correlation over links where both selectivities are at least
    1 dB (a flat `|H|` makes correlation meaningless);
  - the RMS of the dB difference after removing its mean;
  - that mean.
- **(d) RMS delay spread** of each path set:
  `sqrt(sum p (tau - mean)^2 / sum p)`.
- **(e) Frequency selectivity:** the standard deviation (ddof 0) over the 52
  subcarriers of `20 log10 |H|`.

### 4.2 Path matching

With both endpoints fixed, a sequence of mirror planes fixes a path (the
image method). A Room Sim path and a Sionna path with the same sequence of
reflecting shapes (refractions ignored) are therefore the same path: one
per key on each side. Each matched pair gets `dP` (Room Sim minus Sionna,
dB), `|dphase|` and `|dL|` (length). Pairs are grouped by class: LoS;
order 1 at a wall or at the floor/ceiling; order 2 wall–wall,
wall–floor/ceiling or floor–ceiling; object specular. "+T" marks a path
that Sionna refracts somewhere, where A2 applies. Unmatched paths are
counted by kind (Room Sim) or by class (Sionna: whether they touch
furniture).

### 4.3 The full-model comparison (qualitative)

- **Room Sim:** every kind 0–6 term of the reference (its static model
  without people), with each scene's dataset seed. `H` is the reference's
  own `LinkModel.channel([])`. The delay spread is computed from all terms,
  and again with the tail replaced by its expectation: §6.8's exponential
  power–delay profile of total `10^(tail_power_db/10) * P_spec` starting at
  `L_ref`, the longest kind 0–3 path.
  - Components: the delay spread with only the tail, only the clutter or
    only the point scatterers added to kinds 0–3.
  - A what-if diagnostic, labelled as such and **not Room Sim's model**:
    the expected tail started at the LoS delay instead of `L_ref`.
- **Sionna:** `furnished` (`halfspace`) with every material's scattering
  coefficient `S` = 0.3 (0.7 as a sensitivity), Lambertian pattern, diffuse
  reflection on, depth 4, 500,000 rays per source. Every path is used
  (specular, with Sionna's `sqrt(1 - S^2)` reduction, plus diffuse).

Only delay spread and selectivity are compared, plus each side's
stochastic-to-specular power ratio. A diffuse run is a Monte Carlo estimate.

### 4.4 The slab check

1. **Formula.** Room Sim's `materials.slab` against Sionna's
   `itu_coefficients_single_layer_slab`, with Sionna's own ITU permittivity:
   5 wall materials × {0.1, 0.2} m × {0, 15, 30, 45, 60, 75, 85}° × TE/TM.
   `R` must agree, `|T|` must agree, and `arg T` must differ by exactly
   `q0` (A2). The half-space furniture mapping is checked the same way:
   Sionna's `R` against Room Sim's `G`.
2. **Ray traced**, through the whole pipeline (converter material, ray
   tracer, iso/V antennas): one 0.2 m concrete surface.
   - Wall reflection and transmission at 0, 30, 60 and 80° with horizontal
     legs (pure TE).
   - Floor reflection at 30, 60 and 80° in a vertical plane (pure TM).
   - `a * 4 pi L / lambda` is compared against Room Sim's `R` and
     `T * e^{-jq0}`.
   - One tilted wall bounce shows the TE-only approximation (A3).

## 5. Results (all SIM)

<!-- RESULTS -->

## 6. Findings, and what they mean for 10b (all SIM)

<!-- FINDINGS -->

## 7. Limits of this audit

- **Two simulators.** Agreement here does not validate either against a
  real room; only REAL captures can (Phases 9, 11).
- **Ten generated rooms.** Rooms are boxes or Ls, with at most six boxes of
  furniture, brick, drywall or wood walls, and 5–12 wall-mounted nodes.
  The draw contains no concrete or glass walls. Links are counted in both
  directions.
- **Converter choices.** Furniture thickness (`halfspace` or `panel`), the
  fabric stand-in, the diffuse `S` and the Lambertian pattern are choices.
  `furnished_panel` and `S` = 0.7 show how much two of them matter.
- **Sionna settings.** Depth 4, 10^6 rays (5·10^5 for diffuse), float32,
  no diffraction, one receiver per solve. A diffuse run is one Monte Carlo
  draw.
- **Not audited:** people (§8), shadowing, the ESP32 impairment layer (§12)
  and the Rust crate. The crate agrees with the Python reference to ~1e-9
  (spec/roomsim.md §14), so these results carry over to it.
