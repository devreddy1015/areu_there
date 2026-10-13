# Task P0b: device-free localization on synthetic data

You are one of **two Claude sessions working in this folder at the same
time**. The other session (the "main session") is doing phase P0a. It is
changing the breathing pipeline, the synthetic source, and adding a packet
recorder. Your job is phase **P0b**: build and honestly evaluate
link-based localization of one walking person on simulated WiFi links.
You need no hardware.

The project background is in `CAMPUS_SENSING_PROMPT.md` (the camera-free
campus plan; your phase is §11 P0b), `RESEARCH.md` (binding literature
constraints) and `NOTES.md` (how this codebase was built). Read all three
first, then read `wifisense/types.py`, `wifisense/preprocess.py` and
`wifisense/dsp.py`.

---

## 1. Coordination rules (strict, because there is no git here)

**Files you own.** Create and edit only these:

- `wifisense/localize/` (new package, everything inside it)
- `tests/test_localize_*.py` (new test files)
- `evaluate_localize.py` (new, repo root)
- `NOTES-localize.md` (new, repo root): your findings log and your report

**Every other file is read-only for you.** That includes `wifisense/*.py`,
`wifisense/sources/*`, the existing tests, `evaluate.py`, `README.md`,
`NOTES.md`, `RESEARCH.md` and `CAMPUS_SENSING_PROMPT.md`. The main session
is editing several of them right now. If you need a change in a shared
file (for example a `tx_id` field on `CsiFrame`), don't make it. Write it
under **"Requests for the main session"** in `NOTES-localize.md` and work
around it in your own package.

**Public APIs you may import.** The main session guarantees these stay
backward compatible:

- `wifisense.types.CsiFrame(amplitudes, phases, sample_index, sample_rate_hz, rssi_dbm=0, node_id=0)`
- `wifisense.preprocess.StaticSuppressor(n_subcarriers, sample_rate_hz, corner_hz=0.03)`, whose `.process(frame)` returns `(residuals, weights)`
- `wifisense.preprocess.corner_to_alpha`
- `wifisense.dsp.BiquadBandpass(f_low, f_high, sample_rate_hz)` with `.process(x)`, plus `goertzel_power` and `band_peak`

**Transient breakage.** Importing `wifisense.*` runs the package
`__init__` files, which the main session is editing. If an import of a
file you don't own fails briefly, wait a minute and retry. Don't fix it.

**Testing.** Run only your own tests while working:

```
python3 -m unittest discover -s tests -p 'test_localize*.py'
```

Run the full suite (about 3 minutes) once at the end. If a failure is in
a file you don't own, report it rather than fixing it.

**Other limits:**

- Don't use git, don't install packages, and don't reformat existing code.
- **Pure Python standard library only.** numpy is not installed and the project deliberately avoids dependencies so the code ports to C. If you are truly blocked without numpy, stop and ask the user.

## 2. Background: why link-based localization

`RESEARCH.md` §0 establishes the following. ESP32 nodes have one RF chain,
so they can't do angle-of-arrival (MUSIC, SpotFi) and separate nodes don't
form a coherent array. Linear-fit phase sanitization destroys absolute
ranging, so multilateration is off the table.

What single-antenna nodes *can* do is **radio tomographic imaging (RTI)**.
Place N nodes around the room so every pair forms a link. A person
disturbs the links whose paths pass near them, and inverting "which links
are disturbed" gives an image of where they are. The variance-based
version (VRTI) works on the *variance* of each link over a short window,
which captures motion and also sees through walls.

Primary references (look them up and verify; don't trust these summaries):

- Wilson & Patwari, *Radio Tomographic Imaging with Wireless Networks*, IEEE TMC 2010. Introduces the ellipse weight model and regularized inversion.
- Wilson & Patwari, *See-Through Walls: Motion Tracking Using Variance-Based Radio Tomography Networks*, IEEE TMC 2011. This is VRTI.

Those papers used RSS on 802.15.4 radios, not WiFi CSI. Record in
`NOTES-localize.md` what accuracy they actually reported and under which
conditions (node count, room size, through-wall or not), with citations.
Keep the distinction clear: their numbers are not your numbers.

## 3. What to build

### 3.1 Multi-link synthetic source (`wifisense/localize/sim.py`)

Physics-based, deterministic under a seed, configurable:

**Room and nodes**
- A 2-D room, default 8 m × 6 m.
- N nodes evenly spaced around the perimeter, inset about 0.1 m.
- Every ordered pair (tx, rx) is a link.

**Per-link CSI per subcarrier.** Use K = 56 subcarriers, 312.5 kHz
spacing, centre 2.437 GHz. Each subcarrier's channel is the sum of four
terms:

1. **Line of sight.** Amplitude ∝ 1/d, phase −2π f_k d / c, multiplied by a *shadowing* gain.
2. **Static wall reflections.** First-order image method off the 4 walls, reflection coefficient around 0.3–0.5.
3. **The body-scattered path.** Amplitude ∝ 1/(d₁·d₂), phase −2π f_k (d₁+d₂)/c, where d₁ = |tx−p| and d₂ = |p−rx|. Add chest-displacement modulation (breathing) while the person stands still.
4. **Noise.** Complex Gaussian noise, plus a random common phase per packet (CFO).

**Shadowing.** With excess path length δ = d₁ + d₂ − d, attenuate line of
sight by S(δ) = S₀·exp(−δ/δ₀) dB. Make S₀ (around 3–10 dB) and δ₀
tunables, document the values you choose, and sweep them.

**Motion.** Random-waypoint walking at 0.3–1.2 m/s, with pauses of 0–20 s.
During a pause the person stands still and only breathes.

**TDMA timing.** Each superframe lasts N × slot_ms. Node *i* transmits in
slot *i* and every other node receives, so each link is sampled once per
superframe, giving a per-link rate of 1/(N·slot). Add a per-packet loss
probability.

**Output.** A time-ordered stream of `(t, tx_id, rx_id, CsiFrame)` with
`node_id = rx`, plus a ground-truth stream of `(t, x, y, moving)`.

### 3.2 Localization (`wifisense/localize/`)

Suggested split. Adjust it if you have a better design.

- **Geometry.** Room, nodes, links, pixel grid (0.5 m default; also try 0.25 m).
- **Weight model** (Wilson & Patwari ellipse): w(l, p) = 1/√d_l if d(tx,p) + d(p,rx) < d_l + λ, else 0. λ is a tunable, so sweep it.
- **Per-link measurement** (VRTI style): the variance of the link signal over a short window (about 0.5–2 s), with the empty-room variance per link (calibration) subtracted or normalized out. Compare two signals:
  - (a) total received power, as an RSS-like baseline
  - (b) static-suppressed CSI amplitude averaged over subcarriers (`StaticSuppressor`)
- **Inversion.** Regularized least squares with a spatial prior C_pq = σ²·exp(−dist(p,q)/δc). Use the **dual form** x̂ = C Wᵀ (W C Wᵀ + αI)⁻¹ y: it only needs an L×L solve, where L is the number of links. Precompute the P×L projection once per geometry. In pure Python, a 0.25 m grid is too slow with a P×P solve but fine this way.
- **Tracker.** Peak extraction with a sub-pixel centroid, then a constant-velocity Kalman filter. Output `(x, y, sigma_m, confidence)`. **Abstain** when the image peak isn't significant: VRTI can't see a person standing still, and pretending otherwise is the failure we're avoiding.

### 3.3 Evaluation (`evaluate_localize.py`)

- **Sweep:** N ∈ {4, 6, 8, 12}; pixel size; λ; noise level; slot timing (which sets per-link rate); measurement (a) vs (b).
- **Per configuration:** several seeds × 120 s random walks. Report:
  - median and p90 error (m) while **moving**
  - coverage (the fraction of time it doesn't abstain)
  - behaviour while **standing still**, reported separately
  - update rate achieved
  - compute time per update
- **Trivial baselines** on the same data. If RTI can't clearly beat these, say so:
  - always guess the room centre
  - the midpoint of the most-disturbed link
  - link midpoints weighted by disturbance
- Print a clear results table labelled SYNTHETIC. Keep the whole run under about 10 minutes, and offer a `--quick` mode.

### 3.4 Tests (`tests/test_localize_*.py`)

- ellipse membership and symmetry
- inversion recovers a known single blob
- Kalman filter on a straight-line walk
- the simulator is deterministic under a seed
- per-link rate equals 1/(N·slot)
- shadowing increases as the body nears the line of sight
- abstention when nobody is moving

Each test file should run in under about 30 s.

## 4. Standards (same as the rest of this repo)

- `from __future__ import annotations`, type hints, docstrings that explain *why* (match the style of `dsp.py` and `presence.py`).
- Measure before claiming. Every number in your notes comes from a script anyone can rerun. Label synthetic results as synthetic.
- If something doesn't work, report that as a result. Don't tune until the table looks good. Watch especially for cases where the tracker only "works" because the room centre is a good guess.

## 5. Report back

When you're done, `NOTES-localize.md` must contain:

1. What you built, file by file.
2. RTI literature findings with citations, and the reported accuracies with their conditions.
3. The SYNTHETIC results table and what it means for the real 8-node mesh in phase P2: how many nodes, what per-link rate, what accuracy to expect.
4. What failed or surprised you.
5. Requests for the main session (changes needed in shared files).
6. Full-suite test result.

End your session with a ≤ 15-line summary that the user will paste back
to the main session.
