# P0b: device-free localization — findings

Phase P0b of `CAMPUS_SENSING_PROMPT.md`. Build and honestly evaluate
link-based localization of one walking person on simulated WiFi links.

**Headline: at 4–12 single-antenna nodes, radio tomographic imaging does
not beat a trivial weighted-link-midpoint estimator.** RTI reached 0.72 m
median error at 13–18% coverage; the baseline reached 0.709 m at 100%
coverage, and beat or tied RTI in every one of 15 (λ, α) configurations
tested. This matches what the literature predicts for this node count and
is the main result of this phase.

All numbers below are **SYNTHETIC**.

---

## 1. What I built

| File | What |
|---|---|
| `wifisense/localize/geometry.py` | Room, Node, Link, PixelGrid, Wilson–Patwari ellipse weight model, canonical link/pixel ordering |
| `wifisense/localize/sim.py` | Multi-link CSI simulator: LoS, first-order image-method walls, body scattering, shadowing, random-waypoint motion, TDMA slotting, packet loss |
| `wifisense/localize/measure.py` | VRTI per-link variance with empty-room calibration; two selectable signals (total power, static-suppressed amplitude) |
| `wifisense/localize/invert.py` | Regularized least squares in **dual form**, `C Wᵀ (W C Wᵀ + αI)⁻¹ y`, with the P×L projection precomputed once |
| `wifisense/localize/track.py` | Sub-pixel centroid, constant-velocity Kalman, significance-gated abstention |
| `evaluate_localize.py` | The sweep harness, with trivial baselines |
| `tests/test_localize_*.py` | 208 tests |

Pure Python standard library throughout — own Gaussian elimination, own
Kalman filter, no numpy. The dual form matters: it needs only an L×L solve
(56 links at N=8) rather than P×P (192–768 pixels), which is what makes a
pure-Python inversion tractable at 0.28 ms per solve.

## 2. RTI literature — verified, with conditions attached

This literature is widely misquoted. Each number below was checked against
a primary source.

| Work | Radio | Reported accuracy | Conditions |
|---|---|---|---|
| Wilson & Patwari, **TMC 2010** (RTI) | RSS on **802.15.4** TelosB motes | **No metre-level error is reported anywhere.** Only a dimensionless normalized-image MSE: 0.021 (one person), 0.036 (two) | 28 nodes, 41 m², **outdoors on grass**, not through-wall |
| Wilson & Patwari, **TMC 2011** (VRTI) | RSS variance on 802.15.4 | Average Euclidean error **1.03 m** walking a known path; **0.44 m** standing at marked spots with 10 s averaging | 34 nodes, 72.5 m², through-wall (incl. interior brick) |
| Zhao & Patwari, **SECON 2011** | RSS variance, same hardware | **RMSE 0.70 m and 1.26 m**; errors up to **6 m** in wind | Repeat of the VRTI experiment, same house, same hardware |
| Kaltiokallio et al., **MASS 2012** | RSS, 5-channel diversity | 0.10 m open room (**below the 0.1524 m voxel width — the authors say so**); 0.96 m cluttered through-wall | 30 nodes, 70 m², **stationary subjects only** |
| Kaltiokallio et al., **ARTI, TVT 2017** | RSS, 16-channel | Real-time through-wall median **43.7 cm** (95th pct 2.37 m); sub-10 cm figures are **offline-smoothed** | 30–33 nodes, 58–86 m² |
| Kim et al., **arXiv 2505.04085** (2025) | **USRP X310, 100 MHz BW, 8-element arrays, shared 10 MHz ref** | "Sub-metre in most scenarios"; the 0.05 m figure is **ray-traced simulation** | 4–12 nodes; 4- and 8-node configs "degrade significantly" |
| Sun et al., **ICEIEC 2018** | **WiFi CSI** — the only peer-reviewed CSI-RTI I found | **No numbers in any accessible source** | Unknown |
| Weerasiri, **BSc thesis 2025** | **WiFi CSI on ESP32**, 4 nodes | **States no error in metres** | Closest existing work to this build; unreviewed |

Five things worth carrying forward:

1. **The number everyone quotes for Wilson & Patwari 2010 does not exist.**
   That paper reports no localization error in metres at all.
2. **The inventors failed to reproduce their own VRTI result.** Zhao &
   Patwari repeated it with identical hardware and got up to 6 m error,
   because wind moved the node stands — calibration-period RSS variance went
   from <5 dB to 30 dB. This is the most honest available figure for VRTI on
   a bad day, and it is from the people who invented it.
3. **Every good RTI number is bought with node count.** 28–34 nodes over
   40–86 m², i.e. 435–561 links. Eight nodes gives 28 links (unordered).
   That is one to two orders of magnitude fewer constraints on an inverse
   problem that is already ill-posed.
4. **There is no verifiable published localization error for RTI on WiFi
   CSI.** Every metre-level RTI number is RSS on motes, mmWave, or
   synchronised SDR.
5. **Channel diversity, not algorithmic cleverness, produced the biggest
   single jump in RSS-RTI accuracy** (MASS 2012, an order of magnitude).
   That is the one historical lesson pointing straight at CSI subcarriers.

Published weight model, both papers:

```
w(l, p) = (1/sqrt(d_l)) * { 1   if d(tx,p) + d(p,rx) < d_l + lambda
                          { 0   otherwise
```

`d_l` is **node separation**, not voxel-to-node distance — an easy bug.
Published λ is 0.01–0.02 m: the ellipse is "essentially the same as using
the LOS model", a computational device for finding voxels on the line, not
a wide Fresnel zone.

## 3. Synthetic results

8×6 m room, 0.5 m grid, 120 s runs × 3 seeds, 20 s empty-room calibration,
S0 = 30 dB shadowing, total-power signal. `wgt` = weighted-link-midpoint
baseline. `cov` = fraction of moving time the tracker answered.

| Variant | links | MOVE med | p90 | cov | STILL med | centre | maxlink | **wgt** |
|---|---|---|---|---|---|---|---|---|
| N=4 | 12 | 0.44 | 1.69 | 15% | 0.43 | 2.21 | 1.19 | 0.72 |
| N=6 | 30 | 0.97 | 1.59 | 17% | 1.57 | 2.20 | 1.49 | **0.65** |
| N=8 | 56 | 0.82 | 1.40 | 18% | 1.02 | 2.18 | 1.58 | **0.76** |
| N=12 | 132 | 0.71 | 1.11 | 13% | 1.10 | 2.26 | 1.38 | **0.69** |

Signal choice, at N=8:

| Signal | MOVE med | cov | wgt |
|---|---|---|---|
| total power (RSS-like) | 0.82 | 18% | 0.76 |
| static-suppressed CSI | 1.28 | 49% | 0.44 |

Exhaustive (λ, α) grid at N=8, 177 moving samples, centroid estimator:

| λ | best RTI centroid | weighted baseline | blind pixels |
|---|---|---|---|
| 0.02 (published) | 0.754 | 0.709 | 26/192 |
| 0.1 | 0.747 | 0.709 | 6/192 |
| 0.3 | 0.715 | 0.709 | 2/192 |
| 0.6 | 0.717 | 0.709 | 0/192 |
| 1.0 | 0.716 | 0.709 | 0/192 |

**RTI did not beat the baseline in any of the 15 configurations.**

Structural observability (pure geometry, algorithm-independent — pixels
inside no ellipse cannot be reconstructed by any inversion):

| N | λ=0.1 | λ=0.3 | λ=0.6 |
|---|---|---|---|
| 4 | 33.3% | 8.3% | 1.0% |
| 6 | 8.3% | 3.1% | 1.0% |
| 8 | 3.1% | 1.0% | **0%** |
| 12 | 1.0% | 1.0% | 0% |

### What this means for the 8-node mesh in P2

- **Do not expect metre-level tracking from 8 ESP32 nodes.** Plan for
  motion detection and zone-level localization (which quadrant/half).
- **The weighted-link-midpoint estimator is the honest default.** It is six
  lines of code, needs no inversion, no grid, no regularization and no
  tuning, had 100% coverage against RTI's 13–18%, and was never worse.
  Ship that first; treat RTI as the thing that must earn its place.
- **N=8 at λ≥0.6 is the first configuration with full structural
  coverage.** N=4 leaves a third of the room unobservable at tight λ.
- **Protect the per-link sample rate above all else.** Adding nodes costs
  rate under TDMA, and rate is what makes variance features work at all.
  Accuracy was nearly flat from 3.1 to 12.5 Hz per link here, but this
  project's own research records a 9-node ESP32 mesh at 1.4 Hz aggregate
  detecting sedentary presence only 38% of the time.
- **Set the acceptance criterion as zone-classification accuracy and
  motion-detection ROC, not metres.**

## 4. What failed, and what surprised me

**RTI losing to a six-line baseline is the main result.** It is also exactly
what the literature predicts at this node count, so it should not be read
as an implementation failure. The inversion is correct — it recovers a
known forward-projected blob to within one pixel, and the dual form matches
a direct primal solve on small geometries.

**Three things I got wrong during this phase, each caught by measurement:**

1. **I scored the still case by whether the tracker answered, calling any
   answer "false confidence".** That was wrong. A still person still
   breathes, so links are still modulated — and the still-case estimates
   turned out to be *accurate* (0.43–1.10 m, comparable to the moving case).
   Measuring "did it answer" without "was it right" mistakes a capability
   for a bug. **Caveat:** this depends entirely on the simulator giving a
   still person meaningful breathing modulation. Per the P0a work and
   RESEARCH.md §5a, a real still person at a Fresnel null produces almost
   nothing at 20 MHz. This result is simulator-optimistic.

2. **I took the image argmax instead of its centroid.** RTI images are
   blurry blobs; argmax on a 0.5 m grid quantises and is noisy. This alone
   understated RTI by ~0.5 m (1.29 → 0.75 m).

3. **I swept α over a range that was a no-op.** α is only meaningful
   relative to the scale of `WᵀW` and `y`. The published `psi` scale factor
   exists precisely to keep them commensurate. My first sweep (0.001–10)
   changed the answer by 0.003 m across four orders of magnitude, which I
   initially read as "the inversion is broken"; the real reading was "α is
   far below the data scale". At α=100 it finally bites (0.80 → 0.72 m).

**The genuine surprise:** the ellipse weight model fits this physics poorly.
Correlation between model weight and measured link disturbance was only
**+0.13**, with **29% of timesteps actively negative** and an
inside/outside ratio of 1.8×. Raising shadowing to S0=30 dB lifted that to
+0.50 and 19×, confirming the mechanism: **RTI's premise is shadowing**
(valid for RSS on 802.15.4, where a body blocks the path), while **CSI
amplitude is dominated by coherent multipath**, which is not monotonic in
distance from the line of sight. This is the same Fresnel physics that
gates the P0a breathing work, seen from the other side.

**Published λ is wrong for sparse deployments.** λ≈0.02 m is right at 28–34
nodes, but at 8 nodes a line-thin ellipse leaves 26/192 pixels blind and
starves the inversion. λ had to be ~30× wider here. Coverage is bought with
resolution.

**Accuracy improved as coverage fell, in every sweep.** z=8 gives 0.52 m at
3% coverage; z=3 gives 1.03 m at 42%. That is correct behaviour — it is
accurate when confident — but it means any headline accuracy number is
selected by the abstention threshold and is meaningless without its
coverage figure. The same effect makes N=4 look best (0.44 m at 15%); it is
selection, not capability.

## 5. Requests for the main session

Nothing blocking. Two optional items:

1. `CsiFrame` has no `tx_id`. I carry `(t, tx_id, rx_id, frame)` tuples in
   my own stream instead. If a `tx_id` field is ever added, my simulator
   output could become a plain frame stream — cosmetic, not needed.
2. If `synthetic.py`'s geometric scene gains a shadowing term, I would like
   to match parameters. My `shadow_s0_db` / `shadow_delta0_m` are mine
   alone; per the P0a session, their model has no LoS shadowing.

Confirmed consistent with P0a: equation 6 pinned at 4.37 m (HT-LTF
17.5 MHz), the torso-extent correction, and the finding that depth-wise
motion along the bisector genuinely averages over Fresnel zones — which is
the physical reason localization-under-motion is tractable where
breathing-while-still is not.

## 6. Test result

`python3 -m unittest discover -s tests -p 'test_localize*.py'` — **208
tests, all passing**, in 8.2 s.

`python3 -m unittest discover -s tests` (full suite, both sessions' work) —
**351 tests, all passing**, in 211.9 s. No failures in files owned by the
main session.

## 7. If this is taken further

In the order the evidence supports:

1. **Protect per-link sample rate.** It is the one quantity that both
   enables variance features and is destroyed by adding nodes.
2. **Exploit subcarrier diversity** with per-link fade-level scoring — the
   CSI analogue of the channel diversity that bought an order of magnitude
   in MASS 2012. This is the highest-value idea in the RSS literature and
   it is free on CSI.
3. **Add nodes.** Conditioning scales with links, not nodes.
4. **Keep the target inside the node perimeter.** Every cited result does.
5. **Do not widen λ before the rest works** — it buys coverage with
   resolution and hides problems.

And test on a second room before believing anything. Every number here came
from one simulator written by the same process that wrote the estimator.
