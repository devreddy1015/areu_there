# ml/sionna_audit — the Sionna RT audit of Room Sim (Phase 10a, SIM)

Compares Room Sim v1 (spec/roomsim.md; the stdlib Python reference
`wifisense/roomsim/`, used read-only) with **Sionna RT 2.2.0** on 10 rooms
drawn from dataset v1's test split. Both are simulators: every number this
produces is **SIM**, an agreement or disagreement between two models of the
same declared room, never a property of a real room. What is compared, how,
every approximation and the results are in **spec/sionna_audit.md**.

## Run

From the repo root, in the venv (numpy, scipy, sionna-rt 2.2.0, mitsuba
3.9.1, drjit 1.5.0):

```sh
.venv/bin/python -m ml.sionna_audit.audit                                  # the audit: 10 scenes, ~25 min on 96 cores
.venv/bin/python -m ml.sionna_audit.audit --markdown /tmp/audit.md         # also the result tables
.venv/bin/python -m ml.sionna_audit.audit --indices 288 1515 --quick       # a smoke run (fewer rays, not the audit)
.venv/bin/python -m unittest ml.tests.test_sionna_audit                    # tests (Sionna parts skip without it)
```

It writes `ml/results/phase10/sionna_audit.json` (`--out` to change): small,
summaries only (median, p10, p90 per metric, per scene and overall), every
block labelled SIM. Options: `--dataset DIR` (or `SAANS_DATASET_DIR`),
`--scenes N`, `--scene-seed`, `--indices I ...`, `--variants shell furnished
furnished_panel`, `--diffuse-s 0.3 ...` (none: skip the full-model check),
`--spec-samples`, `--spec-depth`, `--no-slab-check`.

## Backend (read this if Sionna fails to start)

The audit selects Mitsuba's **`llvm_ad_mono_polarized` (CPU)** variant
before importing `sionna.rt`. On the audit machine the CUDA variant aborts
the whole process inside OptiX's compiler (driver 545 / CUDA 12.3: "New
backend is missing implementation for PTX intrinsic
optix.ptx.copysign.f32"), so it is never tried automatically; set
`SIONNA_AUDIT_VARIANT=cuda_ad_mono_polarized` on a machine where it works.

Dr.Jit's LLVM backend needs a `libLLVM.so` of version >= 15, and LLVM 15
itself cannot lower the `llvm.maximum` intrinsic Dr.Jit 1.5 emits ("Cannot
select: fmaximum"); the system's LLVM 15 also needs a newer libstdc++ than
miniconda's Python carries. The audit therefore looks for the newest
libLLVM >= 17 (`find_libllvm`), which here is the Rust toolchain's
`~/.rustup/toolchains/stable-x86_64-unknown-linux-gnu/lib/libLLVM.so.23.1-rust-1.99.0-stable`
(it links libstdc++ statically). Set `DRJIT_LIBLLVM_PATH` to override.
The process still opens a CUDA context (about 160 MB of GPU memory); no
GPU computation runs.

## Modules

| Module | What |
|---|---|
| `scene_to_sionna.py` | Room Sim scene + config -> Sionna scene (Mitsuba XML + PLY meshes): walls on Room Sim's bounce faces, floor and ceiling, furniture boxes, ITU materials, iso/V antennas, the 52 LLTF frequencies; backend selection; `solve_paths` (one receiver per solve, see the spec §3.4) |
| `roomsim_side.py` | Room Sim's per-link static paths (kinds 0..6) and its own §11 channel with no people and no impairments, through `RoomSim.link_model` |
| `metrics.py` | CFR, RMS delay spread, frequency selectivity, CFR agreement, path matching, distribution summaries |
| `audit.py` | the audit: scene draw, the variants, the full-model check, the slab check, JSON and markdown |

Nothing under `wifisense/`, `rust/`, `spec/golden/` or `edge/` is changed;
Room Sim is only called.
