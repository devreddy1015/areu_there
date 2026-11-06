# ADR-009: Learned models offline, for research only

**Status:** accepted, 2026-10-25, under the owner's blanket permission of
that day ("continue all the task by your self i given all permission to
do"); flagged for the owner's review (ROOMSIM.md open item 3).

## Context
The Room Sim / ML plan (ROOMSIM.md) trains a neural model on simulated
data (Phase 8), fine-tunes it on public ESP32 data (Phase 9) and on the
owner's recordings (Phase 11), and exports it (Phase 12). ADR-007 keeps
neural networks and trained classifiers out of the S0-S5 path until the
DSP version is measured on real rooms and the learned one beats it
leave-one-room-out; ADR-008 decision 9 adds that a learned estimator is
not admissible in the room demo and that beating the baseline in
simulation earns nothing. Neither says whether a model may be trained and
evaluated *offline*, and ROOMSIM.md open item 3 asked for that carve-out
to be written down before Phase 8.

## Decision
1. **Offline only.** Learned models may be trained and evaluated offline
   (`ml/`, `.venv`, the GPU) on SIM data, public data and recordings. They
   never run in the S0-S5 path, the gateway, the room stream or on a node.
   ADR-007 is unchanged for every runtime path.
2. **How results are reported.** Every number is labelled (SIM, REAL,
   SIM->REAL); splits are by scene, room or session, never by window;
   three seeds, mean ± std; the weighted-link-midpoint baseline (and any
   DSP baseline that exists for the task) on the same split; coverage
   beside every accuracy that uses abstention; an estimate is reported
   with its uncertainty.
3. **What a result earns.** A SIM result says nothing about real rooms
   (ADR-008 d9). A learned model becomes a runtime candidate only by
   ADR-007's condition — it beats the DSP measured on real rooms,
   leave-one-room-out — and through a new ADR that also settles what it
   would read (ADR-001 keeps raw CSI on the node) and who may see its
   output (ADR-005, ADR-008 d10).
4. **Exports are artefacts.** Phase 12's ONNX export and the note on
   loading it in the gateway are research artefacts; integrating one is
   the new ADR of decision 3.
5. **What a model may use.** Declared scene geometry (room size, node
   positions: SCENE) and the preprocessed CSI (spec/preprocess.md). A model
   that uses the SIM dataset's idealised empty-room calibration (`calib_*`,
   spec/dataset.md §6) is reported separately, since a deployment's
   calibration drifts.
6. **Data.** No dataset is ever committed (RUN.md §5); public datasets are
   used under their licences, and one without a licence is used only after
   its authors agree.

## Consequences
- Phases 8-12 can proceed without touching the runtime's rules.
- Any accuracy from them is a research figure with its label and its
  baseline, not a product claim.
- Moving a model into the product is a separate, later decision with its
  own evidence on real rooms.
