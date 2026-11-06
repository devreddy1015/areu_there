# Loading the ONNX export with `ort` (a note, not an integration)

**Status:** Phase 12 note, 2026-10-25. **Nothing here is wired into the
gateway**, and nothing should be until a new ADR says so. ADR-009 decision
3 lets a learned model become a runtime candidate only once it beats the
DSP baseline measured on real rooms, leave-one-room-out. That ADR must also
settle two questions:
- where the model runs, given that ADR-001 keeps raw CSI on the node;
- who may see its output (ADR-005, ADR-008 d10).

## The artefact

`ml/export/onnx_export.py` writes one ONNX file per checkpoint (opset 17).
The batch `B` and link count `L` are dynamic. The bin count `T = 5 ℓ` and
the number of reference bins `R` (44 or 40) are fixed by the checkpoint.

| Input | Shape | dtype |
|---|---|---|
| `amp` | [B, L, R, T] | f32 |
| `mu`, `sigma` | [B, L, R] | f32 |
| `sample_mask` | [B, L, T] | bool |
| `link_mask` | [B, L] | bool |
| `rssi` | [B, L, T] | f32 |
| `rssi_mask` | [B, L, T] | bool |
| `link_pos` | [B, L, 2, 3] | f32 |
| `room` | [B, 3] | f32 |
| `geometry_known` | [B] | bool |

| Output | Shape |
|---|---|
| `presence_logit` | [B] |
| `count_logits` | [B, 4] |
| `zone_logits` | [B, 9] |
| `slot_logit` | [B, 3] |
| `slot_xy` | [B, 3, 2] (m) |
| `slot_sigma` | [B, 3] (m) |

The export check runs onnxruntime against PyTorch in float32. The largest
absolute difference is about 2e-7 against the brief's 1e-4
(`ml/tests/test_export.py`; ML_RESULTS.md has the figures for the trained
checkpoints).

## What a Rust consumer would need

1. **The `ort` crate** (2.x, ONNX Runtime bindings). It needs no `unsafe`
   in the caller. Build an environment and a `Session` from the file, then
   feed `ndarray` arrays by input name and read the outputs by name. The
   CPU execution provider is enough: the model is under 3M parameters and
   one window is a few MFLOPs.
2. **The preprocessing, ported exactly.** The model eats spec/preprocess.md's
   outputs, not raw CSI:
   - per-packet L1 gain over the reference bins;
   - 5 Hz integer-ns binning;
   - the per-link z-score with `mu` and `sigma` kept.

   `ml/preprocess_ref.py` is the stdlib transcription a port is checked
   against (§10). Port it first and check it against the Python on a
   recorded window before trusting a single prediction.
3. **The decision rules** (spec/model.md §4), after the logits:
   - presence at sigmoid ≥ 0.5;
   - count by argmax;
   - each reported slot with its 95% radius `max(2.448 σ, 0.3)` m.

   A slot is shown only as an estimate with that radius (ADR-008).
4. **Geometry.** `geometry_known` is false unless node positions come from
   a declared scene (ADR-008 d1). With it false, `link_pos` and `room` are
   ignored and zone and xy are not meaningful.

## What it must not do

- It must not run on the S0–S5 path, in the gateway, or on a node, until
  the ADR above exists (ADR-007, ADR-009 d1).
- It must not present any SIM or SIM->REAL figure as a product property.
  Every number travels with its label.
