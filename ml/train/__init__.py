"""Phase 8 (spec/model.md): SIM pre-training of model v1 — the cache, data,
model, training, evaluation, baselines, metrics and sweep. Every number
these modules produce is SIM (ADR-008 d9, ADR-009).

Data side: `cache` (every second label tick's window preprocessed once into
float16 memory maps, with the labels and the stored WLM/RTI), `data` (the
Dataset, the training sampler and link-subset rules, the batch of
spec/model.md §1), `metrics` (the predictions contract and every §4 figure
with the scene bootstrap), `baselines` (stored and recomputed WLM/RTI, the
variance-feature forest and MLP, the prior). Model side: `model`, `train`,
`evaluate`, `sweep`."""
