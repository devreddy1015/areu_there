# ml/ — dataset v1 tooling (SIM)

Python 3.11 with numpy and matplotlib, run in `.venv`. Reads what
`saans gen-dataset` writes (spec/dataset.md). Every scene, packet, label
and baseline in it is **SIM**: not a room anyone measured.

```sh
.venv/bin/python ml/plot_sample.py DIR --scene 0 --window 3 --out sample.png [--links 3]
.venv/bin/python -m unittest discover -s ml/tests
SAANS_DATASET_DIR=DIR .venv/bin/python -m unittest discover -s ml/tests   # also checks every real shard
```

```python
from ml import roomsim_dataset as rd

manifest = rd.load_manifest(DIR)                 # refuses anything not labelled SIM / "dataset v1"
split = rd.splits(manifest)                      # {"train", "val", "test"}: by scene, never by window
scene = rd.load_scene(DIR, split["train"][0], manifest=manifest)   # sha256, label and §7.2 checked
k = int(scene["window_end_tick"][w])             # the brief's 2 s window w ends at tick 4w + 3
win = rd.window(scene, k, 2.0)                   # amplitude [L, 56, T] float32, mask, link_mask, column, inside
truth = rd.labels_at(scene, k)                   # presence, count, zone_occupied, people
wlm = rd.baseline_at(scene, k)["wlm"]            # centre and radius together, or the abstention reason
rd.impairments(manifest)                         # "on" or "hardware_off" (§8); scene["dataset"] has it too
```

Windows are never resampled. Slot `j` of every link holds that link's one
transmission in the `j`-th of `T = ceil(80ℓ/N)` superframe-long intervals
ending at `t_k`, so the newest packet is always slot `T - 1`. `inside`
marks the slots whose transmission is in the window (`T` or `T - 1` per
link); `mask` the ones received. A masked slot's amplitude is NaN unless
`window(..., fill=0.0)`. Null subcarriers are not masked: use
`scene["subcarrier_active"]`.

Before training (spec/dataset.md §6, §12):

- **AGC gain steps.** With impairments on, Room Sim's AGC steps a link's
  gain by about ±3 dB between packets, uniformly across subcarriers, in
  about 17% of 2 s link-windows, empty room or not. Raw amplitude variance
  tracks those steps as much as people: normalise each packet's gain
  (e.g. divide by its mean over active subcarriers) first. That is
  Phase 7's job; `window` returns raw amplitude. The sample plot ranks
  links by the normalised variance.
- **`calib_*` is an idealised empty room**: the same seed, clutter and
  noise streams as the occupied run, with no drift. It is there to
  recompute the WLM baseline; a model that uses it sees more than a
  deployment would and is reported separately.
- **Baseline validity** is the manifest's `baseline_valid`
  (`baseline_at(...)["valid"]`); the plot says "baseline not valid for
  this dataset" instead of WLM/RTI outcomes when it is false.
