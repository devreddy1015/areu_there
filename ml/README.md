# ml/ — dataset v1 tooling (SIM) and shared preprocessing

Python 3.11 with numpy and matplotlib, run in `.venv`. Reads what
`saans gen-dataset` writes (spec/dataset.md). Every scene, packet, label
and baseline in it is **SIM**: not a room anyone measured. The shared
preprocessing (below) also reads recordings, which are **REAL**, and keeps
each window's label.

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

## Preprocessing (spec/preprocess.md, Phase 7)

`ml/preprocess.py` is the one input stage for every CSI source: the SIM
shards, Room Sim's `.sim.pcap` exports, and (Phases 9 and 11) public
datasets and the owner's recordings, which are **REAL**. Every window and
every output carries its data's label, `"SIM"` or `"REAL"`, and nothing
can change it: a shard is SIM; a pcap is SIM when it is a Room Sim export,
is read with a link map, or comes entirely from Room Sim's
`10.1.<rx>.1:5566` (a caller who says REAL is then refused), and otherwise
needs the caller's label. A REAL stream is checked, not trusted: a node id
from two addresses, a clock step backwards and a receiver sending mostly
another layout are refused; exact duplicates are dropped and counted
(`pp.PcapStats`).

```python
from ml import preprocess as pp

win = pp.from_roomsim_shard(DIR, scene_index, k, 2.0)          # SIM window ending at tick k (a list for a list of k)
wins = pp.from_pcap("run.sim.pcap", 2.0, [pp.sim_end_ns(k) for k in ticks],
                    link_map=pp.truth_link_map("run.truth.jsonl"))          # the same SIM run, read as datagrams
rec = pp.from_pcap("rec.pcap", 2.0, end_ns, label="REAL", single_tx=True)  # a recording: one transmitter heard
out = pp.preprocess(win, pp.PreprocessConfig(phase=False, bin_hz=None, ref_bins="shared44", min_packets=3))
out["amp"], out["mu"], out["sigma"]       # [L, 52, T] z-scores, [L, 52] the removed mean and std
out["sample_mask"], out["link_mask"]      # what is real: masks, never sentinels; no output is NaN
```

The record (`CsiWindow`) holds complex CSI in `(re, im)` order on the 52
LLTF bins `k = -26..-1, 1..26` with a `carried` mask (SIM carries 44),
each link's **received packets only**, right-aligned (newest last), with
integer-nanosecond times; a lost packet occupies no slot and nothing is
interpolated. `preprocess` then: normalises each packet's gain over the
reference bins (`gain_db` keeps the removed scale, which is not channel
power), optionally removes each packet's common phase and timing slope
(`phase`, as `(cos, sin)`; the slope is never returned), optionally bins
to integer-ns bins (`bin_hz`), and z-scores each link and subcarrier over
the window. Float outputs are float32; `dtype=np.float64` gives the values
before the cast.

`ml/preprocess_ref.py` is the same computation in stdlib Python (no numpy:
the golden-vector generator can import it); the tests hold the numpy
version to it. A cross-path test exports a dataset scene with
`saans roomsim-export` and checks the pcap path gives the shard path's
windows bit for bit; it runs when `SAANS_DATASET_DIR` is set and finds the
binary in `SAANS_BIN`, `$CARGO_TARGET_DIR/release/saans`,
`rust/target/release/saans` or `PATH`. Measured on SIM 2 s windows (4, 8 and
12 nodes, one core): about 390 windows/s (2.3 µs per packet) by default,
110 windows/s with phase, 230 windows/s binned at 10 Hz; the shard adapter
about 390 windows/s.
