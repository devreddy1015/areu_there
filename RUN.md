# RUN.md — starting, monitoring and resuming jobs on this machine

Working notes for the GPU box reached over SSH. Written during Phase 0
(2026-10-14). Everything here was measured on the machine, not assumed.

## 1. The machine

| | |
|---|---|
| GPU | Quadro RTX 8000, **sm_75 (Turing)**, 47.5 GiB |
| Driver / CUDA | 545.23.08 / 12.3 |
| CPU / RAM | 96 cores / 503 GB |
| Disk | ~155 GB free on `/` (92% used) |
| OS | Ubuntu 22.04.2, glibc 2.35 |
| tmux | 3.2a |

### The GPU is shared

A second user's job (`python train.py --datasets snapchat`, PID 551428,
running 36+ days) holds ~14.9 GB and sits at 100% SM utilisation. Budget
for **~32 GB and contended compute**, not 48 GB. Check before launching:

```bash
nvidia-smi --query-compute-apps=pid,used_memory --format=csv
```

### bf16 is a trap on this card

Turing has no bf16 tensor cores. `torch.cuda.is_bf16_supported()` returns
`True` anyway, because it reports emulated support. Measured here
(4096³ matmul, GPU contended, so read the ratio not the absolute):

| dtype | ms | TFLOP/s |
|---|---|---|
| fp16 | 3.69 | **37.2** |
| fp32 | 24.71 | 5.6 |
| tf32 | 24.74 | 5.6 (no TF32 below sm_80) |
| bf16 | 45.12 | **3.0** |

**Use fp16 autocast + `torch.amp.GradScaler`.** bf16 here is 12x slower
than fp16 and slower than plain fp32.

### Egress is flaky, not blocked

`pypi.org` and `static.crates.io` reset connections roughly one attempt in
three (TLS `ECONNRESET` at connect). Downloads succeed on retry; no mirror
is needed. The venv ships a `pip.conf` with `retries = 15`, and any script
that fetches should loop. If a `cargo` or `pip` command dies mid-download,
just run it again.

A truncated `rustup` install leaves `error: missing manifest in toolchain`.
Recover with:

```bash
rustup toolchain uninstall stable && rustup toolchain install stable --profile minimal --component clippy --component rustfmt
```

## 2. Toolchains

Rust was installed with `--no-modify-path`, so **your shell rc was not
touched**. Put it on PATH yourself:

```bash
export PATH="$HOME/.cargo/bin:$PATH"
```

rustc 1.99.0, cargo 1.99.0, clippy and rustfmt. The workspace declares
`rust-version = "1.80"`, but `saans-sim/tests/parity.rs` and
`examples/scan_seats.rs` use `slice::as_chunks::<8>()`, which needs a much
newer toolchain — the crates' `src/` honours the MSRV, the tests do not.

Python lives in a venv at the repo root (git-ignored):

```bash
source /home/Aman/are_u_theone/are_U_there/.venv/bin/activate
```

| package | version | for |
|---|---|---|
| torch | 2.5.1+cu121 | pinned: the build already proven on this driver |
| numpy / scipy / pandas / pyarrow | 2.4.6 / 1.17.1 / 3.0.6 / 26.0.0 | data |
| scikit-learn | 1.9.1 | baselines |
| onnx / onnxruntime | 1.23.2 / 1.31.0 | Phase 12 export + check |
| tensorboard | 2.21.0 | training curves |
| matplotlib | 3.11.2 | plots |
| jsonschema | 4.26.0 | validating `spec/scenes/*.json` |
| sionna-rt / mitsuba / drjit | 2.2.0 / 3.9.1 / 1.5.0 | Phase 10 RT audit |

The repo's own Python (`wifisense/`, `tests/`) is **pure standard library
by policy** and must keep running on bare `python3`, outside this venv.

## 3. Running things

### Tests of record

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd rust && cargo test --workspace          # all Rust crates
cd /home/Aman/are_u_theone/are_U_there && python3 -m unittest discover -s tests
make -C edge/dsp/test check                 # C++ edge DSP against golden vectors
```

### Known pre-existing failures

As of commit `a9c5abb`, the Python suite reports **459 tests, 2 failures +
1 error** on this machine. All three reproduce on a clean clone of HEAD
with no local changes, so they are not caused by new work:

| test | why |
|---|---|
| `test_synthetic.TestRandomSceneUnchanged.test_default_output_is_bit_identical_to_original` | SHA-256 bit-identity check. This box runs CPython 3.11.5; from 3.12 `sum()` over floats is Neumaier-compensated, which moves the last bits. NOTES.md §7.2 documents the sensitivity. |
| `test_localize_invert.TestRegularizationSmooths.test_the_conditioning_trend_stops_past_the_room_scale` | numerical conditioning trend |
| `test_recorder.TestCli.test_replay_send_over_udp` | UDP loopback in this environment |

Re-check the baseline before blaming your own change:

```bash
git clone /home/Aman/are_u_theone/are_U_there /tmp/pristine && cd /tmp/pristine
python3 -m unittest discover -s tests
```

### Long jobs: always in tmux

SSH drops kill foreground jobs. tmux survives them.

```bash
tmux new -s train                  # start
#   detach:  Ctrl-b then d
tmux ls                            # list sessions
tmux attach -t train               # resume after a disconnect
tmux kill-session -t train         # stop
```

Run the real work inside, and tee the log so a dead pane still leaves a
record:

```bash
tmux new -s train
source .venv/bin/activate
python -u train.py ... 2>&1 | tee -a runs/train-$(date +%F-%H%M).log
```

### Monitoring

```bash
watch -n 5 nvidia-smi                       # GPU, yours and the neighbour's
tail -f runs/train-*.log                    # progress
tensorboard --logdir runs --port 6006       # curves; then from your laptop:
#   ssh -N -L 6006:localhost:6006 <user>@<this-host>
```

### Resuming

Checkpoint every epoch to `runs/<name>/ckpt-<step>.pt` and keep `last.pt`
as a symlink, so a killed job restarts with
`python train.py --resume runs/<name>/last.pt`. Nothing written yet
depends on this; it is the convention the training phases should follow.

## 4. Scenes (Phase 1)

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd rust && cargo test -p saans-scene
```

The scene contract is `spec/scene.schema.json`; the three worked examples
are `spec/scenes/{bedroom,office,hall-l-shaped}.json`. Validate them
against the schema from the venv:

```bash
source .venv/bin/activate
python - <<'PY'
import json, jsonschema
schema = json.load(open("spec/scene.schema.json"))
for name in ("bedroom", "office", "hall-l-shaped"):
    jsonschema.validate(json.load(open(f"spec/scenes/{name}.json")), schema)
    print(name, "ok")
PY
```

### Importing a room from an iPhone scan

Scan the room with any RoomPlan app that can export `CapturedRoom` JSON,
copy the file over, then:

```bash
cd rust && cargo build --release -p saans-cli
./target/release/saans import-roomplan ~/bedroom-scan.json --out ../spec/scenes/mine.json --name "my bedroom"
```

It prints every assumption it had to make. RoomPlan measures geometry and
nothing else, so **the materials, the wall thickness and all the node
positions are yours to fill in** - the written scene deliberately does not
load until you add at least one node:

```
  assumed: RoomPlan surfaces are planar, so all 4 wall(s) were given thickness 0.1 m
  assumed: materials are assumed, not measured: 4 wall(s) set to drywall ...
  assumed: a scan contains no ESP32 nodes: add them to `nodes` before this scene will load
  this scene does not load yet (1 problem(s) to fix by hand):
    - nodes is empty: a scene with no ESP32 has nothing to simulate
```

Edit the file, then re-run the schema check above against it.

## 5. Datasets

Nothing is downloaded yet. When Phases 9 and 11 need LoFi, OpenCSI,
WiCount or your own pcaps: **no dataset is ever committed**. Keep them
under `saans-data/`, which `.gitignore` already excludes along with
`*.pcap`, for the health-data reasons in ADR-001 and ADR-005.
