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
`rust-version = "1.80"` in `[workspace.package]`, but no crate sets
`rust-version.workspace = true`, so cargo enforces nothing. Everything
here was built and tested on 1.99 only; the code (and certainly
`saans-sim`'s tests, which use `slice::as_chunks`) needs a much newer
toolchain than 1.80. Treat the declared MSRV as aspirational.

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

**Which `python3` you get depends on the shell.** An interactive shell
runs conda's init from `~/.bashrc` and gets miniconda's **3.11.5**; a
non-interactive one (cron, `ssh host cmd`, tool runners) gets
`/usr/bin/python3`, **3.10.12**. Results differ between them (see *Known
pre-existing failures*), so scripts and reports should name the
interpreter: `.venv/bin/python` is 3.11.5, `$(.venv/bin/uv python find
3.12)` is 3.12.15, `$(.venv/bin/uv python find 3.14)` is 3.14.8.

## 3. Running things

### Tests of record

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd rust && cargo test --workspace          # all Rust crates
cargo test -p saans-gateway --features roomsim   # the room demo's tests exist only here
cargo check -p saans-roomsim --locked && cargo test -p saans-roomsim --locked   # alone, as CI does (see 4c)
cd /home/Aman/are_u_theone/are_U_there && python3 -m unittest discover -s tests
make -C edge/dsp/test check                 # C++ edge DSP against golden vectors
```

On this box `make check` runs **g++ only**: clang++ is not installed, the
Makefile prints "clang++ not installed, skipping", and its summary banner
still says "(g++ and clang++)… under both toolchains". The banner is
wrong; the g++ result is real.

### Known pre-existing failures

As of commit `a9c5abb`, the Python suite reports **459 tests, 2 failures +
1 error** on this machine. All three reproduce on a clean clone of HEAD
with no local changes, so none is caused by new work — and on inspection
none is a bug in the code under test:

| test | cause | verified how |
|---|---|---|
| `test_synthetic.TestRandomSceneUnchanged.test_default_output_is_bit_identical_to_original` | CPython version. The hash was taken on CPython ≥ 3.12, where float `sum()` is Neumaier-compensated (NOTES.md §7.2); this box's default `python3` is 3.11.5. | **Passes** under Python 3.12.15. |
| `test_localize_invert.TestRegularizationSmooths.test_the_conditioning_trend_stops_past_the_room_scale` | CPython version, same mechanism. | **Passes** under Python 3.12.15. |
| `test_recorder.TestCli.test_replay_send_over_udp` | Kernel receive-buffer limit. The test asks for a 1 MB `SO_RCVBUF`, but `net.core.rmem_max` here is 212,992, so the socket gets 425,984 bytes; the replay then sends 598 datagrams in one burst (`--speed 0`) and 43 are dropped. | Measured: 598 sent, 555 received. Would pass with a larger `rmem_max`, or if the test read while sending. Not changed: it is not this work's test, and this is a shared machine. |

So the suite to trust here is the one run under Python 3.12. (To
*regenerate* golden vectors, use CPython 3.14 instead — see ROOMSIM.md,
"Changing the simulator"; 3.14 is the only version that reproduces the committed
vectors byte for byte.)

```bash
.venv/bin/pip install uv && .venv/bin/uv python install 3.12
$(.venv/bin/uv python find 3.12) -m unittest discover -s tests
```

which leaves exactly one expected failure, the UDP one. Re-check the
baseline on a clean clone before blaming your own change:

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

## 4b. The room demo (Phase 2)

A static three.js view of the example scenes, served by the gateway at
`/roomsim` (ADR-008). It draws declared geometry only: no people, no
estimates, no sensor data. It is compiled in only with the `roomsim`
cargo feature, which is off by default (ADR-008 decision 13); a normal
build answers 404 there, and a test proves it.

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd rust && cargo run -p saans-gateway --features roomsim -- --config crates/saans-gateway/roomsim-demo.toml
```

Then open `http://127.0.0.1:18080/roomsim`. The gateway is bound to
loopback, so from your laptop tunnel to it rather than widening
`http_addr` (a non-loopback address turns authentication on):

```bash
ssh -N -L 18080:127.0.0.1:18080 <user>@<this-host>
```

Links are shareable and Back/Forward work: `#scene=office`,
`#scene=hall-l-shaped&view=top`, `#dims=off`, `#select=almirah`. With the
view focused, arrow keys pan, Shift/Ctrl+arrows rotate, `+`/`-` zoom and
`0` resets; every object, node and wall is also selectable from the list.

### Checking it in a real browser

The Rust tests prove the page is *served* correctly; this proves it
*works* — WebGL draws, the CSP blocks nothing it needs, clicks select,
nothing overflows at 375 px. It drives the system Chrome, so no browser
download is needed:

```bash
.venv/bin/pip install playwright pillow
.venv/bin/python -I rust/crates/saans-gateway/tests/browser/roomsim_check.py --base http://127.0.0.1:18080
```

It exits non-zero on any failure and writes screenshots to
`roomsim-shots` in Python's temp directory (`/tmp` here; pass `--out` to
choose). Look at them: a page can pass every check and
still look wrong, and two of the bugs fixed while building this page were
found by looking, not by checking.

### Upgrading three.js

See `rust/crates/saans-gateway/assets/vendor/three-0.186.1/VENDOR.md`. The
vendored files are pinned by SHA-256 in `src/roomsim.rs`, so an upgrade
fails the tests until the record is updated with it.

## 4c. Room Sim (Phase 3)

The simulator is specified in `spec/roomsim.md` and implemented twice,
bit for bit: the Python reference `wifisense/roomsim/` and the Rust crate
`rust/crates/saans-roomsim`. Everything it produces is **SIM**. It reads a
scene (SCENE) and writes what ESP32 nodes in that room would send.

```bash
export PATH="$HOME/.cargo/bin:$PATH"
cd rust && cargo build --release -p saans-cli
./target/release/saans roomsim-export ../spec/scenes/bedroom.json --out /tmp/bedroom --duration 60
```

That writes three files and refuses to overwrite them (add
`--overwrite`):

| File | What |
|---|---|
| `/tmp/bedroom.sim.pcap` | the raw-CSI datagrams, byte for byte what `csi_capture.cpp` sends, in `wifisense record`'s pcap format |
| `/tmp/bedroom.truth.jsonl` | one line per transmission: where every person and the phone were, and each packet's fate (lost, seq, rssi) |
| `/tmp/bedroom.manifest.json` | `"label": "SIM"`, the scene's name and SHA-256, every configuration value, per-link and per-receiver rates, counts |

Options: `--seed N`, `--duration S`, `--config CONFIG.json` (any parameter
of spec §3.2, e.g. `{"loss_prob": 0.1, "roles": {"0": "tx"}}`, except
`quantize: false`, which an export refuses because it emits no datagram).
The same from Python:

```python
from wifisense.roomsim import load_scene, RoomSim, Config, export_run
sim = RoomSim(load_scene("spec/scenes/bedroom.json"), Config(duration_s=60.0))
export_run(sim, "/tmp/bedroom-py")      # byte-identical pcap to the CLI's
```

**Gateway-faithful exports.** The default is the planned all-node mesh:
every node transmits in turn, so each receiver's datagrams interleave
several transmitters with no transmitter id, and nothing downstream can
split them per link (the CLI warns). Today's firmware is one transmitter
at 20 Hz; to replay through the gateway or `wifisense replay`, export
that (one transmitter at another rate also works, if the gateway's
`raw.sample_rate_hz` is set to it; the CLI notes this):

```bash
cd /home/Aman/are_u_theone/are_U_there
echo '{"roles": {"0": "tx", "1": "rx", "2": "rx", "3": "rx"}, "slot_s": 0.05}' > /tmp/one-tx.json
rust/target/release/saans roomsim-export spec/scenes/bedroom.json --out /tmp/bedroom-1tx --config /tmp/one-tx.json
python3 -m wifisense stats /tmp/bedroom-1tx.sim.pcap     # prints a SIM banner first
```

Its `seq` has no gaps for lost packets, as on the real firmware, so
`stats` shows 0 % loss; the manifest's `counts.lost` is the true figure.

Tests (the standalone `-p saans-roomsim` build matters: inside the
workspace cargo unifies features and would hide a dependency on
`saans-scene`'s `write`):

```bash
cd /home/Aman/are_u_theone/are_U_there/rust && cargo test -p saans-roomsim --locked && cargo test -p saans-cli
cd .. && python3 -m unittest discover -s tests -p 'test_roomsim*'
```

`saans-roomsim/tests/golden.rs`, `saans-cli/tests/roomsim_export.rs` (the
`roomsim_export` set) and `tests/test_roomsim_golden.py` check both
implementations against `spec/golden/roomsim_*`. A physics change goes
into `spec/roomsim.md`, the Python reference and the Rust crate together,
then the vectors are regenerated **on CPython 3.14** and both tests must
pass: `$(.venv/bin/uv python find 3.14) -I spec/golden/generate.py`, then
`git status spec/golden` should show only the sets you meant to change.

Speed (release, one thread): about 1 s of compute per minute of simulated
bedroom, 3 s per minute of the 8-node hall, a few MB of memory however
long the run; the Python reference takes about 15 s per 10 s of bedroom and
is for checking, not for datasets.

## 5. Datasets

Nothing is downloaded yet. When Phases 9 and 11 need LoFi, OpenCSI,
WiCount or your own pcaps: **no dataset is ever committed**. Keep them
under `saans-data/`, which `.gitignore` already excludes along with
`*.pcap`, for the health-data reasons in ADR-001 and ADR-005.
