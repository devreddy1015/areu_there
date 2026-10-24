# Preprocessing v1 — one function for every CSI source

**Status:** spec for Phase 7, 2026-10-25. Revised after a two-lens review
(A: fitness for Phases 8, 9, 11 and honesty; B: exactness against the code
and the data): 24 findings, all adopted (one adapted to keep owner decision
2, see decision 3). The ML
pipeline's input stage: the same function prepares the simulated dataset
(spec/dataset.md), the public ESP32 datasets (Phase 9) and the owner's
recordings (Phase 11). It is ML tooling (Python with numpy, `ml/`), not a
runtime path: nothing here runs in the gateway or on a node (ADR-001,
ADR-007).

**Owner decisions, 2026-10-24** (asked and answered):
1. **Z-score statistics.** Each link and subcarrier is z-scored over the
   input window itself, and the removed mean and standard deviation are
   returned as side features (§6). In float64 the transform is lossless
   (`a' = z * sigma + mu`); the float32 outputs reconstruct `a'` to float32
   rounding (about 2e-7 relative). It is causal and needs no reference
   recording, so it is identical for SIM, public data and the owner's
   recordings; the presence information a z-score would otherwise erase
   (in SIM nobody enters or leaves during a run) moves into `mu` and
   `sigma` — partly: see §11.
2. **Output grid.** The 52 legacy-preamble (LLTF) subcarriers of HT20,
   `k = -26..-1, 1..26` in ascending order, for every source, with a
   per-bin availability mask (§3). The owner's firmware, and so the SIM
   dataset, carries 44 of them (`k = -8..-1` are never sent); the public
   ESP32-S3 sets LoFi and WiCount carry all 52; OpenCSI's 52 S3/C3 values
   are of unknown LTF and order and its C6 trial has 246 HE20 tones (Phase 9
   maps them or leaves them out).

**Decisions taken here** (flagged in the report):
1. **Where it lives:** `ml/preprocess.py` (numpy, `.venv`), with a slow
   stdlib-only reference of the same steps, `ml/preprocess_ref.py` (no
   numpy import, so `spec/golden/generate.py` on CPython 3.14 can import it
   later), that the numpy version's float64 values match (§10). No Rust port
   now: nothing consumes it at runtime. The arithmetic is normative (§4), so
   a later port can be held to golden vectors made from the reference.
2. **One canonical record, one adapter per source** (§2). Adapters know
   their source's layout; the preprocessing knows only the record.
3. **Normalisation over the shared bins.** Every statistic that pools
   subcarriers (the packet gain, §4; the phase fit, §5) uses the **reference
   bins** `R`: by default the 44 bins every source carries
   (`k = -26..-9, 1..26`), so a 52-bin source and a 44-bin source are
   normalised identically (measured, SIM: normalising the same packets
   over 52 bins instead of 44 shifts `mu` by about 3% and the phase
   residual by about 0.06 rad, more than a window's own variation). The
   other carried bins are still output (owner decision 2), scaled by the
   same gain.
4. **Phase is off by default** (amplitude is what the project's runtime
   and RespirFi use, RESEARCH.md); when on, it is sanitised by an
   unwrap-free linear-phase removal (§5), output as `(cos, sin)`, and the
   removed slope is never returned (CAMPUS_SENSING_PROMPT.md §3: never
   range from it).
5. **No resampling by default.** Each link keeps its own received packets,
   in order, with their integer times; an optional binning to a common
   rate (§7) is there for sources whose rates differ (100 Hz LoFi, about
   87 Hz WiCount, OpenCSI irregular per link with a median about 14 Hz,
   6.7–20 Hz per link in SIM). Phase 8 chooses, and a SIM->REAL comparison
   bins every source at one rate.
6. **Labels are the data's and cannot be overridden.** A window is `SIM` or
   `REAL` (what the data is; `SIM->REAL` labels results, never windows); a
   pcap's label is derived from the file (§2); provenance names the dataset,
   its impairment profile and the window.

**Rules.** No division without its guard stated before it; no NaN in any
output; masks, never sentinel values, say what is missing; nothing is
interpolated across a lost packet, and a lost packet occupies no slot. The
canonical record holds complex CSI in `(re, im)` order; every adapter
states its source's pair order and converts (a swap conjugates the CSI and
flips the phase slope, spec/roomsim.md §12.4). Times are integer
nanoseconds; float seconds are derived once.

---

## 1. The function

```python
out = preprocess(window: CsiWindow, cfg: PreprocessConfig = PreprocessConfig()) -> dict
```

`PreprocessConfig`: `phase: bool = False`; `bin_hz: float | None = None`
(§7); `ref_bins: str = "shared44"` (§3; `"carried"` uses all of a source's
carried bins, for single-source work only); `min_packets: int = 3` (§6,
never below 2). The steps, in this order:

1. **Subcarrier mask** (§3).
2. **Per-packet L1 gain normalisation** (§4).
3. **Phase sanitisation** (§5), when `phase` is on.
4. **Binning** (§7), when `bin_hz` is set.
5. **Per-link z-score** (§6) of the normalised amplitude.

## 2. The canonical record

```python
@dataclass(frozen=True)
class CsiWindow:
    label: str             # "SIM" or "REAL"
    source: str            # provenance (below)
    window_end: str        # "k=<label tick>" (SIM shards) or "t_end_ns=<n>" (pcaps)
    links: np.ndarray      # [L, 2] int16 (tx, rx); tx = -1 when unknown
    tx_known: np.ndarray   # [L] bool
    link_pos: np.ndarray   # [L, 2, 3] float64: tx and rx positions, m; 0 where unknown
    link_pos_known: np.ndarray  # [L, 2] bool
    csi: np.ndarray        # [L, P, 52] complex128, (re, im); 0 where not received or not carried
    received: np.ndarray   # [L, P] bool
    carried: np.ndarray    # [52] bool: grid bins this source carries
    t_rel_ns: np.ndarray   # [L, P] int64: packet time minus the window's end (< 0); 0 where not received
    rssi: np.ndarray       # [L, P] float64 dBm; 0 where absent
    rssi_mask: np.ndarray  # [L, P] bool
    length_ns: int         # the window's length
```

**Packets.** Each link holds its received packets only, in time order,
**right-aligned**: the newest at index `P-1`, padding (`received` false)
first; `P` is the most received packets any link has, so it varies between
windows and a slot index is not a time (`t_rel_ns` is). A lost packet
occupies no slot in any source. Links are listed in ascending `(tx, rx)`;
with a link map, every link of the map is listed even with no packet in the
window. A `CsiWindow` with a non-finite `csi` entry is refused.

**Provenance.** `source` is `"<dataset>[impairments=<on|hardware_off>]:<scene id>"`
for SIM shards and `"pcap:<sha256 of the file, 12 hex>"` for pcaps;
`window_end` names the window. Outputs copy all three.

**Adapters** (each returns `CsiWindow`s; §8 lists the layouts):
- `from_roomsim_shard(dir, scene_index, k, length_s)`: the SIM dataset
  (spec/dataset.md §4.3's integer window: per link the columns whose
  transmission `n = m*N + tx_slot[l]` lies in `[80 (t_k - ℓ), 80 t_k)`,
  `t_k = 0.5 (k + 1)`), keeping the received ones; `t_rel_ns = (n - 40(k+1))
  * 12_500_000`; `tx` known; positions from the shard's `node_pos`; label
  SIM; the impairment profile from the manifest.
- `from_pcap(path, length_s, end_ns, layout="esp32-fw56", port=5566,
  link_map=None, single_tx=False, label=None, scene=None)`: datagrams as
  `python3 -m wifisense record` and `saans roomsim-export` write them
  (spec/roomsim.md §12.4, §13), read with the stdlib reference reader
  `wifisense.sources.recorder.read_pcap` (integer `t_ns`, port filter
  `port`) and parsed with `wifisense.sources.esp32_udp.parse_esp32_frame`.
  A datagram is counted malformed and skipped when parsing raises, when
  `n_antennas == 0` or `n_subcarriers == 0`, or when `n_subcarriers` differs
  from the layout's entry count (the parser zero-fills a short payload, so
  these are not caught later). `end_ns` are integer epoch nanoseconds (tick
  `k` of a `.sim.pcap` ends at `1_790_000_000·10⁹ + 500_000_000·(k+1)`); a
  window is `[end_ns - ℓ_ns, end_ns)`, compared in integers. A datagram
  carries no transmitter, so:
  - with `link_map` (a `.sim.pcap`'s `truth.jsonl` gives `(rx, seq) ->
    link_index -> (tx, rx)`), each packet goes to its link;
  - without it, a receiver's stream is one link `(-1, rx)` only when
    exactly one transmitter is heard: a `.sim.pcap` whose manifest has
    `gateway_faithful` false is refused, and a REAL recording is read this
    way only with `single_tx=True` from the caller (`tx_known` false).
    Pooling transmitters is never silent: a pooled stream's statistics
    measure differences between transmitters, not people (measured, SIM:
    `sigma/mu` 0.28 pooled against 0.02 per link in an empty room).
  The **label**: SIM when the CLI's own export detector
  (`wifisense.cli._sim_banner`, the rule `wifisense stats` uses) finds a
  Room Sim export, and a caller's `label="REAL"` is then refused;
  otherwise the caller's label, which is required. Positions come only from
  a declared `scene` (ADR-008), never guessed. The layout check of §8 runs
  on REAL data. A datagram whose `(rx, seq)` is missing from a supplied link
  map is refused; a `.sim.pcap` without its manifest is read only with
  `single_tx=True`; malformed counts go to a `stats` argument.
- Phase 9 adds the public datasets' adapters on the same record (§8).

## 3. The subcarrier grid and mask

The grid `K = [-26..-1, 1..26]` (52 bins, ascending `k`). An adapter maps
each entry of its source to a `k` with its layout descriptor (§8) and
places it at `K.index(k)`; entries whose `k` is not on the grid (DC, guard,
HT-LTF, STBC) are dropped **by index, never by value** (spec/roomsim.md
§13: hardware nulls need not be zero). `carried[b]` is true when the source
places a value at grid bin `b`; a bin that is not carried is 0 everywhere
and excluded from every statistic. The **reference bins** `R` (decision 3)
are `k = -26..-9, 1..26` (`ref_bins = "shared44"`) or every carried bin
(`"carried"`); a source that does not carry all of `R` is refused under
`"shared44"`. Output: `k_grid` [52] int8, `carried` [52] bool, `ref` [52]
bool.

## 4. Per-packet L1 gain normalisation

For each received packet, over the reference bins `R`: `g = fsum(|h_b|) /
|R|` (an L1 norm divided by the bin count). Guard: if `g == 0` (every
reference bin exactly zero) the packet is dropped as not received (counted
in `zero_gain_packets`). Otherwise, for **every carried bin** (reference or
not), `h'_b = h_b / g` and `a'_b = |h_b| / g`.

**The arithmetic is normative** (two implementations of the earlier draft
differed in the last bits wherever it was left open: numpy's `abs`,
`arctan2` and `exp` differ from `math`'s on this machine): `|h_b| =
sqrt(re*re + im*im)` in float64 for every source (for int8 sources exact,
as Locate computes it, spec/locate.md §4.3), never `abs` or `hypot`;
`g = fsum(|h_b| in ascending k) / |R|`; `a'_b = |h_b| / g`;
`h'_b = complex(re / g, im / g)`; a real sum is `math.fsum`, a complex sum
is `complex(fsum(re), fsum(im))`; an angle is `math.atan2(im, re)` with a
result of `-π` replaced by `π` (so `-0.0` imaginary parts, whose sign
CPython 3.14 and 3.11 treat differently, cannot change an angle).

This removes any per-packet scalar gain exactly (the ESP32's AGC steps,
device gain, Room Sim's `target_counts` scaling; spec/roomsim.md §12.3)
and nothing that varies across subcarriers (ripple, multipath, people).

Output `gain_db` [L, T] float32 = `20 log10 g` (0 where masked): the
removed per-packet scale, in units of the source's own integer CSI. It is
**neither channel power nor comparable across sources**. In SIM it is
Room Sim's `target_counts` level plus the receiver's device gain, the
AGC walk and the LLTF/HT-LTF level ratio (channel power is divided out
before them, spec/roomsim.md §12.3;
measured, SIM: correlation with RSSI within a link 0.00); on hardware it is
the chip's automatic or fixed scale. Channel power survives only in RSSI.
A model that reads `gain_db` is reported as an ablation.

## 5. Phase sanitisation (when `phase` is on)

Per received packet, from `h'` (§4):
1. **Slope:** `S = Σ h'_{k+1} · conj(h'_k)` over every pair of reference
   bins with consecutive `k` (`k` and `k+1` both in `R`; pairs across the
   DC hole `-1 → 1` or the firmware's gap `-9 → 1` are not consecutive and
   never used), summed in ascending `k` with `fsum` on the real and the
   imaginary parts. Guard: `S == 0` → the packet has no phase
   (`phase_mask` false for all its bins). Otherwise `a = arg(S)` (§4's
   angle rule).
2. **Offset:** `C = Σ_{b∈R} h'_b · exp(-j a k_b)`, summed likewise; guard
   `C == 0` → no phase; `c = arg(C)`.
3. **Residual:** `r_b = h'_b · exp(-j (a k_b + c))` for every carried bin.
   Output `phase` [L, 52, T, 2] float32 = `(cos ψ_b, sin ψ_b)` with
   `ψ_b = arg(r_b)` (computed as `r_b / |r_b|`, never as a wrapped angle,
   so there is no ±π wrap and no `-0.0` ambiguity). `phase_mask` [L, 52, T]
   is false where `h'_b == 0` or `a'_b < 0.1` (a bin 20 dB below the
   packet's mean is 20–30 times noisier, measured SIM).

No unwrapping is done, so the DC hole, the firmware's gap and exact-π ties
cannot make two implementations disagree; `S` and `C` are amplitude-
weighted, so near-zero bins barely count. `a` and `c` are **not** output:
the slope mixes the random timing offset with the physical delay
(RESEARCH.md: never range from it). Adding any common phase and any
timing slope in true `k` to every packet leaves `phase` unchanged.

## 6. Per-link z-score (owner decision 1)

Over the window, per link `l` and carried bin `b`, over the link's `n`
received packets (or non-empty bins, §7): `mu = fsum(a'_b) / n`,
`sigma = sqrt(fsum((a'_b - mu)^2) / (n - 1))` (unbiased: a population
`sigma` would scale with `sqrt((n-1)/n)` and so drift with node count,
window length and rate, which Phases 8 and 9 vary). Guards, in order:
- `n < min_packets` (default 3, never below 2; at 2 every `z` is ±1/√2) → the
  link is not used in this window (`link_mask` false; its `amp`, `mu`,
  `sigma` 0; counted in `links_dropped`);
- every received `a'_b` of the bin equal (compared exactly, on the
  values, before any arithmetic; a computed `sigma` of equal values is
  often 1e-16, not 0) → `mu` = that value, `sigma = 0`, `z = 0`;
- otherwise `z = (a'_b - mu) / sigma`.

Outputs: `amp` [L, 52, T] float32 (`z`; 0 where masked or not carried),
`mu` [L, 52] float32, `sigma` [L, 52] float32, `n_packets` [L] int32,
`sample_mask` [L, T] bool = received and `g != 0` and `link_mask`.
`z * sigma + mu` returns `a'` to float32 precision (within
`2^-22 · (|z| sigma + mu)`).
Phase (§5) and RSSI are not z-scored: `rssi` [L, T] float32 in dBm with
`rssi_mask` [L, T] bool, since its level is information the model may use.

## 7. Binning (optional, `bin_hz`)

`nb = bin_hz · length_s` must be an exact integer (`bin_hz` and
`length_s` are read as the decimals their floats print as, so 0.1 Hz is
exactly 1/10) and `length_ns` a
multiple of `nb` (else refused); `w_ns = length_ns / nb`. A packet goes to
bin `(t_rel_ns + length_ns) // w_ns`, in integers (SIM packets sit on a
12.5 ms grid, so many fall exactly on bin edges; float edges put them in
different bins in different adapters). A bin's amplitude is the **mean of its packets'
amplitudes** `a'_b`, never the magnitude of a complex mean (every packet
carries a random common phase and timing slope until §5 removes them;
measured, SIM: a complex mean kept 0.4–0.6 of the amplitude and raised
`sigma/mu` twentyfold, equally in empty and occupied rooms). With `phase`
on, a bin's phase is `arg(Σ_p r_{p,b})` over its packets with `phase_mask`
true, masked when that sum is 0.
Its `gain_db` is the mean of theirs and its `rssi` the mean of those with
one. Every mean here is `fsum / n`, except that a mean of values that
are all exactly equal is that value (`fsum([x]*n)/n` is not always `x`,
which would defeat §6's equality guard). `bin_count` [L, nb] int32 is output (averaging `n` packets lowers the
noise about `sqrt(n)`). A bin with no packet is masked. `T = nb`, and
`t_rel` is the bin centre.

## 8. Source layouts

A layout descriptor says, for each entry of a source's CSI array, its
subcarrier `k` (or none) and the pair order:

| Layout | Entries | `k` of entry `e` | Pairs | Used by |
|---|---|---|---|---|
| `esp32-fw56` | 56 | LLTF FFT order: `e` for `e < 32`, `e - 64` for `32 <= e < 56` (bins `0..31, -32..-9`); on the grid when `1 <= |k| <= 26` | (imag, real) | the owner's firmware; Room Sim's datagrams and the SIM dataset's shards |
| `esp32-lltf64` | 64 | FFT order `e` / `e - 64` | (imag, real) | the full LLTF block (`fw_subcarriers = 64`) |
| `esp-radar-52-asc` | 52 | `-26..-1, 1..26` | per dataset | esp-radar v0.3 and every secondary-channel capture (LoFi, WiCount) |
| `esp-radar-52-pos-first` | 52 | `1..26, -26..-1` | per dataset | esp-radar v0.1–0.2 without a secondary channel |

(Room Sim's in-memory `Packet.buffer` is `(re, im)`; its datagrams and the
shards are `(imag, real)`.) The ESP-IDF buffer is LLTF, then HT-LTF, then
STBC-HT-LTF, each block in FFT order (k = 0..31, then -32..-1) without a
secondary channel; only LLTF is on the grid.

**Layout check (REAL data).** The `esp32-fw56` layout is unverified on
hardware (spec/roomsim.md §4) and a datagram does not say which layout it
used. `from_pcap` checks it on the data, per receiver and recording: the
mean power of the guard entries `e = 27..37` must be at least 20 dB below
that of the active entries (a threshold Phase 11 confirms on its first
capture), else the recording is refused naming the receiver. Masking stays
by index; this is a check only. (A datagram whose `n_subcarriers` is not
the layout's is skipped as malformed, §2, so one window never mixes
layouts.) Phase 9 checks each public
dataset against these layouts (its research found the authors' processed
files pair each imaginary value with the previous subcarrier's real
value; adapters read the raw CSI).

## 9. Outputs

`preprocess` returns a dict: `label`, `source`, `window_end`, `links` [L,
2] int16, `tx_known` [L] bool, `link_pos` [L, 2, 3] float32,
`link_pos_known` [L, 2] bool, `k_grid` [52] int8, `carried` [52] bool,
`ref` [52] bool, `amp` [L, 52, T] float32, `mu`, `sigma` [L, 52] float32,
`n_packets` [L] int32, `sample_mask` [L, T] bool, `link_mask` [L] bool,
`gain_db` [L, T] float32, `rssi` [L, T] float32, `rssi_mask` [L, T] bool,
`t_rel` [L, T] float32 (seconds, from `t_rel_ns`; bin centres with
`bin_hz`), with `bin_hz` also `bin_count` [L, T] int32, and with `phase` on
`phase` [L, 52, T, 2] float32 and `phase_mask` [L, 52, T] bool; plus counts
`zero_gain_packets`, `links_dropped`. **No output contains NaN.**

## 10. Tests (`ml/tests/`)

- **Reference:** `ml/preprocess_ref.py`, a stdlib-only implementation of
  §3–§7 with §4's arithmetic (`math.fsum` for every sum, never builtin
  `sum`, whose rounding changed in CPython 3.12). The numpy version's
  float64 values **before the float32 cast** match it: shapes, masks and
  counts exactly; `mu`, `sigma`, `gain_db`, `t_rel` within
  `1e-12 · max(1, |ref|)`; `z` within `1e-12 · (1 + |z| + mu/sigma)`; phase
  components within 1e-12 (absolute); and the float32 outputs within one
  ulp of the reference's cast — on random windows and on windows of the SIM
  dataset (if `SAANS_DATASET_DIR` is set).
- **Gain invariance:** multiplying any packet by any positive scalar `c`
  leaves `amp`, `mu`, `sigma` and `phase` unchanged within those
  tolerances (bit for bit for a power-of-two `c`), and shifts that packet's
  float64 `gain_db` by `20 log10 c` within `1e-12 · max(1, |gain_db|)` dB.
- **Phase:** adding any common phase and any timing slope in true `k` to
  every packet leaves `phase` unchanged; the DC hole and the `-8..-1` gap are
  never bridged; a bin below `a' = 0.1` is masked; an exact `-π`/`π`
  case fed with both signs of zero gives `π`.
- **Layouts:** each descriptor maps hand-built entries to the right grid bins
  and pair order; `esp32-fw56` reproduces the dataset's `subcarrier_k` and
  `subcarrier_active`; the layout check refuses a synthetic recording with
  energy in the guard entries.
- **Z-score:** `z * sigma + mu` reconstructs `a'` (float64 exactly to the
  tolerance; the float32 outputs to 2e-7 relative); the guards behave as §6
  says, including a constant bin whose computed variance is not exactly
  0; a malformed datagram (zero antennas, short payload, wrong subcarrier
  count) is skipped and counted; a non-finite CSI entry is refused.
- **One function for all data:** a dataset scene exported with `saans
  roomsim-export --config scenes/gen-XXXXXX.roomsim.json` and read back by
  `from_pcap` with its `truth.jsonl` link map gives `CsiWindow`s
  **identical in every field but `source` and `window_end`** to
  `from_roomsim_shard` on the same scene and ticks (same received packets,
  same `t_rel_ns`, same CSI), and so identical `preprocess` outputs, when
  the pcap is given the shard's node positions as its declared geometry
  (the shard stores them as float32, a scene file as float64: with the
  scene file `link_pos` differs by under 1e-6 m and every other field is
  still identical); the same pcap without a link map is refused (not gateway-
  faithful), and its label is SIM even when the caller says REAL (refused).
- **Binning:** packets land in the right integer-ns bins; empty bins are
  masked; amplitude is averaged, not the complex value.
- Throughput measured and reported (windows per second on this machine).

## 11. Limits

- Gain normalisation removes per-packet scalar gain only; the static
  per-device ripple and the multipath stay, and z-scoring over the window
  removes their mean, keeping it in `mu`.
- The phase residual keeps the static ripple's phase and the multipath; it
  is relative phase, not range (decision 4).
- **What `sigma` keeps of presence** (measured, SIM, 2 s windows): walking
  vs empty AUC 0.94, still vs empty 0.63. A still person differs from an
  empty room mainly in `mu`'s level, which nothing inside the window
  calibrates.
- **SIM-only regularities survive preprocessing:** reciprocal links are
  near-identical (`mu` within 0.7%, RSSI within 0.02 dB, measured SIM, where
  real boards differ by dB); `gain_db` is an invented AGC walk; the phase
  impairment is only a common phase plus a linear slope; SIM always carries
  44 bins.
- REAL `(-1, rx)` links can include ambient frames (the node has no MAC
  filter); the layout check (§8) catches a wrong layout, not stray
  transmitters.
- Window statistics need `min_packets` per link: short windows at low rates
  drop links (`links_dropped`).
- The public adapters, their licences (LoFi and WiCount state none) and
  their label alignment are Phase 9's.
