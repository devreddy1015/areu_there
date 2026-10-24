"""Preprocessing v1 (spec/preprocess.md, Phase 7): one function for every CSI source.

The ML pipeline's input stage. The same `preprocess` prepares the SIM
dataset (spec/dataset.md), Room Sim's `.sim.pcap` exports, the public ESP32
datasets (Phase 9) and the owner's recordings (Phase 11). It is ML tooling,
never a runtime path (ADR-001, ADR-007).

**Labels.** Every `CsiWindow` is `"SIM"` or `"REAL"`, what its data is, and
every output copies the window's label, source and window name unchanged;
nothing here can relabel data. A shard is SIM. A pcap is SIM when
`wifisense stats`' own detector finds a Room Sim export (a caller who says
REAL is then refused), otherwise it is whatever the caller says it is,
which must be said. `SIM->REAL` labels results, never windows.

Each source has an adapter that maps its layout (§8) onto one canonical
record, `CsiWindow` (§2): complex CSI in `(re, im)` order on the 52 LLTF
bins `k = -26..-1, 1..26` with a `carried` mask, each link's received
packets only, right-aligned, with integer-nanosecond times and RSSI. Lost
packets occupy no slot and nothing is interpolated.

    from ml import preprocess as pp
    win = pp.from_roomsim_shard(DIR, scene_index, k, 2.0)               # SIM shard window ending at tick k
    wins = pp.from_pcap("x.sim.pcap", 2.0, [pp.sim_end_ns(k) for k in ticks],
                        link_map=pp.truth_link_map("x.truth.jsonl"))     # the same SIM run, as datagrams
    out = pp.preprocess(win, pp.PreprocessConfig(phase=True, bin_hz=None))

`preprocess` runs §1's steps in order: subcarrier mask (§3), per-packet L1
gain normalisation over the reference bins (§4), phase sanitisation (§5,
when `phase`), binning (§7, when `bin_hz`), per-link z-score (§6). Masks,
never sentinels, say what is missing; no output contains NaN; every
division has its guard stated in a comment before it.

The arithmetic is §4's normative arithmetic, vectorised, operation for
operation; `ml/preprocess_ref.py` is the stdlib transcription it is tested
against (§10). Where numpy cannot repeat `math` exactly it stays inside
§10's tolerances: the packet gain, the phase fit's `S` and `C` and every
bin sum use an error-free (TwoSum) compensated sum, which equals `fsum`
except within about `n^2 eps^2 sum|x|` of a rounding tie (so `a'` and `h'`
are, in practice, the reference's bits); the z-score's sums of `a'` and of
squared deviations use numpy's pairwise sum (a few ulps from `fsum`); and
`arctan2` and `log10` are numpy's (within an ulp of `math`'s here).
"""

from __future__ import annotations

import functools
import hashlib
import json
import math
import numbers
import os
from dataclasses import dataclass
from fractions import Fraction

import numpy as np

from ml import preprocess_ref as ref_impl
from ml import roomsim_dataset as rd
from wifisense.cli import _sim_banner
from wifisense.sources.esp32_udp import parse_esp32_frame
from wifisense.sources.recorder import ReadStats, read_pcap

__all__ = [
    "LABELS", "GRID_K", "K_GRID", "SHARED44", "REF_BINS", "PAIRS", "LAYOUTS", "PHASE_MIN_AMP",
    "SIM_T0_NS", "SIM_TICK_NS", "Layout", "CsiWindow", "PreprocessConfig", "layout", "to_grid",
    "from_roomsim_shard", "from_pcap", "truth_link_map", "sim_end_ns", "preprocess", "reference_input",
]

#: What a window's data is (decision 6). `SIM->REAL` labels results, never windows.
LABELS = ("SIM", "REAL")
#: §3: the grid, the 52 LLTF bins of HT20 in ascending k.
GRID_K = ref_impl.GRID_K
K_GRID = np.array(GRID_K, dtype=np.int8)
_K_FLOAT = K_GRID.astype(np.float64)
_GRID_INDEX = {k: b for b, k in enumerate(GRID_K)}
#: §3 / decision 3: the default reference bins, carried by every source (k = -26..-9, 1..26).
SHARED44 = ref_impl.SHARED44
_SHARED44_MASK = np.isin(K_GRID, SHARED44)
REF_BINS = ref_impl.REF_BINS
#: §5: a bin whose normalised amplitude a' is below this has no phase.
PHASE_MIN_AMP = ref_impl.PHASE_MIN_AMP
#: §8: pair orders. ESP-IDF, the datagram and the shards store (imag, real).
PAIRS = ("imag-real", "real-imag")
#: spec/roomsim.md §13: a `.sim.pcap`'s simulated time 0, epoch ns; label ticks are 0.5 s apart.
SIM_T0_NS = 1_790_000_000 * 1_000_000_000
SIM_TICK_NS = 500_000_000
_SLOT_NS = 12_500_000          # spec/dataset.md §4.1: slot_s = 0.0125 s, 80 transmissions per second
_NS_PER_S = 1_000_000_000
_GUARD_ENTRIES = np.arange(27, 38)   # §8's layout check: buffer entries 27..37, k = 27..31, -32..-27


# --------------------------------------------------------------------------
# §8 Layouts
# --------------------------------------------------------------------------


def _fft_k(e: int) -> int:
    """The bin of entry `e` of a 64-entry LTF block in FFT order (k = 0..31, then -32..-1)."""
    return e if e < 32 else e - 64


_LAYOUT_K = {
    "esp32-fw56": tuple(_fft_k(e) for e in range(56)),
    "esp32-lltf64": tuple(_fft_k(e) for e in range(64)),
    "esp-radar-52-asc": GRID_K,
    "esp-radar-52-pos-first": tuple(range(1, 27)) + tuple(range(-26, 0)),
}
#: The layouts whose pair order is fixed: the ESP32 buffer is (imag, real).
_FIXED_PAIRS = {"esp32-fw56": "imag-real", "esp32-lltf64": "imag-real"}
LAYOUTS = tuple(_LAYOUT_K)


@dataclass(frozen=True)
class Layout:
    """A source's CSI layout (§8): for each entry of its CSI array, the
    subcarrier `k` (None for an entry with none) and the pair order,
    `"imag-real"` (ESP-IDF, the datagram, the SIM shards) or `"real-imag"`.
    Only entries whose `k` is on the 52-bin grid are kept, chosen by index,
    never by value (hardware nulls need not be zero)."""

    name: str
    k: tuple
    pairs: str

    @property
    def entries(self) -> int:
        return len(self.k)

    @functools.cached_property
    def grid(self) -> tuple:
        """Per entry, its grid bin (`GRID_K.index(k)`), or None when `k` is off the grid (DC, guard)."""
        return tuple(None if k is None else _GRID_INDEX.get(k) for k in self.k)

    @property
    def carried(self) -> np.ndarray:
        """[52] bool: the grid bins this layout places a value at."""
        out = np.zeros(52, bool)
        out[[b for b in self.grid if b is not None]] = True
        return out


def layout(name: str, pairs: str | None = None) -> Layout:
    """The §8 layout `name`. The ESP32 layouts are always `"imag-real"`
    (passing another order is refused); the esp-radar layouts' order is the
    dataset's and must be given (Phase 9 checks each dataset)."""
    if isinstance(name, Layout):
        return name
    if name not in _LAYOUT_K:
        raise ValueError(f"layout {name!r} is not one of {LAYOUTS} (§8)")
    fixed = _FIXED_PAIRS.get(name)
    if fixed is not None:
        if pairs not in (None, fixed):
            raise ValueError(f"layout {name}: the ESP32 buffer is {fixed}, not {pairs!r} (§8)")
        pairs = fixed
    elif pairs not in PAIRS:
        raise ValueError(f"layout {name}: give its dataset's pair order, one of {PAIRS} (§8), not {pairs!r}")
    return Layout(name, _LAYOUT_K[name], pairs)


_layout = layout   # for functions whose `layout` argument shadows it


def to_grid(entries, lay: Layout | str) -> np.ndarray:
    """A source's CSI entries `[..., E, 2]` (its pair order, any real
    dtype) on the grid: `[..., 52]` complex128 in `(re, im)` order, 0 at
    bins the layout does not carry (`lay.carried`). Values are converted
    exactly (int8 and int16 are exact in float64); entries off the grid are
    dropped by index."""
    lay = layout(lay)
    a = np.asarray(entries)
    if a.ndim < 2 or a.shape[-2:] != (lay.entries, 2):
        raise ValueError(f"layout {lay.name}: want [..., {lay.entries}, 2] entries, got {a.shape}")
    if not np.issubdtype(a.dtype, np.integer) and not np.issubdtype(a.dtype, np.floating):
        raise ValueError(f"layout {lay.name}: entries must be real numbers, got {a.dtype}")
    first, second = a[..., 0], a[..., 1]
    re, im = (second, first) if lay.pairs == "imag-real" else (first, second)
    entry = [e for e, b in enumerate(lay.grid) if b is not None]
    bins = [lay.grid[e] for e in entry]
    out = np.zeros(a.shape[:-2] + (52,), np.complex128)
    out.real[..., bins] = re[..., entry]
    out.imag[..., bins] = im[..., entry]
    return out


# --------------------------------------------------------------------------
# §2 The canonical record
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class CsiWindow:
    """One window of CSI from any source, on the canonical grid (§2).

    `label` is what the data is, "SIM" or "REAL"; `source` its provenance
    (`"<dataset>[impairments=<profile>]:<scene id>"` for a SIM shard,
    `"pcap:<12 hex of the file's sha256>"` for a pcap); `window_end` names
    the window (`"k=<tick>"`, `"t_end_ns=<n>"`). Each link `(tx, rx)`
    (ascending; `tx = -1` when unknown) holds its **received packets only**,
    in time order, **right-aligned**: the newest at `P - 1`, padding
    (`received` false) first, so a slot index is not a time (`t_rel_ns`
    is). The constructor checks every dtype, shape and invariant and
    refuses a non-finite CSI entry. Treat the arrays as read-only.
    """

    label: str
    source: str
    window_end: str
    links: np.ndarray            #: [L, 2] int16 (tx, rx); tx = -1 when unknown
    tx_known: np.ndarray         #: [L] bool
    link_pos: np.ndarray         #: [L, 2, 3] float64: tx and rx positions, m; 0 where unknown
    link_pos_known: np.ndarray   #: [L, 2] bool
    csi: np.ndarray              #: [L, P, 52] complex128 (re, im); 0 where not received or not carried
    received: np.ndarray         #: [L, P] bool
    carried: np.ndarray          #: [52] bool: grid bins this source carries
    t_rel_ns: np.ndarray         #: [L, P] int64: packet time minus the window's end (< 0); 0 where not received
    rssi: np.ndarray             #: [L, P] float64 dBm; 0 where absent
    rssi_mask: np.ndarray        #: [L, P] bool
    length_ns: int               #: the window's length

    def __post_init__(self) -> None:
        _check_window(self)


def _want(name: str, arr, dtype, shape: tuple) -> None:
    """Refuse a record field that is not an ndarray of `dtype` and `shape` (None: any)."""
    if not isinstance(arr, np.ndarray) or arr.dtype != np.dtype(dtype) or arr.ndim != len(shape) or any(
            w is not None and w != g for w, g in zip(shape, arr.shape)):
        got = f"{arr.dtype} {arr.shape}" if isinstance(arr, np.ndarray) else type(arr).__name__
        raise ValueError(f"CsiWindow.{name}: want {np.dtype(dtype)} {shape}, got {got}")


def _check_window(w: CsiWindow) -> None:
    """§2's record, checked: dtypes and shapes, labels, link order, and the
    masks' invariants (0 where not received or not carried, right-aligned
    packets in time order inside the window)."""
    if w.label not in LABELS:
        raise ValueError(f"CsiWindow.label {w.label!r} is not one of {LABELS} (SIM->REAL labels results, "
                         f"never windows)")
    for name in ("source", "window_end"):
        if not isinstance(getattr(w, name), str) or not getattr(w, name):
            raise ValueError(f"CsiWindow.{name} must be a non-empty string")
    if isinstance(w.length_ns, bool) or not isinstance(w.length_ns, numbers.Integral) or w.length_ns <= 0:
        raise ValueError(f"CsiWindow.length_ns {w.length_ns!r} is not a positive integer")
    _want("links", w.links, np.int16, (None, 2))
    n_links = w.links.shape[0]
    _want("csi", w.csi, np.complex128, (n_links, None, 52))
    n_pk = w.csi.shape[1]
    _want("tx_known", w.tx_known, np.bool_, (n_links,))
    _want("link_pos", w.link_pos, np.float64, (n_links, 2, 3))
    _want("link_pos_known", w.link_pos_known, np.bool_, (n_links, 2))
    _want("received", w.received, np.bool_, (n_links, n_pk))
    _want("carried", w.carried, np.bool_, (52,))
    _want("t_rel_ns", w.t_rel_ns, np.int64, (n_links, n_pk))
    _want("rssi", w.rssi, np.float64, (n_links, n_pk))
    _want("rssi_mask", w.rssi_mask, np.bool_, (n_links, n_pk))
    tx, rx = w.links[:, 0].astype(np.int64), w.links[:, 1].astype(np.int64)
    if (tx < -1).any() or (rx < 0).any() or (tx == rx).any():
        raise ValueError("CsiWindow.links: tx must be -1 (unknown) or a node id, rx a node id, tx != rx")
    key = tx * 65536 + rx
    if (np.diff(key) <= 0).any():
        raise ValueError("CsiWindow.links must be unique and in ascending (tx, rx) order (§2)")
    if not np.array_equal(w.tx_known, tx != -1):
        raise ValueError("CsiWindow.tx_known must be true exactly where tx != -1")
    if not np.isfinite(w.link_pos).all() or (w.link_pos[~w.link_pos_known] != 0).any():
        raise ValueError("CsiWindow.link_pos must be finite, and 0 where not known")
    if not np.isfinite(w.csi.real).all() or not np.isfinite(w.csi.imag).all():
        raise ValueError(f"{w.source} {w.window_end}: a non-finite CSI entry; refused (§2)")
    if (w.csi[~w.received] != 0).any() or (w.csi[:, :, ~w.carried] != 0).any():
        raise ValueError("CsiWindow.csi must be 0 where not received and at bins not carried (§2)")
    if (w.received[:, 1:] < w.received[:, :-1]).any():
        raise ValueError("CsiWindow.received must be right-aligned: padding first, the newest packet at P-1 (§2)")
    t = w.t_rel_ns
    if (t[~w.received] != 0).any():
        raise ValueError("CsiWindow.t_rel_ns must be 0 where not received")
    if ((t[w.received] < -w.length_ns) | (t[w.received] >= 0)).any():
        raise ValueError(f"CsiWindow.t_rel_ns must lie in [-length_ns, 0) = [{-w.length_ns}, 0) where received")
    both = w.received[:, 1:] & w.received[:, :-1]
    if (np.diff(t, axis=1)[both] < 0).any():
        raise ValueError("CsiWindow: each link's packets must be in time order (§2)")
    if not np.isfinite(w.rssi).all() or (w.rssi[~w.rssi_mask] != 0).any() or (w.rssi_mask & ~w.received).any():
        raise ValueError("CsiWindow.rssi must be finite, 0 where absent, and present only on received packets")


# --------------------------------------------------------------------------
# §1 Configuration
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class PreprocessConfig:
    """§1: `phase` (§5, off by default), `bin_hz` (§7: None keeps every
    packet), `ref_bins` ("shared44", the default, or "carried" for
    single-source work, §3), `min_packets` (§6, at least 2)."""

    phase: bool = False
    bin_hz: float | None = None
    ref_bins: str = "shared44"
    min_packets: int = 3

    def __post_init__(self) -> None:
        if not isinstance(self.phase, bool):
            raise ValueError(f"phase {self.phase!r} is not true or false")
        if self.bin_hz is not None and (isinstance(self.bin_hz, bool) or not isinstance(self.bin_hz, (int, float))
                                        or not math.isfinite(self.bin_hz) or self.bin_hz <= 0):
            raise ValueError(f"bin_hz {self.bin_hz!r} is not None or a positive finite number (§7)")
        if self.ref_bins not in REF_BINS:
            raise ValueError(f"ref_bins {self.ref_bins!r} is not one of {REF_BINS} (§3)")
        if isinstance(self.min_packets, bool) or not isinstance(self.min_packets, int) or self.min_packets < 2:
            raise ValueError(f"min_packets {self.min_packets!r} is not an integer >= 2 (§6)")


# --------------------------------------------------------------------------
# §2 Adapters
# --------------------------------------------------------------------------


def _scalar_or_list(values, name: str):
    """`([int, ...], True)` for one integer, `(list, False)` for a sequence of them."""
    if isinstance(values, (numbers.Integral, np.integer)) and not isinstance(values, (bool, np.bool_)):
        return [int(values)], True
    if isinstance(values, (str, bytes)) or not hasattr(values, "__iter__"):
        raise ValueError(f"{name}: {values!r} is not an integer or a sequence of integers")
    out = []
    for v in values:
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, (numbers.Integral, np.integer)):
            raise ValueError(f"{name}: {v!r} is not an integer")
        out.append(int(v))
    return out, False


def _pack(mask: np.ndarray):
    """Right-align each row's true slots (§2): `(P, l, j, d)`, slot `(l, j)`
    going to packed slot `(l, d)`; `P` is the most true slots in a row."""
    after = np.cumsum(mask[:, ::-1], axis=1)[:, ::-1]      # true slots at or after j
    p = int(after[:, 0].max(initial=0)) if mask.shape[1] else 0
    l, j = np.nonzero(mask)
    return p, l, j, p - after[l, j]


def _shared(*arrays):
    """Mark arrays that every window of one adapter call shares read-only."""
    for a in arrays:
        a.setflags(write=False)
    return arrays


def _link_positions(links: np.ndarray, pos: dict | None) -> tuple:
    """`(link_pos [L, 2, 3], link_pos_known [L, 2])` from node id -> (x, y, z)."""
    n = links.shape[0]
    link_pos = np.zeros((n, 2, 3))
    known = np.zeros((n, 2), bool)
    if pos is None:
        return link_pos, known
    for i, (tx, rx) in enumerate(links.tolist()):
        for end, node in ((0, tx), (1, rx)):
            if node == -1:
                continue
            if node not in pos:
                raise ValueError(f"node {node} is not in the declared scene; positions are never guessed (ADR-008)")
            link_pos[i, end] = pos[node]
            known[i, end] = True
    return link_pos, known


def from_roomsim_shard(dataset_dir: str, scene_index: int, k, length_s: float, *, manifest: dict | None = None,
                       scene: dict | None = None):
    """SIM dataset v1 windows (spec/dataset.md §4.3) as `CsiWindow`s (§2).

    Per link, the received columns `m` whose transmission `n = m*N +
    tx_slot[l]` lies in `[80 (t_k - length_s), 80 t_k)`, `t_k = 0.5 (k +
    1)`, chosen in integers (`ml.roomsim_dataset.window_iq`, the loader's
    own slots); lost packets are dropped, the rest right-aligned;
    `t_rel_ns = (n - 40 (k + 1)) * 12_500_000`. Layout `esp32-fw56` (44
    bins carried); `tx` known; positions from the shard's `node_pos` (its
    float32 values); label SIM; `source` names the dataset directory, the
    manifest's impairment profile and the scene.

    `k` is one label tick (returns one window) or a sequence of them
    (returns a list; the shard is loaded once). Pass `scene`
    (`rd.load_scene`'s dict) to skip loading, `manifest` to skip re-reading
    it. Everything returned is SIM.
    """
    if scene is None:
        scene = rd.load_scene(dataset_dir, scene_index, manifest=manifest)
    meta = scene["meta"]
    if meta.get("label") != rd.LABEL or meta.get("scene_index") != scene_index:
        raise ValueError(f"scene dict is {meta.get('label')!r} scene {meta.get('scene_index')}, "
                         f"not SIM scene {scene_index}")
    lay = layout("esp32-fw56")
    # §8: the shard must say it holds this layout (by index: k and the active flags).
    if (scene["subcarrier_k"].tolist() != list(lay.k)
            or scene["subcarrier_active"].astype(bool).tolist() != [b is not None for b in lay.grid]):
        raise ValueError(f"scene {scene_index}: subcarrier_k / subcarrier_active are not the esp32-fw56 layout (§8)")
    halves = rd._halves(length_s)
    length_ns = halves * 500_000_000
    ticks, one = _scalar_or_list(k, "k")
    name = os.path.basename(os.path.normpath(os.path.abspath(os.fspath(dataset_dir))))
    source = f"{name}[impairments={scene['dataset']['impairments']}]:{meta['scene_id']}"
    links = scene["links"].astype(np.int16)
    if links.shape[0] and (np.diff(links[:, 0].astype(np.int64) * 65536 + links[:, 1]) <= 0).any():
        raise ValueError(f"scene {scene_index}: links are not in ascending (tx, rx) order (§2)")
    pos = {int(i): tuple(float(v) for v in p) for i, p in zip(scene["node_ids"], scene["node_pos"])}
    link_pos, pos_known = _link_positions(links, pos)
    tx_known = np.ones(links.shape[0], bool)
    carried = lay.carried
    _shared(links, link_pos, pos_known, tx_known, carried)
    out = []
    for tick in ticks:
        iw = rd.window_iq(scene, tick, length_s)
        p, l, j, d = _pack(iw.mask)
        n_links = links.shape[0]
        csi = np.zeros((n_links, p, 52), np.complex128)
        received = np.zeros((n_links, p), bool)
        t_rel_ns = np.zeros((n_links, p), np.int64)
        rssi = np.zeros((n_links, p))
        csi[l, d] = to_grid(iw.iq[l, :, j], lay)                 # [n, 56, 2] (imag, real) -> [n, 52]
        received[l, d] = True
        t_rel_ns[l, d] = (iw.transmission[l, j] - 40 * (tick + 1)) * _SLOT_NS
        rssi[l, d] = iw.rssi[l, j]
        out.append(CsiWindow(
            label=rd.LABEL, source=source, window_end=f"k={tick}", links=links, tx_known=tx_known,
            link_pos=link_pos, link_pos_known=pos_known, csi=csi, received=received, carried=carried,
            t_rel_ns=t_rel_ns, rssi=rssi, rssi_mask=received.copy(), length_ns=length_ns))
    return out[0] if one else out


def sim_end_ns(k: int) -> int:
    """The end of SIM label tick `k`'s window in a `.sim.pcap`, epoch ns:
    `1_790_000_000 s + 0.5 (k + 1) s` (spec/roomsim.md §13)."""
    return SIM_T0_NS + SIM_TICK_NS * (int(k) + 1)


def truth_link_map(truth_path: str) -> dict:
    """`(rx, seq) -> (tx, rx)` for every received packet of a Room Sim
    export's `PREFIX.truth.jsonl` (spec/roomsim.md §13): the only way to
    recover a datagram's transmitter. Checks every line is labelled SIM and
    that each `link_index` names one `(tx, rx)`. SIM."""
    out: dict = {}
    by_index: dict = {}
    with open(truth_path, encoding="utf-8") as fh:
        for line_no, line in enumerate(fh, 1):
            rec = json.loads(line)
            if rec.get("label") != "SIM":
                raise ValueError(f"{truth_path}:{line_no}: label {rec.get('label')!r}, not 'SIM'")
            tx = int(rec["tx"])
            for pk in rec["packets"]:
                rx = int(pk["rx"])
                if by_index.setdefault(pk["link_index"], (tx, rx)) != (tx, rx):
                    raise ValueError(f"{truth_path}:{line_no}: link_index {pk['link_index']} is "
                                     f"{by_index[pk['link_index']]} and {(tx, rx)}")
                if pk["lost"] or pk["seq"] is None:
                    continue                              # a lost packet has no datagram and no seq
                key = (rx, int(pk["seq"]))
                if key in out:
                    raise ValueError(f"{truth_path}:{line_no}: (rx, seq) = {key} appears twice")
                out[key] = (tx, rx)
    return out


def _length_ns(length_s) -> int:
    """A window length in integer nanoseconds, its decimal value exactly
    (2.0 -> 2_000_000_000); refused unless a positive whole number of ns."""
    if isinstance(length_s, bool) or not isinstance(length_s, (int, float)) or not math.isfinite(length_s):
        raise ValueError(f"window length {length_s!r} s is not a finite number")
    ns = Fraction(repr(float(length_s))) * _NS_PER_S
    if ns.denominator != 1 or ns <= 0:
        raise ValueError(f"window length {length_s!r} s is not a positive whole number of nanoseconds")
    return int(ns)


def _scene_positions(scene) -> dict:
    """node id -> (x, y, z) from a declared scene (a scene file's path or its
    parsed JSON, spec/scene.schema.json `nodes`)."""
    if isinstance(scene, (str, os.PathLike)):
        with open(scene, encoding="utf-8") as fh:
            scene = json.load(fh)
    nodes = scene.get("nodes") if isinstance(scene, dict) else None
    if not isinstance(nodes, list):
        raise ValueError("scene: no 'nodes' list (spec/scene.schema.json)")
    out = {}
    for node in nodes:
        nid, p = node.get("id"), node.get("pos")
        if isinstance(nid, bool) or not isinstance(nid, int) or not isinstance(p, (list, tuple)) or len(p) != 3:
            raise ValueError(f"scene: node {node!r} has no integer id and 3-vector pos")
        if nid in out:
            raise ValueError(f"scene: node id {nid} appears twice")
        xyz = tuple(float(v) for v in p)
        if not all(math.isfinite(v) for v in xyz):
            raise ValueError(f"scene: node {nid} has a non-finite position")
        out[nid] = xyz
    return out


def _export_manifest(path: str) -> dict | None:
    """A Room Sim export's `PREFIX.manifest.json` beside `PREFIX[.sim].pcap`, or None."""
    for suffix in (".sim.pcap", ".pcap"):
        if path.endswith(suffix):
            prefix = path[: -len(suffix)]
            break
    else:
        prefix = path
    try:
        with open(prefix + ".manifest.json", encoding="utf-8") as fh:
            doc = json.load(fh)
    except (OSError, ValueError):
        return None
    return doc if isinstance(doc, dict) else None


def _layout_check(path: str, lay: Layout, rx: np.ndarray, iq: np.ndarray) -> None:
    """§8's check on REAL data, per receiver over the recording: the mean
    power of buffer entries 27..37 (guard bins under an ESP32 layout) must
    be at least 20 dB below the active entries', else the recording is
    refused naming the receiver. Integer powers, compared exactly."""
    active = np.array([e for e, b in enumerate(lay.grid) if b is not None])
    v = iq.astype(np.int64)
    power = v[..., 0] * v[..., 0] + v[..., 1] * v[..., 1]          # [n, E]
    for node in np.unique(rx).tolist():
        mine = power[rx == node]
        guard = int(mine[:, _GUARD_ENTRIES].sum())
        act = int(mine[:, active].sum())
        # 10 log10((guard / 11) / (act / n_active)) <= -20  <=>  100 * guard * n_active <= act * 11
        if act == 0 or 100 * guard * len(active) > act * len(_GUARD_ENTRIES):
            if act == 0:
                why = "no energy on the active entries"
            elif guard == 0:
                why = "guard entries silent"
            else:
                db = 10.0 * math.log10((guard / len(_GUARD_ENTRIES)) / (act / len(active)))
                why = f"guard entries 27..37 only {-db:.1f} dB below the active entries"
            raise ValueError(f"{path}: receiver {node}: {why} (need 20 dB): its datagrams do not look like the "
                             f"{lay.name} layout (§8); recording refused")


def from_pcap(path, length_s: float, end_ns, layout: str = "esp32-fw56", port: int | None = 5566,
              link_map: dict | None = None, single_tx: bool = False, label: str | None = None, scene=None, *,
              stats: ReadStats | None = None):
    """Raw-CSI datagrams of a pcap as `CsiWindow`s (§2): a recording
    (`python3 -m wifisense record`, REAL) or a Room Sim export (`saans
    roomsim-export`, SIM).

    Read with `wifisense.sources.recorder.read_pcap` (integer `t_ns`, UDP
    destination `port`; None keeps every port) and parsed with
    `parse_esp32_frame`. A datagram is skipped and counted in
    `stats.malformed` when parsing raises (FeatureFrames included), when
    `n_antennas` or `n_subcarriers` is 0, or when `n_subcarriers` is not
    the layout's entry count (`esp32-fw56`: 56; `esp32-lltf64`: 64).
    `end_ns` is one integer epoch-ns window end (one window returned) or a
    sequence of them (a list); a window is `[end - length, end)` compared in
    integers (`sim_end_ns(k)` gives a `.sim.pcap`'s tick ends).

    Links: with `link_map` (`(rx, seq) -> (tx, rx)`, e.g.
    `truth_link_map(PREFIX.truth.jsonl)`) every datagram goes to its link
    and every link of the map is listed, packets or not; a datagram missing
    from the map is refused. Without it a receiver's stream is one link
    `(-1, rx)`, which is honest only when one transmitter is heard: a Room
    Sim export whose manifest says `gateway_faithful` false is refused, and
    a recording (or an export without its manifest) needs `single_tx=True`
    from the caller.

    Label: SIM when `wifisense.cli._sim_banner` finds a Room Sim export
    (`label="REAL"` is then refused); otherwise `label` ("REAL" or "SIM")
    is required. REAL data must pass §8's layout check. Positions come only
    from a declared `scene` (a scene file's path or parsed JSON), never
    guessed. `source` is `"pcap:"` and 12 hex digits of the file's sha256.
    """
    lay = _layout(layout)
    if lay.name not in _FIXED_PAIRS:
        raise ValueError(f"from_pcap reads ESP32 datagrams: layout {lay.name!r} is not one of {tuple(_FIXED_PAIRS)}")
    path = os.fspath(path)
    length_ns = _length_ns(length_s)
    ends, one = _scalar_or_list(end_ns, "end_ns")
    banner = _sim_banner(path)
    if banner is not None:
        if label not in (None, "SIM"):
            raise ValueError(f"{path} is a Room Sim export, so its label is SIM; label {label!r} refused (§2)")
        label = "SIM"
    elif label not in LABELS:
        raise ValueError(f"{path} is not a Room Sim export: pass label='REAL' for a recording (or 'SIM'), "
                         f"not {label!r} (§2)")
    if link_map is not None and single_tx:
        raise ValueError("single_tx=True is for a stream without a link map; with one, every packet has its link")
    if link_map is None:
        manifest = _export_manifest(path) if banner is not None else None
        if manifest is not None and manifest.get("gateway_faithful") is False:
            raise ValueError(f"{path}: a Room Sim export that is not gateway-faithful (several transmitters "
                             f"interleave in each receiver's stream, spec/roomsim.md §12.4); pass "
                             f"link_map=truth_link_map(<prefix>.truth.jsonl) (§2)")
        faithful = manifest is not None and manifest.get("gateway_faithful") is True
        if not faithful and not single_tx:
            raise ValueError(f"{path}: without a link map a receiver's stream is one link only when one transmitter "
                             f"is heard; pass single_tx=True if that is so, or a link_map (§2)")
    with open(path, "rb") as fh:
        digest = hashlib.sha256(fh.read()).hexdigest()
    st = stats if stats is not None else ReadStats()
    t_ns, rx, seq, rssi, iq = [], [], [], [], []
    for pkt in read_pcap(path, port, st):
        try:
            frame = parse_esp32_frame(pkt.payload)
        except ValueError:
            st.malformed += 1
            continue
        # The parser zero-fills a payload short of n_antennas*n_subcarriers pairs only when either
        # is 0, and a count other than the layout's is another layout: skipped and counted.
        if frame.n_antennas == 0 or frame.n_subcarriers == 0 or frame.n_subcarriers != lay.entries:
            st.malformed += 1
            continue
        t_ns.append(pkt.t_ns)
        rx.append(frame.node_id)
        seq.append(frame.sequence)
        rssi.append(frame.rssi_dbm)
        iq.append(frame.iq)
    t_arr = np.array(t_ns, np.int64)
    rx_arr = np.array(rx, np.int64)
    iq_arr = np.array(iq, np.int8).reshape(len(iq), lay.entries, 2)
    rssi_arr = np.array(rssi, np.float64)
    if label == "REAL":
        _layout_check(path, lay, rx_arr, iq_arr)
    if link_map is not None:
        all_links = sorted({(int(v[0]), int(v[1])) for v in link_map.values()})
        index = {pair: i for i, pair in enumerate(all_links)}
        li = []
        for r, s in zip(rx, seq):
            pair = link_map.get((r, s))
            if pair is None:
                raise ValueError(f"{path}: datagram rx {r} seq {s} is not in the link map")
            if int(pair[1]) != r:
                raise ValueError(f"{path}: the link map sends rx {r}'s seq {s} to link {tuple(pair)}")
            li.append(index[(int(pair[0]), int(pair[1]))])
    else:
        all_links = [(-1, r) for r in sorted(set(rx))]
        index = {pair[1]: i for i, pair in enumerate(all_links)}
        li = [index[r] for r in rx]
    links = np.array(all_links, np.int16).reshape(len(all_links), 2)
    pos = _scene_positions(scene) if scene is not None else None
    link_pos, pos_known = _link_positions(links, pos)
    tx_known = links[:, 0] != -1
    li_arr = np.array(li, np.int64)
    order = np.lexsort((t_arr, li_arr))          # by link, then time; ties keep file order (stable)
    t_sorted, li_sorted = t_arr[order], li_arr[order]
    iq_sorted, rssi_sorted = iq_arr[order], rssi_arr[order]
    n_links = len(all_links)
    bounds = np.searchsorted(li_sorted, np.arange(n_links + 1))
    carried = lay.carried
    _shared(links, link_pos, pos_known, tx_known, carried)
    out = []
    for end in ends:
        lo = end - length_ns
        first = np.empty(n_links, np.int64)
        last = np.empty(n_links, np.int64)
        for i in range(n_links):
            seg = t_sorted[bounds[i]:bounds[i + 1]]
            first[i] = bounds[i] + np.searchsorted(seg, lo, "left")       # t >= end - length
            last[i] = bounds[i] + np.searchsorted(seg, end, "left")       # t < end
        count = last - first
        p = int(count.max(initial=0))
        csi = np.zeros((n_links, p, 52), np.complex128)
        received = np.zeros((n_links, p), bool)
        t_rel_ns = np.zeros((n_links, p), np.int64)
        rssi_w = np.zeros((n_links, p))
        for i in range(n_links):
            c = int(count[i])
            if c == 0:
                continue
            src = slice(int(first[i]), int(last[i]))
            csi[i, p - c:] = to_grid(iq_sorted[src], lay)
            received[i, p - c:] = True
            t_rel_ns[i, p - c:] = t_sorted[src] - end
            rssi_w[i, p - c:] = rssi_sorted[src]
        out.append(CsiWindow(
            label=label, source=f"pcap:{digest[:12]}", window_end=f"t_end_ns={end}", links=links,
            tx_known=tx_known, link_pos=link_pos, link_pos_known=pos_known, csi=csi, received=received,
            carried=carried, t_rel_ns=t_rel_ns, rssi=rssi_w, rssi_mask=received.copy(), length_ns=length_ns))
    return out[0] if one else out


# --------------------------------------------------------------------------
# §3–§7 preprocess
# --------------------------------------------------------------------------


def _csum(x: np.ndarray) -> np.ndarray:
    """Sum over the last axis with error-free transformations: pairwise
    TwoSum, the exact rounding errors summed separately and added once.
    Equal to `math.fsum` (correctly rounded) except within about `n^2 eps^2
    sum |x|` of a rounding tie. Used for the packet gain (so `a'` and `h'`
    carry the reference's bits), the sums that cancel (§5's `S` and `C`)
    and every bin sum (§7)."""
    if x.shape[-1] == 0:
        return np.zeros(x.shape[:-1])
    err = np.zeros(x.shape[:-1])
    while x.shape[-1] > 1:
        if x.shape[-1] % 2:
            x = np.concatenate((x, np.zeros(x.shape[:-1] + (1,))), axis=-1)
        a, b = x[..., 0::2], x[..., 1::2]
        s = a + b
        bv = s - a
        err += ((a - (s - bv)) + (b - bv)).sum(axis=-1)
        x = s
    return x[..., 0] + err


def _angle(im: np.ndarray, re: np.ndarray) -> np.ndarray:
    """§4's angle: `arctan2(im, re)` with a result of -π replaced by π."""
    a = np.arctan2(im, re)
    return np.where(a == -np.pi, np.pi, a)


def _ref_mask(ref_bins: str, carried: np.ndarray) -> np.ndarray:
    """§3: the reference bins R as a [52] mask (refusing a source that lacks shared44's bins)."""
    mask = np.zeros(52, bool)
    mask[ref_impl.reference_bins(ref_bins, carried.tolist())] = True
    return mask


def _sanitise(hre, him, a1, valid, carried, ref):
    """§5 per packet, vectorised: returns `(r_re, r_im, mask)`, the residual
    `r_b = h'_b exp(-j (a k_b + c))` [L, P, 52] and the per-packet phase
    mask (no link mask yet). `a`, `c` are never returned."""
    pairs = ref_impl.consecutive_pairs(np.flatnonzero(ref).tolist())
    lo = np.array([b0 for b0, _ in pairs], np.int64)
    hi = lo + 1
    x0r, x0i, x1r, x1i = hre[..., lo], him[..., lo], hre[..., hi], him[..., hi]
    s = _csum(np.stack((x1r * x0r + x1i * x0i, x1i * x0r - x1r * x0i)))       # S = sum h'_{k+1} conj(h'_k)
    has = valid & ((s[0] != 0.0) | (s[1] != 0.0))                             # guard: S == 0 -> no phase
    a = np.where(has, _angle(s[1], s[0]), 0.0)
    th = a[..., None] * _K_FLOAT[ref]
    cs, sn = np.cos(th), np.sin(th)
    rr, ri = hre[..., ref], him[..., ref]
    c = _csum(np.stack((rr * cs + ri * sn, ri * cs - rr * sn)))               # C = sum h'_b exp(-j a k_b)
    has &= (c[0] != 0.0) | (c[1] != 0.0)                                      # guard: C == 0 -> no phase
    off = np.where(has, _angle(c[1], c[0]), 0.0)
    th = a[..., None] * _K_FLOAT + off[..., None]
    cs, sn = np.cos(th), np.sin(th)
    r_re = hre * cs + him * sn
    r_im = him * cs - hre * sn
    mask = has[..., None] & carried & ((hre != 0.0) | (him != 0.0)) & (a1 >= PHASE_MIN_AMP)
    return r_re, r_im, mask


def _unit(re: np.ndarray, im: np.ndarray, mask: np.ndarray):
    """`(re, im) / |(re, im)|` where `mask` and the magnitude is not 0 (the
    guard), else 0; returns the components and the narrowed mask."""
    m = np.sqrt(re * re + im * im)
    mask = mask & (m != 0.0)
    m = np.where(mask, m, 1.0)
    return np.where(mask, re / m, 0.0), np.where(mask, im / m, 0.0), mask


def _mean(values: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """`ref_impl.mean` over the last axis of `values` where `filled`: the
    common value when every filled value is equal (exactly), else their
    compensated sum over their count; 0 where nothing is filled."""
    count = filled.sum(axis=-1)
    big = np.where(filled, values, -np.inf).max(axis=-1, initial=-np.inf)
    small = np.where(filled, values, np.inf).min(axis=-1, initial=np.inf)
    total = _csum(np.where(filled, values, 0.0))
    mean = total / np.where(count > 0, count, 1)                            # guard: divide only filled bins
    return np.where(count > 0, np.where(big == small, big, mean), 0.0)


def _binned(w: CsiWindow, nb: int, w_ns: int, valid, a1, gdb, rssi, rssi_mask, resid):
    """§7: per link and integer-ns bin, the mean of its packets' `a'`,
    `gain_db` and RSSI (those with one), the count, and with phase the sum
    of the residuals of its packets with a phase mask. Packets of one link
    are in time order, so a bin's packets are consecutive; they are stacked
    on a last `depth` axis to be summed."""
    n_links = valid.shape[0]
    bidx = (w.t_rel_ns + w.length_ns) // w_ns                                # in integers: [0, nb)
    l, p = np.nonzero(valid)
    b = bidx[l, p]
    key = l * nb + b                                                         # non-decreasing
    count = np.bincount(key, minlength=n_links * nb).reshape(n_links, nb)
    if key.size:
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        sizes = np.diff(np.r_[starts, key.size])
        rank = np.arange(key.size) - np.repeat(starts, sizes)
        depth = int(sizes.max())
    else:
        rank, depth = key, 0
    filled = np.zeros((n_links, nb, depth), bool)
    filled[l, b, rank] = True
    stack = np.zeros((n_links, nb, 52, depth))
    stack[l, b, :, rank] = a1[l, p]
    amp = _mean(stack, filled[:, :, None, :])                                # mean amplitude, never |complex mean|
    g = np.zeros((n_links, nb, depth))
    g[l, b, rank] = gdb[l, p]
    gain = _mean(g, filled)
    rv = np.zeros((n_links, nb, depth))
    rm = np.zeros((n_links, nb, depth), bool)
    rv[l, b, rank] = rssi[l, p]
    rm[l, b, rank] = rssi_mask[l, p]
    out = {"x": amp, "present": count > 0, "gain": gain, "rssi": _mean(rv, rm), "rssi_mask": rm.any(axis=-1),
           "count": count}
    if resid is not None:
        r_re, r_im, pmask = resid
        rs = np.zeros((2, n_links, nb, 52, depth))
        rs[0, l, b, :, rank] = np.where(pmask, r_re, 0.0)[l, p]
        rs[1, l, b, :, rank] = np.where(pmask, r_im, 0.0)[l, p]
        total = _csum(rs)                                                    # [2, L, nb, 52]
        out["phase"] = (total[0], total[1], (total[0] != 0.0) | (total[1] != 0.0))
    return out


def _zscore(x: np.ndarray, use: np.ndarray, n: np.ndarray, link_mask: np.ndarray, carried: np.ndarray):
    """§6 on `x` [L, 52, T] over the samples `use` [L, T] of the used links:
    returns `(z, mu, sigma)`. A bin whose used values are all equal
    (compared exactly) gets that value, sigma 0 and z 0."""
    u = use[:, None, :] & carried[None, :, None]
    big = np.where(u, x, -np.inf).max(axis=2, initial=-np.inf)
    small = np.where(u, x, np.inf).min(axis=2, initial=np.inf)
    equal = big == small                                                     # every used value equal
    # Guard: a used link has n >= min_packets >= 2, so n and n - 1 are > 0; a dropped link divides by 1.
    nn = np.where(link_mask, n, 1).astype(np.float64)[:, None]
    dof = np.where(link_mask, n - 1, 1).astype(np.float64)[:, None]
    mu = np.where(u, x, 0.0).sum(axis=2) / nn
    dev = np.where(u, x - mu[..., None], 0.0)
    sigma = np.sqrt((dev * dev).sum(axis=2) / dof)
    mu = np.where(equal, big, mu)
    sigma = np.where(equal, 0.0, sigma)
    ok = sigma != 0.0                                                        # guard: z only where sigma > 0
    z = np.where(u & ok[..., None], (x - mu[..., None]) / np.where(ok, sigma, 1.0)[..., None], 0.0)
    live = link_mask[:, None] & carried[None, :]
    return z, np.where(live, mu, 0.0), np.where(live, sigma, 0.0)


def preprocess(window: CsiWindow, cfg: PreprocessConfig = PreprocessConfig(), *, dtype=np.float32) -> dict:
    """Preprocess one window (spec/preprocess.md §1–§9); the window's label
    (SIM or REAL), source and name are copied to the output unchanged.

    Steps, in order: §3 the reference bins R (`cfg.ref_bins`); §4 per
    received packet `g = sum(|h_b| over R) / |R|` (a packet with `g == 0` is
    dropped, `zero_gain_packets`), `a'_b = |h_b| / g` and `h'_b = h_b / g`
    on every carried bin; §5 with `cfg.phase` the residual phase as
    `(cos, sin)` (the slope and offset are never returned); §7 with
    `cfg.bin_hz` integer-ns bins (amplitude, gain and RSSI averaged per bin);
    §6 the per-link z-score of `a'` over the link's packets (or bins), a
    link with fewer than `cfg.min_packets` dropped (`links_dropped`).

    Returns §9's dict: `label`, `source`, `window_end`, `links`, `tx_known`,
    `link_pos`, `link_pos_known`, `k_grid`, `carried`, `ref`, `amp` [L, 52,
    T], `mu`, `sigma` [L, 52], `n_packets` [L], `sample_mask` [L, T]
    (received, `g != 0` and the link used), `link_mask` [L], `gain_db`,
    `rssi`, `rssi_mask`, `t_rel` [L, T] (s; bin centres with `bin_hz`), with
    `bin_hz` `bin_count` [L, T], with `phase` `phase` [L, 52, T, 2] and
    `phase_mask` [L, 52, T], and the counts. Every per-sample output is 0
    (False) where `sample_mask` is false (bin centres excepted); no output
    is NaN. Float outputs are float32 (§9); `dtype=np.float64` returns the
    values before that cast (§10 compares those with `ml/preprocess_ref.py`).
    `n_packets` is each link's count of usable packets (or non-empty bins),
    also for a dropped link.
    """
    if not isinstance(window, CsiWindow):
        raise TypeError(f"preprocess wants a CsiWindow, got {type(window).__name__}")
    if not isinstance(cfg, PreprocessConfig):
        raise TypeError(f"preprocess wants a PreprocessConfig, got {type(cfg).__name__}")
    dtype = np.dtype(dtype)
    if dtype not in (np.dtype(np.float32), np.dtype(np.float64)):
        raise ValueError(f"dtype {dtype} is not float32 or float64")
    w = window
    carried = w.carried
    # §3: the reference bins (refused when the source lacks them).
    ref = _ref_mask(cfg.ref_bins, carried)
    n_ref = float(ref.sum())
    if cfg.bin_hz is not None:
        nb, w_ns = ref_impl.bin_layout(cfg.bin_hz, w.length_ns)

    # §4: per-packet L1 gain over R, applied to every carried bin.
    re, im = w.csi.real, w.csi.imag
    mag = np.sqrt(re * re + im * im)                                         # |h| = sqrt(re*re + im*im), never hypot
    g = _csum(mag[..., ref]) / n_ref                                         # fsum over R (compensated)
    zero = w.received & (g == 0.0)                                           # guard: g == 0 -> dropped
    valid = w.received & ~zero
    gv = np.where(valid, g, 1.0)[..., None]                                 # divide only where g != 0
    keep = valid[..., None] & carried
    a1 = np.where(keep, mag / gv, 0.0)
    hre = np.where(keep, re / gv, 0.0)
    him = np.where(keep, im / gv, 0.0)
    gdb = np.where(valid, 20.0 * np.log10(gv[..., 0]), 0.0)

    # §5: phase sanitisation per packet.
    resid = _sanitise(hre, him, a1, valid, carried, ref) if cfg.phase else None

    # §7: binning, or the packets themselves.
    if cfg.bin_hz is None:
        x = a1.transpose(0, 2, 1)
        present = valid
        gain = gdb
        rssi_mask = w.rssi_mask & valid
        rssi = np.where(rssi_mask, w.rssi, 0.0)
        t_rel = w.t_rel_ns / 1e9
        if resid is not None:
            pc, ps, pmask = _unit(*resid)
    else:
        bins = _binned(w, nb, w_ns, valid, a1, gdb, w.rssi, w.rssi_mask & valid, resid)
        x = bins["x"].transpose(0, 2, 1)
        present = bins["present"]
        gain, rssi, rssi_mask = bins["gain"], bins["rssi"], bins["rssi_mask"]
        centre = (2 * (np.arange(nb, dtype=np.int64) * w_ns - w.length_ns) + w_ns) / 2e9
        t_rel = np.broadcast_to(centre, present.shape)
        if resid is not None:
            pc, ps, pmask = _unit(*bins["phase"])

    # §6: per-link z-score.
    n = present.sum(axis=1)
    link_mask = n >= cfg.min_packets                                          # guard: fewer -> link not used
    smask = present & link_mask[:, None]
    z, mu, sigma = _zscore(np.ascontiguousarray(x), smask, n, link_mask, carried)

    def per_sample(a, zero_value=0.0):
        return np.where(smask, a, zero_value)

    out = {
        "label": w.label, "source": w.source, "window_end": w.window_end,
        "links": w.links.copy(), "tx_known": w.tx_known.copy(), "link_pos": w.link_pos.astype(dtype),
        "link_pos_known": w.link_pos_known.copy(),
        "k_grid": K_GRID.copy(), "carried": carried.copy(), "ref": ref,
        "amp": z.astype(dtype), "mu": mu.astype(dtype), "sigma": sigma.astype(dtype),
        "n_packets": n.astype(np.int32), "sample_mask": smask, "link_mask": link_mask,
        "gain_db": per_sample(gain).astype(dtype), "rssi": per_sample(rssi).astype(dtype),
        "rssi_mask": smask & rssi_mask,
        "t_rel": (per_sample(t_rel) if cfg.bin_hz is None else np.array(t_rel)).astype(dtype),
    }
    if cfg.bin_hz is not None:
        out["bin_count"] = per_sample(bins["count"], 0).astype(np.int32)
    if resid is not None:
        pm = pmask.transpose(0, 2, 1) & smask[:, None, :] & carried[None, :, None]
        comp = np.stack((pc, ps), axis=-1).transpose(0, 2, 1, 3)              # [L, 52, T, 2]
        out["phase"] = np.where(pm[..., None], comp, 0.0).astype(dtype)
        out["phase_mask"] = pm
    out["zero_gain_packets"] = int(zero.sum())
    out["links_dropped"] = int((~link_mask).sum())
    return out


def reference_input(window: CsiWindow) -> dict:
    """A `CsiWindow` as `ml/preprocess_ref.py` takes it: plain lists, CSI as
    `(re, im)` float pairs (the label stays with the caller)."""
    return {
        "csi": [[[(z.real, z.imag) for z in row] for row in link] for link in window.csi.tolist()],
        "received": window.received.tolist(), "carried": window.carried.tolist(),
        "t_rel_ns": window.t_rel_ns.tolist(), "rssi": window.rssi.tolist(),
        "rssi_mask": window.rssi_mask.tolist(), "length_ns": int(window.length_ns),
    }
