"""OpenCSI adapter (Phase 9, spec/phase9.md §2): a public ESP32 dataset as REAL windows.

The OpenCSI LCN 2026 dataset (CC BY 4.0; four 12-minute sessions in three
rooms, eight ESP32 nodes each, 56 directed links, one subject, labels
empty / occupied_static / occupied_moving) read into `ml.preprocess`'s
canonical record, `CsiWindow`, labelled **REAL**: measured rooms, not
Room Sim. Nothing here invents what the dataset does not give: node
positions are unknown (`link_pos_known` false everywhere), so is where the
subject was; the rooms' declared length and width are kept as data.

    from ml import opencsi as oc
    s = oc.load_session(oc.DEFAULT_ROOT, "envC_C3")           # parsed once, then cached
    ends = oc.window_ends(s, 2.0, 0.5)                        # windows wholly inside one labelled segment
    win = oc.window(s, ends[0].t_end_ns, 2.0)                 # a REAL CsiWindow
    oc.labels(s, ends[0])                                     # presence, count, moving

**Layouts (spec/preprocess.md §8, spec/phase9.md §2.2).** The HT20 sessions
(S3, S3', C3) store the first 52 entries of the ESP32's 64-entry LLTF
buffer in FFT order (`k = 0..31, -32..-13`: entries 27..37 are always 0),
so they carry 40 grid bins, `k = -26..-13, 1..26`; the C6 session stores
256 HE-LTF tones in centred order (`t = -128..127`; the null pattern, 0 at
`t = -128..-124`, 0 and `124..127`, fits no other order), and tone `t = 4k`
lies at LLTF subcarrier `k`'s frequency, so it carries all 52 bins. The
dataset does not document its pair order; ESP-IDF's buffer order
`(imag, real)` is assumed, which changes phase, never amplitude (Phase 9
uses amplitude only).

**Records.** Times are the backend's `server_ts`, parsed exactly (decimal,
never through a float) to integer ns after `t0_s`, the first record's whole
second. A malformed line (each of three files ends in one cut-off record)
is skipped and counted, so is an exact duplicate (same link, firmware
timestamp, CSI and RSSI). Per link, records are put in time order (stable).

**Rate cap (spec/model.md decision 2).** The SIM bins hold at most 4
packets per link (20 Hz at 4 nodes); OpenCSI's links average about 22 Hz
but arrive in bursts of about ten packets within a few ms every ~0.25 s
(firmware timestamps), so a uniform 20 Hz decimation would keep one packet
per burst. `cap` keeps at most 4 packets per link per 200 ms cell of a grid
anchored at `t0_s`, evenly spaced through the cell's packets.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np

from ml import preprocess as pp

__all__ = [
    "DEFAULT_ROOT", "DEFAULT_CACHE", "TRIALS", "LABELS", "LAYOUTS", "Segment", "WindowEnd", "Session",
    "parse_trial", "load_session", "window_ends", "window", "labels",
]

DEFAULT_ROOT = "/home/Aman/are_u_theone/datasets/public/opencsi/opencsi-dataset-release"
DEFAULT_CACHE = "/home/Aman/are_u_theone/datasets/public/opencsi/parsed"
#: Each trial: its room, chip and layout (the dataset's README and manifest).
TRIALS = {
    "envA_S3": ("envA", "ESP32-S3", "opencsi-lltf52"),
    "envB_S3prime": ("envB", "ESP32-S3'", "opencsi-lltf52"),
    "envC_C3": ("envC", "ESP32-C3", "opencsi-lltf52"),
    "envA_C6": ("envA", "ESP32-C6", "opencsi-he256"),
}
#: fine_label -> (presence, count, moving). Count is the dataset's: one subject or nobody.
LABELS = {"empty_room": (False, 0, False), "occupied_static": (True, 1, False), "occupied_moving": (True, 1, True)}
_NS = 1_000_000_000
#: The rate cap: at most CAP_PACKETS per link per CAP_CELL_NS (20 Hz, SIM's densest bin).
CAP_PACKETS, CAP_CELL_NS = 4, 200_000_000
_FORMAT = 2   # the parsed cache's format; a change re-parses


def _he_k(e: int):
    """Entry `e` of the C6's 256 centred HE tones (`t = e - 128`) -> LLTF `k` when `t = 4k` is on the grid."""
    t = e - 128
    if t % 4 or t // 4 not in pp.GRID_K:
        return None
    return t // 4


#: spec/preprocess.md §8 layouts of the dataset's `iq_data` (pair order assumed, see the module doc).
LAYOUTS = {
    "opencsi-lltf52": pp.Layout("opencsi-lltf52", tuple(pp._fft_k(e) for e in range(52)), "imag-real"),
    "opencsi-he256": pp.Layout("opencsi-he256", tuple(_he_k(e) for e in range(256)), "imag-real"),
}


@dataclass(frozen=True)
class Segment:
    """One annotation: `[start_ns, end_ns]` after `t0_s`, its fine label and note."""

    start_ns: int
    end_ns: int
    fine_label: str
    note: str


@dataclass(frozen=True)
class WindowEnd:
    """A window `[t_end_ns - length, t_end_ns)` lying wholly inside segment `segment`."""

    t_end_ns: int
    segment: int


@dataclass(frozen=True)
class Session:
    """One parsed trial, REAL. Records are sorted by link, then time;
    `offsets[l]:offsets[l + 1]` are link `l`'s. Only records the rate cap
    keeps are here (`stats` counts the rest)."""

    trial: str
    environment: str
    chip: str
    layout: pp.Layout
    n_nodes: int
    links: np.ndarray          #: [L, 2] int16 (tx, rx) node ids, ascending; node id = rank of its MAC
    offsets: np.ndarray        #: [L + 1] int64
    t_ns: np.ndarray           #: [n] int64 server time minus t0_s, ns
    iq: np.ndarray             #: [n, E, 2] int8, the dataset's entries
    rssi: np.ndarray           #: [n] int16 dBm
    t0_s: int
    segments: tuple            #: Segment, by start
    room_m: tuple              #: (length, width) as the environment file declares
    stats: dict = field(default_factory=dict)
    label: str = "REAL"


# --------------------------------------------------------------------------
# Parsing (once per trial, then cached)
# --------------------------------------------------------------------------


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _ns(value: Decimal, t0_s: int) -> int:
    """`value - t0_s` seconds as integer ns (exact for up to 9 decimals; finer is floored)."""
    return int(((value - t0_s) * _NS).to_integral_value(rounding="ROUND_FLOOR"))


def parse_trial(root: str, trial: str) -> dict:
    """Parse `trials/<trial>/raw/csi_raw.jsonl.zst` into arrays (records in
    file order, malformed lines and exact duplicates dropped and counted),
    with the trial's annotations and room. Returns a dict of numpy arrays
    and metadata (what `load_session` caches)."""
    import zstandard   # ml/'s one extra dependency, for this dataset only

    if trial not in TRIALS:
        raise ValueError(f"trial {trial!r} is not one of {sorted(TRIALS)}")
    env, _chip, lay_name = TRIALS[trial]
    lay = LAYOUTS[lay_name]
    tdir = os.path.join(root, "trials", trial)
    raw = os.path.join(tdir, "raw", "csi_raw.jsonl.zst")
    macs_tx, macs_rx, ts, rssi, fw = [], [], [], [], []
    from array import array
    iq = array("b")
    lines = malformed = duplicates = other_layout = 0
    seen = set()
    entries = lay.entries
    with open(raw, "rb") as fh:
        text = io.TextIOWrapper(zstandard.ZstdDecompressor().stream_reader(fh), encoding="utf-8")
        for line in text:
            lines += 1
            try:
                r = json.loads(line, parse_float=Decimal)
                pairs = r["iq_data"]
                tx, rx, t, q, f = r["tx_mac"], r["rx_mac"], r["server_ts"], r["rssi"], r["firmware_ts_us"]
            except (ValueError, KeyError, TypeError):
                malformed += 1
                continue
            if len(pairs) != entries or r.get("num_subcarriers") != entries:
                other_layout += 1                 # e.g. the C6 session's 149 HT20 records: skipped, counted
                continue
            if not isinstance(t, Decimal) or not isinstance(q, int) or not isinstance(f, int) or tx == rx:
                raise ValueError(f"{trial} line {lines}: server_ts, rssi or firmware_ts_us malformed, or tx == rx")
            flat = [v for pair in pairs for v in pair]
            if len(flat) != 2 * entries:
                raise ValueError(f"{trial} line {lines}: iq_data is not {entries} pairs")
            key = (tx, rx, f, q, tuple(flat))
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            iq.extend(flat)                       # OverflowError for a component outside int8: refused
            macs_tx.append(tx)
            macs_rx.append(rx)
            ts.append(t)
            rssi.append(q)
            fw.append(f)
    if not ts:
        raise ValueError(f"{trial}: no records")
    if other_layout > 0.01 * (len(ts) + other_layout):
        raise ValueError(f"{trial}: {other_layout} records are not layout {lay.name} (over 1%); refused")
    t0_s = int(min(ts).to_integral_value(rounding="ROUND_FLOOR"))
    macs = sorted(set(macs_tx) | set(macs_rx))
    node = {m: i for i, m in enumerate(macs)}
    segs = []
    with open(os.path.join(tdir, "annotations.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            a = json.loads(line, parse_float=Decimal)
            if a["fine_label"] not in LABELS:
                raise ValueError(f"{trial}: annotation label {a['fine_label']!r} is not one of {sorted(LABELS)}")
            segs.append((_ns(Decimal(a["start_ts"]), t0_s), _ns(Decimal(a["end_ts"]), t0_s), a["fine_label"],
                         a.get("note", "")))
    segs.sort()
    for (s0, e0, _, _), (s1, _, _, _) in zip(segs, segs[1:]):
        if s1 < e0:
            raise ValueError(f"{trial}: annotations overlap")
    with open(os.path.join(root, "environments", f"{env}.json"), encoding="utf-8") as fh:
        dims = json.load(fh)["dimensions_m"]
    return {
        "format": _FORMAT, "trial": trial, "raw_sha256": _sha256(raw), "t0_s": t0_s, "n_nodes": len(macs),
        "tx": np.array([node[m] for m in macs_tx], np.int16), "rx": np.array([node[m] for m in macs_rx], np.int16),
        "t_ns": np.array([_ns(t, t0_s) for t in ts], np.int64),
        "iq": np.frombuffer(iq, np.int8).reshape(len(ts), entries, 2).copy(),
        "rssi": np.array(rssi, np.int16), "firmware_ts_us": np.array(fw, np.int64),
        "segments": segs, "room_m": (float(dims["length"]), float(dims["width"])),
        "lines": lines, "malformed": malformed, "duplicates": duplicates, "other_layout": other_layout,
    }


def _cache_paths(cache_dir: str, trial: str) -> tuple:
    return os.path.join(cache_dir, f"{trial}.npz"), os.path.join(cache_dir, f"{trial}.json")


def _parsed(root: str, trial: str, cache_dir: str | None) -> dict:
    """`parse_trial`, through a cache keyed by the raw file's size and mtime (re-parsed when either changes)."""
    raw = os.path.join(root, "trials", trial, "raw", "csi_raw.jsonl.zst")
    st = os.stat(raw)
    stamp = {"format": _FORMAT, "size": st.st_size, "mtime_ns": st.st_mtime_ns}
    if cache_dir is not None:
        arrays, meta = _cache_paths(cache_dir, trial)
        if os.path.exists(arrays) and os.path.exists(meta):
            with open(meta, encoding="utf-8") as fh:
                m = json.load(fh)
            if m.get("stamp") == stamp:
                with np.load(arrays) as z:
                    out = {k: z[k] for k in z.files}
                out.update({k: v for k, v in m.items() if k != "stamp"})
                out["segments"] = [tuple(s) for s in out["segments"]]
                out["room_m"] = tuple(out["room_m"])
                return out
    p = parse_trial(root, trial)
    if cache_dir is not None:
        os.makedirs(cache_dir, exist_ok=True)
        arrays, meta = _cache_paths(cache_dir, trial)
        keys = ("tx", "rx", "t_ns", "iq", "rssi", "firmware_ts_us")
        tmp = arrays + ".tmp.npz"
        np.savez(tmp, **{k: p[k] for k in keys})
        os.replace(tmp, arrays)
        with open(meta + ".tmp", "w", encoding="utf-8") as fh:
            json.dump({"stamp": stamp, **{k: v for k, v in p.items() if k not in keys}}, fh)
        os.replace(meta + ".tmp", meta)
    return p


def _cap_keep(link: np.ndarray, t_ns: np.ndarray, cap: int, cell_ns: int) -> np.ndarray:
    """Records sorted by (link, t): keep at most `cap` per (link, cell),
    evenly spaced: in a cell of `n > cap` packets, ranks
    `(2 i (n - 1) + cap - 1) // (2 (cap - 1))`, `i = 0..cap-1` (first and last kept)."""
    if cap < 2:
        raise ValueError("cap must be at least 2")
    key = link.astype(np.int64) * (1 << 40) + t_ns // cell_ns
    if key.size == 0:
        return np.zeros(0, bool)
    starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
    sizes = np.diff(np.r_[starts, key.size])
    n = np.repeat(sizes, sizes)
    rank = np.arange(key.size) - np.repeat(starts, sizes)
    keep = n <= cap
    big = ~keep
    nb, rb = n[big], rank[big]
    i_lo = (rb * (cap - 1)) // (nb - 1)
    hit = np.zeros(nb.size, bool)
    for i in (i_lo, i_lo + 1):
        ok = i < cap
        hit |= ok & ((2 * i * (nb - 1) + cap - 1) // (2 * (cap - 1)) == rb)
    keep[big] = hit
    return keep


def load_session(root: str = DEFAULT_ROOT, trial: str = "envC_C3", *, cache_dir: str | None = DEFAULT_CACHE,
                 cap: int | None = CAP_PACKETS, cell_ns: int = CAP_CELL_NS) -> Session:
    """One trial as a `Session` (REAL): parsed (or read from `cache_dir`),
    sorted by link and time, the rate cap applied (`cap=None` keeps every
    packet; spec/phase9.md §2.4)."""
    p = _parsed(root, trial, cache_dir)
    env, chip, lay_name = TRIALS[trial]
    n_nodes = int(p["n_nodes"])
    key = p["tx"].astype(np.int64) * n_nodes + p["rx"]
    by_link = np.argsort(key, kind="stable")      # file order within each link
    same = key[by_link][1:] == key[by_link][:-1]
    backwards = int((same & (np.diff(p["t_ns"][by_link]) < 0)).sum())
    order = np.lexsort((p["t_ns"], key))          # stable: file order breaks ties
    key, t_ns = key[order], p["t_ns"][order]
    uniq, link = np.unique(key, return_inverse=True)
    links = np.stack((uniq // n_nodes, uniq % n_nodes), axis=1).astype(np.int16)
    if cap is None:
        keep = np.ones(t_ns.size, bool)
    else:
        keep = _cap_keep(link, t_ns, int(cap), int(cell_ns))
    link, t_ns = link[keep], t_ns[keep]
    idx = order[keep]
    offsets = np.searchsorted(link, np.arange(links.shape[0] + 1)).astype(np.int64)
    stats = {"lines": int(p["lines"]), "malformed": int(p["malformed"]), "duplicates": int(p["duplicates"]),
             "other_layout": int(p["other_layout"]), "records": int(order.size), "backwards_in_file": backwards, "capped_out": int((~keep).sum()),
             "kept": int(keep.sum()), "cap": None if cap is None else [int(cap), int(cell_ns)],
             "raw_sha256": p["raw_sha256"]}
    return Session(
        trial=trial, environment=env, chip=chip, layout=LAYOUTS[lay_name], n_nodes=n_nodes, links=links,
        offsets=offsets, t_ns=t_ns, iq=p["iq"][idx], rssi=p["rssi"][idx], t0_s=int(p["t0_s"]),
        segments=tuple(Segment(int(a), int(b), str(c), str(d)) for a, b, c, d in p["segments"]),
        room_m=tuple(p["room_m"]), stats=stats)


# --------------------------------------------------------------------------
# Windows and labels
# --------------------------------------------------------------------------


def _length_ns(length_s) -> int:
    n = round(Decimal(str(length_s)) * _NS)
    if n <= 0 or Decimal(n) != Decimal(str(length_s)) * _NS:
        raise ValueError(f"length {length_s!r} s is not a positive whole number of ns")
    return int(n)


def window_ends(session: Session, length_s: float, hop_s: float) -> list:
    """Every window `[t_end - length, t_end)` wholly inside one labelled
    segment: `t_end = start + length + j hop`, `j >= 0`, `t_end <= end`
    (windows across a transition or in the unlabelled gaps between
    segments are never made)."""
    ln, hop = _length_ns(length_s), _length_ns(hop_s)
    out = []
    for i, s in enumerate(session.segments):
        t = s.start_ns + ln
        while t <= s.end_ns:
            out.append(WindowEnd(t, i))
            t += hop
    return out


def labels(session: Session, end: WindowEnd) -> dict:
    """The window's labels from its segment: presence, count (0 or 1),
    moving; and what is unknown (positions, zones)."""
    seg = session.segments[end.segment]
    presence, count, moving = LABELS[seg.fine_label]
    return {"presence": presence, "count": count, "moving": moving, "fine_label": seg.fine_label,
            "segment": end.segment, "positions_known": False}


def window(session: Session, t_end_ns: int, length_s: float) -> pp.CsiWindow:
    """The REAL `CsiWindow` `[t_end_ns - length, t_end_ns)` (ns after
    `t0_s`): every link of the session (a link without a packet in it has
    none received), its kept packets right-aligned, positions unknown."""
    ln = _length_ns(length_s)
    t_end_ns = int(t_end_ns)
    n_links = session.links.shape[0]
    lo = np.empty(n_links, np.int64)
    hi = np.empty(n_links, np.int64)
    for l in range(n_links):
        a, b = session.offsets[l], session.offsets[l + 1]
        tl = session.t_ns[a:b]
        lo[l] = a + np.searchsorted(tl, t_end_ns - ln, "left")
        hi[l] = a + np.searchsorted(tl, t_end_ns, "left")
    count = hi - lo
    p = int(count.max(initial=0))
    csi = np.zeros((n_links, p, 52), np.complex128)
    received = np.zeros((n_links, p), bool)
    t_rel = np.zeros((n_links, p), np.int64)
    rssi = np.zeros((n_links, p))
    for l in range(n_links):
        c = int(count[l])
        if c == 0:
            continue
        sl = slice(int(lo[l]), int(hi[l]))
        csi[l, p - c:] = pp.to_grid(session.iq[sl], session.layout)
        received[l, p - c:] = True
        t_rel[l, p - c:] = session.t_ns[sl] - t_end_ns
        rssi[l, p - c:] = session.rssi[sl]
    zero = np.zeros((n_links, 2), bool)
    return pp.CsiWindow(
        label=session.label, source=f"opencsi:{session.trial}", window_end=f"t_end_ns={session.t0_s * _NS + t_end_ns}",
        links=session.links.copy(), tx_known=np.ones(n_links, bool), link_pos=np.zeros((n_links, 2, 3)),
        link_pos_known=zero, csi=csi, received=received, carried=session.layout.carried, t_rel_ns=t_rel,
        rssi=rssi, rssi_mask=received.copy(), length_ns=ln)
