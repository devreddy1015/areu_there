"""Preprocessing v1 (spec/preprocess.md §10): ml/preprocess.py against the
stdlib reference ml/preprocess_ref.py, the invariances, the guards, the
layouts and both adapters.

Windows here are hand-built or drawn at random: a "channel" window has, per
link, a static multipath response over k, a small per-packet change (a
person), a random gain (AGC), a random common phase and timing slope, noise
and int8 rounding; a "noise" window is uniform random int8 CSI (the hardest
case for the phase fit's sums). Synthetic shards come from
test_roomsim_dataset's builders, synthetic pcaps from wifisense's own
PcapWriter and build_esp32_frame. Everything synthetic here is labelled as
its test says (SIM for shard-like data, REAL for recording-like data).

With SAANS_DATASET_DIR set, TestRealDataset compares the two
implementations on windows of the SIM dataset, and TestCrossPath exports a
dataset scene with `saans roomsim-export` (the binary named by SAANS_BIN,
else $CARGO_TARGET_DIR/release/saans, else rust/target/release/saans, else
`saans` on PATH; skipped when there is none) and checks the pcap path gives
the shard path's windows exactly.

Run from the repo root: .venv/bin/python -m unittest discover -s ml/tests
"""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import warnings
from fractions import Fraction

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml import preprocess as pp  # noqa: E402
from ml import preprocess_ref as pr  # noqa: E402
from ml import roomsim_dataset as rd  # noqa: E402
from ml.tests import test_roomsim_dataset as trd  # noqa: E402  (synthetic SIM shards)
from wifisense.sources.esp32_udp import build_esp32_frame  # noqa: E402
from wifisense.sources.recorder import PcapWriter, ReadStats  # noqa: E402

GRID = np.array(pp.GRID_K)
FW56 = pp.layout("esp32-fw56").carried            # the 44 bins the firmware sends
ALL52 = np.ones(52, bool)
TWO_S = 2_000_000_000
DF_HZ = 312_500.0
T_REAL = 1_700_000_000 * 1_000_000_000            # an epoch for synthetic recordings
CONFIGS = (
    pp.PreprocessConfig(),
    pp.PreprocessConfig(phase=True),
    pp.PreprocessConfig(bin_hz=10),
    pp.PreprocessConfig(phase=True, bin_hz=5.0),
    pp.PreprocessConfig(phase=True, ref_bins="carried", min_packets=2),
)
EXACT = ("sample_mask", "link_mask", "n_packets", "rssi_mask", "bin_count", "phase_mask", "zero_gain_packets",
         "links_dropped", "ref", "carried", "k_grid")
SCALED = ("mu", "sigma", "gain_db", "t_rel", "rssi")
FLOATS = SCALED + ("amp", "phase", "link_pos")


def gi(k):
    """The grid bin of subcarrier k."""
    return pp.GRID_K.index(k)


def window(csi, *, received=None, t_rel_ns=None, rssi=None, rssi_mask=None, carried=None, length_ns=TWO_S,
           links=None, label="SIM", source="test:hand-built", window_end="t_end_ns=0"):
    """A CsiWindow from hand-built packets `csi` [L, P, 52] (complex): every
    slot received unless `received` says otherwise (right-aligned), times
    spread evenly over the window, newest last; RSSI -50 dBm."""
    csi = np.array(csi, np.complex128)
    n_links, n_pk = csi.shape[:2]
    received = np.ones((n_links, n_pk), bool) if received is None else np.array(received, bool)
    carried = (np.abs(csi).sum(axis=(0, 1)) > 0) | FW56 if carried is None else np.array(carried, bool)
    if t_rel_ns is None:
        step = length_ns // (n_pk + 1)
        t_rel_ns = np.where(received, -length_ns + (np.arange(n_pk) + 1) * step, 0)
    t_rel_ns = np.array(t_rel_ns, np.int64)
    rssi_mask = received.copy() if rssi_mask is None else np.array(rssi_mask, bool)
    rssi = np.where(rssi_mask, -50.0, 0.0) if rssi is None else np.where(rssi_mask, rssi, 0.0)
    links = [(0, r + 1) for r in range(n_links)] if links is None else links
    links = np.array(links, np.int16).reshape(n_links, 2)
    return pp.CsiWindow(
        label=label, source=source, window_end=window_end, links=links, tx_known=links[:, 0] != -1,
        link_pos=np.zeros((n_links, 2, 3)), link_pos_known=np.zeros((n_links, 2), bool),
        csi=np.where(received[..., None] & carried, csi, 0), received=received, carried=carried,
        t_rel_ns=t_rel_ns, rssi=rssi.astype(np.float64), rssi_mask=rssi_mask, length_ns=length_ns)


def random_window(rng, n_links=6, n_pk=20, carried=FW56, kind="channel", quantise=True, length_ns=TWO_S,
                  grid_times=False, extras=True):
    """A random window (see the module docstring). Links 0..2 get 0, 1 and
    2 packets (below min_packets); with `extras`, some packets have every
    reference bin zero (g == 0), some bins fade to exactly 0, and one link
    repeats one packet (every bin constant)."""
    k = GRID.astype(np.float64)
    csi = np.zeros((n_links, n_pk, 52), np.complex128)
    received = np.zeros((n_links, n_pk), bool)
    t_rel = np.zeros((n_links, n_pk), np.int64)
    rssi = np.zeros((n_links, n_pk))
    rssi_mask = np.zeros((n_links, n_pk), bool)
    for link in range(n_links):
        n = link if link < 3 else int(rng.integers(3, n_pk + 1))
        if kind == "channel":
            taps = int(rng.integers(2, 7))
            delay = rng.uniform(0.0, 300e-9, taps)
            amp = rng.normal(size=taps) + 1j * rng.normal(size=taps)
            static = (amp[None, :] * np.exp(-2j * np.pi * k[:, None] * DF_HZ * delay[None, :])).sum(axis=1)
            static /= np.sqrt(np.mean(np.abs(static) ** 2))
        rows = []
        for _ in range(n):
            if kind == "channel":
                person = (rng.normal() + 1j * rng.normal()) * np.exp(-2j * np.pi * k * DF_HZ * rng.uniform(0, 80e-9))
                phase = rng.uniform(-np.pi, np.pi) - 2 * np.pi * k * DF_HZ * rng.uniform(-50e-9, 50e-9)
                v = rng.uniform(5.0, 40.0) * (static + 0.2 * person) * np.exp(1j * phase)
                v = v + rng.normal(0, 0.5, 52) + 1j * rng.normal(0, 0.5, 52)
            else:
                v = rng.integers(-40, 41, 52) + 1j * rng.integers(-40, 41, 52)
            if quantise:
                v = np.clip(np.round(v.real), -128, 127) + 1j * np.clip(np.round(v.imag), -128, 127)
            rows.append(v)
        rows = np.array(rows, np.complex128).reshape(n, 52)
        if extras and n >= 3:
            if link == 3:
                rows[:] = rows[0]                                # one packet repeated: every bin constant
            else:
                if rng.random() < 0.5:
                    rows[int(rng.integers(n)), pp._SHARED44_MASK] = 0     # every reference bin zero: g == 0
                rows[rng.random(rows.shape) < 0.01] = 0                   # deep fades: exact zeros
        rows[:, ~carried] = 0
        if grid_times:
            slots = np.sort(rng.choice(length_ns // 12_500_000, n, replace=False))
            times = -length_ns + slots * 12_500_000
        else:
            times = np.sort(rng.integers(-length_ns, 0, n))
        csi[link, n_pk - n:] = rows
        received[link, n_pk - n:] = True
        t_rel[link, n_pk - n:] = times
        rssi_mask[link, n_pk - n:] = rng.random(n) < 0.9
        rssi[link, n_pk - n:] = np.where(rssi_mask[link, n_pk - n:], rng.integers(-90, -30, n), 0)
    links = [(tx, rx) for tx in range(4) for rx in range(4) if rx != tx][:n_links]
    return pp.CsiWindow(
        label="SIM", source="test:random", window_end="k=0", links=np.array(links, np.int16),
        tx_known=np.ones(n_links, bool), link_pos=np.zeros((n_links, 2, 3)),
        link_pos_known=np.zeros((n_links, 2), bool), csi=csi, received=received, carried=np.array(carried),
        t_rel_ns=t_rel, rssi=rssi, rssi_mask=rssi_mask, length_ns=length_ns)


def run_ref(win, cfg):
    return pr.preprocess_ref(pp.reference_input(win), cfg.phase, cfg.bin_hz, cfg.ref_bins, cfg.min_packets)


def z_tolerance(ref):
    """§10: z within 1e-12 (1 + |z| + mu/sigma); exact where sigma is 0."""
    mu, sigma, z = (np.asarray(ref[key], np.float64) for key in ("mu", "sigma", "amp"))
    ratio = np.where(sigma > 0, np.abs(mu) / np.where(sigma > 0, sigma, 1.0), 0.0)
    tol = 1e-12 * (1.0 + np.abs(z) + ratio[..., None])
    return np.where((sigma > 0)[..., None], tol, 0.0)


def tolerance(key, ref):
    b = np.asarray(ref[key], np.float64)
    if key == "amp":
        return z_tolerance(ref)
    if key == "phase":
        return np.full(b.shape, 1e-12)
    return 1e-12 * np.maximum(1.0, np.abs(b))


def invariance_tolerance(key, base):
    """§10's tolerance for an invariance: as `tolerance`, except that z is
    unbounded where the base's sigma is 0 (the bound's mu/sigma term): a
    transform that is exact only to rounding turns a constant bin into one
    whose values differ in the last bit, and z there is rounding noise."""
    tol = tolerance(key, base)
    if key == "amp":
        tol = np.where((base["sigma"] == 0)[..., None], np.inf, tol)
    return tol


class Case(unittest.TestCase):
    def assert_no_nan(self, out):
        for key, v in out.items():
            if isinstance(v, np.ndarray) and v.dtype.kind == "f":
                self.assertFalse(np.isnan(v).any(), key)

    def assert_matches_ref(self, win, cfg, ref=None):
        """§10: the numpy float64 values against the reference (masks and
        counts exact, each float field within its tolerance), and the
        float32 outputs the cast of those values, within one float32 ulp
        (plus that tolerance) of the reference's cast."""
        ref = run_ref(win, cfg) if ref is None else ref
        with warnings.catch_warnings():
            warnings.simplefilter("error")                   # no division by zero, no invalid value
            o64 = pp.preprocess(win, cfg, dtype=np.float64)
            o32 = pp.preprocess(win, cfg)
        self.assertEqual(set(o64) - {"label", "source", "window_end", "links", "tx_known", "link_pos",
                                     "link_pos_known"}, set(ref))
        for key in EXACT:
            if key in ref:
                np.testing.assert_array_equal(np.asarray(o64[key]), np.asarray(ref[key]), err_msg=key)
                self.assertEqual(o32[key].dtype if isinstance(o32[key], np.ndarray) else type(o32[key]),
                                 o64[key].dtype if isinstance(o64[key], np.ndarray) else type(o64[key]))
        for key in FLOATS:
            if key not in ref:
                continue
            a, b = o64[key], np.asarray(ref[key], np.float64)
            self.assertEqual(a.shape, b.shape, key)
            tol = tolerance(key, ref)
            bad = np.abs(a - b) > tol
            self.assertFalse(bad.any(), f"{key}: {np.argwhere(bad)[:3].tolist()} numpy {a[bad][:3]} ref {b[bad][:3]}")
            self.assertEqual(o32[key].dtype, np.float32, key)
            np.testing.assert_array_equal(o32[key], a.astype(np.float32), err_msg=key)
            b32 = b.astype(np.float32)
            gap = np.abs(o32[key].astype(np.float64) - b32.astype(np.float64))
            self.assertFalse((gap > np.spacing(np.abs(b32)).astype(np.float64) + tol).any(), key)
        for key in ("label", "source", "window_end"):
            self.assertEqual(o32[key], getattr(win, key))
        self.assert_no_nan(o32)
        self.assert_no_nan(o64)
        return o64, ref


# ---------------------------------------------------------------------------
# §8 layouts and §3 grid
# ---------------------------------------------------------------------------


class TestLayouts(Case):
    def test_fw56_is_the_shard_layout(self):
        """esp32-fw56 reproduces the dataset's subcarrier_k and
        subcarrier_active (spec/dataset.md §4.2), by index."""
        lay = pp.layout("esp32-fw56")
        self.assertEqual(list(lay.k), trd.BUFFER_K)
        self.assertEqual([int(b is not None) for b in lay.grid], trd.ACTIVE)
        self.assertEqual(lay.pairs, "imag-real")
        np.testing.assert_array_equal(np.flatnonzero(lay.carried), [gi(k) for k in pp.SHARED44])

    def test_hand_built_entries(self):
        """Entry e holds (1000 + e, -2000 - e) in the layout's pair order;
        each lands at its k's grid bin as (re, im)."""
        for name, pairs, count, n_carried in (("esp32-fw56", None, 56, 44), ("esp32-lltf64", None, 64, 52),
                                              ("esp-radar-52-asc", "imag-real", 52, 52),
                                              ("esp-radar-52-asc", "real-imag", 52, 52),
                                              ("esp-radar-52-pos-first", "real-imag", 52, 52)):
            with self.subTest(name=name, pairs=pairs):
                lay = pp.layout(name, pairs)
                entries = np.array([(1000 + e, -2000 - e) for e in range(count)], np.int16)
                csi = pp.to_grid(entries, lay)
                self.assertEqual((csi.shape, csi.dtype, int(lay.carried.sum())), ((52,), np.complex128, n_carried))
                for e, k in enumerate(lay.k):
                    if 1 <= abs(k) <= 26:
                        first, second = 1000 + e, -2000 - e
                        want = complex(second, first) if lay.pairs == "imag-real" else complex(first, second)
                        self.assertEqual(csi[gi(k)], want, (e, k))
                self.assertTrue((csi[~lay.carried] == 0).all())

    def test_entry_to_k(self):
        """Hand-checked k: fw56 entry 31 is k 31 (guard, dropped), 32 is -32,
        38 is -26, 55 is -9; pos-first entry 0 is k 1, 26 is -26, 51 is -1."""
        fw = pp.layout("esp32-fw56")
        self.assertEqual((fw.k[1], fw.k[26], fw.k[27], fw.k[31], fw.k[32], fw.k[38], fw.k[55]),
                         (1, 26, 27, 31, -32, -26, -9))
        self.assertEqual((fw.grid[0], fw.grid[27], fw.grid[37], fw.grid[38]), (None, None, None, gi(-26)))
        full = pp.layout("esp32-lltf64")
        self.assertEqual((full.k[63], full.grid[63], full.grid[32]), (-1, gi(-1), None))
        pos = pp.layout("esp-radar-52-pos-first", "imag-real")
        self.assertEqual((pos.k[0], pos.k[25], pos.k[26], pos.k[51]), (1, 26, -26, -1))
        asc = pp.layout("esp-radar-52-asc", "imag-real")
        self.assertEqual(asc.k, pp.GRID_K)

    def test_pair_order_flags(self):
        """The ESP32 buffer is (imag, real) and cannot be read otherwise; an
        esp-radar layout needs its dataset's order; a swap conjugates."""
        with self.assertRaisesRegex(ValueError, "imag-real"):
            pp.layout("esp32-fw56", "real-imag")
        self.assertEqual(pp.layout("esp32-lltf64", "imag-real").pairs, "imag-real")
        for bad in (None, "re-im"):
            with self.assertRaisesRegex(ValueError, "pair order"):
                pp.layout("esp-radar-52-asc", bad)
        with self.assertRaisesRegex(ValueError, "not one of"):
            pp.layout("esp32-fw128")
        entries = np.array([(3, 4)] * 52, np.int8)
        a = pp.to_grid(entries, pp.layout("esp-radar-52-asc", "imag-real"))
        b = pp.to_grid(entries, pp.layout("esp-radar-52-asc", "real-imag"))
        self.assertEqual((a[0], b[0]), (4 + 3j, 3 + 4j))
        np.testing.assert_array_equal(a, 1j * np.conj(b))              # a swap is j * conj: phase mirrored

    def test_wrong_shapes_refused(self):
        for entries in (np.zeros((55, 2), np.int8), np.zeros((56, 3), np.int8), np.zeros(56, np.int8)):
            with self.assertRaisesRegex(ValueError, "want"):
                pp.to_grid(entries, "esp32-fw56")
        with self.assertRaisesRegex(ValueError, "real numbers"):
            pp.to_grid(np.zeros((56, 2), complex), "esp32-fw56")
        np.testing.assert_array_equal(pp.to_grid(np.ones((3, 4, 56, 2), np.int8), "esp32-fw56").shape, (3, 4, 52))

    def test_grid_and_reference_bins(self):
        """§3: k_grid ascending without 0; shared44 is -26..-9, 1..26 and
        refuses a source without them; 'carried' is every carried bin."""
        self.assertEqual(pp.K_GRID.tolist(), list(range(-26, 0)) + list(range(1, 27)))
        np.testing.assert_array_equal(pp._ref_mask("shared44", ALL52), FW56)
        np.testing.assert_array_equal(pp._ref_mask("carried", ALL52), ALL52)
        lacking = FW56.copy()
        lacking[gi(5)] = False
        with self.assertRaisesRegex(ValueError, r"k = \[5\]"):
            pp._ref_mask("shared44", lacking)
        with self.assertRaisesRegex(ValueError, "carries no grid bin"):
            pp._ref_mask("carried", np.zeros(52, bool))
        with self.assertRaisesRegex(ValueError, "k = \\[5\\]"):
            pp.preprocess(window(np.ones((1, 3, 52)), carried=lacking))


# ---------------------------------------------------------------------------
# §2 the record
# ---------------------------------------------------------------------------


class TestRecord(Case):
    def base(self):
        csi = np.ones((2, 3, 52)) * (3 + 4j)
        return dict(csi=csi, received=[[False, True, True], [True, True, True]])

    def test_valid_and_refusals(self):
        win = window(**self.base())
        self.assertEqual(win.csi.shape, (2, 3, 52))
        good = {f: getattr(win, f) for f in pp.CsiWindow.__dataclass_fields__}
        cases = {
            "non-finite": dict(csi=np.where(np.arange(52) == 3, np.nan, win.csi)),
            "non-finite inf": dict(csi=np.where(np.arange(52) == 3, np.inf * 1j, win.csi)),
            "SIM->REAL": dict(label="SIM->REAL"),
            "want complex128": dict(csi=win.csi.astype(np.complex64)),
            "want int16": dict(links=win.links.astype(np.int64)),
            "right-aligned": dict(received=win.received[:, ::-1].copy(), t_rel_ns=win.t_rel_ns[:, ::-1].copy(),
                                  csi=win.csi[:, ::-1].copy(), rssi=win.rssi[:, ::-1].copy(),
                                  rssi_mask=win.rssi_mask[:, ::-1].copy()),
            "0 where not received": dict(csi=np.full_like(win.csi, 1 + 1j) * FW56),
            "bins not carried": dict(carried=FW56 & (np.arange(52) != gi(3))),
            "length_ns": dict(t_rel_ns=np.where(win.received, -TWO_S - 1, 0)),
            "time order": dict(t_rel_ns=np.where(win.received, -np.arange(3) - 1, 0)),
            "rssi": dict(rssi=np.where(win.received, -50.0, 1.0)),
            "ascending": dict(links=win.links[::-1].copy()),
            "tx_known": dict(tx_known=np.array([True, False])),
            "link_pos": dict(link_pos=np.ones((2, 2, 3))),
            "positive integer": dict(length_ns=0),
        }
        for what, change in cases.items():
            with self.subTest(what=what), self.assertRaises(ValueError):
                pp.CsiWindow(**dict(good, **change))

    def test_messages_name_the_problem(self):
        win = window(**self.base())
        good = {f: getattr(win, f) for f in pp.CsiWindow.__dataclass_fields__}
        with self.assertRaisesRegex(ValueError, "non-finite CSI entry; refused"):
            pp.CsiWindow(**dict(good, csi=np.where(np.arange(52) == 0, np.nan, win.csi)))
        with self.assertRaisesRegex(ValueError, "never windows"):
            pp.CsiWindow(**dict(good, label="SIM->REAL"))

    def test_config_refusals(self):
        for bad in (dict(phase=1), dict(bin_hz=0), dict(bin_hz=float("nan")), dict(bin_hz=True),
                    dict(ref_bins="all"), dict(min_packets=1), dict(min_packets=2.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                pp.PreprocessConfig(**bad)
        with self.assertRaises(ValueError):
            pr.preprocess_ref(pp.reference_input(window(np.ones((1, 3, 52)))), min_packets=1)
        with self.assertRaises(TypeError):
            pp.preprocess(pp.reference_input(window(np.ones((1, 3, 52)))))
        with self.assertRaisesRegex(ValueError, "float32 or float64"):
            pp.preprocess(window(np.ones((1, 3, 52))), dtype=np.float16)


# ---------------------------------------------------------------------------
# §10 the reference
# ---------------------------------------------------------------------------


class TestReference(Case):
    def test_random_windows(self):
        """numpy == the stdlib reference on random windows, every config."""
        rng = np.random.default_rng(20261025)
        for seed in range(6):
            for kind, carried, quantise in (("channel", FW56, True), ("channel", ALL52, False),
                                            ("noise", FW56, True), ("noise", ALL52, True)):
                win = random_window(rng, carried=carried, kind=kind, quantise=quantise, grid_times=seed % 2 == 0)
                for cfg in CONFIGS:
                    with self.subTest(seed=seed, kind=kind, carried=int(carried.sum()), cfg=cfg):
                        self.assert_matches_ref(win, cfg)

    def test_by_hand(self):
        """One link, 3 packets of a 44-bin source: every reference bin is
        (11, 0) but k = 1, which is (11 v, 0) for v = 1, 2, 3. So g = 11 (43
        + v) / 44, a'(k=1) = 44 v / (43 + v), the other bins' a' = 44 / (43 +
        v); mu, the unbiased sigma and z follow (exact fractions here)."""
        csi = np.zeros((1, 3, 52), complex)
        csi[0][:, FW56] = 11
        for p, v in enumerate((1, 2, 3)):
            csi[0, p, gi(1)] = 11 * v
        win = window(csi, carried=FW56)
        out, ref = self.assert_matches_ref(win, pp.PreprocessConfig())
        for b, values in ((gi(1), [Fraction(44 * v, 43 + v) for v in (1, 2, 3)]),
                          (gi(-20), [Fraction(44, 43 + v) for v in (1, 2, 3)])):
            mu = sum(values) / 3
            var = sum((x - mu) ** 2 for x in values) / 2
            sigma = math.sqrt(var)
            self.assertAlmostEqual(out["mu"][0, b], float(mu), delta=1e-15)
            self.assertAlmostEqual(out["sigma"][0, b], sigma, delta=1e-15)
            for p, x in enumerate(values):
                self.assertAlmostEqual(out["amp"][0, b, p], float((x - mu)) / sigma, delta=1e-12)
        for p, v in enumerate((1, 2, 3)):
            self.assertAlmostEqual(out["gain_db"][0, p], 20 * math.log10(11 * (43 + v) / 44), delta=1e-12)
        self.assertEqual(out["n_packets"].tolist(), [3])
        self.assertTrue((out["amp"][0, ~FW56] == 0).all() and (out["mu"][0, ~FW56] == 0).all())

    def test_reference_is_stdlib_only(self):
        """ml/preprocess_ref.py imports no numpy (the golden generator's
        interpreter has none): run in a fresh interpreter with numpy
        blocked, it gives this process's reference values exactly."""
        win = random_window(np.random.default_rng(12), n_links=4, n_pk=6)
        data = json.dumps(pp.reference_input(win))
        code = ("import json, sys; sys.modules['numpy'] = None; sys.path.insert(0, sys.argv[1]); "
                "import preprocess_ref as pr; "
                "out = pr.preprocess_ref(json.loads(sys.stdin.read()), True, 10, 'shared44', 3); "
                "print(json.dumps(out))")
        run = subprocess.run([sys.executable, "-I", "-c", code, os.path.join(ROOT, "ml")], input=data,
                             capture_output=True, text=True, timeout=120)
        self.assertEqual(run.returncode, 0, run.stderr)
        here = run_ref(win, pp.PreprocessConfig(phase=True, bin_hz=10))
        self.assertEqual(json.loads(run.stdout), json.loads(json.dumps(here)))

    def test_normative_helpers(self):
        """§4's angle maps -π to π for both signs of zero; magnitude is
        sqrt(re*re + im*im) (int8 pairs: Locate's amplitude, exactly)."""
        for impl in (pr.angle, lambda im, re: float(pp._angle(np.float64(im), np.float64(re)))):
            self.assertEqual(impl(0.0, -1.0), math.pi)
            self.assertEqual(impl(-0.0, -1.0), math.pi)
            self.assertEqual(impl(-0.0, 1.0), 0.0)
            self.assertEqual(impl(1.0, 0.0), math.pi / 2)
        iq = np.array([(im, re) for im in range(-128, 128) for re in range(-128, 128)], np.int8)
        mags = [pr.magnitude(float(re), float(im)) for im, re in iq.tolist()]
        np.testing.assert_array_equal(np.array(mags), rd.amplitude(iq))          # every int8 pair


# ---------------------------------------------------------------------------
# §4 gain invariance
# ---------------------------------------------------------------------------


class TestGainInvariance(Case):
    def scaled(self, win, scale):
        """`win` with packet (l, p) multiplied by scale[l, p]."""
        fields = {f: getattr(win, f) for f in pp.CsiWindow.__dataclass_fields__}
        return pp.CsiWindow(**dict(fields, csi=win.csi * scale[..., None]))

    def check(self, base, moved, scale, exact):
        for key in ("amp", "mu", "sigma", "phase"):
            if key not in base:
                continue
            a, b = moved[key], base[key]
            if exact:
                np.testing.assert_array_equal(a, b, err_msg=key)
            else:
                self.assertFalse((np.abs(a - b) > invariance_tolerance(key, base)).any(), key)
        for key in EXACT:
            if key in base and key != "bin_count":
                np.testing.assert_array_equal(np.asarray(moved[key]), np.asarray(base[key]), err_msg=key)
        shift = 20 * np.log10(scale)
        ok = base["sample_mask"]
        diff = moved["gain_db"][ok] - (base["gain_db"][ok] + shift[ok])
        self.assertFalse((np.abs(diff) > 1e-12 * np.maximum(1.0, np.abs(base["gain_db"][ok]))).any())

    def test_any_positive_scalar(self):
        """Any positive per-packet scalar leaves amp, mu, sigma, phase and
        every mask unchanged within §10's tolerances and shifts gain_db by
        20 log10 c; a power of two leaves them bit for bit, in both
        implementations."""
        rng = np.random.default_rng(7)
        for seed in range(4):
            win = random_window(rng, carried=ALL52 if seed % 2 else FW56, quantise=seed < 2)
            cfg = pp.PreprocessConfig(phase=True, ref_bins="carried" if seed == 3 else "shared44")
            base = pp.preprocess(win, cfg, dtype=np.float64)
            scale = np.exp(rng.uniform(-7, 7, win.received.shape))
            self.check(base, pp.preprocess(self.scaled(win, scale), cfg, dtype=np.float64), scale, exact=False)
            two = 2.0 ** rng.integers(-20, 21, win.received.shape)
            self.check(base, pp.preprocess(self.scaled(win, two), cfg, dtype=np.float64), two, exact=True)
            ref = run_ref(win, cfg)
            ref_two = run_ref(self.scaled(win, two), cfg)
            for key in ("amp", "mu", "sigma", "phase", "phase_mask", "sample_mask"):
                self.assertEqual(ref_two[key], ref[key], key)

    def test_agc_step(self):
        """A 3 dB step on half the packets of a fixed pattern: after
        normalisation every bin is constant to the last bits (sigma/mu below
        1e-15, against 0.17 raw), and gain_db steps by 20 log10(1.4125).
        (Scaled copies agree only to rounding, so sigma is about 1e-16, not
        0, and z there is noise: §10 bounds z by mu/sigma.) A power-of-two
        step leaves the bins exactly constant: sigma 0, z 0."""
        pattern = (20.0 + np.arange(52) % 7) * np.exp(1j * np.arange(52))
        for step, exact in ((1.4125, False), (2.0, True)):
            csi = np.array([[pattern * (1.0 if p < 3 else step) for p in range(6)]])
            out = pp.preprocess(window(csi, carried=FW56), dtype=np.float64)
            ratio = out["sigma"][0][FW56] / out["mu"][0][FW56]
            self.assertLess(ratio.max(), 1e-15)
            raw = np.abs(csi[0][:, FW56])
            self.assertGreater((raw.std(axis=0, ddof=1) / raw.mean(axis=0)).min(), 0.17)
            if exact:
                self.assertTrue((out["amp"] == 0).all() and (out["sigma"] == 0).all())
            self.assertAlmostEqual(out["gain_db"][0, 3] - out["gain_db"][0, 0], 20 * math.log10(step), delta=1e-12)


# ---------------------------------------------------------------------------
# §5 phase
# ---------------------------------------------------------------------------


class TestPhase(Case):
    def test_common_phase_and_slope_invariance(self):
        """Adding any common phase and any timing slope in true k to every
        packet leaves `phase` (and every mask) unchanged within 1e-12."""
        rng = np.random.default_rng(11)
        k = GRID.astype(np.float64)
        for seed in range(4):
            win = random_window(rng, carried=FW56 if seed % 2 else ALL52, quantise=False)
            fields = {f: getattr(win, f) for f in pp.CsiWindow.__dataclass_fields__}
            phi = rng.uniform(-np.pi, np.pi, win.received.shape)
            slope = rng.uniform(-np.pi, np.pi, win.received.shape)
            turned = pp.CsiWindow(**dict(fields, csi=win.csi * np.exp(1j * (phi[..., None] + slope[..., None] * k))))
            for cfg in (pp.PreprocessConfig(phase=True), pp.PreprocessConfig(phase=True, bin_hz=10),
                        pp.PreprocessConfig(phase=True, ref_bins="carried")):
                a = pp.preprocess(win, cfg, dtype=np.float64)
                b = pp.preprocess(turned, cfg, dtype=np.float64)
                np.testing.assert_array_equal(a["phase_mask"], b["phase_mask"])
                self.assertLess(np.abs(a["phase"] - b["phase"]).max(initial=0), 1e-12)
                for key in ("amp", "mu", "sigma"):
                    self.assertFalse((np.abs(a[key] - b[key]) > invariance_tolerance(key, a)).any(), key)

    def test_pure_linear_phase_is_removed(self):
        """A packet whose phase is exactly a common phase plus a slope in k
        (any amplitudes) has residual (cos, sin) = (1, 0) on every bin."""
        rng = np.random.default_rng(3)
        k = GRID.astype(np.float64)
        csi = np.array([[rng.uniform(5, 30, 52) * np.exp(1j * (rng.uniform(-3, 3) + rng.uniform(-0.3, 0.3) * k))
                         for _ in range(4)]])
        out = pp.preprocess(window(csi, carried=ALL52), pp.PreprocessConfig(phase=True), dtype=np.float64)
        self.assertTrue(out["phase_mask"].all())
        self.assertLess(np.abs(out["phase"][..., 0] - 1).max(), 1e-12)
        self.assertLess(np.abs(out["phase"][..., 1]).max(), 1e-12)

    def test_gaps_never_bridged(self):
        """The slope uses consecutive k only: never -1 -> 1 (DC) or -9 -> 1
        (the firmware's gap). A packet whose only non-zero bins sit across a
        gap has no pair, so S == 0 and no phase, in both implementations."""
        shared = pr.consecutive_pairs([gi(k) for k in pp.SHARED44])
        full = pr.consecutive_pairs(list(range(52)))
        self.assertEqual((len(shared), len(full)), (42, 50))
        for pairs in (shared, full):
            self.assertNotIn((gi(-1), gi(1)), pairs)                   # grid neighbours across DC
            self.assertNotIn(gi(-9), [a for a, _ in pairs] if pairs is shared else [])   # -9 has no partner in R
            self.assertTrue(all(pp.GRID_K[b] == pp.GRID_K[a] + 1 for a, b in pairs))
        for carried, ref_bins, lo in ((FW56, "shared44", -9), (ALL52, "carried", -1)):
            csi = np.zeros((1, 4, 52), complex)
            csi[0, :, gi(lo)] = 10 + 2j
            csi[0, :, gi(1)] = 10 - 2j
            win = window(csi, carried=carried)
            cfg = pp.PreprocessConfig(phase=True, ref_bins=ref_bins)
            out, _ = self.assert_matches_ref(win, cfg)
            self.assertFalse(out["phase_mask"].any(), (lo, ref_bins))
            self.assertTrue(out["sample_mask"].all())

    def test_low_amplitude_bins_masked(self):
        """A bin with a' < 0.1 has no phase; a' = 0.1 exactly keeps it."""
        csi = np.zeros((1, 3, 52), complex)
        csi[0][:, FW56] = 10.0
        csi[0, :, gi(5)] = 0.9 + 0j           # g = (42 * 10 + 0.9 + 2) / 44 = 9.611: a' = 0.094 < 0.1
        csi[0, :, gi(6)] = 2.0                # a' = 0.208
        win = window(csi, carried=FW56)
        out, _ = self.assert_matches_ref(win, pp.PreprocessConfig(phase=True))
        self.assertFalse(out["phase_mask"][0, gi(5)].any())
        self.assertTrue(out["phase_mask"][0, gi(6)].all())
        exact = np.zeros((1, 3, 52), complex)
        exact[0][:, ALL52 & ~FW56] = 1.0       # non-reference bins: a' = 1 / 10 = 0.1 exactly
        exact[0][:, FW56] = 10.0
        out = pp.preprocess(window(exact, carried=ALL52), pp.PreprocessConfig(phase=True))
        self.assertTrue(out["phase_mask"][0, gi(-5)].all())

    def test_zero_guards(self):
        """S == 0 (no consecutive non-zero pair) and C == 0 (contributions
        cancel) leave the packet without phase."""
        csi = np.zeros((1, 3, 52), complex)
        csi[0, :, gi(1)] = csi[0, :, gi(2)] = 1.0           # S = 1 + 1 = 2 (a = 0) ...
        csi[0, :, gi(10)] = csi[0, :, gi(11)] = -1.0        # ... and C = 1 + 1 - 1 - 1 = 0
        out, _ = self.assert_matches_ref(window(csi, carried=FW56), pp.PreprocessConfig(phase=True))
        self.assertFalse(out["phase_mask"].any())

    def test_exact_pi_both_zero_signs(self):
        """Real CSI whose sign alternates with k makes every slope term
        negative and real, so S is exactly negative real and a = π,
        whichever sign its zero imaginary part has: both signs of zero in
        the CSI give bit-identical phase, in both implementations."""
        amp = 10.0 + np.arange(52) % 3
        signs = np.where(GRID % 2, -1.0, 1.0) * amp
        outs = []
        for zero in (0.0, -0.0):
            csi = np.zeros((1, 3, 52), np.complex128)
            csi.real[...] = signs
            csi.imag[...] = zero                    # (1j * -0.0 would give +0.0: set the part itself)
            win = window(csi, carried=ALL52)
            self.assertEqual(math.copysign(1.0, win.csi[0, 0, 0].imag), math.copysign(1.0, zero))
            out, ref = self.assert_matches_ref(win, pp.PreprocessConfig(phase=True, ref_bins="carried"))
            outs.append((out, ref))
            self.assertTrue(out["phase_mask"].all())
            self.assertLess(np.abs(out["phase"][..., 0] - 1).max(), 1e-12)       # the residual is 0
        np.testing.assert_array_equal(outs[0][0]["phase"], outs[1][0]["phase"])
        self.assertEqual(outs[0][1]["phase"], outs[1][1]["phase"])
        g = math.fsum(amp.tolist()) / 52
        pairs = pr.consecutive_pairs(list(range(52)))
        for zero in (0.0, -0.0):
            hp = [(v / g, zero / g) for v in signs.tolist()]
            s_re = math.fsum([hp[b][0] * hp[a][0] + hp[b][1] * hp[a][1] for a, b in pairs])
            s_im = math.fsum([hp[b][1] * hp[a][0] - hp[b][0] * hp[a][1] for a, b in pairs])
            self.assertTrue(s_re < 0 and s_im == 0)
            self.assertEqual(pr.angle(s_im, s_re), math.pi)
            self.assertEqual(pr.angle(-s_im, s_re), math.pi)                      # either sign of zero

    def test_slope_and_offset_never_output(self):
        out = pp.preprocess(random_window(np.random.default_rng(1)), pp.PreprocessConfig(phase=True))
        self.assertEqual(out["phase"].shape[-1], 2)
        self.assertFalse({"slope", "offset", "a", "c", "phase_slope"} & set(out))


# ---------------------------------------------------------------------------
# §6 z-score
# ---------------------------------------------------------------------------


class TestZScore(Case):
    def test_reconstruction(self):
        """z sigma + mu returns a': float64 to the tolerance, float32 within
        2^-22 (|z| sigma + mu)."""
        rng = np.random.default_rng(5)
        for carried in (FW56, ALL52):
            win = random_window(rng, carried=carried, quantise=False)
            o64 = pp.preprocess(win, dtype=np.float64)
            o32 = pp.preprocess(win)
            carried_idx = np.flatnonzero(carried).tolist()
            ref = [gi(k) for k in pp.SHARED44]
            for link in np.flatnonzero(o64["link_mask"]):
                for p in np.flatnonzero(o64["sample_mask"][link]):
                    g, a1, _ = pr.normalise([(z.real, z.imag) for z in win.csi[link, p].tolist()], ref, carried_idx)
                    a1 = np.array(a1)
                    back = o64["amp"][link, :, p] * o64["sigma"][link] + o64["mu"][link]
                    self.assertFalse((np.abs(back - a1) > 1e-12 * np.maximum(1.0, a1)).any())
                    z32, s32, m32 = (o32[key][link].astype(np.float64) for key in ("amp", "sigma", "mu"))
                    back32 = z32[:, p] * s32 + m32
                    bound = 2.0 ** -22 * (np.abs(z32[:, p]) * s32 + m32) + 1e-300
                    self.assertTrue((np.abs(back32 - a1) <= bound).all())

    def test_guards(self):
        """n < min_packets drops the link (link_mask false, amp/mu/sigma 0,
        counted; n_packets keeps its n); min_packets 2 keeps a two-packet
        link, whose z is ±1/sqrt(2) with the unbiased sigma."""
        rng = np.random.default_rng(2)
        csi = rng.integers(1, 40, (3, 4, 52)) + 1j * rng.integers(1, 40, (3, 4, 52))
        received = np.array([[False, False, False, True], [False, False, True, True], [True] * 4])
        win = window(csi, received=received, carried=FW56)
        out, _ = self.assert_matches_ref(win, pp.PreprocessConfig())
        self.assertEqual(out["link_mask"].tolist(), [False, False, True])
        self.assertEqual((out["links_dropped"], out["n_packets"].tolist()), (2, [1, 2, 4]))
        for key in ("amp", "mu", "sigma", "gain_db", "rssi", "t_rel"):
            self.assertTrue((out[key][:2] == 0).all(), key)
        self.assertFalse(out["sample_mask"][:2].any() or out["rssi_mask"][:2].any())
        two, _ = self.assert_matches_ref(win, pp.PreprocessConfig(min_packets=2))
        self.assertEqual(two["link_mask"].tolist(), [False, True, True])
        z = two["amp"][1][FW56][:, 2:]
        np.testing.assert_allclose(np.abs(z), 1 / math.sqrt(2), rtol=1e-12)

    def test_constant_bin_whose_variance_is_not_zero(self):
        """Three identical packets of a 52-bin source: the non-reference
        bin k = -5 has a' = 1 / 10 = 0.1 in each. Computed, fsum([0.1] * 3)
        / 3 is 0.10000000000000002 and sigma about 2e-17, not 0; the guard
        compares the values: mu 0.1, sigma 0, z 0, in both implementations."""
        csi = np.zeros((1, 3, 52), complex)
        csi[0][:, FW56] = 10.0
        csi[0][:, ALL52 & ~FW56] = 1.0
        win = window(csi, carried=ALL52)
        mu = math.fsum([0.1] * 3) / 3
        self.assertNotEqual(mu, 0.1)
        self.assertGreater(math.sqrt(math.fsum([(0.1 - mu) ** 2] * 3) / 2), 0.0)
        out, ref = self.assert_matches_ref(win, pp.PreprocessConfig())
        self.assertEqual((out["mu"][0, gi(-5)], out["sigma"][0, gi(-5)]), (0.1, 0.0))
        self.assertTrue((out["amp"] == 0).all())
        self.assertEqual((ref["mu"][0][gi(-5)], ref["sigma"][0][gi(-5)]), (0.1, 0.0))

    def test_zero_gain_packet_dropped(self):
        """A packet whose reference bins are all exactly 0 is dropped (g ==
        0), even when a non-reference bin is not: counted, never a sample."""
        csi = np.ones((1, 4, 52), complex) * (5 + 5j)
        csi[0, 1][FW56] = 0
        win = window(csi, carried=ALL52)
        out, _ = self.assert_matches_ref(win, pp.PreprocessConfig())
        self.assertEqual(out["zero_gain_packets"], 1)
        self.assertEqual(out["sample_mask"].tolist(), [[True, False, True, True]])
        self.assertEqual((out["gain_db"][0, 1], out["n_packets"][0]), (0.0, 3))


# ---------------------------------------------------------------------------
# §7 binning
# ---------------------------------------------------------------------------


class TestBinning(Case):
    def test_integer_bins_and_edges(self):
        """2 s at 5 Hz: 10 bins of 200 ms from -2 s. A packet exactly on an
        edge goes to the later bin; 1 ns before it, the earlier. Empty bins
        are masked; t_rel is the bin centre."""
        times = [-2_000_000_000, -1_800_000_001, -1_800_000_000, -1_000_000_000, -999_999_999, -1]
        csi = np.random.default_rng(4).integers(1, 30, (1, len(times), 52)) * (1 + 1j)
        win = window(csi, t_rel_ns=[times], carried=FW56)
        out, _ = self.assert_matches_ref(win, pp.PreprocessConfig(bin_hz=5, min_packets=2))
        self.assertEqual(out["bin_count"].tolist(), [[2, 1, 0, 0, 0, 2, 0, 0, 0, 1]])
        self.assertEqual(out["sample_mask"].tolist(), [[c > 0 for c in [2, 1, 0, 0, 0, 2, 0, 0, 0, 1]]])
        np.testing.assert_allclose(out["t_rel"][0], np.arange(10) * 0.2 - 1.9, atol=1e-7)
        self.assertTrue((out["amp"][0][:, ~out["sample_mask"][0]] == 0).all())
        self.assertTrue((out["gain_db"][0][~out["sample_mask"][0]] == 0).all())

    def test_amplitude_mean_not_complex_mean(self):
        """Two packets per bin with opposite phase: the bin's amplitude is
        their mean amplitude (here constant, so mu = a'), not |h - h| / 2 = 0."""
        rng = np.random.default_rng(8)
        h = rng.uniform(5, 30, 52) * np.exp(1j * rng.uniform(-3, 3, 52))
        csi = np.array([[h, -h] * 4])
        times = np.repeat(np.arange(4) * 500_000_000 - 2_000_000_000, 2) + np.tile([0, 1], 4)
        win = window(csi, t_rel_ns=[times], carried=FW56)
        out, _ = self.assert_matches_ref(win, pp.PreprocessConfig(bin_hz=2))
        g = np.abs(h[FW56]).mean()
        np.testing.assert_allclose(out["mu"][0][FW56], np.abs(h[FW56]) / g, rtol=1e-13)
        self.assertEqual(out["bin_count"].tolist(), [[2, 2, 2, 2]])

    def test_gain_rssi_and_phase_of_a_bin(self):
        """A bin's gain_db is its packets' mean, its RSSI the mean of those
        with one (masked when none has); identical packets give the
        packet's own phase."""
        csi = np.ones((1, 4, 52), complex) * (3 + 4j)
        csi[0, 1] *= 2.0
        times = [[-1_900_000_000, -1_800_000_000, -600_000_000, -500_000_000]]
        win = window(csi, t_rel_ns=times, carried=FW56, rssi=[[-40.0, -60.0, -70.0, 0.0]],
                     rssi_mask=[[True, True, False, False]])
        out, _ = self.assert_matches_ref(win, pp.PreprocessConfig(phase=True, bin_hz=1, min_packets=2))
        self.assertAlmostEqual(out["gain_db"][0, 0], 20 * math.log10(5) + 10 * math.log10(2), places=5)
        self.assertEqual((out["rssi"][0].tolist(), out["rssi_mask"][0].tolist()), ([-50.0, 0.0], [True, False]))
        single = pp.preprocess(window(csi[:, :2], carried=FW56), pp.PreprocessConfig(phase=True, min_packets=2))
        np.testing.assert_allclose(out["phase"][0, :, 0], single["phase"][0, :, 0], atol=1e-6)

    def test_refused_rates(self):
        win = window(np.ones((1, 3, 52)), carried=FW56, length_ns=1_000_000_000)
        for bin_hz, why in ((3, "not a multiple"), (0.3, "not a positive integer"), (1e-10, "not a positive integer")):
            with self.subTest(bin_hz=bin_hz), self.assertRaisesRegex(ValueError, why):
                pp.preprocess(win, pp.PreprocessConfig(bin_hz=bin_hz))
            with self.assertRaisesRegex(ValueError, why):
                run_ref(win, pp.PreprocessConfig(bin_hz=bin_hz))
        self.assertEqual(pr.bin_layout(0.1, 10 * 10 ** 9), (1, 10 * 10 ** 9))
        self.assertEqual(pr.bin_layout(20, TWO_S), (40, 50_000_000))


# ---------------------------------------------------------------------------
# §2 adapters: SIM shards
# ---------------------------------------------------------------------------


class TestShardAdapter(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self._tmp.name, "sim-set")
        lost = [(1, 60), (1, 61), (5, 70)]
        trd.write_dataset(self.dir, [{"n_nodes": 4, "lost": lost}, {"n_nodes": 12, "n_people": 2}])

    def tearDown(self):
        self._tmp.cleanup()

    def test_window_iq_is_window(self):
        """window_iq's slots are window's; iq and rssi are the stored ones,
        0 where masked; transmission is m*N + tx_slot."""
        for n in range(4, 13):
            arrays = trd.make_arrays(n_nodes=n)
            arrays["csi_mask"] &= (np.random.default_rng(n).random(arrays["csi_mask"].shape) > 0.2).astype(np.uint8)
            scene = trd.in_memory(arrays)
            for length in (0.5, 2.0, 3.0):
                for k in (0, 3, 7):
                    win = rd.window(scene, k, length)
                    iw = rd.window_iq(scene, k, length)
                    for field in ("mask", "link_mask", "column", "inside"):
                        np.testing.assert_array_equal(getattr(iw, field), getattr(win, field))
                    self.assertEqual((iw.iq.dtype, iw.iq.shape), (np.int8, win.amplitude.shape + (2,)))
                    np.testing.assert_array_equal(
                        np.where(win.mask[:, None, :], rd.amplitude(iw.iq), np.nan).astype(np.float32), win.amplitude)
                    self.assertTrue((iw.iq[~np.repeat(iw.mask[:, None, :], 56, axis=1)] == 0).all())
                    np.testing.assert_array_equal(iw.transmission, win.column * n + scene["tx_slot"][:, None])
                    self.assertTrue((iw.rssi[iw.mask] == -50).all() and (iw.rssi[~iw.mask] == 0).all())

    def test_windows(self):
        """Received packets only, right-aligned; t_rel_ns from the column
        the CSI encodes (real part m + 1, imaginary part the link);
        positions, provenance and label."""
        scene = rd.load_scene(self.dir, 0)
        win = pp.from_roomsim_shard(self.dir, 0, 7, 2.0)
        self.assertEqual((win.label, win.source, win.window_end), ("SIM", "sim-set[impairments=on]:gen-000000", "k=7"))
        self.assertEqual((win.csi.shape, win.length_ns), ((12, 40, 52), TWO_S))
        np.testing.assert_array_equal(win.carried, FW56)
        counts = win.received.sum(axis=1)
        self.assertEqual(counts.tolist(), [40, 38, 40, 40, 40, 39] + [40] * 6)    # n in [160, 320): m = 40..79
        for link in range(12):
            tx = int(scene["links"][link, 0])
            got = win.csi[link][win.received[link]]
            m = got[:, gi(1)].real.astype(int) - 1
            self.assertTrue((got[:, FW56].imag == link).all() and (got[:, ~FW56] == 0).all())
            lost = {1: {60, 61}, 5: {70}}.get(link, set())
            self.assertEqual(m.tolist(), [c for c in range(40, 80) if c not in lost])
            np.testing.assert_array_equal(win.t_rel_ns[link][win.received[link]], (m * 4 + tx - 320) * 12_500_000)
        self.assertFalse(win.received[1, :2].any())
        np.testing.assert_array_equal(win.link_pos[:, 0], scene["node_pos"][scene["links"][:, 0]].astype(np.float64))
        np.testing.assert_array_equal(win.link_pos[:, 1], scene["node_pos"][scene["links"][:, 1]].astype(np.float64))
        self.assertTrue(win.link_pos_known.all() and win.tx_known.all())
        both = pp.from_roomsim_shard(self.dir, 0, [3, 7], 2.0, scene=scene)
        self.assertEqual([w.window_end for w in both], ["k=3", "k=7"])
        np.testing.assert_array_equal(both[1].csi, win.csi)
        out = pp.preprocess(win)
        self.assertEqual((out["label"], out["source"]), ("SIM", win.source))

    def test_partial_and_short_windows(self):
        """N = 12, k = 3: some links hold one column fewer (13 of 14); k = 0
        at 2 s starts before the stream; the record stays right-aligned."""
        win = pp.from_roomsim_shard(self.dir, 1, 3, 2.0)
        self.assertEqual(win.csi.shape[:2], (132, 14))
        self.assertEqual(set(win.received.sum(axis=1).tolist()), {13, 14})
        early = pp.from_roomsim_shard(self.dir, 1, 0, 2.0)
        self.assertEqual(early.csi.shape[:2], (132, 4))
        self.assertTrue((early.t_rel_ns[early.received] >= -500_000_000).all())

    def test_refusals_and_profile(self):
        with self.assertRaises(ValueError):
            pp.from_roomsim_shard(self.dir, 0, 7, 1.25)
        with self.assertRaises(IndexError):
            pp.from_roomsim_shard(self.dir, 0, 8, 2.0)
        scene = rd.load_scene(self.dir, 0)
        with self.assertRaisesRegex(ValueError, "not SIM scene 1"):
            pp.from_roomsim_shard(self.dir, 1, 3, 2.0, scene=scene)
        other = os.path.join(self._tmp.name, "off-set")
        trd.write_dataset(other, [{}], impairments="hardware_off")
        self.assertEqual(pp.from_roomsim_shard(other, 0, 3, 2.0).source, "off-set[impairments=hardware_off]:gen-000000")


# ---------------------------------------------------------------------------
# §2 adapters: pcaps
# ---------------------------------------------------------------------------


def datagram(rx, seq, rssi=-50, entries=56, n_antennas=1, value=None, guard=0):
    """A raw-CSI datagram: active entries (imag, real) = (rx, seq % 100 + 1)
    unless `value`, guard entries 27..37 = (0, guard), DC 0."""
    lay = pp.layout("esp32-fw56" if entries == 56 else "esp32-lltf64") if entries in (56, 64) else None
    iq = []
    for e in range(entries):
        on = lay is not None and lay.grid[e] is not None
        if on:
            iq.append(value if value is not None else (rx, seq % 100 + 1))
        elif 27 <= e <= 37:
            iq.append((0, guard))
        else:
            iq.append((0, 0))
    return build_esp32_frame(iq, node_id=rx, sequence=seq, rssi_dbm=rssi, n_antennas=n_antennas)


def write_pcap(path, records):
    """records: (t_ns, payload[, dport])."""
    with PcapWriter(path) as pw:
        for t_ns, payload, *port in records:
            pw.write(t_ns, ("10.1.1.1", 5566), ("10.0.0.1", port[0] if port else 5566), payload)


class TestPcapAdapter(Case):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = self._tmp.name

    def tearDown(self):
        self._tmp.cleanup()

    def recording(self, name="rec.pcap", guard=0):
        """Receivers 3 and 7, one packet each 100 ms from T_REAL (rx 7 from
        +50 ms), seq counting per receiver."""
        records = []
        for i in range(30):
            records.append((T_REAL + i * 100_000_000, datagram(3, i, guard=guard)))
            records.append((T_REAL + 50_000_000 + i * 100_000_000, datagram(7, 100 + i, rssi=-60)))
        path = os.path.join(self.dir, name)
        write_pcap(path, records)
        return path

    def test_windows(self):
        """[end - length, end) in integer ns, right-aligned, (-1, rx) links
        with single_tx, REAL, provenance from the file's sha256."""
        path = self.recording()
        end = T_REAL + 1_000_000_000                      # rx 3: t = 0.0 .. 0.9 s; rx 7: 0.05 .. 0.95 s
        wins = pp.from_pcap(path, 1.0, [end, end + 50_000_000], label="REAL", single_tx=True)
        win = wins[0]
        self.assertEqual((win.label, win.window_end), ("REAL", f"t_end_ns={end}"))
        with open(path, "rb") as fh:
            self.assertEqual(win.source, "pcap:" + hashlib.sha256(fh.read()).hexdigest()[:12])
        self.assertEqual(win.links.tolist(), [[-1, 3], [-1, 7]])
        self.assertFalse(win.tx_known.any() or win.link_pos_known.any())
        self.assertEqual(win.received.sum(axis=1).tolist(), [10, 10])
        np.testing.assert_array_equal(win.t_rel_ns[0], np.arange(-10, 0) * 100_000_000)
        np.testing.assert_array_equal(win.t_rel_ns[1], np.arange(-10, 0) * 100_000_000 + 50_000_000)
        self.assertTrue((win.csi[0][:, FW56] == complex(1, 3) + np.arange(10)[:, None]).all())
        self.assertEqual(win.rssi[1].tolist(), [-60.0] * 10)
        later = wins[1]                                   # [0.05, 1.05): rx 3 loses t = 0.0, keeps 0.1..0.9 and 1.0
        self.assertEqual(later.received.sum(axis=1).tolist(), [10, 10])
        self.assertEqual(later.t_rel_ns[0, 0], -950_000_000)
        one = pp.from_pcap(path, 0.25, T_REAL + 250_000_000, label="REAL", single_tx=True)
        self.assertEqual(one.received.tolist(), [[True, True, True], [False, True, True]])  # 0, .1, .2 / .05, .15
        out = pp.preprocess(win, pp.PreprocessConfig(min_packets=2))
        self.assertEqual(out["label"], "REAL")

    def test_malformed_skipped_and_counted(self):
        """Parser errors (short payload, bad magic, a FeatureFrame), zero
        antennas, zero subcarriers and a count other than the layout's are
        skipped and counted malformed; another port is skipped by the reader."""
        good = datagram(3, 0)
        records = [
            (T_REAL + 1, good),
            (T_REAL + 2, good[:100]),                                       # truncated payload: parse raises
            (T_REAL + 3, b"SANS" + bytes(64)),                              # a FeatureFrame
            (T_REAL + 4, datagram(3, 1, n_antennas=0)),                     # zero antennas: parser zero-fills
            (T_REAL + 5, build_esp32_frame([], node_id=3, sequence=2)),     # zero subcarriers
            (T_REAL + 6, datagram(3, 3, entries=64)),                       # another layout's count
            (T_REAL + 7, datagram(3, 4), 9999),                             # another port
            (T_REAL + 8, datagram(3, 5)),
        ]
        path = os.path.join(self.dir, "bad.pcap")
        write_pcap(path, records)
        st = ReadStats()
        win = pp.from_pcap(path, 1.0, T_REAL + 100, label="REAL", single_tx=True, stats=st)
        self.assertEqual((st.malformed, st.other_port, int(win.received.sum())), (5, 1, 2))
        full = pp.from_pcap(path, 1.0, T_REAL + 100, layout="esp32-lltf64", label="REAL", single_tx=True,
                            stats=ReadStats(), port=None)
        self.assertEqual((int(full.received.sum()), int(full.carried.sum())), (1, 52))

    def test_label_rules(self):
        """A .sim.pcap is SIM whatever the caller says (REAL refused); a
        plain pcap needs the caller's label; a pcap with a SIM manifest of
        the right datagram count is SIM, of the wrong count is not."""
        rec = self.recording()
        with self.assertRaisesRegex(ValueError, "label='REAL'"):
            pp.from_pcap(rec, 1.0, T_REAL, single_tx=True)
        with self.assertRaisesRegex(ValueError, "not 'SIM->REAL'"):
            pp.from_pcap(rec, 1.0, T_REAL, single_tx=True, label="SIM->REAL")
        sim = os.path.join(self.dir, "run.sim.pcap")
        shutil.copy(rec, sim)
        with open(os.path.join(self.dir, "run.manifest.json"), "w") as fh:
            json.dump({"label": "SIM", "gateway_faithful": True, "counts": {"datagrams": 60}}, fh)
        self.assertEqual(pp.from_pcap(sim, 1.0, T_REAL).label, "SIM")
        self.assertEqual(pp.from_pcap(sim, 1.0, T_REAL, label="SIM").label, "SIM")
        with self.assertRaisesRegex(ValueError, "so its label is SIM"):
            pp.from_pcap(sim, 1.0, T_REAL, label="REAL")
        twin = os.path.join(self.dir, "run.pcap")
        shutil.copy(rec, twin)
        self.assertEqual(pp.from_pcap(twin, 1.0, T_REAL).label, "SIM")          # manifest says SIM, 60 records
        with open(os.path.join(self.dir, "run.manifest.json"), "w") as fh:
            json.dump({"label": "SIM", "gateway_faithful": True, "counts": {"datagrams": 59}}, fh)
        with self.assertRaisesRegex(ValueError, "not a Room Sim export"):
            pp.from_pcap(twin, 1.0, T_REAL)

    def test_link_rules(self):
        """Without a link map: a non-gateway-faithful export is refused, a
        recording or an export without a manifest needs single_tx. With
        one: every map link is listed, packets or not; an unmapped datagram
        is refused."""
        rec = self.recording()
        with self.assertRaisesRegex(ValueError, "single_tx=True"):
            pp.from_pcap(rec, 1.0, T_REAL, label="REAL")
        sim = os.path.join(self.dir, "mesh.sim.pcap")
        shutil.copy(rec, sim)
        with self.assertRaisesRegex(ValueError, "single_tx=True"):                 # no manifest beside it
            pp.from_pcap(sim, 1.0, T_REAL)
        self.assertEqual(pp.from_pcap(sim, 1.0, T_REAL, single_tx=True).links.tolist(), [[-1, 3], [-1, 7]])
        with open(os.path.join(self.dir, "mesh.manifest.json"), "w") as fh:
            json.dump({"label": "SIM", "gateway_faithful": False, "counts": {"datagrams": 60}}, fh)
        for kw in ({}, {"single_tx": True}):
            with self.assertRaisesRegex(ValueError, "not gateway-faithful"):
                pp.from_pcap(sim, 1.0, T_REAL, **kw)
        link_map = {(3, i): (5, 3) for i in range(30)}
        link_map.update({(7, 100 + i): (2 if i % 2 else 9, 7) for i in range(30)})
        link_map[(4, 0)] = (2, 4)                                                   # a link with no datagram
        win = pp.from_pcap(sim, 1.0, T_REAL + 1_000_000_000, link_map=link_map)
        self.assertEqual(win.links.tolist(), [[2, 4], [2, 7], [5, 3], [9, 7]])
        self.assertEqual(win.received.sum(axis=1).tolist(), [0, 5, 10, 5])
        self.assertTrue(win.tx_known.all())
        with self.assertRaisesRegex(ValueError, "single_tx=True is for a stream without"):
            pp.from_pcap(sim, 1.0, T_REAL, link_map=link_map, single_tx=True)
        del link_map[(3, 4)]
        with self.assertRaisesRegex(ValueError, "rx 3 seq 4 is not in the link map"):
            pp.from_pcap(sim, 1.0, T_REAL, link_map=link_map)

    def test_layout_check(self):
        """§8 on REAL data, per receiver: the guard entries' mean power at
        least 20 dB below the active entries'. Active (0, 10) has power 100,
        guard (0, 1) power 1: exactly 20 dB, accepted; active (0, 9), power
        81: 19.1 dB, refused naming the receiver. Receiver 8 is clean."""
        def rec(value, guard, name):
            records = [(T_REAL + i, datagram(3, i, value=value, guard=guard)) for i in range(5)]
            records += [(T_REAL + 10 + i, datagram(8, i, value=(1, 10))) for i in range(5)]
            path = os.path.join(self.dir, name)
            write_pcap(path, records)
            return path
        self.assertEqual(pp.from_pcap(rec((0, 10), 1, "ok.pcap"), 1.0, T_REAL + 100, label="REAL",
                                      single_tx=True).label, "REAL")
        with self.assertRaisesRegex(ValueError, r"receiver 3: guard entries 27..37 only 19.1 dB"):
            pp.from_pcap(rec((0, 9), 1, "bad.pcap"), 1.0, T_REAL + 100, label="REAL", single_tx=True)
        sim = rec((0, 9), 40, "loud.sim.pcap")                                     # SIM: not checked
        self.assertEqual(pp.from_pcap(sim, 1.0, T_REAL + 100, single_tx=True).label, "SIM")
        dead = os.path.join(self.dir, "dead.pcap")
        write_pcap(dead, [(T_REAL, datagram(3, 0, value=(0, 0)))])
        with self.assertRaisesRegex(ValueError, "no energy on the active entries"):
            pp.from_pcap(dead, 1.0, T_REAL + 1, label="REAL", single_tx=True)

    def test_positions_from_a_declared_scene_only(self):
        rec = self.recording()
        scene = {"nodes": [{"id": 3, "pos": [1.0, 2.0, 1.5]}, {"id": 7, "pos": [4.0, 0.5, 2.0]}]}
        win = pp.from_pcap(rec, 1.0, T_REAL + 1_000_000_000, label="REAL", single_tx=True, scene=scene)
        self.assertEqual(win.link_pos_known.tolist(), [[False, True], [False, True]])
        self.assertEqual(win.link_pos[:, 1].tolist(), [[1.0, 2.0, 1.5], [4.0, 0.5, 2.0]])
        self.assertTrue((win.link_pos[:, 0] == 0).all())
        path = os.path.join(self.dir, "scene.json")
        with open(path, "w") as fh:
            json.dump(scene, fh)
        self.assertEqual(pp.from_pcap(rec, 1.0, T_REAL, label="REAL", single_tx=True, scene=path).link_pos.tolist(),
                         win.link_pos.tolist())
        with self.assertRaisesRegex(ValueError, "node 7 is not in the declared scene"):
            pp.from_pcap(rec, 1.0, T_REAL, label="REAL", single_tx=True, scene={"nodes": scene["nodes"][:1]})

    def test_argument_refusals(self):
        rec = self.recording()
        for kw, why in ((dict(end_ns=1.5e18), "not an integer"), (dict(end_ns=[T_REAL, 2.0]), "not an integer"),
                        (dict(length_s=0.0), "positive whole number"), (dict(length_s=1e-10), "whole number"),
                        (dict(layout="esp-radar-52-asc"), "pair order"),
                        (dict(layout=pp.layout("esp-radar-52-asc", "imag-real")), "reads ESP32 datagrams")):
            args = dict(dict(length_s=1.0, end_ns=T_REAL, label="REAL", single_tx=True), **kw)
            with self.subTest(kw=kw), self.assertRaisesRegex(ValueError, why):
                pp.from_pcap(rec, **args)

    def test_truth_link_map(self):
        path = os.path.join(self.dir, "x.truth.jsonl")
        lines = [{"label": "SIM", "m": 0, "slot": 0, "t": 0.0, "tx": 1, "people": [], "phone": None,
                  "packets": [{"rx": 2, "link_index": 0, "lost": False, "seq": 0, "rssi": -40, "rssi_true": -40.2},
                              {"rx": 3, "link_index": 1, "lost": True, "seq": None, "rssi": None, "rssi_true": None}]},
                 {"label": "SIM", "m": 0, "slot": 1, "t": 0.0125, "tx": 2, "people": [], "phone": None,
                  "packets": [{"rx": 3, "link_index": 2, "lost": False, "seq": 0, "rssi": -41, "rssi_true": -41.0}]}]
        with open(path, "w") as fh:
            fh.write("".join(json.dumps(line) + "\n" for line in lines))
        self.assertEqual(pp.truth_link_map(path), {(2, 0): (1, 2), (3, 0): (2, 3)})
        lines[1]["label"] = "REAL"
        with open(path, "w") as fh:
            fh.write("".join(json.dumps(line) + "\n" for line in lines))
        with self.assertRaisesRegex(ValueError, "not 'SIM'"):
            pp.truth_link_map(path)


# ---------------------------------------------------------------------------
# Real data: the SIM dataset, and the same scene through saans roomsim-export
# ---------------------------------------------------------------------------


def saans_binary():
    """The `saans` CLI to export with, or None."""
    candidates = [os.environ.get("SAANS_BIN")]
    if os.environ.get("CARGO_TARGET_DIR"):
        candidates.append(os.path.join(os.environ["CARGO_TARGET_DIR"], "release", "saans"))
    candidates += [os.path.join(ROOT, "rust", "target", "release", "saans"), shutil.which("saans")]
    for path in candidates:
        if path and os.path.isfile(path) and os.access(path, os.X_OK):
            return path
    return None


DATASET = os.environ.get("SAANS_DATASET_DIR")


@unittest.skipUnless(DATASET, "SAANS_DATASET_DIR names no real dataset")
class TestRealDataset(Case):
    """numpy == the reference on SIM dataset windows (4, 8 and 12 nodes),
    and the fw56 layout is every checked shard's. Everything here is SIM."""

    def scenes(self, manifest):
        picked = {}
        for e in manifest["scenes"]:
            picked.setdefault(e["n_nodes"], e["index"])
        return [picked[n] for n in (4, 8, 12) if n in picked]

    def test_sim_windows_match_reference(self):
        manifest = rd.load_manifest(DATASET)
        for index in self.scenes(manifest):
            scene = rd.load_scene(DATASET, index, manifest=manifest)
            lay = pp.layout("esp32-fw56")
            self.assertEqual(scene["subcarrier_k"].tolist(), list(lay.k))
            self.assertEqual(scene["subcarrier_active"].astype(bool).tolist(), [b is not None for b in lay.grid])
            for k, cfg in ((0, CONFIGS[0]), (int(scene["window_end_tick"][3]), CONFIGS[1]),
                           (int(scene["window_end_tick"][20]), CONFIGS[3]), (151, CONFIGS[4])):
                win = pp.from_roomsim_shard(DATASET, index, k, 2.0, scene=scene)
                with self.subTest(scene=index, k=k, cfg=cfg):
                    self.assertEqual(win.label, "SIM")
                    self.assert_matches_ref(win, cfg)


@unittest.skipUnless(DATASET, "SAANS_DATASET_DIR names no real dataset")
class TestCrossPath(Case):
    """One function for all data (§10): a dataset scene exported with `saans
    roomsim-export --config <scene>.roomsim.json` (3.5 s, the shard's run
    from t = 0) and read by from_pcap with its truth.jsonl link map gives
    windows identical in every field but source and window_end to
    from_roomsim_shard's, and identical preprocess outputs. SIM."""

    @classmethod
    def setUpClass(cls):
        cls.binary = saans_binary()
        if cls.binary is None:
            raise unittest.SkipTest("no saans binary (set SAANS_BIN, or build saans-cli --release)")
        cls.manifest = rd.load_manifest(DATASET)
        cls._tmp = tempfile.TemporaryDirectory()
        cls.exports = {}
        for n_nodes in (4, 12):
            entry = next((e for e in cls.manifest["scenes"] if e["n_nodes"] == n_nodes), None)
            if entry is None:
                continue
            here = os.path.join(cls._tmp.name, entry["id"])
            os.makedirs(here)
            for suffix in (".json", ".roomsim.json"):
                shutil.copy(os.path.join(DATASET, "scenes", entry["id"] + suffix), here)
            prefix = os.path.join(here, "export")
            run = subprocess.run([cls.binary, "roomsim-export", os.path.join(here, entry["id"] + ".json"),
                                  "--out", prefix, "--config", os.path.join(here, entry["id"] + ".roomsim.json"),
                                  "--duration", "3.5"], capture_output=True, text=True, timeout=300)
            if run.returncode != 0:
                raise AssertionError(f"saans roomsim-export failed: {run.stderr}")
            cls.exports[entry["index"]] = (entry, prefix)

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_identical_windows_and_outputs(self):
        ticks = [0, 1, 3, 4, 5]                    # windows ending at 0.5 .. 3 s, the first two cut by t = 0
        fields = [f for f in pp.CsiWindow.__dataclass_fields__ if f not in ("source", "window_end")]
        for index, (entry, prefix) in self.exports.items():
            scene = rd.load_scene(DATASET, index, manifest=self.manifest)
            link_map = pp.truth_link_map(prefix + ".truth.jsonl")
            declared = {"nodes": [{"id": int(i), "pos": [float(v) for v in p]}
                                  for i, p in zip(scene["node_ids"], scene["node_pos"])]}
            from_pcap = pp.from_pcap(prefix + ".sim.pcap", 2.0, [pp.sim_end_ns(k) for k in ticks],
                                     link_map=link_map, scene=declared)
            from_shard = pp.from_roomsim_shard(DATASET, index, ticks, 2.0, scene=scene)
            for k, a, b in zip(ticks, from_pcap, from_shard):
                with self.subTest(scene=index, k=k):
                    self.assertEqual((a.window_end, b.window_end), (f"t_end_ns={pp.sim_end_ns(k)}", f"k={k}"))
                    self.assertTrue(a.source.startswith("pcap:") and b.source.endswith(entry["id"]))
                    self.assertGreater(int(b.received.sum()), 0)
                    for f in fields:
                        x, y = getattr(a, f), getattr(b, f)
                        if isinstance(x, np.ndarray):
                            self.assertEqual((x.dtype, x.shape), (y.dtype, y.shape), f)
                            np.testing.assert_array_equal(x, y, err_msg=f)
                        else:
                            self.assertEqual(x, y, f)
                    for cfg in CONFIGS:
                        oa, ob = pp.preprocess(a, cfg), pp.preprocess(b, cfg)
                        for key in oa:
                            if key not in ("source", "window_end"):
                                np.testing.assert_array_equal(oa[key], ob[key], err_msg=f"{key} {cfg}")

    def test_scene_file_positions(self):
        """With the scene file itself (float64 positions) instead of the
        shard's float32 node_pos, link_pos differs only by that rounding,
        and the float32 outputs are identical."""
        index, (entry, prefix) = next(iter(self.exports.items()))
        scene = rd.load_scene(DATASET, index, manifest=self.manifest)
        a = pp.from_pcap(prefix + ".sim.pcap", 2.0, pp.sim_end_ns(3), link_map=pp.truth_link_map(prefix + ".truth.jsonl"),
                         scene=os.path.join(DATASET, "scenes", entry["id"] + ".json"))
        b = pp.from_roomsim_shard(DATASET, index, 3, 2.0, scene=scene)
        np.testing.assert_array_equal(a.link_pos.astype(np.float32), b.link_pos.astype(np.float32))
        self.assertLess(np.abs(a.link_pos - b.link_pos).max(), 1e-6)
        np.testing.assert_array_equal(pp.preprocess(a)["link_pos"], pp.preprocess(b)["link_pos"])

    def test_refusals(self):
        """The same pcap without a link map is refused (several transmitters
        interleave), and it is SIM even when the caller says REAL."""
        for entry, prefix in self.exports.values():
            with self.assertRaisesRegex(ValueError, "not gateway-faithful"):
                pp.from_pcap(prefix + ".sim.pcap", 2.0, pp.sim_end_ns(3))
            with self.assertRaisesRegex(ValueError, "so its label is SIM"):
                pp.from_pcap(prefix + ".sim.pcap", 2.0, pp.sim_end_ns(3), label="REAL",
                             link_map=pp.truth_link_map(prefix + ".truth.jsonl"))


if __name__ == "__main__":
    unittest.main()
