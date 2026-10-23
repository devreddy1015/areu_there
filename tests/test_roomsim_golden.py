"""The committed Room Sim golden vectors still match this reference.

spec/golden/roomsim_* are written by spec/golden/generate.py (CPython
3.14). This test imports that script, replaces its `emit` with a capture,
re-runs every `gen_roomsim_*` generator and compares every scalar and
every array it would write with the committed files, by the §14 rule:
`|a - b| <= 1e-9*|b| + 1e-12*M`, `M` the largest `|b_j|` of the group `b`
belongs to (one packet's 64 bins or 128 buffer entries, one link's path
values; for a complex array, the largest magnitude). Integer values
(kinds, flags, counts, sequence numbers, RSSI, int8 buffers, datagram and
pcap bytes), AGC states and the RNG's uniforms must be identical. Every set
that names a scene records `scene_sha256`, which must be the sha256 of
that scene file's bytes.

It also checks the parts the Rust parity test (saans-roomsim/tests/
golden.rs) relies on: the `cfg.*` scalars rebuild the run's
configuration, and the materials and shadow sets reproduce from the
inputs they store. The array layouts are the generators' docstrings. A
set with a `variants` scalar (roomsim_physics_switches) holds several runs
and prefixes each run's scalars and arrays (`a_cfg.*`, `a_seed`, `a_H_re`).
"""

import hashlib
import importlib.util
import json
import math
import os
import struct
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from roomsim_helpers import ROOT  # noqa: E402

from wifisense.roomsim import Config, load_scene  # noqa: E402

GOLDEN = os.path.join(ROOT, "spec", "golden")

#: Every Room Sim set, in generate.py's main() order.
SETS = ("roomsim_materials", "roomsim_shadow", "roomsim_paths", "roomsim_paths_hall", "roomsim_paths_specular",
        "roomsim_motion", "roomsim_channel", "roomsim_impair", "roomsim_multi", "roomsim_switches",
        "roomsim_export", "roomsim_physics_switches")

#: Float arrays that must still be identical (§14: AGC states, RNG output).
EXACT_FLOAT = ("agc_db", "u_loss", "u_jump", "u_dir")

_generator = None
_captured: dict = {}


def generator():
    """spec/golden/generate.py as a private module whose `emit` captures."""
    global _generator
    if _generator is None:
        spec = importlib.util.spec_from_file_location(
            "roomsim_golden_generate", os.path.join(GOLDEN, "generate.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        def emit(name, scalars, arrays):
            _captured[name] = (scalars, arrays)

        mod.emit = emit
        _generator = mod
    return _generator


def regenerate(name):
    """(scalars, arrays) that gen_<name> would write, without writing."""
    if name not in _captured:
        getattr(generator(), "gen_" + name)()
    return _captured[name]


def load(name):
    with open(os.path.join(GOLDEN, f"{name}.json")) as fh:
        meta = json.load(fh)
    arrays = {}
    for k, (fname, n) in meta["arrays"].items():
        with open(os.path.join(GOLDEN, fname), "rb") as fh:
            data = fh.read()
        assert len(data) == 8 * n, f"{fname}: {len(data)} bytes for {n} values"
        arrays[k] = list(struct.unpack(f"<{n}d", data))
    return meta["scalars"], arrays


def run_prefixes(sc):
    """The run prefixes of a set: "" for one run, "a_", "b_", ... for a set
    that lists its runs in `variants` (roomsim_physics_switches)."""
    if "variants" not in sc:
        return [""]
    return [v + "_" for v in sc["variants"].split(",")]


def config_of(sc, prefix=""):
    """Rebuild the run's Config from `cfg.*` scalars (as golden.rs does);
    `prefix` picks one run of a set with `variants`."""
    d = {}
    for k, v in sc.items():
        if not k.startswith(prefix + "cfg."):
            continue
        k = k[len(prefix):]
        parts = k.split(".")
        name = parts[1]
        default = getattr(Config(), name)
        if len(parts) == 3:
            if isinstance(default, tuple):
                d.setdefault(name, list(default))[int(parts[2])] = v
            else:
                d.setdefault(name, {})[parts[2]] = v
        else:
            d[name] = bool(v) if isinstance(default, bool) else v
    return Config.from_dict(d)


def _strip(key):
    for p in ("paths_", "nq_", "a_", "b_"):
        if key.startswith(p):
            return p, key[len(p):]
    return "", key


def groups(key, want):
    """The §14 groups of array `key`: a list of (lo, hi) index ranges."""
    n = len(want[key])
    prefix, base = _strip(key)
    if base.endswith(("_re", "_im")):
        base = base[:-3]
    if base in ("H", "H0", "dev_rdev"):
        size = 64
    elif base in ("buffer", "buffer_pre"):
        size = 128
    elif base in ("coef", "coef0", "L", "sigma", "pts_x", "pts_y", "pts_z") and prefix + "link_npaths" in want:
        out, row, lo = [], 0, 0
        npts = want[prefix + "npts"]
        for count in want[prefix + "link_npaths"]:
            count = int(count)
            if base.startswith("pts_"):
                hi = lo + int(sum(npts[row:row + count]))
            else:
                hi = lo + count
            out.append((lo, hi))
            row += count
            lo = hi
        assert lo == n, f"{key}: groups cover {lo} of {n}"
        return out
    else:
        return [(0, n)]
    assert n % size == 0, f"{key}: {n} is not a multiple of {size}"
    return [(i, i + size) for i in range(0, n, size)]


class TestGoldenRerun(unittest.TestCase):
    """Re-run each generator and compare with the committed set."""

    def check_set(self, name):
        got_sc, got = regenerate(name)
        want_sc, want = load(name)
        # Scalars: the same keys; numbers by the rule (M = |b|), the rest equal.
        self.assertEqual(sorted(got_sc), sorted(want_sc), f"{name}: scalar keys")
        for k, b in want_sc.items():
            a = got_sc[k]
            if isinstance(b, float) and not isinstance(a, bool):
                self.assertTrue(abs(a - b) <= 1e-9 * abs(b) + 1e-12 * abs(b), f"{name}.{k}: {a!r} != {b!r}")
            else:
                self.assertEqual(a, b, f"{name}.{k}")
        self.assertEqual(want_sc.get("label"), "SIM", f"{name}: label")
        # Arrays: the same names and lengths, then every value.
        self.assertEqual(sorted(got), sorted(want), f"{name}: array names")
        for key, b in want.items():
            a = got[key]
            self.assertEqual(len(a), len(b), f"{name}.{key}: length")
            exact_float = _strip(key)[1] in EXACT_FLOAT
            complex_pair = None
            if key.endswith(("_re", "_im")):
                other = key[:-3] + ("_im" if key.endswith("_re") else "_re")
                if other in want:
                    complex_pair = want[other]
            for lo, hi in groups(key, want):
                if complex_pair is not None:
                    M = max((math.sqrt(x * x + y * y) for x, y in zip(b[lo:hi], complex_pair[lo:hi])), default=0.0)
                else:
                    M = max((abs(x) for x in b[lo:hi]), default=0.0)
                for i in range(lo, hi):
                    ai, bi = a[i], b[i]
                    if isinstance(ai, int) or exact_float:
                        if float(ai) != bi:
                            self.fail(f"{name}.{key}[{i}]: {ai!r} != {bi!r} (must be identical)")
                    elif not abs(ai - bi) <= 1e-9 * abs(bi) + 1e-12 * M:
                        self.fail(f"{name}.{key}[{i}]: {ai!r} != {bi!r}")

    def check_scene(self, name):
        sc, _a = load(name)
        if "scene" not in sc:
            self.assertNotIn("scene_sha256", sc, name)
            return
        with open(os.path.join(ROOT, sc["scene"]), "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        self.assertEqual(sc["scene_sha256"], digest, f"{name}: scene_sha256 is not {sc['scene']}'s")
        self.assertEqual(load_scene(os.path.join(ROOT, sc["scene"])).sha256, digest, name)

    def check_config(self, name):
        sc, _a = load(name)
        if "scene" not in sc:
            return
        for pre in run_prefixes(sc):
            cfg = config_of(sc, pre)
            cfg.validate(load_scene(os.path.join(ROOT, sc["scene"])))
            want = {k[len(pre):]: v for k, v in sc.items() if k.startswith(pre + "cfg.")}
            self.assertEqual(generator()._roomsim_cfg_scalars(cfg), want,
                             f"{name}: {pre}cfg.* scalars rebuild the config")
            self.assertEqual(cfg.seed, sc[pre + "seed"], name)
            self.assertEqual(cfg.duration_s, sc[pre + "duration_s"], name)

    def test_manifest_lists_every_set(self):
        with open(os.path.join(GOLDEN, "MANIFEST")) as fh:
            listed = [x for x in fh.read().split() if x.startswith("roomsim_")]
        self.assertEqual(listed, list(SETS))
        gens = sorted(n[len("gen_"):] for n in dir(generator()) if n.startswith("gen_roomsim_"))
        self.assertEqual(gens, sorted(SETS))

    def test_size_budget(self):
        total = sum(os.path.getsize(os.path.join(GOLDEN, f)) for f in os.listdir(GOLDEN) if f.startswith("roomsim_"))
        self.assertLess(total, 4_200_000, "the roomsim golden sets stay under about 4 MB")


def _add(name):
    def test(self):
        self.check_set(name)
        self.check_scene(name)
        self.check_config(name)
    test.__name__ = f"test_{name}"
    test.__doc__ = f"gen_{name} reproduces spec/golden/{name}.*"
    setattr(TestGoldenRerun, test.__name__, test)


for _name in SETS:
    _add(_name)


class TestGoldenInputs(unittest.TestCase):
    """The sets the Rust test rebuilds from stored inputs, rebuilt the same way."""

    def close(self, got, want, what, M=None):
        self.assertEqual(len(got), len(want), what)
        if M is None:
            M = max((abs(b) for b in want), default=0.0)
        for i, (a, b) in enumerate(zip(got, want)):
            if not abs(a - b) <= 1e-9 * abs(b) + 1e-12 * M:
                self.fail(f"{what}[{i}]: {a!r} != {b!r}")

    def test_materials(self):
        from wifisense.roomsim.materials import eta, fresnel, slab
        from wifisense.roomsim.numerics import C, carrier_hz
        sc, a = load("roomsim_materials")
        self.assertNotIn("R_wall_re", a, "walls reflect with the plain slab R (§5.1)")
        fc = carrier_hz(sc["channel"])
        lam = C / fc
        mats = [sc[f"material_{i}"] for i in range(sc["n_materials"])]
        out = {k: [] for k in ("G_re", "G_im", "R_re", "R_im", "T_re", "T_im")}
        for r in range(len(a["row_pol"])):
            m = mats[int(a["row_material"][r])]
            s, g_te, g_tm = fresnel(eta(m, fc), a["row_cos_t"][r])
            G = g_te if a["row_pol"][r] == 0 else g_tm
            R, T, _q0 = slab(G, s, a["row_cos_t"][r], a["row_thickness"][r], lam)
            for k, v in (("G", G), ("R", R), ("T", T)):
                out[k + "_re"].append(v[0])
                out[k + "_im"].append(v[1])
        for k, v in out.items():
            self.close(v, a[k], k, M=1.0)

    def test_shadow(self):
        from wifisense.roomsim.people import J1, shadow_J, three_edge_J
        sc, a = load("roomsim_shadow")
        self.close([J1(v) for v in a["nu"]], a["J1"], "J1")
        self.close([three_edge_J(n1, n2, nt, sc["shadow_cap_db"])
                    for n1, n2, nt in zip(a["grid_nu_1"], a["grid_nu_2"], a["grid_nu_t"])],
                   a["grid_J"], "grid_J")
        Js = []
        for i in range(sc["n_legs"]):
            P1 = (a["leg_p1x"][i], a["leg_p1y"][i], a["leg_p1z"][i])
            P2 = (a["leg_p2x"][i], a["leg_p2y"][i], a["leg_p2z"][i])
            J = shadow_J(P1, P2, (a["leg_cx"][i], a["leg_cy"][i]), a["leg_top"][i], sc["body_radius_m"],
                         sc["shadow_cap_db"], sc["lambda"])
            Js.append(0.0 if J is None else J)
        self.close(Js, a["leg_J"], "leg_J")

    def test_specular_set_has_object_faces(self):
        _sc, a = load("roomsim_paths_specular")
        self.assertEqual(sum(1 for k in a["kind"] if k == 3), 10, "the set exists for its kind-3 paths")

    def test_switches_branches(self):
        """roomsim_switches exercises what its §14 row promises."""
        sc, a = load("roomsim_switches")
        cfg = config_of(sc)
        self.assertTrue(cfg.impair and cfg.quantize and cfg.smoothing)
        self.assertFalse(cfg.cpe or cfg.ripple or cfg.ltf_merge or cfg.snr_noise)
        self.assertNotEqual(cfg.timing_offset_dev_s, (0.0, 0.0))
        self.assertEqual(sorted(cfg.roles.values()), ["rx", "tx"])
        self.assertTrue(cfg.tx_period and cfg.loss_prob_link and cfg.start_heading_rad)
        self.assertEqual((cfg.fw_subcarriers, cfg.channel, cfg.body_points, cfg.phone_carrier), (64, 11, 16, 1))
        pc = cfg.phone_carrier
        self.assertEqual((sc[f"person{pc}_walking"], sc[f"person{pc}_posture"]), (0, "sit"))
        n = sc["n_people"]
        self.assertEqual(set(a["p_posture"][pc::n]), {1.0})
        self.assertEqual(set(a["p_heading"][pc::n]), {cfg.start_heading_rad[pc]})
        self.assertGreater(sc["n_lost"], 0)
        self.assertEqual(sc["nq_quantize"], 0)
        self.assertGreater(sc["nq_n_received"], 0)
        self.assertEqual((sc["nq_n_datagrams"], len(a["nq_dgram"]), len(a["nq_dgram_len"])), (0, 0, 0))
        self.assertEqual(a["nq_buffer_re"], a["nq_buffer_pre_re"])
        self.assertTrue(any(x != round(x) for x in a["nq_buffer_re"]), "quantize false keeps floats")

    def test_physics_switches_branches(self):
        """roomsim_physics_switches exercises what its §14 row promises, and
        each switch shows in the stored paths and truth."""
        sc, a = load("roomsim_physics_switches")
        self.assertEqual(sc["scene"], "spec/scenes/hall-l-shaped.json")
        self.assertEqual(run_prefixes(sc), ["a_", "b_"])
        ca, cb = config_of(sc, "a_"), config_of(sc, "b_")
        self.assertFalse(ca.impair or cb.impair)
        self.assertEqual(sorted((ca.reflection_order, cb.reflection_order)), [0, 1])
        both = (ca, cb)
        for sw in ("floor_ceiling", "transmission", "object_specular", "object_scatter", "object_blockage", "tail",
                   "shadowing"):
            self.assertTrue(any(not getattr(c, sw) for c in both), f"{sw} false in no run")
        self.assertTrue(any(c.object_scatter and c.object_scatter_scale != 1.0 for c in both))
        self.assertTrue(any(c.clutter_count != 24 for c in both))
        scene = load_scene(os.path.join(ROOT, sc["scene"]))
        n_scene = len(scene.people)
        for pre, c in (("a_", ca), ("b_", cb)):
            kind = [int(k) for k in a[pre + "kind"]]
            surf = [int(x) for x in a[pre + "surf_i"]] + [int(x) for x in a[pre + "surf_j"]]
            self.assertEqual(sc[pre + "n_packets"] * 64, len(a[pre + "H_re"]))
            self.assertEqual(sc[pre + "n_clutter"], c.clutter_count)
            self.assertEqual(kind.count(5), sc[pre + "n_clutter"] * sc[pre + "n_links"])
            if c.reflection_order < 2:
                self.assertNotIn(2, kind)
            if c.reflection_order < 1:
                self.assertNotIn(1, kind)
            if not c.floor_ceiling:
                self.assertEqual(sc[pre + "n_surfaces"], sc[pre + "n_walls"])
                self.assertLess(max(surf), sc[pre + "n_walls"])
            if not c.tail:
                self.assertNotIn(6, kind)
            if not c.object_scatter:
                self.assertNotIn(4, kind)
            if not c.object_specular:
                self.assertNotIn(3, kind)
                self.assertIn(4, kind)
            else:
                self.assertIn(3, kind, "a specular face to keep")
            if c.n_people is not None and c.n_people > n_scene:
                # The added people enter at the door: stationary there, as overridden.
                for i in range(n_scene, c.n_people):
                    self.assertEqual(c.people_mode.get(i), "stationary")
                    self.assertEqual(sc[f"{pre}person{i}_walking"], 0)
                    self.assertEqual(sc[f"{pre}person{i}_heading0"], c.start_heading_rad[i])
                    n = sc[pre + "n_people"]
                    self.assertEqual(set(a[pre + "p_heading"][i::n]), {c.start_heading_rad[i]})
            for i, posture in c.people_posture.items():
                self.assertEqual(sc[f"{pre}person{i}_posture"], posture)
        self.assertTrue(ca.n_people is not None and ca.n_people > n_scene, "a person added at the door")
        self.assertTrue(ca.people_mode and ca.people_posture and ca.start_heading_rad)
        # people[1] sits on the sofa by default; run a makes it stand there.
        self.assertEqual((sc["b_person1_posture"], sc["a_person1_posture"]), ("sit", "stand"))
        self.assertEqual(sc["a_person1_on_object"], sc["b_person1_on_object"])

    def test_export_consistent(self):
        """roomsim_export's truth and manifest values agree with each other."""
        sc, a = load("roomsim_export")
        self.assertEqual(len(a["pcap"]), sc["pcap_len"])
        self.assertTrue(all(0 <= b <= 255 and b == int(b) for b in a["pcap"]))
        self.assertEqual(sum(a["tr_npackets"]), sc["n_packets"])
        self.assertEqual(sc["counts_packets"], sc["n_packets"])
        self.assertEqual(sc["counts_transmissions"], sc["n_transmissions"])
        self.assertEqual(sc["counts_lost"], sum(a["pk_lost"]))
        self.assertEqual(sc["counts_datagrams"], sc["n_packets"] - sc["counts_lost"])
        self.assertGreater(sc["counts_lost"], 0)
        for lost, seq, rssi, rt in zip(a["pk_lost"], a["pk_seq"], a["pk_rssi"], a["pk_rssi_true"]):
            if lost:
                self.assertEqual((seq, rssi, rt), (-1.0, 0.0, 0.0), "lost-packet sentinels")
        rxs = [int(x) for x in sc["receivers"].split(",")]
        self.assertEqual(rxs, sorted(rxs))
        self.assertEqual(len(rxs), sc["n_receivers"])
        for rx in rxs:
            want = 0.0
            for tx_, rx_, rate in zip(a["link_tx"], a["link_rx"], a["link_rate_hz"]):
                if rx_ == rx:
                    want += rate
            self.assertEqual(sc[f"receiver_rate_hz.{rx}"], want, f"receiver {rx}")


if __name__ == "__main__":
    unittest.main()
