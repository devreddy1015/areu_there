"""The committed Locate golden vectors still match this reference.

spec/golden/localize_* (the ported RTI: geometry, invert, measure, track,
and CPython's two-argument math.hypot) and spec/golden/locate_* (the
radius, the baseline, the simulated phone and the Room Sim pipeline) are
written by spec/golden/generate.py on CPython 3.14. This test imports that
script, replaces its `emit` with a capture, re-runs every `gen_localize_*`
and `gen_locate_*` generator and compares every scalar and every array it
would write with the committed files (spec/locate.md §12).

Each array names its own comparison in a `cmp.<array>` scalar (the
generate.py comment block above the Locate generators): `exact` (discrete
values, flags, codes, uniform draws), `bits`, `rule` / `rule:N` /
`rule:each` / `rule:ptr:P` (spec/roomsim.md §14's
`|a - b| <= 1e-9*|b| + 1e-12*M` with M over the stated group), `tie:V`
(an integer RSSI, exact unless the pre-rounding value is within 1e-6 of a
.5 tie) and `le:B` (a rounding-level diagnostic: both at most B). On
CPython < 3.11 `bits` skips only a NaN's sign bit (3.10's `math.hypot`
returns a negative NaN where 3.14's is positive); its payload is still
compared, and from 3.11 on every bit is. A
recorded `input` is compared by the rule (it is regenerated from the same
code, but on CPython < 3.12 the ported modules' builtin `sum()` is not
compensated, so an input built through them moves by an ulp). Every set
that names a scene records `scene_sha256`, which must be the sha256 of
that scene file's bytes. The sets' array layouts are the generators'
docstrings, and every array name must appear in its generator's docstring.

The Room Sim pipeline sets run Room Sim twice each (about 25 s and 50 s
on CPython 3.14). Everything here is SIM.
"""

import hashlib
import importlib.util
import json
import math
import os
import re
import struct
import sys
import unittest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
GOLDEN = os.path.join(ROOT, "spec", "golden")
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from wifisense.roomsim import load_scene  # noqa: E402

#: Every Locate set, in generate.py's main() order.
SETS = (
    "localize_py_hypot", "localize_perimeter", "localize_weights", "localize_weights_bedroom",
    "localize_weights_office", "localize_weights_hall", "localize_prior", "localize_cholesky",
    "localize_inverter_default", "localize_inverter_aniso", "localize_inverter_fine", "localize_rankstarved",
    "localize_forward", "localize_peak", "localize_tracker", "localize_measure",
    "locate_radius", "locate_wlm", "locate_los", "locate_phone_run", "locate_phone_fix", "locate_phone_kf",
    "locate_pipeline_bedroom", "locate_pipeline_office",
)

#: CPython before 3.11 returns a NaN with the sign bit set from
#: `math.hypot(x, nan)` and `math.hypot(nan, x)` for every finite `x` and
#: for a NaN `x` (3.10.12; an infinite `x` gives +inf on every version),
#: where 3.11 and later return a positive NaN; the vectors are 3.14's. On those
#: versions only the sign of a NaN is skipped by the `bits` rule (its
#: payload is still compared); from 3.11 on every bit is compared, and the
#: Rust port (golden.rs) checks every bit against the 3.14 vectors.
NAN_SIGN_UNRELIABLE = sys.version_info < (3, 11)

#: The size the Locate sets stay under (bytes on disk, apparent size).
SIZE_BUDGET = 6_000_000

_generator = None
_captured: dict = {}


def generator():
    """spec/golden/generate.py as a private module whose `emit` captures."""
    global _generator
    if _generator is None:
        spec = importlib.util.spec_from_file_location(
            "locate_golden_generate", os.path.join(GOLDEN, "generate.py"))
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


def groups(rule: str, n: int, arrays: dict) -> list:
    """The (lo, hi) index ranges of a `rule...` comparison over n values."""
    if rule == "rule":
        return [(0, n)]
    if rule == "rule:each":
        return [(i, i + 1) for i in range(n)]
    if rule.startswith("rule:ptr:"):
        out, lo = [], 0
        for c in arrays[rule[len("rule:ptr:"):]]:
            out.append((lo, lo + int(c)))
            lo += int(c)
        assert lo == n, f"{rule}: groups cover {lo} of {n}"
        return out
    size = int(rule[len("rule:"):])
    assert n % size == 0, f"{rule}: {n} values"
    return [(i, i + size) for i in range(0, n, size)]


def close(a: float, b: float, M: float) -> bool:
    """spec/roomsim.md §14's rule; a non-finite b matches only itself."""
    if math.isnan(b):
        return math.isnan(a)
    if math.isinf(b):
        return a == b
    return abs(a - b) <= 1e-9 * abs(b) + 1e-12 * M


def same(a: float, b: float) -> bool:
    return (math.isnan(a) and math.isnan(b)) or float(a) == b


def near_tie(v: float) -> bool:
    return abs(abs(v - math.floor(v)) - 0.5) <= 1e-6


class TestGoldenRerun(unittest.TestCase):
    """Re-run each generator and compare with the committed set."""

    def compare_array(self, name, key, rule, a, b, got, want, want_sc):
        what = f"{name}.{key}"
        self.assertEqual(len(a), len(b), f"{what}: length")
        if rule == "input":
            M = max((abs(x) for x in b if math.isfinite(x)), default=0.0)
            for i, (ai, bi) in enumerate(zip(a, b)):
                ok = same(ai, bi) if isinstance(ai, int) else close(ai, bi, M)
                if not ok:
                    self.fail(f"{what}[{i}] (input): {ai!r} != {bi!r}")
            return
        if rule == "exact":
            for i, (ai, bi) in enumerate(zip(a, b)):
                if not same(ai, bi):
                    self.fail(f"{what}[{i}]: {ai!r} != {bi!r} (must be identical)")
            return
        if rule == "bits":
            for i, (ai, bi) in enumerate(zip(a, b)):
                pa, pb = struct.pack("<d", ai), struct.pack("<d", bi)
                if pa != pb and NAN_SIGN_UNRELIABLE and math.isnan(ai) and math.isnan(bi):
                    # CPython 3.10's hypot returns a negative NaN: compare
                    # every bit but the sign (byte 7's top bit).
                    pa, pb = pa[:7] + bytes([pa[7] & 0x7F]), pb[:7] + bytes([pb[7] & 0x7F])
                if pa != pb:
                    self.fail(f"{what}[{i}]: {ai!r} != {bi!r} (must be bit-identical)")
            return
        if rule.startswith("tie:"):
            pre = want[rule[4:]]
            for i, (ai, bi) in enumerate(zip(a, b)):
                if not same(ai, bi) and not (near_tie(pre[i]) and abs(ai - bi) <= 1.0):
                    self.fail(f"{what}[{i}]: {ai!r} != {bi!r} (an RSSI, not near a tie)")
            return
        if rule.startswith("le:"):
            ref = rule[3:]
            bound = want[ref] if ref in want else [float(ref)] * len(b)
            for i, (ai, bi) in enumerate(zip(a, b)):
                self.assertTrue(ai <= bound[i] and bi <= bound[i], f"{what}[{i}]: {ai!r}, {bi!r} > {bound[i]!r}")
            return
        self.assertTrue(rule.startswith("rule"), f"{what}: unknown cmp {rule!r}")
        for i, ai in enumerate(a):
            if isinstance(ai, int):
                self.fail(f"{what}[{i}]: an integer value compared by {rule!r}, not exactly")
        for lo, hi in groups(rule, len(b), want):
            M = max((abs(x) for x in b[lo:hi] if math.isfinite(x)), default=0.0)
            for i in range(lo, hi):
                if not close(a[i], b[i], M):
                    self.fail(f"{what}[{i}]: {a[i]!r} != {b[i]!r}")

    def compare_scalar(self, name, k, a, b, want_sc):
        what = f"{name}.{k}"
        rule = want_sc.get(f"cmp.{k}")
        if rule is not None and rule.startswith("le:"):
            bound = float(rule[3:])
            if a is None or b is None:
                self.assertEqual(a, b, what)
            else:
                self.assertTrue(a <= bound and b <= bound, f"{what}: {a!r}, {b!r} > {bound!r}")
        elif isinstance(b, float) and not isinstance(a, bool):
            self.assertIsInstance(a, (int, float), what)
            self.assertTrue(close(float(a), b, abs(b) if math.isfinite(b) else 0.0), f"{what}: {a!r} != {b!r}")
        else:
            self.assertEqual(a, int(b) if isinstance(a, bool) else b, what)

    def check_set(self, name):
        got_sc, got = regenerate(name)
        want_sc, want = load(name)
        self.assertEqual(want_sc.get("label"), "SIM", f"{name}: label")
        self.assertEqual(want_sc.get("spec"), "locate_v1", f"{name}: spec")
        self.assertEqual(sorted(got_sc), sorted(want_sc), f"{name}: scalar keys")
        for k, b in want_sc.items():
            self.compare_scalar(name, k, got_sc[k], b, want_sc)
        self.assertEqual(sorted(got), sorted(want), f"{name}: array names")
        for key, b in want.items():
            rule = want_sc.get(f"cmp.{key}")
            self.assertIsNotNone(rule, f"{name}.{key}: no cmp.{key} scalar")
            self.compare_array(name, key, rule, got[key], b, got, want, want_sc)

    def check_scene(self, name):
        sc, _a = load(name)
        if "scene" not in sc:
            self.assertNotIn("scene_sha256", sc, name)
            return
        with open(os.path.join(ROOT, sc["scene"]), "rb") as fh:
            digest = hashlib.sha256(fh.read()).hexdigest()
        self.assertEqual(sc["scene_sha256"], digest, f"{name}: scene_sha256 is not {sc['scene']}'s")
        self.assertEqual(load_scene(os.path.join(ROOT, sc["scene"])).sha256, digest, name)

    def check_documented(self, name):
        """Every array name appears in the generator's docstring (or in the
        docstring of a set it refers to for its layout): as written, without
        its variant prefix or with it spelled `V_`, with its method
        `rtt`/`rssi` spelled `M`, or (the pipelines' `tk_rti_*`) through its
        `tk_wlm_*` twin."""
        sc, arrays = load(name)
        gen = generator()
        doc = getattr(gen, "gen_" + name).__doc__ or ""
        # "layout as <set>": that set's docstring documents the layout.
        seen = {name}
        while True:
            refs = {r for r in re.findall(r"\b((?:localize|locate)_[a-z_]+[a-z])\b", doc)
                    if r not in seen and hasattr(gen, "gen_" + r)}
            if not refs:
                break
            for r in sorted(refs):
                seen.add(r)
                doc += getattr(gen, "gen_" + r).__doc__ or ""
        prefixes = [v + "_" for v in sc["variants"].split(",")] if "variants" in sc else []
        missing = []
        for key in arrays:
            forms = {key}
            for p in prefixes:
                if key.startswith(p):
                    forms.add("V_" + key[len(p):])
                    forms.add(key[len(p):])
            for f in list(forms):
                forms.add(re.sub(r"(^|_)(rtt|rssi)_", r"\1M_", f))
                forms.add(f.replace("tk_rti_", "tk_wlm_"))
            if not any(re.search(r"(?<![A-Za-z0-9_])" + re.escape(f) + r"(?![A-Za-z0-9_])", doc) for f in forms):
                missing.append(key)
        self.assertEqual(missing, [], f"{name}: arrays its docstring does not name")

    def test_manifest_lists_every_set(self):
        with open(os.path.join(GOLDEN, "MANIFEST")) as fh:
            listed = [x for x in fh.read().split() if x.startswith(("localize_", "locate_"))]
        self.assertEqual(listed, list(SETS))
        gens = sorted(n[len("gen_"):] for n in dir(generator()) if n.startswith(("gen_localize_", "gen_locate_")))
        self.assertEqual(gens, sorted(SETS))

    def test_size_budget(self):
        total = sum(os.path.getsize(os.path.join(GOLDEN, f)) for f in os.listdir(GOLDEN)
                    if f.startswith(("localize_", "locate_")))
        self.assertLess(total, SIZE_BUDGET, "the Locate golden sets stay under about 6 MB")


def _add(name):
    def test(self):
        self.check_set(name)
        self.check_scene(name)
        self.check_documented(name)
    test.__name__ = f"test_{name}"
    test.__doc__ = f"gen_{name} reproduces spec/golden/{name}.*"
    setattr(TestGoldenRerun, test.__name__, test)


for _name in SETS:
    _add(_name)


class TestGoldenFacts(unittest.TestCase):
    """What some sets exist to pin, read back from the committed files."""

    def test_py_hypot_hash(self):
        """The random-pair digest is reproducible here, independently of the
        generator's loop."""
        from wifisense.roomsim.rng import Rng
        sc, a = load("localize_py_hypot")
        rng = Rng.stream(sc["rng_seed"], sc["rng_stream"])
        sha = hashlib.sha256()
        pack = struct.Struct("<d").pack
        for i in range(sc["n_random"]):
            x = rng.range(sc["range_lo"], sc["range_hi"])
            y = rng.range(sc["range_lo"], sc["range_hi"])
            sha.update(pack(math.hypot(x, y)))
            if i < len(a["rand_a"]):
                self.assertEqual((x, y), (a["rand_a"][i], a["rand_b"][i]))
        self.assertEqual(sha.hexdigest(), sc["sha256_random"])
        self.assertGreater(sc["n_random_not_sqrt"], 0, "hypot is not sqrt(a*a + b*b)")

    def test_boundary_pin(self):
        sc, a = load("localize_weights")
        self.assertEqual(sc["edge_min_boundary_margin"], 0.0)
        self.assertEqual(sc["edgeup_nnz"], sc["edge_nnz"] + 2, "the boundary pixel joins both directions")
        _sc, p = load("localize_perimeter")
        self.assertEqual(p["ew_weight"][:2], [0.0, 0.5])

    def test_reasons_and_codes_are_known(self):
        from wifisense.localize.radius import REASONS
        for name in SETS:
            sc, a = load(name)
            self.assertEqual(sc["reasons"], ",".join(REASONS), name)
            for k, v in a.items():
                if k.endswith("reason"):
                    self.assertTrue(all(-1 <= x < len(REASONS) and x == int(x) for x in v), f"{name}.{k}")
                if k == "err_code":
                    self.assertTrue(all(0 <= x <= 8 for x in v), f"{name}.{k}")

    def test_phone_fix_pins(self):
        """§12a: noiseless RTT and RSSI recover the truth; the collinear
        nodes are degenerate (at any line direction: node set 5, whose line
        is horizontal but for one ulp); one-sided Huber moves less than
        least squares on a +5 m range (both printed)."""
        from wifisense.localize.radius import REASONS
        sc, a = load("locate_phone_fix")
        self.assertAlmostEqual(a["x"][0], 1.7, delta=1e-6)
        self.assertAlmostEqual(a["y"][0], 2.1, delta=1e-6)
        degenerate = REASONS.index("degenerate")
        self.assertEqual(a["reason"][7], degenerate)
        self.assertLess(sc["huber_shift_m"], sc["ls_shift_m"])
        # §7.1's 200 iterations: case 8 converges in 97 (not within 50) ...
        self.assertEqual((a["reason"][8], a["iterations"][8]), (-1.0, 97.0))
        # ... and two appended cases need more than 200 (the docstring's
        # order: before the last 3 node-set-5 cases, 9th and 3rd from the
        # end); the two before those 3 hit §7.2's start guard (a weight sum
        # of 0, then of inf).
        nc = REASONS.index("not_converged")
        end = len(a["method"]) - 3
        self.assertEqual(a["ns"][end:], [5.0, 5.0, 5.0])
        self.assertEqual((a["reason"][end - 9], a["reason"][end - 3]), (nc, nc))
        self.assertEqual((a["method"][end - 9], a["method"][end - 3]), (0.0, 1.0))
        self.assertEqual(a["reason"][end - 2:end], [degenerate, degenerate])
        # §7's collinear rule: every case on node sets 1, 3, 4 and 5
        # (collinear anchors) with at least 3 reports, the 0.5 m sweep and
        # the last 3 (RTT, RSSI, RSSI rounded on node set 5) included.
        self.assertEqual(a["method"][end:], [0.0, 1.0, 1.0])
        n_line = 0
        for c, (ns, m) in enumerate(zip(a["ns"], a["n_rep"])):
            if ns in (1.0, 3.0, 4.0, 5.0) and m >= 3:
                self.assertEqual(a["reason"][c], degenerate, f"case {c}")
                n_line += 1
        self.assertGreaterEqual(n_line, 4 * 99 + 13)
        print(f"\nlocate_phone_fix (SIM): one range +5 m: one-sided Huber off by {sc['huber_shift_m']:.3f} m, "
              f"least squares by {sc['ls_shift_m']:.3f} m")

    def test_phone_run_edges(self):
        """locate_phone_run variant c reaches what no other set does: a
        non-finite RSSI v gives no report (§6.5), the clamp at -100 and 0,
        `implausible_radius` in both methods, and no_reports, degenerate
        and not_converged from a run; and the min-radius-0 WLM cases abstain
        on a non-finite position (§8), and the last three on a radius that
        is not > 0 though it is >= the minimum."""
        from wifisense.localize.radius import REASONS
        sc, a = load("locate_phone_run")
        bad = [not math.isfinite(v) for v in a["c_rssi_v"]]
        self.assertTrue(any(bad))
        self.assertTrue(all(p == 0.0 for p, b in zip(a["c_rssi_present"], bad) if b))
        dbm = {q for q, p in zip(a["c_rssi_dbm"], a["c_rssi_present"]) if p}
        self.assertTrue({-100.0, 0.0} <= dbm, dbm)
        implausible = REASONS.index("implausible_radius")
        for m in ("rtt", "rssi"):
            self.assertTrue({-1.0, float(implausible)} <= set(a[f"c_{m}_reason"]), m)
            for v in ("a", "b", "c"):
                minr = sc.get(f"{v}_cfg.min_plausible_radius_m", 0.3)
                for reason, r in zip(a[f"{v}_{m}_reason"], a[f"{v}_{m}_radius"]):
                    if reason == -1:
                        self.assertTrue(math.isfinite(r) and r >= minr, f"{v}_{m}: radius {r}")
        need = {float(REASONS.index(r)) for r in ("no_reports", "degenerate", "not_converged")}
        self.assertTrue(need <= set(a["c_rtt_fix_reason"]))
        _sc, w = load("locate_wlm")
        self.assertEqual(w["min_radius"][-6:-3], [0.0, 0.0, 0.0])
        self.assertEqual(w["reason"][-6:-3], [implausible, implausible, -1.0])
        self.assertEqual(w["min_radius"][-3:], [0.0, 0.0, -10.0])
        self.assertEqual(w["radius_scale"][-3:], [0.0, 0.0, -1.0])
        self.assertEqual(math.copysign(1.0, w["radius_scale"][-2]), -1.0)  # -0.0
        self.assertEqual(w["reason"][-3:], [implausible] * 3)

    def test_pipelines_publish_with_a_radius(self):
        """No published estimate has a non-finite radius or one below 0.3 m."""
        for name in ("locate_pipeline_bedroom", "locate_pipeline_office"):
            sc, a = load(name)
            for pre in ("tk_wlm_", "tk_rti_", "ep_rtt_", "ep_rssi_"):
                for reason, r in zip(a[pre + "reason"], a[pre + "radius"]):
                    if reason == -1:
                        self.assertTrue(math.isfinite(r) and r >= 0.3, f"{name}.{pre}radius {r}")
            self.assertEqual(len(a["tk_t"]), sc["n_ticks"])
            self.assertEqual(len(a["ep_t"]), sc["n_epochs"])


if __name__ == "__main__":
    unittest.main()
