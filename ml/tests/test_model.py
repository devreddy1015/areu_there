"""Model v1 (spec/model.md §2, §7): size, determinism, the radius floor,
Hungarian matching and the losses on hand-built cases, padding invariance,
the unknown-geometry rule, and overfitting a tiny batch.

Batches are synthetic (ml.train.train.SyntheticScenes: the exact §1 batch
layout, SIM-like, labelled SIM); everything runs on the CPU in float32.

Run from the repo root: .venv/bin/python -m unittest discover -s ml/tests
"""

import math
import os
import sys
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.train import model as mdl  # noqa: E402
from ml.train import train as tr  # noqa: E402


def synthetic_batch(n=8, seed=0, window_s=2.0, train=False, n_nodes=(4, 6), **kw):
    ds = tr.SyntheticScenes(n, window_s=window_s, seed=seed, train=train, n_nodes=n_nodes, **kw)
    return tr.collate_pad([ds[i] for i in range(n)])


def small_model(d=32, seed=0, **kw):
    m = mdl.build_model(mdl.ModelConfig(d=d, **kw), seed=seed)
    m.eval()
    return m


def outputs(model, batch):
    with torch.no_grad():
        return {k: v.clone() for k, v in model(batch).items()}


def batch_of(xy_people, slots_xy=None, room=(10.0, 10.0, 3.0), known=True):
    """A one-sample loss batch: people at `xy_people` (m)."""
    xy = torch.zeros(1, 3, 2)
    pm = torch.zeros(1, 3, dtype=torch.bool)
    for i, p in enumerate(xy_people):
        xy[0, i] = torch.tensor(p)
        pm[0, i] = True
    return {"xy": xy, "person_mask": pm, "geometry_known": torch.tensor([known]),
            "presence": torch.tensor([float(len(xy_people) > 0)]), "count": torch.tensor([len(xy_people)]),
            "zone_occupied": torch.zeros(1, 9), "zone_reachable": torch.ones(1, 9, dtype=torch.bool),
            "room": torch.tensor([room]), "amp": torch.zeros(1, 1, 44, 10)}


def outputs_of(slots_xy, sigma=(0.5, 0.5, 0.5), logit=(0.0, 0.0, 0.0)):
    return {"presence_logit": torch.zeros(1), "count_logits": torch.zeros(1, 4), "zone_logits": torch.zeros(1, 9),
            "slot_logit": torch.tensor([logit]), "slot_xy": torch.tensor([slots_xy], dtype=torch.float32),
            "slot_sigma": torch.tensor([sigma], dtype=torch.float32)}


class TestSize(unittest.TestCase):
    def test_under_ten_million_at_every_width(self):
        for d in (64, 128, 256):
            for phase, gain in ((False, False), (True, True)):
                n = mdl.count_parameters(mdl.build_model(mdl.ModelConfig(d=d, phase=phase, gain=gain), seed=0))
                self.assertLess(n, 10_000_000, (d, phase, gain))
        # the spec's sketch measured 0.18M, 0.68M, 2.65M; this model is within 6% of it
        counts = [mdl.count_parameters(mdl.build_model(mdl.ModelConfig(d=d), seed=0)) for d in (64, 128, 256)]
        for n, ref in zip(counts, (0.18e6, 0.68e6, 2.65e6)):
            self.assertLess(abs(n / ref - 1), 0.06, counts)

    def test_time_conv_is_a_conv1d(self):
        torch.manual_seed(0)
        tc = mdl.TimeConv(5, 7)
        conv = torch.nn.Conv1d(5, 7, 3, padding=1)
        with torch.no_grad():                                  # lin weight [d, 3C] = taps (t-1, t, t+1) x C
            conv.weight.copy_(tc.lin.weight.view(7, 3, 5).permute(0, 2, 1))
            conv.bias.copy_(tc.lin.bias)
        x = torch.randn(4, 9, 5)
        torch.testing.assert_close(tc(x), conv(x.transpose(1, 2)).transpose(1, 2), atol=1e-5, rtol=1e-5)
        self.assertEqual(sum(p.numel() for p in tc.parameters()), sum(p.numel() for p in conv.parameters()))


class TestDeterminism(unittest.TestCase):
    def test_same_seed_same_forward(self):
        batch = synthetic_batch(6, seed=3)
        a = outputs(small_model(seed=11), batch)
        b = outputs(small_model(seed=11), batch)
        for k in a:
            self.assertTrue(torch.equal(a[k], b[k]), k)
        c = outputs(small_model(seed=12), batch)
        self.assertFalse(torch.equal(a["presence_logit"], c["presence_logit"]))

    def test_build_model_leaves_global_rng_alone(self):
        torch.manual_seed(5)
        x = torch.rand(3)
        torch.manual_seed(5)
        small_model(seed=1)
        self.assertTrue(torch.equal(torch.rand(3), x))

    def test_synthetic_data_deterministic(self):
        a, b = synthetic_batch(4, seed=2, train=True), synthetic_batch(4, seed=2, train=True)
        for k in a:
            if isinstance(a[k], torch.Tensor):
                self.assertTrue(torch.equal(a[k], b[k]), k)


class TestRadius(unittest.TestCase):
    def _forced(self, bias):
        m = small_model(seed=0)
        with torch.no_grad():
            m.head_slot[-1].weight[3].zero_()
            m.head_slot[-1].bias[3] = bias
        return outputs(m, synthetic_batch(4))

    def test_sigma_bounds_and_radius_floor(self):
        lo = self._forced(-1e4)["slot_sigma"]
        hi = self._forced(1e4)["slot_sigma"]
        self.assertTrue(torch.all(lo == mdl.SIGMA_MIN_M))
        self.assertTrue(torch.all(hi == mdl.SIGMA_MAX_M))
        out = outputs(small_model(seed=4), synthetic_batch(8, seed=4))
        s = out["slot_sigma"]
        self.assertTrue(torch.all((s >= mdl.SIGMA_MIN_M) & (s <= mdl.SIGMA_MAX_M)))
        for sig in (lo, hi, s):
            self.assertTrue(torch.all(mdl.radius95(sig) >= 0.3))
            self.assertTrue(np.all(mdl.radius95(sig.numpy()) >= 0.3))

    def test_the_floor_is_what_guarantees_0_3_m(self):
        # 2.448 x 0.1225 = 0.29988 m: sigma's floor alone falls 0.12 mm short; radius95 floors at 0.3 m.
        self.assertLess(mdl.R95 * mdl.SIGMA_MIN_M, 0.3)
        self.assertEqual(float(mdl.radius95(torch.tensor(mdl.SIGMA_MIN_M))), float(np.float32(0.3)))
        self.assertEqual(float(mdl.radius95(np.float64(mdl.SIGMA_MIN_M))), 0.3)
        self.assertAlmostEqual(float(mdl.radius95(torch.tensor(1.0))), 2.448, places=6)
        self.assertAlmostEqual(math.sqrt(-2 * math.log(0.05)), mdl.R95, places=3)

    def test_cap_passes_gradient(self):
        m = small_model(seed=0)
        with torch.no_grad():
            m.head_slot[-1].bias[3] = 50.0                      # far above the cap
        m.train()
        out = m(synthetic_batch(2))
        self.assertTrue(torch.all(out["slot_sigma"] <= mdl.SIGMA_MAX_M))
        out["slot_sigma"].sum().backward()
        self.assertGreater(float(m.head_slot[-1].bias.grad[3].abs()), 0.0)


class TestMatching(unittest.TestCase):
    def _both(self, slots, people, mask=None):
        pred = torch.tensor([slots], dtype=torch.float32)
        xy = torch.zeros(1, 3, 2)
        if people:
            xy[0, :len(people)] = torch.tensor(people, dtype=torch.float32)
        pm = torch.zeros(1, 3, dtype=torch.bool)
        pm[0, :len(people)] = True
        if mask is not None:
            pm = torch.tensor([mask])
        a = mdl.match_enumerate(pred, xy, pm)
        b = mdl.match_scipy(pred, xy, pm)
        self.assertTrue(torch.equal(a, b), (a, b))
        return a[0].tolist()

    def test_hand_built(self):
        slots = [(1, 1), (5, 5), (9, 9)]
        self.assertEqual(self._both(slots, [(5.2, 5.0), (1.0, 0.8)]), [1, 0, -1])
        self.assertEqual(self._both(slots, [(8.0, 9.5)]), [2, -1, -1])
        self.assertEqual(self._both(slots, []), [-1, -1, -1])
        self.assertEqual(self._both(slots, [(9, 9), (1, 1), (5, 5)]), [2, 0, 1])
        # greedy (person 0 to its nearest slot) costs 1 + 5 = 6; the optimum 2 + 2 = 4
        self.assertEqual(self._both([(0, 0), (3, 0), (50, 50)], [(1, 0), (-2, 0)]), [1, 0, -1])
        # a person masked out takes no slot even when it sits on one
        self.assertEqual(self._both(slots, [(1, 1), (5, 5)], mask=[False, True, False]), [-1, 1, -1])

    def test_random_cases_agree_with_scipy(self):
        g = torch.Generator().manual_seed(0)
        pred = torch.rand(500, 3, 2, generator=g) * 10
        xy = torch.rand(500, 3, 2, generator=g) * 10
        pm = torch.rand(500, 3, generator=g) < 0.6
        a = mdl.match_enumerate(pred, xy, pm)
        b = mdl.match_scipy(pred, xy, pm)
        d = mdl.pairwise_dist(pred, xy)

        def cost(s):
            c = torch.zeros(500)
            for p in range(3):
                ok = s[:, p] >= 0
                c[ok] += d[torch.arange(500)[ok], p, s[ok, p]]
            return c

        torch.testing.assert_close(cost(a), cost(b))
        self.assertTrue(torch.equal(a, b))                     # continuous random costs: no ties
        self.assertTrue(torch.equal(a >= 0, pm))

    def test_loss_by_hand(self):
        slots = [(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)]
        sig = (0.5, 1.0, 2.0)
        out = outputs_of(slots, sigma=sig, logit=(0.3, -0.2, 0.1))
        batch = batch_of([(5.0, 6.0), (1.0, 2.0)])
        for matcher in ("enumerate", "scipy"):
            res = mdl.compute_loss(out, batch, matcher=matcher)
            # person 0 -> slot 1 (error 1 m, sigma 1), person 1 -> slot 0 (error 1 m, sigma 0.5)
            nll = [math.log(2 * math.pi) + 2 * math.log(1.0) + 1.0 / 2.0,
                   math.log(2 * math.pi) + 2 * math.log(0.5) + 1.0 / (2 * 0.25)]
            self.assertAlmostEqual(float(res["xy_nll"]), sum(nll) / 2, places=5)
            bce = lambda z, y: math.log1p(math.exp(-z)) if y else math.log1p(math.exp(z))  # noqa: E731
            exist = (bce(0.3, 1) + bce(-0.2, 1) + bce(0.1, 0)) / 3
            self.assertAlmostEqual(float(res["exist"]), exist, places=5)
            self.assertAlmostEqual(float(res["presence"]), math.log(2), places=5)
            self.assertAlmostEqual(float(res["count"]), math.log(4), places=5)
            self.assertAlmostEqual(float(res["zone"]), math.log(2), places=5)
            total = math.log(2) + math.log(4) + math.log(2) + 0.5 * sum(nll) / 2 + exist
            self.assertAlmostEqual(float(res["total"]), total, places=5)
            self.assertEqual(float(res["n_matched"]), 2.0)

    def test_people_order_does_not_matter(self):
        out = outputs_of([(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)], sigma=(0.5, 1.0, 2.0))
        a = mdl.compute_loss(out, batch_of([(5.0, 6.0), (1.0, 2.0), (8.0, 8.0)]))
        b = mdl.compute_loss(out, batch_of([(8.0, 8.0), (5.0, 6.0), (1.0, 2.0)]))
        for k in a:
            self.assertAlmostEqual(float(a[k]), float(b[k]), places=5)

    def test_gaussian_nll(self):
        err = torch.tensor([0.0, 0.3, 2.0])
        sig = torch.tensor([0.2, 1.0, 3.0])
        dens = torch.exp(-err ** 2 / (2 * sig ** 2)) / (2 * math.pi * sig ** 2)
        torch.testing.assert_close(mdl.gaussian_nll_2d(err ** 2, sig), -torch.log(dens))

    def test_unknown_geometry_skips_xy_and_zone(self):
        out = outputs_of([(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)])
        for k in ("slot_xy", "slot_sigma", "slot_logit", "zone_logits"):
            out[k].requires_grad_(True)
        res = mdl.compute_loss(out, batch_of([(5.0, 6.0)], known=False))
        for k in ("zone", "xy_nll", "exist"):
            self.assertEqual(float(res[k]), 0.0, k)
        self.assertAlmostEqual(float(res["total"]), math.log(2) + math.log(4), places=5)
        res["total"].backward()
        for k in ("slot_xy", "slot_sigma", "slot_logit", "zone_logits"):
            self.assertTrue(out[k].grad is None or float(out[k].grad.abs().sum()) == 0.0, k)

    def test_padding_never_enters_the_loss(self):
        out = outputs_of([(1.0, 1.0), (5.0, 5.0), (9.0, 9.0)])
        out["zone_logits"] = torch.randn(1, 9)
        b1 = batch_of([(5.0, 6.0)])
        b1["zone_reachable"][0, [2, 6]] = False
        b2 = {k: v.clone() for k, v in b1.items()}
        b2["xy"][0, 1:] = torch.tensor([[1.0, 1.0], [9.0, 9.0]])   # absent people's coordinates
        b2["zone_occupied"][0, [2, 6]] = 1.0                        # unreachable cells' targets
        a, b = mdl.compute_loss(out, b1), mdl.compute_loss(out, b2)
        for k in a:
            self.assertAlmostEqual(float(a[k]), float(b[k]), places=6, msg=k)


class TestPadding(unittest.TestCase):
    def setUp(self):
        self.model = small_model(d=32, seed=1)
        self.batch = synthetic_batch(6, seed=5)

    def assertSameOutputs(self, a, b, n=None):
        for k in a:
            x, y = a[k], b[k]
            if n is not None:
                y = y[:n]
            torch.testing.assert_close(x, y, atol=2e-5, rtol=2e-5, msg=k)

    def test_padded_links_change_nothing(self):
        base = outputs(self.model, self.batch)
        b = dict(self.batch)
        extra = 7
        g = torch.Generator().manual_seed(9)
        for key in mdl.PER_LINK_KEYS:
            if key not in b:
                continue
            v = b[key]
            shape = (v.shape[0], extra) + tuple(v.shape[2:])
            if v.dtype == torch.bool:
                junk = torch.rand(shape, generator=g) < 0.5
            else:
                junk = (torch.randn(shape, generator=g) * 50).to(v.dtype)  # garbage: masks must hide it
            if key == "link_mask":
                junk = torch.zeros(shape, dtype=torch.bool)
            b[key] = torch.cat([v, junk], dim=1)
        self.assertSameOutputs(base, outputs(self.model, b))

    def test_padded_bins_change_nothing(self):
        base = outputs(self.model, self.batch)
        b = dict(self.batch)
        for key in ("amp", "sample_mask", "rssi", "rssi_mask"):
            v = b[key]
            pad = torch.zeros(tuple(v.shape[:-1]) + (3,), dtype=v.dtype)
            b[key] = torch.cat([v, pad], dim=-1)
        self.assertSameOutputs(base, outputs(self.model, b))

    def test_pad_batch_changes_nothing(self):
        base = outputs(self.model, self.batch)
        padded = tr.pad_batch(self.batch, n_links=40, n_batch=11)
        out = outputs(self.model, padded)
        self.assertEqual(out["presence_logit"].shape[0], 11)
        self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))
        self.assertSameOutputs(base, out, n=6)

    def test_link_order_does_not_matter(self):
        base = outputs(self.model, self.batch)
        perm = torch.randperm(self.batch["amp"].shape[1], generator=torch.Generator().manual_seed(2))
        b = {k: (v[:, perm] if k in mdl.PER_LINK_KEYS else v) for k, v in self.batch.items()}
        self.assertSameOutputs(base, outputs(self.model, b))

    def test_batch_neighbours_do_not_matter(self):
        base = outputs(self.model, self.batch)
        one = {k: (v[2:3] if isinstance(v, torch.Tensor) else v[2:3]) for k, v in self.batch.items()}
        single = outputs(self.model, one)
        n_links = int(self.batch["link_mask"][2].sum())
        self.assertGreater(n_links, 0)
        for k in base:
            torch.testing.assert_close(base[k][2:3], single[k], atol=2e-5, rtol=2e-5, msg=k)

    def test_a_sample_without_links_is_finite(self):
        b = dict(self.batch)
        b["link_mask"] = b["link_mask"].clone()
        b["link_mask"][0] = False
        out = outputs(self.model, b)
        self.assertTrue(all(torch.isfinite(v).all() for v in out.values()))


class TestPacking(unittest.TestCase):
    """The packed link encoder (model.pack_index) is the same function."""

    def test_pack_index(self):
        lm = torch.tensor([[True, False, True], [False, False, False], [True, True, False]])
        idx = mdl.pack_index(lm, n_links=5, bucket=3)
        self.assertEqual(idx.tolist(), [0, 2, 10, 11, 15, 15])          # b * 5 + l, padded with B * L = 15
        self.assertEqual(mdl.pack_index(lm, n_links=5, bucket=4).tolist(), [0, 2, 10, 11])
        self.assertEqual(mdl.pack_index(lm, bucket=256).numel(), 256)
        with self.assertRaises(ValueError):
            mdl.pack_index(lm, n_links=2)

    def test_packed_equals_padded(self):
        m = small_model(d=32, seed=3)
        b = synthetic_batch(9, seed=11, train=True, n_scenes=9)
        for n_links, n_batch in ((None, None), (45, None), (45, 13)):
            fb = tr.pad_batch(b, n_links=n_links, n_batch=n_batch)
            base = outputs(m, fb)
            fp = dict(fb)
            fp["pack_index"] = mdl.pack_index(b["link_mask"], fb["amp"].shape[1], bucket=64)
            out = outputs(m, fp)
            for k in base:
                torch.testing.assert_close(out[k], base[k], atol=1e-5, rtol=1e-5, msg=k)

    def test_packed_gradients_equal_padded(self):
        b = synthetic_batch(8, seed=12, train=True, n_scenes=8)
        fb = tr.pad_batch(b, n_links=40)
        grads = []
        for pack in (False, True):
            m = mdl.build_model(mdl.ModelConfig(d=32, dropout=0.0), seed=0)
            x = dict(fb)
            if pack:
                x["pack_index"] = mdl.pack_index(b["link_mask"], 40, bucket=32)
            mdl.compute_loss(m(x), b)["total"].backward()
            grads.append({n: p.grad.clone() for n, p in m.named_parameters()})
        for n in grads[0]:
            torch.testing.assert_close(grads[1][n], grads[0][n], atol=1e-5, rtol=1e-4, msg=n)


class TestUnknownGeometry(unittest.TestCase):
    def test_geometry_is_not_read_when_unknown(self):
        m = small_model(d=32, seed=2)
        b = synthetic_batch(5, seed=6)
        b["geometry_known"] = torch.zeros(5, dtype=torch.bool)
        base = outputs(m, b)
        b2 = dict(b)
        b2["link_pos"] = torch.randn_like(b["link_pos"]) * 5
        b2["room"] = torch.rand_like(b["room"]) * 10 + 1
        out = outputs(m, b2)
        for k in base:
            torch.testing.assert_close(base[k], out[k], msg=k)
        b3 = dict(b2)
        b3["room"] = torch.zeros_like(b["room"])                  # an unknown room may be 0
        out3 = outputs(m, b3)
        self.assertTrue(all(torch.isfinite(v).all() for v in out3.values()))

    def test_known_geometry_is_read(self):
        m = small_model(d=32, seed=2)
        b = synthetic_batch(5, seed=6)
        b["geometry_known"] = torch.ones(5, dtype=torch.bool)
        base = outputs(m, b)
        b2 = dict(b)
        b2["link_pos"] = b["link_pos"] * 0.5
        self.assertFalse(torch.allclose(base["presence_logit"], outputs(m, b2)["presence_logit"]))


class TestPredictions(unittest.TestCase):
    def test_contract(self):
        m = small_model(seed=3)
        b = synthetic_batch(7, seed=1)
        p = mdl.predictions(outputs(m, b))
        shapes = {"presence_prob": (7,), "count_prob": (7, 4), "zone_prob": (7, 9), "slot_exist": (7, 3),
                  "slot_xy": (7, 3, 2), "slot_sigma": (7, 3)}
        for k, s in shapes.items():
            self.assertEqual(p[k].shape, s, k)
            self.assertEqual(p[k].dtype, np.float32, k)
            self.assertTrue(np.isfinite(p[k]).all(), k)
        np.testing.assert_allclose(p["count_prob"].sum(-1), 1.0, atol=1e-6)
        room = b["room"].numpy()
        self.assertTrue(np.all((p["slot_xy"] >= 0) & (p["slot_xy"] <= room[:, None, :2])))
        self.assertTrue(np.all(p["slot_sigma"] >= mdl.SIGMA_MIN_M))


@unittest.skipUnless(torch.cuda.is_available(), "needs a GPU")
class TestCompiledFloat16(unittest.TestCase):
    """torch.compile + float16 autocast gives eager's gradients (a regression:
    an amax-based max pooling came out NaN in the compiled backward)."""

    def test_compiled_gradients_match_eager(self):
        cpu = synthetic_batch(16, seed=8, n_nodes=(4, 6), n_scenes=16, train=True)
        b = tr.to_device(cpu, "cuda")
        grads = []
        for compiled in (False, True):
            # links padded and packed as in training (pack_index's length marked dynamic when compiled)
            fb = tr.forward_batch(b, cpu["link_mask"], n_links=40, n_batch=None, pack=True, bucket=64,
                                  compiled=compiled)
            m = mdl.build_model(mdl.ModelConfig(d=32, dropout=0.0), seed=0).cuda()
            fwd = torch.compile(m) if compiled else m
            with torch.autocast("cuda", dtype=torch.float16):
                out = fwd(fb)
            res = mdl.compute_loss(out, b)
            (res["total"] * 1024.0).backward()
            grads.append({n: p.grad.float().clone() for n, p in m.named_parameters()})
        for n, g in grads[1].items():
            self.assertTrue(torch.isfinite(g).all(), n)
            e = grads[0][n]
            cos = torch.nn.functional.cosine_similarity(g.flatten(), e.flatten(), dim=0)
            if e.norm() > 0:
                self.assertGreater(float(cos), 0.99, n)


class TestOverfit(unittest.TestCase):
    def test_overfits_a_tiny_batch(self):
        torch.manual_seed(0)
        m = mdl.build_model(mdl.ModelConfig(d=48, dropout=0.0), seed=0)
        m.train()
        b = synthetic_batch(8, seed=21, n_nodes=(4, 5), n_scenes=8)
        self.assertGreaterEqual(len(set(b["count"].tolist())), 3)
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        first = None
        for _ in range(200):
            opt.zero_grad()
            res = mdl.compute_loss(m(b), b)
            res["total"].backward()
            opt.step()
            if first is None:
                first = {k: float(v) for k, v in res.items()}
        m.eval()
        with torch.no_grad():
            out = m(b)
            last = {k: float(v) for k, v in mdl.compute_loss(out, b).items()}
        self.assertLess(last["presence"], 0.05 * first["presence"], (first, last))
        self.assertLess(last["count"], 0.05 * first["count"], (first, last))
        self.assertLess(last["zone"], 0.1 * first["zone"], (first, last))
        self.assertLess(last["exist"], 0.1 * first["exist"], (first, last))
        self.assertLess(last["xy_nll"], first["xy_nll"] - 5.0, (first, last))  # nats: errors far below sigma
        self.assertTrue(torch.equal(out["count_logits"].argmax(-1), b["count"]))
        err = (out["slot_xy"][:, None] - b["xy"][:, :, None]).norm(dim=-1).min(-1).values[b["person_mask"]]
        self.assertLess(float(err.max()), 0.3)


if __name__ == "__main__":
    unittest.main()
