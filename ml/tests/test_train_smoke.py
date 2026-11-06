"""Training, evaluation and sweep smoke tests (spec/model.md §7 "a smoke
run"): a CPU run of one epoch on synthetic data end to end (config, log,
checkpoints), resume after an interruption equal to an uninterrupted run,
the evaluation's predictions contract and SIM-labelled JSON, the sweep's
plan and resume-safety, and, when the data side's real cache exists in
/dev/shm and a GPU is present, a GPU run of a few hundred compiled steps
on it.

Everything is SIM (synthetic windows here are SIM-like and labelled SIM).

Run from the repo root: .venv/bin/python -m unittest discover -s ml/tests
"""

import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.train import evaluate as ev  # noqa: E402
from ml.train import model as mdl  # noqa: E402
from ml.train import sweep as sw  # noqa: E402
from ml.train import train as tr  # noqa: E402

REAL_CACHE = "/dev/shm/saans-phase8-cache/roomsim-v1/2"
SYN = {"n_train": 96, "n_val": 32, "n_test": 32, "n_nodes": [4, 5]}


def cpu_cfg(out_root, **kw):
    cfg = {"synthetic": dict(SYN), "device": "cpu", "epochs": 1, "batch": 16, "eval_batch": 16, "d": 32,
           "warmup_steps": 3, "out_root": out_root, "compile": False, "workers": 0}
    cfg.update(kw)
    return cfg


class TestCpuSmoke(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="p8smoke-")
        cls.done = tr.train(cpu_cfg(cls.tmp, run_id="smoke"))
        cls.run_dir = os.path.join(cls.tmp, "smoke")

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_files(self):
        for name in ("config.json", "log.jsonl", "best.pt", "last.pt", "done.json"):
            self.assertTrue(os.path.isfile(os.path.join(self.run_dir, name)), name)
        self.assertEqual(self.done["label"], "SIM")
        self.assertEqual(self.done["steps"], 6)                     # 96 samples / 16, one epoch
        self.assertEqual(self.done["stop"], "epochs")

    def test_config_lists_everything(self):
        with open(os.path.join(self.run_dir, "config.json")) as f:
            cfg = json.load(f)
        self.assertEqual(cfg["label"], "SIM")
        self.assertEqual(cfg["loss_weights"], {"presence": 1.0, "count": 1.0, "zone": 1.0, "xy_nll": 0.5,
                                               "exist": 1.0})
        self.assertEqual(cfg["subset_rules"], {"p_one_tx": 0.2, "p_one_link": 0.1, "p_drop_link": 0.1,
                                               "p_unknown_geometry": 0.1})
        for key in ("lr", "d", "seed", "window_s", "weight_decay", "warmup_steps", "grad_clip", "patience",
                    "batch", "epochs", "n_params", "model_config", "total_steps", "data", "environment"):
            self.assertIn(key, cfg)
        self.assertEqual(cfg["n_params"], mdl.count_parameters(mdl.build_model(mdl.ModelConfig(d=32))))

    def test_log(self):
        with open(os.path.join(self.run_dir, "log.jsonl")) as f:
            lines = [json.loads(ln) for ln in f if ln.strip()]
        self.assertEqual(len(lines), 1)
        rec = lines[0]
        self.assertEqual(rec["label"], "SIM")
        for key in ("epoch", "step", "lr", "train", "val", "val_metrics", "time", "throughput", "best_val"):
            self.assertIn(key, rec)
        for term in ("total",) + tr.TERMS:
            self.assertTrue(np.isfinite(rec["train"][term]), term)
        self.assertEqual(rec["val"]["n_windows"], 32)
        self.assertEqual(rec["val_metrics"]["source"], "ml.train.metrics")
        self.assertIn("presence_acc", rec["val_metrics"])

    def test_best_checkpoint_reloads(self):
        model, ck = ev.load_checkpoint(os.path.join(self.run_dir, "best.pt"))
        self.assertEqual(ck["label"], "SIM")
        self.assertAlmostEqual(ck["val_loss"], self.done["best_val_loss"], places=6)
        src = tr.make_source(tr.resolve_config(cpu_cfg(self.tmp)))
        res = tr.run_eval(model, src.eval_loader("val"), torch.device("cpu"), amp=False,
                          weights=mdl.DEFAULT_LOSS_WEIGHTS, matcher="enumerate")
        self.assertAlmostEqual(res["loss"], ck["val_loss"], places=5)

    def test_done_run_is_skipped(self):
        again = tr.train(cpu_cfg(self.tmp, run_id="smoke"))
        self.assertEqual(again, self.done)

    def test_evaluate_contract(self):
        doc = ev.evaluate_runs([self.run_dir], "test", device="cpu", workers=0, batch=16, n_boot=20)
        self.assertEqual(doc["label"], "SIM")
        self.assertEqual(doc["metrics"]["label"], "SIM")
        self.assertTrue(os.path.isfile(doc["json"]))
        z = np.load(doc["runs"][0]["predictions"])
        n = 32
        shapes = {"scene": (n,), "k": (n,), "presence_prob": (n,), "count_prob": (n, 4), "zone_prob": (n, 9),
                  "slot_exist": (n, 3), "slot_xy": (n, 3, 2), "slot_sigma": (n, 3), "slot_radius": (n, 3)}
        for key, shape in shapes.items():
            self.assertEqual(z[key].shape, shape, key)
        self.assertEqual(str(z["label"]), "SIM")
        self.assertTrue(np.all(z["slot_radius"] >= np.float32(0.3)))
        self.assertTrue(np.all(z["slot_sigma"] >= np.float32(mdl.SIGMA_MIN_M)))
        np.testing.assert_allclose(z["count_prob"].sum(1), 1.0, atol=1e-5)
        figs = doc["metrics"]["methods"]["model"]["figures"]
        self.assertIn("presence", figs)
        self.assertIn("ci95", figs["presence"]["all"]["accuracy"])


class TestResume(unittest.TestCase):
    def test_interrupted_and_resumed_equals_straight(self):
        tmp = tempfile.mkdtemp(prefix="p8resume-")
        try:
            straight = tr.train(cpu_cfg(tmp, run_id="a", epochs=2, dropout=0.1))
            part = tr.train(cpu_cfg(tmp, run_id="b", epochs=2, dropout=0.1, interrupt_after_epochs=1))
            self.assertEqual(part.get("interrupted_after_epoch"), 0)
            self.assertFalse(os.path.exists(os.path.join(tmp, "b", "done.json")))
            resumed = tr.train(cpu_cfg(tmp, run_id="b", epochs=2, dropout=0.1))
            self.assertEqual(resumed["steps"], straight["steps"])
            self.assertEqual(resumed["best_val_loss"], straight["best_val_loss"])
            a = torch.load(os.path.join(tmp, "a", "last.pt"), weights_only=False)["model"]
            b = torch.load(os.path.join(tmp, "b", "last.pt"), weights_only=False)["model"]
            for k in a:
                self.assertTrue(torch.equal(a[k], b[k]), k)
            with open(os.path.join(tmp, "b", "log.jsonl")) as f:
                epochs = [json.loads(ln)["epoch"] for ln in f if ln.strip()]
            self.assertEqual(epochs, [0, 1])
            tr.train(cpu_cfg(tmp, run_id="c", epochs=2, interrupt_after_epochs=1))
            with self.assertRaises(SystemExit):                     # another config over a partial run
                tr.train(cpu_cfg(tmp, run_id="c", epochs=2, lr=3e-3))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_a_run_in_progress_is_not_trained_twice(self):
        import fcntl

        tmp = tempfile.mkdtemp(prefix="p8lock-")
        try:
            os.makedirs(os.path.join(tmp, "x"))
            with open(os.path.join(tmp, "x", ".lock"), "w") as held:
                fcntl.flock(held, fcntl.LOCK_EX | fcntl.LOCK_NB)  # another trainer holds the run
                with self.assertRaises(SystemExit):
                    tr.train(cpu_cfg(tmp, run_id="x"))
            self.assertEqual(tr.train(cpu_cfg(tmp, run_id="x"))["label"], "SIM")   # released: it runs
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_early_stopping(self):
        tmp = tempfile.mkdtemp(prefix="p8early-")
        try:
            done = tr.train(cpu_cfg(tmp, run_id="e", epochs=6, lr=0.0, patience=2, warmup_steps=1))
            self.assertEqual(done["stop"], "early_stopping")       # lr 0: the val loss never improves
            self.assertEqual(done["best_epoch"], 0)
            with open(os.path.join(tmp, "e", "log.jsonl")) as f:
                self.assertEqual(len([ln for ln in f if ln.strip()]), 3)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


class TestConfig(unittest.TestCase):
    def test_cli_overrides_json(self):
        tmp = tempfile.mkdtemp(prefix="p8cfg-")
        try:
            path = os.path.join(tmp, "c.json")
            with open(path, "w") as f:
                json.dump({"lr": 3e-3, "d": 64, "loss_weights": {"xy_nll": 0.5}}, f)
            cfg = tr.resolve_config(tr.parse_args(["--config", path, "--d", "128", "--window-s", "4"]))
            self.assertEqual((cfg["lr"], cfg["d"], cfg["window_s"]), (3e-3, 128, 4.0))
            self.assertEqual(cfg["run_id"], "on_l4_d128_lr0.003_s0")
            with self.assertRaises(ValueError):
                tr.resolve_config({"no_such_key": 1})
            with self.assertRaises(ValueError):
                tr.resolve_config({"window_s": 0.3})
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_schedule(self):
        f = tr.lr_lambda(500, 10_000)
        self.assertAlmostEqual(f(0), 1 / 500)
        self.assertAlmostEqual(f(499), 1.0)
        self.assertAlmostEqual(f(500), 1.0)
        self.assertAlmostEqual(f(5250), 0.5)
        self.assertAlmostEqual(f(10_000), 0.0)


class TestSweepPlan(unittest.TestCase):
    def test_grid(self):
        runs = sw.sweep_plan()
        self.assertEqual(len(runs), 27)
        self.assertEqual(len({r["run_id"] for r in runs}), 27)
        self.assertEqual({(r["lr"], r["d"], r["window_s"]) for r in runs},
                         {(lr, d, l) for lr in (3e-4, 1e-3, 3e-3) for d in (64, 128, 256) for l in (1.0, 2.0, 4.0)})
        self.assertTrue(all(r["seed"] == 0 for r in runs))

    def test_finished_runs_are_skipped(self):
        tmp = tempfile.mkdtemp(prefix="p8sweep-")
        try:
            runs = [dict(cpu_cfg(tmp), run_id=f"r{i}", lr=1e-3 * (i + 1)) for i in range(2)]
            os.makedirs(os.path.join(tmp, "r0"))
            with open(os.path.join(tmp, "r0", "done.json"), "w") as f:     # done: best epoch, best.pt, same config
                json.dump({"label": "SIM", "best_epoch": 0, "best_val_loss": 1.0}, f)
            with open(os.path.join(tmp, "r0", "config.json"), "w") as f:
                json.dump(tr.resolve_config(runs[0]), f)
            with open(os.path.join(tmp, "r0", "best.pt"), "wb") as f:
                f.write(b"weights")
            todo = sw.pending(runs)
            self.assertEqual([r["run_id"] for r in todo], ["r1"])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_launch_and_collect(self):
        tmp = tempfile.mkdtemp(prefix="p8sweep-")
        try:
            runs = [dict(cpu_cfg(tmp), run_id=f"r{i}", lr=lr) for i, lr in enumerate((1e-3, 3e-3))]
            res = sw.launch(runs, concurrency=2, log_dir=os.path.join(tmp, "logs"), env={"CUDA_VISIBLE_DEVICES": ""})
            self.assertEqual(sorted(res["finished"]), ["r0", "r1"])
            table = sw.collect(runs)
            self.assertEqual(len(table["rows"]), 2)
            self.assertEqual(table["label"], "SIM")
            best = table["rows"][0]
            self.assertLessEqual(best["best_val_loss"], table["rows"][1]["best_val_loss"])
            again = sw.launch(runs, concurrency=2, log_dir=os.path.join(tmp, "logs"))
            self.assertEqual(again["launched"], [])
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(os.path.isfile(os.path.join(REAL_CACHE, "meta.json")) and torch.cuda.is_available(),
                     f"needs the real cache at {REAL_CACHE} and a GPU")
class TestGpuSmokeRealCache(unittest.TestCase):
    """A few hundred compiled fp16 steps on the real SIM cache (ℓ = 2 s)."""

    def test_few_hundred_steps(self):
        tmp = tempfile.mkdtemp(prefix="p8gpu-")
        try:
            cfg = {"cache_dir": REAL_CACHE, "window_s": 2.0, "d": 128, "lr": 1e-3, "epochs": 1, "max_steps": 300,
                   "warmup_steps": 50, "val_max_batches": 8, "out_root": tmp, "run_id": "gpu", "workers": 4,
                   "compile": True}
            done = tr.train(cfg)
            self.assertEqual(done["steps"], 300)
            self.assertTrue(done["compiled"], done.get("compile_error"))
            with open(os.path.join(tmp, "gpu", "log.jsonl")) as f:
                rec = json.loads(f.readline())
            self.assertTrue(np.isfinite(rec["val"]["loss"]))
            self.assertLessEqual(rec["skipped_steps"], 5)         # GradScaler's first overflowing steps only
            # a model that learned something: better than the constant prior on presence and count
            self.assertLess(rec["val"]["presence"], 0.69)
            self.assertLess(rec["val"]["count"], 1.38)
            self.assertGreater(rec["throughput"]["samples_per_s"], 1000)
            doc = ev.evaluate_runs([os.path.join(tmp, "gpu")], "val", "tx0", workers=4, n_boot=0)
            self.assertEqual(doc["label"], "SIM")
            z = np.load(doc["runs"][0]["predictions"])
            self.assertEqual(z["presence_prob"].shape, (doc["metrics"]["n_windows"],))
            self.assertEqual(str(z["subset"]), "tx0")
            # every link, beside the dataset's stored WLM and RTI and the prior, paired against the model
            doc = ev.evaluate_runs([os.path.join(tmp, "gpu")], "val", "all", workers=4, n_boot=0, baselines=True)
            methods = doc["metrics"]["methods"]
            for name in ("model", "wlm_stored", "rti_stored", "prior"):
                self.assertIn(name, methods)
            self.assertTrue(methods["wlm_stored"]["uses_calibration"])
            self.assertIn("model - wlm_stored", doc["metrics"]["differences"])
            self.assertEqual(doc["metrics"]["n_windows"], 201 * 117)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
