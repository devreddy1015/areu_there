"""Run integrity (ml.train.integrity): the rules that keep a frozen, stale,
partial or mislabelled run from passing as a finished Phase 8 result, and a
checkpoint that is not P9-SIM from reaching Phase 9.

The rules themselves need only the standard library, so the first classes
run anywhere (`python3 -m unittest ml.tests.test_integrity`). The classes
marked "needs torch" drive the real trainer, sweep and evaluation on
synthetic CPU runs and are skipped without torch; run them on the GPU box:
.venv/bin/python -m unittest discover -s ml/tests

Everything here is SIM.
"""

import json
import math
import os
import shutil
import sys
import tempfile
import unittest
from unittest import mock

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from ml.train import integrity as integ  # noqa: E402

try:
    import torch
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

CFG = {"run_id": "on_l2_d128_lr0.001_s0", "dataset": "roomsim-v1", "window_s": 2.0, "d": 128, "lr": 1e-3,
       "seed": 0, "epochs": 30, "batch": 128, "patience": 5, "dropout": 0.1,
       "loss_weights": {"presence": 1.0, "count": 1.0, "zone": 1.0, "xy_nll": 0.5, "exist": 1.0},
       "workers": 6, "device": "cuda", "out_root": "/runs/sweep", "synthetic": None, "phase": False, "gain": False}


def write_json(path, obj):
    with open(path, "w") as f:
        json.dump(obj, f)


def make_run(root, cfg, *, done=None, best=True, config=True, run_id=None):
    """A run directory as ml.train.train leaves it (`best.pt` is a placeholder: only its presence is read here).
    `done=False` leaves out done.json."""
    run_dir = os.path.join(root, run_id or cfg["run_id"])
    os.makedirs(run_dir, exist_ok=True)
    if config:
        write_json(os.path.join(run_dir, "config.json"), {"label": "SIM", **cfg, "n_params": 1, "data": {}})
    if best:
        with open(os.path.join(run_dir, "best.pt"), "wb") as f:
            f.write(b"weights")
    if done is not False:
        write_json(os.path.join(run_dir, "done.json"),
                   done or {"label": "SIM", "run_id": cfg["run_id"], "best_epoch": 12, "best_val_loss": 1.25,
                            "steps": 9000, "stop": "early_stopping"})
    return run_dir


class TempDir(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="p8integrity-")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestSameRun(unittest.TestCase):
    """Finding 2: the run id encodes only part of the config; the rest must be compared."""

    def test_settings_that_do_not_change_the_result_are_ignored(self):
        other = dict(CFG, workers=2, device="cpu", out_root="/elsewhere", compile=False, eval_batch=64)
        self.assertTrue(integ.same_run(CFG, other))
        self.assertEqual(integ.config_diff(CFG, other), [])

    def test_every_setting_missing_from_the_run_id_is_compared(self):
        for key, value in (("epochs", 60), ("batch", 64), ("patience", 10), ("dropout", 0.2),
                           ("loss_weights", dict(CFG["loss_weights"], xy_nll=1.0))):
            with self.subTest(key=key):
                self.assertEqual(integ.config_diff(CFG, dict(CFG, **{key: value})), [key])
                self.assertFalse(integ.same_run(CFG, dict(CFG, **{key: value})))

    def test_a_missing_setting_differs(self):
        partial = {k: v for k, v in CFG.items() if k != "epochs"}
        self.assertEqual(integ.config_diff(partial, CFG), ["epochs"])

    def test_json_round_trip_is_the_same_config(self):
        a = dict(CFG, synthetic={"n_nodes": (4, 6)})
        b = json.loads(json.dumps(a))                        # config.json turns the tuple into a list
        self.assertTrue(integ.same_run(a, b))


class TestDoneProblem(TempDir):
    """Findings 1 and 2: only a finished run with a usable best.pt, of the requested config, is done."""

    def test_a_finished_run_of_the_same_config_is_done(self):
        run_dir = make_run(self.tmp, CFG)
        self.assertIsNone(integ.done_problem(run_dir, CFG))
        self.assertIsNone(integ.done_problem(run_dir, dict(CFG, workers=1)))
        self.assertIsNone(integ.done_problem(run_dir))

    def test_no_done_json_is_unfinished(self):
        run_dir = make_run(self.tmp, CFG, done=False)
        self.assertIn("done.json", integ.done_problem(run_dir, CFG))

    def test_a_frozen_run_is_not_done(self):
        # what the trainer used to write when every step was skipped or the val loss was NaN
        for i, (best_epoch, best_val) in enumerate(((-1, math.inf), (-1, None), (3, math.nan), (None, 1.0))):
            with self.subTest(best_epoch=best_epoch, best_val=best_val):
                run_dir = make_run(self.tmp, CFG, run_id=f"r{i}",
                                   done={"label": "SIM", "best_epoch": best_epoch, "best_val_loss": best_val})
                self.assertIn("finite validation loss", integ.done_problem(run_dir, CFG))

    def test_a_done_json_without_best_pt_is_not_done(self):
        run_dir = make_run(self.tmp, CFG, best=False)
        self.assertIn("best.pt", integ.done_problem(run_dir, CFG))

    def test_a_done_run_of_another_config_is_not_done(self):
        run_dir = make_run(self.tmp, CFG)
        problem = integ.done_problem(run_dir, dict(CFG, epochs=60, patience=10))
        self.assertIn("epochs", problem)
        self.assertIn("patience", problem)

    def test_a_config_cannot_be_checked_without_config_json(self):
        run_dir = make_run(self.tmp, CFG, config=False)
        self.assertIn("config.json", integ.done_problem(run_dir, CFG))

    def test_an_unreadable_done_json_is_not_done(self):
        run_dir = make_run(self.tmp, CFG)
        with open(os.path.join(run_dir, "done.json"), "w") as f:
            f.write("{truncated")
        self.assertIn("unreadable", integ.done_problem(run_dir, CFG))


class TestSkipProblem(unittest.TestCase):
    """Finding 1: a run whose optimiser steps are mostly skipped has stopped learning."""

    def test_the_scalers_start_up_skips_pass(self):
        self.assertIsNone(integ.skip_problem(5, integ.SKIP_WINDOW))
        self.assertIsNone(integ.skip_problem(0, integ.SKIP_WINDOW))
        self.assertIsNone(integ.skip_problem(0, 0))

    def test_a_frozen_window_fails(self):
        self.assertIn("50 of 50", integ.skip_problem(50, 50))

    def test_the_threshold(self):
        limit = int(integ.MAX_SKIP_FRACTION * 50)
        self.assertIsNone(integ.skip_problem(limit, 50))
        self.assertIsNotNone(integ.skip_problem(limit + 1, 50))


class TestBestCheckpoint(unittest.TestCase):
    """Findings 1 and 6: done.json only beside the best epoch's best.pt; a kill between the two saves is repaired."""

    def test_no_finite_validation_loss(self):
        self.assertIn("finite", integ.best_problem(-1, math.inf, None))
        self.assertIn("finite", integ.best_problem(2, math.nan, 2))

    def test_best_pt_missing_or_from_another_epoch(self):
        self.assertIn("missing", integ.best_problem(3, 0.5, None))
        self.assertIn("epoch 2", integ.best_problem(3, 0.5, 2))
        self.assertIsNone(integ.best_problem(3, 0.5, 3))

    def test_restore_only_when_last_pt_holds_the_best_model(self):
        self.assertTrue(integ.restore_best(4, 4, 3))         # killed after last.pt, before best.pt
        self.assertTrue(integ.restore_best(4, 4, None))
        self.assertFalse(integ.restore_best(4, 4, 4))        # consistent
        self.assertFalse(integ.restore_best(4, 2, 2))        # the best epoch is older: last.pt is not it
        self.assertFalse(integ.restore_best(4, -1, None))    # nothing to restore


class TestFinalSelection(unittest.TestCase):
    """Finding 3: the final configuration comes from the whole grid, and never from a non-finite row."""

    def rows(self):
        return [{"run_id": "a", "best_val_loss": 1.2}, {"run_id": "b", "best_val_loss": None},
                {"run_id": "c", "best_val_loss": 0.9}, {"run_id": "d", "best_val_loss": math.inf},
                {"run_id": "e", "best_val_loss": math.nan}]

    def test_rank_drops_non_finite_rows(self):
        ranked, dropped = integ.rank_rows(self.rows())
        self.assertEqual([r["run_id"] for r in ranked], ["c", "a"])
        self.assertEqual(sorted(dropped), ["b", "d", "e"])

    def test_an_incomplete_sweep_is_refused(self):
        table = {"n_runs": 27, "n_finished": 2, "best": "c", "not_done": {"x": "unfinished (no done.json)"}}
        with self.assertRaises(SystemExit) as cm:
            integ.pick_final(table)
        self.assertIn("25 of 27", str(cm.exception))
        self.assertIn("--choose", str(cm.exception))

    def test_an_explicit_choice_is_honoured(self):
        table = {"n_runs": 27, "n_finished": 2, "best": "c", "not_done": {"x": "unfinished"}}
        self.assertEqual(integ.pick_final(table, "a"), "a")

    def test_a_complete_sweep_gives_its_best(self):
        self.assertEqual(integ.pick_final({"n_runs": 2, "n_finished": 2, "best": "c", "not_done": {}}), "c")

    def test_nothing_finished(self):
        with self.assertRaises(SystemExit):
            integ.pick_final({"n_runs": 0, "n_finished": 0, "best": None, "not_done": {}})


class TestStaleEval(TempDir):
    """Finding 2: a final evaluation is reused only while its checkpoints are byte for byte the same."""

    def setUp(self):
        super().setUp()
        self.ckpts = []
        for i in range(2):
            path = os.path.join(self.tmp, f"s{i}", "best.pt")
            os.makedirs(os.path.dirname(path))
            with open(path, "wb") as f:
                f.write(f"weights {i}".encode())
            self.ckpts.append(path)
        self.json = os.path.join(self.tmp, "main_on_all.json")
        write_json(self.json, {"label": "SIM", "runs": [
            {"run_id": f"s{i}", "checkpoint": os.path.abspath(p), "checkpoint_sha256": integ.file_sha256(p)}
            for i, p in enumerate(self.ckpts)]})

    def test_unchanged_checkpoints_are_current(self):
        self.assertIsNone(integ.stale_eval(self.json, self.ckpts))

    def test_a_retrained_checkpoint_is_stale(self):
        with open(self.ckpts[1], "wb") as f:
            f.write(b"retrained")
        self.assertIn("changed", integ.stale_eval(self.json, self.ckpts))

    def test_other_runs_are_stale(self):
        self.assertIsNotNone(integ.stale_eval(self.json, self.ckpts[:1]))
        self.assertIsNotNone(integ.stale_eval(self.json, self.ckpts[::-1]))

    def test_an_evaluation_without_hashes_is_stale(self):
        write_json(self.json, {"runs": [{"checkpoint": os.path.abspath(p)} for p in self.ckpts]})
        self.assertIn("hash", integ.stale_eval(self.json, self.ckpts))

    def test_missing_or_unreadable(self):
        self.assertIsNotNone(integ.stale_eval(os.path.join(self.tmp, "none.json"), self.ckpts))
        with open(self.json, "w") as f:
            f.write("{")
        self.assertIsNotNone(integ.stale_eval(self.json, self.ckpts))


class TestSeedRuns(unittest.TestCase):
    """Finding 4: several runs are distinct seeds scored on the same windows."""

    def entry(self, run_id, seed, scene=(0, 0, 1), k=(7, 9, 7)):
        return {"run_id": run_id, "seed": seed, "scene": list(scene), "k": list(k)}

    def test_distinct_seeds_on_the_same_windows(self):
        integ.check_seed_runs([self.entry("s0", 0), self.entry("s1", 1), self.entry("s2", 2)])
        integ.check_seed_runs([self.entry("s0", 0)])

    def test_a_repeated_seed_is_refused(self):
        with self.assertRaises(ValueError) as cm:
            integ.check_seed_runs([self.entry("s0", 0), self.entry("copy", 0), self.entry("s2", 2)])
        self.assertIn("seed", str(cm.exception))

    def test_other_windows_are_refused(self):
        for key, other in (("scene", {"scene": (0, 1, 1)}), ("k", {"k": (7, 9, 9)}), ("len", {"scene": (0, 0)})):
            with self.subTest(key):
                with self.assertRaises(ValueError):
                    integ.check_seed_runs([self.entry("s0", 0), self.entry("s1", 1, **other)])


class TestP9Pretrained(TempDir):
    """Finding 5: Phase 9 starts only from P9-SIM checkpoints (spec/phase9.md §4.1)."""

    def p9(self, **kw):
        return dict(CFG, **{"dataset": "roomsim-v1", "window_s": 2.0, "seed": 1, "phase": False, "gain": False,
                            "synthetic": None, **kw})

    def test_a_p9_sim_checkpoint_passes(self):
        self.assertEqual(integ.p9_pretrained_problems(self.p9(), 2.0, 1, "shared40"), [])
        self.assertEqual(integ.p9_pretrained_problems(self.p9(dataset="/data/roomsim-v1/"), 2, 1, "shared40"), [])

    def test_each_requirement(self):
        cases = {"window_s": (self.p9(window_s=4.0), "shared40"),
                 "dataset": (self.p9(dataset="roomsim-v1-hwoff"), "shared40"),
                 "shared40": (self.p9(), "shared44"), "reference bins": (self.p9(), None),
                 "phase": (self.p9(phase=True), "shared40"), "gain": (self.p9(gain=True), "shared40"),
                 "seed": (self.p9(seed=0), "shared40"), "synthetic": (self.p9(synthetic={}), "shared40")}
        for word, (rc, ref_bins) in cases.items():
            with self.subTest(word):
                problems = integ.p9_pretrained_problems(rc, 2.0, 1, ref_bins)
                self.assertEqual(len(problems), 1, problems)
                self.assertIn(word, problems[0])

    def test_recorded_ref_bins(self):
        run_dir = make_run(self.tmp, CFG, run_id="p9")
        self.assertIsNone(integ.recorded_ref_bins(run_dir))  # an older run records none
        write_json(os.path.join(run_dir, "config.json"), {"label": "SIM", "data": {"ref_bins": "shared40"}})
        self.assertEqual(integ.recorded_ref_bins(run_dir), "shared40")
        self.assertIsNone(integ.recorded_ref_bins(os.path.join(self.tmp, "nowhere")))


# --------------------------------------------------------------------------
# needs torch: the trainer, sweep and evaluation on synthetic CPU runs
# --------------------------------------------------------------------------

SYN = {"n_train": 96, "n_val": 32, "n_test": 32, "n_nodes": [4, 5]}


def cpu_cfg(out_root, **kw):
    cfg = {"synthetic": dict(SYN), "device": "cpu", "epochs": 1, "batch": 16, "eval_batch": 16, "d": 32,
           "warmup_steps": 3, "out_root": out_root, "compile": False, "workers": 0}
    cfg.update(kw)
    return cfg


@unittest.skipUnless(HAVE_TORCH, "needs torch: run on the GPU box (RUN.md §4h)")
class TestTrainerGuards(TempDir):
    """Findings 1, 2 and 6 in ml.train.train."""

    def setUp(self):
        super().setUp()
        from ml.train import model as mdl
        from ml.train import train as tr
        self.tr, self.mdl = tr, mdl

    def files(self, run_id):
        return set(os.listdir(os.path.join(self.tmp, run_id)))

    def test_a_frozen_run_stops_without_done_json(self):
        real = self.mdl.compute_loss

        def nan_loss(*a, **kw):                              # every gradient non-finite: every step skipped
            out = real(*a, **kw)
            return dict(out, total=out["total"] * float("nan"))

        with mock.patch.object(self.mdl, "compute_loss", nan_loss), mock.patch.object(self.tr, "SKIP_WINDOW", 3):
            with self.assertRaises(SystemExit) as cm:
                self.tr.train(cpu_cfg(self.tmp, run_id="frozen"))
        self.assertIn("skipped", str(cm.exception))
        self.assertFalse({"done.json", "best.pt"} & self.files("frozen"))

    def test_a_non_finite_val_loss_stops_without_done_json(self):
        real = self.tr.run_eval

        def nan_eval(*a, **kw):
            return dict(real(*a, **kw), loss=float("nan"))

        with mock.patch.object(self.tr, "run_eval", nan_eval):
            with self.assertRaises(SystemExit) as cm:
                self.tr.train(cpu_cfg(self.tmp, run_id="nanval"))
        self.assertIn("validation loss", str(cm.exception))
        self.assertFalse({"done.json", "best.pt", "last.pt"} & self.files("nanval"))
        with open(os.path.join(self.tmp, "nanval", "log.jsonl")) as f:
            self.assertEqual(len([ln for ln in f if ln.strip()]), 1)   # the epoch is logged for the post-mortem

    def test_a_finished_run_is_reused_only_for_the_same_config(self):
        done = self.tr.train(cpu_cfg(self.tmp, run_id="r"))
        self.assertEqual(self.tr.train(cpu_cfg(self.tmp, run_id="r", eval_batch=8)), done)
        for change in ({"epochs": 2}, {"batch": 8}, {"patience": 1}, {"dropout": 0.2},
                       {"loss_weights": {"xy_nll": 1.0}}):
            with self.subTest(change=change), self.assertRaises(SystemExit) as cm:
                self.tr.train(cpu_cfg(self.tmp, run_id="r", **change))
            self.assertIn(next(iter(change)), str(cm.exception))
        again = self.tr.train(cpu_cfg(self.tmp, run_id="r", epochs=2, overwrite=True))
        self.assertEqual(again["steps"], 2 * done["steps"])

    def test_a_legacy_frozen_done_json_is_refused(self):
        self.tr.train(cpu_cfg(self.tmp, run_id="old"))
        os.remove(os.path.join(self.tmp, "old", "best.pt"))
        write_json(os.path.join(self.tmp, "old", "done.json"), {"label": "SIM", "best_epoch": -1,
                                                               "best_val_loss": math.inf})
        with self.assertRaises(SystemExit):
            self.tr.train(cpu_cfg(self.tmp, run_id="old"))

    def test_a_resumed_run_does_not_train_past_its_early_stop(self):
        cfg = cpu_cfg(self.tmp, run_id="e", epochs=6, lr=0.0, patience=2, warmup_steps=1)
        done = self.tr.train(cfg)
        self.assertEqual(done["stop"], "early_stopping")
        os.remove(os.path.join(self.tmp, "e", "done.json"))   # killed after last.pt, before done.json
        again = self.tr.train(cfg)
        self.assertEqual((again["stop"], again["steps"], again["best_epoch"]), ("early_stopping", done["steps"], 0))
        with open(os.path.join(self.tmp, "e", "log.jsonl")) as f:
            self.assertEqual(len([ln for ln in f if ln.strip()]), 3)

    def test_best_pt_is_restored_from_last_pt(self):
        done = self.tr.train(cpu_cfg(self.tmp, run_id="k"))
        last = torch.load(os.path.join(self.tmp, "k", "last.pt"), weights_only=False)
        for name in ("best.pt", "done.json"):                # killed after last.pt, before best.pt
            os.remove(os.path.join(self.tmp, "k", name))
        again = self.tr.train(cpu_cfg(self.tmp, run_id="k"))
        self.assertEqual((again["best_epoch"], again["best_val_loss"]), (done["best_epoch"], done["best_val_loss"]))
        best = torch.load(os.path.join(self.tmp, "k", "best.pt"), weights_only=False)
        self.assertEqual((best["label"], best["epoch"], best["val_loss"]), ("SIM", 0, done["best_val_loss"]))
        for key, value in last["model"].items():
            self.assertTrue(torch.equal(best["model"][key], value), key)


@unittest.skipUnless(HAVE_TORCH, "needs torch: run on the GPU box (RUN.md §4h)")
class TestSweepAndEvaluateGuards(TempDir):
    """Findings 2, 3 and 4 in ml.train.sweep and ml.train.evaluate."""

    def setUp(self):
        super().setUp()
        from ml.train import evaluate as ev
        from ml.train import sweep as sw
        from ml.train import train as tr
        self.ev, self.sw, self.tr = ev, sw, tr

    def fake_done(self, cfg, best_val=1.0, **kw):
        cfg = self.tr.resolve_config(cfg)
        make_run(self.tmp, cfg, done={"label": "SIM", "run_id": cfg["run_id"], "best_epoch": 0,
                                      "best_val_loss": best_val, "steps": 6, "stop": "epochs"}, **kw)
        return cfg

    def test_is_done_needs_the_same_config_and_a_best_checkpoint(self):
        cfg = self.fake_done(cpu_cfg(self.tmp, run_id="a"))
        self.assertTrue(self.sw.is_done(cfg))
        self.assertFalse(self.sw.is_done(dict(cfg, epochs=60)))
        os.remove(os.path.join(self.tmp, "a", "best.pt"))
        self.assertFalse(self.sw.is_done(cfg))

    def test_final_refuses_a_partial_sweep(self):
        runs = [self.tr.resolve_config(cpu_cfg(self.tmp, run_id=f"r{i}", lr=1e-3 * (i + 1))) for i in range(3)]
        self.fake_done(runs[0], best_val=0.5)
        self.fake_done(runs[1], best_val=math.inf, best=False)              # a frozen run from the old trainer
        table = self.sw.collect(runs)
        self.assertEqual([r["run_id"] for r in table["rows"]], ["r0"])
        self.assertEqual(sorted(table["not_done"]), ["r1", "r2"])
        with self.assertRaises(SystemExit):
            self.sw.chosen_config(table, runs)
        self.assertEqual(self.sw.chosen_config(table, runs, choose="r0")["run_id"], "r0")
        self.fake_done(runs[1], best_val=0.4)
        self.fake_done(runs[2], best_val=0.6)
        self.assertEqual(self.sw.chosen_config(self.sw.collect(runs), runs)["run_id"], "r1")

    def test_evaluation_records_checkpoint_hashes_and_goes_stale(self):
        self.tr.train(cpu_cfg(self.tmp, run_id="s0"))
        doc = self.ev.evaluate_runs([os.path.join(self.tmp, "s0")], "test", device="cpu", workers=0, batch=16,
                                    n_boot=0)
        ckpt = os.path.join(self.tmp, "s0", "best.pt")
        self.assertEqual(doc["runs"][0]["checkpoint_sha256"], integ.file_sha256(ckpt))
        self.assertIsNone(integ.stale_eval(doc["json"], [ckpt]))
        self.tr.train(cpu_cfg(self.tmp, run_id="s0", lr=3e-3, overwrite=True))
        self.assertIn("changed", integ.stale_eval(doc["json"], [ckpt]))

    def test_evaluation_refuses_repeated_seeds_and_other_windows(self):
        for run_id, kw in (("a", {}), ("b", {}), ("c", {"seed": 1, "synthetic": dict(SYN, n_test=16)})):
            self.tr.train(cpu_cfg(self.tmp, run_id=run_id, **kw))
        runs = [os.path.join(self.tmp, r) for r in "abc"]
        kw = {"device": "cpu", "workers": 0, "batch": 16, "n_boot": 0, "out_dir": os.path.join(self.tmp, "ev")}
        with self.assertRaises(ValueError) as cm:                          # the same seed twice
            self.ev.evaluate_runs(runs[:2], "test", **kw)
        self.assertIn("seed", str(cm.exception))
        with self.assertRaises(ValueError) as cm:                          # 32 test windows against 16
            self.ev.evaluate_runs([runs[0], runs[2]], "test", **kw)
        self.assertIn("windows", str(cm.exception))


if __name__ == "__main__":
    unittest.main()
