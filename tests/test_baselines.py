"""The arithmetic in tools/baselines.py (issue #43).

The measurements themselves (gh, cargo, wall clocks) are not unit-testable
and are not tested here. What is: every number the section prints that is
computed rather than read, because a wrong formula there would print a
plausible table with nothing to say it was wrong.
"""

import importlib.util
import os
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_spec = importlib.util.spec_from_file_location(
    "baselines", os.path.join(ROOT, "tools", "baselines.py"))
baselines = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baselines)


def _job(name, start, end, steps=()):
    return {
        "name": name,
        "conclusion": "success",
        "startedAt": start,
        "completedAt": end,
        "steps": [
            {"name": n, "conclusion": "success", "startedAt": s, "completedAt": e}
            for n, s, e in steps
        ],
    }


class Median(unittest.TestCase):
    def test_odd_and_even(self):
        self.assertEqual(baselines.median([3, 1, 2]), 2)
        self.assertEqual(baselines.median([4, 1, 3, 2]), 2.5)

    def test_empty_refuses(self):
        with self.assertRaises(ValueError):
            baselines.median([])


class MarginalCpu(unittest.TestCase):
    def test_startup_cancels(self):
        # 0.2 s of startup in both runs, 1 ms per link-second on top.
        links, short_s, long_s = 3, 600.0, 2400.0
        short = 0.2 + 1e-3 * links * short_s
        long_ = 0.2 + 1e-3 * links * long_s
        got = baselines.cpu_per_link_s(short, long_, links, short_s, long_s)
        self.assertAlmostEqual(got, 1e-3, places=12)

    def test_refuses_equal_lengths(self):
        with self.assertRaises(ValueError):
            baselines.cpu_per_link_s(1.0, 1.0, 3, 600.0, 600.0)


class Noisy(unittest.TestCase):
    def test_spread_against_median(self):
        self.assertFalse(baselines.noisy({"cpu": 1.0, "spread": 0.2}))
        self.assertTrue(baselines.noisy({"cpu": 1.0, "spread": 0.21}))


class NodeLine(unittest.TestCase):
    LINE = ("NODE config=vitals+heart frames=6000 emitted=600 occupied=600 breathing=553 "
            "heart=528 mean_us=76.517 p50_us=0.618 emit_us=780.944 max_us=927.066 "
            "instr_mean=391903 instr_emit=3806725 instr_max=5537816")

    def test_fields(self):
        d = baselines.parse_kv(self.LINE)
        self.assertEqual(d["config"], "vitals+heart")
        self.assertEqual(d["frames"], 6000)
        self.assertEqual(d["instr_max"], 5537816)
        self.assertAlmostEqual(d["mean_us"], 76.517)

    def test_s3_scaling(self):
        # 400k instructions a frame at 20 Hz is 8M a second; at CPI 1.5 on a
        # 240 MHz core that is 12M cycles, 5% of the core.
        self.assertAlmostEqual(baselines.s3_core_share(400_000, 20.0, 1.5), 0.05)
        # 4.8M instructions at CPI 2 is 9.6M cycles, 40 ms at 240 MHz.
        self.assertAlmostEqual(baselines.s3_ms(4_800_000, 2.0), 40.0)


class Diram(unittest.TestCase):
    def test_links_that_fit(self):
        # 171 KB = 175,104 B; minus one 6,404 B arena leaves 168,700.
        self.assertEqual(baselines.links_in_diram(45_408, 6_404), 3)
        self.assertEqual(baselines.links_in_diram(16_584, 6_404), 10)


class CiRuns(unittest.TestCase):
    RUN = {
        "databaseId": 1,
        "headSha": "abc",
        "createdAt": "2026-11-06T15:51:41Z",
        "jobs": [
            _job("rust", "2026-11-06T15:52:21Z", "2026-11-06T15:57:56Z",
                 [("cargo test", "2026-11-06T15:53:00Z", "2026-11-06T15:57:00Z")]),
            _job("edge", "2026-11-06T15:52:21Z", "2026-11-06T15:54:30Z",
                 [("make check", "2026-11-06T15:52:30Z", "2026-11-06T15:54:20Z")]),
        ],
    }

    def test_job_seconds(self):
        jobs = baselines.job_seconds(self.RUN)
        self.assertEqual(jobs["rust"], 335)
        self.assertEqual(jobs["edge"], 129)

    def test_run_wall_is_first_start_to_last_end(self):
        self.assertEqual(baselines.run_wall_seconds(self.RUN), 335)

    def test_runner_seconds_sum_jobs(self):
        self.assertEqual(baselines.runner_seconds(self.RUN), 335 + 129)

    def test_steps_keyed_by_job(self):
        steps = baselines.step_seconds(self.RUN)
        self.assertEqual(steps[("rust", "cargo test")], 240)
        self.assertEqual(steps[("edge", "make check")], 110)


if __name__ == "__main__":
    unittest.main()
