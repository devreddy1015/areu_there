import unittest

import evaluate


class TestReliability(unittest.TestCase):
    def test_a_confidence_that_means_what_it_says_has_no_error(self):
        # 0.85 confidence, right 17 times in 20 = 0.85: perfectly calibrated.
        pairs = [(0.85, i < 17) for i in range(20)]
        rows, ece = evaluate.reliability(pairs)
        self.assertEqual(len(rows), 1)
        lo, hi, n, mean_conf, share = rows[0]
        self.assertEqual((lo, hi, n), (0.8, 0.9, 20))
        self.assertAlmostEqual(share, 0.85)
        self.assertAlmostEqual(ece, 0.0)

    def test_overconfidence_by_hand(self):
        # Ten readings at 0.95, right 5 times: gap 0.45.
        # Thirty at 0.15, right 3 times (share 0.10): gap 0.05.
        # ECE = 10/40 * 0.45 + 30/40 * 0.05 = 0.1125 + 0.0375 = 0.15.
        pairs = [(0.95, i < 5) for i in range(10)] + [(0.15, i < 3) for i in range(30)]
        rows, ece = evaluate.reliability(pairs)
        self.assertEqual([r[2] for r in rows], [30, 10])
        self.assertAlmostEqual(ece, 0.15)

    def test_a_confidence_of_exactly_one_lands_in_the_top_bin(self):
        rows, _ = evaluate.reliability([(1.0, True), (0.0, False)])
        self.assertEqual([(r[0], r[2]) for r in rows], [(0.0, 1), (0.9, 1)])


if __name__ == "__main__":
    unittest.main()
