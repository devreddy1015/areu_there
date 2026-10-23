"""Room Sim RNG: the Python port of saans_sim::rng::Rng (spec/roomsim.md §2)."""

import unittest

from wifisense.roomsim.rng import Rng


class TestSpecVectors(unittest.TestCase):
    """The §2 vectors, bit-exact."""

    def test_new_next_u64(self):
        r = Rng(7)
        self.assertEqual([r.next_u64() for _ in range(3)],
                         [12923355070828475994, 5142052590334782674, 15488392906492639638])

    def test_stream_next_u64(self):
        r = Rng.stream(42, 1000)
        self.assertEqual([r.next_u64() for _ in range(3)],
                         [12345297164789059879, 1259461603951423963, 15861779893704225696])

    def test_stream_draws(self):
        r = Rng.stream(42, 1)
        self.assertEqual(r.uniform(), float("0.924814964327024258"))
        self.assertEqual(r.range(0.3, 5.7), float("4.97446416539475589"))
        self.assertEqual([r.gauss() for _ in range(3)], [
            float("-0.793888932810311143"), float("-1.59718765083500513"), float("-0.484773194911475802")])
        self.assertEqual(r.exp(3.0), float("2.50672202106860942"))
        self.assertEqual(r.unit_vector(), (float("0.722861780515479668"), float("0.690992652833581578")))


class TestGaussList(unittest.TestCase):
    def test_identical_to_repeated_gauss(self):
        for odd_start in (False, True):
            a = Rng.stream(3, 9)
            b = Rng.stream(3, 9)
            if odd_start:  # leave a cached spare behind
                a.gauss()
                b.gauss()
            self.assertEqual([a.gauss() for _ in range(301)], b.gauss_list(301))
            self.assertEqual(a.spare, b.spare)
            self.assertEqual(a.next_u64(), b.next_u64())

    def test_streams_differ_and_repeat(self):
        self.assertNotEqual(Rng.stream(0, 1).next_u64(), Rng.stream(0, 2).next_u64())
        self.assertEqual(Rng.stream(5, 100000).uniform(), Rng.stream(5, 100000).uniform())


if __name__ == "__main__":
    unittest.main()
