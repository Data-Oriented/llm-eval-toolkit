import unittest

from evaltoolkit.judge_agreement import agreement_report, cohen_kappa
from evaltoolkit.leak_check import accuracy, group_split, make_dataset, random_split


class KappaTests(unittest.TestCase):
    def test_perfect_agreement_is_one(self):
        self.assertAlmostEqual(cohen_kappa(["a", "b", "a"], ["a", "b", "a"]), 1.0)

    def test_known_value(self):
        # 2x2 example: observed 0.7, expected 0.5 -> kappa 0.4
        judge = ["y"] * 5 + ["n"] * 5
        human = ["y"] * 4 + ["n"] + ["n"] * 3 + ["y"] * 2
        self.assertAlmostEqual(cohen_kappa(judge, human), 0.4)

    def test_rejects_unequal_lengths(self):
        with self.assertRaises(ValueError):
            cohen_kappa(["a"], ["a", "b"])

    def test_precision_recall(self):
        rep = agreement_report(["send", "send", "revise"], ["send", "revise", "revise"])
        self.assertAlmostEqual(rep.precision["send"], 0.5)
        self.assertAlmostEqual(rep.recall["revise"], 0.5)


class LeakTests(unittest.TestCase):
    def test_group_split_has_no_shared_groups(self):
        items = make_dataset(n_groups=20, per_group=4)
        test, train = group_split(items)
        self.assertFalse({i["group"] for i in test} & {i["group"] for i in train})

    def test_random_split_inflates_accuracy(self):
        items = make_dataset(n_groups=40, per_group=6)
        leaky = accuracy(*random_split(items))
        honest = accuracy(*group_split(items))
        self.assertGreater(leaky, honest + 0.1)


if __name__ == "__main__":
    unittest.main()
