from __future__ import annotations

import unittest

from src.metrics import recall_at_k


class RecallAtKTest(unittest.TestCase):
    def test_single_relevant_found(self) -> None:
        self.assertEqual(recall_at_k({"q": ["a"]}, {"q": {"a"}}, k=50), 1.0)

    def test_one_of_two_found(self) -> None:
        self.assertEqual(recall_at_k({"q": ["a"]}, {"q": {"a", "b"}}, k=50), 0.5)

    def test_single_relevant_missed(self) -> None:
        self.assertEqual(recall_at_k({"q": ["x"]}, {"q": {"a"}}, k=50), 0.0)

    def test_mean_from_statement(self) -> None:
        predictions = {"A": ["a"], "B": ["b1"], "C": ["x"]}
        relevant = {"A": {"a"}, "B": {"b1", "b2"}, "C": {"c"}}
        self.assertEqual(recall_at_k(predictions, relevant, k=50), 0.5)

    def test_duplicates_do_not_help(self) -> None:
        score = recall_at_k({"q": ["a", "a", "x"]}, {"q": {"a", "b"}}, k=2)
        self.assertEqual(score, 0.5)

    def test_empty_labels_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            recall_at_k({"q": []}, {"q": set()})


if __name__ == "__main__":
    unittest.main()
