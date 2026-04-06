from __future__ import annotations

import unittest

from training.evaluate import ndcg_at_k, recall_at_k


class EvaluationMetricTests(unittest.TestCase):
    def test_recall_at_k(self) -> None:
        score = recall_at_k(["a", "b", "c"], ["b", "d"], 2)
        self.assertEqual(score, 0.5)

    def test_ndcg_at_k(self) -> None:
        score = ndcg_at_k(["a", "b", "c"], ["b", "c"], 3)
        self.assertGreater(score, 0.5)
        self.assertLessEqual(score, 1.0)


if __name__ == "__main__":
    unittest.main()

