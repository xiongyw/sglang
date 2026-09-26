import unittest

import torch

from sglang.srt.mem_cache.kvmem_scorer import mean_k_block_scores


class TestMeanKBlockScores(unittest.TestCase):
    def test_scores_are_softmax_mass_over_blocks(self):
        # One query/head, two blocks, one-dimensional keys.
        q = torch.tensor([[[1.0]]])
        k = torch.tensor([[[1.0]], [[1.0]], [[-1.0]], [[-1.0]]])

        scores = mean_k_block_scores(q, k, block_size=2)

        expected = torch.softmax(torch.tensor([1.0, -1.0]), dim=0)
        torch.testing.assert_close(scores, expected)
        self.assertAlmostEqual(float(scores.sum()), 1.0, places=6)

    def test_gqa_maps_query_heads_to_kv_heads(self):
        # Four query heads share two KV heads: q heads 0/2 use kv 0,
        # q heads 1/3 use kv 1. Each block contains one token.
        q = torch.tensor(
            [
                [[1.0], [1.0], [1.0], [1.0]],
            ]
        )
        k = torch.tensor(
            [
                [[1.0], [-1.0]],
                [[-1.0], [1.0]],
            ]
        )

        scores = mean_k_block_scores(q, k, block_size=1)

        self.assertEqual(tuple(scores.shape), (2,))
        self.assertAlmostEqual(float(scores.sum()), 4.0, places=6)
        self.assertAlmostEqual(float(scores[0]), 2.0, places=5)
        self.assertAlmostEqual(float(scores[1]), 2.0, places=5)

    def test_partial_block_is_included(self):
        q = torch.ones((1, 1, 2))
        k = torch.ones((3, 1, 2))

        scores = mean_k_block_scores(q, k, block_size=2)

        self.assertEqual(tuple(scores.shape), (2,))
        torch.testing.assert_close(scores, torch.tensor([0.5, 0.5]))


if __name__ == "__main__":
    unittest.main()
