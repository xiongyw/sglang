import unittest

import torch

from sglang.srt.mem_cache.kvmem_scorer import (
    mean_k_block_scores,
    mean_k_block_scores_from_means,
)


class TestMeanKBlockScores(unittest.TestCase):
    def test_scores_are_softmax_mass_over_blocks(self):
        q = torch.tensor([[[1.0]]])
        k = torch.tensor([[[1.0]], [[1.0]], [[-1.0]], [[-1.0]]])
        scores = mean_k_block_scores(q, k, block_size=2)
        expected = torch.softmax(torch.tensor([1.0, -1.0]), dim=0)
        torch.testing.assert_close(scores, expected)
        self.assertAlmostEqual(float(scores.sum()), 1.0, places=6)

    def test_precomputed_means_match_direct_path(self):
        q = torch.randn(3, 4, 8)
        k = torch.randn(11, 2, 8)
        direct = mean_k_block_scores(q, k, block_size=4)
        means = torch.stack(
            [k[0:4].mean(0), k[4:8].mean(0), k[8:11].mean(0)], dim=0
        )
        cached = mean_k_block_scores_from_means(q, means)
        torch.testing.assert_close(cached, direct)

    def test_gqa_maps_query_heads_to_kv_heads(self):
        q = torch.tensor([[[1.0], [1.0], [1.0], [1.0]]])
        k = torch.tensor([[[1.0], [-1.0]], [[-1.0], [1.0]]])
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
