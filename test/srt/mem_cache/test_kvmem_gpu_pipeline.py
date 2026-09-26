import unittest

import torch

from sglang.srt.mem_cache.kvmem_selector import (
    KVMemBlock,
    KVMemBlockSelector,
    KVMemSelectionConfig,
)
from sglang.srt.mem_cache.kvmem_scorer import mean_k_block_scores_from_means
from sglang.srt.mem_cache.kvmem_simulation import KVMemResidentSimulation


class TestKVMemGpuResidentPipeline(unittest.TestCase):
    def test_cached_scores_drive_selection_and_residency(self):
        q = torch.tensor([[[1.0], [1.0]]])
        means = torch.tensor(
            [
                [[1.0]],
                [[-1.0]],
                [[0.5]],
                [[-0.5]],
            ]
        )
        scores = mean_k_block_scores_from_means(q, means, scale=1.0)
        blocks = [KVMemBlock(i, i * 4, 4) for i in range(4)]
        selector = KVMemBlockSelector(
            KVMemSelectionConfig(budget_tokens=8, sink_blocks=1)
        )
        sim = KVMemResidentSimulation(selector)

        first = sim.step(blocks, {i: float(scores[i]) for i in range(4)})
        second = sim.step(blocks, {i: float(scores[i]) for i in reversed(range(4))})

        self.assertEqual(first.selected, [0, 2])
        self.assertEqual(first.total_tokens, 8)
        self.assertEqual(second.selected, [0, 2])
        self.assertEqual(second.stage_in, [])
        self.assertEqual(second.stage_out, [])
        self.assertEqual(second.reused, 2)


if __name__ == "__main__":
    unittest.main()
