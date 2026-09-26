import unittest

from sglang.srt.mem_cache.kvmem_selector import (
    KVMemBlock,
    KVMemBlockSelector,
    KVMemSelectionConfig,
)


class TestKVMemBlockSelector(unittest.TestCase):
    def make_blocks(self, n=8, tokens=128):
        return [
            KVMemBlock(block_id=i, start=i * tokens, n_tokens=tokens)
            for i in range(n)
        ]

    def test_budget_keeps_sink_recent_and_highest_scored_middle(self):
        blocks = self.make_blocks()
        selector = KVMemBlockSelector(
            KVMemSelectionConfig(
                budget_tokens=4 * 128,
                sink_blocks=1,
                recent_blocks=1,
            )
        )
        scores = {1: 1.0, 2: 9.0, 3: 3.0, 4: 8.0, 5: 2.0, 6: 7.0}

        result = selector.select(blocks, scores)

        self.assertEqual(result.block_ids, [0, 2, 4, 7])
        self.assertEqual(result.total_tokens, 4 * 128)

    def test_ties_prefer_newer_block_deterministically(self):
        blocks = self.make_blocks(6)
        selector = KVMemBlockSelector(
            KVMemSelectionConfig(budget_tokens=3 * 128, sink_blocks=1)
        )

        result = selector.select(blocks, {1: 5.0, 2: 5.0, 3: 5.0, 4: 5.0})

        self.assertEqual(result.block_ids, [0, 3, 4])

    def test_diff_reports_stage_in_and_stage_out(self):
        blocks = self.make_blocks(5)
        selector = KVMemBlockSelector(
            KVMemSelectionConfig(budget_tokens=3 * 128, sink_blocks=1)
        )
        result = selector.select(blocks, {1: 1.0, 2: 9.0, 3: 8.0, 4: 7.0})

        diff = selector.diff(result, resident_block_ids=[0, 1, 4])

        self.assertEqual(diff.stage_in, [2, 3])
        self.assertEqual(diff.stage_out, [1, 4])
        self.assertEqual(diff.reused, [0])

    def test_partial_last_block_counts_actual_tokens(self):
        blocks = self.make_blocks(3)
        blocks[-1] = KVMemBlock(block_id=2, start=256, n_tokens=17)
        selector = KVMemBlockSelector(
            KVMemSelectionConfig(budget_tokens=273, sink_blocks=1)
        )

        result = selector.select(blocks, {1: 10.0, 2: 20.0})

        self.assertEqual(result.block_ids, [0, 1, 2])
        self.assertEqual(result.total_tokens, 273)


if __name__ == "__main__":
    unittest.main()
