import unittest

import torch

from sglang.srt.mem_cache.kvmem_mean_k import KVMemMeanKAccumulator


class TestKVMemMeanKAccumulator(unittest.TestCase):
    def test_accumulates_across_chunks_and_partial_blocks(self):
        acc = KVMemMeanKAccumulator(block_size=2, kv_heads=1, head_dim=2)
        acc.update(
            torch.tensor([[[1.0, 3.0]], [[3.0, 1.0]]]),
            torch.tensor([0, 1]),
        )
        acc.update(
            torch.tensor([[[5.0, 5.0]], [[7.0, 9.0]], [[9.0, 7.0]]]),
            torch.tensor([2, 3, 4]),
        )

        means, counts = acc.snapshot()

        torch.testing.assert_close(
            means,
            torch.tensor([[[2.0, 2.0]], [[6.0, 7.0]], [[9.0, 7.0]]]),
        )
        torch.testing.assert_close(counts, torch.tensor([2, 2, 1]))

    def test_out_of_order_updates_use_logical_positions(self):
        acc = KVMemMeanKAccumulator(block_size=4, kv_heads=1, head_dim=1)
        acc.update(torch.tensor([[[8.0]], [[2.0]]]), torch.tensor([5, 1]))

        means, counts = acc.snapshot()

        torch.testing.assert_close(means, torch.tensor([[[2.0]], [[8.0]]]))
        torch.testing.assert_close(counts, torch.tensor([1, 1]))

    def test_repeated_position_is_rejected(self):
        acc = KVMemMeanKAccumulator(block_size=2, kv_heads=1, head_dim=1)
        acc.update(torch.tensor([[[1.0]]]), torch.tensor([0]))

        with self.assertRaises(ValueError):
            acc.update(torch.tensor([[[2.0]]]), torch.tensor([0]))

    def test_empty_update_is_noop(self):
        acc = KVMemMeanKAccumulator(block_size=2, kv_heads=1, head_dim=1)
        acc.update(torch.empty((0, 1, 1)), torch.empty((0,), dtype=torch.long))

        means, counts = acc.snapshot()

        self.assertEqual(tuple(means.shape), (0, 1, 1))
        self.assertEqual(tuple(counts.shape), (0,))


if __name__ == "__main__":
    unittest.main()
