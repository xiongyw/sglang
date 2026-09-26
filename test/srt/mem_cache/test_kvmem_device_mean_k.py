import unittest

import torch

from sglang.srt.mem_cache.kvmem_mean_k import KVMemDeviceMeanKAccumulator


class TestKVMemDeviceMeanKAccumulator(unittest.TestCase):
    def test_preallocated_accumulator_matches_expected_means(self):
        acc = KVMemDeviceMeanKAccumulator(
            num_blocks=3, block_size=2, kv_heads=1, head_dim=2, device="cpu"
        )
        acc.update(
            torch.tensor([[[1.0, 3.0]], [[3.0, 1.0]], [[9.0, 7.0]]]),
            torch.tensor([0, 1, 4]),
        )

        means, counts = acc.snapshot()

        torch.testing.assert_close(
            means,
            torch.tensor([[[2.0, 2.0]], [[0.0, 0.0]], [[9.0, 7.0]]]),
        )
        torch.testing.assert_close(counts, torch.tensor([2, 0, 1]))

    def test_rejects_position_outside_preallocated_capacity(self):
        acc = KVMemDeviceMeanKAccumulator(
            num_blocks=2, block_size=2, kv_heads=1, head_dim=1, device="cpu"
        )

        with self.assertRaises(ValueError):
            acc.update(torch.ones((1, 1, 1)), torch.tensor([4]))

    def test_repeated_position_is_rejected(self):
        acc = KVMemDeviceMeanKAccumulator(
            num_blocks=2, block_size=2, kv_heads=1, head_dim=1, device="cpu"
        )
        acc.update(torch.ones((1, 1, 1)), torch.tensor([0]))

        with self.assertRaises(ValueError):
            acc.update(torch.ones((1, 1, 1)), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
