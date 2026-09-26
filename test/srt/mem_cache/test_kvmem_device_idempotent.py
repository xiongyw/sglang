import unittest

import torch

from sglang.srt.mem_cache.kvmem_mean_k import KVMemDeviceMeanKAccumulator


def _acc(num_blocks=8, block_size=2, kv_heads=1, head_dim=2):
    return KVMemDeviceMeanKAccumulator(
        num_blocks=num_blocks,
        block_size=block_size,
        kv_heads=kv_heads,
        head_dim=head_dim,
        device="cpu",
    )


class TestDeviceAccumulatorIdempotent(unittest.TestCase):
    def test_stores_new_positions(self):
        acc = _acc()
        stored = acc.update_idempotent(
            torch.ones((3, 1, 2)), torch.tensor([0, 1, 2])
        )
        self.assertEqual(stored, 3)
        means, counts = acc.snapshot()
        self.assertEqual(int(counts[:2].sum()), 3)

    def test_re_staged_positions_are_dropped(self):
        acc = _acc()
        acc.update_idempotent(torch.ones((4, 1, 2)), torch.tensor([0, 1, 2, 3]))
        stored = acc.update_idempotent(
            torch.full((4, 1, 2), 5.0), torch.tensor([2, 3, 4, 5])
        )
        self.assertEqual(stored, 2)
        means, counts = acc.snapshot()
        self.assertEqual(int(counts.sum()), 6)
        # First sighting wins: block 1 keeps the 1.0 rows for positions 2,3.
        self.assertAlmostEqual(float(means[1][0][0]), 1.0)

    def test_partially_seen_batch_stores_only_the_fresh_rows(self):
        acc = _acc()
        acc.update_idempotent(torch.ones((2, 1, 2)), torch.tensor([4, 5]))
        stored = acc.update_idempotent(
            torch.full((3, 1, 2), 3.0), torch.tensor([5, 6, 7])
        )
        self.assertEqual(stored, 2)
        means, counts = acc.snapshot()
        self.assertAlmostEqual(float(means[2][0][0]), 1.0)  # position 4,5 block
        self.assertAlmostEqual(float(means[3][0][0]), 3.0)

    def test_restart_at_zero_with_overlap_resets(self):
        acc = _acc()
        acc.update_idempotent(torch.ones((4, 1, 2)), torch.tensor([0, 1, 2, 3]))
        stored = acc.update_idempotent(
            torch.full((4, 1, 2), 9.0), torch.tensor([0, 1, 2, 3])
        )
        self.assertEqual(stored, 4)
        means, counts = acc.snapshot()
        self.assertEqual(int(counts.sum()), 4)
        self.assertAlmostEqual(float(means[0][0][0]), 9.0)

    def test_single_row_at_zero_is_not_a_restart(self):
        acc = _acc()
        acc.update_idempotent(torch.ones((2, 1, 2)), torch.tensor([0, 1]))
        stored = acc.update_idempotent(
            torch.full((1, 1, 2), 7.0), torch.tensor([0])
        )
        self.assertEqual(stored, 0)

    def test_empty_batch_is_a_noop(self):
        acc = _acc()
        stored = acc.update_idempotent(
            torch.empty((0, 1, 2)), torch.empty((0,), dtype=torch.long)
        )
        self.assertEqual(stored, 0)

    def test_position_beyond_capacity_is_rejected(self):
        acc = _acc(num_blocks=2, block_size=2)  # 4 positions max
        with self.assertRaises(ValueError):
            acc.update_idempotent(torch.ones((1, 1, 2)), torch.tensor([4]))

    def test_shape_and_dtype_are_validated(self):
        acc = _acc()
        with self.assertRaises(ValueError):
            acc.update_idempotent(torch.ones((1, 3, 2)), torch.tensor([0]))
        with self.assertRaises(ValueError):
            acc.update_idempotent(
                torch.ones((2, 1, 2)), torch.tensor([0.0, 1.0])
            )

    def test_reset_clears_everything(self):
        acc = _acc()
        acc.update_idempotent(torch.ones((3, 1, 2)), torch.tensor([0, 1, 2]))
        acc.reset()
        means, counts = acc.snapshot()
        self.assertEqual(int(counts.sum()), 0)
        self.assertEqual(acc.update_idempotent(torch.ones((1, 1, 2)), torch.tensor([0])), 1)

    def test_seen_any_reports_overlap(self):
        acc = _acc()
        acc.update_idempotent(torch.ones((2, 1, 2)), torch.tensor([0, 1]))
        self.assertTrue(acc.seen_any([1]))
        self.assertFalse(acc.seen_any([2, 3]))

    def test_device_is_preserved(self):
        acc = _acc()
        acc.update_idempotent(torch.ones((2, 1, 2)), torch.tensor([0, 1]))
        means, counts = acc.snapshot()
        self.assertEqual(means.device.type, "cpu")
        self.assertEqual(counts.device.type, "cpu")


if __name__ == "__main__":
    unittest.main()
