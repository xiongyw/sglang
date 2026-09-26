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


class TestDeviceHostParity(unittest.TestCase):
    """The device twin must be numerically equivalent to the host one."""

    def _drive_both(self, batches):
        from sglang.srt.mem_cache.kvmem_mean_k import KVMemMeanKAccumulator

        host = KVMemMeanKAccumulator(block_size=2, kv_heads=1, head_dim=2)
        device = _acc()
        for key, positions in batches:
            host.update_idempotent(key, positions)
            device.update_idempotent(key, positions)
        return host.snapshot(), device.snapshot()

    def _assert_same_used_region(self, host_pair, device_pair):
        """Compare the used blocks: the host grows its buffers, the device one
        is preallocated, so only the prefix that the host materialised is
        comparable."""
        (hm, hc), (dm, dc) = host_pair, device_pair
        n = hm.shape[0]
        torch.testing.assert_close(hm, dm[:n])
        torch.testing.assert_close(hc, dc[:n])

    def test_agree_on_plain_accumulation(self):
        torch.manual_seed(0)
        key = torch.randn(6, 1, 2)
        self._assert_same_used_region(
            *self._drive_both([(key, torch.arange(6))])
        )

    def test_agree_on_overlapping_restaging(self):
        torch.manual_seed(1)
        first = torch.randn(4, 1, 2)
        second = torch.randn(4, 1, 2)
        self._assert_same_used_region(
            *self._drive_both(
                [(first, torch.arange(4)), (second, torch.tensor([2, 3, 4, 5]))]
            )
        )

    def test_agree_on_slot_restart(self):
        torch.manual_seed(2)
        first = torch.randn(4, 1, 2)
        second = torch.randn(4, 1, 2)
        self._assert_same_used_region(
            *self._drive_both([(first, torch.arange(4)), (second, torch.arange(4))])
        )

    def test_agree_on_three_interleaved_batches(self):
        torch.manual_seed(3)
        batches = [
            (torch.randn(3, 1, 2), torch.tensor([0, 1, 2])),
            (torch.randn(4, 1, 2), torch.tensor([3, 4, 5, 6])),
            (torch.randn(3, 1, 2), torch.tensor([5, 6, 7])),
        ]
        self._assert_same_used_region(*self._drive_both(batches))


if __name__ == "__main__":
    unittest.main()
