import unittest

import torch

from sglang.srt.mem_cache.kvmem_slot_plan import (
    KVMemPlanBuffer,
    compacted_slots,
    plan_keep_blocks,
)


def _row(slots, width=16):
    row = torch.full((width,), -1, dtype=torch.int32)
    row[: len(slots)] = torch.tensor(slots, dtype=torch.int32)
    return row


class TestCompactedSlots(unittest.TestCase):
    def test_keeps_selected_positions_in_order(self):
        row = _row([10, 11, 12, 13, 14, 15, 16, 17])
        self.assertEqual(
            compacted_slots(row, 8, [0, 1, 4, 5]), [10, 11, 14, 15]
        )

    def test_keeping_everything_is_identity(self):
        row = _row([10, 11, 12, 13])
        self.assertEqual(compacted_slots(row, 4, [0, 1, 2, 3]), [10, 11, 12, 13])

    def test_partial_history_is_respected(self):
        row = _row([10, 11, 12, 13, 14, 15])
        self.assertEqual(compacted_slots(row, 4, [0, 2]), [10, 12])

    def test_empty_keep_is_empty(self):
        row = _row([10, 11])
        self.assertEqual(compacted_slots(row, 2, []), [])

    def test_positions_beyond_history_rejected(self):
        row = _row([10, 11, 12, 13])
        with self.assertRaises(ValueError):
            compacted_slots(row, 2, [0, 3])


class TestPlanKeepBlocks(unittest.TestCase):
    def test_under_budget_keeps_everything(self):
        keep, evicted = plan_keep_blocks(
            history_len=512,
            block_size=128,
            selected_blocks=[0, 1, 2, 3],
            budget_tokens=1024,
            sink_blocks=1,
            recent_blocks=1,
        )
        self.assertEqual(len(keep), 512)
        self.assertEqual(evicted, [])

    def test_over_budget_evicts_unselected_blocks(self):
        # 8 blocks of 128; mandatory sink+recent = blocks 0 and 7, budget 384
        # leaves room for one more block, which the selector fills with block 1.
        keep, evicted = plan_keep_blocks(
            history_len=1024,
            block_size=128,
            selected_blocks=[1],
            budget_tokens=384,
            sink_blocks=1,
            recent_blocks=1,
        )
        self.assertEqual(len(keep), 384)
        self.assertEqual(evicted, [2, 3, 4, 5, 6])
        self.assertEqual(keep[:128], list(range(0, 128)))
        self.assertEqual(keep[128:256], list(range(128, 256)))
        self.assertEqual(keep[256:], list(range(7 * 128, 1024)))

    def test_already_evicted_blocks_stay_evicted(self):
        # Monotonic: a block evicted earlier must not be re-admitted even if the
        # selector now asks for it.
        keep, evicted = plan_keep_blocks(
            history_len=1024,
            block_size=128,
            selected_blocks=[0, 3],
            budget_tokens=768,
            sink_blocks=1,
            recent_blocks=1,
            already_evicted=[3],
        )
        self.assertNotIn(3 * 128, keep)
        self.assertIn(3, evicted)

    def test_sink_and_recent_are_never_evicted(self):
        keep, evicted = plan_keep_blocks(
            history_len=640,
            block_size=128,
            selected_blocks=[],
            budget_tokens=1,
            sink_blocks=1,
            recent_blocks=2,
        )
        # 5 blocks: sink (0), recent (3,4); blocks 1,2 evictable even though the
        # budget cannot fit the mandatory blocks.
        self.assertEqual(len(keep), 3 * 128)
        self.assertEqual(evicted, [1, 2])

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):
            plan_keep_blocks(
                history_len=0,
                block_size=128,
                selected_blocks=[],
                budget_tokens=64,
            )


class TestPlanBuffer(unittest.TestCase):
    def test_write_and_read_back_a_request(self):
        buffer = KVMemPlanBuffer(max_requests=2, max_kept_tokens=8, device="cpu")
        buffer.write(0, [10, 11, 12])
        buffer.write(1, [20, 21])
        buffer.finalize(batch_size=2)
        self.assertEqual(buffer.indptr().tolist(), [0, 3, 5])
        self.assertEqual(buffer.slots()[:5].tolist(), [10, 11, 12, 20, 21])

    def test_default_empty_plan_is_zero_length(self):
        buffer = KVMemPlanBuffer(max_requests=2, max_kept_tokens=4, device="cpu")
        buffer.finalize(batch_size=2)
        self.assertEqual(buffer.indptr().tolist(), [0, 0, 0])

    def test_overflow_is_rejected(self):
        buffer = KVMemPlanBuffer(max_requests=1, max_kept_tokens=2, device="cpu")
        with self.assertRaises(ValueError):
            buffer.write(0, [1, 2, 3])

    def test_indptr_dtype_matches_kernel_expectation(self):
        buffer = KVMemPlanBuffer(max_requests=1, max_kept_tokens=4, device="cpu")
        buffer.write(0, [1, 2])
        buffer.finalize(batch_size=1)
        self.assertEqual(buffer.indptr().dtype, torch.int32)
        self.assertEqual(buffer.slots().dtype, torch.int64)

    def test_device_publish_is_contiguous_across_rows(self):
        buffer = KVMemPlanBuffer(max_requests=2, max_kept_tokens=6, device="cpu")
        written = buffer.publish_device(
            [
                (torch.tensor([7, 8, 9]), 3),
                (torch.tensor([4]), 1),
            ]
        )
        self.assertEqual(written, 4)
        self.assertEqual(buffer.indptr().tolist(), [0, 3, 4])
        # Contiguous, i.e. row 1 starts right after row 0 (no stride padding).
        self.assertEqual(buffer.slots()[:4].tolist(), [7, 8, 9, 4])

    def test_device_publish_row_without_plan_is_zero_length(self):
        buffer = KVMemPlanBuffer(max_requests=3, max_kept_tokens=4, device="cpu")
        buffer.publish_device([(torch.tensor([1, 2]), 2), (None, 0)])
        self.assertEqual(buffer.indptr().tolist(), [0, 2, 2, 2])

    def test_device_publish_rejects_count_and_slots_disagreement(self):
        buffer = KVMemPlanBuffer(max_requests=1, max_kept_tokens=4, device="cpu")
        with self.assertRaises(ValueError):
            buffer.publish_device([(torch.tensor([1, 2, 3]), 2)])

    def test_device_publish_rejects_capacity_overflow(self):
        buffer = KVMemPlanBuffer(max_requests=1, max_kept_tokens=2, device="cpu")
        with self.assertRaises(ValueError):
            buffer.publish_device([(torch.tensor([1, 2, 3]), 3)])

    def test_device_publish_rejects_missing_slots_for_a_count(self):
        buffer = KVMemPlanBuffer(max_requests=1, max_kept_tokens=4, device="cpu")
        with self.assertRaises(ValueError):
            buffer.publish_device([(None, 2)])


if __name__ == "__main__":
    unittest.main()
