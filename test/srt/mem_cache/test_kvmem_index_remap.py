import unittest

import torch

from sglang.srt.mem_cache.kvmem_index_remap import (
    keep_positions_for_history,
    reduction_ratio,
    remap_kv_indices,
)


def _batched(rows):
    """Build (kv_indptr, kv_indices) from a list of per-request slot lists."""
    flat = [slot for row in rows for slot in row]
    indptr = torch.tensor(
        [0] + list(torch.tensor([len(r) for r in rows]).cumsum(0).tolist()),
        dtype=torch.int32,
    )
    return indptr, torch.tensor(flat, dtype=torch.int64)


class TestKeepPositions(unittest.TestCase):
    def test_keeps_sink_recent_and_tail(self):
        # 10 blocks of 4 = 40 tokens; select only block 3.
        keep = keep_positions_for_history(40, 4, [3], sink_blocks=1, recent_blocks=2)
        self.assertEqual(keep[0:4], [0, 1, 2, 3])          # sink block
        self.assertEqual(keep[4:8], [12, 13, 14, 15])      # block 3
        self.assertEqual(keep[-8:], [32, 33, 34, 35, 36, 37, 38, 39])  # recent 2

    def test_partial_tail_block_is_clamped_and_kept(self):
        # Block 2 is partial (positions 8,9): sink + recent keeps it, and only
        # it, so block 1 is legitimately dropped.
        keep = keep_positions_for_history(10, 4, [], sink_blocks=1, recent_blocks=1)
        self.assertEqual(keep, [0, 1, 2, 3, 8, 9])

    def test_recent_blocks_covering_all_blocks_keeps_everything(self):
        keep = keep_positions_for_history(10, 4, [], sink_blocks=0, recent_blocks=3)
        self.assertEqual(keep, list(range(10)))

    def test_selecting_every_block_is_identity(self):
        keep = keep_positions_for_history(12, 4, [0, 1, 2], recent_blocks=0)
        self.assertEqual(keep, list(range(12)))

    def test_out_of_range_blocks_are_ignored(self):
        keep = keep_positions_for_history(8, 4, [99], sink_blocks=0, recent_blocks=0)
        self.assertEqual(keep, [])

    def test_ascending_and_unique(self):
        keep = keep_positions_for_history(40, 4, [5, 5, 2], recent_blocks=1)
        self.assertEqual(keep, sorted(set(keep)))

    def test_invalid_arguments_rejected(self):
        with self.assertRaises(ValueError):
            keep_positions_for_history(0, 4, [])
        with self.assertRaises(ValueError):
            keep_positions_for_history(8, 0, [])


class TestRemapKvIndices(unittest.TestCase):
    def test_keep_everything_is_exact_identity(self):
        indptr, indices = _batched([[10, 11, 12, 13], [20, 21]])
        new_indptr, new_indices = remap_kv_indices(
            indptr, indices, [list(range(4)), list(range(2))]
        )
        self.assertEqual(new_indptr.tolist(), [0, 4, 6])
        self.assertEqual(new_indices.tolist(), [10, 11, 12, 13, 20, 21])

    def test_drops_selected_positions_and_recomputes_indptr(self):
        indptr, indices = _batched([[10, 11, 12, 13, 14, 15], [20, 21, 22]])
        new_indptr, new_indices = remap_kv_indices(
            indptr, indices, [[0, 1, 4, 5], [1]]
        )
        self.assertEqual(new_indptr.tolist(), [0, 4, 5])
        self.assertEqual(new_indices.tolist(), [10, 11, 14, 15, 21])

    def test_untouched_request_passes_through(self):
        indptr, indices = _batched([[10, 11, 12], [20, 21, 22, 23]])
        new_indptr, new_indices = remap_kv_indices(indptr, indices, [None, [0, 3]])
        self.assertEqual(new_indptr.tolist(), [0, 3, 5])
        self.assertEqual(new_indices.tolist(), [10, 11, 12, 20, 23])

    def test_empty_keep_list_drops_whole_history(self):
        indptr, indices = _batched([[10, 11], [20, 21]])
        new_indptr, new_indices = remap_kv_indices(indptr, indices, [[], None])
        self.assertEqual(new_indptr.tolist(), [0, 0, 2])
        self.assertEqual(new_indices.tolist(), [20, 21])

    def test_inputs_are_not_mutated(self):
        indptr, indices = _batched([[10, 11, 12, 13]])
        before = (indptr.clone(), indices.clone())
        remap_kv_indices(indptr, indices, [[0, 3]])
        torch.testing.assert_close(indptr, before[0])
        torch.testing.assert_close(indices, before[1])

    def test_non_ascending_positions_rejected(self):
        indptr, indices = _batched([[10, 11, 12, 13]])
        with self.assertRaises(ValueError):
            remap_kv_indices(indptr, indices, [[2, 1]])

    def test_out_of_range_position_rejected(self):
        indptr, indices = _batched([[10, 11]])
        with self.assertRaises(ValueError):
            remap_kv_indices(indptr, indices, [[2]])

    def test_dtype_and_device_preserved(self):
        indptr, indices = _batched([[10, 11, 12, 13]])
        new_indptr, new_indices = remap_kv_indices(indptr, indices, [[0, 2]])
        self.assertEqual(new_indptr.dtype, indptr.dtype)
        self.assertEqual(new_indices.dtype, indices.dtype)
        self.assertEqual(new_indices.device, indices.device)


class TestReductionRatio(unittest.TestCase):
    def test_ratio(self):
        self.assertAlmostEqual(reduction_ratio(100, 25), 0.25)
        self.assertEqual(reduction_ratio(0, 0), 1.0)


if __name__ == "__main__":
    unittest.main()
