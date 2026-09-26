import os
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.kvmem_apply import (
    compact_kv_indices_inplace,
    plan_keep_positions_for_batch,
)
from sglang.srt.mem_cache.kvmem_hook import (
    KVMemHookConfig,
    get_kvmem_registry,
    reset_kvmem_hook,
    set_kvmem_hook_config,
)


class FakeMode:
    def __init__(self, extend=True, decode=False, verify=False):
        self._extend = extend
        self._decode = decode
        self._verify = verify

    def is_extend(self):
        return self._extend

    def is_decode(self):
        return self._decode

    def is_decode_or_idle(self):
        return self._decode

    def is_target_verify(self):
        return self._verify

    def is_draft_extend(self):
        return False

    def is_idle(self):
        return False


class FakeForwardBatch:
    def __init__(self, slots=(0,), mode=None, bs=None):
        self.forward_mode = mode or FakeMode()
        self.req_pool_indices = torch.tensor(list(slots), dtype=torch.long)
        self.batch_size = bs if bs is not None else len(slots)


def _enable(block_size=4, budget=4096, max_tokens=65536):
    set_kvmem_hook_config(
        KVMemHookConfig(
            enabled=True,
            block_size=block_size,
            max_tokens=max_tokens,
            selection_budget_tokens=budget,
        )
    )


def _buffers(rows):
    """Build writable indptr/kv_indices buffers like the graph path uses."""
    flat = [slot for row in rows for slot in row]
    indptr = torch.zeros((len(rows) + 2,), dtype=torch.int32)
    indptr[: len(rows) + 1] = torch.tensor(
        [0] + list(torch.tensor([len(r) for r in rows]).cumsum(0).tolist()),
        dtype=torch.int32,
    )
    indices = torch.zeros(
        (max(1, len(flat)),), dtype=torch.int64
    )
    if flat:
        indices[: len(flat)] = torch.tensor(flat, dtype=torch.int64)
    return indptr, indices


class TestPlanKeepPositionsForBatch(unittest.TestCase):
    def setUp(self):
        # The whole path is opt-in; every test runs with it enabled unless it
        # turns it off on purpose.
        self._env = mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        reset_kvmem_hook()

    def test_no_selection_means_nothing_applies(self):
        _enable()
        self.assertIsNone(
            plan_keep_positions_for_batch(
                FakeForwardBatch(), torch.tensor([0, 32], dtype=torch.int32)
            )
        )

    def test_non_budget_limited_selection_is_not_used(self):
        _enable()
        get_kvmem_registry().record_selection("slot:0", [0], budget_limited=False)
        self.assertIsNone(
            plan_keep_positions_for_batch(
                FakeForwardBatch(), torch.tensor([0, 32], dtype=torch.int32)
            )
        )

    def test_budget_limited_selection_yields_keep_positions(self):
        _enable(block_size=4, budget=16)
        # 8 blocks of 4; sink+recent+2 selected blocks of room.
        get_kvmem_registry().record_selection(
            "slot:0", [0, 1], budget_limited=True
        )
        keep = plan_keep_positions_for_batch(
            FakeForwardBatch(), torch.tensor([0, 32], dtype=torch.int32)
        )
        self.assertEqual(len(keep), 1)
        self.assertIsNotNone(keep[0])
        self.assertIn(0, keep[0])
        self.assertLess(len(keep[0]), 32)

    def test_zero_history_row_is_left_alone(self):
        # The first prefill chunk has prefix 0; the planner must not try to bound
        # it (and must not raise).
        _enable(block_size=4, budget=16)
        get_kvmem_registry().record_selection("slot:0", [0], budget_limited=True)
        self.assertIsNone(
            plan_keep_positions_for_batch(
                FakeForwardBatch(), torch.tensor([0, 0], dtype=torch.int32)
            )
        )

    def test_short_indptr_is_refused(self):
        _enable()
        get_kvmem_registry().record_selection("slot:0", [0], budget_limited=True)
        keep = plan_keep_positions_for_batch(
            FakeForwardBatch(slots=(0, 1)), torch.tensor([0, 8], dtype=torch.int32)
        )
        self.assertIsNone(keep)


class TestCompactInPlace(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"})
        self._env.start()

    def tearDown(self):
        self._env.stop()
        reset_kvmem_hook()

    def test_keep_everything_leaves_buffers_untouched(self):
        _enable(block_size=4, budget=4096)
        get_kvmem_registry().record_selection("slot:0", [0, 1], budget_limited=True)
        indptr, indices = _buffers([[10, 11, 12, 13, 14, 15, 16, 17]])
        before = (indptr.clone(), indices.clone())
        result = compact_kv_indices_inplace(
            FakeForwardBatch(), indptr, indices, [list(range(8))]
        )
        self.assertIsNone(result)
        torch.testing.assert_close(indptr, before[0])
        torch.testing.assert_close(indices, before[1])

    def test_compaction_moves_slots_and_rewrites_indptr(self):
        _enable(block_size=4, budget=4096)
        indptr, indices = _buffers([[10, 11, 12, 13, 14, 15, 16, 17]])
        # Keep blocks 0 and 1 only (positions 0-7 minus 4-7 -> keep 0..3 and 4..7
        # is everything; so keep just positions 0..3 plus the tail 4..7 sliced).
        result = compact_kv_indices_inplace(
            FakeForwardBatch(), indptr, indices, [[0, 1, 6, 7]]
        )
        self.assertEqual(result, (4, 8))
        self.assertEqual(indptr.tolist()[:2], [0, 4])
        self.assertEqual(indices[:4].tolist(), [10, 11, 16, 17])

    def test_two_requests_compact_independently(self):
        _enable(block_size=4, budget=4096)
        indptr, indices = _buffers(
            [[10, 11, 12, 13, 14, 15, 16, 17], [20, 21, 22, 23]]
        )
        result = compact_kv_indices_inplace(
            FakeForwardBatch(slots=(0, 1)),
            indptr,
            indices,
            [[0, 1, 6, 7], None],
        )
        self.assertEqual(result, (8, 12))
        self.assertEqual(indptr.tolist()[:3], [0, 4, 8])
        self.assertEqual(indices[:8].tolist(), [10, 11, 16, 17, 20, 21, 22, 23])

    def test_rows_without_a_plan_keep_their_slice(self):
        _enable(block_size=4, budget=4096)
        indptr, indices = _buffers([[10, 11, 12, 13], [20, 21]])
        result = compact_kv_indices_inplace(
            FakeForwardBatch(slots=(0, 1)), indptr, indices, [None, [1]]
        )
        self.assertEqual(result, (5, 6))
        self.assertEqual(indptr.tolist()[:3], [0, 4, 5])
        self.assertEqual(indices[:5].tolist(), [10, 11, 12, 13, 21])

    def test_empty_plan_drops_the_whole_history(self):
        _enable(block_size=4, budget=4096)
        indptr, indices = _buffers([[10, 11, 12, 13]])
        result = compact_kv_indices_inplace(
            FakeForwardBatch(), indptr, indices, [[]]
        )
        self.assertEqual(result, (0, 4))
        self.assertEqual(indptr.tolist()[:2], [0, 0])

    def test_second_compaction_of_the_same_buffer_is_skipped(self):
        # Compacting an already-compacted buffer would shrink it twice, so the
        # guard must treat a shorter-than-already-compacted history as stale.
        _enable(block_size=4, budget=4096)
        fake = FakeForwardBatch()
        indptr, indices = _buffers([[10, 11, 12, 13, 14, 15, 16, 17]])
        first = compact_kv_indices_inplace(fake, indptr, indices, [[0, 1]])
        self.assertIsNotNone(first)
        kept, _history = first
        after_first = indices[:kept].clone()

        # Same (now shorter) buffer again without a refill: nothing to do.
        second = compact_kv_indices_inplace(fake, indptr, indices, [[0, 1]])
        self.assertIsNone(second)
        torch.testing.assert_close(indices[:kept], after_first)

    def test_compaction_proceeds_again_after_a_refill(self):
        _enable(block_size=4, budget=4096)
        fake = FakeForwardBatch()
        indptr, indices = _buffers([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])
        first = compact_kv_indices_inplace(fake, indptr, indices, [[0, 1]])
        self.assertIsNotNone(first)

        # A refill restores the full history (here: longer than what we kept),
        # so the next compaction is allowed.
        indptr, indices = _buffers([[10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]])
        second = compact_kv_indices_inplace(fake, indptr, indices, [[0, 1]])
        self.assertIsNotNone(second)

    def test_capture_in_progress_is_skipped(self):
        _enable()
        indptr, indices = _buffers([[10, 11, 12, 13]])
        before = indices.clone()
        with mock.patch(
            "sglang.srt.mem_cache.kvmem_apply._capture_in_progress",
            return_value=True,
        ):
            result = compact_kv_indices_inplace(
                FakeForwardBatch(), indptr, indices, [[0]]
            )
        self.assertIsNone(result)
        torch.testing.assert_close(indices, before)

    def test_disabled_apply_is_skipped(self):
        _enable()
        indptr, indices = _buffers([[10, 11, 12, 13]])
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_KVMEM_APPLY", None)
            from sglang.srt.mem_cache.kvmem_apply import apply_enabled

            self.assertFalse(apply_enabled())
            self.assertIsNone(
                compact_kv_indices_inplace(FakeForwardBatch(), indptr, indices, [[0]])
            )


if __name__ == "__main__":
    unittest.main()
