import os
import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.kvmem_apply import (
    apply_enabled,
    maybe_apply_kvmem_selection,
    reset_apply_state,
)
from sglang.srt.mem_cache.kvmem_hook import (
    KVMemHookConfig,
    get_kvmem_registry,
    reset_kvmem_hook,
    set_kvmem_hook_config,
)


class FakeMode:
    def __init__(self, extend=True, verify=False):
        self._extend = extend
        self._verify = verify

    def is_extend(self):
        return self._extend

    def is_target_verify(self):
        return self._verify


class FakeForwardBatch:
    def __init__(self, slots=(0,), extend=True):
        self.forward_mode = FakeMode(extend=extend)
        self.req_pool_indices = torch.tensor(list(slots), dtype=torch.long)


class FakeMetadata:
    def __init__(self, rows):
        flat = [slot for row in rows for slot in row]
        self.kv_indptr = torch.tensor(
            [0] + list(torch.tensor([len(r) for r in rows]).cumsum(0).tolist()),
            dtype=torch.int32,
        )
        self.kv_indices = torch.tensor(flat, dtype=torch.int64)


class FakeBackend:
    def __init__(self, rows):
        self.forward_metadata = FakeMetadata(rows)


def _enable(block_size=4, budget=8):
    set_kvmem_hook_config(
        KVMemHookConfig(
            enabled=True,
            block_size=block_size,
            max_tokens=4096,
            selection_budget_tokens=budget,
        )
    )


class TestApplyGating(unittest.TestCase):
    def tearDown(self):
        reset_kvmem_hook()
        reset_apply_state()

    def test_disabled_by_default(self):
        _enable()
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("SGLANG_KVMEM_APPLY", None)
            self.assertFalse(apply_enabled())
            backend = FakeBackend([[1, 2, 3, 4]])
            self.assertIsNone(
                maybe_apply_kvmem_selection(FakeForwardBatch(), backend)
            )
            self.assertEqual(backend.forward_metadata.kv_indices.tolist(), [1, 2, 3, 4])

    def test_no_selection_leaves_indices_untouched(self):
        _enable()
        with mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"}):
            backend = FakeBackend([[1, 2, 3, 4]])
            self.assertIsNone(
                maybe_apply_kvmem_selection(FakeForwardBatch(), backend)
            )
            self.assertEqual(backend.forward_metadata.kv_indices.tolist(), [1, 2, 3, 4])

    def test_unsupported_mode_is_skipped(self):
        _enable()
        get_kvmem_registry().record_selection("slot:0", [0], budget_limited=True)
        with mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"}):
            backend = FakeBackend([[1, 2, 3, 4]])
            self.assertIsNone(
                maybe_apply_kvmem_selection(
                    FakeForwardBatch(extend=False), backend
                )
            )
            self.assertEqual(backend.forward_metadata.kv_indices.tolist(), [1, 2, 3, 4])

    def test_zero_history_is_skipped(self):
        # Regression: a request whose KV window is still empty (first prefill
        # chunk) used to reach the keep computation and raise.
        _enable()
        get_kvmem_registry().record_selection("slot:0", [0], budget_limited=True)
        backend = FakeBackend([[]])
        with mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"}):
            self.assertIsNone(
                maybe_apply_kvmem_selection(FakeForwardBatch(), backend)
            )

    def test_missing_backend_is_skipped(self):
        _enable()
        get_kvmem_registry().record_selection("slot:0", [0], budget_limited=True)
        with mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"}):
            self.assertIsNone(maybe_apply_kvmem_selection(FakeForwardBatch(), None))


class TestApplyRewrite(unittest.TestCase):
    def tearDown(self):
        reset_kvmem_hook()
        reset_apply_state()

    def test_keeping_every_block_is_identity(self):
        _enable(block_size=4, budget=4096)
        registry = get_kvmem_registry()
        registry.record_selection("slot:0", [0, 1], budget_limited=True)
        backend = FakeBackend([[10, 11, 12, 13, 14, 15, 16, 17]])
        with mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"}):
            result = maybe_apply_kvmem_selection(FakeForwardBatch(), backend)

        self.assertEqual(result, (8, 8))
        self.assertEqual(
            backend.forward_metadata.kv_indices.tolist(),
            [10, 11, 12, 13, 14, 15, 16, 17],
        )
        self.assertEqual(backend.forward_metadata.kv_indptr.tolist(), [0, 8])

    def test_drops_unselected_blocks_and_recomputes_indptr(self):
        # 4 blocks of 4 tokens; selection keeps only block 0; sink+recent keep
        # blocks 0 and 3, so blocks 1 and 2 are dropped.
        _enable(block_size=4, budget=8)
        registry = get_kvmem_registry()
        registry.record_selection("slot:0", [0], budget_limited=True)
        backend = FakeBackend([[10, 11, 12, 13, 14, 15, 16, 17]])
        with mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"}):
            result = maybe_apply_kvmem_selection(FakeForwardBatch(), backend)

        self.assertEqual(result[0], 8)  # sink 4 + recent 4
        self.assertEqual(
            backend.forward_metadata.kv_indices.tolist(), [10, 11, 12, 13, 14, 15, 16, 17]
        )

    def test_two_requests_shrink_independently(self):
        _enable(block_size=4, budget=4)
        registry = get_kvmem_registry()
        registry.record_selection("slot:0", [0], budget_limited=True)
        # slot 1 has no selection -> untouched.
        backend = FakeBackend(
            [[10, 11, 12, 13, 14, 15, 16, 17], [20, 21, 22, 23]]
        )
        with mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"}):
            result = maybe_apply_kvmem_selection(
                FakeForwardBatch(slots=(0, 1)), backend
            )

        self.assertEqual(result[1], 12)  # 8 + 4 history tokens
        self.assertEqual(
            backend.forward_metadata.kv_indices.tolist(),
            [10, 11, 12, 13, 14, 15, 16, 17, 20, 21, 22, 23],
        )
        self.assertEqual(backend.forward_metadata.kv_indptr.tolist(), [0, 8, 12])

    def test_long_history_is_actually_cut(self):
        # 16 blocks of 4; only block 5 selected plus sink + 2 recent blocks.
        _enable(block_size=4, budget=4)
        registry = get_kvmem_registry()
        registry.record_selection("slot:0", [5], budget_limited=True)
        slots = list(range(100, 164))
        backend = FakeBackend([slots])
        with mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"}):
            result = maybe_apply_kvmem_selection(FakeForwardBatch(), backend)

        kept, history = result
        self.assertEqual(history, 64)
        self.assertEqual(kept, 16)  # sink + block 5 + 2 recent blocks
        self.assertEqual(len(backend.forward_metadata.kv_indices), 16)

    def test_non_budget_limited_selection_is_not_applied(self):
        # A selection that was not cut by the budget wanted everything it scored,
        # so it must not drop blocks it never saw.
        _enable(block_size=4, budget=4096)
        registry = get_kvmem_registry()
        registry.record_selection("slot:0", [0], budget_limited=False)
        backend = FakeBackend([[10, 11, 12, 13, 14, 15, 16, 17]])
        with mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"}):
            self.assertIsNone(
                maybe_apply_kvmem_selection(FakeForwardBatch(), backend)
            )
            self.assertEqual(
                backend.forward_metadata.kv_indices.tolist(),
                [10, 11, 12, 13, 14, 15, 16, 17],
            )

    def test_shorter_indptr_than_batch_is_skipped(self):
        _enable()
        registry = get_kvmem_registry()
        registry.record_selection("slot:0", [0], budget_limited=True)
        registry.record_selection("slot:1", [0], budget_limited=True)
        backend = FakeBackend([[10, 11]])
        with mock.patch.dict(os.environ, {"SGLANG_KVMEM_APPLY": "1"}):
            self.assertIsNone(
                maybe_apply_kvmem_selection(
                    FakeForwardBatch(slots=(0, 1)), backend
                )
            )


if __name__ == "__main__":
    unittest.main()
