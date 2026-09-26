import unittest

import torch

from sglang.srt.mem_cache.kvmem_hook import (
    KVMemHookConfig,
    _per_row_request_ids,
    capture_attention_k,
    commit_attention_k,
    maybe_capture_self_attention,
    reset_kvmem_hook,
    rollback_attention_k,
    set_kvmem_hook_config,
)


class FakeMode:
    def __init__(self, decode=False, extend=False, verify=False):
        self._decode = decode
        self._extend = extend
        self._verify = verify

    def is_decode(self):
        return self._decode

    def is_extend(self):
        return self._extend or self._verify

    def is_target_verify(self):
        return self._verify


class FakeForwardBatch:
    """Minimal stand-in for ForwardBatch rows/requests metadata."""

    def __init__(
        self,
        req_pool_indices,
        positions,
        seq_lens=None,
        extend_seq_lens_cpu=None,
        mode=None,
    ):
        self.req_pool_indices = torch.tensor(req_pool_indices, dtype=torch.long)
        self.positions = torch.tensor(positions, dtype=torch.long)
        self.seq_lens = torch.tensor(
            seq_lens if seq_lens is not None else [len(positions)],
            dtype=torch.long,
        )
        self.extend_seq_lens_cpu = extend_seq_lens_cpu
        self.forward_mode = mode or FakeMode(decode=True)


class TestPerRowRequestIds(unittest.TestCase):
    def test_decode_keys_by_pool_slot(self):
        batch = FakeForwardBatch([3, 5], [4, 6])
        self.assertEqual(_per_row_request_ids(batch), ["slot:3", "slot:5"])

    def test_verify_repeats_slot_for_each_draft_row(self):
        batch = FakeForwardBatch([1], list(range(8)), mode=FakeMode(verify=True))
        self.assertEqual(_per_row_request_ids(batch), ["slot:1"] * 8)

    def test_extend_uses_host_row_counts(self):
        batch = FakeForwardBatch(
            [0, 1],
            list(range(7)),
            extend_seq_lens_cpu=[3, 4],
            mode=FakeMode(extend=True),
        )
        self.assertEqual(
            _per_row_request_ids(batch),
            ["slot:0"] * 3 + ["slot:1"] * 4,
        )

    def test_uneven_verify_rows_are_rejected(self):
        batch = FakeForwardBatch([0, 1], list(range(7)), mode=FakeMode(verify=True))
        self.assertIsNone(_per_row_request_ids(batch))

    def test_extend_count_mismatch_is_rejected(self):
        batch = FakeForwardBatch(
            [0, 1],
            list(range(7)),
            extend_seq_lens_cpu=[3, 5],
            mode=FakeMode(extend=True),
        )
        self.assertIsNone(_per_row_request_ids(batch))

    def test_empty_batch_returns_none(self):
        batch = FakeForwardBatch([], [])
        self.assertIsNone(_per_row_request_ids(batch))


class TestMaybeCapture(unittest.TestCase):
    def tearDown(self):
        reset_kvmem_hook()

    def _enable(self):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=True,
                block_size=2,
                max_tokens=4096,
                selection_budget_tokens=16,
                kv_heads=2,
                head_dim=4,
            )
        )

    def test_disabled_is_noop(self):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=False,
                block_size=2,
                max_tokens=16,
                selection_budget_tokens=4,
            )
        )
        batch = FakeForwardBatch([0], [0])
        self.assertFalse(
            maybe_capture_self_attention(
                layer_id=0,
                key=torch.ones((1, 8)),
                forward_batch=batch,
                kv_heads=2,
                head_dim=4,
            )
        )

    def test_2d_key_is_reshaped_with_layer_geometry(self):
        self._enable()
        batch = FakeForwardBatch([2], [0, 1], mode=FakeMode(verify=True))
        key = torch.arange(16, dtype=torch.float32).reshape(2, 8)
        self.assertTrue(
            maybe_capture_self_attention(
                layer_id=0,
                key=key,
                forward_batch=batch,
                kv_heads=2,
                head_dim=4,
            )
        )
        commit_attention_k(layer_id=0, accepted={"slot:2": 2})

        from sglang.srt.mem_cache.kvmem_hook import get_kvmem_registry

        means, _ = get_kvmem_registry().controller("slot:2").state.snapshot_layer(0)
        self.assertEqual(tuple(means.shape), (1, 2, 4))

    def test_geometry_mismatch_skips(self):
        self._enable()
        batch = FakeForwardBatch([0], [0])
        self.assertFalse(
            maybe_capture_self_attention(
                layer_id=0,
                key=torch.ones((1, 7)),
                forward_batch=batch,
                kv_heads=2,
                head_dim=4,
            )
        )

    def test_3d_key_is_accepted(self):
        self._enable()
        batch = FakeForwardBatch([0], [0])
        self.assertTrue(
            maybe_capture_self_attention(
                layer_id=0,
                key=torch.ones((1, 2, 4)),
                forward_batch=batch,
                kv_heads=2,
                head_dim=4,
            )
        )


class TestCommitLifecycle(unittest.TestCase):
    def tearDown(self):
        reset_kvmem_hook()

    def _enable(self, kv_heads=2, head_dim=4):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=True,
                block_size=2,
                max_tokens=16,
                selection_budget_tokens=4,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
        )

    def test_rollback_clears_staged_rows(self):
        self._enable()
        capture_attention_k(
            layer_id=0,
            key=torch.ones((2, 2, 4)),
            positions=torch.tensor([0, 1]),
            request_ids=["slot:0", "slot:0"],
        )
        self.assertTrue(rollback_attention_k(layer_id=0))

        from sglang.srt.mem_cache.kvmem_hook import get_kvmem_registry

        _, counts = get_kvmem_registry().controller("slot:0").state.snapshot_layer(0)
        self.assertEqual(tuple(counts.shape), (0,))

    def test_slot_reuse_resets_stale_state(self):
        self._enable(kv_heads=1, head_dim=1)
        capture_attention_k(
            layer_id=0,
            key=torch.ones((2, 1, 1)),
            positions=torch.tensor([0, 1]),
            request_ids=["slot:7", "slot:7"],
        )
        commit_attention_k(layer_id=0, accepted={"slot:7": 2})

        # A new request lands in the same slot and re-stages position 0.
        capture_attention_k(
            layer_id=0,
            key=torch.full((1, 1, 1), 9.0),
            positions=torch.tensor([0]),
            request_ids=["slot:7"],
        )
        commit_attention_k(layer_id=0, accepted={"slot:7": 1})

        from sglang.srt.mem_cache.kvmem_hook import get_kvmem_registry

        means, counts = get_kvmem_registry().controller("slot:7").state.snapshot_layer(0)
        torch.testing.assert_close(means, torch.tensor([[[9.0]]]))
        torch.testing.assert_close(counts, torch.tensor([1]))


if __name__ == "__main__":
    unittest.main()
