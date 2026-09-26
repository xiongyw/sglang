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
    def __init__(self, decode=False, extend=False):
        self._decode = decode
        self._extend = extend

    def is_decode(self):
        return self._decode

    def is_extend(self):
        return self._extend


class FakeForwardBatch:
    def __init__(
        self,
        rids,
        seq_lens,
        positions,
        extend_start_loc=None,
        extend_seq_lens_cpu=None,
        mode=None,
    ):
        self.rids = rids
        self.seq_lens = seq_lens
        self.positions = positions
        self.extend_start_loc = extend_start_loc
        self.extend_seq_lens_cpu = extend_seq_lens_cpu
        self.forward_mode = mode or FakeMode(decode=True)


class TestKVMemHook(unittest.TestCase):
    def tearDown(self):
        reset_kvmem_hook()

    def test_disabled_hook_is_noop(self):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=False,
                block_size=2,
                max_tokens=8,
                selection_budget_tokens=4,
            )
        )

        result = capture_attention_k(
            layer_id=0,
            key=torch.ones((2, 1, 1)),
            positions=torch.tensor([0, 1]),
            request_ids=["a", "a"],
        )

        self.assertFalse(result)
        reset_kvmem_hook()
        self.assertFalse(
            capture_attention_k(
                layer_id=0,
                key=torch.ones((2, 1, 1)),
                positions=torch.tensor([0, 1]),
                request_ids=["a", "a"],
            )
        )

    def test_enabled_hook_stages_and_commit_persists(self):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=True,
                block_size=2,
                max_tokens=8,
                selection_budget_tokens=4,
                kv_heads=1,
                head_dim=1,
            )
        )

        staged = capture_attention_k(
            layer_id=0,
            key=torch.tensor([[[1.0]], [[1.0]], [[5.0]]]),
            positions=torch.tensor([0, 1, 0]),
            request_ids=["a", "a", "b"],
        )
        self.assertTrue(staged)

        committed = commit_attention_k(layer_id=0, accepted={"a": 2, "b": 1})
        self.assertTrue(committed)

        from sglang.srt.mem_cache.kvmem_hook import get_kvmem_registry

        registry = get_kvmem_registry()
        means_a, _ = registry.controller("a").state.snapshot_layer(0)
        means_b, _ = registry.controller("b").state.snapshot_layer(0)
        torch.testing.assert_close(means_a, torch.tensor([[[1.0]]]))
        torch.testing.assert_close(means_b, torch.tensor([[[5.0]]]))

    def test_enabled_hook_rollback_discards(self):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=True,
                block_size=2,
                max_tokens=8,
                selection_budget_tokens=4,
            )
        )
        capture_attention_k(
            layer_id=0,
            key=torch.ones((2, 1, 1)),
            positions=torch.tensor([0, 1]),
            request_ids=["a", "a"],
        )

        rollback_attention_k(layer_id=0)

        from sglang.srt.mem_cache.kvmem_hook import get_kvmem_registry

        _, counts = get_kvmem_registry().controller("a").state.snapshot_layer(0)
        self.assertEqual(tuple(counts.shape), (0,))


class TestPerRowRequestIds(unittest.TestCase):
    def test_decode_exact_mapping(self):
        batch = FakeForwardBatch(
            rids=["a", "b"],
            seq_lens=torch.tensor([5, 7]),
            positions=torch.tensor([4, 6]),
        )
        self.assertEqual(_per_row_request_ids(batch), ["a", "b"])

    def test_decode_count_mismatch_returns_none(self):
        batch = FakeForwardBatch(
            rids=["a"],
            seq_lens=torch.tensor([5, 7]),
            positions=torch.tensor([4, 6]),
        )
        self.assertIsNone(_per_row_request_ids(batch))

    def test_extend_expands_rows(self):
        batch = FakeForwardBatch(
            rids=["a", "b"],
            seq_lens=torch.tensor([9, 12]),
            positions=torch.arange(7),
            extend_start_loc=torch.tensor([0, 3]),
            extend_seq_lens_cpu=[3, 4],
            mode=FakeMode(extend=True),
        )
        self.assertEqual(
            _per_row_request_ids(batch),
            ["a", "a", "a", "b", "b", "b", "b"],
        )

    def test_missing_rids_returns_none(self):
        batch = FakeForwardBatch(
            rids=None,
            seq_lens=torch.tensor([5]),
            positions=torch.tensor([4]),
        )
        self.assertIsNone(_per_row_request_ids(batch))


class TestMaybeCapture(unittest.TestCase):
    def tearDown(self):
        reset_kvmem_hook()

    def test_capture_disabled_is_noop(self):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=False,
                block_size=2,
                max_tokens=8,
                selection_budget_tokens=4,
            )
        )
        batch = FakeForwardBatch(
            rids=["a"],
            seq_lens=torch.tensor([5]),
            positions=torch.tensor([4]),
        )
        self.assertFalse(
            maybe_capture_self_attention(
                layer_id=0, key=torch.ones((1, 1, 1)), forward_batch=batch
            )
        )

    def test_capture_decode_stages_and_commits(self):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=True,
                block_size=2,
                max_tokens=8,
                selection_budget_tokens=4,
                kv_heads=1,
                head_dim=1,
            )
        )
        batch = FakeForwardBatch(
            rids=["a", "a"],
            seq_lens=torch.tensor([5, 5]),
            positions=torch.tensor([3, 4]),
        )
        key = torch.tensor([[[1.0]], [[3.0]]])
        self.assertTrue(
            maybe_capture_self_attention(
                layer_id=0, key=key, forward_batch=batch
            )
        )

        commit_attention_k(layer_id=0, accepted={"a": 2})

        from sglang.srt.mem_cache.kvmem_hook import get_kvmem_registry

        means, _ = get_kvmem_registry().controller("a").state.snapshot_layer(0)
        # positions 3 and 4 land in blocks 1 and 2 (block_size=2)
        torch.testing.assert_close(means[1], torch.tensor([[1.0]]))
        torch.testing.assert_close(means[2], torch.tensor([[3.0]]))
        _, counts = get_kvmem_registry().controller("a").state.snapshot_layer(0)
        torch.testing.assert_close(counts, torch.tensor([0, 1, 1]))

    def test_key_row_mismatch_skips(self):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=True,
                block_size=2,
                max_tokens=8,
                selection_budget_tokens=4,
            )
        )
        batch = FakeForwardBatch(
            rids=["a", "a"],
            seq_lens=torch.tensor([5, 5]),
            positions=torch.tensor([3, 4]),
        )
        self.assertFalse(
            maybe_capture_self_attention(
                layer_id=0, key=torch.ones((1, 1, 1)), forward_batch=batch
            )
        )


if __name__ == "__main__":
    unittest.main()
