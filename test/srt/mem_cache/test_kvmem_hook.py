import unittest

import torch

from sglang.srt.mem_cache.kvmem_hook import (
    KVMemHookConfig,
    capture_attention_k,
    reset_kvmem_hook,
    set_kvmem_hook_config,
)


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


def commit_attention_k(layer_id, accepted):
    from sglang.srt.mem_cache.kvmem_hook import commit_attention_k as impl

    return impl(layer_id=layer_id, accepted=accepted)


def rollback_attention_k(layer_id):
    from sglang.srt.mem_cache.kvmem_hook import rollback_attention_k as impl

    return impl(layer_id=layer_id)


if __name__ == "__main__":
    unittest.main()
