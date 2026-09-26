import unittest

import torch

from sglang.srt.mem_cache.kvmem_capture_adapter import KVMemCaptureBatch
from sglang.srt.mem_cache.kvmem_registry import (
    KVMemRequestRegistry,
    KVMemRegistryConfig,
)


class TestKVMemRequestRegistry(unittest.TestCase):
    def config(self):
        return KVMemRegistryConfig(
            block_size=2,
            max_tokens=8,
            selection_budget_tokens=4,
            kv_heads=1,
            head_dim=1,
        )

    def batch(self):
        return KVMemCaptureBatch(
            key=torch.tensor([[[1.0]], [[1.0]], [[5.0]]]),
            positions=torch.tensor([0, 1, 0]),
            request_ids=["a", "a", "b"],
        )

    def test_routes_rows_to_isolated_request_controllers(self):
        registry = KVMemRequestRegistry(self.config())
        registry.capture_batch(layer_id=0, batch=self.batch())
        registry.commit_batch(layer_id=0, accepted={"a": 2, "b": 1})

        means_a, _ = registry.controller("a").state.snapshot_layer(0)
        means_b, _ = registry.controller("b").state.snapshot_layer(0)
        torch.testing.assert_close(means_a, torch.tensor([[[1.0]]]))
        torch.testing.assert_close(means_b, torch.tensor([[[5.0]]]))

    def test_commit_per_request_accepted_tokens(self):
        registry = KVMemRequestRegistry(self.config())
        registry.capture_batch(layer_id=0, batch=self.batch())

        registry.commit_batch(layer_id=0, accepted={"a": 1, "b": 1})

        _, counts_a = registry.controller("a").state.snapshot_layer(0)
        _, counts_b = registry.controller("b").state.snapshot_layer(0)
        torch.testing.assert_close(counts_a, torch.tensor([1]))
        torch.testing.assert_close(counts_b, torch.tensor([1]))

    def test_rollback_discards_all_staged_batches(self):
        registry = KVMemRequestRegistry(self.config())
        registry.capture_batch(layer_id=0, batch=self.batch())

        registry.rollback_batch(layer_id=0)

        _, counts_a = registry.controller("a").state.snapshot_layer(0)
        _, counts_b = registry.controller("b").state.snapshot_layer(0)
        self.assertEqual(tuple(counts_a.shape), (0,))
        self.assertEqual(tuple(counts_b.shape), (0,))

    def test_commit_requires_active_capture(self):
        registry = KVMemRequestRegistry(self.config())
        with self.assertRaises(RuntimeError):
            registry.commit_batch(layer_id=0, accepted={"a": 1})


    def test_selection_record_round_trip(self):
        registry = KVMemRequestRegistry(
            KVMemRegistryConfig(
                block_size=4, max_tokens=64, selection_budget_tokens=16
            )
        )
        self.assertIsNone(registry.last_selection("slot:0"))
        registry.record_selection("slot:0", [1, 2], budget_limited=True)
        self.assertEqual(registry.last_selection("slot:0"), ([1, 2], True))
        registry.clear_selection("slot:0")
        self.assertIsNone(registry.last_selection("slot:0"))

    def test_remove_request_drops_selection(self):
        registry = KVMemRequestRegistry(
            KVMemRegistryConfig(
                block_size=4, max_tokens=64, selection_budget_tokens=16
            )
        )
        registry.record_selection("slot:0", [1], budget_limited=False)
        registry.remove_request("slot:0")
        self.assertIsNone(registry.last_selection("slot:0"))

    def test_clear_drops_selections(self):
        registry = KVMemRequestRegistry(
            KVMemRegistryConfig(
                block_size=4, max_tokens=64, selection_budget_tokens=16
            )
        )
        registry.record_selection("slot:0", [1], budget_limited=True)
        registry.clear()
        self.assertIsNone(registry.last_selection("slot:0"))


if __name__ == "__main__":
    unittest.main()
