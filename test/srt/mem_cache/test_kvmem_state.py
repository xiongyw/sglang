import unittest

import torch

from sglang.srt.mem_cache.kvmem_state import KVMemRequestState


class TestKVMemRequestState(unittest.TestCase):
    def test_layers_are_isolated(self):
        state = KVMemRequestState(block_size=2, max_tokens=8)
        state.update_layer(
            layer_id=3,
            key=torch.tensor([[[1.0]]]),
            positions=torch.tensor([0]),
        )
        state.update_layer(
            layer_id=7,
            key=torch.tensor([[[5.0]]]),
            positions=torch.tensor([0]),
        )

        means3, counts3 = state.snapshot_layer(3)
        means7, counts7 = state.snapshot_layer(7)

        torch.testing.assert_close(means3, torch.tensor([[[1.0]]]))
        torch.testing.assert_close(means7, torch.tensor([[[5.0]]]))
        torch.testing.assert_close(counts3, torch.tensor([1]))
        torch.testing.assert_close(counts7, torch.tensor([1]))

    def test_clear_request_removes_all_layers(self):
        state = KVMemRequestState(block_size=2, max_tokens=8)
        state.update_layer(3, torch.ones((1, 1, 1)), torch.tensor([0]))
        state.clear()

        self.assertEqual(state.layer_ids, [])

    def test_re_staging_a_position_is_idempotent(self):
        state = KVMemRequestState(block_size=2, max_tokens=8)
        self.assertEqual(
            state.update_layer(3, torch.ones((1, 1, 1)), torch.tensor([0])), 1
        )
        # The same logical position again: stored once, so nothing new lands.
        self.assertEqual(
            state.update_layer(3, torch.full((1, 1, 1), 9.0), torch.tensor([0])), 0
        )
        means, counts = state.snapshot_layer(3)
        torch.testing.assert_close(counts, torch.tensor([1]))
        self.assertAlmostEqual(float(means[0][0][0]), 1.0)


    def test_device_argument_selects_device_accumulator(self):
        from sglang.srt.mem_cache.kvmem_mean_k import (
            KVMemDeviceMeanKAccumulator,
            KVMemMeanKAccumulator,
        )

        state = KVMemRequestState(block_size=2, max_tokens=8)
        host = state.get_or_create_layer(3, 1, 1)
        self.assertIsInstance(host, KVMemMeanKAccumulator)
        # A device other than cpu selects the device-resident twin; meta keeps
        # the test GPU-free while still exercising the branch.
        state2 = KVMemRequestState(block_size=2, max_tokens=8)
        device_acc = state2.get_or_create_layer(
            3, 1, 1, device=torch.device("meta")
        )
        self.assertIsInstance(device_acc, KVMemDeviceMeanKAccumulator)

    def test_geometry_change_rebuilds_the_accumulator(self):
        state = KVMemRequestState(block_size=2, max_tokens=8)
        first = state.get_or_create_layer(3, 1, 1)
        second = state.get_or_create_layer(3, 2, 4)
        self.assertIsNot(first, second)
        self.assertEqual(second.kv_heads, 2)
        self.assertEqual(second.head_dim, 4)
        self.assertIs(state.get_or_create_layer(3, 2, 4), second)


if __name__ == "__main__":
    unittest.main()
