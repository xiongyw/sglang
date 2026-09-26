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

    def test_reusing_position_in_one_layer_is_rejected(self):
        state = KVMemRequestState(block_size=2, max_tokens=8)
        state.update_layer(3, torch.ones((1, 1, 1)), torch.tensor([0]))

        with self.assertRaises(ValueError):
            state.update_layer(3, torch.ones((1, 1, 1)), torch.tensor([0]))


if __name__ == "__main__":
    unittest.main()
