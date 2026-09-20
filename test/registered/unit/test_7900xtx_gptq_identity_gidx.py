from __future__ import annotations

import unittest

import torch

from sglang.srt.hardware_backend.gpu.quantization.gptq_rdna3 import (
    _is_identity_group_index,
)


class Test7900XtxIdentityGroupIndex(unittest.TestCase):
    def test_full_k_identity_is_accepted(self):
        n, group_size = 5120, 64
        g_idx = torch.arange(n, dtype=torch.int32) // group_size
        self.assertTrue(_is_identity_group_index(g_idx, group_size))

    def test_tp_k_shard_with_group_offset_is_accepted(self):
        # TP=2 rank 1 slice of a 5120-wide identity: local groups 40..79.
        n, group_size, offset = 2560, 64, 40
        g_idx = torch.arange(n, dtype=torch.int32) // group_size + offset
        self.assertTrue(_is_identity_group_index(g_idx, group_size))

    def test_empty_group_index_is_accepted(self):
        self.assertTrue(
            _is_identity_group_index(torch.empty(0, dtype=torch.int32), 64)
        )

    def test_act_order_reordering_is_rejected(self):
        n, group_size = 128, 64
        g_idx = (torch.arange(n, dtype=torch.int32) // group_size).flip(0)
        self.assertFalse(_is_identity_group_index(g_idx, group_size))

    def test_compressed_group_blocks_are_rejected(self):
        n, group_size = 128, 64
        g_idx = torch.zeros(n, dtype=torch.int32)
        self.assertFalse(_is_identity_group_index(g_idx, group_size))


if __name__ == "__main__":
    unittest.main()
