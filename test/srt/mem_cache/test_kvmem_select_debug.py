import unittest

import torch

from sglang.srt.mem_cache.kvmem_hook import (
    KVMemHookConfig,
    capture_attention_k,
    commit_attention_k,
    reset_kvmem_hook,
    set_kvmem_hook_config,
)
from sglang.srt.mem_cache.kvmem_select_debug import select_debug


class FakeMode:
    def __init__(self, verify=True):
        self._verify = verify

    def is_decode(self):
        return not self._verify

    def is_extend(self):
        return self._verify

    def is_target_verify(self):
        return self._verify


class FakeForwardBatch:
    def __init__(self, slot=0, seq_len=8, rows=8):
        self.forward_mode = FakeMode()
        self.req_pool_indices = torch.tensor([slot], dtype=torch.long)
        self.seq_lens = torch.tensor([seq_len], dtype=torch.long)
        self.positions = torch.arange(rows, dtype=torch.long)


class TestSelectDebug(unittest.TestCase):
    def tearDown(self):
        reset_kvmem_hook()

    def _enable(self, kv_heads=1, head_dim=2, block_size=2, budget=4):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=True,
                block_size=block_size,
                max_tokens=4096,
                selection_budget_tokens=budget,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
        )

    def _capture_blocks(self, slot=0, keys=None, positions=None):
        keys = keys or [[[1.0, 1.0]], [[1.0, 1.0]], [[-1.0, -1.0]], [[-1.0, -1.0]]]
        positions = positions or [0, 1, 2, 3]
        n = len(positions)
        capture_attention_k(
            layer_id=3,
            key=torch.tensor(keys, dtype=torch.float32),
            positions=torch.tensor(positions, dtype=torch.long),
            request_ids=[f"slot:{slot}"] * n,
        )
        commit_attention_k(layer_id=3, accepted={f"slot:{slot}": n})

    def test_returns_none_when_disabled(self):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=False,
                block_size=2,
                max_tokens=64,
                selection_budget_tokens=4,
            )
        )
        self.assertIsNone(
            select_debug(
                layer_id=3, query=torch.ones((1, 1, 2)), forward_batch=FakeForwardBatch()
            )
        )

    def test_selects_blocks_ranked_by_query_similarity(self):
        self._enable(kv_heads=1, head_dim=2, block_size=2, budget=4)
        self._capture_blocks()

        result = select_debug(
            layer_id=3,
            query=torch.tensor([[[1.0, 1.0]]]),
            forward_batch=FakeForwardBatch(),
        )

        self.assertIsNotNone(result)
        self.assertEqual(len(result), 1)
        entry = result[0]
        self.assertEqual(entry.request_key, "slot:0")
        self.assertEqual(entry.n_blocks, 2)
        # Query matches block 0; block 0 must outrank block 1.
        self.assertIn(0, entry.selected_block_ids)
        self.assertGreater(entry.scores[0], entry.scores[1])
        self.assertLessEqual(entry.selected_tokens, 4)

    def test_returns_none_for_unknown_request(self):
        self._enable()
        result = select_debug(
            layer_id=3,
            query=torch.ones((1, 1, 2)),
            forward_batch=FakeForwardBatch(slot=9),
        )
        self.assertIsNone(result)

    def test_returns_none_without_captured_state(self):
        self._enable()
        result = select_debug(
            layer_id=3, query=torch.ones((1, 1, 2)), forward_batch=FakeForwardBatch()
        )
        self.assertIsNone(result)


if __name__ == "__main__":
    unittest.main()
