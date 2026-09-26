import unittest
from unittest import mock

import torch

from sglang.srt.mem_cache.kvmem_hook import (
    KVMemHookConfig,
    capture_attention_k,
    get_kvmem_registry,
    hook_enabled,
    reset_kvmem_hook,
    set_kvmem_hook_config,
)
from sglang.srt.mem_cache.kvmem_scheduler_hook import (
    finish_kvmem_forward,
    maybe_enable_kvmem_hook,
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

    def is_target_verify(self):
        return self._verify

    def is_draft_extend(self):
        return False

    def is_idle(self):
        return False


class FakeForwardBatch:
    """ForwardBatch stand-in: one request on one slot."""

    def __init__(self, mode=None, slot=0, rows=8):
        self.forward_mode = mode or FakeMode()
        self.req_pool_indices = torch.tensor([slot], dtype=torch.long)
        self.positions = torch.arange(rows, dtype=torch.long)
        self.extend_seq_lens_cpu = [rows]


class TestKVMemSchedulerHook(unittest.TestCase):
    def tearDown(self):
        reset_kvmem_hook()

    def _enable(self, kv_heads=1, head_dim=1, block_size=2):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=True,
                block_size=block_size,
                max_tokens=4096,
                selection_budget_tokens=16,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
        )

    def _stage(self, layer, key_rows, positions, request_id="slot:0", value=1.0):
        capture_attention_k(
            layer_id=layer,
            key=torch.full((key_rows, 1, 1), float(value)),
            positions=torch.tensor(positions, dtype=torch.long),
            request_ids=[request_id] * key_rows,
        )

    def test_maybe_enable_initializes_from_env_off(self):
        with mock.patch.dict("os.environ", {"SGLANG_KVMEM_ENABLED": "0"}):
            maybe_enable_kvmem_hook()
        self.assertFalse(hook_enabled())

    def test_maybe_enable_initializes_from_env_on(self):
        with mock.patch.dict(
            "os.environ",
            {
                "SGLANG_KVMEM_ENABLED": "1",
                "SGLANG_KVMEM_BLOCK_SIZE": "4",
                "SGLANG_KVMEM_MAX_TOKENS": "64",
                "SGLANG_KVMEM_BUDGET_TOKENS": "16",
                "SGLANG_KVMEM_KV_HEADS": "1",
                "SGLANG_KVMEM_HEAD_DIM": "1",
            },
        ):
            maybe_enable_kvmem_hook()
        self.assertTrue(hook_enabled())

    def test_extend_stores_all_staged_rows(self):
        self._enable()
        for layer in (0, 4):
            self._stage(layer, 2, [0, 1])
        finish_kvmem_forward(FakeForwardBatch(rows=2))

        for layer in (0, 4):
            _, counts = (
                get_kvmem_registry().controller("slot:0").state.snapshot_layer(layer)
            )
            torch.testing.assert_close(counts, torch.tensor([2]))

    def test_verify_stores_rows_without_acceptance_input(self):
        self._enable()
        self._stage(0, 8, list(range(8)))
        finish_kvmem_forward(FakeForwardBatch(mode=FakeMode(verify=True), rows=8))

        _, counts = (
            get_kvmem_registry().controller("slot:0").state.snapshot_layer(0)
        )
        self.assertEqual(int(counts.sum()), 8)

    def test_restaged_positions_are_not_double_counted(self):
        self._enable()
        self._stage(0, 4, [0, 1, 2, 3])
        finish_kvmem_forward(FakeForwardBatch(rows=4))

        # The next verify step re-stages an overlapping window.
        self._stage(0, 4, [2, 3, 4, 5], value=2.0)
        finish_kvmem_forward(FakeForwardBatch(mode=FakeMode(verify=True), rows=4))

        means, counts = (
            get_kvmem_registry().controller("slot:0").state.snapshot_layer(0)
        )
        # block_size=2: blocks are [0,1], [2,3], [4,5].
        self.assertEqual(int(counts.sum()), 6)
        self.assertEqual(int(counts[0]), 2)
        self.assertEqual(int(counts[1]), 2)
        self.assertEqual(int(counts[2]), 2)
        # First sighting wins for the overlapping positions.
        self.assertAlmostEqual(float(means[0][0][0]), 1.0)

    def test_slot_reuse_restart_at_zero_resets_state(self):
        self._enable()
        self._stage(0, 4, [0, 1, 2, 3])
        finish_kvmem_forward(FakeForwardBatch(rows=4))

        self._stage(0, 4, [0, 1, 2, 3], value=7.0)
        finish_kvmem_forward(FakeForwardBatch(rows=4))

        means, counts = (
            get_kvmem_registry().controller("slot:0").state.snapshot_layer(0)
        )
        self.assertEqual(int(counts.sum()), 4)
        self.assertAlmostEqual(float(means[0][0][0]), 7.0)

    def test_unknown_mode_rolls_back(self):
        self._enable()
        self._stage(0, 1, [2])
        finish_kvmem_forward(
            FakeForwardBatch(mode=FakeMode(extend=False, decode=False, verify=False))
        )

        _, counts = (
            get_kvmem_registry().controller("slot:0").state.snapshot_layer(0)
        )
        self.assertEqual(tuple(counts.shape), (0,))

    def test_disabled_is_noop(self):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=False,
                block_size=2,
                max_tokens=8,
                selection_budget_tokens=4,
            )
        )
        finish_kvmem_forward(FakeForwardBatch())
        self.assertFalse(hook_enabled())


if __name__ == "__main__":
    unittest.main()
