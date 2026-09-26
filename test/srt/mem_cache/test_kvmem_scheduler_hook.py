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
    _last_seq_lens,
    finish_kvmem_forward,
    maybe_enable_kvmem_hook,
)


class FakeMode:
    def __init__(self, mode):
        self._mode = mode

    def is_decode(self):
        return self._mode == "decode"

    def is_extend(self):
        return self._mode in ("extend", "target_verify", "mixed")

    def is_target_verify(self):
        return self._mode == "target_verify"


class FakeForwardBatch:
    """ForwardBatch stand-in: one request, configurable context length."""

    def __init__(self, mode="target_verify", slot=0, seq_len=100, rows=8):
        self.forward_mode = FakeMode(mode)
        self.req_pool_indices = torch.tensor([slot], dtype=torch.long)
        self.seq_lens = torch.tensor([seq_len], dtype=torch.long)
        self.positions = torch.arange(rows, dtype=torch.long)
        self.extend_seq_lens_cpu = [rows] if mode == "extend" else None


class TestKVMemSchedulerHook(unittest.TestCase):
    def setUp(self):
        _last_seq_lens.clear()

    def tearDown(self):
        reset_kvmem_hook()
        _last_seq_lens.clear()

    def _enable(self, kv_heads=1, head_dim=1):
        set_kvmem_hook_config(
            KVMemHookConfig(
                enabled=True,
                block_size=2,
                max_tokens=4096,
                selection_budget_tokens=16,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
        )

    def _stage(self, layer, key_rows, positions, request_id="slot:0"):
        capture_attention_k(
            layer_id=layer,
            key=torch.ones((key_rows, 1, 1)),
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

    def test_extend_commits_all_staged_rows(self):
        self._enable()
        for layer in (0, 4):
            self._stage(layer, 2, [0, 1])
        finish_kvmem_forward(FakeForwardBatch(mode="extend", rows=2))

        for layer in (0, 4):
            _, counts = get_kvmem_registry().controller("slot:0").state.snapshot_layer(layer)
            torch.testing.assert_close(counts, torch.tensor([2]))

    def test_verify_first_sight_commits_nothing(self):
        self._enable()
        self._stage(0, 8, list(range(8)))
        finish_kvmem_forward(FakeForwardBatch(mode="target_verify", seq_len=100))

        _, counts = get_kvmem_registry().controller("slot:0").state.snapshot_layer(0)
        self.assertEqual(int(counts.sum()), 0)

    def test_verify_commits_only_accepted_delta(self):
        self._enable()
        # First forward records seq_len=100.
        self._stage(0, 8, list(range(8)))
        finish_kvmem_forward(FakeForwardBatch(mode="target_verify", seq_len=100))

        # Second forward: context grew by 3 -> exactly 3 draft rows accepted.
        self._stage(0, 8, list(range(8, 16)))
        finish_kvmem_forward(FakeForwardBatch(mode="target_verify", seq_len=103))

        _, counts = get_kvmem_registry().controller("slot:0").state.snapshot_layer(0)
        self.assertEqual(int(counts.sum()), 3)

    def test_verify_delta_clamped_to_staged_rows(self):
        self._enable()
        self._stage(0, 2, [0, 1])
        finish_kvmem_forward(FakeForwardBatch(mode="target_verify", seq_len=100))

        self._stage(0, 2, [2, 3])
        # Growth larger than the staged batch must not over-commit.
        finish_kvmem_forward(FakeForwardBatch(mode="target_verify", seq_len=200))

        _, counts = get_kvmem_registry().controller("slot:0").state.snapshot_layer(0)
        self.assertEqual(int(counts.sum()), 2)

    def test_unknown_mode_rolls_back(self):
        self._enable()
        self._stage(0, 1, [2])
        finish_kvmem_forward(FakeForwardBatch(mode="idle"))

        _, counts = get_kvmem_registry().controller("slot:0").state.snapshot_layer(0)
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
