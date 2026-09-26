import unittest

import torch

from sglang.srt.mem_cache.kvmem_capture_adapter import (
    KVMemCaptureBatch,
    group_k_capture_rows,
)


class TestKVMemCaptureAdapter(unittest.TestCase):
    def test_groups_rows_by_request_and_preserves_logical_positions(self):
        batch = KVMemCaptureBatch(
            key=torch.tensor(
                [[[1.0]], [[2.0]], [[3.0]], [[4.0]]]
            ),
            positions=torch.tensor([10, 20, 11, 21]),
            request_ids=["a", "b", "a", "b"],
        )

        groups = group_k_capture_rows(batch)

        self.assertEqual(list(groups), ["a", "b"])
        torch.testing.assert_close(groups["a"].key, torch.tensor([[[1.0]], [[3.0]]]))
        torch.testing.assert_close(groups["a"].positions, torch.tensor([10, 11]))
        torch.testing.assert_close(groups["b"].key, torch.tensor([[[2.0]], [[4.0]]]))
        torch.testing.assert_close(groups["b"].positions, torch.tensor([20, 21]))

    def test_empty_batch_returns_no_groups(self):
        batch = KVMemCaptureBatch(
            key=torch.empty((0, 1, 1)),
            positions=torch.empty((0,), dtype=torch.long),
            request_ids=[],
        )

        self.assertEqual(group_k_capture_rows(batch), {})

    def test_rejects_mismatched_rows(self):
        with self.assertRaises(ValueError):
            group_k_capture_rows(
                KVMemCaptureBatch(
                    key=torch.ones((2, 1, 1)),
                    positions=torch.tensor([0]),
                    request_ids=["a", "b"],
                )
            )

    def test_rejects_duplicate_request_position_in_one_batch(self):
        with self.assertRaises(ValueError):
            group_k_capture_rows(
                KVMemCaptureBatch(
                    key=torch.ones((2, 1, 1)),
                    positions=torch.tensor([0, 0]),
                    request_ids=["a", "a"],
                )
            )


if __name__ == "__main__":
    unittest.main()
