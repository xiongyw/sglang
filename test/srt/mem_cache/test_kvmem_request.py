import unittest

import torch

from sglang.srt.mem_cache.kvmem_request import (
    KVMemRequestController,
    KVMemRequestConfig,
)


class TestKVMemRequestController(unittest.TestCase):
    def test_capture_score_select_and_diff(self):
        controller = KVMemRequestController(
            KVMemRequestConfig(
                block_size=2,
                max_tokens=8,
                selection_budget_tokens=4,
                sink_blocks=1,
            )
        )
        controller.begin_capture(layer_id=0)
        controller.stage_k(
            layer_id=0,
            key=torch.tensor(
                [[[1.0]], [[1.0]], [[-1.0]], [[-1.0]], [[0.5]], [[0.5]]]
            ),
            positions=torch.arange(6),
        )
        controller.commit_capture(layer_id=0, accepted_tokens=6)

        scores = controller.score_query(0, torch.tensor([[[1.0]]]))
        result = controller.select(0, scores)

        self.assertEqual(result.selection.block_ids, [0, 2])
        self.assertEqual(result.diff.stage_in, [0, 2])
        self.assertEqual(result.diff.stage_out, [])

    def test_rollback_does_not_change_request_state(self):
        controller = KVMemRequestController(
            KVMemRequestConfig(
                block_size=2,
                max_tokens=8,
                selection_budget_tokens=4,
            )
        )
        controller.begin_capture(0)
        controller.stage_k(0, torch.ones((2, 1, 1)), torch.tensor([0, 1]))
        controller.rollback_capture(0)

        with self.assertRaises(KeyError):
            controller.score_query(0, torch.ones((1, 1, 1)))


if __name__ == "__main__":
    unittest.main()
