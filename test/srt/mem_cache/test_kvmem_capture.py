import unittest

import torch

from sglang.srt.mem_cache.kvmem_mean_k import KVMemMeanKAccumulator
from sglang.srt.mem_cache.kvmem_capture import KVMemKCaptureSession


class TestKVMemKCaptureSession(unittest.TestCase):
    def make_session(self):
        return KVMemKCaptureSession(
            KVMemMeanKAccumulator(block_size=2, kv_heads=1, head_dim=1)
        )

    def test_commit_only_adds_accepted_rows(self):
        session = self.make_session()
        session.begin()
        session.stage(
            torch.tensor([[[1.0]], [[3.0]], [[9.0]]]),
            torch.tensor([0, 1, 2]),
        )

        session.commit(accepted_tokens=2)
        means, counts = session.accumulator.snapshot()

        torch.testing.assert_close(means, torch.tensor([[[2.0]]]))
        torch.testing.assert_close(counts, torch.tensor([2]))
        self.assertFalse(session.active)

    def test_rejected_draft_rows_do_not_consume_positions(self):
        session = self.make_session()
        session.begin()
        session.stage(torch.tensor([[[1.0]], [[3.0]]]), torch.tensor([0, 1]))
        session.rollback()

        session.begin()
        session.stage(torch.tensor([[[5.0]]]), torch.tensor([0]))
        session.commit(accepted_tokens=1)

        means, counts = session.accumulator.snapshot()
        torch.testing.assert_close(means, torch.tensor([[[5.0]]]))
        torch.testing.assert_close(counts, torch.tensor([1]))

    def test_commit_can_accept_zero_and_discard_everything(self):
        session = self.make_session()
        session.begin()
        session.stage(torch.tensor([[[1.0]]]), torch.tensor([0]))
        session.commit(accepted_tokens=0)

        _, counts = session.accumulator.snapshot()
        torch.testing.assert_close(counts, torch.tensor([], dtype=torch.long))

    def test_invalid_lifecycle_is_rejected(self):
        session = self.make_session()
        with self.assertRaises(RuntimeError):
            session.stage(torch.ones((1, 1, 1)), torch.tensor([0]))
        session.begin()
        with self.assertRaises(RuntimeError):
            session.begin()
        with self.assertRaises(ValueError):
            session.commit(accepted_tokens=2)


if __name__ == "__main__":
    unittest.main()
