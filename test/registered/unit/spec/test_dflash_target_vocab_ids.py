import unittest

import torch

from sglang.srt.speculative.dflash_worker_v2 import _validate_dflash_target_vocab_ids


class TestDflashTargetVocabIds(unittest.TestCase):
    def test_out_of_range_draft_id_is_rejected_before_target_embedding(self):
        ids = torch.tensor([[7, 248320]], dtype=torch.int64)
        with self.assertRaisesRegex(ValueError, r"range=\[7, 248320\].*vocab_size=248320"):
            _validate_dflash_target_vocab_ids(ids, vocab_size=248320)


if __name__ == "__main__":
    unittest.main()
