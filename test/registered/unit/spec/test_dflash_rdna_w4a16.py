import unittest

import torch

from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    _rdna3_symmetric_zero_points,
)


class TestDflashRdnaW4A16(unittest.TestCase):
    def test_symmetric_4bit_zero_points_use_neutral_packed_value(self):
        zeros = _rdna3_symmetric_zero_points(
            groups=2, out_features=16, pack_factor=8, device="cpu"
        )
        self.assertEqual(zeros.shape, (2, 2))
        self.assertEqual(zeros.dtype, torch.int32)
        self.assertTrue(torch.equal(zeros, torch.full((2, 2), 0x77777777, dtype=torch.int32)))


if __name__ == "__main__":
    unittest.main()
