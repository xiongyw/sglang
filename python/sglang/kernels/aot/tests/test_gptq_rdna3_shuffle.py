from __future__ import annotations

import unittest

import torch


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires a ROCm device")
class TestGptqRdna3Shuffle(unittest.TestCase):
    def test_four_bit_shuffle_reorders_one_packed_word(self) -> None:
        import sgl_kernel  # noqa: F401

        # Low-to-high nibbles 0..7. The exllama layout interleaves the four
        # even nibbles in bits 0..15 and the four odd nibbles in bits 16..31.
        packed = torch.tensor([[0x76543210]], dtype=torch.int32, device="cuda")
        expected = torch.tensor([[0x75316420]], dtype=torch.int32, device="cuda")
        empty_permutation = torch.empty((0,), dtype=torch.int32, device="cuda")

        torch.ops.sgl_kernel.gptq_shuffle_rdna3(packed, empty_permutation)
        torch.cuda.synchronize()

        self.assertTrue(torch.equal(packed, expected))


if __name__ == "__main__":
    unittest.main()
