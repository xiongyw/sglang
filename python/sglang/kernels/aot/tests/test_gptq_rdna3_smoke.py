from __future__ import annotations

import unittest

import torch


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires a ROCm device")
class TestGptqRdna3Smoke(unittest.TestCase):
    def test_scalar_operator_returns_finite_bf16_for_valid_w4a16_shape(self) -> None:
        import sgl_kernel  # noqa: F401

        device = "cuda"
        # Scalar kernel contract: M=1, K multiple of 32, N multiple of 8,
        # and one GPTQ group. All zero 4-bit weights produce a finite result.
        activations = torch.randn((1, 32), dtype=torch.bfloat16, device=device)
        qweight = torch.zeros((4, 8), dtype=torch.int32, device=device)
        qzeros = torch.zeros((1, 1), dtype=torch.int32, device=device)
        scales = torch.ones((1, 8), dtype=torch.bfloat16, device=device)
        g_idx = torch.empty((0,), dtype=torch.int32, device=device)

        output = torch.ops.sgl_kernel.gptq_gemm_rdna3(
            activations.contiguous(),
            qweight.contiguous(),
            qzeros.contiguous(),
            scales.contiguous(),
            g_idx.contiguous(),
            False,
        )
        torch.cuda.synchronize()

        self.assertEqual(output.shape, (1, 8))
        self.assertEqual(output.dtype, torch.bfloat16)
        self.assertTrue(torch.isfinite(output).all())


if __name__ == "__main__":
    unittest.main()
