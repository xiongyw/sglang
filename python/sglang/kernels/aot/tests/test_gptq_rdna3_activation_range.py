from __future__ import annotations

import unittest

import torch

# FP16 maximum finite value. The RDNA3 W4A16 scalar kernel converts bf16
# activations to fp16 in-register, so any activation above this bound becomes
# inf and poisons the GEMM with NaN.
FP16_MAX = 65504.0

PACK_FACTOR = 8  # int4: 8 nibbles per int32
GROUP_SIZE = 32
K = 32
N_OUT = 8


def _pack_symmetric_4bit(w: torch.Tensor, zero_point: int = 8):
    """Quantize [N, K] weights to symmetric 4-bit, one group, and pack them.

    Nibble j of word w holds K index ``w * PACK_FACTOR + j`` (low nibble first),
    which is the layout the RDNA3 kernel consumes after ``gptq_shuffle_rdna3``.
    """
    scale = (w.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-12)
    q = (w / scale).round() + zero_point
    q = q.clamp(0, 15).to(torch.int64)
    words = K // PACK_FACTOR
    packed = torch.zeros(N_OUT, words, dtype=torch.int64)
    for word in range(words):
        for j in range(PACK_FACTOR):
            packed[:, word] |= q[:, word * PACK_FACTOR + j] << (4 * j)
    return q, scale, packed.to(torch.int32)


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires a ROCm device")
class TestGptqRdna3ActivationRange(unittest.TestCase):
    def _run(self, x: torch.Tensor):
        from sgl_kernel import common_ops  # noqa: F401

        device = "cuda"
        torch.manual_seed(0)
        # All-positive weights and activations keep the K-sum free of catastrophic
        # cancellation, so a mismatch reflects range/precision rather than conditioning.
        w = (torch.randn(N_OUT, K, dtype=torch.float32) * 0.5).abs() + 0.25
        q, scale, packed = _pack_symmetric_4bit(w)
        w_ref = (q.float() - 8.0) * scale

        b_q_weight = packed.t().contiguous().to(device)
        g_idx = torch.empty((0,), dtype=torch.int32, device=device)
        torch.ops.sgl_kernel.gptq_shuffle_rdna3(b_q_weight, g_idx)

        # Stored nibble 7 yields effective zero point 8, as the appliance path builds it.
        fill = sum(7 << (4 * i) for i in range(PACK_FACTOR))
        qzeros = torch.full((1, N_OUT // PACK_FACTOR), fill, dtype=torch.int32, device=device)
        scales = scale.t().contiguous().to(torch.float16).to(device)

        y = torch.ops.sgl_kernel.gptq_gemm_rdna3(
            x.to(device).contiguous(), b_q_weight, qzeros, scales, g_idx, False
        )
        torch.cuda.synchronize()
        ref = (x.float() @ w_ref.t().cpu())
        return y.float().cpu(), ref

    def test_matches_reference_for_small_activations(self) -> None:
        x = torch.randn(1, K, dtype=torch.bfloat16).abs() + 0.25
        y, ref = self._run(x)
        self.assertTrue(torch.isfinite(y).all(), "small activations must stay finite")
        # The kernel computes at fp16 grade (A is narrowed to fp16, so ~1e-3
        # relative per element, and accumulation is lossy), which is why the
        # bound is looser than an fp32 GEMM would need. It still rejects
        # clamping to the fp16 maximum, which would miss by ~35% on this case.
        self.assertTrue(torch.allclose(y, ref, rtol=1e-1, atol=1e-1), f"{y} vs {ref}")

    @unittest.expectedFailure
    def test_matches_reference_for_activations_above_fp16_max(self) -> None:
        """Known kernel limitation: the fp16-scale instantiation narrows activations.

        bf16 provides far more range than the kernel narrows to: an activation of 1e5
        is an ordinary bf16 value, and narrowing it to fp16 produces inf and then NaN.
        Callers avoid this by selecting the bf16-scale instantiation instead (see
        RDNA3_W4A16_SCALES_DTYPE in the compressed-tensors W4A16 scheme and the
        companion test_gptq_rdna3_scale_dtype_range.py). Fixing the kernel itself, so
        that fp16 scales can safely accompany bf16 activations, is outstanding: it
        would let the fp16 (faster) dot-product path keep full bf16 range.
        """
        x = torch.randn(1, K, dtype=torch.bfloat16).abs() + 0.25
        x[0, 0] = 1.0e5
        self.assertGreater(float(x.abs().max()), FP16_MAX)

        y, ref = self._run(x)
        self.assertTrue(
            torch.isfinite(y).all(),
            f"activations above the fp16 maximum must not produce inf/NaN, got {y}",
        )
        self.assertTrue(torch.allclose(y, ref, rtol=1e-1, atol=1e-1), f"{y} vs {ref}")


if __name__ == "__main__":
    unittest.main()
