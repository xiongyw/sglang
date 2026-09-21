"""RDNA3 W4A16 must not narrow activations to a range they overflow.

The kernel keeps the activation dtype and the scale storage dtype independent: it
reads bf16 activations as bf16 (never converting them to fp16, whose maximum is
65504) and widens the scales to fp32, which is exact for both fp16 and bf16
scales. The GEMM is therefore range-safe for any dtype pairing the op accepts.

This is not hypothetical. The DFlash2 draft legitimately reaches ~95,000 inside
its MLP; the old fp16-narrowing instantiation turned that into inf and then NaN,
which silently zeroed speculative acceptance while looking perfectly healthy.

Pre-rewrite history: bf16 activations with fp16 scales selected a "mixed"
instantiation that narrowed A to fp16, and the only way to avoid it was to
force bf16 scales. Both dtypes are safe now.
"""
from __future__ import annotations

import unittest

import torch

PACK_FACTOR = 8
K = 32
N_OUT_SCALAR = 8  # scalar path: N % 8 == 0
N_OUT_WMMA = 16  # WMMA path: N % 16 == 0
M_WMMA = 64  # bf16 activations take WMMA from M >= 16
FP16_MAX = 65504.0
ACT_MAX = 1.0e5


def _pack_symmetric_4bit(w: torch.Tensor, zero_point: int = 8):
    """Quantize [N, K] weights to symmetric 4-bit, one group, and pack them.

    Nibble j of word w holds K index ``w * PACK_FACTOR + j`` (low nibble first),
    which is the layout the RDNA3 kernel consumes after ``gptq_shuffle_rdna3``.
    """
    n_out = w.shape[0]
    scale = (w.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-12)
    q = (w / scale).round() + zero_point
    q = q.clamp(0, 15).to(torch.int64)
    words = K // PACK_FACTOR
    packed = torch.zeros(n_out, words, dtype=torch.int64)
    for word in range(words):
        for j in range(PACK_FACTOR):
            packed[:, word] |= q[:, word * PACK_FACTOR + j] << (4 * j)
    return q, scale, packed.to(torch.int32)


def _run(size_m: int, n_out: int, scale_dtype: torch.dtype, act_max: float = 1.0):
    """Return (kernel_output, exact_reference) for one activation/scale pairing."""
    from sgl_kernel import common_ops  # noqa: F401  (registers the ops)

    device = "cuda"
    torch.manual_seed(0)
    # All-positive weights and activations keep the K-sum free of catastrophic
    # cancellation, so a mismatch reflects range/precision, not conditioning.
    w = (torch.randn(n_out, K, dtype=torch.float32) * 0.5).abs() + 0.25
    q, scale, packed = _pack_symmetric_4bit(w)
    w_ref = (q.float() - 8.0) * scale

    b_q_weight = packed.t().contiguous().to(device)
    g_idx = torch.empty((0,), dtype=torch.int32, device=device)
    torch.ops.sgl_kernel.gptq_shuffle_rdna3(b_q_weight, g_idx)

    # Stored nibble 7 yields effective zero point 8, as the appliance path builds it.
    fill = sum(7 << (4 * i) for i in range(PACK_FACTOR))
    qzeros = torch.full((1, n_out // PACK_FACTOR), fill, dtype=torch.int32, device=device)
    scales = scale.t().contiguous().to(scale_dtype).to(device)

    x = torch.randn(size_m, K, dtype=torch.bfloat16).abs() + 0.25
    if act_max != 1.0:
        x[0, 0] = act_max
    x = x.to(device)

    y = torch.ops.sgl_kernel.gptq_gemm_rdna3(
        x.contiguous(), b_q_weight, qzeros, scales, g_idx, False
    )
    torch.cuda.synchronize()
    return y.float().cpu(), x.float().cpu() @ w_ref.t()


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires a ROCm device")
class TestGptqRdna3ActivationRange(unittest.TestCase):
    def _assert_finite_and_close(self, y, ref, label: str) -> None:
        self.assertTrue(torch.isfinite(y).all(), f"{label}: expected finite, got {y}")
        self.assertTrue(
            torch.allclose(y, ref, rtol=1e-1, atol=1e-1), f"{label}: {y} vs {ref}"
        )

    def test_scalar_path_matches_reference_for_small_activations(self) -> None:
        y, ref = _run(1, N_OUT_SCALAR, torch.float16)
        self._assert_finite_and_close(y, ref, "scalar/small/fp16 scales")

    def test_scalar_path_survives_activations_above_fp16_max(self) -> None:
        x_max = torch.tensor([ACT_MAX])
        self.assertGreater(float(x_max), FP16_MAX)

        y, ref = _run(1, N_OUT_SCALAR, torch.float16, act_max=ACT_MAX)
        self._assert_finite_and_close(y, ref, "scalar/1e5/fp16 scales")

    def test_scalar_path_survives_activations_above_fp16_max_with_bf16_scales(self) -> None:
        y, ref = _run(1, N_OUT_SCALAR, torch.bfloat16, act_max=ACT_MAX)
        self._assert_finite_and_close(y, ref, "scalar/1e5/bf16 scales")

    def test_wmma_prefill_path_survives_activations_above_fp16_max(self) -> None:
        """bf16 activations from M >= 16 take the WMMA kernel, not the scalar one.

        WMMA needs A and the scales in one dtype, so this path used to hand WMMA a
        narrowed fp16 copy of A. It must widen the scales instead.
        """
        y, ref = _run(M_WMMA, N_OUT_WMMA, torch.float16, act_max=ACT_MAX)
        self._assert_finite_and_close(y, ref, "wmma/1e5/fp16 scales")

    def test_wmma_prefill_path_survives_activations_above_fp16_max_with_bf16_scales(self) -> None:
        y, ref = _run(M_WMMA, N_OUT_WMMA, torch.bfloat16, act_max=ACT_MAX)
        self._assert_finite_and_close(y, ref, "wmma/1e5/bf16 scales")


if __name__ == "__main__":
    unittest.main()
