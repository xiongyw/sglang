"""The RDNA3 compressed-tensors W4A16 path must not narrow activations to a range it overflows.

The kernel picks its activation behaviour from the scale dtype:

* bf16 activations + fp16 scales -> ``launch_gemm_q4_mixed`` -> activations are
  converted to fp16 in-register, so anything above 65504 becomes inf, then NaN.
* bf16 activations + bf16 scales -> ``launch_gemm_q4<bf16_t>`` -> activations are
  widened to fp32 and never narrowed.

A quantized weight-only draft must run its activations on the second path: the
DFlash2 draft legitimately reaches ~95,000 inside its MLP, and NaN draft states
silently zero speculative acceptance. fp16 scales therefore cannot be used for
this scheme, whatever they cost in scaling precision.
"""
from __future__ import annotations

import unittest

import torch

PACK_FACTOR = 8
K = 32
N_OUT = 8


def _pack_symmetric_4bit(w: torch.Tensor, zero_point: int = 8):
    scale = (w.abs().amax(dim=1, keepdim=True) / 7.0).clamp_min(1e-12)
    q = (w / scale).round() + zero_point
    q = q.clamp(0, 15).to(torch.int64)
    words = K // PACK_FACTOR
    packed = torch.zeros(N_OUT, words, dtype=torch.int64)
    for word in range(words):
        for j in range(PACK_FACTOR):
            packed[:, word] |= q[:, word * PACK_FACTOR + j] << (4 * j)
    return q, scale, packed.to(torch.int32)


def _run_with_scale_dtype(scale_dtype: torch.dtype, act_max: float):
    """Return (all_finite, relative_error) for the kernel at this scale dtype."""
    from sgl_kernel import common_ops  # noqa: F401

    device = "cuda"
    torch.manual_seed(0)
    w = (torch.randn(N_OUT, K, dtype=torch.float32) * 0.5).abs() + 0.25
    q, scale, packed = _pack_symmetric_4bit(w)
    w_ref = (q.float() - 8.0) * scale

    b_q_weight = packed.t().contiguous().to(device)
    g_idx = torch.empty((0,), dtype=torch.int32, device=device)
    torch.ops.sgl_kernel.gptq_shuffle_rdna3(b_q_weight, g_idx)

    fill = sum(7 << (4 * i) for i in range(PACK_FACTOR))
    qzeros = torch.full((1, N_OUT // PACK_FACTOR), fill, dtype=torch.int32, device=device)
    scales = scale.t().contiguous().to(scale_dtype).to(device)

    x = (torch.randn(1, K, dtype=torch.bfloat16).abs() + 0.25)
    x[0, 0] = act_max
    x = x.to(device)
    y = torch.ops.sgl_kernel.gptq_gemm_rdna3(
        x.contiguous(), b_q_weight, qzeros, scales, g_idx, False
    )
    torch.cuda.synchronize()
    ref = x.float().cpu() @ w_ref.t()
    finite = bool(torch.isfinite(y).all())
    err = float((y.float().cpu() - ref).norm() / ref.norm())
    return finite, err


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires a ROCm device")
class TestGptqRdna3ScaleDtypeRange(unittest.TestCase):
    def test_scale_dtype_keeps_large_activations_finite(self) -> None:
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_wNa16 as scheme,
        )

        production = scheme.RDNA3_W4A16_SCALES_DTYPE
        finite_used, err_used = _run_with_scale_dtype(production, act_max=1.0e5)
        self.assertTrue(
            finite_used,
            f"scale dtype {production} narrows activations and overflows: "
            "expect inf/NaN above the fp16 maximum",
        )
        self.assertLess(err_used, 1e-1, f"unexpected error with {production}: {err_used:.4f}")

    def test_fp16_scales_are_rejected_by_the_same_measurement(self) -> None:
        """Documents why fp16 scales are unusable here, so the test cannot pass vacuously."""
        finite_fp16, _ = _run_with_scale_dtype(torch.float16, act_max=1.0e5)
        self.assertFalse(
            finite_fp16,
            "fp16 scales were expected to overflow on this activation; if the kernel "
            "changed, re-derive whether RDNA3_W4A16_SCALES_DTYPE is still needed",
        )


if __name__ == "__main__":
    unittest.main()
