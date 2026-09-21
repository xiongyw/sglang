"""The scale storage dtype must not change the activation range the kernel can carry.

fp16 and bf16 scales are both widened to fp32 exactly, so neither narrows an
activation and neither is more range-limited than the other. The two dtypes must
also agree with each other to within the bf16 rounding of the scale values
themselves (~0.4% relative), which is the only difference between them.

This pins the pairing that used to be broken: bf16 activations with fp16 scales —
the shape of the compressed-tensors W4A16 checkpoint this appliance serves.
"""
from __future__ import annotations

import unittest

import torch

PACK_FACTOR = 8
K = 32
N_OUT = 8
ACT_MAX = 1.0e5


def _pack_symmetric_4bit(w: torch.Tensor, zero_point: int = 8):
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


def _run(scale_dtype: torch.dtype, act_max: float):
    """Return (output, all_finite, relative_error) for this scale dtype."""
    from sgl_kernel import common_ops  # noqa: F401  (registers the ops)

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

    x = torch.randn(1, K, dtype=torch.bfloat16).abs() + 0.25
    x[0, 0] = act_max
    x = x.to(device)

    y = torch.ops.sgl_kernel.gptq_gemm_rdna3(
        x.contiguous(), b_q_weight, qzeros, scales, g_idx, False
    )
    torch.cuda.synchronize()
    ref = x.float().cpu() @ w_ref.t()
    y = y.float().cpu()
    finite = bool(torch.isfinite(y).all())
    err = float((y - ref).norm() / ref.norm())
    return y, finite, err


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires a ROCm device")
class TestGptqRdna3ScaleDtypeRange(unittest.TestCase):
    def test_every_supported_scale_dtype_carries_large_activations(self) -> None:
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_wNa16 as scheme,
        )

        for dtype in scheme.RDNA3_W4A16_SCALES_DTYPES:
            with self.subTest(scale_dtype=dtype):
                _, finite, err = _run(dtype, ACT_MAX)
                self.assertTrue(
                    finite,
                    f"{dtype} scales produced inf/NaN on a {ACT_MAX:g} activation; the "
                    "kernel must widen scales, never narrow activations",
                )
                self.assertLess(err, 1e-1, f"unexpected error with {dtype}: {err:.4f}")

    def test_scale_dtype_does_not_change_the_result_materially(self) -> None:
        y_fp16, finite_fp16, _ = _run(torch.float16, ACT_MAX)
        y_bf16, finite_bf16, _ = _run(torch.bfloat16, ACT_MAX)
        self.assertTrue(finite_fp16 and finite_bf16)

        # Same scale values in two dtypes: only the scale rounding differs.
        rel = float((y_fp16 - y_bf16).norm() / y_fp16.norm())
        self.assertLess(rel, 2e-2, f"scale dtype changed the result by {rel:.4f}")


if __name__ == "__main__":
    unittest.main()
