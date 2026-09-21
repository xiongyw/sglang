"""The RDNA3 compressed-tensors W4A16 path must use the checkpoint's zero-point convention.

compressed-tensors ``pack-quantized`` serializes ``qzeros`` as ``zero_point - 1``
(the GPTQ-v1 convention), so the kernel must apply its ``+1`` zero offset. The
target GPTQ path derives this from ``checkpoint_format == "gptq_v2"``; the
compressed-tensors path must not hard-code the v2 (no-offset) behaviour, which
biases every dequantized weight by one scale unit and inflates draft activations
until they overflow the fp16 range of the kernel.
"""
from __future__ import annotations

import unittest

import torch

PACK_FACTOR = 8
GROUP_SIZE = 32
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


def _kernel_relative_error(use_v2_format: bool) -> float:
    """Relative error of the kernel against exact dequantization for a stored zero point of 7."""
    from sgl_kernel import common_ops  # noqa: F401

    device = "cuda"
    torch.manual_seed(0)
    w = (torch.randn(N_OUT, K, dtype=torch.float32) * 0.5).abs() + 0.25
    q, scale, packed = _pack_symmetric_4bit(w)
    # Effective zero point 8, stored as 7 in the checkpoint tensor.
    w_ref = (q.float() - 8.0) * scale

    b_q_weight = packed.t().contiguous().to(device)
    g_idx = torch.empty((0,), dtype=torch.int32, device=device)
    torch.ops.sgl_kernel.gptq_shuffle_rdna3(b_q_weight, g_idx)

    fill = sum(7 << (4 * i) for i in range(PACK_FACTOR))
    qzeros = torch.full((1, N_OUT // PACK_FACTOR), fill, dtype=torch.int32, device=device)
    scales = scale.t().contiguous().to(torch.float16).to(device)

    x = (torch.randn(1, K, dtype=torch.bfloat16).abs() + 0.25).to(device)
    y = torch.ops.sgl_kernel.gptq_gemm_rdna3(
        x.contiguous(), b_q_weight, qzeros, scales, g_idx, use_v2_format
    )
    torch.cuda.synchronize()
    ref = (x.float().cpu() @ w_ref.t())
    return ((y.float().cpu() - ref).norm() / ref.norm()).item()


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires a ROCm device")
class TestGptqRdna3CompressedTensorsZeroOffset(unittest.TestCase):
    def test_production_flag_selects_the_checkpoint_zero_point_convention(self) -> None:
        from sglang.srt.layers.quantization.compressed_tensors.schemes import (
            compressed_tensors_wNa16 as scheme,
        )

        flag = scheme.RDNA3_COMPRESSED_TENSORS_V2_ZERO_OFFSET
        err_used = _kernel_relative_error(flag)
        err_other = _kernel_relative_error(not flag)

        self.assertLess(
            err_used,
            err_other,
            f"the production flag ({flag}) picks the worse zero-point convention: "
            f"rel_err={err_used:.4f} vs {err_other:.4f} for the other setting",
        )
        self.assertLess(err_used, 0.08, f"zero-point convention is wrong: rel_err={err_used:.4f}")


if __name__ == "__main__":
    unittest.main()
