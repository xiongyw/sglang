from __future__ import annotations

import unittest

import torch

from sglang.srt.hardware_backend.gpu.quantization.gptq_rdna3 import GPTQLinearKernel
from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig


@unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires ROCm")
class Test7900XtxGptqAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GPTQConfig(
            weight_bits=4,
            group_size=64,
            desc_act=False,
            lm_head_quantized=False,
            dynamic={},
        )

    def test_loaded_identity_gidx_layer_is_packed_and_runs(self) -> None:
        class Layer(torch.nn.Module):
            pass

        layer = Layer().cuda()
        layer.qweight = torch.nn.Parameter(
            torch.zeros((8, 8), dtype=torch.int32, device="cuda"), requires_grad=False
        )
        layer.qzeros = torch.nn.Parameter(
            torch.zeros((1, 1), dtype=torch.int32, device="cuda"), requires_grad=False
        )
        layer.scales = torch.nn.Parameter(
            torch.ones((1, 8), dtype=torch.bfloat16, device="cuda"), requires_grad=False
        )
        layer.g_idx = torch.nn.Parameter(
            torch.arange(64, dtype=torch.int32, device="cuda") // 64,
            requires_grad=False,
        )
        kernel = GPTQLinearKernel(self.config)

        kernel.process_weights_after_loading(layer)
        output = kernel.apply(
            layer,
            torch.randn((1, 64), dtype=torch.bfloat16, device="cuda"),
        )
        torch.cuda.synchronize()

        self.assertEqual(layer.g_idx.numel(), 0)
        self.assertEqual(output.shape, (1, 8))
        self.assertTrue(torch.isfinite(output).all())

    def test_nonidentity_gidx_is_rejected(self) -> None:
        class Layer(torch.nn.Module):
            pass

        layer = Layer().cuda()
        layer.qweight = torch.nn.Parameter(
            torch.zeros((16, 8), dtype=torch.int32, device="cuda"), requires_grad=False
        )
        layer.qzeros = torch.nn.Parameter(
            torch.zeros((2, 1), dtype=torch.int32, device="cuda"), requires_grad=False
        )
        layer.scales = torch.nn.Parameter(
            torch.ones((2, 8), dtype=torch.bfloat16, device="cuda"), requires_grad=False
        )
        layer.g_idx = torch.nn.Parameter(
            torch.zeros((128,), dtype=torch.int32, device="cuda"), requires_grad=False
        )

        with self.assertRaisesRegex(ValueError, "identity GPTQ g_idx"):
            GPTQLinearKernel(self.config).process_weights_after_loading(layer)


if __name__ == "__main__":
    unittest.main()
