from __future__ import annotations

import unittest

import torch

from sglang.srt.layers.quantization.gptq.gptq import GPTQConfig
from sglang.srt.layers.quantization.gptq.schemes.gptq_linear import GPTQLinearScheme


class Test7900XtxGptqDispatch(unittest.TestCase):
    def setUp(self) -> None:
        self.config = GPTQConfig(
            weight_bits=4,
            group_size=64,
            desc_act=False,
            lm_head_quantized=False,
            dynamic={},
        )

    @unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires ROCm")
    def test_gfx1100_gptq_allows_bfloat16_activations(self) -> None:
        self.assertIn(torch.bfloat16, self.config.get_supported_act_dtypes())

    @unittest.skipUnless(torch.version.hip and torch.cuda.is_available(), "requires ROCm")
    def test_gfx1100_constructs_dense_gptq_scheme(self) -> None:
        scheme = GPTQLinearScheme(self.config)

        self.assertEqual(scheme.kernel.__class__.__name__, "GPTQLinearKernel")
        self.assertTrue(scheme.kernel.use_shuffle)


if __name__ == "__main__":
    unittest.main()
