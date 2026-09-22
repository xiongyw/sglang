import unittest
from types import SimpleNamespace

import torch

from sglang.srt.speculative.dflash_utils import (
    can_dflash_dequant_fused_qkv_proj,
)
from sglang.srt.layers.quantization.compressed_tensors.compressed_tensors import (
    CompressedTensorsLinearMethod,
)
from sglang.srt.layers.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    CompressedTensorsWNA16,
)


class TestDflashFusedKvQuantizedQkv(unittest.TestCase):
    @staticmethod
    def _qkv(**overrides):
        fields = {
            "quant_method": CompressedTensorsLinearMethod(None),
            "scheme": CompressedTensorsWNA16(
                strategy="group", num_bits=4, group_size=128, symmetric=True
            ),
            "weight_packed": torch.empty((64, 128), dtype=torch.int32),
            "weight_scale": torch.empty((128, 4), dtype=torch.float16),
            "weight_zero_point": torch.empty((16, 4), dtype=torch.int32),
            "weight_shape": torch.tensor([512, 128], dtype=torch.int64),
            "weight_g_idx": torch.empty((0,), dtype=torch.int32),
            "bias": None,
        }
        fields.update(overrides)
        return SimpleNamespace(**fields)

    def test_packed_qkv_is_eligible_for_fused_context_kv(self):
        eligible, reason = can_dflash_dequant_fused_qkv_proj(self._qkv())

        self.assertTrue(eligible)
        self.assertEqual(reason, "")

    def test_missing_packed_tensor_fails_closed(self):
        qkv = self._qkv(weight_scale=None)

        eligible, reason = can_dflash_dequant_fused_qkv_proj(qkv)

        self.assertFalse(eligible)
        self.assertIn("weight_scale", reason)

    def test_wrong_quant_method_fails_closed(self):
        qkv = self._qkv(quant_method=object())

        eligible, reason = can_dflash_dequant_fused_qkv_proj(qkv)

        self.assertFalse(eligible)
        self.assertIn("CompressedTensorsLinearMethod", reason)

    def test_malformed_metadata_fails_closed(self):
        qkv = self._qkv(weight_scale=torch.empty((128, 3), dtype=torch.float16))

        eligible, reason = can_dflash_dequant_fused_qkv_proj(qkv)

        self.assertFalse(eligible)
        self.assertIn("weight_scale shape", reason)

    def test_nonempty_gidx_fails_closed(self):
        qkv = self._qkv(weight_g_idx=torch.zeros(1, dtype=torch.int32))

        eligible, reason = can_dflash_dequant_fused_qkv_proj(qkv)

        self.assertFalse(eligible)
        self.assertIn("weight_g_idx", reason)

    def test_qkv_bias_fails_closed(self):
        eligible, reason = can_dflash_dequant_fused_qkv_proj(
            self._qkv(bias=torch.zeros(128))
        )

        self.assertFalse(eligible)
        self.assertIn("bias", reason)


if __name__ == "__main__":
    unittest.main()
