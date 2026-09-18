import unittest

from sglang.srt.models.dflash import _dflash_triton_selector_supported


class TestDflashHipSelectorPolicy(unittest.TestCase):
    def test_hip_uses_eager_selector_walk(self):
        self.assertFalse(_dflash_triton_selector_supported(is_hip=True))

    def test_cuda_retains_triton_selector_walk(self):
        self.assertTrue(_dflash_triton_selector_supported(is_hip=False))


if __name__ == "__main__":
    unittest.main()
