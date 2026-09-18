import unittest

from sglang.srt.speculative.dflash_worker_v2 import _dflash_triton_prepare_block_supported


class TestDflashHipPreparePolicy(unittest.TestCase):
    def test_hip_uses_eager_prepare_block_until_triton_path_is_validated(self):
        self.assertFalse(_dflash_triton_prepare_block_supported(is_cuda=False, is_hip=True))

    def test_cuda_retains_triton_prepare_block(self):
        self.assertTrue(_dflash_triton_prepare_block_supported(is_cuda=True, is_hip=False))


if __name__ == "__main__":
    unittest.main()
