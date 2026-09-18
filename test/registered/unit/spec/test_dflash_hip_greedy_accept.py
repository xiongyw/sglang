import unittest

from sglang.srt.speculative.dflash_utils import _dflash_triton_greedy_accept_supported


class TestDflashHipGreedyAccept(unittest.TestCase):
    def test_hip_does_not_compile_triton_greedy_accept_kernel(self):
        self.assertFalse(_dflash_triton_greedy_accept_supported(is_hip=True))

    def test_cuda_retains_triton_greedy_accept_kernel(self):
        self.assertTrue(_dflash_triton_greedy_accept_supported(is_hip=False))


if __name__ == "__main__":
    unittest.main()
