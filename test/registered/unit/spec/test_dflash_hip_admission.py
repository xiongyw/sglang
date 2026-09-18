import unittest

from sglang.srt.arg_groups.speculative_hook import _dflash_device_supported


class TestDflashHipAdmission(unittest.TestCase):
    def test_hip_is_a_supported_dflash_device(self):
        self.assertTrue(_dflash_device_supported(device="cuda", is_hip=True))

    def test_cpu_is_not_a_supported_dflash_device(self):
        self.assertFalse(_dflash_device_supported(device="cpu", is_hip=False))


if __name__ == "__main__":
    unittest.main()
