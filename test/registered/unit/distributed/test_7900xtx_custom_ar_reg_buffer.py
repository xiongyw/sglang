# SPDX-License-Identifier: Apache-2.0
"""The registered all-reduce data buffer must be uncached, IPC-exportable memory.

Why this test exists
--------------------
On HIP the custom-AR data buffer is shared with peers: each rank reads the other's
copy through a peer pointer obtained from an IPC handle, and the 1-stage kernel
reduces across them. Upstream allocates it with `torch.empty`, which comes from
the PyTorch caching allocator. Measured on this box:

    hipIpcGetMemHandle on a pooled torch tensor   -> rc=1 (hipErrorInvalidValue)
    hipIpcGetMemHandle on hipMalloc / uncached    -> rc=0

and a cached buffer additionally lets the owning rank read a stale local L2 copy,
so peer contributions never become visible. The observed symptom of getting this
wrong is a result equal to the *local* contribution instead of the sum - which is
exactly what the end-to-end AR probe measured before this change.

So the HIP path must obtain the buffer from the dedicated uncached allocator, and
must fail fast if that op is unavailable rather than silently falling back to a
pooled allocation (a silent fallback is the bug, not a safety net).
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.distributed.device_communicators import custom_all_reduce as car
from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    _allocate_registered_buffer,
)
from sglang.srt.utils import is_hip

MAX_SIZE = 2 * 8192 * 1024


class _FakeOps:
    def __init__(self, *, with_op: bool) -> None:
        if with_op:
            self.allocate_reg_buffer = self._allocate_reg_buffer
        self.calls = []

    def _allocate_reg_buffer(self, size):
        self.calls.append(size)
        return ("uncached-reg-buffer", size)


@unittest.skipUnless(is_hip(), "HIP registered-buffer rule is ROCm specific")
class TestRegisteredBufferAllocation(unittest.TestCase):
    def test_uses_dedicated_uncached_allocator(self):
        ops = _FakeOps(with_op=True)
        buf = _allocate_registered_buffer(ops, MAX_SIZE, torch.device("cuda:0"))
        self.assertEqual(buf, ("uncached-reg-buffer", MAX_SIZE))
        self.assertEqual(ops.calls, [MAX_SIZE])

    def test_missing_op_fails_fast(self):
        """No silent fallback to a pooled (non-exportable) allocation."""
        ops = _FakeOps(with_op=False)
        with self.assertRaises(RuntimeError):
            _allocate_registered_buffer(ops, MAX_SIZE, torch.device("cuda:0"))

    def test_real_extension_exposes_the_op(self):
        """Integration: the built extension must actually provide the op.

        Note: this must interrogate the *extension*, not `custom_all_reduce.ops`,
        which is a pure-Python shim. Asserting on the shim would pass regardless
        of whether the C++ op exists - a vacuous test.
        """
        import sgl_kernel.allreduce as custom_ar

        self.assertTrue(
            hasattr(custom_ar, "allocate_reg_buffer"),
            "sgl_kernel.allreduce is missing allocate_reg_buffer; rebuild the "
            "gfx1100 AOT extension before using custom all-reduce",
        )
        self.assertTrue(
            hasattr(torch.ops.sgl_kernel, "allocate_reg_buffer"),
            "torch.ops.sgl_kernel.allocate_reg_buffer is not registered",
        )


if __name__ == "__main__":
    unittest.main()
