# SPDX-License-Identifier: Apache-2.0
"""May a graph-captured all-reduce publish the caller's buffer to the peers?

Why this test exists
--------------------
The 1-stage custom all-reduce reads each rank's contribution directly out of that
rank's buffer through a peer pointer, so every rank must be able to observe the
producer's stores. That is only true on a coherent fabric.

Measured on this box (two RX 7900 XTX behind a PLX switch, PCIe, no XGMI):

  * publishing the graph's cacheable activation buffers makes the reduce return the
    local contribution only (or a stale value from a previous invocation)
  * staging the contribution through the uncached registered buffer, which the peers
    read instead, makes the same graph correct on every replay

Upstream never hits this because it gates HIP custom all-reduce on `full_nvlink`,
i.e. on XGMI, where a peer does see the producer's dirty L2 lines.

So: on non-XGMI AMD the input buffer must not be published; elsewhere behaviour is
unchanged, and the memory-saver graph mode keeps using the staged route as before.
"""

from __future__ import annotations

import unittest

from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)
from sglang.srt.utils import is_hip


def _comm(*, full_nvlink: bool, tms_cudagraph: bool) -> CustomAllreduce:
    comm = CustomAllreduce.__new__(CustomAllreduce)
    comm.full_nvlink = full_nvlink
    comm.tms_cudagraph = tms_cudagraph
    comm.disabled = True
    comm.original_disabled = True
    comm._ptr = None  # keep __del__/close() inert
    return comm


class TestPublishPolicy(unittest.TestCase):
    def test_memory_saver_graph_never_publishes(self):
        """Unchanged: the memory-saver graph mode always staged its contribution."""
        self.assertFalse(
            _comm(full_nvlink=True, tms_cudagraph=True)._may_publish_input_buffer()
        )

    @unittest.skipUnless(is_hip(), "XGMI classification is AMD specific")
    def test_xgmi_amd_may_publish(self):
        """Coherent fabric: the fast path stays intact."""
        self.assertTrue(
            _comm(full_nvlink=True, tms_cudagraph=False)._may_publish_input_buffer()
        )

    @unittest.skipUnless(is_hip(), "PCIe AMD case is AMD specific")
    def test_pcie_amd_must_not_publish(self):
        """No XGMI: a peer read of a cacheable buffer is unreliable."""
        self.assertFalse(
            _comm(full_nvlink=False, tms_cudagraph=False)._may_publish_input_buffer()
        )


if __name__ == "__main__":
    unittest.main()
