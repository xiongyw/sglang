# SPDX-License-Identifier: Apache-2.0
"""Target-profile policy for custom all-reduce selection on the dual 7900 XTX box.

Why this test exists
--------------------
Upstream gates HIP custom all-reduce on `full_nvlink`, which on ROCm means XGMI
(`amdsmi_topo_get_link_type(...)["type"] == 2`). The two RX 7900 XTX cards are
behind a PLX switch and link over PCIe, never XGMI, so upstream declines custom AR
by policy - even though peer access is functional (`hipDeviceCanAccessPeer` is
true in both directions on this box).

The appliance profile therefore selects custom AR for the only topology it is
tuned for: `world_size == 2` with the input inside `max_size`. Everything else -
larger worlds, oversized inputs, misaligned or non-contiguous inputs, disabled
comms - must still be declined, and the AMD deterministic path must keep working.

The full `max_size` and byte-size semantics are verified against the real
communicator in the end-to-end AR probe, not here.
"""

from __future__ import annotations

import unittest

import torch

from sglang.srt.distributed.device_communicators.custom_all_reduce import (
    CustomAllreduce,
)
from sglang.srt.utils import is_hip


def _make_comm(
    *,
    world_size: int,
    max_size: int,
    full_nvlink: bool,
    deterministic: bool,
    disabled: bool = False,
) -> CustomAllreduce:
    """A CustomAllreduce with only the fields should_custom_ar() consults."""
    comm = CustomAllreduce.__new__(CustomAllreduce)
    comm.disabled = disabled
    comm.original_disabled = disabled
    comm._ptr = None  # keeps __del__/close() inert for a partially built object
    comm.world_size = world_size
    comm.max_size = max_size
    comm.full_nvlink = full_nvlink
    comm.use_amd_deterministic_impl = deterministic
    return comm


def _bytes_tensor(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes, dtype=torch.int8)


@unittest.skipUnless(is_hip(), "target policy is ROCm/HIP specific")
class TestTargetCustomArPolicy(unittest.TestCase):
    MAX_SIZE = 2 * 8192 * 1024

    def test_two_ranks_over_pcie_select_custom_ar(self):
        """The appliance case: ws=2, no XGMI, input within max_size."""
        comm = _make_comm(
            world_size=2,
            max_size=self.MAX_SIZE,
            full_nvlink=False,
            deterministic=False,
        )
        self.assertTrue(comm.should_custom_ar(_bytes_tensor(10240)))

    def test_two_ranks_at_exactly_max_size(self):
        comm = _make_comm(
            world_size=2,
            max_size=self.MAX_SIZE,
            full_nvlink=False,
            deterministic=False,
        )
        self.assertTrue(comm.should_custom_ar(_bytes_tensor(self.MAX_SIZE)))

    def test_two_ranks_above_max_size_declined(self):
        comm = _make_comm(
            world_size=2,
            max_size=self.MAX_SIZE,
            full_nvlink=False,
            deterministic=False,
        )
        self.assertFalse(comm.should_custom_ar(_bytes_tensor(self.MAX_SIZE + 16)))

    def test_larger_world_without_nvlink_declined(self):
        """We only tune the two-card profile; do not widen selection blindly."""
        comm = _make_comm(
            world_size=4,
            max_size=self.MAX_SIZE,
            full_nvlink=False,
            deterministic=False,
        )
        self.assertFalse(comm.should_custom_ar(_bytes_tensor(10240)))

    def test_larger_world_with_full_nvlink_still_selected(self):
        """Preserve upstream behaviour where XGMI is genuinely available."""
        comm = _make_comm(
            world_size=4,
            max_size=self.MAX_SIZE,
            full_nvlink=True,
            deterministic=False,
        )
        self.assertTrue(comm.should_custom_ar(_bytes_tensor(10240)))

    def test_deterministic_impl_unchanged(self):
        comm = _make_comm(
            world_size=2,
            max_size=self.MAX_SIZE,
            full_nvlink=False,
            deterministic=True,
        )
        self.assertTrue(comm.should_custom_ar(_bytes_tensor(10240)))

    def test_disabled_comm_declines(self):
        comm = _make_comm(
            world_size=2,
            max_size=self.MAX_SIZE,
            full_nvlink=False,
            deterministic=False,
            disabled=True,
        )
        self.assertFalse(comm.should_custom_ar(_bytes_tensor(10240)))

    def test_non_multiple_of_16_declined(self):
        comm = _make_comm(
            world_size=2,
            max_size=self.MAX_SIZE,
            full_nvlink=False,
            deterministic=False,
        )
        self.assertFalse(comm.should_custom_ar(_bytes_tensor(10241)))


if __name__ == "__main__":
    unittest.main()
