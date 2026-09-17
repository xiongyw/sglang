"""AOT build decisions for the RX 7900 XTX specialization target."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

TARGET_ARCH = "gfx1100"
_ALLREDUCE_PREFIX = "csrc/allreduce/"


@dataclass(frozen=True)
class RdnaBuildPlan:
    """The build-time contract for the supported 7900 XTX architecture."""

    is_target: bool
    sources: tuple[str, ...]
    topk_dynamic_smem_bytes: int
    hipcc_flags: tuple[str, ...]
    cxx_flags: tuple[str, ...]


def make_build_plan(amdgpu_target: str, sources: Iterable[str]) -> RdnaBuildPlan:
    """Return the exact AOT plan for gfx1100, rejecting other architectures."""

    if amdgpu_target != TARGET_ARCH:
        raise ValueError(
            f"7900xtx-qwen38-27b expects {TARGET_ARCH}, got {amdgpu_target!r}"
        )
    return RdnaBuildPlan(
        is_target=True,
        sources=tuple(source for source in sources if not source.startswith(_ALLREDUCE_PREFIX)),
        topk_dynamic_smem_bytes=48 * 1024,
        hipcc_flags=("-DSGL_IS_RDNA",),
        cxx_flags=("-DSGL_IS_RDNA",),
    )
