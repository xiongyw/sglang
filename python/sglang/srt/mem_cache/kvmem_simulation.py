"""GPU-free Phase-0 simulation for KVMem selection churn."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from sglang.srt.mem_cache.kvmem_selector import (
    KVMemBlock,
    KVMemBlockSelector,
)


@dataclass(frozen=True)
class KVMemSimulationStep:
    selected: list[int]
    stage_in: list[int]
    stage_out: list[int]
    reused: int
    total_tokens: int


class KVMemResidentSimulation:
    """Simulate selection/residency transitions without touching KV storage."""

    def __init__(self, selector: KVMemBlockSelector):
        self.selector = selector
        self.reset()

    def reset(self) -> None:
        self.resident_block_ids: list[int] = []
        self.total_steps = 0
        self.total_stage_in = 0
        self.total_stage_out = 0
        self.total_reused = 0

    def step(
        self,
        blocks: Sequence[KVMemBlock],
        scores: Mapping[int, float],
    ) -> KVMemSimulationStep:
        selection = self.selector.select(blocks, scores)
        diff = self.selector.diff(selection, self.resident_block_ids)
        self.resident_block_ids = list(selection.block_ids)
        self.total_steps += 1
        self.total_stage_in += len(diff.stage_in)
        self.total_stage_out += len(diff.stage_out)
        self.total_reused += len(diff.reused)
        return KVMemSimulationStep(
            selected=selection.block_ids,
            stage_in=diff.stage_in,
            stage_out=diff.stage_out,
            reused=len(diff.reused),
            total_tokens=selection.total_tokens,
        )
