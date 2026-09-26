"""Pure host-side working-set selection primitives for KVMem experiments.

This module deliberately owns no KV storage and performs no device transfers.
It is the Phase-0 selector seam: logical blocks and scores in, deterministic
selected blocks and a residency diff out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True)
class KVMemBlock:
    block_id: int
    start: int
    n_tokens: int


@dataclass(frozen=True)
class KVMemSelectionConfig:
    budget_tokens: int
    sink_blocks: int = 1
    recent_blocks: int = 0


@dataclass(frozen=True)
class KVMemSelection:
    block_ids: list[int]
    total_tokens: int


@dataclass(frozen=True)
class KVMemResidencyDiff:
    stage_in: list[int]
    stage_out: list[int]
    reused: list[int]


class KVMemBlockSelector:
    """Select a bounded, chronologically ordered block working set."""

    def __init__(self, config: KVMemSelectionConfig):
        if config.budget_tokens <= 0:
            raise ValueError("budget_tokens must be positive")
        if config.sink_blocks < 0 or config.recent_blocks < 0:
            raise ValueError("sink_blocks and recent_blocks must be non-negative")
        self.config = config

    def select(
        self,
        blocks: Sequence[KVMemBlock],
        scores: Mapping[int, float],
    ) -> KVMemSelection:
        ordered = list(blocks)
        if any(block.n_tokens <= 0 for block in ordered):
            raise ValueError("block n_tokens must be positive")
        if len({block.block_id for block in ordered}) != len(ordered):
            raise ValueError("block_id values must be unique")

        mandatory_count = min(
            len(ordered), self.config.sink_blocks + self.config.recent_blocks
        )
        sink_count = min(self.config.sink_blocks, len(ordered))
        recent_start = max(sink_count, len(ordered) - self.config.recent_blocks)
        mandatory_ids = {
            block.block_id for block in ordered[:sink_count]
        } | {block.block_id for block in ordered[recent_start:]}
        mandatory = ordered[:sink_count] + ordered[recent_start:]

        if sum(block.n_tokens for block in mandatory) > self.config.budget_tokens:
            raise ValueError(
                "budget_tokens is smaller than mandatory sink/recent blocks"
            )

        remaining_budget = self.config.budget_tokens - sum(
            block.n_tokens for block in mandatory
        )
        candidates = [block for block in ordered if block.block_id not in mandatory_ids]
        candidates.sort(key=lambda b: (scores.get(b.block_id, 0.0), b.block_id), reverse=True)
        selected = list(mandatory)
        for block in candidates:
            if block.n_tokens <= remaining_budget:
                selected.append(block)
                remaining_budget -= block.n_tokens

        selected.sort(key=lambda b: b.start)
        return KVMemSelection(
            block_ids=[block.block_id for block in selected],
            total_tokens=sum(block.n_tokens for block in selected),
        )

    def diff(
        self,
        selection: KVMemSelection,
        resident_block_ids: Sequence[int],
    ) -> KVMemResidencyDiff:
        selected = set(selection.block_ids)
        resident = set(resident_block_ids)
        return KVMemResidencyDiff(
            stage_in=[block_id for block_id in selection.block_ids if block_id not in resident],
            stage_out=[block_id for block_id in resident_block_ids if block_id not in selected],
            reused=[block_id for block_id in selection.block_ids if block_id in resident],
        )
