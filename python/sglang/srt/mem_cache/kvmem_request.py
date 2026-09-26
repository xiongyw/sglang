"""Experimental end-to-end KVMem request controller.

This composes the Phase-0 primitives without integrating them into SGLang's
scheduler or attention backend.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.mem_cache.kvmem_capture import KVMemKCaptureSession
from sglang.srt.mem_cache.kvmem_scorer import mean_k_block_scores_from_means
from sglang.srt.mem_cache.kvmem_selector import (
    KVMemBlock,
    KVMemBlockSelector,
    KVMemResidencyDiff,
    KVMemSelection,
    KVMemSelectionConfig,
)
from sglang.srt.mem_cache.kvmem_state import KVMemRequestState


@dataclass(frozen=True)
class KVMemRequestConfig:
    block_size: int
    max_tokens: int
    selection_budget_tokens: int
    kv_heads: int = 1
    head_dim: int = 1
    sink_blocks: int = 1
    recent_blocks: int = 0


@dataclass(frozen=True)
class KVMemSelectionResult:
    selection: KVMemSelection
    diff: KVMemResidencyDiff


class KVMemRequestController:
    """Compose KVMem primitives for one request in a testable lifecycle."""

    def __init__(self, config: KVMemRequestConfig):
        self.config = config
        self.state = KVMemRequestState(
            block_size=config.block_size, max_tokens=config.max_tokens
        )
        self.selector = KVMemBlockSelector(
            KVMemSelectionConfig(
                budget_tokens=config.selection_budget_tokens,
                sink_blocks=config.sink_blocks,
                recent_blocks=config.recent_blocks,
            )
        )
        self._captures: dict[int, KVMemKCaptureSession] = {}
        self._resident: dict[int, list[int]] = {}

    def begin_capture(self, layer_id: int) -> None:
        if layer_id in self._captures and self._captures[layer_id].active:
            raise RuntimeError("layer capture already active")
        # The accumulator's geometry is fixed at stage time from the observed
        # K tensor (TP-sharded), so it is deliberately not created here.
        self._captures[layer_id] = KVMemKCaptureSession(
            self.state._layers.get(layer_id)
        )
        self._captures[layer_id].begin()

    def stage_k(self, layer_id: int, key: torch.Tensor, positions: torch.Tensor) -> None:
        session = self._capture(layer_id)
        session.accumulator = self.state.get_or_create_layer(
            layer_id, int(key.shape[1]), int(key.shape[2])
        )
        session.stage(key, positions)

    def commit_capture(self, layer_id: int, accepted_tokens: int) -> None:
        self._capture(layer_id).commit(accepted_tokens)

    def rollback_capture(self, layer_id: int) -> None:
        self._capture(layer_id).rollback()

    def reset_layer(self, layer_id: int) -> None:
        """Drop layer state; used when a request slot is reused."""
        self.state.reset_layer(layer_id)
        self._captures.pop(layer_id, None)
        self._resident.pop(layer_id, None)

    def score_query(self, layer_id: int, query: torch.Tensor) -> torch.Tensor:
        means, counts = self.state.snapshot_layer(layer_id)
        if means.shape[0] == 0:
            raise KeyError(f"layer {layer_id} has no captured K")
        return mean_k_block_scores_from_means(query, means)

    def select(self, layer_id: int, scores: torch.Tensor) -> KVMemSelectionResult:
        means, counts = self.state.snapshot_layer(layer_id)
        blocks = [
            KVMemBlock(
                block_id=i,
                start=i * self.config.block_size,
                n_tokens=int(counts[i]),
            )
            for i in range(len(counts))
            if int(counts[i]) > 0
        ]
        score_map = {block.block_id: float(scores[block.block_id]) for block in blocks}
        selection = self.selector.select(blocks, score_map)
        resident = self._resident.get(layer_id, [])
        diff = self.selector.diff(selection, resident)
        self._resident[layer_id] = selection.block_ids
        return KVMemSelectionResult(selection, diff)

    def _capture(self, layer_id: int) -> KVMemKCaptureSession:
        try:
            return self._captures[layer_id]
        except KeyError as exc:
            raise RuntimeError("begin_capture is required") from exc
