"""Multi-request KVMem registry routing capture batches to controllers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import torch

from sglang.srt.mem_cache.kvmem_capture_adapter import (
    KVMemCaptureBatch,
    group_k_capture_rows,
)
from sglang.srt.mem_cache.kvmem_request import (
    KVMemRequestConfig,
    KVMemRequestController,
)


@dataclass(frozen=True)
class KVMemRegistryConfig:
    block_size: int
    max_tokens: int
    selection_budget_tokens: int
    kv_heads: int = 1
    head_dim: int = 1
    sink_blocks: int = 1
    recent_blocks: int = 0


class KVMemRequestRegistry:
    """Own per-request controllers and route one attention layer's batch."""

    def __init__(self, config: KVMemRegistryConfig):
        self.config = config
        self._controllers: dict[str, KVMemRequestController] = {}
        self._staged: dict[tuple[str, int], int] = {}

    def controller(self, request_id: str) -> KVMemRequestController:
        controller = self._controllers.get(request_id)
        if controller is None:
            controller = KVMemRequestController(
                KVMemRequestConfig(
                    block_size=self.config.block_size,
                    max_tokens=self.config.max_tokens,
                    selection_budget_tokens=self.config.selection_budget_tokens,
                    kv_heads=self.config.kv_heads,
                    head_dim=self.config.head_dim,
                    sink_blocks=self.config.sink_blocks,
                    recent_blocks=self.config.recent_blocks,
                )
            )
            self._controllers[request_id] = controller
        return controller

    def request_ids(self) -> list[str]:
        return sorted(self._controllers)

    def capture_batch(self, layer_id: int, batch: KVMemCaptureBatch) -> None:
        groups = group_k_capture_rows(batch)
        for request_id, rows in groups.items():
            controller = self.controller(request_id)
            # Overlapping positions are normal here (verify steps re-stage the
            # window), so staging never resets: the accumulator stores each
            # logical position once and drops the repeats.
            controller.begin_capture(layer_id)
            controller.stage_k(layer_id, rows.key, rows.positions)
            self._staged[(request_id, layer_id)] = rows.key.shape[0]

    def commit_batch(
        self, layer_id: int, accepted: Optional[Mapping[str, int]] = None
    ) -> int:
        """Store staged rows for one layer; return rows newly stored.

        ``accepted=None`` stores every staged row: spec-decode acceptance is not
        known when the batch is staged, and position identity in the accumulator
        keeps re-staged rows from double counting.
        """
        staged = {
            request_id: n
            for (request_id, lid), n in self._staged.items()
            if lid == layer_id
        }
        if not staged:
            raise RuntimeError(f"no active capture for layer {layer_id}")
        if accepted is not None:
            for request_id, n_staged in staged.items():
                n_accepted = accepted.get(request_id, 0)
                if not 0 <= n_accepted <= n_staged:
                    raise ValueError(
                        f"accepted {n_accepted} outside [0, {n_staged}] for {request_id}"
                    )
        stored = 0
        for (request_id, lid) in list(self._staged):
            if lid == layer_id:
                if accepted is None:
                    stored += self.controller(request_id).commit_capture(layer_id)
                else:
                    stored += self.controller(request_id).commit_capture(
                        layer_id, accepted.get(request_id, 0)
                    )
                del self._staged[(request_id, lid)]
        return stored

    def rollback_batch(self, layer_id: int) -> None:
        for (request_id, lid) in list(self._staged):
            if lid == layer_id:
                self.controller(request_id).rollback_capture(layer_id)
                del self._staged[(request_id, lid)]

    def remove_request(self, request_id: str) -> None:
        for (rid, layer_id) in list(self._staged):
            if rid == request_id:
                del self._staged[(rid, layer_id)]
        self._controllers.pop(request_id, None)

    def clear(self) -> None:
        self._controllers.clear()
        self._staged.clear()
