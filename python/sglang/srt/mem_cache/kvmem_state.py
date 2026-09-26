"""Per-request, per-attention-layer KVMem state ownership."""

from __future__ import annotations

import torch

from sglang.srt.mem_cache.kvmem_mean_k import KVMemMeanKAccumulator


class KVMemRequestState:
    """Own independent Mean-K accumulators for one logical request."""

    def __init__(self, *, block_size: int, max_tokens: int):
        if max_tokens <= 0:
            raise ValueError("max_tokens must be positive")
        self.block_size = block_size
        self.max_tokens = max_tokens
        self._layers: dict[int, KVMemMeanKAccumulator] = {}

    @property
    def layer_ids(self) -> list[int]:
        return sorted(self._layers)

    def update_layer(
        self, layer_id: int, key: torch.Tensor, positions: torch.Tensor
    ) -> None:
        if layer_id < 0:
            raise ValueError("layer_id must be non-negative")
        if positions.numel() and int(positions.max()) >= self.max_tokens:
            raise ValueError("position exceeds request state capacity")
        accumulator = self._layers.get(layer_id)
        if accumulator is None:
            accumulator = KVMemMeanKAccumulator(
                block_size=self.block_size,
                kv_heads=key.shape[1],
                head_dim=key.shape[2],
            )
            self._layers[layer_id] = accumulator
        accumulator.update(key, positions)

    def snapshot_layer(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        accumulator = self._layers.get(layer_id)
        if accumulator is None:
            raise KeyError(f"layer {layer_id} has no captured K")
        return accumulator.snapshot()

    def clear(self) -> None:
        self._layers.clear()
