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
        accumulator = self.get_or_create_layer(
            layer_id, int(key.shape[1]), int(key.shape[2])
        )
        accumulator.update(key, positions)

    def get_or_create_layer(
        self, layer_id: int, kv_heads: int, head_dim: int
    ) -> KVMemMeanKAccumulator:
        """Return this layer's accumulator, creating or re-geometrying it.

        Geometry comes from the observed K tensor (TP-sharded), not from
        configuration: the two must agree or staging would raise mid-forward.
        """
        accumulator = self._layers.get(layer_id)
        if accumulator is None or (
            accumulator.kv_heads != kv_heads or accumulator.head_dim != head_dim
        ):
            accumulator = KVMemMeanKAccumulator(
                block_size=self.block_size,
                kv_heads=kv_heads,
                head_dim=head_dim,
            )
            self._layers[layer_id] = accumulator
        return accumulator

    def snapshot_layer(
        self, layer_id: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        accumulator = self._layers.get(layer_id)
        if accumulator is None:
            raise KeyError(f"layer {layer_id} has no captured K")
        return accumulator.snapshot()

    def conflicts(self, layer_id: int, positions: list[int]) -> bool:
        """True when these positions collide with existing layer state.

        A collision means the request slot was reused by a new request, so the
        stale accumulator must be dropped instead of raising.
        """
        accumulator = self._layers.get(layer_id)
        if accumulator is None:
            return False
        return accumulator.seen_any(positions)

    def reset_layer(self, layer_id: int) -> None:
        self._layers.pop(layer_id, None)

    def clear(self) -> None:
        self._layers.clear()
