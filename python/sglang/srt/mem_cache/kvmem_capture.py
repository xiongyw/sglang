"""Transactional K capture lifecycle for prefill/speculative experiments."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.mem_cache.kvmem_mean_k import KVMemMeanKAccumulator


class KVMemKCaptureSession:
    """Stage newly produced K rows and commit only accepted logical positions."""

    def __init__(self, accumulator: KVMemMeanKAccumulator):
        self.accumulator = accumulator
        self.active = False
        self._key: Optional[torch.Tensor] = None
        self._positions: Optional[torch.Tensor] = None

    def begin(self) -> None:
        if self.active:
            raise RuntimeError("K capture session is already active")
        self.active = True
        self._key = None
        self._positions = None

    def stage(self, key: torch.Tensor, positions: torch.Tensor) -> None:
        if not self.active:
            raise RuntimeError("begin() is required before stage()")
        if key.ndim != 3 or positions.ndim != 1 or key.shape[0] != positions.shape[0]:
            raise ValueError("key and positions have incompatible shapes")
        if self._key is not None:
            raise RuntimeError("a capture session accepts one staged K batch")
        self._key = key
        self._positions = positions

    def commit(self, accepted_tokens: int) -> None:
        self._require_active()
        if accepted_tokens < 0 or self._key is None or self._positions is None:
            raise ValueError("accepted_tokens is outside staged batch")
        if accepted_tokens > self._key.shape[0]:
            raise ValueError("accepted_tokens is outside staged batch")
        if accepted_tokens:
            self.accumulator.update(
                self._key[:accepted_tokens], self._positions[:accepted_tokens]
            )
        self._clear()

    def rollback(self) -> None:
        self._require_active()
        self._clear()

    def _require_active(self) -> None:
        if not self.active:
            raise RuntimeError("capture session is not active")

    def _clear(self) -> None:
        self.active = False
        self._key = None
        self._positions = None
