"""Transactional K capture lifecycle for prefill/speculative experiments."""

from __future__ import annotations

from typing import Optional

import torch

from sglang.srt.mem_cache.kvmem_mean_k import KVMemMeanKAccumulator


class KVMemKCaptureSession:
    """Stage newly produced K rows and commit only accepted logical positions."""

    def __init__(self, accumulator: Optional[KVMemMeanKAccumulator] = None):
        # Bound at stage time so the accumulator geometry follows the observed
        # (TP-sharded) K tensor rather than static configuration.
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

    def commit(self, accepted_tokens: Optional[int] = None) -> int:
        """Store staged rows and return how many were newly stored.

        ``accepted_tokens`` is optional: the production path passes nothing
        because spec-decode acceptance is unknown at staging time, and position
        identity (in ``update_idempotent``) prevents double counting. Tests and
        callers that *do* know the accepted count may still pass it to store a
        prefix only. Negative counts and counts beyond the staged batch raise.
        """
        self._require_active()
        if self._key is None or self._positions is None:
            raise ValueError("accepted_tokens is outside staged batch")
        n_staged = self._key.shape[0]
        n_accept: int = n_staged if accepted_tokens is None else int(accepted_tokens)
        if n_accept < 0 or n_accept > n_staged:
            raise ValueError("accepted_tokens is outside staged batch")
        stored = 0
        if n_accept:
            if self.accumulator is None:
                raise RuntimeError("no accumulator bound; stage() ran without one")
            stored = self.accumulator.update_idempotent(
                self._key[:n_accept], self._positions[:n_accept]
            )
        self._clear()
        return stored

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
