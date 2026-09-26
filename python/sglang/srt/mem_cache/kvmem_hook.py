"""Opt-in attention hook wiring the KVMem registry into model execution.

Default state: disabled. When disabled, every entry point is a no-op and
model output is bit-identical to a build without this module.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping, Optional, Sequence

import torch

from sglang.srt.mem_cache.kvmem_capture_adapter import KVMemCaptureBatch
from sglang.srt.mem_cache.kvmem_registry import (
    KVMemRegistryConfig,
    KVMemRequestRegistry,
)
from sglang.srt.model_executor.runner import get_is_capture_mode


@dataclass(frozen=True)
class KVMemHookConfig(KVMemRegistryConfig):
    enabled: bool = False


_registry: Optional[KVMemRequestRegistry] = None
_config: Optional[KVMemHookConfig] = None


def set_kvmem_hook_config(config: KVMemHookConfig) -> None:
    """Install hook configuration; recreates the registry on change."""
    global _config, _registry
    _config = config
    if config.enabled:
        _registry = KVMemRequestRegistry(
            KVMemRegistryConfig(
                block_size=config.block_size,
                max_tokens=config.max_tokens,
                selection_budget_tokens=config.selection_budget_tokens,
                kv_heads=config.kv_heads,
                head_dim=config.head_dim,
                sink_blocks=config.sink_blocks,
                recent_blocks=config.recent_blocks,
            )
        )
    else:
        _registry = None


def set_kvmem_hook_config_from_env() -> KVMemHookConfig:
    """Build config from SGLANG_KVMEM_* environment variables."""
    enabled = os.environ.get("SGLANG_KVMEM_ENABLED", "0") not in ("", "0", "false")
    config = KVMemHookConfig(
        enabled=enabled,
        block_size=int(os.environ.get("SGLANG_KVMEM_BLOCK_SIZE", 128)),
        max_tokens=int(os.environ.get("SGLANG_KVMEM_MAX_TOKENS", 262144)),
        selection_budget_tokens=int(
            os.environ.get("SGLANG_KVMEM_BUDGET_TOKENS", 32768)
        ),
        kv_heads=int(os.environ.get("SGLANG_KVMEM_KV_HEADS", 4)),
        head_dim=int(os.environ.get("SGLANG_KVMEM_HEAD_DIM", 256)),
    )
    set_kvmem_hook_config(config)
    return config


def reset_kvmem_hook() -> None:
    """Disable the hook and drop all captured state."""
    global _config, _registry
    _config = None
    _registry = None


def hook_enabled() -> bool:
    return _config is not None and _config.enabled and _registry is not None


def get_kvmem_registry() -> KVMemRequestRegistry:
    if _registry is None:
        raise RuntimeError("KVMem hook is not enabled")
    return _registry


def capture_attention_k(
    *,
    layer_id: int,
    key: torch.Tensor,
    positions: torch.Tensor,
    request_ids: Sequence[str],
) -> bool:
    """Stage current K rows; returns False when the hook is disabled."""
    if not hook_enabled():
        return False
    batch = KVMemCaptureBatch(
        key=key, positions=positions, request_ids=list(request_ids)
    )
    get_kvmem_registry().capture_batch(layer_id=layer_id, batch=batch)
    return True


def commit_attention_k(*, layer_id: int, accepted: Mapping[str, int]) -> bool:
    """Commit accepted rows per request; returns False when disabled."""
    if not hook_enabled():
        return False
    get_kvmem_registry().commit_batch(layer_id=layer_id, accepted=dict(accepted))
    return True


def rollback_attention_k(*, layer_id: int) -> bool:
    """Discard staged rows for one layer; returns False when disabled."""
    if not hook_enabled():
        return False
    get_kvmem_registry().rollback_batch(layer_id=layer_id)
    return True


def _per_row_request_ids(forward_batch) -> Optional[list[str]]:
    """Derive one request id per K row, or None when it cannot be exact.

    Exact cases: decode with rids matching batch rows; extend with host-side
    start locations. Inexact cases (CUDA-graph capture/replay padding, missing
    rids, mismatched counts) return None so the caller skips capture.
    """
    if get_is_capture_mode():
        return None
    rids = getattr(forward_batch, "rids", None)
    if not rids:
        return None
    mode = forward_batch.forward_mode
    n_rows = int(forward_batch.seq_lens.shape[0])
    if mode.is_decode():
        if len(rids) != n_rows or (forward_batch.positions is None):
            return None
        if forward_batch.positions.shape[0] != n_rows:
            return None
        return list(rids)
    if mode.is_extend():
        start_loc = forward_batch.extend_start_loc
        seq_lens_cpu = forward_batch.extend_seq_lens_cpu
        if start_loc is None or seq_lens_cpu is None:
            return None
        if len(rids) != len(seq_lens_cpu):
            return None
        starts = start_loc[: len(seq_lens_cpu)].tolist()
        row_ids: list[str] = []
        for request_id, start, length in zip(rids, starts, seq_lens_cpu):
            row_ids.extend([request_id] * int(length))
        positions = forward_batch.positions
        if positions is None or positions.shape[0] != len(row_ids):
            return None
        return row_ids
    return None


def maybe_capture_self_attention(
    *,
    layer_id: int,
    key: torch.Tensor,
    forward_batch,
) -> bool:
    """Stage current K rows when the row<->request mapping is exact.

    No-op (returns False) when the hook is disabled or when the mapping
    cannot be derived exactly (graph capture/replay, padded batches,
    speculative modes without exact metadata).
    """
    if not hook_enabled():
        return False
    request_ids = _per_row_request_ids(forward_batch)
    if request_ids is None or key.shape[0] != len(request_ids):
        return False
    return capture_attention_k(
        layer_id=layer_id,
        key=key,
        positions=forward_batch.positions,
        request_ids=request_ids,
    )
