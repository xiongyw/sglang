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
