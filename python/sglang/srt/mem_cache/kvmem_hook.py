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
from sglang.srt.model_executor.runner_utils import capture_mode as _capture_mode


@dataclass(frozen=True)
class KVMemHookConfig(KVMemRegistryConfig):
    enabled: bool = False


_registry: Optional[KVMemRequestRegistry] = None
_config: Optional[KVMemHookConfig] = None
_debug_stats: Optional[dict] = None
# Read-only selection probe: layer to probe (-1 = off) and call cadence.
_select_layer = -1
_select_every = 0
_select_calls = 0


def _refresh_debug_flags() -> None:
    global _select_layer, _select_every, _select_calls
    if os.environ.get("SGLANG_KVMEM_DEBUG_SELECT", "0") not in ("", "0", "false"):
        _select_layer = int(os.environ.get("SGLANG_KVMEM_SELECT_LAYER", 3))
        _select_every = max(1, int(os.environ.get("SGLANG_KVMEM_SELECT_EVERY", 32)))
    else:
        _select_layer = -1
        _select_every = 0
    _select_calls = 0


def get_kvmem_debug_stats() -> Optional[dict]:
    """Return capture counters, or None when the hook is off."""
    return _debug_stats


def set_kvmem_hook_config(config: KVMemHookConfig) -> None:
    """Install hook configuration; recreates the registry on change."""
    global _config, _registry, _debug_stats
    _config = config
    _refresh_debug_flags()
    if config.enabled:
        _debug_stats = {"captured_rows": 0, "committed_rows": 0}
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
        _debug_stats = None


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
    global _config, _registry, _debug_stats
    _config = None
    _registry = None
    _debug_stats = None
    _refresh_debug_flags()


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
    if _debug_stats is not None:
        _debug_stats["captured_rows"] += key.shape[0]
        if os.environ.get("SGLANG_KVMEM_DEBUG_STATS"):
            _debug_stats["stage_calls"] = _debug_stats.get("stage_calls", 0) + 1
            _debug_stats["last_stage_rows"] = int(key.shape[0])
            _debug_stats["last_stage_pos"] = (
                int(positions[0]),
                int(positions[-1]),
            )
    return True


def commit_attention_k(
    *, layer_id: int, accepted: Optional[Mapping[str, int]] = None
) -> bool:
    """Store staged rows for one layer; returns False when disabled.

    ``accepted=None`` stores every staged row. Spec-decode acceptance is not
    known where the batch is staged, so the engine stores all of them and lets
    logical-position identity discard the re-staged ones.
    """
    if not hook_enabled():
        return False
    accepted_map = None if accepted is None else dict(accepted)
    stored = get_kvmem_registry().commit_batch(
        layer_id=layer_id, accepted=accepted_map
    )
    if _debug_stats is not None:
        _debug_stats["committed_rows"] += int(stored)
    return True


def rollback_attention_k(*, layer_id: int) -> bool:
    """Discard staged rows for one layer; returns False when disabled."""
    if not hook_enabled():
        return False
    get_kvmem_registry().rollback_batch(layer_id=layer_id)
    return True


def _rows_per_request(forward_batch, n_rows: int) -> Optional[list[int]]:
    """Rows contributed by each request in this batch, or None if not exact."""
    mode = forward_batch.forward_mode
    rpi = getattr(forward_batch, "req_pool_indices", None)
    if rpi is None or n_rows <= 0:
        return None
    n_req = int(rpi.shape[0])
    if n_req <= 0:
        return None
    if mode.is_extend() and not mode.is_target_verify():
        lens = getattr(forward_batch, "extend_seq_lens_cpu", None)
        if lens is None or len(lens) != n_req:
            return None
        lens = [int(x) for x in lens]
        return lens if sum(lens) == n_rows else None
    # Decode and target-verify carry a uniform number of rows per request
    # (1 for decode, num_draft_tokens for verify).
    if n_rows % n_req:
        return None
    return [n_rows // n_req] * n_req


def _per_row_request_ids(
    forward_batch, n_rows: Optional[int] = None
) -> Optional[list[str]]:
    """Derive one request key per K row, or None when it cannot be exact.

    Requests are keyed by their ``req_pool_indices`` slot (the engine's stable
    per-request identity). ``rids`` is deliberately not used: it is empty on
    the DFlash2 appliance path. Inexact cases (capture mode, missing/uneven
    metadata, mismatched counts) return None so the caller skips capture.
    """
    if _capture_mode.is_capture_mode:
        return None
    positions = getattr(forward_batch, "positions", None)
    if n_rows is None:
        n_rows = 0 if positions is None else int(positions.shape[0])
    if n_rows <= 0:
        return None
    rpi = getattr(forward_batch, "req_pool_indices", None)
    if rpi is None:
        return None
    if positions is None or int(positions.shape[0]) != n_rows:
        return None
    lens = _rows_per_request(forward_batch, n_rows)
    if lens is None:
        return None
    keys: list[str] = []
    for slot, count in zip(rpi.tolist(), lens):
        keys.extend([f"slot:{int(slot)}"] * int(count))
    return keys if len(keys) == n_rows else None


def _log_gate_skip(layer_id: int, reason: str) -> None:
    import logging

    logging.getLogger(__name__).info(
        f"KVMem gate skip layer={layer_id}: {reason}"
    )


def _diag_fields(forward_batch, key) -> str:
    """Compact description of the row/request metadata at this seam."""
    mode = forward_batch.forward_mode
    mode_name = getattr(mode, "name", str(mode))
    rids = getattr(forward_batch, "rids", None)
    rpi = getattr(forward_batch, "req_pool_indices", None)
    pos = getattr(forward_batch, "positions", None)
    sl = getattr(forward_batch, "seq_lens", None)
    esc = getattr(forward_batch, "extend_seq_lens_cpu", None)
    return (
        f"mode={mode_name} key={tuple(key.shape)} "
        f"rids={0 if not rids else len(rids)} "
        f"rpi={0 if rpi is None else rpi.shape[0]} "
        f"pos={0 if pos is None else pos.shape[0]} "
        f"seq_lens={0 if sl is None else sl.shape[0]} "
        f"batch_size={getattr(forward_batch, 'batch_size', None)} "
        f"ext_lens={0 if esc is None else len(esc)}"
    )


def _capture_skip_reason(forward_batch, key) -> Optional[str]:
    if _capture_mode.is_capture_mode:
        return "capture_mode"
    n_rows = int(key.shape[0])
    if n_rows <= 0:
        return "no_rows"
    rpi = getattr(forward_batch, "req_pool_indices", None)
    if rpi is None:
        return "no_req_pool_indices"
    lens = _rows_per_request(forward_batch, n_rows)
    if lens is None:
        return "rows_per_request_inexact"
    ids = _per_row_request_ids(forward_batch, n_rows)
    if ids is None:
        return "row_mapping_failed"
    if len(ids) != n_rows:
        return f"id_row_mismatch {len(ids)}/{n_rows}"
    return None


def _slot_diag(forward_batch) -> str:
    """Slot list plus whether captured state exists for them (probe only)."""
    rpi = getattr(forward_batch, "req_pool_indices", None)
    if rpi is None:
        return "no_rpi"
    slots = [f"slot:{int(s)}" for s in rpi.tolist()]
    registry = _registry
    if registry is None:
        return f"{slots} registry=None"
    state = []
    for slot in slots:
        controller = registry._controllers.get(slot)
        layers = controller.state.layer_ids if controller is not None else []
        state.append(f"{slot}:layers={layers}")
    return f"{state}"


def _log_select_first_call(layer_id, query, q_heads, head_dim, forward_batch) -> None:
    """One-shot probe diagnostic: geometry and which slots carry state."""
    import logging

    logging.getLogger(__name__).info(
        f"KVMem select probe: first call layer={layer_id} "
        f"q_ndim={query.ndim} q_shape={tuple(query.shape)} "
        f"q_heads={q_heads} head_dim={head_dim} "
        f"slots={_slot_diag(forward_batch)}"
    )


def maybe_select_debug(
    *,
    layer_id: int,
    query: torch.Tensor,
    forward_batch,
    q_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> bool:
    """Rank captured blocks for the current query (read-only probe).

    Off unless SGLANG_KVMEM_DEBUG_SELECT is set; probes one layer and every
    SGLANG_KVMEM_SELECT_EVERY-th call. Never mutates KV state.
    """
    global _select_calls
    if _select_every == 0 or layer_id != _select_layer or not hook_enabled():
        return False
    if _capture_mode.is_capture_mode:
        # Host syncs are illegal while a graph is being captured.
        return False
    _select_calls += 1
    if _select_calls == 1:
        _log_select_first_call(layer_id, query, q_heads, head_dim, forward_batch)
    if _select_every > 1 and _select_calls % _select_every:
        return False
    if query.ndim == 2:
        if q_heads is None or head_dim is None:
            return False
        if query.shape[1] != q_heads * head_dim:
            return False
        query = query.view(query.shape[0], q_heads, head_dim)
    elif query.ndim != 3:
        return False

    from sglang.srt.mem_cache.kvmem_select_debug import (
        log_selection,
        select_debug,
    )

    results = select_debug(
        layer_id=layer_id, query=query, forward_batch=forward_batch
    )
    if results:
        log_selection(results, layer_id)
        return True
    return False


def maybe_capture_self_attention(
    *,
    layer_id: int,
    key: torch.Tensor,
    forward_batch,
    kv_heads: Optional[int] = None,
    head_dim: Optional[int] = None,
) -> bool:
    """Stage current K rows when the row<->request mapping is exact.

    Returns False (no-op) when the hook is disabled, when the mapping cannot be
    derived exactly (capture mode, padded/uneven batches, unknown spec mode), or
    when ``key`` cannot be reshaped with the given layer geometry.
    """
    if not hook_enabled():
        return False
    if os.environ.get("SGLANG_KVMEM_DEBUG_GATES"):
        reason = _capture_skip_reason(forward_batch, key)
        if reason:
            _log_gate_skip(layer_id, f"{reason} | {_diag_fields(forward_batch, key)}")
    request_ids = _per_row_request_ids(forward_batch, int(key.shape[0]))
    if request_ids is None or key.shape[0] != len(request_ids):
        return False
    if key.ndim == 2:
        # self_attention hands K as [tokens, kv_heads * head_dim] (TP-sharded).
        if kv_heads is None or head_dim is None:
            assert _config is not None
            kv_heads, head_dim = _config.kv_heads, _config.head_dim
        if key.shape[1] != kv_heads * head_dim:
            return False
        key = key.view(key.shape[0], kv_heads, head_dim)
    elif key.ndim != 3:
        return False
    return capture_attention_k(
        layer_id=layer_id,
        key=key,
        positions=forward_batch.positions,
        request_ids=request_ids,
    )
