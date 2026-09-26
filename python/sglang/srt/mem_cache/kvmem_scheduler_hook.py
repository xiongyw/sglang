"""Batch-end KVMem capture commit/rollback.

Runs at the top of ModelRunner.forward and settles whatever the previous
forward staged:

- extend/decode/verify: store every staged row. Spec-decode acceptance is not
  knowable at staging time, and ``forward_batch.seq_lens`` at this seam is not
  the context length for the DFlash2 path, so acceptance is not inferred at
  all: logical-position identity in the accumulator stores each position once
  and drops the re-staged rows (including rejected draft rows).
- unknown modes: roll back.

Disabled hook makes every path here a no-op.
"""

from __future__ import annotations

import os
from typing import Optional

from sglang.srt.mem_cache.kvmem_hook import (
    hook_enabled,
    rollback_attention_k,
    set_kvmem_hook_config_from_env,
)

def maybe_enable_kvmem_hook() -> bool:
    """One-time env-based hook init; safe to call from any worker."""
    if os.environ.get("SGLANG_KVMEM_ENABLED", "0") not in ("", "0", "false"):
        set_kvmem_hook_config_from_env()
        return True
    return False


def _is_supported_mode(forward_batch) -> bool:
    mode = forward_batch.forward_mode
    return mode.is_extend() or mode.is_decode() or mode.is_target_verify()


def _mode_name(forward_batch) -> str:
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return "?"
    for name in (
        "is_extend",
        "is_decode",
        "is_target_verify",
        "is_draft_extend",
        "is_idle",
    ):
        fn = getattr(mode, name, None)
        if fn is not None and fn():
            return name[3:]
    return getattr(mode, "name", str(mode))


def finish_kvmem_forward(forward_batch) -> Optional[int]:
    """Commit or roll back all pending staged K captures for this forward."""
    if not hook_enabled():
        return
    registry = _get_registry()
    staged = dict(registry._staged)
    if not staged:
        return

    if not _is_supported_mode(forward_batch):
        for layer in sorted({layer for (_key, layer) in staged}):
            rollback_attention_k(layer_id=layer)
        return

    layers = sorted({layer for (_key, layer) in staged})
    stored = 0
    for layer in layers:
        stored += _commit_attention_k(layer_id=layer) or 0

    stats = _get_debug_stats()
    if stats is not None and os.environ.get("SGLANG_KVMEM_DEBUG_STATS"):
        stats["commit_calls"] = stats.get("commit_calls", 0) + 1
        call = stats["commit_calls"]
        if call <= 3 or stored > 0:
            _log_info(
                f"KVMem commit: calls={call} mode={_mode_name(forward_batch)} "
                f"stage_calls={stats.get('stage_calls', 0)} "
                f"staged_rows={stats['captured_rows']} "
                f"committed_rows={stats['committed_rows']} "
                f"stored_now={stored} layers={len(layers)} "
                f"last_stage=({stats.get('last_stage_rows')} rows, "
                f"pos {stats.get('last_stage_pos')})"
            )
    return stored


def _get_registry():
    from sglang.srt.mem_cache.kvmem_hook import get_kvmem_registry

    return get_kvmem_registry()


def _get_debug_stats():
    from sglang.srt.mem_cache.kvmem_hook import get_kvmem_debug_stats

    return get_kvmem_debug_stats()


def _log_info(message: str) -> None:
    import logging

    logging.getLogger(__name__).info(message)


def _commit_attention_k(*, layer_id: int) -> Optional[int]:
    """Store every staged row for one layer; position identity dedupes them."""
    from sglang.srt.mem_cache.kvmem_hook import commit_attention_k

    return commit_attention_k(layer_id=layer_id)
