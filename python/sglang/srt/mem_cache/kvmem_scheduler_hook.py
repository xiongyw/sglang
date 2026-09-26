"""Batch-end KVMem capture commit/rollback driven by ForwardMode.

Rows staged during forward N can only be judged at the start of forward N+1,
because spec-decode acceptance is known only after N's sampling. This module
therefore runs at the top of ModelRunner.forward and decides, per request:

- extend/prefill: every staged row is real context -> commit all;
- decode/verify: commit ``seq_lens`` growth since the previous forward, i.e.
  exactly the rows the engine actually accepted (rejected draft rows dropped);
- unknown modes: roll back.

Disabled hook makes every path here a no-op.
"""

from __future__ import annotations

import os

from sglang.srt.mem_cache.kvmem_hook import (
    hook_enabled,
    rollback_attention_k,
    set_kvmem_hook_config_from_env,
)

# Last observed context length per request key, used for the accept delta.
_last_seq_lens: dict[str, int] = {}


def maybe_enable_kvmem_hook() -> bool:
    """One-time env-based hook init; safe to call from any worker."""
    if os.environ.get("SGLANG_KVMEM_ENABLED", "0") not in ("", "0", "false"):
        set_kvmem_hook_config_from_env()
        return True
    return False


def _request_keys(forward_batch) -> tuple[list[str], list[int]]:
    rpi = getattr(forward_batch, "req_pool_indices", None)
    seq_lens = getattr(forward_batch, "seq_lens", None)
    if rpi is None or seq_lens is None:
        return [], []
    keys = [f"slot:{int(slot)}" for slot in rpi.tolist()]
    lens = [int(x) for x in seq_lens.tolist()]
    if len(keys) != len(lens):
        return [], []
    return keys, lens


def _is_supported_mode(forward_batch) -> bool:
    mode = forward_batch.forward_mode
    return mode.is_extend() or mode.is_decode() or mode.is_target_verify()


def _commits_all_rows(forward_batch) -> bool:
    mode = forward_batch.forward_mode
    return mode.is_extend() and not mode.is_target_verify()


def finish_kvmem_forward(forward_batch) -> None:
    """Commit or roll back all pending staged K captures for this forward."""
    if not hook_enabled():
        return
    registry = _get_registry()
    staged = dict(registry._staged)
    if not staged:
        return

    keys, lens = _request_keys(forward_batch)
    current = dict(zip(keys, lens))

    if not _is_supported_mode(forward_batch):
        for layer in sorted({layer for (_key, layer) in staged}):
            rollback_attention_k(layer_id=layer)
        _record_seq_lens(current)
        return

    commits_all = _commits_all_rows(forward_batch)
    accepted: dict[str, int] = {}
    for (request_key, _layer), n_staged in staged.items():
        if commits_all:
            accepted[request_key] = n_staged
            continue
        current_len = current.get(request_key)
        previous_len = _last_seq_lens.get(request_key)
        if current_len is None or previous_len is None:
            # First sight (or a reused slot): no reliable accept delta, so
            # nothing from this batch is committed.
            accepted[request_key] = 0
            continue
        accepted[request_key] = max(0, min(current_len - previous_len, n_staged))

    layers = sorted({layer for (_key, layer) in staged})
    for layer in layers:
        _commit_attention_k(layer_id=layer, accepted=accepted)
    _record_seq_lens(current)

    stats = _get_debug_stats()
    if stats is not None and os.environ.get("SGLANG_KVMEM_DEBUG_STATS"):
        stats["commit_calls"] = stats.get("commit_calls", 0) + 1
        call = stats["commit_calls"]
        if call <= 3 or sum(accepted.values()) > 0 or call % 100 == 0:
            _log_info(
                f"KVMem commit: calls={call} "
                f"stage_calls={stats.get('stage_calls', 0)} "
                f"staged_rows={stats['captured_rows']} "
                f"committed_rows={stats['committed_rows']} "
                f"accepted_now={sum(accepted.values())} "
                f"layers={len(layers)} "
                f"last_stage=({stats.get('last_stage_rows')} rows, "
                f"pos {stats.get('last_stage_pos')})"
            )


def _record_seq_lens(current: dict[str, int]) -> None:
    for key, length in current.items():
        _last_seq_lens[key] = length


def _get_registry():
    from sglang.srt.mem_cache.kvmem_hook import get_kvmem_registry

    return get_kvmem_registry()


def _get_debug_stats():
    from sglang.srt.mem_cache.kvmem_hook import get_kvmem_debug_stats

    return get_kvmem_debug_stats()


def _log_info(message: str) -> None:
    import logging

    logging.getLogger(__name__).info(message)


def _commit_attention_k(*, layer_id: int, accepted: dict[str, int]) -> None:
    from sglang.srt.mem_cache.kvmem_hook import commit_attention_k

    commit_attention_k(layer_id=layer_id, accepted=accepted)
