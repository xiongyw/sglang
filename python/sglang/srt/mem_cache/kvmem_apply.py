"""Apply a KVMem selection to the attention KV index list.

This is the step that makes selection real: after the backend has filled
``kv_indptr``/``kv_indices`` from ``req_to_token`` and before the model runs, the
index list of each selected request is rewritten to hold only the kept logical
blocks. The attention kernels then read a bounded working set instead of the
full history -- no data movement, no kernel changes.

Scope of this version (deliberately narrow, opt-in via SGLANG_KVMEM_APPLY=1):

- TARGET_VERIFY forwards only. That is where this appliance spends its context
  attention, and it uses standard causal masking with ``custom_mask=None``, so
  dropping whole history blocks cannot desynchronise a mask: every query token
  in the batch sits at a position >= the history length and therefore attends to
  all history entries either way.
- A request with no recorded selection is left completely untouched, so the path
  is an exact identity unless a selection exists.
- Never runs while a CUDA graph is being captured (the index buffers belong to
  the graph).
"""

from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import torch

from sglang.srt.mem_cache.kvmem_hook import (
    get_kvmem_registry,
    hook_enabled,
)
from sglang.srt.mem_cache.kvmem_index_remap import (
    keep_positions_for_history,
    remap_kv_indices,
)

logger = logging.getLogger(__name__)

_APPROX_SINK_BLOCKS = 1
_APPROX_RECENT_BLOCKS = 2
_apply_calls = 0
_last_report = {"history": 0, "kept": 0}


def apply_enabled() -> bool:
    """True when SGLANG_KVMEM_APPLY requests that selection shape attention."""
    return os.environ.get("SGLANG_KVMEM_APPLY", "0") not in ("", "0", "false")


def _mode_is_supported(forward_batch) -> bool:
    """True for batches whose KV window is prefix history only.

    On this build DFlash2 verify does not report ``is_target_verify()``: it
    arrives through the backend's generic extend path (``spec_info is None``,
    prefix = the context, queries = the draft tokens). Filtering the *prefix*
    part of an extend batch is causally safe for any extend batch: ``kv_indices``
    holds only prefix entries (positions < prefix_len) while every query token
    sits at a position >= prefix_len, so removing whole prefix blocks cannot
    change the causal relation between a query and the entries it keeps.
    """
    mode = getattr(forward_batch, "forward_mode", None)
    if mode is None:
        return False
    checker = getattr(mode, "is_target_verify", None)
    if checker is not None and checker():
        return True
    extend = getattr(mode, "is_extend", None)
    return bool(extend()) if extend is not None else False


def _capture_in_progress() -> bool:
    from sglang.srt.model_executor.runner_utils import capture_mode as _capture_mode

    return bool(_capture_mode.is_capture_mode)


_trace_calls = 0


def _trace_entry(forward_batch, attn_backend) -> None:
    """Log why an apply attempt did or did not proceed (diagnostics only)."""
    global _trace_calls
    _trace_calls += 1
    if _trace_calls > 40 and _trace_calls % 256:
        return
    mode = getattr(forward_batch, "forward_mode", None)
    names = [
        name
        for name in ("is_extend", "is_decode", "is_target_verify", "is_draft_extend")
        if getattr(mode, name, None) is not None and getattr(mode, name)()
    ]
    backend = attn_backend if attn_backend is not None else getattr(
        forward_batch, "attn_backend", None
    )
    metadata = getattr(backend, "forward_metadata", None)
    indptr = getattr(metadata, "kv_indptr", None)
    registry = get_kvmem_registry()
    rpi = getattr(forward_batch, "req_pool_indices", None)
    slots = [] if rpi is None else [f"slot:{int(s)}" for s in rpi.tolist()]
    recorded = {
        slot: registry.last_selection(slot) is not None for slot in slots
    }
    logger.info(
        f"KVMem apply trace #{_trace_calls}: mode={names or '?'} "
        f"backend={type(backend).__name__} capture={_capture_in_progress()} "
        f"indptr={None if indptr is None else int(indptr.shape[0])} "
        f"md_id={id(metadata)} "
        f"spec={type(getattr(forward_batch, 'spec_info', None)).__name__} "
        f"hist={_history_lens(indptr)} "
        f"slots={slots} recorded={recorded}"
    )


def _history_lens(indptr) -> str:
    """Per-request KV history lengths from kv_indptr, for diagnostics."""
    if indptr is None or int(indptr.shape[0]) < 2:
        return "[]"
    values = indptr.tolist()
    return str([int(values[i + 1] - values[i]) for i in range(len(values) - 1)])


def maybe_apply_kvmem_selection(forward_batch, attn_backend=None):
    """Rewrite the KV index list for selected requests; returns kept/history.

    Returns ``None`` when nothing was applied. Callers can ignore the result.
    """
    global _apply_calls
    if os.environ.get("SGLANG_KVMEM_DEBUG_APPLY"):
        _trace_entry(forward_batch, attn_backend)
    if not hook_enabled() or not apply_enabled():
        return None
    if _capture_in_progress() or not _mode_is_supported(forward_batch):
        return None

    backend = attn_backend if attn_backend is not None else getattr(
        forward_batch, "attn_backend", None
    )
    metadata = getattr(backend, "forward_metadata", None)
    if metadata is None:
        return None
    kv_indptr = getattr(metadata, "kv_indptr", None)
    kv_indices = getattr(metadata, "kv_indices", None)
    if kv_indptr is None or kv_indices is None:
        return None

    req_pool_indices = getattr(forward_batch, "req_pool_indices", None)
    if req_pool_indices is None:
        return None
    batch_size = int(req_pool_indices.shape[0])
    if batch_size <= 0 or int(kv_indptr.shape[0]) < batch_size + 1:
        return None

    registry = get_kvmem_registry()
    block_size = int(registry.config.block_size)
    keep_lists: list[Optional[Sequence[int]]] = []
    history_total = 0
    for index in range(batch_size):
        history_len = int(kv_indptr[index + 1]) - int(kv_indptr[index])
        history_total += history_len
        slot = int(req_pool_indices[index])
        recorded = registry.last_selection(f"slot:{slot}")
        if recorded is None:
            keep_lists.append(None)
            continue
        selected, budget_limited = recorded
        if not budget_limited:
            # Nothing was cut when this selection was made, so applying it now
            # (with a longer history than it saw) could drop unscored blocks.
            keep_lists.append(None)
            continue
        keep_lists.append(
            keep_positions_for_history(
                history_len,
                block_size,
                selected,
                sink_blocks=_APPROX_SINK_BLOCKS,
                recent_blocks=_APPROX_RECENT_BLOCKS,
            )
        )

    if all(keep is None for keep in keep_lists):
        return None

    try:
        new_indptr, new_indices = remap_kv_indices(
            kv_indptr[: batch_size + 1], kv_indices, keep_lists
        )
    except ValueError as exc:
        logger.warning(f"KVMem apply skipped: {exc}")
        return None

    metadata.kv_indptr = new_indptr
    metadata.kv_indices = new_indices

    kept = int(new_indices.shape[0])
    _apply_calls += 1
    _last_report["history"] = history_total
    _last_report["kept"] = kept
    if os.environ.get("SGLANG_KVMEM_DEBUG_APPLY"):
        # Sample hard on the small batches (they repeat every step) but always
        # report a batch once its history is real, so the big-context path is
        # never invisible.
        interesting = history_total >= 4 * block_size
        if _apply_calls <= 5 or interesting or _apply_calls % 256 == 0:
            histories = [
                int(kv_indptr[index + 1] - kv_indptr[index])
                for index in range(batch_size)
            ]
            applied = sum(1 for keep in keep_lists if keep is not None)
            ratio = kept / history_total if history_total else 1.0
            logger.info(
                f"KVMem apply: call={_apply_calls} bs={batch_size} "
                f"applied={applied} histories={histories} "
                f"history={history_total} kept={kept} ratio={ratio:.3f} "
                f"block={block_size} budget={registry.config.selection_budget_tokens}"
            )
    return kept, history_total


def last_apply_report() -> dict:
    """Last applied (kept, history) totals, for diagnostics."""
    return dict(_last_report)


def reset_apply_state() -> None:
    global _apply_calls
    _apply_calls = 0
    _last_report["history"] = 0
    _last_report["kept"] = 0
