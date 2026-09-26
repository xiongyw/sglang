"""Read-only selection probe: rank captured blocks for the current query.

This exists to observe what the KVMem selector would pick on real captured
Mean-K, without touching attention. It never mutates KV state and never
influences the forward pass; it only reports scores and a selection.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional, Sequence

import torch

from sglang.srt.mem_cache.kvmem_selector import (
    KVMemBlock,
    KVMemBlockSelector,
    KVMemSelectionConfig,
)

logger = logging.getLogger(__name__)

# Rate-limited diagnostics for the read-only probe.
_diag_calls = 0


def _diag(message: str) -> None:
    """Log a probe skip reason, sampled so a long run stays readable."""
    global _diag_calls
    _diag_calls += 1
    if _diag_calls <= 3 or _diag_calls % 512 == 0:
        logger.info(f"KVMem select diag #{_diag_calls}: {message}")


@dataclass(frozen=True)
class KVMemSelectionDebug:
    request_key: str
    n_blocks: int
    scored_tokens: int
    scores: list[float]
    selected_block_ids: list[int]
    selected_tokens: int


def _request_keys_for_batch(forward_batch) -> list[str]:
    rpi = getattr(forward_batch, "req_pool_indices", None)
    if rpi is None:
        return []
    return [f"slot:{int(slot)}" for slot in rpi.tolist()]


def select_debug(
    *,
    layer_id: int,
    query: torch.Tensor,
    forward_batch,
    sink_blocks: int = 1,
    recent_blocks: int = 0,
) -> Optional[list[KVMemSelectionDebug]]:
    """Rank captured blocks per request for the given query tokens.

    ``query`` is ``[query_tokens, q_heads, head_dim]`` for one layer. Returns
    None when the hook is off, when no blocks were captured for this layer, or
    when the batch carries no request keys.
    """
    from sglang.srt.mem_cache.kvmem_hook import (
        get_kvmem_registry,
        hook_enabled,
    )

    if not hook_enabled():
        return None

    request_keys = _request_keys_for_batch(forward_batch)
    if not request_keys:
        return None

    registry = get_kvmem_registry()
    block_size = registry.config.block_size
    budget = registry.config.selection_budget_tokens
    selector = KVMemBlockSelector(
        KVMemSelectionConfig(
            budget_tokens=budget,
            sink_blocks=sink_blocks,
            recent_blocks=recent_blocks,
        )
    )

    query_cpu = query.detach().to(device="cpu", dtype=torch.float32)
    results: list[KVMemSelectionDebug] = []
    for key in dict.fromkeys(request_keys):
        controller = registry._controllers.get(key)
        if controller is None:
            _diag(f"no_controller key={key}")
            continue
        if layer_id not in controller.state.layer_ids:
            _diag(f"layer_absent key={key} layers={controller.state.layer_ids}")
            continue
        means, counts = controller.state.snapshot_layer(layer_id)
        if means.shape[0] == 0:
            _diag(f"empty_means key={key}")
            continue
        blocks = [
            KVMemBlock(
                block_id=i,
                start=i * block_size,
                n_tokens=int(counts[i]),
            )
            for i in range(int(counts.shape[0]))
            if int(counts[i]) > 0
        ]
        if not blocks:
            continue

        from sglang.srt.mem_cache.kvmem_scorer import (
            mean_k_block_scores_from_means,
        )

        scores = mean_k_block_scores_from_means(query_cpu, means)
        score_list = [float(x) for x in scores]
        score_map = {block.block_id: score_list[block.block_id] for block in blocks}
        selection = selector.select(blocks, score_map)
        results.append(
            KVMemSelectionDebug(
                request_key=key,
                n_blocks=len(blocks),
                scored_tokens=sum(block.n_tokens for block in blocks),
                scores=score_list,
                selected_block_ids=list(selection.block_ids),
                selected_tokens=selection.total_tokens,
            )
        )
    return results or None


def log_selection(results: Sequence[KVMemSelectionDebug], layer_id: int) -> None:
    """Emit one compact line per request; used by the env-gated probe."""
    for entry in results:
        head = entry.selected_block_ids[:12]
        logger.info(
            f"KVMem select: layer={layer_id} req={entry.request_key} "
            f"blocks={entry.n_blocks} tokens={entry.scored_tokens} "
            f"budget_used={entry.selected_tokens} "
            f"selected={head}{'...' if len(entry.selected_block_ids) > 12 else ''} "
            f"top_scores={[round(s, 5) for s in sorted(entry.scores, reverse=True)[:4]]}"
        )


def select_debug_enabled() -> bool:
    return os.environ.get("SGLANG_KVMEM_DEBUG_SELECT", "0") not in ("", "0", "false")
