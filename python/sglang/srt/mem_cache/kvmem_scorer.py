"""Query-conditioned Mean-K scoring for KVMem Phase 0/1 experiments."""

from __future__ import annotations

import torch


def mean_k_block_scores(
    query: torch.Tensor,
    key: torch.Tensor,
    *,
    block_size: int,
    scale: float | None = None,
) -> torch.Tensor:
    """Return summed softmax attention mass for each historical KV block.

    Args:
        query: ``[query_tokens, query_heads, head_dim]``.
        key: ``[key_tokens, kv_heads, head_dim]``.
        block_size: Logical KVMem block size in tokens. The final partial block
            is included and divided by its actual token count.
        scale: Optional attention-logit scale. Defaults to ``1/sqrt(head_dim)``.

    Query heads are mapped to KV heads by grouped-query attention, requiring
    ``query_heads % kv_heads == 0``. The scorer is intentionally separate from
    attention execution and is not used by the production path yet.
    """
    if query.ndim != 3 or key.ndim != 3:
        raise ValueError("query and key must have shape [tokens, heads, dim]")
    if query.shape[2] != key.shape[2]:
        raise ValueError("query and key head dimensions must match")
    if block_size <= 0:
        raise ValueError("block_size must be positive")
    if key.shape[0] == 0:
        return query.new_empty((0,))
    q_tokens, q_heads, head_dim = query.shape
    _, kv_heads, _ = key.shape
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if scale is None:
        scale = head_dim ** -0.5

    n_blocks = (key.shape[0] + block_size - 1) // block_size
    means = []
    for block_id in range(n_blocks):
        start = block_id * block_size
        end = min(start + block_size, key.shape[0])
        means.append(key[start:end].mean(dim=0))
    block_means = torch.stack(means, dim=0)  # [blocks, kv_heads, dim]
    return mean_k_block_scores_from_means(query, block_means, scale=scale)


def mean_k_block_scores_from_means(
    query: torch.Tensor,
    block_means: torch.Tensor,
    *,
    scale: float | None = None,
) -> torch.Tensor:
    """Score query tokens against precomputed ``[blocks, kv_heads, dim]`` means."""
    if query.ndim != 3 or block_means.ndim != 3:
        raise ValueError("query and block_means must be rank-3 tensors")
    if query.shape[2] != block_means.shape[2]:
        raise ValueError("query and block mean head dimensions must match")
    q_tokens, q_heads, head_dim = query.shape
    n_blocks, kv_heads, _ = block_means.shape
    if n_blocks == 0:
        return query.new_empty((0,))
    if q_heads % kv_heads:
        raise ValueError("query heads must be divisible by KV heads")
    if scale is None:
        scale = head_dim ** -0.5
    group = q_heads // kv_heads
    q_by_kv = query.reshape(q_tokens, kv_heads, group, head_dim)
    logits = torch.einsum("tkgd,bkd->tkgb", q_by_kv, block_means)
    logits = logits.permute(0, 2, 1, 3).reshape(q_tokens, q_heads, n_blocks)
    return torch.softmax(logits * scale, dim=-1).sum(dim=(0, 1))
