"""Pure planning/rewriting of the attention KV index list for KVMem selection.

The verify/decode attention reads K/V slots from a flat index list that the
backend fills from ``req_to_token``: for request i, ``kv_indices`` holds
``kv_indptr[i]:kv_indptr[i+1]`` in ascending logical-position order. Dropping
whole logical blocks therefore means shrinking that slice and recomputing
``kv_indptr``.

Causality is unaffected on the verify path: every query token in the batch sits
at a position >= the history length, so it attends to *all* history entries;
the appended current-chunk entries keep their order and count. Every function
here is pure so the rewrite can be unit-tested without a GPU.
"""

from __future__ import annotations

from typing import Sequence

import torch


def keep_positions_for_history(
    history_len: int,
    block_size: int,
    selected_blocks: Sequence[int],
    *,
    sink_blocks: int = 1,
    recent_blocks: int = 1,
) -> list[int]:
    """Logical positions of history to keep, ascending.

    Always keeps the sink block, the most recent ``recent_blocks`` blocks, and
    the partial tail block. ``budget >= history`` keeps everything, which makes
    the whole path an identity rewrite.
    """
    if history_len <= 0 or block_size <= 0:
        raise ValueError("history_len and block_size must be positive")
    if sink_blocks < 0 or recent_blocks < 0:
        raise ValueError("sink_blocks and recent_blocks must be non-negative")
    keep_blocks = {int(b) for b in selected_blocks}
    keep_blocks.update(range(min(sink_blocks, (history_len + block_size - 1) // block_size)))
    n_blocks = (history_len + block_size - 1) // block_size
    keep_blocks.update(range(max(0, n_blocks - recent_blocks), n_blocks))
    positions: list[int] = []
    for block_id in sorted(keep_blocks):
        if block_id < 0 or block_id >= n_blocks:
            continue
        start = block_id * block_size
        stop = min(start + block_size, history_len)
        positions.extend(range(start, stop))
    return positions


def remap_kv_indices(
    kv_indptr: torch.Tensor,
    kv_indices: torch.Tensor,
    keep_lists: Sequence[Sequence[int] | None],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Keep only the listed logical positions of each request's slice.

    ``keep_lists[i] is None`` leaves request i untouched. Positions must be
    ascending and within the request's history length; the result keeps the
    original relative order, so a "keep everything" list is an exact identity.
    Returns new ``(kv_indptr, kv_indices)`` tensors; inputs are not modified.
    """
    n_req = len(keep_lists)
    if kv_indptr.shape[0] < n_req + 1:
        raise ValueError("kv_indptr is shorter than the request count")
    device = kv_indices.device
    pieces: list[torch.Tensor] = []
    lengths: list[int] = []
    for index, keep in enumerate(keep_lists):
        start = int(kv_indptr[index])
        stop = int(kv_indptr[index + 1])
        if keep is None:
            piece = kv_indices[start:stop]
        else:
            history_len = stop - start
            positions = list(keep)
            if positions and (
                positions[0] < 0
                or positions[-1] >= history_len
                or any(
                    positions[i] >= positions[i + 1] for i in range(len(positions) - 1)
                )
            ):
                raise ValueError("keep positions must be ascending and in range")
            if not positions:
                piece = kv_indices[0:0]
            else:
                offsets = torch.tensor(positions, dtype=torch.long, device=device)
                piece = kv_indices[start:stop].index_select(0, offsets)
        pieces.append(piece)
        lengths.append(int(piece.shape[0]))
    new_indices = (
        torch.cat(pieces) if pieces else kv_indices[0:0]
    )
    new_indptr = torch.tensor(
        [0] + list(torch.tensor(lengths).cumsum(0).tolist()),
        dtype=kv_indptr.dtype,
        device=kv_indptr.device,
    )
    return new_indptr, new_indices


def reduction_ratio(history_len: int, kept: int) -> float:
    """Fraction of history kept; 1.0 means no reduction."""
    if history_len <= 0:
        return 1.0
    return kept / history_len
