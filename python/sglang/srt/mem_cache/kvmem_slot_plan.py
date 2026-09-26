"""Eager-side plan for a device-driven, graph-safe KV working set.

The problem this solves: decode steps on this appliance replay captured CUDA
graphs, so no Python runs and metadata cannot be rewritten per forward. What a
replayed graph *does* read is data. So the plan is computed eagerly (outside the
graph), written into stable device buffers, and consumed by the index build that
runs inside the graph.

Two pieces:

- ``plan_keep_blocks``: which logical blocks to keep for a request, monotonic.
  Eviction is one-way by design: an evicted block is not re-admitted, because
  re-admission would need the evicted K/V back and this path never keeps a copy.
- ``compacted_slots`` / ``KVMemPlanBuffer``: turn the keep set into the compacted
  list of KV slot ids for a request and publish it to device buffers whose layout
  matches what an index-building kernel wants (flat slots plus indptr).

Nothing here touches the allocator, the page table or the KV pool: it only
decides and publishes.
"""

from __future__ import annotations

from typing import Sequence

import torch


def plan_keep_blocks(
    *,
    history_len: int,
    block_size: int,
    selected_blocks: Sequence[int],
    budget_tokens: int,
    sink_blocks: int = 1,
    recent_blocks: int = 2,
    already_evicted: Sequence[int] = (),
) -> tuple[list[int], list[int]]:
    """Return ``(keep_positions, evicted_block_ids)`` for one request.

    Mandatory blocks (sink and the most recent ``recent_blocks``) are always
    kept, even when they alone exceed the budget: dropping them would break the
    request, not just its quality. Blocks in ``already_evicted`` are never
    re-admitted.
    """
    if history_len <= 0 or block_size <= 0:
        raise ValueError("history_len and block_size must be positive")
    if budget_tokens < 0:
        raise ValueError("budget_tokens must not be negative")
    n_blocks = (history_len + block_size - 1) // block_size
    evicted_before = {int(b) for b in already_evicted}

    sink = set(range(min(sink_blocks, n_blocks)))
    recent = set(range(max(0, n_blocks - recent_blocks), n_blocks))
    mandatory = sink | recent

    wanted: list[int] = []
    for block_id in selected_blocks:
        block_id = int(block_id)
        if block_id < 0 or block_id >= n_blocks:
            continue
        if block_id in evicted_before or block_id in mandatory:
            continue
        if block_id not in wanted:
            wanted.append(block_id)

    budget_blocks = max(len(mandatory), budget_tokens // block_size)
    room = max(0, budget_blocks - len(mandatory))
    keep_blocks = set(mandatory) | set(wanted[:room])

    keep_positions: list[int] = []
    for block_id in sorted(keep_blocks):
        start = block_id * block_size
        stop = min(start + block_size, history_len)
        keep_positions.extend(range(start, stop))

    evicted = [
        block_id
        for block_id in range(n_blocks)
        if block_id not in keep_blocks
    ]
    return keep_positions, evicted


def compacted_slots(
    page_table_row: torch.Tensor, history_len: int, keep_positions: Sequence[int]
) -> list[int]:
    """KV slot ids for the kept logical positions, in ascending position order.

    ``page_table_row`` is the request's ``req_to_token`` row. This only reads it;
    the page table itself is never modified, so KV-slot ownership and the free
    path stay intact.
    """
    if history_len < 0:
        raise ValueError("history_len must not be negative")
    positions = list(keep_positions)
    if positions and (
        positions[0] < 0
        or positions[-1] >= history_len
        or any(positions[i] >= positions[i + 1] for i in range(len(positions) - 1))
    ):
        raise ValueError("keep positions must be ascending and within history")
    if not positions:
        return []
    if page_table_row.numel() < history_len:
        raise ValueError("page table row is shorter than the history")
    row = page_table_row[:history_len]
    return [int(row[position]) for position in positions]


class KVMemPlanBuffer:
    """Stable device buffers holding per-request compacted slot lists.

    ``finalize`` writes the indptr so a kernel (inside the captured graph) can
    read ``slots[indptr[i]:indptr[i+1]]`` as request i's KV list.
    """

    def __init__(self, *, max_requests: int, max_kept_tokens: int, device="cuda"):
        if max_requests <= 0 or max_kept_tokens <= 0:
            raise ValueError("buffer dimensions must be positive")
        self.max_requests = int(max_requests)
        self.max_kept_tokens = int(max_kept_tokens)
        self.device = torch.device(device)
        self._slots = torch.zeros(
            (self.max_requests * self.max_kept_tokens,),
            dtype=torch.int64,
            device=self.device,
        )
        self._indptr = torch.zeros(
            (self.max_requests + 1,), dtype=torch.int32, device=self.device
        )
        self._pending: dict[int, list[int]] = {}

    def write(self, request_index: int, slots: Sequence[int]) -> None:
        """Stage one request's kept slot list (bounds-checked)."""
        if request_index < 0 or request_index >= self.max_requests:
            raise ValueError("request_index out of range")
        slots = [int(s) for s in slots]
        if len(slots) > self.max_kept_tokens:
            raise ValueError("plan exceeds the buffer's per-request capacity")
        self._pending[int(request_index)] = slots

    def finalize(self, batch_size: int) -> None:
        """Publish staged plans as the flat buffer plus indptr."""
        if batch_size < 0 or batch_size > self.max_requests:
            raise ValueError("batch_size out of range")
        lengths = [len(self._pending.get(i, ())) for i in range(batch_size)]
        if sum(lengths) > self._slots.numel():
            raise ValueError("plans exceed the buffer capacity")
        flat: list[int] = []
        for index in range(batch_size):
            flat.extend(self._pending.get(index, ()))
        if flat:
            self._slots[: len(flat)] = torch.tensor(
                flat, dtype=torch.int64, device=self.device
            )
        cumulative = [0]
        for length in lengths:
            cumulative.append(cumulative[-1] + length)
        self._indptr[: batch_size + 1] = torch.tensor(
            cumulative, dtype=torch.int32, device=self.device
        )

    def slots(self) -> torch.Tensor:
        return self._slots

    def indptr(self) -> torch.Tensor:
        return self._indptr
