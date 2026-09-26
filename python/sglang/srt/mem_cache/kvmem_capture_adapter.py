"""Normalize attention K rows into per-request capture groups."""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class KVMemCaptureBatch:
    key: torch.Tensor
    positions: torch.Tensor
    request_ids: list[str]


@dataclass(frozen=True)
class KVMemCaptureRows:
    key: torch.Tensor
    positions: torch.Tensor


def group_k_capture_rows(
    batch: KVMemCaptureBatch,
) -> dict[str, KVMemCaptureRows]:
    """Group current K rows by request without changing row order."""
    if batch.key.ndim != 3:
        raise ValueError("key must have shape [tokens, kv_heads, head_dim]")
    if batch.positions.ndim != 1:
        raise ValueError("positions must be one-dimensional")
    n_rows = batch.key.shape[0]
    if batch.positions.shape[0] != n_rows or len(batch.request_ids) != n_rows:
        raise ValueError("key, positions, and request_ids must have equal rows")
    if n_rows == 0:
        return {}

    row_ids: OrderedDict[str, list[int]] = OrderedDict()
    seen: set[tuple[str, int]] = set()
    for row, (request_id, position) in enumerate(
        zip(batch.request_ids, batch.positions.tolist())
    ):
        if not request_id:
            raise ValueError("request_ids must be non-empty")
        key = (request_id, int(position))
        if key in seen:
            raise ValueError("duplicate logical position for one request")
        seen.add(key)
        row_ids.setdefault(request_id, []).append(row)

    return {
        request_id: KVMemCaptureRows(
            key=batch.key[rows],
            positions=batch.positions[rows],
        )
        for request_id, rows in row_ids.items()
    }
