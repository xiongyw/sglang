"""Persistent logical-position block mean-K accumulation."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class KVMemMeanKAccumulator:
    """Accumulate pre-RoPE or post-RoPE K means by logical token position."""

    block_size: int
    kv_heads: int
    head_dim: int

    def __post_init__(self) -> None:
        if self.block_size <= 0 or self.kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("block_size, kv_heads, and head_dim must be positive")
        self._sum = torch.empty((0, self.kv_heads, self.head_dim), dtype=torch.float32)
        self._count = torch.empty((0,), dtype=torch.long)
        self._seen: set[int] = set()

    def update(self, key: torch.Tensor, positions: torch.Tensor) -> None:
        if key.ndim != 3 or tuple(key.shape[1:]) != (self.kv_heads, self.head_dim):
            raise ValueError("key must have shape [tokens, kv_heads, head_dim]")
        if positions.ndim != 1 or positions.shape[0] != key.shape[0]:
            raise ValueError("positions must be one-dimensional and match key tokens")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use int32 or int64")
        if key.shape[0] == 0:
            return
        positions_cpu = positions.detach().to(device="cpu", dtype=torch.long).tolist()
        if len(set(positions_cpu)) != len(positions_cpu) or any(
            position in self._seen for position in positions_cpu
        ):
            raise ValueError("each logical position may be accumulated only once")
        max_block = max(positions_cpu) // self.block_size
        self._grow(max_block + 1)
        key_f32 = key.detach().to(device="cpu", dtype=torch.float32)
        for row, position in enumerate(positions_cpu):
            block_id = position // self.block_size
            self._sum[block_id] += key_f32[row]
            self._count[block_id] += 1
            self._seen.add(position)

    def seen_any(self, positions: list[int]) -> bool:
        """True if any logical position was already accumulated."""
        return any(position in self._seen for position in positions)

    def reset(self) -> None:
        """Drop all accumulated blocks (request slot reused by a new request)."""
        self._sum = torch.empty((0, self.kv_heads, self.head_dim), dtype=torch.float32)
        self._count = torch.empty((0,), dtype=torch.long)
        self._seen = set()

    def update_idempotent(self, key: torch.Tensor, positions: torch.Tensor) -> int:
        """Accumulate rows for positions not seen yet; return rows stored.

        Spec-decode acceptance is not known at staging time, so the engine
        commits every staged row and relies on position identity here: a row is
        stored once, at its first sighting, and later re-stagings of the same
        logical position are dropped. A batch that restarts at position 0 means
        the slot was reused by a new request, so the stale state is dropped.
        """
        if key.ndim != 3 or tuple(key.shape[1:]) != (self.kv_heads, self.head_dim):
            raise ValueError("key must have shape [tokens, kv_heads, head_dim]")
        if positions.ndim != 1 or positions.shape[0] != key.shape[0]:
            raise ValueError("positions must be one-dimensional and match key tokens")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use int32 or int64")
        if key.shape[0] == 0:
            return 0
        positions_cpu = positions.detach().to(device="cpu", dtype=torch.long).tolist()
        if (
            positions_cpu[0] == 0
            and len(positions_cpu) > 1
            and any(position in self._seen for position in positions_cpu)
        ):
            # A multi-row batch that restarts at 0 and overlaps what is already
            # stored is a new sequence on a reused slot: the old rows are stale.
            # Single-row staging at position 0 is a first token, not a restart.
            self.reset()
        fresh_rows = [
            row
            for row, position in enumerate(positions_cpu)
            if position not in self._seen
        ]
        if not fresh_rows:
            return 0
        fresh_positions = [positions_cpu[row] for row in fresh_rows]
        max_block = max(fresh_positions) // self.block_size
        self._grow(max_block + 1)
        key_f32 = key.detach().to(device="cpu", dtype=torch.float32)
        for row, position in zip(fresh_rows, fresh_positions):
            block_id = position // self.block_size
            self._sum[block_id] += key_f32[row]
            self._count[block_id] += 1
        self._seen.update(fresh_positions)
        return len(fresh_rows)

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Block means and counts.

        Sizing differs from the host accumulator by design: this one is
        preallocated to its full block capacity, while the host one grows to the
        blocks it has actually touched. Callers must therefore filter on
        ``counts > 0`` rather than assume the two shapes match.
        """
        means = self._sum.clone()
        nonzero = self._count > 0
        if torch.any(nonzero):
            means[nonzero] /= self._count[nonzero].to(torch.float32).view(-1, 1, 1)
        return means, self._count.clone()

    def _grow(self, size: int) -> None:
        if size <= self._sum.shape[0]:
            return
        extra = size - self._sum.shape[0]
        self._sum = torch.cat(
            [
                self._sum,
                torch.zeros((extra, self.kv_heads, self.head_dim), dtype=torch.float32),
            ],
            dim=0,
        )
        self._count = torch.cat(
            [self._count, torch.zeros((extra,), dtype=torch.long)], dim=0
        )


@dataclass
class KVMemDeviceMeanKAccumulator:
    """Preallocated device-resident block mean-K accumulator."""

    num_blocks: int
    block_size: int
    kv_heads: int
    head_dim: int
    device: str | torch.device = "cuda"

    def __post_init__(self) -> None:
        if self.num_blocks <= 0 or self.kv_heads <= 0 or self.head_dim <= 0:
            raise ValueError("accumulator dimensions must be positive")
        self.device = torch.device(self.device)
        self._sum = torch.zeros(
            (self.num_blocks, self.kv_heads, self.head_dim),
            dtype=torch.float32,
            device=self.device,
        )
        self._count = torch.zeros(
            (self.num_blocks,), dtype=torch.long, device=self.device
        )
        self._seen: set[int] = set()
        self._seen_positions: torch.Tensor | None = None

    def update(self, key: torch.Tensor, positions: torch.Tensor) -> None:
        if key.ndim != 3 or tuple(key.shape[1:]) != (self.kv_heads, self.head_dim):
            raise ValueError("key must have shape [tokens, kv_heads, head_dim]")
        if positions.ndim != 1 or positions.shape[0] != key.shape[0]:
            raise ValueError("positions must be one-dimensional and match key tokens")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use int32 or int64")
        if key.shape[0] == 0:
            return
        positions_cpu = positions.detach().to(device="cpu", dtype=torch.long).tolist()
        if len(set(positions_cpu)) != len(positions_cpu) or any(
            position in self._seen for position in positions_cpu
        ):
            raise ValueError("each logical position may be accumulated only once")
        block_ids_cpu = [position // self.block_size for position in positions_cpu]
        if any(block_id < 0 or block_id >= self.num_blocks for block_id in block_ids_cpu):
            raise ValueError("logical position exceeds preallocated block capacity")
        block_ids = torch.tensor(block_ids_cpu, dtype=torch.long, device=self.device)
        key_device = key.to(device=self.device, dtype=torch.float32)
        contiguous_full_blocks = (
            positions_cpu[-1] - positions_cpu[0] + 1 == len(positions_cpu)
            and positions_cpu[0] % self.block_size == 0
            and len(positions_cpu) % self.block_size == 0
        )
        if contiguous_full_blocks:
            first_block = positions_cpu[0] // self.block_size
            n_blocks = len(positions_cpu) // self.block_size
            block_means = key_device.reshape(
                n_blocks, self.block_size, self.kv_heads, self.head_dim
            ).mean(dim=1)
            self._sum[first_block : first_block + n_blocks] = (
                block_means * self.block_size
            )
            self._count[first_block : first_block + n_blocks] = self.block_size
        else:
            self._sum.index_add_(0, block_ids, key_device)
            self._count.index_add_(
                0, block_ids, torch.ones_like(block_ids, dtype=torch.long)
            )
        self._seen.update(positions_cpu)

    def reset(self) -> None:
        """Drop all accumulated blocks (request slot reused by a new request)."""
        self._sum.zero_()
        self._count.zero_()
        self._seen = set()
        self._seen_positions = None

    def seen_any(self, positions) -> bool:
        """True if any logical position was already accumulated."""
        seen = self._seen_positions
        if seen is None:
            return False
        return bool(seen[torch.as_tensor(list(positions), device=seen.device)].any())

    def _ensure_seen_positions(self, device) -> None:
        if self._seen_positions is None:
            self._seen_positions = torch.zeros(
                (self.num_blocks * self.block_size,), dtype=torch.bool, device=device
            )

    def update_idempotent(self, key: torch.Tensor, positions: torch.Tensor) -> int:
        """Accumulate rows for positions not seen yet; return rows stored.

        Device-side twin of the host accumulator's method, same restart rule: a
        multi-row batch that starts at 0 and overlaps what is already stored
        means the slot was reused by a new request. The rows never leave the
        device -- only the single overlapping/restart decision is read back --
        which is what keeps the capture path off the host.
        """
        if key.ndim != 3 or tuple(key.shape[1:]) != (self.kv_heads, self.head_dim):
            raise ValueError("key must have shape [tokens, kv_heads, head_dim]")
        if positions.ndim != 1 or positions.shape[0] != key.shape[0]:
            raise ValueError("positions must be one-dimensional and match key tokens")
        if positions.dtype not in (torch.int32, torch.int64):
            raise ValueError("positions must use int32 or int64")
        if key.shape[0] == 0:
            return 0
        capacity = self.num_blocks * self.block_size
        positions = positions.to(device=key.device, dtype=torch.long)
        if int(positions.min()) < 0 or int(positions.max()) >= capacity:
            raise ValueError("logical position exceeds preallocated capacity")

        self._ensure_seen_positions(key.device)
        assert self._seen_positions is not None
        seen_before = self._seen_positions[positions]
        if bool(seen_before.any()) and int(positions[0]) == 0 and positions.shape[0] > 1:
            # New sequence on a reused slot: the stored rows are stale.
            self.reset()
            self._ensure_seen_positions(key.device)
            assert self._seen_positions is not None
            seen_before = self._seen_positions[positions]

        fresh = ~seen_before
        n_fresh = int(fresh.sum())
        if n_fresh == 0:
            return 0
        fresh_positions = positions[fresh]
        key_fresh = key[fresh].to(device=self.device, dtype=torch.float32)
        block_ids = fresh_positions // self.block_size
        self._sum.index_add_(0, block_ids, key_fresh)
        self._count.index_add_(
            0, block_ids, torch.ones_like(block_ids, dtype=torch.long)
        )
        self._seen_positions[fresh_positions] = True
        self._seen.update(int(p) for p in fresh_positions.tolist())
        return n_fresh

    def snapshot(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Block means and counts.

        Sizing differs from the host accumulator by design: this one is
        preallocated to its full block capacity, while the host one grows to the
        blocks it has actually touched. Callers must therefore filter on
        ``counts > 0`` rather than assume the two shapes match.
        """
        means = self._sum.clone()
        nonzero = self._count > 0
        if torch.any(nonzero):
            means[nonzero] /= self._count[nonzero].to(torch.float32).view(-1, 1, 1)
        return means, self._count.clone()
