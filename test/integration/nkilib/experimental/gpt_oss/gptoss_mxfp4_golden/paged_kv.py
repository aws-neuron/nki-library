# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# SPDX-License-Identifier: Apache-2.0
"""
Paged KV cache for the standalone GPT-OSS decode golden.

Implements the production block-table / slot-mapping contract so the golden
doubles as a *paging correctness oracle*: a fragmented block table (blocks
scattered across the pool) must produce bit-identical logits to a contiguous
one, because attention indexes the gathered buffer by *logical* position.

Cache layout (one per attention layer):
    K_cache / V_cache : [num_blocks, kv_heads, block_size, head_dim]

Per layer the model holds its own K/V tensors (SWA layers may use a shorter
context window than full layers, exactly like production trims the block table).
For simplicity this manager allocates one shared pool geometry and hands out
per-layer caches; the block table + slot mapping are computed per step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import torch
from torch import Tensor


@dataclass
class SequenceState:
    """Per-sequence paging state: which physical blocks hold its tokens."""

    block_ids: List[int] = field(default_factory=list)  # physical block per logical block
    length: int = 0  # number of tokens currently cached


class PagedKVManager:
    """Allocates physical blocks and builds block tables + slot mappings.

    Usage:
        >>> mgr = PagedKVManager(num_blocks=64, block_size=16, kv_heads=8,
        ...                      head_dim=64, num_layers=2, dtype=torch.bfloat16)
        >>> mgr.allocate_sequence(seq_id=0, num_tokens=40)   # reserves ceil(40/16)=3 blocks
        >>> bt = mgr.block_table([0])                        # [1, max_blocks_per_seq]
        >>> slots = mgr.decode_slot_mapping([0])             # next-token write slots
    """

    def __init__(
        self,
        num_blocks: int,
        block_size: int,
        kv_heads: int,
        head_dim: int,
        num_layers: int,
        dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str = "cpu",
        shuffle_blocks: bool = False,
        seed: int = 0,
    ):
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.kv_heads = kv_heads
        self.head_dim = head_dim
        self.num_layers = num_layers
        self.dtype = dtype
        self.device = torch.device(device)

        # Free-block pool. Optionally shuffled so allocations are fragmented —
        # used by the paging-equivalence test.
        order = list(range(num_blocks))
        if shuffle_blocks:
            g = torch.Generator().manual_seed(seed)
            perm = torch.randperm(num_blocks, generator=g).tolist()
            order = perm
        self._free: List[int] = order

        # One K/V pair per layer.
        self.k_caches = [
            torch.zeros(num_blocks, kv_heads, block_size, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]
        self.v_caches = [
            torch.zeros(num_blocks, kv_heads, block_size, head_dim, dtype=dtype, device=device)
            for _ in range(num_layers)
        ]

        self.seqs: dict[int, SequenceState] = {}

    # ── Allocation ─────────────────────────────────────────────────────────

    def allocate_sequence(self, seq_id: int, num_tokens: int) -> SequenceState:
        """Reserve enough physical blocks to hold ``num_tokens`` and one more
        decode step (so the next-token slot is always available)."""
        n_blocks = (num_tokens + self.block_size - 1) // self.block_size
        # ensure room for the next decode token if it crosses a block boundary
        if num_tokens % self.block_size == 0:
            n_blocks += 1
        assert len(self._free) >= n_blocks, "out of KV blocks"
        blocks = [self._free.pop(0) for _ in range(n_blocks)]
        st = SequenceState(block_ids=blocks, length=num_tokens)
        self.seqs[seq_id] = st
        return st

    def ensure_capacity(self, seq_id: int) -> None:
        """Grow a sequence by one physical block if the next write would spill."""
        st = self.seqs[seq_id]
        needed = (st.length + 1 + self.block_size - 1) // self.block_size
        while len(st.block_ids) < needed:
            assert self._free, "out of KV blocks"
            st.block_ids.append(self._free.pop(0))

    # ── Table / slot construction ───────────────────────────────────────────

    def max_blocks_per_seq(self, seq_ids: List[int]) -> int:
        return max(len(self.seqs[s].block_ids) for s in seq_ids)

    def block_table(self, seq_ids: List[int], max_blocks: int | None = None) -> Tensor:
        """[B, max_blocks_per_seq] int32, -1 for unused block slots."""
        if max_blocks is None:
            max_blocks = self.max_blocks_per_seq(seq_ids)
        bt = torch.full((len(seq_ids), max_blocks), -1, dtype=torch.int32)
        for i, s in enumerate(seq_ids):
            ids = self.seqs[s].block_ids
            bt[i, : len(ids)] = torch.tensor(ids, dtype=torch.int32)
        return bt.to(self.device)

    def decode_slot_mapping(self, seq_ids: List[int]) -> Tensor:
        """Write slot for each sequence's *next* token = block*block_size+off.

        Advances each sequence length by one (call once per decode step, after
        ``ensure_capacity``).
        """
        slots = []
        for s in seq_ids:
            self.ensure_capacity(s)
            st = self.seqs[s]
            pos = st.length
            logical_block = pos // self.block_size
            off = pos % self.block_size
            phys = st.block_ids[logical_block]
            slots.append(phys * self.block_size + off)
            st.length += 1
        return torch.tensor(slots, dtype=torch.int64, device=self.device)

    def current_positions(self, seq_ids: List[int]) -> Tensor:
        """Absolute position of each sequence's next token (before it is written)."""
        return torch.tensor(
            [self.seqs[s].length for s in seq_ids],
            dtype=torch.int64,
            device=self.device,
        )

    def prefill_write(
        self,
        layer_idx: int,
        seq_id: int,
        k: Tensor,  # [num_tokens, kv_heads, head_dim]
        v: Tensor,
    ) -> None:
        """Directly write prefill K/V for a sequence into its blocks.

        Used to seed the cache when validating a real forward. ``k``/``v`` are
        in logical-position order [0, num_tokens).
        """
        st = self.seqs[seq_id]
        n = k.shape[0]
        kc, vc = self.k_caches[layer_idx], self.v_caches[layer_idx]
        for pos in range(n):
            lb = pos // self.block_size
            off = pos % self.block_size
            phys = st.block_ids[lb]
            kc[phys, :, off, :] = k[pos].to(self.dtype)
            vc[phys, :, off, :] = v[pos].to(self.dtype)
