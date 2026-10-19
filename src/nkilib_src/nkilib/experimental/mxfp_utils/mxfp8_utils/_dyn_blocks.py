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

"""Runtime block scheduling for LNC2 blockwise kernels.

Dynamic ``nl.fori_loop`` bounds must be shard-independent because the loop
synchronizes the two cores.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, List, Optional

import nki.isa as nisa
import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert

# This module is LNC2-only: the token-axis split assumes exactly two cores.
LNC_CORES = 2


class BlockSharding(Enum):
    """Kernel sharding mode."""

    SHARD_ON_BLOCK = "shard_on_block"
    SHARD_ON_FREE = "shard_on_free"


# Candidate row counts for a shortened trailing block, smallest first: a trailing
# block holding R real rows runs the smallest candidate >= R, capped at block_size.
TAIL_BLOCK_SIZES = (1024, 2048, 4096)


def tail_sizes_for(block_size: int) -> List[int]:
    return [size for size in TAIL_BLOCK_SIZES if size < block_size] + [block_size]


@dataclass
class BlockSchedule:
    """How ``n_blocks`` blocks divide into a static prefix and a dynamic tail.

    Attributes:
        n_blocks: Total block slots (``N``).
        block_size: Tokens per block.
        n_static: Blocks run in the statically unrolled loop.
        split_static_tail: Halve the last static block across the cores.
        can_split_halves: Whether a block can be halved along the token axis. False
            makes both cores run the unpaired last block in full.
        tail_sizes: Candidates from :data:`TAIL_BLOCK_SIZES`; empty when fully static.
        sharding: Kernel sharding mode.
    """

    n_blocks: int
    block_size: int
    n_static: int
    split_static_tail: bool
    can_split_halves: bool
    tail_sizes: List[int] = field(default_factory=list)
    sharding: BlockSharding = BlockSharding.SHARD_ON_BLOCK

    @property
    def n_dynamic(self) -> int:
        return self.n_blocks - self.n_static

    @property
    def static_whole(self) -> int:
        """Whole blocks the static loop walks: n_static less any half-split block."""
        return self.n_static - 1 if self.split_static_tail else self.n_static


@dataclass
class BlockRef:
    """One unit of work: a whole block, a token-half of one, or a shortened tail.

    Attributes:
        index: Which block -- a Python ``int``, or a (1, 1) SBUF int32 tensor when
            ``is_dynamic``. In the dynamic case the runner must address through it (a
            ``scalar_offset`` on a block-major view), never do arithmetic on it.
        rows: Tokens this call computes.
        block_size: The stride between blocks. Stays the full block size even when
            ``rows`` is smaller, so per-block stores keep addressing whole slots.
        tag: Name fragment unique to this emitted body, and to this core where the
            two cores share a block. Buffer names must include it: shared_hbm
            allocations alias across cores.
        row_base: Token offset of ``rows`` within the block.
        is_dynamic: Whether ``index`` is a runtime tensor.
        is_half: Whether this is a token-half of a block shared with the other core.
        next_index: The next block this core will own in the static loop, else None.
            Lets a runner prefetch. Always None in the dynamic region, whose body is
            traced once.
    """

    index: Any
    rows: int
    block_size: int
    tag: str
    row_base: int = 0
    is_dynamic: bool = False
    is_half: bool = False
    next_index: Optional[int] = None


def resolve_blocks(
    n_blocks: int,
    block_size: int,
    num_static_block: Optional[int] = None,
    has_dynamic: bool = False,
    can_split_halves: bool = True,
    sharding: BlockSharding = BlockSharding.SHARD_ON_BLOCK,
) -> BlockSchedule:
    """Resolve the static/dynamic block split.

    Args:
        n_blocks: Total block slots (``N``).
        block_size: Tokens per block.
        num_static_block: Requested static prefix length. None auto-picks
            ``n_blocks - LNC_CORES``. Clamped to
            ``[min(LNC_CORES, n_blocks), n_blocks]`` so a short schedule never
            addresses a nonexistent block; its parity is free, since an odd prefix
            half-splits its last block.
        has_dynamic: False (no valid-token count) makes the whole range static,
            leaving the caller's loop exactly as it was.
        can_split_halves: Whether a block can be halved along the token axis.
        sharding: Kernel sharding mode.

    Returns:
        BlockSchedule: the resolved split.
    """
    if sharding == BlockSharding.SHARD_ON_FREE:
        can_split_halves = False

    if has_dynamic:
        kernel_assert(
            block_size > 0 and (block_size & (block_size - 1)) == 0,
            f"block_size must be a power of two so the runtime block index is a shift "
            f"and the remainder a mask, got {block_size}",
        )
        min_static = min(LNC_CORES, n_blocks)
        requested_static = n_blocks - LNC_CORES if num_static_block is None else num_static_block
        n_static = max(min_static, min(requested_static, n_blocks))
    else:
        n_static = n_blocks

    return BlockSchedule(
        n_blocks=n_blocks,
        block_size=block_size,
        n_static=n_static,
        split_static_tail=has_dynamic and can_split_halves and n_static % LNC_CORES != 0,
        can_split_halves=can_split_halves,
        tail_sizes=_tail_sizes(block_size, has_dynamic, sharding),
        sharding=sharding,
    )


def _tail_sizes(block_size: int, has_dynamic: bool, sharding: BlockSharding) -> List[int]:
    """Candidate sizes for the unpaired last block.

    SHARD_ON_FREE gets a single full-size candidate: shortening only pays off if the
    runner narrows its own work to ``BlockRef.rows``, and the backward threads the full
    block size through its phases instead. Emitting the smaller candidates there would
    cost one extra identical block body each for nothing.
    """
    if not has_dynamic:
        return []
    if sharding == BlockSharding.SHARD_ON_FREE:
        return [block_size]
    return tail_sizes_for(block_size)


def _log2(value: int) -> int:
    return value.bit_length() - 1


class _BlockLoop:
    """Emits the static prefix and the runtime-length remainder.

    A class so the ``nl.fori_loop`` bodies are bound methods carrying the loop state
    rather than closures nested in the caller.
    """

    def __init__(self, schedule: BlockSchedule, num_valid_tokens, shard_id: int, sbm, runner):
        self.schedule = schedule
        self.num_valid_tokens = num_valid_tokens
        self.block_size = schedule.block_size
        self.half = schedule.block_size // LNC_CORES
        self.shard_id = shard_id
        self.sbm = sbm
        self.runner = runner
        self.shard_on_block = schedule.sharding == BlockSharding.SHARD_ON_BLOCK
        self.stride = LNC_CORES if self.shard_on_block else 1
        self.first_block = shard_id if self.shard_on_block else 0
        self.last_rows = 0
        self.last_tag = ""
        # Set by _plan().
        self.blk = None
        self.n_full = None
        self.last_selectors = ()
        self.last_blk = None

    def _tag(self, base: str) -> str:
        """Keep shard-on-free shared buffers on the same name."""
        return f"{base}{self.shard_id}" if self.shard_on_block else base

    def _scalar(self, name: str):
        return self.sbm.alloc_stack((1, 1), dtype=nl.int32, name=name, align=32)

    def _run(self, scope: str, ref: BlockRef) -> None:
        self.sbm.open_scope(name=scope)
        self.runner.process_block(ref, self.runner.prep_inputs(ref))
        self.sbm.close_scope()

    def run_static_prefix(self) -> None:
        """Whole blocks this core owns, then its half of an odd leftover."""
        for block_idx in range(self.first_block, self.schedule.static_whole, self.stride):
            next_index = block_idx + self.stride
            self._run(
                f"Block {block_idx}",
                BlockRef(
                    index=block_idx,
                    rows=self.block_size,
                    block_size=self.block_size,
                    tag=f"{block_idx}",
                    next_index=next_index if next_index < self.schedule.static_whole else None,
                ),
            )

        if self.schedule.split_static_tail:
            tail_idx = self.schedule.n_static - 1
            self._run(
                f"BlockHalf {tail_idx}_{self.shard_id}",
                BlockRef(
                    index=tail_idx,
                    rows=self.half,
                    block_size=self.block_size,
                    tag=f"{tail_idx}h{self.shard_id}",
                    row_base=self.shard_id * self.half,
                    is_half=True,
                ),
            )

    def _plan(self):
        """Derive the trip counts from the valid token count ``V``.

        Sets ``self.n_full``, ``self.last_selectors`` and ``self.last_blk`` -- all
        shard-independent (1, 1) int32 SBUF tensors.
        """
        block_size = self.block_size
        shift = _log2(block_size)

        valid = self._scalar("dynblk_valid")
        nisa.dma_copy(dst=valid, src=self.num_valid_tokens.reshape((1, 1)))

        # Valid tokens the dynamic region owns, i.e. past the static prefix.
        v_dyn_unbounded = self._scalar("dynblk_v_dyn_unbounded")
        nisa.tensor_scalar(
            dst=v_dyn_unbounded,
            data=valid,
            op0=nl.subtract,
            operand0=self.schedule.n_static * block_size,
            op1=nl.maximum,
            operand1=0,
        )
        v_dyn = self._scalar("dynblk_v_dyn")
        nisa.tensor_scalar(
            dst=v_dyn,
            data=v_dyn_unbounded,
            op0=nl.minimum,
            operand0=self.schedule.n_dynamic * block_size,
        )

        # Occupied blocks = ceil(v_dyn / block_size). The add and the shift are separate
        # ops: a chained op1 runs on a float intermediate, which a shift cannot take.
        rounded = self._scalar("dynblk_rounded")
        nisa.tensor_scalar(dst=rounded, data=v_dyn, op0=nl.add, operand0=block_size - 1)
        n_active = self._scalar("dynblk_active")
        nisa.tensor_scalar(dst=n_active, data=rounded, op0=nl.right_shift, operand0=shift)

        # Blocks this core runs at full size, and whether it also runs a shortened last
        # block. STRIDED pairs blocks across the cores, so only an ODD count leaves the
        # last block alone -- and only an unpaired block is worth shortening, since a
        # paired one runs alongside a full-size block and its core is not the
        # bottleneck either way. REPLICATED has no pairing, so the last occupied block
        # is always emitted separately; its sole candidate remains the full block size.
        n_full = self._scalar("dynblk_full")
        shorten = self._scalar("dynblk_shorten")
        if self.shard_on_block:
            nisa.tensor_scalar(dst=n_full, data=n_active, op0=nl.right_shift, operand0=_log2(LNC_CORES))
            nisa.tensor_scalar(dst=shorten, data=n_active, op0=nl.bitwise_and, operand0=LNC_CORES - 1)
        else:
            nisa.tensor_scalar(dst=n_full, data=n_active, op0=nl.subtract, operand0=1, op1=nl.maximum, operand1=0)
            nisa.tensor_scalar(dst=shorten, data=n_active, op0=nl.greater_equal, operand0=1)

        # Real tokens in that last block, in [1, block_size].
        preceding = self._scalar("dynblk_preceding")
        nisa.tensor_scalar(
            dst=preceding, data=n_active, op0=nl.subtract, operand0=1, op1=nl.multiply, operand1=block_size
        )
        rows_needed = self._scalar("dynblk_rows_needed")
        nisa.tensor_tensor(data1=v_dyn, data2=preceding, op=nl.subtract, dst=rows_needed)

        # n_active == 0 produces an in-bounds index that no selector uses.
        nisa.tensor_scalar(
            dst=self.last_blk,
            data=n_active,
            op0=nl.add,
            operand0=self.schedule.n_static - 1,
            op1=nl.minimum,
            operand1=self.schedule.n_blocks - 1,
        )

        # Size k is selected iff rows_needed > bound[k] and rows_needed <= bound[k+1].
        # Since rows_needed > bound[k+1] implies rows_needed > bound[k], that difference
        # of two `greater` results is the 0/1 selector. Gating on `shorten` leaves every
        # selector 0 when there is no unpaired last block.
        bounds = [0] + list(self.schedule.tail_sizes)
        greater = []
        for bound in bounds:
            flag = self._scalar(f"dynblk_gt{bound}")
            nisa.tensor_scalar(dst=flag, data=rows_needed, op0=nl.greater, operand0=bound)
            greater.append(flag)
        selectors = []
        for idx in range(len(self.schedule.tail_sizes)):
            sel = self._scalar(f"dynblk_sel{idx}")
            nisa.tensor_tensor(data1=greater[idx], data2=greater[idx + 1], op=nl.subtract, dst=sel)
            nisa.tensor_tensor(data1=sel, data2=shorten, op=nl.multiply, dst=sel)
            selectors.append(sel)
        self.n_full = n_full
        self.last_selectors = selectors

    def _full_iteration(self, _i):
        self._run(
            f"DynBlock {self.shard_id}",
            BlockRef(
                index=self.blk,
                rows=self.block_size,
                block_size=self.block_size,
                tag=self._tag("dyn"),
                is_dynamic=True,
            ),
        )
        nisa.tensor_scalar(dst=self.blk, data=self.blk, op0=nl.add, operand0=self.stride)

    def _last_block_iteration(self, _i):
        """The unpaired last occupied block, at the size selected for this body."""
        split = self.schedule.can_split_halves
        self._run(
            f"DynBlockLast {self.last_tag}",
            BlockRef(
                index=self.last_blk,
                rows=self.last_rows,
                block_size=self.block_size,
                tag=self.last_tag,
                row_base=self.shard_id * self.last_rows if split else 0,
                is_dynamic=True,
                is_half=split,
            ),
        )

    def run_dynamic(self) -> None:
        # Loop-carried values must outlive the setup scope.
        self.blk = self._scalar("dynblk_idx")
        self.last_blk = self._scalar("dynblk_last_idx")
        self.sbm.open_scope(name="DynBlockSetup")
        self._plan()
        n_full_reg = nisa.register_alloc()
        nisa.register_load(n_full_reg, self.n_full)
        last_regs = []
        for sel in self.last_selectors:
            reg = nisa.register_alloc()
            nisa.register_load(reg, sel)
            last_regs.append(reg)
        self.sbm.close_scope()

        nisa.memset(self.blk, value=self.schedule.n_static + self.first_block)
        nl.fori_loop(0, n_full_reg, self._full_iteration)

        # One body per candidate size, at most one selector ever 1. A fori_loop over a
        # 0/1 bound is the run-once-or-skip construct: NKI has no data-dependent branch.
        for size, reg in zip(self.schedule.tail_sizes, last_regs):
            self.last_rows = size // LNC_CORES if self.schedule.can_split_halves else size
            self.last_tag = self._tag(f"dynlast{size}_")
            nl.fori_loop(0, reg, self._last_block_iteration)


def run_with_dyn_blocks(
    schedule: BlockSchedule,
    num_valid_tokens: Optional[nl.NkiTensor],
    shard_id: int,
    sbm,
    runner,
) -> None:
    """Run every block that holds real tokens, across the two cores.

    Args:
        schedule: From :func:`resolve_blocks`.
        num_valid_tokens: int32 HBM tensor holding the number of leading rows that
            hold real tokens. Unused when ``schedule.n_dynamic == 0``.
        shard_id: This core's index.
        sbm: Active SBUF manager.
        runner: Supplies ``prep_inputs(ref)`` and ``process_block(ref, inputs)``.
    """
    loop = _BlockLoop(schedule, num_valid_tokens, shard_id, sbm, runner)
    loop.run_static_prefix()
    if schedule.n_dynamic > 0:
        loop.run_dynamic()
