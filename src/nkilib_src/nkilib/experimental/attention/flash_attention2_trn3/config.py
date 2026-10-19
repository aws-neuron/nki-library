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

"""Tuning knobs and walk geometry for the FlashAttention-2 forward kernel.

Pure trace-time integer math: how Sq and Sk are tiled and grouped into blocks, how the pipeline walks
those blocks, and what each step's operands actually span. Holding this apart from the compute keeps it
constructible -- and therefore unit-testable -- outside a tracing context.
"""

import nki.language as nl

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from .mixed_radix import radix_size, unflatten
from .remainder_stream import block_table

SQ_TILE = nl.tile_size.pmax  # Sq granule: partition axis of S/P and of the O accumulator
SV_TILE = nl.tile_size.pmax  # Sk granule of V's tiling; MM2's contraction, so P^T matches
D_TILE = nl.tile_size.pmax  # MM1's contraction granule

_MM_MOVING_FMAX = nl.tile_size.gemm_moving_fmax
_MAX_SV_TILES_PER_SK_TILE = _MM_MOVING_FMAX // SV_TILE


class StepInfo(nl.NKIObject):
    """Everything position-dependent about one step of the walk.

    A step is a ``(sk_blk, sq_tile, sk_tile)`` grid point. The walk is ceil-divided and therefore
    uniform, so a trailing block contributes steps whose tiles do not exist; ``exists`` is how a stage
    tells the difference. Built by :meth:`FlashAttentionGeometry.step_info`, never by hand.

    Attributes:
        index: the grid index, for :meth:`FlashAttentionFwd.get_buffer`.
        exists: False when this step's Sq or Sk tile is past the real sequence length. Stages emit
            nothing, so the step consumes a pipeline slot and no instructions.
        is_first_tile: first Sk tile of its point -- resets both reduce chains, and MM2's accumulate.
        is_last_tile: **last existing** Sk tile of its point, which on a short trailing block is not
            ``sk_tiles_per_blk - 1``. Reads out ``block_max`` / ``block_sum`` and fires the merge.
        sk_valid: free extent of this Sk tile, short of ``sk_tile`` only on the final one.
        sv_tiles_valid: V tiles this Sk tile spans, the last of which may be partial.
        sv_last_extent: rows of the final V tile, ``SV_TILE`` when whole. MM2 contracts over the V
            tile's partition axis, so this is the contraction count for that one matmul.
    """

    def __init__(self, index, exists, is_first_tile, is_last_tile, sk_valid, sv_tiles_valid, sv_last_extent):
        self.index = index
        self.exists = exists
        self.is_first_tile = is_first_tile
        self.is_last_tile = is_last_tile
        self.sk_valid = sk_valid
        self.sv_tiles_valid = sv_tiles_valid
        self.sv_last_extent = sv_last_extent


class FlashAttentionConfig(nl.NKIObject):
    """Tuning knobs for :func:`flash_attention_2`.

    Carries no shape: the kernel derives its geometry from the tensors it is handed, via
    :meth:`geometry`. Every knob is a count of the level below it, so extents are built
    multiplicatively and no inter-level divisibility can be violated by construction.

    Args:
        sq_tiles_per_blk: Sq tiles per Sq block -- how many accumulators and Q tiles stay resident.
            At ``ceil(Sq / 128)`` all of Q fits one block, so K and V are streamed once for the whole
            of Q instead of once per Sq block.
        sk_tiles_per_blk: Sk tiles per Sk block. Sets the KV DMA size, the softmax merge period, and
            the depth of both reduce chains at once. Larger means fewer points, so fewer per-point
            merges on Vector, at the cost of deeper scores/probs buffers.
        kv_buffer_count: Rotating-buffer depth of the K/V streams.
    """

    def __init__(self, sq_tiles_per_blk: int = 4, sk_tiles_per_blk: int = 4, kv_buffer_count: int = 2):
        kernel_assert(sk_tiles_per_blk > 0, f"sk_tiles_per_blk={sk_tiles_per_blk} must be positive")
        kernel_assert(sq_tiles_per_blk > 0, f"sq_tiles_per_blk={sq_tiles_per_blk} must be positive")
        kernel_assert(kv_buffer_count > 0, f"kv_buffer_count={kv_buffer_count} must be positive")
        self.sq_tiles_per_blk = sq_tiles_per_blk
        self.sk_tiles_per_blk = sk_tiles_per_blk
        self.kv_buffer_count = kv_buffer_count

    def geometry(self, D: int, Sq: int, Sk: int):
        """Resolve these knobs against a concrete shape.

        Args:
            D: Head dimension.
            Sq: Query sequence length. Arbitrary.
            Sk: Key/value sequence length. Arbitrary.

        Returns:
            FlashAttentionGeometry: the tiling, pipeline and buffer geometry for that shape.
        """
        return FlashAttentionGeometry(self, D=D, Sq=Sq, Sk=Sk)


class FlashAttentionGeometry(nl.NKIObject):
    """Tiling, pipeline and buffer geometry for one shape, derived from a :class:`FlashAttentionConfig`.

    Built by :meth:`FlashAttentionConfig.geometry`, not directly. Everything derived -- tile extents,
    the per-stage pipeline offsets and the buffer depths that follow from them -- is computed here,
    so the compute primitive reads geometry rather than recomputing it.

    Attributes:
        D, Sq, Sk: the shape this geometry was resolved against.
        sq_tiles_per_blk, sk_tiles_per_blk, kv_buffer_count: the tuning knobs, forwarded.
        sk_tile, sk_blk, sq_blk, d_tiles: derived tile and block extents.
        sk_blocks: one ``(tile_count, last_tile_extent)`` per Sk block, matching the K/V streams.
        num_sk_blks, num_sq_tiles, num_sq_blks: block and tile counts, ceil-divided.
        grid, num_steps: the ``(sk_blk, sq_tile, sk_tile)`` walk and its length in steps.
        scores_depth, probs_depth, probs_t_depth, stats_depth: pipeline slot counts.
        qk_offset, max_update_offset, exp_offset, tp_offset, pv_offset: steps each stage runs ahead of
            MM2. The merge shares ``pv_offset``: MM2's accumulation group stays open until it
            drains ``block_out``, so it cannot trail into the next point.
    """

    def __init__(self, config: FlashAttentionConfig, D: int, Sq: int, Sk: int):
        # D bounds MM2's moving free size, since MM2's moving operand is V.
        kernel_assert(
            D % D_TILE == 0 and 0 < D <= _MM_MOVING_FMAX,
            f"D={D} must be a positive multiple of {D_TILE} and at most {_MM_MOVING_FMAX}",
        )
        # Arbitrary lengths. Sq is padded up to a whole tile -- invalid Sq rows are never
        # written. Sk is cut at SV_TILE so no block mixes whole and partial V tiles.
        kernel_assert(Sq > 0, f"Sq={Sq} must be positive")
        kernel_assert(Sk > 0, f"Sk={Sk} must be positive")
        self.D = D
        self.Sq = Sq
        self.Sk = Sk
        self.sv_tiles_per_sk_tile = _MAX_SV_TILES_PER_SK_TILE
        self.sk_tiles_per_blk = config.sk_tiles_per_blk
        self.sq_tiles_per_blk = config.sq_tiles_per_blk
        self.kv_buffer_count = config.kv_buffer_count

        self.sk_tile = self.sv_tiles_per_sk_tile * SV_TILE
        self.sk_blk = self.sk_tiles_per_blk * self.sk_tile
        self.sq_blk = self.sq_tiles_per_blk * SQ_TILE
        self.d_tiles = D // D_TILE

        # No block may mix whole tiles with a partial one: V carries Sk on partitions, where that is not
        # a rectangle and cannot be one DMA. Each leftover becomes its own block.
        self.sk_blocks = block_table(Sk - Sk % SV_TILE, self.sk_tile, self.sk_tiles_per_blk)
        if Sk % SV_TILE > 0:
            self.sk_blocks.append((1, Sk % SV_TILE))
        self.num_sk_blks = len(self.sk_blocks)
        self.num_sq_tiles = div_ceil(Sq, SQ_TILE)
        self.num_sq_blks = div_ceil(self.num_sq_tiles, self.sq_tiles_per_blk)

        # Six stages over the flat walk of (sk_blk, sq_tile, sk_tile). A *point* is one
        # (sk_blk, sq_tile) and spans sk_tiles_per_blk steps. Each stage runs a fixed number of steps
        # ahead of MM2, so four points are in flight at once -- with t the point MM1 is working on:
        #
        #   t             0      1      2      3      4      5
        #   qk          (0,0)  (0,1)  (0,2)  (0,3)  (1,0)  (1,1)   4 points ahead of MM2
        #   max_update    .    (0,0)  (0,1)  (0,2)  (0,3)  (1,0)   3 points
        #   exp           .      .    (0,0)  (0,1)  (0,2)  (0,3)   2 points
        #   transpose     .      .      .      .    (0,0)  (0,1)   1 step
        #   pv, sm_merge  .      .      .      .    (0,0)  (0,1)   0  -- the reference point
        #
        # max_update fires on a point's first Sk tile and sm_merge (the softmax merge) on its last;
        # the others run every step. The three upstream offsets are whole Sk blocks so that no stage
        # boundary lands inside an open reduce chain (the max chains on Scalar, the row sum on Vector).
        # The prologue leads are fixed.
        self.pv_offset = 0
        self.tp_offset = 1
        self.exp_offset = 2 * self.sk_tiles_per_blk
        self.max_update_offset = 3 * self.sk_tiles_per_blk
        self.qk_offset = 4 * self.sk_tiles_per_blk

        # The carried state is per Sq tile, so a block must hold more Sq tiles than max_update trails MM1 by.
        kernel_assert(
            self.sq_tiles_per_blk > self.max_update_offset // self.sk_tiles_per_blk,
            f"sq_tiles_per_blk={self.sq_tiles_per_blk} must exceed the max_update lead of "
            f"{self.max_update_offset // self.sk_tiles_per_blk} Sq tiles",
        )

        # Slots per staged intermediate: the prologue holds a producer's whole lead before its consumer
        # starts, which is a deeper peak than the steady state, so it is the lead plus one.
        self.scores_depth = self.qk_offset + 1
        self.probs_depth = self.exp_offset + 1
        # tp_offset + 2, not + 1: probs_t is written a step before MM2 reads it, so the slot being
        # filled and the slot being consumed are both live, on top of the lead itself.
        self.probs_t_depth = self.tp_offset + 2
        # Per-block softmax stats
        self.stats_depth = self.qk_offset // self.sk_tiles_per_blk
        # computational tile grid
        self.grid = (self.num_sk_blks, self.sq_tiles_per_blk, self.sk_tiles_per_blk)
        # number of tiles in the grid
        self.num_steps = radix_size(self.grid)

    def step_info(self, step: int, sq_tiles_valid: int) -> StepInfo:
        """Resolve flat ``step`` into a :class:`StepInfo`.

        The single place that knows the walk is ceil-divided, so no stage reasons about remainders.

        Args:
            step: flat position in the walk.
            sq_tiles_valid: Sq tiles present in the block being computed. The trailing Sq block holds
                fewer than ``sq_tiles_per_blk``, and which block it is belongs to the caller.
        """
        index = unflatten(step, self.grid)
        sk_tile_idx = index[2]
        tiles_in_blk = self.sk_blocks[index[0]][0]
        is_last_tile = sk_tile_idx == tiles_in_blk - 1
        sk_valid = self.sk_blocks[index[0]][1] if is_last_tile else self.sk_tile
        sv_tiles_valid = div_ceil(sk_valid, SV_TILE)
        return StepInfo(
            index=index,
            exists=index[1] < sq_tiles_valid and sk_tile_idx < tiles_in_blk,
            is_first_tile=sk_tile_idx == 0,
            is_last_tile=is_last_tile,
            sk_valid=sk_valid,
            sv_tiles_valid=sv_tiles_valid,
            sv_last_extent=sk_valid - (sv_tiles_valid - 1) * SV_TILE,
        )
