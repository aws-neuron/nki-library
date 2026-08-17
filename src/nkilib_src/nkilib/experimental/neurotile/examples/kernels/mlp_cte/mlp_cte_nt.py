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
"""MLP CTE: single-rank MLP SwiGLU kernel using NeuroTile.

Computes ``y = (silu(x @ gate) * (x @ up)) @ down`` for a single rank with M-axis sharding for LNC.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.language import NKIObject

from nkilib_src.nkilib.experimental import neurotile as nt


@dataclass(frozen=True)
class MLPConfig(NKIObject):
    """Configuration for the single-rank MLP SwiGLU kernel.

    Args:
        K: Input hidden dimension (matmul reduction dim of gate/up).
        I: Intermediate dimension between gate/up and down projections.
        H: Output hidden dimension (matmul output dim of down projection).
        tile_m: M-dim tile size in elements.
        tile_k: K-dim tile size in elements.
        tile_i: I-dim tile size for gate/up matmuls in elements.
        tile_down_i: I-dim tile size for the down projection in elements.
        tile_h: H-dim tile size in elements.
        src_proj_buffer_count: Rotating-buffer count for gate/up weight stream.
        down_proj_buffer_count: Rotating-buffer count for down weight stream.
        bxs_subtile_count: Number of M-subtiles processed per loaded x block.
        num_cc_channels: Number of communication channels (reserved).
        dge_mode: Optional DMA gather/scatter mode passed to weight loads.

    Notes:
        Frozen=True is required for JAX tracing compatibility (kernel args
        must be hashable and immutable). Block-size fields (``K_BLK``,
        ``I_BLK``, ``H_BLK``, ``I_down_blk``) are derived in ``__post_init__``.
    """

    K: int
    I: int
    H: int
    tile_m: int = 128
    tile_k: int = 128
    tile_i: int = 512
    tile_down_i: int = 128
    tile_h: int = 512
    src_proj_buffer_count: int = 2
    down_proj_buffer_count: int = 2
    bxs_subtile_count: int = 2
    num_cc_channels: int = 1
    dge_mode: Optional[Any] = None
    K_BLK: int = field(init=False)
    I_BLK: int = field(init=False)
    H_BLK: int = field(init=False)
    I_down_blk: int = field(init=False)

    def __post_init__(self):
        assert self.I % self.tile_down_i == 0, (
            f"I={self.I} must be divisible by tile_down_i={self.tile_down_i}. "
            "Non-square transpose remainder not yet supported."
        )
        # Derived block sizes for nt.blocks() construction. Use object.__setattr__ since
        # frozen=True blocks ordinary attribute assignment in __post_init__.
        object.__setattr__(self, "K_BLK", nt.largest_divisor(self.K // self.tile_k, max_val=8))
        object.__setattr__(self, "I_BLK", nt.largest_divisor(nt.ceiling_div(self.I, self.tile_i), max_val=4))
        object.__setattr__(
            self,
            "H_BLK",
            nt.largest_divisor(self.H // self.tile_h, max_val=min(4, 8 // self.bxs_subtile_count)),
        )
        object.__setattr__(self, "I_down_blk", nt.largest_divisor(self.I // self.tile_down_i, max_val=4))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_bank_ids(count: int) -> list:
    """Build PSUM bank ID list [0, 1, ..., count-1]."""
    ids = []
    for bank_idx in range(count):
        ids.append(bank_idx)
    return ids


def transpose_in_place(block):
    """Transpose (tile_p x tile_p) subtiles in-place via PSUM bank pipelining.

    Re-tiles the block into square subtiles, transposes each through PSUM,
    and writes back to the same SBUF location. Interleaves across PSUM banks
    for pipeline efficiency, then bulk-copies each whole bank back to SBUF
    (one tensor_copy per bank, not per subtile).

    Args:
        block: NeuroTile SBUF block view; ``block.tile_size[0] == tile_size[1]``
            on the inner subtile grid is enforced by re-tiling.

    Returns:
        The same ``block`` (now containing the transposed subtiles).
    """
    PSUM_SLOT_SIZE = 512
    NUM_HW_BANKS = 8

    tile_p = block.tile_size[0]

    # Re-view the block as a grid of square (tile_p x tile_p) subtiles.
    subtiles = nt.tiles(block, tile_size=(tile_p, tile_p))
    n_rows = subtiles.shape[0]
    n_cols = subtiles.shape[1]

    # Pack subtiles across NUM_HW_BANKS PSUM banks; each bank holds tiles_per_bank slots.
    tiles_per_slot = PSUM_SLOT_SIZE // tile_p
    tiles_per_bank = min(n_cols, tiles_per_slot)
    num_banks = min(NUM_HW_BANKS, (n_cols + tiles_per_bank - 1) // tiles_per_bank)

    bank_ids = _make_bank_ids(num_banks)

    # Per row: fan transposes across banks/slots, then bulk-evict each bank with one tensor_copy.
    for row_idx in range(n_rows):
        """Allocate one wider PSUM tile per bank: tile holds tiles_per_bank slots side-by-side
        so the whole bank evicts in a single tensor_copy."""
        xpose_banks = nt.psum_pool(
            tile_size=(tile_p, tiles_per_bank * tile_p),
            grid=(num_banks, 1),
            bank_ids=tuple(bank_ids),
            dtype=block.dtype,
        )

        # Phase 1: SBUF -> PSUM transposes, fanned across banks (interleaved for pipelining).
        for slot_idx in range(tiles_per_bank):
            for bank_idx in range(num_banks):
                col_idx = bank_idx * tiles_per_bank + slot_idx
                if col_idx < n_cols:
                    nisa.nc_transpose(
                        xpose_banks[bank_idx, 0].data[:, nl.ds(slot_idx * tile_p, tile_p)],
                        subtiles[row_idx, col_idx].data,
                    )
        # Phase 2: bulk PSUM -> SBUF copy-back -- one tensor_copy per bank with the whole
        # bank addressed as a single contiguous F-slab on the SBUF side.
        for bank_idx in range(num_banks):
            col_start = bank_idx * tiles_per_bank
            cols_in_bank = min(tiles_per_bank, n_cols - col_start)
            nisa.tensor_copy(
                subtiles[row_idx, col_start : col_start + cols_in_bank].data,
                xpose_banks[bank_idx, 0].data[:, 0 : cols_in_bank * tile_p],
            )

    return block


# ---------------------------------------------------------------------------
# Stage 2 -- Gate/Up Projection + SwiGLU Activation
# ---------------------------------------------------------------------------


def compute_swiglu(x_block, gate, up, bias_vector, config: MLPConfig):
    """Compute ``silu(x @ gate) * (x @ up)`` with K-streamed weights.

    Streams weight K-blocks (dim 0) with rotating SBUF buffers for DMA/compute
    overlap. Partitions I-tiles into groups of MAX_I_PER_GROUP to fit PSUM budget.
    Loop order: I-group (outer) -> K-block stream -> BXS * K * I matmul (inner).
    PSUM reuse: gate -> silu evict -> up reuses same PSUMs -> multiply.

    Args:
        x_block: NeuroTile SBUF block view of the input ``[M, K]`` for one shard.
        gate: NeuroTile HBM block view of the gate weights ``[K, I]``.
        up: NeuroTile HBM block view of the up weights ``[K, I]``.
        bias_vector: SBUF tile providing the silu bias (typically zeros).
        config: Kernel configuration carrying tile/block sizes and DGE mode.

    Returns:
        SBUF tiles view of ``silu(x @ gate) * (x @ up)`` shaped ``[M, I]``.
    """
    assert x_block.is_blocked, "x_block must be a block view (need K-block iteration)"
    assert x_block.element_shape[1] == config.K, "x must span full K for correct accumulation"

    tile_m = x_block.tile_size[0]
    m_tiles = x_block.tile_shape[0]
    tile_i = gate.tile_size[1]
    dtype = gate.dtype
    I_tiles = gate.shape[1]

    # Split I-tiles into groups so (m_tiles * i_count) PSUM banks fit in the 8-bank budget.
    MAX_I_PER_GROUP = 8 // m_tiles
    has_i_remainder = (config.I % tile_i) != 0
    if has_i_remainder and I_tiles > 1:
        MAX_I_PER_GROUP = min(MAX_I_PER_GROUP, I_tiles - 1)
    I_groups = nt.ceiling_div(I_tiles, MAX_I_PER_GROUP)

    # Output buffer holds gate*up across all I-groups; written one slice per group below.
    gated_tiles = nt.alloc_tiles(
        tile_size=(tile_m, tile_i),
        buffer_type=nl.sbuf,
        dtype=dtype,
        element_shape=(x_block.element_shape[0], gate.element_shape[1]),
    )

    for i_group_idx in range(I_groups):
        i_start = i_group_idx * MAX_I_PER_GROUP
        i_count = min(MAX_I_PER_GROUP, I_tiles - i_start)
        gated_slice = gated_tiles[:, i_start : i_start + i_count]

        # slice gate/up weight cols for current i_group
        gate_cols = gate[:, i_start : i_start + i_count]
        up_cols = up[:, i_start : i_start + i_count]

        psum_bank_ids = _make_bank_ids(m_tiles * i_count)

        """Step 1: gate matmul -- accumulate x @ gate into PSUM, K-streamed for DMA overlap.
        Allocate psums; element_shape= will allocate partial remainder tiles when present."""
        psum_element_shape = (
            x_block.element_shape[0],
            gate_cols.element_shape[1],
        )
        gate_psums = nt.psum_pool(
            tile_size=(tile_m, tile_i),
            element_shape=psum_element_shape,
            bank_ids=psum_bank_ids,
            dtype=nl.float32,
        )

        gate_k_stream = gate_cols.stream(buffer_count=config.src_proj_buffer_count)
        n_k_blocks = x_block.shape[0]
        for k_block_idx in range(n_k_blocks):
            x_k_block = x_block[k_block_idx]
            gate_k_block = gate_k_stream.load(k_block_idx, dge_mode=config.dge_mode)
            gate_weight_tiles = nt.tiles(gate_k_block)
            x_k_block_tiles = nt.tiles(x_k_block)
            k_tiles = gate_weight_tiles.shape[0]
            for m_tile_idx in range(m_tiles):
                for k_tile_idx in range(k_tiles):
                    for i_tile_idx in range(i_count):
                        nisa.nc_matmul(
                            gate_psums[m_tile_idx, i_tile_idx].data,
                            x_k_block_tiles[m_tile_idx, k_tile_idx].data,
                            gate_weight_tiles[k_tile_idx, i_tile_idx].data,
                        )

        # Step 2: silu(gate) -> SBUF; evicting PSUM frees the same banks for up matmul.
        for m_tile_idx in range(m_tiles):
            for i_tile_idx in range(i_count):
                nisa.activation(
                    gated_slice[m_tile_idx, i_tile_idx].data,
                    op=nl.silu,
                    data=gate_psums[m_tile_idx, i_tile_idx].data,
                    bias=bias_vector,
                )

        # Step 3: up matmul -- accumulate x @ up into a fresh PSUM (reuses freed banks).
        up_psums = nt.psum_pool(
            tile_size=(tile_m, tile_i),
            element_shape=psum_element_shape,
            bank_ids=psum_bank_ids,
            dtype=nl.float32,
        )
        up_k_stream = up_cols.stream(buffer_count=config.src_proj_buffer_count)
        for k_block_idx in range(n_k_blocks):
            x_k_block = x_block[k_block_idx]
            up_k_block = up_k_stream.load(k_block_idx, dge_mode=config.dge_mode)
            up_weight_tiles = nt.tiles(up_k_block)
            x_k_block_tiles = nt.tiles(x_k_block)
            k_tiles = up_weight_tiles.shape[0]
            for m_tile_idx in range(m_tiles):
                for k_tile_idx in range(k_tiles):
                    for i_tile_idx in range(i_count):
                        nisa.nc_matmul(
                            up_psums[m_tile_idx, i_tile_idx].data,
                            x_k_block_tiles[m_tile_idx, k_tile_idx].data,
                            up_weight_tiles[k_tile_idx, i_tile_idx].data,
                        )

        # Step 4: silu(gate) * up, in-place into the output slice.
        for m_tile_idx in range(m_tiles):
            for i_tile_idx in range(i_count):
                nisa.tensor_tensor(
                    gated_slice[m_tile_idx, i_tile_idx].data,
                    gated_slice[m_tile_idx, i_tile_idx].data,
                    up_psums[m_tile_idx, i_tile_idx].data,
                    op=nl.multiply,
                )

    return gated_tiles


# ---------------------------------------------------------------------------
# Stage 3 -- Down Projection
# ---------------------------------------------------------------------------


def compute_down_matmul(gated_T_block, down, config: MLPConfig):
    """Down projection with I-streaming.

    Receives a transposed (m_tiles, I_down_tiles) block from
    :func:`transpose_in_place`. Streams down weights along I with rotating SBUF
    buffers for DMA/compute overlap.

    Args:
        gated_T_block: SBUF block view of the transposed gated activations
            with shape ``[I_down, M]`` per the layout produced by transpose.
        down: NeuroTile HBM block view of the down weights ``[I, H]``.
        config: Kernel configuration carrying tile/block sizes and DGE mode.

    Returns:
        SBUF blocks view of the output ``[M, H]`` shaped as
        ``(m_tiles, H_tiles)`` for one M-batch.
    """
    assert down.is_blocked, "down must be a block view"

    tile_i_down, tile_h = down.tile_size
    I_down_blk, H_BLK = down.block_size
    tile_m = gated_T_block.tile_size[0]
    H_cols = down.shape[1]

    gated_T = nt.tiles(gated_T_block, tile_size=(tile_i_down, tile_m))
    m_tiles = gated_T.tile_shape[0]

    down_psum_bank_ids = _make_bank_ids(m_tiles * H_BLK)

    out_block = nt.alloc_blocks(
        tile_size=(tile_m, tile_h),
        block_size=(m_tiles, H_BLK),
        grid=(1, H_cols),
        buffer_type=nl.sbuf,
        dtype=nl.bfloat16,
    )

    for h_col_idx in range(H_cols):
        down_col = down[:, h_col_idx]
        down_psums = nt.psum_pool(
            tile_size=(tile_m, tile_h),
            grid=(m_tiles, H_BLK),
            bank_ids=down_psum_bank_ids,
            dtype=nl.float32,
        )

        down_weight_stream = down_col.stream(buffer_count=config.down_proj_buffer_count)

        for i_block_idx in range(down_col.shape[0]):
            down_weight_block = down_weight_stream.load(i_block_idx, dge_mode=config.dge_mode)
            down_weight_tiles = nt.tiles(down_weight_block)
            i_tile_count = down_weight_tiles.shape[0]
            i_offset = i_block_idx * I_down_blk
            gated_T_chunk = gated_T[:, i_offset : i_offset + i_tile_count]

            for m_tile_idx in range(m_tiles):
                for i_tile_idx in range(i_tile_count):
                    for h_tile_idx in nl.affine_range(H_BLK):
                        nisa.nc_matmul(
                            down_psums[m_tile_idx, h_tile_idx].data,
                            gated_T_chunk[m_tile_idx, i_tile_idx].data,
                            down_weight_tiles[i_tile_idx, h_tile_idx].data,
                        )

        out_h = out_block[0, h_col_idx]
        for m_tile_idx in range(m_tiles):
            for h_tile_idx in nl.affine_range(H_BLK):
                nisa.tensor_copy(out_h[m_tile_idx, h_tile_idx].data, down_psums[m_tile_idx, h_tile_idx].data)

    return out_block


# ---------------------------------------------------------------------------
# Kernel -- Single-rank MLP (no allgather)
# ---------------------------------------------------------------------------


@nki.jit
def mlp_cte(
    x: nl.ndarray,
    gate_proj: nl.ndarray,
    up_proj: nl.ndarray,
    down_proj: nl.ndarray,
    config: MLPConfig,
) -> nl.ndarray:
    """Single-rank MLP SwiGLU kernel using NeuroTile.

    Computes ``y = (silu(x @ gate_proj) * (x @ up_proj)) @ down_proj`` for a
    single rank with M-axis sharding for LNC. Demonstrates NeuroTile's
    block/tile/stream abstractions for cache-friendly weight streaming and
    PSUM-bank pipelined transposes.

    TODO: Specify intended usage range (e.g., recommended H, K, I size bounds
    and BxS where this kernel performs best).

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension (input/output of the MLP)
        K: Input projection reduction dim (== H of input)
        I: Intermediate dimension between gate/up and down
        M: Flattened B * S

    Args:
        x (nl.ndarray): [B, S, H] input hidden states in HBM.
        gate_proj (nl.ndarray): [K, I] gate projection weights in HBM.
        up_proj (nl.ndarray): [K, I] up projection weights in HBM.
        down_proj (nl.ndarray): [I, H] down projection weights in HBM.
        config (MLPConfig): Kernel configuration (tile/block sizes, buffering).

    Returns:
        output (nl.ndarray): [B, S, H] MLP result in HBM.

    Notes:
        - LNC sharding partitions the M = B * S axis across cores; each core
          processes its slice of x_blocks and writes its slice of output_blocks.
        - x_blocks uses a 2-deep rotating stream so DMA of M-batch ``k+1``
          overlaps compute on batch ``k``.

    Pseudocode:
        x = x.reshape((B * S, H))
        for m_batch in shard(x_blocks):
            x_loaded = stream_load(x_blocks, m_batch)
            x_T = transpose_in_place(x_loaded)
            gated = silu(x_T @ gate_proj) * (x_T @ up_proj)
            gated_T = transpose_in_place(gated)
            out = gated_T @ down_proj
            output_blocks[m_batch] = out
        return output.reshape((B, S, H))
    """
    assert len(x.shape) == 3, "x must be [B, S, H]"
    B, S, H = x.shape
    M = B * S
    x = x.reshape((M, H))

    """Shard input/output along M for LNC. Slice the block view on dim 0
    (block-granular shard); ``block_size=(bxs_subtile_count, K_BLK)`` determines
    the unit count on the shard dim."""
    shard_range = nt.uneven_block_range(
        rank=nl.program_id(0),
        num_shards=nl.num_programs(0),
        total=nt.ceiling_div(M, config.tile_m * config.bxs_subtile_count),
    )

    output = nl.ndarray((M, H), dtype=x.dtype, buffer=nl.shared_hbm)

    x_blocks = nt.blocks(
        x,
        tile_size=(config.tile_m, config.tile_k),
        block_size=(config.bxs_subtile_count, config.K_BLK),
    )[shard_range, :]

    output_blocks = nt.blocks(
        output,
        tile_size=(config.tile_m, config.tile_h),
        block_size=(config.bxs_subtile_count, config.H // config.tile_h),
    )[shard_range, :]

    # Hoist per-batch invariant views and SBUF allocations.
    gate = nt.blocks(gate_proj, tile_size=(config.tile_k, config.tile_i), block_size=(config.K_BLK, 1))
    up = nt.blocks(up_proj, tile_size=(config.tile_k, config.tile_i), block_size=(config.K_BLK, 1))
    down = nt.blocks(
        down_proj,
        tile_size=(config.tile_down_i, config.tile_h),
        block_size=(config.I_down_blk, config.H_BLK),
    )

    bias_vector = nl.ndarray((config.tile_m, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(bias_vector, value=0.0)

    """x_blocks stream lives across batches: a 2-deep rotating SBUF prefetches
    batch k+1 while compute on batch k is in flight."""
    x_stream = x_blocks.stream(buffer_count=2)

    for m_batch_idx in range(x_blocks.shape[0]):
        x_loaded = x_stream.load(m_batch_idx, dge_mode=config.dge_mode)
        x_T = transpose_in_place(x_loaded)
        gated_tiles = compute_swiglu(x_T, gate, up, bias_vector, config)
        gated_T = transpose_in_place(gated_tiles)
        out_block = compute_down_matmul(gated_T, down, config)
        output_blocks[m_batch_idx].store(out_block.data)

    return output.reshape((B, S, H))
