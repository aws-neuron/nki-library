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

"""DeepSeek V3.2 specialized MLP kernel (MX prequantized packed block-scale input).

This is a fresh, self-contained kernel specialized for DeepSeek V3.2 shared-experts and
the first dense MLP layers.

  * Input is MX-prequantized packed hidden states exactly as ``rmsnorm_mx_prefill`` (with
    ``pack_scales=True``) emits them: a token row is ``[H fp8 bytes | scale_region uint8 bytes]``
    where ``scale_region = ceil(H/512/4)*128``.
  * No normalization, no bias, no fused-add.
  * MX (MXFP8) quantization only; weights + scales come from the checkpoint.

Performance intent:

  * When the full gate/up/down weights + scales fit in SBUF, they are hoisted (loaded once)
    before the token loop, so the token loop only DMAs input tiles and runs matmuls -- no
    per-tile weight reloads. This is the first-dense-MLP case (large T, small I).
  * When the full weights do NOT fit (shared-experts case: H=7168, I=2048, small T), the kernel
    instead hoists only the weight *scales* (always small) and streams the gate/up/down weight
    slices through small ring buffers, tiling over H (source) / I x H (down). The input hiddens
    are still loaded per token-tile. The mode is chosen automatically from an SBUF-budget check.
  * The input hidden tiles are double-buffered so tile N+1's DMA overlaps tile N's compute.

Compute per token tile:
    transpose fp8 hidden -> swizzle layout  (DMA transpose, fp32-reinterpret)
    nc_matmul_mx gate & up   (hidden = moving, weight = stationary; block scales / weight scales)
    SiLU(gate) * up
    quantize_mx intermediate
    nc_matmul_mx down
    -> bf16 output [T, H]

"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.mlp.mlp_cte.mlp_cte_constants import MAX_AVAILABLE_SBUF_SIZE
from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.common_types import ActFnType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import (
    NUM_HW_PSUM_BANKS,
    PSUM_BANK_SIZE,
    div_ceil,
    get_nl_act_fn_from_type,
    get_program_sharding_info,
)
from ...core.utils.logging import get_logger
from ...core.utils.tiled_range import TiledRange

# ---------------------------------------------------------------------------
# Fixed layout constants for the MX (MXFP8) DeepSeek path.
# ---------------------------------------------------------------------------
_PMAX = 128  # SBUF partition dim / matmul contraction rows per subtile
_Q = 4  # MX x4 packing width
_H512 = 512  # hidden elements per H-tile (matmul contraction tile)
_MX_BLOCK_32 = 32  # native MX scaling-group size (8 partitions x 4 x4-lanes = 32 contraction elems)
_SCALE_BLOCK_128 = 128  # DeepSeek compact scale block (one uint8 per 128x128 weight block)
_H_PACK = 4  # 512-tile / 128-block = 4 compact blocks (== 512/128); also the swizzle tiling factor
_SWIZZLE_RUN = _SCALE_BLOCK_128 // _H_PACK  # 32-column run within the gate/up (128,4)->(4,128) swizzle
_MX_PSUM_FMAX = 1024  # free dim of a PSUM bank for MX (bf16 accumulation)
_SRC_BXS_SUBTILE = 256  # wide token subtile used for the gate/up (source) projection
_DOWN_BXS_SUBTILE = 128  # token subtile used for down projection / intermediate quant
_FP32_FP8_RATIO = 4  # 4 fp8 bytes reinterpreted as 1 fp32 element for DMA transpose

# Internal token tile size (how many tokens each iteration processes). Kept modest so that,
# together with fully-hoisted weights + double-buffered input, everything fits in SBUF.
_TOKEN_TILE = 256

# Ring-buffer depth for the weight-tiling fallback path. Depth 2 lets the next weight slice's DMA
# overlap the current slice's matmul; the NKI scheduler handles the double-buffering.
_WEIGHT_RING_DEPTH = 2


def _fp8x4():
    return nl.float8_e4m3fn_x4


def _largest_divisor_at_most(n, cap):
    """Largest divisor of ``n`` not exceeding ``cap`` (used to pick a uniform I-group size)."""
    divisor = min(n, cap)
    while n % divisor != 0:
        divisor -= 1
    return divisor


# ---------------------------------------------------------------------------
# Weight / scale hoisting (loaded once, before the token loop).
# ---------------------------------------------------------------------------
def _load_full_src_weight(w_hbm, sbuf, H, I, n_H512, n_I512):
    """Load a full gate/up weight [128, H/512, I/512, 4, 128, 4] fp8 into SBUF.

    Per partition holds ``n_H512`` contiguous blocks, each of ``I*4`` fp8 bytes.
    """
    # HBM view: [128, H/512, I*4]  (I*4 == n_I512 * 4 * 128 * 4)
    w_hbm_view = w_hbm.reshape((_PMAX, n_H512, I * _Q))
    sbuf_view = sbuf.reshape((_PMAX, n_H512, I * _Q))
    nisa.dma_copy(dst=sbuf_view[:, :, :], src=w_hbm_view[:, :, :])


def _load_full_src_weight_scales(scale_hbm, sbuf, H, I, n_H512, n_I512, n_scale_packed, i_offset=0, i_local=None):
    """Load gate/up MX weight scales [16, H/512, I/512, 4, 128] into SBUF.

    Folds the 16 source rows into 128 partitions (4 rows per 32-partition quadrant)
    and packs 4 H-tiles per column slot.
    Result SBUF shape: [128, ceil(H/512 / 4), i_local].

    ``i_offset``/``i_local`` select an I-shard slice: each core loads only its I-half of the
    scales into a local-sized buffer (I-sharded weight-tiling path).
    """
    if i_local == None:
        i_local = I
    QUADRANT_SIZE = 32
    PARTITIONS_PER_SLOT = 4
    NUM_SLOTS = 4
    # scale HBM view: [16, H/512, I]  (I == I/512 * 4 * 128) -- full I, we slice the shard below.
    scale_hbm_view = scale_hbm.reshape((scale_hbm.shape[0], n_H512, I))
    for quadrant_idx in range(div_ceil(_PMAX, QUADRANT_SIZE)):
        for h_tile in TiledRange(H, _H512):
            slot_idx = h_tile.index % NUM_SLOTS
            nisa.dma_copy(
                dst=sbuf[
                    nl.ds(quadrant_idx * QUADRANT_SIZE + slot_idx * PARTITIONS_PER_SLOT, PARTITIONS_PER_SLOT),
                    h_tile.index // NUM_SLOTS,
                    :i_local,
                ],
                src=scale_hbm_view[
                    nl.ds(quadrant_idx * PARTITIONS_PER_SLOT, PARTITIONS_PER_SLOT),
                    h_tile.index,
                    nl.ds(i_offset, i_local),
                ],
            )


def _load_full_src_weight_scales_compact(
    scale_hbm, sbuf, staging_sbm, H, I, n_H512, n_I512, n_scale_packed, i_offset=0, i_local=None, name="src_scale"
):
    """Load COMPACT block-128 gate/up scales and expand them in-kernel to the native block-32 layout.

    Compact HBM shape ``[H/128, I/128]`` uint8 (one scale per 128-K x 128-N weight block). The
    result SBUF matches ``_load_full_src_weight_scales`` byte-for-byte: ``[128, ceil(H/512/4), i_local]``
    quadrant-folded native block-32 scales the matmul consumes - so nothing downstream changes.

    Two-stage:
      * Stage 1 - DMA the compact scales into a small scratch ``[128, n_scale_packed, i_local/128]``
        with the SAME quadrant fold as the native path. Each 512-K-tile's 4 quadrants are its 4
        compact K-blocks (quadrant ``q`` of h_tile ``k`` == compact K-row ``k*4 + q``); the 4
        partitions of a slot share one compact row via a stride-0 partition broadcast.
      * Stage 2 - Vector-engine broadcast the ``i_local/128`` compact N-blocks up to the full
        ``i_local`` physical columns, applying the gate/up ``(128,4)->(4,128)`` column swizzle:
        within each 512-N-tile the physical column is ``lane*128 + block*32 + run`` (lane 0..3,
        block 0..3, run 0..31), so a compact N-block fans out as a 32-run tiled 4x across lanes.
    """
    if i_local == None:
        i_local = I
    QUADRANT_SIZE = 32
    PARTITIONS_PER_SLOT = 4
    NUM_SLOTS = 4
    n_i128_local = i_local // _SCALE_BLOCK_128  # compact N-blocks in this shard
    n_i128_full = I // _SCALE_BLOCK_128
    n_i512_local = i_local // _H512
    i128_offset = i_offset // _SCALE_BLOCK_128

    staging_sbm.open_scope()
    compact = staging_sbm.alloc_stack(
        (_PMAX, n_scale_packed, n_i128_local), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name}_compact"
    )
    # ---- Stage 1: DMA compact scales into the quadrant-folded scratch. ----
    for quadrant_idx in range(div_ceil(_PMAX, QUADRANT_SIZE)):
        for h_tile in TiledRange(H, _H512):
            slot_idx = h_tile.index % NUM_SLOTS
            # Compact K-row for this (h_tile, quadrant): 4 compact K-blocks per 512-K-tile.
            compact_k_row = h_tile.index * _H_PACK + quadrant_idx
            nisa.dma_copy(
                dst=compact[
                    nl.ds(quadrant_idx * QUADRANT_SIZE + slot_idx * PARTITIONS_PER_SLOT, PARTITIONS_PER_SLOT),
                    h_tile.index // NUM_SLOTS,
                    :n_i128_local,
                ],
                src=scale_hbm.ap(
                    # [0, 4]: broadcast the single compact K-row across the slot's 4 partitions.
                    pattern=[[0, PARTITIONS_PER_SLOT], [1, n_i128_local]],
                    offset=compact_k_row * n_i128_full + i128_offset,
                    dtype=nl.uint8,
                ),
            )
    # ---- Stage 2: swizzled broadcast compact N-blocks -> full physical I columns. ----
    """
    One tensor_copy per 512-N-tile so the access pattern stays 5-D (the HW limit): within a
    512-N-tile the physical column is ``lane*128 + block*32 + run`` (lane/block 0..3, run 0..31),
    and the compact N-block for that column is ``it*4 + block`` (independent of lane and run).
    """
    for it in range(n_i512_local):
        src = (
            compact[:, :, nl.ds(it * _H_PACK, _H_PACK)]  # [128, n_packed, block]
            .expand_dim(dim=2)
            .broadcast(dim=2, size=_H_PACK)  # insert lane axis -> [128, n_packed, lane, block]
            .expand_dim(dim=4)
            .broadcast(dim=4, size=_SWIZZLE_RUN)  # insert run axis -> [128, n_packed, lane, block, run]
        )
        dst = sbuf[:, :, nl.ds(it * _H512, _H512)].reshape_dim(dim=2, shape=(_H_PACK, _H_PACK, _SWIZZLE_RUN))
        nisa.tensor_copy(dst=dst, src=src)
    staging_sbm.close_scope()


def _load_full_down_weight(w_hbm, sbuf, H, n_I512):
    """Load full down weight [128, I/512, H, 4] fp8 into SBUF (straight contiguous copy)."""
    w_hbm_view = w_hbm.reshape((_PMAX, n_I512, H, _Q))
    sbuf_view = sbuf.reshape((_PMAX, n_I512, H, _Q))
    nisa.dma_copy(dst=sbuf_view[:, :, :, :], src=w_hbm_view[:, :, :, :])


def _load_full_down_weight_scales(scale_hbm, sbuf, H, n_I512, i_tile_offset=0, n_I512_local=None):
    """Load down MX weight scales [16, I/512, H] into SBUF [128, n_I512_local, H].

    Folds 16 source rows into 128 partitions (4 rows per 32-partition quadrant).

    ``i_tile_offset``/``n_I512_local`` select an I-shard slice along the I/512 dim so each core
    loads only its I-half of the down scales (I-sharded weight-tiling path).
    """
    if n_I512_local == None:
        n_I512_local = n_I512
    QUADRANT_SIZE = 32
    PARTITIONS_PER_SLOT = 4
    for quadrant_idx in range(div_ceil(_PMAX, QUADRANT_SIZE)):
        nisa.dma_copy(
            dst=sbuf[nl.ds(quadrant_idx * QUADRANT_SIZE, PARTITIONS_PER_SLOT), :n_I512_local, :H],
            src=scale_hbm[
                nl.ds(quadrant_idx * PARTITIONS_PER_SLOT, PARTITIONS_PER_SLOT),
                nl.ds(i_tile_offset, n_I512_local),
                :H,
            ],
        )


def _load_full_down_weight_scales_compact(
    scale_hbm, sbuf, staging_sbm, H, n_I512, i_tile_offset=0, n_I512_local=None, name="down_scale"
):
    """Load COMPACT block-128 down scales and expand in-kernel to the native block-32 layout.

    Compact HBM shape ``[I/128, H/128]`` uint8. Result SBUF matches
    ``_load_full_down_weight_scales`` byte-for-byte: ``[128, n_I512_local, H]``, quadrant-folded.
    Down has NO column swizzle, so the H fan-out is a plain 128-contiguous repeat. Along I each
    512-tile's 4 quadrants are its 4 compact I-blocks (all 4 partitions of a quadrant share one
    compact I-row via a stride-0 partition broadcast).
    """
    if n_I512_local == None:
        n_I512_local = n_I512
    QUADRANT_SIZE = 32
    PARTITIONS_PER_SLOT = 4
    n_h128 = H // _SCALE_BLOCK_128

    staging_sbm.open_scope()
    compact = staging_sbm.alloc_stack(
        (_PMAX, n_I512_local, n_h128), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name}_compact"
    )
    # ---- Stage 1: DMA compact scales into the quadrant-folded scratch. ----
    for quadrant_idx in range(div_ceil(_PMAX, QUADRANT_SIZE)):
        # Compact I-row for (512-tile, quadrant): 4 compact I-blocks per 512-I-tile.
        compact_i_row_stride = _H_PACK  # 4 compact I-blocks per 512-tile
        nisa.dma_copy(
            dst=compact[nl.ds(quadrant_idx * QUADRANT_SIZE, PARTITIONS_PER_SLOT), :n_I512_local, :n_h128],
            src=scale_hbm.ap(
                # [0, 4]: broadcast one compact I-row across the quadrant's 4 partitions.
                # middle dim strides one 512-I-tile (== 4 compact I-rows) per local I-tile.
                pattern=[[0, PARTITIONS_PER_SLOT], [compact_i_row_stride * n_h128, n_I512_local], [1, n_h128]],
                offset=(i_tile_offset * _H_PACK + quadrant_idx) * n_h128,
                dtype=nl.uint8,
            ),
        )
    # ---- Stage 2: plain 128-contiguous broadcast compact H-blocks -> full H columns. ----
    src = compact.expand_dim(dim=3).broadcast(dim=3, size=_SCALE_BLOCK_128)
    dst = sbuf.reshape_dim(dim=2, shape=(n_h128, _SCALE_BLOCK_128))
    nisa.tensor_copy(dst=dst, src=src)
    staging_sbm.close_scope()


def _load_down_scale_tile_compact(scale_hbm, sbuf, staging_sbm, int_tile_global, h_start, h_size, name="down_scale"):
    """Stream ONE (int_tile, H-window) COMPACT down-scale slice and expand it in-kernel.

    Compact-scale analogue of ``_load_down_scale_tile``: reads compact block-128 down scales
    ``[I/128, H/128]`` for one 512-I-tile and an ``h_size``-wide (128-multiple) H window, expanding
    to the native block-32 ``[128, h_size]`` quadrant fold. ``h_start``/``h_size`` are multiples of
    ``_SCALE_BLOCK_128``.
    """
    QUADRANT_SIZE = 32
    PARTITIONS_PER_SLOT = 4
    n_h128 = h_size // _SCALE_BLOCK_128
    h128_start = h_start // _SCALE_BLOCK_128
    n_h128_full = scale_hbm.shape[-1]  # compact H-blocks in the full tensor

    staging_sbm.open_scope()
    compact = staging_sbm.alloc_stack((_PMAX, n_h128), dtype=nl.uint8, buffer=nl.sbuf, name=f"{name}_tile_compact")
    for quadrant_idx in range(div_ceil(_PMAX, QUADRANT_SIZE)):
        compact_i_row = int_tile_global * _H_PACK + quadrant_idx
        nisa.dma_copy(
            dst=compact[nl.ds(quadrant_idx * QUADRANT_SIZE, PARTITIONS_PER_SLOT), :n_h128],
            src=scale_hbm.ap(
                pattern=[[0, PARTITIONS_PER_SLOT], [1, n_h128]],
                offset=compact_i_row * n_h128_full + h128_start,
                dtype=nl.uint8,
            ),
        )
    src = compact.expand_dim(dim=2).broadcast(dim=2, size=_SCALE_BLOCK_128)
    dst = sbuf[:, :h_size].reshape_dim(dim=1, shape=(n_h128, _SCALE_BLOCK_128))
    nisa.tensor_copy(dst=dst, src=src)
    staging_sbm.close_scope()


def _load_down_scale_tile(scale_hbm, sbuf, int_tile_global, h_start, h_size):
    """Stream ONE (int_tile, hidden_tile) down-scale slice [16, h_size] into a ring buffer.

    Mirrors the 16->128 partition fold of ``_load_full_down_weight_scales`` (4 rows per
    32-partition quadrant) but for a single I-512 tile and a single H_TILE_SIZE-wide H window,
    so the full down scale never has to stay resident (large-I / SP case).
    Result: ``sbuf`` partitions [q*32 : q*32+4] hold the fold for each quadrant; columns are
    the ``h_size`` hidden elements of this tile.
    """
    QUADRANT_SIZE = 32
    PARTITIONS_PER_SLOT = 4
    for quadrant_idx in range(div_ceil(_PMAX, QUADRANT_SIZE)):
        nisa.dma_copy(
            dst=sbuf[nl.ds(quadrant_idx * QUADRANT_SIZE, PARTITIONS_PER_SLOT), :h_size],
            src=scale_hbm[
                nl.ds(quadrant_idx * PARTITIONS_PER_SLOT, PARTITIONS_PER_SLOT),
                int_tile_global,
                nl.ds(h_start, h_size),
            ],
        )


# ---------------------------------------------------------------------------
# Per-token-tile input loading.
# ---------------------------------------------------------------------------
def _load_and_transpose_hidden(hidden_hbm, hidden_sbuf_list, tok_off, tok_size, H, n_H512, hidden_size_hbm):
    """DMA-transpose the fp8 hidden tile into swizzle layout [128_H, H/512, 256_T, 4_H].

    Uses the fp32-reinterpret trick (4 fp8 bytes -> 1 fp32) so the transpose runs on the DMA
    engine.
    """
    _H512_FP32 = _H512 // _FP32_FP8_RATIO  # fp32 stride between adjacent H-512 tiles (== 128)
    bxs_x4_fp32 = _SRC_BXS_SUBTILE * _Q // _FP32_FP8_RATIO  # fp32 stride between H-512 slots in dst
    for src_sub in TiledRange(tok_size, _SRC_BXS_SUBTILE):  # 256 tokens each
        n_hidden_subtiles = _H512 // _Q  # 128 subtiles of 4 within one H-512 tile
        src_pattern = [
            [hidden_size_hbm // _FP32_FP8_RATIO, src_sub.size],
            [_H512_FP32, n_H512],
            [1, 1],
            [1, n_hidden_subtiles],
        ]
        src_offset = ((tok_off + src_sub.start_offset) * hidden_size_hbm) // _FP32_FP8_RATIO

        dst_pattern = [
            [n_H512 * bxs_x4_fp32, n_hidden_subtiles],
            [bxs_x4_fp32, n_H512],
            [1, 1],
            [1, src_sub.size],
        ]

        nisa.dma_transpose(
            src=hidden_hbm.ap(src_pattern, dtype=nl.float32, offset=src_offset),
            dst=hidden_sbuf_list[src_sub.index].ap(dst_pattern, dtype=nl.float32, offset=0),
        )


def _load_hidden_block_scales(
    hidden_hbm, scales_sbuf_list, staging_sbuf, tok_off, tok_size, H, n_packed, hidden_size_hbm
):
    """Load MX per-block hidden scales from the packed region into [128_H, n_packed, T].

    DMA copy the scale bytes (flat) into a staging buffer 128 tokens at a time,
    then PE-transpose each 128-wide pack through PSUM.
    """
    TILE_H = _PMAX
    FP8_TP_OUT_STEP = 2  # PE fp8 transpose output interleave factor
    UINT8_TP_VIEW = nl.float8_e5m2  # same byte width as uint8
    scale_region = n_packed * TILE_H

    scale_psum = nl.ndarray(
        (TILE_H, 1, _PMAX, FP8_TP_OUT_STEP),
        dtype=UINT8_TP_VIEW,
        buffer=nl.psum,
    )

    for src_sub in TiledRange(tok_size, _SRC_BXS_SUBTILE):  # 256 tokens
        for chunk in TiledRange(src_sub.size, _PMAX):  # 128 tokens at a time
            chunk_off = src_sub.start_offset + chunk.start_offset
            # Step 1: DMA scale bytes HBM -> staging SBUF (flat, no transpose)
            nisa.dma_copy(
                src=hidden_hbm.ap(
                    pattern=[[hidden_size_hbm, chunk.size], [1, scale_region]],
                    dtype=UINT8_TP_VIEW,
                    offset=(tok_off + chunk_off) * hidden_size_hbm + H,
                ),
                dst=staging_sbuf[: chunk.size, :scale_region],
            )
            # Step 2/3: PE-transpose each pack [chunk, 128] -> [128, chunk], evict to scale buffer
            for pack_idx in range(n_packed):
                nisa.nc_transpose(
                    data=staging_sbuf.ap(
                        pattern=[[scale_region, chunk.size], [1, TILE_H]],
                        offset=pack_idx * TILE_H,
                        dtype=UINT8_TP_VIEW,
                    ),
                    dst=scale_psum[:, 0, : chunk.size, 0],
                )
                nisa.tensor_copy(
                    src=scale_psum[:, 0, : chunk.size, 0],
                    dst=scales_sbuf_list[src_sub.index].ap(
                        pattern=[[n_packed * _SRC_BXS_SUBTILE, TILE_H], [1, chunk.size]],
                        offset=pack_idx * _SRC_BXS_SUBTILE + chunk.start_offset,
                        dtype=UINT8_TP_VIEW,
                    ),
                )


# ---------------------------------------------------------------------------
# Gate / up (source) projection with hoisted weights.
# ---------------------------------------------------------------------------
def _src_projection(
    hidden_sbuf_list,
    hidden_scales_sbuf_list,
    w_sbuf,
    w_scale_sbuf,
    psum_list,
    tok_size,
    H,
    I,
    n_H512,
    n_I512,
    w_hbm=None,
    w_ring_list=None,
    full_I=None,
    i_offset=0,
    int_tile_start=0,
    n_i512_group=None,
):
    """Compute one source projection (gate or up) into ``psum_list``.

    weight = stationary, hidden = moving; hidden block scales are the moving scale and the
    MX weight scales are the stationary scale. The weight access pattern is extended to
    index into the hoisted (full-H) weight buffer.

    ``I``/``n_I512`` are the LOCAL intermediate size this call produces (== full I unless
    I-sharded). ``full_I``/``i_offset`` describe the HBM slice: when I-sharded, each core streams
    only its ``I``-wide slice of the source weight (output rows ``[i_offset, i_offset+I)``).

    ``int_tile_start``/``n_i512_group`` select an I-GROUP: this call fills only PSUM banks
    ``[0, n_i512_group)`` for the ``n_i512_group`` I-512 tiles starting at ``int_tile_start``.
    Grouping lets large-I shapes (``n_I512`` > available PSUM banks) run: the caller loops the
    group, draining each group's banks before the next. Defaults (start=0, group=n_I512) reproduce
    the single-pass all-I-tiles-live behavior exactly. Only the PSUM bank mapping and (tiled) the
    streamed I-columns depend on the group; the SBUF drain buffers stay full-I sized.

    Two weight-placement modes:
      * Hoisted (``w_hbm is None``): ``w_sbuf`` already holds the full-H weight; the stationary
        AP indexes into it with the full per-partition stride and the GLOBAL int_tile index.
        (Never I-sharded.)
      * Tiled (``w_hbm``/``w_ring_list`` given): only the current group's I-columns of each H-tile
        are DMA'd from HBM into a small ring buffer just before consumption, so the full weight
        never has to fit in SBUF (shared-experts / SP case). Different groups read DISJOINT
        I-columns of the same H-tile, so every weight byte is still DMA'd exactly once across the
        whole group loop. The stationary AP then uses the group stride and a LOCAL int_tile index.
    """
    if full_I == None:
        full_I = I
    if n_i512_group == None:
        n_i512_group = n_I512
    SLOTS_PER_QUADRANT = 4
    PARTITIONS_PER_SLOT = 4
    per_h_block = n_I512 * _Q * _PMAX  # x4-element count per H-tile in the (local) weight buffer
    per_part_x4 = n_H512 * per_h_block  # full per-partition x4-element count (partition stride)
    group_per_h_block = n_i512_group * _Q * _PMAX  # x4-element count per H-tile for THIS group
    tiled_weights = w_hbm != None
    if tiled_weights:
        group_block_bytes = group_per_h_block * _Q  # fp8 bytes per H-tile per partition for this group
        full_block_bytes = full_I * _Q  # HBM per-H-tile stride spans the full (unsharded) I
        # Byte offset of this core's group within the H-block: I-shard base + group's I-column start.
        i_byte_offset = (i_offset + int_tile_start * _H512) * _Q
        w_hbm_view = w_hbm.reshape((_PMAX, n_H512, full_block_bytes))

    for h_tile in TiledRange(H, _H512):  # 512 in H
        hidden_subtiles = TiledRange(h_tile, _Q)  # 128 subtiles
        if tiled_weights:
            # Stream this H-tile's (I-sharded) group of I-columns into the next ring buffer.
            w_buf = w_ring_list[h_tile.index % len(w_ring_list)]
            nisa.dma_copy(
                dst=w_buf[:, :group_block_bytes],
                src=w_hbm_view[:, h_tile.index, nl.ds(i_byte_offset, group_block_bytes)],
            )
            w_partition_stride = group_per_h_block
            w_offset_base = 0
        else:
            w_buf = w_sbuf
            w_partition_stride = per_part_x4
            w_offset_base = h_tile.index * per_h_block
        for src_sub in TiledRange(tok_size, _SRC_BXS_SUBTILE):  # 256 in T
            packed_buf_idx = h_tile.index // SLOTS_PER_QUADRANT
            slot_part_off = (h_tile.index % SLOTS_PER_QUADRANT) * PARTITIONS_PER_SLOT
            moving_scale = hidden_scales_sbuf_list[src_sub.index][slot_part_off:, packed_buf_idx, : src_sub.size]

            for int_tile in TiledRange(I, _H512)[int_tile_start : int_tile_start + n_i512_group]:  # group's I-512 tiles
                # PSUM bank is LOCAL to the group; the weight offset uses the GLOBAL int_tile for
                # the resident full buffer (hoisted) or a LOCAL index into the group ring (tiled).
                psum_bank = src_sub.index * n_i512_group + (int_tile.index - int_tile_start)
                w_int_idx = (int_tile.index - int_tile_start) if tiled_weights else int_tile.index
                for int_row_tile in TiledRange(int_tile, _PMAX):  # 128 in 512
                    stationary_scale = w_scale_sbuf[
                        PARTITIONS_PER_SLOT * (h_tile.index % SLOTS_PER_QUADRANT) : len(hidden_subtiles),
                        h_tile.index // SLOTS_PER_QUADRANT,
                        nl.ds(int_row_tile.start_offset, int_row_tile.size),
                    ]
                    nisa.nc_matmul_mx(
                        dst=psum_list[psum_bank].ap(
                            pattern=[
                                [_SRC_BXS_SUBTILE * _Q, int_row_tile.size],
                                [1, src_sub.size],
                            ],
                            offset=int_row_tile.index * _SRC_BXS_SUBTILE,
                        ),
                        stationary=w_buf.ap(
                            pattern=[
                                [w_partition_stride, len(hidden_subtiles)],
                                [1, int_row_tile.size],
                            ],
                            offset=w_offset_base + w_int_idx * _Q * _PMAX + int_row_tile.index * _PMAX,
                            dtype=_fp8x4(),
                        ),
                        moving=hidden_sbuf_list[src_sub.index].ap(
                            pattern=[
                                [n_H512 * _SRC_BXS_SUBTILE, len(hidden_subtiles)],
                                [1, src_sub.size],
                            ],
                            offset=h_tile.index * _SRC_BXS_SUBTILE,
                            dtype=_fp8x4(),
                        ),
                        stationary_scale=stationary_scale,
                        moving_scale=moving_scale,
                    )


def _src_psum_read(psum_list, down_sub, int_tile, n_I512, int_tile_start=0, n_i512_group=None):
    """Stride-4 interleaved read of a source-projection PSUM bank.

    Reads with a stride of 4 so 4 adjacent I elements land contiguously, ready to be contracted
    together during the down projection. A 256-wide source subtile packs two 128-wide down
    subtiles into one PSUM bank; the second is at offset 128.

    ``int_tile_start``/``n_i512_group`` mirror ``_src_projection``: the bank is LOCAL to the
    current I-group, so subtract the group start and stride by the group width.
    """
    if n_i512_group == None:
        n_i512_group = n_I512
    int_subtiles = TiledRange(int_tile, _Q)
    psum_bank = (down_sub.index // 2) * n_i512_group + (int_tile.index - int_tile_start)
    return psum_list[psum_bank].ap(
        pattern=[
            [_SRC_BXS_SUBTILE * _Q, len(int_subtiles)],
            [1, down_sub.size],
            [_SRC_BXS_SUBTILE, _Q],
        ],
        offset=(down_sub.index % 2) * _DOWN_BXS_SUBTILE,
    )


def _apply_activation(
    gate_psum_list, src_proj_res_list, act_fn, bias_vector, tok_size, I, n_I512, int_tile_start=0, n_i512_group=None
):
    """SiLU(gate) from PSUM into the bf16 ``src_proj_res_list`` SBUF buffers.

    Drains only the current I-group's PSUM banks (``[int_tile_start, +n_i512_group)``) into the
    matching (global-I) slice of the full-I ``src_proj_res_list``.
    """
    if n_i512_group == None:
        n_i512_group = n_I512
    for down_sub in TiledRange(tok_size, _DOWN_BXS_SUBTILE):  # 128 in T
        for int_tile in TiledRange(I, _H512)[int_tile_start : int_tile_start + n_i512_group]:  # group's I-512 tiles
            int_subtiles = TiledRange(int_tile, _Q)
            dst_tile = src_proj_res_list[down_sub.index].ap(
                pattern=[
                    [n_I512 * _DOWN_BXS_SUBTILE * _Q, len(int_subtiles)],
                    [_Q, down_sub.size],
                    [1, _Q],
                ],
                offset=int_tile.index * _DOWN_BXS_SUBTILE * _Q,  # global-I offset into the full-I buffer
            )
            nisa.activation(
                dst=dst_tile,
                op=act_fn,
                data=_src_psum_read(gate_psum_list, down_sub, int_tile, n_I512, int_tile_start, n_i512_group),
                bias=bias_vector[: len(int_subtiles), 0:1],
            )


def _multiply_up(up_psum_list, src_proj_res_list, tok_size, I, n_I512, int_tile_start=0, n_i512_group=None):
    """Multiply the SiLU(gate) result (already in ``src_proj_res_list``) by up (from PSUM).

    Drains only the current I-group's ``up`` PSUM banks into the matching (global-I) slice.
    """
    if n_i512_group == None:
        n_i512_group = n_I512
    for down_sub in TiledRange(tok_size, _DOWN_BXS_SUBTILE):  # 128 in T
        for int_tile in TiledRange(I, _H512)[int_tile_start : int_tile_start + n_i512_group]:  # group's I-512 tiles
            int_subtiles = TiledRange(int_tile, _Q)
            dst_tile = src_proj_res_list[down_sub.index].ap(
                pattern=[
                    [n_I512 * _DOWN_BXS_SUBTILE * _Q, len(int_subtiles)],
                    [_Q, down_sub.size],
                    [1, _Q],
                ],
                offset=int_tile.index * _DOWN_BXS_SUBTILE * _Q,  # global-I offset into the full-I buffer
            )
            nisa.tensor_tensor(
                dst=dst_tile,
                data1=dst_tile,
                data2=_src_psum_read(up_psum_list, down_sub, int_tile, n_I512, int_tile_start, n_i512_group),
                op=nl.multiply,
            )


def _quantize_intermediate(src_proj_res_list, quantized_list, dequant_scales_list, tok_size, I, n_I512):
    """MX-quantize the intermediate activation (bf16 -> fp8x4 + uint8 block scales)."""
    for down_sub in TiledRange(tok_size, _DOWN_BXS_SUBTILE):  # 128 in T
        for int_tile in TiledRange(I, _H512):  # 512 in I
            int_subtiles = TiledRange(int_tile, _Q)
            bf16_pattern = [
                [n_I512 * _DOWN_BXS_SUBTILE * _Q, len(int_subtiles)],
                [_Q, down_sub.size],
                [1, _Q],
            ]
            fp8x4_pattern = [
                [n_I512 * _DOWN_BXS_SUBTILE, len(int_subtiles)],
                [1, down_sub.size],
            ]
            nisa.quantize_mx(
                dst=quantized_list[down_sub.index].ap(
                    fp8x4_pattern, dtype=_fp8x4(), offset=int_tile.index * _DOWN_BXS_SUBTILE
                ),
                src=src_proj_res_list[down_sub.index].ap(bf16_pattern, offset=int_tile.index * _DOWN_BXS_SUBTILE * _Q),
                dst_scale=dequant_scales_list[down_sub.index][: len(int_subtiles), int_tile.index, : down_sub.size],
            )


_DOWN_H_TILE_SIZE = 1024  # output columns produced per PSUM pass (gemm_moving_fmax)


def _down_projection(
    quantized_list,
    dequant_scales_list,
    down_w_sbuf,
    down_w_scale_sbuf,
    output_sbuf_list,
    tok_size,
    H,
    I,
    n_I512,
    down_w_hbm=None,
    down_w_ring_list=None,
    full_n_I512=None,
    i_tile_offset=0,
    down_w_scale_hbm=None,
    down_w_scale_ring_list=None,
    h_tiles_per_group=1,
    compact_scales=False,
    scale_staging_sbm=None,
):
    """Down projection, writing bf16 output [T, H] to ``output_sbuf_list``.

    intermediate = stationary, down weight = moving; intermediate dequant scales are the
    stationary scale and MX down weight scales are the moving scale.

    ``I``/``n_I512`` are the LOCAL intermediate size this core contracts over. When I-sharded,
    ``full_n_I512``/``i_tile_offset`` describe the down-weight HBM slice (I-tiles
    ``[i_tile_offset, i_tile_offset+n_I512)``); each core produces a full-H PARTIAL that the
    caller reduce-scatters across cores. Output columns are still the full H.

    H-tile grouping: each matmul still produces one ``H_TILE_SIZE``-wide PSUM output (the bank-width
    limit), but ``h_tiles_per_group`` of those H-tiles are processed together so ONE weight/scale
    DMA fetches the whole group's H window per int_tile (a contiguous HBM run) instead of one
    ``H_TILE_SIZE`` slice at a time. This decouples the DMA granularity from the PSUM tiling,
    turning many small (e.g. 4KB) transfers into few large ones -- the dominant win in the
    bandwidth-bound SP case. The group holds ``h_tiles_per_group * n_down_sub`` PSUM banks live
    across the int_tile (contraction) loop; the caller sizes the group so that stays <= banks.

    Two weight-placement modes:
      * Hoisted (``down_w_hbm is None``): ``down_w_sbuf`` holds the full down weight and
        ``down_w_scale_sbuf`` the full down scale; the moving AP / scale index them directly.
        (Never I-sharded.)
      * Tiled (``down_w_hbm``/``down_w_ring_list`` given): each (h_group, int_tile) down-weight slice
        ``[128, group_H, 4]`` is DMA'd into a ring buffer just before it is consumed, so the full
        down weight never has to fit in SBUF. When ``down_w_scale_hbm``/``down_w_scale_ring_list``
        are also given, the matching scale slice ``[16->128, group_H]`` is streamed the same way, so
        the (large) full down scale need not stay resident either (SP / large-I case).
    """
    if full_n_I512 == None:
        full_n_I512 = n_I512
    H_TILE_SIZE = _DOWN_H_TILE_SIZE
    group_h_max = h_tiles_per_group * H_TILE_SIZE  # ring free width (H elements) per int_tile
    n_int = I // _H512
    down_per_part_x4 = n_I512 * H  # full per-partition x4-element count (partition stride)
    tiled_weights = down_w_hbm != None
    tiled_scales = down_w_scale_hbm != None  # stream down scales too (SP / large-I)
    if tiled_weights:
        down_w_hbm_view = down_w_hbm.reshape((_PMAX, full_n_I512, H, _Q))

    all_htiles = TiledRange(H, H_TILE_SIZE)
    n_htiles = len(all_htiles)
    for h_group_start in range(0, n_htiles, h_tiles_per_group):
        group_htiles = all_htiles[h_group_start : h_group_start + h_tiles_per_group]
        group_h_off = group_htiles[0].start_offset
        group_h_size = group_htiles[-1].end_offset - group_h_off
        psum_list = []
        for bank in range(NUM_HW_PSUM_BANKS):
            psum_list.append(
                nl.ndarray(
                    (_PMAX, _MX_PSUM_FMAX),
                    dtype=nl.bfloat16,
                    buffer=nl.psum,
                    address=(0, bank * PSUM_BANK_SIZE),
                )
            )
        for int_tile in TiledRange(I, _H512):  # 512 in I (contraction, accumulated in the banks)
            int_subtiles = TiledRange(int_tile, _Q)  # 128 subtiles of 4
            ring_slot = ((h_group_start // h_tiles_per_group) * n_int + int_tile.index) % (
                len(down_w_ring_list) if tiled_weights else 1
            )
            if tiled_weights:
                # ONE DMA for the whole group's H window of this int_tile (contiguous in HBM).
                w_buf = down_w_ring_list[ring_slot]
                w_buf_view = w_buf.reshape((_PMAX, group_h_max, _Q))
                nisa.dma_copy(
                    dst=w_buf_view[:, :group_h_size, :],
                    src=down_w_hbm_view[:, i_tile_offset + int_tile.index, nl.ds(group_h_off, group_h_size), :],
                )
                w_partition_stride = group_h_max
            else:
                w_buf = down_w_sbuf
                w_partition_stride = down_per_part_x4
            if tiled_scales:
                # Stream the matching down-scale group window (folded 16->128 parts).
                scale_buf = down_w_scale_ring_list[ring_slot]
                if compact_scales:
                    _load_down_scale_tile_compact(
                        down_w_scale_hbm,
                        scale_buf,
                        scale_staging_sbm,
                        i_tile_offset + int_tile.index,
                        group_h_off,
                        group_h_size,
                    )
                else:
                    _load_down_scale_tile(
                        down_w_scale_hbm, scale_buf, i_tile_offset + int_tile.index, group_h_off, group_h_size
                    )
            for hidden_tile in group_htiles:
                h_local = hidden_tile.start_offset - group_h_off  # H offset within the group window
                h_local_idx = hidden_tile.index - h_group_start
                if tiled_weights:
                    w_offset_base = h_local  # ring is group-relative
                else:
                    w_offset_base = int_tile.index * H + hidden_tile.start_offset
                for down_sub in TiledRange(tok_size, _DOWN_BXS_SUBTILE):  # 128 in T
                    psum_bank = h_local_idx * div_ceil(tok_size, _DOWN_BXS_SUBTILE) + down_sub.index
                    stationary_scale = dequant_scales_list[down_sub.index][
                        : len(int_subtiles), int_tile.index, : down_sub.size
                    ]
                    if tiled_scales:
                        # Ring holds the group's H window; index this H-tile within it.
                        moving_scale = scale_buf[: len(int_subtiles), nl.ds(h_local, hidden_tile.size)]
                    else:
                        # Hoisted down scales cover all of H, so index at this hidden tile's offset.
                        moving_scale = down_w_scale_sbuf[
                            : len(int_subtiles), int_tile.index, nl.ds(hidden_tile.start_offset, hidden_tile.size)
                        ]
                    nisa.nc_matmul_mx(
                        dst=psum_list[psum_bank][: down_sub.size, : hidden_tile.size],
                        stationary=quantized_list[down_sub.index].ap(
                            pattern=[
                                [n_I512 * _DOWN_BXS_SUBTILE, len(int_subtiles)],
                                [1, down_sub.size],
                            ],
                            offset=int_tile.index * _DOWN_BXS_SUBTILE,
                            dtype=_fp8x4(),
                        ),
                        moving=w_buf.ap(
                            pattern=[
                                [w_partition_stride, len(int_subtiles)],
                                [1, hidden_tile.size],
                            ],
                            offset=w_offset_base,
                            dtype=_fp8x4(),
                        ),
                        stationary_scale=stationary_scale,
                        moving_scale=moving_scale,
                    )
        """
        Contraction complete for this group: drain each (hidden_tile, down_sub) bank to output.

        Alternate banks across the two engines so the eviction traffic is distributed across
        Vector or Scalar engine.
        """
        for hidden_tile in group_htiles:
            h_local_idx = hidden_tile.index - h_group_start
            for down_sub in TiledRange(tok_size, _DOWN_BXS_SUBTILE):
                psum_bank = h_local_idx * div_ceil(tok_size, _DOWN_BXS_SUBTILE) + down_sub.index
                drain_engine = nisa.vector_engine if psum_bank % 2 == 0 else nisa.scalar_engine
                nisa.tensor_copy(
                    dst=output_sbuf_list[down_sub.index][
                        : down_sub.size, nl.ds(hidden_tile.start_offset, hidden_tile.size)
                    ],
                    src=psum_list[psum_bank][: down_sub.size, : hidden_tile.size],
                    engine=drain_engine,
                )


def _store_output(output_sbuf_list, out_hbm, tok_off, tok_size, T, H, routed_hbm=None):
    """Store the bf16 output tile [T, H] to HBM [1, T, H].

    When ``routed_hbm`` is given, the store DMA also reads the matching routed-expert slice and
    writes ``our_result + routed`` (summed in fp32 inside the DMA engine) -- shared-experts mode.
    """
    out_view = out_hbm.reshape((1, T, H))
    routed_view = routed_hbm.reshape((1, T, H)) if routed_hbm != None else None
    for down_sub in TiledRange(tok_size, _DOWN_BXS_SUBTILE):
        sub_start = tok_off + down_sub.start_offset
        dst_ap = out_view.ap([[H, down_sub.size], [1, H]], offset=sub_start * H)
        src_sbuf = output_sbuf_list[down_sub.index][: down_sub.size, :H]
        if routed_view == None:
            nisa.dma_copy(dst=dst_ap, src=src_sbuf)
        else:
            nisa.dma_compute(
                dst=dst_ap,
                srcs=[src_sbuf, routed_view.ap([[H, down_sub.size], [1, H]], offset=sub_start * H)],
                reduce_op=nl.add,
            )


_PIPE_ID_DOWN_REDUCE = 1


def _reduce_scatter_down(output_sbuf_list, peer_sbuf_list, tok_size, H, n_prgs, prg_id):
    """Combine per-core full-H down-proj PARTIALS across the ``n_prgs`` cores (I-sharded path).

    Each core holds a full-H partial (it contracted only its I-shard). We only need the final,
    fully-reduced values for THIS core's H-half (the half it will store). So each core sends its
    partial for the OTHER core's H-half and receives the other core's partial for ITS half, then
    adds -- a reduce-scatter.
    """
    half_H = H // n_prgs
    my_h_off = prg_id * half_H
    other = 1 - prg_id
    other_h_off = other * half_H
    for down_sub in TiledRange(tok_size, _DOWN_BXS_SUBTILE):  # 128 in T
        # Exchange: send our partial for the other core's half; receive their partial for our half.
        nisa.sendrecv(
            send_to_rank=other,
            recv_from_rank=other,
            src=output_sbuf_list[down_sub.index][: down_sub.size, nl.ds(other_h_off, half_H)],
            dst=peer_sbuf_list[down_sub.index][: down_sub.size, :half_H],
            pipe_id=_PIPE_ID_DOWN_REDUCE,
        )
        # Add the received partial into our half -> final values for our half.
        nisa.tensor_tensor(
            dst=output_sbuf_list[down_sub.index][: down_sub.size, nl.ds(my_h_off, half_H)],
            data1=output_sbuf_list[down_sub.index][: down_sub.size, nl.ds(my_h_off, half_H)],
            data2=peer_sbuf_list[down_sub.index][: down_sub.size, :half_H],
            op=nl.add,
        )


def _store_output_half(output_sbuf_list, out_hbm, tok_off, tok_size, T, H, n_prgs, prg_id, routed_hbm=None):
    """Store only this core's H-half of the reduced output to HBM (I-sharded path).

    When ``routed_hbm`` is given, the store fuses ``our_half + routed_half`` (fp32 in the DMA
    engine). Each core adds only its own H-half slice of the routed output, so there is no
    double-counting across cores.
    """
    out_view = out_hbm.reshape((1, T, H))
    routed_view = routed_hbm.reshape((1, T, H)) if routed_hbm != None else None
    half_H = H // n_prgs
    my_h_off = prg_id * half_H
    for down_sub in TiledRange(tok_size, _DOWN_BXS_SUBTILE):
        sub_start = tok_off + down_sub.start_offset
        dst_ap = out_view.ap([[H, down_sub.size], [1, half_H]], offset=sub_start * H + my_h_off)
        src_sbuf = output_sbuf_list[down_sub.index][: down_sub.size, nl.ds(my_h_off, half_H)]
        if routed_view == None:
            nisa.dma_copy(dst=dst_ap, src=src_sbuf)
        else:
            nisa.dma_compute(
                dst=dst_ap,
                srcs=[
                    src_sbuf,
                    routed_view.ap([[H, down_sub.size], [1, half_H]], offset=sub_start * H + my_h_off),
                ],
                reduce_op=nl.add,
            )


# ---------------------------------------------------------------------------
# Kernel entry.
# ---------------------------------------------------------------------------
@nki.jit
def mlp_deepseek_mx(
    hidden_tensor: nl.NkiTensor,
    gate_proj_weights_tensor: nl.NkiTensor,
    up_proj_weights_tensor: nl.NkiTensor,
    down_proj_weights_tensor: nl.NkiTensor,
    gate_w_scale: nl.NkiTensor,
    up_w_scale: nl.NkiTensor,
    down_w_scale: nl.NkiTensor,
    activation_fn: ActFnType = ActFnType.SiLU,
    output_dtype=None,
    routed_expert_output: Optional[nl.NkiTensor] = None,
    sbm: Optional[SbufManager] = None,
    compact_scales: bool = False,
) -> list:
    """DeepSeek V3.2 MLP kernel (MX prequantized packed block-scale input).

    Specialized for DeepSeek V3.2 shared-experts and first dense MLP layers. Supports MX (MXFP8)
    quantization only, with MX-prequantized packed block-scale hidden input; no norm / bias /
    fused-add. Intended for prefill: large-T small-I shapes (e.g. T=8192, H=7168, I=512) hoist all
    weights and token-shard across cores, while small-T large-I shared-experts shapes (e.g. T<=512,
    H=7168, I=2048) stream weights tiled and I-shard across cores. See module docstring for details.

    Dimensions:
        B: Batch size
        S: Sequence length (T = B * S total tokens)
        H: Hidden dimension size (multiple of 512)
        I: Intermediate dimension size (multiple of 512)

    Args:
        hidden_tensor (nl.NkiTensor): [B, S, H + scale_region] fp8, MX-prequantized packed hidden
            states where ``scale_region = ceil(H/512/4)*128`` uint8 block-scale bytes follow the H
            fp8 bytes.
        gate_proj_weights_tensor (nl.NkiTensor): [128, H/512, I/512, 4, 128, 4] fp8 gate weight
            (MX ``H_X4_INNERMOST`` layout).
        up_proj_weights_tensor (nl.NkiTensor): [128, H/512, I/512, 4, 128, 4] fp8 up weight.
        down_proj_weights_tensor (nl.NkiTensor): [128, I/512, H, 4] fp8 down weight.
        gate_w_scale (nl.NkiTensor): MX gate weight scales. Native block-32 layout
            [16, H/512, I/512, 4, 128] when ``compact_scales`` is False; compact block-128 layout
            [H/128, I/128] when True.
        up_w_scale (nl.NkiTensor): MX up weight scales (same layouts as ``gate_w_scale``).
        down_w_scale (nl.NkiTensor): MX down weight scales. Native block-32 [16, I/512, H] when
            ``compact_scales`` is False; compact block-128 [I/128, H/128] when True.
        activation_fn (ActFnType): Gate activation (default SiLU).
        output_dtype: Output dtype (default bf16).
        routed_expert_output (Optional[nl.NkiTensor]): [B, S, H] routed-expert output to add to this
            kernel's result. When provided (shared-experts mode), the kernel stores
            ``mlp_result + routed_expert_output`` (summed in fp32 inside the store DMA). When None
            (plain MLP mode), the kernel's own result is stored unchanged.
        sbm (Optional[SbufManager]): Optional SbufManager for the shared HBM output allocation.
        compact_scales (bool): When True, gate/up/down weight scales are DeepSeek compact block-128
            (one uint8 per 128x128 weight block) and are expanded + swizzled to the native block-32
            layout in-kernel. When False (default), scales are already the native block-32 layout.

    Returns:
        output_tensor_hbm (nl.NkiTensor): [B, S, H] bf16, returned as a single-element list. When
            ``routed_expert_output`` is given, this is the elementwise sum of the two.

    Pseudocode:
        output = zeros([B, S, H])
        for token_tile in tokens:
            # DMA-transpose fp8 hidden into swizzle layout + load MX block scales
            hidden_sw, hidden_scale = load_and_transpose(hidden_tensor[token_tile])
            # gate/up: hidden (moving) @ weight (stationary), MX block scales
            gate = nc_matmul_mx(hidden_sw, gate_w, hidden_scale, gate_w_scale)
            up = nc_matmul_mx(hidden_sw, up_w, hidden_scale, up_w_scale)
            intermediate = activation_fn(gate) * up
            # MX-quantize intermediate, then down projection
            inter_q, inter_scale = quantize_mx(intermediate)
            output[token_tile] = nc_matmul_mx(inter_q, down_w, inter_scale, down_w_scale)
    """
    if output_dtype == None:
        output_dtype = nl.bfloat16

    """
    Derive shapes directly from the weight tensors (no MLPParameters machinery).

    down weight: [128_I, I/512, H, 4]; gate/up weight: [128_H, H/512, I/512, 4, 128_I, 4_H].
    """
    H = down_proj_weights_tensor.shape[2]
    n_I512 = down_proj_weights_tensor.shape[1]
    I = gate_proj_weights_tensor.shape[2] * gate_proj_weights_tensor.shape[3] * gate_proj_weights_tensor.shape[4]
    n_H512 = H // _H512
    n_packed = div_ceil(n_H512, _Q)
    n_scale_packed = div_ceil(n_H512, _Q)  # gate/up scale column-packing factor

    batch_size = hidden_tensor.shape[0]
    sequence_len = hidden_tensor.shape[1]
    hidden_size_hbm = hidden_tensor.shape[-1]  # H + n_packed*128
    T = batch_size * sequence_len

    kernel_assert(H % _H512 == 0, "H must be a multiple of 512")
    kernel_assert(I % _H512 == 0, "I must be a multiple of 512")

    act_fn = get_nl_act_fn_from_type(activation_fn)

    # Allocate the shared HBM output tensor.
    out_sbm = sbm
    if out_sbm == None:
        out_sbm = SbufManager(0, 200 * 1024, get_logger("mlp_deepseek_mx"))
        out_sbm.set_name_prefix("mlp_ds_")
    out_hbm = out_sbm.alloc(
        (batch_size, sequence_len, H),
        dtype=output_dtype,
        buffer=nl.shared_hbm,
        name="output_tensor_hbm",
    )

    _, n_prgs, prg_id = get_program_sharding_info()

    local_sbm = SbufManager(
        sb_lower_bound=0,
        sb_upper_bound=MAX_AVAILABLE_SBUF_SIZE,
        logger=get_logger("mlp_deepseek_mx"),
    )
    local_sbm.open_scope(interleave_degree=1, name="mlp_ds")
    heap_alloc = local_sbm.alloc_heap

    # --- Constants ---
    bias_vector = heap_alloc((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"bias_vector__prog{prg_id}")
    nisa.memset(bias_vector, value=0.0)

    # --- Decide weight placement: hoist full weights, or stream them tiled through ring buffers ---
    """
    Estimate the per-partition working set of one double-buffered (interleave=2) token tile so the
    decision leaves room for it.

    Hoisting is the fast path (weights loaded once); tiling is the fallback for the shared-experts
    shape (H=7168, I=2048) where full weights do not fit.
    """
    per_h_block_bytes = n_I512 * _Q * _PMAX * _Q  # fp8 bytes per H-tile per partition (== I*4)
    full_src_w_bytes = n_H512 * per_h_block_bytes  # per partition, per source weight (gate or up)
    full_down_w_bytes = n_I512 * H * _Q  # per partition
    full_weight_bytes = 2 * full_src_w_bytes + full_down_w_bytes
    est_tok = min(_TOKEN_TILE, T)
    est_n_src = div_ceil(est_tok, _SRC_BXS_SUBTILE)
    est_n_down = div_ceil(est_tok, _DOWN_BXS_SUBTILE)
    out_bytes = sizeinbytes(output_dtype)
    working_set_bytes = (
        est_n_src * (n_H512 * _SRC_BXS_SUBTILE * _Q + n_packed * _SRC_BXS_SUBTILE)
        + n_packed * _PMAX  # scale staging
        + est_n_down
        * (
            n_I512 * _DOWN_BXS_SUBTILE * _Q * sizeinbytes(nl.bfloat16)  # src_proj_res (bf16)
            + n_I512 * _DOWN_BXS_SUBTILE * _Q  # quantized intermediate (fp8)
            + n_I512 * _DOWN_BXS_SUBTILE  # dequant scales (uint8)
            + H * out_bytes  # output tile
        )
    )
    _TOKEN_TILE_INTERLEAVE = 2
    hoist_weights = full_weight_bytes + _TOKEN_TILE_INTERLEAVE * working_set_bytes <= local_sbm.get_free_space()

    # --- Choose the LNC sharding axis ---
    """
    I-sharding is enabled ONLY in the shared-experts scenario: i.e. when weights do NOT fit and
    must be streamed (not hoisted), AND we have >1 core. There, splitting I halves each core's
    weight-DMA volume (the actual bottleneck) instead of halving already-tiny compute, and the two
    cores' full-H down-proj partials are combined with a reduce-scatter. When weights DO fit (the
    large-T first-dense case), they are hoisted once so a collective would only add overhead --
    we keep token-sharding there.
    """
    shard_on_i = (not hoist_weights) and (n_prgs > 1)
    if shard_on_i:
        kernel_assert(I % (n_prgs * _H512) == 0, "I-sharded MLP requires I divisible by n_prgs*512")
        kernel_assert(H % n_prgs == 0, "I-sharded MLP requires H divisible by n_prgs")
        I_local = I // n_prgs
        n_I512_local = n_I512 // n_prgs
        i_offset = prg_id * I_local  # source-weight output-row offset for this core
        i_tile_offset = prg_id * n_I512_local  # down-weight I-tile offset for this core
        loop_tok_start, loop_tok_end = 0, T  # each core processes ALL tokens
    else:
        # Token-sharding at TOKEN granularity (not tile granularity): split T tokens evenly across
        # programs so both cores stay busy even when T fits in a single _TOKEN_TILE.
        I_local = I
        n_I512_local = n_I512
        i_offset = 0
        i_tile_offset = 0
        tokens_per_prog = div_ceil(T, n_prgs)
        loop_tok_start = prg_id * tokens_per_prog
        loop_tok_end = min(loop_tok_start + tokens_per_prog, T)

    # --- Hoist weight scales (always small enough to keep resident in both modes) ---
    """
    Sized to this core's I-shard (== full I unless I-sharded).

    Hoisted path: gate/up scales are loaded once and stay resident on the heap for the whole token
    loop. Tiled path: they are instead (re)loaded per token tile into a per-tile "srcphase" stack
    scope, so their SBUF (~72KB) is freed before the down phase -- that reclaimed space lets the
    down-weight ring hold a larger H window per DMA (bigger, more efficient transfers). The reload
    is 0 extra DMAs for single-token-tile shapes (SP / shared-experts) and one small DMA otherwise.
    """
    n_scale_packed_local = n_scale_packed
    gate_scale_sbuf = up_scale_sbuf = down_scale_sbuf = None
    if hoist_weights:
        gate_scale_sbuf = heap_alloc(
            (_PMAX, n_scale_packed_local, I_local), dtype=nl.uint8, name=f"gate_scale__prog{prg_id}"
        )
        up_scale_sbuf = heap_alloc(
            (_PMAX, n_scale_packed_local, I_local), dtype=nl.uint8, name=f"up_scale__prog{prg_id}"
        )
        if compact_scales:
            _load_full_src_weight_scales_compact(
                gate_w_scale,
                gate_scale_sbuf,
                local_sbm,
                H,
                I,
                n_H512,
                n_I512,
                n_scale_packed,
                i_offset=i_offset,
                i_local=I_local,
                name=f"gate_scale__prog{prg_id}",
            )
            _load_full_src_weight_scales_compact(
                up_w_scale,
                up_scale_sbuf,
                local_sbm,
                H,
                I,
                n_H512,
                n_I512,
                n_scale_packed,
                i_offset=i_offset,
                i_local=I_local,
                name=f"up_scale__prog{prg_id}",
            )
        else:
            _load_full_src_weight_scales(
                gate_w_scale, gate_scale_sbuf, H, I, n_H512, n_I512, n_scale_packed, i_offset=i_offset, i_local=I_local
            )
            _load_full_src_weight_scales(
                up_w_scale, up_scale_sbuf, H, I, n_H512, n_I512, n_scale_packed, i_offset=i_offset, i_local=I_local
            )
        # down scale resident only in the hoisted path; tiled streams it alongside the down ring.
        down_scale_sbuf = heap_alloc((_PMAX, n_I512_local, H), dtype=nl.uint8, name=f"down_scale__prog{prg_id}")
        if compact_scales:
            _load_full_down_weight_scales_compact(
                down_w_scale,
                down_scale_sbuf,
                local_sbm,
                H,
                n_I512,
                i_tile_offset=i_tile_offset,
                n_I512_local=n_I512_local,
                name=f"down_scale__prog{prg_id}",
            )
        else:
            _load_full_down_weight_scales(
                down_w_scale, down_scale_sbuf, H, n_I512, i_tile_offset=i_tile_offset, n_I512_local=n_I512_local
            )

    # --- I-group sizing for the gate/up (source) projection ---
    """
    gate/up hold one PSUM bank per live I-512 tile; with only NUM_HW_PSUM_BANKS banks, large I
    (SP case, e.g. I_local=9216 -> n_I512_local=18) cannot keep all I live at once. Split the
    local I into groups of at most NUM_HW_PSUM_BANKS tiles and drain each group before the next.
    When n_I512_local already fits, this is a single group == the original single-pass behavior.

    Groups must be UNIFORM (an even divisor of n_I512_local): the (tiled) src weight ring is a
    fixed-size buffer, and nc_matmul_mx requires the stationary partition stride to equal the
    buffer's full free dim -- a smaller partial tail group would violate that. So pick the largest
    divisor of n_I512_local not exceeding the bank cap.
    """
    _group_cap = min(n_I512_local, NUM_HW_PSUM_BANKS)
    n_src_i_group = _largest_divisor_at_most(n_I512_local, _group_cap)
    n_src_i_groups = n_I512_local // n_src_i_group

    # The (tiled) src weight ring only needs to hold ONE group's I-columns for one H-tile.
    local_group_h_block_bytes = n_src_i_group * _Q * _PMAX * _Q  # fp8 bytes per H-tile per partition, one group

    # --- Down-projection H-tile grouping (tiled path only) ---
    """
    Each down matmul still emits one 1024-wide PSUM output, but we process several H-tiles per
    group so ONE weight/scale DMA covers the group's whole (contiguous) H window per int_tile,
    replacing many tiny transfers (e.g. 4KB) with a few large ones. The group is capped by BOTH:
      * PSUM banks: it holds h_tiles_per_group * n_down_sub banks live -> <= NUM_HW_PSUM_BANKS.
      * SBUF: the depth-D down weight+scale rings grow with the group, so they must fit in the
        free space that remains after reserving the interleaved token-tile working set.
    The hoisted path keeps grouping at 1 (no DMA to amortize) so it stays byte-identical.
    """
    n_down_htiles = div_ceil(H, _DOWN_H_TILE_SIZE)
    est_tok_span = max(loop_tok_end - loop_tok_start, 1)
    est_n_down_sub = div_ceil(min(_TOKEN_TILE, est_tok_span), _DOWN_BXS_SUBTILE)
    if hoist_weights:
        down_h_tiles_per_group = 1
    else:
        group_by_banks = min(n_down_htiles, max(1, NUM_HW_PSUM_BANKS // est_n_down_sub))
        # Bytes of down ring (weight fp8 + scale uint8, depth D) per H-tile of group width.
        ring_bytes_per_htile = _WEIGHT_RING_DEPTH * (_DOWN_H_TILE_SIZE * _Q + _DOWN_H_TILE_SIZE)
        """
        The token-tile body uses two SEQUENTIAL sibling stack scopes that share memory:
          srcphase: hidden + hidden_scales + staging + src_proj_res + gate/up scales + src_w_ring
          downphase: down rings
        plus a set of buffers that PERSIST across both (quantized, dequant scales, output, peer).
        The stack peak is  persistent + max(srcphase, downphase)  -- the two phases NEVER coexist,
        so the down rings REUSE the srcphase space. Thus the down ring is bounded by
        (free - persistent - margin), NOT by what's left after also reserving srcphase (that would
        double-count and needlessly shrink the ring). We additionally require the downphase working
        set not to exceed srcphase by so much that the peak grows past budget -- but srcphase (incl.
        the ~72KB gate/up scales) already dominates a full-H down ring, so this is not binding here.
        """
        persistent_ws = est_n_down * (
            n_I512_local * _DOWN_BXS_SUBTILE * _Q  # quantized intermediate (fp8)
            + n_I512_local * _DOWN_BXS_SUBTILE  # dequant scales (uint8)
            + H * out_bytes  # output tile
            + (H // n_prgs) * out_bytes  # peer scratch (I-sharded reduce-scatter)
        )
        srcphase_ws = (
            est_n_src * (n_H512 * _SRC_BXS_SUBTILE * _Q + n_packed * _SRC_BXS_SUBTILE)
            + n_packed * _PMAX  # scale staging
            + est_n_down * (n_I512_local * _DOWN_BXS_SUBTILE * _Q * sizeinbytes(nl.bfloat16))  # src_proj_res (bf16)
            + 2 * n_scale_packed * I_local  # gate + up scales (now stack-resident in srcphase)
            + _WEIGHT_RING_DEPTH * local_group_h_block_bytes  # src_w_ring
        )
        """
        Tiled path forces interleave=1 (no cross-tile double-buffering) so these scopes are not
        doubled. The down rings reuse srcphase's freed space, so reserve only persistent + margin;
        then clamp so the downphase peak (persistent + down rings) cannot exceed the srcphase peak
        (persistent + srcphase) -- i.e. the down rings may grow up to srcphase_ws for free.
        """
        margin = 4 * 1024
        avail_for_down_ring = local_sbm.get_free_space() - persistent_ws - margin
        group_by_mem = max(1, avail_for_down_ring // ring_bytes_per_htile)
        down_h_tiles_per_group = max(1, min(group_by_banks, group_by_mem))
    down_ring_h_elems = down_h_tiles_per_group * _DOWN_H_TILE_SIZE  # H elements per ring slot

    gate_w_sbuf = up_w_sbuf = down_w_sbuf = None
    if hoist_weights:
        # Fully-hoisted path: load full gate/up/down weights once, before the token loop.
        gate_w_sbuf = heap_alloc(
            (_PMAX, n_H512, per_h_block_bytes), dtype=nl.float8_e4m3fn, name=f"gate_w__prog{prg_id}"
        )
        up_w_sbuf = heap_alloc((_PMAX, n_H512, per_h_block_bytes), dtype=nl.float8_e4m3fn, name=f"up_w__prog{prg_id}")
        down_w_sbuf = heap_alloc((_PMAX, n_I512, H, _Q), dtype=nl.float8_e4m3fn, name=f"down_w__prog{prg_id}")
        _load_full_src_weight(gate_proj_weights_tensor, gate_w_sbuf, H, I, n_H512, n_I512)
        _load_full_src_weight(up_proj_weights_tensor, up_w_sbuf, H, I, n_H512, n_I512)
        _load_full_down_weight(down_proj_weights_tensor, down_w_sbuf, H, n_I512)
    """
    Tiled path: gate/up scales and all ring buffers are NOT hoisted here. They are allocated per
    token tile inside two sequential sibling stack scopes ("srcphase" then "downphase") so the
    srcphase space (hidden + gate/up scales + src ring) is freed and REUSED by the larger
    downphase down rings. This is what lets the down-weight DMA cover a big H window per transfer.
    """

    # --- Token-tile loop ---
    """
    Hoisted path keeps cross-tile double-buffering (interleave=2). Tiled path uses interleave=1:
    its per-tile phase scopes need the full free space (no doubling) to grow the down ring, and
    cross-tile prefetch is a no-op for the single-token-tile shapes this path targets.
    """
    scale_region = n_packed * _PMAX
    token_interleave = _TOKEN_TILE_INTERLEAVE if hoist_weights else 1
    local_sbm.open_scope(interleave_degree=token_interleave, name="token_tiles")
    stack_alloc = local_sbm.alloc_stack

    n_prog_tiles = div_ceil(max(loop_tok_end - loop_tok_start, 0), _TOKEN_TILE)
    for tile_idx in range(n_prog_tiles):
        tok_off = loop_tok_start + tile_idx * _TOKEN_TILE  # absolute token offset into [T]
        tok_size = min(_TOKEN_TILE, loop_tok_end - tok_off)
        n_src_subtiles = div_ceil(tok_size, _SRC_BXS_SUBTILE)
        n_down_subtiles = div_ceil(tok_size, _DOWN_BXS_SUBTILE)

        """
        Persistent buffers -- live across BOTH the src (gate/up) and down phases:
        quantized intermediate + dequant scales (produced by src, consumed by down),
        output tile, and (I-sharded) peer scratch for the reduce-scatter.
        """
        quantized_list = []
        dequant_scales_list = []
        output_sbuf_list = []
        peer_sbuf_list = []
        for subtile_idx in range(n_down_subtiles):
            quantized_list.append(
                stack_alloc(
                    (_PMAX, n_I512_local, _DOWN_BXS_SUBTILE, _Q),
                    dtype=nl.float8_e4m3fn,
                    name=f"int_q__prog{prg_id}__tile{tile_idx}__s{subtile_idx}",
                )
            )
            dequant_scales_list.append(
                stack_alloc(
                    (_PMAX, n_I512_local, _DOWN_BXS_SUBTILE),
                    dtype=nl.uint8,
                    name=f"int_scale__prog{prg_id}__tile{tile_idx}__s{subtile_idx}",
                )
            )
            output_sbuf_list.append(
                stack_alloc(
                    (_PMAX, H), dtype=output_dtype, name=f"output__prog{prg_id}__tile{tile_idx}__s{subtile_idx}"
                )
            )
        if shard_on_i:
            for subtile_idx in range(n_down_subtiles):
                peer_sbuf_list.append(
                    stack_alloc(
                        (_PMAX, H // n_prgs),
                        dtype=output_dtype,
                        name=f"peer_out__prog{prg_id}__tile{tile_idx}__s{subtile_idx}",
                    )
                )

        # --- Source (gate/up) phase ---
        """
        In the tiled path this is a nested scope: hidden + gate/up scales + src ring live here and
        are freed at close, so the down phase's rings reuse their space. In the hoisted path the
        scales/weights are already resident on the heap; we open the scope regardless (harmless --
        it just brackets the per-tile hidden/intermediate stack allocations).
        """
        local_sbm.open_scope(interleave_degree=1, name="srcphase")
        hidden_sbuf_list = []
        hidden_scales_sbuf_list = []
        for subtile_idx in range(n_src_subtiles):
            hidden_sbuf_list.append(
                stack_alloc(
                    (_PMAX, n_H512, _SRC_BXS_SUBTILE, _Q),
                    dtype=nl.float8_e4m3fn,
                    align=32,  # xbar transpose requires 32B alignment
                    name=f"hidden__prog{prg_id}__tile{tile_idx}__s{subtile_idx}",
                )
            )
            hidden_scales_sbuf_list.append(
                stack_alloc(
                    (_PMAX, n_packed, _SRC_BXS_SUBTILE),
                    dtype=nl.uint8,
                    align=32,
                    name=f"hidden_scale__prog{prg_id}__tile{tile_idx}__s{subtile_idx}",
                )
            )
        staging_sbuf = stack_alloc(
            (_PMAX, scale_region),
            dtype=nl.float8_e5m2,
            name=f"scale_staging__prog{prg_id}__tile{tile_idx}",
        )
        # src_proj_res (bf16) is the activation staging; only needed within the src phase.
        src_proj_res_list = []
        for subtile_idx in range(n_down_subtiles):
            src_proj_res_list.append(
                stack_alloc(
                    (_PMAX, n_I512_local, _DOWN_BXS_SUBTILE, _Q),
                    dtype=nl.bfloat16,
                    name=f"src_res__prog{prg_id}__tile{tile_idx}__s{subtile_idx}",
                )
            )
        # Tiled path: (re)load gate/up scales + allocate the src weight ring inside this scope.
        if hoist_weights:
            tile_gate_scale = gate_scale_sbuf
            tile_up_scale = up_scale_sbuf
            tile_src_w_ring = None
        else:
            tile_gate_scale = stack_alloc(
                (_PMAX, n_scale_packed_local, I_local), dtype=nl.uint8, name=f"gate_scale__prog{prg_id}__t{tile_idx}"
            )
            tile_up_scale = stack_alloc(
                (_PMAX, n_scale_packed_local, I_local), dtype=nl.uint8, name=f"up_scale__prog{prg_id}__t{tile_idx}"
            )
            if compact_scales:
                _load_full_src_weight_scales_compact(
                    gate_w_scale,
                    tile_gate_scale,
                    local_sbm,
                    H,
                    I,
                    n_H512,
                    n_I512,
                    n_scale_packed,
                    i_offset=i_offset,
                    i_local=I_local,
                    name=f"gate_scale__prog{prg_id}__t{tile_idx}",
                )
                _load_full_src_weight_scales_compact(
                    up_w_scale,
                    tile_up_scale,
                    local_sbm,
                    H,
                    I,
                    n_H512,
                    n_I512,
                    n_scale_packed,
                    i_offset=i_offset,
                    i_local=I_local,
                    name=f"up_scale__prog{prg_id}__t{tile_idx}",
                )
            else:
                _load_full_src_weight_scales(
                    gate_w_scale,
                    tile_gate_scale,
                    H,
                    I,
                    n_H512,
                    n_I512,
                    n_scale_packed,
                    i_offset=i_offset,
                    i_local=I_local,
                )
                _load_full_src_weight_scales(
                    up_w_scale, tile_up_scale, H, I, n_H512, n_I512, n_scale_packed, i_offset=i_offset, i_local=I_local
                )
            tile_src_w_ring = []
            for ring_idx in range(_WEIGHT_RING_DEPTH):
                tile_src_w_ring.append(
                    stack_alloc(
                        (_PMAX, local_group_h_block_bytes),
                        dtype=nl.float8_e4m3fn,
                        name=f"src_w_ring__prog{prg_id}__t{tile_idx}__r{ring_idx}",
                    )
                )

        _load_and_transpose_hidden(hidden_tensor, hidden_sbuf_list, tok_off, tok_size, H, n_H512, hidden_size_hbm)
        _load_hidden_block_scales(
            hidden_tensor, hidden_scales_sbuf_list, staging_sbuf, tok_off, tok_size, H, n_packed, hidden_size_hbm
        )

        """
        Gate/up (source) projection, tiled over I-GROUPS so at most NUM_HW_PSUM_BANKS I-512 tiles
        are live in PSUM at once. For each group: gate -> PSUM -> SiLU into (global-I) SBUF slice;
        then up reuses the same banks -> multiply into the same SBUF slice. Groups drain into
        disjoint I-columns of the full-I ``src_proj_res_list``. PSUM banks are re-declared per
        group so each group's h_tile accumulation is a distinct accumulation group (the scheduler
        rejects reusing one logical PSUM tensor across groups with drains interleaved).
        """
        for i_group in range(n_src_i_groups):
            int_tile_start = i_group * n_src_i_group
            n_i512_group = min(n_src_i_group, n_I512_local - int_tile_start)
            gate_psum_list = []
            up_psum_list = []
            for psum_bank_idx in range(NUM_HW_PSUM_BANKS):
                gate_psum_list.append(
                    nl.ndarray(
                        (_PMAX, _MX_PSUM_FMAX),
                        dtype=nl.bfloat16,
                        buffer=nl.psum,
                        address=(0, psum_bank_idx * PSUM_BANK_SIZE),
                    )
                )
                # up reuses the same physical banks after gate has been drained.
                up_psum_list.append(
                    nl.ndarray(
                        (_PMAX, _MX_PSUM_FMAX),
                        dtype=nl.bfloat16,
                        buffer=nl.psum,
                        address=(0, psum_bank_idx * PSUM_BANK_SIZE),
                    )
                )
            _src_projection(
                hidden_sbuf_list,
                hidden_scales_sbuf_list,
                gate_w_sbuf,
                tile_gate_scale,
                gate_psum_list,
                tok_size,
                H,
                I_local,
                n_H512,
                n_I512_local,
                w_hbm=None if hoist_weights else gate_proj_weights_tensor,
                w_ring_list=tile_src_w_ring,
                full_I=I,
                i_offset=i_offset,
                int_tile_start=int_tile_start,
                n_i512_group=n_i512_group,
            )
            _apply_activation(
                gate_psum_list,
                src_proj_res_list,
                act_fn,
                bias_vector,
                tok_size,
                I_local,
                n_I512_local,
                int_tile_start=int_tile_start,
                n_i512_group=n_i512_group,
            )
            _src_projection(
                hidden_sbuf_list,
                hidden_scales_sbuf_list,
                up_w_sbuf,
                tile_up_scale,
                up_psum_list,
                tok_size,
                H,
                I_local,
                n_H512,
                n_I512_local,
                w_hbm=None if hoist_weights else up_proj_weights_tensor,
                w_ring_list=tile_src_w_ring,
                full_I=I,
                i_offset=i_offset,
                int_tile_start=int_tile_start,
                n_i512_group=n_i512_group,
            )
            _multiply_up(
                up_psum_list,
                src_proj_res_list,
                tok_size,
                I_local,
                n_I512_local,
                int_tile_start=int_tile_start,
                n_i512_group=n_i512_group,
            )

        _quantize_intermediate(src_proj_res_list, quantized_list, dequant_scales_list, tok_size, I_local, n_I512_local)
        local_sbm.close_scope()  # srcphase -- frees hidden + gate/up scales + src ring

        # --- Down phase ---
        """
        The down rings are allocated here, in a sibling scope that reuses the just-freed srcphase
        space, so they can hold a large H window per DMA (see down_ring_h_elems sizing above).
        """
        local_sbm.open_scope(interleave_degree=1, name="downphase")
        if hoist_weights:
            tile_down_w_ring = None
            tile_down_scale_ring = None
        else:
            tile_down_w_ring = []
            for ring_idx in range(_WEIGHT_RING_DEPTH):
                tile_down_w_ring.append(
                    stack_alloc(
                        (_PMAX, down_ring_h_elems * _Q),
                        dtype=nl.float8_e4m3fn,
                        name=f"down_w_ring__prog{prg_id}__t{tile_idx}__r{ring_idx}",
                    )
                )
            tile_down_scale_ring = []
            for ring_idx in range(_WEIGHT_RING_DEPTH):
                tile_down_scale_ring.append(
                    stack_alloc(
                        (_PMAX, down_ring_h_elems),
                        dtype=nl.uint8,
                        name=f"down_scale_ring__prog{prg_id}__t{tile_idx}__r{ring_idx}",
                    )
                )

        _down_projection(
            quantized_list,
            dequant_scales_list,
            down_w_sbuf,
            down_scale_sbuf,
            output_sbuf_list,
            tok_size,
            H,
            I_local,
            n_I512_local,
            down_w_hbm=None if hoist_weights else down_proj_weights_tensor,
            down_w_ring_list=tile_down_w_ring,
            full_n_I512=n_I512,
            i_tile_offset=i_tile_offset,
            down_w_scale_hbm=None if hoist_weights else down_w_scale,
            down_w_scale_ring_list=tile_down_scale_ring,
            h_tiles_per_group=down_h_tiles_per_group,
            compact_scales=compact_scales,
            scale_staging_sbm=local_sbm,
        )
        local_sbm.close_scope()  # downphase -- frees down rings

        if shard_on_i:
            # Each core holds a full-H partial (contracted only its I-shard). Reduce-scatter across
            # cores so each core owns the final values for its H-half, then store just that half.
            _reduce_scatter_down(output_sbuf_list, peer_sbuf_list, tok_size, H, n_prgs, prg_id)
            _store_output_half(
                output_sbuf_list, out_hbm, tok_off, tok_size, T, H, n_prgs, prg_id, routed_hbm=routed_expert_output
            )
        else:
            _store_output(output_sbuf_list, out_hbm, tok_off, tok_size, T, H, routed_hbm=routed_expert_output)

        local_sbm.increment_section()

    local_sbm.close_scope()  # token_tiles

    # Pop heap allocations (LIFO). Hoisted path holds weights + scales on the heap; tiled path keeps
    # only the bias here (its scales/rings were per-tile stack allocations, already freed).
    if hoist_weights:
        local_sbm.pop_heap()  # down_w
        local_sbm.pop_heap()  # up_w
        local_sbm.pop_heap()  # gate_w
        local_sbm.pop_heap()  # down_scale
        local_sbm.pop_heap()  # up_scale
        local_sbm.pop_heap()  # gate_scale
    local_sbm.pop_heap()  # bias_vector
    local_sbm.close_scope()  # mlp_ds

    return [out_hbm]
