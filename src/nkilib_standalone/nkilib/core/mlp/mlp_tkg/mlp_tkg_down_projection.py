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

"""
kernels - high performance MLP kernels

"""

import math
from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager

# common utils
from ...utils.kernel_assert import kernel_assert
from ...utils.logging import Logger
from ...utils.tiled_range import TiledRange
from ...utils.interleave_copy import interleave_copy
from ...utils.common_types import QuantizationType
from ...utils.tensor_view import TensorView
from ...utils.kernel_helpers import get_max_positive_value_for_dtype

# MLP utils
from ..mlp_parameters import MLPParameters, mlpp_has_down_projection_bias
from .mlp_tkg_constants import (
    MLPTKGConstants,
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsDownTileCounts,
    MLPTKGConstantsGateUpTileCounts,
)


def down_projection(
    hidden: nl.ndarray,
    weight: nl.ndarray,
    bias: nl.ndarray,
    output_tile: nl.ndarray,
    weight_tiles: list[nl.ndarray],
    bias_tile: nl.ndarray,
    dequant_tile: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsDownTileCounts,
    params: MLPParameters,
):
    """
    Performs a single Down projection shard on the H.

    Computes: Hidden[I, T] @ Weight[I, H] + Optional(Bias[1, H]) → [T, H]
    - Hidden is the stationary tensor, Weight is the moving tensor.

    Tiled computation:
    H/512 * [ I/128 * (Hidden[128, T] @ Weight[128, 512]) ]

    Tile Load:
    Weight tiles are loaded [HTile, I] at a time for efficient memory access:
    H/HTile * [ HTile/512 * [ I/128 * (Hidden[128, T] @ Weight[128, 512]) ] ]

    Column Tiling Optimization:
    For small T, column tiling improves performance by fully utilizing PE engine space.
    E.g., if T=32, the hidden tile [128, 32] leaves unused 32:128 column space in PE engine.

    After Column Tiling:
    ---------------------------
    | col_tile_1 | col_tile_2 | col_tile_3 | col_tile_4 |
    | 32 columns | 32 columns | 32 columns | 32 columns |
    ---------------------------
    - `column_tiling_dim` = [32, 64, 128], chosen based on T.
    - `column_tiling_factor` = 128 / column_tiling_dim, with a maximum factor of 4 → up to 4× speedup.
    - `column_tile` = HTile / column_tiling_factor
    H/HTile * HTile/column_tiling_factor(parallel execution) * column_tile/512 * [ I/128 * (Hidden[128, T] @ Weight[128, 512]) ]

    Key Points:
    -----------
    - Matrix multiplication is tiled along H and I
    - Column tiling improves PE utilization for small T

    Returns:
        Output tensor with shape [T, H]
    """

    # ---------- Configuration and Dimension Setup ----------
    I0, I1, T = hidden.shape
    I, H = weight.shape[0], dims.H_per_shard
    kernel_assert(
        H == dims.H_per_shard,
        f"Weight Sharding mismatch: expected {dims.H_per_shard}, got {H}",
    )

    # Calculate the starting weight index for the down projection.
    # offset the index to avoid anti-dependencies when loading weights in sequence.
    weight_base_idx = tiles.weight_base_idx

    # ---------- Bias handling ----------
    is_bias = bias is not None
    if is_bias:
        # Load bias
        nisa.dma_copy(
            dst=bias_tile[0:1, 0:H],
            src=bias[:, nl.ds(dims.shard_id * H, H)],
            dge_mode=nisa.dge_mode.none,
        )

        # Broadcast bias across T dimension in chunks of 32
        for T_tile in range(T // 32):
            shuffle_mask = [0] * 32
            nisa.nc_stream_shuffle(
                dst=bias_tile[nl.ds(T_tile * 32, 32), 0:H],
                src=bias_tile[0:1, 0:H],
                shuffle_mask=shuffle_mask,
            )

        # Handle remainder of T not divisible by 32
        T_remainder = T % 32
        if T_remainder != 0:
            shuffle_mask = [0] * T_remainder + [255] * (32 - T_remainder)
            nisa.nc_stream_shuffle(
                dst=bias_tile[nl.ds(32 * (T // 32), T_remainder), 0:H],
                src=bias_tile[0:1, 0:H],
                shuffle_mask=shuffle_mask,
            )

    # ---------- Compute matmul  ----------
    # Compute the H size that each column tile will process
    column_tiled_hidden_tile_size = tiles.HTile // dims.column_tiling_factor

    # The number of required PSUM banks per HTile
    num_required_psums_per_HTile = column_tiled_hidden_tile_size // dims._psum_fmax
    kernel_assert(
        column_tiled_hidden_tile_size % dims._psum_fmax == 0,
        "column_tiled_hidden_tile_size must be a multiple of _psum_fmax (512).",
    )

    # Compute available PSUM groups
    num_available_psum_group = dims._psum_bmax // num_required_psums_per_HTile

    for hidden_tiles in TiledRange(H, tiles.HTile):
        # Calculate start offset for H
        h_offset = hidden_tiles.start_offset
        h1_tiles = hidden_tiles.size // dims._psum_fmax
        res_h1_tile = hidden_tiles.size % dims._psum_fmax

        # Allocate PSUM
        psum_offset = (hidden_tiles.index % num_available_psum_group) * num_required_psums_per_HTile
        result_psums = []
        for i in range(num_required_psums_per_HTile):
            result_psum = nl.ndarray(
                shape=(dims._pmax, dims._psum_fmax),
                dtype=nl.float32,
                name=f"down_psum_{hidden_tiles.index}_{psum_offset + i}",
                buffer=nl.psum,
                address=(0, (psum_offset + i) * dims._psum_fmax * 4),
            )
            result_psums.append(result_psum)

        for i_tiles in TiledRange(I, I0):
            # Load weight of [I, HTile] elements into [I0, HTile]
            weight_idx = (
                weight_base_idx + hidden_tiles.index * dims.num_total_128_tiles_per_I + i_tiles.index
            ) % tiles.num_allocated_w_tile
            nisa.dma_copy(
                dst=weight_tiles[weight_idx].ap(
                    pattern=[[weight_tiles[weight_idx].shape[1], i_tiles.size], [1, hidden_tiles.size]],
                    dtype=nl.float8_e4m3 if str(weight.dtype) == "float8e4" else weight.dtype,
                ),
                src=weight.ap(
                    pattern=[[weight.shape[1], i_tiles.size], [1, hidden_tiles.size]],
                    offset=i_tiles.start_offset * weight.shape[1] + dims.shard_id * dims.H_per_shard + h_offset,
                    dtype=nl.float8_e4m3 if str(weight.dtype) == "float8e4" else weight.dtype,
                ),
                dge_mode=nisa.dge_mode.none,
            )

            # Matmult
            for column_tile in TiledRange(h1_tiles, dims.column_tiling_factor):
                for column_idx in range(column_tile.size):
                    weight_offset = (column_tile.start_offset + column_idx) * dims._psum_fmax
                    nisa.nc_matmul(
                        dst=result_psums[column_tile.index][
                            nl.ds(dims.column_tiling_dim * column_idx, T),
                            0 : dims._psum_fmax,
                        ],
                        stationary=hidden[0 : i_tiles.size, i_tiles.index, 0:T],
                        moving=weight_tiles[weight_idx][0 : i_tiles.size, nl.ds(weight_offset, dims._psum_fmax)],
                        tile_position=(0, dims.column_tiling_dim * column_idx),
                        tile_size=(I0, dims.column_tiling_dim),
                    )

            # remainder h1_tile, not a multipe of 512(_psum_fmax)
            if res_h1_tile != 0:
                weight_offset = h1_tiles * dims._psum_fmax
                nisa.nc_matmul(
                    dst=result_psums[h1_tiles][0:T, 0:res_h1_tile],
                    stationary=hidden[0 : i_tiles.size, i_tiles.index, 0:T],
                    moving=weight_tiles[weight_idx][0 : i_tiles.size, nl.ds(weight_offset, res_h1_tile)],
                    tile_position=(0, 0),
                    tile_size=(I0, res_h1_tile),
                )

        # ---------- Accumulate partial PSUM output to SB and optionally apply dequant scale and bias ----------
        bias_tile_view = TensorView(bias_tile) if is_bias else None
        dequant_tile_view = TensorView(dequant_tile) if params.quant_params.is_quant() else None
        for column_tile in TiledRange(h1_tiles, dims.column_tiling_factor):
            for column_idx in range(column_tile.size):
                dst_offset = h_offset + (column_tile.index * dims.column_tiling_factor + column_idx) * dims._psum_fmax
                interleave_copy(
                    index=column_idx + column_tile.index * column_tile.size,
                    dst=output_tile[0:T, nl.ds(dst_offset, dims._psum_fmax)],
                    src=result_psums[column_tile.index][
                        nl.ds(dims.column_tiling_dim * column_idx, T),
                        0 : dims._psum_fmax,
                    ],
                    scale=dequant_tile_view.slice(dim=1, start=dst_offset, end=dst_offset + dims._psum_fmax)
                    if params.quant_params.is_quant_row()
                    else dequant_tile_view,
                    bias=bias_tile_view.slice(dim=1, start=dst_offset, end=dst_offset + dims._psum_fmax)
                    if is_bias
                    else None,
                )

        if res_h1_tile != 0:
            dst_offset = h_offset + h1_tiles * dims._psum_fmax
            interleave_copy(
                index=0,
                dst=output_tile[0:T, nl.ds(dst_offset, res_h1_tile)],
                src=result_psums[h1_tiles][0:T, 0:res_h1_tile],
                scale=dequant_tile_view.slice(dim=1, start=dst_offset, end=dst_offset + res_h1_tile)
                if params.quant_params.is_quant_row()
                else dequant_tile_view,
                bias=bias_tile_view.slice(dim=1, start=dst_offset, end=dst_offset + res_h1_tile) if is_bias else None,
            )


def down_projection_lhs_rhs_swap(
    hidden: nl.ndarray,
    weight: nl.ndarray,
    bias: nl.ndarray,
    output_tile: nl.ndarray,
    weight_tiles: list[nl.ndarray],
    bias_tile: nl.ndarray,
    dequant_tile: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsDownTileCounts,
    params: MLPParameters,
):
    """
    Performs a single Down projection shard on the H using regular matmult with operands swapped

    Computes: Weight[I, H] @ Hidden[I, T] + Optional(Bias[1, H]) → [T, H]
    - Hidden is the moving tensor, Weight is the stationary tensor.

    Tiled computation:
        H/128 * [ I/128 * (Weight[128, 128] @ Hidden[128, T]) ]

    Returns:
        Output tensor with shape [128, H//128, T]
    """

    # ---------- Configuration and Dimension Setup ----------
    I0, I1, T = hidden.shape
    I, H = weight.shape[0], dims.H_per_shard
    H0 = dims.H0

    kernel_assert(
        H == dims.H_per_shard,
        f"Weight sharding mismatch: expected {dims.H_per_shard}, got {H}",
    )

    # Calculate the starting weight index for the down projection.
    # Offset to prevent anti-dependencies with gate/up weight loads, enabling efficient weight tile loading.
    weight_base_idx = tiles.weight_base_idx

    # ---------- Load Bias ----------
    is_bias = bias != None
    if is_bias:
        nisa.dma_copy(
            src=bias.ap(pattern=[[dims.H1_shard, H0], [1, dims.H1_shard]], offset=dims.shard_id * dims.H_per_shard),
            dst=bias_tile[0:H0, 0 : dims.H1_shard],
            dge_mode=nisa.dge_mode.none,
        )

    # ---------- Compute matmul ----------
    perBankT = dims._psum_fmax // T
    num_required_down_psum_banks = math.ceil(dims.H1_shard / perBankT)
    kernel_assert(
        num_required_down_psum_banks <= dims._psum_bmax,
        f"Required psum banks for down projection: {num_required_down_psum_banks}, which exceeds hardware limit of {dims._psum_bmax}, please use CTE mode",
    )

    # Allocate PSUM buffers to store output
    result_psums = []
    for i in range(num_required_down_psum_banks):
        result_psum = nl.ndarray(
            (dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"down_psum_{i}",
            buffer=nl.psum,
            address=(0, i * dims._psum_fmax * 4),
        )
        result_psums.append(result_psum)

    for hidden_tiles in TiledRange(H, tiles.HTile):
        # Calculate starting offset
        h_offset = hidden_tiles.start_offset
        h1_offset = h_offset // H0

        i_tiles = TiledRange(I, I0)
        for i_tile in i_tiles:
            # Load weight of [I, HTile] elements into [128, HTile]
            weight_idx = (
                weight_base_idx + hidden_tiles.index * len(i_tiles) + i_tile.index
            ) % tiles.num_allocated_w_tile
            nisa.dma_copy(
                dst=weight_tiles[weight_idx][0 : i_tile.size, 0 : hidden_tiles.size],
                src=weight[
                    nl.ds(i_tile.start_offset, i_tile.size),
                    nl.ds(dims.shard_id * dims.H_per_shard + h_offset, hidden_tiles.size),
                ],
                dge_mode=nisa.dge_mode.none,
            )

            # When use_tkg_down_proj_optimized_layout is disabled,
            # the weight-tile access pattern uses a stride of H1_shard.
            # When it is enabled, the framework is expected to permute the weights
            # so that the weight tile can be accessed without any stride.
            h1_tiles = TiledRange(hidden_tiles.size, H0)
            if params.use_tkg_down_proj_optimized_layout:
                for h1_tile in h1_tiles:
                    psum_idx = (h1_offset + h1_tile.index) // perBankT
                    psum_offset = (h1_offset + h1_tile.index) % perBankT
                    nisa.nc_matmul(
                        dst=result_psums[psum_idx][0:H0, nl.ds(psum_offset * T, T)],
                        stationary=weight_tiles[weight_idx][0 : i_tile.size, nl.ds(h1_tile.index * H0, H0)],
                        moving=hidden[0 : i_tile.size, i_tile.index, 0:T],
                    )
            else:
                for h1_tile in h1_tiles:
                    psum_idx = (h1_offset + h1_tile.index) // perBankT
                    psum_offset = (h1_offset + h1_tile.index) % perBankT
                    nisa.nc_matmul(
                        dst=result_psums[psum_idx][0:H0, nl.ds(psum_offset * T, T)],
                        stationary=weight_tiles[weight_idx].ap(
                            pattern=[
                                [hidden_tiles.size, i_tile.size],
                                [len(h1_tiles), H0],
                            ],
                            offset=h1_tile.index,
                        ),
                        moving=hidden[0 : i_tile.size, i_tile.index, 0:T],
                    )

    # Reshape output to 2D
    output_tile = output_tile.reshape((H0, dims.H1_shard * T))

    # Copy PSUM output to SB
    bias_tile_view = TensorView(bias_tile) if is_bias else None
    for psum_tiles in TiledRange(dims.H1_shard, perBankT):
        # Number of elements that each PSUM can hold
        perBankElem = perBankT * T
        # Actual number of elements in the current PSUM
        numElements = psum_tiles.size * T

        if params.quant_params.is_quant_row():
            dequant_tile_view = (
                TensorView(dequant_tile)
                .slice(dim=1, start=psum_tiles.index * psum_tiles.size, end=(psum_tiles.index + 1) * psum_tiles.size)
                .expand_dim(dim=2)
                .broadcast(dim=2, size=T)
            )
        else:
            dequant_tile_view = TensorView(dequant_tile) if params.quant_params.is_quant() else None

        bias_tile_view = None
        if is_bias:
            bias_tile_view = (
                TensorView(bias_tile)
                .slice(dim=1, start=psum_tiles.index * psum_tiles.size, end=(psum_tiles.index + 1) * psum_tiles.size)
                .expand_dim(dim=2)
                .broadcast(dim=2, size=T)
            )

        interleave_copy(
            index=psum_tiles.index,
            dst=output_tile[0:H0, nl.ds(psum_tiles.index * perBankElem, numElements)],
            src=result_psums[psum_tiles.index][0:H0, 0:numElements],
            scale=dequant_tile_view,
            bias=bias_tile_view,
        )

    # Reshape output back to 3D
    output_tile = output_tile.reshape((H0, dims.H1_shard, T))


def process_down_projection(
    hidden: nl.ndarray,
    output: nl.ndarray,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    gate_tile_info: MLPTKGConstantsGateUpTileCounts,
    sbm: SbufManager,
):
    """
    Performs the Down projection for MLP (T = BxS).
    Expected hidden tensor shape is [128(I0), I/128, T],
    with a remainder tile shape of [res_I, I/128, T] if I is not a multiple of 128.

    Overview:
    ---------
    hidden @ down_weight + optional(down_bias)
    # [T, H] = [T, I] @ [I, H] + optional([1, H])

    Hardware constraints (max partition size of 128) require tiling along the I dimension:
    # hidden [128, I//128, T] @ down_weight [128, I//128, H]

    Behavior based on `use_tkg_down_proj_column_tiling`:
    ---------------------------------------
    - False: column tiling(`down_projection`)
        hidden[128, T] @ down_weight[128, H] → [T, H]
        Output shape: [T, H]

    - True: operands swapped(`down_projection_lhs_rhs_swap`)
        down_weight[128, H] @ hidden[128, T] → [H, T]
        Further tiling along H: [128, H//128, T]
        Output shape: [128, H//128, T]

    DMA mode:
    ---------
    Based on experiments, Static DMA provides better performance.
    The MLP TKG implementation therefore uses Static DMA for tensor loads.
    If HBM out-of-memory (OOM) issues arise, we can fall back to DGE mode.

    """
    down_w, down_b, down_w_scale, down_in_scale = (
        params.down_proj_weights_tensor,
        params.bias_params.down_proj_bias_tensor,
        params.quant_params.down_w_scale,
        params.quant_params.down_in_scale,
    )

    sbm.open_scope()  # Begin SBUF allocation scope

    # ---------------- Allocate Bias Tile ----------------
    bias_tile = None
    if mlpp_has_down_projection_bias(params):
        if params.use_tkg_down_proj_column_tiling:
            bias_tile = sbm.alloc_stack(
                (dims.T, dims.H_per_shard),
                dtype=down_b.dtype,
                name=f"down_broadcasted_bias",
                buffer=nl.sbuf,
            )
        else:
            bias_tile = sbm.alloc_stack(
                (dims.H0, dims.H1_shard),
                dtype=down_b.dtype,
                name=f"down_bias",
                buffer=nl.sbuf,
            )

    dequant_tile = None
    # ---------------- Static quantization ----------------
    if params.quant_params.is_quant_static():
        par_dims = dims.T if params.use_tkg_down_proj_column_tiling else dims.H0
        dequant_tile = sbm.alloc_stack((par_dims, 1), dtype=down_w_scale.dtype, name=f"down_w_scale_sb", align=4)
        nisa.dma_copy(dst=dequant_tile[:par_dims, :], src=down_w_scale[:par_dims, :], dge_mode=nisa.dge_mode.none)

        # does not need to update addr since it's going to be dropped after input quant
        in_scale_sb = sbm.alloc_heap((nl.tile_size.pmax, 1), dtype=nl.float32)
        nisa.dma_copy(dst=in_scale_sb, src=down_in_scale, dge_mode=nisa.dge_mode.none)

        # pre-apply input scales onto the weight scaling
        nisa.activation(dst=dequant_tile, op=nl.copy, data=dequant_tile, scale=in_scale_sb[:par_dims, :])

        # quantize the inputs
        nisa.reciprocal(dst=in_scale_sb, data=in_scale_sb)
        nisa.activation(dst=hidden, op=nl.copy, data=hidden, scale=in_scale_sb[: dims.H0, :])
        max_pos_val = get_max_positive_value_for_dtype(down_w.dtype)
        nisa.tensor_scalar(
            dst=hidden, data=hidden, op0=nl.minimum, operand0=max_pos_val, op1=nl.maximum, operand1=-max_pos_val
        )
        sbm.pop_heap()  # in_scale_sb

    # ---------------- Row quantization ----------------
    elif params.quant_params.is_quant_row():
        if params.use_tkg_down_proj_column_tiling:
            dequant_tile = sbm.alloc_stack(
                (dims.T, dims.H_per_shard), dtype=down_w_scale.dtype, name=f"down_w_scale_sb", align=4
            )
            nisa.dma_copy(
                dst=dequant_tile[0 : dims.T, 0 : dims.H_per_shard],
                src=down_w_scale[0 : dims.T, nl.ds(dims.shard_id * dims.H_per_shard, dims.H_per_shard)],
                dge_mode=nisa.dge_mode.none,
            )
        else:
            dequant_tile = sbm.alloc_stack(
                (dims.H0, dims.H1_shard), dtype=down_w_scale.dtype, name=f"down_w_scale_sb", align=4
            )
            nisa.dma_copy(
                src=down_w_scale.ap(
                    pattern=[[dims.H1_shard, dims.H0], [1, dims.H1_shard]], offset=dims.shard_id * dims.H_per_shard
                ),
                dst=dequant_tile[0 : dims.H0, 0 : dims.H1_shard],
                dge_mode=nisa.dge_mode.none,
            )

    # ---------------- Allocate Weight Tiles ----------------
    # By calculating the remaining SBUF space, we allocate as many weight tiles as possible
    remaining_space = sbm.get_free_space()
    kernel_assert(remaining_space > 0, f"Not enough memory for down projection weight")

    # Calculate tile info
    current_address = sbm.get_stack_curr_addr()
    tiles = MLPTKGConstants.calculate_down_tiles(current_address, remaining_space, params, dims, gate_tile_info)

    weight_tiles = []
    for i in range(tiles.num_allocated_w_tile):
        weight_tile = sbm.alloc_stack(
            (dims.I0, tiles.HTile),
            name=f"down_w_tile_{i}",
            dtype=nl.float8_e4m3 if str(down_w.dtype) == "float8e4" else down_w.dtype,
            buffer=nl.sbuf,
        )
        weight_tiles.append(weight_tile)

    # ---------------- Down Projection ----------------
    if params.use_tkg_down_proj_column_tiling:
        down_projection(
            hidden=hidden,
            weight=down_w,
            bias=down_b,
            output_tile=output,
            weight_tiles=weight_tiles,
            bias_tile=bias_tile,
            dequant_tile=dequant_tile,
            dims=dims,
            tiles=tiles,
            params=params,
        )

    else:
        down_projection_lhs_rhs_swap(
            hidden=hidden,
            weight=down_w,
            bias=down_b,
            output_tile=output,
            weight_tiles=weight_tiles,
            bias_tile=bias_tile,
            dequant_tile=dequant_tile,
            dims=dims,
            tiles=tiles,
            params=params,
        )

    sbm.close_scope()  # Close SBUF scope

    return output, tiles
