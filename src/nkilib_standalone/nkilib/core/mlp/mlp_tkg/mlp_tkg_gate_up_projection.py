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

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager, sizeinbytes

# common utils
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_nl_act_fn_from_type, get_max_positive_value_for_dtype
from ...utils.logging import Logger
from ...utils.tiled_range import TiledRange
from ...utils.interleave_copy import interleave_copy
from ...utils.common_types import QuantizationType
from ...utils.tensor_view import TensorView

# MLP utils
from ..mlp_parameters import (
    MLPParameters,
    mlpp_has_gate_projection_bias,
    mlpp_has_up_projection_bias,
)
from .mlp_tkg_constants import (
    MLPTKGConstants,
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsGateUpTileCounts,
)


def gate_up_projection(
    hidden: nl.ndarray,
    unsharded_weight: nl.ndarray,
    shard_dim0: tuple[int, int],
    shard_dim1: tuple[int, int],
    bias: nl.ndarray,
    deqaunt_scale: nl.ndarray,
    output_tile: nl.ndarray,
    weight_tiles: list[nl.ndarray],
    bias_tile: nl.ndarray,
    dequant_tile: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsGateUpTileCounts,
    params: MLPParameters,
    op_name: str,
):
    """
    Performs a single Gate or Up projection shard on the H.

    Computes: Hidden[H, T] @ Weight[H, I] + Optional(Bias[1, I]) → [T, I]
    - Hidden is the stationary tensor, Weight is the moving tensor.

    Tiled computation:
    H/128 * [ I/512 * (Hidden[128, T] @ Weight[128, 512]) ]

    Tile Load:
    Weight tiles are loaded [HTile, I] at a time for efficient memory access:
    H/HTile * [ HTile/128 * [ I/512 * (Hidden[128, T] @ Weight[128, 512]) ] ]

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
    H/HTile * HTile/column_tiling_factor(parallel execution) * column_tile/128 * [ I/512 * (Hidden[128, T] @ Weight[128, 512]) ]

    Key Points:
    -----------
    - Intermediate projection tensors are always fp32 for better numerical accuracy
    - Bias is applied on one core only to avoid double-counting (sharding along H)
    - Matrix multiplication is tiled along H and I
    - Column tiling improves PE utilization for small T

    Returns:
        Output tensor with shape [T, I]
    """

    # ---------- Configuration and Dimension Setup ----------
    H0, T, H1 = hidden.shape
    H = shard_dim0[1] - shard_dim0[0]
    I = shard_dim1[1] - shard_dim1[0]
    i_offset = shard_dim1[0]
    num_allocated_w_tile = tiles.num_allocated_w_tile

    # Sanity checks for sharding
    kernel_assert(
        I <= dims.max_I_shard_size,
        f"{op_name}_projection supports I <= {dims.max_I_shard_size}",
    )
    kernel_assert(
        H == dims.H_per_shard,
        f"Weight sharding mismatch: expected {dims.H_per_shard}, got {H}",
    )

    # For 'up' projection, offset weight index to avoid anti-dependencies with gate weights.
    # The kernel shares weight tiles for gate and up projection
    # this treats them as a ring buffer so up weights load after gate weights for efficient reuse.
    weight_base_idx = tiles.num_HTiles % num_allocated_w_tile if op_name == "up" else 0

    # ---------- Allocate PSUM buffers ----------
    result_psums = []
    for i in range(tiles.num_allocated_psums):
        result_psum = nl.ndarray(
            (dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"{op_name}_psum_shard_{i_offset}_{i}",
            buffer=nl.psum,
            address=(0, i * dims._psum_fmax * 4),
        )
        result_psums.append(result_psum)

    # ---------- Bias handling ----------
    # Only apply bias on one core to avoid double-counting (sharding along H)
    is_bias = bias is not None and dims.shard_id == 0
    if is_bias:
        # Load bias
        nisa.dma_copy(
            dst=bias_tile[0:1, 0:I],
            src=bias[0:1, nl.ds(i_offset, I)],
            dge_mode=nisa.dge_mode.none,
        )

        # Broadcast bias across T dimension in chunks of 32
        for T_tile in range(T // 32):
            shuffle_mask = [0] * 32
            nisa.nc_stream_shuffle(
                dst=bias_tile[nl.ds(T_tile * 32, 32), 0:I],
                src=bias_tile[0:1, 0:I],
                shuffle_mask=shuffle_mask,
            )

        # Handle remainder of T not divisible by 32
        T_remainder = T % 32
        if T_remainder != 0:
            shuffle_mask = [0] * T_remainder + [255] * (32 - T_remainder)
            nisa.nc_stream_shuffle(
                dst=bias_tile[nl.ds(32 * (T // 32), T_remainder), 0:I],
                src=bias_tile[0:1, 0:I],
                shuffle_mask=shuffle_mask,
            )

    # ---------- Load dequant scale ----------
    if params.quant_params.is_quant_row():
        nisa.dma_copy(
            dst=dequant_tile[0:T, 0:I],
            src=deqaunt_scale[0:T, nl.ds(i_offset, I)],
            dge_mode=nisa.dge_mode.none,
        )

    # ---------- Matrix multiplication ----------
    used_columns = 0

    # Gate Up Projection
    for hidden_tiles in TiledRange(H, tiles.HTile):
        # Compute start offset
        h_offset = hidden_tiles.index * tiles.num_128_tiles_per_HTile
        weight_shard_offset = shard_dim0[0] * unsharded_weight.shape[1] + i_offset
        h1_tiles = hidden_tiles.size // H0

        # Load weight tile [HTile, I] → SBUF layout [H0, HTile/H0, I]
        weight_idx = (weight_base_idx + hidden_tiles.index) % num_allocated_w_tile
        nisa.dma_copy(
            dst=weight_tiles[weight_idx][0:H0, 0:h1_tiles, 0:I],
            src=unsharded_weight.ap(
                pattern=[
                    [H1 * unsharded_weight.shape[1], H0],
                    [unsharded_weight.shape[1], h1_tiles],
                    [1, I],
                ],
                offset=h_offset * dims.I + weight_shard_offset,
                dtype=nl.float8_e4m3 if str(unsharded_weight.dtype) == "float8e4" else unsharded_weight.dtype,
            ),
            dge_mode=nisa.dge_mode.none,
        )

        # Matmult
        for column_tile in TiledRange(h1_tiles, dims.column_tiling_factor):
            for column_idx in range(column_tile.size):
                column_tile_offset = dims.column_tiling_factor * column_tile.index + column_idx
                for i_tiles in TiledRange(I, dims._psum_fmax):
                    nisa.nc_matmul(
                        dst=result_psums[i_tiles.index][
                            nl.ds(dims.column_tiling_dim * column_idx, T),
                            0 : i_tiles.size,
                        ],
                        stationary=hidden.ap(
                            pattern=[[T * H1, H0], [H1, T]],
                            offset=h_offset + column_tile_offset,
                        ),
                        moving=weight_tiles[weight_idx][
                            0:H0,
                            column_tile_offset,
                            nl.ds(i_tiles.start_offset, i_tiles.size),
                        ],
                        tile_position=(0, dims.column_tiling_dim * column_idx),
                        tile_size=(H0, dims.column_tiling_dim),
                    )
            # Update used column numbers
            used_columns = max(used_columns, column_tile.size)

    # ---------- Accumulate PSUMs into output ----------
    for i_tiles in TiledRange(I, dims._psum_fmax):
        dst_offset = shard_dim1[0] + i_tiles.start_offset
        # Copy PSUM to SBUF
        nisa.activation(
            dst=output_tile[0:T, nl.ds(dst_offset, i_tiles.size)],
            data=result_psums[i_tiles.index][0:T, 0 : i_tiles.size],
            op=nl.copy,
        )

        # Accumulate PSUMs to SBUF
        for factor_idx in range(1, used_columns):
            nisa.tensor_tensor(
                dst=output_tile[0:T, nl.ds(dst_offset, i_tiles.size)],
                data1=result_psums[i_tiles.index][nl.ds(dims.column_tiling_dim * factor_idx, T), 0 : i_tiles.size],
                data2=output_tile[0:T, nl.ds(dst_offset, i_tiles.size)],
                op=nl.add,
            )

    if params.quant_params.is_quant() or is_bias:
        if params.quant_params.is_quant():
            dequant_tile_view = TensorView(dequant_tile)
            if params.quant_params.is_quant_row():
                dequant_tile_view = dequant_tile_view.slice(dim=1, start=0, end=I)

        if is_bias:
            bias_tile_view = TensorView(bias_tile).slice(dim=1, start=0, end=I)

        interleave_copy(
            dst=output_tile[0:T, nl.ds(i_offset, I)],
            src=output_tile[0:T, nl.ds(i_offset, I)],
            scale=dequant_tile_view if params.quant_params.is_quant() else None,
            bias=bias_tile_view if is_bias else None,
        )


def gate_up_projection_lhs_rhs_swap(
    hidden: nl.ndarray,
    unsharded_weight: nl.ndarray,
    shard_dim0: tuple[int, int],
    shard_dim1: tuple[int, int],
    bias: nl.ndarray,
    deqaunt_scale: nl.ndarray,
    output_tile: nl.ndarray,
    weight_tiles: list[nl.ndarray],
    bias_tile: nl.ndarray,
    dequant_tile: nl.ndarray,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsGateUpTileCounts,
    params: MLPParameters,
    op_name: str,
):
    """
    Performs a single Gate or Up projection shard on the H using regular matmult with operands swapped

    Computes: Weight[H, I] @ Hidden[H, T] + Optional(Bias[1, I]) → [T, I]
    - Hidden is the moving tensor, Weight is the stationary tensor.

    Tiled computation:
        H/128 * [ I/128 * (Weight[128, 128] @ Hidden[128, T]) ]

    Returns:
        Output tensor with shape [128, I/128, T]
    """

    # ---------- Configuration and Dimension Setup ----------
    H0, T, H1 = hidden.shape
    H = shard_dim0[1] - shard_dim0[0]
    I = shard_dim1[1] - shard_dim1[0]
    I0 = dims.I0
    i_offset = shard_dim1[0]
    i1_offset = shard_dim1[0] // I0
    weight_shard_offset = shard_dim0[0] * unsharded_weight.shape[1] + i_offset
    num_allocated_w_tile = tiles.num_allocated_w_tile

    # Sanity checks for sharding
    kernel_assert(
        I <= dims.max_I_shard_size,
        f"{op_name}_projection only supports I <= {dims.max_I_shard_size}",
    )
    kernel_assert(
        H == dims.H_per_shard,
        f"Weight sharding mismatch: expected {dims.H_per_shard}, got {H}",
    )

    # For 'up' projection, offset weight index to avoid anti-dependencies with gate weights.
    # The kernel shares weight tiles for gate and up projection
    # this treats them as a ring buffer so up weights load after gate weights for efficient reuse.
    weight_base_idx = tiles.num_HTiles % num_allocated_w_tile if op_name == "up" else 0

    # Allocate PSUM buffers to store output
    result_psums = []
    for i_tiles in TiledRange(I, I0):
        result_psum = nl.ndarray(
            shape=(dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"{op_name}_psum_shard_{shard_dim1[1]}_{i_tiles.index}",
            buffer=nl.psum,
            address=(0, i_tiles.index * dims._psum_fmax * 4),
        )
        result_psums.append(result_psum)

    # ---------- Bias handling ----------
    # Only apply bias on one core to avoid double-counting (sharding along H)
    is_bias = bias != None and dims.shard_id == 0

    # Number of full/res 128(_pmax)-elements tiles along I
    num_128_I_tiles = I // I0
    res_128_I_tiles = I % I0
    num_total_128_I_tiles = num_128_I_tiles + (res_128_I_tiles != 0)

    if is_bias:
        # Load bias tensor with proper reshaping based on intermediate dimension alignment
        # Case 1: Intermediate dimension is multiple of 128
        I0_res = I % I0
        if I0_res == 0:
            # Bias shape [1, I] can be cleanly reshaped to [I0, I/I0] tiles
            # Load all complete 128-element tiles for this shard
            nisa.dma_transpose(
                src=bias.ap(
                    pattern=[[I0, num_total_128_I_tiles], [1, 1], [1, 1], [1, I0]],
                    offset=i_offset,
                ),
                dst=bias_tile.ap(
                    pattern=[
                        [num_total_128_I_tiles, I0],
                        [1, 1],
                        [1, 1],
                        [1, num_total_128_I_tiles],
                    ]
                ),
                dge_mode=nisa.dge_mode.none,
            )
        # Case 2: Intermediate dimension not multiple of 128 (requires remainder handling)
        elif dims.I % 128 != 0:
            # Load I0 elements first
            nisa.dma_transpose(
                src=bias.ap(
                    pattern=[[I0, num_128_I_tiles], [1, 1], [1, 1], [1, I0]],
                    offset=i_offset,
                ),
                dst=bias_tile.ap(
                    pattern=[
                        [num_total_128_I_tiles, I0],
                        [1, 1],
                        [1, 1],
                        [1, num_128_I_tiles],
                    ]
                ),
                dge_mode=nisa.dge_mode.none,
            )
            # Load remainder elements
            if I0_res != 0:
                nisa.dma_copy(
                    src=bias.ap(
                        pattern=[[1, I0_res], [1, 1]],
                        offset=num_128_I_tiles * I0 + i_offset,
                    ),
                    dst=bias_tile.ap(
                        pattern=[[num_total_128_I_tiles, I0_res], [1, 1]],
                        offset=num_128_I_tiles,
                    ),
                    dge_mode=nisa.dge_mode.none,
                )

    # ---------- Load dequant scale ----------
    if params.quant_params.is_quant_row():
        # Load dequant_πscale tensor with proper reshaping based on intermediate dimension alignment
        # Case 1: Intermediate dimension is multiple of 128
        I0_res = I % I0
        if I0_res == 0:
            # dequant_scale shape [1, I] can be cleanly reshaped to [I0, I/I0] tiles
            # Load all complete 128-element tiles for this shard
            nisa.dma_transpose(
                src=deqaunt_scale.ap(
                    pattern=[[I0, num_total_128_I_tiles], [1, 1], [1, 1], [1, I0]],
                    offset=i_offset,
                ),
                dst=dequant_tile.ap(
                    pattern=[
                        [dequant_tile.shape[1], I0],
                        [1, 1],
                        [1, 1],
                        [1, num_total_128_I_tiles],
                    ]
                ),
                dge_mode=nisa.dge_mode.none,
            )
        # Case 2: Intermediate dimension not multiple of 128 (requires remainder handling)
        elif dims.I % 128 != 0:
            # Load I0 elements first
            nisa.dma_transpose(
                src=deqaunt_scale.ap(
                    pattern=[[I0, num_128_I_tiles], [1, 1], [1, 1], [1, I0]],
                    offset=i_offset,
                ),
                dst=dequant_tile.ap(
                    pattern=[
                        [dequant_tile.shape[1], I0],
                        [1, 1],
                        [1, 1],
                        [1, num_128_I_tiles],
                    ]
                ),
                dge_mode=nisa.dge_mode.none,
            )
            # Load remainder elements
            if I0_res != 0:
                nisa.dma_copy(
                    src=deqaunt_scale.ap(
                        pattern=[[1, I0_res], [1, 1]],
                        offset=num_128_I_tiles * I0 + i_offset,
                    ),
                    dst=dequant_tile.ap(
                        pattern=[[dequant_tile.shape[1], I0_res], [1, 1]],
                        offset=num_128_I_tiles,
                    ),
                    dge_mode=nisa.dge_mode.none,
                )

    # ---------- Matrix multiplication ----------
    # Gate Up Projection
    for hidden_tiles in TiledRange(H, tiles.HTile):
        # Compute start offset
        h_start_offset = hidden_tiles.index * (tiles.HTile // H0)

        # Load weight tile [HTile, I] → SBUF layout [H0, HTile/H0, I]
        h1_size = hidden_tiles.size // H0
        weight_idx = (weight_base_idx + hidden_tiles.index) % num_allocated_w_tile
        nisa.dma_copy(
            dst=weight_tiles[weight_idx][0:H0, 0:h1_size, 0:I],
            src=unsharded_weight.ap(
                pattern=[
                    [unsharded_weight.shape[1] * dims.H1_shard, H0],
                    [unsharded_weight.shape[1], h1_size],
                    [1, I],
                ],
                offset=h_start_offset * unsharded_weight.shape[1] + weight_shard_offset,
            ),
            dge_mode=nisa.dge_mode.none,
        )

        # Matmult
        for h1_tiles in TiledRange(hidden_tiles.size, H0):
            for i_tiles in TiledRange(I, I0):
                nisa.nc_matmul(
                    result_psums[i_tiles.index][0 : i_tiles.size, 0:T],
                    weight_tiles[weight_idx][0:H0, h1_tiles.index, nl.ds(i_tiles.index * I0, i_tiles.size)],
                    hidden[0:H0, 0:T, h_start_offset + h1_tiles.index],
                )

    # ---------- Accumulate partial PSUMs to output ----------
    for i_tiles in TiledRange(I, I0):
        # Set tile view for dequant tile
        dequant_tile_view = None
        if params.quant_params.is_quant():
            dequant_tile_view = TensorView(dequant_tile).slice(dim=0, start=0, end=i_tiles.size)
            if params.quant_params.is_quant_row():
                dequant_tile_view = dequant_tile_view.slice(
                    dim=1, start=i_tiles.index, end=i_tiles.index + 1
                ).broadcast(dim=1, size=T)
        # Set tile view for bias tile
        bias_tile_view = None
        if is_bias:
            bias_tile_view = (
                TensorView(bias_tile)
                .slice(dim=0, start=0, end=i_tiles.size)
                .slice(dim=1, start=i_tiles.index, end=i_tiles.index + 1)
                .broadcast(dim=1, size=T)
            )

        # PSUM to SBUF copy while applying dequant and bias tensor optionally
        interleave_copy(
            index=i_tiles.index,
            dst=output_tile.ap(
                pattern=[[dims.num_total_128_tiles_per_I * T, i_tiles.size], [1, T]],
                offset=(i1_offset + i_tiles.index) * T,
            ),
            src=result_psums[i_tiles.index][0 : i_tiles.size, 0:T],
            scale=dequant_tile_view,
            bias=bias_tile_view,
        )


def process_gate_up_projection(
    hidden: nl.ndarray,
    output: nl.ndarray,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
):
    """
    Performs the Gate/Up projection for MLP (T = BxS).
    Expected hidden tensor shape is [128(H0), T, H//128]

    Overview:
    ---------
    gate_proj_out [T, I] = hidden [H, T] @ gate_weight [H, I] + optional(gate_bias [1, I])
    act_gate_proj [T, I] = Activation_Fn(gate_proj_out [T, I])
    up_proj_out [T, I]   = hidden [H, T] @ up_weight [H, I] + optional(up_bias)
    hidden[T, I] = act_gate_proj [T, I] * up_proj_out [T, I]  # elementwise multiplication

    Hardware constraints (max partition size of 128) require tiling along the H dimension:
    # hidden [128, BxS, H//128] @ gate/up_weight [128, H//128, I]

    Behavior based on `use_tkg_gate_up_proj_column_tiling`:
    ------------------------------------------
    - True: column tiling(`gate_up_projection`)
        hidden[128, BxS] @ gate/up_weight[128, I] → [T, I]
    - False: regular matmult with operands swapped(`gate_up_projection_lhs_rhs_swap`)
        gate/up_weight[128, I] @ hidden[128, BxS] → [I, T]
        Further tiling along I: [128, I//128, T]

    DMA mode:
    ---------
    Based on experiments, Static DMA provides better performance.
    The MLP TKG implementation therefore uses Static DMA for tensor loads.
    If HBM out-of-memory (OOM) issues arise, we can fall back to DGE mode.

    Note:
    -----
    Intermediate gate/up projection tensors are always fp32 to improve numerical accuracy.
    Hidden tensor in SBUF has layout [H, T], tiled as [128(H0), T, H//128] to fully utilize the partition dimension.

    """
    gate_w, up_w = params.gate_proj_weights_tensor, params.up_proj_weights_tensor
    gate_b, up_b = (
        params.bias_params.gate_proj_bias_tensor,
        params.bias_params.up_proj_bias_tensor,
    )
    gate_w_scale, up_w_scale = (
        params.quant_params.gate_w_scale,
        params.quant_params.up_w_scale,
    )
    input_scale = params.quant_params.gate_up_in_scale

    sbm.open_scope()  # Begin SBUF allocation scope

    # ---------------- Allocate Gate/Up/Bias/DequantScale Tiles ----------------
    # Note: intermediate tiles are fp32 for better numerical accuracy
    bias_tile = None
    bias_size = 0
    if params.use_tkg_gate_up_proj_column_tiling:
        if not params.skip_gate_proj:
            gate_sb_fp32 = sbm.alloc_stack(
                (dims.T, dims.I),
                dtype=nl.float32,
                name="gate_sbuf_fp32",
                buffer=nl.sbuf,
                align=4,
            )
        up_sb_fp32 = sbm.alloc_stack(
            (dims.T, dims.I),
            dtype=nl.float32,
            name="up_sbuf_fp32",
            buffer=nl.sbuf,
            align=4,
        )
        if mlpp_has_gate_projection_bias(params) or mlpp_has_up_projection_bias(params):
            bias_tile = sbm.alloc_stack(
                (dims.T, dims.max_I_shard_size),
                dtype=gate_b.dtype,
                name="gate_up_broadcasted_bias",
                buffer=nl.sbuf,
            )
    else:
        if not params.skip_gate_proj:
            gate_sb_fp32 = sbm.alloc_stack(
                (dims.I0, dims.num_total_128_tiles_per_I, dims.T),
                dtype=nl.float32,
                name="gate_sbuf_fp32",
                buffer=nl.sbuf,
                align=4,
            )
        up_sb_fp32 = sbm.alloc_stack(
            (dims.I0, dims.num_total_128_tiles_per_I, dims.T),
            dtype=nl.float32,
            name="up_sbuf_fp32",
            buffer=nl.sbuf,
            align=4,
        )
        # Allocate the bias/dequant tile inside the loop due to the DMA transpose 32-byte address alignment requirement.
        if mlpp_has_gate_projection_bias(params) or mlpp_has_up_projection_bias(params):
            bias_size = dims.num_total_128_tiles_per_I * sizeinbytes(gate_b.dtype)

    # ---------------- Static quantization ----------------
    gate_dequant_tile = up_dequant_tile = None
    # For static quantization, directly load the scales into sbuf
    if params.quant_params.is_quant_static():
        par_dim = dims.T if params.use_tkg_gate_up_proj_column_tiling else dims.I0

        # Allocate Dequant tile
        gate_dequant_tile = sbm.alloc_stack(
            (par_dim, 1),
            dtype=gate_w_scale.dtype,
            name=f"gate_w_scale_sb",
            buffer=nl.sbuf,
            align=4,
        )
        up_dequant_tile = sbm.alloc_stack(
            (par_dim, 1),
            dtype=up_w_scale.dtype,
            name=f"up_w_scale_sb",
            buffer=nl.sbuf,
            align=4,
        )

        # Load gate up dequantization scale
        nisa.dma_copy(dst=gate_dequant_tile[0:par_dim, :], src=gate_w_scale[0:par_dim, :], dge_mode=nisa.dge_mode.none)
        nisa.dma_copy(dst=up_dequant_tile[0:par_dim, :], src=up_w_scale[0:par_dim, :], dge_mode=nisa.dge_mode.none)

        in_scale_sb = sbm.alloc_heap((nl.tile_size.pmax, 1), dtype=nl.float32)
        nisa.dma_copy(dst=in_scale_sb, src=input_scale, dge_mode=nisa.dge_mode.none)

        # pre-apply input scales onto the weight scaling
        nisa.activation(dst=gate_dequant_tile, op=nl.copy, data=gate_dequant_tile, scale=in_scale_sb[0:par_dim, :])
        nisa.activation(dst=up_dequant_tile, op=nl.copy, data=up_dequant_tile, scale=in_scale_sb[0:par_dim, :])

        # quantize the inputs
        nisa.reciprocal(dst=in_scale_sb, data=in_scale_sb)
        nisa.activation(dst=hidden, op=nl.copy, data=hidden, scale=in_scale_sb[: dims.H0, :])
        max_pos_val = get_max_positive_value_for_dtype(gate_w.dtype)
        nisa.tensor_scalar(
            dst=hidden, data=hidden, op0=nl.minimum, operand0=max_pos_val, op1=nl.maximum, operand1=-max_pos_val
        )
        sbm.pop_heap()  # in_scale_sb

    # ---------------- Row quantization ----------------
    # Scale data is loaded in projection function
    elif params.quant_params.is_quant_row():
        if params.use_tkg_gate_up_proj_column_tiling:
            gate_dequant_tile = sbm.alloc_stack(
                (dims.T, min(dims.max_I_shard_size, dims.I)),
                dtype=gate_w_scale.dtype,
                name=f"gate_w_scale_sb",
                buffer=nl.sbuf,
                align=4,
            )
        else:
            gate_dequant_tile = sbm.alloc_stack(
                (dims.I0, min(dims.max_I_shard_size // dims.I0, dims.num_total_128_tiles_per_I)),
                dtype=gate_w_scale.dtype,
                name=f"gate_w_scale_sb",
                buffer=nl.sbuf,
                align=32,
            )
        up_dequant_tile = gate_dequant_tile

    # ---------------- Allocate Receive Buffer for LNC > 1 ----------------
    gate_up_recv = None
    if dims.num_shards > 1:
        gate_up_recv = sbm.alloc_stack(
            up_sb_fp32.shape,
            dtype=nl.float32,
            buffer=nl.sbuf,
            name="gate_up_recv_buffer_fp32",
        )

    # ---------------- Allocate Weight Tiles ----------------
    # By calculating the remaining SBUF space, we allocate as many weight tiles as possible
    remaining_space = sbm.get_free_space() - bias_size
    kernel_assert(remaining_space > 0, "Not enough memory for gate/up weights")
    current_address = sbm.get_stack_curr_addr()
    tiles = MLPTKGConstants.calculate_gate_up_tiles(current_address, remaining_space, params, dims)

    sbm.open_scope()  # Begin Weight SBUF allocation scope

    weight_tiles = []
    for i in range(tiles.num_allocated_w_tile):
        weight_tile = sbm.alloc_stack(
            (dims.H0, tiles.num_128_tiles_per_HTile, dims.I),
            name=f"gate_up_w_tile_{i}",
            dtype=nl.float8_e4m3 if str(gate_w.dtype) == "float8e4" else gate_w.dtype,
        )
        weight_tiles.append(weight_tile)

    # ---------------- Gate/Up Projection ----------------
    if params.use_tkg_gate_up_proj_column_tiling:
        for i_tiles in TiledRange(dims.I, dims.max_I_shard_size):
            h_offset = dims.H1_offset * dims.H0
            I_start = i_tiles.start_offset
            I_end = min(I_start + dims.max_I_shard_size, dims.I)

            if not params.skip_gate_proj:
                # Gate projection
                gate_up_projection(
                    hidden=hidden,
                    unsharded_weight=gate_w,
                    shard_dim0=(h_offset, h_offset + dims.H_per_shard),
                    shard_dim1=(I_start, I_end),
                    bias=gate_b,
                    deqaunt_scale=gate_w_scale,
                    output_tile=gate_sb_fp32,
                    weight_tiles=weight_tiles,
                    bias_tile=bias_tile,
                    dequant_tile=gate_dequant_tile,
                    dims=dims,
                    tiles=tiles,
                    params=params,
                    op_name="gate",
                )

            # Up projection
            gate_up_projection(
                hidden=hidden,
                unsharded_weight=up_w,
                shard_dim0=(h_offset, h_offset + dims.H_per_shard),
                shard_dim1=(I_start, I_end),
                bias=up_b,
                deqaunt_scale=up_w_scale,
                output_tile=up_sb_fp32,
                weight_tiles=weight_tiles,
                bias_tile=bias_tile,
                dequant_tile=up_dequant_tile,
                dims=dims,
                tiles=tiles,
                params=params,
                op_name="up",
            )
    else:
        for i_tiles in TiledRange(dims.I, dims.max_I_shard_size):
            h_offset = dims.H1_offset * dims.H0
            I_start = i_tiles.start_offset
            I_end = min(I_start + dims.max_I_shard_size, dims.I)
            num_total_128_I_tiles = math.ceil(i_tiles.size / dims.I0)

            if mlpp_has_gate_projection_bias(params) or mlpp_has_up_projection_bias(params):
                bias_tile = sbm.alloc_stack(
                    (dims.I0, num_total_128_I_tiles),
                    dtype=gate_b.dtype,
                    name=f"gate_up_bias_{i_tiles.index}",
                    buffer=nl.sbuf,
                    align=32,
                )

            if not params.skip_gate_proj:
                # Gate projection
                gate_up_projection_lhs_rhs_swap(
                    hidden=hidden,
                    unsharded_weight=gate_w,
                    shard_dim0=(h_offset, h_offset + dims.H_per_shard),
                    shard_dim1=(I_start, I_end),
                    bias=gate_b,
                    deqaunt_scale=gate_w_scale,
                    output_tile=gate_sb_fp32,
                    weight_tiles=weight_tiles,
                    bias_tile=bias_tile,
                    dequant_tile=gate_dequant_tile,
                    dims=dims,
                    tiles=tiles,
                    params=params,
                    op_name="gate",
                )

            # Up projection
            gate_up_projection_lhs_rhs_swap(
                hidden=hidden,
                unsharded_weight=up_w,
                shard_dim0=(h_offset, h_offset + dims.H_per_shard),
                shard_dim1=(I_start, I_end),
                bias=up_b,
                deqaunt_scale=up_w_scale,
                output_tile=up_sb_fp32,
                weight_tiles=weight_tiles,
                bias_tile=bias_tile,
                dequant_tile=up_dequant_tile,
                dims=dims,
                tiles=tiles,
                params=params,
                op_name="up",
            )

    sbm.close_scope()  # Close Weight SBUF allocation scope

    if params.skip_gate_proj:
        # ---------------- Up Projection Multi-Shard Communication ----------------
        # Receive up projection output from the other neuron core when LNC > 1
        if dims.num_shards > 1:
            nisa.sendrecv(
                src=up_sb_fp32,
                dst=gate_up_recv,
                send_to_rank=(1 - dims.shard_id),
                recv_from_rank=(1 - dims.shard_id),
                pipe_id=0,
            )
            nisa.tensor_tensor(dst=up_sb_fp32, data1=up_sb_fp32, data2=gate_up_recv, op=nl.add)

        #  ---------------- Optionally perform clamping on up projection results  ----------------
        if params.up_clamp_upper_limit is not None:
            nisa.tensor_scalar(data=up_sb_fp32, dst=up_sb_fp32, op0=nl.minimum, operand0=params.up_clamp_upper_limit)
        if params.up_clamp_lower_limit is not None:
            nisa.tensor_scalar(data=up_sb_fp32, dst=up_sb_fp32, op0=nl.maximum, operand0=params.up_clamp_lower_limit)

        # ---------------- Up Activation ----------------
        nisa.activation(
            dst=up_sb_fp32[:, :],
            op=get_nl_act_fn_from_type(params.activation_fn),
            data=up_sb_fp32,
            scale=1.0,
        )
        nisa.tensor_copy(dst=output, src=up_sb_fp32, engine=nki.isa.vector_engine)

    else:
        # ---------------- Gate Projection Multi-Shard Communication ----------------
        # Receive gate projection output from the other neuron core when LNC > 1
        if dims.num_shards > 1:
            nisa.sendrecv(
                src=gate_sb_fp32,
                dst=gate_up_recv,
                send_to_rank=(1 - dims.shard_id),
                recv_from_rank=(1 - dims.shard_id),
                pipe_id=0,
            )
            nisa.tensor_tensor(dst=gate_sb_fp32, data1=gate_sb_fp32, data2=gate_up_recv, op=nl.add)

        #  ---------------- Optionally perform clamping on gate projection results  ----------------
        if params.gate_clamp_upper_limit is not None:
            nisa.tensor_scalar(
                data=gate_sb_fp32, dst=gate_sb_fp32, op0=nl.minimum, operand0=params.gate_clamp_upper_limit
            )
        if params.gate_clamp_lower_limit is not None:
            nisa.tensor_scalar(
                data=gate_sb_fp32, dst=gate_sb_fp32, op0=nl.maximum, operand0=params.gate_clamp_lower_limit
            )

        # ---------------- Gate Activation ----------------
        nisa.activation(
            dst=gate_sb_fp32[:, :],
            op=get_nl_act_fn_from_type(params.activation_fn),
            data=gate_sb_fp32,
            scale=1.0,
        )

        # ---------------- Up Projection Multi-Shard Communication ----------------
        # Receive up projection output from the other neuron core when LNC > 1
        if dims.num_shards > 1:
            nisa.sendrecv(
                src=up_sb_fp32,
                dst=gate_up_recv,
                send_to_rank=(1 - dims.shard_id),
                recv_from_rank=(1 - dims.shard_id),
                pipe_id=0,
            )
            nisa.tensor_tensor(dst=up_sb_fp32, data1=up_sb_fp32, data2=gate_up_recv, op=nl.add)

        #  ---------------- Optionally perform clamping on up projection results  ----------------
        if params.up_clamp_upper_limit is not None:
            nisa.tensor_scalar(data=up_sb_fp32, dst=up_sb_fp32, op0=nl.minimum, operand0=params.up_clamp_upper_limit)
        if params.up_clamp_lower_limit is not None:
            nisa.tensor_scalar(data=up_sb_fp32, dst=up_sb_fp32, op0=nl.maximum, operand0=params.up_clamp_lower_limit)

        # ---------------- Multiply Gate and Up Outputs ----------------
        nisa.tensor_tensor(dst=output, data1=gate_sb_fp32, data2=up_sb_fp32, op=nl.multiply)

    sbm.close_scope()  # Close SBUF scope

    return tiles
