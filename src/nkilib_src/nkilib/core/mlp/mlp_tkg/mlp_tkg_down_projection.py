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

"""Down projection sub-kernels for MLP TKG with column tiling and LHS/RHS swap modes."""

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_helpers import div_ceil, resolve_fp8_e4m3_dtype
from ...utils.tiled_range import TiledRange
from ..mlp_parameters import MLPParameters, mlpp_has_down_projection_bias
from .mlp_tkg_constants import (
    MLPTKGConstants,
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsDownTileCounts,
    MLPTKGConstantsGateUpTileCounts,
)
from .mlp_tkg_down_projection_lhs_rhs_swap import down_projection_lhs_rhs_swap
from .mlp_tkg_utils import adaptive_dge_mode, prepare_down_bias_and_scale

_DGE_MODE_UNKNOWN = 0  # Compiler decides best DMA mode internally
_DGE_MODE_NONE = 3  # Use STATIC DMA mode


def down_projection(
    hidden: nl.NkiTensor,
    weight: nl.NkiTensor,
    output_tile: nl.NkiTensor,
    weight_tiles: list[nl.NkiTensor],
    bias_tile: nl.NkiTensor,
    dequant_tile: nl.NkiTensor,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsDownTileCounts,
    params: MLPParameters,
    sbm: SbufManager,
):
    """
    Performs a single Down projection shard on the H.

    All weight/bias inputs are pre-sharded NkiTensor instances — callers handle LNC/shard slicing.

    Computes: Hidden[I, T] @ Weight[I, H_per_shard] + Optional(bias_tile) → [T, H_per_shard]
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

    Args:
        hidden (nl.NkiTensor): [I0, I1, T] — hidden activations in SBUF
        weight (NkiTensor): [I, H_per_shard] — pre-sharded weight matrix
        output_tile (nl.NkiTensor): [T, H_per_shard] — output buffer in SBUF

    Returns:
        Output tensor with shape [T, H_per_shard]
    """

    # ---------- Configuration and Dimension Setup ----------
    I0, I1, T = hidden.shape
    I = weight.shape[0]
    H = dims.H_per_shard
    num_Itiles = div_ceil(I, I0)

    weight_base_idx = tiles.weight_base_idx

    # ---------- Compute matmul  ----------
    num_required_psums_per_HTile = div_ceil(tiles.HTile, dims._psum_fmax)
    num_required_psum_after_column_tiling = div_ceil(num_required_psums_per_HTile, dims.column_tiling_factor)

    # Total PSUM banks divided by PSUMs needed per HTile
    num_available_psum_group = dims._psum_bmax // num_required_psum_after_column_tiling

    for hidden_tiles in TiledRange(H, tiles.HTile):
        h_offset = hidden_tiles.start_offset

        # Allocate PSUM
        psum_offset = (hidden_tiles.index % num_available_psum_group) * num_required_psum_after_column_tiling
        result_psums = []
        for psum_idx in range(num_required_psum_after_column_tiling):
            result_psum = nl.ndarray(
                shape=(dims._pmax, dims._psum_fmax),
                dtype=nl.float32,
                name=f"down_psum_{sbm.get_name_prefix()}_{hidden_tiles.index}_{psum_offset + psum_idx}",
                buffer=nl.psum,
                address=None if sbm.is_auto_alloc() else (0, (psum_offset + psum_idx) * dims._psum_fmax * 4),
            )
            result_psums.append(result_psum)

        for i_tile in TiledRange(I, I0):
            weight_idx = (weight_base_idx + hidden_tiles.index * num_Itiles + i_tile.index) % tiles.num_allocated_w_tile

            hidden_sb_tile_slice = hidden.slice(dim=0, start=0, end=i_tile.size).slice(
                dim=1, start=i_tile.index, end=i_tile.index + 1
            )
            weight_sb_tile_slice = (
                weight_tiles[weight_idx]
                .slice(dim=0, start=0, end=i_tile.size)
                .slice(dim=1, start=0, end=hidden_tiles.size)
            )

            # Load weight [I0, HTile] from HBM into SBUF tile
            weight_view = weight.slice(dim=0, start=i_tile.start_offset, end=i_tile.end_offset).slice(
                dim=1, start=h_offset, end=h_offset + hidden_tiles.size
            )
            nisa.dma_copy(
                dst=weight_sb_tile_slice,
                src=weight_view,
                dge_mode=_DGE_MODE_NONE,
            )

            # Matmul with column tiling: distribute tiles across columns and PSUM banks
            for compute_tile in TiledRange(hidden_tiles.size, dims._psum_fmax):
                column_tile_index = compute_tile.index % dims.column_tiling_factor
                psum_bank_index = compute_tile.index // dims.column_tiling_factor
                nisa.nc_matmul(
                    dst=result_psums[psum_bank_index][
                        nl.ds(dims.column_tiling_dim * column_tile_index, T),
                        0 : compute_tile.size,
                    ],
                    stationary=hidden_sb_tile_slice,
                    moving=weight_sb_tile_slice.slice(
                        dim=1, start=compute_tile.start_offset, end=compute_tile.end_offset
                    ),
                    tile_position=(0, dims.column_tiling_dim * column_tile_index),
                    tile_size=(I0, dims.column_tiling_dim),
                )

        # ---------- Copy PSUM output to SBUF, optionally applying dequant scale ----------
        for compute_tile in TiledRange(hidden_tiles.size, dims._psum_fmax):
            column_tile_index = compute_tile.index % dims.column_tiling_factor
            psum_bank_index = compute_tile.index // dims.column_tiling_factor
            dst_offset = h_offset + compute_tile.index * dims._psum_fmax
            interleave_copy(
                index=column_tile_index,
                dst=output_tile.slice(dim=1, start=dst_offset, end=dst_offset + compute_tile.size),
                src=result_psums[psum_bank_index][
                    nl.ds(dims.column_tiling_dim * column_tile_index, T),
                    0 : compute_tile.size,
                ],
                scale=dequant_tile.slice(dim=1, start=dst_offset, end=dst_offset + compute_tile.size)
                if params.quant_params.is_quant_row()
                else dequant_tile,
                bias=None,
            )

    # ---------- Apply bias ----------
    is_bias = bias_tile is not None
    if is_bias:
        nisa.tensor_tensor(
            dst=output_tile,
            data1=output_tile,
            data2=bias_tile,
            op=nl.add,
        )


def process_down_projection(
    hidden: nl.NkiTensor,
    output: nl.NkiTensor,
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

    Note:
    ---------
    Caller will have the flexibility to manage sbm:sbufManager's scope and interleave degree.

    """
    down_w = params.down_proj_weights_tensor
    down_b, down_w_scale = prepare_down_bias_and_scale(params, dims)
    down_w_view = down_w.slice(dim=1, start=dims.H1_offset * dims.H0, end=dims.H1_offset * dims.H0 + dims.H_per_shard)

    # ---------------- Allocate and Load Bias Tile ----------------
    bias_tile = None
    if mlpp_has_down_projection_bias(params):
        bias_tile = sbm.alloc_stack(
            down_b.shape,
            dtype=down_b.dtype,
            name=f"down_bias",
            buffer=nl.sbuf,
        )
        nisa.dma_copy(
            dst=bias_tile,
            src=down_b,
            dge_mode=adaptive_dge_mode(down_b),
        )

    dequant_tile = None
    # ---------------- Quantization Scale ----------------
    if params.quant_params.is_quant():
        dequant_tile = sbm.alloc_stack(
            down_w_scale.shape,
            dtype=down_w_scale.dtype,
            name=f"down_w_scale_sb",
            align=4,
        )
        nisa.dma_copy(
            dst=dequant_tile,
            src=down_w_scale,
            dge_mode=adaptive_dge_mode(down_w_scale),
        )

    # ---------------- Allocate Weight Tiles ----------------
    tiles = MLPTKGConstants.calculate_down_tiles(params, dims, gate_tile_info, sbm)

    _fp8_e4m3_tile_dtype = resolve_fp8_e4m3_dtype(params.dtype_mode)

    weight_tiles = []
    for w_tile_idx in range(tiles.num_allocated_w_tile):
        weight_tile = sbm.alloc_stack(
            (dims.I0, tiles.HTile),
            name=f"down_w_tile_{w_tile_idx}",
            dtype=_fp8_e4m3_tile_dtype if str(down_w.dtype) == "float8e4" else down_w.dtype,
            buffer=nl.sbuf,
        )
        weight_tiles.append(weight_tile)

    # ---------------- Down Projection ----------------
    if params.use_tkg_down_proj_column_tiling:
        down_projection(
            hidden=hidden,
            weight=down_w_view,
            output_tile=output,
            weight_tiles=weight_tiles,
            bias_tile=bias_tile,
            dequant_tile=dequant_tile,
            dims=dims,
            tiles=tiles,
            params=params,
            sbm=sbm,
        )

    else:
        down_projection_lhs_rhs_swap(
            hidden=hidden,
            weight=down_w_view,
            output_tile=output,
            weight_tiles=weight_tiles,
            bias_tile=bias_tile,
            dequant_tile=dequant_tile,
            dims=dims,
            tiles=tiles,
            params=params,
            sbm=sbm,
        )

    return output, tiles
