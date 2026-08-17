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

"""Gate and Up projection sub-kernel for MLP TKG with LHS/RHS swap mode."""

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, resolve_fp8_e4m3_dtype
from ...utils.tiled_range import TiledRange
from ..mlp_parameters import MLPParameters
from .mlp_tkg_constants import (
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsGateUpTileCounts,
)
from .mlp_tkg_utils import _load_transposed_tile


def gate_up_projection_lhs_rhs_swap(
    hidden: nl.NkiTensor,
    weight: nl.NkiTensor,
    bias: nl.NkiTensor,
    dequant_scale: nl.NkiTensor,
    output_tile: nl.NkiTensor,
    weight_tiles: list[nl.NkiTensor],
    bias_tile: nl.NkiTensor,
    dequant_tile: nl.NkiTensor,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsGateUpTileCounts,
    params: MLPParameters,
    op_name: str,
    sbm: SbufManager,
):
    """
    Performs a single Gate or Up projection shard on the H using regular matmult with operands swapped.

    All inputs are pre-sharded NkiTensor instances — callers handle LNC/shard slicing.
    Hidden is pre-sliced on T dimension by the caller (no T_offset parameter).

    Computes: Weight[H, I] @ Hidden[H, T] + Optional(Bias[1, I]) → [T, I]
    - Hidden is the moving tensor, Weight is the stationary tensor.

    Tiled computation:
        H/128 * [ I/128 * (Weight[128, 128] @ Hidden[128, T]) ]

    Args:
        hidden (NkiTensor): [H0, T, H1_shard] — pre-sharded, pre-sliced on T
        weight (NkiTensor): [H_per_shard, I_shard] — pre-sharded weight matrix
        bias (NkiTensor): [I_shard] or [1, I_shard] or None — pre-sharded bias
        dequant_scale (NkiTensor): dequantization scale or None
        output_tile (NkiTensor): [I0, num_I_tiles_shard, T] — pre-sharded output buffer

    Returns:
        Output tensor with shape [128, I/128, T]
    """

    # ---------- Configuration and Dimension Setup ----------
    H0 = dims.H0
    T = dims.T
    H = dims.H_per_shard
    I = weight.shape[1]
    I0 = dims.I0
    num_allocated_w_tile = tiles.num_allocated_w_tile
    is_up_proj = "up" in op_name

    # Determine hidden layout: [H0, H1_shard, T] or [H0, T, H1_shard]
    hidden_h1_dim = dims.hidden_layout.get_h1_dim()

    # Sanity checks for sharding
    kernel_assert(
        I <= dims.max_I_shard_size,
        f"{op_name}_projection only supports I <= {dims.max_I_shard_size}",
    )

    # ---------- Bias handling ----------
    # Only apply bias on one core to avoid double-counting (sharding along H)
    is_bias = bias is not None and dims.shard_id == 0

    if is_bias:
        _load_transposed_tile(bias, bias_tile, I, I0, f"{op_name}_b", sbm)

    # ---------- Load dequant scale ----------
    if params.quant_params.is_quant_row():
        dequant_scale_1d = dequant_scale.select(dim=0, index=0)
        _load_transposed_tile(dequant_scale_1d, dequant_tile, I, I0, f"{op_name}_dq", sbm)
    # For 'up' projection, offset weight index to avoid anti-dependencies with gate weights.
    # The kernel shares weight tiles as a ring buffer so up weights load after gate weights.
    weight_base_idx = div_ceil(H, tiles.HTile) % num_allocated_w_tile if "up" in op_name else 0

    # Allocate PSUM buffers to store output
    result_psums = []
    for i_tile in TiledRange(I, I0):
        psum_idx = (i_tile.index + tiles.up_psum_base_bank) % dims._psum_bmax if is_up_proj else i_tile.index
        result_psum = nl.ndarray(
            shape=(dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"{op_name}_{sbm.get_name_prefix()}_psum_{psum_idx}",
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, psum_idx * dims._psum_fmax * 4),
        )
        result_psums.append(result_psum)

    # ---------- Matrix multiplication ----------
    # Gate Up Projection
    for hidden_tiles in TiledRange(H, tiles.HTile):
        # Compute start offset
        h_start_offset = hidden_tiles.index * (tiles.HTile // H0)

        # Load weight tile [HTile, I] → SBUF layout [H0, HTile/H0, I]
        h1_size = hidden_tiles.size // H0
        weight_idx = (weight_base_idx + hidden_tiles.index) % num_allocated_w_tile
        weight_view = weight.reshape_dim(dim=0, shape=(H0, dims.H1_shard)).slice(
            dim=1, start=h_start_offset, end=h_start_offset + h1_size
        )
        # Slice weight tile for this hidden tile
        weight_sb_tile_slice = weight_tiles[weight_idx].slice(dim=1, start=0, end=h1_size).slice(dim=2, start=0, end=I)

        nisa.dma_copy(
            dst=weight_sb_tile_slice,
            src=weight_view,
            dge_mode=nisa.dge_mode.hwdge,
        )

        # Matmult
        for h1_tile in TiledRange(hidden_tiles.size, H0):
            for i_tile in TiledRange(I, I0):
                nisa.nc_matmul(
                    result_psums[i_tile.index][0 : i_tile.size, 0:T],
                    weight_sb_tile_slice.select(dim=1, index=h1_tile.index).slice(
                        dim=1, start=i_tile.start_offset, end=i_tile.end_offset
                    ),
                    hidden.select(dim=hidden_h1_dim, index=h_start_offset + h1_tile.index),
                )

    # ---------- Accumulate partial PSUMs to output ----------
    for i_tile in TiledRange(I, I0):
        # Set tile view for dequant tile
        dequant_tile_view = None
        if params.quant_params.is_quant():
            dequant_tile_view = dequant_tile.slice(dim=0, start=0, end=i_tile.size)
            if params.quant_params.is_quant_row():
                dequant_tile_view = dequant_tile_view.slice(dim=1, start=i_tile.index, end=i_tile.index + 1).broadcast(
                    dim=1, size=T
                )

        # Create output tile view for this I tile — output_tile is pre-sharded
        output_tile_view = output_tile.slice(dim=0, start=0, end=i_tile.size).select(dim=1, index=i_tile.index)

        # PSUM to SBUF copy while applying dequant tensor optionally
        interleave_copy(
            index=i_tile.index,
            dst=output_tile_view,
            src=result_psums[i_tile.index][0 : i_tile.size, 0:T],
            scale=dequant_tile_view,
            bias=None,
        )

    # ---------- Apply bias ----------
    if is_bias:
        num_I_tiles = div_ceil(I, I0)
        bias_tile_view = bias_tile.slice(dim=1, start=0, end=num_I_tiles).expand_dim(dim=2).broadcast(dim=2, size=T)
        nisa.tensor_tensor(
            dst=output_tile,
            data1=output_tile,
            data2=bias_tile_view,
            op=nl.add,
        )


from ..mlp_parameters import (
    mlpp_has_gate_projection_bias,
    mlpp_has_up_projection_bias,
)
from .mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_utils import (
    adaptive_dge_mode,
    prepare_gate_up_bias_and_scale,
)


def run_gate_up_projection_lhs_rhs_swap(
    hidden: nl.NkiTensor,
    output: nl.NkiTensor,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
    T_offset: int,
    share_memory_scope: bool,
):
    """
    Runs the Gate/Up projection using the LHS/RHS swap matmul path.

    This is the `else` branch of `process_gate_up_projection` when
    `params.use_tkg_gate_up_proj_column_tiling` is False.

    Returns:
        tiles, gate_sb_view, up_sb_view, gate_up_recv
    """
    gate_w, up_w = params.gate_proj_weights_tensor, params.up_proj_weights_tensor
    gate_b, up_b, gate_w_scale, up_w_scale = prepare_gate_up_bias_and_scale(params, dims)

    num_I_tiles = div_ceil(dims.I, dims.I0)
    bias_tile = None
    bias_size = 0
    tile_shape = (dims.I0, num_I_tiles, dims.T)

    if not params.skip_gate_proj:
        gate_sb_view = sbm.alloc_stack(
            tile_shape,
            dtype=nl.float32,
            name="gate_sbuf_fp32",
            buffer=nl.sbuf,
            align=4,
        )
        up_sb_view = sbm.alloc_stack(
            tile_shape,
            dtype=nl.float32,
            name="up_sbuf_fp32",
            buffer=nl.sbuf,
            align=4,
        )
    else:
        up_sb_view = sbm.alloc_stack(
            tile_shape,
            dtype=nl.float32,
            name="up_sbuf_fp32",
            buffer=nl.sbuf,
            align=4,
        )
        gate_sb_view = None

    # ---------------- quantization ----------------
    I_shard_size = min(dims.max_I_shard_size, dims.I)
    gate_dequant_tile = up_dequant_tile = None
    if params.quant_params.is_quant_static():
        par_dim = dims.I0
        gate_dequant_tile = sbm.alloc_stack(
            (par_dim, 1),
            dtype=gate_w_scale.dtype,
            name="gate_w_scale_sb",
            align=4,
        )
        up_dequant_tile = sbm.alloc_stack(
            (par_dim, 1),
            dtype=up_w_scale.dtype,
            name="up_w_scale_sb",
            align=4,
        )
        gate_w_scale_view = gate_w_scale.slice(dim=0, start=0, end=par_dim)
        up_w_scale_view = up_w_scale.slice(dim=0, start=0, end=par_dim)
        nisa.dma_copy(
            dst=gate_dequant_tile,
            src=gate_w_scale_view,
            dge_mode=adaptive_dge_mode(gate_w_scale_view),
        )
        nisa.dma_copy(dst=up_dequant_tile, src=up_w_scale_view, dge_mode=adaptive_dge_mode(up_w_scale_view))

    elif params.quant_params.is_quant_row():
        row_dequant_shape = (dims.I0, div_ceil(I_shard_size, dims.I0))
        gate_dequant_tile = sbm.alloc_stack(
            row_dequant_shape,
            dtype=gate_w_scale.dtype,
            name="gate_w_scale_sb",
            align=32,
        )
        up_dequant_tile = sbm.alloc_stack(
            row_dequant_shape,
            dtype=up_w_scale.dtype,
            name="up_w_scale_sb",
            align=32,
        )

    # ---------------- bias ----------------
    if mlpp_has_gate_projection_bias(params) or mlpp_has_up_projection_bias(params):
        bias_tile = sbm.alloc_stack(
            (dims.I0, div_ceil(I_shard_size, dims.I0)),
            dtype=nl.float32 if gate_b.is_indirect() else gate_b.dtype,
            name="gate_up_bias",
            align=32,
        )

    # ---------------- Allocate Receive Buffer for LNC > 1 ----------------
    gate_up_recv = None
    if dims.num_shards > 1:
        gate_up_recv = sbm.alloc_stack(
            up_sb_view.shape,
            dtype=nl.float32,
            buffer=nl.sbuf,
            name="gate_up_recv_buffer_fp32",
        )

    # ---------------- Allocate Weight Tiles ----------------
    tiles = MLPTKGConstants.calculate_gate_up_tiles(params, dims, sbm, share_memory_scope)
    use_old_sharding_shape = sbm.is_auto_alloc() or share_memory_scope
    weight_tiles = []
    HTile_h1 = div_ceil(tiles.HTile, dims.H0)
    weight_shape = (dims.H0, HTile_h1, dims.I) if use_old_sharding_shape else (dims.H0, HTile_h1, tiles.I_shard_size)
    _fp8_e4m3_tile_dtype = resolve_fp8_e4m3_dtype(params.dtype_mode)
    for w_tile_idx in range(tiles.num_allocated_w_tile):
        weight_tile = sbm.alloc_stack(
            weight_shape,
            name=f"gate_up_w_tile_{w_tile_idx}",
            dtype=_fp8_e4m3_tile_dtype if str(up_w.dtype) == "float8e4" else up_w.dtype,
        )
        weight_tiles.append(weight_tile)

    # ---------------- Pre-shard weights on H dimension ----------------
    h_offset = dims.H1_offset * dims.H0
    up_w_h_sharded = up_w.slice(dim=0, start=h_offset, end=h_offset + dims.H_per_shard)
    gate_w_h_sharded = (
        gate_w.slice(dim=0, start=h_offset, end=h_offset + dims.H_per_shard) if not params.skip_gate_proj else None
    )
    is_row_quant = params.quant_params.is_quant_row()

    # ---------------- Gate/Up Projection (lhs_rhs_swap) ----------------
    t_dim = dims.hidden_layout.get_t_dim()
    hidden_view = hidden.slice(dim=t_dim, start=T_offset, end=T_offset + dims.T)
    I_tiling_size = dims.max_I_shard_size if use_old_sharding_shape else tiles.I_shard_size
    for i_tile in TiledRange(dims.I, I_tiling_size):
        I_start = i_tile.start_offset
        I_end = i_tile.end_offset
        num_total_128_I_tiles = div_ceil(i_tile.size, dims.I0)

        up_w_view = up_w_h_sharded.slice(dim=1, start=I_start, end=I_end)
        up_b_view = up_b.slice(dim=0, start=I_start, end=I_end) if up_b is not None else None
        i1_start = I_start // dims.I0
        i1_end = i1_start + num_total_128_I_tiles
        up_out_view = up_sb_view.slice(dim=1, start=i1_start, end=i1_end)

        if not params.skip_gate_proj:
            gate_w_view = gate_w_h_sharded.slice(dim=1, start=I_start, end=I_end)
            gate_b_view = gate_b.slice(dim=0, start=I_start, end=I_end) if gate_b is not None else None
            gate_out_view = gate_sb_view.slice(dim=1, start=i1_start, end=i1_end)
            gate_up_projection_lhs_rhs_swap(
                hidden=hidden_view,
                weight=gate_w_view,
                bias=gate_b_view,
                dequant_scale=gate_w_scale.slice(dim=1, start=I_start, end=I_end) if is_row_quant else gate_w_scale,
                output_tile=gate_out_view,
                weight_tiles=weight_tiles,
                bias_tile=bias_tile,
                dequant_tile=gate_dequant_tile,
                dims=dims,
                tiles=tiles,
                params=params,
                op_name=f"gate_i{i_tile.index}",
                sbm=sbm,
            )

        gate_up_projection_lhs_rhs_swap(
            hidden=hidden_view,
            weight=up_w_view,
            bias=up_b_view,
            dequant_scale=up_w_scale.slice(dim=1, start=I_start, end=I_end) if is_row_quant else up_w_scale,
            output_tile=up_out_view,
            weight_tiles=weight_tiles,
            bias_tile=bias_tile,
            dequant_tile=up_dequant_tile,
            dims=dims,
            tiles=tiles,
            params=params,
            op_name=f"up_i{i_tile.index}",
            sbm=sbm,
        )

    return tiles, gate_sb_view, up_sb_view, gate_up_recv
