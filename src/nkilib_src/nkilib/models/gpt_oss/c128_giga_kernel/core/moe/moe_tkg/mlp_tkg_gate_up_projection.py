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

"""Gate and Up projection sub-kernels for MLP TKG with column tiling and LHS/RHS swap modes."""

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type, resolve_fp8_e4m3_dtype
from ...utils.tiled_range import TiledRange
from .mlp_parameters import (
    MLPParameters,
    mlpp_has_gate_projection_bias,
    mlpp_has_up_projection_bias,
)
from .mlp_tkg_constants import (
    MLPTKGConstants,
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsGateUpTileCounts,
)
from .mlp_tkg_gate_up_projection_lhs_rhs_swap import (
    run_gate_up_projection_lhs_rhs_swap,
)
from .projection_utils import (
    _clamp_lower_upper_limit,
    adaptive_dge_mode,
    prepare_gate_up_bias_and_scale,
)

_DGE_MODE_UNKNOWN = 0  # Compiler decides best DMA mode internally
_DGE_MODE_NONE = 3  # Use STATIC DMA mode


def gate_up_projection(
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
    Performs a single Gate or Up projection shard on the H.

    All inputs are pre-sharded NkiTensor instances — callers handle LNC/shard slicing.

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

    Args:
        hidden (NkiTensor): [H0, T, H1_shard] — pre-sharded hidden activations
        weight (NkiTensor): [H_per_shard, I_shard] — pre-sharded weight matrix
        bias (NkiTensor): [1, I_shard] or None — pre-sharded bias
        dequant_scale (NkiTensor): [T, I_shard] or static scale — dequantization scale
        output_tile (NkiTensor): [T, I_shard] — pre-sharded output buffer in SBUF

    Returns:
        Output tensor with shape [T, I_shard]
    """

    # ---------- Configuration and Dimension Setup ----------
    H0 = dims.H0
    T = dims.T
    H = dims.H_per_shard
    I = weight.shape[1]
    H1_shard = H // H0
    num_allocated_w_tile = tiles.num_allocated_w_tile
    is_up_proj = "up" in op_name

    # Determine hidden layout: [H0, H1_shard, T] or [H0, T, H1_shard]
    hidden_h1_dim = dims.hidden_layout.get_h1_dim()

    # Sanity checks for sharding
    kernel_assert(
        I <= dims.max_I_shard_size,
        f"{op_name}_projection supports I <= {dims.max_I_shard_size}",
    )

    # For 'up' projection, offset weight index to avoid anti-dependencies with gate weights.
    # The kernel shares weight tiles as a ring buffer so up weights load after gate weights.
    weight_base_idx = div_ceil(H, tiles.HTile) % num_allocated_w_tile if is_up_proj else 0

    # ---------- Allocate PSUM buffers ----------
    result_psums = []
    for i_tile in TiledRange(I, dims._psum_fmax):
        psum_idx = (i_tile.index + tiles.up_psum_base_bank) % dims._psum_bmax if is_up_proj else i_tile.index
        result_psum = nl.ndarray(
            shape=(dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"{op_name}_{sbm.get_name_prefix()}_psum_{psum_idx}",
            buffer=nl.psum,
            address=None if sbm.is_auto_alloc() else (0, psum_idx * dims._psum_fmax * 4),
        )
        result_psums.append(result_psum)

    # ---------- Bias handling ----------
    # Only apply bias on one core to avoid double-counting (sharding along H)
    is_bias = bias is not None and dims.shard_id == 0
    if is_bias:
        # bias is already pre-sharded [1, I_shard] — broadcast across T dimension
        bias_hbm_view = bias.broadcast(dim=0, size=T)
        bias_tile_view = bias_tile.slice(dim=1, start=0, end=I)
        nisa.dma_copy(
            dst=bias_tile_view,
            src=bias_hbm_view,
            dge_mode=_DGE_MODE_NONE,
        )

    # ---------- Load dequant scale ----------
    if params.quant_params.is_quant_row():
        dequant_scale_view = dequant_scale.slice(dim=0, start=0, end=T).slice(dim=1, start=0, end=I)
        nisa.dma_copy(
            dst=dequant_tile.slice(dim=1, start=0, end=I),
            src=dequant_scale_view,
            dge_mode=_DGE_MODE_NONE,
        )

    # ---------- Matrix multiplication ----------
    used_columns = 0

    # Gate Up Projection
    for hidden_tiles in TiledRange(H, tiles.HTile):
        # Compute start offset
        h_offset = hidden_tiles.index * (tiles.HTile // H0)
        h1_tiles = hidden_tiles.size // H0

        # Load weight tile [HTile, I] → SBUF layout [H0, HTile/H0, I]
        weight_idx = (weight_base_idx + hidden_tiles.index) % num_allocated_w_tile
        weight_view = weight.reshape_dim(dim=0, shape=(H0, H1_shard)).slice(
            dim=1, start=h_offset, end=h_offset + h1_tiles
        )
        # Slice weight tile
        weight_sb_tile_slice = weight_tiles[weight_idx].slice(dim=1, start=0, end=h1_tiles).slice(dim=2, start=0, end=I)

        nisa.dma_copy(
            dst=weight_sb_tile_slice,
            src=weight_view,
            dge_mode=_DGE_MODE_NONE,
        )

        # Matmul
        for i_tile in TiledRange(I, dims._psum_fmax):
            for column_tile in TiledRange(h1_tiles, dims.column_tiling_factor):
                for column_idx in range(column_tile.size):
                    column_tile_offset = dims.column_tiling_factor * column_tile.index + column_idx
                    nisa.nc_matmul(
                        dst=result_psums[i_tile.index][
                            nl.ds(dims.column_tiling_dim * column_idx, T),
                            0 : i_tile.size,
                        ],
                        stationary=hidden.select(dim=hidden_h1_dim, index=h_offset + column_tile_offset),
                        moving=weight_sb_tile_slice.select(dim=1, index=column_tile_offset).slice(
                            dim=1, start=i_tile.start_offset, end=i_tile.end_offset
                        ),
                        tile_position=(0, dims.column_tiling_dim * column_idx),
                        tile_size=(H0, dims.column_tiling_dim),
                    )
            # Update used column numbers
            used_columns = max(used_columns, column_tile.size)

    # ---------- Accumulate PSUMs into output ----------
    for i_tile in TiledRange(I, dims._psum_fmax):
        # Create sliced output view for this I tile
        sliced_output = output_tile.slice(dim=1, start=i_tile.start_offset, end=i_tile.end_offset)
        # Copy PSUM to SBUF
        nisa.activation(
            dst=sliced_output,
            data=result_psums[i_tile.index][0:T, 0 : i_tile.size],
            op=nl.copy,
        )

        # Accumulate PSUMs to SBUF
        for factor_idx in range(1, used_columns):
            nisa.tensor_tensor(
                dst=sliced_output,
                data1=result_psums[i_tile.index][nl.ds(dims.column_tiling_dim * factor_idx, T), 0 : i_tile.size],
                data2=sliced_output,
                op=nl.add,
            )

    if params.quant_params.is_quant():
        dequant_tile_view = dequant_tile
        if params.quant_params.is_quant_row():
            dequant_tile_view = dequant_tile_view.slice(dim=1, start=0, end=I)

        interleave_copy(
            dst=output_tile,
            src=output_tile,
            scale=dequant_tile_view,
            bias=None,
        )

    # ---------- Apply bias separately from matmul pipeline ----------
    if is_bias:
        bias_tile_view = bias_tile.slice(dim=1, start=0, end=I)
        nisa.tensor_tensor(
            dst=output_tile,
            data1=output_tile,
            data2=bias_tile_view,
            op=nl.add,
        )


def run_gate_up_projection_non_lhs_rhs_swap(
    hidden: nl.NkiTensor,
    output: nl.NkiTensor,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
):
    """
    Runs the Gate/Up projection using the column tiling matmul path.

    This is the `if` branch of `process_gate_up_projection` when
    `params.use_tkg_gate_up_proj_column_tiling` is True.

    Returns:
        tiles, gate_sb_view, up_sb_view, gate_up_recv,
        use_fused_gate_up_sendrecv, gate_up_sb_fp32
    """
    gate_w, up_w = params.gate_proj_weights_tensor, params.up_proj_weights_tensor
    gate_b, up_b, gate_w_scale, up_w_scale = prepare_gate_up_bias_and_scale(params, dims)

    bias_tile = None
    gate_up_sb_fp32 = None

    use_fused_gate_up_sendrecv = dims.num_shards > 1 and not params.skip_gate_proj
    tile_shape = (dims.T, dims.I)

    if not params.skip_gate_proj:
        if use_fused_gate_up_sendrecv:
            gate_up_tile_shape = (dims.T, 2 * dims.I)
            gate_up_sb_fp32 = sbm.alloc_stack(
                gate_up_tile_shape,
                dtype=nl.float32,
                name="gate_up_sbuf_fp32",
                buffer=nl.sbuf,
                align=4,
            )
            gate_up_tv = gate_up_sb_fp32
            gate_sb_view = gate_up_tv.slice(dim=1, start=0, end=tile_shape[1])
            up_sb_view = gate_up_tv.slice(dim=1, start=tile_shape[1], end=2 * tile_shape[1])
        else:
            gate_sb_fp32 = sbm.alloc_stack(tile_shape, dtype=nl.float32, name="gate_sbuf_fp32", buffer=nl.sbuf, align=4)
            up_sb_fp32 = sbm.alloc_stack(tile_shape, dtype=nl.float32, name="up_sbuf_fp32", buffer=nl.sbuf, align=4)
            gate_sb_view = gate_sb_fp32
            up_sb_view = up_sb_fp32
    else:
        up_sb_fp32 = sbm.alloc_stack(tile_shape, dtype=nl.float32, name="up_sbuf_fp32", buffer=nl.sbuf, align=4)
        up_sb_view = up_sb_fp32
        gate_sb_view = None

    # ---------------- quantization ----------------
    I_shard_size = min(dims.I, dims.max_I_shard_size)
    gate_dequant_tile = up_dequant_tile = None
    if params.quant_params.is_quant_static():
        par_dim = dims.T
        gate_dequant_tile = sbm.alloc_stack((par_dim, 1), dtype=gate_w_scale.dtype, name="gate_w_scale_sb", align=4)
        up_dequant_tile = sbm.alloc_stack((par_dim, 1), dtype=up_w_scale.dtype, name="up_w_scale_sb", align=4)
        gate_w_scale_view = gate_w_scale.slice(dim=0, start=0, end=par_dim)
        up_w_scale_view = up_w_scale.slice(dim=0, start=0, end=par_dim)
        nisa.dma_copy(
            dst=gate_dequant_tile,
            src=gate_w_scale_view,
            dge_mode=adaptive_dge_mode(gate_w_scale_view),
        )
        nisa.dma_copy(dst=up_dequant_tile, src=up_w_scale_view, dge_mode=adaptive_dge_mode(up_w_scale_view))

    elif params.quant_params.is_quant_row():
        row_dequant_shape = (dims.T, I_shard_size)
        gate_dequant_tile = sbm.alloc_stack(
            row_dequant_shape, dtype=gate_w_scale.dtype, name="gate_w_scale_sb", align=4
        )
        up_dequant_tile = sbm.alloc_stack(row_dequant_shape, dtype=up_w_scale.dtype, name="up_w_scale_sb", align=4)

    # ---------------- bias ----------------
    if mlpp_has_gate_projection_bias(params) or mlpp_has_up_projection_bias(params):
        bias_tile = sbm.alloc_stack(
            (dims.T, I_shard_size),
            dtype=gate_b.dtype,
            name="gate_up_broadcasted_bias",
            buffer=nl.sbuf,
        )

    # ---------------- Allocate Receive Buffer for LNC > 1 ----------------
    gate_up_recv = None
    if dims.num_shards > 1:
        if use_fused_gate_up_sendrecv:
            send_recv_buffer_shape = gate_up_sb_fp32.shape
        else:
            send_recv_buffer_shape = up_sb_view.shape
        gate_up_recv = sbm.alloc_stack(
            send_recv_buffer_shape,
            dtype=nl.float32,
            buffer=nl.sbuf,
            name="gate_up_recv_buffer_fp32",
        )

    # ---------------- Allocate Weight Tiles ----------------
    tiles = MLPTKGConstants.calculate_gate_up_tiles(params, dims, sbm)

    _fp8_e4m3_tile_dtype = resolve_fp8_e4m3_dtype(params.dtype_mode)

    weight_tiles = []
    for w_tile_idx in range(tiles.num_allocated_w_tile):
        weight_tile = sbm.alloc_stack(
            (dims.H0, div_ceil(tiles.HTile, dims.H0), tiles.I_shard_size),
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

    # ---------------- Gate/Up Projection (column tiling) ----------------
    hidden_view = hidden
    for i_tile in TiledRange(dims.I, tiles.I_shard_size):
        I_start = i_tile.start_offset
        I_end = i_tile.end_offset

        up_w_view = up_w_h_sharded.slice(dim=1, start=I_start, end=I_end)
        up_b_view = up_b.slice(dim=1, start=I_start, end=I_end) if up_b is not None else None
        up_out_view = up_sb_view.slice(dim=1, start=I_start, end=I_end)

        if not params.skip_gate_proj:
            gate_w_view = gate_w_h_sharded.slice(dim=1, start=I_start, end=I_end)
            gate_b_view = gate_b.slice(dim=1, start=I_start, end=I_end) if gate_b is not None else None
            gate_out_view = gate_sb_view.slice(dim=1, start=I_start, end=I_end)
            gate_up_projection(
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

        gate_up_projection(
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

    return tiles, gate_sb_view, up_sb_view, gate_up_recv, use_fused_gate_up_sendrecv, gate_up_sb_fp32


def _post_process_gate_up(
    output: nl.NkiTensor,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
    gate_sb_view,
    up_sb_view,
    gate_up_recv,
    use_fused_gate_up_sendrecv: bool,
    gate_up_sb_fp32,
    tiles: MLPTKGConstantsGateUpTileCounts,
):
    """Post-processing: sendrecv, activation, multiply, transpose."""
    if params.skip_gate_proj:
        if dims.num_shards > 1:
            nisa.sendrecv(
                src=up_sb_view,
                dst=gate_up_recv,
                send_to_rank=(1 - dims.shard_id),
                recv_from_rank=(1 - dims.shard_id),
                pipe_id=0,
            )
            nisa.tensor_tensor(dst=up_sb_view, data1=up_sb_view, data2=gate_up_recv, op=nl.add)

        _clamp_lower_upper_limit(up_sb_view, params.up_clamp_lower_limit, params.up_clamp_upper_limit)

        nisa.activation(
            dst=up_sb_view if params.use_tkg_gate_up_proj_column_tiling else output,
            op=get_nl_act_fn_from_type(params.activation_fn),
            data=up_sb_view,
            scale=1.0,
        )
    else:
        if dims.num_shards > 1:
            if use_fused_gate_up_sendrecv:
                nisa.sendrecv(
                    src=gate_up_sb_fp32,
                    dst=gate_up_recv,
                    send_to_rank=(1 - dims.shard_id),
                    recv_from_rank=(1 - dims.shard_id),
                    pipe_id=0,
                )
                nisa.tensor_tensor(dst=gate_up_sb_fp32, data1=gate_up_sb_fp32, data2=gate_up_recv, op=nl.add)
            else:
                nisa.sendrecv(
                    src=gate_sb_view,
                    dst=gate_up_recv,
                    send_to_rank=(1 - dims.shard_id),
                    recv_from_rank=(1 - dims.shard_id),
                    pipe_id=0,
                )
                nisa.tensor_tensor(dst=gate_sb_view, data1=gate_sb_view, data2=gate_up_recv, op=nl.add)

                nisa.sendrecv(
                    src=up_sb_view,
                    dst=gate_up_recv,
                    send_to_rank=(1 - dims.shard_id),
                    recv_from_rank=(1 - dims.shard_id),
                    pipe_id=0,
                )
                nisa.tensor_tensor(dst=up_sb_view, data1=up_sb_view, data2=gate_up_recv, op=nl.add)

        _clamp_lower_upper_limit(gate_sb_view, params.gate_clamp_lower_limit, params.gate_clamp_upper_limit)

        nisa.activation(
            dst=gate_sb_view,
            op=get_nl_act_fn_from_type(params.activation_fn),
            data=gate_sb_view,
            scale=1.0,
        )

        _clamp_lower_upper_limit(up_sb_view, params.up_clamp_lower_limit, params.up_clamp_upper_limit)

        mul_dst = up_sb_view if params.use_tkg_gate_up_proj_column_tiling else output
        nisa.tensor_tensor(dst=mul_dst, data1=gate_sb_view, data2=up_sb_view, op=nl.multiply)

    # ---------- Transpose hidden if column tiling is enabled ----------
    if params.use_tkg_gate_up_proj_column_tiling:
        for i_tile in TiledRange(dims.I, dims.I0):
            psum_idx = i_tile.index % dims._psum_bmax
            tp_psum = nl.ndarray(
                (i_tile.size, dims.T),
                dtype=up_sb_view.dtype,
                buffer=nl.psum,
                name=f"{sbm.get_name_prefix()}transpose_psum_{i_tile.index}",
                address=None if sbm.is_auto_alloc() else (0, psum_idx * dims._psum_fmax * 4),
            )
            nisa.nc_transpose(
                dst=tp_psum,
                data=up_sb_view.slice(dim=0, start=0, end=dims.T).slice(
                    dim=1, start=i_tile.index * dims.I0, end=i_tile.index * dims.I0 + i_tile.size
                ),
            )
            nisa.tensor_copy(
                dst=output.slice(dim=0, start=0, end=i_tile.size)
                .slice(dim=1, start=i_tile.index, end=i_tile.index + 1)
                .slice(dim=2, start=0, end=dims.T),
                src=tp_psum,
            )

    return tiles


def process_gate_up_projection(
    hidden: nl.NkiTensor,
    output: nl.NkiTensor,
    params: MLPParameters,
    dims: MLPTKGConstantsDimensionSizes,
    sbm: SbufManager,
    T_offset: int = 0,
    share_memory_scope: bool = False,
    use_dge: bool = False,
    prealloc_gate_up_buffers=None,
):
    """
    Performs the Gate/Up projection for MLP (T = BxS).

    Dispatches to column tiling or LHS/RHS swap path based on
    `params.use_tkg_gate_up_proj_column_tiling`, then runs shared post-processing.
    """
    if params.use_tkg_gate_up_proj_column_tiling:
        kernel_assert(not share_memory_scope, "share_memory_scope is not supported with column tiling")
        tiles, gate_sb_view, up_sb_view, gate_up_recv, use_fused_gate_up_sendrecv, gate_up_sb_fp32 = (
            run_gate_up_projection_non_lhs_rhs_swap(
                hidden=hidden,
                output=output,
                params=params,
                dims=dims,
                sbm=sbm,
            )
        )
    else:
        tiles, gate_sb_view, up_sb_view, gate_up_recv = run_gate_up_projection_lhs_rhs_swap(
            hidden=hidden,
            output=output,
            params=params,
            dims=dims,
            sbm=sbm,
            T_offset=T_offset,
            share_memory_scope=share_memory_scope,
            use_dge=use_dge,
            prealloc_gate_up_buffers=prealloc_gate_up_buffers,
        )
        use_fused_gate_up_sendrecv = False
        gate_up_sb_fp32 = None

    return _post_process_gate_up(
        output=output,
        params=params,
        dims=dims,
        sbm=sbm,
        gate_sb_view=gate_sb_view,
        up_sb_view=up_sb_view,
        gate_up_recv=gate_up_recv,
        use_fused_gate_up_sendrecv=use_fused_gate_up_sendrecv,
        gate_up_sb_fp32=gate_up_sb_fp32,
        tiles=tiles,
    )
