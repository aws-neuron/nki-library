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

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager, sizeinbytes
from ...utils.interleave_copy import interleave_copy
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, resolve_fp8_e4m3_dtype
from ...utils.tiled_range import TiledRange
from .mlp_parameters import MLPParameters
from .mlp_tkg_constants import (
    MLPTKGConstantsDimensionSizes,
    MLPTKGConstantsGateUpTileCounts,
)
from .projection_utils import _load_transposed_tile

# Fused gate/up tile size per partition; covers a decode-size H shard in a few DMAs (not tuned)
_FUSED_GATE_UP_TILE_BYTES = 8 * 1024


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
    use_dge: bool = False,
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
        _load_transposed_tile(bias, bias_tile, I, I0, f"{op_name}_b", sbm, use_dge=use_dge)

    # ---------- Load dequant scale ----------
    if params.quant_params.is_quant_row():
        dequant_scale_1d = dequant_scale.select(dim=0, index=0)
        _load_transposed_tile(dequant_scale_1d, dequant_tile, I, I0, f"{op_name}_dq", sbm, use_dge=use_dge)
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

    # ---------- Accumulate partial PSUMs to output, then apply bias ----------
    _copy_psums_to_output(
        psums=result_psums,
        output_tile=output_tile,
        dequant_tile=dequant_tile,
        bias_tile=bias_tile,
        is_bias=is_bias,
        I=I,
        dims=dims,
        params=params,
    )


def _copy_psums_to_output(
    psums: list[nl.NkiTensor],
    output_tile: nl.NkiTensor,
    dequant_tile: nl.NkiTensor,
    bias_tile: nl.NkiTensor,
    is_bias: bool,
    I: int,
    dims: MLPTKGConstantsDimensionSizes,
    params: MLPParameters,
):
    """Copy one projection's PSUM tiles to output_tile, applying the optional dequant scale, then add the bias."""
    T = dims.T
    I0 = dims.I0
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
            src=psums[i_tile.index][0 : i_tile.size, 0:T],
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


def _gate_up_projection_fused_lhs_rhs_swap(
    hidden: nl.NkiTensor,
    weight: nl.NkiTensor,
    gate_bias: nl.NkiTensor,
    up_bias: nl.NkiTensor,
    gate_dequant_tile: nl.NkiTensor,
    up_dequant_tile: nl.NkiTensor,
    gate_output_tile: nl.NkiTensor,
    up_output_tile: nl.NkiTensor,
    weight_tiles: list[nl.NkiTensor],
    bias_tile: nl.NkiTensor,
    dims: MLPTKGConstantsDimensionSizes,
    tiles: MLPTKGConstantsGateUpTileCounts,
    params: MLPParameters,
    sbm: SbufManager,
    use_dge: bool = False,
):
    """Gate and up projections of one H shard from its [H_per_shard, 2 * I] weight view, one DMA per H tile."""
    H0 = dims.H0
    T = dims.T
    H = dims.H_per_shard
    I = weight.shape[1] // 2
    I0 = dims.I0
    hidden_h1_dim = dims.hidden_layout.get_h1_dim()
    kernel_assert(2 * div_ceil(I, I0) <= dims._psum_bmax, f"Fused gate/up needs 2 * ceil({I} / {I0}) PSUM banks")
    is_gate_bias = gate_bias is not None and dims.shard_id == 0
    is_up_bias = up_bias is not None and dims.shard_id == 0

    # Gate bias loads ahead of the weight stream; up bias reuses bias_tile once gate is done
    if is_gate_bias:
        _load_transposed_tile(gate_bias, bias_tile, I, I0, "gate_fused_b", sbm, use_dge=use_dge)

    if params.quant_params.is_quant_row():
        gate_scale = params.quant_params.gate_w_scale.select(dim=0, index=0)
        up_scale = params.quant_params.up_w_scale.select(dim=0, index=0)
        _load_transposed_tile(gate_scale, gate_dequant_tile, I, I0, "gate_fused_dq", sbm, use_dge=use_dge)
        _load_transposed_tile(up_scale, up_dequant_tile, I, I0, "up_fused_dq", sbm, use_dge=use_dge)

    # Gate and up accumulators stay live together
    gate_psums = []
    up_psums = []
    for i_tile in TiledRange(I, I0):
        up_psum_idx = (i_tile.index + tiles.up_psum_base_bank) % dims._psum_bmax
        gate_psum = nl.ndarray(
            shape=(dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"gate_fused_{sbm.get_name_prefix()}_psum_{i_tile.index}",
            buffer=nl.psum,
        )
        gate_psums.append(gate_psum)
        up_psum = nl.ndarray(
            shape=(dims._pmax, dims._psum_fmax),
            dtype=nl.float32,
            name=f"up_fused_{sbm.get_name_prefix()}_psum_{up_psum_idx}",
            buffer=nl.psum,
        )
        up_psums.append(up_psum)

    # Each row holds gate then up, so one H tile of both is one contiguous run per partition
    weight_h = weight.reshape_dim(dim=0, shape=(H0, dims.H1_shard))
    for hidden_tiles in TiledRange(H, tiles.HTile):
        h_start_offset = hidden_tiles.index * (tiles.HTile // H0)
        h1_size = hidden_tiles.size // H0
        weight_idx = hidden_tiles.index % len(weight_tiles)
        weight_sb = weight_tiles[weight_idx].slice(dim=1, start=0, end=h1_size)
        nisa.dma_copy(
            dst=weight_sb,
            src=weight_h.slice(dim=1, start=h_start_offset, end=h_start_offset + h1_size),
            dge_mode=nisa.dge_mode.hwdge,
        )
        for h1_tile in TiledRange(hidden_tiles.size, H0):
            hidden_tile = hidden.select(dim=hidden_h1_dim, index=h_start_offset + h1_tile.index)
            weight_row = weight_sb.select(dim=1, index=h1_tile.index)
            for i_tile in TiledRange(I, I0):
                nisa.nc_matmul(
                    gate_psums[i_tile.index][0 : i_tile.size, 0:T],
                    weight_row.slice(dim=1, start=i_tile.start_offset, end=i_tile.end_offset),
                    hidden_tile,
                )
                nisa.nc_matmul(
                    up_psums[i_tile.index][0 : i_tile.size, 0:T],
                    weight_row.slice(dim=1, start=I + i_tile.start_offset, end=I + i_tile.end_offset),
                    hidden_tile,
                )

    _copy_psums_to_output(
        psums=gate_psums,
        output_tile=gate_output_tile,
        dequant_tile=gate_dequant_tile,
        bias_tile=bias_tile,
        is_bias=is_gate_bias,
        I=I,
        dims=dims,
        params=params,
    )
    if is_up_bias:
        _load_transposed_tile(up_bias, bias_tile, I, I0, "up_fused_b", sbm, use_dge=use_dge)
    _copy_psums_to_output(
        psums=up_psums,
        output_tile=up_output_tile,
        dequant_tile=up_dequant_tile,
        bias_tile=bias_tile,
        is_bias=is_up_bias,
        I=I,
        dims=dims,
        params=params,
    )


from .mlp_parameters import (
    mlpp_has_gate_projection_bias,
    mlpp_has_up_projection_bias,
)
from .mlp_tkg_constants import MLPTKGConstants
from .projection_utils import (
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
    use_dge: bool = False,
    prealloc_gate_up_buffers=None,
    gate_up_weights: Optional[nl.NkiTensor] = None,
):
    """
    Runs the Gate/Up projection using the LHS/RHS swap matmul path.

    This is the `else` branch of `process_gate_up_projection` when
    `params.use_tkg_gate_up_proj_column_tiling` is False.
    gate_up_weights is this expert's optional [H, 2 * I] view, used for the fused gate/up load when it applies
    (Trn2, auto-alloc, I <= 512).

    Returns:
        tiles, gate_sb_view, up_sb_view, up_sb_fp32, gate_up_recv
    """
    gate_w, up_w = params.gate_proj_weights_tensor, params.up_proj_weights_tensor
    gate_b, up_b, gate_w_scale, up_w_scale = prepare_gate_up_bias_and_scale(params, dims)

    num_I_tiles = div_ceil(dims.I, dims.I0)
    bias_tile = None
    bias_size = 0
    tile_shape = (dims.I0, num_I_tiles, dims.T)

    if not params.skip_gate_proj:
        if prealloc_gate_up_buffers is not None:
            gate_sb_fp32, up_sb_fp32 = prealloc_gate_up_buffers
            nisa.memset(gate_sb_fp32, 0)
            nisa.memset(up_sb_fp32, 0)
        else:
            gate_sb_fp32 = sbm.alloc_stack(
                tile_shape,
                dtype=nl.float32,
                name="gate_sbuf_fp32",
                buffer=nl.sbuf,
                align=4,
            )
            up_sb_fp32 = sbm.alloc_stack(
                tile_shape,
                dtype=nl.float32,
                name="up_sbuf_fp32",
                buffer=nl.sbuf,
                align=4,
            )
        gate_sb_view = gate_sb_fp32
        up_sb_view = up_sb_fp32
    else:
        up_sb_fp32 = sbm.alloc_stack(
            tile_shape,
            dtype=nl.float32,
            name="up_sbuf_fp32",
            buffer=nl.sbuf,
            align=4,
        )
        up_sb_view = up_sb_fp32
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
            dge_mode=nisa.dge_mode.hwdge if use_dge else adaptive_dge_mode(gate_w_scale_view),
        )
        nisa.dma_copy(
            dst=up_dequant_tile,
            src=up_w_scale_view,
            dge_mode=nisa.dge_mode.hwdge if use_dge else adaptive_dge_mode(up_w_scale_view),
        )

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
            dtype=nl.float32
            if (gate_b._scalar_offset is not None and gate_b._indirect_dim is not None)
            else gate_b.dtype,
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
    w_tile_dtype = _fp8_e4m3_tile_dtype if str(up_w.dtype) == "float8e4" else up_w.dtype

    # Trn2 auto-alloc only: gate and up load as one [H, 2 * I] stream, with both PSUM accumulator sets live
    use_fused_gate_up_load = (
        gate_up_weights is not None
        and nisa.get_nc_version() == nisa.nc_version.gen3
        and sbm.is_auto_alloc()
        and not params.skip_gate_proj
        and 2 * num_I_tiles <= dims._psum_bmax
    )
    if use_fused_gate_up_load:
        # Auto-alloc pins HTile at its minimum; size the fused tile by bytes per partition, in balanced H tiles,
        # and re-tile only when that at least halves the number of DMAs
        h1_cap = _FUSED_GATE_UP_TILE_BYTES // (2 * dims.I * sizeinbytes(w_tile_dtype))
        if h1_cap >= 2 * HTile_h1:
            HTile_h1 = div_ceil(dims.H1_shard, div_ceil(dims.H1_shard, h1_cap))
            tiles = MLPTKGConstantsGateUpTileCounts(
                HTile=HTile_h1 * dims.H0,
                num_allocated_w_tile=tiles.num_allocated_w_tile,
                last_accessed_addr=tiles.last_accessed_addr,
                up_psum_base_bank=tiles.up_psum_base_bank,
                I_shard_size=tiles.I_shard_size,
            )
        weight_shape = (dims.H0, HTile_h1, 2 * dims.I)
    for w_tile_idx in range(tiles.num_allocated_w_tile):
        weight_tile = sbm.alloc_stack(
            weight_shape,
            name=f"gate_up_fused_w_tile_{w_tile_idx}" if use_fused_gate_up_load else f"gate_up_w_tile_{w_tile_idx}",
            dtype=w_tile_dtype,
        )
        weight_tiles.append(weight_tile)

    if use_fused_gate_up_load:
        h_offset = dims.H1_offset * dims.H0
        t_dim = dims.hidden_layout.get_t_dim()
        _gate_up_projection_fused_lhs_rhs_swap(
            hidden=hidden.slice(dim=t_dim, start=T_offset, end=T_offset + dims.T),
            weight=gate_up_weights.slice(dim=0, start=h_offset, end=h_offset + dims.H_per_shard),
            gate_bias=gate_b.slice(dim=0, start=0, end=dims.I) if gate_b is not None else None,
            up_bias=up_b.slice(dim=0, start=0, end=dims.I) if up_b is not None else None,
            gate_dequant_tile=gate_dequant_tile,
            up_dequant_tile=up_dequant_tile,
            gate_output_tile=gate_sb_view,
            up_output_tile=up_sb_view,
            weight_tiles=weight_tiles,
            bias_tile=bias_tile,
            dims=dims,
            tiles=tiles,
            params=params,
            sbm=sbm,
            use_dge=use_dge,
        )
        return tiles, gate_sb_view, up_sb_view, gate_up_recv

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
                use_dge=use_dge,
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
            use_dge=use_dge,
        )

    return tiles, gate_sb_view, up_sb_view, gate_up_recv
