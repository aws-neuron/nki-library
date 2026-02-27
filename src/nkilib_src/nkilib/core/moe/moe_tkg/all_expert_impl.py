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

"""All-expert MoE token generation implementation that processes all experts sequentially."""

from dataclasses import dataclass

import nki.isa as nisa
import nki.language as nl
import nki.tensor as ntensor

from ...mlp.mlp_parameters import (
    MLPBiasParameters,
    MLPParameters,
    MLPQuantizationParameters,
)

# MLP utils
from ...mlp.mlp_tkg.mlp_tkg_constants import MLPTKGConstants
from ...mlp.mlp_tkg.mlp_tkg_down_projection import process_down_projection
from ...mlp.mlp_tkg.mlp_tkg_gate_up_projection import process_gate_up_projection
from ...mlp.mlp_tkg.mlp_tkg_utils import input_norm_load, transpose_store
from ...utils.allocator import SbufManager

# common utils
from ...utils.common_types import ExpertAffinityScaleMode, GateUpDim, QuantizationType
from ...utils.kernel_assert import kernel_assert
from ...utils.logging import get_logger
from ...utils.tensor_view import TensorView
from ...utils.tiled_range import TiledRange
from .moe_tkg_utils import reshape_scale_for_mlp, safe_tensor_view


@dataclass
class TokenTilingConfig(nl.NKIObject):
    """Configuration for T-dimension tiling in all-expert MoE kernel.

    Encapsulates tiling parameters to avoid mutating dims.T during iteration.
    """

    T_total: int  # Original total token count
    tile_T: int  # Size of each tile (min(T_total, pmax))
    num_tiles: int  # Number of T-tiles


def _create_token_tiling_config(dims) -> TokenTilingConfig:
    """Create TokenTilingConfig from MLPTKGConstants dimensions."""
    T_total = dims.T
    pmax = dims._pmax
    tile_T = min(T_total, pmax)
    num_tiles = (T_total + pmax - 1) // pmax
    return TokenTilingConfig(T_total=T_total, tile_T=tile_T, num_tiles=num_tiles)


def _all_expert_moe_tkg(
    params: MLPParameters,
    output: nl.ndarray,
) -> nl.ndarray:
    """
    All-expert Mixture of Experts (MoE) kernel for token generation (TKG).

    Processes all experts sequentially, computing MLP projections for each expert
    and accumulating results weighted by expert affinities.

    Supports T > 128 via T-tiling: when T exceeds pmax (128), the token dimension
    is processed in tiles of pmax, with each tile going through all experts before
    storing results and moving to the next tile.

    Args:
        params (MLPParameters): MLPParameters containing model configuration, weights, and input tensors.
        output (nl.ndarray): Output tensor to store the final result.

    Returns:
        output (nl.ndarray): Output tensor with accumulated expert results.

    Notes:
        - Column tiling is not supported; only swapped LHS/RHS projection is used.

    Pseudocode:
        output_sb[tile_T, num_T_tiles, H] = zeros()

        for expert_idx in range(E):
            weights = load_expert_weights(expert_idx)
            for t_tile in range(num_T_tiles):
                input_sb[H0, tile_T, H1] = load(hidden_tensor[t_offset:t_offset+tile_T, H])
                gate_up[I0, I1, tile_T] = gate_up_proj(input_sb, weights)
                down[H0, H1_shard, tile_T] = down_proj(gate_up, weights)
                if POST_SCALE: down *= affinity
                output_sb[:, t_tile, :] += transpose(down)

        for t_tile in range(num_T_tiles):
            output[t_offset:t_offset+tile_T, H] = output_sb[:, t_tile, :]
    """
    io_dtype = params.hidden_tensor.dtype
    expert_affinities = params.expert_params.expert_affinities
    expert_affinities_in_sbuf = expert_affinities.buffer == nl.sbuf

    if params.use_tkg_gate_up_proj_column_tiling or params.use_tkg_down_proj_column_tiling:
        kernel_assert(False, "Column tiling is not supported in all-expert MLP kernel, only swapped LHS/RHS projection")

    hidden_in_sbuf = params.hidden_tensor.buffer == nl.sbuf

    dims = MLPTKGConstants.calculate_constants(params)

    # Create T-tiling config (preserves original dims.T, avoids mutation)
    t_cfg = _create_token_tiling_config(dims)

    kernel_assert(
        not hidden_in_sbuf or params.hidden_tensor.shape[1] == t_cfg.T_total,
        f"SBUF input shape mismatch: expected T dim={t_cfg.T_total}, got {params.hidden_tensor.shape[1]}",
    )

    H = params.hidden_tensor.shape[-1]
    # Note: always use auto allocation since expert_affinities is always in SBUF
    use_auto_alloc = H >= 16 * 1024 or hidden_in_sbuf or expert_affinities_in_sbuf
    sbm = SbufManager(0, 200000, get_logger("all_expert_moe_tkg"), use_auto_alloc=use_auto_alloc)

    sbm.open_scope()
    allocator = sbm.alloc_stack

    # Wrap weight/bias tensors in TensorView for slicing (shared across all T-tiles)
    gate_proj_weights_view = TensorView(params.gate_proj_weights_tensor)
    up_proj_weights_view = TensorView(params.up_proj_weights_tensor)
    down_proj_weights_view = TensorView(params.down_proj_weights_tensor)

    gate_proj_bias_view = safe_tensor_view(params.bias_params.gate_proj_bias_tensor)
    up_proj_bias_view = safe_tensor_view(params.bias_params.up_proj_bias_tensor)
    down_proj_bias_view = safe_tensor_view(params.bias_params.down_proj_bias_tensor)

    gate_w_scale_view = safe_tensor_view(params.quant_params.gate_w_scale)
    up_w_scale_view = safe_tensor_view(params.quant_params.up_w_scale)
    down_w_scale_view = safe_tensor_view(params.quant_params.down_w_scale)
    gate_up_in_scale_view = safe_tensor_view(params.quant_params.gate_up_in_scale)
    down_in_scale_view = safe_tensor_view(params.quant_params.down_in_scale)

    # Allocate accumulation buffer [H0, H1_shard, tile_T * num_tiles]
    output_temp = allocator(
        (dims.H0, dims.H1_shard, t_cfg.tile_T * t_cfg.num_tiles), dtype=io_dtype, buffer=nl.sbuf, name="output_temp"
    )
    output_in_sbuf = output.buffer == nl.sbuf

    # Load expert affinities for all T-tiles upfront
    pmax = dims._pmax
    if expert_affinities_in_sbuf:
        expert_affinities_sb = expert_affinities
    else:
        if t_cfg.num_tiles == 1:
            expert_affinities_sb = allocator(
                (t_cfg.T_total, dims.E), dtype=expert_affinities.dtype, buffer=nl.sbuf, name="expertAffinityAll"
            )
            nisa.dma_copy(
                dst=expert_affinities_sb[: t_cfg.T_total, : dims.E], src=expert_affinities[: t_cfg.T_total, : dims.E]
            )
        else:
            expert_affinities_sb = allocator(
                (t_cfg.tile_T, t_cfg.num_tiles, dims.E),
                dtype=expert_affinities.dtype,
                buffer=nl.sbuf,
                name="expertAffinityAll",
            )
            for t_tile in TiledRange(t_cfg.T_total, t_cfg.tile_T):
                nisa.dma_copy(
                    dst=expert_affinities_sb[: t_tile.size, t_tile.index, : dims.E],
                    src=expert_affinities[nl.ds(t_tile.start_offset, t_tile.size), : dims.E],
                )

    memory_safe_degree = 2 if t_cfg.tile_T * dims.H * dims.I <= 32 * 3072 * 1024 else 1

    # Pre-load identity matrix (reused across all experts)
    identity_hbm = nl.shared_constant(ntensor.identity(t_cfg.tile_T, nl.int8))
    identity_sb = allocator((t_cfg.tile_T, t_cfg.tile_T), dtype=io_dtype, buffer=nl.sbuf, name="identity_sb")
    nisa.dma_copy(dst=identity_sb, src=identity_hbm)

    # Pre-load all input tiles (reused across all experts, avoids redundant DMA per expert)
    if hidden_in_sbuf:
        input_sb_tiles = None
    else:
        input_sb_tiles = []
        for t_tile in TiledRange(t_cfg.T_total, t_cfg.tile_T):
            dims.T = t_tile.size  # Temporarily set for input_norm_load
            isb = allocator(
                [dims.H0, t_tile.size, dims.H1],
                dtype=io_dtype,
                buffer=nl.sbuf,
                name=f"input_sb_t{t_tile.index}",
            )
            input_norm_load(params.hidden_tensor, isb, params, dims, sbm=sbm, T_offset=t_tile.start_offset)
            input_sb_tiles.append(isb)

    # E-then-T: outer loop over experts (load weights once), inner loop over T-tiles
    for expertIdx in range(dims.E):
        sbm.set_name_prefix(f"expert{expertIdx}_")

        # Pre-slice weights for this expert
        expert_gate_w = gate_proj_weights_view.select(dim=0, index=expertIdx)
        expert_up_w = up_proj_weights_view.select(dim=0, index=expertIdx)
        expert_down_w = down_proj_weights_view.select(dim=0, index=expertIdx)

        if len(expert_gate_w.shape) > 2:
            expert_gate_w = expert_gate_w.select(dim=1, index=GateUpDim.GATE.value)
            expert_up_w = expert_up_w.select(dim=1, index=GateUpDim.UP.value)

        expert_gate_b, expert_up_b, expert_down_b = None, None, None
        if gate_proj_bias_view != None:
            expert_gate_b = gate_proj_bias_view.select(dim=0, index=expertIdx)
            if len(expert_gate_b.shape) > 1:
                expert_gate_b = expert_gate_b.select(dim=0, index=GateUpDim.GATE.value)
        if up_proj_bias_view != None:
            expert_up_b = up_proj_bias_view.select(dim=0, index=expertIdx)
            if len(expert_up_b.shape) > 1:
                expert_up_b = expert_up_b.select(dim=0, index=GateUpDim.UP.value)
        if down_proj_bias_view != None:
            expert_down_b = down_proj_bias_view.select(dim=0, index=expertIdx)

        params.gate_proj_weights_tensor = expert_gate_w
        params.up_proj_weights_tensor = expert_up_w
        params.down_proj_weights_tensor = expert_down_w
        params.bias_params = MLPBiasParameters(
            gate_proj_bias_tensor=expert_gate_b,
            up_proj_bias_tensor=expert_up_b,
            down_proj_bias_tensor=expert_down_b,
        )

        if params.quant_params.quantization_type != QuantizationType.NONE:
            params.quant_params = _select_quant_scales(
                params.quant_params,
                gate_w_scale_view,
                up_w_scale_view,
                down_w_scale_view,
                gate_up_in_scale_view,
                down_in_scale_view,
                expertIdx,
            )

        # Inner loop over T-tiles for this expert
        sbm.open_scope(interleave_degree=memory_safe_degree)
        for t_tile in TiledRange(t_cfg.T_total, t_cfg.tile_T):
            current_tile_T = t_tile.size
            t_offset = t_tile.start_offset
            t_idx = t_tile.index
            sbm.set_name_prefix(f"expert{expertIdx}_t{t_idx}_")

            dims.T = current_tile_T  # Temporarily set for sub-kernels

            # Get expert affinity for this T-tile
            if t_cfg.num_tiles == 1:
                expertAffinityLoc = expert_affinities_sb
            else:
                expertAffinityLoc = allocator(
                    (current_tile_T, dims.E), dtype=expert_affinities_sb.dtype, buffer=nl.sbuf, name="expertAffinityLoc"
                )
                nisa.tensor_copy(
                    dst=expertAffinityLoc[:current_tile_T, : dims.E],
                    src=expert_affinities_sb[:current_tile_T, t_idx, : dims.E],
                )

            # Use pre-loaded input for this T-tile
            if hidden_in_sbuf:
                input_sb = params.hidden_tensor
            else:
                input_sb = input_sb_tiles[t_idx]

            # Slice pre-allocated buffers to current tile size
            gate_up_sb = allocator(
                (dims.I0, dims.num_total_128_tiles_per_I, current_tile_T),
                dtype=nl.float32,
                name="gate_up_sbuf",
                buffer=nl.sbuf,
            )
            down_sb = allocator(
                (dims.H0, dims.H1_shard, current_tile_T), dtype=io_dtype, name="down_sbuf", buffer=nl.sbuf
            )

            # Gate Up projection
            gate_tile_info = process_gate_up_projection(
                hidden=input_sb,
                output=gate_up_sb,
                params=params,
                dims=dims,
                sbm=sbm,
                T_offset=t_offset if hidden_in_sbuf else 0,
            )

            # Compute POST_SCALE affinity broadcast
            if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
                expert_affinities_broadcast = allocator(
                    (pmax, current_tile_T), dtype=io_dtype, buffer=nl.sbuf, name="expert_affinities_sb"
                )
                affinityTpPsum = nl.ndarray(
                    (1, current_tile_T), dtype=nl.float32, buffer=nl.psum, address=None if use_auto_alloc else (0, 0)
                )
                expertAffinityLoc_cast = allocator(
                    (current_tile_T, 1), dtype=io_dtype, buffer=nl.sbuf, name="expertAffinityLoc_cast"
                )
                nisa.activation(
                    dst=expertAffinityLoc_cast[:current_tile_T, :],
                    op=nl.copy,
                    data=expertAffinityLoc[:current_tile_T, nl.ds(expertIdx, 1)],
                )
                nisa.nc_matmul(
                    dst=affinityTpPsum[:, :],
                    stationary=expertAffinityLoc_cast,
                    moving=identity_sb[:current_tile_T, :current_tile_T],
                )
                nisa.tensor_copy(
                    src=affinityTpPsum[:, :],
                    dst=expert_affinities_broadcast[:1, :current_tile_T],
                )
                for partition_group_idx in range(4):
                    nisa.nc_stream_shuffle(
                        dst=expert_affinities_broadcast[nl.ds(32 * partition_group_idx, 32), :current_tile_T],
                        src=expert_affinities_broadcast[:1, :current_tile_T],
                        shuffle_mask=[0] * 32,
                    )

            # Down projection
            gate_up_sb_casted = allocator(
                (dims.I0, dims.num_total_128_tiles_per_I, current_tile_T),
                dtype=io_dtype,
                name="gate_up_sbuf_with_io_dtype",
                buffer=nl.sbuf,
            )
            nisa.tensor_copy(dst=gate_up_sb_casted, src=gate_up_sb)
            process_down_projection(
                hidden=gate_up_sb_casted,
                output=down_sb,
                params=params,
                dims=dims,
                gate_tile_info=gate_tile_info,
                sbm=sbm,
            )

            # Apply affinity scaling to down_sb [H0, H1_shard, current_tile_T]
            if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
                for shard in range(dims.H1_shard):
                    nisa.tensor_tensor(
                        data1=down_sb[: dims.H0, shard, :current_tile_T],
                        data2=expert_affinities_broadcast,
                        op=nl.multiply,
                        dst=down_sb[: dims.H0, shard, :current_tile_T],
                    )

            # Accumulate down_sb [H0, H1_shard, tile_T] into output_temp
            t_free_offset = t_idx * t_cfg.tile_T
            if expertIdx == 0:
                nisa.tensor_copy(
                    dst=output_temp[: dims.H0, : dims.H1_shard, nl.ds(t_free_offset, current_tile_T)],
                    src=down_sb[: dims.H0, : dims.H1_shard, :current_tile_T],
                )
            else:
                nisa.tensor_tensor(
                    data1=down_sb[: dims.H0, : dims.H1_shard, :current_tile_T],
                    data2=output_temp[: dims.H0, : dims.H1_shard, nl.ds(t_free_offset, current_tile_T)],
                    op=nl.add,
                    dst=output_temp[: dims.H0, : dims.H1_shard, nl.ds(t_free_offset, current_tile_T)],
                )

            sbm.increment_section()
        sbm.close_scope()

    # Store: transpose [H0, H1_shard, tile_T] to [tile_T, H] and write to HBM (once per T-tile)
    if output_in_sbuf:
        for t_tile in TiledRange(t_cfg.T_total, t_cfg.tile_T):
            t_free_offset = t_tile.index * t_cfg.tile_T
            for h1 in range(dims.H1_shard):
                nisa.tensor_copy(
                    dst=output[:, nl.ds(t_tile.start_offset, t_tile.size), h1],
                    src=output_temp[:, h1, nl.ds(t_free_offset, t_tile.size)],
                )
    else:
        for t_tile in TiledRange(t_cfg.T_total, t_cfg.tile_T):
            sbm.set_name_prefix(f"store_t{t_tile.index}_")
            t_free_offset = t_tile.index * t_cfg.tile_T
            dims.T = t_tile.size  # Temporarily set for transpose_store
            transpose_store(
                output_temp[: dims.H0, : dims.H1_shard, nl.ds(t_free_offset, t_tile.size)],
                output,
                dims,
                params.output_dtype,
                sbm,
                T_offset=t_tile.start_offset,
            )

    sbm.set_name_prefix("")

    sbm.close_scope()
    return output


def _select_quant_scales(
    quant_params: MLPQuantizationParameters,
    gate_w_scale_view: TensorView,
    up_w_scale_view: TensorView,
    down_w_scale_view: TensorView,
    gate_up_in_scale_view: TensorView,
    down_in_scale_view: TensorView,
    expertIdx: int,
):
    """
    Select and reshape quantization scales for a specific expert.

    Args:
        quant_params (MLPQuantizationParameters): Quantization parameters.
        gate_w_scale_view (TensorView): Gate weight scale tensor view.
        up_w_scale_view (TensorView): Up weight scale tensor view.
        down_w_scale_view (TensorView): Down weight scale tensor view.
        gate_up_in_scale_view (TensorView): Gate/up input scale tensor view.
        down_in_scale_view (TensorView): Down input scale tensor view.
        expertIdx (int): Expert index to select scales for.

    Returns:
        MLPQuantizationParameters: Quantization parameters with scales for the specified expert.
    """
    quantization_type = quant_params.quantization_type
    expert_gate_w_scale = None
    if gate_w_scale_view != None:
        expert_gate_w_scale = gate_w_scale_view.select(dim=0, index=expertIdx).select(dim=0, index=GateUpDim.GATE.value)
        expert_gate_w_scale = reshape_scale_for_mlp(expert_gate_w_scale)

    expert_up_w_scale = None
    if up_w_scale_view != None:
        expert_up_w_scale = up_w_scale_view.select(dim=0, index=expertIdx).select(dim=0, index=GateUpDim.UP.value)
        expert_up_w_scale = reshape_scale_for_mlp(expert_up_w_scale)

    expert_down_w_scale = None
    if down_w_scale_view != None:
        expert_down_w_scale = down_w_scale_view.select(dim=0, index=expertIdx)
        expert_down_w_scale = reshape_scale_for_mlp(expert_down_w_scale)

    expert_gate_up_in_scale = None
    if gate_up_in_scale_view != None:
        expert_gate_up_in_scale = reshape_scale_for_mlp(gate_up_in_scale_view.select(dim=0, index=expertIdx))

    expert_down_in_scale = None
    if down_in_scale_view != None:
        expert_down_in_scale = reshape_scale_for_mlp(down_in_scale_view.select(dim=0, index=expertIdx))

    return MLPQuantizationParameters(
        quantization_type=quantization_type,
        gate_w_scale=expert_gate_w_scale,
        up_w_scale=expert_up_w_scale,
        down_w_scale=expert_down_w_scale,
        gate_up_in_scale=expert_gate_up_in_scale,
        down_in_scale=expert_down_in_scale,
        clipping_bound=quant_params.clipping_bound if quant_params != None else None,
    )
