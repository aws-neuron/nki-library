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

"""All-expert MoE token generation implementation with static and dynamic (DLoC) dispatch."""

from dataclasses import dataclass

import nki.isa as nisa
import nki.language as nl
from nki.isa import oob_mode

from ...utils.allocator import SbufManager

# common utils
from ...utils.common_types import ExpertAffinityScaleMode, GateUpDim, MoEAllToAllVStrategy, QuantizationType
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...utils.logging import get_logger
from ...utils.tiled_range import TiledRange
from .mlp_parameters import (
    MLPBiasParameters,
    MLPParameters,
    MLPQuantizationParameters,
)

# MLP utils
from .mlp_tkg_constants import MLPTKGConstants
from .mlp_tkg_down_projection import process_down_projection
from .mlp_tkg_gate_up_projection import process_gate_up_projection
from .moe_tkg_utils import (
    broadcast_all_expert_affinity,
    get_all_expert_tile_affinities,
    load_all_expert_affinities,
    reshape_scale_for_mlp,
)
from .projection_utils import input_norm_load, transpose_store


@dataclass
class TokenTilingConfig(nl.NKIObject):
    """Configuration for T-dimension tiling in all-expert MoE kernel.

    Encapsulates tiling parameters to avoid mutating dims.T during iteration.
    """

    T_total: int  # Original total token count
    tile_T: int  # Size of each tile (min(T_total, pmax))
    num_tiles: int  # Number of T-tiles


NUM_DYNAMIC_ALGO_STATIC_BLOCKS = 1
NONZERO_WITH_COUNT_PAD_VAL = -1  # Pad indices with -1 to utilize DMA oob_mode.skip
BF16_PER_INT32 = 2  # Number of bf16 elements per int32 (for token index passthrough)


@dataclass
class AllExpertDynamismConfig(nl.NKIObject):
    """Dynamic control flow config for all-expert kernel."""

    is_all_expert_dynamic: bool = False
    block_size: int = None
    all_to_all_v_strategy: MoEAllToAllVStrategy = MoEAllToAllVStrategy.DISABLED

    def __post_init__(self):
        """Initialize derived fields to zero; populated by derive_from_dims."""
        self.n_blocks = 0
        self.n_static_blocks = NUM_DYNAMIC_ALGO_STATIC_BLOCKS
        self.n_dynamic_blocks = 0
        self.T_plus_1 = 0
        self.n_dynamic_blocks_plus_1 = 0
        self.blk_tile_T = 0
        self.blk_n_T_tiles = 0

    def derive_from_dims(self, T: int):
        """Derive dynamic tiling constants from T and block_size."""
        pmax = nl.tile_size.pmax

        self.n_blocks = T // self.block_size
        self.n_dynamic_blocks = self.n_blocks - self.n_static_blocks

        self.T_plus_1 = T + 1
        self.n_dynamic_blocks_plus_1 = self.n_dynamic_blocks + 1

        self.blk_tile_T = min(pmax, self.block_size)
        self.blk_n_T_tiles = div_ceil(self.block_size, pmax)


def _init_dynamism_config(params: MLPParameters) -> AllExpertDynamismConfig:
    """Extract dynamism config from MLPParameters."""
    T = params.batch_size * params.sequence_len
    cfg = AllExpertDynamismConfig(
        is_all_expert_dynamic=params.expert_params.is_all_expert_dynamic,
        block_size=params.expert_params.block_size if params.expert_params.block_size is not None else T,
        all_to_all_v_strategy=params.expert_params.all_to_all_v_strategy,
    )
    if cfg.all_to_all_v_strategy != MoEAllToAllVStrategy.DISABLED:
        cfg.n_static_blocks = 0
    if cfg.is_all_expert_dynamic:
        cfg.derive_from_dims(T)
    return cfg


def _create_token_tiling_config(dims) -> TokenTilingConfig:
    """Create TokenTilingConfig from MLPTKGConstants dimensions."""
    T_total = dims.T
    # LHS/RHS swap path places T in the free dimension (psum_fmax),
    # but the down projection packs H1_shard tiles into PSUM banks,
    # constraining T to: psum_fmax // ceil(H1_shard / psum_bmax).
    pmax = dims._pmax
    psum_fmax = dims._psum_fmax
    psum_bmax = dims._psum_bmax
    min_perBankT = div_ceil(dims.H1_shard, psum_bmax)
    tile_limit = psum_fmax // max(min_perBankT, 1)
    tile_limit = min(tile_limit, psum_fmax)
    # Round down to multiple of pmax so affinity chunking aligns cleanly
    tile_limit = (tile_limit // pmax) * pmax
    tile_limit = max(tile_limit, pmax)  # at least pmax
    tile_T = min(T_total, tile_limit)
    num_tiles = div_ceil(T_total, tile_limit)
    return TokenTilingConfig(T_total=T_total, tile_T=tile_T, num_tiles=num_tiles)


def _extract_a2av_affinities(hidden_input, H, E_L, T, pmax):
    """Extract expert affinities from concatenated A2AV input buffer.

    Input layout: [T, H + E_L + 2] in bf16.
    Affinities are at columns [H : H+E_L], shape [T, E_L].

    Returns:
        expert_affinities_hbm (nl.NkiTensor): [T, E_L] affinities in private HBM.
    """
    expert_affinities_hbm = nl.ndarray((T, E_L), dtype=hidden_input.dtype, buffer=nl.private_hbm)
    # Copy affinity columns from concatenated input to separate HBM buffer
    nisa.dma_copy(
        src=hidden_input[:T, nl.ds(H, E_L)],
        dst=expert_affinities_hbm,
    )
    return expert_affinities_hbm


def _copy_a2av_token_indices_to_output(hidden_input, output, H, E_L, T, strategy):
    """Copy token indices from A2AV input to output buffer.

    Input token_index at columns [H+E_L : H+E_L+2] (int32 as 2xbf16).
    Output token_index at columns [H : H+2] (appended after hidden output).

    For PRESERVE_ROW_ORDER: direct copy of all T rows (both NCs shard on T).
    For PACK_OUTPUT_ROWS: deferred — indices are scattered during output write.
    """
    if strategy == MoEAllToAllVStrategy.PRESERVE_ROW_ORDER:
        _, n_prgs, prg_id = _get_program_sharding_info()
        T_local = T // n_prgs
        T_offset = prg_id * T_local
        nisa.dma_copy(
            src=hidden_input[nl.ds(T_offset, T_local), nl.ds(H + E_L, BF16_PER_INT32)],
            dst=output[nl.ds(T_offset, T_local), nl.ds(H, BF16_PER_INT32)],
        )


def _get_program_sharding_info():
    """Get verified program sharding info for LNC-2."""
    return get_verified_program_sharding_info()


def _build_pack_output_indices(expert_affinities, T, E, pmax, dynamism_cfg, hidden_input, output, H):
    """Build packed output index mapping for PACK_OUTPUT_ROWS with E>1.

    Finds all tokens routed to ANY expert (nonzero affinity in any column),
    builds output_indices_hbm[routed_pos] = packed_pos, and copies token indices
    to packed output positions.

    Args:
        expert_affinities (nl.NkiTensor): [T, E] affinities in HBM.
        T: Total tokens.
        E: Number of local experts.
        pmax: Partition max.
        dynamism_cfg: Dynamism config.
        hidden_input (nl.NkiTensor): [T, H+E+2] concatenated input in HBM.
        output (nl.NkiTensor): [T, H+2] output in HBM.
        H: Hidden dimension.

    Returns:
        output_indices_hbm (nl.NkiTensor): [T, 1] in private HBM. Maps routed input position → packed output row.
    """
    tile_T = dynamism_cfg.blk_tile_T
    n_T_tiles = dynamism_cfg.blk_n_T_tiles
    if T > pmax:
        tile_T = pmax
        n_T_tiles = div_ceil(T, pmax)

    # Step 1: Sum affinities across experts to find tokens routed to ANY expert
    # Load [T, E] affinities, reduce across E to get [1, T] indicator
    aff_sum_sb = nl.ndarray((pmax, T), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(aff_sum_sb, 0)
    for e in range(E):
        expert_aff_sb = nl.ndarray((1, T), dtype=expert_affinities.dtype, buffer=nl.sbuf)
        src_view = expert_affinities.select(dim=1, index=e).expand_dim(dim=1).expand_dim(dim=1).expand_dim(dim=1)
        dst_view = expert_aff_sb.expand_dim(dim=1).expand_dim(dim=1)
        nisa.dma_transpose(src=src_view, dst=dst_view)
        # Accumulate absolute value (affinity can be negative)
        aff_abs_sb = nl.ndarray((1, T), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(data=expert_aff_sb, op0=nl.abs, operand0=0, dst=aff_abs_sb)
        nisa.tensor_tensor(data1=aff_sum_sb[0, :T], data2=aff_abs_sb[0, :T], op=nl.add, dst=aff_sum_sb[0, :T])

    # Step 2: nonzero_with_count on the sum to find all routed positions
    routed_indices_sb = nl.ndarray((pmax, T + 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(src=aff_sum_sb[...], padding_val=NONZERO_WITH_COUNT_PAD_VAL, dst=routed_indices_sb[...])

    # Step 3: Build output_indices_hbm: scatter iota into HBM at routed positions
    # output_indices_hbm[routed_indices[i]] = i (packed position)
    output_indices_hbm = nl.ndarray((T, 1), dtype=nl.int32, buffer=nl.private_hbm)
    # Init to -1 so unrouted positions produce OOB (skipped by oob_mode.skip downstream)
    neg_one_sb = nl.ndarray((tile_T, 1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.memset(neg_one_sb, NONZERO_WITH_COUNT_PAD_VAL)
    for t_idx in range(n_T_tiles):
        t_actual = min(tile_T, T - t_idx * tile_T)
        nisa.dma_copy(src=neg_one_sb[:t_actual, :], dst=output_indices_hbm[nl.ds(t_idx * tile_T, t_actual), :])

    iota_sb = nl.ndarray((tile_T, n_T_tiles), dtype=nl.int32, buffer=nl.sbuf)
    nisa.iota(dst=iota_sb, pattern=[[tile_T, n_T_tiles]], offset=0, channel_multiplier=1)

    # Transpose routed_indices to [tile_T, n_T_tiles] for use as vector_offset
    routed_T_psum = nl.ndarray((tile_T, n_T_tiles), dtype=nl.float32, buffer=nl.psum)
    routed_T_sb = nl.ndarray((tile_T, n_T_tiles), dtype=nl.float32, buffer=nl.sbuf)
    for t_idx in range(n_T_tiles):
        nisa.nc_transpose(
            data=routed_indices_sb.view(nl.float32)[0, nl.ds(tile_T * t_idx, tile_T)],
            dst=routed_T_psum[:, t_idx],
        )
    nisa.tensor_copy(src=routed_T_psum, dst=routed_T_sb)
    routed_T_int32_sb = routed_T_sb.view(nl.int32)

    for t_idx in range(n_T_tiles):
        nisa.dma_copy(
            src=iota_sb[:, t_idx : t_idx + 1],
            dst=output_indices_hbm.ap(
                pattern=[[1, tile_T], [1, 1]],
                offset=0,
                vector_offset=routed_T_int32_sb.ap(
                    pattern=[[n_T_tiles, tile_T], [1, 1]],
                    offset=t_idx,
                ),
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip,
            dge_mode=nisa.dge_mode.swdge,
        )

    # Step 4: Copy token indices from input to packed output positions
    _, n_prgs, prg_id = _get_program_sharding_info()
    input_row_stride = hidden_input.shape[-1]
    E_L = E
    for t_idx in range(n_T_tiles):
        tile_T_actual = min(tile_T, T - t_idx * tile_T)
        token_idx_tile_sb = nl.ndarray((tile_T, BF16_PER_INT32), dtype=hidden_input.dtype, buffer=nl.sbuf)
        nisa.memset(token_idx_tile_sb, 0)
        # Gather token indices from input at routed positions
        nisa.dma_copy(
            src=hidden_input.ap(
                pattern=[[input_row_stride, tile_T], [1, BF16_PER_INT32]],
                offset=H + E_L,
                vector_offset=routed_T_int32_sb.ap(
                    pattern=[[n_T_tiles, tile_T], [1, 1]],
                    offset=t_idx,
                ),
                indirect_dim=0,
            ),
            dst=token_idx_tile_sb[:tile_T_actual, :BF16_PER_INT32],
            oob_mode=oob_mode.skip,
            dge_mode=nisa.dge_mode.swdge,
        )
        # Write to packed output positions (sequential since iota ordering)
        if prg_id == 0:
            nisa.dma_copy(
                src=token_idx_tile_sb[:tile_T_actual, :BF16_PER_INT32],
                dst=output[nl.ds(t_idx * tile_T, tile_T_actual), nl.ds(H, BF16_PER_INT32)],
            )

    return output_indices_hbm


def _all_expert_moe_tkg(
    params: MLPParameters,
    output: nl.NkiTensor,
) -> nl.NkiTensor:
    """
    All-expert MoE kernel for token generation (TKG). Dispatches to static or dynamic implementation.

    Args:
        params (MLPParameters): MLPParameters containing model configuration, weights, and input tensors.
        output (nl.NkiTensor): Output tensor to store the final result.

    Returns:
        output (nl.NkiTensor): Output tensor with accumulated expert results.
    """
    dynamism_cfg = _init_dynamism_config(params)

    if dynamism_cfg.is_all_expert_dynamic:
        return _all_expert_moe_tkg_dynamic(params, dynamism_cfg, output)
    else:
        return _all_expert_moe_tkg_static(params, output)


def _all_expert_moe_tkg_dynamic(
    params: MLPParameters,
    dynamism_cfg: AllExpertDynamismConfig,
    output: nl.NkiTensor,
) -> nl.NkiTensor:
    """
    All-expert MoE with dynamic control flow (DLoC).

    For each expert, finds routed tokens via nonzero_with_count, then processes
    static blocks (always executed) and dynamic blocks (skipped at runtime if
    no routed tokens remain). Each block gathers input by token indices, computes
    the expert MLP, and scatters output back to HBM.

    Currently restricted to E == 1 (single local expert after expert parallelism),
    matching the MX dynamic path constraint.

    Args:
        params (MLPParameters): MLPParameters containing model configuration, weights, and input tensors.
        dynamism_cfg (AllExpertDynamismConfig): Dynamic control flow configuration.
        output (nl.NkiTensor): Output tensor [T, H] in HBM.

    Returns:
        output (nl.NkiTensor): Output tensor with accumulated expert results.
    """
    io_dtype = params.hidden_tensor.dtype
    expert_affinities = params.expert_params.expert_affinities

    kernel_assert(
        not (params.use_tkg_gate_up_proj_column_tiling or params.use_tkg_down_proj_column_tiling),
        "Column tiling is not supported in all-expert dynamic MLP kernel",
    )
    kernel_assert(
        params.hidden_tensor.buffer != nl.sbuf,
        "Dynamic all-expert MoE TKG requires hidden input in HBM",
    )
    kernel_assert(
        output.buffer != nl.sbuf,
        "Dynamic all-expert MoE TKG requires output in HBM",
    )

    dims = MLPTKGConstants.calculate_constants(params)

    T = dims.T
    H = dims.H
    pmax = dims._pmax

    # A2AV setup: extract affinities from concatenated input and copy token indices to output
    a2av_enabled = dynamism_cfg.all_to_all_v_strategy != MoEAllToAllVStrategy.DISABLED
    output_indices_hbm = None
    if a2av_enabled:
        E_L = dims.E
        expert_affinities = _extract_a2av_affinities(params.hidden_tensor, H, E_L, T, pmax)
        if dims.E > 1 and dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.PACK_OUTPUT_ROWS:
            # E>1 PACK_OUTPUT_ROWS: build packed index mapping and copy token indices to packed positions
            output_indices_hbm = _build_pack_output_indices(
                expert_affinities,
                T,
                E_L,
                pmax,
                dynamism_cfg,
                params.hidden_tensor,
                output,
                H,
            )
        else:
            _copy_a2av_token_indices_to_output(
                params.hidden_tensor,
                output,
                H,
                E_L,
                T,
                dynamism_cfg.all_to_all_v_strategy,
            )

    # For A2AV, input row stride is H_concat (H + E_L + 2); otherwise it's H
    input_row_stride = params.hidden_tensor.shape[-1] if a2av_enabled else None

    use_auto_alloc = H >= 16 * 1024
    sbm = SbufManager(0, 200000, get_logger("all_expert_moe_tkg_dynamic"), use_auto_alloc=use_auto_alloc)
    sbm.open_scope()
    allocator = sbm.alloc_stack

    # Pre-allocate projection buffers outside dynamic loops to avoid NCC_IGCA108
    # (compiler requires tensors with loop-carried dependency to be defined outside of loop)
    block_T = dynamism_cfg.block_size
    num_I_tiles = div_ceil(dims.I, dims.I0)
    prealloc_gate_up_sb = nl.ndarray((dims.I0, num_I_tiles, block_T), dtype=nl.float32, buffer=nl.sbuf)
    prealloc_gate_up_sb_casted = nl.ndarray((dims.I0, num_I_tiles, block_T), dtype=io_dtype, buffer=nl.sbuf)
    # Internal gate/up projection accumulation buffers
    gate_up_proj_tile_shape = (dims.I0, num_I_tiles, block_T)
    prealloc_gate_sb_fp32 = nl.ndarray(gate_up_proj_tile_shape, dtype=nl.float32, buffer=nl.sbuf)
    prealloc_up_sb_fp32 = nl.ndarray(gate_up_proj_tile_shape, dtype=nl.float32, buffer=nl.sbuf)
    prealloc_gate_up_buffers = (prealloc_gate_sb_fp32, prealloc_up_sb_fp32)

    # Save original tensors before expert loop (params is mutated by _select_expert_params)
    gate_proj_weights = params.gate_proj_weights_tensor
    up_proj_weights = params.up_proj_weights_tensor
    down_proj_weights = params.down_proj_weights_tensor
    gate_proj_bias = params.bias_params.gate_proj_bias_tensor
    up_proj_bias = params.bias_params.up_proj_bias_tensor
    down_proj_bias = params.bias_params.down_proj_bias_tensor
    gate_w_scale = params.quant_params.gate_w_scale
    up_w_scale = params.quant_params.up_w_scale
    down_w_scale = params.quant_params.down_w_scale
    gate_up_in_scale = params.quant_params.gate_up_in_scale
    down_in_scale = params.quant_params.down_in_scale

    # Zero-initialize output buffer: each NC memsets its own [T, H_per_shard] slice
    # For PACK_OUTPUT_ROWS: ensures padding positions beyond routed tokens are deterministic
    # For PRESERVE_ROW_ORDER: ensures unrouted token positions are zero for correct combine-sum
    H_per_shard = dims.H_per_shard
    h_offset = dims.H1_offset * dims.H0
    zero_tile_T = min(pmax, T)
    zero_sb = nl.ndarray((zero_tile_T, H_per_shard), dtype=io_dtype, buffer=nl.sbuf)
    nisa.memset(zero_sb, 0)
    for t_start in range(0, T, zero_tile_T):
        tile_T_actual = min(zero_tile_T, T - t_start)
        nisa.dma_copy(
            src=zero_sb[:tile_T_actual, :H_per_shard],
            dst=output[nl.ds(t_start, tile_T_actual), nl.ds(h_offset, H_per_shard)],
        )

    # For PACK_OUTPUT_ROWS with E=1, track write position for sequential output.
    # For E>1, we use scatter (same as PRESERVE_ROW_ORDER) since packing is only valid for single expert.
    write_offset = None
    if dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.PACK_OUTPUT_ROWS and dims.E == 1:
        write_offset = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(write_offset, 0)

    # Process each expert
    for expertIdx in nl.sequential_range(dims.E):
        sbm.set_name_prefix(f"expert{expertIdx}_")

        # Select weights for this expert
        _select_expert_params(
            params,
            expertIdx,
            gate_proj_weights,
            up_proj_weights,
            down_proj_weights,
            gate_proj_bias,
            up_proj_bias,
            down_proj_bias,
            gate_w_scale,
            up_w_scale,
            down_w_scale,
            gate_up_in_scale,
            down_in_scale,
        )

        # Find routed tokens and build dynamic decision vector
        routed_token_indices_sb, dynamic_decision_sb = _find_routed_tokens(
            expert_affinities=expert_affinities,
            expert_idx=expertIdx,
            T=T,
            pmax=pmax,
            dynamism_cfg=dynamism_cfg,
        )

        # Process static blocks
        for static_block_idx in nl.sequential_range(dynamism_cfg.n_static_blocks):
            _compute_dynamic_block(
                params=params,
                dims=dims,
                dynamism_cfg=dynamism_cfg,
                io_dtype=io_dtype,
                routed_token_indices=routed_token_indices_sb,
                expert_affinities=expert_affinities,
                expert_idx=expertIdx,
                block_idx=static_block_idx,
                output_hbm=output,
                is_dynamic_block=False,
                input_row_stride=input_row_stride,
                write_offset=write_offset,
                is_first_expert=(expertIdx == 0),
                output_indices_hbm=output_indices_hbm,
            )

        # Move dynamic block indices to HBM for indirect access
        dynamic_block_token_indices_hbm = _init_dynamic_indices_hbm(
            dynamism_cfg=dynamism_cfg,
            routed_token_indices_sb=routed_token_indices_sb,
            dynamic_decision_sb=dynamic_decision_sb,
        )

        # Dynamic loop: sum decision vector to get total dynamic block count, then iterate
        n_dynamic_blocks_sb = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.tensor_reduce(dst=n_dynamic_blocks_sb, data=dynamic_decision_sb, op=nl.add, axis=1)
        n_dynamic_blocks_reg = nisa.register_alloc()
        nisa.register_load(src=n_dynamic_blocks_sb, dst=n_dynamic_blocks_reg)

        dynamic_block_idx = nl.ndarray((1, 1), dtype=nl.int32, buffer=nl.sbuf)
        nisa.memset(dynamic_block_idx, 0)

        def _dynamic_block_body(_):
            _compute_dynamic_block(
                params=params,
                dims=dims,
                dynamism_cfg=dynamism_cfg,
                io_dtype=io_dtype,
                routed_token_indices=dynamic_block_token_indices_hbm,
                expert_affinities=expert_affinities,
                expert_idx=expertIdx,
                block_idx=dynamic_block_idx,
                output_hbm=output,
                is_dynamic_block=True,
                input_row_stride=input_row_stride,
                write_offset=write_offset,
                is_first_expert=(expertIdx == 0),
                output_indices_hbm=output_indices_hbm,
                prealloc_gate_up_sb=prealloc_gate_up_sb,
                prealloc_gate_up_sb_casted=prealloc_gate_up_sb_casted,
                prealloc_gate_up_buffers=prealloc_gate_up_buffers,
            )

            # Advance to next dynamic block
            nisa.tensor_scalar(data=dynamic_block_idx, op0=nl.add, operand0=1, dst=dynamic_block_idx)

        nl.fori_loop(0, n_dynamic_blocks_reg, _dynamic_block_body)

    sbm.close_scope()
    return output


def _all_expert_moe_tkg_static(
    params: MLPParameters,
    output: nl.NkiTensor,
) -> nl.NkiTensor:
    """
    Static all-expert MoE computation without dynamic loop on chip (DLoC).

    Processes all experts sequentially, computing MLP projections for each expert
    and accumulating results weighted by expert affinities.

    Supports T > 128 via T-tiling: when T exceeds pmax (128), the token dimension
    is processed in tiles of pmax, with each tile going through all experts before
    storing results and moving to the next tile.

    Args:
        params (MLPParameters): MLPParameters containing model configuration, weights, and input tensors.
        output (nl.NkiTensor): Output tensor to store the final result.

    Returns:
        output (nl.NkiTensor): Output tensor with accumulated expert results.

    Notes:
        - Column tiling for down projection is not supported.

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

    if params.use_tkg_down_proj_column_tiling:
        kernel_assert(False, "Column tiling for down proj is not supported in all-expert MLP kernel")

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

    # Save original tensors before expert loop (params is mutated by _select_expert_params)
    hidden_tensor = params.hidden_tensor
    gate_proj_weights = params.gate_proj_weights_tensor
    up_proj_weights = params.up_proj_weights_tensor
    down_proj_weights = params.down_proj_weights_tensor

    gate_proj_bias = params.bias_params.gate_proj_bias_tensor
    up_proj_bias = params.bias_params.up_proj_bias_tensor
    down_proj_bias = params.bias_params.down_proj_bias_tensor

    gate_w_scale = params.quant_params.gate_w_scale
    up_w_scale = params.quant_params.up_w_scale
    down_w_scale = params.quant_params.down_w_scale
    gate_up_in_scale = params.quant_params.gate_up_in_scale
    down_in_scale = params.quant_params.down_in_scale

    # Allocate accumulation buffer [H0, H1_shard, tile_T * num_tiles]
    output_temp = allocator((dims.H0, dims.H1_shard, t_cfg.tile_T * t_cfg.num_tiles), dtype=io_dtype, buffer=nl.sbuf)
    output_in_sbuf = (output.buffer == nl.sbuf) if isinstance(output, nl.NkiTensor) else output.buffer == nl.sbuf

    # Load expert affinities for all T-tiles upfront
    pmax = dims._pmax
    expert_affinities_sb, aff_num_tiles = load_all_expert_affinities(
        expert_affinities, expert_affinities_in_sbuf, t_cfg.T_total, dims, allocator
    )

    memory_safe_degree = 2 if t_cfg.tile_T * dims.H * dims.I <= 32 * 3072 * 1024 else 1

    # Pre-load identity matrix (reused across all experts, only for T <= pmax)
    pmax = dims._pmax
    identity_sb = nl.shared_identity_matrix(t_cfg.tile_T, dtype=io_dtype) if t_cfg.tile_T <= pmax else None

    # Pre-load all input tiles (reused across all experts, avoids redundant DMA per expert)
    if hidden_in_sbuf:
        input_sb_tiles = None
    else:
        input_sb_tiles = []
        for t_tile in TiledRange(t_cfg.T_total, t_cfg.tile_T):
            dims.T = t_tile.size  # Temporarily set for input_norm_load
            isb = allocator(
                [dims.H0, t_tile.size, dims.H1_shard],
                dtype=io_dtype,
                buffer=nl.sbuf,
                name=f"input_sb_t{t_tile.index}",
            )
            input_norm_load(hidden_tensor, isb, params, dims, sbm=sbm, T_offset=t_tile.start_offset)
            input_sb_tiles.append(isb)

    # E-then-T: outer loop over experts (load weights once), inner loop over T-tiles
    for expertIdx in range(dims.E):
        sbm.set_name_prefix(f"expert{expertIdx}_")

        # Select weights, biases, and quant scales for this expert
        _select_expert_params(
            params,
            expertIdx,
            gate_proj_weights,
            up_proj_weights,
            down_proj_weights,
            gate_proj_bias,
            up_proj_bias,
            down_proj_bias,
            gate_w_scale,
            up_w_scale,
            down_w_scale,
            gate_up_in_scale,
            down_in_scale,
        )

        # Inner loop over T-tiles for this expert
        sbm.open_scope(interleave_degree=memory_safe_degree)
        for t_tile in TiledRange(t_cfg.T_total, t_cfg.tile_T):
            current_tile_T = t_tile.size
            t_offset = t_tile.start_offset
            t_idx = t_tile.index
            sbm.set_name_prefix(f"expert{expertIdx}_t{t_idx}_")

            dims.T = current_tile_T  # Temporarily set for sub-kernels

            # Use pre-loaded input for this T-tile
            if hidden_in_sbuf:
                input_sb = hidden_tensor
            else:
                input_sb = input_sb_tiles[t_idx]

            _compute_expert_mlp_tkg(
                input_sb=input_sb,
                params=params,
                dims=dims,
                sbm=sbm,
                allocator=allocator,
                expert_affinities_sb=expert_affinities_sb,
                aff_num_tiles=aff_num_tiles,
                identity_sb=identity_sb,
                use_auto_alloc=use_auto_alloc,
                io_dtype=io_dtype,
                output_temp=output_temp,
                expertIdx=expertIdx,
                t_offset=t_offset,
                t_idx=t_idx,
                t_cfg=t_cfg,
                current_tile_T=current_tile_T,
                hidden_in_sbuf=hidden_in_sbuf,
            )

            sbm.increment_section()
        sbm.close_scope()

    # Store: transpose [H0, H1_shard, T] to [T, H] and write to HBM
    # transpose_store requires T <= pmax, so tile in pmax-sized chunks
    store_tile_T = min(t_cfg.T_total, pmax)
    if output_in_sbuf:
        for t_tile in TiledRange(t_cfg.T_total, store_tile_T):
            t_free_offset = t_tile.start_offset  # offset into output_temp's free dim
            for h1 in range(dims.H1_shard):
                nisa.tensor_copy(
                    dst=output[:, nl.ds(t_tile.start_offset, t_tile.size), h1],
                    src=output_temp[:, h1, nl.ds(t_free_offset, t_tile.size)],
                )
    elif params.transposed_out:
        for t_tile in TiledRange(t_cfg.T_total, t_cfg.tile_T):
            t_free_offset = t_tile.index * t_cfg.tile_T
            tile_nc_size = dims.H1_shard * t_tile.size
            full_nc_size = dims.H1_shard * t_cfg.T_total
            nc_offset = dims.shard_id * full_nc_size + dims.H1_shard * t_tile.start_offset
            nisa.dma_copy(
                dst=output.reshape((dims.H0, dims.num_shards * full_nc_size))[:, nc_offset : nc_offset + tile_nc_size],
                src=output_temp[: dims.H0, : dims.H1_shard, nl.ds(t_free_offset, t_tile.size)].reshape(
                    (dims.H0, tile_nc_size)
                ),
            )
    else:
        for t_tile in TiledRange(t_cfg.T_total, store_tile_T):
            sbm.set_name_prefix(f"store_t{t_tile.index}_")
            t_free_offset = t_tile.start_offset
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


def _select_expert_params(
    params,
    expertIdx,
    gate_proj_weights,
    up_proj_weights,
    down_proj_weights,
    gate_proj_bias,
    up_proj_bias,
    down_proj_bias,
    gate_w_scale,
    up_w_scale,
    down_w_scale,
    gate_up_in_scale,
    down_in_scale,
):
    """Select weights, biases, and quant scales for a single expert, mutating params in-place."""
    expert_gate_w = gate_proj_weights.select(dim=0, index=expertIdx)
    expert_up_w = up_proj_weights.select(dim=0, index=expertIdx)
    expert_down_w = down_proj_weights.select(dim=0, index=expertIdx)

    if len(expert_gate_w.shape) > 2:
        expert_gate_w = expert_gate_w.select(dim=1, index=GateUpDim.GATE.value)
        expert_up_w = expert_up_w.select(dim=1, index=GateUpDim.UP.value)

    expert_gate_b, expert_up_b, expert_down_b = None, None, None
    if gate_proj_bias != None:
        expert_gate_b = gate_proj_bias.select(dim=0, index=expertIdx)
        if len(expert_gate_b.shape) > 1:
            expert_gate_b = expert_gate_b.select(dim=0, index=GateUpDim.GATE.value)
    if up_proj_bias != None:
        expert_up_b = up_proj_bias.select(dim=0, index=expertIdx)
        if len(expert_up_b.shape) > 1:
            expert_up_b = expert_up_b.select(dim=0, index=GateUpDim.UP.value)
    if down_proj_bias != None:
        expert_down_b = down_proj_bias.select(dim=0, index=expertIdx)

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
            gate_w_scale,
            up_w_scale,
            down_w_scale,
            gate_up_in_scale,
            down_in_scale,
            expertIdx,
        )


def _find_routed_tokens(expert_affinities, expert_idx, T, pmax, dynamism_cfg):
    """
    Find indices of tokens routed to a specific expert and build dynamic block decision vector.

    Args:
        expert_affinities (nl.NkiTensor): [T, E] expert affinities in HBM.
        expert_idx: Index of the expert.
        T: Total number of tokens.
        pmax: Partition max size.
        dynamism_cfg (AllExpertDynamismConfig): Dynamism parameters.

    Returns:
        routed_token_indices_sb (nl.NkiTensor): [pmax, T+1] in SBUF. Partition 0 has routed indices + count.
        dynamic_decision_sb (nl.NkiTensor): [1, n_dynamic_blocks+1] in SBUF. Decision vector.
    """
    # Load expert affinities for this expert: transpose [T, E] -> [1, T]
    expert_aff_f32_sb = nl.ndarray((pmax, T), dtype=nl.float32, buffer=nl.sbuf)
    needs_cast = expert_affinities.dtype != nl.float32
    if needs_cast:
        expert_aff_sb = nl.ndarray((1, T), dtype=expert_affinities.dtype, buffer=nl.sbuf)
        load_dst = expert_aff_sb
    else:
        load_dst = expert_aff_f32_sb

    src_view = expert_affinities.select(dim=1, index=expert_idx).expand_dim(dim=1).expand_dim(dim=1).expand_dim(dim=1)
    dst_view = load_dst.expand_dim(dim=1).expand_dim(dim=1)
    nisa.dma_transpose(src=src_view, dst=dst_view)
    if needs_cast:
        nisa.tensor_copy(src=expert_aff_sb[...], dst=expert_aff_f32_sb[0, :])

    # nonzero_with_count
    routed_token_indices_sb = nl.ndarray((pmax, dynamism_cfg.T_plus_1), dtype=nl.int32, buffer=nl.sbuf)
    nisa.nonzero_with_count(
        src=expert_aff_f32_sb[...],
        padding_val=NONZERO_WITH_COUNT_PAD_VAL,
        dst=routed_token_indices_sb[...],
    )

    # Build dynamic decision vector: iota=[block_size*n_static, ...], less(iota, count) -> [1, 0, ...]
    dynamic_decision_sb = nl.ndarray((1, dynamism_cfg.n_dynamic_blocks_plus_1), dtype=nl.int32, buffer=nl.sbuf)
    count_f32 = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(src=routed_token_indices_sb[0, T], dst=count_f32)
    nisa.iota(
        dst=dynamic_decision_sb[...],
        pattern=[[dynamism_cfg.block_size, dynamism_cfg.n_dynamic_blocks_plus_1]],
        offset=dynamism_cfg.n_static_blocks * dynamism_cfg.block_size,
    )
    nisa.tensor_scalar(
        data=dynamic_decision_sb[...],
        op0=nl.less,
        operand0=count_f32,
        dst=dynamic_decision_sb[...],
    )

    return routed_token_indices_sb, dynamic_decision_sb


def _init_dynamic_indices_hbm(dynamism_cfg, routed_token_indices_sb, dynamic_decision_sb):
    """
    Move routed token indices for dynamic blocks to HBM.

    Returns:
        dynamic_block_token_indices_hbm (nl.NkiTensor): [n_dynamic_blocks, block_size] in private HBM.
    """
    n_static_tokens = dynamism_cfg.n_static_blocks * dynamism_cfg.block_size
    n_dynamic_tokens = dynamism_cfg.n_dynamic_blocks * dynamism_cfg.block_size

    dynamic_block_token_indices_hbm = nl.ndarray(
        (1, n_dynamic_tokens),
        dtype=routed_token_indices_sb.dtype,
        buffer=nl.private_hbm,
    )

    nisa.dma_copy(
        src=routed_token_indices_sb[0, nl.ds(n_static_tokens, n_dynamic_tokens)],
        dst=dynamic_block_token_indices_hbm[0, :],
    )
    dynamic_block_token_indices_hbm = dynamic_block_token_indices_hbm.reshape(
        (dynamism_cfg.n_dynamic_blocks, dynamism_cfg.block_size)
    )

    return dynamic_block_token_indices_hbm


def _get_block_token_indices(dynamism_cfg, routed_token_indices, block_idx, is_dynamic_block):
    """
    Get token indices for a block and transpose them for indirect DMA.

    Args:
        dynamism_cfg (AllExpertDynamismConfig): Dynamism parameters.
        routed_token_indices: Static: [pmax, T+1] SBUF. Dynamic: [n_dynamic_blocks, block_size] HBM.
        block_idx: Static: int literal. Dynamic: [1, 1] SBUF tensor.
        is_dynamic_block (bool): Whether this is a dynamic block.

    Returns:
        token_indices_T_sb (nl.NkiTensor): [blk_tile_T, blk_n_T_tiles] transposed indices in SBUF (int32).
    """
    # Load indices to SBUF
    if is_dynamic_block:
        token_indices_sb = nl.ndarray((1, dynamism_cfg.block_size), dtype=nl.int32, buffer=nl.sbuf)
        nisa.dma_copy(
            src=routed_token_indices.ap(
                pattern=[[dynamism_cfg.block_size, 1], [1, dynamism_cfg.block_size]],
                offset=0,
                scalar_offset=block_idx,
                indirect_dim=0,
            ),
            dst=token_indices_sb,
        )
    else:
        token_indices_sb = routed_token_indices[0, nl.ds(dynamism_cfg.block_size * block_idx, dynamism_cfg.block_size)]

    # Transpose indices: [1, block_size] -> [blk_tile_T, blk_n_T_tiles]
    # Reinterpret int32 as float32 since PE transpose doesn't support int32
    token_indices_sb = token_indices_sb.view(nl.float32)
    token_indices_T_psum = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles),
        dtype=nl.float32,
        buffer=nl.psum,
    )
    token_indices_T_sb = nl.ndarray(
        (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles),
        dtype=nl.float32,
        buffer=nl.sbuf,
    )
    for tile_out in range(dynamism_cfg.blk_n_T_tiles):
        nisa.nc_transpose(
            data=token_indices_sb[0, nl.ds(dynamism_cfg.blk_tile_T * tile_out, dynamism_cfg.blk_tile_T)],
            dst=token_indices_T_psum[:, tile_out],
        )
    nisa.tensor_copy(src=token_indices_T_psum[...], dst=token_indices_T_sb[...])
    token_indices_T_sb = token_indices_T_sb.view(nl.int32)

    return token_indices_T_sb


def _gather_input_block(hidden_input, token_indices_T_sb, dims, dynamism_cfg, io_dtype, sbm, input_row_stride=None):
    """
    Gather input hidden states for a block of tokens via indirect DMA.

    Loads from [T, H] or [T, H_concat] HBM into [H0, block_T, H1_shard] SBUF layout using
    indirect DMA gather + on-chip transpose.

    Args:
        hidden_input (nl.NkiTensor): [T, H] or [T, H_concat] input in HBM.
        token_indices_T_sb (nl.NkiTensor): [blk_tile_T, blk_n_T_tiles] transposed token indices.
        dims: MLPTKGConstantsDimensionSizes.
        dynamism_cfg (AllExpertDynamismConfig): Dynamism parameters.
        io_dtype: Data type.
        sbm (SbufManager): SBUF manager.
        input_row_stride (int): Row stride of hidden_input. Defaults to dims.H.

    Returns:
        input_sb (nl.NkiTensor): [H0, block_T, H1_shard] in SBUF.
    """
    block_T = dynamism_cfg.block_size
    H0 = dims.H0  # pmax = 128
    H1_shard = dims.H1_shard
    H_per_shard = dims.H_per_shard
    shard_id = dims.shard_id
    h_offset = dims.H1_offset * H0
    row_stride = input_row_stride if input_row_stride is not None else dims.H

    input_sb = nl.ndarray((H0, block_T, H1_shard), dtype=io_dtype, buffer=nl.sbuf)
    # Zero-init ensures unrouted token positions (skipped by oob_mode.skip) produce
    # deterministic zeros through the MLP, avoiding non-determinism from uninitialized SBUF.
    nisa.memset(input_sb, 0)

    # Indirect DMA: gather [block_T, H_per_shard] from [T, H] HBM
    # Then transpose on-chip to [H0, block_T, H1_shard]
    for tile_t in range(dynamism_cfg.blk_n_T_tiles):
        tile_T_actual = min(dynamism_cfg.blk_tile_T, block_T - tile_t * dynamism_cfg.blk_tile_T)
        # Load [blk_tile_T, H_per_shard] via indirect DMA
        input_tile_sb = nl.ndarray((tile_T_actual, H_per_shard), dtype=io_dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            src=hidden_input.ap(
                pattern=[[row_stride, dynamism_cfg.blk_tile_T], [1, H_per_shard]],
                offset=h_offset,
                vector_offset=token_indices_T_sb.ap(
                    pattern=[[dynamism_cfg.blk_n_T_tiles, dynamism_cfg.blk_tile_T], [1, 1]],
                    offset=tile_t,
                ),
                indirect_dim=0,
            ),
            dst=input_tile_sb[:tile_T_actual, :H_per_shard],
            oob_mode=oob_mode.skip,
            dge_mode=nisa.dge_mode.swdge,
        )
        # Transpose [tile_T_actual, H_per_shard] -> [H0, tile_T_actual, H1_shard]
        # HBM layout: hidden[t, h0*H1_shard + h1] (H0 outer, H1_shard inner)
        # Reshape to [T, H0, H1_shard], then transpose [T, H0] -> [H0, T] per H1 slice
        input_tile_reshaped = input_tile_sb.reshape((tile_T_actual, H0, H1_shard))
        for h1_idx in range(H1_shard):
            tp_psum = nl.ndarray((H0, tile_T_actual), dtype=io_dtype, buffer=nl.psum)
            nisa.nc_transpose(
                data=input_tile_reshaped[:tile_T_actual, :H0, h1_idx],
                dst=tp_psum[:H0, :tile_T_actual],
            )
            t_offset = tile_t * dynamism_cfg.blk_tile_T
            nisa.tensor_copy(
                dst=input_sb[:H0, nl.ds(t_offset, tile_T_actual), h1_idx],
                src=tp_psum[:H0, :tile_T_actual],
            )

    return input_sb


def _scatter_output_block(
    output_sb,
    output_hbm,
    token_indices_T_sb,
    dims,
    dynamism_cfg,
    io_dtype,
    output_row_stride=None,
    is_first_expert=True,
):
    """
    Transpose block output from [H0, H1_shard, block_T] to [block_T, H] and scatter to HBM.

    Args:
        output_sb (nl.NkiTensor): [H0, H1_shard, block_T] expert MLP output in SBUF.
        output_hbm (nl.NkiTensor): [T, H] or [T, H+2] output in HBM.
        token_indices_T_sb (nl.NkiTensor): [blk_tile_T, blk_n_T_tiles] transposed token indices.
        dims: MLPTKGConstantsDimensionSizes.
        dynamism_cfg (AllExpertDynamismConfig): Dynamism parameters.
        io_dtype: Data type.
        output_row_stride (int): Row stride of output_hbm. Defaults to dims.H.
        is_first_expert (bool): If True, overwrite output. If False, accumulate via atomic add.
    """
    block_T = dynamism_cfg.block_size
    H0 = dims.H0
    H1_shard = dims.H1_shard
    H_per_shard = dims.H_per_shard
    shard_id = dims.shard_id
    h_offset = dims.H1_offset * H0
    row_stride = output_row_stride if output_row_stride is not None else dims.H

    # Transpose [H0, H1_shard, block_T] -> [block_T, H_per_shard] in SBUF
    out_transposed = nl.ndarray((block_T, H_per_shard), dtype=io_dtype, buffer=nl.sbuf)
    out_transposed_reshaped = out_transposed.reshape((block_T, H0, H1_shard))
    for h1_idx in range(H1_shard):
        tp_psum = nl.ndarray((block_T, H0), dtype=io_dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=tp_psum[:block_T, :H0], data=output_sb[:H0, h1_idx, :block_T])
        nisa.tensor_copy(
            dst=out_transposed_reshaped[:block_T, :H0, h1_idx],
            src=tp_psum[:block_T, :H0],
        )

    # Indirect scatter to HBM
    for tile_t in range(dynamism_cfg.blk_n_T_tiles):
        tile_T_actual = min(dynamism_cfg.blk_tile_T, block_T - tile_t * dynamism_cfg.blk_tile_T)
        dst_ap = output_hbm.ap(
            pattern=[[row_stride, dynamism_cfg.blk_tile_T], [1, H_per_shard]],
            offset=h_offset,
            vector_offset=token_indices_T_sb.ap(
                pattern=[[dynamism_cfg.blk_n_T_tiles, dynamism_cfg.blk_tile_T], [1, 1]],
                offset=tile_t,
            ),
            indirect_dim=0,
        )
        src_tile = out_transposed[nl.ds(tile_t * dynamism_cfg.blk_tile_T, tile_T_actual), :H_per_shard]
        if is_first_expert:
            nisa.dma_copy(
                src=src_tile,
                dst=dst_ap,
                oob_mode=oob_mode.skip,
                dge_mode=nisa.dge_mode.swdge,
            )
        else:
            nisa.dma_compute(
                dst=dst_ap,
                srcs=[dst_ap, src_tile],
                scales=[1.0, 1.0],
                reduce_op=nl.add,
                oob_mode=oob_mode.skip,
            )


def _sequential_store_block(
    output_sb,
    output_hbm,
    token_indices_T_sb,
    hidden_input,
    write_offset,
    dims,
    dynamism_cfg,
    io_dtype,
    output_row_stride,
):
    """
    Sequential store for PACK_OUTPUT_ROWS: write block output to consecutive rows at write_offset.

    Also copies token indices from input to output at the packed position.

    Args:
        output_sb (nl.NkiTensor): [H0, H1_shard, block_T] expert MLP output in SBUF.
        output_hbm (nl.NkiTensor): [T, H+2] output in HBM.
        token_indices_T_sb (nl.NkiTensor): [blk_tile_T, blk_n_T_tiles] transposed token indices (original positions).
        hidden_input (nl.NkiTensor): [T, H_concat] input in HBM (for token index extraction).
        write_offset (nl.NkiTensor): [1, 1] SBUF tensor tracking current write position.
        dims: MLPTKGConstantsDimensionSizes.
        dynamism_cfg (AllExpertDynamismConfig): Dynamism parameters.
        io_dtype: Data type.
        output_row_stride (int): Row stride of output (H + 2).
    """
    block_T = dynamism_cfg.block_size
    H0 = dims.H0
    H1_shard = dims.H1_shard
    H_per_shard = dims.H_per_shard
    shard_id = dims.shard_id
    h_offset = dims.H1_offset * H0
    H = dims.H
    E_L = dims.E
    input_row_stride = hidden_input.shape[-1]

    # Transpose [H0, H1_shard, block_T] -> [block_T, H_per_shard]
    out_transposed = nl.ndarray((block_T, H_per_shard), dtype=io_dtype, buffer=nl.sbuf)
    out_transposed_reshaped = out_transposed.reshape((block_T, H0, H1_shard))
    for h1_idx in range(H1_shard):
        tp_psum = nl.ndarray((block_T, H0), dtype=io_dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=tp_psum[:block_T, :H0], data=output_sb[:H0, h1_idx, :block_T])
        nisa.tensor_copy(
            dst=out_transposed_reshaped[:block_T, :H0, h1_idx],
            src=tp_psum[:block_T, :H0],
        )

    # Sequential write to output at write_offset (indirect on row dimension)
    for tile_t in range(dynamism_cfg.blk_n_T_tiles):
        tile_T_actual = min(dynamism_cfg.blk_tile_T, block_T - tile_t * dynamism_cfg.blk_tile_T)
        # Write hidden output
        nisa.dma_copy(
            src=out_transposed[nl.ds(tile_t * dynamism_cfg.blk_tile_T, tile_T_actual), :H_per_shard],
            dst=output_hbm.ap(
                pattern=[[output_row_stride, dynamism_cfg.blk_tile_T], [1, H_per_shard]],
                offset=h_offset,
                scalar_offset=write_offset,
                indirect_dim=0,
            ),
            dge_mode=nisa.dge_mode.swdge,
        )

        # Copy token indices from input to output (only NC0 writes the 2 trailing columns)
        if shard_id == 0:
            token_idx_sb = nl.ndarray((dynamism_cfg.blk_tile_T, BF16_PER_INT32), dtype=io_dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                src=hidden_input.ap(
                    pattern=[[input_row_stride, dynamism_cfg.blk_tile_T], [1, BF16_PER_INT32]],
                    offset=H + E_L,
                    vector_offset=token_indices_T_sb.ap(
                        pattern=[[dynamism_cfg.blk_n_T_tiles, dynamism_cfg.blk_tile_T], [1, 1]],
                        offset=tile_t,
                    ),
                    indirect_dim=0,
                ),
                dst=token_idx_sb[:tile_T_actual, :BF16_PER_INT32],
                oob_mode=oob_mode.skip,
                dge_mode=nisa.dge_mode.swdge,
            )
            nisa.dma_copy(
                src=token_idx_sb[:tile_T_actual, :BF16_PER_INT32],
                dst=output_hbm.ap(
                    pattern=[[output_row_stride, dynamism_cfg.blk_tile_T], [1, BF16_PER_INT32]],
                    offset=H,
                    scalar_offset=write_offset,
                    indirect_dim=0,
                ),
                dge_mode=nisa.dge_mode.swdge,
            )

        # Increment write offset
        nisa.tensor_scalar(data=write_offset, op0=nl.add, operand0=tile_T_actual, dst=write_offset)


def _compute_dynamic_block(
    params,
    dims,
    dynamism_cfg,
    io_dtype,
    routed_token_indices,
    expert_affinities,
    expert_idx,
    block_idx,
    output_hbm,
    is_dynamic_block,
    input_row_stride=None,
    write_offset=None,
    is_first_expert=True,
    output_indices_hbm=None,
    prealloc_gate_up_sb=None,
    prealloc_gate_up_sb_casted=None,
    prealloc_gate_up_buffers=None,
):
    """
    Compute expert MLP for a single block of routed tokens.

    Gathers input, computes gate_up -> down -> affinity scale, and scatters output.

    Args:
        params (MLPParameters): MLP parameters with expert weights already selected.
        dims: MLPTKGConstantsDimensionSizes.
        dynamism_cfg (AllExpertDynamismConfig): Dynamism parameters.
        sbm (SbufManager): SBUF manager.
        allocator: SBUF allocator callable.
        io_dtype: I/O data type.
        routed_token_indices: Static: [pmax, T+1] SBUF. Dynamic: [n_dynamic_blocks, block_size] HBM.
        expert_affinities (nl.NkiTensor): [T, E] expert affinities.
        expert_idx: Current expert index.
        block_idx: Block index (int literal for static, [1,1] SBUF tensor for dynamic).
        output_hbm (nl.NkiTensor): [T, H] output in HBM.
        is_dynamic_block (bool): Whether this is a dynamic block.
        is_first_expert (bool): Whether this is the first expert (overwrite vs accumulate output).
    """
    block_T = dynamism_cfg.block_size
    pmax = dims._pmax

    block_sbm = SbufManager(0, 200000, get_logger("dynamic_block"), use_auto_alloc=True)
    block_sbm.open_scope()
    if is_dynamic_block:
        block_sbm.set_name_prefix(f"e{expert_idx}_dyn_blk_")
    else:
        block_sbm.set_name_prefix(f"e{expert_idx}_static_blk{block_idx}_")

    token_indices_T_sb = _get_block_token_indices(
        dynamism_cfg,
        routed_token_indices,
        block_idx,
        is_dynamic_block,
    )
    input_sb = _gather_input_block(
        params.hidden_tensor,
        token_indices_T_sb,
        dims,
        dynamism_cfg,
        io_dtype,
        None,
        input_row_stride=input_row_stride,
    )

    orig_T = dims.T
    dims.T = block_T

    # Gate Up projection
    if prealloc_gate_up_sb is not None:
        gate_up_sb = prealloc_gate_up_sb
        nisa.memset(gate_up_sb, 0)
    else:
        gate_up_sb = nl.ndarray(
            (dims.I0, div_ceil(dims.I, dims.I0), block_T),
            dtype=nl.float32,
            buffer=nl.sbuf,
        )
    gate_tile_info = process_gate_up_projection(
        hidden=input_sb,
        output=gate_up_sb,
        params=params,
        dims=dims,
        sbm=block_sbm,
        T_offset=0,
        share_memory_scope=True,
        use_dge=is_dynamic_block,
        prealloc_gate_up_buffers=prealloc_gate_up_buffers if is_dynamic_block else None,
    )

    # Load and broadcast expert affinity for this block
    if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
        block_aff_sb = nl.ndarray(
            (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles, 1), dtype=expert_affinities.dtype, buffer=nl.sbuf
        )
        # Zero-init ensures padded positions (oob_mode.skip) have zero affinity,
        # so their MLP output is zeroed out before scatter (safe even if dma_compute doesn't skip).
        nisa.memset(block_aff_sb, 0)
        for tile_t in range(dynamism_cfg.blk_n_T_tiles):
            nisa.dma_copy(
                src=expert_affinities.ap(
                    pattern=[[dims.E, dynamism_cfg.blk_tile_T], [1, 1], [1, 1]],
                    offset=expert_idx,
                    vector_offset=token_indices_T_sb.ap(
                        pattern=[[dynamism_cfg.blk_n_T_tiles, dynamism_cfg.blk_tile_T], [1, 1]],
                        offset=tile_t,
                    ),
                    indirect_dim=0,
                ),
                dst=block_aff_sb[:, tile_t, 0],
                oob_mode=oob_mode.skip,
            )
        block_aff_flat = block_aff_sb.reshape((block_T, 1))
        identity_sb = nl.shared_identity_matrix(block_T, dtype=io_dtype)
        block_aff_cast = nl.ndarray((block_T, 1), dtype=io_dtype, buffer=nl.sbuf)
        nisa.activation(dst=block_aff_cast[:block_T, :], op=nl.copy, data=block_aff_flat[:block_T, :])
        aff_psum = nl.ndarray((1, block_T), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=aff_psum[:, :], stationary=block_aff_cast, moving=identity_sb[:block_T, :block_T])
        expert_affinities_broadcast = nl.ndarray((pmax, block_T), dtype=io_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(src=aff_psum[:, :], dst=expert_affinities_broadcast[:1, :block_T])
        for pg in range(4):
            nisa.nc_stream_shuffle(
                dst=expert_affinities_broadcast[nl.ds(32 * pg, 32), :block_T],
                src=expert_affinities_broadcast[:1, :block_T],
                shuffle_mask=[0] * 32,
            )

    # Down projection
    down_sb = nl.ndarray(
        (dims.H0, dims.H1_shard, block_T),
        dtype=io_dtype,
        buffer=nl.sbuf,
    )
    if prealloc_gate_up_sb_casted is not None:
        gate_up_sb_casted = prealloc_gate_up_sb_casted
    else:
        gate_up_sb_casted = nl.ndarray(
            (dims.I0, div_ceil(dims.I, dims.I0), block_T),
            dtype=io_dtype,
            buffer=nl.sbuf,
        )
    nisa.tensor_copy(dst=gate_up_sb_casted, src=gate_up_sb)
    process_down_projection(
        hidden=gate_up_sb_casted,
        output=down_sb,
        params=params,
        dims=dims,
        gate_tile_info=gate_tile_info,
        sbm=block_sbm,
        use_dge=is_dynamic_block,
    )

    # Apply affinity scaling
    if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
        for shard in range(dims.H1_shard):
            nisa.tensor_tensor(
                data1=down_sb[: dims.H0, shard, :block_T],
                data2=expert_affinities_broadcast,
                op=nl.multiply,
                dst=down_sb[: dims.H0, shard, :block_T],
            )

    dims.T = orig_T

    # Scatter output to HBM
    # PACK_OUTPUT_ROWS sequential store only valid for E=1 (single local expert after EP).
    # For E>1, scatter to original positions with accumulation (same as PRESERVE_ROW_ORDER).
    use_sequential_store = (
        dynamism_cfg.all_to_all_v_strategy == MoEAllToAllVStrategy.PACK_OUTPUT_ROWS
        and is_first_expert
        and write_offset is not None
    )
    if use_sequential_store:
        # E=1 or first expert: sequential store packs output + token indices
        output_row_stride = dims.H + BF16_PER_INT32
        _sequential_store_block(
            output_sb=down_sb,
            output_hbm=output_hbm,
            token_indices_T_sb=token_indices_T_sb,
            hidden_input=params.hidden_tensor,
            write_offset=write_offset,
            dims=dims,
            dynamism_cfg=dynamism_cfg,
            io_dtype=io_dtype,
            output_row_stride=output_row_stride,
        )
    else:
        output_row_stride = (
            (dims.H + BF16_PER_INT32) if dynamism_cfg.all_to_all_v_strategy != MoEAllToAllVStrategy.DISABLED else None
        )
        # For PACK_OUTPUT_ROWS E>1: remap token indices to packed output positions
        scatter_indices = token_indices_T_sb
        if output_indices_hbm is not None:
            scatter_indices = nl.ndarray(
                (dynamism_cfg.blk_tile_T, dynamism_cfg.blk_n_T_tiles), dtype=nl.int32, buffer=nl.sbuf
            )
            nisa.memset(scatter_indices, NONZERO_WITH_COUNT_PAD_VAL)
            for tile_t in range(dynamism_cfg.blk_n_T_tiles):
                nisa.dma_copy(
                    src=output_indices_hbm.ap(
                        pattern=[[1, dynamism_cfg.blk_tile_T], [1, 1]],
                        offset=0,
                        vector_offset=token_indices_T_sb.ap(
                            pattern=[[dynamism_cfg.blk_n_T_tiles, dynamism_cfg.blk_tile_T], [1, 1]],
                            offset=tile_t,
                        ),
                        indirect_dim=0,
                    ),
                    dst=scatter_indices[:, tile_t : tile_t + 1],
                    oob_mode=oob_mode.skip,
                    dge_mode=nisa.dge_mode.swdge,
                )
        _scatter_output_block(
            down_sb,
            output_hbm,
            scatter_indices,
            dims,
            dynamism_cfg,
            io_dtype,
            output_row_stride=output_row_stride,
            is_first_expert=is_first_expert,
        )


def _compute_expert_mlp_tkg(
    input_sb,
    params,
    dims,
    sbm,
    allocator,
    expert_affinities_sb,
    aff_num_tiles,
    identity_sb,
    use_auto_alloc,
    io_dtype,
    output_temp,
    expertIdx,
    t_offset,
    t_idx,
    t_cfg,
    current_tile_T,
    hidden_in_sbuf,
):
    """
    Core expert MLP computation for one (expert, T-tile) pair.

    Computes gate/up projection, down projection, affinity scaling, and accumulation
    into the output buffer. This is the reusable computation unit shared by both
    static and (future) dynamic paths.

    Args:
        input_sb: Input hidden states for this T-tile (nl.NkiTensor in SBUF).
        params (MLPParameters): MLP parameters with expert weights already selected.
        dims: MLPTKGConstantsDimensionSizes with T set to current_tile_T.
        sbm (SbufManager): SBUF memory manager.
        allocator: SBUF allocator callable.
        expert_affinities_sb: Expert affinities in SBUF.
        aff_num_tiles: Number of affinity tiles.
        identity_sb: Pre-loaded identity matrix (or None).
        use_auto_alloc: Whether auto allocation is enabled.
        io_dtype: I/O data type.
        output_temp: Accumulation buffer [H0, H1_shard, tile_T * num_tiles] in SBUF.
        expertIdx: Current expert index.
        t_offset: Token offset for this T-tile.
        t_idx: T-tile index.
        t_cfg (TokenTilingConfig): Token tiling configuration.
        current_tile_T: Number of tokens in this tile.
        hidden_in_sbuf: Whether hidden input is in SBUF.
    """
    # Get expert affinity for this T-tile
    expertAffinityLoc = get_all_expert_tile_affinities(
        expert_affinities_sb, aff_num_tiles, t_offset, current_tile_T, dims, allocator
    )

    # Allocate buffers for this tile
    gate_up_sb = allocator(
        (dims.I0, div_ceil(dims.I, dims.I0), current_tile_T),
        dtype=nl.float32,
        name="gate_up_sbuf",
        buffer=nl.sbuf,
    )
    down_sb = allocator((dims.H0, dims.H1_shard, current_tile_T), dtype=io_dtype, name="down_sbuf", buffer=nl.sbuf)

    # Gate Up projection
    gate_tile_info = process_gate_up_projection(
        hidden=input_sb,
        output=gate_up_sb,
        params=params,
        dims=dims,
        sbm=sbm,
        T_offset=t_offset if hidden_in_sbuf else 0,
        share_memory_scope=not params.use_tkg_gate_up_proj_column_tiling,
    )

    # Compute POST_SCALE affinity broadcast
    if params.expert_params.expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
        expert_affinities_broadcast = broadcast_all_expert_affinity(
            expert_affinities_sb,
            aff_num_tiles,
            expertIdx,
            t_offset,
            current_tile_T,
            io_dtype,
            identity_sb,
            dims,
            allocator,
            use_auto_alloc,
        )

    # Down projection
    gate_up_sb_casted = allocator(
        (dims.I0, div_ceil(dims.I, dims.I0), current_tile_T),
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


def _select_quant_scales(
    quant_params: MLPQuantizationParameters,
    gate_w_scale: nl.NkiTensor,
    up_w_scale: nl.NkiTensor,
    down_w_scale: nl.NkiTensor,
    gate_up_in_scale: nl.NkiTensor,
    down_in_scale: nl.NkiTensor,
    expertIdx: int,
):
    """
    Select and reshape quantization scales for a specific expert.

    Args:
        quant_params (MLPQuantizationParameters): Quantization parameters.
        gate_w_scale (nl.NkiTensor): Gate weight scale tensor.
        up_w_scale (nl.NkiTensor): Up weight scale tensor.
        down_w_scale (nl.NkiTensor): Down weight scale tensor.
        gate_up_in_scale (nl.NkiTensor): Gate/up input scale tensor.
        down_in_scale (nl.NkiTensor): Down input scale tensor.
        expertIdx (int): Expert index to select scales for.

    Returns:
        MLPQuantizationParameters: Quantization parameters with scales for the specified expert.
    """
    quantization_type = quant_params.quantization_type
    expert_gate_w_scale = None
    if gate_w_scale != None:
        expert_gate_w_scale = gate_w_scale.select(dim=0, index=expertIdx).select(dim=0, index=GateUpDim.GATE.value)
        expert_gate_w_scale = reshape_scale_for_mlp(expert_gate_w_scale)

    expert_up_w_scale = None
    if up_w_scale != None:
        expert_up_w_scale = up_w_scale.select(dim=0, index=expertIdx).select(dim=0, index=GateUpDim.UP.value)
        expert_up_w_scale = reshape_scale_for_mlp(expert_up_w_scale)

    expert_down_w_scale = None
    if down_w_scale != None:
        expert_down_w_scale = down_w_scale.select(dim=0, index=expertIdx)
        expert_down_w_scale = reshape_scale_for_mlp(expert_down_w_scale)

    expert_gate_up_in_scale = None
    if gate_up_in_scale != None:
        expert_gate_up_in_scale = reshape_scale_for_mlp(gate_up_in_scale.select(dim=0, index=expertIdx))

    expert_down_in_scale = None
    if down_in_scale != None:
        expert_down_in_scale = reshape_scale_for_mlp(down_in_scale.select(dim=0, index=expertIdx))

    return MLPQuantizationParameters(
        quantization_type=quantization_type,
        gate_w_scale=expert_gate_w_scale,
        up_w_scale=expert_up_w_scale,
        down_w_scale=expert_down_w_scale,
        gate_up_in_scale=expert_gate_up_in_scale,
        down_in_scale=expert_down_in_scale,
        clipping_bound=quant_params.clipping_bound if quant_params != None else None,
    )
