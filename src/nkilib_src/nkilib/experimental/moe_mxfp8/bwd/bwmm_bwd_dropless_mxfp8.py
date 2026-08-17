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

"""MXFP8 backward pass implementation for blockwise MoE matrix multiplication.

This module implements the core backward pass logic for Mixture of Experts using
MXFP8 quantized matrix multiplication. It combines:
- Block-loop structure from BF16 MOE BWD (per-expert iteration, indirect token indexing)
- MXFP8 matmul infrastructure (TensorDescriptor, generic_matmul_mxfp8_api, spill/reload)
- A single-expert path for inputs already packed in contiguous token order

Architecture:
    For each block b (expert e = block_to_expert[b]):
        Phase 1: d_intermediate = output_grad[block] @ W_down[e].T; SwiGLU bwd → d_gate, d_up
        Phase 2: hidden_states_grad[block] += d_gate_up @ W_gate_up[e] (scatter via token indices)
        Phase 3: dW_gate_up[e] += d_gate_up.T @ hidden_states[block] (accumulate)
        Phase 4: dW_down[e] += output_grad[block].T @ intermediate[block] (accumulate)
"""

import nki  # noqa: F401
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode

from ....core.moe.moe_cte.bwmm_shard_on_I import DimensionSizes
from ....core.utils.kernel_helpers import div_ceil, get_program_sharding_info
from ....core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ...matmul_mxfp8.matmul_mxfp8_generic_api import generic_matmul_mxfp8_api
from ...mlp_mxfp8.common_utils import (
    MATMUL_TILE_K_PHYSICAL,
    TILE_M,
    TILE_N,
    _allocate_spill_buffer,
    _build_matmul_params,
    _compute_load_tile_shape,
    apply_gradient_clamp,
    build_tile_sizes,
)
from ...moe.bwd.bwmm_bwd_dropless import (
    _compute_down_proj_bias_grad,
    _compute_gate_up_proj_bias_grad,
    _generate_dynamic_offsets,
    _initialize_gradient_outputs_shard,
)
from ...moe.bwd.moe_bwd_parameters import AffinityOption, ClampLimits
from ...mxfp_utils.mxfp8_utils.common_dataclasses import QuantScheme, SwizzleMode, TensorDescriptor
from ...mxfp_utils.mxfp8_utils.common_utils import create_and_set_active_sbm, get_active_sbm, with_active_sbm
from .config import MXFP8MOEBwdConfig, TransposeMode


def _valid_tiles_in_block(total_tiles, block_idx, tiles_per_block):
    """Return a full block's tile count, trimmed for the final partial block."""
    return min(tiles_per_block, total_tiles - block_idx * tiles_per_block)


def _load_token_indices_dgt(token_position_to_id, block_idx, B, NUM_TILES, sbm=None, dst=None, src_token_offset=0):
    """Load and transpose token indices for the current block using DGT.

    Single hardware-accelerated `nisa.dma_transpose` (DGT) — no PE, no PSUM —
    to gather-transpose tokens [block_idx*B + src_token_offset : ... + NUM_TILES*TILE_M]
    into SBUF in [TILE_M, NUM_TILES] partition-major layout. Lets prefetch
    overlap freely with PE/PSUM compute in the block loop.

    Local copy of `core.moe.moe_cte.moe_cte_utils.load_token_indices` extended
    with an optional pre-allocated `dst` so the MXFP8 caller can reuse a
    double-buffered slot. The upstream version always allocates internally.
    TODO: when refactoring, push `dst=` into the upstream helper and delete this.

    Args:
        token_position_to_id (nl.ndarray): [N*B] full token position map.
        block_idx (int): Current block index.
        B (int): Block size.
        NUM_TILES (int): Number of TILE_M-sized tiles to load. Normally B // TILE_M
            (the whole block); a half-block tail (MXFP8 fwd, odd N) passes
            (B // TILE_M) // 2 to load only its half of the block's tokens.
        sbm (SbufManager, optional): Used for the internal allocation when dst is None.
        dst (nl.ndarray, optional): Pre-allocated [TILE_M, NUM_TILES] SBUF buffer.
            When provided, no internal allocation happens.
        src_token_offset (int): Token offset within the block to start loading at
            (default 0 = start of block). The MXFP8 fwd half-block tail passes
            shard_id * (B // 2) so each core loads its contiguous half of the tail
            block's tokens into a compact [TILE_M, NUM_TILES] buffer.

    Returns:
        nl.ndarray: [TILE_M, NUM_TILES] transposed token indices in SBUF (== dst when provided).
    """
    if dst == None:
        result = sbm.alloc_stack(
            (TILE_M, NUM_TILES),
            dtype=nl.int32,
            name=f"token_indices_{block_idx}",
            align=32,
        )
    else:
        result = dst

    offset = block_idx * B + src_token_offset
    nisa.dma_transpose(
        dst=result.ap(pattern=[[NUM_TILES, TILE_M], [1, 1], [1, 1], [1, NUM_TILES]]),
        src=token_position_to_id.ap(
            pattern=[[TILE_M, NUM_TILES], [1, 1], [1, 1], [1, TILE_M]],
            offset=offset,
        ),
    )
    return result


# Phase 1: Down Projection Output Grad + SwiGLU Backward


def _compute_phase1_down_proj_output_grad_mxfp8(
    output_grad_td,
    down_weight_td,
    gate_up_proj_act_checkpoint_T,
    block_idx,
    expert_idx,
    B,
    H,
    I_TP,
    E,
    shard_id,
    num_shards,
    blocking,
    config,
    sbm,
    block_token_pos_to_id_full,
    expert_affinities_masked,
    expert_affinities_masked_grad,
    expert_idx_broadcast,
    clamp_limits: ClampLimits = ClampLimits(),
    scaled_intermediate_checkpoint_T_td=None,
    cache_weight=False,
    store_d_gate_up_transpose=True,
):
    """Phase 1: Compute gradient through down projection + SwiGLU backward.

    Steps:
        A) Matmul: d_intermediate = output_grad[token_idx] @ W_down[expert].T  → [B, I_TP]
           LHS is loaded via indirect DMA from the global [T, H] output_grad,
           gathering scattered token rows using block_token_pos_to_id_full as
           the row-index vector.
        B) SwiGLU backward using checkpointed activations:
            gate_pre = checkpoint[block, 0, :, :]  (gate pre-activation)
            gate_act = SiLU(gate_pre)
            up = checkpoint[block, 1, :, :]
            silu_dx = SiLU'(gate_pre)
            d_gate = d_intermediate * (silu_dx * up)
            d_up = d_intermediate * gate_act
        C) Store d_gate, d_up into the interleaved d_gate_up
           [B, 2, I_TP] (Phase 2's input).
        D) For routed execution, transpose d_gate and d_up into
           d_gate_up_T [2*I_TP, B] when requested by Phase 3. Dense execution
           can instead feed d_gate_up [B, 2*I_TP] directly to the wrapX
           PE-swizzle loader.
        E) core_barrier both outputs to make cross-shard writes visible.

    Args:
        output_grad_td (TensorDescriptor): Global [T, H] output gradient. The matmul
            LHS is gathered per-block via indirect DMA — no caller-side gather needed.
        down_weight_td (TensorDescriptor): Down weight descriptor ([E, I_TP, H] slice for expert).
        cache_weight (bool): Keep the quantized down weight for reuse by later
            dense blocks. A quantized input under this policy is block-local.
        gate_up_proj_act_checkpoint_T (nl.ndarray): [N, 2, I_TP, B], Checkpointed activations.
        block_idx (int): Current block index.
        expert_idx: Expert index (dynamic).
        B (int): Block size.
        H (int): Hidden dimension.
        I_TP (int): Intermediate dimension.
        E (int): Number of experts.
        shard_id (int): LNC shard ID.
        num_shards (int): Number of LNC shards.
        blocking (MatmulMxfp8KernelConfig): Blocking parameters for this phase.
    config (MXFP8MOEBwdConfig): Kernel configuration.
        sbm (SbufManager): SBUF memory manager.
        block_token_pos_to_id_full (nl.ndarray): [TILE_M=128, NUM_B_TILES] SBUF int32
            tensor of global token indices for this block. Used as the indirect-DMA
            vector_offset to gather output_grad rows for the matmul LHS.
        expert_affinities_masked (nl.ndarray): [T*E, 1] global affinities. Phase 1
            (AFFINITY_ON_I) gathers a per-token EA scalar per b_tile and uses it
            to (a) pre-scale scaled_intermediate and (b) post-scale
            d_intermediate before SwiGLU bwd.
        expert_affinities_masked_grad (nl.ndarray): [T*E, 1] global EA grad output.
            Phase 1 accumulates dEA[t] = sum_i(d_intermediate[t,i] * intermediate_tile[t,i])
            per b_tile and scatters it (after sendrecv across LNC shards under
            SHARD_ON_FREE).
        expert_idx_broadcast (nl.ndarray): [TILE_M, N] SBUF int32 tensor with the
            expert ID for each block broadcast across all 128 partitions. Sliced
            per block to feed _generate_dynamic_offsets.
        scaled_intermediate_checkpoint_T_td (TensorDescriptor, optional): If the
            caller saved the post-EA scaled intermediate from the forward pass,
            Phase 1 skips computing and storing scaled_intermediate —
            its content (gate_act * up * EA, the same value Phase 4 wants as RHS)
            is already in scaled_intermediate_checkpoint_T_td.data[block_idx].
            None means Phase 1 must produce scaled_intermediate itself.

    Returns:
        tuple:
            - d_gate_up (nl.ndarray): [B, 2, I_TP] shared_hbm
              with d_gate at [:, 0, :] and d_up at [:, 1, :]. Phase 2 LHS.
            - scaled_intermediate (nl.ndarray or None): [B, I_TP] shared_hbm
              holding gate_act * up * EA (= the scaled intermediate, AFFINITY_ON_I).
              Phase 4 RHS source. Returns None when the caller passed
              scaled_intermediate_checkpoint_T_td — the kernel body's Phase 4
              RHS dispatch reads from the user-provided tensor instead.
            - d_gate_up_T (nl.ndarray or None): [2*I_TP, B] shared_hbm with
              the transposed d_gate||d_up for routed Phase 3. None for
              dense PE-swizzle execution, where Phase 3 loads d_gate_up as
              K-by-F.
            - down_weightq_td (TensorDescriptor or None): Quantized down weight
              for reuse by later dense blocks.

    NOTE: The routed path keeps the transposed Phase 3 operand because its
    gathered execution uses the F-by-K DGT path:
      - Phase 2 contracts over I_TP, so its LHS must have I_TP at data.shape[1]
        — d_gate_up[B, 2, I_TP] (or [B, 2*I_TP]) has B at
        dim 0 and works.
      - Phase 3 contracts over B, so its LHS must have B at data.shape[1]
        — that's d_gate_up_T[2*I_TP, B].
    Dense execution instead describes d_gate_up[B, 2*I_TP] as K-by-F. The
    generic loader then uses load_tile_PE_swizzle_wrapX to transpose and
    interleave at load time, avoiding the transposed HBM copy.
    """
    # Allocate per-block HBM outputs. Names follow the BF16 BWD kernel
    # convention so the scheduler can identify them per block.
    d_gate_up = nl.ndarray(
        (B, 2, I_TP),
        dtype=config.compute_dtype,
        buffer=nl.shared_hbm,
        name=f"d_gate_up_shared_block_{block_idx}",
    )
    """
    gate_act * up * EA (the scaled intermediate, Phase 4 RHS). Skipped when
    the caller saved scaled_intermediate from the forward pass — that user-
    provided tensor holds the same value, and the kernel body's Phase 4 RHS
    dispatch will read from it directly.
    """
    skip_scaled_intermediate_compute = scaled_intermediate_checkpoint_T_td != None
    if skip_scaled_intermediate_compute:
        scaled_intermediate = None
    else:
        scaled_intermediate = nl.ndarray(
            (B, I_TP),
            dtype=config.compute_dtype,
            buffer=nl.shared_hbm,
            name=f"scaled_intermediate_shared_block_{block_idx}",
        )
    if store_d_gate_up_transpose:
        d_gate_up_T = nl.ndarray(
            (2 * I_TP, B),
            dtype=config.compute_dtype,
            buffer=nl.shared_hbm,
            name=f"d_gate_up_T_shared_block_{block_idx}",
        )
    else:
        d_gate_up_T = None

    TILES_IN_BLOCK_M = blocking.TILES_IN_BLOCK_M
    TILES_IN_BLOCK_N = blocking.TILES_IN_BLOCK_N
    TILES_IN_BLOCK_K = blocking.TILES_IN_BLOCK_K
    """
    SHARD_ON_FREE: split I_TP across LNC cores. Each core processes only its
    half of the I_TP free dim — matmul, SwiGLU bwd, and HBM stores all use
    I_TP_SHARD_OFFSET to map per-shard local indices to global I_TP positions.
    Down weight load uses rhs_n_offset; HBM writes (d_gate_up, scaled_intermediate,
    d_gate_up_T) and checkpoint reads add the offset directly to i_off.
    """
    I_TP_PER_SHARD = I_TP // num_shards
    I_TP_SHARD_OFFSET = I_TP_PER_SHARD * shard_id
    rhs_n_offset = 0 if cache_weight and down_weight_td.is_quantized else I_TP_SHARD_OFFSET

    # Tile shapes are normalized on the phase config at the kernel boundary.
    # Phase 1: d_intermediate = output_grad[B,H] @ W_down[H,I].T
    #   -> M=B, K=H, N=I_TP_PER_SHARD.
    tile_m = blocking.tile_m
    tile_n = blocking.tile_n
    l_tile_k = blocking.tile_k
    tiles = build_tile_sizes(tile_m=tile_m, l_tile_k=l_tile_k, tile_n=tile_n)

    BLOCK_N = TILES_IN_BLOCK_N * tile_n
    NUM_B_TILES = div_ceil(B, tile_m)
    NUM_I_TILES = div_ceil(I_TP_PER_SHARD, tile_n)
    NUM_K_TILES = div_ceil(H, l_tile_k)
    NUM_M_BLOCKS = div_ceil(NUM_B_TILES, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_I_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

    # Build block descriptor for the matmul
    lhs_load_tile_shape = _compute_load_tile_shape(output_grad_td, tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(down_weight_td, tiles, tile_n)
    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    # Spill/reload buffers. Weight storage can outlive this phase and become
    # the primary descriptor for later dense blocks.
    output_gradq_td = None
    down_weightq_td = None
    spill_reload = config.phase1_config.spill_reload
    if spill_reload or cache_weight:
        data_buffer = nl.private_hbm if config.phase1_config.run_with_lnc2 else nl.hbm
        if spill_reload and not output_grad_td.is_quantized and NUM_N_BLOCKS > 1:
            output_gradq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=config.phase1_config.enable_scale_packing,
                data_buffer=data_buffer,
            )
        if not down_weight_td.is_quantized and (NUM_M_BLOCKS > 1 or cache_weight):
            down_weightq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=config.phase1_config.enable_scale_packing,
                data_buffer=data_buffer,
            )

    sbuf_step_p = TILES_IN_BLOCK_M * BLOCK_N

    # output_grad_td is a pre-gathered block-local [B, H] tensor.
    # The caller handles OOB via _gather_block_tokens with oob_mode.skip.

    """
    ------------------------------------------------------------------------
    AFFINITY_ON_I — EA pre-load and grad accumulator setup.
    ------------------------------------------------------------------------
    Pre-load one EA scalar per token of the block:
      - ea_offsets_all[b_tile]: [TILE_M, 1] int32 token-major flat addresses
        (token_id * E + expert_idx) for the b_tile, used as vector_offset for
        both the EA gather and the EA-grad scatter.
      - ea_tiles_all[:, b_tile]: [TILE_M] fp32 EA values for the b_tile,
        consumed by the per-tile pre-scale and post-scale steps.
    Per-b_tile fp32 EA-grad accumulators are zeroed; each accumulates
      dEA[t] = sum_i (d_intermediate[t,i] * gate_up_mult_unscaled[t,i])
    over all (n_block, tile_n) iterations of the matmul loop, then is
    combined across LNC shards and scattered after the loop.
    """
    is_affinity_i = config.affinity_option == AffinityOption.AFFINITY_ON_I

    if is_affinity_i:
        if not config.single_expert_dense:
            ea_expert_idx_tensor = expert_idx_broadcast[0:tile_m, block_idx : block_idx + 1]
        ea_tiles_all = sbm.alloc_stack(
            (tile_m, NUM_B_TILES),
            dtype=nl.float32,
            name=f"ea_tiles_all_{block_idx}",
            align=32,
        )
        ea_offsets_all = []
        ea_grad_accum_list = []
        ea_grad_reduced_list = []
        for b_tile in range(NUM_B_TILES):
            ea_dst = sbm.alloc_stack(
                (tile_m, 1),
                dtype=nl.float32,
                name=f"ea_load_{block_idx}_{b_tile}",
                align=32,
            )
            if config.single_expert_dense:
                token_start = block_idx * B + b_tile * tile_m
                nisa.dma_copy(
                    dst=ea_dst,
                    src=expert_affinities_masked[token_start : token_start + tile_m, 0:1],
                )
            else:
                token_off = sbm.alloc_stack(
                    (tile_m, 1),
                    dtype=nl.int32,
                    name=f"ea_off_{block_idx}_{b_tile}",
                    align=32,
                )
                addr_tmp = sbm.alloc_stack(
                    (tile_m, 1),
                    dtype=nl.int32,
                    name=f"ea_addr_{block_idx}_{b_tile}",
                    align=32,
                )
                _generate_dynamic_offsets(
                    block_token_pos_to_id_full,
                    ea_expert_idx_tensor,
                    token_off,
                    addr_tmp,
                    b_tile,
                    config.skip_dma,
                    E,
                )
                ea_offsets_all.append(token_off)
                if config.skip_dma.skip_token:
                    nisa.memset(ea_dst, value=0.0)
                nisa.dma_copy(
                    dst=ea_dst,
                    src=expert_affinities_masked.ap(
                        pattern=[[expert_affinities_masked.shape[1], tile_m], [1, 1]],
                        offset=0,
                        vector_offset=token_off,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if config.skip_dma.skip_token else oob_mode.error,
                )
            nisa.tensor_copy(dst=ea_tiles_all[:, b_tile], src=ea_dst)

            acc = sbm.alloc_stack(
                (tile_m, 1),
                dtype=nl.float32,
                name=f"ea_grad_acc_{block_idx}_{b_tile}",
                align=32,
            )
            nisa.memset(acc, value=0.0)
            ea_grad_accum_list.append(acc)
            ea_grad_reduced_list.append(
                sbm.alloc_stack(
                    (tile_m, 1),
                    dtype=config.compute_dtype,
                    name=f"ea_gr_{block_idx}_{b_tile}",
                    align=32,
                )
            )

    for m_block_idx in nl.sequential_range(NUM_M_BLOCKS):
        m_block_start = m_block_idx * TILES_IN_BLOCK_M

        for n_block_idx in range(NUM_N_BLOCKS):
            n_block_start = n_block_idx * TILES_IN_BLOCK_N

            # Accumulator for matmul result
            acc_sbuf = sbm.alloc_stack(shape=(tile_m, TILES_IN_BLOCK_M * BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
            acc_td = TensorDescriptor(data=acc_sbuf)

            for k_block_idx in nl.sequential_range(NUM_K_BLOCKS):
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=output_grad_td,
                    rhs_hbm_td=down_weight_td,
                    bd=bd,
                    output_td=acc_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_m_offset=0,  # Block-local: output_grad is [B, H] per-block
                    rhs_n_offset=rhs_n_offset,
                    TILES_IN_LOAD_M=blocking.TILES_IN_LOAD_M,
                    TILES_IN_LOAD_N=blocking.TILES_IN_LOAD_N,
                    lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                    rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                    lhs_load_tile_shape=lhs_load_tile_shape,
                    rhs_load_tile_shape=rhs_load_tile_shape,
                    lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                    rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                    initialize_accumulator=(k_block_idx == 0),
                    spill_reload=spill_reload,
                    lhsq_td=output_gradq_td,
                    rhsq_td=down_weightq_td,
                    use_scale_packing=config.phase1_config.enable_scale_packing,
                )

            # SwiGLU backward: compute d_gate and d_up from matmul result
            m_tiles_in_block = _valid_tiles_in_block(NUM_B_TILES, m_block_idx, TILES_IN_BLOCK_M)
            n_tiles_in_block = _valid_tiles_in_block(NUM_I_TILES, n_block_idx, TILES_IN_BLOCK_N)
            for tile_m_idx in nl.affine_range(m_tiles_in_block):
                m_idx = m_block_start + tile_m_idx
                local_s = m_idx * tile_m  # offset within the block
                actual_m = min(tile_m, B - local_s)

                for tile_n_idx in nl.affine_range(n_tiles_in_block):
                    n_idx = n_block_start + tile_n_idx
                    actual_n = min(tile_n, I_TP_PER_SHARD - n_idx * tile_n)
                    # Global I_TP position for this tile: per-shard local index
                    # plus the shard's I_TP base offset (SHARD_ON_FREE).
                    i_off = I_TP_SHARD_OFFSET + n_idx * tile_n

                    """
                    Load checkpointed activations for this tile
                    gate_up_proj_act_checkpoint_T: [N, 2, I_TP, B]
                    gate_pre = checkpoint[block_idx, 0, i_off:i_off+TILE_N, local_s:local_s+TILE_M]

                    NOTE: Assumption is that if clamping is used, then the saved gate/up activations have
                    already been clamped and hence the clamping of the projections is not done here.
                    This is a specific scenario used by GPT-OSS.
                    """
                    gate_pre_checkpoint = sbm.alloc_stack(
                        shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf
                    )
                    up_checkpoint = sbm.alloc_stack(
                        shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf
                    )

                    """
                    Load gate_pre via DGT: checkpoint[block_idx, 0, i_off:, local_s:]
                    Source HBM layout: [N, 2, I_TP, B] — reading TILE_N rows of
                    TILE_M contiguous B-elements each. DGT transposes
                    [I_TP_tile, B_tile] → [B_part, I_TP_free] in SBUF.
                    """
                    gate_pre_offset = (block_idx * 2 * I_TP * B) + (0 * I_TP * B) + i_off * B + local_s
                    nisa.dma_transpose(
                        dst=gate_pre_checkpoint.ap(
                            pattern=[[actual_n, actual_m], [1, 1], [1, 1], [1, actual_n]],
                        ),
                        src=gate_up_proj_act_checkpoint_T.ap(
                            pattern=[[B, actual_n], [1, 1], [1, 1], [1, actual_m]],
                            offset=gate_pre_offset,
                        ),
                    )

                    # Load up via DGT: checkpoint[block_idx, 1, i_off:, local_s:]
                    up_offset = (block_idx * 2 * I_TP * B) + (1 * I_TP * B) + i_off * B + local_s
                    nisa.dma_transpose(
                        dst=up_checkpoint.ap(
                            pattern=[[actual_n, actual_m], [1, 1], [1, 1], [1, actual_n]],
                        ),
                        src=gate_up_proj_act_checkpoint_T.ap(
                            pattern=[[B, actual_n], [1, 1], [1, 1], [1, actual_m]],
                            offset=up_offset,
                        ),
                    )

                    # Compute gate_act = SiLU(gate_pre) and silu_dx = SiLU'(gate_pre)
                    silu_dx = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf)
                    nisa.activation(dst=silu_dx, op=nl.silu_dx, data=gate_pre_checkpoint, bias=None, scale=1.0)

                    gate_act = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf)
                    nisa.activation(dst=gate_act, op=nl.silu, data=gate_pre_checkpoint, bias=None, scale=1.0)

                    # d_gate_act = silu_dx * up
                    d_gate_act = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=d_gate_act, data1=silu_dx, data2=up_checkpoint, op=nl.multiply, engine=nisa.vector_engine
                    )

                    """
                    Read matmul result from accumulator. Under AFFINITY_ON_I the
                    EA grad uses the *unscaled* d_intermediate, so we compute
                    EA grad first, then post-scale the accumulator before
                    deriving d_gate / d_up.
                    """
                    sbuf_offset = tile_m_idx * BLOCK_N + tile_n_idx * tile_n
                    acc_tile = acc_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)

                    # Compute the pre-EA intermediate: gate_act * up.
                    # This unscaled value feeds the EA-grad reduction below.
                    intermediate_tile = sbm.alloc_stack(
                        shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf
                    )
                    nisa.tensor_tensor(
                        dst=intermediate_tile,
                        data1=gate_act,
                        data2=up_checkpoint,
                        op=nl.multiply,
                    )

                    if is_affinity_i:
                        global_b_tile_idx = m_idx  # m_idx is already the per-block tile index
                        ea_scalar = ea_tiles_all[:, global_b_tile_idx]

                        """
                        EA grad: dEA[t] += sum_i (d_intermediate[t,i] * intermediate_tile[t,i]).
                        Uses the *unscaled* intermediate (pre-EA) — that's the
                        mathematically correct partner for the d_intermediate sum.
                        """
                        ea_product = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_tensor(
                            dst=ea_product,
                            data1=acc_tile,
                            data2=intermediate_tile,
                            op=nl.multiply,
                        )
                        ea_reduce = sbm.alloc_stack(shape=(actual_m, 1), dtype=nl.float32, buffer=nl.sbuf)
                        nisa.tensor_reduce(
                            dst=ea_reduce,
                            op=nl.add,
                            data=ea_product,
                            axis=1,
                        )
                        nisa.tensor_tensor(
                            dst=ea_grad_accum_list[global_b_tile_idx],
                            op=nl.add,
                            data1=ea_grad_accum_list[global_b_tile_idx],
                            data2=ea_reduce,
                        )

                        """
                        Build scaled_intermediate_tile = intermediate_tile * EA
                        (the post-EA value, Phase 4 RHS). Skipped when the
                        caller forward-saved scaled_intermediate — we won't
                        store it below, so the scaling is wasted.
                        """
                        if not skip_scaled_intermediate_compute:
                            scaled_intermediate_tile = sbm.alloc_stack(
                                shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf
                            )
                            nisa.tensor_scalar(
                                dst=scaled_intermediate_tile,
                                data=intermediate_tile,
                                op0=nl.multiply,
                                operand0=ea_scalar,
                            )

                        """
                        Post-scale d_intermediate (acc_tile) by EA. In-place
                        write back into the matmul accumulator slice — this
                        is the AFFINITY_ON_I post-scale on d_intermediate
                        before SwiGLU bwd reads acc_tile.
                        """
                        nisa.tensor_scalar(
                            dst=acc_tile,
                            data=acc_tile,
                            op0=nl.multiply,
                            operand0=ea_scalar,
                        )

                    # Allocate clamping buffers for non-linear activation gradient clamping
                    if (
                        clamp_limits.non_linear_clamp_upper_limit != None
                        or clamp_limits.non_linear_clamp_lower_limit != None
                    ):
                        clamp_mask1 = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=config.compute_dtype, align=32)
                        clamp_mask2 = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=config.compute_dtype, align=32)
                        clamp_mask = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=config.compute_dtype, align=32)

                    # Allocate clamping buffers for linear activation gradient clamping
                    if clamp_limits.linear_clamp_upper_limit != None or clamp_limits.linear_clamp_lower_limit != None:
                        linear_clamp_mask1 = sbm.alloc_stack(
                            shape=(actual_m, actual_n), dtype=config.compute_dtype, align=32
                        )
                        linear_clamp_mask2 = sbm.alloc_stack(
                            shape=(actual_m, actual_n), dtype=config.compute_dtype, align=32
                        )
                        linear_clamp_mask = sbm.alloc_stack(
                            shape=(actual_m, actual_n), dtype=config.compute_dtype, align=32
                        )

                    # d_gate = d_intermediate * d_gate_act
                    d_gate = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=d_gate, data1=acc_tile, data2=d_gate_act, op=nl.multiply)

                    # Apply gradient clamping for non-linear activation backward pass
                    apply_gradient_clamp(
                        d_gate,
                        gate_pre_checkpoint,
                        clamp_limits.non_linear_clamp_upper_limit,
                        clamp_limits.non_linear_clamp_lower_limit,
                        nl.bfloat16,
                    )

                    # d_up = d_intermediate * gate_act
                    d_up = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=d_up, data1=acc_tile, data2=gate_act, op=nl.multiply)

                    # Apply gradient clamping for linear activation (up) backward pass
                    apply_gradient_clamp(
                        d_up,
                        up_checkpoint,
                        clamp_limits.linear_clamp_upper_limit,
                        clamp_limits.linear_clamp_lower_limit,
                        nl.bfloat16,
                    )

                    # Store d_gate and d_up into the consolidated [B, 2, I_TP]
                    # HBM tensor (Phase 2 LHS).
                    nisa.dma_copy(
                        dst=d_gate_up[local_s : local_s + actual_m, 0, i_off : i_off + actual_n],
                        src=d_gate,
                    )
                    nisa.dma_copy(
                        dst=d_gate_up[local_s : local_s + actual_m, 1, i_off : i_off + actual_n],
                        src=d_up,
                    )

                    """
                    Store scaled_intermediate_tile to scaled_intermediate[B, I_TP]
                    — Phase 4's dW_down RHS. Skipped when scaled_intermediate
                    was forward-saved (see signature).
                    """
                    if not skip_scaled_intermediate_compute:
                        nisa.dma_copy(
                            dst=scaled_intermediate[local_s : local_s + actual_m, i_off : i_off + actual_n],
                            src=scaled_intermediate_tile,
                        )

                    if store_d_gate_up_transpose:
                        """
                        Transpose d_gate and d_up into scratch for routed Phase 3.
                        Dense Phase 3 consumes d_gate_up directly as K-by-F.
                        """
                        _transpose_tile_to_scratch(
                            d_gate,
                            d_gate_up_T,
                            local_s,
                            i_off,
                            I_TP,
                            B,
                            config.compute_dtype,
                            sbm,
                        )
                        _transpose_tile_to_scratch(
                            d_up,
                            d_gate_up_T,
                            local_s,
                            I_TP + i_off,
                            I_TP,
                            B,
                            config.compute_dtype,
                            sbm,
                        )

    """
    ------------------------------------------------------------------------
    AFFINITY_ON_I — combine EA grad across LNC shards and scatter to HBM.
    Under SHARD_ON_FREE each shard holds a partial sum over its half of I_TP;
    sendrecv with the peer adds the two halves to recover the full
        dEA[t] = sum_i (d_intermediate[t,i] * gate_up_mult_unscaled[t,i])
    before scattering one EA grad per (token, expert) pair.
    ------------------------------------------------------------------------
    """
    if is_affinity_i:
        recv_buf_list = []

        for b_tile in range(NUM_B_TILES):
            recv_buf_list.append(
                sbm.alloc_stack(
                    (tile_m, 1),
                    dtype=nl.float32,
                    name=f"ea_recv_{block_idx}_{b_tile}",
                    align=32,
                )
            )
            nisa.sendrecv(
                src=ea_grad_accum_list[b_tile],
                dst=recv_buf_list[b_tile],
                send_to_rank=(1 - shard_id),
                recv_from_rank=(1 - shard_id),
                pipe_id=0,
            )
            nisa.tensor_tensor(
                dst=ea_grad_accum_list[b_tile],
                op=nl.add,
                data1=ea_grad_accum_list[b_tile],
                data2=recv_buf_list[b_tile],
            )

        # Each shard scatters its assigned half of the b_tiles, no overlap
        # so there's no WAW hazard. (Shard-on-free splits B-tiles half/half.)
        if NUM_B_TILES >= num_shards:
            tiles_per_shard = NUM_B_TILES // num_shards
            shard_tile_start = tiles_per_shard * shard_id
        else:
            tiles_per_shard = NUM_B_TILES
            shard_tile_start = 0

        for b_tile in range(tiles_per_shard):
            tile_idx = shard_tile_start + b_tile
            nisa.tensor_copy(
                dst=ea_grad_reduced_list[tile_idx],
                src=ea_grad_accum_list[tile_idx],
                engine=nisa.scalar_engine,
            )
            if config.single_expert_dense:
                token_start = block_idx * B + tile_idx * tile_m
                nisa.dma_copy(
                    dst=expert_affinities_masked_grad[token_start : token_start + tile_m, 0:1],
                    src=ea_grad_reduced_list[tile_idx],
                )
            else:
                nisa.dma_copy(
                    dst=expert_affinities_masked_grad.ap(
                        pattern=[[1, tile_m], [1, 1]],
                        offset=0,
                        vector_offset=ea_offsets_all[tile_idx],
                        indirect_dim=0,
                    ),
                    src=ea_grad_reduced_list[tile_idx],
                    oob_mode=oob_mode.skip if config.skip_dma.skip_token else oob_mode.error,
                )

    # Make per-shard writes visible to the other LNC core before any consumer
    # phase reads these tensors.
    nisa.core_barrier(d_gate_up, (0, 1))
    if not skip_scaled_intermediate_compute:
        nisa.core_barrier(scaled_intermediate, (0, 1))
    if store_d_gate_up_transpose:
        nisa.core_barrier(d_gate_up_T, (0, 1))

    if cache_weight and down_weight_td.is_quantized:
        down_weightq_td = down_weight_td
    return d_gate_up, scaled_intermediate, d_gate_up_T, down_weightq_td if cache_weight else None


def _transpose_tile_to_scratch(src_sbuf, dst_scratch, s_offset, dst_row_offset, I, B, dtype, sbm):
    """Transpose an [actual_m, actual_n] SBUF tile and store to scratch in transposed layout.

    Transposes src_sbuf[actual_m, actual_n] → dst_scratch[dst_row_offset:, s_offset:] as [actual_n, actual_m].

    Args:
        src_sbuf (nl.ndarray): [actual_m, actual_n] source tile in SBUF.
        dst_scratch (nl.ndarray): [2*I, B] transposed scratch in HBM.
        s_offset (int): Column offset in scratch (maps to sequence position).
        dst_row_offset (int): Row offset in scratch for this tile.
        I (int): Intermediate dimension (unused, kept for compatibility).
        B (int): Block size (column count of scratch).
        dtype: Data type.
        sbm (SbufManager): SBUF memory manager.
    """
    actual_m = src_sbuf.shape[0]
    actual_n = src_sbuf.shape[1]
    NUM_SUB = div_ceil(actual_n, actual_m)
    for subtile_idx in range(NUM_SUB):
        sub_off = subtile_idx * actual_m
        sub_width = min(actual_m, actual_n - sub_off)
        # Transpose on PE (PSUM dst routes to PE; SBUF dst routes to DVE
        # which caps at 32x32). Tiles up to 128x128 use PE.
        tile_t_psum = nl.ndarray((sub_width, actual_m), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=tile_t_psum, data=src_sbuf[:, sub_off : sub_off + sub_width])
        # PSUM -> SBUF before DMA (HBM cannot read directly from PSUM)
        tile_t = sbm.alloc_stack(shape=(sub_width, actual_m), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=tile_t, src=tile_t_psum, engine=nisa.scalar_engine)
        # Store to scratch
        nisa.dma_copy(
            dst=dst_scratch[
                dst_row_offset + sub_off : dst_row_offset + sub_off + sub_width, s_offset : s_offset + actual_m
            ],
            src=tile_t,
        )


# Phase 2: Hidden States Gradient (with scatter via token indices)


def _compute_phase2_hidden_states_grad_mxfp8(
    d_gate_up_td,
    gate_up_weight_td,
    hidden_states_grad,
    block_token_pos_to_id_full,
    B,
    H,
    I_TP,
    shard_id,
    num_shards,
    blocking,
    config,
    sbm,
    block_token_offset,
    cache_weight=False,
):
    """Phase 2: Compute hidden states gradient and write it to the output.

    Computes: partial_hidden_grad = d_gate_up[B, 2*I_TP] @ W[E*H, 2*I_TP].T → [B, H_shard]
    This is equivalent to d_gate @ W_gate.T + d_up @ W_up.T in a single matmul
    because [d_gate|d_up] @ [W_gate|W_up].T = d_gate @ W_gate.T + d_up @ W_up.T.
    The general path scatters to hidden_states_grad[token_ids, H_shard]. The
    single-expert dense path writes directly to its contiguous block slice.

    Args:
        d_gate_up_td (TensorDescriptor): [B, 2*I_TP], Combined gate/up gradient (block-local).
        gate_up_weight_td (TensorDescriptor): [E*H, 2*I_TP], Combined gate/up weights with
            scalar_offset set for per-expert indexing.
        cache_weight (bool): Keep the quantized gate/up weight for reuse by
            later dense blocks. A quantized input under this policy is local.
        hidden_states_grad (nl.ndarray): [T, H], Global output tensor.
        block_token_pos_to_id_full (nl.ndarray): [TILE_M, NUM_B_TILES] token
            indices, or None for single_expert_dense.
        B (int): Block size.
        H (int): Hidden dimension.
        I_TP (int): Intermediate dimension.
        shard_id (int): LNC shard ID.
        num_shards (int): Number of LNC shards.
        blocking (MatmulMxfp8KernelConfig): Phase 2 blocking parameters.
        config (MXFP8MOEBwdConfig): Kernel configuration.
        sbm (SbufManager): SBUF memory manager.
    """
    TILES_IN_BLOCK_M = blocking.TILES_IN_BLOCK_M
    TILES_IN_BLOCK_N = blocking.TILES_IN_BLOCK_N
    TILES_IN_BLOCK_K = blocking.TILES_IN_BLOCK_K
    H_PER_SHARD = H // num_shards
    H_SHARD_OFFSET = H_PER_SHARD * shard_id
    rhs_n_offset = 0 if cache_weight and gate_up_weight_td.is_quantized else H_SHARD_OFFSET

    # Tile shapes: phase config tile_* FIRST, else shape-derived (exp-one config.py pattern).
    # Phase 2: hidden_states_grad = d_gate_up[B,2I] @ W_gate_up[2I,H].T -> M=B, K=2*I_TP, N=H_PER_SHARD.
    tile_m = blocking.tile_m
    tile_n = blocking.tile_n
    l_tile_k = blocking.tile_k
    tiles = build_tile_sizes(tile_m=tile_m, l_tile_k=l_tile_k, tile_n=tile_n)

    BLOCK_N = TILES_IN_BLOCK_N * tile_n
    NUM_B_TILES = div_ceil(B, tile_m)
    NUM_H_TILES = div_ceil(H_PER_SHARD, tile_n)
    NUM_K_TILES = div_ceil((2 * I_TP), l_tile_k)
    NUM_M_BLOCKS = div_ceil(NUM_B_TILES, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_H_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

    # Build block descriptor
    lhs_load_tile_shape = _compute_load_tile_shape(d_gate_up_td, tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(gate_up_weight_td, tiles, tile_n)
    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        rhs_load_tile_shape=rhs_load_tile_shape,
        lhs_load_tile_shape=lhs_load_tile_shape,
        tiles=tiles,
    )

    # Spill/reload buffers. Weight storage can outlive this phase and become
    # the primary descriptor for later dense blocks.
    d_gate_upq_td = None
    gate_up_weightq_td = None
    spill_reload = config.phase2_config.spill_reload
    if spill_reload or cache_weight:
        data_buffer = nl.private_hbm if config.phase2_config.run_with_lnc2 else nl.hbm
        if spill_reload and NUM_N_BLOCKS > 1:
            d_gate_upq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=config.phase2_config.enable_scale_packing,
                data_buffer=data_buffer,
            )
        if not gate_up_weight_td.is_quantized and (NUM_M_BLOCKS > 1 or cache_weight):
            gate_up_weightq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=config.phase2_config.enable_scale_packing,
                data_buffer=data_buffer,
            )

    for idx_m in range(NUM_M_BLOCKS):
        for idx_n in range(NUM_N_BLOCKS):
            output_sbuf = sbm.alloc_stack(shape=(tile_m, TILES_IN_BLOCK_M, BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
            output_sbuf_td = TensorDescriptor(data=output_sbuf)

            # Combined matmul: d_gate_up[B, 2*I_TP] @ gate_up_weight[E*H, 2*I_TP].T → [B, H]
            generic_matmul_mxfp8_api(
                lhs_hbm_td=d_gate_up_td,
                rhs_hbm_td=gate_up_weight_td,
                bd=bd,
                output_td=output_sbuf_td,
                block_idx_m=(idx_m, idx_m + 1),
                block_idx_n=(idx_n, idx_n + 1),
                lhs_m_offset=0,
                rhs_n_offset=rhs_n_offset,
                TILES_IN_LOAD_M=blocking.TILES_IN_LOAD_M,
                TILES_IN_LOAD_N=blocking.TILES_IN_LOAD_N,
                lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                lhs_load_tile_shape=lhs_load_tile_shape,
                rhs_load_tile_shape=rhs_load_tile_shape,
                lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                spill_reload=spill_reload,
                lhsq_td=d_gate_upq_td,
                rhsq_td=gate_up_weightq_td,
                use_scale_packing=config.phase2_config.enable_scale_packing,
            )

            """
            Scatter result to hidden_states_grad via token indices.
            When accumulate_hidden_states_grad=True, multiple experts contribute
            to the same token's hidden grad (top-K > 1 routing): the existing
            value at hidden_states_grad[token_id, :] is gathered, summed with
            this block's contribution, and scattered back. When False, we
            overwrite — correct only when each token is touched by exactly one
            block (top-K = 1).
            """
            actual_n = min(BLOCK_N, H_PER_SHARD - idx_n * BLOCK_N)
            sbuf_step = TILES_IN_BLOCK_M * BLOCK_N
            m_tiles_in_block = _valid_tiles_in_block(NUM_B_TILES, idx_m, TILES_IN_BLOCK_M)
            for tile_m_idx in nl.affine_range(m_tiles_in_block):
                b_tile_idx = idx_m * TILES_IN_BLOCK_M + tile_m_idx
                h_col_base = idx_n * BLOCK_N + H_SHARD_OFFSET
                actual_m = min(tile_m, B - b_tile_idx * tile_m)

                sbuf_offset = tile_m_idx * BLOCK_N
                result_tile = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf)
                # Cast from fp32 accumulator to compute dtype
                nisa.tensor_copy(
                    dst=result_tile,
                    src=output_sbuf.ap(pattern=[[sbuf_step, actual_m], [1, actual_n]], offset=sbuf_offset),
                )

                if config.single_expert_dense:
                    token_start = block_token_offset + b_tile_idx * tile_m
                    nisa.dma_copy(
                        dst=hidden_states_grad[
                            token_start : token_start + actual_m,
                            h_col_base : h_col_base + actual_n,
                        ],
                        src=result_tile,
                    )
                else:
                    # Build the token-index vector for this B-tile (one int32 per partition).
                    token_indices_col = block_token_pos_to_id_full[:, b_tile_idx : b_tile_idx + 1]

                    if config.accumulate_hidden_states_grad:
                        """
                        Read-modify-write: gather existing grad, add this block's
                        contribution, then scatter back. Out-of-bounds tokens are
                        zeroed first so masked-out lanes contribute nothing.
                        """
                        existing_tile = sbm.alloc_stack(
                            shape=(actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf
                        )
                        if config.skip_dma.skip_token:
                            nisa.memset(existing_tile, value=0)
                        nisa.dma_copy(
                            dst=existing_tile,
                            src=hidden_states_grad.ap(
                                pattern=[[H, actual_m], [1, actual_n]],
                                offset=h_col_base,
                                vector_offset=token_indices_col,
                                indirect_dim=0,
                            ),
                            oob_mode=oob_mode.skip if config.skip_dma.skip_token else oob_mode.error,
                        )
                        nisa.tensor_tensor(
                            dst=result_tile,
                            op=nl.add,
                            data1=result_tile,
                            data2=existing_tile,
                        )

                    # Scatter using token indices (indirect DMA).
                    nisa.dma_copy(
                        dst=hidden_states_grad.ap(
                            pattern=[[H, actual_m], [1, actual_n]],
                            offset=h_col_base,
                            vector_offset=token_indices_col,
                            indirect_dim=0,
                        ),
                        src=result_tile,
                        oob_mode=oob_mode.skip if config.skip_dma.skip_token else oob_mode.error,
                    )

    if cache_weight and gate_up_weight_td.is_quantized:
        gate_up_weightq_td = gate_up_weight_td
    return gate_up_weightq_td if cache_weight else None


def _transpose_accumulate_weight_grad_tile(
    acc_tile,
    weight_grad_hbm,
    expert_idx,
    pair0_stride,
    base_offset,
    dtype,
    sbm,
    accumulate,
):
    """Transpose an [actual_m, actual_n] matmul result and write an expert's weight grad.

    For each actual_m x actual_m sub-tile of acc_tile:
      1. nc_transpose to PSUM (PE route for tiles up to 128x128)
      2. tensor_copy PSUM → SBUF
      3. When requested, load and accumulate the existing gradient
      4. DMA store back to HBM

    ``accumulate`` controls whether the existing gradient is loaded and added
    before the result is stored.

    The transpose swaps partition and free dims so the DMA access pattern's
    pair 0 stride > 1 (required by HWDGE hardware).

    Sub-tile offsets are computed as: base_offset + sub_idx * sub_size * pair0_stride.
    This follows from the HBM layout: consecutive sub_size-wide slices along the
    pair0 dimension are separated by sub_size * pair0_stride elements.

    Args:
        acc_tile (nl.ndarray): [actual_m, actual_n] SBUF tile from matmul accumulator.
        weight_grad_hbm (nl.ndarray): The full HBM weight gradient tensor
            (e.g., [E, H, 2, I_TP] or [E, I_TP, H]).
        expert_idx: Runtime expert index (SBUF scalar for scalar_offset).
        pair0_stride (int): Stride for pair 0 of the access pattern. This is the
            stride of the dimension that maps to SBUF partitions after transpose
            (e.g., 2*I_TP for gate_up, H for down_proj).
        base_offset (int): Flat HBM offset for the first sub-tile (sub_idx=0).
        dtype: Compute data type.
        sbm (SbufManager): SBUF memory manager.
    """
    actual_m = acc_tile.shape[0]
    actual_n = acc_tile.shape[1]
    NUM_SUBTILES = div_ceil(actual_n, actual_m)
    for sub_idx in range(NUM_SUBTILES):
        sub_width = min(actual_m, actual_n - sub_idx * actual_m)
        sub_t_psum = nl.ndarray((sub_width, actual_m), dtype=dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=sub_t_psum, data=acc_tile[:, sub_idx * actual_m : (sub_idx * actual_m) + sub_width])
        sub_t = sbm.alloc_stack((sub_width, actual_m), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=sub_t, src=sub_t_psum, engine=nisa.scalar_engine)

        offset = base_offset + sub_idx * actual_m * pair0_stride
        if expert_idx is None:
            weight_grad_ap = weight_grad_hbm.ap(
                pattern=[[pair0_stride, sub_width], [1, actual_m]],
                offset=offset,
            )
        else:
            weight_grad_ap = weight_grad_hbm.ap(
                pattern=[[pair0_stride, sub_width], [1, actual_m]],
                offset=offset,
                scalar_offset=expert_idx,
                indirect_dim=0,
            )

        if accumulate:
            existing = sbm.alloc_stack((sub_width, actual_m), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=existing, src=weight_grad_ap, dge_mode=dge_mode.hwdge)
            nisa.tensor_tensor(dst=sub_t, op=nl.add, data1=sub_t, data2=existing)

        nisa.dma_copy(dst=weight_grad_ap, src=sub_t, dge_mode=dge_mode.hwdge)


# Phase 3: Gate/Up Weight Gradient (accumulated across blocks)


def _compute_phase3_gate_up_weight_grad_mxfp8(
    lhs_td,
    rhs_td,
    gate_up_proj_weight_grad,
    expert_idx,
    B,
    H,
    I_TP,
    shard_id,
    num_shards,
    blocking,
    config,
    sbm,
    accumulate_weight_grad,
):
    """Phase 3: Compute gate/up weight gradient.

    Computes: dW_gate_up[expert] += d_gate_up.T @ hidden_states[block]  → [I_TP, H]
    Accumulates into the expert's weight gradient slice.

    Routed execution uses d_gate_up_T [2*I_TP, B] and hidden_states_T
    [H, B]. Dense execution passes the natural [B, 2*I_TP] and [B, H]
    tensors as K-by-F operands to load_tile_PE_swizzle_wrapX.
    Output is accumulated into gate_up_proj_weight_grad[expert, :, :, :].

    Args:
        lhs_td (TensorDescriptor): Routed [2*I_TP, B] F-by-K or dense
            [B, 2*I_TP] K-by-F d_gate||d_up.
        rhs_td (TensorDescriptor): Routed [H, B] F-by-K or dense
            [B, H] K-by-F hidden states.
        gate_up_proj_weight_grad (nl.ndarray): [E, H, 2, I_TP], Output weight gradient tensor.
        expert_idx: Expert index (dynamic).
        B (int): Block size.
        H (int): Hidden dimension.
        I_TP (int): Intermediate dimension.
        shard_id (int): LNC shard ID.
        num_shards (int): Number of LNC shards.
        blocking (MatmulMxfp8KernelConfig): Phase 3 blocking parameters.
        config (MXFP8MOEBwdConfig): Kernel configuration.
        sbm (SbufManager): SBUF memory manager.
        accumulate_weight_grad (bool): Add this block to an existing gradient.
    """
    TILES_IN_BLOCK_M = blocking.TILES_IN_BLOCK_M
    TILES_IN_BLOCK_N = blocking.TILES_IN_BLOCK_N
    TILES_IN_BLOCK_K = blocking.TILES_IN_BLOCK_K
    H_PER_SHARD = H // num_shards
    H_SHARD_OFFSET = H_PER_SHARD * shard_id

    # Tile shapes: phase config tile_* FIRST, else shape-derived (exp-one config.py pattern).
    # The routed path computes gate/up separately with M=I_TP. The dense path
    # follows the exp-one/k3g shape and computes both in one M=2*I_TP matmul.
    tile_m = blocking.tile_m
    tile_n = blocking.tile_n
    l_tile_k = blocking.tile_k
    tiles = build_tile_sizes(tile_m=tile_m, l_tile_k=l_tile_k, tile_n=tile_n)

    BLOCK_N = TILES_IN_BLOCK_N * tile_n
    phase3_m = 2 * I_TP if config.single_expert_dense else I_TP
    NUM_I_TILES = div_ceil(phase3_m, tile_m)
    NUM_H_TILES = div_ceil(H_PER_SHARD, tile_n)
    NUM_K_TILES = div_ceil(B, l_tile_k)
    # Clamp TILES_IN_BLOCK_K to actual K-tiles available to prevent OOB loads
    TILES_IN_BLOCK_K = min(TILES_IN_BLOCK_K, NUM_K_TILES)
    NUM_M_BLOCKS = div_ceil(NUM_I_TILES, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_H_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = max(1, div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K))

    # Build block descriptor
    lhs_load_tile_shape = _compute_load_tile_shape(lhs_td, tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(rhs_td, tiles, tile_n)
    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    """
    Spill/reload buffers — gate and up need separate LHS spill buffers
    because both matmuls share the same source tensor but use
    different M offsets (0 vs I_TP). A single shared buffer gets overwritten
    by the up matmul before the gate matmul can reload on subsequent N-blocks.
    """
    gate_lhsq_td = None
    up_lhsq_td = None
    rhsq_td = None
    if config.phase3_config.spill_reload and NUM_K_BLOCKS > 0:
        data_buffer = nl.private_hbm if config.phase3_config.run_with_lnc2 else nl.hbm
        if not config.single_expert_dense and NUM_N_BLOCKS > 1:
            gate_lhsq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=config.phase3_config.enable_scale_packing,
                data_buffer=data_buffer,
            )
            up_lhsq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=config.phase3_config.enable_scale_packing,
                data_buffer=data_buffer,
            )
        if not rhs_td.is_quantized and NUM_M_BLOCKS > 1:
            rhsq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=config.phase3_config.enable_scale_packing,
                data_buffer=data_buffer,
            )

    for m_block_idx in nl.sequential_range(NUM_M_BLOCKS):
        m_block_start = m_block_idx * TILES_IN_BLOCK_M

        for n_block_idx in range(NUM_N_BLOCKS):
            gate_acc_sbuf = sbm.alloc_stack(
                shape=(tile_m, TILES_IN_BLOCK_M * BLOCK_N), dtype=nl.float32, buffer=nl.sbuf
            )
            gate_acc_td = TensorDescriptor(data=gate_acc_sbuf)
            if not config.single_expert_dense:
                up_acc_sbuf = sbm.alloc_stack(
                    shape=(tile_m, TILES_IN_BLOCK_M * BLOCK_N), dtype=nl.float32, buffer=nl.sbuf
                )
                up_acc_td = TensorDescriptor(data=up_acc_sbuf)

            for k_block_idx in nl.sequential_range(NUM_K_BLOCKS):
                # Dense uses the full [2I,B] operand. Routed execution retains
                # separate gate/up calls because each block accumulates by expert.
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=lhs_td,
                    rhs_hbm_td=rhs_td,
                    bd=bd,
                    output_td=gate_acc_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_m_offset=0,  # Gate rows start at 0
                    rhs_n_offset=H_SHARD_OFFSET,  # SHARD_ON_FREE: each core's H slice
                    TILES_IN_LOAD_M=blocking.TILES_IN_LOAD_M,
                    TILES_IN_LOAD_N=blocking.TILES_IN_LOAD_N,
                    lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                    rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                    lhs_load_tile_shape=lhs_load_tile_shape,
                    rhs_load_tile_shape=rhs_load_tile_shape,
                    lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                    rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                    initialize_accumulator=(k_block_idx == 0),
                    spill_reload=config.phase3_config.spill_reload,
                    lhsq_td=gate_lhsq_td,
                    rhsq_td=rhsq_td,
                    use_scale_packing=config.phase3_config.enable_scale_packing,
                )

                if not config.single_expert_dense:
                    # Up grad: d_up.T @ hidden.T (rows I_TP:2*I_TP of d_gate_up_T)
                    generic_matmul_mxfp8_api(
                        lhs_hbm_td=lhs_td,
                        rhs_hbm_td=rhs_td,
                        bd=bd,
                        output_td=up_acc_td,
                        block_idx_m=(m_block_idx, m_block_idx + 1),
                        block_idx_n=(n_block_idx, n_block_idx + 1),
                        block_idx_k=(k_block_idx, k_block_idx + 1),
                        lhs_m_offset=I_TP,
                        rhs_n_offset=H_SHARD_OFFSET,
                        TILES_IN_LOAD_M=blocking.TILES_IN_LOAD_M,
                        TILES_IN_LOAD_N=blocking.TILES_IN_LOAD_N,
                        lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                        rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                        lhs_load_tile_shape=lhs_load_tile_shape,
                        rhs_load_tile_shape=rhs_load_tile_shape,
                        lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                        rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                        initialize_accumulator=(k_block_idx == 0),
                        spill_reload=config.phase3_config.spill_reload,
                        lhsq_td=up_lhsq_td,
                        rhsq_td=rhsq_td,
                        use_scale_packing=config.phase3_config.enable_scale_packing,
                    )

            """
            Accumulate gate and up weight grads into output
            gate_up_proj_weight_grad: [E, H, 2, I_TP]
            Matmul output: partition=I_TP (tile_m), free=H (tile_n)
            HBM layout: H stride=2*I_TP (partition), I_TP stride=1 (free)
            Must transpose 128x128 sub-tiles so partition maps to H for DMA.
            """
            sbuf_step = TILES_IN_BLOCK_M * BLOCK_N
            m_tiles_in_block = _valid_tiles_in_block(NUM_I_TILES, m_block_idx, TILES_IN_BLOCK_M)
            n_tiles_in_block = _valid_tiles_in_block(NUM_H_TILES, n_block_idx, TILES_IN_BLOCK_N)
            for tile_m_idx in nl.affine_range(m_tiles_in_block):
                i_off = (m_block_start + tile_m_idx) * tile_m
                actual_m = min(tile_m, phase3_m - i_off)
                for tile_n_idx in nl.affine_range(n_tiles_in_block):
                    h_base = H_SHARD_OFFSET + n_block_idx * BLOCK_N + tile_n_idx * tile_n
                    actual_n = min(tile_n, H_PER_SHARD - (n_block_idx * BLOCK_N + tile_n_idx * tile_n))
                    sbuf_offset = tile_m_idx * BLOCK_N + tile_n_idx * tile_n

                    # Read matmul result [tile_m, tile_n] from accumulator
                    acc_tile = sbm.alloc_stack((actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=acc_tile,
                        src=gate_acc_sbuf.ap(pattern=[[sbuf_step, actual_m], [1, actual_n]], offset=sbuf_offset),
                    )
                    # Dense i_off spans [0,2I), matching the flattened [gate,up]
                    # dimension of gate_up_proj_weight_grad[E,H,2,I].
                    gate_grad_base_offset = h_base * 2 * I_TP + i_off
                    _transpose_accumulate_weight_grad_tile(
                        acc_tile=acc_tile,
                        weight_grad_hbm=gate_up_proj_weight_grad,
                        expert_idx=expert_idx,
                        pair0_stride=2 * I_TP,
                        base_offset=gate_grad_base_offset,
                        dtype=config.compute_dtype,
                        sbm=sbm,
                        accumulate=accumulate_weight_grad,
                    )

                    if not config.single_expert_dense:
                        up_acc_tile = sbm.alloc_stack(
                            (actual_m, actual_n),
                            dtype=config.compute_dtype,
                            buffer=nl.sbuf,
                        )
                        nisa.tensor_copy(
                            dst=up_acc_tile,
                            src=up_acc_sbuf.ap(
                                pattern=[[sbuf_step, actual_m], [1, actual_n]],
                                offset=sbuf_offset,
                            ),
                        )
                        up_grad_base_offset = h_base * 2 * I_TP + I_TP + i_off
                        _transpose_accumulate_weight_grad_tile(
                            acc_tile=up_acc_tile,
                            weight_grad_hbm=gate_up_proj_weight_grad,
                            expert_idx=expert_idx,
                            pair0_stride=2 * I_TP,
                            base_offset=up_grad_base_offset,
                            dtype=config.compute_dtype,
                            sbm=sbm,
                            accumulate=accumulate_weight_grad,
                        )


# Phase 4: Down Projection Weight Gradient (accumulated across blocks)


def _compute_phase4_down_weight_grad_mxfp8(
    lhs_td,
    rhs_td,
    down_proj_weight_grad,
    expert_idx,
    B,
    H,
    I_TP,
    shard_id,
    num_shards,
    blocking,
    config,
    sbm,
    accumulate_weight_grad,
):
    """Phase 4: Compute down projection weight gradient.

    Computes: dW_down[expert] += output_grad[block].T @ scaled_intermediate[block]  → [H, I_TP]
    Accumulates into the expert's down weight gradient slice. Under AFFINITY_ON_I
    the RHS is the post-EA scaled intermediate (gate_act * up * EA), matching the
    forward operand of the down projection.

    Routed execution uses output_grad.T [H, B] and scaled_intermediate.T
    [I_TP, B]. Dense execution passes their natural [B, H] and [B, I_TP]
    layouts as K-by-F operands to load_tile_PE_swizzle_wrapX.
    Output accumulates into down_proj_weight_grad[expert, :, :].

    Args:
        lhs_td (TensorDescriptor): Routed [H, B] F-by-K or dense
            [B, H] K-by-F output gradient.
        rhs_td (TensorDescriptor): Routed [I_TP, B] F-by-K
            or dense [B, I_TP] K-by-F post-EA scaled intermediate.
        down_proj_weight_grad (nl.ndarray): [E, I_TP, H], Output weight gradient tensor.
        expert_idx: Expert index (dynamic).
        B (int): Block size.
        H (int): Hidden dimension.
        I_TP (int): Intermediate dimension.
        shard_id (int): LNC shard ID.
        num_shards (int): Number of LNC shards.
        blocking (MatmulMxfp8KernelConfig): Phase 4 blocking parameters.
        config (MXFP8MOEBwdConfig): Kernel configuration.
        sbm (SbufManager): SBUF memory manager.
        accumulate_weight_grad (bool): Add this block to an existing gradient.
    """
    TILES_IN_BLOCK_M = blocking.TILES_IN_BLOCK_M
    TILES_IN_BLOCK_N = blocking.TILES_IN_BLOCK_N
    TILES_IN_BLOCK_K = blocking.TILES_IN_BLOCK_K
    H_PER_SHARD = H // num_shards
    H_SHARD_OFFSET = H_PER_SHARD * shard_id

    # Tile shapes: phase config tile_* FIRST, else shape-derived (exp-one config.py pattern).
    # Phase 4: dW_down = output_grad_T[H,B] @ scaled_intermediate_T[B,I] (contract over B)
    #   -> M=H_PER_SHARD, K=B, N=I_TP.
    tile_m = blocking.tile_m
    tile_n = blocking.tile_n
    l_tile_k = blocking.tile_k
    tiles = build_tile_sizes(tile_m=tile_m, l_tile_k=l_tile_k, tile_n=tile_n)

    BLOCK_N = TILES_IN_BLOCK_N * tile_n
    BLOCK_M = TILES_IN_BLOCK_M * tile_m
    NUM_H_TILES = div_ceil(H_PER_SHARD, tile_m)
    NUM_I_TILES = div_ceil(I_TP, tile_n)
    NUM_K_TILES = div_ceil(B, l_tile_k)
    # Clamp TILES_IN_BLOCK_K to actual K-tiles available to prevent OOB loads
    TILES_IN_BLOCK_K = min(TILES_IN_BLOCK_K, NUM_K_TILES)
    NUM_M_BLOCKS = div_ceil(NUM_H_TILES, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_I_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = max(1, div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K))

    # Build block descriptor
    lhs_load_tile_shape = _compute_load_tile_shape(lhs_td, tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(rhs_td, tiles, tile_n)
    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    # Spill/reload buffers
    lhsq_td = None
    rhsq_td = None
    if config.phase4_config.spill_reload and NUM_K_BLOCKS > 0:
        data_buffer = nl.private_hbm if config.phase4_config.run_with_lnc2 else nl.hbm
        if not lhs_td.is_quantized:
            lhsq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=config.phase4_config.enable_scale_packing,
                data_buffer=data_buffer,
            )
        if not rhs_td.is_quantized:
            rhsq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=config.phase4_config.enable_scale_packing,
                data_buffer=data_buffer,
            )

    for idx_m in nl.sequential_range(NUM_M_BLOCKS):
        for idx_n in nl.sequential_range(NUM_N_BLOCKS):
            output_sbuf = sbm.alloc_stack(shape=(tile_m, TILES_IN_BLOCK_M, BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
            output_sbuf_td = TensorDescriptor(data=output_sbuf)

            generic_matmul_mxfp8_api(
                lhs_hbm_td=lhs_td,
                rhs_hbm_td=rhs_td,
                bd=bd,
                output_td=output_sbuf_td,
                block_idx_m=(idx_m, idx_m + 1),
                block_idx_n=(idx_n, idx_n + 1),
                lhs_m_offset=H_SHARD_OFFSET,  # SHARD_ON_FREE: each core's H slice
                TILES_IN_LOAD_M=blocking.TILES_IN_LOAD_M,
                TILES_IN_LOAD_N=blocking.TILES_IN_LOAD_N,
                lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                lhs_load_tile_shape=lhs_load_tile_shape,
                rhs_load_tile_shape=rhs_load_tile_shape,
                lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                spill_reload=config.phase4_config.spill_reload,
                lhsq_td=lhsq_td,
                rhsq_td=rhsq_td,
                use_scale_packing=config.phase4_config.enable_scale_packing,
            )

            """
            Accumulate into down_proj_weight_grad[expert, :, H_shard_slice]
            down_proj_weight_grad: [E, I_TP, H]
            Matmul output: partition=H (tile_m), free=I_TP (tile_n)
            HBM layout: I_TP stride=H (partition), H stride=1 (free)
            Must transpose tile_m x tile_m sub-tiles so partition maps to I_TP for DMA.
            """
            sbuf_step = TILES_IN_BLOCK_M * BLOCK_N
            m_tiles_in_block = _valid_tiles_in_block(NUM_H_TILES, idx_m, TILES_IN_BLOCK_M)
            n_tiles_in_block = _valid_tiles_in_block(NUM_I_TILES, idx_n, TILES_IN_BLOCK_N)
            for tile_m_idx in nl.affine_range(m_tiles_in_block):
                h_off = H_SHARD_OFFSET + idx_m * BLOCK_M + tile_m_idx * tile_m
                actual_m = min(tile_m, H_PER_SHARD - (idx_m * BLOCK_M + tile_m_idx * tile_m))
                for tile_n_idx in nl.affine_range(n_tiles_in_block):
                    i_base = idx_n * BLOCK_N + tile_n_idx * tile_n
                    actual_n = min(tile_n, I_TP - i_base)
                    sbuf_offset = tile_m_idx * BLOCK_N + tile_n_idx * tile_n

                    acc_tile = sbm.alloc_stack((actual_m, actual_n), dtype=config.compute_dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(
                        dst=acc_tile,
                        src=output_sbuf.ap(pattern=[[sbuf_step, actual_m], [1, actual_n]], offset=sbuf_offset),
                    )

                    _transpose_accumulate_weight_grad_tile(
                        acc_tile=acc_tile,
                        weight_grad_hbm=down_proj_weight_grad,
                        expert_idx=expert_idx,
                        pair0_stride=H,
                        base_offset=i_base * H + h_off,
                        dtype=config.compute_dtype,
                        sbm=sbm,
                        accumulate=accumulate_weight_grad,
                    )


def _set_expert_offset_on_td(
    td,
    expert_idx_broadcast,
    block_idx,
    expert_stride,
    scales_stride,
    effective_f_dim,
    name_prefix,
    sbm,
):
    """Set per-expert scalar_offset on a TensorDescriptor for the current block.

    Handles both the non-prequantized path (float32 broadcast across TILE_M
    partitions) and the prequantized path (uint32 scalar + separate scales offset).
    """
    if td.scales is None:
        offset_buf = sbm.alloc_stack(
            (TILE_M, 1),
            dtype=nl.float32,
            buffer=nl.sbuf,
            name=f"{name_prefix}_expert_offset_{block_idx}",
            align=32,
        )
        nisa.tensor_scalar(
            dst=offset_buf,
            data=expert_idx_broadcast[0:TILE_M, block_idx : block_idx + 1],
            op0=nl.multiply,
            operand0=expert_stride,
        )
        td.scalar_offset = offset_buf
        td.effective_f_dim = effective_f_dim
    else:
        offset_buf = sbm.alloc_stack(
            (1, 1),
            dtype=nl.uint32,
            buffer=nl.sbuf,
            name=f"{name_prefix}_expert_offset_{block_idx}",
            align=32,
        )
        nisa.tensor_scalar(
            dst=offset_buf,
            data=expert_idx_broadcast[0:1, block_idx : block_idx + 1],
            op0=nl.multiply,
            operand0=expert_stride,
        )
        scales_offset_buf = sbm.alloc_stack(
            (1, 1),
            dtype=nl.uint32,
            buffer=nl.sbuf,
            name=f"{name_prefix}_expert_offset_scales_{block_idx}",
            align=32,
        )
        nisa.tensor_scalar(
            dst=scales_offset_buf,
            data=expert_idx_broadcast[0:1, block_idx : block_idx + 1],
            op0=nl.multiply,
            operand0=scales_stride,
        )
        td.scalar_offset = offset_buf
        td.scalar_offset_scales = scales_offset_buf
        td.effective_f_dim = effective_f_dim
        td.effective_f_dim_scales = scales_stride


# Main Kernel Implementation


@with_active_sbm
def blockwise_mm_bwd_dropless_mxfp8(
    # --- Input TensorDescriptors (constructed from raw tensors at kernel entry) ---
    hidden_states_td: TensorDescriptor,
    output_grad_td: TensorDescriptor,
    gate_up_weight_td: TensorDescriptor,
    down_weight_td: TensorDescriptor,
    token_position_to_id_td: TensorDescriptor,
    block_to_expert_td: TensorDescriptor,
    expert_affinities_masked_td: TensorDescriptor,
    gate_up_proj_act_checkpoint_T_td: TensorDescriptor,
    gate_act_checkpoint_T_td,
    intermediate_checkpoint_T_td,
    scaled_intermediate_checkpoint_T_td,
    down_proj_act_checkpoint_td,
    # --- Derived dimensions (plain ints) ---
    T: int,
    H: int,
    I_TP: int,
    E: int,
    N: int,
    block_size: int,
    # --- Tensor-local layout conversion modes ---
    output_grad_swizzle_mode: SwizzleMode,
    d_gate_up_swizzle_mode: SwizzleMode,
    d_gate_up_t_swizzle_mode: SwizzleMode,
    hidden_states_t_swizzle_mode: SwizzleMode,
    output_grad_t_swizzle_mode: SwizzleMode,
    scaled_intermediate_t_swizzle_mode: SwizzleMode,
    # --- Config and output gradient buffers ---
    config: MXFP8MOEBwdConfig,
    hidden_states_grad: nl.ndarray,
    expert_affinities_masked_grad: nl.ndarray,
    gate_up_proj_weight_grad: nl.ndarray,
    down_proj_weight_grad: nl.ndarray,
    gate_and_up_proj_bias_grad: nl.ndarray = None,
    down_proj_bias_grad: nl.ndarray = None,
):
    """MXFP8 backward pass implementation for blockwise dropless MoE.

    Orchestrates the block loop and four matmul phases using MXFP8 quantization.

    Dimensions:
        T: Total number of input tokens (after linearizing across batch dimension)
        H: Hidden dimension size
        I_TP: Intermediate size / tensor parallel degree
        E: Number of experts
        B: Number of tokens per block (block_size)
        N: Total number of blocks

    Args:
        hidden_states_td (TensorDescriptor): [T, H] BF16 input hidden states.
        output_grad_td (TensorDescriptor): [T, H] BF16 upstream gradient.
        gate_up_weight_td (TensorDescriptor): [E*H, 2*I_TP] gate/up weights (reshaped from [E, H, 2, I_TP]).
        down_weight_td (TensorDescriptor): [E*I_TP, H] down weights (reshaped from [E, I_TP, H]).
        token_position_to_id_td (TensorDescriptor): [N*B] token index map.
        block_to_expert_td (TensorDescriptor): [N, 1] expert per block.
        expert_affinities_masked_td (TensorDescriptor): [T*E, 1] affinities.
        gate_up_proj_act_checkpoint_T_td (TensorDescriptor): [N, 2, I_TP, B] checkpoint.
        gate_act_checkpoint_T_td (TensorDescriptor, optional): pre-computed SiLU(gate_pre).
        intermediate_checkpoint_T_td (TensorDescriptor, optional): pre-computed gate_act * up.
        scaled_intermediate_checkpoint_T_td (TensorDescriptor, optional): EA-scaled intermediate.
        down_proj_act_checkpoint_td (TensorDescriptor, optional): down output checkpoint
            (AFFINITY_ON_H only).
        T, H, I_TP, E, N (int): Derived dimensions.
        block_size (int): Tokens per block.
        config (MXFP8MOEBwdConfig): Kernel configuration.
        hidden_states_grad (nl.ndarray): [T, H] output gradient for hidden states.
        expert_affinities_masked_grad (nl.ndarray): [T*E, 1] output affinity gradient.
        gate_up_proj_weight_grad (nl.ndarray): [E, H, 2, I_TP] gate/up weight gradient.
        down_proj_weight_grad (nl.ndarray): [E, I_TP, H] down weight gradient.
        gate_and_up_proj_bias_grad (nl.ndarray, optional): [E, 2, I_TP] gate/up bias gradient.
            None when bias=False.
        down_proj_bias_grad (nl.ndarray, optional): [E, H] down projection bias gradient.
            None when bias=False.

    Returns:
        None. Output gradients are written in-place to the passed-in HBM buffers
        (hidden_states_grad, expert_affinities_masked_grad,
        gate_up_proj_weight_grad, down_proj_weight_grad, and optionally
        gate_and_up_proj_bias_grad, down_proj_bias_grad when bias is enabled).

    Pseudocode:
        TODO: Add pseudocode description
    """
    # Initialize SbufManager
    if get_active_sbm() == None:
        create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope(name="MXFP8 MOE BWD")

    B = block_size
    NUM_B_TILES = B // TILE_M

    # Get sharding info
    _, num_shards, shard_id = get_program_sharding_info()
    H_PER_SHARD = H // num_shards

    # Dense execution writes token gradients directly and initializes weight
    # gradients from block 0 before accumulating any later blocks.
    if not config.skip_grad_initialization and not config.single_expert_dense:
        _initialize_gradient_outputs_shard(
            hidden_states_grad=hidden_states_grad,
            gate_up_proj_weight_grad=gate_up_proj_weight_grad,
            down_proj_weight_grad=down_proj_weight_grad,
            num_shards=num_shards,
            shard_id=shard_id,
            down_proj_bias_grad=down_proj_bias_grad,
            gate_and_up_proj_bias_grad=gate_and_up_proj_bias_grad,
            expert_affinities_masked_grad=expert_affinities_masked_grad,
            sbm=sbm,
        )
        nisa.core_barrier(hidden_states_grad, (0, 1))
        nisa.core_barrier(gate_up_proj_weight_grad, (0, 1))
        nisa.core_barrier(down_proj_weight_grad, (0, 1))
        if gate_and_up_proj_bias_grad is not None:
            nisa.core_barrier(gate_and_up_proj_bias_grad, (0, 1))
        if down_proj_bias_grad is not None:
            nisa.core_barrier(down_proj_bias_grad, (0, 1))

    if config.single_expert_dense:
        expert_idx_bufs = None
        expert_idx_broadcast = None
        iota_vec = None
        token_indices_bufs = None
    else:
        # Bulk-load all expert indices into SBUF.
        expert_idx_bufs = sbm.alloc_stack((1, N), dtype=nl.int32, buffer=nl.sbuf, align=32)
        block_to_expert_2d = block_to_expert_td.data.reshape((1, N))
        nisa.dma_copy(expert_idx_bufs[0, 0:N], block_to_expert_2d[0, 0:N])

        """
        iota_vec: [TILE_M, 1] partition-channel iota (0..127). Used by
        _generate_dynamic_offsets to derive per-partition addresses from a
        broadcast scalar.
        """
        iota_vec = sbm.alloc_stack((TILE_M, 1), dtype=nl.int32, buffer=nl.sbuf, name="iota_vec", align=32)
        nisa.iota(dst=iota_vec, pattern=[[0, 1]], offset=0, channel_multiplier=1)

        """
        Broadcast all expert indices from (1, N) to (TILE_M, N) once, so each
        partition holds the same expert ID for every block. Phase 1's EA gather
        slices column block_idx out of this tensor per block.
        """
        expert_idx_broadcast = sbm.alloc_stack(
            (TILE_M, N), dtype=nl.int32, buffer=nl.sbuf, name="expert_idx_broadcast", align=32
        )
        stream_shuffle_broadcast(src=expert_idx_bufs, dst=expert_idx_broadcast)

        # Double-buffer token indices.
        token_indices_bufs = [
            sbm.alloc_stack((TILE_M, NUM_B_TILES), dtype=nl.int32, align=32),
            sbm.alloc_stack((TILE_M, NUM_B_TILES), dtype=nl.int32, align=32),
        ]

    dims = DimensionSizes(T=T, H=H, B=block_size, E=E, N=N, I_TP=I_TP)
    dims.derive_all_dims()

    if not config.single_expert_dense:
        # Prefetch block 0 token indices.
        _load_token_indices_dgt(token_position_to_id_td.data, 0, B, NUM_B_TILES, dst=token_indices_bufs[0])

    """
    Per-block expert offset for the down weight DGT load is allocated
    inside the block loop (see below) — fresh SBUF per block prevents
    cross-iteration aliasing/reordering hazards. down_weight is reshaped
    to [E*I_TP, H] at the wrapper level; per-expert byte stride in
    elements = I_TP * H, and vector_offset values index the flattened
    DGT view in chunks of size vector_size = MATMUL_TILE_K_PHYSICAL.
    """
    if down_weight_td.scales is None:
        DOWN_EXPERT_STRIDE_IN_VS = (I_TP * H) // MATMUL_TILE_K_PHYSICAL
        DOWN_SCALES_EXPERT_STRIDE = None
    else:
        DOWN_EXPERT_STRIDE_IN_VS = down_weight_td.data.shape[0] // E
        DOWN_SCALES_EXPERT_STRIDE = down_weight_td.scales.shape[0] // E

    # Phase 3 can consume natural [B, 2*I_TP] through the K-by-F PE loader and
    # avoid d_gate_up_T when 2*I_TP is divisible by 512. Routing itself does not
    # prevent this, but routed test shapes such as I_TP=128/384/640 do not meet
    # that loader constraint and still require the explicit transpose.
    p3_lhs_pe = config.single_expert_dense and d_gate_up_t_swizzle_mode == SwizzleMode.PE
    p3_rhs_pe = config.single_expert_dense and hidden_states_t_swizzle_mode == SwizzleMode.PE
    p4_lhs_pe = config.single_expert_dense and output_grad_t_swizzle_mode == SwizzleMode.PE
    p4_rhs_pe = config.single_expert_dense and scaled_intermediate_t_swizzle_mode == SwizzleMode.PE

    # Cached quantized weights become the primary operands after dense block 0.
    # Routed spill buffers remain phase-local because experts can change.
    down_weightq_td = None
    gate_up_weightq_td = None
    cache_down_weight = config.single_expert_dense and N > 1 and not down_weight_td.is_quantized
    cache_gate_up_weight = config.single_expert_dense and N > 1 and not gate_up_weight_td.is_quantized

    # Block Loop
    for block_idx in range(N):
        sbm.open_scope(name=f"Block {block_idx}")
        accumulate_weight_grad = not config.single_expert_dense or block_idx > 0
        if config.single_expert_dense:
            expert_idx = None
            block_token_pos_to_id_full = None
        else:
            expert_idx = expert_idx_bufs[0, block_idx]
            cur = block_idx % 2
            block_token_pos_to_id_full = token_indices_bufs[cur]

            _set_expert_offset_on_td(
                td=down_weight_td,
                expert_idx_broadcast=expert_idx_broadcast,
                block_idx=block_idx,
                expert_stride=DOWN_EXPERT_STRIDE_IN_VS,
                scales_stride=DOWN_SCALES_EXPERT_STRIDE,
                effective_f_dim=I_TP if down_weight_td.scales is None else H // 4,
                name_prefix="down",
                sbm=sbm,
            )

        """
        Per-block HBM transpose scratch. A dense operand configured with PE
        swizzle stays in natural [B, F] layout and skips its scratch allocation.
        """
        if p3_rhs_pe:
            hidden_states_block_T = None
        else:
            hidden_states_block_T = nl.ndarray(
                (H, B),
                dtype=config.compute_dtype,
                buffer=nl.shared_hbm,
                name=f"hidden_states_block_T_shared_block_{block_idx}",
            )
        if p4_lhs_pe:
            output_grad_block_T = None
        else:
            output_grad_block_T = nl.ndarray(
                (H, B),
                dtype=config.compute_dtype,
                buffer=nl.shared_hbm,
                name=f"output_grad_block_T_shared_block_{block_idx}",
            )
        if p4_rhs_pe:
            scaled_intermediate_T = None
        else:
            scaled_intermediate_T = nl.ndarray(
                (I_TP, B),
                dtype=config.compute_dtype,
                buffer=nl.shared_hbm,
                name=f"scaled_intermediate_T_shared_block_{block_idx}",
            )

        # Prefetch next block's token indices
        if not config.single_expert_dense and block_idx < N - 1:
            nxt = (block_idx + 1) % 2
            _load_token_indices_dgt(
                token_position_to_id_td.data, block_idx + 1, B, NUM_B_TILES, dst=token_indices_bufs[nxt]
            )

        if config.single_expert_dense:
            token_start = block_idx * B
            hidden_states_block = hidden_states_td.data[nl.ds(token_start, B), 0:H]
            output_grad_block = output_grad_td.data[nl.ds(token_start, B), 0:H]
            if not p3_rhs_pe:
                _load_contiguous_block_transposed(
                    src=hidden_states_block,
                    dst=hidden_states_block_T,
                    B=B,
                    feature_dim=H,
                    sbm=sbm,
                    transpose_mode=config.phase3_transpose_mode,
                )
            if not p4_lhs_pe:
                _load_contiguous_block_transposed(
                    src=output_grad_block,
                    dst=output_grad_block_T,
                    B=B,
                    feature_dim=H,
                    sbm=sbm,
                    transpose_mode=config.phase4_transpose_mode,
                )
        else:
            # --- Phase 3 RHS: hidden states, transposed to [H, B] ---
            _gather_block_tokens_transposed(
                src=hidden_states_td.data,
                dst=hidden_states_block_T,
                token_indices=block_token_pos_to_id_full,
                B=B,
                feature_dim=H,
                skip_dma=config.skip_dma,
                sbm=sbm,
                transpose_mode=config.phase3_transpose_mode,
            )

            # --- Gather + transpose output grad for Phase 4 LHS ---
            _gather_block_tokens_transposed(
                src=output_grad_td.data,
                dst=output_grad_block_T,
                token_indices=block_token_pos_to_id_full,
                B=B,
                feature_dim=H,
                skip_dma=config.skip_dma,
                sbm=sbm,
                transpose_mode=config.phase4_transpose_mode,
            )

        """
        TODO: current version of compiler/nki is giving issues when TILES_IN_LOAD_M!=4, hence we cannot use:
        load_tile_PE_swizzle_wrapX currently.
        Instead of indirect DMA inside Phase 1, gather here and pass as
        a plain F-by-K tensor so Phase 1 uses the conventional DGT path.
        """

        if not config.single_expert_dense:
            output_grad_block = nl.ndarray(
                (B, H),
                dtype=config.compute_dtype,
                buffer=nl.shared_hbm,
                name=f"output_grad_block_shared_block_{block_idx}",
            )
            _gather_block_tokens(
                src=output_grad_td.data,
                dst=output_grad_block,
                token_indices=block_token_pos_to_id_full,
                B=B,
                feature_dim=H,
                skip_dma=config.skip_dma,
                sbm=sbm,
            )
        output_grad_block_td = TensorDescriptor(
            data=output_grad_block,
            swizzle_mode=output_grad_swizzle_mode,
            fast_dma_transpose=config.fast_dma_transpose,
        )

        # Phase 1: Down projection output grad + SwiGLU backward
        """
        Phase 1 always produces these natural-layout HBM outputs:
          - d_gate_up [B, 2, I_TP]: Phase 2 LHS (K=I_TP).
          - scaled_intermediate   [B, I_TP]:    Phase 4 RHS source.
        Phase 1 also produces d_gate_up_T [2*I_TP, B] unless dense Phase 3
        selects PE swizzle and consumes d_gate_up as K-by-F.
        """

        (
            d_gate_up,
            scaled_intermediate,
            d_gate_up_T,
            down_weightq_td,
        ) = _compute_phase1_down_proj_output_grad_mxfp8(
            output_grad_td=output_grad_block_td,
            down_weight_td=down_weightq_td if down_weightq_td is not None else down_weight_td,
            gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T_td.data,
            block_idx=block_idx,
            expert_idx=expert_idx,
            B=B,
            H=H,
            I_TP=I_TP,
            E=E,
            shard_id=shard_id,
            num_shards=num_shards,
            blocking=config.phase1_config,
            config=config,
            sbm=sbm,
            clamp_limits=config.clamp_limits,
            block_token_pos_to_id_full=block_token_pos_to_id_full,
            expert_affinities_masked=expert_affinities_masked_td.data,
            expert_affinities_masked_grad=expert_affinities_masked_grad,
            expert_idx_broadcast=expert_idx_broadcast,
            scaled_intermediate_checkpoint_T_td=scaled_intermediate_checkpoint_T_td,
            cache_weight=cache_down_weight,
            store_d_gate_up_transpose=not p3_lhs_pe,
        )

        # Phase 2: Hidden states gradient (scatter to output)
        """
        Use combined d_gate_up[B, 2*I_TP] @ gate_up_weight[E*H, 2*I_TP].T → [B, H]
        This computes d_gate @ W_gate.T + d_up @ W_up.T in a single matmul.
        """
        d_gate_up_2d_td = TensorDescriptor(
            data=d_gate_up.reshape((B, 2 * I_TP)),
            swizzle_mode=d_gate_up_swizzle_mode,
            quant_scheme=config.phase2_config.quant_scheme,
            fast_dma_transpose=config.fast_dma_transpose,
        )

        """
        Per-expert indexing for gate_up_weight [E*H, 2*I_TP]:
        scalar_offset is in vector_size units (MATMUL_TILE_K_PHYSICAL = 128).
        Per-expert stride in elements = H * 2 * I_TP.
        Per-expert stride in vector_size units = H * 2 * I_TP / MATMUL_TILE_K_PHYSICAL.
        Must be float32 for nisa.tensor_scalar compatibility in set_vector_offset.
        """
        if gate_up_weight_td.scales is None:
            GATE_UP_EXPERT_STRIDE_IN_VS = (H * 2 * I_TP) // MATMUL_TILE_K_PHYSICAL
            GATE_UP_SCALES_EXPERT_STRIDE = None
        else:
            GATE_UP_EXPERT_STRIDE_IN_VS = gate_up_weight_td.data.shape[0] // E
            GATE_UP_SCALES_EXPERT_STRIDE = gate_up_weight_td.scales.shape[0] // E

        if not config.single_expert_dense:
            _set_expert_offset_on_td(
                td=gate_up_weight_td,
                expert_idx_broadcast=expert_idx_broadcast,
                block_idx=block_idx,
                expert_stride=GATE_UP_EXPERT_STRIDE_IN_VS,
                scales_stride=GATE_UP_SCALES_EXPERT_STRIDE,
                effective_f_dim=H if gate_up_weight_td.scales is None else 2 * I_TP // 4,
                name_prefix="gate_up",
                sbm=sbm,
            )

        gate_up_weightq_td = _compute_phase2_hidden_states_grad_mxfp8(
            d_gate_up_td=d_gate_up_2d_td,
            gate_up_weight_td=gate_up_weightq_td if gate_up_weightq_td is not None else gate_up_weight_td,
            hidden_states_grad=hidden_states_grad,
            block_token_pos_to_id_full=block_token_pos_to_id_full,
            B=B,
            H=H,
            I_TP=I_TP,
            shard_id=shard_id,
            num_shards=num_shards,
            blocking=config.phase2_config,
            config=config,
            sbm=sbm,
            block_token_offset=block_idx * B,
            cache_weight=cache_gate_up_weight,
        )

        # Phase 3: Gate/up weight gradient (accumulate)
        if p3_lhs_pe:
            phase3_lhs_td = TensorDescriptor(
                data=d_gate_up.reshape((B, 2 * I_TP)),
                is_f_by_k=False,
                swizzle_mode=SwizzleMode.PE,
                quant_scheme=QuantScheme.WRAPX,
            )
        else:
            phase3_lhs_td = TensorDescriptor(
                data=d_gate_up_T,
                swizzle_mode=d_gate_up_t_swizzle_mode,
                quant_scheme=config.phase3_config.quant_scheme,
                fast_dma_transpose=config.fast_dma_transpose,
            )
        if p3_rhs_pe:
            phase3_rhs_td = TensorDescriptor(
                data=hidden_states_block,
                is_f_by_k=False,
                swizzle_mode=SwizzleMode.PE,
                quant_scheme=QuantScheme.WRAPX,
            )
        else:
            phase3_rhs_td = TensorDescriptor(
                data=hidden_states_block_T,
                swizzle_mode=hidden_states_t_swizzle_mode,
                fast_dma_transpose=config.fast_dma_transpose,
            )

        _compute_phase3_gate_up_weight_grad_mxfp8(
            lhs_td=phase3_lhs_td,
            rhs_td=phase3_rhs_td,
            gate_up_proj_weight_grad=gate_up_proj_weight_grad,
            expert_idx=expert_idx,
            B=B,
            H=H,
            I_TP=I_TP,
            shard_id=shard_id,
            num_shards=num_shards,
            blocking=config.phase3_config,
            config=config,
            sbm=sbm,
            accumulate_weight_grad=accumulate_weight_grad,
        )

        # Phase 4: Down projection weight gradient (accumulate)
        """
        Phase 4 normally prepares scaled_intermediate_T [I_TP, B] for an
        F-by-K load. Dense PE mode consumes scaled_intermediate [B, I_TP]
        directly through the wrapX K-by-F loader.
        """
        if not p4_rhs_pe:
            _transpose_block_to_hbm(
                src=scaled_intermediate,
                dst=scaled_intermediate_T,
                B=B,
                feature_dim=I_TP,
                shard_offset=0,
                sbm=sbm,
                transpose_mode=config.phase4_transpose_mode,
            )
            nisa.core_barrier(scaled_intermediate_T, (0, 1))

        if not p4_lhs_pe:
            nisa.core_barrier(output_grad_block_T, (0, 1))
        if p4_lhs_pe:
            phase4_lhs_td = TensorDescriptor(
                data=output_grad_block,
                is_f_by_k=False,
                swizzle_mode=SwizzleMode.PE,
                quant_scheme=QuantScheme.WRAPX,
            )
        else:
            phase4_lhs_td = TensorDescriptor(
                data=output_grad_block_T,
                swizzle_mode=output_grad_t_swizzle_mode,
                fast_dma_transpose=config.fast_dma_transpose,
            )
        if p4_rhs_pe:
            phase4_rhs_td = TensorDescriptor(
                data=scaled_intermediate,
                is_f_by_k=False,
                swizzle_mode=SwizzleMode.PE,
                quant_scheme=QuantScheme.WRAPX,
            )
        else:
            phase4_rhs_td = TensorDescriptor(
                data=scaled_intermediate_T,
                swizzle_mode=scaled_intermediate_t_swizzle_mode,
                quant_scheme=config.phase4_config.quant_scheme,
                fast_dma_transpose=config.fast_dma_transpose,
            )

        _compute_phase4_down_weight_grad_mxfp8(
            lhs_td=phase4_lhs_td,
            rhs_td=phase4_rhs_td,
            down_proj_weight_grad=down_proj_weight_grad,
            expert_idx=expert_idx,
            B=B,
            H=H,
            I_TP=I_TP,
            shard_id=shard_id,
            num_shards=num_shards,
            blocking=config.phase4_config,
            config=config,
            sbm=sbm,
            accumulate_weight_grad=accumulate_weight_grad,
        )

        # Bias gradients: reduce over B dimension and accumulate per-expert.
        # The down-bias gradient must be affinity-weighted before the token
        # reduction (AFFINITY_ON_I). Pass the global scattered gradient plus the
        # block token indices so the helper's indirect path gathers per-token rows
        # and applies the per-token affinity, matching the BF16 caller.
        if down_proj_bias_grad != None:
            _compute_down_proj_bias_grad(
                down_proj_output_grad_hbm=output_grad_td.data,
                down_proj_bias_grad=down_proj_bias_grad,
                expert_idx=expert_idx,
                B_DIM=B,
                H_DIM=H,
                num_shards=num_shards,
                shard_id=shard_id,
                dtype=config.compute_dtype,
                sbm=sbm,
                block_token_pos_to_id_full=block_token_pos_to_id_full,
                skip_dma=config.skip_dma,
                block_idx=block_idx,
                iota_vec=iota_vec,
                expert_idx_broadcast=expert_idx_broadcast,
                expert_affinities_masked=expert_affinities_masked_td.data,
                E=E,
            )

        if gate_and_up_proj_bias_grad != None:
            _compute_gate_up_proj_bias_grad(
                gate_up_proj_output_grad_hbm=d_gate_up.reshape((B, 2, I_TP)),
                gate_and_up_proj_bias_grad=gate_and_up_proj_bias_grad,
                expert_idx=expert_idx,
                B=B,
                I_TP=I_TP,
                shard_id=shard_id,
                dtype=config.compute_dtype,
                sbm=sbm,
                block_idx=block_idx,
                iota_vec=iota_vec,
                expert_idx_broadcast=expert_idx_broadcast,
            )

        sbm.close_scope()

    sbm.close_scope()


# Helper Functions: Token Gather/Scatter and Transpose


def _gather_block_tokens(src, dst, token_indices, B, feature_dim, skip_dma, sbm):
    """Gather tokens from src (non-transposed): src[token_ids, :] → dst[B, feature_dim].

    Args:
        src (nl.ndarray): [T, feature_dim], Source tensor in HBM.
        dst (nl.ndarray): [B, feature_dim], Destination tensor in HBM.
        token_indices (nl.ndarray): [TILE_M, NUM_B_TILES], Token indices in SBUF.
        B (int): Block size.
        feature_dim (int): Feature dimension (H).
        skip_dma (SkipMode): OOB handling.
        sbm (SbufManager): SBUF memory manager.
    """
    NUM_B_TILES = div_ceil(B, TILE_M)
    NUM_F_TILES = div_ceil(feature_dim, TILE_N)

    for b_tile_idx in range(NUM_B_TILES):
        b_off = b_tile_idx * TILE_M
        actual_b = min(TILE_M, B - b_off)
        vec_ap = token_indices.ap(
            [[NUM_B_TILES, actual_b], [1, 1]],
            offset=b_tile_idx,
        )

        for f_tile_idx in range(NUM_F_TILES):
            f_off = f_tile_idx * TILE_N
            actual_f = min(TILE_N, feature_dim - f_off)
            tile = sbm.alloc_stack((actual_b, actual_f), dtype=src.dtype, buffer=nl.sbuf)

            if skip_dma.skip_token:
                nisa.memset(tile, value=0)
            nisa.dma_copy(
                dst=tile,
                src=src.ap(
                    pattern=[[feature_dim, actual_b], [1, actual_f]],
                    offset=f_off,
                    vector_offset=vec_ap,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )
            nisa.dma_copy(
                dst=dst[b_off : b_off + actual_b, f_off : f_off + actual_f],
                src=tile,
            )


def _gather_block_tokens_transposed(src, dst, token_indices, B, feature_dim, skip_dma, sbm, transpose_mode=None):
    """Gather tokens from src and store transposed: src[token_ids, :].T → dst[feature_dim, B].

    Uses dma_copy with int32 vector offsets to gather rows, then selects the
    configured transpose engine before storing to transposed HBM.

    Args:
        src (nl.ndarray): [T, feature_dim], Source tensor in HBM.
        dst (nl.ndarray): [feature_dim, B], Destination transposed tensor in HBM.
        token_indices (nl.ndarray): [TILE_M, NUM_B_TILES], Token indices in SBUF.
        B (int): Block size.
        feature_dim (int): Feature dimension (H).
        skip_dma (SkipMode): OOB handling.
        sbm (SbufManager): SBUF memory manager.
        transpose_mode (TransposeMode): Transpose engine.
    """
    NUM_B_TILES = div_ceil(B, TILE_M)
    NUM_F_TILES = div_ceil(feature_dim, TILE_M)
    use_dma = transpose_mode == TransposeMode.DMA

    for b_tile_idx in range(NUM_B_TILES):
        actual_b = min(TILE_M, B - b_tile_idx * TILE_M)
        vec_ap = token_indices.ap(
            [[NUM_B_TILES, actual_b], [1, 1]],
            offset=b_tile_idx,
        )

        col_offset = b_tile_idx * TILE_M
        for f_tile_idx in range(NUM_F_TILES):
            f_off = f_tile_idx * TILE_M
            actual_f = min(TILE_M, feature_dim - f_off)

            # Step 1: Gather [actual_b, actual_f] using dma_copy (int32 vector_offset)
            tile = sbm.alloc_stack((actual_b, actual_f), dtype=src.dtype, buffer=nl.sbuf)
            if skip_dma.skip_token:
                nisa.memset(tile, value=0)
            nisa.dma_copy(
                dst=tile,
                src=src.ap(
                    pattern=[[feature_dim, actual_b], [1, actual_f]],
                    offset=f_off,
                    vector_offset=vec_ap,
                    indirect_dim=0,
                ),
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )

            tile_t = sbm.alloc_stack((actual_f, actual_b), dtype=src.dtype, buffer=nl.sbuf)
            if use_dma:
                nisa.dma_transpose(dst=tile_t, src=tile)
            else:
                tile_t_psum = nl.ndarray((actual_f, actual_b), dtype=src.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=tile_t_psum, data=tile)
                nisa.tensor_copy(dst=tile_t, src=tile_t_psum, engine=nisa.scalar_engine)

            nisa.dma_copy(
                dst=dst[f_off : f_off + actual_f, col_offset : col_offset + actual_b],
                src=tile_t,
            )


def _load_contiguous_block_transposed(src, dst, B, feature_dim, sbm, transpose_mode=None):
    """Load contiguous [B, feature_dim] tiles and store them as [feature_dim, B]."""
    NUM_B_TILES = div_ceil(B, TILE_M)
    NUM_F_TILES = div_ceil(feature_dim, TILE_M)
    use_dma = transpose_mode == TransposeMode.DMA

    for b_tile_idx in range(NUM_B_TILES):
        b_off = b_tile_idx * TILE_M
        actual_b = min(TILE_M, B - b_off)
        for f_tile_idx in range(NUM_F_TILES):
            f_off = f_tile_idx * TILE_M
            actual_f = min(TILE_M, feature_dim - f_off)
            tile = sbm.alloc_stack((actual_b, actual_f), dtype=src.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=tile,
                src=src[b_off : b_off + actual_b, f_off : f_off + actual_f],
            )

            tile_t = sbm.alloc_stack((actual_f, actual_b), dtype=src.dtype, buffer=nl.sbuf)
            if use_dma:
                nisa.dma_transpose(dst=tile_t, src=tile)
            else:
                tile_t_psum = nl.ndarray((actual_f, actual_b), dtype=src.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=tile_t_psum, data=tile)
                nisa.tensor_copy(dst=tile_t, src=tile_t_psum, engine=nisa.scalar_engine)

            nisa.dma_copy(
                dst=dst[f_off : f_off + actual_f, b_off : b_off + actual_b],
                src=tile_t,
            )


def _transpose_block_to_hbm(src, dst, B, feature_dim, shard_offset, sbm, transpose_mode=None):
    """Transpose a [B, feature_dim] block to [feature_dim, B] in HBM.

    transpose_mode selects the mechanism:
      NC  (default): load [B,F]->SBUF, nc_transpose->PSUM, copy PSUM->SBUF, store [F,B]->HBM.
                     Uses the PE array for the transpose (burns PE cycles).
      DMA          : load [B,F]->SBUF, then a SINGLE dma_transpose writes [F,B]->HBM on the
                     DMA engine — no nc_transpose, no PSUM hop. Moves the transpose off the
                     PE onto the (often idle) DMA engine.

    Args:
        src (nl.ndarray): [B, feature_dim], Source block in HBM.
        dst (nl.ndarray): [feature_dim, B], Destination transposed block in HBM.
        B (int): Block size.
        feature_dim (int): Feature dimension.
        shard_offset (int): Offset for sharding (unused currently).
        sbm (SbufManager): SBUF memory manager.
        transpose_mode (TransposeMode): Transpose engine.
    """
    NUM_B_TILES = div_ceil(B, TILE_M)
    NUM_F_TILES = div_ceil(feature_dim, TILE_M)
    use_dma = transpose_mode == TransposeMode.DMA

    for b_tile_idx in range(NUM_B_TILES):
        b_off = b_tile_idx * TILE_M
        actual_b = min(TILE_M, B - b_off)
        row_tile = sbm.alloc_stack((actual_b, feature_dim), dtype=src.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=row_tile, src=src[b_off : b_off + actual_b, 0:feature_dim])
        for f_tile_idx in range(NUM_F_TILES):
            f_off = f_tile_idx * TILE_M
            actual_f = min(TILE_M, feature_dim - f_off)
            sub = row_tile[:, f_off : f_off + actual_f]

            if use_dma:
                # DMA-engine transpose OFF the PE. dma_transpose requires an SBUF dst, so
                # transpose SBUF->SBUF then dma_copy the [f,b] tile to HBM. No PE, no PSUM.
                tile_t = sbm.alloc_stack((actual_f, actual_b), dtype=src.dtype, buffer=nl.sbuf)
                nisa.dma_transpose(dst=tile_t, src=sub)
                nisa.dma_copy(dst=dst[f_off : f_off + actual_f, b_off : b_off + actual_b], src=tile_t)
            else:
                # NC path: transpose on PE (PSUM dst routes to PE; SBUF dst routes to DVE
                # which caps at 32x32). Tiles up to 128x128 use PE.
                tile_t_psum = nl.ndarray((actual_f, actual_b), dtype=src.dtype, buffer=nl.psum)
                nisa.nc_transpose(dst=tile_t_psum, data=sub)
                # PSUM -> SBUF before DMA (HBM cannot read directly from PSUM). Alternate the
                # copy across scalar/vector engines to relieve the scalar-COPY bottleneck.
                tile_t = sbm.alloc_stack((actual_f, actual_b), dtype=src.dtype, buffer=nl.sbuf)
                # TODO: Use the base nc_transpose wrapper's ratio-based Scalar/Vector drain selection.
                _cp_eng = nisa.scalar_engine if (f_tile_idx % 2 == 0) else nisa.vector_engine
                nisa.tensor_copy(dst=tile_t, src=tile_t_psum, engine=_cp_eng)
                # Store to dst
                nisa.dma_copy(dst=dst[f_off : f_off + actual_f, b_off : b_off + actual_b], src=tile_t)
