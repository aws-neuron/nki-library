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

"""MXFP8 forward pass implementation for blockwise dropless MoE (shard-on-block).

This is the device-side realization of the forward MoE FFN that the MXFP8 MoE
backward (``moe_mxfp8/bwd/``) consumes. It is built from the same training
helpers as the dense MLP forward (``generic_matmul_mxfp8_api`` +
``TensorDescriptor``) and reuses the backward's MoE orchestration primitives
(``_load_token_indices_dgt``, ``_set_expert_offset_on_td``,
``_gather_block_tokens`` and the EA gather).

Per block (one expert, ``B`` tokens):

    hidden_block = gather(hidden_states, token_ids)             # [B, H]
    gate         = hidden_block @ W_gate[e].T                   # [B, I_TP]
    up           = hidden_block @ W_up[e].T                     # [B, I_TP]
    (clamp gate/up, then checkpoint gate/up pre-activations)
    intermediate = SiLU(gate) * up                             # [B, I_TP]
    scaled       = intermediate * affinity[token, e]           # [B, I_TP]  (AFFINITY_ON_I)
    (checkpoint scaled intermediate, transposed)
    out_block    = scaled @ W_down[e].T                        # [B, H]
    scatter_add(output, out_block, token_ids)                  # -> output[T, H]

Sharding is SHARD_ON_BLOCK: each of the ``num_shards`` cores processes the
strided block subset ``range(shard_id, N, num_shards)`` end-to-end into its own
output slab ``output[shard_id]``; the slabs are summed by ``_reduce_output_shards``.

Activations stay BF16; only weights are MXFP8. Mode: AFFINITY_ON_I, SiLU, E4M3,
LNC2, SHARD_ON_BLOCK.

TODO(moe-fwd): phase bodies below are scaffolded with the exact reused helpers
named in their docstrings; fill them following the milestone plan (M1 single
block -> M2 multi-block -> M3 checkpoints -> M4 round-trip -> M5 features).
"""

import nki  # noqa: F401
import nki.isa as nisa  # noqa: F401
import nki.language as nl
from nki.isa.constants import dge_mode, oob_mode  # noqa: F401

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil, get_program_sharding_info  # noqa: F401
from ....core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ...matmul_mxfp8.matmul_mxfp8_generic_api import generic_matmul_mxfp8_api  # noqa: F401
from ...mlp_mxfp8.common_utils import (  # noqa: F401
    L_TILE_K,
    MATMUL_TILE_K_PHYSICAL,
    MAX_TILES_IN_LOAD_M,
    TILE_M,
    TILE_N,
    _allocate_spill_buffer,
    _build_matmul_params,
    _compute_load_tile_shape,
    apply_activation_clamp,
    build_tile_sizes,
)
from ...moe.bwd.bwmm_bwd_dropless import _generate_dynamic_offsets  # noqa: F401
from ...mxfp_utils.mxfp8_utils.common_dataclasses import TensorDescriptor
from ...mxfp_utils.mxfp8_utils.common_utils import create_and_set_active_sbm, get_active_sbm
from ...mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import MX_PARTITION_SIZE, TILE_SIZE_GEMM_MOVING_MAX

# Reuse the backward's block-orchestration primitives verbatim — same indirect
# DMA / DGT / per-expert offset semantics, so fwd and bwd stay byte-compatible.
from ..bwd.bwmm_bwd_dropless_mxfp8 import (
    _gather_block_tokens,
    _gather_block_tokens_transposed,  # noqa: F401
    _load_token_indices_dgt,
    _set_expert_offset_on_td,
)

# =====================================================================================
# Per-block phase helpers — each owns one stage of the per-block FFN. They take and
# return plain nl.ndarray / TensorDescriptor (no tensor-bearing dataclasses across
# boundaries) so they compose inside the traced block loop.
# =====================================================================================


def _resolve_tiles_in_load(configured_load, tiles_in_block, fallback):
    """Resolve a config's TILES_IN_LOAD_* against the post-clamp block size.

    ``auto_generate_default`` derives TILES_IN_LOAD_* as a DGT-legal divisor of
    the (pre-clamp) TILES_IN_BLOCK_*, but the phase bodies clamp TILES_IN_BLOCK_*
    down to the real tile count. ``generic_matmul_mxfp8_api`` integer-divides the
    block extent by the load factor (see load_block.load_lhs_and_rhs), so the
    effective load must still divide the clamped block and not exceed it. Reduce
    to the largest divisor of the clamped block that is <= the configured load
    (already DGT-capped by auto-gen) so the result stays DGT-legal.

    ``fallback`` is used only when the config carries no value (auto-gen skipped);
    it reproduces the previous hardcoded behavior exactly.
    """
    load = configured_load if configured_load else fallback
    load = min(load, tiles_in_block)
    while load > 1 and tiles_in_block % load != 0:
        load -= 1
    return max(1, load)


def _resolve_phase_tiles(phase_config):
    """Build the matmul ``tiles`` dict for one GEMM phase from its config.

    Uses the config's ``tile_m`` / ``tile_k`` / ``tile_n`` (populated by
    ``auto_generate_default`` — from the autotune cache on a hit, else the
    heuristic) when set, and falls back to the pinned defaults
    (``TILE_M`` / ``L_TILE_K`` / ``TILE_N``) when a field is None.

    Two invariants are enforced because the forward drives
    ``generic_matmul_mxfp8_api`` directly and so bypasses ``validate_shapes``:

      - ``tile_m`` must stay ``TILE_M`` (128): the M dimension partitions the
        matmul accumulator (pmax = 128) and the affinity-fold epilogue indexes the
        EA columns by ``TILE_M`` (``b_tile = m_off // TILE_M``).
      - physical K (``tile_k // INTERLEAVE_FACTOR``) must be a legal MX contraction
        (<= 128 and a multiple of ``MX_PARTITION_SIZE``); e.g. ``tile_k`` must not
        collapse to 384 -> physical K 96, which matmul_mx rejects. The spill
        buffers also size K by the module ``MATMUL_TILE_K_PHYSICAL`` constant, so
        ``tile_k`` must stay ``L_TILE_K`` (512 -> physical 128).

    ``tile_n`` is free: it only needs ``tile_n % MX_PARTITION_SIZE == 0`` and
    ``tile_n <= TILE_SIZE_GEMM_MOVING_MAX`` — a moving-dim tile may span multiple
    PSUM banks on gen4 (e.g. a tuned ``tile_n`` of 768).
    """
    tile_m = phase_config.tile_m if phase_config.tile_m else TILE_M
    l_tile_k = phase_config.tile_k if phase_config.tile_k else L_TILE_K
    tile_n = phase_config.tile_n if phase_config.tile_n else TILE_N

    kernel_assert(
        tile_m == TILE_M,
        f"moe fwd requires tile_m == {TILE_M} (got {tile_m})",
    )
    kernel_assert(
        l_tile_k == L_TILE_K,
        f"moe fwd requires tile_k == {L_TILE_K} so physical K stays a legal 128 (got {l_tile_k})",
    )
    kernel_assert(
        tile_n % MX_PARTITION_SIZE == 0 and tile_n <= TILE_SIZE_GEMM_MOVING_MAX,
        f"moe fwd tile_n ({tile_n}) must be a multiple of {MX_PARTITION_SIZE} and <= {TILE_SIZE_GEMM_MOVING_MAX}",
    )
    return build_tile_sizes(tile_m=tile_m, l_tile_k=l_tile_k, tile_n=tile_n)


def _gather_hidden_block(hidden_states_td, token_indices, B, H, skip_dma, block_idx, config, sbm):
    """Step 3 — indirect-gather this block's tokens into a dense [B, H] HBM tile.

    Reuses :func:`_gather_block_tokens` (indirect ``dma_copy`` with
    ``vector_offset`` + ``oob_mode.skip`` for ``-1`` padding). The result is
    wrapped in a fresh per-block ``TensorDescriptor`` (unswizzled BF16) to feed
    the gate/up GEMMs as the LHS.

    Returns:
        TensorDescriptor: [B, H] gathered hidden block (is_f_by_k defaults).
    """
    buffer_dtype = config.compute_dtype
    hidden_block = nl.ndarray(
        (B, H),
        dtype=buffer_dtype,
        buffer=nl.shared_hbm,
        name=f"fwd_hidden_block_{block_idx}",
    )
    _gather_block_tokens(
        src=hidden_states_td.data,
        dst=hidden_block,
        token_indices=token_indices,
        B=B,
        feature_dim=H,
        skip_dma=skip_dma,
        sbm=sbm,
    )
    return TensorDescriptor(data=hidden_block)


def _gather_block_affinities(
    expert_affinities_masked, token_indices, expert_idx_broadcast, block_idx, B, E, skip_dma, sbm
):
    """Step 7a — gather the per-token expert-affinity scalar for this block.

    Mirrors the AFFINITY_ON_I EA pre-load in the backward's Phase 1: build the
    flat address ``token_id * E + expert_idx`` via :func:`_generate_dynamic_offsets`,
    then indirect ``dma_copy`` (``vector_offset`` + ``oob_mode.skip``) from
    ``expert_affinities_masked [T*E, 1]`` into a per-b-tile [TILE_M, 1] fp32 tile.
    The forward only needs the gather (no EA-grad accumulation — that is bwd-only).

    Returns:
        nl.ndarray: [TILE_M, NUM_B_TILES] fp32 EA scalars, column b_tile holds the
        per-token affinities for that b-tile (broadcast-multiplied across I_TP later).
    """
    NUM_B_TILES = div_ceil(B, TILE_M)
    ea_tiles_all = sbm.alloc_stack((TILE_M, NUM_B_TILES), dtype=nl.float32, name=f"fwd_ea_{block_idx}", align=32)
    ea_expert_idx_tensor = expert_idx_broadcast[0:TILE_M, block_idx : block_idx + 1]

    # Per b_tile: build the flat address token_id*E + expert_idx, indirect-gather
    # the per-token affinity scalar. Mirrors bwd Phase-1 EA pre-load
    # (bwmm_bwd_dropless_mxfp8.py:350-392); the forward needs only the gather.
    for b_tile in range(NUM_B_TILES):
        token_off = sbm.alloc_stack((TILE_M, 1), dtype=nl.int32, name=f"fwd_ea_off_{block_idx}_{b_tile}", align=32)
        addr_tmp = sbm.alloc_stack((TILE_M, 1), dtype=nl.int32, name=f"fwd_ea_addr_{block_idx}_{b_tile}", align=32)
        _generate_dynamic_offsets(
            token_indices,
            ea_expert_idx_tensor,
            token_off,
            addr_tmp,
            b_tile,
            skip_dma,
            E,
        )
        ea_dst = sbm.alloc_stack((TILE_M, 1), dtype=nl.float32, name=f"fwd_ea_load_{block_idx}_{b_tile}", align=32)
        if skip_dma.skip_token:
            nisa.memset(ea_dst, value=0.0)
        nisa.dma_copy(
            dst=ea_dst,
            src=expert_affinities_masked.ap(
                pattern=[[expert_affinities_masked.shape[1], TILE_M], [1, 1]],
                offset=0,
                vector_offset=token_off,
                indirect_dim=0,
            ),
            oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
        )
        nisa.tensor_copy(dst=ea_tiles_all[:, b_tile], src=ea_dst)
    return ea_tiles_all


def _load_block_affinities_contiguous(expert_affinities_masked, block_idx, B, sbm):
    """Load per-token affinities for the directly packed single expert.

    The epilogue multiplies SBUF activation tiles by these row scalars, so the
    affinities still need a direct HBM -> SBUF load even though token routing is
    already contiguous.
    """
    NUM_B_TILES = div_ceil(B, TILE_M)
    ea_tiles_all = sbm.alloc_stack((TILE_M, NUM_B_TILES), dtype=nl.float32, name=f"fwd_ea_direct_{block_idx}", align=32)
    block_offset = block_idx * B
    for b_tile_idx in range(NUM_B_TILES):
        b_off = b_tile_idx * TILE_M
        actual_b = min(TILE_M, B - b_off)
        row_off = block_offset + b_off
        nisa.dma_copy(
            dst=ea_tiles_all[0:actual_b, b_tile_idx : b_tile_idx + 1],
            src=expert_affinities_masked[nl.ds(row_off, actual_b), 0:1],
        )
    return ea_tiles_all


def _store_tile_transposed_to_ckpt(src_tile, ckpt, block_idx, half, i_off, m_off, I_TP, B, sbm):
    """Transpose an [actual_m(B), actual_n(I_TP)] SBUF tile and store into a
    [N, 2, I_TP, B] checkpoint at [block_idx, half, i_off:, m_off:].

    nc_transpose (PE) caps at 128x128, so the N (I_TP) dimension is transposed
    in <=128-wide sub-chunks. The checkpoint's B axis is contiguous, so each
    transposed [n_chunk, actual_m] sub-tile is DMA'd to its I_TP rows (stride B).
    """
    block_half_base = (block_idx * 2 * I_TP * B) + (half * I_TP * B)
    _store_tile_transposed_chunked(src_tile, ckpt, block_half_base, i_off, m_off, B, sbm)


def _store_tile_transposed_to_ckpt_2d(src_tile, ckpt, block_idx, i_off, m_off, I_TP, B, sbm):
    """Transpose an [actual_m(B), actual_n(I_TP)] SBUF tile and store into a
    [N, I_TP, B] checkpoint at [block_idx, i_off:, m_off:] (128-wide N sub-chunks)."""
    block_base = block_idx * I_TP * B
    _store_tile_transposed_chunked(src_tile, ckpt, block_base, i_off, m_off, B, sbm)


def _store_tile_transposed_chunked(src_tile, ckpt, block_base, i_off, m_off, B, sbm):
    """Transpose [actual_m, actual_n] -> store as [n, m] rows into a B-contiguous
    checkpoint, chunking N into <=TILE_M (128) pieces for the PE transpose cap.

    ckpt is addressed flat; the I_TP row r at column m maps to
    offset = block_base + r * B + m. block_base already encodes block (and half).
    """
    actual_m = src_tile.shape[0]
    actual_n = src_tile.shape[1]
    NUM_N_CHUNKS = div_ceil(actual_n, TILE_M)
    for nc in range(NUM_N_CHUNKS):
        n0 = nc * TILE_M
        cn = min(TILE_M, actual_n - n0)
        sub = src_tile[0:actual_m, n0 : n0 + cn]
        t_psum = nl.ndarray((cn, actual_m), dtype=src_tile.dtype, buffer=nl.psum)
        nisa.nc_transpose(dst=t_psum, data=sub)
        t_sbuf = sbm.alloc_stack(shape=(cn, actual_m), dtype=src_tile.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=t_sbuf, src=t_psum, engine=nisa.scalar_engine)
        base = block_base + (i_off + n0) * B + m_off
        nisa.dma_copy(
            dst=ckpt.ap(pattern=[[B, cn], [1, actual_m]], offset=base),
            src=t_sbuf,
        )


def _gate_up_swiglu_affinity_block(
    hidden_block_td,
    gate_up_weight_td,
    ea_tiles_all,
    block_idx,
    expert_idx_broadcast,
    B,
    H,
    I_TP,
    E,
    gate_up_proj_act_checkpoint_T,
    scaled_intermediate_checkpoint_T,
    config,
    sbm,
):
    """Steps 4-8 — gate/up GEMMs, clamp+checkpoint, SwiGLU, affinity fold, checkpoint.

    Drops the dense forward's compute body (``mlp_fwd_mxfp8_kernel``) into the
    per-block loop with three substitutions:
      - LHS is the gathered ``hidden_block_td`` ([B, H]), ``lhs_m_offset=0``.
      - RHS is the per-expert ``gate_up_weight_td`` ([E*H, 2*I_TP] reshape); gate
        vs up via ``rhs_n_offset`` = 0 vs I_TP, composed with the expert
        ``scalar_offset`` set by :func:`_set_expert_offset_on_td`.
      - Quantize the gathered hidden tile once (empty ``lhs_sbuf_td =
        TensorDescriptor(is_quantized=True)``) and reuse it across gate & up.

    Ordering (matches the golden + the bwd's checkpoint expectations):
      gate/up GEMM -> (bias) -> clamp via :func:`apply_activation_clamp`
      -> store clamped gate_pre & up to gate_up_proj_act_checkpoint_T[block, {0,1}]
      -> SiLU(gate) -> * up -> * affinity (AFFINITY_ON_I)
      -> store scaled intermediate to scaled_intermediate_checkpoint_T[block]
      -> return the scaled intermediate (HBM [B, I_TP]) for the down GEMM.

    Both checkpoint stores are optional: when ``gate_up_proj_act_checkpoint_T`` or
    ``scaled_intermediate_checkpoint_T`` is None the corresponding transpose+store
    is skipped (the clamp and SwiGLU compute still run, since they feed the FFN).

    Returns:
        TensorDescriptor: [B, I_TP] EA-scaled intermediate, the down-projection LHS.
    """
    # Per-expert weight slice for gate/up (composed with rhs_n_offset for the up half).
    # Forward-natural gate/up weight is [E, 2*I_TP, H] -> 2D [E*2*I_TP, H] (F-by-K,
    # F = 2*I_TP per expert, K = H), so the gate/up GEMM contracts over H. The
    # per-expert F slice is 2*I_TP rows; the up half is the second I_TP of those.
    if gate_up_weight_td.scales is None:
        gate_up_expert_stride_in_vs = (2 * I_TP * H) // MATMUL_TILE_K_PHYSICAL
        gate_up_scales_stride = None
        gate_up_effective_f_dim = 2 * I_TP
    else:
        gate_up_expert_stride_in_vs = gate_up_weight_td.data.shape[0] // E
        gate_up_scales_stride = gate_up_weight_td.scales.shape[0] // E
        gate_up_effective_f_dim = 2 * I_TP // 4
    if not config.no_indirect_load:
        _set_expert_offset_on_td(
            td=gate_up_weight_td,
            expert_idx_broadcast=expert_idx_broadcast,
            block_idx=block_idx,
            expert_stride=gate_up_expert_stride_in_vs,
            scales_stride=gate_up_scales_stride,
            effective_f_dim=gate_up_effective_f_dim,
            name_prefix="fwd_gate_up",
            sbm=sbm,
        )

    # Single source of truth for the activation/checkpoint buffer dtype, so we
    # can't accidentally allocate an activation buffer in the wrong precision.
    # (The matmul accumulators stay explicit nl.float32 — that is accumulation
    # precision, deliberately not the buffer dtype.)
    buffer_dtype = config.compute_dtype

    scaled_intermediate = nl.ndarray(
        (B, I_TP),
        dtype=buffer_dtype,
        buffer=nl.shared_hbm,
        name=f"fwd_scaled_intermediate_{block_idx}",
    )

    clamp = config.clamp_limits

    # Tile sizes for the gate/up GEMM: M->B, N->I_TP (per gate/up half), K->H.
    # Take tile_m/tile_k/tile_n from the phase config (autotune cache or heuristic),
    # defaulting to the pinned TILE_M/L_TILE_K/TILE_N. tile_m/tile_k are held to
    # their defaults so physical K stays a legal 128 and the EA-fold M indexing
    # holds; only tile_n varies (see _resolve_phase_tiles). The real tile counts
    # come from the actual B/I_TP/H below.
    gu_cfg = config.gate_up_config
    tiles = _resolve_phase_tiles(gu_cfg)
    tile_m = tiles['tile_m']
    tile_n = tiles['tile_n']
    l_tile_k = tiles['l_tile_k']

    NUM_M_TILES = div_ceil(B, tile_m)
    NUM_N_TILES = div_ceil(I_TP, tile_n)
    NUM_K_TILES = div_ceil(H, l_tile_k)
    TILES_IN_BLOCK_M = max(1, min(gu_cfg.TILES_IN_BLOCK_M, NUM_M_TILES))
    TILES_IN_BLOCK_N = max(1, min(gu_cfg.TILES_IN_BLOCK_N, NUM_N_TILES))
    TILES_IN_BLOCK_K = max(1, min(gu_cfg.TILES_IN_BLOCK_K, NUM_K_TILES))
    BLOCK_N = TILES_IN_BLOCK_N * tile_n

    # Load factors from the auto-generated config, reconciled with the clamped
    # blocks. Fall back to the previous hardcoded values if auto-gen was skipped.
    TILES_IN_LOAD_M = _resolve_tiles_in_load(
        gu_cfg.TILES_IN_LOAD_M, TILES_IN_BLOCK_M, min(TILES_IN_BLOCK_M, MAX_TILES_IN_LOAD_M)
    )
    TILES_IN_LOAD_N = _resolve_tiles_in_load(gu_cfg.TILES_IN_LOAD_N, TILES_IN_BLOCK_N, 1)

    lhs_load_tile_shape = _compute_load_tile_shape(hidden_block_td, tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(gate_up_weight_td, tiles, tile_n)
    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )

    NUM_M_BLOCKS = div_ceil(NUM_M_TILES, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_N_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)

    # Spill/reload scratch (fresh per block, per the dense-fwd Phase-1 pattern in
    # mlp_fwd_mxfp8_kernel.py). The gate and up GEMMs share ONE quantized-hidden
    # LHS buffer (hiddenq_td) — that shared reuse is the point of spilling — and
    # each gets its own weight RHS buffer. Only BF16 operands are spilled: the
    # `is_quantized` guard skips a buffer when that operand arrives pre-quantized
    # (weights are gated non-prequant today, so the guards are always true here).
    # data_buffer is private_hbm (LNC2 is asserted on).
    hiddenq_td = None
    gate_wq_td = None
    up_wq_td = None
    if gu_cfg.spill_reload:
        data_buffer = nl.private_hbm
        if not hidden_block_td.is_quantized:
            hiddenq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_M_BLOCKS,
                block_f_logical=bd.BLOCK_M_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=gu_cfg.enable_scale_packing,
                data_buffer=data_buffer,
            )
        if not gate_up_weight_td.is_quantized:
            gate_wq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=gu_cfg.enable_scale_packing,
                data_buffer=data_buffer,
            )
            up_wq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=gu_cfg.enable_scale_packing,
                data_buffer=data_buffer,
            )

    for m_block_idx in nl.sequential_range(NUM_M_BLOCKS):
        m_block_start = m_block_idx * TILES_IN_BLOCK_M
        for n_block_idx in range(NUM_N_BLOCKS):
            n_block_start = n_block_idx * TILES_IN_BLOCK_N
            acc_cols = TILES_IN_BLOCK_M * BLOCK_N
            gate_sbuf = sbm.alloc_stack(shape=(tile_m, acc_cols), dtype=nl.float32, buffer=nl.sbuf)
            up_sbuf = sbm.alloc_stack(shape=(tile_m, acc_cols), dtype=nl.float32, buffer=nl.sbuf)
            gate_output_td = TensorDescriptor(data=gate_sbuf)
            up_output_td = TensorDescriptor(data=up_sbuf)

            for k_block_idx in nl.sequential_range(NUM_K_BLOCKS):
                # Empty TD: the gate call quantizes hidden and fills it; the up
                # call reuses the same quantized hidden tile (gate/up share LHS).
                hidden_sbuf_td = TensorDescriptor(is_quantized=True)

                # Gate GEMM: hidden_block[B, H] @ W_gate[H, I_TP] -> [B, I_TP]
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=hidden_block_td,
                    rhs_hbm_td=gate_up_weight_td,
                    bd=bd,
                    output_td=gate_output_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_sbuf_td=hidden_sbuf_td,
                    lhs_m_offset=0,
                    rhs_n_offset=0,
                    TILES_IN_LOAD_M=TILES_IN_LOAD_M,
                    TILES_IN_LOAD_N=TILES_IN_LOAD_N,
                    lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                    rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                    lhs_load_tile_shape=lhs_load_tile_shape or tiles['lhs_load_tile'],
                    rhs_load_tile_shape=rhs_load_tile_shape or tiles['rhs_load_tile'],
                    lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                    rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                    spill_reload=gu_cfg.spill_reload,
                    lhsq_td=hiddenq_td,
                    rhsq_td=gate_wq_td,
                    use_scale_packing=gu_cfg.enable_scale_packing,
                    initialize_accumulator=(k_block_idx == 0),
                )

                # Up GEMM: reuses the quantized hidden; rhs_n_offset=I_TP selects
                # the up half of the per-expert F slice.
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=hidden_block_td,
                    rhs_hbm_td=gate_up_weight_td,
                    bd=bd,
                    output_td=up_output_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_sbuf_td=hidden_sbuf_td,
                    lhs_m_offset=0,
                    rhs_n_offset=I_TP,
                    TILES_IN_LOAD_M=TILES_IN_LOAD_M,
                    TILES_IN_LOAD_N=TILES_IN_LOAD_N,
                    lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                    rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                    lhs_load_tile_shape=lhs_load_tile_shape or tiles['lhs_load_tile'],
                    rhs_load_tile_shape=rhs_load_tile_shape or tiles['rhs_load_tile'],
                    lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                    rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                    spill_reload=gu_cfg.spill_reload,
                    lhsq_td=hiddenq_td,
                    rhsq_td=up_wq_td,
                    use_scale_packing=gu_cfg.enable_scale_packing,
                    initialize_accumulator=(k_block_idx == 0),
                )

            # Epilogue per (m_tile, n_tile): clamp -> checkpoint gate/up ->
            # SiLU(gate)*up -> fold affinity -> write scaled intermediate (+ ckpt).
            sbuf_step_p = TILES_IN_BLOCK_M * BLOCK_N
            num_m_tiles_in_block = min(TILES_IN_BLOCK_M, div_ceil(B - m_block_start * tile_m, tile_m))
            num_n_tiles_in_block = min(TILES_IN_BLOCK_N, div_ceil(I_TP - n_block_start * tile_n, tile_n))
            for m_tile_idx in range(num_m_tiles_in_block):
                for n_tile_idx in range(num_n_tiles_in_block):
                    m_off = m_block_start * tile_m + m_tile_idx * tile_m
                    i_off = (n_block_start + n_tile_idx) * tile_n
                    actual_m = min(tile_m, B - m_off)
                    actual_n = min(tile_n, I_TP - i_off)
                    sbuf_offset = m_tile_idx * BLOCK_N + n_tile_idx * tile_n

                    gate_tile = gate_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)
                    up_tile = up_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)

                    # Clamp gate (non-linear) and up (linear) BEFORE checkpoint+SiLU,
                    # matching the golden + the backward's no-re-clamp assumption.
                    gate_c = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=buffer_dtype, buffer=nl.sbuf)
                    up_c = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=buffer_dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=gate_c, src=gate_tile)
                    nisa.tensor_copy(dst=up_c, src=up_tile)
                    apply_activation_clamp(
                        gate_c, clamp.non_linear_clamp_upper_limit, clamp.non_linear_clamp_lower_limit
                    )
                    apply_activation_clamp(up_c, clamp.linear_clamp_upper_limit, clamp.linear_clamp_lower_limit)

                    # Checkpoint clamped gate pre-activation and up, transposed to
                    # [I_TP, B] at gate_up_proj_act_checkpoint_T[block_idx, {0,1}].
                    # Skipped when the checkpoint is disabled (None); the clamp above
                    # still runs since it feeds SiLU regardless.
                    if gate_up_proj_act_checkpoint_T is not None:
                        _store_tile_transposed_to_ckpt(
                            gate_c, gate_up_proj_act_checkpoint_T, block_idx, 0, i_off, m_off, I_TP, B, sbm
                        )
                        _store_tile_transposed_to_ckpt(
                            up_c, gate_up_proj_act_checkpoint_T, block_idx, 1, i_off, m_off, I_TP, B, sbm
                        )

                    # SiLU(gate) * up
                    silu_out = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=silu_out, op=nl.silu, data=gate_c)
                    inter = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=buffer_dtype, buffer=nl.sbuf)
                    nisa.tensor_tensor(dst=inter, data1=silu_out, data2=up_c, op=nl.multiply)

                    # Fold affinity (AFFINITY_ON_I): scale each token row by its EA
                    # scalar, broadcast over I_TP. b_tile index = which TILE_M block.
                    b_tile = m_off // TILE_M
                    ea_col = ea_tiles_all[0:actual_m, b_tile : b_tile + 1]
                    scaled = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=buffer_dtype, buffer=nl.sbuf)
                    nisa.tensor_scalar(dst=scaled, data=inter, op0=nl.multiply, operand0=ea_col)

                    nisa.dma_copy(
                        dst=scaled_intermediate[m_off : m_off + actual_m, i_off : i_off + actual_n],
                        src=scaled,
                    )

                    # Optional scaled-intermediate checkpoint, transposed to [I_TP, B].
                    if scaled_intermediate_checkpoint_T is not None:
                        _store_tile_transposed_to_ckpt_2d(
                            scaled, scaled_intermediate_checkpoint_T, block_idx, i_off, m_off, I_TP, B, sbm
                        )

    return TensorDescriptor(data=scaled_intermediate)


def _down_projection_block(
    scaled_intermediate_td, down_weight_td, block_idx, expert_idx_broadcast, B, H, I_TP, E, config, sbm
):
    """Step 9 — down projection: out_block[B, H] = scaled_intermediate[B, I_TP] @ W_down[e].T.

    Per-expert slice on ``down_weight_td`` via :func:`_set_expert_offset_on_td`
    (stride = (I_TP*H)//MATMUL_TILE_K_PHYSICAL non-prequant, else data.shape[0]//E),
    then one :func:`generic_matmul_mxfp8_api` call with the scaled intermediate as
    LHS (quantized on the fly) and the per-expert down weight as RHS.

    Returns:
        nl.ndarray: [B, H] per-block output contribution in HBM (pre-scatter).
    """
    # Forward-natural down weight is [E, H, I_TP] -> 2D [E*H, I_TP] (F-by-K,
    # F = H per expert, K = I_TP), so the down GEMM contracts over I_TP.
    if down_weight_td.scales is None:
        down_expert_stride_in_vs = (H * I_TP) // MATMUL_TILE_K_PHYSICAL
        down_scales_stride = None
        down_effective_f_dim = H
    else:
        down_expert_stride_in_vs = down_weight_td.data.shape[0] // E
        down_scales_stride = down_weight_td.scales.shape[0] // E
        down_effective_f_dim = I_TP // 4
    if not config.no_indirect_load:
        _set_expert_offset_on_td(
            td=down_weight_td,
            expert_idx_broadcast=expert_idx_broadcast,
            block_idx=block_idx,
            expert_stride=down_expert_stride_in_vs,
            scales_stride=down_scales_stride,
            effective_f_dim=down_effective_f_dim,
            name_prefix="fwd_down",
            sbm=sbm,
        )

    buffer_dtype = config.compute_dtype
    out_block = nl.ndarray(
        (B, H),
        dtype=buffer_dtype,
        buffer=nl.shared_hbm,
        name=f"fwd_out_block_{block_idx}",
    )

    # Down GEMM: scaled_intermediate[B, I_TP] @ W_down[I_TP, H] -> [B, H].
    # M->B, N->H, K->I_TP. Take tile_m/tile_k/tile_n from the phase config (autotune
    # cache or heuristic), defaulting to the pinned TILE_M/L_TILE_K/TILE_N; tile_m/
    # tile_k are held to their defaults so physical K stays a legal 128 even when
    # I_TP is a multiple of 128 but not 512 (see _resolve_phase_tiles).
    d_cfg = config.down_config
    tiles = _resolve_phase_tiles(d_cfg)
    NUM_M_TILES = div_ceil(B, tiles['tile_m'])
    NUM_N_TILES = div_ceil(H, tiles['tile_n'])
    NUM_K_TILES = div_ceil(I_TP, tiles['l_tile_k'])
    TILES_IN_BLOCK_M = max(1, min(d_cfg.TILES_IN_BLOCK_M, NUM_M_TILES))
    TILES_IN_BLOCK_N = max(1, min(d_cfg.TILES_IN_BLOCK_N, NUM_N_TILES))
    TILES_IN_BLOCK_K = max(1, min(d_cfg.TILES_IN_BLOCK_K, NUM_K_TILES))

    # Load factors from the auto-generated config, reconciled with the clamped
    # blocks. Fall back to the previous hardcoded values if auto-gen was skipped.
    TILES_IN_LOAD_M = _resolve_tiles_in_load(
        d_cfg.TILES_IN_LOAD_M, TILES_IN_BLOCK_M, min(TILES_IN_BLOCK_M, MAX_TILES_IN_LOAD_M)
    )
    TILES_IN_LOAD_N = _resolve_tiles_in_load(d_cfg.TILES_IN_LOAD_N, TILES_IN_BLOCK_N, 1)

    lhs_load_tile_shape = _compute_load_tile_shape(scaled_intermediate_td, tiles, tiles['tile_m'])
    rhs_load_tile_shape = _compute_load_tile_shape(down_weight_td, tiles, tiles['tile_n'])
    bd = _build_matmul_params(
        TILES_IN_BLOCK_M,
        TILES_IN_BLOCK_N,
        TILES_IN_BLOCK_K,
        lhs_load_tile_shape=lhs_load_tile_shape,
        rhs_load_tile_shape=rhs_load_tile_shape,
        tiles=tiles,
    )
    NUM_M_BLOCKS = div_ceil(NUM_M_TILES, TILES_IN_BLOCK_M)
    NUM_N_BLOCKS = div_ceil(NUM_N_TILES, TILES_IN_BLOCK_N)
    NUM_K_BLOCKS = div_ceil(NUM_K_TILES, TILES_IN_BLOCK_K)
    tile_m = tiles['tile_m']
    tile_n = tiles['tile_n']
    BLOCK_N = TILES_IN_BLOCK_N * tile_n

    # Spill/reload scratch for the down GEMM (mirrors dense-fwd Phase-2). The LHS
    # (scaled_intermediate) is always BF16, so intq_td is always allocated when
    # spilling; the RHS down-weight buffer is skipped when weights are prequant.
    intq_td = None
    downq_td = None
    if d_cfg.spill_reload:
        data_buffer = nl.private_hbm
        intq_td = _allocate_spill_buffer(
            num_k_blocks=NUM_K_BLOCKS,
            num_f_blocks=NUM_M_BLOCKS,
            block_f_logical=bd.BLOCK_M_LOGICAL,
            tiles_in_block_k=TILES_IN_BLOCK_K,
            use_scale_packing=d_cfg.enable_scale_packing,
            data_buffer=data_buffer,
        )
        if not down_weight_td.is_quantized:
            downq_td = _allocate_spill_buffer(
                num_k_blocks=NUM_K_BLOCKS,
                num_f_blocks=NUM_N_BLOCKS,
                block_f_logical=bd.BLOCK_N_LOGICAL,
                tiles_in_block_k=TILES_IN_BLOCK_K,
                use_scale_packing=d_cfg.enable_scale_packing,
                data_buffer=data_buffer,
            )

    # Drive M and N explicitly. The down weight TD is the full stacked-expert
    # [E*H, I_TP] view, so its logical N = E*H. We write each GEMM block into a
    # fixed-size SBUF accumulator (not directly to HBM): a direct HBM output_td
    # would make the API's store clamp against the weight's N_LOGICAL (=E*H) and
    # overrun the per-expert [B, H] output. We then copy the SBUF block into
    # out_block clamped to the actual H. Mirrors the backward Phase-2.
    for idx_m in range(NUM_M_BLOCKS):
        for idx_n in range(NUM_N_BLOCKS):
            output_sbuf = sbm.alloc_stack(shape=(tile_m, TILES_IN_BLOCK_M * BLOCK_N), dtype=nl.float32, buffer=nl.sbuf)
            output_sbuf_td = TensorDescriptor(data=output_sbuf)
            generic_matmul_mxfp8_api(
                lhs_hbm_td=scaled_intermediate_td,
                rhs_hbm_td=down_weight_td,
                bd=bd,
                output_td=output_sbuf_td,
                block_idx_m=(idx_m, idx_m + 1),
                block_idx_n=(idx_n, idx_n + 1),
                lhs_m_offset=0,
                rhs_n_offset=0,
                TILES_IN_LOAD_M=TILES_IN_LOAD_M,
                TILES_IN_LOAD_N=TILES_IN_LOAD_N,
                lhs_matmul_tile_shape_physical=tiles['lhs_matmul_tile_physical'],
                rhs_matmul_tile_shape_physical=tiles['rhs_matmul_tile_physical'],
                lhs_load_tile_shape=lhs_load_tile_shape or tiles['lhs_load_tile'],
                rhs_load_tile_shape=rhs_load_tile_shape or tiles['rhs_load_tile'],
                lhs_quantize_tile_shape=tiles['lhs_quantize_tile'],
                rhs_quantize_tile_shape=tiles['rhs_quantize_tile'],
                spill_reload=d_cfg.spill_reload,
                lhsq_td=intq_td,
                rhsq_td=downq_td,
                use_scale_packing=d_cfg.enable_scale_packing,
            )

            # Copy the SBUF accumulator block into out_block[B, H], clamped to the
            # real per-block M (B) and N (H) extents.
            sbuf_step_p = TILES_IN_BLOCK_M * BLOCK_N
            n_off_base = idx_n * BLOCK_N
            actual_n = min(BLOCK_N, H - n_off_base)
            num_m_tiles_in_block = min(TILES_IN_BLOCK_M, div_ceil(B - idx_m * TILES_IN_BLOCK_M * tile_m, tile_m))
            for tmi in range(num_m_tiles_in_block):
                m_off = (idx_m * TILES_IN_BLOCK_M + tmi) * tile_m
                actual_m = min(tile_m, B - m_off)
                if actual_m <= 0 or actual_n <= 0:
                    continue
                res = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=buffer_dtype, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=res,
                    src=output_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=tmi * BLOCK_N),
                )
                nisa.dma_copy(
                    dst=out_block[m_off : m_off + actual_m, n_off_base : n_off_base + actual_n],
                    src=res,
                )
    return out_block


def _store_output_block_contiguous(out_block, output_shard, block_idx, B, H, sbm):
    """Store a directly packed single-expert output block contiguously."""
    NUM_B_TILES = div_ceil(B, TILE_M)
    NUM_F_TILES = div_ceil(H, TILE_N)
    block_offset = block_idx * B
    for b_tile_idx in range(NUM_B_TILES):
        b_off = b_tile_idx * TILE_M
        actual_b = min(TILE_M, B - b_off)
        row_off = block_offset + b_off
        for f_tile_idx in range(NUM_F_TILES):
            f_off = f_tile_idx * TILE_N
            actual_f = min(TILE_N, H - f_off)
            result_tile = sbm.alloc_stack(shape=(actual_b, actual_f), dtype=out_block.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=result_tile, src=out_block[b_off : b_off + actual_b, f_off : f_off + actual_f])
            nisa.dma_copy(
                dst=output_shard[nl.ds(row_off, actual_b), f_off : f_off + actual_f],
                src=result_tile,
            )


def _scatter_output_block(out_block, output_shard, token_indices, B, H, skip_dma, is_accumulating, block_idx, sbm):
    """Step 10 — indirect scatter of out_block[B, H] back into this shard's output slab.

    Inverse of the gather: ``token_indices`` index destination rows of
    ``output_shard [T, H]``. When ``is_accumulating`` (top_k > 1), do a
    read-modify-write so multiple blocks of the same shard contributing to the
    same token sum; otherwise overwrite. ``-1`` pads are dropped via
    ``oob_mode.skip``. Mirrors the backward's Phase-2 hidden-grad scatter
    (bwmm_bwd_dropless_mxfp8.py:922-980).
    """
    NUM_B_TILES = div_ceil(B, TILE_M)
    NUM_F_TILES = div_ceil(H, TILE_N)

    for b_tile_idx in range(NUM_B_TILES):
        b_off = b_tile_idx * TILE_M
        actual_b = min(TILE_M, B - b_off)
        # One int32 token index per partition for this B-tile.
        token_indices_col = token_indices[:, b_tile_idx : b_tile_idx + 1]

        for f_tile_idx in range(NUM_F_TILES):
            f_off = f_tile_idx * TILE_N
            actual_f = min(TILE_N, H - f_off)

            # Load this block's contribution tile from the per-block HBM result.
            result_tile = sbm.alloc_stack(shape=(actual_b, actual_f), dtype=out_block.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=result_tile, src=out_block[b_off : b_off + actual_b, f_off : f_off + actual_f])

            if is_accumulating:
                # Read-modify-write: gather existing slab value, add, scatter back.
                existing_tile = sbm.alloc_stack(shape=(actual_b, actual_f), dtype=out_block.dtype, buffer=nl.sbuf)
                if skip_dma.skip_token:
                    nisa.memset(existing_tile, value=0)
                nisa.dma_copy(
                    dst=existing_tile,
                    src=output_shard.ap(
                        pattern=[[H, actual_b], [1, actual_f]],
                        offset=f_off,
                        vector_offset=token_indices_col,
                        indirect_dim=0,
                    ),
                    oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
                )
                nisa.tensor_tensor(dst=result_tile, op=nl.add, data1=result_tile, data2=existing_tile)

            nisa.dma_copy(
                dst=output_shard.ap(
                    pattern=[[H, actual_b], [1, actual_f]],
                    offset=f_off,
                    vector_offset=token_indices_col,
                    indirect_dim=0,
                ),
                src=result_tile,
                oob_mode=oob_mode.skip if skip_dma.skip_token else oob_mode.error,
            )


def _zero_init_output(dst, T, H, sbm):
    """Zero an [T, H] HBM tensor in TILE_M-row chunks (memset SBUF -> DMA out).

    Required before the RMW scatter accumulates into the per-shard slab.
    """
    NUM_T_TILES = div_ceil(T, TILE_M)
    NUM_F_TILES = div_ceil(H, TILE_N)
    for t_idx in range(NUM_T_TILES):
        t_off = t_idx * TILE_M
        actual_t = min(TILE_M, T - t_off)
        for f_idx in range(NUM_F_TILES):
            f_off = f_idx * TILE_N
            actual_f = min(TILE_N, H - f_off)
            ztile = sbm.alloc_stack(shape=(actual_t, actual_f), dtype=dst.dtype, buffer=nl.sbuf)
            nisa.memset(ztile, value=0)
            nisa.dma_copy(dst=dst[t_off : t_off + actual_t, f_off : f_off + actual_f], src=ztile)


def _reduce_output_shards(output_hidden_states, output_slabs, num_shards, shard_id, T, H, sbm):
    """Final reduce — sum the per-shard slabs output_slabs[shard, T, H] into output_hidden_states[T, H].

    SHARD_ON_BLOCK gives each core a disjoint set of blocks writing into its own
    slab; the layer output is the elementwise sum across slabs. A core_barrier
    makes every shard's writes visible before the reduce. Each shard reduces a
    disjoint T-tile range so the cores share the reduce work without colliding.
    """
    # core_barrier's rank arg must be a literal tuple; LNC2 (num_shards==2) is
    # required, so (0, 1) is correct (matches the backward).
    for s in range(num_shards):
        nisa.core_barrier(output_slabs[s], (0, 1))

    NUM_T_TILES = div_ceil(T, TILE_M)
    NUM_F_TILES = div_ceil(H, TILE_N)
    for t_idx in range(NUM_T_TILES):
        # Partition the reduce across cores by T-tile to avoid redundant work.
        if num_shards > 1 and (t_idx % num_shards) != shard_id:
            continue
        t_off = t_idx * TILE_M
        actual_t = min(TILE_M, T - t_off)
        for f_idx in range(NUM_F_TILES):
            f_off = f_idx * TILE_N
            actual_f = min(TILE_N, H - f_off)
            acc = sbm.alloc_stack(shape=(actual_t, actual_f), dtype=output_hidden_states.dtype, buffer=nl.sbuf)
            nisa.dma_copy(src=output_slabs[0][t_off : t_off + actual_t, f_off : f_off + actual_f], dst=acc)
            for s in range(1, num_shards):
                other = sbm.alloc_stack(shape=(actual_t, actual_f), dtype=output_hidden_states.dtype, buffer=nl.sbuf)
                nisa.dma_copy(src=output_slabs[s][t_off : t_off + actual_t, f_off : f_off + actual_f], dst=other)
                nisa.tensor_tensor(dst=acc, op=nl.add, data1=acc, data2=other)
            nisa.dma_copy(dst=output_hidden_states[t_off : t_off + actual_t, f_off : f_off + actual_f], src=acc)


# =====================================================================================
# Top-level dropless forward orchestration
# =====================================================================================


def blockwise_mm_fwd_dropless_mxfp8(
    # --- Input TensorDescriptors (passed flat — no tensor-bearing dataclasses across
    #     traced function boundaries) ---
    hidden_states_td: TensorDescriptor,
    gate_up_weight_td: TensorDescriptor,
    down_weight_td: TensorDescriptor,
    token_position_to_id_td: TensorDescriptor,
    block_to_expert_td: TensorDescriptor,
    expert_affinities_masked_td: TensorDescriptor,
    # --- Derived dimensions (plain ints) ---
    T: int,
    H: int,
    I_TP: int,
    E: int,
    N: int,
    block_size: int,
    # --- Config and output buffers ---
    config,
    output_hidden_states: nl.ndarray,
    output_slabs: nl.ndarray,
    gate_up_proj_act_checkpoint_T: nl.ndarray = None,
    scaled_intermediate_checkpoint_T: nl.ndarray = None,
):
    """MXFP8 forward pass implementation for blockwise dropless MoE (shard-on-block).

    Orchestrates the per-block FFN and emits the checkpoints the backward consumes.

    Args:
        hidden_states_td (TensorDescriptor): [T, H] BF16 input hidden states.
        gate_up_weight_td (TensorDescriptor): [E*H, 2*I_TP] gate/up weights
            (reshaped from [E, H, 2, I_TP]); per-expert slice via scalar_offset.
        down_weight_td (TensorDescriptor): [E*I_TP, H] down weights (reshaped from
            [E, I_TP, H]); per-expert slice via scalar_offset.
        token_position_to_id_td (TensorDescriptor): [N*B] int32 token index map
            (pad id = -1 under skip_dma).
        block_to_expert_td (TensorDescriptor): [N, 1] int32 expert per block.
        expert_affinities_masked_td (TensorDescriptor): [T*E, 1] fp32 affinities.
        T, H, I_TP, E, N (int): derived dims.
        block_size (int): tokens per block (B).
        config (MXFP8MOEFwdConfig): kernel configuration.
        output_hidden_states (nl.ndarray): [T, H] final layer output (shared_hbm),
            written by the cross-shard reduce.
        output_slabs (nl.ndarray): [num_shards, T, H] per-shard scratch slabs
            (shared_hbm); each core scatter-accumulates into its own slab, then
            the slabs are summed into output_hidden_states. None under
            no_indirect_load, where each core writes disjoint rows straight into
            output_hidden_states (no slabs, zero-init, or reduce).
        gate_up_proj_act_checkpoint_T (nl.ndarray, optional): [N, 2, I_TP, B] BF16;
            slot[block, 0] = clamped gate pre-activation, slot[block, 1] = clamped
            up. B contiguous (last axis). When None, the store is skipped.
        scaled_intermediate_checkpoint_T (nl.ndarray, optional): [N, I_TP, B] BF16,
            = SiLU(gate)*up*EA transposed. Emitted when not None.

    Returns:
        None. Results are written into output[0] and the checkpoint tensors.
    """
    if get_active_sbm() == None:
        create_and_set_active_sbm()
    sbm = get_active_sbm()
    sbm.open_scope(name="MXFP8 MOE FWD")

    B = block_size
    NUM_B_TILES = B // TILE_M

    # SHARD_ON_BLOCK: each core owns the strided block subset range(shard_id, N, num_shards).
    _, num_shards, shard_id = get_program_sharding_info()

    # no_indirect_load (E=1, top_k=1) gives each core a disjoint set of blocks that
    # map to disjoint contiguous output rows, so every row is written exactly once
    # across cores. Write straight into output_hidden_states and skip the per-shard
    # slab, its zero-init, and the cross-shard reduce — those are only needed for the
    # top_k>1 indirect scatter, where cores can touch the same token row and must be
    # summed. output_slabs is None in this path (see wrapper).
    if config.no_indirect_load:
        output_shard = output_hidden_states
    else:
        output_shard = output_slabs[shard_id]

    scaled_intermediate_checkpoint_T_td = (
        TensorDescriptor(data=scaled_intermediate_checkpoint_T) if scaled_intermediate_checkpoint_T != None else None
    )

    # --- One-time setup (mirrors the bwd) -------------------------------------------------
    if config.no_indirect_load:
        expert_idx_broadcast = None
        token_indices_bufs = None
    else:
        # S1: bulk-load block_to_expert into SBUF and broadcast across partitions.
        expert_idx_bufs = sbm.alloc_stack((1, N), dtype=nl.int32, buffer=nl.sbuf, align=32)
        block_to_expert_2d = block_to_expert_td.data.reshape((1, N))
        nisa.dma_copy(expert_idx_bufs[0, 0:N], block_to_expert_2d[0, 0:N])

        expert_idx_broadcast = sbm.alloc_stack(
            (TILE_M, N), dtype=nl.int32, buffer=nl.sbuf, name="fwd_expert_idx_broadcast", align=32
        )
        stream_shuffle_broadcast(src=expert_idx_bufs, dst=expert_idx_broadcast)

        # S2: double-buffered token-index slots; prefetch this shard's first block.
        token_indices_bufs = [
            sbm.alloc_stack((TILE_M, NUM_B_TILES), dtype=nl.int32, align=32),
            sbm.alloc_stack((TILE_M, NUM_B_TILES), dtype=nl.int32, align=32),
        ]
        if shard_id < N:
            _load_token_indices_dgt(token_position_to_id_td.data, shard_id, B, NUM_B_TILES, dst=token_indices_bufs[0])

    # S3: zero-init this shard's output slab (RMW scatter writes into it). The
    # no_indirect_load path overwrites every owned row directly in output_hidden_states,
    # so there is nothing to pre-zero and no cross-core visibility to barrier on.
    if not config.no_indirect_load:
        _zero_init_output(output_shard, T, H, sbm)
        nisa.core_barrier(output_shard, (0, 1))

    # --- Per-block loop: this core processes blocks [shard_id, shard_id+num_shards, ...] ----
    ring = 0
    for block_idx in range(shard_id, N, num_shards):
        sbm.open_scope(name=f"FwdBlock {block_idx}")
        if config.no_indirect_load:
            # The framework packed this expert's tokens contiguously, so block_idx
            # maps directly to hidden/affinity/output rows.
            hidden_block_td = TensorDescriptor(data=hidden_states_td.data[nl.ds(block_idx * B, B), 0:H])
            ea_tiles_all = _load_block_affinities_contiguous(expert_affinities_masked_td.data, block_idx, B, sbm)
        else:
            block_token_pos_to_id_full = token_indices_bufs[ring]

            # Prefetch the next block this shard will own.
            next_block_idx = block_idx + num_shards
            if next_block_idx < N:
                nxt = 1 - ring
                _load_token_indices_dgt(
                    token_position_to_id_td.data, next_block_idx, B, NUM_B_TILES, dst=token_indices_bufs[nxt]
                )

            # Step 3: gather this block's hidden tokens.
            hidden_block_td = _gather_hidden_block(
                hidden_states_td, block_token_pos_to_id_full, B, H, config.skip_dma, block_idx, config, sbm
            )

            # Step 7a: gather per-token affinities (AFFINITY_ON_I).
            ea_tiles_all = _gather_block_affinities(
                expert_affinities_masked_td.data,
                block_token_pos_to_id_full,
                expert_idx_broadcast,
                block_idx,
                B,
                E,
                config.skip_dma,
                sbm,
            )

        # Steps 4-8: gate/up -> clamp+checkpoint -> SwiGLU -> affinity fold -> checkpoint.
        scaled_intermediate_td = _gate_up_swiglu_affinity_block(
            hidden_block_td=hidden_block_td,
            gate_up_weight_td=gate_up_weight_td,
            ea_tiles_all=ea_tiles_all,
            block_idx=block_idx,
            expert_idx_broadcast=expert_idx_broadcast,
            B=B,
            H=H,
            I_TP=I_TP,
            E=E,
            gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
            scaled_intermediate_checkpoint_T=scaled_intermediate_checkpoint_T,
            config=config,
            sbm=sbm,
        )

        # Step 9: down projection.
        out_block = _down_projection_block(
            scaled_intermediate_td, down_weight_td, block_idx, expert_idx_broadcast, B, H, I_TP, E, config, sbm
        )

        # Step 10: write into this shard's output slab.
        if config.no_indirect_load:
            _store_output_block_contiguous(out_block, output_shard, block_idx, B, H, sbm)
        else:
            _scatter_output_block(
                out_block,
                output_shard,
                block_token_pos_to_id_full,
                B,
                H,
                config.skip_dma,
                config.is_tensor_update_accumulating,
                block_idx,
                sbm,
            )

        ring = 1 - ring
        sbm.close_scope()

    # Final reduce of the per-shard output slabs into the returned [T, H] output.
    # Skipped for no_indirect_load: each core already wrote its disjoint rows straight
    # into output_hidden_states, so there are no slabs to sum.
    if not config.no_indirect_load:
        _reduce_output_shards(output_hidden_states, output_slabs, num_shards, shard_id, T, H, sbm)

    sbm.close_scope()
