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
from ...mxfp_utils.mxfp8_utils.common_utils import create_and_set_active_sbm, get_active_sbm, with_active_sbm
from ...mxfp_utils.mxfp8_utils.quantize_mxfp8_utils import MX_PARTITION_SIZE, TILE_SIZE_GEMM_MOVING_MAX

# Reuse the backward's block-orchestration primitives verbatim — same indirect
# DMA / DGT / per-expert offset semantics, so fwd and bwd stay byte-compatible.
from ..bwd.bwmm_bwd_dropless_mxfp8 import (
    _gather_block_tokens,
    _gather_block_tokens_transposed,  # noqa: F401
    _load_token_indices_dgt,
    _set_expert_offset_on_td,
)
from ..moe_mxfp8_checkpoint_config import CheckpointLayout

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


def _gather_hidden_block(hidden_states_td, token_indices, B, H, skip_dma, block_idx, config, sbm, name_suffix=""):
    """Step 3 — indirect-gather this block's tokens into a dense [B, H] HBM tile.

    Reuses :func:`_gather_block_tokens` (indirect ``dma_copy`` with
    ``vector_offset`` + ``oob_mode.skip`` for ``-1`` padding). The result is
    wrapped in a fresh per-block ``TensorDescriptor`` (unswizzled BF16) to feed
    the gate/up GEMMs as the LHS.

    ``name_suffix`` disambiguates the shared_hbm scratch name when both cores
    process the same ``block_idx`` (the odd-N tail half-block): same-name
    shared_hbm allocations alias across cores, so the two halves must use
    distinct names (e.g. "h0"/"h1"). Empty for the round-robin full blocks,
    where the two cores always own disjoint block_idx.

    Returns:
        TensorDescriptor: [B, H] gathered hidden block (is_f_by_k defaults).
    """
    buffer_dtype = config.compute_dtype
    hidden_block = nl.ndarray(
        (B, H),
        dtype=buffer_dtype,
        buffer=nl.shared_hbm,
        name=f"fwd_hidden_block_{block_idx}{name_suffix}",
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


def _load_block_affinities_contiguous(
    expert_affinities_masked, block_idx, B, sbm, full_block_size=None, m_base=0, name_suffix=""
):
    """Load per-token affinities for the directly packed single expert.

    The epilogue multiplies SBUF activation tiles by these row scalars, so the
    affinities still need a direct HBM -> SBUF load even though token routing is
    already contiguous.

    Half-block tail (odd N): ``B`` is this call's token count (B/2 for a half),
    ``full_block_size`` is the full block size used to locate the block's rows,
    and ``m_base`` is this half's token offset within the block. The packed rows
    start at ``block_idx * full_block_size + m_base``. Defaults
    (``full_block_size=None`` -> B, ``m_base=0``) reproduce the whole-block load.
    """
    full_B = full_block_size if full_block_size is not None else B
    NUM_B_TILES = div_ceil(B, TILE_M)
    ea_tiles_all = sbm.alloc_stack(
        (TILE_M, NUM_B_TILES), dtype=nl.float32, name=f"fwd_ea_direct_{block_idx}{name_suffix}", align=32
    )
    block_offset = block_idx * full_B + m_base
    for b_tile_idx in range(NUM_B_TILES):
        b_off = b_tile_idx * TILE_M
        actual_b = min(TILE_M, B - b_off)
        row_off = block_offset + b_off
        nisa.dma_copy(
            dst=ea_tiles_all[0:actual_b, b_tile_idx : b_tile_idx + 1],
            src=expert_affinities_masked[nl.ds(row_off, actual_b), 0:1],
        )
    return ea_tiles_all


# --- Checkpoint store dispatch ---------------------------------------------------
# One entry point (`_store_checkpoint_block`) routes a WHOLE per-block activation
# buffer to the store worker for its CheckpointLayout, so a new layout is added by
# writing its store worker + adding one branch below (and its shape in
# checkpoint_block_dims); the kernel call sites never change. Dispatch is a static
# if/elif (not a dict of callables) because the NKI parser frontend traces the
# kernel and cannot resolve an indirect call through a looked-up function.
#
# Block granularity (not per-tile) is deliberate and is the whole point of this
# layer: a per-(m_tile, n_tile) store writes only `tile_n` of each token row, so
# every row is an isolated `tile_n * dtype` island on an `I_TP * dtype` pitch and
# the DMA cannot coalesce past it (a 128x512 bf16 tile degenerated into 128 x 1KiB
# packets, ~62% of each packet's time being fixed per-packet overhead). Storing the
# full [B, I_TP] block instead makes consecutive token rows adjacent, so the whole
# region is one contiguous run and the engines emit max-size packets.


def _store_block_token_major(
    src_block, dst, dst_base, row_pitch, num_m_tiles, rows_per_tile, width, sbuf_step_p, src_tile_stride
):
    """Emit ONE DMA moving a whole [num_m_tiles * rows_per_tile, width] SBUF block
    into a token-major (row-pitch ``row_pitch``) HBM region starting at ``dst_base``.

    Shared single-DMA worker behind both the scaled-intermediate store and the
    DIRECT checkpoint store — the two differ only in ``dst``/``dst_base``.

    Geometry. The gate/up/scaled block buffers are all [rows_per_tile,
    num_m_tiles * src_tile_stride]: an m_tile advances the FREE dim by
    ``src_tile_stride`` while the token rows within it sit across the 128
    partitions (partition stride ``sbuf_step_p``). So SBUF element (p, t, n) holds
    token ``t * rows_per_tile + p``, column n — which lands at HBM offset
    ``dst_base + (t * rows_per_tile + p) * row_pitch + n``. Both sides are
    therefore 3-D patterns over (p, t, n), which is what lets one instruction
    cover the block.

    When ``width == row_pitch`` (the block spans the full row) the destination
    strides tile the region exactly and the whole [rows, width] area is ONE
    contiguous run, so the DMA emits max-size packets. A narrower ``width`` still
    coalesces each row's ``width`` elements, just with a gap between rows.
    """
    nisa.dma_copy(
        dst=dst.ap(
            pattern=[[row_pitch, rows_per_tile], [rows_per_tile * row_pitch, num_m_tiles], [1, width]],
            offset=dst_base,
        ),
        src=src_block.ap(
            pattern=[[sbuf_step_p, rows_per_tile], [src_tile_stride, num_m_tiles], [1, width]],
            offset=0,
        ),
    )


def _store_checkpoint_block(
    layout,
    src_block,
    ckpt,
    block_idx,
    half,
    n_halves,
    m_off,
    n_off,
    num_m_tiles,
    rows_per_tile,
    width,
    I_TP,
    ckpt_B,
    sbuf_step_p,
    src_tile_stride,
    sbm,
):
    """Store one whole per-block activation buffer to ``ckpt`` using ``layout``'s worker.

    ``src_block`` is the block-wide SBUF buffer (gate/up accumulator or the scaled
    block); ``num_m_tiles``/``rows_per_tile``/``width`` describe the region of it to
    store, and ``sbuf_step_p``/``src_tile_stride`` its geometry (see
    :func:`_store_block_token_major`).

    ``half``/``n_halves`` index the gate(0)/up(1) planes of the 4D gate/up
    checkpoint ([N, n_halves=2, ...]); for the 2D scaled checkpoint pass
    half=0, n_halves=1 ([N, ...]).

    ``ckpt_B`` is the checkpoint slot's B-axis extent (the full block size) while
    ``m_off`` is this store's token offset within that slot — already including the
    half-block tail's ``m_base``, so a tail half writes its rows into the correct
    half of the shared slot. ``n_off`` is the I_TP-axis offset of this n_block.

    Add a new layout by adding a branch here.
    """
    if layout == CheckpointLayout.DIRECT:
        # token-major [.., ckpt_B, I_TP]: one plain DMA, no transpose (sbm unused).
        block_base = (block_idx * n_halves * ckpt_B * I_TP) + (half * ckpt_B * I_TP)
        _store_block_token_major(
            src_block=src_block,
            dst=ckpt,
            dst_base=block_base + m_off * I_TP + n_off,
            row_pitch=I_TP,
            num_m_tiles=num_m_tiles,
            rows_per_tile=rows_per_tile,
            width=width,
            sbuf_step_p=sbuf_step_p,
            src_tile_stride=src_tile_stride,
        )
    elif layout == CheckpointLayout.TRANSPOSED:
        # I_TP-major [.., I_TP, ckpt_B] — needs a PE nc_transpose of the block before
        # the store. Not implemented at block granularity yet; the per-tile transposed
        # store this layer replaced was itself a source of the tiny-packet stall, so
        # it is not worth reviving tile-wise.
        # TODO(moe-fwd): block-level transposed store — chunk the I_TP axis into
        # <=TILE_M PE transposes into one [width, rows] staging buffer, then a single
        # DMA with row_pitch=ckpt_B.
        kernel_assert(
            False,
            "TRANSPOSED checkpoint layout is not currently supported by the block-level "
            "checkpoint store; use CheckpointLayout.DIRECT",
        )
    else:
        kernel_assert(False, f"unsupported checkpoint layout: {layout}")


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
    full_block_size=None,
    m_base=0,
    name_suffix="",
    gate_wq_td=None,
    up_wq_td=None,
    reuse_weights=False,
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

    Half-block tail (MXFP8 fwd, odd N): ``B`` is the number of tokens this call
    actually computes (B/2 for a tail half), while ``full_block_size`` is the
    checkpoint slot's B-axis extent (the full block size). The compute (GEMMs,
    SwiGLU, EA fold, scaled_intermediate scratch) works on the ``B`` tokens with
    block-relative ``m_off``, but the checkpoint stores use ``full_block_size``
    as the row stride and ``m_off + m_base`` as the token offset, so this core's
    half lands at tokens [m_base : m_base + B] of the full slot. Defaults
    (``full_block_size=None`` -> B, ``m_base=0``) reproduce the whole-block
    behavior exactly.

    Args:
        full_block_size (int, optional): Checkpoint B-axis extent. None -> B (the
            whole block is this call's tokens, no split).
        m_base (int): Token offset of this half within the full block (0 for the
            full-block path; shard_id * (B_full // 2) for the tail half).
        name_suffix (str): Disambiguates the scaled_intermediate shared_hbm name
            when both cores share ``block_idx`` (the tail half). Empty for full
            blocks (disjoint block_idx per core).
        gate_wq_td / up_wq_td (TensorDescriptor, optional): Quantized weight spill
            buffers from an earlier block. None = not yet populated: this call
            allocates them and the GEMMs load+quantize+spill as usual. Non-None =
            already in HBM scratch, fed straight in as the RHS (no load/quantize/spill).
        reuse_weights (bool): Carry the weight buffers across blocks. Set only when
            the phase spills and E == 1.

    Returns:
        tuple: (scaled_intermediate_td, gate_wq_td, up_wq_td) — the [B, I_TP]
        EA-scaled intermediate (down-projection LHS) plus the weight spill buffers
        for the next block to reuse (None when not reusing).
    """
    # Checkpoint B-axis stride: the full block size, which equals B for whole
    # blocks and stays the full size for a half-block tail so the two halves
    # reassemble into one contiguous [.., .., B_full] slot (see backward read).
    ckpt_B = full_block_size if full_block_size is not None else B
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

    # Per-checkpoint store layout (TRANSPOSED [I_TP, B] vs DIRECT [B, I_TP]).
    gate_up_layout = config.checkpoint_config.gate_up_proj_act_layout
    scaled_layout = config.checkpoint_config.scaled_intermediate_layout

    scaled_intermediate = nl.ndarray(
        (B, I_TP),
        dtype=buffer_dtype,
        buffer=nl.shared_hbm,
        name=f"fwd_scaled_intermediate_{block_idx}{name_suffix}",
    )

    clamp = config.clamp_limits
    # clamp_limits=None is normalized to a ClampLimits() with all-None fields in
    # config __post_init__, so test the fields, not the object. Matches the no-op
    # condition inside apply_activation_clamp.
    clamp_active = (
        clamp.non_linear_clamp_upper_limit is not None
        or clamp.non_linear_clamp_lower_limit is not None
        or clamp.linear_clamp_upper_limit is not None
        or clamp.linear_clamp_lower_limit is not None
    )

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

    # Weight reuse: a non-None gate_wq_td means an earlier block already spilled the
    # quantized weights, so the RHS is the x4 spill buffer, not the BF16 tensor. That
    # changes the RHS load tile shape, so resolve it before building the bd.
    weights_cached = reuse_weights and gate_wq_td != None
    rhs_shape_td = gate_wq_td if weights_cached else gate_up_weight_td

    lhs_load_tile_shape = _compute_load_tile_shape(hidden_block_td, tiles, tile_m)
    rhs_load_tile_shape = _compute_load_tile_shape(rhs_shape_td, tiles, tile_n)
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

    # Spill/reload scratch, per the dense-fwd Phase-1 pattern in
    # mlp_fwd_mxfp8_kernel.py. Gate and up GEMMs share ONE quantized-hidden LHS
    # buffer (hiddenq_td) and each gets its own weight RHS buffer. Only BF16 operands
    # are spilled; the `is_quantized` guard skips pre-quantized ones. data_buffer is
    # private_hbm (LNC2 asserted on).
    #
    # The LHS buffer is always fresh per block (its F extent is sized by B, which the
    # odd-N tail half changes). The weight RHS buffers are sized only by H / I_TP and
    # the phase config, so under `reuse_weights` they are allocated on the first block
    # and carried in on later ones (weights_cached), skipping the load+quantize+spill.
    hiddenq_td = None
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
        if not gate_up_weight_td.is_quantized and not weights_cached:
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

                # Gate GEMM: hidden_block[B, H] @ W_gate[H, I_TP] -> [B, I_TP].
                # On reuse blocks the RHS is the x4 spill buffer an earlier block
                # filled; is_quantized makes the API load it and skip quantize+respill
                # (rhsq_td unused, passed as None).
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=hidden_block_td,
                    rhs_hbm_td=gate_wq_td if weights_cached else gate_up_weight_td,
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
                    rhsq_td=None if weights_cached else gate_wq_td,
                    use_scale_packing=gu_cfg.enable_scale_packing,
                    initialize_accumulator=(k_block_idx == 0),
                )

                # Up GEMM: reuses the quantized hidden; rhs_n_offset=I_TP selects the
                # up half of the per-expert F slice. On reuse blocks it must be 0:
                # up_wq_td holds only the up half, and the API does not zero the offset
                # itself because the quantized RHS takes the load-from-hbm branch.
                generic_matmul_mxfp8_api(
                    lhs_hbm_td=hidden_block_td,
                    rhs_hbm_td=up_wq_td if weights_cached else gate_up_weight_td,
                    bd=bd,
                    output_td=up_output_td,
                    block_idx_m=(m_block_idx, m_block_idx + 1),
                    block_idx_n=(n_block_idx, n_block_idx + 1),
                    block_idx_k=(k_block_idx, k_block_idx + 1),
                    lhs_sbuf_td=hidden_sbuf_td,
                    lhs_m_offset=0,
                    rhs_n_offset=0 if weights_cached else I_TP,
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
                    rhsq_td=None if weights_cached else up_wq_td,
                    use_scale_packing=gu_cfg.enable_scale_packing,
                    initialize_accumulator=(k_block_idx == 0),
                )

            # Epilogue. Compute stays per (m_tile, n_tile) — the clamp / SiLU /
            # affinity-fold ops are tile-shaped and their cost does not depend on
            # tiling — but every HBM STORE is hoisted out of both tile loops and
            # issued once for the whole [rows, BLOCK_N] block (see
            # _store_block_token_major for why: a per-tile store writes only tile_n
            # of each token row, so each row is an isolated island on an I_TP pitch
            # and the DMA cannot coalesce past it).
            #
            # The gate/up accumulators are already block-wide — gate_sbuf/up_sbuf are
            # [tile_m, TILES_IN_BLOCK_M * BLOCK_N] with m_tile at free-dim offset
            # m_tile_idx * BLOCK_N — so their checkpoint stores read them directly and
            # need no staging buffer at all. Only the SwiGLU result needs a new
            # block-wide buffer (`scaled_block`) so its rows are contiguous too; the
            # per-tile `scaled` result writes into a view of it instead of a private
            # tile, which costs no extra copy.
            sbuf_step_p = TILES_IN_BLOCK_M * BLOCK_N
            num_m_tiles_in_block = min(TILES_IN_BLOCK_M, div_ceil(B - m_block_start * tile_m, tile_m))
            num_n_tiles_in_block = min(TILES_IN_BLOCK_N, div_ceil(I_TP - n_block_start * tile_n, tile_n))

            # Rows/width this n_block actually covers. The single-DMA store walks the
            # m_tiles with one uniform stride, so every m_tile must hold the same row
            # count — true here because B is always a multiple of tile_m (block_size is
            # one of 128..4096 and tile_m is pinned to TILE_M=128; the odd-N tail half
            # is B/2 with NUM_B_TILES even, so it stays 128-aligned). A ragged last
            # m_tile would need the tail stored separately, so assert rather than
            # silently truncate.
            block_m_off = m_block_start * tile_m
            kernel_assert(
                B % tile_m == 0,
                f"moe fwd block-granular checkpoint store requires B ({B}) to be a multiple of tile_m ({tile_m})",
            )
            rows_per_tile = tile_m
            n_off = n_block_start * tile_n
            # The n_tiles sit at consecutive free-dim offsets k*tile_n, so even a short
            # last n_tile leaves the covered region contiguous and exactly block_width wide.
            block_width = min(BLOCK_N, I_TP - n_off)

            # Block-wide SwiGLU result, same [tile_m, TILES_IN_BLOCK_M * BLOCK_N]
            # geometry as the accumulators so one DMA covers it.
            #
            # buffer_dtype (bf16), NOT fp32: this buffer is block-sized, so in fp32 it
            # is as large as a whole gate/up accumulator (e.g. 8*1536*4 = 48 KiB per
            # partition) and a third co-resident 48 KiB buffer pushed the epilogue past
            # what the SBUF allocator could hold, spilling it to HBM — which cost far
            # more than the store coalescing saved. bf16 halves it and is lossless
            # relative to the previous behavior anyway: both consumers (the
            # scaled_intermediate down-GEMM LHS and the checkpoint) are compute_dtype,
            # so the fp32->bf16 rounding used to happen in the store DMA instead of
            # here. The SiLU/affinity math still runs in fp32 and only the result is
            # narrowed.
            scaled_block = sbm.alloc_stack(
                shape=(tile_m, TILES_IN_BLOCK_M * BLOCK_N), dtype=buffer_dtype, buffer=nl.sbuf
            )

            for m_tile_idx in range(num_m_tiles_in_block):
                for n_tile_idx in range(num_n_tiles_in_block):
                    m_off = block_m_off + m_tile_idx * tile_m
                    i_off = (n_block_start + n_tile_idx) * tile_n
                    actual_m = min(tile_m, B - m_off)
                    actual_n = min(tile_n, I_TP - i_off)
                    sbuf_offset = m_tile_idx * BLOCK_N + n_tile_idx * tile_n

                    gate_tile = gate_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)
                    up_tile = up_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)

                    # Clamp gate (non-linear) and up (linear) BEFORE checkpoint+SiLU,
                    # matching the golden + the backward's no-re-clamp assumption. All
                    # epilogue buffers stay fp32; the checkpoint and intermediate DMA
                    # stores downcast to compute_dtype on the way to HBM (DMA casts
                    # fp32->bf16), so no explicit cast copy is needed here.
                    #
                    # Clamp is in-place, so when it is active it must write back into
                    # the accumulator view itself (not a private tile): the gate/up
                    # checkpoint store below reads the whole block straight out of
                    # gate_sbuf/up_sbuf, so a clamped copy parked elsewhere would not
                    # be seen by it. Clamping the view in place is also one fewer
                    # tensor_copy per tile than the previous code. When clamp is
                    # inactive the accumulators are already what we want to store.
                    if clamp_active:
                        apply_activation_clamp(
                            gate_tile, clamp.non_linear_clamp_upper_limit, clamp.non_linear_clamp_lower_limit
                        )
                        apply_activation_clamp(up_tile, clamp.linear_clamp_upper_limit, clamp.linear_clamp_lower_limit)

                    # scaled = (SiLU(gate) * ea) * up, fusing the affinity fold
                    # (AFFINITY_ON_I) into the up-multiply via one scalar_tensor_tensor.
                    # ea_col is the per-token EA scalar (per-partition [actual_m,1],
                    # broadcast over I_TP); b_tile index = which TILE_M block.
                    # Both write into views of the block-wide buffers.
                    silu_out = sbm.alloc_stack(shape=(actual_m, actual_n), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.activation(dst=silu_out, op=nl.silu, data=gate_tile)
                    b_tile = m_off // TILE_M
                    ea_col = ea_tiles_all[0:actual_m, b_tile : b_tile + 1]
                    scaled = scaled_block.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=sbuf_offset)
                    nisa.scalar_tensor_tensor(
                        dst=scaled,
                        data=silu_out,
                        op0=nl.multiply,
                        operand0=ea_col,
                        op1=nl.multiply,
                        operand1=up_tile,
                    )

            # --- Block-granular stores (one DMA each, outside the tile loops) -------
            # Checkpoint the clamped gate pre-activation and up at
            # gate_up_proj_act_checkpoint_T[block_idx, {0,1}]. Skipped when the
            # checkpoint is disabled (None). The store backend for gate_up_layout
            # picks the on-HBM layout (see dispatch above).
            #
            # Store order: the gate/up checkpoints go first because they read the
            # accumulators, which are dead after this point, whereas scaled_block is
            # still being read by the scaled_intermediate store below. Issuing the
            # accumulator stores first therefore frees their SBUF earlier and shortens
            # the window in which all three block buffers are live.
            if gate_up_proj_act_checkpoint_T is not None:
                _store_checkpoint_block(
                    gate_up_layout,
                    gate_sbuf,
                    gate_up_proj_act_checkpoint_T,
                    block_idx,
                    0,
                    2,
                    block_m_off + m_base,
                    n_off,
                    num_m_tiles_in_block,
                    rows_per_tile,
                    block_width,
                    I_TP,
                    ckpt_B,
                    sbuf_step_p,
                    BLOCK_N,
                    sbm,
                )
                _store_checkpoint_block(
                    gate_up_layout,
                    up_sbuf,
                    gate_up_proj_act_checkpoint_T,
                    block_idx,
                    1,
                    2,
                    block_m_off + m_base,
                    n_off,
                    num_m_tiles_in_block,
                    rows_per_tile,
                    block_width,
                    I_TP,
                    ckpt_B,
                    sbuf_step_p,
                    BLOCK_N,
                    sbm,
                )

            # Scaled intermediate -> the down GEMM's LHS [B, I_TP] (row pitch I_TP).
            _store_block_token_major(
                src_block=scaled_block,
                dst=scaled_intermediate,
                dst_base=block_m_off * I_TP + n_off,
                row_pitch=I_TP,
                num_m_tiles=num_m_tiles_in_block,
                rows_per_tile=rows_per_tile,
                width=block_width,
                sbuf_step_p=sbuf_step_p,
                src_tile_stride=BLOCK_N,
            )

            # Optional scaled-intermediate checkpoint ([N, ...], n_halves=1).
            # The store backend for scaled_layout picks the on-HBM layout. Reads the
            # same scaled_block as the store above, so it is kept adjacent to it.
            if scaled_intermediate_checkpoint_T is not None:
                _store_checkpoint_block(
                    scaled_layout,
                    scaled_block,
                    scaled_intermediate_checkpoint_T,
                    block_idx,
                    0,
                    1,
                    block_m_off + m_base,
                    n_off,
                    num_m_tiles_in_block,
                    rows_per_tile,
                    block_width,
                    I_TP,
                    ckpt_B,
                    sbuf_step_p,
                    BLOCK_N,
                    sbm,
                )

    # Hand the weight buffers back for the next block to reuse (None unless reusing).
    return TensorDescriptor(data=scaled_intermediate), gate_wq_td, up_wq_td


def _down_projection_block(
    scaled_intermediate_td,
    down_weight_td,
    block_idx,
    expert_idx_broadcast,
    B,
    H,
    I_TP,
    E,
    config,
    sbm,
    name_suffix="",
    direct_output_shard=None,
    full_block_size=None,
    m_base=0,
    downq_td=None,
    reuse_weights=False,
):
    """Step 9 — down projection: out_block[B, H] = scaled_intermediate[B, I_TP] @ W_down[e].T.

    Per-expert slice on ``down_weight_td`` via :func:`_set_expert_offset_on_td`
    (stride = (I_TP*H)//MATMUL_TILE_K_PHYSICAL non-prequant, else data.shape[0]//E),
    then one :func:`generic_matmul_mxfp8_api` call with the scaled intermediate as
    LHS (quantized on the fly) and the per-expert down weight as RHS.

    ``name_suffix`` disambiguates the out_block shared_hbm name when both cores
    process the same ``block_idx`` (the odd-N tail half). Empty for full blocks.

    When ``direct_output_shard`` is provided (the ``no_indirect_load`` path, where
    each block maps to a disjoint contiguous output-row range written exactly once),
    each result tile is DMA'd straight into ``direct_output_shard`` at row
    ``block_idx*full_B + m_base + m_off`` (``full_B`` = ``full_block_size`` or ``B``)
    and the function returns ``None`` — this skips the per-block ``out_block`` HBM
    buffer entirely and avoids the redundant write/read round-trip that a separate
    contiguous store would incur. When it is ``None`` (the indirect scatter path),
    the result is materialized in a per-block ``out_block`` HBM tile and returned for
    :func:`_scatter_output_block`.

    Half-block tail (odd N): ``B`` is this call's token count, ``full_block_size``
    the full block size, ``m_base`` this half's offset within the block, so the two
    cores write disjoint contiguous row ranges (defaults reproduce the whole block).

    ``downq_td`` / ``reuse_weights`` mirror the gate/up phase: a non-None ``downq_td``
    is an earlier block's spilled down weight, fed straight in as the RHS. See
    :func:`_gate_up_swiglu_affinity_block` for the full contract.

    Returns:
        tuple: (out_block, downq_td) — the [B, H] per-block output contribution in
        HBM (pre-scatter, or ``None`` under ``direct_output_shard``), plus the weight
        spill buffer for the next block to reuse (None when not reusing).
    """
    write_direct = direct_output_shard is not None
    # Direct-store row base: block_idx * full_block_size + m_base, so the odd-N
    # tail half lands at its own disjoint contiguous rows (full_B == B for whole
    # blocks and m_base == 0, reproducing the plain block store).
    full_B = full_block_size if full_block_size is not None else B
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
    # Only the indirect-scatter path needs a per-block HBM staging buffer; the
    # direct path DMAs each result tile straight into the contiguous output slab.
    out_block = (
        None
        if write_direct
        else nl.ndarray(
            (B, H),
            dtype=buffer_dtype,
            buffer=nl.shared_hbm,
            name=f"fwd_out_block_{block_idx}{name_suffix}",
        )
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

    # See the gate/up phase: a carried-over downq_td makes the RHS an x4 spill buffer,
    # changing the RHS load tile shape, so resolve it before the bd.
    weights_cached = reuse_weights and downq_td != None

    lhs_load_tile_shape = _compute_load_tile_shape(scaled_intermediate_td, tiles, tiles['tile_m'])
    rhs_load_tile_shape = _compute_load_tile_shape(
        downq_td if weights_cached else down_weight_td, tiles, tiles['tile_n']
    )
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
    # spilling; the RHS down-weight buffer is skipped when prequant. The LHS buffer
    # stays per-block (F extent sized by B); only the weight RHS buffer is carried.
    intq_td = None
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
        if not down_weight_td.is_quantized and not weights_cached:
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
                rhs_hbm_td=downq_td if weights_cached else down_weight_td,
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
                rhsq_td=None if weights_cached else downq_td,
                use_scale_packing=d_cfg.enable_scale_packing,
            )

            # Store the SBUF accumulator block, clamped to the real per-block M (B)
            # and N (H) extents. output_sbuf is the fp32 matmul accumulator; the DMA
            # engine casts fp32 -> buffer_dtype (bf16) during the copy, so a strided
            # view of output_sbuf goes straight to HBM with no intermediate cast tile
            # (see _store_output_block_to_hbm in matmul_mxfp8_generic_api). The store
            # already clamps via actual_m/actual_n, so the clamp the old copy did is
            # folded into the DMA.
            sbuf_step_p = TILES_IN_BLOCK_M * BLOCK_N
            n_off_base = idx_n * BLOCK_N
            actual_n = min(BLOCK_N, H - n_off_base)
            num_m_tiles_in_block = min(TILES_IN_BLOCK_M, div_ceil(B - idx_m * TILES_IN_BLOCK_M * tile_m, tile_m))
            for tmi in range(num_m_tiles_in_block):
                m_off = (idx_m * TILES_IN_BLOCK_M + tmi) * tile_m
                actual_m = min(tile_m, B - m_off)
                if actual_m <= 0 or actual_n <= 0:
                    continue
                src_block = output_sbuf.ap(pattern=[[sbuf_step_p, actual_m], [1, actual_n]], offset=tmi * BLOCK_N)
                if write_direct:
                    # no_indirect_load: block_idx maps to a disjoint contiguous
                    # output-row range written exactly once, so store straight from
                    # SBUF into the output slab — no per-block out_block round-trip.
                    row_off = block_idx * full_B + m_base + m_off
                    nisa.dma_copy(
                        dst=direct_output_shard[row_off : row_off + actual_m, n_off_base : n_off_base + actual_n],
                        src=src_block,
                    )
                else:
                    nisa.dma_copy(
                        dst=out_block[m_off : m_off + actual_m, n_off_base : n_off_base + actual_n],
                        src=src_block,
                    )
    return out_block, downq_td


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


def _process_block(
    hidden_states_td,
    gate_up_weight_td,
    down_weight_td,
    expert_affinities_masked_td,
    output_shard,
    gate_up_proj_act_checkpoint_T,
    scaled_intermediate_checkpoint_T,
    expert_idx_broadcast,
    block_token_pos_to_id,
    block_idx,
    B_eff,
    H,
    I_TP,
    E,
    config,
    sbm,
    full_block_size=None,
    m_base=0,
    name_suffix="",
    fast_dma_transpose=False,
    gate_wq_td=None,
    up_wq_td=None,
    downq_td=None,
    gu_reuse_weights=False,
    d_reuse_weights=False,
):
    """Run the FFN for one (possibly partial) block end-to-end.

    Loads this (sub-)block's tokens + affinities, runs gate/up -> SwiGLU ->
    affinity-fold -> down, and writes the result to ``output_shard``. The input
    and output steps differ by path; the gate/up/down middle is shared:

      - Routed (indirect): gather tokens via ``block_token_pos_to_id`` (a compact
        [TILE_M, B_eff // TILE_M] index buffer, NOT a slice of a wider one — a
        slice keeps the wider row stride and mis-addresses the gather) and scatter
        the output into the per-shard slab.
      - ``no_indirect_load``: tokens are packed contiguously, so the block maps to
        rows [block_idx*full_B + m_base : ... + B_eff] directly (no gather), and
        the down projection stores each result tile straight into those contiguous
        output rows (``direct_output_shard``), skipping the per-block out_block
        HBM round-trip. When ``fast_dma_transpose`` is set (no_indirect_load only), the
        hidden LHS and scaled-intermediate LHS also take the fast DGT load path.

    Factored out of the orchestrator's block loop so the odd-N tail reuses it as a
    half-block:

      - Full block: ``B_eff = block_size``, ``full_block_size=None`` (-> B_eff),
        ``m_base=0``, ``name_suffix=""`` — byte-identical to the whole-block path.
      - Tail half: ``B_eff = block_size // 2``, ``full_block_size = block_size``,
        ``m_base = shard_id * (block_size // 2)``, ``name_suffix = f"h{shard_id}"``.
        Compute runs on the half's B_eff tokens; the checkpoint/output stores use
        the full block size as the B-axis stride and shift by ``m_base`` so the two
        cores' halves reassemble into one contiguous slot (invisible to the bwd).
        ``name_suffix`` keeps the shared_hbm scratch disjoint since both cores
        share ``block_idx``.

    The three ``*_wq_td`` weight spill buffers are threaded through unchanged: None on
    the first block a core processes (the phase allocates+fills them), then fed back
    on later blocks to skip the reload+requantize. Only valid under E == 1 (see the
    entry point's assert); returned as a triple for the caller to carry forward.
    """
    full_B = full_block_size if full_block_size is not None else B_eff

    # Steps 3 + 7a: load this (sub-)block's hidden tokens and per-token affinities.
    if config.no_indirect_load:
        hidden_block_td = TensorDescriptor(data=hidden_states_td.data[nl.ds(block_idx * full_B + m_base, B_eff), 0:H])
        hidden_block_td.fast_dma_transpose = fast_dma_transpose
        ea_tiles_all = _load_block_affinities_contiguous(
            expert_affinities_masked_td.data, block_idx, B_eff, sbm, full_B, m_base, name_suffix
        )
    else:
        hidden_block_td = _gather_hidden_block(
            hidden_states_td, block_token_pos_to_id, B_eff, H, config.skip_dma, block_idx, config, sbm, name_suffix
        )
        ea_tiles_all = _gather_block_affinities(
            expert_affinities_masked_td.data,
            block_token_pos_to_id,
            expert_idx_broadcast,
            block_idx,
            B_eff,
            E,
            config.skip_dma,
            sbm,
        )

    # Steps 4-8: gate/up -> clamp+checkpoint -> SwiGLU -> affinity fold -> checkpoint.
    scaled_intermediate_td, gate_wq_td, up_wq_td = _gate_up_swiglu_affinity_block(
        hidden_block_td=hidden_block_td,
        gate_up_weight_td=gate_up_weight_td,
        ea_tiles_all=ea_tiles_all,
        block_idx=block_idx,
        expert_idx_broadcast=expert_idx_broadcast,
        B=B_eff,
        H=H,
        I_TP=I_TP,
        E=E,
        gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
        scaled_intermediate_checkpoint_T=scaled_intermediate_checkpoint_T,
        config=config,
        sbm=sbm,
        full_block_size=full_block_size,
        m_base=m_base,
        name_suffix=name_suffix,
        gate_wq_td=gate_wq_td,
        up_wq_td=up_wq_td,
        reuse_weights=gu_reuse_weights,
    )

    # The scaled intermediate is the down GEMM's LHS (unswizzled BF16, no expert
    # offset), so it can also take the fast DGT path.
    scaled_intermediate_td.fast_dma_transpose = fast_dma_transpose

    # Step 9 (+ Step 10 on the direct path): down projection. On no_indirect_load,
    # each result tile is DMA'd straight into this shard's contiguous output slab
    # (block maps to disjoint rows written once, offset by full_block_size/m_base for
    # the tail half), skipping the per-block out_block HBM buffer and its redundant
    # write/read round-trip. On the scatter path, the result is returned in out_block
    # for the indirect scatter below.
    out_block, downq_td = _down_projection_block(
        scaled_intermediate_td,
        down_weight_td,
        block_idx,
        expert_idx_broadcast,
        B_eff,
        H,
        I_TP,
        E,
        config,
        sbm,
        name_suffix=name_suffix,
        direct_output_shard=output_shard if config.no_indirect_load else None,
        full_block_size=full_block_size,
        m_base=m_base,
        downq_td=downq_td,
        reuse_weights=d_reuse_weights,
    )

    # Step 10: scatter into this shard's output slab (direct path already stored).
    if not config.no_indirect_load:
        _scatter_output_block(
            out_block,
            output_shard,
            block_token_pos_to_id,
            B_eff,
            H,
            config.skip_dma,
            config.is_tensor_update_accumulating,
            block_idx,
            sbm,
        )

    return gate_wq_td, up_wq_td, downq_td


# =====================================================================================
# Top-level dropless forward orchestration
# =====================================================================================


@with_active_sbm
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

    # Odd-N tail balancing (LNC2 only): with an odd block count the round-robin
    # leaves one core idle during the final block. When enabled, process the
    # first N-1 (even) blocks round-robin and split the last block across the two
    # cores along the token (B) axis (core 0 -> tokens [0:B/2], core 1 -> [B/2:B]).
    # Requires B/2 to stay TILE_M(128)-aligned, i.e. NUM_B_TILES even (B >= 256);
    # for B=128 the split is skipped and the plain round-robin covers all N blocks.
    # Applies to both the indirect (routed) and no_indirect_load (contiguous
    # single-expert) paths — both write disjoint token sub-ranges for the two
    # halves. All operands are compile-time Python ints, so both cores take this
    # branch identically.
    split_tail = num_shards == 2 and N % 2 == 1 and NUM_B_TILES % 2 == 0
    num_full_blocks = N - 1 if split_tail else N

    # Fast DGT: load the unswizzled-BF16 operands via the direct 4D access pattern
    # (skips the vector_offset_pattern SBUF buffers). The fast loader addresses the
    # source as f_offset*K + k_offset and ignores per-expert scalar_offset, so it is
    # only valid on the no_indirect_load path — where the weights carry no expert
    # offset (the entry point asserts fast_dma_transpose => no_indirect_load). Enable
    # it on both weight RHS TDs here; the per-block activation LHS TDs get it at
    # construction below.
    fast_dma_transpose = config.fast_dma_transpose and config.no_indirect_load
    if fast_dma_transpose:
        gate_up_weight_td.fast_dma_transpose = True
        down_weight_td.fast_dma_transpose = True

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

        # S2: double-buffered token-index slots; prefetch this shard's first
        # full (round-robin) block. The odd-N tail half loads its own compact
        # index buffer after the loop, so it is excluded from this prefetch.
        token_indices_bufs = [
            sbm.alloc_stack((TILE_M, NUM_B_TILES), dtype=nl.int32, align=32),
            sbm.alloc_stack((TILE_M, NUM_B_TILES), dtype=nl.int32, align=32),
        ]
        if shard_id < num_full_blocks:
            _load_token_indices_dgt(token_position_to_id_td.data, shard_id, B, NUM_B_TILES, dst=token_indices_bufs[0])

    # S3: zero-init this shard's output slab (RMW scatter writes into it). The
    # no_indirect_load path overwrites every owned row directly in output_hidden_states,
    # so there is nothing to pre-zero and no cross-core visibility to barrier on.
    if not config.no_indirect_load:
        _zero_init_output(output_shard, T, H, sbm)
        nisa.core_barrier(output_shard, (0, 1))

    # Weight reuse: carry each phase's quantized weight spill buffer across the block
    # loop so only the first block this core loads+quantizes the weights; later blocks
    # read the copy from HBM scratch. Correct only because no_indirect_load asserts
    # E == 1. Resolved per phase (not once) because the autotune cache can override a
    # phase's spill_reload, and without spilling there is no buffer to carry.
    gu_reuse_weights = config.reuse_spilled_weights and config.gate_up_config.spill_reload
    d_reuse_weights = config.reuse_spilled_weights and config.down_config.spill_reload
    gate_wq_td = None
    up_wq_td = None
    downq_td = None

    # --- Per-block loop: this core processes blocks [shard_id, shard_id+num_shards, ...] ----
    # When split_tail is set this stops at num_full_blocks (= N-1); the odd tail
    # block is handled below as two half-blocks, one per core.
    ring = 0
    for block_idx in range(shard_id, num_full_blocks, num_shards):
        sbm.open_scope(name=f"FwdBlock {block_idx}")
        block_token_pos_to_id = None
        if not config.no_indirect_load:
            block_token_pos_to_id = token_indices_bufs[ring]
            # Prefetch the next full block this shard will own (tail excluded).
            next_block_idx = block_idx + num_shards
            if next_block_idx < num_full_blocks:
                nxt = 1 - ring
                _load_token_indices_dgt(
                    token_position_to_id_td.data, next_block_idx, B, NUM_B_TILES, dst=token_indices_bufs[nxt]
                )
            ring = 1 - ring

        gate_wq_td, up_wq_td, downq_td = _process_block(
            hidden_states_td=hidden_states_td,
            gate_up_weight_td=gate_up_weight_td,
            down_weight_td=down_weight_td,
            expert_affinities_masked_td=expert_affinities_masked_td,
            output_shard=output_shard,
            gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
            scaled_intermediate_checkpoint_T=scaled_intermediate_checkpoint_T,
            expert_idx_broadcast=expert_idx_broadcast,
            block_token_pos_to_id=block_token_pos_to_id,
            block_idx=block_idx,
            B_eff=B,
            H=H,
            I_TP=I_TP,
            E=E,
            config=config,
            sbm=sbm,
            fast_dma_transpose=fast_dma_transpose,
            gate_wq_td=gate_wq_td,
            up_wq_td=up_wq_td,
            downq_td=downq_td,
            gu_reuse_weights=gu_reuse_weights,
            d_reuse_weights=d_reuse_weights,
        )
        sbm.close_scope()

    # --- Odd-N tail: split the last block along B across the two cores ----------------------
    # Core shard_id computes tokens [shard_id*B_half : (shard_id+1)*B_half] of block
    # N-1. Both halves write disjoint token sub-ranges of the same checkpoint slot
    # and disjoint output rows (scatter into per-shard slabs for the routed path,
    # direct contiguous rows for no_indirect_load), so the result is byte-identical
    # to processing the whole block on one core.
    if split_tail:
        tail_idx = N - 1
        B_half = B // 2
        m_base = shard_id * B_half
        sbm.open_scope(name=f"FwdTailHalf {tail_idx}_{shard_id}")

        # Routed path: load a compact [TILE_M, NUM_B_TILES // 2] index buffer for
        # this core's half of the tail block (from token offset m_base). A fresh
        # buffer, not a slice of the full-width one, so the row stride matches the
        # compact width. no_indirect_load needs no index buffer (contiguous rows).
        tail_token_indices = None
        if not config.no_indirect_load:
            tail_token_indices = sbm.alloc_stack(
                (TILE_M, NUM_B_TILES // 2), dtype=nl.int32, name=f"fwd_tail_tok_{tail_idx}_{shard_id}", align=32
            )
            _load_token_indices_dgt(
                token_position_to_id_td.data,
                tail_idx,
                B,
                NUM_B_TILES // 2,
                dst=tail_token_indices,
                src_token_offset=m_base,
            )

        # The tail half reuses the cache: the buffers are sized by H / I_TP and the
        # phase config, never by B, so the halved token count changes nothing. And
        # split_tail implies N >= 3 (num_full_blocks >= 2), so each core ran a full
        # block above and the cache is populated.
        _process_block(
            hidden_states_td=hidden_states_td,
            gate_up_weight_td=gate_up_weight_td,
            down_weight_td=down_weight_td,
            expert_affinities_masked_td=expert_affinities_masked_td,
            output_shard=output_shard,
            gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
            scaled_intermediate_checkpoint_T=scaled_intermediate_checkpoint_T,
            expert_idx_broadcast=expert_idx_broadcast,
            block_token_pos_to_id=tail_token_indices,
            block_idx=tail_idx,
            B_eff=B_half,
            H=H,
            I_TP=I_TP,
            E=E,
            config=config,
            sbm=sbm,
            full_block_size=B,
            m_base=m_base,
            name_suffix=f"h{shard_id}",
            fast_dma_transpose=fast_dma_transpose,
            gate_wq_td=gate_wq_td,
            up_wq_td=up_wq_td,
            downq_td=downq_td,
            gu_reuse_weights=gu_reuse_weights,
            d_reuse_weights=d_reuse_weights,
        )
        sbm.close_scope()

    # Final reduce of the per-shard output slabs into the returned [T, H] output.
    # Skipped for no_indirect_load: each core already wrote its disjoint rows straight
    # into output_hidden_states, so there are no slabs to sum.
    if not config.no_indirect_load:
        _reduce_output_shards(output_hidden_states, output_slabs, num_shards, shard_id, T, H, sbm)

    sbm.close_scope()
