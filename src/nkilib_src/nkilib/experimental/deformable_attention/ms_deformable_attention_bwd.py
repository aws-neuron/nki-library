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
Multi-Scale Deformable Attention Backward kernel for NeuronCore.

This kernel implements the backward pass for multi-scale deformable attention,
computing gradients with respect to value, sampling_locations, and attention_weights.
"""

from dataclasses import dataclass
from typing import Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...core.utils.logging import get_logger

SCATTER_INDEX_ALIGN = 16  # Turbo batched indirect scatter-add DGE indicies rquire 16 byte alignment
MAX_SCATTER_BUFFERS = 8  # Upper bound on the number of K-replicated scatter buffer sets a caller may request
MAX_INTERLEAVE = 8
FP32_EXACT_INT_MAX = 1 << 24
MAX_Q_PACK = 64
SBUF_FIT_RESERVE = 4096


@nki.jit
def ms_deformable_attention_bwd(
    grad_output: nl.NkiTensor,
    value: nl.NkiTensor,
    spatial_shapes: tuple,
    level_start_index: tuple,
    sampling_locations: nl.NkiTensor,
    attention_weights: nl.NkiTensor,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    align_corners: bool = False,
    padding_mode: str = "zeros",
    max_gather_indices_per_indirect: Optional[int] = None,
    max_scatter_indices_per_indirect: Optional[int] = None,
    gather_method: str = "transpose",
    num_scatter_buffers: Optional[int] = None,
    compute_grad_value: bool = True,
    compute_grad_sampling_locations: bool = True,
    compute_grad_attention_weights: bool = True,
    iota_k_scale_in: Optional[nl.NkiTensor] = None,
    grad_value_buffers_in: Optional[nl.NkiTensor] = None,
    prev_reduced_grad_value: Optional[nl.NkiTensor] = None,
    dump_iota_k_scale: bool = False,
    dump_grad_value_buffers: bool = False,
    dump_reduced_grad_value: bool = False,
) -> Tuple[nl.NkiTensor, ...]:
    """
    Multi-scale deformable attention backward pass kernel.

    Computes gradients with respect to value, sampling_locations, and attention_weights
    given the downstream gradient.

    Dimensions:
        B: Batch size
        N_q: Number of queries
        N_h: Number of attention heads
        C_h: Channels per head
        N_l: Number of feature pyramid levels
        N_p: Number of sampling points per query per head per level
        L: Total flattened spatial dimension (sum of H_i * W_i across all levels)
        H_i: Height of feature map at level i
        W_i: Width of feature map at level i

    Args:
        grad_output (nl.NkiTensor): Gradient from downstream in HBM, shape (B, N_q, N_h * C_h)
        value (nl.NkiTensor): Value tensor in HBM. Shape depends on value_layout:
            - If value_layout="BLNC": (B, L, N_h, C_h)
            - If value_layout="BNLC": (B, N_h, L, C_h)
        spatial_shapes (tuple): Tuple of (H_i, W_i) tuples specifying spatial dimensions for each level
        level_start_index (tuple): Tuple of start indices for each level in the flattened L dimension
        sampling_locations (nl.NkiTensor): Normalized sampling coordinates in HBM. Shape depends on layout:
            - If sampling_locations_layout="BQHLP2": (B, N_q, N_h, N_l, N_p, 2)
            - If sampling_locations_layout="B2QHLP": (B, 2, N_q, N_h, N_l, N_p)
        attention_weights (nl.NkiTensor): Attention weights in HBM, shape (B, N_q, N_h, N_l, N_p)
        value_layout (str): Layout of value tensor, either "BLNC" or "BNLC". Default: "BLNC"
        sampling_locations_layout (str): Layout of sampling_locations, either "BQHLP2" or "B2QHLP". Default: "BQHLP2"
        align_corners (bool): If True, coordinates map [0,1] to [0, H-1]. If False, map to [-0.5, H-0.5]. Default: False
        padding_mode (str): Padding mode for out-of-bounds coordinates, either "zeros" or "border". Default: "zeros"
        max_gather_indices_per_indirect (Optional[int]): Cap on gather indices (P_MAX * M_batch)
            per batched gather. Setting this (not None) enables the batched indirect gather
            path (M_batch = cap // P_MAX); None disables it (each (head, level) gathered
            separately). Default: None
        max_scatter_indices_per_indirect (Optional[int]): Controls the scatter-add indirection
            budget. If None, scatters over all P_MAX query partitions with no column batching.
            If set and <= P_MAX, enables the "fewer indirections" path: at most cap query
            partitions are scattered per dma_compute. Default: None
        gather_method (str): How values are gathered: "transpose" (dma_transpose to [channel,
            query] then nc_transpose to [query, channel]) or "copy" (indirect dma_copy to
            [query, channel]). Scatter always uses dma_compute. Default: "transpose"
        iota_k_scale_in (Optional[nl.NkiTensor]): Precomputed K-scaling index tensor in HBM,
            shape (P_MAX, k_scale_cols), uint32. If given, the kernel copies it in instead of
            building it. Default: None
        grad_value_buffers_in (Optional[nl.NkiTensor]): Persistent K replicated scatter buffers
            in HBM, shape (lnc*K, L, N_h, C_h) (BLNC) / (lnc*K, N_h, L, C_h) (BNLC). If given,
            the kernel accumulates into it and skips zeroing; if None, the buffers are allocated
            and zeroed. Default: None
        prev_reduced_grad_value (Optional[nl.NkiTensor]): Reduced cumulative grad_value through
            the previous call, shape/layout of a single value map (L, N_h, C_h) / (N_h, L, C_h).
            When given, grad_value = reduce(K-buffers) - prev_reduced_grad_value, recovering this
            call's own contribution. Default: None (grad_value = reduce(K-buffers)).
        dump_iota_k_scale (bool): If True, also return the built/ingested iota_k_scale in HBM so a
            later call can pass it as iota_k_scale_in. Default: False
        dump_grad_value_buffers (bool): If True, also return the raw K replicated scatter buffers
            in HBM so a later call can pass them as grad_value_buffers_in. Default: False
        dump_reduced_grad_value (bool): If True, also return reduce(K-buffers) (cumulative through
            this call) in HBM so a later call can pass it as prev_reduced_grad_value. Default: False

    Returns:
        grad_value (nl.NkiTensor): Gradient w.r.t. value in HBM, same shape and layout as input value
        grad_sampling_locations (nl.NkiTensor): Gradient w.r.t. sampling_locations in HBM, same shape and layout as input
        grad_attention_weights (nl.NkiTensor): Gradient w.r.t. attention_weights in HBM, shape (B, N_q, N_h, N_l, N_p)
        Followed, in this fixed order, by whichever of these are enabled: iota_k_scale
        (dump_iota_k_scale), grad_value_buffers (dump_grad_value_buffers), reduced_grad_value
        (dump_reduced_grad_value).

    Pseudocode:
        # Initialize gradient buffers
        grad_value = zeros_like(value)
        grad_sampling_locations = zeros_like(sampling_locations)
        grad_attention_weights = zeros_like(attention_weights)

        for batch in range(B):
            for query_tile in range(num_query_tiles):
                for head_tile in range(num_head_tiles):
                    # Load gradients and sampling coordinates
                    grad_out_tile = load(grad_output[batch, queries, heads])
                    x, y = load(sampling_locations[batch, queries, heads])
                    attn_w = load(attention_weights[batch, queries, heads])

                    # Compute bilinear interpolation coordinates
                    x0, y0, x1, y1 = compute_bilinear_coords(x, y, spatial_shapes)

                    # Compute bilinear weights and derivatives
                    bilinear_weights = compute_bilinear_weights(x, y, x0, y0)
                    grad_x_weights, grad_y_weights = compute_bilinear_derivatives(x, y)

                    # Gather values from feature maps and accumulate loc/weight grads
                    for level in range(N_l):
                        for point in range(N_p):
                            sampled_values = gather(value, flat_indices)
                            # Gradient w.r.t. attention weights
                            grad_attention_weights += reduce_sum(grad_out * sampled_values)
                            # Gradient w.r.t. sampling locations
                            grad_sampling_locations += grad_out * grad_weights * attn_w

                    # Scatter gradients to value tensor
                    for level in range(N_l):
                        for point in range(N_p):
                            grad_value_contribution = grad_out * bilinear_weights * attn_w
                            scatter_add(grad_value, grad_value_contribution, flat_indices)

        return grad_value, grad_sampling_locations, grad_attention_weights
    """
    # Build config
    cfg = _build_config(
        value,
        spatial_shapes,
        level_start_index,
        sampling_locations,
        attention_weights,
        value_layout,
        sampling_locations_layout,
        padding_mode,
        max_gather_indices_per_indirect,
        max_scatter_indices_per_indirect,
        gather_method,
        num_scatter_buffers,
        compute_grad_value,
        compute_grad_sampling_locations,
        compute_grad_attention_weights,
    )

    # Extract config values
    P_MAX = cfg.P_MAX
    B, N_q, N_h, C_h = cfg.B, cfg.N_q, cfg.N_h, cfg.C_h
    N_l, N_p = cfg.N_l, cfg.N_p
    L = cfg.L
    Q_tile = cfg.Q_tile
    H_tile = cfg.H_tile
    num_q_tiles = cfg.num_q_tiles
    num_h_tiles = cfg.num_h_tiles
    num_C_h_tiles = cfg.num_C_h_tiles
    dtype = cfg.dtype
    total_sbuf_req = cfg.total_sbuf
    local_N_q = cfg.local_N_q
    q_start_global = cfg.q_start_global
    lnc = cfg.lnc
    shard_id = cfg.shard_id
    K = cfg.K
    N_p_hl = cfg.N_p_hl
    Q_pack = cfg.Q_pack
    hl_tile = cfg.hl_tile
    zero_tile_free_dim = cfg.zero_tile_free_dim
    h_interleave = cfg.h_interleave
    gather_interleave = cfg.gather_interleave
    scatter_interleave = cfg.scatter_interleave

    # Determine if we need to gather and/or scatter depending on selected gradients
    need_gather = compute_grad_sampling_locations or compute_grad_attention_weights
    need_scatter = compute_grad_value
    kernel_assert(
        need_gather or need_scatter,
        "at least one of compute_grad_value / compute_grad_sampling_locations / "
        "compute_grad_attention_weights must be True",
    )

    # Initialize SBUF manager
    logger = get_logger("ms_deformable_attention_bwd")
    sbm = SbufManager(0, total_sbuf_req, logger=logger)

    # Allocate output gradients in HBM
    grad_value = nl.ndarray(value.shape, dtype=dtype, buffer=nl.shared_hbm) if compute_grad_value else None
    grad_sampling_locations = (
        nl.ndarray(sampling_locations.shape, dtype=nl.float32, buffer=nl.shared_hbm)
        if compute_grad_sampling_locations
        else None
    )
    grad_attention_weights = (
        nl.ndarray(attention_weights.shape, dtype=dtype, buffer=nl.shared_hbm)
        if compute_grad_attention_weights
        else None
    )

    # Snake index scratch
    gather_elem_indexed_cfg = (cfg.batched_indirect_gather or cfg.gather_method == "copy") and (
        L * N_h * C_h
    ) < FP32_EXACT_INT_MAX

    snake_hbm = None
    if gather_elem_indexed_cfg and cfg.gather_M_batch > 1:
        snake_cols_static = cfg.gather_hl_group * N_p * 4
        snake_buf_elems_static = (snake_cols_static // cfg.gather_M_batch) * P_MAX * cfg.gather_M_batch
        snake_hbm = nl.ndarray(
            (max(1, cfg.num_gather_bufs) * snake_buf_elems_static,), dtype=nl.uint32, buffer=nl.private_hbm
        )
    snake_hbm_buf_idx = 0

    # Allocate zero buffer in shared HBM
    if value_layout == "BNLC":
        zero_buffer_hbm = nl.ndarray((N_h, L, C_h), dtype=dtype, buffer=nl.shared_hbm)
    else:  # BLNC
        zero_buffer_hbm = nl.ndarray((L, N_h, C_h), dtype=dtype, buffer=nl.shared_hbm)

    # Reduction tree: round 0 reduces each core's K scatter buffers 16-at-a-time
    round0_per_core = div_ceil(K, 16)
    round0_survivors = lnc * round0_per_core
    final_combine_offset = round0_survivors
    _max_combine_srcs = round0_survivors + (B - 1 if B > 1 else 0)
    _region_offset = final_combine_offset
    _s = _max_combine_srcs
    while _s > 16:
        _s = div_ceil(_s, 16)
        _region_offset += _s
    intermediate_buffers = _region_offset

    # Allocate the K scatter buffers in shared HBM
    n_buf_sets = cfg.num_scatter_buffers
    reuse_buffers = grad_value_buffers_in != None
    if reuse_buffers:
        kernel_assert(
            n_buf_sets == 1,
            "grad_value_buffers_in requires num_scatter_buffers=1",
        )

    # Only need replicated HBM on need_scatter
    _alloc_full = need_scatter or dump_grad_value_buffers
    K_alloc = K if _alloc_full else 1
    n_buf_sets_alloc = n_buf_sets if _alloc_full else 1
    intermediate_alloc = intermediate_buffers if _alloc_full else 1
    if value_layout == "BLNC":
        if reuse_buffers:
            grad_value_buffers_all_sets = [grad_value_buffers_in]
        else:
            grad_value_buffers_all_sets = [
                nl.ndarray((lnc * K_alloc, L, N_h, C_h), dtype=dtype, buffer=nl.shared_hbm)
                for _ in range(n_buf_sets_alloc)
            ]
        temp_intermediate = nl.ndarray((intermediate_alloc, L, N_h, C_h), dtype=dtype, buffer=nl.shared_hbm)
    else:  # BNLC
        if reuse_buffers:
            grad_value_buffers_all_sets = [grad_value_buffers_in]
        else:
            grad_value_buffers_all_sets = [
                nl.ndarray((lnc * K_alloc, N_h, L, C_h), dtype=dtype, buffer=nl.shared_hbm)
                for _ in range(n_buf_sets_alloc)
            ]
        temp_intermediate = nl.ndarray((intermediate_alloc, N_h, L, C_h), dtype=dtype, buffer=nl.shared_hbm)

    # Per-core K-buffer view for each set
    grad_value_buffers_sets = [s[shard_id * K_alloc : (shard_id + 1) * K_alloc] for s in grad_value_buffers_all_sets]

    # Default alias used by the non-double-buffer paths
    grad_value_buffers_all = grad_value_buffers_all_sets[0]
    grad_value_buffers = grad_value_buffers_sets[0]

    # Row-flattened views for the spatially-sharded final combine
    n_value_rows = L * N_h
    rows_per_core = div_ceil(n_value_rows, lnc)
    temp_intermediate_rows = temp_intermediate.reshape((intermediate_alloc, n_value_rows, C_h))

    # Reuse-across-calls: the reduced cumulative grad_value through this call
    subtract_prev = prev_reduced_grad_value != None
    reduced_grad_value_out = None
    reduced_grad_value_rows = None
    if dump_reduced_grad_value:
        if value_layout == "BLNC":
            reduced_grad_value_out = nl.ndarray((L, N_h, C_h), dtype=dtype, buffer=nl.shared_hbm)
        else:  # BNLC
            reduced_grad_value_out = nl.ndarray((N_h, L, C_h), dtype=dtype, buffer=nl.shared_hbm)
        reduced_grad_value_rows = reduced_grad_value_out.reshape((n_value_rows, C_h))
    if subtract_prev:
        prev_reduced_rows = prev_reduced_grad_value.reshape((n_value_rows, C_h))

    # ====================================================================
    # Step 1: Zero out a tile in HBM using a memset tile in SBUF
    # ====================================================================
    zero_tile = sbm.alloc_heap((P_MAX, zero_tile_free_dim), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_tile, value=0, engine=nisa.engine.vector)

    if need_scatter:
        _zero_hbm_from_tile(
            zero_tile,
            zero_buffer_hbm.reshape((N_h * L * C_h,)),
            N_h * L * C_h,
            P_MAX,
            zero_tile_free_dim,
        )

    sbm.pop_heap()

    # ====================================================================
    # Step 2: Construct the K-scaling tensor for use in atomic scatter-add
    # ====================================================================
    k_scale_cols = hl_tile * N_p_hl * 4
    initial_width = min(P_MAX // 4, k_scale_cols)

    K_base = cfg.K_base
    M_batch = cfg.scatter_M_batch

    iota_k_scale = sbm.alloc_heap((P_MAX, k_scale_cols), dtype=nl.uint32, buffer=nl.sbuf)

    if iota_k_scale_in != None:
        # Reuse-across-calls: copy the precomputed, already-K-stride-scaled index
        # tensor in from HBM instead of rebuilding it
        nisa.dma_copy(dst=iota_k_scale, src=iota_k_scale_in, dge_mode=nisa.dge_mode.none)
    else:
        iota_temp = sbm.alloc_heap((M_batch, P_MAX), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(
            dst=iota_temp,
            pattern=[[0, P_MAX // K_base], [1, K_base]],  # swizzled level/head ramp per group
            offset=0,
            channel_multiplier=K_base,  # group m -> disjoint K-block base m*K_base
        )

        # Transpose iota_temp into the first M_batch columns, tiling P_MAX in 32-chunks
        for i in range(P_MAX // 32):
            nisa.nc_transpose(
                dst=iota_k_scale[i * 32 : (i + 1) * 32, 0:M_batch],
                data=iota_temp[0:M_batch, i * 32 : (i + 1) * 32],
                engine=nisa.engine.vector,
            )
        sbm.pop_heap()

        # Replicate the M_batch columns across the full k_scale_cols width
        if k_scale_cols > M_batch:
            current_width = M_batch
            while current_width < k_scale_cols:
                copy_width = min(current_width, k_scale_cols - current_width)
                nisa.tensor_copy(
                    dst=iota_k_scale[:, current_width : current_width + copy_width],
                    src=iota_k_scale[:, 0:copy_width],
                    engine=nisa.engine.vector,
                )
                current_width += copy_width

        # Scale by K-buffer stride based on layout
        k_buffer_stride = L * N_h

        nisa.tensor_scalar(
            dst=iota_k_scale,
            data=iota_k_scale,
            op0=nl.multiply,
            operand0=int(k_buffer_stride),
            engine=nisa.engine.scalar,
        )

    # Optionally dump the iota_k_scale to HBM for later calls
    iota_k_scale_out = None
    if dump_iota_k_scale:
        iota_k_scale_out = nl.ndarray((P_MAX, k_scale_cols), dtype=nl.uint32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=iota_k_scale_out, src=iota_k_scale, dge_mode=nisa.dge_mode.none)

    # Zero the K-buffers once
    if not reuse_buffers and need_scatter:
        for gvb in grad_value_buffers_sets:
            for k in range(K):
                nisa.dma_copy(dst=gvb[k], src=zero_buffer_hbm, dge_mode=nisa.dge_mode.none)

    hl_off = sbm.alloc_heap((P_MAX, Q_pack * N_h, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
    for level_idx in range(N_l):
        if cfg.value_layout == "BLNC":
            # memory order [L, N_h]: offset = level_start * N_h + h_global
            level_head_base = level_start_index[level_idx] * N_h
            head_step = 1
        else:  # BNLC, memory order [N_h, L]
            level_head_base = level_start_index[level_idx]
            head_step = L
        nisa.iota(
            dst=hl_off[:, :, level_idx, :],
            pattern=[[0, Q_pack], [head_step, N_h], [0, N_p]],
            offset=int(level_head_base),
            channel_multiplier=0,
        )

    query_segments = []
    covered = 0
    if Q_pack > 1:
        n_groups = local_N_q // Q_pack
        n_full = (n_groups // Q_tile) * Q_tile
        for group_start in range(0, n_full, Q_tile):
            query_segments.append((group_start * Q_pack, Q_tile, Q_pack))
        covered = n_full * Q_pack
    rest = local_N_q - covered
    done = 0
    while done < rest:
        seg_parts = min(Q_tile, rest - done)
        query_segments.append((covered + done, seg_parts, 1))
        done = done + seg_parts

    # Open H-tile scope
    sbm.open_scope(interleave_degree=h_interleave)

    # Loop over batches
    for batch_idx in range(B):
        # Loop over queries
        for q_start_local, q_actual, q_pack_tile in query_segments:
            q_end_local = q_start_local + q_actual * q_pack_tile

            # Calculate global query indices for reading from HBM
            q_start = q_start_global + q_start_local
            q_end = q_start_global + q_end_local

            # Loop over head tiles
            for h_tile_idx in range(num_h_tiles):
                h_start = h_tile_idx * H_tile
                h_end = min(h_start + H_tile, N_h)
                h_actual = h_end - h_start

                qh = q_pack_tile * h_actual

                # ====================================================================
                # Step 4: Load grad_output, x, y, attn_w from HBM
                # ====================================================================
                grad_out = sbm.alloc_stack((q_actual, qh, C_h), dtype=dtype, buffer=nl.sbuf)

                nisa.dma_copy(
                    dst=grad_out.reshape((q_actual, qh * C_h)),
                    src=grad_output[batch_idx, q_start:q_end, h_start * C_h : h_end * C_h],
                    dge_mode=nisa.dge_mode.none,
                )

                # Load sampling locations (x, y)
                x = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                y = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)

                if cfg.sampling_locations_layout == "BQHLP2":
                    # sampling_locations: (B, N_q, N_h, N_l, N_p, 2)
                    xy = sbm.alloc_stack((q_actual, qh, N_l, N_p, 2), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.dma_copy(
                        dst=xy,
                        src=sampling_locations[batch_idx, q_start:q_end, h_start:h_end, :, :, :],
                        dge_mode=nisa.dge_mode.none,
                    )
                    # Strided source views
                    x_src = xy[:, :, :, :, 0]
                    y_src = xy[:, :, :, :, 1]
                else:
                    # sampling_locations: (B, 2, N_q, N_h, N_l, N_p)
                    nisa.dma_copy(
                        dst=x,
                        src=sampling_locations[batch_idx, 0, q_start:q_end, h_start:h_end, :, :],
                        dge_mode=nisa.dge_mode.none,
                    )
                    nisa.dma_copy(
                        dst=y,
                        src=sampling_locations[batch_idx, 1, q_start:q_end, h_start:h_end, :, :],
                        dge_mode=nisa.dge_mode.none,
                    )
                    # Contigious source views
                    x_src = x
                    y_src = y

                # Load attn_w
                attn_w = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=attn_w,
                    src=attention_weights[batch_idx, q_start:q_end, h_start:h_end, :, :],
                    dge_mode=nisa.dge_mode.none,
                )

                # ====================================================================
                # Step 5: Scale coordinates by spatial dimensions
                # ====================================================================
                for l in range(N_l):
                    H_l, W_l = spatial_shapes[l]
                    if align_corners:
                        # align_corners=True: [0,1] -> [0, W_l-1] / [0, H_l-1]
                        x_scale, y_scale, shift = float(W_l - 1), float(H_l - 1), None
                    else:
                        # align_corners=False: [0,1] -> [-0.5, W_l-0.5] / [-0.5, H_l-0.5]
                        x_scale, y_scale, shift = float(W_l), float(H_l), -0.5
                    for dst_t, src_t, scale in ((x, x_src, x_scale), (y, y_src, y_scale)):
                        kw = {} if shift == None else {"op1": nl.add, "operand1": shift}
                        nisa.tensor_scalar(
                            dst=dst_t[:, :, l, :],
                            data=src_t[:, :, l, :],
                            op0=nl.multiply,
                            operand0=scale,
                            engine=nisa.engine.scalar,
                            **kw,
                        )

                # ====================================================================
                # Step 6: Compute bilinear coordinates
                # ====================================================================
                x0_unclamped = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                y0_unclamped = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                x1_unclamped = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                y1_unclamped = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)

                x0_unclamped[0:q_actual, :, :, :] = nl.floor(x=x[0:q_actual, :, :, :], dtype=nl.int32)
                y0_unclamped[0:q_actual, :, :, :] = nl.floor(x=y[0:q_actual, :, :, :], dtype=nl.int32)

                nisa.tensor_scalar(
                    dst=x1_unclamped,
                    data=x0_unclamped,
                    op0=nl.add,
                    operand0=1,
                    engine=nisa.engine.scalar,
                )
                nisa.tensor_scalar(
                    dst=y1_unclamped,
                    data=y0_unclamped,
                    op0=nl.add,
                    operand0=1,
                    engine=nisa.engine.scalar,
                )

                # ====================================================================
                # Step 7: Clamp coordinates
                # ====================================================================
                x0_clamped = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
                y0_clamped = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
                x1_clamped = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
                y1_clamped = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)

                for l in range(N_l):
                    H_l, W_l = spatial_shapes[l]

                    # x0, x1
                    nisa.tensor_scalar(
                        dst=x0_clamped[:, :, l, :],
                        data=x0_unclamped[:, :, l, :],
                        op0=nl.minimum,
                        operand0=float(W_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_scalar(
                        dst=x1_clamped[:, :, l, :],
                        data=x1_unclamped[:, :, l, :],
                        op0=nl.minimum,
                        operand0=float(W_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                        engine=nisa.engine.vector,
                    )

                    # y0, y1
                    nisa.tensor_scalar(
                        dst=y0_clamped[:, :, l, :],
                        data=y0_unclamped[:, :, l, :],
                        op0=nl.minimum,
                        operand0=float(H_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_scalar(
                        dst=y1_clamped[:, :, l, :],
                        data=y1_unclamped[:, :, l, :],
                        op0=nl.minimum,
                        operand0=float(H_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                        engine=nisa.engine.vector,
                    )

                # ====================================================================
                # Step 8: Compute fractional contributions
                # ====================================================================
                dx_dy = sbm.alloc_stack((q_actual, 2, qh, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                one_minus_dx_dy = sbm.alloc_stack((q_actual, 2, qh, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                dx_dy_minus_one = sbm.alloc_stack((q_actual, 2, qh, N_l, N_p), dtype=dtype, buffer=nl.sbuf)

                # Compute dx, dy
                nisa.tensor_tensor(
                    dst=dx_dy[:, 0, :, :, :],
                    data1=x,
                    data2=x0_unclamped,
                    op=nl.subtract,
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=dx_dy[:, 1, :, :, :],
                    data1=y,
                    data2=y0_unclamped,
                    op=nl.subtract,
                    engine=nisa.engine.vector,
                )

                # Compute 1 - dx, 1 - dy
                dx_dy_2d = dx_dy.reshape((q_actual, 2 * qh * N_l * N_p))
                one_minus_dx_dy_2d = one_minus_dx_dy.reshape((q_actual, 2 * qh * N_l * N_p))
                nisa.tensor_scalar(
                    dst=one_minus_dx_dy_2d[0:q_actual, :],
                    data=dx_dy_2d[0:q_actual, :],
                    op0=nl.multiply,
                    operand0=-1.0,
                    op1=nl.add,
                    operand1=1.0,
                    engine=nisa.engine.scalar,
                )

                # Compute dx - 1, dy - 1
                dx_dy_minus_one_2d = dx_dy_minus_one.reshape((q_actual, 2 * qh * N_l * N_p))
                nisa.tensor_scalar(
                    dst=dx_dy_minus_one_2d[0:q_actual, :],
                    data=one_minus_dx_dy_2d[0:q_actual, :],
                    op0=nl.multiply,
                    operand0=-1.0,
                    engine=nisa.engine.scalar,
                )

                # ====================================================================
                # Step 9: Compute bilinear weights
                # ====================================================================
                bilinear_weights = sbm.alloc_stack((q_actual, 4, qh, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=bilinear_weights[0:q_actual, 0, :, :, :],
                    data1=one_minus_dx_dy[0:q_actual, 0, :, :, :],
                    data2=one_minus_dx_dy[0:q_actual, 1, :, :, :],
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=bilinear_weights[0:q_actual, 1, :, :, :],
                    data1=one_minus_dx_dy[0:q_actual, 1, :, :, :],
                    data2=dx_dy[0:q_actual, 0, :, :, :],
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=bilinear_weights[0:q_actual, 2, :, :, :],
                    data1=dx_dy[0:q_actual, 1, :, :, :],
                    data2=one_minus_dx_dy[0:q_actual, 0, :, :, :],
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=bilinear_weights[0:q_actual, 3, :, :, :],
                    data1=dx_dy[0:q_actual, 0, :, :, :],
                    data2=dx_dy[0:q_actual, 1, :, :, :],
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )

                # ====================================================================
                # Step 10: Compute flat indexes
                # ====================================================================
                flat_idx_all = sbm.alloc_stack((q_actual, qh, N_l, N_p, 4), dtype=nl.uint32, buffer=nl.sbuf)
                row_stride = N_h if cfg.value_layout == "BLNC" else 1
                x_term = sbm.alloc_stack((q_actual, 2, qh, N_p), dtype=nl.uint32, buffer=nl.sbuf)

                for l in range(N_l):
                    H_l, W_l = spatial_shapes[l]

                    for x_sel in range(2):
                        nisa.scalar_tensor_tensor(
                            dst=x_term[:, x_sel, :, :],
                            data=(x0_clamped if x_sel == 0 else x1_clamped)[:, :, l, :],
                            op0=nl.multiply,
                            operand0=float(row_stride),
                            op1=nl.add,
                            operand1=hl_off[0:q_actual, 0:qh, l, :],
                        )

                    # corner order is (y0x0, y0x1, y1x0, y1x1)
                    for corner in range(4):
                        y_clamped = y0_clamped if corner // 2 == 0 else y1_clamped
                        nisa.scalar_tensor_tensor(
                            dst=flat_idx_all[:, :, l, :, corner],
                            data=y_clamped[:, :, l, :],
                            op0=nl.multiply,
                            operand0=float(W_l * row_stride),
                            op1=nl.add,
                            operand1=x_term[:, corner % 2, :, :],
                        )

                # ====================================================================
                # Step 11: Handle padding_mode="zeros" OOB masking
                # ====================================================================
                if padding_mode == "zeros":
                    oob_scope = cfg.value_layout != "BNLC" or not cfg.combined_scatter_corners
                    if oob_scope:
                        sbm.open_scope()
                    # Compute coordinate OOB masks
                    coord_oob = sbm.alloc_stack((q_actual, 4, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                    x0_clamped_i = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                    y0_clamped_i = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                    x1_clamped_i = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                    y1_clamped_i = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=x0_clamped_i, src=x0_clamped, engine=nisa.engine.scalar)
                    nisa.tensor_copy(dst=y0_clamped_i, src=y0_clamped, engine=nisa.engine.vector)
                    nisa.tensor_copy(dst=x1_clamped_i, src=x1_clamped, engine=nisa.engine.vector)
                    nisa.tensor_copy(dst=y1_clamped_i, src=y1_clamped, engine=nisa.engine.scalar)
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 0, :, :, :],
                        data1=x0_clamped_i,
                        data2=x0_unclamped,
                        op=nl.subtract,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 1, :, :, :],
                        data1=x1_unclamped,
                        data2=x1_clamped_i,
                        op=nl.subtract,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 2, :, :, :],
                        data1=y0_clamped_i,
                        data2=y0_unclamped,
                        op=nl.subtract,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 3, :, :, :],
                        data1=y1_unclamped,
                        data2=y1_clamped_i,
                        op=nl.subtract,
                        engine=nisa.engine.vector,
                    )

                    # Normalize coord_oob to a strict 0/1 "is this coordinate OOB?" boolean
                    coord_oob_2d = coord_oob.reshape((q_actual, 4 * qh * N_l * N_p))
                    nisa.tensor_scalar(
                        dst=coord_oob_2d,
                        data=coord_oob_2d,
                        op0=nl.minimum,
                        operand0=1,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_scalar(
                        dst=coord_oob_2d,
                        data=coord_oob_2d,
                        op0=nl.maximum,
                        operand0=-1,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=coord_oob_2d,
                        data1=coord_oob_2d,
                        data2=coord_oob_2d,
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )

                    # Compute inverse coordinate OOB masks
                    one_minus_coord_oob = sbm.alloc_stack((q_actual, 4, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)

                    one_minus_coord_oob_2d = one_minus_coord_oob.reshape((q_actual, 4 * qh * N_l * N_p))
                    nisa.tensor_scalar(
                        dst=one_minus_coord_oob_2d,
                        data=coord_oob_2d,
                        op0=nl.multiply,
                        operand0=-1,
                        op1=nl.add,
                        operand1=1,
                        engine=nisa.engine.scalar,
                    )

                    # Compute corner OOB masks
                    zeros_mask = sbm.alloc_stack((q_actual, 4, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=zeros_mask[:, 0, :, :, :],
                        data1=one_minus_coord_oob[:, 0, :, :, :],
                        data2=one_minus_coord_oob[:, 2, :, :, :],
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=zeros_mask[:, 1, :, :, :],
                        data1=one_minus_coord_oob[:, 1, :, :, :],
                        data2=one_minus_coord_oob[:, 2, :, :, :],
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=zeros_mask[:, 2, :, :, :],
                        data1=one_minus_coord_oob[:, 0, :, :, :],
                        data2=one_minus_coord_oob[:, 3, :, :, :],
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=zeros_mask[:, 3, :, :, :],
                        data1=one_minus_coord_oob[:, 1, :, :, :],
                        data2=one_minus_coord_oob[:, 3, :, :, :],
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )

                    # Update bilinear weights with corner OOB masks
                    bilinear_weights_2d = bilinear_weights.reshape((q_actual, 4 * qh * N_l * N_p))
                    zeros_mask_2d = zeros_mask.reshape((q_actual, 4 * qh * N_l * N_p))
                    nisa.tensor_tensor(
                        dst=bilinear_weights_2d,
                        data1=bilinear_weights_2d,
                        data2=zeros_mask_2d,
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )

                    flat_idx_shifted = None
                    # ====================================================================
                    # Step 11b: Shift for combined corners in BNLC layout
                    # ====================================================================
                    if cfg.value_layout == "BNLC" and cfg.combined_scatter_corners:
                        coord_oob_f = sbm.alloc_stack((q_actual, 4, qh, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                        one_minus_coord_oob_f = sbm.alloc_stack(
                            (q_actual, 4, qh, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf
                        )
                        nisa.tensor_copy(
                            dst=coord_oob_f.reshape((q_actual, 4 * qh * N_l * N_p)),
                            src=coord_oob_2d,
                            engine=nisa.engine.vector,
                        )
                        nisa.tensor_copy(
                            dst=one_minus_coord_oob_f.reshape((q_actual, 4 * qh * N_l * N_p)),
                            src=one_minus_coord_oob_2d,
                            engine=nisa.engine.vector,
                        )

                        # Apply shift to flat_idx_00 (only x1_oob matters for x-direction shift)
                        flat_idx_shifted = sbm.alloc_stack((q_actual, qh, N_l, N_p, 2), dtype=nl.uint32, buffer=nl.sbuf)

                        # uint32 copy of the x1_oob 0/1 mask
                        x1_oob_u = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
                        nisa.tensor_copy(dst=x1_oob_u, src=coord_oob[:, 1, :, :, :], engine=nisa.engine.scalar)

                        # Left pair: flat_idx_00 - (x1_oob * C_h)
                        # (corners 00 and 10 share the same x0 coordinate, so they collapse together)
                        nisa.tensor_tensor(
                            dst=flat_idx_shifted[:, :, :, :, 0],
                            data1=flat_idx_all[:, :, :, :, 0],  # flat_idx_00
                            data2=x1_oob_u,  # x1_oob
                            op=nl.subtract,
                            engine=nisa.engine.vector,
                        )

                        # Right pair: flat_idx_10 - (x1_oob * C_h)
                        # (corners 01 and 11 share the same x1 coordinate, so they collapse together)
                        nisa.tensor_tensor(
                            dst=flat_idx_shifted[:, :, :, :, 1],
                            data1=flat_idx_all[:, :, :, :, 2],  # flat_idx_10
                            data2=x1_oob_u,  # x1_oob
                            op=nl.subtract,
                            engine=nisa.engine.vector,
                        )

                        # Scale the shifted indices by C_h
                        flat_idx_shifted_2d = flat_idx_shifted.reshape((q_actual, qh * N_l * N_p * 2))
                        nisa.tensor_scalar(
                            dst=flat_idx_shifted_2d,
                            data=flat_idx_shifted_2d,
                            op0=nl.multiply,
                            operand0=int(C_h),
                            engine=nisa.engine.scalar,
                        )
                    if oob_scope:
                        sbm.close_scope()

                # ====================================================================
                # Step 12: Compute combined weights (bilinear weights * attention weights)
                # ====================================================================
                combined_weights = sbm.alloc_stack((q_actual, 4, qh, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 0, :, :, :],
                    data1=bilinear_weights[0:q_actual, 0, :, :, :],
                    data2=attn_w[0:q_actual, :, :, :],
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 1, :, :, :],
                    data1=bilinear_weights[0:q_actual, 1, :, :, :],
                    data2=attn_w[0:q_actual, :, :, :],
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 2, :, :, :],
                    data1=bilinear_weights[0:q_actual, 2, :, :, :],
                    data2=attn_w[0:q_actual, :, :, :],
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 3, :, :, :],
                    data1=bilinear_weights[0:q_actual, 3, :, :, :],
                    data2=attn_w[0:q_actual, :, :, :],
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )

                # ====================================================================
                # Step 13: Compute gradient weight masks for sampling locations
                # ====================================================================

                """
                Compute gradient weights for ∂bilinear/∂x and ∂bilinear/∂y
                ∂bilinear/∂x = [-(1-dy), (1-dy), -dy,  dy ]
                ∂bilinear/∂y = [-(1-dx), -dx,   (1-dx), dx ]
                """
                grad_x_weights = sbm.alloc_stack((q_actual, 4, qh, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                grad_y_weights = sbm.alloc_stack((q_actual, 4, qh, N_l, N_p), dtype=dtype, buffer=nl.sbuf)

                # corner 0: -(1-dy) = (dy-1)
                nisa.tensor_copy(
                    dst=grad_x_weights[:, 0, :, :, :],
                    src=dx_dy_minus_one[:, 1, :, :, :],
                    engine=nisa.engine.vector,
                )
                # corner 1: (1-dy)
                nisa.tensor_copy(
                    dst=grad_x_weights[:, 1, :, :, :],
                    src=one_minus_dx_dy[:, 1, :, :, :],
                    engine=nisa.engine.vector,
                )
                # corner 2: -dy
                nisa.tensor_scalar(
                    dst=grad_x_weights[:, 2, :, :, :],
                    data=dx_dy[:, 1, :, :, :],
                    op0=nl.multiply,
                    operand0=-1.0,
                    engine=nisa.engine.scalar,
                )
                # corner 3: dy
                nisa.tensor_copy(
                    dst=grad_x_weights[:, 3, :, :, :],
                    src=dx_dy[:, 1, :, :, :],
                    engine=nisa.engine.scalar,
                )

                # corner 0: -(1-dx) = (dx-1)
                nisa.tensor_copy(
                    dst=grad_y_weights[:, 0, :, :, :],
                    src=dx_dy_minus_one[:, 0, :, :, :],
                    engine=nisa.engine.vector,
                )
                # corner 1: -dx
                nisa.tensor_scalar(
                    dst=grad_y_weights[:, 1, :, :, :],
                    data=dx_dy[:, 0, :, :, :],
                    op0=nl.multiply,
                    operand0=-1.0,
                    engine=nisa.engine.scalar,
                )
                # corner 2: (1-dx)
                nisa.tensor_copy(
                    dst=grad_y_weights[:, 2, :, :, :],
                    src=one_minus_dx_dy[:, 0, :, :, :],
                    engine=nisa.engine.vector,
                )
                # corner 3: dx
                nisa.tensor_copy(
                    dst=grad_y_weights[:, 3, :, :, :],
                    src=dx_dy[:, 0, :, :, :],
                    engine=nisa.engine.scalar,
                )

                # ====================================================================
                # Step 13b: Apply OOB masking if padding_mode="zeros"
                # ====================================================================
                if padding_mode == "zeros":
                    grad_x_weights_2d = grad_x_weights.reshape((q_actual, 4 * qh * N_l * N_p))
                    grad_y_weights_2d = grad_y_weights.reshape((q_actual, 4 * qh * N_l * N_p))

                    nisa.tensor_tensor(
                        dst=grad_x_weights_2d,
                        data1=grad_x_weights_2d,
                        data2=zeros_mask_2d,
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )

                    nisa.tensor_tensor(
                        dst=grad_y_weights_2d,
                        data1=grad_y_weights_2d,
                        data2=zeros_mask_2d,
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )

                # ====================================================================
                # Step 14: Gather and compute attention_weights and sampling_locations gradients
                # ====================================================================
                grad_attn_w_local = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                grad_sampling_loc_local = sbm.alloc_stack((q_actual, qh, N_l, N_p, 2), dtype=nl.float32, buffer=nl.sbuf)

                if not need_gather:
                    nisa.memset(dst=grad_attn_w_local, value=0, engine=nisa.engine.vector)
                    nisa.memset(dst=grad_sampling_loc_local, value=0, engine=nisa.engine.vector)

                if need_gather:
                    total_corners = N_p * 4
                    num_C_h_chunks = div_ceil(C_h, P_MAX)
                    min_C_h_P_MAX = min(C_h, P_MAX)

                    batch_offset = batch_idx * L * N_h * C_h

                    grouped_gather = cfg.batched_indirect_gather or cfg.gather_method == "copy"

                    gather_elem_indexed = gather_elem_indexed_cfg
                    value_rows = (
                        value.reshape((B * L * N_h * C_h,))
                        if gather_elem_indexed
                        else value.reshape((B * L * N_h, C_h))
                    )
                    flat_idx_all_reshaped = flat_idx_all.reshape((q_actual, qh, N_l, total_corners))

                    hl_pairs = [(h, l) for h in range(qh) for l in range(N_l)]
                    hl_group = max(1, min(cfg.gather_hl_group, len(hl_pairs)))
                    flat_idx_hl = flat_idx_all.reshape((q_actual, qh * N_l * total_corners))

                    sbm.open_scope(interleave_degree=gather_interleave, name="msda_bwd_gather_group_scope")

                    for g0 in range(0, len(hl_pairs), hl_group):
                        group = hl_pairs[g0 : g0 + hl_group]
                        group_cols = len(group) * total_corners

                        gathered_group = None
                        if grouped_gather:
                            # Zero-pad the index to P_MAX partitions.
                            effective_q = P_MAX
                            gathered_group_flat = sbm.alloc_stack(
                                (P_MAX, group_cols * C_h), dtype=dtype, buffer=nl.sbuf, align=32
                            )
                            gathered_group_cols = gathered_group_flat.reshape((P_MAX, group_cols, C_h))

                            idx_group = sbm.alloc_stack((P_MAX, group_cols), dtype=nl.uint32, buffer=nl.sbuf)
                            if q_actual < P_MAX:
                                nisa.memset(dst=idx_group, value=0, engine=nisa.engine.vector)
                            idx_src = flat_idx_hl[:, g0 * total_corners : (g0 + len(group)) * total_corners]
                            if gather_elem_indexed:
                                nisa.tensor_copy(
                                    dst=idx_group[0:q_actual, :],
                                    src=idx_src,
                                    engine=nisa.engine.vector,
                                )
                                nisa.tensor_scalar(
                                    dst=idx_group[0:q_actual, :],
                                    data=idx_group[0:q_actual, :],
                                    op0=nl.multiply,
                                    operand0=int(C_h),
                                    engine=nisa.engine.scalar,
                                )
                            else:
                                nisa.tensor_copy(
                                    dst=idx_group[0:q_actual, :],
                                    src=idx_src,
                                    engine=nisa.engine.vector,
                                )

                            M = cfg.gather_M_batch if cfg.batched_indirect_gather else 1
                            M = min(M, group_cols)
                            while M > 1 and group_cols % M != 0:
                                M -= 1

                            idx_gather = idx_group
                            if gather_elem_indexed and M > 1 and snake_hbm != None:
                                snake_ng = group_cols // M
                                snake_buf_elems = snake_ng * effective_q * M
                                snake_buf_off = snake_hbm_buf_idx * snake_buf_elems
                                nisa.dma_copy(
                                    dst=snake_hbm.ap(
                                        [[M, effective_q], [effective_q * M, snake_ng], [1, M]],
                                        offset=snake_buf_off,
                                    ),
                                    src=idx_group[0:effective_q, 0:group_cols].reshape_dim(1, [snake_ng, M]),
                                    dge_mode=nisa.dge_mode.hwdge,
                                )
                                idx_gather = sbm.alloc_stack(
                                    (P_MAX, group_cols), dtype=nl.uint32, buffer=nl.sbuf, align=32
                                )
                                nisa.dma_transpose(
                                    dst=idx_gather,
                                    src=snake_hbm.ap(
                                        [[effective_q, snake_ng * M], [1, effective_q]], offset=snake_buf_off
                                    ),
                                    axes=(1, 0),
                                )
                                snake_hbm_buf_idx = (snake_hbm_buf_idx + 1) % max(1, cfg.num_gather_bufs)

                            for run in range(group_cols // M):
                                c0 = run * M
                                NTOT = effective_q * M
                                idx_run = idx_gather[0:effective_q, c0 : c0 + M]

                                if cfg.gather_method == "copy":
                                    nisa.dma_copy(
                                        dst=gathered_group_flat[0:effective_q, c0 * C_h : (c0 + M) * C_h],
                                        src=value_rows.ap(
                                            pattern=[[C_h, NTOT], [1, C_h]],
                                            offset=batch_offset,
                                            vector_offset=idx_run,
                                            indirect_dim=0,
                                        ),
                                    )
                                else:
                                    gathered_cq = sbm.alloc_stack(
                                        (min_C_h_P_MAX, num_C_h_tiles, M * P_MAX),
                                        dtype=dtype,
                                        buffer=nl.sbuf,
                                        align=32,
                                    )
                                    for c_tile_idx in range(num_C_h_tiles):
                                        c_start = c_tile_idx * P_MAX
                                        c_end = min(c_start + P_MAX, C_h)
                                        c_actual = c_end - c_start
                                        nisa.dma_transpose(
                                            dst=gathered_cq[0:c_actual, c_tile_idx, :],
                                            src=value_rows.ap(
                                                pattern=[[c_actual, NTOT], [1, c_actual]],
                                                offset=c_start + batch_offset,
                                                vector_offset=idx_run,
                                                indirect_dim=0,
                                            ),
                                            axes=(1, 0),
                                        )

                                    gathered_cq_r = gathered_cq.reshape((min_C_h_P_MAX, num_C_h_tiles, P_MAX, M))
                                    for m in range(M):
                                        corner_ps = nl.ndarray((P_MAX, C_h), dtype=dtype, buffer=nl.psum)
                                        for c_tile_idx in range(num_C_h_tiles):
                                            c_start = c_tile_idx * P_MAX
                                            c_end = min(c_start + P_MAX, C_h)
                                            c_actual = c_end - c_start
                                            nisa.nc_transpose(
                                                dst=corner_ps[:, c_start:c_end],
                                                data=gathered_cq_r[0:c_actual, c_tile_idx, :, m],
                                                engine=nisa.engine.tensor,
                                            )
                                        nisa.tensor_copy(
                                            dst=gathered_group_cols[:, c0 + m, :],
                                            src=corner_ps,
                                            engine=nisa.engine.scalar,
                                        )

                            gathered_group = gathered_group_flat.reshape((P_MAX, len(group), total_corners, C_h))

                        sbm.open_scope(
                            interleave_degree=max(1, min(gather_interleave, len(group))),
                            name="msda_bwd_gather_block_scope",
                        )

                        for gi, (h, l) in enumerate(group):
                            H_l, W_l = spatial_shapes[l]

                            # Pre-scale attention weights by H_l and W_l for this level
                            attn_w_scaled_x = sbm.alloc_stack((q_actual, N_p), dtype=dtype, buffer=nl.sbuf)
                            attn_w_scaled_y = sbm.alloc_stack((q_actual, N_p), dtype=dtype, buffer=nl.sbuf)

                            nisa.tensor_scalar(
                                dst=attn_w_scaled_x,
                                data=attn_w[:, h, l, :],
                                op0=nl.multiply,
                                operand0=float(W_l),
                                engine=nisa.engine.scalar,
                            )

                            nisa.tensor_scalar(
                                dst=attn_w_scaled_y,
                                data=attn_w[:, h, l, :],
                                op0=nl.multiply,
                                operand0=float(H_l),
                                engine=nisa.engine.scalar,
                            )

                            if gathered_group != None:
                                gathered_reshaped = None
                                gathered_qc = gathered_group[:, gi, :, :]
                            else:
                                gathered_all = sbm.alloc_stack(
                                    (P_MAX, num_C_h_chunks * total_corners * P_MAX),
                                    dtype=dtype,
                                    buffer=nl.sbuf,
                                    align=32,
                                )

                                vector_offset_view = None
                                if q_actual < P_MAX:
                                    flat_idx_padded = sbm.alloc_stack(
                                        (P_MAX, total_corners), dtype=nl.uint32, buffer=nl.sbuf
                                    )
                                    nisa.memset(dst=flat_idx_padded, value=0, engine=nisa.engine.vector)
                                    nisa.tensor_copy(
                                        dst=flat_idx_padded[0:q_actual, :],
                                        src=flat_idx_all_reshaped[:, h, l, :],
                                        engine=nisa.engine.vector,
                                    )
                                    vector_offset_view = flat_idx_padded
                                else:
                                    vector_offset_view = flat_idx_all_reshaped[:, h, l, :]

                                for c_tile_idx in range(num_C_h_tiles):
                                    c_start = c_tile_idx * P_MAX
                                    c_end = min(c_start + P_MAX, C_h)
                                    c_actual = c_end - c_start

                                    chunk_idx = c_start // P_MAX
                                    offset_in_chunk = c_start % P_MAX
                                    col_start = chunk_idx * total_corners * P_MAX
                                    col_end = col_start + total_corners * P_MAX

                                    nisa.dma_transpose(
                                        dst=gathered_all[
                                            offset_in_chunk : offset_in_chunk + c_actual, col_start:col_end
                                        ],
                                        src=value_rows.ap(
                                            pattern=[[c_actual, total_corners * P_MAX], [1, c_actual]],
                                            offset=c_start + batch_offset,
                                            vector_offset=vector_offset_view,
                                            indirect_dim=0,
                                        ),
                                        axes=(1, 0),
                                    )

                                gathered_reshaped = gathered_all.reshape((P_MAX, num_C_h_chunks, N_p, 4, P_MAX))
                                gathered_qc = None

                            if gathered_qc != None:
                                gathered_pc = gathered_qc.reshape((P_MAX, N_p, 4, C_h))

                                # prod_all[:, p, corner, :] = grad_out * G[:, p, corner, :]
                                grad_out_bc = (
                                    grad_out[:, h, :].expand_dim(1).expand_dim(1).broadcast(1, N_p).broadcast(2, 4)
                                )
                                prod_all = sbm.alloc_stack((q_actual, N_p, 4, C_h), dtype=nl.float32, buffer=nl.sbuf)
                                nisa.tensor_tensor(
                                    dst=prod_all,
                                    data1=gathered_pc[0:q_actual, :, :, :],
                                    data2=grad_out_bc,
                                    op=nl.multiply,
                                    engine=nisa.engine.vector,
                                )

                                # dc_all[:, p, corner] = sum_c grad_out * G  (shared dot product)
                                dc_all = sbm.alloc_stack((q_actual, N_p, 4), dtype=nl.float32, buffer=nl.sbuf)
                                nisa.tensor_reduce(
                                    dst=dc_all,
                                    op=nl.add,
                                    data=prod_all,
                                    axis=(3,),
                                )

                                grad_x_dot = sbm.alloc_stack((q_actual, N_p), dtype=nl.float32, buffer=nl.sbuf)
                                grad_y_dot = sbm.alloc_stack((q_actual, N_p), dtype=nl.float32, buffer=nl.sbuf)
                                wtmp = sbm.alloc_stack((q_actual, N_p, 4), dtype=nl.float32, buffer=nl.sbuf)

                                bw = bilinear_weights[:, :, h, l, :].permute((0, 2, 1))
                                gx = grad_x_weights[:, :, h, l, :].permute((0, 2, 1))
                                gy = grad_y_weights[:, :, h, l, :].permute((0, 2, 1))

                                nisa.tensor_tensor(
                                    dst=wtmp, data1=dc_all, data2=bw, op=nl.multiply, engine=nisa.engine.vector
                                )
                                if compute_grad_attention_weights:
                                    nisa.tensor_reduce(
                                        dst=grad_attn_w_local[:, h, l, :], op=nl.add, data=wtmp, axis=(2,)
                                    )

                                if compute_grad_sampling_locations:
                                    nisa.tensor_tensor(
                                        dst=wtmp, data1=dc_all, data2=gx, op=nl.multiply, engine=nisa.engine.vector
                                    )
                                    nisa.tensor_reduce(dst=grad_x_dot, op=nl.add, data=wtmp, axis=(2,))

                                    nisa.tensor_tensor(
                                        dst=wtmp, data1=dc_all, data2=gy, op=nl.multiply, engine=nisa.engine.vector
                                    )
                                    nisa.tensor_reduce(dst=grad_y_dot, op=nl.add, data=wtmp, axis=(2,))

                                    nisa.tensor_tensor(
                                        dst=grad_sampling_loc_local[:, h, l, :, 0],
                                        data1=grad_x_dot,
                                        data2=attn_w_scaled_x,
                                        op=nl.multiply,
                                        engine=nisa.engine.vector,
                                    )
                                    nisa.tensor_tensor(
                                        dst=grad_sampling_loc_local[:, h, l, :, 1],
                                        data1=grad_y_dot,
                                        data2=attn_w_scaled_y,
                                        op=nl.multiply,
                                        engine=nisa.engine.vector,
                                    )
                            else:
                                for p in range(N_p):
                                    dc = sbm.alloc_stack((q_actual, 1), dtype=nl.float32, buffer=nl.sbuf)
                                    prod = sbm.alloc_stack((q_actual, C_h), dtype=nl.float32, buffer=nl.sbuf)

                                    grad_attn_acc = sbm.alloc_stack((q_actual, 1), dtype=nl.float32, buffer=nl.sbuf)
                                    grad_x_dot = sbm.alloc_stack((q_actual, 1), dtype=nl.float32, buffer=nl.sbuf)
                                    grad_y_dot = sbm.alloc_stack((q_actual, 1), dtype=nl.float32, buffer=nl.sbuf)
                                    nisa.memset(dst=grad_attn_acc, value=0.0, engine=nisa.engine.vector)
                                    nisa.memset(dst=grad_x_dot, value=0.0, engine=nisa.engine.vector)
                                    nisa.memset(dst=grad_y_dot, value=0.0, engine=nisa.engine.vector)

                                    for corner in range(4):
                                        gathered_corner = nl.ndarray((P_MAX, C_h), dtype=dtype, buffer=nl.psum)

                                        for c_tile_idx in range(num_C_h_tiles):
                                            c_start = c_tile_idx * P_MAX
                                            c_end = min(c_start + P_MAX, C_h)
                                            c_actual = c_end - c_start

                                            chunk_idx = c_start // P_MAX
                                            offset_in_chunk = c_start % P_MAX

                                            nisa.nc_transpose(
                                                dst=gathered_corner[:, c_start:c_end],
                                                data=gathered_reshaped[0:c_actual, chunk_idx, p, corner, :],
                                                engine=nisa.engine.tensor,
                                            )

                                        nisa.tensor_tensor(
                                            dst=prod[0:q_actual, :],
                                            data1=grad_out[:, h, :],
                                            data2=gathered_corner[0:q_actual, :],
                                            op=nl.multiply,
                                            engine=nisa.engine.vector,
                                        )
                                        nisa.tensor_reduce(
                                            dst=dc,
                                            op=nl.add,
                                            data=prod[0:q_actual, :],
                                            axis=(1,),
                                            keepdims=True,
                                        )
                                        nisa.scalar_tensor_tensor(
                                            dst=grad_attn_acc,
                                            data=dc,
                                            op0=nl.multiply,
                                            operand0=bilinear_weights[:, corner, h, l, p],
                                            op1=nl.add,
                                            operand1=grad_attn_acc,
                                        )
                                        nisa.scalar_tensor_tensor(
                                            dst=grad_x_dot,
                                            data=dc,
                                            op0=nl.multiply,
                                            operand0=grad_x_weights[:, corner, h, l, p],
                                            op1=nl.add,
                                            operand1=grad_x_dot,
                                        )
                                        nisa.scalar_tensor_tensor(
                                            dst=grad_y_dot,
                                            data=dc,
                                            op0=nl.multiply,
                                            operand0=grad_y_weights[:, corner, h, l, p],
                                            op1=nl.add,
                                            operand1=grad_y_dot,
                                        )

                                    nisa.tensor_copy(
                                        dst=grad_attn_w_local[:, h, l, p], src=grad_attn_acc, engine=nisa.engine.vector
                                    )
                                    nisa.tensor_tensor(
                                        dst=grad_sampling_loc_local[:, h, l, p, 0],
                                        data1=grad_x_dot,
                                        data2=attn_w_scaled_x[:, p : p + 1],
                                        op=nl.multiply,
                                        engine=nisa.engine.vector,
                                    )
                                    nisa.tensor_tensor(
                                        dst=grad_sampling_loc_local[:, h, l, p, 1],
                                        data1=grad_y_dot,
                                        data2=attn_w_scaled_y[:, p : p + 1],
                                        op=nl.multiply,
                                        engine=nisa.engine.vector,
                                    )

                            sbm.increment_section()

                        sbm.close_scope()
                        sbm.increment_section()

                    sbm.close_scope()

                # ====================================================================
                # Step 15: Compute grad_value and scatter to K-buffers
                # ====================================================================

                total_hl_combinations = qh * N_l
                num_hl_batches = div_ceil(total_hl_combinations, hl_tile)

                if need_scatter:
                    sbm.open_scope(interleave_degree=scatter_interleave)

                    if cfg.value_layout == "BLNC" or cfg.combined_scatter_corners == False:
                        grad_value_buffers_rows_sets = [
                            gvb.reshape((K * L * N_h, C_h)) for gvb in grad_value_buffers_sets
                        ]
                        grad_value_buffers_rows = grad_value_buffers_rows_sets[0]
                        scatter_iter = 0
                        flat_idx_all_corners = flat_idx_all.reshape((q_actual, qh, N_l, N_p, 4))

                        for hl_batch_idx in range(num_hl_batches):
                            hl_start = hl_batch_idx * hl_tile
                            hl_end = min(hl_start + hl_tile, total_hl_combinations)
                            hl_actual = hl_end - hl_start

                            num_p_groups = div_ceil(N_p, N_p_hl)

                            for p_group_idx in range(num_p_groups):
                                p_start = p_group_idx * N_p_hl
                                p_end = min(p_start + N_p_hl, N_p)
                                p_actual = p_end - p_start

                                num_cols = hl_actual * p_actual * 4

                                grad_out_scaled = sbm.alloc_stack(
                                    (q_actual, num_cols, C_h), dtype=dtype, buffer=nl.sbuf
                                )
                                flat_idx_all_scaled = sbm.alloc_stack(
                                    (q_actual, num_cols),
                                    dtype=nl.uint32,
                                    buffer=nl.sbuf,
                                    align=SCATTER_INDEX_ALIGN,
                                )

                                for hl_idx in range(hl_actual):
                                    hl_global = hl_start + hl_idx
                                    h = hl_global // N_l
                                    l = hl_global % N_l

                                    # Scale grad_out by combined_weights
                                    for p_idx in range(p_actual):
                                        p = p_start + p_idx

                                        col_base = (hl_idx * p_actual + p_idx) * 4
                                        nisa.tensor_tensor(
                                            dst=grad_out_scaled[:, col_base : col_base + 4, :],
                                            data1=grad_out[:, h, :].expand_dim(1).broadcast(1, 4),
                                            data2=combined_weights[:, :, h, l, p].expand_dim(2).broadcast(2, C_h),
                                            op=nl.multiply,
                                            engine=nisa.engine.vector,
                                        )

                                    col_start = hl_idx * p_actual * 4
                                    col_end = col_start + p_actual * 4

                                    nisa.tensor_copy(
                                        dst=flat_idx_all_scaled[:, col_start:col_end],
                                        src=flat_idx_all_corners.select(dim=1, index=h)
                                        .select(dim=1, index=l)
                                        .slice(dim=1, start=p_start, end=p_end)
                                        .reshape((q_actual, p_actual * 4)),
                                        engine=nisa.engine.vector,
                                    )

                                swizzle_degree = min(cfg.scatter_q_tile // cfg.K_base, hl_actual)

                                # Effective partition size, padded to P_MAX to avoid a DMA abort.
                                effective_q = P_MAX if q_actual < P_MAX else q_actual

                                if swizzle_degree == 1:
                                    if q_actual < P_MAX:
                                        grad_out_swizzled = sbm.alloc_stack(
                                            (P_MAX, num_cols, C_h), dtype=dtype, buffer=nl.sbuf
                                        )
                                        flat_idx_all_swizzled = sbm.alloc_stack(
                                            (P_MAX, num_cols),
                                            dtype=nl.uint32,
                                            buffer=nl.sbuf,
                                            align=SCATTER_INDEX_ALIGN,
                                        )
                                        nisa.memset(dst=grad_out_swizzled, value=0, engine=nisa.engine.vector)
                                        nisa.memset(dst=flat_idx_all_swizzled, value=0, engine=nisa.engine.vector)
                                        nisa.tensor_copy(
                                            dst=grad_out_swizzled[0:q_actual, :, :],
                                            src=grad_out_scaled,
                                            engine=nisa.engine.scalar,
                                        )
                                        nisa.tensor_copy(
                                            dst=flat_idx_all_swizzled[0:q_actual, :],
                                            src=flat_idx_all_scaled,
                                            engine=nisa.engine.vector,
                                        )
                                    else:
                                        grad_out_swizzled = grad_out_scaled
                                        flat_idx_all_swizzled = flat_idx_all_scaled
                                else:
                                    grad_out_swizzled = sbm.alloc_stack(
                                        (effective_q, num_cols, C_h), dtype=dtype, buffer=nl.sbuf
                                    )
                                    flat_idx_all_swizzled = sbm.alloc_stack(
                                        (effective_q, num_cols),
                                        dtype=nl.uint32,
                                        buffer=nl.sbuf,
                                        align=SCATTER_INDEX_ALIGN,
                                    )

                                    if q_actual < P_MAX:
                                        nisa.memset(dst=grad_out_swizzled, value=0, engine=nisa.engine.vector)
                                        nisa.memset(dst=flat_idx_all_swizzled, value=0, engine=nisa.engine.vector)

                                    if swizzle_degree == 2 and cfg.scatter_q_tile == P_MAX:
                                        partition_offsets = [0, 64]

                                        for partition_offset_idx in range(len(partition_offsets)):
                                            partition_offset = partition_offsets[partition_offset_idx]

                                            if partition_offset >= q_actual:
                                                continue

                                            partition_height = min(64, q_actual - partition_offset)

                                            offset = (num_cols // swizzle_degree) * partition_offset_idx
                                            remaining = num_cols - offset

                                            # First copy
                                            nisa.tensor_copy(
                                                dst=flat_idx_all_swizzled[
                                                    partition_offset : partition_offset + partition_height,
                                                    offset:num_cols,
                                                ],
                                                src=flat_idx_all_scaled[
                                                    partition_offset : partition_offset + partition_height, 0:remaining
                                                ],
                                                engine=nisa.engine.vector,
                                            )
                                            nisa.tensor_copy(
                                                dst=grad_out_swizzled[
                                                    partition_offset : partition_offset + partition_height,
                                                    offset:num_cols,
                                                    :,
                                                ],
                                                src=grad_out_scaled[
                                                    partition_offset : partition_offset + partition_height,
                                                    0:remaining,
                                                    :,
                                                ],
                                                engine=nisa.engine.scalar,
                                            )

                                            # Second copy, skip if first copy spans entire free dimension
                                            if offset != 0:
                                                nisa.tensor_copy(
                                                    dst=flat_idx_all_swizzled[
                                                        partition_offset : partition_offset + partition_height, 0:offset
                                                    ],
                                                    src=flat_idx_all_scaled[
                                                        partition_offset : partition_offset + partition_height,
                                                        remaining:num_cols,
                                                    ],
                                                    engine=nisa.engine.vector,
                                                )
                                                nisa.tensor_copy(
                                                    dst=grad_out_swizzled[
                                                        partition_offset : partition_offset + partition_height,
                                                        0:offset,
                                                        :,
                                                    ],
                                                    src=grad_out_scaled[
                                                        partition_offset : partition_offset + partition_height,
                                                        remaining:num_cols,
                                                        :,
                                                    ],
                                                    engine=nisa.engine.vector,
                                                )

                                    elif swizzle_degree >= 2:
                                        populated_quadrants = min(4, div_ceil(swizzle_degree * cfg.K_base, 32))
                                        num_quadrants = 1
                                        for _cand in range(populated_quadrants, 0, -1):
                                            if (
                                                swizzle_degree % _cand == 0
                                                and (swizzle_degree // _cand) * cfg.K_base <= 32
                                            ):
                                                num_quadrants = _cand
                                                break
                                        partition_offsets = [0, 32, 64, 96][:num_quadrants]

                                        quadrant_copy_degree = swizzle_degree // num_quadrants

                                        for quadrant_copy_idx in range(quadrant_copy_degree):
                                            for partition_offset_idx in range(len(partition_offsets)):
                                                partition_offset = partition_offsets[partition_offset_idx]

                                                if partition_offset >= q_actual:
                                                    continue

                                                partition_height_raw = 32 - (
                                                    quadrant_copy_idx * (32 // quadrant_copy_degree)
                                                )
                                                partition_height = min(
                                                    partition_height_raw, q_actual - partition_offset
                                                )

                                                offset = (p_actual * 4) * (
                                                    quadrant_copy_degree
                                                    - 1
                                                    - quadrant_copy_idx
                                                    + partition_offset_idx * quadrant_copy_degree
                                                )
                                                remaining = num_cols - offset

                                                # Skip this iteration if offset is out of bounds or no data to copy
                                                if offset >= num_cols or remaining <= 0:
                                                    continue

                                                # First copy
                                                nisa.tensor_copy(
                                                    dst=flat_idx_all_swizzled[
                                                        partition_offset : partition_offset + partition_height,
                                                        offset:num_cols,
                                                    ],
                                                    src=flat_idx_all_scaled[
                                                        partition_offset : partition_offset + partition_height,
                                                        0:remaining,
                                                    ],
                                                    engine=nisa.engine.scalar,
                                                )
                                                nisa.tensor_copy(
                                                    dst=grad_out_swizzled[
                                                        partition_offset : partition_offset + partition_height,
                                                        offset:num_cols,
                                                        :,
                                                    ],
                                                    src=grad_out_scaled[
                                                        partition_offset : partition_offset + partition_height,
                                                        0:remaining,
                                                        :,
                                                    ],
                                                    engine=nisa.engine.vector,
                                                )

                                                # Second copy, skip if first copy spans entire free dimension
                                                if offset != 0:
                                                    nisa.tensor_copy(
                                                        dst=flat_idx_all_swizzled[
                                                            partition_offset : partition_offset + partition_height,
                                                            0:offset,
                                                        ],
                                                        src=flat_idx_all_scaled[
                                                            partition_offset : partition_offset + partition_height,
                                                            remaining:num_cols,
                                                        ],
                                                        engine=nisa.engine.vector,
                                                    )
                                                    nisa.tensor_copy(
                                                        dst=grad_out_swizzled[
                                                            partition_offset : partition_offset + partition_height,
                                                            0:offset,
                                                            :,
                                                        ],
                                                        src=grad_out_scaled[
                                                            partition_offset : partition_offset + partition_height,
                                                            remaining:num_cols,
                                                            :,
                                                        ],
                                                        engine=nisa.engine.scalar,
                                                    )

                                # Scale indirect indicies
                                nisa.tensor_tensor(
                                    dst=flat_idx_all_swizzled[0:q_actual, :],
                                    data1=flat_idx_all_swizzled[0:q_actual, :],
                                    data2=iota_k_scale[0:q_actual, 0:num_cols],
                                    op=nl.add,
                                    engine=nisa.engine.vector,
                                )

                                # Scatter
                                if (
                                    cfg.batched_indirect_scatter
                                    and cfg.scatter_M_batch > 1
                                    and num_cols % cfg.scatter_M_batch == 0
                                ):
                                    scatter_iter += _batched_scatter_add(
                                        sbm,
                                        cfg,
                                        grad_value_buffers_rows_sets,
                                        scatter_iter,
                                        flat_idx_all_swizzled,
                                        grad_out_swizzled,
                                        effective_q,
                                        num_cols,
                                        C_h,
                                    )
                                else:
                                    for col in range(num_cols):
                                        gvb_col = grad_value_buffers_rows_sets[
                                            scatter_iter % len(grad_value_buffers_rows_sets)
                                        ]
                                        scatter_iter += 1
                                        col_ap = gvb_col.ap(
                                            pattern=[[1, effective_q], [1, C_h]],
                                            offset=0,
                                            vector_offset=flat_idx_all_swizzled[0:effective_q, col : col + 1],
                                            indirect_dim=0,
                                        )
                                        nisa.dma_compute(
                                            dst=col_ap,
                                            srcs=[
                                                col_ap,
                                                grad_out_swizzled[0:effective_q, col, :],
                                            ],
                                            reduce_op=nl.add,
                                            unique_indices=True,
                                        )

                                sbm.increment_section()
                    else:
                        grad_value_buffers_flat = grad_value_buffers.reshape((K * N_h * L * C_h,))
                        flat_idx_shifted_reshaped = flat_idx_shifted.reshape((q_actual, qh, N_l, N_p, 2))

                        for hl_batch_idx in range(num_hl_batches):
                            hl_start = hl_batch_idx * hl_tile
                            hl_end = min(hl_start + hl_tile, total_hl_combinations)
                            hl_actual = hl_end - hl_start

                            num_p_groups = div_ceil(N_p, N_p_hl)

                            for p_group_idx in range(num_p_groups):
                                p_start = p_group_idx * N_p_hl
                                p_end = min(p_start + N_p_hl, N_p)
                                p_actual = p_end - p_start

                                # 2 columns per point
                                num_cols = hl_actual * p_actual * 2

                                # Each column holds 2*C_h
                                grad_out_scaled = sbm.alloc_stack(
                                    (q_actual, num_cols, C_h * 2), dtype=dtype, buffer=nl.sbuf
                                )
                                flat_idx_scaled = sbm.alloc_stack((q_actual, num_cols), dtype=nl.uint32, buffer=nl.sbuf)

                                # Prepare gradient data with adjacent corner contributions
                                for hl_idx in range(hl_actual):
                                    hl_global = hl_start + hl_idx
                                    h = hl_global // N_l
                                    l = hl_global % N_l

                                    # Scale grad_out by combined_weights
                                    for p_idx in range(p_actual):
                                        p = p_start + p_idx
                                        """
                                        All four corner scalings in one instruction: grad_out
                                        broadcasts across the corner axis and the weight across
                                        channels, two free dimensions either side. corner_a and
                                        corner_b below are strided slices of the result.
                                        """
                                        corner_scaled = sbm.alloc_stack((q_actual, 4, C_h), dtype=dtype, buffer=nl.sbuf)
                                        nisa.tensor_tensor(
                                            dst=corner_scaled,
                                            data1=grad_out[:, h, :].expand_dim(1).broadcast(1, 4),
                                            data2=combined_weights[:, :, h, l, p].expand_dim(2).broadcast(2, C_h),
                                            op=nl.multiply,
                                            engine=nisa.engine.vector,
                                        )

                                        col_pair = (hl_idx * p_actual + p_idx) * 2
                                        corner_a = corner_scaled.slice(dim=1, start=0, end=3, step=2)
                                        corner_b = corner_scaled.slice(dim=1, start=1, end=4, step=2)

                                        for half in range(2):
                                            lo = half * C_h
                                            if half == 0:
                                                w_a = one_minus_coord_oob_f[:, 1, h, l, p : p + 1]
                                                w_b = coord_oob_f[:, 0, h, l, p : p + 1]
                                            else:
                                                w_a = coord_oob_f[:, 1, h, l, p : p + 1]
                                                w_b = one_minus_coord_oob_f[:, 0, h, l, p : p + 1]
                                            dst_pair = grad_out_scaled[:, col_pair : col_pair + 2, lo : lo + C_h]
                                            nisa.tensor_scalar(
                                                dst=dst_pair,
                                                data=corner_a,
                                                op0=nl.multiply,
                                                operand0=w_a,
                                                engine=nisa.engine.scalar,
                                            )
                                            nisa.scalar_tensor_tensor(
                                                dst=dst_pair,
                                                data=corner_b,
                                                op0=nl.multiply,
                                                operand0=w_b,
                                                op1=nl.add,
                                                operand1=dst_pair,
                                            )

                                    # Copy shifted indices for this (h, l) pair
                                    col_start = hl_idx * p_actual * 2
                                    col_end = col_start + p_actual * 2

                                    nisa.tensor_copy(
                                        dst=flat_idx_scaled[:, col_start:col_end],
                                        src=flat_idx_shifted_reshaped.select(dim=1, index=h)
                                        .select(dim=1, index=l)
                                        .slice(dim=1, start=p_start, end=p_end)
                                        .reshape((q_actual, p_actual * 2)),
                                        engine=nisa.engine.vector,
                                    )

                                swizzle_degree = min(cfg.scatter_q_tile // cfg.K_base, hl_actual)

                                if swizzle_degree == 1:
                                    grad_out_swizzled = grad_out_scaled
                                    flat_idx_swizzled = flat_idx_scaled
                                else:
                                    grad_out_swizzled = sbm.alloc_stack(
                                        (q_actual, num_cols, C_h * 2), dtype=dtype, buffer=nl.sbuf
                                    )
                                    flat_idx_swizzled = sbm.alloc_stack(
                                        (q_actual, num_cols), dtype=nl.uint32, buffer=nl.sbuf
                                    )

                                    if swizzle_degree == 2 and cfg.scatter_q_tile == P_MAX:
                                        partition_offsets = [0, 64]

                                        for partition_offset_idx in range(len(partition_offsets)):
                                            partition_offset = partition_offsets[partition_offset_idx]

                                            if partition_offset >= q_actual:
                                                continue

                                            partition_height = min(64, q_actual - partition_offset)

                                            offset = (num_cols // swizzle_degree) * partition_offset_idx
                                            remaining = num_cols - offset

                                            # First copy
                                            nisa.tensor_copy(
                                                dst=flat_idx_swizzled[
                                                    partition_offset : partition_offset + partition_height,
                                                    offset:num_cols,
                                                ],
                                                src=flat_idx_scaled[
                                                    partition_offset : partition_offset + partition_height, 0:remaining
                                                ],
                                                engine=nisa.engine.vector,
                                            )
                                            nisa.tensor_copy(
                                                dst=grad_out_swizzled[
                                                    partition_offset : partition_offset + partition_height,
                                                    offset:num_cols,
                                                    :,
                                                ],
                                                src=grad_out_scaled[
                                                    partition_offset : partition_offset + partition_height,
                                                    0:remaining,
                                                    :,
                                                ],
                                                engine=nisa.engine.scalar,
                                            )

                                            # Second copy, skip if first copy spans entire free dimension
                                            if offset != 0:
                                                nisa.tensor_copy(
                                                    dst=flat_idx_swizzled[
                                                        partition_offset : partition_offset + partition_height, 0:offset
                                                    ],
                                                    src=flat_idx_scaled[
                                                        partition_offset : partition_offset + partition_height,
                                                        remaining:num_cols,
                                                    ],
                                                    engine=nisa.engine.vector,
                                                )
                                                nisa.tensor_copy(
                                                    dst=grad_out_swizzled[
                                                        partition_offset : partition_offset + partition_height,
                                                        0:offset,
                                                        :,
                                                    ],
                                                    src=grad_out_scaled[
                                                        partition_offset : partition_offset + partition_height,
                                                        remaining:num_cols,
                                                        :,
                                                    ],
                                                    engine=nisa.engine.vector,
                                                )

                                    elif swizzle_degree >= 2:
                                        populated_quadrants = min(4, div_ceil(swizzle_degree * cfg.K_base, 32))
                                        num_quadrants = 1
                                        for _cand in range(populated_quadrants, 0, -1):
                                            if (
                                                swizzle_degree % _cand == 0
                                                and (swizzle_degree // _cand) * cfg.K_base <= 32
                                            ):
                                                num_quadrants = _cand
                                                break
                                        partition_offsets = [0, 32, 64, 96][:num_quadrants]

                                        quadrant_copy_degree = swizzle_degree // num_quadrants

                                        for quadrant_copy_idx in range(quadrant_copy_degree):
                                            for partition_offset_idx in range(len(partition_offsets)):
                                                partition_offset = partition_offsets[partition_offset_idx]

                                                if partition_offset >= q_actual:
                                                    continue

                                                partition_height_raw = 32 - (
                                                    quadrant_copy_idx * (32 // quadrant_copy_degree)
                                                )
                                                partition_height = min(
                                                    partition_height_raw, q_actual - partition_offset
                                                )

                                                offset = (p_actual * 2) * (
                                                    quadrant_copy_degree
                                                    - 1
                                                    - quadrant_copy_idx
                                                    + partition_offset_idx * quadrant_copy_degree
                                                )
                                                remaining = num_cols - offset

                                                # Skip if offset is out of bounds or remaining is non-positive
                                                if offset >= num_cols or remaining <= 0:
                                                    continue

                                                # First copy
                                                nisa.tensor_copy(
                                                    dst=flat_idx_swizzled[
                                                        partition_offset : partition_offset + partition_height,
                                                        offset:num_cols,
                                                    ],
                                                    src=flat_idx_scaled[
                                                        partition_offset : partition_offset + partition_height,
                                                        0:remaining,
                                                    ],
                                                    engine=nisa.engine.scalar,
                                                )
                                                nisa.tensor_copy(
                                                    dst=grad_out_swizzled[
                                                        partition_offset : partition_offset + partition_height,
                                                        offset:num_cols,
                                                        :,
                                                    ],
                                                    src=grad_out_scaled[
                                                        partition_offset : partition_offset + partition_height,
                                                        0:remaining,
                                                        :,
                                                    ],
                                                    engine=nisa.engine.vector,
                                                )

                                                # Second copy, skip if first copy spans entire free dimension
                                                if offset != 0 and remaining < num_cols:
                                                    nisa.tensor_copy(
                                                        dst=flat_idx_swizzled[
                                                            partition_offset : partition_offset + partition_height,
                                                            0:offset,
                                                        ],
                                                        src=flat_idx_scaled[
                                                            partition_offset : partition_offset + partition_height,
                                                            remaining:num_cols,
                                                        ],
                                                        engine=nisa.engine.vector,
                                                    )
                                                    nisa.tensor_copy(
                                                        dst=grad_out_swizzled[
                                                            partition_offset : partition_offset + partition_height,
                                                            0:offset,
                                                            :,
                                                        ],
                                                        src=grad_out_scaled[
                                                            partition_offset : partition_offset + partition_height,
                                                            remaining:num_cols,
                                                            :,
                                                        ],
                                                        engine=nisa.engine.scalar,
                                                    )

                                # Scale indirect indicies
                                nisa.tensor_tensor(
                                    dst=flat_idx_swizzled[0:q_actual, :],
                                    data1=flat_idx_swizzled[0:q_actual, :],
                                    data2=iota_k_scale[0:q_actual, 0:num_cols],
                                    op=nl.add,
                                    engine=nisa.engine.vector,
                                )

                                # Scatter
                                if q_actual < P_MAX:
                                    effective_q = P_MAX
                                    flat_idx_padded = sbm.alloc_stack(
                                        (P_MAX, num_cols), dtype=nl.uint32, buffer=nl.sbuf
                                    )
                                    grad_out_padded = sbm.alloc_stack(
                                        (P_MAX, num_cols, C_h * 2), dtype=dtype, buffer=nl.sbuf
                                    )
                                    nisa.memset(dst=flat_idx_padded, value=0, engine=nisa.engine.vector)
                                    nisa.memset(dst=grad_out_padded, value=0, engine=nisa.engine.vector)
                                    nisa.tensor_copy(
                                        dst=flat_idx_padded[0:q_actual, :],
                                        src=flat_idx_swizzled[0:q_actual, :],
                                        engine=nisa.engine.vector,
                                    )
                                    nisa.tensor_copy(
                                        dst=grad_out_padded[0:q_actual, :, :],
                                        src=grad_out_swizzled[0:q_actual, :, :],
                                        engine=nisa.engine.vector,
                                    )

                                    flat_idx_to_use = flat_idx_padded
                                    grad_out_to_use = grad_out_padded
                                else:
                                    effective_q = q_actual
                                    flat_idx_to_use = flat_idx_swizzled
                                    grad_out_to_use = grad_out_swizzled

                                if (
                                    cfg.batched_indirect_scatter
                                    and cfg.scatter_M_batch > 1
                                    and num_cols % cfg.scatter_M_batch == 0
                                ):
                                    _batched_scatter_add(
                                        sbm,
                                        cfg,
                                        grad_value_buffers_flat,
                                        flat_idx_to_use,
                                        grad_out_to_use,
                                        effective_q,
                                        num_cols,
                                        C_h * 2,
                                        row_stride=C_h * 2,
                                    )
                                else:
                                    for col in range(num_cols):
                                        nisa.dma_compute(
                                            dst=grad_value_buffers_flat.ap(
                                                pattern=[[C_h * 2, effective_q], [1, C_h * 2]],
                                                offset=0,
                                                vector_offset=flat_idx_to_use[0:effective_q, col : col + 1],
                                                indirect_dim=0,
                                            ),
                                            srcs=[
                                                grad_value_buffers_flat.ap(
                                                    pattern=[[C_h * 2, effective_q], [1, C_h * 2]],
                                                    offset=0,
                                                    vector_offset=flat_idx_to_use[0:effective_q, col : col + 1],
                                                    indirect_dim=0,
                                                ),
                                                grad_out_to_use[0:effective_q, col, :],
                                            ],
                                            reduce_op=nl.add,
                                            unique_indices=True,
                                        )

                                sbm.increment_section()

                    sbm.close_scope()

                # ====================================================================
                # Step 16: Write gradients back to HBM
                # ====================================================================

                # Write grad_attention_weights
                if compute_grad_attention_weights:
                    nisa.dma_copy(
                        dst=grad_attention_weights[batch_idx, q_start:q_end, h_start:h_end, :, :],
                        src=grad_attn_w_local,
                        dge_mode=nisa.dge_mode.none,
                    )

                # Write grad_sampling_locations
                if not compute_grad_sampling_locations:
                    pass
                elif cfg.sampling_locations_layout == "BQHLP2":
                    grad_sampling_loc_local_2d = grad_sampling_loc_local.reshape((q_actual, qh, N_l, N_p * 2))
                    nisa.dma_copy(
                        dst=grad_sampling_locations.select(dim=0, index=batch_idx)
                        .slice(dim=0, start=q_start, end=q_end, step=1)
                        .slice(dim=1, start=h_start, end=h_end, step=1)
                        .reshape((q_actual, qh, N_l, N_p * 2)),
                        src=grad_sampling_loc_local_2d,
                        dge_mode=nisa.dge_mode.none,
                    )
                else:  # B2QHLP
                    nisa.dma_copy(
                        dst=grad_sampling_locations[batch_idx, 0, q_start:q_end, h_start:h_end, :, :],
                        src=grad_sampling_loc_local[:, :, :, :, 0],
                        dge_mode=nisa.dge_mode.none,
                    )
                    nisa.dma_copy(
                        dst=grad_sampling_locations[batch_idx, 1, q_start:q_end, h_start:h_end, :, :],
                        src=grad_sampling_loc_local[:, :, :, :, 1],
                        dge_mode=nisa.dge_mode.none,
                    )

                sbm.increment_section()

        if need_scatter:
            # ====================================================================
            # Step 17: Reduce K buffers for this batch
            # ====================================================================
            if lnc > 1:
                barrier_cores = []
                for core_idx in range(lnc):
                    barrier_cores.append(core_idx)
                for gvb_all in grad_value_buffers_all_sets:
                    nisa.core_barrier(gvb_all, cores=barrier_cores)

            grad_value_buffers_rows_r_sets = [gvb.reshape((K, n_value_rows, C_h)) for gvb in grad_value_buffers_sets]
            if K > 1:
                for group in range(round0_per_core):
                    start_idx = group * 16
                    end_idx = min(start_idx + 16, K)
                    dst_slot = temp_intermediate_rows[shard_id * round0_per_core + group]
                    first = grad_value_buffers_rows_r_sets[0]
                    nisa.dma_compute(
                        dst=dst_slot,
                        srcs=[first[k] for k in range(start_idx, end_idx)],
                        reduce_op=nl.add,
                    )
                    for si in range(1, len(grad_value_buffers_rows_r_sets)):
                        gvb_r = grad_value_buffers_rows_r_sets[si]
                        k = start_idx
                        while k < end_idx:
                            chunk_end = min(k + 15, end_idx)
                            nisa.dma_compute(
                                dst=dst_slot,
                                srcs=[dst_slot] + [gvb_r[kk] for kk in range(k, chunk_end)],
                                reduce_op=nl.add,
                            )
                            k = chunk_end
            else:
                dst_slot_k1 = temp_intermediate_rows[shard_id * round0_per_core]
                if len(grad_value_buffers_rows_r_sets) == 1:
                    nisa.dma_copy(
                        dst=dst_slot_k1,
                        src=grad_value_buffers_rows_r_sets[0][0],
                        dge_mode=nisa.dge_mode.none,
                    )
                else:
                    # Chunked so the source count stays within one dma_compute at any set count
                    srcs_all = [gvb_r[0] for gvb_r in grad_value_buffers_rows_r_sets]
                    nisa.dma_compute(dst=dst_slot_k1, srcs=srcs_all[0:16], reduce_op=nl.add)
                    done = 16
                    while done < len(srcs_all):
                        chunk_end = min(done + 15, len(srcs_all))
                        nisa.dma_compute(
                            dst=dst_slot_k1,
                            srcs=[dst_slot_k1] + srcs_all[done:chunk_end],
                            reduce_op=nl.add,
                        )
                        done = chunk_end

            # All cores must finish writing their round-0 region before any core reads all regions
            if lnc > 1:
                cores = []
                for core_idx in range(lnc):
                    cores.append(core_idx)
                nisa.core_barrier(temp_intermediate, cores=cores)

            # Spatially shard the value rows across cores
            row_lo = shard_id * rows_per_core
            row_hi = min(row_lo + rows_per_core, n_value_rows)
            grad_value_rows = grad_value.reshape((B, n_value_rows, C_h))

            if row_hi > row_lo:
                if reuse_buffers or dump_reduced_grad_value or subtract_prev:
                    survivor_srcs = [
                        temp_intermediate_rows[survivor_idx, row_lo:row_hi, :]
                        for survivor_idx in range(round0_survivors)
                    ]

                    # Optionally materialize R_b for the next call
                    if dump_reduced_grad_value:
                        _grouped_reduce(
                            reduced_grad_value_rows[row_lo:row_hi, :],
                            survivor_srcs,
                            [1.0] * len(survivor_srcs),
                            temp_intermediate_rows,
                            final_combine_offset,
                            row_slice=(row_lo, row_hi),
                        )

                    if subtract_prev:
                        if dump_reduced_grad_value:
                            # R_b already materialized; grad_value = R_b - prev
                            nisa.dma_compute(
                                dst=grad_value_rows[batch_idx, row_lo:row_hi, :],
                                srcs=[
                                    reduced_grad_value_rows[row_lo:row_hi, :],
                                    prev_reduced_rows[row_lo:row_hi, :],
                                ],
                                scales=[1.0, -1.0],
                                reduce_op=nl.add,
                            )
                        else:
                            # Fold the -prev term straight into the survivor reduction
                            _grouped_reduce(
                                grad_value_rows[batch_idx, row_lo:row_hi, :],
                                survivor_srcs + [prev_reduced_rows[row_lo:row_hi, :]],
                                [1.0] * len(survivor_srcs) + [-1.0],
                                temp_intermediate_rows,
                                final_combine_offset,
                                row_slice=(row_lo, row_hi),
                            )
                    else:
                        # First call of a reuse sequence: grad_value = R_b
                        _grouped_reduce(
                            grad_value_rows[batch_idx, row_lo:row_hi, :],
                            survivor_srcs,
                            [1.0] * len(survivor_srcs),
                            temp_intermediate_rows,
                            final_combine_offset,
                            row_slice=(row_lo, row_hi),
                        )
                else:
                    combine_srcs = []
                    combine_scales = []
                    for survivor_idx in range(round0_survivors):
                        combine_srcs.append(temp_intermediate_rows[survivor_idx, row_lo:row_hi, :])
                        combine_scales.append(1.0)
                    if cfg.accumulate_scatter_across_batches and batch_idx > 0:
                        for prev_batch in range(batch_idx):
                            combine_srcs.append(grad_value_rows[prev_batch, row_lo:row_hi, :])
                            combine_scales.append(-1.0)

                    _grouped_reduce(
                        grad_value_rows[batch_idx, row_lo:row_hi, :],
                        combine_srcs,
                        combine_scales,
                        temp_intermediate_rows,
                        final_combine_offset,
                        row_slice=(row_lo, row_hi),
                    )

    sbm.close_scope()

    # Free iota_k_scale
    sbm.pop_heap()  # iota_k_scale

    # Base outputs, then any requested dumps in fixed order:
    outputs = [t for t in (grad_value, grad_sampling_locations, grad_attention_weights) if t != None]
    if dump_iota_k_scale:
        outputs.append(iota_k_scale_out)
    if dump_grad_value_buffers:
        outputs.append(grad_value_buffers_all_sets[0])
    if dump_reduced_grad_value:
        outputs.append(reduced_grad_value_out)
    return tuple(outputs)


def _grouped_reduce(dst, srcs, scales, scratch, scratch_offset, row_slice=None):
    """Reduce a signed source list into dst via a 16-at-a-time reduction tree."""
    cur_srcs = srcs
    cur_scales = scales
    write_offset = scratch_offset
    while len(cur_srcs) > 16:
        next_srcs = []
        next_scales = []
        for lo in range(0, len(cur_srcs), 16):
            hi = min(lo + 16, len(cur_srcs))
            if row_slice == None:
                dst_scratch = scratch[write_offset]
            else:
                dst_scratch = scratch[write_offset, row_slice[0] : row_slice[1], :]
            nisa.dma_compute(
                dst=dst_scratch,
                srcs=cur_srcs[lo:hi],
                scales=cur_scales[lo:hi],
                reduce_op=nl.add,
            )
            next_srcs.append(dst_scratch)
            next_scales.append(1.0)
            write_offset += 1
        cur_srcs = next_srcs
        cur_scales = next_scales
    nisa.dma_compute(dst=dst, srcs=cur_srcs, scales=cur_scales, reduce_op=nl.add)


def _batched_scatter_add(
    sbm,
    cfg,
    buffer_sets,
    set_base,
    flat_idx_all_swizzled,
    grad_out_swizzled,
    effective_q,
    num_cols,
    elem_size,
    row_stride=None,
):
    """Collapse the per-column scatter loop into batched >P_MAX indirection dma_computes, one per run of cfg.scatter_M_batch columns."""
    M = cfg.scatter_M_batch
    P = effective_q
    if row_stride == None:
        row_stride = 1

    idx_gather = flat_idx_all_swizzled
    n_sets = len(buffer_sets)

    num_runs = num_cols // M
    for run in range(num_runs):
        c0 = run * M
        NTOT = P * M
        idx_run = idx_gather[0:P, c0 : c0 + M]
        data_run = grad_out_swizzled[0:P, c0 : c0 + M, :]
        dst_ap = buffer_sets[(set_base + run) % n_sets].ap(
            pattern=[[row_stride, NTOT], [1, elem_size]],
            offset=0,
            vector_offset=idx_run,
            indirect_dim=0,
        )
        nisa.dma_compute(
            dst=dst_ap,
            srcs=[dst_ap, data_run],
            reduce_op=nl.add,
            unique_indices=True,
        )
    return num_runs


def _zero_hbm_from_tile(zero_tile, dst_flat, total_elements, P_MAX, zero_tile_free_dim):
    """Write total_elements zeros into a flat HBM view, a zeroed SBUF tile row at a time."""
    tile_elements = P_MAX * zero_tile_free_dim
    for offset in range(0, total_elements, tile_elements):
        chunk_size = min(tile_elements, total_elements - offset)
        num_full_rows = chunk_size // zero_tile_free_dim
        for row_idx in range(num_full_rows):
            dst_offset = offset + row_idx * zero_tile_free_dim
            nisa.dma_copy(
                dst=dst_flat[dst_offset : dst_offset + zero_tile_free_dim],
                src=zero_tile[row_idx, :],
                dge_mode=nisa.dge_mode.none,
            )
        remainder = chunk_size % zero_tile_free_dim
        if remainder > 0:
            dst_offset = offset + num_full_rows * zero_tile_free_dim
            nisa.dma_copy(
                dst=dst_flat[dst_offset : dst_offset + remainder],
                src=zero_tile[num_full_rows : num_full_rows + 1, 0:remainder][0, :],
                dge_mode=nisa.dge_mode.none,
            )


def _calculate_h_scope_memory(
    h_tile: int,
    N_l: int,
    N_p: int,
    C_h: int,
    dtype_size: int,
    padding_mode: str,
    sampling_locations_layout: str,
    value_layout: str,
    combined_scatter_corners: bool,
) -> int:
    """H-scope footprint in bytes PER PARTITION."""
    base = h_tile * N_l * N_p
    mem = 0
    mem += h_tile * C_h * dtype_size  # grad_out
    mem += base * 4 * 2  # x, y (float32)
    if sampling_locations_layout == "BQHLP2":
        mem += base * 2 * 4  # xy staging buffer
    mem += base * dtype_size  # attn_w
    mem += base * dtype_size * 2  # x0, y0 unclamped
    mem += base * 4 * 2  # x1, y1 unclamped (int32)
    mem += base * 4 * 4  # x0/x1/y0/y1 clamped (uint32)
    mem += 2 * base * dtype_size * 3  # dx_dy, one_minus_dx_dy, dx_dy_minus_one
    mem += 4 * base * 4  # bilinear_weights (float32)
    mem += base * 4 * 4  # flat_idx_all (uint32)
    mem += 2 * h_tile * N_p * 4  # x_term (uint32)
    if padding_mode == "zeros":
        # Step 11's masking temporaries live in their own nested scope and are freed before the
        # gather and scatter scopes open, so they do not add to the persistent H figure -- they
        # only have to fit alongside it, which _oob_scope_memory below accounts for.
        if value_layout == "BNLC" and combined_scatter_corners:
            mem += 4 * base * 4 * 2  # coord_oob_f, one_minus_coord_oob_f
            mem += base * 2 * 4  # flat_idx_shifted
            mem += base * 4  # x1_oob_u
    mem += 4 * base * 4  # combined_weights (float32)
    mem += 4 * base * dtype_size * 2  # grad_x_weights, grad_y_weights
    mem += base * dtype_size  # grad_attn_w_local
    mem += base * 2 * 4  # grad_sampling_loc_local (float32)
    return mem


def _oob_scope_memory(h_tile: int, N_l: int, N_p: int) -> int:
    """Step 11's OOB-masking temporaries, PER PARTITION, for one nested-scope instance.

    Freed before the gather/scatter scopes open, so this competes with them for the same space
    rather than adding to the persistent H-scope figure.
    """
    base = h_tile * N_l * N_p
    mem = 4 * base * 4  # coord_oob (int32)
    mem += base * 4 * 4  # x0/y0/x1/y1 as int32
    mem += 4 * base * 4  # one_minus_coord_oob
    mem += 4 * base * 4  # zeros_mask
    return mem


def _calculate_gather_scope_memory(
    N_p: int,
    C_h: int,
    dtype_size: int,
    P_MAX: int,
    gather_method: str,
    batched_indirect_gather: bool,
    gather_M_batch: int,
    num_C_h_tiles: int,
    gather_hl_group: int = 1,
) -> int:
    """Gather-scope footprint PER PARTITION for ONE section (one (head, level) group)."""
    total_corners = N_p * 4
    # Shared by a whole group, so it rotates once per group.
    group_mem = 0
    # Private to one block; the nested scope rotates these per block.
    block_mem = 0
    block_mem += N_p * dtype_size * 2  # attn_w_scaled_x, attn_w_scaled_y
    if gather_method == "transpose" and not batched_indirect_gather:
        block_mem += div_ceil(C_h, P_MAX) * total_corners * P_MAX * dtype_size  # gathered_all
        block_mem += total_corners * 4  # flat_idx_padded
    else:
        group_mem += gather_hl_group * total_corners * C_h * dtype_size  # gathered_group_flat
        group_mem += gather_hl_group * total_corners * 4  # idx_group
        if gather_method != "copy":
            # gathered_cq is transposed, so its partition axis is C_h, not the query
            group_mem += num_C_h_tiles * gather_M_batch * P_MAX * dtype_size
    block_mem += N_p * 4 * C_h * 4  # prod_all (float32)
    block_mem += N_p * 4 * 4  # dc_all (float32)
    block_mem += N_p * 4 * 2  # grad_x_dot, grad_y_dot (float32)
    block_mem += N_p * 4 * 4  # wtmp (float32)
    return group_mem + gather_hl_group * block_mem


def _calculate_scatter_scope_memory(
    num_cols: int,
    C_h: int,
    dtype_size: int,
    swizzle_degree: int,
    needs_padded_tile: bool,
) -> int:
    """Scatter-scope footprint PER PARTITION for ONE section (one point group)."""
    mem = num_cols * C_h * dtype_size + num_cols * 4  # grad_out_scaled, flat_idx_all_scaled
    if swizzle_degree >= 2 or needs_padded_tile:
        mem += num_cols * C_h * dtype_size + num_cols * 4  # the swizzled pair
    return mem


@dataclass(frozen=True)
class MSDeformAttnBwdConfig(nl.NKIObject):
    """Configuration for multi-scale deformable attention backward pass."""

    # Input dimensions
    B: int  # Batch size
    N_q: int  # Global number of queries
    N_h: int  # Number of heads
    C_h: int  # Channels per head
    N_l: int  # Number of levels
    N_p: int  # Number of sampling points per query per head per level
    L: int  # Total flattened spatial dimension (sum of H_i * W_i)
    spatial_shapes: tuple  # ((H_0, W_0), (H_1, W_1), ...)
    level_start_index: tuple  # (0, H_0*W_0, H_0*W_0 + H_1*W_1, ...)
    dtype: type  # dtype
    value_layout: str  # "BLNC" or "BNLC"
    sampling_locations_layout: str  # "BQHLP2" (B,N_q,N_h,N_l,N_p,2) or "B2QHLP" (B,2,N_q,N_h,N_l,N_p)

    # Hardware constants
    P_MAX: int  # Partition dimension size

    # Tiling configuration
    Q_tile: int  # Queries per tile
    H_tile: int  # Number of heads per tile
    num_q_tiles: int  # Number of query tiles needed
    num_h_tiles: int  # Number of head tiles needed
    num_C_h_tiles: int  # Number of C_h tiles needed

    # Interleave degrees
    h_interleave: int  # Interleave degree for h-scope
    gather_interleave: int  # Interleave degree for gather-scope
    scatter_interleave: int  # Interleave degree for scatter-scope

    # Memory
    total_sbuf: int  # Total SBUF memory available

    # Sharding info
    lnc: int  # Logical NeuronCore count
    shard_id: int  # Current shard ID
    q_start_global: int  # Global start index for queries on this core
    local_N_q: int  # Number of queries on this core

    # Calculated dimensions
    K: int  # Number of K-buffers for scatter
    N_p_hl: int  # Number of points per HL group
    Q_pack: int  # Queries packed per partition (free-dimension query packing)
    hl_tile: int  # HL tile size (P_MAX // K)
    zero_tile_free_dim: int  # Free dimension for zero buffer initialization
    K_rep: int  # K replication factor
    combined_scatter_corners: bool  # Process adjacent corners together during scatter
    accumulate_scatter_across_batches: (
        bool  # Skip per-batch memset; accumulate cumulative sum, recover via cum[N]-cum[N-1]
    )

    # Batched indirect scatter
    batched_indirect_scatter: bool
    max_scatter_indices_per_indirect: int
    scatter_M_batch: int  # [P_MAX] scatter columns batched per dma_compute (>=1)
    scatter_q_tile: int  # query partitions processed per scatter dma_compute (<=P_MAX)
    K_base: int
    num_scatter_buffers: int  # K-replicated scatter buffer sets held in HBM

    # Batched indirect gather
    batched_indirect_gather: bool
    max_gather_indices_per_indirect: int
    gather_M_batch: int  # P_MAX gather columns batched per run (>=1)
    gather_hl_group: int  # (head, level) blocks sharing one gather tile / indirect DMA (>=1)
    num_gather_bufs: int  # SBUF gather tiles for double-buffering (>=1)
    gather_method: str  # "transpose" or "copy"


def _build_config(
    value: nl.NkiTensor,
    spatial_shapes: tuple,
    level_start_index: tuple,
    sampling_locations: nl.NkiTensor,
    attention_weights: nl.NkiTensor,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    padding_mode: str = "zeros",
    max_gather_indices_per_indirect: Optional[int] = None,
    max_scatter_indices_per_indirect: Optional[int] = None,
    gather_method: str = "transpose",
    num_scatter_buffers: Optional[int] = None,
    compute_grad_value: bool = True,
    compute_grad_sampling_locations: bool = True,
    compute_grad_attention_weights: bool = True,
) -> MSDeformAttnBwdConfig:
    """Build config with sharding and tiling information.

    Args:
        value (nl.NkiTensor): Value tensor in either (B, L, N_h, C_h) or (B, N_h, L, C_h) layout
        spatial_shapes (tuple): Spatial dimensions for each level
        level_start_index (tuple): Start indices for each level
        sampling_locations (nl.NkiTensor): Sampling coordinates in (B,N_q,N_h,N_l,N_p,2) or (B,2,N_q,N_h,N_l,N_p)
        attention_weights (nl.NkiTensor): Attention weights
        value_layout (str): "BLNC" for (B, L, N_h, C_h) or "BNLC" for (B, N_h, L, C_h)
        sampling_locations_layout (str): "BQHLP2" or "B2QHLP"
        padding_mode (str): Padding mode ("zeros" or "border")
        max_gather_indices_per_indirect (Optional[int]): Cap on gather indices per batched gather
        max_scatter_indices_per_indirect (Optional[int]): Cap on scatter indices per batched dma_compute
        gather_method (str): "transpose" or "copy"

    Returns:
        MSDeformAttnBwdConfig: Configuration object with sharding and tiling parameters
    """
    # Parse value shape based on layout
    if value_layout == "BLNC":
        B, L, N_h, C_h = value.shape
    else:  # BNLC
        B, N_h, L, C_h = value.shape

    # Parse sampling_locations shape based on layout
    if sampling_locations_layout == "BQHLP2":
        _, N_q, _, N_l, N_p, _ = sampling_locations.shape
    else:  # B2QHLP
        _, _, N_q, _, N_l, N_p = sampling_locations.shape

    # Get sharding info
    _, lnc, shard_id = get_verified_program_sharding_info("ms_deformable_attention_bwd", (0, 1))

    # Shard queries across
    queries_per_nc = div_ceil(N_q, lnc)
    q_start_global = shard_id * queries_per_nc
    q_end_global = min(q_start_global + queries_per_nc, N_q)
    local_N_q = q_end_global - q_start_global

    # Hardware constants
    P_MAX = nl.tile_size.pmax

    # Default param values
    default_h_interleave = 2
    default_gather_interleave = 2
    default_scatter_interleave = 2
    default_max_indices_per_indirect = 128
    default_kbase_mult = 1

    # Default all interleave degrees to 2
    h_interleave = default_h_interleave
    gather_interleave = default_gather_interleave
    scatter_interleave = default_scatter_interleave

    # Calculate available memory
    RESERVED_SBUF = 1024
    total_sbuf = nl.tile_size.total_available_sbuf_size - RESERVED_SBUF

    # Hardcode H_tile to use all heads
    H_tile = N_h
    num_h_tiles = 1

    # Calculate num_C_h_tiles
    num_C_h_tiles = div_ceil(C_h, P_MAX)

    resolved_num_scatter_buffers = 1 if num_scatter_buffers == None else num_scatter_buffers
    kernel_assert(
        resolved_num_scatter_buffers >= 1 and resolved_num_scatter_buffers <= MAX_SCATTER_BUFFERS,
        f"num_scatter_buffers must be in [1, {MAX_SCATTER_BUFFERS}], got {resolved_num_scatter_buffers=}",
    )
    kernel_assert(
        resolved_num_scatter_buffers <= 2 or C_h <= 256,
        f"num_scatter_buffers > 2 is not correct for C_h > 256 (got {resolved_num_scatter_buffers=}, "
        f"{C_h=}): the Step 17 reduce races the still-draining indirect scatters. Use 1 or 2.",
    )

    N_p_hl = 1  # Number of points per HL group
    zero_tile_free_dim = 16384  # Free dimension for zero buffer initialization
    K_rep = min(P_MAX, P_MAX // (N_l * N_h))  # K replication factor
    combined_scatter_corners = True  # Process adjacent corners together during scatter
    accumulate_scatter_across_batches = B > 1

    batched_indirect_scatter = max_scatter_indices_per_indirect != None
    if max_scatter_indices_per_indirect == None:
        max_scatter_indices_per_indirect = default_max_indices_per_indirect

    dtype_size = sizeinbytes(value.dtype)

    if batched_indirect_scatter and max_scatter_indices_per_indirect <= P_MAX:
        _cap = max(1, max_scatter_indices_per_indirect)
        scatter_q_tile = 1
        while scatter_q_tile * 2 <= min(_cap, P_MAX):
            scatter_q_tile *= 2
        scatter_M_batch = 1
    else:
        scatter_q_tile = P_MAX
        scatter_M_batch = max(1, max_scatter_indices_per_indirect // P_MAX) if batched_indirect_scatter else 1

    K = max(1, div_ceil(scatter_q_tile, N_l * N_h))
    hl_tile = scatter_q_tile // K

    max_cols_per_run = hl_tile * N_p_hl * 4
    scatter_M_batch = min(scatter_M_batch, max_cols_per_run)
    K_base = min(K * default_kbase_mult, scatter_q_tile)
    K = K_base * scatter_M_batch

    _k_row_stride = L * N_h
    if _k_row_stride > 0:
        _k_limit = max(1, (FP32_EXACT_INT_MAX - 1) // _k_row_stride)
        if K > _k_limit:
            while scatter_M_batch > 1 and K_base * scatter_M_batch > _k_limit:
                scatter_M_batch -= 1
            while K_base > 1 and K_base * scatter_M_batch > _k_limit:
                K_base -= 1
            K = K_base * scatter_M_batch
            kernel_assert(
                K * _k_row_stride < FP32_EXACT_INT_MAX,
                f"K * L * N_h must stay under the fp32 exact-integer limit for the scatter index to "
                f"be exact, got {K=} * {L=} * {N_h=} = {K * _k_row_stride} >= {FP32_EXACT_INT_MAX}",
            )

    _set_bytes = lnc * K * L * N_h * C_h * dtype_size
    if _set_bytes > 0:
        while resolved_num_scatter_buffers > 1 and resolved_num_scatter_buffers * _set_bytes > (1 << 32):
            resolved_num_scatter_buffers -= 1

    Q_tile = min(P_MAX, scatter_q_tile, local_N_q)
    num_q_tiles = div_ceil(local_N_q, Q_tile)

    batched_indirect_gather = max_gather_indices_per_indirect != None
    if max_gather_indices_per_indirect == None:
        max_gather_indices_per_indirect = default_max_indices_per_indirect

    gather_cols_per_block = N_p * 4
    _gather_M_req = max(1, max_gather_indices_per_indirect // P_MAX) if batched_indirect_gather else 1
    _ratio_per_M = 1 if gather_method == "copy" else max(1, P_MAX // min(C_h, P_MAX))
    _gather_M_cap = max(1, 64 // _ratio_per_M)

    def _widest_run(cols: int) -> int:
        """Widest run width within the budget and packet limit that divides `cols` columns."""
        m = min(_gather_M_req, cols, _gather_M_cap)
        while m > 1 and cols % m != 0:
            m -= 1
        return m

    gather_hl_group = 1
    gather_M_batch = _widest_run(gather_cols_per_block) if batched_indirect_gather else 1

    _need_gather_cfg = compute_grad_sampling_locations or compute_grad_attention_weights
    _g_sec_q = (
        _calculate_gather_scope_memory(
            N_p,
            C_h,
            dtype_size,
            P_MAX,
            gather_method,
            batched_indirect_gather,
            gather_M_batch,
            num_C_h_tiles,
            gather_hl_group,
        )
        if _need_gather_cfg
        else 0
    )
    _hl_act_q = min(hl_tile, N_h * N_l)
    _swz_q = min(scatter_q_tile // K_base, _hl_act_q)
    _s_sec_q = (
        _calculate_scatter_scope_memory(_hl_act_q * N_p_hl * 4, C_h, dtype_size, _swz_q, local_N_q < P_MAX)
        if compute_grad_value
        else 0
    )

    q_pack_cap = min(MAX_Q_PACK, max(1, local_N_q // P_MAX))
    Q_pack = 1
    while Q_pack * 2 <= q_pack_cap:
        _try = Q_pack * 2
        _h_own_q = _calculate_h_scope_memory(
            _try * H_tile,
            N_l,
            N_p,
            C_h,
            dtype_size,
            padding_mode,
            sampling_locations_layout,
            value_layout,
            combined_scatter_corners,
        )
        _need = h_interleave * (_h_own_q + max(gather_interleave * _g_sec_q, scatter_interleave * _s_sec_q))
        if _need + _try * N_h * N_l * N_p * 4 + hl_tile * N_p_hl * 16 + SBUF_FIT_RESERVE > total_sbuf:
            break
        Q_pack = _try

    num_q_tiles = div_ceil(local_N_q // Q_pack, Q_tile) + div_ceil(local_N_q % Q_pack, Q_tile)

    h_sections = B * num_q_tiles * num_h_tiles
    gather_sections = div_ceil(Q_pack * N_h * N_l, gather_hl_group)
    total_hl_combinations = Q_pack * N_h * N_l
    hl_actual_est = min(hl_tile, total_hl_combinations)
    scatter_sections = div_ceil(total_hl_combinations, hl_tile) * div_ceil(N_p, N_p_hl)
    num_cols_est = hl_actual_est * N_p_hl * 4
    swizzle_degree_est = min(scatter_q_tile // K_base, hl_actual_est)
    needs_padded_tile = local_N_q < P_MAX

    h_own = _calculate_h_scope_memory(
        Q_pack * H_tile,
        N_l,
        N_p,
        C_h,
        dtype_size,
        padding_mode,
        sampling_locations_layout,
        value_layout,
        combined_scatter_corners,
    )
    g_sec = _g_sec_q
    s_sec = (
        _calculate_scatter_scope_memory(num_cols_est, C_h, dtype_size, swizzle_degree_est, needs_padded_tile)
        if compute_grad_value
        else 0
    )

    # zero_tile is popped before the loop, so only these two are live while the stack is deep
    live_heap = hl_tile * N_p_hl * 4 * 4 + Q_pack * N_h * N_l * N_p * 4  # iota_k_scale + hl_off
    stack_budget = total_sbuf - live_heap - SBUF_FIT_RESERVE

    oob_sec = _oob_scope_memory(Q_pack * H_tile, N_l, N_p) if padding_mode == "zeros" else 0

    def _fits(h_il: int, g_il: int, s_il: int) -> bool:
        per_h = h_own + max(
            oob_sec,
            min(g_il, gather_sections) * g_sec,
            min(s_il, scatter_sections) * s_sec,
        )
        return min(h_il, h_sections) * per_h <= stack_budget

    if _fits(h_interleave, gather_interleave, scatter_interleave):
        for cand in range(MAX_INTERLEAVE, scatter_interleave, -1):
            if _fits(h_interleave, gather_interleave, cand):
                scatter_interleave = cand
                break
        if s_sec == 0:
            free_gather = MAX_INTERLEAVE
        else:
            free_gather = max(1, (min(scatter_interleave, scatter_sections) * s_sec) // max(1, g_sec))
        for cand in range(min(MAX_INTERLEAVE, free_gather), gather_interleave, -1):
            if _fits(h_interleave, cand, scatter_interleave):
                gather_interleave = cand
                break
    else:
        while h_interleave > 1 and not _fits(h_interleave, gather_interleave, scatter_interleave):
            h_interleave -= 1
        while scatter_interleave > 1 and not _fits(h_interleave, gather_interleave, scatter_interleave):
            scatter_interleave -= 1
        while gather_interleave > 1 and not _fits(h_interleave, gather_interleave, scatter_interleave):
            gather_interleave -= 1

    gather_interleave = min(gather_interleave, gather_sections)
    scatter_interleave = min(scatter_interleave, scatter_sections)
    h_interleave = min(h_interleave, h_sections)

    est_stack = min(h_interleave, h_sections) * (
        h_own
        + max(
            min(gather_interleave, gather_sections) * g_sec,
            min(scatter_interleave, scatter_sections) * s_sec,
        )
    )

    num_gather_bufs = gather_interleave if batched_indirect_gather else 1

    # Log comprehensive config
    logger = get_logger("ms_deformable_attention_bwd")
    logger.info(
        "Config: B=" + str(B) + ", N_q=" + str(N_q) + ", local_N_q=" + str(local_N_q) + ", "
        "N_h=" + str(N_h) + ", C_h=" + str(C_h) + ", N_l=" + str(N_l) + ", N_p=" + str(N_p) + ", "
        "L=" + str(L) + ", dtype=" + str(value.dtype) + ", value_layout=" + str(value_layout) + ", "
        "sampling_locations_layout=" + str(sampling_locations_layout) + ", "
        "padding_mode=" + str(padding_mode) + ", "
        "num_scatter_buffers=" + str(resolved_num_scatter_buffers) + ", "
        "h_interleave=" + str(h_interleave) + ", gather_interleave=" + str(gather_interleave) + ", "
        "scatter_interleave=" + str(scatter_interleave) + ", est_stack=" + str(est_stack) + ", "
        "stack_budget=" + str(stack_budget) + ", "
        "spatial_shapes=" + str(spatial_shapes) + ", level_start_index=" + str(level_start_index) + ", "
        "P_MAX=" + str(P_MAX) + ", Q_tile=" + str(Q_tile) + ", H_tile=" + str(H_tile) + ", "
        "num_q_tiles=" + str(num_q_tiles) + ", num_h_tiles=" + str(num_h_tiles) + ", "
        "num_C_h_tiles=" + str(num_C_h_tiles) + ", "
        "lnc=" + str(lnc) + ", shard_id=" + str(shard_id) + ", q_start_global=" + str(q_start_global) + ", "
        "K="
        + str(K)
        + ", Q_pack="
        + str(Q_pack)
        + ", N_p_hl="
        + str(N_p_hl)
        + ", hl_tile="
        + str(hl_tile)
        + ", zero_tile_free_dim="
        + str(zero_tile_free_dim)
        + ", combined_scatter_corners="
        + str(combined_scatter_corners)
        + ", accumulate_scatter_across_batches="
        + str(accumulate_scatter_across_batches)
        + ", batched_indirect_scatter="
        + str(batched_indirect_scatter)
        + ", max_scatter_indices_per_indirect="
        + str(max_scatter_indices_per_indirect)
        + ", scatter_M_batch="
        + str(scatter_M_batch)
        + ", K_base="
        + str(K_base)
        + ", batched_indirect_gather="
        + str(batched_indirect_gather)
        + ", max_gather_indices_per_indirect="
        + str(max_gather_indices_per_indirect)
        + ", gather_M_batch="
        + str(gather_M_batch)
        + ", gather_hl_group="
        + str(gather_hl_group)
        + ", num_gather_bufs="
        + str(num_gather_bufs)
        + ", gather_method="
        + str(gather_method)
    )

    if _need_gather_cfg:
        _rows = B * N_q * N_h * N_l * N_p * 4
        logger.info(
            "Gather issues "
            + str(_rows)
            + " descriptors of "
            + str(C_h * dtype_size)
            + " B ("
            + str(_rows * C_h * dtype_size // 1024 // 1024)
            + " MiB) against a "
            + str(B * L * N_h * C_h * dtype_size // 1024)
            + " KiB value tensor. Descriptor size is set by the innermost access-pattern "
            + "dimension and cannot be widened by batching; only a longer contiguous run per "
            + "index (e.g. combining horizontally adjacent corners, which needs a w-stride of "
            + "C_h) reduces it."
        )

    return MSDeformAttnBwdConfig(
        B=B,
        N_q=N_q,
        N_h=N_h,
        C_h=C_h,
        N_l=N_l,
        N_p=N_p,
        L=L,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        dtype=value.dtype,
        value_layout=value_layout,
        sampling_locations_layout=sampling_locations_layout,
        P_MAX=P_MAX,
        Q_tile=Q_tile,
        H_tile=H_tile,
        num_q_tiles=num_q_tiles,
        num_h_tiles=num_h_tiles,
        num_C_h_tiles=num_C_h_tiles,
        h_interleave=h_interleave,
        gather_interleave=gather_interleave,
        scatter_interleave=scatter_interleave,
        total_sbuf=total_sbuf,
        lnc=lnc,
        shard_id=shard_id,
        q_start_global=q_start_global,
        local_N_q=local_N_q,
        K=K,
        N_p_hl=N_p_hl,
        Q_pack=Q_pack,
        hl_tile=hl_tile,
        zero_tile_free_dim=zero_tile_free_dim,
        K_rep=K_rep,
        combined_scatter_corners=combined_scatter_corners,
        accumulate_scatter_across_batches=accumulate_scatter_across_batches,
        batched_indirect_scatter=batched_indirect_scatter,
        max_scatter_indices_per_indirect=max_scatter_indices_per_indirect,
        scatter_M_batch=scatter_M_batch,
        scatter_q_tile=scatter_q_tile,
        K_base=K_base,
        num_scatter_buffers=resolved_num_scatter_buffers,
        batched_indirect_gather=batched_indirect_gather,
        max_gather_indices_per_indirect=max_gather_indices_per_indirect,
        gather_M_batch=gather_M_batch,
        gather_hl_group=gather_hl_group,
        num_gather_bufs=num_gather_bufs,
        gather_method=gather_method,
    )
