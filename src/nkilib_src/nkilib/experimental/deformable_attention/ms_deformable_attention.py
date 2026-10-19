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
Multi-scale deformable attention kernel for NeuronCore.

This kernel implements multi-scale deformable attention using indirect DMA transpose for efficient
gathering of values from multiple feature pyramid levels.
"""

from dataclasses import dataclass
from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...core.utils.logging import get_logger

INDEX_ALIGN = 16  # Turbo batched indirect scatter-add DGE indices require 16 byte alignment
FP32_EXACT_INT_MAX = 1 << 24  # Largest integer exactly representable in fp32 (2**24)


@nki.jit
def ms_deformable_attention(
    value: nl.NkiTensor,
    spatial_shapes: tuple,
    level_start_index: tuple,
    sampling_locations: nl.NkiTensor,
    attention_weights: nl.NkiTensor,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    align_corners: bool = False,
    padding_mode: str = "zeros",
    max_indices_per_indirect: Optional[int] = None,
    gather_method: str = "transpose",
) -> nl.NkiTensor:
    """
    Multi-scale deformable attention kernel that uses indirect DMA transpose.

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
        max_indices_per_indirect (Optional[int]): Cap on gather indices (P_MAX * M_batch) per batched
            gather. Setting this enables the batched indirect gather path (M_batch =
            cap // P_MAX); None disables it (each (head, level) gathered separately). Default: None
        gather_method (str): How values are gathered: "transpose" (dma_transpose to [channel, query]
            then nc_transpose to [query, channel]) or "copy" (indirect dma_copy to [query, channel]).
            Default: "transpose"

    Returns:
        output (nl.NkiTensor): Attention output in HBM, shape (B, N_q, N_h * C_h)

    Pseudocode:
        # For each batch, query, and head:
        for b in range(B):
            for q in range(N_q):
                for h in range(N_h):
                    output[b, q, h*C_h:(h+1)*C_h] = 0
                    # For each level and sampling point:
                    for l in range(N_l):
                        for p in range(N_p):
                            # Get normalized coordinates and scale to pixel space
                            x, y = sampling_locations[b, q, h, l, p]
                            x_scaled = x * (H_l - 1) if align_corners else x * H_l - 0.5
                            y_scaled = y * (W_l - 1) if align_corners else y * W_l - 0.5

                            # Compute bilinear interpolation corners
                            x0, y0 = floor(x_scaled), floor(y_scaled)
                            x1, y1 = x0 + 1, y0 + 1

                            # Clamp to valid range
                            x0_c, y0_c = clamp(x0, 0, H_l-1), clamp(y0, 0, W_l-1)
                            x1_c, y1_c = clamp(x1, 0, H_l-1), clamp(y1, 0, W_l-1)

                            # Compute bilinear weights
                            dx, dy = x_scaled - x0, y_scaled - y0
                            w_00 = (1 - dx) * (1 - dy)
                            w_01 = (1 - dx) * dy
                            w_10 = dx * (1 - dy)
                            w_11 = dx * dy

                            # Apply padding mode (zeros: mask OOB, border: use clamped)
                            if padding_mode == "zeros":
                                # Zero out weights for out-of-bounds corners
                                w_00 *= (x0 == x0_c) * (y0 == y0_c)
                                w_01 *= (x0 == x0_c) * (y1 == y1_c)
                                w_10 *= (x1 == x1_c) * (y0 == y0_c)
                                w_11 *= (x1 == x1_c) * (y1 == y1_c)

                            # Gather values and accumulate weighted sum
                            attn_w = attention_weights[b, q, h, l, p]
                            output[b, q, h*C_h:(h+1)*C_h] += (
                                w_00 * attn_w * value[b, level_start_index[l] + x0_c*W_l + y0_c, h, :] +
                                w_01 * attn_w * value[b, level_start_index[l] + x0_c*W_l + y1_c, h, :] +
                                w_10 * attn_w * value[b, level_start_index[l] + x1_c*W_l + y0_c, h, :] +
                                w_11 * attn_w * value[b, level_start_index[l] + x1_c*W_l + y1_c, h, :]
                            )
    """
    # LNC sharding on Q dimension
    _, lnc, shard_id = get_verified_program_sharding_info("ms_deformable_attention", (0, 1))

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
        lnc,
        shard_id,
        max_indices_per_indirect,
        gather_method,
    )

    # Extract config values
    P_MAX = cfg.P_MAX
    B, N_q, N_h, C_h = cfg.B, cfg.N_q, cfg.N_h, cfg.C_h
    N_l, N_p = cfg.N_l, cfg.N_p
    L = cfg.L
    spatial_shapes = cfg.spatial_shapes
    level_start_index = cfg.level_start_index
    Q_pack = cfg.Q_pack
    Q_tile = cfg.Q_tile
    H_tile = cfg.H_tile
    num_q_tiles = cfg.num_q_tiles
    num_h_tiles = cfg.num_h_tiles
    num_C_h_tiles = cfg.num_C_h_tiles
    dtype = cfg.dtype
    h_interleave = cfg.h_interleave
    gather_interleave = cfg.gather_interleave
    total_sbuf_req = cfg.total_sbuf
    q_start_global = cfg.q_start_global
    local_N_q = cfg.local_N_q

    # Engine rotation state
    data_copy_modulo = cfg.data_copy_scalar_modulo
    index_copy_modulo = cfg.index_copy_scalar_modulo
    scale_engine_modulo = cfg.scale_scalar_modulo
    data_copy_engine_idx = 0
    index_copy_engine_idx = 0
    scale_engine_idx = 0

    # Allocate output in HBM
    output = nl.ndarray((B, N_q, N_h * C_h), dtype=dtype, buffer=nl.shared_hbm)

    # Initialize SBUF manager
    logger = get_logger("ms_deformable_attention")
    sbm = SbufManager(0, total_sbuf_req, logger=logger)

    num_snake_hbm_bufs = cfg.num_gather_bufs
    snake_hbm_buf_idx = 0

    # Pre-allocate the snake index HBM scratch space
    if cfg.batched_indirect_gather and cfg.M_batch > 1:
        snake_total_cols = Q_pack * N_h * N_l * (N_p * 4)
        snake_ng_static = snake_total_cols // cfg.M_batch
        snake_buf_elems_static = snake_ng_static * P_MAX * cfg.M_batch
        snake_hbm = nl.ndarray((num_snake_hbm_bufs * snake_buf_elems_static,), dtype=nl.uint32, buffer=nl.private_hbm)

    # ====================================================================
    # Step 0: Per (head, level) base offsets into the flattened value rows
    # ====================================================================
    head_level_offset_tiles = []
    for h_tile_idx in range(num_h_tiles):
        h_start_tile = h_tile_idx * H_tile
        h_actual_tile = min(h_start_tile + H_tile, N_h) - h_start_tile
        hl_off_tile = sbm.alloc_heap((P_MAX, Q_pack * h_actual_tile, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
        for level_idx in range(N_l):
            if cfg.value_layout == "BLNC":
                # memory order [L, N_h]: offset = level_start * N_h + h_global
                level_head_base = level_start_index[level_idx] * N_h + h_start_tile
                head_step = 1
            else:  # BNLC, memory order [N_h, L]
                level_head_base = h_start_tile * L + level_start_index[level_idx]
                head_step = L
            nisa.iota(
                dst=hl_off_tile[:, :, level_idx, :],
                pattern=[[0, Q_pack], [head_step * C_h, h_actual_tile], [0, N_p]],
                offset=int(level_head_base * C_h),
                channel_multiplier=0,
            )
        head_level_offset_tiles.append(hl_off_tile)

    # Query tiles as (first query, partitions, queries per partition)
    query_segments = []
    n_groups = local_N_q // Q_pack
    group_start = 0
    while group_start < n_groups:
        seg_parts = min(P_MAX, n_groups - group_start)
        query_segments.append((group_start * Q_pack, seg_parts, Q_pack))
        group_start = group_start + seg_parts
    remainder_start = n_groups * Q_pack
    remainder = local_N_q - remainder_start
    rem_done = 0
    while rem_done < remainder:
        seg_parts = min(P_MAX, remainder - rem_done)
        query_segments.append((remainder_start + rem_done, seg_parts, 1))
        rem_done = rem_done + seg_parts

    # Open H-tile scope
    sbm.open_scope(interleave_degree=h_interleave, name="ms_deformable_attention_h_scope")

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
                # Step 1: load x, y, attn_w from HBM
                # ====================================================================
                x = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                y = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                attn_w = sbm.alloc_stack((q_actual, qh, N_l, N_p), dtype=dtype, buffer=nl.sbuf)

                # Load x and y
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
                nisa.dma_copy(
                    dst=attn_w,
                    src=attention_weights[batch_idx, q_start:q_end, h_start:h_end, :, :],
                    dge_mode=nisa.dge_mode.none,
                )

                # ====================================================================
                # Step 2: Scale coordinates by spatial dimensions based on align_corners
                # ====================================================================
                for level_idx in range(N_l):
                    H_l, W_l = spatial_shapes[level_idx]
                    if align_corners:
                        # align_corners=True: [0,1] -> [0, W_l-1]
                        x_scale, y_scale = float(W_l - 1), float(H_l - 1)
                        x_shift, y_shift = None, None
                    else:
                        # align_corners=False: [0,1] -> [-0.5, W_l-0.5]
                        x_scale, y_scale = float(W_l), float(H_l)
                        x_shift, y_shift = -0.5, -0.5

                    nisa.tensor_scalar(
                        dst=x[:, :, level_idx, :],
                        data=x_src[:, :, level_idx, :],
                        op0=nl.multiply,
                        operand0=x_scale,
                        op1=None if x_shift == None else nl.add,
                        operand1=x_shift,
                        engine=_get_tensor_scalar_engine(scale_engine_idx + 2 * level_idx, scale_engine_modulo),
                    )
                    nisa.tensor_scalar(
                        dst=y[:, :, level_idx, :],
                        data=y_src[:, :, level_idx, :],
                        op0=nl.multiply,
                        operand0=y_scale,
                        op1=None if y_shift == None else nl.add,
                        operand1=y_shift,
                        engine=_get_tensor_scalar_engine(scale_engine_idx + 2 * level_idx + 1, scale_engine_modulo),
                    )

                # ====================================================================
                # Step 3: Compute bilinear coordinates x0, x1, y0, y1
                # ====================================================================
                hlp_total = qh * N_l * N_p

                unclamped = sbm.alloc_stack((q_actual, 4, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                x0_unclamped = unclamped[:, 0]
                x1_unclamped = unclamped[:, 1]
                y0_unclamped = unclamped[:, 2]
                y1_unclamped = unclamped[:, 3]

                unclamped_hlp = unclamped.reshape((q_actual, 4, hlp_total))
                x_hlp = x.reshape((q_actual, hlp_total))
                y_hlp = y.reshape((q_actual, hlp_total))

                round_trip = sbm.alloc_stack((q_actual, 2, hlp_total), dtype=nl.float32, buffer=nl.sbuf)
                rounded_up = sbm.alloc_stack((q_actual, 2, hlp_total), dtype=nl.int32, buffer=nl.sbuf)

                nisa.tensor_copy(
                    dst=unclamped_hlp[:, 0],
                    src=x_hlp,
                    engine=_get_tensor_copy_engine(index_copy_engine_idx, index_copy_modulo),
                )
                nisa.tensor_copy(
                    dst=unclamped_hlp[:, 2],
                    src=y_hlp,
                    engine=_get_tensor_copy_engine(index_copy_engine_idx + 1, index_copy_modulo),
                )
                index_copy_engine_idx += 2

                nisa.tensor_copy(
                    dst=round_trip,
                    src=unclamped_hlp.slice(dim=1, start=0, end=3, step=2),
                    engine=_get_tensor_copy_engine(index_copy_engine_idx, index_copy_modulo),
                )
                index_copy_engine_idx += 1
                nisa.tensor_tensor(
                    dst=rounded_up[:, 0],
                    data1=round_trip[:, 0],
                    data2=x_hlp,
                    op=nl.greater,
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=rounded_up[:, 1],
                    data1=round_trip[:, 1],
                    data2=y_hlp,
                    op=nl.greater,
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=unclamped_hlp.slice(dim=1, start=0, end=3, step=2),
                    data1=unclamped_hlp.slice(dim=1, start=0, end=3, step=2),
                    data2=rounded_up,
                    op=nl.subtract,
                    engine=nisa.engine.vector,
                )
                # x1 = x0 + 1 and y1 = y0 + 1 in one strided instruction
                nisa.tensor_scalar(
                    dst=unclamped_hlp.slice(dim=1, start=1, end=4, step=2),
                    data=unclamped_hlp.slice(dim=1, start=0, end=3, step=2),
                    op0=nl.add,
                    operand0=1,
                    engine=_get_tensor_scalar_engine(scale_engine_idx + 1, scale_engine_modulo),
                )

                frac_terms = sbm.alloc_stack((q_actual, qh, N_l, N_p, 4), dtype=nl.float32, buffer=nl.sbuf)
                frac_hlp = frac_terms.reshape((q_actual, hlp_total, 4))
                d_terms = frac_hlp.slice(dim=2, start=1, end=4, step=2)
                one_minus_d_terms = frac_hlp.slice(dim=2, start=0, end=3, step=2)

                nisa.tensor_tensor(
                    dst=frac_hlp[:, :, 1],
                    data1=x_hlp,
                    data2=unclamped_hlp[:, 0],
                    op=nl.subtract,
                    engine=nisa.engine.vector,
                )
                nisa.tensor_tensor(
                    dst=frac_hlp[:, :, 3],
                    data1=y_hlp,
                    data2=unclamped_hlp[:, 2],
                    op=nl.subtract,
                    engine=nisa.engine.vector,
                )

                clamped = sbm.alloc_stack((q_actual, 4, qh, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                clamped_hlp = clamped.reshape((q_actual, 4, hlp_total))
                x0_clamped = clamped[:, 0]
                x1_clamped = clamped[:, 1]
                y0_clamped = clamped[:, 2]
                y1_clamped = clamped[:, 3]

                for level_idx in range(N_l):
                    H_l, W_l = spatial_shapes[level_idx]
                    for coord in range(4):
                        nisa.tensor_scalar(
                            dst=clamped[:, coord, :, level_idx, :],
                            data=unclamped[:, coord, :, level_idx, :],
                            op0=nl.minimum,
                            operand0=float(W_l - 1) if coord < 2 else float(H_l - 1),
                            op1=nl.maximum,
                            operand1=0.0,
                        )

                # ====================================================================
                # Step 4: Compute fractional contributions: 1-dx, 1-dy
                # ====================================================================
                # dx and dy were already produced in Step 3 as a by-product of the floor;
                # 1-dx and 1-dy fill the remaining two slots in one strided instruction.
                nisa.tensor_scalar(
                    dst=one_minus_d_terms,
                    data=d_terms,
                    op0=nl.multiply,
                    operand0=-1.0,
                    op1=nl.add,
                    operand1=1.0,
                    engine=nisa.engine.scalar,
                )

                # ====================================================================
                # Step 4b: Mask out-of-bounds coordinates for padding_mode zeros
                # ====================================================================
                if padding_mode == "zeros":
                    coord_in_bounds = sbm.alloc_stack((q_actual, hlp_total, 4), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=coord_in_bounds,
                        data1=clamped_hlp.permute((0, 2, 1)),
                        data2=unclamped_hlp.permute((0, 2, 1)),
                        op=nl.equal,
                        engine=nisa.engine.vector,
                    )
                    nisa.tensor_tensor(
                        dst=frac_hlp,
                        data1=frac_hlp,
                        data2=coord_in_bounds,
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )

                # ====================================================================
                # Step 5: Compute bilinear weights: w_00, w_01, w_10, w_11
                # ====================================================================
                combined_weights = sbm.alloc_stack((q_actual, qh, N_l, N_p, 4), dtype=nl.float32, buffer=nl.sbuf)
                cw_hlp = combined_weights.reshape((q_actual, hlp_total, 4))
                for y_sel in range(2):
                    nisa.tensor_tensor(
                        dst=cw_hlp.slice(dim=2, start=2 * y_sel, end=2 * y_sel + 2, step=1),
                        data1=frac_hlp.slice(dim=2, start=0, end=2, step=1),
                        data2=frac_hlp[:, :, 2 + y_sel].expand_dim(2).broadcast(2, 2),
                        op=nl.multiply,
                        engine=nisa.engine.vector,
                    )

                # ====================================================================
                # Step 6: Compute flat indexes
                # ====================================================================
                flat_idx_all = sbm.alloc_stack(
                    (q_actual, qh, N_l, N_p, 4), dtype=nl.uint32, buffer=nl.sbuf, align=INDEX_ALIGN
                )

                hl_off = head_level_offset_tiles[h_tile_idx]
                row_stride = (N_h if cfg.value_layout == "BLNC" else 1) * C_h
                x_term = sbm.alloc_stack((q_actual, 2, qh, N_p), dtype=nl.uint32, buffer=nl.sbuf)

                for level_idx in range(N_l):
                    H_l, W_l = spatial_shapes[level_idx]

                    for x_sel in range(2):
                        nisa.scalar_tensor_tensor(
                            dst=x_term[:, x_sel, :, :],
                            data=(x0_clamped if x_sel == 0 else x1_clamped)[:, :, level_idx, :],
                            op0=nl.multiply,
                            operand0=float(row_stride),
                            op1=nl.add,
                            operand1=hl_off[0:q_actual, 0:qh, level_idx, :],
                        )

                    # corner order is (y0x0, y0x1, y1x0, y1x1)
                    for corner in range(4):
                        y_clamped = y0_clamped if corner // 2 == 0 else y1_clamped
                        nisa.scalar_tensor_tensor(
                            dst=flat_idx_all[:, :, level_idx, :, corner],
                            data=y_clamped[:, :, level_idx, :],
                            op0=nl.multiply,
                            operand0=float(W_l * row_stride),
                            op1=nl.add,
                            operand1=x_term[:, corner % 2, :, :],
                        )

                # ====================================================================
                # Step 7: Fold in the attention weights
                # ====================================================================
                nisa.tensor_tensor(
                    dst=cw_hlp,
                    data1=cw_hlp,
                    data2=attn_w.reshape((q_actual, hlp_total)).expand_dim(2).broadcast(2, 4),
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )

                # ====================================================================
                # Step 8: Gather and accumulate
                # ====================================================================
                accum = sbm.alloc_stack((q_actual, q_pack_tile, h_actual, C_h), dtype=nl.float32, buffer=nl.sbuf)
                accum_g = accum.reshape((q_actual, qh, C_h))
                nisa.memset(dst=accum.reshape((q_actual, qh * C_h)), value=0.0)

                value_flat_1d = value.reshape((B * L * N_h * C_h,))
                batch_offset = batch_idx * L * N_h * C_h

                sbm.open_scope(interleave_degree=gather_interleave, name="ms_deformable_attention_gather_scope")

                total_corners = N_p * 4
                cols_per_group_k = N_l * N_p * 4
                min_C_h_P_MAX = min(C_h, P_MAX)

                if cfg.batched_indirect_gather:
                    num_cols = qh * N_l * total_corners
                    M_batch = 1
                    m_candidate = 1
                    while m_candidate <= min(cfg.M_batch, num_cols):
                        legal = cols_per_group_k % m_candidate == 0
                        if m_candidate % cols_per_group_k == 0:
                            legal = True
                        if legal and num_cols % m_candidate == 0:
                            M_batch = m_candidate
                        m_candidate = m_candidate + 1

                    flat_idx_cols = flat_idx_all.reshape((q_actual, num_cols))

                    effective_q = P_MAX
                    if q_actual < P_MAX:
                        flat_idx_padded = sbm.alloc_stack(
                            (P_MAX, num_cols), dtype=nl.uint32, buffer=nl.sbuf, align=INDEX_ALIGN
                        )
                        nisa.memset(dst=flat_idx_padded, value=0)
                        nisa.tensor_copy(
                            dst=flat_idx_padded[0:q_actual, :],
                            src=flat_idx_cols,
                            engine=_get_tensor_copy_engine(index_copy_engine_idx, index_copy_modulo),
                        )
                        index_copy_engine_idx += 1
                        idx_2d = flat_idx_padded
                    else:
                        idx_2d = flat_idx_cols

                    if M_batch == 1:
                        idx_snake = idx_2d
                    else:
                        snake_ng = num_cols // M_batch
                        snake_N = M_batch
                        snake_buf_elems = snake_ng * effective_q * snake_N
                        snake_buf_off = snake_hbm_buf_idx * snake_buf_elems
                        snake_src_view = idx_2d[0:effective_q, 0:num_cols].reshape_dim(1, [snake_ng, snake_N])
                        nisa.dma_copy(
                            dst=snake_hbm.ap(
                                [[snake_N, effective_q], [effective_q * snake_N, snake_ng], [1, snake_N]],
                                offset=snake_buf_off,
                            ),
                            src=snake_src_view,
                            dge_mode=nisa.dge_mode.hwdge,
                        )
                        idx_snake = sbm.alloc_stack((P_MAX, num_cols), dtype=nl.uint32, buffer=nl.sbuf, align=32)
                        nisa.dma_transpose(
                            dst=idx_snake,
                            src=snake_hbm.ap(
                                [[effective_q, snake_ng * snake_N], [1, effective_q]], offset=snake_buf_off
                            ),
                            axes=(1, 0),
                        )

                    num_runs = num_cols // M_batch

                    cols_per_group = cols_per_group_k
                    weights_col = combined_weights.reshape((q_actual, num_cols))

                    sbm.open_scope(
                        interleave_degree=cfg.num_gather_bufs,
                        name="ms_deformable_attention_gather_run_scope",
                    )
                    for run in range(num_runs):
                        c0 = run * M_batch
                        NTOT = effective_q * M_batch
                        idx_run = idx_snake[0:effective_q, c0 : c0 + M_batch]

                        gathered_run_flat = sbm.alloc_stack(
                            (P_MAX, M_batch * C_h), dtype=dtype, buffer=nl.sbuf, align=32
                        )
                        gathered_run = gathered_run_flat.reshape((P_MAX, M_batch, C_h))

                        if cfg.gather_method == "transpose":
                            gathered_cq = sbm.alloc_stack(
                                (min_C_h_P_MAX, num_C_h_tiles, M_batch * P_MAX),
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
                                    src=value_flat_1d.ap(
                                        pattern=[[c_actual, NTOT], [1, c_actual]],
                                        offset=c_start + batch_offset,
                                        vector_offset=idx_run,
                                        indirect_dim=0,
                                    ),
                                    axes=(1, 0),
                                )
                            gathered_cq_r = gathered_cq.reshape((min_C_h_P_MAX, num_C_h_tiles, P_MAX, M_batch))
                            for m in range(M_batch):
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
                                    dst=gathered_run[:, m, :],
                                    src=corner_ps,
                                    engine=_get_tensor_copy_engine(data_copy_engine_idx, data_copy_modulo),
                                )
                                data_copy_engine_idx += 1
                        else:
                            nisa.dma_copy(
                                dst=gathered_run_flat[0:effective_q, :],
                                src=value_flat_1d.ap(
                                    pattern=[[C_h, NTOT], [1, C_h]],
                                    offset=batch_offset,
                                    vector_offset=idx_run,
                                    indirect_dim=0,
                                ),
                            )

                        weighted_run = sbm.alloc_stack((q_actual, M_batch, C_h), dtype=nl.float32, buffer=nl.sbuf)
                        weights_run = weights_col[0:q_actual, c0 : c0 + M_batch].expand_dim(2).broadcast(2, C_h)
                        nisa.tensor_tensor(
                            dst=weighted_run,
                            data1=gathered_run[0:q_actual, :, :],
                            data2=weights_run,
                            op=nl.multiply,
                        )

                        if M_batch <= cols_per_group:
                            span = M_batch
                            while span > 1:
                                half = (span + 1) // 2
                                tail = span - half
                                nisa.tensor_tensor(
                                    dst=weighted_run[0:q_actual, 0:tail, :],
                                    data1=weighted_run[0:q_actual, 0:tail, :],
                                    data2=weighted_run[0:q_actual, half:span, :],
                                    op=nl.add,
                                )
                                span = half
                            group_idx = c0 // cols_per_group
                            nisa.tensor_tensor(
                                dst=accum_g[0:q_actual, group_idx, :],
                                data1=accum_g[0:q_actual, group_idx, :],
                                data2=weighted_run[0:q_actual, 0, :],
                                op=nl.add,
                            )
                        else:
                            for m in range(M_batch):
                                nisa.tensor_tensor(
                                    dst=accum_g[0:q_actual, (c0 + m) // cols_per_group, :],
                                    data1=accum_g[0:q_actual, (c0 + m) // cols_per_group, :],
                                    data2=weighted_run[0:q_actual, m, :],
                                    op=nl.add,
                                )

                        sbm.increment_section()

                    sbm.close_scope()

                else:
                    # group_idx walks the packed (query, head) axis
                    for group_idx in range(qh):
                        for level_idx in range(N_l):
                            flat_idx_all_reshaped = flat_idx_all.reshape((q_actual, qh, N_l, N_p * 4))

                            if cfg.gather_method == "copy":
                                effective_q = P_MAX
                                block_idx = flat_idx_all_reshaped[:, group_idx, level_idx, :]
                                idx_block = sbm.alloc_stack(
                                    (P_MAX, total_corners), dtype=nl.uint32, buffer=nl.sbuf, align=INDEX_ALIGN
                                )

                                if q_actual < P_MAX:
                                    nisa.memset(dst=idx_block, value=0)
                                nisa.tensor_copy(
                                    dst=idx_block[0:q_actual, :],
                                    src=block_idx,
                                    engine=_get_tensor_copy_engine(index_copy_engine_idx, index_copy_modulo),
                                )
                                index_copy_engine_idx += 1

                                gathered_flat = sbm.alloc_stack(
                                    (P_MAX, total_corners * C_h), dtype=dtype, buffer=nl.sbuf, align=32
                                )
                                gathered = gathered_flat.reshape((P_MAX, total_corners, C_h))
                                NTOT = effective_q
                                for corner_col in range(total_corners):
                                    nisa.dma_copy(
                                        dst=gathered[0:effective_q, corner_col, :],
                                        src=value_flat_1d.ap(
                                            pattern=[[C_h, NTOT], [1, C_h]],
                                            offset=batch_offset,
                                            vector_offset=idx_block[0:effective_q, corner_col : corner_col + 1],
                                            indirect_dim=0,
                                        ),
                                    )

                                for point_idx in range(N_p):
                                    for corner_idx in range(4):
                                        col = point_idx * 4 + corner_idx
                                        gathered_corner = gathered[:, col, :]
                                        if corner_idx % 2 == 0:
                                            weighted = sbm.alloc_stack((q_actual, C_h), dtype=dtype, buffer=nl.sbuf)
                                            nisa.tensor_scalar(
                                                dst=weighted,
                                                data=gathered_corner[0:q_actual, :],
                                                operand0=combined_weights[
                                                    :, group_idx, level_idx, point_idx, corner_idx
                                                ],
                                                op0=nl.multiply,
                                            )
                                            nisa.tensor_tensor(
                                                dst=accum_g[0:q_actual, group_idx, :],
                                                data1=accum_g[0:q_actual, group_idx, :],
                                                data2=weighted,
                                                op=nl.add,
                                            )
                                        else:
                                            nisa.scalar_tensor_tensor(
                                                dst=accum_g[0:q_actual, group_idx, :],
                                                data=gathered_corner[0:q_actual, :],
                                                op0=nl.multiply,
                                                operand0=combined_weights[
                                                    :, group_idx, level_idx, point_idx, corner_idx
                                                ],
                                                op1=nl.add,
                                                operand1=accum_g[0:q_actual, group_idx, :],
                                            )

                                sbm.increment_section()
                            else:
                                gathered_all = sbm.alloc_stack(
                                    (min_C_h_P_MAX, num_C_h_tiles, total_corners * P_MAX),
                                    dtype=dtype,
                                    buffer=nl.sbuf,
                                    align=32,
                                )

                                vector_offset_view = None
                                if q_actual < P_MAX:
                                    flat_idx_padded = sbm.alloc_stack(
                                        (P_MAX, total_corners), dtype=nl.uint32, buffer=nl.sbuf, align=INDEX_ALIGN
                                    )
                                    nisa.memset(dst=flat_idx_padded, value=0)

                                    nisa.tensor_copy(
                                        dst=flat_idx_padded[0:q_actual, :],
                                        src=flat_idx_all_reshaped[:, group_idx, level_idx, :],
                                        engine=_get_tensor_copy_engine(index_copy_engine_idx, index_copy_modulo),
                                    )
                                    index_copy_engine_idx += 1
                                    vector_offset_view = flat_idx_padded
                                else:
                                    vector_offset_view = flat_idx_all_reshaped[:, group_idx, level_idx, :]

                                for c_tile_idx in range(num_C_h_tiles):
                                    c_start = c_tile_idx * P_MAX
                                    c_end = min(c_start + P_MAX, C_h)
                                    c_actual = c_end - c_start

                                    nisa.dma_transpose(
                                        dst=gathered_all[0:c_actual, c_tile_idx, :],
                                        src=value_flat_1d.ap(
                                            pattern=[[c_actual, total_corners * P_MAX], [1, c_actual]],
                                            offset=c_start + batch_offset,
                                            vector_offset=vector_offset_view,
                                            indirect_dim=0,
                                        ),
                                        axes=(1, 0),
                                    )

                                gathered_reshaped = gathered_all.reshape((min_C_h_P_MAX, num_C_h_tiles, N_p, 4, P_MAX))

                                for point_idx in range(N_p):
                                    for corner_idx in range(4):
                                        gathered_corner = nl.ndarray((P_MAX, C_h), dtype=dtype, buffer=nl.psum)

                                        for c_tile_idx in range(num_C_h_tiles):
                                            c_start = c_tile_idx * P_MAX
                                            c_end = min(c_start + P_MAX, C_h)
                                            c_actual = c_end - c_start

                                            nisa.nc_transpose(
                                                dst=gathered_corner[:, c_start:c_end],
                                                data=gathered_reshaped[
                                                    0:c_actual, c_tile_idx, point_idx, corner_idx, :
                                                ],
                                                engine=nisa.engine.tensor,
                                            )

                                        if corner_idx % 2 == 0:
                                            weighted = sbm.alloc_stack((q_actual, C_h), dtype=dtype, buffer=nl.sbuf)

                                            nisa.tensor_scalar(
                                                dst=weighted,
                                                data=gathered_corner[0:q_actual, :],
                                                operand0=combined_weights[
                                                    :, group_idx, level_idx, point_idx, corner_idx
                                                ],
                                                op0=nl.multiply,
                                            )

                                            nisa.tensor_tensor(
                                                dst=accum_g[0:q_actual, group_idx, :],
                                                data1=accum_g[0:q_actual, group_idx, :],
                                                data2=weighted,
                                                op=nl.add,
                                            )
                                        else:
                                            nisa.scalar_tensor_tensor(
                                                dst=accum_g[0:q_actual, group_idx, :],
                                                data=gathered_corner[0:q_actual, :],
                                                op0=nl.multiply,
                                                operand0=combined_weights[
                                                    :, group_idx, level_idx, point_idx, corner_idx
                                                ],
                                                op1=nl.add,
                                                operand1=accum_g[0:q_actual, group_idx, :],
                                            )

                                sbm.increment_section()

                sbm.close_scope()

                # ====================================================================
                # Step 10: Write accumulated results to HBM
                # ====================================================================
                output_flat = output.reshape((B * N_q * N_h * C_h,))
                out_base = batch_idx * N_q * N_h * C_h + q_start * N_h * C_h
                if N_h == 1:
                    store_pattern = [[q_pack_tile * C_h, q_actual], [1, q_pack_tile * C_h]]
                else:
                    store_pattern = [
                        [q_pack_tile * N_h * C_h, q_actual],
                        [N_h * C_h, q_pack_tile],
                        [1, C_h],
                    ]
                for head_idx in range(h_actual):
                    h_global = h_start + head_idx
                    nisa.dma_copy(
                        dst=output_flat.ap(
                            pattern=store_pattern,
                            offset=out_base + h_global * C_h,
                        ),
                        src=accum[0:q_actual, :, head_idx, :],
                        dge_mode=nisa.dge_mode.none,
                    )

                # Increment section for next h_tile
                sbm.increment_section()

                # Rotate to the next HBM snake buffer
                snake_hbm_buf_idx = (snake_hbm_buf_idx + 1) % num_snake_hbm_bufs

    sbm.close_scope()

    return output


def _get_tensor_copy_engine(idx: int, modulo: int) -> Any:
    """Alternate tensor_copy between the Scalar and Vector engines for pipelining."""
    return nisa.engine.scalar if idx % modulo == 0 else nisa.engine.vector


def _get_tensor_scalar_engine(idx: int, modulo: int) -> Any:
    """Alternate tensor_scalar between the Scalar and Vector engines for pipelining."""
    return nisa.engine.scalar if idx % modulo == 0 else nisa.engine.vector


@dataclass(frozen=True)
class MSDeformAttnConfig(nl.NKIObject):
    """Configuration for multi-scale deformable attention."""

    # Input dimensions
    B: int  # Batch size
    N_q: int  # Number of queries (global)
    N_h: int  # Number of heads
    C_h: int  # Channels per head
    N_l: int  # Number of levels
    N_p: int  # Number of sampling points per query per head per level
    L: int  # Total flattened spatial dimension (sum of H_i * W_i)

    # Spatial information
    spatial_shapes: tuple  # ((H_0, W_0), (H_1, W_1), ...)
    level_start_index: tuple  # (0, H_0*W_0, H_0*W_0 + H_1*W_1, ...)

    # Layout parameters
    value_layout: str  # "BLNC" or "BNLC"
    sampling_locations_layout: str  # "BQHLP2" or "B2QHLP"
    padding_mode: str  # "zeros" or "border"

    # Hardware constants
    P_MAX: int  # Partition dimension size (128)
    dtype: type  # Data type
    dtype_size: int  # Size of dtype in bytes

    # Sharding parameters
    q_start_global: int  # Global query start index for this shard
    local_N_q: int  # Number of queries for this shard

    # Tiling parameters
    Q_pack: int  # Queries packed per partition (free-dimension query packing)
    Q_tile: int  # Queries per tile
    H_tile: int  # Number of heads per tile
    num_q_tiles: int  # Number of query tiles needed
    num_h_tiles: int  # Number of head tiles needed
    num_C_h_tiles: int  # Number of C_h tiles needed

    # Memory parameters
    h_interleave: int  # Interleave degree for h-scope
    gather_interleave: int  # Interleave degree for gather-scope
    h_scope_mem: int  # H-scope memory usage in bytes
    gather_scope_mem: int  # Gather-scope memory usage in bytes
    total_sbuf: int  # Total available SBUF in bytes

    # Batched indirect gather
    batched_indirect_gather: bool  # Whether batched indirect gather is enabled
    max_indices_per_indirect: int  # Cap on indices (P_MAX*M_batch) per batched dma_transpose
    M_batch: int  # Number of [P_MAX] gather columns batched per dma_transpose (>=1)
    num_gather_bufs: int  # Distinct SBUF gather tiles
    gather_method: str  # "transpose" (dma_transpose + flip) or "copy" (indirect dma_copy)

    # Scalar/Vector engine rotation. One of every `modulo` instructions at a site goes to
    # the Scalar engine and the rest to Vector; GpSimd is excluded so it stays free for
    # indirect DMA descriptor generation. See _get_tensor_copy_engine.
    data_copy_scalar_modulo: int  # PSUM -> SBUF gather result copies
    index_copy_scalar_modulo: int  # uint32 index copies
    scale_scalar_modulo: int  # coordinate/weight tensor_scalar ops


def _calculate_h_scope_memory(
    q_pack: int,
    h_tile: int,
    N_l: int,
    N_p: int,
    C_h: int,
    dtype_size: int,
    padding_mode: str,
    sampling_locations_layout: str,
) -> int:
    """Calculate H-scope memory usage.

    Args:
        q_pack: Queries packed per partition
        h_tile: Number of heads in tile
        N_l: Number of levels
        N_p: Number of sampling points per level
        C_h: Channels per head
        dtype_size: Size of data type in bytes
        padding_mode: "zeros" or "border"
        sampling_locations_layout: "BQHLP2" or "B2QHLP"

    Returns:
        H-scope memory usage in bytes PER PARTITION
    """
    mem = 0
    base = q_pack * h_tile * N_l * N_p

    # Step 1: Load x, y, attn_w
    mem += base * 4  # x (float32)
    mem += base * 4  # y (float32)
    mem += base * dtype_size  # attn_w
    if sampling_locations_layout == "BQHLP2":
        mem += base * 2 * 4  # xy staging buffer (float32, x/y interleaved)

    # Step 3: Bilinear coordinates
    mem += base * 4 * 4  # unclamped, (x0, x1, y0, y1) in one block (int32)
    mem += base * 4 * 4  # clamped, (x0, x1, y0, y1) in one block (int32)
    mem += base * 4 * 4  # frac_terms, (1-dx, dx, 1-dy, dy) (float32)

    # floor scratch: round_trip (float32) and rounded_up (int32), each (q, 2, base)
    mem += 2 * base * 4
    mem += 2 * base * 4

    # Step 5: Bilinear weights (reused in place as the combined weights)
    mem += base * 4 * 4  # combined_weights (float32)

    # Step 6: Flat indices
    mem += base * 4 * 4  # flat_idx_all (uint32)
    mem += 2 * q_pack * h_tile * N_p * 4  # x_term (uint32)

    # Step 4b: Padding mode "zeros" specific
    if padding_mode == "zeros":
        mem += base * 4 * 4  # coord_in_bounds (float32)

    # Step 8: Output accumulator, (q_pack, h_tile, C_h)
    mem += q_pack * h_tile * C_h * 4  # accum (float32)

    return mem


def _calculate_gather_scope_memory(
    q_pack: int,
    h_tile: int,
    N_l: int,
    N_p: int,
    C_h: int,
    dtype_size: int,
    P_MAX: int,
    batched_indirect_gather: bool = False,
    M_batch: int = 1,
    num_gather_bufs: int = 1,
    gather_method: str = "transpose",
) -> int:
    """Calculate gather-scope memory usage.

    Args:
        q_tile: Number of queries in tile
        h_tile: Number of heads in tile
        N_l: Number of levels
        N_p: Number of sampling points per level
        C_h: Channels per head
        dtype_size: Size of data type in bytes
        P_MAX: Partition dimension size
        batched_indirect_gather: Whether the batched indirect gather path is used.
        M_batch: Number of columns batched per indirect DMA.
        num_gather_bufs: Number of SBUF gather tiles (batched path only).

    Returns:
        Gather-scope memory usage in bytes PER PARTITION
    """
    mem = 0

    total_corners = N_p * 4
    num_C_h_tiles = div_ceil(C_h, P_MAX)
    num_cols = q_pack * h_tile * N_l * total_corners

    if batched_indirect_gather:
        # Per gather run, interleaved num_gather_bufs deep:
        #   gathered_run_flat (P_MAX, M_batch * C_h)
        #   gathered_cq       (C_h, num_C_h_tiles, M_batch * P_MAX)
        #   weighted_run      (q_tile, M_batch, C_h) fp32
        per_run = M_batch * C_h * dtype_size
        if gather_method == "transpose":
            per_run += num_C_h_tiles * M_batch * P_MAX * dtype_size
        per_run += M_batch * C_h * 4
        mem += num_gather_bufs * per_run

        # flat_idx_padded, only when a tile does not fill every partition
        mem += num_cols * 4
        # idx_snake, only when the columns are snaked through HBM (M_batch > 1)
        if M_batch > 1:
            mem += num_cols * 4
    else:
        # gathered_all plus one index block
        mem += num_C_h_tiles * total_corners * P_MAX * dtype_size
        mem += total_corners * 4  # uint32
        mem += N_p * 2 * C_h * dtype_size

    return mem


def _build_config(
    value: nl.NkiTensor,
    spatial_shapes: tuple,
    level_start_index: tuple,
    sampling_locations: nl.NkiTensor,
    attention_weights: nl.NkiTensor,
    value_layout: str,
    sampling_locations_layout: str,
    padding_mode: str,
    lnc: int,
    shard_id: int,
    max_indices_per_indirect: Optional[int] = None,
    gather_method: str = "transpose",
) -> MSDeformAttnConfig:
    """Build configuration from input tensors with sharding.

    Args:
        value: Value tensor in either (B, L, N_h, C_h) or (B, N_h, L, C_h) layout
        spatial_shapes: Spatial dimensions for each level
        level_start_index: Start indices for each level
        sampling_locations: Sampling coordinates in (B,N_q,N_h,N_l,N_p,2) or (B,2,N_q,N_h,N_l,N_p)
        attention_weights: Attention weights
        value_layout: "BLNC" for (B, L, N_h, C_h) or "BNLC" for (B, N_h, L, C_h)
        sampling_locations_layout: "BQHLP2" or "B2QHLP"
        padding_mode: "zeros" or "border"
        lnc: Number of NeuronCores for sharding
        shard_id: ID of this shard
        max_indices_per_indirect: Cap on gather indices (P_MAX*M_batch) per batched gather; not None
            enables the batched path, None disables it.

    Returns:
        MSDeformAttnConfig with sharding info
    """
    # Parse input dimensions
    if value_layout == "BLNC":
        B, L, N_h, C_h = value.shape
    else:  # BNLC
        B, N_h, L, C_h = value.shape

    if sampling_locations_layout == "BQHLP2":
        _, N_q, _, N_l, N_p, _ = sampling_locations.shape
    else:  # B2QHLP
        _, _, N_q, _, N_l, N_p = sampling_locations.shape

    # Distribute queries across NeuronCores
    queries_per_nc = div_ceil(N_q, lnc)
    q_start_global = shard_id * queries_per_nc
    q_end_global = min(q_start_global + queries_per_nc, N_q)
    local_N_q = q_end_global - q_start_global

    # Hardware constants
    P_MAX = nl.tile_size.pmax
    dtype = value.dtype
    dtype_size = sizeinbytes(dtype)
    RESERVED_SBUF = 1024
    total_sbuf = nl.tile_size.total_available_sbuf_size - RESERVED_SBUF

    kernel_assert(
        L * N_h * C_h < FP32_EXACT_INT_MAX,
        "ms_deformable_attention: L * N_h * C_h must be < 2**24 so flat value element offsets "
        "stay exact on the fp32 compute datapath",
    )

    H_tile = N_h

    # Prioritize h_interleave=2, then maximize gather_interleave
    h_interleave = 2

    num_h_tiles = div_ceil(N_h, H_tile)
    num_C_h_tiles = div_ceil(C_h, P_MAX)

    kernel_assert(
        gather_method in ("transpose", "copy"),
        f"gather_method must be 'transpose' or 'copy', got {gather_method=}",
    )
    batched_indirect_gather = max_indices_per_indirect != None

    default_max_indices_per_indirect = 128
    if max_indices_per_indirect == None:
        max_indices_per_indirect = default_max_indices_per_indirect

    cols_per_group = N_l * N_p * 4
    M_batch_cap = max(1, max_indices_per_indirect // P_MAX) if batched_indirect_gather else 1
    M_batch = 1

    SBUF_FIT_RESERVE = 8192
    MIN_GATHER_BUFS = 2
    MIN_WITHIN_GROUP_M_BATCH = 8
    q_pack_cap = local_N_q // P_MAX if local_N_q >= P_MAX else 1

    Q_pack = 1
    M_batch = 1
    num_gather_bufs = 1

    for search_pass in range(4):
        allow_spanning = search_pass % 2 == 1
        buf_floor = MIN_GATHER_BUFS if search_pass < 2 else 1
        best_m = 0
        best_bufs = 0
        best_pack = 0

        pack_candidate = 1
        while pack_candidate <= q_pack_cap:
            if True:
                cand_h = _calculate_h_scope_memory(
                    pack_candidate,
                    H_tile,
                    N_l,
                    N_p,
                    C_h,
                    dtype_size,
                    padding_mode,
                    sampling_locations_layout,
                )
                cand_heap = num_h_tiles * pack_candidate * H_tile * N_l * N_p * 4
                cand_total_cols = pack_candidate * H_tile * N_l * N_p * 4
                gather_room = (total_sbuf - cand_heap - SBUF_FIT_RESERVE) // h_interleave - cand_h

                m_candidate = 1
                while m_candidate <= min(M_batch_cap, cand_total_cols):
                    legal = cols_per_group % m_candidate == 0
                    if allow_spanning and m_candidate % cols_per_group == 0:
                        legal = True
                    if legal and cand_total_cols % m_candidate == 0:
                        for buf_candidate in [8, 7, 6, 5, 4, 3, 2, 1]:
                            if buf_candidate < buf_floor:
                                break
                            cand_g = _calculate_gather_scope_memory(
                                pack_candidate,
                                H_tile,
                                N_l,
                                N_p,
                                C_h,
                                dtype_size,
                                P_MAX,
                                batched_indirect_gather,
                                m_candidate,
                                num_gather_bufs=buf_candidate,
                                gather_method=gather_method,
                            )
                            if cand_g <= gather_room:
                                better = m_candidate > best_m
                                if m_candidate == best_m:
                                    better = buf_candidate > best_bufs
                                    if buf_candidate == best_bufs:
                                        better = pack_candidate > best_pack
                                if better:
                                    best_m = m_candidate
                                    best_bufs = buf_candidate
                                    best_pack = pack_candidate
                                break
                    m_candidate = m_candidate + 1
            pack_candidate = pack_candidate * 2

        if best_m > 0 and (allow_spanning or best_m >= MIN_WITHIN_GROUP_M_BATCH):
            Q_pack = best_pack
            M_batch = best_m
            num_gather_bufs = best_bufs
            break

    # Queries per tile, and the partition count a full tile uses
    Q_tile = min(P_MAX * Q_pack, local_N_q) if local_N_q > 0 else P_MAX * Q_pack
    num_q_tiles = div_ceil(local_N_q, Q_tile) if local_N_q > 0 else 0
    total_gather_cols = Q_pack * H_tile * N_l * N_p * 4
    scope_sbuf = total_sbuf - num_h_tiles * Q_pack * H_tile * N_l * N_p * 4

    h_scope_mem = _calculate_h_scope_memory(
        Q_pack, H_tile, N_l, N_p, C_h, dtype_size, padding_mode, sampling_locations_layout
    )
    gather_budget = (scope_sbuf - SBUF_FIT_RESERVE) // h_interleave - h_scope_mem
    gather_scope_mem = _calculate_gather_scope_memory(
        Q_pack,
        H_tile,
        N_l,
        N_p,
        C_h,
        dtype_size,
        P_MAX,
        batched_indirect_gather,
        M_batch,
        num_gather_bufs=num_gather_bufs,
        gather_method=gather_method,
    )

    use_batched_gather = batched_indirect_gather and M_batch >= 1

    if use_batched_gather:
        gather_interleave = 1
    else:
        num_gather_bufs = 1
        gather_interleave = 2
        gather_scope_mem = _calculate_gather_scope_memory(
            Q_pack,
            H_tile,
            N_l,
            N_p,
            C_h,
            dtype_size,
            P_MAX,
            False,
            M_batch,
            num_gather_bufs=1,
            gather_method=gather_method,
        )
        gather_budget = (scope_sbuf - SBUF_FIT_RESERVE) // h_interleave - h_scope_mem
        for candidate_interleave in [8, 7, 6, 5, 4, 3, 2]:
            if gather_scope_mem <= gather_budget // candidate_interleave:
                gather_interleave = candidate_interleave
                break

    default_data_copy_scalar_modulo = 3
    default_index_copy_scalar_modulo = 3
    default_scale_scalar_modulo = 1

    # Log configuration
    logger = get_logger("ms_deformable_attention")
    logger.info(
        "MSDeformAttnConfig: "
        "B=" + str(B) + ", "
        "N_q=" + str(N_q) + " (global), "
        "local_N_q=" + str(local_N_q) + ", "
        "q_start_global=" + str(q_start_global) + ", "
        "N_h=" + str(N_h) + ", "
        "C_h=" + str(C_h) + ", "
        "N_l=" + str(N_l) + ", "
        "N_p=" + str(N_p) + ", "
        "L=" + str(L) + ", "
        "value_layout=" + str(value_layout) + ", "
        "sampling_locations_layout=" + str(sampling_locations_layout) + ", "
        "padding_mode=" + str(padding_mode) + ", "
        "P_MAX=" + str(P_MAX) + ", "
        "dtype=" + str(dtype) + ", "
        "dtype_size=" + str(dtype_size) + ", "
        "Q_pack=" + str(Q_pack) + ", "
        "Q_tile=" + str(Q_tile) + ", "
        "H_tile=" + str(H_tile) + ", "
        "num_q_tiles=" + str(num_q_tiles) + ", "
        "num_h_tiles=" + str(num_h_tiles) + ", "
        "num_C_h_tiles=" + str(num_C_h_tiles) + ", "
        "h_interleave=" + str(h_interleave) + ", "
        "gather_interleave=" + str(gather_interleave) + ", "
        "h_scope_mem=" + str(h_scope_mem) + ", "
        "gather_scope_mem=" + str(gather_scope_mem) + ", "
        "total_sbuf=" + str(total_sbuf) + ", "
        "lnc=" + str(lnc) + ", "
        "batched_indirect_gather=" + str(batched_indirect_gather) + ", "
        "max_indices_per_indirect=" + str(max_indices_per_indirect) + ", "
        "M_batch=" + str(M_batch) + ", "
        "num_gather_bufs=" + str(num_gather_bufs) + ", "
        "use_batched_gather=" + str(use_batched_gather)
    )

    return MSDeformAttnConfig(
        B=B,
        N_q=N_q,
        N_h=N_h,
        C_h=C_h,
        N_l=N_l,
        N_p=N_p,
        L=L,
        spatial_shapes=spatial_shapes,
        level_start_index=level_start_index,
        value_layout=value_layout,
        sampling_locations_layout=sampling_locations_layout,
        padding_mode=padding_mode,
        P_MAX=P_MAX,
        dtype=dtype,
        dtype_size=dtype_size,
        q_start_global=q_start_global,
        local_N_q=local_N_q,
        Q_pack=Q_pack,
        Q_tile=Q_tile,
        H_tile=H_tile,
        num_q_tiles=num_q_tiles,
        num_h_tiles=num_h_tiles,
        num_C_h_tiles=num_C_h_tiles,
        h_interleave=h_interleave,
        gather_interleave=gather_interleave,
        h_scope_mem=h_scope_mem,
        gather_scope_mem=gather_scope_mem,
        total_sbuf=total_sbuf,
        batched_indirect_gather=batched_indirect_gather,
        max_indices_per_indirect=max_indices_per_indirect,
        M_batch=M_batch,
        num_gather_bufs=num_gather_bufs,
        gather_method=gather_method,
        data_copy_scalar_modulo=default_data_copy_scalar_modulo,
        index_copy_scalar_modulo=default_index_copy_scalar_modulo,
        scale_scalar_modulo=default_scale_scalar_modulo,
    )
