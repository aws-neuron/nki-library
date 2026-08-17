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
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager
from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ...core.utils.logging import get_logger


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
) -> Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
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

    Returns:
        grad_value (nl.NkiTensor): Gradient w.r.t. value in HBM, same shape and layout as input value
        grad_sampling_locations (nl.NkiTensor): Gradient w.r.t. sampling_locations in HBM, same shape and layout as input
        grad_attention_weights (nl.NkiTensor): Gradient w.r.t. attention_weights in HBM, shape (B, N_q, N_h, N_l, N_p)

    Notes:
        - Computes actual gradients using bilinear interpolation derivatives
        - Supports both BLNC and BNLC value layouts
        - Supports both BQHLP2 and B2QHLP sampling_locations layouts
        - Padding modes: "zeros" (OOB returns 0) and "border" (clamps to edge)
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
    hl_tile = cfg.hl_tile
    zero_tile_free_dim = cfg.zero_tile_free_dim
    h_interleave = cfg.h_interleave
    gather_interleave = cfg.gather_interleave
    scatter_interleave = cfg.scatter_interleave

    # Initialize SBUF manager
    logger = get_logger("ms_deformable_attention_bwd")
    sbm = SbufManager(0, total_sbuf_req, logger=logger)

    # Allocate output gradients in HBM
    grad_value = nl.ndarray(value.shape, dtype=dtype, buffer=nl.shared_hbm)
    grad_sampling_locations = nl.ndarray(sampling_locations.shape, dtype=nl.float32, buffer=nl.shared_hbm)
    grad_attention_weights = nl.ndarray(attention_weights.shape, dtype=dtype, buffer=nl.shared_hbm)

    # Allocate zero buffer in shared HBM
    if value_layout == "BNLC":
        zero_buffer_hbm = nl.ndarray((N_h, L, C_h), dtype=dtype, buffer=nl.shared_hbm)
    else:  # BLNC
        zero_buffer_hbm = nl.ndarray((L, N_h, C_h), dtype=dtype, buffer=nl.shared_hbm)

    # Calculate intermediate buffers needed
    intermediate_buffers = div_ceil(K, 16) * 2

    # Allocate per-core K-buffers in private HBM
    if value_layout == "BLNC":
        grad_value_buffers = nl.ndarray((K, L, N_h, C_h), dtype=dtype, buffer=nl.private_hbm)
        temp_intermediate = nl.ndarray((intermediate_buffers, L, N_h, C_h), dtype=dtype, buffer=nl.shared_hbm)
    else:  # BNLC
        grad_value_buffers = nl.ndarray((K, N_h, L, C_h), dtype=dtype, buffer=nl.private_hbm)
        temp_intermediate = nl.ndarray((intermediate_buffers, N_h, L, C_h), dtype=dtype, buffer=nl.shared_hbm)

    # ====================================================================
    # Step 1: Zero out a tile in HBM using a memset tile in SBUF
    # ====================================================================
    zero_tile = sbm.alloc_heap((P_MAX, zero_tile_free_dim), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(dst=zero_tile, value=0)

    total_elements = N_h * L * C_h
    zero_buffer_flat = zero_buffer_hbm.reshape((total_elements,))
    ZERO_TILE_SIZE = P_MAX * zero_tile_free_dim

    for offset in range(0, total_elements, ZERO_TILE_SIZE):
        chunk_size = min(ZERO_TILE_SIZE, total_elements - offset)

        if chunk_size == ZERO_TILE_SIZE:
            # Full tile
            for row_idx in range(P_MAX):
                dst_offset = offset + row_idx * zero_tile_free_dim
                nisa.dma_copy(
                    dst=zero_buffer_flat[dst_offset : dst_offset + zero_tile_free_dim],
                    src=zero_tile[row_idx, :],
                )
        else:
            # Partial tile
            num_full_rows = chunk_size // zero_tile_free_dim
            remainder = chunk_size % zero_tile_free_dim

            # Copy full rows
            for row_idx in range(num_full_rows):
                dst_offset = offset + row_idx * zero_tile_free_dim
                nisa.dma_copy(
                    dst=zero_buffer_flat[dst_offset : dst_offset + zero_tile_free_dim],
                    src=zero_tile[row_idx, :],
                )

            # Copy partial last row if needed
            if remainder > 0:
                dst_offset = offset + num_full_rows * zero_tile_free_dim
                src_row_sliced = zero_tile[num_full_rows : num_full_rows + 1, 0:remainder]
                nisa.dma_copy(
                    dst=zero_buffer_flat[dst_offset : dst_offset + remainder],
                    src=src_row_sliced[0, :],
                )

    sbm.pop_heap()

    # ====================================================================
    # Step 2: Construct the K-scaling tensor for use in atomic scatter-add
    # ====================================================================
    k_scale_cols = hl_tile * N_p_hl * 4
    initial_width = min(P_MAX // 4, k_scale_cols)

    iota_temp = sbm.alloc_heap((initial_width, P_MAX), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(
        dst=iota_temp,
        pattern=[[0, P_MAX // K], [1, K]],
        offset=0,
        channel_multiplier=0,
    )

    iota_k_scale = sbm.alloc_heap((P_MAX, k_scale_cols), dtype=nl.uint32, buffer=nl.sbuf)

    # Transpose to first initial_width columns
    for i in range(P_MAX // 32):
        nisa.nc_transpose(
            dst=iota_k_scale[i * 32 : (i + 1) * 32, 0:initial_width],
            data=iota_temp[0:initial_width, i * 32 : (i + 1) * 32],
            engine=nisa.engine.vector,
        )
    sbm.pop_heap()

    # Replicate to fill remaining columns if needed
    if k_scale_cols > initial_width:
        current_width = initial_width
        while current_width < k_scale_cols:
            copy_width = min(current_width, k_scale_cols - current_width)
            nisa.tensor_copy(
                dst=iota_k_scale[:, current_width : current_width + copy_width],
                src=iota_k_scale[:, 0:copy_width],
                engine=nisa.engine.vector,
            )
            current_width += copy_width

    # Scale by K-buffer stride based on layout
    k_buffer_stride = L * N_h * C_h

    nisa.tensor_scalar(dst=iota_k_scale, data=iota_k_scale, op0=nl.multiply, operand0=int(k_buffer_stride))

    # Open H-tile scope
    sbm.open_scope(interleave_degree=h_interleave)

    # Loop over batches
    for batch_idx in range(B):
        # ====================================================================
        # Step 3: Replicate zero HBM buffer to replicated K buffers
        # ====================================================================
        for k in range(K):
            nisa.dma_copy(
                dst=grad_value_buffers[k],
                src=zero_buffer_hbm,
            )

        # Loop over queries
        for q_tile_idx in range(num_q_tiles):
            q_start_local = q_tile_idx * Q_tile
            q_end_local = min(q_start_local + Q_tile, local_N_q)
            q_actual = q_end_local - q_start_local

            # Calculate global query indices for reading from HBM
            q_start = q_start_global + q_start_local
            q_end = q_start_global + q_end_local

            # Loop over head tiles
            for h_tile_idx in range(num_h_tiles):
                h_start = h_tile_idx * H_tile
                h_end = min(h_start + H_tile, N_h)
                h_actual = h_end - h_start

                # ====================================================================
                # Step 4: Load grad_output, x, y, attn_w from HBM
                # ====================================================================
                grad_out = sbm.alloc_stack((q_actual, h_actual, C_h), dtype=dtype, buffer=nl.sbuf)

                for h in range(h_actual):
                    h_global = h_start + h
                    nisa.dma_copy(
                        dst=grad_out[:, h, :],
                        src=grad_output[batch_idx, q_start:q_end, h_global * C_h : (h_global + 1) * C_h],
                    )

                # Load sampling locations (x, y)
                x = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                y = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)

                if cfg.sampling_locations_layout == "BQHLP2":
                    # sampling_locations: (B, N_q, N_h, N_l, N_p, 2)
                    nisa.dma_copy(
                        dst=x,
                        src=sampling_locations[batch_idx, q_start:q_end, h_start:h_end, :, :, 0],
                    )
                    nisa.dma_copy(
                        dst=y,
                        src=sampling_locations[batch_idx, q_start:q_end, h_start:h_end, :, :, 1],
                    )
                else:
                    # sampling_locations: (B, 2, N_q, N_h, N_l, N_p)
                    nisa.dma_copy(
                        dst=x,
                        src=sampling_locations[batch_idx, 0, q_start:q_end, h_start:h_end, :, :],
                    )
                    nisa.dma_copy(
                        dst=y,
                        src=sampling_locations[batch_idx, 1, q_start:q_end, h_start:h_end, :, :],
                    )

                # Load attn_w
                attn_w = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    dst=attn_w,
                    src=attention_weights[batch_idx, q_start:q_end, h_start:h_end, :, :],
                )

                # ====================================================================
                # Step 5: Scale coordinates by spatial dimensions
                # ====================================================================
                for h in range(h_actual):
                    for l in range(N_l):
                        H_l, W_l = spatial_shapes[l]
                        if align_corners:
                            # align_corners=True: [0,1] -> [0, W_l-1]
                            nisa.tensor_scalar(
                                dst=x[:, h, l, :],
                                data=x[:, h, l, :],
                                op0=nl.multiply,
                                operand0=float(W_l - 1),
                                engine=nisa.engine.scalar,
                            )
                            # align_corners=True: [0,1] -> [0, H_l-1]
                            nisa.tensor_scalar(
                                dst=y[:, h, l, :],
                                data=y[:, h, l, :],
                                op0=nl.multiply,
                                operand0=float(H_l - 1),
                                engine=nisa.engine.scalar,
                            )
                        else:
                            # align_corners=False: [0,1] -> [-0.5, W_l-0.5]
                            nisa.tensor_scalar(
                                dst=x[:, h, l, :],
                                data=x[:, h, l, :],
                                op0=nl.multiply,
                                operand0=float(W_l),
                                op1=nl.add,
                                operand1=-0.5,
                                engine=nisa.engine.scalar,
                            )
                            # align_corners=False: [0,1] -> [-0.5, H_l-0.5]
                            nisa.tensor_scalar(
                                dst=y[:, h, l, :],
                                data=y[:, h, l, :],
                                op0=nl.multiply,
                                operand0=float(H_l),
                                op1=nl.add,
                                operand1=-0.5,
                                engine=nisa.engine.scalar,
                            )

                # ====================================================================
                # Step 6: Compute bilinear coordinates
                # ====================================================================
                x0_unclamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                y0_unclamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                x1_unclamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
                y1_unclamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)

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
                x0_clamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
                y0_clamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
                x1_clamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)
                y1_clamped = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=nl.uint32, buffer=nl.sbuf)

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
                    )
                    nisa.tensor_scalar(
                        dst=x1_clamped[:, :, l, :],
                        data=x1_unclamped[:, :, l, :],
                        op0=nl.minimum,
                        operand0=float(W_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                    )

                    # y0, y1
                    nisa.tensor_scalar(
                        dst=y0_clamped[:, :, l, :],
                        data=y0_unclamped[:, :, l, :],
                        op0=nl.minimum,
                        operand0=float(H_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                    )
                    nisa.tensor_scalar(
                        dst=y1_clamped[:, :, l, :],
                        data=y1_unclamped[:, :, l, :],
                        op0=nl.minimum,
                        operand0=float(H_l - 1),
                        op1=nl.maximum,
                        operand1=0.0,
                    )

                # ====================================================================
                # Step 8: Compute fractional contributions
                # ====================================================================
                dx_dy = sbm.alloc_stack((q_actual, 2, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                one_minus_dx_dy = sbm.alloc_stack((q_actual, 2, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                dx_dy_minus_one = sbm.alloc_stack((q_actual, 2, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)

                # Compute dx, dy
                nisa.tensor_tensor(
                    dst=dx_dy[:, 0, :, :, :],
                    data1=x,
                    data2=x0_unclamped,
                    op=nl.subtract,
                )
                nisa.tensor_tensor(
                    dst=dx_dy[:, 1, :, :, :],
                    data1=y,
                    data2=y0_unclamped,
                    op=nl.subtract,
                )

                # Compute 1 - dx, 1 - dy
                dx_dy_2d = dx_dy.reshape((q_actual, 2 * h_actual * N_l * N_p))
                one_minus_dx_dy_2d = one_minus_dx_dy.reshape((q_actual, 2 * h_actual * N_l * N_p))
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
                dx_dy_minus_one_2d = dx_dy_minus_one.reshape((q_actual, 2 * h_actual * N_l * N_p))
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
                bilinear_weights = sbm.alloc_stack((q_actual, 4, h_actual, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=bilinear_weights[0:q_actual, 0, :, :, :],
                    data1=one_minus_dx_dy[0:q_actual, 0, :, :, :],
                    data2=one_minus_dx_dy[0:q_actual, 1, :, :, :],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=bilinear_weights[0:q_actual, 1, :, :, :],
                    data1=one_minus_dx_dy[0:q_actual, 1, :, :, :],
                    data2=dx_dy[0:q_actual, 0, :, :, :],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=bilinear_weights[0:q_actual, 2, :, :, :],
                    data1=dx_dy[0:q_actual, 1, :, :, :],
                    data2=one_minus_dx_dy[0:q_actual, 0, :, :, :],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=bilinear_weights[0:q_actual, 3, :, :, :],
                    data1=dx_dy[0:q_actual, 0, :, :, :],
                    data2=dx_dy[0:q_actual, 1, :, :, :],
                    op=nl.multiply,
                )

                # ====================================================================
                # Step 10: Compute flat indexes
                # ====================================================================
                flat_idx_all = sbm.alloc_stack((q_actual, h_actual, N_l, N_p, 4), dtype=nl.uint32, buffer=nl.sbuf)

                for h in range(h_actual):
                    h_global = h_start + h
                    for l in range(N_l):
                        H_l, W_l = spatial_shapes[l]
                        level_offset = level_start_index[l]

                        # Compute offset based on value_layout
                        # For BLNC: memory order is [L, N_h], so offset = level_offset * N_h + h_global
                        # For BNLC: memory order is [N_h, L], so offset = h_global * L + level_offset
                        if cfg.value_layout == "BLNC":
                            level_and_head_offset = level_offset * N_h + h_global
                        else:  # BNLC
                            level_and_head_offset = h_global * L + level_offset

                        # flat_idx_00
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h, l, :, 0],
                            data=y0_clamped[:, h, l, :],
                            op0=nl.multiply,
                            operand0=int(W_l),
                            engine=nisa.engine.scalar,
                        )
                        nisa.tensor_tensor(
                            dst=flat_idx_all[:, h, l, :, 0],
                            data1=flat_idx_all[:, h, l, :, 0],
                            data2=x0_clamped[:, h, l, :],
                            op=nl.add,
                        )
                        if cfg.value_layout == "BLNC":
                            nisa.tensor_scalar(
                                dst=flat_idx_all[:, h, l, :, 0],
                                data=flat_idx_all[:, h, l, :, 0],
                                op0=nl.multiply,
                                operand0=int(N_h),
                                op1=nl.add,
                                operand1=int(level_and_head_offset),
                                engine=nisa.engine.scalar,
                            )
                        else:  # BNLC
                            nisa.tensor_scalar(
                                dst=flat_idx_all[:, h, l, :, 0],
                                data=flat_idx_all[:, h, l, :, 0],
                                op0=nl.add,
                                operand0=int(level_and_head_offset),
                                engine=nisa.engine.scalar,
                            )

                        # flat_idx_01
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h, l, :, 1],
                            data=y0_clamped[:, h, l, :],
                            op0=nl.multiply,
                            operand0=int(W_l),
                            engine=nisa.engine.scalar,
                        )
                        nisa.tensor_tensor(
                            dst=flat_idx_all[:, h, l, :, 1],
                            data1=flat_idx_all[:, h, l, :, 1],
                            data2=x1_clamped[:, h, l, :],
                            op=nl.add,
                        )
                        if cfg.value_layout == "BLNC":
                            nisa.tensor_scalar(
                                dst=flat_idx_all[:, h, l, :, 1],
                                data=flat_idx_all[:, h, l, :, 1],
                                op0=nl.multiply,
                                operand0=int(N_h),
                                op1=nl.add,
                                operand1=int(level_and_head_offset),
                                engine=nisa.engine.scalar,
                            )
                        else:  # BNLC
                            nisa.tensor_scalar(
                                dst=flat_idx_all[:, h, l, :, 1],
                                data=flat_idx_all[:, h, l, :, 1],
                                op0=nl.add,
                                operand0=int(level_and_head_offset),
                                engine=nisa.engine.scalar,
                            )

                        # flat_idx_10
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h, l, :, 2],
                            data=y1_clamped[:, h, l, :],
                            op0=nl.multiply,
                            operand0=int(W_l),
                            engine=nisa.engine.scalar,
                        )
                        nisa.tensor_tensor(
                            dst=flat_idx_all[:, h, l, :, 2],
                            data1=flat_idx_all[:, h, l, :, 2],
                            data2=x0_clamped[:, h, l, :],
                            op=nl.add,
                        )
                        if cfg.value_layout == "BLNC":
                            nisa.tensor_scalar(
                                dst=flat_idx_all[:, h, l, :, 2],
                                data=flat_idx_all[:, h, l, :, 2],
                                op0=nl.multiply,
                                operand0=int(N_h),
                                op1=nl.add,
                                operand1=int(level_and_head_offset),
                                engine=nisa.engine.scalar,
                            )
                        else:  # BNLC
                            nisa.tensor_scalar(
                                dst=flat_idx_all[:, h, l, :, 2],
                                data=flat_idx_all[:, h, l, :, 2],
                                op0=nl.add,
                                operand0=int(level_and_head_offset),
                                engine=nisa.engine.scalar,
                            )

                        # flat_idx_11
                        nisa.tensor_scalar(
                            dst=flat_idx_all[:, h, l, :, 3],
                            data=y1_clamped[:, h, l, :],
                            op0=nl.multiply,
                            operand0=int(W_l),
                            engine=nisa.engine.scalar,
                        )
                        nisa.tensor_tensor(
                            dst=flat_idx_all[:, h, l, :, 3],
                            data1=flat_idx_all[:, h, l, :, 3],
                            data2=x1_clamped[:, h, l, :],
                            op=nl.add,
                        )
                        if cfg.value_layout == "BLNC":
                            nisa.tensor_scalar(
                                dst=flat_idx_all[:, h, l, :, 3],
                                data=flat_idx_all[:, h, l, :, 3],
                                op0=nl.multiply,
                                operand0=int(N_h),
                                op1=nl.add,
                                operand1=int(level_and_head_offset),
                                engine=nisa.engine.scalar,
                            )
                        else:  # BNLC
                            nisa.tensor_scalar(
                                dst=flat_idx_all[:, h, l, :, 3],
                                data=flat_idx_all[:, h, l, :, 3],
                                op0=nl.add,
                                operand0=int(level_and_head_offset),
                                engine=nisa.engine.scalar,
                            )

                # ====================================================================
                # Step 11: Handle padding_mode="zeros" OOB masking
                # ====================================================================
                if padding_mode == "zeros":
                    # Compute coordinate OOB masks
                    coord_oob = sbm.alloc_stack((q_actual, 4, h_actual, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 0, :, :, :],
                        data1=x0_clamped,
                        data2=x0_unclamped,
                        op=nl.subtract,
                    )
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 1, :, :, :],
                        data1=x1_unclamped,
                        data2=x1_clamped,
                        op=nl.subtract,
                    )
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 2, :, :, :],
                        data1=y0_clamped,
                        data2=y0_unclamped,
                        op=nl.subtract,
                    )
                    nisa.tensor_tensor(
                        dst=coord_oob[:, 3, :, :, :],
                        data1=y1_unclamped,
                        data2=y1_clamped,
                        op=nl.subtract,
                    )

                    # Compute inverse coordinate OOB masks
                    one_minus_coord_oob = sbm.alloc_stack(
                        (q_actual, 4, h_actual, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf
                    )

                    coord_oob_2d = coord_oob.reshape((q_actual, 4 * h_actual * N_l * N_p))
                    one_minus_coord_oob_2d = one_minus_coord_oob.reshape((q_actual, 4 * h_actual * N_l * N_p))
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
                    zeros_mask = sbm.alloc_stack((q_actual, 4, h_actual, N_l, N_p), dtype=nl.int32, buffer=nl.sbuf)
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
                    bilinear_weights_2d = bilinear_weights.reshape((q_actual, 4 * h_actual * N_l * N_p))
                    zeros_mask_2d = zeros_mask.reshape((q_actual, 4 * h_actual * N_l * N_p))
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
                        # Apply shift to flat_idx_00 (only x1_oob matters for x-direction shift)
                        flat_idx_shifted = sbm.alloc_stack(
                            (q_actual, h_actual, N_l, N_p, 2), dtype=nl.uint32, buffer=nl.sbuf
                        )
                        # Left pair: flat_idx_00 - (x1_oob * C_h)
                        # (corners 00 and 10 share the same x0 coordinate, so they collapse together)
                        nisa.tensor_tensor(
                            dst=flat_idx_shifted[:, :, :, :, 0],
                            data1=flat_idx_all[:, :, :, :, 0],  # flat_idx_00
                            data2=coord_oob[:, 1, :, :, :],  # x1_oob
                            op=nl.subtract,
                        )

                        # Right pair: flat_idx_10 - (x1_oob * C_h)
                        # (corners 01 and 11 share the same x1 coordinate, so they collapse together)
                        nisa.tensor_tensor(
                            dst=flat_idx_shifted[:, :, :, :, 1],
                            data1=flat_idx_all[:, :, :, :, 2],  # flat_idx_10
                            data2=coord_oob[:, 1, :, :, :],  # x1_oob
                            op=nl.subtract,
                        )

                        # Scale the shifted indices by C_h
                        flat_idx_shifted_2d = flat_idx_shifted.reshape((q_actual, h_actual * N_l * N_p * 2))
                        nisa.tensor_scalar(
                            dst=flat_idx_shifted_2d,
                            data=flat_idx_shifted_2d,
                            op0=nl.multiply,
                            operand0=int(C_h),
                            engine=nisa.engine.scalar,
                        )

                # Scale all flat indices by C_h
                flat_idx_all_2d = flat_idx_all.reshape((q_actual, h_actual * N_l * N_p * 4))
                nisa.tensor_scalar(
                    dst=flat_idx_all_2d,
                    data=flat_idx_all_2d,
                    op0=nl.multiply,
                    operand0=int(C_h),
                    engine=nisa.engine.scalar,
                )

                # ====================================================================
                # Step 12: Compute combined weights (bilinear weights * attention weights)
                # ====================================================================
                combined_weights = sbm.alloc_stack((q_actual, 4, h_actual, N_l, N_p), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 0, :, :, :],
                    data1=bilinear_weights[0:q_actual, 0, :, :, :],
                    data2=attn_w[0:q_actual, :, :, :],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 1, :, :, :],
                    data1=bilinear_weights[0:q_actual, 1, :, :, :],
                    data2=attn_w[0:q_actual, :, :, :],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 2, :, :, :],
                    data1=bilinear_weights[0:q_actual, 2, :, :, :],
                    data2=attn_w[0:q_actual, :, :, :],
                    op=nl.multiply,
                )
                nisa.tensor_tensor(
                    dst=combined_weights[0:q_actual, 3, :, :, :],
                    data1=bilinear_weights[0:q_actual, 3, :, :, :],
                    data2=attn_w[0:q_actual, :, :, :],
                    op=nl.multiply,
                )

                # ====================================================================
                # Step 13: Compute gradient weight masks for sampling locations
                # ====================================================================

                # Compute gradient weights for ∂bilinear/∂x and ∂bilinear/∂y
                # ∂bilinear/∂x = [-(1-dy), (1-dy), -dy,  dy ]
                # ∂bilinear/∂y = [-(1-dx), -dx,   (1-dx), dx ]
                grad_x_weights = sbm.alloc_stack((q_actual, 4, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                grad_y_weights = sbm.alloc_stack((q_actual, 4, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)

                # corner 0: -(1-dy) = (dy-1)
                nisa.tensor_copy(
                    dst=grad_x_weights[:, 0, :, :, :],
                    src=dx_dy_minus_one[:, 1, :, :, :],
                )
                # corner 1: (1-dy)
                nisa.tensor_copy(
                    dst=grad_x_weights[:, 1, :, :, :],
                    src=one_minus_dx_dy[:, 1, :, :, :],
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
                )

                # corner 0: -(1-dx) = (dx-1)
                nisa.tensor_copy(
                    dst=grad_y_weights[:, 0, :, :, :],
                    src=dx_dy_minus_one[:, 0, :, :, :],
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
                )
                # corner 3: dx
                nisa.tensor_copy(
                    dst=grad_y_weights[:, 3, :, :, :],
                    src=dx_dy[:, 0, :, :, :],
                )

                # ====================================================================
                # Step 13b: Apply OOB masking if padding_mode="zeros"
                # ====================================================================
                if padding_mode == "zeros":
                    grad_x_weights_2d = grad_x_weights.reshape((q_actual, 4 * h_actual * N_l * N_p))
                    grad_y_weights_2d = grad_y_weights.reshape((q_actual, 4 * h_actual * N_l * N_p))

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
                grad_attn_w_local = sbm.alloc_stack((q_actual, h_actual, N_l, N_p), dtype=dtype, buffer=nl.sbuf)
                grad_sampling_loc_local = sbm.alloc_stack(
                    (q_actual, h_actual, N_l, N_p, 2), dtype=nl.float32, buffer=nl.sbuf
                )

                sbm.open_scope(interleave_degree=gather_interleave)

                for h in range(h_actual):
                    for l in range(N_l):
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

                        # Reshape to combine points and corners, then select this head and level
                        flat_idx_all_reshaped = flat_idx_all.reshape((q_actual, h_actual, N_l, N_p * 4))
                        total_corners = N_p * 4
                        num_C_h_chunks = div_ceil(C_h, P_MAX)
                        gathered_all = sbm.alloc_stack(
                            (P_MAX, num_C_h_chunks * total_corners * P_MAX), dtype=dtype, buffer=nl.sbuf, align=32
                        )

                        # Calculate batch offset to add to gather offset
                        batch_offset = batch_idx * L * N_h * C_h

                        # Prepare vector offset tensor
                        vector_offset_tensor_view = None
                        if q_actual < P_MAX:
                            flat_idx_padded = sbm.alloc_stack((P_MAX, total_corners), dtype=nl.uint32, buffer=nl.sbuf)
                            nisa.memset(dst=flat_idx_padded, value=0)
                            nisa.tensor_copy(
                                dst=flat_idx_padded[0:q_actual, :],
                                src=flat_idx_all_reshaped[:, h, l, :],
                            )
                            vector_offset_tensor_view = flat_idx_padded
                        else:
                            vector_offset_tensor_view = flat_idx_all_reshaped[:, h, l, :]

                        # Reshape value for vector_select
                        value_flat_1d = value.reshape((B * L * N_h * C_h,))

                        for c_tile_idx in range(num_C_h_tiles):
                            c_start = c_tile_idx * P_MAX
                            c_end = min(c_start + P_MAX, C_h)
                            c_actual = c_end - c_start

                            # Calculate which C_h chunk and offset within chunk
                            chunk_idx = c_start // P_MAX
                            offset_in_chunk = c_start % P_MAX

                            # Column offset in gathered_all
                            col_start = chunk_idx * total_corners * P_MAX
                            col_end = col_start + total_corners * P_MAX

                            nisa.dma_transpose(
                                dst=gathered_all[offset_in_chunk : offset_in_chunk + c_actual, col_start:col_end],
                                src=value_flat_1d.ap(
                                    pattern=[[c_actual, total_corners * P_MAX], [1, c_actual]],
                                    offset=c_start + batch_offset,
                                    vector_offset=vector_offset_tensor_view,
                                    indirect_dim=0,
                                ),
                                axes=(1, 0),
                            )

                        gathered_reshaped = gathered_all.reshape((P_MAX, num_C_h_chunks, N_p, 4, P_MAX))

                        # Process each point
                        for p in range(N_p):
                            # Compute sampled value
                            sampled_accum = sbm.alloc_stack((q_actual, C_h), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.memset(dst=sampled_accum, value=0.0)

                            # Accumulate gradient weights for sampling locations
                            grad_x_accum = sbm.alloc_stack((q_actual, C_h), dtype=nl.float32, buffer=nl.sbuf)
                            grad_y_accum = sbm.alloc_stack((q_actual, C_h), dtype=nl.float32, buffer=nl.sbuf)
                            nisa.memset(dst=grad_x_accum, value=0.0)
                            nisa.memset(dst=grad_y_accum, value=0.0)

                            for corner in range(4):
                                gathered_corner = nl.ndarray((P_MAX, C_h), dtype=dtype, buffer=nl.psum)

                                for c_tile_idx in range(num_C_h_tiles):
                                    c_start = c_tile_idx * P_MAX
                                    c_end = min(c_start + P_MAX, C_h)
                                    c_actual = c_end - c_start

                                    # Calculate which chunk and offset
                                    chunk_idx = c_start // P_MAX
                                    offset_in_chunk = c_start % P_MAX

                                    nisa.nc_transpose(
                                        dst=gathered_corner[:, c_start:c_end],
                                        data=gathered_reshaped[0:c_actual, chunk_idx, p, corner, :],
                                        engine=nisa.engine.tensor,
                                    )

                                # Accumulate for sampled value
                                nisa.scalar_tensor_tensor(
                                    dst=sampled_accum[0:q_actual, :],
                                    data=gathered_corner[0:q_actual, :],
                                    op0=nl.multiply,
                                    operand0=bilinear_weights[:, corner, h, l, p],
                                    op1=nl.add,
                                    operand1=sampled_accum[0:q_actual, :],
                                )

                                # Accumulate for grad_x: grad_x_accum += grad_x_weight[corner] * value[corner]
                                nisa.scalar_tensor_tensor(
                                    dst=grad_x_accum[0:q_actual, :],
                                    data=gathered_corner[0:q_actual, :],
                                    op0=nl.multiply,
                                    operand0=grad_x_weights[:, corner, h, l, p],
                                    op1=nl.add,
                                    operand1=grad_x_accum[0:q_actual, :],
                                )

                                # Accumulate for grad_y: grad_y_accum += grad_y_weight[corner] * value[corner]
                                nisa.scalar_tensor_tensor(
                                    dst=grad_y_accum[0:q_actual, :],
                                    data=gathered_corner[0:q_actual, :],
                                    op0=nl.multiply,
                                    operand0=grad_y_weights[:, corner, h, l, p],
                                    op1=nl.add,
                                    operand1=grad_y_accum[0:q_actual, :],
                                )

                            # Allocate temp buffers
                            temp = sbm.alloc_stack((q_actual, C_h), dtype=nl.float32, buffer=nl.sbuf)
                            temp_x = sbm.alloc_stack((q_actual, C_h), dtype=nl.float32, buffer=nl.sbuf)
                            temp_y = sbm.alloc_stack((q_actual, C_h), dtype=nl.float32, buffer=nl.sbuf)

                            # Compute grad_attention_weights
                            # temp = grad_out * sampled_accum
                            nisa.tensor_tensor(
                                dst=temp,
                                data1=grad_out[:, h, :],
                                data2=sampled_accum,
                                op=nl.multiply,
                            )

                            # Reduce sum along C_h dimension
                            nisa.tensor_reduce(
                                dst=grad_attn_w_local[:, h, l, p],
                                op=nl.add,
                                data=temp,
                                axis=(1,),
                                keepdims=True,
                            )

                            # Compute grad_sampling_locations
                            # temp_x = grad_out * grad_x_accum
                            nisa.tensor_tensor(
                                dst=temp_x,
                                data1=grad_out[:, h, :],
                                data2=grad_x_accum,
                                op=nl.multiply,
                            )

                            # temp_y = grad_out * grad_y_accum
                            nisa.tensor_tensor(
                                dst=temp_y,
                                data1=grad_out[:, h, :],
                                data2=grad_y_accum,
                                op=nl.multiply,
                            )

                            # Allocate buffers for dot products
                            grad_x_dot = sbm.alloc_stack((q_actual, 1), dtype=nl.float32, buffer=nl.sbuf)
                            grad_y_dot = sbm.alloc_stack((q_actual, 1), dtype=nl.float32, buffer=nl.sbuf)

                            # Reduce sum to get dot products
                            nisa.tensor_reduce(
                                dst=grad_x_dot,
                                op=nl.add,
                                data=temp_x,
                                axis=(1,),
                                keepdims=True,
                            )
                            nisa.tensor_reduce(
                                dst=grad_y_dot,
                                op=nl.add,
                                data=temp_y,
                                axis=(1,),
                                keepdims=True,
                            )

                            # Multiply by scaled attention weights and store
                            # grad_sampling_loc[:, h, l, p, 0] = grad_x_dot * attn_w_scaled_x
                            nisa.tensor_tensor(
                                dst=grad_sampling_loc_local[:, h, l, p, 0],
                                data1=grad_x_dot,
                                data2=attn_w_scaled_x[:, p : p + 1],
                                op=nl.multiply,
                            )

                            # grad_sampling_loc[:, h, l, p, 1] = grad_y_dot * attn_w_scaled_y
                            nisa.tensor_tensor(
                                dst=grad_sampling_loc_local[:, h, l, p, 1],
                                data1=grad_y_dot,
                                data2=attn_w_scaled_y[:, p : p + 1],
                                op=nl.multiply,
                            )

                        sbm.increment_section()

                sbm.close_scope()

                # ====================================================================
                # Step 15: Compute grad_value and scatter to K-buffers
                # ====================================================================

                # Process in batches of (head, level) combinations
                total_hl_combinations = h_actual * N_l
                num_hl_batches = div_ceil(total_hl_combinations, hl_tile)

                sbm.open_scope(interleave_degree=scatter_interleave)

                if cfg.value_layout == "BLNC" or cfg.combined_scatter_corners == False:
                    grad_value_buffers_flat = grad_value_buffers.reshape((K * L * N_h * C_h,))
                    flat_idx_all_corners = flat_idx_all.reshape((q_actual, h_actual, N_l, N_p, 4))

                    for hl_batch_idx in range(num_hl_batches):
                        hl_start = hl_batch_idx * hl_tile
                        hl_end = min(hl_start + hl_tile, total_hl_combinations)
                        hl_actual = hl_end - hl_start

                        # Process points in groups of N_p_hl
                        num_p_groups = div_ceil(N_p, N_p_hl)

                        # Process each point
                        for p_group_idx in range(num_p_groups):
                            p_start = p_group_idx * N_p_hl
                            p_end = min(p_start + N_p_hl, N_p)
                            p_actual = p_end - p_start

                            num_cols = hl_actual * p_actual * 4

                            grad_out_scaled = sbm.alloc_stack((q_actual, num_cols, C_h), dtype=dtype, buffer=nl.sbuf)
                            flat_idx_all_scaled = sbm.alloc_stack((q_actual, num_cols), dtype=nl.uint32, buffer=nl.sbuf)

                            for hl_idx in range(hl_actual):
                                hl_global = hl_start + hl_idx
                                h = hl_global // N_l
                                l = hl_global % N_l

                                # Scale grad_out by combined_weights
                                for p_idx in range(p_actual):
                                    p = p_start + p_idx

                                    for corner in range(4):
                                        col = (hl_idx * p_actual + p_idx) * 4 + corner

                                        nisa.tensor_scalar(
                                            dst=grad_out_scaled[:, col, :],
                                            data=grad_out[:, h, :],
                                            op0=nl.multiply,
                                            operand0=combined_weights[:, corner, h, l, p],
                                        )

                                # Copy in indirect indices
                                col_start = hl_idx * p_actual * 4
                                col_end = col_start + p_actual * 4

                                nisa.tensor_copy(
                                    dst=flat_idx_all_scaled[:, col_start:col_end],
                                    src=flat_idx_all_corners.select(dim=1, index=h)
                                    .select(dim=1, index=l)
                                    .slice(dim=1, start=p_start, end=p_end)
                                    .reshape((q_actual, p_actual * 4)),
                                )

                            # Scale indirect indicies
                            nisa.tensor_tensor(
                                dst=flat_idx_all_scaled[0:q_actual, :],
                                data1=flat_idx_all_scaled[0:q_actual, :],
                                data2=iota_k_scale[0:q_actual, 0:num_cols],
                                op=nl.add,
                            )

                            # Swizzle
                            swizzle_degree = min(P_MAX // K, hl_actual)

                            # Determine effective partition size to avoid DMA abort
                            effective_q = P_MAX if q_actual < P_MAX else q_actual

                            if swizzle_degree == 1:
                                # No swizzling needed
                                if q_actual < P_MAX:
                                    grad_out_swizzled = sbm.alloc_stack(
                                        (P_MAX, num_cols, C_h), dtype=dtype, buffer=nl.sbuf
                                    )
                                    flat_idx_all_swizzled = sbm.alloc_stack(
                                        (P_MAX, num_cols), dtype=nl.uint32, buffer=nl.sbuf
                                    )
                                    nisa.memset(dst=grad_out_swizzled, value=0)
                                    nisa.memset(dst=flat_idx_all_swizzled, value=0)
                                    nisa.tensor_copy(
                                        dst=grad_out_swizzled[0:q_actual, :, :],
                                        src=grad_out_scaled,
                                    )
                                    nisa.tensor_copy(
                                        dst=flat_idx_all_swizzled[0:q_actual, :],
                                        src=flat_idx_all_scaled,
                                    )
                                else:
                                    grad_out_swizzled = grad_out_scaled
                                    flat_idx_all_swizzled = flat_idx_all_scaled
                            else:
                                # Allocate swizzled tensors
                                grad_out_swizzled = sbm.alloc_stack(
                                    (effective_q, num_cols, C_h), dtype=dtype, buffer=nl.sbuf
                                )
                                flat_idx_all_swizzled = sbm.alloc_stack(
                                    (effective_q, num_cols), dtype=nl.uint32, buffer=nl.sbuf
                                )

                                # Initialize padding region with zeros if needed
                                if q_actual < P_MAX:
                                    nisa.memset(dst=grad_out_swizzled, value=0)
                                    nisa.memset(dst=flat_idx_all_swizzled, value=0)

                                if swizzle_degree == 2:
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
                                                partition_offset : partition_offset + partition_height, offset:num_cols
                                            ],
                                            src=flat_idx_all_scaled[
                                                partition_offset : partition_offset + partition_height, 0:remaining
                                            ],
                                        )
                                        nisa.tensor_copy(
                                            dst=grad_out_swizzled[
                                                partition_offset : partition_offset + partition_height,
                                                offset:num_cols,
                                                :,
                                            ],
                                            src=grad_out_scaled[
                                                partition_offset : partition_offset + partition_height, 0:remaining, :
                                            ],
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
                                            )
                                            nisa.tensor_copy(
                                                dst=grad_out_swizzled[
                                                    partition_offset : partition_offset + partition_height, 0:offset, :
                                                ],
                                                src=grad_out_scaled[
                                                    partition_offset : partition_offset + partition_height,
                                                    remaining:num_cols,
                                                    :,
                                                ],
                                            )

                                elif swizzle_degree >= 4:
                                    partition_offsets = [0, 32, 64, 96]

                                    quadrant_copy_degree = swizzle_degree // 4

                                    for quadrant_copy_idx in range(quadrant_copy_degree):
                                        for partition_offset_idx in range(len(partition_offsets)):
                                            partition_offset = partition_offsets[partition_offset_idx]

                                            if partition_offset >= q_actual:
                                                continue

                                            partition_height_raw = 32 - (
                                                quadrant_copy_idx * (32 // quadrant_copy_degree)
                                            )
                                            partition_height = min(partition_height_raw, q_actual - partition_offset)

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
                                                    partition_offset : partition_offset + partition_height, 0:remaining
                                                ],
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
                                                )

                            # Scatter
                            for col in range(num_cols):
                                nisa.dma_compute(
                                    dst=grad_value_buffers_flat.ap(
                                        pattern=[[C_h, effective_q], [1, C_h]],
                                        offset=0,
                                        vector_offset=flat_idx_all_swizzled[0:effective_q, col : col + 1],
                                        indirect_dim=0,
                                    ),
                                    srcs=[
                                        grad_value_buffers_flat.ap(
                                            pattern=[[C_h, effective_q], [1, C_h]],
                                            offset=0,
                                            vector_offset=flat_idx_all_swizzled[0:effective_q, col : col + 1],
                                            indirect_dim=0,
                                        ),
                                        grad_out_swizzled[0:effective_q, col, :],
                                    ],
                                    reduce_op=nl.add,
                                    unique_indices=True,
                                )

                            sbm.increment_section()
                else:
                    grad_value_buffers_flat = grad_value_buffers.reshape((K * N_h * L * C_h,))
                    flat_idx_shifted_reshaped = flat_idx_shifted.reshape((q_actual, h_actual, N_l, N_p, 2))

                    for hl_batch_idx in range(num_hl_batches):
                        hl_start = hl_batch_idx * hl_tile
                        hl_end = min(hl_start + hl_tile, total_hl_combinations)
                        hl_actual = hl_end - hl_start

                        # Process points in groups of N_p_hl
                        num_p_groups = div_ceil(N_p, N_p_hl)

                        # Process each point
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
                                    temp_00 = sbm.alloc_stack((q_actual, C_h), dtype=dtype, buffer=nl.sbuf)
                                    temp_01 = sbm.alloc_stack((q_actual, C_h), dtype=dtype, buffer=nl.sbuf)
                                    temp_10 = sbm.alloc_stack((q_actual, C_h), dtype=dtype, buffer=nl.sbuf)
                                    temp_11 = sbm.alloc_stack((q_actual, C_h), dtype=dtype, buffer=nl.sbuf)

                                    nisa.tensor_scalar(
                                        dst=temp_00,
                                        data=grad_out[:, h, :],
                                        op0=nl.multiply,
                                        operand0=combined_weights[:, 0, h, l, p : p + 1],
                                    )
                                    nisa.tensor_scalar(
                                        dst=temp_01,
                                        data=grad_out[:, h, :],
                                        op0=nl.multiply,
                                        operand0=combined_weights[:, 1, h, l, p : p + 1],
                                    )
                                    nisa.tensor_scalar(
                                        dst=temp_10,
                                        data=grad_out[:, h, :],
                                        op0=nl.multiply,
                                        operand0=combined_weights[:, 2, h, l, p : p + 1],
                                    )
                                    nisa.tensor_scalar(
                                        dst=temp_11,
                                        data=grad_out[:, h, :],
                                        op0=nl.multiply,
                                        operand0=combined_weights[:, 3, h, l, p : p + 1],
                                    )

                                    col_left = (hl_idx * p_actual + p_idx) * 2 + 0
                                    col_right = (hl_idx * p_actual + p_idx) * 2 + 1

                                    # Left col
                                    nisa.tensor_scalar(
                                        dst=grad_out_scaled[:, col_left, 0:C_h],
                                        data=temp_00,
                                        op0=nl.multiply,
                                        operand0=one_minus_coord_oob[:, 1, h, l, p : p + 1],
                                    )
                                    nisa.scalar_tensor_tensor(
                                        dst=grad_out_scaled[:, col_left, 0:C_h],
                                        data=temp_01,
                                        op0=nl.multiply,
                                        operand0=coord_oob[:, 0, h, l, p : p + 1],
                                        op1=nl.add,
                                        operand1=grad_out_scaled[:, col_left, 0:C_h],
                                    )
                                    nisa.tensor_scalar(
                                        dst=grad_out_scaled[:, col_left, C_h : C_h * 2],
                                        data=temp_00,
                                        op0=nl.multiply,
                                        operand0=coord_oob[:, 1, h, l, p : p + 1],
                                    )
                                    nisa.scalar_tensor_tensor(
                                        dst=grad_out_scaled[:, col_left, C_h : C_h * 2],
                                        data=temp_01,
                                        op0=nl.multiply,
                                        operand0=one_minus_coord_oob[:, 0, h, l, p : p + 1],
                                        op1=nl.add,
                                        operand1=grad_out_scaled[:, col_left, C_h : C_h * 2],
                                    )

                                    # Right col
                                    nisa.tensor_scalar(
                                        dst=grad_out_scaled[:, col_right, 0:C_h],
                                        data=temp_10,
                                        op0=nl.multiply,
                                        operand0=one_minus_coord_oob[:, 1, h, l, p : p + 1],
                                    )
                                    nisa.scalar_tensor_tensor(
                                        dst=grad_out_scaled[:, col_right, 0:C_h],
                                        data=temp_11,
                                        op0=nl.multiply,
                                        operand0=coord_oob[:, 0, h, l, p : p + 1],
                                        op1=nl.add,
                                        operand1=grad_out_scaled[:, col_right, 0:C_h],
                                    )
                                    nisa.tensor_scalar(
                                        dst=grad_out_scaled[:, col_right, C_h : C_h * 2],
                                        data=temp_10,
                                        op0=nl.multiply,
                                        operand0=coord_oob[:, 1, h, l, p : p + 1],
                                    )
                                    nisa.scalar_tensor_tensor(
                                        dst=grad_out_scaled[:, col_right, C_h : C_h * 2],
                                        data=temp_11,
                                        op0=nl.multiply,
                                        operand0=one_minus_coord_oob[:, 0, h, l, p : p + 1],
                                        op1=nl.add,
                                        operand1=grad_out_scaled[:, col_right, C_h : C_h * 2],
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
                                )

                            # Scale indices by K-buffer stride
                            nisa.tensor_tensor(
                                dst=flat_idx_scaled[0:q_actual, :],
                                data1=flat_idx_scaled[0:q_actual, :],
                                data2=iota_k_scale[0:q_actual, 0:num_cols],
                                op=nl.add,
                            )

                            # Swizzle
                            swizzle_degree = min(P_MAX // K, hl_actual)

                            if swizzle_degree == 1:
                                # No swizzling needed
                                grad_out_swizzled = grad_out_scaled
                                flat_idx_swizzled = flat_idx_scaled
                            else:
                                # Allocate swizzled tensors
                                grad_out_swizzled = sbm.alloc_stack(
                                    (q_actual, num_cols, C_h * 2), dtype=dtype, buffer=nl.sbuf
                                )
                                flat_idx_swizzled = sbm.alloc_stack(
                                    (q_actual, num_cols), dtype=nl.uint32, buffer=nl.sbuf
                                )

                                if swizzle_degree == 2:
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
                                                partition_offset : partition_offset + partition_height, offset:num_cols
                                            ],
                                            src=flat_idx_scaled[
                                                partition_offset : partition_offset + partition_height, 0:remaining
                                            ],
                                        )
                                        nisa.tensor_copy(
                                            dst=grad_out_swizzled[
                                                partition_offset : partition_offset + partition_height,
                                                offset:num_cols,
                                                :,
                                            ],
                                            src=grad_out_scaled[
                                                partition_offset : partition_offset + partition_height, 0:remaining, :
                                            ],
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
                                            )
                                            nisa.tensor_copy(
                                                dst=grad_out_swizzled[
                                                    partition_offset : partition_offset + partition_height, 0:offset, :
                                                ],
                                                src=grad_out_scaled[
                                                    partition_offset : partition_offset + partition_height,
                                                    remaining:num_cols,
                                                    :,
                                                ],
                                            )

                                elif swizzle_degree >= 4:
                                    partition_offsets = [0, 32, 64, 96]
                                    quadrant_copy_degree = swizzle_degree // 4

                                    for quadrant_copy_idx in range(quadrant_copy_degree):
                                        for partition_offset_idx in range(len(partition_offsets)):
                                            partition_offset = partition_offsets[partition_offset_idx]

                                            if partition_offset >= q_actual:
                                                continue

                                            partition_height_raw = 32 - (
                                                quadrant_copy_idx * (32 // quadrant_copy_degree)
                                            )
                                            partition_height = min(partition_height_raw, q_actual - partition_offset)

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
                                                    partition_offset : partition_offset + partition_height, 0:remaining
                                                ],
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
                                            )

                                            # Second copy, skip if first copy spans entire free dimension
                                            if offset != 0 and remaining < num_cols:
                                                nisa.tensor_copy(
                                                    dst=flat_idx_swizzled[
                                                        partition_offset : partition_offset + partition_height, 0:offset
                                                    ],
                                                    src=flat_idx_scaled[
                                                        partition_offset : partition_offset + partition_height,
                                                        remaining:num_cols,
                                                    ],
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
                                                )

                            # Scatter
                            if q_actual < P_MAX:
                                effective_q = P_MAX
                                flat_idx_padded = sbm.alloc_stack((P_MAX, num_cols), dtype=nl.uint32, buffer=nl.sbuf)
                                grad_out_padded = sbm.alloc_stack(
                                    (P_MAX, num_cols, C_h * 2), dtype=dtype, buffer=nl.sbuf
                                )
                                nisa.memset(dst=flat_idx_padded, value=0)
                                nisa.memset(dst=grad_out_padded, value=0)
                                nisa.tensor_copy(
                                    dst=flat_idx_padded[0:q_actual, :],
                                    src=flat_idx_swizzled[0:q_actual, :],
                                )
                                nisa.tensor_copy(
                                    dst=grad_out_padded[0:q_actual, :, :],
                                    src=grad_out_swizzled[0:q_actual, :, :],
                                )

                                flat_idx_to_use = flat_idx_padded
                                grad_out_to_use = grad_out_padded
                            else:
                                effective_q = q_actual
                                flat_idx_to_use = flat_idx_swizzled
                                grad_out_to_use = grad_out_swizzled

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
                nisa.dma_copy(
                    dst=grad_attention_weights[batch_idx, q_start:q_end, h_start:h_end, :, :],
                    src=grad_attn_w_local,
                )

                # Write grad_sampling_locations
                if cfg.sampling_locations_layout == "BQHLP2":
                    grad_sampling_loc_local_2d = grad_sampling_loc_local.reshape((q_actual, h_actual, N_l, N_p * 2))
                    nisa.dma_copy(
                        dst=grad_sampling_locations.select(dim=0, index=batch_idx)
                        .slice(dim=0, start=q_start, end=q_end, step=1)
                        .slice(dim=1, start=h_start, end=h_end, step=1)
                        .reshape((q_actual, h_actual, N_l, N_p * 2)),
                        src=grad_sampling_loc_local_2d,
                    )
                else:  # B2QHLP
                    nisa.dma_copy(
                        dst=grad_sampling_locations[batch_idx, 0, q_start:q_end, h_start:h_end, :, :],
                        src=grad_sampling_loc_local[:, :, :, :, 0],
                    )
                    nisa.dma_copy(
                        dst=grad_sampling_locations[batch_idx, 1, q_start:q_end, h_start:h_end, :, :],
                        src=grad_sampling_loc_local[:, :, :, :, 1],
                    )

                sbm.increment_section()

        # ====================================================================
        # Step 17: Reduce K buffers for this batch
        # ====================================================================

        # Each core reduces its K buffers to intermediate results
        if K > 1:
            num_groups_round1 = div_ceil(K, 16)

            for group in range(num_groups_round1):
                buffer_views = []
                start_idx = group * 16
                end_idx = min(start_idx + 16, K)

                for k_idx in range(start_idx, end_idx):
                    buffer_views.append(grad_value_buffers[k_idx])

                nisa.dma_compute(
                    dst=temp_intermediate[shard_id * num_groups_round1 + group],
                    srcs=buffer_views,
                    reduce_op=nl.add,
                )
        else:
            # K=1: Just copy the single buffer to temp_intermediate
            nisa.dma_copy(
                dst=temp_intermediate[shard_id],
                src=grad_value_buffers[0],
            )

        # Reduce all intermediate buffers from both cores to final output
        total_intermediates = div_ceil(K, 16) * 2 if K > 1 else 2
        buffer_views = []
        for i in range(total_intermediates):
            buffer_views.append(temp_intermediate[i])

        nisa.dma_compute(
            dst=grad_value[batch_idx],
            srcs=buffer_views,
            reduce_op=nl.add,
        )

    sbm.close_scope()

    # Free iota_k_scale
    sbm.pop_heap()

    return grad_value, grad_sampling_locations, grad_attention_weights


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
    hl_tile: int  # HL tile size (P_MAX // K)
    zero_tile_free_dim: int  # Free dimension for zero buffer initialization
    K_rep: int  # K replication factor
    combined_scatter_corners: bool  # Process adjacent corners together during scatter


def _build_config(
    value: nl.NkiTensor,
    spatial_shapes: tuple,
    level_start_index: tuple,
    sampling_locations: nl.NkiTensor,
    attention_weights: nl.NkiTensor,
    value_layout: str = "BLNC",
    sampling_locations_layout: str = "BQHLP2",
    padding_mode: str = "zeros",
) -> MSDeformAttnBwdConfig:
    """Build config with sharding and tiling information.

    Args:
        value: Value tensor in either (B, L, N_h, C_h) or (B, N_h, L, C_h) layout
        spatial_shapes: Spatial dimensions for each level
        level_start_index: Start indices for each level
        sampling_locations: Sampling coordinates in (B,N_q,N_h,N_l,N_p,2) or (B,2,N_q,N_h,N_l,N_p)
        attention_weights: Attention weights
        value_layout: "BLNC" for (B, L, N_h, C_h) or "BNLC" for (B, N_h, L, C_h)
        sampling_locations_layout: "BQHLP2" or "B2QHLP"
        padding_mode: Padding mode ("zeros" or "border")

    Returns:
        MSDeformAttnBwdConfig
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

    # Hardcode all interleave degrees to 2, except if C_h > 256
    h_interleave = 2
    gather_interleave = 2
    scatter_interleave = 2
    if C_h > 256:
        h_interleave = 1
        gather_interleave = 1
        scatter_interleave = 1

    # Tiling configuration using local_N_q
    Q_tile = min(P_MAX, local_N_q)
    num_q_tiles = div_ceil(local_N_q, Q_tile)

    # Calculate available memory
    RESERVED_SBUF = 1024
    total_sbuf = nl.tile_size.total_available_sbuf_size - RESERVED_SBUF

    # Hardcode H_tile to use all heads
    H_tile = N_h
    num_h_tiles = 1

    # Calculate num_C_h_tiles
    num_C_h_tiles = div_ceil(C_h, P_MAX)

    K = max(1, div_ceil(P_MAX, N_l * N_h))
    N_p_hl = 1  # Number of points per HL group
    hl_tile = P_MAX // K  # HL tile size
    zero_tile_free_dim = 16384  # Free dimension for zero buffer initialization
    K_rep = min(P_MAX, P_MAX // (N_l * N_h))  # K replication factor
    combined_scatter_corners = True  # Process adjacent corners together during scatter

    # Log comprehensive config
    logger = get_logger("ms_deformable_attention_bwd")
    logger.info(
        "Config: B=" + str(B) + ", N_q=" + str(N_q) + ", local_N_q=" + str(local_N_q) + ", "
        "N_h=" + str(N_h) + ", C_h=" + str(C_h) + ", N_l=" + str(N_l) + ", N_p=" + str(N_p) + ", "
        "L=" + str(L) + ", dtype=" + str(value.dtype) + ", value_layout=" + str(value_layout) + ", "
        "sampling_locations_layout=" + str(sampling_locations_layout) + ", "
        "padding_mode=" + str(padding_mode) + ", "
        "spatial_shapes=" + str(spatial_shapes) + ", level_start_index=" + str(level_start_index) + ", "
        "P_MAX=" + str(P_MAX) + ", Q_tile=" + str(Q_tile) + ", H_tile=" + str(H_tile) + ", "
        "num_q_tiles=" + str(num_q_tiles) + ", num_h_tiles=" + str(num_h_tiles) + ", "
        "num_C_h_tiles=" + str(num_C_h_tiles) + ", "
        "lnc=" + str(lnc) + ", shard_id=" + str(shard_id) + ", q_start_global=" + str(q_start_global) + ", "
        "K="
        + str(K)
        + ", N_p_hl="
        + str(N_p_hl)
        + ", hl_tile="
        + str(hl_tile)
        + ", zero_tile_free_dim="
        + str(zero_tile_free_dim)
        + ", combined_scatter_corners="
        + str(combined_scatter_corners)
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
        hl_tile=hl_tile,
        zero_tile_free_dim=zero_tile_free_dim,
        K_rep=K_rep,
        combined_scatter_corners=combined_scatter_corners,
    )
