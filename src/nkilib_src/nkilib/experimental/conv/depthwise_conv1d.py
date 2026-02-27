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

"""Depthwise Conv1D kernel using implicit GEMM approach for TRN2."""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil


@nki.jit(platform_target="trn2")
def depthwise_conv1d_implicit_gemm(
    img_ref: nl.ndarray,
    filter_ref: nl.ndarray,
    padding: tuple = ((0, 0), (0, 0)),
    stride: tuple = (1, 1),
    rhs_dilation: tuple = (1, 1),
    lhs_dilation: tuple = (1, 1),
    feature_group_count: int = 1,
    batch_group_count: int = 1,
    in_perm: tuple = None,
    kern_perm: tuple = None,
    out_perm: tuple = None,
) -> nl.ndarray:
    """
    Depthwise Conv1D using implicit GEMM without full im2col materialization.

    Performs depthwise 1D convolution by loading input with shape [S_TILE, Q] where
    row k contains elements starting at index k (i.e., input[k:k+Q*stride:stride]), enabling implicit
    im2col via offset-based loading. Tiles on S dimension for S > 128. Optimized for
    TRN2 platform with LNC2 sharding on channel dimension. Supports arbitrary stride values.

    TODO: Specify intended usage range (e.g., typical kernel sizes S, input widths W, channel counts C)

    Dimensions:
        N: Batch size
        C: Number of channels
        W: Input width (spatial dimension)
        S: Kernel size
        Q: Output width ((W + W_pad_l + W_pad_r - S) // stride_w + 1)

    Args:
        img_ref (nl.ndarray): [N, C, 1, W], Input tensor on HBM
        filter_ref (nl.ndarray): [C, 1, 1, S], Depthwise kernel weights on HBM
        padding (tuple): Padding as ((H_pad_l, H_pad_r), (W_pad_l, W_pad_r)) (default: ((0,0),(0,0)), zero padding supported)
        stride (tuple): Stride values (stride_h, stride_w) (default: (1, 1), stride_h must be 1, stride_w can be any positive integer)
        rhs_dilation (tuple): RHS dilation (default: (1, 1))
        lhs_dilation (tuple): LHS dilation (default: (1, 1))
        feature_group_count (int): Number of feature groups (default: 1)
        batch_group_count (int): Number of batch groups (default: 1)
        in_perm (tuple): Input permutation (default: None)
        kern_perm (tuple): Kernel permutation (default: None)
        out_perm (tuple): Output permutation (default: None)

    Returns:
        output (nl.ndarray): [N, C, 1, Q], Convolution output on HBM where Q = (W + W_pad_l + W_pad_r - S) // stride_w + 1

    Notes:
        - Only supports stride_h=1, stride_w can be any positive integer
        - Only supports zero padding
        - Requires C to be divisible by NUM_SHARDS (2)
        - Uses LNC2 sharding on channel dimension
        - For stride>1, uses bulk load + strided tensor_copy (efficient DMA + on-chip striding)

    Pseudocode:
        for batch_idx in range(N):
            for c_tile_idx in range(num_c_tiles):
                # Preload and transpose all filter tiles for this channel tile
                for s_tile_idx in range(num_s_tiles):
                    filter_all_tiles[s_tile_idx] = load_and_transpose(filter[c_tile, s_tile])

                for channel_idx in range(c_tile_size):
                    # Load input tiles: input_tile[s, q] = input[s_start + s + q*stride_w]
                    for s_tile_idx in range(num_s_tiles):
                        for q in range(Q):
                            input_tiles[s_tile_idx][s, q] = input[batch_idx, channel_idx, 0, s_start+s+q*stride_w]

                    for q_tile_idx in range(num_q_tiles):
                        # Accumulate across S tiles
                        for s_tile_idx in range(num_s_tiles):
                            output[..., q_start:q_end] += matmul(
                                filter_all_tiles[s_tile_idx][:, channel_idx],
                                input_tiles[s_tile_idx][:, q_start:q_start+Q_TILE]
                            )
    """
    W_padding_l, W_padding_r = padding[1]
    stride_h, stride_w = stride
    kernel_assert(stride_h == 1, f"Only stride_h=1 is supported, got stride_h={stride_h}")
    kernel_assert(
        lhs_dilation[0] == 1 and lhs_dilation[1] == 1, f"Only lhs_dilation=(1,1) is supported, got {lhs_dilation}"
    )
    kernel_assert(
        rhs_dilation[0] == 1 and rhs_dilation[1] == 1, f"Only rhs_dilation=(1,1) is supported, got {rhs_dilation}"
    )
    kernel_assert(batch_group_count == 1, f"Only batch_group_count=1 is supported, got {batch_group_count}")

    N = img_ref.shape[0]
    C = img_ref.shape[1]
    W = img_ref.shape[3]
    S = filter_ref.shape[3]
    W_padded = W + W_padding_l + W_padding_r
    Q = (W_padded - S) // stride_w + 1

    kernel_assert(
        feature_group_count == C,
        f"Only depthwise convolution is supported (feature_group_count must equal C={C}), got {feature_group_count}",
    )

    """
    Calculate tiling parameters for depthwise convolution.

    Tiling Strategy:
    - Input: [N, C, W] tiled as [N, C_TILES, C_TILE] x [S_TILES, S_TILE, Q]
    - Filter: [C, S] tiled as [C_TILES, C_TILE] x [S_TILES, S_TILE]
    - Output: [N, C, Q] accumulated in [Q_TILES, Q_TILE] chunks

    Tile Size Selection:
    - S_TILE = min(S, 128): Matches partition dimension (P_MAX=128)
    - Q_TILE = min(Q, 512): Matches free dimension (F_MAX=512)
    - C_TILE = min(C_per_shard, 128): Balances parallelism and memory

    Optimization Rationale:
    - Filter preloading (outer loop): Amortizes transpose cost across channels
    - Implicit im2col: Avoids materializing full im2col matrix (saves W*S*C memory)
    - Sequential S-tile accumulation: Enables pipelining and reduces PSUM pressure
    """

    P_MAX = nl.tile_size.pmax
    F_MAX = nl.tile_size.psum_fmax
    S_TILE = min(S, P_MAX)
    Q_TILE = min(Q, F_MAX)
    NUM_SHARDS = nl.num_programs()
    C_per_shard = C // NUM_SHARDS
    C_TILE = min(C_per_shard, P_MAX)
    num_s_tiles = div_ceil(S, S_TILE)
    num_q_tiles = div_ceil(Q, Q_TILE)
    num_c_tiles = div_ceil(C_per_shard, C_TILE)

    shard_id = nl.program_id(axis=0)

    output = nl.ndarray((N, C, 1, Q), dtype=img_ref.dtype, buffer=nl.hbm)

    for batch_idx in nl.affine_range(N):
        for c_tile_idx in nl.affine_range(num_c_tiles):
            c_tile_start = c_tile_idx * C_TILE
            c_tile_size = min(C_TILE, C_per_shard - c_tile_start)
            c_global_start = shard_id * C_per_shard + c_tile_start

            # Preload and transpose filter tiles for this channel tile
            filter_all_tiles = []
            for s_tile_idx in nl.affine_range(num_s_tiles):
                s_start = s_tile_idx * S_TILE
                s_tile_size = min(S_TILE, S - s_start)

                filter_tmp = nl.ndarray((C_TILE, S_TILE), dtype=filter_ref.dtype, buffer=nl.sbuf)
                if c_tile_size < C_TILE or s_tile_size < S_TILE:
                    nisa.memset(filter_tmp, 0)
                nisa.dma_copy(
                    dst=filter_tmp[0:c_tile_size, 0:s_tile_size],
                    src=filter_ref.ap(
                        pattern=[[S, c_tile_size], [1, s_tile_size]], offset=s_start + c_global_start * S
                    ),
                )

                filter_psum = nl.ndarray((S_TILE, C_TILE), dtype=filter_ref.dtype, buffer=nl.psum)
                nisa.nc_transpose(
                    dst=filter_psum[0:s_tile_size, 0:c_tile_size], data=filter_tmp[0:c_tile_size, 0:s_tile_size]
                )

                filter_tile = nl.ndarray((S_TILE, C_TILE), dtype=filter_ref.dtype, buffer=nl.sbuf)
                if c_tile_size < C_TILE or s_tile_size < S_TILE:
                    nisa.memset(filter_tile, 0)
                nisa.tensor_copy(
                    dst=filter_tile[0:s_tile_size, 0:c_tile_size], src=filter_psum[0:s_tile_size, 0:c_tile_size]
                )
                filter_all_tiles.append(filter_tile)

            for channel_idx in nl.affine_range(c_tile_size):
                c_global = c_global_start + channel_idx
                input_base_offset = batch_idx * C * W + c_global * W

                """
                Load input tiles using implicit im2col pattern with padding support.
                
                Strategy: Split tiles into three regions:
                1. Start tiles affected by left padding - load row by row
                2. Middle tiles not affected by padding - bulk load
                3. End tiles affected by right padding - load row by row
                
                Tile s_tile_idx is affected by left padding if s_start < W_padding_l.
                Tile s_tile_idx is affected by right padding if s_start + S_TILE - 1 + Q - 1 >= W_padding_l + W.
                """
                input_tiles = []

                # Determine which tiles are affected by padding
                first_unaffected_tile = div_ceil(W_padding_l, S_TILE) if W_padding_l > 0 else 0
                last_unaffected_tile = (W_padding_l + W - Q * stride_w) // S_TILE if W_padding_r > 0 else num_s_tiles

                for s_tile_idx in nl.affine_range(num_s_tiles):
                    s_start = s_tile_idx * S_TILE
                    s_tile_size = min(S_TILE, S - s_start)

                    input_tile = nl.ndarray((S_TILE, Q), dtype=img_ref.dtype, buffer=nl.sbuf)

                    if s_tile_idx < first_unaffected_tile or s_tile_idx >= last_unaffected_tile:
                        """
                        Tile affected by padding - identify contiguous unaffected regions for bulk loading.
                        
                        Strategy:
                        1. Find first/last columns where padding affects the tile
                        2. Load affected columns individually (partial rows)
                        3. Bulk load the middle unaffected region (full rows)
                        """
                        nisa.memset(input_tile, 0)

                        # Identify padding-affected columns
                        q_first_full = None  # First column with full s_tile_size rows
                        q_last_full = None  # Last column with full s_tile_size rows

                        for q_idx in range(Q):
                            q_pos = q_idx * stride_w
                            s_start_valid = max(0, W_padding_l - s_start - q_pos)
                            s_end_valid = min(s_tile_size, W + W_padding_l - s_start - q_pos)

                            if s_start_valid == 0 and s_end_valid == s_tile_size:
                                if q_first_full is None:
                                    q_first_full = q_idx
                                q_last_full = q_idx

                        # Load columns before bulk region (each column may have partial rows)
                        if q_first_full is not None and q_first_full > 0:
                            for q_idx in nl.affine_range(q_first_full):
                                q_pos = q_idx * stride_w
                                s_start_valid = max(0, W_padding_l - s_start - q_pos)
                                s_end_valid = min(s_tile_size, W + W_padding_l - s_start - q_pos)
                                if s_start_valid < s_end_valid:
                                    load_size = s_end_valid - s_start_valid
                                    input_pos = s_start + s_start_valid + q_pos - W_padding_l
                                    nisa.dma_copy(
                                        dst=input_tile[s_start_valid:s_end_valid, q_idx],
                                        src=img_ref.ap(
                                            pattern=[[1, load_size], [1, 1]], offset=input_base_offset + input_pos
                                        ),
                                    )

                        # Bulk load unaffected middle region (full columns with full rows)
                        if q_first_full is not None:
                            q_bulk_count = q_last_full - q_first_full + 1
                            input_offset = input_base_offset + s_start + q_first_full * stride_w - W_padding_l
                            nisa.dma_copy(
                                dst=input_tile[0:s_tile_size, q_first_full : q_last_full + 1],
                                src=img_ref.ap(
                                    pattern=[[1, s_tile_size], [stride_w, q_bulk_count]], offset=input_offset
                                ),
                            )

                        # Load columns after bulk region (each column may have partial rows)
                        if q_last_full is not None and q_last_full < Q - 1:
                            for q_idx in nl.affine_range(q_last_full + 1, Q):
                                q_pos = q_idx * stride_w
                                s_start_valid = max(0, W_padding_l - s_start - q_pos)
                                s_end_valid = min(s_tile_size, W + W_padding_l - s_start - q_pos)
                                if s_start_valid < s_end_valid:
                                    load_size = s_end_valid - s_start_valid
                                    input_pos = s_start + s_start_valid + q_pos - W_padding_l
                                    nisa.dma_copy(
                                        dst=input_tile[s_start_valid:s_end_valid, q_idx],
                                        src=img_ref.ap(
                                            pattern=[[1, load_size], [1, 1]], offset=input_base_offset + input_pos
                                        ),
                                    )

                        # Handle case where no full columns exist (entire tile affected)
                        if q_first_full is None:
                            for q_idx in nl.affine_range(Q):
                                q_pos = q_idx * stride_w
                                s_start_valid = max(0, W_padding_l - s_start - q_pos)
                                s_end_valid = min(s_tile_size, W + W_padding_l - s_start - q_pos)
                                if s_start_valid < s_end_valid:
                                    load_size = s_end_valid - s_start_valid
                                    input_pos = s_start + s_start_valid + q_pos - W_padding_l
                                    nisa.dma_copy(
                                        dst=input_tile[s_start_valid:s_end_valid, q_idx],
                                        src=img_ref.ap(
                                            pattern=[[1, load_size], [1, 1]], offset=input_base_offset + input_pos
                                        ),
                                    )
                    else:
                        """
                        Middle tile not affected by padding - use bulk load with optional strided copy.
                        
                        For stride=1: Direct bulk load to output tile.
                        For stride>1: Bulk load to temporary buffer, then strided tensor_copy to output tile.
                        """
                        if s_tile_size < S_TILE:
                            nisa.memset(input_tile, 0)

                        if stride_w == 1:
                            # Bulk load for stride=1 (optimal path)
                            input_offset = input_base_offset + s_start - W_padding_l
                            nisa.dma_copy(
                                dst=input_tile[0:s_tile_size, 0:Q],
                                src=img_ref.ap(pattern=[[1, s_tile_size], [1, Q]], offset=input_offset),
                            )
                        else:
                            # Bulk load + strided copy for stride>1
                            # Load contiguous chunk: need elements from s_start to s_start + (Q-1)*stride_w
                            bulk_load_size = (Q - 1) * stride_w + 1
                            input_bulk = nl.ndarray((S_TILE, bulk_load_size), dtype=img_ref.dtype, buffer=nl.sbuf)

                            input_offset = input_base_offset + s_start - W_padding_l
                            nisa.dma_copy(
                                dst=input_bulk[0:s_tile_size, 0:bulk_load_size],
                                src=img_ref.ap(pattern=[[1, s_tile_size], [1, bulk_load_size]], offset=input_offset),
                            )

                            # Strided copy: extract every stride_w-th element
                            # Pattern: [[free_dim_size, partition_count], [stride_on_free, num_elements]]
                            # First tuple: step=bulk_load_size (free dim), num=s_tile_size (partitions)
                            # Second tuple: step=stride_w (skip elements), num=Q (output columns)
                            nisa.tensor_copy(
                                dst=input_tile[0:s_tile_size, 0:Q],
                                src=input_bulk.ap(pattern=[[bulk_load_size, s_tile_size], [stride_w, Q]]),
                            )

                    input_tiles.append(input_tile)

                for q_tile_idx in nl.affine_range(num_q_tiles):
                    q_start = q_tile_idx * Q_TILE
                    q_tile_size = min(Q_TILE, Q - q_start)

                    result_psum = nl.ndarray((1, Q_TILE), dtype=nl.float32, buffer=nl.psum)
                    nisa.memset(result_psum, 0)

                    for s_tile_idx in nl.sequential_range(num_s_tiles):
                        s_tile_size = min(S_TILE, S - s_tile_idx * S_TILE)

                        """
                        Matmul computes: output = filter_tile.T @ input_tile where
                        filter_tile = (s_tile_size,1), input_tile = (s_tile_size, q_tile_size)
                        This is the convolution operation for output positions q_start:q_start+Q_TILE
                        """
                        nisa.nc_matmul(
                            dst=result_psum[0:1, 0:q_tile_size],
                            stationary=filter_all_tiles[s_tile_idx][0:s_tile_size, channel_idx : channel_idx + 1],
                            moving=input_tiles[s_tile_idx][0:s_tile_size, q_start : q_start + q_tile_size],
                        )

                    result_sbuf = nl.ndarray((1, Q_TILE), dtype=output.dtype, buffer=nl.sbuf)
                    nisa.tensor_copy(dst=result_sbuf[0:1, 0:q_tile_size], src=result_psum[0:1, 0:q_tile_size])

                    out_offset = batch_idx * C * Q + c_global * Q + q_start
                    nisa.dma_copy(
                        dst=output.ap(pattern=[[1, 1], [1, q_tile_size]], offset=out_offset),
                        src=result_sbuf[0:1, 0:q_tile_size],
                    )

    return output
