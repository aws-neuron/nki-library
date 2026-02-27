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

"""1D Convolution kernels for NeuronCore with replication strategy for efficient tensor engine utilization."""

from dataclasses import dataclass
from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.allocator import SbufManager, sizeinbytes
from ...core.utils.common_types import ActFnType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_nl_act_fn_from_type, get_verified_program_sharding_info
from ...core.utils.logging import get_logger
from ...core.utils.tensor_view import TensorView

# Partition alignment constants for K replication strategy
_PARTITION_STRIDE_32 = 32  # Partition stride when c_in_tile_size <= 32
_PARTITION_STRIDE_64 = 64  # Partition stride when 32 < c_in_tile_size <= 64
_MAX_K_REP_AT_32 = 4  # Max K replication factor when c_in_tile_size <= 32
_MAX_K_REP_AT_64 = 2  # Max K replication factor when 32 < c_in_tile_size <= 64

# PSUM hardware constants
_PSUM_BANK_SIZE = 2048  # Elements per PSUM bank
_NUM_PSUM_BANKS = 8  # Total PSUM banks


@dataclass
class Conv1dConfig(nl.NKIObject):
    """
    Configuration for conv1d kernel parameters.

    Captures user-facing convolution parameters and computed dimension values.
    Used to pass configuration between kernel functions.

    Args:
        stride (int): Convolution stride.
        pad_left (int): Left padding amount.
        pad_right (int): Right padding amount.
        dilation (int): Dilation factor.
        has_bias (bool): Whether bias is present.
        has_activation (bool): Whether activation is applied.
        activation_fn (Optional[ActFnType]): Activation function type.
        lnc_shard (bool): Whether LNC sharding is enabled.
        B (int): Batch size.
        C_in (int): Input channels.
        C_out (int): Output channels.
        L (int): Input sequence length.
        K (int): Kernel size.
        L_out (int): Output sequence length.

    Notes:
        - All dimension values are extracted from input tensor shapes.
        - L_out is computed from L, padding, dilation, stride, and K.
        - Dimensions: B (batch), C_in (input channels), C_out (output channels),
          L (input length), K (kernel size), L_out (output length).
    """

    stride: int
    pad_left: int
    pad_right: int
    dilation: int
    has_bias: bool
    has_activation: bool
    activation_fn: Optional[ActFnType]
    lnc_shard: bool

    # Tensor dimensions
    B: int  # Batch size
    C_in: int  # Input channels
    C_out: int  # Output channels
    L: int  # Input sequence length
    K: int  # Kernel size
    L_out: int  # Output sequence length


@dataclass
class Conv1dTileConfig(nl.NKIObject):
    """
    Configuration for conv1d tiling parameters.

    Captures hardware constants, tile sizes, tile counts, and K replication
    parameters computed from Conv1dConfig.

    Args:
        P_MAX (int): Maximum partition size (128).
        F_MAX (int): Maximum free dimension size (512).
        PSUM_BANK_SIZE (int): Size of each PSUM bank (2048 elements).
        NUM_PSUM_BANKS (int): Number of PSUM banks (8).
        L_tile (int): Output sequence tile size.
        c_out_tile_size_max (int): Maximum C_out tile size.
        c_in_tile_size_max (int): Maximum C_in tile size.
        c_in_tile_count (int): Number of C_in tiles.
        c_out_tile_count (int): Number of C_out tiles.
        K_outer_tile_count (int): Number of K outer tiles.
        K_REP_max (int): Maximum K replication factor.
        partition_stride_max (int): Maximum partition stride.
        stacked_filter_dim_max (int): Maximum stacked filter dimension.
        C_out_start (int): Start of local C_out range.
        C_out_end (int): End of local C_out range.
        C_out_local (int): Local C_out size.

    Notes:
        - K replication parameters depend on C_in tile size for partition alignment.
        - P_MAX (128) is the maximum partition dimension for tensor engine operations.
        - F_MAX (512) is the maximum free dimension for PSUM accumulation.
    """

    # Hardware constants
    P_MAX: int  # Maximum partition size
    F_MAX: int  # Maximum free dimension size
    PSUM_BANK_SIZE: int  # Size of each PSUM bank
    NUM_PSUM_BANKS: int  # Number of PSUM banks

    # Tile sizes
    L_tile: int  # Output sequence tile size
    c_out_tile_size_max: int  # Maximum C_out tile size
    c_in_tile_size_max: int  # Maximum C_in tile size

    # Tile counts
    c_in_tile_count: int  # Number of C_in tiles
    c_out_tile_count: int  # Number of C_out tiles
    K_outer_tile_count: int  # Number of K outer tiles

    # K replication parameters
    K_REP_max: int  # Maximum K replication factor
    partition_stride_max: int  # Maximum partition stride
    stacked_filter_dim_max: int  # Maximum stacked filter dimension

    # LNC sharding
    C_out_start: int  # Start of local C_out range
    C_out_end: int  # End of local C_out range
    C_out_local: int  # Local C_out size


@dataclass
class Conv1dMemoryConfig(nl.NKIObject):
    """
    Configuration for conv1d memory allocation parameters.

    Captures memory sizes, grouping factors, and interleaving parameters
    for efficient SBUF utilization.

    Args:
        dtype_size (int): Size of data type in bytes.
        TOTAL_SBUF (int): Total SBUF size available.
        bias_size_per_tile (int): Bias memory per C_out tile.
        filters_per_c_out_tile (int): Filter memory per C_out tile.
        per_c_out_tile_outer (int): Outer section memory per C_out tile.
        per_c_out_tile_inner (int): Inner section memory per C_out tile.
        per_c_in_tile_inner (int): Memory per C_in tile iteration.
        window_size_max (int): Maximum input window size.
        c_out_group_size (int): Number of C_out tiles per group.
        c_in_interleave (int): C_in double buffering factor (1 or 2).
        l_out_interleave (int): L_out double buffering factor (1 or 2).
        c_out_group_interleave (int): C_out group double buffering factor (1 or 2).

    Notes:
        - Memory allocation follows priority order: c_out_group_size > c_in_interleave > l_out_interleave > c_out_group_interleave.
    """

    dtype_size: int  # Size of data type in bytes
    TOTAL_SBUF: int  # Total SBUF size

    # Per-tile memory sizes
    bias_size_per_tile: int  # Bias memory per C_out tile
    filters_per_c_out_tile: int  # Filter memory per C_out tile
    per_c_out_tile_outer: int  # Outer section memory per C_out tile
    per_c_out_tile_inner: int  # Inner section memory per C_out tile
    per_c_in_tile_inner: int  # Memory per C_in tile iteration
    window_size_max: int  # Maximum input window size

    # Grouping and interleaving
    c_out_group_size: int  # Number of C_out tiles per group
    c_in_interleave: int  # C_in double buffering factor
    l_out_interleave: int  # L_out double buffering factor
    c_out_group_interleave: int  # C_out group double buffering factor


def _build_conv1d_config(
    x_in: nl.ndarray,
    filters: nl.ndarray,
    bias: Optional[nl.ndarray],
    stride: int,
    padding: tuple[int, int],
    dilation: int,
    activation_fn: Optional[ActFnType],
    lnc_shard: bool,
) -> Conv1dConfig:
    """
    Build Conv1dConfig from kernel inputs.

    Args:
        x_in (nl.ndarray): [B, C_in, L], Input tensor.
        filters (nl.ndarray): [K, C_in, C_out], Filter weights.
        bias (Optional[nl.ndarray]): [C_out], Optional bias tensor.
        stride (int): Stride for convolution.
        padding (tuple[int, int]): Tuple of (left_pad, right_pad).
        dilation (int): Dilation factor.
        activation_fn (Optional[ActFnType]): Optional activation function.
        lnc_shard (bool): Whether to enable LNC sharding.

    Returns:
        Conv1dConfig: Configuration object with extracted parameters.
    """
    pad_left, pad_right = padding
    B, C_in, L = x_in.shape
    K, _, C_out = filters.shape
    L_out = (L + pad_left + pad_right - dilation * (K - 1) - 1) // stride + 1

    return Conv1dConfig(
        stride=stride,
        pad_left=pad_left,
        pad_right=pad_right,
        dilation=dilation,
        has_bias=bias != None,
        has_activation=activation_fn != None,
        activation_fn=activation_fn,
        lnc_shard=lnc_shard,
        B=B,
        C_in=C_in,
        C_out=C_out,
        L=L,
        K=K,
        L_out=L_out,
    )


def _build_tile_config(cfg: Conv1dConfig, lnc_shard: bool) -> Conv1dTileConfig:
    """
    Build Conv1dTileConfig from Conv1dConfig.

    Args:
        cfg (Conv1dConfig): Convolution configuration object.
        lnc_shard (bool): Whether to enable LNC sharding.

    Returns:
        Conv1dTileConfig: Tile configuration object.

    Notes:
        Tile Size Justification:
        - P_MAX (128): Maximum partition dimension for tensor engine operations.
          C_in and C_out tiles are bounded by P_MAX to fit in partition dimension.
        - F_MAX (512): Maximum free dimension for PSUM accumulation.
          L_tile is bounded by F_MAX to ensure output fits in PSUM banks.
        - PSUM_BANK_SIZE (2048): Each PSUM bank holds 2048 elements.
          Used for round-robin bank allocation to avoid memory anti dependencies.
        - NUM_PSUM_BANKS (8): Total PSUM banks available.
          c_out_group_size is bounded by this to maximize parallelism.

        K Replication Strategy:
        To efficiently use the tensor engine, we stack multiple K positions
        along the partition dimension. The replication factor depends on
        C_in tile size to maintain partition alignment:
        - c_in_tile_size <= 32: K_REP = min(K, 4), partition_stride = 32
          Allows 4 filter positions stacked in 128 partitions (4 x 32)
        - 32 < c_in_tile_size <= 64: K_REP = min(K, 2), partition_stride = 64
          Allows 2 filter positions stacked in 128 partitions (2 x 64)
        - c_in_tile_size > 64: K_REP = 1, partition_stride = c_in_tile_size
          No stacking, each K position processed separately
    """
    # Hardware constants
    P_MAX = nl.tile_size.pmax  # Max partition dimension
    F_MAX = nl.tile_size.psum_fmax  # Max free dimension for PSUM

    # Determine LNC sharding
    n_prgs, prg_id = 1, 0
    if lnc_shard:
        _, n_prgs, prg_id = get_verified_program_sharding_info("conv1d", (0, 1))

    # Calculate per NC C_out range
    C_out_per_nc = div_ceil(cfg.C_out, n_prgs)
    C_out_start = C_out_per_nc * prg_id
    C_out_end = min(C_out_start + C_out_per_nc, cfg.C_out)
    C_out_local = C_out_end - C_out_start

    # Calculate tile sizes
    L_tile = min(cfg.L_out, F_MAX)
    c_out_tile_size_max = min(P_MAX, C_out_local)
    c_in_tile_size_max = min(P_MAX, cfg.C_in)

    # Calculate K replication parameters
    K_REP_max, partition_stride_max = _get_k_replication_params(c_in_tile_size_max, cfg.K)

    # Calculate tile counts
    K_outer_tile_count = div_ceil(cfg.K, K_REP_max)
    c_in_tile_count = div_ceil(cfg.C_in, P_MAX)
    c_out_tile_count = div_ceil(C_out_local, P_MAX)
    stacked_filter_dim_max = partition_stride_max * K_REP_max

    return Conv1dTileConfig(
        P_MAX=P_MAX,
        F_MAX=F_MAX,
        PSUM_BANK_SIZE=_PSUM_BANK_SIZE,
        NUM_PSUM_BANKS=_NUM_PSUM_BANKS,
        L_tile=L_tile,
        c_out_tile_size_max=c_out_tile_size_max,
        c_in_tile_size_max=c_in_tile_size_max,
        c_in_tile_count=c_in_tile_count,
        c_out_tile_count=c_out_tile_count,
        K_outer_tile_count=K_outer_tile_count,
        K_REP_max=K_REP_max,
        partition_stride_max=partition_stride_max,
        stacked_filter_dim_max=stacked_filter_dim_max,
        C_out_start=C_out_start,
        C_out_end=C_out_end,
        C_out_local=C_out_local,
    )


def _build_memory_config(
    cfg: Conv1dConfig,
    tile_cfg: Conv1dTileConfig,
    dtype_size: int,
) -> Conv1dMemoryConfig:
    """
    Build Conv1dMemoryConfig from Conv1dConfig and Conv1dTileConfig.

    Args:
        cfg (Conv1dConfig): Convolution configuration object.
        tile_cfg (Conv1dTileConfig): Tile configuration object.
        dtype_size (int): Size of data type in bytes.

    Returns:
        Conv1dMemoryConfig: Memory configuration object.

    Notes:
        Memory Allocation Priority:
        1. c_out_group_size: Maximize PSUM bank utilization (up to 8 tiles)
        2. c_in_interleave:  Enable double buffering for C_in tiles (1 or 2)
        3. l_out_interleave: Enable double buffering for L_out tiles (1 or 2)
        4. c_out_group_interleave: Enable double buffering for C_out groups (1 or 2)
    """
    TOTAL_SBUF = nl.tile_size.total_available_sbuf_size

    # Calculate per-tile memory sizes
    bias_size_per_tile = 1 * dtype_size if cfg.has_bias else 0
    filters_per_c_out_tile = (
        tile_cfg.c_in_tile_count * tile_cfg.K_outer_tile_count * tile_cfg.c_out_tile_size_max * dtype_size
    )
    per_c_out_tile_outer = bias_size_per_tile + filters_per_c_out_tile

    # Calculate worst-case input window size
    window_size_max = (tile_cfg.L_tile - 1) * cfg.stride + (cfg.K - 1) * cfg.dilation + 1

    # Memory per C_out tile (result_sbuf)
    per_c_out_tile_inner = tile_cfg.L_tile * dtype_size

    # Memory per C_in tile iteration
    per_c_in_tile_inner = window_size_max * dtype_size + tile_cfg.K_outer_tile_count * tile_cfg.L_tile * dtype_size

    # Total memory per C_out tile
    per_c_out_tile_total = per_c_out_tile_outer + per_c_out_tile_inner

    # Calculate remaining SBUF after reserving space for C_in tile
    remaining_sbuf = TOTAL_SBUF - per_c_in_tile_inner

    # Priority 1: c_out_group_size to maximize PSUM bank utilization
    max_c_out_group_size = min(tile_cfg.NUM_PSUM_BANKS, tile_cfg.c_out_tile_count)
    if per_c_out_tile_total > 0 and remaining_sbuf >= per_c_out_tile_total:
        c_out_group_size = min(max_c_out_group_size, remaining_sbuf // per_c_out_tile_total)
        remaining_sbuf -= c_out_group_size * per_c_out_tile_total
    else:
        c_out_group_size = 1
        remaining_sbuf -= per_c_out_tile_total

    # Calculate memory sizes for the chosen c_out_group_size
    outer_section_size = c_out_group_size * per_c_out_tile_outer
    per_l_out_tile_inner = c_out_group_size * per_c_out_tile_inner

    # Priority 2: c_in_interleave
    if per_c_in_tile_inner > 0 and remaining_sbuf >= per_c_in_tile_inner:
        c_in_interleave = min(2, 1 + remaining_sbuf // per_c_in_tile_inner)
        remaining_sbuf -= (c_in_interleave - 1) * per_c_in_tile_inner
    else:
        c_in_interleave = 1

    # Priority 3: l_out_interleave
    if per_l_out_tile_inner > 0 and remaining_sbuf >= per_l_out_tile_inner:
        l_out_interleave = min(2, 1 + remaining_sbuf // per_l_out_tile_inner)
        remaining_sbuf -= (l_out_interleave - 1) * per_l_out_tile_inner
    else:
        l_out_interleave = 1

    # Priority 4: c_out_group_interleave
    if outer_section_size > 0 and remaining_sbuf >= outer_section_size:
        c_out_group_interleave = min(2, 1 + remaining_sbuf // outer_section_size)
    else:
        c_out_group_interleave = 1

    return Conv1dMemoryConfig(
        dtype_size=dtype_size,
        TOTAL_SBUF=TOTAL_SBUF,
        bias_size_per_tile=bias_size_per_tile,
        filters_per_c_out_tile=filters_per_c_out_tile,
        per_c_out_tile_outer=per_c_out_tile_outer,
        per_c_out_tile_inner=per_c_out_tile_inner,
        per_c_in_tile_inner=per_c_in_tile_inner,
        window_size_max=window_size_max,
        c_out_group_size=c_out_group_size,
        c_in_interleave=c_in_interleave,
        l_out_interleave=l_out_interleave,
        c_out_group_interleave=c_out_group_interleave,
    )


def _get_k_replication_params(c_in_tile_size: int, K: int) -> tuple[int, int]:
    """
    Calculate K replication factor and partition stride based on C_in tile size.

    Args:
        c_in_tile_size (int): Size of the C_in tile.
        K (int): Total kernel size.

    Returns:
        tuple[int, int]: (K_REP, partition_stride) where K_REP is the replication
            factor and partition_stride is the stride between replicated filters.

    Notes:
        - c_in_tile_size <= 32: K_REP = min(K, 4), partition_stride = 32
        - 32 < c_in_tile_size <= 64: K_REP = min(K, 2), partition_stride = 64
        - c_in_tile_size > 64: K_REP = 1, partition_stride = c_in_tile_size
    """
    if c_in_tile_size <= _PARTITION_STRIDE_32:
        return min(K, _MAX_K_REP_AT_32), _PARTITION_STRIDE_32
    elif c_in_tile_size <= _PARTITION_STRIDE_64:
        return min(K, _MAX_K_REP_AT_64), _PARTITION_STRIDE_64
    else:
        return 1, c_in_tile_size


def _get_tensor_copy_engine(idx: int):
    """
    Get tensor copy engine based on index for round-robin alternation.

    Args:
        idx (int): Index for engine selection.

    Returns:
        Engine type: nisa.vector_engine for even indices, nisa.scalar_engine for odd.
    """
    return nisa.vector_engine if idx % 2 == 0 else nisa.scalar_engine


def _validate_conv1d_inputs(
    x_in: nl.ndarray,
    filters: nl.ndarray,
    bias: nl.ndarray,
    stride: int,
    pad_left: int,
    pad_right: int,
    dilation: int,
) -> None:
    """
    Validate all input parameters for conv1d kernel.

    Args:
        x_in (nl.ndarray): [B, C_in, L], Input tensor.
        filters (nl.ndarray): [K, C_in, C_out], Filter weights.
        bias (nl.ndarray): Optional bias tensor.
        stride (int): Stride for convolution.
        pad_left (int): Left padding.
        pad_right (int): Right padding.
        dilation (int): Dilation factor.

    Raises:
        AssertionError: If any validation check fails with descriptive message.

    Notes:
        - All tensors must have matching dtypes.
        - Input channels must match filter channels.
    """
    C_in = x_in.shape[1]
    C_in_filter = filters.shape[1]

    # Parameter constraints
    kernel_assert(dilation >= 1, f"Dilation must be >= 1, got {dilation=}")
    kernel_assert(stride >= 1, f"Stride must be >= 1, got {stride=}")
    kernel_assert(
        pad_left >= 0 and pad_right >= 0,
        f"Padding must be non-negative, got ({pad_left=}, {pad_right=})",
    )

    # Shape validation
    kernel_assert(
        C_in == C_in_filter,
        f"Input channels must match filter channels, got {C_in=} vs {C_in_filter=}",
    )

    # Dtype checks
    kernel_assert(
        x_in.dtype == filters.dtype,
        f"x_in.dtype ({x_in.dtype}) must match filters.dtype ({filters.dtype})",
    )
    if bias != None:
        kernel_assert(
            x_in.dtype == bias.dtype,
            f"x_in.dtype ({x_in.dtype}) must match bias.dtype ({bias.dtype})",
        )


# =============================================================================
# Memory Hierarchy 3: SBUF operations
# =============================================================================


def _scatter_input_to_stacked(
    input_view: Optional[TensorView],
    sbm: SbufManager,
    c_in_start: int,
    c_in_end: int,
    l_start: int,
    l_end: int,
    L: int,
    K: int,
    stride: int,
    dilation: int,
    pad_left: int,
    valid_window_start: int,
    valid_window_end: int,
    name_suffix: str,
    tensor_copy_engine_idx: int,
) -> list[nl.ndarray]:
    """
    Scatter input window to K-replicated stacked format for tensor engine matmul.

    Transforms input data from contiguous layout to a stacked format where multiple
    K positions are stacked along the partition dimension. This enables efficient
    tensor engine utilization by processing multiple filter positions in parallel.

    Args:
        input_view (Optional[TensorView]): [c_in_tile_size, valid_window_size], Input window in SBUF.
        sbm (SbufManager): SBUF memory manager.
        c_in_start (int): Start index of C_in tile.
        c_in_end (int): End index of C_in tile.
        l_start (int): Start index of L_out tile.
        l_end (int): End index of L_out tile.
        L (int): Total input sequence length.
        K (int): Kernel size.
        stride (int): Convolution stride.
        dilation (int): Dilation factor.
        pad_left (int): Left padding amount.
        valid_window_start (int): Start of valid (non-padded) input window.
        valid_window_end (int): End of valid (non-padded) input window.
        name_suffix (str): Prefix for buffer naming.
        tensor_copy_engine_idx (int): Starting index for tensor copy engine alternation.

    Returns:
        list[nl.ndarray]: List of stacked input buffers, one per K outer tile.
            Each buffer has shape [stacked_filter_dim, l_tile_size].

    Notes:
        - Handles padding by initializing buffers to zero when needed.
        - Uses round-robin tensor copy engine alternation for efficiency.
    """
    l_tile_size = l_end - l_start
    c_in_tile_size = c_in_end - c_in_start
    K_REP, partition_stride = _get_k_replication_params(c_in_tile_size, K) if c_in_tile_size > 0 else (1, 1)
    K_outer_tile_count_local = div_ceil(K, K_REP)

    input_stacked_list = []

    # Replication loop over K (outer tiles)
    for k_outer_idx in range(K_outer_tile_count_local):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K)
        k_actual = k_end - k_start
        stacked_filter_dim = partition_stride * k_actual

        # Create buffer for input_stacked
        input_stacked = sbm.alloc_stack(
            shape=(stacked_filter_dim, l_tile_size),
            dtype=input_view.dtype,
            name=f"input_stacked_{name_suffix}_k{k_outer_idx}",
        )

        # Check if this k_outer range needs padding (zeros)
        k_outer_window_start = l_start * stride + k_start * dilation - pad_left
        k_outer_window_end = (l_end - 1) * stride + (k_end - 1) * dilation - pad_left + 1

        # Initialize tile to zero if input padding is needed (k_outer_window_start < 0 or
        # k_outer_window_end > L) or partition padding is needed (partition_stride > c_in_tile_size)
        if k_outer_window_start < 0 or k_outer_window_end > L or partition_stride > c_in_tile_size:
            nisa.memset(dst=input_stacked, value=0.0)

        # Scatter from input_window to input_stacked
        for k_replicate_idx in range(k_actual):
            k_position = k_start + k_replicate_idx
            partition_offset = k_replicate_idx * partition_stride

            base_in_pos = l_start * stride + k_position * dilation - pad_left

            if base_in_pos >= valid_window_start:
                first_valid_out = 0
            else:
                first_valid_out = div_ceil(valid_window_start - base_in_pos, stride)

            if base_in_pos + (l_tile_size - 1) * stride < valid_window_end:
                last_valid_out = l_tile_size
            else:
                last_valid_out = div_ceil(valid_window_end - base_in_pos, stride)

            first_valid_out = max(0, min(first_valid_out, l_tile_size))
            last_valid_out = max(0, min(last_valid_out, l_tile_size))

            if first_valid_out < last_valid_out and input_view != None:
                src_start = base_in_pos + first_valid_out * stride - valid_window_start
                num_elements = last_valid_out - first_valid_out
                src_end = src_start + (num_elements - 1) * stride + 1

                nisa.tensor_copy(
                    dst=TensorView(input_stacked)
                    .slice(dim=0, start=partition_offset, end=partition_offset + c_in_tile_size, step=1)
                    .slice(dim=1, start=first_valid_out, end=last_valid_out, step=1)
                    .get_view(),
                    src=input_view.slice(dim=1, start=src_start, end=src_end, step=stride).get_view(),
                    engine=_get_tensor_copy_engine(tensor_copy_engine_idx + k_outer_idx * K_REP + k_replicate_idx),
                )

        input_stacked_list.append(input_stacked)

    return input_stacked_list


def _conv1d_matmul(
    input_stacked_list: list[nl.ndarray],
    filters_stacked_list: list[list[nl.ndarray]],
    psum_tiles: list[nl.ndarray],
) -> None:
    """
    Perform matrix multiplication between stacked inputs and filters, accumulating to PSUM.

    Args:
        input_stacked_list (list[nl.ndarray]): List of stacked input buffers from scatter operation.
        filters_stacked_list (list[list[nl.ndarray]]): Filter buffers organized as
            [c_out_tile_idx][k_outer_idx].
        psum_tiles (list[nl.ndarray]): PSUM accumulation buffers for each C_out tile.

    Notes:
        - Accumulates results across all K outer tiles.
    """
    num_c_out_tiles = len(psum_tiles)

    for k_outer_idx in range(len(input_stacked_list)):
        input_stacked = input_stacked_list[k_outer_idx]
        for c_out_tile_idx in range(num_c_out_tiles):
            filters_stacked = filters_stacked_list[c_out_tile_idx][k_outer_idx]
            nisa.nc_matmul(
                dst=psum_tiles[c_out_tile_idx],
                stationary=filters_stacked,
                moving=input_stacked,
            )


def _conv1d_cin_tile(
    input_view: Optional[TensorView],
    filters_for_cin: list[list[nl.ndarray]],
    psum_tiles: list[nl.ndarray],
    sbm: SbufManager,
    c_in_start: int,
    c_in_end: int,
    l_start: int,
    l_end: int,
    L: int,
    K: int,
    stride: int,
    dilation: int,
    pad_left: int,
    valid_window_start: int,
    valid_window_end: int,
    name_suffix: str,
    tensor_copy_engine_idx: int,
) -> None:
    """
    Process a single C_in tile for conv1d computation.

    Scatters input to K-replicated stacked format and performs matrix multiplication
    to accumulate partial results in PSUM.

    Args:
        input_view (Optional[TensorView]): [c_in_tile_size, valid_window_size], Input window in SBUF.
        filters_for_cin (list[list[nl.ndarray]]): Filter buffers for this C_in tile,
            organized as [c_out_tile_idx][k_outer_idx].
        psum_tiles (list[nl.ndarray]): PSUM accumulation buffers for each C_out tile.
        sbm (SbufManager): SBUF memory manager.
        c_in_start (int): Start index of C_in tile.
        c_in_end (int): End index of C_in tile.
        l_start (int): Start index of L_out tile.
        l_end (int): End index of L_out tile.
        L (int): Total input sequence length.
        K (int): Kernel size.
        stride (int): Convolution stride.
        dilation (int): Dilation factor.
        pad_left (int): Left padding amount.
        valid_window_start (int): Start of valid (non-padded) input window.
        valid_window_end (int): End of valid (non-padded) input window.
        name_suffix (str): Prefix for buffer naming.
        tensor_copy_engine_idx (int): Starting index for tensor copy engine alternation.

    Notes:
        - Accumulates results to existing PSUM values (does not initialize).
    """
    # Scatter input to K-replicated stacked format
    input_stacked_list = _scatter_input_to_stacked(
        input_view=input_view,
        sbm=sbm,
        c_in_start=c_in_start,
        c_in_end=c_in_end,
        l_start=l_start,
        l_end=l_end,
        L=L,
        K=K,
        stride=stride,
        dilation=dilation,
        pad_left=pad_left,
        valid_window_start=valid_window_start,
        valid_window_end=valid_window_end,
        name_suffix=name_suffix,
        tensor_copy_engine_idx=tensor_copy_engine_idx,
    )

    # Matmul accumulation
    _conv1d_matmul(
        input_stacked_list=input_stacked_list,
        filters_stacked_list=filters_for_cin,
        psum_tiles=psum_tiles,
    )


# =============================================================================
# Memory Hierarchy 2: HBM <-> SBUF interleaved with SBUF operations
# =============================================================================


def _apply_bias_activation_and_copy(
    psum_tiles: list[nl.ndarray],
    result_sbufs: list[nl.ndarray],
    bias_sbufs: list[Optional[nl.ndarray]],
    has_bias: bool,
    has_activation: bool,
    activation_fn: Optional[ActFnType],
) -> None:
    """
    Apply optional bias and activation, then copy from PSUM to SBUF.

    Args:
        psum_tiles (list[nl.ndarray]): PSUM accumulation buffers for each C_out tile.
        result_sbufs (list[nl.ndarray]): Destination SBUF buffers for each C_out tile.
        bias_sbufs (list[Optional[nl.ndarray]]): Bias buffers for each C_out tile (may be None).
        has_bias (bool): Whether to apply bias addition.
        has_activation (bool): Whether to apply activation function.
        activation_fn (Optional[ActFnType]): Activation function type to apply.

    Notes:
        - Fuses bias addition or activation with PSUM-to-SBUF copy when possible.
    """
    num_c_out_tiles = len(result_sbufs)

    for c_out_tile_idx in range(num_c_out_tiles):
        psum_tile = psum_tiles[c_out_tile_idx]
        result_sbuf = result_sbufs[c_out_tile_idx]
        bias_sbuf = bias_sbufs[c_out_tile_idx]

        # Copy output tile from PSUM to SBUF, with optional fused bias and activation
        if has_bias and has_activation:
            nisa.tensor_scalar(
                dst=result_sbuf,
                data=psum_tile,
                op0=nl.add,
                operand0=TensorView(bias_sbuf).get_view(),
            )
            nisa.activation(
                dst=result_sbuf,
                data=result_sbuf,
                op=get_nl_act_fn_from_type(activation_fn),
            )
        elif has_bias:
            nisa.tensor_scalar(
                dst=result_sbuf,
                data=psum_tile,
                op0=nl.add,
                operand0=TensorView(bias_sbuf).get_view(),
            )
        elif has_activation:
            nisa.activation(
                dst=result_sbuf,
                data=psum_tile,
                op=get_nl_act_fn_from_type(activation_fn),
            )
        else:
            nisa.tensor_copy(dst=result_sbuf, src=psum_tile)


def _load_input_window_to_sbuf(
    x_in_view: TensorView,
    sbm: SbufManager,
    l_start: int,
    l_end: int,
    L: int,
    K: int,
    stride: int,
    dilation: int,
    pad_left: int,
    name: str,
) -> tuple[Optional[TensorView], int, int]:
    """
    Load input window from HBM to SBUF for a given L_out tile.

    Args:
        x_in_view (TensorView): [c_in_tile_size, L], Input tensor view on HBM.
        sbm (SbufManager): SBUF memory manager.
        l_start (int): Start index of L_out tile.
        l_end (int): End index of L_out tile.
        L (int): Total input sequence length.
        K (int): Kernel size.
        stride (int): Convolution stride.
        dilation (int): Dilation factor.
        pad_left (int): Left padding amount.
        name (str): Name for the allocated buffer.

    Returns:
        tuple containing:
            - input_view_sbuf (Optional[TensorView]): Input window in SBUF, or None if fully padded.
            - valid_window_start (int): Start of valid (non-padded) input window.
            - valid_window_end (int): End of valid (non-padded) input window.

    Notes:
        - Returns None for input_view_sbuf if the entire window is in padding region.
    """
    c_in_tile_size = x_in_view.shape[0]
    full_window_start = l_start * stride - pad_left
    full_window_end = (l_end - 1) * stride + (K - 1) * dilation - pad_left + 1
    valid_window_start = max(0, full_window_start)
    valid_window_end = min(L, full_window_end)

    input_view_sbuf = None
    if valid_window_start < valid_window_end:
        valid_window_size = valid_window_end - valid_window_start
        input_window = sbm.alloc_stack(shape=(c_in_tile_size, valid_window_size), dtype=x_in_view.dtype, name=name)
        nisa.dma_copy(
            dst=input_window,
            src=x_in_view.slice(dim=1, start=valid_window_start, end=valid_window_end, step=1).get_view(),
        )
        input_view_sbuf = TensorView(input_window)

    return input_view_sbuf, valid_window_start, valid_window_end


def _conv1d_lout_tile(
    x_in_view: TensorView,
    filters_cache: list[list[list[nl.ndarray]]],
    bias_cache: list[Optional[nl.ndarray]],
    result_sbufs: list[nl.ndarray],
    sbm: SbufManager,
    cfg: Conv1dConfig,
    tile_cfg: Conv1dTileConfig,
    mem_cfg: Conv1dMemoryConfig,
    c_out_tile_sizes: list[int],
    l_start: int,
    l_end: int,
    psum_bank_idx: int,
    name_suffix: str,
) -> None:
    """
    Process a single L_out tile for conv1d computation.

    Orchestrates the full computation for one output sequence tile: allocates PSUM,
    iterates over C_in tiles, performs matmul accumulation, and applies bias/activation.

    Args:
        x_in_view (TensorView): [C_in, L], Input tensor view for current batch on HBM.
        filters_cache (list[list[list[nl.ndarray]]]): Pre-loaded filter buffers organized as
            [c_out_tile_idx][c_in_tile_idx][k_outer_idx].
        bias_cache (list[Optional[nl.ndarray]]): Pre-loaded bias buffers for each C_out tile.
        result_sbufs (list[nl.ndarray]): Destination SBUF buffers for each C_out tile.
        sbm (SbufManager): SBUF memory manager.
        cfg (Conv1dConfig): Convolution configuration.
        tile_cfg (Conv1dTileConfig): Tile configuration.
        mem_cfg (Conv1dMemoryConfig): Memory configuration.
        c_out_tile_sizes (list[int]): Size of each C_out tile.
        l_start (int): Start index of L_out tile.
        l_end (int): End index of L_out tile.
        psum_bank_idx (int): Starting PSUM bank index for round-robin allocation.
        name_suffix (str): Prefix for buffer naming.

    Notes:
        - Uses round-robin PSUM bank allocation to avoid memory anti dependencies.
        - Applies C_in interleaving for double buffering when memory permits.
    """
    num_c_out_tiles = len(result_sbufs)
    l_tile_size = l_end - l_start

    # Allocate PSUM buffers for accumulation
    psum_tiles = []
    for c_out_tile_idx in range(num_c_out_tiles):
        psum_tile = nl.ndarray(
            shape=(c_out_tile_sizes[c_out_tile_idx], l_tile_size),
            dtype=nl.float32,
            buffer=nl.psum,
            address=(0, (psum_bank_idx + c_out_tile_idx) % _NUM_PSUM_BANKS * _PSUM_BANK_SIZE),
        )
        psum_tiles.append(psum_tile)

    # Open scope for C_in tiles with interleaving
    sbm.open_scope(interleave_degree=mem_cfg.c_in_interleave, name=f"c_in_{name_suffix}")

    # Process all C_in tiles
    c_in_tile_idx = 0
    for c_in_start in range(0, cfg.C_in, tile_cfg.P_MAX):
        c_in_end = min(c_in_start + tile_cfg.P_MAX, cfg.C_in)

        # Load input window from HBM
        x_in_cin_view = x_in_view.slice(dim=0, start=c_in_start, end=c_in_end, step=1)
        input_window, valid_window_start, valid_window_end = _load_input_window_to_sbuf(
            x_in_cin_view,
            sbm,
            l_start,
            l_end,
            cfg.L,
            cfg.K,
            cfg.stride,
            cfg.dilation,
            cfg.pad_left,
            f"{name_suffix}_cin{c_in_start}",
        )

        # Get filters for this C_in tile
        filters_for_cin = []
        for c_out_tile_idx in range(num_c_out_tiles):
            filters_for_cin.append(filters_cache[c_out_tile_idx][c_in_tile_idx])

        # Process this C_in tile
        _conv1d_cin_tile(
            input_view=input_window,
            filters_for_cin=filters_for_cin,
            psum_tiles=psum_tiles,
            sbm=sbm,
            c_in_start=c_in_start,
            c_in_end=c_in_end,
            l_start=l_start,
            l_end=l_end,
            L=cfg.L,
            K=cfg.K,
            stride=cfg.stride,
            dilation=cfg.dilation,
            pad_left=cfg.pad_left,
            valid_window_start=valid_window_start,
            valid_window_end=valid_window_end,
            name_suffix=f"{name_suffix}_cin{c_in_tile_idx}",
            tensor_copy_engine_idx=c_in_tile_idx * cfg.K,
        )

        # Increment section to next C_in tile
        sbm.increment_section()
        c_in_tile_idx += 1

    # Close C_in scope
    sbm.close_scope()

    # Apply bias/activation and copy to result SBUF
    _apply_bias_activation_and_copy(
        psum_tiles=psum_tiles,
        result_sbufs=result_sbufs,
        bias_sbufs=bias_cache,
        has_bias=cfg.has_bias,
        has_activation=cfg.has_activation,
        activation_fn=cfg.activation_fn,
    )


# =============================================================================
# Memory Hierarchy 1: HBM <-> SBUF
# =============================================================================


def _load_bias_to_sbuf(
    bias: TensorView,
    sbm: SbufManager,
    name: str,
) -> nl.ndarray:
    """
    Load bias from HBM to SBUF.

    Args:
        bias (TensorView): [c_out_tile_size], Bias tensor view on HBM.
        sbm (SbufManager): SBUF memory manager.
        name (str): Name for the allocated buffer.

    Returns:
        nl.ndarray: [c_out_tile_size, 1], Bias buffer in SBUF.

    Notes:
        - Bias is always loaded as fp32 because tensor_scalar requires fp32 operands
          when the operand is a tile.
    """
    c_out_tile_size = bias.shape[0]
    bias_sbuf = sbm.alloc_stack(shape=(c_out_tile_size, 1), dtype=nl.float32, name=name)
    nisa.dma_copy(dst=bias_sbuf[:, 0], src=bias.get_view())
    return bias_sbuf


def _load_filters_to_sbuf(
    filters: TensorView,
    sbm: SbufManager,
    K: int,
    name_suffix: str,
) -> list[nl.ndarray]:
    """
    Load filters from HBM to SBUF with K-replication stacking.

    Args:
        filters (TensorView): [K, c_in_tile_size, c_out_tile_size], Filter tensor view on HBM.
        sbm (SbufManager): SBUF memory manager.
        K (int): Total kernel size.
        name_suffix (str): Prefix for buffer naming.

    Returns:
        list[nl.ndarray]: List of stacked filter buffers, one per K outer tile.
            Each buffer has shape [stacked_filter_dim, c_out_tile_size].

    Notes:
        - Stacks multiple K positions along partition dimension for efficient tensor engine use.
        - Initializes buffers to zero when partition padding is needed.
    """
    c_in_tile_size = filters.shape[1]
    c_out_tile_size = filters.shape[2]
    K_REP, partition_stride = _get_k_replication_params(c_in_tile_size, K)
    K_outer_tile_count_local = div_ceil(K, K_REP)

    filters_stacked_list = []
    for k_outer_idx in range(K_outer_tile_count_local):
        k_start = k_outer_idx * K_REP
        k_end = min(k_start + K_REP, K)
        k_actual = k_end - k_start
        stacked_filter_dim = partition_stride * k_actual

        filters_stacked = sbm.alloc_stack(
            shape=(stacked_filter_dim, c_out_tile_size),
            dtype=filters.dtype,
            name=f"filters_stacked_{name_suffix}_k{k_outer_idx}",
        )

        if partition_stride > c_in_tile_size:
            nisa.memset(dst=filters_stacked, value=0.0)

        for k_replicate_idx in range(k_actual):
            k_position = k_start + k_replicate_idx
            partition_offset = k_replicate_idx * partition_stride

            nisa.dma_copy(
                dst=TensorView(filters_stacked)
                .slice(dim=0, start=partition_offset, end=partition_offset + c_in_tile_size, step=1)
                .get_view(),
                src=filters.select(dim=0, index=k_position).get_view(),
            )

        filters_stacked_list.append(filters_stacked)

    return filters_stacked_list


def _store_result_to_hbm(
    y_out_view: TensorView,
    result_sbuf: nl.ndarray,
) -> None:
    """
    Store result from SBUF to HBM.

    Args:
        y_out_view (TensorView): [c_out_tile_size, l_tile_size], Output tensor view on HBM.
        result_sbuf (nl.ndarray): [c_out_tile_size, l_tile_size], Result buffer in SBUF.
    """
    nisa.dma_copy(dst=y_out_view.get_view(), src=result_sbuf)


def _load_bias_and_filters_for_c_out_group(
    filters: nl.ndarray,
    bias: Optional[nl.ndarray],
    sbm: SbufManager,
    cfg: Conv1dConfig,
    tile_cfg: Conv1dTileConfig,
    c_out_group_start: int,
    c_out_group_end: int,
) -> tuple[list[int], list[Optional[nl.ndarray]], list[list[list[nl.ndarray]]]]:
    """
    Load bias and filters for all C_out tiles in a C_out group.

    Args:
        filters (nl.ndarray): [K, C_in, C_out], Filter weights on HBM.
        bias (Optional[nl.ndarray]): [C_out], Optional bias tensor on HBM.
        sbm (SbufManager): SBUF memory manager.
        cfg (Conv1dConfig): Convolution configuration.
        tile_cfg (Conv1dTileConfig): Tile configuration.
        c_out_group_start (int): Start index of C_out group.
        c_out_group_end (int): End index of C_out group.

    Returns:
        tuple containing:
            - c_out_tile_sizes (list[int]): Size of each C_out tile.
            - bias_cache (list[Optional[nl.ndarray]]): Bias buffers for each C_out tile.
            - filters_cache (list[list[list[nl.ndarray]]]): Filter buffers organized as
              [c_out_tile_idx][c_in_tile_idx][k_outer_idx].
    """
    actual_group_size = div_ceil(c_out_group_end - c_out_group_start, tile_cfg.P_MAX)

    c_out_tile_sizes = []
    bias_cache = []
    filters_cache = []

    for c_out_tile_idx in range(actual_group_size):
        c_out_tile_start = c_out_group_start + c_out_tile_idx * tile_cfg.P_MAX
        c_out_tile_end = min(c_out_tile_start + tile_cfg.P_MAX, tile_cfg.C_out_end)
        c_out_tile_size = c_out_tile_end - c_out_tile_start
        c_out_tile_sizes.append(c_out_tile_size)

        # Load bias
        bias_sbuf = None
        if cfg.has_bias:
            bias_view = TensorView(bias).slice(dim=0, start=c_out_tile_start, end=c_out_tile_end, step=1)
            bias_sbuf = _load_bias_to_sbuf(bias_view, sbm, f"bias_c{c_out_tile_start}")
        bias_cache.append(bias_sbuf)

        # Load filters for all C_in tiles
        filters_cache_tile = []
        for c_in_tile_idx in range(tile_cfg.c_in_tile_count):
            c_in_start = c_in_tile_idx * tile_cfg.P_MAX
            c_in_end = min(c_in_start + tile_cfg.P_MAX, cfg.C_in)
            filters_view = (
                TensorView(filters)
                .slice(dim=1, start=c_in_start, end=c_in_end, step=1)
                .slice(dim=2, start=c_out_tile_start, end=c_out_tile_end, step=1)
            )
            filters_stacked_list = _load_filters_to_sbuf(
                filters_view,
                sbm,
                cfg.K,
                f"filters_c{c_out_tile_start}_cin{c_in_start}",
            )
            filters_cache_tile.append(filters_stacked_list)
        filters_cache.append(filters_cache_tile)

    return c_out_tile_sizes, bias_cache, filters_cache


@nki.jit
def conv1d(
    x_in: nl.ndarray,
    filters: nl.ndarray,
    bias: Optional[nl.ndarray] = None,
    stride: int = 1,
    padding: tuple[int, int] = (0, 0),
    dilation: int = 1,
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
) -> nl.ndarray:
    """
    1D Convolution operation using tensor engine with replication strategy.

    Applies 1D convolution filters across the input sequence dimension. Supports
    stride, padding, dilation, optional bias addition, and activation function fusion.
    The kernel uses a replication strategy to efficiently utilize the tensor engine
    by stacking multiple filter positions along the partition dimension.

    Intended Usage Range:
        - Kernel size (K): 1 to 128
        - Sequence length (L): 1 to 4096
        - Input channels (C_in): 1 to 4096
        - Output channels (C_out): 1 to 4096
        - Batch size (B): Any positive integer

    Dimensions:
        B: Batch size
        C_in: Number of input channels
        C_out: Number of output channels
        L: Input sequence length
        L_out: Output sequence length = (L + pad_left + pad_right - dilation * (K - 1) - 1) // stride + 1
        K: Kernel/filter size

    Args:
        x_in (nl.ndarray): [B, C_in, L], Input tensor on HBM.
        filters (nl.ndarray): [K, C_in, C_out], Convolution filter weights on HBM.
        bias (Optional[nl.ndarray]): [C_out], Optional bias tensor on HBM. Default None.
        stride (int): Stride for convolution. Must be >= 1. Default 1.
        padding (tuple[int, int]): Tuple of (left_pad, right_pad). Must be non-negative. Default (0, 0).
        dilation (int): Dilation factor for dilated convolution. Must be >= 1. Default 1.
        activation_fn (Optional[ActFnType]): Optional activation function to fuse. Default None.
        lnc_shard (bool): If True, shard computation across LNC cores on C_out dimension. Default False.

    Returns:
        y_out (nl.ndarray): [B, C_out, L_out], Output tensor on HBM where
            L_out = (L + pad_left + pad_right - dilation * (K - 1) - 1) // stride + 1

    Notes:
        - All input tensors (x_in, filters, bias) must have the same dtype
        - Input channels C_in must match filter channels
        - Uses replication strategy to stack K filter positions along partition dimension
        - Partition alignment rules limit K replication factor based on C_in tile size
        - Memory management uses SbufManager with multi-buffering for efficiency

    Pseudocode:
        y_out = zeros(B, C_out, L_out)
        for batch_idx in range(B):
            for c_out_group in range(0, C_out, c_out_group_size * P_MAX):
                # Load bias and filters for C_out group
                bias_cache = load_bias(bias[c_out_group:c_out_group_end])
                filters_cache = load_filters(filters[:, :, c_out_group:c_out_group_end])

                for l_start in range(0, L_out, L_tile):
                    # Initialize PSUM accumulators
                    psum_tiles = zeros(c_out_tile_size, l_tile_size)

                    for c_in_tile in range(0, C_in, P_MAX):
                        # Load input window from HBM to SBUF
                        input_window = load_input(x_in[batch_idx, c_in_tile:c_in_end, window_start:window_end])

                        for k_outer in range(0, K, K_REP):
                            # Scatter input to K-replicated stacked format
                            input_stacked = scatter_to_stacked(input_window, k_outer, K_REP)

                            # Matrix multiply: filters_stacked @ input_stacked -> psum
                            psum_tiles += matmul(filters_cache[c_in_tile][k_outer], input_stacked)

                    # Apply bias and activation, copy from PSUM to SBUF
                    result = apply_bias_activation(psum_tiles, bias_cache, activation_fn)

                    # Store result from SBUF to HBM
                    y_out[batch_idx, c_out_group:c_out_group_end, l_start:l_end] = result
        return y_out
    """
    # Build configuration objects
    cfg = _build_conv1d_config(x_in, filters, bias, stride, padding, dilation, activation_fn, lnc_shard)
    tile_cfg = _build_tile_config(cfg, lnc_shard)
    mem_cfg = _build_memory_config(cfg, tile_cfg, sizeinbytes(x_in.dtype))

    # Input validation
    _validate_conv1d_inputs(x_in, filters, bias, cfg.stride, cfg.pad_left, cfg.pad_right, cfg.dilation)

    # Create output tensor in HBM
    y_out = nl.ndarray(
        shape=(cfg.B, cfg.C_out, cfg.L_out),
        dtype=x_in.dtype,
        buffer=nl.shared_hbm if cfg.lnc_shard else nl.hbm,
    )

    # Track PSUM bank for round-robin scheduling
    psum_bank_idx = 0

    # Initialize SBUF manager
    sbm = SbufManager(0, mem_cfg.TOTAL_SBUF, logger=get_logger("conv1d"))

    # Open outermost scope for C_out group interleaving
    sbm.open_scope(interleave_degree=mem_cfg.c_out_group_interleave, name="cout_group")

    # Tile on C_out group
    for c_out_group_start in range(tile_cfg.C_out_start, tile_cfg.C_out_end, mem_cfg.c_out_group_size * tile_cfg.P_MAX):
        c_out_group_end = min(c_out_group_start + mem_cfg.c_out_group_size * tile_cfg.P_MAX, tile_cfg.C_out_end)
        actual_group_size = div_ceil(c_out_group_end - c_out_group_start, tile_cfg.P_MAX)

        # Load bias and filters for C_out tiles in this C_out group
        c_out_tile_sizes, bias_cache, filters_cache = _load_bias_and_filters_for_c_out_group(
            filters=filters,
            bias=bias,
            sbm=sbm,
            cfg=cfg,
            tile_cfg=tile_cfg,
            c_out_group_start=c_out_group_start,
            c_out_group_end=c_out_group_end,
        )

        # Open scope for L_out tile interleaving
        sbm.open_scope(interleave_degree=mem_cfg.l_out_interleave, name=f"l_out_cg{c_out_group_start}")

        # Tile on B (batch dimension)
        for batch_idx in range(cfg.B):
            # Tile on L_out (output sequence dimension)
            for l_start in range(0, cfg.L_out, tile_cfg.L_tile):
                l_end = min(l_start + tile_cfg.L_tile, cfg.L_out)
                l_tile_size = l_end - l_start

                # Allocate result SBUF buffers
                result_sbufs = []
                for c_out_tile_idx in range(actual_group_size):
                    result_sbuf = sbm.alloc_stack(
                        shape=(c_out_tile_sizes[c_out_tile_idx], l_tile_size),
                        dtype=x_in.dtype,
                        name=f"result_sbuf_cg{c_out_group_start}_b{batch_idx}_l{l_start}_t{c_out_tile_idx}",
                    )
                    result_sbufs.append(result_sbuf)

                # Process L_out tile
                _conv1d_lout_tile(
                    x_in_view=TensorView(x_in).select(dim=0, index=batch_idx),
                    filters_cache=filters_cache,
                    bias_cache=bias_cache,
                    result_sbufs=result_sbufs,
                    sbm=sbm,
                    cfg=cfg,
                    tile_cfg=tile_cfg,
                    mem_cfg=mem_cfg,
                    c_out_tile_sizes=c_out_tile_sizes,
                    l_start=l_start,
                    l_end=l_end,
                    psum_bank_idx=psum_bank_idx,
                    name_suffix=f"cg{c_out_group_start}_b{batch_idx}_l{l_start}",
                )

                # Store results from SBUF to HBM
                for c_out_tile_idx in range(actual_group_size):
                    c_out_start_tile = c_out_group_start + c_out_tile_idx * tile_cfg.P_MAX
                    c_out_end_tile = c_out_start_tile + c_out_tile_sizes[c_out_tile_idx]

                    _store_result_to_hbm(
                        TensorView(y_out)
                        .select(dim=0, index=batch_idx)
                        .slice(dim=0, start=c_out_start_tile, end=c_out_end_tile, step=1)
                        .slice(dim=1, start=l_start, end=l_end, step=1),
                        result_sbufs[c_out_tile_idx],
                    )

                # Increment section for L_out interleaving
                sbm.increment_section()
                psum_bank_idx = (psum_bank_idx + actual_group_size) % tile_cfg.NUM_PSUM_BANKS

        # Close L_out scope
        sbm.close_scope()

        # Increment section for C_out group interleaving
        sbm.increment_section()

    # Close the C_out group scope
    sbm.close_scope()

    return y_out
