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

"""Configuration parameters, constants, and tiling strategies for output projection CTE kernels."""

from dataclasses import dataclass
from typing import Optional, Tuple

import nki.language as nl

from ...utils.common_types import QuantizationType
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import div_ceil
from ...utils.tile_info import TiledDimInfo

# Hardware constants - use literal values since nl.tile_size.* returns -1 at module load time
P_MAX = 128
F_MAX = 512

# Maximum SBUF space (in bytes) allowed for weight tensors
_MAX_WEIGHT_SBUF_BYTES = 10 * 1024 * 1024  # 10MB

# Validation limits for testing
_MAX_VALIDATED_H_SIZE = 16384 + 4321
_MAX_VALIDATED_B_TIMES_S_SIZE = 128 * 1024
_MAX_VALIDATED_N_SIZE = 17


@dataclass
class QuantizationConfig(nl.NKIObject):
    """
    Configuration for quantization in output projection.

    Captures quantization parameters including scales, data types, and optimization flags.

    Args:
        is_enabled (bool): Whether quantization is enabled.
        input_scales (Optional[nl.ndarray]): Input quantization scales.
        weight_scales (Optional[nl.ndarray]): Weight quantization scales.
        quant_data_type (Optional[type]): Quantized data type (e.g., nl.float8_e4m3).
        input_quantized (bool): Whether input is already quantized.
        input_data_type (Optional[type]): Original input data type.
        weight_data_type (Optional[type]): Original weight data type.
        use_double_row (bool): Whether to use double row matmul optimization.
        is_fp8_quantized (bool): Whether FP8 quantization is enabled.

    Notes:
        - For STATIC quantization, both input_scales and weight_scales are required.
        - Double row optimization is enabled for STATIC quantization.
    """

    is_enabled: bool
    input_scales: Optional[nl.ndarray] = None
    weight_scales: Optional[nl.ndarray] = None
    quant_data_type: Optional[type] = None
    input_quantized: bool = False
    input_data_type: Optional[type] = None
    weight_data_type: Optional[type] = None
    use_double_row: bool = False
    is_fp8_quantized: bool = False


@dataclass
class TilingConfig(nl.NKIObject):
    """
    Tiling configuration for output projection CTE kernel.

    Contains dimension sizes, tiling parameters, and tile info objects for
    calculating actual tile sizes with boundary handling.

    Args:
        num_prgs (int): Number of logical cores (LNC).
        b_size (int): Batch size.
        n_size (int): Number of heads (after packing).
        d_size (int): Head dimension (after packing).
        s_size (int): Sequence length.
        h_size (int): Hidden dimension.
        h_sharded_size (int): Hidden dimension per shard.
        num_h_blocks_per_prg (int): Number of H blocks per program.
        group_size (int): Number of heads packed together (1 = no packing).
        s_tile (TiledDimInfo): Tile info for S dimension.
        h_tile (TiledDimInfo): Tile info for H dimension.

    Notes:
        - n_size and d_size may differ from original due to head packing.
        - Last block may be smaller than block_size (handled by TiledDimInfo).
    """

    num_prgs: int
    b_size: int
    n_size: int
    d_size: int
    s_size: int
    h_size: int
    h_sharded_size: int
    num_h_blocks_per_prg: int
    group_size: int
    s_tile: TiledDimInfo
    h_tile: TiledDimInfo


def _get_dtype_size(dtype) -> int:
    """
    Return size in bytes for NKI dtype.

    Args:
        dtype: NKI data type.

    Returns:
        int: Size in bytes for the given dtype.
    """
    if dtype == nl.float32:
        return 4
    if dtype in (nl.float16, nl.bfloat16):
        return 2
    if dtype == nl.float8_e4m3:
        return 1
    return 2


def _calculate_head_packing(n_size: int, d_size: int, partition_size: int) -> Tuple[int, int, int]:
    """
    Optimize contraction dimension by folding N into D when D < partition_size.

    Maximizes PE engine utilization by packing multiple heads into the partition dimension.

    Args:
        n_size (int): Number of heads.
        d_size (int): Head dimension size.
        partition_size (int): Hardware constraint (typically P_MAX=128).

    Returns:
        Tuple[int, int, int]: (new_n_size, new_d_size, group_size).

    Notes:
        - group_size indicates how many heads are packed together.
        - new_n_size = n_size // group_size, new_d_size = d_size * group_size.
    """
    group_size = n_size
    while (n_size % group_size) or (group_size * d_size) > partition_size:
        group_size -= 1
    kernel_assert(group_size > 0, f"group_size should be >= 1, got {group_size}.")

    new_n_size = n_size // group_size
    new_d_size = d_size * group_size
    return new_n_size, new_d_size, group_size


def _calculate_double_row_head_packing(
    n_size: int,
    d_size: int,
    is_fp8_quantized: bool,
) -> Tuple[int, int, bool]:
    """
    Adjust N and D dimensions for double row matmul when N is odd.

    When using double row matmul with odd number of heads, splits head dimension
    to increase number of heads, making it even.

    Args:
        n_size (int): Number of heads.
        d_size (int): Head dimension size.
        is_fp8_quantized (bool): Whether FP8 quantization is enabled.

    Returns:
        Tuple[int, int, bool]: (new_n_size, new_d_size, use_double_row).

    Notes:
        - Only applies for FP8 quantization.
        - If D is odd and N is odd, disables double row optimization.
    """
    use_double_row = is_fp8_quantized
    if is_fp8_quantized:
        if n_size % 2 != 0:
            if d_size % 2 == 0:
                n_size = n_size * 2
                d_size = d_size // 2
            else:
                use_double_row = False
    return n_size, d_size, use_double_row


def _calculate_h_block_size(
    n_size: int,
    d_size: int,
    h_sharded_size: int,
    weight_dtype,
    max_sbuf_bytes: int = _MAX_WEIGHT_SBUF_BYTES,
) -> Tuple[int, int]:
    """
    Calculate H dimension tiling to fit weight tensors within SBUF budget.

    Weight tensor for a single h_block has shape [n_size][d_size, h_block_size].
    Total bytes = n_size * d_size * h_block_size * dtype_size.

    Args:
        n_size (int): Number of heads.
        d_size (int): Head dimension size.
        h_sharded_size (int): Size of H dimension for this shard.
        weight_dtype: Data type of weight tensor.
        max_sbuf_bytes (int): Maximum SBUF bytes allowed for weights.

    Returns:
        Tuple[int, int]: (h_block_size, num_h_blocks_per_prg).

    Notes:
        - h_block_size is aligned to F_MAX (512).
        - Last block may be smaller (handled by TiledDimInfo).
    """
    dtype_size = _get_dtype_size(weight_dtype)
    bytes_per_h_element = n_size * d_size * dtype_size
    max_h_block_size = max_sbuf_bytes // bytes_per_h_element
    max_h_block_size = (max_h_block_size // F_MAX) * F_MAX
    max_h_block_size = max(max_h_block_size, F_MAX)
    h_block_size = min(max_h_block_size, h_sharded_size)
    h_block_size = (h_block_size // F_MAX) * F_MAX
    h_block_size = max(h_block_size, F_MAX)
    num_h_blocks_per_prg = div_ceil(h_sharded_size, h_block_size)

    return h_block_size, num_h_blocks_per_prg


def build_tiling_config(
    b_size: int,
    n_size: int,
    d_size: int,
    s_size: int,
    h_size: int,
    n_prgs: int,
    quant_config: QuantizationConfig,
    weight_dtype=nl.bfloat16,
) -> TilingConfig:
    """
    Create TilingConfig with dimension sizes and tiling decisions.

    Handles head packing optimizations:
    1. Standard head packing: Folds N into D when D < P_MAX.
    2. Double row head packing: Adjusts N and D for FP8 to ensure N is even.

    Args:
        b_size (int): Batch size.
        n_size (int): Number of heads (original).
        d_size (int): Head dimension size (original).
        s_size (int): Sequence length.
        h_size (int): Hidden dimension size.
        n_prgs (int): Number of logical cores.
        quant_config (QuantizationConfig): Quantization configuration.
        weight_dtype: Data type of weight tensor.

    Returns:
        TilingConfig: Configuration with all tiling parameters.

    Notes:
        - n_size and d_size in config may differ from inputs due to head packing.
    """
    n_size, d_size, group_size = _calculate_head_packing(n_size, d_size, P_MAX)

    n_size, d_size, use_double_row = _calculate_double_row_head_packing(
        n_size=n_size,
        d_size=d_size,
        is_fp8_quantized=quant_config.is_fp8_quantized,
    )
    quant_config.use_double_row = use_double_row

    h_sharded_size = h_size // n_prgs
    h_block_size, num_h_blocks_per_prg = _calculate_h_block_size(
        n_size=n_size,
        d_size=d_size,
        h_sharded_size=h_sharded_size,
        weight_dtype=weight_dtype,
    )

    s_block_size = F_MAX

    s_tile = TiledDimInfo.build_with_subtiling(
        tiled_dim_size=s_size,
        tile_size=s_block_size,
        subtile_size=P_MAX,
    )

    h_tile = TiledDimInfo.build_with_subtiling(
        tiled_dim_size=h_sharded_size,
        tile_size=h_block_size,
        subtile_size=F_MAX,
    )

    return TilingConfig(
        num_prgs=n_prgs,
        b_size=b_size,
        n_size=n_size,
        d_size=d_size,
        s_size=s_size,
        h_size=h_size,
        h_sharded_size=h_sharded_size,
        num_h_blocks_per_prg=num_h_blocks_per_prg,
        group_size=group_size,
        s_tile=s_tile,
        h_tile=h_tile,
    )


def build_quantization_config(
    quantization_type: QuantizationType,
    input_scales: Optional[nl.ndarray],
    weight_scales: Optional[nl.ndarray],
    input_data_type: Optional[type],
    weight_data_type: Optional[type],
) -> QuantizationConfig:
    """
    Setup quantization configuration including scale loading and double row optimization.

    Args:
        quantization_type (QuantizationType): Type of quantization to use.
        input_scales (Optional[nl.ndarray]): Input quantization scales.
        weight_scales (Optional[nl.ndarray]): Weight quantization scales.
        input_data_type (Optional[type]): Data type of input tensor.
        weight_data_type (Optional[type]): Data type of weight tensor.

    Returns:
        QuantizationConfig: Configuration with quantization parameters.

    Notes:
        - For STATIC quantization, both input_scales and weight_scales are required.
        - Double row optimization is enabled for STATIC quantization.
    """
    if quantization_type == QuantizationType.NONE:
        return QuantizationConfig(is_enabled=False)

    quant_data_type = None
    is_fp8_quantized = False

    if quantization_type == QuantizationType.STATIC:
        quant_data_type = nl.float8_e4m3
        is_fp8_quantized = True

    return QuantizationConfig(
        is_enabled=True,
        input_scales=input_scales,
        weight_scales=weight_scales,
        quant_data_type=quant_data_type,
        input_quantized=False,
        input_data_type=input_data_type,
        weight_data_type=weight_data_type,
        use_double_row=False,
        is_fp8_quantized=is_fp8_quantized,
    )


def validate_output_projection_inputs(
    b_size: int,
    n_size: int,
    d_size: int,
    s_size: int,
    h_size: int,
    n_prgs: int,
    attention_dtype,
    weight_dtype,
) -> None:
    """
    Validate input parameters for output projection CTE kernel.

    Checks dimension sizes, data types, and sharding constraints.

    Args:
        b_size (int): Batch size.
        n_size (int): Number of heads.
        d_size (int): Head dimension size.
        s_size (int): Sequence length.
        h_size (int): Hidden dimension size.
        n_prgs (int): Number of logical cores (LNC).
        attention_dtype: Data type of attention tensor.
        weight_dtype: Data type of weight tensor.

    Raises:
        AssertionError: If any validation check fails.

    Notes:
        - MAX_VALIDATED_B_TIMES_S_SIZE: 131072
        - MAX_VALIDATED_H_SIZE: 20705
        - MAX_VALIDATED_N_SIZE: 17
    """
    kernel_assert(
        b_size * s_size <= _MAX_VALIDATED_B_TIMES_S_SIZE,
        f"Product B * S must not exceed {_MAX_VALIDATED_B_TIMES_S_SIZE}. "
        f"Got B={b_size}, S={s_size}, B*S={b_size * s_size}",
    )

    kernel_assert(
        d_size <= P_MAX,
        f"Head dimension D must not exceed {P_MAX}. Got D={d_size}",
    )

    kernel_assert(
        h_size <= _MAX_VALIDATED_H_SIZE,
        f"Hidden dimension H must not exceed {_MAX_VALIDATED_H_SIZE}. Got H={h_size}",
    )

    kernel_assert(
        n_size <= _MAX_VALIDATED_N_SIZE,
        f"Number of heads N must not exceed {_MAX_VALIDATED_N_SIZE}. Got N={n_size}",
    )

    kernel_assert(
        h_size % n_prgs == 0,
        f"Hidden dimension H must be divisible by LNC. Got H={h_size}, LNC={n_prgs}, H % LNC = {h_size % n_prgs}",
    )
