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
MLP CTE Utilities Module

Provides unified utility functions for MLP CTE kernels including bias operations,
elementwise multiplication, and activation functions with boolean parameter support.

"""

from dataclasses import dataclass
from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_ceil_quotient, get_nl_act_fn_from_type
from ..mlp_parameters import MLPParameters
from .mlp_cte_constants import MLPCTEConstants
from .mlp_cte_tile_info import MLPCTETileInfo

#
# ***************
# General Helpers
# ***************
#


def is_launch_grid_valid_for_mlp() -> bool:
    """Check if launch grid configuration is valid for MLP operations.

    Validates that the launch grid supports MLP CTE sharding requirements.

    Args:
        None

    Returns:
        True if launch grid is valid, False otherwise

    Intended Usage:
        Called at kernel startup to validate execution environment
    """
    # We only support sharding on 1 dimension
    grid_ndim = nl.program_ndim()
    if grid_ndim < 0 or grid_ndim > 1:
        return False

    return True


#
# Calculates the number of elements per partition for vectors that are to be loaded across partitions like so:
#   partition 0, free 0 = element 0
#   partition 1, free 0 = element 1
#   partition 2, free 0 = element 2
#   ...
#   partition pmax, free 0 = element pmax - 1
#   partition    0, free 1 = element pmax
#   ...
def calc_vec_crossload_free_dim_len(vector_tensor_hbm: nl.ndarray) -> tuple[int, int]:
    """Calculate vector cross-load free dimension length.

    Computes the number of elements per partition for vectors loaded across partitions.

    Args:
        vector_tensor_hbm: Input vector tensor in HBM

    Returns:
        Tuple of vector length and elements per partition

    Intended Usage:
        Used when loading vectors that need to be distributed across partition dimensions
    """
    # We are expecting a vector here of one of these shapes: [1, X] or [X, 1]
    kernel_assert(
        (len(vector_tensor_hbm.shape) == 2) and (vector_tensor_hbm.shape[0] == 1 or vector_tensor_hbm.shape[1] == 1),
        f"Unexpected HBM tensor shape of {vector_tensor_hbm.shape}. Expected a vector with shape [1, X] or [X, 1].",
    )

    vec_len = max(vector_tensor_hbm.shape[0], vector_tensor_hbm.shape[1])
    return (vec_len, get_ceil_quotient(vec_len, nl.tile_size.pmax))


#
# ***************
# Bias Operations
# ***************
#


#
# Add bias to a source projection (up or gate) on PSUM and store the result in SBUF
def apply_source_projection_bias(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    proj_psum_list: list[nl.ndarray],
    bias_tensor_sbuf: nl.ndarray,
    bias_res_sbuf_list: list[nl.ndarray],
):
    """Add bias to source projection results in PSUM and store in SBUF.

    Applies bias addition to projection results stored in PSUM memory and copies
    the results to SBUF for further processing.
    All inputs and outputs to this function are assumed to be in PSUM/SBUF.

    Args:
        mlp_params: MLP configuration parameters
        tile_info: Tiling information for the computation
        constants: MLP CTE constants configuration
        bxs_tile_idx: Current batch×sequence tile index
        proj_psum_list: List of projection result tensors in PSUM
        bias_tensor_sbuf: Bias tensor in SBUF
        bias_res_sbuf_list: Output list for bias results in SBUF

    Returns:
        None

    Intended Usage:
        Called after projection operations to apply bias when bias is enabled
    """
    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count

    # This is the total size from the tensor that we are computing
    tensor_bxs_size = constants.get_bxs_size(mlp_params)

    # Loop over all the intermediate tiles within the B x S subtiles and compute the bias addition
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        for intermediate_tile_idx in range(int_dim_tile.tile_count):
            p_bxs_size = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)
            f_int_size = int_dim_tile.get_tile_bound(intermediate_tile_idx)

            if p_bxs_size <= 0 or f_int_size <= 0:
                continue

            psum_bank = bxs_subtile_idx * int_dim_tile.tile_count + intermediate_tile_idx
            nisa.tensor_tensor(
                dst=bias_res_sbuf_list[bxs_subtile_idx][:p_bxs_size, intermediate_tile_idx, :f_int_size],
                data1=proj_psum_list[psum_bank][:p_bxs_size, :f_int_size],
                data2=bias_tensor_sbuf[
                    :p_bxs_size,
                    int_dim_tile.get_tile_indices(intermediate_tile_idx, f_int_size),
                ],
                op=nl.add,
            )


#
# ***************
# Multiplication Operations
# ***************
#


#
# Perform element-wise multiply between gate and up projections
def perform_elementwise_multiply(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    gate_tile_sbuf_list: list[nl.ndarray],
    up_tile_data_list: list[nl.ndarray],
    up_weight_row_scales_sbuf: Optional[nl.ndarray],
    up_static_scales_sbuf: Optional[nl.ndarray],
    hidden_scales_sbuf_list: Optional[nl.ndarray],
    output_tile_sbuf_list: list[nl.ndarray],
    up_data_is_psum: bool = False,
):
    """Perform elementwise multiplication between gate and up projection results.

    Multiplies gate projection results with up projection results element-wise,
    supporting both PSUM and SBUF input data formats.

    Args:
        mlp_params: MLP configuration parameters
        tile_info: Tiling information for the computation
        constants: MLP CTE constants configuration
        bxs_tile_idx: Current batch×sequence tile index
        gate_tile_sbuf_list: Gate projection results in SBUF
        up_tile_data_list: Up projection data (PSUM or SBUF)
        up_weight_row_scales_sbuf: Optional up weight row dequant scales in SBUF
        up_static_scales_sbuf: Optional up static dequant scales in SBUF
        hidden_scales_sbuf_list: Optional hidden scales in SBUF
        output_tile_sbuf_list: Output tensors in SBUF
        up_data_is_psum: Whether up projection data is in PSUM format

    Returns:
        None

    Intended Usage:
        Called to combine gate and up projection results in gated MLP architectures
    """
    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count

    # Loop over all the intermediate tiles within the B x S subtiles and compute the multiply
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        for intermediate_tile_idx in range(int_dim_tile.tile_count):
            p_bxs_size = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)
            f_int_size = int_dim_tile.get_tile_bound(intermediate_tile_idx)

            if p_bxs_size <= 0 or f_int_size <= 0:
                continue

            gate_data = gate_tile_sbuf_list[bxs_subtile_idx][:p_bxs_size, intermediate_tile_idx, :f_int_size]

            if up_data_is_psum:
                psum_bank = bxs_subtile_idx * int_dim_tile.tile_count + intermediate_tile_idx
                up_data = up_tile_data_list[psum_bank][:p_bxs_size, :f_int_size]
            else:
                up_data = up_tile_data_list[bxs_subtile_idx][:p_bxs_size, intermediate_tile_idx, :f_int_size]
            dst_tile = output_tile_sbuf_list[bxs_subtile_idx][:p_bxs_size, intermediate_tile_idx, :f_int_size]

            nisa.tensor_tensor(dst=dst_tile, data1=gate_data, data2=up_data, op=nl.multiply)

            if mlp_params.quant_params.is_quant_static():
                nisa.activation(
                    dst=dst_tile,
                    op=nl.copy,
                    data=dst_tile,
                    scale=up_static_scales_sbuf[:p_bxs_size, 0:1],
                    bias=bias_vector[:p_bxs_size, 0:1],
                )
            elif mlp_params.quant_params.is_quant_row():
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=dst_tile,
                    data2=up_weight_row_scales_sbuf[
                        :p_bxs_size, nl.ds(intermediate_tile_idx * int_dim_tile.tile_size, f_int_size)
                    ],
                )
                nisa.activation(
                    dst=dst_tile,
                    op=nl.copy,
                    data=dst_tile,
                    scale=hidden_scales_sbuf_list[bxs_subtile_idx][:p_bxs_size, 0:1],
                    bias=bias_vector[:p_bxs_size, 0:1],
                )


#
# ***************
# Activation Operations
# ***************
#


#
# Apply an activation function on a source projection (up or gate)
def apply_source_projection_activation(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    proj_data_list: list[nl.ndarray],
    src_weight_row_scales_sbuf: Optional[nl.ndarray],
    src_static_scales_sbuf: Optional[nl.ndarray],
    hidden_scales_sbuf_list: Optional[nl.ndarray],
    act_fn_res_sbuf_list: list[nl.ndarray],
    data_is_psum: bool = False,
):
    """Apply activation function to source projection results.

    Applies the specified activation function to projection data, supporting both
    PSUM and SBUF input formats with optional scaling.

    Args:
        mlp_params: MLP configuration parameters
        tile_info: Tiling information for the computation
        constants: MLP CTE constants configuration
        bxs_tile_idx: Current batch×sequence tile index
        proj_data_list: Projection data (PSUM or SBUF)
        src_weight_row_scales_sbuf: Optional source weight row dequant scales
        src_static_scales_sbuf: Optional source projection static dequant scales
        hidden_scales_sbuf_list: Optional hidden scales
        act_fn_res_sbuf_list: Output activation results in SBUF
        data_is_psum: Whether input data is in PSUM format

    Returns:
        None

    Intended Usage:
        Called to apply activation functions like ReLU, GELU, or SiLU to projection results
    """
    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count

    # Loop over all the intermediate tiles within the B x S subtiles and compute the activation
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        for intermediate_tile_idx in range(int_dim_tile.tile_count):
            # Calculate valid ranges
            p_bxs_size = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)
            f_int_size = int_dim_tile.get_tile_bound(intermediate_tile_idx)

            if p_bxs_size <= 0 or f_int_size <= 0:
                continue

            # Calculate bias sizes (might be different from data sizes)
            p_bias_size = min(p_bxs_size, bias_vector.shape[0])
            f_bias_size = min(f_int_size, bias_vector.shape[1])

            if data_is_psum:
                psum_bank = bxs_subtile_idx * int_dim_tile.tile_count + intermediate_tile_idx
                proj_data = proj_data_list[psum_bank][:p_bxs_size, :f_int_size]
            else:
                proj_data = proj_data_list[bxs_subtile_idx][:p_bxs_size, intermediate_tile_idx, :f_int_size]
            dst_tile = act_fn_res_sbuf_list[bxs_subtile_idx][:p_bxs_size, intermediate_tile_idx, :f_int_size]

            if mlp_params.quant_params.is_quant_static():
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=proj_data,
                    scale=src_static_scales_sbuf[:p_bxs_size, 0:1],
                    bias=bias_vector[:p_bias_size, :f_bias_size],
                )
            elif mlp_params.quant_params.is_quant_row():
                nisa.tensor_tensor(
                    dst=dst_tile,
                    op=nl.multiply,
                    data1=proj_data,
                    data2=src_weight_row_scales_sbuf[:p_bxs_size, :f_int_size],
                )
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=dst_tile,
                    scale=hidden_scales_sbuf_list[bxs_subtile_idx][:p_bxs_size, 0:1],
                    bias=bias_vector[:p_bias_size, :f_bias_size],
                )
            else:
                nisa.activation(
                    dst=dst_tile,
                    op=get_nl_act_fn_from_type(mlp_params.activation_fn),
                    data=proj_data,
                    bias=bias_vector[:p_bias_size, :f_bias_size],
                )
