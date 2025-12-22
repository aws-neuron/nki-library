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
MLP CTE Quantization Module

Handles tensor quantization for the MLP CTE kernel with support for multiple
quantization schemes, including dynamic row-wise and static tensor-wise.

"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import SbufManager
from ...utils.kernel_helpers import get_max_positive_value_for_dtype
from ..mlp_parameters import MLPParameters
from .mlp_cte_constants import MLPCTEConstants
from .mlp_cte_tile_info import MLPCTETileInfo

#
# Local constants
_MINVAL = 1e-6
_FP8_E4M3_MAX_POS_VAL = 240.0


def invert_static_scales(scales_sbuf: nl.ndarray):
    nisa.reciprocal(
        dst=scales_sbuf[0 : nl.tile_size.pmax, 0:1],
        data=scales_sbuf[0 : nl.tile_size.pmax, 0:1],
    )


def perform_intermediate_static_quantization(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    src_proj_res_sbuf_list: list[nl.ndarray],
    quantized_output_sbuf_list: list[nl.ndarray],
    static_input_quant_scale_sbuf: Optional[nl.ndarray],
):
    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count
    max_pos_val = _FP8_E4M3_MAX_POS_VAL
    rounded_intermediate_dim = int_dim_tile.tile_count * int_dim_tile.tile_size

    # Reshape to get the shapes we need
    src_proj_res_sbuf_view_list = []
    quantized_output_sbuf_view_list = []
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        src_proj_res_sbuf_view = src_proj_res_sbuf_list[bxs_subtile_idx].reshape(
            (
                bxs_dim_tile.subtile_dim_info.tile_size,
                rounded_intermediate_dim,
            )
        )
        src_proj_res_sbuf_view_list.append(src_proj_res_sbuf_view)
        quantized_output_sbuf_view = quantized_output_sbuf_list[bxs_subtile_idx].reshape(
            (
                bxs_dim_tile.subtile_dim_info.tile_size,
                rounded_intermediate_dim,
            )
        )
        quantized_output_sbuf_view_list.append(quantized_output_sbuf_view)

    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        p_bxs_size = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)
        nisa.activation(
            dst=src_proj_res_sbuf_view_list[bxs_subtile_idx][:p_bxs_size, : mlp_params.intermediate_size],
            op=nl.copy,
            data=src_proj_res_sbuf_view_list[bxs_subtile_idx][:p_bxs_size, : mlp_params.intermediate_size],
            scale=static_input_quant_scale_sbuf[:p_bxs_size, 0:1],
            bias=bias_vector[:p_bxs_size, 0:1],
        )
        nisa.tensor_scalar(
            dst=quantized_output_sbuf_view_list[bxs_subtile_idx][:p_bxs_size, : mlp_params.intermediate_size],
            data=src_proj_res_sbuf_view_list[bxs_subtile_idx][:p_bxs_size, : mlp_params.intermediate_size],
            op0=nl.minimum,
            operand0=max_pos_val,
            op1=nl.maximum,
            operand1=-max_pos_val,
        )


def perform_intermediate_row_quantization(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    src_proj_res_sbuf_list: list[nl.ndarray],
    quantized_output_sbuf_list: list[nl.ndarray],
    row_dequant_scales_sbuf_list: Optional[nl.ndarray],
    sbm: SbufManager,
):
    # Alias these to cut down on the code size for tile information references
    bxs_dim_tile = tile_info.bxs_dim_tile
    int_dim_tile = tile_info.src_proj_intermediate_dim_tile
    bias_vector = constants.bxs_dim_subtile_zero_bias_vector_sbuf
    BXS_SUBTILE_COUNT = bxs_dim_tile.subtile_dim_info.tile_count
    max_pos_val = _FP8_E4M3_MAX_POS_VAL
    rounded_intermediate_dim = int_dim_tile.tile_count * int_dim_tile.tile_size

    # Allocate buffers to store intermediate tensors
    quant_abs_sbuf_list = []
    quant_scales_sbuf_list = []
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        alloc_stack = sbm.alloc_stack if sbm else nl.ndarray
        quant_abs_sbuf = alloc_stack(
            (bxs_dim_tile.subtile_dim_info.tile_size, rounded_intermediate_dim),
            dtype=constants.compute_data_type,
        )
        quant_abs_sbuf_list.append(quant_abs_sbuf)
        quant_scales_sbuf = alloc_stack(
            (bxs_dim_tile.subtile_dim_info.tile_size, 1),
            dtype=nl.float32,
        )
        quant_scales_sbuf_list.append(quant_scales_sbuf)

    # Reshape to get the shapes we need
    src_proj_res_sbuf_view_list = []
    quantized_output_sbuf_view_list = []
    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        src_proj_res_sbuf_view = src_proj_res_sbuf_list[bxs_subtile_idx].reshape(
            (
                bxs_dim_tile.subtile_dim_info.tile_size,
                rounded_intermediate_dim,
            )
        )
        src_proj_res_sbuf_view_list.append(src_proj_res_sbuf_view)
        quantized_output_sbuf_view = quantized_output_sbuf_list[bxs_subtile_idx].reshape(
            (
                bxs_dim_tile.subtile_dim_info.tile_size,
                rounded_intermediate_dim,
            )
        )
        quantized_output_sbuf_view_list.append(quantized_output_sbuf_view)

    for bxs_subtile_idx in range(BXS_SUBTILE_COUNT):
        p_bxs_size = bxs_dim_tile.get_subtile_bound(bxs_tile_idx, bxs_subtile_idx)

        # clip values to [-clipping_bound, clipping_bound]
        if mlp_params.quant_params.has_clipping_bound():
            nisa.tensor_scalar(
                dst=src_proj_res_sbuf_view_list[bxs_subtile_idx][:p_bxs_size, : mlp_params.intermediate_size],
                data=src_proj_res_sbuf_view_list[bxs_subtile_idx][:p_bxs_size, : mlp_params.intermediate_size],
                op0=nl.minimum,
                operand0=mlp_params.quant_params.clipping_bound,
                op1=nl.maximum,
                operand1=-mlp_params.quant_params.clipping_bound,
            )

        # compute quant_scales = 1 / (absmax(x) / max_pos)
        nisa.tensor_scalar_reduce(
            dst=quant_abs_sbuf_list[bxs_subtile_idx][:p_bxs_size, : mlp_params.intermediate_size],
            data=src_proj_res_sbuf_view_list[bxs_subtile_idx][:p_bxs_size, : mlp_params.intermediate_size],
            op0=nl.abs,
            operand0=0.0,
            reduce_op=nl.maximum,
            reduce_res=row_dequant_scales_sbuf_list[bxs_subtile_idx][:p_bxs_size, 0:1],
        )
        nisa.tensor_scalar(
            dst=row_dequant_scales_sbuf_list[bxs_subtile_idx][:p_bxs_size, 0:1],
            data=row_dequant_scales_sbuf_list[bxs_subtile_idx][:p_bxs_size, 0:1],
            op0=nl.multiply,
            operand0=1.0 / max_pos_val,
            op1=nl.maximum,
            operand1=_MINVAL,
        )
        nisa.reciprocal(
            dst=quant_scales_sbuf_list[bxs_subtile_idx][:p_bxs_size, 0:1],
            data=row_dequant_scales_sbuf_list[bxs_subtile_idx][:p_bxs_size, 0:1],
        )

        # apply quant_scales
        nisa.activation(
            dst=quantized_output_sbuf_view_list[bxs_subtile_idx][:p_bxs_size, : mlp_params.intermediate_size],
            data=src_proj_res_sbuf_view_list[bxs_subtile_idx][:p_bxs_size, : mlp_params.intermediate_size],
            op=nl.copy,
            scale=quant_scales_sbuf_list[bxs_subtile_idx][:p_bxs_size, 0:1],
            bias=bias_vector[:p_bxs_size, 0:1],
        )


def perform_intermediate_quantization(
    mlp_params: MLPParameters,
    tile_info: MLPCTETileInfo,
    constants: MLPCTEConstants,
    bxs_tile_idx: int,
    src_proj_res_sbuf_list: list[nl.ndarray],
    quantized_output_sbuf_list: list[nl.ndarray],
    row_dequant_scales_sbuf_list: Optional[nl.ndarray],
    static_input_quant_scale_sbuf: Optional[nl.ndarray],
    sbm: SbufManager,
):
    if mlp_params.quant_params.is_quant_static():
        perform_intermediate_static_quantization(
            mlp_params,
            tile_info,
            constants,
            bxs_tile_idx,
            src_proj_res_sbuf_list,
            quantized_output_sbuf_list,
            static_input_quant_scale_sbuf,
        )
    elif mlp_params.quant_params.is_quant_row():
        perform_intermediate_row_quantization(
            mlp_params,
            tile_info,
            constants,
            bxs_tile_idx,
            src_proj_res_sbuf_list,
            quantized_output_sbuf_list,
            row_dequant_scales_sbuf_list,
            sbm,
        )
