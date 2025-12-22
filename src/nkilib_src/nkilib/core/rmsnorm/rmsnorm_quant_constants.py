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

"""Constants and configuration for the RMS Norm Quantization kernel."""

from dataclasses import dataclass

import nki.isa as nisa
import nki.language as nl
import numpy as np

from ..utils.kernel_assert import kernel_assert
from .rmsnorm_quant_tile_info import RMSNormQuantTileInfo


@dataclass
class RMSNormQuantConstants(nl.NKIObject):
    """
    Constants required by the RMS Norm Quantization kernel.

    Args:
        num_hw_psum_banks (int): Number of hardware PSUM banks
        compute_data_type (np.dtype): Data type for matmuls and compute
        quant_data_type (np.dtype): Data type for quantized output
        quant_data_type_range (int): Maximum representable value in quant dtype
        dequant_scale_size (int): Number of output elements for dequant scale
        min_dequant_scale_value (float): Minimum dequant scale for numerical stability
        outer_dim_tile_zero_bias_vector_sbuf (nl.ndarray): Zero bias vector in SBUF
        rmsn_eps_bias_sbuf (nl.ndarray): Epsilon bias vector for RMS norm
        pe_broadcast_ones_vector_sbuf (nl.ndarray): Ones vector for PE broadcasting
        outer_dim_size (int): Size of outer dimension
        proc_dim_size (int): Size of processing dimension
        MAX_S (int): Maximum supported sequence length
        MAX_H (int): Maximum supported hidden dimension
        MAX_B (int): Maximum supported batch size
    """

    num_hw_psum_banks: int
    compute_data_type: np.dtype
    quant_data_type: np.dtype
    quant_data_type_range: int
    dequant_scale_size: int
    min_dequant_scale_value: float
    outer_dim_tile_zero_bias_vector_sbuf: nl.ndarray
    rmsn_eps_bias_sbuf: nl.ndarray
    pe_broadcast_ones_vector_sbuf: nl.ndarray
    outer_dim_size: int
    proc_dim_size: int
    MAX_S: int = 32768
    MAX_H: int = 16384
    MAX_B: int = 2


def build_rms_norm_quant_constants(
    tile_info: RMSNormQuantTileInfo,
    eps: float,
    processing_shape: tuple[int, int],
) -> RMSNormQuantConstants:
    """
    Factory method to construct RMSNormQuantConstants.

    Args:
        tile_info (RMSNormQuantTileInfo): Tile configuration info
        eps (float): Epsilon value for numerical stability
        processing_shape (tuple[int, int]): (outer_dim_size, proc_dim_size)

    Returns:
        RMSNormQuantConstants: Initialized constants for the kernel
    """
    # TODO: Get this constant from the NKI API once it is available
    num_hw_psum_banks = 8
    # Data types
    compute_data_type = nl.bfloat16
    quant_data_type = nl.float8_e4m3
    # Range calculation: 3 bits of fraction, max exponent is (2^4 - 1) - 2^3 = 7...  2 ^ 7 * (1 + (2^7 / 2^8)) = 240
    quant_data_type_range = 240
    # The dequantizing scale factors are added to the end of each processed dimension.  The unit of this constant
    # is in output tensor elements.  Each dequantizing scale factor is a 32-bit float and the element size of
    # quant_data_type (i.e. the output type) is 8 bits.  Therefore, it takes 4 elements to store each dequantizing factor.
    # We do the math here for illustrative purposes plus the kernel_assert as a sanity check.
    float32_bytes = 4
    quant_type_bytes = 1  # float8_e4m3 is 1 byte
    kernel_assert(float32_bytes % quant_type_bytes == 0, "float32_bytes must be divisible by quant_type_bytes")
    dequant_scale_size = float32_bytes // quant_type_bytes
    # Used for numerical stability for quantizing
    min_dequant_scale_value = 1e-6
    # We need a zero bias vector for activations to work around a runtime issue when no bias vector is supplied to the activation method
    bias_vector_sbuf = nl.ndarray((tile_info.outer_dim_tile.tile_size, 1), nl.float32, buffer=nl.sbuf)
    nisa.memset(bias_vector_sbuf, value=0.0)
    # Epsilon is added to values across the outer dimension tile for RMS norm
    rmsn_eps_bias_sbuf = nl.ndarray((tile_info.outer_dim_tile.tile_size, 1), dtype=compute_data_type, buffer=nl.sbuf)
    nisa.memset(rmsn_eps_bias_sbuf, value=eps)
    # Used for broadcasting using the PE array
    ones_vector_sbuf = nl.ndarray((1, nl.tile_size.pmax), dtype=compute_data_type, buffer=nl.sbuf)
    nisa.memset(ones_vector_sbuf, value=1.0)

    return RMSNormQuantConstants(
        num_hw_psum_banks,
        compute_data_type,
        quant_data_type,
        quant_data_type_range,
        dequant_scale_size,
        min_dequant_scale_value,
        bias_vector_sbuf,
        rmsn_eps_bias_sbuf,
        ones_vector_sbuf,
        processing_shape[0],
        processing_shape[1],
    )
