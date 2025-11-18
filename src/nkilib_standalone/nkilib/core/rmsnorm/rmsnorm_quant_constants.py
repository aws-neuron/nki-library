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


import numpy as np
from dataclasses import dataclass

import nki
import nki.isa as nisa
import nki.language as nl
from nki.language import NKIObject

from .rmsnorm_quant_tile_info import RMSNormQuantTileInfo


#
#
# Primary tuple that holds miscellaneous constants required by the kernel
#
@dataclass
class RMSNormQuantConstants(NKIObject):
    num_hw_psum_banks: int
    # Compute data type used for matmuls, etc.
    compute_data_type: np.dtype
    # Data type used for quantizing the inputs along with its range
    quant_data_type: np.dtype
    quant_data_type_range: int
    # The number of elements in the output tensor used to store each dequantizing scale factor
    dequant_scale_size: int
    # A small constant used to put an upper bound on the quantization scales for numerical stability.  This number is a minimum
    # applied to the dequantization scales and serves our purpose since quantization scales are computed as 1 / dequantization scale.
    min_dequant_scale_value: float
    # Bias vector of zeros used for activation functions
    outer_dim_tile_zero_bias_vector_sbuf: nl.ndarray
    # Epsilon vector used for numerical stability in RMS norm
    rmsn_eps_bias_sbuf: nl.ndarray
    # Ones vector for broadcasting via the PE
    pe_broadcast_ones_vector_sbuf: nl.ndarray
    # The dimension we use for processing
    outer_dim_size: int
    proc_dim_size: int
    # The currently support max value for S dimension
    MAX_S: int = 32768
    # The currently support max value for H dimension
    MAX_H: int = 16384
    # The currently support max value for B dimension
    MAX_B: int = 2


# Factory method
# ONLY CONSTRUCT THIS USING THE FACTORY METHOD BELOW


def build_rms_norm_quant_constants(
    tile_info: RMSNormQuantTileInfo, eps: float, processing_shape: tuple[int, int]
) -> "RMSNormQuantConstants":
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
    # We do the math here for illustrative purposes plus the assert as a sanity check.
    float32_bytes = 4
    quant_type_bytes = 1  # float8_e4m3 is 1 byte
    assert float32_bytes % quant_type_bytes == 0
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
