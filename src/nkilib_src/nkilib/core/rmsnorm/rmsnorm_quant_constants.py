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

from ..utils.common_types import DtypeMode
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_max_positive_value_for_dtype, resolve_fp8_e4m3_dtype
from .rmsnorm_quant_tile_info import RMSNormQuantTileInfo


@dataclass
class RMSNormQuantConstants(nl.NKIObject):
    """
    Constants required by the RMS Norm Quantization kernel.

    Args:
        num_hw_psum_banks (int): Number of hardware PSUM banks
        compute_data_type (np.dtype): Data type for matmuls and compute
        quant_data_type (np.dtype): Data type for quantized output
        quant_data_type_range (float): Maximum representable value in quant dtype (240.0 for ``nl.float8_e4m3``, 448.0 for ``nl.float8_e4m3fn``)
        dequant_scale_size (int): Number of output elements for dequant scale
        min_dequant_scale_value (float): Minimum dequant scale for numerical stability
        outer_dim_tile_zero_bias_vector_sbuf (nl.NkiTensor): Zero bias vector in SBUF
        rmsn_eps_bias_sbuf (nl.NkiTensor): Epsilon bias vector for RMS norm
        pe_broadcast_ones_vector_sbuf (nl.NkiTensor): Ones vector for PE broadcasting
        outer_dim_size (int): Size of outer dimension
        proc_dim_size (int): Size of processing dimension
        MAX_S (int): Maximum supported sequence length
        MAX_H (int): Maximum supported hidden dimension
        MAX_B (int): Maximum supported batch size
    """

    num_hw_psum_banks: int
    compute_data_type: np.dtype
    quant_data_type: np.dtype
    quant_data_type_range: float
    dequant_scale_size: int
    min_dequant_scale_value: float
    outer_dim_tile_zero_bias_vector_sbuf: nl.NkiTensor
    rmsn_eps_bias_sbuf: nl.NkiTensor
    pe_broadcast_ones_vector_sbuf: nl.NkiTensor
    outer_dim_size: int
    proc_dim_size: int
    MAX_S: int = 32768
    MAX_H: int = 16384
    MAX_B: int = 2


def build_rms_norm_quant_constants(
    tile_info: RMSNormQuantTileInfo,
    eps: float,
    processing_shape: tuple[int, int],
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> RMSNormQuantConstants:
    """
    Factory method to construct RMSNormQuantConstants.

    Args:
        tile_info (RMSNormQuantTileInfo): Tile configuration info
        eps (float): Epsilon value for numerical stability
        processing_shape (tuple[int, int]): (outer_dim_size, proc_dim_size)
        dtype_mode (DtypeMode): Explicit FP8 E4M3 dtype selection for the
            quantized output.
            - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240).
            - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448). TRN3 only.
            - ``DtypeMode.AUTO``: ``nl.float8_e4m3fn`` on TRN3, ``nl.float8_e4m3``
              elsewhere (resolved at kernel trace time).

    Returns:
        RMSNormQuantConstants: Initialized constants for the kernel
    """
    # TODO: Get this constant from the NKI API once it is available
    num_hw_psum_banks = 8
    # Data types
    compute_data_type = nl.bfloat16
    quant_data_type = resolve_fp8_e4m3_dtype(dtype_mode)
    quant_data_type_range = get_max_positive_value_for_dtype(quant_data_type)
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
