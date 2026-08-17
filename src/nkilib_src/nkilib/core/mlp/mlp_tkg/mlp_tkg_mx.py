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

"""MLP TKG MX quantization dispatcher.

Routes to the appropriate quantization-specific implementation based on
the quantization type in MLPParameters:
- MX (hardware): mlp_tkg_quad_fp8_mx
- STATIC_MX (tensor-wise): mlp_tkg_quad_fp8_static
- ROW_MX (row-wise): mlp_tkg_quad_fp8_row
"""

import nki.language as nl

from ...utils.kernel_assert import kernel_assert
from ..mlp_parameters import MLPParameters

# Re-export for backward compatibility
from .mlp_tkg_quad_fp8_mx import (
    _mlp_tkg_hw_mx_impl,  # noqa: F401
    mlp_tkg_quad_fp8_mx,
)
from .mlp_tkg_quad_fp8_row import (
    RowMxPreallocBuffers,  # noqa: F401
    _mlp_tkg_row_mx_impl,  # noqa: F401
    mlp_tkg_quad_fp8_row,
)
from .mlp_tkg_quad_fp8_static import (
    StaticMxPreallocBuffers,  # noqa: F401
    _mlp_tkg_static_mx_impl,  # noqa: F401
    mlp_tkg_quad_fp8_static,
)

# Backward-compatible alias
MxPreallocBuffers = StaticMxPreallocBuffers


def mlp_tkg_mx(
    params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
) -> list[nl.NkiTensor]:
    """
    Dispatcher that routes to the appropriate MX quantization implementation.

    Selects among three MX quantization variants based on ``params.quant_params``:
      - MX (hardware):    ``mlp_tkg_quad_fp8_mx``
      - STATIC_MX:        ``mlp_tkg_quad_fp8_static``
      - ROW_MX:           ``mlp_tkg_quad_fp8_row``

    Args:
        params (MLPParameters): MLP configuration. ``params.quant_params`` must
            report one of MX, STATIC_MX, or ROW_MX quantization.
        output_tensor_hbm (nl.NkiTensor): [B, S, H], Output tensor in HBM.
        output_stored_add_tensor_hbm (nl.NkiTensor): Optional fused-add output in HBM.

    Returns:
        list[nl.NkiTensor]: The output list returned by the selected implementation.
            Typically ``[output_tensor_hbm]`` or ``[output_tensor_hbm, output_stored_add_tensor_hbm]``
            when fused add is stored; may be SBUF-resident if ``store_output_in_sbuf=True``.

    Notes:
        - Raises via ``kernel_assert`` if quantization is not MX, STATIC_MX, or ROW_MX.
        - This function is an undecorated sub-kernel; it is called from the top-level
          ``@nki.jit`` MLP kernel.
    """

    if params.quant_params.is_quant_mx():
        return mlp_tkg_quad_fp8_mx(params, output_tensor_hbm, output_stored_add_tensor_hbm)
    elif params.quant_params.is_quant_static_mx():
        return mlp_tkg_quad_fp8_static(params, output_tensor_hbm, output_stored_add_tensor_hbm)
    elif params.quant_params.is_quant_row_mx():
        return mlp_tkg_quad_fp8_row(params, output_tensor_hbm, output_stored_add_tensor_hbm)
    else:
        kernel_assert(False, "mlp_tkg_mx requires MX, STATIC_MX, or ROW_MX quantization")
