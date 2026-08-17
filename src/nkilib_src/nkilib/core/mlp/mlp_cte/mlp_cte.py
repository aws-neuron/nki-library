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

"""MLP CTE kernel router - dispatches to basic or MX implementation based on quantization type."""

import nki.language as nl

from ..mlp_parameters import MLPParameters
from .basic.mlp_cte_basic import mlp_cte_basic
from .mx.mlp_cte_mx import mlp_cte_mx


def mlp_cte(
    mlp_params: MLPParameters,
    output_tensor_hbm: nl.NkiTensor,
    output_stored_add_tensor_hbm: nl.NkiTensor,
):
    """
    MLP Context Encoding (CTE) kernel with SPMD support and automatic sharding.

    Routes to either the basic implementation (for NONE, ROW, STATIC quantization)
    or the MX implementation (for MX, STATIC_MX, ROW_MX quantization).

    Args:
        mlp_params (MLPParameters): Complete MLP configuration parameters including
            hidden_tensor, weight tensors, normalization params, and quantization params
        output_tensor_hbm (nl.NkiTensor): [B, S, H], Pre-allocated HBM output tensor for MLP results
        output_stored_add_tensor_hbm (nl.NkiTensor): [B, S, H], Pre-allocated HBM tensor for
            fused addition results (optional, can be None)
    """
    if mlp_params.quant_params.is_dtype_mx():
        mlp_cte_mx(mlp_params, output_tensor_hbm, output_stored_add_tensor_hbm)
    else:
        mlp_cte_basic(mlp_params, output_tensor_hbm, output_stored_add_tensor_hbm)
