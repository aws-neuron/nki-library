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

"""PyTorch reference for the LNC1 blockwise-MX MoE CTE kernel (``bwmm_block_mx``).

Delegates to the SHARED numpy/torch oracle ``bwmm_mx_blockwise_loop`` -- the same one the
LNC2 ``bwmm_shard_on_block_mx_torch_ref`` uses -- so there is no second implementation of the
MoE math to keep in sync. The ONLY difference is the output shape:

``bwmm_mx_blockwise_loop`` allocates ``[lnc_degree=2, T+1, H]`` when ``separate_outputs=True``
and ``[T+1, H]`` when False. Crucially it accumulates into shard 0 ONLY
(``output[0, local_ids, :] += ...``); shard 1 is never written, because the oracle models the
POST-reduce result that the LNC2 kernel produces by merging its two half-outputs. With one
core there is nothing to merge, so the LNC1 kernel returns a plain ``[T, H]`` buffer and this
reference passes ``separate_outputs=False`` -- yielding bit-identical values in the
un-sharded shape.

The signature mirrors ``bwmm_block_mx`` exactly (parameter NAMES must match for
``test/unit/test_torch_ref_dispatch_audit.py``).
"""

from typing import Any

import nki.language as nl
import torch

from ....core.mlp.mlp_parameters import MLPQuantizationParameters
from ....core.moe.moe_cte.bwmm_mx_torch_common import bwmm_mx_blockwise_loop
from ....core.utils.common_types import ActFnType, ExpertAffinityScaleMode, QuantizationType
from ._utils import SkipMode


def bwmm_block_mx_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    conditions: torch.Tensor | None = None,
    gate_and_up_proj_bias: torch.Tensor | None = None,
    down_proj_bias: torch.Tensor | None = None,
    gate_up_proj_scale: torch.Tensor | None = None,
    down_proj_scale: torch.Tensor | None = None,
    block_size: int | None = None,
    n_static_blocks: int = -1,
    n_dynamic_blocks: int = 55,
    top_k: int = 1,
    ep_degree: int = 1,
    gate_up_activations_T: torch.Tensor | None = None,
    down_activations: torch.Tensor | None = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype: Any = nl.bfloat16,
    weight_dtype: Any = None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: float | None = None,
    gate_clamp_lower_limit: float | None = None,
    up_clamp_lower_limit: float | None = None,
    up_clamp_upper_limit: float | None = None,
    # Declared for exact kernel-vs-ref signature parity (test_torch_ref_dispatch_audit).
    # The oracle reads the DENSE scale tensors either way -- packing is a pure HBM layout
    # change on the kernel side -- so it is accepted and ignored here.
    use_packed_scales: bool = False,  # noqa: ARG001 - kernel-side parity
    quantization_type: QuantizationType = QuantizationType.NONE,
    gate_up_in_scale: torch.Tensor | None = None,
    down_in_scale: torch.Tensor | None = None,
) -> dict:
    """PyTorch reference for bwmm_block_mx. Signature matches the kernel exactly.

    Returns ``{"output": [T, H]}`` -- single-core, so no shard dimension.
    """
    quant_params = None
    if quantization_type == QuantizationType.STATIC_MX:
        """
        STATIC_MX reuses gate_up_proj_scale / down_proj_scale to carry the per-expert weight scales (gate/up packed [E,
        2, 1], down [E, 1]). Split the packed gate/up tensor into separate [E, 1] gate and up views (idx 0 = gate, idx 1
        = up).
        """
        quant_params = MLPQuantizationParameters(
            quantization_type=quantization_type,
            gate_w_scale=gate_up_proj_scale[:, 0],
            up_w_scale=gate_up_proj_scale[:, 1],
            down_w_scale=down_proj_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            clipping_bound=0.0,
        )
    return bwmm_mx_blockwise_loop(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        block_size=block_size,
        activation_function=activation_function,
        skip_dma=skip_dma,
        weight_dtype=weight_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        conditions=conditions,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        # LNC1: no shard dim. The oracle only ever accumulates into shard 0, so this yields
        # the same values as the LNC2 ref, shaped [T, H] instead of [2, T, H].
        separate_outputs=False,
        quant_params=quant_params,
    )
