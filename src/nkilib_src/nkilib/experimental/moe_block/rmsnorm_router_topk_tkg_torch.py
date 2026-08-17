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

"""PyTorch reference for rmsnorm_router_topk_tkg."""

from typing import Optional

import nki.language as nl
import torch

from ...core.router_topk.router_topk import XSBLayout_tp201__2
from ...core.router_topk.router_topk_torch import router_topk_torch_ref
from ...core.subkernels.rmsnorm_mx_quantize_tkg_torch import rmsnorm_mx_quantize_tkg_wrapper_torch_ref
from ...core.subkernels.rmsnorm_torch import rms_norm_torch_ref
from ...core.utils.common_types import QuantizationType, RouterActFnType

# Map NKI dtypes to torch dtypes for narrowing-precision casts.
_NL_TO_TORCH_DTYPE = {
    nl.bfloat16: torch.bfloat16,
    nl.float16: torch.float16,
    nl.float32: torch.float32,
}


def rmsnorm_router_topk_tkg_torch_ref(
    hidden_states: torch.Tensor,
    gamma: torch.Tensor,
    router_weights: torch.Tensor,
    router_bias: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    hidden_actual: Optional[int] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    router_mm_dtype=nl.bfloat16,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
) -> dict:
    """Torch reference matching rmsnorm_router_topk_tkg outputs.

    Returns dict with keys: norm_output, expert_index, expert_affinities.
    """
    B, S, H = hidden_states.shape
    T = B * S
    router_torch_dtype = _NL_TO_TORCH_DTYPE[router_mm_dtype]

    # Unquantized RMSNorm output drives the router for both NONE and MX paths
    # (the kernel feeds the unquantized SBUF buffer into the router matmul).
    norm_unquant = rms_norm_torch_ref(hidden_states, gamma, eps=eps, hidden_actual=hidden_actual).reshape(T, H)
    norm_for_router = norm_unquant

    if quantization_type == QuantizationType.MX:
        # Use the MX ref to produce the packed FP8 quant‖scales output for validation.
        mx_result = rmsnorm_mx_quantize_tkg_wrapper_torch_ref(
            inp=hidden_states,
            gamma=gamma,
            hidden_actual=hidden_actual,
            output_quant_in_sbuf=False,
            output_quant_packed=True,
            _out_quant_dtype=nl.float8_e4m3fn,
        )
        norm_output = mx_result["out_quant"]
    elif quantization_type == QuantizationType.NONE:
        norm_output = norm_unquant
    else:
        raise ValueError(f"Unsupported quantization_type: {quantization_type}")

    # Cast operands to router_mm_dtype then back to fp32 — applies the cast's precision
    # loss while keeping fp32 matmul accumulation (matching the tensor engine's behavior).
    norm_for_router = norm_for_router.to(router_torch_dtype).float()

    router_outputs = router_topk_torch_ref(
        x=norm_for_router,
        w=router_weights.to(router_torch_dtype).float(),
        w_bias=router_bias.to(router_torch_dtype).float() if router_bias is not None else None,
        router_logits=None,
        expert_affinities=None,
        expert_index=None,
        act_fn=router_act_fn,
        k=top_k,
        x_hbm_layout=1,
        x_sb_layout=XSBLayout_tp201__2,
        router_pre_norm=False,
        norm_topk_prob=False,
    )

    return {
        "norm_output": norm_output,
        "expert_index": router_outputs["expert_index"],
        "expert_affinities": router_outputs["expert_affinities"],
    }
