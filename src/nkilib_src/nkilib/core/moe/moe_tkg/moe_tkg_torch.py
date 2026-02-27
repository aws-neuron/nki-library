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

"""PyTorch reference implementation for MoE TKG kernel."""

import math

import numpy as np
import torch
import torch.nn.functional as F

from ...mlp.mlp_tkg.mlp_proj_mxfp4_torch import (
    down_proj_mxfp4_torch_ref,
    gate_up_proj_mxfp4_torch_ref,
)
from ...utils.common_types import ActFnType
from ...utils.mx_torch_common import (
    quantize_to_mx_fp8,
    unpack_float4_x4,
    unpack_float8_e4m3fn_x4,
)


def moe_tkg_torch_ref(
    hidden_input: torch.Tensor,
    expert_gate_up_weights: torch.Tensor,
    expert_down_weights: torch.Tensor,
    expert_affinities: torch.Tensor,
    expert_index: torch.Tensor,
    is_all_expert: bool,
    rank_id: torch.Tensor = None,
    expert_gate_up_bias: torch.Tensor = None,
    expert_down_bias: torch.Tensor = None,
    expert_gate_up_weights_scale: torch.Tensor = None,
    expert_down_weights_scale: torch.Tensor = None,
    hidden_input_scale: torch.Tensor = None,
    gate_up_input_scale: torch.Tensor = None,
    down_input_scale: torch.Tensor = None,
    mask_unselected_experts: bool = False,
    expert_affinities_eager: torch.Tensor = None,
    expert_affinities_scaling_mode=None,
    activation_fn=None,
    output_dtype=None,
    gate_clamp_upper_limit: float = None,
    gate_clamp_lower_limit: float = None,
    up_clamp_upper_limit: float = None,
    up_clamp_lower_limit: float = None,
    output_in_sbuf: bool = False,
    is_all_expert_dynamic: bool = False,
    block_size: int = None,
) -> dict:
    """
    PyTorch reference implementation of Mixture of Experts Token Generation (MoE TKG).

    Signature matches moe_tkg kernel.

    Args:
        hidden_input: [T, H] input tensor
        expert_gate_up_weights: [E, 2, ...] gate/up projection weights
        expert_down_weights: [E, ...] down projection weights
        expert_affinities: [T, E] expert affinity scores
        expert_index: [T, K] selected expert indices (selective mode)
        is_all_expert: if True, all experts process all tokens; else top-k selective
        rank_id: [T, 1] rank IDs for all-expert affinity scaling
        expert_gate_up_bias: [E, 2, I] optional gate/up bias
        expert_down_bias: [E, H] optional down projection bias
        expert_gate_up_weights_scale: [E, ...] FP8 row scales for gate/up weights
        expert_down_weights_scale: [E, ...] FP8 row scales for down weights
        expert_affinities_scaling_mode: ExpertAffinityScaleMode enum (0=NO_SCALE, 1=POST_SCALE)
        activation_fn: ActFnType enum (0=SiLU, 1=GELU, 2=GELU_Tanh, 3=Swish)
        output_dtype: output tensor dtype
        gate_clamp_upper_limit: upper clamp for gate projection output
        gate_clamp_lower_limit: lower clamp for gate projection output
        up_clamp_upper_limit: upper clamp for up projection output
        up_clamp_lower_limit: lower clamp for up projection output

    Unused params (signature compatibility with kernel):
        hidden_input_scale, gate_up_input_scale, down_input_scale,
        mask_unselected_experts, expert_affinities_eager, output_in_sbuf,
        is_all_expert_dynamic, block_size

    Returns:
        dict with "out" key containing output tensor [T, H]
    """
    # Convert activation_fn enum to string
    act_fn = "silu"
    if activation_fn is not None:
        if hasattr(activation_fn, 'value'):
            act_fn = {0: "silu", 1: "gelu", 2: "gelu_tanh", 3: "swish"}.get(int(activation_fn.value), "silu")
        elif isinstance(activation_fn, int):
            act_fn = {0: "silu", 1: "gelu", 2: "gelu_tanh", 3: "swish"}.get(activation_fn, "silu")

    # Convert scaling mode enum to int
    scale_mode = 0
    if expert_affinities_scaling_mode is not None:
        if hasattr(expert_affinities_scaling_mode, 'value'):
            scale_mode = int(expert_affinities_scaling_mode.value)
        elif isinstance(expert_affinities_scaling_mode, int):
            scale_mode = expert_affinities_scaling_mode

    # Check if FP8 ROW quantization (scale tensors provided, shape [E, 2, I] for gate_up)
    is_fp8_row = (
        expert_gate_up_weights_scale is not None
        and not _is_mx_weight(expert_gate_up_weights)
        and len(expert_gate_up_weights_scale.shape) == 3
    )

    # Check if MX quantization
    is_mx = _is_mx_weight(expert_gate_up_weights)

    if is_mx:
        return _moe_tkg_mx_ref(
            hidden_input=hidden_input,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            is_all_expert=is_all_expert,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            expert_gate_up_bias=expert_gate_up_bias,
            expert_down_bias=expert_down_bias,
            act_fn=act_fn,
            scale_mode=scale_mode,
            dtype=hidden_input.numpy().dtype if isinstance(hidden_input, torch.Tensor) else hidden_input.dtype,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
        )

    T, H = hidden_input.shape
    E = expert_gate_up_weights.shape[0]

    # Initialize output
    output = torch.zeros(T, H, dtype=hidden_input.dtype, device=hidden_input.device)

    if is_all_expert:
        # All-expert mode: process all experts for all tokens
        for e in range(E):
            affinity = expert_affinities[:, e : e + 1]  # [T, 1]
            if affinity.sum() == 0:
                continue

            gate_up_scale = expert_gate_up_weights_scale[e] if is_fp8_row else None
            down_scale = expert_down_weights_scale[e] if is_fp8_row else None

            expert_out = _compute_expert_mlp(
                hidden_input,
                expert_gate_up_weights[e],
                expert_down_weights[e],
                expert_gate_up_bias[e] if expert_gate_up_bias is not None else None,
                expert_down_bias[e] if expert_down_bias is not None else None,
                act_fn,
                gate_clamp_upper_limit,
                up_clamp_upper_limit,
                up_clamp_lower_limit,
                gate_up_scale,
                down_scale,
            )

            # Apply affinity scaling (POST_SCALE mode = 1)
            if scale_mode == 1:
                expert_out = affinity * expert_out

            output = output + expert_out
    else:
        # Selective-expert mode: process only top-k selected experts per token
        T, K = expert_index.shape  # K is top_k
        for t in range(T):
            for k in range(K):
                e = int(expert_index[t, k].item())
                affinity = expert_affinities[t, e].unsqueeze(0).unsqueeze(0)  # [1, 1]

                gate_up_scale = expert_gate_up_weights_scale[e] if is_fp8_row else None
                down_scale = expert_down_weights_scale[e] if is_fp8_row else None

                token_input = hidden_input[t : t + 1]  # [1, H]
                expert_out = _compute_expert_mlp(
                    token_input,
                    expert_gate_up_weights[e],
                    expert_down_weights[e],
                    expert_gate_up_bias[e] if expert_gate_up_bias is not None else None,
                    expert_down_bias[e] if expert_down_bias is not None else None,
                    act_fn,
                    gate_clamp_upper_limit,
                    up_clamp_upper_limit,
                    up_clamp_lower_limit,
                    gate_up_scale,
                    down_scale,
                )

                # Apply affinity scaling (POST_SCALE mode = 1)
                if scale_mode == 1:
                    expert_out = affinity * expert_out

                output[t] = output[t] + expert_out.squeeze(0)

    return {"out": output}


def _is_mx_weight(weight):
    """Check if weight is MX packed x4 format (numpy array with x4 dtype)."""
    return isinstance(weight, np.ndarray) and 'x4' in str(weight.dtype)


def _to_numpy(t):
    """Convert torch tensor or numpy array to numpy."""
    if isinstance(t, torch.Tensor):
        return t.numpy()
    return t


def _compute_expert_mlp(
    hidden_input,
    gate_up_weight,
    down_weight,
    gate_up_bias,
    down_bias,
    act_fn,
    gate_clamp_upper,
    up_clamp_upper,
    up_clamp_lower,
    gate_up_scale=None,
    down_scale=None,
):
    """Compute MLP for a single expert.

    For FP8 ROW quantization:
    - gate_up_scale: [2, I] - per-row scale for gate and up weights
    - down_scale: [H] - per-row scale for down weights
    """
    gate_weight = gate_up_weight[:, 0, :]  # [H, I]
    up_weight = gate_up_weight[:, 1, :]  # [H, I]

    # Dequantize weights if FP8 ROW
    if gate_up_scale is not None:
        # gate_up_scale shape: [2, I], broadcast to [H, I]
        gate_weight = gate_weight.float() * gate_up_scale[0:1, :]  # [H, I] * [1, I]
        up_weight = up_weight.float() * gate_up_scale[1:2, :]

    # Gate projection
    gate_out = torch.matmul(hidden_input.float(), gate_weight.float())
    if gate_up_bias is not None:
        gate_out = gate_out + gate_up_bias[0, :]

    if gate_clamp_upper is not None:
        gate_out = torch.clamp(gate_out, max=gate_clamp_upper)

    # Apply activation
    if act_fn == "silu":
        gate_out = F.silu(gate_out)
    elif act_fn == "swish":
        gate_out = gate_out * torch.sigmoid(1.702 * gate_out)
    elif act_fn == "gelu":
        gate_out = F.gelu(gate_out)
    elif act_fn == "gelu_tanh":
        gate_out = F.gelu(gate_out, approximate="tanh")

    # Up projection
    up_out = torch.matmul(hidden_input.float(), up_weight.float())
    if gate_up_bias is not None:
        up_out = up_out + gate_up_bias[1, :]

    if up_clamp_upper is not None or up_clamp_lower is not None:
        up_out = torch.clamp(
            up_out,
            min=up_clamp_lower if up_clamp_lower is not None else float('-inf'),
            max=up_clamp_upper if up_clamp_upper is not None else float('inf'),
        )

    # Element-wise multiply and down projection
    intermediate = gate_out * up_out

    # Dequantize down weights if FP8 ROW
    if down_scale is not None:
        # down_scale shape: [H], broadcast to [I, H]
        down_weight = down_weight.float() * down_scale.unsqueeze(0)  # [I, H] * [1, H]

    expert_out = torch.matmul(intermediate, down_weight.float())
    if down_bias is not None:
        expert_out = expert_out + down_bias

    return expert_out.to(hidden_input.dtype)


def _moe_tkg_mx_ref(
    hidden_input,
    expert_gate_up_weights,
    expert_down_weights,
    expert_affinities,
    expert_index,
    is_all_expert,
    expert_gate_up_weights_scale,
    expert_down_weights_scale,
    expert_gate_up_bias,
    expert_down_bias,
    act_fn,
    scale_mode,
    dtype,
    gate_clamp_upper_limit,
    gate_clamp_lower_limit,
    up_clamp_upper_limit,
    up_clamp_lower_limit,
):
    """MX quantization path using mlp_proj_mxfp4_torch helpers.

    Reuses gate_up_proj_mxfp4_torch_ref for gate/up projections and
    down_proj_mxfp4_torch_ref for down projection. Hidden is quantized
    to float8_e4m3fn_x4 (matching kernel behavior).
    """

    inp = _to_numpy(hidden_input)
    affinities = _to_numpy(expert_affinities)
    exp_idx = _to_numpy(expert_index)
    gate_up_w = expert_gate_up_weights
    down_w = expert_down_weights
    gate_up_w_scale = expert_gate_up_weights_scale
    down_w_scale = expert_down_weights_scale
    gate_up_b = expert_gate_up_bias
    down_b = expert_down_bias

    act_fn_map = {
        "silu": ActFnType.SiLU,
        "swish": ActFnType.Swish,
        "gelu": ActFnType.GELU,
        "gelu_tanh": ActFnType.GELU_Tanh_Approx,
    }
    act_fn_type = act_fn_map.get(act_fn, ActFnType.SiLU)

    act_fns = {
        ActFnType.SiLU: lambda x: x * (1 / (1 + np.exp(-x))),
        ActFnType.Swish: lambda x: x * (1 / (1 + np.exp(-1.702 * x))),
        ActFnType.GELU: lambda x: 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2))),
        ActFnType.GELU_Tanh_Approx: lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
    }
    act_fn_func = act_fns[act_fn_type]

    T, H = inp.shape
    E = gate_up_w.shape[0]
    I = gate_up_w.shape[-1]
    _pmax = 128
    _q_width = 4
    n_H512 = H // _pmax // _q_width
    result = np.zeros((T, H), dtype=np.float32)

    # Select weight unpack function based on dtype
    is_float4 = 'float4' in str(gate_up_w.dtype)
    w_unpack = unpack_float4_x4 if is_float4 else unpack_float8_e4m3fn_x4

    def _compute_one_expert(active_in, expert_idx):
        BxS = active_in.shape[0]
        # Quantize hidden to mxfp8 (float8_e4m3fn_x4)
        h = active_in.reshape(BxS, _q_width, n_H512, _pmax).transpose(3, 2, 0, 1).reshape(_pmax, -1)
        hidden_mx, hidden_scale = quantize_to_mx_fp8(h)
        # Reshape to [_pmax, n_H512, BxS] matching gate_up_proj_mxfp4_torch_ref input
        hidden_mx = hidden_mx.reshape(_pmax, n_H512, BxS)
        hidden_scale_t = torch.from_numpy(hidden_scale.reshape(_pmax // 8, n_H512, BxS))

        # Gate projection
        gw = gate_up_w[expert_idx][:, 0, :, :]
        gs_t = gate_up_w_scale[expert_idx][:, 0, :, :]
        gb_t = gate_up_b[expert_idx][:, 0, :, :].float() if gate_up_b is not None else None
        gate_out = gate_up_proj_mxfp4_torch_ref(
            hidden_mx,
            hidden_scale_t,
            gw,
            gs_t,
            gb_t,
            H,
            I,
            BxS,
            hidden_unpack_fn=unpack_float8_e4m3fn_x4,
            weight_unpack_fn=w_unpack,
        )["out"].numpy()

        # Up projection
        uw = gate_up_w[expert_idx][:, 1, :, :]
        us_t = gate_up_w_scale[expert_idx][:, 1, :, :]
        ub_t = gate_up_b[expert_idx][:, 1, :, :].float() if gate_up_b is not None else None
        up_out = gate_up_proj_mxfp4_torch_ref(
            hidden_mx,
            hidden_scale_t,
            uw,
            us_t,
            ub_t,
            H,
            I,
            BxS,
            hidden_unpack_fn=unpack_float8_e4m3fn_x4,
            weight_unpack_fn=w_unpack,
        )["out"].numpy()

        # Clamp
        if up_clamp_upper_limit is not None:
            up_out = np.minimum(up_out, up_clamp_upper_limit)
        if up_clamp_lower_limit is not None:
            up_out = np.maximum(up_out, up_clamp_lower_limit)
        if gate_clamp_upper_limit is not None:
            gate_out = np.minimum(gate_out, gate_clamp_upper_limit)
        if gate_clamp_lower_limit is not None:
            gate_out = np.maximum(gate_out, gate_clamp_lower_limit)

        # Activation + multiply
        mult = torch.from_numpy(act_fn_func(gate_out) * up_out)

        # Down projection
        dw = down_w[expert_idx]
        ds_t = down_w_scale[expert_idx]
        db_t = down_b[expert_idx].float() if down_b is not None else None
        return down_proj_mxfp4_torch_ref(mult, dw, ds_t, db_t, H, I, BxS, weight_unpack_fn=w_unpack)["out"].numpy()

    if is_all_expert:
        for e in range(E):
            for t in range(T):
                token_out = _compute_one_expert(inp[t : t + 1], e).reshape(H)
                if scale_mode == 1:
                    token_out = token_out * affinities[t, e]
                result[t] += token_out
    else:
        T_idx, K = exp_idx.shape
        for t in range(T_idx):
            for k in range(K):
                e = int(exp_idx[t, k])
                token_out = _compute_one_expert(inp[t : t + 1], e).reshape(H)
                if scale_mode == 1:
                    token_out = token_out * affinities[t, e]
                result[t] += token_out

    return {"out": result.astype(dtype)}
