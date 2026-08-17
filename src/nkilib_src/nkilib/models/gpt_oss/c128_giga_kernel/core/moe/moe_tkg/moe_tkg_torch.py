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

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import torch
import torch.nn.functional as F

from ...utils.common_types import ActFnType, DtypeMode, MoEAllToAllVStrategy
from ...utils.kernel_helpers import get_max_positive_value_for_dtype
from ...utils.mx_torch_common import (
    quantize_to_mx,
    unpack_float4_x4,
    unpack_float8_e4m3fn_x4,
)
from .mlp_proj_mx_torch import (
    down_proj_mx_torch_ref,
    gate_up_proj_mx_torch_ref,
)

_TORCH_NP_FP8_DTYPE_MAP = {
    torch.float8_e4m3fn: nl.float8_e4m3fn,
    torch.float8_e5m2: nl.float8_e5m2,
}

# Clamp bound for np.exp to prevent float32 overflow (exp(88) ≈ 1.6e38, exp(89) overflows).
# For |x| > 88 the sigmoid is effectively 0 or 1, so clamping has no numerical impact.
_EXP_CLAMP = 88

# Safe magnitude for float32 multiply operands: sqrt(FLT_MAX) ensures a*b won't overflow.
_F32_SAFE = np.sqrt(np.finfo(np.float32).max)


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
    expert_gate_up_input_scale: torch.Tensor = None,
    expert_down_input_scale: torch.Tensor = None,
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
    input_dequant_scale: torch.Tensor = None,
    all_to_all_v_strategy: MoEAllToAllVStrategy = MoEAllToAllVStrategy.DISABLED,
    output_layout=None,
    output: torch.Tensor = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> dict:
    """
    PyTorch reference implementation of Mixture of Experts Token Generation (MoE TKG).

    Signature matches moe_tkg kernel.

    Args:
        hidden_input: [T, H] or [T, H_concat] input tensor.
            When all_to_all_v_strategy != DISABLED, [T, H_concat] layout is expected with fp8 dtype,
            where H_concat = H + H/4 + E_L * 2 + 4 (hidden_quant | hidden_scale | expert_affinities | token_indices).
        expert_gate_up_weights: [E, 2, ...] gate/up projection weights
        expert_down_weights: [E, ...] down projection weights
        expert_affinities: [T, E] expert affinity scores (None when all_to_all_v_strategy != DISABLED)
        expert_index: [T, K] selected expert indices (None when all_to_all_v_strategy != DISABLED)
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
        all_to_all_v_strategy (MoEAllToAllVStrategy): Input/output permutation strategy when all_to_all_v (A2A-v) is used.
            Currently only supported on Trn3 with MX weights.
            - DISABLED: Default; A2A-v is not used.
            - PRESERVE_ROW_ORDER: Output row ordering matches input row ordering. Token indices are appended as trailing 2 columns of output.
            - PACK_OUTPUT_ROWS: Output rows are packed, with routed tokens placed in the first N rows, where N is the number of routed tokens.
                Final T-N rows are padded with 0s. Token indices are appended as trailing 2 columns of output.
                When this strategy is used, the final 4 elements of hidden_input must be 0 for all padded rows.

    Unused params (signature compatibility with kernel):
        hidden_input_scale, mask_unselected_experts, expert_affinities_eager,
        output_in_sbuf, is_all_expert_dynamic, block_size, input_dequant_scale

    Returns:
        dict with "out" key containing output tensor [T, H]
    """
    # OCP → 448, NON_OCP → 240. Callers pre-resolve AUTO (see
    # ``resolve_dtype_mode_for_torch_ref``); the torch ref runs on CPU and can't
    # query hardware directly. The clip must match the kernel's FP8 range so
    # goldens align with the subkernel output.
    assert dtype_mode != DtypeMode.AUTO, (  # noqa: S101
        "moe_tkg_torch_ref requires DtypeMode.AUTO to be pre-resolved by the caller."
    )
    if dtype_mode == DtypeMode.OCP:
        _fp8_max = get_max_positive_value_for_dtype(nl.float8_e4m3fn)
    else:
        _fp8_max = get_max_positive_value_for_dtype(nl.float8_e4m3)

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
        and expert_gate_up_weights_scale.shape[-1] > 1
    )

    # Check if FP8 STATIC quantization (scale tensors + input scales provided, non-MX)
    is_fp8_static = (
        not _is_mx_weight(expert_gate_up_weights)
        and expert_gate_up_input_scale is not None
        and expert_down_input_scale is not None
    )

    # Check if MX quantization
    is_mx = _is_mx_weight(expert_gate_up_weights)
    is_static_mx = is_mx and expert_gate_up_input_scale != None and expert_down_input_scale != None
    is_row_quant = (
        is_mx
        and expert_gate_up_weights_scale is not None
        and expert_down_weights_scale is not None
        and not is_static_mx
        and _to_numpy(expert_gate_up_weights_scale).dtype == np.float32
        and expert_gate_up_weights_scale.shape[-1] > 1
    )

    # Resolve output dtype: use output_dtype if provided, otherwise fall back to input dtype
    # FIXME[accuracy]: propagate output dtype to non-MX path
    output_dtype = _resolve_output_dtype(hidden_input, output_dtype)

    # Shapes
    T, H = hidden_input.shape
    E = expert_gate_up_weights.shape[0]

    # Unpack input for a2av - a2av input is packed as [hidden_quant, hidden_scale, expert_affinites, token_indices]
    if all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED:
        hidden_input_scale = None
        token_indices = None
    else:
        _EXPECTED_A2AV_DTYPES = [torch.float8_e4m3fn, torch.float8_e5m2, torch.bfloat16, torch.float16, torch.float32]
        assert (  # noqa: S101
            hidden_input.dtype in _EXPECTED_A2AV_DTYPES
        ), f"Expected hidden_input.dtype in {_EXPECTED_A2AV_DTYPES}, got {hidden_input.dtype=}"

        H_actual = expert_down_weights.shape[-1]
        _pmax = 128
        _q_width = 4
        _q_height = 8
        n_H512 = H_actual // _pmax // _q_width
        hidden_input_concat = hidden_input.clone()

        # Non-MX A2AV: bf16/fp16 concatenated input [T, H + E_L + 2]
        is_non_mx_a2av = hidden_input.dtype in (torch.bfloat16, torch.float16, torch.float32)
        if is_non_mx_a2av:
            E_L = expert_gate_up_weights.shape[0]
            hidden_input = hidden_input_concat[:, :H_actual]
            hidden_input_scale = None
            expert_affinities = hidden_input_concat[:, H_actual : H_actual + E_L].to(torch.float32)
            token_indices = hidden_input_concat[:, H_actual + E_L : H_actual + E_L + 2].clone().view(torch.int32)
            H = H_actual
        else:
            affinities_offset = H_actual + H_actual // 4

            # Slice first H columns of hidden_input_concat, convert to np, perform [T, H/512 * 128_H * 4_H]f8 -> reshape/bitcast [T, H/512, 128_H]f8x4 -> transpose [128_H, H/512, T]f8x4
            hidden_input = (
                hidden_input_concat[:, :H_actual]
                .view(torch.uint8)
                .numpy()
                .view(_TORCH_NP_FP8_DTYPE_MAP[hidden_input_concat.dtype])
            )
            hidden_input = hidden_input.view(nl.float8_e4m3fn_x4)
            hidden_input = hidden_input.reshape(T, n_H512, _pmax).transpose(2, 1, 0)

            # Slice cols H : H + H/4, perform [T, H/512 * 128_H]f8 -> transpose [128_H, H/512, T]u8 -> unstride [16_H, H/512, T]u8
            _n_q_blocks_per_col = _pmax // _q_height
            hidden_input_scale = torch.zeros(_pmax // _n_q_blocks_per_col, n_H512, T, dtype=torch.uint8)
            hidden_input_scale_strided = (
                hidden_input_concat[:, H_actual : H_actual + H_actual // 4]
                .view(torch.uint8)
                .reshape(T, n_H512, _pmax)
                .permute(2, 1, 0)
            )
            unstride_indices = (torch.arange(4) + torch.arange(4).unsqueeze(1) * 32).flatten()
            hidden_input_scale = hidden_input_scale_strided[unstride_indices, :, :]

            # View affinities as bf16, upcast to fp32, convert to numpy.
            # NOTE: right now, affinities are hardcoded to be bf16.
            expert_affinities = (
                hidden_input_concat[:, affinities_offset : affinities_offset + 2 * E]
                .clone()
                .view(torch.bfloat16)
                .to(torch.float32)
                .numpy()
            )

            # FIXME: no magic numbers
            token_indices = hidden_input_concat[:, -4:].clone().view(torch.int32)

    if is_row_quant:
        return _moe_tkg_row_mx_ref(
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
            dtype=output_dtype,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
        )

    if is_static_mx:
        return _moe_tkg_static_mx_ref(
            hidden_input=hidden_input,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            is_all_expert=is_all_expert,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            expert_gate_up_input_scale=expert_gate_up_input_scale,
            expert_down_input_scale=expert_down_input_scale,
            expert_gate_up_bias=expert_gate_up_bias,
            expert_down_bias=expert_down_bias,
            act_fn=act_fn,
            scale_mode=scale_mode,
            dtype=output_dtype,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
        )

    if is_mx:
        return _moe_tkg_mx_ref(
            hidden_input=hidden_input,
            hidden_input_scale=hidden_input_scale,
            expert_gate_up_weights=expert_gate_up_weights,
            expert_down_weights=expert_down_weights,
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            is_all_expert=is_all_expert,
            expert_gate_up_weights_scale=expert_gate_up_weights_scale,
            expert_down_weights_scale=expert_down_weights_scale,
            expert_gate_up_bias=expert_gate_up_bias,
            expert_down_bias=expert_down_bias,
            token_indices=token_indices,
            act_fn=act_fn,
            scale_mode=scale_mode,
            dtype=output_dtype,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            all_to_all_v_strategy=all_to_all_v_strategy,
        )

    # Initialize output
    output = torch.zeros(T, H, dtype=hidden_input.dtype, device=hidden_input.device)

    if is_all_expert:
        # All-expert mode: process all experts for all tokens
        for e in range(E):
            affinity = expert_affinities[:, e : e + 1]  # [T, 1]
            if affinity.sum() == 0:
                continue

            gate_up_scale = expert_gate_up_weights_scale[e] if (is_fp8_row or is_fp8_static) else None
            down_scale = expert_down_weights_scale[e] if (is_fp8_row or is_fp8_static) else None

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
                fp8_max=_fp8_max,
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

                gate_up_scale = expert_gate_up_weights_scale[e] if (is_fp8_row or is_fp8_static) else None
                down_scale = expert_down_weights_scale[e] if (is_fp8_row or is_fp8_static) else None

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
                    fp8_max=_fp8_max,
                )

                # Apply affinity scaling (POST_SCALE mode = 1)
                if scale_mode == 1:
                    expert_out = affinity * expert_out

                output[t] = output[t] + expert_out.squeeze(0)

    # For A2AV, handle output packing and token index concatenation
    if all_to_all_v_strategy == MoEAllToAllVStrategy.PACK_OUTPUT_ROWS and token_indices is not None:
        # Pack: move routed tokens (nonzero affinity) to first N rows
        routed_mask = expert_affinities.sum(dim=1) != 0
        count = int(routed_mask.sum().item())
        packed_output = torch.zeros_like(output)
        packed_output[:count] = output[routed_mask]
        # Token indices: pack routed ones to first N rows
        packed_token_indices = torch.zeros_like(token_indices)
        packed_token_indices[:count] = token_indices[routed_mask]
        output = torch.cat([packed_output, packed_token_indices.view(output.dtype)], dim=1)
    elif all_to_all_v_strategy != MoEAllToAllVStrategy.DISABLED and token_indices is not None:
        output = torch.cat([output, token_indices.view(output.dtype)], dim=1)

    return {"out": output}


def _is_mx_weight(weight):
    """Check if weight is MX packed x4 format (numpy array with x4 dtype)."""
    return isinstance(weight, np.ndarray) and 'x4' in str(weight.dtype)


def _to_numpy(t):
    """Convert torch tensor or numpy array to numpy."""
    if isinstance(t, torch.Tensor):
        return t.numpy()
    return t


def _resolve_output_dtype(hidden_input, output_dtype):
    """Resolve output dtype. When output_dtype is None, fallback to hidden input dtype"""

    if output_dtype is not None:
        return output_dtype
    elif isinstance(hidden_input, torch.Tensor):
        # For fp8, use the specified torch/numpy conversion map; otherwise, call .numpy().dtype directly
        return _TORCH_NP_FP8_DTYPE_MAP.get(hidden_input.dtype, hidden_input.numpy().dtype)
    else:
        return hidden_input.dtype


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
    gate_up_in_scale=None,
    down_in_scale=None,
    fp8_max: float = 240.0,
):
    """Compute MLP for a single expert.

    For FP8 ROW quantization:
    - gate_up_scale: [2, I] - per-row scale for gate and up weights
    - down_scale: [H] - per-row scale for down weights

    For FP8 STATIC quantization:
    - gate_up_scale: [2, 1] - per-tensor scale for gate and up weights
    - down_scale: [1] - per-tensor scale for down weights
    - gate_up_in_scale: [1] - per-tensor input scale for gate/up projection
    - down_in_scale: [1] - per-tensor input scale for down projection
    """
    FP8_E4M3_MAX = fp8_max
    gate_weight = gate_up_weight[:, 0, :]  # [H, I]
    up_weight = gate_up_weight[:, 1, :]  # [H, I]

    # Dequantize weights if FP8 ROW or STATIC
    if gate_up_scale is not None:
        gate_weight = gate_weight.float() * gate_up_scale[0:1, :]
        up_weight = up_weight.float() * gate_up_scale[1:2, :]

    # STATIC: quantize input for gate/up projection
    inp = hidden_input.float()
    if gate_up_in_scale is not None:
        in_scale = float(gate_up_in_scale.reshape(-1)[0])
        inp = (inp / in_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    # Gate projection
    gate_out = torch.matmul(inp, gate_weight.float())
    if gate_up_in_scale is not None:
        gate_out = gate_out * in_scale
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
    up_out = torch.matmul(inp, up_weight.float())
    if gate_up_in_scale is not None:
        up_out = up_out * in_scale
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

    # Dequantize down weights if FP8 ROW or STATIC
    if down_scale is not None:
        down_weight = down_weight.float() * down_scale.unsqueeze(0)

    # STATIC: quantize intermediate for down projection
    if down_in_scale is not None:
        dn_scale = float(down_in_scale.reshape(-1)[0])
        intermediate = (intermediate / dn_scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX)

    expert_out = torch.matmul(intermediate, down_weight.float())
    if down_in_scale is not None:
        expert_out = expert_out * dn_scale
    if down_bias is not None:
        expert_out = expert_out + down_bias

    return expert_out.to(hidden_input.dtype)


def _moe_tkg_mx_ref(
    hidden_input,
    hidden_input_scale,
    expert_gate_up_weights,
    expert_down_weights,
    expert_affinities,
    expert_index,
    is_all_expert,
    expert_gate_up_weights_scale,
    expert_down_weights_scale,
    expert_gate_up_bias,
    expert_down_bias,
    token_indices,
    act_fn,
    scale_mode,
    dtype,
    gate_clamp_upper_limit,
    gate_clamp_lower_limit,
    up_clamp_upper_limit,
    up_clamp_lower_limit,
    all_to_all_v_strategy,
):
    """MX quantization path using mlp_proj_mx_torch helpers.

    Reuses gate_up_proj_mx_torch_ref for gate/up projections and
    down_proj_mx_torch_ref for down projection. Hidden is quantized
    to float8_e4m3fn_x4 (matching kernel behavior).
    """

    inp = _to_numpy(hidden_input)
    inp_scale = hidden_input_scale
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
        ActFnType.SiLU: lambda x: x * (1 / (1 + np.exp(-np.clip(x, -_EXP_CLAMP, _EXP_CLAMP)))),
        ActFnType.Swish: lambda x: x * (1 / (1 + np.exp(np.clip(-1.702 * x, -_EXP_CLAMP, _EXP_CLAMP)))),
        ActFnType.GELU: lambda x: 0.5 * x * (1 + np.vectorize(math.erf)(x / np.sqrt(2))),
        ActFnType.GELU_Tanh_Approx: lambda x: 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3))),
    }
    act_fn_func = act_fns[act_fn_type]

    # T is outermost dim when inp is unquantized, and innermost when imp is quantized
    T = BxS = inp.shape[0] if inp_scale is None else inp.shape[-1]
    E = gate_up_w.shape[0]
    I = gate_up_w.shape[-1]
    H = down_w.shape[-1]
    _pmax = 128
    _q_width = 4
    n_H512 = H // _pmax // _q_width
    result = np.zeros((T, H), dtype=np.float32)

    # Select weight unpack function based on dtype
    is_float4 = 'float4' in str(gate_up_w.dtype)
    w_unpack = unpack_float4_x4 if is_float4 else unpack_float8_e4m3fn_x4

    def _compute_one_expert(active_in, active_scale, expert_idx):
        # When input is not already quantized, quantize hidden to mxfp8 (float8_e4m3fn_x4)
        BxS_local = active_in.shape[0] if active_scale is None else active_in.shape[-1]
        if active_scale is None:
            h = active_in.reshape(BxS_local, _q_width, n_H512, _pmax).transpose(3, 2, 0, 1).reshape(_pmax, -1)
            hidden_mx, hidden_scale = quantize_to_mx(h, nl.float8_e4m3fn_x4)
            # Reshape to [_pmax, n_H512, BxS] matching gate_up_proj_mx_torch_ref input
            hidden_mx = hidden_mx.reshape(_pmax, n_H512, BxS_local)
            hidden_scale_t = torch.from_numpy(hidden_scale.reshape(_pmax // 8, n_H512, BxS_local))
        else:
            # Prequantized:
            hidden_mx = active_in
            hidden_scale_t = active_scale

        # Gate projection
        gw = gate_up_w[expert_idx][:, 0, :, :]
        gs_t = gate_up_w_scale[expert_idx][:, 0, :, :]
        gb_t = gate_up_b[expert_idx][:, 0, :, :].float() if gate_up_b is not None else None
        gate_out = gate_up_proj_mx_torch_ref(
            hidden_mx,
            hidden_scale_t,
            gw,
            gs_t,
            gb_t,
            H,
            I,
            BxS_local,
            hidden_unpack_fn=unpack_float8_e4m3fn_x4,
            weight_unpack_fn=w_unpack,
        )["out"].numpy()

        # Up projection
        uw = gate_up_w[expert_idx][:, 1, :, :]
        us_t = gate_up_w_scale[expert_idx][:, 1, :, :]
        ub_t = gate_up_b[expert_idx][:, 1, :, :].float() if gate_up_b is not None else None
        up_out = gate_up_proj_mx_torch_ref(
            hidden_mx,
            hidden_scale_t,
            uw,
            us_t,
            ub_t,
            H,
            I,
            BxS_local,
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

        # Clamp to prevent float32 overflow in activation * up_out multiply
        gate_out = np.clip(gate_out, -_F32_SAFE, _F32_SAFE)
        up_out = np.clip(up_out, -_F32_SAFE, _F32_SAFE)

        # Activation + multiply
        mult = torch.from_numpy(act_fn_func(gate_out) * up_out)

        # Down projection
        dw = down_w[expert_idx]
        ds_t = down_w_scale[expert_idx]
        db_t = down_b[expert_idx].float() if down_b is not None else None
        return down_proj_mx_torch_ref(mult, dw, ds_t, db_t, H, I, BxS_local, weight_unpack_fn=w_unpack)["out"].numpy()

    if is_all_expert:
        for e in range(E):
            # Batch all T tokens in a single call (BxS=T) instead of looping token-by-token
            expert_out = _compute_one_expert(inp, inp_scale, e).reshape(T, H)
            if scale_mode == 1:
                expert_out = expert_out * affinities[:, e : e + 1]
            result += expert_out
    else:
        T_idx, K = exp_idx.shape
        for t in range(T_idx):
            for k in range(K):
                e = int(exp_idx[t, k])
                if inp_scale is not None:
                    token_in = inp[:, :, t : t + 1]
                    token_scale = inp_scale[:, :, t : t + 1]
                else:
                    token_in = inp[t : t + 1]
                    token_scale = None
                token_out = _compute_one_expert(token_in, token_scale, e).reshape(H)
                if scale_mode == 1:
                    token_out = token_out * affinities[t, e]
                result[t] += token_out

    # When using a2av, token indices are appended to the end of the output, bitcast to output dtype
    output = result.astype(dtype)
    if all_to_all_v_strategy == MoEAllToAllVStrategy.PACK_OUTPUT_ROWS:
        # Unpermute: routed tokens (token_idx != 0) first, then zeros
        token_idx_int32 = token_indices.numpy().view(np.int32).flatten()
        routed_mask = token_idx_int32 != 0
        count = int(routed_mask.sum())
        PACK_OUTPUT_ROWS = np.zeros_like(output)
        PACK_OUTPUT_ROWS[:count] = output[routed_mask]
        # Trailing columns: token indices in unpermuted order
        unpermuted_token_indices = np.zeros((T, token_indices.shape[1]), dtype=token_indices.numpy().dtype)
        unpermuted_token_indices[:count] = token_indices.numpy()[routed_mask]
        output = np.concatenate([PACK_OUTPUT_ROWS, unpermuted_token_indices.view(dtype)], axis=1)
    elif all_to_all_v_strategy != MoEAllToAllVStrategy.DISABLED:
        output = np.concatenate([output, token_indices.numpy().view(dtype)], axis=1)

    return {"out": output}


def _moe_tkg_static_mx_ref(
    hidden_input,
    expert_gate_up_weights,
    expert_down_weights,
    expert_affinities,
    expert_index,
    is_all_expert,
    expert_gate_up_weights_scale,
    expert_down_weights_scale,
    expert_gate_up_input_scale,
    expert_down_input_scale,
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
    """STATIC_MX golden: FP8 round-trip on input + MX matmul + post-matmul dequant.

    In STATIC_MX, expert_gate_up_weights_scale and expert_down_weights_scale are float32
    per-expert weight dequant scales (not uint8 MX block scales). The golden generates
    dummy-127 block scales internally for the MX matmul.
    """
    import nki.language as nl

    # Activation dequant scales (float32 [E_L, 1])
    gate_up_in_dequant_np = _to_numpy(expert_gate_up_input_scale)
    down_in_dequant_np = _to_numpy(expert_down_input_scale)
    # Weight dequant scales (float32 [E_L, 2, 1] for gate/up, [E_L, 1] for down)
    gate_up_w_dequant_np = _to_numpy(expert_gate_up_weights_scale)
    down_w_dequant_np = _to_numpy(expert_down_weights_scale)

    inp = _to_numpy(hidden_input)
    affinities = _to_numpy(expert_affinities)
    exp_idx = _to_numpy(expert_index)
    gate_up_w = expert_gate_up_weights
    down_w = expert_down_weights
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
        ActFnType.SiLU: lambda x: x * (1 / (1 + np.exp(-np.clip(x, -_EXP_CLAMP, _EXP_CLAMP)))),
        ActFnType.Swish: lambda x: x * (1 / (1 + np.exp(np.clip(-1.702 * x, -_EXP_CLAMP, _EXP_CLAMP)))),
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

    is_float4 = 'float4' in str(gate_up_w.dtype)
    w_unpack = unpack_float4_x4 if is_float4 else unpack_float8_e4m3fn_x4

    def _compute_one_expert(active_in, expert_idx):
        BxS = active_in.shape[0]
        # Step 1: FP8 quantize input (matching kernel's static_quantization → fp8 cast)
        # Kernel: quantized = FP8(input / in_scale), then feeds quantized into nc_matmul_mx
        # Post-matmul: result *= in_scale * w_dequant to recover original scale
        in_scale = float(gate_up_in_dequant_np[expert_idx, 0])
        inp_f32 = active_in.astype(np.float32)
        quantized = np.clip(inp_f32 / in_scale, -448.0, 448.0)

        # Step 2: Feed QUANTIZED values (not dequantized!) into MX matmul
        # This mirrors the kernel which feeds raw FP8 values (input/in_scale) into nc_matmul_mx
        h = quantized.reshape(BxS, _q_width, n_H512, _pmax).transpose(3, 2, 0, 1).reshape(_pmax, -1)
        # Round to FP8 precision first (matching kernel's static_quantization → fp8 cast),
        # then quantize to MX. Without this, quantize_to_mx normalizes float32 values by
        # block scale before FP8 rounding, producing different FP8 bit patterns than the
        # kernel's direct float32→FP8 cast. The MX roundtrip is lossless on FP8-precision
        # inputs, so the real block scales correctly reconstruct the kernel's FP8 values.
        h_fp8 = unpack_float8_e4m3fn_x4(dt.static_cast(np.ascontiguousarray(h.astype(np.float32)), nl.float8_e4m3fn_x4))
        hidden_mx, hidden_scale = quantize_to_mx(h_fp8.numpy(), nl.float8_e4m3fn_x4)
        hidden_mx = hidden_mx.reshape(_pmax, n_H512, BxS)
        hidden_scale_t = torch.from_numpy(hidden_scale.reshape(_pmax // 8, n_H512, BxS))

        # Gate projection (MX matmul with dummy-127 weight block scales, no bias)
        # Dummy-127 block scales (= scale factor 1.0) match the kernel's memset(127)
        gw = gate_up_w[expert_idx][:, 0, :, :]
        dummy_w_scale = torch.full((gw.shape[0] // 8,) + gw.shape[1:], 127, dtype=torch.uint8)
        gate_out = gate_up_proj_mx_torch_ref(
            hidden_mx,
            hidden_scale_t,
            gw,
            dummy_w_scale,
            None,
            H,
            I,
            BxS,
            hidden_unpack_fn=unpack_float8_e4m3fn_x4,
            weight_unpack_fn=w_unpack,
        )["out"].numpy()

        # Up projection
        uw = gate_up_w[expert_idx][:, 1, :, :]
        up_out = gate_up_proj_mx_torch_ref(
            hidden_mx,
            hidden_scale_t,
            uw,
            dummy_w_scale,
            None,
            H,
            I,
            BxS,
            hidden_unpack_fn=unpack_float8_e4m3fn_x4,
            weight_unpack_fn=w_unpack,
        )["out"].numpy()

        # Post-matmul dequant: result *= in_scale * w_dequant_scale
        gate_w_dequant = float(gate_up_w_dequant_np[expert_idx, 0, 0])
        up_w_dequant = float(gate_up_w_dequant_np[expert_idx, 1, 0])
        gate_out = gate_out * in_scale * gate_w_dequant
        up_out = up_out * in_scale * up_w_dequant

        # Bias after dequant — extract raw bias contribution via double-matmul trick:
        # bias_contribution = mx_matmul(with_bias) - mx_matmul(without_bias)
        # Then add to already-dequanted result: final = dequanted_matmul + bias
        if gate_up_b != None:
            gb_t = torch.from_numpy(np.array(gate_up_b[expert_idx][:, 0, :, :])).float()
            ub_t = torch.from_numpy(np.array(gate_up_b[expert_idx][:, 1, :, :])).float()
            gate_with_bias = gate_up_proj_mx_torch_ref(
                hidden_mx,
                hidden_scale_t,
                gw,
                dummy_w_scale,
                gb_t,
                H,
                I,
                BxS,
                hidden_unpack_fn=unpack_float8_e4m3fn_x4,
                weight_unpack_fn=w_unpack,
            )["out"].numpy()
            up_with_bias = gate_up_proj_mx_torch_ref(
                hidden_mx,
                hidden_scale_t,
                uw,
                dummy_w_scale,
                ub_t,
                H,
                I,
                BxS,
                hidden_unpack_fn=unpack_float8_e4m3fn_x4,
                weight_unpack_fn=w_unpack,
            )["out"].numpy()
            # Extract raw bias (matmul result without dequant)
            gate_raw_matmul = gate_out / (in_scale * gate_w_dequant)
            up_raw_matmul = up_out / (in_scale * up_w_dequant)
            gate_bias = gate_with_bias - gate_raw_matmul
            up_bias = up_with_bias - up_raw_matmul
            gate_out = gate_out + gate_bias
            up_out = up_out + up_bias

        # Clamp
        if gate_clamp_upper_limit != None:
            gate_out = np.minimum(gate_out, gate_clamp_upper_limit)
        if gate_clamp_lower_limit != None:
            gate_out = np.maximum(gate_out, gate_clamp_lower_limit)
        if up_clamp_upper_limit != None:
            up_out = np.minimum(up_out, up_clamp_upper_limit)
        if up_clamp_lower_limit != None:
            up_out = np.maximum(up_out, up_clamp_lower_limit)

        # Activation + multiply
        intermediate = act_fn_func(gate_out) * up_out

        # Step 3: Down projection with MX quantization.
        down_in_scale = float(down_in_dequant_np[expert_idx, 0])
        # Pre-quantize intermediate by dividing by down_in_scale
        intermediate_for_down = np.clip(intermediate / down_in_scale, -448.0, 448.0).astype(np.float32)
        mult_t = torch.from_numpy(intermediate_for_down)
        dw = down_w[expert_idx]
        dummy_down_w_scale = torch.full((dw.shape[0] // 8,) + dw.shape[1:], 127, dtype=torch.uint8)
        down_out = down_proj_mx_torch_ref(mult_t, dw, dummy_down_w_scale, None, H, I, BxS, weight_unpack_fn=w_unpack)[
            "out"
        ].numpy()

        # Post-down dequant: result *= down_in_scale * down_w_dequant
        down_out = down_out * down_in_scale * float(down_w_dequant_np[expert_idx, 0])

        # Down bias after dequant — down_b is [E, H] (already in standard layout, no MX packing)
        if down_b != None:
            db_np = np.array(down_b[expert_idx]).flatten()[:H].astype(np.float32)
            down_out = down_out + db_np.reshape(1, H)

        return down_out

    if is_all_expert:
        for e in range(E):
            expert_out = _compute_one_expert(inp, e).reshape(T, H)
            if scale_mode == 1:
                expert_out = expert_out * affinities[:, e : e + 1]
            result += expert_out
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


def _moe_tkg_row_mx_ref(
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
    """ROW_MX golden: per-token dynamic FP8 input quantization + MX matmul + per-row weight dequant.

    In ROW_MX, expert_gate_up_weights_scale is [E_L, 2, n_I512*4] (per-row weight dequant)
    and expert_down_weights_scale is [E_L, H//128] (per-row weight dequant).
    Input quantization is per-token dynamic FP8 (not per-tensor static).
    """
    import nki.language as nl

    gate_up_w_dequant_np = _to_numpy(expert_gate_up_weights_scale)
    down_w_dequant_np = _to_numpy(expert_down_weights_scale)

    inp = _to_numpy(hidden_input)
    affinities = _to_numpy(expert_affinities)
    exp_idx = _to_numpy(expert_index)
    gate_up_w = expert_gate_up_weights
    down_w = expert_down_weights
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
        ActFnType.SiLU: lambda x: x * (1 / (1 + np.exp(-np.clip(x, -_EXP_CLAMP, _EXP_CLAMP)))),
        ActFnType.Swish: lambda x: x * (1 / (1 + np.exp(np.clip(-1.702 * x, -_EXP_CLAMP, _EXP_CLAMP)))),
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
    n_I512 = math.ceil(I / (_pmax * _q_width))
    FP8_MAX = 448.0
    result = np.zeros((T, H), dtype=np.float32)

    is_float4 = 'float4' in str(gate_up_w.dtype)
    w_unpack = unpack_float4_x4 if is_float4 else unpack_float8_e4m3fn_x4

    def _per_token_fp8_quantize(data):
        """Per-token dynamic FP8 quantization with bf16 round-trip."""
        BxS = data.shape[0]
        dequant_scales = np.zeros(BxS, dtype=np.float32)
        quantized = np.zeros_like(data, dtype=np.float32)
        for t in range(BxS):
            absmax = np.max(np.abs(data[t].astype(np.float32)))
            dequant = max(absmax / FP8_MAX, 1e-12)
            dequant_scales[t] = np.float32(dequant)
            scaled = np.clip(data[t].astype(np.float32) / dequant, -FP8_MAX, FP8_MAX)
            # Round to bf16 precision (matching kernel's bf16→fp8 cast)
            scaled_bf16 = dt.static_cast(scaled, nl.bfloat16).astype(np.float32)
            quantized[t] = scaled_bf16
        return quantized, dequant_scales

    def _compute_one_expert(active_in, expert_idx):
        BxS = active_in.shape[0]

        # Step 1: Per-token FP8 quantize input
        quantized, in_dequant_scales = _per_token_fp8_quantize(active_in.astype(np.float32))

        # Step 2: MX matmul with dummy-127 block scales
        h = quantized.reshape(BxS, _q_width, n_H512, _pmax).transpose(3, 2, 0, 1).reshape(_pmax, -1)
        h_fp8 = unpack_float8_e4m3fn_x4(dt.static_cast(np.ascontiguousarray(h.astype(np.float32)), nl.float8_e4m3fn_x4))
        hidden_mx, hidden_scale = quantize_to_mx(h_fp8.numpy(), nl.float8_e4m3fn_x4)
        hidden_mx = hidden_mx.reshape(_pmax, n_H512, BxS)
        hidden_scale_t = torch.from_numpy(hidden_scale.reshape(_pmax // 8, n_H512, BxS))

        gw = gate_up_w[expert_idx][:, 0, :, :]
        dummy_w_scale = torch.full((gw.shape[0] // 8,) + gw.shape[1:], 127, dtype=torch.uint8)
        gate_out = gate_up_proj_mx_torch_ref(
            hidden_mx,
            hidden_scale_t,
            gw,
            dummy_w_scale,
            None,
            H,
            I,
            BxS,
            hidden_unpack_fn=unpack_float8_e4m3fn_x4,
            weight_unpack_fn=w_unpack,
        )["out"].numpy()

        uw = gate_up_w[expert_idx][:, 1, :, :]
        up_out = gate_up_proj_mx_torch_ref(
            hidden_mx,
            hidden_scale_t,
            uw,
            dummy_w_scale,
            None,
            H,
            I,
            BxS,
            hidden_unpack_fn=unpack_float8_e4m3fn_x4,
            weight_unpack_fn=w_unpack,
        )["out"].numpy()

        # Post-matmul dequant: per-row weight scale * per-token input scale
        # gate_out shape is tiled: (_pmax, n_I512, BxS, 4) from gate_up_proj_mx_torch_ref
        # The pre-shuffled weight scale [n_I512*4] maps directly to the tiled layout:
        # scale[i_tile * 4 + i_q] applies to gate_out[:cur_I_pdim, i_tile, :, i_q]
        gate_w_scale_shuffled = gate_up_w_dequant_np[expert_idx, 0, :]  # [n_I512*4]
        up_w_scale_shuffled = gate_up_w_dequant_np[expert_idx, 1, :]

        for i_tile in range(n_I512):
            for i_q in range(_q_width):
                i_col = i_tile * _q_width + i_q
                if i_col < len(gate_w_scale_shuffled):
                    gate_out[:, i_tile, :, i_q] *= gate_w_scale_shuffled[i_col]
                    up_out[:, i_tile, :, i_q] *= up_w_scale_shuffled[i_col]
        # Per-token input dequant: broadcasts over P and I dimensions
        for t in range(BxS):
            gate_out[:, :, t, :] *= in_dequant_scales[t]
            up_out[:, :, t, :] *= in_dequant_scales[t]

        # Bias after dequant — extract raw bias contribution via double-matmul trick:
        # bias_contribution = mx_matmul(with_bias) - mx_matmul(without_bias)
        # Then add to already-dequanted result: final = dequanted_matmul + bias
        if gate_up_b is not None:
            gb_t = torch.from_numpy(np.array(gate_up_b[expert_idx][:, 0, :, :])).float()
            ub_t = torch.from_numpy(np.array(gate_up_b[expert_idx][:, 1, :, :])).float()
            gate_with_bias = gate_up_proj_mx_torch_ref(
                hidden_mx,
                hidden_scale_t,
                gw,
                dummy_w_scale,
                gb_t,
                H,
                I,
                BxS,
                hidden_unpack_fn=unpack_float8_e4m3fn_x4,
                weight_unpack_fn=w_unpack,
            )["out"].numpy()
            up_with_bias = gate_up_proj_mx_torch_ref(
                hidden_mx,
                hidden_scale_t,
                uw,
                dummy_w_scale,
                ub_t,
                H,
                I,
                BxS,
                hidden_unpack_fn=unpack_float8_e4m3fn_x4,
                weight_unpack_fn=w_unpack,
            )["out"].numpy()
            # Extract raw bias: (with_bias - without_bias) in the raw MX matmul output
            # gate_out already has dequant applied, so divide it back to get raw matmul
            gate_raw = gate_out.copy()
            up_raw = up_out.copy()
            for i_tile in range(n_I512):
                for i_q in range(_q_width):
                    i_col = i_tile * _q_width + i_q
                    if i_col < len(gate_w_scale_shuffled):
                        gate_raw[:, i_tile, :, i_q] /= gate_w_scale_shuffled[i_col]
                        up_raw[:, i_tile, :, i_q] /= up_w_scale_shuffled[i_col]
            for t in range(BxS):
                gate_raw[:, :, t, :] /= in_dequant_scales[t]
                up_raw[:, :, t, :] /= in_dequant_scales[t]
            gate_bias = gate_with_bias - gate_raw
            up_bias = up_with_bias - up_raw
            gate_out = gate_out + gate_bias
            up_out = up_out + up_bias

        # Clamp
        if gate_clamp_upper_limit is not None:
            gate_out = np.minimum(gate_out, gate_clamp_upper_limit)
        if gate_clamp_lower_limit is not None:
            gate_out = np.maximum(gate_out, gate_clamp_lower_limit)
        if up_clamp_upper_limit is not None:
            up_out = np.minimum(up_out, up_clamp_upper_limit)
        if up_clamp_lower_limit is not None:
            up_out = np.maximum(up_out, up_clamp_lower_limit)

        # Activation + multiply (tiled layout [_pmax, n_I512, BxS, 4])
        intermediate = act_fn_func(gate_out) * up_out

        # Step 3: Per-token FP8 quantize intermediate for down projection.
        # intermediate is tiled [_pmax, n_I512, BxS, 4]. Flatten per-token for quantization,
        # then reshape back to tiled for down_proj_mx_torch_ref which expects [_pmax, n_I512, BxS, 4].
        inter_for_quant = intermediate.transpose(2, 0, 1, 3).reshape(BxS, -1)  # [BxS, _pmax*n_I512*4]
        inter_quantized_flat, inter_dequant_scales = _per_token_fp8_quantize(inter_for_quant)
        inter_tiled = inter_quantized_flat.reshape(BxS, _pmax, n_I512, _q_width).transpose(1, 2, 0, 3)
        inter_t = torch.from_numpy(inter_tiled)

        dw = down_w[expert_idx]
        dummy_down_w_scale = torch.full((dw.shape[0] // 8,) + dw.shape[1:], 127, dtype=torch.uint8)
        down_out = down_proj_mx_torch_ref(inter_t, dw, dummy_down_w_scale, None, H, I, BxS, weight_unpack_fn=w_unpack)[
            "out"
        ].numpy()

        # Post-down dequant: per-row weight scale * per-token intermediate scale
        # down_out shape: [BxS, H] from down_proj_mx_torch_ref
        down_w_scale = down_w_dequant_np[expert_idx, :]  # [H//128]
        down_w_scale_full = np.repeat(down_w_scale, _pmax)[:H]
        for t in range(BxS):
            down_out[t] = down_out[t] * down_w_scale_full * inter_dequant_scales[t]

        if down_b is not None:
            db_np = np.array(down_b[expert_idx]).flatten()[:H].astype(np.float32)
            down_out = down_out + db_np.reshape(1, H)

        return down_out

    if is_all_expert:
        for e in range(E):
            expert_out = _compute_one_expert(inp, e).reshape(T, H)
            if scale_mode == 1:
                expert_out = expert_out * affinities[:, e : e + 1]
            result += expert_out
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
