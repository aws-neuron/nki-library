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

import math
from test.integration.nkilib.core.mlp.test_mlp_common import (
    gen_moe_mx_weights,
    norm_mlp_ref,
    norm_mlp_ref_mx,
    rmsnorm_quant_mlp_ref,
)
from test.integration.nkilib.utils.dtype_helper import dt
from test.integration.nkilib.utils.tensor_generators import generate_stabilized_mx_data
from typing import Callable

import nki.language as nl
import numpy as np
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    GateUpDim,
    NormType,
    QuantizationType,
)

# Constants
_pmax = 128  # sbuf max partition dim
_q_width = 4  # quantization width
_q_height = 8  # quantization height


def is_dtype_mx(dtype):
    return dtype in (nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4, nl.float8_e5m2_x4)


def is_dtype_fp8(dtype):
    return dtype in (nl.float8_e4m3, nl.float8_e5m2)


def is_dtype_low_precision(dtype):
    return is_dtype_mx(dtype) or is_dtype_fp8(dtype)


def build_moe_tkg(
    tokens,
    hidden,
    intermediate,
    expert,
    top_k,
    act_fn,
    expert_affinities_scaling_mode,
    is_all_expert,
    quant_dtype=None,
    quant_type=QuantizationType.NONE,
    in_dtype=nl.bfloat16,
    out_dtype=None,
    bias=False,
    clamp=False,
    tensor_generator: Callable = None,
    rank_id: int = 0,
    mask_unselected_experts: bool = False,
):
    """Build input tensors for MoE TKG kernel testing.

    Args:
        rank_id: Rank ID for all-expert mode (default 0, meaning all experts are local)
        mask_unselected_experts: Whether to apply affinity masking based on expert_index in all-expert mode
    """
    # Default out_dtype to in_dtype if not provided
    if out_dtype is None:
        out_dtype = in_dtype

    is_mx_quant = is_dtype_mx(quant_dtype)

    # For all-expert mode, top_k defaults to expert count if not specified
    effective_top_k = top_k if top_k is not None else expert

    # Pre-generate expert indices (used for both MX and non-MX paths)
    np.random.seed(0)
    if is_all_expert:
        exp_idx = np.zeros((tokens, effective_top_k), dtype=np.int32)
        for t in range(tokens):
            exp_idx[t] = np.random.choice(expert, size=effective_top_k, replace=False)
        exp_idx = dt.static_cast(exp_idx, np.int32)
    else:
        exp_idx = dt.static_cast(np.random.randint(0, expert, size=(tokens, effective_top_k)), np.int32)

    # Pre-generate weights if using MX quantization
    if is_mx_quant:
        mx_weights = gen_moe_mx_weights(hidden, intermediate, expert, quant_dtype)

    # Use custom tensor generator if not provided
    if tensor_generator is None:

        def default_tensor_generator(inp):
            rng = np.random.default_rng(0)
            if inp.name == "expert_index":
                return dt.static_cast(exp_idx, inp.dtype)
            elif inp.name == "rank_id":
                return np.array([[rank_id]], dtype=inp.dtype)
            elif inp.name == "expert_affinities":
                # Create [T, K] affinities and place them at expert_index positions
                random_matrix = np.random.rand(tokens, effective_top_k)
                normalized_matrix = random_matrix / random_matrix.sum(axis=1, keepdims=True)

                # Create [T, E] with values at exp_idx positions
                res = np.zeros((tokens, expert), dtype=inp.dtype)
                for i_t in range(tokens):
                    for i_k in range(effective_top_k):
                        i_e = exp_idx[i_t, i_k]
                        res[i_t, i_e] = normalized_matrix[i_t, i_k]
                return dt.static_cast(res, inp.dtype)

            elif inp.name in ("gate_up_b", "down_b"):
                return rng.normal(size=inp.shape).astype(inp.dtype)
            elif inp.name == 'gate_up_w' and is_mx_quant:
                return mx_weights.gate_up_w_qtz
            elif inp.name == 'down_w' and is_mx_quant:
                return mx_weights.down_w_qtz
            elif inp.name == 'gate_up_w_scale' and is_mx_quant:
                return mx_weights.gate_up_w_scale
            elif inp.name == 'down_w_scale' and is_mx_quant:
                return mx_weights.down_w_scale
            elif inp.name == 'hidden_input' and is_mx_quant:
                n_H512_tile = hidden // 512
                hidden_states, _, _ = generate_stabilized_mx_data(
                    mx_dtype=nl.float8_e4m3fn_x4, shape=(tokens * n_H512_tile * _pmax, _q_width), val_range=5
                )
                hidden_states = (
                    hidden_states.reshape(tokens, n_H512_tile, _pmax, _q_width)
                    .transpose(0, 3, 1, 2)
                    .reshape(tokens, hidden)
                )
                return dt.static_cast(hidden_states, inp.dtype)
            else:
                mean = 0.0
                std = 1.0
                return (rng.normal(size=inp.shape) * std + mean).astype(inp.dtype)

        tensor_generator = default_tensor_generator

    # Create a simple object to hold shape, dtype, and name for the generator
    class TensorTemplate:
        def __init__(self, shape, dtype, name):
            self.shape = shape
            self.dtype = dtype
            self.name = name

    # Init helper dims
    intermediate_p = (intermediate // 4) if intermediate < 512 else _pmax  # for mxfp4
    n_H512_tile = hidden // (_pmax * _q_width)
    n_I512_tile = math.ceil(intermediate / (_pmax * _q_width))

    # Init quant dtypes - use in_dtype for input tensor
    hidden_dtype = in_dtype
    weight_dtype = quant_dtype if quant_dtype is not None else in_dtype
    scale_dtype = np.uint8 if is_mx_quant else np.float32

    # Init weight and scale shapes
    gate_up_w_shape = (
        (expert, _pmax, 2, n_H512_tile, intermediate) if is_mx_quant else (expert, hidden, 2, intermediate)
    )
    down_w_shape = (expert, intermediate_p, n_I512_tile, hidden) if is_mx_quant else (expert, intermediate, hidden)

    gate_up_w_scale_shape = None
    down_w_scale_shape = None
    if is_mx_quant:
        gate_up_w_scale_shape = (expert, _pmax // _q_height, 2, n_H512_tile, intermediate)
        down_w_scale_shape = (expert, intermediate_p // _q_height, n_I512_tile, hidden)
    elif quant_type == QuantizationType.ROW:
        gate_up_w_scale_shape = (expert, 2, intermediate)
        down_w_scale_shape = (expert, hidden)
    elif quant_type == QuantizationType.STATIC:
        gate_up_w_scale_shape = (expert, 2, 1)
        down_w_scale_shape = (expert, 1)

    # Init bias shape
    gate_up_bias_shape = (
        (expert, intermediate_p, 2, n_I512_tile, _q_width) if is_mx_quant else (expert, 2, intermediate)
    )
    down_bias_shape = (expert, hidden)

    # Create input tensors
    hidden_input = tensor_generator(TensorTemplate((tokens, hidden), hidden_dtype, "hidden_input"))
    expert_affinities = tensor_generator(TensorTemplate((tokens, expert), nl.float32, "expert_affinities"))
    # expert_index is always required now
    expert_index = tensor_generator(TensorTemplate((tokens, effective_top_k), nl.int32, "expert_index"))
    # rank_id for all-expert mode with affinity scaling
    rank_id_tensor = tensor_generator(TensorTemplate((1, 1), nl.uint32, "rank_id"))
    gate_up_w = tensor_generator(TensorTemplate(gate_up_w_shape, weight_dtype, "gate_up_w"))
    down_w = tensor_generator(TensorTemplate(down_w_shape, weight_dtype, "down_w"))

    gate_up_b = tensor_generator(TensorTemplate(gate_up_bias_shape, hidden_dtype, "gate_up_b")) if bias else None
    down_b = tensor_generator(TensorTemplate(down_bias_shape, hidden_dtype, "down_b")) if bias else None

    gate_up_w_scale = None
    down_w_scale = None
    if gate_up_w_scale_shape is not None:
        gate_up_w_scale = tensor_generator(TensorTemplate(gate_up_w_scale_shape, scale_dtype, "gate_up_w_scale"))
    if down_w_scale_shape is not None:
        down_w_scale = tensor_generator(TensorTemplate(down_w_scale_shape, scale_dtype, "down_w_scale"))

    gate_up_in_scale = None
    down_in_scale = None
    if quant_type == QuantizationType.STATIC:
        gate_up_in_scale = tensor_generator(TensorTemplate((expert, 1), dtype=np.float32, name="gate_up_input_scale"))
        down_in_scale = tensor_generator(TensorTemplate((expert, 1), dtype=np.float32, name="down_input_scale"))

    kernel_input = {
        "hidden_input": hidden_input,
        "expert_gate_up_weights": gate_up_w,
        "expert_down_weights": down_w,
        "expert_affinities": expert_affinities,
        "expert_index": expert_index,
        "is_all_expert": is_all_expert,
        "rank_id": rank_id_tensor
        if is_all_expert and expert_affinities_scaling_mode != ExpertAffinityScaleMode.NO_SCALE
        else None,
        "mask_unselected_experts": mask_unselected_experts,
        "expert_gate_up_bias": gate_up_b,
        "expert_down_bias": down_b,
        "expert_gate_up_weights_scale": gate_up_w_scale,
        "expert_down_weights_scale": down_w_scale,
        "gate_up_input_scale": gate_up_in_scale,
        "down_input_scale": down_in_scale,
        "expert_affinities_eager": None,
        "expert_affinities_scaling_mode": expert_affinities_scaling_mode,
        "activation_fn": act_fn,
        "output_dtype": out_dtype,
        "gate_clamp_upper_limit": float(7.0) if clamp else None,
        "gate_clamp_lower_limit": None,
        "up_clamp_upper_limit": float(8.0) if clamp else None,
        "up_clamp_lower_limit": float(-6.0) if clamp else None,
    }
    return kernel_input


def _get_clamp_limits(clamp):
    """Get clamp limits for gate and up projections."""
    return (
        float(7.0) if clamp else None,  # gate_clamp_upper_limit
        None,  # gate_clamp_lower_limit
        float(8.0) if clamp else None,  # up_clamp_upper_limit
        float(-6.0) if clamp else None,  # up_clamp_lower_limit
    )


def _compute_expert_mlp(
    active_in,
    gate_w_select,
    up_w_select,
    down_w_select,
    gate_w_scale,
    up_w_scale,
    down_w_scale,
    gate_up_in_scale,
    down_in_scale,
    dtype,
    quant_dtype,
    quant_type,
    act_fn_type,
    gate_b,
    up_b_select,
    down_b,
    clamp_limits,
):
    """Compute MLP output for a single expert."""
    gate_clamp_upper_limit, gate_clamp_lower_limit, up_clamp_upper_limit, up_clamp_lower_limit = clamp_limits

    if quant_type != QuantizationType.NONE:
        # reshape the tensor so that they can reuse the same rmsnorm_quant_mlp_ref golden function

        # expand to [H, 1, T] with adaptive shape of [H, B, S]
        active_in_3d = active_in.reshape(active_in.shape[0], 1, -1)

        gate_w_scale = np.tile(gate_w_scale, (128, 1))
        up_w_scale = np.tile(up_w_scale, (128, 1))
        down_w_scale = np.tile(down_w_scale, (128, 1))

        if quant_type == QuantizationType.STATIC:
            gate_up_in_scale = np.tile(gate_up_in_scale, (128, 1))
            down_in_scale = np.tile(down_in_scale, (128, 1))

        mlp_out_3d, _ = rmsnorm_quant_mlp_ref(
            [active_in_3d],
            None,  # gamma
            gate=gate_w_select,
            up=up_w_select,
            down=down_w_select,
            quantization_type=quant_type,
            gate_w_scale=gate_w_scale,
            up_w_scale=up_w_scale,
            down_w_scale=down_w_scale,
            gate_in_scale=gate_up_in_scale,
            up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            dtype=dtype,
            quant_dtype=quant_dtype,
            rmsnorm=None,
            store_add=False,
            skip_gate=False,
            act_fn_type=act_fn_type,
            gate_b=gate_b,
            up_b=up_b_select,
            down_b=down_b,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            quantize_activation=True,
        )

        mlp_out = mlp_out_3d.squeeze()

    else:
        mlp_out, _ = norm_mlp_ref(
            [active_in],
            None,  # gamma
            gate_w_select,
            up_w_select,
            down_w_select,
            dtype,
            norm_type=NormType.NO_NORM,
            store_add=False,
            skip_gate=False,
            act_fn_type=act_fn_type,
            gate_b=gate_b,
            up_b=up_b_select,
            down_b=down_b,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
        )

    return mlp_out


def _compute_expert_mlp_mx(
    active_in,
    gate_w,
    gate_scale,
    up_w,
    up_scale,
    down_w_select,
    down_scale,
    dtype,
    act_fn_type,
    gate_b,
    up_b,
    down_b,
    clamp_limits,
):
    """Compute MLP output for a single expert."""
    gate_clamp_upper_limit, gate_clamp_lower_limit, up_clamp_upper_limit, up_clamp_lower_limit = clamp_limits

    mlp_out, _ = norm_mlp_ref_mx(
        [active_in],
        None,
        gate_w,
        gate_scale,
        up_w,
        up_scale,
        down_w_select,
        down_scale,
        dtype,
        act_fn_type=act_fn_type,
        gate_b=gate_b,
        up_b=up_b,
        down_b=down_b,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
    )
    return mlp_out


def _apply_pre_scale(active_in, affinity, expert_affinities_scaling_mode):
    """Apply pre-scaling to input if needed."""
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.PRE_SCALE:
        return active_in * affinity
    return active_in


def _apply_post_scale(mlp_out, affinity, expert_affinities_scaling_mode):
    """Apply post-scaling to output if needed."""
    if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
        return mlp_out * affinity
    return mlp_out


def _extract_tensors(inp_np, bias=False):
    """Extract common tensors from input dictionary."""
    inp = inp_np["hidden_input"]
    gate_up_w = inp_np["expert_gate_up_weights"]
    down_w = inp_np["expert_down_weights"]
    gate_up_b = inp_np["expert_gate_up_bias"] if bias else None
    down_b = inp_np["expert_down_bias"] if bias else None
    expert_affinities = inp_np["expert_affinities"]
    return inp, gate_up_w, down_w, gate_up_b, down_b, expert_affinities


def _extract_quant_tensors(inp_np, quant_type):
    """Extract scale tensors from input dictionary."""
    gate_w_scale = None
    up_w_scale = None
    down_w_scale = None
    gate_up_in_scale = None
    down_in_scale = None
    if quant_type == QuantizationType.NONE:
        return gate_w_scale, up_w_scale, down_w_scale, gate_up_in_scale, down_in_scale

    gate_w_scale = inp_np["expert_gate_up_weights_scale"]
    up_w_scale = inp_np["expert_gate_up_weights_scale"]
    down_w_scale = inp_np["expert_down_weights_scale"]

    if quant_type == QuantizationType.STATIC:
        gate_up_in_scale = inp_np["gate_up_input_scale"]
        down_in_scale = inp_np["down_input_scale"]

    return gate_w_scale, up_w_scale, down_w_scale, gate_up_in_scale, down_in_scale


def _select_tensors(gate_up_w, down_w, expert_idx, gate_up_b=None, down_b=None, bias=False):
    """Extract expert weights and biases, return processed weights."""
    gate_w_select = gate_up_w[expert_idx][:, GateUpDim.GATE.value, :]
    up_w_select = gate_up_w[expert_idx][:, GateUpDim.UP.value, :]
    down_w_select = down_w[expert_idx]
    gate_b_select = gate_up_b[expert_idx][GateUpDim.GATE.value, :] if bias else None
    up_b_select = gate_up_b[expert_idx][GateUpDim.UP.value, :] if bias else None
    down_b_select = down_b[expert_idx] if bias else None

    return gate_w_select, up_w_select, down_w_select, gate_b_select, up_b_select, down_b_select


def _select_quant_tensors(gate_w_scale, up_w_scale, down_w_scale, gate_up_in_scale, down_in_scale, expert_idx):
    """Select scale tensors for a given expert."""
    if gate_w_scale is not None:
        gate_w_scale = gate_w_scale[expert_idx][GateUpDim.GATE.value, :]

    if up_w_scale is not None:
        up_w_scale = up_w_scale[expert_idx][GateUpDim.UP.value, :]

    if down_w_scale is not None:
        down_w_scale = down_w_scale[expert_idx]

    if gate_up_in_scale is not None:
        gate_up_in_scale = gate_up_in_scale[expert_idx]

    if down_in_scale is not None:
        down_in_scale = down_in_scale[expert_idx]
    return gate_w_scale, up_w_scale, down_w_scale, gate_up_in_scale, down_in_scale


def golden_all_expert_moe_tkg(
    inp_np,
    dtype,
    quant_dtype=None,
    quant_type=QuantizationType.NONE,
    act_fn_type=ActFnType.SiLU,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.NO_SCALE,
    bias=False,
    clamp=False,
    rank_id=0,
    mask_unselected_experts=False,
    # **kwargs
):
    """Golden function for all-expert MoE TKG kernel.

    Args:
        rank_id: Rank ID specifying which local experts this worker processes.
                 Local experts are [E_L * rank_id, E_L * (rank_id + 1)).
        mask_unselected_experts: If True, mask affinities based on expert_index (zero out
                         affinities for experts not selected by each token).
    """
    inp, gate_up_w, down_w, gate_up_b, down_b, expert_affinities = _extract_tensors(inp_np, bias)
    gate_w_scale, up_w_scale, down_w_scale, gate_up_in_scale, down_in_scale = _extract_quant_tensors(inp_np, quant_type)
    expert_index = inp_np["expert_index"]
    is_mx_kernel = is_dtype_mx(gate_up_w.dtype)

    clamp_limits = _get_clamp_limits(clamp)
    H = down_w.shape[-1]
    T, _ = expert_affinities.shape
    E_L = down_w.shape[0]  # Number of local experts from weight shape
    result = np.zeros((T, H), dtype=dtype)

    # Calculate expert offset based on rank_id
    expert_offset = rank_id * E_L

    # Slice affinities to local experts and apply masking if needed
    if expert_affinities_scaling_mode != ExpertAffinityScaleMode.NO_SCALE:
        # Slice to local experts: [T, E] -> [T, E_L]
        local_expert_affinities = expert_affinities[:, expert_offset : expert_offset + E_L].copy()

        # Apply masking based on expert_index when mask_unselected_experts=True
        if mask_unselected_experts:
            for expert_idx in range(E_L):
                global_expert_idx = expert_offset + expert_idx
                # Check if this expert was selected for each token
                expert_match = np.any(expert_index == global_expert_idx, axis=1).astype(dtype)
                local_expert_affinities[:, expert_idx] *= expert_match
    else:
        local_expert_affinities = expert_affinities[:, expert_offset : expert_offset + E_L]

    for local_expert_idx in range(E_L):
        active_in = inp
        affinity = local_expert_affinities[:, local_expert_idx : local_expert_idx + 1]

        active_in = _apply_pre_scale(active_in, affinity, expert_affinities_scaling_mode)

        if is_mx_kernel:
            # MX weight shapes: gate_up_w[local_expert_idx] = (pmax, 2, n_H512_tile, I)
            gate_w_select = gate_up_w[local_expert_idx][:, 0, :, :]
            up_w_select = gate_up_w[local_expert_idx][:, 1, :, :]
            gate_b_select = gate_up_b[local_expert_idx][:, 0, :, :] if bias else None
            up_b_select = gate_up_b[local_expert_idx][:, 1, :, :] if bias else None
            down_b_select = down_b[local_expert_idx] if bias else None
            down_w_select = down_w[local_expert_idx]

            # Load mx scales
            gate_scale = inp_np['expert_gate_up_weights_scale'][local_expert_idx][
                :, 0, :, :
            ]  # shape: [pmax/8, H/512, I]
            up_scale = inp_np['expert_gate_up_weights_scale'][local_expert_idx][:, 1, :, :]  # shape: [pmax/8, H/512, I]
            down_scale = inp_np['expert_down_weights_scale'][local_expert_idx]  # shape: [I_p/8, ceil(I/512), H]

            # For all-expert MX, we need to process each token separately
            mlp_out = np.zeros((T, H), dtype=dtype)
            for t in range(T):
                token_out = _compute_expert_mlp_mx(
                    active_in[t : t + 1],
                    gate_w_select,
                    gate_scale,
                    up_w_select,
                    up_scale,
                    down_w_select,
                    down_scale,
                    dtype,
                    act_fn_type,
                    gate_b_select,
                    up_b_select,
                    down_b_select,
                    clamp_limits,
                ).reshape(H)
                mlp_out[t] = token_out
        else:
            gate_w_select, up_w_select, down_w_select, gate_b_select, up_b_select, down_b_select = _select_tensors(
                gate_up_w, down_w, local_expert_idx, gate_up_b, down_b, bias
            )

            (
                gate_w_scale_select,
                up_w_scale_select,
                down_w_scale_select,
                gate_up_in_scale_select,
                down_in_scale_select,
            ) = _select_quant_tensors(
                gate_w_scale, up_w_scale, down_w_scale, gate_up_in_scale, down_in_scale, local_expert_idx
            )

            mlp_out = _compute_expert_mlp(
                active_in=active_in,
                gate_w_select=gate_w_select,
                up_w_select=up_w_select,
                down_w_select=down_w_select,
                gate_w_scale=gate_w_scale_select,
                up_w_scale=up_w_scale_select,
                down_w_scale=down_w_scale_select,
                gate_up_in_scale=gate_up_in_scale_select,
                down_in_scale=down_in_scale_select,
                dtype=dtype,
                quant_dtype=quant_dtype,
                quant_type=quant_type,
                act_fn_type=act_fn_type,
                gate_b=gate_b_select,
                up_b_select=up_b_select,
                down_b=down_b_select,
                clamp_limits=clamp_limits,
            )

        mlp_out = _apply_post_scale(mlp_out, affinity, expert_affinities_scaling_mode)
        result += mlp_out

    return {'out': dt.static_cast(result, dtype)}


def golden_selective_expert_moe_tkg(
    inp_np,
    dtype,
    quant_dtype=None,
    quant_type=QuantizationType.NONE,
    bias=False,
    clamp=False,
    act_fn_type=ActFnType.SiLU,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.NO_SCALE,
    # **kwargs
):
    inp, gate_up_w, down_w, gate_up_b, down_b, expert_affinities = _extract_tensors(inp_np, bias)
    gate_w_scale, up_w_scale, down_w_scale, gate_up_in_scale, down_in_scale = _extract_quant_tensors(inp_np, quant_type)
    is_mx_kernel = is_dtype_mx(gate_up_w.dtype)
    clamp_limits = _get_clamp_limits(clamp)
    expert_index = inp_np["expert_index"]
    H = down_w.shape[-1]
    T, K = expert_index.shape
    result = np.zeros((T, H), dtype=dtype)

    for t in range(T):
        for k in range(K):
            index = expert_index[t, k]
            affinity = expert_affinities[t, index]

            active_in = inp[t : t + 1]  # Keep 2D shape (1, H) instead of 1D (H,)
            active_in = _apply_pre_scale(active_in, affinity, expert_affinities_scaling_mode)

            if is_mx_kernel:
                gate_w_select = gate_up_w[index][:, 0, :, :]
                up_w_select = gate_up_w[index][:, 1, :, :]
                gate_b_select = gate_up_b[index][:, 0, :, :] if bias else None
                up_b_select = gate_up_b[index][:, 1, :, :] if bias else None
                down_b_select = down_b[index] if bias else None
                down_w_select = down_w[index]

                # Load mx scales
                gate_scale = inp_np['expert_gate_up_weights_scale'][index][:, 0, :, :]  # shape: [pmax/8, H/512, I]
                up_scale = inp_np['expert_gate_up_weights_scale'][index][:, 1, :, :]  # shape: [pmax/8, H/512, I]
                down_scale = inp_np['expert_down_weights_scale'][index]  # shape: [I_p/8, ceil(I/512), H]

                mlp_out = _compute_expert_mlp_mx(
                    active_in.reshape(1, H),
                    gate_w_select,
                    gate_scale,
                    up_w_select,
                    up_scale,
                    down_w_select,
                    down_scale,
                    dtype,
                    act_fn_type,
                    gate_b_select,
                    up_b_select,
                    down_b_select,
                    clamp_limits,
                ).reshape(H)

            else:
                gate_w_select, up_w_select, down_w_select, gate_b_select, up_b_select, down_b_select = _select_tensors(
                    gate_up_w, down_w, index, gate_up_b, down_b, bias
                )
                (
                    gate_w_scale_select,
                    up_w_scale_select,
                    down_w_scale_select,
                    gate_up_in_scale_select,
                    down_in_scale_select,
                ) = _select_quant_tensors(
                    gate_w_scale, up_w_scale, down_w_scale, gate_up_in_scale, down_in_scale, index
                )
                mlp_out = _compute_expert_mlp(
                    active_in=active_in,
                    gate_w_select=gate_w_select,
                    up_w_select=up_w_select,
                    down_w_select=down_w_select,
                    gate_w_scale=gate_w_scale_select,
                    up_w_scale=up_w_scale_select,
                    down_w_scale=down_w_scale_select,
                    gate_up_in_scale=gate_up_in_scale_select,
                    down_in_scale=down_in_scale_select,
                    dtype=dtype,
                    quant_dtype=quant_dtype,
                    quant_type=quant_type,
                    act_fn_type=act_fn_type,
                    gate_b=gate_b_select,
                    up_b_select=up_b_select,
                    down_b=down_b_select,
                    clamp_limits=clamp_limits,
                )
                mlp_out = mlp_out.squeeze()

            mlp_out = _apply_post_scale(mlp_out, affinity, expert_affinities_scaling_mode)
            result[t] += mlp_out

    return {"out": dt.static_cast(result, dtype)}
