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
from test.integration.nkilib.core.mlp.test_mlp_common import gen_moe_mx_weights
from test.integration.nkilib.utils.tensor_generators import generate_stabilized_mx_data
from test.integration.nkilib.utils.test_kernel_common import is_dtype_mx
from typing import Callable

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
from nkilib_src.nkilib.core.utils.common_types import (
    ExpertAffinityScaleMode,
    QuantizationType,
)

# Constants
_pmax = 128  # sbuf max partition dim
_q_width = 4  # quantization width
_q_height = 8  # quantization height


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
    expert_affinities_dtype=nl.bfloat16,
    bias=False,
    clamp=False,
    tensor_generator: Callable = None,
    rank_id: int = 0,
    mask_unselected_experts: bool = False,
    is_all_expert_dynamic: bool = False,
    routed_token_ratio: float = 1.0,
    block_size: int = None,
):
    """Build input tensors for MoE TKG kernel testing.

    Args:
        rank_id: Rank ID for all-expert mode (default 0, meaning all experts are local)
        mask_unselected_experts: Whether to apply affinity masking based on expert_index in all-expert mode
        routed_token_ratio: Fraction of tokens routed to experts (0.0 to 1.0). Default 1.0 routes all tokens.
        block_size: Block size for dynamic algorithm. Required when is_all_expert_dynamic=True.
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

                # Determine which tokens are routed based on routed_token_ratio
                num_routed = int(tokens * routed_token_ratio)
                routed_mask = np.zeros(tokens, dtype=bool)
                routed_mask[:num_routed] = True
                np.random.shuffle(routed_mask)

                # Create [T, E] with values at exp_idx positions only for routed tokens
                res = np.zeros((tokens, expert), dtype=inp.dtype)
                for i_t in range(tokens):
                    if routed_mask[i_t]:
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
    intermediate_p = math.ceil(intermediate / 4 / 8) * 8 if intermediate < 512 else _pmax  # for mxfp4
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
        down_w_scale_shape = (expert, math.ceil(intermediate_p / _q_height), n_I512_tile, hidden)
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
    expert_affinities = tensor_generator(TensorTemplate((tokens, expert), expert_affinities_dtype, "expert_affinities"))
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
        "is_all_expert_dynamic": is_all_expert_dynamic,
        "block_size": block_size,
    }
    return kernel_input


def get_expert_affinity_dtype(is_all_expert):
    """Get expert affinity initialization dtype based on expert MLP algorithm."""
    # FIXME: ensure selective load is compatible with non fp32 expert affinity dtypes, default to bf16 for all algorithms
    return nl.bfloat16 if is_all_expert else nl.float32


def _get_clamp_limits(clamp):
    """Get clamp limits for gate and up projections."""
    return (
        float(7.0) if clamp else None,  # gate_clamp_upper_limit
        None,  # gate_clamp_lower_limit
        float(8.0) if clamp else None,  # up_clamp_upper_limit
        float(-6.0) if clamp else None,  # up_clamp_lower_limit
    )
