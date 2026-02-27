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

"""
Test utilities for MoE BWMM MXFP4/MXFP8 CTE kernel tests.

Provides input builders and golden functions for testing the blockwise
matrix multiplication kernel with MXFP4/MXFP8 quantization.
"""

import hashlib
import os
import pickle
from test.integration.nkilib.core.mlp.test_mlp_common import (
    _down_proj_golden_mx,
    _gate_up_proj_golden_mx,
)
from test.integration.nkilib.core.moe.moe_cte.test_moe_cte import (
    generate_token_position_to_id_and_experts,
    get_n_blocks,
    map_skip_mode,
)
from test.integration.nkilib.utils.tensor_generators import generate_stabilized_mx_data
from test.integration.nkilib.utils.test_kernel_common import (
    act_fn_type2func,
)
from test.utils.mx_utils import is_mx_quantize
from typing import Optional

import nki.language as nl
import numpy as np
from nkilib_src.nkilib.core.moe.moe_cte.moe_cte_utils import SkipMode
from nkilib_src.nkilib.core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
)
from nkilib_src.nkilib.core.utils.kernel_assert import kernel_assert

# MXFP4 quantization block dimensions
_q_width = 4  # quantization width
_q_height = 8  # quantization height
_pmax = 128  # sbuf max partition dim (128)

# Explicit parameter ordering per kernel variant, matching kernel function signatures.
# From bwmm_shard_on_block_mx.py::bwmm_shard_on_block_mx
_SHARD_ON_BLOCK_MX_ORDER = [
    'hidden_states',
    'expert_affinities_masked',
    'gate_up_proj_weight',
    'down_proj_weight',
    'token_position_to_id',
    'block_to_expert',
    'conditions',
    'gate_and_up_proj_bias',
    'down_proj_bias',
    'gate_up_proj_scale',
    'down_proj_scale',
    'block_size',
    'n_static_blocks',
    'n_dynamic_blocks',
    'gate_up_activations_T',
    'down_activations',
    'activation_function',
    'skip_dma',
    'compute_dtype',
    'weight_dtype',
    'is_tensor_update_accumulating',
    'expert_affinities_scaling_mode',
    'gate_clamp_upper_limit',
    'gate_clamp_lower_limit',
    'up_clamp_lower_limit',
    'up_clamp_upper_limit',
]

# From bwmm_shard_on_I_mx.py::blockwise_mm_shard_intermediate_mx
_SHARD_ON_I_MX_ORDER = [
    'hidden_states',
    'expert_affinities_masked',
    'gate_up_proj_weight',
    'down_proj_weight',
    'token_position_to_id',
    'block_to_expert',
    'gate_and_up_proj_bias',
    'down_proj_bias',
    'gate_up_proj_scale',
    'down_proj_scale',
    'block_size',
    'activation_function',
    'skip_dma',
    'compute_dtype',
    'weight_dtype',
    'is_tensor_update_accumulating',
    'expert_affinities_scaling_mode',
    'gate_clamp_upper_limit',
    'gate_clamp_lower_limit',
    'up_clamp_lower_limit',
    'up_clamp_upper_limit',
]

# From bwmm_shard_on_I_mx.py::blockwise_mm_shard_intermediate_mx_hybrid
_SHARD_ON_I_MX_HYBRID_ORDER = [
    'conditions',
    'hidden_states',
    'expert_affinities_masked',
    'gate_up_proj_weight',
    'down_proj_weight',
    'token_position_to_id',
    'block_to_expert',
    'gate_and_up_proj_bias',
    'down_proj_bias',
    'gate_up_proj_scale',
    'down_proj_scale',
    'block_size',
    'num_static_block',
    'activation_function',
    'skip_dma',
    'compute_dtype',
    'weight_dtype',
    'is_tensor_update_accumulating',
    'expert_affinities_scaling_mode',
    'gate_clamp_upper_limit',
    'gate_clamp_lower_limit',
    'up_clamp_lower_limit',
    'up_clamp_upper_limit',
]

_KERNEL_INPUT_ORDER = {
    'shard_on_block_mx': _SHARD_ON_BLOCK_MX_ORDER,
    'shard_on_I_mx': _SHARD_ON_I_MX_ORDER,
    'shard_on_I_mx_hybrid': _SHARD_ON_I_MX_HYBRID_ORDER,
}


def order_kernel_input(kernel_input, variant):
    """Reorder kernel_input dict to match kernel function signature ordering.

    Args:
        kernel_input: Dict from build_moe_bwmm_mx_cte.
        variant: One of 'shard_on_block_mx', 'shard_on_I_mx', 'shard_on_I_mx_hybrid'.

    Returns:
        New dict with keys ordered to match the kernel's parameter list.
        Internal keys (prefixed with '_') are excluded.
    """
    key_order = _KERNEL_INPUT_ORDER[variant]
    ordered = {}
    for key in key_order:
        if key in kernel_input:
            ordered[key] = kernel_input[key]
    for key in kernel_input:
        if key not in ordered and not key.startswith('_'):
            ordered[key] = kernel_input[key]
    return ordered


# Golden cache directory (same pattern as test_nki_moe.py)
_GOLDEN_CACHE_DIR = os.path.expanduser('~/unit_test_input_golden_cache/moe_bwmm_mxfp4_cte')


def _compute_golden_cache_key(
    H: int,
    T: int,
    E: int,
    B: int,
    TOPK: int,
    I_TP: int,
    dtype,
    weight_dtype,
    skip_mode: int,
    bias: bool,
    activation_function,
    expert_affinities_scaling_mode,
    is_dynamic: bool,
    vnc_degree: int,
    gate_clamp_upper_limit,
    gate_clamp_lower_limit,
    up_clamp_upper_limit,
    up_clamp_lower_limit,
    alpha,
) -> str:
    """Compute a hash key from test parameters for caching."""
    key_data = (
        H,
        T,
        E,
        B,
        TOPK,
        I_TP,
        str(dtype),
        str(weight_dtype),
        skip_mode,
        bias,
        activation_function.value if hasattr(activation_function, 'value') else activation_function,
        expert_affinities_scaling_mode.value
        if hasattr(expert_affinities_scaling_mode, 'value')
        else expert_affinities_scaling_mode,
        is_dynamic,
        vnc_degree,
        gate_clamp_upper_limit,
        gate_clamp_lower_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
        alpha,
    )
    key_str = str(key_data)
    return hashlib.sha256(key_str.encode()).hexdigest()[:16]


# Input Builder
def build_moe_bwmm_mx_cte(
    H: int,
    T: int,
    E: int,
    B: int,
    TOPK: int,
    I_TP: int,
    dtype=nl.bfloat16,
    weight_dtype=nl.float4_e2m1fn_x4,
    skip_mode: int = 0,
    bias: bool = False,
    activation_function: ActFnType = ActFnType.SiLU,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    is_dynamic: bool = False,
    vnc_degree: int = 2,
    n_dynamic_blocks: int = 55,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    alpha: Optional[float] = None,
    use_cache: bool = False,
    is_shard_on_I: bool = False,
) -> dict:
    """
    Build input tensors for MoE BWMM MXFP4/MXFP8 CTE kernel testing.

    Args:
        H: Hidden dimension size
        T: Total number of tokens
        E: Number of experts
        B: Block size (tokens per block)
        TOPK: Top-K experts per token
        I_TP: Intermediate size per TP degree
        dtype: Data type for activations
        weight_dtype: Data type for weights (e.g., nl.float4_e2m1fn_x4 for MXFP4,
                     nl.float8_e4m3fn_x4 or nl.float8_e5m2_x4 for MXFP8)
        skip_mode: DMA skip mode (0-3)
        bias: Whether to include bias tensors
        activation_function: Activation function type
        expert_affinities_scaling_mode: Expert affinity scaling mode
        is_dynamic: Whether to use dynamic loop
        vnc_degree: LNC sharding degree
        n_dynamic_blocks: Number of blocks to process with dynamic loop (default: 55)
        gate_clamp_upper_limit: Upper clamp limit for gate projection
        gate_clamp_lower_limit: Lower clamp limit for gate projection
        up_clamp_upper_limit: Upper clamp limit for up projection
        up_clamp_lower_limit: Lower clamp limit for up projection
        alpha: Expert distribution sparsity parameter (None for uniform distribution)
        use_cache: Whether to use cached inputs if available (default: False)

    Returns:
        Dictionary with all kernel input tensors and parameters
    """
    # Check for cached inputs
    cache_key = _compute_golden_cache_key(
        H,
        T,
        E,
        B,
        TOPK,
        I_TP,
        dtype,
        weight_dtype,
        skip_mode,
        bias,
        activation_function,
        expert_affinities_scaling_mode,
        is_dynamic,
        vnc_degree,
        gate_clamp_upper_limit,
        gate_clamp_lower_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
        alpha,
    )
    cache_file = os.path.join(_GOLDEN_CACHE_DIR, f"input_{cache_key}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Found cached inputs in {cache_file}, reusing...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    np.random.seed(0)

    dma_skip = map_skip_mode(skip_mode)
    if is_shard_on_I:
        N = get_n_blocks(T, TOPK, E, B, n_block_per_iter=1)
    else:
        N = get_n_blocks(T, TOPK, E, B, n_block_per_iter=vnc_degree)

    # Generate token assignments
    expert_masks, token_position_to_id, block_to_expert, conditions = generate_token_position_to_id_and_experts(
        T,
        TOPK,
        E,
        B,
        dma_skip,
        N,
        vnc_degree=vnc_degree,
        alpha=alpha,
        is_block_parallel=False if is_shard_on_I else True,
        quantize=weight_dtype,
    )

    # Calculate MXFP4 tensor dimensions
    kernel_assert(H % (_pmax * _q_width) == 0, f"H must be divisible by {_pmax * _q_width}, got {H}")
    n_H512_tile = H // (_pmax * _q_width)

    kernel_assert(
        I_TP % (_pmax * _q_width) == 0 or (I_TP < (_pmax * _q_width) and I_TP % (_q_height * _q_width) == 0),
        f"I_TP must be divisible by {_pmax * _q_width} or (I_TP < {_pmax * _q_width} and I_TP divisible by {_q_height * _q_width}), got {I_TP}",
    )
    n_I512_tile, r_I512_tile = divmod(I_TP, _pmax * _q_width)
    I_TP_par_dim = _pmax
    if r_I512_tile > 0:
        kernel_assert(n_I512_tile == 0, f"Expected n_I512_tile == 0 when remainder exists, got {n_I512_tile}")
        n_I512_tile = 1
        I_TP_par_dim = r_I512_tile // _q_width

    # Generate hidden states with MXFP4-compatible layout
    # When skip_token is True, we use T tokens; otherwise T+1 (with padding token)
    if dma_skip.skip_token:
        hidden_T = T
    else:
        hidden_T = T + 1  # Include padding token

    hidden_states_fp32, _, _ = generate_stabilized_mx_data(
        mx_dtype=nl.float8_e4m3fn_x4,
        shape=(hidden_T * n_H512_tile * _pmax, _q_width),
        val_range=5,
    )
    hidden_states = (
        hidden_states_fp32.reshape(hidden_T, n_H512_tile, _pmax, _q_width)
        .transpose(0, 3, 1, 2)
        .reshape(hidden_T, H)
        .astype(dtype)
    )

    # Zero out padding token (only when not skipping tokens)
    if not dma_skip.skip_token:
        hidden_states[T, :] = 0

    # Generate expert affinities
    if dma_skip.skip_token:
        expert_affinities_masked = np.random.random_sample([T, E]).astype(dtype)
        expert_affinities_masked = (expert_affinities_masked * expert_masks).astype(dtype)
    else:
        expert_affinities_masked = np.random.random_sample([T + 1, E]).astype(dtype)
        expert_affinities_masked[:T] = (expert_affinities_masked[:T] * expert_masks).astype(dtype)
        expert_affinities_masked[T] = 0  # Zero padding token affinities

    # Generate MXFP4 gate/up projection weights
    gate_up_proj_weights_fp32, gate_up_proj_weights, gate_up_proj_scale = generate_stabilized_mx_data(
        mx_dtype=weight_dtype,
        shape=(E * _pmax, 2 * n_H512_tile * I_TP * _q_width),
    )
    gate_up_proj_weights = gate_up_proj_weights.reshape(E, _pmax, 2, n_H512_tile, I_TP)
    gate_up_proj_scale = gate_up_proj_scale.reshape(E, _pmax // _q_height, 2, n_H512_tile, I_TP)

    # Generate MXFP4 down projection weights
    down_proj_weights_fp32, down_proj_weights, down_proj_scale = generate_stabilized_mx_data(
        mx_dtype=weight_dtype,
        shape=(E * I_TP_par_dim, n_I512_tile * H * _q_width),
    )
    down_proj_weights = down_proj_weights.reshape(E, I_TP_par_dim, n_I512_tile, H)
    down_proj_scale = down_proj_scale.reshape(E, I_TP_par_dim // _q_height, n_I512_tile, H)

    # Build kernel input dictionary in exact KLIR test order
    # Order must match build_blockwise_mm input_list:
    # [hidden_states, expert_affinities, gate_and_up_proj_weights, down_proj_weights,
    #  token_position_to_id, block_to_expert]
    # then: conditions (if dynamic), bias tensors (if bias), scale tensors (if quantize)

    kernel_input = {
        'hidden_states': hidden_states,
        'expert_affinities_masked': expert_affinities_masked.reshape(-1, 1),
        'gate_up_proj_weight': gate_up_proj_weights,
        'down_proj_weight': down_proj_weights,
        'block_size': B,
        'token_position_to_id': token_position_to_id,
        'block_to_expert': block_to_expert,
        'skip_dma': dma_skip,
        'compute_dtype': dtype,
        'is_tensor_update_accumulating': TOPK != 1,
        'expert_affinities_scaling_mode': expert_affinities_scaling_mode,
    }

    if is_dynamic and not is_shard_on_I:
        kernel_input['n_dynamic_blocks'] = n_dynamic_blocks
    # Add clamp limits only if they have non-None values
    if gate_clamp_upper_limit is not None:
        kernel_input['gate_clamp_upper_limit'] = gate_clamp_upper_limit
    if gate_clamp_lower_limit is not None:
        kernel_input['gate_clamp_lower_limit'] = gate_clamp_lower_limit
    if up_clamp_lower_limit is not None:
        kernel_input['up_clamp_lower_limit'] = up_clamp_lower_limit
    if up_clamp_upper_limit is not None:
        kernel_input['up_clamp_upper_limit'] = up_clamp_upper_limit

    # Add activation function after clamp limits
    kernel_input['activation_function'] = activation_function

    # Add weight_dtype to specify target MXFP format
    kernel_input['weight_dtype'] = weight_dtype

    # Add dynamic conditions BEFORE bias (matches build_blockwise_mm order)
    if is_dynamic:
        kernel_input['conditions'] = conditions

    # Add bias tensors (matches build_blockwise_mm order: gate_and_up_proj_bias, down_proj_bias)
    if bias:
        gate_and_up_proj_bias = np.random.uniform(
            -2.0625, 0.52, size=(E, I_TP_par_dim, 2, n_I512_tile, _q_width)
        ).astype(dtype)
        down_proj_bias = np.random.uniform(-1.632, 1.4375, size=[E, H]).astype(dtype)
        kernel_input['gate_and_up_proj_bias'] = gate_and_up_proj_bias
        kernel_input['down_proj_bias'] = down_proj_bias

    # Add scale tensors AFTER bias (matches build_blockwise_mm order)
    kernel_input['gate_up_proj_scale'] = gate_up_proj_scale
    kernel_input['down_proj_scale'] = down_proj_scale

    # Store additional data needed for golden computation
    kernel_input['_internal'] = {
        'gate_up_proj_weights_fp32': gate_up_proj_weights_fp32,
        'down_proj_weights_fp32': down_proj_weights_fp32,
        'expert_masks': expert_masks,
        'N': N,
        'n_H512_tile': n_H512_tile,
        'n_I512_tile': n_I512_tile,
        'I_TP_par_dim': I_TP_par_dim,
    }

    # Cache the generated inputs for future reuse
    if use_cache:
        try:
            os.makedirs(_GOLDEN_CACHE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(kernel_input, f)
            print(f"Cached inputs saved to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to cache inputs to {cache_file}: {e}")

    return kernel_input


def generate_blockwise_numpy_golden(
    test_config,
    expert_affinities,
    down_proj_weights,
    token_position_to_id,
    block_to_expert,
    gate_and_up_proj_weights,
    hidden_states,
    T,
    H,
    B,
    N,
    E,
    I_TP,
    dtype,
    dma_skip: SkipMode,
    quantize=False,
    quantize_strategy=5,
    expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
    activation_function=ActFnType.SiLU,
    gate_up_proj_bias=None,
    down_proj_bias=None,
    gate_up_proj_scale=np.empty([]),
    down_proj_scale=np.empty([]),
    checkpoint_activation=False,
    separate_outputs=False,
    conditions=None,
    n_block_per_iter=1,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    DBG_KERNEL=False,
    kernel_input=None,
):
    DBG_golden_tensors = {}

    lnc_degree = 1 if test_config.target_instance_family == 'trn1' else 2
    output_shape = [lnc_degree, T + 1, H] if separate_outputs else [T + 1, H]
    output_np = np.zeros(output_shape).astype(dtype)
    token_position_to_id = token_position_to_id.reshape(N, B)

    if checkpoint_activation:
        gate_up_activations_T = np.zeros([N, 2, I_TP, B]).astype(dtype)
        down_activations = np.zeros([N, B, H]).astype(dtype)

    if dma_skip.skip_weight:
        is_weight_same_as_prev = np.zeros((N))
        is_weight_same_as_prev[1:] = block_to_expert[1:] == block_to_expert[:-1]
        is_weight_same_as_prev = is_weight_same_as_prev.astype(np.uint8)

    gate_up_weights = None
    down_weights = None
    if quantize and quantize_strategy == 5:
        # FAL should write the down scales to 128, H//128 in column major order so our load is faster
        down_proj_scale = np.transpose(down_proj_scale.reshape((E, 128, H // 128)), (0, 2, 1)).reshape((E, 1, H))

    if not is_mx_quantize(quantize):
        E_local = gate_and_up_proj_weights.shape[0]
        gate_and_up_proj_weights = gate_and_up_proj_weights.reshape(E_local, H, 2 * I_TP).astype(np.float32)
        down_proj_weights = down_proj_weights.astype(np.float32)

    for b in range(N):
        if conditions is not None and conditions[b] == 0:
            break

        local_token_position_to_id = token_position_to_id[b, :]
        # [B, H]
        if dma_skip.skip_token:
            zeros_hidden = np.zeros((1, H)).astype(dtype)
            hidden_states = np.concatenate([hidden_states, zeros_hidden], axis=0)
            zeros_exaf = np.zeros((1, E)).astype(dtype)
            expert_affinities = np.concatenate([expert_affinities, zeros_exaf], axis=0)

        local_hidden_states = hidden_states[local_token_position_to_id[:], :].astype(np.float32)
        if DBG_KERNEL and b == 0:
            """
            1. MoE kernel will load and transpose the input tensor,
            2. Input layout in HBM: [T, 4_H, H/512, 16_H, 8_H] 
            3. Load input to SBUF: [32_T * 4_H (P), T/32, H/512, 16_H * 8_H]
            4. T/32 * H/512 number of transpose operations will be performed to swap the outermost and innermost dims of above SBUF layout
            5. Obtaining the swizzle layout: [16_H * 8_H(P), H/512, T/32, 32_T * 4_H]
            """

            # hidden = hidden.reshape(BxS, _q_width, H // _pmax // _q_width, _pmax).transpose(3, 2, 0, 1).reshape(_pmax, -1)
            DBG_golden_tensors['dbg_hidden_states'] = (
                local_hidden_states.reshape(
                    -1,  # 0: T_div_32
                    32 * _q_width,  # 1: 128
                    H // _pmax // _q_width,  # 2: H_div_512
                    _pmax,  # 3: 128
                )
                .transpose(3, 2, 0, 1)
                .astype(dtype)
            )

        expert_idx = block_to_expert[b]
        local_expert_affinities = expert_affinities[local_token_position_to_id, expert_idx].reshape(-1, 1).astype(dtype)

        if expert_affinities_scaling_mode in [
            ExpertAffinityScaleMode.PRE_SCALE,
            ExpertAffinityScaleMode.PRE_SCALE_DELAYED,
        ]:
            local_hidden_states = local_expert_affinities * local_hidden_states

        if dma_skip.skip_weight:
            expert_idx = E if is_weight_same_as_prev[b] else expert_idx

        # Select expert weights, scale, and bias
        if expert_idx < E:  # weight skip
            # [H, 2, I]
            gate_up_weights = gate_and_up_proj_weights[expert_idx]
            # [H, I]
            down_weights = down_proj_weights[expert_idx, :, :]
            if (quantize and gate_up_proj_scale.shape[0] == E) or is_mx_quantize(quantize):
                gup_scale = gate_up_proj_scale[expert_idx]
                down_scale = down_proj_scale[expert_idx]
            if gate_up_proj_bias is not None:
                gate_up_bias = gate_up_proj_bias[expert_idx]

            if down_proj_bias is not None:
                down_bias = down_proj_bias[expert_idx]

        if is_mx_quantize(quantize):

            class SimpleConfig:
                def __init__(self, H, I, BxS):
                    self.H = H
                    self.I = I
                    self.BxS = BxS

            # Use original MXFP4 tensors directly, not logical layout
            gate_up_weights_mxfp4 = kernel_input['gate_up_proj_weight'][expert_idx]  # [_pmax, 2, n_H512_tile, I_TP]
            gup_scale_mxfp4 = kernel_input['gate_up_proj_scale'][expert_idx]  # [_pmax//8, 2, n_H512_tile, I_TP]

            gate_activation = _gate_up_proj_golden_mx(
                hidden=local_hidden_states,
                hidden_scale=None,
                weight=gate_up_weights_mxfp4[:, 0, :, :],  # [_pmax, n_H512_tile, I_TP]
                weight_scale=gup_scale_mxfp4[:, 0, :, :],
                bias=gate_up_bias[:, 0, :, :] if gate_up_proj_bias is not None else None,
                cfg=SimpleConfig(H, I_TP, B),
            )
            up_activation = _gate_up_proj_golden_mx(
                hidden=local_hidden_states,
                hidden_scale=None,
                weight=gate_up_weights_mxfp4[:, 1, :, :],  # [_pmax, n_H512_tile, I_TP]
                weight_scale=gup_scale_mxfp4[:, 1, :, :],
                bias=gate_up_bias[:, 1, :, :] if gate_up_proj_bias is not None else None,
                cfg=SimpleConfig(H, I_TP, B),
            )
        else:
            # [B, 2, I]
            gate_up_activation = np.matmul(local_hidden_states, gate_up_weights).reshape(B, 2, I_TP)
            gate_activation = gate_up_activation[:, 0, :]
            up_activation = gate_up_activation[:, 1, :]

        """
        Compute intermediate state = 
        silu(dequantize(gate_proj) + gate_bias) * (dequantize(up_proj) + up_bias) #when activation function is silu
        or
        swiglu(dequantize(gate_proj) + gate_bias) * (dequantize(up_proj) + up_bias) #when activation function is swiglu
        Note that we expect (up_bias = up_bias + 1) (ie FAL should have added 1 to it before calling the kernel)
        """
        if not is_mx_quantize(quantize):  # Dequantization and bias are done in _gate_up_proj_golden_mx.
            # [B, I_TP] - Dequantize before adding bias
            if quantize and gate_up_proj_scale.shape[0] == 1:
                gate_activation *= gate_up_proj_scale.squeeze()[:I_TP]
                up_activation *= gate_up_proj_scale.squeeze()[I_TP:]
            elif quantize and gate_up_proj_scale.shape[0] == E:
                gate_activation *= gup_scale[0, :I_TP]
                up_activation *= gup_scale[0, I_TP:]

            # Add bias
            if gate_up_proj_bias is not None:
                gate_activation += gate_up_bias[0, :]
                up_activation += gate_up_bias[1, :]

        if gate_clamp_lower_limit is not None or gate_clamp_upper_limit is not None:
            np.clip(gate_activation, a_min=gate_clamp_lower_limit, a_max=gate_clamp_upper_limit, out=gate_activation)
        if up_clamp_lower_limit is not None or up_clamp_upper_limit is not None:
            np.clip(up_activation, a_min=up_clamp_lower_limit, a_max=up_clamp_upper_limit, out=up_activation)

        # Debug goldens for gate_proj and up_proj (after clipping)
        if DBG_KERNEL and b == 0:
            n_I512_tile = max(1, I_TP // (_pmax * _q_width))
            flatten_free_dim = n_I512_tile * B * _q_width
            DBG_golden_tensors['dbg_gate_proj'] = gate_activation.reshape(_pmax, flatten_free_dim).astype(dtype)
            DBG_golden_tensors['dbg_up_proj'] = up_activation.reshape(_pmax, flatten_free_dim).astype(dtype)

        if checkpoint_activation:
            assert not is_mx_quantize(quantize)
            gate_up_activations_T[b] = gate_up_activation.transpose(1, 2, 0)

        act_res = act_fn_type2func[activation_function](gate_activation)

        # Debug golden for gate_after_act
        if DBG_KERNEL and b == 0:
            pass

        multiply_1 = act_res * up_activation

        if is_mx_quantize(quantize):
            # Use original MXFP4 tensors directly for down projection
            down_weights_mxfp4 = kernel_input['down_proj_weight'][expert_idx]  # [I_TP_par_dim, n_I512_tile, H]
            down_scale_mxfp4 = kernel_input['down_proj_scale'][expert_idx]  # [I_TP_par_dim//8, n_I512_tile, H]

            down_activation = _down_proj_golden_mx(
                multiply_1,
                down_weights_mxfp4,
                down_scale_mxfp4,
                down_bias if down_proj_bias is not None else None,
                SimpleConfig(H, I_TP, B),
            )
        else:
            down_activation = np.matmul(multiply_1, down_weights)

        """
        dequantize and add bias to down projection
        """
        if not is_mx_quantize(quantize):  # Dequantization and bias are done in _down_proj_golden_mx.
            # dequantize before adding bias
            if quantize and gate_up_proj_scale.shape[0] == 1:
                # [B, H]
                # Quantize down proj before expert affinities
                down_activation = down_activation * down_proj_scale.squeeze()
            elif quantize and gate_up_proj_scale.shape[0] == E:
                down_activation = down_activation * down_scale[0, :]

            # add bias
            if down_proj_bias is not None:
                down_activation += down_bias

        # Debug golden for down_proj (after bias)
        if DBG_KERNEL and b == 0:
            n_B128_tiles = (B + _pmax - 1) // _pmax
            DBG_golden_tensors['dbg_down_proj'] = (
                down_activation.reshape(n_B128_tiles, _pmax, H).transpose(1, 0, 2).astype(dtype)
            )

        if checkpoint_activation:
            down_activations[b] = down_activation

        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE:
            scale = down_activation * local_expert_affinities
        else:
            scale = down_activation

        if separate_outputs:
            output_np[0, local_token_position_to_id[:], :] += scale.astype(output_np.dtype)
        else:
            output_np[local_token_position_to_id[:], :] += scale.astype(output_np.dtype)

    if separate_outputs:
        out_return = output_np[:, :T, :] if dma_skip.skip_token else output_np
    else:
        out_return = output_np[:T, :] if dma_skip.skip_token else output_np

    if checkpoint_activation:
        return out_return, gate_up_activations_T, down_activations

    return out_return, DBG_golden_tensors


def golden_moe_bwmm_mx_cte(
    kernel_input: dict,
    dtype,
    lnc_degree: int = 2,
    use_cache: bool = False,
    is_shard_on_I: bool = False,
) -> dict:
    """
    Compute golden output for MoE BWMM MXFP4/MXFP8 CTE kernel.

    Uses generate_blockwise_numpy_golden from the original framework
    to ensure 100% matching behavior.

    Args:
        kernel_input: Dictionary with kernel input tensors (from build_moe_bwmm_mx_cte)
        dtype: Data type for output
        lnc_degree: LNC sharding degree
        use_cache: Whether to use cached goldens if available (default: False)

    Returns:
        Dictionary with golden output tensors
    """
    # Compute cache key from kernel_input parameters
    internal = kernel_input['_internal']
    E = kernel_input['gate_up_proj_weight'].shape[0]
    H = kernel_input['hidden_states'].shape[1]
    I_TP = kernel_input['gate_up_proj_weight'].shape[-1]
    T_dim = kernel_input['hidden_states'].shape[0]
    T = T_dim if kernel_input['skip_dma'].skip_token else T_dim - 1
    B = kernel_input['block_size']

    cache_key = _compute_golden_cache_key(
        H,
        T,
        E,
        B,
        TOPK=4 if kernel_input.get('is_tensor_update_accumulating', False) else 1,  # Infer TOPK
        I_TP=I_TP,
        dtype=dtype,
        weight_dtype=kernel_input['gate_up_proj_weight'].dtype,
        skip_mode=(1 if kernel_input['skip_dma'].skip_token else 0)
        + (2 if kernel_input['skip_dma'].skip_weight else 0),
        bias='gate_and_up_proj_bias' in kernel_input,
        activation_function=kernel_input['activation_function'],
        expert_affinities_scaling_mode=kernel_input['expert_affinities_scaling_mode'],
        is_dynamic='conditions' in kernel_input,
        vnc_degree=lnc_degree,
        gate_clamp_upper_limit=kernel_input.get('gate_clamp_upper_limit'),
        gate_clamp_lower_limit=kernel_input.get('gate_clamp_lower_limit'),
        up_clamp_upper_limit=kernel_input.get('up_clamp_upper_limit'),
        up_clamp_lower_limit=kernel_input.get('up_clamp_lower_limit'),
        alpha=None,  # Not stored in kernel_input
    )
    cache_file = os.path.join(_GOLDEN_CACHE_DIR, f"golden_{cache_key}.pkl")

    if use_cache and os.path.exists(cache_file):
        print(f"Found cached golden in {cache_file}, reusing...")
        with open(cache_file, 'rb') as f:
            return pickle.load(f)

    # Create mock test_config
    class MockTestConfig:
        def __init__(self, lnc_degree):
            self.target_instance_family = 'trn1' if lnc_degree == 1 else 'trn2'

    test_config = MockTestConfig(lnc_degree)

    # Extract data from kernel_input
    internal = kernel_input['_internal']

    E = kernel_input['gate_up_proj_weight'].shape[0]
    H = kernel_input['hidden_states'].shape[1]
    I_TP = kernel_input['gate_up_proj_weight'].shape[-1]

    # For MXFP4, don't convert to logical layout - use original tensors directly
    # The golden function expects MXFP4 format when is_mx_quantize() is true
    gate_and_up_proj_weights_for_golden = kernel_input['gate_up_proj_weight']  # Keep MXFP4 format
    down_proj_weights_for_golden = kernel_input['down_proj_weight']  # Keep MXFP4 format

    # Determine dimensions
    T_dim = kernel_input['hidden_states'].shape[0]
    T = T_dim if kernel_input['skip_dma'].skip_token else T_dim - 1
    B = kernel_input['block_size']
    N = internal['N']
    is_accumulating = kernel_input.get('is_tensor_update_accumulating', False)

    # Call the original golden function
    output_np, dbg_tensors = generate_blockwise_numpy_golden(
        test_config=test_config,
        expert_affinities=kernel_input['expert_affinities_masked'].reshape(-1, E),
        down_proj_weights=down_proj_weights_for_golden,
        token_position_to_id=kernel_input['token_position_to_id'],
        block_to_expert=kernel_input['block_to_expert'],
        gate_and_up_proj_weights=gate_and_up_proj_weights_for_golden,
        hidden_states=kernel_input['hidden_states'],
        T=T,
        H=H,
        B=B,
        N=N,
        E=E,
        I_TP=I_TP,
        dtype=dtype,
        dma_skip=kernel_input['skip_dma'],
        quantize=kernel_input['gate_up_proj_weight'].dtype,  # MXFP4 quantization
        quantize_strategy=6,  # Strategy 6 for MXFP4
        expert_affinities_scaling_mode=kernel_input['expert_affinities_scaling_mode'],
        activation_function=kernel_input['activation_function'],
        gate_up_proj_bias=kernel_input.get('gate_and_up_proj_bias'),
        down_proj_bias=kernel_input.get('down_proj_bias'),
        gate_up_proj_scale=kernel_input['gate_up_proj_scale'],
        down_proj_scale=kernel_input['down_proj_scale'],
        checkpoint_activation=False,
        separate_outputs=is_accumulating and not is_shard_on_I,
        conditions=kernel_input.get('conditions'),
        n_block_per_iter=1,
        gate_clamp_upper_limit=kernel_input.get('gate_clamp_upper_limit'),
        gate_clamp_lower_limit=kernel_input.get('gate_clamp_lower_limit'),
        up_clamp_lower_limit=kernel_input.get('up_clamp_lower_limit'),
        up_clamp_upper_limit=kernel_input.get('up_clamp_upper_limit'),
        DBG_KERNEL=False,
        kernel_input=kernel_input,
    )

    # Return golden output with debug tensors
    golden_output = {'output': output_np}
    for key, value in dbg_tensors.items():
        golden_output[key] = value

    # Cache the generated golden for future reuse
    if use_cache:
        try:
            os.makedirs(_GOLDEN_CACHE_DIR, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(golden_output, f)
            print(f"Cached golden saved to {cache_file}")
        except Exception as e:
            print(f"Warning: Failed to cache golden to {cache_file}: {e}")

    return golden_output
