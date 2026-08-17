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

"""Test-only input generation for the MXFP8 MoE forward tests.

The pure-PyTorch reference of the kernel itself lives in
``nkilib.experimental.moe_mxfp8.fwd.blockwise_mm_forward_mxfp8_torch``
(co-located with the kernel per coding guidelines). This module hosts the
test-fixture utilities: random input generation, routing-table construction,
and per-expert weight pre-quantization.

Unlike the backward, the forward *produces* the activation checkpoints, so this
builder does not seed them — it only generates the FFN inputs (hidden states,
weights, affinities) and the routing tables.
"""

import hashlib

import nki.language as nl
import numpy as np
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ShardOption,
    SkipMode,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.moe_mxfp8_checkpoint_config import MXFP8MOECheckpointConfig

from test.integration.nkilib.experimental.moe.test_bwmm_bwd_common import (
    generate_token_position_to_id_and_experts,
)

# Reuse the backward's helpers verbatim: block-count formula and per-expert
# weight pre-quantization produce byte-identical layouts on both sides.
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_bwd_test_utils import (
    _get_n_blocks,
    prequantize_moe_weights,
)


def build_mxfp8_moe_fwd_inputs(
    tokens,
    hidden,
    intermediate,
    expert,
    block_size,
    top_k,
    dtype=nl.bfloat16,
    run_with_lnc2=True,
    blocking_params=None,
    spill_reload=False,
    use_scale_packing=False,
    bias=False,
    clamp_limits=None,
    prequantize_weights=False,
    checkpoint_config=None,
    no_indirect_load=False,
):
    """Build the kernel-input dict for ``blockwise_mm_fwd_mxfp8``.

    Mirrors ``build_mxfp8_moe_bwd_inputs`` but for the forward: no
    ``output_hidden_states_grad`` and no checkpoint inputs (the forward produces
    those), and ``shard_option=SHARD_ON_BLOCK``. Random seeding uses the same
    parameter-hash scheme so a given shape is reproducible across runs.

    Returns:
        dict of kernel inputs. When ``prequantize_weights`` is True, returns
        ``(inputs, orig_gate_up_weight, orig_down_weight)`` so the test can feed
        the un-quantized weights to the golden (which is precision-agnostic).
    """
    T, H, I_TP, E, B = tokens, hidden, intermediate, expert, block_size
    if no_indirect_load:
        assert E == 1, f"no_indirect_load test inputs require expert=1, got {E}"
        assert top_k == 1, f"no_indirect_load test inputs require top_k=1, got {top_k}"
        assert T % B == 0, f"no_indirect_load test inputs require tokens divisible by block_size, got T={T}, B={B}"
        N = T // B
        dma_skip = SkipMode(False, False)
        token_experts = np.ones((T, E), dtype=np.int32)
        token_position_to_id = np.zeros((1,), dtype=np.int32)
        block_to_expert = np.zeros((1, 1), dtype=np.int32)
    else:
        N = _get_n_blocks(T, top_k, E, B)
        dma_skip = SkipMode(True, False)
        token_experts, token_position_to_id, block_to_expert = generate_token_position_to_id_and_experts(
            T, top_k, E, B, dma_skip, N
        )

    param_string = f"{T}_{top_k}_{B}_{E}_{I_TP}_{H}_mxfp8_fwd_{int(no_indirect_load)}"
    seed = int(hashlib.sha256(param_string.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    # Standard (backward-natural) weights, used by the golden:
    #   down:    [E, I_TP, H]      gate_up: [E, H, 2, I_TP]
    down_proj_weights = np.random.uniform(-0.1, 0.1, size=[E, I_TP, H]).astype(dtype)
    gate_and_up_proj_weights = np.random.uniform(-0.1, 0.1, size=[E, H, 2, I_TP]).astype(dtype)

    expert_affinities_masked = (np.random.random_sample([T, E]) * token_experts).astype(dtype)
    hidden_states = np.random.random_sample([T, H]).astype(dtype)

    # Forward-natural (transposed) weights, consumed by the kernel:
    #   gate_up: [E, H, 2, I_TP] -> [E, 2, I_TP, H]
    #   down:    [E, I_TP, H]    -> [E, H, I_TP]
    gate_up_fwd = np.ascontiguousarray(gate_and_up_proj_weights.transpose(0, 2, 3, 1))  # [E, 2, I_TP, H]
    down_fwd = np.ascontiguousarray(down_proj_weights.transpose(0, 2, 1))  # [E, H, I_TP]

    if prequantize_weights:
        # prequantize_moe_weights expects the standard layouts and produces the
        # x4 per-expert layouts the kernel's prequant path consumes.
        gate_up_data, gate_up_scales, down_data, down_scales = prequantize_moe_weights(
            gate_and_up_proj_weights, down_proj_weights, use_scale_packing
        )
    else:
        gate_up_data = gate_up_fwd
        gate_up_scales = None
        down_data = down_fwd
        down_scales = None

    inputs = {
        "hidden_states": hidden_states,
        "expert_affinities_masked": expert_affinities_masked.reshape(-1, 1),
        "gate_up_proj_weight": gate_up_data,
        "down_proj_weight": down_data,
        "token_position_to_id": token_position_to_id,
        "block_to_expert": block_to_expert.reshape(-1, 1),
        "block_size": block_size,
        "run_with_lnc2": run_with_lnc2,
        "affinity_option": AffinityOption.AFFINITY_ON_I,
        "shard_option": ShardOption.SHARD_ON_BLOCK,
        "activation_type": ActFnType.SiLU,
        "is_tensor_update_accumulating": False if no_indirect_load else top_k != 1,
        "no_indirect_load": no_indirect_load,
        "spill_reload": spill_reload,
        "use_scale_packing": use_scale_packing,
        "skip_dma": dma_skip,
        "bias": bias,
        "clamp_limits": clamp_limits,
        "checkpoint_config": checkpoint_config if checkpoint_config is not None else MXFP8MOECheckpointConfig(),
    }

    if gate_up_scales is not None:
        inputs["gate_up_weight_scales"] = gate_up_scales
    if down_scales is not None:
        inputs["down_weight_scales"] = down_scales

    if blocking_params is not None:
        inputs["gate_up_config"] = blocking_params.gate_up
        inputs["down_config"] = blocking_params.down

    if prequantize_weights:
        return inputs, gate_and_up_proj_weights, down_proj_weights

    return inputs
