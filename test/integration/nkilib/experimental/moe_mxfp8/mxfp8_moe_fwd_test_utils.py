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

# Reuse the backward's block-count formula. The forward's weight *layouts* differ
# from the backward's (it contracts over different axes), so it needs its own
# per-expert prequantizer, built here on the shared 2D quantize/swizzle/scale
# primitives so it can select the WRAPX vs 1x32 swizzle.
import nki.language as nl
import numpy as np
from nkilib_src.nkilib.experimental.matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from nkilib_src.nkilib.experimental.moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)
from nkilib_src.nkilib.experimental.moe_mxfp8.config_helpers import ExecutionMode
from nkilib_src.nkilib.experimental.moe_mxfp8.fwd.config import MXFP8MOEFwdConfig
from nkilib_src.nkilib.experimental.moe_mxfp8.moe_mxfp8_checkpoint_config import MXFP8MOECheckpointConfig
from nkilib_src.nkilib.experimental.mxfp_utils.mxfp8_utils.common_dataclasses import (
    QuantScheme,
    TensorDescriptor,
    TensorOrientation,
)

from test.integration.nkilib.experimental.matmul_mxfp8 import utils as matmul_utils
from test.integration.nkilib.experimental.moe.test_bwmm_bwd_common import (
    generate_token_position_to_id_and_experts,
)
from test.integration.nkilib.experimental.moe_mxfp8.mxfp8_moe_bwd_test_utils import _get_n_blocks
from test.utils.mx_utils import quantize_mx_golden


def _prequantize_2d_fwd(tensor_2d, use_1x32):
    """Pre-quantize a single 2D [K, F] tensor to MXFP8 x4, WRAPX or 1x32 swizzle.

    Mirrors the backward's ``_prequantize_2d`` but selects the swizzle: WRAPX
    (``swizzle_tensor``, 4-partition interleave — the DGT / fast_dma_transpose
    layout) or 1x32 (``swizzle_tensor_1x32``, contiguous-K — the
    ``load_tile_PE_Swizzle_1x32`` layout). The quantize + scale-pack steps are
    identical; only the K permutation baked into the x4 data (and thus which
    activation load mode it must be paired with) differs. Returns (data, scales).
    """
    K, F = tensor_2d.shape
    swizzled = matmul_utils.swizzle_tensor_1x32(tensor_2d) if use_1x32 else matmul_utils.swizzle_tensor(tensor_2d)
    fp8_x4_dtype = nl.float8_e4m3fn_x4
    data, compact_scales = quantize_mx_golden(swizzled, fp8_x4_dtype)
    return data, matmul_utils.to_hbm_scales(compact_scales)


def prequantize_moe_weights_fwd(gate_up_proj_weight, down_proj_weight, use_1x32=False):
    """Pre-quantize MoE weights per-expert into the FORWARD-natural x4 layouts.

    The kernel takes gate_up as [E, I_TP, 2, H] ([E*I_TP, 2*H], gate/up a K-slice) and
    down as [E, H, I_TP]. Each weight is quantized as [K=contraction, F=output]:

    gate_up: standard [H, 2, I_TP] -> [K=2*H (gate|up), F=I_TP] -> x4 [2*H/4, I_TP]
    down:    standard [I_TP, H]    -> [K=I_TP, F=H]             -> x4 [I_TP/4, H]

    ``use_1x32`` selects the 1x32 (contiguous-K) swizzle so the x4 weight pairs with
    the ``use_1x32_pe_swizzle`` activation load; otherwise the WRAPX (DGT) swizzle,
    which pairs with the default / fast_dma_transpose activation load.

    Returns:
        (gate_up_data, gate_up_scales, down_data, down_scales)
        gate_up_data: [E, 2*H//4, I_TP]   gate_up_scales: [E, 2*H//32, I_TP]
        down_data:    [E, I_TP//4, H]      down_scales:    [E, I_TP//32, H]
    """
    E, H, _, I_TP = gate_up_proj_weight.shape

    gate_up_data_list = []
    gate_up_scales_list = []
    down_data_list = []
    down_scales_list = []

    for e in range(E):
        # gate_up: the kernel reads a K-slice of the [K=2*H, F=I_TP] weight (gate K in
        # [0,H), up in [H,2H)). Swizzle/quantize each H-half INDEPENDENTLY and stack, so
        # each half's swizzle (and its <512 K-remainder decomposition, e.g. H=768 = 512+256)
        # aligns with the kernel's per-half K-tile reads -- swizzling the full 2*H as one
        # would mix K across the 512-tile that straddles the gate/up boundary.
        gate_2d = gate_up_proj_weight[e][:, 0, :].astype(np.float32)  # [K=H, F=I_TP]
        up_2d = gate_up_proj_weight[e][:, 1, :].astype(np.float32)  # [K=H, F=I_TP]
        g_data, g_scales = _prequantize_2d_fwd(gate_2d, use_1x32)  # x4 [H/4, I_TP]
        u_data, u_scales = _prequantize_2d_fwd(up_2d, use_1x32)
        gu_data = np.concatenate([g_data, u_data], axis=0)  # [2*H/4, I_TP], [gate | up] on K
        gu_scales = np.concatenate([g_scales, u_scales], axis=0)
        gate_up_data_list.append(gu_data)
        gate_up_scales_list.append(gu_scales)

        # down: [I_TP, H] = [K=I_TP, F=H]  (no transpose)
        down_2d = down_proj_weight[e].astype(np.float32)
        d_data, d_scales = _prequantize_2d_fwd(down_2d, use_1x32)
        down_data_list.append(d_data)
        down_scales_list.append(d_scales)

    gate_up_data = np.stack(gate_up_data_list, axis=0)
    gate_up_scales = np.stack(gate_up_scales_list, axis=0)
    down_data = np.stack(down_data_list, axis=0)
    down_scales = np.stack(down_scales_list, axis=0)

    return gate_up_data, gate_up_scales, down_data, down_scales


def build_mxfp8_moe_fwd_inputs(
    tokens,
    hidden,
    intermediate,
    expert,
    block_size,
    top_k,
    dtype=nl.bfloat16,
    blocking_params=None,
    spill_reload=False,
    reuse_spilled_weights=False,
    use_scale_packing=False,
    fast_dma_transpose=False,
    use_1x32_pe_swizzle=False,
    bias=False,
    clamp_limits=None,
    prequantize_weights=False,
    save_gate_up_proj_act=True,
    single_expert_dense=False,
    weights_k_by_f=False,
    gate_up_weight_orientation=None,
    down_weight_orientation=None,
    num_valid_tokens=None,
    num_static_block=None,
):
    """Build the kernel-input dict for ``blockwise_mm_fwd_mxfp8``.

    Mirrors ``build_mxfp8_moe_bwd_inputs`` but for the forward: no
    ``output_hidden_states_grad`` and no checkpoint inputs (the forward produces
    those), and ``shard_option=SHARD_ON_BLOCK``. Random seeding uses the same
    parameter-hash scheme so a given shape is reproducible across runs.

    ``num_valid_tokens`` (single_expert_dense only) is how many leading token rows are
    real, enabling the kernel's hardware dynamic block loop. Counts above T exercise
    the kernel's capacity clamp; padding affinities after the effective count are
    zeroed so the golden's skipped region is exactly 0.
    ``num_static_block`` sets the static/dynamic split (None = auto).

    Returns:
        dict of kernel inputs. When ``prequantize_weights`` is True, returns
        ``(inputs, orig_gate_up_weight, orig_down_weight)`` so the test can feed
        the un-quantized weights to the golden (which is precision-agnostic).
    """
    T, H, I_TP, E, B = tokens, hidden, intermediate, expert, block_size
    legacy_orientation = TensorOrientation.K_BY_F if weights_k_by_f else TensorOrientation.F_BY_K
    if gate_up_weight_orientation is None:
        gate_up_weight_orientation = legacy_orientation
    if down_weight_orientation is None:
        down_weight_orientation = legacy_orientation
    if weights_k_by_f:
        assert gate_up_weight_orientation == TensorOrientation.K_BY_F
        assert down_weight_orientation == TensorOrientation.K_BY_F
    if single_expert_dense:
        assert E == 1, f"single_expert_dense test inputs require expert=1, got {E}"
        assert top_k == 1, f"single_expert_dense test inputs require top_k=1, got {top_k}"
        assert T % B == 0, f"single_expert_dense test inputs require tokens divisible by block_size, got T={T}, B={B}"
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

    param_string = f"{T}_{top_k}_{B}_{E}_{I_TP}_{H}_mxfp8_fwd_{int(single_expert_dense)}"
    seed = int(hashlib.sha256(param_string.encode()).hexdigest(), 16) % (2**32)
    np.random.seed(seed)

    # Standard (backward-natural) weights, used by the golden:
    #   down:    [E, I_TP, H]      gate_up: [E, H, 2, I_TP]
    down_proj_weights = np.random.uniform(-0.1, 0.1, size=[E, I_TP, H]).astype(dtype)
    gate_and_up_proj_weights = np.random.uniform(-0.1, 0.1, size=[E, H, 2, I_TP]).astype(dtype)

    expert_affinities_masked = (np.random.random_sample([T, E]) * token_experts).astype(dtype)
    hidden_states = np.random.random_sample([T, H]).astype(dtype)

    # Padding affinities are zeroed so the reference produces exactly 0 for the region
    # the kernel never writes.
    valid_tokens = None
    if num_valid_tokens is not None:
        assert single_expert_dense, "num_valid_tokens is only supported on the single_expert_dense path"
        assert num_valid_tokens > 0, f"num_valid_tokens must be positive, got {num_valid_tokens}"
        valid_tokens = np.array([num_valid_tokens], dtype=np.int32)
        expert_affinities_masked[min(num_valid_tokens, T) :] = 0

    # Forward-natural (transposed) weights, consumed by the kernel:
    #   gate_up: [E, H, 2, I_TP] -> [E, I_TP, 2, H]
    #   down:    [E, I_TP, H]    -> [E, H, I_TP]
    gate_up_fwd = np.ascontiguousarray(gate_and_up_proj_weights.transpose(0, 3, 2, 1))  # [E, I_TP, 2, H]
    down_fwd = np.ascontiguousarray(down_proj_weights.transpose(0, 2, 1))  # [E, H, I_TP]

    if prequantize_weights:
        assert gate_up_weight_orientation == TensorOrientation.F_BY_K
        assert down_weight_orientation == TensorOrientation.F_BY_K
        # prequantize_moe_weights_fwd expects the standard layouts and produces the
        # forward-natural x4 per-expert layouts the kernel's prequant path consumes
        # (gate_up [E, 2*H/4, I_TP], down [E, I_TP/4, H]).
        gate_up_data, gate_up_scales, down_data, down_scales = prequantize_moe_weights_fwd(
            gate_and_up_proj_weights, down_proj_weights, use_1x32=use_1x32_pe_swizzle
        )
    else:
        gate_up_data = (
            gate_and_up_proj_weights if gate_up_weight_orientation == TensorOrientation.K_BY_F else gate_up_fwd
        )
        gate_up_scales = None
        down_data = down_proj_weights if down_weight_orientation == TensorOrientation.K_BY_F else down_fwd
        down_scales = None

    # spill/reload and scale packing are per-phase settings, not config-level ones.
    gate_up_config = blocking_params.gate_up if blocking_params is not None else MatmulMxfp8KernelConfig(M=0, K=0, N=0)
    down_config = blocking_params.down if blocking_params is not None else MatmulMxfp8KernelConfig(M=0, K=0, N=0)
    for phase_config in (gate_up_config, down_config):
        phase_config.spill_reload = spill_reload
        phase_config.enable_scale_packing = use_scale_packing

    # Pre-quantized weights: the kernel reads the x4 swizzle scheme from the weight TD
    # (source of truth). Its default is 1x32 on the single_expert_dense (E=1) path and
    # WRAPX on the routed path; only set an explicit weight TD when the offline swizzle
    # differs from that default, so the matching cells exercise the default (None) path.
    prequant_weight_td = None
    if prequantize_weights:
        kernel_default_1x32 = single_expert_dense
        if use_1x32_pe_swizzle != kernel_default_1x32:
            scheme = QuantScheme._1x32 if use_1x32_pe_swizzle else QuantScheme.WRAPX
            prequant_weight_td = TensorDescriptor(quant_scheme=scheme)
    gate_up_bf16_weight_td = (
        TensorDescriptor(orientation=TensorOrientation.K_BY_F)
        if gate_up_weight_orientation == TensorOrientation.K_BY_F
        else None
    )
    down_bf16_weight_td = (
        TensorDescriptor(orientation=TensorOrientation.K_BY_F)
        if down_weight_orientation == TensorOrientation.K_BY_F
        else None
    )

    # Collapse every non-tensor knob into the fused forward config. The kernel entry
    # takes only the tensors + block_size + this config (+ optional prequant scales).
    config = MXFP8MOEFwdConfig(
        affinity_option=AffinityOption.AFFINITY_ON_I,
        shard_option=ShardOption.SHARD_ON_BLOCK,
        activation_type=ActFnType.SiLU,
        is_tensor_update_accumulating=False if single_expert_dense else top_k != 1,
        execution_mode=(ExecutionMode.DENSE_MOE if single_expert_dense else ExecutionMode.ROUTED_MOE),
        num_static_block=num_static_block,
        reuse_spilled_weights=reuse_spilled_weights,
        fast_dma_transpose=fast_dma_transpose,
        use_1x32_pe_swizzle=use_1x32_pe_swizzle,
        skip_dma=dma_skip,
        bias=bias,
        # Pass concrete objects (not None) so the type checker is satisfied; these
        # match what MXFP8MOEFwdConfig.__post_init__ would build for a None field.
        clamp_limits=clamp_limits if clamp_limits is not None else ClampLimits(),
        checkpoint_config=MXFP8MOECheckpointConfig(save_gate_up_proj_act=save_gate_up_proj_act),
        gate_up_weight_td=prequant_weight_td if prequantize_weights else gate_up_bf16_weight_td,
        down_weight_td=prequant_weight_td if prequantize_weights else down_bf16_weight_td,
        gate_up_config=gate_up_config,
        down_config=down_config,
    )

    inputs = {
        "hidden_states": hidden_states,
        "expert_affinities_masked": expert_affinities_masked.reshape(-1, 1),
        "gate_up_proj_weight": gate_up_data,
        "down_proj_weight": down_data,
        "token_position_to_id": token_position_to_id,
        "block_to_expert": block_to_expert.reshape(-1, 1),
        "block_size": block_size,
        "config": config,
    }
    if valid_tokens is not None:
        inputs["num_valid_tokens"] = valid_tokens

    if gate_up_scales is not None:
        inputs["gate_up_weight_scales"] = gate_up_scales
    if down_scales is not None:
        inputs["down_weight_scales"] = down_scales

    if prequantize_weights:
        return inputs, gate_and_up_proj_weights, down_proj_weights

    return inputs
