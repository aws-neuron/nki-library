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

"""PyTorch reference implementation of the blockwise MoE MXFP8 forward kernel.

Delegates to the shared BF16 MoE forward golden (``_generate_fwd_golden``) for the
output and the ``gate_up_proj_act_checkpoint_T`` checkpoint, then derives the
``scaled_intermediate_checkpoint_T`` (= SiLU(gate_clamped) * up_clamped * EA,
transposed to [N, I_TP, B]) that the forward kernel emits.

The parameter list mirrors ``blockwise_mm_fwd_mxfp8`` exactly so the test
framework's ``validate_torch_ref_signature`` passes; hardware/quantization-only
arguments are accepted and intentionally ignored (they do not change the math).
"""

import numpy as np
import torch

from test.integration.nkilib.experimental.moe.test_bwmm_bwd_common import _generate_fwd_golden
from test.integration.nkilib.utils.test_kernel_common import silu

from ....core.utils.kernel_assert import kernel_assert
from ...moe.bwd.moe_bwd_parameters import ActFnType, ClampLimits, SkipMode
from ..moe_mxfp8_checkpoint_config import CheckpointLayout, MXFP8MOECheckpointConfig


def blockwise_mm_fwd_mxfp8_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    block_size: int,
    gate_up_weight_scales=None,
    gate_up_weight_is_swizzled: bool = False,
    down_weight_scales=None,
    down_weight_is_swizzled: bool = False,
    gate_up_config=None,
    down_config=None,
    fp8_x4_dtype=None,
    spill_reload: bool = False,
    reuse_spilled_weights: bool = False,
    use_scale_packing: bool = True,
    fast_dma_transpose: bool = False,
    run_with_lnc2: bool = True,
    shard_option=None,
    affinity_option=None,
    compute_dtype=None,
    skip_dma: SkipMode = None,
    is_tensor_update_accumulating: bool = True,
    no_indirect_load: bool = False,
    clamp_limits: ClampLimits = None,
    activation_type=None,
    bias: bool = False,
    checkpoint_config: MXFP8MOECheckpointConfig = None,
) -> dict:
    """PyTorch reference for ``blockwise_mm_fwd_mxfp8``.

    Parameter list matches the kernel entry exactly (framework requirement). Only
    the tensor inputs + ``block_size``, ``skip_dma``, ``no_indirect_load``,
    ``clamp_limits``, ``bias`` and ``checkpoint_config`` affect the result; the
    remaining hardware/quantization arguments are accepted for signature
    compatibility and referenced below as no-ops so the framework's
    unused-parameter check passes.

    Returns:
        dict with keys matching the kernel's positional outputs (a checkpoint key
        is present only when its ``checkpoint_config`` save flag is set):
            - output_hidden_states: [T, H]
            - gate_up_proj_act_checkpoint_T: [N, 2, I_TP, B]  (when saved)
            - scaled_intermediate_checkpoint_T: [N, I_TP, B]  (when saved)
    """
    # Accept-and-ignore: hardware/quantization knobs do not change the reference math.
    _ignored = (
        gate_up_weight_scales,
        gate_up_weight_is_swizzled,
        down_weight_scales,
        down_weight_is_swizzled,
        gate_up_config,
        down_config,
        fp8_x4_dtype,
        spill_reload,
        reuse_spilled_weights,
        use_scale_packing,
        fast_dma_transpose,
        run_with_lnc2,
        shard_option,
        affinity_option,
        compute_dtype,
        is_tensor_update_accumulating,
        activation_type,
    )
    del _ignored

    if skip_dma is None:
        skip_dma = SkipMode(False, False)
    if clamp_limits is None:
        clamp_limits = ClampLimits()
    if checkpoint_config is None:
        checkpoint_config = MXFP8MOECheckpointConfig()

    hidden_np = hidden_states.numpy()
    expert_aff_np = expert_affinities_masked.numpy()
    gate_up_w_np = gate_up_proj_weight.numpy()
    down_w_np = down_proj_weight.numpy()
    tok_pos_np = token_position_to_id.numpy()
    blk_exp_np = block_to_expert.reshape(-1).numpy()

    T = hidden_np.shape[0]
    H = hidden_np.shape[1]
    E = down_w_np.shape[0]

    # The kernel consumes forward-natural weights (gate_up [E, 2, I_TP, H],
    # down [E, H, I_TP]); the golden needs the standard backward-natural layout
    # (gate_up [E, H, 2, I_TP], down [E, I_TP, H]). Transpose back here. (When the
    # prequantized path supplies the original fp32 standard-layout weights via the
    # test's _pq_torch_ref wrapper, they are already standard — detect by rank.)
    if gate_up_w_np.ndim == 4 and gate_up_w_np.shape == (E, 2, gate_up_w_np.shape[2], H):
        I_TP = gate_up_w_np.shape[2]
        gate_up_w_np = np.ascontiguousarray(gate_up_w_np.transpose(0, 3, 1, 2))  # -> [E, H, 2, I_TP]
    else:
        I_TP = gate_up_w_np.shape[3]  # already standard [E, H, 2, I_TP]
    if down_w_np.shape == (E, H, I_TP):
        down_w_np = np.ascontiguousarray(down_w_np.transpose(0, 2, 1))  # -> [E, I_TP, H]
    B = block_size
    N = T // B if no_indirect_load else tok_pos_np.shape[0] // B
    expert_aff_2d = expert_aff_np.reshape(-1, E)
    dtype = hidden_np.dtype

    if no_indirect_load:
        kernel_assert(E == 1, f"no_indirect_load requires one expert, got E={E}")
        kernel_assert(T % B == 0, "no_indirect_load requires T to be divisible by block_size")
        kernel_assert(
            tok_pos_np.shape == (1,),
            f"no_indirect_load expects dummy token_position_to_id [1], got {tok_pos_np.shape}",
        )
        kernel_assert(
            blk_exp_np.shape == (1,),
            f"no_indirect_load expects dummy block_to_expert [1, 1], got {blk_exp_np.shape}",
        )

        hidden_f32 = hidden_np.astype(np.float32)
        gate = hidden_f32 @ gate_up_w_np[0, :, 0, :].astype(np.float32)
        up = hidden_f32 @ gate_up_w_np[0, :, 1, :].astype(np.float32)

        if clamp_limits.non_linear_clamp_upper_limit is not None:
            gate = np.minimum(gate, clamp_limits.non_linear_clamp_upper_limit)
        if clamp_limits.non_linear_clamp_lower_limit is not None:
            gate = np.maximum(gate, clamp_limits.non_linear_clamp_lower_limit)
        if clamp_limits.linear_clamp_upper_limit is not None:
            up = np.minimum(up, clamp_limits.linear_clamp_upper_limit)
        if clamp_limits.linear_clamp_lower_limit is not None:
            up = np.maximum(up, clamp_limits.linear_clamp_lower_limit)

        gate_c = gate.astype(dtype)
        up_c = up.astype(dtype)
        inter = silu(gate_c.astype(np.float32)) * up_c.astype(np.float32)
        scaled = (inter * expert_aff_2d[:, 0:1].astype(np.float32)).astype(dtype)
        output_np = (scaled.astype(np.float32) @ down_w_np[0].astype(np.float32)).astype(dtype)

        result = {
            "output_hidden_states": torch.from_numpy(np.ascontiguousarray(output_np)),
        }
        # Each checkpoint is emitted only when its save flag is set (matches the
        # kernel's checkpoint_config gating and the routed path below). Per-block
        # layout follows the config: TRANSPOSED stores [I_TP, B] (X[block].T),
        # DIRECT stores the natural [B, I_TP] (X[block]).
        gate_up_direct = checkpoint_config.gate_up_proj_act_layout == CheckpointLayout.DIRECT
        scaled_direct = checkpoint_config.scaled_intermediate_layout == CheckpointLayout.DIRECT
        if checkpoint_config.save_gate_up_proj_act:
            gate_up_shape = (N, 2, B, I_TP) if gate_up_direct else (N, 2, I_TP, B)
            gate_up_activations_T = np.zeros(gate_up_shape, dtype=dtype)
            for block_idx in range(N):
                start = block_idx * B
                end = start + B
                if gate_up_direct:
                    gate_up_activations_T[block_idx, 0] = gate_c[start:end, :]
                    gate_up_activations_T[block_idx, 1] = up_c[start:end, :]
                else:
                    gate_up_activations_T[block_idx, 0] = gate_c[start:end, :].T
                    gate_up_activations_T[block_idx, 1] = up_c[start:end, :].T
            result["gate_up_proj_act_checkpoint_T"] = torch.from_numpy(np.ascontiguousarray(gate_up_activations_T))
        if checkpoint_config.save_scaled_intermediate:
            scaled_shape = (N, B, I_TP) if scaled_direct else (N, I_TP, B)
            scaled_intermediate_checkpoint_T = np.zeros(scaled_shape, dtype=dtype)
            for block_idx in range(N):
                start = block_idx * B
                end = start + B
                scaled_intermediate_checkpoint_T[block_idx] = (
                    scaled[start:end, :] if scaled_direct else scaled[start:end, :].T
                )
            result["scaled_intermediate_checkpoint_T"] = torch.from_numpy(
                np.ascontiguousarray(scaled_intermediate_checkpoint_T)
            )
        return result

    gate_up_bias = None
    down_bias = None
    if bias:
        gate_up_bias = np.zeros((E, 2, I_TP), dtype=gate_up_w_np.dtype)
        down_bias = np.zeros((E, H), dtype=down_w_np.dtype)

    output_np, gate_up_activations_T, _down_activations = _generate_fwd_golden(
        expert_affinities=expert_aff_2d,
        down_proj_weights=down_w_np,
        token_position_to_id=tok_pos_np,
        block_to_expert=blk_exp_np,
        gate_and_up_proj_weights=gate_up_w_np,
        hidden_states=hidden_np,
        T=T,
        H=H,
        B=B,
        N=N,
        E=E,
        I_TP=I_TP,
        dtype=dtype,
        dma_skip=skip_dma,
        activation_function=ActFnType.SiLU,
        gate_up_proj_bias=gate_up_bias,
        down_proj_bias=down_bias,
        clamp_limits=clamp_limits,
    )

    result = {
        "output_hidden_states": torch.from_numpy(np.ascontiguousarray(output_np)),
    }
    # gate_up_activations_T (transposed [N, 2, I_TP, B]) is always computed by the
    # golden (it is needed to derive the scaled intermediate below) but only
    # returned as a checkpoint when the save flag is set. When the config selects
    # DIRECT, transpose it back to token-major [N, 2, B, I_TP] for the return.
    gate_up_direct = checkpoint_config.gate_up_proj_act_layout == CheckpointLayout.DIRECT
    scaled_direct = checkpoint_config.scaled_intermediate_layout == CheckpointLayout.DIRECT
    if checkpoint_config.save_gate_up_proj_act:
        gate_up_out = gate_up_activations_T.transpose(0, 1, 3, 2) if gate_up_direct else gate_up_activations_T
        result["gate_up_proj_act_checkpoint_T"] = torch.from_numpy(np.ascontiguousarray(gate_up_out))

    # Derive the scaled intermediate from the (already clamped) gate/up checkpoint +
    # affinity: SiLU(gate) * up * EA. gate_up_activations_T is [N, 2, I_TP, B] with
    # gate at [:,0], up at [:,1]. Store transposed [N, I_TP, B] or, for DIRECT,
    # token-major [N, B, I_TP].
    if checkpoint_config.save_scaled_intermediate:
        tok_pos_2d = tok_pos_np.reshape(N, B)
        scaled_shape = [N, B, I_TP] if scaled_direct else [N, I_TP, B]
        scaled_intermediate_checkpoint_T = np.zeros(scaled_shape).astype(dtype)
        for b in range(N):
            gate_t = gate_up_activations_T[b, 0].astype(np.float32)  # [I_TP, B]
            up_t = gate_up_activations_T[b, 1].astype(np.float32)  # [I_TP, B]
            inter_t = silu(gate_t) * up_t  # [I_TP, B]
            expert_idx = blk_exp_np[b]
            local_ids = tok_pos_2d[b, :]
            ea = expert_aff_2d[local_ids, expert_idx].reshape(1, B).astype(np.float32)  # [1, B]
            scaled_t = (inter_t * ea).astype(dtype)  # [I_TP, B]
            scaled_intermediate_checkpoint_T[b] = scaled_t.T if scaled_direct else scaled_t
        result["scaled_intermediate_checkpoint_T"] = torch.from_numpy(
            np.ascontiguousarray(scaled_intermediate_checkpoint_T)
        )

    return result
