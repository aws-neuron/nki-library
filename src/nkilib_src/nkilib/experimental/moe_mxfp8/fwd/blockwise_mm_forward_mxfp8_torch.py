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
output and the ``gate_up_proj_act_checkpoint_T`` checkpoint (the only checkpoint the
forward emits; its store layout follows config.gate_up_proj_act_td.orientation).

The parameter list mirrors ``blockwise_mm_fwd_mxfp8`` exactly so the test
framework's ``validate_torch_ref_signature`` passes; hardware/quantization-only
arguments are accepted and intentionally ignored (they do not change the math).
"""

import numpy as np
import torch

from test.integration.nkilib.experimental.moe.test_bwmm_bwd_common import _generate_fwd_golden
from test.integration.nkilib.utils.test_kernel_common import silu

from ....core.utils.kernel_assert import kernel_assert
from ...moe.bwd.moe_bwd_parameters import ActFnType
from ...mxfp_utils.mxfp8_utils.common_dataclasses import TensorOrientation
from ...mxfp_utils.mxfp8_utils.dyn_blocks import resolve_blocks, tail_sizes_for
from ..moe_mxfp8_checkpoint_config import CheckpointLayout
from .config import MXFP8MOEFwdConfig


def _dynamic_checkpoint_extent(T, B, num_valid_tokens, num_static_block):
    """Return produced checkpoint blocks and any shortened final-block row count."""
    N = T // B
    schedule = resolve_blocks(
        n_blocks=N,
        block_size=B,
        num_static_block=num_static_block,
        has_dynamic=True,
    )
    valid_tokens = min(max(int(num_valid_tokens.reshape(-1)[0].item()), 0), T)
    dynamic_tokens = max(0, valid_tokens - schedule.n_static * B)
    active_dynamic = min((dynamic_tokens + B - 1) // B, schedule.n_dynamic)
    produced_blocks = schedule.n_static + active_dynamic

    shortened_rows = None
    if active_dynamic % 2 == 1:
        rows_needed = dynamic_tokens - (active_dynamic - 1) * B
        selected_rows = next(size for size in tail_sizes_for(B) if rows_needed <= size)
        if selected_rows < B:
            shortened_rows = selected_rows
    return produced_blocks, shortened_rows


def blockwise_mm_fwd_mxfp8_torch_ref(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    block_size: int = 4096,
    config: MXFP8MOEFwdConfig = None,
    gate_up_weight_scales=None,
    down_weight_scales=None,
    num_valid_tokens: torch.Tensor = None,
) -> dict:
    """PyTorch reference for ``blockwise_mm_fwd_mxfp8``.

    Parameter list matches the kernel entry exactly (framework requirement). Only
    the tensor inputs + ``block_size`` and the ``config`` fields ``skip_dma``,
    ``single_expert_dense``, ``clamp_limits``, ``bias`` and ``checkpoint_config``
    affect the result; the remaining hardware/quantization knobs on ``config`` (and
    the pre-quantized weight scales) do not change the reference math.

    Rows past ``num_valid_tokens`` are undefined on device and are not compared.

    Returns:
        dict with keys matching the kernel's positional outputs (a checkpoint key
        is present only when its ``checkpoint_config`` save flag is set):
            - output_hidden_states: [T, H]
            - gate_up_proj_act_checkpoint_T: [N, 2, I_TP, B]  (when saved)
    """
    if config is None:
        config = MXFP8MOEFwdConfig()
    # Pre-quantized scales do not change the reference math. The dynamic loop
    # changes which blocks run, not the math.
    _ignored = (gate_up_weight_scales, down_weight_scales)
    del _ignored

    # The only config fields that change the reference math.
    skip_dma = config.skip_dma
    clamp_limits = config.clamp_limits
    checkpoint_config = config.checkpoint_config
    # Store layout for the gate/up checkpoint comes from the activation TD orientation.
    gate_up_direct = config.gate_up_proj_act_layout == CheckpointLayout.DIRECT
    single_expert_dense = config.single_expert_dense
    bias = config.bias
    kernel_assert(
        num_valid_tokens is None or single_expert_dense,
        "num_valid_tokens (hardware dynamic block loop) requires single_expert_dense=True (E=1)",
    )

    hidden_np = hidden_states.numpy()
    expert_aff_np = expert_affinities_masked.numpy()
    gate_up_w_np = gate_up_proj_weight.numpy()
    down_w_np = down_proj_weight.numpy()
    tok_pos_np = token_position_to_id.numpy()
    blk_exp_np = block_to_expert.reshape(-1).numpy()

    T = hidden_np.shape[0]
    H = hidden_np.shape[1]
    E = down_w_np.shape[0]

    # The golden consumes backward-natural weights. Orientation is the source of
    # truth for BF16 inputs, including square H == I_TP shapes where shape cannot
    # distinguish the two layouts. The pre-quantized test wrapper substitutes its
    # original fp32 backward-natural weights while retaining the non-None scales.
    gate_up_orientation = (
        config.gate_up_weight_td.orientation if config.gate_up_weight_td is not None else TensorOrientation.F_BY_K
    )
    down_orientation = (
        config.down_weight_td.orientation if config.down_weight_td is not None else TensorOrientation.F_BY_K
    )
    gate_up_already_backward_natural = (
        gate_up_weight_scales is not None or gate_up_orientation == TensorOrientation.K_BY_F
    )
    down_already_backward_natural = down_weight_scales is not None or down_orientation == TensorOrientation.K_BY_F
    if not gate_up_already_backward_natural:
        gate_up_w_np = np.ascontiguousarray(gate_up_w_np.transpose(0, 3, 2, 1))  # -> [E, H, 2, I_TP]
    I_TP = gate_up_w_np.shape[3]
    if not down_already_backward_natural:
        down_w_np = np.ascontiguousarray(down_w_np.transpose(0, 2, 1))  # -> [E, I_TP, H]
    B = block_size
    N = T // B if single_expert_dense else tok_pos_np.shape[0] // B
    expert_aff_2d = expert_aff_np.reshape(-1, E)
    dtype = hidden_np.dtype

    if single_expert_dense:
        kernel_assert(E == 1, f"single_expert_dense requires one expert, got E={E}")
        kernel_assert(T % B == 0, "single_expert_dense requires T to be divisible by block_size")
        kernel_assert(
            tok_pos_np.shape == (1,),
            f"single_expert_dense expects dummy token_position_to_id [1], got {tok_pos_np.shape}",
        )
        kernel_assert(
            blk_exp_np.shape == (1,),
            f"single_expert_dense expects dummy block_to_expert [1, 1], got {blk_exp_np.shape}",
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

        # Blocks the kernel skips are never written; zero them so the golden encodes the
        # same "not produced" region the comparator masks off.

        result = {
            "output_hidden_states": torch.from_numpy(np.ascontiguousarray(output_np)),
        }
        # Each checkpoint is emitted only when its save flag is set (matches the
        # kernel's checkpoint_config gating and the routed path below). Per-block
        # layout follows the config: TRANSPOSED stores [I_TP, B] (X[block].T),
        # DIRECT stores the natural [B, I_TP] (X[block]).
        if checkpoint_config.save_gate_up_proj_act:
            gate_up_shape = (N, 2, B, I_TP) if gate_up_direct else (N, 2, I_TP, B)
            gate_up_activations_T = np.zeros(gate_up_shape, dtype=dtype)
            produced_blocks = N
            shortened_rows = None
            if num_valid_tokens is not None:
                produced_blocks, shortened_rows = _dynamic_checkpoint_extent(
                    T,
                    B,
                    num_valid_tokens,
                    config.num_static_block,
                )
            for block_idx in range(produced_blocks):
                start = block_idx * B
                end = start + B
                if gate_up_direct:
                    gate_up_activations_T[block_idx, 0] = gate_c[start:end, :]
                    gate_up_activations_T[block_idx, 1] = up_c[start:end, :]
                else:
                    gate_up_activations_T[block_idx, 0] = gate_c[start:end, :].T
                    gate_up_activations_T[block_idx, 1] = up_c[start:end, :].T
            if shortened_rows is not None:
                last_block = produced_blocks - 1
                if gate_up_direct:
                    gate_up_activations_T[last_block, :, shortened_rows:, :] = 0
                else:
                    gate_up_activations_T[last_block, :, :, shortened_rows:] = 0
            result["gate_up_proj_act_checkpoint_T"] = torch.from_numpy(np.ascontiguousarray(gate_up_activations_T))
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

    # The kernel always returns [T, H]; the shared golden keeps a T+1 sentinel row
    # only when skip_token is False (it returns [T, H] pre-sliced when True), so slice
    # to [:T] here to match the kernel for both skip modes.
    result = {
        "output_hidden_states": torch.from_numpy(np.ascontiguousarray(output_np[:T])),
    }
    # gate_up_activations_T (transposed [N, 2, I_TP, B]) is returned as a checkpoint
    # when the save flag is set. When the config selects DIRECT, transpose it back to
    # token-major [N, 2, B, I_TP] for the return.
    if checkpoint_config.save_gate_up_proj_act:
        gate_up_out = gate_up_activations_T.transpose(0, 1, 3, 2) if gate_up_direct else gate_up_activations_T
        result["gate_up_proj_act_checkpoint_T"] = torch.from_numpy(np.ascontiguousarray(gate_up_out))

    return result
