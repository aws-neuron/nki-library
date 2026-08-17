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

"""PyTorch reference implementation for MoE CTE blockwise matrix multiplication kernels."""

from typing import Optional

import torch

from ...utils.common_types import ActFnType, ExpertAffinityScaleMode
from .moe_cte_torch_utils import torch_act_fn
from .moe_cte_utils import SkipMode


def _moe_cte_torch_ref_impl(
    hidden_states: torch.Tensor,
    expert_affinities_masked: torch.Tensor,
    gate_up_proj_weight: torch.Tensor,
    down_proj_weight: torch.Tensor,
    token_position_to_id: torch.Tensor,
    block_to_expert: torch.Tensor,
    block_size: int,
    bwmm_func,
    lnc_degree: int = 2,
    conditions: Optional[torch.Tensor] = None,
    gate_and_up_proj_bias: Optional[torch.Tensor] = None,
    down_proj_bias: Optional[torch.Tensor] = None,
    gate_up_proj_scale: Optional[torch.Tensor] = None,
    down_proj_scale: Optional[torch.Tensor] = None,
    gate_up_hidden_scale: Optional[torch.Tensor] = None,
    down_hidden_scale: Optional[torch.Tensor] = None,
    is_block_quant: bool = False,
    is_per_tensor: bool = False,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype=None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    checkpoint_activation: bool = False,
    expert_affinity_multiply_on_I: bool = False,
    n_block_per_iter: int = 1,
    block_sharding_strategy=None,
    num_static_block: Optional[int] = None,
    gate_up_activations_T: Optional[torch.Tensor] = None,
    down_activations: Optional[torch.Tensor] = None,
    top_k: int = 1,
    down_bias_tp_degree: Optional[int] = None,
    down_bias_tp_rank: Optional[int] = None,
    accumulation_dtype=None,
    skip_gate_proj: bool = False,
) -> dict:
    """
    PyTorch reference implementation of blockwise MoE matrix multiplication.

    This is a line-by-line port of the numpy golden (generate_blockwise_numpy_golden)
    to PyTorch, used for testing the NKI MoE CTE kernels.

    Args:
        hidden_states: [T+1, H], Input token embeddings (T+1 includes padding token)
        expert_affinities_masked: [(T+1)*E, 1], Expert routing weights
        gate_up_proj_weight: [E, H, 2, I_TP], Gate and up projection weights
        down_proj_weight: [E, I_TP_padded, H], Down projection weights (may be padded on I dimension)
        token_position_to_id: [N*B], Token to block position mapping
        block_to_expert: [N], Expert assignment per block
        block_size: Tokens per block
        bwmm_func: BWMMFunc enum indicating kernel variant
        lnc_degree: LNC degree
        top_k: Number of top experts per token
        down_bias_tp_degree: TP degree for down projection bias sharding
        down_bias_tp_rank: TP rank for down projection bias sharding

    Returns:
        dict with 'output' and optionally 'gate_up_activations_T', 'down_activations'
    """
    from .bwmm_func import BWMMFunc

    # is_per_tensor is part of the kernel signature; the torch ref dispatches per-tensor
    # vs per-token activation rescale via the gate_up_hidden_scale shape, so this flag is
    # informational here. Kept in the signature to mirror the kernel exactly.
    _ = is_per_tensor

    E = gate_up_proj_weight.shape[0]
    H = hidden_states.shape[-1]
    I_TP = gate_up_proj_weight.shape[-1]
    B = block_size

    is_block_parallel = bwmm_func in (BWMMFunc.SHARD_ON_BLOCK, BWMMFunc.SHARD_ON_BLOCK_V2, BWMMFunc.SHARD_ON_BLOCK_HW)
    is_dropping = bwmm_func == BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING
    is_shard_block = bwmm_func in (BWMMFunc.SHARD_ON_BLOCK, BWMMFunc.SHARD_ON_BLOCK_V2, BWMMFunc.SHARD_ON_BLOCK_HW)
    do_checkpoint = checkpoint_activation or is_dropping

    has_quantize = gate_up_proj_scale is not None and gate_up_proj_scale.numel() > 0
    quantize_strategy = 6 if has_quantize else 0

    block_to_expert_flat = block_to_expert.flatten().long()
    N = block_to_expert_flat.shape[0]
    separate_outputs = is_block_parallel and top_k > 1

    # Match numpy golden: T is derived from hidden_states shape
    # When skip_token=False: hidden_states is [T+1, H], so T = shape[0] - 1
    # When skip_token=True: hidden_states is [T, H], so T = shape[0]
    T = hidden_states.shape[0] - 1 if not skip_dma.skip_token else hidden_states.shape[0]

    is_v2_shard_block = bwmm_func in (BWMMFunc.SHARD_ON_BLOCK_V2, BWMMFunc.SHARD_ON_BLOCK_HW)

    # Output shape matches numpy golden: always [T+1, ...]
    if is_shard_block:
        if is_v2_shard_block:
            output_shape = [T + 1, H] if separate_outputs else [T + 1, H]
        else:
            output_shape = [T + 1, lnc_degree, H] if separate_outputs else [T + 1, H]
    else:
        output_shape = [lnc_degree, T + 1, H] if separate_outputs else [T + 1, H]

    # Use bfloat16 for output accumulation to match numpy golden behavior
    output = torch.zeros(output_shape, dtype=torch.float32)
    token_pos_2d = token_position_to_id.long().reshape(N, B)

    if do_checkpoint:
        ckpt_gate_up = torch.zeros(N, 2, I_TP, B, dtype=torch.float32)
        ckpt_down = torch.zeros(N, B, H, dtype=torch.float32)

    # Reshape weights: [E, H, 2*I_TP]
    gate_up_w = gate_up_proj_weight.float().reshape(E, H, 2 * I_TP)
    down_w = down_proj_weight.float()

    # Handle quantize_strategy == 5 scale transpose (matching numpy golden)
    down_scale_work = None
    if has_quantize and quantize_strategy == 5 and down_proj_scale is not None:
        ds = down_proj_scale.float()
        if ds.shape[0] == E:
            down_scale_work = ds.reshape(E, 128, H // 128).permute(0, 2, 1).reshape(E, 1, H)
        else:
            down_scale_work = ds
    elif has_quantize and down_proj_scale is not None:
        down_scale_work = down_proj_scale.float()

    # Working copies that grow for skip_token (matching numpy golden)
    hidden_work = hidden_states.float()
    affinities_2d = expert_affinities_masked.float().reshape(-1, E)

    gup_scale = gate_up_proj_scale.float() if has_quantize and gate_up_proj_scale is not None else None

    for b_idx in range(N):
        # Padded blocks are skipped via expert_idx >= E check below

        local_ids = token_pos_2d[b_idx]
        expert_idx = block_to_expert_flat[b_idx].item()

        # For skip_token, append zero row each iteration (matching numpy golden exactly)
        if skip_dma.skip_token:
            hidden_work = torch.cat([hidden_work, torch.zeros(1, H)], dim=0)
            affinities_2d = torch.cat([affinities_2d, torch.zeros(1, E)], dim=0)

        local_hidden = hidden_work[local_ids].float()
        local_affinities = affinities_2d[local_ids, expert_idx].unsqueeze(1).to(hidden_states.dtype)

        if (
            expert_affinities_scaling_mode
            in [
                ExpertAffinityScaleMode.PRE_SCALE,
                ExpertAffinityScaleMode.PRE_SCALE_DELAYED,
            ]
            and not expert_affinity_multiply_on_I
        ):
            local_hidden = local_affinities * local_hidden

        if expert_idx >= E:
            continue

        # Gate-up projection: [B, H] @ [H, 2*I_TP] -> [B, 2, I_TP]
        gate_up_act = torch.matmul(local_hidden, gate_up_w[expert_idx]).reshape(B, 2, I_TP)
        gate_act = gate_up_act[:, 0, :].clone()
        up_act = gate_up_act[:, 1, :].clone()

        # Apply quantization scales
        if has_quantize and gup_scale is not None:
            if is_block_quant:
                # Block quant: scale shape [E, H//256, 2, I_TP//256, TILE_SIZE] — take first element
                bq_scale = gup_scale[expert_idx, :, :, :, 0].float()  # [H//256, 2, I_TP//256]
                BQ = 256
                H_blocks = H // BQ
                I_blocks = I_TP // BQ
                gate_act_bq = torch.zeros_like(gate_act)
                up_act_bq = torch.zeros_like(up_act)
                for h_b in range(H_blocks):
                    h_s, h_e = h_b * BQ, (h_b + 1) * BQ
                    for i_b in range(I_blocks):
                        i_s, i_e = i_b * BQ, (i_b + 1) * BQ
                        gate_act_bq[:, i_s:i_e] += (
                            local_hidden[:, h_s:h_e]
                            @ gate_up_w[expert_idx, h_s:h_e, :I_TP][:, i_s:i_e]
                            * bq_scale[h_b, 0, i_b]
                        )
                        up_act_bq[:, i_s:i_e] += (
                            local_hidden[:, h_s:h_e]
                            @ gate_up_w[expert_idx, h_s:h_e, I_TP:][:, i_s:i_e]
                            * bq_scale[h_b, 1, i_b]
                        )
                gate_act = gate_act_bq
                up_act = up_act_bq
            elif gup_scale.shape[0] == 1:
                gate_act *= gup_scale.squeeze()[:I_TP]
                up_act *= gup_scale.squeeze()[I_TP:]
            elif gup_scale.dim() == 3 and gup_scale.shape[1] == 2 and gup_scale.shape[2] == 1:
                # Per-tensor: [E, 2, 1]
                gate_act *= gup_scale[expert_idx, 0, 0].float()
                up_act *= gup_scale[expert_idx, 1, 0].float()
            elif gup_scale.shape[0] == E:
                gate_act *= gup_scale[expert_idx, 0, :I_TP]
                up_act *= gup_scale[expert_idx, 0, I_TP:]

        # Apply per-token hidden scale for static quantization (post-dequant)
        if has_quantize and gate_up_hidden_scale is not None:
            if (
                gate_up_hidden_scale.dim() == 3
                and gate_up_hidden_scale.shape[1] == 2
                and gate_up_hidden_scale.shape[2] == 1
            ):
                # Per-tensor activation: [E, 2, 1]
                gate_act *= gate_up_hidden_scale[expert_idx, 0, 0].float()
                up_act *= gate_up_hidden_scale[expert_idx, 1, 0].float()
            else:
                # Per-token activation: [T+1, 1]
                local_gup_hs = gate_up_hidden_scale[local_ids].float()  # [B, 1]
                gate_act *= local_gup_hs
                up_act *= local_gup_hs

        # Apply bias
        if gate_and_up_proj_bias is not None:
            gate_act += gate_and_up_proj_bias[expert_idx, 0, :]
            up_act += gate_and_up_proj_bias[expert_idx, 1, :]

        # Apply clamping
        if gate_clamp_lower_limit is not None or gate_clamp_upper_limit is not None:
            gate_act = torch.clamp(gate_act, min=gate_clamp_lower_limit, max=gate_clamp_upper_limit)
        if up_clamp_lower_limit is not None or up_clamp_upper_limit is not None:
            up_act = torch.clamp(up_act, min=up_clamp_lower_limit, max=up_clamp_upper_limit)

        if do_checkpoint:
            ckpt_gate_up[b_idx] = gate_up_act.permute(1, 2, 0)

        # Activation + element-wise multiply
        if skip_gate_proj:
            intermediate = torch_act_fn(up_act, activation_function)
        else:
            intermediate = torch_act_fn(gate_act, activation_function) * up_act

        # Apply affinity on intermediate if multiply_on_I
        if expert_affinity_multiply_on_I:
            intermediate = intermediate * local_affinities

        # Down projection: [B, I_TP] @ [I_TP_padded, H] -> [B, H]
        down_act = torch.matmul(intermediate, down_w[expert_idx])

        # Apply down quantization scale
        if has_quantize and down_scale_work is not None:
            if is_block_quant:
                # Block quant: scale shape [E, I_TP//256, H//256, TILE_SIZE] — take first element
                ds = down_scale_work[expert_idx, :, :, 0].float()  # [I_TP//256, H//256]
                BQ = 256
                I_blocks = intermediate.shape[1] // BQ
                H_blocks = H // BQ
                down_act_bq = torch.zeros(B, H, dtype=torch.float32)
                for i_b in range(I_blocks):
                    i_s, i_e = i_b * BQ, (i_b + 1) * BQ
                    for h_b in range(H_blocks):
                        h_s, h_e = h_b * BQ, (h_b + 1) * BQ
                        down_act_bq[:, h_s:h_e] += (
                            intermediate[:, i_s:i_e] @ down_w[expert_idx, i_s:i_e, h_s:h_e] * ds[i_b, h_b]
                        )
                down_act = down_act_bq
            elif down_scale_work.shape[0] == 1:
                down_act = down_act * down_scale_work.squeeze()
            elif down_scale_work.dim() == 2 and down_scale_work.shape[1] == 1 and down_scale_work.shape[0] == E:
                # Per-tensor: [E, 1]
                down_act = down_act * down_scale_work[expert_idx, 0].float()
            elif down_scale_work.shape[0] == E:
                down_act = down_act * down_scale_work[expert_idx, 0, :]

        # Apply per-token intermediate scale for static quantization (post-dequant)
        if has_quantize and down_hidden_scale is not None:
            if down_hidden_scale.dim() == 2 and down_hidden_scale.shape[1] == 1 and down_hidden_scale.shape[0] == E:
                # Per-tensor activation: [E, 1]
                down_act = down_act * down_hidden_scale[expert_idx, 0].float()
            else:
                # Per-token activation: [T+1, 1]
                local_down_hs = down_hidden_scale[local_ids].float()
                down_act = down_act * local_down_hs

        # Apply down bias
        if down_proj_bias is not None:
            if down_bias_tp_degree is not None:
                # TP-sharded bias: apply only to the H slice for this TP rank
                _h_per_tp = H // down_bias_tp_degree
                _h_start = down_bias_tp_rank * _h_per_tp
                _h_end = _h_start + _h_per_tp
                if expert_affinity_multiply_on_I:
                    down_act[:, _h_start:_h_end] += down_proj_bias[expert_idx] * local_affinities
                else:
                    down_act[:, _h_start:_h_end] += down_proj_bias[expert_idx]
            else:
                if expert_affinity_multiply_on_I:
                    down_act += down_proj_bias[expert_idx] * local_affinities
                else:
                    down_act += down_proj_bias[expert_idx]

        if do_checkpoint:
            ckpt_down[b_idx] = down_act

        # Apply expert affinity scaling
        if expert_affinities_scaling_mode == ExpertAffinityScaleMode.POST_SCALE and not expert_affinity_multiply_on_I:
            scaled = down_act * local_affinities
        else:
            scaled = down_act

        # Accumulate output (cast to bfloat16 then back, matching numpy golden .astype(dtype))
        scaled_bf16 = scaled.to(torch.bfloat16).to(torch.float32)
        if separate_outputs:
            if is_v2_shard_block:
                output[local_ids, :] += scaled_bf16
            elif is_shard_block:
                output[local_ids, 0, :] += scaled_bf16
            else:
                output[0, local_ids, :] += scaled_bf16
        else:
            output[local_ids, :] += scaled_bf16

    # Slice for skip_token (matching numpy golden)
    if skip_dma.skip_token:
        if separate_outputs:
            if is_v2_shard_block:
                output = output[:T, :]
            elif is_shard_block:
                output = output[:T, :, :]
            else:
                output = output[:, :T, :]
        else:
            output = output[:T, :]

    result = {"output": output}
    if do_checkpoint:
        result["gate_up_activations_T"] = ckpt_gate_up
        if not expert_affinity_multiply_on_I:
            result["down_activations"] = ckpt_down
    return result


def moe_cte_torch_ref(
    hidden_states,
    expert_affinities_masked,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    block_size: int,
    spec,
    conditions=None,
    gate_and_up_proj_bias=None,
    down_proj_bias=None,
    quantization_config=None,
    gate_up_proj_scale=None,
    down_proj_scale=None,
    gate_up_activations_T=None,
    down_activations=None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: SkipMode = SkipMode(False, False),
    compute_dtype=None,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    gate_up_in_scale=None,
    down_in_scale=None,
    accumulation_dtype=None,
    skip_gate_proj: bool = False,
) -> dict:
    """
    PyTorch reference for the unified moe_cte() entry point.

    Signature matches moe_cte() exactly. Extracts implementation-specific params
    from spec and quantization_config, then delegates to _moe_cte_torch_ref_impl.

    Args:
        spec: MoECTESpec with implementation type and config
        quantization_config: QuantizationConfig with optional scales
        (all other args match moe_cte() signature)

    Returns:
        dict with 'output' and optionally 'gate_up_activations_T', 'down_activations'
    """
    from .bwmm_func import BWMMFunc
    from .moe_cte import MoECTEImplementation

    # Map MoECTEImplementation -> BWMMFunc
    impl_to_bwmm = {
        MoECTEImplementation.shard_on_block: BWMMFunc.SHARD_ON_BLOCK,
        MoECTEImplementation.shard_on_i: BWMMFunc.SHARD_ON_INTERMEDIATE,
        MoECTEImplementation.shard_on_i_hybrid: BWMMFunc.SHARD_ON_INTERMEDIATE_HW,
        MoECTEImplementation.shard_on_i_dropping: BWMMFunc.SHARD_ON_INTERMEDIATE_DROPPING,
    }

    # Extract scales: direct params take precedence over quantization_config
    if gate_up_proj_scale is None and quantization_config is not None:
        gate_up_proj_scale = quantization_config.gate_up_proj_scale
    if down_proj_scale is None and quantization_config is not None:
        down_proj_scale = quantization_config.down_proj_scale
    # Convert numpy arrays to torch tensors (torch_ref_wrapper doesn't recurse into dataclasses)
    import numpy as np

    if isinstance(gate_up_proj_scale, np.ndarray):
        gate_up_proj_scale = torch.from_numpy(gate_up_proj_scale.astype(np.float32))
    if isinstance(down_proj_scale, np.ndarray):
        down_proj_scale = torch.from_numpy(down_proj_scale.astype(np.float32))

    # Extract config from spec
    checkpoint_activation = False
    expert_affinity_multiply_on_I = False
    if spec.shard_on_I is not None:
        checkpoint_activation = spec.shard_on_I.checkpoint_activation
        expert_affinity_multiply_on_I = spec.shard_on_I.expert_affinity_multiply_on_I

    impl = spec.implementation

    # MX path: dequantize weights to fp32, then delegate to existing torch ref
    if impl in (
        MoECTEImplementation.shard_on_block_mx,
        MoECTEImplementation.shard_on_i_mx,
        MoECTEImplementation.shard_on_i_mx_hybrid,
    ):
        import numpy as np

        from .bwmm_func import _mx_internal_data

        internal = _mx_internal_data.get(id(spec))
        if internal is None:
            raise ValueError("MX path requires '_internal' in _mx_internal_data (set by input generator)")

        # Get fp32 weights stored during input generation (exact dequantized equivalents)
        gup_fp32 = internal['gate_up_proj_weights_fp32']  # (E*128, 2*n_H512_tile*I_TP*4)
        down_fp32 = internal['down_proj_weights_fp32']  # (E*I_TP_par_dim, n_I512_tile*H*4)

        E = gate_up_proj_weight.shape[0]
        H = hidden_states.shape[-1] if isinstance(hidden_states, torch.Tensor) else hidden_states.shape[-1]
        # Infer I_TP from fp32 weight size: total = E*2*I_TP*H
        total_gup = gup_fp32.size
        I_TP = total_gup // (E * 2 * H)

        # Reshape to standard layout [E, H, 2, I_TP]
        gup_reshaped = gup_fp32.reshape(E, H, 2 * I_TP).reshape(E, H, 2, I_TP)
        gup_torch = torch.from_numpy(gup_reshaped.astype(np.float32))

        # Down: total = E*I_TP*H elements
        down_reshaped = down_fp32.reshape(E, I_TP, H)
        down_torch = torch.from_numpy(down_reshaped.astype(np.float32))

        # Map to BWMMFunc for torch ref
        mx_to_bwmm = {
            MoECTEImplementation.shard_on_block_mx: BWMMFunc.SHARD_ON_BLOCK,
            MoECTEImplementation.shard_on_i_mx: BWMMFunc.SHARD_ON_INTERMEDIATE,
            MoECTEImplementation.shard_on_i_mx_hybrid: BWMMFunc.SHARD_ON_INTERMEDIATE_HW,
        }
        bwmm_func = mx_to_bwmm[impl]
        top_k = 2 if is_tensor_update_accumulating else 1

        return _moe_cte_torch_ref_impl(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gup_torch,
            down_proj_weight=down_torch,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            block_size=block_size,
            bwmm_func=bwmm_func,
            lnc_degree=2,
            conditions=conditions,
            gate_and_up_proj_bias=None,  # MX bias has different layout, skip for now
            down_proj_bias=None,
            gate_up_proj_scale=None,
            down_proj_scale=None,
            activation_function=activation_function,
            skip_dma=skip_dma,
            compute_dtype=compute_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            checkpoint_activation=checkpoint_activation,
            expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
            top_k=top_k,
        )

    # Non-MX path: delegate to existing torch ref
    bwmm_func = impl_to_bwmm.get(impl)
    if bwmm_func is None:
        raise ValueError(f"No BWMMFunc mapping for {impl}")

    # Infer top_k for torch ref: exact value doesn't matter, only whether > 1
    top_k = 2 if is_tensor_update_accumulating else 1

    return _moe_cte_torch_ref_impl(
        hidden_states=hidden_states,
        expert_affinities_masked=expert_affinities_masked,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        block_size=block_size,
        bwmm_func=bwmm_func,
        lnc_degree=2,
        conditions=conditions,
        gate_and_up_proj_bias=gate_and_up_proj_bias,
        down_proj_bias=down_proj_bias,
        gate_up_proj_scale=gate_up_proj_scale,
        down_proj_scale=down_proj_scale,
        activation_function=activation_function,
        skip_dma=skip_dma,
        compute_dtype=compute_dtype,
        is_tensor_update_accumulating=is_tensor_update_accumulating,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        checkpoint_activation=checkpoint_activation,
        expert_affinity_multiply_on_I=expert_affinity_multiply_on_I,
        top_k=top_k,
    )
