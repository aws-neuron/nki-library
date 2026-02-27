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

"""MoE Block kernel for token generation with RMSNorm, RouterTopK, and Expert MLPs."""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ..moe.moe_tkg.moe_tkg import moe_tkg as _moe_tkg
from ..router_topk.router_topk import XSBLayout_tp102__0, XSBLayout_tp201__2, XSBLayout_tp2013__1
from ..router_topk.router_topk import router_topk as _router_topk
from ..subkernels.rmsnorm_mx_quantize_tkg import rmsnorm_mx_quantize_tkg as _rmsnorm_mx_quantize_tkg
from ..subkernels.rmsnorm_tkg import rmsnorm_tkg as _rmsnorm_tkg
from ..utils.common_types import ActFnType, ExpertAffinityScaleMode, RouterActFnType
from .moe_block_tkg_utils import (
    _pmax,
    _q_width,
    get_sbuf_tensor_shape,
    parse_moe_block_config,
    validate_moe_block_inputs,
)


@nki.jit
def moe_block_tkg(
    inp: nl.ndarray,
    gamma: nl.ndarray,
    router_weights: nl.ndarray,
    expert_gate_up_weights: nl.ndarray,
    expert_down_weights: nl.ndarray,
    shared_expert_gate_w: Optional[nl.ndarray] = None,
    shared_expert_up_w: Optional[nl.ndarray] = None,
    shared_expert_down_w: Optional[nl.ndarray] = None,
    expert_gate_up_weights_scale: Optional[nl.ndarray] = None,
    expert_down_weights_scale: Optional[nl.ndarray] = None,
    router_bias: Optional[nl.ndarray] = None,
    expert_gate_up_bias: Optional[nl.ndarray] = None,
    expert_down_bias: Optional[nl.ndarray] = None,
    shared_expert_gate_bias: Optional[nl.ndarray] = None,
    shared_expert_up_bias: Optional[nl.ndarray] = None,
    shared_expert_down_bias: Optional[nl.ndarray] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
    router_pre_norm: bool = True,
    norm_topk_prob: bool = False,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    hidden_act_fn: ActFnType = ActFnType.SiLU,
    hidden_act_scale_factor: Optional[float] = None,
    hidden_act_bias: Optional[float] = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    router_mm_dtype=nl.bfloat16,
    hidden_actual: Optional[int] = None,
    skip_router_logits: bool = False,
    is_all_expert: bool = False,
    rank_id: Optional[nl.ndarray] = None,
    residual: Optional[nl.ndarray] = None,
):
    """
    Unified MoE Block kernel for token generation supporting selective-expert and all-expert modes.

    Performs RMSNorm + RouterTopK + (optional) SharedExpert + ExpertMLPs on the input.
    Optimized for token generation with T <= 128 tokens in selective-expert mode, or T divisible
    by 4 for MXFP in all-expert mode. Requires LNC-2 sharding configuration.

    Dimensions:
        B: Batch size
        S: Sequence length
        T: Total number of input tokens (equivalent to B x S)
        H: Hidden dimension size of the model
        I: Intermediate dimension size of the model after tensor parallelism
        E: Number of experts
        K: Top K experts selected for each token

    Args:
        inp (nl.ndarray): [B, S, H], Active input tensor on HBM.
        gamma (nl.ndarray): [1, H], Normalization weights on HBM.
        router_weights (nl.ndarray): [H, E], Router weights on HBM.
        expert_gate_up_weights (nl.ndarray): [E, H, 2, I] for bf16/fp16 OR [E, 128, 2, ceil(H/512), I] for MX,
            Fused gate and up projection weights on HBM.
        expert_down_weights (nl.ndarray): [E, I, H] for bf16/fp16 OR [E, I_p, ceil(I/512), H] for MX,
            Down projection weights on HBM. I_p = I//4 if I <= 512 else 128.
        shared_expert_gate_w (nl.ndarray): [H, I], Optional gate projection weights for shared expert on HBM.
        shared_expert_up_w (nl.ndarray): [H, I], Optional up projection weights for shared expert on HBM.
        shared_expert_down_w (nl.ndarray): [I, H], Optional down projection weights for shared expert on HBM.
        expert_gate_up_weights_scale (nl.ndarray): [E, 16, 2, ceil(H/512), I], Optional MxFP quantization scales
            for gate and up projection weights. Required when expert_gate_up_weights dtype is MX.
        expert_down_weights_scale (nl.ndarray): [E, I_p/8, ceil(I/512), H], Optional MxFP quantization scales
            for down projection weights. Required when expert_down_weights dtype is MX.
        router_bias (nl.ndarray): [1, E], Optional bias for router computation.
        expert_gate_up_bias (nl.ndarray): [E, 2, I] for non-MX OR [E, I_p, 2, ceil(I/512), 4] for MX,
            Optional fused gate and up projection bias.
        expert_down_bias (nl.ndarray): [E, H], Optional down projection bias for expert computation.
        shared_expert_gate_bias (nl.ndarray): [1, I], Optional gate projection bias for shared expert. Placeholder.
        shared_expert_up_bias (nl.ndarray): [1, I], Optional up projection bias for shared expert. Placeholder.
        shared_expert_down_bias (nl.ndarray): [1, H], Optional down projection bias for shared expert. Placeholder.
        eps (float): Epsilon value used in RMSNorm.
        top_k (int): Number of top K experts selected for each token.
        router_act_fn (RouterActFnType): Activation function (softmax/sigmoid) applied on router_logits.
        router_pre_norm (bool): If True, apply softmax/sigmoid before TopK; otherwise after TopK.
        norm_topk_prob (bool): Whether to normalize top-k expert affinity values. Combined with router_pre_norm=True.
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Enum indicating whether and how to apply
            affinity scales.
        hidden_act_fn (ActFnType): Activation function for expert gate projection (SiLU, GELU, or Swish).
        hidden_act_scale_factor (float): Scaling factor (alpha) for expert activation function. Placeholder.
        hidden_act_bias (float): Scaling bias (beta) for expert activation function. Placeholder.
        gate_clamp_upper_limit (float): Upper bound to clamp expert MLP gate projection results. None to skip.
        gate_clamp_lower_limit (float): Lower bound to clamp expert MLP gate projection results. None to skip.
        up_clamp_upper_limit (float): Upper bound to clamp expert MLP up projection results. None to skip.
        up_clamp_lower_limit (float): Lower bound to clamp expert MLP up projection results. None to skip.
        router_mm_dtype: Dtype for Router matmul (nl.bfloat16, nl.float16, or nl.float32).
            Performs tensor copy cast if different from inp.dtype.
        hidden_actual (int): If specified, use this value for RMSNorm mean calculation instead of H.
            Handles padded hidden dimensions.
        skip_router_logits (bool): Whether to skip returning router logits tensor.
        is_all_expert (bool): If True, use all-expert mode (iterate over local experts).
            If False, use selective-expert mode (iterate over tokens).
        rank_id (nl.ndarray): [1, 1], Worker rank for expert sharding. Required when is_all_expert=True.
        residual (nl.ndarray): [B, S, H] or [T, H], Optional residual tensor for fused residual add.
            Only supported for MXFP in all_expert mode.

    Returns:
        out (nl.ndarray): [T, H], Output tensor of the kernel.
        router_logits (nl.ndarray): [T, E], Router logits. Returned when skip_router_logits is False.
        residual_out (nl.ndarray): [T, H], Residual output. Returned when residual is provided (all_expert mode).

    Notes:
        - H must be divisible by 128 (partition size) and by 256 (128 * n_prgs for LNC-2)
        - Selective-expert mode: T <= 128
        - All-expert mode with MXFP: T must be divisible by 4
        - All-expert mode without MXFP: T <= 128
        - Shared expert support is not yet implemented
        - hidden_act_scale_factor and hidden_act_bias are placeholders (must be None)

    Pseudocode:
        # Step 1: RMSNorm
        rmsnorm_out = rmsnorm(inp, gamma, eps)

        # Step 2: Router + TopK
        router_logits = rmsnorm_out @ router_weights + router_bias
        expert_affinities = act_fn(router_logits)  # softmax or sigmoid
        expert_index, expert_affinities = topk(expert_affinities, k=top_k)

        # Step 3: (Optional) Shared Expert MLP
        if has_shared_expert:
            shared_out = shared_expert_mlp(rmsnorm_out)

        # Step 4: Expert MLPs
        out = moe_tkg(rmsnorm_out, expert_weights, expert_index, expert_affinities)

        return out, router_logits
    """

    # Parse dimensions and validate inputs
    dims, quant_config, expert_config = parse_moe_block_config(
        inp, router_weights, expert_gate_up_weights, shared_expert_gate_w, top_k, hidden_actual, is_all_expert
    )
    validate_moe_block_inputs(
        dims,
        quant_config,
        expert_config,
        shared_expert_gate_w,
        shared_expert_up_w,
        shared_expert_down_w,
        hidden_act_scale_factor,
        hidden_act_bias,
        router_mm_dtype,
        rank_id,
        residual,
    )

    # Convenience flags
    is_mxfp_all_expert = quant_config.is_moe_weight_mx and expert_config.is_all_expert

    # Step 1: perform RMSNorm (with optional MX quantization for MXFP all-expert mode)
    rmsnorm_out = nl.ndarray((_pmax, dims.T, dims.H_free), dtype=inp.dtype, buffer=nl.sbuf)
    rmsnorm_out_quant = None
    rmsnorm_out_scale = None
    residual_out = None

    if is_mxfp_all_expert:
        # MXFP all-expert mode: use fused RMSNorm + MX quantization
        num_H512_tiles = dims.H // (_pmax * _q_width)
        quant_shape = (_pmax, num_H512_tiles, dims.T)
        rmsnorm_out_quant = nl.ndarray(quant_shape, dtype=nl.float8_e4m3fn_x4, buffer=nl.sbuf)
        rmsnorm_out_scale = nl.ndarray(quant_shape, dtype=nl.uint8, buffer=nl.sbuf)
        residual_out = nl.ndarray((dims.T, dims.H), dtype=inp.dtype, buffer=nl.shared_hbm) if residual != None else None

        _rmsnorm_mx_quantize_tkg(
            input=inp,
            gamma=gamma,
            output=rmsnorm_out,
            output_quant=rmsnorm_out_quant,
            output_scale=rmsnorm_out_scale,
            residual=residual,
            output_residual=residual_out,
            eps=eps,
            hidden_actual=dims.hidden_actual,
            hidden_dim_tp=True,
        )
    else:
        # Non-MXFP or selective-expert mode: use standard RMSNorm
        rmsnorm_out = _rmsnorm_tkg(
            input=inp,
            gamma=gamma,
            output=rmsnorm_out,
            eps=eps,
            hidden_actual=dims.hidden_actual,
            hidden_dim_tp=quant_config.is_moe_weight_mx,
            single_core_forced=True
            if (not quant_config.is_moe_weight_mx and not expert_config.is_all_expert and dims.T > 1)
            else False,
        )

    router_in = rmsnorm_out
    if rmsnorm_out.dtype != router_mm_dtype:
        router_in = nl.ndarray((_pmax, dims.T, dims.H_free), dtype=router_mm_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=router_in, src=rmsnorm_out)

    # Step 2: compute router & topK
    router_logits = None if skip_router_logits else nl.ndarray((dims.T, dims.E), dtype=inp.dtype, buffer=nl.shared_hbm)
    # Selective-load: expert_index in SBUF, always stored (needed for selective loading)
    # All-expert: expert_index in HBM, skip store when not router_pre_norm
    skip_store_expert_index = expert_config.is_all_expert and not router_pre_norm
    expert_index = nl.ndarray(
        get_sbuf_tensor_shape(dims.T, dims.K, is_sbuf=True),
        dtype=nl.uint32,
        buffer=nl.sbuf,
        name='expert_index',
    )
    # For all-expert mode, expert_affinities goes to HBM (moe_tkg handles slicing to local experts)
    # For selective-expert mode, expert_affinities stays in SBUF
    affinities_in_sbuf = not expert_config.is_all_expert
    expert_affinities = nl.ndarray(
        get_sbuf_tensor_shape(dims.T, dims.E, is_sbuf=affinities_in_sbuf),
        dtype=nl.float32,
        buffer=nl.sbuf if affinities_in_sbuf else nl.shared_hbm,
        name='expert_affinities',
    )
    expert_affinities_eager = (
        nl.ndarray(get_sbuf_tensor_shape(dims.T, dims.K, is_sbuf=True), dtype=nl.float32, buffer=nl.sbuf)
        if not expert_config.is_all_expert
        else None
    )
    # Determine x_sb_layout based on rmsnorm output layout
    if quant_config.is_moe_weight_mx:
        router_x_sb_layout = XSBLayout_tp201__2
    elif not expert_config.is_all_expert and dims.T > 1:
        router_x_sb_layout = XSBLayout_tp102__0
    else:
        router_x_sb_layout = XSBLayout_tp2013__1

    router_outputs = _router_topk(
        x=router_in,
        w=router_weights,
        w_bias=router_bias,
        router_logits=router_logits,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=router_act_fn,
        k=dims.K,
        x_hbm_layout=0,  # Since x_input_in_sbuf=True, this value is not used but required by function signature
        x_sb_layout=router_x_sb_layout,
        output_in_sbuf=affinities_in_sbuf,
        router_pre_norm=router_pre_norm,
        norm_topk_prob=norm_topk_prob,
        use_column_tiling=True,
        return_eager_affi=not expert_config.is_all_expert
        and quant_config.is_moe_weight_mx,  # Only needed for selective-expert mxfp mode
        use_PE_broadcast_w_bias=is_mxfp_all_expert,
        shard_on_tokens=is_mxfp_all_expert or (not expert_config.is_all_expert and dims.T > 1),
        skip_store_expert_index=skip_store_expert_index,
        skip_store_router_logits=skip_router_logits,
        x_input_in_sbuf=True,
        expert_affin_in_sb=affinities_in_sbuf,
    )
    router_logits, expert_index, expert_affinities = router_outputs[0], router_outputs[1], router_outputs[2]
    if not expert_config.is_all_expert and quant_config.is_moe_weight_mx:
        expert_affinities_eager = router_outputs[3].reshape((dims.T, dims.K))

    # Step 3: [Optional] compute shared expert
    if expert_config.has_shared_expert:
        # TODO
        pass

    # Step 4: compute expert MLPs
    expert_mlp_in_scale = rmsnorm_out_scale if is_mxfp_all_expert else None
    # Determine if we're using shard_on_T for selective expert
    selective_expert_shard_on_T = not expert_config.is_all_expert and dims.T > 1
    if is_mxfp_all_expert:
        expert_mlp_in = rmsnorm_out_quant
    elif quant_config.is_moe_weight_mx or selective_expert_shard_on_T:
        # MXFP selective-expert or shard_on_T mode: use full rmsnorm output
        expert_mlp_in = rmsnorm_out
    else:
        # Non-MXFP mode without shard_on_T: shard the hidden dimension
        expert_mlp_in = nl.ndarray((_pmax, dims.T, dims.H_free_shard), dtype=inp.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(
            dst=expert_mlp_in, src=rmsnorm_out[:, :, nl.ds(dims.prg_id * dims.H_free_shard, dims.H_free_shard)]
        )

    result = _moe_tkg(
        hidden_input=expert_mlp_in,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        is_all_expert=expert_config.is_all_expert,
        rank_id=rank_id,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        hidden_input_scale=expert_mlp_in_scale,
        mask_unselected_experts=router_pre_norm,
        expert_affinities_eager=expert_affinities_eager if not expert_config.is_all_expert else None,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        activation_fn=hidden_act_fn,
        output_dtype=inp.dtype,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
    )

    # Process and return the output
    outputs = [result]
    if not skip_router_logits:
        outputs.append(router_logits)
    if residual_out != None:
        outputs.append(residual_out)

    return tuple(outputs)
