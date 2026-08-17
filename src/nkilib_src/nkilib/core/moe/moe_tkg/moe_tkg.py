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

"""This module implements the Mixture of Experts (MoE) token generation kernel with support for all-expert and selective-expert modes."""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...utils.allocator import sizeinbytes

# common utils
from ...utils.common_types import (
    ActFnType,
    DtypeMode,
    ExpertAffinityScaleMode,
    MoEAllToAllVStrategy,
    MoEBlockIOLayout,
    NormType,
    QuantizationType,
)
from ...utils.kernel_assert import kernel_assert
from ...utils.kernel_helpers import get_verified_program_sharding_info
from .all_expert_impl import _all_expert_moe_tkg
from .all_expert_mx_impl import BF16_PER_INT32, _all_expert_moe_tkg_mx

# MLP utils
from .mlp_parameters import MLPExpertParameters, MLPParameters, get_T_from_hidden_input
from .moe_tkg_affinity_masking import mask_expert_affinities
from .selective_expert_impl import _selective_expert_moe_tkg
from .selective_expert_mx_impl import _selective_expert_moe_tkg_mxfp4

# Constants
_SUPPORTED_MX_DTYPES = (nl.float4_e2m1fn_x4, nl.float8_e4m3fn_x4)
_SUPPORTED_ALLTOALLV_STRATEGIES = (
    MoEAllToAllVStrategy.DISABLED,
    MoEAllToAllVStrategy.PRESERVE_ROW_ORDER,
    MoEAllToAllVStrategy.PACK_OUTPUT_ROWS,
)
_MOE_TKG_ERROR_PREFIX = "[MoE TKG Kernel]"


@nki.jit
def moe_tkg(
    hidden_input: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_affinities: nl.NkiTensor,
    expert_index: nl.NkiTensor,
    is_all_expert: bool,
    rank_id: Optional[nl.NkiTensor] = None,
    expert_gate_up_bias: Optional[nl.NkiTensor] = None,
    expert_down_bias: Optional[nl.NkiTensor] = None,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor] = None,
    expert_down_weights_scale: Optional[nl.NkiTensor] = None,
    hidden_input_scale: Optional[nl.NkiTensor] = None,
    expert_gate_up_input_scale: Optional[nl.NkiTensor] = None,
    expert_down_input_scale: Optional[nl.NkiTensor] = None,
    mask_unselected_experts: bool = False,
    expert_affinities_eager: Optional[nl.NkiTensor] = None,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.NO_SCALE,
    activation_fn: ActFnType = ActFnType.SiLU,
    output_dtype: nki.dtype = None,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    output_in_sbuf: bool = False,
    is_all_expert_dynamic: bool = False,
    block_size: int = None,
    input_dequant_scale: Optional[nl.NkiTensor] = None,
    all_to_all_v_strategy: MoEAllToAllVStrategy = MoEAllToAllVStrategy.DISABLED,
    output_layout: MoEBlockIOLayout = MoEBlockIOLayout.B_S_H,
    output: Optional[nl.NkiTensor] = None,
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
) -> nl.NkiTensor:
    """
    Mixture of Experts (MoE) MLP token generation kernel.

    Performs MoE computation with support for both all-expert and selective-expert modes.
    Supports various quantization types including FP8 row/static quantization and MxFP4.
    Optimized for token generation scenarios with T <= 128 (except MX all-expert mode).

    Supported input data types: bfloat16, float16, float4_e2m1fn_x4 (MxFP4).

    Dimensions:
        T: Number of tokens (batch_size * seq_len)
        H: Hidden dimension
        I: Intermediate dimension
        E: Number of global experts
        E_L: Number of local experts processed by this kernel
        K: Top-k experts per token
        I_p: I//4 if I <= 512 else 128
        H_concat: H + H/4 + E_L * 2 + 4. Number of columns when hidden_input, hidden_input_scale, local expert affinities,
            and global token index are concatenated and bitcast to a common fp8 dtype.

    Args:
        hidden_input (nl.NkiTensor): [T, H] or [T, H_concat] in HBM or [H0, T, H1] in SBUF, Input hidden states tensor.
            When all_to_all_v_strategy != DISABLED, input is expected to have layout [T, H_concat] and fp8 dtype,
            where H_concat = H + H/4 + E_L * 2 + 4 (hidden_quant | hidden_scale | expert_affinities | token_indices).
        expert_gate_up_weights (nl.NkiTensor): [E_L, H, 2, I] for bf16/fp16 or [E_L, 128, 2, ceil(H/512), I] for MxFP4,
            Fused gate and up projection weights.
        expert_down_weights (nl.NkiTensor): [E_L, I, H] for bf16/fp16 or [E_L, I_p, ceil(I/512), H] for MxFP4,
            Down projection weights.
        expert_affinities (nl.NkiTensor): [T, E], Expert routing weights/affinities. None when
            all_to_all_v_strategy != DISABLED (affinities are packed in hidden_input). For all-expert mode with
            affinity scaling, this will be sliced to [T, E_L] internally.
        expert_index (nl.NkiTensor): [T, K], Top-K expert indices per token. None when
            all_to_all_v_strategy != DISABLED.
        is_all_expert (bool): If True, process all experts for all tokens; otherwise, process only selected
            top-k experts.
        rank_id (nl.NkiTensor, optional): [1, 1], Rank ID tensor specifying which worker processes experts
            [E_L * rank_id, E_L * (rank_id + 1)). Required for all-expert mode with affinity scaling enabled.
        expert_gate_up_bias (nl.NkiTensor, optional): [E_L, 2, I] for non-MX or [E_L, I_p, 2, ceil(I/512), 4]
            for MX, Bias for gate/up projections.
        expert_down_bias (nl.NkiTensor, optional): [E_L, H], Bias for down projection.
        expert_gate_up_weights_scale (nl.NkiTensor, optional): [E_L, 2, I] for FP8 row quantization, [E_L, 2, 1] for
            FP8 static quantization, or [E_L, 128/8, 2, ceil(H/512), I] for MxFP4, Quantization scales for
            gate/up weights.
        expert_down_weights_scale (nl.NkiTensor, optional): [E_L, H] for FP8 row quantization, [E_L, 1] for FP8 static
            quantization, or [E_L, I_p/8, ceil(I/512), H] for MxFP4, Quantization scales for down weights.
        hidden_input_scale (nl.NkiTensor, optional): [H0, H/512, T], MX quantization scale for pre-quantized
            hidden_input in SBUF. When provided with MX weights in all-expert mode, indicates that hidden_input
            is already quantized and skips internal swizzle + quantization. The hidden_input buffer must be in
            SBUF when hidden_input_scale is provided. dtype: nl.uint8.
        expert_gate_up_input_scale (nl.NkiTensor, optional): [E_L, 1], FP8 dequantization scales for gate/up input.
            Used for static quantization.
        expert_down_input_scale (nl.NkiTensor, optional): [E_L, 1], FP8 dequantization scales for down input. Used for
            static quantization.
        mask_unselected_experts (bool): Whether to apply expert affinity masking based on expert_index. When
            True, affinities are masked to zero for experts not selected by each token. Only used in all-expert
            mode with affinity scaling. (default: False)
        expert_affinities_eager (nl.NkiTensor, optional): [T, K], Eager expert affinities. Not used in
            all_expert mode.
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): When to apply affinity scaling. Supported
            values: NO_SCALE, POST_SCALE. (default: NO_SCALE)
        activation_fn (ActFnType): Activation function type. (default: SiLU)
        output_dtype: Output tensor data type. Defaults to None.
            If None, uses hidden_input dtype when hidden_input is a ≥ 2 byte dtype or bf16 when hidden_input is quantized.
        gate_clamp_upper_limit (float, optional): Upper bound value to clamp gate projection results.
        gate_clamp_lower_limit (float, optional): Lower bound value to clamp gate projection results.
        up_clamp_upper_limit (float, optional): Upper bound value to clamp up projection results.
        up_clamp_lower_limit (float, optional): Lower bound value to clamp up projection results.
        output_in_sbuf (bool): If True, allocate output in SBUF with same shape as hidden_input. If False
            (default), allocate output in HBM with shape [T, H].
        is_all_expert_dynamic (bool): If True, configures all-expert algorithm to use dynamic control flow.
            If False (default), utilizes all-expert algorithm without dynamic control flow. Only valid when is_all_expert=True.
        block_size (int): Block size for all-expert dynamic algorithm, used to group tokens for dynamic control flow. Required argument
            when is_all_expert_dynamic=True. block_size must:
            - Evenly divide T, resulting in at least 2 blocks.
            - Be divisible by 4 and ≤ 32, divisible by 32 and ≤ 128, or divisible by 128.
        input_dequant_scale (nl.NkiTensor, optional): [128, 1] in SBUF, Pre-computed input FP8 dequantization
            scale for STATIC_MX mode. Passed from moe_block_tkg which computes it during the fused
            RMSNorm+quantize step. Used by the all-expert MX path to combine with per-expert weight
            dequant scales for post-matmul dequantization. Derived from expert_gate_up_input_scale.
        all_to_all_v_strategy (MoEAllToAllVStrategy): Input/output permutation strategy when all_to_all_v (A2A-v) is used.
            Currently only supported on Trn3 with MX weights.
            - DISABLED: Default; A2A-v is not used.
            - PRESERVE_ROW_ORDER: Output row ordering matches input row ordering. Token indices are appended as trailing 2 columns of output.
            - PACK_OUTPUT_ROWS: Output rows are packed, with routed tokens placed in the first N rows, where N is the number of routed tokens.
                Final T-N rows are padded with 0s. Token indices are appended as trailing 2 columns of output.
                When this strategy is used, the final 4 elements of hidden_input must be 0 for all padded rows, and the real token indices must be 1-indexed.
        output_layout (MoEBlockIOLayout): Output tensor layout. When _128_Nprgs_Hfree_T, output is
            [128, n_prgs, H//128//n_prgs, T]. Not supported with output_in_sbuf. Default is B_S_H.
        dtype_mode (DtypeMode): Explicit FP8 E4M3 dtype selection for STATIC/ROW
            quantization weight tiles (mirrors core/mlp).
            - ``DtypeMode.NON_OCP`` (default): ``nl.float8_e4m3`` (max=240).
            - ``DtypeMode.OCP``: ``nl.float8_e4m3fn`` (max=448). TRN3 only.
            - ``DtypeMode.AUTO``: ``nl.float8_e4m3fn`` on TRN3, ``nl.float8_e4m3``
              elsewhere.

    Returns:
        output (nl.NkiTensor): [T, H] or [128, n_prgs, H//128//n_prgs, T] depending on output_layout,
            or same shape as hidden_input if output_in_sbuf=True. Output tensor with
            MoE computation results.

    Notes:
        - T <= 128 (batch_size * seq_len must be <= 128, except for MX all-expert mode)
        - PRE_SCALE and PRE_SCALE_DELAYED modes are not supported
        - Column tiling is disabled for MoE kernels
        - STATIC_MX quantization requires MX weights with expert_gate_up_input_scale and expert_down_input_scale

    Pseudocode:
        # Mask expert affinities if needed (all-expert mode with affinity scaling)
        if is_all_expert and expert_affinities_scaling_mode != NO_SCALE:
            masked_expert_affinities = mask_expert_affinities(expert_affinities, expert_index, rank_id)

        # Process experts
        output = zeros([T, H])
        for each expert (all-expert) or selected expert (selective-expert):
            gate_proj_out = hidden_states @ gate_weights
            act_gate_proj = activation_fn(gate_proj_out)
            up_proj_out = hidden_states @ up_weights
            intermediate = act_gate_proj * up_proj_out
            expert_out = intermediate @ down_weights
            if expert_affinities_scaling_mode == POST_SCALE:
                expert_out *= affinity
            output += expert_out
    """

    # Extract quantization type
    quant_type, is_mx_kernel = _extract_quantization_type(
        expert_gate_up_weights=expert_gate_up_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        expert_gate_up_input_scale=expert_gate_up_input_scale,
        expert_down_input_scale=expert_down_input_scale,
    )

    # For all-expert mode with affinity scaling and without A2A-v, mask expert affinities based on rank_id
    if (
        is_all_expert
        and expert_affinities_scaling_mode != ExpertAffinityScaleMode.NO_SCALE
        and all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED
    ):
        kernel_assert(
            rank_id != None, f"{_MOE_TKG_ERROR_PREFIX} rank_id is required for all-expert mode with affinity scaling"
        )

        # Get dimensions for masking
        E_L = expert_gate_up_weights.shape[0]
        T = get_T_from_hidden_input(hidden_input, hidden_input_scale)
        K = expert_index.shape[-1]

        masked_expert_affinities = mask_expert_affinities(
            expert_affinities=expert_affinities,
            expert_index=expert_index,
            rank_id=rank_id,
            E_L=E_L,
            T=T,
            K=K,
            io_dtype=expert_affinities.dtype,
            mask_unselected_experts=mask_unselected_experts,
            output_in_sbuf=not is_all_expert_dynamic,
        )
    else:
        masked_expert_affinities = expert_affinities

    # Initialize config objects
    expert_params = MLPExpertParameters(
        expert_affinities=masked_expert_affinities,
        expert_index=expert_index,
        expert_affinities_eager=expert_affinities_eager,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        is_all_expert_dynamic=is_all_expert_dynamic,
        all_to_all_v_strategy=all_to_all_v_strategy,
        block_size=block_size,
    )

    # Default output_dtype to hidden_input.dtype when hidden_input is not quantized, and bf16 when hidden_input is quantized.
    if output_dtype is None:
        output_dtype = hidden_input.dtype if sizeinbytes(hidden_input.dtype) >= 2 else nl.bfloat16

    # Column tiling improves PE utilization for small T (32, 64, 128) but requires:
    #   - Quantized weights (FP8): unquantized bf16/fp16 weights are not supported
    #   - 32 <= T <= 128: the column tiling path does not tile on the T dimension
    #   - Non-MX quantization: MX path has its own path
    T_for_heuristic = get_T_from_hidden_input(hidden_input, hidden_input_scale)
    _use_gate_up_col_tiling = (
        is_all_expert
        and not is_mx_kernel
        and quant_type in (QuantizationType.STATIC,)
        and 32 <= T_for_heuristic
        and T_for_heuristic <= 128
    )

    mlp_params = MLPParameters(
        hidden_tensor=hidden_input,
        gate_proj_weights_tensor=expert_gate_up_weights,
        up_proj_weights_tensor=expert_gate_up_weights,
        down_proj_weights_tensor=expert_down_weights,
        activation_fn=activation_fn,
        normalization_type=NormType.NO_NORM,
        gate_proj_bias_tensor=expert_gate_up_bias,
        up_proj_bias_tensor=expert_gate_up_bias,
        down_proj_bias_tensor=expert_down_bias,
        gate_w_scale=expert_gate_up_weights_scale,
        up_w_scale=expert_gate_up_weights_scale,
        down_w_scale=expert_down_weights_scale,
        gate_up_in_scale=expert_gate_up_input_scale,
        down_in_scale=expert_down_input_scale,
        hidden_input_scale=hidden_input_scale,
        input_dequant_scale=input_dequant_scale,
        output_dtype=output_dtype,
        use_tkg_gate_up_proj_column_tiling=_use_gate_up_col_tiling,
        use_tkg_down_proj_column_tiling=False,
        shard_on_h_disabled=is_mx_kernel and not is_all_expert,
        expert_params=expert_params,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        gate_clamp_lower_limit=gate_clamp_lower_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        quantization_type=quant_type,
        transposed_out=output_layout == MoEBlockIOLayout._128_Nprgs_Hfree_T,
        dtype_mode=dtype_mode,
    )

    T = mlp_params.sequence_len
    H = mlp_params.hidden_size

    # Validate inputs
    _validate_moe_tkg_inputs(
        T=T,
        is_all_expert=is_all_expert,
        is_all_expert_dynamic=is_all_expert_dynamic,
        all_to_all_v_strategy=all_to_all_v_strategy,
        block_size=block_size,
        is_mx_kernel=is_mx_kernel,
        expert_weight_dtype=expert_gate_up_weights.dtype,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        hidden_input_scale=hidden_input_scale,
        expert_affinities_scaling_mode=expert_affinities_scaling_mode,
        expert_gate_up_input_scale=expert_gate_up_input_scale,
        expert_down_input_scale=expert_down_input_scale,
        expert_affinities_eager=expert_affinities_eager,
        output_dtype=output_dtype,
    )

    # Allocate output tensor if not provided by caller
    _T_LAYOUT = MoEBlockIOLayout._128_Nprgs_Hfree_T
    kernel_assert(
        not (output_layout == _T_LAYOUT and output_in_sbuf),
        f"{_MOE_TKG_ERROR_PREFIX} output_layout=_128_Nprgs_Hfree_T is not supported with output_in_sbuf=True",
    )
    if output is None:
        if output_in_sbuf:
            output = nl.ndarray(hidden_input.shape, dtype=output_dtype, buffer=nl.sbuf)
        elif output_layout == _T_LAYOUT:
            _, n_prgs, _ = get_verified_program_sharding_info("moe_tkg", (0, 1))
            H0 = 128
            H1_shard = H // (H0 * n_prgs)
            output = nl.ndarray((H0, n_prgs, H1_shard, T), dtype=output_dtype, buffer=nl.shared_hbm)
        elif all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED:
            output = nl.ndarray((T, H), dtype=output_dtype, buffer=nl.shared_hbm)
        else:
            # We add BF16_PER_INT32 additional columns to end when using all_to_all_v, to store concatenated token indices
            output = nl.ndarray((T, H + BF16_PER_INT32), dtype=output_dtype, buffer=nl.shared_hbm)

    # Dispatch to expert MLP implementation
    if is_all_expert:
        if is_mx_kernel:
            _all_expert_moe_tkg_mx(mlp_params, output, output_t_offset=0)
        else:
            _all_expert_moe_tkg(mlp_params, output)
    else:
        if is_mx_kernel:
            _selective_expert_moe_tkg_mxfp4(mlp_params, output)
        else:
            _selective_expert_moe_tkg(mlp_params, output)

    return output


def _extract_quantization_type(
    expert_gate_up_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor],
    expert_down_weights_scale: Optional[nl.NkiTensor],
    expert_gate_up_input_scale: Optional[nl.NkiTensor],
    expert_down_input_scale: Optional[nl.NkiTensor],
) -> tuple[QuantizationType, bool]:
    """
    Extract quantization type from kernel parameters.

    Args:
        expert_gate_up_weights (nl.NkiTensor): Gate/up projection weights tensor.
        expert_gate_up_weights_scale (nl.NkiTensor, optional): Quantization scale for gate/up weights.
        expert_down_weights_scale (nl.NkiTensor, optional): Quantization scale for down weights.
        expert_gate_up_input_scale (nl.NkiTensor, optional): FP8 dequantization scale for gate/up input.
        expert_down_input_scale (nl.NkiTensor, optional): FP8 dequantization scale for down input.

    Returns:
        tuple[QuantizationType, bool]: (quant_type, is_mx_kernel) tuple indicating the detected
            quantization type and whether MX quantization is used.
    """

    quant_type = QuantizationType.NONE
    is_mx_kernel = False
    if expert_gate_up_weights.dtype in _SUPPORTED_MX_DTYPES:
        is_mx_kernel = True
        if expert_gate_up_input_scale != None and expert_down_input_scale != None:
            quant_type = QuantizationType.STATIC_MX
        elif (
            expert_gate_up_weights_scale != None
            and expert_down_weights_scale != None
            and expert_gate_up_weights_scale.dtype != nl.uint8
            and expert_gate_up_weights_scale.shape[-1] > 1
        ):
            quant_type = QuantizationType.ROW_MX
        else:
            quant_type = QuantizationType.MX
    elif expert_gate_up_input_scale != None and expert_down_input_scale != None:
        quant_type = QuantizationType.STATIC
    elif expert_gate_up_weights_scale != None and expert_down_weights_scale != None:
        quant_type = QuantizationType.ROW

    return quant_type, is_mx_kernel


def _validate_moe_tkg_inputs(
    T: int,
    is_all_expert: bool,
    is_all_expert_dynamic: bool,
    all_to_all_v_strategy: MoEAllToAllVStrategy,
    block_size: int,
    is_mx_kernel: bool,
    expert_weight_dtype: nki.dtype,
    expert_gate_up_weights_scale: Optional[nl.NkiTensor],
    expert_down_weights_scale: Optional[nl.NkiTensor],
    hidden_input_scale: Optional[nl.NkiTensor],
    expert_affinities_scaling_mode: ExpertAffinityScaleMode,
    expert_gate_up_input_scale: Optional[nl.NkiTensor],
    expert_down_input_scale: Optional[nl.NkiTensor],
    expert_affinities_eager: Optional[nl.NkiTensor],
    output_dtype: nki.dtype,
) -> None:
    """
    Validate MoE TKG kernel input parameters.

    Args:
        T (int): Number of tokens.
        is_all_expert (bool): Whether using all-expert mode.
        is_all_expert_dynamic (bool): Whether all-expert mode uses dynamic control flow.
        is_all_to_all_v (bool): Whether MoE layer uses all_to_all_v collective, which provides sparse input and requires shuffled output.
        is_mx_kernel (bool): Whether using MX quantization.
        expert_gate_up_weights_scale (nl.NkiTensor, optional): Quantization scale for gate/up weights.
        expert_down_weights_scale (nl.NkiTensor, optional): Quantization scale for down weights.
        hidden_input_scale (nl.NkiTensor, optional): MX quantization scale for hidden input.
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): When to apply affinity scaling.
        expert_gate_up_input_scale (nl.NkiTensor, optional): FP8 dequantization scale for gate/up input.
        expert_down_input_scale (nl.NkiTensor, optional): FP8 dequantization scale for down input.
        expert_affinities_eager (nl.NkiTensor, optional): Eager expert affinities.
        output_dtype (nki.dtype): Resolved output dtype (must be at least 2 bytes).

    Returns:
        None
    """
    # MX quantization requires scales
    if is_mx_kernel:
        kernel_assert(
            expert_gate_up_weights_scale != None and expert_down_weights_scale != None,
            f"{_MOE_TKG_ERROR_PREFIX} Scales must be set when using MX weights",
        )
        kernel_assert(
            nisa.get_nc_version() >= nisa.nc_version.gen4,
            f"{_MOE_TKG_ERROR_PREFIX} MX weights are only supported on gen4+ (Trn3+) but got {nisa.get_nc_version()=}",
        )

    # Dynamic control flow requires is_all_expert=True and block_size != None
    if is_all_expert_dynamic:
        kernel_assert(
            is_all_expert,
            f"{_MOE_TKG_ERROR_PREFIX} is_all_expert_dynamic=True requires is_all_expert=True, but got {is_all_expert=}",
        )
        kernel_assert(
            block_size != None,
            f"{_MOE_TKG_ERROR_PREFIX} is_all_expert_dynamic=True requires block_size != None, but got {block_size=}",
        )
        # Validate all_to_all_v_strategy supported with dynamic control flow
        kernel_assert(
            all_to_all_v_strategy in _SUPPORTED_ALLTOALLV_STRATEGIES,
            f"{_MOE_TKG_ERROR_PREFIX} Unsupported all_to_all_v_strategy={all_to_all_v_strategy}. Supported: {_SUPPORTED_ALLTOALLV_STRATEGIES}",
        )
    # all_to_all_v requires is_all_expert_dynamic
    else:
        kernel_assert(
            all_to_all_v_strategy == MoEAllToAllVStrategy.DISABLED,
            f"{_MOE_TKG_ERROR_PREFIX} all_to_all_v_strategy != DISABLED is only supported with is_all_expert_dynamic=True, but got {is_all_expert_dynamic=}, {all_to_all_v_strategy=}",
        )

    # hidden_input_scale only supported with MX weights in ALL_EXPERT mode
    kernel_assert(
        hidden_input_scale == None or (is_mx_kernel and is_all_expert),
        f"{_MOE_TKG_ERROR_PREFIX} hidden_input_scale is only supported with MX weights in all-expert mode",
    )

    # Token count limitation (except for all-expert mode which supports T-tiling)
    kernel_assert(
        T <= 128 or is_all_expert,
        f"{_MOE_TKG_ERROR_PREFIX} Currently only batch size * seq len <= 128 is supported (except for all-expert mode)",
    )

    # Affinity scaling mode restrictions
    kernel_assert(
        expert_affinities_scaling_mode != ExpertAffinityScaleMode.PRE_SCALE_DELAYED,
        f"{_MOE_TKG_ERROR_PREFIX} PRE_SCALE_DELAYED option is only applicable in CTE expert_mlp case",
    )

    kernel_assert(
        expert_affinities_scaling_mode != ExpertAffinityScaleMode.PRE_SCALE,
        f"{_MOE_TKG_ERROR_PREFIX} Kernel does not support pre-scale mode",
    )

    kernel_assert(
        sizeinbytes(output_dtype) >= 2,
        f"{_MOE_TKG_ERROR_PREFIX} output_dtype must be at least a 2-byte dtype, got {output_dtype}",
    )

    # Static quantization without MX weights: input stays BF16, TRN2 does BF16 × FP8 matmul.
    # gate_up_in_scale / down_in_scale are passed for detection but not used in the kernel.

    # All-expert mode restrictions
    if is_all_expert:
        kernel_assert(
            expert_affinities_eager == None,
            f"{_MOE_TKG_ERROR_PREFIX} expert_affinities eager mode not supported with is_all_expert=True",
        )
