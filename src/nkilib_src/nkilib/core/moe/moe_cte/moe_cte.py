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

"""Unified entry point for MoE CTE blockwise matrix multiplication kernels."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional

import nki
import nki.language as nl

from ....experimental.moe.moe_cte.bwmm_shard_on_block_v2 import bwmm_shard_on_block as bwmm_shard_on_block_v2
from ...utils.common_types import ActFnType, ExpertAffinityScaleMode, QuantizationType
from .bwmm_shard_on_block import bwmm_shard_on_block
from .bwmm_shard_on_block_mx import bwmm_shard_on_block_mx
from .bwmm_shard_on_I import (
    blockwise_mm_baseline_shard_intermediate,
    blockwise_mm_baseline_shard_intermediate_hybrid,
    blockwise_mm_shard_intermediate_dropping,
)
from .moe_cte_utils import BlockShardStrategy, SkipMode


class MoECTEImplementation(Enum):
    """
    Implementation strategy for MoE CTE kernel.

    Variants:
        shard_on_block: Shards blocks across cores. Best for many blocks. (TRN2)
        shard_on_i: Shards intermediate dimension across cores. (TRN2)
        shard_on_i_hybrid: Shard on I with hybrid static/dynamic loop. (TRN2)
        shard_on_i_dropping: Shard on I for dropping layer. (TRN2)
        shard_on_block_mx: Shard on block with MxFP4/MxFP8 quantization. (TRN3)
        shard_on_i_mx: Shard on I with MxFP4/MxFP8 quantization. (TRN3)
        shard_on_i_mx_hybrid: Shard on I with MxFP4/MxFP8 and hybrid loop. (TRN3)
    """

    shard_on_block = 0
    shard_on_i = 1
    shard_on_i_hybrid = 2
    shard_on_i_dropping = 3
    shard_on_block_mx = 4
    shard_on_i_mx = 5
    shard_on_i_mx_hybrid = 6


# =============================================================================
# Implementation-specific configuration classes
# =============================================================================


@dataclass(frozen=True)
class QuantizationConfig(nl.NKIObject):
    """
    Configuration for quantization-related parameters in MoE CTE kernels.

    Contains dequantization scales for weight tensors used in quantized inference.
    These scales are applied during matrix multiplication to convert quantized
    weights back to the target compute precision.

    Attributes:
        gate_up_proj_scale (nl.NkiTensor, optional): Dequantization scales for gate/up
            projection weights. Shape depends on quantization granularity (per-tensor,
            per-channel, or per-group). Required for quantized gate_up_proj_weight.
            Default: None (no quantization)

        down_proj_scale (nl.NkiTensor, optional): Dequantization scales for down
            projection weights. Shape depends on quantization granularity.
            Required for quantized down_proj_weight.
            Default: None (no quantization)

    Example:
        # No quantization (default)
        quant_cfg = QuantizationConfig()

        # With per-tensor scales
        quant_cfg = QuantizationConfig(
            gate_up_proj_scale=gate_up_scale_tensor,
            down_proj_scale=down_scale_tensor,
        )
    """

    gate_up_proj_scale: Optional[nl.NkiTensor] = None
    down_proj_scale: Optional[nl.NkiTensor] = None


@dataclass
class ShardOnBlockConfig(nl.NKIObject):
    """
    Configuration for block-sharding implementations.

    Applies to:
        - shard_on_block: Standard block sharding on TRN2
        - shard_on_block_mx: Block sharding with MxFP4/MxFP8 quantization on TRN3

    Attributes:
        block_sharding_strategy (BlockShardStrategy): Strategy for distributing blocks.
            - PING_PONG: Alternates blocks between shards (0, 1, 0, 1, ...)
            - HI_LO: First half to shard 0, second half to shard 1
            Used by: shard_on_block
            Default: PING_PONG

        n_static_blocks (int): Number of blocks processed in static loop (known at compile time).
            Set to -1 to auto-compute based on N and E.
            Used by: shard_on_block_mx
            Default: -1

        n_dynamic_blocks (int): Maximum number of blocks in dynamic loop (runtime-dependent).
            Used for hybrid static/dynamic loop optimization with padded sequences.
            Used by: shard_on_block_mx
            Default: 55

        use_packed_scales (bool): When True, gate_up_proj_scale and down_proj_scale are
            in packed HBM layout — packed as H_2k instead of H_512, so each 32-partition
            quadrant holds 4 MX scale tiles stacked in a 4-partition stripe (totalling
            up to 16 partitions of the quadrant), vs. one tile per quadrant in the
            standard layout. Producers must emit both scales packed.
            Used by: shard_on_block_mx
            Default: False

        weight_dtype (nki.dtype, optional): Target dtype for MX weight conversion when
            weights are passed as uint/int/float carrier types. For MXFP4:
            nl.float4_e2m1fn_x4. For MXFP8: nl.float8_e4m3fn_x4 or nl.float8_e5m2_x4.
            If None, the kernel auto-detects.
            Used by: shard_on_block_mx
            Default: None

        top_k (int): Top-K experts per token. Used together with ep_degree for the
            kernel's auto-computed best-case static-block estimate when n_static_blocks
            is unset and an over-range n_dynamic_blocks is given.
            Used by: shard_on_block_mx
            Default: 1

        ep_degree (int): Expert-parallelism degree the routing is sharded across.
            Used with top_k for the static-block estimate (see above).
            Used by: shard_on_block_mx
            Default: 1

        quantization_type (QuantizationType): Selects the quantization path. When
            STATIC_MX, the kernel takes the per-tensor MXFP (FP8/FP4 + identity
            micro-scale) path. The per-expert weight scales reuse gate_up_proj_scale
            ([E, 2, 1]) and down_proj_scale ([E, 1]); the per-tensor input scales are
            passed via gate_up_in_scale / down_in_scale. MX (default) is the standard
            microscaling path; any non-STATIC_MX value routes to it.
            Used by: shard_on_block_mx
            Default: QuantizationType.MX
    """

    # shard_on_block parameters
    n_block_per_iter: int = 1
    block_sharding_strategy: BlockShardStrategy = BlockShardStrategy.PING_PONG
    non_overlapping_shards: bool = False

    # shard_on_block_mx parameters (MxFP4/MxFP8 quantization)
    n_static_blocks: int = -1
    n_dynamic_blocks: int = 55
    use_packed_scales: bool = False
    weight_dtype: Any = None
    top_k: int = 1
    ep_degree: int = 1
    quantization_type: QuantizationType = QuantizationType.MX


@dataclass
class ShardOnIConfig(nl.NKIObject):
    """
    Configuration for intermediate-dimension-sharding implementations.

    Applies to:
        - shard_on_i: Standard I-sharding, processes all blocks sequentially
        - shard_on_i_hybrid: I-sharding with hybrid static/dynamic loop for padded sequences
        - shard_on_i_dropping: I-sharding for token-dropping MoE layers

    Attributes:
        checkpoint_activation (bool): Save gate/up activations for backward pass.
            When True, returns (output, gate_up_activations_T) or
            (output, gate_up_activations_T, down_activations) depending on
            expert_affinity_multiply_on_I setting.
            Used by: shard_on_i, shard_on_i_dropping
            Default: False

        expert_affinity_multiply_on_I (bool): Where to apply expert affinity scaling.
            - True: Apply on intermediate dimension (I) after activation function
            - False: Apply on hidden dimension (H) after down projection
            Affects which activations are saved when checkpoint_activation=True.
            Used by: shard_on_i, shard_on_i_dropping
            Default: False

        num_static_block (int, optional): Number of non-padded blocks for static loop.
            When provided, the kernel uses a hybrid loop: static loop for the first
            num_static_block blocks (compile-time known), then dynamic loop for
            remaining potentially-padded blocks (runtime-dependent).
            Set to None to auto-compute as (N - E).
            Used by: shard_on_i_hybrid
            Default: None
    """

    # shard_on_i and shard_on_i_dropping parameters
    checkpoint_activation: bool = False
    expert_affinity_multiply_on_I: bool = False

    # shard_on_i_hybrid parameters
    num_static_block: Optional[int] = None


# =============================================================================
# Main specification class
# =============================================================================


@dataclass(frozen=True)
class MoECTESpec(nl.NKIObject):
    """
    User-facing specification for MoE CTE kernel execution.

    This is the primary interface for configuring MoE CTE kernels. Users create instances
    of this class and pass them to moe_cte(). Internally, MoECTESpec is converted to
    MoECTEShardingSpecs which handles default initialization based on the implementation.

    Uses composition pattern with two config objects based on sharding strategy:
        - shard_on_block: For block-sharding implementations (shard_on_block, shard_on_block_mx)
        - shard_on_I: For I-sharding implementations (shard_on_i, shard_on_i_hybrid, shard_on_i_dropping)

    Attributes:
        implementation (MoECTEImplementation): Which implementation variant to use.

        shard_on_block (ShardOnBlockConfig, optional): Config for block-sharding implementations.
            Required for: shard_on_block, shard_on_block_mx
            Auto-initialized with defaults if not provided.

        shard_on_I (ShardOnIConfig, optional): Config for I-sharding implementations.
            Required for: shard_on_i, shard_on_i_hybrid, shard_on_i_dropping
            Auto-initialized with defaults if not provided.

    Example:
        # shard_on_block with custom block iteration
        spec = MoECTESpec(
            implementation=MoECTEImplementation.shard_on_block,
            shard_on_block=ShardOnBlockConfig(n_block_per_iter=2),
        )

        # shard_on_block_mx with dynamic blocks
        spec = MoECTESpec(
            implementation=MoECTEImplementation.shard_on_block_mx,
            shard_on_block=ShardOnBlockConfig(n_static_blocks=10, n_dynamic_blocks=20),
        )

        # shard_on_i with activation checkpointing
        spec = MoECTESpec(
            implementation=MoECTEImplementation.shard_on_i,
            shard_on_I=ShardOnIConfig(checkpoint_activation=True),
        )

        # shard_on_i_hybrid with known static block count
        spec = MoECTESpec(
            implementation=MoECTEImplementation.shard_on_i_hybrid,
            shard_on_I=ShardOnIConfig(num_static_block=100),
        )

        # Simple usage (defaults auto-initialized internally)
        spec = MoECTESpec(implementation=MoECTEImplementation.shard_on_i)
    """

    implementation: MoECTEImplementation
    shard_on_block: Optional[ShardOnBlockConfig] = None
    shard_on_I: Optional[ShardOnIConfig] = None


@dataclass
class MoECTEShardingSpecs(nl.NKIObject):
    """
    Internal specification with initialized defaults for MoE CTE kernel execution.

    This class is used internally by moe_cte() to convert frozen, user-provided
    MoECTESpec into a fully-initialized configuration. It ensures that the appropriate
    config object (shard_on_block or shard_on_I) is populated with defaults based
    on the selected implementation.

    Users should not instantiate this class directly. Instead, create MoECTESpec
    and pass it to moe_cte(), which handles the conversion internally.

    Attributes:
        implementation (MoECTEImplementation): Which implementation variant to use.
            Copied from the input MoECTESpec.

        shard_on_block (ShardOnBlockConfig, optional): Config for block-sharding implementations.
            Initialized with ShardOnBlockConfig() defaults if the implementation is
            shard_on_block or shard_on_block_mx and no config was provided.

        shard_on_I (ShardOnIConfig, optional): Config for I-sharding implementations.
            Initialized with ShardOnIConfig() defaults if the implementation is
            shard_on_i, shard_on_i_hybrid, or shard_on_i_dropping and no config was provided.

    Example:
        # Internal usage within moe_cte():
        def moe_cte(..., spec: MoECTESpec, ...):
            sharding_spec = MoECTEShardingSpecs(spec)  # Converts and initializes defaults
            if sharding_spec.implementation == MoECTEImplementation.shard_on_i:
                cfg = sharding_spec.shard_on_I  # Guaranteed to be non-None
                ...
    """

    implementation: MoECTEImplementation
    shard_on_block: Optional[ShardOnBlockConfig] = None
    shard_on_I: Optional[ShardOnIConfig] = None

    def __init__(self, spec: MoECTESpec):
        self.implementation = spec.implementation
        self.shard_on_block = None
        self.shard_on_I = None

        if (
            spec.implementation == MoECTEImplementation.shard_on_block
            or spec.implementation == MoECTEImplementation.shard_on_block_mx
        ):
            if spec.shard_on_block is None:
                self.shard_on_block = ShardOnBlockConfig()
            else:
                self.shard_on_block = spec.shard_on_block

        elif (
            spec.implementation == MoECTEImplementation.shard_on_i
            or spec.implementation == MoECTEImplementation.shard_on_i_hybrid
            or spec.implementation == MoECTEImplementation.shard_on_i_dropping
        ):
            if spec.shard_on_I is None:
                self.shard_on_I = ShardOnIConfig()
            else:
                self.shard_on_I = spec.shard_on_I


@nki.jit
def moe_cte(
    hidden_states: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    token_position_to_id: nl.NkiTensor,
    block_to_expert: nl.NkiTensor,
    block_size: int,
    spec: MoECTESpec,
    conditions: Optional[nl.NkiTensor] = None,
    gate_and_up_proj_bias: Optional[nl.NkiTensor] = None,
    down_proj_bias: Optional[nl.NkiTensor] = None,
    quantization_config: Optional[QuantizationConfig] = None,
    gate_up_proj_scale: Optional[nl.NkiTensor] = None,
    down_proj_scale: Optional[nl.NkiTensor] = None,
    gate_up_activations_T: Optional[nl.NkiTensor] = None,
    down_activations: Optional[nl.NkiTensor] = None,
    activation_function: ActFnType = ActFnType.SiLU,
    skip_dma: Optional[SkipMode] = None,
    compute_dtype: Any = nl.bfloat16,
    is_tensor_update_accumulating: bool = True,
    expert_affinities_scaling_mode: ExpertAffinityScaleMode = ExpertAffinityScaleMode.POST_SCALE,
    gate_clamp_upper_limit: Optional[float] = None,
    gate_clamp_lower_limit: Optional[float] = None,
    up_clamp_upper_limit: Optional[float] = None,
    up_clamp_lower_limit: Optional[float] = None,
    # Per-expert STATIC_MX input scales. The per-expert weight scales reuse the existing
    # gate_up_proj_scale / down_proj_scale tensors (gate/up packed [E, 2, 1], down [E, 1]).
    gate_up_in_scale: Optional[nl.NkiTensor] = None,
    down_in_scale: Optional[nl.NkiTensor] = None,
    # shard_on_block_mx only: [1, 1] int32 expert-parallel rank of this instance; required on the
    # packed-affinity path (the kernel derives its global expert base as ep_rank * E_local on-device).
    ep_rank: Optional[nl.NkiTensor] = None,
    accumulation_dtype=None,
    skip_gate_proj: bool = False,
):
    """
    Unified entry point for MoE CTE blockwise matrix multiplication kernels.

    Dispatches to appropriate implementation based on spec.implementation. Supports multiple
    sharding strategies and quantization modes for different hardware targets.

    Dimensions:
        T: Total number of input tokens
        H: Hidden dimension size
        B: Block size (tokens per block)
        E: Number of experts
        N: Total number of blocks
        I_TP: Intermediate size divided by tensor parallelism degree

    Args:
        hidden_states (nl.NkiTensor): [T+1, H], Input token embeddings in HBM
        expert_affinities_masked (nl.NkiTensor): [(T+1)*E, 1], Expert routing weights in HBM
        gate_up_proj_weight (nl.NkiTensor): [E, H, 2, I_TP], Gate and up projection weights in HBM
        down_proj_weight (nl.NkiTensor): [E, I_TP, H], Down projection weights in HBM
        token_position_to_id (nl.NkiTensor): [N*B], Token to block position mapping in HBM
        block_to_expert (nl.NkiTensor): [N, 1], Expert assignment per block in HBM
        block_size (int): Number of tokens per block
        spec (MoECTESpec): Implementation selection and configuration
        conditions (nl.NkiTensor, optional): [N+1], Block padding indicators (HYBRID, BLOCK_MX)
        gate_and_up_proj_bias (nl.NkiTensor, optional): Bias for gate/up projections
        down_proj_bias (nl.NkiTensor, optional): Bias for down projection
        quantization_config (QuantizationConfig, optional): Quantization scales configuration
        gate_up_proj_scale (nl.NkiTensor, optional): Direct scale tensor for gate/up weights.
            Takes precedence over quantization_config. Use for MX quantization to avoid
            wrapping tensors in dataclass (NKI tracer limitation). Under STATIC_MX this
            instead carries the per-expert gate/up weight scales as [E, 2, 1].
        down_proj_scale (nl.NkiTensor, optional): Direct scale tensor for down weights.
            Takes precedence over quantization_config. Under STATIC_MX this instead
            carries the per-expert down weight scale as [E, 1].
        gate_up_activations_T (nl.NkiTensor, optional): Storage for gate/up activations
        down_activations (nl.NkiTensor, optional): Storage for down activations
        activation_function (ActFnType): Activation function type (default: SiLU)
        skip_dma (SkipMode): DMA skip configuration
        compute_dtype (nki.dtype): Data type for computations (default: bfloat16)
        is_tensor_update_accumulating (bool): Enable accumulation for TopK > 1
        expert_affinities_scaling_mode (ExpertAffinityScaleMode): Affinity scaling mode
        gate_clamp_upper_limit (float, optional): Upper clamp for gate projection
        gate_clamp_lower_limit (float, optional): Lower clamp for gate projection
        up_clamp_upper_limit (float, optional): Upper clamp for up projection
        up_clamp_lower_limit (float, optional): Lower clamp for up projection
        gate_up_in_scale, down_in_scale (nl.NkiTensor, optional): Per-expert STATIC_MX
            input quant scales for the gate/up and down projections respectively.
            Only used by shard_on_block_mx under STATIC_MX. (The matching weight
            dequant scales reuse gate_up_proj_scale / down_proj_scale.)

    Returns:
        output (nl.NkiTensor): Expert-processed token representations in HBM

    Notes:
        - shard_on_block, shard_on_i, shard_on_i_hybrid, shard_on_i_dropping target TRN2
        - shard_on_block_mx, shard_on_i_mx target TRN3 with MxFP4/MxFP8 quantization
        - Implementation-specific parameters are passed via MoECTESpec

    Pseudocode:
        if spec.implementation == shard_on_block:
            return bwmm_shard_on_block(...)
        elif spec.implementation == shard_on_i:
            return blockwise_mm_baseline_shard_intermediate(...)
        elif spec.implementation == shard_on_i_hybrid:
            return blockwise_mm_baseline_shard_intermediate_hybrid(...)
        elif spec.implementation == shard_on_i_dropping:
            return blockwise_mm_shard_intermediate_dropping(...)
        elif spec.implementation == shard_on_block_mx:
            return bwmm_shard_on_block_mx(...)
        elif spec.implementation == shard_on_i_mx:
            return blockwise_mm_shard_intermediate_mx(...)
        elif spec.implementation == shard_on_i_mx_hybrid:
            return blockwise_mm_shard_intermediate_mx_hybrid(...)
    """

    if skip_dma is None:
        skip_dma = SkipMode(False, False)

    sharding_spec = MoECTEShardingSpecs(spec)
    print(f"spec: {sharding_spec}")

    # Extract quantization scales: direct params take precedence over config
    if gate_up_proj_scale is None or down_proj_scale is None:
        quant_cfg = quantization_config or QuantizationConfig()
        if gate_up_proj_scale is None:
            gate_up_proj_scale = quant_cfg.gate_up_proj_scale
        if down_proj_scale is None:
            down_proj_scale = quant_cfg.down_proj_scale

    if sharding_spec.implementation == MoECTEImplementation.shard_on_block:
        print(f"bwmm_shard_block_branch")
        cfg = sharding_spec.shard_on_block or ShardOnBlockConfig()
        if cfg.non_overlapping_shards:
            return bwmm_shard_on_block_v2(
                hidden_states=hidden_states,
                expert_affinities_masked=expert_affinities_masked,
                gate_up_proj_weight=gate_up_proj_weight,
                down_proj_weight=down_proj_weight,
                block_size=block_size,
                token_position_to_id=token_position_to_id,
                block_to_expert=block_to_expert,
                gate_and_up_proj_bias=gate_and_up_proj_bias,
                down_proj_bias=down_proj_bias,
                gate_up_proj_scale=gate_up_proj_scale,
                down_proj_scale=down_proj_scale,
                down_activations=down_activations,
                activation_function=activation_function,
                skip_dma=skip_dma,
                compute_dtype=compute_dtype,
                is_tensor_update_accumulating=is_tensor_update_accumulating,
                expert_affinities_scaling_mode=expert_affinities_scaling_mode,
                n_block_per_iter=cfg.n_block_per_iter,
                gate_clamp_upper_limit=gate_clamp_upper_limit,
                gate_clamp_lower_limit=gate_clamp_lower_limit,
                up_clamp_upper_limit=up_clamp_upper_limit,
                up_clamp_lower_limit=up_clamp_lower_limit,
                block_sharding_strategy=cfg.block_sharding_strategy,
                non_overlapping_shards=cfg.non_overlapping_shards,
            )
        return bwmm_shard_on_block(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            down_activations=down_activations,
            activation_function=activation_function,
            skip_dma=skip_dma,
            compute_dtype=compute_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            n_block_per_iter=cfg.n_block_per_iter,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            block_sharding_strategy=cfg.block_sharding_strategy,
        )

    elif sharding_spec.implementation == MoECTEImplementation.shard_on_i:
        print(f"bwmm_shard_I_branch")
        cfg = sharding_spec.shard_on_I or ShardOnIConfig()
        return blockwise_mm_baseline_shard_intermediate(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
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
            up_clamp_lower_limit=up_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            checkpoint_activation=cfg.checkpoint_activation,
            expert_affinity_multiply_on_I=cfg.expert_affinity_multiply_on_I,
            accumulation_dtype=accumulation_dtype,
        )

    elif sharding_spec.implementation == MoECTEImplementation.shard_on_i_hybrid:
        print(f"bwmm_shard_I_hybrid_branch")
        cfg = sharding_spec.shard_on_I or ShardOnIConfig()
        return blockwise_mm_baseline_shard_intermediate_hybrid(
            conditions=conditions,
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            num_static_block=cfg.num_static_block,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            gate_up_activations_T=gate_up_activations_T,
            down_activations=down_activations,
            activation_function=activation_function,
            skip_dma=skip_dma,
            compute_dtype=compute_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            accumulation_dtype=accumulation_dtype,
        )

    elif sharding_spec.implementation == MoECTEImplementation.shard_on_i_dropping:
        cfg = sharding_spec.shard_on_I or ShardOnIConfig()
        return blockwise_mm_shard_intermediate_dropping(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            block_size=block_size,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            activation_function=activation_function,
            skip_dma=skip_dma,
            compute_dtype=compute_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            expert_affinity_multiply_on_I=cfg.expert_affinity_multiply_on_I,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            accumulation_dtype=accumulation_dtype,
        )

    elif sharding_spec.implementation == MoECTEImplementation.shard_on_block_mx:
        cfg = sharding_spec.shard_on_block or ShardOnBlockConfig()
        return bwmm_shard_on_block_mx(
            hidden_states=hidden_states,
            expert_affinities_masked=expert_affinities_masked,
            gate_up_proj_weight=gate_up_proj_weight,
            down_proj_weight=down_proj_weight,
            token_position_to_id=token_position_to_id,
            block_to_expert=block_to_expert,
            conditions=conditions,
            gate_and_up_proj_bias=gate_and_up_proj_bias,
            down_proj_bias=down_proj_bias,
            gate_up_proj_scale=gate_up_proj_scale,
            down_proj_scale=down_proj_scale,
            ep_rank=ep_rank,
            block_size=block_size,
            n_static_blocks=cfg.n_static_blocks,
            n_dynamic_blocks=cfg.n_dynamic_blocks,
            gate_up_activations_T=gate_up_activations_T,
            down_activations=down_activations,
            activation_function=activation_function,
            skip_dma=skip_dma,
            compute_dtype=compute_dtype,
            weight_dtype=cfg.weight_dtype,
            is_tensor_update_accumulating=is_tensor_update_accumulating,
            expert_affinities_scaling_mode=expert_affinities_scaling_mode,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            use_packed_scales=cfg.use_packed_scales,
            top_k=cfg.top_k,
            ep_degree=cfg.ep_degree,
            quantization_type=cfg.quantization_type,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
        )
