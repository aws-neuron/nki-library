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

"""MXFP8 backward pass kernel entry point for blockwise MoE matrix multiplication."""

from typing import Optional

import nki
import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ....core.utils.kernel_assert import kernel_assert
from ...matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig
from ...mlp_mxfp8.common_utils import get_tile_sizes
from ...moe.bwd.moe_bwd_parameters import ActFnType, AffinityOption, ClampLimits, ShardOption, SkipMode
from ...mxfp_utils.mxfp8_utils.common_dataclasses import QuantScheme, SwizzleMode, TensorDescriptor
from .bwmm_bwd_dropless_mxfp8 import blockwise_mm_bwd_dropless_mxfp8
from .config import MXFP8MOEBwdConfig, TransposeMode


def _validate_kernel_options(
    config: MXFP8MOEBwdConfig,
    down_proj_act_checkpoint,
    bias: bool,
    clamp_limits,
    activation_type: ActFnType,
    run_with_lnc2: bool = True,
    gate_act_checkpoint_T: nl.ndarray = None,
    intermediate_checkpoint_T: nl.ndarray = None,
    scaled_intermediate_checkpoint_T: nl.ndarray = None,
    gate_up_weight_is_swizzled: nl.ndarray = None,
    down_weight_is_swizzled: nl.ndarray = None,
):
    """Validate kernel-level options against currently supported feature set.

    Anything not yet wired into the MXFP8 dropless impl is gated here so callers
    fail loudly instead of silently getting wrong gradients. TODOs flag the
    features still to be implemented.
    """
    # SHARD_ON_HIDDEN is only valid with AFFINITY_ON_I — the H-shard reduce-scatter
    # on B tiles assumes the gate/up function produces the per-token EA grad inline.
    if config.shard_option == ShardOption.SHARD_ON_HIDDEN:
        kernel_assert(
            config.affinity_option == AffinityOption.AFFINITY_ON_I,
            "SHARD_ON_HIDDEN only supports AFFINITY_ON_I",
        )

    """
    TODO: support AFFINITY_ON_H once the AFFINITY_ON_H code paths are wired
    into the MXFP8 dropless impl (down_proj_act_checkpoint consumption,
    separate EA-grad function, etc.).
    """
    kernel_assert(
        config.affinity_option == AffinityOption.AFFINITY_ON_I,
        "blockwise_mm_bwd_mxfp8 currently only supports AFFINITY_ON_I",
    )

    # TODO: support SHARD_ON_HIDDEN once the H-shard reduce-scatter and
    # per-core B-tile partitioning are wired into the MXFP8 dropless impl.
    kernel_assert(
        config.shard_option == ShardOption.SHARD_ON_FREE,
        "blockwise_mm_bwd_mxfp8 currently only supports SHARD_ON_FREE",
    )

    kernel_assert(
        clamp_limits != None,
        "clamp_limits object should not be None",
    )
    kernel_assert(
        activation_type == ActFnType.SiLU,
        "only ActFnType.SiLU is implemented in blockwise_mm_bwd_mxfp8",
    )

    # down_proj_act_checkpoint is required for AFFINITY_ON_H (used to compute d_affinity)
    # and must be None for AFFINITY_ON_I (EA grad is derived inline during Phase 2).
    if config.affinity_option == AffinityOption.AFFINITY_ON_I:
        kernel_assert(
            down_proj_act_checkpoint == None,
            "down_proj_act_checkpoint must be None for AFFINITY_ON_I",
        )
    else:
        kernel_assert(
            down_proj_act_checkpoint != None,
            "down_proj_act_checkpoint is required for AFFINITY_ON_H",
        )

    kernel_assert(gate_act_checkpoint_T == None, "gate_act_checkpoint_T is not currently supported")
    kernel_assert(intermediate_checkpoint_T == None, "intermediate_checkpoint_T is not currently supported")
    kernel_assert(
        scaled_intermediate_checkpoint_T == None, "scaled_intermediate_checkpoint_T is not currently supported"
    )
    kernel_assert(gate_up_weight_is_swizzled == False, "gate_up_weight_is_swizzled is not currently supported")
    kernel_assert(down_weight_is_swizzled == False, "down_weight_is_swizzled is not currently supported")
    kernel_assert(config.fp8_x4_dtype == float8_e4m3fn_x4, "Only E4M3 is tested, E5M2 works, but not tested")
    kernel_assert(run_with_lnc2 == True, "Kernel is expected to run only with LNC2")
    kernel_assert(config.compute_dtype == nl.bfloat16, "Only BF16 is supported, DGT does not support FP32")
    if config.single_expert_dense:
        kernel_assert(
            not config.accumulate_hidden_states_grad,
            "single_expert_dense requires accumulate_hidden_states_grad=False",
        )
        kernel_assert(not bias, "single_expert_dense does not currently support bias gradients")


def _validate_inputs_and_derive_dims(
    hidden_states,
    output_hidden_states_grad,
    gate_up_proj_act_checkpoint_T,
    gate_up_proj_weight,
    down_proj_weight,
    gate_up_weight_scales,
    down_proj_weight_scales,
    token_position_to_id,
    block_to_expert,
    expert_affinities_masked,
    block_size,
    num_shards,
    single_expert_dense=False,
):
    """Validate raw inputs and return derived dimensions as plain ints.

    NKI does not allow tensors inside dataclasses, so all checks operate on the
    raw nl.ndarray inputs directly and only ints are returned. Validation covers:
      - mandatory tensors are present
      - weight ranks (gate_up: 4D, down: 3D) and full shape consistency
      - activation shapes and dtypes (BF16/FP16 only — they cannot be MXFP8 since
        per-block indirect-DMA gather breaks 32-element quantization groups)
      - routing tensor shapes
      - contraction-dim divisibility (H, I_TP, B all multiples of L_TILE_K=512)
      - LNC sharding divisibility for both H (SHARD_ON_HIDDEN) and I_TP (SHARD_ON_FREE)

    Returns:
        (T, H, I_TP, E, N): derived ints used to allocate output buffers and
        thread through the dropless impl.
    """
    # Mandatory inputs must be present.
    required = (
        ("hidden_states", hidden_states),
        ("output_hidden_states_grad", output_hidden_states_grad),
        ("gate_up_proj_act_checkpoint_T", gate_up_proj_act_checkpoint_T),
        ("gate_up_proj_weight", gate_up_proj_weight),
        ("down_proj_weight", down_proj_weight),
        ("token_position_to_id", token_position_to_id),
        ("block_to_expert", block_to_expert),
        ("expert_affinities_masked", expert_affinities_masked),
    )
    for entry in required:
        name = entry[0]
        tensor = entry[1]
        kernel_assert(tensor != None, f"{name} is required")

    # Derive dimensions from raw shapes. hidden_states is always [T, H];
    # down_weight is always [E, I_TP, H]; N follows from token_position_to_id.
    gate_up_weight_quantized = gate_up_weight_scales is not None
    down_weight_quantized = down_proj_weight_scales is not None

    T = hidden_states.shape[0]
    H = hidden_states.shape[1]
    E = down_proj_weight.shape[0]
    I_TP = gate_up_proj_act_checkpoint_T.shape[2]
    N = token_position_to_id.shape[0] // block_size
    if single_expert_dense:
        kernel_assert(E == 1, f"single_expert_dense requires E=1, got E={E}")
        kernel_assert(T % block_size == 0, f"single_expert_dense requires T={T} divisible by block_size={block_size}")
        kernel_assert(
            N == T // block_size,
            f"single_expert_dense requires N=T/B={T // block_size}, got N={N}",
        )

    # TODO: support gate_up_proj_act_checkpoint_T=None by re-running the
    # gate_up_shape = gate_up_proj_weight.shape
    # gate/up forward matmul (hidden_states @ gate_up_proj_weight).

    kernel_assert(
        gate_up_proj_act_checkpoint_T != None,
        "gate_up_proj_act_checkpoint_T is currently required by blockwise_mm_bwd_mxfp8 — "
        "recompute of gate and up activations is not yet supported",
    )

    # gate_up_proj_act_checkpoint_T ranks + full shape match
    gate_up_proj_act_checkpoint_T_shape = gate_up_proj_act_checkpoint_T.shape

    kernel_assert(
        len(gate_up_proj_act_checkpoint_T_shape) == 4,
        f"gate_up_proj_act_checkpoint_T must be 4D [N, 2, I_TP, B], got rank {len(gate_up_proj_act_checkpoint_T_shape)}",
    )

    kernel_assert(
        gate_up_proj_act_checkpoint_T_shape == (N, 2, I_TP, block_size),
        f"gate_up_proj_act_checkpoint_T shape {tuple(gate_up_proj_act_checkpoint_T_shape)} must match [N={N}, 2, I_TP={I_TP}, B={block_size}]",
    )

    # Weight ranks + full shape match.
    if not gate_up_weight_quantized:
        # TODO: Add asserts for gate/up wt. scales.
        gate_up_shape = gate_up_proj_weight.shape
        kernel_assert(
            len(gate_up_shape) == 4,
            f"gate_up_weight must be 4D [E, H, 2, I_TP], got rank {len(gate_up_shape)}",
        )
        kernel_assert(
            gate_up_shape == (E, H, 2, I_TP),
            f"gate_up_weight shape {tuple(gate_up_shape)} must match [E={E}, H={H}, 2, I_TP={I_TP}]",
        )
    else:
        gate_up_shape = gate_up_proj_weight.shape
        kernel_assert(
            len(gate_up_shape) == 3,
            f"gate_up_weight must be 3D [E, 2*I_TP/4, H], got rank {len(gate_up_shape)}",
        )
        kernel_assert(
            gate_up_shape == (E, 2 * I_TP // 4, H),
            f"gate_up_weight shape {tuple(gate_up_shape)} must match [E={E}, 2*I_TP={2 * I_TP // 4}, H={H}]",
        )

    if not down_weight_quantized:
        # TODO: Add asserts for down wt. scales.
        down_shape = down_proj_weight.shape
        kernel_assert(
            len(down_shape) == 3,
            f"down_weight must be 3D [E, I_TP, H], got rank {len(down_shape)}",
        )
        kernel_assert(
            down_shape == (E, I_TP, H),
            f"down_weight shape {tuple(down_shape)} must match [E={E}, I_TP={I_TP}, H={H}]",
        )
    else:
        down_shape = down_proj_weight.shape
        kernel_assert(
            len(down_shape) == 3,
            f"down_weight must be 3D [E, H/4, I_TP] for x4 data, got rank {len(down_shape)}",
        )
        kernel_assert(
            down_shape == (E, H // 4, I_TP),
            f"down_weight shape {tuple(down_shape)} must match [E={E}, H/4={H // 4}, I_TP={I_TP}]",
        )

    # Activations: [T, H], BF16 or FP16 only (gathered per-block via indirect DMA).
    hs_shape = hidden_states.shape
    kernel_assert(
        len(hs_shape) == 2 and hs_shape == (T, H),
        f"hidden_states shape {tuple(hs_shape)} must match [T={T}, H={H}]",
    )
    kernel_assert(
        hidden_states.dtype in (nl.bfloat16, nl.float16),
        f"hidden_states dtype must be bfloat16 or float16, got {hidden_states.dtype}",
    )
    og_shape = output_hidden_states_grad.shape
    kernel_assert(
        len(og_shape) == 2 and og_shape == (T, H),
        f"output_hidden_states_grad shape {tuple(og_shape)} must match [T={T}, H={H}]",
    )
    kernel_assert(
        output_hidden_states_grad.dtype in (nl.bfloat16, nl.float16),
        f"output_hidden_states_grad dtype must be bfloat16 or float16, got {output_hidden_states_grad.dtype}",
    )

    """
    TODO: add dtype asserts for token_position_to_id (expected int32),
    block_to_expert (expected int32), and expert_affinities_masked
    (expected matching activation dtype).
    """

    # Routing tensor shapes.
    tpti_shape = token_position_to_id.shape
    kernel_assert(
        len(tpti_shape) == 1 and tpti_shape[0] == N * block_size,
        f"token_position_to_id shape {tuple(tpti_shape)} must be [N*B = {N * block_size}]",
    )
    bte_shape = block_to_expert.shape
    kernel_assert(
        len(bte_shape) == 2 and bte_shape == (N, 1),
        f"block_to_expert shape {tuple(bte_shape)} must match [N={N}, 1]",
    )
    ea_shape = expert_affinities_masked.shape
    kernel_assert(
        len(ea_shape) == 2 and ea_shape == (T * E, 1),
        f"expert_affinities_masked shape {tuple(ea_shape)} must match [T*E = {T * E}, 1]",
    )

    """
    Dimension alignment: H, I_TP, and block_size must be multiples of 128
    (the PE partition dimension / minimum tile size). The kernel handles
    non-512-aligned dimensions via partial-tile loading and remainder logic.

    Allowed block_size: 128, 256, 512, 1024, 2048, 4096. The 2048/4096 sizes were added
    (previously capped at 1024) so a dense run (E=1/TOP_K=1) can use larger blocks — up to a
    single dense block over all tokens when block_size == T — which matches the standalone
    matmul shapes. Blocks may exceed T because routing tensors pad unused positions.
    """
    kernel_assert(
        block_size in (128, 256, 512, 1024, 2048, 4096),
        f"block_size must be one of 128/256/512/1024/2048/4096, got {block_size}",
    )
    kernel_assert(H % 128 == 0, f"H={H} must be divisible by 128")
    kernel_assert(I_TP % 128 == 0, f"I_TP={I_TP} must be divisible by 128")

    # LNC sharding: SHARD_ON_HIDDEN splits H, SHARD_ON_FREE splits I_TP — both
    # must divide evenly across cores.
    kernel_assert(H % num_shards == 0, f"H={H} must be divisible by num_shards={num_shards}")
    kernel_assert(I_TP % num_shards == 0, f"I_TP={I_TP} must be divisible by num_shards={num_shards}")

    return T, H, I_TP, E, N


def _resolve_phase_configs(
    config,
    provided_phase_configs,
    phase_shapes,
    hidden_size,
    spill_reload,
    use_scale_packing,
    run_with_lnc2,
):
    """Resolve phase configs before kernel execution."""
    phase_configs = (
        config.phase1_config,
        config.phase2_config,
        config.phase3_config,
        config.phase4_config,
    )
    phase_names = ("phase1", "phase2", "phase3", "phase4")

    for phase_name, provided_config, phase_config, phase_shape in zip(
        phase_names,
        provided_phase_configs,
        phase_configs,
        phase_shapes,
    ):
        phase_m, phase_k, phase_n = phase_shape
        phase_config.M = phase_m
        phase_config.K = phase_k
        phase_config.N = phase_n

        # Wrapper defaults apply only to omitted phases.
        if provided_config is None:
            phase_config.spill_reload = spill_reload
            phase_config.enable_scale_packing = use_scale_packing

        phase_config.run_with_lnc2 = run_with_lnc2
        if phase_config.TILES_IN_BLOCK_M is None:
            phase_config.TILES_IN_BLOCK_M = 1
        if phase_config.TILES_IN_BLOCK_N is None:
            phase_config.TILES_IN_BLOCK_N = 1
        if phase_config.TILES_IN_BLOCK_K is None:
            phase_config.TILES_IN_BLOCK_K = 1
        if phase_config.TILES_IN_LOAD_M is None:
            phase_config.TILES_IN_LOAD_M = 1
        if phase_config.TILES_IN_LOAD_N is None:
            phase_config.TILES_IN_LOAD_N = 1

        # H=384 retains the original padded tile geometry.
        default_tiles = (
            get_tile_sizes(512, 512, 512) if hidden_size == 384 else get_tile_sizes(phase_k, phase_m, phase_n)
        )
        if phase_config.tile_m is None:
            phase_config.tile_m = default_tiles["tile_m"]
        if phase_config.tile_k is None:
            phase_config.tile_k = default_tiles["l_tile_k"]
        if phase_config.tile_n is None:
            phase_config.tile_n = default_tiles["tile_n"]

        kernel_assert(
            phase_config.quant_scheme == QuantScheme.WRAPX,
            f"{phase_name}: unsupported quant_scheme {phase_config.quant_scheme!r}; "
            "only QuantScheme.WRAPX is supported",
        )
        kernel_assert(
            phase_config.tile_k in (128, 256, 512),
            f"{phase_name}: tile_k must be 128, 256, or 512, got {phase_config.tile_k}",
        )


def blockwise_mm_bwd_mxfp8(
    # --- Required input tensors ---
    hidden_states: nl.ndarray,
    expert_affinities_masked: nl.ndarray,
    gate_up_proj_weight: nl.ndarray,
    down_proj_weight: nl.ndarray,
    token_position_to_id: nl.ndarray,
    block_to_expert: nl.ndarray,
    output_hidden_states_grad: nl.ndarray,
    block_size: int,
    # --- Optional pre-computed intermediate tensors ---
    gate_up_proj_act_checkpoint_T: Optional[nl.ndarray] = None,
    gate_act_checkpoint_T: nl.ndarray = None,
    intermediate_checkpoint_T: nl.ndarray = None,
    # Affinity I: gate_act * up * ea_scale (scaled intermediate for Phase 4 dW_down).
    # If None, recomputed per-block as intermediate * expert_affinity[token].
    scaled_intermediate_checkpoint_T: nl.ndarray = None,
    # Down projection activation checkpoint — required for AFFINITY_ON_H, must be None for AFFINITY_ON_I.
    down_proj_act_checkpoint: Optional[nl.ndarray] = None,
    # --- Optional pre-quantized weight support ---
    gate_up_weight_scales: nl.ndarray = None,
    gate_up_weight_is_swizzled: bool = False,
    down_weight_scales: nl.ndarray = None,
    down_weight_is_swizzled: bool = False,
    # --- Per-phase matmul configs (None = default TILES_IN_BLOCK_*=1) ---
    phase1_config: Optional[MatmulMxfp8KernelConfig] = None,
    phase2_config: Optional[MatmulMxfp8KernelConfig] = None,
    phase3_config: Optional[MatmulMxfp8KernelConfig] = None,
    phase4_config: Optional[MatmulMxfp8KernelConfig] = None,
    # --- MXFP8 configuration ---
    fp8_x4_dtype: type = float8_e4m3fn_x4,
    spill_reload: bool = False,
    use_scale_packing: bool = True,
    run_with_lnc2: bool = True,
    # --- Sharding & affinity placement ---
    shard_option: ShardOption = ShardOption.SHARD_ON_FREE,
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_I,
    # --- Compute / DMA / accumulation knobs (wired through to the dropless impl) ---
    compute_dtype: nki.dtype = nl.bfloat16,
    skip_dma: SkipMode = None,
    skip_grad_initialization: bool = False,
    single_expert_dense: bool = False,
    fast_dma_transpose: bool = False,
    # --- Tensor-local matmul layout conversion ---
    output_grad_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    down_weight_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    d_gate_up_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    gate_up_weight_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    d_gate_up_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    hidden_states_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    output_grad_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    scaled_intermediate_t_swizzle_mode: SwizzleMode = SwizzleMode.DGT,
    # Override the P3/P4 in-kernel transpose engine.
    phase3_transpose_mode: TransposeMode = TransposeMode.NC,
    phase4_transpose_mode: TransposeMode = TransposeMode.NC,
    accumulate_hidden_states_grad: bool = True,
    # --- Reserved API surface — accepted but not yet implemented in MXFP8 ---
    clamp_limits: ClampLimits = None,
    activation_type: ActFnType = ActFnType.SiLU,
    # --- Bias gradients (reserved API surface) ---
    bias: bool = False,
) -> tuple:
    """
    MXFP8 backward pass for blockwise Mixture of Experts.

    Computes gradients for all parameters in a Mixture of Experts layer using
    MXFP8 quantized matrix multiplication. Processes tokens in blocks assigned
    to specific experts.

    Only weights (gate_up_proj_weight, down_proj_weight) support pre-quantized
    MXFP8 inputs. Activations (hidden_states, output_hidden_states_grad) must be
    BF16. The default path gathers them via token indices; single_expert_dense
    reads contiguous block slices directly.

    TODO: Specify intended usage range (e.g., recommended T, H, I_TP, B, E ranges
    where this kernel is performance-optimized).

    Dimensions:
        T: Total number of input tokens (after linearizing across batch dimension)
        H: Hidden dimension size
        I_TP: Intermediate size / tensor parallel degree
        E: Number of experts
        B: Number of tokens per block (block_size)
        N: Total number of blocks ((T*TopK - (E-1) )/ B + E-1)

    Args:
        hidden_states (nl.ndarray): [T, H], Input hidden states (BF16) on HBM.
        expert_affinities_masked (nl.ndarray): [T * E, 1], Expert affinities on HBM.
        gate_up_proj_weight (nl.ndarray): [E, H, 2, I_TP], Gate/up projection weights on HBM.
        down_proj_weight (nl.ndarray): [E, I_TP, H], Down projection weights on HBM.
        token_position_to_id (nl.ndarray): [N * B], Token position to block mapping.
        block_to_expert (nl.ndarray): [N, 1], Expert index per block.
        output_hidden_states_grad (nl.ndarray): [T, H], Upstream gradient (BF16) from output.
        block_size (int): Number of tokens per block (128, 256, 512, or 1024).
        gate_up_proj_act_checkpoint_T (nl.ndarray, optional): [N, 2, I_TP, B], Checkpointed
            gate/up activations (gate_pre = checkpoint[block, 0], up = checkpoint[block, 1]).
            If None, gate_act_checkpoint_T and intermediate_checkpoint_T must be provided
            so the kernel can avoid recomputing from this checkpoint.
        gate_act_checkpoint_T (nl.ndarray, optional): [N, I_TP, B], Pre-computed SiLU(gate_pre).
            If None, recomputed per-block as SiLU(gate_up_proj_act_checkpoint_T[block, 0]).
        intermediate_checkpoint_T (nl.ndarray, optional): [N, I_TP, B], Pre-computed gate_act * up.
            If None, recomputed per-block as gate_act * up. Used for Phase 4 (dW_down).
        scaled_intermediate_checkpoint_T (nl.ndarray, optional): [N, I_TP, B], Pre-computed
            intermediate * expert_affinity (Affinity I mode), saved from the forward pass.
            If provided, Phase 4 reads its per-block slice directly as the dW_down RHS.
            If None, Phase 4 reuses Phase 1's scaled_intermediate (already EA-scaled
            under AFFINITY_ON_I) and transposes it inline — no separate recompute.
        down_proj_act_checkpoint (nl.ndarray, optional): [N, B, H], Pre-computed
            output_grad * expert_affinity (Affinity H mode). If None, recomputed per-block
            as output_grad[block] * ea_scale. Used for Phase 1 when affinity_option=AFFINITY_ON_H.
        gate_up_weight_scales (nl.ndarray, optional): MXFP8 scales for pre-quantized gate/up weights.
        gate_up_weight_is_swizzled (bool): Whether gate/up weights are pre-swizzled.
        down_weight_scales (nl.ndarray, optional): MXFP8 scales for pre-quantized down weights.
        down_weight_is_swizzled (bool): Whether down weights are pre-swizzled.
        phase1_config..phase4_config (MatmulMxfp8KernelConfig, optional): Per-phase
            matmul tiling, blocking, quantization, and spill/reload configuration.
        fp8_x4_dtype (type): MXFP8 packed data type (default: float8_e4m3fn_x4).
        spill_reload (bool): Spill/reload setting used by default phase configs.
        use_scale_packing (bool): Scale-packing setting used by default phase configs and inputs.
        run_with_lnc2 (bool): Whether to shard across 2 LNC cores.
        shard_option (ShardOption): LNC2 sharding strategy (default: SHARD_ON_FREE).
            SHARD_ON_HIDDEN requires affinity_option=AFFINITY_ON_I.
        affinity_option (AffinityOption): Where the expert affinity scalar is folded into
            the FFN chain. Must match the forward kernel's choice.
            AFFINITY_ON_H: requires down_proj_act_checkpoint.
            AFFINITY_ON_I: requires down_proj_act_checkpoint=None.
        compute_dtype (nki.dtype): Dtype for SBUF/HBM intermediates (default: bf16).
        skip_dma (SkipMode): OOB handling mode for indirect DMA operations.
        skip_grad_initialization (bool): If True, skip the zero-init of grad outputs.
        single_expert_dense (bool): Use direct contiguous block addressing for a
            single expert. Requires E=1, full blocks, and top-k=1 semantics.
        fast_dma_transpose (bool): Use direct 4D DMA gather-transpose addressing
            when swizzling unswizzled BF16 matmul operands. Default: False.
        *_swizzle_mode (SwizzleMode): Tensor-local DGT or PE conversion mode for
            each matmul operand descriptor.
        phase3_transpose_mode (TransposeMode): Transpose engine used by Phase 3.
        phase4_transpose_mode (TransposeMode): Transpose engine used by Phase 4.
        accumulate_hidden_states_grad (bool): If True (default), the Phase 2 hidden_states_grad
            scatter does a read-modify-write so multiple experts contributing to the same
            token (top-K > 1 routing) accumulate correctly. If False, the scatter overwrites
            — correct only when each token is touched by exactly one block (top-K = 1).
        clamp_limits (ClampLimits): Optional gradient clamping limits.
        activation_type (ActFnType): NOT YET IMPLEMENTED. SiLU is hardcoded in the
            MXFP8 dropless impl; passing a different activation will raise.
        bias (bool): Whether to compute bias gradients (default: False).

    Returns:
        tuple: Gradient tensors:
            - hidden_states_grad (nl.ndarray): [T, H], Gradient for hidden states.
            - expert_affinities_masked_grad (nl.ndarray): [T * E, 1], Gradient for affinities.
            - gate_up_proj_weight_grad (nl.ndarray): [E, H, 2, I_TP], Gradient for gate/up weights.
            - down_proj_weight_grad (nl.ndarray): [E, I_TP, H], Gradient for down weights.
            - gate_and_up_proj_bias_grad (nl.ndarray, optional): [E, 2, I_TP], if bias=True.
            - down_proj_bias_grad (nl.ndarray, optional): [E, H], if bias=True.

    Pseudocode:
        initialize_gradient_outputs()
        prefetch block_to_expert, token_indices[0]

        for block_idx in range(N):
            expert_idx = block_to_expert[block_idx]

            Phase 1: d_intermediate = output_grad[block] @ W_down[expert].T
                     SwiGLU_bwd(d_intermediate, checkpoint) → d_gate, d_up
                     compute affinity_grad (if AFFINITY_ON_H)

            Phase 2: hidden_states_grad[block] += d_gate_up @ W_gate_up[expert]
                     (scatter via token_position_to_id)

            Phase 3: dW_gate_up[expert] += d_gate_up.T @ hidden_states[block]

            Phase 4: dW_down[expert] += output_grad[block].T @ intermediate[block]
    """
    if skip_dma == None:
        skip_dma = SkipMode(False, False)
    if clamp_limits == None:
        clamp_limits = ClampLimits()

    kernel_assert(
        phase3_transpose_mode in (TransposeMode.NC, TransposeMode.DMA),
        f"Unsupported phase3_transpose_mode: {phase3_transpose_mode}",
    )
    kernel_assert(
        phase4_transpose_mode in (TransposeMode.NC, TransposeMode.DMA),
        f"Unsupported phase4_transpose_mode: {phase4_transpose_mode}",
    )
    for tensor_name, swizzle_mode in (
        ("output_grad", output_grad_swizzle_mode),
        ("down_weight", down_weight_swizzle_mode),
        ("d_gate_up", d_gate_up_swizzle_mode),
        ("gate_up_weight", gate_up_weight_swizzle_mode),
        ("d_gate_up_T", d_gate_up_t_swizzle_mode),
        ("hidden_states_T", hidden_states_t_swizzle_mode),
        ("output_grad_T", output_grad_t_swizzle_mode),
        ("scaled_intermediate_T", scaled_intermediate_t_swizzle_mode),
    ):
        kernel_assert(
            swizzle_mode in (SwizzleMode.DGT, SwizzleMode.PE),
            f"Unsupported {tensor_name}_swizzle_mode: {swizzle_mode}",
        )

    num_shards = nl.num_programs(axes=0) if run_with_lnc2 else 1
    T, H, I_TP, E, N = _validate_inputs_and_derive_dims(
        hidden_states=hidden_states,
        output_hidden_states_grad=output_hidden_states_grad,
        gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        gate_up_weight_scales=gate_up_weight_scales,
        down_proj_weight_scales=down_weight_scales,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
        block_size=block_size,
        num_shards=num_shards,
        single_expert_dense=single_expert_dense,
    )
    I_TP_PER_SHARD = I_TP // num_shards
    H_PER_SHARD = H // num_shards

    config = MXFP8MOEBwdConfig(
        compute_dtype=compute_dtype,
        fp8_x4_dtype=fp8_x4_dtype,
        activation_type=activation_type,
        shard_option=shard_option,
        affinity_option=affinity_option,
        skip_dma=skip_dma,
        skip_grad_initialization=skip_grad_initialization,
        single_expert_dense=single_expert_dense,
        fast_dma_transpose=fast_dma_transpose,
        accumulate_hidden_states_grad=accumulate_hidden_states_grad,
        clamp_limits=clamp_limits,
        phase1_config=phase1_config,
        phase2_config=phase2_config,
        phase3_config=phase3_config,
        phase4_config=phase4_config,
        bias=bias,
        phase3_transpose_mode=phase3_transpose_mode,
        phase4_transpose_mode=phase4_transpose_mode,
    )
    _resolve_phase_configs(
        config=config,
        provided_phase_configs=(phase1_config, phase2_config, phase3_config, phase4_config),
        phase_shapes=(
            (block_size, H, I_TP_PER_SHARD),
            (block_size, 2 * I_TP, H_PER_SHARD),
            (2 * I_TP if single_expert_dense else I_TP, block_size, H_PER_SHARD),
            (H_PER_SHARD, block_size, I_TP),
        ),
        hidden_size=H,
        spill_reload=spill_reload,
        use_scale_packing=use_scale_packing,
        run_with_lnc2=run_with_lnc2,
    )

    _validate_kernel_options(
        config=config,
        down_proj_act_checkpoint=down_proj_act_checkpoint,
        bias=bias,
        clamp_limits=clamp_limits,
        activation_type=activation_type,
        run_with_lnc2=run_with_lnc2,
        gate_act_checkpoint_T=gate_act_checkpoint_T,
        intermediate_checkpoint_T=intermediate_checkpoint_T,
        scaled_intermediate_checkpoint_T=scaled_intermediate_checkpoint_T,
        gate_up_weight_is_swizzled=gate_up_weight_is_swizzled,
        down_weight_is_swizzled=down_weight_is_swizzled,
    )

    # Per-phase blocking is resolved entirely by the CALLER: either an explicit phase_config
    # (fast, shape-tuned blocking — the caller looks it up, e.g. via get_shape_tuned_config in the
    # test harness), or None -> the TILES_IN_BLOCK_*=1 config default.
    # The kernel does NOT consult a tuning table; blocking is a caller concern.
    #
    # The P3/P4 transpose engine is an independent caller knob. NC uses nc_transpose on the
    # idle PE; DMA moves the [F,B] transpose onto the DMA engine.
    """
    Tensor-bearing descriptors cannot cross the kernel-entry boundary. Build
    them here from raw tensor arguments for use by internal helpers.
    """
    hidden_states_td = TensorDescriptor(
        data=hidden_states,
        swizzle_mode=hidden_states_t_swizzle_mode,
        fast_dma_transpose=fast_dma_transpose,
    )
    output_grad_td = TensorDescriptor(
        data=output_hidden_states_grad,
        swizzle_mode=output_grad_swizzle_mode,
        fast_dma_transpose=fast_dma_transpose,
    )

    gate_up_weight_quantized = gate_up_weight_scales is not None
    down_weight_quantized = down_weight_scales is not None

    """
    Phase 2 needs gate_up_weight as 2D for generic_matmul_mxfp8_api.
    Reshape [E, H, 2, I_TP] → [E*H, 2*I_TP]. Per-expert indexing via
    scalar_offset (indirect_dim=0, stride = H * 2 * I_TP). Phase 2 uses
    d_gate_up[B, 2*I_TP] @ W[E*H, 2*I_TP].T → [B, H], which computes
    d_gate @ W_gate.T + d_up @ W_up.T in a single matmul (K = 2*I_TP).
    """
    if not gate_up_weight_quantized:
        gate_up_weight_td = TensorDescriptor(
            data=gate_up_proj_weight.reshape((E * H, 2 * I_TP)),
            scales=gate_up_weight_scales,
            is_swizzled=gate_up_weight_is_swizzled,
            swizzle_mode=gate_up_weight_swizzle_mode,
            fast_dma_transpose=fast_dma_transpose,
        )
    else:
        # Pre-quantized: data [E, 2*I_TP//4, H] → 2D [E*2*I_TP//4, H]
        # Scales [E, K_per_expert, F_scales] → 2D [E*K_per_expert, F_scales]
        gate_up_scales_2d = gate_up_weight_scales.reshape(
            (E * gate_up_weight_scales.shape[1], gate_up_weight_scales.shape[2])
        )
        gate_up_weight_td = TensorDescriptor(
            data=gate_up_proj_weight.reshape((E * 2 * I_TP // 4, H)),
            scales=gate_up_scales_2d,
            is_swizzled=gate_up_weight_is_swizzled,
            scales_are_packed=use_scale_packing,
            swizzle_mode=gate_up_weight_swizzle_mode,
        )
    """
    Reshape from [E, I_TP, H] to a 2D [E*I_TP, H] view so the TD/matmul
    path (which assumes 2D data.shape) can parse it. The per-block
    expert offset is applied at runtime via TD.scalar_offset inside the
    block loop (see bwmm_bwd_dropless_mxfp8). Same underlying memory —
    the 3D output buffers are allocated separately and untouched here.
    """
    if not down_weight_quantized:
        down_weight_td = TensorDescriptor(
            data=down_proj_weight.reshape((E * I_TP, H)),
            scales=down_weight_scales,
            is_swizzled=down_weight_is_swizzled,
            swizzle_mode=down_weight_swizzle_mode,
            fast_dma_transpose=fast_dma_transpose,
        )
    else:
        # Pre-quantized: data [E, H//4, I_TP] → 2D [E*H//4, I_TP]
        # Scales [E, K_per_expert, F_scales] → 2D [E*K_per_expert, F_scales]
        down_scales_2d = down_weight_scales.reshape((E * down_weight_scales.shape[1], down_weight_scales.shape[2]))
        down_weight_td = TensorDescriptor(
            data=down_proj_weight.reshape((E * H // 4, I_TP)),
            scales=down_scales_2d,
            is_swizzled=down_weight_is_swizzled,
            scales_are_packed=use_scale_packing,
            swizzle_mode=down_weight_swizzle_mode,
        )

    token_position_to_id_td = TensorDescriptor(data=token_position_to_id)
    block_to_expert_td = TensorDescriptor(data=block_to_expert)
    expert_affinities_masked_td = TensorDescriptor(data=expert_affinities_masked)
    gate_up_proj_act_checkpoint_T_td = TensorDescriptor(data=gate_up_proj_act_checkpoint_T)
    gate_act_checkpoint_T_td = TensorDescriptor(data=gate_act_checkpoint_T) if gate_act_checkpoint_T != None else None
    intermediate_checkpoint_T_td = (
        TensorDescriptor(data=intermediate_checkpoint_T) if intermediate_checkpoint_T != None else None
    )
    scaled_intermediate_checkpoint_T_td = (
        TensorDescriptor(data=scaled_intermediate_checkpoint_T) if scaled_intermediate_checkpoint_T != None else None
    )
    down_proj_act_checkpoint_td = (
        TensorDescriptor(data=down_proj_act_checkpoint) if down_proj_act_checkpoint != None else None
    )

    # Allocate output gradient tensors
    hbm_buffer = nl.shared_hbm if run_with_lnc2 else nl.hbm

    hidden_states_grad = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=hbm_buffer)
    expert_affinities_masked_grad = nl.ndarray(
        expert_affinities_masked.shape, dtype=expert_affinities_masked.dtype, buffer=hbm_buffer
    )
    gate_up_proj_weight_grad = nl.ndarray((E, H, 2, I_TP), dtype=nl.bfloat16, buffer=hbm_buffer)
    down_proj_weight_grad = nl.ndarray((E, I_TP, H), dtype=nl.bfloat16, buffer=hbm_buffer)

    gate_and_up_proj_bias_grad = None
    down_proj_bias_grad = None
    if bias:
        gate_and_up_proj_bias_grad = nl.ndarray(shape=(E, 2, I_TP), dtype=compute_dtype, buffer=hbm_buffer)
        down_proj_bias_grad = nl.ndarray(shape=(E, H), dtype=compute_dtype, buffer=hbm_buffer)

    # TensorDescriptors are safe to pass after reconstruction inside the kernel.
    blockwise_mm_bwd_dropless_mxfp8(
        hidden_states_td=hidden_states_td,
        output_grad_td=output_grad_td,
        gate_up_weight_td=gate_up_weight_td,
        down_weight_td=down_weight_td,
        token_position_to_id_td=token_position_to_id_td,
        block_to_expert_td=block_to_expert_td,
        expert_affinities_masked_td=expert_affinities_masked_td,
        gate_up_proj_act_checkpoint_T_td=gate_up_proj_act_checkpoint_T_td,
        gate_act_checkpoint_T_td=gate_act_checkpoint_T_td,
        intermediate_checkpoint_T_td=intermediate_checkpoint_T_td,
        scaled_intermediate_checkpoint_T_td=scaled_intermediate_checkpoint_T_td,
        down_proj_act_checkpoint_td=down_proj_act_checkpoint_td,
        T=T,
        H=H,
        I_TP=I_TP,
        E=E,
        N=N,
        block_size=block_size,
        config=config,
        hidden_states_grad=hidden_states_grad,
        expert_affinities_masked_grad=expert_affinities_masked_grad,
        gate_up_proj_weight_grad=gate_up_proj_weight_grad,
        down_proj_weight_grad=down_proj_weight_grad,
        gate_and_up_proj_bias_grad=gate_and_up_proj_bias_grad,
        down_proj_bias_grad=down_proj_bias_grad,
        output_grad_swizzle_mode=output_grad_swizzle_mode,
        d_gate_up_swizzle_mode=d_gate_up_swizzle_mode,
        d_gate_up_t_swizzle_mode=d_gate_up_t_swizzle_mode,
        hidden_states_t_swizzle_mode=hidden_states_t_swizzle_mode,
        output_grad_t_swizzle_mode=output_grad_t_swizzle_mode,
        scaled_intermediate_t_swizzle_mode=scaled_intermediate_t_swizzle_mode,
    )

    if bias:
        return (
            hidden_states_grad,
            expert_affinities_masked_grad,
            gate_up_proj_weight_grad,
            down_proj_weight_grad,
            gate_and_up_proj_bias_grad,
            down_proj_bias_grad,
        )

    return (
        hidden_states_grad,
        expert_affinities_masked_grad,
        gate_up_proj_weight_grad,
        down_proj_weight_grad,
    )
