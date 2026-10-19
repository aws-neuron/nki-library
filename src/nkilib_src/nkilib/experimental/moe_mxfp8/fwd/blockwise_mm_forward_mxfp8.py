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

"""MXFP8 forward pass kernel entry point for blockwise MoE matrix multiplication.

Public wrapper for the training forward MoE FFN. Mirrors the backward wrapper
(``blockwise_mm_backward_mxfp8.py``): validates inputs, derives dims, builds the
2-D stacked-expert weight ``TensorDescriptor``s, allocates the output slabs and
the activation checkpoints, then delegates to the dropless impl. Returns the
layer output plus the checkpoints the backward consumes.
"""

import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ...moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ShardOption,
)
from ...mxfp_utils.mxfp8_utils._dyn_blocks import LNC_CORES
from ...mxfp_utils.mxfp8_utils.common_dataclasses import (
    QuantScheme,
    TensorDescriptor,
    TensorOrientation,
)
from .._config_cache import get_cached_moe_fwd_config
from .._config_helpers import (
    copy_config,
    input_shape_from_tensors,
    make_moe_config_key,
)
from ..config_helpers import ExecutionMode
from .bwmm_fwd_dropless_mxfp8 import blockwise_mm_fwd_dropless_mxfp8
from .config import MXFP8MOEFwdConfig, _auto_generate_moe_fwd_configs


def _weight_orientation(config_td):
    """Resolve an optional config descriptor to the BF16 input orientation."""
    if config_td is None:
        return TensorOrientation.F_BY_K
    return config_td.orientation


def _build_bf16_weight_td(
    data,
    config_td,
    f_by_k_shape,
    k_by_f_shape,
    f_by_k_effective_k_dim=None,
    shard_k_by_f=False,
):
    """Attach config metadata and reshape one BF16 weight for its declared orientation."""
    if config_td is None:
        config_td = TensorDescriptor()
    k_by_f = config_td.orientation == TensorOrientation.K_BY_F
    return TensorDescriptor(
        data=data.reshape(k_by_f_shape if k_by_f else f_by_k_shape),
        effective_k_dim=None if k_by_f else f_by_k_effective_k_dim,
        swizzle_mode=config_td.swizzle_mode,
        quant_scheme=config_td.quant_scheme,
        orientation=config_td.orientation,
        scales_are_packed=config_td.scales_are_packed,
        is_col_parallel_sharded=shard_k_by_f and k_by_f,
        disable_dma_transpose=config_td.disable_dma_transpose,
    )


def _validate_kernel_options(
    config: MXFP8MOEFwdConfig,
    gate_up_weight_scales=None,
    down_weight_scales=None,
    num_valid_tokens=None,
):
    """Gate features not yet wired into the MXFP8 forward dropless impl.

    Keeps the forward in lockstep with the backward's supported set: AFFINITY_ON_I,
    SiLU, E4M3, LNC2, BF16 compute. Sharding for the forward is SHARD_ON_BLOCK.
    """
    kernel_assert(
        config.affinity_option == AffinityOption.AFFINITY_ON_I,
        "blockwise_mm_fwd_mxfp8 currently only supports AFFINITY_ON_I",
    )

    # Pre-quantized weights arrive in the forward-natural x4 layout (gate_up
    # [E, 2*H/4, I_TP], down [E, I_TP/4, H]) plus their MX scales, so the on-chip
    # quantize path is skipped and the matmul consumes the weight directly. The x4
    # data must be swizzled with the SAME K permutation as the activation load mode:
    # WRAPX (the DGT / fast_dma_transpose default) or 1x32 (use_1x32_pe_swizzle). The
    # caller (weight producer) is responsible for matching the two; the dropless sets
    # quant_scheme=_1x32 on the weight TDs when use_1x32_pe_swizzle is on.
    kernel_assert(
        config.shard_option == ShardOption.SHARD_ON_BLOCK,
        "blockwise_mm_fwd_mxfp8 currently only supports SHARD_ON_BLOCK",
    )
    if config.execution_mode == ExecutionMode.MLP:
        kernel_assert(config.execution_mode.is_dense, "mlp_mode requires single_expert_dense=True")
        kernel_assert(num_valid_tokens is None, "mlp_mode does not support dynamic blocks")
    kernel_assert(
        config.activation_type == ActFnType.SiLU,
        "only ActFnType.SiLU is implemented in blockwise_mm_fwd_mxfp8",
    )
    kernel_assert(config.fp8_x4_dtype == float8_e4m3fn_x4, "Only E4M3 is tested, E5M2 works, but not tested")
    kernel_assert(config.compute_dtype == nl.bfloat16, "Only BF16 is supported, DGT does not support FP32")
    # Fast DMA transpose only supports the single-expert (E=1) case: the fast DGT
    # loader addresses the source directly and carries no per-expert offset, so it
    # is gated to the contiguous single_expert_dense path (which asserts E == 1).
    kernel_assert(
        not config.fast_dma_transpose or config.execution_mode.is_dense,
        "fast_dma_transpose only supports the E=1 case (requires single_expert_dense=True)",
    )
    # The 1x32 PE-swizzle loader addresses the source directly (no per-expert
    # offset), so like fast_dma_transpose it is gated to the contiguous E=1 path.
    kernel_assert(
        not config.use_1x32_pe_swizzle or config.execution_mode.is_dense,
        "use_1x32_pe_swizzle only supports the E=1 case (requires single_expert_dense=True)",
    )
    # Reusing one quantized weight copy across blocks is only correct when every
    # block uses the same expert. single_expert_dense asserts E == 1, making that a
    # compile-time fact; the routed path would need a runtime same-expert predicate.
    kernel_assert(
        not config.reuse_spilled_weights or config.execution_mode.is_dense,
        "reuse_spilled_weights requires single_expert_dense=True (E=1)",
    )
    # The dynamic block loop indexes every per-block address at runtime via
    # scalar_offset on a block-major view, which the routed path's indirect token
    # gather/scatter does not go through. Dense-only until that is wired.
    kernel_assert(
        num_valid_tokens is None or config.execution_mode.is_dense,
        "num_valid_tokens (hardware dynamic block loop) requires single_expert_dense=True (E=1)",
    )


def _validate_inputs_and_derive_dims(
    hidden_states,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    expert_affinities_masked,
    block_size,
    single_expert_dense=False,
    gate_up_weight_scales=None,
    down_weight_scales=None,
    gate_up_weight_orientation=TensorOrientation.F_BY_K,
    down_weight_orientation=TensorOrientation.F_BY_K,
    num_valid_tokens=None,
    mlp_mode=False,
):
    """Validate raw inputs and return derived dims (T, H, I_TP, E, N) as plain ints.

    Mirrors the backward's validator but on the forward's input set (no
    output_hidden_states_grad, no checkpoints as inputs). Activations are BF16
    only (gathered per-block via indirect DMA, which breaks MXFP8 quant groups).
    Weights are either unswizzled BF16 or pre-quantized MXFP8 x4 (when the
    matching ``*_weight_scales`` are provided).
    """
    required = (
        ("hidden_states", hidden_states),
        ("gate_up_proj_weight", gate_up_proj_weight),
        ("down_proj_weight", down_proj_weight),
    )
    if not mlp_mode:
        required += (
            ("token_position_to_id", token_position_to_id),
            ("block_to_expert", block_to_expert),
            ("expert_affinities_masked", expert_affinities_masked),
        )
    for entry in required:
        name = entry[0]
        tensor = entry[1]
        kernel_assert(tensor != None, f"{name} is required")

    gate_up_weight_quantized = gate_up_weight_scales is not None
    down_weight_quantized = down_weight_scales is not None

    T = hidden_states.shape[0]
    H = hidden_states.shape[1]
    E = down_proj_weight.shape[0]
    # Both GEMMs contract over the input dim (gate/up over H, down over I_TP).
    # Raw BF16 accepts either forward-natural F_BY_K or backward-natural K_BY_F.
    # Pre-quantized x4 keeps its fixed forward-kernel layout:
    #   BF16 F_BY_K: gate_up [E, I_TP, 2, H]  down [E, H, I_TP]
    #   BF16 K_BY_F: gate_up [E, H, 2, I_TP]  down [E, I_TP, H]
    #   x4:          gate_up [E, 2*H/4, I_TP] down [E, I_TP/4, H]
    # gate_up packs the "2" (gate/up) into the K axis: per intermediate channel the 2*H
    # columns are [gate(H), up(H)], so gate/up is a K-slice (rhs_k_offset=0/H). H is read
    # from hidden_states ([T, H]); I_TP from the down weight (its trailing dim for BF16,
    # or dim-1 * 4 for the x4 K-by-F layout — down is unchanged by the K-slice migration).
    if down_weight_quantized:
        I_TP = down_proj_weight.shape[1] * 4
    elif down_weight_orientation == TensorOrientation.K_BY_F:
        I_TP = down_proj_weight.shape[1]
    else:
        I_TP = down_proj_weight.shape[2]
    N = div_ceil(T, block_size) if single_expert_dense else token_position_to_id.shape[0] // block_size

    # Weight ranks + full shapes, per orientation.
    gate_up_shape = gate_up_proj_weight.shape
    if not gate_up_weight_quantized:
        if gate_up_weight_orientation == TensorOrientation.K_BY_F:
            kernel_assert(
                len(gate_up_shape) == 4 and gate_up_shape == (E, H, 2, I_TP),
                f"gate_up_weight shape {tuple(gate_up_shape)} must match backward-natural "
                f"[E={E}, H={H}, 2, I_TP={I_TP}] (gate_up_weight_td.orientation=K_BY_F)",
            )
        else:
            kernel_assert(
                len(gate_up_shape) == 4 and gate_up_shape == (E, I_TP, 2, H),
                f"gate_up_weight shape {tuple(gate_up_shape)} must match "
                f"[E={E}, I_TP={I_TP}, 2, H={H}] (gate_up_weight_td.orientation=F_BY_K)",
            )
    else:
        kernel_assert(
            len(gate_up_shape) == 3 and gate_up_shape == (E, 2 * H // 4, I_TP),
            f"pre-quantized gate_up_weight shape {tuple(gate_up_shape)} must match "
            f"[E={E}, 2*H/4={2 * H // 4}, I_TP={I_TP}]",
        )

    down_shape = down_proj_weight.shape
    if not down_weight_quantized:
        if down_weight_orientation == TensorOrientation.K_BY_F:
            kernel_assert(
                len(down_shape) == 3 and down_shape == (E, I_TP, H),
                f"down_weight shape {tuple(down_shape)} must match backward-natural "
                f"[E={E}, I_TP={I_TP}, H={H}] (down_weight_td.orientation=K_BY_F)",
            )
        else:
            kernel_assert(
                len(down_shape) == 3 and down_shape == (E, H, I_TP),
                f"down_weight shape {tuple(down_shape)} must match "
                f"[E={E}, H={H}, I_TP={I_TP}] (down_weight_td.orientation=F_BY_K)",
            )
    else:
        kernel_assert(
            len(down_shape) == 3 and down_shape == (E, I_TP // 4, H),
            f"pre-quantized down_weight shape {tuple(down_shape)} must match [E={E}, I_TP/4={I_TP // 4}, H={H}]",
        )

    # Activations: [T, H], BF16/FP16 only.
    hs_shape = hidden_states.shape
    kernel_assert(
        len(hs_shape) == 2 and hs_shape == (T, H),
        f"hidden_states shape {tuple(hs_shape)} must match [T={T}, H={H}]",
    )
    kernel_assert(
        hidden_states.dtype in [nl.bfloat16],
        f"hidden_states dtype must be bfloat16, got {hidden_states.dtype}",
    )

    # Routing tensor shapes. In no-indirect mode the framework has already
    # packed one expert's tokens, so routing tensors are single-entry dummies.
    if not mlp_mode:
        tpti_shape = token_position_to_id.shape
        bte_shape = block_to_expert.shape
        if single_expert_dense:
            kernel_assert(
                len(tpti_shape) == 1 and tpti_shape[0] == 1,
                f"single_expert_dense expects dummy token_position_to_id shape [1], got {tuple(tpti_shape)}",
            )
            kernel_assert(
                len(bte_shape) == 2 and bte_shape == (1, 1),
                f"single_expert_dense expects dummy block_to_expert shape [1, 1], got {tuple(bte_shape)}",
            )
        else:
            kernel_assert(
                len(tpti_shape) == 1 and tpti_shape[0] == N * block_size,
                f"token_position_to_id shape {tuple(tpti_shape)} must be [N*B = {N * block_size}]",
            )
            kernel_assert(
                len(bte_shape) == 2 and bte_shape == (N, 1),
                f"block_to_expert shape {tuple(bte_shape)} must match [N={N}, 1]",
            )
        ea_shape = expert_affinities_masked.shape
        kernel_assert(
            len(ea_shape) == 2 and ea_shape == (T * E, 1),
            f"expert_affinities_masked shape {tuple(ea_shape)} must match [T*E = {T * E}, 1]",
        )

    # Dimension alignment.
    if mlp_mode:
        kernel_assert(
            N == 2 and block_size % 128 == 0,
            "MLP requires two tile-aligned sequence blocks",
        )
    else:
        kernel_assert(
            block_size in (128, 256, 512, 1024, 2048, 4096),
            f"block_size must be 128, 256, 512, 1024, 2048, or 4096, got {block_size}",
        )
    kernel_assert(H % 128 == 0, f"H={H} must be divisible by 128")
    kernel_assert(I_TP % 128 == 0, f"I_TP={I_TP} must be divisible by 128")
    # The static SHARD_ON_BLOCK path distributes whole blocks across cores, so N=1
    # leaves one core idle. The dynamic scheduler can instead split its odd static
    # prefix block across both cores, which makes the single-block dense case valid.
    dynamic_single_block = N == 1 and num_valid_tokens is not None and single_expert_dense
    kernel_assert(
        N >= LNC_CORES or dynamic_single_block,
        f"SHARD_ON_BLOCK requires N>={LNC_CORES} so each core owns a block, "
        f"got N={N} (T={T}, block_size={block_size}). "
        f"Single-block requires num_valid_tokens and single_expert_dense.",
    )
    if single_expert_dense:
        kernel_assert(E == 1, f"single_expert_dense requires exactly one expert weight, got E={E}")
        kernel_assert(T % block_size == 0, "single_expert_dense requires T to be divisible by block_size")

    if num_valid_tokens is not None:
        kernel_assert(
            len(num_valid_tokens.shape) == 1 and num_valid_tokens.shape[0] == 1,
            f"num_valid_tokens shape {tuple(num_valid_tokens.shape)} must be [1]",
        )
        kernel_assert(
            num_valid_tokens.dtype == nl.int32,
            f"num_valid_tokens dtype must be int32, got {num_valid_tokens.dtype}",
        )

    return T, H, I_TP, E, N


def blockwise_mm_fwd_mxfp8(
    # --- Required input tensors ---
    hidden_states: nl.NkiTensor,
    expert_affinities_masked: nl.NkiTensor,
    gate_up_proj_weight: nl.NkiTensor,
    down_proj_weight: nl.NkiTensor,
    token_position_to_id: nl.NkiTensor,
    block_to_expert: nl.NkiTensor,
    block_size: int = 4096,
    # --- Fused kernel configuration (all non-tensor knobs live here) ---
    config: MXFP8MOEFwdConfig = None,
    # --- Optional pre-quantized weight scales (enables the MXFP8 x4 weight path) ---
    gate_up_weight_scales: nl.NkiTensor = None,
    down_weight_scales: nl.NkiTensor = None,
    # --- Optional hardware dynamic block loop (mirrors the bf16 hybrid kernel) ---
    num_valid_tokens: nl.NkiTensor = None,
) -> tuple:
    """MXFP8 forward pass for blockwise (dropless) Mixture of Experts.

    Computes the MoE FFN output and emits the activation checkpoints the MXFP8
    MoE backward (``blockwise_mm_bwd_mxfp8``) consumes, so fwd + bwd form a
    validated training pair. Tokens are processed in fixed-size blocks already
    assigned to a single expert each by an upstream router; this kernel never
    computes routing.

    Only weights support pre-quantized MXFP8 inputs. Activations (hidden_states)
    must be BF16 because they are gathered per-block via indirect DMA, which would
    break MXFP8 32-element quantization groups. When single_expert_dense is True,
    hidden_states must already contain block-aligned tokens for one expert and
    both weight tensors must have E=1.

    Dimensions:
        T: total tokens (linearized across batch)
        H: hidden dimension
        I_TP: intermediate size / tensor-parallel degree
        E: number of experts
        B: tokens per block (block_size)
        N: total blocks ((T*top_k - (E-1)) / B + (E-1))

    Args:
        hidden_states (nl.NkiTensor): [T, H], input hidden states (BF16) on HBM.
        expert_affinities_masked (nl.NkiTensor): [T*E, 1], expert affinities (fp32) on HBM.
        gate_up_proj_weight (nl.NkiTensor): Gate/up weights on HBM. BF16 defaults
            to forward-natural [E, I_TP, 2, H], reinterpreted [E, I_TP, 2*H] (F-by-K,
            F=I_TP, K=2*H): per intermediate channel the 2*H columns are [gate(H), up(H)]
            contiguous (the transpose of the backward's [E, H, 2, I_TP]). The forward
            GEMMs contract over H, so gate/up is a K-slice of the 2*H weight (gate K in
            [0,H), up K in [H,2H)) via rhs_k_offset while the hidden LHS stays H-long.
            Setting config.gate_up_weight_td.orientation=K_BY_F accepts the
            backward-natural BF16 layout [E, H, 2, I_TP] instead; gate/up are then
            selected as free-axis slices.
            When gate_up_weight_scales is provided the weight is pre-quantized MXFP8 x4
            in the K-by-F layout [E, 2*H/4, I_TP] (K=2*H packed by 4 on dim-1).
        down_proj_weight (nl.NkiTensor): Down weights on HBM. BF16 defaults to
            forward-natural [E, H, I_TP] (F-by-K, transpose of the backward's
            [E, I_TP, H]); setting config.down_weight_td.orientation=K_BY_F accepts
            backward-natural [E, I_TP, H]. The two weight orientations are independent.
            The down GEMM contracts over I_TP. When down_weight_scales
            is provided the weight is pre-quantized MXFP8 x4 in the K-by-F layout
            [E, I_TP/4, H] (K=I_TP packed by 4 on dim-1).
        token_position_to_id (nl.NkiTensor): [N*B] int32, token -> block-position map
            (pad id = -1 under skip_dma). Use a dummy [1] tensor when
            single_expert_dense is True.
        block_to_expert (nl.NkiTensor): [N, 1] int32, expert index per block. Use
            a dummy [1, 1] tensor when single_expert_dense is True.
        num_valid_tokens (nl.NkiTensor, optional): [1] int32, how many leading rows of
            hidden_states hold real tokens. Enables the hardware dynamic block loop.
            Real tokens MUST be the leading rows. Values above T are clamped to T.
            Requires single_expert_dense. Output rows and checkpoint slots past the
            effective count are UNDEFINED; consumers must ignore that region, and the
            backward must be given the same count.
        block_size (int): tokens per block (128/256/512/1024/2048/4096).
            MLP wrappers pass T/2 so compute and checkpoint use two
            blocks, one per core.
        config (MXFP8MOEFwdConfig, optional): fused forward configuration carrying every
            non-tensor knob — compute/quant dtypes, activation, sharding + affinity
            placement, the two per-phase matmul configs (gate_up/down), the checkpoint
            emission config, clamp limits, skip-DMA mode, and the load-mode / spill /
            scale-packing / LNC2 flags. When None a default ``MXFP8MOEFwdConfig()`` is
            used, so the framework can call the kernel with only the tensors +
            ``block_size``. See ``MXFP8MOEFwdConfig`` for per-field docs and defaults.
        gate_up_weight_scales (nl.NkiTensor, optional): MXFP8 scales for pre-quantized
            gate/up weights. Passing scales auto-selects the pre-quantized path:
            gate_up_proj_weight is consumed as MXFP8 x4 in the K-by-F layout
            [E, 2*H/4, I_TP] and the on-chip quantize is skipped; when None,
            gate_up_proj_weight is unswizzled BF16 in the orientation selected by
            config.gate_up_weight_td. The x4 swizzle scheme
            is taken from config.gate_up_weight_td.quant_scheme (the source of truth); left
            unset it defaults to 1x32 PE-swizzle on the single_expert_dense (E=1) path and
            WRAPX/DGT on the routed path, with scale packing on. Set
            config.gate_up_weight_td=TensorDescriptor(quant_scheme=QuantScheme.WRAPX) to
            force WRAPX. The offline weight swizzle must match this scheme.
        down_weight_scales (nl.NkiTensor, optional): MXFP8 scales for pre-quantized down
            weights. When provided, down_proj_weight is consumed as MXFP8 x4 in the
            K-by-F layout [E, I_TP/4, H] (same scheme resolution as gate/up); when None,
            it is unswizzled BF16 in the orientation selected by
            config.down_weight_td.

    Returns:
        tuple:
            - output_hidden_states (nl.NkiTensor): [T, H] MoE FFN output.
          followed by the gate/up activation checkpoint, present only when
          config.checkpoint_config.save_gate_up_proj_act is set:
            - gate_up_proj_act_checkpoint_T (nl.NkiTensor): clamped gate pre-activation
              at [block, 0] and up at [block, 1], stored token-major as
              [N, 2, B, I_TP].
    """
    config = MXFP8MOEFwdConfig() if config == None else copy_config(config)
    shape = input_shape_from_tensors(
        hidden_states,
        gate_up_proj_weight,
        down_proj_weight,
        block_size,
        config.execution_mode,
        token_position_to_id,
        block_to_expert,
        gate_up_weight_scales,
        down_weight_scales,
        num_valid_tokens=num_valid_tokens,
    )
    key = make_moe_config_key(
        shape,
        (gate_up_proj_weight.dtype, down_proj_weight.dtype),
        config,
        False,
    )
    cached_config = get_cached_moe_fwd_config(key, config)
    if cached_config != None:
        config = cached_config

    # single_expert_dense (E=1, top_k=1) writes disjoint output rows directly, so the
    # per-block output scatter never accumulates; force accumulation off to match the
    # direct path regardless of what the caller set.
    if config.execution_mode.is_dense:
        config.is_tensor_update_accumulating = False

    # Resolve optional weight descriptors once. Their orientation governs BF16
    # shapes; their quant_scheme governs the pre-quantized load path.
    gate_up_cfg_td = config.gate_up_weight_td
    down_cfg_td = config.down_weight_td
    # MLP's shared BF16 layout is a tensor fact, resolved after matching caller presets.
    if config.execution_mode == ExecutionMode.MLP:
        if gate_up_cfg_td == None and gate_up_weight_scales == None:
            gate_up_cfg_td = TensorDescriptor(orientation=TensorOrientation.K_BY_F)
        if down_cfg_td == None and down_weight_scales == None:
            down_cfg_td = TensorDescriptor(orientation=TensorOrientation.K_BY_F)

    # Auto-detect pre-quantized weights (scales present) and resolve the weight load
    # scheme from the weight TD, which is the source of truth. When the caller left the
    # weight TD unset (None) the default is 1x32 PE-swizzle on the single_expert_dense
    # (E=1) path and WRAPX/DGT on the routed path (1x32 is E=1-only); scale packing is on
    # by default (config.*_config.enable_scale_packing). A caller-supplied weight TD's
    # quant_scheme overrides it (e.g. TensorDescriptor(quant_scheme=WRAPX) forces WRAPX).
    # The resolved scheme is folded into use_1x32_pe_swizzle so the dropless orchestrator
    # drives the weight AND the on-the-fly activation load with one consistent scheme;
    # this runs before _validate_kernel_options so the routed default clears the
    # "use_1x32 requires single_expert_dense" assert without the caller touching the bool.
    prequantized = gate_up_weight_scales is not None or down_weight_scales is not None
    if prequantized:
        gu_td = config.gate_up_weight_td
        dn_td = config.down_weight_td
        # gate/up and down share a single x4 swizzle scheme: it drives the one on-the-fly
        # activation load, which cannot straddle two schemes. If the caller supplies both
        # weight TDs their quant_scheme must agree.
        kernel_assert(
            gu_td is None or dn_td is None or gu_td.quant_scheme == dn_td.quant_scheme,
            "gate_up and down weight quant_scheme must match, got "
            f"{gu_td.quant_scheme if gu_td is not None else None} (gate_up) vs "
            f"{dn_td.quant_scheme if dn_td is not None else None} (down)",
        )
        default_scheme = QuantScheme._1x32 if config.execution_mode.is_dense else QuantScheme.WRAPX
        weight_scheme = default_scheme
        if gu_td is not None:
            weight_scheme = gu_td.quant_scheme
        elif dn_td is not None:
            weight_scheme = dn_td.quant_scheme
        config.use_1x32_pe_swizzle = weight_scheme == QuantScheme._1x32

    config.resolve_phase_defaults()
    _validate_kernel_options(
        config=config,
        gate_up_weight_scales=gate_up_weight_scales,
        down_weight_scales=down_weight_scales,
        num_valid_tokens=num_valid_tokens,
    )

    # LNC2 sharding is a launch fact, not a config knob: read the shard count from
    # the grid. The kernel is only supported under LNC2 (num_shards == 2).
    num_shards = nl.num_programs(axes=0)
    kernel_assert(num_shards == LNC_CORES, f"Kernel is LNC{LNC_CORES}-only, got {num_shards} cores")
    T, H, I_TP, E, N = _validate_inputs_and_derive_dims(
        hidden_states=hidden_states,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
        block_size=block_size,
        single_expert_dense=config.execution_mode.is_dense,
        gate_up_weight_scales=gate_up_weight_scales,
        down_weight_scales=down_weight_scales,
        gate_up_weight_orientation=_weight_orientation(gate_up_cfg_td),
        down_weight_orientation=_weight_orientation(down_cfg_td),
        num_valid_tokens=num_valid_tokens,
        mlp_mode=(config.execution_mode == ExecutionMode.MLP),
    )

    # The shared MXFP8 matmul mis-computes a pre-quantized GEMM that is LOGICALLY SQUARE
    # (contraction K == free/output N) on the mixed BF16-LHS x x4-RHS path. I_TP == H makes
    # BOTH forward GEMMs square (gate/up: K=H, N=I_TP; down: K=I_TP, N=H), so the gate/up
    # result is already wrong (its checkpoint fails first) -- this is NOT a down-phase issue,
    # and it is independent of single_expert_dense, the gate/up weight slicing, tiling, and the
    # load mode. Fail loudly until the shared matmul's square-K==N pre-quant path is fixed; real
    # MoE shapes never have hidden == intermediate/TP (Qwen3 H=4096, I_TP in {1536, 768}).
    # TODO(matmul-mxfp8): fix the K==N pre-quantized matmul + remove this guard.
    kernel_assert(
        not ((gate_up_weight_scales is not None or down_weight_scales is not None) and I_TP == H),
        f"pre-quantized weights with I_TP == H ({I_TP}) are not yet supported: the shared MXFP8 "
        "matmul is incorrect for a logically-square (K == N) pre-quantized GEMM; real MoE shapes "
        "have hidden != intermediate/TP",
    )

    # Auto-generate the per-phase matmul configs from the derived dims. The
    # forward drives generic_matmul_mxfp8_api directly (not the generic matmul
    # kernel entry point), so its configs never otherwise pass through
    # auto_generate_default — this fills the derived fields (tile sizes,
    # TILES_IN_LOAD_M/N) the dropless impl feeds into the matmul calls.
    _auto_generate_moe_fwd_configs(config, block_size=block_size, H=H, I_TP=I_TP)

    # Build data-bearing TensorDescriptors locally (passed flat into the dropless
    # impl). Weights are either unswizzled BF16 (F-by-K) or pre-quantized MXFP8 x4
    # (K-by-F, /4); the dropless impl branches on ``weight_td.scales is None`` and the
    # matmul API skips the on-chip quantize/spill for an already-quantized operand.
    # For the BF16 path the layout facts (swizzle_mode / quant_scheme / orientation /
    # scales_are_packed) come from the caller's config weight TDs (a default TensorDescriptor
    # when unset), so a caller-supplied swizzle wins; when left at the defaults the
    # orchestrator's fast_dma_transpose / use_1x32_pe_swizzle bools still apply. For the x4
    # path those layout facts are auto-resolved by TensorDescriptor (quantized => is_swizzled,
    # K_BY_F orientation, packed-scales set from the phase config) and the orchestrator applies
    # the WRAPX/1x32 swizzle picked by the weight-TD-derived use_1x32_pe_swizzle above.
    hidden_states_td = TensorDescriptor(data=hidden_states)

    gate_up_weight_quantized = gate_up_weight_scales is not None
    down_weight_quantized = down_weight_scales is not None

    # Build the two data-bearing descriptors through one orientation-processing
    # helper. For F_BY_K gate/up, [gate|up] is a K-slice and effective_k_dim=H
    # caps each half. For K_BY_F it is an N-slice, so the stored K is already H;
    # sharding the doubled free dimension presents one I_TP-wide half to the GEMM.
    if not gate_up_weight_quantized:
        gate_up_weight_td = _build_bf16_weight_td(
            data=gate_up_proj_weight,
            config_td=gate_up_cfg_td,
            f_by_k_shape=(E * I_TP, 2 * H),
            k_by_f_shape=(E * H, 2 * I_TP),
            f_by_k_effective_k_dim=H,
            shard_k_by_f=True,
        )
    else:
        # x4 [E, 2*H/4, I_TP] -> 2D [E*2*H/4, I_TP] (K-by-F, K/4=2*H/4, F=I_TP).
        # Dense scales [E, K // 32, F] -> 2D [E * K // 32, F].
        gate_up_scales_2d = gate_up_weight_scales.reshape(
            (E * gate_up_weight_scales.shape[1], gate_up_weight_scales.shape[2])
        )
        gate_up_weight_td = TensorDescriptor(
            data=gate_up_proj_weight.reshape((E * (2 * H // 4), I_TP)),
            scales=gate_up_scales_2d,
            effective_k_dim=H,
        )

    if not down_weight_quantized:
        down_weight_td = _build_bf16_weight_td(
            data=down_proj_weight,
            config_td=down_cfg_td,
            f_by_k_shape=(E * H, I_TP),
            k_by_f_shape=(E * I_TP, H),
        )
    else:
        # x4 [E, I_TP/4, H] -> 2D [E*I_TP/4, H] (K-by-F, K/4=I_TP/4, F=H).
        # Dense scales [E, K // 32, F] -> 2D [E * K // 32, F].
        down_scales_2d = down_weight_scales.reshape((E * down_weight_scales.shape[1], down_weight_scales.shape[2]))
        down_weight_td = TensorDescriptor(
            data=down_proj_weight.reshape((E * (I_TP // 4), H)),
            scales=down_scales_2d,
        )

    if not config.dma_transpose_enabled:
        for operand in (hidden_states_td, gate_up_weight_td, down_weight_td):
            operand.disable_dma_transpose = True

    if config.execution_mode == ExecutionMode.MLP:
        token_position_to_id_td = None
        block_to_expert_td = None
        expert_affinities_masked_td = None
    else:
        token_position_to_id_td = TensorDescriptor(data=token_position_to_id)
        block_to_expert_td = TensorDescriptor(data=block_to_expert)
        expert_affinities_masked_td = TensorDescriptor(data=expert_affinities_masked)
    num_valid_tokens_td = TensorDescriptor(data=num_valid_tokens) if num_valid_tokens is not None else None

    # Allocate outputs. SHARD_ON_BLOCK gives each core its own scratch output
    # slab (output_slabs[shard_id]); the per-shard slabs are summed into the
    # returned [T, H] output by the final reduce. Separate slabs avoid the
    # cross-core write race when top_k > 1 routes a token's blocks onto
    # different cores. Checkpoints are written per-block, so no cross-core
    # aliasing there (each core owns a disjoint block subset).
    #
    # single_expert_dense (E=1, top_k=1) writes disjoint output rows per core
    # directly into output_hidden_states, so the scratch slabs are unnecessary
    # (skips their alloc + zero-init + reduce in the dropless impl).
    checkpoint_config = config.checkpoint_config
    hbm_buffer = nl.shared_hbm
    output_hidden_states = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=hbm_buffer)
    output_slabs = (
        None
        if config.execution_mode.is_dense
        else nl.ndarray((LNC_CORES, T, H), dtype=hidden_states.dtype, buffer=hbm_buffer)
    )
    # Allocate the gate/up checkpoint only when its save flag is set; a disabled
    # checkpoint is passed to the impl as None so it skips the store.
    gate_up_proj_act_checkpoint_T = None
    if checkpoint_config.save_gate_up_proj_act:
        gate_up_shape = (N, 2, block_size, I_TP)
        gate_up_proj_act_checkpoint_T = nl.ndarray(gate_up_shape, dtype=config.compute_dtype, buffer=hbm_buffer)

    blockwise_mm_fwd_dropless_mxfp8(
        hidden_states_td=hidden_states_td,
        gate_up_weight_td=gate_up_weight_td,
        down_weight_td=down_weight_td,
        token_position_to_id_td=token_position_to_id_td,
        block_to_expert_td=block_to_expert_td,
        expert_affinities_masked_td=expert_affinities_masked_td,
        T=T,
        H=H,
        I_TP=I_TP,
        E=E,
        N=N,
        block_size=block_size,
        config=config,
        output_hidden_states=output_hidden_states,
        output_slabs=output_slabs,
        gate_up_proj_act_checkpoint_T=gate_up_proj_act_checkpoint_T,
        num_valid_tokens_td=num_valid_tokens_td,
    )

    # Return arity tracks the save flag (mirrors the backward's bias-conditional
    # return): the gate/up checkpoint is appended only when saved, so the test
    # framework's positional output mapping never sees a None. Order is fixed: output
    # first, then the gate/up activation checkpoint.
    outputs = [output_hidden_states]
    if checkpoint_config.save_gate_up_proj_act:
        outputs.append(gate_up_proj_act_checkpoint_T)
    return tuple(outputs)
