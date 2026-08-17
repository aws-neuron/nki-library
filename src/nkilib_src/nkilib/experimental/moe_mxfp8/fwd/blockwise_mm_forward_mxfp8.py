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

from typing import Optional

import nki
import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil
from ...moe.bwd.moe_bwd_parameters import ActFnType, AffinityOption, ClampLimits, ShardOption, SkipMode
from ...mxfp_utils.mxfp8_utils.common_dataclasses import TensorDescriptor
from ..moe_mxfp8_checkpoint_config import MXFP8MOECheckpointConfig, checkpoint_block_dims
from .bwmm_fwd_dropless_mxfp8 import blockwise_mm_fwd_dropless_mxfp8
from .moe_fwd_mxfp8_config import MatmulMxfp8KernelConfig, MXFP8MOEFwdConfig, auto_generate_moe_fwd_configs


def _validate_kernel_options(
    config: MXFP8MOEFwdConfig,
    activation_type: ActFnType,
    run_with_lnc2: bool,
    gate_up_weight_scales=None,
    down_weight_scales=None,
):
    """Gate features not yet wired into the MXFP8 forward dropless impl.

    Keeps the forward in lockstep with the backward's supported set: AFFINITY_ON_I,
    SiLU, E4M3, LNC2, BF16 compute. Sharding for the forward is SHARD_ON_BLOCK.
    """
    kernel_assert(
        config.affinity_option == AffinityOption.AFFINITY_ON_I,
        "blockwise_mm_fwd_mxfp8 currently only supports AFFINITY_ON_I",
    )

    # Pre-quantized weights are not yet supported in the forward: the forward
    # contracts over different axes than the backward, so it needs a distinct
    # forward-natural x4 layout (gate_up [E, H/4, 2*I_TP], down [E, I_TP/4, H])
    # plus matching MX scales. That layout is not yet produced/validated.
    # TODO(moe-fwd / M5): add forward weight prequantization + enable this path.
    kernel_assert(
        gate_up_weight_scales is None and down_weight_scales is None,
        "blockwise_mm_fwd_mxfp8 does not yet support pre-quantized weights",
    )
    kernel_assert(
        config.shard_option == ShardOption.SHARD_ON_BLOCK,
        "blockwise_mm_fwd_mxfp8 currently only supports SHARD_ON_BLOCK",
    )
    kernel_assert(
        activation_type == ActFnType.SiLU,
        "only ActFnType.SiLU is implemented in blockwise_mm_fwd_mxfp8",
    )
    kernel_assert(config.fp8_x4_dtype == float8_e4m3fn_x4, "Only E4M3 is tested, E5M2 works, but not tested")
    kernel_assert(run_with_lnc2 == True, "Kernel is expected to run only with LNC2")
    kernel_assert(config.compute_dtype == nl.bfloat16, "Only BF16 is supported, DGT does not support FP32")
    # Fast DMA transpose only supports the single-expert (E=1) case: the fast DGT
    # loader addresses the source directly and carries no per-expert offset, so it
    # is gated to the contiguous no_indirect_load path (which asserts E == 1).
    kernel_assert(
        not config.fast_dma_transpose or config.no_indirect_load,
        "fast_dma_transpose only supports the E=1 case (requires no_indirect_load=True)",
    )
    # Reusing one quantized weight copy across blocks is only correct when every
    # block uses the same expert. no_indirect_load asserts E == 1, making that a
    # compile-time fact; the routed path would need a runtime same-expert predicate.
    kernel_assert(
        not config.reuse_spilled_weights or config.no_indirect_load,
        "reuse_spilled_weights requires no_indirect_load=True (E=1)",
    )


def _validate_inputs_and_derive_dims(
    hidden_states,
    gate_up_proj_weight,
    down_proj_weight,
    token_position_to_id,
    block_to_expert,
    expert_affinities_masked,
    block_size,
    num_shards,
    no_indirect_load=False,
):
    """Validate raw inputs and return derived dims (T, H, I_TP, E, N) as plain ints.

    Mirrors the backward's validator but on the forward's input set (no
    output_hidden_states_grad, no checkpoints as inputs). Activations are BF16
    only (gathered per-block via indirect DMA, which breaks MXFP8 quant groups),
    and weights are unswizzled BF16 (pre-quantized weights are gated off).
    """
    required = (
        ("hidden_states", hidden_states),
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

    T = hidden_states.shape[0]
    H = hidden_states.shape[1]
    E = down_proj_weight.shape[0]
    # FORWARD-NATURAL weight layout (transpose of the backward's): the gate/up and
    # down GEMMs both contract over the input dim, and the per-expert DGT load path
    # contracts over data.shape[1] (K). So weights are stored [out, in] per expert:
    #   gate_up: [E, 2, I_TP, H]   down: [E, H, I_TP]
    # (Pre-quantized weights are gated off in _validate_kernel_options, so only the
    # non-prequant layouts are handled here.)
    I_TP = down_proj_weight.shape[2]
    N = div_ceil(T, block_size) if no_indirect_load else token_position_to_id.shape[0] // block_size

    # Weight ranks + full shapes (forward-natural orientation).
    gate_up_shape = gate_up_proj_weight.shape
    kernel_assert(
        len(gate_up_shape) == 4 and gate_up_shape == (E, 2, I_TP, H),
        f"gate_up_weight shape {tuple(gate_up_shape)} must match [E={E}, 2, I_TP={I_TP}, H={H}]",
    )

    down_shape = down_proj_weight.shape
    kernel_assert(
        len(down_shape) == 3 and down_shape == (E, H, I_TP),
        f"down_weight shape {tuple(down_shape)} must match [E={E}, H={H}, I_TP={I_TP}]",
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
    tpti_shape = token_position_to_id.shape
    bte_shape = block_to_expert.shape
    if no_indirect_load:
        kernel_assert(
            len(tpti_shape) == 1 and tpti_shape[0] == 1,
            f"no_indirect_load expects dummy token_position_to_id shape [1], got {tuple(tpti_shape)}",
        )
        kernel_assert(
            len(bte_shape) == 2 and bte_shape == (1, 1),
            f"no_indirect_load expects dummy block_to_expert shape [1, 1], got {tuple(bte_shape)}",
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
    kernel_assert(
        block_size in (128, 256, 512, 1024, 2048, 4096),
        f"block_size must be 128, 256, 512, 1024, 2048, or 4096, got {block_size}",
    )
    kernel_assert(H % 128 == 0, f"H={H} must be divisible by 128")
    kernel_assert(I_TP % 128 == 0, f"I_TP={I_TP} must be divisible by 128")
    # SHARD_ON_BLOCK distributes whole blocks across cores (range(shard_id, N, num_shards)),
    # so every core needs at least one block. When N < num_shards the idle core's output
    # tiles are never produced. A single-block (N < num_shards) col-parallel path is not yet
    # implemented; guard it out until then.
    kernel_assert(
        N >= num_shards,
        f"SHARD_ON_BLOCK requires N>=num_shards so each core owns a block, "
        f"got N={N} < num_shards={num_shards} (T={T}, block_size={block_size}). "
        f"Single-block (N < num_shards) is not yet supported.",
    )
    if no_indirect_load:
        kernel_assert(E == 1, f"no_indirect_load requires exactly one expert weight, got E={E}")
        kernel_assert(T % block_size == 0, "no_indirect_load requires T to be divisible by block_size")

    return T, H, I_TP, E, N


def blockwise_mm_fwd_mxfp8(
    # --- Required input tensors ---
    hidden_states: nl.ndarray,
    expert_affinities_masked: nl.ndarray,
    gate_up_proj_weight: nl.ndarray,
    down_proj_weight: nl.ndarray,
    token_position_to_id: nl.ndarray,
    block_to_expert: nl.ndarray,
    block_size: int,
    # --- Optional pre-quantized weight support ---
    gate_up_weight_scales: nl.ndarray = None,
    gate_up_weight_is_swizzled: bool = False,
    down_weight_scales: nl.ndarray = None,
    down_weight_is_swizzled: bool = False,
    # --- Per-phase matmul configs (None = default TILES_IN_BLOCK_*=1) ---
    gate_up_config: Optional[MatmulMxfp8KernelConfig] = None,
    down_config: Optional[MatmulMxfp8KernelConfig] = None,
    # --- MXFP8 configuration ---
    fp8_x4_dtype: type = float8_e4m3fn_x4,
    spill_reload: bool = False,
    reuse_spilled_weights: bool = False,
    use_scale_packing: bool = True,
    fast_dma_transpose: bool = False,
    run_with_lnc2: bool = True,
    # --- Sharding & affinity placement ---
    shard_option: ShardOption = ShardOption.SHARD_ON_BLOCK,
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_I,
    # --- Compute / DMA / accumulation knobs ---
    compute_dtype: nki.dtype = nl.bfloat16,
    skip_dma: SkipMode = None,
    is_tensor_update_accumulating: bool = True,
    no_indirect_load: bool = False,
    clamp_limits: ClampLimits = None,
    activation_type: ActFnType = ActFnType.SiLU,
    bias: bool = False,
    # --- Checkpoint emission ---
    checkpoint_config: Optional[MXFP8MOECheckpointConfig] = None,
) -> tuple:
    """MXFP8 forward pass for blockwise (dropless) Mixture of Experts.

    Computes the MoE FFN output and emits the activation checkpoints the MXFP8
    MoE backward (``blockwise_mm_bwd_mxfp8``) consumes, so fwd + bwd form a
    validated training pair. Tokens are processed in fixed-size blocks already
    assigned to a single expert each by an upstream router; this kernel never
    computes routing.

    Only weights support pre-quantized MXFP8 inputs. Activations (hidden_states)
    must be BF16 because they are gathered per-block via indirect DMA, which would
    break MXFP8 32-element quantization groups. When no_indirect_load is True,
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
        hidden_states (nl.ndarray): [T, H], input hidden states (BF16) on HBM.
        expert_affinities_masked (nl.ndarray): [T*E, 1], expert affinities (fp32) on HBM.
        gate_up_proj_weight (nl.ndarray): [E, 2, I_TP, H], gate/up weights on HBM
            in forward-natural [out, in] orientation (the transpose of the
            backward's [E, H, 2, I_TP]). The forward GEMMs contract over the input
            dim H, so the per-expert DGT load needs H as the contraction axis.
        down_proj_weight (nl.ndarray): [E, H, I_TP], down weights on HBM in
            forward-natural [out, in] orientation (transpose of the backward's
            [E, I_TP, H]); the down GEMM contracts over I_TP.
        token_position_to_id (nl.ndarray): [N*B] int32, token -> block-position map
            (pad id = -1 under skip_dma). Use a dummy [1] tensor when
            no_indirect_load is True.
        block_to_expert (nl.ndarray): [N, 1] int32, expert index per block. Use
            a dummy [1, 1] tensor when no_indirect_load is True.
        block_size (int): tokens per block (128/256/512/1024/2048/4096).
        gate_up_weight_scales (nl.ndarray, optional): MXFP8 scales for pre-quantized gate/up.
        gate_up_weight_is_swizzled (bool): whether gate/up weights are pre-swizzled.
        down_weight_scales (nl.ndarray, optional): MXFP8 scales for pre-quantized down.
        down_weight_is_swizzled (bool): whether down weights are pre-swizzled.
        gate_up_config / down_config (MatmulMxfp8KernelConfig, optional): per-phase
            matmul blocking. When None, defaults are used.
        fp8_x4_dtype (type): MXFP8 packed weight dtype (default float8_e4m3fn_x4).
        spill_reload (bool): spill quantized tiles to HBM for K-block reuse.
        reuse_spilled_weights (bool): carry quantized weight spill buffers across the
            block loop so only the first block each core loads+quantizes weights.
            Requires no_indirect_load=True (E=1); only affects phases that spill.
            Default False.
        use_scale_packing (bool): packed MXFP8 scale layout.
        fast_dma_transpose (bool): load unswizzled-BF16 operands via the direct-4D
            DGT access pattern (no vector_offset_pattern SBUF buffers). Only valid
            with no_indirect_load=True (the fast loader ignores per-expert
            scalar_offset); the dropless impl asserts this. Default False.
        run_with_lnc2 (bool): shard across 2 LNC cores.
        shard_option (ShardOption): sharding strategy (default SHARD_ON_BLOCK).
        affinity_option (AffinityOption): affinity placement; must match the backward
            (AFFINITY_ON_I — the forward folds affinity on the intermediate).
        compute_dtype (nki.dtype): dtype for SBUF/HBM intermediates + checkpoints (bf16).
        skip_dma (SkipMode): OOB handling for indirect-DMA token gather/scatter.
        is_tensor_update_accumulating (bool): when True (top_k>1) the output scatter
            does read-modify-write so experts touching the same token accumulate.
            Ignored when no_indirect_load is True because direct outputs do not scatter.
        no_indirect_load (bool): use contiguous single-expert inputs/weights and skip
            token-index gather, expert-indexed weight loads, affinity gather, and scatter.
        clamp_limits (ClampLimits): optional gate/up clamp, applied BEFORE the
            gate_up checkpoint + SiLU so the checkpoint matches the backward.
        activation_type (ActFnType): SiLU only (hardcoded in the dropless impl).
        bias (bool): whether gate/up + down biases are added (reserved surface).
        checkpoint_config (MXFP8MOECheckpointConfig, optional): per-checkpoint save
            flags selecting which activation checkpoints the forward emits for the
            backward. When a checkpoint is disabled the kernel skips computing +
            storing it and does not allocate/return it. Defaults to saving both.

    Returns:
        tuple:
            - output_hidden_states (nl.ndarray): [T, H] MoE FFN output.
          followed by each saved checkpoint, in this fixed order (an entry is
          present only when its checkpoint_config flag is set):
            - gate_up_proj_act_checkpoint_T (nl.ndarray): clamped gate pre-activation
              at [block, 0] and up at [block, 1]; present when
              checkpoint_config.save_gate_up_proj_act. Shape follows
              gate_up_proj_act_layout: TRANSPOSED -> [N, 2, I_TP, B] (B contiguous,
              the backward's layout); DIRECT -> [N, 2, B, I_TP] (I_TP contiguous).
            - scaled_intermediate_checkpoint_T (nl.ndarray): SiLU(gate)*up*EA;
              present when checkpoint_config.save_scaled_intermediate. Shape follows
              scaled_intermediate_layout: TRANSPOSED -> [N, I_TP, B]; DIRECT ->
              [N, B, I_TP].
    """
    if skip_dma == None:
        skip_dma = SkipMode(False, False)
    if clamp_limits == None:
        clamp_limits = ClampLimits()
    if checkpoint_config == None:
        checkpoint_config = MXFP8MOECheckpointConfig()

    config = MXFP8MOEFwdConfig(
        compute_dtype=compute_dtype,
        fp8_x4_dtype=fp8_x4_dtype,
        activation_type=activation_type,
        shard_option=shard_option,
        affinity_option=affinity_option,
        gate_up_config=gate_up_config,
        down_config=down_config,
        is_tensor_update_accumulating=False if no_indirect_load else is_tensor_update_accumulating,
        no_indirect_load=no_indirect_load,
        clamp_limits=clamp_limits,
        skip_dma=skip_dma,
        bias=bias,
        checkpoint_config=checkpoint_config,
        fast_dma_transpose=fast_dma_transpose,
        reuse_spilled_weights=reuse_spilled_weights,
    )
    config.gate_up_config.spill_reload = spill_reload
    config.gate_up_config.enable_scale_packing = use_scale_packing
    config.down_config.spill_reload = spill_reload
    config.down_config.enable_scale_packing = use_scale_packing

    _validate_kernel_options(
        config=config,
        activation_type=activation_type,
        run_with_lnc2=run_with_lnc2,
        gate_up_weight_scales=gate_up_weight_scales,
        down_weight_scales=down_weight_scales,
    )

    num_shards = nl.num_programs(axes=0) if run_with_lnc2 else 1
    T, H, I_TP, E, N = _validate_inputs_and_derive_dims(
        hidden_states=hidden_states,
        gate_up_proj_weight=gate_up_proj_weight,
        down_proj_weight=down_proj_weight,
        token_position_to_id=token_position_to_id,
        block_to_expert=block_to_expert,
        expert_affinities_masked=expert_affinities_masked,
        block_size=block_size,
        num_shards=num_shards,
        no_indirect_load=no_indirect_load,
    )

    # Auto-generate the per-phase matmul configs from the derived dims. The
    # forward drives generic_matmul_mxfp8_api directly (not the generic matmul
    # kernel entry point), so its configs never otherwise pass through
    # auto_generate_default — this fills the derived fields (tile sizes,
    # TILES_IN_LOAD_M/N) the dropless impl feeds into the matmul calls.
    auto_generate_moe_fwd_configs(config, block_size=block_size, H=H, I_TP=I_TP)

    # Build TensorDescriptors locally (passed flat into the dropless impl).
    # Pre-quantized weights are gated off in _validate_kernel_options, so weights
    # are always unswizzled BF16 in the forward-natural [out, in] orientation.
    hidden_states_td = TensorDescriptor(data=hidden_states)

    # Forward-natural gate_up: [E, 2, I_TP, H] -> 2D [E*2*I_TP, H] (F-by-K,
    # F=2*I_TP per expert, K=H). Per-expert slice via scalar_offset; the up half
    # is the second I_TP of the F slice (selected by rhs_n_offset=I_TP).
    gate_up_weight_td = TensorDescriptor(data=gate_up_proj_weight.reshape((E * 2 * I_TP, H)))

    # Forward-natural down: [E, H, I_TP] -> 2D [E*H, I_TP] (F-by-K, F=H per
    # expert, K=I_TP). Per-expert slice via scalar_offset.
    down_weight_td = TensorDescriptor(data=down_proj_weight.reshape((E * H, I_TP)))

    token_position_to_id_td = TensorDescriptor(data=token_position_to_id)
    block_to_expert_td = TensorDescriptor(data=block_to_expert)
    expert_affinities_masked_td = TensorDescriptor(data=expert_affinities_masked)

    # Allocate outputs. SHARD_ON_BLOCK gives each core its own scratch output
    # slab (output_slabs[shard_id]); the per-shard slabs are summed into the
    # returned [T, H] output by the final reduce. Separate slabs avoid the
    # cross-core write race when top_k > 1 routes a token's blocks onto
    # different cores. Checkpoints are written per-block, so no cross-core
    # aliasing there (each core owns a disjoint block subset).
    #
    # no_indirect_load (E=1, top_k=1) writes disjoint output rows per core
    # directly into output_hidden_states, so the scratch slabs are unnecessary
    # (skips their alloc + zero-init + reduce in the dropless impl).
    hbm_buffer = nl.shared_hbm if run_with_lnc2 else nl.hbm
    output_hidden_states = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=hbm_buffer)
    output_slabs = (
        None if no_indirect_load else nl.ndarray((num_shards, T, H), dtype=hidden_states.dtype, buffer=hbm_buffer)
    )
    # Allocate each checkpoint only when its save flag is set; a disabled
    # checkpoint is passed to the impl as None so it skips the store.
    # Per-block trailing dims come from checkpoint_block_dims (single source of
    # truth shared with the torch ref + test): DIRECT is token-major (the only
    # layout the block-granular store implements), TRANSPOSED is I_TP-major (the
    # backward's contract; not currently supported, see CheckpointLayout).
    gate_up_proj_act_checkpoint_T = None
    if checkpoint_config.save_gate_up_proj_act:
        gate_up_shape = (N, 2) + checkpoint_block_dims(checkpoint_config.gate_up_proj_act_layout, I_TP, block_size)
        gate_up_proj_act_checkpoint_T = nl.ndarray(gate_up_shape, dtype=compute_dtype, buffer=hbm_buffer)
    scaled_intermediate_checkpoint_T = None
    if checkpoint_config.save_scaled_intermediate:
        scaled_shape = (N,) + checkpoint_block_dims(checkpoint_config.scaled_intermediate_layout, I_TP, block_size)
        scaled_intermediate_checkpoint_T = nl.ndarray(scaled_shape, dtype=compute_dtype, buffer=hbm_buffer)

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
        scaled_intermediate_checkpoint_T=scaled_intermediate_checkpoint_T,
    )

    # Return arity tracks the save flags (mirrors the backward's bias-conditional
    # return): each checkpoint is appended only when saved, so the test framework's
    # positional output mapping never sees a None. Order is fixed: output first,
    # then gate/up, then scaled intermediate.
    outputs = [output_hidden_states]
    if checkpoint_config.save_gate_up_proj_act:
        outputs.append(gate_up_proj_act_checkpoint_T)
    if checkpoint_config.save_scaled_intermediate:
        outputs.append(scaled_intermediate_checkpoint_T)
    return tuple(outputs)
