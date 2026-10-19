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

"""Configuration classes for the MXFP8 MoE forward pass kernel."""

from dataclasses import dataclass
from typing import Optional

import nki
import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ....core.utils.kernel_helpers import _NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE
from ...matmul_mxfp8._matmul_mxfp8_constants import PRECISION_BFLOAT16
from ...matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig, auto_generate_default
from ...moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)
from ...mxfp_utils.mxfp8_utils.common_dataclasses import LncShardingMode, QuantScheme, TensorDescriptor
from ..config_helpers import ExecutionMode
from ..moe_mxfp8_checkpoint_config import MXFP8MOECheckpointConfig


def _default_matmul_config(quant_scheme):
    """Per-phase defaults use the fused kernel's resolved quantization scheme."""
    return MatmulMxfp8KernelConfig(
        M=0, K=0, N=0, quant_scheme=quant_scheme, spill_reload=True, enable_scale_packing=True
    )


@dataclass
class MXFP8MOEFwdConfig(nl.NKIObject):
    """User presets for the fused kernel.

    Phase configs remain None until the cache has been checked. A miss or a
    conflicting hit resolves them through the existing phase defaults.
    execution_mode is the source of truth for dense MoE, routed MoE and MLP.
    """

    # Compute settings
    compute_dtype: nki.dtype = nl.bfloat16
    fp8_x4_dtype: type = float8_e4m3fn_x4
    activation_type: ActFnType = ActFnType.SiLU

    # Sharding & affinity
    shard_option: ShardOption = ShardOption.SHARD_ON_BLOCK
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_I

    # Shape-dependent phase defaults are resolved after the cache lookup.
    gate_up_config: MatmulMxfp8KernelConfig = None
    down_config: MatmulMxfp8KernelConfig = None

    # Accumulation across blocks (top_k > 1 routing)
    is_tensor_update_accumulating: bool = False

    # Single-expert contiguous mode
    # Contiguous MLP without routing, expert affinities, or dynamic blocks.
    execution_mode: ExecutionMode = ExecutionMode.DENSE_MOE
    num_static_block: Optional[int] = None

    # Gate/up activation clamping
    clamp_limits: ClampLimits = None

    # OOB handling for indirect DMA
    skip_dma: SkipMode = None

    # Bias (reserved surface)
    bias: bool = False

    # Per-checkpoint save flags (which activation checkpoints the forward emits)
    checkpoint_config: MXFP8MOECheckpointConfig = None

    # Data-free tensor-local layout descriptors. Specify the quantization scheme,
    # swizzle mode, orientation, and scale packing for the gate/up + down weights;
    # the entry point attaches the real tensor data (+ scales) and copies these facts
    # onto the data-bearing TDs it passes to the dropless impl.
    #
    # The two weight TDs stay None when unset (see __post_init__). On the pre-quantized
    # path they are the source of truth for the x4 swizzle scheme: None -> default (1x32
    # on single_expert_dense, WRAPX on the routed path); set quant_scheme=WRAPX to force
    # WRAPX. On the BF16 path, orientation independently selects each input layout:
    # F_BY_K (default) is forward-natural, while K_BY_F accepts the backward-natural
    # layout so one BF16 copy can feed both passes. Dense K_BY_F uses PE transpose
    # directly; routed K_BY_F is canonicalized once into shared forward-natural
    # storage before the per-expert block loop. When H == I_TP the two layouts have the
    # same shape, so the descriptor remains the only source of truth; a wrong
    # orientation cannot be inferred from shape and produces wrong numerics.
    gate_up_weight_td: Optional[TensorDescriptor] = None
    down_weight_td: Optional[TensorDescriptor] = None

    # Fast DGT: direct 4D access pattern on unswizzled-BF16 operands (skips the
    # vector_offset_pattern SBUF buffers). Only valid on the single_expert_dense
    # path (the fast loader ignores per-expert scalar_offset).
    fast_dma_transpose: Optional[bool] = None

    # Carry the quantized weight spill buffers across the block loop instead of
    # rebuilding per block. Only valid on single_expert_dense (E=1); only affects
    # phases that spill.
    reuse_spilled_weights: bool = True

    # Load unswizzled-BF16 operands via the 1x32 PE-swizzle transpose
    # (load_tile_PE_Swizzle_1x32) instead of DGT. Only valid on single_expert_dense
    # (E=1); H / I_TP need only be % 128 (flexible-K loader tiles K as 512 + a
    # <512 remainder). This path uses PE regardless of fast_dma_transpose.
    use_1x32_pe_swizzle: bool = True

    dma_transpose_enabled: bool = not _NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE

    def __post_init__(self):
        """Initialize optional metadata while retaining unset phase presets."""
        if self.fast_dma_transpose == None:
            self.fast_dma_transpose = self.dma_transpose_enabled
        # TODO: deprecate explicit fast_dma_transpose=False when DMA transpose is enabled.
        if self.checkpoint_config == None:
            self.checkpoint_config = MXFP8MOECheckpointConfig()
        if self.clamp_limits == None:
            self.clamp_limits = ClampLimits()
        if self.skip_dma == None:
            self.skip_dma = SkipMode()
        # Weight layout descriptors stay None when unset (the "caller did not specify"
        # sentinel): the entry falls back to plain unswizzled BF16 for the BF16 path,
        # and for the pre-quantized path a None weight TD selects the default x4 scheme
        # (1x32 on single_expert_dense, WRAPX on the routed path). A supplied TD's
        # quant_scheme overrides that.

    def resolve_phase_defaults(self):
        scheme = QuantScheme._1x32 if self.use_1x32_pe_swizzle else QuantScheme.WRAPX
        if self.gate_up_config == None:
            self.gate_up_config = _default_matmul_config(scheme)
        if self.down_config == None:
            self.down_config = _default_matmul_config(scheme)


def _auto_generate_phase_config(phase_config, M, K, N):
    """Fill the derived fields (tile sizes, TILES_IN_LOAD_*) on one phase config.

    Both forward GEMMs run unswizzled BF16 activations against unswizzled BF16
    weights, and SHARD_ON_BLOCK places each whole block on a single core, so the
    matmul blocking spans the full (unsharded) H / I_TP — ``run_with_lnc2`` is
    False for the blocking math so auto-gen derives TILES_IN_LOAD_M/N against the
    full (non-halved) N, matching the tiles the dropless impl loads at.

    Only unset fields are generated; the whole-kernel cache is checked earlier.
    """
    if phase_config.M == 0:
        phase_config.M = M
    if phase_config.K == 0:
        phase_config.K = K
    if phase_config.N == 0:
        phase_config.N = N
    phase_config.lnc_sharding = LncShardingMode.OFF
    auto_generate_default(
        phase_config,
        PRECISION_BFLOAT16,
        PRECISION_BFLOAT16,
        PRECISION_BFLOAT16,
        # Standalone GEMM winners have not been measured together as this fused kernel.
        use_cache=False,
        lhs_is_swizzled=False,
        rhs_is_swizzled=False,
        lhs_load_with_PE_swizzle=False,
        rhs_load_with_PE_swizzle=False,
    )


def _auto_generate_moe_fwd_configs(config: MXFP8MOEFwdConfig, block_size: int, H: int, I_TP: int):
    """Auto-generate the per-phase matmul configs from the derived MoE dimensions.

    The forward drives ``generic_matmul_mxfp8_api`` directly (not the generic
    matmul kernel entry point), so its per-phase ``MatmulMxfp8KernelConfig``s
    never pass through ``auto_generate_default`` and their derived fields stay
    None. This fills them so the dropless impl can feed the auto-generated
    TILES_IN_LOAD_M/N into the matmul calls.

    Args:
        config (MXFP8MOEFwdConfig): forward config whose gate_up_config /
            down_config are populated in place.
        block_size (int): tokens per block (B), the M dimension of both GEMMs.
        H (int): hidden dimension.
        I_TP (int): intermediate size / tensor-parallel degree.
    """
    config.resolve_phase_defaults()
    # gate/up GEMM: hidden_block[B, H] @ W_gate_up[H, I_TP] -> M=B, K=H, N=I_TP.
    _auto_generate_phase_config(config.gate_up_config, M=block_size, K=H, N=I_TP)
    # down GEMM: scaled_intermediate[B, I_TP] @ W_down[I_TP, H] -> M=B, K=I_TP, N=H.
    _auto_generate_phase_config(config.down_config, M=block_size, K=I_TP, N=H)
