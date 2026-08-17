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

import nki
import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ...matmul_mxfp8.matmul_mxfp8_config import MatmulMxfp8KernelConfig, auto_generate_default
from ...matmul_mxfp8.matmul_mxfp8_constants import PRECISION_BFLOAT16
from ...moe.bwd.moe_bwd_parameters import ActFnType, AffinityOption, ClampLimits, ShardOption, SkipMode
from ..moe_mxfp8_checkpoint_config import MXFP8MOECheckpointConfig


@dataclass
class MXFP8MOEFwdConfig(nl.NKIObject):
    """Configuration for the MXFP8 MoE forward pass kernel.

    Mirrors :class:`MXFP8MOEBwdConfig` so the forward and backward share the same
    tuning surface and stay symmetric. Groups the compute / sharding / affinity
    knobs plus the per-phase matmul blocking configs.

    The forward has only two matmul phases (gate_up and down), versus the
    backward's four, so it carries two per-phase configs.

    Args:
        compute_dtype (nki.dtype): Compute dtype for SBUF/HBM intermediates and
            the emitted activations/checkpoints (default: nl.bfloat16).
        fp8_x4_dtype (type): MXFP8 packed weight dtype (default: float8_e4m3fn_x4).
        activation_type (ActFnType): Gate activation (default: SiLU).
        shard_option (ShardOption): LNC2 sharding strategy. The forward kernel is
            built for SHARD_ON_BLOCK (each core owns a disjoint subset of the N
            blocks and writes its own output slab; slabs are reduced at the end).
        affinity_option (AffinityOption): Where the expert affinity scalar is
            folded into the FFN chain. Must match the backward's choice. The
            forward folds on the intermediate (AFFINITY_ON_I) so the pre-down
            value equals the scaled_intermediate checkpoint.
        gate_up_config (MatmulMxfp8KernelConfig): Matmul blocking for the gate/up
            projection. When None, a default with TILES_IN_BLOCK_*=1 is built.
        down_config (MatmulMxfp8KernelConfig): Matmul blocking for the down
            projection. When None, a default with TILES_IN_BLOCK_*=1 is built.
        is_tensor_update_accumulating (bool): When True (top_k > 1), the per-block
            output scatter does a read-modify-write so multiple experts touching
            the same token accumulate. When False (top_k == 1), it overwrites.
        no_indirect_load (bool): When True, the kernel assumes hidden states are
            already packed for one expert and weights contain only that expert.
            Token gather, expert-indexed weight loads, affinity gather, and output
            scatter use contiguous direct addressing instead of indirect DMA.
        clamp_limits (ClampLimits): Optional gate/up activation clamp limits.
            Applied BEFORE the gate_up checkpoint is written and BEFORE SiLU, so
            the checkpoint matches what the backward expects (it does not re-clamp).
        skip_dma (SkipMode): OOB handling for indirect-DMA token gather/scatter.
        bias (bool): Whether gate/up and down biases are added (reserved surface).
        checkpoint_config (MXFP8MOECheckpointConfig): Per-checkpoint save flags
            controlling which activation checkpoints the forward emits for the
            backward. When a checkpoint is disabled the forward skips computing +
            storing it (and the entry point does not allocate/return it).
    """

    # Compute settings
    compute_dtype: nki.dtype = nl.bfloat16
    fp8_x4_dtype: type = float8_e4m3fn_x4
    activation_type: ActFnType = ActFnType.SiLU

    # Sharding & affinity
    shard_option: ShardOption = ShardOption.SHARD_ON_BLOCK
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_I

    # Per-phase matmul configs. When None, defaults are constructed with
    # TILES_IN_BLOCK_*=1 and the remaining fields filled from tensor shapes by
    # the kernel entry point.
    gate_up_config: MatmulMxfp8KernelConfig = None
    down_config: MatmulMxfp8KernelConfig = None

    # Accumulation across blocks (top_k > 1 routing)
    is_tensor_update_accumulating: bool = True

    # Single-expert contiguous mode
    no_indirect_load: bool = False

    # Gate/up activation clamping
    clamp_limits: ClampLimits = None

    # OOB handling for indirect DMA
    skip_dma: SkipMode = None

    # Bias (reserved surface)
    bias: bool = False

    # Per-checkpoint save flags (which activation checkpoints the forward emits)
    checkpoint_config: MXFP8MOECheckpointConfig = None

    def __post_init__(self):
        """Initialize default matmul configs and optional fields."""
        if self.checkpoint_config == None:
            self.checkpoint_config = MXFP8MOECheckpointConfig()
        if self.gate_up_config == None:
            self.gate_up_config = MatmulMxfp8KernelConfig(
                M=0,
                K=0,
                N=0,
            )
        if self.down_config == None:
            self.down_config = MatmulMxfp8KernelConfig(
                M=0,
                K=0,
                N=0,
            )
        if self.clamp_limits == None:
            self.clamp_limits = ClampLimits()
        if self.skip_dma == None:
            self.skip_dma = SkipMode()


def _auto_generate_phase_config(phase_config, M, K, N):
    """Fill the derived fields (tile sizes, TILES_IN_LOAD_*) on one phase config.

    Both forward GEMMs run unswizzled BF16 activations against unswizzled BF16
    weights, and SHARD_ON_BLOCK places each whole block on a single core, so the
    matmul blocking spans the full (unsharded) H / I_TP — ``run_with_lnc2`` is
    False for the blocking math so auto-gen derives TILES_IN_LOAD_M/N against the
    full (non-halved) N, matching the tiles the dropless impl loads at.

    The autotune cache is consulted (use_cache=True): with both operands
    unswizzled BF16 the lookup key resolves to ``{M}x{K}x{N}_bf16_bf16_dgt``, so a
    shape with a tuned DGT entry gets its offline blocking; otherwise auto-gen
    fills the fields heuristically.
    """
    if phase_config.M == 0:
        phase_config.M = M
    if phase_config.K == 0:
        phase_config.K = K
    if phase_config.N == 0:
        phase_config.N = N
    phase_config.run_with_lnc2 = False
    phase_config.lhs_is_swizzled = False
    phase_config.rhs_is_swizzled = False
    # use_cache=True: the autotune cache is now keyed by load method, so the
    # forward's unswizzled BF16 GEMMs hit their own DGT entries
    # ({M}x{K}x{N}_bf16_bf16_dgt) rather than the swizzled standalone-matmul ones.
    # On a hit the cached TILES_IN_BLOCK_*/TILES_IN_LOAD_* (and spill_reload) are
    # used; on a miss auto-gen fills the None fields heuristically. The dropless
    # impl re-clamps whatever it consumes to legal values and ignores the cached
    # tile_m/tile_k/tile_n (it pins its own geometry via get_tile_sizes).
    auto_generate_default(
        phase_config,
        PRECISION_BFLOAT16,
        PRECISION_BFLOAT16,
        PRECISION_BFLOAT16,
        use_cache=True,
    )


def auto_generate_moe_fwd_configs(config: MXFP8MOEFwdConfig, block_size: int, H: int, I_TP: int):
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
    # gate/up GEMM: hidden_block[B, H] @ W_gate_up[H, I_TP] -> M=B, K=H, N=I_TP.
    _auto_generate_phase_config(config.gate_up_config, M=block_size, K=H, N=I_TP)
    # down GEMM: scaled_intermediate[B, I_TP] @ W_down[I_TP, H] -> M=B, K=I_TP, N=H.
    _auto_generate_phase_config(config.down_config, M=block_size, K=I_TP, N=H)
