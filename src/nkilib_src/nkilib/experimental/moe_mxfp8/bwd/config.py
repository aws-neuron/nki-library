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

"""Configuration for the MXFP8 MoE backward kernel."""

from dataclasses import dataclass, field
from enum import Enum

import nki
import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ...matmul_mxfp8.matmul_mxfp8_config import (
    MatmulMxfp8KernelConfig,
)
from ...moe.bwd.moe_bwd_parameters import ActFnType, AffinityOption, ClampLimits, ShardOption, SkipMode
from ...mxfp_utils.mxfp8_utils.common_dataclasses import QuantScheme, SwizzleMode


class TransposeMode(Enum):
    """Engine used to materialize a transposed operand for one backward phase."""

    NC = "nc"
    DMA = "dma"


__all__ = [
    "MXFP8MOEBwdConfig",
    "MatmulMxfp8KernelConfig",
    "QuantScheme",
    "SwizzleMode",
    "TransposeMode",
]


def _default_matmul_config():
    return MatmulMxfp8KernelConfig(
        M=0,
        K=0,
        N=0,
        TILES_IN_BLOCK_M=1,
        TILES_IN_BLOCK_N=1,
        TILES_IN_BLOCK_K=1,
        quant_scheme=QuantScheme.WRAPX,
    )


@dataclass
class MXFP8MOEBwdConfig(nl.NKIObject):
    """Typed configuration for the MXFP8 MoE backward kernel.

    This configuration contains no tensors.

    Args:
        compute_dtype (nki.dtype): Compute data type for intermediate results (default: nl.bfloat16).
        fp8_x4_dtype (type): MXFP8 packed data type (default: float8_e4m3fn_x4).
        activation_type (ActFnType): Activation function type (default: SiLU).
        shard_option (ShardOption): LNC2 sharding strategy (default: SHARD_ON_FREE).
        affinity_option (AffinityOption): Affinity scaling dimension (default: AFFINITY_ON_H).
        phase1_config (MatmulMxfp8KernelConfig): Matmul config for Phase 1 (gate_up_proj_output_grad + SwiGLU bwd + ea grad).
        phase2_config (MatmulMxfp8KernelConfig): Matmul config for Phase 2 (hidden_states_grad).
        phase3_config (MatmulMxfp8KernelConfig): Matmul config for Phase 3 (gate_up_weight_grad).
        phase4_config (MatmulMxfp8KernelConfig): Matmul config for Phase 4 (down_weight_grad).
        accumulate_hidden_states_grad (bool): Whether Phase 2 accumulates into
            existing hidden-state gradients.
        skip_grad_initialization (bool): Whether to skip zeroing gradient outputs.
        single_expert_dense (bool): Whether inputs are packed contiguously for one expert.
        fast_dma_transpose (bool): Whether unswizzled BF16 matmul operands use direct DGT addressing.
        clamp_limits (ClampLimits): Optional gradient clamping limits.
        skip_dma (SkipMode): OOB handling mode for DMA operations.
        bias (bool): Whether to compute bias gradients.
    """

    # Compute settings
    compute_dtype: nki.dtype = nl.bfloat16
    fp8_x4_dtype: type = float8_e4m3fn_x4
    activation_type: ActFnType = ActFnType.SiLU

    # Sharding & affinity
    shard_option: ShardOption = ShardOption.SHARD_ON_FREE
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_H

    # Per-phase matmul configs default to one tile per block. The kernel fills
    # shape-dependent fields after input validation.
    phase1_config: MatmulMxfp8KernelConfig = field(default_factory=_default_matmul_config)
    phase2_config: MatmulMxfp8KernelConfig = field(default_factory=_default_matmul_config)
    phase3_config: MatmulMxfp8KernelConfig = field(default_factory=_default_matmul_config)
    phase4_config: MatmulMxfp8KernelConfig = field(default_factory=_default_matmul_config)

    # Accumulation
    accumulate_hidden_states_grad: bool = True
    skip_grad_initialization: bool = False

    # Single-expert inputs packed in block order. Routing tensors are not read.
    single_expert_dense: bool = False

    # Direct 4D DMA gather-transpose for unswizzled BF16 matmul operands.
    fast_dma_transpose: bool = False

    # Gradient clamping
    clamp_limits: ClampLimits = None

    # Skip DMA for OOB
    skip_dma: SkipMode = None

    # Bias
    bias: bool = False

    # Per-phase transpose mechanism for BUILDING an [F,B] transpose buffer on the DGT path
    # (TransposeMode.{NC,DMA}). NC = nc_transpose on PE (original); DMA = dma_transpose on the
    # DMA engine (no PE cycles, no PSUM). Only P3 (hidden_states_T) and P4 (output_grad_T,
    # scaled_intermediate_T) build transposed operands.
    phase3_transpose_mode: TransposeMode = TransposeMode.NC
    phase4_transpose_mode: TransposeMode = TransposeMode.NC

    def __post_init__(self):
        """Validate typed fields and initialize optional values."""
        for name in ("phase1_config", "phase2_config", "phase3_config", "phase4_config"):
            phase_config = getattr(self, name)
            if phase_config is None:
                phase_config = _default_matmul_config()
                setattr(self, name, phase_config)
            if not isinstance(phase_config, MatmulMxfp8KernelConfig):
                raise TypeError(f"{name} must be a MatmulMxfp8KernelConfig")
            if not isinstance(phase_config.quant_scheme, QuantScheme):
                raise TypeError(f"{name}.quant_scheme must be a QuantScheme")
        if not isinstance(self.phase3_transpose_mode, TransposeMode):
            raise TypeError("phase3_transpose_mode must be a TransposeMode")
        if not isinstance(self.phase4_transpose_mode, TransposeMode):
            raise TypeError("phase4_transpose_mode must be a TransposeMode")
        if self.clamp_limits == None:
            self.clamp_limits = ClampLimits()
