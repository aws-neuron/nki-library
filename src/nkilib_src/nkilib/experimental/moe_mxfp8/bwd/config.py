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

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import nki
import nki.language as nl
from nki.dtype import float8_e4m3fn_x4

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import _NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE
from ...matmul_mxfp8.matmul_mxfp8_config import (
    MatmulMxfp8KernelConfig,
)
from ...moe.bwd.moe_bwd_parameters import (
    ActFnType,
    AffinityOption,
    ClampLimits,
    ShardOption,
    SkipMode,
)
from ...mxfp_utils.mxfp8_utils.common_dataclasses import (
    QuantScheme,
    SwizzleMode,
    TensorDescriptor,
)
from ..config_helpers import ExecutionMode


class TransposeMode(Enum):
    """Engine used to materialize a transposed operand for one backward phase."""

    NC = "nc"
    DMA = "dma"


__all__ = [
    "MXFP8MOEBwdConfig",
    "MatmulMxfp8KernelConfig",
    "QuantScheme",
    "SwizzleMode",
    "TensorDescriptor",
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
        quant_scheme=QuantScheme._1x32,
        spill_reload=True,
        enable_scale_packing=True,
    )


@dataclass
class MXFP8MOEBwdConfig(nl.NKIObject):
    """User presets for the fused kernel.

    Phase configs remain None until the cache has been checked. A miss or a
    conflicting hit resolves them through the existing phase defaults.
    execution_mode is the source of truth for dense MoE, routed MoE and MLP.
    """

    # Compute settings
    compute_dtype: nki.dtype = nl.bfloat16
    fp8_x4_dtype: type = float8_e4m3fn_x4
    activation_type: ActFnType = ActFnType.SiLU

    # AFFINITY_ON_I is the only mode the backward implements, so it is the default.
    shard_option: ShardOption = ShardOption.SHARD_ON_FREE
    affinity_option: AffinityOption = AffinityOption.AFFINITY_ON_I

    # Unset phases receive one-tile-per-block defaults after the cache lookup.
    # The kernel fills shape-dependent fields after input validation.
    phase1_config: MatmulMxfp8KernelConfig = None
    phase2_config: MatmulMxfp8KernelConfig = None
    phase3_config: MatmulMxfp8KernelConfig = None
    phase4_config: MatmulMxfp8KernelConfig = None

    # Accumulation
    accumulate_hidden_states_grad: bool = False
    skip_grad_initialization: bool = False

    # Single-expert inputs packed in block order. Routing tensors are not read.
    # Contiguous MLP without routing, expert affinities, or dynamic blocks.
    execution_mode: ExecutionMode = ExecutionMode.DENSE_MOE
    num_static_block: Optional[int] = None

    # Direct 4D DMA gather-transpose for unswizzled BF16 matmul operands.
    fast_dma_transpose: Optional[bool] = None

    # Leave tuning choices unset until the whole-kernel cache has been checked.
    pe_transpose_only: Optional[bool] = None

    # Gradient clamping
    clamp_limits: ClampLimits = None

    # Skip DMA for OOB
    skip_dma: SkipMode = None

    # Bias
    bias: bool = False

    # Engine that builds the [F,B] transpose buffer on the DGT path: NC = nc_transpose on
    # the PE, DMA = dma_transpose (no PE cycles, no PSUM). Only P3/P4 transpose operands.
    phase3_transpose_mode: Optional[TransposeMode] = None
    phase4_transpose_mode: Optional[TransposeMode] = None

    # Per-operand tensor-local matmul layout conversion mode (DGT or PE).
    # TODO: support PE for the two weight modes with E > 1 (only E == 1 works today).
    output_grad_swizzle_mode: Optional[SwizzleMode] = None
    down_weight_swizzle_mode: Optional[SwizzleMode] = None
    d_gate_up_swizzle_mode: Optional[SwizzleMode] = None
    gate_up_weight_swizzle_mode: Optional[SwizzleMode] = None
    d_gate_up_t_swizzle_mode: Optional[SwizzleMode] = None
    hidden_states_t_swizzle_mode: Optional[SwizzleMode] = None
    output_grad_t_swizzle_mode: Optional[SwizzleMode] = None
    scaled_intermediate_t_swizzle_mode: Optional[SwizzleMode] = None

    # Optional data-free raw-BF16 weight layout descriptors. None preserves the
    # backward-natural input contract. Pre-quantized x4 weights keep their fixed layout.
    gate_up_weight_td: Optional[TensorDescriptor] = None
    down_weight_td: Optional[TensorDescriptor] = None

    dma_transpose_enabled: bool = not _NKI_DMA_TRANSPOSE_AS_PE_TRANSPOSE

    def __post_init__(self):
        """Validate supplied phases without filling unset phase presets."""
        if self.fast_dma_transpose == None:
            self.fast_dma_transpose = self.dma_transpose_enabled
        # TODO: deprecate explicit fast_dma_transpose=False when DMA transpose is enabled.
        quant_scheme = None
        for name in ("phase1_config", "phase2_config", "phase3_config", "phase4_config"):
            phase_config = getattr(self, name)
            if phase_config is None:
                continue
            if not isinstance(phase_config, MatmulMxfp8KernelConfig):
                raise TypeError(f"{name} must be a MatmulMxfp8KernelConfig")
            if not isinstance(phase_config.quant_scheme, QuantScheme):
                raise TypeError(f"{name}.quant_scheme must be a QuantScheme")
            if quant_scheme is None:
                quant_scheme = phase_config.quant_scheme
            kernel_assert(
                phase_config.quant_scheme == quant_scheme,
                "All MXFP8 MoE backward phases must use the same quantization scheme",
            )
        if self.phase3_transpose_mode is not None and not isinstance(self.phase3_transpose_mode, TransposeMode):
            raise TypeError("phase3_transpose_mode must be a TransposeMode")
        if self.phase4_transpose_mode is not None and not isinstance(self.phase4_transpose_mode, TransposeMode):
            raise TypeError("phase4_transpose_mode must be a TransposeMode")
        for mode_name in (
            "output_grad_swizzle_mode",
            "down_weight_swizzle_mode",
            "d_gate_up_swizzle_mode",
            "gate_up_weight_swizzle_mode",
            "d_gate_up_t_swizzle_mode",
            "hidden_states_t_swizzle_mode",
            "output_grad_t_swizzle_mode",
            "scaled_intermediate_t_swizzle_mode",
        ):
            if getattr(self, mode_name) not in (None, SwizzleMode.DGT, SwizzleMode.PE):
                raise ValueError(f"Unsupported {mode_name}: {getattr(self, mode_name)}")
        if self.clamp_limits == None:
            self.clamp_limits = ClampLimits()

    def resolve_phase_defaults(self):
        if self.pe_transpose_only == None:
            self.pe_transpose_only = False
        for name in ("phase3_transpose_mode", "phase4_transpose_mode"):
            if getattr(self, name) == None:
                setattr(self, name, TransposeMode.NC)
        for name in (
            "output_grad_swizzle_mode",
            "down_weight_swizzle_mode",
            "d_gate_up_swizzle_mode",
            "gate_up_weight_swizzle_mode",
            "d_gate_up_t_swizzle_mode",
            "hidden_states_t_swizzle_mode",
            "output_grad_t_swizzle_mode",
            "scaled_intermediate_t_swizzle_mode",
        ):
            if getattr(self, name) == None:
                setattr(self, name, SwizzleMode.PE)
        scheme = QuantScheme._1x32
        for phase in (self.phase1_config, self.phase2_config, self.phase3_config, self.phase4_config):
            if phase != None:
                scheme = phase.quant_scheme
                break
        for name in ("phase1_config", "phase2_config", "phase3_config", "phase4_config"):
            if getattr(self, name) == None:
                phase = _default_matmul_config()
                phase.quant_scheme = scheme
                setattr(self, name, phase)
        for phase in (self.phase1_config, self.phase2_config, self.phase3_config, self.phase4_config):
            kernel_assert(
                phase.quant_scheme == scheme,
                "All MXFP8 MoE backward phases must use the same quantization scheme",
            )
