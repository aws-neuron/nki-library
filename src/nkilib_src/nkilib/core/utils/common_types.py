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

"""Common type definitions (enums) shared across NKI Library kernels."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional

import nki.language as nl


class QKVOutputLayout(Enum):
    BSD = 0  # (b, s, (n_q_heads + 2 * n_kv_heads) * d_head)
    NBSd = 1  # (num_heads, b, s, d_head)
    NBdS = 2  # (num_heads, b, d_head, s)


class NormType(Enum):
    NO_NORM = 0
    RMS_NORM = 1
    LAYER_NORM = 2
    RMS_NORM_SKIP_GAMMA = 3


class ActFnType(Enum):
    SiLU = 0
    GELU = 1
    GELU_Tanh_Approx = 2
    Swish = 3
    ReLU = 4
    SquaredReLU = 5


class RouterActFnType(Enum):
    """Supported activation types for RouterTopK kernel"""

    SIGMOID = 0
    SOFTMAX = 1
    NOAUX_TC = 2

    def __str__(self):
        return self.name.lower()


class ExpertAffinityScaleMode(Enum):
    NO_SCALE = 0
    POST_SCALE = 1
    PRE_SCALE = 2
    PRE_SCALE_DELAYED = 3


class QuantizationType(Enum):
    NONE = 0  # No quantization; data remains in native precision (BF16/FP16)
    STATIC = 1  # Per-tensor quantization to FP8 using a single scalar scale (TRN2)
    ROW = 2  # Per-channel quantization to FP8 (TRN2): for a [T, H] tensor, each row is independently
    # scaled along the H dimension by absmax / max_representable_fp8_value
    MX = 3  # Microscaling (MX) quantization (TRN3): along the H dimension, every 32 elements share a micro-scale factor
    STATIC_MX = 4  # Per-tensor quantization in MXFP format (TRN3): FP8/FP4 with a single scalar scale
    # and identity micro-scale factors (127) to satisfy the hardware tensor engine
    ROW_MX = 5  # Per-channel quantization in MXFP format (TRN3): FP8/FP4 with per-row scaling
    # and identity micro-scale factors (127) to satisfy the hardware tensor engine

    def is_mx(self) -> bool:
        """Whether this quantization type uses the MXFP hardware format (TRN3)."""
        return self in (QuantizationType.MX, QuantizationType.STATIC_MX, QuantizationType.ROW_MX)

    def is_logical_static(self) -> bool:
        """Whether this quantization type implements static per-tensor quantization."""
        return self in (QuantizationType.STATIC, QuantizationType.STATIC_MX)

    def is_logical_row(self) -> bool:
        """Whether this quantization type implements dynamic per-channel quantization."""
        return self in (QuantizationType.ROW, QuantizationType.ROW_MX)


class DtypeMode(Enum):
    """FP8 E4M3 dtype variant selection for quantized kernels.

    NON_OCP: nl.float8_e4m3 (max=240, with NaN). Default.
    OCP:     nl.float8_e4m3fn (max=448, no NaN). OCP finite variant.
    AUTO:    Resolve by hardware at kernel trace time via resolve_dtype_to_nki.
             OCP when supported, NON_OCP otherwise.
    """

    NON_OCP = "non_ocp"
    OCP = "ocp"
    AUTO = "auto"


class ComputationMode(Enum):
    AUTO = 0
    PREFILL = 1
    DECODE = 2


class QKVWeightLayout(Enum):
    """Layout of fused QKV weight tensor passed to the kernel.

    The kernel requires weights in a specific layout depending on quantization
    type and whether the DMA transpose optimization is enabled. Starting from a
    standard float checkpoint with shape [H, I] (I = fused QKV output dim):

    CONTIGUOUS (non-MX):
        Use the checkpoint weights as-is in [H, I] layout.

    MX_CONTIGUOUS (MX/ROW_MX):
        Group every 4 consecutive H rows as the innermost dimension:
            w.reshape(H//4, 4, I).transpose(0, 2, 1)
        Result shape: [H//4, I, 4] in fp8 dtype.

    MX_INTERLEAVED (MX with DMA transpose — FP8 or BF16+static-dequant input):
        Starting from [H, I] fp8 weights, reorder rows so that the quads
        match the interleaved layout produced by DMA transpose on the input:
            h_idx = np.arange(H).reshape(2, H//4, 2).transpose(1, 0, 2).reshape(H)
            w_reordered = w[h_idx, :]
        then group into quads:
            w_reordered.reshape(H//4, 4, I).transpose(0, 2, 1)
        Result shape: [H//4, I, 4] in fp8 dtype.
    """

    CONTIGUOUS = 0
    MX_CONTIGUOUS = 1
    MX_INTERLEAVED = 2


class GateUpDim(Enum):
    GATE = 0
    UP = 1


class MLPGateUpWeightLayout(Enum):
    """Layout of gate and up projection weights passed to the kernel.

    The kernel requires different weight layouts depending on whether or not
    MX quantization is enabled. If it is enabled, it provides two alternative
    functional layouts.

    CONTIGUOUS:
        Use the checkpoint weights as-is in [H, I] layout. This layout should be
        used for non-MX flows.

    H_X4_INNERMOST:
        This layout should be used for MX quantized flows with fp8 weights. It is
        most optimal when the hidden tensor is pre-quantized before it is passed
        to the MLP kernel.

        Starting from MLPGateUpWeightLayout.CONTIGUOUS, perform this transformation:
            w.reshape(
                H/512, 128, 4, I/512, 128, 4
            ).transpose(
                1, 0, 3, 5, 4, 2
            )
        Both H and I will need to be padded up to the nearest 512.
        The final shape will be [128, H/512, I/512, 4, 128, 4]

    H_X4_MIDDLE:
        This layout should be used for MX quantized flows with fp8 weights. It is
        most optimal when the hidden tensor is not quantized when it is passed to the
        kernel. The quantization step will be fused to the start of the kernel.

        Starting from MLPGateUpWeightLayout.CONTIGUOUS, perform this transformation:
            w.reshape(
                H/512, 4, 128, I/512, 128, 4
            ).transpose(
                2, 0, 3, 5, 4, 1
            )
        Both H and I will need to be padded up to the nearest 512.
        The final shape will be [128, H/512, I/512, 4, 128, 4]
    """

    CONTIGUOUS = 0
    H_X4_INNERMOST = 1
    H_X4_MIDDLE = 2


class OProjAttentionLayout(Enum):
    """Layout of the attention tensor passed to the output projection CTE kernel.

    BNdS:
        [B, N, D, S]. Attention is in transposed layout.

    BSNd:
        [B, S, N, D], Attention is in native untransposed layout.

    BNSd:
        [B, N, S, D], Attention is untransposed with heads outer. This is the
        attention CTE output when heads are folded into the batch dimension and
        ``tp_out=False``, i.e. [B * N, S, D] viewed as 4D.
    """

    BNdS = 0
    BSNd = 1
    BNSd = 2


class HiddenLayout(Enum):
    """Layout of hidden activations in SBUF for TKG kernels."""

    H0_T_H1 = 0  # [H0, T, H1]
    H0_H1_T = 1  # [H0, H1, T]

    def get_t_dim(self):
        return 1 if self == HiddenLayout.H0_T_H1 else 2

    def get_h1_dim(self):
        return 2 if self == HiddenLayout.H0_T_H1 else 1


class MoELNCShardingStrategy(Enum):
    """LNC (Logical NeuronCore) sharding strategies for MoE kernels.

    Defines how computation is distributed across NeuronCores when LNC=2.
    Not all strategies are supported by all MoE kernels - check kernel documentation
    or the SUPPORTED_MOE_SHARDING_STRATEGIES list for kernel-specific support.
    """

    NO_SHARD = 0  # No sharding: each NC computes full result independently
    SHARD_I = 1  # Shard on I (intermediate) dimension - default for most workloads
    SHARD_T = 2  # Shard on T (token) dimension - useful when T is large
    SHARD_E = 3  # Future: Shard on E (expert) dimension


class MoEAllToAllVStrategy(Enum):
    """MoE all_to_all_v (A2A-v) strategy for MoE kernels.

    Defines how the MoE kernel processes hidden input and MoE output.
    Not all strategies are supported by all MoE kernels.
    """

    DISABLED = 0  # A2A-v not used
    PRESERVE_ROW_ORDER = 1  # A2A-v used; output row ordering matches input row ordering.
    PACK_OUTPUT_ROWS = 2  # A2A-v used; output rows are packed, with routed tokens placed in the first N rows, where N is the number of routed tokens.


class MoEBlockIOLayout(Enum):
    """I/O tensor layout for MoE Block TKG kernel.

    B_S_H: Standard [B, S, H] layout (input) / [T, H] layout (output).
    _128_Nprgs_Hfree_T: [128, n_prgs, H//128//n_prgs, T] layout in HBM.
        Avoids intermediate layout conversions between transformer layers.
    """

    B_S_H = 0
    _128_Nprgs_Hfree_T = 1


@dataclass
class QKNormConfig(nl.NKIObject):
    """Configuration for per-head QK-norm on Q and K projections.

    Each head group independently specifies which norm to apply (or None to skip).

    Args:
        q_norm: Normalization type for Q heads, or None to skip Q norm.
            Only RMS_NORM is registered by default.
        k_norm: Normalization type for K heads, or None to skip K norm.
            Only RMS_NORM is registered by default.
        eps: Epsilon for numerical stability. Default matches norm_eps used throughout
            the QKV CTE codebase.
        q_gamma_norm_weights: [1, d_head] gamma weights for Q heads, or None for
            pure RMSNorm without affine scale.
        k_gamma_norm_weights: [1, d_head] gamma weights for K heads, or None for
            pure RMSNorm without affine scale.
        q_beta_norm_weights: [1, d_head] beta weights for Q heads (LayerNorm only).
            Not yet implemented.
        k_beta_norm_weights: [1, d_head] beta weights for K heads (LayerNorm only).
            Not yet implemented.
        gamma_fused_in_rope_caches: When True, gamma weights have been pre-multiplied
            into the cos/sin RoPE caches by the caller. The kernel fuses the rsqrt
            into the RoPE multiply instructions, eliminating the per-head gamma multiply.
            Requires fused_rope=True, q/k_gamma_norm_weights=None, and only valid on
            qk_norm_pre_rope (not qk_norm_post_rope). The sin cache must be [B, S, d_head]
            with sin_lo_fused in [:,:,0:d_half] and sin_hi_fused in [:,:,d_half:d_head].
    """

    q_norm: Optional[NormType] = NormType.RMS_NORM
    k_norm: Optional[NormType] = NormType.RMS_NORM
    eps: float = 1e-6
    q_gamma_norm_weights: Optional[nl.ndarray] = None
    k_gamma_norm_weights: Optional[nl.ndarray] = None
    q_beta_norm_weights: Optional[nl.ndarray] = None
    k_beta_norm_weights: Optional[nl.ndarray] = None
    gamma_fused_in_rope_caches: bool = False


@dataclass(frozen=True)
class StridedInputConfig(nl.NKIObject):
    """Gather uniformly-spaced blocks from the input sequence dimension.

    The kernel reads num_local_tokens // block_len blocks from the input,
    starting at token offset block_offset, with block_stride tokens between
    the start of consecutive blocks. Output is written contiguously with
    S = num_local_tokens.

    Requires caller-provided output_hbm (output S != input S).

    block_len must either divide pmax (128) or be a multiple of pmax. Accepted:
    {1, 2, 4, 8, 16, 32, 64, 128, 256, 384, 512, ...}. The kernel also requires
    this shard's S window (S_shard_offset, S_shard) to be block-aligned.

    Visual example (block_len=16, block_stride=64, block_offset=16, num_local_tokens=32):

        Input HBM [S_full=128 tokens]:
        Block:    0       1       2       3       4       5       6       7
        Tokens: [0-15] [16-31] [32-47] [48-63] [64-79] [80-95] [96-111] [112-127]
                         ^^^^^                           ^^^^^
                         OURS                            OURS
                 offset=16 -------- stride=64 ---------->

        Output HBM [num_local_tokens=32]:
        [16-31] [80-95]   <- gathered blocks, written contiguously
    """

    block_len: int  # Contiguous tokens per block
    block_stride: int  # Tokens between start of consecutive owned blocks
    block_offset: int  # Token offset of first owned block
    num_local_tokens: int  # Total tokens to process (= output S dimension)
