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

"""
Input Projection TKG Kernel

This kernel implements the fused block-entry (input) projection operation with
optional residual addition and normalization (RMSNorm/LayerNorm), commonly used
before the attention block in transformer models. The kernel is specifically optimized
for Token Generation (TKG, also known as Decoding) scenarios where batch_size * seqlen
is small.

This kernel is designed with LNC support. When LNC>1, the H dimension is sharded across cores.
Multiple output layouts (BSD, NBSd) are supported to match downstream kernel requirements.

"""

from dataclasses import dataclass
from typing import Optional, Tuple, Union

import nki.isa as nisa
import nki.language as nl

from ...core.subkernels.layernorm_tkg import (
    SHARDING_THRESHOLD as layernorm_sharding_threshold,
)
from ...core.subkernels.layernorm_tkg import layernorm_tkg as _layernorm_tkg
from ...core.subkernels.rmsnorm_tkg import SHARDING_THRESHOLD as rmsnorm_sharding_threshold
from ...core.subkernels.rmsnorm_tkg import rmsnorm_tkg as _rmsnorm_tkg
from ...core.utils.allocator import (
    SbufManager,
    create_auto_alloc_manager,
    sizeinbytes,
)
from ...core.utils.common_types import NormType, QKVOutputLayout, QuantizationType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_max_positive_value_for_dtype, get_verified_program_sharding_info
from ...core.utils.logging import get_logger
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ...core.utils.tiled_range import TiledRange, TiledRangeIterator

P_MAX = 128
F_MAX = 512
NUM_PSUM_BANKS = 8

# Heuristic tile size for H dimension weight loading
H_BLOCK_SIZE = 2048
NUM_TILES_PER_H_BLOCK = H_BLOCK_SIZE // P_MAX

_DGE_MODE_NONE = 3

logger = get_logger("input_projection_tkg")


def input_projection_tkg(
    hidden: nl.NkiTensor,
    proj_w: nl.NkiTensor,
    norm_w: Optional[nl.NkiTensor] = None,
    fused_add: bool = False,
    mlp_prev: Optional[nl.NkiTensor] = None,
    attn_prev: Optional[nl.NkiTensor] = None,
    d_head: Optional[int] = None,
    num_kv_heads: Optional[int] = None,
    num_q_heads: Optional[int] = None,
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD,
    eps: float = 1e-6,
    norm_type: NormType = NormType.RMS_NORM,
    quantization_type: QuantizationType = QuantizationType.NONE,
    proj_w_scale: Optional[nl.NkiTensor] = None,
    proj_in_scale: Optional[nl.NkiTensor] = None,
    output_in_sbuf: bool = False,
    proj_bias: Optional[nl.NkiTensor] = None,
    norm_bias: Optional[nl.NkiTensor] = None,
    hidden_actual: Optional[int] = None,
    sbm: Optional[SbufManager] = None,
    transposed_in: bool = False,
) -> nl.NkiTensor | Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Input Projection Kernel for Token Generation

    This kernel computes the fused input projection operation:
        hidden' = norm(hidden + attn_prev + mlp_prev)  # optional fused add and norm
        output = hidden' @ proj_w + proj_bias
    typically used before the attention block in transformer models.

    This kernel is optimized for Token Generation (aka Decoding) use cases where
    batch_size * seqlen is small. Using this kernel with B*S > 128 may result in
    degraded performance - use the CTE variant for large sequence lengths.

    The kernel supports optional fused residual addition and normalization (RMSNorm/LayerNorm)
    to reduce HBM traffic and improve performance.

    Data Types:
        This kernel supports nl.float32, nl.float16, and nl.bfloat16 data types.

    Dimensions:
        B: Batch size
        S: Sequence length
        H: Hidden dimension size
        I: Fused projection output dimension. For a QKV projection this is
           (Q + K + V) * N * D; any concatenation of linear maps of the same
           input is valid (e.g. q | k | v | out_gate | beta for KDA).
        N: Number of heads (for NBSd output layout)
        D: Head dimension size (for NBSd output layout)

    Args:
        hidden (nl.NkiTensor):
            Input hidden states tensor in HBM or SBUF.
            Shape:
                [B, S, H]         when in HBM
                [H0=128, BxS, H1] when in SBUF
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        proj_w (nl.NkiTensor):
            Fused input projection weight tensor in HBM.
            Shape:
                - QuantizationType.NONE/STATIC/ROW: [H, I]
            Dtype:
                - QuantizationType.NONE: nl.float32, nl.float16, or nl.bfloat16
                - QuantizationType.STATIC: nl.float8_e4m3 or nl.float8_e4m3fn
                - QuantizationType.ROW: nl.float8_e4m3 or nl.float8_e4m3fn
        norm_w (nl.NkiTensor, optional):
            Normalization weight tensor in HBM. Required when norm_type is RMS_NORM or LAYER_NORM.
            Shape:    [1, H]
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        fused_add (bool):
            Enable fused residual addition (hidden + attn_prev + mlp_prev). Default: False.
        mlp_prev (nl.NkiTensor, optional):
            Previous MLP residual tensor in HBM. Required when fused_add is True.
            Shape:    [B, S, H]
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        attn_prev (nl.NkiTensor, optional):
            Previous attention residual tensor in HBM. Required when fused_add is True.
            Shape:    [B, S, H]
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        d_head (int, optional):
            Head dimension size D. Required for static quantization and NBSd and NBdS output layouts.
        num_q_heads : Optional[int], default=None
            Number of query heads (required for FP8 quantization)
        num_kv_heads : Optional[int], default=None
            Number of key/value heads (required for FP8 quantization)
        output_layout (QKVOutputLayout):
            Output tensor layout format. BSD: [B, S, I] or NBSd: [N, B, S, D]. Default: QKVOutputLayout.BSD.
        eps (float):
            Epsilon value to maintain numerical stability in normalization. Default: 1e-6.
        norm_type (NormType):
            Type of normalization to apply (NO_NORM, RMS_NORM, or LAYER_NORM). Default: NormType.RMS_NORM.
        quantization_type (QuantizationType):
            Type of quantization to apply (NONE, ROW, STATIC). MX variants are not supported
            by this kernel. Default: QuantizationType.NONE.
        proj_w_scale (nl.NkiTensor, optional):
            Weight dequantization scale tensor in HBM.
                - QuantizationType.STATIC: [1, 3] or [P_MAX, 3] pre-broadcasted
                - QuantizationType.ROW: [1, I] or [P_MAX, I] pre-broadcasted
            Dtype: nl.float32
        proj_in_scale (nl.NkiTensor, optional):
            Input scale tensor in HBM. Required for STATIC quantization: the input
            quantization and dequantization scale.
            Shape:    [1, 1] or [P_MAX, 1] pre-broadcasted
            Dtype: nl.float32
        output_in_sbuf (bool):
            If True, output is kept in SBUF; otherwise stored to HBM. Default: False.
        proj_bias (nl.NkiTensor, optional):
            Bias tensor in HBM for the input projection.
            Shape:    [1, I]
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        norm_bias (nl.NkiTensor, optional):
            LayerNorm beta parameter tensor in HBM. Required when norm_type is LAYER_NORM.
            Shape:    [1, H]
            Dtype: nl.float32, nl.float16, or nl.bfloat16
        hidden_actual (int, optional):
            Actual hidden dimension for padded input tensors. If specified, normalization
            uses this value instead of H for mean calculation.
        sbm (SbufManager, optional):
            Instance of SbufManager responsible for handling SBUF allocation.
            If None, auto-allocation manager is created.
        transposed_in (bool):
            When True, input is in transposed HBM layout [H0, n_prgs, H1_shard, BxS].
            The kernel loads the per-NC shard, permutes to [H0, BxS, H1_shard], applies
            shard_on_h RMSNorm if norm_type is RMS_NORM, and returns the per-shard result
            for the input projection. Default: False.
        output (nl.NkiTensor | Tuple[nl.NkiTensor, nl.NkiTensor]):
            Input projection output tensor. The tensor can reside in either SBUF or HBM.
            Shape:    [B, S, I] for BSD layout, [N, B, S, D] for NBSd layout.
            When fused_add is True, returns tuple (output, fused_hidden) where
            fused_hidden is the result of the fused residual addition.

    Notes:
        - H must be divisible by 128 (nl.tile_size.pmax).
        - H1 (H//128) must be divisible by number of shards for multi-core execution.

    Pseudocode:
        # Optional fused residual add
        if fused_add:
            hidden = hidden + attn_prev + mlp_prev

        # Optional normalization
        if norm_type != NO_NORM:
            hidden = norm(hidden, norm_w, norm_bias, eps)

        # input projection with tiled matmul
        output = zeros(B, S, I)
        for i_block in range(0, I, I_BLOCK_SIZE):
            for h_block in range(0, H_shard, H_BLOCK_SIZE):
                output[:, i_block:i_block+I_BLOCK_SIZE] += hidden[:, h_block:h_block+H_BLOCK_SIZE] @ proj_w[h_block:h_block+H_BLOCK_SIZE, i_block:i_block+I_BLOCK_SIZE]
            output[:, i_block:i_block+I_BLOCK_SIZE] += proj_bias[i_block:i_block+I_BLOCK_SIZE]
    """

    if not sbm:
        sbm = create_auto_alloc_manager()

    sbm.open_scope(name="input_projection_tkg")

    # Validate inputs and create config
    cfg = _validate_and_create_config(
        hidden=hidden,
        proj_w=proj_w,
        proj_bias=proj_bias,
        norm_w=norm_w,
        norm_bias=norm_bias,
        norm_type=norm_type,
        output_layout=output_layout,
        output_in_sbuf=output_in_sbuf,
        fused_add=fused_add,
        attn_prev=attn_prev,
        mlp_prev=mlp_prev,
        quantization_type=quantization_type,
        proj_w_scale=proj_w_scale,
        proj_in_scale=proj_in_scale,
        d_head=d_head,
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        transposed_in=transposed_in,
    )

    io_dtype = hidden.dtype
    quant_dtype = proj_w.dtype if quantization_type != QuantizationType.NONE else None
    if hidden_actual == None:
        hidden_actual = cfg.H

    # Perform optional fused residual add in HBM
    if fused_add:
        hidden = _fused_residual_add_hbm2hbm(
            hidden_hbm=hidden, attn_prev_hbm=attn_prev, mlp_prev_hbm=mlp_prev, cfg=cfg, norm_type=norm_type
        )

    # Load quantization scales
    if quantization_type == QuantizationType.STATIC:
        w_scale_tile = sbm.alloc_stack(
            shape=(P_MAX, 3), dtype=proj_w_scale.dtype, name="proj_w_scale_sb", buffer=nl.sbuf
        )
        if proj_w_scale.shape[0] == 1:
            nisa.dma_copy(dst=w_scale_tile[0, :], src=proj_w_scale[0, :], dge_mode=_DGE_MODE_NONE)
            stream_shuffle_broadcast(w_scale_tile, w_scale_tile)
        else:
            nisa.dma_copy(dst=w_scale_tile, src=proj_w_scale, dge_mode=_DGE_MODE_NONE)

        in_scale_tile = sbm.alloc_heap(shape=(P_MAX, 1), dtype=proj_in_scale.dtype)
        if proj_in_scale.shape[0] == 1:
            nisa.dma_copy(dst=in_scale_tile[0, :], src=proj_in_scale[0, :], dge_mode=_DGE_MODE_NONE)
            stream_shuffle_broadcast(in_scale_tile, in_scale_tile)
        else:
            nisa.dma_copy(dst=in_scale_tile, src=proj_in_scale, dge_mode=_DGE_MODE_NONE)
        nisa.activation(dst=w_scale_tile, op=nl.copy, data=w_scale_tile, scale=in_scale_tile)
        quant_config = StaticQuantConfig(combined_scale_sb=w_scale_tile, in_scale_tile=in_scale_tile)
    elif quantization_type == QuantizationType.ROW:
        quant_config = RowQuantConfig(weight_scale_hbm=proj_w_scale)
    else:
        quant_config = None

    # Perform optional fused norm and load
    hidden_sb = _fused_norm_and_load(
        hidden=hidden,
        norm_type=norm_type,
        norm_w=norm_w,
        norm_bias=norm_bias,
        eps=eps,
        hidden_actual=hidden_actual,
        cfg=cfg,
        sbm=sbm,
        quantization_type=quantization_type,
        quant_config=quant_config,
        quant_dtype=quant_dtype,
        transposed_in=transposed_in,
    )

    # Shard on H for proj_w: (H0, H1_sharded, I)
    proj_w_hbm = proj_w.reshape_dim(dim=0, shape=(cfg.num_shards, cfg.H0, cfg.H1_shard)).select(
        dim=0, index=cfg.shard_id
    )

    # Dispatch to appropriate projection path based output buffer
    if output_in_sbuf:
        output = _projection_sbuf_output(
            hidden_sb=hidden_sb,
            proj_w=proj_w_hbm,
            proj_bias=proj_bias,
            cfg=cfg,
            sbm=sbm,
            io_dtype=io_dtype,
            quantization_type=quantization_type,
            quant_config=quant_config,
        )
    else:
        output = _projection_hbm_output(
            hidden_sb=hidden_sb,
            proj_w=proj_w_hbm,
            proj_bias=proj_bias,
            cfg=cfg,
            output_layout=output_layout,
            sbm=sbm,
            io_dtype=io_dtype,
            quantization_type=quantization_type,
            quant_config=quant_config,
        )

    sbm.close_scope()

    if fused_add:
        return output, hidden
    else:
        return output


@dataclass
class StaticQuantConfig(nl.NKIObject):
    """Configuration for STATIC quantization.

    Holds pre-computed combined scale (weight_scale * input_scale) in SBUF.
    """

    combined_scale_sb: nl.NkiTensor  # [P_MAX, 3], pre-computed in SBUF
    in_scale_tile: nl.NkiTensor  # [P_MAX, 1], input scale in SBUF (for norm quantization)


@dataclass
class RowQuantConfig(nl.NKIObject):
    """Configuration for ROW quantization.

    Holds weight scale tensor in HBM to be loaded inside impl functions.
    """

    weight_scale_hbm: nl.NkiTensor  # [1, I] or [P_MAX, I], in HBM


@dataclass
class InputProjectionTkgConfig(nl.NKIObject):
    """Configuration for the input projection TKG kernel: input dimensions, sharding, and tiling."""

    # Input dimensions
    B: Optional[int]
    S: Optional[int]
    BxS: int
    H: int
    I: int
    H0: int
    H1: int
    d_head: int
    n_q_heads: int
    n_kv_heads: int
    # Sharding
    num_shards: int
    shard_id: int
    H_shard: int
    H1_shard: int
    H1_offset: int
    # Array tiling
    array_tiling_dim: int
    array_tiling_factor: int
    remainder_array_tiling_dim: int
    remainder_array_tiling_factor: int
    array_tiled_H1: int
    remainder_array_tiled_H1: int
    # I dimension tiling
    i_tile_size: int
    i_block_size: int
    # Column tiling strategy
    use_I_column_tiling: bool


def _validate_and_create_config(
    hidden: nl.NkiTensor,
    proj_w: nl.NkiTensor,
    proj_bias: Optional[nl.NkiTensor],
    norm_w: Optional[nl.NkiTensor],
    norm_bias: Optional[nl.NkiTensor],
    norm_type: NormType,
    output_layout: QKVOutputLayout,
    output_in_sbuf: bool,
    fused_add: bool,
    attn_prev: Optional[nl.NkiTensor],
    mlp_prev: Optional[nl.NkiTensor],
    quantization_type: QuantizationType,
    proj_w_scale: Optional[nl.NkiTensor],
    proj_in_scale: Optional[nl.NkiTensor],
    d_head: Optional[int],
    num_q_heads: Optional[int],
    num_kv_heads: Optional[int],
    transposed_in: bool = False,
) -> InputProjectionTkgConfig:
    """
    Validate inputs and create kernel configuration.

    Performs comprehensive validation of input tensor shapes, quantization settings,
    and layout constraints. Computes derived tiling parameters.

    Args:
        hidden: Input hidden states tensor
        proj_w: fused input projection weight tensor
        proj_bias: Optional bias tensor
        norm_w: Optional normalization weight tensor
        norm_bias: Optional normalization bias tensor (LayerNorm only)
        norm_type: Type of normalization
        output_layout: Output tensor layout
        output_in_sbuf: Whether output stays in SBUF
        fused_add: Whether to fuse residual addition
        attn_prev: Previous attention residual (required if fused_add)
        mlp_prev: Previous MLP residual (required if fused_add)
        quantization_type: Quantization mode
        proj_w_scale: Weight scale for quantization
        proj_in_scale: Input scale for quantization
        d_head: Head dimension
        num_q_heads: Number of query heads
        num_kv_heads: Number of key/value heads
        transposed_in: When True, hidden is in transposed HBM layout [H0, n_prgs, H1_shard, BxS]

    Returns:
        InputProjectionTkgConfig with validated configuration and computed tiling parameters
    """
    # Get sharding info
    _, num_shards, shard_id = get_verified_program_sharding_info("input_projection_tkg", (0, 1))

    input_in_sbuf = hidden.buffer == nl.sbuf
    if input_in_sbuf:
        kernel_assert(not fused_add, "fused residual add is only supported when input is in hbm")
        kernel_assert(output_in_sbuf, "sb2hbm is not yet supported for input_projection_tkg")

    # Get input shapes
    if input_in_sbuf:
        H0, BxS, H1 = hidden.shape
        kernel_assert(H0 == nl.tile_size.pmax, f"invalid input shape - H0 (first dimension) must equal 128, got {H0}")
        H = H0 * H1
        B = None
        S = None
    elif transposed_in:
        # Transposed HBM input: [H0, n_prgs, H1_shard, BxS]
        H0 = hidden.shape[0]
        kernel_assert(H0 == nl.tile_size.pmax, f"invalid input shape - H0 (first dimension) must equal 128, got {H0}")
        H = H0 * hidden.shape[1] * hidden.shape[2]
        H1 = H // H0
        BxS = hidden.shape[3]
        B = None
        S = None
    else:
        B, S, H = hidden.shape
        H0 = nl.tile_size.pmax
        H1 = H // H0
        BxS = B * S
    _H, I = proj_w.shape

    # Validate kernel inputs
    kernel_assert(
        H % nl.tile_size.pmax == 0,
        f"H must be divisible by {nl.tile_size.pmax}, got {H} % {nl.tile_size.pmax}={H % nl.tile_size.pmax}",
    )
    kernel_assert(
        _H == H,
        f"Weight tensor reduction dimension must match hidden dimension, got weight H={_H}, hidden H={H}",
    )
    if proj_bias != None:
        kernel_assert(
            proj_bias.shape == (1, I),
            f"Bias shape must be [1, I], got {proj_bias.shape}, expected {(1, I)}",
        )
    kernel_assert(
        norm_type == NormType.LAYER_NORM or not norm_bias,
        f"norm_bias only supported for LAYER_NORM, got norm_type={norm_type} with norm_bias={norm_bias != None}",
    )
    kernel_assert(
        output_layout != QKVOutputLayout.NBdS,
        f"NBdS output layout is not supported in input projection TKG, got output_layout={output_layout}",
    )
    kernel_assert(
        H1 % num_shards == 0,
        f"H1 must be divisible by num_shards, got H={H}, H1={H1}, num_shards={num_shards}, H1 % num_shards={H1 % num_shards}",
    )
    kernel_assert(
        not fused_add or (attn_prev != None and mlp_prev != None),
        f"attn_prev and mlp_prev must be provided when fused_add=True, got fused_add={fused_add}, "
        f"attn_prev provided={attn_prev != None}, mlp_prev provided={mlp_prev != None}",
    )

    # Validate quantization inputs
    kernel_assert(
        quantization_type == QuantizationType.NONE
        or quantization_type == QuantizationType.STATIC
        or quantization_type == QuantizationType.ROW,
        f"Only QuantizationType.NONE, QuantizationType.STATIC, and QuantizationType.ROW are supported, got {quantization_type}",
    )
    if quantization_type == QuantizationType.STATIC:
        kernel_assert(proj_w_scale != None, "proj_w_scales must be provided when quantization_type is STATIC")
        kernel_assert(proj_in_scale != None, "proj_in_scales must be provided when quantization_type is STATIC")
        if proj_w_scale.shape[0] == 1:
            logger.warn(
                f"For static quantization, recommend to pre-broadcast scales to ({nl.tile_size.pmax}, 3) for better performance"
            )
        else:
            kernel_assert(
                proj_w_scale.shape == (nl.tile_size.pmax, 3),
                f"Incorrect shape for qkv weight scale for static per tensor quantization, expected ({nl.tile_size.pmax}, 3), got {proj_w_scale.shape}",
            )
        if proj_in_scale.shape[0] == 1:
            logger.warn(
                f"For static quantization, recommend to pre-broadcast scales to ({nl.tile_size.pmax}, 3) for better performance"
            )
        else:
            kernel_assert(
                proj_in_scale.shape == (nl.tile_size.pmax, 1),
                f"Incorrect shape for qkv input scale for static per tensor quantization, expected ({nl.tile_size.pmax}, 1), got {proj_in_scale.shape}",
            )
        kernel_assert(
            d_head != None and num_kv_heads != None and num_q_heads != None,
            f"d_head, num_q_heads, num_kv_heads must be provided when quant_type is STATIC, got d_head={d_head}, num_q_heads={num_q_heads}, num_kv_heads={num_kv_heads}",
        )
    elif quantization_type == QuantizationType.ROW:
        kernel_assert(proj_w_scale != None, "proj_w_scale must be provided when quantization_type is ROW")
        if proj_w_scale.shape[0] == 1:
            logger.warn(
                f"For row quantization, recommend to pre-broadcast scales to ({nl.tile_size.pmax}, {I}) for better performance"
            )
        else:
            kernel_assert(
                proj_w_scale.shape == (nl.tile_size.pmax, I),
                f"Incorrect shape for qkv weight scale for row quantization, expected ({nl.tile_size.pmax}, {I}), got {proj_w_scale.shape}",
            )

    # Explicit Enum checks
    if norm_type != NormType.NO_NORM and norm_type != NormType.RMS_NORM and norm_type != NormType.LAYER_NORM:
        kernel_assert(False, f"NormType {norm_type} is not supported")
    if output_layout != QKVOutputLayout.BSD and output_layout != QKVOutputLayout.NBSd:
        kernel_assert(False, f"OutputLayout {output_layout} is not supported")

    # Validate HBM output requirements
    if not output_in_sbuf:
        kernel_assert(
            (B != None and S != None) or transposed_in,
            "B and S must be present when output is in HBM (input must be in HBM or transposed_in)",
        )
        if output_layout == QKVOutputLayout.NBSd:
            kernel_assert(
                d_head != None,
                f"d_head must be specified for NBSd output layout, got output_layout={output_layout}, d_head={d_head}",
            )
            kernel_assert(
                I % d_head == 0,
                f"I must be divisible by d_head for NBSd output layout, got I={I}, d_head={d_head}, I % d_head = {I % d_head}",
            )

    # Compute sharding
    H1_sharded = H1 // num_shards
    H1_remainder = H1 % num_shards

    kernel_assert(
        H1_remainder == 0,
        f"H1 must be evenly divisible by num_shards, got H={H}, H1={H1}, num_shards={num_shards}, "
        f"H1 % num_shards = {H1_remainder}",
    )
    if H1_remainder == 0:
        H1_shard = H1_sharded
        H1_offset = shard_id * H1_sharded
    else:
        if shard_id < H1_remainder:
            H1_shard = H1_sharded + 1
            H1_offset = shard_id * (H1_sharded + 1)
        else:
            H1_shard = H1_sharded
            H1_offset = H1_remainder * (H1_sharded + 1) + (shard_id - H1_remainder) * H1_sharded

    H_shard = H1_shard * H0

    # Intermediate dimension tiling
    remainder_H_block = H_shard % H_BLOCK_SIZE
    num_128_tiles_per_remainder_H_block = remainder_H_block // 128

    # Choose Array tiling strategy depending on BxS
    if BxS <= 32:  # do 4x 128P*32F PE array tiling
        array_tiling_dim = 32
    elif BxS <= 64:  # do 2x 128P*64F PE array tiling
        array_tiling_dim = 64
    else:
        array_tiling_dim = 128

    # Adjust hardware-specific logic for column tiling on NeuronCore-v2
    if nisa.get_nc_version() == nisa.nc_version.gen2:
        # Both the row and column sizes in tile_size cannot be 32
        array_tiling_dim = 64

    array_tiling_factor = 128 // array_tiling_dim

    '''
    We have two variants of column-tiling: on H or on I.
    # H-tiling (array tiling): Packs multiple H chunks into one matmul to fill partition rows. Same output column, different partial sums → must reduce.
    # I-tiling (column tiling): Packs multiple output columns into one matmul to fill partition rows. Same input, different outputs → no reduce needed.
    '''
    # For now, only use I_tiling on GPT-OSS config. After more performance testing, it should be enabled in general.
    use_I_column_tiling = False
    if H == 3072:
        use_I_column_tiling = True

    # Only used in case of H-column-tiling.
    array_tiled_H1 = NUM_TILES_PER_H_BLOCK // array_tiling_factor
    # If H is not multiple of H_BLOCK_SIZE and num_128_tiles_per_remainder_H_block is not multiple of array_tiling_factor,
    # kernel won't use array tiling
    remainder_array_tiling_dim = array_tiling_dim
    remainder_array_tiling_factor = array_tiling_factor
    if num_128_tiles_per_remainder_H_block % array_tiling_factor != 0:
        remainder_array_tiling_dim = 128
        remainder_array_tiling_factor = 1
    remainder_array_tiled_H1 = num_128_tiles_per_remainder_H_block // remainder_array_tiling_factor

    if use_I_column_tiling and I <= F_MAX * array_tiling_factor:
        # We are losing on column-tiling when I is small. Use smaller I_tile in this case.
        # In this case moving f-dim will be smaller than F_MAX, and we have more PSUM evitions.
        # However, due to faster matmult perf is still better.
        i_tile_size = I // array_tiling_factor
    else:
        i_tile_size = F_MAX
    i_block_size = NUM_PSUM_BANKS * i_tile_size

    return InputProjectionTkgConfig(
        B=B,
        S=S,
        BxS=BxS,
        H=H,
        I=I,
        H0=H0,
        H1=H1,
        d_head=d_head,
        n_q_heads=num_q_heads,
        n_kv_heads=num_kv_heads,
        num_shards=num_shards,
        shard_id=shard_id,
        H_shard=H_shard,
        H1_shard=H1_shard,
        H1_offset=H1_offset,
        array_tiling_dim=array_tiling_dim,
        array_tiling_factor=array_tiling_factor,
        remainder_array_tiling_dim=remainder_array_tiling_dim,
        remainder_array_tiling_factor=remainder_array_tiling_factor,
        array_tiled_H1=array_tiled_H1,
        remainder_array_tiled_H1=remainder_array_tiled_H1,
        i_tile_size=i_tile_size,
        i_block_size=i_block_size,
        use_I_column_tiling=use_I_column_tiling,
    )


def _fused_residual_add_hbm2hbm(
    hidden_hbm: nl.NkiTensor,
    attn_prev_hbm: nl.NkiTensor,
    mlp_prev_hbm: nl.NkiTensor,
    cfg: InputProjectionTkgConfig,
    norm_type: NormType,
) -> nl.NkiTensor:
    """
    Perform fused residual addition in HBM: hidden + attn_prev + mlp_prev.

    Args:
        hidden_hbm: Hidden states in HBM. Shape: (B, S, H)
        attn_prev_hbm: Previous attention residual in HBM. Shape: (B, S, H)
        mlp_prev_hbm: Previous MLP residual in HBM. Shape: (B, S, H)
        cfg: input projection TKG config
        norm_type: Type of normalization (affects sharding strategy)

    Returns:
        Fused hidden states in HBM. Shape: (B, S, H)
    """

    sharding_threshold = rmsnorm_sharding_threshold if norm_type == NormType.RMS_NORM else layernorm_sharding_threshold

    B, S, BxS, H = cfg.B, cfg.S, cfg.BxS, cfg.H
    num_shards, shard_id = cfg.num_shards, cfg.shard_id

    # Allocate Fused hidden hbm tensor
    if num_shards > 1 and (norm_type == NormType.NO_NORM or BxS > sharding_threshold and norm_type != NormType.NO_NORM):
        fused_hidden = nl.ndarray(
            (BxS, H),
            dtype=hidden_hbm.dtype,
            buffer=nl.shared_hbm,
            name="fused_hidden_shared_hbm",
        )
    else:
        fused_hidden = nl.ndarray((BxS, H), dtype=hidden_hbm.dtype, buffer=nl.shared_hbm, name="fused_hidden_hbm")

    # To prevent non-determinism, a different access pattern is needed for offloaded FMA instruction
    if num_shards > 1 and norm_type == NormType.NO_NORM:
        hidden_hbm = hidden_hbm.reshape((BxS, 2, H // 2))
        attn_prev_hbm = attn_prev_hbm.reshape((BxS, 2, H // 2))
        mlp_prev_hbm = mlp_prev_hbm.reshape((BxS, 2, H // 2))
        fused_hidden = fused_hidden.reshape((BxS, 2, H // 2))
        nisa.dma_compute(
            fused_hidden[0:BxS, shard_id, 0 : (H // 2)],
            (
                hidden_hbm[0:BxS, shard_id, 0 : (H // 2)],
                attn_prev_hbm[0:BxS, shard_id, 0 : (H // 2)],
                mlp_prev_hbm[0:BxS, shard_id, 0 : (H // 2)],
            ),
            scales=(1.0, 1.0, 1.0),
            reduce_op=nl.add,
        )
    elif norm_type != NormType.NO_NORM and num_shards > 1 and BxS > sharding_threshold:
        kernel_assert(
            BxS % 2 == 0,
            f"expected BxS divisible by 2 when BxS={BxS} > {sharding_threshold}",
        )
        hidden_hbm = hidden_hbm.reshape((2, BxS // 2, H))
        attn_prev_hbm = attn_prev_hbm.reshape((2, BxS // 2, H))
        mlp_prev_hbm = mlp_prev_hbm.reshape((2, BxS // 2, H))
        fused_hidden = fused_hidden.reshape((2, BxS // 2, H))
        nisa.dma_compute(
            fused_hidden[shard_id, 0 : (BxS // 2), 0:H],
            (
                hidden_hbm[shard_id, 0 : (BxS // 2), 0:H],
                attn_prev_hbm[shard_id, 0 : (BxS // 2), 0:H],
                mlp_prev_hbm[shard_id, 0 : (BxS // 2), 0:H],
            ),
            scales=(1.0, 1.0, 1.0),
            reduce_op=nl.add,
        )
    else:
        hidden_hbm = hidden_hbm.reshape((1, BxS * H))
        attn_prev_hbm = attn_prev_hbm.reshape((1, BxS * H))
        mlp_prev_hbm = mlp_prev_hbm.reshape((1, BxS * H))
        fused_hidden = fused_hidden.reshape((1, BxS * H))
        nisa.dma_compute(
            fused_hidden,
            (hidden_hbm, attn_prev_hbm, mlp_prev_hbm),
            scales=(1.0, 1.0, 1.0),
            reduce_op=nl.add,
        )

    return fused_hidden.reshape((B, S, H))


def _fused_norm_and_load(
    hidden: nl.NkiTensor,
    norm_type: NormType,
    norm_w: Optional[nl.NkiTensor],
    norm_bias: Optional[nl.NkiTensor],
    eps: float,
    hidden_actual: int,
    cfg: InputProjectionTkgConfig,
    sbm: SbufManager,
    quantization_type: QuantizationType,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]],
    quant_dtype=None,
    transposed_in: bool = False,
) -> nl.NkiTensor:
    """
    Perform fused normalization and load from HBM to SBUF when input is in HBM.

    Handles three normalization paths:
    - NO_NORM:
        Input in HBM:  Direct load with shape (H0, BxS, H1_sharded)
        Input in SBUF: Return as-is with shape (H0, BxS, H1)
    - RMS_NORM: RMSNorm (with HBM -> SBUF load if needed) with shape (H0, BxS, H1)
    - LAYER_NORM: LayerNorm (with HBM -> SBUF load if needed) with shape (H0, BxS, H1)

    Args:
        hidden: Input hidden states in HBM or SBUF.
            Shape:
                (B, S, H)         when in HBM
                (H0=128, BxS, H1) when in SBUF
        norm_type: Type of normalization (NO_NORM, RMS_NORM, or LAYER_NORM)
        norm_w: Normalization weights (required for RMS/LAYER_NORM). Shape: (1, H)
        norm_bias: LayerNorm bias (required for LAYER_NORM). Shape: (1, H)
        eps: Epsilon for numerical stability
        hidden_actual: Actual hidden dimension for padded tensors
        cfg: input projection TKG config
        sbm: SbufManager object for SBUF allocation
        quantization_type: Type of quantization
        quant_config: Quantization config (StaticQuantConfig or RowQuantConfig)
        transposed_in: When True, hidden is in transposed HBM layout [H0, n_prgs, H1_shard, BxS].
            Loads per-NC shard, permutes, applies shard_on_h RMSNorm if enabled, and returns
            the shard directly (H1_shard == H1_sharded).

    Returns:
        nl.NkiTensor wrapping hidden states in SBUF:
          Shape: (H0, BxS, H1_sharded)
    """

    BxS, H0, H1 = cfg.BxS, cfg.H0, cfg.H1
    num_shards, shard_id = cfg.num_shards, cfg.shard_id
    H1_sharded = cfg.H1_shard

    hidden_in_sbuf = hidden.buffer == nl.sbuf

    hidden_sb = None
    hidden_sb_quantized = None

    hidden_shape = (H0, BxS, H1)
    hidden_sharded_shape = (H0, BxS, H1_sharded)

    if quantization_type != QuantizationType.NONE:
        hidden_sb_quantized = sbm.alloc_stack(hidden_sharded_shape, dtype=quant_dtype, buffer=nl.sbuf)

    if transposed_in:
        # Transposed HBM input: [H0, n_prgs, H1_shard, BxS]
        # Load per-NC shard, permute, optionally shard_on_h RMSNorm
        nc_size = H1_sharded * BxS
        nc_offset = shard_id * nc_size
        flat_size = num_shards * nc_size

        # Step 1: Load per-NC shard [H0, H1_sharded*BxS] contiguously from HBM
        x_raw_sb = nl.ndarray((H0, H1_sharded, BxS), dtype=hidden.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=x_raw_sb.reshape((H0, nc_size)),
            src=hidden.reshape((H0, flat_size))[:, nc_offset : nc_offset + nc_size],
        )

        # Step 2: Permute [H0, H1_sharded, BxS] → [H0, BxS, H1_sharded]
        x_shard_sb = nl.ndarray((H0, BxS, H1_sharded), dtype=hidden.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=x_shard_sb, src=x_raw_sb.permute(dims=(0, 2, 1)))

        # Step 3: RMSNorm with shard_on_h (if enabled)
        if norm_type == NormType.RMS_NORM:
            x_normed_sb = nl.ndarray((H0, BxS, H1_sharded), dtype=hidden.dtype, buffer=nl.sbuf)
            _rmsnorm_tkg(
                input=x_shard_sb,
                gamma=norm_w,
                output=x_normed_sb,
                eps=eps,
                hidden_actual=hidden_actual if hidden_actual != None else H0 * num_shards * H1_sharded,
                shard_on_h=True,
                sbm=sbm,
            )
            x_shard_sb = x_normed_sb

        # Step 4: Return shard directly
        hidden_sb = x_shard_sb
    elif norm_type == NormType.NO_NORM:
        if hidden_in_sbuf:
            hidden_sb = hidden.slice(dim=2, start=shard_id * H1_sharded, end=(shard_id + 1) * H1_sharded)
        else:
            if quantization_type != QuantizationType.NONE:
                hidden_sb = sbm.alloc_heap(hidden_sharded_shape, dtype=hidden.dtype, buffer=nl.sbuf)
            else:
                hidden_sb = sbm.alloc_stack(hidden_sharded_shape, dtype=hidden.dtype, buffer=nl.sbuf)
            # Perform direct input load with no norm
            # hidden_sb: (H0, BxS, H1_sharded)
            hidden_sb = _input_load(hidden, hidden_sb, cfg, sbm)
            hidden_sb = hidden_sb
    elif norm_type == NormType.RMS_NORM or norm_type == NormType.LAYER_NORM:
        if hidden_in_sbuf:
            hidden_sb = hidden
        elif quantization_type != QuantizationType.NONE:
            hidden_sb = sbm.alloc_heap(hidden_shape, dtype=hidden.dtype, buffer=nl.sbuf)
        else:
            hidden_sb = sbm.alloc_stack(hidden_shape, dtype=hidden.dtype, buffer=nl.sbuf)
        # Perform norm with load
        if norm_type == NormType.RMS_NORM:
            # Perform rmsnorm with load
            # hidden_sb: (H0, BxS, H1)
            hidden_sb = _rmsnorm_tkg(
                input=hidden,
                gamma=norm_w,
                output=hidden_sb,
                eps=eps,
                hidden_actual=hidden_actual,
                sbm=sbm,
            )
        elif norm_type == NormType.LAYER_NORM:
            # Perform layernorm with load
            # hidden_sb: (H0, BxS, H1)
            hidden_sb = _layernorm_tkg(
                input=hidden,
                gamma=norm_w,
                beta=norm_bias,
                output=hidden_sb,
                eps=eps,
                sbm=sbm,
            )
        hidden_sb = hidden_sb.slice(dim=2, start=shard_id * H1_sharded, end=(shard_id + 1) * H1_sharded)

    # optionally quantize the inputs
    if quantization_type == QuantizationType.STATIC:
        in_scale_tile = quant_config.in_scale_tile
        nisa.reciprocal(dst=in_scale_tile, data=in_scale_tile)
        nisa.activation(dst=hidden_sb, op=nl.copy, data=hidden_sb, scale=in_scale_tile[:H0, :])
        max_pos_val = get_max_positive_value_for_dtype(quant_dtype)
        nisa.tensor_scalar(
            dst=hidden_sb_quantized,
            data=hidden_sb,
            op0=nl.minimum,
            operand0=max_pos_val,
            op1=nl.maximum,
            operand1=-max_pos_val,
        )
        if not hidden_in_sbuf and not transposed_in:
            sbm.pop_heap()  # hidden_sb
        sbm.pop_heap()  # in_scale_tile
        hidden_sb = hidden_sb_quantized

    return hidden_sb


def _input_load(
    hidden_hbm: nl.NkiTensor,
    hidden_sb: nl.NkiTensor,
    cfg: InputProjectionTkgConfig,
    sbm: SbufManager,
) -> nl.NkiTensor:
    """
    Load hidden states from HBM to SBUF without normalization.

    Args:
        hidden_hbm: Hidden states in HBM. Shape: (B, S, H)
        hidden_sb: Hidden states in SBUF to be loaded to. Shape: (H0, BxS, H1_sharded)
        cfg: input projection TKG config
        sbm: SbufManager for SBUF allocation

    Returns:
        Hidden states in SBUF. Shape: (H0, BxS, H1_sharded)
    """
    BxS, H0, H1 = cfg.BxS, cfg.H0, cfg.H1
    num_shards, shard_id = cfg.num_shards, cfg.shard_id
    H1_sharded = cfg.H1_shard

    if num_shards > 1:
        # Reshape (B, S, H) to (BxS, num_shards, H0, H1_sharded), select shard, permute to (H0, BxS, H1_sharded)
        hidden_hbm = hidden_hbm.reshape((BxS, num_shards, H0, H1_sharded))
        hidden_hbm = hidden_hbm.select(dim=1, index=shard_id).permute((1, 0, 2))
        nisa.dma_copy(hidden_sb, hidden_hbm)
    else:
        # Reshape (B, S, H) to (BxS, H0, H1), permute to (H0, BxS, H1)
        hidden_hbm = hidden_hbm.reshape((BxS, H0, H1))
        hidden_hbm = hidden_hbm.permute((1, 0, 2))
        nisa.dma_copy(hidden_sb, hidden_hbm)

    return hidden_sb


def _initialize_proj_out(
    proj_out_sb: nl.NkiTensor,
    proj_bias: Optional[nl.NkiTensor],
    cfg: InputProjectionTkgConfig,
    i_block_idx: int,
    sbm: SbufManager,
) -> None:
    """
    Initialize the projection output buffer with bias or zeros.

    When bias is provided and this is shard_id 0, loads the bias into SBUF
    and broadcasts it across all BxS partitions. Otherwise initializes to zeros.

    Args:
        proj_out_sb: Output buffer to initialize. Shape: (BxS, I_block_size)
        proj_bias: Optional bias tensor. Shape: (1, I_block_size) or None
        cfg: input projection TKG config
        sbm: SbufManager for SBUF allocation
        i_block_idx: I-block index for scope/buffer naming
    """
    if proj_bias != None and cfg.shard_id == 0:
        scope_name = f"qkv_bias_init_block_{i_block_idx}" if i_block_idx != None else "proj_bias_init_block"
        sbm.open_scope(name=scope_name)

        buffer_name = f"proj_bias_sb_{i_block_idx}" if i_block_idx != None else "proj_bias_sb"
        proj_bias_sb = sbm.alloc_stack(
            proj_bias.shape,
            dtype=proj_bias.dtype,
            buffer=nl.sbuf,
            name=buffer_name,
        )
        nisa.dma_copy(proj_bias_sb, proj_bias)
        # Broadcast bias to all BxS partitions
        print(f'input projection pre-applying bias')
        stream_shuffle_broadcast(proj_bias_sb, proj_out_sb)
        sbm.close_scope()
    else:
        nisa.memset(proj_out_sb, value=0)


def _static_dequantize(
    output_sb: nl.NkiTensor,
    dequant_scale_sb: nl.NkiTensor,
    cfg: InputProjectionTkgConfig,
    I_start: int = 0,
):
    pdim, I_size = output_sb.shape
    I_end = I_start + I_size

    # Q heads dequant
    q_start = 0
    q_end = min(I_end, cfg.d_head * cfg.n_q_heads) - I_start
    if q_start < q_end:
        nisa.tensor_scalar(
            dst=output_sb[:pdim, q_start:q_end],
            data=output_sb[:pdim, q_start:q_end],
            op0=nl.multiply,
            operand0=dequant_scale_sb[:pdim, 0:1],
            engine=nisa.vector_engine,
        )

    # K head dequant
    k_start = max(I_start, cfg.d_head * cfg.n_q_heads) - I_start
    k_end = min(I_end, cfg.d_head * (cfg.n_q_heads + cfg.n_kv_heads)) - I_start
    if k_start < k_end:
        nisa.activation(
            dst=output_sb[:pdim, k_start:k_end],
            data=output_sb[:pdim, k_start:k_end],
            op=nl.copy,
            scale=dequant_scale_sb[:pdim, 1:2],
        )
    # V head dequant
    v_start = max(I_start, cfg.d_head * (cfg.n_q_heads + cfg.n_kv_heads)) - I_start
    v_end = min(I_end, cfg.d_head * (cfg.n_q_heads + 2 * cfg.n_kv_heads)) - I_start
    if v_start < v_end:
        nisa.activation(
            dst=output_sb[:pdim, v_start:v_end],
            data=output_sb[:pdim, v_start:v_end],
            op=nl.copy,
            scale=dequant_scale_sb[:pdim, 2:3],
        )
    return output_sb


def _row_dequantize(
    output_sb: nl.NkiTensor,
    weight_scale: nl.NkiTensor,
    cfg: InputProjectionTkgConfig,
    sbm: SbufManager,
) -> None:
    """
    Apply row dequantization to the projection output.

    Loads weight scale from HBM, broadcasts if needed, and applies element-wise multiply.

    Args:
        output_sb: projection output in SBUF. Shape: (BxS, i_size)
        weight_scale: Pre-sliced weight scale nl.NkiTensor in HBM. Shape: (P_MAX, i_size) or (1, i_size)
        cfg: input projection TKG config
        sbm: SbufManager for allocation
    """
    BxS = cfg.BxS
    i_size = output_sb.shape[1]

    weight_scale_sb = sbm.alloc_heap((P_MAX, i_size), dtype=weight_scale.dtype, buffer=nl.sbuf)
    if weight_scale.shape[0] == 1:
        nisa.dma_copy(dst=weight_scale_sb[0:1, :], src=weight_scale)
        stream_shuffle_broadcast(weight_scale_sb, weight_scale_sb)
    else:
        nisa.dma_copy(dst=weight_scale_sb, src=weight_scale)
    nisa.tensor_tensor(dst=output_sb, data1=output_sb, data2=weight_scale_sb[:BxS, :], op=nl.multiply)
    sbm.pop_heap()  # weight_scale_sb


def _apply_bias(
    output_sb: nl.NkiTensor,
    proj_bias: nl.NkiTensor,
    sbm: SbufManager,
) -> None:
    """
    Apply bias to the projection output.

    Loads bias from HBM, broadcasts to all BxS partitions, and adds to output.

    Args:
        output_sb: projection output in SBUF. Shape: (BxS, i_size)
        proj_bias: Pre-sliced bias nl.NkiTensor in HBM. Shape: (1, i_size)
        sbm: SbufManager for allocation
    """
    proj_bias_sb = sbm.alloc_heap(output_sb.shape, dtype=proj_bias.dtype, buffer=nl.sbuf)
    # DMA based broacast is usually faster, especially for high batch.
    use_stream_shuffle_bias_broadcast = False
    if use_stream_shuffle_bias_broadcast:
        nisa.dma_copy(proj_bias_sb[0:1, :], proj_bias)
        stream_shuffle_broadcast(proj_bias_sb, proj_bias_sb)
    else:
        # Single DMA that broadcasts the bias (HBM, shape (1, i_size)) across all
        # BxS partitions by setting the partition-dim stride to 0 on the source.
        BxS = output_sb.shape[0]
        proj_bias_bcast = proj_bias.broadcast(dim=0, size=BxS)
        nisa.dma_copy(proj_bias_sb, proj_bias_bcast)

    nisa.tensor_tensor(dst=output_sb, data1=output_sb, data2=proj_bias_sb, op=nl.add)
    sbm.pop_heap()  # proj_bias_sb


def _compute_proj_i_block(
    hidden_sb: nl.NkiTensor,
    proj_w: nl.NkiTensor,
    proj_bias: nl.NkiTensor,
    proj_out_sb: nl.NkiTensor,
    i_block: TiledRangeIterator,
    cfg: InputProjectionTkgConfig,
    sbm: SbufManager,
    quantization_type: QuantizationType,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]],
) -> nl.NkiTensor:
    """
    Compute a single I-block of the input projection: bias init, matmul, optional dequant, and cross-core reduce.

    Args:
        hidden_sb: Input hidden states in SBUF (nl.NkiTensor). Shape: (H0, BxS, H1_sharded)
        proj_w: fused input projection weights (nl.NkiTensor). Shape: (H0, H1_sharded, I)
        proj_bias: Optional bias in HBM. Shape: (1, I) or None
        proj_out_sb: Pre-allocated SBUF output for this I-block. Shape: (BxS, i_block.size)
        cfg: input projection TKG config
        i_block: Current I-block from TiledRange
        sbm: SbufManager for SBUF allocation
        quantization_type: Quantization mode
        quant_config: Quantization config (StaticQuantConfig or RowQuantConfig)

    Returns:
        Computed output in SBUF. Shape: (BxS, i_block.size)
    """
    num_shards, shard_id = cfg.num_shards, cfg.shard_id

    # Create bias slice for this I block if bias exists for use in initializing qkv out.
    # When dtypes differ cannot preapply so skip this step and apply after projection.
    preapply_bias = (
        quantization_type == QuantizationType.NONE and proj_bias != None and proj_bias.dtype == proj_out_sb.dtype
    )

    # Not pre-applying bias is slightly efficient if we use I_column_tiling.
    # In this case no reductoin is needed post-matmult, so we can engine balance tensor_copy and apply bias later.
    if cfg.use_I_column_tiling and cfg.array_tiling_factor == 4:
        preapply_bias = False

    if preapply_bias:
        proj_bias_block = proj_bias[:, i_block.start_offset : i_block.end_offset]
    else:
        proj_bias_block = None

    _initialize_proj_out(proj_out_sb, proj_bias_block, cfg, i_block.index, sbm)

    # Slice proj_w for this I block
    proj_w_block = proj_w.slice(dim=2, start=i_block.start_offset, end=i_block.end_offset)

    # Perform the input projection for this I block
    # output_sb: (BxS, i_block.size)
    output_sb = _projection(
        hidden_sb=hidden_sb,
        proj_w_hbm=proj_w_block,
        proj_out_sb=proj_out_sb,
        cfg=cfg,
        i_block_idx=i_block.index,
        sbm=sbm,
        has_preapplied_bias=preapply_bias,
    )

    if quantization_type == QuantizationType.STATIC:
        output_sb = _static_dequantize(output_sb, quant_config.combined_scale_sb, cfg, I_start=i_block.start_offset)
    elif quantization_type == QuantizationType.ROW:
        weight_scale_block = quant_config.weight_scale_hbm.slice(
            dim=1, start=i_block.start_offset, end=i_block.end_offset
        )
        _row_dequantize(output_sb, weight_scale_block, cfg, sbm)

    if not preapply_bias and proj_bias != None and cfg.shard_id == 0:
        proj_bias_block = proj_bias.slice(dim=1, start=i_block.start_offset, end=i_block.end_offset)
        _apply_bias(output_sb, proj_bias_block, sbm)

    # Receive qkv projection output from the other neuron core when LNC > 1
    if num_shards > 1:
        sbm.open_scope(name=f"output_store_sendrecv_block_{i_block.index}")
        proj_recv = sbm.alloc_stack((cfg.BxS, i_block.size), dtype=output_sb.dtype, buffer=nl.sbuf)
        other_core = 1 - shard_id
        nisa.sendrecv(
            src=output_sb,
            dst=proj_recv,
            send_to_rank=other_core,
            recv_from_rank=other_core,
            pipe_id=0,
        )
        nisa.tensor_tensor(output_sb, output_sb, proj_recv, op=nl.add)
        sbm.close_scope()

    return output_sb


def _projection_sbuf_output(
    hidden_sb: nl.NkiTensor,
    proj_w: nl.NkiTensor,
    proj_bias: nl.NkiTensor,
    cfg: InputProjectionTkgConfig,
    sbm: SbufManager,
    io_dtype,
    quantization_type: QuantizationType = QuantizationType.NONE,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]] = None,
) -> nl.NkiTensor:
    """
    Input projection with SBUF output (output_in_sbuf=True path).

    Loops over I-blocks (each up to I_BLOCK_SIZE) with output kept in SBUF.
    Includes neuron core cross-communication when sharded.

    Args:
        hidden_sb: Input hidden states in SBUF (nl.NkiTensor). Shape: (H0, BxS, H1_sharded)
        proj_w: fused input projection weights (nl.NkiTensor). Shape: (H0, H1_sharded, I)
        proj_bias: Optional bias in HBM. Shape: (1, I)
        cfg: input projection TKG config
        sbm: SbufManager for SBUF allocation
        io_dtype: Data type for input/output tensors
        quantization_type: Quantization mode
        quant_config: Quantization config (StaticQuantConfig or RowQuantConfig)

    Returns:
        Projection output in SBUF. Shape: (BxS, I)
    """

    BxS, I = cfg.BxS, cfg.I

    # Allocate full (BxS, I) output on heap — persists for caller
    proj_out_sb = sbm.alloc_heap((BxS, I), dtype=io_dtype, buffer=nl.sbuf)

    # Process each I-block
    for i_block in TiledRange(I, cfg.i_block_size):
        sbm.open_scope(name=f"proj_sbuf_output_i_block_{i_block.index}")

        # Output slice for this I-block
        output_slice = proj_out_sb[:, i_block.start_offset : i_block.end_offset]

        _compute_proj_i_block(
            hidden_sb, proj_w, proj_bias, output_slice, i_block, cfg, sbm, quantization_type, quant_config
        )

        sbm.close_scope()

    return proj_out_sb


def _projection_hbm_output(
    hidden_sb: nl.NkiTensor,
    proj_w: nl.NkiTensor,
    proj_bias: nl.NkiTensor,
    cfg: InputProjectionTkgConfig,
    output_layout: QKVOutputLayout,
    sbm: SbufManager,
    io_dtype,
    quantization_type: QuantizationType = QuantizationType.NONE,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]] = None,
) -> nl.NkiTensor:
    """
    Input projection with HBM output (output_in_sbuf=False path).

    Handles multiple I-blocks when I > I_BLOCK_SIZE. Each block is computed
    in SBUF then stored to HBM with layout-specific transformation.
    Includes neuron core cross-communication when sharded.

    Args:
        hidden_sb: Input hidden states in SBUF (nl.NkiTensor). Shape: (H0, BxS, H1_sharded)
        proj_w: fused input projection weights (nl.NkiTensor). Shape: (H0, H1_sharded, I)
        proj_bias: Optional bias in HBM. Shape: (1, I) or None
        cfg: input projection TKG config
        output_layout: Target layout (BSD or NBSd)
        sbm: SbufManager for SBUF allocation
        io_dtype: Dtype of the input hidden, which should also be the output dtype
        quant_config: Quantization config (StaticQuantConfig or RowQuantConfig)

    Returns:
        Projection output tensor. Shape depends on output_layout:
        - BSD: (B, S, I)
        - NBSd: (num_heads, B, S, d_head)
    """

    B, S, BxS, I = cfg.B, cfg.S, cfg.BxS, cfg.I

    # Allocate output tensor with layout-specific shape
    if output_layout == QKVOutputLayout.BSD:
        output = nl.ndarray(
            (BxS, I), dtype=io_dtype, buffer=nl.shared_hbm, name=f"{sbm.get_name_prefix()}proj_output_bsd"
        )
    elif output_layout == QKVOutputLayout.NBSd:
        nh = I // cfg.d_head
        output = nl.ndarray(
            (nh, BxS, cfg.d_head), dtype=io_dtype, buffer=nl.shared_hbm, name=f"{sbm.get_name_prefix()}proj_output_nbsd"
        )

    # Process each I block
    for i_block in TiledRange(I, cfg.i_block_size):
        sbm.open_scope(name=f"proj_hbm_output_i_block_{i_block.index}")

        # Allocate output SB that gets accumulated in HBM
        proj_out_sb = sbm.alloc_stack((BxS, i_block.size), dtype=io_dtype, buffer=nl.sbuf)

        output_sb = _compute_proj_i_block(
            hidden_sb, proj_w, proj_bias, proj_out_sb, i_block, cfg, sbm, quantization_type, quant_config
        )

        # Store to HBM with layout-specific transformation
        _store_projection_output_to_hbm(
            output_hbm=output,
            output_sb=output_sb,
            I_block_start=i_block.start_offset,
            I_block_end=i_block.end_offset,
            output_layout=output_layout,
            cfg=cfg,
        )

        sbm.close_scope()

    # Reshape to expected output shape (skip when B/S are None, e.g. transposed_in)
    if B != None and S != None:
        if output_layout == QKVOutputLayout.BSD:
            output = output.reshape((B, S, I))
        elif output_layout == QKVOutputLayout.NBSd:
            n_heads = I // cfg.d_head
            output = output.reshape((n_heads, B, S, cfg.d_head))

    # Return output in HBM
    return output


def _store_projection_output_to_hbm(
    output_hbm: nl.NkiTensor,
    output_sb: nl.NkiTensor,
    I_block_start: int,
    I_block_end: int,
    output_layout: QKVOutputLayout,
    cfg: InputProjectionTkgConfig,
) -> None:
    """
    Store the projection output from SBUF to HBM with layout-specific transformation.

    Handles two output layouts:
    - BSD: (Batch, Seqlen, fused_proj_dim) -> direct 2D copy
    - NBSd: (num_heads, Batch, Seqlen, d_head) -> reshape with head dimension split

    Args:
        output_hbm: Destination HBM tensor
                    Shape: (BxS, I) for BSD layout
                           (num_heads, BxS, d_head) for NBSd layout
        output_sb: Source SBUF tensor with shape (BxS, I_block_size)
        I_block_start: Starting index in I dimension for this block
        I_block_end: Ending index in I dimension for this block (exclusive)
        output_layout: Target layout (BSD or NBSd)
        cfg: input projection TKG config
    """
    if output_layout == QKVOutputLayout.BSD:
        nisa.dma_copy(output_hbm[:, I_block_start:I_block_end], output_sb)

    elif output_layout == QKVOutputLayout.NBSd:
        # TODO: Change to nl.NkiTensor
        I_block_size = I_block_end - I_block_start
        ns = I_block_start // cfg.d_head
        ne = I_block_end // cfg.d_head

        for i_n in range(ns, ne):
            output_pattern = [[cfg.d_head, cfg.BxS], [1, cfg.d_head]]
            output_offset = i_n * cfg.BxS * cfg.d_head
            output_sb_pattern = [[I_block_size, cfg.BxS], [1, cfg.d_head]]
            output_sb_offset = (i_n - ns) * cfg.d_head
            nisa.dma_copy(
                output_hbm.ap(pattern=output_pattern, offset=output_offset),
                output_sb.ap(pattern=output_sb_pattern, offset=output_sb_offset),
            )


def _projection(
    hidden_sb: nl.NkiTensor,
    proj_w_hbm: nl.NkiTensor,
    proj_out_sb: nl.NkiTensor,
    cfg: InputProjectionTkgConfig,
    i_block_idx: int,
    sbm: SbufManager,
    has_preapplied_bias: bool = False,
) -> nl.NkiTensor:
    # Note: "column_tiling" here refers to special perf-mode of nc_matmult, that allows us fill PE-array with multiple tiles.
    if cfg.use_I_column_tiling:
        return _projection_I_column_tiled(
            hidden_sb, proj_w_hbm, proj_out_sb, cfg, i_block_idx, sbm, has_preapplied_bias
        )
    else:
        return _projection_H_column_tiled(
            hidden_sb, proj_w_hbm, proj_out_sb, cfg, i_block_idx, sbm, has_preapplied_bias
        )


def _projection_H_column_tiled(
    hidden_sb: nl.NkiTensor,
    proj_w_hbm: nl.NkiTensor,
    proj_out_sb: nl.NkiTensor,
    cfg: InputProjectionTkgConfig,
    i_block_idx: int,
    sbm: SbufManager,
    has_preapplied_bias: bool = False,
) -> nl.NkiTensor:
    """Original array-tiling path: tiles H chunks across partition rows, requires reduction."""
    _, _, I = proj_w_hbm.shape
    output_dtype = hidden_sb.dtype
    weight_dtype = proj_w_hbm.dtype

    sbm.open_scope(name=f"projection_block_{i_block_idx}")

    # Reserve space for post-projection buffers (bias broadcast, sendrecv) that will be
    # allocated at the same scope level after the projection scope closes.
    # get_free_space() already accounts for all prior allocations (stack and heap),
    # but not for these future sibling-scope allocations.
    extra_space_needed = sizeinbytes(nl.float32)
    for i in range(1, len(proj_out_sb.shape)):
        extra_space_needed *= proj_out_sb.shape[i]

    remaining_space = sbm.get_free_space() - extra_space_needed
    size_of_proj_w_block = I * NUM_TILES_PER_H_BLOCK * sizeinbytes(weight_dtype)
    num_available_w_blocks = remaining_space // size_of_proj_w_block
    num_H_blocks = div_ceil(cfg.H_shard, H_BLOCK_SIZE)
    num_w_blocks = min(num_H_blocks, num_available_w_blocks)
    # With auto_alloc, remaining_space underestimates available memory due to automatic reuse,
    # so ensure at least one tile can be allocate
    if sbm.is_auto_alloc():
        num_w_blocks = max(1, num_w_blocks)
    kernel_assert(num_w_blocks > 0, f"Not enough SBUF space for qkv projection weight")

    # Allocate weight tiles
    proj_w_sb = sbm.alloc_stack(
        (cfg.H0, num_w_blocks, NUM_TILES_PER_H_BLOCK, I),
        name=f"proj_w_sb_block_{i_block_idx}",
        dtype=weight_dtype,
        buffer=nl.sbuf,
    )
    proj_w_sb = proj_w_sb

    # Allocate PSUM tiles - one per I tile in this I-block
    n_psum = div_ceil(I, cfg.i_tile_size)
    result_psum = []
    prefix = sbm.get_name_prefix()
    for psum_idx in range(n_psum):
        psum_tensor = nl.ndarray(
            (128, cfg.i_tile_size),
            dtype=nl.float32,
            name=f"{prefix}batch_result_psum_{i_block_idx}_{psum_idx}",
            buffer=nl.psum,
            address=None
            if sbm.is_auto_alloc()
            else (0, (psum_idx % NUM_PSUM_BANKS) * (F_MAX * sizeinbytes(nl.float32))),
        )
        result_psum.append(psum_tensor)

    # Process all H blocks with array tiling
    for h_block in TiledRange(cfg.H1_shard, NUM_TILES_PER_H_BLOCK):
        is_remainder = h_block.size < NUM_TILES_PER_H_BLOCK
        hidden_block = hidden_sb.slice(dim=2, start=h_block.start_offset, end=h_block.end_offset)
        proj_w_block = proj_w_hbm.slice(dim=1, start=h_block.start_offset, end=h_block.end_offset)

        array_tiling_dim = cfg.remainder_array_tiling_dim if is_remainder else cfg.array_tiling_dim
        array_tiling_factor = cfg.remainder_array_tiling_factor if is_remainder else cfg.array_tiling_factor
        array_tiled_H1 = cfg.remainder_array_tiled_H1 if is_remainder else cfg.array_tiled_H1

        w_block_slot = h_block.index % num_w_blocks
        proj_w_sb_block = proj_w_sb.select(dim=1, index=w_block_slot).slice(dim=1, start=0, end=h_block.size)
        nisa.dma_copy(proj_w_sb_block, proj_w_block)

        for h1_tile in range(array_tiled_H1):
            array_tile_offset = array_tiling_factor * h1_tile
            for factor in range(array_tiling_factor):
                h1_tile_idx = array_tile_offset + factor
                hidden_tile = hidden_block.select(dim=2, index=h1_tile_idx)

                for i_tile in TiledRange(I, cfg.i_tile_size):
                    proj_w_sb_tile = proj_w_sb_block.select(dim=1, index=h1_tile_idx).slice(
                        dim=1, start=i_tile.start_offset, end=i_tile.end_offset
                    )
                    psum_row_start = array_tiling_dim * factor
                    result_slice = result_psum[i_tile.index][psum_row_start : psum_row_start + cfg.BxS, 0 : i_tile.size]
                    nisa.nc_matmul(
                        result_slice,
                        hidden_tile,
                        proj_w_sb_tile,
                        tile_position=(0, array_tiling_dim * factor),
                        tile_size=(cfg.H0, array_tiling_dim),
                    )

    # Accumulate: reduce array-tiled PSUM partitions
    _i_array_tiling_factor = cfg.array_tiling_factor
    _i_array_tiling_dim = cfg.array_tiling_dim
    has_only_remainder_H_block = cfg.H_shard < H_BLOCK_SIZE
    if has_only_remainder_H_block:
        _i_array_tiling_factor = cfg.remainder_array_tiling_factor
        _i_array_tiling_dim = cfg.remainder_array_tiling_dim

    for i_tile in TiledRange(I, cfg.i_tile_size):
        for factor in range(_i_array_tiling_factor):
            result_psum_slice_start = _i_array_tiling_dim * factor
            nisa.tensor_tensor(
                proj_out_sb[0 : cfg.BxS, i_tile.start_offset : i_tile.end_offset],
                proj_out_sb[0 : cfg.BxS, i_tile.start_offset : i_tile.end_offset],
                result_psum[i_tile.index][result_psum_slice_start : result_psum_slice_start + cfg.BxS, 0 : i_tile.size],
                op=nl.add,
            )

    sbm.close_scope()
    return proj_out_sb


def _projection_I_column_tiled(
    hidden_sb: nl.NkiTensor,
    proj_w_hbm: nl.NkiTensor,
    proj_out_sb: nl.NkiTensor,
    cfg: InputProjectionTkgConfig,
    i_block_idx: int,
    sbm: SbufManager,
    has_preapplied_bias: bool = False,
) -> nl.NkiTensor:
    """I-column-tiling path: tiles I across partition rows, no reduction needed."""
    _, _, I = proj_w_hbm.shape
    output_dtype = hidden_sb.dtype
    weight_dtype = proj_w_hbm.dtype

    sbm.open_scope(name=f"projection_block_{i_block_idx}")

    col_tiling_dim = cfg.array_tiling_dim
    col_tiling_factor = cfg.array_tiling_factor

    # Reserve space for post-projection buffers
    extra_space_needed = sizeinbytes(nl.float32)
    for i in range(1, len(proj_out_sb.shape)):
        extra_space_needed *= proj_out_sb.shape[i]

    remaining_space = sbm.get_free_space() - extra_space_needed
    size_of_proj_w_block = I * NUM_TILES_PER_H_BLOCK * sizeinbytes(weight_dtype)
    num_available_w_blocks = remaining_space // size_of_proj_w_block
    num_H_blocks = div_ceil(cfg.H_shard, H_BLOCK_SIZE)
    num_w_blocks = min(num_H_blocks, num_available_w_blocks)
    if sbm.is_auto_alloc():
        num_w_blocks = max(1, num_w_blocks)
    kernel_assert(num_w_blocks > 0, f"Not enough SBUF space for qkv projection weight")

    proj_w_sb = sbm.alloc_stack(
        (cfg.H0, num_w_blocks, NUM_TILES_PER_H_BLOCK, I),
        name=f"proj_w_sb_block_{i_block_idx}",
        dtype=weight_dtype,
        buffer=nl.sbuf,
    )
    proj_w_sb = proj_w_sb

    # Allocate PSUM - one per group of col_tiling_factor I tiles
    n_psum = div_ceil(I, cfg.i_tile_size * col_tiling_factor)
    psum_pdim = min(128, col_tiling_dim * col_tiling_factor)
    result_psum = []
    prefix = sbm.get_name_prefix()
    for psum_idx in range(n_psum):
        psum_tensor = nl.ndarray(
            (psum_pdim, cfg.i_tile_size),
            dtype=nl.float32,
            name=f"{prefix}batch_result_psum_{i_block_idx}_{psum_idx}",
            buffer=nl.psum,
            address=None
            if sbm.is_auto_alloc()
            else (0, (psum_idx % NUM_PSUM_BANKS) * (F_MAX * sizeinbytes(nl.float32))),
        )
        result_psum.append(psum_tensor)

    # Matmul: column tiling on I with all H1 tiles accumulating naturally
    for h_block in TiledRange(cfg.H1_shard, NUM_TILES_PER_H_BLOCK):
        hidden_block = hidden_sb.slice(dim=2, start=h_block.start_offset, end=h_block.end_offset)
        proj_w_block = proj_w_hbm.slice(dim=1, start=h_block.start_offset, end=h_block.end_offset)

        w_block_slot = h_block.index % num_w_blocks
        proj_w_sb_block = proj_w_sb.select(dim=1, index=w_block_slot).slice(dim=1, start=0, end=h_block.size)
        nisa.dma_copy(proj_w_sb_block, proj_w_block)

        for h1_tile_idx in range(h_block.size):
            hidden_tile = hidden_block.select(dim=2, index=h1_tile_idx)

            for i_group in TiledRange(I, cfg.i_tile_size * col_tiling_factor):
                n_col_tiles = min(col_tiling_factor, div_ceil(i_group.size, cfg.i_tile_size))
                group_idx = i_group.index

                for col_idx in range(n_col_tiles):
                    i_offset = i_group.start_offset + col_idx * cfg.i_tile_size
                    i_size = min(cfg.i_tile_size, I - i_offset)
                    proj_w_sb_tile = proj_w_sb_block.select(dim=1, index=h1_tile_idx).slice(
                        dim=1, start=i_offset, end=i_offset + i_size
                    )
                    nisa.nc_matmul(
                        result_psum[group_idx][nl.ds(col_tiling_dim * col_idx, cfg.BxS), 0:i_size],
                        hidden_tile,
                        proj_w_sb_tile,
                        tile_position=(0, col_tiling_dim * col_idx),
                        tile_size=(cfg.H0, col_tiling_dim),
                    )

    # Evict: one copy per column tile (no reduction)
    for i_group in TiledRange(I, cfg.i_tile_size * col_tiling_factor):
        n_col_tiles = min(col_tiling_factor, div_ceil(i_group.size, cfg.i_tile_size))
        group_idx = i_group.index

        for col_idx in range(n_col_tiles):
            i_offset = i_group.start_offset + col_idx * cfg.i_tile_size
            i_size = min(cfg.i_tile_size, I - i_offset)
            psum_slice = result_psum[group_idx][nl.ds(col_tiling_dim * col_idx, cfg.BxS), 0:i_size]
            out_slice = proj_out_sb[0 : cfg.BxS, nl.ds(i_offset, i_size)]

            if has_preapplied_bias:
                nisa.tensor_tensor(out_slice, out_slice, psum_slice, op=nl.add)
            # Unlike with H-column-tiling, there is no reduction needed.
            else:
                if col_idx % 2 == 0:
                    nisa.tensor_copy(out_slice, psum_slice, engine=nisa.engine.vector)
                else:
                    nisa.tensor_copy(out_slice, psum_slice, engine=nisa.engine.scalar)

    sbm.close_scope()
    return proj_out_sb
