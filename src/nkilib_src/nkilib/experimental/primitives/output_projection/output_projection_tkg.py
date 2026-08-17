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
Output Projection TKG Kernel

This kernel implements the output projection operation (attention @ weight +
bias) commonly used after attention blocks in transformer models. The kernel is
specifically optimized for Token Generation (TKG, also known as Decode) scenarios
where the sequence length S is small (often 1 or a small number for spec. decode).

Remark: The input layouts expected for this kernel are different from those for the
CTE kernel. The reason for this is the broader impacts of such layouts on performance
(not just on this kernel but also on other kernels).

In CTE workloads, where sequence length is large, we generally expect to have to reload
it from HBM more frequently. Placing the large S dimension at the end allows more efficient
HBM loads.

For TKG workloads, the S dimension is small, so placing the N dimension next to it
allows more efficient GQA implementations by loading multiple heads at once.

This kernel is designed with LNC support. When LNC>1, the H dimension is sharded
across the cores. We choose to shard on H as this avoids the need for any
inter-core collective operations, as each core produces part of the output tensor.

"""

from dataclasses import dataclass
from typing import Any, Optional, Union

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import matmul_perf_mode
from nki.language import affine_range, static_range

from ....core.output_projection.output_projection_tkg_mx_impl import _output_projection_tkg_mx
from ....core.output_projection.output_projection_utils import calculate_head_packing
from ....core.utils.allocator import BufferManager, align_to, create_auto_alloc_manager, sizeinbytes
from ....core.utils.common_types import QuantizationType
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import (
    div_ceil,
    get_max_positive_value_for_dtype,
    get_program_sharding_info,
)
from ....core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ....core.utils.tiled_range import TiledRange
from ....core.utils.tiled_tensor import TiledTensor
from ..matmul_loop_nest import matmul_loop_nest

P_MAX = 128
F_MAX = 512
NUM_PSUM_BANKS = 8

# Conservative limit assuming 2 bytes per scalar and LNC=2, ~20MB SBUF per core for projection weights
MAX_VALIDATED_N_TIMES_H_SIZE = 163840
MAX_VALIDATED_N_TIMES_H_SIZE_FP32 = MAX_VALIDATED_N_TIMES_H_SIZE // 2


@nki.jit
def output_projection_tkg(
    attention: nl.NkiTensor,
    weight: nl.NkiTensor,
    bias: Optional[nl.NkiTensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    weight_scale: Optional[nl.NkiTensor] = None,
    input_scale: Optional[nl.NkiTensor] = None,
    TRANSPOSE_OUT: bool = False,
    OUT_IN_SB: bool = False,
    sbm: Optional[BufferManager] = None,
    dtype_mode=None,
) -> nl.NkiTensor:
    """
    Output Projection Kernel

    This kernel computes
      out = attention @ weight + bias
    typically used to project the output scores after an attention block in transformer models.

    This kernel is optimized for Token Generation (aka Decode) use cases where sequence length S is
    small.

    Dimensions:
        B: Batch size
        N: Number of heads
        S: Sequence length
        H: Hidden dimension size
        D: Head dimension size

    Args:
        attention (nl.NkiTensor): Input tensor in HBM or SBUF, typically the scores output from an attention block.
            Shape:    [D, B, N, S]
            Indexing: [d, b, n, s]
            Dtype:    nl.float32, nl.float16, or nl.bfloat16
        weight (nl.NkiTensor): Weight tensor in HBM
            Shape:    [N * D,     H] or [N*D // 4, H] if QuantizationType.MX
            Indexing: [n * D + d, h]
            Dtype:
                - QuantizationType.NONE: nl.float32, nl.float16, or nl.bfloat16
                - QuantizationType.STATIC: nl.float8_e4m3
                - QuantizationType.ROW: nl.float8_e4m3
                - QuantizationType.MX: nl.float8_e4m3fn_x4
        bias (Optional[nl.NkiTensor]): Optional bias tensor in HBM
            Shape:    [1, H]
            Indexing: [1, h]
            Dtype:    nl.float32, nl.float16, or nl.bfloat16
        quantization_type (QuantizationType): Type of quantization to apply (NONE, STATIC, ROW, MX).
            Default: QuantizationType.NONE.
        weight_scale (Optional[nl.NkiTensor]): Weight dequantization scale tensor in HBM
            Shape:    [P_MAX, 1] for STATIC, [P_MAX, H] for ROW (all P_MAX rows identical),
                      [N*D // 32, H] for MX.
            Dtype:    nl.float32 for STATIC/ROW, nl.uint8 for MX
        input_scale (Optional[nl.NkiTensor]): Input dequantization scale tensor in HBM
            Shape:    [P_MAX, 1] for STATIC (not used for ROW)
            Dtype:    nl.float32
        TRANSPOSE_OUT (bool): Whether to store the output in transposed shape.
            If False, the output tensor has the following shape and indexing:
              Shape:    [B * S,     H]
              Indexing: [b * S + s, h]
            If True, the output is instead kept in a different shape, which may be
            advantageous for other kernels' performance.
              Shape:    [H_1, H_0, H_2, B * S    ]
              Indexing: [h_1, h_0, h_2, b * S + s]
            where
              H_0 = logical core size (LNC = 1 or LNC = 2),
              H_1 = 128,
              H_2 = H // H_0 // H_1,
            such that h = h_0 * H_1 * H_2 + h_1 * H_2 + h_2.
        OUT_IN_SB (bool): If True, output is in SBUF. Else, it is written out to HBM.
        sbm (BufferManager): Optional BufferManager for tensor allocation with consistent naming.

    Returns:
        out (nl.NkiTensor): Output tensor in HBM. Shape depends on `TRANSPOSE_OUT` parameter.

    Notes:
        - This kernel supports nl.float32, nl.float16 and nl.bfloat16 data types.
          However, for nl.float32, large inputs may not fit in SBUF.
        - The product B * S must not exceed 128.
        - Head dimension D must not exceed 128.
        - When TRANSPOSE_OUT is False: H must be divisible by LNC.
        - When TRANSPOSE_OUT is True: H must be divisible by 128 * LNC.
        - When TRANSPOSE_OUT is True with float32 dtype: N * H must not exceed 81920.
        - When TRANSPOSE_OUT is True with float16/bfloat16 dtype: N * H must not exceed 163840.

    Pseudocode:
        # Load attention scores
        attn_sb = load_to_sbuf(attention)

        # Shuffle attention from [D, B, N, S] to [D, N * B * S]
        attn_shuffled = shuffle(attn_sb)

        # Compute projection
        if TRANSPOSE_OUT:
            for h_tile in range(H // (128 * LNC)):
                result[h_tile] = attn_shuffled @ weight[h_tile]
                if bias is not None:
                    result[h_tile] += bias[h_tile]
        else:
            for h_block in range(H // (512 * LNC)):
                result[h_block] = attn_shuffled @ weight[h_block]
                if bias is not None:
                    result[h_block] += bias[h_block]

        return result
    """

    cfg = _validate_and_create_config(
        attention=attention,
        weight=weight,
        bias=bias,
        quantization_type=quantization_type,
        weight_scale=weight_scale,
        input_scale=input_scale,
        transpose_out=TRANSPOSE_OUT,
        out_in_sb=OUT_IN_SB,
    )

    if quantization_type == QuantizationType.MX:
        return _output_projection_tkg_mx(
            attention=attention,
            weights_qtz=weight,
            weight_scales_hbm=weight_scale,
            bias=bias,
            TRANSPOSE_OUT=TRANSPOSE_OUT,
            quantization_type=quantization_type,
            input_scale=input_scale,
        )

    if sbm == None:
        sbm = create_auto_alloc_manager()

    sbm.open_scope(name="output_projection_tkg")

    quant_config, input_scale_sb = _prepare_quant_scales(
        weight_scale=weight_scale,
        input_scale=input_scale,
        quantization_type=quantization_type,
        cfg=cfg,
        sbm=sbm,
    )

    attn_shuffled_shape = (
        (cfg.d_size, 2, cfg.n_size // 2 * cfg.b_size * cfg.s_size)
        if cfg.use_double_row
        else (cfg.d_size, cfg.n_size * cfg.b_size * cfg.s_size)
    )

    bxs_size = cfg.b_size * cfg.s_size
    """
    SBUF memory layout for non-transpose path with bias (manual allocation):

    The goal is to separate the address ranges of all three 
    DMA streams (bias, attention, weights), eliminating
    anti-dependencies so they can all proceed in parallel.

    bias_sb_1d is a temporary heap buffer used to broadcast bias from
    [1, H] to [B*S, H]. Crucially, bias_sb_1d is freed (pop_heap) only
    *after* attn_sb is freed (which happens after _shuffle_attn returns).
    This ordering ensures bias_sb_1d and
    attn_sb occupy separate heap addresses during the load phase. If we
    freed bias_sb_1d before allocating attn_sb, the heap would reclaim
    that address range and attn_sb would overlap with bias_sb_1d,
    creating an anti-dependency between the bias and attention DMA loads.
    Allocating on the stack will cause the same anti-dependency with weight load.

      SBUF address space (stack grows ↓ from top, heap grows ↑ from bottom)
      ┌─────────────────────────┐ ◄─ upper bound (stack starts here, grows ↓)
      │  (quant scales, if any) │    stack
      ├─────────────────────────┤
      │  bias_sb                │    stack: allocated 1st ← keeps bias away
      ├─────────────────────────┤       from weights below
      │  attn_shuffled          │    stack: allocated 2nd
      ├─────────────────────────┤
      │  weights (loaded later  │    stack: allocated in _impl
      │  in _impl, below attn)  │
      ├─────────────────────────┤
      │      (free space)       │
      ├─────────────────────────┤
      │  attn_sb (temporary)    │    heap: freed after _shuffle_attn
      ├─────────────────────────┤
      │  bias_sb_1d (temporary) │    heap: freed after attn_sb is freed
      └─────────────────────────┘ ◄─ lower bound (heap starts here, grows ↑)

    The heap during the load phase (both alive simultaneously):
      bias_sb_1d ← allocated 1st (lowest addr)
      attn_sb    ← allocated 2nd (above bias_sb_1d, separate addr)
    Because they coexist, their DMA loads target disjoint addresses,
    preventing anti-dependencies.
    """
    if not cfg.transpose_out and cfg.has_bias:
        BxS_block_size = min(P_MAX, bxs_size)
        bias_sb = _prepare_bias(bias=bias, cfg=cfg, sbm=sbm)
        bias_sb_1d_allocated = True
    else:
        bias_sb = None
        bias_sb_1d_allocated = False

    attn_shuffled_dtype = cfg.quant_dtype if cfg.quantization_type == QuantizationType.STATIC else attention.dtype

    attn_shuffled = sbm.alloc_stack(attn_shuffled_shape, dtype=attn_shuffled_dtype, buffer=nl.sbuf, align=cfg.align)

    attn_sb = _load_attn_to_sbuf(attention=attention, cfg=cfg, sbm=sbm)
    if quantization_type == QuantizationType.STATIC:
        attn_sb = _quantize_attn(attn_sb=attn_sb, input_scale_sb=input_scale_sb, cfg=cfg, sbm=sbm)
    _shuffle_attn(attn_sb=attn_sb, attn_shuffled=attn_shuffled, cfg=cfg)

    if cfg.quantization_type == QuantizationType.STATIC:
        sbm.pop_heap()  # attn_quantized
    if attention.buffer != nl.sbuf:
        sbm.pop_heap()  # attn_sb

    if bias_sb_1d_allocated:
        sbm.pop_heap()  # bias_sb_1d

    w_reshaped = weight.reshape((cfg.n_size, cfg.d_size, cfg.h_size))
    # Pre-shard weights by prg_id: [N, D, H] -> [N, D, h_sharded]
    w_shard_hbm = w_reshaped.reshape_dim(2, (cfg.num_prgs, cfg.h_sharded)).select(dim=2, index=cfg.prg_id)

    if not cfg.transpose_out:
        out = (
            sbm.alloc(
                (cfg.b_size * cfg.s_size, cfg.h_size),
                dtype=cfg.io_dtype,
                buffer=nl.shared_hbm,
                name="output_projection_tkg_out",
            )
            if not cfg.out_in_sb
            else sbm.alloc_stack(
                (min(bxs_size, P_MAX), cfg.h_sharded),
                dtype=cfg.io_dtype,
                buffer=nl.sbuf,
                align=cfg.align,
            )
        )

        out_hbm_view = (
            out.reshape_dim(1, (cfg.num_prgs, cfg.h_sharded)).select(dim=1, index=cfg.prg_id) if out != None else None
        )

        result = _output_projection_tkg_impl(
            out_hbm_buffer=out if not cfg.out_in_sb else None,
            out_hbm_view=out_hbm_view,
            out_sb=out if cfg.out_in_sb else None,
            bias_sb=bias_sb,
            w_shard_hbm=w_shard_hbm,
            quant_config=quant_config,
            attn_shuffled=attn_shuffled,
            cfg=cfg,
            sbm=sbm,
        )

    else:  # TRANSPOSE_OUT == True
        """
        Notes on iteration order:
        
        cfg.h_0_size corresponds to the outermost logical iterator h_0 = prg_id from 0 to cfg.num_prgs - 1. This corresponds to LNC sharding.
        cfg.h_1_size corresponds to the mid logical iterator h_1 from 0 to P_MAX - 1. This is placed in partition dim.
        cfg.h_2_size corresponds to the innermost logical iterator h_2. This is placed in free dim.
        
        Check for h_size % (P_MAX * n_prgs) == 0 above should cover this
        """
        kernel_assert(
            cfg.h_size == cfg.h_0_size * cfg.h_1_size * cfg.h_2_size,
            f"H decomposition mismatch: {cfg.h_size} != {cfg.h_0_size} * {cfg.h_1_size} * {cfg.h_2_size}",
        )
        out = (
            sbm.alloc(
                (cfg.h_1_size, cfg.h_0_size, cfg.h_2_size, cfg.b_size * cfg.s_size),
                dtype=cfg.io_dtype,
                buffer=nl.shared_hbm,
                name="output_projection_tkg_out",
            )
            if not cfg.out_in_sb
            else sbm.alloc_stack(
                (cfg.h_1_size, cfg.h_2_size * bxs_size),
                dtype=cfg.io_dtype,
                buffer=nl.sbuf,
                align=cfg.align,
            )
        )

        out_hbm_view = out.select(dim=1, index=cfg.prg_id) if out != None else None

        result = _output_projection_tkg_transpose_out_impl(
            out_hbm_buffer=out if not cfg.out_in_sb else None,
            out_hbm_view=out_hbm_view,
            out_sb=out if cfg.out_in_sb else None,
            bias_sb=_prepare_bias_transposed(bias=bias, cfg=cfg, sbm=sbm),
            w_shard_hbm=w_shard_hbm,
            quant_config=quant_config,
            attn_shuffled=attn_shuffled,
            cfg=cfg,
            sbm=sbm,
        )

    sbm.close_scope()
    return result


@dataclass
class StaticQuantConfig(nl.NKIObject):
    """Configuration for STATIC quantization.

    Holds pre-computed combined scale (weight_scale * input_scale) in SBUF.
    """

    combined_scale_sb: nl.NkiTensor  # [P_MAX, 1], pre-computed in SBUF


@dataclass
class RowQuantConfig(nl.NKIObject):
    """Configuration for ROW (per-output-channel) quantization.

    Holds weight scale tensor in HBM to be loaded inside impl functions.
    """

    weight_scale_hbm: nl.NkiTensor  # [P_MAX, H], in HBM


@dataclass
class OutputProjectionTkgConfig(nl.NKIObject):
    """Configuration and validation for output projection TKG kernel.

    Input tensors:
        attention: [D, B, N, S]
        weight:    [N*D, H]
        bias:      [1, H] (optional)
    """

    # Input dimensions
    d_original_size: int
    """Head dimension before packing. From attention.shape[0]."""
    b_size: int
    """Batch size. From attention.shape[1]."""
    n_original_size: int
    """Number of attention heads before packing. From attention.shape[2]."""
    s_size: int
    """Sequence length. From attention.shape[3]."""
    h_size: int
    """Hidden dimension. From weight.shape[1]."""

    # Execution parameters
    num_prgs: int
    """Number of logical neuron cores (LNC shards)."""
    prg_id: int
    """Current logical neuron core index."""
    align: int
    """Alignment for intermediate tensors."""

    # Kernel options
    transpose_out: bool
    """If True, output shape is [h_1, h_0, h_2, B*S]; if False, [B*S, H]."""
    out_in_sb: bool
    """If True, output stays in SBUF instead of being written to HBM."""
    is_quantized: bool
    """True when quantization_type is STATIC or ROW."""
    quantization_type: QuantizationType
    """Quantization mode: NONE, STATIC, or ROW."""
    use_double_row: bool
    """If True, pairs two heads per matmul using double_row perf mode.
    Requires STATIC quant, even n_size, n_size*b_size % 32 == 0, B*S >= 64."""

    # Optional tensors (for validation)
    has_bias: bool = False
    """Whether a bias tensor [1, H] was provided."""
    has_weight_scale: bool = False
    """Whether a weight_scale tensor was provided."""
    has_input_scale: bool = False
    """Whether an input_scale tensor was provided (STATIC quant only)."""

    # Data types
    io_dtype: Any = None
    """Data type of attention input and kernel output (e.g. bfloat16)."""
    quant_dtype: Any = None
    """Data type of quantized weight (e.g. float8_e4m3). Set to weight.dtype."""

    # Computed fields (set by _validate_and_create_config)
    n_size: int = None
    """Number of heads after packing: n_original_size // group_size."""
    d_size: int = None
    """Head dimension after packing: d_original_size * group_size. d_size <= P_MAX."""
    group_size: int = None
    """Head packing factor. Multiple heads folded into the D dimension when d_original_size < P_MAX."""
    h_sharded: int = None
    """Per-core hidden dimension: h_size // num_prgs."""
    h_0_size: int = None
    """Transpose-out only. Outermost H tiling dim = num_prgs (LNC sharding). -1 when transpose_out=False."""
    h_1_size: int = None
    """Transpose-out only. Mid H tiling dim = P_MAX (partition dim). -1 when transpose_out=False."""
    h_2_size: int = None
    """Transpose-out only. Innermost H tiling dim = h_sharded // P_MAX (free dim). -1 when transpose_out=False."""


def _validate_and_create_config(
    attention: nl.NkiTensor,
    weight: nl.NkiTensor,
    bias: Optional[nl.NkiTensor] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    weight_scale: Optional[nl.NkiTensor] = None,
    input_scale: Optional[nl.NkiTensor] = None,
    transpose_out: bool = False,
    out_in_sb: bool = False,
) -> OutputProjectionTkgConfig:
    """
    Validate inputs and create kernel configuration.

    Performs comprehensive validation of input tensor shapes, quantization settings,
    and layout constraints. Computes derived tiling parameters including head packing.

    Args:
        attention (nl.NkiTensor): [D, B, N, S], Input attention tensor.
        weight (nl.NkiTensor): [N*D, H], Weight tensor.
        bias (Optional[nl.NkiTensor]): [1, H], Optional bias tensor.
        quantization_type (QuantizationType): Quantization mode (NONE, STATIC, or ROW).
        weight_scale (Optional[nl.NkiTensor]): [P_MAX, 1] for STATIC, [P_MAX, H] for ROW.
        input_scale (Optional[nl.NkiTensor]): [P_MAX, 1] for STATIC (not used for ROW).
        transpose_out (bool): Whether to produce transposed output layout.
        out_in_sb (bool): Whether output stays in SBUF.

    Returns:
        OutputProjectionTkgConfig: Validated configuration with computed tiling parameters.
    """
    d_original_size, b_size, n_original_size, s_size = attention.shape
    n_d, h_size = weight.shape
    io_dtype = attention.dtype
    _, n_prgs, prg_id = get_program_sharding_info()

    # Hardware constraints
    kernel_assert(nl.tile_size.pmax == nl.tile_size.gemm_stationary_fmax, "pmax must equal gemm_stationary_fmax")
    kernel_assert(nl.tile_size.psum_fmax == nl.tile_size.gemm_moving_fmax, "psum_fmax must equal gemm_moving_fmax")

    # MX-specific validation
    if quantization_type == QuantizationType.MX:
        # d_"original"_size and n_"original"_size are there for head packing, unrelated to MX.

        ################### Verify Dtypes #############
        kernel_assert(
            weight.dtype == nl.float8_e4m3fn_x4,
            f"MX quantization requires weight dtype float8_e4m3fn_x4, got {weight.dtype}",
        )
        kernel_assert(
            weight_scale != None,
            "weight_scale must be provided for MX quantization",
        )
        kernel_assert(
            weight_scale.dtype == nl.uint8,
            f"MX quantization requires weight_scale dtype nl.uint8, got {weight_scale.dtype}",
        )

        ##################### Verify Shapes #############
        # n_d variable is inferred from weight shape above
        n_d_not_packed = n_original_size * d_original_size
        n_d_packed = n_d_not_packed // 4  # (= n_d)
        kernel_assert(
            weight.shape[0] == n_d_packed,
            f"MX output_projection_tkg kernel requires weight in shape ({n_d_packed}, H = {h_size}), but got {weight.shape}.\n",
        )
        kernel_assert(
            weight_scale.shape == (n_d_not_packed // 32, h_size),
            f"MX quantization requires weight_scale shape [{n_d_not_packed // 32}, {h_size}], got {weight_scale.shape}",
        )
        kernel_assert(
            (b_size * s_size) % 4 == 0,
            f"MX quantization requires B*S ({b_size * s_size}) to be divisible by 4",
        )
        kernel_assert(
            h_size % 4 == 0,
            f"MX quantization requires H ({h_size}) to be divisible by 4",
        )
        kernel_assert(
            (b_size * s_size) <= 128,
            f"MX quantization requires B*S ({b_size * s_size}) to be <=128",
        )
        kernel_assert(
            n_d_not_packed % P_MAX == 0,
            f"MX quantization requires N*D ({n_d_not_packed}) to be divisible by {P_MAX}",
        )

        ######## Unsupported Features ####################
        kernel_assert(
            attention.buffer != nl.sbuf,
            "MX quantization only accepts HBM input.",
        )
        kernel_assert(
            not transpose_out,
            "MX quantization only supports TRANSPOSE_OUT=False",
        )
        kernel_assert(
            not out_in_sb,
            "MX quantization only supports out_in_sb=False",
        )
    else:
        # Validate weight shape
        kernel_assert(
            n_d == n_original_size * d_original_size,
            f"output_projection_tkg kernel requires weight in shape (N * D = {n_original_size * d_original_size}, H = {h_size}), but got {weight.shape}.\n"
            f"Note: N and D inferred from attention score shape: {attention.shape}: {n_original_size} * {d_original_size} = {n_original_size * d_original_size}.",
        )

    # Validate bias shape
    if bias != None:
        kernel_assert(
            bias.shape[0] == 1,
            f"output_projection_tkg kernel requires bias in shape (1, H = {h_size}), but got {bias.shape}.\n"
            f"Note: H inferred from weight shape: {weight.shape}.",
        )
        kernel_assert(
            bias.shape[1] == h_size,
            f"output_projection_tkg kernel requires bias in shape (1, H = {h_size}), but got {bias.shape}.\n"
            f"Note: H inferred from weight shape: {weight.shape}.",
        )

    # Validate quantization type
    kernel_assert(
        quantization_type == QuantizationType.NONE
        or quantization_type == QuantizationType.STATIC
        or quantization_type == QuantizationType.ROW
        or quantization_type == QuantizationType.MX,
        f"Only QuantizationType.NONE, QuantizationType.STATIC, QuantizationType.ROW, and QuantizationType.MX are supported, but got {quantization_type}",
    )

    # Validate quantization scale shapes
    if quantization_type == QuantizationType.STATIC:
        kernel_assert(weight_scale != None, f"Weight scale must be provided for quantization type {quantization_type}")
        kernel_assert(input_scale != None, f"Input scale must be provided for quantization type {quantization_type}")
        kernel_assert(
            weight_scale.shape == (P_MAX, 1),
            f"Incorrect shape for weight scale for static per tensor quantization, expected ({P_MAX}, 1), got {weight_scale.shape}",
        )
        kernel_assert(
            input_scale.shape == (P_MAX, 1),
            f"Incorrect shape for input scale for static per tensor quantization, expected ({P_MAX}, 1), got {input_scale.shape}",
        )
    elif quantization_type == QuantizationType.ROW:
        kernel_assert(weight_scale != None, f"Weight scale must be provided for quantization type {quantization_type}")
        kernel_assert(
            weight_scale.shape == (P_MAX, h_size),
            f"Incorrect shape for weight scale for row quantization, expected ({P_MAX}, {h_size}), got {weight_scale.shape}",
        )

    # Kernel shape validation
    kernel_assert(
        h_size % n_prgs == 0,
        f"output_projection_tkg kernel requires hidden dimension (H = {h_size}) to be divisible by logical core size of {n_prgs}.",
    )

    # Dimension constraints
    if out_in_sb and not transpose_out:
        kernel_assert(
            b_size * s_size <= P_MAX,
            f"When OUT_IN_SB=True and TRANSPOSE_OUT=False, output_projection_tkg kernel does not support (B * S = {b_size * s_size}) > {P_MAX}.",
        )

    kernel_assert(
        d_original_size <= P_MAX,
        f"output_projection_tkg kernel does not support head dimension (D = {d_original_size}) greater than {P_MAX}.",
    )

    # Layout-specific validation
    if not transpose_out:
        kernel_assert(
            h_size % n_prgs == 0,
            f"When `TRANSPOSE_OUT` is False, output_projection_tkg kernel requires hidden dimension (H = {h_size}) to be a multiple of logical core size, where logical core size is {n_prgs}.",
        )
    else:
        kernel_assert(
            h_size % (P_MAX * n_prgs) == 0,
            f"When `TRANSPOSE_OUT` is True, output_projection_tkg kernel requires hidden dimension (H = {h_size}) to be a multiple of {P_MAX} * logical core size, where logical core size is {n_prgs}.",
        )

        # Size limits for transpose mode
        if weight.dtype == nl.float32:
            kernel_assert(
                n_original_size * h_size <= MAX_VALIDATED_N_TIMES_H_SIZE_FP32,
                f"When `TRANSPOSE_OUT` is True and using 32bit floats, output_projection_tkg kernel is not tested for (N * H = {n_original_size * h_size}) greater than {MAX_VALIDATED_N_TIMES_H_SIZE_FP32}.",
            )
        else:
            kernel_assert(
                n_original_size * h_size <= MAX_VALIDATED_N_TIMES_H_SIZE,
                f"When `TRANSPOSE_OUT` is True, output_projection_tkg kernel is not tested for (N * H = {n_original_size * h_size}) greater than {MAX_VALIDATED_N_TIMES_H_SIZE}.",
            )

    # Head packing: pack small heads into partition dim when aligned to 32 (hardware vector width)
    if d_original_size % 32 == 0:
        n_size, d_size, group_size = calculate_head_packing(n_original_size, d_original_size, P_MAX)
    else:
        n_size, d_size, group_size = n_original_size, d_original_size, 1

    # Tiling parameters
    h_sharded = h_size // n_prgs

    if transpose_out:
        h_0_size = n_prgs
        h_1_size = P_MAX
        h_2_size = h_size // n_prgs // P_MAX
    else:
        h_0_size = -1
        h_1_size = -1
        h_2_size = -1

    is_quantized = (
        quantization_type == QuantizationType.STATIC
        or quantization_type == QuantizationType.ROW
        or quantization_type == QuantizationType.MX
    )

    # TODO: support padding for odd number of heads
    # n_heads // 2 * batch needs to be multiple of 16 for double row stride access
    use_double_row = (
        quantization_type == QuantizationType.STATIC
        and n_size % 2 == 0
        and n_size * b_size % 32 == 0
        and b_size * s_size >= 64
    )

    align = 32 if nisa.get_nc_version() == nisa.nc_version.gen4 else 4

    return OutputProjectionTkgConfig(
        d_original_size=d_original_size,
        b_size=b_size,
        n_original_size=n_original_size,
        s_size=s_size,
        h_size=h_size,
        num_prgs=n_prgs,
        prg_id=prg_id,
        align=align,
        transpose_out=transpose_out,
        out_in_sb=out_in_sb,
        is_quantized=is_quantized,
        quantization_type=quantization_type,
        use_double_row=use_double_row,
        has_bias=bias != None,
        has_weight_scale=weight_scale != None,
        has_input_scale=input_scale != None,
        io_dtype=io_dtype,
        quant_dtype=weight.dtype,
        n_size=n_size,
        d_size=d_size,
        group_size=group_size,
        h_sharded=h_sharded,
        h_0_size=h_0_size,
        h_1_size=h_1_size,
        h_2_size=h_2_size,
    )


def _prepare_quant_scales(
    weight_scale: Optional[nl.NkiTensor],
    input_scale: Optional[nl.NkiTensor],
    quantization_type: QuantizationType,
    cfg: OutputProjectionTkgConfig,
    sbm: BufferManager,
):
    """Prepare quantization scale tensors in SBUF.

    For STATIC: allocates and computes combined_scale_sb = weight_scale * input_scale.
    Returns (quant_config, input_scale_sb). input_scale_sb is needed by _quantize_attn
    and will be destructively modified there via nisa.reciprocal.

    This function MUST be called before _quantize_attn.
    """
    if quantization_type == QuantizationType.STATIC:
        input_scale_sb = sbm.alloc_stack(
            weight_scale.shape, dtype=weight_scale.dtype, buffer=nl.sbuf, align=cfg.align, name="input_scale_sb"
        )
        nisa.dma_copy(dst=input_scale_sb, src=input_scale)
        combined_scale_sb = sbm.alloc_stack(
            weight_scale.shape, dtype=weight_scale.dtype, buffer=nl.sbuf, align=cfg.align, name="combined_scale_sb"
        )
        nisa.dma_copy(dst=combined_scale_sb, src=weight_scale)
        nisa.activation(dst=combined_scale_sb, op=nl.copy, data=combined_scale_sb, scale=input_scale_sb)
        return StaticQuantConfig(combined_scale_sb=combined_scale_sb), input_scale_sb
    elif quantization_type == QuantizationType.ROW:
        return RowQuantConfig(weight_scale_hbm=weight_scale), None
    else:
        return None, None


def _prepare_bias(
    bias: nl.NkiTensor,
    cfg: OutputProjectionTkgConfig,
    sbm: BufferManager,
) -> nl.NkiTensor:
    """Prepare bias for non-transposed path: slice by prg_id, DMA, broadcast.

    Allocates bias_sb on stack and bias_sb_1d on heap. The caller MUST call
    sbm.pop_heap() after _shuffle_attn to free bias_sb_1d.
    See the SBUF memory layout comment in output_projection_tkg for details.

    Returns bias_sb [BxS_block_size, h_sharded] in SBUF.
    """
    bxs_size = cfg.b_size * cfg.s_size
    BxS_block_size = min(P_MAX, bxs_size)
    bias_sb = sbm.alloc_stack((BxS_block_size, cfg.h_sharded), dtype=bias.dtype, buffer=nl.sbuf, align=cfg.align)
    bias_sb_1d = sbm.alloc_heap((1, cfg.h_sharded), dtype=bias.dtype, buffer=nl.sbuf, align=cfg.align)
    nisa.dma_copy(
        src=bias.reshape_dim(1, (cfg.num_prgs, cfg.h_sharded)).select(dim=1, index=cfg.prg_id),
        dst=bias_sb_1d,
    )
    stream_shuffle_broadcast(bias_sb_1d, bias_sb)
    return bias_sb


def _prepare_bias_transposed(
    bias: Optional[nl.NkiTensor],
    cfg: OutputProjectionTkgConfig,
    sbm: BufferManager,
) -> Optional[nl.NkiTensor]:
    """Prepare bias for transposed path: reshape, select prg_id, DMA to SBUF.

    Returns bias_sb [h_1_size, h_2_size] in SBUF, or None if no bias.
    """
    if bias == None:
        return None
    bias_sb = sbm.alloc_stack((cfg.h_1_size, cfg.h_2_size), dtype=bias.dtype, buffer=nl.sbuf, align=cfg.align)
    nisa.dma_copy(
        dst=bias_sb,
        src=bias.reshape_dim(1, (cfg.h_0_size, cfg.h_1_size, cfg.h_2_size))
        .select(dim=1, index=cfg.prg_id)
        .squeeze_dim(0),
    )
    return bias_sb


def _prepare_weight_scales(
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]],
    cfg: OutputProjectionTkgConfig,
    sbm: BufferManager,
    h_block_sizes: list,
    h_block_offsets: list,
    num_h_blocks_per_prg: int,
):
    """Prepare weight scales for non-transposed path.

    For STATIC: returns (combined_scale_sb, None).
    For ROW: slices weight_scale_hbm by prg_id, blocks and DMAs to SBUF.
        Returns (None, weight_scale_blocks).
    For NONE: returns (None, None).
    """
    if cfg.quantization_type == QuantizationType.STATIC:
        return quant_config.combined_scale_sb, None
    elif cfg.quantization_type == QuantizationType.ROW:
        weight_scale_hbm = quant_config.weight_scale_hbm
        weight_scale_shard = weight_scale_hbm.reshape_dim(1, (cfg.num_prgs, cfg.h_sharded)).select(
            dim=1, index=cfg.prg_id
        )
        w_scale_dtype = weight_scale_hbm.dtype
        weight_scale_blocks = []
        for h_block_idx in affine_range(num_h_blocks_per_prg):
            cur_size = h_block_sizes[h_block_idx]
            cur_offset = h_block_offsets[h_block_idx]
            scale_tensor = sbm.alloc_stack((P_MAX, cur_size), dtype=w_scale_dtype, align=cfg.align, buffer=nl.sbuf)
            nisa.dma_copy(
                src=weight_scale_shard.slice(dim=1, start=cur_offset, end=cur_offset + cur_size),
                dst=scale_tensor,
            )
            weight_scale_blocks.append(scale_tensor)
        return None, weight_scale_blocks
    else:
        return None, None


def _prepare_weight_scales_transposed(
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]],
    cfg: OutputProjectionTkgConfig,
    sbm: BufferManager,
):
    """Prepare weight scales for transposed path.

    For STATIC: returns combined_scale_sb.
    For ROW: selects row 0 of weight_scale_hbm, reshapes to [h_0, h_1, h_2],
        selects prg_id, DMAs [h_1, h_2] to SBUF.
    For NONE: returns None.
    """
    if cfg.quantization_type == QuantizationType.STATIC:
        return quant_config.combined_scale_sb
    elif cfg.quantization_type == QuantizationType.ROW:
        weight_scale_hbm = quant_config.weight_scale_hbm
        weight_scale_sb = sbm.alloc_stack(
            (cfg.h_1_size, cfg.h_2_size),
            dtype=weight_scale_hbm.dtype,
            buffer=nl.sbuf,
            align=cfg.align,
            name="weight_scale_sb",
        )
        nisa.dma_copy(
            dst=weight_scale_sb,
            src=weight_scale_hbm.select(dim=0, index=0)
            .reshape_dim(0, (cfg.h_0_size, cfg.h_1_size, cfg.h_2_size))
            .select(dim=0, index=cfg.prg_id),
        )
        return weight_scale_sb
    else:
        return None


def _build_psum_and_out_views(
    out_sb: nl.NkiTensor,
    res_psum: nl.NkiTensor,
    bxs_size: int,
    bxs_tile_offset: int,
    bxs_tile_size: int,
    h_2_base: int,
    num_valid_bs_groups: int,
    out_sb_h_2_size: int,
):
    """Build out_sb and psum NkiTensors for the current PSUM tile.

    When bxs_size <= F_MAX, multiple B*S groups are packed per PSUM bank.
    When bxs_size > F_MAX, B*S is tiled across multiple PSUM tiles.

    Returns (out_sb_view, psum_view).
    """
    if bxs_size <= F_MAX:
        out_sb_view = out_sb.reshape_dim(1, (out_sb_h_2_size, bxs_size)).slice(
            dim=1, start=h_2_base, end=h_2_base + num_valid_bs_groups
        )
        psum_view = res_psum.slice(dim=1, start=0, end=num_valid_bs_groups * bxs_size).reshape_dim(
            1, (num_valid_bs_groups, bxs_size)
        )
    else:
        out_sb_view = (
            out_sb.reshape_dim(1, (out_sb_h_2_size, bxs_size))
            .select(dim=1, index=h_2_base)
            .slice(dim=1, start=bxs_tile_offset, end=bxs_tile_offset + bxs_tile_size)
            .expand_dim(2)
        )
        psum_view = res_psum.slice(dim=1, start=0, end=bxs_tile_size).expand_dim(2)
    return out_sb_view, psum_view


def _load_attn_to_sbuf(
    attention: nl.NkiTensor,
    cfg: OutputProjectionTkgConfig,
    sbm: BufferManager,
) -> nl.NkiTensor:
    """Load attention tensor from HBM to SBUF, or return as-is if already in SBUF.

    Allocates on heap to prevent anti-dependency with later weight/bias loading.

    Args:
        attention: [d_original_size, B, n_original_size, S], Input attention tensor.
        cfg: Kernel configuration.
        sbm: Buffer manager for heap allocation.

    Returns:
        Attention tensor in SBUF with shape [d_original_size, B, n_original_size, S].
    """
    if attention.buffer == nl.sbuf:
        return attention
    attn_sb = sbm.alloc_heap(
        (cfg.d_original_size, cfg.b_size, cfg.n_original_size, cfg.s_size),
        dtype=attention.dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(dst=attn_sb[...], src=attention[...])
    return attn_sb


def _quantize_attn(
    attn_sb: nl.NkiTensor,
    input_scale_sb: nl.NkiTensor,
    cfg: OutputProjectionTkgConfig,
    sbm: BufferManager,
) -> nl.NkiTensor:
    """Quantize attention tensor to FP8 for STATIC quantization.

    Applies reciprocal of input_scale, scales attention, and clamps to FP8 range.

    IMPORTANT: Destructively modifies input_scale_sb via nisa.reciprocal.
    Must be called AFTER _prepare_quant_scales has finished computing combined_scale_sb.

    Args:
        attn_sb: [d_original_size, B, n_original_size, S], Attention tensor in SBUF.
        input_scale_sb: [P_MAX, 1], Input scale tensor in SBUF.
        cfg: Kernel configuration.
        sbm: Buffer manager for heap allocation.

    Returns:
        Quantized attention tensor in SBUF with quant_dtype.
    """
    attn_quantized = sbm.alloc_heap(
        (cfg.d_original_size, cfg.b_size, cfg.n_original_size, cfg.s_size),
        dtype=cfg.quant_dtype,
        buffer=nl.sbuf,
        align=cfg.align,
    )
    nisa.reciprocal(dst=input_scale_sb, data=input_scale_sb)
    nisa.activation(dst=attn_sb, op=nl.copy, data=attn_sb, scale=input_scale_sb[: cfg.d_original_size, :])
    max_pos_val = get_max_positive_value_for_dtype(cfg.quant_dtype)
    nisa.tensor_scalar(
        dst=attn_quantized,
        data=attn_sb,
        op0=nl.minimum,
        operand0=max_pos_val,
        op1=nl.maximum,
        operand1=-max_pos_val,
    )
    return attn_quantized


def _shuffle_attn(
    attn_sb: nl.NkiTensor,
    attn_shuffled: nl.NkiTensor,
    cfg: OutputProjectionTkgConfig,
) -> nl.NkiTensor:
    """Shuffle attention from [d_original_size, B, n_original_size, S] to [D, N * B * S].

    Pure SBUF-to-SBUF operation. When group_size > 1, packs multiple heads into the
    partition dimension. For double_row, produces [D, 2, N//2 * B * S] with interleaved heads:
    - Head 0 -> row 0, pair 0: attn_shuffled[d, 0, 0:bxs]
    - Head 1 -> row 1, pair 0: attn_shuffled[d, 1, 0:bxs]
    - Head 2 -> row 0, pair 1: attn_shuffled[d, 0, bxs:2*bxs]
    - Head 3 -> row 1, pair 1: attn_shuffled[d, 1, bxs:2*bxs]

    Args:
        attn_sb: Attention tensor in SBUF, shape [d_original_size, B, n_original_size, S].
        attn_shuffled: Pre-allocated output buffer in SBUF.
        cfg: Kernel configuration.

    Returns:
        Shuffled attention tensor in attn_shuffled.
    """
    bxs_size = cfg.b_size * cfg.s_size

    for n_orig in static_range(cfg.n_original_size):
        n_group, n_offset = divmod(n_orig, cfg.group_size)
        dst_p_start = n_offset * cfg.d_original_size

        src_view = attn_sb.select(dim=2, index=n_orig)
        dst_view = attn_shuffled.slice(dim=0, start=dst_p_start, end=dst_p_start + cfg.d_original_size)

        if cfg.use_double_row:
            # Double row: attn_shuffled[d_size, 2, n_size//2 * b * s]
            # Interleave heads: row_idx = n_group % 2, pair_idx = n_group // 2
            row_idx = n_group % 2
            pair_offset = (n_group // 2) * bxs_size
            dst_view = dst_view.select(dim=1, index=row_idx).slice(dim=1, start=pair_offset, end=pair_offset + bxs_size)
        else:
            # Single row: attn_shuffled[d_size, n_size * b * s]
            n_group_offset = n_group * bxs_size
            dst_view = dst_view.slice(dim=1, start=n_group_offset, end=n_group_offset + bxs_size)

        nisa.tensor_copy(dst=dst_view, src=src_view)

    return attn_shuffled


def _get_block_sizes_and_offsets(h_sharded, block_size):
    """Return (sizes_list, offsets_list) for the given block_size, with a remainder."""
    num_full, remainder = divmod(h_sharded, block_size)
    sizes = [block_size] * num_full
    if remainder > 0:
        sizes.append(remainder)
    offsets = []
    off = 0
    for s in sizes:
        offsets.append(off)
        off += s
    return sizes, offsets


def _budget_weight_blocks(
    num_bxs_tiles: int, w_dtype, cfg: OutputProjectionTkgConfig, sbm: BufferManager, w_scale_dtype=None
) -> tuple:
    """
    Choose h_block_size, weight slot count, and interleave degree for the
    non-transpose output projection path.

    Three-step priority:
    1. Block size: Try h_sharded down to F_MAX (halving). Pick the largest
       where at least 1 weight block + 1 out_sb fits in SBUF. Remainder
       blocks are handled by _get_block_sizes_and_offsets.
    2. Weight slots: Maximize the number of weight blocks that fit, reducing
       reload pressure across B×S tile iterations.
    3. Interleave: Use remaining space for out_sb multi-buffering across
       B×S tiles, capped by num_bxs_tiles.

    When cfg.out_in_sb is True, the output stays in SBUF with no DMA store
    to pipeline, so interleave degree is always 1.

    Args:
        num_bxs_tiles: Number of bxs tiles in the outer tile loop.
        w_dtype: Weight data type.
        cfg: Kernel configuration.
        sbm: Buffer manager.
        w_scale_dtype: Weight scale data type for ROW quantization. When set,
            the budget reserves space for one weight scale block before
            computing available space for weights and interleave.

    Returns:
        (h_block_sizes, h_block_offsets, num_w_h_blocks, out_sb_interleave_degree):
        h_block_sizes is a list of per-block sizes (length num_h_blocks_per_prg),
        h_block_offsets is a list of cumulative offsets within h_sharded,
        num_w_h_blocks is the number of weight h_block SBUF slots to allocate
        (may be less than num_h_blocks_per_prg for circular buffering),
        and out_sb_interleave_degree is the interleave degree for the bxs tile loop.
    """

    INITIAL_H_BLOCK_SIZE = 2048
    if sbm.is_auto_alloc():
        block_size = INITIAL_H_BLOCK_SIZE if cfg.h_sharded >= INITIAL_H_BLOCK_SIZE else cfg.h_sharded
        sizes, offsets = _get_block_sizes_and_offsets(cfg.h_sharded, block_size)
        return sizes, offsets, len(sizes), 1

    out_sb_bytes = align_to(cfg.h_sharded * sizeinbytes(cfg.io_dtype), cfg.align)
    n_heads_per_block = cfg.n_size // 2 if cfg.use_double_row else cfg.n_size

    free_space = sbm.get_free_space()
    if cfg.quantization_type == QuantizationType.ROW:
        free_space -= align_to(cfg.h_sharded * sizeinbytes(w_scale_dtype), cfg.align)

    # Build candidate h_block_sizes: h_sharded down to F_MAX (halving)
    block_size_candidates = []
    size = cfg.h_sharded
    while size >= F_MAX:
        block_size_candidates.append(size)
        # TODO: Investigate if div_ceil will yield better results for
        # cases where floor yields tiny remainders
        size //= 2
    if not block_size_candidates:
        block_size_candidates.append(cfg.h_sharded)  # h_sharded < F_MAX, use as-is

    # Step 1: Find the largest block size where at least 1 weight block fits
    best_block_size = None
    for block_size in block_size_candidates:
        w_elts_per_head = (2 if cfg.use_double_row else 1) * block_size
        bytes_per_h_block = align_to(n_heads_per_block * w_elts_per_head * sizeinbytes(w_dtype), cfg.align)
        # Need at least 1 weight block + 1 out_sb
        if bytes_per_h_block + out_sb_bytes <= free_space:
            best_block_size = block_size
            break

    kernel_assert(
        best_block_size != None,
        f"Not enough SBUF space for even 1 weight block, free={free_space}, out_sb={out_sb_bytes}",
    )

    if best_block_size % INITIAL_H_BLOCK_SIZE == 0:
        best_block_size = INITIAL_H_BLOCK_SIZE

    # Step 2: Maximize weight blocks first (reduces reload pressure across B×S tiles)
    num_h_blocks = div_ceil(cfg.h_sharded, best_block_size)
    w_elts_per_head = (2 if cfg.use_double_row else 1) * best_block_size
    bytes_per_h_block = align_to(n_heads_per_block * w_elts_per_head * sizeinbytes(w_dtype), cfg.align)
    if cfg.out_in_sb:
        best_w_blocks = min(num_h_blocks, free_space // bytes_per_h_block)
    else:
        best_w_blocks = min(num_h_blocks, (free_space - out_sb_bytes) // bytes_per_h_block)

    kernel_assert(
        best_w_blocks >= 1,
        f"Not enough SBUF space for even 1 weight slot, free={free_space}, needs={bytes_per_h_block}",
    )

    # Step 3: Use remaining space for interleave degree
    remaining_after_weights = free_space - best_w_blocks * bytes_per_h_block
    if cfg.out_in_sb:
        best_interleave = 1
    else:
        best_interleave = min(num_bxs_tiles, remaining_after_weights // out_sb_bytes)

    kernel_assert(
        best_interleave >= 1,
        f"Not enough SBUF space for even 1 out_sb buffer, free={remaining_after_weights}, needs={out_sb_bytes}",
    )

    sizes, offsets = _get_block_sizes_and_offsets(cfg.h_sharded, best_block_size)
    return sizes, offsets, best_w_blocks, best_interleave


def _load_weight_h_block(
    w_sbuf_slot,
    w_shard_hbm: nl.NkiTensor,
    h_block_size: int,
    h_block_offset: int,
    cfg: OutputProjectionTkgConfig,
):
    """
    Load weights for a single h_block into the given SBUF slot.

    Args:
        w_sbuf_slot: TiledTensor or list of pre-allocated weight tensors for one h_block slot.
        w_shard_hbm: [N, D, h_sharded] pre-sharded weight nl.NkiTensor.
        h_block_size: Size of this h_block.
        h_block_offset: Offset of this h_block within h_sharded.
        cfg: Kernel configuration.
    """
    w_h_sliced = w_shard_hbm.slice(dim=2, start=h_block_offset, end=h_block_offset + h_block_size)
    if not cfg.use_double_row:
        for head_idx in affine_range(cfg.n_size):
            nisa.dma_copy(
                src=w_h_sliced.select(dim=0, index=head_idx),
                dst=w_sbuf_slot[head_idx, 0][:, :h_block_size],
            )
    else:
        for head_idx in affine_range(0, cfg.n_size, 2):
            pair_idx = head_idx // 2
            nisa.dma_copy(
                src=w_h_sliced.select(dim=0, index=head_idx),
                dst=w_sbuf_slot[pair_idx, 0][:, 0, :h_block_size],
            )
            nisa.dma_copy(
                src=w_h_sliced.select(dim=0, index=head_idx + 1),
                dst=w_sbuf_slot[pair_idx, 0][:, 1, :h_block_size],
            )


def _output_projection_tkg_impl(
    out_hbm_buffer: Optional[nl.NkiTensor],
    out_hbm_view: Optional[nl.NkiTensor],
    out_sb: Optional[nl.NkiTensor],
    bias_sb: Optional[nl.NkiTensor],
    w_shard_hbm: nl.NkiTensor,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]],
    attn_shuffled: nl.NkiTensor,
    cfg: OutputProjectionTkgConfig,
    sbm: BufferManager,
) -> nl.NkiTensor:
    """
    Core implementation for regular (non-transposed) output projection.

    Computes attention @ weight + bias with output shape [B*S, h_sharded]. Tiles computation
    across H dimension in F_MAX-sized blocks and B*S dimension in P_MAX-sized blocks.

    When manual allocation is active, uses multi-buffering for the bxs tile loop and
    budgets weight blocks based on available SBUF space, loading them on-demand with
    circular buffer indexing.

    Args:
        out_hbm_buffer (Optional[nl.NkiTensor]): Full output buffer in HBM [B*S, H], returned as-is.
        out_hbm_view: Pre-sliced nl.NkiTensor of out_hbm_buffer for this shard [B*S, h_sharded].
        out_sb (Optional[nl.NkiTensor]): Pre-allocated output buffer in SBUF or None.
        bias_sb (Optional[nl.NkiTensor]): [B*S, H] broadcast bias in SBUF, or None if no bias.
        w_shard_hbm: [N, D, h_sharded] pre-sharded weight nl.NkiTensor.
        quant_config (Optional[Union[StaticQuantConfig, RowQuantConfig]]): Quantization config.
        attn_shuffled (nl.NkiTensor): [D, N*B*S], Shuffled attention tensor in SBUF.
        cfg (OutputProjectionTkgConfig): Kernel configuration containing dimensions and options.
        sbm (BufferManager): Buffer manager for SBUF allocation.

    Returns:
        nl.NkiTensor: Output tensor with shape [B*S, H], either from HBM buffer or SBUF.
    """
    kernel_assert(cfg.out_in_sb == (out_sb != None), "Expected pre-allocated out_sb when cfg.out_in_sb == True")
    sbm.open_scope("output_projection_tkg_impl")
    bxs_size = cfg.b_size * cfg.s_size

    w_scale_dtype = quant_config.weight_scale_hbm.dtype if cfg.quantization_type == QuantizationType.ROW else None
    num_bxs_tiles = div_ceil(bxs_size, P_MAX)
    h_block_sizes, h_block_offsets, num_w_h_blocks, out_sb_interleave_degree = _budget_weight_blocks(
        num_bxs_tiles, w_shard_hbm.dtype, cfg, sbm, w_scale_dtype
    )
    num_h_blocks_per_prg = len(h_block_sizes)
    max_h_block_size = h_block_sizes[0]  # First (full-size) block is always the largest
    kernel_assert(
        sum(h_block_sizes) * cfg.num_prgs == cfg.h_size,
        f"Weight blocking mismatch: sum(h_block_sizes) * {cfg.num_prgs} != {cfg.h_size}",
    )

    weight_scale_sb, weight_scale_blocks = _prepare_weight_scales(
        quant_config=quant_config,
        cfg=cfg,
        sbm=sbm,
        h_block_sizes=h_block_sizes,
        h_block_offsets=h_block_offsets,
        num_h_blocks_per_prg=num_h_blocks_per_prg,
    )

    all_weights_preloaded = num_w_h_blocks == num_h_blocks_per_prg

    # Allocate circular buffer slots at max_h_block_size (remainder block fits in same slot)
    w_sbuf_blocks = []
    for _h_block_idx in affine_range(num_w_h_blocks):
        if not cfg.use_double_row:
            w_sb = TiledTensor.alloc(
                grid=(cfg.n_size, 1),
                tile_size=(cfg.d_size, max_h_block_size),
                dtype=w_shard_hbm.dtype,
                buffer=nl.sbuf,
                sbm=sbm,
                align=cfg.align,
            )
        else:
            w_sb = TiledTensor.alloc(
                grid=(cfg.n_size // 2, 1),
                tile_size=(cfg.d_size, 2, max_h_block_size),
                dtype=w_shard_hbm.dtype,
                buffer=nl.sbuf,
                sbm=sbm,
                align=cfg.align,
            )
        w_sbuf_blocks.append(w_sb)

    if all_weights_preloaded:
        for h_block_idx in affine_range(num_h_blocks_per_prg):
            _load_weight_h_block(
                w_sbuf_blocks[h_block_idx],
                w_shard_hbm,
                h_block_sizes[h_block_idx],
                h_block_offsets[h_block_idx],
                cfg,
            )

    global_psum_idx = 0
    sbm.open_scope(interleave_degree=out_sb_interleave_degree, name="bxs_tile_loop")
    # Compute and write out attention @ weight (+ bias) blocks
    matmul_idx = 0  # for load balancing tensor_copy
    for bxs_block in TiledRange(bxs_size, P_MAX):
        if not cfg.out_in_sb:
            out_sb = sbm.alloc_stack(
                (bxs_block.size, cfg.h_sharded), dtype=cfg.io_dtype, buffer=nl.sbuf, align=cfg.align
            )

        for h_block_idx in affine_range(num_h_blocks_per_prg):
            cur_h_block_size = h_block_sizes[h_block_idx]
            cur_h_block_offset = h_block_offsets[h_block_idx]
            w_slot = h_block_idx % num_w_h_blocks
            if not all_weights_preloaded:
                _load_weight_h_block(
                    w_sbuf_blocks[w_slot],
                    w_shard_hbm,
                    cur_h_block_size,
                    cur_h_block_offset,
                    cfg,
                )

            h_block_f_tiles = list(TiledRange(cur_h_block_size, F_MAX))
            n_heads = cfg.n_size if not cfg.use_double_row else cfg.n_size // 2
            num_f = len(h_block_f_tiles)

            # Stationary: reshape to isolate N dim, tile, select current bxs_block
            if not cfg.use_double_row:
                attn_view = attn_shuffled.reshape_dim(1, (n_heads, bxs_size))
                attn_tiled = TiledTensor(attn_view, tile_size=(cfg.d_size, 1, P_MAX))
                stat_tiled = attn_tiled.select(dim=2, index=bxs_block.index).squeeze_dim(0).expand_dim(1)
                mk = {}
            else:
                attn_view = attn_shuffled.reshape_dim(2, (n_heads, bxs_size))
                attn_tiled = TiledTensor(attn_view, tile_size=(cfg.d_size, 2, 1, P_MAX))
                stat_tiled = attn_tiled.select(dim=3, index=bxs_block.index).squeeze_dim(0).squeeze_dim(0).expand_dim(1)
                mk = {"matmul_kwargs": lambda k, c: {"perf_mode": matmul_perf_mode.double_row}}

            # Moving: sub-tile weight buffer on free dim
            mov_tiled = w_sbuf_blocks[w_slot].sub_tile(dim=-1, size=F_MAX, actual_size=cur_h_block_size)

            # PSUM: one per f_tile, with bank rotation
            psum_tiled = TiledTensor.alloc(
                grid=(1, num_f),
                tile_size=(bxs_block.size, F_MAX),
                dtype=nl.float32,
                buffer=nl.psum,
                num_banks=num_f,
            )
            matmul_idx += num_f

            # Output: grid (1, num_f), tile (bxs, F_MAX)
            out_view = out_sb.slice(dim=1, start=cur_h_block_offset, end=cur_h_block_offset + cur_h_block_size)
            out_tiled = TiledTensor(out_view, tile_size=(bxs_block.size, F_MAX))

            # Auxiliaries: same grid (1, num_f)
            auxiliaries = []
            if cfg.quantization_type == QuantizationType.ROW:
                scale_view = (weight_scale_blocks[h_block_idx]).slice(dim=1, start=0, end=cur_h_block_size)
                scale_tiled = TiledTensor(scale_view, tile_size=(bxs_block.size, F_MAX))
                auxiliaries.append(scale_tiled)
            if cfg.has_bias:
                bias_view = bias_sb.slice(dim=1, start=cur_h_block_offset, end=cur_h_block_offset + cur_h_block_size)
                bias_tiled = TiledTensor(bias_view, tile_size=(bxs_block.size, F_MAX))
                auxiliaries.append(bias_tiled)

            engine_toggle = [matmul_idx]

            def on_output_site1(psum_tile, out_tile, *aux_tiles):
                idx = 0
                if cfg.quantization_type == QuantizationType.ROW:
                    nisa.tensor_tensor(dst=out_tile, data1=psum_tile, data2=aux_tiles[idx], op=nl.multiply)
                    idx = idx + 1
                elif cfg.quantization_type == QuantizationType.STATIC:
                    nisa.activation(
                        dst=out_tile, op=nl.copy, data=psum_tile, scale=weight_scale_sb[: bxs_block.size, :]
                    )
                res = out_tile if cfg.is_quantized else psum_tile
                if cfg.has_bias:
                    nisa.tensor_tensor(dst=out_tile, data1=res, data2=aux_tiles[idx], op=nl.add)
                else:
                    if engine_toggle[0] % 2 == 0:
                        nisa.tensor_copy(dst=out_tile, src=res, engine=nisa.scalar_engine)
                    else:
                        nisa.tensor_copy(dst=out_tile, src=res, engine=nisa.vector_engine)
                    engine_toggle[0] = engine_toggle[0] + 1

            matmul_loop_nest(
                stationary=stat_tiled,
                moving=mov_tiled,
                dst_psum=psum_tiled,
                output=out_tiled,
                auxiliaries=auxiliaries,
                on_output=on_output_site1,
                **mk,
            )

        if out_hbm_view != None:
            nisa.dma_copy(
                dst=out_hbm_view.slice(
                    dim=0, start=bxs_block.start_offset, end=bxs_block.start_offset + bxs_block.size
                ),
                src=out_sb,
            )
            sbm.increment_section()
        else:
            sbm.close_scope()
            sbm.close_scope()
            return out_sb

    sbm.close_scope()
    sbm.close_scope()

    return out_hbm_buffer


def _budget_weight_blocks_transpose(
    w_dtype,
    cfg: OutputProjectionTkgConfig,
    sbm: BufferManager,
) -> tuple:
    """
    Choose h_2_block_size, weight slot count, and interleave degree for the
    transpose output projection path.

    Three-step priority:
    1. Block size: Try h_2_size down to 1 (halving). Pick the largest where
       at least 1 weight block + 1 out_sb fits in SBUF. Remainder blocks
       are handled by _get_block_sizes_and_offsets.
    2. Weight slots: Maximize the number of weight blocks that fit.
    3. Interleave: Use remaining space for out_sb multi-buffering across
       h_2 blocks, capped by num_h2_blocks.

    When cfg.out_in_sb is True, the caller already allocated out_sb, so no
    out_sb space is budgeted and interleave degree is always 1.

    Args:
        w_dtype: Weight data type.
        cfg: Kernel configuration.
        sbm: Buffer manager.

    Returns:
        (h_2_block_sizes, h_2_block_offsets, num_w_h2_slots, out_sb_interleave_degree):
        h_2_block_sizes is a list of per-block sizes (length num_h2_blocks),
        h_2_block_offsets is a list of cumulative offsets within h_2_size,
        num_w_h2_slots is the number of weight SBUF slots to allocate
        (may be less than num_h2_blocks for circular buffering),
        and out_sb_interleave_degree is the interleave degree for the h_2 block loop.
    """
    bxs_size = cfg.b_size * cfg.s_size

    if sbm.is_auto_alloc():
        sizes, offsets = _get_block_sizes_and_offsets(cfg.h_2_size, cfg.h_2_size)
        return sizes, offsets, 1, 1

    free_space = sbm.get_free_space()

    # Step 1: Find largest block size where at least 1 weight block fits
    block_size_candidates = []
    size = cfg.h_2_size
    while size >= 1:
        block_size_candidates.append(size)
        # TODO: Investigate if div_ceil will yield better results for
        # cases where floor yields tiny remainders
        size //= 2

    best_block_size = None
    for block_size in block_size_candidates:
        bytes_per_w_slot = align_to(cfg.n_size * cfg.h_1_size * block_size * sizeinbytes(w_dtype), cfg.align)
        out_sb_bytes = align_to(block_size * bxs_size * sizeinbytes(cfg.io_dtype), cfg.align)
        needed = bytes_per_w_slot + (0 if cfg.out_in_sb else out_sb_bytes)
        if needed <= free_space:
            best_block_size = block_size
            break

    kernel_assert(
        best_block_size != None,
        f"Not enough SBUF space for even 1 weight block in transpose path, free={free_space}",
    )

    # Step 2: Maximize weight slots with remaining space
    num_h2_blocks = div_ceil(cfg.h_2_size, best_block_size)
    bytes_per_w_slot = align_to(cfg.n_size * cfg.h_1_size * best_block_size * sizeinbytes(w_dtype), cfg.align)
    out_sb_bytes = align_to(best_block_size * bxs_size * sizeinbytes(cfg.io_dtype), cfg.align)
    if cfg.out_in_sb:
        best_w_slots = min(num_h2_blocks, free_space // bytes_per_w_slot)
    else:
        best_w_slots = min(num_h2_blocks, (free_space - out_sb_bytes) // bytes_per_w_slot)
    kernel_assert(
        best_w_slots >= 1,
        f"Not enough SBUF space for even 1 weight slot in transpose path, free={free_space}, needs={bytes_per_w_slot}",
    )

    # Step 3: Use remaining space for interleave degree
    remaining_after_weights = free_space - best_w_slots * bytes_per_w_slot
    if cfg.out_in_sb:
        best_interleave = 1
    else:
        best_interleave = min(num_h2_blocks, remaining_after_weights // out_sb_bytes)

    kernel_assert(
        best_interleave >= 1,
        f"Not enough SBUF space for even 1 out_sb buffer in transpose path, free={remaining_after_weights}, needs={out_sb_bytes}",
    )

    sizes, offsets = _get_block_sizes_and_offsets(cfg.h_2_size, best_block_size)
    return sizes, offsets, best_w_slots, best_interleave


def _load_weight_h2_block(
    w_sbuf_slot: nl.NkiTensor,
    w_shard_hbm: nl.NkiTensor,
    h_2_block_size: int,
    h_2_block_offset: int,
    max_h_2_block_size: int,
    cfg: OutputProjectionTkgConfig,
):
    """
    Load weights for a single h_2 block into the given SBUF slot.

    Args:
        w_sbuf_slot: Pre-allocated weight tensor in SBUF, shape (d_size, n_size, h_1_size * max_h_2_block_size).
        w_shard_hbm: [N, D, h_sharded] pre-sharded weight nl.NkiTensor.
        h_2_block_size: Number of h_2 values in this block.
        h_2_block_offset: Offset of this block within h_2_size.
        max_h_2_block_size: Allocation size of the weight slot (for dst AP strides).
        cfg: Kernel configuration.
    """
    # src: w_shard_hbm [N, D, h_sharded] -> permute to [D, N, h_sharded] -> reshape -> slice h_2 block
    src_view = (
        w_shard_hbm.permute((1, 0, 2))
        .reshape_dim(2, (cfg.h_1_size, cfg.h_2_size))
        .slice(dim=3, start=h_2_block_offset, end=h_2_block_offset + h_2_block_size)
    )
    # dst: w_sbuf_slot (d_size, n_size, h_1_size * max_h_2_block_size) -> reshape -> slice h_2 block
    dst_view = w_sbuf_slot.reshape_dim(2, (cfg.h_1_size, max_h_2_block_size)).slice(dim=3, start=0, end=h_2_block_size)
    nisa.dma_copy(dst=dst_view, src=src_view)


def _output_projection_tkg_transpose_out_impl(
    out_hbm_buffer: Optional[nl.NkiTensor],
    out_hbm_view: Optional[nl.NkiTensor],
    out_sb: Optional[nl.NkiTensor],
    bias_sb: Optional[nl.NkiTensor],
    w_shard_hbm: nl.NkiTensor,
    quant_config: Optional[Union[StaticQuantConfig, RowQuantConfig]],
    attn_shuffled: nl.NkiTensor,
    cfg: OutputProjectionTkgConfig,
    sbm: BufferManager,
) -> nl.NkiTensor:
    """
    Core implementation for transposed output projection.

    Computes attention @ weight + bias with transposed output shape [H_1, H_2, B*S].
    Uses weights-stationary matmul with attention moving. Packs multiple B*S groups in
    PSUM banks for efficiency. Tiles across h_2 dimension to reduce SBUF pressure from
    weight storage and increase overlap. Uses circular buffering for weight blocks.

    Args:
        out_hbm_buffer (Optional[nl.NkiTensor]): Full output buffer in HBM [h_1, h_0, h_2, bxs], returned as-is.
        out_hbm_view: Pre-sliced nl.NkiTensor of out_hbm_buffer for this shard [h_1, h_2, bxs].
        out_sb (Optional[nl.NkiTensor]): Pre-allocated output buffer in SBUF or None.
        bias_sb (Optional[nl.NkiTensor]): [h_1_size, h_2_size], Pre-prepared bias in SBUF, or None.
        w_shard_hbm: [N, D, h_sharded] pre-sharded weight nl.NkiTensor.
        quant_config (Optional[Union[StaticQuantConfig, RowQuantConfig]]): Quantization config.
        attn_shuffled (nl.NkiTensor): [D, N*B*S], Shuffled attention tensor in SBUF.
        cfg (OutputProjectionTkgConfig): Kernel configuration containing dimensions and options.
        sbm (BufferManager): Buffer manager for SBUF allocation.

    Returns:
        nl.NkiTensor: Output tensor in transposed layout, either from HBM buffer or SBUF.
    """
    sbm.open_scope("output_projection_tkg_transpose_out_impl")
    bxs_size = cfg.b_size * cfg.s_size

    weight_scale_sb = _prepare_weight_scales_transposed(quant_config=quant_config, cfg=cfg, sbm=sbm)

    # Budget weight blocks across h_2 dimension
    h_2_block_sizes, h_2_block_offsets, num_w_h2_slots, out_sb_interleave_degree = _budget_weight_blocks_transpose(
        w_shard_hbm.dtype, cfg, sbm
    )
    num_h2_blocks = len(h_2_block_sizes)
    max_h_2_block_size = h_2_block_sizes[0]
    all_weights_preloaded = num_w_h2_slots == num_h2_blocks

    # Allocate circular weight buffer slots at max block size
    w_sbuf_slots = []
    for _slot_idx in range(num_w_h2_slots):
        w_sbuf_slots.append(
            sbm.alloc_stack(
                (cfg.d_size, cfg.n_size, cfg.h_1_size * max_h_2_block_size),
                dtype=w_shard_hbm.dtype,
                buffer=nl.sbuf,
                align=cfg.align,
            )
        )

    # Preload all weight blocks if they fit
    if all_weights_preloaded:
        for h2_block_idx in affine_range(num_h2_blocks):
            _load_weight_h2_block(
                w_sbuf_slots[h2_block_idx],
                w_shard_hbm,
                h_2_block_sizes[h2_block_idx],
                h_2_block_offsets[h2_block_idx],
                max_h_2_block_size,
                cfg,
            )

    # For out_in_sb, caller provides full-sized out_sb; otherwise allocate per-block
    if cfg.out_in_sb:
        kernel_assert(out_sb != None, "Expected pre-allocated out_sb when cfg.out_in_sb == True")
        full_out_sb = out_sb
    else:
        full_out_sb = None

    sbm.open_scope(interleave_degree=out_sb_interleave_degree, name="h2_block_loop")
    for h2_block_idx in affine_range(num_h2_blocks):
        cur_h_2_block_size = h_2_block_sizes[h2_block_idx]
        cur_h_2_block_offset = h_2_block_offsets[h2_block_idx]
        w_slot = h2_block_idx % num_w_h2_slots

        # Load weights for this h_2 block if not preloaded
        if not all_weights_preloaded:
            _load_weight_h2_block(
                w_sbuf_slots[w_slot],
                w_shard_hbm,
                cur_h_2_block_size,
                cur_h_2_block_offset,
                max_h_2_block_size,
                cfg,
            )

        w_sbuf = w_sbuf_slots[w_slot]

        # Determine out_sb for this block
        if cfg.out_in_sb:
            # Write into the correct slice of the full out_sb
            block_out_sb = full_out_sb
            # out_sb is (h_1_size, h_2_size * bxs_size), we write at offset cur_h_2_block_offset * bxs_size
            out_sb_h_2_size = cfg.h_2_size
        else:
            block_out_sb = sbm.alloc_stack(
                (cfg.h_1_size, cur_h_2_block_size * bxs_size),
                dtype=cfg.io_dtype,
                buffer=nl.sbuf,
                align=cfg.align,
            )
            out_sb_h_2_size = cur_h_2_block_size

        """
        Tiling Strategy (unified):
        - Weights stationary (K=heads, N0=h2), attn moving (K=heads, N1=bxs_tiles)
        - Factored-N: loop iterates N0 × N1, stat indexed by n//N1, mov by n%N1
        - n_packing groups consecutive N tiles into one PSUM bank when bxs <= F_MAX
        """
        n_heads = cfg.n_size if not cfg.use_double_row else cfg.n_size // 2
        num_bxs_tiles = div_ceil(bxs_size, F_MAX)
        bxs_tile_sz = min(bxs_size, F_MAX)
        N_total = cur_h_2_block_size * num_bxs_tiles
        NUM_BS_PER_PSUM_BANK = F_MAX // bxs_tile_sz  # >=1; >1 when bxs <= F_MAX

        # Stationary (weight): grid (n_heads, cur_h2_block_size)
        if not cfg.use_double_row:
            w_view = w_sbuf.reshape_dim(2, (cfg.h_1_size, max_h_2_block_size))
            w_tiled = TiledTensor(w_view, tile_size=(cfg.d_size, 1, cfg.h_1_size, 1))
            stat_tiled = w_tiled.slice(dim=3, start=0, end=cur_h_2_block_size).squeeze_dim(0).squeeze_dim(1)
            mk = {}
        else:
            w_view = w_sbuf.reshape_dim(2, (cfg.h_1_size, max_h_2_block_size))
            w_tiled = TiledTensor(w_view, tile_size=(cfg.d_size, 2, cfg.h_1_size, 1))
            stat_tiled = w_tiled.slice(dim=3, start=0, end=cur_h_2_block_size).squeeze_dim(0).squeeze_dim(1)
            mk = {"matmul_kwargs": lambda k, c: {"perf_mode": matmul_perf_mode.double_row}}

        # Moving (attn): grid (n_heads, num_bxs_tiles)
        if not cfg.use_double_row:
            attn_view = attn_shuffled.reshape_dim(1, (n_heads, bxs_size))
            attn_tiled = TiledTensor(attn_view, tile_size=(cfg.d_size, 1, bxs_tile_sz))
            # grid: (1, n_heads, num_bxs_tiles)
            mov_tiled = attn_tiled.squeeze_dim(0)
            # grid: (n_heads, num_bxs_tiles) — if num_bxs_tiles==1, this is (n_heads, 1) already
        else:
            attn_view = attn_shuffled.reshape_dim(2, (n_heads, bxs_size))
            attn_tiled = TiledTensor(attn_view, tile_size=(cfg.d_size, 2, 1, bxs_tile_sz))
            # grid: (1, 1, n_heads, num_bxs_tiles)
            mov_tiled = attn_tiled.squeeze_dim(0).squeeze_dim(0)
            # grid: (n_heads, num_bxs_tiles) — if num_bxs_tiles==1, this is (n_heads, 1) already

        # PSUM: with n_packing, each bank holds NUM_BS_PER_PSUM_BANK tiles
        num_psum_groups = div_ceil(N_total, NUM_BS_PER_PSUM_BANK)
        psum_tiled = TiledTensor.alloc(
            grid=(1, num_psum_groups),
            tile_size=(cfg.h_1_size, F_MAX),
            dtype=nl.float32,
            buffer=nl.psum,
            num_banks=num_psum_groups,
        )

        # Output: grid (1, num_psum_groups), tile (h1, NUM_BS_PER_PSUM_BANK * bxs_size)
        packed_bxs = min(NUM_BS_PER_PSUM_BANK, cur_h_2_block_size) * bxs_size
        if cfg.out_in_sb:
            out_start = cur_h_2_block_offset * bxs_size
            out_end = (cur_h_2_block_offset + cur_h_2_block_size) * bxs_size
            out_view = block_out_sb.slice(dim=1, start=out_start, end=out_end)
            out_tiled = TiledTensor(out_view, tile_size=(cfg.h_1_size, packed_bxs))
        else:
            out_tiled = TiledTensor(block_out_sb, tile_size=(cfg.h_1_size, packed_bxs))

        # Auxiliaries: grid (1, num_psum_groups), tile (h1, NUM_BS_PER_PSUM_BANK)
        auxiliaries = []
        if cfg.quantization_type == QuantizationType.ROW and weight_scale_sb is not None:
            scale_view = weight_scale_sb.slice(
                dim=1, start=cur_h_2_block_offset, end=cur_h_2_block_offset + cur_h_2_block_size
            )
            scale_tiled = TiledTensor(scale_view, tile_size=(cfg.h_1_size, NUM_BS_PER_PSUM_BANK))
            auxiliaries.append(scale_tiled)
        if cfg.has_bias and bias_sb is not None:
            bias_view = bias_sb.slice(dim=1, start=cur_h_2_block_offset, end=cur_h_2_block_offset + cur_h_2_block_size)
            bias_tiled = TiledTensor(bias_view, tile_size=(cfg.h_1_size, NUM_BS_PER_PSUM_BANK))
            auxiliaries.append(bias_tiled)

        def on_output_site2(psum_tile, out_tile, *aux_tiles):
            idx = 0
            tile_cols = psum_tile.shape[1]
            if cfg.quantization_type == QuantizationType.ROW:
                scale_bc = (aux_tiles[idx]).expand_dim(2).broadcast(dim=2, size=bxs_tile_sz)
                nisa.tensor_tensor(dst=out_tile, data1=psum_tile, data2=scale_bc, op=nl.multiply)
                idx = idx + 1
            elif cfg.quantization_type == QuantizationType.STATIC:
                nisa.activation(dst=out_tile, op=nl.copy, data=psum_tile, scale=weight_scale_sb[: cfg.h_1_size, :])
            res = out_tile if cfg.is_quantized else psum_tile
            if cfg.has_bias:
                bias_bc = (aux_tiles[idx]).expand_dim(2).broadcast(dim=2, size=bxs_tile_sz)
                nisa.tensor_tensor(dst=out_tile, data1=res, data2=bias_bc, op=nl.add)
            elif not cfg.is_quantized:
                nisa.tensor_copy(dst=out_tile, src=psum_tile)

        matmul_loop_nest(
            stationary=stat_tiled,
            moving=mov_tiled,
            dst_psum=psum_tiled,
            stationary_dims={"K": 0, "N": 1},
            moving_dims={"K": 0},
            n_packing=NUM_BS_PER_PSUM_BANK,
            output=out_tiled,
            auxiliaries=auxiliaries,
            on_output=on_output_site2,
            **mk,
        )

        # Store this h_2 block to HBM incrementally
        if not cfg.out_in_sb:
            block_out_reshaped = block_out_sb.reshape((cfg.h_1_size, cur_h_2_block_size, bxs_size))
            nisa.dma_copy(
                dst=(
                    out_hbm_view.slice(dim=1, start=cur_h_2_block_offset, end=cur_h_2_block_offset + cur_h_2_block_size)
                ),
                src=block_out_reshaped,
            )
            sbm.increment_section()

    sbm.close_scope()
    sbm.close_scope()

    return full_out_sb if cfg.out_in_sb else out_hbm_buffer
