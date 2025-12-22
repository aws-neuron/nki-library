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

import math
from dataclasses import dataclass
from typing import Any, Optional

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import matmul_perf_mode
from nki.language import affine_range, static_range

from ..utils.common_types import QuantizationType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_max_positive_value_for_dtype, get_program_sharding_info
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .output_projection_utils import calculate_head_packing

P_MAX = 128
F_MAX = 512

# Conservative limit assuming 2 bytes per scalar and LNC=2, ~20MB SBUF per core for projection weights
MAX_VALIDATED_N_TIMES_H_SIZE = 163840
MAX_VALIDATED_N_TIMES_H_SIZE_FP32 = MAX_VALIDATED_N_TIMES_H_SIZE // 2


def output_projection_tkg(
    attention: nl.ndarray,
    weight: nl.ndarray,
    bias: Optional[nl.ndarray] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    weight_scale: Optional[nl.ndarray] = None,
    input_scale: Optional[nl.ndarray] = None,
    TRANSPOSE_OUT=False,
    OUT_IN_SB=False,
):
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
        attention (nl.ndarray): Input tensor in HBM or SBUF, typically the scores output from an attention block.
            Shape:    [D, B, N, S]
            Indexing: [d, b, n, s]
        weight (nl.ndarray): Weight tensor in HBM
            Shape:    [N * D,     H]
            Indexing: [n * D + d, h]
        bias (Optional[nl.ndarray]): Optional bias tensor in HBM
            Shape:    [1, H]
            Indexing: [1, h]
        quantization_type (QuantizationType): Type of quantization to apply (NONE, STATIC). Default: QuantizationType.NONE.
        weight_scale (Optional[nl.ndarray]): Optional weight scale tensor for static quantization in HBM
            Shape:    [P_MAX, 1]
        input_scale (Optional[nl.ndarray]): Optional input scale tensor for static quantization in HBM
            Shape:    [P_MAX, 1]
        TRANSPOSE_OUT (bool): Whether need to store the output in transposed shape.
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

    Returns:
        out (nl.ndarray): Output tensor in HBM. Shape depends on `TRANSPOSE_OUT` parameter.

    Notes:
        - This kernel supports nl.float32, nl.float16 and nl.bfloat16 data types.
          However, for nl.float32, large inputs may not fit in SBUF.
        - The product B * S must not exceed 128.
        - Head dimension D must not exceed 128.
        - When TRANSPOSE_OUT is False: H must be divisible by 512 * LNC,
          where LNC is the logical neuron core count (1 or 2).
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
    d_original_size, b_size, n_original_size, s_size = attention.shape
    h_size = weight.shape[1]
    io_dtype = attention.dtype
    _, n_prgs, prg_id = get_program_sharding_info()

    # Validate weight shape
    kernel_assert(
        weight.shape[0] == n_original_size * d_original_size,
        f"output_projection_tkg kernel requires weight in shape (N * D = {n_original_size * d_original_size}, H = {h_size}), but got {weight.shape}.\n"
        f"Note: N and D inferred from attention score shape: {attention.shape}.",
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

    # Validate quantization scale shapes
    if quantization_type == QuantizationType.STATIC:
        kernel_assert(
            weight_scale.shape == (P_MAX, 1),
            f"Incorrect shape for weight scale for static per tensor quantization, expected ({P_MAX}, 1), got {weight_scale.shape}",
        )
        kernel_assert(
            input_scale.shape == (P_MAX, 1),
            f"Incorrect shape for qkv scale for static per tensor quantization, expected ({P_MAX}, 1), got {input_scale.shape}",
        )

    # Create and validate configuration
    cfg = OutputProjectionTkgConfig(
        d_original_size=d_original_size,
        b_size=b_size,
        n_original_size=n_original_size,
        s_size=s_size,
        h_size=h_size,
        num_prgs=n_prgs,
        prg_id=prg_id,
        transpose_out=TRANSPOSE_OUT,
        out_in_sb=OUT_IN_SB,
        quantization_type=quantization_type,
        has_bias=(bias != None),
        has_weight_scale=(weight_scale != None),
        has_input_scale=(input_scale != None),
        io_dtype=io_dtype,
        weight_dtype=weight.dtype,
    )

    # Validate configuration
    cfg.validate()

    # Compute derived configuration (head packing + tiling)
    cfg.compute_derived_config()

    quant_dtype = weight.dtype

    # TODO: support padding for odd number of heads
    # n_heads // 2 * batch needs to be multiple of 16 for double row stride access
    use_double_row = (
        quantization_type == QuantizationType.STATIC
        and cfg.n_size % 2 == 0
        and cfg.n_size * b_size % 32 == 0
        and b_size * s_size >= 64
    )

    # ================================================================================
    # Execution
    # ================================================================================

    # Load quantization scales
    weight_scale_sb = None
    input_scale_sb = None
    if quantization_type == QuantizationType.STATIC:
        weight_scale_sb = nl.ndarray(
            weight_scale.shape, dtype=weight_scale.dtype, buffer=nl.sbuf, name="weight_scale_sb"
        )
        input_scale_sb = nl.ndarray(weight_scale.shape, dtype=weight_scale.dtype, buffer=nl.sbuf, name="input_scale_sb")
        nisa.dma_copy(dst=weight_scale_sb, src=weight_scale)
        nisa.dma_copy(dst=input_scale_sb, src=input_scale)
        # pre-apply input scales onto the weight scaling
        nisa.activation(dst=weight_scale_sb, op=nl.copy, data=weight_scale_sb, scale=input_scale_sb)

    w_reshaped = weight.reshape((cfg.n_size, cfg.d_size, h_size))
    attn_shuffled = load_and_shuffle_attn(
        attention=attention,
        quantization_type=quantization_type,
        input_scale_sb=input_scale_sb,
        quant_dtype=quant_dtype,
        d_original_size=d_original_size,
        n_original_size=n_original_size,
        group_size=cfg.group_size,
        use_double_row=use_double_row,
        cfg=cfg,
    )

    if not TRANSPOSE_OUT:
        out = (
            nl.ndarray(
                (b_size * s_size, h_size),
                dtype=io_dtype,
                buffer=nl.shared_hbm,
                name="output_projection_tkg_out",
            )
            if not OUT_IN_SB
            else None
        )

        return _output_projection_tkg_impl(
            out_hbm_buffer=out,
            bias=bias,
            prg_id=prg_id,
            w_reshaped=w_reshaped,
            weight_scale_sb=weight_scale_sb,
            attn_shuffled=attn_shuffled,
            quantization_type=quantization_type,
            io_dtype=io_dtype,
            use_double_row=use_double_row,
            cfg=cfg,
        )

    else:  # TRANSPOSE_OUT == True
        """
        Notes on iteration order:
        
        cfg.h_0_size corresponds to the outermost logical iterator h_0 = prg_id from 0 to cfg.num_prgs - 1. This corresponds to LNC sharding.
        cfg.h_1_size corresponds to the mid logical iterator h_1 from 0 to P_MAX - 1. This is placed in partition dim.
        cfg.h_2_size corresponds to the innermost logical iterator h_2. This is placed in free dim.
        
        Check for h_size % (P_MAX * n_prgs) == 0 above should cover this
        """
        kernel_assert(h_size == cfg.h_0_size * cfg.h_1_size * cfg.h_2_size, "")
        out = (
            nl.ndarray(
                (cfg.h_1_size, cfg.h_0_size, cfg.h_2_size, b_size * s_size),
                dtype=io_dtype,
                buffer=nl.shared_hbm,
                name="output_projection_tkg_out",
            )
            if not OUT_IN_SB
            else None
        )
        return _output_projection_tkg_transpose_out_impl(
            out_hbm_buffer=out,
            bias=bias,
            prg_id=prg_id,
            w_reshaped=w_reshaped,
            weight_scale_sb=weight_scale_sb,
            attn_shuffled=attn_shuffled,
            quantization_type=quantization_type,
            io_dtype=io_dtype,
            use_double_row=use_double_row,
            cfg=cfg,
        )


@dataclass()
class OutputProjectionTkgConfig(nl.NKIObject):
    """Configuration and validation for output projection TKG kernel."""

    # Input dimensions
    d_original_size: int
    b_size: int
    n_original_size: int
    s_size: int
    h_size: int

    # Execution parameters
    num_prgs: int
    prg_id: int

    # Kernel options
    transpose_out: bool
    out_in_sb: bool
    quantization_type: QuantizationType

    # Optional tensors (for validation)
    has_bias: bool = False
    has_weight_scale: bool = False
    has_input_scale: bool = False

    # Data types
    io_dtype: Any = None
    weight_dtype: Any = None

    # Computed fields (set by compute_derived_config)
    n_size: int = None
    d_size: int = None
    group_size: int = None
    h_sharded: int = None
    h_0_size: int = None
    h_1_size: int = None
    h_2_size: int = None

    def validate(self) -> None:
        """Validate all configuration parameters."""

        # Hardware constraints
        kernel_assert(nl.tile_size.pmax == nl.tile_size.gemm_stationary_fmax, "")
        kernel_assert(nl.tile_size.psum_fmax == nl.tile_size.gemm_moving_fmax, "")

        # Shape validation
        kernel_assert(
            self.h_size % self.num_prgs == 0,
            f"output_projection_tkg kernel requires hidden dimension (H = {self.h_size}) to be divisible by logical core size of {self.num_prgs}.",
        )

        # Dimension constraints
        if self.out_in_sb and not self.transpose_out:
            kernel_assert(
                self.b_size * self.s_size <= P_MAX,
                f"When OUT_IN_SB=True and TRANSPOSE_OUT=False, output_projection_tkg kernel does not support (B * S = {self.b_size * self.s_size}) > {P_MAX}.",
            )

        if self.transpose_out:
            kernel_assert(
                self.b_size * self.s_size <= F_MAX,
                f"When TRANSPOSE_OUT=True, output_projection_tkg kernel does not support (B * S = {self.b_size * self.s_size}) > {F_MAX}.",
            )

        kernel_assert(
            self.d_original_size <= P_MAX,
            f"output_projection_tkg kernel does not support head dimension (D = {self.d_original_size}) greater than {P_MAX}.",
        )

        # Quantization validation
        kernel_assert(
            self.quantization_type != QuantizationType.ROW,
            "QuantizationType ROW in not supported",
        )

        if self.quantization_type == QuantizationType.STATIC:
            kernel_assert(
                self.has_weight_scale,
                "weight_scale must be provided when quantization_type is STATIC",
            )
            kernel_assert(
                self.has_input_scale,
                "input_scale must be provided when quantization_type is STATIC",
            )

        # Layout-specific validation
        if not self.transpose_out:
            kernel_assert(
                self.h_size % (F_MAX * self.num_prgs) == 0,
                f"When `TRANSPOSE_OUT` is False, output_projection_tkg kernel requires hidden dimension (H = {self.h_size}) to be a multiple of {F_MAX} * logical core size, where logical core size is {self.num_prgs}.",
            )
        else:
            kernel_assert(
                self.h_size % (P_MAX * self.num_prgs) == 0,
                f"When `TRANSPOSE_OUT` is True, output_projection_tkg kernel requires hidden dimension (H = {self.h_size}) to be a multiple of {P_MAX} * logical core size, where logical core size is {self.num_prgs}.",
            )

            # Size limits for transpose mode
            if self.weight_dtype == nl.float32:
                kernel_assert(
                    self.n_original_size * self.h_size <= MAX_VALIDATED_N_TIMES_H_SIZE_FP32,
                    f"When `TRANSPOSE_OUT` is True and using 32bit floats, output_projection_tkg kernel is not tested for (N * H = {self.n_original_size * self.h_size}) greater than {MAX_VALIDATED_N_TIMES_H_SIZE_FP32}.",
                )
            else:
                kernel_assert(
                    self.n_original_size * self.h_size <= MAX_VALIDATED_N_TIMES_H_SIZE,
                    f"When `TRANSPOSE_OUT` is True, output_projection_tkg kernel is not tested for (N * H = {self.n_original_size * self.h_size}) greater than {MAX_VALIDATED_N_TIMES_H_SIZE}.",
                )

    def compute_derived_config(self) -> None:
        """Compute head packing and tiling parameters."""

        # Head packing optimization
        if self.d_original_size % 32 == 0:
            n_size, d_size, group_size = calculate_head_packing(self.n_original_size, self.d_original_size, P_MAX)
            self.n_size = n_size
            self.d_size = d_size
            self.group_size = group_size
        else:
            self.n_size = self.n_original_size
            self.d_size = self.d_original_size
            self.group_size = 1

        # Tiling parameters
        self.h_sharded = self.h_size // self.num_prgs

        if self.transpose_out:
            self.h_0_size = self.num_prgs
            self.h_1_size = P_MAX
            self.h_2_size = self.h_size // self.num_prgs // P_MAX
        else:
            self.h_0_size = -1
            self.h_1_size = -1
            self.h_2_size = -1


def load_and_shuffle_attn(
    attention: nl.ndarray,
    d_original_size: int,
    n_original_size: int,
    quantization_type: QuantizationType,
    input_scale_sb: Optional[nl.ndarray],
    quant_dtype,
    group_size: int,
    use_double_row: bool,
    cfg: OutputProjectionTkgConfig,
):
    """
    Load attention tensor to SBUF, optionally quantize, and shuffle to optimized layout.

    Transforms attention from [d_original_size, B, n_original_size, S] to [D, N * B * S] or
    [D, 2, N//2 * B * S] for efficient matrix multiplication. When group_size > 1, multiple
    heads are packed into the partition dimension D (head packing optimization). Applies
    static quantization if specified.

    Args:
        attention (nl.ndarray): [d_original_size, B, n_original_size, S], Input attention tensor.
        d_original_size (int): Original head dimension size before packing.
        n_original_size (int): Original number of heads before packing.
        quantization_type (QuantizationType): Type of quantization (NONE or STATIC).
        input_scale_sb (Optional[nl.ndarray]): [P_MAX, 1], Input scale tensor in SBUF for quantization.
        quant_dtype: Target dtype for quantization.
        group_size (int): Number of heads packed together in partition dimension D.
        use_double_row (bool): Whether to use double row layout for matmul.
        cfg (OutputProjectionTkgConfig): Configuration with D=d_original_size*group_size, N=n_original_size//group_size.

    Returns:
        nl.ndarray: Shuffled attention tensor in SBUF. Shape [D, N * B * S] or [D, 2, N//2 * B * S].

    Notes:
        - When group_size > 1, cfg.d_size = d_original_size * group_size and cfg.n_size = n_original_size // group_size
        - The shuffle operation maps n_orig to (n_group, n_offset) where n_group = n_orig // group_size
    """
    if attention.buffer == nl.sbuf:
        attn_sb = attention
    else:
        attn_sb = nl.ndarray(
            (d_original_size, cfg.b_size, n_original_size, cfg.s_size),
            dtype=attention.dtype,
            buffer=nl.sbuf,
        )
        nisa.dma_copy(dst=attn_sb[...], src=attention[...])

    if quantization_type == QuantizationType.STATIC:
        attn_quantized = nl.ndarray(
            (d_original_size, cfg.b_size, n_original_size, cfg.s_size),
            dtype=quant_dtype,
            buffer=nl.sbuf,
        )
        nisa.reciprocal(dst=input_scale_sb, data=input_scale_sb)
        nisa.activation(dst=attn_sb, op=nl.copy, data=attn_sb, scale=input_scale_sb[:d_original_size, :])
        max_pos_val = get_max_positive_value_for_dtype(quant_dtype)
        nisa.tensor_scalar(
            dst=attn_quantized,
            data=attn_sb,
            op0=nl.minimum,
            operand0=max_pos_val,
            op1=nl.maximum,
            operand1=-max_pos_val,
        )
        attn_sb = attn_quantized

    """
    Shuffle from attn_sb[d_original_size, B, n_original_size, S] to attn_shuffled[D, N * B * S]
    Indexing is attn_shuffled[d, n * B * S + b * S + s]
    Combined reshape + shuffle when group_size > 1
    For double row, the shape is attn_shuffled[D, 2, N // 2 * B * S]
    Indexing is attn_shuffled[d, n // (N//2), (n % (N//2) * B * S + b * S + s]
    """
    tensor_shape = (
        (cfg.d_size, 2, cfg.n_size // 2 * cfg.b_size * cfg.s_size)
        if use_double_row
        else (cfg.d_size, cfg.n_size * cfg.b_size * cfg.s_size)
    )

    attn_shuffled = nl.ndarray(
        tensor_shape,
        dtype=attn_sb.dtype,
        buffer=nl.sbuf,
    )
    # Use original n_original_size before it was divided by group_siz
    for n_orig in static_range(n_original_size):
        # Map original n to new (n_group, n_offset) coordinates
        n_group = n_orig // group_size
        n_offset = n_orig % group_size
        nisa.tensor_copy(
            dst=attn_shuffled.ap(
                pattern=[
                    [cfg.n_size * cfg.b_size * cfg.s_size, d_original_size],
                    [cfg.s_size, cfg.b_size],
                    [1, cfg.s_size],
                ],
                offset=n_offset * d_original_size * cfg.n_size * cfg.b_size * cfg.s_size
                + n_group * cfg.b_size * cfg.s_size,
            ),
            src=attn_sb.ap(
                pattern=[
                    [cfg.b_size * n_original_size * cfg.s_size, d_original_size],
                    [n_original_size * cfg.s_size, cfg.b_size],
                    [1, cfg.s_size],
                ],
                offset=cfg.s_size * n_orig,
            ),
        )

    return attn_shuffled


def _output_projection_tkg_impl(
    out_hbm_buffer,
    bias,
    prg_id,
    w_reshaped,
    weight_scale_sb,
    attn_shuffled,
    quantization_type,
    io_dtype,
    use_double_row,
    cfg: OutputProjectionTkgConfig,
):
    """
    Core implementation for regular (non-transposed) output projection.

    Computes attention @ weight + bias with output shape [B*S, H]. Tiles computation
    across H dimension in F_MAX-sized blocks and B*S dimension in P_MAX-sized blocks.

    Args:
        out_hbm_buffer: Output buffer in HBM or None if output in SBUF.
        bias: Optional bias tensor in HBM.
        prg_id (int): Current logical core ID.
        w_reshaped: Weight tensor reshaped to [N, D, H].
        weight_scale_sb: Weight scale tensor in SBUF for quantization.
        attn_shuffled: Shuffled attention tensor in SBUF.
        quantization_type (QuantizationType): Quantization type.
        io_dtype: Output data type.
        use_double_row (bool): Whether using double row matmul mode.
        cfg (OutputProjectionTkgTilingStrategy): Tiling configuration.

    Returns:
        nl.ndarray: Output tensor, either from HBM buffer or SBUF.
    """
    bxs_size = cfg.b_size * cfg.s_size
    num_BxS_blocks = math.ceil(bxs_size / P_MAX)
    BxS_block_size = min(P_MAX, bxs_size)
    if bias != None:
        # Load bias at once, this can be improved if cfg.h_size is large.
        bias_sb_1d = nl.ndarray((1, cfg.h_sharded), dtype=bias.dtype)
        nisa.dma_copy(
            src=bias.ap(
                pattern=[[cfg.h_sharded, 1], [1, cfg.h_sharded]],
                offset=prg_id * cfg.h_sharded,
            ),
            dst=bias_sb_1d,
        )
        bias_sb = nl.ndarray((BxS_block_size, cfg.h_sharded), dtype=bias.dtype, buffer=nl.sbuf)
        # Broadcast bias from [1, cfg.h_sharded] to [B*S, cfg.h_sharded] to match out_sb shape below.
        stream_shuffle_broadcast(bias_sb_1d, bias_sb)

    """
    Potentially load w_reshaped into separate blocks in sbuf to allow prefetching.
    2K block size, otherwise just 1 block.
    TODO: Fine tune this number.
    """
    h_block_size = 2048 if cfg.h_sharded % 2048 == 0 else cfg.h_sharded
    num_h_blocks_per_prg = cfg.h_sharded // h_block_size
    kernel_assert(num_h_blocks_per_prg * h_block_size * cfg.num_prgs == cfg.h_size, "")

    """
    By loading into separate sbuf tensors, we give the compiler finer control over pre-fetching.
    overall shape is [num_h_blocks_per_prg][cfg.n_size][cfg.d_size, h_block_size]
    for double row, shape is [num_h_blocks_per_prg][cfg.n_size // 2][cfg.d_size, 2, h_block_size]
    """
    w_sbuf_blocks = []
    for h_block in affine_range(num_h_blocks_per_prg):
        w_row = []
        if not use_double_row:
            for n in affine_range(cfg.n_size):
                w_tensor = nl.ndarray((cfg.d_size, h_block_size), dtype=w_reshaped.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    src=w_reshaped.ap(
                        pattern=[[cfg.h_size, cfg.d_size], [1, h_block_size]],
                        offset=n * cfg.d_size * cfg.h_size + (prg_id * num_h_blocks_per_prg + h_block) * h_block_size,
                    ),
                    dst=w_tensor,
                )
                w_row.append(w_tensor)
        else:
            for n in affine_range(0, cfg.n_size, 2):
                w_tensor = nl.ndarray((cfg.d_size, 2, h_block_size), dtype=w_reshaped.dtype, buffer=nl.sbuf)
                nisa.dma_copy(
                    src=w_reshaped.ap(
                        pattern=[[cfg.h_size, cfg.d_size], [1, h_block_size]],
                        offset=n * cfg.d_size * cfg.h_size + (prg_id * num_h_blocks_per_prg + h_block) * h_block_size,
                    ),
                    dst=w_tensor[:, 0, :],
                )
                nisa.dma_copy(
                    src=w_reshaped.ap(
                        pattern=[[cfg.h_size, cfg.d_size], [1, h_block_size]],
                        offset=(n + 1) * cfg.d_size * cfg.h_size
                        + (prg_id * num_h_blocks_per_prg + h_block) * h_block_size,
                    ),
                    dst=w_tensor[:, 1, :],
                )
                w_row.append(w_tensor)
        w_sbuf_blocks.append(w_row)

    num_free_dim_tiles_per_h_block = h_block_size // F_MAX

    # Compute and write out attention @ weight (+ bias) blocks
    for BxS_block in affine_range(num_BxS_blocks):
        curr_BxS_block_size = min(P_MAX, bxs_size - BxS_block * P_MAX)

        out_sb = nl.ndarray(
            (curr_BxS_block_size, cfg.h_sharded),
            dtype=io_dtype,
            buffer=nl.sbuf,
        )

        for h_block in affine_range(num_h_blocks_per_prg):
            for h_block_f_tile in affine_range(num_free_dim_tiles_per_h_block):
                res_psum = nl.ndarray((curr_BxS_block_size, F_MAX), dtype=nl.float32, buffer=nl.psum)

                # Accumulate (B*S, F_MAX) tiled attn @ weight blocks for all cfg.n_size heads
                if not use_double_row:
                    for n in affine_range(cfg.n_size):
                        stationary = attn_shuffled[:, nl.ds(n * bxs_size + BxS_block * P_MAX, curr_BxS_block_size)]
                        moving = w_sbuf_blocks[h_block][n][:, nl.ds(h_block_f_tile * F_MAX, F_MAX)]
                        nisa.nc_matmul(res_psum, stationary, moving)
                else:
                    # for double row we use the leading free dimension of 2 and double the data per matmul
                    for n in affine_range(cfg.n_size // 2):
                        stationary = attn_shuffled[:, :, nl.ds(n * bxs_size + BxS_block * P_MAX, curr_BxS_block_size)]
                        moving = w_sbuf_blocks[h_block][n][:, :, nl.ds(h_block_f_tile * F_MAX, F_MAX)]
                        nisa.nc_matmul(res_psum, stationary, moving, perf_mode=matmul_perf_mode.double_row)

                # Read out from psum, possibly adding bias if present.
                h_offset = h_block * h_block_size + h_block_f_tile * F_MAX
                if quantization_type == QuantizationType.STATIC:
                    nisa.activation(
                        dst=out_sb[:, nl.ds(h_offset, F_MAX)],
                        data=res_psum[:, :F_MAX],
                        op=nl.copy,
                        scale=weight_scale_sb[:curr_BxS_block_size, :],
                    )
                    if bias != None:
                        nisa.tensor_tensor(
                            dst=out_sb[:, nl.ds(h_offset, F_MAX)],
                            data1=out_sb[:, nl.ds(h_offset, F_MAX)],
                            data2=bias_sb[:curr_BxS_block_size, nl.ds(h_offset, F_MAX)],
                            op=nl.add,
                        )
                elif bias != None:
                    nisa.tensor_tensor(
                        dst=out_sb[:, nl.ds(h_offset, F_MAX)],
                        data1=res_psum[:, :F_MAX],
                        data2=bias_sb[:curr_BxS_block_size, nl.ds(h_offset, F_MAX)],
                        op=nl.add,
                    )
                else:
                    nisa.tensor_copy(dst=out_sb[:, nl.ds(h_offset, F_MAX)], src=res_psum[:, :F_MAX])

        if out_hbm_buffer is not None:
            nisa.dma_copy(
                dst=out_hbm_buffer[
                    nl.ds(BxS_block * P_MAX, curr_BxS_block_size), nl.ds(prg_id * cfg.h_sharded, cfg.h_sharded)
                ],
                src=out_sb,
            )
        else:
            return out_sb
    return out_hbm_buffer


def _output_projection_tkg_transpose_out_impl(
    out_hbm_buffer,
    bias,
    prg_id,
    w_reshaped,
    weight_scale_sb,
    attn_shuffled,
    quantization_type,
    io_dtype,
    use_double_row,
    cfg: OutputProjectionTkgConfig,
):
    """
    Core implementation for transposed output projection.

    Computes attention @ weight + bias with transposed output shape [H_1, H_0, H_2, B*S].
    Uses weights-stationary matmul with attention moving. Packs multiple B*S groups in
    PSUM banks for efficiency.

    Args:
        out_hbm_buffer: Output buffer in HBM or None if output in SBUF.
        bias: Optional bias tensor in HBM.
        prg_id (int): Current logical core ID.
        w_reshaped: Weight tensor reshaped to [N, D, H].
        weight_scale_sb: Weight scale tensor in SBUF for quantization.
        attn_shuffled: Shuffled attention tensor in SBUF.
        quantization_type (QuantizationType): Quantization type.
        io_dtype: Output data type.
        use_double_row (bool): Whether using double row matmul mode.
        cfg (OutputProjectionTkgTilingStrategy): Tiling configuration.

    Returns:
        nl.ndarray: Output tensor in transposed layout, either from HBM buffer or SBUF.
    """
    bxs_size = cfg.b_size * cfg.s_size
    out_sb = nl.ndarray(
        (cfg.h_1_size, cfg.h_2_size * bxs_size),
        dtype=io_dtype,
        buffer=nl.sbuf,
    )

    if bias != None:
        bias_sb = nl.ndarray((cfg.h_1_size, cfg.h_2_size), dtype=bias.dtype, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=bias_sb,
            src=bias.ap(
                pattern=[[cfg.h_2_size, cfg.h_1_size], [1, cfg.h_2_size]],
                offset=prg_id * cfg.h_1_size * cfg.h_2_size,
            ),
        )

    # Load weights at once.
    w_sbuf = nl.ndarray(
        (cfg.d_size, cfg.n_size, cfg.h_1_size * cfg.h_2_size),
        dtype=w_reshaped.dtype,
        buffer=nl.sbuf,
    )
    nisa.dma_copy(
        dst=w_sbuf.ap(
            pattern=[
                [cfg.n_size * (cfg.h_1_size * cfg.h_2_size), cfg.d_size],
                [cfg.h_1_size * cfg.h_2_size, cfg.n_size],
                [1, cfg.h_1_size * cfg.h_2_size],
            ]
        ),
        src=w_reshaped.ap(
            pattern=[
                [cfg.h_size, cfg.d_size],
                [cfg.d_size * cfg.h_size, cfg.n_size],
                [1, cfg.h_1_size * cfg.h_2_size],
            ],
            offset=(cfg.h_1_size * cfg.h_2_size) * prg_id,
        ),
    )

    """
    Weights stationary, input (attn_shuffled) moving in, with B*S on free-dim each time.
    For a PSUM bank with 512 (F_MAX) entries, we can pack multiple B*S groups (i.e. different values of h_c)
    in one bank and copy the whole bank to SBUF at once.
    """
    NUM_BS_PER_PSUM_BANK = F_MAX // (bxs_size)
    NUM_PSUM_TILES = div_ceil(cfg.h_2_size, NUM_BS_PER_PSUM_BANK)

    for i in affine_range(NUM_PSUM_TILES):
        """
        Accumulate (cfg.h_1_size, F_MAX) sized attn @ weight blocks for all cfg.n_size heads,
        for a total of NUM_PSUM_TILES different resulting psum tiles.
        Note that each PSUM row will have NUM_BS_PER_PSUM_BANK many B*S groups.
        In effect, we are packing part of cfg.h_2_size dimension together with B*S.
        """
        res_psum = nl.ndarray((cfg.h_1_size, F_MAX), dtype=nl.float32, buffer=nl.psum)
        for j in affine_range(NUM_BS_PER_PSUM_BANK):
            h_c_offset = i * NUM_BS_PER_PSUM_BANK + j
            if not use_double_row:
                for n in affine_range(cfg.n_size):
                    moving = attn_shuffled[:, nl.ds(n * bxs_size, bxs_size)]
                    stationary = w_sbuf.ap(
                        pattern=[
                            [cfg.n_size * (cfg.h_1_size * cfg.h_2_size), cfg.d_size],
                            [cfg.h_2_size, cfg.h_1_size],
                        ],
                        offset=n * (cfg.h_1_size * cfg.h_2_size) + h_c_offset,
                    )
                    # if cfg.h_2_size is not divsible by NUM_BS_PER_PSUM_BANK,
                    # the last psum tile may be "incomplete".
                    if h_c_offset < cfg.h_2_size:
                        nisa.nc_matmul(
                            dst=res_psum[
                                :,
                                nl.ds(j * bxs_size, bxs_size),
                            ],
                            stationary=stationary,
                            moving=moving,
                        )
            else:
                for n in affine_range(cfg.n_size // 2):
                    moving = attn_shuffled[:, :, nl.ds(n * bxs_size, bxs_size)]
                    stationary = w_sbuf.ap(
                        pattern=[
                            [cfg.n_size * (cfg.h_1_size * cfg.h_2_size), cfg.d_size],
                            [cfg.h_1_size * cfg.h_2_size, 2],
                            [cfg.h_2_size, cfg.h_1_size],
                        ],
                        offset=n * (cfg.h_1_size * cfg.h_2_size) + h_c_offset,
                    )
                    # if cfg.h_2_size is not divsible by NUM_BS_PER_PSUM_BANK,
                    # the last psum tile may be "incomplete".
                    if h_c_offset < cfg.h_2_size:
                        nisa.nc_matmul(
                            dst=res_psum[
                                :,
                                nl.ds(j * bxs_size, bxs_size),
                            ],
                            stationary=stationary,
                            moving=moving,
                            perf_mode=matmul_perf_mode.double_row,
                        )

        # Last psum tile may be "incomplete" if cfg.h_2_size is not divisible by NUM_BS_PER_PSUM_BANK
        num_BS_for_current_psum_tile = min(NUM_BS_PER_PSUM_BANK, cfg.h_2_size - i * NUM_BS_PER_PSUM_BANK)

        # Read out from psum, possibly adding bias if present.ß
        dst_sb_access_pattern = [
            [cfg.h_2_size * bxs_size, cfg.h_1_size],
            [bxs_size, num_BS_for_current_psum_tile],
            [1, bxs_size],
        ]
        dst_sb_access_offset = i * NUM_BS_PER_PSUM_BANK * bxs_size
        psum_access_pattern = [
            [F_MAX, cfg.h_1_size],
            [bxs_size, num_BS_for_current_psum_tile],
            [1, bxs_size],
        ]
        bias_access_pattern = [
            [cfg.h_2_size, cfg.h_1_size],
            [1, num_BS_for_current_psum_tile],
            [0, bxs_size],
        ]
        bias_access_offset = i * NUM_BS_PER_PSUM_BANK
        if quantization_type == QuantizationType.STATIC:
            nisa.activation(
                dst=out_sb.ap(
                    pattern=dst_sb_access_pattern,
                    offset=dst_sb_access_offset,
                ),
                op=nl.copy,
                data=res_psum.ap(pattern=psum_access_pattern),
                scale=weight_scale_sb[: cfg.h_1_size, :],
            )
            if bias != None:
                nisa.tensor_tensor(
                    dst=out_sb.ap(
                        pattern=dst_sb_access_pattern,
                        offset=dst_sb_access_offset,
                    ),
                    data1=out_sb.ap(
                        pattern=dst_sb_access_pattern,
                        offset=dst_sb_access_offset,
                    ),
                    data2=bias_sb.ap(
                        pattern=bias_access_pattern,
                        offset=bias_access_offset,
                    ),
                    op=nl.add,
                )
        elif bias != None:
            nisa.tensor_tensor(
                dst=out_sb.ap(
                    pattern=dst_sb_access_pattern,
                    offset=dst_sb_access_offset,
                ),
                data1=res_psum.ap(pattern=psum_access_pattern),
                data2=bias_sb.ap(
                    pattern=bias_access_pattern,
                    offset=bias_access_offset,
                ),
                op=nl.add,
            )
        else:
            nisa.tensor_copy(
                dst=out_sb.ap(
                    pattern=dst_sb_access_pattern,
                    offset=dst_sb_access_offset,
                ),
                src=res_psum.ap(pattern=psum_access_pattern),
            )

    # Store out as transposed.
    out_sb = out_sb.reshape((cfg.h_1_size, cfg.h_2_size, bxs_size))

    if out_hbm_buffer is not None:
        nisa.dma_copy(
            dst=out_hbm_buffer.ap(
                pattern=[
                    [cfg.h_0_size * cfg.h_2_size * bxs_size, cfg.h_1_size],
                    [bxs_size, cfg.h_2_size],
                    [1, bxs_size],
                ],
                offset=prg_id * cfg.h_2_size * bxs_size,
            ),
            src=out_sb,
        )
        return out_hbm_buffer
    else:
        return out_sb
