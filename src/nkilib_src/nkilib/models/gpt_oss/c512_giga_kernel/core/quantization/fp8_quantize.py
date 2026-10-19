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
Shared FP8 quantization functions for MLP TKG and MOE TKG kernels.

Provides static (tensor-wise) and row-wise (dynamic) FP8 quantization,
plus a helper to pre-combine dequant scales.
These functions operate on SBUF tensors and are independent of
kernel-specific tiling for reusability across MLP and MOE paths.
"""

from typing import Optional

import nki.isa as nisa
import nki.language as nl

from ..utils.allocator import BufferManager
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_max_positive_value_for_dtype
from ..utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .constants import MINVAL


def static_quantization(
    hidden_state,
    input_dequant_scale,
    quant_scale=None,
    dtype=nl.float8_e4m3fn,
    sbm: Optional[BufferManager] = None,
    quantized=None,
):
    """
    Tensor-wise static FP8 quantization.

    Multiplies input by quant_scale (= 1/dequant_scale), clips to [-FP8_MAXVAL, FP8_MAXVAL].
    The caller can pre-compute quant_scale via nisa.reciprocal to avoid redundant computation
    when the same dequant_scale is reused (e.g., gate and up projections share gate_up_in_scale).

    Pseudocode:
        # Inputs: hidden_state bf16[P, F], input_dequant_scale fp32[P, 1]
        quant_scale = 1 / input_dequant_scale                       # fp32[P, 1]
        input_scaled = hidden_state * quant_scale                   # bf16[P, F]
        quantized = clip(input_scaled, -MAXVAL, MAXVAL)             #  fp8[P, F]

    :param hidden_state: bf16 tensor in SBUF, shape [P, F]
    :param input_dequant_scale: fp32 scalar scale [128, 1] from checkpoint (returned as-is)
    :param quant_scale: optional pre-computed 1/dequant_scale [128, 1]. If None, computed internally.
    :param dtype: FP8 dtype for clipping bound (MAXVAL) and output buffer allocation (default: nl.float8_e4m3fn).
                  Must be OCP-compliant (float8_e4m3fn) when used with float8_e4m3fn_x4 weights.
    :param sbm: optional BufferManager for SBUF allocation management
    :param quantized: optional pre-allocated tensor for storing the quantized output.
        Expected shape: same as hidden_state [P, F], dtype: ``dtype`` param (default fp8_e4m3fn).
        If None, allocated internally.
    :return: (quantized tensor same shape as hidden_state, input_dequant_scale passed through)
    """
    _alloc = sbm.alloc_stack if sbm else nl.ndarray
    max_pos_val = get_max_positive_value_for_dtype(dtype)

    # quant_scale = 1 / input_dequant_scale  →  fp32[P, 1]
    if quant_scale == None:
        quant_scale = _alloc(input_dequant_scale.shape, dtype=nl.float32, buffer=nl.sbuf)  # fp32[P, 1]
        nisa.reciprocal(dst=quant_scale, data=input_dequant_scale)

    # input_scaled = hidden_state * quant_scale  →  bf16[P, F]
    # quant_scale [P, 1] broadcasts over F
    nisa.activation(
        dst=hidden_state,  # bf16[P, F]
        op=nl.copy,
        data=hidden_state,  # bf16[P, F]
        scale=quant_scale,  # fp32[P, 1]  (broadcasts over F)
    )

    # quantized = clip(input_scaled, -MAXVAL, MAXVAL)
    if quantized == None:
        quantized = _alloc(hidden_state.shape, dtype=dtype, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=quantized,
        data=hidden_state,  # bf16[P, F]
        op0=nl.minimum,
        operand0=max_pos_val,
        op1=nl.maximum,
        operand1=-max_pos_val,
    )

    return quantized, input_dequant_scale


def row_quantization(
    hidden_state,
    dtype=nl.float8_e4m3fn,
    sbm: Optional[BufferManager] = None,
    output_dtype=None,
    quantized=None,
    dequant_scale=None,
):
    """
    Row-wise dynamic FP8 quantization.

    Computes per-row absmax, derives dequant_scale = absmax / FP8_MAXVAL (clamped to MINVAL),
    then applies quant_scale = 1/dequant_scale.

    Handles two input layouts by branching on shape rank:

      Rank 2 — [P, F]:
        Standard layout where each partition row is an independent row to quantize.
        Uses tensor_scalar_reduce (reduces across F) to get per-row absmax.

        Pseudocode:
            # Input: hidden_state bf16[P, F]
            absmax = max(abs(hidden_state), dim=-1)            # fp32[P, 1]
            dequant_scale = max(absmax / MAXVAL, MINVAL)       # fp32[P, 1]
            quant_scale = 1 / dequant_scale                    # fp32[P, 1]
            quantized = hidden_state * quant_scale             # fp8[P, F]

      Rank 3 — [P0, BxS, F0]:
        MLP TKG layout where the full token vector spans both the partition dim (P0)
        and free dim (F0). Per-token absmax requires two reduction stages.

        We reshape 3D → 2D [P0, BxS*F0] and process one token at a time:
          Per token t:
            Stage 1: tensor_scalar_reduce on [P0, F0] slice across F0
                     → partial max [P0, 1]
            Stage 2: tensor_partition_reduce across P0 → reduced max [P0, 1]
                     (result on partition 0 only on hardware)
            Stage 3: stream_shuffle_broadcast: replicate partition 0 → all P0
            Apply: nisa.activation with quant_scale [P0, 1] broadcasting over F0

        Note: tensor_partition_reduce writes the result to partition 0 only on
        hardware. To broadcast to all P0 partitions, we use stream_shuffle_broadcast
        which copies partition 0's value to all 128 partitions via nc_stream_shuffle
        (4 iterations of 32-partition quadrant broadcasts). This avoids the DMA
        compiler limitations with [1,1] shaped SBUF tensors.

        Pseudocode:
            # Input: hidden_state bf16[P0, BxS, F0]
            flat = hidden_state.reshape(P0, BxS * F0)
            for t in range(BxS):
                token = flat[:, t*F0:(t+1)*F0]                   # bf16[P0, F0]
                partial_max = max(abs(token), dim=-1)            # fp32[P0, 1]
                reduced_max = partition_reduce(partial_max, max) # fp32[P0, 1] (p0 only)
                dequant = max(reduced_max / MAXVAL, MINVAL)      # fp32[P0, 1] (p0 only)
                dequant_scale[t] = shuffle_broadcast(dequant)    # fp32[P0, 1] (all P0)
                quant_scale = 1 / dequant_scale[t]               # fp32[P0, 1]
                quantized_token = token * quant_scale            # bf16[P0, F0]

        Use cases in MLP TKG:
          - Input quantization: hidden_state is [H0, BxS, H1] from rmsnorm output.
            Per-token absmax spans H = H0 * H1.
          - Intermediate quantization: gate*up result is [I0, BxS, I1] before down projection.
            Per-token absmax spans I = I0 * I1.

    :param hidden_state: bf16 tensor in SBUF, shape [P, F] or [P0, BxS, F0]
    :param dtype: FP8 dtype for get_max_positive_value_for_dtype (default: nl.float8_e4m3)
    :param sbm: optional BufferManager for SBUF allocation management
    :param output_dtype: when set to an fp8 dtype (e.g. nl.float8_e4m3fn), fuse the
        scale+clip into the quantization loop and return fp8 directly, eliminating
        the need for a separate bf16→fp8 tensor_copy at the call site.
        When None (default), returns bf16 for backward compatibility.
    :param quantized: optional pre-allocated tensor for storing the quantized output.
        Rank-2: shape [P, F], dtype hidden_state.dtype (bf16).
        Rank-3 with output_dtype: shape [P0, BxS, F0], dtype ``output_dtype``.
        Rank-3 without output_dtype: shape [P0, BxS, F0], dtype hidden_state.dtype.
        If None, allocated internally.
    :param dequant_scale: optional pre-allocated tensor for storing the dequant scale.
        Rank-2: shape [P, 1], dtype fp32.
        Rank-3: shape [P0, BxS], dtype fp32 (returned reshaped as [P0, BxS, 1]).
        If None, allocated internally.
    :return: (quantized tensor same shape as hidden_state,
              dequant_scale [P, 1] for rank-2 or [P0, BxS, 1] for rank-3)
    """
    rank = len(hidden_state.shape)
    kernel_assert(rank in (2, 3), f"row_quantization expects rank-2 [P, F] or rank-3 [P0, BxS, F0], got rank {rank}")
    if rank == 2:
        return _row_quantization_2d(hidden_state, dtype, sbm, quantized, dequant_scale)
    else:
        return _row_quantization_3d(hidden_state, dtype, sbm, output_dtype, quantized, dequant_scale)


def _row_quantization_2d(hidden_state, dtype, sbm, quantized=None, dequant_scale=None):
    """Rank-2 path: hidden_state is [P, F]. Reduces across F only."""
    _alloc = sbm.alloc_stack if sbm else nl.ndarray
    max_pos_val = get_max_positive_value_for_dtype(dtype)
    p_size = hidden_state.shape[0]

    # absmax = max(abs(hidden_state), dim=-1)  →  fp32[P, 1]
    """
    tensor_scalar_reduce: applies abs element-wise, then reduces across F (free dim)
    producing one max value per partition row.
    """
    if dequant_scale == None:
        dequant_scale = _alloc((p_size, 1), dtype=nl.float32, buffer=nl.sbuf)  # fp32[P, 1]
    abs_buf = _alloc(hidden_state.shape, dtype=hidden_state.dtype, buffer=nl.sbuf)  # bf16[P, F]
    nisa.tensor_scalar_reduce(
        dst=abs_buf,  # bf16[P, F]  (abs values, side output)
        data=hidden_state,  # bf16[P, F]
        op0=nl.abs,
        operand0=0.0,
        reduce_op=nl.maximum,
        reduce_res=dequant_scale,  # fp32[P, 1]  (per-row absmax)
    )

    # dequant_scale = max(absmax / MAXVAL, MINVAL)  →  fp32[P, 1]
    nisa.tensor_scalar(
        dst=dequant_scale,  # fp32[P, 1]
        data=dequant_scale,  # fp32[P, 1]
        op0=nl.multiply,
        operand0=1.0 / max_pos_val,
        op1=nl.maximum,
        operand1=MINVAL,
    )

    # quant_scale = 1 / dequant_scale  →  fp32[P, 1]
    quant_scale = _alloc(dequant_scale.shape, dtype=nl.float32, buffer=nl.sbuf)  # fp32[P, 1]
    nisa.reciprocal(dst=quant_scale, data=dequant_scale)

    # quantized = hidden_state * quant_scale  →  bf16[P, F]
    # quant_scale [P, 1] broadcasts over F
    if quantized == None:
        quantized = _alloc(hidden_state.shape, dtype=hidden_state.dtype, buffer=nl.sbuf)  # bf16[P, F]
    nisa.activation(
        dst=quantized,  # bf16[P, F]
        op=nl.copy,
        data=hidden_state,  # bf16[P, F]
        scale=quant_scale,  # fp32[P, 1]  (broadcasts over F)
    )

    return quantized, dequant_scale


def _row_quantization_3d(hidden_state, dtype, sbm, output_dtype=None, quantized=None, dequant_scale=None):
    """Rank-3 path: hidden_state is [P0, BxS, F0]. Per-token quantization with cross-partition reduction.

    The full token vector spans both the partition dim (P0) and free dim (F0),
    so per-token absmax requires two reduction stages per token.

    Vectorized approach — eliminates the BxS serial loop:
      Scale computation (NO per-token loop):
        1. tensor_scalar(abs) on full [P0, BxS, F0]
        2. tensor_reduce(maximum, axis=[2]) on [P0, BxS, F0] → [P0, BxS]
           Per-partition per-token absmax in ONE instruction instead of BxS iterations.
        3. tensor_partition_reduce(maximum) on [P0, BxS] → [P0, BxS] (p0 only)
        4. tensor_scalar(multiply 1/MAXVAL, maximum MINVAL) on [P0, BxS]
        5. stream_shuffle_broadcast on [P0, BxS] — broadcast p0 → all partitions
        6. reciprocal on [P0, BxS] → quant_scale

      Quantization (F0-loop scale expansion + bulk multiply):
        Expand quant_scale [P0, BxS] → [P0, BxS*F0] by tiling F0 copies then permuting.
        tensor_tensor(multiply) applies all scales in one instruction.
        tensor_scalar(clip ±MAXVAL) clips and casts to fp8 in one instruction.
    """
    _alloc = sbm.alloc_stack if sbm else nl.ndarray
    max_pos_val = get_max_positive_value_for_dtype(dtype)
    emit_fp8 = output_dtype != None
    P0, BxS, F0 = hidden_state.shape

    if emit_fp8:
        out_max_pos_val = get_max_positive_value_for_dtype(output_dtype)

    # ── Phase 1: Vectorized scale computation (no per-token loop) ──

    # Step 1+2: Vectorized abs + per-token reduce across F0 (axis 2)
    """
    tensor_reduce can only reduce the last contiguous dims. For [P0, BxS, F0],
    axis=2 (F0) is the most-minor free axis — exactly what we need.
    First compute abs on the full 3D tensor, then reduce across F0.
    """
    abs_3d = _alloc((P0, BxS, F0), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=abs_3d,
        data=hidden_state,
        op0=nl.abs,
        operand0=0.0,
    )

    # Per-token reduce across F0: [P0, BxS, F0] → [P0, BxS]
    # ONE instruction gives per-partition per-token absmax for ALL tokens.
    partial_max_all = _alloc((P0, BxS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(
        dst=partial_max_all,
        op=nl.maximum,
        data=abs_3d,
        axis=[2],
    )

    # Step 3: Vectorized cross-partition max on [P0, BxS] → [P0, BxS] (p0 only)
    # Reuse partial_max_all buffer (in-place is safe for tensor_partition_reduce).
    nisa.tensor_partition_reduce(partial_max_all, nl.max, partial_max_all)

    # Step 4: Vectorized scale computation on [P0, BxS]
    # dequant_scale = max(reduced_max / MAXVAL, MINVAL)
    if dequant_scale == None:
        dequant_scale = _alloc((P0, BxS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=dequant_scale,
        data=partial_max_all,
        op0=nl.multiply,
        operand0=1.0 / max_pos_val,
        op1=nl.maximum,
        operand1=MINVAL,
    )

    # ── Phase 2: Broadcast reduced scales from partition 0 → all partitions ──
    """
    tensor_partition_reduce writes the correct result to partition 0 but other
    partitions may contain stale/incorrect values. stream_shuffle_broadcast
    copies partition 0's values to all 128 partitions.
    """
    stream_shuffle_broadcast(src=dequant_scale, dst=dequant_scale)

    # ── Phase 2.5: Vectorized reciprocal for all tokens at once ──
    quant_scale_all = _alloc((P0, BxS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.reciprocal(dst=quant_scale_all, data=dequant_scale)

    # ── Phase 3: Vectorized quantize via broadcast scale ──
    """
    Use NkiTensor.broadcast to create a stride-0 view of quant_scale that
    expands [P0, BxS, 1] → [P0, BxS, F0] without materializing the expansion.
    This eliminates the F0-copy loop + permute + tensor_copy.
    """
    quant_scale_3d = quant_scale_all.reshape((P0, BxS, 1)).broadcast(2, F0)
    # Work in 3D [P0, BxS, F0] to avoid reshape on non-contiguous broadcast view.
    input_3d = hidden_state  # already [P0, BxS, F0]

    if emit_fp8:
        scaled_3d = _alloc((P0, BxS, F0), dtype=hidden_state.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=scaled_3d,
            data1=input_3d,
            data2=quant_scale_3d,
            op=nl.multiply,
        )
        if quantized == None:
            quantized = _alloc((P0, BxS, F0), dtype=output_dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=quantized,
            data=scaled_3d,
            op0=nl.minimum,
            operand0=out_max_pos_val,
            op1=nl.maximum,
            operand1=-out_max_pos_val,
        )
    else:
        if quantized == None:
            quantized = _alloc((P0, BxS, F0), dtype=hidden_state.dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(
            dst=quantized,
            data1=input_3d,
            data2=quant_scale_3d,
            op=nl.multiply,
        )

    # quantized is already 3D [P0, BxS, F0]
    dequant_scale_3d = dequant_scale.reshape((P0, BxS, 1))
    return quantized, dequant_scale_3d


def pre_combine_dequant_scales(input_dequant_scale, weight_dequant_scale, sbm: Optional[BufferManager] = None):
    """
    Pre-combine input and weight dequant scales: combined = w_dequant * in_dequant.

    Used by the STATIC_MX path inside projection kernels to fuse two scalar
    dequant scales into one, so the post-matmul dequant is a single
    nisa.activation call instead of two.

    Pseudocode:
        # Inputs: input_dequant_scale fp32[P, 1], weight_dequant_scale fp32[P, 1]
        combined = weight_dequant_scale * input_dequant_scale   # fp32[P, 1]

    :param input_dequant_scale: fp32 [P, 1] per-tensor activation dequant scale
    :param weight_dequant_scale: fp32 [P, 1] per-tensor weight dequant scale
    :param sbm: optional BufferManager for SBUF allocation management
    :return: combined dequant scale fp32 [P, 1]
    """
    _alloc = sbm.alloc_stack if sbm else nl.ndarray
    # combined = weight_dequant_scale * input_dequant_scale  →  fp32[P, F_w]
    # input_dequant_scale [P, 1] broadcasts over F_w when weight_dequant_scale is [P, F_w]
    combined = _alloc(weight_dequant_scale.shape, dtype=nl.float32, buffer=nl.sbuf)  # fp32[P, F_w]
    nisa.activation(
        dst=combined,  # fp32[P, F_w]
        op=nl.copy,
        data=weight_dequant_scale,  # fp32[P, F_w]  (data to scale)
        scale=input_dequant_scale,  # fp32[P, 1]    (broadcasts over F_w)
    )
    return combined
