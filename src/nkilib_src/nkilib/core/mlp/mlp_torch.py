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
PyTorch reference for MLP kernel.
"""

import math
from typing import Optional, Protocol

import neuron_dtypes as dt

# nki.language is needed for the nl.float8_e4m3 custom dtype used in FP8 round-trip
# simulation. There is no pure-PyTorch equivalent for this dtype on CPU — the numpy
# golden reference (golden_quant_mlp) uses the same approach:
#   quantized.astype(quant_dtype).astype(nl.float32)
import nki.language as nl
import numpy as np
import torch

from ..rmsnorm.rmsnorm_mx_prefill_torch import decode_packed_output
from ..subkernels.norm_torch_dispatch import norm_name2func_torch
from ..utils.common_types import (
    ActFnType,
    ComputationMode,
    DtypeMode,
    MLPGateUpWeightLayout,
    NormType,
    QuantizationType,
)
from ..utils.kernel_helpers import get_max_positive_value_for_dtype
from ..utils.lnc_subscriptable import LncSubscriptable
from ..utils.mx_torch_common import (
    get_float32_exp,
    get_mx_fp_max,
    nc_matmul_mx_golden,
    quantize_to_mx,
    unpack_float4_x4,
    unpack_float8_e4m3fn_x4,
    unpack_float8_e5m2_x4,
)
from .mlp_parameters import TKG_BS_SEQLEN_THRESHOLD
from .mlp_tkg.mlp_proj_mx_torch import (
    down_proj_mx_torch_ref,
    gate_up_proj_mx_torch_ref,
)
from .mlp_tkg.projection_mx_constants import _pmax, _psum_fmax, _q_height, _q_width

# FP8 E4M3 max. Tests pick via ``dtype_mode`` so goldens clip to the
# same range as the kernel.
_FP8_E4M3_MAX = get_max_positive_value_for_dtype(nl.float8_e4m3)
_FP8_E4M3FN_MAX = get_max_positive_value_for_dtype(nl.float8_e4m3fn)


# Forward declaration for the module-level docstring (assigned at end of file).
mlp_torch_ref: "LncSubscriptable[_MlpTorchRefFn]"  # noqa: F842 — reassigned at end of file
"""
PyTorch reference implementation for the MLP kernel (mlp.mlp.mlp).

Performs the standard MLP computation using PyTorch operations in float32 precision.
This is a reference implementation for testing the MLP kernel — it implements the same
mathematical operation and supports all quantization modes.

Computation flow:
    if fused_add_tensor is provided:
        hidden_states = hidden_states + fused_add_tensor

    if normalization is applied:
        hidden_states = normalization_type(hidden_states)

    gate_proj_out = hidden_states @ gate_proj_weights_tensor
    act_gate_proj = activation_fn(gate_proj_out)

    up_proj_out = hidden_states @ up_proj_weights_tensor
    hidden_states = act_gate_proj * up_proj_out

    down_proj_out = hidden_states @ down_proj_weights_tensor
    output = down_proj_out

Dimensions:
    B: Batch size
    S: Sequence length
    H: Hidden dimension size
    I: Intermediate dimension size

Usage:
    mlp_torch_ref[lnc](hidden_tensor=..., gate_proj_weights_tensor=..., ...)

Args:
    hidden_tensor (torch.Tensor): Input hidden states tensor. Shape: [B, S, H].
    gate_proj_weights_tensor (torch.Tensor): Gate projection weight matrix. Shape: [H, I].
    up_proj_weights_tensor (torch.Tensor): Up projection weight matrix. Shape: [H, I].
    down_proj_weights_tensor (torch.Tensor): Down projection weight matrix. Shape: [I, H].
    normalization_weights_tensor (torch.Tensor, optional): Normalization weights. Shape: [1, H].
        Required when normalization_type is RMS_NORM or LAYER_NORM.
    gate_proj_bias_tensor (torch.Tensor, optional): Bias for gate projection. Shape: [1, I].
    up_proj_bias_tensor (torch.Tensor, optional): Bias for up projection. Shape: [1, I].
    down_proj_bias_tensor (torch.Tensor, optional): Bias for down projection. Shape: [1, H].
    normalization_bias_tensor (torch.Tensor, optional): Bias for normalization. Shape: [1, H].
        Used when normalization_type is LAYER_NORM. If None, no bias is applied.
    fused_add_tensor (torch.Tensor, optional): Residual tensor for fused addition. Shape: [B, S, H].
    store_fused_add_result (bool): If True, returns the fused add result alongside the
        MLP output. Default: False.
    activation_fn (ActFnType): Activation function type (SiLU, GELU, GELU_Tanh_Approx).
        Default: ActFnType.SiLU.
    normalization_type (NormType): Type of normalization (NO_NORM, RMS_NORM, LAYER_NORM,
        RMS_NORM_SKIP_GAMMA). Default: NormType.NO_NORM.
    quantization_type (QuantizationType): Quantization type. Default: QuantizationType.NONE.
        Supported values:
        - NONE: No quantization
        - ROW: FP8 per-row quantization
        - STATIC: FP8 tensor-wise quantization
        - STATIC_MX: FP8 tensor-wise with MX weight swizzle (CTE only)
        - MX: MXFP quantization (MXFP4/MXFP8, TKG only)
    gate_w_scale (torch.Tensor, optional): Dequantization scales for gate weights.
        Shape: [128, I] for ROW, [128, 1] for STATIC. Required for quantized modes.
    up_w_scale (torch.Tensor, optional): Dequantization scales for up weights.
        Shape: [128, I] for ROW, [128, 1] for STATIC. Required for quantized modes.
    down_w_scale (torch.Tensor, optional): Dequantization scales for down weights.
        Shape: [128, I] for ROW, [128, 1] for STATIC. Required for quantized modes.
    gate_up_in_scale (torch.Tensor, optional): FP8 input scales for gate/up projections.
        Shape: [128, 1]. Required for STATIC/STATIC_MX in CTE mode.
    down_in_scale (torch.Tensor, optional): FP8 input scales for down projection.
        Shape: [128, 1]. Required for STATIC/STATIC_MX in CTE mode.
    quant_clipping_bound (float): Clipping boundary for FP8 row quantization. Default: 0.0.
    output_dtype: Output data type. Default: None (ignored in torch ref).
    store_output_in_sbuf (bool): Store output in SBUF. Default: False.
        Hardware-only parameter, ignored in torch ref.
    eps (float): Epsilon for numerical stability in normalization. Default: 1e-6.
    skip_gate_proj (bool): If True, skips gate projection and applies activation
        directly to up projection output. Default: False.
    use_tkg_gate_up_proj_column_tiling (bool): Column tiling for gate/up projection.
        Default: True. Hardware-only parameter, ignored in torch ref.
    use_tkg_down_proj_column_tiling (bool): Column tiling for down projection.
        Default: True. Hardware-only parameter, ignored in torch ref.
    use_tkg_down_proj_optimized_layout (bool): If True, undoes the optimized down weight
        layout [I, lnc, 128, H//(128*lnc)] back to [I, H] using the lnc value from
        LncSubscriptable. Default: False.
    gate_clamp_upper_limit (float, optional): Upper clamp for gate projection output.
    gate_clamp_lower_limit (float, optional): Lower clamp for gate projection output.
    up_clamp_upper_limit (float, optional): Upper clamp for up projection output.
    up_clamp_lower_limit (float, optional): Lower clamp for up projection output.
    force_cte_mode (bool): Force CTE mode. Default: False.
        Hardware-only parameter, ignored in torch ref.
    sbm: Buffer manager. Default: None.
        Hardware-only parameter, ignored in torch ref.

Returns:
    Dict[str, torch.Tensor]: Dictionary with the following keys:
        - "out": MLP output tensor. Shape: [B, S, H].
        - "add_out" (only when fused_add_tensor is provided and store_fused_add_result
          is True): Result of the fused residual addition. Shape: [B, S, H].

Notes:
    - All computation is performed in float32 precision
    - Hardware-specific parameters (store_output_in_sbuf, use_tkg_gate_up_proj_column_tiling,
      use_tkg_down_proj_column_tiling, force_cte_mode, sbm) are accepted for signature
      compatibility but do not affect the computation
    - use_tkg_down_proj_optimized_layout is NOT hardware-only — it triggers a weight
      layout undo using the lnc value from LncSubscriptable
"""


def _apply_activation(x: torch.Tensor, act_fn: ActFnType) -> torch.Tensor:
    """Apply activation function."""
    if act_fn == ActFnType.SiLU:
        return torch.nn.functional.silu(x)
    elif act_fn == ActFnType.GELU:
        return torch.nn.functional.gelu(x)
    elif act_fn == ActFnType.GELU_Tanh_Approx:
        return torch.nn.functional.gelu(x, approximate="tanh")
    else:
        raise ValueError(f"Unsupported activation function: {act_fn}")


def _apply_clamp(
    x: torch.Tensor,
    upper: Optional[float] = None,
    lower: Optional[float] = None,
) -> torch.Tensor:
    """Clamp tensor to optional upper/lower bounds."""
    if upper is not None:
        x = torch.clamp(x, max=upper)
    if lower is not None:
        x = torch.clamp(x, min=lower)
    return x


def _fp8_round_trip(tensor: torch.Tensor, quant_dtype=nl.float8_e4m3) -> torch.Tensor:
    """Simulate FP8 e4m3 round-trip: cast to fp8 then back to float32.

    ``quant_dtype`` selects the FP8 variant (``nl.float8_e4m3`` for NON_OCP,
    ``nl.float8_e4m3fn`` for OCP); using the wrong one hides real discrepancies
    under loose tolerances.

    This uses Neuron SDK custom numpy dtypes because there is no pure-PyTorch CPU
    equivalent. The numpy golden reference uses the same approach:
        quantized.astype(quant_dtype).astype(nl.float32)
    """
    return torch.from_numpy(tensor.numpy().astype(quant_dtype).astype(np.float32))


def _extract_precomputed_row_scale(hidden_tensor: torch.Tensor, H: int) -> tuple:
    """Extract pre-quantized input and its scale from a ROW quant hidden tensor.

    For CTE ROW quant, the hidden tensor has shape [B, S, H+4] where:
    - First H columns: quantized fp8 values (already cast to float32 by torch_ref_wrapper)
    - Last 4 columns: 4 fp8 bytes encoding a single fp32 per-row scale

    The scale reconstruction: convert the 4 float32 values back to fp8 bytes,
    then view those 4 bytes as a single float32 value.
    """
    quantized_input = hidden_tensor[..., :H]
    scale_cols = hidden_tensor[..., H:]  # [B, S, 4]

    # Convert float32 -> fp8 bytes -> view as float32 to reconstruct the original scale
    scale_np = scale_cols.numpy().astype(nl.float8_e4m3fn)
    input_quant_scale = scale_np.view(np.float32)  # [B, S, 1]
    input_quant_scale = torch.from_numpy(input_quant_scale.astype(np.float32))

    return quantized_input, input_quant_scale.flatten().unsqueeze(1)


def _row_quantize(x: torch.Tensor, clip_bound: float, fp8_max: float = _FP8_E4M3_MAX) -> tuple:
    """Per-row FP8 quantization: compute scale and quantize.

    ``fp8_max`` selects the FP8 range (240 for NON_OCP, 448 for OCP) to match
    the kernel's per-row scale.

    Returns (quantized_tensor, per_row_scale) where scale shape is [..., 1].
    """
    abs_max = x.abs().max(dim=-1, keepdim=True).values
    if clip_bound is not None and clip_bound > 0:
        abs_max = abs_max.clamp(max=clip_bound)
        x = x.clamp(-clip_bound, clip_bound)
    min_scale = torch.tensor(1e-5, dtype=x.dtype)
    quant_scale = torch.max(abs_max / fp8_max, min_scale)
    quant_scale = quant_scale.broadcast_to(x.shape)
    return (x / quant_scale).to(torch.float32), quant_scale.to(torch.float32)


def _static_quantize(x: torch.Tensor, quant_scale: torch.Tensor, fp8_max: float = _FP8_E4M3_MAX) -> torch.Tensor:
    """Static FP8 quantization: scale input by 1/quant_scale, clamp to FP8 range.

    ``fp8_max`` selects the FP8 range (240 for NON_OCP, 448 for OCP) to match
    the kernel's clip.
    """
    scaled = x / quant_scale
    return torch.clamp(scaled, -fp8_max, fp8_max).to(torch.float32)


def _mx_quantize(x: torch.Tensor) -> tuple:
    """MX FP8 quantization following the trn3 microscaling speculation.
    The MX quantization block is a 32-element subsection of the final dim of x.
    """
    Q_AREA = _q_height * _q_width
    FP32_EXP_BIAS = 127
    fp8_max = get_mx_fp_max(nl.float8_e4m3fn_x4)

    # exp = get_float32_exp(x.numpy())
    exp = get_float32_exp(x.numpy()).view(np.int32)
    q_area_shape = x.shape[:-1] + (x.shape[-1] // Q_AREA, Q_AREA)
    exp = exp.reshape(q_area_shape)
    # choose 7 instead of get_mx_max_exp(nl.float8_e4m3fn_x4) because 7 is default
    exp_max = exp.max(axis=-1) - 7
    scale = 2.0 ** (exp_max.astype(np.int32) - FP32_EXP_BIAS)
    scale = np.broadcast_to(np.expand_dims(scale, axis=-1), q_area_shape).reshape(x.shape)
    scale = torch.from_numpy(scale)

    q_x = torch.clamp(x / scale, -fp8_max, fp8_max)
    return q_x.to(torch.float32), scale.to(torch.float32)


def _undo_mx_gate_up_w_reshape(weight: torch.Tensor, layout: MLPGateUpWeightLayout) -> torch.Tensor:
    """Undo MX gate/up weight swizzle for MX quant."""
    if layout == MLPGateUpWeightLayout.H_X4_INNERMOST:
        # fp8[128_H, H/512, I/512, 4_I, 128_I, 4_H]
        shape = list(weight.shape)
        H = shape[0] * shape[1] * shape[5]
        I = shape[2] * shape[3] * shape[4]
        weight = weight.reshape(shape).permute(1, 0, 5, 2, 4, 3).reshape((H, I))
        return weight
    elif layout == MLPGateUpWeightLayout.H_X4_MIDDLE:
        shape = list(weight.shape)
        H = shape[0] * shape[1] * shape[5]
        I = shape[2] * shape[3] * shape[4]
        weight = weight.reshape(shape).permute(1, 5, 0, 2, 4, 3).reshape((H, I))
        return weight
    elif layout == MLPGateUpWeightLayout.CONTIGUOUS:
        return weight.to(torch.float32)


def _undo_mx_down_w_reshape(weight: torch.Tensor) -> torch.Tensor:
    """Undo MX down weight swizzle for MX quant."""
    # fp8[128_I, I/512, H, 4_I]
    shape = list(weight.shape)
    H = shape[2]
    I = shape[0] * shape[1] * shape[3]
    weight = weight.reshape(shape).permute(1, 0, 3, 2).reshape((I, H))
    return weight


# Physical fold constants for CTE MX weight scales (mirror the kernel's load-time fold).
_MX_QUADRANT_SIZE = 32
_MX_PARTITIONS_PER_SLOT = 4
_MX_NUM_SLOTS = 4


def _unfold_mx_gate_up_scale(folded: torch.Tensor, H: int, I: int) -> torch.Tensor:
    """Recover standard gate/up MX scale [16, H/512, I/512, 4, 128] from folded [128, n_packed, I].

    Inverse of the kernel's quadrant fold: partition ``q*32 + (k%4)*4 + r`` of buffer ``k//4``
    holds standard scale row ``q*4 + r`` of H/512 tile ``k``. The recovered ``[16, H/512, I]``
    already carries the physical I order (n_I512, 4, 128), so the final reshape splits I in that
    order without a further transpose.
    """
    n_H512 = H // 512
    n_I512 = math.ceil(I / (_pmax * _q_width))
    scale_flat = torch.zeros((16, n_H512, I), dtype=folded.dtype)
    for q in range(_pmax // _MX_QUADRANT_SIZE):
        for k in range(n_H512):
            slot = k % _MX_NUM_SLOTS
            dst_p = q * _MX_QUADRANT_SIZE + slot * _MX_PARTITIONS_PER_SLOT
            src_p = q * _MX_PARTITIONS_PER_SLOT
            scale_flat[src_p : src_p + _MX_PARTITIONS_PER_SLOT, k, :] = folded[
                dst_p : dst_p + _MX_PARTITIONS_PER_SLOT, k // _MX_NUM_SLOTS, :
            ]
    return scale_flat.reshape(16, n_H512, n_I512, _q_width, _pmax)


def _unfold_mx_down_scale(folded: torch.Tensor) -> torch.Tensor:
    """Recover standard down MX scale [16, I/512, H] from folded [128, I/512, H].

    Inverse of the kernel's quadrant fold: partition ``q*32 + r`` holds standard row ``q*4 + r``.
    """
    _, n_I512, H = folded.shape
    scale = torch.zeros((16, n_I512, H), dtype=folded.dtype)
    for q in range(_pmax // _MX_QUADRANT_SIZE):
        dst_p = q * _MX_PARTITIONS_PER_SLOT
        src_p = q * _MX_QUADRANT_SIZE
        scale[dst_p : dst_p + _MX_PARTITIONS_PER_SLOT, :, :] = folded[src_p : src_p + _MX_PARTITIONS_PER_SLOT, :, :]
    return scale


def _undo_mx_gate_up_sc_reshape(scale: torch.Tensor, layout: MLPGateUpWeightLayout, H: int, I: int) -> torch.Tensor:
    """Undo MX gate/up scale swizzle for MX quant and broadcast to full [H, I]."""
    # Gate/up scale shape: [16, H/512, I/512, 4, 128] (physical I order)
    # Undo the transpose to recover logical I order: (4, 128) -> (128, 4) -> flatten to 512 per tile
    FP32_EXP_BIAS = 127.0
    Q_AREA = _q_height * _q_width
    n_H_scales = scale.shape[0]  # 16
    n_H512 = scale.shape[1]  # H/512
    if layout == MLPGateUpWeightLayout.H_X4_INNERMOST:
        scale_unswizzled = (
            scale.permute(1, 0, 2, 4, 3)
            .reshape(n_H512, n_H_scales, 1, -1)
            .broadcast_to((n_H512, n_H_scales, Q_AREA, I))
            .reshape(H, I)
            .float()
        )
    elif layout == MLPGateUpWeightLayout.H_X4_MIDDLE:
        scale_unswizzled = (
            scale.permute(1, 0, 2, 4, 3)
            .reshape(n_H512, 1, n_H_scales, 1, -1)
            .broadcast_to((n_H512, _q_width, n_H_scales, _q_height, I))
            .reshape(H, I)
            .float()
        )
    return torch.pow(2.0, scale_unswizzled - FP32_EXP_BIAS)


def _undo_mx_down_sc_reshape(scale: torch.Tensor, H: int, I: int) -> torch.Tensor:
    """Broadcase MX down scale to full [I, H]."""
    FP32_EXP_BIAS = 127.0
    n_I_scales, n_I512_down, _ = scale.shape  # [16, I/512, H]
    Q_AREA = _q_height * _q_width
    scale_broadcasted = (
        scale.permute((1, 0, 2)).unsqueeze(2).broadcast_to((n_I512_down, n_I_scales, Q_AREA, H)).reshape(I, H).float()
    )
    return torch.pow(2.0, scale_broadcasted - FP32_EXP_BIAS)


def _is_tkg(bxs: int, mode: ComputationMode):
    if mode == ComputationMode.AUTO:
        return bxs <= TKG_BS_SEQLEN_THRESHOLD
    else:
        return mode == ComputationMode.DECODE


def _mlp_ref_standard(
    hidden: torch.Tensor,
    gate_w: torch.Tensor,
    up_w: torch.Tensor,
    down_w: torch.Tensor,
    quantization_type: QuantizationType,
    gate_w_scale: Optional[torch.Tensor],
    up_w_scale: Optional[torch.Tensor],
    down_w_scale: Optional[torch.Tensor],
    gate_up_in_scale: Optional[torch.Tensor],
    down_in_scale: Optional[torch.Tensor],
    quant_clipping_bound: Optional[float],
    gate_proj_bias_tensor: Optional[torch.Tensor],
    up_proj_bias_tensor: Optional[torch.Tensor],
    down_proj_bias_tensor: Optional[torch.Tensor],
    skip_gate_proj: bool,
    activation_fn: ActFnType,
    gate_clamp_upper_limit: Optional[float],
    gate_clamp_lower_limit: Optional[float],
    up_clamp_upper_limit: Optional[float],
    up_clamp_lower_limit: Optional[float],
    gate_up_w_layout: MLPGateUpWeightLayout,
    mode: ComputationMode = ComputationMode.AUTO,
    fp8_max: float = _FP8_E4M3_MAX,
    fp8_round_trip_dtype=nl.float8_e4m3,
    use_folded_mx_scales: bool = False,
):
    """Standard MLP projection path for NONE / ROW / STATIC / STATIC_MX quantization.

    Mirrors the numpy golden ref's perform_projection pattern: the gate/up/down
    sequence is written once, with quant-specific setup (input preparation,
    scale computation, intermediate quantization) handled before/after.

    Args:
        mode: ComputationMode.AUTO, ComputationMode.PREFILL or ComputationMode.DECODE.
    """
    # For CTE STATIC_MX and ROW_MX: undo MX weight swizzle before matmul
    # (CTE receives swizzled 2D weights that need to be unswizzled for the golden ref)
    if quantization_type.is_mx():
        gate_w = _undo_mx_gate_up_w_reshape(gate_w, gate_up_w_layout)
        up_w = _undo_mx_gate_up_w_reshape(up_w, gate_up_w_layout)
        down_w = _undo_mx_down_w_reshape(down_w)

    # --- Determine whether to quantize activations ---
    # ROW quant always quantizes activations (both TKG and CTE).
    # STATIC/STATIC_MX only quantize in CTE mode (large BxS or mode==2).
    # NONE quant never quantizes.
    B, S = hidden.shape[0], hidden.shape[1]
    H, I = up_w.shape[0], up_w.shape[1]
    BxS = B * S
    is_tkg = _is_tkg(BxS, mode)
    is_llama3_70b_specialized_config = all(
        [
            B == 256,
            S == 1,
            H == 8192,
            I == 3584,
            quantization_type == QuantizationType.STATIC,
        ]
    )
    quantize_activations = quantization_type != QuantizationType.NONE and (
        not is_tkg or quantization_type.is_logical_row() or is_llama3_70b_specialized_config
    )

    # Prepare input and per-row scale (ROW quant only)
    if quantization_type.is_logical_row():
        H_up = up_w.shape[0]
        # Note: we detect pre-quantized input via shape, not dtype, because
        # torch_ref_wrapper converts fp8 inputs to float32 before they reach here.
        # Pre-quantized ROW input has shape [B, S, H+4] (4 extra fp8 bytes encoding
        # a per-row fp32 scale); non-pre-quantized input has shape [B, S, H].
        is_input_prequantized = hidden.shape[-1] > H_up
        if is_input_prequantized:
            if hidden.shape[-1] != H_up + 4:
                raise ValueError(
                    f"Pre-quantized ROW input must include 4 scale bytes: expected {H_up + 4}, got {hidden.shape[-1]}"
                )
            # Pre-quantized input: extract quantized values and embedded per-row scale
            hidden, gate_up_in_scale = _extract_precomputed_row_scale(hidden, H_up)
        else:
            hidden, gate_up_in_scale = _row_quantize(hidden.reshape(BxS, H), quant_clipping_bound, fp8_max=fp8_max)
            gate_up_in_scale = gate_up_in_scale.reshape(BxS, H)
        if quantization_type == QuantizationType.ROW:
            gate_w_scale = gate_w_scale[0] if not skip_gate_proj else None
            up_w_scale = up_w_scale[0]
        elif quantization_type == QuantizationType.ROW_MX:
            # Undo MX row weight scale transpose
            gate_w_scale = gate_w_scale.permute((1, 0, 2)).flatten() if not skip_gate_proj else None
            up_w_scale = up_w_scale.permute((1, 0, 2)).flatten()
        down_w_scale = down_w_scale[0]

        gate_w_scale = gate_w_scale.broadcast_to((H, I)).to(torch.float32) if not skip_gate_proj else None
        up_w_scale = up_w_scale.broadcast_to((H, I)).to(torch.float32)
        down_w_scale = down_w_scale.broadcast_to((I, H)).to(torch.float32)
        gate_up_in_scale = gate_up_in_scale.broadcast_to((BxS, H)).to(torch.float32)
    elif quantization_type.is_logical_static():
        gate_w_scale = torch.full(size=(H, I), fill_value=gate_w_scale[0, 0]) if not skip_gate_proj else None
        up_w_scale = torch.full(size=(H, I), fill_value=up_w_scale[0, 0])
        down_w_scale = torch.full(size=(I, H), fill_value=down_w_scale[0, 0])
        gate_up_in_scale_fill_value = gate_up_in_scale[0, 0] if quantize_activations else 1
        gate_up_in_scale = torch.full(size=(BxS, H), fill_value=gate_up_in_scale_fill_value)
        down_in_scale_fill_value = down_in_scale[0, 0] if quantize_activations else 1
        down_in_scale = torch.full(size=(BxS, I), fill_value=down_in_scale_fill_value)
        if quantize_activations and is_tkg:
            hidden = _fp8_round_trip(
                _static_quantize(hidden.reshape(BxS, H), gate_up_in_scale, fp8_max=fp8_max),
                quant_dtype=fp8_round_trip_dtype,
            )
    elif quantization_type == QuantizationType.MX:
        if use_folded_mx_scales:
            # The kernel receives pre-folded MX weight scales; recover the standard
            # layout the dequant reshape helpers below expect.
            if not skip_gate_proj:
                gate_w_scale = _unfold_mx_gate_up_scale(gate_w_scale, H, I)
            up_w_scale = _unfold_mx_gate_up_scale(up_w_scale, H, I)
            down_w_scale = _unfold_mx_down_scale(down_w_scale)
        gate_w_scale = _undo_mx_gate_up_sc_reshape(gate_w_scale, gate_up_w_layout, H, I) if not skip_gate_proj else None
        up_w_scale = _undo_mx_gate_up_sc_reshape(up_w_scale, gate_up_w_layout, H, I)
        down_w_scale = _undo_mx_down_sc_reshape(down_w_scale, H, I)
        H_up = up_w.shape[0]
        is_input_prequantized = hidden.shape[-1] > H_up
        if is_input_prequantized:
            n_H512 = H_up // 512
            n_packed = math.ceil(n_H512 / 4)
            mx_block_scale_region = n_packed * 128
            tail_size = hidden.shape[-1] - H_up
            if tail_size >= mx_block_scale_region:
                hidden_np = hidden.numpy().reshape(BxS, hidden.shape[-1])
                hidden_fp8 = dt.static_cast(hidden_np, nl.float8_e4m3fn)
                hidden_fp32 = decode_packed_output(hidden_fp8, BxS, H_up, pack_scales=True)
                hidden = torch.from_numpy(hidden_fp32).reshape(BxS, H_up)
                gate_up_in_scale = torch.ones(BxS, H_up)
            elif tail_size == 4:
                hidden, gate_up_in_scale = _extract_precomputed_row_scale(hidden, H_up)
                gate_up_in_scale = gate_up_in_scale.broadcast_to((BxS, H)).to(torch.float32)
            else:
                raise ValueError(
                    f"Pre-quantized MX input tail size {tail_size} doesn't match "
                    f"ROW (4) or MX block-scale ({mx_block_scale_region}) format"
                )
    else:
        gate_w_scale = torch.ones(H, I)
        up_w_scale = torch.ones(H, I)
        down_w_scale = torch.ones(I, H)
        gate_up_in_scale = torch.ones(BxS, H)
        down_in_scale = torch.ones(BxS, I)

    hidden = hidden.reshape(BxS, H)

    gate_b = (
        gate_proj_bias_tensor.broadcast_to((BxS, I)).to(torch.float32)
        if gate_proj_bias_tensor is not None
        else torch.zeros(BxS, I)
    )
    up_b = (
        up_proj_bias_tensor.broadcast_to((BxS, I)).to(torch.float32)
        if up_proj_bias_tensor is not None
        else torch.zeros(BxS, I)
    )
    down_b = (
        down_proj_bias_tensor.broadcast_to((BxS, H)).to(torch.float32)
        if down_proj_bias_tensor is not None
        else torch.zeros(BxS, H)
    )

    # --- Gate/Up projections ---
    if not skip_gate_proj:
        gate_out = (hidden * gate_up_in_scale) @ (gate_w * gate_w_scale) + gate_b
        gate_out = _apply_clamp(gate_out, gate_clamp_upper_limit, gate_clamp_lower_limit)

        up_out = (hidden * gate_up_in_scale) @ (up_w * up_w_scale) + up_b
        up_out = _apply_clamp(up_out, up_clamp_upper_limit, up_clamp_lower_limit)

        intermediate = _apply_activation(gate_out, activation_fn) * up_out
    else:
        up_out = (hidden * gate_up_in_scale) @ (up_w * up_w_scale) + up_b
        up_out = _apply_clamp(up_out, up_clamp_upper_limit, up_clamp_lower_limit)
        intermediate = _apply_activation(up_out, activation_fn)

    # --- Quantization ---
    if quantize_activations:
        if quantization_type == QuantizationType.MX or (quantization_type == QuantizationType.ROW_MX and not is_tkg):
            intermediate, down_in_scale = _mx_quantize(intermediate)
            fp8_round_trip_dtype = nl.float8_e4m3fn
        elif quantization_type.is_logical_row():
            intermediate, down_in_scale = _row_quantize(intermediate, quant_clipping_bound, fp8_max=fp8_max)
        elif quantization_type.is_logical_static():
            intermediate = _static_quantize(intermediate, down_in_scale, fp8_max=fp8_max)
        intermediate = _fp8_round_trip(intermediate, quant_dtype=fp8_round_trip_dtype)

    # --- Down projection ---
    output = (intermediate * down_in_scale) @ (down_w * down_w_scale) + down_b

    return output


def _mx_fused_add_and_norm(hidden_tensor, fused_add_tensor, gamma, normalization_type, norm_bias, eps):
    """Shared: fused add + normalization in MX tiled layout (STATIC_MX / ROW_MX)."""
    B, S, H = hidden_tensor.shape
    BxS = B * S
    n_H512 = H // (_pmax * _q_width)

    hidden_mx = hidden_tensor.to(torch.float32)
    if fused_add_tensor is not None:
        hidden_mx = hidden_mx + fused_add_tensor.to(torch.float32)

    hidden_flat = hidden_mx.reshape(BxS, H)

    if gamma is not None:
        hidden_flat = hidden_flat.reshape(BxS, _q_width, n_H512, _pmax).permute(0, 2, 3, 1).reshape(BxS, H)
        gamma_mx = gamma.reshape(1, _q_width, n_H512, _pmax).permute(0, 2, 3, 1).reshape(1, H)
        norm_gamma_mx = None if normalization_type == NormType.RMS_NORM_SKIP_GAMMA else gamma_mx
        hidden_flat = norm_name2func_torch[normalization_type](hidden_flat, norm_gamma_mx, eps=eps, norm_b=norm_bias)
        hidden_flat = hidden_flat.reshape(BxS, n_H512, _pmax, _q_width).permute(0, 3, 1, 2).reshape(BxS, H)
    else:
        hidden_flat = norm_name2func_torch[normalization_type](hidden_flat, None, eps=eps, norm_b=norm_bias)

    return hidden_flat


def _mx_reshape_hidden_4d(hidden_flat, H):
    """Shared: reshape hidden to 4D MX layout [_pmax, n_H512, T_padded, _q_width] and pad T.

    Uses the interleaved-4 H packing convention (used by MX hardware quantization
    and the _layout_adapter_hbm path):
        reshape(BxS, _q_width, n_H512, _pmax).transpose(3, 2, 0, 1)
    The 4 values in _q_width at (p, h512, t, :) are H1 indices strided by n_H512.
    """
    BxS = hidden_flat.shape[0]
    n_H512 = H // (_pmax * _q_width)
    T_padded = math.ceil(BxS / 4) * 4

    hidden_np = hidden_flat.numpy()
    hidden_4d = hidden_np.reshape(BxS, _q_width, n_H512, _pmax).transpose(3, 2, 0, 1)
    if T_padded > BxS:
        padded = np.zeros((_pmax, n_H512, T_padded, _q_width), dtype=hidden_4d.dtype)
        padded[:, :, :BxS, :] = hidden_4d
        hidden_4d = padded

    return hidden_4d, T_padded, n_H512


def _mx_reshape_hidden_4d_consecutive(hidden_flat, H):
    """Reshape hidden to 4D layout with consecutive-4 H packing for STATIC_MX/ROW_MX.

    Uses the consecutive-4 H packing convention that matches the kernel's
    natural [H0, T, H1] SBUF layout with reinterpret_cast:
        quantized_input.reshape(_pmax, T, n_H512, 4) → reinterpret_cast(fp8_x4)

    In SBUF, H index = p + _pmax * h1, where h1 = h512*4 + q.
    So H = p + 128*(h512*4 + q).

    From [BxS, H]:
        reshape(BxS, n_H512, _q_width, _pmax).transpose(3, 1, 0, 2)
        → [_pmax, n_H512, BxS, _q_width]
    where q at (p, h512, t, q) gives H index p + 128*(h512*4 + q).

    This matches the H-contiguous _fp8_to_gate_up_x4 weight packing.
    """
    BxS = hidden_flat.shape[0]
    n_H512 = H // (_pmax * _q_width)
    T_padded = math.ceil(BxS / 4) * 4

    hidden_np = hidden_flat.numpy()
    # [BxS, H] → [BxS, n_H512, 4, 128] → transpose → [128, n_H512, BxS, 4]
    hidden_4d = hidden_np.reshape(BxS, n_H512, _q_width, _pmax).transpose(3, 1, 0, 2)
    if T_padded > BxS:
        padded = np.zeros((_pmax, n_H512, T_padded, _q_width), dtype=hidden_4d.dtype)
        padded[:, :, :BxS, :] = hidden_4d
        hidden_4d = padded

    return hidden_4d, T_padded, n_H512


def _mx_reshape_hidden_4d_contiguous(hidden_flat, H):
    """Reshape hidden to 4D layout with contiguous-4 H packing.

    Uses the contiguous-4 H packing convention where 4 adjacent H values are
    packed into one x4 element:
        H index at (p, h512, t, q) = 512*h512 + 4*p + q   (stride-1)

    From [BxS, H]:
        reshape(BxS, n_H512, _pmax, _q_width).transpose(2, 1, 0, 3)
        → [_pmax, n_H512, BxS, _q_width]
    where q at (p, h512, t, q) gives H index 512*h512 + 4*p + q.

    This matches _fp8_to_gate_up_x4(contiguous_x4=True) weight packing.
    """
    BxS = hidden_flat.shape[0]
    n_H512 = H // (_pmax * _q_width)
    T_padded = math.ceil(BxS / 4) * 4

    hidden_np = hidden_flat.numpy()
    # [BxS, H] → [BxS, n_H512, 128, 4] → transpose → [128, n_H512, BxS, 4]
    hidden_4d = hidden_np.reshape(BxS, n_H512, _pmax, _q_width).transpose(2, 1, 0, 3)
    if T_padded > BxS:
        padded = np.zeros((_pmax, n_H512, T_padded, _q_width), dtype=hidden_4d.dtype)
        padded[:, :, :BxS, :] = hidden_4d
        hidden_4d = padded

    return hidden_4d, T_padded, n_H512


def _convert_6d_weight_to_x4(weight_6d, I):
    """Convert 6D scalar fp8 weight [128, n_H512, n_I512, 4, 128, 4] to 3D x4 [128, n_H512, I_padded].

    Mirrors the kernel's reshape+view conversion. The trailing dim-4 becomes the x4 pack.
    I_padded is derived from the shape (may be larger than I due to 512-alignment padding).
    """
    import neuron_dtypes as dt

    n_H512 = weight_6d.shape[1]
    I_padded = weight_6d.shape[2] * weight_6d.shape[3] * weight_6d.shape[4]
    # [128, n_H512, n_I512, 4, 128, 4] → [128, n_H512, I_padded*4]
    flat = weight_6d.reshape(_pmax, n_H512, I_padded * _q_width)
    # Pack trailing 4 into x4: reshape to [..., I_padded, 4] then static_cast
    grouped = flat.reshape(_pmax, n_H512, I_padded, _q_width)
    return dt.static_cast(grouped.astype(np.float32), nl.float8_e4m3fn_x4).reshape(_pmax, n_H512, I_padded)


def _convert_4d_down_weight_to_x4(weight_4d, I, H):
    """Convert 4D scalar fp8 down weight [128_I, n_I512, H, 4] to 3D x4 [128_I, n_I512, H].

    Mirrors the kernel's reshape+view conversion. The trailing dim-4 becomes the x4 pack.
    """
    import neuron_dtypes as dt

    p_I = weight_4d.shape[0]
    n_I512 = weight_4d.shape[1]
    # [p_I, n_I512, H, 4] → pack trailing 4 into x4
    return dt.static_cast(weight_4d.astype(np.float32), nl.float8_e4m3fn_x4).reshape(p_I, n_I512, H)


def _mx_gate_up_matmul_and_shuffle(hidden_x4, hidden_dummy_scale, weight_x4, H, I, T_padded):
    """Shared: gate/up matmul with nc_matmul_mx_golden + I-tile shuffle.

    Returns raw result in tiled layout [_pmax, n_I512, T_padded, 4] BEFORE dequant.
    """

    n_H512 = H // (_pmax * _q_width)
    w = weight_x4.transpose(1, 0, 2).reshape((H // _q_width, I))
    w_dummy = np.full((H // _q_width // _q_height, I), 127, dtype=np.uint8)
    h_x4 = hidden_x4.reshape(_pmax, n_H512, T_padded).transpose(1, 0, 2).reshape(H // _q_width, T_padded)
    h_sc = (
        hidden_dummy_scale.reshape(_pmax // _q_height, n_H512, T_padded)
        .transpose(1, 0, 2)
        .reshape(H // _q_width // _q_height, T_padded)
    )

    result = nc_matmul_mx_golden(stationary_x4=w, moving_x4=h_x4, stationary_scale=w_dummy, moving_scale=h_sc)

    n_I512 = math.ceil(I / _psum_fmax)
    res_shfl = np.zeros((_pmax, n_I512, T_padded, _q_width), dtype=result.dtype)
    for i in range(n_I512):
        rows_filled = min(_psum_fmax, I - i * _psum_fmax)
        cur = result[i * _psum_fmax : i * _psum_fmax + rows_filled, :]
        rows_padded = math.ceil(rows_filled / 8) * 8
        if rows_padded > rows_filled:
            cur = np.pad(cur, ((0, rows_padded - rows_filled), (0, 0)))
        res_shfl[: rows_padded // _q_width, i, :, :] = cur.reshape(
            _q_width, rows_padded // _q_width, T_padded
        ).transpose(1, 2, 0)
    return res_shfl


def _mx_add_bias(res_shfl, bias, I):
    """Shared: add bias to gate/up projection result."""
    if bias is not None:
        bias_np = bias.numpy() if isinstance(bias, torch.Tensor) else np.asarray(bias)
        if I < _psum_fmax:
            bias_np = np.pad(bias_np, ((0, _pmax - bias_np.shape[0]), *((0, 0),) * (bias_np.ndim - 1)))
        res_shfl += bias_np[:, :, np.newaxis, :]
    return res_shfl


def _mx_down_proj_matmul(inter_x4, down_proj_weights_tensor, H, I, T_padded):
    """Shared: down projection tile loop with nc_matmul_mx_golden. Returns raw result BEFORE dequant."""

    n_I512 = math.ceil(I / _psum_fmax)
    H1 = H // _pmax
    inter_x4_3d = inter_x4.reshape(_pmax, n_I512, T_padded)
    inter_dummy = np.full((_pmax // _q_height, n_I512, T_padded), 127, dtype=np.uint8)
    p_I = down_proj_weights_tensor.shape[0]
    w_sb = np.zeros((_pmax, n_I512, H), dtype=down_proj_weights_tensor.dtype)
    w_sb[:p_I, :, :] = down_proj_weights_tensor
    w_dummy = np.full((_pmax // _q_height, n_I512, H), 127, dtype=np.uint8)

    out = np.zeros((_pmax, H1, T_padded), dtype=np.float32)
    for i_H1 in range(H1):
        h_s, h_e = i_H1 * _pmax, (i_H1 + 1) * _pmax
        tile = np.zeros((_pmax, T_padded), dtype=np.float32)
        for i_I512 in range(n_I512):
            tile += nc_matmul_mx_golden(
                stationary_x4=w_sb[:, i_I512, h_s:h_e],
                moving_x4=inter_x4_3d[:, i_I512, :],
                stationary_scale=w_dummy[:, i_I512, h_s:h_e],
                moving_scale=inter_dummy[:, i_I512, :],
            )
        out[:, i_H1, :] = tile
    return out


def _mx_transpose_output(out, down_proj_bias_tensor, B, S, H, BxS, T_padded):
    """Shared: transpose [_pmax, H1, T] → [BxS, H] + optional bias."""
    H1 = H // _pmax
    output = np.zeros((T_padded, H), dtype=np.float32)
    for i_H1 in range(H1):
        output[:, i_H1 * _pmax : (i_H1 + 1) * _pmax] = out[:, i_H1, :].T
    output = output[:BxS, :]
    if down_proj_bias_tensor is not None:
        down_b = (
            down_proj_bias_tensor.to(torch.float32).numpy()
            if isinstance(down_proj_bias_tensor, torch.Tensor)
            else np.asarray(down_proj_bias_tensor, dtype=np.float32)
        )
        output += down_b.reshape((1, H))
    return torch.from_numpy(output.astype(np.float32)).reshape(B, S, H)


def _mlp_ref_row_mx(
    hidden_tensor,
    fused_add_tensor,
    gate_proj_weights_tensor,
    up_proj_weights_tensor,
    down_proj_weights_tensor,
    gamma,
    normalization_type,
    norm_bias,
    eps,
    gate_w_scale,
    up_w_scale,
    down_w_scale,
    gate_proj_bias_tensor,
    up_proj_bias_tensor,
    down_proj_bias_tensor,
    skip_gate_proj,
    activation_fn,
    gate_clamp_upper_limit,
    gate_clamp_lower_limit,
    up_clamp_upper_limit,
    up_clamp_lower_limit,
    use_contiguous_x4_gate_up=False,
):
    """ROW_MX quantization path: per-token row-wise dynamic quantization with dummy MX scales."""
    import ml_dtypes

    B, S, H = hidden_tensor.shape
    I = gate_proj_weights_tensor.shape[-1]
    # For 6D scalar fp8 weights, derive real I from down weight shape
    if hasattr(gate_proj_weights_tensor, 'ndim') and gate_proj_weights_tensor.ndim == 6:
        p_I = down_proj_weights_tensor.shape[0]
        I = (
            p_I * _q_width
            if p_I < _pmax
            else gate_proj_weights_tensor.shape[2]
            * gate_proj_weights_tensor.shape[3]
            * gate_proj_weights_tensor.shape[4]
        )
    BxS = B * S
    fp8_max = 448.0  # max for float8_e4m3fn (OCP, no NaN — matches kernel's row_quantization dtype)
    bf16 = ml_dtypes.bfloat16
    MINVAL = 1e-5

    hidden_flat = _mx_fused_add_and_norm(hidden_tensor, fused_add_tensor, gamma, normalization_type, norm_bias, eps)
    if use_contiguous_x4_gate_up:
        hidden_4d, T_padded, n_H512 = _mx_reshape_hidden_4d_contiguous(hidden_flat, H)
    else:
        hidden_4d, T_padded, n_H512 = _mx_reshape_hidden_4d_consecutive(hidden_flat, H)

    def row_quant_to_x4(data_4d):
        P, n_tiles, T, Q = data_4d.shape
        data_3d = data_4d.transpose(0, 2, 1, 3).reshape(P, T, n_tiles * Q)
        dequant_scale = np.zeros((P, T, 1), dtype=np.float32)
        for t in range(T):
            global_max = np.max(np.abs(data_3d[:, t, :].astype(np.float32)))
            dequant_scale[:, t, 0] = max(global_max / fp8_max, MINVAL)
        quant_scale = 1.0 / dequant_scale
        scaled = np.clip((data_3d.astype(np.float32) * quant_scale).astype(bf16), bf16(-fp8_max), bf16(fp8_max))
        scaled_4d = scaled.reshape(P, T, n_tiles, Q).transpose(0, 2, 1, 3)
        return dt.static_cast(scaled_4d.reshape(P, -1).astype(np.float32), nl.float8_e4m3fn_x4), dequant_scale

    hidden_x4, input_dequant_scale = row_quant_to_x4(hidden_4d)
    hidden_dummy_scale = np.full((_pmax // _q_height, hidden_x4.shape[1]), 127, dtype=np.uint8)

    # Convert 6D scalar fp8 weights to 3D x4 if needed (slice to real I after padding)
    if hasattr(gate_proj_weights_tensor, 'ndim') and gate_proj_weights_tensor.ndim == 6:
        gate_np = (
            gate_proj_weights_tensor.numpy()
            if isinstance(gate_proj_weights_tensor, torch.Tensor)
            else np.asarray(gate_proj_weights_tensor)
        )
        up_np = (
            up_proj_weights_tensor.numpy()
            if isinstance(up_proj_weights_tensor, torch.Tensor)
            else np.asarray(up_proj_weights_tensor)
        )
        gate_proj_weights_tensor = _convert_6d_weight_to_x4(gate_np, I)[:, :, :I]
        up_proj_weights_tensor = _convert_6d_weight_to_x4(up_np, I)[:, :, :I]
    if hasattr(down_proj_weights_tensor, 'ndim') and down_proj_weights_tensor.ndim == 4:
        down_np = (
            down_proj_weights_tensor.numpy()
            if isinstance(down_proj_weights_tensor, torch.Tensor)
            else np.asarray(down_proj_weights_tensor)
        )
        down_proj_weights_tensor = _convert_4d_down_weight_to_x4(down_np, I, H)

    def gate_up_proj_row_mx(weight_x4, w_dequant_scale, bias):
        res_shfl = _mx_gate_up_matmul_and_shuffle(hidden_x4, hidden_dummy_scale, weight_x4, H, I, T_padded)
        w_np = w_dequant_scale.numpy() if isinstance(w_dequant_scale, torch.Tensor) else np.asarray(w_dequant_scale)
        n_I512 = math.ceil(I / _psum_fmax)
        for i_I in range(n_I512):
            for i_q in range(_q_width):
                i_col = i_I * _q_width + i_q
                if i_col < w_np.shape[1]:
                    res_shfl[:, i_I, :, i_q] *= w_np[:, i_col : i_col + 1]
        for i_I in range(n_I512):
            for i_q in range(_q_width):
                res_shfl[:, i_I, :, i_q] *= input_dequant_scale[:, :, 0]
        return _mx_add_bias(res_shfl, bias, I)

    gate_b = gate_proj_bias_tensor if gate_proj_bias_tensor is not None else None
    up_b = up_proj_bias_tensor if up_proj_bias_tensor is not None else None

    up_out = torch.from_numpy(gate_up_proj_row_mx(up_proj_weights_tensor, up_w_scale, up_b))
    up_out = _apply_clamp(up_out, up_clamp_upper_limit, up_clamp_lower_limit)
    if not skip_gate_proj:
        gate_out = torch.from_numpy(gate_up_proj_row_mx(gate_proj_weights_tensor, gate_w_scale, gate_b))
        gate_out = _apply_clamp(gate_out, gate_clamp_upper_limit, gate_clamp_lower_limit)
        mult = _apply_activation(gate_out, activation_fn) * up_out
    else:
        mult = _apply_activation(up_out, activation_fn)

    inter_x4, inter_dequant_scale = row_quant_to_x4(mult.float().numpy())
    out = _mx_down_proj_matmul(inter_x4, down_proj_weights_tensor, H, I, T_padded)

    down_np = down_w_scale.numpy() if isinstance(down_w_scale, torch.Tensor) else np.asarray(down_w_scale)
    H1 = H // _pmax
    for i_H1 in range(H1):
        out[:, i_H1, :] *= down_np[:, i_H1 : i_H1 + 1]
    for i_H1 in range(H1):
        out[:, i_H1, :] *= inter_dequant_scale[:, :, 0]

    return _mx_transpose_output(out, down_proj_bias_tensor, B, S, H, BxS, T_padded)


def _mlp_ref_static_mx(
    hidden_tensor,
    fused_add_tensor,
    gate_proj_weights_tensor,
    up_proj_weights_tensor,
    down_proj_weights_tensor,
    gamma,
    normalization_type,
    norm_bias,
    eps,
    gate_w_scale,
    up_w_scale,
    down_w_scale,
    gate_up_in_scale,
    down_in_scale,
    gate_proj_bias_tensor,
    up_proj_bias_tensor,
    down_proj_bias_tensor,
    skip_gate_proj,
    activation_fn,
    gate_clamp_upper_limit,
    gate_clamp_lower_limit,
    up_clamp_upper_limit,
    up_clamp_lower_limit,
    use_contiguous_x4_gate_up=False,
):
    """STATIC_MX quantization path for MLP torch reference.

    Uses MX matmul infrastructure with dummy scales (all 127 = 2^0 = 1.0)
    and per-tensor static quantization. Mirrors the numpy norm_mlp_ref_static_mx.

    Key differences from MX path:
    - Hidden is statically quantized (per-tensor scale) instead of MX block quantized
    - All MX scales are dummy (127) — the real dequant uses combined_scale = in_scale * w_scale
    - Intermediate is also statically quantized before down projection
    """
    import ml_dtypes

    B, S, H = hidden_tensor.shape
    I = gate_proj_weights_tensor.shape[-1]
    # For 6D scalar fp8 weights, derive real I from down weight shape
    if hasattr(gate_proj_weights_tensor, 'ndim') and gate_proj_weights_tensor.ndim == 6:
        p_I = down_proj_weights_tensor.shape[0]
        I = (
            p_I * _q_width
            if p_I < _pmax
            else gate_proj_weights_tensor.shape[2]
            * gate_proj_weights_tensor.shape[3]
            * gate_proj_weights_tensor.shape[4]
        )
    BxS = B * S
    fp8_max = 448.0  # max for float8_e4m3fn_x4
    bf16 = ml_dtypes.bfloat16

    hidden_flat = _mx_fused_add_and_norm(hidden_tensor, fused_add_tensor, gamma, normalization_type, norm_bias, eps)
    if use_contiguous_x4_gate_up:
        hidden_4d, T_padded, n_H512 = _mx_reshape_hidden_4d_contiguous(hidden_flat, H)
    else:
        hidden_4d, T_padded, n_H512 = _mx_reshape_hidden_4d_consecutive(hidden_flat, H)
    hidden_mx_flat = hidden_4d.reshape(_pmax, -1)

    if isinstance(gate_up_in_scale, torch.Tensor):
        gate_up_in_scale_val = float(gate_up_in_scale.ravel()[0].item())
    elif isinstance(gate_up_in_scale, np.ndarray):
        gate_up_in_scale_val = float(gate_up_in_scale.ravel()[0])
    else:
        gate_up_in_scale_val = float(gate_up_in_scale)
    quant_scale = np.float32(1.0 / gate_up_in_scale_val)
    data_bf16 = hidden_mx_flat.astype(bf16)
    scaled = (data_bf16.astype(np.float32) * quant_scale).astype(bf16)
    scaled = np.clip(scaled, bf16(-fp8_max), bf16(fp8_max))

    hidden_x4 = dt.static_cast(scaled.astype(np.float32), nl.float8_e4m3fn_x4)
    hidden_dummy_scale = np.full((_pmax // _q_height, hidden_x4.shape[1]), 127, dtype=np.uint8)

    # Convert 6D scalar fp8 weights to 3D x4 if needed (slice to real I after padding)
    if hasattr(gate_proj_weights_tensor, 'ndim') and gate_proj_weights_tensor.ndim == 6:
        gate_np = (
            gate_proj_weights_tensor.numpy()
            if isinstance(gate_proj_weights_tensor, torch.Tensor)
            else np.asarray(gate_proj_weights_tensor)
        )
        up_np = (
            up_proj_weights_tensor.numpy()
            if isinstance(up_proj_weights_tensor, torch.Tensor)
            else np.asarray(up_proj_weights_tensor)
        )
        gate_proj_weights_tensor = _convert_6d_weight_to_x4(gate_np, I)[:, :, :I]
        up_proj_weights_tensor = _convert_6d_weight_to_x4(up_np, I)[:, :, :I]
    if hasattr(down_proj_weights_tensor, 'ndim') and down_proj_weights_tensor.ndim == 4:
        down_np = (
            down_proj_weights_tensor.numpy()
            if isinstance(down_proj_weights_tensor, torch.Tensor)
            else np.asarray(down_proj_weights_tensor)
        )
        down_proj_weights_tensor = _convert_4d_down_weight_to_x4(down_np, I, H)

    def _gate_up_proj_static_mx(weight_x4, w_dequant_scale, bias, in_dequant_scale):
        res_shfl = _mx_gate_up_matmul_and_shuffle(hidden_x4, hidden_dummy_scale, weight_x4, H, I, T_padded)
        in_val = (
            float(in_dequant_scale.ravel()[0].item())
            if isinstance(in_dequant_scale, torch.Tensor)
            else float(np.asarray(in_dequant_scale).ravel()[0])
        )
        w_val = (
            float(w_dequant_scale.ravel()[0].item())
            if isinstance(w_dequant_scale, torch.Tensor)
            else float(np.asarray(w_dequant_scale).ravel()[0])
        )
        res_shfl = res_shfl * (in_val * w_val)
        return _mx_add_bias(res_shfl, bias, I)

    gate_b = gate_proj_bias_tensor if gate_proj_bias_tensor is not None else None
    up_b = up_proj_bias_tensor if up_proj_bias_tensor is not None else None

    gate_out = torch.from_numpy(
        _gate_up_proj_static_mx(gate_proj_weights_tensor, gate_w_scale, gate_b, gate_up_in_scale)
    )
    up_out = torch.from_numpy(_gate_up_proj_static_mx(up_proj_weights_tensor, up_w_scale, up_b, gate_up_in_scale))
    gate_out = _apply_clamp(gate_out, gate_clamp_upper_limit, gate_clamp_lower_limit)
    up_out = _apply_clamp(up_out, up_clamp_upper_limit, up_clamp_lower_limit)

    if not skip_gate_proj:
        mult = _apply_activation(gate_out, activation_fn) * up_out
    else:
        mult = _apply_activation(up_out, activation_fn)

    down_in_scale_val = (
        float(down_in_scale.ravel()[0].item())
        if isinstance(down_in_scale, torch.Tensor)
        else float(np.asarray(down_in_scale).ravel()[0])
    )
    down_w_scale_val = (
        float(down_w_scale.ravel()[0].item())
        if isinstance(down_w_scale, torch.Tensor)
        else float(np.asarray(down_w_scale).ravel()[0])
    )

    n_I512 = math.ceil(I / _psum_fmax)
    inter_flat = mult.float().numpy().astype(np.float32).reshape(_pmax, n_I512 * T_padded * _q_width)
    quant_scale_down = np.float32(1.0 / down_in_scale_val)
    inter_bf16 = inter_flat.astype(bf16)
    inter_scaled = np.clip(
        (inter_bf16.astype(np.float32) * quant_scale_down).astype(bf16), bf16(-fp8_max), bf16(fp8_max)
    )
    inter_x4 = dt.static_cast(inter_scaled.astype(np.float32), nl.float8_e4m3fn_x4)

    out = _mx_down_proj_matmul(inter_x4, down_proj_weights_tensor, H, I, T_padded)
    result = out * (down_in_scale_val * down_w_scale_val)

    return _mx_transpose_output(result, down_proj_bias_tensor, B, S, H, BxS, T_padded)


def _mlp_ref_mx(
    hidden_tensor,
    fused_add_tensor,
    gate_proj_weights_tensor,
    up_proj_weights_tensor,
    down_proj_weights_tensor,
    gamma,
    normalization_type,
    norm_bias,
    eps,
    gate_w_scale,
    up_w_scale,
    down_w_scale,
    gate_proj_bias_tensor,
    up_proj_bias_tensor,
    down_proj_bias_tensor,
    skip_gate_proj,
    activation_fn,
    gate_clamp_upper_limit,
    gate_clamp_lower_limit,
    up_clamp_upper_limit,
    up_clamp_lower_limit,
):
    """MX quantization path for MLP torch reference.

    Delegates to gate_up_proj_mx_torch_ref and down_proj_mx_torch_ref.
    MX normalization requires special layout handling: hidden and gamma must be
    reshaped to MX tiled layout before normalization, then reshaped back.
    """
    _pmax = 128
    _q_width = 4

    B, S, H = hidden_tensor.shape
    I = gate_proj_weights_tensor.shape[-1]
    BxS = B * S

    # Derive pre-norm hidden from original inputs (normalization in the caller
    # was skipped for MX — MX needs special tiled layout for normalization).
    hidden_mx = hidden_tensor.to(torch.float32)
    if fused_add_tensor is not None:
        hidden_mx = hidden_mx + fused_add_tensor.to(torch.float32)

    # Reshape to [BxS, H] for MX processing
    hidden_flat = hidden_mx.reshape(BxS, H)

    # Apply normalization in MX tiled layout
    if gamma is not None:
        n_H512 = H // (_pmax * _q_width)
        hidden_flat = hidden_flat.reshape(BxS, _q_width, n_H512, _pmax).permute(0, 2, 3, 1).reshape(BxS, H)
        gamma_mx = gamma.reshape(1, _q_width, n_H512, _pmax).permute(0, 2, 3, 1).reshape(1, H)
        norm_gamma_mx = None if normalization_type == NormType.RMS_NORM_SKIP_GAMMA else gamma_mx
        hidden_flat = norm_name2func_torch[normalization_type](
            hidden_flat,
            norm_gamma_mx,
            eps=eps,
            norm_b=norm_bias,
        )
        hidden_flat = hidden_flat.reshape(BxS, n_H512, _pmax, _q_width).permute(0, 3, 1, 2).reshape(BxS, H)
    else:
        hidden_flat = norm_name2func_torch[normalization_type](
            hidden_flat,
            None,
            eps=eps,
            norm_b=norm_bias,
        )

    n_H512 = H // (_pmax * _q_width)

    # Select weight unpack function based on quant_dtype
    gate_w_np = gate_proj_weights_tensor
    dtype_str = str(gate_w_np.dtype) if hasattr(gate_w_np, 'dtype') else ''
    is_float4 = 'float4' in dtype_str
    is_float8_e5m2 = 'float8_e5m2' in dtype_str
    if is_float4:
        w_unpack = unpack_float4_x4
    elif is_float8_e5m2:
        w_unpack = unpack_float8_e5m2_x4
    else:
        w_unpack = unpack_float8_e4m3fn_x4

    # Cast to bfloat16 to match kernel SBUF precision at each stage.
    # The kernel performs normalization in bfloat16 on SBUF, then quantizes to mxfp8.
    # Gate/up matmul outputs go through PSUM (float32) -> SBUF (bfloat16).
    # Activation and element-wise multiply happen in bfloat16.
    hidden_flat = hidden_flat.to(torch.bfloat16).to(torch.float32)

    # Quantize hidden to mxfp8
    h = hidden_flat.numpy().reshape(BxS, _q_width, n_H512, _pmax).transpose(3, 2, 0, 1).reshape(_pmax, -1)
    hidden_mx_qtz, hidden_scale = quantize_to_mx(h, nl.float8_e4m3fn_x4)
    hidden_mx_qtz = hidden_mx_qtz.reshape(_pmax, n_H512, BxS)
    hidden_scale_t = torch.from_numpy(hidden_scale.reshape(_pmax // 8, n_H512, BxS))

    def _to_torch_scale(scale):
        return scale if isinstance(scale, torch.Tensor) else torch.from_numpy(np.asarray(scale))

    # Gate projection
    gate_out = gate_up_proj_mx_torch_ref(
        hidden_mx_qtz,
        hidden_scale_t,
        gate_w_np,
        _to_torch_scale(gate_w_scale),
        gate_proj_bias_tensor.to(torch.float32) if gate_proj_bias_tensor is not None else None,
        H,
        I,
        BxS,
        hidden_unpack_fn=unpack_float8_e4m3fn_x4,
        weight_unpack_fn=w_unpack,
    )["out"]

    # Up projection
    up_out = gate_up_proj_mx_torch_ref(
        hidden_mx_qtz,
        hidden_scale_t,
        up_proj_weights_tensor,
        _to_torch_scale(up_w_scale),
        up_proj_bias_tensor.to(torch.float32) if up_proj_bias_tensor is not None else None,
        H,
        I,
        BxS,
        hidden_unpack_fn=unpack_float8_e4m3fn_x4,
        weight_unpack_fn=w_unpack,
    )["out"]

    # Cast to bfloat16 to match kernel SBUF precision (PSUM float32 -> SBUF bfloat16)
    gate_out = gate_out.to(torch.bfloat16).to(torch.float32)
    up_out = up_out.to(torch.bfloat16).to(torch.float32)

    # Clamp
    gate_out = _apply_clamp(gate_out, gate_clamp_upper_limit, gate_clamp_lower_limit)
    up_out = _apply_clamp(up_out, up_clamp_upper_limit, up_clamp_lower_limit)

    # Activation + multiply (or skip_gate)
    if not skip_gate_proj:
        mult = _apply_activation(gate_out, activation_fn) * up_out
    else:
        mult = _apply_activation(up_out, activation_fn)

    # Cast to bfloat16 to match kernel SBUF precision before down projection quantization
    mult = mult.to(torch.bfloat16).to(torch.float32)

    # Down projection
    down_result = down_proj_mx_torch_ref(
        mult,
        down_proj_weights_tensor,
        _to_torch_scale(down_w_scale),
        down_proj_bias_tensor.to(torch.float32) if down_proj_bias_tensor is not None else None,
        H,
        I,
        BxS,
        weight_unpack_fn=w_unpack,
    )["out"]

    return down_result.reshape(B, S, H)


def _mlp_torch_ref_impl(
    hidden_tensor,
    gate_proj_weights_tensor,
    up_proj_weights_tensor,
    down_proj_weights_tensor,
    normalization_weights_tensor=None,
    gate_proj_bias_tensor=None,
    up_proj_bias_tensor=None,
    down_proj_bias_tensor=None,
    normalization_bias_tensor=None,
    fused_add_tensor=None,
    store_fused_add_result=False,
    activation_fn=ActFnType.SiLU,
    normalization_type=NormType.NO_NORM,
    quantization_type=QuantizationType.NONE,
    gate_w_scale=None,
    up_w_scale=None,
    down_w_scale=None,
    gate_up_in_scale=None,
    down_in_scale=None,
    quant_clipping_bound=0.0,
    output_dtype=None,
    store_output_in_sbuf=False,
    eps=1e-6,
    skip_gate_proj=False,
    use_tkg_gate_up_proj_column_tiling=True,
    use_tkg_down_proj_column_tiling=True,
    use_tkg_down_proj_optimized_layout=False,
    use_contiguous_x4_gate_up=False,
    gate_clamp_upper_limit=None,
    gate_clamp_lower_limit=None,
    up_clamp_upper_limit=None,
    up_clamp_lower_limit=None,
    force_cte_mode=False,
    mode=ComputationMode.AUTO,
    sbm=None,
    mx_dummy_scale_hbm=None,  # noqa: ARG001 — hardware-only buffer, kernel-side parity
    transposed_in=False,
    transposed_out=False,
    # FP8 E4M3 dtype for STATIC / ROW quant ref (matches kernel's allocation).
    # NON_OCP (default) → 240 clip, OCP → 448.
    dtype_mode: DtypeMode = DtypeMode.NON_OCP,
    gate_up_w_layout=MLPGateUpWeightLayout.CONTIGUOUS,
    use_folded_mx_scales=False,
) -> dict:
    # --- Input validation ---
    is_tkg = _is_tkg(hidden_tensor.shape[0] * hidden_tensor.shape[1], mode)
    if normalization_type in (NormType.RMS_NORM, NormType.LAYER_NORM):
        if normalization_weights_tensor is None:
            raise ValueError(f"normalization_weights_tensor required when normalization_type is {normalization_type}")
    if quantization_type.is_logical_static() or quantization_type.is_logical_row():
        if gate_w_scale is None:
            raise ValueError(f"gate_w_scale required for {quantization_type} quantization")
        if up_w_scale is None:
            raise ValueError(f"up_w_scale required for {quantization_type} quantization")
        if down_w_scale is None:
            raise ValueError(f"down_w_scale required for {quantization_type} quantization")
        if quantization_type.is_logical_static():
            if gate_up_in_scale is None:
                raise ValueError(f"gate_up_in_scale required for {quantization_type} in CTE mode")
            if down_in_scale is None:
                raise ValueError(f"down_in_scale required for {quantization_type} in CTE mode")
    if use_tkg_down_proj_optimized_layout:
        if not hasattr(mlp_torch_ref, 'lnc') or mlp_torch_ref.lnc is None:
            raise ValueError(
                "use_tkg_down_proj_optimized_layout=True requires lnc to be set via mlp_torch_ref[lnc](...)"
            )

    # --- Convert inputs to float32 because CPU doesn't support half precision ---
    # MX quant weights are numpy arrays with packed dtypes (float8_e4m3fn_x4 etc.)
    # that cannot be converted via .to(torch.float32). The MX path handles them directly.
    if transposed_in and len(hidden_tensor.shape) == 4:
        # Convert transposed [H0, n_prgs, H1_shard, BxS] back to [1, BxS, H] for torch ref
        H0, n_prgs, H1_shard, BxS = hidden_tensor.shape
        hidden = hidden_tensor.to(torch.float32).permute(3, 1, 0, 2).reshape(BxS, n_prgs * H0 * H1_shard).unsqueeze(0)
    else:
        hidden = hidden_tensor.to(torch.float32)
    # Detect STATIC_MX mode: TKG uses 3D x4-packed numpy weights, CTE uses 2D torch weights
    _is_mx_tkg = quantization_type == QuantizationType.MX and is_tkg
    _is_static_mx_tkg = quantization_type == QuantizationType.STATIC_MX and is_tkg
    _is_row_mx_tkg = quantization_type == QuantizationType.ROW_MX and is_tkg
    if quantization_type.is_mx() and not is_tkg:
        # MX/TKG-STATIC_MX path uses gate_proj_weights_tensor directly (x4-packed numpy)
        gate_proj_weights_tensor = gate_proj_weights_tensor.to(torch.float32)
        up_proj_weights_tensor = up_proj_weights_tensor.to(torch.float32)
        down_proj_weights_tensor = down_proj_weights_tensor.to(torch.float32)
    gamma = normalization_weights_tensor.to(torch.float32) if normalization_weights_tensor is not None else None
    norm_bias = normalization_bias_tensor.to(torch.float32) if normalization_bias_tensor is not None else None

    # Undo optimized down weight layout if needed
    if use_tkg_down_proj_optimized_layout:
        LNC = mlp_torch_ref.lnc
        I, H = down_proj_weights_tensor.shape
        down_proj_weights_tensor = (
            down_proj_weights_tensor.reshape((I, LNC, H // 128 // LNC, 128)).permute(0, 1, 3, 2).reshape((I, H))
        )

    # Fused add: add residual to hidden before normalization
    add_out = None
    if fused_add_tensor is not None:
        fused_add = fused_add_tensor.to(torch.float32)
        hidden = hidden + fused_add
        if store_fused_add_result:
            add_out = hidden.clone()

    # Normalization (non-MX paths use norm_name2func_torch from test_kernel_common;
    # MX/TKG-STATIC_MX path handles normalization separately due to special layout requirements)
    if not (_is_mx_tkg or _is_static_mx_tkg or _is_row_mx_tkg):
        # For RMS_NORM_SKIP_GAMMA, pass gamma=None so rms_norm_torch_ref skips the
        # gamma multiply (norm_name2func_torch maps SKIP_GAMMA to rms_norm_torch_ref).
        norm_gamma = None if normalization_type == NormType.RMS_NORM_SKIP_GAMMA else gamma
        hidden = norm_name2func_torch[normalization_type](
            hidden,
            norm_gamma,
            eps=eps,
            norm_b=norm_bias,
        )

    # H_X4_INNERMOST uses contiguous hidden reshape (4 adjacent H values packed)
    _cx4 = gate_up_w_layout == MLPGateUpWeightLayout.H_X4_INNERMOST

    if _is_row_mx_tkg:
        output = _mlp_ref_row_mx(
            hidden_tensor=hidden_tensor,
            fused_add_tensor=fused_add_tensor,
            gate_proj_weights_tensor=gate_proj_weights_tensor,
            up_proj_weights_tensor=up_proj_weights_tensor,
            down_proj_weights_tensor=down_proj_weights_tensor,
            gamma=gamma,
            normalization_type=normalization_type,
            norm_bias=norm_bias,
            eps=eps,
            gate_w_scale=gate_w_scale,
            up_w_scale=up_w_scale,
            down_w_scale=down_w_scale,
            gate_proj_bias_tensor=gate_proj_bias_tensor,
            up_proj_bias_tensor=up_proj_bias_tensor,
            down_proj_bias_tensor=down_proj_bias_tensor,
            skip_gate_proj=skip_gate_proj,
            activation_fn=activation_fn,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            use_contiguous_x4_gate_up=_cx4,
        )
    elif _is_static_mx_tkg:
        output = _mlp_ref_static_mx(
            hidden_tensor=hidden_tensor,
            fused_add_tensor=fused_add_tensor,
            gate_proj_weights_tensor=gate_proj_weights_tensor,
            up_proj_weights_tensor=up_proj_weights_tensor,
            down_proj_weights_tensor=down_proj_weights_tensor,
            gamma=gamma,
            normalization_type=normalization_type,
            norm_bias=norm_bias,
            eps=eps,
            gate_w_scale=gate_w_scale,
            up_w_scale=up_w_scale,
            down_w_scale=down_w_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            gate_proj_bias_tensor=gate_proj_bias_tensor,
            up_proj_bias_tensor=up_proj_bias_tensor,
            down_proj_bias_tensor=down_proj_bias_tensor,
            skip_gate_proj=skip_gate_proj,
            activation_fn=activation_fn,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            use_contiguous_x4_gate_up=_cx4,
        )
    elif not _is_mx_tkg:
        # Callers pre-resolve AUTO (see ``resolve_dtype_mode_for_torch_ref``);
        # the torch ref runs on CPU and can't query hardware directly.
        assert dtype_mode != DtypeMode.AUTO, (  # noqa: S101
            "mlp_torch_ref requires DtypeMode.AUTO to be pre-resolved by the caller."
        )
        # OCP → 448, NON_OCP → 240. Must match the kernel's chosen FP8 range.
        _fp8_max = _FP8_E4M3FN_MAX if dtype_mode == DtypeMode.OCP else _FP8_E4M3_MAX
        _fp8_round_trip_dtype = nl.float8_e4m3fn if dtype_mode == DtypeMode.OCP else nl.float8_e4m3
        output = _mlp_ref_standard(
            hidden=hidden,
            gate_w=gate_proj_weights_tensor,
            up_w=up_proj_weights_tensor,
            down_w=down_proj_weights_tensor,
            quantization_type=quantization_type,
            gate_w_scale=gate_w_scale,
            up_w_scale=up_w_scale,
            down_w_scale=down_w_scale,
            gate_up_in_scale=gate_up_in_scale,
            down_in_scale=down_in_scale,
            quant_clipping_bound=quant_clipping_bound,
            gate_proj_bias_tensor=gate_proj_bias_tensor,
            up_proj_bias_tensor=up_proj_bias_tensor,
            down_proj_bias_tensor=down_proj_bias_tensor,
            skip_gate_proj=skip_gate_proj,
            activation_fn=activation_fn,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
            gate_up_w_layout=gate_up_w_layout,
            mode=mode,
            fp8_max=_fp8_max,
            fp8_round_trip_dtype=_fp8_round_trip_dtype,
            use_folded_mx_scales=use_folded_mx_scales,
        )
    else:
        output = _mlp_ref_mx(
            hidden_tensor=hidden_tensor,
            fused_add_tensor=fused_add_tensor,
            gate_proj_weights_tensor=gate_proj_weights_tensor,
            up_proj_weights_tensor=up_proj_weights_tensor,
            down_proj_weights_tensor=down_proj_weights_tensor,
            gamma=gamma,
            normalization_type=normalization_type,
            norm_bias=norm_bias,
            eps=eps,
            gate_w_scale=gate_w_scale,
            up_w_scale=up_w_scale,
            down_w_scale=down_w_scale,
            gate_proj_bias_tensor=gate_proj_bias_tensor,
            up_proj_bias_tensor=up_proj_bias_tensor,
            down_proj_bias_tensor=down_proj_bias_tensor,
            skip_gate_proj=skip_gate_proj,
            activation_fn=activation_fn,
            gate_clamp_upper_limit=gate_clamp_upper_limit,
            gate_clamp_lower_limit=gate_clamp_lower_limit,
            up_clamp_upper_limit=up_clamp_upper_limit,
            up_clamp_lower_limit=up_clamp_lower_limit,
        )

    if output_dtype is None:
        output_dtype = torch.bfloat16
    elif isinstance(output_dtype, str):
        output_dtype = getattr(torch, output_dtype)

    # Build result dict
    result = {"out": output.to(output_dtype)}
    if fused_add_tensor is not None and store_fused_add_result:
        result["add_out"] = add_out

    return result


# NOTE: This Protocol must match the signature of _mlp_torch_ref_impl above.
# It exists solely for IDE parameter hints when using mlp_torch_ref[lnc](...).
class _MlpTorchRefFn(Protocol):
    def __call__(
        self,
        hidden_tensor,
        gate_proj_weights_tensor,
        up_proj_weights_tensor,
        down_proj_weights_tensor,
        normalization_weights_tensor=None,
        gate_proj_bias_tensor=None,
        up_proj_bias_tensor=None,
        down_proj_bias_tensor=None,
        normalization_bias_tensor=None,
        fused_add_tensor=None,
        store_fused_add_result=False,
        activation_fn=ActFnType.SiLU,
        normalization_type=NormType.NO_NORM,
        quantization_type=QuantizationType.NONE,
        gate_w_scale=None,
        up_w_scale=None,
        down_w_scale=None,
        gate_up_in_scale=None,
        down_in_scale=None,
        quant_clipping_bound=0.0,
        output_dtype=None,
        store_output_in_sbuf=False,
        eps=1e-6,
        skip_gate_proj=False,
        use_tkg_gate_up_proj_column_tiling=True,
        use_tkg_down_proj_column_tiling=True,
        use_tkg_down_proj_optimized_layout=False,
        use_contiguous_x4_gate_up=False,
        gate_clamp_upper_limit=None,
        gate_clamp_lower_limit=None,
        up_clamp_upper_limit=None,
        up_clamp_lower_limit=None,
        force_cte_mode=False,
        mode=ComputationMode.AUTO,
        sbm=None,
        mx_dummy_scale_hbm=None,
        transposed_in=False,
        transposed_out=False,
        dtype_mode: DtypeMode = DtypeMode.NON_OCP,
        gate_up_w_layout=MLPGateUpWeightLayout.CONTIGUOUS,
        use_folded_mx_scales=False,
    ) -> dict:
        """
        PyTorch reference implementation for the MLP kernel (mlp.mlp.mlp).

        Performs the standard MLP computation using PyTorch operations in float32 precision.
        This is a reference implementation for testing the MLP kernel — it implements the same
        mathematical operation and supports all quantization modes.

        Computation flow:
            if fused_add_tensor is provided:
                hidden_states = hidden_states + fused_add_tensor

            if normalization is applied:
                hidden_states = normalization_type(hidden_states)

            gate_proj_out = hidden_states @ gate_proj_weights_tensor
            act_gate_proj = activation_fn(gate_proj_out)

            up_proj_out = hidden_states @ up_proj_weights_tensor
            hidden_states = act_gate_proj * up_proj_out

            down_proj_out = hidden_states @ down_proj_weights_tensor
            output = down_proj_out

        Dimensions:
            B: Batch size
            S: Sequence length
            H: Hidden dimension size
            I: Intermediate dimension size

        Usage:
            mlp_torch_ref[lnc](hidden_tensor=..., gate_proj_weights_tensor=..., ...)

        Args:
            hidden_tensor (torch.Tensor): Input hidden states tensor. Shape: [B, S, H].
            gate_proj_weights_tensor (torch.Tensor): Gate projection weight matrix. Shape: [H, I].
            up_proj_weights_tensor (torch.Tensor): Up projection weight matrix. Shape: [H, I].
            down_proj_weights_tensor (torch.Tensor): Down projection weight matrix. Shape: [I, H].
            normalization_weights_tensor (torch.Tensor, optional): Normalization weights. Shape: [1, H].
                Required when normalization_type is RMS_NORM or LAYER_NORM.
            gate_proj_bias_tensor (torch.Tensor, optional): Bias for gate projection. Shape: [1, I].
            up_proj_bias_tensor (torch.Tensor, optional): Bias for up projection. Shape: [1, I].
            down_proj_bias_tensor (torch.Tensor, optional): Bias for down projection. Shape: [1, H].
            normalization_bias_tensor (torch.Tensor, optional): Bias for normalization. Shape: [1, H].
                Used when normalization_type is LAYER_NORM. If None, no bias is applied.
            fused_add_tensor (torch.Tensor, optional): Residual tensor for fused addition. Shape: [B, S, H].
            store_fused_add_result (bool): If True, returns the fused add result alongside the
                MLP output. Default: False.
            activation_fn (ActFnType): Activation function type (SiLU, GELU, GELU_Tanh_Approx).
                Default: ActFnType.SiLU.
            normalization_type (NormType): Type of normalization (NO_NORM, RMS_NORM, LAYER_NORM,
                RMS_NORM_SKIP_GAMMA). Default: NormType.NO_NORM.
            quantization_type (QuantizationType): Quantization type. Default: QuantizationType.NONE.
                Supported values:
                - NONE: No quantization
                - ROW: FP8 per-row quantization
                - STATIC: FP8 tensor-wise quantization
                - STATIC_MX: FP8 tensor-wise with MX weight swizzle (CTE only)
                - MX: MXFP quantization (MXFP4/MXFP8, TKG only)
            gate_w_scale (torch.Tensor, optional): Dequantization scales for gate weights.
                Shape: [128, I] for ROW, [128, 1] for STATIC. Required for quantized modes.
            up_w_scale (torch.Tensor, optional): Dequantization scales for up weights.
                Shape: [128, I] for ROW, [128, 1] for STATIC. Required for quantized modes.
            down_w_scale (torch.Tensor, optional): Dequantization scales for down weights.
                Shape: [128, I] for ROW, [128, 1] for STATIC. Required for quantized modes.
            gate_up_in_scale (torch.Tensor, optional): FP8 input scales for gate/up projections.
                Shape: [128, 1]. Required for STATIC/STATIC_MX in CTE mode.
            down_in_scale (torch.Tensor, optional): FP8 input scales for down projection.
                Shape: [128, 1]. Required for STATIC/STATIC_MX in CTE mode.
            quant_clipping_bound (float): Clipping boundary for FP8 row quantization. Default: 0.0.
            output_dtype: Output data type. Default: None (ignored in torch ref).
            store_output_in_sbuf (bool): Store output in SBUF. Default: False.
                Hardware-only parameter, ignored in torch ref.
            eps (float): Epsilon for numerical stability in normalization. Default: 1e-6.
            skip_gate_proj (bool): If True, skips gate projection and applies activation
                directly to up projection output. Default: False.
            use_tkg_gate_up_proj_column_tiling (bool): Column tiling for gate/up projection.
                Default: True. Hardware-only parameter, ignored in torch ref.
            use_tkg_down_proj_column_tiling (bool): Column tiling for down projection.
                Default: True. Hardware-only parameter, ignored in torch ref.
            use_tkg_down_proj_optimized_layout (bool): If True, undoes the optimized down weight
                layout [I, lnc, 128, H//(128*lnc)] back to [I, H] using the lnc value from
                LncSubscriptable. Default: False.
            gate_clamp_upper_limit (float, optional): Upper clamp for gate projection output.
            gate_clamp_lower_limit (float, optional): Lower clamp for gate projection output.
            up_clamp_upper_limit (float, optional): Upper clamp for up projection output.
            up_clamp_lower_limit (float, optional): Lower clamp for up projection output.
            force_cte_mode (bool): Force CTE mode. Default: False.
                Hardware-only parameter, ignored in torch ref.
            sbm: Buffer manager. Default: None.
                Hardware-only parameter, ignored in torch ref.

        Returns:
            Dict[str, torch.Tensor]: Dictionary with the following keys:
                - "out": MLP output tensor. Shape: [B, S, H].
                - "add_out" (only when fused_add_tensor is provided and store_fused_add_result
                  is True): Result of the fused residual addition. Shape: [B, S, H].

        Notes:
            - All computation is performed in float32 precision
            - Hardware-specific parameters (store_output_in_sbuf, use_tkg_gate_up_proj_column_tiling,
              use_tkg_down_proj_column_tiling, force_cte_mode, sbm) are accepted for signature
              compatibility but do not affect the computation
            - use_tkg_down_proj_optimized_layout is NOT hardware-only — it triggers a weight
              layout undo using the lnc value from LncSubscriptable
        """
        ...


mlp_torch_ref = LncSubscriptable(_mlp_torch_ref_impl, _MlpTorchRefFn)
