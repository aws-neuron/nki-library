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

"""Utility classes and functions for the QKV TKG MXFP Kernel."""

from dataclasses import dataclass
from typing import Any, Optional

import nki.language as nl

from ..utils.allocator import SbufManager
from ..utils.common_types import NormType, QKVOutputLayout, QuantizationType
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info

P_MAX = 128


@dataclass
class QKV_TKG_MXFP_UserInput(nl.NKIObject):
    """
    Container for unmodified user inputs to QKV TKG MXFP kernel.

    This dataclass captures all user-provided parameters before validation
    and configuration building.
    """

    hidden: nl.ndarray
    weights_qtz_hbm: nl.ndarray
    norm_w: Optional[nl.ndarray] = None
    fused_add: bool = False
    mlp_prev: Optional[nl.ndarray] = None
    attn_prev: Optional[nl.ndarray] = None
    d_head: Optional[int] = None
    num_kv_heads: Optional[int] = None
    num_q_heads: Optional[int] = None
    output_layout: QKVOutputLayout = QKVOutputLayout.BSD
    eps: float = 1e-6
    norm_type: NormType = NormType.NO_NORM
    quantization_type: QuantizationType = QuantizationType.MX
    is_h_dim_4h_transposed: bool = False
    weight_scales_hbm: Optional[nl.ndarray] = None
    output_in_sbuf: bool = False
    qkv_bias: Optional[nl.ndarray] = None
    norm_bias: Optional[nl.ndarray] = None
    hidden_actual: Optional[int] = None
    sbm: Optional[SbufManager] = None


def _validate_user_inputs(args: QKV_TKG_MXFP_UserInput) -> None:
    """
    Validate all user inputs to the QKV TKG MXFP kernel.

    Args:
        args (QKV_TKG_MXFP_UserInput): User input container with all kernel parameters.

    Raises:
        AssertionError: If any validation check fails with descriptive message.
    """
    B, S, H = args.hidden.shape
    H_packed, I = args.weights_qtz_hbm.shape

    # Dimensions
    kernel_assert(H % 512 == 0, f"[QKV TKG MXFP] H must be divisible by 512 for MXFP, got H={H}.")
    kernel_assert(
        H_packed == H // 4,
        f"[QKV TKG MXFP] weights_qtz_hbm.shape[0] must equal H//4, got {H_packed}, expected {H // 4}.",
    )
    kernel_assert(
        args.weight_scales_hbm.shape == (H // 32, I),
        f"[QKV TKG MXFP] weight_scales_hbm.shape must be ({H // 32}, {I}), got {args.weight_scales_hbm.shape}.",
    )
    kernel_assert(B * S <= P_MAX, f"[QKV TKG MXFP] BxS must be <= {P_MAX} for TKG, got BxS={B * S}.")
    kernel_assert(
        (B * S) % 4 == 0, f"[QKV TKG MXFP] BxS must be divisible by 4 for MXFP quantization, got BxS={B * S}."
    )

    # Dtypes
    kernel_assert(
        args.weights_qtz_hbm.dtype == nl.float8_e4m3fn_x4,
        f"[QKV TKG MXFP] weights_qtz_hbm.dtype must be nl.float8_e4m3fn_x4, got {args.weights_qtz_hbm.dtype}.",
    )
    kernel_assert(
        args.weight_scales_hbm.dtype == nl.uint8,
        f"[QKV TKG MXFP] weight_scales_hbm.dtype must be nl.uint8, got {args.weight_scales_hbm.dtype}.",
    )
    # H pre-shuffle conditions.
    kernel_assert(args.is_h_dim_4h_transposed == True, f"[QKV TKG MXFP] is_h_dim_4h_transposed must be True for MXFP.")
    # QKV_MXFP does not support these at the moment:
    # fused_add is not supported
    kernel_assert(
        args.fused_add == False, f"[QKV TKG MXFP] fused_add is not supported, got fused_add={args.fused_add}."
    )
    # output_layout only BSD
    kernel_assert(
        args.output_layout == QKVOutputLayout.BSD,
        f"[QKV TKG MXFP] Only BSD output_layout is supported, got output_layout={args.output_layout}.",
    )
    # No support for LayerNorm at the moment.
    kernel_assert(
        args.norm_type == NormType.NO_NORM or args.norm_type == NormType.RMS_NORM,
        f"[QKV TKG MXFP] Only NO_NORM / RMS_NORM is supported, got norm_type={args.norm_type}.",
    )
    # Ensure quantization type is always MX
    kernel_assert(
        args.quantization_type == QuantizationType.MX,
        f"[QKV TKG MXFP] quantization_type must be MX, got quantization_type={args.quantization_type}.",
    )
    # norm_bias is not supported either
    kernel_assert(args.norm_bias is None, f"[QKV TKG MXFP] norm_bias is not supported.")
    # Input must be on HBM
    kernel_assert(
        args.hidden.buffer == nl.hbm or args.hidden.buffer == nl.shared_hbm,
        f"[QKV TKG MXFP] hidden input must be on HBM, SBUF input not supported.",
    )
    # Output must be on HBM, output_in_sbuf not yet supported
    kernel_assert(
        args.output_in_sbuf == False, f"[QKV TKG MXFP] output_in_sbuf is not supported, output must be on HBM."
    )
    # When output is on SBUF size of I is restricted.
    if args.output_in_sbuf == True:
        kernel_assert(I <= 4096, f"[QKV TKG MXFP] If output_in_sbuf=True, then I <= 4096, but got {I}")


@dataclass
class QKV_TKG_MXFP_Config(nl.NKIObject):
    """
    Kernel configuration for QKV TKG MXFP.

    Contains computed configuration values derived from user inputs,
    including dimensions, sharding info, and feature flags.
    """

    # User Requested
    output_layout: QKVOutputLayout
    add_bias: bool
    fused_residual_add: bool
    fused_norm_type: NormType
    add_layer_norm_bias: bool
    output_in_sbuf: bool

    # Dimensions
    B: int
    S: int
    H: int
    I: int
    BxS: int
    H0: int
    H1: int
    H_packed: int
    H1_packed: int
    hidden_orig_dtype: Any

    # Sharding
    num_shards: int
    shard_id: int
    H1_packed_shard: int
    H_packed_shard: int

    # Flags
    is_h_dim_4h_transposed: bool
    force_lnc1: bool  # True if H too small to shard


def _build_config(args: QKV_TKG_MXFP_UserInput) -> QKV_TKG_MXFP_Config:
    """
    Build kernel configuration from user inputs.

    Args:
        args (QKV_TKG_MXFP_UserInput): Validated user input container.

    Returns:
        QKV_TKG_MXFP_Config: Computed kernel configuration.
    """
    B, S, H = args.hidden.shape
    H_packed, I = args.weights_qtz_hbm.shape
    BxS = B * S
    H0 = P_MAX
    H1 = H // H0
    H1_packed = H // 512

    _, num_shards, shard_id = get_verified_program_sharding_info("qkv_tkg_mxfp", (0, 1))

    # Force LNC1 if H too small to shard
    force_lnc1 = H1_packed <= 1
    if force_lnc1:
        num_shards = 1
        shard_id = 0

    H1_packed_shard = H1_packed // num_shards
    H_packed_shard = H0 * H1_packed_shard

    add_bias = args.qkv_bias is not None
    add_layer_norm_bias = args.norm_bias is not None

    return QKV_TKG_MXFP_Config(
        # User Requested
        output_layout=args.output_layout,
        add_bias=add_bias,
        fused_residual_add=args.fused_add,
        fused_norm_type=args.norm_type,
        add_layer_norm_bias=add_layer_norm_bias,
        output_in_sbuf=args.output_in_sbuf,
        # Dimensions
        B=B,
        S=S,
        H=H,
        I=I,
        BxS=BxS,
        H0=H0,
        H1=H1,
        H_packed=H_packed,
        H1_packed=H1_packed,
        hidden_orig_dtype=args.hidden.dtype,
        num_shards=num_shards,
        shard_id=shard_id,
        H1_packed_shard=H1_packed_shard,
        H_packed_shard=H_packed_shard,
        is_h_dim_4h_transposed=args.is_h_dim_4h_transposed,
        force_lnc1=force_lnc1,
    )
