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

"""Utility functions and constants for MoE Block TKG kernel."""

from dataclasses import dataclass
from typing import Optional

import nki.language as nl

from ..utils.common_types import MoEBlockIOLayout
from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import div_ceil, get_verified_program_sharding_info

# Constants
_pmax = 128  # sbuf max partition dim (nl.tile_size.pmax)
_q_width = 4  # MX quantization block width
_MX_SUPPORTED_DTYPES = (
    nl.float4_e2m1fn_x4,
    nl.float8_e4m3fn_x4,
)  # TODO: add support for MXFP8


def get_sbuf_tensor_shape(T: int, free_dim: int, is_sbuf: bool) -> tuple:
    """
    Get tensor shape for SBUF or HBM buffer.

    SBUF has max partition dim of _pmax (128). When T > _pmax, fold to 3D tensor.
    When num_t_tiles == 1, keep 2D shape.

    Args:
        T: Number of tokens (partition dimension)
        free_dim: Size of free dimension
        is_sbuf: Whether tensor is in SBUF

    Returns:
        Shape tuple: (T, free_dim) for HBM or T <= _pmax
                     (_pmax, num_t_tiles, free_dim) for SBUF when T > _pmax
    """
    if not is_sbuf or T <= _pmax:
        return (T, free_dim)

    num_t_tiles = div_ceil(T, _pmax)
    return (_pmax, num_t_tiles, free_dim)


@dataclass
class QuantizationConfig(nl.NKIObject):
    """Weight quantization configuration for MoE Block TKG kernel."""

    is_moe_weight_mx: bool  # Whether MoE weights use MX format
    is_static_quant: bool  # Whether using per-tensor static quantization
    is_row_quant: bool  # Whether using per-token row-wise quantization


@dataclass
class ExpertConfig(nl.NKIObject):
    """Expert execution configuration for MoE Block TKG kernel."""

    is_all_expert: bool  # Whether using all-expert mode
    has_shared_expert: bool  # Whether shared expert is enabled


@dataclass
class MoEBlockTKGDims(nl.NKIObject):
    """
    Dimension constants for MoE Block TKG kernel.

    Captures all dimension parameters parsed from input tensors.

    Args:
        B (int): Batch size.
        S (int): Sequence length.
        T (int): Total tokens (B * S).
        H (int): Hidden dimension size.
        H_free (int): Hidden dimension free tiles (H // 128).
        H_free_shard (int): Hidden dimension free tiles per shard.
        E (int): Number of experts.
        K (int): Top-K experts per token.
        n_prgs (int): Number of programs (LNC shards).
        prg_id (int): Current program ID.
        hidden_actual (int): Actual hidden dimension for RMSNorm.
    """

    B: int
    S: int
    T: int
    H: int
    H_free: int
    H_free_shard: int
    E: int
    K: int
    n_prgs: int
    prg_id: int
    hidden_actual: int


def parse_moe_block_config(
    inp: nl.ndarray,
    router_weights: nl.ndarray,
    expert_gate_up_weights: nl.ndarray,
    shared_expert_gate_w: Optional[nl.ndarray],
    top_k: int,
    hidden_actual: Optional[int],
    is_all_expert: bool,
    expert_gate_up_weights_scale: Optional[nl.ndarray] = None,
    expert_gate_up_input_scale: Optional[nl.ndarray] = None,
    inp_layout: MoEBlockIOLayout = MoEBlockIOLayout.B_S_H,
) -> tuple[MoEBlockTKGDims, QuantizationConfig, ExpertConfig]:
    """
    Parse input tensors and compute dimension constants.

    Args:
        inp (nl.ndarray): [B, S, H] or [H0, n_prgs, H1_shard, BxS] depending on inp_layout.
        router_weights (nl.ndarray): [H, E], Router weights tensor.
        expert_gate_up_weights (nl.ndarray): Expert gate/up projection weights.
        shared_expert_gate_w (nl.ndarray): Optional shared expert gate weights.
        top_k (int): Number of top-K experts.
        hidden_actual (int): Optional actual hidden dimension for RMSNorm.
        is_all_expert (bool): Whether using all-expert mode.
        expert_gate_up_weights_scale (nl.ndarray): Optional quantization scales for gate/up weights.
        expert_gate_up_input_scale (nl.ndarray): Optional FP8 dequant scale for gate/up input (STATIC_MX).
        inp_layout (MoEBlockIOLayout): Input tensor layout.

    Returns:
        tuple: (MoEBlockTKGDims, QuantizationConfig, ExpertConfig)
    """
    if inp_layout == MoEBlockIOLayout._128_Nprgs_Hfree_T:
        H = router_weights.shape[0]
        B = 1
        S = inp.shape[3]
    else:
        B, S, H = inp.shape
    hidden_actual = H if hidden_actual == None else hidden_actual
    H_free = H // _pmax
    T = B * S
    _, E = router_weights.shape
    is_moe_weight_mx = expert_gate_up_weights.dtype in _MX_SUPPORTED_DTYPES
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_block_tkg_kernel", (0, 1), 2)

    dims = MoEBlockTKGDims(
        B=B,
        S=S,
        T=T,
        H=H,
        H_free=H_free,
        H_free_shard=H_free // n_prgs if prg_id < n_prgs - 1 else H_free - H_free // n_prgs * (n_prgs - 1),
        E=E,
        K=top_k,
        n_prgs=n_prgs,
        prg_id=prg_id,
        hidden_actual=hidden_actual,
    )

    is_static_quant = expert_gate_up_input_scale != None and is_moe_weight_mx
    is_row_quant = (
        is_moe_weight_mx
        and expert_gate_up_weights_scale != None
        and expert_gate_up_weights_scale.dtype != nl.uint8
        and expert_gate_up_weights_scale.shape[-1] > 1
        and expert_gate_up_input_scale == None
    )
    quant_config = QuantizationConfig(
        is_moe_weight_mx=is_moe_weight_mx,
        is_static_quant=is_static_quant,
        is_row_quant=is_row_quant,
    )

    expert_config = ExpertConfig(
        is_all_expert=is_all_expert,
        has_shared_expert=shared_expert_gate_w != None,
    )

    return dims, quant_config, expert_config


def validate_moe_block_inputs(
    dims: MoEBlockTKGDims,
    quant_config: QuantizationConfig,
    expert_config: ExpertConfig,
    shared_expert_gate_w: Optional[nl.ndarray],
    shared_expert_up_w: Optional[nl.ndarray],
    shared_expert_down_w: Optional[nl.ndarray],
    hidden_act_scale_factor: Optional[float],
    hidden_act_bias: Optional[nl.ndarray],
    router_mm_dtype,
    rank_id: Optional[nl.ndarray],
    residual: Optional[nl.ndarray],
):
    """
    Validate input parameters for MoE Block TKG kernel.

    Args:
        dims (MoEBlockTKGDims): Parsed dimension constants.
        quant_config (QuantizationConfig): Quantization configuration.
        expert_config (ExpertConfig): Expert execution configuration.
        shared_expert_gate_w (nl.ndarray): Optional shared expert gate weights.
        shared_expert_up_w (nl.ndarray): Optional shared expert up weights.
        shared_expert_down_w (nl.ndarray): Optional shared expert down weights.
        hidden_act_scale_factor (float): Optional activation scale factor.
        hidden_act_bias (nl.ndarray): Optional activation bias.
        router_mm_dtype: Router matmul dtype.
        rank_id (nl.ndarray): Optional rank ID for all-expert mode.
        residual (nl.ndarray): Optional residual tensor.

    Raises:
        AssertionError: If any validation check fails.
    """
    # Basic parameter checks
    kernel_assert(dims.H % _pmax == 0, f"H={dims.H} must be divisible by {_pmax}")

    # Token size constraints
    if not expert_config.is_all_expert:
        kernel_assert(dims.T <= 128, f"selective_load mode currently supports T <= 128, got {dims.T}")

    kernel_assert(
        not expert_config.has_shared_expert, "shared_expert has not been supported in moe_block_tkg kernel yet"
    )

    # Shared expert validation (for future support)
    if expert_config.has_shared_expert:
        kernel_assert(
            shared_expert_up_w != None,
            "shared expert up weight must be a valid tensor when shared expert is enabled",
        )
        kernel_assert(
            shared_expert_down_w != None,
            "shared expert down weight must be a valid tensor when shared expert is enabled",
        )
        kernel_assert(
            shared_expert_gate_w.shape == shared_expert_up_w.shape,
            "shared gate & up weight shapes must match",
        )
        kernel_assert(
            shared_expert_gate_w.shape[0] == shared_expert_down_w.shape[1]
            and shared_expert_gate_w.shape[1] == shared_expert_down_w.shape[0],
            "shared gate/up weight and down weight shapes must match",
        )

    # All-expert mode specific validation
    if expert_config.is_all_expert:
        kernel_assert(rank_id != None, "rank_id is required for all_expert mode")
        if residual != None:
            kernel_assert(
                quant_config.is_moe_weight_mx, "fused residual add is only supported for MXFP in all_expert mode"
            )

    # Current implementation limitations
    # LNC=1 is only supported for MX weight paths (all-expert MX and selective-load MX).
    # The BF16/non-MX all-expert path has LNC-2-specific sharding that doesn't support n_prgs=1.
    kernel_assert(
        dims.n_prgs == 2 or quant_config.is_moe_weight_mx,
        f"moe_block_tkg with LNC=1 only supports MX weight dtypes; got n_prgs={dims.n_prgs}, is_moe_weight_mx={quant_config.is_moe_weight_mx}",
    )
    # Unbalanced H-sharding (H not divisible by 128*n_prgs) is only supported for BF16 all-expert path
    if not (expert_config.is_all_expert and not quant_config.is_moe_weight_mx):
        kernel_assert(dims.H % (_pmax * dims.n_prgs) == 0, f"H={dims.H} must be divisible by {_pmax * dims.n_prgs}")
    kernel_assert(
        hidden_act_scale_factor == None,
        "hidden_act_scale_factor is currently a placeholder in moe_block_tkg kernel",
    )
    kernel_assert(hidden_act_bias == None, "hidden_act_bias is currently a placeholder in moe_block_tkg")

    # Router dtype validation
    kernel_assert(
        router_mm_dtype in (nl.bfloat16, nl.float16, nl.float32),
        f"moe_block_tkg expects router_mm_dtype to be one of (nl.bfloat16, nl.float16, nl.float32), got {router_mm_dtype}",
    )


# Byte sizes for NKI dtypes used in SBUF estimation
_DTYPE_SIZE = {
    nl.bfloat16: 2,
    nl.float16: 2,
    nl.float32: 4,
    nl.float8_e4m3: 1,
    nl.float8_e4m3fn: 1,
    nl.float4_e2m1fn_x4: 2,
    nl.float8_e4m3fn_x4: 4,
    nl.float8_e5m2_x4: 4,
    nl.uint8: 1,
}

# SBUF usable bytes per partition by target (from trn_hw_info.md)
_SBUF_USABLE_PER_PARTITION = {
    "trn1": 180224,
    "trn2": 212984,
    "trn3": 245752,
}


def _dtype_size(dtype) -> int:
    """Return size in bytes for a given NKI dtype."""
    return _DTYPE_SIZE.get(dtype, 2)


def _get_tile_size(
    T: int,
    H: int,
    E_L: int,
    I: int,
    is_mx: bool,
    input_dtype=nl.bfloat16,
    weight_dtype=nl.bfloat16,
    is_all_expert_static_mx: bool = False,
) -> int:
    """
    Determine the tile size for T-dimension tiling based on estimated SBUF capacity.

    Estimates peak SBUF usage per partition across two execution phases:
    - Phase 1 (RMSNorm + Router): rmsnorm output + router intermediates
    - Phase 2 (Expert MLP): weights + quantized input + affinities + output accumulator

    The estimate is intentionally conservative. When tile_T == T, no tiling occurs.

    Args:
        T: Total number of tokens.
        H: Hidden dimension size.
        E_L: Number of local experts.
        I: Intermediate dimension size.
        is_mx: Whether using MX quantized weights.
        input_dtype: Input/activation dtype (e.g., nl.bfloat16).
        weight_dtype: Expert weight dtype (e.g., nl.float4_e2m1fn_x4).
        is_all_expert_static_mx: If True, accounts for bf16 input residing in SBUF during expert MLP (where per-expert quantization happens)

    Returns:
        Tile size (multiple of _pmax) that fits within SBUF capacity.
    """
    input_bytes = _dtype_size(input_dtype)
    # Use trn2 capacity as the conservative default (smallest modern target)
    sbuf_capacity = _SBUF_USABLE_PER_PARTITION["trn2"]

    H_free = H // _pmax

    # Expert weight footprint per partition (constant, doesn't scale with T)
    if is_mx:
        n_I512 = div_ceil(I, 512)
        n_I512_local = div_ceil(n_I512, 2)  # LNC-2 sharding
        weight_bytes = _dtype_size(weight_dtype)
        weight_per_part = n_I512_local * H * weight_bytes + n_I512_local * H  # weight + scale
    else:
        weight_per_part = I * H * _dtype_size(weight_dtype) // _pmax

    tile_T = T
    while tile_T > _pmax:
        n_T128 = div_ceil(tile_T, _pmax)

        # Phase 1: RMSNorm output lives through router computation
        # rmsnorm_out: [_pmax, tile_T, H_free] in input_dtype
        phase1 = tile_T * H_free * input_bytes

        # Phase 2: Expert MLP
        if is_mx and not is_all_expert_static_mx:
            num_H512_tiles = H // (_pmax * _q_width)
            # Quantized input (fp8x4 = 1B) + scale (uint8 = 1B) per element
            input_per_part = num_H512_tiles * tile_T * 2
        else:
            input_per_part = tile_T * H_free * input_bytes

        affinities_per_part = n_T128 * E_L * 4  # float32 affinities
        output_per_part = n_T128 * H * input_bytes  # output accumulator

        phase2 = weight_per_part + input_per_part + affinities_per_part + output_per_part

        if max(phase1, phase2) <= sbuf_capacity:
            break
        tile_T = tile_T // 2

    # Ensure tile_T is at least _pmax and a multiple of _pmax
    tile_T = max(tile_T, _pmax)
    tile_T = (tile_T // _pmax) * _pmax
    return tile_T
