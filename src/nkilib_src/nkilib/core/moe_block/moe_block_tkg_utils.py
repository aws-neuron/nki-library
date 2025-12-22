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

from ..utils.kernel_assert import kernel_assert
from ..utils.kernel_helpers import get_verified_program_sharding_info

# Constants
_pmax = 128  # sbuf max partition dim (nl.tile_size.pmax)
_MX_SUPPORTED_DTYPES = (nl.float4_e2m1fn_x4,)  # TODO: add support for MXFP8


@dataclass
class MoEBlockTKGDims(nl.NKIObject):
    """
    Dimension constants for MoE Block TKG kernel.

    Captures all dimension parameters and configuration flags parsed from input tensors.

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
        is_moe_weight_mx (bool): Whether MoE weights use MX format.
        has_shared_expert (bool): Whether shared expert is enabled.
        is_all_expert (bool): Whether using all-expert mode.
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
    is_moe_weight_mx: bool
    has_shared_expert: bool
    is_all_expert: bool


def parse_moe_block_dims(
    inp: nl.ndarray,
    router_weights: nl.ndarray,
    expert_gate_up_weights: nl.ndarray,
    shared_expert_gate_w: Optional[nl.ndarray],
    top_k: int,
    hidden_actual: Optional[int],
    is_all_expert: bool,
) -> MoEBlockTKGDims:
    """
    Parse input tensors and compute dimension constants.

    Args:
        inp (nl.ndarray): [B, S, H], Input tensor.
        router_weights (nl.ndarray): [H, E], Router weights tensor.
        expert_gate_up_weights (nl.ndarray): Expert gate/up projection weights.
        shared_expert_gate_w (nl.ndarray): Optional shared expert gate weights.
        top_k (int): Number of top-K experts.
        hidden_actual (int): Optional actual hidden dimension for RMSNorm.
        is_all_expert (bool): Whether using all-expert mode.

    Returns:
        MoEBlockTKGDims: Parsed dimension constants.
    """
    B, S, H = inp.shape
    hidden_actual = H if hidden_actual == None else hidden_actual
    H_free = H // _pmax
    T = B * S
    _, E = router_weights.shape
    is_moe_weight_mx = expert_gate_up_weights.dtype in _MX_SUPPORTED_DTYPES
    _, n_prgs, prg_id = get_verified_program_sharding_info("moe_block_tkg_kernel", (0, 1), 2)

    return MoEBlockTKGDims(
        B=B,
        S=S,
        T=T,
        H=H,
        H_free=H_free,
        H_free_shard=H_free // n_prgs,
        E=E,
        K=top_k,
        n_prgs=n_prgs,
        prg_id=prg_id,
        hidden_actual=hidden_actual,
        is_moe_weight_mx=is_moe_weight_mx,
        has_shared_expert=shared_expert_gate_w != None,
        is_all_expert=is_all_expert,
    )


def validate_moe_block_inputs(
    dims: MoEBlockTKGDims,
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

    # Token size constraints differ between selective and all-expert modes
    if dims.is_all_expert:
        if dims.is_moe_weight_mx:
            kernel_assert(dims.T % 4 == 0, f"all_expert mode with MXFP requires T divisible by 4, got {dims.T}")
        else:
            kernel_assert(dims.T <= 128, f"all_expert mode currently supports T <= 128, got {dims.T}")
    else:
        kernel_assert(dims.T <= 128, f"selective_load mode currently supports T <= 128, got {dims.T}")

    kernel_assert(not dims.has_shared_expert, "shared_expert has not been supported in moe_block_tkg kernel yet")

    # Shared expert validation (for future support)
    if dims.has_shared_expert:
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
    if dims.is_all_expert:
        kernel_assert(rank_id != None, "rank_id is required for all_expert mode")
        if residual != None:
            kernel_assert(dims.is_moe_weight_mx, "fused residual add is only supported for MXFP in all_expert mode")

    # Current implementation limitations
    kernel_assert(dims.n_prgs == 2, f"moe_block_tkg only supports LNC-2; but got a spmd grid size of {dims.n_prgs}")
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
