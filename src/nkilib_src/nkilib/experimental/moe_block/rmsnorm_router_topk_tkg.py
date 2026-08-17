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

"""Fused RMSNorm + Router TopK kernel for MoE token generation."""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.moe_block.moe_block_tkg_utils import _pmax
from ...core.router_topk.router_topk import XSBLayout_tp102__0, XSBLayout_tp201__2
from ...core.router_topk.router_topk import router_topk as _router_topk
from ...core.subkernels.rmsnorm_mx_quantize_tkg import rmsnorm_mx_quantize_tkg as _rmsnorm_mx_quantize_tkg
from ...core.subkernels.rmsnorm_tkg import _rmsnorm_tkg_dloc
from ...core.utils.common_types import QuantizationType, RouterActFnType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_verified_program_sharding_info


@nki.jit
def rmsnorm_router_topk_tkg(
    hidden_states: nl.ndarray,
    gamma: nl.ndarray,
    router_weights: nl.ndarray,
    router_bias: Optional[nl.ndarray] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    hidden_actual: Optional[int] = None,
    quantization_type: QuantizationType = QuantizationType.NONE,
    router_mm_dtype=nl.bfloat16,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
    store_eager_affi_only: bool = False,
):
    """Fused RMSNorm (+ optional MX quantize) + Router TopK.

    Args:
        hidden_states (nl.ndarray): [B, S, H], Input tensor on HBM.
        gamma (nl.ndarray): [1, H], RMSNorm weights on HBM.
        router_weights (nl.ndarray): [H, E], Router weights on HBM.
        router_bias (Optional[nl.ndarray]): [1, E], Optional router bias on HBM.
        eps (float): Epsilon for RMSNorm. Default 1e-6.
        top_k (int): Number of top experts per token. Default 1.
        hidden_actual (Optional[int]): Actual hidden dim for padded inputs.
        quantization_type (QuantizationType): NONE or MX. Default NONE.
        router_mm_dtype: Dtype for router matmul. Default nl.bfloat16.
        router_act_fn (RouterActFnType): SOFTMAX or SIGMOID. Default SIGMOID.
        store_eager_affi_only (bool): If True, skip building/storing the sparse [T, E] expert
                        affinities (no [T, E] HBM allocation, no indirect-DMA scatter) and instead
                        return the dense [T, K] eager affinities. Requires T <= 128. Downstream
                        consumers must read affinities in the dense [T, K] form (co-indexed with
                        expert_index) rather than the sparse [T, E] form.

    Returns:
        norm_output: [T, H] (NONE) or [T, H + H/4] FP8 packed quant‖scales (MX).
        expert_index: [T, K] int32 top-K indices.
        expert_affinities: when store_eager_affi_only=False, the sparse [T, E] bfloat16 masked top-K
            affinities (zero elsewhere); when store_eager_affi_only=True, the dense [T, K] bf16 eager
            affinities co-indexed with expert_index.

    Notes:
        - Supports LNC=1 (single core, no token sharding) and LNC=2 (token-sharded).
        - NONE: H must be divisible by 128; T must be a multiple of 256 (DLoC tiling).
        - MX: H must be divisible by 512 (MX block size).
    """
    B, S, H = hidden_states.shape
    T = B * S
    _, E = router_weights.shape
    H_free = H // _pmax

    # Logical-NC config (number of SPMD cores): drives token-sharding decisions.
    # LNC=1 runs everything on a single core (no sendrecv / core_barrier).
    _, n_prgs, _ = get_verified_program_sharding_info("rmsnorm_router_topk_tkg", (0, 1))

    expert_index = nl.ndarray((T, top_k), dtype=nl.int32, buffer=nl.shared_hbm)
    if store_eager_affi_only:
        expert_affinities = None
        expert_affinities_eager = nl.ndarray((T, top_k), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    else:
        expert_affinities = nl.ndarray((T, E), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        expert_affinities_eager = None

    rmsnorm_out_sb = nl.ndarray((_pmax, T, H_free), dtype=hidden_states.dtype, buffer=nl.sbuf)

    if quantization_type == QuantizationType.MX:
        # MX path produces SBUF in XSBLayout_tp201__2 layout for the router.
        norm_output = nl.ndarray((T, H + H // 4), dtype=nl.float8_e4m3fn, buffer=nl.shared_hbm)
        _rmsnorm_mx_quantize_tkg(
            input=hidden_states,
            gamma=gamma,
            output=rmsnorm_out_sb,
            output_quant=norm_output,
            output_scale=None,
            eps=eps,
            hidden_actual=hidden_actual,
            hidden_dim_tp=True,
        )
        x_sb_layout = XSBLayout_tp201__2
    elif quantization_type == QuantizationType.NONE:
        # DLoC RMSNorm produces SBUF in XSBLayout_tp102__0 layout.
        norm_output = nl.ndarray((T, H), dtype=router_mm_dtype, buffer=nl.shared_hbm)
        _rmsnorm_tkg_dloc(
            input_hbm=hidden_states,
            gamma=gamma,
            output_hbm=norm_output,
            output_sb=rmsnorm_out_sb,
            eps=eps,
            hidden_actual=hidden_actual,
            sync_output=True,
        )
        x_sb_layout = XSBLayout_tp102__0
    else:
        kernel_assert(
            False,
            f"rmsnorm_router_topk_tkg only supports QuantizationType.NONE or MX, got {quantization_type}",
        )

    router_in = rmsnorm_out_sb
    if rmsnorm_out_sb.dtype != router_mm_dtype:
        router_in = nl.ndarray((_pmax, T, H_free), dtype=router_mm_dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=router_in, src=rmsnorm_out_sb)

    # ACT2 path (router_pre_norm=False) returns masked top-K affinities (zero elsewhere).
    # use_indirect_dma_scatter=True is required for SIGMOID in ACT2.
    _router_topk(
        x=router_in,
        w=router_weights,
        w_bias=router_bias,
        router_logits=None,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=router_act_fn,
        k=top_k,
        x_hbm_layout=0,
        x_sb_layout=x_sb_layout,
        router_pre_norm=False,
        norm_topk_prob=False,
        use_column_tiling=True,
        use_indirect_dma_scatter=True,
        use_PE_broadcast_w_bias=True,
        # Token-sharding needs >1 core; router_topk asserts n_prgs>1 when this is
        # set. At LNC=1 the single core processes all T tokens (no sharding).
        shard_on_tokens=T > 1 and n_prgs > 1,
        skip_store_expert_index=False,
        skip_store_router_logits=True,
        return_eager_affi=store_eager_affi_only,
        skip_store_expert_affinities=store_eager_affi_only,
        expert_affinities_eager_out=expert_affinities_eager if store_eager_affi_only else None,
    )

    if store_eager_affi_only:
        # _router_topk wrote the dense [T, k] eager affinities directly into expert_affinities_eager.
        return norm_output, expert_index, expert_affinities_eager

    return norm_output, expert_index, expert_affinities
