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

"""Fused RMSNorm + Router TopK for small-T MoE token generation with A2A-v."""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.router_topk.router_topk import XSBLayout_tp102__0
from ...core.router_topk.router_topk import router_topk as _router_topk
from ...core.subkernels.rmsnorm_tkg import _process_rmsnorm_tile
from ...core.utils.allocator import SbufManager
from ...core.utils.common_types import RouterActFnType
from ...core.utils.kernel_helpers import get_verified_program_sharding_info
from ...core.utils.tiled_range import TiledRange

_pmax = 128
_BxS_FULL_TILE_SIZE = 128


def _rmsnorm_small_t(hidden_states, gamma, eps, output_sb, output_hbm, T, H, H0, H_free, n_prgs, prg_id):
    """RMSNorm sub-kernel for small T. Both NCs compute full T.

    Loads input, calls _process_rmsnorm_tile, stores to HBM (sharded on T across NCs).
    Output SBUF tensor is kept for downstream router fusion.
    """
    # Load input: HBM [T, H] → SBUF [H0, T, H_free]
    input_hbm_view = (
        hidden_states.flatten_dims(start_dim=0, end_dim=1)  # [T, H]
        .reshape_dim(dim=1, shape=[H0, H_free])  # [T, H0, H_free]
        .permute(dims=[1, 0, 2])  # [H0, T, H_free]
    )
    # Reuse output_sb as input buffer (loaded in-place, then overwritten by normalization)
    nisa.dma_copy(dst=output_sb, src=input_hbm_view)

    # Load gamma: [1, H] → [H0, H_free]
    gamma_sb = nl.ndarray((H0, H_free), dtype=gamma.dtype, buffer=nl.sbuf)
    gamma_hbm_view = gamma.reshape_dim(dim=1, shape=[H0, H_free]).permute(dims=[1, 0, 2])
    nisa.dma_copy(dst=gamma_sb, src=gamma_hbm_view)

    # Prepare eps and reduction constant
    eps_sb = nl.ndarray((H0, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, value=eps)
    matmul_reduction_const = nl.ndarray((H0, H0), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(matmul_reduction_const, value=1.0)

    # Process all T tokens
    input_sb_view = output_sb
    output_sb_view = output_sb

    sbm = SbufManager(0, nl.tile_size.total_available_sbuf_size, use_auto_alloc=True)
    sbm.open_scope()

    for bxs_tile in TiledRange(T, _BxS_FULL_TILE_SIZE):
        input_tile = input_sb_view.slice(dim=1, start=bxs_tile.start_offset, end=bxs_tile.start_offset + bxs_tile.size)
        gamma_tile = gamma_sb.expand_dim(dim=1).broadcast(dim=1, size=bxs_tile.size)
        output_tile = output_sb_view.slice(
            dim=1, start=bxs_tile.start_offset, end=bxs_tile.start_offset + bxs_tile.size
        )
        _process_rmsnorm_tile(
            input_sb_view=input_tile,
            gamma_sb_view=gamma_tile,
            output_sb_view=output_tile,
            eps_view=eps_sb,
            matmul_reduction_const_view=matmul_reduction_const,
            bxs_tile=bxs_tile,
            hidden_actual=H,
            shard_on_h=False,
            sbm=sbm,
        )

    sbm.close_scope()

    # Store to HBM [T, H]: each NC writes its shard of T
    if T > 1 and n_prgs > 1:
        T_shard = T // n_prgs
        T_offset = prg_id * T_shard
    else:
        T_shard = T
        T_offset = 0

    output_hbm_shard_view = (
        output_hbm.slice(dim=0, start=T_offset, end=T_offset + T_shard)  # [T_shard, H]
        .reshape_dim(dim=1, shape=[H0, H_free])  # [T_shard, H0, H_free]
        .permute(dims=[1, 0, 2])  # [H0, T_shard, H_free]
    )
    nisa.dma_copy(
        dst=output_hbm_shard_view,
        src=output_sb[:, nl.ds(T_offset, T_shard), :],
    )


@nki.jit
def rmsnorm_router_topk_a2av(
    hidden_states: nl.NkiTensor,
    gamma: nl.NkiTensor,
    router_weights: nl.NkiTensor,
    router_bias: Optional[nl.NkiTensor] = None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
):
    """Fused RMSNorm + Router TopK for MoE token generation (small T).

    Both NCs duplicate RMSNorm + Router (no token sharding on compute).
    HBM norm_output store is sharded across NCs for bandwidth.

    Dimensions:
        B: Batch size (typically 1 for TKG)
        S: Sequence length (tokens per batch, S <= 128)
        H: Hidden dimension
        E: Number of experts
        K: top_k experts per token

    Args:
        hidden_states (nl.NkiTensor): [B, S, H]@HBM, bf16/fp16 input.
        gamma (nl.NkiTensor): [1, H]@HBM, bf16/fp16 RMSNorm scale weights.
        router_weights (nl.NkiTensor): [H, E]@HBM, bf16/fp16 router projection.
        router_bias (Optional[nl.NkiTensor]): [1, E]@HBM, optional router bias.
        eps (float): RMSNorm epsilon for numerical stability.
        top_k (int): Number of top experts to select per token.
        router_act_fn (RouterActFnType): Activation for router (SOFTMAX or SIGMOID).

    Returns:
        norm_output (nl.NkiTensor): [T, H]@HBM, normalized hidden states.
        expert_index (nl.NkiTensor): [T, K]@HBM, int32 top-K expert indices.
        expert_affinities (nl.NkiTensor): [T, E]@HBM, bf16 masked affinities.

    Notes:
        - T = B * S must be <= 128 (single tile processing)
        - H must be divisible by 128 (pmax)
        - Both NCs compute identical results; HBM store is sharded for bandwidth
        - Router uses ACT2 pipeline (router_pre_norm=False): top-K on raw logits,
          then activation applied only to selected experts

    Pseudocode:
        norm = rms_norm(hidden_states, gamma, eps)
        logits = norm @ router_weights + router_bias
        expert_index = argsort(-logits)[:, :top_k]
        expert_affinities = act_fn(logits[expert_index])  # masked [T, E]
    """
    B, S, H = hidden_states.shape
    T = B * S
    _, E = router_weights.shape
    H0 = _pmax
    H_free = H // H0

    _, n_prgs, prg_id = get_verified_program_sharding_info()

    # Allocate outputs
    norm_output = nl.ndarray((T, H), dtype=hidden_states.dtype, buffer=nl.shared_hbm)
    expert_index = nl.ndarray((T, top_k), dtype=nl.int32, buffer=nl.shared_hbm)
    expert_affinities = nl.ndarray((T, E), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    # SBUF buffer for rmsnorm output (reused as router input)
    rmsnorm_out_sb = nl.ndarray((H0, T, H_free), dtype=hidden_states.dtype, buffer=nl.sbuf)

    # ── RMSNorm ──
    _rmsnorm_small_t(hidden_states, gamma, eps, rmsnorm_out_sb, norm_output, T, H, H0, H_free, n_prgs, prg_id)

    # ── Router TopK ──
    _router_topk(
        x=rmsnorm_out_sb,
        w=router_weights,
        w_bias=router_bias,
        router_logits=None,
        expert_affinities=expert_affinities,
        expert_index=expert_index,
        act_fn=router_act_fn,
        k=top_k,
        x_hbm_layout=0,
        x_sb_layout=XSBLayout_tp102__0,
        router_pre_norm=False,
        norm_topk_prob=False,
        use_column_tiling=True,
        use_indirect_dma_scatter=False,
        use_PE_broadcast_w_bias=False,
        shard_on_tokens=False,
        skip_store_expert_index=False,
        skip_store_router_logits=True,
    )

    return norm_output, expert_index, expert_affinities
