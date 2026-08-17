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

"""Torch reference for ``attention_block_tp_kernel``.

Same signature as the kernel. Mirrors its collective structure via
``torch.distributed`` (SimDistAdapter in sim / real backend on hardware) using
``get_pg`` to route to this rank's TP subgroup:

    AllGather[TP] token shards -> AttentionBlockTkgTorchRef(this rank's head shard)
                               -> ReduceScatter[TP] the O-proj partials

Because the ref is fed this rank's row-parallel ``W_out`` slice, its O-proj output
is a per-rank **partial** (this rank's head contribution) exactly like the kernel;
summing the partials over the TP subgroup (the ReduceScatter) reconstructs the
full multi-head attention output. Kernel and ref take identical per-rank inputs
and run the identical collective, so they agree by construction.

The in-place KV-cache write is a rank-local side effect on this rank's replicated
KV head; it is not part of the reduced output and needs no collective.
"""

import numpy as np
import torch
import torch.distributed as dist

from ...core.utils.common_types import QuantizationType
from ..collectives.distributed_adapter import get_pg
from ..transformer.attention_block_tkg_torch import AttentionBlockTkgTorchRef


def attention_block_tp_torch_ref(
    X_shard: np.ndarray,
    W_qkv: np.ndarray,
    bias_qkv: np.ndarray,
    W_out: np.ndarray,
    bias_out: np.ndarray,
    cos: np.ndarray,
    sin: np.ndarray,
    sink: np.ndarray,
    K_cache: np.ndarray,
    V_cache: np.ndarray,
    active_blocks_table: np.ndarray,
    attention_mask: np.ndarray,
    kv_cache_update_idx: np.ndarray,
    pos_ids: np.ndarray,
    swa_start_pos_ids: np.ndarray,
    replica_group,
    num_ranks: int,
    B: int,
    S_tkg: int,
    X_hidden_dim_actual: int,
    softmax_scale: float,
) -> dict:
    """Reference matching ``attention_block_tp_kernel`` (see module docstring)."""
    pg = get_pg(replica_group)
    dtype = X_shard.dtype
    T_shard, H = X_shard.shape

    # ── Dispatch: AllGather[TP] the SP token shards -> full [T, H].
    shard_t = torch.from_numpy(X_shard.astype(np.float32))
    gathered = [torch.zeros_like(shard_t) for _ in range(num_ranks)]
    dist.all_gather(gathered, shard_t, group=pg)
    X_full = torch.cat(gathered, dim=0).numpy().astype(dtype)  # [T, H]

    # ── Compute this rank's head-shard attention via the validated torch ref.
    # numpy<->torch conversion mirrors what the framework's torch_ref_wrapper does
    # (float16/bfloat16 -> float32 for the CPU ref); done inline so we can also
    # run the collectives (which need numpy/torch, not the wrapped signature).
    def _to_t(a):
        if a is None:
            return None
        s = str(a.dtype)
        if "bfloat16" in s or "float8" in s or "float16" in s:
            return torch.from_numpy(a.astype(np.float32))
        if a.dtype == np.uint32:
            return torch.from_numpy(a.astype(np.int32))
        return torch.from_numpy(a.copy())

    ref = AttentionBlockTkgTorchRef(lnc=1)
    result = ref.forward(
        _to_t(X_full).reshape(B, S_tkg, H),
        X_hidden_dim_actual=X_hidden_dim_actual,
        rmsnorm_X_enabled=False,
        rmsnorm_X_eps=None,
        rmsnorm_X_gamma=None,
        W_qkv=_to_t(W_qkv),
        bias_qkv=_to_t(bias_qkv),
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=0.0,
        rmsnorm_QK_pre_rope_W_Q=None,
        rmsnorm_QK_pre_rope_W_K=None,
        cos=_to_t(cos),
        sin=_to_t(sin),
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        skip_attention=False,
        K_cache_transposed=False,
        active_blocks_table=_to_t(active_blocks_table),
        K_cache=_to_t(K_cache),
        V_cache=_to_t(V_cache),
        attention_mask=_to_t(attention_mask),
        sink=_to_t(sink),
        softmax_scale=softmax_scale,
        k_scale=None,
        v_scale=None,
        update_cache=False,
        kv_cache_update_idx=_to_t(kv_cache_update_idx),
        W_out=_to_t(W_out),
        bias_out=_to_t(bias_out),
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        transposed_in=False,
        out_in_sb=False,
        pos_ids=_to_t(pos_ids),
        swa_start_pos_ids=_to_t(swa_start_pos_ids),
    )
    partial = result["X_out"].to(torch.float32).reshape(-1, H)  # [T, H] this rank's partial

    # ── Combine: ReduceScatter[TP] the O-proj partials -> [T_shard, H].
    chunks = list(partial.chunk(num_ranks, dim=0))
    out = torch.zeros_like(chunks[0])
    dist.reduce_scatter(out, chunks, op=dist.ReduceOp.SUM, group=pg)
    return {"out": out.numpy().astype(dtype)}
