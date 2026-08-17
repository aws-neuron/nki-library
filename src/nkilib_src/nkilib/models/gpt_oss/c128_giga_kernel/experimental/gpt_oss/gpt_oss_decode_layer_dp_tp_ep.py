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

"""Full GPT-OSS decoder layer with collectives: DP x TP attention + EP MoE (LNC=1).

Composes the two validated collective blocks into one fused ``@nki.jit`` layer:

    residual = x
    x = AllGather[TP] -> attention_block_tkg(input_layernorm + QKV + RoPE +
                          paged attn + sinks + O-proj partial) -> ReduceScatter[TP]
    x = residual + x
    residual = x
    x = AllGather[EP] -> moe_block_tkg(post_attn_layernorm + router top-k +
                          MXFP4 SwiGLU expert) -> ReduceScatter[EP]
    x = residual + x

Two collective topologies coexist in one NEFF (precedent: attention_block_tkg
composes KVDP+CP groups): ``tp_replica_group`` is the per-DP-replica TP group
(e.g. ``[[0..7],[8..15]]`` for DP2TP8), ``ep_replica_group`` spans all ranks
(``[[0..15]]`` for EP16). ncc collectives take the group per call.

Token bookkeeping (DP2 x TP8, EP16, global batch B tokens):
  * rank r is in DP replica ``d = r // TP`` at TP position ``p = r % TP``; it owns
    the SP shard ``B/(DP*TP)`` of its replica's ``B/DP`` tokens.
  * Attention AllGather[TP] reconstructs the replica's ``B/DP`` tokens for QKV;
    O-proj RS[TP] returns ``B/(DP*TP)`` per rank. So each rank holds ``B/16`` after
    attention. The residual add is at that ``B/16`` shard granularity.
  * MoE AllGather[EP] over all 16 ranks concatenates ``16 * B/16 = B`` tokens (both
    DP replicas), so every expert sees every token; RS[EP] returns ``B/16`` per
    rank. The layout is consistent end-to-end (token-SP over the 16-rank world).

All attention weights are the per-rank TP head shard (bf16); the expert weights
are this rank's single MXFP4 local expert; the router is replicated.
"""

import os as _os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.attention.attention_tkg import _ABT_CACHE, _ABT_V_REARRANGED_CACHE
from ...core.attention.attention_tkg_utils import is_qk_swapped
from ...core.attention.gen_mask_tkg import gen_mask_tkg_hbm
from ...core.moe.moe_tkg.down_projection_mx import _DOWN_PREFETCH_FIFO, _MOE_OUT_SBUF_CAPTURE, prefetch_down_weight_sb
from ...core.moe.moe_tkg.gate_up_projection_mx import _GATE_UP_PREFETCH_FIFO, prefetch_gate_up_weight_sb
from ...core.moe_block.moe_block_tkg import moe_block_tkg
from ...core.router_topk.router_topk import XSBLayout_tp201__2 as _ROUTER_X_SB_LAYOUT
from ...core.router_topk.router_topk import router_topk as _router_topk
from ...core.subkernels.rmsnorm_mx_quantize_tkg import rmsnorm_mx_quantize_tkg as _rmsnorm_mx_quantize_tkg
from ...core.utils.allocator import create_auto_alloc_manager, sizeinbytes
from ...core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    QuantizationType,
    RouterActFnType,
)
from ..transformer.attention_block_tkg import _ABT_FOLD_CACHE, attention_block_tkg


@nki.jit
def gpt_oss_decode_layer_dp_tp_ep_kernel(
    # ── input (SP token shard) ──
    x_shard: nl.ndarray,  # [B/(DP*TP), H]
    # ── attention (per-rank TP head shard) ──
    input_layernorm_weight: nl.ndarray,  # [1, H] input RMSNorm gamma
    W_qkv: nl.ndarray,
    bias_qkv: nl.ndarray,
    W_out: nl.ndarray,
    bias_out: nl.ndarray,
    cos: nl.ndarray,
    sin: nl.ndarray,
    sink: nl.ndarray,
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    active_blocks_table: nl.ndarray,
    attention_mask: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    pos_ids: nl.ndarray,
    swa_start_pos_ids: nl.ndarray,
    # ── MoE (this rank's 1 MXFP4 expert; router replicated) ──
    post_attn_layernorm_weight: nl.ndarray,  # [1, H] MoE RMSNorm gamma
    router_weights: nl.ndarray,
    router_bias: nl.ndarray,
    expert_gate_up_weights: nl.ndarray,
    expert_down_weights: nl.ndarray,
    expert_gate_up_weights_scale: nl.ndarray,
    expert_down_weights_scale: nl.ndarray,
    expert_gate_up_bias: nl.ndarray,
    expert_down_bias: nl.ndarray,
    rank_id: nl.ndarray,
    # ── groups + scalars ──
    tp_replica_group: ReplicaGroup,
    ep_replica_group: ReplicaGroup,
    tp_size: int,
    ep_size: int,
    B_attn: int,  # tokens per DP replica seen by attention = B / DP
    S_tkg: int,
    top_k: int,
    eps: float,
    hidden_actual: int,
    softmax_scale: float,
    gate_clamp_upper_limit: float,
    up_clamp_upper_limit: float,
    up_clamp_lower_limit: float,
):
    """One rank's full decoder layer (see module docstring). Returns [B/(DP*TP), H]."""
    # Positional call (NKI tracer rejects keyword args to nested functions).
    return _decode_layer_body(
        x_shard,
        input_layernorm_weight,
        W_qkv,
        bias_qkv,
        W_out,
        bias_out,
        cos,
        sin,
        sink,
        K_cache,
        V_cache,
        active_blocks_table,
        attention_mask,
        kv_cache_update_idx,
        pos_ids,
        swa_start_pos_ids,
        post_attn_layernorm_weight,
        router_weights,
        router_bias,
        expert_gate_up_weights,
        expert_down_weights,
        expert_gate_up_weights_scale,
        expert_down_weights_scale,
        expert_gate_up_bias,
        expert_down_bias,
        rank_id,
        tp_replica_group,
        ep_replica_group,
        tp_size,
        ep_size,
        B_attn,
        S_tkg,
        top_k,
        eps,
        hidden_actual,
        softmax_scale,
        gate_clamp_upper_limit,
        up_clamp_upper_limit,
        up_clamp_lower_limit,
    )


def _unpack_prequant_rows(rows_hbm, T_ep, H, quant_dtype, row_width=None):
    """AllGather'd per-token rows [T_ep, row_width] -> moe's [H0, H/512, T_ep] quant/scale pair.

    The row layout exists because HBM AllGather concatenates on dim 0 only, while moe wants T
    innermost. Quant data is [:, :H], MX scales [:, H:H+H/4]; both need the same token->partition
    transpose. Per-tile widths SUM to the global width -- they do not multiply it.
    ``row_width`` > H+H/4 means extra payload (affinities) rides in the same row; it is ignored
    here but must be sliced off before the views, whose reshapes assume the H-derived widths.
    """
    H0 = nl.tile_size.pmax
    n_h512 = H // 512
    H_pack = H + H // 4
    quant_sb = nl.ndarray((H0, n_h512, T_ep), dtype=quant_dtype, buffer=nl.sbuf)
    scale_sb = nl.ndarray((H0, n_h512, T_ep), dtype=nl.uint8, buffer=nl.sbuf)
    # PE has no uint8 input and needs dst dtype == src dtype, so bitcast scales to a 1-byte
    # float for the transpose (same trick as norm_tkg_utils._UINT8_TP_VIEW_DTYPE); 1-byte PSUM
    # needs the stride-2 layout.
    _u8 = nl.float8_e5m2
    # T_ep rides the PARTITION axis on load and the transpose FREE axis, both capped at 128. At
    # T_shard=1, T_ep == 128 and this is a single tile; T_shard>1 needs it tiled or the load
    # asserts "dma_copy dst partition dimension 256 exceeds maximum 128".
    for t0 in range(0, T_ep, H0):
        t_n = min(H0, T_ep - t0)
        rows_sb = nl.load(rows_hbm[t0 : t0 + t_n, 0:H_pack])
        q_src = rows_sb[:, 0:H].view(nl.float32).reshape((t_n, n_h512, H0))
        s_src = rows_sb[:, H:H_pack].view(_u8).reshape((t_n, n_h512, H0))
        for h in nl.affine_range(n_h512):
            q_psum = nl.ndarray((H0, t_n), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(data=q_src[:, h, :], dst=q_psum)
            nisa.tensor_copy(dst=quant_sb[:, h, t0 : t0 + t_n].view(nl.float32), src=q_psum)
            s_psum = nl.ndarray((H0, t_n, 2), dtype=_u8, buffer=nl.psum)
            nisa.nc_transpose(data=s_src[:, h, :], dst=s_psum[:, :, 0])
            nisa.tensor_copy(dst=scale_sb[:, h, t0 : t0 + t_n].view(_u8), src=s_psum[:, :, 0])
    return quant_sb, scale_sb


def _decode_layer_body(
    x_shard,
    input_layernorm_weight,
    W_qkv,
    bias_qkv,
    W_out,
    bias_out,
    cos,
    sin,
    sink,
    K_cache,
    V_cache,
    active_blocks_table,
    attention_mask,
    kv_cache_update_idx,
    pos_ids,
    swa_start_pos_ids,
    post_attn_layernorm_weight,
    router_weights,
    router_bias,
    expert_gate_up_weights,
    expert_down_weights,
    expert_gate_up_weights_scale,
    expert_down_weights_scale,
    expert_gate_up_bias,
    expert_down_bias,
    rank_id,
    tp_replica_group,
    ep_replica_group,
    tp_size,
    ep_size,
    B_attn,
    S_tkg,
    top_k,
    eps,
    hidden_actual,
    softmax_scale,
    gate_clamp_upper_limit,
    up_clamp_upper_limit,
    up_clamp_lower_limit,
    layer_tag="",
    is_last_layer=True,
    pregen_mask=None,
    k_scale=None,
    v_scale=None,
    fp8_packed=False,
):
    """One decoder layer's compute (NOT @nki.jit — a reusable body).

    NOTE: called POSITIONALLY (the NKI tracer does not support keyword arguments
    on nested-function calls inside a kernel). Argument order must match both
    call sites (the single-layer entry and the mega-kernel loop).

    Takes this rank's SP token shard ``[B/(DP*TP), H]`` plus this layer's weights
    and KV cache, and returns the post-layer SP token shard ``[B/(DP*TP), H]`` in
    the SAME layout, so a multi-layer kernel can feed it straight into the next
    layer. All collectives/tensors are per-call (the two ReplicaGroups and the
    shared cos/sin/pos_ids/block table are reused across layers; the weights and
    KV cache are per-layer).

    ``layer_tag``: a unique per-layer suffix appended to every named HBM buffer,
    so that when the body is inlined once per layer in the mega kernel the op/
    buffer names stay globally unique (NKI requires unique op names in one NEFF).
    """
    _t = str(layer_tag)
    T_shard, H = x_shard.shape
    dtype = x_shard.dtype
    T_attn = T_shard * tp_size  # B/DP: tokens per DP replica (attention gather)
    T_ep = T_shard * ep_size  # B: global tokens (EP gather)

    # PROTOTYPE (tracer only): inter-layer SBUF hand-off. When on, the previous
    # layer's output arrives as an SBUF [T_shard, H] tensor (x_shard) instead of HBM,
    # and this layer keeps its own output in SBUF too (returned to the next layer),
    # storing to HBM only on the last layer. x_shard is consumed twice below: as the
    # AllGather[TP] source (dma_copy accepts SBUF src) and in residual add #1
    # (nl.load only reads HBM, so branch on the buffer type).
    _sbuf_interlayer = bool(_os.environ.get("VLLM_NEURON_MEGA_SBUF_INTERLAYER"))
    _moe_sbuf_collective = bool(_os.environ.get("VLLM_NEURON_MEGA_MOE_SBUF_COLLECTIVE"))
    # FP8 EP ReduceScatter: RS is ~3.8x slower per byte than pure movement (31.1 vs 118.6 GB/s),
    # so halving the payload is the lever that shortens it. Safe because with 128 experts and
    # top_k=4 each rank's partial is zero for ~124 of the 128 global tokens.
    _fp8_rs = bool(_os.environ.get("VLLM_NEURON_MEGA_FP8_RS"))
    # Pre-gather quantize+router: RMSNorm+MX-quantize+router are per-token ops, so running them
    # after the EP AllGather makes all T_ep ranks recompute T_ep identical copies (the swizzle
    # alone moves exactly T*H*2 bytes). Do them for THIS rank's token and gather the results.
    _prequant_moe = bool(_os.environ.get("VLLM_NEURON_MEGA_PREQUANT_MOE"))
    # FUSE_RESID2 handoff: set by the fp8-RS branch, consumed by residual add #2. Declared here so
    # every other path falls through to the plain tensor_tensor add.
    _fuse_resid2 = False
    _fuse_rs_src = None
    _fuse_rs_scale = 1.0
    _prequant_aff16 = bool(_os.environ.get("VLLM_NEURON_MEGA_PREQUANT_AFF16"))
    _fp8_rs_scale = float(_os.environ.get("VLLM_NEURON_MEGA_FP8_RS_SCALE", "1") or 1)
    # Attention output stays in SBUF and the TP ReduceScatter runs as a SBUF collective;
    # residual add #1 stays in SBUF; only the resid1 result is written to HBM for the EP AllGather.
    _sbuf_attn_rs = bool(_os.environ.get("VLLM_NEURON_MEGA_SBUF_ATTN_RS"))
    # KV CACHE UPDATE (VLLM_NEURON_MEGA_KV_UPDATE=1): write this decode token's K/V back into
    # the paged K_cache/V_cache in-place (attention_block_tkg(update_cache=True) returns the
    # updated caches). The @nki.jit mega kernel returns them so the harness can alias them onto
    # the K_cache/V_cache input buffers (test passes *.must_alias_input). Correctness note: each
    # layer writes its own K_cache[li_k] slice, so caches MUST be stacked per-layer (leading dim
    # == num_layers); do NOT combine with SHARE_LAYER_INPUTS (all layers would alias slice 0).
    # PROFILING note: SHARE_LAYER_INPUTS must be DISABLED for a representative profile. With it on,
    # every layer indexes the SAME shared weight/cache tensors (leading dim 1 -> li_*=0), so the
    # compiler keeps per-layer weights SBUF-resident across layers and skips the re-load DMA that a
    # real 36-distinct-layer decode pays. A shared-input profile therefore UNDERCOUNTS weight DMA
    # (and any KV-update-vs-baseline DMA delta is dominated by that layer_input_count 1<->36
    # difference, not the KV write). Profile KV-update AND its baseline with sharing off.
    _kv_update = bool(_os.environ.get("VLLM_NEURON_MEGA_KV_UPDATE"))
    _x_in_sbuf = _sbuf_interlayer and (getattr(x_shard, "buffer", None) == nl.sbuf)

    # Per-layer SBUF manager with a unique name prefix so attention_block_tkg's
    # internal named buffers stay globally unique when inlined once per layer.
    _sbm = create_auto_alloc_manager()
    _sbm.set_name_prefix(f"attn{_t}_")

    # ── PROTOTYPE: prefetch this layer's MoE gate/up weight into SBUF NOW, before
    # attention, so the ~9 MB HBM->SBUF DMA overlaps the attention matmuls instead
    # of stalling the first MoE gate matmul. moe_block_tkg's leaf loader pops the
    # enqueued SBUF buffer (FIFO per gate/up idx) instead of alloc+DMA. LNC=1
    # no-shard => full I, no I-offset/padding. (Tracer frontend only.)
    # This placement (before attention) is the measured optimum. Five alternatives were tried and
    # all lost: after attention (+92 us, starves both EP collectives), after the EP collectives
    # (INFERENCE_FAILURE at L36), split into 4/8/16 DMAs (monotonically worse), immediately before
    # RS-TP (inflates RS-TP 7.89 -> 12.83 us), and true depth-2 cross-layer pipelining (+0.53 ms,
    # destroys the cross-layer residency the single-buffer form already gets). See
    # scheduling-experiment/FULL128_TUNING.md for the per-arm numbers.
    _I_gu = expert_gate_up_weights.shape[-1]
    for _gu_idx in (0, 1):  # 0=gate, 1=up (fused)
        prefetch_gate_up_weight_sb(
            weight=expert_gate_up_weights,
            expert_idx=0,
            gate_or_up_idx=_gu_idx,
            H=H,
            I_local=_I_gu,
            I_offset=0,
            I_local_padded=0,
            name=f"gup_prefetch_{_gu_idx}" + _t,
        )
    # Down weight too: gate/up is already prefetched, so the down stream is the last large DMA
    # still on the critical path. LNC=1 => no I-sharding, so n_I512_tiles is the full I/512 and
    # tile_offset is 0 (shape [E_L, I_p, I/512, H]).
    if _os.environ.get("VLLM_NEURON_MEGA_DOWN_PREFETCH"):
        prefetch_down_weight_sb(
            weight=expert_down_weights,
            expert_idx=0,
            H=H,
            tile_I=nl.tile_size.pmax,
            n_I512_tiles=expert_down_weights.shape[2],
            tile_offset=0,
            name="down_prefetch" + _t,
        )

    # ════════════════ Attention (input RMSNorm fused inside) ════════════════
    # Dispatch: AllGather[TP] SP shard -> this replica's full [B/DP, H].
    a_src = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.shared_hbm, name="a_src" + _t)
    a_dst = nl.ndarray((T_attn, H), dtype=dtype, buffer=nl.shared_hbm, name="a_dst" + _t)
    nisa.dma_copy(dst=a_src, src=x_shard)
    ncc.all_gather(dsts=[a_dst], srcs=[a_src], replica_group=tp_replica_group, collective_dim=0)
    X = a_dst.reshape((B_attn, S_tkg, H))

    # KV UPDATE: capture the updated caches attention returns (they alias K_cache/V_cache
    # in-place when update_cache=True). When off, these are the new-token HBM tensors (unused).
    attn_out, _K_upd, _V_upd = attention_block_tkg(
        X=X,
        X_hidden_dim_actual=hidden_actual,
        rmsnorm_X_enabled=True,  # <-- the layer's input_layernorm
        rmsnorm_X_eps=eps,
        rmsnorm_X_gamma=input_layernorm_weight,
        W_qkv=W_qkv,
        bias_qkv=bias_qkv,
        quantization_type_qkv=QuantizationType.NONE,
        weight_dequant_scale_qkv=None,
        input_dequant_scale_qkv=None,
        rmsnorm_QK_pre_rope_enabled=False,
        rmsnorm_QK_pre_rope_eps=0.0,
        rmsnorm_QK_pre_rope_W_Q=None,
        rmsnorm_QK_pre_rope_W_K=None,
        cos=cos,
        sin=sin,
        rope_contiguous_layout=True,
        rmsnorm_QK_post_rope_enabled=False,
        rmsnorm_QK_post_rope_eps=0.0,
        rmsnorm_QK_post_rope_W_Q=None,
        rmsnorm_QK_post_rope_W_K=None,
        K_cache_transposed=False,
        active_blocks_table=active_blocks_table,
        K_cache=K_cache,
        V_cache=V_cache,
        # MASK HOIST (tracer only): when a pre-generated FULL mask is supplied, pass it
        # as attention_mask with pos_ids/swa_start=None so attention_block_tkg LOADS it
        # (cheap DMA) instead of regenerating gen_mask_tkg on-chip every layer. The mask
        # is layer-invariant (depends only on pos_ids/swa_start/block_len, not the block
        # table), so the caller builds one full-attn + one SWA mask once and reuses them.
        attention_mask=pregen_mask if pregen_mask is not None else attention_mask,
        sink=sink,
        update_cache=_kv_update,
        kv_cache_update_idx=kv_cache_update_idx,
        W_out=W_out,
        bias_out=bias_out,
        quantization_type_out=QuantizationType.NONE,
        weight_dequant_scale_out=None,
        input_dequant_scale_out=None,
        transposed_out=False,
        out_in_sb=_sbuf_attn_rs,  # SBUF [T_attn, H] output -> SBUF TP ReduceScatter (no HBM staging)
        transposed_in=False,
        softmax_scale=softmax_scale,
        pos_ids=None if pregen_mask is not None else pos_ids,
        swa_start_pos_ids=None if pregen_mask is not None else swa_start_pos_ids,
        # FP8 KV cache: raw fp8 K/V + scales; softmax_scale is pre-fused with k_scale
        # and W_out pre-fused with v_scale by the caller. None (unset) => bf16/fp16 KV.
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_packed=fp8_packed,  # packed K [nb, bl//2, d, 2] -> DMA-gather-transpose path
        sbm=_sbm,  # per-layer name prefix -> unique attention buffer names
    )  # [B/DP, H] per-rank O-proj partial
    resid1 = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.shared_hbm, name="resid1" + _t)
    if _sbuf_attn_rs:
        # PROTOTYPE (tracer only): attention_block_tkg emitted its [T_attn, H] O-proj partial
        # straight to SBUF (out_in_sb=True). Feed it to reduce_scatter as a SBUF collective
        # (src SBUF + dst SBUF, collective_dim=0), dropping the ars_src HBM staging copy.
        # Residual add #1 then runs fully in SBUF; only the result is written to HBM (resid1)
        # for the pre-MoE EP AllGather (whose RMSNorm reads its input from HBM).
        attn_shard_sb = nl.ndarray((T_shard, H), dtype=attn_out.dtype, buffer=nl.sbuf)
        ncc.reduce_scatter(
            dsts=[attn_shard_sb], srcs=[attn_out], op=nl.add, replica_group=tp_replica_group, collective_dim=0
        )
        x_sb = x_shard if _x_in_sbuf else nl.load(x_shard)
        resid1_sb = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=resid1_sb, data1=x_sb, data2=attn_shard_sb, op=nl.add)
        nl.store(resid1, value=resid1_sb)
    else:
        ars_src = nl.ndarray((T_attn, H), dtype=attn_out.dtype, buffer=nl.shared_hbm, name="ars_src" + _t)
        attn_shard = nl.ndarray((T_shard, H), dtype=attn_out.dtype, buffer=nl.shared_hbm, name="attn_shard" + _t)
        nisa.dma_copy(dst=ars_src, src=attn_out)
        ncc.reduce_scatter(
            dsts=[attn_shard], srcs=[ars_src], op=nl.add, replica_group=tp_replica_group, collective_dim=0
        )

        # ── residual add #1 (token-SP, B/(DP*TP) shard) ──
        # Element-wise add of two [T_shard, H] HBM tensors: DMA both into SBUF
        # ([T_shard<=128] on the partition axis, H on the free axis), tensor_tensor
        # add on-chip, then store the SBUF result back to HBM.
        # x_shard may already be resident in SBUF (inter-layer hand-off); nl.load reads HBM
        # only, so use the SBUF tensor directly in that case.
        x_sb = x_shard if _x_in_sbuf else nl.load(x_shard)
        a_sb = nl.load(attn_shard)
        resid1_sb = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=resid1_sb, data1=x_sb, data2=a_sb, op=nl.add)
        nl.store(resid1, value=resid1_sb)
    m_src = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.shared_hbm, name="m_src" + _t)
    m_dst = nl.ndarray((T_ep, H), dtype=dtype, buffer=nl.shared_hbm, name="m_dst" + _t)
    _pq_quant = None
    _pq_scale = None
    _pq_affin = None
    if not _prequant_moe:
        nisa.dma_copy(dst=m_src, src=resid1)
        ncc.all_gather(dsts=[m_dst], srcs=[m_src], replica_group=ep_replica_group, collective_dim=0)
    moe_in = m_dst.reshape((1, T_ep, H))

    if _prequant_moe:
        H_pack = H + H // 4
        E_total = router_weights.shape[-1]
        # Dense buffers on BOTH ends: a collective cannot read/write a sub-range of a wider
        # tensor ("Output pattern is not contiguous"), and feeding a gather from a bitcast slice
        # of a shared row silently corrupts the payload (rank0 fine, all peers garbage).
        pq_src = nl.ndarray((T_shard, H_pack), dtype=nl.float8_e4m3fn, buffer=nl.shared_hbm, name="pq_src" + _t)
        pq_norm_sb = nl.ndarray((nl.tile_size.pmax, T_shard, H // nl.tile_size.pmax), dtype=nl.float16, buffer=nl.sbuf)
        # Packed layout is selected by output_quant.shape == (T, H+H/4) with output_scale=None.
        _rmsnorm_mx_quantize_tkg(
            input=resid1.reshape((1, T_shard, H)),
            gamma=post_attn_layernorm_weight,
            output=pq_norm_sb,
            output_quant=pq_src,
            output_scale=None,
            eps=eps,
            hidden_actual=hidden_actual,
            hidden_dim_tp=True,
        )
        # Router on this rank's own token; its affinities are what every other rank would have
        # computed for it. router_topk stores via _hbm_tiled_store_view, which assumes the dst's
        # last dim IS its row pitch -> must be a dense [T, E] tensor.
        # The affinity gather costs 21.6 us to move only 64 KB (13.5 us above the AllGather
        # model) while the 8x-larger quant gather takes 13.2 us -- it is overhead-dominated, so
        # halving the payload is the only handle. Affinities are softmax probabilities in [0,1]
        # used to rank/scale top_k=4; moe already runs them in fp16 on its dynamic path
        # (moe_block_tkg.py:324) and io_dtype is derived from this buffer's dtype, so fp16 flows
        # through consistently.
        _aff_dt = nl.float16 if _prequant_aff16 else nl.float32
        aff_src = nl.ndarray((T_shard, E_total), dtype=_aff_dt, buffer=nl.shared_hbm, name="aff_src" + _t)
        eidx_sb = nl.ndarray((T_shard, top_k), dtype=nl.uint32, buffer=nl.sbuf)
        quant_dst = nl.ndarray((T_ep, H_pack), dtype=nl.float8_e4m3fn, buffer=nl.shared_hbm, name="quant_dst" + _t)
        _router_topk(
            x=pq_norm_sb,
            w=router_weights,
            w_bias=router_bias,
            router_logits=None,
            expert_affinities=aff_src,
            expert_index=eidx_sb,
            act_fn=RouterActFnType.SOFTMAX,
            k=top_k,
            x_hbm_layout=0,
            x_sb_layout=_ROUTER_X_SB_LAYOUT,
            router_pre_norm=False,
            norm_topk_prob=False,
            use_column_tiling=True,
            use_PE_broadcast_w_bias=True,
            skip_store_expert_index=True,
            skip_store_router_logits=True,
        )
        # Two separate gathers. Fusing them into one wider collective was measured twice and
        # regressed (+0.15 ms): the collective it removes is the already-hidden 256 B affinity
        # gather (10.7 of 12.5 us overlapped), while the local DMAs that build the fused row add
        # unhidden serial work. Issuing the quant gather before the router also lost.
        aff_dst = nl.ndarray((T_ep, E_total), dtype=_aff_dt, buffer=nl.shared_hbm, name="aff_dst" + _t)
        # fp8 AllGather is unsupported (runtime error 1202) -> bitcast to bf16, identical bytes.
        # Two separate gathers: ncc's coalesced list-form is correct but pathologically slow for
        # these mismatched widths (490 us vs 13.7 us, +235% e2e).
        ncc.all_gather(
            dsts=[quant_dst.view(nl.bfloat16)],
            srcs=[pq_src.view(nl.bfloat16)],
            replica_group=ep_replica_group,
            collective_dim=0,
        )
        ncc.all_gather(dsts=[aff_dst], srcs=[aff_src], replica_group=ep_replica_group, collective_dim=0)
        _pq_quant, _pq_scale = _unpack_prequant_rows(
            quant_dst,
            T_ep=T_ep,
            H=H,
            quant_dtype=nl.float8_e4m3fn_x4,
        )
        _pq_affin = aff_dst

    (moe_out,) = moe_block_tkg(
        inp=moe_in,
        gamma=post_attn_layernorm_weight,  # <-- the layer's post_attention_layernorm
        router_weights=router_weights,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        router_bias=router_bias,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        eps=eps,
        top_k=top_k,
        router_act_fn=RouterActFnType.SOFTMAX,
        router_pre_norm=False,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        hidden_act_fn=ActFnType.Swish,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        router_mm_dtype=nl.float16,
        hidden_actual=hidden_actual,
        skip_router_logits=True,
        is_all_expert=True,
        rank_id=rank_id,
        prequant_input=_pq_quant,
        prequant_scale=_pq_scale,
        prequant_affinities=_pq_affin,
    )  # [B, H] this expert's affinity-scaled partial over all global tokens

    # Combine: ReduceScatter[EP] sum per-expert partials + re-shard -> [B/(DP*TP), H].
    moe_shard = nl.ndarray((T_shard, H), dtype=moe_out.dtype, buffer=nl.shared_hbm, name="moe_shard" + _t)
    moe_shard_sb = None
    if _moe_sbuf_collective:
        # PROTOTYPE (tracer only): the MX all-expert MoE already accumulated its full
        # [T_ep, H] result in an SBUF buffer; down_projection_mx stashed it in a FIFO
        # instead of spilling to HBM. Pop it and feed reduce_scatter straight from SBUF
        # (src SBUF + dst SBUF, collective_dim=0), removing the ~T_ep*H HBM staging copy.
        moe_out_sb = _MOE_OUT_SBUF_CAPTURE.pop(0).reshape((T_ep, H))
        # reduce_scatter requires src/dst same dtype; the captured accumulator is
        # activation_compute_dtype (bf16), which may differ from moe_out.dtype. Keep the
        # collective in the accumulator dtype, then let the HBM store cast to moe_out.dtype.
        if _fp8_rs:
            _rs_dt = nl.float8_e4m3fn  # OCP E4M3 (max 448); NEURON_RT_ENABLE_OCP=1 on trn3
            rs_src = nl.ndarray((T_ep, H), dtype=_rs_dt, buffer=nl.sbuf)
            if _fp8_rs_scale != 1:
                nisa.tensor_scalar(dst=rs_src, data=moe_out_sb, op0=nl.multiply, operand0=_fp8_rs_scale)
            else:
                nisa.tensor_copy(dst=rs_src, src=moe_out_sb)
            rs_dst = nl.ndarray((T_shard, H), dtype=_rs_dt, buffer=nl.sbuf)
            # NOTE: all_to_all + a local partition-sum was tested here as a movement-only
            # replacement (arm `pqA2Apatch`, needs the mesh-a2a descriptor patch to run at all)
            # and is NOT faster: 23.15 us vs 23.56 us for the same 393216 B, e2e -7.78% vs
            # -14.65%. The 118.6 GB/s "movement" rate was fitted on AllGather (one row
            # broadcast); an a2a is a full transpose, so per-rank wire traffic is ~T_ep x larger
            # for the same dst size. Do not retry.
            # RS-EP has the least compute hidden behind it (0.64 of 22 us). DMA QoS priority was
            # tested at both extremes (0 and 3) and is inert for this traffic mix; multi-stream CC
            # is unreachable from the NKI path. Staging through dense HBM to make the collective
            # ONE_D also did not help. See FULL128_TUNING.md.
            ncc.reduce_scatter(
                dsts=[rs_dst], srcs=[rs_src], op=nl.add, replica_group=ep_replica_group, collective_dim=0
            )
            # FUSE_RESID2: the fp8-RS dequant here and residual add #2 below are two full-width
            # Vector passes over the same [T_shard, H] data. scalar_tensor_tensor computes
            # (rs_dst * 1/scale) + resid1 in ONE pass, removing a Vector op that the
            # exclusive-critical-path analysis measures at ~1 us/layer (Vector is the busiest
            # engine and is exclusively on the critical path 17.67 us/layer). Deferred: the
            # residual operand is loaded below, so set a marker and let residual #2 do the fusion.
            _fuse_resid2 = bool(_os.environ.get("VLLM_NEURON_MEGA_FUSE_RESID2"))
            moe_shard_sb = nl.ndarray((T_shard, H), dtype=moe_out_sb.dtype, buffer=nl.sbuf)
            if _fuse_resid2:
                _fuse_rs_src = rs_dst
                _fuse_rs_scale = 1.0 / _fp8_rs_scale if _fp8_rs_scale != 1 else 1.0
            elif _fp8_rs_scale != 1:
                nisa.tensor_scalar(dst=moe_shard_sb, data=rs_dst, op0=nl.multiply, operand0=1.0 / _fp8_rs_scale)
            else:
                nisa.tensor_copy(dst=moe_shard_sb, src=rs_dst)
        else:
            moe_shard_sb = nl.ndarray((T_shard, H), dtype=moe_out_sb.dtype, buffer=nl.sbuf)
            ncc.reduce_scatter(
                dsts=[moe_shard_sb], srcs=[moe_out_sb], op=nl.add, replica_group=ep_replica_group, collective_dim=0
            )
        # Inter-layer SBUF keeps moe_shard_sb resident and skips the HBM round-trip;
        # otherwise store back to HBM (reloaded below).
        if not _sbuf_interlayer:
            nl.store(moe_shard, value=moe_shard_sb)
    else:
        mrs_src = nl.ndarray((T_ep, H), dtype=moe_out.dtype, buffer=nl.shared_hbm, name="mrs_src" + _t)
        nisa.dma_copy(dst=mrs_src, src=moe_out)
        ncc.reduce_scatter(
            dsts=[moe_shard], srcs=[mrs_src], op=nl.add, replica_group=ep_replica_group, collective_dim=0
        )

    # ── residual add #2 -> layer output [B/(DP*TP), H] ──
    # Same HBM -> SBUF -> tensor_tensor(add) -> HBM idiom as residual add #1.
    out = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.shared_hbm, name="mega_out" + _t)
    out_sb = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.sbuf)
    r_sb = nl.load(resid1)
    # residual1 add MoE: source the MoE shard from SBUF when we kept it resident
    # (inter-layer), else reload from HBM. tensor_tensor casts moe_shard_sb (bf16) to
    # out_sb dtype (fp16) implicitly.
    if _sbuf_interlayer and moe_shard_sb is not None:
        mo_sb = moe_shard_sb
    else:
        mo_sb = nl.load(moe_shard)
    if _fuse_resid2 and _fuse_rs_src is not None:
        # One pass: (rs_dst * 1/scale) + resid1 -> out_sb, replacing dequant + add.
        nisa.scalar_tensor_tensor(
            dst=out_sb, data=_fuse_rs_src, op0=nl.multiply, operand0=_fuse_rs_scale, op1=nl.add, operand1=r_sb
        )
    else:
        nisa.tensor_tensor(dst=out_sb, data1=r_sb, data2=mo_sb, op=nl.add)
    # Inter-layer SBUF: hand the output to the next layer straight from SBUF; store to
    # HBM only on the final layer (the @nki.jit kernel must return an HBM tensor).
    if _sbuf_interlayer and not is_last_layer:
        return out_sb
    nl.store(out, value=out_sb)
    return out


@nki.jit
def gpt_oss_decode_mega_kernel(
    # ── input (SP token shard) ──
    x_shard: nl.ndarray,  # [B/(DP*TP), H]
    # ── per-layer weights: STACKED on a leading layer axis [num_layers, ...] ──
    # (NKI rejects Python-list kernel args; integer-indexing an HBM tensor removes
    #  dim 0, so W[i] yields that layer's tensor in the exact per-layer shape.)
    input_layernorm_weight: nl.ndarray,  # [L, 1, H]
    W_qkv: nl.ndarray,  # [L, H, (q + 2*kv)*d]
    bias_qkv: nl.ndarray,  # [L, 1, (q + 2*kv)*d]
    W_out: nl.ndarray,  # [L, q*d, H]
    bias_out: nl.ndarray,  # [L, 1, H]
    sink: nl.ndarray,  # [L, q, 1]
    K_cache: nl.ndarray,  # [L, nb, kv, block_len, d]
    V_cache: nl.ndarray,  # [L, nb, kv, block_len, d]
    post_attn_layernorm_weight: nl.ndarray,  # [L, 1, H]
    router_weights: nl.ndarray,  # [L, H, E]
    router_bias: nl.ndarray,  # [L, 1, E]
    expert_gate_up_weights: nl.ndarray,  # [L, ...] this rank's expert / layer
    expert_down_weights: nl.ndarray,  # [L, ...]
    expert_gate_up_weights_scale: nl.ndarray,  # [L, ...]
    expert_down_weights_scale: nl.ndarray,  # [L, ...]
    expert_gate_up_bias: nl.ndarray,  # [L, ...]
    expert_down_bias: nl.ndarray,  # [L, ...]
    swa_start_pos_ids: nl.ndarray,  # [L, B_attn, S_tkg] per-layer SWA starts
    #   (full layers = 0 -> window [0,pos] = full causal)
    # ── shared attention control (same absolute positions + active mask) ──
    cos: nl.ndarray,
    sin: nl.ndarray,
    active_blocks_table: nl.ndarray,  # FULL block table [B_attn, nbf]; used by full (odd) layers
    active_blocks_table_swa: nl.ndarray,  # TRIMMED block table [B_attn, num_swa]; used by sliding (even) layers
    attention_mask: nl.ndarray,  # active-token mask; identical for SWA/full layers
    kv_cache_update_idx: nl.ndarray,
    pos_ids: nl.ndarray,  # ABSOLUTE positions [B_attn, S_tkg]; used by full (odd) layers
    pos_ids_swa: nl.ndarray,  # positions SHIFTED into the trimmed frame; used by sliding (even) layers
    rank_id: nl.ndarray,
    # ── groups + scalars ──
    tp_replica_group: ReplicaGroup,
    ep_replica_group: ReplicaGroup,
    num_layers: int,
    tp_size: int,
    ep_size: int,
    B_attn: int,
    S_tkg: int,
    top_k: int,
    eps: float,
    hidden_actual: int,
    softmax_scale: float,
    gate_clamp_upper_limit: float,
    up_clamp_upper_limit: float,
    up_clamp_lower_limit: float,
    # ── optional FP8 KV cache scales (layer-invariant; None => bf16/fp16 KV) ──
    # softmax_scale must already be k_scale-fused, W_out already v_scale-fused (caller).
    k_scale: nl.ndarray = None,
    v_scale: nl.ndarray = None,
):
    """Multi-layer GPT-OSS decode "mega kernel": stack ``num_layers`` fused
    DP x TP attention + EP MoE decoder layers in one NEFF.

    Each layer's output ``[B/(DP*TP), H]`` is the next layer's input (same SP
    layout), so the layers chain by threading the running ``x`` shard through
    :func:`_decode_layer_body`. Per-layer weights, KV caches, per-layer SWA starts
    and per-layer attention masks are passed **stacked on a leading layer axis**
    ``[num_layers, ...]`` (NKI kernels do not accept Python-list arguments;
    integer-indexing an HBM tensor removes dim 0, so ``W_qkv[i]`` is layer ``i``'s
    tensor in its natural per-layer shape). The RoPE tables, active-token mask,
    kv-update index, and both ReplicaGroups are shared across layers.

    **SWA block-table trimming (matches production vllm-neuron decode).** Real
    GPT-OSS alternates even = sliding-window (window 128), odd = full attention.
    Rather than reading the full context and masking scores (equal cost), the
    sliding layers read a **trimmed** block table covering only the window's
    ``num_swa`` blocks, so they touch far fewer KV blocks from HBM (e.g. 2 blocks
    vs 80 at block_len 128 / S_ctx 10240) — genuinely cheaper, exactly as the
    model runner's ``_compute_swa_decode_tensors`` does. So two block tables and
    two position vectors are passed: the full ``active_blocks_table`` / absolute
    ``pos_ids`` for odd (full) layers, and the trimmed ``active_blocks_table_swa``
    / window-frame ``pos_ids_swa`` (= pos - start_block*block_len) for even
    (sliding) layers. Because the stored K carries its own absolute-position RoPE,
    relabelling positions into the trimmed frame leaves the attention output
    bit-identical to the full-frame windowed result. ``swa_start_pos_ids`` is
    per-layer so the in-kernel banded window mask still matches.

    **Shared (layer-invariant) stacked inputs.** Every per-layer stacked tensor
    may be passed with a leading axis of either ``num_layers`` (distinct per
    layer) OR ``1`` (one copy, reused for every layer). This lets a caller avoid
    dumping/uploading ``num_layers`` identical copies of large layer-invariant
    tensors (e.g. the MoE expert weights are the same for every layer; a profiling
    run may also share one KV cache) — cutting host-side input serialization and
    transfer, which dominate wall time at long context. Selection is by leading
    dim: ``W[i]`` when ``W.shape[0] == num_layers``, else ``W[0]``. Correctness
    runs pass distinct ``[num_layers, ...]`` tensors as before; only the leading
    dim changes, not the per-layer shape. The leading dim and ``num_layers`` are
    both trace-time constants, so this is a pure Python index choice (NOT a
    runtime tensor branch, and NOT a nested traced call — both of which the NKI
    tracer rejects).

    Returns the final post-layer SP token shard ``[B/(DP*TP), H]``.
    """
    # ── Reset the trace-scoped module caches (MANDATORY) ─────────────────────
    # The prefetch FIFOs and the block-table memos are module globals. They are only valid
    # WITHIN one trace: the FIFOs hand SBUF buffers from a producer to a consumer inside this
    # kernel, and the memo keys use id() of trace-local tensors, which CPython recycles across
    # traces. Left over, a second trace in the same process pops another trace's buffers (the
    # down-weight FIFO consumer is unconditional -- it pops whenever non-empty) and silently
    # computes with the wrong weights. That regressed the golden gate to cosine 0.887 while the
    # profile runs, which trace once per process, looked fine. Clear on every kernel entry.
    _GATE_UP_PREFETCH_FIFO[0].clear()
    _GATE_UP_PREFETCH_FIFO[1].clear()
    _DOWN_PREFETCH_FIFO.clear()
    _MOE_OUT_SBUF_CAPTURE.clear()
    _ABT_CACHE.clear()
    _ABT_V_REARRANGED_CACHE.clear()
    _ABT_FOLD_CACHE.clear()

    # ── MASK HOIST (tracer only, env-gated) ──────────────────────────────────
    # The prior causal/SWA attention mask is layer-invariant (depends only on
    # pos_ids/swa_start/block_len, not the block table). Instead of regenerating it
    # in-kernel every layer (gen_mask_tkg, once per FA-tile per layer), build ONE
    # full-attn mask + ONE SWA mask to HBM here via gen_mask_tkg_hbm, then pass the
    # matching pre-generated mask per layer with pos_ids=None so attention just LOADS
    # it. Full layers use the full block-table span (S_ctx); SWA layers use the
    # trimmed span (num_swa*block_len). Derive each mask layout with the same
    # is_qk_swapped predicate used by attention because the two prior lengths can
    # select different paths.
    _hoist_mask = bool(_os.environ.get("VLLM_NEURON_MEGA_HOIST_MASK"))
    # Packed fp8 KV: stacked K_cache is 6D [L, nb, kv, block_len//2, d, 2] vs the normal
    # 5D [L, nb, kv, block_len, d]. Derive the TRUE block_len (2x the stored dim when packed).
    _kv_packed = K_cache.ndim == 6
    _mask_full = None
    _mask_swa = None
    if _hoist_mask:
        _blk_len = K_cache.shape[3] * 2 if _kv_packed else K_cache.shape[3]
        _q_head = sink.shape[1]  # [L, q, 1] -> per-rank q heads
        _kv_heads = K_cache.shape[2]
        _d_head = K_cache.shape[-2] if _kv_packed else K_cache.shape[-1]
        _s_prior_full = active_blocks_table.shape[-1] * _blk_len
        _s_prior_swa = active_blocks_table_swa.shape[-1] * _blk_len
        _qk_swapped_full = is_qk_swapped(
            bs=B_attn,
            q_head=_q_head,
            d_head=_d_head,
            s_active=S_tkg,
            curr_sprior=_s_prior_full,
            lnc=nl.num_programs(0),
            p_max=nl.tile_size.pmax,
            block_len=_blk_len,
            is_2byte_kv=sizeinbytes(K_cache.dtype) == 2,
            fp8_packed=_kv_packed,
            fuse_rope=False,
            kv_heads=_kv_heads,
        )
        _qk_swapped_swa = is_qk_swapped(
            bs=B_attn,
            q_head=_q_head,
            d_head=_d_head,
            s_active=S_tkg,
            curr_sprior=_s_prior_swa,
            lnc=nl.num_programs(0),
            p_max=nl.tile_size.pmax,
            block_len=_blk_len,
            is_2byte_kv=sizeinbytes(K_cache.dtype) == 2,
            fp8_packed=_kv_packed,
            fuse_rope=False,
            kv_heads=_kv_heads,
        )
        _pos_full = pos_ids.reshape((1, B_attn * S_tkg))
        _pos_swa = pos_ids_swa.reshape((1, B_attn * S_tkg))
        # swa_start for the SWA mask is layer-invariant across sliding layers; take
        # any sliding layer's (they're all equal). Use layer 0 (sliding).
        _swa_start0 = swa_start_pos_ids[0].reshape((1, B_attn * S_tkg))
        # active_mask overlay: the small active-token mask the caller already passes.
        # Distinct name_prefix per call: gen_mask_tkg_hbm uses hardcoded op/buffer
        # names, so two calls in one NEFF collide ("duplicate instruction name")
        # without a prefix.
        _mask_full = gen_mask_tkg_hbm(
            _pos_full,
            B_attn,
            _q_head,
            S_tkg,
            _s_prior_full,
            start_pos_hbm=None,
            block_len=_blk_len,
            active_mask=attention_mask,
            transposed_out=_qk_swapped_full,
            name_prefix="megamask_full_",
        )
        _mask_swa = gen_mask_tkg_hbm(
            _pos_swa,
            B_attn,
            _q_head,
            S_tkg,
            _s_prior_swa,
            start_pos_hbm=_swa_start0,
            block_len=_blk_len,
            active_mask=attention_mask,
            transposed_out=_qk_swapped_swa,
            name_prefix="megamask_swa_",
        )

    x = x_shard
    for i in range(num_layers):
        # Real GPT-OSS: even layers slide (trimmed table + window-frame positions),
        # odd layers are full attention (full table + absolute positions). Selecting
        # the trimmed table for sliding layers is what makes them cheaper on device,
        # mirroring the production decode path.
        _sliding = i % 2 == 0
        _abt = active_blocks_table_swa if _sliding else active_blocks_table
        _pos = pos_ids_swa if _sliding else pos_ids
        # Pre-generated mask for this layer (None unless the hoist is on): sliding
        # layers get the SWA mask, full layers the full-attn mask.
        _pregen_mask = (_mask_swa if _sliding else _mask_full) if _hoist_mask else None
        # Per-tensor layer index: ``i`` if stacked per-layer (leading dim ==
        # num_layers), else ``0`` to reuse the single shared copy (leading dim ==
        # 1). Both operands are trace-time Python ints (static shape, scalar arg),
        # so each subscript is plain HBM integer-indexing (dim 0 removed) decided
        # during tracing -- NOT a runtime tensor branch and NOT a nested traced
        # call (the NKI tracer rejects both). Computed inline per tensor (no helper
        # fn, since the tracer forbids direct calls to inner functions).
        li_iln = i if input_layernorm_weight.shape[0] == num_layers else 0
        li_qkv = i if W_qkv.shape[0] == num_layers else 0
        li_bqkv = i if bias_qkv.shape[0] == num_layers else 0
        li_wout = i if W_out.shape[0] == num_layers else 0
        li_bout = i if bias_out.shape[0] == num_layers else 0
        li_sink = i if sink.shape[0] == num_layers else 0
        li_k = i if K_cache.shape[0] == num_layers else 0
        li_v = i if V_cache.shape[0] == num_layers else 0
        li_swa = i if swa_start_pos_ids.shape[0] == num_layers else 0
        li_paln = i if post_attn_layernorm_weight.shape[0] == num_layers else 0
        li_rw = i if router_weights.shape[0] == num_layers else 0
        li_rb = i if router_bias.shape[0] == num_layers else 0
        li_egu = i if expert_gate_up_weights.shape[0] == num_layers else 0
        li_edn = i if expert_down_weights.shape[0] == num_layers else 0
        li_egus = i if expert_gate_up_weights_scale.shape[0] == num_layers else 0
        li_edns = i if expert_down_weights_scale.shape[0] == num_layers else 0
        li_egub = i if expert_gate_up_bias.shape[0] == num_layers else 0
        li_edb = i if expert_down_bias.shape[0] == num_layers else 0
        # Positional call (NKI tracer rejects keyword args to nested functions);
        # order must match _decode_layer_body's signature.
        x = _decode_layer_body(
            x,
            input_layernorm_weight[li_iln],
            W_qkv[li_qkv],
            bias_qkv[li_bqkv],
            W_out[li_wout],
            bias_out[li_bout],
            cos,
            sin,
            sink[li_sink],
            K_cache[li_k],
            V_cache[li_v],
            _abt,
            attention_mask,
            kv_cache_update_idx,
            _pos,
            swa_start_pos_ids[li_swa],
            post_attn_layernorm_weight[li_paln],
            router_weights[li_rw],
            router_bias[li_rb],
            expert_gate_up_weights[li_egu],
            expert_down_weights[li_edn],
            expert_gate_up_weights_scale[li_egus],
            expert_down_weights_scale[li_edns],
            expert_gate_up_bias[li_egub],
            expert_down_bias[li_edb],
            rank_id,
            tp_replica_group,
            ep_replica_group,
            tp_size,
            ep_size,
            B_attn,
            S_tkg,
            top_k,
            eps,
            hidden_actual,
            softmax_scale,
            gate_clamp_upper_limit,
            up_clamp_upper_limit,
            up_clamp_lower_limit,
            f"_L{i}",  # unique per-layer buffer-name suffix
            (i == num_layers - 1),  # is_last_layer: only the final layer spills to HBM
            _pregen_mask,  # hoisted mask (None unless VLLM_NEURON_MEGA_HOIST_MASK)
            k_scale,  # FP8 KV scales (layer-invariant; None unless FP8-KV on)
            v_scale,
            _kv_packed,  # packed fp8 K layout (6D stacked K_cache)
        )
    # KV UPDATE: each layer wrote its token's K/V in-place into K_cache[li_k]/V_cache[li_v]
    # (views of these input tensors), so the full input caches now hold all updates. Returning
    # them makes the compiler emit input<->output aliases (test passes *.must_alias_input), so
    # the golden can compare the post-update caches. Caches MUST be stacked per-layer here.
    if bool(_os.environ.get("VLLM_NEURON_MEGA_KV_UPDATE")):
        return x, K_cache, V_cache
    return x
