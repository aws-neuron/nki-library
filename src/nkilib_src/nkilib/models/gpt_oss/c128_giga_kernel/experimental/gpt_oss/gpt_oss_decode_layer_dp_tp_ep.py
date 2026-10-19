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

from ...core.attention.attention_tkg import _rope
from ...core.attention.attention_tkg_utils import is_qk_swapped
from ...core.attention.gen_mask_tkg import gen_mask_tkg_hbm
from ...core.moe.moe_tkg.down_projection_mx import _MOE_OUT_SBUF_CAPTURE, prefetch_down_bias_sb, prefetch_down_weight_sb
from ...core.moe.moe_tkg.gate_up_projection_mx import prefetch_gate_up_weight_sb
from ...core.moe_block.moe_block_tkg import moe_block_tkg
from ...core.qkv import qkv_tkg as _qkv_tkg_mod
from ...core.router_topk.router_topk import XSBLayout_tp201__2 as _ROUTER_X_SB_LAYOUT
from ...core.router_topk.router_topk import router_topk as _router_topk
from ...core.subkernels import rmsnorm_tkg as _rmsnorm_tkg_mod
from ...core.subkernels.rmsnorm_mx_quantize_tkg import rmsnorm_mx_quantize_tkg as _rmsnorm_mx_quantize_tkg
from ...core.utils.allocator import create_auto_alloc_manager, sizeinbytes
from ...core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    QuantizationType,
    RouterActFnType,
)
from ...core.utils.dma_names import dma_name, set_dma_name_scope
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import mega_flag_default_on as _mega_flag_default_on
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from ..trace_caches import reset_trace_state
from ..transformer.attention_block_tkg import attention_block_tkg

# oob_mode.skip for the fused-embedding gather. Instantiate at module scope
# (not inside the @nki.jit kernel), the same as attention_tkg's OOB_MODE_SKIP.
_OOB_SKIP = nisa.oob_mode.skip

# Buffer class for every HBM tensor that is an OPERAND of a collective. Two incompatible rules
# exist in the wild, and which one an nki build enforces CANNOT be derived from its version:
#   * strict -- collectives/_ops.py::_validate_tensors_buffer_types asserts that every src AND dst
#     share one buffer type ("All src & dst tensors must have the same buffer type"). This is what
#     mainline's pinned nki 0.6.x does.
#   * shared-dst -- at LNC=1 shared_hbm is reserved for collective DESTINATIONS (plus kernel I/O)
#     and a shared src is rejected outright, via
#     _validate_tensors_buffer_types_shared_output_buffers.
#
# Version gating was tried and is WRONG: two builds both calling themselves 0.7.0b1 enforce
# OPPOSITE rules. nki-0.7.0b1+31030513877.g7fadadb6 (the CR dry-run's wheel) is strict -- its
# all_gather calls _validate_tensors_buffer_types -- while the 0.7.0b1 on the DMA-order tuning
# toolchain is shared-dst. A `>= 0.7` test therefore picked private_hbm for the dry-run and failed
# all four test_gpt_oss_decode_layer_dp_tp_ep_golden rank configs on the a_src/a_dst AllGather.
#
# Use private collective operands by default. This satisfies the shared-dst rule because its
# sources are private, and it satisfies strict toolchains because each source/destination pair
# uses the same buffer type. Keep an explicit ``=0`` escape hatch for older reproductions that
# require shared operands.
#
# ⚠️ APPLY THIS TO BOTH ENDS OF A PAIR. All HBM collective sources and destinations use
# _COLLECTIVE_HBM. Plain scratch and kernel I/O are independent of this compatibility switch.
#
# resid1 is deliberately NOT switched: plain per-layer scratch, never a collective operand, so
# private_hbm is legal under both. Kernel I/O (out/mega_out) is forced shared regardless
# ("'nisa.func' op function argument #30 must be a shared_hbm memref").
#
# The 128-rank DMA-order arms already export VLLM_NEURON_MEGA_COLLECTIVE_PRIVATE_SRC=1. Older
# strict-toolchain reproductions can explicitly set it to 0.
_COLLECTIVE_PRIVATE_SRC = _os.environ.get("VLLM_NEURON_MEGA_COLLECTIVE_PRIVATE_SRC", "1") != "0"
_COLLECTIVE_HBM = nl.private_hbm if _COLLECTIVE_PRIVATE_SRC else nl.shared_hbm


def _resolve_fa_s_prior_tiling(seq_packed_attention_mask) -> bool:
    """Resolve FA s_prior tiling: the env default, forced on by packed metadata.

    Banded single-10K softmax: shortening s_prior beats FA s_prior tiling on that
    shape, so DISABLE_FA is part of the measured-optimal bundle and is ON by default.
    Packing structurally requires the tiling -- attention_block_tkg asserts it, because
    the packed kernel has no untiled formulation -- and packing targets 128K, where
    there is no band to shorten. So packed metadata forces it on and the dense
    resolution is untouched: with no packed mask the first disjunct is False and this
    reduces to today's expression exactly.

    One function rather than two copies because both call sites must agree. The mega
    entry builds the hoisted mask from this value while the layer body passes it to
    attention, and a mismatch hands attention a mask of the wrong layout.
    """
    if seq_packed_attention_mask is not None:
        return True
    return not _mega_flag_default_on("VLLM_NEURON_MEGA_DISABLE_FA")


@nki.jit
def gpt_oss_decode_layer_dp_tp_ep_kernel(
    # ── input (SP token shard) ──
    x_shard: nl.NkiTensor,  # [B/(DP*TP), H]
    # ── attention (per-rank TP head shard) ──
    input_layernorm_weight: nl.NkiTensor,  # [1, H] input RMSNorm gamma
    W_qkv: nl.NkiTensor,
    bias_qkv: nl.NkiTensor,
    W_out: nl.NkiTensor,
    bias_out: nl.NkiTensor,
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    sink: nl.NkiTensor,
    K_cache: nl.NkiTensor,
    V_cache: nl.NkiTensor,
    active_blocks_table: nl.NkiTensor,
    attention_mask: nl.NkiTensor,
    kv_cache_update_idx: nl.NkiTensor,
    pos_ids: nl.NkiTensor,
    swa_start_pos_ids: nl.NkiTensor,
    # ── MoE (this rank's 1 MXFP4 expert; router replicated) ──
    post_attn_layernorm_weight: nl.NkiTensor,  # [1, H] MoE RMSNorm gamma
    router_weights: nl.NkiTensor,
    router_bias: nl.NkiTensor,
    expert_gate_up_weights: nl.NkiTensor,
    expert_down_weights: nl.NkiTensor,
    expert_gate_up_weights_scale: nl.NkiTensor,
    expert_down_weights_scale: nl.NkiTensor,
    expert_gate_up_bias: nl.NkiTensor,
    expert_down_bias: nl.NkiTensor,
    rank_id: nl.NkiTensor,
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
    seq_packed_attention_mask: nl.NkiTensor = None,
    seq_packed_active_blocks_table: nl.NkiTensor = None,
    seq_packed_q_index_table: nl.NkiTensor = None,
    seq_packed_seq_id_table: nl.NkiTensor = None,
    seq_packed_accumulator_partial_route_table: nl.NkiTensor = None,
    seq_packed_accumulator_group_state_route_table: nl.NkiTensor = None,
):
    """One rank's full decoder layer (see module docstring). Returns [B/(DP*TP), H].

    The four schedule ``seq_packed_*`` tensors are optional and required as a group;
    supplying them dispatches attention to the sequence-packed kernel. The two
    accumulator route tables are a further all-or-none pair that selects the grouped
    associative accumulator merge over the ordered one.
    """
    reset_trace_state()
    # Positional call (NKI tracer rejects keyword args to nested functions).
    _layer_out = _decode_layer_body(
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
        seq_packed_attention_mask,
        seq_packed_active_blocks_table,
        seq_packed_q_index_table,
        seq_packed_seq_id_table,
        seq_packed_accumulator_partial_route_table,
        seq_packed_accumulator_group_state_route_table,
    )
    # _decode_layer_body sets a per-layer DMA-name scope and deliberately does not clear it (the
    # mega kernel re-scopes on every layer). Clear it at this trace boundary so the scope cannot
    # leak into the NEXT kernel traced in the same process: a leaked scope makes unrelated kernels
    # emit names and trip dma_name()'s duplicate guard -- observed as moe_block_dp_ep failing on a
    # duplicate '_prequant_gamma_load' after this entry ran in the same pytest worker.
    set_dma_name_scope(None)
    return _layer_out


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
        # nisa.dma_copy (not nl.load) so the load can carry a name= for DMA-order pinning.
        rows_sb = nl.ndarray((t_n, H_pack), dtype=rows_hbm.dtype, buffer=nl.sbuf)
        h_split = n_h512 // 2
        q_cut = h_split * 512  # quant bytes per token row of chunks [0, h_split)  (512 fp8 per H512 chunk)
        s_cut = H + h_split * (H // 4 // n_h512)  # scale bytes per token row of chunks [0, h_split)
        for _k, (_c0, _c1, _sfx) in enumerate(
            ((0, q_cut, "p0q"), (H, s_cut, "p0s"), (q_cut, H, "p1q"), (s_cut, H_pack, "p1s"))
        ):
            nisa.dma_copy(
                dst=rows_sb[:, _c0:_c1],
                src=rows_hbm[t0 : t0 + t_n, _c0:_c1],
                name=dma_name(f"prequant_rows_load_t{t0}_{_sfx}"),
            )
        q_src = rows_sb[:, 0:H].view(nl.float32).reshape((t_n, n_h512, H0))
        s_src = rows_sb[:, H:H_pack].view(_u8).reshape((t_n, n_h512, H0))
        _prev_tp = None
        _prev_cp = None
        for h in range(n_h512):
            q_psum = nl.ndarray((H0, t_n), dtype=nl.float32, buffer=nl.psum)
            _tq = nisa.nc_transpose(data=q_src[:, h, :], dst=q_psum)
            _cq = nisa.tensor_copy(dst=quant_sb[:, h, t0 : t0 + t_n].view(nl.float32), src=q_psum)
            s_psum = nl.ndarray((H0, t_n, 2), dtype=_u8, buffer=nl.psum)
            _ts = nisa.nc_transpose(data=s_src[:, h, :], dst=s_psum[:, :, 0])
            _cs = nisa.tensor_copy(dst=scale_sb[:, h, t0 : t0 + t_n].view(_u8), src=s_psum[:, :, 0])
            kernel_assert(
                _tq is not None and _cq is not None and _ts is not None and _cs is not None,
                "nc_transpose / tensor_copy returned no instruction handle",
            )
            if _prev_tp is not None:
                _tq.depends_on(_prev_tp, kind=nisa.dependency.schedule)
            _ts.depends_on(_tq, kind=nisa.dependency.schedule)
            if _prev_cp is not None:
                _cq.depends_on(_prev_cp, kind=nisa.dependency.schedule)
            _cs.depends_on(_cq, kind=nisa.dependency.schedule)
            _prev_tp, _prev_cp = _ts, _cs
    return quant_sb, scale_sb


def _token_rows_to_hidden_sbuf(rows_sb, T, H):
    """Convert token-partitioned [T, H] SBUF rows to QKV's [H0, T, H1] layout."""
    H0 = nl.tile_size.pmax
    H1 = H // H0
    rows_view = rows_sb.reshape((T, H1, H0))
    # PSUM transpose destinations must start on a 4-byte boundary.
    padded_T = ((T * 2 + 3) // 4) * 4 // 2
    hidden_psum = nl.ndarray((H0, H1 * padded_T), dtype=rows_sb.dtype, buffer=nl.psum)
    for h1 in nl.affine_range(H1):
        col_offset = h1 * padded_T
        nisa.nc_transpose(
            data=rows_view[:, h1, :],
            dst=hidden_psum[:, col_offset : col_offset + T],
        )
    hidden_psum_view = hidden_psum.reshape_dim(dim=1, shape=[H1, padded_T]).slice(dim=2, start=0, end=T)
    hidden_sb = nl.ndarray((H0, T, H1), dtype=rows_sb.dtype, buffer=nl.sbuf)
    nisa.tensor_copy(dst=hidden_sb, src=hidden_psum_view.permute([0, 2, 1]))
    return hidden_sb


def _token_rows_to_hidden_psum(rows_sb, T, H, sched_after=None):
    """_token_rows_to_hidden_sbuf, stopped at PSUM: returns the [H0, T, H1] PSUM view.

    For the LATE operand of residual add #1 the PSUM->SBUF copy is dead weight (the add can read
    PSUM directly) and it sits on the serial reduce_scatter -> add -> rmsnorm chain.
    ``sched_after``: every transpose takes a schedule-only dependency on this handle (the flow-gate
    copy), so that copy is issued right at the RS completion, with the first post-RS compute.
    """
    H0 = nl.tile_size.pmax
    H1 = H // H0
    rows_view = rows_sb.reshape((T, H1, H0))
    # PSUM transpose destinations must start on a 4-byte boundary.
    padded_T = ((T * 2 + 3) // 4) * 4 // 2
    hidden_psum = nl.ndarray((H0, H1 * padded_T), dtype=rows_sb.dtype, buffer=nl.psum)
    for h1 in nl.affine_range(H1) if sched_after is None else range(H1):
        col_offset = h1 * padded_T
        _tp = nisa.nc_transpose(
            data=rows_view[:, h1, :],
            dst=hidden_psum[:, col_offset : col_offset + T],
        )
        if sched_after is not None:
            kernel_assert(_tp is not None, "nc_transpose returned no instruction handle")
            _tp.depends_on(sched_after, kind=nisa.dependency.schedule)
    return hidden_psum.reshape_dim(dim=1, shape=[H1, padded_T]).slice(dim=2, start=0, end=T).permute([0, 2, 1])


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
    seq_packed_attention_mask=None,
    seq_packed_active_blocks_table=None,
    seq_packed_q_index_table=None,
    seq_packed_seq_id_table=None,
    seq_packed_accumulator_partial_route_table=None,
    seq_packed_accumulator_group_state_route_table=None,
    layer_tag="",
    is_last_layer=True,
    pregen_mask=None,
    k_scale=None,
    v_scale=None,
    fp8_packed=False,
    x_full=None,
    enable_pv_swapped_column_tiling=False,
    use_swdge_for_wout_load=None,
    wout_load_engine=None,
    order_wout_after_v_prior=False,
    queue_pin_io=None,
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

    ``seq_packed_*``: OPTIONAL sequence-packing metadata, required as a group and
    forwarded verbatim so ``attention_block_tkg`` owns the contract. A packed layer
    also needs a hoisted mask, because packed asserts ``swa_start_pos_ids == None``
    and only the hoist path nulls it; the packed mask carries the window instead.
    The two ``seq_packed_accumulator_*_route_table`` tensors are a separate all-or-none
    pair that selects the grouped associative accumulator merge; both windows share
    them because slot-to-sequence routing does not depend on the window.

    ``layer_tag``: a unique per-layer suffix appended to every named HBM buffer,
    so that when the body is inlined once per layer in the mega kernel the op/
    buffer names stay globally unique (NKI requires unique op names in one NEFF).

    ``x_full``: OPTIONAL already-gathered ``[T_attn, H]`` attention input. Only
    the FIRST layer supplies it: its input is the decode hidden states, which the
    caller holds TP-REPLICATED (identical on every rank of a DP replica), so the
    per-layer AllGather[TP] that reconstructs ``[B/DP, H]`` from an SP shard is
    redundant for layer 0 — the full tensor is already present. When ``x_full`` is
    given, the layer uses it directly as the attention input and SKIPS the
    ``dma_copy(x_shard)`` + ``AllGather``; ``x_shard`` (this rank's ``[T_shard,H]``
    slice) is still used for residual add #1. ``None`` (layers >= 1) keeps the
    AllGather, since those inputs arrive SP-sharded from the prior layer's RS.
    """
    _t = str(layer_tag)
    # Scope every DMA name emitted while tracing this layer, so the subkernels inlined below
    # (attention, qkv, moe, ...) produce stable per-layer instruction names for the DMA-order
    # constraint file instead of positional I-<n> ones. See core/utils/dma_names.py.
    set_dma_name_scope(f"{_t.lstrip('_')}_")
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
    _sbuf_interlayer = _mega_flag_default_on("VLLM_NEURON_MEGA_SBUF_INTERLAYER")
    _moe_sbuf_collective = _mega_flag_default_on("VLLM_NEURON_MEGA_MOE_SBUF_COLLECTIVE")
    # FP8 EP ReduceScatter: RS is ~3.8x slower per byte than pure movement (31.1 vs 118.6 GB/s),
    # so halving the payload is the lever that shortens it. Safe because with 128 experts and
    # top_k=4 each rank's partial is zero for ~124 of the 128 global tokens.
    _fp8_rs = _mega_flag_default_on("VLLM_NEURON_MEGA_FP8_RS")
    # Pre-gather quantize+router: RMSNorm+MX-quantize+router are per-token ops, so running them
    # after the EP AllGather makes all T_ep ranks recompute T_ep identical copies (the swizzle
    # alone moves exactly T*H*2 bytes). Do them for THIS rank's token and gather the results.
    _prequant_moe = _mega_flag_default_on("VLLM_NEURON_MEGA_PREQUANT_MOE")
    # FUSE_RESID2 handoff: set by the fp8-RS branch, consumed by residual add #2. Declared here so
    # every other path falls through to the plain tensor_tensor add.
    _fuse_resid2 = False
    _fuse_rs_src = None
    _fuse_rs_scale = 1.0
    _qpin_prev = queue_pin_io.get("prev_resid2") if queue_pin_io is not None else None
    _rmsnorm_tkg_mod._PIN_AFTER[0] = _qpin_prev
    _qkv_tkg_mod._PIN_AFTER[0] = _qpin_prev
    _prequant_aff16 = _mega_flag_default_on("VLLM_NEURON_MEGA_PREQUANT_AFF16")
    _fp8_rs_scale = float(_os.environ.get("VLLM_NEURON_MEGA_FP8_RS_SCALE", "1") or 1)
    # Attention output stays in SBUF and the TP ReduceScatter runs as a SBUF collective;
    # residual add #1 stays in SBUF. Prequant MoE can consume that result directly.
    _sbuf_attn_rs = _mega_flag_default_on("VLLM_NEURON_MEGA_SBUF_ATTN_RS")
    # Convert this rank's token rows to QKV's hidden-partitioned SBUF layout, gather
    # along the free axis, and pass the gathered SBUF tensor directly to attention.
    # Keep this default-off: the row-to-hidden Tensor Engine transposes currently
    # offset the scheduling benefit of removing the HBM collective staging path.
    _sbuf_attn_ag = bool(_os.environ.get("VLLM_NEURON_MEGA_SBUF_ATTN_AG"))
    # Also measured negative; kept selectable so the negative result stays reproducible.
    _keep_resid1_in_sbuf = _sbuf_attn_rs and _prequant_moe and bool(_os.environ.get("VLLM_NEURON_MEGA_SBUF_RESID1"))
    # H0-MAJOR RESIDUAL ADD #1 (VLLM_NEURON_MEGA_H0_RESID1=1).
    #
    # At T_shard=1 resid1_sb is ONE SBUF partition, so the stock path runs the add 1-of-128
    # partitions wide and then pays rmsnorm's own H1=24 [1,128] transposes to widen it -- a serial
    # prologue (add -> transposes -> rmsnorm -> router) with nothing overlapping it, 87.7 us exposed.
    # The 1->128 expansion is unavoidable (the TP ReduceScatter scatters on dim 0, which for an SBUF
    # collective is the partition dim) but it need not be paid on the sum. Instead transpose the two
    # operands separately: x_shard is ready at layer entry so its transpose hoists above attention,
    # only the late operand is transposed on the critical path, and the add itself runs H0-major.
    # The token-major sum is still needed by residual add #2, so it is emitted after the router.
    #
    # ⚠ Do NOT combine with SBUF_RESID1: the token-major resid1 then stays resident across the MoE,
    # reserving 6144 B of every partition's free axis, which evicts a prefetched gate/up buffer
    # (gate_up input DMA 174.6 -> 335.0 MB, ~+50 us exposed).
    _h0_resid1 = _sbuf_attn_rs and _prequant_moe and bool(_os.environ.get("VLLM_NEURON_MEGA_H0_RESID1"))
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
    _kv_update = _mega_flag_default_on("VLLM_NEURON_MEGA_KV_UPDATE")
    _enable_fa_s_prior_tiling = _resolve_fa_s_prior_tiling(seq_packed_attention_mask)
    _x_in_sbuf = _sbuf_interlayer and (getattr(x_shard, "buffer", None) == nl.sbuf)

    # Per-layer SBUF manager with a unique name prefix so attention_block_tkg's
    # internal named buffers stay globally unique when inlined once per layer.
    _sbm = create_auto_alloc_manager()
    _sbm.set_name_prefix(f"attn{_t}_")

    # Prefetch this layer's MoE gate/up weight into SBUF before attention so the
    # ~9 MB HBM->SBUF DMA overlaps the attention matmuls instead of stalling the
    # first MoE gate matmul; moe_block_tkg's leaf loader pops the enqueued SBUF
    # buffer (FIFO per gate/up idx) instead of alloc+DMA. Tracer frontend only.
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
    if _mega_flag_default_on("VLLM_NEURON_MEGA_DOWN_PREFETCH"):
        prefetch_down_weight_sb(
            weight=expert_down_weights,
            expert_idx=0,
            H=H,
            tile_I=nl.tile_size.pmax,
            n_I512_tiles=expert_down_weights.shape[2],
            tile_offset=0,
            name="down_prefetch" + _t,
        )

    # HOISTED half of residual add #1 (see _h0_resid1 above): x_shard is this layer's input, i.e.
    # already resident at layer entry, so converting it to [H0, T_shard, H1] here lets those 24
    # nc_transposes overlap the entire attention block instead of joining the serial prologue.
    _x_sb_pre = None
    _x_h0_sb = None
    if _h0_resid1:
        if _x_in_sbuf:
            _x_sb_pre = x_shard
        else:
            # nisa.dma_copy, not nl.load, so the hoisted load carries a stable U- name. An unnamed
            # load gets a positional one (I-185 in the first derive of this build) that renumbers
            # whenever any earlier DMA moves -- the exact fragility this package's dma_name scopes
            # exist to remove. Same name as the non-hoisted site below, so one schedule binds either.
            _x_sb_pre = nl.ndarray((T_shard, H), dtype=x_shard.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=_x_sb_pre, src=x_shard, name=dma_name("x_shard_load"))
        _x_h0_sb = _token_rows_to_hidden_sbuf(_x_sb_pre, T_shard, H)

    # ════════════════ Attention (input RMSNorm fused inside) ════════════════
    # Reconstruct this replica's full [T_attn, H] for QKV. Layer 0 passes the
    # already-gathered (TP-replicated) input as x_full, so skip the dma_copy +
    # AllGather[TP]; layers >= 1 arrive SP-sharded and AllGather to rebuild it.
    if x_full is not None:
        X = x_full.reshape((B_attn, S_tkg, H))
    elif _sbuf_attn_ag:
        # Gather along the free axis straight out of SBUF, dropping the HBM staging copy.
        H0 = nl.tile_size.pmax
        H1 = H // H0
        x_rows_sb = x_shard if _x_in_sbuf else nl.load(x_shard)
        a_src_sb = _token_rows_to_hidden_sbuf(x_rows_sb, T_shard, H)
        a_dst_sb = nl.ndarray((H0, T_attn * H1), dtype=dtype, buffer=nl.sbuf)
        ncc.all_gather(
            dsts=[a_dst_sb],
            srcs=[a_src_sb.reshape((H0, T_shard * H1))],
            replica_group=tp_replica_group,
            collective_dim=1,
        )
        X = a_dst_sb.reshape((H0, T_attn, H1))
    else:
        # HBM collective operands use _COLLECTIVE_HBM; see the compatibility note above.
        a_src = nl.ndarray((T_shard, H), dtype=dtype, buffer=_COLLECTIVE_HBM, name="a_src" + _t)
        a_dst = nl.ndarray((T_attn, H), dtype=dtype, buffer=_COLLECTIVE_HBM, name="a_dst" + _t)
        nisa.dma_copy(dst=a_src, src=x_shard, name=dma_name("attn_ag_src_store"))
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
        use_swdge_for_out_proj_weight_load=use_swdge_for_wout_load,
        out_proj_weight_load_engine=wout_load_engine,
        order_out_proj_after_v_prior_load=order_wout_after_v_prior,
        queue_pin_attn=True,
        queue_pin_early_after=_qpin_prev,
        softmax_scale=softmax_scale,
        enable_fa_s_prior_tiling=_enable_fa_s_prior_tiling,
        enable_pv_moving_free_dim_padding=False,
        enable_pv_swapped_column_tiling=enable_pv_swapped_column_tiling,
        pos_ids=None if pregen_mask is not None else pos_ids,
        swa_start_pos_ids=None if pregen_mask is not None else swa_start_pos_ids,
        S_tkg=S_tkg,
        # SEQUENCE PACKING: forwarded verbatim, so attention_block_tkg's own validation
        # owns the contract. Packed carries its window in seq_packed_attention_mask and
        # asserts swa_start_pos_ids == None, so a packed layer needs the hoisted mask
        # (which nulls it) -- deliberately an assert rather than a silent override here.
        seq_packed_attention_mask=seq_packed_attention_mask,
        seq_packed_active_blocks_table=seq_packed_active_blocks_table,
        seq_packed_q_index_table=seq_packed_q_index_table,
        seq_packed_seq_id_table=seq_packed_seq_id_table,
        seq_packed_accumulator_partial_route_table=seq_packed_accumulator_partial_route_table,
        seq_packed_accumulator_group_state_route_table=seq_packed_accumulator_group_state_route_table,
        # FP8 KV cache: raw fp8 K/V + scales; softmax_scale is pre-fused with k_scale
        # and W_out pre-fused with v_scale by the caller. None (unset) => bf16/fp16 KV.
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_packed=fp8_packed,  # packed K [nb, bl//2, d, 2] -> DMA-gather-transpose path
        sbm=_sbm,  # per-layer name prefix -> unique attention buffer names
    )  # [B/DP, H] per-rank O-proj partial
    # H0_RESID1 with resid1 resident in SBUF: residual add #1 happens in the H0 domain and the
    # token-major sum is emitted later (after the router), so it is not produced here.
    _no_token_resid1 = _h0_resid1 and _keep_resid1_in_sbuf
    resid1 = (
        nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.private_hbm, name="resid1" + _t)
        if not _keep_resid1_in_sbuf
        else None
    )
    _resid1_h0_sb = None
    if _sbuf_attn_rs:
        # PROTOTYPE (tracer only): attention_block_tkg emitted its [T_attn, H] O-proj partial
        # straight to SBUF (out_in_sb=True). Feed it to reduce_scatter as a SBUF collective
        # (src SBUF + dst SBUF, collective_dim=0), dropping the ars_src HBM staging copy.
        # Residual add #1 then runs fully in SBUF. The experiment can extend its lifetime.
        attn_shard_sb = nl.ndarray((T_shard, H), dtype=attn_out.dtype, buffer=nl.sbuf)
        ncc.reduce_scatter(
            dsts=[attn_shard_sb], srcs=[attn_out], op=nl.add, replica_group=tp_replica_group, collective_dim=0
        )
        # GU_RS_FLOWGATE (VLLM_NEURON_MEGA_GU_RS_FLOWGATE=1): an SBUF->SBUF copy between the
        # attention TP ReduceScatter and its consumers, so the expert gate/up prefetch can be given
        # a COMPLETION dependency on that collective.
        #
        # The pinned DMA-order JSON's `Ordered` edge orders against a collective's ISSUE, not its
        # completion, so any placement of gup_prefetch_0 above resid1 runs INSIDE the RS. Only a DMA
        # predecessor is completion-ordered, and its result must be READ or DCE removes it.
        # Costs 5.76 KB/layer to gate a 2 MB/layer prefetch.
        if _os.environ.get("VLLM_NEURON_MEGA_GU_RS_FLOWGATE"):
            _gate_sb = nl.ndarray((T_shard, H), dtype=attn_out.dtype, buffer=nl.sbuf, name="gugate" + _t)
            _gate_dma = nisa.dma_copy(dst=_gate_sb, src=attn_shard_sb, name=dma_name("gugate"))
            kernel_assert(_gate_dma is not None, "dma_copy returned no instruction handle")
            # The copy needs exactly ONE reader: unread, DCE drops it and the prefetch loses its
            # completion edge; read by every consumer, it stays live to the end of the post-attention
            # region and collides with the prequant PSUM->SBUF copy (measured worse). The token-major
            # resid1 add is that reader.
            _h0_gate_src = _gate_sb
        else:
            _h0_gate_src = None
            _gate_dma = None
        if _x_sb_pre is not None:
            # H0_RESID1 already loaded x_shard to build its hoisted transpose; reuse that tile.
            x_sb = _x_sb_pre
        elif _x_in_sbuf:
            x_sb = x_shard
        else:
            # nisa.dma_copy (not nl.load) so the load can carry a name= for DMA-order pinning.
            x_sb = nl.ndarray((T_shard, H), dtype=x_shard.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb, src=x_shard, name=dma_name("x_shard_load"))
        resid1_sb = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.sbuf)
        if _h0_resid1:
            # Transpose only the late operand (the RS output) and add H0-major: the same 24
            # nc_transposes rmsnorm used to do on the sum, but feeding a 128-partition-wide add,
            # and read straight out of PSUM. Ordered after the flow-gate copy.
            _attn_h0_psum = _token_rows_to_hidden_psum(attn_shard_sb, T_shard, H, sched_after=_gate_dma)
            _resid1_h0_sb = nl.ndarray(
                (nl.tile_size.pmax, T_shard, H // nl.tile_size.pmax), dtype=dtype, buffer=nl.sbuf
            )
            nisa.tensor_tensor(dst=_resid1_h0_sb, data1=_attn_h0_psum, data2=_x_h0_sb, op=nl.add)
        # Token-major sum. Sinks past the router only when resid1 stays in SBUF: in HBM the
        # store must precede the rmsnorm call, whose `input=` is a view of that tensor.
        # When the flow gate is on, read the RS output through the gate copy here, so the gate has
        # its one reader and the expert prefetch keeps its completion edge on the RS.
        _resid1_add_src = attn_shard_sb
        if _h0_gate_src is not None:
            _resid1_add_src = _h0_gate_src
        if not _no_token_resid1:
            nisa.tensor_tensor(dst=resid1_sb, data1=x_sb, data2=_resid1_add_src, op=nl.add)
            if not _keep_resid1_in_sbuf:
                nisa.dma_copy(dst=resid1, src=resid1_sb, name=dma_name("resid1_store"))
    else:
        ars_src = nl.ndarray((T_attn, H), dtype=attn_out.dtype, buffer=_COLLECTIVE_HBM, name="ars_src" + _t)
        attn_shard = nl.ndarray((T_shard, H), dtype=attn_out.dtype, buffer=_COLLECTIVE_HBM, name="attn_shard" + _t)
        nisa.dma_copy(dst=ars_src, src=attn_out, name=dma_name("attn_rs_src_store"))
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
    m_src = nl.ndarray((T_shard, H), dtype=dtype, buffer=_COLLECTIVE_HBM, name="m_src" + _t)
    m_dst = nl.ndarray((T_ep, H), dtype=dtype, buffer=_COLLECTIVE_HBM, name="m_dst" + _t)
    _pq_quant = None
    _pq_scale = None
    _pq_affin = None
    if not _prequant_moe:
        nisa.dma_copy(dst=m_src, src=resid1, name=dma_name("moe_ag_src_store"))
        ncc.all_gather(dsts=[m_dst], srcs=[m_src], replica_group=ep_replica_group, collective_dim=0)
    moe_in = m_dst.reshape((1, T_ep, H))

    if _prequant_moe:
        if _no_token_resid1:
            # hidden_h0_sb carries the hidden state, so `input` is read only for shape/dtype.
            # It has to be x_sb, not resid1_sb: the token-major sum is emitted after this call.
            pq_input = x_sb.reshape((T_shard, 1, H))
        else:
            pq_input = resid1_sb.reshape((T_shard, 1, H)) if _keep_resid1_in_sbuf else resid1.reshape((1, T_shard, H))
        H_pack = H + H // 4
        E_total = router_weights.shape[-1]
        # Dense buffers on BOTH ends: a collective cannot read/write a sub-range of a wider
        # tensor ("Output pattern is not contiguous"), and feeding a gather from a bitcast slice
        # of a shared row silently corrupts the payload (rank0 fine, all peers garbage).
        pq_src = nl.ndarray((T_shard, H_pack), dtype=nl.float8_e4m3fn, buffer=_COLLECTIVE_HBM, name="pq_src" + _t)
        pq_norm_sb = nl.ndarray((nl.tile_size.pmax, T_shard, H // nl.tile_size.pmax), dtype=nl.float16, buffer=nl.sbuf)
        # Packed layout is selected by output_quant.shape == (T, H+H/4) with output_scale=None.
        _rmsnorm_mx_quantize_tkg(
            input=pq_input,
            gamma=post_attn_layernorm_weight,
            output=pq_norm_sb,
            output_quant=pq_src,
            output_scale=None,
            eps=eps,
            hidden_actual=hidden_actual,
            hidden_dim_tp=True,
            # Residual add #1 already produced the H0-partitioned hidden state, in SBUF, so
            # rmsnorm skips its own H1 [1,128] nc_transposes.
            hidden_h0_sb=_resid1_h0_sb,
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
        aff_src = nl.ndarray((T_shard, E_total), dtype=_aff_dt, buffer=_COLLECTIVE_HBM, name="aff_src" + _t)
        eidx_sb = nl.ndarray((T_shard, top_k), dtype=nl.uint32, buffer=nl.sbuf)
        _qd_shared = queue_pin_io.get("quant_dst_shared") if queue_pin_io is not None else None
        if queue_pin_io is not None and _qd_shared is None:
            _qd_shared = nl.ndarray(
                (T_ep, H_pack), dtype=nl.float8_e4m3fn, buffer=_COLLECTIVE_HBM, name="quant_dst_shared"
            )
            queue_pin_io["quant_dst_shared"] = _qd_shared
        if _qd_shared is not None:
            quant_dst = _qd_shared
        else:
            quant_dst = nl.ndarray(
                (T_ep, H_pack), dtype=nl.float8_e4m3fn, buffer=_COLLECTIVE_HBM, name="quant_dst" + _t
            )
        _rmsnorm_tkg_mod._PIN_AFTER[0] = None
        _qkv_tkg_mod._PIN_AFTER[0] = None
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
        if _h0_resid1 and _keep_resid1_in_sbuf:
            # Token-major resid1, needed only by residual add #2. Emitted after the router so the
            # 1-partition-wide add lands in the MoE shadow rather than the rmsnorm prologue;
            # nothing between here and residual #2 reads it, so the scheduler can sink it.
            nisa.tensor_tensor(dst=resid1_sb, data1=x_sb, data2=attn_shard_sb, op=nl.add)
        # Two separate gathers. Fusing them into one wider collective was measured twice and
        # regressed (+0.15 ms): the collective it removes is the already-hidden 256 B affinity
        # gather (10.7 of 12.5 us overlapped), while the local DMAs that build the fused row add
        # unhidden serial work. Issuing the quant gather before the router also lost.
        aff_dst = nl.ndarray((T_ep, E_total), dtype=_aff_dt, buffer=_COLLECTIVE_HBM, name="aff_dst" + _t)
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
        prefetch_down_bias_sb(
            bias=expert_down_bias,
            expert_idx=0,
            H=H,
            tile_T=T_ep,
            activation_compute_dtype=nl.bfloat16,  # moe_block_tkg's activation_compute_dtype default; asserted by the loader
        )
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
    moe_shard = nl.ndarray((T_shard, H), dtype=moe_out.dtype, buffer=_COLLECTIVE_HBM, name="moe_shard" + _t)
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
            if moe_out_sb.dtype == _rs_dt:
                # MOE_OUT_FP8: down_projection_mx already wrote the affinity-scaled result as fp8, so the
                # collective reads it directly (no [T_ep, H] bf16->fp8 pass on the moe_rs critical path).
                kernel_assert(_fp8_rs_scale == 1, "MOE_OUT_FP8 requires FP8_RS_SCALE == 1")
                rs_src = moe_out_sb
            else:
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
            _fuse_resid2 = _mega_flag_default_on("VLLM_NEURON_MEGA_FUSE_RESID2")
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
        mrs_src = nl.ndarray((T_ep, H), dtype=moe_out.dtype, buffer=_COLLECTIVE_HBM, name="mrs_src" + _t)
        nisa.dma_copy(dst=mrs_src, src=moe_out, name=dma_name("moe_rs_src_store"))
        ncc.reduce_scatter(
            dsts=[moe_shard], srcs=[mrs_src], op=nl.add, replica_group=ep_replica_group, collective_dim=0
        )

    # ── residual add #2 -> layer output [B/(DP*TP), H] ──
    # Reuse resid1 from SBUF when available; otherwise load it from HBM.
    out = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.shared_hbm, name="mega_out" + _t)
    out_sb = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.sbuf)
    # nisa.dma_copy (not nl.load/nl.store) on these paths so each can carry a name= for
    # DMA-order pinning; nl.load/nl.store take no name.
    if _keep_resid1_in_sbuf:
        r_sb = resid1_sb
    else:
        r_sb = nl.ndarray((T_shard, H), dtype=resid1.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=r_sb, src=resid1, name=dma_name("resid1_load"))
    # residual1 add MoE: source the MoE shard from SBUF when we kept it resident
    # (inter-layer), else reload from HBM. tensor_tensor casts moe_shard_sb (bf16) to
    # out_sb dtype (fp16) implicitly.
    if _sbuf_interlayer and moe_shard_sb is not None:
        mo_sb = moe_shard_sb
    else:
        mo_sb = nl.ndarray(moe_shard.shape, dtype=moe_shard.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=mo_sb, src=moe_shard, name=dma_name("moe_shard_load"))
    if _fuse_resid2 and _fuse_rs_src is not None and _fuse_rs_scale == 1.0:
        _r2 = nisa.tensor_tensor(dst=out_sb, data1=r_sb, data2=_fuse_rs_src, op=nl.add)
    elif _fuse_resid2 and _fuse_rs_src is not None:
        # One pass: (rs_dst * 1/scale) + resid1 -> out_sb, replacing dequant + add.
        _r2 = nisa.scalar_tensor_tensor(
            dst=out_sb, data=_fuse_rs_src, op0=nl.multiply, operand0=_fuse_rs_scale, op1=nl.add, operand1=r_sb
        )
    else:
        _r2 = nisa.tensor_tensor(dst=out_sb, data1=r_sb, data2=mo_sb, op=nl.add)
    if queue_pin_io is not None:
        kernel_assert(_r2 is not None, "residual add #2 returned no instruction handle")
        queue_pin_io["prev_resid2"] = _r2
    # Inter-layer SBUF: hand the output to the next layer straight from SBUF; store to
    # HBM only on the final layer (the @nki.jit kernel must return an HBM tensor).
    if _sbuf_interlayer and not is_last_layer:
        return out_sb
    nisa.dma_copy(dst=out, src=out_sb, name=dma_name("mega_out_store"))
    return out


@nki.jit
def gpt_oss_decode_mega_kernel(
    # ── input (SP token shard) ──
    x_shard: nl.NkiTensor,  # [B/(DP*TP), H]
    # ── per-layer weights: STACKED on a leading layer axis [num_layers, ...] ──
    # (NKI rejects Python-list kernel args; integer-indexing an HBM tensor removes
    #  dim 0, so W[i] yields that layer's tensor in the exact per-layer shape.)
    input_layernorm_weight: nl.NkiTensor,  # [L, 1, H]
    W_qkv: nl.NkiTensor,  # [L, H, (q + 2*kv)*d]
    bias_qkv: nl.NkiTensor,  # [L, 1, (q + 2*kv)*d]
    W_out: nl.NkiTensor,  # [L, q*d, H]
    bias_out: nl.NkiTensor,  # [L, 1, H]
    sink: nl.NkiTensor,  # [L, q, 1]
    K_cache: nl.NkiTensor,  # [L, nb, kv, block_len, d]
    V_cache: nl.NkiTensor,  # [L, nb, kv, block_len, d]
    post_attn_layernorm_weight: nl.NkiTensor,  # [L, 1, H]
    router_weights: nl.NkiTensor,  # [L, H, E]
    router_bias: nl.NkiTensor,  # [L, 1, E]
    expert_gate_up_weights: nl.NkiTensor,  # [L, ...] this rank's expert / layer
    expert_down_weights: nl.NkiTensor,  # [L, ...]
    expert_gate_up_weights_scale: nl.NkiTensor,  # [L, ...]
    expert_down_weights_scale: nl.NkiTensor,  # [L, ...]
    expert_gate_up_bias: nl.NkiTensor,  # [L, ...]
    expert_down_bias: nl.NkiTensor,  # [L, ...]
    swa_start_pos_ids: nl.NkiTensor,  # [L, B_attn, S_tkg] per-layer SWA starts
    #   (full layers = 0 -> window [0,pos] = full causal)
    # ── shared attention control (same absolute positions + active mask) ──
    cos: nl.NkiTensor,
    sin: nl.NkiTensor,
    active_blocks_table: nl.NkiTensor,  # FULL block table [B_attn, nbf]; used by full (odd) layers
    active_blocks_table_swa: nl.NkiTensor,  # TRIMMED block table [B_attn, num_swa]; used by sliding (even) layers
    attention_mask: nl.NkiTensor,  # active-token mask; identical for SWA/full layers
    kv_cache_update_idx: nl.NkiTensor,
    pos_ids: nl.NkiTensor,  # ABSOLUTE positions [B_attn, S_tkg]; used by full (odd) layers
    pos_ids_swa: nl.NkiTensor,  # positions SHIFTED into the trimmed frame; used by sliding (even) layers
    rank_id: nl.NkiTensor,
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
    k_scale: nl.NkiTensor = None,
    v_scale: nl.NkiTensor = None,
    # SWA-group KV write index: paired SWA/full layers share a buffer but get
    # disjoint block IDs, so sliding layers write with this and full layers with
    # kv_cache_update_idx. None => both use kv_cache_update_idx (golden path).
    kv_cache_update_idx_swa: nl.NkiTensor = None,
    # Layer-0 pre-gathered (TP-replicated) attention input, letting layer 0 skip
    # its redundant input AllGather. None => layer 0 AllGathers x_shard (golden).
    x_first_full: nl.NkiTensor = None,
    # HOST GEN-MASK: pre-generated full-attn + SWA masks built on the host and
    # DMA-loaded here, replacing the on-chip gen_mask_tkg. When provided (not
    # None), on-chip mask generation is skipped. Each must match the layout that
    # gen_mask_tkg_hbm would produce for its s_prior (see is_qk_swapped).
    pregen_mask_full: nl.NkiTensor = None,
    pregen_mask_swa: nl.NkiTensor = None,
    # FUSED EMBEDDING: build layer 0's input in-NEFF from the embedding weight, so
    # the host does not compute the hidden and it never spills to HBM. Two shardings,
    # auto-selected by embed_weight's last dim:
    #   (a) VOCAB-dim  -- embed_weight [vocab_per_rank, H]; embed_ids are LOCAL vocab
    #       indices (-1 for a token outside this rank's shard). Gather full-H rows,
    #       then reduce_scatter[TP] + all_gather[TP] (a cross-rank sum).
    #   (b) HIDDEN-dim -- embed_weight [vocab, H/tp_size]; embed_ids are GLOBAL token
    #       ids. Gather this rank's H/tp_size slice for all tokens, then ONE SBUF
    #       all_gather along the hidden dim (no cross-rank sum). embed_x_shard_ids
    #       (this rank's token rows) recovers the SP shard for layer 0's residual.
    # None (default) keeps the external x_shard / x_first_full path.
    embed_weight: nl.NkiTensor = None,
    embed_ids: nl.NkiTensor = None,
    embed_x_shard_ids: nl.NkiTensor = None,
    # FUSED ROPE: this model's inverse frequencies [d_head//2, 1] and the RoPE
    # concentration scalar. When inv_freq is provided, the kernel computes the
    # cos/sin tables ON-CHIP from pos_ids + inv_freq (reusing the _rope helper),
    # replacing the external cos/sin inputs. None (default) uses cos/sin as passed.
    inv_freq: nl.NkiTensor = None,
    rope_concentration: float = 1.0,
    # -- HMA individual KV-cache buffers (bw-compatible alt to stacked K_cache/V_cache) --
    # Under HMA the caller registers group_size = num_layers//2 SEPARATE combined
    # full+SWA block pools instead of one stacked [L, ...] tensor. Pass those pools here
    # (each per-buffer [nb, kv, block_len, d], or packed [nb, kv, block_len//2, d, 2]) and
    # set the stacked K_cache/V_cache to None. Used ONLY when stacked K_cache is None.
    # Per-layer routing mirrors the stacked leading-dim logic by the count of provided
    # buffers: count == num_layers -> per-layer (i); count == 1 -> shared (0); else the
    # HMA pairing where SWA layer 2j and full layer 2j+1 share buffer j == i//2.
    # These sit LAST in the signature and the caller passes them by KEYWORD, so adding
    # them shifts no existing positional argument.
    K_cache_0: nl.NkiTensor = None,
    K_cache_1: nl.NkiTensor = None,
    K_cache_2: nl.NkiTensor = None,
    K_cache_3: nl.NkiTensor = None,
    K_cache_4: nl.NkiTensor = None,
    K_cache_5: nl.NkiTensor = None,
    K_cache_6: nl.NkiTensor = None,
    K_cache_7: nl.NkiTensor = None,
    K_cache_8: nl.NkiTensor = None,
    K_cache_9: nl.NkiTensor = None,
    K_cache_10: nl.NkiTensor = None,
    K_cache_11: nl.NkiTensor = None,
    K_cache_12: nl.NkiTensor = None,
    K_cache_13: nl.NkiTensor = None,
    K_cache_14: nl.NkiTensor = None,
    K_cache_15: nl.NkiTensor = None,
    K_cache_16: nl.NkiTensor = None,
    K_cache_17: nl.NkiTensor = None,
    V_cache_0: nl.NkiTensor = None,
    V_cache_1: nl.NkiTensor = None,
    V_cache_2: nl.NkiTensor = None,
    V_cache_3: nl.NkiTensor = None,
    V_cache_4: nl.NkiTensor = None,
    V_cache_5: nl.NkiTensor = None,
    V_cache_6: nl.NkiTensor = None,
    V_cache_7: nl.NkiTensor = None,
    V_cache_8: nl.NkiTensor = None,
    V_cache_9: nl.NkiTensor = None,
    V_cache_10: nl.NkiTensor = None,
    V_cache_11: nl.NkiTensor = None,
    V_cache_12: nl.NkiTensor = None,
    V_cache_13: nl.NkiTensor = None,
    V_cache_14: nl.NkiTensor = None,
    V_cache_15: nl.NkiTensor = None,
    V_cache_16: nl.NkiTensor = None,
    V_cache_17: nl.NkiTensor = None,
    # ── optional sequence packing (all None => dense attention on every layer) ──
    # Split swa/full like active_blocks_table: a packed mask and its block routes
    # are window-specific, the q-index and seq-id routes are not. Both _swa
    # variants are REQUIRED when packed is on -- this stack always alternates.
    seq_packed_attention_mask: nl.NkiTensor = None,
    seq_packed_attention_mask_swa: nl.NkiTensor = None,
    seq_packed_active_blocks_table: nl.NkiTensor = None,
    seq_packed_active_blocks_table_swa: nl.NkiTensor = None,
    seq_packed_q_index_table: nl.NkiTensor = None,
    seq_packed_seq_id_table: nl.NkiTensor = None,
    # Accumulator routes are an all-or-none pair and, like the q-index and seq-id
    # routes, are window-invariant: they encode slot-to-sequence grouping only.
    seq_packed_accumulator_partial_route_table: nl.NkiTensor = None,
    seq_packed_accumulator_group_state_route_table: nl.NkiTensor = None,
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
    reset_trace_state()
    # Name the prologue's DMAs (mask hoist, block tables) under their own scope; each layer
    # re-scopes on entry. Cleared before returning so a kernel traced later in this process
    # is unaffected. See core/utils/dma_names.py.
    set_dma_name_scope("pro_")

    # -- HMA individual KV buffers vs stacked K_cache/V_cache (see signature) --
    # Gather the optional per-buffer params into Python tuples (trace-time only) and
    # decide, once, whether to use them. Used ONLY when the stacked K_cache is None,
    # keeping the original stacked interface fully backward compatible.
    _k_indiv = (
        K_cache_0,
        K_cache_1,
        K_cache_2,
        K_cache_3,
        K_cache_4,
        K_cache_5,
        K_cache_6,
        K_cache_7,
        K_cache_8,
        K_cache_9,
        K_cache_10,
        K_cache_11,
        K_cache_12,
        K_cache_13,
        K_cache_14,
        K_cache_15,
        K_cache_16,
        K_cache_17,
    )
    _v_indiv = (
        V_cache_0,
        V_cache_1,
        V_cache_2,
        V_cache_3,
        V_cache_4,
        V_cache_5,
        V_cache_6,
        V_cache_7,
        V_cache_8,
        V_cache_9,
        V_cache_10,
        V_cache_11,
        V_cache_12,
        V_cache_13,
        V_cache_14,
        V_cache_15,
        V_cache_16,
        V_cache_17,
    )
    # Exactly one KV form is provided: stacked K_cache OR per-buffer K_cache_0.
    # If the stacked K_cache is None, use the per-buffer form, and vice versa.
    kernel_assert(
        (K_cache is None) != (K_cache_0 is None),
        "GK decode needs exactly one KV form: pass either the stacked K_cache or "
        "the per-buffer K_cache_0.., with the other None.",
    )
    _use_indiv_kv = K_cache is None
    _n_kv = sum(1 for t in _k_indiv if t is not None)  # provided-buffer count (trace-time)

    # This stack alternates sliding/full unconditionally (``_sliding = i % 2 == 0``), so
    # there is no uniform-window case: a packed sliding layer whose _swa metadata is
    # missing would attend over the full context instead of its window, silently.
    if seq_packed_attention_mask is not None:
        kernel_assert(
            seq_packed_attention_mask_swa is not None and seq_packed_active_blocks_table_swa is not None,
            "sequence-packed decode requires both seq_packed_attention_mask_swa and "
            "seq_packed_active_blocks_table_swa: sliding layers carry their window inside the packed mask",
        )

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
    # HOST GEN-MASK: if the host supplied the masks, use them directly and skip
    # the on-chip gen_mask_tkg. Otherwise fall back to the env-gated on-chip hoist.
    _host_mask = pregen_mask_full is not None
    _hoist_mask = _host_mask or _mega_flag_default_on("VLLM_NEURON_MEGA_HOIST_MASK")
    # Same resolution the layer body uses, or the hoisted mask layout will not match the
    # shape attention expects: a packed layer consumes this mask (it is what nulls
    # swa_start_pos_ids for it).
    _enable_fa_s_prior_tiling = _resolve_fa_s_prior_tiling(seq_packed_attention_mask)
    # Packed fp8 KV: a per-buffer K is 5D [nb, kv, block_len//2, d, 2] (packed) vs 4D
    # [nb, kv, block_len, d]; the stacked form carries an extra leading layer axis
    # (6D/5D). Derive the TRUE block_len (2x the stored dim when packed). Individual
    # buffers have no leading axis, so the shape indices shift down by one.
    if _use_indiv_kv:
        _kv_packed = K_cache_0.ndim == 5
        _kv_blk_stored = K_cache_0.shape[2]
        _kv_heads_cnt = K_cache_0.shape[1]
        _kv_d_head = K_cache_0.shape[-2] if _kv_packed else K_cache_0.shape[-1]
        _kv_dtype = K_cache_0.dtype
    else:
        _kv_packed = K_cache.ndim == 6
        _kv_blk_stored = K_cache.shape[3]
        _kv_heads_cnt = K_cache.shape[2]
        _kv_d_head = K_cache.shape[-2] if _kv_packed else K_cache.shape[-1]
        _kv_dtype = K_cache.dtype

    _mask_full = None
    _mask_swa = None
    if _host_mask:
        # Host supplies both masks (generation moved off device). The host mask
        # is byte-identical to gen_mask_tkg_hbm, so the load path is the same as
        # the on-chip hoist below. If the host supplies only the full mask, the
        # SWA layers fall back to in-kernel generation (see the per-layer select).
        _mask_full = pregen_mask_full
        _mask_swa = pregen_mask_swa
    elif _hoist_mask:
        _blk_len = _kv_blk_stored * 2 if _kv_packed else _kv_blk_stored
        _q_head = sink.shape[1]  # [L, q, 1] -> per-rank q heads
        _kv_heads = _kv_heads_cnt
        _d_head = _kv_d_head
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
            is_2byte_kv=sizeinbytes(_kv_dtype) == 2,
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
            is_2byte_kv=sizeinbytes(_kv_dtype) == 2,
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
            enable_fa_s_prior_tiling=_enable_fa_s_prior_tiling,
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
            enable_fa_s_prior_tiling=_enable_fa_s_prior_tiling,
            transposed_out=_qk_swapped_swa,
            name_prefix="megamask_swa_",
        )

    # ── FUSED EMBEDDING PROLOGUE (env-gated by the caller) ───────────────────
    # When the caller supplies the vocab-shard weight + local token indices, build
    # layer 0's input INSIDE the NEFF instead of reading a host-computed hidden:
    #   1. gather this rank's embedding rows for every replica token (indirect DMA,
    #      oob_mode.skip zeros the tokens outside this rank's vocab shard);
    #   2. reduce_scatter[TP] the partials -> this rank's summed SP shard;
    #   3. all_gather[TP] -> the full TP-replicated [T_attn, H] layer-0 input.
    # Each token lives in exactly one vocab shard, so the reduce sums one real row
    # and zeros, which is bit-exact regardless of the reduce order. The result
    # feeds layer 0 directly (SBUF/HBM, no external hidden), so the embedding does
    # not spill to HBM and back on the decode critical path.
    x = x_shard
    x_layer0_full = x_first_full
    if embed_weight is not None:
        T_shard_e, H_e = x_shard.shape
        T_attn_e = T_shard_e * tp_size
        _p_max = nl.tile_size.pmax
        _h_slice = embed_weight.shape[1]
        if _h_slice == H_e:
            # ── OPTION (a): VOCAB-dim sharding (full-H gather + TP all-reduce) ──
            ids_hbm = embed_ids.reshape((T_attn_e, 1))
            # ReduceScatter src -> _COLLECTIVE_HBM (see the module-scope note).
            emb_partial = nl.ndarray((T_attn_e, H_e), dtype=x_shard.dtype, buffer=_COLLECTIVE_HBM, name="embed_partial")
            for _c0 in range(0, T_attn_e, _p_max):
                _cn = min(_p_max, T_attn_e - _c0)
                ids_sb = nl.load(ids_hbm[_c0 : _c0 + _cn])
                emb_sb = nl.ndarray((_cn, H_e), dtype=x_shard.dtype, buffer=nl.sbuf)
                # oob_mode.skip leaves skipped partitions untouched, so start at zero.
                nisa.memset(emb_sb, value=0)
                nisa.dma_copy(
                    dst=emb_sb,
                    src=embed_weight.ap(
                        [[H_e, _cn], [1, H_e]],
                        offset=0,
                        vector_offset=ids_sb,
                        indirect_dim=0,
                    ),
                    oob_mode=_OOB_SKIP,
                    name=dma_name(f"embed_gather_c{_c0}"),
                )
                nl.store(emb_partial[_c0 : _c0 + _cn], value=emb_sb)
            # ReduceScatter dst AND the AllGather src below, so it moves as a src:
            # both ends of that pair use _COLLECTIVE_HBM.
            emb_shard = nl.ndarray((T_shard_e, H_e), dtype=x_shard.dtype, buffer=_COLLECTIVE_HBM, name="embed_shard")
            ncc.reduce_scatter(
                dsts=[emb_shard],
                srcs=[emb_partial],
                op=nl.add,
                replica_group=tp_replica_group,
                collective_dim=0,
            )
            emb_full = nl.ndarray((T_attn_e, H_e), dtype=x_shard.dtype, buffer=nl.shared_hbm, name="embed_full")
            ncc.all_gather(
                dsts=[emb_full],
                srcs=[emb_shard],
                replica_group=tp_replica_group,
                collective_dim=0,
            )
            x = emb_shard
            x_layer0_full = emb_full
        else:
            # ── OPTION (b): HIDDEN-dim sharding (H/tp_size gather + one SBUF
            # all_gather along the hidden dim; no cross-rank sum) ──
            # T_attn <= P_MAX for decode, so the whole slice fits one SBUF tile.
            # Gather this rank's H/tp_size slice for every replica token; the SBUF
            # all_gather(collective_dim=1) concatenates the tp_size slices in rank
            # order to rebuild the full [T_attn, H]. embed_ids are GLOBAL token ids
            # (a full-vocab lookup, so no masking); oob_mode.skip only guards padded
            # ids.
            ids_hbm = embed_ids.reshape((T_attn_e, 1))
            ids_sb = nl.load(ids_hbm)
            hslice_sb = nl.ndarray((T_attn_e, _h_slice), dtype=x_shard.dtype, buffer=nl.sbuf)
            nisa.memset(hslice_sb, value=0)
            nisa.dma_copy(
                dst=hslice_sb,
                src=embed_weight.ap(
                    [[_h_slice, T_attn_e], [1, _h_slice]],
                    offset=0,
                    vector_offset=ids_sb,
                    indirect_dim=0,
                ),
                oob_mode=_OOB_SKIP,
                name=dma_name("embedb_gather"),
            )
            xfull_sb = nl.ndarray((T_attn_e, H_e), dtype=x_shard.dtype, buffer=nl.sbuf)
            ncc.all_gather(
                srcs=[hslice_sb],
                dsts=[xfull_sb],
                replica_group=tp_replica_group,
                collective_dim=1,
            )
            emb_full = nl.ndarray((T_attn_e, H_e), dtype=x_shard.dtype, buffer=nl.shared_hbm, name="embedb_full")
            nl.store(emb_full, value=xfull_sb)
            # SP shard for layer 0's residual: this rank's token rows of the full
            # hidden (embed_x_shard_ids = tp_rank*T_shard + arange, built on host).
            sid_sb = nl.load(embed_x_shard_ids.reshape((T_shard_e, 1)))
            xshard_sb = nl.ndarray((T_shard_e, H_e), dtype=x_shard.dtype, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=xshard_sb,
                src=emb_full.ap(
                    [[H_e, T_shard_e], [1, H_e]],
                    offset=0,
                    vector_offset=sid_sb,
                    indirect_dim=0,
                ),
                name=dma_name("embedb_xshard"),
            )
            emb_shard = nl.ndarray((T_shard_e, H_e), dtype=x_shard.dtype, buffer=nl.shared_hbm, name="embedb_shard")
            nl.store(emb_shard, value=xshard_sb)
            x = emb_shard
            x_layer0_full = emb_full

    # ── FUSED ROPE PROLOGUE (env-gated by the caller) ────────────────────────
    # Compute the RoPE cos/sin ON-CHIP from pos_ids + inv_freq (rotary_emb fused
    # into the GK) instead of receiving pre-computed cos/sin. Reuse the kernel's
    # own _rope helper (same math the fuse_rope path uses), then feed the result
    # into the normal cos/sin RoPE-application path (FP8-safe) -- the fp8-blocked
    # fuse_rope flag is NOT used. _rope returns [d_head, T] with both halves equal
    # and no concentration, so take the first half_d rows and scale by the
    # concentration to match the external cos/sin ([half_d, B_attn, S_tkg]). RoPE
    # uses ABSOLUTE positions, so one cos/sin set is shared across all layers.
    cos_r = cos
    sin_r = sin
    if inv_freq is not None:
        _half_d = inv_freq.shape[0]
        _d_head_r = _half_d * 2
        _T = B_attn * S_tkg
        cos_r = nl.ndarray((_half_d, B_attn, S_tkg), dtype=cos.dtype, buffer=nl.shared_hbm, name="rope_cos")
        sin_r = nl.ndarray((_half_d, B_attn, S_tkg), dtype=sin.dtype, buffer=nl.shared_hbm, name="rope_sin")
        _sbm_rope = create_auto_alloc_manager()
        _sbm_rope.set_name_prefix("rope_")
        _sbm_rope.open_scope()  # _rope allocates its cos/sin on the caller's open scope
        _invf_sb = _sbm_rope.alloc_stack((_half_d, 1), dtype=inv_freq.dtype, buffer=nl.sbuf)
        nisa.dma_copy(_invf_sb, inv_freq.reshape((_half_d, 1)))
        # positions broadcast across the partition dim (see _rope's pos_ids contract):
        # load to one partition, then GPSIMD-broadcast to half_d partitions (a view
        # cannot broadcast the SBUF partition dim).
        _pos1 = _sbm_rope.alloc_stack((1, _T), dtype=pos_ids.dtype, buffer=nl.sbuf, align=4)
        nisa.dma_copy(_pos1, pos_ids.reshape((1, _T)))
        _pos_sb = _sbm_rope.alloc_stack((_half_d, _T), dtype=pos_ids.dtype, buffer=nl.sbuf)
        stream_shuffle_broadcast(src=_pos1, dst=_pos_sb)
        _cos_full, _sin_full = _rope(_invf_sb, _pos_sb, bs=B_attn, s_a=S_tkg, d_head=_d_head_r, sbm=_sbm_rope)
        _cos_sc = _sbm_rope.alloc_stack((_half_d, _T), dtype=cos.dtype, buffer=nl.sbuf)
        _sin_sc = _sbm_rope.alloc_stack((_half_d, _T), dtype=sin.dtype, buffer=nl.sbuf)
        nisa.tensor_scalar(_cos_sc, _cos_full[0:_half_d, :], op0=nl.multiply, operand0=rope_concentration)
        nisa.tensor_scalar(_sin_sc, _sin_full[0:_half_d, :], op0=nl.multiply, operand0=rope_concentration)
        nl.store(cos_r, value=_cos_sc.reshape((_half_d, B_attn, S_tkg)))
        nl.store(sin_r, value=_sin_sc.reshape((_half_d, B_attn, S_tkg)))
        _sbm_rope.close_scope()

    _qpin_io = {}  # trace-time hand-off between layers (see _decode_layer_body)
    for i in range(num_layers):
        # Real GPT-OSS: even layers slide (trimmed table + window-frame positions),
        # odd layers are full attention (full table + absolute positions). Selecting
        # the trimmed table for sliding layers is what makes them cheaper on device,
        # mirroring the production decode path.
        _sliding = i % 2 == 0
        _abt = active_blocks_table_swa if _sliding else active_blocks_table
        _pos = pos_ids_swa if _sliding else pos_ids
        # Packed metadata follows the same swa/full split, with no fallback: a packed
        # sliding layer carries its window inside the mask, so substituting the full
        # variant would silently give it full-context attention.
        _spm = seq_packed_attention_mask_swa if _sliding else seq_packed_attention_mask
        _sabt = seq_packed_active_blocks_table_swa if _sliding else seq_packed_active_blocks_table
        # Sliding layers write with the SWA-group index when provided, else the
        # shared kv_cache_update_idx (see the param docstring).
        _upd = kv_cache_update_idx_swa if (_sliding and kv_cache_update_idx_swa is not None) else kv_cache_update_idx
        # Pre-generated mask for this layer (None unless the hoist is on): sliding
        # layers get the SWA mask, full layers the full-attn mask.
        # Host-mask mode: full-attention (odd) layers load the host full mask.
        # Sliding (even) layers load the host SWA mask when the host supplies it,
        # else they generate in-kernel (pos_ids passed) as a fallback.
        # On-chip hoist uses the pre-generated mask for both parities.
        if _host_mask:
            if _sliding:
                _pregen_mask = _mask_swa  # None -> in-kernel SWA fallback
            else:
                _pregen_mask = _mask_full
        elif _hoist_mask:
            _pregen_mask = _mask_swa if _sliding else _mask_full
        else:
            _pregen_mask = None
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
        # KV cache selection. Stacked path: leading dim == num_layers -> per-layer
        # (li=i, golden); == 1 -> shared copy (li=0); else the HMA group_size ==
        # num_layers//2 stack where SWA layer 2j and full layer 2j+1 share buffer
        # j==i//2. Individual-buffer path: identical routing by the count of provided
        # buffers instead of the stacked leading dim. Checked in this order so a small
        # num_layers can't misclassify.
        if _use_indiv_kv:
            _li_kv = i if _n_kv == num_layers else (0 if _n_kv == 1 else i // 2)
            _k_buf = _k_indiv[_li_kv]
            _v_buf = _v_indiv[_li_kv]
        else:
            li_k = i if K_cache.shape[0] == num_layers else (0 if K_cache.shape[0] == 1 else i // 2)
            li_v = i if V_cache.shape[0] == num_layers else (0 if V_cache.shape[0] == 1 else i // 2)
            _k_buf = K_cache[li_k]
            _v_buf = V_cache[li_v]
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
            cos_r,
            sin_r,
            sink[li_sink],
            _k_buf,
            _v_buf,
            _abt,
            attention_mask,
            _upd,
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
            _spm,  # packed mask for this layer's window (None => dense attention)
            _sabt,  # packed block routes for this layer's window
            seq_packed_q_index_table,  # window-invariant
            seq_packed_seq_id_table,  # window-invariant
            seq_packed_accumulator_partial_route_table,  # window-invariant
            seq_packed_accumulator_group_state_route_table,  # window-invariant
            f"_L{i}",  # unique per-layer buffer-name suffix
            (i == num_layers - 1),  # is_last_layer: only the final layer spills to HBM
            _pregen_mask,  # hoisted mask (None unless VLLM_NEURON_MEGA_HOIST_MASK)
            k_scale,  # FP8 KV scales (layer-invariant; None unless FP8-KV on)
            v_scale,
            _kv_packed,  # packed fp8 K layout (6D stacked K_cache)
            (x_layer0_full if i == 0 else None),  # layer-0 pre-gathered input
            (not _sliding),  # swapped column-tiled PV is enabled only for full attention
            # Full attention uses Sync-triggered HWDGE; an explicit dependency preserves
            # final-V-prior-load -> first-W_out-load order across trigger engines. SWA
            # retains Scalar-triggered HWDGE with no additional dependency.
            (not _sliding),
            ("sync" if not _sliding else None),
            (not _sliding),
            _qpin_io,
        )
    # KV UPDATE: each layer wrote its token's K/V in-place into its cache buffer (a view
    # of an input tensor), so the input caches now hold all updates. Returning them makes
    # the compiler emit input<->output aliases (test passes *.must_alias_input), so the
    # golden can compare the post-update caches.
    set_dma_name_scope(None)
    if _mega_flag_default_on("VLLM_NEURON_MEGA_KV_UPDATE"):
        if _use_indiv_kv:
            # Return the provided individual buffers so each aliases its registered HMA pool.
            _k_out = tuple(t for t in _k_indiv if t is not None)
            _v_out = tuple(t for t in _v_indiv if t is not None)
            return (x, *_k_out, *_v_out)
        return x, K_cache, V_cache
    return x
