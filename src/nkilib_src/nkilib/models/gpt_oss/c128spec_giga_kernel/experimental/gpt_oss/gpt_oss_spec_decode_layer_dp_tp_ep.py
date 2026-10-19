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

"""GPT-OSS spec-decode TARGET (verify) decoder layer: DP x TP attention + EP DLoC MoE (LNC=1).

Speculative-decode target pass for ``S_tkg = spec_len + 1`` active tokens per sequence
(spec_len 3 -> S_tkg 4). Structurally identical to the non-spec c128 giga-kernel layer,
which is already ``S_tkg``-generic: this file is that layer with the MoE half switched
to the DLoC (dynamic on-chip routing) all-expert kernel.

Why the MoE half changes while attention does not. The two halves see different token
counts, and ``S_tkg`` scales them differently:

  * ATTENTION is per-DP-replica: ``T_attn = T_shard * tp_size = B_attn * S_tkg``. At 128
    sequences over DP16 x TP8 that is 8 sequences x 4 = 32 rows -- the same small-batch
    regime the c128 kernel is tuned for, so the attention half is used verbatim.
  * MoE is global: ``T_ep = T_shard * ep_size``. At 128 sequences x S_tkg 4 over EP128
    that is **512** tokens -- the c512 giga-kernel's regime, where a static all-expert
    pass over all 512 rows is wasteful and DLoC's per-block dynamic dispatch wins.

So this layer is "c128 attention + c512 MoE", which is exactly what spec_len 3 at
concurrency 128 asks for.

    residual = x
    x = AllGather[TP] -> attention_block_tkg(input_layernorm + QKV + RoPE +
                          paged attn + sinks + O-proj partial) -> ReduceScatter[TP]
    x = residual + x
    residual = x
    x = AllGather[EP] -> moe_tkg(is_all_expert_dynamic, DLoC) -> ReduceScatter[EP]
    x = residual + x

Concretely at DP16 x TP8 x EP128, 128 sequences, spec_len 3:

    T_shard (rows/rank) = 4      T_attn = 32 = B_attn(8) x S_tkg(4)      T_ep = 512

Mask topology. The target pass verifies a linear draft CHAIN, not a tree, so the active
block is a dense lower-triangular causal mask over the ``S_tkg`` active tokens, overlaid
by ``gen_mask_tkg_hbm(active_mask=...)`` onto the last ``S_tkg`` prior slots. That matches
what vllm-neuron builds in production, where the same tensor is expressed as
``triu(ones(S_tkg, S_tkg))`` in ``[key, B, q_head, query]`` order -- the identical tensor,
transposed. No tree-attention path exists on either side.

Token bookkeeping, replica groups, and weight sharding are unchanged from the non-spec
layer: ``tp_replica_group`` is the per-DP-replica TP group, ``ep_replica_group`` spans all
ranks, attention weights are the per-rank TP head shard (bf16), the expert weights are
this rank's single MXFP4 local expert, and the router is replicated.
"""

import os as _os

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl
from nki.collectives import ReplicaGroup

from ...core.attention.attention_tkg_utils import is_qk_swapped
from ...core.attention.gen_mask_tkg import gen_mask_tkg_hbm
from ...core.moe.moe_tkg.moe_tkg import moe_tkg as _moe_tkg
from ...core.router_topk.router_topk import XSBLayout_tp201__2 as _ROUTER_X_SB_LAYOUT
from ...core.router_topk.router_topk import router_topk as _router_topk
from ...core.subkernels.rmsnorm_mx_quantize_tkg import rmsnorm_mx_quantize_tkg as _rmsnorm_mx_quantize_tkg
from ...core.utils.allocator import create_auto_alloc_manager, sizeinbytes
from ...core.utils.common_types import (
    ActFnType,
    ExpertAffinityScaleMode,
    MoEAllToAllVStrategy,
    QuantizationType,
    RouterActFnType,
)
from ...core.utils.dma_names import dma_name, inst_name
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import mega_flag_default_on as _mega_flag_default_on
from ..trace_caches import reset_trace_state
from ..transformer.attention_block_tkg import attention_block_tkg

# Buffer class for every HBM tensor that is an OPERAND of a collective: private HBM. At LNC=1 the nki
# collectives reject a shared_hbm SOURCE and accept private srcs and dsts alike, so every collective
# operand here (AllGather srcs/dsts, ReduceScatter srcs/dsts) is private. Kernel I/O stays shared_hbm.
_COLLECTIVE_HBM = nl.private_hbm


@nki.jit
def gpt_oss_spec_decode_layer_dp_tp_ep_kernel(
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
):
    """One rank's full decoder layer (see module docstring). Returns [B/(DP*TP), H]."""
    # MANDATORY on every outermost @nki.jit entry, not just the mega one. This gate traces many
    # ranks, and a leaked MoE SBUF-capture FIFO entry from a previous trace is specifically
    # dangerous for THIS kernel: the combine below keys the SBUF path off that FIFO being
    # non-empty, so a stale entry would route DLoC down the static path with a foreign buffer.
    reset_trace_state()
    # Positional call (NKI tracer rejects keyword args to nested functions).
    _layer_out = _spec_decode_layer_body(
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
    # _spec_decode_layer_body sets a per-layer DMA-name scope and deliberately does not clear it (the
    # mega kernel re-scopes on every layer). Clear it at this trace boundary so the scope cannot
    # leak into the NEXT kernel traced in the same process: a leaked scope makes unrelated kernels
    # emit names and trip dma_name()'s duplicate guard -- observed as moe_block_dp_ep failing on a
    # duplicate '_prequant_gamma_load' after this entry ran in the same pytest worker.
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
        nisa.dma_copy(dst=rows_sb, src=rows_hbm[t0 : t0 + t_n, 0:H_pack], name=dma_name(f"prequant_rows_load_t{t0}"))
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


def _spec_decode_layer_body(
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
    x_full=None,
    enable_pv_swapped_column_tiling=False,
    use_swdge_for_wout_load=None,
    wout_load_engine=None,
    order_wout_after_v_prior=False,
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
    # The previous layer may hand its output over as the [128, T_shard * H / 128] wide SBUF view
    # (RESID_P128 below), so take H from the layer's norm gamma and T_shard from the attention token count.
    H = input_layernorm_weight.shape[-1]
    T_shard = B_attn * S_tkg // tp_size
    kernel_assert(x_shard.shape[0] * x_shard.shape[1] == T_shard * H, "layer input size mismatch")
    dtype = x_shard.dtype
    T_attn = T_shard * tp_size  # B/DP: tokens per DP replica (attention gather)
    T_ep = T_shard * ep_size  # B: global tokens (EP gather)
    # SPEC TARGET SHAPE CONTRACT. T_shard counts ROWS of x_shard, so it already carries the
    # S_tkg factor: T_shard = (B_attn / tp_size) * S_tkg. Attention reshapes the TP-gathered
    # activation to (B_attn, S_tkg, H), so that product has to close exactly.
    kernel_assert(
        T_attn == B_attn * S_tkg,
        f"spec target: T_attn={T_attn} (=T_shard {T_shard} * tp_size {tp_size}) must equal "
        f"B_attn {B_attn} * S_tkg {S_tkg}. x_shard must hold this rank's (B_attn/tp_size) "
        f"sequences x S_tkg active tokens.",
    )
    # DLoC block_size 64 needs T_ep/2 >= 64, i.e. T_ep a multiple of 128. At 128 sequences x
    # S_tkg 4 -> T_ep=512. A non-spec S_tkg=1 run at this topology gives T_ep=128, which also
    # satisfies it, so the guard does not forbid reusing this kernel at S_tkg=1.
    kernel_assert(
        T_ep >= 256 and T_ep % 128 == 0,
        f"spec target: DLoC block_size 128 requires T_ep >= 256 and a multiple of 128, got "
        f"T_ep={T_ep} = T_shard {T_shard} * ep_size {ep_size}; raise S_tkg or ep_size.",
    )

    # PROTOTYPE (tracer only): inter-layer SBUF hand-off. When on, the previous
    # layer's output arrives as an SBUF [T_shard, H] tensor (x_shard) instead of HBM,
    # and this layer keeps its own output in SBUF too (returned to the next layer),
    # storing to HBM only on the last layer. x_shard is consumed twice below: as the
    # AllGather[TP] source (dma_copy accepts SBUF src) and in residual add #1
    # (nl.load only reads HBM, so branch on the buffer type).
    _sbuf_interlayer = _mega_flag_default_on("VLLM_NEURON_MEGA_SBUF_INTERLAYER")
    # fp8 EP ReduceScatter at the source: the MoE writes its [T_ep, H] output as fp8_e4m3fn and the EP
    # ReduceScatter moves and sums 1 B/elt (RS is ~3.8x slower per byte than plain movement, so payload
    # size is the lever; with 128 experts and top_k=4 each rank's partial is zero for ~124 of 128 tokens).
    # Needs NEURON_RT_ENABLE_OCP=1; the EP128 accuracy gate is calibrated for it (rtol 0.12).
    # Pre-gather quantize+router: RMSNorm+MX-quantize+router are per-token ops, so running them
    # after the EP AllGather makes all T_ep ranks recompute T_ep identical copies (the swizzle
    # alone moves exactly T*H*2 bytes). Do them for THIS rank's token and gather the results.
    # Pinned ON, not a flag: the DLoC MoE consumes the packed [T_ep, H+H/4] fp8 dispatch
    # buffer directly (_layout_adapter_prequant_mx_hbm gathers+transposes it per block), so
    # there is no unpacked-input path to fall back to.
    _prequant_moe = True
    _prequant_aff16 = _mega_flag_default_on("VLLM_NEURON_MEGA_PREQUANT_AFF16")
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
    # Banded single-10K softmax: shortening s_prior beats FA s_prior tiling on this shape, so
    # DISABLE_FA is part of the measured-optimal bundle and is ON by default.
    _enable_fa_s_prior_tiling = not _mega_flag_default_on("VLLM_NEURON_MEGA_DISABLE_FA")
    _x_in_sbuf = _sbuf_interlayer and (getattr(x_shard, "buffer", None) == nl.sbuf)
    # Residual add #2 -- the fp8 ReduceScatter result dequant + add -- runs on the [128, T_shard * H / 128] view of
    # the same bytes instead of [T_shard=4, H] (c512 CR-310882146 trick 8; -182/-163 us min/median here, 2026-10-08):
    # the Vector engine streams a partition's elements serially, so the 4-wide layout costs ~2.8 us per layer on
    # the critical path between the MoE ReduceScatter and the next layer's AllGather (~0.2 us at 128 wide). The
    # layer output is handed to the next layer in that layout; the next layer stores it to its AllGather source
    # through the same view and reloads the [T_shard, H] copy it needs for residual add #1 from there.
    _P = nl.tile_size.pmax
    _resid_p128 = _sbuf_interlayer and (T_shard * H) % _P == 0
    _x_p128 = _x_in_sbuf and tuple(x_shard.shape) != (T_shard, H)  # previous layer handed over the wide view

    # Per-layer SBUF manager with a unique name prefix so attention_block_tkg's
    # internal named buffers stay globally unique when inlined once per layer.
    _sbm = create_auto_alloc_manager()
    _sbm.set_name_prefix(f"attn{_t}_")

    # No cross-layer weight prefetch: the MoE is the c512 weight-split implementation (CR-302598650),
    # which loads the gate weight up front and the up/down weights at the compute sites (HW-DGE inside
    # the DLoC dynamic loop), and its loaders assert the prefetch FIFOs are empty. Prefetch off is also
    # the measured optimum on this kernel (2026-09-19, steering_private/gptoss_c128spec_json/README.md).

    # MoE output buffer, allocated and zero-filled HERE, before attention (from c512 CR-306569240, A/B'd on
    # this kernel 2026-09-23: -282 us / step at 36 layers, bit-identical outputs):
    #  * private HBM is a legal LNC1 collective source, so the EP ReduceScatter reads the MoE output
    #    directly and the 1.5 MB moe_out -> staging copy per layer is gone (-231 us alone);
    #  * zero-filling it at layer start instead of inside the MoE after the router keeps the fill off the
    #    DMA rings when the router's small stores feed the EP AllGather trigger (-50 us on top; NOTE the
    #    same relocation WITHOUT the private buffer measured +350 us, the two are not separable).
    # The MoE is told the buffer is pre-zeroed (output_prezeroed=True) and skips its own zero pass.
    _moe_out_dtype = nl.float8_e4m3fn
    _moe_out_esize = 1
    moe_out_buf = nl.ndarray((T_ep, H), dtype=_moe_out_dtype, buffer=nl.private_hbm, name="moe_out" + _t)
    _pz_tile_T = min(T_ep, nl.tile_size.pmax)
    _pz_H_u32 = H * _moe_out_esize // 4
    _pz_zero_sb = nl.ndarray((_pz_tile_T, _pz_H_u32), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.memset(_pz_zero_sb, 0, engine=nisa.vector_engine, name=inst_name("prezero_output_memset"))
    for _pz_t in range(T_ep // _pz_tile_T):
        nisa.dma_copy(
            src=_pz_zero_sb[:, :],
            dst=moe_out_buf.ap(
                [[_pz_H_u32, _pz_tile_T], [1, _pz_H_u32]],
                offset=_pz_t * _pz_tile_T * _pz_H_u32,
                dtype=nl.uint32,
            ),
            dge_mode=nisa.dge_mode.none,
            name=dma_name(f"prezero_output_t{_pz_t}"),
        )

    # ════════════════ Attention (input RMSNorm fused inside) ════════════════
    # Reconstruct this replica's full [T_attn, H] for QKV. Layer 0 passes the
    # already-gathered (TP-replicated) input as x_full, so skip the dma_copy +
    # AllGather[TP]; layers >= 1 arrive SP-sharded and AllGather to rebuild it.
    if x_full is not None:
        X = x_full.reshape((B_attn, S_tkg, H))
    elif _sbuf_attn_ag:
        kernel_assert(not _x_p128, "RESID_P128 hand-off needs the HBM AllGather source path (SBUF_ATTN_AG off)")
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
        # Collective srcs use _COLLECTIVE_HBM (shared on nki 0.6, private on >=0.7) --
        # see its definition at the top of this module.
        a_src = nl.ndarray((T_shard, H), dtype=dtype, buffer=_COLLECTIVE_HBM, name="a_src" + _t)
        a_dst = nl.ndarray((T_attn, H), dtype=dtype, buffer=nl.shared_hbm, name="a_dst" + _t)
        nisa.dma_copy(
            dst=a_src.reshape(tuple(x_shard.shape)) if _x_p128 else a_src,
            src=x_shard,
            name=dma_name("attn_ag_src_store"),
        )
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
        softmax_scale=softmax_scale,
        enable_fa_s_prior_tiling=_enable_fa_s_prior_tiling,
        enable_pv_moving_free_dim_padding=False,
        enable_pv_swapped_column_tiling=enable_pv_swapped_column_tiling,
        pos_ids=None if pregen_mask is not None else pos_ids,
        swa_start_pos_ids=None if pregen_mask is not None else swa_start_pos_ids,
        S_tkg=S_tkg,
        # FP8 KV cache: raw fp8 K/V + scales; softmax_scale is pre-fused with k_scale
        # and W_out pre-fused with v_scale by the caller. None (unset) => bf16/fp16 KV.
        k_scale=k_scale,
        v_scale=v_scale,
        fp8_packed=fp8_packed,  # packed K [nb, bl//2, d, 2] -> DMA-gather-transpose path
        sbm=_sbm,  # per-layer name prefix -> unique attention buffer names
    )  # [B/DP, H] per-rank O-proj partial
    resid1 = (
        nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.private_hbm, name="resid1" + _t)
        if not _keep_resid1_in_sbuf
        else None
    )
    if _sbuf_attn_rs:
        # PROTOTYPE (tracer only): attention_block_tkg emitted its [T_attn, H] O-proj partial
        # straight to SBUF (out_in_sb=True). Feed it to reduce_scatter as a SBUF collective
        # (src SBUF + dst SBUF, collective_dim=0), dropping the ars_src HBM staging copy.
        # Residual add #1 then runs fully in SBUF. The experiment can extend its lifetime.
        attn_shard_sb = nl.ndarray((T_shard, H), dtype=attn_out.dtype, buffer=nl.sbuf)
        ncc.reduce_scatter(
            dsts=[attn_shard_sb], srcs=[attn_out], op=nl.add, replica_group=tp_replica_group, collective_dim=0
        )
        if _x_p128:
            # the [T_shard, H] copy of this layer's input, from the AllGather source just stored above
            x_sb = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb, src=a_src, name=dma_name("resid1_x_reload"))
        elif _x_in_sbuf:
            x_sb = x_shard
        else:
            # nisa.dma_copy (not nl.load) so the load can carry a name= for DMA-order pinning.
            x_sb = nl.ndarray((T_shard, H), dtype=x_shard.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb, src=x_shard, name=dma_name("x_shard_load"))
        resid1_sb = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=resid1_sb, data1=x_sb, data2=attn_shard_sb, op=nl.add)
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
        if _x_p128:
            x_sb = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb, src=a_src, name=dma_name("resid1_x_reload"))
        else:
            x_sb = x_shard if _x_in_sbuf else nl.load(x_shard)
        a_sb = nl.load(attn_shard)
        resid1_sb = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=resid1_sb, data1=x_sb, data2=a_sb, op=nl.add)
        nl.store(resid1, value=resid1_sb)
    # No plain [T_ep, H] EP AllGather buffer here: the DLoC path dispatches only the packed
    # fp8 quant + affinity tensors built below, so allocating m_src/m_dst would reserve a dead
    # [T_ep, H] shared_hbm tile per layer.

    if _prequant_moe:
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
        # Expert-major affinity gather (c512 CR-310882146 trick 5; -546/-457 us min/median here, 2026-10-08): the
        # router leaves its one-hot scattered [T_shard, E] fp32 affinities in SBUF; they are PE-transposed to
        # [E, T_shard] and stored expert-major into the AllGather source, so the MoE's rank slice of the gathered
        # buffer is T_shard contiguous values per rank (128 descriptors) instead of one element per token (T_ep
        # descriptors on the SWDGE path). Needs E <= 128 and E / T_shard a power of two (E=128, T_shard=4 here);
        # other shapes keep the token-major [T_shard, E] gather.
        _epr = E_total // T_shard if E_total % T_shard == 0 else 0
        _aff_expert_major = _epr > 0 and _epr & (_epr - 1) == 0 and E_total <= nl.tile_size.pmax
        if _aff_expert_major:
            _aff_router_sb = nl.ndarray((T_shard, E_total), dtype=nl.float32, buffer=nl.sbuf)
            aff_src = None
        else:
            _aff_router_sb = None
            aff_src = nl.ndarray((T_shard, E_total), dtype=_aff_dt, buffer=_COLLECTIVE_HBM, name="aff_src" + _t)
        eidx_sb = nl.ndarray((T_shard, top_k), dtype=nl.uint32, buffer=nl.sbuf)
        quant_dst = nl.ndarray((T_ep, H_pack), dtype=nl.float8_e4m3fn, buffer=nl.shared_hbm, name="quant_dst" + _t)
        # With an SBUF expert_affinities the router returns its own one-hot scattered [T_shard, E] fp32 tile as
        # outputs[2] (it does not write the caller's buffer), so the expert-major path reads that return value.
        _router_outs = _router_topk(
            x=pq_norm_sb,
            w=router_weights,
            w_bias=router_bias,
            router_logits=None,
            expert_affinities=_aff_router_sb if _aff_expert_major else aff_src,
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
        _aff_ag_src = aff_src
        if _aff_expert_major:
            # [T_shard, E] fp32 one-hot scattered tile -> PE transpose -> [E, T_shard] in the gather dtype ->
            # packed store (small, latency-critical: Sync HWDGE queue, away from bulk weight traffic).
            _aff_pk = nl.ndarray((T_shard, E_total), dtype=_aff_dt, buffer=_COLLECTIVE_HBM, name="aff_pk" + _t)
            _aff_em_sb = _router_outs[2]  # the router's one-hot scattered [T_shard, E] fp32 tile
            _aff_em_ps = nl.ndarray((E_total, T_shard), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_transpose(dst=_aff_em_ps, data=_aff_em_sb, name=inst_name("aff_em_xpose"))
            _aff_emT_sb = nl.ndarray((E_total, T_shard), dtype=_aff_dt, buffer=nl.sbuf)
            nisa.tensor_copy(dst=_aff_emT_sb, src=_aff_em_ps, name=inst_name("aff_em_xpose_copy"))
            nisa.dma_copy(
                dst=_aff_pk.reshape((E_total, T_shard)),
                src=_aff_emT_sb,
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
                name=dma_name("aff_em_store"),
            )
            _aff_ag_src = _aff_pk
        # fp8 AllGather is unsupported (runtime error 1202) -> bitcast to bf16, identical bytes.
        # Two separate gathers: ncc's coalesced list-form is correct but pathologically slow for
        # these mismatched widths (490 us vs 13.7 us, +235% e2e).
        ncc.all_gather(
            dsts=[quant_dst.view(nl.bfloat16)],
            srcs=[pq_src.view(nl.bfloat16)],
            replica_group=ep_replica_group,
            collective_dim=0,
        )
        ncc.all_gather(dsts=[aff_dst], srcs=[_aff_ag_src], replica_group=ep_replica_group, collective_dim=0)
    # DLoC (dynamic on-chip routing) all-expert MoE. The gathered tensors are fed in directly,
    # with no packed [T, H_concat] intermediate. quant_dst is the packed [T_ep, H_pack] fp8
    # (hidden_quant | hidden_scale): moe_tkg reads H from the down weights and T from
    # hidden_input.shape[0], so the trailing scale columns are transparent, and
    # _layout_adapter_prequant_mx_hbm gathers+transposes it per block WITHOUT re-quantizing (a
    # 1-byte hidden_input means "already MX-quantized"). That per-block gather is why the packed
    # buffer is handed over as-is instead of being unpacked up front the way the static
    # all-expert path does. aff_dst is the router's sparse [T_ep, E] scatter
    # (router_pre_norm=False leaves E-k structural zeros per row); the kernel rank-slices it to
    # this rank's E_L experts and nonzero_with_count skips blocks with no routed tokens, which
    # is what makes the dispatch genuinely dynamic.
    #
    # block_size 128 (c512 CR-302598650 rev 2): T_ep=512 -> 4 blocks = 1 static + 3 dynamic; fewer blocks mean less
    # DMA throttling and a lower probability that a dynamic block is triggered. Was 64 (8 blocks).
    # It requires T_ep to be a multiple of 128, asserted at the top of this body (moe_tkg's
    # _is_valid_block_size would otherwise catch it).
    _dloc_block_size = 128
    # Dummy expert_index: moe_tkg's affinity-mask block reads K = expert_index.shape[-1] for
    # DISABLED + POST_SCALE, but with mask_unselected_experts=False it only rank-slices and
    # never reads the values, so a shape-only [T_ep, top_k] placeholder suffices.
    _eidx_dummy = nl.ndarray((T_ep, top_k), dtype=nl.uint32, buffer=nl.shared_hbm, name="eidx_dummy" + _t)
    moe_out = _moe_tkg(
        hidden_input=quant_dst,
        expert_gate_up_weights=expert_gate_up_weights,
        expert_down_weights=expert_down_weights,
        expert_affinities=aff_dst,  # sparse [T_ep, E]; kernel rank-slices to [T_ep, E_L]
        expert_index=_eidx_dummy,  # shape-only (K); values unused when mask_unselected=False
        is_all_expert=True,
        is_all_expert_dynamic=True,
        block_size=_dloc_block_size,
        all_to_all_v_strategy=MoEAllToAllVStrategy.DISABLED,
        mask_unselected_experts=False,  # affinities are already sparse from the router scatter
        expert_affinities_packed_expert_major=T_shard if _aff_expert_major else 0,
        rank_id=rank_id,
        expert_gate_up_weights_scale=expert_gate_up_weights_scale,
        expert_down_weights_scale=expert_down_weights_scale,
        expert_gate_up_bias=expert_gate_up_bias,
        expert_down_bias=expert_down_bias,
        expert_affinities_scaling_mode=ExpertAffinityScaleMode.POST_SCALE,
        activation_fn=ActFnType.Swish,
        gate_clamp_upper_limit=gate_clamp_upper_limit,
        up_clamp_upper_limit=up_clamp_upper_limit,
        up_clamp_lower_limit=up_clamp_lower_limit,
        # fp8-RS at source: the MoE writes its output directly in fp8, so the staging copy and
        # the EP ReduceScatter below both move 1 B/elt. moe_shard derives its dtype
        # from moe_out, so the whole combine follows automatically and the collective stays
        # single-dtype either way; residual #2 dequantizes. E_L==1 here, so the DLoC scatter is a
        # single write per row and the cast happens once.
        output_dtype=nl.float8_e4m3fn,
        output=moe_out_buf,
        output_prezeroed=True,
    )  # [T_ep, H] position-aligned (no token-index columns), private HBM, ready for ReduceScatter

    # Combine: ReduceScatter[EP] sum per-expert partials + re-shard -> [B/(DP*TP), H].
    moe_shard = nl.ndarray((T_shard, H), dtype=moe_out.dtype, buffer=_COLLECTIVE_HBM, name="moe_shard" + _t)
    # moe_out is the private-HBM buffer allocated at layer start: reduce-scatter it directly.
    ncc.reduce_scatter(dsts=[moe_shard], srcs=[moe_out], op=nl.add, replica_group=ep_replica_group, collective_dim=0)

    # ── residual add #2 -> layer output [B/(DP*TP), H] ──
    # Reuse resid1 from SBUF when available; otherwise load it from HBM.
    out = nl.ndarray((T_shard, H), dtype=dtype, buffer=nl.shared_hbm, name="mega_out" + _t)
    # RESID_P128: every operand of residual add #2 is HBM here, so view them all as [128, T_shard * H / 128]
    # (same bytes) and hand that view to the next layer; SBUF-resident resid1 cannot be re-viewed across partitions.
    _wide = _resid_p128 and not _keep_resid1_in_sbuf
    _o_shape = (_P, T_shard * H // _P) if _wide else (T_shard, H)
    out_sb = nl.ndarray(_o_shape, dtype=dtype, buffer=nl.sbuf)
    # nisa.dma_copy (not nl.load/nl.store) on these paths so each can carry a name= for
    # DMA-order pinning; nl.load/nl.store take no name.
    if _keep_resid1_in_sbuf:
        r_sb = resid1_sb
    else:
        r_sb = nl.ndarray(_o_shape, dtype=resid1.dtype, buffer=nl.sbuf)
        nisa.dma_copy(dst=r_sb, src=resid1.reshape(_o_shape) if _wide else resid1, name=dma_name("resid1_load"))
    # residual1 add MoE: reload the reduced shard from HBM.
    mo_sb = nl.ndarray(_o_shape, dtype=moe_shard.dtype, buffer=nl.sbuf)
    nisa.dma_copy(dst=mo_sb, src=moe_shard.reshape(_o_shape) if _wide else moe_shard, name=dma_name("moe_shard_load"))
    if mo_sb.dtype != dtype:
        # Dequantize the fp8 EP-reduced shard to the layer dtype before residual #2, rather than
        # relying on an implicit mixed-dtype tensor_tensor operand. A plain cast is correct: the
        # MoE's output cast carried no scale, so there is nothing to undo. Cost is one Vector pass over [T_shard, H] only -- T_shard is the
        # per-rank shard (4 rows at 128 sequences x S_tkg 4), not T_ep, so this is negligible
        # against the T_ep-sized bytes the fp8 RS just saved.
        _mo_fp8 = mo_sb
        mo_sb = nl.ndarray(_o_shape, dtype=dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=mo_sb, src=_mo_fp8)
    nisa.tensor_tensor(dst=out_sb, data1=r_sb, data2=mo_sb, op=nl.add)
    # Inter-layer SBUF: hand the output to the next layer straight from SBUF; store to
    # HBM only on the final layer (the @nki.jit kernel must return an HBM tensor).
    if _sbuf_interlayer and not is_last_layer:
        return out_sb
    nisa.dma_copy(dst=out.reshape(_o_shape) if _wide else out, src=out_sb, name=dma_name("mega_out_store"))
    return out


@nki.jit
def gpt_oss_spec_decode_mega_kernel(
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
):
    """Multi-layer GPT-OSS decode "mega kernel": stack ``num_layers`` fused
    DP x TP attention + EP MoE decoder layers in one NEFF.

    Each layer's output ``[B/(DP*TP), H]`` is the next layer's input (same SP
    layout), so the layers chain by threading the running ``x`` shard through
    :func:`_spec_decode_layer_body`. Per-layer weights, KV caches, per-layer SWA starts
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
    _hoist_mask = _mega_flag_default_on("VLLM_NEURON_MEGA_HOIST_MASK")
    # Must match the value the layer body passes to attention, or the hoisted mask layout
    # will not match the shape attention expects.
    _enable_fa_s_prior_tiling = not _mega_flag_default_on("VLLM_NEURON_MEGA_DISABLE_FA")
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

    x = x_shard
    for i in range(num_layers):
        # Real GPT-OSS: even layers slide (trimmed table + window-frame positions),
        # odd layers are full attention (full table + absolute positions). Selecting
        # the trimmed table for sliding layers is what makes them cheaper on device,
        # mirroring the production decode path.
        _sliding = i % 2 == 0
        _abt = active_blocks_table_swa if _sliding else active_blocks_table
        _pos = pos_ids_swa if _sliding else pos_ids
        # Sliding layers write with the SWA-group index when provided, else the
        # shared kv_cache_update_idx (see the param docstring).
        _upd = kv_cache_update_idx_swa if (_sliding and kv_cache_update_idx_swa is not None) else kv_cache_update_idx
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
        # KV cache axis: leading dim == num_layers -> per-layer (li=i, golden);
        # == 1 -> shared copy (li=0); else the HMA group_size==num_layers//2 stack
        # where SWA layer 2j and full layer 2j+1 share buffer j==i//2. Checked in
        # this order so a small num_layers can't misclassify.
        li_k = i if K_cache.shape[0] == num_layers else (0 if K_cache.shape[0] == 1 else i // 2)
        li_v = i if V_cache.shape[0] == num_layers else (0 if V_cache.shape[0] == 1 else i // 2)
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
        # order must match _spec_decode_layer_body's signature.
        x = _spec_decode_layer_body(
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
            f"_L{i}",  # unique per-layer buffer-name suffix
            (i == num_layers - 1),  # is_last_layer: only the final layer spills to HBM
            _pregen_mask,  # hoisted mask (None unless VLLM_NEURON_MEGA_HOIST_MASK)
            k_scale,  # FP8 KV scales (layer-invariant; None unless FP8-KV on)
            v_scale,
            _kv_packed,  # packed fp8 K layout (6D stacked K_cache)
            (x_first_full if i == 0 else None),  # layer-0 pre-gathered input
            (not _sliding),  # swapped column-tiled PV is enabled only for full attention
            # Full attention uses Sync-triggered HWDGE; an explicit dependency preserves
            # final-V-prior-load -> first-W_out-load order across trigger engines. SWA
            # retains Scalar-triggered HWDGE with no additional dependency.
            (not _sliding),
            ("sync" if not _sliding else None),
            (not _sliding),
        )
    # KV UPDATE: each layer wrote its token's K/V in-place into K_cache[li_k]/V_cache[li_v]
    # (views of these input tensors), so the full input caches now hold all updates. Returning
    # them makes the compiler emit input<->output aliases (test passes *.must_alias_input), so
    # the golden can compare the post-update caches. Caches MUST be stacked per-layer here.
    if _mega_flag_default_on("VLLM_NEURON_MEGA_KV_UPDATE"):
        return x, K_cache, V_cache
    return x
