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

"""Standalone sparse MLA latent + RoPE attention (CTE).

KERNEL A of the split DeepSeek-V3.2 sparse-MLA forward. Computes, per query, the
absorbed-latent sparse attention output attn_latent[H, L] over the topk-selected cache
rows and writes out_attn_hbm[B=1, S, H*L] (row-major h*L + l). This is the un-projected
value path; pair with ``mla_vupmx_oproj_cte_kernel`` (kernel B) for V-up + o_proj.

S-sharded across cores. Under Context Parallelism the framework gathers the latent KV
(c_kv / k_pe) to the full S_kv before calling this kernel; queries stay this rank's
S-shard and topk indices address the full [0, S_kv) gathered range.

The MM2 output's latent column layout follows the DOWNSTREAM o_proj kernel's precision (the
``precision`` arg): MX pre-permutes each head's latent into 4-pack order (natural latent
l = 4*group + sub stored at physical column sub*(L//4) + group) to feed the MX o_proj's
contiguous DMA-transpose load; BF16 emits natural order. See ``_mm2_out_aps``. This kernel's
own arithmetic is bf16 in both cases.

QUERY-LOOP DRIVERS (sparse stage). The per-query work is emitted by ``_query_body``, driven one of
two ways depending on S_local -- see ``_UNROLL_MAX_S_LOCAL`` / ``_QUERY_CHUNK``:

* ``S_local <= _UNROLL_MAX_S_LOCAL`` (and the whole resident path): a FULLY UNROLLED trace-time
  loop. Every address is a compile-time constant. This is the faster code and the default.
* ``S_local >  _UNROLL_MAX_S_LOCAL``: an ``nl.fori_loop`` DEVICE loop over query chunks. Needed
  because unrolling makes program size scale with S_local , and pushes the serialized colz body blob
  past its uint32 (4 GiB) offset range.
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ....core.qkv.qkv_cte import _get_psum_bank_size
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ....core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast
from .mla_common_cte import (
    _H_PACK,
    _K_CHUNK,
    _MM1_TILE,
    _NUM_HW_PSUM_BANKS,
    _P_MAX,
    _SM_TILE,
    _TI_REPLICATE_MASK,
    MlaPrecision,
    _new_sbm,
)
from .mla_validate_params import _validate_mla_attention_inputs

_FLOAT32_MIN = -3.4028235e38  # most-negative finite fp32; dense causal mask fills future keys
_UNROLL_MAX_S_LOCAL = 2048
_QUERY_CHUNK = 256


# Max gathered S_kv the SBUF-RESIDENT path (use_sbuf_indirect=True) supports, by cache dtype.
# _load_kv_cache makes the whole cache resident: n_l tiles of [128, S_kv] plus [R, S_kv], i.e.
# (L + R) * itemsize bytes/partition. Against the 245,752 B stack minus the ~80 KB per-query
# working set, bf16 measured OK at 12288 and OOM at 14336; these are the round numbers below
# that edge, doubled for a 1-byte cache since the residency halves.
_SBUF_RESIDENT_MAX_S_KV = {2: 10240, 1: 20480}


def _resident_s_kv_limit(c_kv_hbm):
    """Ceiling for the SBUF-resident path, from the KV cache element size."""
    itemsize = 1 if c_kv_hbm.dtype in (nl.float8_e4m3, nl.float8_e4m3fn, nl.float8_e5m2) else 2
    return _SBUF_RESIDENT_MAX_S_KV[itemsize], itemsize


def _mm2_out_aps(out_bf16, pv_psum, H, L, precision):
    """Access patterns for the MM2 [H, L] eviction, per the consumer kernel's precision.

    MX: each head's latent columns are PRE-PERMUTED into 4-pack order (natural latent
    ``l = 4*group + sub`` stored at physical column ``sub*(L//4) + group``, i.e. sub-major) so
    the MX o_proj can transpose contiguous groups with a plain ``dma_transpose`` rather than an
    ``nc_transpose``, and ``W_uv`` is loaded with the SAME permutation. Done here as a vector
    free-axis permute because a permuted HBM write would be a ~14x-slower scatter.

    BF16: NATURAL column order. The bf16 o_proj has no MX layout to feed — it transposes the
    latent on the Tensor Engine — so the permute would be work that the consumer then has to
    undo. This is the ONLY thing ``precision`` changes in this kernel; the math is bf16 either
    way. Keep in lockstep with the o_proj load, the W_uv load, and both refs.
    """
    if precision.is_bf16():
        return out_bf16.ap(pattern=[[L, H], [1, L]]), pv_psum.ap(pattern=[[L, H], [1, L]])
    return (
        out_bf16.ap(pattern=[[L, H], [1, L // _H_PACK], [L // _H_PACK, _H_PACK]]),
        pv_psum.ap(pattern=[[L, H], [_H_PACK, L // _H_PACK], [1, _H_PACK]]),
    )


@nki.jit
def mla_sparse_attention_cte_kernel(
    q_lift_hbm: nl.NkiTensor,
    q_pe_hbm: nl.NkiTensor,
    c_kv_hbm: nl.NkiTensor,
    k_pe_hbm: nl.NkiTensor,
    softmax_scale: float,
    topk_indices_hbm: nl.NkiTensor = None,
    topk_tiled: bool = False,
    dense: bool = False,
    q_pos_offset_hbm: nl.NkiTensor = None,
    precision: MlaPrecision = MlaPrecision.MX,
    use_sbuf_indirect: bool = False,
) -> nl.NkiTensor:
    """Standalone sparse latent + RoPE attention (S-sharded across cores).

    KERNEL A of the split DeepSeek-V3.2 sparse-MLA forward: the un-projected latent
    value path. Computes, per query, absorbed-latent sparse attention over the
    topk-selected cache rows. Intended for Context Encoding with DeepSeek-V3.2 dims
    (L == 512 kv_lora_rank, R == 64, up to 128 heads, topk K a multiple of 128 up to
    ~2048); pair with the o_proj kernel B for V-up + o_proj. Requires B == 1 and S
    divisible by the number of cores.

    Dimensions:
        B: Batch size (must be 1)
        S: Query sequence length (this rank's S-shard)
        S_kv: Cache (key/value) sequence length
        H: Number of attention heads
        L: Latent (kv_lora_rank) dimension (must be 512 = P_MAX * 4)
        R: RoPE head dimension
        K: Number of topk-selected cache rows per query

    Args:
        q_lift_hbm (nl.NkiTensor): [B, S, H, L] bf16, per-head absorbed Q latent.
        q_pe_hbm (nl.NkiTensor): [B, S, H, R] bf16, per-head pre-rotated RoPE queries.
        c_kv_hbm (nl.NkiTensor): [B, S_kv, L] bf16, latent KV cache.
        k_pe_hbm (nl.NkiTensor): [B, S_kv, R] bf16, pre-rotated RoPE key cache.
        topk_indices_hbm (nl.NkiTensor): int32 topk cache-row indices. Flat [B, S, K]
            when topk_tiled is False; partition-tiled
            [num_s_tiles, NUM_TOPK_BATCHES, P_MAX, K // 16] when topk_tiled is True.
            Required for the sparse path; unused (and may be None) when dense=True.
        softmax_scale (float): Scaling factor applied to the attention scores. Must be
            positive. DeepSeek's scale is head_dim**-0.5 times a
            squared mscale correction, so it is always positive in practice.
        topk_tiled (bool): Select the topk_indices_hbm layout (default False = flat).
        q_pos_offset_hbm (nl.NkiTensor): [1, 1] int32/float32 GLOBAL sequence offset of this
            rank's query shard (``cp_rank * S_local`` under Context Parallelism), used by the
            causal mask. A TENSOR rather than a Python int so the framework traces ONE graph
            for every CP rank instead of specializing (and recompiling) per rank. None (the
            default) means offset 0 -- the non-CP case.
        precision (MlaPrecision): precision of the DOWNSTREAM V-up + o_proj kernel (kernel B).
            This kernel's arithmetic is bf16 regardless; ``precision`` selects only the
            ``out_attn`` latent COLUMN LAYOUT so it matches what kernel B reads.
            ``MX`` (default) pre-permutes each head's latent into 4-pack order; ``BF16`` emits
            NATURAL order (the bf16 o_proj transposes on the Tensor Engine and has no MX layout
            to feed). See :func:`_mm2_out_aps`. Every GLM-MoE-DSA dim this kernel constrains
            (L == 512, R == 64, H == 64, topk K == 2048) is already legal, so this layout switch
            is the only change GLM needs here.
        use_sbuf_indirect (bool): select how the topk-selected KV rows are gathered.
            False (default): DMA-INDIRECT gather straight from HBM, per query, via 2D
                ``vector_offset`` DMA batching. SBUF holds only the K gathered rows, so it is
                INDEPENDENT of S_kv -- this is what allows long context. Measured 24% fewer
                cycles than the resident path at S_kv=8192/S_local=128/K=2048 (the resident
                path's cost is not the load, it is the per-query SBUF tensor-indirection gather
                saturating the vector engine). Requires topk_tiled=False.
            True: legacy path -- make the WHOLE gathered cache SBUF-resident, then gather
                columns with SBUF tensor indirection. Costs (L + R) * itemsize bytes/partition,
                so it is capped at S_kv <= 10240 (bf16) or 20480 (fp8); exceeding that raises a
                kernel_assert naming the limit instead of a bare "Stack out of memory".

    Returns:
        out_attn (nl.NkiTensor): [B, S, H * L] bf16 latent attention output, row-major
            h * L + l. Each head's latent columns are pre-permuted into MX 4-pack order when
            ``precision`` is MX, or left in natural order when BF16.

    Notes:
        - S-sharded across cores; under Context Parallelism the framework gathers the
          latent KV to the full S_kv before this kernel is called.
        - The MM2 output's latent column layout is a CROSS-KERNEL CONTRACT with the o_proj
          kernel (and its W_uv load and both refs) selected by ``precision``; see
          :func:`_mm2_out_aps`.

    Pseudocode:
        for q_idx in range(S_shard):
            c_g = c_kv[topk_indices[q_idx]]     # gather K latent rows
            k_pe_g = k_pe[topk_indices[q_idx]]  # gather K RoPE key rows
            scores = q_lift[q_idx] @ c_g.T + q_pe[q_idx] @ k_pe_g.T
            weights = softmax(scores * softmax_scale)
            out_attn[q_idx] = weights @ c_g     # [H, L], 4-pack permuted (MX) or natural (BF16)
    """
    _validate_mla_attention_inputs(
        q_lift_hbm,
        q_pe_hbm,
        c_kv_hbm,
        k_pe_hbm,
        topk_indices_hbm,
        topk_tiled,
        dense=dense,
        q_pos_offset_hbm=q_pos_offset_hbm,
    )
    B, S, H, L = q_lift_hbm.shape
    HL = H * L

    _, n_prgs, prg_id = get_verified_program_sharding_info("mla_sparse_attention_cte_kernel", (0, 1))
    kernel_assert(S % n_prgs == 0, f"S={S} must be divisible by n_prgs={n_prgs}")
    s_per_core = S // n_prgs
    s_start = prg_id * s_per_core

    out_attn_hbm = nl.ndarray((B, S, HL), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    sbm = _new_sbm("sparse_mla_latent_attn")
    # Sparse (indexer topk-gather) and dense (no-indexer, all-keys + causal mask) are separate
    # Dense is the S <= index_topk case where the indexer is skipped;
    # it attends every gathered key with an in-kernel causal mask.
    if dense:
        _attention_stage_dense(
            q_lift_hbm,
            q_pe_hbm,
            c_kv_hbm,
            k_pe_hbm,
            out_attn_hbm,
            softmax_scale,
            sbm,
            s_start,
            s_per_core,
            q_pos_offset_hbm=q_pos_offset_hbm,
            precision=precision,
        )
    else:
        _attention_stage_sparse(
            q_lift_hbm,
            q_pe_hbm,
            c_kv_hbm,
            k_pe_hbm,
            topk_indices_hbm,
            out_attn_hbm,
            softmax_scale,
            sbm,
            s_start,
            s_per_core,
            topk_tiled=topk_tiled,
            q_pos_offset_hbm=q_pos_offset_hbm,
            precision=precision,
            use_sbuf_indirect=use_sbuf_indirect,
        )
    return out_attn_hbm


def _load_kv_cache(c_kv_hbm, k_pe_hbm, R, L, n_l, sbm, kv_sbuf):
    """Make the latent KV cache resident in the [L_partition, S_kv_free] layout the
    attention stages consume: n_l tiles of [P_MAX, S_kv] (latent) plus [R, S_kv] (RoPE key).

    kv_sbuf: optional (c_sb_tiles, k_pe_sb, S_kv) tuple of PRE-GATHERED SBUF KV already in
    this layout (e.g. from a SB2SB CP all-gather). When given, it is used in place and the
    HBM cache load is SKIPPED (c_kv_hbm / k_pe_hbm are then unused). When None, the cache is
    transpose-loaded from HBM ([S_kv, L] -> [L, S_kv]).

    Returns:
        (c_sb_tiles, k_pe_sb, S_kv)
    """
    if kv_sbuf != None:
        return kv_sbuf

    S_kv = c_kv_hbm.shape[1]
    c_sb_tiles = []
    for li in range(n_l):
        c_sb_tiles.append(sbm.alloc_stack((_P_MAX, S_kv), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"c_sb_{li}"))
    k_pe_sb = sbm.alloc_stack((R, S_kv), dtype=nl.bfloat16, buffer=nl.sbuf, name="k_pe_sb")

    for li in range(n_l):
        nisa.dma_transpose(dst=c_sb_tiles[li], src=c_kv_hbm.ap(pattern=[[L, S_kv], [1, _P_MAX]], offset=li * _P_MAX))
    nisa.dma_transpose(dst=k_pe_sb, src=k_pe_hbm.ap(pattern=[[R, S_kv], [1, R]], offset=0))
    return c_sb_tiles, k_pe_sb, S_kv


def _chunks_per_psum_bank(num_chunks, bytes_per_chunk, psum_bank_size):
    """How many chunk transposes to stage in ONE PSUM bank before evicting them as a single copy.

    The staging tiles are far smaller than a bank ([_K_CHUNK, L] bf16 is half a bank, [_K_CHUNK, H]
    an eighth), so evicting per chunk pays an instruction issue -- and makes the PE wait on a bank
    turnaround -- for every fraction of a bank. Returns the largest DIVISOR of num_chunks that
    still fits a bank, so the batched loop covers every chunk with no tail.

    Capped at half the chunks: folding every chunk into ONE eviction collapses the two-bank
    rotation, leaving the transposes nothing to overlap with and MM2 waiting on the last one
    (measured H=64: 0.96 -> 1.13 ms). Keeping >= 2 batches preserves the rotation.
    """
    cap = min(num_chunks, psum_bank_size // bytes_per_chunk)
    if num_chunks > 1:
        cap = min(cap, num_chunks // 2)
    cap = max(1, cap)
    while cap > 1 and num_chunks % cap != 0:
        cap -= 1
    return cap


def _qap(tensor, pattern, elem_offset, s_dim, qbase_sb):
    """HBM access pattern for one query, with the query base optionally supplied at RUNTIME.

    ``elem_offset`` is the trace-time part (the query's offset within its chunk, times the
    per-query stride). When the query loop is a device loop, ``qbase_sb`` is a ``[1, 1]`` int32
    SBUF tile holding the chunk's base query index, and ``s_dim`` names the tensor dimension whose
    stride scales it -- i.e. the S dimension, so the hardware adds ``base * per_query_stride``
    without the kernel ever computing that product. ``qbase_sb=None`` is the fully unrolled case,
    where the whole offset is a compile-time constant.
    """
    if qbase_sb is None:
        return tensor.ap(pattern=pattern, offset=elem_offset)
    return tensor.ap(pattern=pattern, offset=elem_offset, scalar_offset=qbase_sb, indirect_dim=s_dim)


def _load_q_pos_offset(q_pos_offset_hbm, sbm):
    """Broadcast the global query-shard offset to a [P_MAX, 1] float32 SBUF column.

    The offset arrives as a TENSOR (not a kernel constant) so the framework traces one graph
    for every CP rank instead of specializing per rank. Both causal masks consume it as a
    per-partition scalar operand -- tensor_scalar's operand and range_select's bound both
    require float32, one element per partition -- so the int is cast on a compute engine
    (DMA cannot cast) and replicated across all partitions once, outside the query loop.

    q_pos_offset_hbm None is the non-CP case: offset 0.
    """
    q_pos_offset_sb = sbm.alloc_stack((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="q_pos_offset_sb")
    if q_pos_offset_hbm is None:
        nisa.memset(q_pos_offset_sb, value=0.0)
    else:
        q_pos_offset_in = sbm.alloc_stack((1, 1), dtype=q_pos_offset_hbm.dtype, buffer=nl.sbuf, name="q_pos_offset_in")
        nisa.dma_copy(dst=q_pos_offset_in, src=q_pos_offset_hbm)
        nisa.tensor_copy(dst=q_pos_offset_sb[0:1, :], src=q_pos_offset_in, engine=nisa.engine.vector)
        stream_shuffle_broadcast(src=q_pos_offset_sb[0:1, :], dst=q_pos_offset_sb)
    return q_pos_offset_sb


def _attention_stage_sparse(
    q_lift_hbm,
    q_pe_hbm,
    c_kv_hbm,
    k_pe_hbm,
    topk_indices_hbm,
    out_attn_hbm,
    softmax_scale,
    sbm,
    s_start,
    s_per_core,
    kv_sbuf=None,
    topk_tiled=False,
    q_pos_offset_hbm=None,
    precision=MlaPrecision.MX,
    use_sbuf_indirect=False,
):
    """Sparse latent + RoPE attention, S-sharded.

    Computes attn_latent[H, L] per query for queries [s_start, s_start+s_per_core) and
    writes them to out_attn_hbm[B, S, H*L] (row-major h*L + l). Single source of truth
    for the standalone attention kernel (and reusable by a fused parent).

    The per-query work lives in the nested ``_query_body``; it is driven either fully unrolled or
    by an ``nl.fori_loop`` device loop over query chunks, chosen by ``s_per_core`` against
    ``_UNROLL_MAX_S_LOCAL``. ``_query_body`` takes ``qbase_sb`` (the runtime chunk base, or None
    when unrolled) and ``qpos_eff`` (the causal mask's runtime offset, with the chunk base already
    folded in).

    Causal re-mask of the gathered topk keys (mirrors the reference's ``index_mask +=
    causal_mask``): a query at GLOBAL position ``q_global = q_pos_offset + q_idx`` with
    fewer than K valid causal keys gets its topk padded by the indexer with FUTURE
    positions (key position > q_global). Those keys are set to -inf here so they drop out
    of the softmax; it is a no-op for queries whose topk are all causally valid. Under CP
    the caller passes the runtime tensor ``q_pos_offset_hbm = cp_rank * S_local`` so
    q_global is the true global query position (the gathered c_kv / k_pe are the full
    all-gathered S_kv).

    kv_sbuf: optional PRE-GATHERED SBUF KV, see _load_kv_cache.
    """
    B, S, H, L = q_lift_hbm.shape
    R = q_pe_hbm.shape[3]
    """
    topk_indices layout: FLAT [B, S, K] (topk_tiled=False) -> K = shape[2];
    TILED [num_s_tiles, NUM_TOPK_BATCHES, P_MAX, K//16] (topk_tiled=True, the
    split-fix from the indexer) -> K = shape[3] * 16. In TILED mode query
    (s_tile*128 + t*8 + g) reads its [16, K//16] tile straight from partitions
    [16g,16g+16) of block [s_tile, t] -- no flat re-tile (see load below).
    """
    K = (topk_indices_hbm.shape[3] * 16) if topk_tiled else topk_indices_hbm.shape[2]
    HL = H * L
    n_l = L // _P_MAX

    num_k_chunks = K // _K_CHUNK
    mm1_tile = min(_MM1_TILE, K)
    num_mm1_tiles = K // mm1_tile
    sm_tile = min(_SM_TILE, K)
    num_sm_tiles = K // sm_tile

    q_s_stride = H * L
    q_pe_s_stride = H * R
    topk_s_stride = K
    PSUM_BANK_SIZE = _get_psum_bank_size()

    sbm.open_scope()

    S_kv = c_kv_hbm.shape[1]
    n_idx = K // _P_MAX  # gathered-row batches for the DMA path; == num_k_chunks

    if use_sbuf_indirect:
        _limit, _itemsize = _resident_s_kv_limit(c_kv_hbm)
        kernel_assert(
            S_kv <= _limit,
            f"[MLA sparse attn] use_sbuf_indirect=True keeps the WHOLE gathered cache SBUF-resident "
            f"({(L + R) * _itemsize} B/partition per position), which caps S_kv at {_limit} for a "
            f"{_itemsize}-byte cache; got S_kv={S_kv}. Pass use_sbuf_indirect=False to gather from "
            f"HBM per query instead (SBUF cost then independent of S_kv).",
        )
        c_sb_tiles, k_pe_sb, S_kv = _load_kv_cache(c_kv_hbm, k_pe_hbm, R, L, n_l, sbm, kv_sbuf)
        """
        Cache-position column [0, 1, ..., S_kv-1] (same on every partition). Gathering it with
        the SAME idx_u16 the KV gather uses yields each score column's ORIGINAL key position in
        the exact gather order, so the causal re-mask below stays column-aligned regardless of
        the topk layout. uint16 matches the gather index dtype.
        """
        pos_col = sbm.alloc_stack((_P_MAX, S_kv), dtype=nl.uint16, buffer=nl.sbuf, name="pos_col")
        nisa.iota(dst=pos_col, pattern=[[1, S_kv]], offset=0, channel_multiplier=0)
    else:
        """
        DMA-indirect path: nothing resident. The gather needs its index in the 2D vector_offset
        arrangement -- P must be exactly 128 and the hardware reads the index column-major, so
        slot s lands at idx_dma[s % 128, s // 128]. Building it as idx_dma[p, m] = topk[m*128 + p]
        therefore puts gathered slot m*128+p at c_g_nat[p, m], i.e. slots in natural topk order.
        The TILED topk layout has no cheap rearrangement into that form, so it stays on the
        resident path.
        """
        kernel_assert(
            not topk_tiled,
            "[MLA sparse attn] topk_tiled=True is only supported with use_sbuf_indirect=True; the "
            "DMA-indirect gather needs the flat [B, S, K] topk to build its column-major index.",
        )
        kernel_assert(
            K % _P_MAX == 0,
            f"[MLA sparse attn] the DMA-indirect gather batches K in {_P_MAX}-row groups, so K must "
            f"be a multiple of {_P_MAX}; got K={K}.",
        )
        c_sb_tiles, k_pe_sb, pos_col = None, None, None
    q_pos_offset_sb = _load_q_pos_offset(q_pos_offset_hbm, sbm)

    """
    Query-chunk size for the DEVICE loop (see the driver at the bottom of this function).

    ``_q_chunk == s_per_core`` selects the fully unrolled driver -- the faster code, kept for every
    S_local that still compiles unrolled. Otherwise the queries are emitted as ``_q_chunk`` unrolled
    iterations inside one nl.fori_loop trip, so program size is O(_q_chunk) rather than
    O(s_per_core). _q_chunk must DIVIDE s_per_core so the loop has no tail.

    The resident path (use_sbuf_indirect) is pinned to the unrolled driver because its topk_tiled
    branch does trace-time arithmetic on the absolute query index; it is capped at S_kv <= 10240
    anyway, so it cannot reach the sizes that need a device loop.
    """
    if use_sbuf_indirect or s_per_core <= _UNROLL_MAX_S_LOCAL:
        _q_chunk = s_per_core
    else:
        _q_chunk = min(_QUERY_CHUNK, s_per_core)
    while s_per_core % _q_chunk != 0:
        _q_chunk -= 1
    if _q_chunk < s_per_core:
        qbase_i32 = sbm.alloc_stack((1, 1), dtype=nl.int32, buffer=nl.sbuf, name="qbase_i32")
        qbase_f32 = sbm.alloc_stack((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qbase_f32")
        qpos_chunk_sb = sbm.alloc_stack((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="qpos_chunk_sb")
    else:
        qbase_i32, qbase_f32, qpos_chunk_sb = None, None, None

    ti_f = K // 16
    NUM_INPUT_BUFFERS = 2
    idx_i32_bufs, idx_u16_bufs, q_lift_t_bufs, q_pe_t_bufs, c_g_bufs, k_pe_g_bufs = [], [], [], [], [], []
    gathered_pos_bufs = []
    c_g_nat_bufs, kpe_nat_bufs, idx_dma_bufs, gpos_i32_bufs, gpos_nat_bufs = [], [], [], [], []
    q_lift_nat_bufs, q_pe_nat_bufs = [], []
    for buf_idx in range(NUM_INPUT_BUFFERS):
        idx_i32_bufs.append(sbm.alloc_stack((_P_MAX, ti_f), dtype=nl.int32, buffer=nl.sbuf, name=f"idx_i32_{buf_idx}"))
        idx_u16_bufs.append(sbm.alloc_stack((_P_MAX, ti_f), dtype=nl.uint16, buffer=nl.sbuf, name=f"idx_u16_{buf_idx}"))
        q_lift_t_bufs.append(
            sbm.alloc_stack((_P_MAX, n_l, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qlt_{buf_idx}")
        )
        q_pe_t_bufs.append(sbm.alloc_stack((R, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qpt_{buf_idx}"))
        if _q_chunk < s_per_core:
            # Natural-layout q staging for the device-loop path (see the q load in the body).
            q_lift_nat_bufs.append(sbm.alloc_stack((H, L), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"qln_{buf_idx}"))
            q_pe_nat_bufs.append(sbm.alloc_stack((H, R), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"qpn_{buf_idx}"))
        c_g_bufs.append(sbm.alloc_stack((_P_MAX, n_l, K), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"c_g_{buf_idx}"))
        k_pe_g_bufs.append(sbm.alloc_stack((R, K), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_pe_g_{buf_idx}"))
        gathered_pos_bufs.append(
            sbm.alloc_stack((_P_MAX, K), dtype=nl.uint16, buffer=nl.sbuf, name=f"gathered_pos_{buf_idx}")
        )
        if not use_sbuf_indirect:
            """
            DMA-path buffers. c_g_nat / kpe_nat hold the gathered rows NATURAL ([slot, feature],
            each row contiguous -> 1024 B per descriptor). c_g_nat doubles as MM2's moving operand
            (MM2 contracts over K, so it wants keys-on-partition) -- which is why the DMA path
            drops c_g_t_all entirely; only MM1 needs the [L, K] transpose.
            idx_dma is the column-major 2D vector_offset; gpos_i32 carries the key positions for
            the causal re-mask straight from topk (no pos_col gather needed).
            """
            c_g_nat_bufs.append(
                sbm.alloc_stack((_P_MAX, n_idx, L), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"c_g_nat_{buf_idx}")
            )
            kpe_nat_bufs.append(
                sbm.alloc_stack((_P_MAX, n_idx, R), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"kpe_nat_{buf_idx}")
            )
            idx_dma_bufs.append(
                sbm.alloc_stack((_P_MAX, n_idx), dtype=nl.uint32, buffer=nl.sbuf, name=f"idx_dma_{buf_idx}")
            )
            gpos_i32_bufs.append(sbm.alloc_stack((H, K), dtype=nl.int32, buffer=nl.sbuf, name=f"gpos_i32_{buf_idx}"))
            gpos_nat_bufs.append(sbm.alloc_stack((H, K), dtype=nl.int32, buffer=nl.sbuf, name=f"gpos_nat_{buf_idx}"))

    scores_sb = sbm.alloc_stack((H, K), dtype=nl.float32, buffer=nl.sbuf, name="scores_sb")
    mask_add = sbm.alloc_stack((H, K), dtype=nl.float32, buffer=nl.sbuf, name="mask_add")
    p = sbm.alloc_stack((H, K), dtype=nl.bfloat16, buffer=nl.sbuf, name="p")
    neg_row_max = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="neg_row_max")
    exp_bias = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="exp_bias")
    row_sum = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="row_sum")
    recip = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="recip")
    out_bf16 = sbm.alloc_stack((H, L), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_bf16")
    # MM2 emits latent-on-partition out_T; staged here so nc_transpose (stationary must be
    # SBUF) can put it back into the [H, L] the HBM output contract requires.
    out_t_sb = sbm.alloc_stack((_P_MAX, n_l, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name="out_t_sb")
    c_g_t_all = (
        sbm.alloc_stack((_K_CHUNK, num_k_chunks, L), dtype=nl.bfloat16, buffer=nl.sbuf, name="c_g_t_all")
        if use_sbuf_indirect
        else None
    )
    p_t_all = sbm.alloc_stack((_K_CHUNK, num_k_chunks, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="p_t_all")

    def _query_body(s_local, qbase_sb, qpos_eff):
        """One query. ``s_local`` is trace-time; ``qbase_sb`` is the runtime chunk base
        ([1, 1] int32 SBUF) or None when the whole query loop is unrolled."""
        q_loc = s_start + s_local
        buf_idx = s_local % NUM_INPUT_BUFFERS
        idx_i32 = idx_i32_bufs[buf_idx]
        idx_u16 = idx_u16_bufs[buf_idx]
        q_lift_t = q_lift_t_bufs[buf_idx]
        q_pe_t = q_pe_t_bufs[buf_idx]
        q_lift_nat = q_lift_nat_bufs[buf_idx] if _q_chunk < s_per_core else None
        q_pe_nat = q_pe_nat_bufs[buf_idx] if _q_chunk < s_per_core else None
        c_g = c_g_bufs[buf_idx]
        k_pe_g = k_pe_g_bufs[buf_idx]
        gathered_pos = gathered_pos_bufs[buf_idx]
        c_g_nat = c_g_nat_bufs[buf_idx] if not use_sbuf_indirect else None
        kpe_nat = kpe_nat_bufs[buf_idx] if not use_sbuf_indirect else None
        idx_dma = idx_dma_bufs[buf_idx] if not use_sbuf_indirect else None
        gpos_i32 = gpos_i32_bufs[buf_idx] if not use_sbuf_indirect else None
        gpos_nat = gpos_nat_bufs[buf_idx] if not use_sbuf_indirect else None

        if not use_sbuf_indirect:
            """
            idx_dma[p, m] = topk[m*128 + p] -- one strided DMA (partition stride 1 over the topk
            row, m stride 128), which is exactly the column-major arrangement 2D vector_offset
            wants, so gathered slot m*128+p lands at c_g_nat[p, m] in natural topk order.
            gpos_i32 is the same topk broadcast across the H score partitions, giving the causal
            re-mask each column's key position without a pos_col gather.
            """
            _topk_2d = topk_indices_hbm.reshape((B * S, K))
            nisa.dma_copy(
                dst=idx_dma,
                src=_qap(_topk_2d, [[1, _P_MAX], [_P_MAX, n_idx]], q_loc * topk_s_stride, 0, qbase_sb),
            )
            """
            The mask must line up with the SCORE COLUMNS, and those carry the gather's slot
            permutation: the DMA writes fetched rows partition-major, so c_g_nat[p, m] holds topk
            slot p*n_idx + m, and MM1 lands it at score column m*128 + p. (MM1 and MM2 are
            self-consistent under that permutation -- MM2 pairs p_t chunk m with c_g_nat[:, m, :]
            -- and attention pools keys order-invariantly, so only the mask needs to follow it.)
            Hence score column m*128+p needs position topk[p*n_idx + m]: one strided 3D DMA over
            (H broadcast, m, p) with src stride n_idx on p and 1 on m.
            """
            """
            Two steps ON PURPOSE. Doing the permuted read as a single strided DMA from HBM emits
            one descriptor per 4-byte element (~131k per query) and measured ~64 M cycles/query --
            it was the whole cost of this path. Instead: DMA the topk row CONTIGUOUSLY (inner run
            = K * 4 B, stride-0 partition broadcast, so descriptors are large), then do the
            permutation as an SBUF->SBUF strided copy on the vector engine, where a strided access
            pattern is free.
            """
            nisa.dma_copy(
                dst=gpos_nat[0:H, 0:K],
                src=_qap(_topk_2d, [[0, H], [1, K]], q_loc * topk_s_stride, 0, qbase_sb),
            )
            nisa.tensor_copy(
                dst=gpos_i32.ap(pattern=[[K, H], [_P_MAX, n_idx], [1, _P_MAX]], offset=0),
                src=gpos_nat.ap(pattern=[[K, H], [1, n_idx], [n_idx, _P_MAX]], offset=0),
                engine=nisa.engine.vector,
            )
        elif topk_tiled:
            """
            Read the indexer's natural partition tile directly (no flat re-tile, no
            scatter). Safe: sparse attn pools the K keys order-invariantly. Query q_loc's
            [16, ti_f] tile is at partitions [16g, 16g+16) of block [s_tile, t] in
            topk_tiled_hbm[num_s_tiles, NUM_TOPK_BATCHES, P_MAX, ti_f].
            """
            _s_tile = q_loc // _P_MAX
            _q_local = q_loc % _P_MAX
            _t = _q_local // 8
            _g = _q_local % 8
            nisa.dma_copy(
                dst=idx_i32[0:16, :],
                src=topk_indices_hbm[_s_tile, _t, 16 * _g : 16 * _g + 16, :],
            )
        else:
            nisa.dma_copy(
                dst=idx_i32[0:16, :],
                src=topk_indices_hbm.ap(pattern=[[ti_f, 16], [1, ti_f]], offset=q_loc * topk_s_stride),
            )
        if use_sbuf_indirect:
            nisa.tensor_scalar(
                idx_u16[0:16, :], idx_i32[0:16, :], op0=nl.multiply, operand0=1, engine=nisa.engine.vector
            )
            nisa.nc_stream_shuffle(dst=idx_u16[0:32, :], src=idx_u16[0:32, :], shuffle_mask=_TI_REPLICATE_MASK)
            for quad in range(1, _P_MAX // 32):
                nisa.dma_copy(dst=idx_u16[quad * 32 : quad * 32 + 32, :], src=idx_u16[0:32, :])

        if qbase_sb is None:
            # Load q_lift transposed to latent-on-partition in ONE dma_transpose: 3D source
            # (H, n_l, 128_latent) transposed [2,1,0] -> q_lift_t [128_latent, n_l, H].
            nisa.dma_transpose(
                dst=q_lift_t,
                src=q_lift_hbm.ap(pattern=[[L, H], [_P_MAX, n_l], [1, _P_MAX]], offset=q_loc * q_s_stride),
            )
            nisa.dma_transpose(dst=q_pe_t, src=q_pe_hbm.ap(pattern=[[R, H], [1, R]], offset=q_loc * q_pe_s_stride))
        else:
            """
            Same two loads, but NATURAL DMA + Tensor-Engine transpose instead of dma_transpose.

            Inside a device loop a dma_transpose off a kernel INPUT is rejected:
              [NCC_IBIR251] Static DMA that is in a dynamic loop can only read from Internal
              tensor, but it reads from Input tensor
            A dma_copy carrying a runtime scalar_offset counts as dynamic and is accepted (the
            topk / out_attn DMAs above and below prove it), but dma_transpose is not, so the
            transpose has to move off the DMA engine. The natural load is actually the friendlier
            DMA -- one contiguous L*2 B (resp. R*2 B) run per head -- and the transpose costs
            n_l + 1 nc_transposes of dst free H, i.e. ~(n_l+1)*(H + instr) PE cycles per query
            (~1.4 % of this kernel's PE time at the GLM dims).
            """
            nisa.dma_copy(dst=q_lift_nat, src=_qap(q_lift_hbm, [[L, H], [1, L]], q_loc * q_s_stride, 1, qbase_sb))
            nisa.dma_copy(dst=q_pe_nat, src=_qap(q_pe_hbm, [[R, H], [1, R]], q_loc * q_pe_s_stride, 1, qbase_sb))
            qlt_psum = nl.ndarray(
                (_P_MAX, n_l * H), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 6 * PSUM_BANK_SIZE + 1024)
            )
            for li in range(n_l):
                nisa.nc_transpose(qlt_psum[:, li * H : (li + 1) * H], q_lift_nat[:, li * _P_MAX : (li + 1) * _P_MAX])
            nisa.tensor_copy(dst=q_lift_t, src=qlt_psum, engine=nisa.engine.vector)
            qpt_psum = nl.ndarray((R, H), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 7 * PSUM_BANK_SIZE + 1024))
            nisa.nc_transpose(qpt_psum, q_pe_nat)
            nisa.tensor_copy(dst=q_pe_t, src=qpt_psum, engine=nisa.engine.scalar)

        """
        TI gather via tensor_copy (supports indirection on BOTH vector + scalar on
        gen4). Alternate the engine across latent tiles so the n_l gathers overlap
        instead of serializing — the vector engine is the kernel-A bottleneck (~80%)
        while scalar idles (~27%). tensor_copy needs no ti_ones multiply operand.
        """
        if use_sbuf_indirect:
            for li in range(n_l):
                c_g_view = c_sb_tiles[li].indirect(idx_u16, num_elem=K)
                g_engine = nisa.engine.vector if li % 2 == 0 else nisa.engine.scalar
                nisa.tensor_copy(dst=c_g[:, li, :], src=c_g_view, engine=g_engine)
            k_pe_view = k_pe_sb.indirect(idx_u16[0:R, :], num_elem=K)
            nisa.tensor_copy(dst=k_pe_g, src=k_pe_view, engine=nisa.engine.scalar)

            # Gather each score column's ORIGINAL key position (same idx_u16 -> same gather order
            # as c_g), so the causal re-mask after MM1 is column-aligned regardless of topk layout.
            pos_view = pos_col.indirect(idx_u16, num_elem=K)
            nisa.tensor_copy(dst=gathered_pos, src=pos_view, engine=nisa.engine.vector)
        else:
            """
            One DMA-indirect call gathers all K rows of c_kv (and of k_pe) straight from HBM.
            Indirect DMA is SWDGE-only -- hardware DGE is rejected -- so descriptors are software
            generated; each descriptor still moves a full contiguous row (L * 2 = 1024 B), which
            is what keeps this efficient.
            """
            nisa.dma_copy(
                dst=c_g_nat,
                src=c_kv_hbm.reshape((S_kv, L)).ap(
                    pattern=[[L, _P_MAX * n_idx], [1, L]], vector_offset=idx_dma, indirect_dim=0
                ),
                dge_mode=dge_mode.swdge,
            )
            nisa.dma_copy(
                dst=kpe_nat,
                src=k_pe_hbm.reshape((S_kv, R)).ap(
                    pattern=[[R, _P_MAX * n_idx], [1, R]], vector_offset=idx_dma, indirect_dim=0
                ),
                dge_mode=dge_mode.swdge,
            )
            """
            MM1 contracts over L, so it needs latent-on-partition [L, K]; the gather landed
            [slot, feature]. Transpose it here -- n_idx * n_l tiles, the SAME count the resident
            path spends transposing c_g the other way for MM2 (which this path gets for free from
            c_g_nat). Banks 4/5 rotate, matching the freed c_g_t staging banks.
            """
            for m in range(n_idx):
                cgn_psum = nl.ndarray(
                    (_P_MAX, L), dtype=nl.bfloat16, buffer=nl.psum, address=(0, (4 + m % 2) * PSUM_BANK_SIZE)
                )
                for li in range(n_l):
                    nisa.nc_transpose(
                        cgn_psum[:, li * _P_MAX : (li + 1) * _P_MAX],
                        c_g_nat[:, m, li * _P_MAX : (li + 1) * _P_MAX],
                    )
                t_engine = nisa.engine.vector if m % 2 == 0 else nisa.engine.scalar
                for li in range(n_l):
                    nisa.tensor_copy(
                        dst=c_g[:, li, m * _P_MAX : (m + 1) * _P_MAX],
                        src=cgn_psum[:, li * _P_MAX : (li + 1) * _P_MAX],
                        engine=t_engine,
                    )
                # k_pe: [slot, R] -> [R, slot] for the same K slice.
                kpe_psum = nl.ndarray(
                    (R, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum, address=(0, (6 + m % 2) * PSUM_BANK_SIZE)
                )
                nisa.nc_transpose(kpe_psum, kpe_nat[:, m, 0:R])
                nisa.tensor_copy(dst=k_pe_g[:, m * _P_MAX : (m + 1) * _P_MAX], src=kpe_psum, engine=nisa.engine.scalar)

        scores_psum = nl.ndarray((H, K), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
        for mm1_idx in range(num_mm1_tiles):
            off = mm1_idx * mm1_tile
            for li in range(n_l):
                nisa.nc_matmul(
                    scores_psum[:, off : off + mm1_tile],
                    q_lift_t[:, li, :],
                    c_g[:, li, off : off + mm1_tile],
                    accumulate=(li > 0),
                )
            nisa.nc_matmul(
                scores_psum[:, off : off + mm1_tile], q_pe_t, k_pe_g[:, off : off + mm1_tile], accumulate=True
            )

        nisa.tensor_scalar(
            mask_add,
            gathered_pos[0:H, :] if use_sbuf_indirect else gpos_i32[0:H, :],
            op0=nl.subtract,
            operand0=qpos_eff[0:H, :],
            op1=nl.greater,
            operand1=q_loc,
            engine=nisa.engine.vector,
        )
        nisa.tensor_scalar(mask_add, mask_add, op0=nl.multiply, operand0=_FLOAT32_MIN, engine=nisa.engine.scalar)
        nisa.tensor_tensor(scores_sb, scores_psum, mask_add, op=nl.add)

        """
        c_g transpose for MM2 (keys-on-partition), hoisted before softmax (depends only on
        the gather, not on p) so the TE transpose overlaps softmax latency.
        """
        if use_sbuf_indirect:
            # Resident path only: build MM2's keys-on-partition operand. The DMA path already has it
            # (c_g_nat), so it skips this whole stage.
            cgt_per_bank = _chunks_per_psum_bank(num_k_chunks, L * 2, PSUM_BANK_SIZE)
            for batch_idx in range(num_k_chunks // cgt_per_bank):
                par = batch_idx % 2
                c_g_t_psum = nl.ndarray(
                    (_K_CHUNK, cgt_per_bank * L),
                    dtype=nl.bfloat16,
                    buffer=nl.psum,
                    address=(0, (4 + par) * PSUM_BANK_SIZE),
                )
                for j in range(cgt_per_bank):
                    ks = (batch_idx * cgt_per_bank + j) * _K_CHUNK
                    for li in range(n_l):
                        nisa.nc_transpose(
                            c_g_t_psum[:, j * L + li * _P_MAX : j * L + (li + 1) * _P_MAX],
                            c_g[:, li, ks : ks + _K_CHUNK],
                        )
                e_engine = nisa.engine.vector if batch_idx % 2 == 0 else nisa.engine.scalar
                nisa.tensor_scalar(
                    c_g_t_all[:, batch_idx * cgt_per_bank : (batch_idx + 1) * cgt_per_bank, :],
                    c_g_t_psum,
                    op0=nl.multiply,
                    operand0=1.0,
                    engine=e_engine,
                )

        nisa.tensor_reduce(neg_row_max, op=nl.maximum, data=scores_sb, axis=1, negate=True)
        nisa.tensor_scalar(exp_bias, neg_row_max, op0=nl.multiply, operand0=softmax_scale, engine=nisa.engine.vector)
        for sm_idx in range(num_sm_tiles):
            so = sm_idx * sm_tile
            nisa.activation(
                dst=p[:, so : so + sm_tile],
                op=nl.exp,
                data=scores_sb[:, so : so + sm_tile],
                bias=exp_bias,
                scale=softmax_scale,
                reduce_op=nl.add,
                reduce_res=row_sum if sm_idx == num_sm_tiles - 1 else None,
                reduce_cmd=nisa.reduce_cmd.reset_reduce if sm_idx == 0 else nisa.reduce_cmd.reduce,
            )
        nisa.reciprocal(recip, row_sum)

        # Same batching for p_t: a [_K_CHUNK, H] bf16 tile is only H*2 bytes (1/8 of a bank at
        # H=128), so stage a full bank of chunks per eviction.
        pt_per_bank = _chunks_per_psum_bank(num_k_chunks, H * 2, PSUM_BANK_SIZE)
        for batch_idx in range(num_k_chunks // pt_per_bank):
            par = batch_idx % 2
            p_t_psum = nl.ndarray(
                (_K_CHUNK, pt_per_bank * H), dtype=nl.bfloat16, buffer=nl.psum, address=(0, (1 + par) * PSUM_BANK_SIZE)
            )
            for j in range(pt_per_bank):
                ks = (batch_idx * pt_per_bank + j) * _K_CHUNK
                nisa.nc_transpose(p_t_psum[:, j * H : (j + 1) * H], p[:, ks : ks + _K_CHUNK])
            nisa.tensor_scalar(
                p_t_all[:, batch_idx * pt_per_bank : (batch_idx + 1) * pt_per_bank, :],
                p_t_psum,
                op0=nl.multiply,
                operand0=1.0,
                engine=nisa.engine.scalar,
            )

        # MM2: out_attn[h, d] = sum_j p[h,j] * c_g[d,j], contracting over the K gathered keys.
        pvt_psum = []
        for li in range(n_l):
            pvt_psum.append(
                nl.ndarray((_P_MAX, H), dtype=nl.float32, buffer=nl.psum, address=(0, (n_l + li) * PSUM_BANK_SIZE))
            )
        for chunk_idx in range(num_k_chunks):
            # Keys-on-partition KV: c_g_t_all on the resident path, and the natural gather
            # output itself on the DMA path (n_idx == num_k_chunks, c_g_nat is [slot, feature]).
            _mm2_kv = c_g_t_all[:, chunk_idx, :] if use_sbuf_indirect else c_g_nat[:, chunk_idx, :]
            for li in range(n_l):
                nisa.nc_matmul(
                    pvt_psum[li],
                    _mm2_kv[:, li * _P_MAX : (li + 1) * _P_MAX],
                    p_t_all[:, chunk_idx, :],
                    accumulate=(chunk_idx > 0),
                )

        # out_T [L, H] -> [H, L]. nc_transpose's stationary must come from SBUF, so evict
        # first; the extra bf16 rounding is on the UN-normalized sum, and the final
        # multiply by recip preserves relative error (~0.4 %, vs a 5e-2 tolerance).
        for li in range(n_l):
            nisa.tensor_copy(
                dst=out_t_sb[:, li, :],
                src=pvt_psum[li],
                engine=nisa.engine.vector if li % 2 == 0 else nisa.engine.scalar,
            )
        out_psum = nl.ndarray((H, L), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 0))
        for li in range(n_l):
            nisa.nc_transpose(out_psum[:, li * _P_MAX : (li + 1) * _P_MAX], out_t_sb[:, li, :])

        # CROSS-KERNEL LAYOUT CONTRACT: 4-pack permuted (MX) or natural (BF16); see _mm2_out_aps.
        _out_ap, _psum_ap = _mm2_out_aps(out_bf16, out_psum, H, L, precision)
        nisa.tensor_scalar(
            _out_ap,
            _psum_ap,
            op0=nl.multiply,
            operand0=recip,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(dst=_qap(out_attn_hbm, [[L, H], [1, L]], q_loc * HL, 1, qbase_sb), src=out_bf16)

    if _q_chunk >= s_per_core:
        for s_local in range(s_per_core):
            _query_body(s_local, None, q_pos_offset_sb)
    else:
        # DEVICE loop over query chunks.
        def _query_chunk(_qbase_reg):
            nisa.register_store(dst=qbase_i32, src=_qbase_reg)
            nisa.tensor_copy(dst=qbase_f32[0:1, :], src=qbase_i32, engine=nisa.engine.vector)
            stream_shuffle_broadcast(src=qbase_f32[0:1, :], dst=qbase_f32)
            nisa.tensor_tensor(qpos_chunk_sb, q_pos_offset_sb, qbase_f32, op=nl.add)
            for s_local in range(_q_chunk):
                _query_body(s_local, qbase_i32, qpos_chunk_sb)

        nl.fori_loop(0, s_per_core, _query_chunk, step=_q_chunk)

    sbm.close_scope()


def _attention_stage_dense(
    q_lift_hbm,
    q_pe_hbm,
    c_kv_hbm,
    k_pe_hbm,
    out_attn_hbm,
    softmax_scale,
    sbm,
    s_start,
    s_per_core,
    kv_sbuf=None,
    q_pos_offset_hbm=None,
    precision=MlaPrecision.MX,
):
    """Dense latent + RoPE attention (no indexer), S-sharded, causal.

    The S <= index_topk case: the indexer is skipped and every query attends ALL S_kv keys,
    so there is NO topk gather -- MM1 reads the resident cache (c_sb_tiles / k_pe_sb) directly
    and K == S_kv. Causality is applied per query as a range_select on the scores: key column j
    is set to -inf when j > q_global (= q_pos_offset + s_start + s_local). range_select takes its
    bound as a runtime per-partition tile, which affine_select (compile-time affine offset only)
    cannot do; it also evicts PSUM -> SBUF and reduces the row max in the same instruction.
    Split from _attention_stage_sparse so the dense path can be optimized independently without
    touching the sparse gather flow.

    Same output contract as the sparse stage: out_attn_hbm[B, S, H*L], each head's latent
    columns pre-permuted into MX 4-pack order (o_proj / W_uv / refs depend on it).

    kv_sbuf: optional PRE-GATHERED SBUF KV, see _load_kv_cache.
    """
    B, S, H, L = q_lift_hbm.shape
    R = q_pe_hbm.shape[3]
    HL = H * L
    n_l = L // _P_MAX

    # Dense attends all keys: K == S_kv (a multiple of 128, from the cache).
    S_kv_arg = kv_sbuf[2] if kv_sbuf is not None else c_kv_hbm.shape[1]
    K = S_kv_arg

    num_k_chunks = K // _K_CHUNK
    mm1_tile = min(_MM1_TILE, K)
    num_mm1_tiles = K // mm1_tile
    sm_tile = min(_SM_TILE, K)
    num_sm_tiles = K // sm_tile

    q_s_stride = H * L
    q_pe_s_stride = H * R
    PSUM_BANK_SIZE = _get_psum_bank_size()

    sbm.open_scope()

    c_sb_tiles, k_pe_sb, S_kv = _load_kv_cache(c_kv_hbm, k_pe_hbm, R, L, n_l, sbm, kv_sbuf)
    q_pos_offset_sb = _load_q_pos_offset(q_pos_offset_hbm, sbm)

    # range_select needs two bounds; the causal condition only constrains the upper one, so the
    # lower bound is the always-true key j >= 0.
    zero_bound = sbm.alloc_stack((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="zero_bound")
    nisa.memset(zero_bound, value=0.0)

    # PSUM is bounded (8 HW banks), so bank usage MUST NOT scale with S_kv.
    _PT_ROT = 2  # rotating banks for the p_t / c_g_t chunk transposes
    sc_banks = div_ceil(K * 4, PSUM_BANK_SIZE)  # scores [H,K] fp32 bank span
    per_set_banks = sc_banks + _PT_ROT
    kernel_assert(
        per_set_banks <= _NUM_HW_PSUM_BANKS,
        f"[MLA dense] PSUM overflow: scores({sc_banks}) + p_t({_PT_ROT}) = {per_set_banks} banks "
        f"> {_NUM_HW_PSUM_BANKS} for K=S_kv={K}. Dense S_kv must keep sc_banks <= "
        f"{_NUM_HW_PSUM_BANKS - _PT_ROT} (K <= {(_NUM_HW_PSUM_BANKS - _PT_ROT) * PSUM_BANK_SIZE // 4}).",
    )
    NUM_BUF = max(1, min(3, _NUM_HW_PSUM_BANKS // per_set_banks))

    psum_base = []
    for s in range(NUM_BUF):
        b0 = s * per_set_banks
        psum_base.append({"scores": b0, "p_t": b0 + sc_banks})

    # Query-independent MM2 stationary operand: transpose the resident cache to keys-on-partition
    # once. bf16 eviction on Vector (kept out of the per-query loop entirely).
    c_g_t_all = sbm.alloc_stack((_K_CHUNK, num_k_chunks, L), dtype=nl.bfloat16, buffer=nl.sbuf, name="c_g_t_all")

    # SBUF buffer sets (rotated by pbuf). Per-query-private buffers get NUM_BUF copies; the
    # tiny scalar reductions (neg_row_max/exp_bias/row_sum/recip) are cheap and also duplicated.
    q_lift_t_bufs, q_pe_t_bufs, scores_sb_bufs, p_bufs = [], [], [], []
    p_t_all_bufs, out_bf16_bufs = [], []
    qg_bufs, rmax_bufs, eb_bufs, rs_bufs, rc_bufs = [], [], [], [], []
    for b in range(NUM_BUF):
        q_lift_t_bufs.append(
            sbm.alloc_stack((_P_MAX, n_l, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qlt_{b}")
        )
        q_pe_t_bufs.append(sbm.alloc_stack((R, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qpt_{b}"))
        # range_select writes to SBUF, so scores land in scores_sb already masked (key
        # j > q_global -> -inf) and softmax reads scores_sb.
        scores_sb_bufs.append(sbm.alloc_stack((H, K), dtype=nl.float32, buffer=nl.sbuf, name=f"scores_sb_{b}"))
        p_bufs.append(sbm.alloc_stack((H, K), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"p_{b}"))
        p_t_all_bufs.append(
            sbm.alloc_stack((_K_CHUNK, num_k_chunks, H), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"p_t_all_{b}")
        )
        out_bf16_bufs.append(sbm.alloc_stack((H, L), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"out_bf16_{b}"))
        qg_bufs.append(sbm.alloc_stack((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"q_global_{b}"))
        rmax_bufs.append(sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"row_max_{b}"))
        eb_bufs.append(sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"exp_bias_{b}"))
        rs_bufs.append(sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"row_sum_{b}"))
        rc_bufs.append(sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name=f"recip_{b}"))

    # Hoisted, query-independent: c_g_t[k, l] = c_kv[l, k] (keys-on-partition), evicted once.
    # Runs BEFORE the query loop, so it reuses buffer-0's rotating p_t banks (no live query PSUM
    # yet); each chunk is evicted to c_g_t_all (SBUF), so 2 rotating banks suffice for all chunks.
    _cgt_bank = psum_base[0]["p_t"]
    for chunk_idx in range(num_k_chunks):
        ks = chunk_idx * _K_CHUNK
        par = chunk_idx % _PT_ROT
        c_g_t_psum = nl.ndarray(
            (_K_CHUNK, L), dtype=nl.bfloat16, buffer=nl.psum, address=(0, (_cgt_bank + par) * PSUM_BANK_SIZE)
        )
        for li in range(n_l):
            nisa.nc_transpose(c_g_t_psum[:, li * _P_MAX : (li + 1) * _P_MAX], c_sb_tiles[li][:, ks : ks + _K_CHUNK])
        nisa.tensor_scalar(
            c_g_t_all[:, chunk_idx, :], c_g_t_psum, op0=nl.multiply, operand0=1.0, engine=nisa.engine.vector
        )

    for s_local in range(s_per_core):
        # LOCAL q sharded index
        q_idx = s_start + s_local
        pbuf = s_local % NUM_BUF
        q_lift_t = q_lift_t_bufs[pbuf]
        q_pe_t = q_pe_t_bufs[pbuf]
        scores_sb = scores_sb_bufs[pbuf]
        p = p_bufs[pbuf]
        p_t_all = p_t_all_bufs[pbuf]
        out_bf16 = out_bf16_bufs[pbuf]
        q_global_sb = qg_bufs[pbuf]
        row_max, exp_bias, row_sum, recip = rmax_bufs[pbuf], eb_bufs[pbuf], rs_bufs[pbuf], rc_bufs[pbuf]
        bank = psum_base[pbuf]

        # GLOBAL q sharded position for the causal mask, as the runtime per-partition bound
        # range_select needs: q_pos_offset (runtime) + q_idx (trace-time immediate).
        nisa.tensor_scalar(q_global_sb, q_pos_offset_sb, op0=nl.add, operand0=q_idx, engine=nisa.engine.scalar)

        nisa.dma_transpose(
            dst=q_lift_t,
            src=q_lift_hbm.ap(pattern=[[L, H], [_P_MAX, n_l], [1, _P_MAX]], offset=q_idx * q_s_stride),
        )
        nisa.dma_transpose(dst=q_pe_t, src=q_pe_hbm.ap(pattern=[[R, H], [1, R]], offset=q_idx * q_pe_s_stride))

        # MM1: scores[H, K] = q_lift @ c_kv + q_pe @ k_pe, reading the resident cache directly.
        scores_psum = nl.ndarray((H, K), dtype=nl.float32, buffer=nl.psum, address=(0, bank["scores"] * PSUM_BANK_SIZE))
        for mm1_idx in range(num_mm1_tiles):
            off = mm1_idx * mm1_tile
            for li in range(n_l):
                nisa.nc_matmul(
                    scores_psum[:, off : off + mm1_tile],
                    q_lift_t[:, li, :],
                    c_sb_tiles[li][:, off : off + mm1_tile],
                    accumulate=(li > 0),
                )
            nisa.nc_matmul(
                scores_psum[:, off : off + mm1_tile], q_pe_t, k_pe_sb[:, off : off + mm1_tile], accumulate=True
            )

        nisa.range_select(
            dst=scores_sb,
            on_true_tile=scores_psum,
            on_false_value=_FLOAT32_MIN,
            comp_op0=nl.greater_equal,
            comp_op1=nl.less_equal,
            bound0=zero_bound[0:H, :],
            bound1=q_global_sb[0:H, :],
            reduce_op=nl.maximum,
            reduce_res=row_max,
        )
        nisa.tensor_scalar(exp_bias, row_max, op0=nl.multiply, operand0=-softmax_scale, engine=nisa.engine.scalar)
        for sm_idx in range(num_sm_tiles):
            so = sm_idx * sm_tile
            nisa.activation(
                dst=p[:, so : so + sm_tile],
                op=nl.exp,
                data=scores_sb[:, so : so + sm_tile],
                bias=exp_bias,
                scale=softmax_scale,
                reduce_op=nl.add,
                reduce_res=row_sum if sm_idx == num_sm_tiles - 1 else None,
                reduce_cmd=nisa.reduce_cmd.reset_reduce if sm_idx == 0 else nisa.reduce_cmd.reduce,
            )
        nisa.reciprocal(recip, row_sum)

        for chunk_idx in range(num_k_chunks):
            ks = chunk_idx * _K_CHUNK
            par = chunk_idx % _PT_ROT
            p_t_psum = nl.ndarray(
                (_K_CHUNK, H),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, (bank["p_t"] + par) * PSUM_BANK_SIZE),
            )
            nisa.nc_transpose(p_t_psum, p[:, ks : ks + _K_CHUNK])
            nisa.tensor_scalar(
                p_t_all[:, chunk_idx, :], p_t_psum, op0=nl.multiply, operand0=1.0, engine=nisa.engine.scalar
            )

        # MM2: out_attn[h, d] = sum_j p[h,j] * c[d,j]. Emitted [H, L] (H-on-partition).
        # pv reuses the scores banks (scores is fully consumed by softmax before MM2).
        pv_psum = nl.ndarray((H, L), dtype=nl.float32, buffer=nl.psum, address=(0, bank["scores"] * PSUM_BANK_SIZE))
        for chunk_idx in range(num_k_chunks):
            nisa.nc_matmul(pv_psum, p_t_all[:, chunk_idx, :], c_g_t_all[:, chunk_idx, :], accumulate=(chunk_idx > 0))

        # Same output layout contract as the sparse stage; see _mm2_out_aps.
        _out_ap, _psum_ap = _mm2_out_aps(out_bf16, pv_psum, H, L, precision)
        nisa.tensor_scalar(
            _out_ap,
            _psum_ap,
            op0=nl.multiply,
            operand0=recip,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(dst=out_attn_hbm.ap(pattern=[[L, H], [1, L]], offset=q_idx * HL), src=out_bf16)

    sbm.close_scope()
