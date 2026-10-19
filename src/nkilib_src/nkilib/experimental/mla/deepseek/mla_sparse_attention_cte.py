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
  loop, every address a compile-time constant.
* ``S_local >  _UNROLL_MAX_S_LOCAL``: an ``nl.fori_loop`` DEVICE loop over query chunks, which
  keeps program size O(_q_chunk) instead of O(S_local).
"""

import math

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

_FP8_TP_OUT_STEP = 2  # fp8 nc_transpose must write PSUM with an output element step of 2 (HW)
_FLOAT32_MIN = -3.4028235e38  # most-negative finite fp32; nisa.range_select requires exactly this
# Masked fill for the bf16 mask tile; range_select requires the exact fp32 min, which bf16 cannot hold.
_BF16_MASK_FILL = -3.3895314e38
_UNROLL_MAX_S_LOCAL = 64
_QUERY_CHUNK = 64
# Gather-path input buffers. In the pipelined A(s), C(s - 1) order, query s + 1's gather reuses the
# buffer of a query whose MM2 may still be running, so two would serialize them.
_GATHER_BUFS = 4
# Queries per topk key-position load on the gather path (one DMA per group, double-buffered).
_TOPK_GROUP = 8


# Max gathered S_kv the SBUF-resident path supports, keyed by cache itemsize: the whole cache
# is resident at (L + R) * itemsize bytes/partition.
_SBUF_RESIDENT_MAX_S_KV = {2: 10240, 1: 20480}


def _resident_s_kv_limit(c_kv_hbm):
    """Ceiling for the SBUF-resident path, from the KV cache element size."""
    itemsize = 1 if c_kv_hbm.dtype in (nl.float8_e4m3, nl.float8_e4m3fn, nl.float8_e5m2) else 2
    return _SBUF_RESIDENT_MAX_S_KV[itemsize], itemsize


def _mm2_out_aps(out_bf16, pv_psum, H, L, precision):
    """Access patterns for the MM2 [H, L] eviction, per the consumer kernel's precision.

    MX pre-permutes each head's latent into 4-pack order: natural latent ``l = 4*group + sub``
    is stored at physical column ``sub*(L//4) + group``, matching the MX o_proj's load and
    ``W_uv``. BF16 emits natural column order. Keep in lockstep with the o_proj load, the W_uv
    load, and both refs.
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
    prior_tokens_hbm: nl.NkiTensor = None,
    cp_rank_hbm: nl.NkiTensor = None,
    cp_world_size: int = 1,
    cp_kv_cache_interleave_size: int = 1,
    precision: MlaPrecision = MlaPrecision.MX,
    use_sbuf_indirect: bool = False,
    kv_cache_packed: bool = None,
    kv_cache_fp8_dtype=nl.float8_e4m3,
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
        prior_tokens_hbm (nl.NkiTensor): [1, 1] int32 LOCAL prior/cached token count on this
            rank (``= global_prior / cp_world_size``), used by the causal mask to place this
            chunk's queries in their rank's owned-token stream. A TENSOR (runtime) so ONE graph
            serves every CP rank/step. None (default) = 0 (first chunk / non-CP).
        cp_rank_hbm (nl.NkiTensor): [1, 1] int32 this rank's CP index, used to convert
            owned-token indices to GLOBAL sequence positions for the causal mask. A TENSOR so
            one graph serves all ranks. None (default) = rank 0.
        cp_world_size (int): CP degree W (static). 1 (default) = non-CP: the KV cache is in
            natural order and physical index == global position, so no interleave conversion is
            done and the fast masks are used. > 1: the cache is rank-major INTERLEAVED and topk
            indices are physical slots; the mask converts them (and the query rows) to global
            positions via ``(p // I) % W`` ownership. Must be a power of two.
        cp_kv_cache_interleave_size (int): interleave granularity I (static, == block_size);
            whole-block round-robin. Must be a power of two dividing S_kv // W. Only used when
            cp_world_size > 1.
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

    The packed fp8 cache is auto-detected from the cache tensor's rank: the fused cache is 2-D
    [S_kv, kv_row], the unpacked pair is 3-D [B, S_kv, L]. Pass kv_cache_packed to override.
    """
    if kv_cache_packed is None:
        kv_cache_packed = len(c_kv_hbm.shape) == 2

    _validate_mla_attention_inputs(
        q_lift_hbm,
        q_pe_hbm,
        c_kv_hbm,
        k_pe_hbm,
        topk_indices_hbm,
        topk_tiled,
        dense=dense,
        prior_tokens_hbm=prior_tokens_hbm,
        cp_rank_hbm=cp_rank_hbm,
        cp_world_size=cp_world_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        kv_cache_packed=kv_cache_packed,
    )
    B, S, H, L = q_lift_hbm.shape
    HL = H * L

    _, n_prgs, prg_id = get_verified_program_sharding_info("mla_sparse_attention_cte_kernel", (0, 1))
    kernel_assert(S % n_prgs == 0, f"S={S} must be divisible by n_prgs={n_prgs}")
    s_per_core = S // n_prgs
    s_start = prg_id * s_per_core

    out_attn_hbm = nl.ndarray((B, S, HL), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    sbm = _new_sbm("sparse_mla_latent_attn")
    # Dense is the S <= index_topk case: no indexer, every key attended under an in-kernel causal mask.
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
            prior_tokens_hbm=prior_tokens_hbm,
            cp_rank_hbm=cp_rank_hbm,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            precision=precision,
            kv_cache_packed=kv_cache_packed,
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
            prior_tokens_hbm=prior_tokens_hbm,
            cp_rank_hbm=cp_rank_hbm,
            cp_world_size=cp_world_size,
            cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
            precision=precision,
            use_sbuf_indirect=use_sbuf_indirect,
            kv_cache_packed=kv_cache_packed,
            kv_cache_fp8_dtype=kv_cache_fp8_dtype,
        )
    return out_attn_hbm


def _load_packed_kv_resident(c_kv_hbm, R, L, n_l, sbm, num_k_chunks):
    """Make a PACKED fp8 KV cache resident for the dense (no-indexer) path, DEQUANTIZED once.

    Dense attends every key, so the cache is query-independent: the dequantization is hoisted out
    of the query loop and the per-query path is then identical to the unpacked one.

    Returns (c_sb_tiles, k_pe_sb, S_kv, scale_col, c_g_t_fp8):
      c_sb_tiles  n_l x [_P_MAX, S_kv] bf16, MM1's moving operand, in the pair-transposed grouping
                  (group li = par * n_pair + pi holds channels 2 * (pi * _P_MAX + j) + par on
                  partition j); q_lift is loaded with the matching permutation.
      k_pe_sb     [R, S_kv] bf16, stored unquantized in the row so it needs no scaling.
      scale_col   [_P_MAX, num_k_chunks] fp32, one scale per key with keys on the PARTITION axis.
      c_g_t_fp8   [_K_CHUNK, num_k_chunks, L] bf16, MM2's keys-on-partition stationary, converted
                  straight off the row by DMA.
    """
    S_kv = c_kv_hbm.shape[0]
    kv_row = c_kv_hbm.shape[1]
    kv_lat_bf16 = L // 2
    n_pair = max(1, kv_lat_bf16 // _P_MAX)
    _row_b = kv_row * 2  # row stride in fp8 BYTES

    c_sb_tiles = [
        sbm.alloc_stack((_P_MAX, S_kv), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"c_sb_{li}") for li in range(n_l)
    ]
    k_pe_sb = sbm.alloc_stack((R, S_kv), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name="k_pe_sb")
    # Two copies: fp32 for tensor_scalar's operand0, bf16 for nc_transpose (which cannot change dtype).
    scale_col = sbm.alloc_stack((_P_MAX, num_k_chunks), dtype=nl.float32, buffer=nl.sbuf, name="kv_scale_col")
    scale_bf16 = sbm.alloc_stack((_P_MAX, num_k_chunks), dtype=nl.bfloat16, buffer=nl.sbuf, name="kv_scale_bf16")
    s_bcast = sbm.alloc_stack((_P_MAX, S_kv), dtype=nl.bfloat16, buffer=nl.sbuf, name="kv_scale_bcast")
    pair_tiles = [
        sbm.alloc_stack((_P_MAX, S_kv), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"kv_pair_{pi}")
        for pi in range(n_pair)
    ]
    c_g_t_fp8 = sbm.alloc_stack((_K_CHUNK, num_k_chunks, L), dtype=nl.bfloat16, buffer=nl.sbuf, name="c_g_t_all")
    PSUM_BANK_SIZE = _get_psum_bank_size()

    # Latent as bf16 PAIRS: [S_kv, _P_MAX containers] -> [_P_MAX, S_kv] per pair tile.
    for pi in range(n_pair):
        nisa.dma_transpose(
            dst=pair_tiles[pi], src=c_kv_hbm.ap(pattern=[[kv_row, S_kv], [1, _P_MAX]], offset=pi * _P_MAX)
        )
    # k_pe sits after the scale and is genuine bf16.
    nisa.dma_transpose(dst=k_pe_sb, src=c_kv_hbm.ap(pattern=[[kv_row, S_kv], [1, R]], offset=kv_lat_bf16 + 1))
    # One scale per key, keys on the partition axis -- the orientation MM2's stationary wants.
    _scale_src = c_kv_hbm.ap(pattern=[[kv_row, _K_CHUNK], [_K_CHUNK * kv_row, num_k_chunks]], offset=kv_lat_bf16)
    nisa.dma_copy(dst=scale_bf16, src=_scale_src)
    nisa.tensor_copy(dst=scale_col, src=scale_bf16, engine=nisa.engine.vector)
    # MM2 stationary straight off the row: the DMA converts fp8 -> bf16, then one [P, 1] multiply
    # per chunk dequantizes it.
    nisa.dma_copy(
        dst=c_g_t_fp8,
        src=c_kv_hbm.ap(
            pattern=[[_row_b, _K_CHUNK], [_K_CHUNK * _row_b, num_k_chunks], [1, L]],
            offset=0,
            dtype=nl.float8_e4m3,
        ),
    )
    for m in range(num_k_chunks):
        nisa.tensor_scalar(
            c_g_t_fp8[:, m, :],
            c_g_t_fp8[:, m, :],
            op0=nl.multiply,
            operand0=scale_col[:, m : m + 1],
            engine=nisa.engine.vector if m % 2 == 0 else nisa.engine.scalar,
        )
    # s_bcast[j, k] = scale[k]: a free-axis broadcast is a stride-0 view, and the transpose lands
    # it on every partition at once.
    for m in range(num_k_chunks):
        sc_psum = nl.ndarray((_P_MAX, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum, address=(0, (m % 2) * PSUM_BANK_SIZE))
        nisa.nc_transpose(sc_psum, scale_bf16[:, m : m + 1].broadcast(dim=1, size=_P_MAX))
        nisa.tensor_copy(
            dst=s_bcast[:, m * _P_MAX : (m + 1) * _P_MAX],
            src=sc_psum,
            engine=nisa.engine.vector if m % 2 == 0 else nisa.engine.scalar,
        )
    # Dequantize into MM1's moving operand: a stride-2 fp8 view picks group li's channel out of
    # the pair tile and the multiply upcasts it to bf16.
    for li in range(n_l):
        nisa.tensor_tensor(
            c_sb_tiles[li],
            pair_tiles[li % n_pair].ap(
                pattern=[[_FP8_TP_OUT_STEP * S_kv, _P_MAX], [_FP8_TP_OUT_STEP, S_kv]],
                offset=li // n_pair,
                dtype=nl.float8_e4m3,
            ),
            s_bcast,
            op=nl.multiply,
            engine=nisa.engine.vector,
        )
    return c_sb_tiles, k_pe_sb, S_kv, scale_col, c_g_t_fp8


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


def _load_cp_scalar(hbm, sbm, dtype, name, default=0):
    """Load a [1, 1] runtime scalar tensor and broadcast it to a [P_MAX, 1] SBUF column.

    The scalar (prior_tokens or cp_rank) arrives as a TENSOR, not a kernel constant, so the
    framework traces one graph for every CP rank/step instead of specializing per rank. It is
    consumed as a per-partition operand, so the value is cast on a compute engine (DMA cannot
    cast) and replicated across all partitions once, outside the query loop.

    ``hbm is None`` is the non-CP / first-chunk case: the column is memset to ``default``.
    """
    col = sbm.alloc_stack((_P_MAX, 1), dtype=dtype, buffer=nl.sbuf, name=name)
    if hbm is None:
        nisa.memset(col, value=default)
    else:
        scalar_in = sbm.alloc_stack((1, 1), dtype=hbm.dtype, buffer=nl.sbuf, name=f"{name}_in")
        nisa.dma_copy(dst=scalar_in, src=hbm)
        nisa.tensor_copy(dst=col[0:1, :], src=scalar_in, engine=nisa.engine.vector)
        stream_shuffle_broadcast(src=col[0:1, :], dst=col)
    return col


def _owned_positions(dst, rank_tile, l_tile, hi_tile, lo_tile, interleave_size, cp_world_size):
    """dst = owned(rank, l) elementwise (int32), the CP interleave slot->global position map.

    owned(rank, l) = (l // I) * (I*W) + rank * I + (l % I), with I = interleave_size a power of
    two and W = cp_world_size, so // and % are shifts/masks. ``rank_tile`` / ``l_tile`` and the
    two scratch tiles ``hi_tile`` / ``lo_tile`` are all int32 and shaped like ``dst`` (arithmetic
    and bitvector ops cannot mix in one tensor_scalar, so each step is a separate instruction).
    """
    lg_i = int(math.log2(interleave_size))
    lg_p = int(math.log2(interleave_size * cp_world_size))
    # hi = (l >> lgI) << lgP  == (l // I) * (I*W)
    nisa.tensor_scalar(hi_tile, l_tile, op0=nl.right_shift, operand0=lg_i, engine=nisa.engine.vector)
    nisa.tensor_scalar(hi_tile, hi_tile, op0=nl.left_shift, operand0=lg_p, engine=nisa.engine.vector)
    # lo = l & (I-1)  == l % I
    nisa.tensor_scalar(lo_tile, l_tile, op0=nl.bitwise_and, operand0=interleave_size - 1, engine=nisa.engine.vector)
    # dst = rank << lgI  == rank * I
    nisa.tensor_scalar(dst, rank_tile, op0=nl.left_shift, operand0=lg_i, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst, dst, hi_tile, op=nl.add, engine=nisa.engine.vector)
    nisa.tensor_tensor(dst, dst, lo_tile, op=nl.add, engine=nisa.engine.vector)


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
    prior_tokens_hbm=None,
    cp_rank_hbm=None,
    cp_world_size=1,
    cp_kv_cache_interleave_size=1,
    precision=MlaPrecision.MX,
    use_sbuf_indirect=False,
    kv_cache_packed=False,
    kv_cache_fp8_dtype=nl.float8_e4m3,
):
    """Sparse latent + RoPE attention, S-sharded.

    Computes attn_latent[H, L] per query for queries [s_start, s_start+s_per_core) and
    writes them to out_attn_hbm[B, S, H*L] (row-major h*L + l).

    Causal re-mask of the gathered topk keys: a query with fewer than K valid causal keys gets its
    topk padded by the indexer with future positions, which are set to -inf here.

    CP interleave (cp_world_size > 1): the gathered cache is rank-major interleaved and topk
    indices are PHYSICAL slots, so the mask converts each key slot and query row to its GLOBAL
    position. The SBUF-resident path does not support CP.

    kv_sbuf: optional PRE-GATHERED SBUF KV, see _load_kv_cache.
    """
    if use_sbuf_indirect:
        _attention_stage_sparse_resident(
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
            kv_sbuf=kv_sbuf,
            topk_tiled=topk_tiled,
            prior_tokens_hbm=prior_tokens_hbm,
            cp_world_size=cp_world_size,
            precision=precision,
            kv_cache_packed=kv_cache_packed,
        )
        return
    _attention_stage_sparse_gather(
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
        prior_tokens_hbm=prior_tokens_hbm,
        cp_rank_hbm=cp_rank_hbm,
        cp_world_size=cp_world_size,
        cp_kv_cache_interleave_size=cp_kv_cache_interleave_size,
        precision=precision,
        kv_cache_packed=kv_cache_packed,
        kv_cache_fp8_dtype=kv_cache_fp8_dtype,
    )


def _attention_stage_sparse_gather(
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
    topk_tiled=False,
    prior_tokens_hbm=None,
    cp_rank_hbm=None,
    cp_world_size=1,
    cp_kv_cache_interleave_size=1,
    precision=MlaPrecision.MX,
    kv_cache_packed=False,
    kv_cache_fp8_dtype=nl.float8_e4m3,
):
    """DMA-indirect sparse attention: each query's K rows are gathered straight from HBM.

    Scores are computed KEYS-ON-PARTITION: MM1 takes the gathered (transposed) KV as its
    stationary and q as moving, so scores_t is [key, head]. The per-key dequant scale and the
    causal mask are then [P, 1] operands, and the exp output is directly MM2's moving operand.
    Softmax reduces over keys, i.e. across partitions: a log-tree fold down the chunk axis, then
    one transpose + free-axis reduce.

    Queries are software-pipelined in three stages, emitted A(s), C(s - 1), T(s):
      A: topk / q loads, gather, cache transposes, mask, MM1, max fold
      T: the two softmax transposes (row max, replicated shift)
      C: shift, exp, p_t, MM2, out transposes, row sum, output store
    so the in-order Tensor Engine always has the other query's matmuls queued. Per-query buffers
    of A -> C are double-buffered by query parity, and the gather inputs are _GATHER_BUFS deep.

    BF16 cache: c_kv and k_pe are first interleaved into one [S_kv, L + R] HBM copy so each query
    is a single gather (under LNC2 the two cores each copy half and barrier). Packed fp8 cache:
    the row already carries [fp8 latent pairs | scale | k_pe].
    """
    is_cp = cp_world_size > 1
    # A packed cache is [S_kv, row], so tokens are the MAJOR dim there; unpacked is [B, S_kv, L].
    S_kv_local = (c_kv_hbm.shape[0] if kv_cache_packed else c_kv_hbm.shape[1]) // cp_world_size
    B, S, H, L = q_lift_hbm.shape
    R = q_pe_hbm.shape[3]
    K = topk_indices_hbm.shape[2]
    HL = H * L
    n_l = L // _P_MAX
    num_k_chunks = K // _K_CHUNK
    n_idx = K // _P_MAX  # gathered-row batches; == num_k_chunks
    q_s_stride = H * L
    q_pe_s_stride = H * R
    topk_s_stride = K
    PSUM_BANK_SIZE = _get_psum_bank_size()
    # MM1 chunks per PSUM bank: scores_t is [_K_CHUNK, H] fp32, a quarter of a bank's worth of
    # chunks per bank keeps two banks rotating with few evictions.
    mm1_cpb = max(1, _chunks_per_psum_bank(num_k_chunks, H * 4, PSUM_BANK_SIZE) // 4)

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

    sbm.open_scope()

    # Packed: c_kv_hbm is [S_kv, kv_row] bf16, each row [L//2 fp8 latent pairs | scale | R k_pe]; k_pe_hbm unused.
    kv_lat_bf16 = L // 2
    # Packed latent counted in bf16 pair slots: 2 fp8 channels per slot.
    n_pair = max(1, kv_lat_bf16 // _P_MAX)
    kv_row = c_kv_hbm.shape[1] if kv_cache_packed else L
    S_kv = c_kv_hbm.shape[0] if kv_cache_packed else c_kv_hbm.shape[1]

    if not kv_cache_packed:
        # One [S_kv, L + R] copy so the per-query gather fetches latent and k_pe in one descriptor
        # per key. Under LNC2 each core interleaves half the rows into shared HBM, then barriers.
        _, n_prg, prg = get_verified_program_sharding_info("mla_sparse_attention_cte_kernel", (0, 1))
        split = n_prg == 2 and S_kv % 2 == 0
        kv_fused_hbm = nl.ndarray(
            (S_kv, L + R),
            dtype=nl.bfloat16,
            buffer=nl.shared_hbm if split else nl.private_hbm,
            name="kv_fused_hbm",
        )
        r_lo, r_hi = (prg * (S_kv // 2), (prg + 1) * (S_kv // 2)) if split else (0, S_kv)
        rows = min(r_hi - r_lo, 2048)
        for r0 in range(r_lo, r_hi, rows):
            nisa.dma_copy(
                dst=kv_fused_hbm.ap(pattern=[[L + R, rows], [1, L]], offset=r0 * (L + R)),
                src=c_kv_hbm.reshape((S_kv, L)).ap(pattern=[[L, rows], [1, L]], offset=r0 * L),
            )
            nisa.dma_copy(
                dst=kv_fused_hbm.ap(pattern=[[L + R, rows], [1, R]], offset=r0 * (L + R) + L),
                src=k_pe_hbm.reshape((S_kv, R)).ap(pattern=[[R, rows], [1, R]], offset=r0 * R),
            )
        if split:
            nisa.core_barrier(data=kv_fused_hbm, cores=(0, 1))

    # Causal-mask base column. Non-CP: float32 prior, the (gathered_pos - base) > q_loc operand.
    # CP: int32 prior and cp_rank feed the on-device owned() conversion (bitvector ops need int).
    base_dtype = nl.int32 if is_cp else nl.float32
    prior_col = _load_cp_scalar(prior_tokens_hbm, sbm, base_dtype, "prior_col", default=0)
    cp_rank_col = _load_cp_scalar(cp_rank_hbm, sbm, nl.int32, "cp_rank_col", default=0) if is_cp else None

    # Query-chunk size for the device loop; must divide s_per_core. _q_chunk == s_per_core selects the unrolled driver.
    _q_chunk = s_per_core if s_per_core <= _UNROLL_MAX_S_LOCAL else min(_QUERY_CHUNK, s_per_core)
    while s_per_core % _q_chunk != 0:
        _q_chunk -= 1
    if _q_chunk < s_per_core:
        qbase_i32 = sbm.alloc_stack((1, 1), dtype=nl.int32, buffer=nl.sbuf, name="qbase_i32")
        qbase_bcast = sbm.alloc_stack((_P_MAX, 1), dtype=base_dtype, buffer=nl.sbuf, name="qbase_bcast")
        # chunk base column = prior + qbase (folds the runtime chunk offset into the mask base).
        base_chunk_col = sbm.alloc_stack((_P_MAX, 1), dtype=base_dtype, buffer=nl.sbuf, name="base_chunk_col")
    else:
        qbase_i32, qbase_bcast, base_chunk_col = None, None, None
    # q needs a natural load + PE transpose wherever dma_transpose off HBM is unavailable: inside the
    # device loop (dynamic address) and for packed (its fp8 grouping is an element-stride-2 read).
    q_natural = _q_chunk < s_per_core or kv_cache_packed

    # Gather-side buffers, _GATHER_BUFS deep: in the A(s), C(s - 1) order query s + 1's gather
    # reuses the buffer of a query whose MM2 is still in flight at depth 2.
    q_lift_t_bufs, q_pe_t_bufs, q_lift_nat_bufs, q_pe_nat_bufs = [], [], [], []
    c_g_bufs, k_pe_g_bufs, c_g_nat_bufs, idx_dma_bufs = [], [], [], []
    for buf_idx in range(_GATHER_BUFS):
        q_lift_t_bufs.append(
            sbm.alloc_stack((_P_MAX, n_l, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qlt_{buf_idx}")
        )
        q_pe_t_bufs.append(sbm.alloc_stack((R, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qpt_{buf_idx}"))
        if q_natural:
            q_lift_nat_bufs.append(sbm.alloc_stack((H, L), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"qln_{buf_idx}"))
            q_pe_nat_bufs.append(sbm.alloc_stack((H, R), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"qpn_{buf_idx}"))
        # MM1's stationary, latent on partitions. Packed holds bf16 PAIRS (fp8 channels 2j, 2j+1 on
        # partition j); MM1 re-views it as fp8 with an element stride of 2.
        c_g_bufs.append(
            sbm.alloc_stack(
                (_P_MAX, n_pair if kv_cache_packed else n_l, K),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=f"c_g_{buf_idx}",
            )
        )
        k_pe_g_bufs.append(sbm.alloc_stack((R, K), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_pe_g_{buf_idx}"))
        # The gathered rows in natural [slot, feature] order; also MM2's moving operand. Packed is
        # 2-D so the fp8 view's partition stride is a plain 2 * n_idx * kv_row; BF16 rows are
        # [L latent | R k_pe].
        c_g_nat_bufs.append(
            sbm.alloc_stack(
                (_P_MAX, n_idx * kv_row) if kv_cache_packed else (_P_MAX, n_idx, L + R),
                dtype=nl.bfloat16,
                buffer=nl.sbuf,
                name=f"kv_nat_{buf_idx}",
            )
        )
        # The gather's 2D vector_offset: idx_dma[p, m] = topk[m*128 + p] (it is read column-major).
        idx_dma_bufs.append(
            sbm.alloc_stack((_P_MAX, n_idx), dtype=nl.uint32, buffer=nl.sbuf, name=f"idx_dma_{buf_idx}")
        )

    def _qalloc(*args, name, **kwargs):
        """One buffer per query parity, for values written in stage A and read in stage C."""
        return [sbm.alloc_stack(*args, name=f"{name}_{i}", **kwargs) for i in range(2)]

    # Per-key dequant scale (packed) and its reciprocal, for pre-dividing k_pe so RoPE can share
    # MM1's accumulator.
    scale_col = (
        _qalloc((_P_MAX, n_idx), dtype=nl.float32, buffer=nl.sbuf, name="kv_scale_col") if kv_cache_packed else None
    )
    scale_recip = (
        _qalloc((_P_MAX, n_idx), dtype=nl.float32, buffer=nl.sbuf, name="kv_scale_recip") if kv_cache_packed else None
    )
    # Scores with keys on the partition axis and the CHUNK AXIS IN THE MIDDLE, so any run of chunks
    # is one contiguous [_K_CHUNK, n, H] tile that a single instruction can cover.
    scores_t_all = _qalloc((_K_CHUNK, num_k_chunks, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="scores_t_all")
    # UNSCALED exp: the softmax denominator must not carry the per-key dequant scale.
    exp_all = _qalloc((_K_CHUNK, num_k_chunks, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="exp_all")
    # MM2's moving operand (packed: exp times the per-key scale).
    p_t_all = _qalloc((_K_CHUNK, num_k_chunks, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="p_t_all")
    # Log-tree fold scratch, reused by both the max and the sum.
    fold_half = _qalloc((_K_CHUNK, max(1, num_k_chunks // 2), H), dtype=nl.bfloat16, buffer=nl.sbuf, name="fold_half")
    fold_out = _qalloc((_K_CHUNK, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="fold_out")
    mask_col = _qalloc((_P_MAX, n_idx), dtype=nl.float32, buffer=nl.sbuf, name="mask_col")
    neg_row_max = _qalloc((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="neg_row_max")
    row_sum = _qalloc((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="row_sum")
    recip = _qalloc((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="recip")
    # MM2 emits latent-on-partition out_T; staged here because nc_transpose's stationary must be SBUF.
    out_t_sb = _qalloc((_P_MAX, n_l, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name="out_t_sb")
    out_bf16 = _qalloc((H, L), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_bf16")
    if is_cp:
        # Query owned(): l = base + q_loc, then q_global = owned(cp_rank, l).
        q_global_sb = _qalloc((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name="q_global_sb")
        # tensor_scalar's operand0 must be float32, so the int32 owned() result is cast here.
        q_global_f32 = _qalloc((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="q_global_f32")
        l_sb = _qalloc((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name="l_sb")
        hi_sb = _qalloc((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name="hi_sb")
        lo_sb = _qalloc((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name="lo_sb")

    # Key positions for the causal mask, loaded _TOPK_GROUP queries at a time (one DMA instead of
    # one per query), double-buffered so group g + 1 loads while group g runs. gpos_grp[p, q, m] =
    # topk[q, p*n_idx + m], the key behind score partition p of chunk m.
    tk_g = min(_TOPK_GROUP, _q_chunk)
    while _q_chunk % tk_g:
        tk_g -= 1
    gpos_grp = [
        sbm.alloc_stack((_P_MAX, tk_g * n_idx), dtype=nl.int32, buffer=nl.sbuf, name=f"gpos_grp{i}") for i in range(2)
    ]
    gpos_cvt = (
        [sbm.alloc_stack((_P_MAX, tk_g * n_idx), dtype=nl.int32, buffer=nl.sbuf, name=f"gpos_cvt{i}") for i in range(2)]
        if is_cp
        else None
    )

    def _tree_fold(src, op, fold_out, fold_half):
        """Reduce ``src`` [_K_CHUNK, num_k_chunks, H] down the CHUNK axis into ``fold_out``.

        The chunk axis is the middle axis, so any run of chunks is one contiguous tile and a halving
        tree costs log2(num_k_chunks) instructions. Falls back to serial when num_k_chunks is not a
        power of two.
        """
        if num_k_chunks == 1:
            nisa.tensor_copy(dst=fold_out, src=src[:, 0, :], engine=nisa.engine.vector)
            return
        if num_k_chunks & (num_k_chunks - 1):
            nisa.tensor_tensor(fold_out, src[:, 0, :], src[:, 1, :], op=op, engine=nisa.engine.vector)
            for m in range(2, num_k_chunks):
                nisa.tensor_tensor(fold_out, fold_out, src[:, m, :], op=op, engine=nisa.engine.vector)
            return
        n = num_k_chunks // 2
        nisa.tensor_tensor(
            fold_out if n == 1 else fold_half[:, 0:n, :],
            src[:, 0:n, :],
            src[:, n : 2 * n, :],
            op=op,
            engine=nisa.engine.vector,
        )
        while n > 1:
            h = n // 2
            nisa.tensor_tensor(
                fold_out if h == 1 else fold_half[:, 0:h, :],
                fold_half[:, 0:h, :],
                fold_half[:, h : 2 * h, :],
                op=op,
                engine=nisa.engine.vector,
            )
            n = h

    def _owned_key_positions(gp, gs1, gs2):
        """Physical cache slot -> global position, in place: owned(idx // S_kv_local, idx % S_kv_local).

        owned()'s three terms occupy disjoint bit-fields, so they combine with bitwise-OR:
        l%I -> bits [0, lgI); rank*I -> bits [lgI, lgP) (rank in [0, W)); (l//I)*P -> bits [lgP, ...).
        """
        lg_sl = int(math.log2(S_kv_local))
        lg_i = int(math.log2(cp_kv_cache_interleave_size))
        lg_p = int(math.log2(cp_kv_cache_interleave_size * cp_world_size))
        # C = ((idx & maskSL) >> lgI) << lgP  == (l // I) * (I*W), l = idx % S_kv_local
        nisa.tensor_scalar(
            gs1,
            gp,
            op0=nl.bitwise_and,
            operand0=S_kv_local - 1,
            op1=nl.right_shift,
            operand1=lg_i,
            engine=nisa.engine.vector,
        )
        nisa.tensor_scalar(gs1, gs1, op0=nl.left_shift, operand0=lg_p, engine=nisa.engine.vector)
        # B = (idx >> lgSL) << lgI  == rank * I,  rank = idx // S_kv_local
        nisa.tensor_scalar(
            gs2, gp, op0=nl.right_shift, operand0=lg_sl, op1=nl.left_shift, operand1=lg_i, engine=nisa.engine.vector
        )
        # A = idx & maskI  == l % I, then OR the disjoint fields: owned = A | B | C
        nisa.tensor_scalar(
            gp, gp, op0=nl.bitwise_and, operand0=cp_kv_cache_interleave_size - 1, engine=nisa.engine.vector
        )
        nisa.tensor_tensor(gp, gp, gs1, op=nl.bitwise_or, engine=nisa.engine.vector)
        nisa.tensor_tensor(gp, gp, gs2, op=nl.bitwise_or, engine=nisa.engine.vector)

    def _load_topk_group(qbase_sb, grp):
        """Key positions for queries [grp * tk_g, (grp + 1) * tk_g) of the current chunk."""
        topk_2d = topk_indices_hbm.reshape((B * S, K))
        nisa.dma_copy(
            dst=gpos_grp[grp % 2],
            src=_qap(
                topk_2d, [[n_idx, _P_MAX], [K, tk_g], [1, n_idx]], (s_start + grp * tk_g) * topk_s_stride, 0, qbase_sb
            ),
            dge_mode=dge_mode.hwdge,
            engine=nisa.engine.sync,
        )

    def _query_body(s_local, qbase_sb, base_col):
        """Emit stage A for one query and return its (stage T, stage C) closures.

        ``s_local`` is trace-time; ``qbase_sb`` is the runtime chunk base ([1, 1] int32 SBUF) or None
        when the whole query loop is unrolled. ``base_col`` is the per-partition mask base
        ``prior + qbase`` (non-CP: float32; CP: int32, the owned() query-position base).
        """
        q_loc = s_start + s_local
        buf_idx = s_local % _GATHER_BUFS
        qb = s_local % 2
        q_lift_t, q_pe_t = q_lift_t_bufs[buf_idx], q_pe_t_bufs[buf_idx]
        c_g, k_pe_g, c_g_nat, idx_dma = (
            c_g_bufs[buf_idx],
            k_pe_g_bufs[buf_idx],
            c_g_nat_bufs[buf_idx],
            idx_dma_bufs[buf_idx],
        )
        scores_t, exp_t, p_t = scores_t_all[qb], exp_all[qb], p_t_all[qb]
        f_out, f_half, msk = fold_out[qb], fold_half[qb], mask_col[qb]
        nrm, rsum, rcp, o_t, o_bf16 = neg_row_max[qb], row_sum[qb], recip[qb], out_t_sb[qb], out_bf16[qb]
        sc_col = scale_col[qb] if kv_cache_packed else None
        sc_rcp = scale_recip[qb] if kv_cache_packed else None
        gp0 = (s_local % tk_g) * n_idx
        gpos_col = gpos_grp[(s_local // tk_g) % 2][:, gp0 : gp0 + n_idx]

        # ---- Stage A: loads and gather.
        nisa.dma_copy(
            dst=idx_dma,
            src=_qap(
                topk_indices_hbm.reshape((B * S, K)), [[1, _P_MAX], [_P_MAX, n_idx]], q_loc * topk_s_stride, 0, qbase_sb
            ),
        )
        if not q_natural:
            # One dma_transpose: 3D source (H, n_l, 128_latent) permuted [2,1,0] -> [128_latent, n_l, H].
            nisa.dma_transpose(
                dst=q_lift_t,
                src=q_lift_hbm.ap(pattern=[[L, H], [_P_MAX, n_l], [1, _P_MAX]], offset=q_loc * q_s_stride),
            )
            nisa.dma_transpose(dst=q_pe_t, src=q_pe_hbm.ap(pattern=[[R, H], [1, R]], offset=q_loc * q_pe_s_stride))
        else:
            # Natural load + PE transpose: in a device loop a dma_transpose off a kernel input is
            # rejected (NCC_IBIR251), and packed's q grouping is an element-stride-2 read.
            q_lift_nat, q_pe_nat = q_lift_nat_bufs[buf_idx], q_pe_nat_bufs[buf_idx]
            nisa.dma_copy(
                dst=q_lift_nat,
                src=_qap(q_lift_hbm, [[L, H], [1, L]], q_loc * q_s_stride, 1, qbase_sb),
                dge_mode=dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
            nisa.dma_copy(
                dst=q_pe_nat,
                src=_qap(q_pe_hbm, [[R, H], [1, R]], q_loc * q_pe_s_stride, 1, qbase_sb),
                dge_mode=dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
            qlt_psum = nl.ndarray(
                (_P_MAX, n_l * H), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 6 * PSUM_BANK_SIZE + 1024)
            )
            for li in range(n_l):
                # Packed: group par * n_pair + pi holds channel 2 * (pi * _P_MAX + j) + par on partition j.
                qsrc = (
                    q_lift_nat.ap(
                        pattern=[[L, H], [_FP8_TP_OUT_STEP, _P_MAX]],
                        offset=(li % n_pair) * _FP8_TP_OUT_STEP * _P_MAX + li // n_pair,
                    )
                    if kv_cache_packed
                    else q_lift_nat[:, li * _P_MAX : (li + 1) * _P_MAX]
                )
                nisa.nc_transpose(qlt_psum[:, li * H : (li + 1) * H], qsrc)
            nisa.tensor_copy(dst=q_lift_t, src=qlt_psum, engine=nisa.engine.scalar)
            qpt_psum = nl.ndarray((R, H), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 7 * PSUM_BANK_SIZE + 1024))
            nisa.nc_transpose(qpt_psum, q_pe_nat)
            nisa.tensor_copy(dst=q_pe_t, src=qpt_psum, engine=nisa.engine.scalar)

        # One indirect DMA gathers all K rows (one descriptor per key). Indirect DMA is SWDGE-only.
        gather_src = (
            c_kv_hbm.reshape((S_kv, kv_row)).ap(
                pattern=[[kv_row, _P_MAX * n_idx], [1, kv_row]], vector_offset=idx_dma, indirect_dim=0
            )
            if kv_cache_packed
            else kv_fused_hbm.ap(pattern=[[L + R, _P_MAX * n_idx], [1, L + R]], vector_offset=idx_dma, indirect_dim=0)
        )
        nisa.dma_copy(dst=c_g_nat, src=gather_src, dge_mode=dge_mode.swdge)

        # ---- Stage A: cache transposes. MM1 contracts over L, so it needs latent-on-partition [L, K].
        if kv_cache_packed:
            # One scale per key, keys on the partition axis; one strided read covers every chunk.
            nisa.tensor_copy(
                dst=sc_col,
                src=c_g_nat.ap(pattern=[[n_idx * kv_row, _P_MAX], [kv_row, n_idx]], offset=kv_lat_bf16),
                engine=nisa.engine.vector,
            )
            nisa.reciprocal(sc_rcp, sc_col)
            # Transpose the latent as bf16 PAIRS: an fp8 transpose must write PSUM with an output
            # element step of 2. MM1 separates the two channels with a stride-2 fp8 view. Batches
            # are capped at half a bank so MM2's accumulators at +1.5 KB stay clear.
            cg_cpb = _chunks_per_psum_bank(n_idx, n_pair * _P_MAX * 2, PSUM_BANK_SIZE // 2)
            for b in range(n_idx // cg_cpb):
                cgn_psum = nl.ndarray(
                    (_P_MAX, cg_cpb * n_pair * _P_MAX),
                    dtype=nl.bfloat16,
                    buffer=nl.psum,
                    address=(0, (4 + b % 2) * PSUM_BANK_SIZE),
                )
                for j in range(cg_cpb):
                    m = b * cg_cpb + j
                    for li in range(n_pair):
                        c0 = (j * n_pair + li) * _P_MAX
                        nisa.nc_transpose(
                            cgn_psum[:, c0 : c0 + _P_MAX],
                            c_g_nat.ap(
                                pattern=[[n_idx * kv_row, _P_MAX], [1, _P_MAX]], offset=m * kv_row + li * _P_MAX
                            ),
                        )
                k0 = b * cg_cpb * _P_MAX
                for li in range(n_pair):
                    # The bank holds (chunk, pair) interleaved; one strided read per pair tile.
                    nisa.tensor_copy(
                        dst=c_g[:, li, k0 : k0 + cg_cpb * _P_MAX],
                        src=cgn_psum.ap(
                            pattern=[[cg_cpb * n_pair * _P_MAX, _P_MAX], [n_pair * _P_MAX, cg_cpb], [1, _P_MAX]],
                            offset=li * _P_MAX,
                        ),
                        engine=nisa.engine.vector if b % 2 == 0 else nisa.engine.scalar,
                    )
            kpe_cpb = _chunks_per_psum_bank(n_idx, _P_MAX * 2, PSUM_BANK_SIZE // 2)
            for b in range(n_idx // kpe_cpb):
                kpe_psum = nl.ndarray(
                    (R, kpe_cpb * _P_MAX), dtype=nl.bfloat16, buffer=nl.psum, address=(0, (6 + b % 2) * PSUM_BANK_SIZE)
                )
                for j in range(kpe_cpb):
                    m = b * kpe_cpb + j
                    kpe_ap = c_g_nat.ap(pattern=[[n_idx * kv_row, _P_MAX], [1, R]], offset=m * kv_row + kv_lat_bf16 + 1)
                    # Pre-divide k_pe by the per-key scale while keys are on the partition axis, so
                    # RoPE accumulates into MM1's PSUM and the dequant multiply on eviction restores it.
                    nisa.tensor_scalar(
                        kpe_ap,
                        kpe_ap,
                        op0=nl.multiply,
                        operand0=sc_rcp[:, m : m + 1],
                        engine=nisa.engine.vector if b % 2 == 0 else nisa.engine.scalar,
                    )
                    nisa.nc_transpose(kpe_psum[:, j * _P_MAX : (j + 1) * _P_MAX], kpe_ap)
                k0 = b * kpe_cpb * _P_MAX
                nisa.tensor_copy(
                    dst=k_pe_g[:, k0 : k0 + kpe_cpb * _P_MAX],
                    src=kpe_psum,
                    engine=nisa.engine.scalar if b % 2 == 0 else nisa.engine.vector,
                )
        else:
            for m in range(n_idx):
                # The first latent tile of each chunk on the DMA engine, the rest on PE: balances the two.
                nisa.dma_transpose(dst=c_g[:, 0, m * _P_MAX : (m + 1) * _P_MAX], src=c_g_nat[:, m, 0:_P_MAX])
                cgn_psum = nl.ndarray(
                    (_P_MAX, n_l - 1, _P_MAX),
                    dtype=nl.bfloat16,
                    buffer=nl.psum,
                    address=(0, (4 + m % 2) * PSUM_BANK_SIZE),
                )
                for li in range(1, n_l):
                    nisa.nc_transpose(cgn_psum[:, li - 1, :], c_g_nat[:, m, li * _P_MAX : (li + 1) * _P_MAX])
                # One eviction per chunk for all its PE tiles.
                nisa.tensor_copy(
                    dst=c_g[:, 1:, m * _P_MAX : (m + 1) * _P_MAX],
                    src=cgn_psum,
                    engine=nisa.engine.vector if m % 2 == 0 else nisa.engine.scalar,
                )
                # k_pe: [slot, R] -> [R, slot]. Evicted on Vector: Scalar's in-order queue delays it.
                kpe_psum = nl.ndarray(
                    (R, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum, address=(0, (6 + m % 2) * PSUM_BANK_SIZE)
                )
                nisa.nc_transpose(kpe_psum, c_g_nat[:, m, L : L + R])
                nisa.tensor_copy(dst=k_pe_g[:, m * _P_MAX : (m + 1) * _P_MAX], src=kpe_psum, engine=nisa.engine.vector)

        # ---- Stage A: causal mask column, one fp32 per key per chunk.
        if is_cp:
            # gpos_col already holds global key positions; q_global = owned(cp_rank, base + q_loc).
            nisa.tensor_scalar(l_sb[qb], base_col, op0=nl.add, operand0=q_loc, engine=nisa.engine.vector)
            _owned_positions(
                q_global_sb[qb], cp_rank_col, l_sb[qb], hi_sb[qb], lo_sb[qb], cp_kv_cache_interleave_size, cp_world_size
            )
            nisa.tensor_copy(dst=q_global_f32[qb], src=q_global_sb[qb], engine=nisa.engine.vector)
            nisa.tensor_scalar(
                msk,
                gpos_col,
                op0=nl.greater,
                operand0=q_global_f32[qb],
                op1=nl.multiply,
                operand1=_BF16_MASK_FILL,
                engine=nisa.engine.vector,
            )
        else:
            nisa.tensor_scalar(
                msk,
                gpos_col,
                op0=nl.subtract,
                operand0=base_col[0:_P_MAX, :],
                op1=nl.greater,
                operand1=q_loc,
                engine=nisa.engine.vector,
            )
            nisa.tensor_scalar(msk, msk, op0=nl.multiply, operand0=_BF16_MASK_FILL, engine=nisa.engine.scalar)

        # ---- Stage A: MM1, mm1_cpb chunks per PSUM bank, each bank drained by one eviction.
        for b in range(num_k_chunks // mm1_cpb):
            scores_t_psum = nl.ndarray(
                (_K_CHUNK, mm1_cpb * H), dtype=nl.float32, buffer=nl.psum, address=(0, (b % 2) * PSUM_BANK_SIZE)
            )
            for j in range(mm1_cpb):
                ks = (b * mm1_cpb + j) * _K_CHUNK
                dst = scores_t_psum[:, j * H : (j + 1) * H]
                for li in range(n_l):
                    # Packed: group li = par * n_pair + pi selects pair tile pi and parity par, matching q_lift_t.
                    mm1_kv = (
                        c_g.ap(
                            pattern=[[2 * n_pair * K, _P_MAX], [_FP8_TP_OUT_STEP, _K_CHUNK]],
                            offset=2 * ((li % n_pair) * K + ks) + li // n_pair,
                            dtype=kv_cache_fp8_dtype,
                        )
                        if kv_cache_packed
                        else c_g[:, li, ks : ks + _K_CHUNK]
                    )
                    nisa.nc_matmul(dst, mm1_kv, q_lift_t[:, li, :], accumulate=(li > 0))
                nisa.nc_matmul(dst, k_pe_g[:, ks : ks + _K_CHUNK], q_pe_t, accumulate=True)
            sl = scores_t[:, b * mm1_cpb : (b + 1) * mm1_cpb, :]
            if kv_cache_packed:
                # Dequant rides the eviction: one scale per key, stride-0 across the H heads.
                nisa.tensor_tensor(
                    sl,
                    scores_t_psum,
                    sc_col[:, b * mm1_cpb : (b + 1) * mm1_cpb].expand_dim(dim=2).broadcast(dim=2, size=H),
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )
            else:
                nisa.tensor_copy(dst=sl, src=scores_t_psum, engine=nisa.engine.scalar if b % 2 else nisa.engine.vector)

        # Causal fill for every chunk in one pass, then the per-head max over keys (log-tree fold).
        nisa.tensor_tensor(
            scores_t,
            scores_t,
            msk[:, 0:num_k_chunks].expand_dim(dim=2).broadcast(dim=2, size=H),
            op=nl.add,
            engine=nisa.engine.vector,
        )
        _tree_fold(scores_t, nl.maximum, f_out, f_half)

        # Softmax-transpose PSUM, offset by query parity so T(s) never waits on C(s - 1)'s reads.
        mx_psum = nl.ndarray((H, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 2 * PSUM_BANK_SIZE + 256 * qb))
        sh_psum = nl.ndarray(
            (_K_CHUNK, H), dtype=nl.float32, buffer=nl.psum, address=(0, 2 * PSUM_BANK_SIZE + 1024 + 256 * qb)
        )

        def _stage_t():
            # Keys are the partition axis, so the per-head max is cross-partition: transpose the
            # [P, H] partial and reduce its free axis.
            nisa.nc_transpose(mx_psum, f_out)
            nisa.tensor_reduce(nrm, op=nl.maximum, data=mx_psum, axis=1, negate=True)
            # Heads are on the FREE axis, so activation's per-partition bias cannot carry the shift.
            # Transposing the stride-0 [H, _K_CHUNK] broadcast lands it replicated across keys.
            nisa.nc_transpose(sh_psum, nrm.broadcast(dim=1, size=_K_CHUNK))

        def _stage_c():
            # exp(softmax_scale * (scores - per_head_max)), each one instruction over all chunks.
            nisa.tensor_tensor(
                scores_t,
                scores_t,
                sh_psum.expand_dim(dim=1).broadcast(dim=1, size=num_k_chunks),
                op=nl.add,
                engine=nisa.engine.vector,
            )
            nisa.activation(dst=exp_t, op=nl.exp, data=scores_t, scale=softmax_scale)
            if kv_cache_packed:
                # MM2's fp8 latent carries no scale and MM2 contracts over K, so the per-key factor rides p_t.
                nisa.tensor_tensor(
                    p_t,
                    exp_t,
                    sc_col[:, 0:num_k_chunks].expand_dim(dim=2).broadcast(dim=2, size=H),
                    op=nl.multiply,
                    engine=nisa.engine.vector,
                )
            else:
                nisa.tensor_copy(dst=p_t, src=exp_t, engine=nisa.engine.scalar)

            # MM2: out_T[d, h] = sum_k c_g_nat[k, d] * p_t[k, h]. One accumulation bank per latent tile,
            # at +1.5 KB, clear of the next query's stage-A transposes in the same banks.
            pvt_psum = [
                nl.ndarray((_P_MAX, H), dtype=nl.float32, buffer=nl.psum, address=(0, (4 + li) * PSUM_BANK_SIZE + 1536))
                for li in range(n_l)
            ]
            for chunk_idx in range(num_k_chunks):
                mm2_kv = (
                    # Mixed-dtype matmul: fp8 moving operand against the bf16 p_t stationary.
                    c_g_nat.ap(
                        pattern=[[2 * n_idx * kv_row, _K_CHUNK], [1, L]],
                        offset=2 * chunk_idx * kv_row,
                        dtype=kv_cache_fp8_dtype,
                    )
                    if kv_cache_packed
                    else c_g_nat[:, chunk_idx, 0:L]
                )
                for li in range(n_l):
                    nisa.nc_matmul(
                        pvt_psum[li],
                        mm2_kv[:, li * _P_MAX : (li + 1) * _P_MAX],
                        p_t[:, chunk_idx, :],
                        accumulate=(chunk_idx > 0),
                    )

            # out_T [L, H] -> [H, L]; evict first because nc_transpose's stationary must be SBUF.
            for li in range(n_l):
                nisa.tensor_copy(dst=o_t[:, li, :], src=pvt_psum[li], engine=nisa.engine.scalar)
            out_psum = nl.ndarray((H, L), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 3 * PSUM_BANK_SIZE + 1024))
            for li in range(n_l):
                nisa.nc_transpose(out_psum[:, li * _P_MAX : (li + 1) * _P_MAX], o_t[:, li, :])

            # Row sum over keys from the UNSCALED exp; only the final scale needs it, so it is emitted
            # after the out transposes rather than ahead of them in the in-order queues.
            _tree_fold(exp_t, nl.add, f_out, f_half)
            sm_psum = nl.ndarray((H, _P_MAX), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 3 * PSUM_BANK_SIZE))
            nisa.nc_transpose(sm_psum, f_out)
            nisa.tensor_reduce(rsum, op=nl.add, data=sm_psum, axis=1)
            nisa.reciprocal(rcp, rsum)

            # CROSS-KERNEL LAYOUT CONTRACT: 4-pack permuted (MX) or natural (BF16); see _mm2_out_aps.
            out_ap, psum_ap = _mm2_out_aps(o_bf16, out_psum, H, L, precision)
            nisa.tensor_scalar(out_ap, psum_ap, op0=nl.multiply, operand0=rcp, engine=nisa.engine.scalar)
            nisa.dma_copy(
                dst=_qap(out_attn_hbm, [[L, H], [1, L]], q_loc * HL, 1, qbase_sb),
                src=o_bf16,
                dge_mode=dge_mode.hwdge,
                engine=nisa.engine.sync,
            )

        return _stage_t, _stage_c

    def _run_queries(n, qbase_sb, base_col):
        pending_c = None
        _load_topk_group(qbase_sb, 0)
        for s_local in range(n):
            if s_local % tk_g == 0:
                # Engine queues are in order: issue group g + 1's load now, but convert group g only
                # here, right before its first use, so Vector never blocks on an in-flight load.
                if is_cp:
                    _owned_key_positions(gpos_grp[(s_local // tk_g) % 2], gpos_cvt[0], gpos_cvt[1])
                if s_local + tk_g < n:
                    _load_topk_group(qbase_sb, s_local // tk_g + 1)
            stage_t, stage_c = _query_body(s_local, qbase_sb, base_col)
            if pending_c is not None:
                pending_c()
            stage_t()
            pending_c = stage_c
        pending_c()

    if _q_chunk >= s_per_core:
        _run_queries(s_per_core, None, prior_col)
    else:
        # Device loop over query chunks; base_chunk_col = prior + qbase folds the runtime chunk
        # offset into the mask base.
        def _query_chunk(qbase_reg):
            nisa.register_store(dst=qbase_i32, src=qbase_reg)
            nisa.tensor_copy(dst=qbase_bcast[0:1, :], src=qbase_i32, engine=nisa.engine.vector)
            stream_shuffle_broadcast(src=qbase_bcast[0:1, :], dst=qbase_bcast)
            nisa.tensor_tensor(base_chunk_col, prior_col, qbase_bcast, op=nl.add, engine=nisa.engine.vector)
            _run_queries(_q_chunk, qbase_i32, base_chunk_col)

        nl.fori_loop(0, s_per_core, _query_chunk, step=_q_chunk)

    sbm.close_scope()


def _attention_stage_sparse_resident(
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
    prior_tokens_hbm=None,
    cp_world_size=1,
    precision=MlaPrecision.MX,
    kv_cache_packed=False,
):
    """SBUF-resident sparse attention (``use_sbuf_indirect=True``).

    The whole gathered cache is made SBUF-resident and each query's K columns are gathered with
    SBUF tensor indirection; scores are heads-on-partition [H, K]. Fully unrolled query loop,
    non-CP, unpacked cache only.
    """
    kernel_assert(
        cp_world_size == 1,
        "[MLA sparse attn] CP interleave (cp_world_size > 1) is not yet supported on the "
        "SBUF-resident path (use_sbuf_indirect=True); it caps S_kv at 10240/20480 and is not "
        "the long-context target. Pass use_sbuf_indirect=False (DMA-indirect gather).",
    )
    kernel_assert(
        not kv_cache_packed,
        "[MLA sparse attn] the packed fp8 cache is only supported with use_sbuf_indirect=False.",
    )
    B, S, H, L = q_lift_hbm.shape
    R = q_pe_hbm.shape[3]
    # topk_indices: FLAT [B, S, K] -> K = shape[2]; TILED [num_s_tiles, NUM_TOPK_BATCHES, P_MAX, K//16] -> K = shape[3] * 16.
    K = (topk_indices_hbm.shape[3] * 16) if topk_tiled else topk_indices_hbm.shape[2]
    HL = H * L
    n_l = L // _P_MAX
    num_k_chunks = K // _K_CHUNK
    mm1_tile = min(_MM1_TILE, K)
    num_mm1_tiles = K // mm1_tile
    sm_tile = min(_SM_TILE, K)
    num_sm_tiles = K // sm_tile
    ti_f = K // 16
    PSUM_BANK_SIZE = _get_psum_bank_size()

    sbm.open_scope()
    S_kv = c_kv_hbm.shape[1]
    limit, itemsize = _resident_s_kv_limit(c_kv_hbm)
    kernel_assert(
        S_kv <= limit,
        f"[MLA sparse attn] use_sbuf_indirect=True keeps the WHOLE gathered cache SBUF-resident "
        f"({(L + R) * itemsize} B/partition per position), which caps S_kv at {limit} for a "
        f"{itemsize}-byte cache; got S_kv={S_kv}. Pass use_sbuf_indirect=False to gather from "
        f"HBM per query instead (SBUF cost then independent of S_kv).",
    )
    c_sb_tiles, k_pe_sb, S_kv = _load_kv_cache(c_kv_hbm, k_pe_hbm, R, L, n_l, sbm, kv_sbuf)
    # Cache-position column [0, S_kv), gathered with idx_u16 so each score column carries its key position.
    pos_col = sbm.alloc_stack((_P_MAX, S_kv), dtype=nl.uint16, buffer=nl.sbuf, name="pos_col")
    nisa.iota(dst=pos_col, pattern=[[1, S_kv]], offset=0, channel_multiplier=0)
    prior_col = _load_cp_scalar(prior_tokens_hbm, sbm, nl.float32, "prior_col", default=0)

    idx_i32_bufs, idx_u16_bufs, q_lift_t_bufs, q_pe_t_bufs, c_g_bufs, k_pe_g_bufs, gathered_pos_bufs = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for buf_idx in range(2):
        idx_i32_bufs.append(sbm.alloc_stack((_P_MAX, ti_f), dtype=nl.int32, buffer=nl.sbuf, name=f"idx_i32_{buf_idx}"))
        idx_u16_bufs.append(sbm.alloc_stack((_P_MAX, ti_f), dtype=nl.uint16, buffer=nl.sbuf, name=f"idx_u16_{buf_idx}"))
        q_lift_t_bufs.append(
            sbm.alloc_stack((_P_MAX, n_l, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qlt_{buf_idx}")
        )
        q_pe_t_bufs.append(sbm.alloc_stack((R, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qpt_{buf_idx}"))
        c_g_bufs.append(sbm.alloc_stack((_P_MAX, n_l, K), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"c_g_{buf_idx}"))
        k_pe_g_bufs.append(sbm.alloc_stack((R, K), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_pe_g_{buf_idx}"))
        gathered_pos_bufs.append(
            sbm.alloc_stack((_P_MAX, K), dtype=nl.uint16, buffer=nl.sbuf, name=f"gathered_pos_{buf_idx}")
        )
    # bf16: the exp still subtracts an exact fp32 per-head max through activation's bias.
    scores_sb = sbm.alloc_stack((H, K), dtype=nl.bfloat16, buffer=nl.sbuf, name="scores_sb")
    # bf16: an unmasked key adds exactly 0.0 and the masked fill is representable in bf16.
    mask_add = sbm.alloc_stack((H, K), dtype=nl.bfloat16, buffer=nl.sbuf, name="mask_add")
    qpos_thr = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="qpos_thr")
    p = sbm.alloc_stack((H, K), dtype=nl.bfloat16, buffer=nl.sbuf, name="p")
    neg_row_max = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="neg_row_max")
    exp_bias = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="exp_bias")
    row_sum = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="row_sum")
    recip = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="recip")
    out_bf16 = sbm.alloc_stack((H, L), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_bf16")
    # MM2 emits latent-on-partition out_T; staged here because nc_transpose's stationary must be SBUF.
    out_t_sb = sbm.alloc_stack((_P_MAX, n_l, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name="out_t_sb")
    # MM2's keys-on-partition stationary, transposed from the gathered c_g.
    c_g_t_all = sbm.alloc_stack((_K_CHUNK, num_k_chunks, L), dtype=nl.bfloat16, buffer=nl.sbuf, name="c_g_t_all")
    p_t_all = sbm.alloc_stack((_K_CHUNK, num_k_chunks, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="p_t_all")

    for s_local in range(s_per_core):
        q_loc = s_start + s_local
        buf_idx = s_local % 2
        idx_i32, idx_u16 = idx_i32_bufs[buf_idx], idx_u16_bufs[buf_idx]
        q_lift_t, q_pe_t = q_lift_t_bufs[buf_idx], q_pe_t_bufs[buf_idx]
        c_g, k_pe_g, gathered_pos = c_g_bufs[buf_idx], k_pe_g_bufs[buf_idx], gathered_pos_bufs[buf_idx]

        if topk_tiled:
            # Query q_loc's [16, ti_f] tile sits at partitions [16g, 16g+16) of block [s_tile, t].
            s_tile, q_local = q_loc // _P_MAX, q_loc % _P_MAX
            t, g = q_local // 8, q_local % 8
            nisa.dma_copy(dst=idx_i32[0:16, :], src=topk_indices_hbm[s_tile, t, 16 * g : 16 * g + 16, :])
        else:
            nisa.dma_copy(
                dst=idx_i32[0:16, :], src=topk_indices_hbm.ap(pattern=[[ti_f, 16], [1, ti_f]], offset=q_loc * K)
            )
        nisa.tensor_scalar(idx_u16[0:16, :], idx_i32[0:16, :], op0=nl.multiply, operand0=1, engine=nisa.engine.vector)
        nisa.nc_stream_shuffle(dst=idx_u16[0:32, :], src=idx_u16[0:32, :], shuffle_mask=_TI_REPLICATE_MASK)
        for quad in range(1, _P_MAX // 32):
            nisa.dma_copy(dst=idx_u16[quad * 32 : quad * 32 + 32, :], src=idx_u16[0:32, :])

        # One dma_transpose: 3D source (H, n_l, 128_latent) permuted [2,1,0] -> [128_latent, n_l, H].
        nisa.dma_transpose(
            dst=q_lift_t, src=q_lift_hbm.ap(pattern=[[L, H], [_P_MAX, n_l], [1, _P_MAX]], offset=q_loc * HL)
        )
        nisa.dma_transpose(dst=q_pe_t, src=q_pe_hbm.ap(pattern=[[R, H], [1, R]], offset=q_loc * H * R))

        # Indirect gather via tensor_copy, alternating engine across latent tiles.
        for li in range(n_l):
            c_g_view = c_sb_tiles[li].indirect(idx_u16, num_elem=K)
            nisa.tensor_copy(
                dst=c_g[:, li, :], src=c_g_view, engine=nisa.engine.vector if li % 2 == 0 else nisa.engine.scalar
            )
        nisa.tensor_copy(dst=k_pe_g, src=k_pe_sb.indirect(idx_u16[0:R, :], num_elem=K), engine=nisa.engine.scalar)
        # Each score column's key position, gathered with the same idx_u16 as c_g.
        nisa.tensor_copy(dst=gathered_pos, src=pos_col.indirect(idx_u16, num_elem=K), engine=nisa.engine.vector)

        # Causal mask gpos > threshold, one pass: the fill multiply rides tensor_scalar's second op slot.
        nisa.tensor_scalar(qpos_thr, prior_col[0:H, :], op0=nl.add, operand0=q_loc, engine=nisa.engine.vector)
        nisa.tensor_scalar(
            mask_add,
            gathered_pos[0:H, :],
            op0=nl.greater,
            operand0=qpos_thr,
            op1=nl.multiply,
            operand1=_BF16_MASK_FILL,
            engine=nisa.engine.vector,
        )
        # MM1 one tile at a time into two rotating PSUM banks, each tile evicted as its matmuls land.
        for mm1_idx in range(num_mm1_tiles):
            off = mm1_idx * mm1_tile
            scores_psum = nl.ndarray(
                (H, mm1_tile), dtype=nl.float32, buffer=nl.psum, address=(0, (mm1_idx % 2) * PSUM_BANK_SIZE)
            )
            for li in range(n_l):
                nisa.nc_matmul(scores_psum, q_lift_t[:, li, :], c_g[:, li, off : off + mm1_tile], accumulate=(li > 0))
            nisa.nc_matmul(scores_psum, q_pe_t, k_pe_g[:, off : off + mm1_tile], accumulate=True)
            nisa.tensor_tensor(
                scores_sb[:, off : off + mm1_tile],
                scores_psum,
                mask_add[:, off : off + mm1_tile],
                op=nl.add,
                engine=nisa.engine.vector,
            )

        # MM2's keys-on-partition operand; depends only on the gather, so it is emitted before softmax.
        cgt_per_bank = _chunks_per_psum_bank(num_k_chunks, L * 2, PSUM_BANK_SIZE)
        for batch_idx in range(num_k_chunks // cgt_per_bank):
            c_g_t_psum = nl.ndarray(
                (_K_CHUNK, cgt_per_bank * L),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, (4 + batch_idx % 2) * PSUM_BANK_SIZE),
            )
            for j in range(cgt_per_bank):
                ks = (batch_idx * cgt_per_bank + j) * _K_CHUNK
                for li in range(n_l):
                    nisa.nc_transpose(
                        c_g_t_psum[:, j * L + li * _P_MAX : j * L + (li + 1) * _P_MAX], c_g[:, li, ks : ks + _K_CHUNK]
                    )
            nisa.tensor_scalar(
                c_g_t_all[:, batch_idx * cgt_per_bank : (batch_idx + 1) * cgt_per_bank, :],
                c_g_t_psum,
                op0=nl.multiply,
                operand0=1.0,
                engine=nisa.engine.vector if batch_idx % 2 == 0 else nisa.engine.scalar,
            )

        nisa.tensor_reduce(neg_row_max, op=nl.maximum, data=scores_sb, axis=1, negate=True)
        nisa.tensor_scalar(exp_bias, neg_row_max, op0=nl.multiply, operand0=softmax_scale, engine=nisa.engine.scalar)
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

        # p [H, K] -> p_t [K, H], a full bank of chunks per eviction.
        pt_per_bank = _chunks_per_psum_bank(num_k_chunks, H * 2, PSUM_BANK_SIZE)
        for batch_idx in range(num_k_chunks // pt_per_bank):
            p_t_psum = nl.ndarray(
                (_K_CHUNK, pt_per_bank * H),
                dtype=nl.bfloat16,
                buffer=nl.psum,
                address=(0, (1 + batch_idx % 2) * PSUM_BANK_SIZE),
            )
            for j in range(pt_per_bank):
                ks = (batch_idx * pt_per_bank + j) * _K_CHUNK
                nisa.nc_transpose(p_t_psum[:, j * H : (j + 1) * H], p[:, ks : ks + _K_CHUNK])
            nisa.tensor_copy(
                dst=p_t_all[:, batch_idx * pt_per_bank : (batch_idx + 1) * pt_per_bank, :],
                src=p_t_psum,
                engine=nisa.engine.scalar,
            )

        # MM2: out_T[d, h] = sum_k c_g_t[k, d] * p_t[k, h]; one accumulation bank per latent tile.
        pvt_psum = [
            nl.ndarray((_P_MAX, H), dtype=nl.float32, buffer=nl.psum, address=(0, (n_l + li) * PSUM_BANK_SIZE))
            for li in range(n_l)
        ]
        for chunk_idx in range(num_k_chunks):
            for li in range(n_l):
                nisa.nc_matmul(
                    pvt_psum[li],
                    c_g_t_all[:, chunk_idx, li * _P_MAX : (li + 1) * _P_MAX],
                    p_t_all[:, chunk_idx, :],
                    accumulate=(chunk_idx > 0),
                )
        for li in range(n_l):
            nisa.tensor_copy(dst=out_t_sb[:, li, :], src=pvt_psum[li], engine=nisa.engine.scalar)
        out_psum = nl.ndarray((H, L), dtype=nl.bfloat16, buffer=nl.psum, address=(0, 0))
        for li in range(n_l):
            nisa.nc_transpose(out_psum[:, li * _P_MAX : (li + 1) * _P_MAX], out_t_sb[:, li, :])

        # CROSS-KERNEL LAYOUT CONTRACT: 4-pack permuted (MX) or natural (BF16); see _mm2_out_aps.
        out_ap, psum_ap = _mm2_out_aps(out_bf16, out_psum, H, L, precision)
        nisa.tensor_scalar(out_ap, psum_ap, op0=nl.multiply, operand0=recip, engine=nisa.engine.scalar)
        nisa.dma_copy(dst=out_attn_hbm.ap(pattern=[[L, H], [1, L]], offset=q_loc * HL), src=out_bf16)

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
    prior_tokens_hbm=None,
    cp_rank_hbm=None,
    cp_world_size=1,
    cp_kv_cache_interleave_size=1,
    precision=MlaPrecision.MX,
    kv_cache_packed=False,
):
    """Dense latent + RoPE attention (no indexer), S-sharded, causal.

    The S <= index_topk case: the indexer is skipped and every query attends ALL S_kv keys,
    so there is NO topk gather -- MM1 reads the resident cache (c_sb_tiles / k_pe_sb) directly
    and K == S_kv.

    Non-CP (cp_world_size == 1): causality is a range_select on the scores, setting key column j
    to -inf when j > q_global (= prior + s_start + s_local); it also evicts PSUM -> SBUF and
    reduces the row max in the same instruction.

    CP interleave (cp_world_size > 1): key column j's GLOBAL position is
    owned(j // S_kv_local, j % S_kv_local), a non-monotonic permutation range_select's
    column-index bound cannot express, so a query-independent key-position row is built once and
    the mask is additive (key_global > q_global -> -inf) with a separate row-max reduce.

    Same output contract as the sparse stage: out_attn_hbm[B, S, H*L], each head's latent
    columns pre-permuted into MX 4-pack order (o_proj / W_uv / refs depend on it).

    kv_sbuf: optional PRE-GATHERED SBUF KV, see _load_kv_cache.
    """
    is_cp = cp_world_size > 1
    B, S, H, L = q_lift_hbm.shape
    R = q_pe_hbm.shape[3]
    HL = H * L
    n_l = L // _P_MAX

    # Dense attends all keys: K == S_kv. A packed cache is [S_kv, row], so S_kv is its major dim.
    if kv_sbuf is not None:
        S_kv_arg = kv_sbuf[2]
    elif kv_cache_packed:
        S_kv_arg = c_kv_hbm.shape[0]
    else:
        S_kv_arg = c_kv_hbm.shape[1]
    K = S_kv_arg
    n_pair = max(1, (L // 2) // _P_MAX)

    num_k_chunks = K // _K_CHUNK
    mm1_tile = min(_MM1_TILE, K)
    num_mm1_tiles = K // mm1_tile
    sm_tile = min(_SM_TILE, K)
    num_sm_tiles = K // sm_tile

    q_s_stride = H * L
    q_pe_s_stride = H * R
    PSUM_BANK_SIZE = _get_psum_bank_size()

    sbm.open_scope()

    if kv_cache_packed:
        # Dequantized ONCE here, so everything downstream is identical to the unpacked path.
        c_sb_tiles, k_pe_sb, S_kv, _scale_col, _c_g_t_packed = _load_packed_kv_resident(
            c_kv_hbm, R, L, n_l, sbm, num_k_chunks
        )
    else:
        c_sb_tiles, k_pe_sb, S_kv = _load_kv_cache(c_kv_hbm, k_pe_hbm, R, L, n_l, sbm, kv_sbuf)
        _c_g_t_packed = None
    base_dtype = nl.int32 if is_cp else nl.float32
    prior_col = _load_cp_scalar(prior_tokens_hbm, sbm, base_dtype, "prior_col", default=0)
    cp_rank_col = _load_cp_scalar(cp_rank_hbm, sbm, nl.int32, "cp_rank_col", default=0) if is_cp else None

    if is_cp:
        # CP interleave uses an additive mask: range_select's column-index bound cannot express the
        # non-monotonic physical->global permutation. key_global[j] = owned(j // S_kv_local,
        # j % S_kv_local) is built once and broadcast across the H score partitions.
        S_kv_local = S_kv // cp_world_size
        key_global = sbm.alloc_stack((H, K), dtype=nl.int32, buffer=nl.sbuf, name="key_global")
        _kg_rank = sbm.alloc_stack((H, K), dtype=nl.int32, buffer=nl.sbuf, name="kg_rank")
        _kg_hi = sbm.alloc_stack((H, K), dtype=nl.int32, buffer=nl.sbuf, name="kg_hi")
        _kg_lo = sbm.alloc_stack((H, K), dtype=nl.int32, buffer=nl.sbuf, name="kg_lo")
        nisa.iota(dst=key_global, pattern=[[1, K]], offset=0, channel_multiplier=0)  # j on free axis
        nisa.tensor_scalar(
            _kg_rank, key_global, op0=nl.right_shift, operand0=int(math.log2(S_kv_local)), engine=nisa.engine.vector
        )  # rank = j // S_kv_local
        nisa.tensor_scalar(
            key_global, key_global, op0=nl.bitwise_and, operand0=S_kv_local - 1, engine=nisa.engine.vector
        )  # key_global := l = j % S_kv_local
        _owned_positions(key_global, _kg_rank, key_global, _kg_hi, _kg_lo, cp_kv_cache_interleave_size, cp_world_size)
        # Per-query scratch (single: the additive mask serializes queries; correctness first).
        mask_add_dense = sbm.alloc_stack((H, K), dtype=nl.float32, buffer=nl.sbuf, name="mask_add_dense")
        q_global_i32 = sbm.alloc_stack((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name="q_global_i32")
        # operand0 must be float32; positions <= ~1M are exact in fp32. Cast per query.
        q_global_f32 = sbm.alloc_stack((_P_MAX, 1), dtype=nl.float32, buffer=nl.sbuf, name="q_global_f32")
        l_sb = sbm.alloc_stack((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name="l_sb")
        hi_sb = sbm.alloc_stack((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name="hi_sb")
        lo_sb = sbm.alloc_stack((_P_MAX, 1), dtype=nl.int32, buffer=nl.sbuf, name="lo_sb")
    else:
        # range_select needs two bounds; the causal condition only constrains the upper one.
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

    # Query-independent MM2 stationary: the resident cache transposed to keys-on-partition once.
    c_g_t_all = (
        _c_g_t_packed
        if _c_g_t_packed is not None
        else sbm.alloc_stack((_K_CHUNK, num_k_chunks, L), dtype=nl.bfloat16, buffer=nl.sbuf, name="c_g_t_all")
    )

    # SBUF buffer sets rotated by pbuf: per-query-private buffers get NUM_BUF copies.
    q_lift_t_bufs, q_pe_t_bufs, scores_sb_bufs, p_bufs = [], [], [], []
    q_lift_nat_bufs = []
    p_t_all_bufs, out_bf16_bufs = [], []
    qg_bufs, rmax_bufs, eb_bufs, rs_bufs, rc_bufs = [], [], [], [], []
    for b in range(NUM_BUF):
        q_lift_t_bufs.append(
            sbm.alloc_stack((_P_MAX, n_l, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qlt_{b}")
        )
        q_pe_t_bufs.append(sbm.alloc_stack((R, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qpt_{b}"))
        # Packed stages q in natural [H, L] order; the Tensor Engine applies the channel grouping below.
        q_lift_nat_bufs.append(
            sbm.alloc_stack((H, L), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"qln_{b}") if kv_cache_packed else None
        )
        # range_select writes SBUF, so scores land in scores_sb already masked.
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

    # Query-independent c_g_t[k, l] = c_kv[l, k], evicted once before the query loop, so it can
    # reuse buffer-0's rotating p_t banks.
    _cgt_bank = psum_base[0]["p_t"]
    for chunk_idx in range(0 if kv_cache_packed else num_k_chunks):
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
        q_lift_nat = q_lift_nat_bufs[pbuf]
        scores_sb = scores_sb_bufs[pbuf]
        p = p_bufs[pbuf]
        p_t_all = p_t_all_bufs[pbuf]
        out_bf16 = out_bf16_bufs[pbuf]
        q_global_sb = qg_bufs[pbuf]
        row_max, exp_bias, row_sum, recip = rmax_bufs[pbuf], eb_bufs[pbuf], rs_bufs[pbuf], rc_bufs[pbuf]
        bank = psum_base[pbuf]

        # Global query position for the causal mask. Non-CP: prior + q_idx, the range_select bound.
        # CP: owned(cp_rank, prior + q_idx) via the interleave map.
        if is_cp:
            nisa.tensor_scalar(l_sb, prior_col, op0=nl.add, operand0=q_idx, engine=nisa.engine.vector)
            _owned_positions(q_global_i32, cp_rank_col, l_sb, hi_sb, lo_sb, cp_kv_cache_interleave_size, cp_world_size)
            nisa.tensor_copy(dst=q_global_f32, src=q_global_i32, engine=nisa.engine.vector)
        else:
            nisa.tensor_scalar(q_global_sb, prior_col, op0=nl.add, operand0=q_idx, engine=nisa.engine.scalar)

        # Packed: q is loaded in c_sb_tiles' grouping, group li = par * n_pair + pi holding
        # channels 2 * (pi * _P_MAX + j) + par.
        if kv_cache_packed:
            # q is loaded in c_sb_tiles' grouping: group li = par * n_pair + pi holds latent channels
            # 2 * (pi * _P_MAX + j) + par on partition j.
            nisa.dma_copy(dst=q_lift_nat, src=q_lift_hbm.ap(pattern=[[L, H], [1, L]], offset=q_idx * q_s_stride))
            qlt_psum = nl.ndarray(
                (_P_MAX, n_l * H), dtype=nl.bfloat16, buffer=nl.psum, address=(0, bank["p_t"] * PSUM_BANK_SIZE)
            )
            for li in range(n_l):
                nisa.nc_transpose(
                    qlt_psum[:, li * H : (li + 1) * H],
                    q_lift_nat.ap(
                        pattern=[[L, H], [_FP8_TP_OUT_STEP, _P_MAX]],
                        offset=(li % n_pair) * _FP8_TP_OUT_STEP * _P_MAX + li // n_pair,
                    ),
                )
            nisa.tensor_copy(dst=q_lift_t, src=qlt_psum, engine=nisa.engine.vector)
        else:
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

        if is_cp:
            # Additive causal mask key_global[j] > q_global -> -inf, compared as
            # (key_global - q_global) > 0 because operand0 must be float32, then a separate row max.
            nisa.tensor_scalar(
                mask_add_dense,
                key_global[0:H, :],
                op0=nl.subtract,
                operand0=q_global_f32[0:H, :],
                op1=nl.greater,
                operand1=0,
                engine=nisa.engine.vector,
            )
            nisa.tensor_scalar(
                mask_add_dense, mask_add_dense, op0=nl.multiply, operand0=_FLOAT32_MIN, engine=nisa.engine.scalar
            )
            nisa.tensor_tensor(scores_sb, scores_psum, mask_add_dense, op=nl.add)
            nisa.tensor_reduce(row_max, op=nl.maximum, data=scores_sb, axis=1)
        else:
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

        # MM2: out_attn[h, d] = sum_j p[h,j] * c[d,j], emitted [H, L]. pv reuses the scores banks.
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
