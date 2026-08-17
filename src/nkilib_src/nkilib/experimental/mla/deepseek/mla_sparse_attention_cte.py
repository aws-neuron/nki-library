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

The MM2 output is written with each head's latent columns PRE-PERMUTED into MX 4-pack
order (natural latent l = 4*group + sub stored at physical column sub*(L//4) + group), the
cross-kernel layout contract the o_proj kernel's contiguous DMA-transpose load depends on.
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ....core.qkv.qkv_cte import _get_psum_bank_size
from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import get_verified_program_sharding_info
from .mla_common_cte import (
    _H_PACK,
    _K_CHUNK,
    _MM1_TILE,
    _P_MAX,
    _SM_TILE,
    _TI_REPLICATE_MASK,
    _new_sbm,
)
from .mla_validate_params import _validate_mla_attention_inputs


@nki.jit
def mla_sparse_attention_cte_kernel(
    q_lift_hbm: nl.NkiTensor,
    q_pe_hbm: nl.NkiTensor,
    c_kv_hbm: nl.NkiTensor,
    k_pe_hbm: nl.NkiTensor,
    topk_indices_hbm: nl.NkiTensor,
    softmax_scale: float,
    topk_tiled: bool = False,
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
        softmax_scale (float): Scaling factor applied to the attention scores. Must be
            positive. DeepSeek's scale is head_dim**-0.5 times a
            squared mscale correction, so it is always positive in practice.
        topk_tiled (bool): Select the topk_indices_hbm layout (default False = flat).

    Returns:
        out_attn (nl.NkiTensor): [B, S, H * L] bf16 latent attention output, row-major
            h * L + l, with each head's latent columns pre-permuted into MX 4-pack order.

    Notes:
        - S-sharded across cores; under Context Parallelism the framework gathers the
          latent KV to the full S_kv before this kernel is called.
        - The MM2 output is written with each head's latent columns pre-permuted into MX
          4-pack order (natural latent l = 4*group + sub at physical column sub*(L//4) +
          group), the cross-kernel layout contract the o_proj kernel's DMA-transpose load
          depends on. Keep in lockstep with the o_proj load, W_uv load, and both refs.

    Pseudocode:
        for q_idx in range(S_shard):
            c_g = c_kv[topk_indices[q_idx]]     # gather K latent rows
            k_pe_g = k_pe[topk_indices[q_idx]]  # gather K RoPE key rows
            scores = q_lift[q_idx] @ c_g.T + q_pe[q_idx] @ k_pe_g.T
            weights = softmax(scores * softmax_scale)
            out_attn[q_idx] = weights @ c_g     # [H, L], written 4-pack permuted
    """
    _validate_mla_attention_inputs(q_lift_hbm, q_pe_hbm, c_kv_hbm, k_pe_hbm, topk_indices_hbm, topk_tiled)
    B, S, H, L = q_lift_hbm.shape
    HL = H * L

    _, n_prgs, prg_id = get_verified_program_sharding_info("mla_sparse_attention_cte_kernel", (0, 1))
    kernel_assert(S % n_prgs == 0, f"S={S} must be divisible by n_prgs={n_prgs}")
    s_per_core = S // n_prgs
    s_start = prg_id * s_per_core

    out_attn_hbm = nl.ndarray((B, S, HL), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    sbm = _new_sbm("sparse_mla_latent_attn")
    _attention_stage(
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
    )
    return out_attn_hbm


def _attention_stage(
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
):
    """Sparse latent + RoPE attention, S-sharded.

    Computes attn_latent[H, L] per query for queries [s_start, s_start+s_per_core) and
    writes them to out_attn_hbm[B, S, H*L] (row-major h*L + l). Single source of truth
    for the standalone attention kernel (and reusable by a fused parent).

    kv_sbuf: optional (c_sb_tiles, k_pe_sb, S_kv) tuple of PRE-GATHERED SBUF KV in the
    [L_partition, S_kv_free] layout this stage consumes (n_l tiles of [P_MAX, S_kv] plus
    [R, S_kv]). When given, the KV is already resident (e.g. from a SB2SB CP all-gather)
    so this stage SKIPS allocating + dma_transpose-loading the cache from HBM (c_kv_hbm /
    k_pe_hbm are then unused for the cache). When None (default), the cache is loaded from
    HBM as before.
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

    if kv_sbuf != None:
        # KV already gathered & resident in [L, S_kv] SBUF layout (SB2SB CP gather): use
        # in place, no HBM cache load. S_kv derived from the provided tiles, not c_kv_hbm.
        c_sb_tiles, k_pe_sb, S_kv = kv_sbuf
    else:
        S_kv = c_kv_hbm.shape[1]
        c_sb_tiles = []
        for li in range(n_l):
            c_sb_tiles.append(sbm.alloc_stack((_P_MAX, S_kv), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"c_sb_{li}"))
        k_pe_sb = sbm.alloc_stack((R, S_kv), dtype=nl.bfloat16, buffer=nl.sbuf, name="k_pe_sb")

    ti_f = K // 16
    # Single-buffered: the kernel is dependency-bound on the per-query
    # MM1->softmax->transpose->MM2 chain, so double-buffering the inputs measured no gain.
    NUM_INPUT_BUFFERS = 1
    idx_i32_bufs, idx_u16_bufs, q_lift_t_bufs, q_pe_t_bufs, c_g_bufs, k_pe_g_bufs = [], [], [], [], [], []
    for buf_idx in range(NUM_INPUT_BUFFERS):
        idx_i32_bufs.append(sbm.alloc_stack((_P_MAX, ti_f), dtype=nl.int32, buffer=nl.sbuf, name=f"idx_i32_{buf_idx}"))
        idx_u16_bufs.append(sbm.alloc_stack((_P_MAX, ti_f), dtype=nl.uint16, buffer=nl.sbuf, name=f"idx_u16_{buf_idx}"))
        q_lift_t_bufs.append(
            sbm.alloc_stack((_P_MAX, n_l, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qlt_{buf_idx}")
        )
        q_pe_t_bufs.append(sbm.alloc_stack((R, H), dtype=nl.bfloat16, buffer=nl.sbuf, align=32, name=f"qpt_{buf_idx}"))
        c_g_bufs.append(sbm.alloc_stack((_P_MAX, n_l, K), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"c_g_{buf_idx}"))
        k_pe_g_bufs.append(sbm.alloc_stack((R, K), dtype=nl.bfloat16, buffer=nl.sbuf, name=f"k_pe_g_{buf_idx}"))

    p = sbm.alloc_stack((H, K), dtype=nl.bfloat16, buffer=nl.sbuf, name="p")
    neg_row_max = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="neg_row_max")
    exp_bias = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="exp_bias")
    row_sum = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="row_sum")
    recip = sbm.alloc_stack((H, 1), dtype=nl.float32, buffer=nl.sbuf, name="recip")
    out_bf16 = sbm.alloc_stack((H, L), dtype=nl.bfloat16, buffer=nl.sbuf, name="out_bf16")
    c_g_t_all = sbm.alloc_stack((_K_CHUNK, num_k_chunks, L), dtype=nl.bfloat16, buffer=nl.sbuf, name="c_g_t_all")
    p_t_all = sbm.alloc_stack((_K_CHUNK, num_k_chunks, H), dtype=nl.bfloat16, buffer=nl.sbuf, name="p_t_all")

    if kv_sbuf == None:
        # HBM path: transpose-load the cache [S_kv, L] -> [L, S_kv] SBUF tiles. (SB2SB
        # path already has these resident in the right layout — skip.)
        for li in range(n_l):
            nisa.dma_transpose(
                dst=c_sb_tiles[li], src=c_kv_hbm.ap(pattern=[[L, S_kv], [1, _P_MAX]], offset=li * _P_MAX)
            )
        nisa.dma_transpose(dst=k_pe_sb, src=k_pe_hbm.ap(pattern=[[R, S_kv], [1, R]], offset=0))

    for s_local in nl.affine_range(s_per_core):
        q_idx = s_start + s_local
        buf_idx = s_local % NUM_INPUT_BUFFERS
        idx_i32 = idx_i32_bufs[buf_idx]
        idx_u16 = idx_u16_bufs[buf_idx]
        q_lift_t = q_lift_t_bufs[buf_idx]
        q_pe_t = q_pe_t_bufs[buf_idx]
        c_g = c_g_bufs[buf_idx]
        k_pe_g = k_pe_g_bufs[buf_idx]

        if topk_tiled:
            """
            Read the indexer's natural partition tile directly (no flat re-tile, no
            scatter). Safe: sparse attn pools the K keys order-invariantly. Query q_idx's
            [16, ti_f] tile is at partitions [16g, 16g+16) of block [s_tile, t] in
            topk_tiled_hbm[num_s_tiles, NUM_TOPK_BATCHES, P_MAX, ti_f].
            """
            _s_tile = q_idx // _P_MAX
            _q_local = q_idx % _P_MAX
            _t = _q_local // 8
            _g = _q_local % 8
            nisa.dma_copy(
                dst=idx_i32[0:16, :],
                src=topk_indices_hbm[_s_tile, _t, 16 * _g : 16 * _g + 16, :],
            )
        else:
            nisa.dma_copy(
                dst=idx_i32[0:16, :],
                src=topk_indices_hbm.ap(pattern=[[ti_f, 16], [1, ti_f]], offset=q_idx * topk_s_stride),
            )
        nisa.tensor_scalar(idx_u16[0:16, :], idx_i32[0:16, :], op0=nl.multiply, operand0=1, engine=nisa.engine.vector)
        nisa.nc_stream_shuffle(dst=idx_u16[0:32, :], src=idx_u16[0:32, :], shuffle_mask=_TI_REPLICATE_MASK)
        for quad in range(1, _P_MAX // 32):
            nisa.dma_copy(dst=idx_u16[quad * 32 : quad * 32 + 32, :], src=idx_u16[0:32, :])

        # Load q_lift transposed to latent-on-partition in ONE dma_transpose: 3D source
        # (H, n_l, 128_latent) transposed [2,1,0] -> q_lift_t [128_latent, n_l, H].
        nisa.dma_transpose(
            dst=q_lift_t,
            src=q_lift_hbm.ap(pattern=[[L, H], [_P_MAX, n_l], [1, _P_MAX]], offset=q_idx * q_s_stride),
        )
        nisa.dma_transpose(dst=q_pe_t, src=q_pe_hbm.ap(pattern=[[R, H], [1, R]], offset=q_idx * q_pe_s_stride))

        """
        TI gather via tensor_copy (supports indirection on BOTH vector + scalar on
        gen4). Alternate the engine across latent tiles so the n_l gathers overlap
        instead of serializing — the vector engine is the kernel-A bottleneck (~80%)
        while scalar idles (~27%). tensor_copy needs no ti_ones multiply operand.
        """
        for li in range(n_l):
            c_g_view = c_sb_tiles[li].indirect(idx_u16, num_elem=K)
            g_engine = nisa.engine.vector if li % 2 == 0 else nisa.engine.scalar
            nisa.tensor_copy(dst=c_g[:, li, :], src=c_g_view, engine=g_engine)
        k_pe_view = k_pe_sb.indirect(idx_u16[0:R, :], num_elem=K)
        nisa.tensor_copy(dst=k_pe_g, src=k_pe_view, engine=nisa.engine.scalar)

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

        """
        c_g transpose for MM2 (keys-on-partition), hoisted before softmax (depends only on
        the gather, not on p) so the TE transpose overlaps softmax latency.
        """
        for chunk_idx in range(num_k_chunks):
            ks = chunk_idx * _K_CHUNK
            par = chunk_idx % 2
            c_g_t_psum = nl.ndarray(
                (_K_CHUNK, L), dtype=nl.bfloat16, buffer=nl.psum, address=(0, (4 + par) * PSUM_BANK_SIZE)
            )
            for li in range(n_l):
                nisa.nc_transpose(c_g_t_psum[:, li * _P_MAX : (li + 1) * _P_MAX], c_g[:, li, ks : ks + _K_CHUNK])
            nisa.tensor_scalar(
                c_g_t_all[:, chunk_idx, :], c_g_t_psum, op0=nl.multiply, operand0=1.0, engine=nisa.engine.vector
            )

        nisa.tensor_reduce(neg_row_max, op=nl.maximum, data=scores_psum, axis=1, negate=True)
        nisa.tensor_scalar(exp_bias, neg_row_max, op0=nl.multiply, operand0=softmax_scale, engine=nisa.engine.vector)
        for sm_idx in range(num_sm_tiles):
            so = sm_idx * sm_tile
            nisa.activation(
                dst=p[:, so : so + sm_tile],
                op=nl.exp,
                data=scores_psum[:, so : so + sm_tile],
                bias=exp_bias,
                scale=softmax_scale,
                reduce_op=nl.add,
                reduce_res=row_sum if sm_idx == num_sm_tiles - 1 else None,
                reduce_cmd=nisa.reduce_cmd.reset_reduce if sm_idx == 0 else nisa.reduce_cmd.reduce,
            )
        nisa.reciprocal(recip, row_sum)

        for chunk_idx in range(num_k_chunks):
            ks = chunk_idx * _K_CHUNK
            par = chunk_idx % 2
            p_t_psum = nl.ndarray(
                (_K_CHUNK, H), dtype=nl.bfloat16, buffer=nl.psum, address=(0, (1 + par) * PSUM_BANK_SIZE)
            )
            nisa.nc_transpose(p_t_psum, p[:, ks : ks + _K_CHUNK])
            nisa.tensor_scalar(
                p_t_all[:, chunk_idx, :], p_t_psum, op0=nl.multiply, operand0=1.0, engine=nisa.engine.scalar
            )

        # MM2: out_attn[h, d] = sum_j p[h,j] * c_g[d,j]. Emitted [H, L] (H-on-partition).
        pv_psum = nl.ndarray((H, L), dtype=nl.float32, buffer=nl.psum, address=(0, 0))
        for chunk_idx in range(num_k_chunks):
            nisa.nc_matmul(pv_psum, p_t_all[:, chunk_idx, :], c_g_t_all[:, chunk_idx, :], accumulate=(chunk_idx > 0))

        """
        CROSS-KERNEL MX LAYOUT CONTRACT: write each head's latent pre-permuted into 4-pack
        order (natural l = 4*group + sub at column sub*(L//4) + group), so the o_proj consumer
        transposes contiguous groups via dma_transpose (not nc_transpose); W_uv loads with the
        SAME permutation. Done here as a vector free-axis permute (a permuted HBM write would be
        a ~14x-slower scatter). Keep in lockstep with the o_proj load, W_uv load, and both refs.
        """
        nisa.tensor_scalar(
            out_bf16.ap(pattern=[[L, H], [1, L // _H_PACK], [L // _H_PACK, _H_PACK]]),
            pv_psum.ap(pattern=[[L, H], [_H_PACK, L // _H_PACK], [1, _H_PACK]]),
            op0=nl.multiply,
            operand0=recip,
            engine=nisa.engine.vector,
        )
        nisa.dma_copy(dst=out_attn_hbm.ap(pattern=[[L, H], [1, L]], offset=q_idx * HL), src=out_bf16)

    sbm.close_scope()
