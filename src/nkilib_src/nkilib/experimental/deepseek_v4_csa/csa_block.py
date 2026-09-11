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

"""Whole DeepSeek-V4 CSA attention blocks -- the composition layer over the kernels.

This is where the kernels in ``csa_prefill_attention`` / ``csa_decode_attention``
and the collective in ``csa_tp_all_reduce`` are wired into complete attention
blocks that take a raw hidden state and return the projected block output. It also
holds the host-side glue the kernels need (index reformatting, mask assembly,
tensor layout changes) and a ``main()`` that traces a block and checks it against
the CPU reference in ``csa_block_torch``.

Topology
--------
One chip holds the model as ``tp_size`` HEAD-PARALLEL tensor-parallel ranks, so
each rank owns ``n_heads / tp_size`` query heads and the matching output
projection groups. Each rank traces at ``--logical-nc-config=2`` and its
``ncc.all_reduce`` sums the RowParallelLinear partials across ranks, so one
``torch_neuronx.trace`` of one rank emits ONE NEFF holding the projections, the
sparse attention and the cross-rank reduction.

Decode issues three ``@nki.jit`` launches per step:

======================================== ======== =========================================
launch                                   grid     work
======================================== ======== =========================================
``nki_qkv_rms_rope_kernel``               ``[1]``  RMSNorm + RoPE for the query heads and
                                                   the new KV token, on one packed tile
``nki_indexer_qproj_gemv``                ``[2]``  the indexer query projection as a
                                                   hand-tiled GEMV
``nki_indexer_score_topk_gather_2core``   ``[2]``  indexer score, top-k, and the O(k)
                                                   sparse attention, fused
======================================== ======== =========================================

Prefill instead calls the RMS+RoPE kernel three times (q, kv, output de-RoPE)
around the compressor, indexer and the two sparse-attention kernels.

Not covered by the integration tests
------------------------------------
The classes here interleave torch projections with NKI launches and, on the
multi-worker path, span several ranks, so they are outside what the kernel test
framework traces. The per-kernel numerics live in the integration tests; this
module's own end-to-end check is ``main()`` against ``csa_block_torch``.
"""

import os

import torch
import torch.nn.functional as F
from torch import nn

from .csa_common import (
    CSAConfig,
    RMSNorm,
    apply_rotary_emb_functional,
    hadamard_transform,
    precompute_freqs_cos_sin,
    precompute_win_bias_parts,
)
from .csa_decode_attention import (
    NISA_TOPK_GROUP_SIZE,
    NISA_TOPK_GROUPS_PER_CALL,
    NISA_TOPK_PARTITIONS,
    nisa_topk_snake_kernel,
    nki_decode_gather_ok_kernel,
    nki_indexer_qproj_gemv,
    nki_indexer_score_2core,
    nki_indexer_score_kernel,
    nki_indexer_score_topk_2core,
    nki_indexer_score_topk_gather_2core,
    nki_indexer_score_topk_kernel,
    nki_qkv_rms_rope_kernel,
)
from .csa_prefill_attention import (
    nki_compressor_core_kernel,
    nki_fused_csa_attn_kernel,
    nki_gather_csa_attn_kernel,
    nki_indexer_score_mask_kernel,
    nki_prefill_sparse_attn_kernel,
    nki_prefill_topk_kernel,
    nki_rms_rope_kernel,
)
from .csa_tp_all_reduce import tp_all_gather_rows, tp_all_reduce


_SPARSE_PREFILL_MODE = os.environ.get("CSA_SPARSE_PREFILL", "auto")
_SPARSE_MIN_T_C = 4096
# Proven-safe nisa.topk width; see the note at the padding site below.
_SAFE_TOPK_N = 8192
_SPARSE_TILE_Q = int(os.environ.get("CSA_SPARSE_TILE_Q", "1024"))


def sparse_prefill_q_range(seq_len: int, t_c: int, index_topk: int, ratio: int, tp_rank: int, tp_size: int):
    """This rank's contiguous output-row range under SEQUENCE-parallel prefill.

    The sparse kernel needs all ``n_heads`` on one core, so the ranks split the
    SEQUENCE instead of the heads: replicated compressed KV, queries divided, softmax
    therefore entirely local -- the reduction runs over the key axis, which sequence
    sharding does not split.

    Queries below ``split_pos = index_topk * ratio`` have fewer causal compressed
    positions than ``index_topk``, so they select ALL of them and there is no sparsity
    to exploit; that region stays on the dense kernel. Rank 0 owns it, because keeping
    every rank's rows CONTIGUOUS lets the driver concatenate the partials instead of
    gathering them.

    The row counts are equalised, though. Handing rank 0 the whole dense region on top
    of a full share of the scored region gave it 11264 of 32768 rows against 7168 for
    the others -- 1.57x the work on what is the tp4 critical path, since the ranks run
    concurrently and the block finishes with the slowest. Rank 0 instead takes exactly
    ``seq_len / tp_size`` rows (the dense region plus however much of the scored region
    fills its share) and the remainder divides among the rest.

    Returns ``(lo, hi)`` half-open, in query positions.
    """
    del t_c, index_topk, ratio
    per = seq_len // tp_size
    if tp_rank == 0:
        return 0, per
    share = (seq_len - per) // (tp_size - 1)
    lo = per + (tp_rank - 1) * share
    return lo, lo + share


def compress_sharded(compressor, x, start_pos, freqs_cos_sin, tp_shard):
    """Compressed KV, computed on this rank's slice only and all-gathered to full.

    ``tp_shard`` is ``(tp_rank, replica_ranks)`` or None. This is the "each rank does a
    gather of KV" half of the sequence-parallel design: a query's top-k may select ANY
    compressed position, so every rank needs all ``T_c`` of them -- but only 1/tp of them
    need to be COMPUTED locally. The compressor is pointwise up to a one-group halo, so
    the shards are independent and the concatenation is bit-exact against computing the
    whole thing (verified: max_abs 0.0 at seq_len 8192/16384/32768, tp=4).
    """
    if tp_shard is None:
        return compressor(x, start_pos, freqs_cos_sin)
    tp_rank, replica_ranks = tp_shard
    world = len(replica_ranks)
    if world == 1:
        return compressor(x, start_pos, freqs_cos_sin)

    t_c = x.shape[1] // compressor.compress_ratio
    if t_c % world != 0:
        # An uneven split needs all_gather_v; fall back rather than mis-gather.
        return compressor(x, start_pos, freqs_cos_sin)
    per = t_c // world
    shard = compressor(x, start_pos, freqs_cos_sin, t_range=(tp_rank * per, (tp_rank + 1) * per))
    head_dim = shard.shape[-1]
    gathered = tp_all_gather_rows(shard.reshape(per, head_dim), replica_ranks)
    return gathered.reshape(1, t_c, head_dim)


def _seq_parallel_prefill(phase: str, full_config: CSAConfig) -> bool:
    """Does this prefill run shard the SEQUENCE rather than the heads?

    Only when the sparse second half is selected, because that kernel needs all
    ``n_heads`` on one core. Dense prefill stays head-parallel and byte-identical.
    """
    if phase != "prefill":
        return False
    if os.environ.get("CSA_SEQ_PARALLEL", "") == "1":
        # Diagnostic: sequence-parallel sharding INDEPENDENT of the sparse dispatch, so
        # the sharding and the sparse kernel can be bisected against each other.
        return True
    return _use_sparse_prefill(full_config.compressed_len)


def _use_sparse_prefill(t_c: int) -> bool:
    """Trace-time: is the O(k) sparse second half the cheaper kernel at this T_c?"""
    if _SPARSE_PREFILL_MODE == "1":
        return True
    if _SPARSE_PREFILL_MODE == "0":
        return False
    return t_c >= _SPARSE_MIN_T_C


def encode_snake(scores, n):
    """Encode [rows, n] scores into snake layout [rows//8, 128, n//16] for nisa.topk."""
    rows = scores.shape[0]
    src_x = n // NISA_TOPK_GROUP_SIZE
    num_batches = rows // NISA_TOPK_GROUPS_PER_CALL
    # scores[row, j] → snake[row_local*16 + j%16, j//16]
    # Reshape [rows, n] → [rows, src_x, 16] → permute → [rows, 16, src_x]
    snake = scores.reshape(rows, src_x, NISA_TOPK_GROUP_SIZE).permute(0, 2, 1).contiguous()
    # Pack batches of 8 rows into 128 partitions: [num_batches, 128, src_x]
    snake = snake.reshape(num_batches, NISA_TOPK_PARTITIONS, src_x)
    # Flatten batch dim for kernel: [num_batches * 128, src_x]
    return snake.reshape(num_batches * NISA_TOPK_PARTITIONS, src_x)


def decode_snake(vals_snake, idxs_snake, rows, k):
    """Decode snake layout [rows//8 * 128, k] → [rows, k] values and indices."""
    num_batches = rows // NISA_TOPK_GROUPS_PER_CALL
    k_cols = k // NISA_TOPK_GROUP_SIZE
    # Reshape to [num_batches, 8, 16, k] and take meaningful columns
    vals = vals_snake.reshape(num_batches, NISA_TOPK_GROUPS_PER_CALL, NISA_TOPK_GROUP_SIZE, k)
    idxs = idxs_snake.reshape(num_batches, NISA_TOPK_GROUPS_PER_CALL, NISA_TOPK_GROUP_SIZE, k)
    # Only first k_cols columns per partition are meaningful
    vals = vals[:, :, :, :k_cols]  # [num_batches, 8, 16, k_cols]
    idxs = idxs[:, :, :, :k_cols]
    # Snake decode: result j → partition j%16, column j//16
    # Permute [batch, group, part, col] → [batch, group, col, part] then reshape
    vals = vals.permute(0, 1, 3, 2).reshape(rows, k)
    idxs = idxs.permute(0, 1, 3, 2).reshape(rows, k)
    return vals, idxs


def nisa_topk_batched(scores, k, n_cores=2):
    """GpSimd top-k over [rows, n] scores -> (values [rows, k], indices [rows, k]).

    Uses nisa.topk (GPSIMD) with snake layout encode/decode.
    rows must be divisible by 8. n must be divisible by 16. k must be divisible by 16.
    """
    rows, n = scores.shape
    scores_bf16 = scores.to(torch.bfloat16)
    snake_input = encode_snake(scores_bf16, n)
    vals_snake, idxs_snake = nisa_topk_snake_kernel[n_cores](snake_input, k, n)
    vals, idxs = decode_snake(vals_snake, idxs_snake, rows, k)
    return vals, idxs.int()


def prefill_second_half_attention(
    selection: torch.Tensor,  # [S_q, T_c] 0/-1e9 bias (dense) or [S_q, k] positions (sparse)
    q_second: torch.Tensor,  # [S_q, n_heads, head_dim] fp16, already softmax-scaled
    win_K_T: torch.Tensor,  # [head_dim, S_q + W] window K^T, front-padded by W
    win_V: torch.Tensor,  # [S_q + W, head_dim] window V, front-padded by W
    compress_K_T: torch.Tensor,  # [head_dim, T_c] compressed K^T
    compress_V: torch.Tensor,  # [T_c, head_dim] compressed V
    win_bias_base: torch.Tensor,  # [S_q, W] window causal bias
    win_bias_sink_ind: torch.Tensor,  # [S_q, W] sink-column indicator
    attn_sink: torch.Tensor,  # [1, n_heads] per-head sink scalar
    s_lo: int,  # global position of this block's first query row
    ratio: int,  # compress_ratio
    sparse: bool,  # which kernel; a trace-time choice on a compile-time shape
    tile_q: int = _SPARSE_TILE_Q,
) -> torch.Tensor:
    """Attention for the SCORED half of prefill. Returns [n_heads, S_q, head_dim].

    One signature for both kernels, because they compute the same thing from the same
    operands. What differs is internal and stays internal:

    * ``selection`` is a 0/-1e9 additive bias over all ``T_c`` for the dense kernel and a
      list of ``k`` compressed POSITIONS for the sparse one -- the same operand slot, a
      different encoding, which is the whole point of the sparse path.
    * ``q_second`` arrives QUERY-major for both. The sparse kernel wants exactly that (a
      query's heads are then contiguous, so it can ``dma_transpose`` them instead of
      paying one DMA descriptor per element); the dense kernel wants head-major and gets
      it from the permute below, which is the same permute it used to do at the call site.
    * the sparse kernel is launched per query tile because it unrolls over its query
      count, so one launch for the whole half would not compile. The dense kernel takes
      the half in one launch. Either way the caller gets one ``[n_heads, S_q, head_dim]``.
    """
    S_q, n_heads, head_dim = q_second.shape
    win = win_K_T.shape[1] - S_q

    if not sparse:
        # Head-major Q^T: [head_dim, n_heads * S_q] indexed h * S_q + s.
        all_q_T = q_second.permute(2, 1, 0).reshape(head_dim, n_heads * S_q)
        out_flat = nki_gather_csa_attn_kernel[2](
            selection.to(torch.bfloat16),
            all_q_T,
            win_K_T,
            win_V,
            compress_K_T,
            compress_V,
            win_bias_base,
            win_bias_sink_ind,
            attn_sink,
            s_lo,
            ratio,
        )
        return out_flat.reshape(n_heads, S_q, head_dim)

    topk_idx = selection.to(torch.int32)
    q_major = q_second.reshape(S_q * n_heads, head_dim)
    compress_V_f16 = compress_V.to(torch.float16)
    win_K_T_f16 = win_K_T.to(torch.float16)
    win_V_f16 = win_V.to(torch.float16)
    tiles = []
    for t0 in range(0, S_q, tile_q):
        tq = min(tile_q, S_q - t0)
        out_t = nki_prefill_sparse_attn_kernel[2](
            topk_idx[t0 : t0 + tq].transpose(0, 1).contiguous(),
            q_major[t0 * n_heads : (t0 + tq) * n_heads],
            win_K_T_f16[:, t0 : t0 + tq + win],
            win_V_f16[t0 : t0 + tq + win],
            compress_V_f16,
            attn_sink,
        )
        tiles.append(out_t.reshape(n_heads, tq, head_dim))
    return tiles[0] if len(tiles) == 1 else torch.cat(tiles, dim=1)


def nki_fused_csa_attn(
    q,
    kv_raw,
    kv_compress,
    attn_sink,
    window_size,
    compress_sel_mask,
    softmax_scale,
    win_bias_base,
    win_bias_sink_ind,
    split_pos=0,
    T_c_first=0,
    first_mask=None,
):
    """NKI fused attention: window + compressed with online softmax.

    Uses multi-head kernel that processes all heads in a single call per sub-split,
    sharing mask, K^T, and V loads across heads to reduce DMA bandwidth.
    Window bias is computed inline in the kernel from base + attn_sink * sink_ind.
    """
    B, S, n_heads, head_dim = q.shape
    T_c = kv_compress.shape[1]
    W = window_size

    q_scaled = (q * softmax_scale).to(torch.float16)
    q_T = q_scaled.permute(0, 2, 3, 1).reshape(B * n_heads, head_dim, S)

    kv_raw_f16 = kv_raw.to(torch.float16)
    kv_raw_bf16 = kv_raw.to(torch.bfloat16)
    kv_compress_f16 = kv_compress.to(torch.float16)
    kv_compress_bf16 = kv_compress.to(torch.bfloat16)

    kv_raw_K_T = kv_raw_f16.transpose(1, 2)
    kv_compress_K_T = kv_compress_f16.transpose(1, 2)

    # attn_sink reshaped to [1, n_heads] for kernel consumption
    attn_sink_2d = attn_sink.detach().view(1, n_heads).float().contiguous()

    if split_pos > 0 and split_pos < S and T_c_first > 0 and T_c_first < T_c:
        raw_padded_K_T = F.pad(kv_raw_K_T, (W, 0)).reshape(head_dim, S + W)
        raw_padded_V = F.pad(kv_raw_bf16, (0, 0, W, 0)).reshape(S + W, head_dim)
        compress_K_T_2d = kv_compress_K_T.reshape(head_dim, T_c)
        compress_V_2d = kv_compress_bf16.reshape(T_c, head_dim)

        # First half: T_c_first columns of compressed KV
        first_mask_2d = first_mask.reshape(split_pos, T_c_first)
        first_kt = torch.cat([raw_padded_K_T[:, : split_pos + W], compress_K_T_2d[:, :T_c_first]], dim=1)
        first_v = torch.cat([raw_padded_V[: split_pos + W, :], compress_V_2d[:T_c_first, :]], dim=0)

        all_q_T_first = q_T[:, :, :split_pos].permute(1, 0, 2).reshape(head_dim, n_heads * split_pos)
        out_first_flat = nki_fused_csa_attn_kernel[2](
            first_mask_2d,
            all_q_T_first,
            first_kt,
            first_v,
            win_bias_base[:split_pos],
            win_bias_sink_ind[:split_pos],
            attn_sink_2d,
        )
        out_first_all = out_first_flat.reshape(n_heads, split_pos, head_dim)

        # Second half: full T_c compressed columns
        second_mask_2d = compress_sel_mask.reshape(S - split_pos, T_c).to(torch.bfloat16)
        second_kt = torch.cat([raw_padded_K_T[:, split_pos:], compress_K_T_2d], dim=1)
        second_v = torch.cat([raw_padded_V[split_pos:, :], compress_V_2d], dim=0)

        S_second_len = S - split_pos
        all_q_T_second = q_T[:, :, split_pos:].permute(1, 0, 2).reshape(head_dim, n_heads * S_second_len)
        out_second_flat = nki_fused_csa_attn_kernel[2](
            second_mask_2d,
            all_q_T_second,
            second_kt,
            second_v,
            win_bias_base[split_pos:],
            win_bias_sink_ind[split_pos:],
            attn_sink_2d,
        )
        out_second_all = out_second_flat.reshape(n_heads, S - split_pos, head_dim)

        out_all = torch.cat([out_first_all, out_second_all], dim=1)
        out = out_all.permute(1, 0, 2)  # [S, n_heads, head_dim]
    else:
        mask_2d = compress_sel_mask.reshape(S, T_c).to(torch.bfloat16)
        all_K_T = F.pad(torch.cat([kv_raw_K_T, kv_compress_K_T], dim=2), (W, 0))
        all_V = F.pad(torch.cat([kv_raw_bf16, kv_compress_bf16], dim=1), (0, 0, W, 0))
        total_free = S + W + T_c
        kt_2d = all_K_T.reshape(head_dim, total_free)
        v_2d = all_V.reshape(total_free, head_dim)

        all_q_T_full = q_T.permute(1, 0, 2).reshape(head_dim, n_heads * S)
        out_flat = nki_fused_csa_attn_kernel[2](
            mask_2d, all_q_T_full, kt_2d, v_2d, win_bias_base, win_bias_sink_ind, attn_sink_2d
        )
        out = out_flat.reshape(n_heads, S, head_dim).permute(1, 0, 2)

    return out.unsqueeze(0)  # [1, S, n_heads, head_dim]


# --------------------------------------------------------------------------
# Compressor (stateless, functional)
# --------------------------------------------------------------------------
class CompressorNKI(nn.Module):
    def __init__(self, config, head_dim: int = 512, rotate: bool = False, use_nki: bool = True):
        super().__init__()
        self.dim = config.dim
        self.head_dim = head_dim
        self.rope_head_dim = config.rope_head_dim
        self.compress_ratio = config.compress_ratio
        self.overlap = config.compress_ratio == 4
        self.rotate = rotate
        self.use_nki = use_nki
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(config.compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = nn.Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.out_dim = coff * self.head_dim
        self.norm = RMSNorm(self.head_dim, config.norm_eps)

    def overlap_transform_functional(self, tensor, value=0):
        b, s, ratio, _ = tensor.size()
        d = self.head_dim
        first_half = tensor[..., :d]
        second_half = tensor[..., d:]
        top = F.pad(first_half[:, :-1], (0, 0, 0, 0, 1, 0), value=value)
        return torch.cat([top, second_half], dim=2)

    def _compress_from_kv_score(self, kv_score, seqlen, freqs_cos_sin):
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        kv = kv_score[..., : self.out_dim]
        score = kv_score[..., self.out_dim :]

        remainder = seqlen % ratio
        cutoff = seqlen - remainder

        if remainder > 0:
            kv = kv[:, :cutoff]
            score = score[:, :cutoff]

        kv = kv.unflatten(1, (-1, ratio))
        score = score.unflatten(1, (-1, ratio)) + self.ape

        if self.overlap:
            kv = self.overlap_transform_functional(kv, 0)
            score = self.overlap_transform_functional(score, -1e9)

        freqs_cos, freqs_sin = freqs_cos_sin
        compress_cos = freqs_cos[:cutoff:ratio]
        compress_sin = freqs_sin[:cutoff:ratio]

        if self.use_nki and not self.rotate and self.overlap and kv.shape[0] == 1:
            return self._compress_core_nki(kv, score, compress_cos, compress_sin)

        weights = score.softmax(dim=2)
        kv = (kv * weights).sum(dim=2)

        kv = self.norm(kv.to(torch.bfloat16))

        kv_nope = kv[..., :-rd]
        kv_rope = apply_rotary_emb_functional(kv[..., -rd:], (compress_cos, compress_sin))
        kv = torch.cat([kv_nope, kv_rope], dim=-1)

        if self.rotate:
            kv = hadamard_transform(kv)

        return kv

    def _compress_core_nki(self, kv, score, compress_cos, compress_sin):
        """Run the gated-pooling + RMSNorm + RoPE core in a single NKI kernel.

        Args (post overlap_transform):
            kv:    [1, T_c, ratio2, head_dim] (fp32)
            score: [1, T_c, ratio2, head_dim] (fp32)
            compress_cos/sin: [T_c, rope_head_dim // 2] (fp32)
        Returns:
            [1, T_c, head_dim] bf16
        """
        T_c = kv.shape[1]
        hd = self.head_dim

        # Drop the batch dim and make slot-major contiguous: [T_c, ratio2, head_dim].
        kv8 = kv[0].contiguous().float()
        score8 = score[0].contiguous().float()

        norm_weight = self.norm.weight.detach().view(1, hd).float().contiguous()

        # Repeat each per-pair cos/sin so adjacent channels share the same value:
        # cos_rep[t, 2i] = cos_rep[t, 2i+1] = compress_cos[t, i].
        cos_rep = compress_cos.float().repeat_interleave(2, dim=-1).contiguous()
        sin_rep = compress_sin.float().repeat_interleave(2, dim=-1).contiguous()

        # SPMD across compressed-position tiles (query-row-analog for the compressor):
        # the kernel splits its 128-position tiles across cores with no cross-core
        # reduction. Use 2 cores when the tile count splits evenly; else single core.
        TILE_P = 128
        num_tiles = (T_c + TILE_P - 1) // TILE_P
        n_cores = 2 if (T_c % TILE_P == 0 and num_tiles % 2 == 0) else 1
        out = nki_compressor_core_kernel[n_cores](kv8, score8, norm_weight, cos_rep, sin_rep, float(self.norm.eps))
        return out.unsqueeze(0)

    def forward(self, x, start_pos, freqs_cos_sin, t_range=None):
        """Compress ``x`` into compressed cache positions.

        ``t_range=(t0, t1)`` computes only compressed positions ``[t0, t1)``, which is
        what sequence-parallel prefill wants: each rank produces its own slice and the
        ranks all-gather. It is NOT simply ``x[ratio*t0 : ratio*t1]``, because with
        ``overlap`` (compress_ratio == 4) compressed position ``t`` pools the second half
        of raw group ``t`` AND the first half of group ``t - 1`` -- see
        ``overlap_transform_functional``, which shifts ``first_half`` down by one group.
        So the slice carries a ONE-GROUP (``ratio`` raw tokens) halo at the front and the
        halo's own compressed position is dropped afterwards. Its ``first_half`` would have
        been zero-padded, which is only the right answer at the true sequence start, i.e.
        for ``t0 == 0`` -- and there no halo is taken, so the padding is genuine.
        """
        bsz, seqlen, _ = x.size()

        if seqlen < self.compress_ratio:
            return None

        if t_range is not None:
            ratio = self.compress_ratio
            t0, t1 = t_range
            halo = 1 if t0 > 0 else 0
            lo = ratio * (t0 - halo)
            hi = ratio * t1
            freqs_cos, freqs_sin = freqs_cos_sin
            shard = self.forward(x[:, lo:hi], start_pos + lo, (freqs_cos[lo:], freqs_sin[lo:]))
            return shard if halo == 0 else shard[:, halo:]

        W = torch.cat([self.wkv.weight, self.wgate.weight], dim=0).to(torch.bfloat16)
        kv_score = F.linear(x.to(torch.bfloat16), W).float()
        return self._compress_from_kv_score(kv_score, seqlen, freqs_cos_sin)


# --------------------------------------------------------------------------
# Indexer (stateless, functional) — with split scoring optimization
# --------------------------------------------------------------------------
class IndexerNKI(nn.Module):
    def __init__(self, config, use_nki: bool = True):
        super().__init__()
        # Sequence-parallel prefill: (lo, hi) rows this rank scores; None = all rows.
        self.q_range = None
        # (tp_rank, replica_ranks) when the indexer's compressed KV is sharded + gathered.
        self.tp_shard = None
        self.dim = config.dim
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False, dtype=torch.bfloat16)
        self.weight_scale = self.softmax_scale * (self.n_heads**-0.5)
        self.compressor = CompressorNKI(config, self.head_dim, rotate=True, use_nki=use_nki)

        # Precompute causal bias masks for fixed seq_len.
        # Collapsed (single-kernel) path: one full causal bias [S_q, T_c] covering
        # the scored second half [split_pos:seqlen] against all T_c compressed kv.
        seqlen = config.seq_len
        ratio = config.compress_ratio
        T_c_idx = seqlen // ratio
        k = min(config.index_topk, seqlen // ratio)
        if k < T_c_idx:
            split_pos = k * ratio

            # first_mask: causal mask for positions 0..split_pos-1. Only need [split_pos, k]
            # columns since positions beyond k are always -inf for those query rows.
            kv_pos_first = torch.arange(k).view(1, -1)
            vc_first = (torch.arange(1, split_pos + 1) // ratio).view(-1, 1)
            first_mask_buf = torch.where(kv_pos_first < vc_first, torch.zeros(1), torch.tensor(-1e9)).to(torch.bfloat16)
            self.register_buffer("first_mask_buf", first_mask_buf, persistent=False)

            # Full causal bias for the scored second half: rows = queries
            # [split_pos:seqlen], cols = compressed kv [0:T_c_idx].
            S_q = seqlen - split_pos
            kv_pos = torch.arange(T_c_idx).unsqueeze(0)
            valid_counts_idx = (torch.arange(split_pos + 1, seqlen + 1) // ratio).unsqueeze(1)
            causal_bias = torch.where(kv_pos < valid_counts_idx, torch.zeros(1), torch.tensor(-1e9))
            self.register_buffer("causal_bias_full", causal_bias.float().contiguous(), persistent=False)
            # Zero bias used for start_pos != 0 (decode) — no causal masking on scores.
            self.register_buffer("zero_bias_full", torch.zeros(S_q, T_c_idx, dtype=torch.float32), persistent=False)

    def _build_mask_from_scores(self, scores, k, T_c_out, device):
        """Build selection mask from scores using binary-search threshold finding.
        Uses bisection to find the k-th largest value per row, then generates mask.
        Selects all elements >= threshold (may select slightly more than k in case
        of ties, which is acceptable for attention masking)."""
        _NEG_INF = -1e9
        T_c_local = scores.shape[2]

        scores = scores.float()
        hi = scores.max(dim=-1, keepdim=True).values
        lo = torch.where(scores > -1e8, scores, hi).min(dim=-1, keepdim=True).values

        for _ in range(9):
            mid = (lo + hi) * 0.5
            count = (scores >= mid).to(scores.dtype).sum(dim=-1, keepdim=True)
            lo = torch.where(count >= k, mid, lo)
            hi = torch.where(count < k, mid, hi)

        sel_mask = torch.where(scores >= lo, 0.0, _NEG_INF)

        if T_c_local < T_c_out:
            sel_mask = F.pad(sel_mask, (0, T_c_out - T_c_local), value=_NEG_INF)
        return sel_mask

    def forward(self, x, qr, start_pos, offset, freqs_cos_sin):
        bsz, seqlen, _ = x.size()
        freqs_cos, freqs_sin = freqs_cos_sin
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen
        _NEG_INF = -1e9

        T_c_idx = seqlen // ratio
        k = min(self.index_topk, end_pos // ratio)

        if k >= T_c_idx:
            if start_pos == 0:
                kv_pos = torch.arange(T_c_idx, device=x.device).view(1, 1, -1)
                valid_counts = (torch.arange(1, seqlen + 1, device=x.device) // ratio).view(1, -1, 1)
                mask = torch.where(kv_pos < valid_counts, 0.0, _NEG_INF)
            else:
                mask = torch.zeros(bsz, seqlen, T_c_idx, device=x.device, dtype=torch.float32)
            return None, mask

        split_pos = k * ratio
        split_pos = min(split_pos, seqlen)

        # Sequence-parallel: score only THIS rank's slice of the scored region.
        lo, hi = self.q_range if self.q_range is not None else (0, seqlen)
        s_lo, s_hi = max(lo, split_pos), hi
        S_q = s_hi - s_lo

        # --- First half: precomputed causal mask (queries select all valid kv) ---
        first_mask = self.first_mask_buf

        if S_q == 0:
            return first_mask, None

        # --- Second half: project + RoPE + Hadamard the queries [s_lo:s_hi] ---
        seq_cos_second = freqs_cos[start_pos + s_lo : start_pos + s_hi]
        seq_sin_second = freqs_sin[start_pos + s_lo : start_pos + s_hi]

        qr_second = qr[:, s_lo:s_hi, :]
        q_second = self.wq_b(qr_second)
        q_second = q_second.unflatten(-1, (self.n_heads, self.head_dim))
        q_rope = apply_rotary_emb_functional(q_second[..., -rd:], (seq_cos_second, seq_sin_second))
        q_second = torch.cat([q_second[..., :-rd], q_rope], dim=-1)
        q_second = hadamard_transform(q_second)  # [1, S_q, n_heads, head_dim] bf16

        indexer_kv = compress_sharded(self.compressor, x, start_pos, freqs_cos_sin, self.tp_shard)
        indexer_kv_t = indexer_kv.transpose(1, 2)  # [1, head_dim, T_c_idx]

        # Match the ground-truth dtype: weights_proj (bf16) * scale -> bf16, F.linear in bf16.
        weights_second = F.linear(x[:, s_lo:s_hi, :], (self.weights_proj.weight * self.weight_scale).to(torch.bfloat16))
        # [1, S_q, n_heads] bf16; widened to fp32 for the kernel's per-head accumulate.

        # --- Stack operands for the single fused NKI kernel ---
        # q_T_all[d, h*S_q + s] = q_second[0, s, h, d]: need [head_dim, n_heads, S_q]
        # then flatten the last two axes (h outer, s inner) to match the kernel's
        # q_global = h * S_q + q_start indexing. permute(2, 1, 0) gives head_dim first.
        q_T_all = q_second[0].permute(2, 1, 0).reshape(self.head_dim, self.n_heads * S_q).contiguous()
        kv_t_2d = indexer_kv_t[0].contiguous()  # [head_dim, T_c_idx] bf16
        weights_2d = weights_second[0].float().contiguous()  # [S_q, n_heads] fp32

        # The bias buffers cover rows [split_pos, seqlen); take this rank's rows.
        b0, b1 = s_lo - split_pos, s_hi - split_pos
        cbias = (self.causal_bias_full if start_pos == 0 else self.zero_bias_full)[b0:b1].contiguous()

        # Compute scores and bisection-based selection mask in one kernel
        TILE_Q = 128
        num_q_tiles = S_q // TILE_Q
        n_cores = 2 if (S_q % TILE_Q == 0 and num_q_tiles % 2 == 0) else 1
        if _use_sparse_prefill(kv_t_2d.shape[1]):
            scores_2d = nki_indexer_score_kernel[1](q_T_all, kv_t_2d, weights_2d, cbias)
            topk_idx = nki_prefill_topk_kernel[2](scores_2d.to(torch.bfloat16), int(k), _SAFE_TOPK_N)
            return first_mask, topk_idx.to(torch.int32).unsqueeze(0)

        second_mask_2d = nki_indexer_score_mask_kernel[n_cores](q_T_all, kv_t_2d, weights_2d, cbias, int(k))
        second_mask = second_mask_2d.unsqueeze(0)  # [1, S_q, T_c_idx]

        return first_mask, second_mask


# --------------------------------------------------------------------------
# Core Attention Module — NKI version
# --------------------------------------------------------------------------
class CSAAttentionCoreNKI(nn.Module):
    """Core attention with split window/compressed and NKI kernel integration."""

    def __init__(self, config, use_dense_attn: bool = False, use_nki: bool = True):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.head_dim**-0.5
        self.use_dense_attn = use_dense_attn

        self.attn_sink = nn.Parameter(torch.empty(config.n_heads, dtype=torch.float32))
        self.compressor = CompressorNKI(config, config.head_dim, rotate=False, use_nki=use_nki)
        self.indexer = IndexerNKI(config, use_nki=use_nki)
        # Sequence-parallel prefill: (lo, hi) output rows this rank owns; None = all rows.
        self.q_range = None
        # First global query row present in the `q` tensor handed to forward().
        self.q_row_base = 0
        # (tp_rank, replica_ranks) when the compressed KV is sharded + all-gathered.
        self.tp_shard = None
        self.out_head_major = False

        max_seq_len = config.seq_len
        freqs_cos, freqs_sin = precompute_freqs_cos_sin(
            self.rope_head_dim,
            max_seq_len,
            config.original_seq_len,
            config.compress_rope_theta,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

        win_bias_base, win_bias_sink_ind = precompute_win_bias_parts(max_seq_len, config.window_size)
        self.register_buffer("win_bias_base", win_bias_base, persistent=False)
        self.register_buffer("win_bias_sink_ind", win_bias_sink_ind, persistent=False)

    def forward(self, q, kv, x, qr, start_pos=0):
        bsz, seqlen, _ = x.size()
        win = self.window_size
        ratio = self.compress_ratio
        full_freqs_cs = (self.freqs_cos, self.freqs_sin)

        first_mask, second_mask = self.indexer(x, qr, start_pos, 0, full_freqs_cs)

        kv_compress = compress_sharded(self.compressor, x, start_pos, full_freqs_cs, self.tp_shard)

        T_c_idx = seqlen // ratio
        k = min(self.config.index_topk, seqlen // ratio)

        if first_mask is not None:
            split_pos = min(k * ratio, seqlen)
            T_c_first = split_pos // ratio
            # Sequence-parallel: this rank owns output rows [lo, hi). The dense first
            # half and the scored second half are each intersected with that range;
            # either intersection may be empty.
            lo, hi = self.q_range if self.q_range is not None else (0, seqlen)
            f_lo, f_hi = lo, min(hi, split_pos)
            s_lo, s_hi = max(lo, split_pos), hi
            S_q = s_hi - s_lo

            qb = self.q_row_base
            n_q_local = q.shape[1]
            q_scaled = (q * self.softmax_scale).to(torch.float16)
            q_T = q_scaled.permute(0, 2, 3, 1).reshape(bsz * self.n_heads, self.head_dim, n_q_local)

            kv_f16 = kv.to(torch.float16)
            kv_bf16 = kv.to(torch.bfloat16)
            kv_compress_f16 = kv_compress.to(torch.float16)
            kv_compress_bf16 = kv_compress.to(torch.bfloat16)
            kv_raw_K_T = kv_f16.transpose(1, 2)
            compress_V_2d = kv_compress_bf16.reshape(T_c_idx, self.head_dim)
            compress_K_T_2d = kv_compress_f16.transpose(1, 2).reshape(self.head_dim, T_c_idx)
            attn_sink_2d = self.attn_sink.detach().view(1, self.n_heads).float().contiguous()

            raw_padded_K_T = F.pad(kv_raw_K_T, (win, 0)).reshape(self.head_dim, seqlen + win)
            raw_padded_V = F.pad(kv_bf16, (0, 0, win, 0)).reshape(seqlen + win, self.head_dim)

            # First half: mask-based kernel (unchanged)
            parts = []
            if f_hi > f_lo:
                n_first = f_hi - f_lo
                first_mask_2d = first_mask.reshape(split_pos, T_c_first)[f_lo:f_hi].contiguous()
                first_kt = torch.cat([raw_padded_K_T[:, f_lo : f_hi + win], compress_K_T_2d[:, :T_c_first]], dim=1)
                first_v = torch.cat([raw_padded_V[f_lo : f_hi + win, :], compress_V_2d[:T_c_first, :]], dim=0)
                all_q_T_first = (
                    q_T[:, :, f_lo - qb : f_hi - qb].permute(1, 0, 2).reshape(self.head_dim, self.n_heads * n_first)
                )
                num_q_tiles_first = n_first // 128
                n_cores_first = 2 if (num_q_tiles_first % 2 == 0 and num_q_tiles_first >= 2) else 1
                out_first_flat = nki_fused_csa_attn_kernel[n_cores_first](
                    first_mask_2d,
                    all_q_T_first,
                    first_kt,
                    first_v,
                    self.win_bias_base[f_lo:f_hi],
                    self.win_bias_sink_ind[f_lo:f_hi],
                    attn_sink_2d,
                )
                parts.append(out_first_flat.reshape(self.n_heads, n_first, self.head_dim))

            # Second half: static causal-bound sparse attention (global-max softmax
            # + sel_bias predication, per-tile compile-time causal chunk bound).
            second_win_K_T = raw_padded_K_T[:, s_lo : s_lo + S_q + win]
            second_win_V = raw_padded_V[s_lo : s_lo + S_q + win, :]

            if S_q > 0:
                out_second_all = prefill_second_half_attention(
                    second_mask.reshape(S_q, -1),
                    q_scaled[0, s_lo - qb : s_hi - qb],
                    second_win_K_T,
                    second_win_V,
                    compress_K_T_2d,
                    compress_V_2d,
                    self.win_bias_base[s_lo:s_hi],
                    self.win_bias_sink_ind[s_lo:s_hi],
                    attn_sink_2d,
                    int(s_lo),
                    int(ratio),
                    sparse=_use_sparse_prefill(T_c_idx),
                )
                parts.append(out_second_all)

            out_all = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
            if self.out_head_major:
                return out_all
            o = out_all.permute(1, 0, 2).unsqueeze(0)
        else:
            o = nki_fused_csa_attn(
                q,
                kv,
                kv_compress,
                self.attn_sink,
                win,
                second_mask,
                self.softmax_scale,
                self.win_bias_base,
                self.win_bias_sink_ind,
            )
            if self.out_head_major:
                # Dense fallback (k >= T_c, i.e. short contexts only): reshape to the
                # head-major contract, which costs a copy this path is small enough to pay.
                o = o[0].permute(1, 0, 2)

        return o


class CSAAttentionXLA(nn.Module):
    def __init__(self, config: CSAConfig, replica_ranks=None):
        super().__init__()
        self.config = config
        self.replica_ranks = list(replica_ranks) if replica_ranks is not None else None
        # Sequence-parallel prefill: (lo, hi) output rows this rank owns, or None for
        # the head-parallel path where every rank produces all rows.
        self._q_range = None
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.n_local_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.n_groups = config.o_groups
        self.n_local_groups = config.o_groups
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.eps = config.norm_eps

        pdt = torch.bfloat16

        # Query path
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=pdt)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=pdt)

        # KV path
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False, dtype=pdt)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # Output path (grouped low-rank)
        self.group_in = self.n_heads * self.head_dim // self.n_groups
        self.wo_a = nn.Linear(self.group_in, self.n_groups * self.o_lora_rank, bias=False, dtype=pdt)
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False, dtype=pdt)

        self.core = CSAAttentionCoreNKI(config, use_dense_attn=False, use_nki=True)
        # Take the core's native head-major output; the de-RoPE kernel re-lays it out.
        self.core.out_head_major = True

        # Precompute RoPE frequencies as real cos/sin (no complex on NeuronX)
        freqs_cos, freqs_sin = precompute_freqs_cos_sin(
            self.rope_head_dim,
            config.seq_len,
            config.original_seq_len,
            config.compress_rope_theta,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def set_q_range(self, q_range) -> None:
        """Sequence-parallel prefill: restrict this rank to output rows ``[lo, hi)``.

        Pushed to the attention core and the indexer, which is where the query range
        actually reduces work; ``None`` restores the head-parallel behaviour of every
        rank producing every row.
        """
        self._q_range = q_range
        self.core.q_range = q_range
        self.core.q_row_base = 0 if q_range is None else q_range[0]
        self.core.indexer.q_range = q_range

    def set_tp_shard(self, tp_shard) -> None:
        """Shard the COMPRESSED KV across ranks and all-gather it.

        ``tp_shard`` is ``(tp_rank, replica_ranks)``, or None to compute the whole
        compressed cache locally on every rank. Pushed to the attention core and the
        indexer, which own the two compressor call sites. This is orthogonal to
        ``set_q_range``: that shards which QUERIES a rank produces, this shards which
        compressed KV positions it COMPUTES -- the KV itself ends up complete on every
        rank either way, because the top-k can select any position.
        """
        self.core.tp_shard = tp_shard
        self.core.indexer.tp_shard = tp_shard

    def forward(self, x: torch.Tensor, start_pos: int = 0):
        """
        Args:
            x: [B, S, dim]
            start_pos: 0 for prefill
        Returns:
            output: [B, S, dim]
        """
        bsz, seqlen, _ = x.size()
        # Slice cos/sin for current positions
        seq_cos = self.freqs_cos[start_pos : start_pos + seqlen]
        seq_sin = self.freqs_sin[start_pos : start_pos + seqlen]
        H, D = self.n_local_heads, self.head_dim
        x_bf = x.to(torch.bfloat16)

        q_lo, q_hi = self._q_range if self._q_range is not None else (0, seqlen)
        n_q = q_hi - q_lo

        cos_qs = seq_cos[q_lo:q_hi].float().contiguous()
        sin_qs = seq_sin[q_lo:q_hi].float().contiguous()
        cos_s = seq_cos.float().contiguous()
        sin_s = seq_sin.float().contiguous()

        # ===== Query Path =====
        qr = self.q_norm(self.wq_a(x_bf))  # [B, S, q_lora_rank]
        qr_q = qr if self._q_range is None else qr[:, q_lo:q_hi, :]
        q = self.wq_b(qr_q)  # [B, n_q, H*D]

        q_out = nki_rms_rope_kernel(
            q.reshape(bsz * n_q, H * D)[0:n_q].to(torch.bfloat16),
            cos_qs,
            sin_qs,
            None,
            self.eps,
            do_rms=1,
            inverse=0,
            heads=H,
            in_head_major=0,
            out_head_major=0,
        )
        q = q_out.reshape(bsz, n_q, H, D)

        # ===== KV Path =====
        # Learnable-gain RMSNorm + RoPE, same kernel with gain_in = kv_norm.weight.
        kv_lin = self.wkv(x_bf)  # [B, S, D]
        kv_out = nki_rms_rope_kernel(
            kv_lin.reshape(seqlen, D).to(torch.bfloat16),
            cos_s,
            sin_s,
            self.kv_norm.weight.reshape(1, D).float().contiguous(),
            self.eps,
            do_rms=1,
            inverse=0,
        )
        kv = kv_out.reshape(bsz, seqlen, D)

        # ===== NKI Attention Core =====
        # Replaces window+compressed index computation, KV compression and
        # sparse_attn_xla. Same operands and same [B, S, n_heads, head_dim] out.
        o = self.core(q, kv, x_bf, qr, start_pos=start_pos)

        # Sequence-parallel: the core returned only THIS rank's rows, so everything
        # downstream (de-RoPE positions, output projection) runs on that row count and
        # at that position offset, not on the full sequence.
        o_lo, o_hi = self._q_range if self._q_range is not None else (0, seqlen)
        n_out = o_hi - o_lo
        # Same rows as the q path (both ranges are self._q_range), so the same table.
        cos_o, sin_o = cos_qs, sin_qs

        # ===== Output de-RoPE =====
        # Rotation only (do_rms=0) with inverse=1, same fused kernel. The attention core
        # hands back HEAD-MAJOR [H, n_out, D] and the output projection wants QUERY-MAJOR
        # [n_out, H*D], so the kernel reads one layout and writes the other: the transpose
        # rides along in the DMA it was already issuing. The call site used to permute to
        # query-major, back to head-major for this kernel, and to query-major again.
        o_out = nki_rms_rope_kernel(
            o.reshape(H * n_out, D).to(torch.bfloat16),
            cos_o,
            sin_o,
            None,
            self.eps,
            do_rms=0,
            inverse=1,
            heads=H,
            in_head_major=1,
            out_head_major=0,
        )
        o = o_out.reshape(bsz, n_out, H * D)

        # ===== Output Projection (grouped low-rank) =====
        # Two orderings. Fusing composes wo_a into wo_b and then needs ONE matmul;
        # unfused projects to o_lora_rank first and then out.
        G, R, Din = self.n_local_groups, self.o_lora_rank, self.group_in
        o = o.reshape(bsz, n_out, G, Din)
        rows = bsz * n_out
        fused_macs = self.dim * G * R * Din + rows * G * Din * self.dim
        unfused_macs = rows * G * Din * R + rows * G * R * self.dim
        
        _FUSED_WEIGHT_BUDGET = 1 << 27  # 1.34e8 elements
        if fused_macs <= unfused_macs and self.dim * G * Din <= _FUSED_WEIGHT_BUDGET:
            wo_a = self.wo_a.weight.view(G, R, Din)
            wo_b = self.wo_b.weight.view(self.dim, G, R)
            wfused = torch.einsum("cgr,grd->cgd", wo_b, wo_a).reshape(self.dim, G * Din)
            output = torch.matmul(o.reshape(bsz, n_out, G * Din), wfused.t())
        else:
            wo_a = self.wo_a.weight.view(G, R, Din)
            lat = torch.einsum("bsgd,grd->bsgr", o, wo_a)  # [B, S, G, o_lora]
            output = self.wo_b(lat.reshape(bsz, n_out, G * R))

        # ===== Cross-rank all-reduce (merged): RowParallelLinear sum over ranks =====
        # When replica_ranks is set (multi-worker torchrun), append the 2-LNC
        # ncc.all_reduce(op=add) as the block's FINAL op so the traced block is one
        # integrated lnc=2 NEFF returning the full [B,S,dim]. Otherwise return the
        # rank-local partial and let the caller host-sum the ranks.
        if self.replica_ranks is not None:
            return tp_all_reduce(output, self.replica_ranks)
        return output


# --------------------------------------------------------------------------
# Decode Indexer — uses bisection mask for large T_c support
# --------------------------------------------------------------------------
class DecodeIndexerGatheredNKI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = self.head_dim**-0.5

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False, dtype=torch.bfloat16)
        self.weight_scale = self.softmax_scale * (self.n_heads**-0.5)
        self.compressor = CompressorNKI(config, self.head_dim, rotate=True, use_nki=True)

    def _score_inputs(self, x, qr, start_pos, indexer_kv_cache, freqs_cos_sin):
        """Host-side prep shared by `forward` and `fused_single_chunk_inputs`.

        Factored out (with its op sequence VERBATIM) so the fused
        score+topk+attention path consumes bit-identical scoring inputs to the
        two-launch path it replaces, rather than a re-derivation of them.

        Returns (q_T_all, weights_2d, indexer_kv_t, T_c, k, S_q).
        """
        freqs_cos, freqs_sin = freqs_cos_sin
        rd = self.rope_head_dim
        T_c = indexer_kv_cache.shape[1]
        k = min(self.index_topk, T_c)
        TILE_Q = 128
        S_q = TILE_Q

        seq_cos = freqs_cos[start_pos : start_pos + 1]
        seq_sin = freqs_sin[start_pos : start_pos + 1]

        wT = self.wq_b.weight.t().contiguous().reshape(self.q_lora_rank // 128, 128, self.n_heads * self.head_dim)
        qr_2d = qr.reshape(1, self.q_lora_rank)
        qT = nki_indexer_qproj_gemv[2](wT, qr_2d)  # [head_dim, n_heads] bf16
        q = qT.t().contiguous().reshape(1, 1, self.n_heads, self.head_dim)
        q_rope = apply_rotary_emb_functional(q[..., -rd:], (seq_cos, seq_sin))
        q = torch.cat([q[..., :-rd], q_rope], dim=-1)
        q = hadamard_transform(q)

        indexer_kv_t = indexer_kv_cache.transpose(1, 2)  # [1, head_dim, T_c]

        weights = F.linear(x, (self.weights_proj.weight * self.weight_scale).to(torch.bfloat16))

        q_single = q[0, 0]
        q_T_all = (
            q_single.permute(1, 0)
            .unsqueeze(2)
            .expand(self.head_dim, self.n_heads, S_q)
            .reshape(self.head_dim, self.n_heads * S_q)
            .contiguous()
        )
        weights_2d = weights[0, 0:1].float().expand(S_q, -1).contiguous()

        return q_T_all, weights_2d, indexer_kv_t, T_c, k, S_q

    IDX_CHUNK = 8192
    SCORE_CHUNK = 512
    SAFE_TOPK_N = 8192

    def fused_single_chunk_inputs(self, x, qr, start_pos, indexer_kv_cache, freqs_cos_sin, gather_chunk):
        """Scoring inputs for the FUSED score+topk+attention launch, or None.

        Returns (q_T_all, kv_t_seg, weights_2d, k, SAFE_TOPK_N) when this decode
        step qualifies for `nki_indexer_score_topk_gather_2core[2]` — i.e. when it
        already qualified for the merged `nki_indexer_score_topk_2core[2]` scoring
        path AND the top-k width divides the attention gather's chunk. Returns None
        otherwise, in which case the caller falls back to the unchanged
        `forward()` -> `nki_decode_gather_ok_kernel[1]` two-launch pipeline.

        The gate clauses are exactly `forward`'s, so no seq-len that used to take
        the merged scoring path can silently drop to a slower one; the only added
        clause is `k % gather_chunk == 0`, which the attention kernel's
        num_k_chunks = k // COMP_CHUNK tiling already required of every config it
        ran on (k=1024, COMP_CHUNK=128).
        """
        T_c = indexer_kv_cache.shape[1]
        k = min(self.index_topk, T_c)
        num_idx_chunks = (T_c + self.IDX_CHUNK - 1) // self.IDX_CHUNK
        if not (num_idx_chunks == 1 and T_c % 128 == 0 and k % 16 == 0):
            return None
        if not (T_c % 2 == 0 and (T_c // 2) % self.SCORE_CHUNK == 0 and T_c <= self.SAFE_TOPK_N):
            return None
        if k % gather_chunk != 0:
            return None

        q_T_all, weights_2d, indexer_kv_t, T_c, k, _ = self._score_inputs(
            x, qr, start_pos, indexer_kv_cache, freqs_cos_sin
        )
        kv_t_seg = indexer_kv_t[0, :, 0:T_c].contiguous()  # [head_dim, T_c]
        return q_T_all, kv_t_seg, weights_2d, k, self.SAFE_TOPK_N

    def forward(self, x, qr, start_pos, indexer_kv_cache, freqs_cos_sin):
        """Score all T_c in chunks → bisection per chunk → merge top-k indices.

        For large T_c (e.g. 16384), the full score array doesn't fit in SBUF.
        Strategy: split indexer_kv_cache into segments of IDX_CHUNK, score each
        with nki_indexer_score_kernel (which writes scores to HBM), concatenate,
        then use nkilib topk or bisection on the concatenated scores.

        Still the entry point for the MULTI-chunk path (and for any single-chunk
        config the fused kernel's gate rejects); the single-chunk decode path now
        goes through `fused_single_chunk_inputs` +
        `nki_indexer_score_topk_gather_2core[2]`, which folds this scoring, its
        top-k, AND the attention body into one launch.
        """
        q_T_all, weights_2d, indexer_kv_t, T_c, k, S_q = self._score_inputs(
            x, qr, start_pos, indexer_kv_cache, freqs_cos_sin
        )

        IDX_CHUNK = 8192
        num_idx_chunks = (T_c + IDX_CHUNK - 1) // IDX_CHUNK

        if num_idx_chunks == 1 and T_c % 128 == 0 and k % 16 == 0:
            kv_t_seg = indexer_kv_t[0, :, 0:T_c].contiguous()  # [head_dim, T_c]
            SCORE_CHUNK = 512
            SAFE_TOPK_N = 8192
            if T_c % 2 == 0 and (T_c // 2) % SCORE_CHUNK == 0 and T_c <= SAFE_TOPK_N:
                topk_idx_hbm = nki_indexer_score_topk_2core[2](q_T_all, kv_t_seg, weights_2d, int(k), SAFE_TOPK_N)
            else:
                topk_idx_hbm = nki_indexer_score_topk_kernel[1](
                    q_T_all, kv_t_seg, weights_2d, int(k)
                )  # [TOPK_ROWS, k] uint32
            topk_head = topk_idx_hbm.int()
            return topk_head[0:1].contiguous()

        SCORE_CHUNK_2C = 512
        TOPK_ROWS = 8
        tail_len = T_c - (num_idx_chunks - 1) * IDX_CHUNK  # IDX_CHUNK if T_c % IDX_CHUNK == 0
        use_2core_score = T_c % 128 == 0 and (T_c // 2) % SCORE_CHUNK_2C == 0 and k % 16 == 0 and tail_len >= k
        if use_2core_score:
            kv_t_full = indexer_kv_t[0, :, 0:T_c].contiguous()  # [head_dim, T_c]
            scores_full = nki_indexer_score_2core[2](q_T_all, kv_t_full, weights_2d)  # [1, T_c] bf16

            chunk_rows = []
            for c in range(num_idx_chunks):
                seg_start = c * IDX_CHUNK
                seg_end = min(seg_start + IDX_CHUNK, T_c)
                row = scores_full[0:1, seg_start:seg_end]  # [1, seg_len]
                seg_len = seg_end - seg_start
                if seg_len < IDX_CHUNK:
                    row = F.pad(row, (0, IDX_CHUNK - seg_len), value=-1e9)
                chunk_rows.append(row)
            scores_chunks = torch.cat(chunk_rows, dim=0)  # [num_idx_chunks, IDX_CHUNK]
            # Pad rows up to TOPK_ROWS=8 (nisa.topk needs rows % 8 == 0); only the
            # first num_idx_chunks rows are consumed (pad with a copy of row 0).
            if num_idx_chunks < TOPK_ROWS:
                pad = scores_chunks[0:1].expand(TOPK_ROWS - num_idx_chunks, IDX_CHUNK)
                scores_batched = torch.cat([scores_chunks, pad], dim=0).contiguous()
            else:
                scores_batched = scores_chunks.contiguous()
            # Batched Pass-1: ONE launch, each row/group gets its own local top-k.
            k_padded = ((k + 15) // 16) * 16
            pv, pl = nisa_topk_batched(scores_batched, k=int(k_padded))  # [8, k_padded]
            # Merge candidates from the num_idx_chunks chunks (indices -> global).
            merged_scores = torch.cat([pv[s : s + 1, :k] for s in range(num_idx_chunks)], dim=1)  # [1, C*k]
            merged_indices = torch.cat([pl[s : s + 1, :k] + s * IDX_CHUNK for s in range(num_idx_chunks)], dim=1)
            # Pass-2: final top-k over the merged candidates (n = C*k, proven safe).
            merge_n = merged_scores.shape[1]
            pad_merge = (16 - merge_n % 16) % 16
            if pad_merge > 0:
                merged_scores = F.pad(merged_scores, (0, pad_merge), value=-1e9)
                merged_indices = F.pad(merged_indices, (0, pad_merge), value=0)
            merged_scores_b = merged_scores.expand(TOPK_ROWS, -1).contiguous()
            _, ml = nisa_topk_batched(merged_scores_b, k=int(k_padded))
            ml = ml[0:1, :k]  # row 0 only (all rows identical in decode)
            topk_head = torch.gather(merged_indices, dim=1, index=ml.long()).int()  # [1, k]
            # S_out=1 (see single-chunk path): kernel reads only column 0.
            return topk_head[0:1].contiguous()

        # ---- Fallback (old proven path): per-chunk single-core fp32 scoring ------
        score_chunks = []
        for seg in range(num_idx_chunks):
            seg_start = seg * IDX_CHUNK
            seg_end = min(seg_start + IDX_CHUNK, T_c)
            seg_len = seg_end - seg_start

            kv_t_seg = indexer_kv_t[0, :, seg_start:seg_end].contiguous()  # [head_dim, seg_len]
            zero_bias_seg = torch.zeros_like(kv_t_seg[0:1]).float().expand(S_q, -1).contiguous()

            # S_q = TILE_Q = 128 → exactly one query tile, so launch on 1 core
            # (the kernel does num_q_tiles // n_cores tiles per core; with 2 cores
            # that would be 1 // 2 = 0 and produce no scores).
            scores_seg = nki_indexer_score_kernel[1](q_T_all, kv_t_seg, weights_2d, zero_bias_seg)
            score_chunks.append(scores_seg)  # [S_q, seg_len]

        TOPK_ROWS = 8

        # Pass 1: top-k per chunk, collecting (scores, global_indices)
        candidate_scores = []
        candidate_indices = []
        for seg_idx, scores_seg in enumerate(score_chunks):
            seg_start = seg_idx * IDX_CHUNK
            seg_len = scores_seg.shape[1]
            seg_k = min(k, seg_len)
            scores_head = scores_seg[0:TOPK_ROWS]  # identical rows → only need 8
            if seg_len < IDX_CHUNK:
                scores_padded = F.pad(scores_head, (0, IDX_CHUNK - seg_len), value=-1e9)
            else:
                pad_n = (16 - seg_len % 16) % 16
                scores_padded = F.pad(scores_head, (0, pad_n), value=-1e9) if pad_n > 0 else scores_head
            # Pad seg_k to multiple of 16 if needed
            seg_k_padded = ((seg_k + 15) // 16) * 16
            top_vals, top_local_idx = nisa_topk_batched(scores_padded, k=int(seg_k_padded))
            # Trim to actual seg_k
            top_vals = top_vals[:, :seg_k]
            top_local_idx = top_local_idx[:, :seg_k]
            # Convert local indices to global: add segment offset
            top_global_idx = top_local_idx + seg_start
            candidate_scores.append(top_vals)  # [TOPK_ROWS, seg_k]
            candidate_indices.append(top_global_idx)  # [TOPK_ROWS, seg_k]

        if num_idx_chunks == 1:
            topk_head = candidate_indices[0][:, :k].int()
        else:
            # Pass 2: merge all candidates and take final top-k
            merged_scores = torch.cat(candidate_scores, dim=1)  # [TOPK_ROWS, num_chunks * k]
            merged_indices = torch.cat(candidate_indices, dim=1)  # [TOPK_ROWS, num_chunks * k]

            # nisa.topk on merged set: [TOPK_ROWS, num_chunks*k] → [TOPK_ROWS, k]
            merge_n = merged_scores.shape[1]
            pad_merge = (16 - merge_n % 16) % 16
            if pad_merge > 0:
                merged_scores_padded = F.pad(merged_scores, (0, pad_merge), value=-1e9)
            else:
                merged_scores_padded = merged_scores
            k_padded = ((k + 15) // 16) * 16
            _, merge_local_idx = nisa_topk_batched(merged_scores_padded, k=int(k_padded))
            merge_local_idx = merge_local_idx[:, :k]
            # merge_local_idx[s, i] indexes into merged_indices[s, :] → use torch.gather
            topk_head = torch.gather(merged_indices, dim=1, index=merge_local_idx.long()).int()

        # Return a single index row [1, k]: the attention kernel batches heads on
        # partitions and reads only column 0 of topk_indices_T (its 2-core split is
        # by-HEAD, S-independent), so the S=256 broadcast was pure dead weight.
        return topk_head[0:1].contiguous()


# --------------------------------------------------------------------------
# Decode Attention Module — O(k) with on-device gathered K+V
# --------------------------------------------------------------------------
class CSADecodeAttentionGatheredNKI(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.head_dim**-0.5

        self.attn_sink = nn.Parameter(torch.empty(config.n_heads, dtype=torch.float32))
        self.compressor = CompressorNKI(config, config.head_dim, rotate=False, use_nki=True)
        self.indexer = DecodeIndexerGatheredNKI(config)

        max_seq_len = config.seq_len
        freqs_cos, freqs_sin = precompute_freqs_cos_sin(
            self.rope_head_dim,
            max_seq_len + 1,
            config.original_seq_len,
            config.compress_rope_theta,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    def forward(self, q, kv_window, kv_compress, x, qr, indexer_kv_cache):
        """O(k) decode: gathered K scoring + gathered V matmul, all on device.

        Attention kernel cost: O(W + k) — independent of T_c.
        Indexer cost: O(T_c) — unavoidable (must score all to find top-k).
        """
        W = self.window_size
        T_c = kv_compress.shape[1]
        S = 1
        start_pos = self.config.seq_len
        full_freqs_cs = (self.freqs_cos, self.freqs_sin)

        # ---- FUSED indexer-score + top-k + attention (single launch) ----------
        # `fused_single_chunk_inputs` returns the indexer's scoring inputs when this
        # step qualifies for the fused [2]-grid kernel (both graded seq-lens do), or
        # None to fall back to the unchanged two-launch pipeline. Asking for it here,
        # BEFORE the attention-side host prep, keeps the indexer's own op sequence in
        # the same relative position in the traced graph as the `self.indexer(...)`
        # call it replaces.
        COMP_CHUNK = 128  # the attention kernel's gather chunk (k must divide it)
        fused_inputs = self.indexer.fused_single_chunk_inputs(
            x, qr, start_pos, indexer_kv_cache, full_freqs_cs, COMP_CHUNK
        )

        # --- Indexer (fallback only): score all T_c → top-k indices [S, k] ---
        if fused_inputs is None:
            topk_indices = self.indexer(x, qr, start_pos, indexer_kv_cache, full_freqs_cs)

        # --- Prepare Q: replicate to S ---
        q_scaled = (q * self.softmax_scale).to(torch.float16)
        q_single = q_scaled[0, 0]
        all_q_T = (
            q_single.permute(1, 0)
            .unsqueeze(2)
            .expand(self.head_dim, self.n_heads, S)
            .reshape(self.head_dim, self.n_heads * S)
            .contiguous()
        )

        WIN_SIZE = W
        kv_win_f16 = kv_window.to(torch.float16)
        all_K_T_win = kv_win_f16.transpose(1, 2).reshape(self.head_dim, WIN_SIZE)
        all_V_win = kv_win_f16.reshape(WIN_SIZE, self.head_dim)

        compress_kv = kv_compress.reshape(T_c, self.head_dim).contiguous()

        attn_sink_2d = self.attn_sink.detach().view(1, self.n_heads).float().contiguous()

        derope_cos = self.freqs_cos[start_pos : start_pos + 1].contiguous()  # [1, half_rope]
        derope_sin = self.freqs_sin[start_pos : start_pos + 1].contiguous()  # [1, half_rope]
        if fused_inputs is not None:
            idx_q_T_all, idx_kv_t_seg, idx_weights_2d, fused_k, fused_n = fused_inputs
            out_flat = nki_indexer_score_topk_gather_2core[2](
                idx_q_T_all,
                idx_kv_t_seg,
                idx_weights_2d,
                int(fused_k),
                int(fused_n),
                all_q_T,
                all_K_T_win,
                all_V_win,
                compress_kv,
                attn_sink_2d,
                derope_cos,
                derope_sin,
            )
        else:
            topk_indices_T = topk_indices.t().contiguous()  # [k, S]
            out_flat = nki_decode_gather_ok_kernel[1](
                topk_indices_T, all_q_T, all_K_T_win, all_V_win, compress_kv, attn_sink_2d, derope_cos, derope_sin
            )

        # --- Extract ---
        out_all = out_flat.reshape(self.n_heads, S, self.head_dim)
        o = out_all[:, 0, :].unsqueeze(0).unsqueeze(1)
        return o


class CSADecodeAttentionBlockNKI(nn.Module):
    """Complete CSA attention block (compress_ratio=4), decode phase, on NKI.

    Owns the projection weights and a `CSADecodeAttentionGatheredNKI` core
    (which itself owns attn_sink, the indexer, and the compressor). The output
    projection is sharded for `tp_size`-way tensor parallelism; this module
    holds and computes ONLY rank `tp_rank`'s shard.
    """

    def __init__(self, config, tp_size: int = 4, tp_rank: int = 0, replica_ranks=None):
        super().__init__()
        self.config = config
        # None -> return the rank-local partial (host sums the partials).
        # list -> append a 2-LNC ncc.all_reduce(op=add) over these ranks as the
        # final forward op, so the traced block returns the full output.
        self.replica_ranks = list(replica_ranks) if replica_ranks is not None else None
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.n_groups = config.o_groups
        self.window_size = config.window_size
        self.eps = config.norm_eps
        self.softmax_scale = config.head_dim**-0.5

        if self.n_groups % tp_size != 0:
            raise ValueError(f"o_groups={self.n_groups} must be divisible by tp_size={tp_size}")
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.n_local_groups = self.n_groups // tp_size  # groups this rank owns
        self.group_in = self.n_heads * self.head_dim // self.n_groups  # per-group wo_a input width

        pdt = torch.bfloat16

        # ----- Query projection -----
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=pdt)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=pdt)

        # ----- KV projection (for the new decode token) -----
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False, dtype=pdt)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # ----- Output projection (grouped low-rank), rank-local shard -----
        self.wo_a = nn.Linear(self.group_in, self.n_local_groups * self.o_lora_rank, bias=False, dtype=pdt)
        self.wo_b = nn.Linear(self.n_local_groups * self.o_lora_rank, self.dim, bias=False, dtype=pdt)

        # ----- Library core (gathered O(k) decode attention) -----
        self.core = CSADecodeAttentionGatheredNKI(config)

        # RoPE tables for the block's q/kv rotation (start_pos == seq_len needs
        # index seq_len, so provision seq_len + 1 entries).
        freqs_cos, freqs_sin = precompute_freqs_cos_sin(
            self.rope_head_dim,
            config.seq_len + 1,
            config.original_seq_len,
            config.compress_rope_theta,
            config.rope_factor,
            config.beta_fast,
            config.beta_slow,
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

    # ---- projection helpers -------------------------------------------------
    def _project_qkv(self, x, freqs_cs):
        """Fused q-path AND kv-path RMS+RoPE via ONE merged NKI kernel.

        Both projection tails do the identical per-partition op (RMS over the free
        axis + RoPE on the last rope_head_dim channels). Packing the n_heads q rows
        and the single kv row onto one [n_heads+1, head_dim] partition tile lets a
        single @nki.jit launch replace the two separate kernels — dropping a launch
        boundary, a shared_hbm alloc, and an HBM round-trip. The q/kv path
        difference (kv has a learnable gain, q does not) is a per-partition gain
        tile: 1.0 on the q rows (x*1.0==x exact in fp32), kv_norm.weight on the kv
        row. Bit-identical to the two-kernel path.
        """
        # q-path pre-norm latent (shared with the indexer via qr).
        qr = self.q_norm(self.wq_a(x))
        q_lin = self.wq_b(qr)  # [1,1,n_heads*head_dim] bf16
        q_2d = q_lin.reshape(self.n_heads, self.head_dim).contiguous()  # [n_heads, head_dim] bf16
        # kv-path pre-norm latent (the single decode-token KV).
        kv_lin = self.wkv(x)  # [1,1,head_dim] bf16
        kv_2d = kv_lin.reshape(1, self.head_dim).contiguous()  # [1, head_dim] bf16
        weight = self.kv_norm.weight.reshape(1, self.head_dim)  # [1, head_dim] fp32 gain

        cos, sin = freqs_cs
        # ONE launch assembles the [n_heads+1, head_dim] tile on-chip and fuses
        # RMS(+gain)+RoPE for both paths (q rows + kv row).
        out = nki_qkv_rms_rope_kernel(q_2d, kv_2d, weight, cos, sin, self.eps)

        q = out[: self.n_heads].reshape(1, 1, self.n_heads, self.head_dim)
        kv = out[self.n_heads : self.n_heads + 1].reshape(1, 1, self.head_dim)
        return q, qr, kv

    def _output_projection(self, o, bsz, seqlen):
        """Rank-local grouped low-rank output projection.

        o: [B, S, n_heads, head_dim] (post de-RoPE). Consumes only the
        n_local_groups groups this rank owns.

        Decode is S=1, so both wo_a and wo_b are memory-bound GEMVs — latency is
        dominated by the bf16 weight bytes streamed from HBM. There are two ways to
        run the grouped low-rank projection, and which is cheaper depends on the
        relation between group_in and o_lora_rank:

          two-step (wo_a then wo_b):  reads  n_local_groups*o_lora*group_in
                                            + dim*n_local_groups*o_lora  bytes
          fused (compose wo_b@wo_a):  reads  dim*n_local_groups*group_in  bytes

        The fused single-matmul weight is [dim, n_local_groups*group_in]; composing
        wo_a into wo_b EXPANDS the projected width from o_lora_rank back up to
        group_in, so fusing only wins when group_in <= o_lora_rank (the reduced
        test model: group_in=1024=o_lora_rank). The PRODUCTION shard
        (original_model.py world_size=4) has group_in = n_heads*head_dim/n_groups =
        128*512/16 = 4096 > o_lora_rank=1024, where fusing would stream
        dim*4*4096 = 234MB vs the two-step's 92MB — a 2.55x HBM blow-up that stalls
        the PE. So keep the low-rank o_lora bottleneck: apply wo_a (compress
        group_in->o_lora per group) THEN wo_b, exactly as original_model.py's
        einsum + RowParallelLinear. We pick whichever reads fewer weight bytes so
        the reduced-model fusion win is preserved and the full model takes the
        cheap two-step path.
        """
        o = o.reshape(bsz, seqlen, self.n_groups, self.group_in)
        g0 = self.tp_rank * self.n_local_groups
        o_local = o[:, :, g0 : g0 + self.n_local_groups, :].contiguous()
        G, R, D = self.n_local_groups, self.o_lora_rank, self.group_in

        fused_bytes = self.dim * G * D
        twostep_bytes = G * R * D + self.dim * G * R
        if fused_bytes <= twostep_bytes:
            # Fused single matmul (constant-folded weight compose). Cheaper only
            # when group_in <= o_lora_rank (reduced model).
            wo_a = self.wo_a.weight.view(G, R, D)
            wo_b = self.wo_b.weight.view(self.dim, G, R)
            wfused = torch.einsum("cgr,grd->cgd", wo_b, wo_a).reshape(self.dim, G * D)
            out_partial = torch.matmul(o_local.reshape(bsz, seqlen, G * D).to(torch.bfloat16), wfused.t())
        else:
            wo_a = self.wo_a.weight.view(G, R, D)
            lat = torch.einsum("bsgd,grd->bsgr", o_local.to(torch.bfloat16), wo_a)  # [B,S,G,o_lora]
            out_partial = self.wo_b(lat.reshape(bsz, seqlen, G * R))  # [B,S,dim]

        return out_partial

    # ---- decode forward -----------------------------------------------------
    def forward(self, x, kv_window, kv_compress, indexer_kv_cache):
        """Single-token decode over the whole attention block.

        Args:
            x:                [B, 1, dim]           raw hidden state for the new token
            kv_window:        [B, W, head_dim]      window KV cache (post-prefill, pre-decode)
            kv_compress:      [B, T_c, head_dim]    compressed KV cache
            indexer_kv_cache: [B, T_c, index_head_dim]
        Returns:
            [B, 1, dim] — the full all-reduced output when `replica_ranks` was
            given at construction, else rank `tp_rank`'s partial (host-sum path).
        """
        bsz, seqlen, _ = x.size()
        if seqlen != 1:
            raise ValueError(f"Decode expects seqlen=1, got {seqlen}")
        W = self.window_size
        start_pos = self.config.seq_len

        seq_cos = self.freqs_cos[start_pos : start_pos + 1]
        seq_sin = self.freqs_sin[start_pos : start_pos + 1]
        freqs_cs = (seq_cos, seq_sin)

        # ----- Projections (q latent shared with the indexer via qr) -----
        # ONE merged NKI kernel fuses BOTH the q-path per-head RMS+RoPE and the
        # kv-path learnable RMSNorm+RoPE (see _project_qkv).
        q, qr, kv = self._project_qkv(x, freqs_cs)  # q:[B,1,n_heads,head_dim] kv:[B,1,head_dim]

        # ----- Insert the new token into the window cache at (start_pos % W) -----
        p = start_pos % W
        kv_slot = kv.reshape(bsz, 1, self.head_dim).to(kv_window.dtype)
        kv_window = torch.cat([kv_window[:, :p], kv_slot, kv_window[:, p + 1 :]], dim=1)

        # ----- Core sparse attention (gathered O(k)) -----
        o = self.core(q, kv_window, kv_compress, x, qr, indexer_kv_cache)

        # ----- Rank-local output projection -----
        partial = self._output_projection(o, bsz, seqlen)  # [B,1,dim] rank partial

        # ----- Cross-rank all-reduce (merged): RowParallelLinear sum over ranks -----
        if self.replica_ranks is not None:
            return tp_all_reduce(partial, self.replica_ranks)
        return partial


# ------------------------------------------------------------------------
# Runnable driver: trace one rank's block and grade it against the CPU golden
# ------------------------------------------------------------------------
# The blocks above cannot be graded by the kernel integration tests (they mix
# torch projections with NKI launches, and the multi-worker path spans ranks), so
# this is their end-to-end check. Two launch modes:
#
#   sequential   one process traces each rank in turn on one core-set and sums the
#                partials on the host. `replica_ranks=None`, so no collective is
#                traced -- this validates the block compute alone.
#   distributed  `torchrun --nproc_per_node=<tp_size>`; every rank traces its own
#                block with the 2-LNC ncc.all_reduce MERGED in as the final op, so
#                each rank's single NEFF returns the FULL all-reduced output. This
#                is the real configuration, and the only one that exercises the
#                collective.
#
# Both grade against the same 128-head CPU golden, so a head-parallel sharding
# mistake shows up as a correctness failure rather than as a plausible number.
_ATOL = 2e-3

_WEIGHT_GAIN = {"decode": 0.33, "prefill": 0.2}


def _rank_config(full_config: CSAConfig, tp_size: int) -> CSAConfig:
    """One rank's self-contained config: ``n_heads`` and ``o_groups`` both divided by ``tp_size``.

    Dividing BOTH keeps ``group_in = n_heads * head_dim / o_groups`` at the full
    model's value, which is what ``wo_a`` expects -- the production model's
    ``ColumnParallelLinear`` is built from the global head and group counts. A rank
    is then an ordinary block of its own size, constructed with ``tp_size=1``.
    """
    from .csa_common import shard_for_tp

    return shard_for_tp(full_config, tp_size)


def _rank_config_for(phase: str, full_config: CSAConfig, tp_size: int) -> CSAConfig:
    """Rank config for `phase`: unsharded under sequence parallelism, else head-sharded."""
    if _seq_parallel_prefill(phase, full_config):
        return full_config
    return _rank_config(full_config, tp_size)


def _load_rank_weights(model: nn.Module, rank_weights: dict) -> None:
    """Copy one rank's reference shard into ``model``, reporting keys that found no home."""
    sd = model.state_dict()
    missing = [k for k in rank_weights if k not in sd]
    for k, v in rank_weights.items():
        if k in sd:
            sd[k].copy_(v.to(sd[k].dtype))
    model.load_state_dict(sd, strict=False)
    if missing:
        print(f"  [warn] {len(missing)} reference weights had no match in the NKI block:")
        for k in missing[:12]:
            print(f"         {k}")


def _check(out: torch.Tensor, ref: torch.Tensor, label: str) -> bool:
    """Report max/mean absolute and RMS-relative error against ``ref``; pass on ``_ATOL``."""
    diff = (out.float() - ref.float()).abs()
    max_abs = diff.max().item()
    rms_rel = (diff.pow(2).sum().sqrt() / (ref.float().pow(2).sum().sqrt() + 1e-12)).item()
    passed = max_abs < _ATOL
    print(
        f"  [{label}] ref std={ref.float().std().item():.4e}  max_abs_diff={max_abs:.2e}  "
        f"mean_abs_diff={diff.mean().item():.2e}  rms_rel={rms_rel:.2e}"
    )
    print(f"  [{label}] [{'PASS' if passed else 'FAIL'}] max_abs {max_abs:.2e} {'<' if passed else '>='} {_ATOL:.0e}")
    return passed


def _build_reference(phase: str, full_config: CSAConfig, tp_size: int) -> dict:
    """Run the CPU golden for ``phase`` and return its reference dict."""
    from .csa_block_torch import (
        generate_decode_block_reference_tp,
        generate_prefill_block_reference_tp,
    )

    gen = generate_prefill_block_reference_tp if phase == "prefill" else generate_decode_block_reference_tp
    # Sequence-parallel ranks each hold the FULL weights (all heads, all o_groups), so
    # the reference is generated unsharded and the ranks differ only in which output
    # rows they produce.
    ref_tp = 1 if _seq_parallel_prefill(phase, full_config) else tp_size
    return gen(full_config, tp_size=ref_tp, weight_gain=_WEIGHT_GAIN[phase])


def _reference_inputs(phase: str, ref: dict) -> tuple:
    """The trace inputs for ``phase``, taken from the reference so both see identical data."""
    if phase == "prefill":
        return (ref["x"],)
    return (ref["x_dec"], ref["kv_window"], ref["kv_compress"], ref["indexer_kv_cache"])


def _trace_rank(phase, full_config, tp_size, tp_rank, ref, inputs, workdir, replica_ranks=None):
    """Trace rank ``tp_rank``'s block into one NEFF, loading its reference weight shard.

    The rank is constructed as a self-contained ``tp_size=1`` block of its own
    (already divided) size -- see ``_rank_config``. With ``replica_ranks`` set, the
    cross-rank ``ncc.all_reduce`` is merged in and the traced block returns the
    full output; otherwise it returns this rank's partial.
    """
    import torch_neuronx

    seq_par = _seq_parallel_prefill(phase, full_config)
    cfg = _rank_config_for(phase, full_config, tp_size)
    if phase == "prefill":
        model = CSAAttentionXLA(cfg, replica_ranks=replica_ranks)
        if seq_par:
            model.set_q_range(
                sparse_prefill_q_range(
                    full_config.seq_len,
                    full_config.compressed_len,
                    full_config.index_topk,
                    full_config.compress_ratio,
                    tp_rank,
                    tp_size,
                )
            )

            if replica_ranks is not None:
                model.set_tp_shard((tp_rank, list(replica_ranks)))
    else:
        model = CSADecodeAttentionBlockNKI(cfg, tp_size=1, tp_rank=0, replica_ranks=replica_ranks)
    if ref is not None:
        # Sequence-parallel ranks all load the SAME (full) weight set.
        _load_rank_weights(model, ref["per_rank_weights"][0 if seq_par else tp_rank])
    model.eval()
    return torch_neuronx.trace(model, inputs, compiler_workdir=workdir)


def warm_up(traced, inputs) -> None:
    """Execute ``traced`` once and discard the result, before any graded execution.

    The graded runs execute each NEFF exactly once, so without this they would grade a
    NEFF's FIRST execution -- and the indexer top-k has a first-execution hazard that
    only shows up there. The one instance that has been root-caused is a uint32
    bitvec chain over ``nisa.topk``'s index output disagreeing with the same arithmetic
    on the host on run 0 and agreeing on every run after (see the snake-layout note in
    ``csa_decode_attention``); because run 1+ reads back the value the previous run
    left in that SBUF, a warm-up hides it rather than fixing it. The shipped fill does
    no such arithmetic and measures 1024/1024 winners on run 0, so this is now
    defensive rather than load-bearing.
    """
    traced(*inputs)


def run_sequential(phase: str, full_config: CSAConfig, tp_size: int) -> bool:
    """Trace the ranks one at a time on one core-set and sum their partials on the host.

    No collective is traced, so this isolates the block compute: if it passes here
    but fails under ``run_distributed``, the collective or the rank topology is at
    fault rather than the kernels.
    """
    print(f"=== CSA {phase} block, {tp_size}-rank head-parallel, sequential (host-summed) ===")
    ref = _build_reference(phase, full_config, tp_size)
    print(f"  reference: ||sum_r partial - full||_inf = {ref['max_sum_err']:.3e}")
    inputs = _reference_inputs(phase, ref)

    partials = []
    for r in range(tp_size):
        traced = _trace_rank(
            phase, full_config, tp_size, r, ref, inputs, workdir=f"./compiler_workdir_{phase}_block_rank{r}"
        )
        warm_up(traced, inputs)
        partials.append(traced(*inputs).float())
        print(f"  rank {r} traced and run")

    if _seq_parallel_prefill(phase, full_config):
        # Sequence-parallel: each rank produced a DISJOINT, contiguous block of output
        # rows, in rank order, so the full output is their concatenation -- there is no
        # cross-rank reduction to undo.
        combined = torch.cat(partials, dim=1)
    else:
        combined = torch.stack(partials, 0).sum(0)
    return _check(combined, ref["ref_output_full"], label=f"{phase}_full")


_LNC = 2
"""Logical NeuronCores each rank runs on. The attention and indexer kernels use both."""


def _pin_this_worker_to_its_cores() -> str | None:
    """Give this ``torchrun`` worker its own ``_LNC`` physical cores, so the ranks run concurrently.

    Rank ``r`` takes cores ``[base + r * _LNC, base + r * _LNC + _LNC - 1]``, with
    ``base`` read from the inherited ``NEURON_RT_VISIBLE_CORES`` (default 8, per the
    launch convention) so 4 ranks fill cores 8 to 15. Without this every worker sees
    the same cores and the ranks serialize instead of overlapping.

    If the inherited value ALREADY spans ``world * _LNC`` cores, it is left untouched
    and this returns it as-is. That matters when the range is chosen to avoid cores
    another job holds: the runtime allocates one logical core per process index out of
    the visible set, so re-narrowing each worker to a single pair here would discard
    the offset and every rank would fall back to logical cores ``0..world-1`` --
    observed as ``Requested:lnc0..lnc3 Available:0 (cores busy, ret=-16)`` in
    ``nrt_allocate_neuron_cores`` while an unrelated job held the low cores.

    MUST run before ``torch_neuronx`` is imported, which is why the driver imports
    it lazily inside ``_trace_rank`` rather than at module scope. Returns the pinned
    range, or None outside a multi-worker launch.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible = os.environ.get("NEURON_RT_VISIBLE_CORES", "8")
    bounds = visible.split("-")
    base = int(bounds[0])
    if len(bounds) == 2 and int(bounds[1]) - base + 1 >= world * _LNC:
        return visible
    first = base + local_rank * _LNC
    pinned = f"{first}-{first + _LNC - 1}"
    os.environ["NEURON_RT_VISIBLE_CORES"] = pinned
    return pinned


def run_distributed(phase: str, full_config: CSAConfig, tp_size: int) -> bool:
    """One rank per ``torchrun`` worker, each with the all-reduce merged into its NEFF.

    Every rank returns the FULL all-reduced output, so rank 0's result is graded
    directly against the 128-head golden.
    """
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    if world != tp_size:
        raise ValueError(f"launch with --nproc_per_node={tp_size} (got WORLD_SIZE={world})")

    # Pin FIRST, before anything that can touch the Neuron runtime: the process group
    # and the tracer both read NEURON_RT_VISIBLE_CORES when they initialize, so a pin
    # applied after them lands too late and every rank shares one core-set.
    pinned = _pin_this_worker_to_its_cores()

    import torch.distributed as dist

    # Registers the "xla" process-group backend. Without it init_process_group raises
    # `AssertionError: Unknown backend type xla` -- importing torch_xla alone is not
    # enough, the backend module itself has to be imported for its side effect.
    import torch_xla.distributed.xla_backend  # noqa: F401

    dist.init_process_group(backend="xla", init_method="env://", rank=rank, world_size=world)

    def barrier():
        try:
            dist.barrier()
        except Exception:
            pass

    if rank == 0:
        print(f"=== CSA {phase} block, {tp_size}-rank head-parallel x {_LNC} LNC, merged all-reduce ===")
    print(f"  rank {rank} pinned to physical cores [{pinned}]")

    # Every rank builds the reference itself: it is deterministic, so this is
    # cheaper and simpler than broadcasting it, and it keeps the ranks independent.
    ref = _build_reference(phase, full_config, tp_size)
    if rank == 0:
        print(f"  reference: ||sum_r partial - full||_inf = {ref['max_sum_err']:.3e}")
    inputs = _reference_inputs(phase, ref)

    traced = _trace_rank(
        phase,
        full_config,
        tp_size,
        rank,
        ref,
        inputs,
        workdir=f"./compiler_workdir_{phase}_block_rank{rank}",
        replica_ranks=list(range(tp_size)),
    )
    barrier()
    # Every rank warms before any rank grades: the collective is a rendezvous, so a
    # warm-up on one rank has to be matched on all of them.
    warm_up(traced, inputs)
    barrier()
    full_out = traced(*inputs)
    barrier()

    passed = True
    if rank == 0:
        passed = _check(full_out, ref["ref_output_full"], label=f"{phase}_full")
    barrier()
    dist.destroy_process_group()
    return passed


def emit_neff(phase: str, full_config: CSAConfig, tp_size: int, tp_rank: int, workdir: str, merged: bool) -> None:
    """Trace ONE rank's block into ``workdir`` for profiling, then stop.

    ``merged`` picks WHAT gets profiled. With it on, the cross-rank
    ``ncc.all_reduce`` is traced into the block, giving the real end-to-end artifact;
    profile it with ``neuron-explorer capture --collectives-workers-per-node=<tp_size>``.
    With it off the block is compute-only and captures as a single rank, which is the
    lower-variance instrument for A/B-ing kernel changes -- a collective capture adds
    hundreds of microseconds of cross-rank rendezvous to the reported total.

    Inputs are random rather than taken from the golden: nothing is graded here, and
    building the CPU reference for a long context costs more than the trace does.
    """
    import torch_neuronx  # noqa: F401  (imported for its side effect on tracing)

    T_c = full_config.compressed_len
    torch.manual_seed(200)
    if phase == "prefill":
        inputs = ((torch.randn(1, full_config.seq_len, full_config.dim) * 0.02).to(torch.bfloat16),)
    else:
        inputs = (
            (torch.randn(1, 1, full_config.dim) * 0.02).to(torch.bfloat16),
            (torch.randn(1, full_config.window_size, full_config.head_dim) * 0.01).to(torch.bfloat16),
            (torch.randn(1, T_c, full_config.head_dim) * 0.01).to(torch.bfloat16),
            (torch.randn(1, T_c, full_config.index_head_dim) * 0.01).to(torch.bfloat16),
        )

    kind = "merged block + all-reduce" if merged else "compute-only block"
    print(f"Emitting rank {tp_rank}/{tp_size} {phase} {kind} NEFF -> {workdir}")
    _trace_rank(
        phase,
        full_config,
        tp_size,
        tp_rank,
        ref=None,
        inputs=inputs,
        workdir=workdir,
        replica_ranks=list(range(tp_size)) if merged else None,
    )
    print(f"  NEFF at {os.path.join(workdir, 'graph.neff')}")


def main(argv=None) -> int:
    """Trace a CSA block and grade it against the CPU golden. Returns a process exit code."""
    import argparse

    from .csa_common import CSAConfigFull

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--phase", choices=("prefill", "decode"), default="decode")
    parser.add_argument("--seq-len", type=int, default=8192)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument(
        "--sequential",
        action="store_true",
        help="trace the ranks one at a time and sum on the host, instead of one rank per torchrun worker",
    )
    parser.add_argument(
        "--emit-neff", metavar="WORKDIR", help="trace one rank's block to WORKDIR for profiling, then exit"
    )
    parser.add_argument("--emit-rank", type=int, default=0)
    parser.add_argument(
        "--emit-merged", action="store_true", help="with --emit-neff, trace the all-reduce into the block"
    )
    args = parser.parse_args(argv)

    full_config = CSAConfigFull(seq_len=args.seq_len)
    if args.emit_neff:
        emit_neff(args.phase, full_config, args.tp_size, args.emit_rank, args.emit_neff, args.emit_merged)
        return 0

    distributed = not args.sequential and "RANK" in os.environ
    run = run_distributed if distributed else run_sequential
    passed = run(args.phase, full_config, args.tp_size)
    print("RESULT: PASSED" if passed else "RESULT: FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
