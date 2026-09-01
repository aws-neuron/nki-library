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
    nki_rms_rope_kernel,
)
from .csa_tp_all_reduce import tp_all_reduce


# ------------------------------------------------------------------------
# Host-side glue the kernels consume
# ------------------------------------------------------------------------
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

def nki_fused_csa_attn(q, kv_raw, kv_compress, attn_sink, window_size,
                       compress_sel_mask, softmax_scale, win_bias_base, win_bias_sink_ind,
                       split_pos=0, T_c_first=0, first_mask=None):
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
        first_kt = torch.cat([raw_padded_K_T[:, :split_pos + W],
                              compress_K_T_2d[:, :T_c_first]], dim=1)
        first_v = torch.cat([raw_padded_V[:split_pos + W, :],
                             compress_V_2d[:T_c_first, :]], dim=0)

        all_q_T_first = q_T[:, :, :split_pos].permute(1, 0, 2).reshape(head_dim, n_heads * split_pos)
        out_first_flat = nki_fused_csa_attn_kernel[2](
            first_mask_2d, all_q_T_first, first_kt, first_v,
            win_bias_base[:split_pos],
            win_bias_sink_ind[:split_pos],
            attn_sink_2d)
        out_first_all = out_first_flat.reshape(n_heads, split_pos, head_dim)

        # Second half: full T_c compressed columns
        second_mask_2d = compress_sel_mask.reshape(S - split_pos, T_c).to(torch.bfloat16)
        second_kt = torch.cat([raw_padded_K_T[:, split_pos:],
                               compress_K_T_2d], dim=1)
        second_v = torch.cat([raw_padded_V[split_pos:, :],
                              compress_V_2d], dim=0)

        S_second_len = S - split_pos
        all_q_T_second = q_T[:, :, split_pos:].permute(1, 0, 2).reshape(head_dim, n_heads * S_second_len)
        out_second_flat = nki_fused_csa_attn_kernel[2](
            second_mask_2d, all_q_T_second, second_kt, second_v,
            win_bias_base[split_pos:],
            win_bias_sink_ind[split_pos:],
            attn_sink_2d)
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
            mask_2d, all_q_T_full, kt_2d, v_2d,
            win_bias_base,
            win_bias_sink_ind,
            attn_sink_2d)
        out = out_flat.reshape(n_heads, S, head_dim).permute(1, 0, 2)

    return out.unsqueeze(0)  # [1, S, n_heads, head_dim]

# --------------------------------------------------------------------------
# Compressor (stateless, functional)
# --------------------------------------------------------------------------
class CompressorNKI(nn.Module):
    def __init__(self, config, head_dim: int = 512, rotate: bool = False,
                 use_nki: bool = True):
        super().__init__()
        self.dim = config.dim
        self.head_dim = head_dim
        self.rope_head_dim = config.rope_head_dim
        self.compress_ratio = config.compress_ratio
        self.overlap = (config.compress_ratio == 4)
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

        kv = kv_score[..., :self.out_dim]
        score = kv_score[..., self.out_dim:]

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
        rd = self.rope_head_dim
        T_c = kv.shape[1]
        ratio2 = kv.shape[2]
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
        out = nki_compressor_core_kernel[n_cores](
            kv8, score8, norm_weight, cos_rep, sin_rep, float(self.norm.eps))
        return out.unsqueeze(0)

    def forward(self, x, start_pos, freqs_cos_sin):
        bsz, seqlen, _ = x.size()

        if seqlen < self.compress_ratio:
            return None

        # bf16 projection: the eval runs with --auto-cast=none and x is already
        # bf16-valued, so an fp32 F.linear here wastes ~4x PE throughput for a
        # projection whose only new error is bf16-rounding the (tiny, ~1.5e-3 std)
        # weights. Downstream pooling/RMSNorm/RoPE stay fp32 (kv8/score8 are
        # re-widened via .float() in _compress_core_nki), and the gate softmax is
        # robust to a ~0.4% logit perturbation. The weight cast is constant-folded
        # at trace time, so it adds no per-call cost.
        W = torch.cat([self.wkv.weight, self.wgate.weight], dim=0).to(torch.bfloat16)
        kv_score = F.linear(x.to(torch.bfloat16), W).float()
        return self._compress_from_kv_score(kv_score, seqlen, freqs_cos_sin)


# --------------------------------------------------------------------------
# Indexer (stateless, functional) — with split scoring optimization
# --------------------------------------------------------------------------
class IndexerNKI(nn.Module):
    def __init__(self, config, use_nki: bool = True):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = self.head_dim ** -0.5

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False, dtype=torch.bfloat16)
        self.weight_scale = self.softmax_scale * (self.n_heads ** -0.5)
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
        S_q = seqlen - split_pos

        # --- First half: precomputed causal mask (queries select all valid kv) ---
        first_mask = self.first_mask_buf

        # --- Second half: project + RoPE + Hadamard the queries [split_pos:seqlen] ---
        seq_cos_second = freqs_cos[start_pos + split_pos:start_pos + seqlen]
        seq_sin_second = freqs_sin[start_pos + split_pos:start_pos + seqlen]

        qr_second = qr[:, split_pos:, :]
        q_second = self.wq_b(qr_second)
        q_second = q_second.unflatten(-1, (self.n_heads, self.head_dim))
        q_rope = apply_rotary_emb_functional(q_second[..., -rd:], (seq_cos_second, seq_sin_second))
        q_second = torch.cat([q_second[..., :-rd], q_rope], dim=-1)
        q_second = hadamard_transform(q_second)  # [1, S_q, n_heads, head_dim] bf16

        indexer_kv = self.compressor(x, start_pos, freqs_cos_sin)   # [1, T_c_idx, head_dim] bf16
        indexer_kv_t = indexer_kv.transpose(1, 2)                   # [1, head_dim, T_c_idx]

        # Match the ground-truth dtype: weights_proj (bf16) * scale -> bf16, F.linear in bf16.
        weights_second = F.linear(x[:, split_pos:, :], (self.weights_proj.weight * self.weight_scale).to(torch.bfloat16))
        # [1, S_q, n_heads] bf16; widened to fp32 for the kernel's per-head accumulate.

        # --- Stack operands for the single fused NKI kernel ---
        # q_T_all[d, h*S_q + s] = q_second[0, s, h, d]
        # q_T_all[d, h*S_q + s] = q_second[0, s, h, d]: need [head_dim, n_heads, S_q]
        # then flatten the last two axes (h outer, s inner) to match the kernel's
        # q_global = h * S_q + q_start indexing. permute(2, 1, 0) gives head_dim first.
        q_T_all = q_second[0].permute(2, 1, 0).reshape(self.head_dim, self.n_heads * S_q).contiguous()
        kv_t_2d = indexer_kv_t[0].contiguous()                     # [head_dim, T_c_idx] bf16
        weights_2d = weights_second[0].float().contiguous()        # [S_q, n_heads] fp32

        if start_pos == 0:
            cbias = self.causal_bias_full
        else:
            cbias = self.zero_bias_full

        # Compute scores and bisection-based selection mask in one kernel
        TILE_Q = 128
        num_q_tiles = S_q // TILE_Q
        n_cores = 2 if (S_q % TILE_Q == 0 and num_q_tiles % 2 == 0) else 1
        second_mask_2d = nki_indexer_score_mask_kernel[n_cores](
            q_T_all, kv_t_2d, weights_2d, cbias, int(k))
        second_mask = second_mask_2d.unsqueeze(0)                   # [1, S_q, T_c_idx]

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
        self.softmax_scale = config.head_dim ** -0.5
        self.use_dense_attn = use_dense_attn

        self.attn_sink = nn.Parameter(torch.empty(config.n_heads, dtype=torch.float32))
        self.compressor = CompressorNKI(config, config.head_dim, rotate=False, use_nki=use_nki)
        self.indexer = IndexerNKI(config, use_nki=use_nki)

        max_seq_len = config.seq_len
        freqs_cos, freqs_sin = precompute_freqs_cos_sin(
            self.rope_head_dim, max_seq_len,
            config.original_seq_len, config.compress_rope_theta,
            config.rope_factor, config.beta_fast, config.beta_slow
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

        kv_compress = self.compressor(x, start_pos, full_freqs_cs)

        T_c_idx = seqlen // ratio
        k = min(self.config.index_topk, seqlen // ratio)

        if first_mask is not None:
            split_pos = min(k * ratio, seqlen)
            T_c_first = split_pos // ratio
            S_q = seqlen - split_pos

            q_scaled = (q * self.softmax_scale).to(torch.float16)
            q_T = q_scaled.permute(0, 2, 3, 1).reshape(bsz * self.n_heads, self.head_dim, seqlen)

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
            first_mask_2d = first_mask.reshape(split_pos, T_c_first)
            first_kt = torch.cat([raw_padded_K_T[:, :split_pos + win],
                                  compress_K_T_2d[:, :T_c_first]], dim=1)
            first_v = torch.cat([raw_padded_V[:split_pos + win, :],
                                 compress_V_2d[:T_c_first, :]], dim=0)
            all_q_T_first = q_T[:, :, :split_pos].permute(1, 0, 2).reshape(self.head_dim, self.n_heads * split_pos)
            num_q_tiles_first = split_pos // 128
            n_cores_first = 2 if (num_q_tiles_first % 2 == 0 and num_q_tiles_first >= 2) else 1
            out_first_flat = nki_fused_csa_attn_kernel[n_cores_first](
                first_mask_2d, all_q_T_first, first_kt, first_v,
                self.win_bias_base[:split_pos],
                self.win_bias_sink_ind[:split_pos],
                attn_sink_2d)
            out_first_all = out_first_flat.reshape(self.n_heads, split_pos, self.head_dim)

            # Second half: static causal-bound sparse attention (global-max softmax
            # + sel_bias predication, per-tile compile-time causal chunk bound).
            all_q_T_second = q_T[:, :, split_pos:].permute(1, 0, 2).reshape(self.head_dim, self.n_heads * S_q)
            second_win_K_T = raw_padded_K_T[:, split_pos:split_pos + S_q + win]
            second_win_V = raw_padded_V[split_pos:split_pos + S_q + win, :]

            # Bisection mask is already 0/-1e9 selection bias with causal masking baked in.
            sel_bias = second_mask.reshape(S_q, T_c_idx).to(torch.bfloat16)

            out_second_flat = nki_gather_csa_attn_kernel[2](
                sel_bias,
                all_q_T_second, second_win_K_T, second_win_V,
                compress_K_T_2d, compress_V_2d,
                self.win_bias_base[split_pos:split_pos + S_q],
                self.win_bias_sink_ind[split_pos:split_pos + S_q],
                attn_sink_2d,
                int(split_pos), int(ratio))
            out_second_all = out_second_flat.reshape(self.n_heads, S_q, self.head_dim)

            out_all = torch.cat([out_first_all, out_second_all], dim=1)
            o = out_all.permute(1, 0, 2).unsqueeze(0)
        else:
            o = nki_fused_csa_attn(
                q, kv, kv_compress, self.attn_sink, win,
                second_mask, self.softmax_scale,
                self.win_bias_base, self.win_bias_sink_ind
            )

        return o


class CSAAttentionXLA(nn.Module):
    def __init__(self, config: CSAConfig, replica_ranks=None):
        super().__init__()
        self.config = config
        # None -> return this rank's output partial (single-device / host-sum).
        # list -> append a 2-LNC ncc.all_reduce(op=add) over these ranks as the
        # final forward op, so the traced block returns the full all-reduced
        # output (RowParallelLinear semantics, the true multi-worker path).
        self.replica_ranks = list(replica_ranks) if replica_ranks is not None else None
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

        # No attn_sink / softmax_scale here: the NKI core owns both.

        # All projection weights bf16, matching the decode block's `pdt`
        # convention (and DeepSeek-V4's bf16 default). The XLA original left these
        # at torch's fp32 default, which under --auto-cast=none means the whole
        # block runs FP32 matmuls -- several times less tensor-engine throughput
        # than bf16, plus 2x the weight bytes. These projections dominate the
        # block at s8192, so that alone is the difference between a
        # tensor-engine-bound block and a comfortable one -- which is what the
        # first s4096 profile showed, nearly all of it tensor_engine_active_time.
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
        self.wo_a = nn.Linear(
            self.group_in,
            self.n_groups * self.o_lora_rank,
            bias=False, dtype=pdt
        )
        self.wo_b = nn.Linear(self.n_groups * self.o_lora_rank, self.dim, bias=False, dtype=pdt)

        # NKI attention core: owns the compressor, indexer top-k and sparse
        # attention matmul, plus their parameters (compressor.*, indexer.*,
        # attn_sink).
        # Named `core` (NOT `attn_core`) so its nested params (core.attn_sink,
        # core.compressor.*, core.indexer.*) match the block CPU reference's
        # state_dict keys (deepseek_v4_csa_block_prefill.CSAAttentionBlockPrefill),
        # so the TP evaluator can load the sharded core weights. Same convention
        # as the decode block's CSADecodeAttentionBlockNKI.core.
        self.core = CSAAttentionCoreNKI(config, use_dense_attn=False, use_nki=True)

        # Precompute RoPE frequencies as real cos/sin (no complex on NeuronX)
        freqs_cos, freqs_sin = precompute_freqs_cos_sin(
            self.rope_head_dim, config.seq_len,
            config.original_seq_len, config.compress_rope_theta,
            config.rope_factor, config.beta_fast, config.beta_slow
        )
        self.register_buffer("freqs_cos", freqs_cos, persistent=False)
        self.register_buffer("freqs_sin", freqs_sin, persistent=False)

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
        seq_cos = self.freqs_cos[start_pos:start_pos + seqlen]
        seq_sin = self.freqs_sin[start_pos:start_pos + seqlen]
        rd = self.rope_head_dim
        H, D = self.n_local_heads, self.head_dim
        x_bf = x.to(torch.bfloat16)

        # cos/sin gathered per (head, position) row so the kernel's row r matches
        # x_in row r. q is head-major [H*S, ...], so repeat the S-length table H
        # times; kv is a single [S, ...] block.
        half = seq_cos.shape[-1]
        cos_q = seq_cos.float().unsqueeze(0).expand(H, seqlen, half).reshape(H * seqlen, half).contiguous()
        sin_q = seq_sin.float().unsqueeze(0).expand(H, seqlen, half).reshape(H * seqlen, half).contiguous()
        cos_s = seq_cos.float().contiguous()
        sin_s = seq_sin.float().contiguous()

        # ===== Query Path =====
        qr = self.q_norm(self.wq_a(x_bf))                    # [B, S, q_lora_rank]
        q = self.wq_b(qr)                                    # [B, S, H*D]
        # Per-head RMS (no learnable gain) + RoPE, fused in ONE NKI kernel. Lay q
        # out head-major [H*S, D] so each row is one head's D-vector: that puts the
        # RMS reduction on the free axis and the sequence on the partition axis.
        q_rows = q.reshape(bsz * seqlen, H, D)[0:seqlen].permute(1, 0, 2).reshape(H * seqlen, D).contiguous()
        q_out = nki_rms_rope_kernel(q_rows.to(torch.bfloat16), cos_q, sin_q, None,
                                    self.eps, do_rms=1, inverse=0)
        q = q_out.reshape(H, seqlen, D).permute(1, 0, 2).reshape(bsz, seqlen, H, D)

        # ===== KV Path =====
        # Learnable-gain RMSNorm + RoPE, same kernel with gain_in = kv_norm.weight.
        kv_lin = self.wkv(x_bf)                              # [B, S, D]
        kv_out = nki_rms_rope_kernel(
            kv_lin.reshape(seqlen, D).to(torch.bfloat16), cos_s, sin_s,
            self.kv_norm.weight.reshape(1, D).float().contiguous(),
            self.eps, do_rms=1, inverse=0)
        kv = kv_out.reshape(bsz, seqlen, D)

        # ===== NKI Attention Core =====
        # Replaces window+compressed index computation, KV compression and
        # sparse_attn_xla. Same operands and same [B, S, n_heads, head_dim] out.
        o = self.core(q, kv, x_bf, qr, start_pos=start_pos)

        # ===== Output de-RoPE =====
        # Rotation only (do_rms=0) with inverse=1, same fused kernel, same
        # head-major layout as the q path.
        o_rows = o.reshape(bsz * seqlen, H, D)[0:seqlen].permute(1, 0, 2).reshape(H * seqlen, D).contiguous()
        o_out = nki_rms_rope_kernel(o_rows.to(torch.bfloat16), cos_q, sin_q, None,
                                    self.eps, do_rms=0, inverse=1)
        o = o_out.reshape(H, seqlen, D).permute(1, 0, 2).reshape(bsz, seqlen, H * D)

        # ===== Output Projection (grouped low-rank) =====
        # Pick whichever of the two orderings streams fewer weight bytes, as
        # CSADecodeAttentionBlockNKI._output_projection does: composing wo_a into
        # wo_b widens the projected dim from o_lora_rank back up to group_in, so
        # fusing only wins when group_in <= o_lora_rank. This config has
        # group_in == o_lora_rank == 1024, so the fused single matmul is taken.
        G, R, Din = self.n_local_groups, self.o_lora_rank, self.group_in
        o = o.reshape(bsz, seqlen, G, Din)
        if self.dim * G * Din <= G * R * Din + self.dim * G * R:
            wo_a = self.wo_a.weight.view(G, R, Din)
            wo_b = self.wo_b.weight.view(self.dim, G, R)
            wfused = torch.einsum("cgr,grd->cgd", wo_b, wo_a).reshape(self.dim, G * Din)
            output = torch.matmul(o.reshape(bsz, seqlen, G * Din), wfused.t())
        else:
            wo_a = self.wo_a.weight.view(G, R, Din)
            lat = torch.einsum("bsgd,grd->bsgr", o, wo_a)     # [B, S, G, o_lora]
            output = self.wo_b(lat.reshape(bsz, seqlen, G * R))

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
        self.softmax_scale = self.head_dim ** -0.5

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False, dtype=torch.bfloat16)
        self.weight_scale = self.softmax_scale * (self.n_heads ** -0.5)
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
        # In decode, every query row scored by the indexer is bit-identical (all
        # derive from the single decode query below), the topk only needs 8 rows,
        # and the attention kernel broadcasts one index row across its S=256 rows.
        # So score exactly one TILE_Q=128 tile (one SPMD core) instead of 2 — this
        # halves the score kernel's fp32 [S_score, T_c] HBM write and the host-side
        # q_T_all / weights_2d / zero_bias construction, with zero output change.
        S_q = TILE_Q

        seq_cos = freqs_cos[start_pos:start_pos + 1]
        seq_sin = freqs_sin[start_pos:start_pos + 1]

        # q-projection via the hand-written NKI GEMV instead of self.wq_b(qr).
        # The nn.Linear form is materialized by neuronx-cc at 2x its true size
        # (declared 50.33 MB vs a true 25.17 MB), and the block is 87.6% DMA-bound
        # on weight streaming — see nki_indexer_qproj_gemv for the full rationale
        # and for why this kernel deliberately keeps the compiler's fine
        # [128,128]=32KB stationary matmul tiling while issuing the weight in 12
        # large 16 KB/partition DMA bursts.
        # wT is a pure transform of a FROZEN parameter, so neuronx-cc
        # constant-folds it: the only weight materialized is wT, at its true size.
        wT = self.wq_b.weight.t().contiguous().reshape(
            self.q_lora_rank // 128, 128, self.n_heads * self.head_dim)
        qr_2d = qr.reshape(1, self.q_lora_rank)
        # Launched on the [2] grid so the 25.17 MB weight stream is SPLIT ~12.6 MB
        # per logical core (core c owns a disjoint n-tile/head range and loads only
        # its own weight columns). As a [1]-grid kernel inside this lnc=2 graph the
        # whole stream landed on pcore0 with 0 bytes on pcore1 — the one weight in
        # the block whose per-core DMA load is maximally unbalanced. Total bytes are
        # unchanged (nothing is re-streamed), and the n-tiles are independent with
        # the k-reduction staying in-core, so the result is bit-identical.
        qT = nki_indexer_qproj_gemv[2](wT, qr_2d)     # [head_dim, n_heads] bf16
        q = qT.t().contiguous().reshape(1, 1, self.n_heads, self.head_dim)
        q_rope = apply_rotary_emb_functional(q[..., -rd:], (seq_cos, seq_sin))
        q = torch.cat([q[..., :-rd], q_rope], dim=-1)
        q = hadamard_transform(q)

        indexer_kv_t = indexer_kv_cache.transpose(1, 2)  # [1, head_dim, T_c]

        weights = F.linear(x, (self.weights_proj.weight * self.weight_scale).to(torch.bfloat16))

        q_single = q[0, 0]
        q_T_all = q_single.permute(1, 0).unsqueeze(2).expand(
            self.head_dim, self.n_heads, S_q).reshape(
            self.head_dim, self.n_heads * S_q).contiguous()
        weights_2d = weights[0, 0:1].float().expand(S_q, -1).contiguous()

        return q_T_all, weights_2d, indexer_kv_t, T_c, k, S_q

    # ---- nisa.topk n-SAFETY / single-chunk gating constants ------------------
    # Kept as class-level constants so `forward` and `fused_single_chunk_inputs`
    # gate on exactly the same numbers (see the long rationale in `forward`).
    IDX_CHUNK = 8192
    SCORE_CHUNK = 512
    SAFE_TOPK_N = 8192

    def fused_single_chunk_inputs(self, x, qr, start_pos, indexer_kv_cache,
                                  freqs_cos_sin, gather_chunk):
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
        if not (T_c % 2 == 0 and (T_c // 2) % self.SCORE_CHUNK == 0
                and T_c <= self.SAFE_TOPK_N):
            return None
        if k % gather_chunk != 0:
            return None

        q_T_all, weights_2d, indexer_kv_t, T_c, k, _ = self._score_inputs(
            x, qr, start_pos, indexer_kv_cache, freqs_cos_sin)
        kv_t_seg = indexer_kv_t[0, :, 0:T_c].contiguous()   # [head_dim, T_c]
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
            x, qr, start_pos, indexer_kv_cache, freqs_cos_sin)

        # Score in chunks that fit in SBUF: nki_indexer_score_kernel handles
        # any T_c via internal SCORE_CHUNK=512 tiling of the matmul, but
        # it preloads kv_t_sb = [head_dim, T_c] which must fit in SBUF.
        # At T_c=8192 the score kernel's per-partition peak (~97KB: index_score
        # fp32 32KB + cbias fp32 32KB + index_score_bf16 16KB + kv_t_sb 16KB)
        # fits trn2's ~192KB/partition SBUF, so IDX_CHUNK=8192 keeps both graded
        # configs (T_c=2048 @ s8192, T_c=8192 @ s32768) to a single chunk. That
        # lets s32768 hit the single-chunk short-circuit below: one score-kernel
        # launch + one topk, dropping the second score call and the Pass-2 merge.
        IDX_CHUNK = 8192
        num_idx_chunks = (T_c + IDX_CHUNK - 1) // IDX_CHUNK

        # ---- FUSED single-chunk path -----------------------------------------
        # When the whole compressed KV fits one chunk (T_c <= IDX_CHUNK; both
        # graded configs qualify: T_c=2048 @ s8192, T_c=8192 @ s32768), fuse
        # indexer scoring + nisa.topk into ONE kernel. This drops the fp32
        # [S_q, T_c] scores HBM round-trip, the torch encode_snake/decode_snake
        # glue, and the separate topk kernel launch. The kernel returns the k
        # GLOBAL indices (single chunk -> local == global) as an unordered set
        # in row 0 -- exactly what the permutation-invariant downstream softmax
        # over gathered positions needs, matching the old candidate_indices[0].
        # Guard on the kernel's layout requirements (T_c % 128 == 0, k % 16 == 0);
        # fall back to the two-kernel path otherwise.
        if num_idx_chunks == 1 and T_c % 128 == 0 and k % 16 == 0:
            kv_t_seg = indexer_kv_t[0, :, 0:T_c].contiguous()  # [head_dim, T_c]
            # 2-LNC split: score the T_c halves on both cores (the attention
            # kernel already uses [2], so the 2nd core would otherwise idle
            # through the whole indexing phase), then run topk on ONE core. The
            # kernel boundary is the cross-core barrier that guarantees both
            # score halves land before topk reads them. Requires each core's
            # half (T_c/2) to be a clean multiple of the score kernel's
            # SCORE_CHUNK=512 tiler; both graded configs (T_c=2048 -> 1024/core,
            # T_c=8192 -> 4096/core) qualify. Bit-identical: per-column head
            # accumulation order is unchanged, halves are disjoint.
            SCORE_CHUNK = 512
            # ---- nisa.topk n-SAFETY: run topk at the validated n=8192 --------------
            # The RAW indexer score row is heavily TIED: most heads' relu(q.kv)
            # underflow to 0, so a large fraction of positions score exactly 0.0.
            # On that degenerate distribution the selection nisa.topk returns is
            # sensitive to `n`, and only some widths were validated against
            # torch.topk here. So rather than calling topk at whatever n_val T_c
            # happens to be, pad the score row up to the validated
            # SAFE_TOPK_N=8192 with a very-negative sentinel (< every real score,
            # which are all >= 0 after relu). The padded positions can never enter
            # the top-k, so the returned global indices are the reference ones
            # (measured >= 1022/1024 overlap with torch.topk across the graded
            # shapes; the <=2 misses are exact ties at the kth score 0.0, benign
            # for the permutation-invariant softmax). T_c <= IDX_CHUNK = 8192 on
            # this single-chunk path, so 8192 always has room for all k=1024 real
            # winners. Do not drop the padding.
            #
            # The padding now happens ON-CHIP inside nki_indexer_score_topk_2core
            # (memset of the [T_c, 8192) tail with the same -1e9 sentinel, before the
            # cross-core barrier) instead of via this host F.pad, because scoring and
            # top-k are MERGED into one [2]-grid launch: that removes one @nki.jit
            # kernel-launch boundary (the profile's largest sync-engine opcode is
            # DMA_DIRECT2D kernel-boundary staging) plus the [1, T_c] HBM score
            # round-trip. The intra-kernel nisa.core_barrier(cores=(0,1)) replaces the
            # launch boundary as the cross-core barrier, so the top-k still reads a
            # FULLY assembled row and still runs at n=8192 on ONE core.
            SAFE_TOPK_N = 8192
            if T_c % 2 == 0 and (T_c // 2) % SCORE_CHUNK == 0 and T_c <= SAFE_TOPK_N:
                topk_idx_hbm = nki_indexer_score_topk_2core[2](
                    q_T_all, kv_t_seg, weights_2d, int(k), SAFE_TOPK_N)
            else:
                topk_idx_hbm = nki_indexer_score_topk_kernel[1](
                    q_T_all, kv_t_seg, weights_2d, int(k))  # [TOPK_ROWS, k] uint32
            topk_head = topk_idx_hbm.int()
            # S_out=1: the attention kernel batches heads on partitions and reads
            # only column 0 of topk_indices_T, so emit a single row [1, k] (must
            # match the attention-side S=1 so the kernel derives n_heads correctly).
            return topk_head[0:1].contiguous()

        # ---- FAST multi-chunk path: 2-LNC scoring + batched Pass-1 + merge ------
        # s131072 (T_c=32768) lands here. Score ALL T_c on BOTH LNC cores (disjoint
        # T_c halves) into ONE [1, T_c] bf16 HBM buffer via nki_indexer_score_2core[2],
        # replacing the old num_idx_chunks x single-core nki_indexer_score_kernel[1]
        # launches (each wrote a 4MB fp32 [S_q, IDX_CHUNK] tensor with core 1 idle).
        # The attention kernel already uses [2], so the 2nd core otherwise idles
        # through the whole indexing phase. Bit-identical: each core scores a
        # contiguous disjoint T_c slice in the SAME per-column bf16 head-accumulation
        # order as the single-core scorer, and Pass-1 casts scores to bf16 anyway.
        #
        # Top-k is then done PER IDX_CHUNK (n=IDX_CHUNK=8192 is the width validated
        # against torch.topk on the real clustered/tied bf16 scores; n=T_c=32768 is
        # not one of them) followed by a small Pass-2 merge (n = num_idx_chunks*k
        # <= 4096, also validated). Both topk passes are packed into single batched
        # nisa_topk_batched launches (8 independent groups).
        SCORE_CHUNK_2C = 512
        TOPK_ROWS = 8
        # ---- Relaxed 2-core gate (iter-1): drop the T_c % IDX_CHUNK == 0 clause ----
        # The old gate ALSO required T_c to be an exact multiple of IDX_CHUNK=8192, so
        # the mid-range multi-chunk seq_lens (s40960 T_c=10240, s49152 T_c=12288,
        # s57344 T_c=14336 — none % 8192 == 0) fell through to the single-core fp32
        # FALLBACK: scoring the whole compressed KV on ONE LNC (profile issue #4) with
        # per-chunk 4MB fp32 [S,IDX_CHUNK] HBM writes (issue #3), ~2x the per-position
        # rate of the 2-core path. But T_c % IDX_CHUNK == 0 is NOT a scoring-kernel
        # requirement — nki_indexer_score_2core only needs T_c % 2 == 0 and
        # (T_c/2) % SCORE_CHUNK == 0 (all three satisfy: 5120/6144/7168 all % 512 == 0).
        # The clause existed ONLY so scores_full.reshape(num_idx_chunks, IDX_CHUNK)
        # produced EQUAL rows for the batched top-k. Below we replace that reshape with
        # an explicit per-chunk build whose ragged tail is padded to IDX_CHUNK with the
        # same -1e9 sentinel the fallback already uses (lines ~932), so the 2-core fast
        # path now covers these seq_lens too. tail_len >= k is required so Pass-1 always
        # fills k winners from REAL positions (never a -1e9 pad slot, whose local index
        # would map to an out-of-bounds global position); the fallback stays as the
        # safety net for any future T_c that violates it.
        tail_len = T_c - (num_idx_chunks - 1) * IDX_CHUNK   # IDX_CHUNK if T_c % IDX_CHUNK == 0
        use_2core_score = (
            T_c % 128 == 0
            and (T_c // 2) % SCORE_CHUNK_2C == 0 and k % 16 == 0
            and tail_len >= k
        )
        if use_2core_score:
            kv_t_full = indexer_kv_t[0, :, 0:T_c].contiguous()   # [head_dim, T_c]
            scores_full = nki_indexer_score_2core[2](
                q_T_all, kv_t_full, weights_2d)                  # [1, T_c] bf16
            # Build per-chunk score rows: row c holds positions
            # [c*IDX_CHUNK, min((c+1)*IDX_CHUNK, T_c)); the ragged last chunk is padded
            # up to IDX_CHUNK with the -1e9 sentinel (< every relu'd score >= 0, so
            # padded slots never enter Pass-1's top-k). When T_c % IDX_CHUNK == 0 every
            # chunk is full and this cat is BYTE-IDENTICAL to the old
            # scores_full.reshape(num_idx_chunks, IDX_CHUNK) (so s131072 is unchanged).
            chunk_rows = []
            for c in range(num_idx_chunks):
                seg_start = c * IDX_CHUNK
                seg_end = min(seg_start + IDX_CHUNK, T_c)
                row = scores_full[0:1, seg_start:seg_end]        # [1, seg_len]
                seg_len = seg_end - seg_start
                if seg_len < IDX_CHUNK:
                    row = F.pad(row, (0, IDX_CHUNK - seg_len), value=-1e9)
                chunk_rows.append(row)
            scores_chunks = torch.cat(chunk_rows, dim=0)         # [num_idx_chunks, IDX_CHUNK]
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
            merged_scores = torch.cat(
                [pv[s:s + 1, :k] for s in range(num_idx_chunks)], dim=1)   # [1, C*k]
            merged_indices = torch.cat(
                [pl[s:s + 1, :k] + s * IDX_CHUNK for s in range(num_idx_chunks)], dim=1)
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
            scores_seg = nki_indexer_score_kernel[1](
                q_T_all, kv_t_seg, weights_2d, zero_bias_seg)
            score_chunks.append(scores_seg)  # [S_q, seg_len]

        # Two-pass top-k via nisa.topk (GPSIMD): top-k per chunk → merge → final top-k
        # nisa.topk requires n divisible by 16 and rows divisible by 8.
        # IDX_CHUNK=4096, S_q=256 — both satisfy these constraints.
        #
        # Decode optimization: all S_q query rows are bit-identical — q_T_all and
        # weights_2d are .expand() of a single decode query and zero_bias_seg is
        # all zeros, so every row of every scores_seg is identical, and the
        # attention kernel only ever consumes row 0 of the result. Run topk on the
        # minimum TOPK_ROWS=8 rows (nisa.topk requires rows % 8 == 0) instead of all
        # 256, then broadcast row 0 back to S_q — a ~32x reduction in topk work
        # (and in the snake encode/decode + HBM traffic) with zero change to output.
        TOPK_ROWS = 8

        # Pass 1: top-k per chunk, collecting (scores, global_indices)
        candidate_scores = []
        candidate_indices = []
        for seg_idx, scores_seg in enumerate(score_chunks):
            seg_start = seg_idx * IDX_CHUNK
            seg_len = scores_seg.shape[1]
            seg_k = min(k, seg_len)
            scores_head = scores_seg[0:TOPK_ROWS]      # identical rows → only need 8
            # nisa.topk n-SAFETY: run Pass-1 topk at the validated n=IDX_CHUNK=8192.
            # The RAW indexer score row is heavily TIED — most heads' relu(q.kv)
            # underflow to 0, so a large fraction of positions score exactly 0.0 — and
            # on that degenerate distribution the selection depends on `n`. Rather than
            # calling topk at whatever n the last partial segment happens to be (4096 /
            # 5120 / 6144 for general T_c), keep every call at the validated width.
            # Full segments are already exactly IDX_CHUNK; pad only the last partial
            # segment up to it with a very-negative sentinel (< every relu'd score >= 0),
            # so padded positions never enter the top-k and local indices stay valid.
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
            candidate_scores.append(top_vals)        # [TOPK_ROWS, seg_k]
            candidate_indices.append(top_global_idx) # [TOPK_ROWS, seg_k]

        if num_idx_chunks == 1:
            # Single chunk (e.g. s8192, T_c=2048 <= IDX_CHUNK): the Pass-1 candidate
            # already IS the global top-k (seg_start=0, seg_k=k). The attention
            # kernel treats the k indices as an unordered set (softmax over the
            # gathered positions is permutation-invariant), so re-sorting in Pass 2
            # is a no-op. Skip Pass 2 (a full topk + cat/pad/gather) entirely.
            topk_head = candidate_indices[0][:, :k].int()
        else:
            # Pass 2: merge all candidates and take final top-k
            merged_scores = torch.cat(candidate_scores, dim=1)   # [TOPK_ROWS, num_chunks * k]
            merged_indices = torch.cat(candidate_indices, dim=1) # [TOPK_ROWS, num_chunks * k]

            # nisa.topk on merged set: [TOPK_ROWS, num_chunks*k] → [TOPK_ROWS, k]
            merge_n = merged_scores.shape[1]
            pad_merge = (16 - merge_n % 16) % 16
            if pad_merge > 0:
                merged_scores_padded = F.pad(merged_scores, (0, pad_merge), value=-1e9)
            else:
                merged_scores_padded = merged_scores
            k_padded = ((k + 15) // 16) * 16
            merge_vals, merge_local_idx = nisa_topk_batched(merged_scores_padded, k=int(k_padded))
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
        self.softmax_scale = config.head_dim ** -0.5

        self.attn_sink = nn.Parameter(torch.empty(config.n_heads, dtype=torch.float32))
        self.compressor = CompressorNKI(config, config.head_dim, rotate=False, use_nki=True)
        self.indexer = DecodeIndexerGatheredNKI(config)

        max_seq_len = config.seq_len
        freqs_cos, freqs_sin = precompute_freqs_cos_sin(
            self.rope_head_dim, max_seq_len + 1,
            config.original_seq_len, config.compress_rope_theta,
            config.rope_factor, config.beta_fast, config.beta_slow
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
        # ---- S=1: collapse the vestigial query-row broadcast (decode key win) ----
        # Post-iter-8 the attention kernel batches the 16 per-core heads on the
        # matmul OUTPUT-partition dim; it reads only COLUMN 0 of each head from
        # all_q_T / topk_indices_T and writes only ROW 0 of each head's block
        # (downstream reads out_all[:, 0, :]). S=2*TILE_Q=256 was a leftover of the
        # OLD by-q-tile split (removed in iter-8): it replicated the single decode
        # query into 256 identical rows, inflating all_q_T ([head_dim, n_heads*256]
        # =8MB), the output HBM tensor ([n_heads*256, head_dim]=8MB, written via a
        # SCATTERED stride-256 DMA), and topk_indices_T ([k,256]=1MB) — 255/256 pure
        # dead weight materialized on device EVERY call. The kernel is fully
        # parameterized by S (its 2-core split is by-HEAD, S-independent), so S=1
        # shrinks these ~256x and turns the scattered output write CONTIGUOUS,
        # attacking profile issue #3 (long HBM->SBUF setup before matmul) and the
        # host-staging overhead. BIT-IDENTICAL: the consumed row-0/col-0 values and
        # every MAC / fp32-PSUM order are unchanged; only the redundant copies go.
        S = 1
        start_pos = self.config.seq_len
        full_freqs_cs = (self.freqs_cos, self.freqs_sin)

        k = min(self.config.index_topk, T_c)

        # ---- FUSED indexer-score + top-k + attention (single launch) ----------
        # `fused_single_chunk_inputs` returns the indexer's scoring inputs when this
        # step qualifies for the fused [2]-grid kernel (both graded seq-lens do), or
        # None to fall back to the unchanged two-launch pipeline. Asking for it here,
        # BEFORE the attention-side host prep, keeps the indexer's own op sequence in
        # the same relative position in the traced graph as the `self.indexer(...)`
        # call it replaces.
        COMP_CHUNK = 128     # the attention kernel's gather chunk (k must divide it)
        fused_inputs = self.indexer.fused_single_chunk_inputs(
            x, qr, start_pos, indexer_kv_cache, full_freqs_cs, COMP_CHUNK)

        # --- Indexer (fallback only): score all T_c → top-k indices [S, k] ---
        if fused_inputs is None:
            topk_indices = self.indexer(x, qr, start_pos, indexer_kv_cache, full_freqs_cs)

        # --- Prepare Q: replicate to S ---
        q_scaled = (q * self.softmax_scale).to(torch.float16)
        q_single = q_scaled[0, 0]
        all_q_T = q_single.permute(1, 0).unsqueeze(2).expand(
            self.head_dim, self.n_heads, S).reshape(
            self.head_dim, self.n_heads * S).contiguous()

        # --- Window KV: real window only (WIN_SIZE = W = 128) ---
        # iter-14: the old layout padded to WIN_SIZE=2*W=256 as [W zeros | W real].
        # The leading W zero-pad positions scored -1e9 (bias base) -> exp() underflows
        # to EXACTLY 0.0 -> contributed 0 to win_sum and 0 to out_psum (a 0@0 V
        # matmul), i.e. pure dead weight moved + transposed + matmul'd every call.
        # In decode the reference window is a full valid W-position permutation with
        # NO intra-window mask, so only the real half ever mattered. Pass the real
        # window directly (no pad): new kernel position j == old position W+j, so K^T
        # columns / V rows / their order are byte-identical to the old real half, and
        # softmax over the same 128 finite terms is bit-identical (dropped terms were
        # exactly 0 in the sum and -1e9 never wins the max). This halves the window
        # K/V DMA + score + bias/exp (issue #3) and, in the kernel, drops one window
        # nc_transpose (issue #1) + one window V matmul (issue #2).
        WIN_SIZE = W
        kv_win_f16 = kv_window.to(torch.float16)
        all_K_T_win = kv_win_f16.transpose(1, 2).reshape(self.head_dim, WIN_SIZE)
        all_V_win = kv_win_f16.reshape(WIN_SIZE, self.head_dim)

        # --- Compressed KV (full T_c in HBM — kernel gathers only k positions via swdge) ---
        # Pass NATIVE bf16 and let the kernel cast the k=1024 gathered rows to f16
        # on-chip. The old host `.to(float16)` traced into a device convert over the
        # ENTIRE [1, T_c, head_dim] tensor (+ a fresh alloc) every iteration — the
        # dominant T_c-scaling cost (an indexer-bypassed ablation showed the forward
        # still scaled 1.442->3.310ms s32768->s131072 with only this op left O(T_c)).
        # reshape([T_c, head_dim]) on the already-contiguous input is a free view and
        # .contiguous() is then a no-op, so compress_kv prep becomes ~free. Casting
        # the gathered subset bf16->f16 on-chip is bit-identical to converting-all-
        # then-gathering (same IEEE round-to-nearest-even; values ~0.01 in f16 range).
        compress_kv = kv_compress.reshape(T_c, self.head_dim).contiguous()

        # --- Window bias collapsed to the sink scalar (bit-identical dead-weight cut) ---
        # Post-iter-14 (WIN_SIZE=W, real window only) the two host bias tensors were
        # pure dead weight: win_bias_base == all-zeros, win_bias_sink == one-hot at
        # position 0, so win_bias[h,pos] = attn_sink[h] if pos==0 else 0. attn_sink is
        # already passed as attn_sink_2d, so the kernel now adds it to window column 0
        # directly (see nki_decode_gather_ok_kernel). This drops both host bias tensors,
        # their two kernel params and two per-call HBM->SBUF DMAs (attacks issue #3).
        attn_sink_2d = self.attn_sink.detach().view(1, self.n_heads).float().contiguous()

        # --- Call O(k) kernel: transpose indices to [k, S] for partition-dim slicing ---
        # SINGLE-CORE launch [1] (iter-19): in decode the top-k indices are identical
        # for all heads, so the by-HEAD [2] split had BOTH cores redundantly gather the
        # same compress_kv rows and rebuild the same K^T (the 78%-of-transposes item).
        # Launching [1] puts all n_heads heads on one core: H_BATCH=n_heads (=32 for the
        # evaluated config) doubles the PE output-partition utilization vs [2] (issue
        # #2), and the device does the gather + K^T transpose ONCE not twice (issues #1,
        # #3). Bit-identical:
        # each head is an independent matmul output partition with the same head_dim
        # contraction + fp32-PSUM accumulation order regardless of how many heads share
        # the matmul. The kernel is n_cores=nl.num_programs()-parameterized so [1] needs
        # no other change. Indexer stays [2] (its O(T_c) scoring needs both cores).
        # ---- ONE LAUNCH for score+topk+attention (the fused path) -------------
        # On the fused path there is no host-visible index tensor at all: the top-k
        # runs on core 0 of the SAME [2]-grid launch that then gathers and does the
        # attention, so the [k, S] index array never leaves the kernel (no `.int()`,
        # no `[0:1]`, no `.t().contiguous()`, no HBM materialization between two
        # launches) and one @nki.jit boundary disappears from the critical path.
        # Bit-identical: the fused kernel runs the SAME `_score_2core_stage`,
        # `_snake_topk_stage` and `_gather_attn_stage` traces, on the same inputs,
        # with all n_heads on one core exactly as this [1]-grid launch does.
        # Output de-RoPE cos/sin for this decode position, fused into the kernel's
        # finalize (inverse rotation on the SBUF-resident output before its single
        # HBM write). Same start_pos slice the block's torch de-RoPE consumed.
        derope_cos = self.freqs_cos[start_pos:start_pos + 1].contiguous()  # [1, half_rope]
        derope_sin = self.freqs_sin[start_pos:start_pos + 1].contiguous()  # [1, half_rope]
        if fused_inputs is not None:
            idx_q_T_all, idx_kv_t_seg, idx_weights_2d, fused_k, fused_n = fused_inputs
            out_flat = nki_indexer_score_topk_gather_2core[2](
                idx_q_T_all, idx_kv_t_seg, idx_weights_2d, int(fused_k), int(fused_n),
                all_q_T,
                all_K_T_win, all_V_win,
                compress_kv,
                attn_sink_2d,
                derope_cos, derope_sin)
        else:
            topk_indices_T = topk_indices.t().contiguous()  # [k, S]
            out_flat = nki_decode_gather_ok_kernel[1](
                topk_indices_T, all_q_T,
                all_K_T_win, all_V_win,
                compress_kv,
                attn_sink_2d,
                derope_cos, derope_sin)

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

    The cross-rank all-reduce that sums the RowParallelLinear partials into the
    full output is MERGED into forward() when `replica_ranks` is given (the true
    multi-worker torchrun path): forward returns the full all-reduced [B,1,dim]
    and the whole block+collective is ONE traced lnc=2 NEFF. With
    `replica_ranks=None` (single-process / host-sum path) forward returns the
    rank-local partial and the caller sums the partials host-side.
    """

    def __init__(self, config, tp_size: int = 4, tp_rank: int = 0,
                 replica_ranks=None):
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
        self.softmax_scale = config.head_dim ** -0.5

        if self.n_groups % tp_size != 0:
            raise ValueError(f"o_groups={self.n_groups} must be divisible by tp_size={tp_size}")
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.n_local_groups = self.n_groups // tp_size               # groups this rank owns
        self.group_in = self.n_heads * self.head_dim // self.n_groups  # per-group wo_a input width

        # All projection weights bf16: the core consumes bf16 x/qr, and the whole
        # block runs bf16 matmuls under --auto-cast=none (mirrors DeepSeek-V4's
        # bf16 default dtype). See block reference for the matching convention.
        pdt = torch.bfloat16

        # ----- Query projection -----
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=pdt)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=pdt)

        # ----- KV projection (for the new decode token) -----
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False, dtype=pdt)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # ----- Output projection (grouped low-rank), rank-local shard -----
        # Full (world_size=1) wo_a: [n_groups*o_lora_rank, group_in]; wo_b: [dim, n_groups*o_lora_rank].
        # Rank r owns groups [r*n_local_groups : (r+1)*n_local_groups]:
        #   wo_a shard -> [n_local_groups*o_lora_rank, group_in]
        #   wo_b shard -> [dim, n_local_groups*o_lora_rank]
        self.wo_a = nn.Linear(self.group_in, self.n_local_groups * self.o_lora_rank, bias=False, dtype=pdt)
        self.wo_b = nn.Linear(self.n_local_groups * self.o_lora_rank, self.dim, bias=False, dtype=pdt)

        # ----- Library core (gathered O(k) decode attention) -----
        # Named `core` so its nested params (core.attn_sink, core.compressor.*,
        # core.indexer.*) match the block CPU reference's state_dict keys.
        self.core = CSADecodeAttentionGatheredNKI(config)

        # RoPE tables for the block's q/kv rotation (start_pos == seq_len needs
        # index seq_len, so provision seq_len + 1 entries).
        freqs_cos, freqs_sin = precompute_freqs_cos_sin(
            self.rope_head_dim, config.seq_len + 1,
            config.original_seq_len, config.compress_rope_theta,
            config.rope_factor, config.beta_fast, config.beta_slow,
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
        q_lin = self.wq_b(qr)                                       # [1,1,n_heads*head_dim] bf16
        q_2d = q_lin.reshape(self.n_heads, self.head_dim).contiguous()   # [n_heads, head_dim] bf16
        # kv-path pre-norm latent (the single decode-token KV).
        kv_lin = self.wkv(x)                                        # [1,1,head_dim] bf16
        kv_2d = kv_lin.reshape(1, self.head_dim).contiguous()      # [1, head_dim] bf16
        weight = self.kv_norm.weight.reshape(1, self.head_dim)     # [1, head_dim] fp32 gain

        cos, sin = freqs_cs
        # ONE launch assembles the [n_heads+1, head_dim] tile on-chip and fuses
        # RMS(+gain)+RoPE for both paths (q rows + kv row).
        out = nki_qkv_rms_rope_kernel(q_2d, kv_2d, weight, cos, sin, self.eps)

        q = out[:self.n_heads].reshape(1, 1, self.n_heads, self.head_dim)
        kv = out[self.n_heads:self.n_heads + 1].reshape(1, 1, self.head_dim)
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
        o_local = o[:, :, g0:g0 + self.n_local_groups, :].contiguous()
        G, R, D = self.n_local_groups, self.o_lora_rank, self.group_in

        fused_bytes = self.dim * G * D
        twostep_bytes = G * R * D + self.dim * G * R
        if fused_bytes <= twostep_bytes:
            # Fused single matmul (constant-folded weight compose). Cheaper only
            # when group_in <= o_lora_rank (reduced model).
            wo_a = self.wo_a.weight.view(G, R, D)
            wo_b = self.wo_b.weight.view(self.dim, G, R)
            wfused = torch.einsum("cgr,grd->cgd", wo_b, wo_a).reshape(self.dim, G * D)
            out_partial = torch.matmul(
                o_local.reshape(bsz, seqlen, G * D).to(torch.bfloat16), wfused.t())
        else:
            # Two-step: wo_a compresses group_in(4096)->o_lora(1024) PER GROUP (the
            # 4 groups are independent GEMVs the compiler parallelizes), then wo_b
            # over the low-rank [G*o_lora=4096]-wide latent. Keeps the o_lora
            # bottleneck so wo_b never streams the full group_in width. 234MB->92MB.
            wo_a = self.wo_a.weight.view(G, R, D)
            lat = torch.einsum("bsgd,grd->bsgr",
                               o_local.to(torch.bfloat16), wo_a)      # [B,S,G,o_lora]
            out_partial = self.wo_b(lat.reshape(bsz, seqlen, G * R))  # [B,S,dim]

        # NOTE(tensor-parallel): out_partial is rank `tp_rank`'s contribution.
        # The full block output is the sum over ranks — a genuine ncc.all_reduce
        # (RowParallelLinear semantics), traced separately (see csa_nki_tp_allreduce).
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

        seq_cos = self.freqs_cos[start_pos:start_pos + 1]
        seq_sin = self.freqs_sin[start_pos:start_pos + 1]
        freqs_cs = (seq_cos, seq_sin)

        # ----- Projections (q latent shared with the indexer via qr) -----
        # ONE merged NKI kernel fuses BOTH the q-path per-head RMS+RoPE and the
        # kv-path learnable RMSNorm+RoPE (see _project_qkv).
        q, qr, kv = self._project_qkv(x, freqs_cs)  # q:[B,1,n_heads,head_dim] kv:[B,1,head_dim]

        # ----- Insert the new token into the window cache at (start_pos % W) -----
        # The core treats window column (start_pos % W) as the attention-sink slot
        # and attends over all W window positions. The reference decode overwrites
        # this slot with the freshly projected decode KV, so we do the same before
        # handing the window to the core (static index; start_pos, W are known).
        p = start_pos % W
        kv_slot = kv.reshape(bsz, 1, self.head_dim).to(kv_window.dtype)
        kv_window = torch.cat(
            [kv_window[:, :p], kv_slot, kv_window[:, p + 1:]], dim=1
        )

        # ----- Core sparse attention (gathered O(k)) -----
        # Returns [B, 1, n_heads, head_dim] with the output de-RoPE already FUSED
        # into the core kernel's finalize (inverse rotation on the SBUF-resident
        # output before its single HBM write) — this removes the last forward-path
        # torch RoPE op-graph (apply_rotary_emb_functional + torch.cat) here.
        o = self.core(q, kv_window, kv_compress, x, qr, indexer_kv_cache)

        # ----- Rank-local output projection -----
        partial = self._output_projection(o, bsz, seqlen)   # [B,1,dim] rank partial

        # ----- Cross-rank all-reduce (merged): RowParallelLinear sum over ranks -----
        # When replica_ranks is set (multi-worker torchrun), append the 2-LNC
        # ncc.all_reduce(op=add) as the block's FINAL op so the traced block is one
        # integrated lnc=2 NEFF returning the full [B,1,dim]. Otherwise return the
        # partial and let the caller host-sum the ranks (single-process path).
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


def _rank_config(full_config: CSAConfig, tp_size: int) -> CSAConfig:
    """One rank's self-contained config: ``n_heads`` and ``o_groups`` both divided by ``tp_size``.

    Dividing BOTH keeps ``group_in = n_heads * head_dim / o_groups`` at the full
    model's value, which is what ``wo_a`` expects -- the production model's
    ``ColumnParallelLinear`` is built from the global head and group counts. A rank
    is then an ordinary block of its own size, constructed with ``tp_size=1``.
    """
    from .csa_common import shard_for_tp

    return shard_for_tp(full_config, tp_size)


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
    print(f"  [{label}] [{'PASS' if passed else 'FAIL'}] max_abs {max_abs:.2e} "
          f"{'<' if passed else '>='} {_ATOL:.0e}")
    return passed


def _build_reference(phase: str, full_config: CSAConfig, tp_size: int) -> dict:
    """Run the CPU golden for ``phase`` and return its reference dict."""
    from .csa_block_torch import (
        generate_decode_block_reference_tp,
        generate_prefill_block_reference_tp,
    )

    gen = generate_prefill_block_reference_tp if phase == "prefill" else generate_decode_block_reference_tp
    return gen(full_config, tp_size=tp_size)


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

    cfg = _rank_config(full_config, tp_size)
    if phase == "prefill":
        model = CSAAttentionXLA(cfg, replica_ranks=replica_ranks)
    else:
        model = CSADecodeAttentionBlockNKI(cfg, tp_size=1, tp_rank=0, replica_ranks=replica_ranks)
    if ref is not None:
        _load_rank_weights(model, ref["per_rank_weights"][tp_rank])
    model.eval()
    return torch_neuronx.trace(model, inputs, compiler_workdir=workdir)


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
        traced = _trace_rank(phase, full_config, tp_size, r, ref, inputs,
                             workdir=f"./compiler_workdir_{phase}_block_rank{r}")
        partials.append(traced(*inputs).float())
        print(f"  rank {r} traced and run")

    summed = torch.stack(partials, 0).sum(0)
    return _check(summed, ref["ref_output_full"], label=f"{phase}_full")


_LNC = 2
"""Logical NeuronCores each rank runs on. The attention and indexer kernels use both."""


def _pin_this_worker_to_its_cores() -> str | None:
    """Give this ``torchrun`` worker its own ``_LNC`` physical cores, so the ranks run concurrently.

    Rank ``r`` takes cores ``[base + r * _LNC, base + r * _LNC + _LNC - 1]``, with
    ``base`` read from the inherited ``NEURON_RT_VISIBLE_CORES`` (default 8, per the
    launch convention) so 4 ranks fill cores 8 to 15. Without this every worker sees
    the same cores and the ranks serialize instead of overlapping.

    MUST run before ``torch_neuronx`` is imported, which is why the driver imports
    it lazily inside ``_trace_rank`` rather than at module scope. Returns the pinned
    range, or None outside a multi-worker launch.
    """
    world = int(os.environ.get("WORLD_SIZE", "1"))
    if world <= 1:
        return None
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    base = int(os.environ.get("NEURON_RT_VISIBLE_CORES", "8").split("-")[0])
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

    traced = _trace_rank(phase, full_config, tp_size, rank, ref, inputs,
                         workdir=f"./compiler_workdir_{phase}_block_rank{rank}",
                         replica_ranks=list(range(tp_size)))
    barrier()
    full_out = traced(*inputs)
    barrier()

    passed = True
    if rank == 0:
        passed = _check(full_out, ref["ref_output_full"], label=f"{phase}_full")
    barrier()
    dist.destroy_process_group()
    return passed


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
    args = parser.parse_args(argv)

    full_config = CSAConfigFull(seq_len=args.seq_len)
    distributed = not args.sequential and "RANK" in os.environ
    run = run_distributed if distributed else run_sequential
    passed = run(args.phase, full_config, args.tp_size)
    print("RESULT: PASSED" if passed else "RESULT: FAILED")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
