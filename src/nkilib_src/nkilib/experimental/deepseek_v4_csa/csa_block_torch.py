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

"""CPU reference for the whole DeepSeek-V4 CSA attention block.

A plain-torch transcription of the model's own attention block, following
DeepSeek-V4-Pro's inference code: it selects with ``torch.topk`` rather than
``nisa.topk``, attends densely over the gathered positions, and keeps the caches
as ordinary tensors. It is deliberately written for clarity over speed -- its job
is to be obviously correct so the fused kernels can be graded against it.

Both phases share one ``CSAAttentionCore``: ``prefill()`` populates the window and
compressed caches over ``S`` positions, and ``forward()`` runs a single decode
step against them. That sharing is what makes the decode reference trustworthy --
the caches the decode step reads are produced by the same code path that the
prefill reference validates.

``generate_prefill_block_reference_tp`` / ``generate_decode_block_reference_tp``
build a full 128-head model, shard its weights head-parallel over ``tp_size``
ranks, and return each rank's weights alongside the FULL-model golden output. The
NKI block then loads one rank's shard and its partials are summed to compare
against that golden -- which is what makes the comparison a test of the tensor
parallelism as well as of the kernels.

Used by ``csa_block.main()``. The integration tests grade individual kernels
against the per-kernel references in the ``*_torch`` modules instead.
"""

import math

import torch
import torch.nn.functional as F
from torch import nn

from .csa_common import CSAConfig
# --------------------------------------------------------------------------
# RoPE
# --------------------------------------------------------------------------
def precompute_freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    def find_correction_dim(num_rotations, dim, base, max_seq_len):
        return dim * math.log(max_seq_len / (num_rotations * 2 * math.pi)) / (2 * math.log(base))

    def find_correction_range(low_rot, high_rot, dim, base, max_seq_len):
        low = math.floor(find_correction_dim(low_rot, dim, base, max_seq_len))
        high = math.ceil(find_correction_dim(high_rot, dim, base, max_seq_len))
        return max(low, 0), min(high, dim - 1)

    def linear_ramp_factor(min_val, max_val, dim):
        if min_val == max_val:
            max_val += 0.001
        linear_func = (torch.arange(dim, dtype=torch.float32) - min_val) / (max_val - min_val)
        return torch.clamp(linear_func, 0, 1)

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
        smooth = 1 - linear_ramp_factor(low, high, dim // 2)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Applies rotary positional embeddings in-place."""
    dtype = x.dtype
    x_complex = torch.view_as_complex(x.float().unflatten(-1, (-1, 2)))
    if inverse:
        freqs_cis = freqs_cis.conj()
    if x_complex.ndim == 3:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), x_complex.size(-1))
    else:
        freqs_cis = freqs_cis.view(1, x_complex.size(1), 1, x_complex.size(-1))
    result = torch.view_as_real(x_complex * freqs_cis).flatten(-2)
    x.copy_(result.to(dtype))
    return x


# --------------------------------------------------------------------------
# RMSNorm
# --------------------------------------------------------------------------
class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x: torch.Tensor):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        x = x * torch.rsqrt(var + self.eps)
        return (self.weight * x).to(dtype)


# --------------------------------------------------------------------------
# Hadamard Transform (for Indexer's rotate_activation)
# --------------------------------------------------------------------------
def hadamard_transform_cpu(x: torch.Tensor) -> torch.Tensor:
    """CPU implementation of Hadamard transform. Works for power-of-2 dims."""
    n = x.shape[-1]
    assert n > 0 and (n & (n - 1)) == 0, f"Dim must be power of 2, got {n}"
    scale = n ** -0.5
    h = x.float()
    step = 1
    while step < n:
        idx_even = torch.arange(0, n, 2 * step)
        for i in range(step):
            a = h[..., idx_even + i].clone()
            b = h[..., idx_even + step + i].clone()
            h[..., idx_even + i] = a + b
            h[..., idx_even + step + i] = a - b
        step *= 2
    return (h * scale).to(x.dtype)


def rotate_activation(x: torch.Tensor) -> torch.Tensor:
    return hadamard_transform_cpu(x)


# --------------------------------------------------------------------------
# Sparse Attention (pure PyTorch)
# --------------------------------------------------------------------------
def sparse_attn_cpu(q, kv, attn_sink, topk_idxs, softmax_scale):
    """
    Pure PyTorch sparse attention.

    Args:
        q: [B, S, n_heads, head_dim] (bf16)
        kv: [B, T_kv, head_dim] (bf16) -- shared KV (K=V in CSA)
        attn_sink: [n_heads] (fp32) -- per-head bias for position 0
        topk_idxs: [B, S, topk_count] -- indices into kv dim 1 (-1 = masked)
        softmax_scale: float

    Returns:
        o: [B, S, n_heads, head_dim] (bf16)
    """
    B, S, n_heads, head_dim = q.shape
    topk_count = topk_idxs.shape[-1]

    mask = (topk_idxs == -1)
    safe_idxs = topk_idxs.clamp(min=0)

    batch_idx = torch.arange(B, device=kv.device)[:, None, None].expand(B, S, topk_count)
    gathered_kv = kv[batch_idx, safe_idxs]

    scores = torch.einsum("bshd,bstd->bsht", q.float(), gathered_kv.float()) * softmax_scale

    sink_mask = (safe_idxs == 0) & (~mask)
    sink_bias = attn_sink[None, None, :, None].expand(B, S, n_heads, topk_count)
    scores = scores + sink_bias * sink_mask[:, :, None, :].float()

    scores = scores.masked_fill(mask[:, :, None, :].expand_as(scores), float("-inf"))

    attn_weights = torch.softmax(scores, dim=-1)
    attn_weights = attn_weights.masked_fill(mask[:, :, None, :].expand_as(attn_weights), 0.0)

    o = torch.einsum("bsht,bstd->bshd", attn_weights, gathered_kv.float())
    return o.to(torch.bfloat16)


# --------------------------------------------------------------------------
# Index computation
# --------------------------------------------------------------------------
def get_window_topk_idxs(window_size, bsz, seqlen, start_pos):
    if start_pos >= window_size - 1:
        start_pos_mod = start_pos % window_size
        matrix = torch.cat([
            torch.arange(start_pos_mod + 1, window_size),
            torch.arange(0, start_pos_mod + 1)
        ], dim=0)
    elif start_pos > 0:
        matrix = F.pad(torch.arange(start_pos + 1), (0, window_size - start_pos - 1), value=-1)
    else:
        base = torch.arange(seqlen).unsqueeze(1)
        matrix = (base - window_size + 1).clamp(0) + torch.arange(min(seqlen, window_size))
        matrix = torch.where(matrix > base, -1, matrix)
    return matrix.unsqueeze(0).expand(bsz, -1, -1)


# --------------------------------------------------------------------------
# Compressor (compress_ratio=4, with overlap)
# --------------------------------------------------------------------------
class Compressor(nn.Module):
    def __init__(self, config: CSAConfig, head_dim: int = 512, rotate: bool = False):
        super().__init__()
        self.dim = config.dim
        self.head_dim = head_dim
        self.rope_head_dim = config.rope_head_dim
        self.compress_ratio = config.compress_ratio
        self.overlap = (config.compress_ratio == 4)
        self.rotate = rotate
        coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(config.compress_ratio, coff * self.head_dim, dtype=torch.float32))
        self.wkv = nn.Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.wgate = nn.Linear(self.dim, coff * self.head_dim, bias=False, dtype=torch.float32)
        self.norm = RMSNorm(self.head_dim, config.norm_eps)

        self.kv_cache = None
        self.freqs_cis = None

    def overlap_transform(self, tensor: torch.Tensor, value=0):
        b, s, _, _ = tensor.size()
        ratio, d = self.compress_ratio, self.head_dim
        new_tensor = tensor.new_full((b, s, 2 * ratio, d), value)
        new_tensor[:, :, ratio:] = tensor[:, :, :, d:]
        new_tensor[:, 1:, :ratio] = tensor[:, :-1, :, :d]
        return new_tensor

    def forward(self, x: torch.Tensor, start_pos: int):
        assert self.kv_cache is not None
        bsz, seqlen, _ = x.size()
        ratio = self.compress_ratio
        rd = self.rope_head_dim

        x_float = x.float()
        kv = self.wkv(x_float)
        score = self.wgate(x_float)

        if start_pos == 0:
            should_compress = seqlen >= ratio
            remainder = seqlen % ratio
            cutoff = seqlen - remainder

            if remainder > 0:
                kv = kv[:, :cutoff]
                score = score[:, :cutoff]

            kv = kv.unflatten(1, (-1, ratio))
            score = score.unflatten(1, (-1, ratio)) + self.ape

            if self.overlap:
                kv = self.overlap_transform(kv, 0)
                score = self.overlap_transform(score, float("-inf"))

            kv = (kv * score.softmax(dim=2)).sum(dim=2)
        else:
            should_compress = (start_pos + 1) % self.compress_ratio == 0
            if not should_compress:
                return None
            return None

        if not should_compress:
            return None

        kv = self.norm(kv.to(torch.bfloat16))

        if start_pos == 0:
            freqs_cis = self.freqs_cis[:cutoff:ratio]
        else:
            freqs_cis = self.freqs_cis[start_pos + 1 - self.compress_ratio].unsqueeze(0)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)

        if self.rotate:
            kv = rotate_activation(kv)

        if start_pos == 0:
            self.kv_cache[:bsz, :seqlen // ratio] = kv

        return kv


# --------------------------------------------------------------------------
# Indexer (for compress_ratio=4 layers)
# --------------------------------------------------------------------------
class Indexer(nn.Module):
    def __init__(self, config: CSAConfig):
        super().__init__()
        self.dim = config.dim
        self.n_heads = config.index_n_heads
        self.n_local_heads = config.index_n_heads
        self.head_dim = config.index_head_dim
        self.rope_head_dim = config.rope_head_dim
        self.index_topk = config.index_topk
        self.q_lora_rank = config.q_lora_rank
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = self.head_dim ** -0.5

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=torch.bfloat16)
        self.weights_proj = nn.Linear(self.dim, self.n_heads, bias=False, dtype=torch.bfloat16)

        self.compressor = Compressor(config, self.head_dim, rotate=True)

        self.kv_cache = None
        self.freqs_cis = None

    def forward(self, x: torch.Tensor, qr: torch.Tensor, start_pos: int, offset: int):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]
        ratio = self.compress_ratio
        rd = self.rope_head_dim
        end_pos = start_pos + seqlen

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache
            self.compressor.freqs_cis = self.freqs_cis

        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_local_heads, self.head_dim))
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        q = rotate_activation(q)

        self.compressor(x, start_pos)

        weights = self.weights_proj(x) * (self.softmax_scale * self.n_heads ** -0.5)

        index_score = torch.einsum(
            "bshd,btd->bsht",
            q,
            self.kv_cache[:bsz, :end_pos // ratio]
        )
        index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)

        if start_pos == 0:
            mask = (
                torch.arange(seqlen // ratio).repeat(seqlen, 1)
                >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            )
            index_score = index_score + torch.where(mask, float("-inf"), torch.zeros_like(mask, dtype=index_score.dtype))

        k = min(self.index_topk, end_pos // ratio)
        topk_idxs = index_score.topk(k, dim=-1)[1]

        if start_pos == 0:
            mask = topk_idxs >= torch.arange(1, seqlen + 1).unsqueeze(1) // ratio
            topk_idxs = torch.where(mask, torch.tensor(-1), topk_idxs + offset)
        else:
            topk_idxs = topk_idxs + offset

        return topk_idxs


# --------------------------------------------------------------------------
# Core Attention Module (Steps 11-16) -- Decode
# --------------------------------------------------------------------------
class CSAAttentionCore(nn.Module):
    """
    Core sparse attention for CSA (compress_ratio=4).

    Owns: Compressor, Indexer, attn_sink, freqs_cis, kv_cache.
    Does NOT own: wq_a, wq_b, wkv, kv_norm, wo_a, wo_b (projection weights).
    """
    def __init__(self, config: CSAConfig):
        super().__init__()
        self.config = config
        self.n_heads = config.n_heads
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.window_size = config.window_size
        self.compress_ratio = config.compress_ratio
        self.softmax_scale = config.head_dim ** -0.5

        self.attn_sink = nn.Parameter(torch.empty(config.n_heads, dtype=torch.float32))
        self.compressor = Compressor(config, config.head_dim, rotate=False)
        self.indexer = Indexer(config)

        max_seq_len = config.seq_len
        freqs_cis = precompute_freqs_cis(
            self.rope_head_dim, max_seq_len + 1,
            config.original_seq_len, config.compress_rope_theta,
            config.rope_factor, config.beta_fast, config.beta_slow
        )
        self.register_buffer("freqs_cis", freqs_cis, persistent=False)

        kv_cache_size = config.window_size + max_seq_len // self.compress_ratio
        self.register_buffer(
            "kv_cache",
            torch.zeros(config.batch_size, kv_cache_size, self.head_dim, dtype=torch.bfloat16)
        )

        indexer_cache_size = max_seq_len // self.compress_ratio
        self.indexer.kv_cache = torch.zeros(
            config.batch_size, indexer_cache_size, config.index_head_dim, dtype=torch.bfloat16
        )

    def prefill(self, q, kv, x, qr):
        """
        Run prefill to populate KV caches. Identical to the prefill path in
        deepseek_v4_csa_attn_core.py.

        Args:
            q: [B, S, n_heads, head_dim]
            kv: [B, S, head_dim]
            x: [B, S, dim]
            qr: [B, S, q_lora_rank]

        Returns:
            o: [B, S, n_heads, head_dim] -- prefill attention output
        """
        bsz, seqlen, _ = x.size()
        win = self.window_size
        ratio = self.compress_ratio
        start_pos = 0

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            self.indexer.freqs_cis = self.freqs_cis

        # Step 11: Window indices
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)

        # Step 13: Indexer
        offset = kv.size(1)
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)

        # Step 14: Merge
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # Step 12 & 15: Compress KV and prepare
        if seqlen <= win:
            self.kv_cache[:bsz, :seqlen] = kv
        else:
            cutoff = seqlen % win
            self.kv_cache[:bsz, cutoff:win], self.kv_cache[:bsz, :cutoff] = \
                kv[:, -win:].split([win - cutoff, cutoff], dim=1)

        kv_compress = self.compressor(x, start_pos)
        if kv_compress is not None:
            kv_full = torch.cat([kv, kv_compress], dim=1)
        else:
            kv_full = kv

        # Step 16: Sparse attention
        o = sparse_attn_cpu(q, kv_full, self.attn_sink, topk_idxs, self.softmax_scale)
        return o

    def forward(self, q, kv, x, qr, start_pos):
        """
        Decode step: compute attention for a single new token.

        Args:
            q: [B, 1, n_heads, head_dim] -- projected query for new token
            kv: [B, 1, head_dim] -- projected KV for new token
            x: [B, 1, dim] -- raw hidden states for new token
            qr: [B, 1, q_lora_rank] -- normalized query latent for new token
            start_pos: int -- position of the new token (== prefill seq_len)

        Returns:
            o: [B, 1, n_heads, head_dim] -- attention output
        """
        bsz, seqlen, _ = x.size()
        assert seqlen == 1, f"Decode expects seqlen=1, got {seqlen}"
        win = self.window_size

        if self.compressor.kv_cache is None:
            self.compressor.kv_cache = self.kv_cache[:, win:]
            self.compressor.freqs_cis = self.freqs_cis
            self.indexer.freqs_cis = self.freqs_cis

        # Step 11: Window indices
        topk_idxs = get_window_topk_idxs(win, bsz, seqlen, start_pos)

        # Step 13: Indexer
        offset = win
        compress_topk_idxs = self.indexer(x, qr, start_pos, offset)

        # Step 14: Merge
        topk_idxs = torch.cat([topk_idxs, compress_topk_idxs], dim=-1)
        topk_idxs = topk_idxs.int()

        # Step 12 & 15: Place new KV in window cache and try compress
        self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
        self.compressor(x, start_pos)

        # Step 16: Sparse attention against full cache (window + compressed)
        o = sparse_attn_cpu(q, self.kv_cache[:bsz], self.attn_sink, topk_idxs, self.softmax_scale)
        return o


def _init_block_weights(model, seed: int = 42, weight_gain: float = 1.0,
                        norm_init: float = 1.0, sink_scale: float = 1.0):
    """Deterministic init.

    2-D weights: xavier_uniform(gain=weight_gain). 1-D params default to a
    small uniform, EXCEPT RMSNorm weights (initialized near `norm_init`, i.e.
    ~identity scaling) and attn_sink (scaled by `sink_scale`). With gain=0.1 and
    norm_init~=0 the output decays to ~1e-6 (bf16 noise floor); gain=1.0 +
    norm_init=1.0 gives an O(1) output where real numerical error is visible.
    """
    torch.manual_seed(seed)
    for name, param in model.named_parameters():
        if param.dim() >= 2:
            nn.init.xavier_uniform_(param, gain=weight_gain)
        elif param.dim() == 1:
            if name.endswith("attn_sink"):
                nn.init.uniform_(param, -0.1 * sink_scale, 0.1 * sink_scale)
            elif "norm" in name:
                # RMSNorm gamma near 1.0 (identity) with a small spread.
                nn.init.uniform_(param, norm_init - 0.05, norm_init + 0.05)
            else:
                nn.init.uniform_(param, -0.1, 0.1)


# --------------------------------------------------------------------------
# Full CSA Attention Block (projections + core) -- Prefill
# --------------------------------------------------------------------------
class CSAAttentionBlockPrefill(nn.Module):
    """Complete CSA attention block (compress_ratio=4), prefill phase.

    Owns the projection weights (wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b)
    plus a CSAAttentionCore (attn_sink, Compressor, Indexer, sparse attention,
    KV caches). Runs the whole sequence through the prefill path in one forward.

    The output projection (wo_a, wo_b) is sharded for `tp_size`-way tensor
    parallelism; this module holds ONLY rank `tp_rank`'s shard and returns that
    rank's partial [B, S, dim] contribution (the all-reduce over ranks is the
    caller's responsibility). Structurally identical to the decode block's
    projection wiring -- only the phase (start_pos=0, full sequence) differs.
    """

    def __init__(self, config: CSAConfig, tp_size: int = 4, tp_rank: int = 0):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.n_groups = config.o_groups
        self.eps = config.norm_eps

        assert self.n_groups % tp_size == 0, "o_groups must be divisible by tp_size"
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.n_local_groups = self.n_groups // tp_size          # groups owned by this rank
        self.group_in = self.n_heads * self.head_dim // self.n_groups   # per-group wo_a input width

        # All projection weights bf16 -- the core consumes bf16 x/qr, and the NKI
        # block runs bf16 matmuls under --auto-cast=none (DeepSeek-V4's bf16
        # default). Keeps the CPU reference and the NKI kernel on the same numeric
        # footing (diff is hardware matmul accumulation only).
        pdt = torch.bfloat16

        # ----- Query projection -----
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=pdt)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=pdt)

        # ----- KV projection -----
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False, dtype=pdt)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # ----- Output projection (grouped low-rank), rank-local shard -----
        # Full (world_size=1) wo_a: [n_groups * o_lora_rank, group_in]
        #                    wo_b: [dim, n_groups * o_lora_rank]
        # Rank r owns groups [r*n_local_groups : (r+1)*n_local_groups]:
        #   wo_a shard rows = those groups' o_lora_rank outputs -> [n_local_groups*o_lora_rank, group_in]
        #   wo_b shard cols = those groups' flattened inputs     -> [dim, n_local_groups*o_lora_rank]
        self.wo_a = nn.Linear(self.group_in, self.n_local_groups * self.o_lora_rank, bias=False, dtype=pdt)
        self.wo_b = nn.Linear(self.n_local_groups * self.o_lora_rank, self.dim, bias=False, dtype=pdt)

        # ----- Core sparse attention (owns caches, attn_sink, compressor, indexer) -----
        self.core = CSAAttentionCore(config)

        # RoPE frequencies (shared with core; used here for q/kv RoPE).
        self.register_buffer("freqs_cis", self.core.freqs_cis, persistent=False)

    # ---- shared projection helpers -----------------------------------------
    def _project_q(self, x, freqs_cis):
        rd = self.rope_head_dim
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        return q, qr

    def _project_kv(self, x, freqs_cis):
        rd = self.rope_head_dim
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        return kv

    def _output_projection(self, o, bsz, seqlen):
        """Rank-local grouped low-rank output projection.

        o: [B, S, n_heads, head_dim] (post de-RoPE). This rank only consumes the
        n_local_groups groups it owns.
        """
        o = o.reshape(bsz, seqlen, self.n_groups, self.group_in)
        g0 = self.tp_rank * self.n_local_groups
        o_local = o[:, :, g0:g0 + self.n_local_groups, :]        # [B, S, n_local_groups, group_in]

        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, self.group_in)
        o_local = torch.einsum("bsgd,grd->bsgr", o_local, wo_a)  # [B, S, n_local_groups, o_lora_rank]
        out_partial = self.wo_b(o_local.flatten(2))              # [B, S, dim] -- rank partial

        # NOTE(tensor-parallel): out_partial is this rank's contribution only.
        # The final output is sum over ranks: dist.all_reduce(out_partial).
        return out_partial

    # ---- prefill forward (whole sequence) ----------------------------------
    @torch.no_grad()
    def forward(self, x, start_pos: int = 0):
        bsz, seqlen, _ = x.size()
        rd = self.rope_head_dim
        freqs_cis = self.freqs_cis[start_pos:start_pos + seqlen]

        q, qr = self._project_q(x, freqs_cis)
        kv = self._project_kv(x, freqs_cis)

        # Core sparse attention (prefill path when start_pos == 0: window indices,
        # KV compression, indexer top-k, sparse attention matmul).
        o = self.core(q, kv, x, qr, start_pos)          # [B, S, n_heads, head_dim]

        # De-rotate RoPE on the output's rope channels.
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        return self._output_projection(o, bsz, seqlen)   # [B, S, dim] rank partial


# --------------------------------------------------------------------------
# Tensor-parallel reference-data generation (head-parallel over `tp_size` ranks)
# --------------------------------------------------------------------------
def generate_prefill_block_reference_tp(full_config, tp_size: int = 4,
                                weight_gain: float = 0.2, norm_init: float = 1.0,
                                sink_scale: float = 1.0, input_scale: float = 1.0):
    """Head-parallel TP decomposition of the FULL prefill model over `tp_size` ranks.

    The production model has full_config.n_heads (=128) query heads and
    full_config.o_groups (=16) output-projection groups. Sharded HEAD-PARALLEL:
    rank r owns query heads [r*Hl : (r+1)*Hl] (Hl = n_heads/tp_size = 32) which,
    since a group tiles n_heads/o_groups (=8) heads, coincide EXACTLY with output
    groups [r*Gl : (r+1)*Gl] (Gl = o_groups/tp_size = 4). Each rank is a
    self-contained Hl-head / Gl-group block with its OWN weights -- no redundant
    compute.

    Builds ONE full 128-head model, runs the prefill forward to the true full
    output [B,S,dim] (the all-reduce target), then for each rank derives (a) its
    sharded weight state_dict for an Hl-head/Gl-group block and (b) its golden
    output-projection PARTIAL. Because prefill attention is per-head (the
    indexer's top-k is head-independent and the KV is shared), heads
    [r*Hl:(r+1)*Hl] of the full model equal an independent Hl-head block's
    output; and because the grouped projection sums over disjoint group blocks,
    sum_r partial_r == full output.

    Returns the shared prefill input `x`, ref_output_full, per-rank partials,
    and per-rank shard state_dicts.
    """
    full = CSAAttentionBlockPrefill(full_config, tp_size=1, tp_rank=0)
    _init_block_weights(full, weight_gain=weight_gain, norm_init=norm_init,
                        sink_scale=sink_scale)
    full.eval()

    B, S = full_config.batch_size, full_config.seq_len
    Hl = full_config.n_heads // tp_size          # 32 query heads per rank
    Gl = full_config.o_groups // tp_size         # 4 output groups per rank
    hd = full_config.head_dim
    R = full_config.o_lora_rank
    group_in = full_config.n_heads * hd // full_config.o_groups   # 4096 (full)

    torch.manual_seed(99)
    x = (torch.randn(B, S, full_config.dim) * input_scale).to(torch.bfloat16)
    rd = full_config.rope_head_dim
    with torch.no_grad():
        # Full prefill up to the per-head attention output o (post de-RoPE), then
        # the full grouped output projection = the golden all-reduce target.
        freqs_cis = full.freqs_cis[0:S]
        q, qr = full._project_q(x, freqs_cis)
        kv = full._project_kv(x, freqs_cis)
        o_full = full.core(q, kv, x, qr, start_pos=0)          # [B,S,128,head_dim]
        apply_rotary_emb(o_full[..., -rd:], freqs_cis, inverse=True)
        ref_output_full = full._output_projection(o_full.clone(), B, S)  # [B,S,dim]

    full_sd = {
        k: v for k, v in full.state_dict().items()
        if not k.startswith("core.kv_cache")
        and not k.startswith("core.freqs_cis")
        and not k.startswith("freqs_cis")
        and not k.startswith("core.indexer.kv_cache")
    }

    per_rank_weights, ref_partials = [], []
    for r in range(tp_size):
        h0, h1 = r * Hl, (r + 1) * Hl                # this rank's query heads
        g0r, g1r = r * Gl * R, (r + 1) * Gl * R      # this rank's wo_a rows / wo_b cols
        sd_r = {}
        for k, v in full_sd.items():
            if k == "wq_b.weight":                   # [n_heads*hd, q_lora] -> this rank's heads
                sd_r[k] = v[h0 * hd:h1 * hd, :].clone()
            elif k == "core.attn_sink":              # [n_heads] -> this rank's heads
                sd_r[k] = v[h0:h1].clone()
            elif k == "wo_a.weight":                 # [o_groups*R, group_in] -> this rank's groups
                sd_r[k] = v[g0r:g1r, :].clone()
            elif k == "wo_b.weight":                 # [dim, o_groups*R] -> this rank's group cols
                sd_r[k] = v[:, g0r:g1r].clone()
            else:                                    # wq_a/q_norm/wkv/kv_norm/indexer/compressor replicated
                sd_r[k] = v.clone()
        per_rank_weights.append(sd_r)
        # Golden partial: an Hl-head/Gl-group block over this rank's head slice of
        # o_full (== what the NKI rank block computes independently).
        with torch.no_grad():
            og = o_full[:, :, h0:h1, :].reshape(B, S, Gl, group_in)        # [B,S,Gl,group_in]
            wo_a_r = sd_r["wo_a.weight"].view(Gl, R, group_in)
            lat = torch.einsum("bsgd,grd->bsgr", og, wo_a_r)               # [B,S,Gl,R]
            partial_r = torch.matmul(lat.reshape(B, S, Gl * R),
                                     sd_r["wo_b.weight"].t())              # [B,S,dim]
        ref_partials.append(partial_r)

    summed = torch.stack([p.float() for p in ref_partials], 0).sum(0)
    max_sum_err = (summed - ref_output_full.float()).abs().max().item()

    return {
        "x": x,                                # prefill input [B,S,dim] (the trace input)
        "ref_output_full": ref_output_full,    # all-reduce target [B,S,dim]
        "ref_partials": ref_partials,          # list of tp_size [B,S,dim] partials
        "per_rank_weights": per_rank_weights,   # list of tp_size shard state_dicts
        "max_sum_err": max_sum_err,
        "tp_size": tp_size,
    }


# --------------------------------------------------------------------------
# Full CSA Attention Block (projections + core) -- Decode
# --------------------------------------------------------------------------
class CSAAttentionBlockDecode(nn.Module):
    """Complete CSA attention block (compress_ratio=4), decode phase.

    Owns the projection weights (wq_a, q_norm, wq_b, wkv, kv_norm, wo_a, wo_b)
    plus a CSAAttentionCore (attn_sink, Compressor, Indexer, sparse attention,
    KV caches). Runs prefill to populate the caches, then a single decode step.

    The output projection (wo_a, wo_b) is sharded for `tp_size`-way tensor
    parallelism; this module holds ONLY rank `tp_rank`'s shard and returns that
    rank's partial [B, 1, dim] contribution (the all-reduce over ranks is the
    caller's responsibility).
    """

    def __init__(self, config: CSAConfig, tp_size: int = 4, tp_rank: int = 0):
        super().__init__()
        self.config = config
        self.dim = config.dim
        self.n_heads = config.n_heads
        self.q_lora_rank = config.q_lora_rank
        self.o_lora_rank = config.o_lora_rank
        self.head_dim = config.head_dim
        self.rope_head_dim = config.rope_head_dim
        self.n_groups = config.o_groups
        self.eps = config.norm_eps

        assert self.n_groups % tp_size == 0, "o_groups must be divisible by tp_size"
        self.tp_size = tp_size
        self.tp_rank = tp_rank
        self.n_local_groups = self.n_groups // tp_size          # groups owned by this rank
        self.group_in = self.n_heads * self.head_dim // self.n_groups   # per-group wo_a input width

        # All projection weights are bf16: the core's indexer/compressor consume
        # bf16 x/qr (their wq_b/weights_proj are bf16), the core's validated
        # contract is bf16 q/kv/x/qr, and the NKI block runs bf16 matmuls
        # (--auto-cast=none). This mirrors the original DeepSeek-V4 model's bf16
        # default dtype and keeps the CPU reference and the NKI kernel on the
        # same numeric footing (diff is hardware matmul accumulation only).
        pdt = torch.bfloat16

        # ----- Query projection -----
        self.wq_a = nn.Linear(self.dim, self.q_lora_rank, bias=False, dtype=pdt)
        self.q_norm = RMSNorm(self.q_lora_rank, self.eps)
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False, dtype=pdt)

        # ----- KV projection (for the new decode token) -----
        self.wkv = nn.Linear(self.dim, self.head_dim, bias=False, dtype=pdt)
        self.kv_norm = RMSNorm(self.head_dim, self.eps)

        # ----- Output projection (grouped low-rank), rank-local shard -----
        # Full (world_size=1) wo_a: [n_groups * o_lora_rank, group_in]
        #                    wo_b: [dim, n_groups * o_lora_rank]
        # Rank r owns groups [r*n_local_groups : (r+1)*n_local_groups]:
        #   wo_a shard rows  = those groups' o_lora_rank outputs -> [n_local_groups*o_lora_rank, group_in]
        #   wo_b shard cols  = those groups' flattened inputs     -> [dim, n_local_groups*o_lora_rank]
        self.wo_a = nn.Linear(self.group_in, self.n_local_groups * self.o_lora_rank, bias=False, dtype=pdt)
        self.wo_b = nn.Linear(self.n_local_groups * self.o_lora_rank, self.dim, bias=False, dtype=pdt)

        # ----- Core sparse attention (owns caches, attn_sink, compressor, indexer) -----
        self.core = CSAAttentionCore(config)

        # RoPE frequencies (shared with core; used here for q/kv RoPE).
        self.register_buffer("freqs_cis", self.core.freqs_cis, persistent=False)

    # ---- shared projection helpers -----------------------------------------
    def _project_q(self, x, freqs_cis):
        rd = self.rope_head_dim
        qr = self.q_norm(self.wq_a(x))
        q = self.wq_b(qr)
        q = q.unflatten(-1, (self.n_heads, self.head_dim))
        q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + self.eps)
        apply_rotary_emb(q[..., -rd:], freqs_cis)
        return q, qr

    def _project_kv(self, x, freqs_cis):
        rd = self.rope_head_dim
        kv = self.wkv(x)
        kv = self.kv_norm(kv)
        apply_rotary_emb(kv[..., -rd:], freqs_cis)
        return kv

    def _output_projection(self, o, bsz, seqlen):
        """Rank-local grouped low-rank output projection.

        o: [B, S, n_heads, head_dim] (post de-RoPE). This rank only consumes the
        n_local_groups groups it owns.
        """
        # Flatten heads then view as groups; keep only this rank's groups.
        o = o.reshape(bsz, seqlen, self.n_groups, self.group_in)
        g0 = self.tp_rank * self.n_local_groups
        o_local = o[:, :, g0:g0 + self.n_local_groups, :]        # [B, S, n_local_groups, group_in]

        wo_a = self.wo_a.weight.view(self.n_local_groups, self.o_lora_rank, self.group_in)
        o_local = torch.einsum("bsgd,grd->bsgr", o_local, wo_a)  # [B, S, n_local_groups, o_lora_rank]
        out_partial = self.wo_b(o_local.flatten(2))              # [B, S, dim] -- rank partial

        # NOTE(tensor-parallel): out_partial is this rank's contribution only.
        # The final output is sum over ranks: dist.all_reduce(out_partial). We
        # return the partial and leave the collective to the caller.
        return out_partial

    # ---- prefill (populate caches) -----------------------------------------
    @torch.no_grad()
    def prefill(self, x):
        bsz, seqlen, _ = x.size()
        freqs_cis = self.freqs_cis[0:seqlen]
        q, qr = self._project_q(x, freqs_cis)
        kv = self._project_kv(x, freqs_cis)
        # core.prefill fills self.core.kv_cache and self.core.indexer.kv_cache.
        self.core.prefill(q, kv, x, qr)

    # ---- decode (single new token) -----------------------------------------
    @torch.no_grad()
    def forward(self, x, start_pos):
        bsz, seqlen, _ = x.size()
        assert seqlen == 1, f"Decode expects seqlen=1, got {seqlen}"
        rd = self.rope_head_dim
        freqs_cis = self.freqs_cis[start_pos:start_pos + 1]

        q, qr = self._project_q(x, freqs_cis)
        kv = self._project_kv(x, freqs_cis)

        # Core sparse attention (inserts the new kv into the window cache and
        # attends over window + top-k compressed positions).
        o = self.core(q, kv, x, qr, start_pos)          # [B, 1, n_heads, head_dim]

        # De-rotate RoPE on the output's rope channels.
        apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)

        return self._output_projection(o, bsz, seqlen)   # [B, 1, dim] rank partial


# --------------------------------------------------------------------------
# Reference data generation (prefill -> extract caches -> decode)
# --------------------------------------------------------------------------
def generate_decode_block_reference(config, tp_size: int = 4, tp_rank: int = 0,
                             weight_gain: float = 0.46, norm_init: float = 1.0,
                             sink_scale: float = 1.0, input_scale: float = 1.0):
    """Build the block, run prefill, extract caches, run one decode step.

    Returns a dict with the raw decode input `x_dec`, the post-prefill KV caches
    (window / compressed / indexer) that the NKI block consumes, the rank-`tp_rank`
    reference output, and the block weights for loading into the NKI module.

    The magnitude knobs control how large the reference output is (useful for
    exposing numerical error — a tiny gain=0.1 init decays the output to ~1e-6,
    where bf16 rounding dominates the relative error):
      weight_gain: xavier gain for 2-D projection weights. Default 0.46 gives an
                   O(1e-2) output whose max_abs_diff (~1.1e-3) sits just below
                   the 2e-3 tolerance; gain=1.0 is standard-xavier (max_abs ~5e-3).
      norm_init:   RMSNorm weights initialized near this value (1.0 = identity).
      sink_scale:  attn_sink magnitude.
      input_scale: std of the random hidden-state inputs (prefill + decode). Note
                   the block is input-scale-invariant (RMSNorm after wq_a/wkv).
    """
    block = CSAAttentionBlockDecode(config, tp_size=tp_size, tp_rank=tp_rank)
    _init_block_weights(block, weight_gain=weight_gain, norm_init=norm_init,
                        sink_scale=sink_scale)
    block.eval()

    B, S = config.batch_size, config.seq_len
    W = config.window_size
    T_c = S // config.compress_ratio

    # Prefill from raw hidden states.
    torch.manual_seed(99)
    x_prefill = (torch.randn(B, S, config.dim) * input_scale).to(torch.bfloat16)
    with torch.no_grad():
        block.prefill(x_prefill)

    # Extract caches after prefill (pre-decode; the NKI block inserts the new
    # token into the window itself, matching core.forward).
    kv_window = block.core.kv_cache[:B, :W].clone()
    kv_compress = block.core.kv_cache[:B, W:W + T_c].clone()
    indexer_kv_cache = block.core.indexer.kv_cache[:B, :T_c].clone()

    # Decode from a raw hidden state.
    torch.manual_seed(200)
    x_dec = (torch.randn(B, 1, config.dim) * input_scale).to(torch.bfloat16)
    with torch.no_grad():
        ref_output = block(x_dec, start_pos=S)

    # Weights to load into the NKI block (skip caches + freqs buffers).
    ref_weights = {
        k: v for k, v in block.state_dict().items()
        if not k.startswith("core.kv_cache")
        and not k.startswith("core.freqs_cis")
        and not k.startswith("freqs_cis")
        and not k.startswith("core.indexer.kv_cache")
    }

    return {
        "x_dec": x_dec,
        "kv_window": kv_window,
        "kv_compress": kv_compress,
        "indexer_kv_cache": indexer_kv_cache,
        "ref_output": ref_output,
        "ref_weights": ref_weights,
        "tp_size": tp_size,
        "tp_rank": tp_rank,
    }


def generate_decode_block_reference_tp(full_config, tp_size: int = 4,
                                weight_gain: float = 0.46, norm_init: float = 1.0,
                                sink_scale: float = 1.0, input_scale: float = 1.0):
    """Head-parallel tensor-parallel decomposition of the FULL model over `tp_size` ranks.

    The production model has full_config.n_heads (=128) query heads and
    full_config.o_groups (=16) output-projection groups. We shard it HEAD-PARALLEL:
    rank r owns query heads [r*Hl : (r+1)*Hl] (Hl = n_heads/tp_size = 32) which,
    since a group tiles n_heads/o_groups (=8) heads, coincide EXACTLY with output
    groups [r*Gl : (r+1)*Gl] (Gl = o_groups/tp_size = 4). So each rank is a
    self-contained Hl-head / Gl-group block with its OWN weights, computing a
    DIFFERENT slice of the attention — no redundant compute.

    Builds ONE full model, runs prefill+decode to the true full output (the
    all-reduce target), then for each rank derives (a) its sharded weight
    state_dict for an Hl-head/Gl-group block and (b) its golden output-projection
    PARTIAL. Because decode attention is per-head (the indexer's top-k is
    head-independent and the KV is shared), heads [r*Hl:(r+1)*Hl] of the full
    model equal an independent Hl-head block's output; and because the grouped
    projection sums over disjoint group blocks, sum_r partial_r == full output.

    Returns shared decode inputs + caches (head-independent, shared by all ranks),
    ref_output_full, the per-rank partials, and the per-rank shard state_dicts.
    """
    full = CSAAttentionBlockDecode(full_config, tp_size=1, tp_rank=0)
    _init_block_weights(full, weight_gain=weight_gain, norm_init=norm_init,
                        sink_scale=sink_scale)
    full.eval()

    B, S = full_config.batch_size, full_config.seq_len
    W = full_config.window_size
    T_c = S // full_config.compress_ratio
    Hl = full_config.n_heads // tp_size          # 32 query heads per rank
    Gl = full_config.o_groups // tp_size         # 4 output groups per rank
    hd = full_config.head_dim
    R = full_config.o_lora_rank

    torch.manual_seed(99)
    x_prefill = (torch.randn(B, S, full_config.dim) * input_scale).to(torch.bfloat16)
    with torch.no_grad():
        full.prefill(x_prefill)

    # Head-independent caches, shared by every rank.
    kv_window = full.core.kv_cache[:B, :W].clone()
    kv_compress = full.core.kv_cache[:B, W:W + T_c].clone()
    indexer_kv_cache = full.core.indexer.kv_cache[:B, :T_c].clone()

    torch.manual_seed(200)
    x_dec = (torch.randn(B, 1, full_config.dim) * input_scale).to(torch.bfloat16)
    rd = full_config.rope_head_dim
    with torch.no_grad():
        # Full decode up to the per-head attention output o (post de-RoPE), then
        # the full grouped output projection = the golden all-reduce target.
        freqs_cis = full.freqs_cis[S:S + 1]
        q, qr = full._project_q(x_dec, freqs_cis)
        kv = full._project_kv(x_dec, freqs_cis)
        o_full = full.core(q, kv, x_dec, qr, start_pos=S)      # [B,1,128,head_dim]
        apply_rotary_emb(o_full[..., -rd:], freqs_cis, inverse=True)
        ref_output_full = full._output_projection(o_full.clone(), B, 1)  # [B,1,dim]

    full_sd = {
        k: v for k, v in full.state_dict().items()
        if not k.startswith("core.kv_cache")
        and not k.startswith("core.freqs_cis")
        and not k.startswith("freqs_cis")
        and not k.startswith("core.indexer.kv_cache")
    }

    per_rank_weights, ref_partials = [], []
    for r in range(tp_size):
        h0, h1 = r * Hl, (r + 1) * Hl                # this rank's query heads
        g0r, g1r = r * Gl * R, (r + 1) * Gl * R      # this rank's wo_a rows / wo_b cols
        sd_r = {}
        for k, v in full_sd.items():
            if k == "wq_b.weight":                   # [n_heads*hd, q_lora] -> this rank's heads
                sd_r[k] = v[h0 * hd:h1 * hd, :].clone()
            elif k == "core.attn_sink":              # [n_heads] -> this rank's heads
                sd_r[k] = v[h0:h1].clone()
            elif k == "wo_a.weight":                 # [o_groups*R, group_in] -> this rank's groups
                sd_r[k] = v[g0r:g1r, :].clone()
            elif k == "wo_b.weight":                 # [dim, o_groups*R] -> this rank's group cols
                sd_r[k] = v[:, g0r:g1r].clone()
            else:                                    # wq_a/q_norm/wkv/kv_norm/indexer/compressor replicated
                sd_r[k] = v.clone()
        per_rank_weights.append(sd_r)
        # Golden partial: an Hl-head/Gl-group block over this rank's head slice of
        # o_full (== what the NKI rank block computes independently).
        with torch.no_grad():
            og = o_full[:, :, h0:h1, :].reshape(B, 1, Gl, Hl * hd // Gl)   # [B,1,Gl,group_in]
            wo_a_r = sd_r["wo_a.weight"].view(Gl, R, Hl * hd // Gl)
            lat = torch.einsum("bsgd,grd->bsgr", og, wo_a_r)               # [B,1,Gl,R]
            partial_r = torch.matmul(lat.reshape(B, 1, Gl * R),
                                     sd_r["wo_b.weight"].t())              # [B,1,dim]
        ref_partials.append(partial_r)

    summed = torch.stack([p.float() for p in ref_partials], 0).sum(0)
    max_sum_err = (summed - ref_output_full.float()).abs().max().item()

    return {
        "x_dec": x_dec,
        "kv_window": kv_window,
        "kv_compress": kv_compress,
        "indexer_kv_cache": indexer_kv_cache,
        "ref_output_full": ref_output_full,   # all-reduce target [B,1,dim]
        "ref_partials": ref_partials,         # list of tp_size [B,1,dim] partials
        "per_rank_weights": per_rank_weights,  # list of tp_size shard state_dicts
        "max_sum_err": max_sum_err,
        "tp_size": tp_size,
    }


