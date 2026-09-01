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

"""Configuration and host-side helpers shared across the DeepSeek-V4 CSA kernels.

``CSAConfigFull`` is the production configuration: 128 query heads and 16 output
projection groups. ``CSAConfig`` is the per-rank shard that the kernels actually
see under 4-way head-parallel tensor parallelism -- 32 query heads and 4 output
groups -- and is what ``shard_for_tp`` produces.

The indexer fields (``index_*``) describe the lightning indexer that scores every
compressed position; ``index_topk`` is how many of those positions the sparse
attention gathers, so the attention body's cost is O(window_size + index_topk)
and does not grow with the context length.

The helpers below are plain torch, not NKI: the RoPE tables and window-bias masks
are built once on the host and handed to the kernels as inputs, and ``RMSNorm`` /
``hadamard_transform`` are used by the block composition and by the CPU
references. Keeping them here is what lets the kernels, the blocks and the
references agree bit-for-bit on the tables they consume.
"""

import math
from dataclasses import dataclass, replace

import torch
from torch import nn


@dataclass
class CSAConfig:
    """One tensor-parallel rank's view of the DeepSeek-V4 CSA attention block.

    ``n_heads`` and ``o_groups`` are the RANK-LOCAL counts. The compressed cache
    holds ``seq_len // compress_ratio`` positions (``T_c``), which is what the
    indexer scores and what the top-k selects from.
    """

    dim: int = 7168
    """Model hidden size."""

    n_heads: int = 32
    """Rank-local query heads (128 in the full model, over 4 ranks)."""

    head_dim: int = 512
    """Attention head dimension."""

    rope_head_dim: int = 64
    """Rotated channels of each head; the leading ``head_dim - rope_head_dim`` pass through."""

    q_lora_rank: int = 1536
    """Query latent rank, shared between the attention q-path and the indexer."""

    o_groups: int = 16
    """Rank-local output projection groups (16 in the full model, over 4 ranks)."""

    o_lora_rank: int = 1024
    """Output projection latent rank -- the low-rank bottleneck between wo_a and wo_b."""

    window_size: int = 128
    """Sliding window positions attended in full, on top of the selected compressed ones."""

    compress_ratio: int = 4
    """Raw tokens folded into one compressed cache position."""

    norm_eps: float = 1e-6
    """RMSNorm epsilon."""

    index_n_heads: int = 64
    """Lightning indexer query heads."""

    index_head_dim: int = 128
    """Lightning indexer head dimension."""

    index_topk: int = 1024
    """Compressed positions the sparse attention selects (``k``)."""

    compress_rope_theta: float = 160000.0
    """RoPE base."""

    original_seq_len: int = 65536
    """RoPE reference length; 0 disables the NTK correction."""

    rope_factor: float = 16.0
    """RoPE scaling factor."""

    beta_fast: int = 32
    """High end of the RoPE correction range."""

    beta_slow: int = 1
    """Low end of the RoPE correction range."""

    batch_size: int = 1
    """Batch size."""

    seq_len: int = 8192
    """Context length. The compressed cache holds ``seq_len // compress_ratio`` positions."""

    @property
    def compressed_len(self) -> int:
        """``T_c`` -- compressed positions in the cache."""
        return self.seq_len // self.compress_ratio

    @property
    def group_in(self) -> int:
        """Per-group input width of the output projection's wo_a."""
        return self.n_heads * self.head_dim // self.o_groups


@dataclass
class CSAConfigFull(CSAConfig):
    """The production DeepSeek-V4-Pro-Max shape: 128 query heads, 16 output groups.

    A whole chip holds this configuration as ``tp_size`` head-parallel ranks; each
    rank runs ``shard_for_tp(config, tp_size)`` on 2 logical NeuronCores.
    """

    n_heads: int = 128
    o_groups: int = 16


def shard_for_tp(config: CSAConfig, tp_size: int) -> CSAConfig:
    """Return one head-parallel rank's config: ``n_heads`` and ``o_groups`` divided by ``tp_size``.

    Head-parallel sharding leaves ``group_in`` unchanged, because both the head
    count and the group count divide by the same factor. Every other field is
    replicated, since the indexer, the compressor and the sliding window are
    shared: each rank scores all ``T_c`` compressed positions and selects the same
    ``index_topk`` of them.
    """
    if config.n_heads % tp_size != 0:
        raise ValueError(f"n_heads={config.n_heads} must be divisible by tp_size={tp_size}")
    if config.o_groups % tp_size != 0:
        raise ValueError(f"o_groups={config.o_groups} must be divisible by tp_size={tp_size}")
    return replace(config, n_heads=config.n_heads // tp_size, o_groups=config.o_groups // tp_size)


def precompute_freqs_cos_sin(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """Build the RoPE rotation tables as real ``(cos, sin)`` of shape ``[seqlen, dim // 2]``.

    Real tables rather than complex ones, because the kernels rotate with
    multiplies and adds on the Vector engine and there is no complex dtype on
    device. ``original_seq_len > 0`` enables the YaRN-style NTK correction: the
    low-frequency channels are divided by ``factor`` and the high-frequency ones
    are left alone, with ``beta_fast``/``beta_slow`` setting the ramp between.
    """

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
    return torch.cos(freqs), torch.sin(freqs)


def apply_rotary_emb_functional(x_rope, freqs_cos_sin, inverse=False):
    """Functional RoPE using real cos/sin. Computes in fp32, returns input dtype."""
    dtype = x_rope.dtype
    cos_f, sin_f = freqs_cos_sin

    x_pairs = x_rope.float().unflatten(-1, (-1, 2))
    x1 = x_pairs[..., 0]
    x2 = x_pairs[..., 1]

    if inverse:
        sin_f = -sin_f

    if x1.ndim == 3:
        cos_f = cos_f.unsqueeze(0)
        sin_f = sin_f.unsqueeze(0)
    else:
        cos_f = cos_f.unsqueeze(0).unsqueeze(2)
        sin_f = sin_f.unsqueeze(0).unsqueeze(2)

    y1 = x1 * cos_f - x2 * sin_f
    y2 = x1 * sin_f + x2 * cos_f

    return torch.stack([y1, y2], dim=-1).flatten(-2).to(dtype)


class RMSNorm(nn.Module):
    """RMSNorm with a learnable fp32 gain, normalizing in fp32 and casting back on the way out.

    The fp32 accumulation and the cast at the output boundary are load-bearing:
    the fused kernels reproduce exactly this dtype flow, so a kernel result can be
    compared against this module bit-for-bit.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim, dtype=torch.float32))

    def forward(self, x):
        dtype = x.dtype
        x = x.float()
        var = x.square().mean(-1, keepdim=True)
        return (self.weight * (x * torch.rsqrt(var + self.eps))).to(dtype)


_hadamard_cache: dict = {}


def get_hadamard_matrix(n, device, dtype):
    """Normalized ``[n, n]`` Sylvester Hadamard matrix, cached per (n, device, dtype).

    ``n`` must be a power of two. Scaled by ``n ** -0.5`` so the transform is
    orthonormal and does not change the magnitude of what it rotates.
    """
    key = (n, device, dtype)
    if key not in _hadamard_cache:
        H = torch.tensor([[1.0]])
        while H.shape[0] < n:
            H = torch.cat([
                torch.cat([H, H], dim=1),
                torch.cat([H, -H], dim=1),
            ], dim=0)
        _hadamard_cache[key] = (H * (n ** -0.5)).to(dtype=dtype, device=device)
    return _hadamard_cache[key]


def hadamard_transform(x):
    """Rotate the last axis of ``x`` by the orthonormal Hadamard matrix, as one matmul.

    The indexer applies this to its query so the scored channels are decorrelated;
    being orthonormal it leaves the dot products the indexer takes unchanged in
    aggregate while spreading each channel's contribution.
    """
    n = x.shape[-1]
    H = get_hadamard_matrix(n, x.device, x.dtype)
    return x @ H


def precompute_win_bias_parts(S, W):
    """Static sliding-window bias parts for prefill, both ``[S, 2 * W]``.

    Returns ``(base, sink_indicator)``:

    * ``base`` is ``0`` at window positions a query may attend and ``-1e9``
      elsewhere, so adding it before ``exp`` masks the rest out.
    * ``sink_indicator`` is ``1.0`` at each query's attention-sink slot and ``0.0``
      elsewhere, so the caller scales it by the per-head sink scalar and adds it.

    Splitting the bias this way keeps both halves independent of the sink weights,
    which are learned -- so these two tables are constant for a given ``(S, W)``
    and are built once at module construction rather than per call.
    """
    TILE_Q = W
    WIN_SIZE = 2 * W

    q_pos = torch.arange(S)
    i_idx = (q_pos % TILE_Q).unsqueeze(1)
    q_start = ((q_pos // TILE_Q) * TILE_Q).unsqueeze(1)
    j_idx = torch.arange(WIN_SIZE).unsqueeze(0)

    valid = (j_idx >= i_idx + 1) & (j_idx <= i_idx + W) & (j_idx >= (W - q_start))
    sink_j = W - q_start
    is_sink = (j_idx == sink_j) & valid

    base = torch.where(valid, torch.zeros(1), torch.tensor(-1e9))
    sink_indicator = is_sink.float()
    return base, sink_indicator

