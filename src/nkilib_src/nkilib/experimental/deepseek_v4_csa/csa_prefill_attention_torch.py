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

"""CPU references for the DeepSeek-V4 CSA prefill kernels.

One reference per tested kernel, taking the SAME parameter names as its kernel so
the test framework can pair them.

The one structural thing to know: the sliding window is tiled by QUERY TILE, not
by query row. A tile of 128 queries starting at ``q_start`` all read the same 256
window key columns ``[q_start, q_start + 256)``, and which of those a given row may
actually attend to is decided by ``win_bias_base`` rather than by the slice. So the
references loop over 128-row tiles exactly as the kernels do -- a reference that
sliced per row would disagree with a correct kernel.

The window key/value tensors arrive left-padded by ``W``, so window column ``j`` of
tile ``q_start`` is sequence position ``q_start + j - W``. That padding is what lets
the first tile use the same slice arithmetic as every other one.
"""

import torch
import torch.nn.functional as F

_TILE_Q = 128
_W = 128
_WIN_SIZE = 2 * _TILE_Q
_NEG_INF = -1e9


def _rope_pairs(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, inverse: bool) -> torch.Tensor:
    """Rotate interleaved (even, odd) channel pairs of ``x`` by ``cos``/``sin``, in fp32.

    ``inverse`` negates ``sin``, which turns the forward rotation into its inverse --
    the form the output de-RoPE uses.
    """
    half_rope = cos.shape[-1]
    pairs = x.float().unflatten(-1, (half_rope, 2))
    x1, x2 = pairs[..., 0], pairs[..., 1]
    s = -sin if inverse else sin
    y1 = x1 * cos - x2 * s
    y2 = x1 * s + x2 * cos
    return torch.stack([y1, y2], dim=-1).flatten(-2)


def nki_rms_rope_torch_ref(
    x_in: torch.Tensor,
    cos_in: torch.Tensor,
    sin_in: torch.Tensor,
    gain_in,
    eps_val: float,
    do_rms: int = 1,
    inverse: int = 0,
) -> dict[str, torch.Tensor]:
    """Oracle for ``nki_rms_rope_kernel``: RMSNorm(+gain) then RoPE over ``[S_rows, head_dim]``.

    Three call sites share this kernel, and the flags are what pick between them:
    the q-path (``gain_in=None``), the kv-path (``gain_in=kv_norm.weight``) and the
    output de-RoPE (``do_rms=0, inverse=1``). ``cos_in``/``sin_in`` are per-ROW, so
    the caller has already gathered the right angle for each (head, position) row.

    The bf16 round after the norm and before the rotation is deliberate: the model
    casts at its RMSNorm output boundary, and the kernel reproduces that, so the
    reference has to as well or it would be systematically more accurate than what
    it grades.
    """
    half_rope = cos_in.shape[1]
    rope_dim = 2 * half_rope
    nope_dim = x_in.shape[1] - rope_dim

    x = x_in.float()
    if do_rms:
        x = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps_val)
        if gain_in is not None:
            x = x * gain_in.float()
    normed = x.to(torch.bfloat16)

    rotated = _rope_pairs(normed[:, nope_dim:], cos_in.float(), sin_in.float(), bool(inverse))
    out = torch.cat([normed[:, :nope_dim], rotated.to(torch.bfloat16)], dim=-1)
    return {"output_0": out}


def nki_compressor_core_torch_ref(
    kv8: torch.Tensor,
    score8: torch.Tensor,
    norm_weight: torch.Tensor,
    cos_rep: torch.Tensor,
    sin_rep: torch.Tensor,
    eps: float,
) -> dict[str, torch.Tensor]:
    """Oracle for ``nki_compressor_core_kernel``: gated pooling, then RMSNorm, then RoPE.

    The gate is a softmax over the ``ratio2`` overlapped slots taken INDEPENDENTLY
    PER CHANNEL -- not per position -- so each channel of a compressed position
    pools its ``ratio2`` candidates with its own weights. That per-channel
    independence is why the kernel can keep positions on partitions and channels on
    the free axis and never reduce across cores.

    ``cos_rep``/``sin_rep`` arrive with each pair's angle DUPLICATED across the two
    channels of the pair, so the kernel can load them with the same access pattern
    as the data; only the even entries are read, which is what this mirrors.
    """
    rope_dim = cos_rep.shape[1]
    head_dim = kv8.shape[2]
    nope_dim = head_dim - rope_dim

    weights = torch.softmax(score8.float(), dim=1)
    pooled = (kv8.float() * weights).sum(dim=1)

    # bf16 at the pooling output, matching the model casting before its RMSNorm.
    x = pooled.to(torch.bfloat16).float()
    normed = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + eps) * norm_weight.float()
    normed = normed.to(torch.bfloat16)

    # Each pair's angle is duplicated across its two channels; take one per pair.
    cos = cos_rep.float()[:, 0::2]
    sin = sin_rep.float()[:, 0::2]
    rotated = _rope_pairs(normed[:, nope_dim:], cos, sin, inverse=False)
    out = torch.cat([normed[:, :nope_dim], rotated.to(torch.bfloat16)], dim=-1)
    return {"output_0": out}


def _indexer_scores(q_T_all: torch.Tensor, kv_t: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``score[s, t] = sum_h relu(q[s, h, :] . kv[t, :]) * weights[s, h]``, as ``[S_q, T_c]``.

    The per-head sum is accumulated in **bf16**, one head at a time, because that is
    what the kernel does: its running total lives in a bf16 SBUF tile and each head's
    contribution is folded in with a single `scalar_tensor_tensor`. Accumulating in
    fp32 here instead would make the reference systematically more precise, which
    matters more than usual downstream -- ``nki_indexer_score_mask_kernel`` turns this
    score into a THRESHOLDED mask, so a score difference of one bf16 ulp near the
    threshold flips a position and shows up as a full ``1e9`` mask error rather than
    as a small numeric one.
    """
    head_dim, total_q_free = q_T_all.shape
    n_heads = weights.shape[1]
    s_q = total_q_free // n_heads

    q = q_T_all.float().reshape(head_dim, n_heads, s_q).permute(2, 1, 0)
    per_head = F.relu(torch.einsum("shd,dt->sht", q, kv_t.float())).to(torch.bfloat16)

    acc = torch.zeros((s_q, kv_t.shape[1]), dtype=torch.bfloat16)
    for h in range(n_heads):
        acc = (per_head[:, h] * weights[:, h : h + 1].float()).to(torch.bfloat16).add(acc).to(torch.bfloat16)
    return acc.float()


def nki_indexer_score_mask_torch_ref(
    q_T_all: torch.Tensor,
    kv_t: torch.Tensor,
    weights: torch.Tensor,
    causal_bias: torch.Tensor,
    k: int,
) -> dict[str, torch.Tensor]:
    """Oracle for ``nki_indexer_score_mask_kernel``: the ``0 / -1e9`` top-k selection mask.

    The threshold comes from 9 rounds of BISECTION on the score range, not from a
    sort, so this reference runs the same bisection rather than calling
    ``torch.topk``. That is the honest comparison: bisection to a fixed depth does
    not always select exactly ``k`` positions (ties, and a residual interval), and a
    ``topk``-based reference would report those as kernel errors.

    ``lo`` starts at the smallest score that is not causally masked (found by
    folding the masked entries up to ``hi``), ``hi`` at the largest. Each round keeps
    the half that still holds at least ``k`` positions, and the final mask admits
    every score at or above ``lo``.
    """
    scores = _indexer_scores(q_T_all, kv_t, weights) + causal_bias.float()

    hi = scores.max(dim=-1, keepdim=True).values
    # Fold causally-masked entries (score <= -1e8) up to `hi` so they cannot become
    # the minimum, then take the min over what is left.
    is_valid = (scores > -1e8).float()
    lo = ((scores - hi) * is_valid + hi).min(dim=-1, keepdim=True).values

    for _ in range(9):
        mid = (lo + hi) * 0.5
        enough = ((scores >= mid).float().sum(dim=-1, keepdim=True) >= float(k)).float()
        lo = lo + (mid - lo) * enough
        hi = hi + (mid - hi) * (1.0 - enough)

    sel = (scores >= lo).float() * (-_NEG_INF) + _NEG_INF
    return {"output_0": sel}


def _tiled_sparse_attention(
    all_q_T: torch.Tensor,
    win_K_T: torch.Tensor,
    win_V: torch.Tensor,
    comp_K_T: torch.Tensor,
    comp_V: torch.Tensor,
    comp_sel: torch.Tensor,
    win_bias_base: torch.Tensor,
    win_bias_sink: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    """Global-max softmax over [window | selected compressed], tiled by query tile.

    Both prefill attention kernels compute exactly this. The window and the
    compressed positions share ONE softmax normalization (a single global max over
    the two score sets), which is what makes the two contributions directly
    comparable and removes any need for online rescaling.

    Masking is additive and happens before ``exp``: ``win_bias_base`` is ``-1e9``
    outside a row's causal window and ``comp_sel`` is ``-1e9`` at unselected
    compressed positions, so those terms underflow to exactly zero and contribute
    nothing to either the denominator or the value sum. That is also why a
    reference over ALL ``T_c`` columns matches a kernel that truncates the
    compressed loop at its causal bound.

    Returns ``[n_heads * S, head_dim]``, head-major as the kernels write it.
    """
    head_dim, total_q_free = all_q_T.shape
    s_len = comp_sel.shape[0]
    n_heads = total_q_free // s_len
    t_c = comp_sel.shape[1]

    # all_q_T[d, h * S + s] == q[s, h, d]  ->  [S, n_heads, head_dim]
    q = all_q_T.float().reshape(head_dim, n_heads, s_len).permute(2, 1, 0)
    out = torch.zeros((n_heads * s_len, head_dim), dtype=torch.bfloat16)

    comp_scores_all = torch.einsum("shd,dt->sht", q, comp_K_T.float()) + comp_sel.float().unsqueeze(1)

    for q_start in range(0, s_len, _TILE_Q):
        rows = slice(q_start, q_start + _TILE_Q)
        q_tile = q[rows]  # [tile, n_heads, head_dim]

        # Every row of the tile reads the same 256 window columns; win_bias_base
        # decides which of them the row may actually attend to.
        k_win = win_K_T.float()[:, q_start : q_start + _WIN_SIZE]  # [head_dim, 256]
        v_win = win_V.float()[q_start : q_start + _WIN_SIZE]  # [256, head_dim]

        bias = win_bias_sink.float()[rows].unsqueeze(1) * attn_sink.float().reshape(
            1, n_heads, 1
        ) + win_bias_base.float()[rows].unsqueeze(1)
        win_scores = torch.einsum("shd,dj->shj", q_tile, k_win) + bias
        comp_scores = comp_scores_all[rows]

        shift = torch.maximum(
            win_scores.max(dim=-1, keepdim=True).values,
            comp_scores.max(dim=-1, keepdim=True).values,
        )
        win_exp = torch.exp(win_scores - shift)
        comp_exp = torch.exp(comp_scores - shift)
        total = win_exp.sum(-1, keepdim=True) + comp_exp.sum(-1, keepdim=True)

        tile_out = (
            torch.einsum("shj,jd->shd", win_exp, v_win) + torch.einsum("sht,td->shd", comp_exp, comp_V.float())
        ) / total

        for h in range(n_heads):
            out[h * s_len + q_start : h * s_len + q_start + q_tile.shape[0]] = tile_out[:, h].to(torch.bfloat16)

    del t_c
    return out


def nki_fused_csa_attn_torch_ref(
    compress_sel: torch.Tensor,
    all_q_T: torch.Tensor,
    all_K_T: torch.Tensor,
    all_V: torch.Tensor,
    win_bias_base_in: torch.Tensor,
    win_bias_sink_in: torch.Tensor,
    attn_sink_in: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Oracle for ``nki_fused_csa_attn_kernel``: mask-predicated attention on concatenated KV.

    This kernel takes the window and compressed keys/values in ONE concatenated
    tensor laid out ``[padded window (S + W) | compressed (T_c)]``, so the reference
    splits them back apart at ``S + W`` before attending.
    """
    s_len, t_c = compress_sel.shape
    split = s_len + _W
    out = _tiled_sparse_attention(
        all_q_T,
        all_K_T[:, 0:split],
        all_V[0:split],
        all_K_T[:, split : split + t_c],
        all_V[split : split + t_c],
        compress_sel,
        win_bias_base_in,
        win_bias_sink_in,
        attn_sink_in,
    )
    return {"output_0": out}


def nki_gather_csa_attn_torch_ref(
    topk_sel_bias: torch.Tensor,
    all_q_T: torch.Tensor,
    all_K_T_win: torch.Tensor,
    all_V_win: torch.Tensor,
    compress_kv_T: torch.Tensor,
    compress_kv: torch.Tensor,
    win_bias_base_in: torch.Tensor,
    win_bias_sink_in: torch.Tensor,
    attn_sink_in: torch.Tensor,
    split_pos: int,
    ratio: int,
) -> dict[str, torch.Tensor]:
    """Oracle for ``nki_gather_csa_attn_kernel``: the same attention with a static causal bound.

    ``split_pos`` and ``ratio`` only set how far the kernel's compressed loop runs:
    it stops at the chunk holding the tile's last causally-reachable compressed
    position. This reference deliberately attends over ALL ``T_c`` compressed
    columns and relies on ``topk_sel_bias`` being ``-1e9`` past the frontier, so if
    the kernel's bound were ever too tight -- dropping a column that mattered -- the
    two would disagree.
    """
    del split_pos, ratio
    out = _tiled_sparse_attention(
        all_q_T,
        all_K_T_win,
        all_V_win,
        compress_kv_T,
        compress_kv,
        topk_sel_bias,
        win_bias_base_in,
        win_bias_sink_in,
        attn_sink_in,
    )
    return {"output_0": out}
