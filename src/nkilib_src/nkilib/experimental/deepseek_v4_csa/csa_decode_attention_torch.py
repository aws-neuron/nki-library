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

"""CPU references for the DeepSeek-V4 CSA decode kernels.

One reference per tested kernel, each taking the SAME parameter names as its
kernel so the test framework can pair them. They are written as the plain
definition of what the kernel computes -- dense torch ops in fp32, no tiling, no
layout tricks -- so a disagreement points at the kernel rather than at a shared
mistake.

Two conventions worth knowing when reading these:

* The kernels take the query already TRANSPOSED and head-major, so
  ``q_T_all[d, h * S_q + s] == q[s, h, d]``. Every reference undoes that
  indexing explicitly rather than reshaping, since getting it wrong silently
  is exactly the bug these tests are for.
* Decode is a single token, so ``S == 1``: each head's query is one column and
  the ``[n_heads * S, head_dim]`` output has one row per head. The references
  keep ``S`` general in the indexing anyway, and read only column 0, matching the
  kernels.
"""

import torch
import torch.nn.functional as F


def _rms_rope_rows(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor, gain, eps: float) -> torch.Tensor:
    """RMSNorm over the free axis then RoPE on the trailing ``2 * cos.shape[-1]`` channels.

    Mirrors the kernels' dtype flow exactly, which is what lets a kernel be graded
    tightly rather than loosely: normalize in fp32, round to bf16 at the RMSNorm
    output boundary (as the model does), then widen those rope channels back to
    fp32 for the rotation and round once more at the end.
    """
    half_rope = cos.shape[-1]
    rope_dim = 2 * half_rope
    nope_dim = x.shape[-1] - rope_dim

    xf = x.float()
    rms = torch.rsqrt(xf.square().mean(-1, keepdim=True) + eps)
    normed = xf * rms
    if gain is not None:
        normed = normed * gain
    normed = normed.to(torch.bfloat16)

    rope = normed[:, nope_dim:].float().unflatten(-1, (half_rope, 2))
    x1, x2 = rope[..., 0], rope[..., 1]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    rotated = torch.stack([y1, y2], dim=-1).flatten(-2).to(torch.bfloat16)
    return torch.cat([normed[:, :nope_dim], rotated], dim=-1)


def _inverse_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Undo the query rotation on the trailing rope channels of an attention output.

    The inverse of the forward rotation is the forward formula with ``sin``
    negated, which is what the kernels fuse into their finalize step:
    ``y1 = x1 * cos + x2 * sin``, ``y2 = x2 * cos - x1 * sin``.
    """
    half_rope = cos.shape[-1]
    rope_dim = 2 * half_rope
    nope_dim = x.shape[-1] - rope_dim

    rope = x[:, nope_dim:].float().unflatten(-1, (half_rope, 2))
    x1, x2 = rope[..., 0], rope[..., 1]
    y1 = x1 * cos + x2 * sin
    y2 = x2 * cos - x1 * sin
    rotated = torch.stack([y1, y2], dim=-1).flatten(-2).to(torch.bfloat16)
    return torch.cat([x[:, :nope_dim], rotated], dim=-1)


def nki_qkv_rms_rope_torch_ref(
    q_in: torch.Tensor,
    kv_in: torch.Tensor,
    weight_in: torch.Tensor,
    cos_in: torch.Tensor,
    sin_in: torch.Tensor,
    eps_val: float,
) -> dict[str, torch.Tensor]:
    """Oracle for ``nki_qkv_rms_rope_kernel``: the q heads and the kv row on one tile.

    The kernel packs both projection tails onto one partition tile and unifies them
    with a per-partition gain of ``1.0`` on the q rows and ``kv_norm.weight`` on the
    kv row. This reference instead runs the two paths SEPARATELY -- q with no gain,
    kv with its learnable gain -- and concatenates. That is the point: if the
    kernel's gain trick were not exact (``x * 1.0 == x`` in fp32), the two would
    disagree.
    """
    q_out = _rms_rope_rows(q_in, cos_in, sin_in, None, eps_val)
    kv_out = _rms_rope_rows(kv_in, cos_in, sin_in, weight_in.float(), eps_val)
    return {"output_0": torch.cat([q_out, kv_out], dim=0)}


def nki_indexer_qproj_gemv_torch_ref(wT: torch.Tensor, qr_in: torch.Tensor) -> dict[str, torch.Tensor]:
    """Oracle for ``nki_indexer_qproj_gemv``: the indexer q-projection, returned transposed.

    ``wT`` is the frozen ``wq_b`` weight pre-tiled on the host so that
    ``wT[t, kk, n] == wq_b.weight[n, t * 128 + kk]``. The reference rebuilds the
    plain ``[N, K]`` weight from that tiling and does one dense matvec, then lays
    the result out as the kernel's PSUM tile is: ``out[c, j] == q[head j, channel c]``
    with ``j`` an N-tile of 128 columns, i.e. one head's channel block.
    """
    n_ktiles, k_tile, n = wT.shape
    weight = wT.float().permute(2, 0, 1).reshape(n, n_ktiles * k_tile)
    q = weight @ qr_in.float().reshape(-1)
    return {"output_0": q.reshape(n // k_tile, k_tile).t().to(torch.bfloat16)}


def _indexer_scores(q_T_all: torch.Tensor, kv_t: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """``score[s, t] = sum_h relu(q[s, h, :] . kv[t, :]) * weights[s, h]``, as ``[S_q, T_c]``.

    The relu comes BEFORE the per-head weight, so a head whose dot product is
    negative contributes nothing at all rather than contributing negatively. That
    is what makes every real indexer score non-negative, which in turn is what lets
    the kernels pad a score row with a negative sentinel and know the padding can
    never win the top-k.
    """
    head_dim, total_q_free = q_T_all.shape
    n_heads = weights.shape[1]
    s_q = total_q_free // n_heads

    # q_T_all[d, h * S_q + s] == q[s, h, d]  ->  [S_q, n_heads, head_dim]
    q = q_T_all.float().reshape(head_dim, n_heads, s_q).permute(2, 1, 0)
    per_head = torch.einsum("shd,dt->sht", q, kv_t.float())
    return torch.einsum("sht,sh->st", F.relu(per_head), weights.float())


def nki_indexer_score_torch_ref(
    q_T_all: torch.Tensor,
    kv_t: torch.Tensor,
    weights: torch.Tensor,
    causal_bias: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Oracle for ``nki_indexer_score_kernel``: raw indexer scores plus the causal bias."""
    return {"output_0": _indexer_scores(q_T_all, kv_t, weights) + causal_bias.float()}


def nki_indexer_score_2core_torch_ref(
    q_T_all: torch.Tensor,
    kv_t: torch.Tensor,
    weights: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Oracle for ``nki_indexer_score_2core``: the assembled ``[1, T_c]`` score row.

    The kernel splits ``T_c`` in half across the two logical cores and each writes
    its own half of a shared buffer, so a reference that scores the WHOLE range is
    what catches a half that never landed -- the failure mode a per-core allocation
    produces, where core 1's half reads back as zeros.

    Decode's query rows are all identical, so the kernel writes only row 0 and this
    returns only row 0.
    """
    scores = _indexer_scores(q_T_all, kv_t, weights)
    return {"output_0": scores[0:1].to(torch.bfloat16)}


def _gather_attention(
    idx: torch.Tensor,
    all_q_T: torch.Tensor,
    win_K_T: torch.Tensor,
    win_V: torch.Tensor,
    compress_kv: torch.Tensor,
    attn_sink_in: torch.Tensor,
    derope_cos: torch.Tensor,
    derope_sin: torch.Tensor,
    n_heads: int,
    s_len: int,
) -> torch.Tensor:
    """Dense O(W + k) sparse attention over the sliding window plus ``idx``'s rows.

    This is the whole point of CSA: the softmax runs over ``window_size + k``
    positions and nothing else, so its cost is independent of how long the context
    is. The window is a full valid permutation in decode with no intra-window mask,
    and the attention sink is a per-head bias on window column 0 only.

    ``all_q_T`` arrives ALREADY scaled by ``softmax_scale`` (the host folds it in),
    so no scaling happens here.
    """
    head_dim = all_q_T.shape[0]

    # Column 0 of each head: [head_dim, n_heads]. All S columns per head are
    # identical in decode, and the kernels read only this one.
    q_hb = all_q_T.float()[:, 0 : n_heads * s_len : s_len]

    win_scores = q_hb.t() @ win_K_T.float()
    win_scores[:, 0] = win_scores[:, 0] + attn_sink_in.float().reshape(-1)

    gathered = compress_kv.float()[idx.long()]  # [k, head_dim]
    comp_scores = q_hb.t() @ gathered.t()  # [n_heads, k]

    both = torch.cat([win_scores, comp_scores], dim=-1)
    shift = both.max(dim=-1, keepdim=True).values
    win_exp = torch.exp(win_scores - shift)
    comp_exp = torch.exp(comp_scores - shift)
    total = win_exp.sum(-1, keepdim=True) + comp_exp.sum(-1, keepdim=True)

    out = (win_exp @ win_V.float() + comp_exp @ gathered) / total
    out = _inverse_rope(out.to(torch.bfloat16), derope_cos.float(), derope_sin.float())

    # The kernel writes head h to output row h * S, leaving the other rows
    # untouched; with S == 1 that is every row.
    full = torch.zeros((n_heads * s_len, head_dim), dtype=torch.bfloat16)
    full[0 : n_heads * s_len : s_len] = out
    return full


def nki_decode_gather_ok_torch_ref(
    topk_indices_T: torch.Tensor,
    all_q_T: torch.Tensor,
    win_K_T: torch.Tensor,
    win_V: torch.Tensor,
    compress_kv: torch.Tensor,
    attn_sink_in: torch.Tensor,
    derope_cos: torch.Tensor,
    derope_sin: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Oracle for ``nki_decode_gather_ok_kernel``: O(k) attention on caller-supplied indices.

    ``topk_indices_T`` is ``[k, S]``, transposed so the kernel can slice ``k`` onto
    the partition axis for the gather; every column holds the same indices in
    decode, so column 0 is what is read.
    """
    n_heads = attn_sink_in.shape[1]
    s_len = all_q_T.shape[1] // n_heads
    out = _gather_attention(
        topk_indices_T[:, 0],
        all_q_T,
        win_K_T,
        win_V,
        compress_kv,
        attn_sink_in,
        derope_cos,
        derope_sin,
        n_heads,
        s_len,
    )
    return {"output_0": out}


def nki_indexer_score_topk_gather_2core_torch_ref(
    q_T_all: torch.Tensor,
    kv_t: torch.Tensor,
    weights: torch.Tensor,
    k_val: int,
    n_val: int,
    all_q_T: torch.Tensor,
    win_K_T: torch.Tensor,
    win_V: torch.Tensor,
    compress_kv: torch.Tensor,
    attn_sink_in: torch.Tensor,
    derope_cos: torch.Tensor,
    derope_sin: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Oracle for the fused decode kernel: indexer score, top-k, and O(k) attention.

    The kernel does all three inside one launch, so nothing intermediate is
    observable and the only way to grade it is end-to-end. This reference scores
    with ``torch``, selects with ``torch.topk``, and attends densely.

    That makes the test sensitive to top-k ties: if the k-th and (k+1)-th scores are
    equal, the kernel and ``torch.topk`` may legitimately pick different positions
    and the outputs will differ. The test's input generator is therefore built to
    give the scores real separation -- which is also the realistic regime, since a
    tied indexer means the selection carries no information.
    """
    del n_val
    n_heads = attn_sink_in.shape[1]
    s_len = all_q_T.shape[1] // n_heads

    scores = _indexer_scores(q_T_all, kv_t, weights)
    idx = torch.topk(scores[0].float(), k_val).indices

    out = _gather_attention(
        idx,
        all_q_T,
        win_K_T,
        win_V,
        compress_kv,
        attn_sink_in,
        derope_cos,
        derope_sin,
        n_heads,
        s_len,
    )
    return {"output_0": out}
