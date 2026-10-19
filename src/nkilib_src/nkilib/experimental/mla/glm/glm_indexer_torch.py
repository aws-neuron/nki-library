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

"""Numpy reference for :mod:`glm_indexer`, staged to match the kernel's build order."""

import ml_dtypes
import nki.language as nl
import numpy as np

from .glm_indexer import MASK_NEG, score_row_width

# Floor mirrors the kernel's _AMAX_FLOOR: an all-zero (padding) token keeps a finite reciprocal.
_FP8_AMAX_FLOOR = 1e-30


def _fp8_np_dtype_and_max(kv_cache_fp8_dtype):
    """Map the nki fp8 dtype the kernel uses to a numpy (ml_dtypes) dtype and its max positive value.

    ``nl.float8_e4m3`` is the NON-OCP format (max 240); ``nl.float8_e4m3fn`` is OCP (max 448). The
    reference runs on CPU, so it cannot call ``nisa.get_nc_version()`` -- the caller passes the
    resolved concrete dtype, exactly as it is passed to the kernel.
    """
    name = str(kv_cache_fp8_dtype)
    if "e4m3fn" in name:
        return ml_dtypes.float8_e4m3fn, 448.0
    if "e5m2" in name:
        return ml_dtypes.float8_e5m2, 57344.0
    # Default / NON-OCP e4m3 (max 240); ml_dtypes.float8_e4m3 is exactly that format.
    return ml_dtypes.float8_e4m3, 240.0


def _quantize_k_per_token_ref(k, kv_cache_fp8_dtype):
    """Per-token fp8 round-trip of K over the full head_dim, mirroring ``quantize_k_per_token``.

    One dequant scale per token (row) = ``max|k| / fp8_max`` floored; quantize by its reciprocal,
    cast to fp8, then dequantize. Returns the fp8-then-back-to-fp32 K the kernel actually scores, so
    the score/top-k goldens reflect the fp8 loss rather than measuring the formula.
    """
    np_fp8, fp8_max = _fp8_np_dtype_and_max(kv_cache_fp8_dtype)
    amax = np.maximum(np.abs(k).max(axis=-1, keepdims=True), _FP8_AMAX_FLOOR)
    dequant = amax / fp8_max
    q = (k / dequant).astype(np_fp8).astype(np.float32)
    return np.clip(q, -fp8_max, fp8_max) * dequant


def kv_row_bf16_width_ref(head_dim):
    """bf16 containers per fused fp8 cache row, mirroring the kernel's ``kv_row_bf16_width``."""
    return -(-(head_dim // 2 + 1) // 8) * 8


def pack_k_fp8_row_ref(k, head_dim, kv_cache_fp8_dtype):
    """Pack dense K ``[n, head_dim]`` into the kernel's fused fp8 cache rows ``[n, row_width]`` bf16.

    Mirrors ``store_k_paged``'s fp8 branch so a test can build a packed ``k_ctx``/``k_cache`` the
    kernel will read: per token, quantize to fp8 (one scale over head_dim), bitcast the fp8 bytes
    two-per-bf16 container in natural channel order, then place the bf16 dequant scale right after
    the ``head_dim // 2`` latent containers.
    """
    np_fp8, fp8_max = _fp8_np_dtype_and_max(kv_cache_fp8_dtype)
    n = k.shape[0]
    row_w = kv_row_bf16_width_ref(head_dim)
    lat_bf16 = head_dim // 2
    amax = np.maximum(np.abs(k).max(axis=-1, keepdims=True), _FP8_AMAX_FLOOR)
    dequant = (amax / fp8_max).astype(np.float32)
    q_fp8 = np.clip((k / dequant), -fp8_max, fp8_max).astype(np_fp8)  # [n, head_dim] fp8
    row = np.zeros((n, row_w), dtype=nl.bfloat16)
    # Bitcast the fp8 bytes two-per-bf16 container: view row as raw bytes and drop the fp8 in.
    row_bytes = row.view(np.uint8).reshape(n, row_w * 2)
    row_bytes[:, :head_dim] = q_fp8.view(np.uint8).reshape(n, head_dim)
    # The per-token dequant scale is the bf16 field right after the latent containers.
    row[:, lat_bf16] = dequant.reshape(n).astype(nl.bfloat16)
    return row


def unpack_k_fp8_row_ref(row, head_dim, kv_cache_fp8_dtype):
    """Inverse of :func:`pack_k_fp8_row_ref`: fused fp8 rows ``[n, row_width]`` -> dense K ``[n, head_dim]``.

    Mirrors the kernel's ``dequant_k_packed``: read the fp8 latent back (paired-channel byte view),
    multiply by the in-row per-token scale. Returns fp32 dense K (bf16-rounded, as the kernel writes
    dense bf16 after dequant).
    """
    np_fp8, _ = _fp8_np_dtype_and_max(kv_cache_fp8_dtype)
    row = np.ascontiguousarray(np.asarray(row).view(nl.bfloat16))
    n, row_w = row.shape
    lat_bf16 = head_dim // 2
    row_bytes = row.view(np.uint8).reshape(n, row_w * 2)
    q_fp8 = row_bytes[:, :head_dim].view(np_fp8).reshape(n, head_dim).astype(np.float32)
    scale = row[:, lat_bf16].astype(np.float32).reshape(n, 1)
    return (q_fp8 * scale).astype(nl.bfloat16).astype(np.float32)


def _as_numpy(x):
    """Accept numpy or torch input and return float32 numpy."""
    a = x.numpy() if hasattr(x, "numpy") else x
    return np.asarray(a, dtype=np.float32)


def _layernorm_ref(x, gamma, beta, eps=1e-6):
    """LayerNorm over the last axis, matching the kernel's bn_stats/bn_aggr path.

    The kernel normalizes with the BIASED variance (``bn_stats`` accumulates sum and sum-of-squares
    over the row, so the divisor is N, not N-1) and applies ``rsqrt(var + eps)`` with eps INSIDE the
    square root. Both details have to match or the comparison measures the formula, not the kernel.
    """
    mean = x.mean(axis=-1, keepdims=True)
    var = ((x - mean) ** 2).mean(axis=-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * gamma + beta


def _pack_outputs(topk, q, gathered, score, emit_proj, emit_score):
    """Assemble the golden dict in the kernel's output order: top-k, then the gated extras.

    Only ``output_N`` keys: the framework derives the kernel's output arity from this dict, so an
    extra key makes it reject the descriptor.
    """
    out = {"output_0": topk}
    n = 1
    if emit_proj:
        out[f"output_{n}"] = q
        out[f"output_{n + 1}"] = gathered
        n += 2
    if emit_score:
        out[f"output_{n}"] = score
    return out


def _rope_interleaved_ref(x, cos, sin, rope_head_dim):
    """INTERLEAVED (adjacent-pair) RoPE on the leading ``rope_head_dim`` columns.

    Pair ``j`` is columns ``(2j, 2j+1)`` rotated by angle ``theta_j``::

        out[2j]   = x[2j] * cos_j - x[2j+1] * sin_j
        out[2j+1] = x[2j] * sin_j + x[2j+1] * cos_j

    This is GLM-MoE-DSA's pairing; DeepSeek-V3.2 uses HALF_SPLIT over contiguous halves with the
    same angle set. The trailing (nope) columns pass through unchanged.
    """
    out = x.copy()
    even = x[:, 0:rope_head_dim:2]
    odd = x[:, 1:rope_head_dim:2]
    out[:, 0:rope_head_dim:2] = even * cos - odd * sin
    out[:, 1:rope_head_dim:2] = even * sin + odd * cos
    return out


def glm_indexer_proj_ref(
    x,
    qr,
    wq,
    wk,
    k_norm_gamma,
    k_norm_beta,
    cos,
    sin,
    slot_mapping,
    weights,
    q_pos,
    end_pos=0,
    cp_degree=1,
    replica_groups=None,
    topk_nseg=0,
    roll_score=0,
    emit_score=False,
    emit_proj=False,
    k_ctx=None,
    k_cache=None,
    zero_topk=False,
    score_chunk=0,
    block_table=None,
    block_size=0,
    cp_interleave=0,
    index_topk=2048,
    kv_fp8=False,
    kv_cache_fp8_dtype=None,
):
    """Reference for ``glm_indexer_proj``: the Q and K projections.

    The kernel matmuls in bf16 and accumulates in fp32, so both operands are rounded to bf16
    before the product. Without that rounding the reference is a different computation and the
    comparison silently measures bf16 error instead of kernel correctness.

    K then goes through LayerNorm and interleaved RoPE, matching the kernel's stage 2. The
    LayerNorm runs in fp32 on the fp32 matmul output, as the kernel does, so no extra rounding is
    introduced there.

    K is then scattered into the paged cache at ``slot_mapping``, matching stage 3. Slots no token
    maps to keep their initial value (0 here), and tokens with ``slot_mapping == -1`` are dropped --
    the kernel does that with ``oob_mode.skip``.

    ``index_topk`` is accepted so the signature matches the kernel entry (the test framework checks
    that); it is unused until later stages. ``kv_fp8`` / ``kv_cache_fp8_dtype`` select the fp8 cache
    path: when set, K is run through the same per-token fp8 round-trip the kernel stores and scores.

    ``output_1`` is the CP all-gather of K. This per-rank reference only sees its OWN chunk, so it
    fills just this rank's slice and leaves the peers' zero; the test compares per rank against the
    global K it built, which is what makes a wrong concatenation order visible.

    Returns:
        dict with ``output_0`` = Q, ``output_1`` = gathered K, ``output_2`` = score.
    """
    # topk_nseg IS consumed: it sets the padded score row width via score_row_width.
    # emit_score is consumed too, and it is a HOST-MEMORY gate, not just an output switch: the
    # score below is [s_local, end_pos] fp32 -- 68.7 GB at the production-ratio 1M shape -- and
    # holding it (plus its bf16-rounded copy) while neuronx-cc runs is what got the compiler
    # OOM-killed ("[F137] neuronx-cc was forcibly killed") on the 512k/1M single-rank cases.
    # When the kernel is not emitting a score, nothing consumes this golden (the index check is
    # golden-free), so it must not be computed.
    del replica_groups, roll_score  # not consumed by the reference

    def to_bf16(a):
        return a.astype(nl.bfloat16).astype(np.float32)

    qr_b = to_bf16(_as_numpy(qr))
    wq_b = to_bf16(_as_numpy(wq))
    x_b = to_bf16(_as_numpy(x))
    wk_b = to_bf16(_as_numpy(wk))

    head_dim = np.asarray(wk).shape[1]
    n_heads = np.asarray(wq).shape[1] // head_dim
    rope_head_dim = 2 * np.asarray(cos).shape[1]
    s_global = np.asarray(k_ctx).shape[0] if k_ctx is not None else cp_degree * np.asarray(x).shape[0]
    # fp8: a caller-supplied k_ctx arrives as the fused packed rows the kernel reads; unpack it to
    # dense head_dim K here so the rest of the reference scores the same values the kernel does.
    if k_ctx is not None and kv_fp8:
        k_ctx = unpack_k_fp8_row_ref(k_ctx, head_dim, kv_cache_fp8_dtype)
    q = qr_b @ wq_b  # [s_local, n_heads * head_dim]
    k = x_b @ wk_b  # [s_local, head_dim]

    # INTERLEAVED RoPE on Q too, per head, on the leading rope_head_dim columns.
    #
    # GLM-MoE-DSA's indexer ropes BOTH Q and K: HF's GlmMoeDsaIndexer does
    #   q_rot, k_rot = apply_rotary_pos_emb_interleave(q_rot, k_rot, cos, sin, unsqueeze_dim=2)
    # and SGLang's dsa_indexer does self.rotary_emb(positions, q_rope, k_rope). This reference
    # previously roped K only, matching the kernel's identical omission, so the comparison agreed to
    # bf16 noise while both scored un-roped Q against roped K.
    #
    # unsqueeze_dim=2 means every head shares the same cos/sin row, hence the per-head loop over one
    # table rather than a per-head table.
    _cos, _sin = _as_numpy(cos), _as_numpy(sin)
    q = q.reshape(q.shape[0], n_heads, head_dim)
    for _h in range(n_heads):
        q[:, _h, :] = _rope_interleaved_ref(q[:, _h, :], _cos, _sin, rope_head_dim)
    q = q.reshape(q.shape[0], n_heads * head_dim)

    k = _layernorm_ref(k, _as_numpy(k_norm_gamma), _as_numpy(k_norm_beta))
    k = _rope_interleaved_ref(k, _as_numpy(cos), _as_numpy(sin), rope_head_dim)

    # FP8 cache: the kernel quantizes K per token over the whole head_dim (RoPE included) before
    # storing, gathers the fp8 K, and scores it. Model that round-trip here so the cache, the
    # gathered K and the score goldens all reflect the fp8 loss rather than bf16 loss.
    if kv_fp8:
        k = _quantize_k_per_token_ref(k, kv_cache_fp8_dtype)

    # Scatter into the paged cache. The cache is bf16, so round K the same way the kernel does.
    slots = np.asarray(slot_mapping.numpy() if hasattr(slot_mapping, "numpy") else slot_mapping).astype(np.int64)
    n_slots = _as_numpy(k_cache).shape[0] if k_cache is not None else x_b.shape[0]
    cache = np.zeros((n_slots, k.shape[1]), dtype=np.float32)
    valid = slots >= 0
    cache[slots[valid]] = to_bf16(k[valid])

    k_bf16 = to_bf16(k)
    # The kernel linearizes this rank's OWN pages before the gather, so mirror that for the local K.
    # Only on the real per-rank path: with a caller-supplied k_ctx the table addresses k_ctx, not the
    # cache, and covers the whole context rather than this rank's s_local rows.
    if block_table is not None and k_ctx is None and cp_degree > 1:
        _bt = np.asarray(block_table).reshape(-1).astype(np.int64)
        _rows = to_bf16(_as_numpy(cache))
        _n_blk = -(-k_bf16.shape[0] // block_size)
        _slots = (_bt[:_n_blk, None] * block_size + np.arange(block_size)[None, :]).reshape(-1)
        k_local_bf16 = _rows[_slots[: k_bf16.shape[0]]]
    else:
        k_local_bf16 = k_bf16

    # Score: sum_h relu(K_p . q_{q,h}) * w[q, h], over [0, end_pos).
    #
    # The kernel scores the bf16 Q it wrote against the bf16 K it gathered, accumulating fp32, so the
    # reference must use the SAME rounded operands -- scoring the unrounded fp32 q/k would measure
    # bf16 error instead of the kernel.
    if end_pos > 0 and emit_score:
        q_heads = to_bf16(q).reshape(q.shape[0], n_heads, head_dim)
        w = _as_numpy(weights)
        # Context source, in precedence order:
        #   k_ctx -- caller-supplied context, already in gathered order (the 1-rank path).
        #   else  -- this rank's own pages, made contiguous through the block table.
        # Mirrors the kernel: at cp_degree == 1 the table is resolved over the whole context, so the
        # golden must read the same permuted rows.
        if block_table is not None:
            bt = np.asarray(block_table).reshape(-1).astype(np.int64)
            rows = to_bf16(_as_numpy(k_ctx)) if k_ctx is not None else to_bf16(_as_numpy(cache))
            n_blk = -(-end_pos // block_size)
            slots = (bt[:n_blk, None] * block_size + np.arange(block_size)[None, :]).reshape(-1)
            ctx_k = rows[slots[:end_pos]]
        elif k_ctx is not None:
            ctx_k = to_bf16(_as_numpy(k_ctx))
        else:
            ctx_k = k_local_bf16
        k_ctx_slice = ctx_k[:end_pos]
        # Accumulate PER HEAD. The one-shot einsum materializes logits[q, h, p], which is
        # s_local * n_heads * end_pos fp32 = 8 GB per rank at 64k -- and the collective framework
        # computes all ranks' goldens in parallel, so 64 ranks wanted 512 GB on a 373 GB host. It
        # did not OOM, it just wedged at 0% CPU and never reached the compiler. Per head the
        # intermediate is [q, p] = 268 MB, 32x smaller, for identical arithmetic.
        score = np.zeros((q_heads.shape[0], end_pos), dtype=np.float32)
        for h in range(n_heads):
            score += np.maximum(q_heads[:, h, :] @ k_ctx_slice.T, 0.0) * w[:, h : h + 1]
        # Causal mask on GLOBAL positions: query q sees key p iff k_pos[p] <= q_pos[q]. At cp_degree=1
        # the gathered positions are just this rank's own, so k_pos == q_pos[:end_pos].
        qp = _as_numpy(q_pos).reshape(-1)
        # Same slot->global map the kernel's mask derives.
        cpd = max(1, (s_global or end_pos) // max(1, len(qp)))
        s_kv_local = max(1, end_pos // cpd)
        inter = cp_interleave or s_kv_local
        g = np.arange(end_pos, dtype=np.int64)
        rank, l = g // s_kv_local, g % s_kv_local
        kp = ((l // inter) * (inter * cpd) + rank * inter + (l % inter)).astype(np.float32)
        score = np.where(kp[None, :] > qp[:, None], np.float32(MASK_NEG), score)
    else:
        score = np.zeros((q.shape[0], 1), dtype=np.float32)

    # Top-k: the k highest-scoring positions per query. The kernel ranks the score in bf16 (the
    # snake it gathers is bf16), so round before ranking or ties resolve differently.
    #
    # Returned as an unordered SET per query: nisa.topk with sorted=False gives an unspecified order,
    # so the comparison can only be set-based. argpartition matches that semantics.
    if end_pos > 0 and emit_score:
        # PAD to the kernel's row width. The kernel's score buffer is score_row_width(end_pos) wide,
        # which exceeds end_pos whenever the segment count does not divide the context evenly -- at
        # 1M, nseg=24 gives T=1049088 against end_pos=1048576. Every other shape happens to come out
        # equal, so an unpadded golden only fails at 1M, with "cannot reshape array of size
        # 1074266112 into shape (1024,1048576)". The tail is -inf because those columns are ranked
        # with the real ones and must never win.
        W = score_row_width(end_pos, topk_nseg)
        if W > end_pos:
            score = np.concatenate([score, np.full((score.shape[0], W - end_pos), -np.inf, dtype=np.float32)], axis=1)
        # GOLDEN top-k: the reference's own top-k of its own score, which is what the kernel's picks
        # are judged against. ROW BY ROW because np.argpartition returns int64 -- a whole-array call
        # is 2 GiB at 1M, while one row is 8 MB.
        #
        # It is returned as an unordered SET, and the comparison must treat it that way: nisa.topk
        # runs sorted=False, so the order within a query's k picks is unspecified, and near the cut
        # positions whose scores differ by under one bf16 quantum may legitimately swap. An
        # elementwise comparison of these indices is meaningless -- that is why this was removed
        # once, but removing it left the top-k unvalidated instead of validated differently.
        # [:, :end_pos] -- the score has ALREADY been padded to score_row_width above, and at 1M that
        # is 1049088 against end_pos 1048576. Ranking the padded row makes the [cut:] slice 2560 long
        # instead of index_topk ("could not broadcast input array from shape (2560,) into shape
        # (2048,)"), and would let -inf pad columns be selected.
        sb_g = score[:, :end_pos].astype(nl.bfloat16).astype(np.float32)
        cut = end_pos - index_topk
        topk_pos = np.empty((score.shape[0], index_topk), dtype=np.int32)
        for _m in range(score.shape[0]):
            topk_pos[_m] = np.argpartition(sb_g[_m], cut)[cut:].astype(np.int32)
    elif end_pos > 0:
        # emit_score off: the kernel returns a (1, 1) score stub and the top-k is checked
        # golden-free (TopkStructureValidator), so both goldens are placeholders. Shaped to the
        # DECLARED outputs so dtype/shape inference still holds.
        score = np.zeros((q.shape[0], 1), dtype=np.float32)
        topk_pos = np.zeros((q.shape[0], index_topk), dtype=np.int32)
    else:
        topk_pos = np.zeros((q.shape[0], 1), dtype=np.int32)

    # The kernel writes Q as bf16, so round the reference the same way. The gathered K is fp32 DTYPE
    # with bf16-rounded VALUES: glm_indexer_proj_cp_ref does torch.from_numpy on it, and torch
    # rejects ml_dtypes.bfloat16. topk_pos is a PLACEHOLDER -- nisa.topk runs sorted=False, so an
    # elementwise comparison against a reference ranking is meaningless; the suites check it against
    # the kernel's own score instead.
    return _pack_outputs(
        topk_pos,
        to_bf16(q),
        to_bf16(_as_numpy(k_ctx)) if k_ctx is not None else k_local_bf16,
        score.astype(np.float32),
        emit_proj,
        emit_score,
    )


def glm_indexer_proj_cp_ref(
    x,
    qr,
    wq,
    wk,
    k_norm_gamma,
    k_norm_beta,
    cos,
    sin,
    slot_mapping,
    weights,
    q_pos,
    end_pos=0,
    cp_degree=1,
    replica_groups=None,
    topk_nseg=0,
    roll_score=0,
    emit_score=False,
    emit_proj=False,
    k_ctx=None,
    k_cache=None,
    zero_topk=False,
    score_chunk=0,
    block_table=None,
    block_size=0,
    cp_interleave=0,
    index_topk=2048,
    kv_fp8=False,
    kv_cache_fp8_dtype=None,
):
    """Multi-rank reference: same as :func:`glm_indexer_proj_ref`, but ``output_1`` is really gathered.

    Runs under ``torch.distributed`` (the collective test framework sets up the process group), so
    the all-gather here is a genuine cross-rank operation rather than a local concatenation. That is
    the point of this reference: it is the only way to catch a wrong ``collective_dim``, a replica
    group listing ranks out of context order, or a gather that silently never ran.

    ``output_0`` (Q) stays PER RANK -- only K is shared.
    """
    import torch
    import torch.distributed as dist
    from nki.collectives import ReplicaGroup

    from ...collectives.collectives_torch import get_pg

    # KEYWORD arguments, deliberately: positional forwarding silently shifts every later argument
    # when a parameter is inserted into glm_indexer_proj_ref's signature.
    local = glm_indexer_proj_ref(
        x,
        qr,
        wq,
        wk,
        k_norm_gamma,
        k_norm_beta,
        cos,
        sin,
        slot_mapping,
        weights,
        q_pos,
        end_pos=0,
        cp_degree=cp_degree,
        replica_groups=replica_groups,
        topk_nseg=0,
        roll_score=0,
        # Both forced on: this call is internal, and the CP reference needs Q, the gathered K and the
        # score regardless of what the kernel under test emits.
        emit_score=True,
        emit_proj=True,
        k_ctx=None,
        k_cache=k_cache,
        zero_topk=False,
        score_chunk=0,
        block_table=block_table,
        block_size=block_size,
        cp_interleave=cp_interleave,
        index_topk=index_topk,
        kv_fp8=kv_fp8,
        kv_cache_fp8_dtype=kv_cache_fp8_dtype,
    )

    head_dim = np.asarray(wk).shape[1]
    n_heads = np.asarray(wq).shape[1] // head_dim
    k_local = torch.from_numpy(np.ascontiguousarray(local["output_2"]))
    rg = ReplicaGroup([list(g) for g in replica_groups] if replica_groups else [list(range(cp_degree))])
    gathered = [torch.zeros_like(k_local) for _ in range(cp_degree)]
    dist.all_gather(gathered, k_local, group=get_pg(rg))

    # Positions are gathered too, exactly as the kernel does (a SEPARATE collective, because the
    # dtypes differ and a collective cannot coalesce mismatched dtypes). Gathering rather than
    # assuming q_pos == arange + rank*s_local is what makes this test able to fail on a wrong
    # replica-group order or a CP token interleaving the kernel does not expect.
    qp_local = torch.from_numpy(np.ascontiguousarray(_as_numpy(q_pos).reshape(-1)))
    qp_gathered = [torch.zeros_like(qp_local) for _ in range(cp_degree)]
    dist.all_gather(qp_gathered, qp_local, group=get_pg(rg))

    k_all = torch.cat(gathered, dim=0).numpy().astype(np.float32)
    kp_all = torch.cat(qp_gathered, dim=0).numpy().astype(np.float32)

    # Score over the GATHERED context. This is the whole point of this reference: the single-rank
    # one scores against its own K only, so it cannot catch a rank picking positions out of the
    # wrong peer's chunk. index_topk must be <= the number of positions VISIBLE to the earliest
    # query, since the causal mask hides everything at a later global position.
    if end_pos > 0:
        q_heads = local["output_1"].astype(np.float32).reshape(-1, n_heads, head_dim)
        w = _as_numpy(weights)
        qp = _as_numpy(q_pos).reshape(-1)
        k_ctx = k_all[:end_pos]
        # Accumulate PER HEAD. The one-shot einsum materializes logits[q, h, p], which is
        # s_local * n_heads * end_pos fp32 = 8 GB per rank at 64k -- and the collective framework
        # computes all ranks' goldens in parallel, so 64 ranks wanted 512 GB on a 373 GB host. It
        # did not OOM, it just wedged at 0% CPU and never reached the compiler. Per head the
        # intermediate is [q, p] = 268 MB, 32x smaller, for identical arithmetic.
        score = np.zeros((q_heads.shape[0], end_pos), dtype=np.float32)
        for h in range(n_heads):
            score += np.maximum(q_heads[:, h, :] @ k_ctx.T, 0.0) * w[:, h : h + 1]
        score = np.where(kp_all[:end_pos][None, :] > qp[:, None], np.float32(MASK_NEG), score)
        # PAD to the kernel's row width. The kernel's score buffer is score_row_width(end_pos) wide,
        # which exceeds end_pos whenever the segment count does not divide the context evenly -- at
        # 1M, nseg=24 gives T=1049088 against end_pos=1048576. Every other shape happens to come out
        # equal, so an unpadded golden only fails at 1M, with "cannot reshape array of size
        # 1074266112 into shape (1024,1048576)". The tail is -inf because those columns are ranked
        # with the real ones and must never win.
        W = score_row_width(end_pos, topk_nseg)
        if W > end_pos:
            score = np.concatenate([score, np.full((score.shape[0], W - end_pos), -np.inf, dtype=np.float32)], axis=1)
        # NOT computed: np.argpartition over [s_local, end_pos] returns int64, so at 64k that is
        # 537 MB per rank -- larger than the score itself -- and NOTHING compares it. Index sets
        # cannot be compared elementwise anyway (nisa.topk runs sorted=False, and near the cut a
        # 0.1% score difference legitimately reorders winners), so every suite either omits
        # output_3 from output_keys or checks the kernel's picks against the kernel's OWN score with
        # a golden-free validator. A placeholder keeps the output arity and dtype intact.
    else:
        score = local["output_3"]

    # bf16 goldens: the collective framework infers each output's dtype from its golden, so an fp32
    # golden for a bf16 output makes it read half the elements and fail the reshape. The top-k golden
    # is ZEROS, matching the kernel's zero_topk -- a real comparison is impossible here.
    return _pack_outputs(
        np.zeros((score.shape[0], index_topk), dtype=np.int32),
        local["output_1"].astype(nl.bfloat16),
        k_all.astype(nl.bfloat16),
        score.astype(np.float32),
        emit_proj,
        emit_score,
    )


# The dispatch audit resolves a kernel's reference by name as ``<kernel_func>_torch_ref``.
glm_indexer_proj_torch_ref = glm_indexer_proj_ref
