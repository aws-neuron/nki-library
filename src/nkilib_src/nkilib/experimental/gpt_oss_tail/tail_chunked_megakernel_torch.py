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

"""Numpy oracle for the single-collective GPT-OSS sampling tail.

PRECISION. The lm_head is bf16 (the checkpoint's own precision) and so are the matmul
activations; the accumulation, the norm and the returned logits are fp32. ``rmsnorm_tail`` is
fp32 end to end. Getting this wrong in either direction shows up as index disagreements on
tie-heavy rows rather than as value errors, which is much harder to read.

Reproduces the DATAFLOW structurally -- the per-rank lm_head shard, the all-to-all that turns
vocab-parallelism into batch-parallelism, and which batch rows each rank ends up owning -- so
a mis-sharded kernel FAILS here instead of quietly agreeing. What it does NOT reproduce is the
top-k itself: that is a plain global argsort over the assembled row, deliberately independent
of the kernel's chunked two-stage form, because the kernel's whole claim is that the two are
equivalent.

The bf16 round-trip IS modelled. The collective carries bf16 and the top-k consumes bf16, so
an fp32 oracle would disagree with the kernel on roughly half the logits.
"""

from typing import Any, Optional

import ml_dtypes
import numpy as np

from .tail_chunked_megakernel import owned_row_block, owned_rows

_MXFP4_SHUFFLE_GROUPS = 4

# Mirrored from the kernel; these are the model's sampler semantics, not arbitrary choices.
_TOPK_MASK_SENTINEL = -3000.0
_SAMPLING_EPS = 1e-5
_DETERMINISTIC_DRAW = 0.5


def _fp32(t: Any) -> np.ndarray:
    """Coerce a numpy array or torch Tensor to float32 numpy.

    The harness hands every numpy input over as a torch Tensor, so a bare ``.astype`` would
    raise ``'Tensor' object has no attribute 'astype'``.
    """
    if isinstance(t, np.ndarray):
        return t.astype(np.float32)
    detach = getattr(t, "detach", None)
    if detach is not None:
        return detach().to("cpu").float().numpy()
    return np.asarray(t, dtype=np.float32)


def _bf16(x: np.ndarray) -> np.ndarray:
    """Round through bfloat16 and back, exactly as a bf16 store would (round-to-nearest-even).

    ``ml_dtypes`` is a hard dependency of this package, so this is the ONE bf16 rounding
    routine: a private bit-twiddling fallback that once lived here drifted from IEEE on NaN
    (its rounding carry propagated through the exponent) and had to be kept in lockstep.
    """
    return x.astype(ml_dtypes.bfloat16).astype(np.float32)


# The row-ownership map lives in the KERNEL module (next to compose_hops -- the two must
# agree, and production callers de-permute with it, so it cannot live in a test-oriented
# module). Re-exported under its historical name for the tests and pod-side verify scripts.
rank_row_block = owned_row_block


def rmsnorm_tail(x, gamma, eps: float, hidden_actual: Optional[int] = None, zero_interleaved_pad: bool = True):
    """Final RMSNorm, fp32, mxfp4 convention: variance is ``sum(x^2)/H_u``.

    Because the padding lanes are exactly zero this equals ``mean(x^2 over H) * H/H_u``.
    Padding is re-zeroed AFTER the gamma multiply (gamma's padding entries are not guaranteed
    zero), through a ``[4, H/4]`` view because the mxfp4 checkpoint INTERLEAVES the padding
    lanes rather than leaving a contiguous tail.
    """
    xf = _fp32(x)
    hidden = xf.shape[-1]
    h_u = hidden if hidden_actual is None else hidden_actual
    var = np.sum(np.square(xf), axis=-1, keepdims=True) / h_u
    out = xf * (1.0 / np.sqrt(var + eps)) * _fp32(gamma).reshape(1, -1)
    if h_u < hidden:
        if zero_interleaved_pad:
            gw, keep = hidden // _MXFP4_SHUFFLE_GROUPS, h_u // _MXFP4_SHUFFLE_GROUPS
            v = out.reshape(*out.shape[:-1], _MXFP4_SHUFFLE_GROUPS, gw)
            v[..., keep:] = 0.0
            out = v.reshape(*out.shape[:-1], hidden)
        else:
            out[..., h_u:] = 0.0
    return out


def sample_rows(vals: np.ndarray, idx: np.ndarray, sp: np.ndarray, return_cdf: bool = False):
    """Sampler on ``[rows, k]``: top-k mask, temperature, softmax, top-p, position, gather.

    Two behaviours that look like bugs but are the model's semantics: top-p uses a STRICT
    ``>`` on the inclusive cumsum so the crossing token is DROPPED, and the mask sentinel is
    a finite ``-3000.0``, not -inf (an -inf sentinel makes a fully-masked row's manual softmax
    produce NaN).

    ``sp`` is ``[rows, 3]`` or ``[rows, 4]``, mirroring the kernel: an optional 4th column is
    the per-row inverse-transform draw ``u``; without it the compiled-in constant applies.
    """
    top_k_v, top_p_v, temp_v = sp[:, 0:1], sp[:, 1:2], sp[:, 2:3]
    draw = sp[:, 3:4] if sp.shape[1] > 3 else np.full_like(temp_v, _DETERMINISTIC_DRAW)
    v = vals.astype(np.float32)
    kk = v.shape[1]
    kth = np.clip(np.minimum(top_k_v, kk) - 1.0, 0, kk - 1).astype(np.int64)
    masked = np.where(v < np.take_along_axis(v, kth, axis=-1), _TOPK_MASK_SENTINEL, v)
    greedy = temp_v < _SAMPLING_EPS
    scaled = masked / np.where(greedy, 1.0, temp_v)
    ex = np.exp(scaled - scaled.max(axis=-1, keepdims=True))
    probs = ex / ex.sum(axis=-1, keepdims=True)
    over = np.cumsum(probs, axis=-1) > top_p_v
    over[:, 0] = False
    filt = np.where(over, 0.0, probs)
    filt = filt / filt.sum(axis=-1, keepdims=True)
    cdf = np.cumsum(filt, axis=-1)
    cdf = cdf / cdf[:, -1:]
    # Greedy is slot 0 BY CONTRACT, not argmax(filt): the input is descending and slot 0 can
    # never be masked (the threshold compare is strict-less against a value <= vals[0]), so
    # slot 0 is always the highest KEPT candidate. argmax(filt) agrees everywhere except the
    # sub-sentinel regime (every value below the finite -3000 sentinel), where it would pick a
    # masked slot -- i.e. return a token the caller's top_k cut explicitly excluded. A masked
    # slot is never an eligible greedy answer, so the contract pins slot 0 and the kernel's
    # single-memset greedy path is exact in every regime.
    pos = np.where(greedy.reshape(-1), 0, np.sum(draw > cdf, axis=-1))
    tokens = np.take_along_axis(idx, pos.reshape(-1, 1), axis=-1).astype(np.uint32)
    if return_cdf:
        # For gates that need to recognise draw-boundary coincidences: the kernel evaluates
        # the position compare in multiplied form (fcum < draw * last) where this reference
        # divides, and the two round differently in the last ulp. A row whose draw sits
        # within an ulp of a cdf step can legally land one slot apart -- provable only with
        # the cdf in hand.
        return tokens, cdf
    return tokens


def gpt_oss_tail_chunked_ref(
    hidden_states: np.ndarray,
    gamma: np.ndarray,
    lm_head_weight_t_shards: list,
    k: int,
    eps: float = 1e-5,
    hidden_actual: Optional[int] = None,
    zero_interleaved_pad: bool = True,
    sampling_params: Optional[np.ndarray] = None,
    return_full_rows: bool = False,
    hops=None,
    batch_groups: int = 1,
) -> dict:
    """Oracle for the single-collective tail. Returns per-rank results keyed by rank id.

    Stages, mirroring the kernel:
      1. norm (replicated) + per-rank lm_head shard        ``[gbs, V_s]``
      2. all_to_all over ALL VTP ranks, in rank order      rank r holds ``[R, V]``
      3. global top-k on the assembled full-vocab row      ``[R, k]``
      4. sampler                                          ``[R, 1]``
      5. rank-ordered token gather (the identity)          ``[gbs, 1]``

    ``hops`` is the all-to-all hop decomposition (None = one hop over all ranks). It changes
    only WHICH batch rows each rank ends up owning, never the vocab column order, so a kernel
    that got the hop composition wrong disagrees here on rows rather than silently agreeing.
    ``batch_groups`` is the batch-grouped-exchange form (single hop only): ownership becomes
    the STRIDED ``owned_rows`` map, again changing rows and nothing else.

    ``sampling_params`` is the FULL ``[gbs, 3]`` batch here and is sliced per rank with
    ``owned_rows``. The KERNEL by contract receives only its own rows, so this is where
    the two conventions are reconciled -- deliberately on the oracle side, so a kernel that
    ignored the contract disagrees.

    MEMORY. The assembled rows are built ONE RANK AT A TIME rather than by materialising every
    shard's logits first. At the production shape that is the difference between a 6.4 MB
    working set and 823 MB, and the flop count is identical either way
    (``gbs * H * V`` total). ``return_full_rows`` additionally keeps every rank's ``[R, V]``
    assembled logits for a strict pairing check; leave it off above small VTP, where it would
    cost ``vtp * R * V * 4`` bytes.
    """
    vtp = len(lm_head_weight_t_shards)
    gbs = hidden_states.shape[0]
    if gbs % vtp:
        raise ValueError(f"gbs ({gbs}) must be divisible by VTP ({vtp})")
    rows_per_rank = gbs // vtp
    # The lm_head is bf16 in the checkpoint and the kernel matmuls it in bf16, so the oracle
    # rounds BOTH operands to bf16 before multiplying. Doing the reference in fp32 would hold the
    # kernel to a precision the model itself does not have, and at k=256 on a tie-heavy row that
    # shows up as index disagreements rather than value ones.
    shards = [_bf16(_fp32(w)) for w in lm_head_weight_t_shards]
    v_s = shards[0].shape[1]

    # ---- 1. norm, replicated across ranks exactly as in the model ----
    # The norm itself stays fp32; only the matmul operands are bf16 (PSUM accumulates fp32).
    normed = rmsnorm_tail(hidden_states, gamma, eps, hidden_actual, zero_interleaved_pad)
    normed_bf = _bf16(normed)
    sp = None if sampling_params is None else _fp32(sampling_params)

    out_vals, out_idx, out_tok, full_rows = {}, {}, {}, {}
    for r in range(vtp):
        rows = owned_rows(r, vtp, gbs, hops, batch_groups)
        # ---- 2. all_to_all: this rank's rows against EVERY peer's vocab shard, IN RANK
        # ORDER. Concatenating in rank order is what makes a column index a global vocab id.
        # ``rows`` is contiguous at batch_groups == 1 (owned_row_block), strided otherwise;
        # numpy fancy-indexing handles both, byte-identically to the old slice at 1.
        nb = normed_bf[rows]
        full = _bf16(np.concatenate([nb @ w for w in shards], axis=-1))  # [nr, V]
        # ---- 3. global top-k. Column index IS the global vocab id, so no correction. ----
        order = np.argsort(-full, axis=-1, kind="stable")[:, :k]
        out_vals[r] = np.take_along_axis(full, order, axis=-1)
        out_idx[r] = order.astype(np.uint32)
        if sp is not None:
            # ---- 4. sampler on this rank's rows ----
            out_tok[r] = sample_rows(out_vals[r], out_idx[r], sp[rows, :])
        if return_full_rows:
            full_rows[r] = full

    res = {
        "topk_values": out_vals,
        "topk_indices": out_idx,
        "rows_per_rank": rows_per_rank,
        "vocab": vtp * v_s,
    }
    if return_full_rows:
        res["full_rows"] = full_rows
    if sampling_params is not None:
        # ---- 5. rank-ordered gather. Identity, but verified rather than assumed. ----
        full_tok = np.zeros((gbs, 1), dtype=np.uint32)
        seen = np.zeros(gbs, dtype=np.int64)
        for r in range(vtp):
            # The SAME map step 2 owned by -- strided with batch_groups > 1. Assembling with
            # the contiguous block here silently row-permuted tokens_full for G > 1.
            rows = owned_rows(r, vtp, gbs, hops, batch_groups)
            full_tok[rows] = out_tok[r]
            seen[rows] += 1
        if not (seen == 1).all():
            raise AssertionError("row ownership does not tile the batch exactly once")
        res["tokens"] = out_tok
        res["tokens_full"] = full_tok
    return res
