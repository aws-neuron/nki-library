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

"""
GDN chunked prefill: one entry point, two intra-chunk inverse algorithms.

`gdn_cte` is the only `@nki.jit` kernel here. `gdn_cte_schur` and `gdn_cte_neumann` are
plain traced functions this dispatcher inlines selected by `algorithm=`. Both compute the
gated delta-rule (GDN) linear-attention recurrence in chunk-parallel form from the same
inputs and are validated against one CPU reference, `gdn_cte_torch_ref`. They differ in
how the intra-chunk triangular system is solved:

  ALGORITHM_SCHUR (default; outperforms neumann at every benchmarked shape)
      Forms (I + A)^-1 by a recursive SCHUR-COMPLEMENT block merge,
      inv([[P, 0], [R, Q]]) = [[P^-1, 0], [-Q^-1 R P^-1, Q^-1]], with the block size
      doubling from 1 to CHUNK=128: log2(128) = 7 merge levels, 12 matmuls where the
      literal recursion would be 127 block operations. No power of A is formed, so nothing
      depends on A being small. The levels are emitted LEVEL-MAJOR across a group of
      chunks, amortizing each level's dependency stall over `group_size` chunks.

  ALGORITHM_NEUMANN
      Sums a terminating NEUMANN SERIES, (I - A_ii)^-1 = sum_j A_ii^j, on 16x16
      strictly-lower-triangular diagonal blocks by recursive doubling (exact -- such a
      block is nilpotent), then stitches the blocks with block forward substitution.
      Powers of A are only safe at that block size: at 128x128 a row of A has up to 127
      nonzeros and the powers grow geometrically. CHUNK=64 is used here, since the
      substitution's coupling term grows as the square of the sub-block count, and half
      the chunk length doubles the per-token cost of the CHUNK x CHUNK decay mask and Gram
      matrix. It does accept S a multiple of 64 rather than 128.

Prefer the default unless the sequence length is a multiple of 64 but not 128.

LAYOUTS. `gdn_cte_schur` takes this dispatcher's arguments directly. `gdn_cte_neumann`
uses flat [B, S, D] q/k/v with [B, S] gate/beta, so `_dispatch_neumann` folds the head axis
into the batch axis (a free reshape -- the layout is token-major and contiguous) and folds
it back out of the results. That reshape cannot supply three of the family's features, so
ALGORITHM_NEUMANN restricts the call:

  * n_k_heads must EQUAL n_v_heads: a flat [B*Hv, S, D] layout gives every value head its
    own q/k rows, so GQA head sharing requires the caller to repeat the key/query heads.
  * `recurrent_state` must be omitted: that path always starts from a zero state and
    allocates its own state output.
  * Dv must EQUAL Dk: the flat [B, S, D] signature carries one head dim for q, k and v.

Those, and the family bounds (Dq == Dk, and all head dims <= MAX_HEAD_DIM=128), are enforced
with kernel_assert at trace time.
"""

from typing import Optional

import nki
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from .gdn_cte_neumann import gdn_cte_neumann
from .gdn_cte_schur import DEFAULT_GROUP_SIZE, gdn_cte_schur
from .gdn_cte_utils import (
    ALGORITHM_NEUMANN,
    ALGORITHM_SCHUR,
    ALGORITHMS,
    CHUNK,
    DEFAULT_ALGORITHM,
    MAX_HEAD_DIM,
)


@nki.jit
def gdn_cte(
    query: nl.NkiTensor,
    key: nl.NkiTensor,
    value: nl.NkiTensor,
    g_log: nl.NkiTensor,
    beta: nl.NkiTensor,
    scale: float = 1.0,
    algorithm: str = DEFAULT_ALGORITHM,
    mask_pack: Optional[nl.NkiTensor] = None,
    group_size: int = DEFAULT_GROUP_SIZE,
    recurrent_state: Optional[nl.NkiTensor] = None,
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Chunked gated delta-rule (GDN) prefill kernel.

    Computes the gated delta-rule linear-attention recurrence in chunk-parallel form and
    returns the output hidden states plus the final recurrent state. `algorithm` selects
    the intra-chunk triangular-inverse method. See the module docstring for each.

    Intended for prefill of a few hundred tokens and up; the default algorithm's chunk
    grouping needs at least `group_size` chunks (1024 tokens at the default of 8) to
    amortize, and its measured advantage grows from 1.8x at S=128 to about 3.7x at S>=2048
    (package README).

    Dimensions:
        B: Batch size
        Hk: Key/query head count
        Hv: Value head count, a multiple of Hk (GQA ratio Hv // Hk)
        Dk: Key/query/state head dimension (<= 128)
        Dv: Value/state head dimension
        S: Sequence length

    Args:
        query (nl.NkiTensor): [B, Hk, S, Dk], Query tensor in HBM, TOKEN-major; the permute
            the matmuls need rides on the load DMA. ITS DTYPE SETS THE COMPUTE DTYPE --
            matmul operands, q/k/v tiles and `out`. Must be L2-normalized, see Notes.
        key (nl.NkiTensor): [B, Hk, S, Dk], Key tensor in HBM, TOKEN-major, L2-normalized,
            same dtype as `query`.
        value (nl.NkiTensor): [B, Hv, S, Dv], Value tensor in HBM, same dtype as `query`.
        g_log (nl.NkiTensor): [B, Hv, S], Per-token log-decay (negative) in HBM, fp32.
        beta (nl.NkiTensor): [B, Hv, S], Per-token update strength in (0, 1) in HBM, fp32.
        scale (float): Query scaling, typically 1/sqrt(Dk). Applied ON DEVICE.
        algorithm (str): Intra-chunk inverse to use, `"schur"` (default) or `"neumann"`.
            Trace-time constant.
        mask_pack (Optional[nl.NkiTensor]): [N_MASKS, CHUNK, CHUNK] fp32 mask pack in HBM.
            REQUIRED by `"schur"`; build it with `build_mask_pack()`.
        group_size (int): Chunks per grouped inverse for `"schur"`.
        recurrent_state (Optional[nl.NkiTensor]): [B, Hv, Dk, Dv] fp32, OPTIONAL MUTABLE IN/OUT.
            Supplied, it is aliased to the second output, updated IN PLACE, and its
            incoming contents are the initial state. Omitted, the kernel allocates the
            output and starts from a zero state with no load emitted. `"neumann"` supports
            ONLY the omitted form.

    Returns:
        out (nl.NkiTensor): [B, Hv, S, Dv], Output hidden states in HBM in the compute
            dtype, HEAD-major and UN-normalized (no fused output epilogue).
        recurrent_state (nl.NkiTensor): [B, Hv, Dk, Dv], Final recurrent state, always
            fp32 -- the caller's buffer when one was supplied.

    Notes:
        - Dq must equal Dk (they contract), and Dk, Dv <= MAX_HEAD_DIM=128.
        - S must be divisible by 128 for `"schur"` and by 64 for `"neumann"`. This
          dispatcher enforces only the selected algorithm's constraint.
        - Hv must be divisible by Hk; value head h reads key/query head h // (Hv//Hk).
          `"neumann"` additionally requires Hk == Hv (no head sharing).
        - q/k MUST arrive L2-normalized. That bounds q.k in [-1, 1], which is what makes
          bf16 matmul operands safe here; un-normalized inputs put the (I+A) inverse
          outside bf16's usable range.
        - The recurrent state is fp32 whatever the compute dtype: it accumulates across
          the whole sequence, so it is the one place a narrow mantissa would compound.
    """
    kernel_assert(
        algorithm in ALGORITHMS,
        f"algorithm must be one of {ALGORITHMS}, got {algorithm!r}",
    )
    if algorithm == ALGORITHM_NEUMANN:
        return _dispatch_neumann(
            query,
            key,
            value,
            g_log=g_log,
            beta=beta,
            scale=scale,
            recurrent_state=recurrent_state,
        )
    kernel_assert(
        mask_pack is not None,
        f"algorithm={ALGORITHM_SCHUR!r} requires `mask_pack`. Build one with build_mask_pack().",
    )
    """KEYWORDS past the tensors: `gdn_cte_schur` carries scheduling knobs that get added over
    time, and a positional call silently reassigns them all when one is inserted."""
    return gdn_cte_schur(
        query,
        key,
        value,
        g_log,
        beta,
        mask_pack,
        scale=scale,
        group_size=group_size,
        recurrent_state=recurrent_state,
    )


def _dispatch_neumann(
    query: nl.NkiTensor,
    key: nl.NkiTensor,
    value: nl.NkiTensor,
    g_log: nl.NkiTensor,
    beta: nl.NkiTensor,
    scale: float,
    recurrent_state: Optional[nl.NkiTensor],
) -> tuple[nl.NkiTensor, nl.NkiTensor]:
    """Adapt the family's multi-head layout onto `gdn_cte_neumann`'s [B, S, D] interface.

    That path takes flat [B, S, D] q/k/v with [B, S] gate and beta, a zero initial state,
    and builds its own masks. The adaptation is only a reshape: the family's layout is
    token-major and contiguous, so folding the head axis into the batch axis is a view, and
    so is folding it back out of the results. GQA head sharing and a non-zero initial state
    cannot be expressed that way and are rejected below.

    Args:
        query (nl.NkiTensor): [B, Hk, S, Dk], token-major.
        key (nl.NkiTensor): [B, Hk, S, Dk], token-major.
        value (nl.NkiTensor): [B, Hv, S, Dv].
        g_log (nl.NkiTensor): [B, Hv, S], per-token log decay.
        beta (nl.NkiTensor): [B, Hv, S], per-token update strength.
        scale (float): Query scaling, applied on device.
        recurrent_state (nl.NkiTensor): Must be None -- see below.

    Returns:
        tuple[nl.NkiTensor, nl.NkiTensor]: `(out, recurrent_state)` in the family's shapes,
            [B, Hv, S, Dv] and [B, Hv, Dk, Dv].
    """
    batch, n_k_heads, seqlen, key_head_dim = query.shape
    n_v_heads = value.shape[1]
    value_head_dim = value.shape[-1]

    """Head dims: nothing downstream checks them on this path (`gdn_cte_neumann` derives D
    from `q.shape`), so the family's bounds are enforced here. Dv == Dk is an extra
    restriction of this path only."""
    kernel_assert(
        key_head_dim == key.shape[-1],
        f"query and key must have the same head dim, got {key_head_dim} and {key.shape[-1]}",
    )
    kernel_assert(
        key_head_dim <= MAX_HEAD_DIM and value_head_dim <= MAX_HEAD_DIM,
        f"head dims must be at most MAX_HEAD_DIM={MAX_HEAD_DIM}, got key_head_dim="
        f"{key_head_dim} and value_head_dim={value_head_dim}",
    )
    kernel_assert(
        key_head_dim == value_head_dim,
        f"algorithm={ALGORITHM_NEUMANN!r} requires key_head_dim == value_head_dim, got "
        f"{key_head_dim} and {value_head_dim}: it takes flat [B, S, D] q/k/v, so one head "
        f"dim covers all three. Use algorithm={ALGORITHM_SCHUR!r} for differing head dims.",
    )

    # No GQA: the flat layout gives every value head its own q/k rows, so Hk < Hv would
    # need those rows duplicated, an HBM copy of q and k on every call.
    kernel_assert(
        n_k_heads == n_v_heads,
        f"algorithm={ALGORITHM_NEUMANN!r} requires n_k_heads == n_v_heads, got "
        f"{n_k_heads} and {n_v_heads}. It takes flat [B, S, D] q/k/v and so has no "
        "head-sharing; repeat the key/query heads to n_v_heads before calling, or use "
        f"algorithm={ALGORITHM_SCHUR!r}, which does GQA natively.",
    )
    # No initial state: that path always starts from a memset-zero state and allocates its
    # own state output, so it can neither read an incoming state nor alias the caller's.
    kernel_assert(
        recurrent_state is None,
        f"algorithm={ALGORITHM_NEUMANN!r} does not accept a `recurrent_state`: it always "
        "starts from a zero state and allocates its own state output. Omit the argument, "
        f"or use algorithm={ALGORITHM_SCHUR!r} to continue from a state.",
    )

    n_flat_lanes = batch * n_v_heads
    """KEYWORDS, and not for tidiness: this signature orders the two per-token scalars
    `(g_log, beta)` while `gdn_cte_neumann` orders them `(beta, gate)`. Both are [B, S] fp32,
    so passing them positionally means the swap is correct only by inspection, and reordering
    either signature would exchange the decay gate with the update strength silently -- no
    shape or dtype check can catch it, and the error is a plausible-looking wrong answer."""
    out_flat, state_flat = gdn_cte_neumann(
        q=query.reshape((n_flat_lanes, seqlen, key_head_dim)),
        k=key.reshape((n_flat_lanes, seqlen, key_head_dim)),
        v=value.reshape((n_flat_lanes, seqlen, value_head_dim)),
        beta=beta.reshape((n_flat_lanes, seqlen)),
        gate=g_log.reshape((n_flat_lanes, seqlen)),
        scale=scale,
    )
    return (
        out_flat.reshape((batch, n_v_heads, seqlen, value_head_dim)),
        state_flat.reshape((batch, n_v_heads, key_head_dim, value_head_dim)),
    )


# Re-exported so a caller can select an algorithm and check the sequence-length
# constraint without importing the individual kernel modules.
__all__ = [
    "ALGORITHMS",
    "ALGORITHM_NEUMANN",
    "ALGORITHM_SCHUR",
    "CHUNK",
    "DEFAULT_ALGORITHM",
    "DEFAULT_GROUP_SIZE",
    "MAX_HEAD_DIM",
    "gdn_cte",
]
