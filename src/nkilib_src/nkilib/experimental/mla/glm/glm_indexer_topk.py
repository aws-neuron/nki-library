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

"""Segment-and-merge top-k for :mod:`glm_indexer`, for contexts past ``nisa.topk``'s reach.

Self-contained so this kernel can evolve its segment geometry, roll policy and output layout without
perturbing the first indexer's validated configs. Four shape rules are load-bearing:

  * Stage 1 spends a call's 8 snakes on 8 position SEGMENTS of one query, not on 8 queries -- only
    the former collapses to a single DMA descriptor. Invisible in simulation, which does not lower DMA.
  * The position lookup is a one-hot MATMUL, never a gather: a winner usually lives on a different
    partition, and only ``nc_matmul`` contracts along the partition axis.
  * The merge tree takes the largest DIVISOR of n_sets that fits, not the largest power of two,
    which silently drops a candidate set at ``nseg=9``.
  * The query-batch loop is rolled with ``nl.fori_loop``; unrolled, 512k exceeds BIR's uint32 body
    offset range.

The roll is NOT gated on ``nseg >= 16`` as in the original: trace size is driven by
``tiles_per_core * batches``, which ``nseg`` only correlates with, and that gate is why 256k emitted
444244 instructions while 512k compiled with 115588. Here the loop rolls whenever it is legal.
"""

import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import dge_mode

from ....core.utils.kernel_assert import kernel_assert
from ....core.utils.kernel_helpers import div_ceil

# --------------------------------------------------------------------------------------------
# Hardware constants (duplicated deliberately -- see the module docstring on self-containment)
# --------------------------------------------------------------------------------------------

P_MAX = 128
TOPK_QUERY_CHUNKS = 16
TOPK_QUERIES_PER_BATCH = 8
TOPK_N_MAX = 65536
TOPK_N_MIN = 8

NUM_TOPK_BATCHES = P_MAX // TOPK_QUERIES_PER_BATCH
SCORE_RING = 2
_PSUM_MOVING_F32 = 512
SEG_ALIGN = 2 * TOPK_QUERY_CHUNKS
SEG_MAX = ((TOPK_N_MAX - 1) // SEG_ALIGN) * SEG_ALIGN  # = 65504
_NSEG_CHOICES = (1, 2, 4, 8, 16, 32, 64, 128)

_UNIQ = [0]


def _u():
    """Monotonic suffix for ``alloc_stack`` names, which must be unique across the whole trace."""
    _UNIQ[0] += 1
    return f"_gt{_UNIQ[0]}"


def round_up_to_chunks(end_pos: int) -> int:
    """Round ``end_pos`` up to a whole number of snakes."""
    return div_ceil(end_pos, TOPK_QUERY_CHUNKS) * TOPK_QUERY_CHUNKS


def topk_geometry(end_pos: int, nseg_override: int = 0):
    """Padded score width and stage-1 segmentation for one top-k over ``end_pos`` positions.

    Returns ``(T, nseg, seg, spc)``:
      * ``T`` -- the width ``score_padded_hbm`` must be, with ``[end_pos, T)`` filled -inf.
      * ``nseg`` -- position segments per query; 1 means the single-stage path.
      * ``seg`` -- elements each segment's snake ranks (the ``n`` passed to ``nisa.topk``).
      * ``spc`` -- segments one ``nisa.topk`` CALL covers; ``8 // spc`` queries ride with them.

    One stage-1 call must read a contiguous run of ONE query's row, so that the access pattern
    collapses to an outer dimension divisible by 128. Hence ``spc = min(nseg, 8)`` and ``nseg`` is
    always a multiple of it; spending a call on 8 separate queries does not lower.

    Two regimes: powers of two up to 8 segments (``seg = T // nseg`` exactly), and past that the
    smallest ``nseg`` with ``nseg * SEG_MAX >= T`` rounded up to a multiple of 8. The second keeps the
    segment width from halving unnecessarily, which at 1M removes a whole merge level.

    ``seg`` rounds up to ``SEG_ALIGN`` (32), not 16 -- an odd ``seg // 16`` does not compile. The cost
    is padding: ``T = nseg * seg`` exceeds ``end_pos`` by up to ``seg``, and those columns MUST be
    -inf or they win the top-k.
    """
    T0 = round_up_to_chunks(end_pos)
    if T0 < TOPK_N_MAX and not nseg_override:
        # An explicit nseg_override BYPASSES this early return, so segmentation is reachable
        return T0, 1, T0, 1
    if nseg_override:
        nseg = nseg_override
    else:
        nseg = next(
            (n for n in _NSEG_CHOICES if n <= TOPK_QUERIES_PER_BATCH and T0 % n == 0 and T0 // n < TOPK_N_MAX),
            0,
        ) or (div_ceil(div_ceil(T0, SEG_MAX), TOPK_QUERIES_PER_BATCH) * TOPK_QUERIES_PER_BATCH)
    spc = min(nseg, TOPK_QUERIES_PER_BATCH)
    pow2_in_call = nseg <= TOPK_QUERIES_PER_BATCH and (nseg & (nseg - 1)) == 0 and T0 % nseg == 0
    seg = T0 // nseg if pow2_in_call else div_ceil(div_ceil(T0, nseg), SEG_ALIGN) * SEG_ALIGN
    kernel_assert(
        nseg % spc == 0,
        f"nseg must be a multiple of the segments-per-call ({spc}) so every stage-1 call reads a "
        f"whole contiguous row slice; got nseg={nseg}. A partial call cannot be lowered to DMA.",
    )
    kernel_assert(
        TOPK_N_MIN <= seg <= SEG_MAX,
        f"segment width {seg} (nseg={nseg}, end_pos={end_pos}) must lie in [{TOPK_N_MIN}, {SEG_MAX}]; "
        f"nisa.topk needs n < {TOPK_N_MAX} and a whole {TOPK_QUERY_CHUNKS}-element snake.",
    )
    kernel_assert(
        nseg * seg >= T0,
        f"segmentation does not cover the context: nseg*seg={nseg * seg} < T={T0}",
    )
    kernel_assert(
        seg % SEG_ALIGN == 0,
        f"segment width {seg} (nseg={nseg}, end_pos={end_pos}) must be a multiple of {SEG_ALIGN} so "
        f"src_x_seg = seg // {TOPK_QUERY_CHUNKS} is EVEN; an odd extent fails the stage-1 nisa.topk "
        f"backend ISA check (NCC_IXCG164). See SEG_ALIGN.",
    )
    return nseg * seg, nseg, seg, spc


def _merge_plan(nseg: int, k: int) -> list[int]:
    """Sets merged per ``nisa.topk`` call, one entry per merge level.

    Stage 1 leaves ``nseg`` candidate sets of ``k`` per query. A merge call can rank at most
    ``TOPK_N_MAX - 1`` candidates in one snake, so it absorbs at most ``m_max = (TOPK_N_MAX-1)//k``
    sets. If every set fits in one call the plan is a single level; otherwise a level takes the
    largest DIVISOR of ``n_sets`` that fits, so the tree stays uniform, and repeats.

    The divisor requirement is the subtle part. An earlier version took the largest POWER OF TWO
    below ``n_sets``, which is correct only when ``nseg`` is itself a power of two -- and became
    wrong the moment ``topk_geometry`` started returning 9 and 17. At ``nseg = 9`` it planned a
    single 8-set merge and SILENTLY DROPPED the ninth set: the score was unaffected, and the top-k
    lost only the candidates of one segment in nine, so it surfaced as a 49% error on the smallest
    of the 2048 selected values and nothing else. Hence the product assert below -- a plan that
    does not account for every set is a correctness bug that looks like a tolerance problem.

    Every level keeps ``k`` survivors per output set -- it CANNOT keep fewer. A global top-k
    element only has to be in the top-k of the set that holds it, so all ``k`` of any set could be
    global winners; trimming a level to ``k/2`` silently drops real winners in exactly the cases
    where the candidates are unevenly spread.

    Both long shapes resolve in ONE level, so the multi-level path below is reached only by an
    explicit ``topk_nseg`` override. Kept because past 1M ``nseg * k`` climbs back over
    ``TOPK_N_MAX``.
    """
    m_max = (TOPK_N_MAX - 1) // k
    kernel_assert(m_max >= 2, f"a merge must combine at least two sets; k={k} allows only {m_max}")
    plan = []
    n_sets = nseg
    while n_sets > 1:
        if n_sets <= m_max:
            m = n_sets
        else:
            m = max(d for d in range(2, m_max + 1) if n_sets % d == 0)
        plan.append(m)
        n_sets //= m
    merged = 1
    for m in plan:
        merged *= m
    kernel_assert(
        merged == nseg,
        f"merge plan {plan} accounts for {merged} of {nseg} candidate sets; the rest would be "
        f"dropped from the selection without any error",
    )
    return plan


def make_onehot_consts(sbm):
    """Loop-invariant operands for :func:`_lookup_positions_onehot`.

    Built once per s-tile and passed in: the partition ramp and the two ones-vectors do not
    depend on the query group, and rebuilding them inside the per-group loop emitted them
    ``S/qpc`` times (64x at 128k), which is pure instruction-count bloat.
    """
    iota_u = sbm.alloc_stack((P_MAX, 1), nl.uint32, name=f"oh_iota_u{_u()}")
    iota_f = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"oh_iota_f{_u()}")
    nisa.iota(dst=iota_u, pattern=[[0, 1]], channel_multiplier=1)
    nisa.tensor_copy(dst=iota_f, src=iota_u)
    ones_row_bf = sbm.alloc_stack((1, P_MAX), nl.bfloat16, name=f"oh_ones_row{_u()}")
    nisa.memset(dst=ones_row_bf, value=1.0)
    ones_col = sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"oh_ones_col{_u()}")
    nisa.memset(dst=ones_col, value=1.0)
    return iota_f, ones_row_bf, ones_col


def _alloc_chunked_scratch(sbm, k, src_x_seg, cand_x, k_cols, plan, nseg):
    """Every SBUF buffer the chunked top-k needs, allocated ONCE and reused per query batch.

    Two reasons this is one flat set rather than per-iteration allocations in nested scopes.

    **Aliasing.** ``close_scope`` releases addresses but the compiler cannot see that the next
    scope's tensor aliases the previous one, so a still-in-flight DMA reading a released buffer
    reads whatever the next iteration wrote -- the failure documented for ``persist_buf`` in
    ``score_and_persist_blocked_bf16`` (7939 of 8256 score cells wrong). The chunked path is full
    of SBUF->SBUF DMAs (the candidate relayout, the winner flatten), so it is exposed to exactly
    that. Reusing ONE tensor per role makes every hazard a same-tensor dependency the compiler
    orders for us.

    **Trace size.** These allocations sit inside a Python-unrolled loop over query batches and
    s-tiles; naming them per iteration produced thousands of ``alloc_stack`` names and as many
    address computations. At 1M (16384 queries) that is the difference between a trace that
    compiles and one that does not.

    Peak is ``~130 KB/partition`` at 1M and ``~100 KB`` at 512k, against the ~245 KB budget:
    candidates ``cand_x * 6 B``, stage-1 scratch ``~5 * k * 4 B``, and the gather's fixed set.
    """
    n_blk_max = (cand_x + P_MAX - 1) // P_MAX
    # One tag per call: this runs once per S-TILE and alloc_stack names must be unique across
    # the whole trace (close_scope frees addresses, not names).
    t = _u()
    buf = {
        # Stage 1: snake source, its top-k outputs, and the position decode's scratch.
        "s1_src": sbm.alloc_stack((P_MAX, src_x_seg), nl.bfloat16, name=f"s1_src{t}"),
        "s1_val": sbm.alloc_stack((P_MAX, k), nl.bfloat16, name=f"s1_val{t}"),
        "s1_idx": sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"s1_idx{t}"),
        "s1_pos": sbm.alloc_stack((P_MAX, k), nl.float32, name=f"s1_pos{t}"),
        "s1_hi": sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"s1_hi{t}"),
        "s1_lo": sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"s1_lo{t}"),
        "s1_pidx": sbm.alloc_stack((P_MAX, 1), nl.uint32, name=f"s1_pidx{t}"),
        "s1_segoff_u": sbm.alloc_stack((P_MAX, 1), nl.uint32, name=f"s1_segoff_u{t}"),
        "s1_segoff_f": sbm.alloc_stack((P_MAX, 1), nl.float32, name=f"s1_segoff_f{t}"),
        # Candidate arrays: one snake per query, nseg * k slots wide.
        "cand_v": sbm.alloc_stack((P_MAX, cand_x), nl.bfloat16, name=f"cand_v{t}"),
        "cand_p": sbm.alloc_stack((P_MAX, cand_x), nl.float32, name=f"cand_p{t}"),
        # Merge outputs (shared by every level; k wide regardless of how many sets it absorbed).
        "m_val": sbm.alloc_stack((P_MAX, k), nl.bfloat16, name=f"m_val{t}"),
        "m_idx": sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"m_idx{t}"),
        # One-hot gather: transposed source blocks, index rows, and the PSUM-width staging.
        # FULL 128-column width, not 16, so one dma_transpose can cover all 8 queries at once --
        "oh_srct": sbm.alloc_stack((P_MAX, n_blk_max, P_MAX), nl.float32, name=f"oh_srct{t}", align=32),
        "oh_srct_b": sbm.alloc_stack((P_MAX, n_blk_max, P_MAX), nl.float32, name=f"oh_srctb{t}", align=32),
        "oh_widx": sbm.alloc_stack((1, k), nl.uint32, name=f"oh_widx{t}"),
        "oh_sidx": sbm.alloc_stack((1, k), nl.uint32, name=f"oh_sidx{t}"),
        "oh_gout": sbm.alloc_stack((1, k), nl.float32, name=f"oh_gout{t}"),
        "oh_flat_pos_i": sbm.alloc_stack((1, k), nl.int32, name=f"oh_flat_pos_i{t}"),
        "oh_col_u": sbm.alloc_stack((1, _PSUM_MOVING_F32), nl.uint32, name=f"oh_col_u{t}"),
        "oh_row_u": sbm.alloc_stack((1, _PSUM_MOVING_F32), nl.uint32, name=f"oh_row_u{t}"),
        "oh_col_bf": sbm.alloc_stack((1, _PSUM_MOVING_F32), nl.bfloat16, name=f"oh_col_bf{t}"),
        "oh_row_bf": sbm.alloc_stack((1, _PSUM_MOVING_F32), nl.bfloat16, name=f"oh_row_bf{t}"),
        "oh_col_hot": sbm.alloc_stack((P_MAX, _PSUM_MOVING_F32), nl.float32, name=f"oh_col_hot{t}"),
        "oh_row_sel": sbm.alloc_stack((P_MAX, _PSUM_MOVING_F32), nl.float32, name=f"oh_row_sel{t}"),
    }
    # Per-level intermediates, only for a multi-level plan (1M). The last level writes the
    # kernel output directly, so it needs none.
    n_sets = nseg
    for lvl, m in enumerate(plan):
        n_sets //= m
        if n_sets > 1:
            buf[f"mv{lvl}"] = sbm.alloc_stack((P_MAX, n_sets * k_cols), nl.bfloat16, name=f"mv{lvl}{t}")
            buf[f"ms{lvl}"] = sbm.alloc_stack((P_MAX, n_sets * k_cols), nl.float32, name=f"ms{lvl}{t}")
    return buf


def _gather_stage1_snakes(dst_sb, q_base, score_padded_hbm, sc_slot, T, spc, qs, seg, src_x_seg, seg_base, q_dyn=None):
    """Gather ``qs`` queries x ``spc`` position segments into the 8 snakes of one tile.

    Group ``g`` of the tile (partitions ``16g .. 16g+15``) is assigned query ``g // spc`` and
    position segment ``seg_base + g % spc``, so the 8 independent top-k the instruction already
    performs per issue cover (query, segment) pairs instead of 8 queries. That is the whole
    trick: the position split costs nothing to build because it IS the group wiring.

    A tile holds only 8 snakes, so one call covers at most 8 segments: ``spc = min(nseg, 8)``
    and ``qs = 8 // spc``. When ``nseg > 8`` the caller issues ``nseg // spc`` calls per query,
    passing ``seg_base = call_idx * spc`` -- that is the only thing that makes 512k (NSEG=16)
    and 1M (NSEG=32) reachable. ``seg_base`` shifts the source window; the matching offset is
    added back to the decoded position by :func:`_decode_stage1_positions`.

    The free axis stays contiguous (stride 1) for the same reason as :func:`_gather_batch` -- a
    round-robin free axis fails hardware DMA lowering. So within a segment, snake element ``i``
    is chunk ``i % 16``, offset ``i // 16``, and the position decode carries the matching
    ``(i % 16) * src_x_seg + (i // 16)`` remap plus the segment's base.

    Partition index is the row-major product of the leading three dims:
    ``j*(spc*16) + s*16 + p`` = ``16*(j*spc + s) + p`` = ``16g + p``.

    ``q_dyn``, when given, is the ROLLED query-batch index -- a ``fori_loop`` variable rather
    than a trace-time int (see :func:`topk_over_score_chunked`). It cannot go in ``offset``,
    which must be a Python int, so it rides ``scalar_offset``: the effective address is
    ``base + offset + scalar_offset * stride(shape[indirect_dim])``, and
    ``score_padded_hbm`` is ``(SCORE_RING, P_MAX, T)``, so ``indirect_dim=1`` has stride ``T``
    and contributes exactly the ``q_dyn * T`` the unrolled form folds into ``offset``.
    """
    pattern = [
        [T, qs],  # j: query, stride T
        [seg, spc],  # s: position segment within this call's window
        [src_x_seg, TOPK_QUERY_CHUNKS],  # p: contiguous chunk -> partition low bits
        [1, src_x_seg],  # c: column (free, contiguous)
    ]
    offset = (sc_slot * P_MAX + q_base) * T + seg_base * seg
    if q_dyn is None:
        src = score_padded_hbm.ap(pattern=pattern, offset=offset)
    else:
        src = score_padded_hbm.ap(pattern=pattern, offset=offset, scalar_offset=q_dyn, indirect_dim=1)
    nisa.dma_copy(dst=dst_sb[0:P_MAX, 0:src_x_seg], src=src, dge_mode=dge_mode.swdge)


def _decode_stage1_positions(buf, k, spc, seg, src_x_seg, seg_base):
    """Snake indices -> GLOBAL cache positions.

    Stage-1 group ``g = j*spc + s`` ranked segment ``seg_base + s``, so its snake index ``i``
    means position ``(seg_base + s)*seg + (i % 16) * src_x_seg + (i // 16)``. The last two terms
    are the remap the single-stage FLAT path also does; the segment base is what the position
    split adds.
    """
    idx_s1, pos_s1, hi, lo = buf["s1_idx"], buf["s1_pos"], buf["s1_hi"], buf["s1_lo"]
    pidx, segoff_u, segoff_f = buf["s1_pidx"], buf["s1_segoff_u"], buf["s1_segoff_f"]
    nisa.iota(dst=pidx, pattern=[[0, 1]], channel_multiplier=1)
    nisa.tensor_scalar(
        segoff_u[:P_MAX, 0:1],
        pidx[:P_MAX, 0:1],
        op0=nl.right_shift,
        operand0=4,
        op1=nl.bitwise_and,
        operand1=spc - 1,
    )
    nisa.tensor_scalar(
        segoff_u[:P_MAX, 0:1], segoff_u[:P_MAX, 0:1], op0=nl.add, operand0=seg_base, op1=nl.multiply, operand1=seg
    )
    nisa.tensor_copy(dst=segoff_f[:P_MAX, 0:1], src=segoff_u[:P_MAX, 0:1])
    nisa.tensor_scalar(hi[:P_MAX, :k], idx_s1[:P_MAX, :k], op0=nl.right_shift, operand0=4, engine=nisa.engine.vector)
    nisa.tensor_scalar(lo[:P_MAX, :k], idx_s1[:P_MAX, :k], op0=nl.bitwise_and, operand0=15, engine=nisa.engine.vector)
    nisa.tensor_scalar(
        pos_s1[:P_MAX, :k],
        lo[:P_MAX, :k],
        op0=nl.multiply,
        operand0=src_x_seg,
        op1=nl.add,
        operand1=segoff_f[:P_MAX, 0:1],
        engine=nisa.engine.scalar,
    )
    nisa.tensor_tensor(dst=pos_s1[:P_MAX, :k], data1=pos_s1[:P_MAX, :k], data2=hi[:P_MAX, :k], op=nl.add)


def _relayout_candidates(cand_v, cand_p, val_s1, pos_s1, k, spc, qs, q_off, seg_base):
    """Scatter one stage-1 call's 8 per-(query, segment) results into the per-QUERY snakes.

    Stage 1 leaves group ``g = jj*spc + s`` in partitions ``[16g, 16g+16)``, each holding its
    own ``k`` winners snake-arranged (rank ``r`` at partition ``16g + r%16``, column ``r//16``).
    The merge must rank all ``nseg*k`` of a query's candidates as a SINGLE snake, i.e. in
    partitions ``[16j, 16j+16)`` of ``cand_v`` where ``j = q_off + jj`` is the query's slot in
    the outer 8-query batch.

    Choosing candidate slot ``e = s_global*k + r`` makes the move a pure partition-block copy:
    since ``k % TOPK_QUERY_CHUNKS == 0``, ``e % 16 == r % 16`` (low partition bits are
    preserved) and ``e // 16 == s_global*(k//16) + r//16`` (the segment only shifts the column
    block). So each of the 8 groups is one contiguous ``[16, k//16]`` rectangle moved to a
    column offset -- no intra-partition shuffle, no transpose.

    ``cand_v``/``cand_p`` are owned by the caller and filled across ``nseg // spc`` calls, so
    the slot numbering stays global: this call's segment ``s`` is ``seg_base + s``.
    """
    k_cols = k // TOPK_QUERY_CHUNKS
    for jj in range(qs):
        for s in range(spc):
            g = jj * spc + s
            src_p = nl.ds(g * TOPK_QUERY_CHUNKS, TOPK_QUERY_CHUNKS)
            dst_p = nl.ds((q_off + jj) * TOPK_QUERY_CHUNKS, TOPK_QUERY_CHUNKS)
            cols = nl.ds((seg_base + s) * k_cols, k_cols)
            nisa.dma_copy(dst=cand_v[dst_p, cols], src=val_s1[src_p, 0:k_cols])
            nisa.dma_copy(dst=cand_p[dst_p, cols], src=pos_s1[src_p, 0:k_cols])


def _flatten_winner_idx(dst, r_idx, k, qp0):
    """Flatten one query's ``k`` snake-arranged winner indices to a single ``[1, k]`` row.

    Winner ``w`` sits at partition ``qp0 + w % 16``, column ``w // 16``, so a partition-minor
    walk visits them in ``w`` order. 16 DMAs (one per snake row), not ``k``.

    Separated from :func:`_onehot_gather` because a CHAINED gather (merge level -> original
    candidate array, which is how 1M resolves its two-level merge) already holds flat indices
    and must not flatten twice.
    """
    C = k // TOPK_QUERY_CHUNKS
    for rp in range(TOPK_QUERY_CHUNKS):
        nisa.dma_copy(dst=dst[0:1, nl.ds(rp * C, C)], src=r_idx[nl.ds(qp0 + rp, 1), 0:C])


def _pretranspose_snake(buf, key, src_sb, src_x, col_off=0):
    """Transpose ALL 128 partitions of a candidate snake once, for every query in the batch.

    Hoisted out of :func:`_onehot_gather`: a per-query ``[16, w] -> [w, 16]`` transpose writes each
    destination row from 16 different source partitions, so every descriptor moves a single fp32.

    Transposing the whole tile instead gives destination rows of 128 fp32 = 512-byte runs and 8x
    fewer calls. Each query then slices its own 16 columns out of the shared result.
    """
    src_t = buf[key]
    n_blk = (src_x + P_MAX - 1) // P_MAX
    for b in range(n_blk):
        w = min(P_MAX, src_x - b * P_MAX)
        nisa.dma_transpose(
            dst=src_t[0:w, b, 0:P_MAX],
            src=src_sb[0:P_MAX, nl.ds(col_off + b * P_MAX, w)],
        )


def _onehot_gather(buf, out, src_sb, qp0, flat_i, k, src_x, consts, col_off=0, src_t_key="oh_srct"):
    """``out[0, w] = src_sb[query snake at qp0, slot flat_i[0, w]]`` via one-hot matmuls.

    See :func:`_lookup_positions_onehot` for the shape reasoning; this is that function's
    gather half, factored out so it can run twice per query. ``src_sb`` is fp32 and holds one
    query's ``16 x src_x`` snake at partitions ``[qp0, qp0+16)``, columns
    ``[col_off, col_off + src_x)``; ``flat_i`` is a ``[1, k]`` uint32 row of slot numbers into
    that snake. Returns ``[1, k]`` fp32.
    """
    n_blk = (src_x + P_MAX - 1) // P_MAX
    iota_f, ones_row_bf, ones_col = consts
    src_t = buf[src_t_key]

    col_u, row_u = buf["oh_col_u"], buf["oh_row_u"]
    col_bf, row_bf = buf["oh_col_bf"], buf["oh_row_bf"]
    col_hot, row_sel = buf["oh_col_hot"], buf["oh_row_sel"]

    for w0 in range(0, k, _PSUM_MOVING_F32):
        n_w = min(_PSUM_MOVING_F32, k - w0)
        wt = nl.ds(w0, n_w)
        nisa.tensor_scalar(col_u[0:1, 0:n_w], flat_i[0:1, wt], op0=nl.right_shift, operand0=4)
        nisa.tensor_scalar(row_u[0:1, 0:n_w], flat_i[0:1, wt], op0=nl.bitwise_and, operand0=15)
        nisa.tensor_copy(dst=row_bf[0:1, 0:n_w], src=row_u[0:1, 0:n_w])

        """
        Column select, accumulated over 128-column blocks.

        The block base is subtracted BEFORE the bf16 cast, so every value that reaches the
        comparison is a block-LOCAL column in [0, 128). nc_matmul's operand must be bf16, and
        bf16 has 8 mantissa bits -- integers are exact only to 256. src_x = nseg * k/16
        reaches 512 at the target index_topk of 2048 (2048 at 512k, 4096 at 1M), so casting a
        GLOBAL column index would round anything above 256 to a neighbour and the one-hot would
        select the wrong column. That is invisible at index_topk 512 (src_x = 128, all columns
        exact).

        Columns outside block b land outside [0, 128) after the subtraction and so match no
        partition; a rounded value >= 256 cannot round into [0, 128) either. Exactly one block
        therefore contributes, and the PSUM accumulation across blocks selects it.
        """
        m_ps = nl.ndarray((P_MAX, _PSUM_MOVING_F32), dtype=nl.float32, buffer=nl.psum)
        for b in range(n_blk):
            w = min(P_MAX, src_x - b * P_MAX)
            # Subtract straight into the bf16 matmul operand -- the cast on the result is
            # free, so no separate tensor_copy is needed.
            # add of a NEGATIVE rather than subtract: the Scalar engine supports multiply+add,
            # multiply and add, but not subtract. Arithmetically identical modulo 2^32 on the uint32
            # data, so columns outside block b still land outside [0, 128) and match no iota entry --
            # which is what the accumulate-over-blocks relies on.
            nisa.tensor_scalar(
                dst=col_bf[0:1, 0:n_w],
                data=col_u[0:1, 0:n_w],
                op0=nl.add,
                operand0=-(b * P_MAX),
                engine=nisa.engine.scalar,
            )
            col_b = nl.ndarray((P_MAX, _PSUM_MOVING_F32), dtype=nl.float32, buffer=nl.psum)
            nisa.nc_matmul(dst=col_b[0:w, 0:n_w], stationary=ones_row_bf[0:1, 0:w], moving=col_bf[0:1, 0:n_w])
            nisa.tensor_scalar(dst=col_hot[0:w, 0:n_w], data=col_b[0:w, 0:n_w], op0=nl.equal, operand0=iota_f[0:w, 0:1])
            nisa.nc_matmul(
                dst=m_ps[0:TOPK_QUERY_CHUNKS, 0:n_w],
                stationary=src_t[0:w, b, nl.ds(qp0, TOPK_QUERY_CHUNKS)],
                moving=col_hot[0:w, 0:n_w],
                accumulate=(b > 0),
            )

        # Row select over the snake's 16 rows, then collapse the partition axis.
        row_b = nl.ndarray((P_MAX, _PSUM_MOVING_F32), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=row_b[0:TOPK_QUERY_CHUNKS, 0:n_w],
            stationary=ones_row_bf[0:1, 0:TOPK_QUERY_CHUNKS],
            moving=row_bf[0:1, 0:n_w],
        )
        nisa.tensor_scalar(
            dst=row_sel[0:TOPK_QUERY_CHUNKS, 0:n_w],
            data=row_b[0:TOPK_QUERY_CHUNKS, 0:n_w],
            op0=nl.equal,
            operand0=iota_f[0:TOPK_QUERY_CHUNKS, 0:1],
        )
        nisa.tensor_tensor(
            dst=row_sel[0:TOPK_QUERY_CHUNKS, 0:n_w],
            data1=m_ps[0:TOPK_QUERY_CHUNKS, 0:n_w],
            data2=row_sel[0:TOPK_QUERY_CHUNKS, 0:n_w],
            op=nl.multiply,
        )
        sel = nl.ndarray((1, _PSUM_MOVING_F32), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=sel[0:1, 0:n_w],
            stationary=ones_col[0:TOPK_QUERY_CHUNKS, 0:1],
            moving=row_sel[0:TOPK_QUERY_CHUNKS, 0:n_w],
        )
        nisa.tensor_copy(dst=out[0:1, wt], src=sel[0:1, 0:n_w])


def _lookup_positions_onehot(
    buf,
    cand_p,
    r_idx,
    k,
    qpc,
    src_x2,
    topk_flat_hbm,
    q_row_base,
    q_row_hi,
    consts,
    slot_sb=None,
    slot_x=0,
    q_dyn=None,
):
    """``out[j, w] = cand_p[query j, slot(r_idx[j, w])]`` via one-hot matmuls on the PE array.

    Deliberately NOT ``nisa.nc_n_gather``, even though a per-partition gather is the obvious
    primitive. ``chunked_topk`` documents that gather as unusable for exactly this lookup: its
    ucode signals completion before its stores are visible, and four escalating fence
    constructions (a flow-dependent Vector consumer 500 ns downstream, an SBUF->SBUF DMA fence,
    a GpSIMD copy fence the compiler copy-propagated away, and an unfoldable cast fence) each
    still read stale data. It is also only 8 columns per issue -- 256 issues at k=2048.

    Three shape rules keep this cheap:

    1. **Flatten the winners to one partition row.** A winner tile is [16, k/16] (snake), so
       both indices of the lookup vary per element. Flattened to ``[1, k]`` the one-hot
       broadcast has a single moving operand and the tile finishes in ``k/512`` PSUM-width
       passes; looping the 16 partitions instead would multiply every matmul by 16.
    2. **Split the slot the way the snake already splits it**: ``col = e // 16``,
       ``row = e % 16``. A column select then a 16-wide row select, never one one-hot over all
       ``nseg*k`` slots.
    3. **Block the column select at 128.** ``src_x2`` reaches 512 at 128k, past the partition
       dim a one-hot can span, so the select runs per 128-column block and ACCUMULATES in PSUM.
       Exactly one block holds the true column, so the sum selects it -- no masking needed.

    Cost: ``2 + ceil(src_x2/128)`` matmuls per 512 winners per query.
    """
    widx, sidx = buf["oh_widx"], buf["oh_sidx"]
    gout, flat_pos_i = buf["oh_gout"], buf["oh_flat_pos_i"]

    # ONCE for the whole batch, not once per query: see _pretranspose_snake.
    _pretranspose_snake(buf, "oh_srct", cand_p, src_x2)
    if slot_sb is not None:
        _pretranspose_snake(buf, "oh_srct_b", slot_sb, slot_x)

    for j in range(qpc):
        qp0 = j * TOPK_QUERY_CHUNKS
        _flatten_winner_idx(widx, r_idx, k, qp0)
        sel_i = widx
        if slot_sb is not None:
            """
            CHAINED resolve, used only by the multi-level merge (1M).

            The final merge ranked candidates whose slots index the LAST level's array, not the
            stage-1 one, so the winner's slot must be translated before the position lookup:
            ``slot_sb`` holds, per level-input slot, the slot it came from in ``cand_p``. It is
            built ARITHMETICALLY when the level's input is stage 1 (slot = o*m*k + idx), so the
            only gathers ever paid are these two -- on the FINAL k winners, not on every
            intermediate level's k winners.
            """
            _onehot_gather(buf, gout, slot_sb, qp0, widx, k, slot_x, consts, src_t_key="oh_srct_b")
            nisa.tensor_copy(dst=sidx[0:1, 0:k], src=gout[0:1, 0:k])
            sel_i = sidx
        _onehot_gather(buf, gout, cand_p, qp0, sel_i, k, src_x2, consts)

        """
        Cast to the int32 FLAT contract and write this query's k positions in one DMA.

        No scatter back to the [16, C] snake: the FLAT consumer takes a flat unordered row.

        Two addressing forms. Unrolled: the row is a trace-time int, and the ``q_row_hi`` guard
        drops the tail rows of a short final batch. Rolled (``q_dyn``): the row is
        ``q_dyn + j`` with ``q_dyn`` a ``fori_loop`` variable, so it rides ``scalar_offset``
        on dim 0 (stride ``k``) and the static ``j`` part stays in ``offset``. The guard is gone
        there by construction -- the rolled path is only taken when the query range divides
        evenly into batches, so every row it writes is in range (asserted by the caller).
        """
        nisa.tensor_copy(dst=flat_pos_i[0:1, 0:k], src=gout[0:1, 0:k])
        if q_dyn is not None:
            nisa.dma_copy(
                dst=topk_flat_hbm.ap(
                    pattern=[[k, 1], [1, k]],
                    offset=(q_row_base + j) * k,
                    scalar_offset=q_dyn,
                    indirect_dim=0,
                ),
                src=flat_pos_i[0:1, 0:k],
            )
        elif q_row_base + j < q_row_hi:
            nisa.dma_copy(dst=topk_flat_hbm[nl.ds(q_row_base + j, 1), 0:k], src=flat_pos_i[0:1, 0:k])


def topk_over_score_chunked(
    sbm,
    S,
    end_pos,
    index_topk,
    score_padded_hbm,
    s_tile_idx,
    topk_flat_hbm,
    sc_slot=None,
    s_tile_base_q=0,
    batch_start=0,
    batch_end=NUM_TOPK_BATCHES,
    part_offset=0,
    topk_nseg=0,
):
    """Multi-stage top-k for contexts past ``nisa.topk``'s ``n < 65536`` ceiling.

    Stage 1 spends the tile's 8 snakes on POSITION segments rather than 8 queries, so each
    snake ranks ``seg`` positions (see :func:`topk_geometry`); the merge stage then
    reduces the ``NSEG * k`` survivors per query down to ``k``. Correct because a global top-k
    element is necessarily in the top-k of whichever segment holds it.

    Three geometries stack up, and they are deliberately independent:

    * ``spc = min(NSEG, 8)`` position segments and ``qs = 8 // spc`` queries per stage-1 CALL.
      A tile holds only 8 snakes, so 8 is the hard cap on segments per call -- past it stage 1
      issues ``NSEG // spc`` calls per query with a shifted segment window. That is what lifts
      the old ``NSEG <= 8`` ceiling (which came from ``qpc = 8 // NSEG`` reaching 0) and makes
      512k (NSEG=16) and 1M (NSEG=32) reachable at all.
    * ``QB = 8`` queries per OUTER iteration, independent of the above. Candidates live in
      ``cand_v``/``cand_p`` at partitions ``[16j, 16j+16)``, so 8 queries fill all 128
      partitions and the merge ranks 8 snakes per issue -- one GpSIMD core each. The previous
      code tied the outer batch to ``8 // NSEG``, which at NSEG >= 8 left 7 of the 8 cores idle
      in every merge call.
    * The MERGE is a tree, ``_merge_plan(NSEG, k)``: one level through 512k, two at 1M.

    Perf notes:
      * The instruction's ``4*n`` term is invariant under the split -- the same elements are
        ranked either way -- so the split multiplies only the per-call ``+K``. NSEG is chosen
        as small as legality allows for that reason.
      * No descending sort: attention gathers all k selected positions, so their order is
        irrelevant. That skips the bitonic network chunked_topk needs for its sorted contract.

    Emits FLAT decoded positions only. The single-stage SNAKE contract cannot be reproduced --
    it hands the consumer a raw snake index to remap with one global ``src_x``, and after a
    position split there is no single such mapping. FLAT is what
    ``sparse_mla_latent_attn_cte``'s ``topk_indices`` wants anyway.

    ``topk_nseg`` (0 = auto) forces the segment count; see :func:`topk_geometry`.
    """
    k = index_topk
    sc_slot = s_tile_idx if sc_slot is None else sc_slot
    T, nseg, seg, spc = topk_geometry(end_pos, topk_nseg)
    qs = TOPK_QUERIES_PER_BATCH // spc  # queries per stage-1 call
    n_s1_calls = nseg // spc  # stage-1 calls per query group
    src_x_seg = seg // TOPK_QUERY_CHUNKS
    k_cols = k // TOPK_QUERY_CHUNKS
    cand_x = nseg * k_cols  # candidate-array width, in columns
    plan = _merge_plan(nseg, k)

    kernel_assert(topk_flat_hbm != None, "chunked topk emits FLAT positions; pass emit_flat_topk=True")
    kernel_assert(k % TOPK_QUERY_CHUNKS == 0, f"chunked topk needs k % {TOPK_QUERY_CHUNKS} == 0, got k={k}")
    kernel_assert(k <= seg, f"stage-1 topk needs k <= n; k={k}, segment width={seg}")
    kernel_assert(
        n_s1_calls * spc == nseg,
        f"segments per call must divide nseg; nseg={nseg} spc={spc}",
    )

    sbm.open_scope(name=f"topk_chunked{_u()}")
    consts = make_onehot_consts(sbm)
    buf = _alloc_chunked_scratch(sbm, k, src_x_seg, cand_x, k_cols, plan, nseg)
    """
    Query range, in ABSOLUTE s-tile rows.

    ``S`` is this core's LOCAL query count (the caller passes ``sc_S``), while q_lo/q_hi index the
    s-tile's rows absolutely, so the upper bound is ``part_offset + S`` -- the end of the range
    this core owns -- NOT ``S``. Clamping with ``S`` gave core 1 ``range(64, 64)``: an empty loop,
    so it emitted nothing and its rows read back as zeros. That looked segment-selective rather
    than missing, because a zero passes segment 0's ``[0, seg)`` range check and fails segment
    1's, and it was invisible at LNC=1 where part_offset is 0 and S is the full 128.
    """
    q_lo = batch_start * TOPK_QUERIES_PER_BATCH
    q_hi = min(batch_end * TOPK_QUERIES_PER_BATCH, part_offset + S)
    cand_v, cand_p = buf["cand_v"], buf["cand_p"]

    q_lo = batch_start * TOPK_QUERIES_PER_BATCH
    q_hi = min(batch_end * TOPK_QUERIES_PER_BATCH, part_offset + S)
    cand_v, cand_p = buf["cand_v"], buf["cand_p"]

    """
    Query-batch loop: UNROLLED in Python, or ROLLED with nl.fori_loop.

    Unrolled, this loop is the trace-size multiplier that makes the long shapes uncompilable.
    Its body is ~200 instructions and it runs ``s_tiles * (q_hi - q_lo) / 8`` times -- 4096 copies
    of the lookup at 512k -- and the 512k attempt died in BIR emission with
    "body blob exceeds uint32 offset range" after 93 minutes. Not a timeout: the emitted program
    overflowed a 32-bit offset field, so no amount of waiting or memory would have helped.

    ``nl.fori_loop`` is the only construct that actually keeps ONE body in the trace.
    ``nl.affine_range`` / ``sequential_range`` / ``static_range`` are all documented as "fully
    unrolled ... alias for Python's range()" and are deprecated -- they would change nothing.

    Rolling costs no overlap here: the whole chunked path shares one scratch set
    (:func:`_alloc_chunked_scratch`), so WAR dependencies on those buffers already serialized
    consecutive batches regardless of which engines the body touches. The score path's
    ``interleave_degree=2`` double-buffering lives in the caller and is untouched.

    Only ``q0`` is rolled. The inner 8-query loop cannot be: its index scales an SBUF PARTITION
    base (``qp0 = j*16``), and partition addressing is static -- a 16-partition access must start
    at 0/32/64/96. The s-tile loop is a further 64x but drives ``sc_slot`` and the interleave
    scheme, which are allocation-time decisions.

    Rolled whenever legal; see the divisibility clause below.
    """
    n_batches = (q_hi - q_lo) // TOPK_QUERIES_PER_BATCH
    # Roll whenever it is LEGAL, with no nseg gate. The divisibility clause is load-bearing -- the
    # rolled body assumes every batch is full (nq = 8 statically) -- but the original's extra
    # "nseg >= 16" was conservatism that made 256k the trace worst case; see the module docstring.
    roll = (q_hi - q_lo) % TOPK_QUERIES_PER_BATCH == 0 and n_batches > 1

    def _query_batch(q0, q_dyn):
        """One batch of up to 8 queries. ``q_dyn`` is None when ``q0`` is a trace-time int."""
        # Static in the rolled path: the range divides evenly, so every batch is full.
        nq = TOPK_QUERIES_PER_BATCH if q_dyn is not None else min(TOPK_QUERIES_PER_BATCH, q_hi - q0)
        np_use = nq * TOPK_QUERY_CHUNKS  # partitions actually holding candidates

        # ---- Stage 1: one snake per (query, position segment), over n_s1_calls windows ----
        for q_off in range(0, nq, qs):
            for c in range(n_s1_calls):
                seg_base = c * spc
                _gather_stage1_snakes(
                    buf["s1_src"],
                    q0 + q_off,
                    score_padded_hbm,
                    sc_slot,
                    T,
                    spc,
                    qs,
                    seg,
                    src_x_seg,
                    seg_base,
                    q_dyn=q_dyn,
                )
                nisa.topk(
                    val_dst=buf["s1_val"][:P_MAX, :k],
                    idx_dst=buf["s1_idx"][:P_MAX, :k],
                    src=buf["s1_src"][:P_MAX, :src_x_seg],
                    n=seg,
                )
                _decode_stage1_positions(buf, k, spc, seg, src_x_seg, seg_base)
                _relayout_candidates(cand_v, cand_p, buf["s1_val"], buf["s1_pos"], k, spc, qs, q_off, seg_base)

        # ---- Merge tree: nseg sets of k -> k, plan[lvl] sets absorbed per call ----
        cur_v, cur_slot, cur_x = cand_v, None, cand_x
        n_sets = nseg
        for lvl, m in enumerate(plan):
            n_out = n_sets // m
            in_x = m * k_cols  # columns of one merge call's input snake
            val_m, idx_m = buf["m_val"], buf["m_idx"]
            for o in range(n_out):
                nisa.topk(
                    val_dst=val_m[:np_use, :k],
                    idx_dst=idx_m[:np_use, :k],
                    src=cur_v[:np_use, nl.ds(o * in_x, in_x)],
                    n=m * k,
                )
                if n_out == 1:
                    # Final level: resolve winners to positions and write the FLAT output.
                    _lookup_positions_onehot(
                        buf,
                        cand_p,
                        idx_m,
                        k,
                        nq,
                        cand_x,
                        topk_flat_hbm,
                        s_tile_base_q + q0,
                        s_tile_base_q + q_hi,
                        consts,
                        slot_sb=cur_slot,
                        slot_x=cur_x,
                        q_dyn=q_dyn,
                    )
                else:
                    """
                    Intermediate level: carry values and ORIGINATING SLOTS, not positions.

                    Values move by a plain column-block copy: winner ``r`` of output set ``o``
                    takes slot ``e = o*k + r``, and ``k % 16 == 0`` makes ``e % 16 == r % 16``,
                    so the snake's partition assignment is unchanged.

                    The slot array is pure arithmetic -- ``idx_m`` indexes this call's input
                    window, whose slot ``i`` is input-array slot ``o*m*k + i``, and when the input
                    IS stage 1 that is already the ``cand_p`` slot. So an intermediate level pays
                    NO gather; only the final winners do (twice, chained). That is the whole
                    reason a two-level merge costs one extra small top-k plus one extra gather
                    rather than a position lookup per level.
                    """
                    new_v, new_slot = buf[f"mv{lvl}"], buf[f"ms{lvl}"]
                    cols = nl.ds(o * k_cols, k_cols)
                    nisa.tensor_copy(dst=new_v[:np_use, cols], src=val_m[:np_use, 0:k_cols])
                    if cur_slot is None:
                        nisa.tensor_scalar(
                            dst=new_slot[:np_use, cols],
                            data=idx_m[:np_use, 0:k_cols],
                            op0=nl.add,
                            operand0=o * m * k,
                        )
                    else:
                        """
                        Three or more merge levels (past 1M): the input slots are themselves
                        indirect, so this level's slots need the same one-hot gather the final
                        resolve uses, then a scatter back to the snake. Unreachable for every
                        shape this kernel ships -- 1M is two levels -- and left in so the plan
                        loop is total rather than silently wrong at the next context step.
                        """
                        widx, gout = buf["oh_widx"], buf["oh_gout"]
                        _pretranspose_snake(buf, "oh_srct", cur_slot, cur_x)
                        for j in range(nq):
                            qp0 = j * TOPK_QUERY_CHUNKS
                            _flatten_winner_idx(widx, idx_m, k, qp0)
                            nisa.tensor_scalar(widx[0:1, 0:k], widx[0:1, 0:k], op0=nl.add, operand0=o * m * k)
                            _onehot_gather(buf, gout, cur_slot, qp0, widx, k, cur_x, consts)
                            for rp in range(TOPK_QUERY_CHUNKS):
                                nisa.dma_copy(
                                    dst=new_slot[nl.ds(qp0 + rp, 1), cols],
                                    src=gout[0:1, nl.ds(rp * k_cols, k_cols)],
                                )
            if n_out > 1:
                cur_v, cur_slot, cur_x = buf[f"mv{lvl}"], buf[f"ms{lvl}"], n_out * k_cols
            n_sets = n_out

    if roll:
        # fori_loop hands the body its induction value, so stepping by 8 makes it q0 itself --
        # no register arithmetic to reconstruct the batch base.
        kernel_assert(
            n_batches * TOPK_QUERIES_PER_BATCH == q_hi - q_lo,
            f"rolled query loop needs the range to divide evenly; q_lo={q_lo} q_hi={q_hi}",
        )
        nl.fori_loop(q_lo, q_hi, lambda i: _query_batch(0, i), step=TOPK_QUERIES_PER_BATCH)
    else:
        for q0 in range(q_lo, q_hi, TOPK_QUERIES_PER_BATCH):
            _query_batch(q0, None)
    sbm.close_scope()


# --------------------------------------------------------------------------------------------
# Chunked score + accumulated candidates (stage 8)
# --------------------------------------------------------------------------------------------


def alloc_chunk_candidates(cfg, n_chunks: int, k: int, group_q: int):
    """HBM candidate arrays for a CHUNKED score: ``[group_q, n_chunks * k]`` values and positions.

    This is what lets the score ring shrink from "every tile's whole row" to two chunk-sized slots.
    Today the ring is ``tiles_per_core * P_MAX * T``, which is 16.4 GB at cp64/1M because the top-k
    runs after all groups and so every tile's score must persist. Ranking each chunk as it is produced
    means only the chunk being scored and the one being ranked need to exist -- 32 MB -- and what
    survives instead is k candidates per query per chunk.

    Sized per GROUP, not per core: a chunk's score covers every tile in the panel, so candidates for
    the whole group accumulate together. That makes the panel size a MEMORY knob as well as a K-traffic
    one -- at 1M, k=2048, 17 chunks it is 25.5 MB per tile, so panel 3 fits a 192 MB budget and panel 6
    does not.
    """
    n_cand = n_chunks * k
    cand_v = nl.ndarray((group_q, n_cand), dtype=nl.bfloat16, buffer=nl.private_hbm, name="cand_v_hbm")
    cand_p = nl.ndarray((group_q, n_cand), dtype=nl.int32, buffer=nl.private_hbm, name="cand_p_hbm")
    return cand_v, cand_p


def topk_chunk_stage1(
    sbm,
    score_ring,
    slot: int,
    cand_v_hbm,
    cand_p_hbm,
    q_lo: int,
    n_q: int,
    chunk_idx: int,
    chunk: int,
    k: int,
    cand_row_q0: int,
) -> None:
    """Rank ONE chunk for ``n_q`` queries and append their k winners to the candidate arrays.

    ``score_ring[slot]`` is ``[P_MAX, chunk]`` holding this chunk's scores for a tile's queries.

    The gather is the existing stage-1 snake with ``seg == chunk == T``, i.e. spc=1 and 8 queries per
    call. That form LOWERS here even though it does not in the segmented path: the failing case reads
    ``seg`` elements from each of 8 rows separated by gaps of ``T - seg``, and when seg == T there is no
    gap, so 8 adjacent query rows are one contiguous run.

    Positions come out ABSOLUTE via the decode's segment base: passing ``seg_base = chunk_idx`` with
    spc=1 makes the offset ``chunk_idx * chunk``.
    """
    src_x = chunk // TOPK_QUERY_CHUNKS
    k_cols = k // TOPK_QUERY_CHUNKS
    kernel_assert(
        chunk <= SEG_MAX,
        f"chunk ({chunk}) must not exceed {SEG_MAX}: one chunk is ONE nisa.topk call, which is the "
        f"point -- it removes segmentation, its merge tree and the SEG_ALIGN parity rule entirely",
    )
    kernel_assert(src_x % 2 == 0, f"chunk/16 ({src_x}) must be even or nisa.topk fails [NCC_IXCG164]")

    def _batch(qb, qb_dyn):
        """One batch of 8 queries. ``qb_dyn`` is None when ``qb`` is a trace-time int.

        ROLLED when it legally can be. This loop used to be a plain Python range, which emitted
        ``n_q / 8`` bodies PER CHUNK PER TILE: at 1M with the production s_local=16384 that is
        17 chunks x 16 batches x 64 tiles/core = 17408 unrolled bodies, and neuronx-cc is
        OOM-killed building it ("[F137] neuronx-cc was forcibly killed"). The segmented path
        already rolls the identical loop; this is the same fix.

        Every rolled batch is FULL (nq_b == 8), asserted by the caller, so rows_out is static.
        """
        nq_b = TOPK_QUERIES_PER_BATCH if qb_dyn is not None else min(TOPK_QUERIES_PER_BATCH, n_q - qb)
        sbm.open_scope(name="chunk_s1")
        src = sbm.alloc_stack((P_MAX, src_x), nl.bfloat16, name=f"cs_src{_u()}")
        val = sbm.alloc_stack((P_MAX, k), nl.bfloat16, name=f"cs_val{_u()}")
        idx = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"cs_idx{_u()}")
        pos = sbm.alloc_stack((P_MAX, k), nl.int32, name=f"cs_pos{_u()}")

        # 8 queries as 8 snakes over the WHOLE chunk. Contiguous because seg == T.
        # Rolled: the batch base rides scalar_offset (units of the leading stride, i.e. rows of
        # `chunk`), and the static q_lo part stays in `offset`.
        src_ap = (
            score_ring[slot].ap(
                pattern=[[chunk, TOPK_QUERIES_PER_BATCH], [src_x, TOPK_QUERY_CHUNKS], [1, src_x]],
                offset=q_lo * chunk,
                scalar_offset=qb_dyn,
                indirect_dim=0,
            )
            if qb_dyn is not None
            else score_ring[slot].ap(
                pattern=[[chunk, TOPK_QUERIES_PER_BATCH], [src_x, TOPK_QUERY_CHUNKS], [1, src_x]],
                offset=(q_lo + qb) * chunk,
            )
        )
        nisa.dma_copy(dst=src[:P_MAX, :src_x], src=src_ap, dge_mode=dge_mode.swdge)
        nisa.topk(val_dst=val[:P_MAX, :k], idx_dst=idx[:P_MAX, :k], src=src[:P_MAX, :src_x], n=chunk)
        _decode_chunk_positions(sbm, idx, pos, src_x, chunk_idx * chunk, k)

        # Append at this chunk's column block. One DMA each; the snake rows of a query are adjacent
        # in the candidate array, so the destination AP collapses the same way the score write does.
        rows_out = nq_b * TOPK_QUERY_CHUNKS
        n_cand = cand_v_hbm.shape[1]
        cand_cols = n_cand // TOPK_QUERY_CHUNKS
        for dst_hbm, src_sb in ((cand_v_hbm, val), (cand_p_hbm, pos)):
            if qb_dyn is not None:
                """
                Addressed on the UN-RESHAPED [group_q, n_cand] tensor, whose row IS one query.

                scalar_offset is added in units of the TENSOR's row stride -- not the pattern's
                leading stride. Reshaping to [group_q * 16, cand_cols] first makes a row one SNAKE
                ROW, and a query owns TOPK_QUERY_CHUNKS of them, so the induction value advanced 16x
                too little and consecutive batches overlapped. That showed up as duplicate winners
                ("row 0: only 2034 unique positions") with the score and mask still exact, and
                restating the pattern's leading extent did not change it -- which is what identified
                the units. The working precedent is _lookup_positions_onehot's topk_flat_hbm, a
                [s_local, k] tensor whose row is likewise exactly one query.

                Within a query's row the 16 snake rows sit at stride cand_cols, so the inner two
                pattern dims walk them.
                """
                dst_ap = dst_hbm.ap(
                    pattern=[[n_cand, TOPK_QUERIES_PER_BATCH], [cand_cols, TOPK_QUERY_CHUNKS], [1, k_cols]],
                    offset=cand_row_q0 * n_cand + chunk_idx * k_cols,
                    scalar_offset=qb_dyn,
                    indirect_dim=0,
                )
            else:
                flat = dst_hbm.reshape((dst_hbm.shape[0] * TOPK_QUERY_CHUNKS, cand_cols))
                base = (cand_row_q0 + qb) * TOPK_QUERY_CHUNKS
                dst_ap = flat[base : base + rows_out, chunk_idx * k_cols : (chunk_idx + 1) * k_cols]
            nisa.dma_copy(dst=dst_ap, src=src_sb[:rows_out, :k_cols], dge_mode=dge_mode.swdge)
        sbm.close_scope()

    n_batches = n_q // TOPK_QUERIES_PER_BATCH
    tail_q = n_q - n_batches * TOPK_QUERIES_PER_BATCH
    if n_batches > 1 and tail_q == 0:
        nl.fori_loop(0, n_q, lambda i: _batch(0, i), step=TOPK_QUERIES_PER_BATCH)
    else:
        for b in range(div_ceil(n_q, TOPK_QUERIES_PER_BATCH)):
            _batch(b * TOPK_QUERIES_PER_BATCH, None)


def _decode_chunk_positions(sbm, idx_sb, pos_sb, src_x: int, base: int, k: int) -> None:
    """Snake index -> absolute position: ``pos = (i & 15) * src_x + (i >> 4) + base``."""
    sbm.open_scope(name="cs_decode")
    hi = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"cd_hi{_u()}")
    lo = sbm.alloc_stack((P_MAX, k), nl.uint32, name=f"cd_lo{_u()}")
    nisa.tensor_scalar(hi[:P_MAX, :k], idx_sb[:P_MAX, :k], op0=nl.right_shift, operand0=4)
    nisa.tensor_scalar(lo[:P_MAX, :k], idx_sb[:P_MAX, :k], op0=nl.bitwise_and, operand0=15)
    if base:
        nisa.tensor_scalar(hi[:P_MAX, :k], hi[:P_MAX, :k], op0=nl.add, operand0=base)
    nisa.scalar_tensor_tensor(
        dst=pos_sb[:P_MAX, :k],
        data=lo[:P_MAX, :k],
        op0=nl.multiply,
        operand0=src_x,
        op1=nl.add,
        operand1=hi[:P_MAX, :k],
    )
    sbm.close_scope()


def topk_chunk_merge(
    sbm,
    cand_v_hbm,
    cand_p_hbm,
    topk_flat_hbm,
    n_chunks: int,
    chunk: int,
    k: int,
    cand_row_q0: int,
    n_q: int,
    q_row_base: int,
) -> None:
    """Merge the per-chunk candidate sets into the final k, and resolve their positions.

    Structurally the SECOND HALF of :func:`topk_over_score_chunked`: n_chunks sets of k per query,
    reduced by the same merge tree, with the same one-hot position lookup. The only difference is
    where the candidates came from -- accumulated chunk by chunk as the score was produced, rather
    than read out of a full-width score row.

    The merge work is LOWER than the segmented path it replaces. At 1M the segmented geometry derives
    nseg=24, so 24 sets; chunking at 65504 gives 17. And within a chunk nothing is segmented at all,
    so SEG_ALIGN parity, the segment-per-call packing and the two-level tree stop applying.
    """
    k_cols = k // TOPK_QUERY_CHUNKS
    cand_x = n_chunks * k_cols
    src_x_chunk = chunk // TOPK_QUERY_CHUNKS
    plan = _merge_plan(n_chunks, k)
    kernel_assert(
        n_chunks * k < TOPK_N_MAX or len(plan) > 1,
        f"{n_chunks} chunks x k={k} candidates needs a merge plan; got {plan}",
    )

    sbm.open_scope(name="chunk_merge")
    consts = make_onehot_consts(sbm)
    buf = _alloc_chunked_scratch(sbm, k, src_x_chunk, cand_x, k_cols, plan, n_chunks)
    cand_v, cand_p = buf["cand_v"], buf["cand_p"]
    n_cand_cols = cand_v_hbm.shape[1] // TOPK_QUERY_CHUNKS

    def _merge_batch(qb, qb_dyn):
        """One batch of 8 queries. ``qb_dyn`` is None when ``qb`` is a trace-time int.

        ROLLED when legal. Unrolled this emitted n_q/8 bodies PER TILE, and each body carries the
        one-hot position gather -- ~19 matmuls x 4 PSUM passes x 8 queries. At 1M with the production
        s_local=16384 that is 16 batches x 64 tiles/core, and neuronx-cc is OOM-killed building it
        ("[F137] neuronx-cc was forcibly killed"). Rolling is the same fix the segmented path already
        applies to its own query-batch loop.
        """
        nq_b = TOPK_QUERIES_PER_BATCH if qb_dyn is not None else min(TOPK_QUERIES_PER_BATCH, n_q - qb)
        np_use = nq_b * TOPK_QUERY_CHUNKS

        # Candidates back from HBM. The snake rows of a query are adjacent, so this is one DMA each.
        # Rolled: addressed on the UN-RESHAPED [group_q, n_cand] tensor, because scalar_offset counts
        # the TENSOR's rows and only there is a row exactly one query -- see topk_chunk_stage1.
        for dst_sb, src_hbm in ((cand_v, cand_v_hbm), (cand_p, cand_p_hbm)):
            if qb_dyn is not None:
                src_ap = src_hbm.ap(
                    pattern=[
                        [src_hbm.shape[1], TOPK_QUERIES_PER_BATCH],
                        [n_cand_cols, TOPK_QUERY_CHUNKS],
                        [1, cand_x],
                    ],
                    offset=cand_row_q0 * src_hbm.shape[1],
                    scalar_offset=qb_dyn,
                    indirect_dim=0,
                )
            else:
                base = (cand_row_q0 + qb) * TOPK_QUERY_CHUNKS
                src_ap = src_hbm.reshape((src_hbm.shape[0] * TOPK_QUERY_CHUNKS, n_cand_cols))[
                    base : base + np_use, 0:cand_x
                ]
            nisa.dma_copy(dst=dst_sb[:np_use, :cand_x], src=src_ap)

        cur_v, cur_slot, cur_x = cand_v, None, cand_x
        n_sets = n_chunks
        for lvl, m in enumerate(plan):
            n_out = n_sets // m
            in_x = m * k_cols
            val_m, idx_m = buf["m_val"], buf["m_idx"]
            for o in range(n_out):
                nisa.topk(
                    val_dst=val_m[:np_use, :k],
                    idx_dst=idx_m[:np_use, :k],
                    src=cur_v[:np_use, nl.ds(o * in_x, in_x)],
                    n=m * k,
                )
                if n_out == 1:
                    # TODO: replace this merge + position gather with the GPSIMD custom op
                    _lookup_positions_onehot(
                        buf,
                        cand_p,
                        idx_m,
                        k,
                        nq_b,
                        cand_x,
                        topk_flat_hbm,
                        q_row_base if qb_dyn is not None else q_row_base + qb,
                        q_row_base + n_q,
                        consts,
                        slot_sb=cur_slot,
                        slot_x=cur_x,
                        q_dyn=qb_dyn,
                    )
                else:
                    kernel_assert(
                        False,
                        f"multi-level chunk merge is not wired: plan={plan} for {n_chunks} chunks. "
                        f"n_chunks*k must stay under {TOPK_N_MAX}; at k=2048 that caps chunks at 31, "
                        f"i.e. contexts to ~2M at chunk=65504",
                    )
            n_sets = n_out

    n_batches = n_q // TOPK_QUERIES_PER_BATCH
    if n_batches > 1 and n_q % TOPK_QUERIES_PER_BATCH == 0:
        nl.fori_loop(0, n_q, lambda i: _merge_batch(0, i), step=TOPK_QUERIES_PER_BATCH)
    else:
        for b in range(div_ceil(n_q, TOPK_QUERIES_PER_BATCH)):
            _merge_batch(b * TOPK_QUERIES_PER_BATCH, None)
    sbm.close_scope()
