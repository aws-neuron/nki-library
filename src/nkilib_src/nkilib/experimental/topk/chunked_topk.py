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

"""CHUNKED top-k: a top-k over a vocab WIDER than ``nisa.topk``'s 65,535 ceiling.

WHY THIS EXISTS
---------------
``nisa.topk`` reduces at most ``n < 65536`` elements per call, so a single top-k over the
GPT-OSS vocab (``V = 201,088``) is not expressible. The existing way around that is to
shard the vocab across RANKS and gather the per-rank top-k with a collective. This kernel
does it with no collective at all: it splits the vocab into ``P`` chunks ON ONE CORE, takes
a top-k of each, and reduces the ``P*k`` candidates with a second top-k. Both stages are
local, so there is nothing to reshard between them.

THE IDEA: LET THE INSTRUCTION'S OWN GROUP STRUCTURE BE THE VOCAB SPLIT
---------------------------------------------------------------------
``nisa.topk`` runs one independent top-k per group of ``PARTS_PER_GROUP == 16`` partitions,
and a 128-partition tile holds ``GROUPS_PER_TILE == 8`` of them. So a 128-partition tile
already computes 8 independent top-k per issue. If the tile is loaded as ONE FLAT
CONTIGUOUS SWEEP of the vocab -- partition ``q`` gets vocab ``[q*m, (q+1)*m)`` with
``m = V/128`` -- then group ``g`` (partitions ``16g .. 16g+15``) automatically covers the
contiguous vocab range ``[16g*m, (16g+16)*m)``. The 8-way vocab split is a SIDE EFFECT of
how the instruction is wired; nothing has to build it.

For GPT-OSS the numbers land on it exactly. ``V = 201,088 = 2^7 * 1571`` and
``m = V/128 = 1571``, so each group covers ``16 * 1571 = 25,136 == V/8``:

    25,136 < 65,536              -> legal for nisa.topk
    25,136 % 16 == 0             -> has_pad False, so the coalesced prefetch path is
                                    taken (see fast_coalesced), and stage 1 can skip its
                                    whole phase 2 (see _stage1_desnake_only)
    n_cols = 1,571               -> 3,142 B contiguous DMA run per partition

WHAT IS AND IS NOT AVOIDED
--------------------------
* **Chunk concatenation: avoided entirely.** Stage 1 returns ``[BxS*P, k]``, and
  ``reshape(BxS, P*k)`` puts chunk ``c`` at columns ``[c*k, (c+1)*k)`` of token ``j``. Also
  a pure view. This is why the row order is token-major (``j*P + c``) and not chunk-major:
  token-major makes BOTH reshapes free.
* **Index normalisation: free, or one add.** A stage-1 index is chunk-local, in ``[0, V/P)``,
  and the global id is ``c*(V/P) + local`` with ``c`` fixed per column block. On the fast path
  it costs NOTHING extra: the offset is a whole number of snake partitions, so it folds into
  the snake -> vocab remap's existing multiply (see ``_stage1_desnake_only``). Otherwise it is
  ``P-1`` ``tensor_scalar`` adds on free-dim slices, the cheapest form of vector op there is.
  Compare the sharded alternative, which needs a boundary-counting divide plus a
  select-and-sum over a runtime ``group_map`` tensor.
* **Snaking on the LOAD: avoided entirely -- there is no rearrangement at all.** This is
  the crux, so it is worth being exact. ``inp.reshape(BxS*P, V/P)`` is a pure view of a
  C-contiguous ``[BxS, V]`` tensor (row ``j*P + c`` IS chunk ``c`` of token ``j``), and
  the blocked load is
  ``ap(pattern=[[n_cols, 128], [128*n_cols, tiles], [1, n_cols]], offset=row_start*vocab)``.
  When ``vocab == 16*n_cols`` exactly -- which a chunk width of ``16*1571 = 25,136``
  satisfies -- partition ``q`` reads flat offset ``q*n_cols``, i.e. ONE contiguous
  128-partition sweep of HBM. The "snake" is then purely nominal: it is how the 16
  partitions are interpreted, not a permutation anything performs. Data is loaded directly
  and the instruction's group boundaries ARE the vocab division.
* **De-snaking on the OUTPUT: still required.** After ``nisa.topk`` each group's k results
  are spread across its 16 partitions (snake positions ``[0,k)`` -> partitions 0..15,
  columns ``0..k/16-1``), and both consumers need them on ONE partition per row: the phase-2
  ops are per-partition free-dim ops, and the ``[BxS, k]`` output is row-major. So
  each stage de-snakes once per tile -- two stages, two de-snakes. It cannot be dropped:
  ``nisa.topk`` returns ORIGINAL snake positions rather than output positions, so the
  value<->index pairing is not recoverable from the output layout, and cross-partition
  movement needs a DMA or a transpose either way.

  **Stage 1's phase 2, however, is gone.** It used to reload those de-snake buffers, compact
  the valid columns, remap the snake positions and store ``[BxS*P, k]`` -- which this kernel
  then viewed as ``[BxS, P*k]`` and handed to the reduce. When ``k % 16 == 0`` that entire
  pass is a COPY: ``asc_val_hbm`` is already ``[BxS*P, k]``, so viewed as ``[BxS, P*k]`` it IS
  the candidate array, in an order the index buffer beside it matches slot for slot -- and the
  concat convention only ever required the two to agree, never a particular order. So the
  de-snake buffers are used directly and the pass is deleted, along with the separate padded
  reduce-input staging. The snake -> vocab remap moves to where it is 16x narrower: onto
  ``idx_snake``, ``k_cols`` columns wide instead of ``k``, with the per-chunk offset joining it
  as one add per 16-partition group. See ``_stage1_desnake_only``.

  WHAT IS STILL LEFT, and the trap in it. The inter-stage HBM bounce itself: stage 1 writes
  the candidates to HBM and the reduce loads them straight back. Collapsing that means keeping
  the inter-stage snake in SBUF, and the value side is genuinely cheap -- stage-2 snake
  partition ``p2`` takes stage-1 tile partitions ``{p2, 16+p2, ..., 16(P-1)+p2}``, so with
  chunk ``g`` placed at stage-2 columns ``[g*k_cols, (g+1)*k_cols)`` it is P SBUF->SBUF block
  copies of ``[16, k_cols]``, not a general permutation. The index mapping stays closed-form:

      stage-2 snake position t  ->  g = (t >> 4) // k_cols,   p2 = t & 15
      stage-1 slot              ->  s = idx_snake1[16g + p2, (t >> 4) % k_cols]
      global vocab id            =  (16g + (s & 15)) * n_cols1 + (s >> 4)

  (``16g + (s & 15)`` is just the tile partition that held the value, so the id reduces to
  ``partition * n_cols1 + column``. Verified against a global top-k over 5 geometries x 4
  adversarial distributions x 25 seeds with the modelled hardware output order randomised, and
  with four injected mutations -- dropped chunk term, swapped shift/mask, wrong shuffle
  partition, wrong lookup slot -- all caught.)

  The trap is the INDEX side, and it is why this is not done. That layout makes the reduce's
  candidate position ``v = q*(P*k_cols) + g*k_cols + c1``, which is NOT affine in the source
  partition ``pp = 16g + q`` -- the strides over ``g`` and over ``q`` differ. So the one
  contiguous ``par_dim``-partition de-snake store has to become P stores of 16 partitions
  each, and merging those very stores 8:1 is what an earlier round measured as a large win
  (they were the top serial contributors in the profile). The value side saves DMAs and the
  index side spends them back, with the balance depending on P and the tile count -- so
  measure both ends before committing, and do not assume the SBUF version is faster because
  it moves fewer bytes.

THE HARDWARE SORT IS THE ONLY SORT (THE SOFTWARE SORT IS GONE)
--------------------------------------------------------------
``nisa.topk(sorted=True)`` makes the instruction emit rank ``j`` at snake position
``j``. Because the de-snake mapping is also a compile-time fact, recovering descending order is
then a FIXED PERMUTATION of columns. The software sort that used to serve ``sorted=True`` -- 16
``max8``, 16 ``nc_match_replace8``, 16 gathers, two reverse-gathers, 8 bitonic merge stages,
three synchronising DMAs and both sort-key clamps per sort tile -- WAS DELETED, not kept as a
fallback: a sorted request on a toolchain without the capability (patched nki + neuronx-cc >=
2.0.280935) or with ``k > 8192`` is refused at trace time. Beyond the speed, the deletion
removes the value-addressed index recovery and its tie hazards outright, and it removes a
family of GpSIMD-producer handoffs (max8 / nc_match_replace8 / nc_n_gather feeding cross-engine
readers) whose sync-DMA fences the store-visibility work below showed to be DISTANCE rather
than ordering. The ``sorted=False`` compaction path is not a sort and stays -- stage 1 runs on
it. The ``p == 1`` small-vocab case runs through this kernel's own inlined top-k for the same
reason, rather than delegating to ``gpsimd_topk`` and silently re-entering a software sort.

**Deleting that work moved the wall clock by NOTHING** -- the first landing (permutation folded
into the reload's access pattern) measured 170.3 -> 172.1 us at 128 ranks, gbs=128 -- and the
work/wait split of the profile says why (summed occupancy over all engines):

                    instructions   occupancy
    software sort:  work 644        214.3 us
                    wait 101        167.1 us
    hardware sort:  work 464        173.9 us      -180 instructions, -40 us
                    wait  61        167.5 us      -40 instructions,  +0.4 us

The waiting was invariant: 40 us of work vanished and the wait absorbed it exactly. The machine
was fully idle for only ~13 us of the ~171 us body -- a timeline full of semaphore waits, with
the critical path the serial chain norm -> GEMM -> hop 0 -> hop 1 -> stage 1 -> reduce ->
sampler. **The lever was therefore never the work; it was the synchronisation on that chain**,
and the strided reload was itself a synchronisation point: 1,968 serialised 4-byte DMA transfers
(see the next section). Replacing it with a contiguous reload + on-chip permutation, and the
fused tail's gather with the Tensor-Engine one-hot, is what finally moved the body:

    gbs=128:  170.0 -> 146.2 us   (-14.0% vs the shipped software path; -18.8% vs baseline)
    gbs=512:  348.5 -> 334.4 us   (-4.0%)
    gbs=1024: 580.1 -> 577.9 us   (-0.4% -- the body there is transfer/compute bound)
    semaphore stall: -90 / -194 / -341 us respectively; 4-byte packets 3,070 -> 1,054 at gbs=128

Two lessons worth keeping: a standalone measurement of this kernel is NOT predictive of the tail
(the same sort is worth 1.32x alone and nothing in situ until the synchronisation went too); and
before optimising anything here, split the profile into work and wait -- if wait occupancy is
~equal to the body, instruction count is the wrong lever, and a store/reload or a serialised DMA
burst on the critical chain is the right one.

THE 4-BYTE DMA PROBLEM (solved -- the permutation moved on chip, which exposed a deeper bug)
--------------------------------------------------------------------------------------------
Folding the rank permutation into the reload's access pattern was NOT free, and the wording that
said so was wrong. **The last pattern level is the FASTEST-VARYING**, so a pattern ending in
``[k_cols, PARTS_PER_GROUP]`` gives the innermost dimension a stride of ``k_cols`` rather than 1,
and every single element becomes its own transfer. Measured at gbs=128 against the software sort:

    4-byte DMA packets     1,102 -> 3,070      (+1,968)
    their summed time        7.3 -> 57.3 us    (+50 us)

Rank order and de-snaked order are a TRANSPOSE of one another (rank ``j = p + 16c``, de-snaked
column ``m = p*k_cols + c``), so exactly one side has to be strided, and the strided side belongs
ON CHIP. That is what the hw_sort branch now does: the reload is one contiguous run per row, and
the permutation is 16 strided ``tensor_copy`` ops per tensor, k_cols wide each.

**Every earlier attempt at exactly this broke correctness at BxS == 1, and the fault was never
the permutation.** Moving ~30 Vector ops into phase 2 shifted the engine schedule enough to
expose a LATENT RACE in the fused tail's ``nc_n_gather`` -- present since the first version of
this kernel, hit by no schedule before it. The debug taps (L4) plus the Instruction/Flow tables
of four failing NEFFs established, against escalating fences, that the gather's completion
signal does not mean its stores are visible: a Vector consumer with a correct FLOW_DEPENDENCE
edge read the gather's tail as stale zeros 500 ns after it retired; a synchronising SBUF->SBUF
DMA read the same mid-flight state; a same-dtype GpSIMD fence copy was copy-propagated away;
and an unfoldable GpSIMD CAST fence was itself entered by its consumer 119 ns into its 900 ns
execution, with two GpSIMD instructions visibly overlapping on the engine. Meanwhile
``nisa.topk``'s consumers at SHORTER distances are always correct -- the defect is the gather
ucode's, not GpSIMD stores in general (the multi-group corner of the same defect).

So the gather is not fenced, it is GONE: the position -> id lookup is an exact one-hot
contraction on the Tensor Engine (see the fused tail), whose PSUM -> Vector handoff is the
best-validated producer edge on the chip. Everything is fp32, ids are < 2^24, every one-hot
column selects exactly one partition, so every sum is one exact product plus zeros. The software
sort's own gathers (max8 / nc_match_replace8 / the bitonic reverse) carried the same hazard
class behind sync-DMA fences that this evidence showed to be DISTANCE rather than ordering --
one more reason that code is now deleted rather than kept as a fallback.

The synchronisation-removal lesson (see THE HARDWARE SORT above, where it is now measured at
-23.8 us) also reframes the remaining inter-stage HBM bounce: its value is not the 2.6 us of DMA
bytes, it is that a store followed by a reload of the same region is a serialisation point on
the critical chain. Same for the hops -- hop 1 costs 2.8x hop 0 for identical bytes purely
because it crosses devices.

STAGE 1 IS NEVER SORTED
-----------------------
Stage 1 always runs with ``sorted=False``, and this is deliberately NOT a caller option.
Its ordering is unobservable -- stage-1 values and indices never leave this kernel, and the
reduce re-sorts -- so the choice cannot change the output, only the cost. Measured on trn3
at LNC=1, ``V = 201,088``, ``k = 256``:

    BxS=1 (P=8):  97.32 us -> 76.33 us   (-21.6%)
    BxS=8 (P=4): 176.53 us -> 151.88 us  (-14.0%)

The profile explains the size of it: the VECTOR engine dominates this kernel, not GpSIMD
(54.24 of 97.32 us at BxS=1), because the phase-2 descending sort is sized by
``k`` rather than by the vocab width. Dropping stage 1's sort removed 132 Vector
instructions (317 -> 185) and 21.2 us.

The wider lesson for anyone extending this: cost here tracks the NUMBER OF TOP-K
INVOCATIONS times ``k``, not the number of elements scanned. A three-stage tree reduce
would add a third phase-2 sort and lose more than the narrower reduce could win.

CHOOSING P
----------
Stage-1 cost has a closed form. With ``rows = P*R`` and ``width = V/P``, GpSIMD cost
``ceil(rows/8) * ceil(width/16)`` becomes ``R*V/128`` -- the ``P`` cancels. So P CANNOT make
stage 1 faster; it can only make it worse by failing to fill a tile, shortening the DMA
runs, or losing the ``% 16`` coalesced path. Meanwhile the reduce grows as ``k*P``. The
optimum is therefore the SMALLEST P that fills a whole tile and is legal:

    P = min { P : (BxS*P) % 8 == 0,  V/P < 65536,  (V/P) % 16 == 0,  P*k < 65536 }

``create_chunked_topk_config`` picks it. At ``V = 201,088, k = 256`` that is ``P = 8`` for
``BxS = 1`` (8 rows == exactly one tile) and ``P = 4`` for ``BxS >= 4``.

LNC=1 BY DESIGN
---------------
Both stages pass ``num_programs=1``. NOTE that this does not pin anything: the inlined top-k
re-reads the program count from the live grid (``get_verified_program_sharding_info``), so at
LNC=2 its ``per_lnc_BxS`` halves and ``fast_dma_safe`` can silently flip off. This kernel is
written for and tested at LNC=1 only, and asserts it at trace time.

DEBUG TAPS
----------
Every correctness defect this kernel has had was invisible except on hardware, under full
engine/DMA overlap: the simulator serialises, so does a device dump, and both return the right
answer while the device returns the wrong one. Diagnosing that by hypothesis costs a 128-rank
sweep per guess. ``config.debug_taps`` instead makes the index chain OBSERVABLE in situ -- it
appends seven extra outputs (``TAP_NAMES``) to the return tuple, each a copy of one link:

    s1_pos    [BxS*P, k]       stage-1 snake positions phase 2 worked from, pre-remap
    s1_gid    [BxS*P, k]       stage-1 chunk-local vocab ids, post-remap
    s1_late   [BxS*P, k_pad]   a SECOND read of stage 1's de-snake buffer
    cand_i    [BxS, P*k]       global ids after the chunk offsets -- the gather's data
    r_pos     [BxS, k]         the reduce's positions -- the gather's indices
    gathered  [BxS, k]         the gather's result, before the uint32 narrowing
    s1_val    [BxS*P, k]       the VALUE side of stage 1's candidates (appended last)

A bad returned id is then localised in ONE run rather than one guess per run: walk the chain
backwards and the first tap that disagrees with an offline oracle names the culprit. Values have
never been the failing side, which is itself diagnostic -- values and indices ride in separate
buffers, so a defect that takes only the indices is a per-buffer fault (a race, a dropped
strided store) and not a bad selection.

TWO PROPERTIES MAKE THE TAPS TRUSTWORTHY, and both are easy to lose:

* **A tap must not create the ordering it is measuring.** Adding a reader of a buffer adds a
  dependency edge on it, which can hide the very race being hunted. So every tap point above is
  a buffer that ALREADY has a full-width consumer immediately after it (``out_i`` is read whole
  by the remap, ``i_f`` and ``pf`` by the gather, ``got`` by the output cast). The tap rides
  along an edge that exists; it does not add one.
* **``s1_late`` works without being guaranteed late.** It is traced last so it TENDS to be
  scheduled late, but nothing pins it there. It does not need to be: the signature of a
  store/reload race is TWO READS OF THE SAME ADDRESS DISAGREEING, which cannot happen unless a
  write was in flight between them. ``s1_pos`` wrong + ``s1_late`` right means the de-snake
  store landed after phase 2 read it; both wrong means ``nisa.topk`` reported those positions
  and nothing about DMA ordering is involved.

Taps change the kernel's arity, so they are opt-in and off by default. The harness maps
validators to outputs POSITIONALLY, so a compile+infer test run must not enable them; the
128-rank pod path is unaffected because the on-device profiler's capture flow saves every
output as ``output_<n>.2.npy`` regardless of how many there are.

Complementary, and the right first question for any new DMA pattern: ``--enable-perf-analysis``
emits ``analysis_nc00.log`` with a ``critical_dep_type`` per instruction, where ``Flow`` means a
semaphore edge exists and ``Engine`` means the only ordering is same-engine issue. If a
consumer's dependency on a producer is not ``Flow``, it is unordered by construction -- no
amount of measurement is needed to know it is a bug waiting for the right traffic.
"""

import inspect
from dataclasses import dataclass
from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl
import numpy as np

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from .gpsimd_topk import (
    BFLOAT16_MIN,
    GROUPS_PER_TILE,
    PARTS_PER_GROUP,
    PMAX,
    _valid_desnake_col_runs,
    _validate_gpsimd_topk,
    create_gpsimd_topk_config,
    snake_padded_n,
)

# nisa.topk reduces n < _TOPK_N_MAX elements per call. This is the instruction's own
# bound (the returned index is a snake POSITION), not a kernel choice -- which is why
# P >= ceil(V / (_TOPK_N_MAX - 1)) is a hardware floor rather than a tuning knob.
_TOPK_N_MAX = 65536
# nisa.topk's k bound, mirrored so an out-of-envelope k fails here with a clear message.
_TOPK_K_MAX = 32768

# --- Hardware sort capability ------------------------------------------------------------
# nisa.topk gains a `sorted=` flag in newer NKI toolchains, which makes the
# GpSIMD instruction emit rank j at snake position j. That turns the software descending sort
# into a COMPILE-TIME COLUMN PERMUTATION -- see the hw_sort branch in _inline_topk, where it
# folds into the reload's access pattern -- which is NOT free, see THE 4-BYTE DMA PROBLEM.
#
# Probed at TRACE TIME rather than version-gated, because passing `sorted=` to an nki that does
# not have it is a TypeError while tracing, not a graceful fallback. A version check would also
# be wrong twice over: the flag needs BOTH a patched nki AND the BIR field from
# neuronx-cc >= 2.0.280935, and the two move independently.
_HW_SORT_AVAILABLE = "sorted" in inspect.signature(nisa.topk).parameters
# The ISA caps the sorted mode at k <= 8192. Sorted requests above it are REFUSED: the
# software sort that used to serve them was deleted.
_HW_SORT_MAX_K = 8192


@dataclass(frozen=True, eq=True)
class ChunkedTopkConfig(nl.NKIObject):
    """Configuration for the chunked (wide-vocab) top-k kernel.

    Attributes:
        BxS: Combined batch*sequence dimension (number of logical rows / tokens).
        vocab_size: Full vocab length reduced over. May exceed nisa.topk's 65,535.
        k: Number of largest elements to return.
        chunks: P, the number of disjoint vocab chunks stage 1 reduces independently.
            ``chunks == 1`` short-circuits to a plain gpsimd_topk.
        sorted: Whether the FINAL output must be descending.
        inp_dtype: Input/value dtype (must be bfloat16 -- gpsimd_topk's own requirement).
        index_dtype: Output index dtype (uint32).
        out_shape: Logical output shape (input leading dims + k).
        debug_taps: Append the seven diagnostic taps to the return tuple. See DEBUG TAPS in the
            module docstring. Off by default and free when off; it changes the kernel's arity,
            so a caller that turns it on has to consume the extra outputs.
        skip_nan_fold: Skip stage 1's NaN -> BFLOAT16_MIN fold, under a CALLER CONTRACT that
            the input contains no NaN. The fold exists because nisa.topk is undefined on NaN
            (see the fold site); on a NaN-free input it can never fire, yet it still costs a
            compare plus a predicated copy over the FULL input width per tile -- the widest
            non-topk work in stage 1. +/-inf need no fold and stay legal either way (they are
            orderable; the bitpattern gate covers them). Default False: only a caller that can
            PROVE its input finite (e.g. the GPT-OSS tail, whose logits are its own GEMM
            output) should set it, because with NaN present the fold is what keeps the output
            defined. The reduce stage always skips independently of this flag -- its input is
            stage 1's own already-folded output.
        interleaved_peer_width: 0 (default) means ``inp`` is the ordinary C-contiguous
            ``[BxS, vocab]``. A value ``w > 0`` declares the ALL-TO-ALL RECEIVE LAYOUT
            instead: ``inp`` is ``[BxS * (vocab/w), w]`` where row ``p*BxS + j`` holds token
            ``j``'s vocab columns ``[p*w, (p+1)*w)``. Stage 1 then folds the row -> column
            de-interleave into its own load patterns (three levels, unit innermost stride),
            which DELETES the caller's [rows, vocab] materialisation -- a full HBM round trip
            of the received logits. Results are BITWISE identical to de-interleaving first:
            only the DMA descriptors change, not one value or its snake position. Fast-path
            only: requires the desnake-only stage-1 shape and ``w | (vocab/P)/16`` so every
            snake partition covers whole peer blocks; anything else is refused at trace time.
        interleaved_batch_groups: 1 (default) means one receive buffer. ``G > 1`` declares a
            BATCH-GROUP SLABBED receive: the producer ran G separate exchanges, each delivering
            tokens ``[g*BxS/G, (g+1)*BxS/G)`` as its own peer-major slab (what lets the
            exchange start under the producer's GEMM), and ``inp`` is a LIST of those G slab
            tensors (jit-in-jit callers only). Each slab is read in place with its own data
            dependency, so group g's stage-1 sorts start when ITS exchange lands. Requires
            ``interleaved_peer_width > 0`` and ``G | BxS``. A stage-1 sort tile MAY straddle
            two slabs (each 16-partition group's load selects its slab per token); it then
            waits for the later exchange, so groups aligned to whole 8-row tiles
            (``(BxS/G) * P % 8 == 0``) overlap best.
    """

    BxS: int
    vocab_size: int
    k: int
    chunks: int
    sorted: bool
    inp_dtype: np.dtype
    index_dtype: np.dtype
    out_shape: tuple
    debug_taps: bool = False
    skip_nan_fold: bool = False
    interleaved_peer_width: int = 0
    # > 1 when the receive buffer is BATCH-GROUP slabbed: the producer ran G separate
    # exchanges, each delivering tokens [g*T/G, (g+1)*T/G) as its own peer-major slab (what
    # lets the exchange start under the producer's GEMM). Token j's slab base is
    # (j // (T/G)) * (T/G) * vocab elements; within a slab the layout is exactly the
    # interleaved_peer_width one with T/G tokens. Requires interleaved_peer_width > 0.
    interleaved_batch_groups: int = 1

    @property
    def topk_config(self):
        """Self-reference so a shared torch_ref reading ``.topk_config.k`` /
        ``.topk_config.sorted`` works against this config without a separate object."""
        return self


def _chunk_legal(width: int, p: int, k: int) -> bool:
    """THE legality predicate for one (chunk width, P, k) triple, shared by ``pick_chunks``
    and ``_validate_chunked_topk`` so the two can never disagree (an auto-picked P the
    validator then refuses, while another P is legal, was exactly that disagreement).

    In order: P is a power of two (the stage-1 bypass derives chunk offsets with a bitwise
    mask that is a valid modulus only for pow2); the instruction bound applies to the
    16-PADDED snake width (see ``snake_padded_n``); k fits one chunk; and for P > 1 the
    fused reduce's one-hot position lookup holds P*k positions across at most 128 chunks of
    128 partitions -- tighter than the instruction's own ``n < 65536``.
    """
    if p < 1 or p & (p - 1):
        return False
    if width < 8 or snake_padded_n(width) >= _TOPK_N_MAX or k > width:
        return False
    if p > 1 and not (8 <= p * k <= PMAX * PMAX):
        return False
    return True


def _slab_sizes_match(shapes, expected_elems: int) -> bool:
    """Every receive slab holds exactly ``expected_elems`` (= tokens-per-slab * vocab).

    The single-tensor path gets this check for free from its ``reshape``; a slab LIST is
    flattened per tensor, which cannot fail, so a mis-sized slab would otherwise be read
    through the hand-built flat pattern at wrong token/peer bases with no trace-time error.
    """
    return all(int(np.prod(shape)) == expected_elems for shape in shapes)


def pick_chunks(BxS: int, vocab: int, k: int) -> int:
    """Smallest legal P that fills a whole 128-partition tile. See CHOOSING P above.

    Ranking, in order: fills a tile (``BxS*P % 8 == 0``), then hits the coalesced load
    path (``(vocab/P) % 16 == 0``), then smallest P. Ties on the first two keys are what
    make this pick ``P = 8`` at ``BxS = 1`` (where ``P = 4`` would leave half a tile
    padding) and ``P = 4`` at ``BxS >= 4`` (where both fill, so the smaller reduce wins).

    Returns 1 when the vocab already fits one nisa.topk call (its PADDED snake width under
    the bound), i.e. no chunking is needed. Legality per P is ``_chunk_legal`` -- the same
    predicate the validator applies -- so whatever this returns traces.
    """
    if snake_padded_n(vocab) < _TOPK_N_MAX:
        return 1
    best_key, best_p = None, None
    p = 1
    while p <= PMAX:
        if vocab % p == 0:
            width = vocab // p
            if _chunk_legal(width, p, k):
                key = (
                    0 if (BxS * p) % GROUPS_PER_TILE == 0 else 1,
                    0 if width % PARTS_PER_GROUP == 0 else 1,
                    p,
                )
                if best_key is None or key < best_key:
                    best_key, best_p = key, p
        p *= 2
    if best_p is None:
        raise ValueError(
            f"no legal chunk count for vocab={vocab}, k={k}: need a power-of-two P dividing "
            f"vocab with 8 <= vocab/P, 16*ceil((vocab/P)/16) < {_TOPK_N_MAX}, k <= vocab/P and "
            f"8 <= P*k <= {PMAX * PMAX}"
        )
    return best_p


def interleaved_fold_supported(bxs: int, vocab: int, k: int, peer_width: int) -> bool:
    """True iff this kernel accepts ``interleaved_peer_width=peer_width`` at this geometry.

    THE eligibility predicate for the receive-layout fold, exported so callers decide
    "fold or explicit de-interleave" with the callee's OWN rules instead of re-deriving a
    subset at the call site (an incomplete re-derivation turned legal configs into
    trace-time refusals -- e.g. a vocab small enough that ``pick_chunks`` returns 1
    delegates to the single-call path, which does not implement the fold at all).
    Mirrors, in order: the p == 1 delegate refusal, the desnake-only stage-1 path
    requirements (k, row and width alignment -- which also imply the coalesced whole-tile
    load path), and the peer-width divisibility asserts.
    """
    if peer_width <= 0:
        return False
    p = pick_chunks(bxs, vocab, k)
    if p == 1:
        return False
    width = vocab // p
    s1_rows = bxs * p
    if not (k % PARTS_PER_GROUP == 0 and s1_rows % GROUPS_PER_TILE == 0 and width % PARTS_PER_GROUP == 0):
        return False
    n_cols_s1 = div_ceil(width, PARTS_PER_GROUP)
    return vocab % peer_width == 0 and n_cols_s1 % peer_width == 0 and width % peer_width == 0


def create_chunked_topk_config(
    inp_shape: Tuple,
    inp_dtype: np.dtype,
    k: int,
    chunks: int = 0,
    sorted: bool = True,
    debug_taps: bool = False,
    skip_nan_fold: bool = False,
    interleaved_peer_width: int = 0,
    interleaved_batch_groups: int = 1,
) -> ChunkedTopkConfig:
    """Build a ChunkedTopkConfig. ``chunks=0`` (default) auto-picks via ``pick_chunks``.

    ``inp_shape`` is always the LOGICAL ``[..., vocab]`` shape, including when
    ``interleaved_peer_width`` declares the receive layout (the physical tensor then has the
    same element count in ``[BxS*(vocab/w), w]`` order; every internal access is a reshape or
    an explicit pattern over the same flat buffer).
    """
    BxS = 1
    for d in inp_shape[:-1]:
        BxS *= d
    vocab_size = inp_shape[-1]
    return ChunkedTopkConfig(
        BxS=BxS,
        vocab_size=vocab_size,
        k=k,
        chunks=chunks if chunks else pick_chunks(BxS, vocab_size, k),
        sorted=sorted,
        inp_dtype=inp_dtype,
        index_dtype=nl.uint32,
        out_shape=tuple(list(inp_shape[:-1]) + [k]),
        debug_taps=debug_taps,
        skip_nan_fold=skip_nan_fold,
        interleaved_peer_width=interleaved_peer_width,
        interleaved_batch_groups=interleaved_batch_groups,
    )


#: Names of the diagnostic taps, in the order ``chunked_topk`` appends them to its return tuple
#: when ``config.debug_taps``. Exposed so a caller that forwards them (the GPT-OSS tail does) can
#: label them without hard-coding the count. See DEBUG TAPS in the module docstring.
TAP_NAMES = ("s1_pos", "s1_gid", "s1_late", "cand_i", "r_pos", "gathered", "s1_val")


def _validate_chunked_topk(config: ChunkedTopkConfig) -> None:
    """Validate the requested shape against every nisa.topk constraint, per stage."""
    v, k, p = config.vocab_size, config.k, config.chunks
    kernel_assert(config.inp_dtype == nl.bfloat16, "chunked_topk requires bfloat16 input")
    kernel_assert(1 <= k < _TOPK_K_MAX, f"chunked_topk requires 1 <= k ({k}) < {_TOPK_K_MAX}")
    # Sorted output is hardware-sort ONLY. The software sort was deleted (see THE HARDWARE SORT
    # IS THE ONLY SORT in the module docstring), so a sorted request on a toolchain without the
    # capability is REFUSED at trace time rather than silently served by a slower, race-prone
    # fallback. The capability needs BOTH a patched nki and neuronx-cc >= 2.0.280935.
    if config.sorted:
        kernel_assert(
            _HW_SORT_AVAILABLE,
            "chunked_topk sorted=True requires nisa.topk(sorted=True); this toolchain does not "
            "have it (patched nki + neuronx-cc >= 2.0.280935 required) and the software sort "
            "was removed. Pass sorted=False or upgrade the toolchain.",
        )
        kernel_assert(k <= _HW_SORT_MAX_K, f"sorted top-k is capped at k <= {_HW_SORT_MAX_K} by the ISA, got k={k}")
    kernel_assert(p >= 1, f"chunks ({p}) must be >= 1")
    # Power-of-two P is a CORRECTNESS requirement, not a picker preference: the stage-1 bypass
    # derives each row's chunk offset with a bitwise-AND mask (16*P - 1), which is a valid
    # modulus only for power-of-two P. pick_chunks only ever returns powers of two; this guard
    # is for an explicit caller-set chunks, where e.g. chunks=3 passed every divisibility check
    # and returned in-range, exactly-right VALUES with silently wrong global ids.
    # The per-bound asserts below spell out _chunk_legal (the predicate pick_chunks selects
    # with) one named message at a time; test_pick_chunks_agrees_with_validator pins that
    # the two stay the same set.
    kernel_assert(p & (p - 1) == 0, f"chunks ({p}) must be a power of two")
    kernel_assert(v % p == 0, f"vocab ({v}) must be divisible by chunks ({p})")
    width = v // p
    # Stage 1 sees [BxS*P, V/P], so the CHUNK width is what faces the instruction bound -- and
    # what faces it is the 16-PADDED snake size 16*ceil(width/16), which reaches 65536 (past
    # the bound) for widths 65521..65535. Bound the padded width, not the raw one.
    kernel_assert(8 <= width, f"chunk width V/P ({width}) must be >= 8")
    kernel_assert(
        PARTS_PER_GROUP * div_ceil(width, PARTS_PER_GROUP) < _TOPK_N_MAX,
        f"chunk width V/P ({width}) padded to a whole 16-column snake "
        f"({PARTS_PER_GROUP * div_ceil(width, PARTS_PER_GROUP)}) must be < {_TOPK_N_MAX}",
    )
    kernel_assert(k <= width, f"k ({k}) must be <= chunk width ({width})")
    if p > 1:
        # The reduce sees [BxS, P*k]; P*k is a SECOND, independent bound -- and a TIGHTER one
        # than the instruction's: the fused reduce translates sort positions through the
        # candidate-id array with a Tensor-Engine one-hot lookup that lays the P*k positions
        # across ceil(P*k / 128) partition chunks, so P*k must fit in 128 chunks of 128.
        # Batch sharding keeps it small (2,048 at P=8) but a large P or k inflates it.
        red_w = p * k
        kernel_assert(
            8 <= red_w <= PMAX * PMAX,
            f"reduce width P*k ({red_w}) must be in [8, {PMAX * PMAX}]: the fused reduce's "
            f"one-hot position lookup holds all P*k positions across at most {PMAX} "
            f"{PMAX}-wide partition chunks",
        )
    if config.interleaved_peer_width:
        w = config.interleaved_peer_width
        n_cols_s1 = div_ceil(width, PARTS_PER_GROUP)
        kernel_assert(
            v % w == 0 and n_cols_s1 % w == 0,
            f"interleaved_peer_width ({w}) must divide the vocab ({v}) and the stage-1 snake "
            f"row width ({n_cols_s1}); the receive-layout fold needs whole peer blocks per "
            f"snake partition",
        )
    g = config.interleaved_batch_groups
    kernel_assert(g >= 1, f"interleaved_batch_groups ({g}) must be >= 1")
    if g > 1:
        kernel_assert(
            config.interleaved_peer_width > 0,
            "interleaved_batch_groups > 1 describes a slabbed RECEIVE layout and needs interleaved_peer_width",
        )
        bxs = config.BxS
        kernel_assert(bxs % g == 0, f"batch ({bxs}) must divide by interleaved_batch_groups ({g})")
        # No sort-tile alignment requirement: every 16-partition group's load selects its
        # slab per TOKEN (inp_flats[j // tok_span]), so a stage-1 tile may straddle two slabs
        # and simply waits for the later of its two exchanges. Aligned groups
        # (((bxs/g)*p) % 8 == 0) overlap best -- a performance preference, not legality.


# =======================================================================================
# THE INLINED TOP-K
# =======================================================================================
# ``_inline_topk`` is a VERBATIM inline of ``gpsimd_topk``'s ``@nki.jit`` body: the same
# instructions, in the same order, over the same buffers, with the same DMA modes. Read that
# way it buys nothing -- and measured that way it is worth about -1% -- but it is what puts
# BOTH ``nisa.topk`` calls inside one traced body, which is the precondition for every
# inter-stage optimisation this kernel wants: keeping the candidate array in SBUF instead of
# bouncing it through HBM, and handing stage 1's snake layout straight to stage 2 instead of
# de-snaking and re-snaking around a [BxS, k] HBM interface. None of that is expressible while
# the stages are separate kernels whose only interface is that HBM array.
#
# It is a COPY rather than a shared helper on purpose. ``gpsimd_topk`` is a public kernel with
# its own callers and its own contract; the specialisations wanted here -- a stage that never
# de-snakes, a stage whose "vocab" is a candidate array rather than logits -- are not modes
# worth bolting onto it. This is a peak-performance path, and the duplication is the cheaper
# side of that trade.
#
# THE CORRECTNESS RATIONALE IS DELIBERATELY NOT DUPLICATED. Nearly every non-obvious line
# below is there because of a specific measured defect, documented at length beside the SAME
# line in ``gpsimd_topk``: the three unconditional synchronising SBUF->SBUF DMAs, the
# ``dge_mode.unknown`` on the de-snake round trip, the two NaN folds, the ``NEG_INF_FLOOR``
# sort-key clamp, the two gather-index clamps, and the compile-time padding-column
# enumeration. Each comment here names its counterpart. Do not "simplify" any of them without
# reading the original first -- several look redundant and are not.
#: Slots of ``_inline_topk``'s ``fused`` argument. A plain tuple rather than a dataclass on
#: purpose: nki hashes config objects for its trace cache, and a dataclass holding live tensor
#: handles is a bad thing to put in a hash key.
_FUSE_CAND_I = 0  # [rows, P*k] uint32 HBM: stage-1 chunk-local ids, the gather's data
_FUSE_CHUNKS = 1  # P
_FUSE_WIDTH = 2  # vocab // P, the per-chunk id offset
_FUSE_OUT_VALS = 3  # [rows, k] bf16 shared_hbm destination
_FUSE_OUT_IDX = 4  # [rows, k] uint32 shared_hbm destination
_FUSE_N_ROWS = 5  # rows to emit; the reduce's own row count may be padded above it
_FUSE_TAP_CAND = 6
_FUSE_TAP_POS = 7
_FUSE_TAP_GOT = 8
_FUSE_IDS_GLOBAL = 9  # cand_i already holds float32 GLOBAL ids: skip the cast and the offsets


def _stage1_desnake_only(
    inp: nl.NkiTensor,
    config,
    chunks: int,
    chunk_width: int,
    pad_rows: int,
    tap_pos=None,
    skip_nan_fold=False,
    peer_width=0,
    batch_groups=1,
):
    """Stage 1 with NO PHASE 2 AT ALL: the de-snake buffers ARE the candidate arrays.

    THE OBSERVATION. ``gpsimd_topk``'s phase 1 de-snakes each tile into ``[BxS, k_pad]`` HBM
    buffers, and phase 2 then reloads them, compacts the valid columns to ``[0, k)``, remaps the
    snake positions, and stores ``[BxS, k]`` results -- which ``chunked_topk`` immediately views
    as ``[BxS/P, P*k]`` and hands to the reduce. When ``k_pad == k`` that whole pass is a COPY.
    ``asc_val_hbm`` is ``[BxS*P, k]``, so viewed as ``[BxS, P*k]`` it already IS the candidate
    value array, in a candidate order the index buffer beside it matches slot for slot -- and the
    concat convention only requires the two agree, never a particular order. So phase 2 is not
    optimised here, it is DELETED: two reloads, two compacting copies, the remap over a k-wide
    tile, two stores, and (in the caller) the P-1 chunk-offset adds and the uint32 -> float32
    cast of a P*k-wide tile all go away, and so does the separate padded reduce-input staging.

    What has to happen instead, and why it is cheaper: the snake -> vocab remap still has to be
    applied, but it can be applied to ``idx_snake`` BEFORE the de-snake store, where the tile is
    ``k_cols`` wide instead of ``k``. That is 16x narrower for the same op count, and Vector cost
    tracks free-dim width, not partition count -- the same work over 16 columns of 128 partitions
    instead of 256 columns of 8. The per-chunk global-id offset joins it there, as one add per
    16-partition group on a compile-time constant, replacing P-1 adds over k columns.

    Args:
        inp: ``[BxS*P, V/P]`` bfloat16 view of the caller's vocab, one chunk per row.
        config: a ``GpsimdTopkConfig`` for that view. ``k % 16 == 0`` is REQUIRED (it is what
            makes ``k_pad == k`` and therefore makes phase 2 a copy), as is a shard of whole
            128-partition tiles (which is what keeps the value buffer bfloat16 -- see
            ``fast_dma_safe`` -- and the reduce needs bfloat16 input).
        chunks: P. Row ``r`` of ``inp`` is chunk ``r % P``, which is what makes the offset a
            per-group compile-time constant.
        chunk_width: ``V/P``, the per-chunk id offset unit.
        pad_rows: total rows to allocate, ``red_rows * P`` -- so the ``[red_rows, P*k]`` view the
            reduce wants exists, with its tail rows carrying the sentinel. Those rows are whole
            reduce ROWS whose results are never read, so their garbage indices cannot leak into
            a real row's answer.

    Returns:
        ``(cand_v [pad_rows, k] bf16, cand_i [pad_rows, k] f32)``, whose ``[pad_rows/P, P*k]``
        views are the reduce's input and the gather's data. Indices are GLOBAL vocab ids.
    """
    _validate_gpsimd_topk(config)

    BxS = config.BxS
    vocab = config.vocab_size
    k = config.k

    shard_info = get_verified_program_sharding_info("chunked_topk_stage1", (0, 1), 2)
    if BxS > 1:
        n_prgs = shard_info[1]
        prg_id = shard_info[2]
    else:
        n_prgs = 1
        prg_id = 0
    per_lnc_BxS = div_ceil(BxS, n_prgs)

    n_cols = div_ceil(vocab, PARTS_PER_GROUP)
    full_n = PARTS_PER_GROUP * n_cols
    has_pad = full_n != vocab
    k_cols = div_ceil(k, PARTS_PER_GROUP)
    k_pad = k_cols * PARTS_PER_GROUP
    fast_dma_safe = (per_lnc_BxS % GROUPS_PER_TILE == 0) and (BxS == n_prgs * per_lnc_BxS)
    # The two preconditions of the whole scheme, asserted rather than assumed: the caller picks
    # this path on exactly these conditions, and if either were false the buffers below would not
    # be the candidate arrays (k_pad > k leaves unwritten columns interleaved with the valid ones)
    # or would not be bfloat16 (which nisa.topk requires of the reduce's source).
    kernel_assert(k_pad == k, f"_stage1_desnake_only needs k % 16 == 0, got k={k}")
    kernel_assert(fast_dma_safe, "_stage1_desnake_only needs whole 128-partition tiles")
    kernel_assert(pad_rows % chunks == 0, f"pad_rows ({pad_rows}) must be a multiple of chunks ({chunks})")
    # chunk_width == 16*n_cols is what makes the chunk offset a whole number of snake
    # partitions, which is what lets it fold into the remap's multiply (see chunk_base).
    kernel_assert(
        not has_pad and chunk_width == PARTS_PER_GROUP * n_cols,
        f"_stage1_desnake_only needs 16 | chunk width, got {chunk_width}",
    )

    # inp may be a LIST of batch-group slab tensors (interleaved_batch_groups > 1,
    # jit-in-jit only): slab g holds tokens [g*tok_span, (g+1)*tok_span)'s receive layout
    # as its own tensor, so a stage-1 tile's loads data-depend only on ITS slab's producing
    # exchange. One tensor is the one-slab degenerate case.
    inp_slabs = list(inp) if isinstance(inp, (list, tuple)) else [inp]
    inp_flats = [t.reshape((int(np.prod(t.shape)),)) for t in inp_slabs]
    inp_flat = inp_flats[0]
    # --- The interleaved (all-to-all receive) source layout --------------------------------
    # peer_width > 0 says inp is the RECEIVE layout: token j's vocab column v = p*w + q lives
    # at flat address (p*BxS_tok + j)*w + q. The blocked snake load still wants vocab-in-chunk
    # index u = part*n_cols + col at snake (part, col) -- and because w | n_cols (asserted),
    # each snake partition covers WHOLE peer blocks, so the load stays a three-level pattern
    # with a UNIT innermost stride (see THE 4-BYTE DMA PROBLEM for why that matters):
    #     [[ppp*BxS_tok*w, 16], [BxS_tok*w, ppp], [1, w]]   ppp = n_cols // w
    #     offset(j, c) = c*(chunk_width//w)*BxS_tok*w + j*w
    # The snake CONTENT is identical to loading a de-interleaved [BxS_tok, vocab] -- only the
    # descriptors change -- so everything downstream (topk, de-snake, remap, ids) is bitwise
    # the same. What it deletes is the caller's de-interleave: a full HBM round trip of the
    # received logits. Cost: one DMA per 16-partition GROUP instead of one per 128-partition
    # tile (the partition stride is uniform only within a group).
    BxS_tok = BxS // chunks
    ppp = ppc = 0
    # tok_span: how many TOKENS share one peer-major interleave. One exchange (batch_groups
    # == 1) interleaves all BxS_tok tokens; a batch-group-slabbed receive (see
    # interleaved_batch_groups in ChunkedTopkConfig) interleaves tok_span = BxS_tok/G tokens
    # per slab, and token j's slab starts at (j // tok_span) * tok_span * vocab elements.
    tok_span = BxS_tok
    if peer_width:
        kernel_assert(
            not has_pad and n_cols % peer_width == 0 and chunk_width % peer_width == 0,
            f"interleaved_peer_width ({peer_width}) must divide the snake row width "
            f"({n_cols}) and the chunk width ({chunk_width})",
        )
        ppp = n_cols // peer_width
        ppc = chunk_width // peer_width
        kernel_assert(BxS_tok % batch_groups == 0, f"tokens ({BxS_tok}) must divide by groups ({batch_groups})")
        tok_span = BxS_tok // batch_groups
    kernel_assert(
        len(inp_slabs) == max(1, batch_groups if peer_width else 1),
        f"got {len(inp_slabs)} input slabs for batch_groups={batch_groups}",
    )
    kernel_assert(
        _slab_sizes_match([tuple(t.shape) for t in inp_slabs], tok_span * chunks * chunk_width),
        f"each input slab must hold tokens-per-slab * vocab = {tok_span * chunks * chunk_width} elements, "
        f"got {[int(np.prod(t.shape)) for t in inp_slabs]}",
    )
    # See the desnake_dge comment in _inline_topk: unconditional, and the reason is a measured
    # silent index corruption, not tidiness.
    desnake_dge = nisa.dge_mode.unknown
    cand_v = nl.ndarray((pad_rows, k), dtype=nl.bfloat16, buffer=nl.private_hbm)
    cand_i = nl.ndarray((pad_rows, k), dtype=nl.float32, buffer=nl.private_hbm)
    cand_v_flat = cand_v.reshape((pad_rows * k,))
    cand_i_flat = cand_i.reshape((pad_rows * k,))

    # Sentinel the pad rows once. They are whole reduce rows, so BFLOAT16_MIN here cannot
    # outrank a real -inf in a REAL row -- it only has to lose within its own row, and that row's
    # result is never read. cand_i's pad rows are left as they are for the same reason.
    if pad_rows > BxS:
        n_pad = pad_rows - BxS
        for t in range(div_ceil(n_pad, PMAX)):
            rows = min(PMAX, n_pad - t * PMAX)
            pad_tile = nl.ndarray((PMAX, k), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=pad_tile[0:rows, :], value=BFLOAT16_MIN)
            nisa.dma_copy(dst=cand_v[nl.ds(BxS + t * PMAX, rows), :], src=pad_tile[0:rows, :])

    lnc_row_start = prg_id * per_lnc_BxS
    n_tiles = div_ceil(per_lnc_BxS, GROUPS_PER_TILE)
    lnc_row_end = min(lnc_row_start + per_lnc_BxS, BxS)

    # Coalesced + double-buffered load, exactly as in _inline_topk's phase 1.
    fast_coalesced = (not has_pad) and (per_lnc_BxS % GROUPS_PER_TILE == 0) and (lnc_row_start + per_lnc_BxS <= BxS)
    kernel_assert(
        peer_width == 0 or fast_coalesced,
        "interleaved_peer_width requires the coalesced whole-tile load path",
    )
    CHUNK_TILES = 1
    n_chunks = div_ceil(n_tiles, CHUNK_TILES)
    _load_engines = [nisa.engine.sync, nisa.engine.scalar]
    # Interleaved path: one buffer PER TILE (four at the production shapes, ~3.1 MB of SBUF
    # that is entirely free post-collective). With buffer REUSE, a reused buffer's load
    # must wait for the sort still reading it, and everything queued behind that descriptor
    # stalls too -- measured as a ~7us gap between the first two sorts at gbs=1024. With no
    # reuse every load free-runs from the collective's completion. The contiguous path keeps
    # its historical 3 (it serves every other suite; don't perturb those schedules).
    N_PREFETCH_BUFS = min(4 if peer_width else 3, n_chunks)
    chunk_bufs = []

    def _load_chunk(dst3, first_s1_row, tiles, engine_phase):
        """One chunk-buffer fill from either source layout.

        Contiguous [BxS_tok, vocab]: the whole 128-partition tile is ONE run-per-partition
        DMA. Interleaved receive layout: the partition stride is uniform only WITHIN a
        16-partition group (each group is a different (token, chunk) with its own base), so
        it is one DMA per group -- 8 descriptors instead of 1, all unit-innermost-stride,
        ALTERNATING engines per group: descriptor generation (~0.9us per dma_copy on one
        engine) is what paces these loads, so splitting a tile's eight across both load
        engines halves its ready time.
        """
        if not peer_width:
            nisa.dma_copy(
                dst=dst3[nl.ds(0, PMAX), nl.ds(0, tiles), nl.ds(0, n_cols)],
                src=inp_flat.ap(
                    pattern=[[n_cols, PMAX], [PMAX * n_cols, tiles], [1, n_cols]],
                    offset=first_s1_row * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=_load_engines[engine_phase % 2],
            )
            return
        for tt in range(tiles):
            for grp in range(GROUPS_PER_TILE):
                r = first_s1_row + tt * GROUPS_PER_TILE + grp
                j = r // chunks
                c = r % chunks
                nisa.dma_copy(
                    dst=dst3[nl.ds(grp * PARTS_PER_GROUP, PARTS_PER_GROUP), nl.ds(tt, 1), nl.ds(0, n_cols)],
                    # Token j's SLAB (index j // tok_span; slab 0 for every j at batch_groups
                    # == 1, where tok_span == BxS_tok), then the chunk and token offsets
                    # WITHIN that slab's peer-major interleave. At one slab this is the
                    # original pattern verbatim -- bitwise-identical descriptors.
                    src=inp_flats[j // tok_span].ap(
                        pattern=[
                            [ppp * tok_span * peer_width, PARTS_PER_GROUP],
                            [tok_span * peer_width, ppp],
                            [1, peer_width],
                        ],
                        offset=c * ppc * tok_span * peer_width + (j % tok_span) * peer_width,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=_load_engines[(engine_phase + grp) % 2],
                )

    if fast_coalesced:
        for _b in range(N_PREFETCH_BUFS):
            chunk_bufs.append(nl.ndarray((PMAX, CHUNK_TILES, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf))
        for pc in range(min(N_PREFETCH_BUFS - 1, n_chunks)):
            prime_row_start = lnc_row_start + pc * CHUNK_TILES * GROUPS_PER_TILE
            prime_tiles = min(CHUNK_TILES, n_tiles - pc * CHUNK_TILES)
            _load_chunk(chunk_bufs[pc % N_PREFETCH_BUFS], prime_row_start, prime_tiles, pc)

    # The constant NaN floor, hoisted OUT of the tile loop: it is the same tile every time, and
    # at the GPT-OSS chunk width the memset covers 128 x 3,142 bf16, which is not a rounding
    # error when there are four tiles. Not built at all under the caller's finite-input
    # contract (config.skip_nan_fold) -- see the fold site in the tile loop.
    _nan_floor = None
    if not skip_nan_fold:
        _nan_floor = nl.ndarray((PMAX, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=_nan_floor, value=BFLOAT16_MIN)

    # --- The per-chunk global-id offset, as a PER-PARTITION CONSTANT ---------------------
    # The offset a row needs is (chunk index) * chunk_width, and within a tile the chunk index
    # is g % P where g = pp // 16 is the partition's group. The obvious spelling -- one
    # tensor_scalar add per group, on partitions [16g, 16g+16) -- IS NOT EXPRESSIBLE: the
    # hardware requires a 16-partition access to start at partition 0, 32, 64 or 96, and the
    # compiler rejects anything else ("Expected 'dst' partition start in {0, 32, 64, 96} for a
    # 16-partition access"). So build it as a tile instead, with iota's channel_multiplier,
    # which is the one thing that varies a generated value BY PARTITION.
    #
    # It is then folded into the remap rather than added afterwards. Because chunk_width is
    # exactly 16*n_cols (asserted below), the offset is a whole number of snake partitions:
    #     global = (r % P)*chunk_width + (s % 16)*n_cols + (s // 16)
    #            = (16*(r % P) + (s % 16)) * n_cols + (s // 16)
    # where r is the s1 ROW index (row j*P + c is chunk c of token j, so chunk == r % P).
    # so it costs one uint32 add on a k_cols-wide tile instead of P-1 adds on a P*k-wide one.
    #
    # The row's chunk is derived from the PARTITION index: iota seeds 16*(first_row % P) + pp
    # and a bitwise_and with (16P - 1) & ~15 (both powers of two minus one; pick_chunks only
    # returns power-of-two P) reduces it to 16*((first_row + pp//16) % P) exactly. When P
    # divides GROUPS_PER_TILE the tile's first row is always 0 mod P, so ONE hoisted tile
    # serves every iteration; for P > GROUPS_PER_TILE the residue changes per tile and the
    # base must be rebuilt with that tile's first row (a hoisted P<=8 tile reused at P=16
    # gave every row past tile 0 an id short by GROUPS_PER_TILE*chunk_width -- in-range,
    # silently wrong, found by review simulation 2026-08-31).
    _base_mask = (PARTS_PER_GROUP * chunks - 1) & ~(PARTS_PER_GROUP - 1)

    def _build_chunk_base(first_row):
        cb = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(
            dst=cb,
            pattern=[[0, k_cols]],
            offset=PARTS_PER_GROUP * (first_row % chunks),
            channel_multiplier=1,
        )
        nisa.tensor_scalar(dst=cb, data=cb, op0=nl.bitwise_and, operand0=_base_mask)
        return cb

    chunk_base = _build_chunk_base(0) if GROUPS_PER_TILE % chunks == 0 else None

    for tile_idx in nl.sequential_range(n_tiles):
        tile_row_start = lnc_row_start + tile_idx * GROUPS_PER_TILE
        tile_row_end = min(tile_row_start + GROUPS_PER_TILE, lnc_row_end)
        n_rows = tile_row_end - tile_row_start
        if n_rows <= 0:
            continue
        par_dim = n_rows * PARTS_PER_GROUP

        if fast_coalesced:
            chunk_idx = tile_idx // CHUNK_TILES
            tile_in_chunk = tile_idx % CHUNK_TILES
            prefetch_chunk = chunk_idx + (N_PREFETCH_BUFS - 1)
            if tile_in_chunk == 0 and prefetch_chunk < n_chunks:
                pf_first_tile = prefetch_chunk * CHUNK_TILES
                pf_row_start = lnc_row_start + pf_first_tile * GROUPS_PER_TILE
                pf_tiles = min(CHUNK_TILES, n_tiles - pf_first_tile)
                _load_chunk(
                    chunk_bufs[prefetch_chunk % N_PREFETCH_BUFS],
                    pf_row_start,
                    pf_tiles,
                    prefetch_chunk,
                )
            src_snake = chunk_bufs[chunk_idx % N_PREFETCH_BUFS][nl.ds(0, PMAX), tile_in_chunk, nl.ds(0, n_cols)]
        elif not has_pad:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=inp_flat.ap(pattern=[[n_cols, par_dim], [1, n_cols]], offset=tile_row_start * vocab),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        else:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # -inf, NOT the most negative finite bf16: a finite sentinel outranks a real -inf
            # logit and pulls padding into the top-k. See _inline_topk's memset comment.
            nisa.memset(dst=src_snake, value=float("-inf"))
            for g in nl.affine_range(n_rows):
                row_off = (tile_row_start + g) * vocab
                base = g * PARTS_PER_GROUP
                q_full = vocab // n_cols
                r_partial = vocab % n_cols
                if q_full > 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base, q_full), nl.ds(0, n_cols)],
                        src=inp_flat.ap(pattern=[[n_cols, q_full], [1, n_cols]], offset=row_off),
                    )
                if r_partial != 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base + q_full, 1), nl.ds(0, r_partial)],
                        src=inp_flat.ap(pattern=[[1, r_partial]], offset=row_off + q_full * n_cols),
                    )

        # Fold NaN out of the snake BEFORE nisa.topk selects from it: afterwards the instruction
        # has already emitted a (value, position) pair that does not correspond, and no later
        # clamp can undo that. (x != x) is true only for NaN. Skipped under the caller's
        # finite-input contract (config.skip_nan_fold): with no NaN the predicate can never
        # fire, yet the compare + predicated copy span the full chunk width -- the widest
        # non-topk work in this loop. +/-inf are orderable and never needed the fold.
        if not skip_nan_fold:
            _nan_mask = nl.ndarray((par_dim, n_cols), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=_nan_mask[nl.ds(0, par_dim), :],
                data1=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                data2=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                op=nl.not_equal,
            )
            nisa.tensor_copy_predicated(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=_nan_floor[nl.ds(0, par_dim), :],
                predicate=_nan_mask[nl.ds(0, par_dim), :],
            )

        val_snake = nl.ndarray((par_dim, k_pad), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx_snake = nl.ndarray((par_dim, k_pad), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.topk(val_dst=val_snake[:, nl.ds(0, k)], idx_dst=idx_snake[:, nl.ds(0, k)], src=src_snake, n=full_n)

        # Stage the positions through a dge_mode.unknown DMA before ANY compute engine reads
        # them. The remap below used to read idx_snake directly on Vector -- a direct
        # GpSIMD-write -> Vector-read edge that neither gpsimd_topk's phase 2 (which reads its
        # positions only after the de-snake store/reload round trip) nor any other path in this
        # file has. That edge is timing-unsafe: with the debug taps shifting the schedule, the
        # bxs=128 seed sweep FAILED THE PAIRING GATE on hardware (58 slots on one row, ids
        # plausible and in-range, values exactly right) and the L2 tap named this remap -- the
        # Vector ops had read idx_snake before nisa.topk's stores were visible, exactly the
        # store-visibility defect measured on nc_n_gather in the fused tail. The staging DMA is
        # the same handoff the de-snake stores themselves use (descriptor generation on the
        # GpSIMD queue, completion on a real DMA semaphore), which is the highest-validated
        # GpSIMD-producer idiom in this kernel.
        idx_stage = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=idx_stage[nl.ds(0, par_dim), :],
            src=idx_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

        # --- The remap, ON THE SNAKE TILE: k_cols wide, not k ---
        # The blocked load put vocab index p*n_cols + c at snake position s = p + 16*c, so the
        # chunk-local id is (s % 16)*n_cols + (s // 16), and the global id folds the chunk
        # offset into the same multiply (see chunk_base):
        #     global = (16*(g % P) + (s % 16)) * n_cols + (s // 16)
        # 16 is a power of two, so the split is EXACT with bit-ops on the uint32 the instruction
        # already reports -- and unlike the phase-2 version there is no float32 -> uint32 cast
        # first, because idx_stage IS uint32.
        c_u32 = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        q_u32 = nl.ndarray((PMAX, k_cols), dtype=nl.uint32, buffer=nl.sbuf)
        c_f32 = nl.ndarray((PMAX, k_cols), dtype=nl.float32, buffer=nl.sbuf)
        gid = nl.ndarray((PMAX, k_cols), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=c_u32[nl.ds(0, par_dim), :],
            data=idx_stage[nl.ds(0, par_dim), :],
            op0=nl.right_shift,
            operand0=4,  # log2(PARTS_PER_GROUP); an int method call is rejected by the parser
        )
        # Read from idx_stage, NOT from c_u32: two ops reading one source and writing two
        # distinct destinations have no hazard between them, whereas chaining them through a
        # shared buffer creates exactly the write-after-read pair that has bitten this kernel.
        nisa.tensor_scalar(
            dst=q_u32[nl.ds(0, par_dim), :],
            data=idx_stage[nl.ds(0, par_dim), :],
            op0=nl.bitwise_and,
            operand0=PARTS_PER_GROUP - 1,
        )
        # + 16*(row % P), the chunk offset expressed in snake partitions. Both operands are
        # uint32 and bounded by 15 and 16*(P-1) <= 2032, so the sum cannot overflow anything
        # (and the fp32 gid below stays exact: ids are < 2^24 by kernel contract).
        _cb = chunk_base if chunk_base is not None else _build_chunk_base(tile_row_start)
        nisa.tensor_tensor(
            dst=q_u32[nl.ds(0, par_dim), :],
            data1=q_u32[nl.ds(0, par_dim), :],
            data2=_cb[nl.ds(0, par_dim), :],
            op=nl.add,
        )
        nisa.tensor_copy(dst=c_f32[nl.ds(0, par_dim), :], src=c_u32[nl.ds(0, par_dim), :])
        nisa.tensor_scalar(
            dst=gid[nl.ds(0, par_dim), :],
            data=q_u32[nl.ds(0, par_dim), :],
            op0=nl.multiply,
            operand0=float(n_cols),
        )
        nisa.tensor_tensor(
            dst=gid[nl.ds(0, par_dim), :],
            data1=gid[nl.ds(0, par_dim), :],
            data2=c_f32[nl.ds(0, par_dim), :],
            op=nl.add,
        )
        # Every id must land in [0, full vocab). It can only escape if the snake position was
        # itself garbage, which happens when the values are not orderable (nisa.topk's reported
        # position is then undefined). Such an id is already unpaired from its value, so the only
        # thing left to guarantee is the range -- an out-of-range id handed to a table gather
        # faults the device. Clamping to the FULL vocab rather than to this chunk is a deliberate
        # weakening: the offset is now folded in before the clamp, so a garbage id can land in a
        # neighbouring chunk's range. Both are in-contract, and neither is paired.
        nisa.tensor_scalar(
            dst=gid[nl.ds(0, par_dim), :],
            data=gid[nl.ds(0, par_dim), :],
            op0=nl.minimum,
            operand0=float(chunks * vocab - 1),
        )

        # TAP: the RAW snake positions, de-snaked through the same pattern as everything else so
        # the tap means exactly what it means on the general path. It reads idx_stage -- the
        # staged buffer the remap consumed -- so an L2 break really does mean the remap's own
        # arithmetic disagreed with its own input, not that the two taps read one racy buffer
        # at different times.
        if tap_pos is not None:
            nisa.dma_copy(
                dst=tap_pos.reshape((BxS * k,)).ap(
                    pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_row_start * k_pad
                ),
                src=idx_stage[nl.ds(0, par_dim), :],
                dge_mode=desnake_dge,
            )

        # De-snake straight into the candidate arrays. One contiguous par_dim-partition DMA per
        # tensor per tile, as in _inline_topk -- the only difference is where it lands.
        tile_out_off = tile_row_start * k_pad
        nisa.dma_copy(
            dst=cand_v_flat.ap(pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off),
            src=val_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )
        nisa.dma_copy(
            dst=cand_i_flat.ap(pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off),
            src=gid[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

    return cand_v, cand_i


def _inline_topk(
    inp: nl.NkiTensor, config, tap_pos=None, fused=None, skip_nan_fold=False
) -> Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """Top-k over ``inp[BxS, V]`` with ``nisa.topk`` -- inlined copy of ``gpsimd_topk``.

    Args:
        inp: ``[BxS, V]`` bfloat16 tensor in HBM, ``8 <= V < 65536``.
        config: a ``GpsimdTopkConfig`` (build it with ``create_gpsimd_topk_config``).
        tap_pos: optional ``[BxS, k]`` float32 HBM tap. Receives the RAW snake positions phase 2
            worked from, before the snake->vocab remap. See DEBUG TAPS in the module docstring.
        fused: optional tuple (slots ``_FUSE_*``) that replaces the plain output store with the
            REDUCE'S TAIL: translate this call's positions through the caller's candidate-id
            array and write the final values and global ids directly. Only meaningful for the
            second stage, whose "vocab index" IS a candidate position. See WHY THE REDUCE FUSES
            ITS TAIL below.

    Returns:
        ``(values, indices, desnaked_idx [BxS, k_pad] f32)``. Without ``fused`` the first two are
        this call's own ``[BxS, k]`` private_hbm buffers, holding indices into ``inp``'s last
        dimension, descending when ``config.sorted``; with ``fused`` they are the caller's own
        destinations, already carrying global ids. The third element is the phase-1 de-snake
        index buffer, handed back so a caller can re-read it (the tap that distinguishes a
        late-landing store from a bad top-k) -- returning a handle to a buffer that already
        exists costs nothing.
    """
    _validate_gpsimd_topk(config)

    BxS = config.BxS
    vocab = config.vocab_size
    k = config.k
    index_dtype = config.index_dtype

    # Resolve runtime sharding (LNC). BxS == 1 collapses to a single program. chunked_topk has
    # already asserted n_prgs == 1, so this only ever resolves to (1, 0) here; it is kept
    # verbatim so the inline stays a faithful copy.
    shard_info = get_verified_program_sharding_info("gpsimd_topk", (0, 1), 2)
    if BxS > 1:
        n_prgs = shard_info[1]
        prg_id = shard_info[2]
    else:
        n_prgs = 1
        prg_id = 0

    per_lnc_BxS = div_ceil(BxS, n_prgs)

    # Without fusion this stage publishes its own [BxS, k] results; with it the caller's
    # destinations are written in place and these are never allocated.
    if fused is None:
        topk_values = nl.ndarray((BxS, k), dtype=inp.dtype, buffer=nl.private_hbm)
        topk_indices = nl.ndarray((BxS, k), dtype=index_dtype, buffer=nl.private_hbm)
    else:
        topk_values = fused[_FUSE_OUT_VALS]
        topk_indices = fused[_FUSE_OUT_IDX]

    # Snake free-dim sizes (BLOCKED layout: snake[p, c] = inp[row, p*n_cols + c], so snake
    # position s = p + 16*c holds vocab index p*n_cols + c -- inverted by the remap below).
    n_cols = div_ceil(vocab, PARTS_PER_GROUP)
    full_n = PARTS_PER_GROUP * n_cols  # the n passed to nisa.topk; padded slots hold -inf
    q_full = vocab // n_cols  # fully-valid partitions
    r_partial = vocab % n_cols  # valid elements in the partial partition q_full
    has_pad = full_n != vocab  # some snake slots are padding (vocab % 16 != 0)

    inp_flat = inp.reshape((BxS * vocab,))

    k_cols = div_ceil(k, PARTS_PER_GROUP)
    k_pad = k_cols * PARTS_PER_GROUP  # >= k; padded snake-column width

    # fast_dma_safe: this shard's phase-1 tiles are ALL full 128-partition tiles, which is what
    # makes a bf16 de-snake round trip safe against the narrow-dtype strided store. It does NOT
    # select the DGE mode -- see the desnake_dge comment.
    fast_dma_safe = (per_lnc_BxS % GROUPS_PER_TILE == 0) and (BxS == n_prgs * per_lnc_BxS)
    # Sorted output is served EXCLUSIVELY by the hardware sort -- the software sort was
    # deleted (see THE SOFTWARE SORT IS GONE in the module docstring). _validate_chunked_topk
    # has already refused a sorted request without the capability; this assert is the
    # belt-and-suspenders for any direct caller of this helper.
    if config.sorted:
        kernel_assert(
            _HW_SORT_AVAILABLE and k <= _HW_SORT_MAX_K,
            "sorted top-k requires nisa.topk(sorted=True) (patched nki + neuronx-cc >= 2.0.280935) "
            f"and k <= {_HW_SORT_MAX_K}; the software sort was removed",
        )
    hw_sort = bool(config.sorted)
    # dge_mode.unknown UNCONDITIONALLY. dge_mode.none pre-generates the de-snake descriptors
    # out-of-band and takes the store off GpSIMD, so nothing orders it against the phase-2
    # reload of the same private_hbm region; measured at 128 ranks, 2 of 128 row blocks came
    # back with REPEATED vocab ids and exactly-right values. Costs +2.7%. See THE STORE/RELOAD
    # RACE IS REAL in gpsimd_topk -- and its warning that a variant which merely narrows the
    # window passes a 128-rank run by luck.
    desnake_dge = nisa.dge_mode.unknown
    asc_val_dtype = nl.bfloat16 if fast_dma_safe else nl.float32
    asc_val_hbm = nl.ndarray((BxS, k_pad), dtype=asc_val_dtype, buffer=nl.private_hbm)
    asc_idx_hbm = nl.ndarray((BxS, k_pad), dtype=nl.float32, buffer=nl.private_hbm)
    asc_val_flat = asc_val_hbm.reshape((BxS * k_pad,))
    asc_idx_flat = asc_idx_hbm.reshape((BxS * k_pad,))

    lnc_row_start = prg_id * per_lnc_BxS
    n_tiles = div_ceil(per_lnc_BxS, GROUPS_PER_TILE)
    lnc_row_end = min(lnc_row_start + per_lnc_BxS, BxS)

    # --- Coalesced + double-buffered phase-1 load ------------------------------------------
    # When 16 | vocab and every tile is full, CHUNK_TILES tiles are ONE contiguous HBM block
    # (partition stride n_cols, tile stride 128*n_cols, free stride 1), so the load is a few
    # large HWDGE transfers into rolling buffers instead of one latency-bound transfer per
    # tile. N_PREFETCH_BUFS distinct buffers keep loads in flight while GpSIMD drains another.
    fast_coalesced = (not has_pad) and (per_lnc_BxS % GROUPS_PER_TILE == 0) and (lnc_row_start + per_lnc_BxS <= BxS)
    CHUNK_TILES = 1
    n_chunks = div_ceil(n_tiles, CHUNK_TILES)
    # Round-robin the loads across two DGE sequencer queues so consecutive loads run
    # concurrently instead of serializing on one sync queue.
    _load_engines = [nisa.engine.sync, nisa.engine.scalar]
    N_PREFETCH_BUFS = min(3, n_chunks)
    chunk_bufs = []
    if fast_coalesced:
        for _b in range(N_PREFETCH_BUFS):
            chunk_bufs.append(nl.ndarray((PMAX, CHUNK_TILES, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf))
        n_prime = min(N_PREFETCH_BUFS - 1, n_chunks)
        for pc in range(n_prime):
            prime_first_tile = pc * CHUNK_TILES
            prime_row_start = lnc_row_start + prime_first_tile * GROUPS_PER_TILE
            # The last chunk may hold fewer tiles; clamp so the DMA never over-reads.
            prime_tiles = min(CHUNK_TILES, n_tiles - prime_first_tile)
            nisa.dma_copy(
                dst=chunk_bufs[pc % N_PREFETCH_BUFS][nl.ds(0, PMAX), nl.ds(0, prime_tiles), nl.ds(0, n_cols)],
                src=inp_flat.ap(
                    pattern=[[n_cols, PMAX], [PMAX * n_cols, prime_tiles], [1, n_cols]],
                    offset=prime_row_start * vocab,
                ),
                dge_mode=nisa.dge_mode.hwdge,
                engine=_load_engines[pc % 2],
            )

    # =====================================================================
    # PHASE 1 -- per-8-row tile: blocked load -> nisa.topk -> de-snake the K
    # (value, snake-index) pairs into the contiguous [BxS, k_pad] HBM buffers.
    # The descending sort is NOT done here: it runs ONCE in phase 2 over all
    # this core's rows (one row per partition, up to 128), because max8 /
    # nc_match_replace8 / nc_n_gather are per-partition free-dim ops, so
    # widening the sort from 8 to 128 partitions is free.
    # =====================================================================
    for tile_idx in nl.sequential_range(n_tiles):
        tile_row_start = lnc_row_start + tile_idx * GROUPS_PER_TILE
        tile_row_end = min(tile_row_start + GROUPS_PER_TILE, lnc_row_end)
        n_rows = tile_row_end - tile_row_start
        if n_rows <= 0:
            continue

        par_dim = n_rows * PARTS_PER_GROUP  # multiple of 16

        if fast_coalesced:
            chunk_idx = tile_idx // CHUNK_TILES
            tile_in_chunk = tile_idx % CHUNK_TILES
            # Keep the pipeline full: at the first tile of a chunk, prefetch the chunk
            # (N_PREFETCH_BUFS-1) ahead into a slot nothing is still consuming, so no
            # anti-dependency forces the load to wait.
            prefetch_chunk = chunk_idx + (N_PREFETCH_BUFS - 1)
            if tile_in_chunk == 0 and prefetch_chunk < n_chunks:
                pf_first_tile = prefetch_chunk * CHUNK_TILES
                pf_row_start = lnc_row_start + pf_first_tile * GROUPS_PER_TILE
                pf_tiles = min(CHUNK_TILES, n_tiles - pf_first_tile)
                nisa.dma_copy(
                    dst=chunk_bufs[prefetch_chunk % N_PREFETCH_BUFS][
                        nl.ds(0, PMAX), nl.ds(0, pf_tiles), nl.ds(0, n_cols)
                    ],
                    src=inp_flat.ap(
                        pattern=[[n_cols, PMAX], [PMAX * n_cols, pf_tiles], [1, n_cols]],
                        offset=pf_row_start * vocab,
                    ),
                    dge_mode=nisa.dge_mode.hwdge,
                    engine=_load_engines[prefetch_chunk % 2],
                )
            src_snake = chunk_bufs[chunk_idx % N_PREFETCH_BUFS][nl.ds(0, PMAX), tile_in_chunk, nl.ds(0, n_cols)]
        elif not has_pad:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # 16 | vocab but the coalesced precondition does not hold. The whole-tile blocked
            # load is still ONE contiguous HBM run (partition stride n_cols, free stride 1),
            # emitted as a single par_dim-partition DMA. HWDGE on Sync keeps descriptor
            # generation off GpSIMD so the load overlaps the previous tile's top-k.
            nisa.dma_copy(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=inp_flat.ap(pattern=[[n_cols, par_dim], [1, n_cols]], offset=tile_row_start * vocab),
                dge_mode=nisa.dge_mode.hwdge,
                engine=nisa.engine.sync,
            )
        else:
            src_snake = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            # vocab % 16 != 0: some snake slots are padding. The sentinel is -inf, NOT the
            # most-negative FINITE bf16: a finite sentinel outranks a real -inf logit, so on a
            # masked row padding wins the top-k and the kernel returns a value the input never
            # contained paired with a clamped index. See gpsimd_topk's memset comment.
            nisa.memset(dst=src_snake, value=float("-inf"))

            for g in nl.affine_range(n_rows):
                row = tile_row_start + g
                base = g * PARTS_PER_GROUP
                row_off = row * vocab
                if q_full > 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base, q_full), nl.ds(0, n_cols)],
                        src=inp_flat.ap(pattern=[[n_cols, q_full], [1, n_cols]], offset=row_off),
                    )
                if r_partial != 0:
                    nisa.dma_copy(
                        dst=src_snake[nl.ds(base + q_full, 1), nl.ds(0, r_partial)],
                        src=inp_flat.ap(pattern=[[1, r_partial]], offset=row_off + q_full * n_cols),
                    )

        # Fold NaN out of the snake BEFORE nisa.topk selects from it -- the earliest point NaN
        # can do damage, and the phase-2 clamps are far too late (the topk has already emitted
        # a (value, position) pair that does not correspond). (x != x) is true only for NaN.
        # Fold to BFLOAT16_MIN, not NEG_INF_FLOOR: src_snake is BF16, which cannot hold the
        # floor finitely. See the phase-1 NaN-fold comment in gpsimd_topk. The REDUCE stage
        # skips this (skip_nan_fold): its input is stage 1's own output plus BFLOAT16_MIN pad
        # rows, and stage 1 already folded -- so its fold can never fire, and it sat as two
        # full-width Vector ops on the reduce's critical path anyway.
        if not skip_nan_fold:
            _nan_mask = nl.ndarray((par_dim, n_cols), dtype=nl.uint8, buffer=nl.sbuf)
            nisa.tensor_tensor(
                dst=_nan_mask[nl.ds(0, par_dim), :],
                data1=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                data2=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                op=nl.not_equal,
            )
            _nan_floor = nl.ndarray((par_dim, n_cols), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.memset(dst=_nan_floor[nl.ds(0, par_dim), :], value=BFLOAT16_MIN)
            nisa.tensor_copy_predicated(
                dst=src_snake[nl.ds(0, par_dim), nl.ds(0, n_cols)],
                src=_nan_floor[nl.ds(0, par_dim), :],
                predicate=_nan_mask[nl.ds(0, par_dim), :],
            )

        # n = full_n (NOT vocab): with the blocked layout the valid vocab elements are not a
        # contiguous prefix of snake positions, so all full_n slots are active.
        val_snake = nl.ndarray((par_dim, k_pad), dtype=nl.bfloat16, buffer=nl.sbuf)
        idx_snake = nl.ndarray((par_dim, k_pad), dtype=nl.uint32, buffer=nl.sbuf)
        if hw_sort:
            # sorted=True puts rank j at snake position j, which is what lets phase 2 replace
            # the entire descending sort with a fixed column permutation. Two call forms because
            # passing `sorted=` to an nki without it is a trace-time TypeError.
            nisa.topk(
                val_dst=val_snake[:, nl.ds(0, k)],
                idx_dst=idx_snake[:, nl.ds(0, k)],
                src=src_snake,
                n=full_n,
                sorted=True,
            )
        else:
            nisa.topk(val_dst=val_snake[:, nl.ds(0, k)], idx_dst=idx_snake[:, nl.ds(0, k)], src=src_snake, n=full_n)
        # k_pad > k leaves snake positions [k, k_pad) unwritten. They live at a PARTITION
        # SUBRANGE here, which a full-partition memset cannot target, so they are masked after
        # the de-snake reload in phase 2 where the layout makes it a legal free-dim write.
        # (The hw_sort path needs no masking at all: it reads BY POSITION, and positions
        # [k, k_pad) are never among the ranks it asks for.)

        # --- De-snake the topk output into a CONTIGUOUS [rows, k_pad] HBM tile ---
        # asc[row, p*k_cols + c] = val_snake[base + p, c]. Because k_pad == 16*k_cols,
        # consecutive row-groups are exactly k_pad apart, so the WHOLE par_dim-partition tile
        # is ONE contiguous HBM run: a single DMA per tile rather than n_rows of them (the
        # de-snake stores were the top serial contributors in the HW profile).
        tile_out_off = tile_row_start * k_pad
        nisa.dma_copy(
            dst=asc_val_flat.ap(pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off),
            src=val_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )
        nisa.dma_copy(
            dst=asc_idx_flat.ap(pattern=[[k_cols, par_dim], [1, k_cols]], offset=tile_out_off),
            src=idx_snake[nl.ds(0, par_dim), nl.ds(0, k_cols)],
            dge_mode=desnake_dge,
        )

    # =====================================================================
    # PHASE 2 -- batch-parallel descending sort, ONE invocation over up to 128
    # rows (one row per partition) rather than one per 8-row tile.
    # =====================================================================
    n_pass = div_ceil(k, 8)
    log2_ppg = 4  # == log2(PARTS_PER_GROUP)
    n_sort_tiles = div_ceil(per_lnc_BxS, PMAX)

    for sort_idx in nl.sequential_range(n_sort_tiles):
        sort_row_start = lnc_row_start + sort_idx * PMAX
        sort_row_end = min(sort_row_start + PMAX, lnc_row_end)
        n_srows = sort_row_end - sort_row_start
        if n_srows <= 0:
            continue

        out_v = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)
        out_i = nl.ndarray((PMAX, n_pass * 8), dtype=nl.float32, buffer=nl.sbuf)

        if hw_sort:
            # ===== THERE IS NO SORT ============================================
            # nisa.topk(sorted=True) emitted rank j at snake position j, and the de-snake places
            # snake position pos at column (pos % 16)*k_cols + (pos // 16). Both are compile-time
            # facts, so recovering descending order is a FIXED PERMUTATION of columns. Writing
            # j = a + 16b with a = j % 16 and b = j // 16, rank j lives at row offset
            # a*k_cols + b -- so a [k_cols, 16] destination whose flat column index is b*16 + a
            # holds the ranks in order once source column a*k_cols + b lands at (b, a).
            #
            # The reload itself -- which the software path performs anyway -- stays ONE
            # CONTIGUOUS RUN PER ROW, and the permutation runs ON CHIP: 16 tensor_copy ops per
            # tensor, each moving the contiguous k_cols-wide block for one value of `a` into the
            # strided [:, :, a] slice. Folding the permutation into the reload's access pattern
            # instead is NOT free -- the last pattern level is the fastest-varying, so it makes
            # every element its own 4-byte transfer (+1,968 packets, +50 us at gbs=128; see THE
            # 4-BYTE DMA PROBLEM in the module docstring, including why landing this exposed a
            # latent race elsewhere).
            #
            # What the branch deletes, per sort tile: 16 max8, 16 nc_match_replace8, 16 gathers,
            # the two reverse-gathers, 8 bitonic merge stages of 7 ops each, three synchronising
            # SBUF->SBUF DMAs, and the two sort-key clamps. About 110 instructions become 34.
            #
            # It also removes two whole classes of hazard rather than mitigating them:
            #   * the value-addressed index recovery. nc_match_replace8 found a value's position
            #     by equality, so duplicate extremes left a dst_idx UNDEFINED and a real -inf
            #     could alias the marker it stamps -- the defect _clamp_sort_keys and
            #     NEG_INF_FLOOR exist to work around. Here the instruction reports the index of
            #     the value it selected, so there is nothing to look up and nothing to alias.
            #     Both clamps are therefore absent, not merely unnecessary.
            #   * the k_pad > k padding. The software path must mask the unwritten snake
            #     positions [k, k_pad) because garbage there wins max8. This path never reads
            #     them: it asks for ranks, and no rank maps to an unwritten position.
            #
            # The NaN fold in phase 1 STAYS. The ISA is explicit that "the behavior for NaN
            # inputs is undefined; a NaN may appear in the output spuriously" -- sorting does not
            # make an unorderable key orderable, so it must still be folded before selection.
            pv = nl.ndarray((PMAX, k_cols, PARTS_PER_GROUP), dtype=nl.float32, buffer=nl.sbuf)
            pi = nl.ndarray((PMAX, k_cols, PARTS_PER_GROUP), dtype=nl.float32, buffer=nl.sbuf)
            lv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            li = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=lv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=li[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            for _a in range(PARTS_PER_GROUP):
                nisa.tensor_copy(
                    dst=pv[nl.ds(0, n_srows), :, _a], src=lv[nl.ds(0, n_srows), nl.ds(_a * k_cols, k_cols)]
                )
                nisa.tensor_copy(
                    dst=pi[nl.ds(0, n_srows), :, _a], src=li[nl.ds(0, n_srows), nl.ds(_a * k_cols, k_cols)]
                )
            # The 3D tile's free dims are contiguous, so this view is the [n_srows, k_pad]
            # descending array the shared tail below expects. k <= k_pad, so the tail's
            # [:, 0:k] slice takes ranks 0..k-1.
            out_v = pv.reshape((PMAX, k_pad))
            out_i = pi.reshape((PMAX, k_pad))
        else:
            # ===== Unsorted fast path (config.sorted == False) ================
            # Skip the sort; COMPACT the valid de-snaked columns into the output prefix
            # [0, k). nisa.topk writes only snake positions [0, k), and the de-snake scatters
            # those among the k_pad columns when k % 16 != 0; for k % 16 == 0 this is the
            # single full-width run (0, 0, k).
            cv = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            ci = nl.ndarray((PMAX, k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=cv[nl.ds(0, n_srows), :],
                src=asc_val_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            nisa.dma_copy(
                dst=ci[nl.ds(0, n_srows), :],
                src=asc_idx_hbm[nl.ds(sort_row_start, n_srows), :],
                dge_mode=desnake_dge,
            )
            _valid_runs = _valid_desnake_col_runs(k, k_cols, k_pad)
            # The parser frontend rejects tuple-unpacking in a for-target, so index explicitly.
            for _run_idx in range(len(_valid_runs)):
                _src_col = _valid_runs[_run_idx][0]
                _dst_col = _valid_runs[_run_idx][1]
                _run_len = _valid_runs[_run_idx][2]
                nisa.tensor_copy(
                    dst=out_v[nl.ds(0, n_srows), nl.ds(_dst_col, _run_len)],
                    src=cv[nl.ds(0, n_srows), nl.ds(_src_col, _run_len)],
                )
                nisa.tensor_copy(
                    dst=out_i[nl.ds(0, n_srows), nl.ds(_dst_col, _run_len)],
                    src=ci[nl.ds(0, n_srows), nl.ds(_src_col, _run_len)],
                )

        # TAP: the raw snake positions phase 2 ended up with, before the remap. Reads out_i
        # full-width, which the remap immediately below does too, so it adds no ordering edge
        # that is not already there -- the property that keeps a tap from hiding the race it is
        # meant to catch (see DEBUG TAPS).
        if tap_pos is not None:
            nisa.dma_copy(dst=tap_pos[nl.ds(sort_row_start, n_srows), :], src=out_i[nl.ds(0, n_srows), nl.ds(0, k)])

        # --- Blocked-layout snake-position -> vocab-index remap ---
        # The blocked load put vocab index p*n_cols + c at snake position s = p + 16*c, so
        # v = (s % 16) * n_cols + (s // 16). 16 is a power of two, so the split is EXACT via
        # bit-ops on uint32 (c = s >> 4, p = s & 15); the recombine is float32 (exact for
        # v < 2^24) and casts back to uint32 on the store.
        # Exactly k columns wide, not the sort buffers' width: only [0, k) is ever stored, and
        # the branches above leave out_i at different widths (n_pass*8 for the unsorted compaction,
        # k_pad for the hardware one), so k is both the cheapest and the only common one.
        s_u32 = nl.ndarray((PMAX, k), dtype=nl.uint32, buffer=nl.sbuf)
        c_u32 = nl.ndarray((PMAX, k), dtype=nl.uint32, buffer=nl.sbuf)
        p_u32 = nl.ndarray((PMAX, k), dtype=nl.uint32, buffer=nl.sbuf)
        c_f32 = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
        remapped_i = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=s_u32[nl.ds(0, n_srows), :], src=out_i[nl.ds(0, n_srows), nl.ds(0, k)])
        nisa.tensor_scalar(
            dst=c_u32[nl.ds(0, n_srows), :],
            data=s_u32[nl.ds(0, n_srows), :],
            op0=nl.right_shift,
            operand0=log2_ppg,
        )
        nisa.tensor_scalar(
            dst=p_u32[nl.ds(0, n_srows), :],
            data=s_u32[nl.ds(0, n_srows), :],
            op0=nl.bitwise_and,
            operand0=PARTS_PER_GROUP - 1,
        )
        nisa.tensor_copy(dst=c_f32[nl.ds(0, n_srows), :], src=c_u32[nl.ds(0, n_srows), :])
        nisa.tensor_scalar(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data=p_u32[nl.ds(0, n_srows), :],
            op0=nl.multiply,
            operand0=float(n_cols),
        )
        nisa.tensor_tensor(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data1=remapped_i[nl.ds(0, n_srows), :],
            data2=c_f32[nl.ds(0, n_srows), :],
            op=nl.add,
        )
        # Every returned index must be in [0, vocab). Two ways it could escape: a padding slot
        # that tied real -inf data and remapped up to full_n - 1, or a gathered snake position
        # that was itself garbage. An out-of-range index breaks the contract and, if it feeds a
        # table gather, faults the device. Unconditional -- one Vector op on k columns.
        nisa.tensor_scalar(
            dst=remapped_i[nl.ds(0, n_srows), :],
            data=remapped_i[nl.ds(0, n_srows), :],
            op0=nl.minimum,
            operand0=float(vocab - 1),
        )

        if fused is None:
            nisa.dma_copy(dst=topk_values[nl.ds(sort_row_start, n_srows), :], src=out_v[nl.ds(0, n_srows), nl.ds(0, k)])
            nisa.dma_copy(
                dst=topk_indices[nl.ds(sort_row_start, n_srows), :], src=remapped_i[nl.ds(0, n_srows), nl.ds(0, k)]
            )
            continue

        # ===== WHY THE REDUCE FUSES ITS TAIL ==============================================
        # For the second stage `remapped_i` is not a vocab index at all -- this call's "vocab"
        # IS the candidate array -- so it is a POSITION that still has to be translated through
        # the caller's chunk-local ids. Unfused, that translation costs a full round trip: store
        # the positions to HBM as uint32, reload them, widen back to float32, and separately
        # store the values only for the caller to reload and re-store them. Every one of those
        # is removed here. The positions are already in SBUF, already float32, and already
        # clamped into [0, red_w) by the range clamp above -- which is exactly what
        # nc_n_gather requires of its indices -- so they can be used as they stand.
        n_out = fused[_FUSE_N_ROWS]
        rows = min(n_srows, n_out - sort_row_start)
        if rows <= 0:
            continue
        p_chunks = fused[_FUSE_CHUNKS]
        chunk_width = fused[_FUSE_WIDTH]
        red_w = p_chunks * k

        # Values: ONE fp32 SBUF -> bf16 HBM DMA, the cast riding on the transfer.
        nisa.dma_copy(dst=fused[_FUSE_OUT_VALS][nl.ds(sort_row_start, rows), :], src=out_v[nl.ds(0, rows), nl.ds(0, k)])

        # Index normalisation. fp32 throughout: every global id is < vocab <= 2^24, so the
        # uint32 -> fp32 -> uint32 round trip is exact.
        if fused[_FUSE_IDS_GLOBAL]:
            # _stage1_desnake_only already remapped and offset the ids, on a k_cols-wide tile,
            # and wrote them as float32 -- and the Tensor-Engine lookup below reads them
            # STRAIGHT FROM HBM (the d_all loads), so no SBUF copy of the full candidate row
            # exists on this path at all. Only the cand_i tap, which captures exactly what the
            # lookup consumes, materialises one.
            i_f = None
            if fused[_FUSE_TAP_CAND] is not None:
                i_f = nl.ndarray((PMAX, red_w), dtype=nl.float32, buffer=nl.sbuf)
                nisa.dma_copy(dst=i_f[0:rows, :], src=fused[_FUSE_CAND_I][nl.ds(sort_row_start, rows), :])
        else:
            i_f = nl.ndarray((PMAX, red_w), dtype=nl.float32, buffer=nl.sbuf)
            iu = nl.ndarray((PMAX, red_w), dtype=nl.uint32, buffer=nl.sbuf)
            nisa.dma_copy(dst=iu[0:rows, :], src=fused[_FUSE_CAND_I][nl.ds(sort_row_start, rows), :])
            nisa.tensor_copy(dst=i_f[0:rows, :], src=iu[0:rows, :])
            # global = c*width + chunk_local. c is constant per column block, so this is a
            # free-dim slice add per chunk. Chunk 0 needs no offset.
            for c in range(1, p_chunks):
                nisa.tensor_scalar(
                    dst=i_f[0:rows, nl.ds(c * k, k)],
                    data=i_f[0:rows, nl.ds(c * k, k)],
                    op0=nl.add,
                    operand0=float(c * chunk_width),
                )
        if fused[_FUSE_TAP_CAND] is not None:
            nisa.dma_copy(dst=fused[_FUSE_TAP_CAND][nl.ds(sort_row_start, rows), :], src=i_f[0:rows, :])
        if fused[_FUSE_TAP_POS] is not None:
            nisa.dma_copy(dst=fused[_FUSE_TAP_POS][nl.ds(sort_row_start, rows), :], src=remapped_i[0:rows, nl.ds(0, k)])

        # ===== THE POSITION -> ID LOOKUP RUNS ON THE TENSOR ENGINE, NOT ON A GATHER =========
        # got[r, j] = i_f[r, pos] with pos = remapped_i[r, j] is the one data-dependent lookup
        # in the kernel, and nc_n_gather CANNOT BE MADE SAFE HERE. That is a measured statement,
        # not a cautious one -- four escalating fence constructions all failed, each localised by
        # debug_taps (the L4 tap) plus the Instruction/Flow tables of its failing NEFF:
        #   * a Vector consumer with a correct FLOW_DEPENDENCE edge, dispatched 500 ns after the
        #     gather retired, read the row TAIL of its output as stale zeros;
        #   * a default-mode SBUF->SBUF DMA fence read the same mid-flight state (its descriptors
        #     are generated out-of-band, so nothing places it after the gather);
        #   * a same-dtype GpSIMD COPY fence was copy-propagated away by the compiler, re-exposing
        #     exactly the window whose fence was elided;
        #   * with an unfoldable GpSIMD CAST fence, the profile shows the Vector consumer STARTING
        #     119 ns into the cast's own 900 ns execution -- its semaphore wait already satisfied
        #     -- and two GpSIMD instructions visibly overlapping on the engine. A GpSIMD
        #     instruction's completion signal does not mean its stores are visible, and same-queue
        #     issue order does not serialise execution either.
        # nisa.topk's own consumers at SHORTER distances are always correct, so this is the gather
        # ucode's early completion signal, not a property of GpSIMD stores in general (the
        # multi-group corner of the same defect). The sort paths' gathers keep their historical
        # sync-DMA fences and their far-downstream consumers; THIS site had neither distance nor a
        # working fence, so the gather is not fenced here -- it is REMOVED.
        #
        # The replacement is an exact one-hot contraction on the Tensor Engine, whose PSUM handoff
        # to Vector is the most-validated producer/consumer edge on the chip. Split each position
        # as pos = q*c_w + c (c_w = min(128, red_w)):
        #     D[(g, q), c] = i_f[r0+g, q*c_w + c]                 the group's candidates, fp32
        #     B[(g, q), (g', j)] = (g*n_q + q == key[g', j])      key = g'*n_q + q_{g',j}
        #     M[c, (g', j)] = sum_p D[p, c] * B[p, (g', j)]       = i_f[r0+g', q_j*c_w + c]
        #     got[g', j] = sum_c M[c, (g', j)] * (c == c_{g',j})  the c-selection + ones-reduce
        # EXACTNESS: everything is fp32. Ids are < vocab <= 2^24, so fp32 holds them exactly; keys
        # and iotas are small integers; each one-hot column matches exactly one partition, so every
        # PSUM sum is one exact product plus zeros -- no accumulation error exists to reason about.
        # A padded tail chunk (c_w does not divide red_w) is memset to 0.0 so the matmul multiplies
        # B's zeros against 0.0 rather than uninitialised bits (0 * NaN is NaN).
        c_w = min(PMAX, red_w)
        n_q = div_ceil(red_w, c_w)
        # Rows per matmul group: the block-diagonal key trick batches g rows into one contraction
        # so long as their K fits the partition dim and their flattened outputs fit one PSUM-bank
        # moving tile. k > _PSUM_MOVING falls back to one row per group, tiled along k.
        _PSUM_MOVING = 512
        g_max = max(1, min(PMAX // n_q, _PSUM_MOVING // k if k <= _PSUM_MOVING else 1))
        # Shared small constants, sliced per use: one integer ramp down the partitions (serves
        # both the K-side and the c-side one-hot compares) and a ones vector for the broadcasts
        # and the partition reduction.
        iota_pf = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        _iota_pu = nl.ndarray((PMAX, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.iota(dst=_iota_pu, pattern=[[0, 1]], channel_multiplier=1)
        nisa.tensor_copy(dst=iota_pf, src=_iota_pu)
        ones_row = nl.ndarray((1, PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_row, value=1.0)
        ones_col = nl.ndarray((PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.memset(dst=ones_col, value=1.0)
        # bf16 twin of the ones vector, for the two BROADCAST matmuls (key_b, c_b) below.
        # Those carry only small integers (keys < 128, c < c_w <= 128 -- all exact in bf16's
        # 8 significand bits) and every PSUM element is ONE exact product, so running them at
        # the PE's bf16 rate instead of the quarter-rate fp32 mode is BITWISE free. The id
        # contraction (m_ps) and the ones-reduce (sel) carry vocab ids up to 2^24 and STAY
        # fp32 -- see the exactness paragraph above.
        ones_row_bf = nl.ndarray((1, PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.memset(dst=ones_row_bf, value=1.0)

        for r0 in range(0, rows, g_max):
            g = min(g_max, rows - r0)
            n_flat = g * k
            k_g = g * n_q
            # The group's positions, flattened onto one partition, and split into (q, c). The
            # per-row DMAs are partition moves; everything after them is exact fp32 arithmetic.
            pos_flat = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            for gi in range(g):
                nisa.dma_copy(
                    dst=pos_flat[0:1, nl.ds(gi * k, k)],
                    src=remapped_i[nl.ds(r0 + gi, 1), nl.ds(0, k)],
                )
            key_f = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            c_f = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            key_bf = nl.ndarray((1, n_flat), dtype=nl.bfloat16, buffer=nl.sbuf)
            c_bf = nl.ndarray((1, n_flat), dtype=nl.bfloat16, buffer=nl.sbuf)
            if n_q == 1:
                # red_w <= 128: q is identically zero and c is the position itself.
                nisa.tensor_copy(dst=c_f[0:1, :], src=pos_flat[0:1, :])
                nisa.memset(dst=key_f, value=0.0)
            else:
                # c_w == 128 here (red_w > 128), so the split is exact bit arithmetic.
                pos_u = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
                q_u = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
                c_u = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=pos_u[0:1, :], src=pos_flat[0:1, :])
                nisa.tensor_scalar(dst=q_u[0:1, :], data=pos_u[0:1, :], op0=nl.right_shift, operand0=7)
                nisa.tensor_scalar(dst=c_u[0:1, :], data=pos_u[0:1, :], op0=nl.bitwise_and, operand0=PMAX - 1)
                nisa.tensor_copy(dst=key_f[0:1, :], src=q_u[0:1, :])
                nisa.tensor_copy(dst=c_f[0:1, :], src=c_u[0:1, :])
            if g > 1:
                # key = g'*n_q + q: fold the row-within-group block into the key so ONE compare
                # builds the block-diagonal one-hot.
                for gi in range(g):
                    nisa.tensor_scalar(
                        dst=key_f[0:1, nl.ds(gi * k, k)],
                        data=key_f[0:1, nl.ds(gi * k, k)],
                        op0=nl.add,
                        operand0=float(gi * n_q),
                    )

            # bf16 twins of the finished key/c vectors, for the broadcast matmuls: keys and
            # c are integers below 128/c_w, exact in bf16, so this is a pure rate change.
            nisa.tensor_copy(dst=key_bf[0:1, :], src=key_f[0:1, :])
            nisa.tensor_copy(dst=c_bf[0:1, :], src=c_f[0:1, :])

            # The group's candidate ids, reshaped one row -> n_q partition chunks of c_w. A DMA
            # cannot fan one SBUF partition out to n_q partitions, so the reshape happens where
            # each path can afford it: the bypass path's ids are already global float32 in HBM,
            # where an access pattern reshapes for free (contiguous c_w-runs); the general path
            # computed i_f in SBUF, so it moves one single-partition chunk at a time.
            d_all = nl.ndarray((PMAX, c_w), dtype=nl.float32, buffer=nl.sbuf)
            if n_q * c_w != red_w:
                nisa.memset(dst=d_all[0:k_g, :], value=0.0)
            q_full = red_w // c_w
            rem = red_w % c_w
            if fused[_FUSE_IDS_GLOBAL]:
                cand_flat = fused[_FUSE_CAND_I].reshape((fused[_FUSE_CAND_I].shape[0] * red_w,))
                for gi in range(g):
                    row_off = (sort_row_start + r0 + gi) * red_w
                    if q_full > 0:
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q, q_full), nl.ds(0, c_w)],
                            src=cand_flat.ap(pattern=[[c_w, q_full], [1, c_w]], offset=row_off),
                        )
                    if rem:
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q + q_full, 1), nl.ds(0, rem)],
                            src=cand_flat.ap(pattern=[[1, rem]], offset=row_off + q_full * c_w),
                        )
            else:
                for gi in range(g):
                    for q in range(q_full):
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q + q, 1), nl.ds(0, c_w)],
                            src=i_f[nl.ds(r0 + gi, 1), nl.ds(q * c_w, c_w)],
                        )
                    if rem:
                        nisa.dma_copy(
                            dst=d_all[nl.ds(gi * n_q + q_full, 1), nl.ds(0, rem)],
                            src=i_f[nl.ds(r0 + gi, 1), nl.ds(q_full * c_w, rem)],
                        )

            got_flat = nl.ndarray((1, n_flat), dtype=nl.uint32, buffer=nl.sbuf)
            got_tap = None
            if fused[_FUSE_TAP_GOT] is not None:
                got_tap = nl.ndarray((1, n_flat), dtype=nl.float32, buffer=nl.sbuf)
            for t0 in range(0, n_flat, _PSUM_MOVING):
                n_t = min(_PSUM_MOVING, n_flat - t0)
                tile = nl.ds(t0, n_t)
                # Broadcast the tile's keys down K_g partitions and compare against the ramp:
                # the block-diagonal one-hot, one column of which selects one (row, q) chunk.
                key_b = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=key_b[0:k_g, 0:n_t], stationary=ones_row_bf[0:1, 0:k_g], moving=key_bf[0:1, tile])
                b_hot = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=b_hot[0:k_g, 0:n_t], data=key_b[0:k_g, 0:n_t], op0=nl.equal, operand0=iota_pf[0:k_g, 0:1]
                )
                # M[c, col] = i_f[row_col, q_col*c_w + c] for every c at once.
                m_ps = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=m_ps[0:c_w, 0:n_t], stationary=d_all[0:k_g, 0:c_w], moving=b_hot[0:k_g, 0:n_t])
                # The c-selection: broadcast c down c_w partitions, one-hot it against the same
                # ramp, zero everything but row c_col, and sum the partitions with a ones matmul.
                c_b = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=c_b[0:c_w, 0:n_t], stationary=ones_row_bf[0:1, 0:c_w], moving=c_bf[0:1, tile])
                m_sel = nl.ndarray((PMAX, _PSUM_MOVING), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(
                    dst=m_sel[0:c_w, 0:n_t], data=c_b[0:c_w, 0:n_t], op0=nl.equal, operand0=iota_pf[0:c_w, 0:1]
                )
                nisa.tensor_tensor(
                    dst=m_sel[0:c_w, 0:n_t], data1=m_ps[0:c_w, 0:n_t], data2=m_sel[0:c_w, 0:n_t], op=nl.multiply
                )
                sel = nl.ndarray((1, _PSUM_MOVING), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_matmul(dst=sel[0:1, 0:n_t], stationary=ones_col[0:c_w, 0:1], moving=m_sel[0:c_w, 0:n_t])
                nisa.tensor_copy(dst=got_flat[0:1, nl.ds(t0, n_t)], src=sel[0:1, 0:n_t])
                if got_tap is not None:
                    nisa.tensor_copy(dst=got_tap[0:1, nl.ds(t0, n_t)], src=sel[0:1, 0:n_t])

            # The destination rows are contiguous in HBM, so the whole group's ids leave in
            # ONE flat DMA rather than one per row.
            out_idx_flat = fused[_FUSE_OUT_IDX].reshape((fused[_FUSE_OUT_IDX].shape[0] * k,))
            nisa.dma_copy(
                dst=out_idx_flat.ap(pattern=[[1, n_flat]], offset=(sort_row_start + r0) * k),
                src=got_flat[0:1, :],
            )
            if got_tap is not None:
                tap_flat = fused[_FUSE_TAP_GOT].reshape((fused[_FUSE_TAP_GOT].shape[0] * k,))
                nisa.dma_copy(
                    dst=tap_flat.ap(pattern=[[1, n_flat]], offset=(sort_row_start + r0) * k),
                    src=got_tap[0:1, :],
                )

    return topk_values, topk_indices, asc_idx_hbm


@nki.jit
def chunked_topk(inp: nl.NkiTensor, config: ChunkedTopkConfig) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """Top-k over ``inp[BxS, vocab]`` for a vocab wider than nisa.topk's ceiling.

    Dimensions:
        BxS:   number of logical rows (tokens)
        vocab: full reduction length; may exceed 65,535
        k:     number of largest elements, 1 <= k <= vocab/P

    Args:
        inp: ``[BxS, vocab]`` bfloat16 tensor in HBM -- or, when
            ``config.interleaved_batch_groups > 1`` (jit-in-jit callers only), a LIST of
            batch-group slab tensors in the slabbed receive layout, one per producing
            exchange, so stage-1 tiles data-depend per slab.
        config: ChunkedTopkConfig. ``config.chunks == 1`` delegates to gpsimd_topk.

    Returns:
        ``(topk_values [BxS, k] bf16, topk_indices [BxS, k] uint32)``, both in HBM.
        Indices are GLOBAL vocab indices in ``[0, vocab)``. Descending along the last dim
        when ``config.sorted``. With ``config.debug_taps`` the taps of ``TAP_NAMES`` are
        APPENDED, so these two keep their positions.

    Pseudocode:
        W = vocab // P
        s1_v, s1_i = topk(inp.reshape(BxS*P, W), k)              # free reshape: chunking
        cand_v = s1_v.reshape(BxS, P*k)                          # free reshape: concat
        cand_i = s1_i.reshape(BxS, P*k) + c*W  per column block  # P-1 scalar adds
        r_v, r_pos = topk(cand_v, k)                             # local reduce, no collective
        out_i = nc_n_gather(cand_i, r_pos)                       # positions -> global ids
        # both topk calls are INLINED here, and the reduce's r_pos never leaves SBUF
    """
    _validate_chunked_topk(config)

    bxs = config.BxS
    vocab = config.vocab_size
    k = config.k
    p = config.chunks

    # LNC=1 is a correctness requirement, so assert it rather than document it -- and assert it
    # BEFORE the p == 1 short-circuit, so no path can reach the kernel body at LNC=2.
    #
    # Passing num_programs=1 to the inner calls pins nothing: gpsimd_topk re-reads the count from the
    # live grid, so at LNC=2 stage 1 would shard its rows across programs while the reduce below is
    # NOT sharded. Every program would then read the full candidate array, including the rows the
    # other program produced, with no core_barrier between them, and both would write the same output
    # rows. fast_dma_safe would also silently flip off as per_lnc_BxS halves. A caller that wired up
    # two programs has a bug either way, and failing loudly at trace time beats returning
    # half-written output rows.
    _, n_prgs, _ = get_verified_program_sharding_info("chunked_topk", (0, 1), 2)
    kernel_assert(n_prgs == 1, f"chunked_topk is LNC=1 only, got n_prgs={n_prgs}")

    # A vocab that already fits one call needs none of the chunking machinery -- but it runs
    # through THIS kernel's inlined top-k, not gpsimd_topk: chunked_topk is hardware-sort only,
    # and delegating a sorted request to gpsimd_topk would silently re-enter the software sort
    # this kernel deleted. The unfused _inline_topk publishes private buffers, so its results
    # are staged into the kernel's own shared outputs.
    if p == 1:
        # Every tap point is a link of the CHUNKING chain, none of which exists here, so asking
        # for taps on this path is a caller mistake rather than something to silently return
        # fewer outputs for.
        kernel_assert(not config.debug_taps, "debug_taps has nothing to observe when chunks == 1")
        kernel_assert(
            config.interleaved_peer_width == 0,
            "interleaved_peer_width is not implemented on the chunks == 1 delegate path",
        )
        p1_vals, p1_idx, _ = _inline_topk(
            inp,
            create_gpsimd_topk_config(
                inp_shape=(bxs, vocab), inp_dtype=config.inp_dtype, k=k, sorted=config.sorted, num_programs=1
            ),
            skip_nan_fold=config.skip_nan_fold,
        )
        p1_out_v = nl.ndarray((bxs, k), dtype=config.inp_dtype, buffer=nl.shared_hbm)
        p1_out_i = nl.ndarray((bxs, k), dtype=config.index_dtype, buffer=nl.shared_hbm)
        for t in range(div_ceil(bxs, PMAX)):
            r0 = t * PMAX
            rows = min(PMAX, bxs - r0)
            sv = nl.ndarray((PMAX, k), dtype=config.inp_dtype, buffer=nl.sbuf)
            si = nl.ndarray((PMAX, k), dtype=config.index_dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=sv[0:rows, :], src=p1_vals[r0 : r0 + rows, :])
            nisa.dma_copy(dst=si[0:rows, :], src=p1_idx[r0 : r0 + rows, :])
            nisa.dma_copy(dst=p1_out_v[nl.ds(r0, rows), :], src=sv[0:rows, :])
            nisa.dma_copy(dst=p1_out_i[nl.ds(r0, rows), :], src=si[0:rows, :])
        return p1_out_v, p1_out_i

    width = vocab // p
    s1_rows = bxs * p
    red_w = p * k
    # Stage 1's de-snake width, needed here only to shape the s1_late tap.
    s1_k_pad = div_ceil(k, PARTS_PER_GROUP) * PARTS_PER_GROUP

    # ---- Diagnostic taps (config.debug_taps) ------------------------------------------
    # Seven extra outputs that make the index chain observable ON HARDWARE, in a fully
    # overlapped run. See DEBUG TAPS in the module docstring for the method and for why each
    # tap point was chosen to perturb the schedule as little as possible.
    tap_s1_pos = None
    tap_s1_gid = None
    tap_s1_late = None
    tap_cand_i = None
    tap_r_pos = None
    tap_gathered = None
    tap_s1_val = None
    if config.debug_taps:
        tap_s1_pos = nl.ndarray((s1_rows, k), dtype=nl.float32, buffer=nl.shared_hbm)
        # float32 on both paths so the reader never has to know which one produced it: the
        # general path's uint32 ids cast exactly on the way out (every id is < 2^24).
        tap_s1_gid = nl.ndarray((s1_rows, k), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_s1_late = nl.ndarray((s1_rows, s1_k_pad), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_cand_i = nl.ndarray((bxs, red_w), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_r_pos = nl.ndarray((bxs, k), dtype=nl.float32, buffer=nl.shared_hbm)
        tap_gathered = nl.ndarray((bxs, k), dtype=nl.float32, buffer=nl.shared_hbm)
        # The VALUE side of stage 1, appended LAST so every earlier tap keeps its position.
        # It makes the intermediate candidate VALUES checkable on hardware -- per-chunk pairing
        # against the caller's own input row, and per-chunk top-k multiset equality -- closing
        # the one link the index chain does not cover.
        tap_s1_val = nl.ndarray((s1_rows, k), dtype=nl.float32, buffer=nl.shared_hbm)

    # ---- Which stage-1 shape can skip its phase 2 -------------------------------------
    # See _stage1_desnake_only: when k % 16 == 0 its de-snake buffers ARE the candidate arrays,
    # so stage 1's entire phase 2 (reload, compact, remap, store) is a copy and is deleted. The
    # second condition is that every stage-1 tile is a whole 128-partition tile, which is what
    # keeps the value buffer bfloat16 -- the reduce's nisa.topk requires bfloat16 input, and a
    # shard with a partial tile falls back to a float32 de-snake buffer. Both hold for every
    # auto-picked P at the production shape; anything else takes the general path below, which
    # is why both remain covered by the suite rather than one becoming dead code.
    red_rows = max(GROUPS_PER_TILE, div_ceil(bxs, GROUPS_PER_TILE) * GROUPS_PER_TILE)
    bypass_s1_phase2 = (
        (k % PARTS_PER_GROUP == 0) and (s1_rows % GROUPS_PER_TILE == 0) and (width % PARTS_PER_GROUP == 0)
    )

    # ---- Stage 1: one top-k per vocab chunk -------------------------------------------
    # The reshape is a pure view: a C-contiguous [BxS, vocab] tensor viewed as
    # [BxS*P, vocab/P] has row j*P + c == chunk c of token j. So the blocked contiguous load IS
    # the flat vocab sweep, and the instruction's 16-partition groups ARE the chunks.
    #
    # sorted=False is UNCONDITIONAL, not a knob. Stage 1's ordering is unobservable: its
    # values and indices never leave this kernel, and the reduce below re-sorts. So the
    # choice cannot affect the output, only the cost -- and sorting here is pure waste.
    # See the STAGE 1 IS NEVER SORTED section of the module docstring for the measurement.
    s1_cfg = create_gpsimd_topk_config(
        inp_shape=(s1_rows, width),
        inp_dtype=config.inp_dtype,
        k=k,
        sorted=False,
        num_programs=1,
    )
    kernel_assert(
        config.interleaved_peer_width == 0 or bypass_s1_phase2,
        "interleaved_peer_width is only implemented on the desnake-only stage-1 path",
    )
    _slabbed = isinstance(inp, (list, tuple))
    kernel_assert(
        not _slabbed or (config.interleaved_batch_groups > 1 and bypass_s1_phase2),
        "a slab-list input requires interleaved_batch_groups > 1 (and the desnake-only path)",
    )
    if bypass_s1_phase2:
        cand_v, cand_i = _stage1_desnake_only(
            inp if _slabbed else inp.reshape((s1_rows, width)),
            s1_cfg,
            chunks=p,
            chunk_width=width,
            pad_rows=red_rows * p,
            tap_pos=tap_s1_pos,
            skip_nan_fold=config.skip_nan_fold,
            peer_width=config.interleaved_peer_width,
            batch_groups=config.interleaved_batch_groups,
        )
        # The de-snake buffers, viewed as the reduce wants them. Both views are free, and they
        # agree slot for slot because both were written by the same de-snake pattern.
        red_in = cand_v.reshape((red_rows, red_w))
        cand_i_view = cand_i.reshape((red_rows, red_w))
        # On this path stage 1 publishes ONE index buffer, already global, so it is what both
        # the "what stage 1 said" tap and the second-read tap look at -- which makes the s1_late
        # comparison a three-read agreement test on the de-snake store rather than two.
        s1_published = cand_i
        s1_desnaked = cand_i
        # First s1_rows rows of the [pad_rows, k] buffer are the real candidates; the rest pad.
        s1_vals_pub = cand_v
    else:
        s1_vals, s1_idx, s1_desnaked = _inline_topk(
            inp.reshape((s1_rows, width)), s1_cfg, tap_pos=tap_s1_pos, skip_nan_fold=config.skip_nan_fold
        )
        s1_published = s1_idx
        s1_vals_pub = s1_vals
        # ---- Reduce input: another free reshape, row-padded to a whole tile ---------------
        # [BxS*P, k] viewed as [BxS, P*k] places chunk c at columns [c*k, (c+1)*k) of row j.
        cand_v = s1_vals.reshape((bxs, red_w))
        cand_i_view = s1_idx.reshape((bxs, red_w))
        # Pad the row count up to a multiple of GROUPS_PER_TILE so the reduce stays on its
        # fast_dma_safe path (bf16 rather than fp32 de-snake buffers). This is a PERFORMANCE
        # pad, not a correctness one: a partial tile is handled correctly. Pad rows carry the
        # most-negative FINITE bf16, so they can never win a slot within their own row, and
        # their row's results are simply never read.
        if red_rows > bxs:
            red_in = nl.ndarray((red_rows, red_w), dtype=config.inp_dtype, buffer=nl.private_hbm)
            n_pad = red_rows - bxs
            pad_tile = nl.ndarray((PMAX, red_w), dtype=config.inp_dtype, buffer=nl.sbuf)
            # Only the n_pad rows that are actually stored need the sentinel. Memsetting the
            # whole [PMAX, red_w] allocation wrote 128 partitions x 2,048 bf16 to fill 7 rows.
            nisa.memset(dst=pad_tile[0:n_pad, :], value=BFLOAT16_MIN)
            nisa.dma_copy(dst=red_in[nl.ds(bxs, n_pad), :], src=pad_tile[0:n_pad, :])
            for t in range(div_ceil(bxs, PMAX)):
                r0 = t * PMAX
                rows = min(PMAX, bxs - r0)
                stage = nl.ndarray((PMAX, red_w), dtype=config.inp_dtype, buffer=nl.sbuf)
                nisa.dma_copy(dst=stage[0:rows, :], src=cand_v[r0 : r0 + rows, :])
                nisa.dma_copy(dst=red_in[r0 : r0 + rows, :], src=stage[0:rows, :])
        else:
            red_in = cand_v

    # ---- Reduce: top-k over the P*k candidates, entirely local ------------------------
    # The reduce's tail is FUSED into it (see WHY THE REDUCE FUSES ITS TAIL in _inline_topk):
    # its positions never reach HBM, they are translated through the candidate ids and written
    # as global ids from the same SBUF tile the sort produced them in, and its values go
    # straight to out_vals in one cast-on-DMA. Unfused this cost two stores, three loads and
    # two casts, all of them pure interface between two things that are now one body.
    out_vals = nl.ndarray((bxs, k), dtype=config.inp_dtype, buffer=nl.shared_hbm)
    out_idx = nl.ndarray((bxs, k), dtype=config.index_dtype, buffer=nl.shared_hbm)
    _inline_topk(
        red_in,
        create_gpsimd_topk_config(
            inp_shape=(red_rows, red_w),
            inp_dtype=config.inp_dtype,
            k=k,
            sorted=config.sorted,
            num_programs=1,
        ),
        # The reduce input is stage 1's own (already NaN-folded) candidates plus finite
        # sentinel pad rows: the fold can never fire here. See the fold site for the argument.
        skip_nan_fold=True,
        fused=(
            cand_i_view,
            p,
            width,
            out_vals,
            out_idx,
            bxs,
            tap_cand_i,
            tap_r_pos,
            tap_gathered,
            bypass_s1_phase2,
        ),
    )

    if config.debug_taps:
        # A SECOND, independent read of stage 1's de-snake buffer, and a copy of stage 1's
        # remapped ids. Traced last so it tends to be scheduled late, but the diagnostic does
        # NOT depend on that: the signature of a store/reload race is TWO READS OF THE SAME
        # ADDRESS DISAGREEING, which is only possible while a write is in flight. So if s1_pos
        # (what phase 2 worked from) is wrong while s1_late is right, the de-snake store landed
        # after phase 2 read it; if BOTH are wrong the top-k itself reported those positions and
        # the fault is upstream of any DMA ordering.
        for t in range(div_ceil(s1_rows, PMAX)):
            r0 = t * PMAX
            rows = min(PMAX, s1_rows - r0)
            lt = nl.ndarray((PMAX, s1_k_pad), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=lt[0:rows, :], src=s1_desnaked[r0 : r0 + rows, :])
            nisa.dma_copy(dst=tap_s1_late[r0 : r0 + rows, :], src=lt[0:rows, :])
            gt = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=gt[0:rows, :], src=s1_published[r0 : r0 + rows, :])
            nisa.dma_copy(dst=tap_s1_gid[r0 : r0 + rows, :], src=gt[0:rows, :])
            vt = nl.ndarray((PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=vt[0:rows, :], src=s1_vals_pub[r0 : r0 + rows, :])
            nisa.dma_copy(dst=tap_s1_val[r0 : r0 + rows, :], src=vt[0:rows, :])
        return out_vals, out_idx, tap_s1_pos, tap_s1_gid, tap_s1_late, tap_cand_i, tap_r_pos, tap_gathered, tap_s1_val

    return out_vals, out_idx
