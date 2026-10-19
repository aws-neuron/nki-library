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

"""GPT-OSS decode sampling tail: RMSNorm -> lm_head -> top-k -> sampler, ONE collective.

    final RMSNorm -> lm_head GEMM -> all_to_all(all ranks) -> chunked_topk -> sampler

WHY ONE COLLECTIVE
------------------
The vocab is column-parallel across ``VTP`` ranks, so no rank can take a global top-k
alone. The established answer is to top-k each rank's shard and reduce across ranks with a
second collective, which costs two collectives, a group-aware index correction, a packed
value/index collective, and a batch-row permutation the caller has to undo.

This kernel instead spends its ONE collective bringing the FULL vocab onto each rank -- an
all-to-all that trades vocab-parallelism for batch-parallelism -- and then takes the whole
top-k locally with ``chunked_topk``, which exists precisely because a 201,088-wide vocab
exceeds ``nisa.topk``'s per-call ceiling. Everything the two-collective form needs in
between simply has no counterpart here:

* **No index correction.** After the all-to-all over the replica group ``[0 .. VTP-1]`` in
  RANK ORDER, peer ``p``'s shard lands at columns ``[p*V_s, (p+1)*V_s)``, and peer index IS
  vocab-shard index. So a column index in the assembled row already IS the global vocab id,
  and ``chunked_topk`` returns global ids with no fix-up of any kind.
* **No packed collective.** Nothing has to travel with its index, so there is no
  ``[rows, 2k]``-vs-``[2*rows, k]`` packing orientation to get wrong and no fp32 inflation
  of bf16 values.
* **Row ownership is a closed form, not a lookup.** With a single hop it is the identity --
  rank ``r`` owns ``[r*R, (r+1)*R)``, ``R = gbs/VTP`` -- so the caller's final ``all_gather``
  over the same rank-ordered group concatenates straight into batch order. Multi-hop (which
  trn3 forces, see below) makes it a mixed-radix sum instead, still closed-form and provided
  by this module's ``owned_row_block``, but the caller must then de-permute. The
  batch-grouped exchange (``a2a_batch_groups > 1``) adds one more mixed-radix digit and
  makes ownership STRIDED -- ``owned_rows`` is the map in force in every case.

WHICH COLLECTIVE ACTUALLY RUNS (why ``hop_meshes`` exists)
---------------------------------------------------------
"One collective" is the DATAFLOW claim -- one vocab-parallel-to-batch-parallel exchange rather
than a top-k-then-reduce pair. It is not a claim that the exchange is one hardware mesh
operation, because on trn3 it cannot be. Measured on a reserved 16-device trn3 host (16
devices x 8 cores = the 128 LNC=1 ranks of production):

    a2a group                        | device layout        | result
    ---------------------------------+----------------------+---------------------------------
    8 contiguous   {0..7}            | 1 device, 8 cores    | PASS
    4 stride 8     {0,8,16,24}       | 4 devices, 1 core ea | PASS
    16 stride 8    {0,8,...,120}     | 16 devices, 1 core ea| PASS
    16 contiguous  {0..15}           | 2 devices, 8 cores ea| FAIL: op_type 7 unsupported
    128 all ranks                    | 16 devices, 8 cores  | FAIL: op_type 7 unsupported
    4 stride 4     {0,4,8,12}        | 2 devices, 2 cores ea| FAIL: encd_mesh_memcopy

The constraint is the group's DEVICE LAYOUT, not its size. A group must be entirely within one
device, or hold exactly one core per device; a MIXED group -- several devices with several cores
each -- has no mesh subtype that implements all-to-all and is rejected with
``__select_mesh_subtype: not a supported operation for mesh (op_type 7)``. Payload size is not
the trigger: the 16-contiguous rejection carries ``total_sz 0x800`` and the 128-rank one
``0x62300``.

Reading it as a size limit is the easy mistake, and it costs a hop: it makes 8 look like the
ceiling, which forces three hops, when in fact ``(8, 16)`` is legal and sufficient -- hop 0
device-local, hop 1 one core per device across all 16. That is exactly the decomposition the
existing sharded tail uses.

``all_gather`` at 128 also runs, but it is not the right tool here: it hands every rank EVERY
batch row, which at gbs=1024 is 412 MB per rank against 6.4 MB for the two-hop all-to-all, and
extracting just this rank's rows needs a rank-dependent slice one shared NEFF cannot express.

Note these limits differ by host. On the shared fleet's trn3_a0 a single mesh passes at 16 and
32 but aborts at >= 64 in ``encd_mesh_memcopy``, while MORE THAN ONE mesh per NEFF fails to
load at any rank count -- the exact opposite of what reserved trn3 allows. So a decomposition
must be validated on the host it will run on; ``collective_probe.py`` holds both matrices.

STATUS
------
Correct and passing every gate at VTP = 8, 16 and 32 (values, indices, tokens, logits, with
per-row-varying sampling params and degenerate/extreme inputs), single-hop and multi-hop.

CALLER CONTRACTS
----------------
Three, all of them load-bearing:

1. ``lm_head_weight_t`` is ``[H, V_s]`` **bf16**, i.e. ALREADY TRANSPOSED, and C-contiguous.
   bf16 is the GPT-OSS checkpoint's own lm_head precision, so passing fp32 would not be more
   accurate than the model -- it would just double the dominant HBM read. The kernel DMAs it
   straight to bf16 SBUF and matmuls in bf16 with an fp32 PSUM accumulator.
   ``nc_matmul`` contracts the partition dim, so the weight must present H on partitions.
   Reading an untransposed ``[V_s, H]`` shard forces ``V_s`` separate short runs per
   H-chunk instead of one contiguous run; a transposed VIEW (no copy) keeps the original
   strides and the DMA descriptor is built from strides, so it provides none of the benefit.
   Transposing is a one-time host cost on a tensor that never changes.
2. ``replica_groups`` is a single group listing ALL ``VTP`` ranks IN RANK ORDER. The
   no-index-correction property above is a direct consequence of that ordering; a permuted
   group silently returns wrong vocab ids.
3. ``sampling_params`` is ``[R, 3]`` -- THIS RANK'S rows only, not the whole batch. The
   caller slices by ``owned_rows`` (``[r*R, (r+1)*R)`` for a single ungrouped hop; strided
   with ``a2a_batch_groups > 1``; a mixed-radix block multi-hop). Passing the full ``[gbs, 3]``
   would make every rank apply rows ``0..R-1``'s temperature and top-p to its own rows, a
   wrong-token bug invisible to any test whose sampling params happen to be uniform.

THE mxfp4 VARIANCE CONVENTION (do not "simplify")
-------------------------------------------------
GPT-OSS pads hidden 2880 -> 3072, and under the mxfp4 checkpoint the hidden dim is SHUFFLED,
so the padding lanes are INTERLEAVED rather than a contiguous tail. Two consequences:
variance is ``sum(x^2)/H_u`` (equal to ``mean(x^2 over all H) * H/H_u`` only because the
padding is exactly zero), and zeroing the padding (applied to GAMMA at load time -- see
``_load_gamma_broadcast`` for why that is equivalent to re-zeroing the normed output) goes
through a ``[4, H/4]`` view killing columns ``>= H_u/4`` of each group. A contiguous
``[..., H_u:] = 0`` zeroes the WRONG lanes; that is the bf16 checkpoint's convention and it
changes the answer here.

LNC=1 BY DESIGN
---------------
``VTP = 128`` LNC=1 ranks is what makes ``V_s = 201088/128 = 1571``. ``n_prgs == 1`` is
asserted rather than assumed: the stages after the collective are not program-sharded and
carry no ``core_barrier``, so at LNC=2 both programs would race on the same ``shared_hbm``
collective buffers. Note that passing ``num_programs=1`` to an inner kernel does NOT pin
this -- gpsimd_topk re-reads the count from the live grid -- which is why the assert is here.
"""

from typing import Optional, Tuple

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil, get_verified_program_sharding_info
from ..topk.chunked_topk import chunked_topk, create_chunked_topk_config, interleaved_fold_supported

_PMAX = nl.tile_size.pmax  # 128


def compose_hops(vtp: int, hops: Tuple[int, ...]):
    """Replica-group meshes for a sequence of all-to-all hops whose sizes multiply to ``vtp``.

    Hop ``h`` groups ranks that agree on every LATER hop's coordinate, so with ``hops=(8, 16)``
    and ``r = q*8 + s``: hop 0 groups the 8 CONTIGUOUS ranks sharing ``q``, and hop 1 groups
    the 16 ranks sharing ``s``, strided by 8. Listing each group in ascending rank order is
    what keeps a column index equal to a global vocab id.

    Returns a tuple of meshes, one per hop, each a tuple of rank-tuples.
    """
    prod = 1
    for h in hops:
        prod *= h
    if prod != vtp:
        raise ValueError(f"hop sizes {hops} multiply to {prod}, not vtp={vtp}")
    stride = 1
    meshes = []
    for h in hops:
        groups = []
        for base in range(vtp):
            # One group per distinct (everything except this hop's coordinate).
            if (base // stride) % h != 0:
                continue
            groups.append(tuple(base + i * stride for i in range(h)))
        meshes.append(tuple(sorted(groups)))
        stride *= h
    return tuple(meshes)


def owned_row_block(rank: int, vtp: int, gbs: int, hops=None):
    """Batch rows that ``rank`` owns after the all-to-all. Returns ``(row_start, n_rows)``.

    THE caller-facing half of the kernel's row-ownership contract, living next to
    ``compose_hops`` because the two must agree: a production caller de-permutes this
    kernel's per-rank outputs with exactly this map. With a single hop it is the IDENTITY
    -- rank ``r`` owns ``[r*gbs/vtp, (r+1)*gbs/vtp)`` -- which is the point of routing the
    collective over a rank-ordered group. A multi-hop decomposition keeps, per hop, only
    the sub-block selected by this rank's coordinate within that hop's group, so the
    offsets compose as a mixed-radix sum. Kept as one named function so the tests can
    assert the map tiles the batch exactly once and a change to hop structure has exactly
    one place to break.

    ``hops=None`` means a single hop over all ``vtp`` ranks, which the general formula
    reproduces exactly (``start = rank * gbs/vtp``); it is not a separate code path.
    """
    if gbs % vtp:
        raise ValueError(f"gbs ({gbs}) must be divisible by VTP ({vtp})")
    rows = gbs // vtp
    hop_sizes = (vtp,) if hops is None else tuple(hops)
    prod = 1
    for h in hop_sizes:
        prod *= h
    if prod != vtp:
        raise ValueError(f"hop sizes {hop_sizes} multiply to {prod}, not vtp={vtp}")
    span, stride, start = gbs, 1, 0
    for h in hop_sizes:
        span //= h
        start += ((rank // stride) % h) * span
        stride *= h
    return start, rows


def owned_rows(rank: int, vtp: int, gbs: int, hops=None, batch_groups: int = 1):
    """The GLOBAL batch rows ``rank`` owns, IN RECEIVE ORDER, as a list.

    Generalizes ``owned_row_block`` to the batch-group-chunked all-to-all
    (``a2a_batch_groups > 1``), where ownership is no longer one contiguous block: chunk
    ``g``'s collective carries mirror rows ``[g*S, (g+1)*S)`` (``S = gbs/G``, one or more
    whole lm_head tiles -- which is what lets the exchange start under the lm_head), and a
    full-mesh all-to-all of that slab hands rank ``p`` its ``S/vtp`` rows of EVERY group:

        row(g, j) = g*S + p*(S/vtp) + j,   g in [0, G),  j in [0, S/vtp)

    i.e. the same mixed-radix ownership map with the group index as one more (outermost)
    digit. ``batch_groups = 1`` reproduces ``owned_row_block`` exactly. ``batch_groups > 1``
    is defined for the single-hop exchange only, matching the kernel.
    """
    # Derived FROM owned_row_block rather than re-implemented, so its validation (hop sizes
    # multiply to vtp, gbs divisible) covers the grouped form too, and "G = 1 reproduces
    # owned_row_block" holds by construction. With G groups each group is gbs/G rows and
    # this rank's contiguous block splits into G equal sub-blocks, one per group, at the
    # same relative offset (start/G) inside each group.
    start, rows = owned_row_block(rank, vtp, gbs, hops)
    if batch_groups == 1:
        return list(range(start, start + rows))
    if hops is not None and len(tuple(hops)) != 1:
        raise ValueError("batch_groups > 1 is defined for the single-hop exchange only")
    if rows % batch_groups or start % batch_groups:
        raise ValueError(f"rows_per_rank ({rows}) must be divisible by batch_groups ({batch_groups})")
    group_rows = gbs // batch_groups
    return [
        g * group_rows + start // batch_groups + j for g in range(batch_groups) for j in range(rows // batch_groups)
    ]


# The mxfp4 hidden shuffle is defined over 4 interleaved groups, so the padding boundary is
# expressed in a [4, H/4] view. See THE mxfp4 VARIANCE CONVENTION above.
_MXFP4_SHUFFLE_GROUPS = 4

# Sampler constants. The below-threshold sentinel is FINITE by design: -inf would make a
# fully-masked row's manual softmax produce NaN.
_TOPK_MASK_SENTINEL = -3000.0
# temperature < eps => treat the row as greedy.
_SAMPLING_EPS = 1e-5
# The inverse-transform draw when ``sampling_params`` carries no 4th column: COMPILED IN, so
# the 3-column NEFF stays deterministic by construction. A 4-column ``sampling_params``
# supplies the draw per row at RUNTIME instead (see _sampler_stage), which is what lets a
# test sweep u as data without rebuilding the NEFF.
_DETERMINISTIC_DRAW = 0.5


def _const_tile(shape, value: float):
    """Materialize a constant SBUF tile.

    ``nisa.tensor_copy_predicated`` validates its ``src`` as a partitioned tensor and
    rejects a Python scalar despite the type hint.
    """
    tile = nl.ndarray(shape, dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=tile, value=value)
    return tile


def _make_scan_triangle():
    """``[128, 128]`` fp32 upper-triangular ones, ``T[p, n] = 1 iff p <= n``.

    The stationary-side constant of the Tensor Engine prefix sum in ``_inclusive_scan_pe``:
    contracting data against it turns "sum over partitions" into "sum over partitions up to
    n", which IS the inclusive prefix. Built from two iotas and one compare, with no data
    dependencies, so the scheduler hoists it off every critical path.
    """
    part = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(part, pattern=[[1, 1]], offset=0, channel_multiplier=1)
    cols = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(cols, pattern=[[1, _PMAX]], offset=0, channel_multiplier=0)
    tri = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=tri, data=cols, op0=nl.greater_equal, operand0=part)
    return tri


def _inclusive_scan_pe(dst, src, n_rows: int, width: int, tri) -> None:
    """Inclusive prefix sum along the free dim on the TENSOR Engine: ``[rows, w] -> [rows, w]``.

    Per 128-wide chunk: transpose the chunk so its slice of the row sits on partitions, then
    contract against the upper-triangular ones -- ``out[r, n] = sum_{p <= n} chunk_t[p, r]`` --
    and read the PSUM back with the previous chunks' running total fused into the copy as a
    per-partition scalar add (the last column of the previous chunk's output, which already
    carries every earlier chunk).

    This replaces a Hillis-Steele scan whose ceil(log2(w)) shifted adds were the sampler's
    single largest block of Vector work, and whose every step was serial ON VECTOR. Here
    Vector pays two 128-wide copies per chunk and the contraction runs on the PE, which is
    otherwise idle through the whole sampler phase. Each output element is ONE fp32 PE
    contraction of its prefix rather than a tree of partial adds -- a different (not worse)
    fp32 summation order than either the tree or a sequential cumsum, the same class of
    last-ulp wiggle the gates' tie-guards already absorb.
    """
    for c in range(div_ceil(width, _PMAX)):
        c0 = c * _PMAX
        c_len = min(_PMAX, width - c0)
        chunk_ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(
            dst=chunk_ps[0:c_len, 0:n_rows], data=src[0:n_rows, nl.ds(c0, c_len)], engine=nisa.tensor_engine
        )
        chunk_t = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=chunk_t[0:c_len, 0:n_rows], src=chunk_ps[0:c_len, 0:n_rows])
        pre_ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(
            dst=pre_ps[0:n_rows, 0:c_len], stationary=chunk_t[0:c_len, 0:n_rows], moving=tri[0:c_len, 0:c_len]
        )
        if c == 0:
            nisa.tensor_copy(dst=dst[0:n_rows, nl.ds(c0, c_len)], src=pre_ps[0:n_rows, 0:c_len])
        else:
            # The carry rides the PSUM read: previous chunk's last output column is the total
            # of everything before this chunk, and it lives on the same partition as its row.
            nisa.tensor_scalar(
                dst=dst[0:n_rows, nl.ds(c0, c_len)],
                data=pre_ps[0:n_rows, 0:c_len],
                op0=nl.add,
                operand0=dst[0:n_rows, nl.ds(c0 - 1, 1)],
            )


def _skew_fence(hop_meshes, hidden):
    """All-reduce a ZERO on every hop mesh, and return it for the norm to consume.

    PROFILING INSTRUMENT, not computation. Without it the FIRST collective absorbs all the
    rank-arrival skew and every LATER collective absorbs whatever skew re-accumulates while the
    ranks run compute -- so a per-hop duration reads as transfer cost when it is mostly dispatch
    jitter. Fencing each mesh separately moves that charge onto the fence.

    It cannot change the answer: the reduced value is 0.0 and the caller adds it into the
    hidden states, and ``x + 0.0`` is bit-identical for every finite fp32 x. The result MUST
    stay consumed by the norm -- that data dependency is the only thing stopping the scheduler
    from sinking the fence past the real work and making it useless.
    """
    acc = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
    zero_sb = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(dst=zero_sb, value=0.0)
    nisa.dma_copy(dst=acc, src=zero_sb)
    for hop in range(len(hop_meshes)):
        src = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        nisa.dma_copy(dst=src, src=zero_sb)
        dst = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.shared_hbm)
        ncc.all_reduce(srcs=[src], dsts=[dst], replica_group=ncc.ReplicaGroup(hop_meshes[hop]), op=nl.add)
        got = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=got, src=dst)
        prev = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=prev, src=acc)
        nisa.tensor_tensor(dst=prev, data1=prev, data2=got, op=nl.add)
        nisa.dma_copy(dst=acc, src=prev)
    return acc


def _load_gamma_broadcast(gamma_hbm, hidden, hidden_actual, zero_interleaved_pad):
    """gamma ``[1, H]`` -> ``[_PMAX, H]`` SBUF broadcast, with the padding lanes zeroed.

    gamma is ``[1, H]`` and every row needs it, but a PARTITION-dim broadcast of an on-chip
    tensor is illegal -- do it at the DMA boundary with an HBM-side ``.broadcast`` view. Loaded
    ONCE per kernel, not once per 128-row tile: gamma never changes between tiles, and the
    broadcast moves ``_PMAX * H`` fp32 (1.5 MB at the production shape) per load.

    ZEROING GAMMA'S PADDING REPLACES RE-ZEROING THE NORMED OUTPUT. The input's padding lanes
    are exactly zero by contract -- the variance formula divides by ``H_u`` on exactly that
    assumption (see THE mxfp4 VARIANCE CONVENTION) -- so the normed padding is
    ``(0 * rms) * gamma_pad``: zero for any finite ``gamma_pad``, and NaN for a non-finite one
    only when the variance (and therefore every normed lane) is already poisoned anyway. Doing
    it on gamma moves the memsets off the per-tile critical path, where the old form re-ran
    them per tile AFTER the full-width gamma multiply, and lets the whole normed row feed the
    GEMM the moment it is scaled. The mxfp4 padding is INTERLEAVED, so kill columns
    ``>= H_u/4`` of each of 4 groups.
    """
    gamma_sb = nl.ndarray((_PMAX, hidden), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_sb, src=gamma_hbm[0:1, :].broadcast(0, _PMAX))
    if hidden_actual < hidden:
        if zero_interleaved_pad:
            gw = hidden // _MXFP4_SHUFFLE_GROUPS
            keep = hidden_actual // _MXFP4_SHUFFLE_GROUPS
            for g in range(_MXFP4_SHUFFLE_GROUPS):
                if gw - keep > 0:
                    nisa.memset(dst=gamma_sb[:, nl.ds(g * gw + keep, gw - keep)], value=0)
        else:
            nisa.memset(dst=gamma_sb[:, nl.ds(hidden_actual, hidden - hidden_actual)], value=0)
    return gamma_sb


# DMA granule for the row tile's hidden dim in _norm_lm_head_stage. One monolithic [rows, H]
# load serialises the whole norm behind the full transfer (12 us of engine idle at the
# production shape); block loads let the sum of squares run behind each block while the next
# is still in flight, so "rms ready" trails the LAST block's arrival by one 0.7 us activation
# instead of trailing the whole row by three full-width ops.
_NORM_DMA_BLOCK = 6 * _PMAX


def _load_weight_resident(wt_hbm, hidden, vocab_shard):
    """Load the whole ``[H, V_s]`` bf16 lm_head shard into SBUF ONCE, as per-chunk tiles.

    The GEMM was re-reading the shard from HBM once per pair of 128-row tiles (measured:
    2x/4x the 9.65 MB shard at gbs 512/1024). In isolation that costs no wall clock -- the
    head is PE-bound and the stream keeps pace -- but in INTEGRATION the repeated stream is
    contended HBM bandwidth that cannot be prefetched behind preceding work, whereas one
    resident copy (24 x 393 KB = 9.4 MB of SBUF, ~40%, live only through the head) loads
    once, entirely prefetchable before the first matmul needs chunk 0. Static descriptors
    for the same reason as the input blocks above.
    """
    wt_res = []
    for c in range(div_ceil(hidden, _PMAX)):
        h0 = c * _PMAX
        h_len = min(_PMAX, hidden - h0)
        wt_c = nl.ndarray((_PMAX, vocab_shard), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(dst=wt_c[0:h_len, :], src=wt_hbm[nl.ds(h0, h_len), :], dge_mode=nisa.dge_mode.none)
        wt_res.append(wt_c)
    return wt_res


# PSUM is 8 banks of 512 fp32 per partition on every NeuronCore version. Per the
# ``nki.isa.nc_matmul`` doc, NeuronCore-v2/v3 cap a matmul's moving free dim at ONE bank
# (``gemm_moving_fmax`` = 512) while v4+ (trn3) lets an fp32 dst span the entire PSUM.
_PSUM_NUM_BANKS = 8
_GEN4_MATMUL_MOVING_FMAX = _PSUM_NUM_BANKS * nl.tile_size.psum_bank_fmax  # 4096 fp32


def _lm_head_moving_fmax() -> int:
    """Widest matmul moving free dim this target takes for an fp32 PSUM dst.

    Mirrors the compiler's ``validate_matmul_dst_shape``: 512 before gen4, 4096 on gen4. The
    production shard widths (``V_s`` = 1571 and 2048) exceed 512, so trn2 tiles the lm_head
    GEMM over vocab while trn3 keeps ONE matmul per H chunk -- the instruction count the
    PE-bound head was tuned at.
    """
    if nisa.get_nc_version() >= nisa.nc_version.gen4:
        return _GEN4_MATMUL_MOVING_FMAX
    return nl.tile_size.gemm_moving_fmax


def _norm_lm_head_stage(
    x_hbm, gamma_sb, wt_res, eps, n_rows, row_start, hidden, hidden_actual, vocab_shard, fence, deferred_rms
):
    """RMSNorm + lm_head GEMM for one 128-row tile, pipelined at block/chunk granularity.

    Returns ``(acc_ps, rms)``. With ``deferred_rms`` (the default), ``acc_ps`` is the
    ``[_PMAX, V_s]`` fp32 PSUM accumulator holding this tile's UNNORMALIZED logits
    ``(x * gamma) @ W`` and the CALLER multiplies the ``[_PMAX, 1]`` fp32 per-row ``rms`` in
    on its PSUM-read copies. DEFERRED RMS: since ``rms`` is a per-row scalar and the GEMM is
    linear, ``((x*rms)*gamma) @ W == rms * ((x*gamma) @ W)`` exactly -- and deferring it
    removes the one true data dependency that stalled the PE: every chunk used to wait on
    ``rms``, which needs ``sum(x^2)`` over the WHOLE row (~9 us of PE idle at the head of
    every tile; measured -3.8 us on the fenced head window). With ``deferred_rms=False`` the
    legacy in-chain form runs instead -- ``(x * rms) * gamma`` per chunk, ``acc_ps`` already
    normalized, the returned ``rms`` unused -- as a safety fallback with the exact
    pre-deferred numerics realization.

    THE POINT IS THE SCHEDULE'S SHAPE, NOT THE MATH. What the original rewrite fixed was three
    strictly serial phases -- a 12 us monolithic input DMA that every engine waited out, then
    ~15 us of full-width [rows, H] Vector/Scalar ops each gated on the previous, then 24
    transposes and 24 matmuls -- 54 us for a tile whose PE work is 16 us. This form breaks
    every one of those serialisations:

    * The input row loads in ``_NORM_DMA_BLOCK`` column blocks, and each block's
      ``sum(x^2)`` contribution runs on the Scalar Engine (``nisa.activation`` with
      ``op=nl.square`` accumulating into the engine's ``reduce_regs`` across blocks -- the
      idiom of ``mlp_tkg_rmsnorm``) while later blocks are still in flight. NOTE the
      ``reduce_regs`` chain is engine state: it tolerates interleaved ``reduce_cmd.idle``
      activations (every other activation in this kernel), but a hoisted reducing activation
      from other code WOULD corrupt it -- the WAW on ``sq_scratch`` is what pins the chain's
      own order.
    * ``rsqrt(ssq/H_u + eps)`` is ONE activation: the Scalar Engine applies
      ``op(data*scale + bias)`` natively, so the two full-width-free tensor_scalars the old
      form spent on ``/H_u`` and ``+eps`` cost nothing here.
    * Gamma applies in ONE ``scalar_tensor_tensor`` per 128-column GEMM chunk --
      ``(x + fence0) * gamma`` -- feeding that chunk's transpose+matmul immediately. RMS is
      NOT in this chain (see DEFERRED RMS above): the rsqrt runs concurrently on the Scalar
      Engine and lands on the caller's PSUM reads, so no matmul waits for the full row.

    The bf16 boundary moves by exactly one algebraic step and not in magnitude: the bf16
    rounding lands on ``x * gamma`` instead of ``x * rms * gamma``. bf16's relative error is
    scale-invariant (same exponent range as fp32), so the per-element error bound is
    IDENTICAL -- one bf16 rounding either way -- and the trailing fp32 multiply by rms adds
    2^-24, invisible at bf16 output granularity. Verified by simulation at the production
    shape against fp64 ground truth: both forms sit at the same distance from exact (mean
    1.66e-3 relative), greedy argmax flips ZERO rows, and the sampling distributions differ
    by <= 0.7% total variation. What DOES change is the last-ulp REALIZATION -- logits move
    within the bf16 noise floor relative to the previous build, so fixed-seed sampled tokens
    can land on an equally-valid neighbour at cdf crossings. Same class of wiggle as any
    resummation change, arbitrated the same way: tie-guarded gates, regenerated hashes.

    On the GEMM half nothing changed: ``nc_matmul`` contracts the PARTITION dim of both
    operands, so H must be on partitions for both. The activations get an on-chip per-chunk
    ``nc_transpose``; the WEIGHT is required to arrive already transposed as ``[H, V_s]`` --
    see CALLER CONTRACTS -- which makes the per-chunk read ``h_len`` ADJACENT rows of ``V_s``
    contiguous elements, i.e. ONE contiguous run rather than ``V_s`` short strided ones.
    BF16 THROUGHOUT, INCLUDING THE WEIGHT DMA: the GPT-OSS lm_head ships bf16, so fp32 here
    buys no accuracy over the checkpoint -- it just doubles the dominant HBM read (9.65 MB per
    rank per pass instead of 19.30 MB, re-read once per row tile).
    """
    # ---- load + sum of squares, block by block ----
    # The block DMAs are issued FIRST, before the fence broadcast: DMA descriptors on a queue
    # execute in issue order, so a fence-dependent load issued ahead of them would stall the
    # whole input behind the skew-absorbing all-reduce (measured: +14 us of engine idle in a
    # fenced profile). The fence rides TWO operand slots, both computing +0.0: the squares'
    # BIAS (gating ssq -> rms) and each GEMM chunk's scalar_tensor_tensor ADD (gating every
    # matmul and, through the mirror store, the collective). The second one exists because
    # DEFERRED RMS removed the rms edge that used to carry the fence into the matmuls -- an
    # unfenced GEMM would front-run the skew window and the fence-corrected head would
    # under-count it. Both adds are exact (+0.0) and cost no extra instructions.
    # A 16-bit input loads half the bytes of this kernel's largest input DMA, then widens to
    # the fp32 xt through an EXPLICIT per-block tensor_copy. The copy is what makes 16-bit
    # input bitwise-equivalent to a pre-widened fp32 copy of the same data: a dtype-conversion
    # COPY widens exactly (fp16/bf16 are subsets of fp32), whereas feeding 16-bit data
    # straight into the square/gamma chains does NOT -- the engines compute at the INPUT's
    # precision, measured as a real output diff on hardware (2026-08-31). Everything below
    # this loop reads only the fp32 xt, so the numerics are dtype-independent.
    widen = x_hbm.dtype != nl.float32
    x_raw = nl.ndarray((_PMAX, hidden), dtype=x_hbm.dtype, buffer=nl.sbuf) if widen else None
    xt = nl.ndarray((_PMAX, hidden), dtype=nl.float32, buffer=nl.sbuf)
    ssq = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    sq_scratch = nl.ndarray((_PMAX, _NORM_DMA_BLOCK), dtype=nl.float32, buffer=nl.sbuf)
    n_blk = div_ceil(hidden, _NORM_DMA_BLOCK)
    for b in range(n_blk):
        b0 = b * _NORM_DMA_BLOCK
        b_len = min(_NORM_DMA_BLOCK, hidden - b0)
        # Static descriptors (dge_mode.none): generated at compile time, so the load neither
        # waits on a DGE engine nor -- in fenced profiling NEFFs -- queues behind the skew
        # fence's trigger, which lives on the GpSimd DGE queue.
        nisa.dma_copy(
            dst=(x_raw if widen else xt)[0:n_rows, nl.ds(b0, b_len)],
            src=x_hbm[row_start : row_start + n_rows, nl.ds(b0, b_len)],
            dge_mode=nisa.dge_mode.none,
        )
    if widen:
        for b in range(n_blk):
            b0 = b * _NORM_DMA_BLOCK
            b_len = min(_NORM_DMA_BLOCK, hidden - b0)
            nisa.tensor_copy(dst=xt[0:n_rows, nl.ds(b0, b_len)], src=x_raw[0:n_rows, nl.ds(b0, b_len)])
    f_sb = None
    if fence is not None:
        # Broadcast on the partition dim at the DMA boundary, as with gamma.
        f_sb = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=f_sb, src=fence[0:1, 0:1].broadcast(0, _PMAX))
    for b in range(n_blk):
        b0 = b * _NORM_DMA_BLOCK
        b_len = min(_NORM_DMA_BLOCK, hidden - b0)
        nisa.activation(
            dst=sq_scratch[0:n_rows, 0:b_len],
            op=nl.square,
            data=xt[0:n_rows, nl.ds(b0, b_len)],
            bias=f_sb[0:n_rows, :] if f_sb is not None else None,
            reduce_op=nl.add,
            reduce_res=ssq[0:n_rows, :] if b == n_blk - 1 else None,
            reduce_cmd=nisa.reduce_cmd.reset_reduce if b == 0 else nisa.reduce_cmd.reduce,
        )
    # Divide by H_u, NOT H: the mxfp4 convention (padding contributes exactly zero to ssq).
    # DEFERRED: nothing below consumes rms -- it runs concurrently with the GEMM on the
    # Scalar Engine and the CALLER multiplies it onto the PSUM-read copies.
    rms = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=rms[0:n_rows, :], op=nl.rsqrt, data=ssq[0:n_rows, :], scale=1.0 / float(hidden_actual), bias=float(eps)
    )

    # ---- per-128-column chunk: scale -> transpose -> bf16 -> matmul ----
    # Deferred: (x + fence0) * gamma -- chunk c depends only on its OWN input block (plus the
    # fence), so the first transpose can start one block-DMA after the fence instead of after
    # the whole row's ssq. Legacy (deferred_rms=False): (x * rms) * gamma, the rms edge
    # carrying the fence into the matmuls exactly as before.
    acc_ps = nl.ndarray((_PMAX, vocab_shard), dtype=nl.float32, buffer=nl.psum)
    v_tile = min(vocab_shard, _lm_head_moving_fmax())
    for c in range(div_ceil(hidden, _PMAX)):
        h0 = c * _PMAX
        h_len = min(_PMAX, hidden - h0)
        scaled_c = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.sbuf)
        if deferred_rms:
            nisa.scalar_tensor_tensor(
                dst=scaled_c[0:n_rows, 0:h_len],
                data=xt[0:n_rows, nl.ds(h0, h_len)],
                op0=nl.add,
                operand0=f_sb[0:n_rows, :] if f_sb is not None else 0.0,
                op1=nl.multiply,
                operand1=gamma_sb[0:n_rows, nl.ds(h0, h_len)],
            )
        else:
            nisa.scalar_tensor_tensor(
                dst=scaled_c[0:n_rows, 0:h_len],
                data=xt[0:n_rows, nl.ds(h0, h_len)],
                op0=nl.multiply,
                operand0=rms[0:n_rows, :],
                op1=nl.multiply,
                operand1=gamma_sb[0:n_rows, nl.ds(h0, h_len)],
            )
        xt_ps = nl.ndarray((_PMAX, _PMAX), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=xt_ps[0:h_len, 0:n_rows], data=scaled_c[0:n_rows, 0:h_len], engine=nisa.tensor_engine)
        xt_sb = nl.ndarray((_PMAX, _PMAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.tensor_copy(dst=xt_sb[0:h_len, 0:n_rows], src=xt_ps[0:h_len, 0:n_rows])
        # The weight chunk is RESIDENT (loaded once for all row tiles by
        # _load_weight_resident), not re-streamed per tile. Its moving free dim is tiled to
        # the target's PSUM-bank limit (_lm_head_moving_fmax): one matmul per chunk on trn3,
        # four at the production widths on trn2, each accumulating its own column range of
        # acc_ps across the H chunks.
        for v0 in range(0, vocab_shard, v_tile):
            v_len = min(v_tile, vocab_shard - v0)
            nisa.nc_matmul(
                dst=acc_ps[0:n_rows, nl.ds(v0, v_len)],
                stationary=xt_sb[0:h_len, 0:n_rows],
                moving=wt_res[c][0:h_len, nl.ds(v0, v_len)],
            )
    return acc_ps, rms


def _select_by_position(data, ramp, position, n_rows: int, k: int, inf_safe: bool = False):
    """Return ``data[row, position[row]]`` as a ``[P, 1]`` fp32 tile, without nc_n_gather.

    Selection is a masked sum: build ``ramp == position`` and reduce ``mask * data``. That is
    four vector ops on a ``[rows, k]`` tile, against one gather instruction -- but the gather
    is avoided deliberately. ``nc_n_gather`` returned ZERO here for a narrow ``data`` tile
    (k=8) even though the position and the data were both verified correct on device, and the
    same primitive is documented as corrupting results in its multi-group form above
    ``gather_group_size`` (see rotational_topk_utils). A masked sum has no such
    width-dependent behaviour, so this is correct for every k the kernel accepts.

    ``position`` is fp32 holding a small exact integer, and ``ramp`` holds exact integers, so
    the equality is exact rather than approximate. ``(ramp == position) * data`` is ONE fused
    ``scalar_tensor_tensor`` on the Vector Engine -- the compare's 1.0/0.0 feeds the multiply
    inside the instruction -- so the whole selection is two instructions, and every one of the
    sampler's instructions counts: at ``gbs = VTP`` the sampler owns a single batch row, every
    op runs on ONE partition in ~0.3 us of mostly dispatch, and this function sits twice on
    that serial chain.

    ``inf_safe``: the fused form NaN-poisons on ``+/-inf`` data -- ``0.0 * inf = NaN`` in
    every NON-selected slot, and the add-reduce then returns NaN instead of the selected
    value. The values this kernel samples may legally be ``+/-inf`` (GEMM overflow saturates,
    declared in-contract by ``skip_nan_fold``), so the top-k THRESHOLD selection must use the
    inf-safe form: ONE ``nisa.range_select`` (index-range mask + max-reduce, no multiply).
    It requires ``data`` to be DESCENDING along the free dim (the sorted top-k values are)
    and FLOORS ``position``: it returns the max over slots >= floor(position), which on a
    descending row is the slot floor(position) itself. The fused form needs an exact
    integer position, which its one caller (the sampled position, a count) always is.
    """
    out = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
    prod = nl.ndarray((_PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
    if inf_safe:
        # ONE Vector instruction: range_select compares each slot's index + range_start against
        # two per-partition bounds and max-reduces the selected elements. With range_start=1
        # the mask is (s + 1 > position) AND (s + 1 >= position), i.e. every slot s >= FLOOR
        # (position) -- so position may be a non-integer (top_k is fp32 runtime data and the
        # model floors it) and the reduce over a DESCENDING row returns exactly the slot
        # floor(position). No multiply anywhere, so +/-inf data is safe. Unselected slots read
        # FP32_MIN; the one observable edge is a row whose selected slot IS -inf: the
        # threshold comes back FP32_MIN rather than -inf, so -inf values below it get the
        # finite sentinel instead of staying -inf -- both exp to exactly 0 in the softmax.
        # Replaces six ops (iota, two compares, AND, memset, predicated copy, add-reduce).
        sel = nl.ndarray((_PMAX, k), dtype=nl.float32, buffer=nl.sbuf)
        nisa.range_select(
            dst=sel[0:n_rows, :],
            on_true_tile=data[0:n_rows, :],
            comp_op0=nl.greater,
            comp_op1=nl.greater_equal,
            bound0=position[0:n_rows, :],
            bound1=position[0:n_rows, :],
            reduce_op=nl.maximum,
            reduce_res=out[0:n_rows, :],
            reduce_cmd=nisa.reduce_cmd.reset_reduce,
            range_start=1,
        )
        return out
    else:
        nisa.scalar_tensor_tensor(
            dst=prod[0:n_rows, :],
            data=ramp[0:n_rows, :],
            op0=nl.equal,
            operand0=position[0:n_rows, :],
            op1=nl.multiply,
            operand1=data[0:n_rows, :],
        )
    nisa.tensor_reduce(dst=out[0:n_rows, :], data=prod[0:n_rows, :], op=nl.add, axis=1)
    return out


def _sampler_stage(vals_hbm, idx_hbm, params_hbm, n_rows, k, token_out):
    """Sampler on [rows, k]: top-k mask, temperature, softmax, top-p, position, gather.

    Each operand is loaded from HBM INSIDE this function, immediately before its use, and the
    index tile in particular is loaded last. That is not stylistic: an earlier version took
    the index tile as an SBUF argument from the caller, and because the gather is the very
    last thing this function does, that tile was reused by the sampler's own allocations
    before it got there. The gather then read zeros and every rank emitted token id 0 --
    while ``pos`` was provably correct and the top-k outputs validated clean, so only an
    end-to-end token check catches it.

    Two behaviours that look like bugs but are the model's semantics: top-p uses a STRICT
    ``>`` on the inclusive cumsum so the crossing token is DROPPED, and the mask sentinel is
    a finite ``-3000.0`` (which rounds to -3008.0 in bf16), not -inf.
    """
    P = _PMAX
    vals = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=vals[0:n_rows, :], src=vals_hbm[0:n_rows, :])
    # Column ramp 0..k-1, identical on every partition (channel_multiplier=0). Used to pick a
    # slot by EQUALITY instead of by nc_n_gather -- see _select_by_position.
    ramp = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(ramp, pattern=[[1, k]], offset=0, channel_multiplier=0)
    # 3 columns [top_k, top_p, temperature], or 4 with a per-row runtime draw appended -- the
    # width is a trace-time fact of the caller's tensor, so each form specialises cleanly.
    p_cols = params_hbm.shape[1]
    params = nl.ndarray((P, p_cols), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=params[0:n_rows, :], src=params_hbm[0:n_rows, :])
    top_k_col, top_p_col, temp_col = params[0:n_rows, 0:1], params[0:n_rows, 1:2], params[0:n_rows, 2:3]

    # --- top-k threshold: mask everything below the (top_k-1)-th value ---
    kth = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=kth[0:n_rows, :], data=top_k_col, op0=nl.minimum, operand0=float(k))
    nisa.tensor_scalar(dst=kth[0:n_rows, :], data=kth[0:n_rows, :], op0=nl.subtract, operand0=1.0)
    nisa.tensor_scalar(dst=kth[0:n_rows, :], data=kth[0:n_rows, :], op0=nl.maximum, operand0=0.0)
    # kth may be NON-INTEGER (top_k is fp32 runtime data; the model floors it): the inf-safe
    # select below floors it by construction (range select), so no separate floor op runs.
    thresh = _select_by_position(vals, ramp, kth, n_rows, k, inf_safe=True)

    below = nl.ndarray((P, k), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=below[0:n_rows, :], data=vals[0:n_rows, :], op0=nl.less, operand0=thresh[0:n_rows, :])
    masked = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=masked[0:n_rows, :], src=vals[0:n_rows, :])
    sent = _const_tile((P, k), _TOPK_MASK_SENTINEL)
    nisa.tensor_copy_predicated(dst=masked[0:n_rows, :], src=sent[0:n_rows, :], predicate=below[0:n_rows, :])

    # --- temperature (greedy rows divide by 1.0 instead) ---
    greedy = nl.ndarray((P, 1), dtype=nl.uint8, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=greedy[0:n_rows, :], data=temp_col, op0=nl.less, operand0=_SAMPLING_EPS)
    temp_safe = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=temp_safe[0:n_rows, :], src=temp_col)
    ones = _const_tile((P, 1), 1.0)
    nisa.tensor_copy_predicated(dst=temp_safe[0:n_rows, :], src=ones[0:n_rows, :], predicate=greedy[0:n_rows, :])
    inv_t = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=inv_t[0:n_rows, :], data=temp_safe[0:n_rows, :], op=nl.reciprocal)

    # --- softmax (max-shifted), NO normalisation anywhere downstream ---
    # Two ops, and zero divisions on the serial chain. The temperature ratio rides the exp's
    # SCALE operand and the max shift rides its BIAS (the Scalar Engine computes
    # op(data*scale + bias)), so ``exp((masked - rmax)/temp)`` is the max-reduce plus one
    # activation, which also emits the denominator from its reduce registers. The denominator
    # is never divided by: every consumer below compares against a threshold instead, because
    # ``cum/den > p  <=>  cum > p*den`` and ``fcum/last < draw  <=>  fcum < draw*last`` for the
    # positive den/last this softmax produces. That removes two reciprocal activations, two
    # full-width normalising multiplies, and the activation-table reload the reciprocals forced
    # between the exp and the scans (1.28 us on the serial chain). The division and the
    # multiplication round differently in the last ulp -- the same class of wiggle as the PE
    # scan's summation order, arbitrated the same way: by the tie-guarded gates on hardware.
    rmax_neg = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=rmax_neg[0:n_rows, :], data=masked[0:n_rows, :], op=nl.max, axis=1, negate=True)
    exp_bias = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=exp_bias[0:n_rows, :], data1=rmax_neg[0:n_rows, :], data2=inv_t[0:n_rows, :], op=nl.multiply)
    expv = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    den = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(
        dst=expv[0:n_rows, :],
        data=masked[0:n_rows, :],
        op=nl.exp,
        scale=inv_t[0:n_rows, :],
        bias=exp_bias[0:n_rows, :],
        reduce_op=nl.add,
        reduce_res=den[0:n_rows, :],
        reduce_cmd=nisa.reduce_cmd.reset_reduce,
    )

    # --- top-p: strict > on the INCLUSIVE cumsum, so the crossing token is dropped ---
    # The compare is against top_p * den, so the scan runs on the raw exponentials.
    tri = _make_scan_triangle()
    cum = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    _inclusive_scan_pe(cum, expv, n_rows, k, tri)
    pden = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(dst=pden[0:n_rows, :], data1=top_p_col, data2=den[0:n_rows, :], op=nl.multiply)
    # The compare writes 1.0/0.0 straight into fp32 so the mask multiplies without a cast.
    over_f = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=over_f[0:n_rows, :], data=cum[0:n_rows, :], op0=nl.greater, operand0=pden[0:n_rows, :])
    nisa.memset(dst=over_f[0:n_rows, 0:1], value=0)  # never drop the argmax
    # filt = (1 - over) * expv in ONE fused op.
    filt = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.scalar_tensor_tensor(
        dst=filt[0:n_rows, :],
        data=over_f[0:n_rows, :],
        op0=nl.subtract,
        operand0=1.0,
        reverse0=True,
        op1=nl.multiply,
        operand1=expv[0:n_rows, :],
    )
    # --- position: count cdf entries below the draw; greedy rows take slot 0 ---
    # The cdf is never materialised: the reference's ``cumsum(filt)/cumsum(filt)[k-1] < draw``
    # is evaluated as ``fcum < draw * last`` (same last-ulp caveat as the top-p compare above).
    fcum = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    _inclusive_scan_pe(fcum, filt, n_rows, k, tri)
    # With a 4-column params tensor the draw u is a RUNTIME per-row operand, so sweeping u is
    # a data change rather than a recompile; with 3 columns the compiled-in constant keeps the
    # legacy contract (and its bit-exact determinism) unchanged.
    draw_thresh = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(
        dst=draw_thresh[0:n_rows, :],
        data=fcum[0:n_rows, k - 1 : k],
        op0=nl.multiply,
        operand0=params[0:n_rows, 3:4] if p_cols == 4 else _DETERMINISTIC_DRAW,
    )
    lt_f = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_scalar(dst=lt_f[0:n_rows, :], data=fcum[0:n_rows, :], op0=nl.less, operand0=draw_thresh[0:n_rows, :])
    pos = nl.ndarray((P, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_reduce(dst=pos[0:n_rows, :], data=lt_f[0:n_rows, :], op=nl.add, axis=1)

    # Greedy rows take slot 0 BY CONTRACT, so it costs one memset and no scan. The contract
    # (mirrored verbatim in the reference sampler): greedy returns the highest KEPT candidate,
    # and slot 0 is always kept -- the input is DESCENDING and the top-k threshold compare is
    # strict-less against a value <= vals[0]. This is deliberately NOT "argmax of the masked
    # array": in the sub-sentinel regime (every real value below the finite -3000 sentinel,
    # unreachable for real logits but legal bf16) the masked slots numerically outrank the
    # kept ones, and an argmax would return a token the caller's top_k cut explicitly
    # excluded. A masked slot is never an eligible greedy answer. On every other input the
    # two formulations agree (mask replaces a suffix with a smaller sentinel; temperature,
    # exp and top-p all preserve "slot 0 is a maximum"; ties resolve to the first maximum),
    # including fully degenerate all-equal rows.
    #
    # This replaces a max-reduce, a compare, two casts, a full ceil(log2(k))-step prefix scan and
    # a reduce (~30 Vector instructions at k=256) with a single [rows, 1] memset. If a future
    # caller feeds an UNSORTED value tile, this is the line that breaks -- the descending-order
    # gate in the tests is what protects it.
    zero_pos = _const_tile((P, 1), 0.0)
    nisa.tensor_copy_predicated(dst=pos[0:n_rows, :], src=zero_pos[0:n_rows, :], predicate=greedy[0:n_rows, :])

    # --- gather the token id at the chosen position ---
    idx_u32 = nl.ndarray((P, k), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(dst=idx_u32[0:n_rows, :], src=idx_hbm[0:n_rows, :])
    idx_f = nl.ndarray((P, k), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=idx_f[0:n_rows, :], src=idx_u32[0:n_rows, :])
    tok_f = _select_by_position(idx_f, ramp, pos, n_rows, k)
    tok_u32 = nl.ndarray((P, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=tok_u32[0:n_rows, :], src=tok_f[0:n_rows, :])
    nisa.dma_copy(dst=token_out[0:n_rows, :], src=tok_u32[0:n_rows, :])


def _exchange_hops(cur, hops, rows_per_rank, vocab, k):
    """Stages 3-4, plain form: raw logits through every hop, de-interleaving between.

    Returns ``(full_vocab, recv_peer_width)``: the ``[rows_per_rank, vocab]`` assembled rows
    (or, when the last hop's de-interleave folds into chunked_topk, that hop's RECEIVE
    buffer with its peer width so chunked_topk reads it in place -- see
    ``interleaved_peer_width``).
    """
    cur_rows, cur_w = cur.shape
    # The LAST hop's de-interleave can be FOLDED INTO chunked_topk's own load patterns when
    # every stage-1 snake partition covers whole peer blocks (see interleaved_peer_width in
    # ChunkedTopkConfig): stage 5 then reads the receive buffer directly and the [rows, vocab]
    # materialisation -- a full HBM round trip of the received logits, the head of the
    # post-collective serial chain -- is deleted. Bitwise identical: only DMA descriptors
    # change. Holds for the single-hop production shape (peer width == V_s divides the snake
    # row width); the two-hop form's last-hop peer blocks are wider than a snake partition,
    # so it keeps the explicit de-interleave.
    recv_peer_width = 0
    for hop in range(len(hops)):
        groups = hops[hop]
        g = len(groups[0])
        kernel_assert(cur_rows % g == 0, f"hop {hop}: rows ({cur_rows}) must divide by group size ({g})")
        out_rows = cur_rows // g

        recv = nl.ndarray((cur_rows, cur_w), dtype=nl.bfloat16, buffer=nl.shared_hbm)
        ncc.all_to_all(dsts=[recv], srcs=[cur], replica_group=ncc.ReplicaGroup(groups), collective_dim=0)

        # Eligibility is the CALLEE's predicate, not a re-derivation: interleaved_fold_supported
        # carries every chunked_topk precondition (chunks > 1, the desnake-path alignments,
        # the peer-width divisibilities). An earlier geometry-only condition here turned legal
        # small-vocab configs into trace-time refusals on the chunks == 1 delegate path.
        fold_deint = (
            hop == len(hops) - 1 and out_rows > 1 and interleaved_fold_supported(rows_per_rank, vocab, k, cur_w)
        )
        # De-interleave so peer p's block becomes columns [p*cur_w, (p+1)*cur_w). At
        # out_rows==1 the received rows are ALREADY in peer order, so laying them end to
        # end is exactly the target layout and the de-interleave is a pure reshape.
        if fold_deint:
            cur = recv
            recv_peer_width = cur_w
        elif out_rows == 1:
            cur = recv.reshape((1, g * cur_w))
        else:
            # shared_hbm, not private_hbm: this buffer is the SOURCE of the next hop's
            # collective, and every collective src/dst must be a shared buffer. One
            # descriptor per peer block: splitting each block into row halves was tried and
            # measured as no gain (the de-interleave is an HBM round trip at ~90 GB/s,
            # bound by the round trip rather than by descriptor count).
            wide = nl.ndarray((out_rows, g * cur_w), dtype=nl.bfloat16, buffer=nl.shared_hbm)
            for p in range(g):
                nisa.dma_copy(dst=wide[:, nl.ds(p * cur_w, cur_w)], src=recv[nl.ds(p * out_rows, out_rows), :])
            cur = wide
        cur_rows, cur_w = out_rows, g * cur_w

    kernel_assert(
        cur_rows == rows_per_rank and cur_w == vocab,
        f"hops left [{cur_rows}, {cur_w}], expected [{rows_per_rank}, {vocab}]",
    )
    return cur, recv_peer_width


@nki.jit
def gpt_oss_tail_chunked_megakernel(
    hidden_states: nl.NkiTensor,
    gamma: nl.NkiTensor,
    lm_head_weight_t: nl.NkiTensor,
    eps: float = 1e-5,
    hidden_actual: Optional[int] = None,
    zero_interleaved_pad: bool = True,
    k: int = 256,
    sampling_params: Optional[nl.NkiTensor] = None,
    replica_groups: Optional[tuple] = None,
    hop_meshes: Optional[tuple] = None,
    dummy_collectives: bool = False,
    emit_logits: bool = True,
    debug_taps: bool = False,
    deferred_rms: bool = True,
    skip_nan_fold: bool = True,
    a2a_batch_groups: int = 1,
) -> Tuple[nl.NkiTensor, ...]:
    """GPT-OSS decode sampling tail with a single collective. See the module docstring.

    Args:
        hidden_states: ``[gbs, H]`` post-MoE hidden states in HBM; fp32, fp16, or bf16.
            A 16-bit input is widened to fp32 through an explicit on-chip conversion
            copy immediately after its (half-sized) load, which is EXACT (both half
            formats are subsets of fp32) -- so it produces bitwise the same outputs as
            passing a pre-widened fp32 copy, while halving this kernel's largest input
            DMA and letting a 16-bit producer skip an HBM widen round trip. The copy is
            load-bearing: engines fed 16-bit data directly compute at input precision.
        gamma: ``[1, H]`` final RMSNorm weight, fp32.
        lm_head_weight_t: ``[H, V_s]`` **bfloat16** lm_head shard for this rank, ALREADY
            TRANSPOSED and C-contiguous. ``V_s = V/VTP``. bf16 because that is the checkpoint's
            own precision -- see CALLER CONTRACTS.
        eps: RMSNorm epsilon (1e-5 for GPT-OSS).
        hidden_actual: ``H_u``, the unpadded hidden (2880). Defaults to H.
        zero_interleaved_pad: True for the mxfp4 checkpoint (interleaved padding lanes).
        k: candidates kept (256 in production).
        emit_logits: whether to return the fp32 ``[gbs, V_s]`` pre-collective logits. They are a
            DEBUG/GATING output -- nothing downstream of this kernel reads them; the collective
            carries the bf16 mirror instead. Keeping them costs a full extra HBM write of the
            shard (805 KB at the production shape, roughly half this kernel's hbm_write) plus its
            DMA and the fp32 SBUF tile. The tests set it True so the norm and the lm_head GEMM
            stay independently gated; a deployment that only wants tokens should set it False.
        sampling_params: ``[gbs/VTP, 3]`` or ``[gbs/VTP, 4]`` fp32. Columns are
            ``[top_k, top_p, temperature]``; the optional 4th column is the per-row
            inverse-transform draw ``u`` in ``[0, 1)`` -- the RUNTIME form of the compiled-in
            ``_DETERMINISTIC_DRAW`` (0.5), so sweeping the draw is a data change rather than a
            NEFF rebuild. Rows are THIS RANK'S only -- see ``owned_row_block`` for which rows
            those are when more than one hop is used. When None the sampler is skipped and no
            token is returned.
        replica_groups: one group listing all ``VTP`` ranks IN RANK ORDER. Shorthand for a
            single-hop ``hop_meshes=(replica_groups,)``; supply exactly one of the two.
        hop_meshes: one replica-group mesh per all-to-all hop, from
            ``collective_probe.compose_hops``. Hop sizes must multiply to ``VTP``. Needed
            because an all-to-all replica group must be device-local or one-core-per-device --
            see WHICH COLLECTIVE ACTUALLY RUNS in the module docstring.
        dummy_collectives: prepend a zero all-reduce fence per hop mesh so the fence, not the
            real hops, absorbs rank-arrival skew. A PROFILING instrument: it cannot change the
            answer but it adds a collective per mesh to every call, so it defaults off and
            profiling runs opt in. Without it a per-hop duration is mostly dispatch jitter.
        debug_taps: forward ``chunked_topk``'s seven diagnostic taps as extra outputs, APPENDED
            after everything else so the existing output positions do not move. A DIAGNOSTIC
            instrument for hardware-only index defects -- see DEBUG TAPS in ``chunked_topk``.
            It changes this kernel's arity, so leave it off unless the consumer expects them.
        deferred_rms: apply the per-row RMSNorm scale on the GEMM's PSUM-read copies instead
            of on the GEMM's input (exact algebra: ``rms`` is a per-row scalar and the GEMM
            is linear), which stops every matmul from waiting on the full row's sum of
            squares -- measured -3.8 us on the fenced gbs=128 head. Numerically the SAME
            one-bf16-rounding-per-element bound; only the last-ulp realization moves, so
            fixed-seed sampled tokens can shift at cdf crossings (same class as any
            resummation change). Default True; a SAFETY FALLBACK to the legacy in-chain
            form -- ``(x*rms)*gamma`` feeding the GEMM, the pre-deferred realization
            bit-for-bit -- is one trace-time flag away if a consumer needs it.
        skip_nan_fold: skip ``chunked_topk``'s stage-1 NaN fold. Sound here because the top-k
            input is this kernel's OWN GEMM output: finite hidden/gamma/weight inputs give
            finite bf16 products summed in fp32, and even an overflow saturates to +/-inf --
            which ``nisa.topk`` orders fine (the fold only exists for NaN, which would take
            inf arithmetic INSIDE the GEMM, i.e. already-garbage inputs). The fold costs a
            full-width compare + predicated copy per stage-1 tile on the post-collective
            serial chain. Default True; the fold is one trace-time flag away for a caller
            that wants the belt-and-suspenders behavior back. Outputs are BITWISE identical
            either way on any NaN-free input -- unlike ``deferred_rms`` this cannot move even
            an ulp, because it deletes ops whose effect on finite data is the identity.

        a2a_batch_groups: split the exchange into this many batch-group collectives, one
            per gbs/G whole lm_head tiles (single-hop only; G must divide rows_per_rank and
            the geometry must support the receive-layout fold -- asserted; groups aligned to
            whole 8-row sort tiles overlap best but are not required). Group g's
            exchange fires as soon as its own lm_head tiles' mirror stores land -- under
            the rest of the GEMM -- pulling the exchange forward by up to (G-1)/G of its
            wire time minus (G-1) launch serializations. OWNERSHIP CONTRACT CHANGES with
            G > 1: rank r owns the STRIDED row set given by ``owned_rows`` (one extra
            mixed-radix digit), not ``owned_row_block``'s contiguous block --
            ``sampling_params`` rows and every consumer of the per-rank outputs must use
            that map. Gate-validated, not bitwise: regrouping a collective changes the
            runtime's internal transfer realization.

    Returns:
        ``(logits, topk_values, topk_indices)`` and, when sampling, ``tokens``.
        ``logits`` is this rank's ``[gbs, V_s]`` fp32 shard. ``topk_values`` and
        ``topk_indices`` are this rank's ``[gbs/VTP, k]`` rows, with indices being GLOBAL
        vocab ids. ``tokens`` is ``[gbs/VTP, 1]``. Which batch rows this rank owns is
        ``owned_rows(rank, VTP, gbs, hops, a2a_batch_groups)``: the IDENTITY for a single
        ungrouped hop, so a rank-ordered ``all_gather`` concatenates straight into batch
        order, but a mixed-radix shuffle for multi-hop or a STRIDED set for the batch-grouped
        exchange, either of which the caller must undo.
    """
    # hidden_states may also be a LIST of row-slab tensors [rows_i, H] (concatenated-by-rows
    # == one [gbs, H] tensor) -- the pipelined-seam form, reachable only jit-in-jit (a
    # top-level kernel invocation rejects list args). Separate slab tensors are the point:
    # each lm_head row tile then data-depends only on ITS slab's producing collective, so a
    # caller that splits its final all_reduce into K row-slab collectives gets the head
    # overlapped under the later slabs (measured on the fused GPT-OSS decode: the tail
    # window from seam-collective start to tail end shrinks ~6-8% at gbs 512/1024 with
    # K = 2; gate-validated, not bitwise -- splitting a collective changes the runtime's
    # internal reduction realization).
    hidden_slabs = list(hidden_states) if isinstance(hidden_states, (list, tuple)) else [hidden_states]
    for hs in hidden_slabs:
        kernel_assert(len(hs.shape) == 2, "hidden_states (slab) must be [rows, H]")
        kernel_assert(
            hs.dtype in (nl.float32, nl.float16, nl.bfloat16),
            f"hidden_states must be fp32/fp16/bf16 (16-bit widens exactly), got {hs.dtype}",
        )
        kernel_assert(hs.shape[1] == hidden_slabs[0].shape[1], "hidden_states slabs must share H")
        if len(hidden_slabs) > 1:
            kernel_assert(
                hs.shape[0] % _PMAX == 0,
                f"each hidden_states slab must hold whole {_PMAX}-row tiles, got {hs.shape[0]} rows",
            )
    kernel_assert(len(lm_head_weight_t.shape) == 2, "lm_head_weight_t must be [H, V_s]")
    # Enforced, not just documented: any other dtype would still trace and compute correctly
    # (the resident load casts on DMA), but silently doubles the kernel's dominant HBM read --
    # the exact failure mode only a profile would ever catch.
    kernel_assert(
        lm_head_weight_t.dtype == nl.bfloat16,
        f"lm_head_weight_t must be bfloat16 (caller contract 1; any wider dtype would silently "
        f"inflate the dominant HBM read via cast-on-DMA), got {lm_head_weight_t.dtype}",
    )
    kernel_assert(
        (replica_groups is None) != (hop_meshes is None),
        "supply exactly one of replica_groups (single hop, all VTP ranks in rank order) or hop_meshes",
    )
    hops = (replica_groups,) if hop_meshes is None else tuple(hop_meshes)
    kernel_assert(len(hops) >= 1, "at least one hop is required")

    gbs = sum(hs.shape[0] for hs in hidden_slabs)
    hidden = hidden_slabs[0].shape[1]
    w_hidden, v_s = lm_head_weight_t.shape
    kernel_assert(w_hidden == hidden, f"weight hidden {w_hidden} must match hidden_states hidden {hidden}")

    vtp = 1
    for mesh in hops:
        vtp *= len(mesh[0])
    kernel_assert(gbs % vtp == 0, f"gbs ({gbs}) must be divisible by VTP ({vtp})")
    rows_per_rank = gbs // vtp
    vocab = vtp * v_s

    h_actual = hidden if hidden_actual is None else hidden_actual
    kernel_assert(0 < h_actual <= hidden, f"hidden_actual ({h_actual}) must be in (0, {hidden}]")
    if h_actual < hidden and zero_interleaved_pad:
        kernel_assert(
            hidden % _MXFP4_SHUFFLE_GROUPS == 0 and h_actual % _MXFP4_SHUFFLE_GROUPS == 0,
            f"mxfp4 padding needs H ({hidden}) and H_u ({h_actual}) divisible by {_MXFP4_SHUFFLE_GROUPS}",
        )
    if sampling_params is not None:
        kernel_assert(len(sampling_params.shape) == 2, "sampling_params must be [gbs/VTP, 3]")
        kernel_assert(
            sampling_params.shape[0] == rows_per_rank,
            f"sampling_params must hold THIS RANK'S {rows_per_rank} rows, got {sampling_params.shape[0]}; "
            f"passing the full [gbs, 3] batch makes every rank use rows 0..{rows_per_rank - 1}",
        )
        kernel_assert(
            sampling_params.shape[1] in (3, 4),
            f"sampling_params needs columns [top_k, top_p, temperature] plus an optional 4th "
            f"per-row draw in [0, 1), got {sampling_params.shape[1]} columns",
        )

    # Every stage after the collective is unsharded and carries no core_barrier, so two
    # programs would race on the same shared_hbm buffers. Assert rather than assume: passing
    # num_programs=1 to an inner kernel does not pin the live grid.
    _, n_prgs, _ = get_verified_program_sharding_info("gpt_oss_tail_chunked_megakernel", (0, 1), 2)
    kernel_assert(n_prgs == 1, f"this kernel is LNC=1 only, got n_prgs={n_prgs}")

    fence = _skew_fence(hops, hidden) if dummy_collectives else None

    # DEBUG-ONLY output, gated: see emit_logits. When off, neither the buffer nor its store
    # exists, so it costs nothing at all rather than costing a store nobody reads.
    logits = nl.ndarray((gbs, v_s), dtype=nl.float32, buffer=nl.shared_hbm) if emit_logits else None
    # bf16 mirror for the collective: chunked_topk is bf16-only and the collective then moves
    # half the bytes. shared_hbm because every collective src/dst must share a buffer type.
    # With a2a_batch_groups > 1 the mirror is G SEPARATE tensors, one per batch group of
    # gbs/G rows: separate tensors are what give each group's exchange a data dependency on
    # ONLY its own lm_head tiles, so group g's all-to-all fires as soon as tile-group g's
    # stores land -- under the rest of the GEMM -- instead of after the whole head.
    _bg = a2a_batch_groups
    kernel_assert(_bg >= 1 and gbs % _bg == 0, f"a2a_batch_groups ({_bg}) must divide gbs ({gbs})")
    _bg_rows = gbs // _bg
    if _bg > 1:
        kernel_assert(len(hops) == 1, "a2a_batch_groups > 1 is defined for the single-hop exchange only")
        kernel_assert(_bg_rows % _PMAX == 0, f"each batch group ({_bg_rows} rows) must be whole lm_head tiles")
        kernel_assert(rows_per_rank % _bg == 0, f"a2a_batch_groups ({_bg}) must divide rows_per_rank ({rows_per_rank})")
        kernel_assert(
            interleaved_fold_supported(rows_per_rank, vocab, k, v_s),
            "a2a_batch_groups > 1 requires the receive-layout fold (single-hop production-class geometry)",
        )
        # OWNERSHIP CONTRACT CHANGE: with G groups rank r owns rows {g*gbs/G + r*rpc + j}
        # (see owned_rows) -- strided, not the contiguous owned_row_block. sampling_params
        # and every consumer of this kernel's per-rank outputs must use that map.
    mirrors = [
        nl.ndarray((_bg_rows, v_s), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=f"logits_bf16_g{i}")
        for i in range(_bg)
    ]

    # ---- Stages 1-2: norm + lm_head, fused through SBUF, per 128-row tile ----
    gamma_sb = _load_gamma_broadcast(gamma, hidden, h_actual, zero_interleaved_pad)
    wt_res = _load_weight_resident(lm_head_weight_t, hidden, v_s)
    # Slab-start offsets, so each 128-row tile reads ITS slab at a local offset (a tile
    # never straddles slabs: slab rows are whole multiples of _PMAX, asserted above).
    slab_row0 = []
    _acc_rows = 0
    for hs in hidden_slabs:
        slab_row0.append(_acc_rows)
        _acc_rows += hs.shape[0]
    for t in range(div_ceil(gbs, _PMAX)):
        row0 = t * _PMAX
        rows = min(_PMAX, gbs - row0)
        s = max(i for i in range(len(hidden_slabs)) if slab_row0[i] <= row0)
        acc_ps, rms = _norm_lm_head_stage(
            hidden_slabs[s],
            gamma_sb,
            wt_res,
            eps,
            rows,
            row0 - slab_row0[s],
            hidden,
            h_actual,
            v_s,
            fence,
            deferred_rms,
        )
        # Both readers pull straight from PSUM. With deferred_rms the per-row rms rides each
        # read as the multiply that replaces what was a plain copy -- the bf16 cast and the
        # fp32 debug copy both had to happen anyway, so normalization costs zero extra ops
        # here; without it acc_ps is already normalized and the reads are plain copies.
        # The two reads stay independent rather than the cast chaining behind the fp32 copy.
        # The mirror store is split by row halves (two contiguous descriptors on separate
        # queues): it is the last transfer standing between the GEMM and the first hop's
        # trigger, so its latency is bare critical path.
        # (An engine-parallel column-half split of this read -- Scalar || Vector, to free
        # acc_ps sooner for the next tile's PSUM -- measured ZERO at 512 and 1024 on
        # 2026-08-28: the tile-boundary PE gap does not gate on this drain. Reverted.)
        bf = nl.ndarray((_PMAX, v_s), dtype=nl.bfloat16, buffer=nl.sbuf)
        if deferred_rms:
            nisa.tensor_scalar(dst=bf[0:rows, :], data=acc_ps[0:rows, :], op0=nl.multiply, operand0=rms[0:rows, :])
        else:
            nisa.tensor_copy(dst=bf[0:rows, :], src=acc_ps[0:rows, :])
        st_half = rows // 2
        _mg, _mr0 = row0 // _bg_rows, row0 % _bg_rows
        if st_half > 0:
            nisa.dma_copy(dst=mirrors[_mg][_mr0 : _mr0 + st_half, :], src=bf[0:st_half, :])
            nisa.dma_copy(dst=mirrors[_mg][_mr0 + st_half : _mr0 + rows, :], src=bf[st_half:rows, :])
        else:
            nisa.dma_copy(dst=mirrors[_mg][_mr0 : _mr0 + rows, :], src=bf[0:rows, :])
        if emit_logits:
            logits_sb = nl.ndarray((_PMAX, v_s), dtype=nl.float32, buffer=nl.sbuf)
            if deferred_rms:
                nisa.tensor_scalar(
                    dst=logits_sb[0:rows, :], data=acc_ps[0:rows, :], op0=nl.multiply, operand0=rms[0:rows, :]
                )
            else:
                nisa.tensor_copy(dst=logits_sb[0:rows, :], src=acc_ps[0:rows, :])
            nisa.dma_copy(dst=logits[row0 : row0 + rows, :], src=logits_sb[0:rows, :])

    # ---- Stages 3-4, plain: raw logits through every hop, de-interleaving between ----
    # Each hop is an all_to_all followed by a rows->columns de-interleave, and hops
    # compose: a hop over g peers turns [rows, w] into [rows/g, g*w]. HBM forces
    # collective_dim=0, so the all_to_all splits AND concatenates on rows -- rank r ships
    # row block p to peer p and gets every peer's block back. Because every hop's group is
    # listed in ASCENDING RANK ORDER and the first hop's is 8 contiguous ranks, peer p's
    # block is peer p's vocab shard at every hop, so after the last hop a column index
    # already IS a global vocab id and no index correction of any kind appears. What
    # multi-hop costs instead is row ownership: see owned_row_block.
    if _bg > 1:
        # G exchanges, one per batch group: exchange g's src is mirror g (whole lm_head
        # tiles), so it fires as soon as those tiles' stores land -- under the rest of the
        # GEMM -- and the last exchange trails the head by only ~1/G of the wire time plus
        # one launch. Same-mesh collectives serialize end-to-end (measured), which is why G
        # stays small and why the win is overlap, not wire speed. Each exchange delivers
        # every rank its rpc = rows_per_rank/G rows of that group as a complete peer-major
        # slab; the slabs go to chunked_topk as a LIST (per-slab data dependencies), whose
        # receive-layout fold reads each in place -- no de-interleave, no concat.
        _recv_slabs = []
        for _g in range(_bg):
            _rv = nl.ndarray((_bg_rows, v_s), dtype=nl.bfloat16, buffer=nl.shared_hbm, name=f"a2a_recv_g{_g}")
            ncc.all_to_all(dsts=[_rv], srcs=[mirrors[_g]], replica_group=ncc.ReplicaGroup(hops[0]), collective_dim=0)
            _recv_slabs.append(_rv)
        full_vocab = _recv_slabs
        recv_peer_width = v_s
    else:
        full_vocab, recv_peer_width = _exchange_hops(mirrors[0], hops, rows_per_rank, vocab, k)

    # ---- Stage 5: the whole top-k, locally. Indices come back as GLOBAL vocab ids ----
    topk_out = chunked_topk(
        full_vocab,
        create_chunked_topk_config(
            inp_shape=(rows_per_rank, vocab),
            inp_dtype=nl.bfloat16,
            k=k,
            sorted=True,
            debug_taps=debug_taps,
            # The logits are this kernel's own GEMM output, provably NaN-free -- see the
            # skip_nan_fold docstring for the argument and the flag to restore the fold.
            skip_nan_fold=skip_nan_fold,
            # Non-zero when the last hop's de-interleave was folded into stage 1's loads:
            # full_vocab is then the RECEIVE buffer and stage 1 reads it in place.
            interleaved_peer_width=recv_peer_width,
            interleaved_batch_groups=_bg,
        ),
    )
    tk_vals = topk_out[0]
    tk_idx = topk_out[1]
    # The taps, if any, ride through unchanged and are appended to this kernel's outputs
    # LAST, so no existing output position moves and a consumer that ignores them works.
    tk_taps = topk_out[2:] if debug_taps else ()

    # ---- Stage 6: publish the top-k results as this kernel's own outputs ----
    # chunked_topk's buffers are private_hbm (invisible to the caller), so a republish is
    # required -- but it is a straight HBM->HBM DMA per tensor, not a bounce through SBUF.
    topk_values = nl.ndarray((rows_per_rank, k), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    topk_indices = nl.ndarray((rows_per_rank, k), dtype=nl.uint32, buffer=nl.shared_hbm)
    nisa.dma_copy(dst=topk_values, src=tk_vals)
    nisa.dma_copy(dst=topk_indices, src=tk_idx)

    if sampling_params is None:
        base = (logits, topk_values, topk_indices) if emit_logits else (topk_values, topk_indices)
        return base + tk_taps

    # ---- Stage 7: sampler, in its OWN loop over this rank's rows ----
    # Separate from the loop above so each operand is read from HBM inside the sampler, close
    # to its use. See _sampler_stage for why that ordering is load-bearing. The sampler reads
    # chunked_topk's own tk_* buffers, NOT the published shared copies: same bytes, but this
    # way the two republish DMAs above run concurrently with the sampler instead of gating it.
    tokens = nl.ndarray((rows_per_rank, 1), dtype=nl.uint32, buffer=nl.shared_hbm)
    for t in range(div_ceil(rows_per_rank, _PMAX)):
        r0 = t * _PMAX
        rows = min(_PMAX, rows_per_rank - r0)
        _sampler_stage(
            tk_vals[r0 : r0 + rows, :],
            tk_idx[r0 : r0 + rows, :],
            sampling_params[r0 : r0 + rows, :],
            rows,
            k,
            tokens[r0 : r0 + rows, :],
        )
    base = (logits, topk_values, topk_indices, tokens) if emit_logits else (topk_values, topk_indices, tokens)
    return base + tk_taps
