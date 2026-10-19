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
"""Ring Attention Backward pass kernel using collective permute for distributed gradient computation."""

import nki
import nki.collectives as ncc
import nki.isa as nisa
import nki.language as nl

from ...core.attention.attention_bwd import (
    _flash_attn_bwd_core,
    compute_rowsum_single_tile,
    get_required_tiles_mask,
    load_kv,
    load_q_dy,
    ndarray,
    setup_config,
    transpose_tiles,
)
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.stream_shuffle_broadcast import stream_shuffle_broadcast


@nki.jit
def ring_attention_spmd_bwd(
    q_ref: nl.NkiTensor,
    k_ref: nl.NkiTensor,
    v_ref: nl.NkiTensor,
    o_ref: nl.NkiTensor,
    dy_ref: nl.NkiTensor,
    lse_ref: nl.NkiTensor,
    use_causal_mask: bool = False,
    mixed_precision: bool = True,
    softmax_scale: float = None,
    num_workers: int = 1,
    lnc_size: int = 1,
    replica_groups: tuple = None,
    striped_attention: bool = False,
    bound_min: nl.NkiTensor = None,
    bound_max: nl.NkiTensor = None,
):
    """
    Ring Attention Backward SPMD kernel.

    Computes gradients dQ, dK, dV for ring attention using collective permute
    operations to circulate Q, dY, LSE, and dy_o_sum across workers while
    keeping K, V local. Supports causal masking and striped attention.
    TODO: Specify intended usage range (e.g., sequence length, batch size)

    Dimensions:
        B: Batch size
        N: Number of attention heads
        D: Head dimension
        S: Sequence length per shard

    Args:
        q_ref (nl.NkiTensor): [B, N, D, S], Query tensor in HBM.
        k_ref (nl.NkiTensor): [B, N, D, S], Key tensor in HBM.
        v_ref (nl.NkiTensor): [B, N, D, S], Value tensor in HBM.
        o_ref (nl.NkiTensor): [B, N, D, S], Forward output tensor in HBM.
        dy_ref (nl.NkiTensor): [B, N, D, S], Upstream gradient tensor in HBM.
        lse_ref (nl.NkiTensor): [B, N, 128, S//128], Log-sum-exp from forward pass in HBM.
        use_causal_mask (bool): Whether to apply causal masking. Default: False.
        mixed_precision (bool): Whether to use mixed precision (fp32 accumulators). Default: True.
        softmax_scale (float): Softmax scale factor. Default: 1/sqrt(D).
        num_workers (int): Number of workers in the ring. Default: 1.
        lnc_size (int): LNC size (number of logical cores). Default: 1.
        replica_groups (list): Replica groups for collective communication. Default: None.
        striped_attention (bool): Whether to use striped attention layout. Default: False.
        bound_min (nl.NkiTensor, optional): Sequence packing lower bound. Shape
            (bs, seqlen_per_rank), fp32. Per-local-Q-token inclusive lower bound on
            the local K index of the document this Q token belongs to. Requires
            use_causal_mask=True and striped_attention=True. Identical across ranks
            (striped CP invariant with doc length divisible by num_workers).
            Use test/integration/nkilib/utils/sequence_packing_helpers.py::
            cu_seqlens_to_striped_bounds() to build it.
        bound_max (nl.NkiTensor, optional): Sequence packing upper bound (exclusive).
            Same shape/dtype/semantics as bound_min. Must be provided together.

    Returns:
        out_dq (nl.NkiTensor): [B, N, D, S], Query gradient in HBM (float32).
        out_dk (nl.NkiTensor): [B, N, D, S], Key gradient in HBM (float32).
        out_dv (nl.NkiTensor): [B, N, D, S], Value gradient in HBM (float32).

    Notes:
        - Sequence length S must be divisible by 128.
        - striped_attention requires use_causal_mask=True.
        - When B is not divisible by lnc_size, the last batch is handled with duplicate work.

    Pseudocode:
        # For each (batch, head) pair:
        for bid in range(B):
            for hid in range(N):
                # Step 0: compute local Q against local K/V
                dQ_local, dK, dV = bwd_core(Q_local, K_local, V_local, dY_local)

                # Ring steps 1..nw-1: circulate Q, dY, LSE, dy_o_sum
                for step in range(1, num_workers):
                    permute(Q, dY, LSE, dy_o_sum)
                    dQ_step = bwd_core(Q_recv, K_local, V_local, dY_recv)
                    dQ = reduce(dQ_step, dQ_prev)  # collective reduce
                    dK += dK_step
                    dV += dV_step

        return dQ, dK, dV
    """
    if striped_attention:
        kernel_assert(use_causal_mask, "striped_attention requires use_causal_mask")
    bs, nh, dh, sl = q_ref.shape
    kernel_assert(sl % 128 == 0, f"seqlen must be divisible by 128, got {sl}")

    # Sequence packing: bound_min and bound_max must be provided together. Under
    # striped CP with each padded doc length a multiple of num_workers, the local
    # document layout is identical across ranks, so the caller produces one pair
    # of bound tensors and replicates them to every rank.
    is_sequence_packed = bound_min is not None
    kernel_assert(
        is_sequence_packed == (bound_max is not None),
        "bound_min and bound_max must both be provided or both be None",
    )
    if is_sequence_packed:
        kernel_assert(use_causal_mask, "bound_min/bound_max require use_causal_mask=True")
        kernel_assert(striped_attention, "bound_min/bound_max require striped_attention=True")

    if replica_groups is None:
        replica_groups = (tuple(range(num_workers)),)

    out_dq = nl.ndarray(q_ref.shape, dtype=nl.float32, buffer=nl.shared_hbm, name="out_dq")
    out_dk = nl.ndarray(k_ref.shape, dtype=nl.float32, buffer=nl.shared_hbm, name="out_dk")
    out_dv = nl.ndarray(v_ref.shape, dtype=nl.float32, buffer=nl.shared_hbm, name="out_dv")

    # Rank ID lookup table for dynamic rank computation (needed for causal bounds)
    iota_nw_sb = nl.ndarray((1, num_workers), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(iota_nw_sb, [[1, num_workers]], offset=0)
    iota_nw = nl.ndarray((1, num_workers), dtype=nl.float32, buffer=nl.shared_hbm, name="iota_nw")
    nisa.dma_copy(dst=iota_nw, src=iota_nw_sb)

    lnc_n = nl.num_programs(axes=0)
    lnc_id = nl.program_id(0) if lnc_n > 1 else 0

    qbs = dh * sl
    lbs = 128 * (sl // 128)
    sq = nl.ndarray((lnc_size * qbs, 1), dtype=q_ref.dtype, buffer=nl.shared_hbm, name="sq")
    rq = nl.ndarray((lnc_size * qbs, 1), dtype=q_ref.dtype, buffer=nl.shared_hbm, name="rq")
    sdy = nl.ndarray((lnc_size * qbs, 1), dtype=q_ref.dtype, buffer=nl.shared_hbm, name="sdy")
    rdy = nl.ndarray((lnc_size * qbs, 1), dtype=q_ref.dtype, buffer=nl.shared_hbm, name="rdy")
    slse = nl.ndarray((lnc_size * lbs, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="slse")
    rlse = nl.ndarray((lnc_size * lbs, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="rlse")
    sdos = nl.ndarray((lnc_size * lbs, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="sdos")
    rdos = nl.ndarray((lnc_size * lbs, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="rdos")
    sdq = nl.ndarray((lnc_size * qbs, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="sdq")
    rdq = nl.ndarray((lnc_size * qbs, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="rdq")
    dq_scratch0 = nl.ndarray((lnc_size * qbs, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="dq_scratch0")
    dq_scratch1 = nl.ndarray((lnc_size * qbs, 1), dtype=nl.float32, buffer=nl.shared_hbm, name="dq_scratch1")

    """
    Partition (batch, head) work across LNC cores.
    Flatten bs*nh work items and assign each to an NC. Both NCs iterate over all
    items but only compute for their assigned ones; SPMD requires all NCs to issue
    the same collective ops.
    """
    total_work = bs * nh
    num_rounds = (total_work + lnc_size - 1) // lnc_size
    has_dummy_round = (total_work % lnc_size) != 0

    for round_idx in range(num_rounds):
        sample_idx = round_idx * lnc_size + lnc_id
        bid = sample_idx // nh
        hid = sample_idx % nh

        # Last round may have fewer items than lnc_size.
        # The NC whose sample_idx >= total_work writes to dummy outputs.
        is_dummy = has_dummy_round and (round_idx == num_rounds - 1) and (lnc_id >= total_work % lnc_size)
        if is_dummy:
            out_dq_eff = nl.ndarray(q_ref.shape, dtype=nl.float32, buffer=nl.shared_hbm, name="out_dq_rem")
            out_dk_eff = nl.ndarray(k_ref.shape, dtype=nl.float32, buffer=nl.shared_hbm, name="out_dk_rem")
            out_dv_eff = nl.ndarray(v_ref.shape, dtype=nl.float32, buffer=nl.shared_hbm, name="out_dv_rem")
            # Clamp bid/hid to valid range for dummy iteration
            bid = 0
            hid = 0
        else:
            out_dq_eff = out_dq
            out_dk_eff = out_dk
            out_dv_eff = out_dv

        _ring_bwd_impl(
            q_ref,
            k_ref,
            v_ref,
            o_ref,
            dy_ref,
            lse_ref,
            out_dq_eff,
            out_dk_eff,
            out_dv_eff,
            sq,
            rq,
            sdy,
            rdy,
            slse,
            rlse,
            sdos,
            rdos,
            sdq,
            rdq,
            dq_scratch0,
            dq_scratch1,
            bid,
            hid,
            lnc_size,
            lnc_id,
            replica_groups,
            iota_nw,
            num_workers,
            use_causal_mask,
            striped=striped_attention,
            mp=mixed_precision,
            ss=softmax_scale,
            bound_min=bound_min,
            bound_max=bound_max,
        )

    return out_dq, out_dk, out_dv


def _build_causal_bounds(qts, qnt, striped, recv_rank_sb=None, my_rank_sb=None, gko_sb=None):
    """
    Build range_select upper/lower bounds for causal masking (no packing).

    For striped: ub = local_q_pos - no_include_diagonal, lb = 0.
    For contiguous: ub = local_q_pos + gko, lb = 0.
    All rank tensors must be (qts, 1) SBUF, broadcast to all partitions.

    Args:
        qts (int): Q sequence tile size.
        qnt (int): Number of Q sequence tiles.
        striped (bool): Whether striped attention layout is used.
        recv_rank_sb (nl.NkiTensor): [qts, 1], Receiver rank ID in SBUF.
        my_rank_sb (nl.NkiTensor): [qts, 1], Current rank ID in SBUF.
        gko_sb (nl.NkiTensor): [qts, 1], Global K offset in SBUF.

    Returns:
        tuple: (rs_ub, rs_lb) upper and lower bound tensors for range_select.
    """
    rs_ub = nl.ndarray((qts, qnt), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(rs_ub, pattern=[[qts, qnt]], channel_multiplier=1)

    if striped:
        no_diag = nl.ndarray((qts, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(no_diag, my_rank_sb, op0=nl.subtract, operand0=recv_rank_sb)
        nisa.tensor_scalar(no_diag, no_diag, op0=nl.maximum, operand0=0.0)
        nisa.tensor_scalar(no_diag, no_diag, op0=nl.minimum, operand0=1.0)
        nisa.tensor_scalar(rs_ub, rs_ub, op0=nl.subtract, operand0=no_diag)
    else:
        if gko_sb is not None:
            nisa.tensor_scalar(rs_ub, rs_ub, nl.add, gko_sb)

    rs_lb = nl.ndarray((qts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(rs_lb, value=0.0)
    return rs_ub, rs_lb


def _build_bound_max_clamped(
    qts,
    qnt,
    bound_max_sb,
    striped,
    recv_rank_sb=None,
    my_rank_sb=None,
    gko_sb=None,
):
    """
    Build bound_max_clamped = min(bound_max, causal_ub_exclusive) for the
    use_sequence_packing path in recompute_qk_softmax.

    The use_sequence_packing path uses ``comp_op1=nl.less`` with
    ``bound1=bound_max_clamped``, so ``bound_max_clamped`` is the EXCLUSIVE
    causal upper bound clamped to the packing upper bound:

        causal_ub_exclusive[p, g] = iota[p, g] + 1 - no_diag   (striped)
                                  = iota[p, g] + 1 + gko      (contiguous)
        bound_max_clamped[p, g]   = min(bound_max_sb[p, g], causal_ub_exclusive[p, g])

    This matches the clamp that attention_bwd does in its non-ring sequence_packing
    code path (see ``_attention_bwd`` where it clamps ``bound_max_sbuf`` with
    ``iota(offset=1 + cp_offset)``), with ``cp_offset = -no_diag`` for striped CP.

    Args:
        qts (int): Q sequence tile size.
        qnt (int): Number of Q sequence tiles.
        bound_max_sb (nl.NkiTensor): [qts, qnt] fp32, base packing upper bound (exclusive).
        striped (bool): Whether striped attention layout is used.
        recv_rank_sb (nl.NkiTensor): [qts, 1], Receiver rank ID in SBUF (striped only).
        my_rank_sb (nl.NkiTensor): [qts, 1], Current rank ID in SBUF (striped only).
        gko_sb (nl.NkiTensor): [qts, 1], Global K offset in SBUF (contiguous only).

    Returns:
        nl.NkiTensor: [qts, qnt] fp32, the composed exclusive upper bound.
    """
    # iota + 1 gives the exclusive causal upper bound (k < iota + 1 means k <= iota).
    causal_ub = nl.ndarray((qts, qnt), dtype=nl.float32, buffer=nl.sbuf)
    nisa.iota(causal_ub, pattern=[[qts, qnt]], offset=1, channel_multiplier=1)

    if striped:
        no_diag = nl.ndarray((qts, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(no_diag, my_rank_sb, op0=nl.subtract, operand0=recv_rank_sb)
        nisa.tensor_scalar(no_diag, no_diag, op0=nl.maximum, operand0=0.0)
        nisa.tensor_scalar(no_diag, no_diag, op0=nl.minimum, operand0=1.0)
        nisa.tensor_scalar(causal_ub, causal_ub, op0=nl.subtract, operand0=no_diag)
    else:
        if gko_sb is not None:
            nisa.tensor_scalar(causal_ub, causal_ub, nl.add, gko_sb)

    bound_max_clamped = nl.ndarray((qts, qnt), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(bound_max_clamped, bound_max_sb, causal_ub, op=nl.minimum)
    return bound_max_clamped


def _load_rank_sb(iota_nw, scalar_rank, qts):
    """
    Load a scalar rank ID into a (qts, 1) SBUF tensor via dma_copy and broadcast.

    Args:
        iota_nw (nl.NkiTensor): Shared constant iota tensor for rank lookup.
        scalar_rank (int): Scalar rank ID to load.
        qts (int): Q sequence tile size (partition dimension).

    Returns:
        nl.NkiTensor: [qts, 1], Rank ID broadcast to all partitions in SBUF.
    """
    sb = nl.ndarray((qts, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=sb[0, 0],
        src=iota_nw.ap(pattern=[[1, 1], [1, 1]], offset=0, scalar_offset=scalar_rank, indirect_dim=1),
    )
    stream_shuffle_broadcast(src=sb, dst=sb)
    return sb


def _permute_all(sq, rq, sdy, rdy, slse, rlse, sdos, rdos, rg, ch, lnc, barrier=True):
    """
    Collective permute Q, dY, LSE, and dy_o_sum buffers to the next ring step.

    Args:
        sq (nl.NkiTensor): Send buffer for Q.
        rq (nl.NkiTensor): Receive buffer for Q.
        sdy (nl.NkiTensor): Send buffer for dY.
        rdy (nl.NkiTensor): Receive buffer for dY.
        slse (nl.NkiTensor): Send buffer for LSE.
        rlse (nl.NkiTensor): Receive buffer for LSE.
        sdos (nl.NkiTensor): Send buffer for dy_o_sum.
        rdos (nl.NkiTensor): Receive buffer for dy_o_sum.
        rg (ReplicaGroup): Replica group for collective communication.
        ch (int): Channel ID for collective permute.
        lnc (int): LNC size (number of logical cores).
        barrier (bool): Whether to issue core_barrier before permute.
            Required when NCs write to send buffers (e.g., initial copy).
            Not needed with ping-pong when send buffer was written by CC.
    """
    if barrier and lnc > 1:
        nisa.core_barrier(sq, (0, 1))
        nisa.core_barrier(sdy, (0, 1))
        nisa.core_barrier(slse, (0, 1))
        nisa.core_barrier(sdos, (0, 1))
    ncc.collective_permute_implicit(
        srcs_by_channel=((sq,),), dsts_by_channel=((rq,),), replica_group=rg, channel_ids=(ch,)
    )
    ncc.collective_permute_implicit(
        srcs_by_channel=((sdy,),), dsts_by_channel=((rdy,),), replica_group=rg, channel_ids=(ch,)
    )
    ncc.collective_permute_implicit(
        srcs_by_channel=((slse,),), dsts_by_channel=((rlse,),), replica_group=rg, channel_ids=(ch,)
    )
    ncc.collective_permute_implicit(
        srcs_by_channel=((sdos,),), dsts_by_channel=((rdos,),), replica_group=rg, channel_ids=(ch,)
    )


def _compute_step(
    cfg,
    q_buf,
    dy_buf,
    exp_bias,
    dos,
    dq_buf,
    k_loaded,
    v_loaded,
    dk_red,
    dv_red,
    kernel_dtype,
    mixed_dtype,
    softmax_scale,
    causal,
    q_seq_tile_size,
    q_seq_n_tiles,
    k_seq_tile_size,
    k_seq_n_tiles,
    d_head_n_tiles,
    d_head_tile_size,
    seqlen,
    global_k_offset,
    q_shard_off,
    dq_shard_off,
    range_select_bounds=None,
    striped=False,
    bound_min_sbuf=None,
    bound_max_sbuf=None,
):
    """
    Run backward core for all Q tiles against all local K tiles.

    q_buf/dy_buf/dq_buf are flat HBM buffers accessed via .ap() with shard offsets.

    Args:
        cfg (AttentionBwdConfig): Backward attention configuration.
        q_buf (nl.NkiTensor): Flat HBM buffer containing Q tiles.
        dy_buf (nl.NkiTensor): Flat HBM buffer containing dY tiles.
        exp_bias (nl.NkiTensor): [q_seq_tile_size, q_seq_n_tiles], Softmax exp bias (-LSE) in SBUF.
        dos (nl.NkiTensor): [q_seq_tile_size, q_seq_n_tiles], dy_o_sum in SBUF.
        dq_buf (nl.NkiTensor): Flat HBM buffer for dQ output.
        k_loaded (list): List of K tiles loaded in SBUF, one per d_head tile.
        v_loaded (list): List of V tiles loaded in SBUF, one per d_head tile.
        dk_red (list): Accumulator for dK reduction, one per d_head tile.
        dv_red (list): Accumulator for dV reduction, one per d_head tile.
        kernel_dtype: Kernel data type.
        mixed_dtype: Mixed precision data type.
        softmax_scale (float): Softmax scale.
        causal (bool): Whether to apply causal masking.
        q_seq_tile_size (int): Q sequence tile size.
        q_seq_n_tiles (int): Number of Q sequence tiles.
        k_seq_tile_size (int): K sequence tile size.
        k_seq_n_tiles (int): Number of K sequence tiles.
        d_head_n_tiles (int): Number of d_head tiles.
        d_head_tile_size (int): d_head tile size.
        seqlen (int): Sequence length.
        global_k_offset (int): Global K sequence offset.
        q_shard_off (int): Q shard offset into flat buffers.
        dq_shard_off (int): dQ shard offset into flat buffers.
        range_select_bounds (tuple): Optional (ub, lb) for range_select causal masking.
        striped (bool): Whether striped attention layout is used. When True, enables
            compile-time tile skipping even when range_select_bounds is set, since
            the local causal pattern (q_local >= k_local) is the same at every ring step.
    """
    for q_tile_idx in range(0, q_seq_n_tiles, cfg.q_tile_group_size):
        current_group = min(cfg.q_tile_group_size, q_seq_n_tiles - q_tile_idx)
        group_seqlen = q_seq_tile_size * current_group

        q_local, dy_local = load_q_dy(
            q_ref_hbm_tile=q_buf,
            dy_ref_hbm_tile=dy_buf,
            dtype=kernel_dtype,
            d_head_qk_n_tiles=d_head_n_tiles,
            d_head_v_n_tiles=d_head_n_tiles,
            d_head_qk_tile_size=d_head_tile_size,
            d_head_v_tile_size=d_head_tile_size,
            q_seq_tile_size=group_seqlen,
            seqlen_q=seqlen,
            offset_q=q_shard_off + q_tile_idx * q_seq_tile_size,
            offset_dy=q_shard_off + q_tile_idx * q_seq_tile_size,
        )

        trans_q_stride = d_head_tile_size * current_group
        trans_q_f = trans_q_stride * d_head_n_tiles
        tq = nl.ndarray((q_seq_tile_size, trans_q_f), dtype=kernel_dtype, buffer=nl.sbuf)
        tdy = nl.ndarray((q_seq_tile_size, trans_q_f), dtype=kernel_dtype, buffer=nl.sbuf)
        for d_tile_idx in range(d_head_n_tiles):
            tq_start = d_tile_idx * trans_q_stride
            tq_end = tq_start + d_head_tile_size * current_group
            transpose_tiles(q_local[d_tile_idx], tq[:, tq_start:tq_end], q_seq_tile_size)
            transpose_tiles(dy_local[d_tile_idx], tdy[:, tq_start:tq_end], q_seq_tile_size)

        dq_grp = ndarray((d_head_n_tiles,), (d_head_tile_size, group_seqlen), dtype=mixed_dtype, value=0.0)

        for k_tile_idx in range(k_seq_n_tiles):
            tr, any_r = get_required_tiles_mask(
                current_group,
                causal and (striped or (range_select_bounds is None and bound_max_sbuf is None)),
                q_tile_idx,
                q_seq_tile_size,
                global_k_offset,
                k_tile_idx,
                k_seq_tile_size,
                0,
            )
            if any_r:
                kl, vl = [], []
                for d_tile_idx in range(d_head_n_tiles):
                    kl.append(
                        k_loaded[d_tile_idx][:, k_tile_idx * k_seq_tile_size : (k_tile_idx + 1) * k_seq_tile_size]
                    )
                    vl.append(
                        v_loaded[d_tile_idx][:, k_tile_idx * k_seq_tile_size : (k_tile_idx + 1) * k_seq_tile_size]
                    )

                _flash_attn_bwd_core(
                    cfg,
                    q_local=q_local,
                    k_local=kl,
                    v_local=vl,
                    dy_local=dy_local,
                    dk_local_reduced=dk_red,
                    dv_local_reduced=dv_red,
                    dq_local=dq_grp,
                    softmax_exp_bias=exp_bias,
                    dy_o_sum=dos[:, q_tile_idx : q_tile_idx + current_group],
                    trans_q_local=tq,
                    trans_dy=tdy,
                    local_i_q_seq_tile=q_tile_idx,
                    local_i_k_seq_tile=k_tile_idx,
                    use_causal_mask=causal,
                    sliding_window=0,
                    tile_required=tr,
                    q_tile_group_size=current_group,
                    global_k_seq_offset=global_k_offset,
                    range_select_bounds=range_select_bounds,
                    bound_min_sbuf=bound_min_sbuf,
                    bound_max_sbuf=bound_max_sbuf,
                )

        dq_st_pat = [[seqlen, d_head_tile_size], [1, group_seqlen]]
        for d_tile_idx in range(d_head_n_tiles):
            nisa.dma_copy(
                dst=dq_buf.ap(
                    pattern=dq_st_pat,
                    offset=dq_shard_off + d_tile_idx * d_head_tile_size * seqlen + q_tile_idx * q_seq_tile_size,
                ),
                src=dq_grp[d_tile_idx],
            )


def _ring_bwd_impl(
    q_ref,
    k_ref,
    v_ref,
    o_ref,
    dy_ref,
    lse_ref,
    out_dq,
    out_dk,
    out_dv,
    sq,
    rq,
    sdy,
    rdy,
    slse,
    rlse,
    sdos,
    rdos,
    sdq,
    rdq,
    dq_scr0,
    dq_scr1,
    bid,
    hid,
    lnc,
    lnc_id,
    rgs,
    iota_nw,
    nw=1,
    causal=False,
    striped=False,
    mp=True,
    ss=None,
    bound_min=None,
    bound_max=None,
):
    """
    Inner implementation for one (batch, head) pair of ring attention backward.

    Performs the full ring communication loop: computes local backward pass (step 0),
    then circulates Q/dY/LSE/dy_o_sum through collective permute while accumulating
    dK/dV locally and reducing dQ across workers.

    Args:
        q_ref (nl.NkiTensor): [B, N, D, S], Query tensor in HBM.
        k_ref (nl.NkiTensor): [B, N, D, S], Key tensor in HBM.
        v_ref (nl.NkiTensor): [B, N, D, S], Value tensor in HBM.
        o_ref (nl.NkiTensor): [B, N, D, S], Forward output tensor in HBM.
        dy_ref (nl.NkiTensor): [B, N, D, S], Upstream gradient tensor in HBM.
        lse_ref (nl.NkiTensor): [B, N, 128, S//128], Log-sum-exp from forward pass.
        out_dq (nl.NkiTensor): Output dQ buffer in HBM.
        out_dk (nl.NkiTensor): Output dK buffer in HBM.
        out_dv (nl.NkiTensor): Output dV buffer in HBM.
        sq (nl.NkiTensor): Send buffer for Q.
        rq (nl.NkiTensor): Receive buffer for Q.
        sdy (nl.NkiTensor): Send buffer for dY.
        rdy (nl.NkiTensor): Receive buffer for dY.
        slse (nl.NkiTensor): Send buffer for LSE.
        rlse (nl.NkiTensor): Receive buffer for LSE.
        sdos (nl.NkiTensor): Send buffer for dy_o_sum.
        rdos (nl.NkiTensor): Receive buffer for dy_o_sum.
        sdq (nl.NkiTensor): Send buffer for dQ.
        rdq (nl.NkiTensor): Receive buffer for dQ.
        dq_scr0 (nl.NkiTensor): dQ scratch buffer 0.
        dq_scr1 (nl.NkiTensor): dQ scratch buffer 1.
        bid (int): Batch index.
        hid (int): Head index.
        lnc (int): LNC size.
        lnc_id (int): LNC core ID.
        rgs (list): Replica groups for collective communication.
        nw (int): Number of workers in the ring.
        causal (bool): Whether to apply causal masking.
        striped (bool): Whether to use striped attention layout.
        mp (bool): Whether to use mixed precision.
        ss (float): Softmax scale factor.
    """
    _, _, dh, sl = q_ref.shape
    kdt = q_ref.dtype
    mdt = nl.float32 if mp else kdt
    ss = ss or 1.0 / float(dh**0.5)
    qbs = dh * sl  # per-shard Q buffer size
    lbs = 128 * (sl // 128)  # per-shard LSE buffer size

    rg = ncc.ReplicaGroup(rgs)
    ch = 0

    # Sequence packing: when bound_min/bound_max are passed, route through
    # attention_bwd's proven use_sequence_packing path (fused PSUM->SBUF
    # range_select with bound_min_sbuf / bound_max_sbuf) instead of the
    # range_select_bounds path. The clamp that attention_bwd does once for
    # non-ring (min(bound_max, iota+1+cp_offset)) is re-done per ring step here
    # because cp_offset (encoded via no_diag) changes with my_rank vs recv_rank.
    is_seq_packed = bound_min is not None

    cfg = setup_config(
        q_ref,
        k_ref,
        v_ref,
        sinks_ref=None,
        mixed_precision=mp,
        softmax_scale=ss,
        use_sequence_packing=is_seq_packed,
    )
    qts = cfg.q_seq_tile_size
    qnt = cfg.q_seq_n_tiles
    dnt = cfg.d_head_qk_n_tiles
    dts = cfg.d_head_qk_tile_size
    kts = cfg.k_seq_tile_size
    knt = cfg.k_seq_n_tiles

    # Shard offsets into flat buffers
    q_so = lnc_id * qbs  # Q/dY/dQ shard offset
    l_so = lnc_id * lbs  # LSE/dos shard offset

    # Access patterns
    q_pat = [[sl, dts], [1, sl]]
    lse_pat = [[qnt, qts], [1, qnt]]
    dos_pat = [[qnt, qts], [1, qnt]]

    # Precompute softmax_exp_bias = -LSE
    seb = ndarray((1,), (qts, qnt), mdt)
    off_lse = bid * cfg.offset_lse_bs + hid * cfg.offset_lse_head
    nisa.dma_copy(dst=seb[0], src=lse_ref.ap(pattern=lse_pat, offset=off_lse))
    nisa.tensor_scalar(seb[0], seb[0], nl.multiply, -1.0)

    # Load sequence-packing bounds once (no rotation needed because the striped CP
    # invariant makes local bounds identical across ranks). Both shapes (qts, qnt)
    # in SBUF fp32.
    bound_min_sb = None
    bound_max_sb = None
    if is_seq_packed:
        # bound_min / bound_max have HBM shape (bs, seqlen_per_rank) fp32 — one row
        # per batch. Access pattern must match iota's layout so that SBUF element
        # (partition p, free g) corresponds to LOCAL Q position g*qts + p (group g,
        # row p within group). Pattern [[1, qts], [qts, qnt]]: partition stride=1
        # count=qts; free stride=qts count=qnt.
        bound_offset = bid * sl
        bound_pat = [[1, qts], [qts, qnt]]
        bound_min_sb = nl.ndarray((qts, qnt), dtype=nl.float32, buffer=nl.sbuf)
        bound_max_sb = nl.ndarray((qts, qnt), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=bound_min_sb, src=bound_min.ap(pattern=bound_pat, offset=bound_offset))
        nisa.dma_copy(dst=bound_max_sb, src=bound_max.ap(pattern=bound_pat, offset=bound_offset))

    # Precompute my_rank_sb once — constant across all ring steps, also needed by
    # step 0 when packing is on (to build the step-0 bounds tuple).
    my_rank_sb = None
    if causal:
        my_rank_sb = _load_rank_sb(
            iota_nw,
            ncc.collective_permute_implicit_current_processing_rank_id(
                iteration_id=0,
                channel_id=ch,
                replica_group=rg,
            ),
            qts,
        )

    # Compute dy_o_sum = rowsum(dO * O)
    dos_sb = ndarray((1,), (qts, qnt), dtype=mdt, value=0.0)
    off_q = bid * cfg.offset_q_bs + hid * cfg.offset_q_head

    for q_tile_idx in range(0, qnt, cfg.q_tile_group_size):
        current_group = min(cfg.q_tile_group_size, qnt - q_tile_idx)
        group_seqlen = qts * current_group
        dy_ls, _ = load_q_dy(
            q_ref_hbm_tile=dy_ref,
            dy_ref_hbm_tile=dy_ref,
            dtype=kdt,
            d_head_qk_n_tiles=dnt,
            d_head_v_n_tiles=dnt,
            d_head_qk_tile_size=dts,
            d_head_v_tile_size=dts,
            q_seq_tile_size=group_seqlen,
            seqlen_q=sl,
            offset_q=off_q + q_tile_idx * qts,
            offset_dy=off_q + q_tile_idx * qts,
        )
        o_l = ndarray((dnt,), (dts, group_seqlen), kdt)
        for d_tile_idx in range(dnt):
            nisa.dma_copy(
                dst=o_l[d_tile_idx],
                src=o_ref.ap(
                    pattern=[[sl, dts], [1, group_seqlen]],
                    offset=off_q + d_tile_idx * dts * sl + q_tile_idx * qts,
                ),
            )
        dy_t = ndarray((dnt,), (qts, dts * current_group), kdt)
        for d_tile_idx in range(dnt):
            transpose_tiles(dy_ls[d_tile_idx], dy_t[d_tile_idx], qts)
        dy_op = ndarray((current_group,), (qts, dnt), mdt)
        for d_tile_idx in range(dnt):
            compute_rowsum_single_tile(
                o_l[d_tile_idx],
                dy_t[d_tile_idx],
                dy_op,
                i_d_head_tile=d_tile_idx,
                q_seq_tile_size=qts,
                d_head_v_tile_size=dts,
                num_tiles=current_group,
            )
        for group_idx in range(current_group):
            nisa.tensor_reduce(dos_sb[0][:, q_tile_idx + group_idx], op=nl.add, data=dy_op[group_idx], axis=1)

    # Load local K, V
    k_ld, v_ld = load_kv(
        k_ref_hbm_tile=k_ref,
        v_ref_hbm_tile=v_ref,
        dtype=kdt,
        d_head_qk_n_tiles=dnt,
        d_head_v_n_tiles=dnt,
        d_head_qk_tile_size=dts,
        d_head_v_tile_size=dts,
        k_seq_tile_size=sl,
        seqlen_k=sl,
        offset_k=bid * cfg.offset_k_bs + hid * cfg.offset_k_head,
        offset_v=bid * cfg.offset_v_bs + hid * cfg.offset_v_head,
    )

    dk_r = ndarray((dnt,), (dts, sl), mdt, value=0.0)
    dv_r = ndarray((dnt,), (dts, sl), mdt, value=0.0)

    # Copy local Q, dY, LSE, dy_o_sum into send buffers
    for d_tile_idx in range(dnt):
        nisa.dma_copy(
            dst=sq.ap(pattern=q_pat, offset=q_so + d_tile_idx * dts * sl),
            src=q_ref.ap(pattern=q_pat, offset=off_q + d_tile_idx * dts * sl),
        )
        nisa.dma_copy(
            dst=sdy.ap(pattern=q_pat, offset=q_so + d_tile_idx * dts * sl),
            src=dy_ref.ap(pattern=q_pat, offset=off_q + d_tile_idx * dts * sl),
        )
    nisa.dma_copy(dst=slse.ap(pattern=lse_pat, offset=l_so), src=lse_ref.ap(pattern=lse_pat, offset=off_lse))
    nisa.dma_copy(dst=sdos.ap(pattern=dos_pat, offset=l_so), src=dos_sb[0])

    """
    Step 0: compute local Q against local K/V (gko = 0).
    Start permuting Q, dY, LSE, dos BEFORE compute — they're already in send buffers
    and compute only reads from them, so CC and compute can overlap.
    """
    _permute_all(sq, rq, sdy, rdy, slse, rlse, sdos, rdos, rg, ch, lnc)

    # Step 0: no range_select_bounds needed — gko=0 means affine_select handles causal
    # correctly, and this enables compile-time tile skipping for upper-triangle tiles.
    # EXCEPTION: when sequence packing is on, we route through the use_sequence_packing
    # path with bound_min_sbuf / bound_max_sbuf (bound_max clamped per step with the
    # striped causal rule). At step 0, my_rank == recv_rank => no_diag = 0 =>
    # causal_ub_exclusive = iota + 1, matching attention_bwd's non-ring packing path.
    step0_bound_max_sbuf = None
    if causal and is_seq_packed:
        step0_bound_max_sbuf = _build_bound_max_clamped(
            qts,
            qnt,
            bound_max_sb=bound_max_sb,
            striped=striped,
            recv_rank_sb=my_rank_sb if striped else None,
            my_rank_sb=my_rank_sb if striped else None,
            gko_sb=None,
        )
    _compute_step(
        cfg,
        sq,
        sdy,
        seb[0],
        dos_sb[0],
        dq_scr0,
        k_ld,
        v_ld,
        dk_r,
        dv_r,
        kdt,
        mdt,
        ss,
        causal,
        qts,
        qnt,
        kts,
        knt,
        dnt,
        dts,
        sl,
        0,
        q_so,
        q_so,
        range_select_bounds=None,
        striped=striped,
        bound_min_sbuf=bound_min_sb if is_seq_packed else None,
        bound_max_sbuf=step0_bound_max_sbuf,
    )

    # Copy dQ step0 into send_dq
    for d_tile_idx in range(dnt):
        nisa.dma_copy(
            dst=sdq.ap(pattern=q_pat, offset=q_so + d_tile_idx * dts * sl),
            src=dq_scr0.ap(pattern=q_pat, offset=q_so + d_tile_idx * dts * sl),
        )

    # First permute dQ
    if lnc > 1:
        nisa.core_barrier(sdq, (0, 1))
    ncc.collective_permute_implicit(
        srcs_by_channel=((sdq,),), dsts_by_channel=((rdq,),), replica_group=rg, channel_ids=(ch,)
    )

    # Ping-pong: after first permute, data is in recv buffers.
    # Alternate which buffer holds current data vs receives next data.
    cur_q, nxt_q = rq, sq
    cur_dy, nxt_dy = rdy, sdy
    cur_lse, nxt_lse = rlse, slse
    cur_dos, nxt_dos = rdos, sdos
    cur_dq_recv, cur_dq_send = rdq, sdq

    # my_rank_sb was precomputed above (before step 0) since step 0 may also need it.

    # Ring steps 1..nw-1
    for ring_step_idx in nl.sequential_range(1, nw):
        # Reload exp_bias and dy_o_sum from current (just-received) buffers
        step_eb = ndarray((1,), (qts, qnt), mdt)
        nisa.dma_copy(dst=step_eb[0], src=cur_lse.ap(pattern=lse_pat, offset=l_so))
        nisa.tensor_scalar(step_eb[0], step_eb[0], nl.multiply, -1.0)

        step_dos = ndarray((1,), (qts, qnt), dtype=mdt)
        nisa.dma_copy(dst=step_dos[0], src=cur_dos.ap(pattern=dos_pat, offset=l_so))

        dq_s = dq_scr1

        # Get the rank ID of the Q data we're currently processing
        recv_rank = ncc.collective_permute_implicit_current_processing_rank_id(
            iteration_id=ring_step_idx,
            channel_id=ch,
            replica_group=rg,
        )

        rs_bounds = None
        step_bound_max_sbuf = None
        if causal:
            recv_rank_sb = _load_rank_sb(iota_nw, recv_rank, qts)

            if striped:
                if is_seq_packed:
                    # Route through use_sequence_packing path: build bound_max_clamped
                    # with the striped causal rule (iota + 1 - no_diag).
                    step_bound_max_sbuf = _build_bound_max_clamped(
                        qts,
                        qnt,
                        bound_max_sb=bound_max_sb,
                        striped=True,
                        recv_rank_sb=recv_rank_sb,
                        my_rank_sb=my_rank_sb,
                    )
                else:
                    rs_bounds = _build_causal_bounds(
                        qts,
                        qnt,
                        striped=True,
                        recv_rank_sb=recv_rank_sb,
                        my_rank_sb=my_rank_sb,
                    )
            else:
                # gko = (recv_rank - my_rank) * sl
                gko_sb = nl.ndarray((qts, 1), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_scalar(gko_sb, recv_rank_sb, op0=nl.subtract, operand0=my_rank_sb)
                nisa.tensor_scalar(gko_sb, gko_sb, op0=nl.multiply, operand0=float(sl))
                if is_seq_packed:
                    step_bound_max_sbuf = _build_bound_max_clamped(
                        qts,
                        qnt,
                        bound_max_sb=bound_max_sb,
                        striped=False,
                        gko_sb=gko_sb,
                    )
                else:
                    rs_bounds = _build_causal_bounds(qts, qnt, striped=False, gko_sb=gko_sb)

        # No barrier needed: send buffer was written by CC, not NCs
        _permute_all(cur_q, nxt_q, cur_dy, nxt_dy, cur_lse, nxt_lse, cur_dos, nxt_dos, rg, ch, lnc, barrier=False)

        # Compute reads from current data buffer
        _compute_step(
            cfg,
            cur_q,
            cur_dy,
            step_eb[0],
            step_dos[0],
            dq_s,
            k_ld,
            v_ld,
            dk_r,
            dv_r,
            kdt,
            mdt,
            ss,
            causal,
            qts,
            qnt,
            kts,
            knt,
            dnt,
            dts,
            sl,
            0,
            q_so,
            q_so,
            range_select_bounds=rs_bounds,
            striped=striped,
            bound_min_sbuf=bound_min_sb if is_seq_packed else None,
            bound_max_sbuf=step_bound_max_sbuf,
        )

        # dQ reduction: combine local dq_s with incoming cur_dq_recv, write to cur_dq_send
        if lnc > 1:
            nisa.core_barrier(cur_dq_send, (0, 1))
        ncc.collective_permute_implicit_reduce(
            srcs0_by_channel=((dq_s,),),
            srcs1_by_channel=((cur_dq_recv,),),
            dsts_by_channel=((cur_dq_send,),),
            replica_group=rg,
            op=nl.add,
            channel_ids=(ch,),
        )

        # Ping-pong: swap buffer roles (no DMA needed)
        cur_q, nxt_q = nxt_q, cur_q
        cur_dy, nxt_dy = nxt_dy, cur_dy
        cur_lse, nxt_lse = nxt_lse, cur_lse
        cur_dos, nxt_dos = nxt_dos, cur_dos
        cur_dq_recv, cur_dq_send = cur_dq_send, cur_dq_recv

    # Write dK, dV
    off_k = bid * cfg.offset_k_bs + hid * cfg.offset_k_head
    dk_pat = [[sl, dts], [1, sl]]
    for d_tile_idx in range(dnt):
        nisa.dma_copy(dst=out_dk.ap(pattern=dk_pat, offset=off_k + d_tile_idx * dts * sl), src=dk_r[d_tile_idx])
        nisa.dma_copy(dst=out_dv.ap(pattern=dk_pat, offset=off_k + d_tile_idx * dts * sl), src=dv_r[d_tile_idx])

    # Write dQ from the buffer that holds the final reduced dQ after ping-pong
    for d_tile_idx in range(dnt):
        nisa.dma_copy(
            dst=out_dq.ap(pattern=dk_pat, offset=off_q + d_tile_idx * dts * sl),
            src=cur_dq_recv.ap(pattern=dk_pat, offset=q_so + d_tile_idx * dts * sl),
        )
