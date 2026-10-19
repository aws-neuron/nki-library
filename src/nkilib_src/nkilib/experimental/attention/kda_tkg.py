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

"""NKI kernel for Kimi Delta Attention (KDA) token generation (TKG).

Per request and head, with a per-channel gate g[dk] (all entries <= 0):

  kv   = (k * exp(g)) @ S                           # [dv]
  diff = v - kv                                     # [dv]
  S    = Diag(exp(g)) @ S + outer(k * beta, diff)   # [dk,dv]
  out  = scale * (q @ S)                            # [dv]
"""

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert

# Shorthand for the hardware tile-size table. Not a module, so it cannot be imported.
nlts = nl.tile_size

# L2-norm epsilon, folded into the rsqrt bias.
_L2_EPS = 1.0e-6

# PE row-tiling granularity. The NKI API exposes no constant for it, so like the
# other kernels in this repo it is a local literal.
_SBUF_QUADRANT_SIZE = 32
# Heads whose state/output transfers are coalesced into one DMA each. A multiple of
# the array-tiling group so the PE work below still fills the quadrants exactly.
_HEAD_GROUP = 8
# Heads processed per array-tiling pass: one per PE quadrant.
_TILE_GROUP = nlts.pmax // _SBUF_QUADRANT_SIZE

# nc_stream_shuffle: mask entry leaving a destination partition unmodified.
_SHUFFLE_NO_CHANGE = 255


@nki.jit
def kda_tkg(
    q: nl.NkiTensor,  # [B, dk, num_head]      raw query (pre-L2-norm)
    k: nl.NkiTensor,  # [B, dk, num_head]      raw key (pre-L2-norm)
    v: nl.NkiTensor,  # [B, num_head, dv]      raw value
    g: nl.NkiTensor,  # [B, dk, num_head]      raw per-channel gate (pre-activation)
    beta: nl.NkiTensor,  # [B, 1, num_head]    beta logits (pre-sigmoid)
    a_log: nl.NkiTensor,  # [1, num_head]      log-gate parameter (shared across batch)
    dt_bias: nl.NkiTensor,  # [dk, num_head]   gate bias (shared across batch)
    out: nl.NkiTensor,  # [B, num_head, dv]    attention output (written)
    S: nl.NkiTensor,  # [B, num_head, dk, dv]  recurrent state, IN/OUT (updated in place)
    scale: nl.NkiTensor,  # [1, 1]             query scale (1/sqrt(dk))
    lower_bound: nl.NkiTensor,  # [1, 1]       gate lower bound (in [-5, 0])
):
    """Kimi Delta Attention (KDA) token generation: output + new state.

    Summary:
        Advances the KDA recurrence by exactly one token for each of B requests in
        a single invocation.

    Dimensions:
        B:        Number of requests in the batch (unconstrained)
        num_head: Number of value heads
        dk:       Key/query head dimension, <= 128
        dv:       Value head dimension, <= 128

    Args:
        q (nl.NkiTensor):
            Query tensor in HBM for the current token, raw (pre-L2-norm, pre-scale).
            Shape: [B, dk, num_head].  Dtype: nl.float32
        k (nl.NkiTensor):
            Key tensor in HBM for the current token, raw (pre-L2-norm).
            Shape: [B, dk, num_head].  Dtype: nl.float32
        v (nl.NkiTensor):
            Value tensor in HBM for the current token.
            Shape: [B, num_head, dv].  Dtype: nl.float32
        g (nl.NkiTensor):
            Per-channel forget-gate tensor in HBM, raw (pre-activation; the kernel
            applies lower_bound * sigmoid(exp(a_log) * (g + dt_bias))).
            Shape: [B, dk, num_head].  Dtype: nl.float32
        beta (nl.NkiTensor):
            Delta-rule beta LOGITS in HBM (the kernel applies the sigmoid).
            Shape: [B, 1, num_head].  Dtype: nl.float32
        a_log (nl.NkiTensor):
            Per-head log-gate parameter in HBM; exp(a_log) scales the gate
            pre-activation. A model parameter, so it carries no batch axis.
            Shape: [1, num_head].  Dtype: nl.float32
        dt_bias (nl.NkiTensor):
            Per-head, per-CHANNEL gate bias in HBM, added to g before the sigmoid.
            A model parameter, so it carries no batch axis.
            Shape: [dk, num_head].  Dtype: nl.float32
        out (nl.NkiTensor):
            Caller-allocated destination for this token's attention output.
            Shape: [B, num_head, dv].  Dtype: nl.float32
        S (nl.NkiTensor):
            Recurrent state carried in from the previous token, in HBM. IN/OUT:
            each item's [dk, dv] slice is read to seed its step, then overwritten
            with the advanced state.
            Shape: [B, num_head, dk, dv].  Dtype: nl.float32
        scale (nl.NkiTensor):
            Query scale, normally 1/sqrt(dk); folded into the L2-normed query so
            the output needs no post-scale.
            Shape: [1, 1].  Dtype: nl.float32
        lower_bound (nl.NkiTensor):
            Gate lower bound, expected in [-5, 0]. The activated gate is
            lower_bound * sigmoid(...), hence <= 0, so exp(gate) is in (0, 1] and
            the decay can never amplify the state.
            Shape: [1, 1].  Dtype: nl.float32

    Returns:
        out (nl.NkiTensor):
            The caller-provided output tensor, filled with this token's attention
            output for every request in the batch.
            Shape: [B, num_head, dv].  Dtype: nl.float32
        S (nl.NkiTensor):
            The caller-provided state tensor, advanced in place by one token for
            every request.
            Shape: [B, num_head, dk, dv].  Dtype: nl.float32

    Pseudocode:
        for b in this core's requests:
            q, k, g = q[b], k[b], g[b]
            ss        = sum_dk(square([q | k]))
            q         = q * rsqrt(ss_q + eps) * scale
            k         = k * rsqrt(ss_k + eps)
            exp_g     = exp(lower_bound * sigmoid(exp(a_log) * (g + dt_bias)))
            k_decayed = k * exp_g
            k_beta    = k * sigmoid(beta)

            for each group of _HEAD_GROUP heads:
                S_g = S[b, group]

                for head in group:
                    kv[head]   = k_decayed[:, head] @ S_g[head]
                diff = v[group] - kv

                for head in group:
                    S_g[head] = S_g[head] * exp_g[:, head] + outer(k_beta[:, head], diff[head])

                for head in group:
                    out[b, head] = q[:, head].T @ S_g[head]

                S[b, group] = S_g
    """
    B, dk, num_head = q.shape
    dv = v.shape[-1]

    # Every per-item vector lives on a partition axis, so each must fit one tile.
    kernel_assert(num_head <= 128, f"num_head must be <= 128, got {num_head}")
    # Head-group coalescing and the PE quadrant tiling both assume full groups. A head
    # count that leaves a partial group needs the recurrence restructured, so reject it
    # rather than silently taking a degenerate path.
    kernel_assert(
        num_head % _HEAD_GROUP == 0,
        f"num_head must be a multiple of {_HEAD_GROUP}, got {num_head}",
    )
    kernel_assert(dk <= 128, f"dk must be <= 128, got {dk}")
    kernel_assert(dv <= 128, f"dv must be <= 128, got {dv}")
    kernel_assert(B >= 1, f"B must be at least 1, got {B}")

    # SPMD sharding over the BATCH: the preprocessing tile is one request's heads.
    grid_ndim = nl.program_ndim()
    if grid_ndim != 0:
        n_progs = nl.num_programs(axes=0)
        prog_id = nl.program_id(axis=0)
    else:
        n_progs = 1
        prog_id = 0

    # The last core absorbs the remainder, so B need not divide the core count. If
    # B < n_progs the low cores get an empty range and simply emit no work.
    batch_shard_size = B // n_progs
    b_start = prog_id * batch_shard_size
    b_end = B if prog_id == n_progs - 1 else b_start + batch_shard_size

    scale_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale_t, src=scale)
    lower_bound_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=lower_bound_t, src=lower_bound)
    # lower_bound multiplies a [dk, num_head] tile, so it needs dk partitions; the
    # value is constant along both axes, so one partition-broadcast covers it.
    lower_bound_bc = nl.broadcast_to(lower_bound_t, (dk, 1))

    # Column of ones for the L2-norm's partition-axis reduction. Built once.
    ones_col = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_col, 1.0)

    # Model parameters: no batch axis, so loaded once for the whole shard.
    dt_bias_in = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dt_bias_in, src=dt_bias)
    a_log_in = nl.ndarray((1, num_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_log_in, src=a_log)
    a_exp_row = nl.ndarray((1, num_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=a_exp_row, op=nl.exp, data=a_log_in)
    a_exp_bc = nl.broadcast_to(a_exp_row, (dk, num_head))

    for b in range(b_start, b_end):
        # One contiguous DMA per input: [dk, num_head] with dk on the partitions.
        q_in = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=q_in, src=q[b])
        k_in = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=k_in, src=k[b])
        g_in = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=g_in, src=g[b])
        beta_in = nl.ndarray((1, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(dst=beta_in, src=beta[b])
        # v is head-major, so the whole batch loads in one contiguous DMA. Heads are
        # redistributed to the array-tiling partitions on-chip (see the group loop).
        # Rounded up to a whole quadrant because the redistribution reads a full
        # quadrant; the pad is zeroed so that read never touches uninitialized data.
        v_all_rows = ((num_head + _SBUF_QUADRANT_SIZE - 1) // _SBUF_QUADRANT_SIZE) * _SBUF_QUADRANT_SIZE
        v_all = nl.ndarray((v_all_rows, dv), dtype=nl.float32, buffer=nl.sbuf)
        if v_all_rows != num_head:
            nisa.memset(dst=v_all[num_head:v_all_rows, 0:dv], value=0)
        nisa.dma_copy(dst=v_all[0:num_head, 0:dv], src=v[b, 0:num_head, 0:dv])

        # L2-norm. The sum over dk is a PARTITION reduction, so it is a matmul
        # against ones_col; q and k are packed side by side on the free axis so one
        # nc_matmul produces both [1, num_head] sums.
        qk_sq = nl.ndarray((dk, 2 * num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=qk_sq[0:dk, 0:num_head], op=nl.square, data=q_in)
        nisa.activation(dst=qk_sq[0:dk, num_head : 2 * num_head], op=nl.square, data=k_in)
        ss_row_p = nl.ndarray((1, 2 * num_head), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ss_row_p, stationary=ones_col, moving=qk_sq)
        ss_row = nl.ndarray((1, 2 * num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ss_row, src=ss_row_p, engine=nisa.scalar_engine)
        rnorm_row = nl.ndarray((1, 2 * num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rnorm_row, op=nl.rsqrt, data=ss_row, bias=_L2_EPS)

        # The query scale folds into the query's reciprocal norm, so out needs no
        # post-scale. Reciprocal norms vary along the FREE axis here, so they are
        # partition-broadcast and applied with tensor_tensor rather than
        # tensor_scalar. The prologue is Scalar-bound, so pin these to Vector.
        q_rnorm_scaled = nl.ndarray((1, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(
            dst=q_rnorm_scaled,
            data=rnorm_row[0:1, 0:num_head],
            op0=nl.multiply,
            operand0=scale_t,
            engine=nisa.vector_engine,
        )
        q_rnorm_bc = nl.broadcast_to(q_rnorm_scaled, (dk, num_head))
        k_rnorm_bc = nl.broadcast_to(rnorm_row[0:1, num_head : 2 * num_head], (dk, num_head))
        q_normed = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=q_normed, data1=q_in, data2=q_rnorm_bc, op=nl.multiply)
        k_normed = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_normed, data1=k_in, data2=k_rnorm_bc, op=nl.multiply)

        # g_act = lower_bound * sigmoid(exp(a_log) * (g_raw + dt_bias)).
        g_shift = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=g_shift, data1=g_in, data2=dt_bias_in, op=nl.add)
        g_scaled = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=g_scaled, data1=g_shift, data2=a_exp_bc, op=nl.multiply)
        g_sig = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=g_sig, op=nl.sigmoid, data=g_scaled)
        # g <= 0, so exp(g) is in (0, 1] -- no overflow.
        g_act = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_scalar(dst=g_act, data=g_sig, op0=nl.multiply, operand0=lower_bound_bc, engine=nisa.vector_engine)
        exp_g_all = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=exp_g_all, op=nl.exp, data=g_act)

        beta_row = nl.ndarray((1, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=beta_row, op=nl.sigmoid, data=beta_in)
        beta_bc = nl.broadcast_to(beta_row, (dk, num_head))

        k_decayed_all = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_decayed_all, data1=k_normed, data2=exp_g_all, op=nl.multiply)
        k_beta_all = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_beta_all, data1=k_normed, data2=beta_bc, op=nl.multiply)

        # Outer pass coalesces transfers; the inner pass does one array-tiling group.
        for head_group_idx in range(0, num_head, _HEAD_GROUP):
            group_size = min(_HEAD_GROUP, num_head - head_group_idx)

            # The whole coalescing group's states arrive and leave in one DMA each.
            S_g = nl.ndarray((dk, group_size, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=S_g, src=S[b, head_group_idx : head_group_idx + group_size].permute((1, 0, 2)))
            S_g_out = nl.ndarray((dk, group_size, dv), dtype=nl.float32, buffer=nl.sbuf)
            out_g = nl.ndarray((group_size, dv), dtype=nl.float32, buffer=nl.sbuf)

            for tile_base in range(0, group_size, _TILE_GROUP):
                tile_n = min(_TILE_GROUP, group_size - tile_base)
                tile_head_idx = head_group_idx + tile_base

                # One tensor_copy replication to place data at the array tiling offsets before
                # transpose. Note: trying to AP broadcast errors with 2D free on input.
                tile_rows = tile_n * _SBUF_QUADRANT_SIZE
                k_beta_broadcasted = nl.ndarray((dk, tile_rows), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(
                    dst=k_beta_broadcasted,
                    src=k_beta_all[0:dk, tile_head_idx : tile_head_idx + tile_n]
                    .expand_dim(2)
                    .broadcast(2, _SBUF_QUADRANT_SIZE),
                    engine=nisa.scalar_engine,
                )
                k_beta_tile_t_psum = nl.ndarray((tile_rows, dk), dtype=nl.float32, buffer=nl.psum)
                nisa.nc_transpose(dst=k_beta_tile_t_psum, data=k_beta_broadcasted, engine=nisa.tensor_engine)
                k_beta_rows = nl.ndarray((tile_rows, dk), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_copy(dst=k_beta_rows, src=k_beta_tile_t_psum, engine=nisa.scalar_engine)

                # Move each head's v row from its staging partition to its array-tiling
                # partition. nc_stream_shuffle pins the src/dst START partitions to a
                # quadrant boundary, so the source is the whole enclosing quadrant and the
                # mask picks the head within it -- the head index never enters src_start.
                v_tile = nl.ndarray((nlts.pmax, dv), dtype=nl.float32, buffer=nl.sbuf)
                for t in range(tile_n):
                    head_idx = tile_head_idx + t
                    src_quadrant = (head_idx // _SBUF_QUADRANT_SIZE) * _SBUF_QUADRANT_SIZE
                    shuffle_mask = [_SHUFFLE_NO_CHANGE] * _SBUF_QUADRANT_SIZE
                    shuffle_mask[0] = head_idx - src_quadrant
                    nisa.nc_stream_shuffle(
                        dst=v_tile[t * _SBUF_QUADRANT_SIZE : t * _SBUF_QUADRANT_SIZE + 1, 0:dv],
                        src=v_all[src_quadrant : src_quadrant + _SBUF_QUADRANT_SIZE, 0:dv],
                        shuffle_mask=shuffle_mask,
                    )

                # Only tile_n of 128 partitions are written by the column tiled matmuls below.
                # nisa.memset rejects psum destinations, so zero it via a multiply-by-0; the
                # uint32 view keeps garbage bit patterns from becoming NaN as an fp multiply would.
                kv_tile_psum = nl.ndarray((nlts.pmax, dv), dtype=nl.float32, buffer=nl.psum)
                nisa.tensor_scalar(
                    dst=kv_tile_psum.reinterpret_cast(nl.uint32),
                    data=kv_tile_psum.reinterpret_cast(nl.uint32),
                    op0=nl.multiply,
                    operand0=0,
                    engine=nisa.scalar_engine,
                )
                # Column tiled matmul between kv = k_exp @ S
                for t in range(tile_n):
                    nisa.nc_matmul(
                        dst=kv_tile_psum[t * _SBUF_QUADRANT_SIZE : t * _SBUF_QUADRANT_SIZE + 1, 0:dv],
                        stationary=k_decayed_all[:, tile_head_idx + t : tile_head_idx + t + 1],
                        moving=S_g[:, tile_base + t],
                        tile_size=(nlts.pmax, _SBUF_QUADRANT_SIZE),
                        tile_position=(0, t * _SBUF_QUADRANT_SIZE),
                    )

                # Single eviction for the whole tile enabled by the memset above.
                diff = nl.ndarray((nlts.pmax, dv), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=diff, data1=v_tile, data2=kv_tile_psum, op=nl.subtract)

                # Row tiled matmul for outer product
                for t in range(tile_n):
                    head_idx = tile_head_idx + t
                    row_tile = t * _SBUF_QUADRANT_SIZE

                    S_head = S_g[:, tile_base + t]
                    exp_g = exp_g_all[:, head_idx : head_idx + 1]
                    k_beta_row = k_beta_rows[row_tile : row_tile + 1, 0:dk]

                    # S_updated = Diag(exp_g) @ S_head + outer(k*beta, diff).

                    # The outer product is a contraction-1 nc_matmul against the tile's
                    # pre-transposed [1, dk] row. Performs outer(k_beta, diff).
                    outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        dst=outer_psum,
                        stationary=k_beta_row,
                        moving=diff[row_tile : row_tile + 1, 0:dv],
                        tile_size=(_SBUF_QUADRANT_SIZE, nlts.pmax),
                        tile_position=(row_tile, 0),
                    )
                    # The decay multiply and the additive update are fused on evict.
                    # Performs Diag(exp_g) @ S_head + outer
                    nisa.scalar_tensor_tensor(
                        dst=S_g_out[:, tile_base + t],
                        data=S_head,
                        op0=nl.multiply,
                        operand0=exp_g,
                        op1=nl.add,
                        operand1=outer_psum,
                    )

                # Column tiled matmul for out = q.T @ S_updated
                out_psum = nl.ndarray((nlts.pmax, dv), dtype=nl.float32, buffer=nl.psum)
                out_temp = nl.ndarray((nlts.pmax, dv), dtype=nl.float32, buffer=nl.sbuf)
                for t in range(tile_n):
                    head_idx = tile_head_idx + t
                    col_tile = t * _SBUF_QUADRANT_SIZE
                    nisa.nc_matmul(
                        dst=out_psum[col_tile : col_tile + 1, 0:dv],
                        stationary=q_normed[:, head_idx : head_idx + 1],
                        moving=S_g_out[:, tile_base + t],
                        tile_size=(nlts.pmax, _SBUF_QUADRANT_SIZE),
                        tile_position=(0, col_tile),
                    )
                    nisa.tensor_copy(
                        dst=out_temp[col_tile : col_tile + 1, 0:dv],
                        src=out_psum[col_tile : col_tile + 1, 0:dv],
                        engine=nisa.scalar_engine,
                    )
                    # SBUF->SBUF DMA copy to place output in correct partition
                    nisa.dma_copy(
                        dst=out_g[tile_base + t : tile_base + t + 1, 0:dv],
                        src=out_temp[col_tile : col_tile + 1, 0:dv],
                    )

            # Grouped writeback: the advanced states and the outputs each leave in one DMA.
            nisa.dma_copy(dst=S[b, head_group_idx : head_group_idx + group_size].permute((1, 0, 2)), src=S_g_out)
            nisa.dma_copy(dst=out[b, head_group_idx : head_group_idx + group_size, 0:dv], src=out_g)

    return out, S
