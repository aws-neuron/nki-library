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

"""NKI kernel for a full Kimi Delta Attention (KDA) block, token generation (TKG).

Spans hidden states in to attention output out: input norm, the fused input
projections, the causal short convolutions, the low-rank forget gate, the KDA
recurrence, the gated output norm, and the output projection. The caller keeps the TP
collective and the residual add, so they can be fused with each other.

Per request and head, with a per-channel gate g[dk] (all entries <= 0):

  kv   = (k * exp(g)) @ S                           # [dv]
  diff = v - kv                                     # [dv]
  S    = Diag(exp(g)) @ S + outer(k * beta, diff)   # [dk,dv]
  o    = scale * (q @ S)                            # [dv]
"""

import nki
import nki.isa as nisa
import nki.language as nl
from nki.language import tile_size as nlts

from ...core.output_projection.output_projection_tkg import output_projection_tkg
from ...core.utils.allocator import sizeinbytes
from ...core.utils.common_types import NormType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_program_sharding_info
from ..input_projection.input_projection_tkg import input_projection_tkg

# PE row-tiling granularity. The NKI API exposes no constant for it, so like the
# other kernels in this repo it is a local literal.
_SBUF_QUADRANT_SIZE = 32
# Heads processed per array-tiling pass: one per PE quadrant.
_TILE_GROUP = nlts.pmax // _SBUF_QUADRANT_SIZE
# Heads whose state/output transfers are coalesced into one DMA each. A multiple of
# the array-tiling group so the PE work still fills the quadrants exactly.
_HEAD_GROUP = 8

# nc_stream_shuffle: mask entry leaving a destination partition unmodified.
_SHUFFLE_NO_CHANGE = 255

# Short-convolution streams (q, k, v), in the order they are stacked in conv_w.
_CONV_STREAMS = 3


@nki.jit
def kda_block_tkg(
    hidden: nl.NkiTensor,  # [B, n, H]              block input, PRE input-norm
    gamma_in: nl.NkiTensor,  # [1, H]               input RMSNorm weight
    W_in: nl.NkiTensor,  # [H, I]                   fused q|k|v|g_a|beta|f_a projection
    W_fb: nl.NkiTensor,  # [dk, dk * num_head]      forget-gate up-projection (head-outer cols)
    W_gb: nl.NkiTensor,  # [dk, dv * num_head]      output-gate up-projection (channel-outer cols)
    conv_w: nl.NkiTensor,  # [3, K, dk * num_head]  causal conv taps for q, k, v
    A_log: nl.NkiTensor,  # [1, num_head]           log-gate parameter
    dt_bias: nl.NkiTensor,  # [dk, num_head]        gate bias
    gamma_o: nl.NkiTensor,  # [1, dv]               output RMSNorm weight (shared across heads)
    W_o: nl.NkiTensor,  # [dv * num_head, H]        output projection
    out: nl.NkiTensor,  # [B, n, H]                 output projection partial (written)
    S: nl.NkiTensor,  # [B, num_head, dk, dv]       recurrent state, IN/OUT
    conv_state: nl.NkiTensor,  # [B, 3, K - 1, dk * num_head]  conv windows, IN/OUT
    scale: nl.NkiTensor,  # [1, 1]                  query scale (1/sqrt(dk))
    lower_bound: nl.NkiTensor,  # [1, 1]            gate lower bound (in [-5, 0])
    rms_norm_eps: float = 1.0e-5,  #                epsilon for BOTH RMSNorms
    l2_eps: float = 1.0e-6,  #                       L2-norm epsilon, folded into the rsqrt bias.
):
    """Kimi Delta Attention block for token generation: hidden states in, attention out.

    Summary:
        Advances the KDA recurrence by n tokens for each of B requests, wrapping the
        recurrence in the whole surrounding block so no intermediate leaves SBUF: input
        RMSNorm, the fused input projections, the causal short convolutions with SiLU,
        the low-rank forget gate, the gated output RMSNorm, and the output projection.

    Dimensions:
        B:        Number of requests in the batch (unconstrained)
        n:        Active tokens per request. Must be 1 today; the axis exists so a
                  speculative verify/replay window can be added without an API change.
        H:        Model hidden size
        num_head: Number of value heads held by THIS rank (num_head_global // TP).
                  Must be a multiple of _HEAD_GROUP, which caps TP at 12 for Kimi K3
                  (96 heads): TP in {1, 2, 3, 4, 6, 12}.
        dk:       Key/query head dimension, <= 128
        dv:       Value head dimension, <= 128. Equal to dk for KDA.
        K:        Short-convolution kernel size (4 for Kimi K3)
        I:        Fused input projection width, 2 * dk * num_head  (q, k)
                  + dv * num_head  (v)  + num_head  (beta)  + 2 * dk  (f_a, g_a).
                  The output gate is low-rank via g_a -> W_gb, not a dv*num_head block:
                  full rank would add 7168 x 12288 = 176 MB per layer in bf16.

    Tensor parallelism:
        KDA has no cross-head reduction, so every stage up to and including the output
        RMSNorm is local to the heads this rank owns: W_in, W_fb, conv_w, A_log, dt_bias
        and W_o are all column-sharded on the head axis, and num_head shrinks with TP.
        Only W_o is contraction-sharded, so ``out`` is a PARTIAL that the caller must
        reduce (all-reduce, or reduce-scatter if the consumer wants sharded activations).
        The caller also owns the residual add, so the two can be fused. The f_a columns
        of W_in are the exception: their output is a single dk-wide latent shared by all
        heads, so that block is replicated rather than sharded. gamma_in and gamma_o are
        replicated.

    Weight dtype:
        W_in, W_fb, conv_w and W_o default to bf16, matching the model's own dtype; fp8
        and MX variants are the intended future options. Weights are never fp32 in
        practice, and for W_in that is not merely a bandwidth preference:
        input_projection_tkg stages the fused weight as 16 x min(I, 4096) elements per
        partition, which is 256 KB at fp32 and exceeds the 192 KB partition budget for
        any real KDA width. Every staging tile below takes its dtype from the weight it
        holds, because DMA cannot convert dtypes in flight. Activations, the recurrent
        state, and the gate parameters (A_log, dt_bias) stay fp32.

    Args:
        hidden (nl.NkiTensor):
            Block input in HBM, before the input RMSNorm (the kernel applies it, so the
            residual the caller adds back is this tensor).
            Shape: [B, n, H].  Dtype: nl.float32
        gamma_in (nl.NkiTensor):
            Input RMSNorm weight. Replicated across ranks.
            Shape: [1, H].  Dtype: nl.float32
        W_in (nl.NkiTensor):
            Fused input projection, stored contraction-first so no runtime transpose is
            needed. Columns are this rank's shard in the order
            q | k | v | g_a | beta | f_a, with widths dk*num_head, dk*num_head,
            dv*num_head, dk, num_head, dk.

            The column order WITHIN each per-head block selects the layout the single
            transpose below produces, so it is chosen per block:
              - q, k channel-outer (d*num_head + h) -> transposes to [dk, BxS, num_head]
              - v     head-outer    (h*dv + d)       -> transposes to [num_head, BxS, dv]
            beta needs no transpose. f_a and g_a are the two low-rank gate latents: both
            are dk wide with no head axis, and each is expanded by its own up-projection
            (W_fb, W_gb). They ride here rather than in their own weights so they see the
            same normed hidden without a second norm; both are replicated across ranks,
            unlike the rest.
            Shape: [H, I].  Dtype: nl.bfloat16 (see the weight dtype note below)
        W_fb (nl.NkiTensor):
            Forget-gate up-projection, latent to this rank's per-head channels. Columns
            are HEAD-outer (h*dk + d), unlike W_in's q/k blocks: that makes each head's
            dk columns contiguous, so one matmul per head lands the result with dk
            already on the partition axis and the stage-5 transpose is not needed here.
            Shape: [dk, dk * num_head].  Dtype: nl.bfloat16
        W_gb (nl.NkiTensor):
            Output-gate up-projection, latent to this rank's per-head channels. Columns
            are CHANNEL-outer (d*num_head + h), unlike W_fb: the gated norm consumes the
            gate head-major, so tiling the matmul over dv with num_head contiguous lands
            [num_head, BxS] per channel directly, with no transpose.
            Shape: [dk, dv * num_head].  Dtype: nl.bfloat16
        conv_w (nl.NkiTensor):
            Depthwise causal convolution taps for the q, k and v streams, one filter per
            channel. Tap 0 is the oldest position in the window. Tap-major so that each
            tap is a contiguous dk*num_head run, loadable as one [dk, num_head] tile.
            Each stream's channel order matches the W_in block it filters, so streams
            0/1 are channel-outer and stream 2 is head-outer.
            Shape: [3, K, dk * num_head].  Dtype: nl.bfloat16
        A_log (nl.NkiTensor):
            Per-head log-gate parameter; exp(A_log) scales the gate pre-activation. A
            model parameter, so it carries no batch axis.
            Shape: [1, num_head].  Dtype: nl.float32
        dt_bias (nl.NkiTensor):
            Per-head, per-CHANNEL gate bias, added before the sigmoid. A model
            parameter, so it carries no batch axis.
            Shape: [dk, num_head].  Dtype: nl.float32
        gamma_o (nl.NkiTensor):
            Output RMSNorm weight, applied per head over dv and shared across heads.
            Shape: [1, dv].  Dtype: nl.float32
        W_o (nl.NkiTensor):
            Output projection, stored contraction-first (dv * num_head on the partition
            axis) so no runtime transpose is needed.
            Shape: [dv * num_head, H].  Dtype: nl.bfloat16
        out (nl.NkiTensor):
            Caller-allocated destination for this rank's output projection PARTIAL. Not
            a usable activation until the caller reduces it across the TP group; the
            residual it is added against is ``hidden``.
            Shape: [B, n, H].  Dtype: nl.float32
        S (nl.NkiTensor):
            Recurrent state carried in from the previous token, in HBM. IN/OUT: each
            head's [dk, dv] slice is read to seed its step, then overwritten with the
            advanced state.
            Shape: [B, num_head, dk, dv].  Dtype: nl.float32
        conv_state (nl.NkiTensor):
            The last K-1 PRE-convolution projected inputs for each of the q, k and v
            streams. IN/OUT: read as the left part of this step's convolution window,
            then overwritten with the window ending at the newest token. Pre-convolution
            is deliberate -- it is what the filter needs, and it is also the payload a
            future speculative replay would cache.
            Shape: [B, 3, dk * num_head, K - 1].  Dtype: nl.float32
        scale (nl.NkiTensor):
            Query scale, normally 1/sqrt(dk); folded into the L2-normed query so the
            output needs no post-scale.
            Shape: [1, 1].  Dtype: nl.float32
        rms_norm_eps (float):
            Epsilon for the input RMSNorm and the gated output RMSNorm. The reference
            implementation drives both from a single ``config.rms_norm_eps`` (1e-5 for
            Kimi K3), so one value covers them.
        lower_bound (nl.NkiTensor):
            Gate lower bound, expected in [-5, 0] (-5 for Kimi K3). The activated gate
            is lower_bound * sigmoid(...), hence <= 0, so exp(gate) is in (0, 1] and the
            decay can never amplify the state. Also what bounds the cumulative decay
            range that the chunkwise prefill kernel depends on.
            Shape: [1, 1].  Dtype: nl.float32

    Returns:
        out (nl.NkiTensor):
            The caller-provided output tensor, filled with this rank's output projection
            partial for every request in the batch. Requires a TP reduction before use.
            Shape: [B, n, H].  Dtype: nl.float32
        S (nl.NkiTensor):
            The caller-provided state tensor, advanced in place by n tokens for every
            request.
            Shape: [B, num_head, dk, dv].  Dtype: nl.float32
        conv_state (nl.NkiTensor):
            The caller-provided convolution windows, advanced in place by n tokens.
            Shape: [B, 3, dk * num_head, K - 1].  Dtype: nl.float32

    Pseudocode:
        for b in this core's requests:
            h = rmsnorm(hidden[b], rms_norm_eps) * gamma_in

            q, k, v, g_out, beta, f_a = split(h @ W_in)
            f_gate                    = f_a @ W_fb

            for stream, x in [(q, q), (k, k), (v, v)]:
                window   = concat(conv_state[b, stream], x)
                x        = silu(sum_K(window * conv_w[stream]))
                conv_state[b, stream] = window[-(K - 1):]

            ss        = sum_dk(square([q | k]))
            q         = q * rsqrt(ss_q + eps) * scale
            k         = k * rsqrt(ss_k + eps)
            exp_g     = exp(lower_bound * sigmoid(exp(A_log) * (f_gate + dt_bias)))
            k_decayed = k * exp_g
            k_beta    = k * sigmoid(beta)

            for each group of _HEAD_GROUP heads:
                S_g = S[b, group]

                for head in group:
                    kv[head] = k_decayed[:, head] @ S_g[head]
                diff = v[group] - kv

                for head in group:
                    S_g[head] = S_g[head] * exp_g[:, head] + outer(k_beta[:, head], diff[head])

                for head in group:
                    o[head] = q[:, head].T @ S_g[head]

                S[b, group] = S_g

            o          = rmsnorm_dv(o, rms_norm_eps) * gamma_o * sigmoid(g_out)
            out[b]     = o @ W_o
    """
    B, n, H = hidden.shape
    num_head = A_log.shape[-1]
    dk = dt_bias.shape[0]
    dv = S.shape[-1]
    conv_taps = conv_w.shape[1]

    # Every per-item vector lives on a partition axis, so each must fit one tile.
    kernel_assert(num_head <= 128, f"num_head must be <= 128, got {num_head}")
    # Head-group coalescing and the PE quadrant tiling both assume full groups. A head
    # count leaving a partial group needs the recurrence restructured, so reject it
    # rather than silently taking a degenerate path.
    kernel_assert(
        num_head % _HEAD_GROUP == 0,
        f"num_head must be a multiple of {_HEAD_GROUP}, got {num_head}",
    )
    kernel_assert(dk <= 128, f"dk must be <= 128, got {dk}")
    kernel_assert(dv <= 128, f"dv must be <= 128, got {dv}")
    # All three conv streams share one rectangular conv_w / conv_state, which needs the
    # q/k width (dk*num_head) to equal the v width (dv*num_head). True for KDA.
    kernel_assert(dv == dk, f"dv must equal dk, got dv={dv}, dk={dk}")
    kernel_assert(B >= 1, f"B must be at least 1, got {B}")
    # The axis is in the signature so a speculative verify/replay window can be added
    # later without an API change; the recurrence below is still single-token.
    kernel_assert(n == 1, f"n must be 1, got {n}")
    kernel_assert(conv_taps >= 2, f"conv kernel size must be >= 2, got {conv_taps}")
    kernel_assert(
        conv_state.shape[2] == conv_taps - 1,
        f"conv_state must hold K-1={conv_taps - 1} positions, got {conv_state.shape[2]}",
    )
    kernel_assert(
        conv_state.shape[1] == _CONV_STREAMS,
        f"conv_state must carry {_CONV_STREAMS} streams (Q, K, V), got {conv_state.shape[1]}",
    )
    kernel_assert(
        conv_w.shape[0] == _CONV_STREAMS,
        f"conv_w must carry {_CONV_STREAMS} streams (Q, K, V), got {conv_w.shape[0]}",
    )
    kernel_assert(
        W_in.shape[0] == H and W_o.shape[-1] == H,
        f"projection weights must contract over H={H}, got {W_in.shape[0]} and {W_o.shape[-1]}",
    )
    i_qk = dk * num_head
    i_v = dv * num_head
    kernel_assert(
        W_in.shape[-1] == 2 * i_qk + i_v + num_head + 2 * dk,
        f"W_in width must be 2*dk*num_head + dv*num_head + num_head + 2*dk, got {W_in.shape[-1]}",
    )

    BxS = B * n

    # ===== Stages 1-2: input RMSNorm + fused input projection =====
    # Output stays in SBUF as [BxS, I]: BxS on the partition axis, every channel on the
    # free axis. output_layout only shapes an HBM destination, so it is irrelevant here.
    proj_sb = input_projection_tkg(
        hidden=hidden,
        proj_w=W_in,
        norm_w=gamma_in,
        norm_type=NormType.RMS_NORM,
        eps=rms_norm_eps,
        output_in_sbuf=True,
    )
    proj_sb = proj_sb.reshape((BxS, W_in.shape[-1]))

    # ===== Stage 5: re-layout =====
    # Each tensor's channels move from the free axis onto partitions with one transpose.
    # Which layout comes out is set by that block's column order in W_in, not by anything
    # here: channel-outer transposes to [dk, BxS, num_head], head-outer to
    # [num_head, BxS, dv]. beta is already [BxS, num_head] and needs no transpose; f_a is
    # a shared latent with no head axis and is consumed by W_fb as-is.
    off_q, off_k = 0, i_qk
    off_v, off_ga = 2 * i_qk, 2 * i_qk + i_v
    off_beta = off_ga + dk
    off_fa = off_beta + num_head

    q_sb = nl.ndarray((dk, BxS, num_head), dtype=hidden.dtype, buffer=nl.sbuf, name="q_cm")
    k_sb = nl.ndarray((dk, BxS, num_head), dtype=hidden.dtype, buffer=nl.sbuf, name="k_cm")
    v_sb = nl.ndarray((num_head, BxS, dv), dtype=hidden.dtype, buffer=nl.sbuf, name="v_hm")
    g_out_sb = nl.ndarray((num_head, BxS, dv), dtype=hidden.dtype, buffer=nl.sbuf, name="g_out_hm")

    # One PE transpose per tile, packed into as few PSUM tiles as capacity allows so a
    # single eviction covers many transposes; evictions alternate Scalar/Vector so the
    # two engines overlap. A tile is [BxS, tile_size] and lands as [tile_size, BxS].
    # PSUM accesses must be psum_min_align bytes aligned, and a transpose's PSUM dst has
    # to share the input's dtype, so at bf16 a BxS of 1 would put chunk slots 2 bytes
    # apart. Pad each slot up to the alignment and skip the padding on eviction.
    slot_align = max(1, nlts.psum_min_align // sizeinbytes(proj_sb.dtype))
    psum_slot = ((BxS + slot_align - 1) // slot_align) * slot_align
    tiles_per_psum = max(1, nlts.psum_fmax // psum_slot)
    chunk_index = 0
    for dst, off, width, tile_size, num_tiles in (
        (q_sb, off_q, i_qk, dk, num_head),
        (k_sb, off_k, i_qk, dk, num_head),
        (v_sb, off_v, i_v, num_head, dv),
    ):
        src = proj_sb[:, off : off + width].reshape_dim(1, [tile_size, num_tiles])
        # permute to tile-major so each PSUM chunk evicts as one contiguous range
        dst_tile_major = dst.permute(dims=[0, 2, 1])
        for chunk_start in range(0, num_tiles, tiles_per_psum):
            chunk = min(tiles_per_psum, num_tiles - chunk_start)
            tp_psum = nl.ndarray((tile_size, chunk * psum_slot), dtype=proj_sb.dtype, buffer=nl.psum)
            for i in range(chunk):
                nisa.nc_transpose(
                    dst=tp_psum[0:tile_size, i * psum_slot : i * psum_slot + BxS],
                    data=src.slice(dim=2, start=chunk_start + i, end=chunk_start + i + 1).squeeze_dim(dim=2),
                )
            nisa.tensor_copy(
                dst=dst_tile_major.slice(dim=1, start=chunk_start, end=chunk_start + chunk),
                src=tp_psum.reshape_dim(1, [chunk, psum_slot]).slice(dim=2, start=0, end=BxS),
                engine=nisa.scalar_engine if chunk_index % 2 == 0 else nisa.vector_engine,
            )
            chunk_index += 1

    beta_sb = proj_sb[:, off_beta : off_beta + num_head]
    f_a_sb = proj_sb[:, off_fa : off_fa + dk]
    g_a_sb = proj_sb[:, off_ga : off_ga + dk]

    # ===== Stage 3: low-rank forget gate =====
    # f_gate = f_a @ W_fb, one matmul per head. W_fb's columns are head-outer, so a head's
    # dk columns are contiguous and serve as the stationary operand directly; the result
    # lands as [dk, BxS] with dk already on partitions, so no transpose is needed here.
    f_a_t_psum = nl.ndarray((dk, BxS), dtype=proj_sb.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=f_a_t_psum, data=f_a_sb)
    f_a_t = nl.ndarray((dk, BxS), dtype=W_fb.dtype, buffer=nl.sbuf, name="f_a_t")
    nisa.tensor_copy(dst=f_a_t, src=f_a_t_psum, engine=nisa.scalar_engine)

    w_fb_sb = nl.ndarray((dk, i_qk), dtype=W_fb.dtype, buffer=nl.sbuf, name="w_fb")
    nisa.dma_copy(dst=w_fb_sb, src=W_fb)

    g_sb = nl.ndarray((dk, BxS, num_head), dtype=nl.float32, buffer=nl.sbuf, name="g_cm")
    g_head_major = g_sb.permute(dims=[0, 2, 1])
    heads_per_psum = max(1, nlts.psum_fmax // BxS)
    for head_start in range(0, num_head, heads_per_psum):
        head_n = min(heads_per_psum, num_head - head_start)
        gate_psum = nl.ndarray((dk, head_n * BxS), dtype=nl.float32, buffer=nl.psum)
        for i in range(head_n):
            head_idx = head_start + i
            nisa.nc_matmul(
                dst=gate_psum[0:dk, i * BxS : (i + 1) * BxS],
                stationary=w_fb_sb[0:dk, head_idx * dk : (head_idx + 1) * dk],
                moving=f_a_t,
            )
        nisa.tensor_copy(
            dst=g_head_major.slice(dim=1, start=head_start, end=head_start + head_n),
            src=gate_psum.reshape_dim(1, [head_n, BxS]),
            engine=nisa.vector_engine,
        )

    # ===== Stage 3b: low-rank output gate =====
    # g_out = g_a @ W_gb. The gated norm wants the gate HEAD-major, so this tiles over dv
    # instead of over heads: W_gb's columns are channel-outer, so for a fixed channel d the
    # num_head columns are contiguous and one matmul yields [num_head, BxS] directly. The
    # PSUM stays fp32 here (it is a matmul dst, not a transpose dst), so no slot padding.
    g_a_t_psum = nl.ndarray((dk, BxS), dtype=proj_sb.dtype, buffer=nl.psum)
    nisa.nc_transpose(dst=g_a_t_psum, data=g_a_sb)
    g_a_t = nl.ndarray((dk, BxS), dtype=W_gb.dtype, buffer=nl.sbuf, name="g_a_t")
    nisa.tensor_copy(dst=g_a_t, src=g_a_t_psum, engine=nisa.scalar_engine)

    w_gb_sb = nl.ndarray((dk, i_v), dtype=W_gb.dtype, buffer=nl.sbuf, name="w_gb")
    nisa.dma_copy(dst=w_gb_sb, src=W_gb)

    g_out_chan_major = g_out_sb.permute(dims=[0, 2, 1])
    chans_per_psum = max(1, nlts.psum_fmax // BxS)
    for chan_start in range(0, dv, chans_per_psum):
        chan_n = min(chans_per_psum, dv - chan_start)
        gate_psum = nl.ndarray((num_head, chan_n * BxS), dtype=nl.float32, buffer=nl.psum)
        for i in range(chan_n):
            chan_idx = chan_start + i
            nisa.nc_matmul(
                dst=gate_psum[0:num_head, i * BxS : (i + 1) * BxS],
                stationary=w_gb_sb[0:dk, chan_idx * num_head : (chan_idx + 1) * num_head],
                moving=g_a_t,
            )
        nisa.tensor_copy(
            dst=g_out_chan_major.slice(dim=1, start=chan_start, end=chan_start + chan_n),
            src=gate_psum.reshape_dim(1, [chan_n, BxS]),
            engine=nisa.scalar_engine,
        )

    # ===== Stage 4 setup: convolution taps =====
    # Model parameters, so loaded once. conv_w is tap-major, so each tap is a contiguous
    # run of the stream's channel vector: one [dk, num_head] tile for the channel-outer
    # streams (q, k) and one [num_head, dv] tile for the head-outer stream (v). The
    # partition axis therefore matches whatever the stage-5 transpose produced.
    conv_w_sb = []
    for stream in range(_CONV_STREAMS):
        if stream < 2:
            tap_tile = nl.ndarray((dk, conv_taps, num_head), dtype=conv_w.dtype, buffer=nl.sbuf)
            tap_pattern = [[num_head, dk], [i_qk, conv_taps], [1, num_head]]
        else:
            tap_tile = nl.ndarray((num_head, conv_taps, dv), dtype=conv_w.dtype, buffer=nl.sbuf)
            tap_pattern = [[dv, num_head], [i_qk, conv_taps], [1, dv]]
        nisa.dma_copy(dst=tap_tile, src=conv_w.ap(pattern=tap_pattern, offset=stream * conv_taps * i_qk))
        conv_w_sb.append(tap_tile)

    # Convolution outputs. v's is padded up to a whole quadrant because the recurrence
    # redistributes heads by reading a full enclosing quadrant; the pad is zeroed so that
    # read never touches uninitialized data.
    num_head_p = ((num_head + _SBUF_QUADRANT_SIZE - 1) // _SBUF_QUADRANT_SIZE) * _SBUF_QUADRANT_SIZE
    q_cv = nl.ndarray((dk, BxS, num_head), dtype=nl.float32, buffer=nl.sbuf, name="q_conv")
    k_cv = nl.ndarray((dk, BxS, num_head), dtype=nl.float32, buffer=nl.sbuf, name="k_conv")
    v_cv = nl.ndarray((num_head_p, BxS, dv), dtype=nl.float32, buffer=nl.sbuf, name="v_conv")
    if num_head_p != num_head:
        # memset needs a quadrant-aligned partition start, and the pad rows begin at
        # num_head, so zero the whole tile; the conv overwrites the live rows below.
        nisa.memset(dst=v_cv, value=0)

    # ===== Stage 6 setup: shared scalars and model parameters =====
    scale_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=scale_t, src=scale)
    lower_bound_t = nl.ndarray((1, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=lower_bound_t, src=lower_bound)
    # lower_bound multiplies a [dk, num_head] tile, so it needs dk partitions; the value is
    # constant along both axes, so one partition-broadcast covers it.
    lower_bound_bc = nl.broadcast_to(lower_bound_t, (dk, 1))

    # Column of ones for the L2-norm's partition-axis reduction. Built once.
    ones_col = nl.ndarray((dk, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_col, 1.0)

    dt_bias_in = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=dt_bias_in, src=dt_bias)
    a_log_in = nl.ndarray((1, num_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=a_log_in, src=A_log)
    a_exp_row = nl.ndarray((1, num_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=a_exp_row, op=nl.exp, data=a_log_in)
    a_exp_bc = nl.broadcast_to(a_exp_row, (dk, num_head))

    gamma_o_row = nl.ndarray((1, dv), dtype=nl.float32, buffer=nl.sbuf)
    nisa.dma_copy(dst=gamma_o_row, src=gamma_o)
    gamma_o_bc = nl.broadcast_to(gamma_o_row, (num_head, dv))

    # Stage 9 input, in the layout output_projection_tkg expects: [D, B, N, S].
    attn_sb = nl.ndarray((dv, B, num_head, n), dtype=W_o.dtype, buffer=nl.sbuf, name="attn")

    for b in range(B):
        # ===== Stage 4: causal short conv + SiLU =====
        # Per channel c: out[c] = silu(sum_j w[c, j] * window[c, j]), where the window is
        # the K-1 cached PRE-conv positions followed by this token. The tap varies along
        # both tile axes, so it cannot ride scalar_tensor_tensor's operand0 (which must be
        # a per-partition scalar); each tap is a multiply into a temp plus an add.
        for stream, (src_sb, dst_sb) in enumerate(((q_sb, q_cv), (k_sb, k_cv), (v_sb, v_cv))):
            if stream < 2:
                part, free = dk, num_head
                state_pattern = [[num_head, dk], [i_qk, conv_taps - 1], [1, num_head]]
                tail_pattern = [[num_head, dk], [i_qk, conv_taps - 2], [1, num_head]]
                last_pattern = [[num_head, dk], [1, num_head]]
            else:
                part, free = num_head, dv
                state_pattern = [[dv, num_head], [i_qk, conv_taps - 1], [1, dv]]
                tail_pattern = [[dv, num_head], [i_qk, conv_taps - 2], [1, dv]]
                last_pattern = [[dv, num_head], [1, dv]]
            state_base = ((b * _CONV_STREAMS + stream) * (conv_taps - 1)) * i_qk

            x_cur = src_sb.slice(dim=1, start=b, end=b + 1).squeeze_dim(dim=1)[0:part, 0:free]
            window = nl.ndarray((part, conv_taps - 1, free), dtype=conv_state.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=window, src=conv_state.ap(pattern=state_pattern, offset=state_base))

            taps = conv_w_sb[stream]
            acc = nl.ndarray((part, free), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_tensor(dst=acc, data1=x_cur, data2=taps[:, conv_taps - 1], op=nl.multiply)
            tap_prod = nl.ndarray((part, free), dtype=nl.float32, buffer=nl.sbuf)
            for tap_idx in range(conv_taps - 1):
                nisa.tensor_tensor(dst=tap_prod, data1=window[:, tap_idx], data2=taps[:, tap_idx], op=nl.multiply)
                nisa.tensor_tensor(dst=acc, data1=acc, data2=tap_prod, op=nl.add)

            # Slide the cached window forward by one token before acc is consumed: the new
            # window is the old one minus its oldest position, with this token appended.
            if conv_taps > 2:
                nisa.dma_copy(
                    dst=conv_state.ap(pattern=tail_pattern, offset=state_base),
                    src=window.slice(dim=1, start=1, end=conv_taps - 1),
                )
            nisa.dma_copy(
                dst=conv_state.ap(pattern=last_pattern, offset=state_base + (conv_taps - 2) * i_qk),
                src=x_cur,
            )

            nisa.activation(
                dst=dst_sb.slice(dim=1, start=b, end=b + 1).squeeze_dim(dim=1)[0:part, 0:free],
                op=nl.silu,
                data=acc,
            )

        # ===== Stage 6: preprocessing =====
        q_in = q_cv.slice(dim=1, start=b, end=b + 1).squeeze_dim(dim=1)
        k_in = k_cv.slice(dim=1, start=b, end=b + 1).squeeze_dim(dim=1)
        g_in = g_sb.slice(dim=1, start=b, end=b + 1).squeeze_dim(dim=1)
        v_all = v_cv.slice(dim=1, start=b, end=b + 1).squeeze_dim(dim=1)

        # L2-norm. The sum over dk is a PARTITION reduction, so it is a matmul against
        # ones_col; q and k are packed side by side on the free axis so one nc_matmul
        # produces both [1, num_head] sums.
        qk_sq = nl.ndarray((dk, 2 * num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=qk_sq[0:dk, 0:num_head], op=nl.square, data=q_in)
        nisa.activation(dst=qk_sq[0:dk, num_head : 2 * num_head], op=nl.square, data=k_in)
        ss_row_p = nl.ndarray((1, 2 * num_head), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_matmul(dst=ss_row_p, stationary=ones_col, moving=qk_sq)
        ss_row = nl.ndarray((1, 2 * num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_copy(dst=ss_row, src=ss_row_p, engine=nisa.scalar_engine)
        rnorm_row = nl.ndarray((1, 2 * num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=rnorm_row, op=nl.rsqrt, data=ss_row, bias=l2_eps)

        # The query scale folds into the query's reciprocal norm, so the output needs no
        # post-scale. Reciprocal norms vary along the FREE axis here, so they are
        # partition-broadcast and applied with tensor_tensor rather than tensor_scalar.
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

        # g_act = lower_bound * sigmoid(exp(A_log) * (f_gate + dt_bias)).
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
        nisa.activation(dst=beta_row, op=nl.sigmoid, data=beta_sb[b : b + 1, 0:num_head])
        beta_bc = nl.broadcast_to(beta_row, (dk, num_head))

        k_decayed_all = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_decayed_all, data1=k_normed, data2=exp_g_all, op=nl.multiply)
        k_beta_all = nl.ndarray((dk, num_head), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=k_beta_all, data1=k_normed, data2=beta_bc, op=nl.multiply)

        # ===== Stage 7: recurrence =====
        # Outer pass coalesces transfers; the inner pass does one array-tiling group.
        o_hm = nl.ndarray((num_head, dv), dtype=nl.float32, buffer=nl.sbuf, name="o_hm")
        for head_group_idx in range(0, num_head, _HEAD_GROUP):
            group_size = min(_HEAD_GROUP, num_head - head_group_idx)

            # The whole coalescing group's states arrive and leave in one DMA each.
            S_g = nl.ndarray((dk, group_size, dv), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=S_g, src=S[b, head_group_idx : head_group_idx + group_size].permute((1, 0, 2)))
            S_g_out = nl.ndarray((dk, group_size, dv), dtype=nl.float32, buffer=nl.sbuf)

            for tile_base in range(0, group_size, _TILE_GROUP):
                tile_n = min(_TILE_GROUP, group_size - tile_base)
                tile_head_idx = head_group_idx + tile_base

                # One tensor_copy replication to place data at the array tiling offsets
                # before transpose. Note: AP broadcast errors with 2D free on input.
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

                # Only tile_n of 128 partitions are written by the column tiled matmuls
                # below. nisa.memset rejects psum destinations, so zero it via a
                # multiply-by-0; the uint32 view keeps garbage bit patterns from becoming
                # NaN as an fp multiply would.
                kv_tile_psum = nl.ndarray((nlts.pmax, dv), dtype=nl.float32, buffer=nl.psum)
                nisa.tensor_scalar(
                    dst=kv_tile_psum.reinterpret_cast(nl.uint32),
                    data=kv_tile_psum.reinterpret_cast(nl.uint32),
                    op0=nl.multiply,
                    operand0=0,
                    engine=nisa.scalar_engine,
                )
                # Column tiled matmul for kv = k_decayed @ S
                for t in range(tile_n):
                    nisa.nc_matmul(
                        dst=kv_tile_psum[t * _SBUF_QUADRANT_SIZE : t * _SBUF_QUADRANT_SIZE + 1, 0:dv],
                        stationary=k_decayed_all[:, tile_head_idx + t : tile_head_idx + t + 1],
                        moving=S_g[:, tile_base + t],
                        tile_size=(nlts.pmax, _SBUF_QUADRANT_SIZE),
                        tile_position=(0, t * _SBUF_QUADRANT_SIZE),
                    )

                # Single eviction for the whole tile, enabled by the zero-fill above.
                diff = nl.ndarray((nlts.pmax, dv), dtype=nl.float32, buffer=nl.sbuf)
                nisa.tensor_tensor(dst=diff, data1=v_tile, data2=kv_tile_psum, op=nl.subtract)

                # Row tiled matmul for the outer product
                for t in range(tile_n):
                    head_idx = tile_head_idx + t
                    row_tile = t * _SBUF_QUADRANT_SIZE

                    S_head = S_g[:, tile_base + t]
                    exp_g = exp_g_all[:, head_idx : head_idx + 1]
                    k_beta_row = k_beta_rows[row_tile : row_tile + 1, 0:dk]

                    # S_updated = Diag(exp_g) @ S_head + outer(k*beta, diff). The outer
                    # product is a contraction-1 nc_matmul against the tile's
                    # pre-transposed [1, dk] row.
                    outer_psum = nl.ndarray((dk, dv), dtype=nl.float32, buffer=nl.psum)
                    nisa.nc_matmul(
                        dst=outer_psum,
                        stationary=k_beta_row,
                        moving=diff[row_tile : row_tile + 1, 0:dv],
                        tile_size=(_SBUF_QUADRANT_SIZE, nlts.pmax),
                        tile_position=(row_tile, 0),
                    )
                    # The decay multiply and the additive update are fused on evict.
                    nisa.scalar_tensor_tensor(
                        dst=S_g_out[:, tile_base + t],
                        data=S_head,
                        op0=nl.multiply,
                        operand0=exp_g,
                        op1=nl.add,
                        operand1=outer_psum,
                    )

                # Column tiled matmul for o = q.T @ S_updated
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
                    # SBUF->SBUF DMA copy to place the output on its head's partition
                    nisa.dma_copy(
                        dst=o_hm[head_idx : head_idx + 1, 0:dv],
                        src=out_temp[col_tile : col_tile + 1, 0:dv],
                    )

            # Grouped writeback: the advanced states leave in one DMA.
            nisa.dma_copy(dst=S[b, head_group_idx : head_group_idx + group_size].permute((1, 0, 2)), src=S_g_out)

        # ===== Stage 8: gated output RMSNorm =====
        # Heads are on partitions and dv on the free axis, so the norm reduces along the
        # free axis and every broadcast is a per-partition scalar. Norm first, then gate.
        o_sq = nl.ndarray((num_head, dv), dtype=nl.float32, buffer=nl.sbuf)
        o_ss = nl.ndarray((num_head, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation_reduce(dst=o_sq, op=nl.square, data=o_hm, reduce_op=nl.add, reduce_res=o_ss)
        # rsqrt(mean(o^2) + eps), the mean folded in as the activation's input scale.
        o_rnorm = nl.ndarray((num_head, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=o_rnorm, op=nl.rsqrt, data=o_ss, scale=1.0 / dv, bias=rms_norm_eps)
        o_norm = nl.ndarray((num_head, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.scalar_tensor_tensor(
            dst=o_norm, data=o_hm, op0=nl.multiply, operand0=o_rnorm, op1=nl.multiply, operand1=gamma_o_bc
        )
        o_gate = nl.ndarray((num_head, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.activation(dst=o_gate, op=nl.sigmoid, data=g_out_sb.slice(dim=1, start=b, end=b + 1).squeeze_dim(dim=1))
        o_gated = nl.ndarray((num_head, dv), dtype=nl.float32, buffer=nl.sbuf)
        nisa.tensor_tensor(dst=o_gated, data1=o_norm, data2=o_gate, op=nl.multiply)

        # Stage 9 wants dv on partitions; one transpose per request moves it off the head
        # axis and straight into the [D, B, N, S] staging tile.
        o_t_psum = nl.ndarray((dv, num_head), dtype=nl.float32, buffer=nl.psum)
        nisa.nc_transpose(dst=o_t_psum, data=o_gated)
        nisa.tensor_copy(dst=attn_sb[0:dv, b, 0:num_head, 0], src=o_t_psum, engine=nisa.vector_engine)

    # ===== Stage 9: output projection =====
    # W_o's rows are head-outer, which is exactly the [n * D + d, h] indexing this kernel
    # wants, and it takes the attention tile straight from SBUF with no round trip. The
    # result is this rank's PARTIAL; the caller reduces it and adds the residual.
    # It LNC-shards H internally, so OUT_IN_SB returns only this core's [BxS, H // LNC]
    # slice; each core writes its own contiguous band of the caller's output.
    _, n_prgs, prg_id = get_program_sharding_info()
    kernel_assert(H % n_prgs == 0, f"H must be divisible by the core count {n_prgs}, got H={H}")
    proj_out_sb = output_projection_tkg(attention=attn_sb, weight=W_o, OUT_IN_SB=True)
    out_shard = out.reshape((BxS, H)).reshape_dim(1, (n_prgs, H // n_prgs)).select(dim=1, index=prg_id)
    nisa.dma_copy(dst=out_shard, src=proj_out_sb)

    return out, S, conv_state
