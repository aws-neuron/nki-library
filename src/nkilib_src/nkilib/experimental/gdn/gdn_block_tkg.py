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
GDN Block TKG Kernel (Partial Megakernel, DECODE)

Assembles the validated NKI components into ONE @nki.jit kernel so the Qwen3.5
GDN decode layer only makes a single kernel call (out_proj + all_reduce aside):

        +-------------------+   +-------------------+   +-------------------+   +-----------------------+
        | hidden (HBM)      |-->| in_proj matmuls   |-->| gdn_conv1d_decode |-->| split + GQA-repeat    |-->
        | [B, 1, H]         |   | (q|k|v, z, a, b)  |   | (conv + silu +    |   | q,k -> flat [BH, .]   |
        +-------------------+   +-------------------+   |  slide-window)    |   +-----------------------+
                                                        +-------------------+

        +-----------------------+   +-------------------+
    --> | gdn_tkg            |-->| scatter -> core_out |
        | (l2norm/beta/g/       |   | [B, 1, value_dim]   |
        |  RMSNormGated + recur)|   +-------------------+
        +-----------------------+

out_proj (and the collective all_reduce) stay in PYTHON, matching the fusion
boundary in docs/gdn-megakernel-design.md and model_bf16.py _forward_decode
MINUS out_proj.

State I/O: conv_state and recurrent_state are read at entry and the updated
copies are returned; the caller aliases them back to the persistent cache
buffers (like attention_block_tkg's update_cache K/V aliasing in
attention_decode.py). The kernel itself is functional (no in-place mutation of
its inputs).

The two component kernels (gdn_conv1d_decode, gdn_tkg) are reused VERBATIM as
nested @nki.jit calls; internal nl.hbm tensors carry the intermediate
activations (mixed, conv_out, per-head flat q/k/v/z, a/b) between stages.

Dims (Qwen3.5): H=2048, key_dim=2048, value_dim=4096, conv_dim=8192,
num_k_heads=16, num_v_heads=32, head_k_dim=head_v_dim=128, conv_kernel=4, gqa=2.
All dims are derived from tensor shapes (no python-int args).
"""

from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import div_ceil
from ...core.utils.logging import get_logger
from .gdn_conv1d import gdn_conv1d_decode
from .gdn_tkg import gdn_tkg

logger = get_logger("gdn_block_tkg")

_P_MAX = 128


def _in_proj_channel_partition(W, out_dim, dst_hbm, hidden_sb, num_h_tiles, B):
    """dst_hbm[b, c, 0] = (hidden @ W)[b, c].  channel-on-middle layout.

    result[c, b] = sum_h W[h, c] * hidden[b, h] via
    nc_matmul(dst[C,B], stationary=W[H,C], moving=hidden[H,B]) accumulated over H tiles.
    """
    for channel_tile_idx in nl.affine_range(div_ceil(out_dim, _P_MAX)):
        channel_start = channel_tile_idx * _P_MAX
        channel_size = min(_P_MAX, out_dim - channel_start)
        w_tile = nl.ndarray((_P_MAX, num_h_tiles, _P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=w_tile[0:_P_MAX, 0:num_h_tiles, 0:channel_size],
            src=W.ap(
                pattern=[[out_dim, _P_MAX], [_P_MAX * out_dim, num_h_tiles], [1, channel_size]],
                offset=channel_start,
            ),
        )
        psum = nl.ndarray((_P_MAX, B), dtype=nl.float32, buffer=nl.psum)
        for h_tile_idx in nl.affine_range(num_h_tiles):
            nisa.nc_matmul(
                psum[0:channel_size, 0:B],
                w_tile[0:_P_MAX, h_tile_idx, 0:channel_size],
                hidden_sb[0:_P_MAX, h_tile_idx, 0:B],
            )
        out_sb = nl.ndarray((_P_MAX, B), dtype=dst_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_sb[0:channel_size, 0:B], src=psum[0:channel_size, 0:B])
        for batch_idx in range(B):
            nisa.dma_copy(
                dst=dst_hbm[batch_idx, channel_start : channel_start + channel_size, 0:1],
                src=out_sb[0:channel_size, batch_idx : batch_idx + 1],
            )


def _in_proj_channel_free(W, out_dim, dst_hbm, hidden_sb, num_h_tiles, B):
    """dst_hbm[b, c] = (hidden @ W)[b, c].  channel-on-free layout (2D)."""
    for channel_tile_idx in nl.affine_range(div_ceil(out_dim, _P_MAX)):
        channel_start = channel_tile_idx * _P_MAX
        channel_size = min(_P_MAX, out_dim - channel_start)
        w_tile = nl.ndarray((_P_MAX, num_h_tiles, _P_MAX), dtype=nl.bfloat16, buffer=nl.sbuf)
        nisa.dma_copy(
            dst=w_tile[0:_P_MAX, 0:num_h_tiles, 0:channel_size],
            src=W.ap(
                pattern=[[out_dim, _P_MAX], [_P_MAX * out_dim, num_h_tiles], [1, channel_size]],
                offset=channel_start,
            ),
        )
        psum = nl.ndarray((_P_MAX, B), dtype=nl.float32, buffer=nl.psum)
        for h_tile_idx in nl.affine_range(num_h_tiles):
            nisa.nc_matmul(
                psum[0:channel_size, 0:B],
                w_tile[0:_P_MAX, h_tile_idx, 0:channel_size],
                hidden_sb[0:_P_MAX, h_tile_idx, 0:B],
            )
        out_sb = nl.ndarray((_P_MAX, B), dtype=dst_hbm.dtype, buffer=nl.sbuf)
        nisa.tensor_copy(dst=out_sb[0:channel_size, 0:B], src=psum[0:channel_size, 0:B])
        # transpose store: SBUF column [channel_size,1] (channel on partition) -> HBM row [1,channel_size].
        for batch_idx in range(B):
            nisa.dma_copy(
                dst=dst_hbm.ap(pattern=[[1, channel_size], [0, 1]], offset=batch_idx * out_dim + channel_start),
                src=out_sb[0:channel_size, batch_idx : batch_idx + 1],
            )


def _in_proj_flat_bh(W, dst_bh, hidden_sb, num_h_tiles, B, num_v_heads):
    """dst_bh[b*num_v_heads + vh] = (hidden @ W)[b, vh].  a/b per-head scalars."""
    nv = num_v_heads
    w_tile = nl.ndarray((_P_MAX, num_h_tiles, nv), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=w_tile,
        src=W.ap(pattern=[[nv, _P_MAX], [_P_MAX * nv, num_h_tiles], [1, nv]], offset=0),
    )
    psum = nl.ndarray((nv, B), dtype=nl.float32, buffer=nl.psum)
    for h_tile_idx in nl.affine_range(num_h_tiles):
        nisa.nc_matmul(psum[0:nv, 0:B], w_tile[0:_P_MAX, h_tile_idx, 0:nv], hidden_sb[0:_P_MAX, h_tile_idx, 0:B])
    out_sb = nl.ndarray((nv, B), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_copy(dst=out_sb, src=psum)
    for batch_idx in range(B):
        nisa.dma_copy(
            dst=dst_bh.ap(pattern=[[1, nv], [0, 1]], offset=batch_idx * nv),
            src=out_sb[0:nv, batch_idx : batch_idx + 1],
        )


@nki.jit
def gdn_block_tkg(
    hidden: nl.NkiTensor,  # [B, 1, H]                     bf16
    W_in_qkv: nl.NkiTensor,  # [H, 2*key_dim + value_dim]    bf16  (== nn.Linear.weight.T)
    W_in_z: nl.NkiTensor,  # [H, value_dim]                bf16
    W_in_a: nl.NkiTensor,  # [H, num_v_heads]              bf16
    W_in_b: nl.NkiTensor,  # [H, num_v_heads]              bf16
    conv_weight: nl.NkiTensor,  # [conv_dim, conv_kernel]       fp32  (self.conv1d.weight squeezed)
    conv_state: nl.NkiTensor,  # [B, conv_dim, conv_kernel]    bf16  (read at entry)
    A_log: nl.NkiTensor,  # [num_v_heads]                 fp32
    dt_bias: nl.NkiTensor,  # [num_v_heads]                 fp32
    norm_weight: nl.NkiTensor,  # [head_v_dim]                  fp32
    recurrent_state: nl.NkiTensor,  # [B, num_v_heads, head_k_dim, head_v_dim] fp32 (read at entry)
) -> Tuple[nl.NkiTensor, nl.NkiTensor, nl.NkiTensor]:
    """
    Fused GDN decode block: in_proj + conv1d + GQA + gated delta-rule in ONE kernel.

    Runs the full Qwen3.5 GDN decode layer except out_proj / all_reduce:
    in_proj matmuls -> gdn_conv1d_decode (conv + silu + slide-window state) ->
    split conv_out + GQA-repeat q,k -> gdn_tkg (l2norm/beta/g/RMSNormGated +
    single-token recurrence) -> scatter to core_out.

    Dimensions:
        B: Batch size
        H: Hidden dimension (must be a multiple of 128)
        conv_dim: Conv channel count = 2*key_dim + value_dim (q|k|v packed)
        key_dim: Query/key projection dim = num_k_heads * head_k_dim
        value_dim: Value projection dim = num_v_heads * head_v_dim
        num_k_heads: Number of key heads
        num_v_heads: Number of value heads (GQA target)
        head_k_dim: Key/state head dim (<= 128)
        head_v_dim: Value head dim (<= 128)
        gqa: Query heads per key head = num_v_heads / num_k_heads
        BH: Flattened B * num_v_heads
        K_win: Conv kernel width

    Args:
        hidden (nl.NkiTensor): [B, 1, H] @ HBM, bf16 decode-step hidden states.
        W_in_qkv (nl.NkiTensor): [H, conv_dim] @ HBM, bf16 q|k|v in_proj weight
            (== nn.Linear.weight.T).
        W_in_z (nl.NkiTensor): [H, value_dim] @ HBM, bf16 gate (z) in_proj weight.
        W_in_a (nl.NkiTensor): [H, num_v_heads] @ HBM, bf16 gate-input (a) in_proj weight.
        W_in_b (nl.NkiTensor): [H, num_v_heads] @ HBM, bf16 beta-input (b) in_proj weight.
        conv_weight (nl.NkiTensor): [conv_dim, K_win] @ HBM, fp32 depthwise conv weight
            (self.conv1d.weight squeezed).
        conv_state (nl.NkiTensor): [B, conv_dim, K_win] @ HBM, bf16 conv window (read at entry).
        A_log (nl.NkiTensor): [num_v_heads] @ HBM, fp32 per-v-head log decay.
        dt_bias (nl.NkiTensor): [num_v_heads] @ HBM, fp32 per-v-head softplus bias.
        norm_weight (nl.NkiTensor): [head_v_dim] @ HBM, fp32 RMSNormGated weight (gamma).
        recurrent_state (nl.NkiTensor): [B, num_v_heads, head_k_dim, head_v_dim] @ HBM,
            fp32 recurrent state (read at entry).

    Returns:
        core_out (nl.NkiTensor): [B, 1, value_dim] @ HBM, RMSNormGated decode output.
        new_conv_state (nl.NkiTensor): [B, conv_dim, K_win] @ HBM, updated conv window.
        new_recurrent_state (nl.NkiTensor): [B, num_v_heads, head_k_dim, head_v_dim] @ HBM,
            updated recurrent state.

    Notes:
        - Functional: inputs are not mutated in place; the caller aliases the
          returned conv_state / recurrent_state back into the persistent caches.
        - gdn_conv1d_decode and gdn_tkg are reused verbatim as nested @nki.jit
          calls; internal nl.hbm scratch carries the inter-stage activations.
        - All dims are derived from tensor shapes (no python-int args).

    Pseudocode:
        # in_proj matmuls (hidden @ W, channel = output feature):
        mixed = hidden @ W_in_qkv        # [B, conv_dim, 1] (q|k|v packed, channel-on-middle)
        z     = hidden @ W_in_z          # [B, value_dim]
        a     = hidden @ W_in_a          # [BH] per-v-head scalar
        b     = hidden @ W_in_b          # [BH] per-v-head scalar

        # conv1d decode (validated component, reused verbatim):
        conv_out, new_conv_state = gdn_conv1d_decode(mixed, conv_state, conv_weight)

        # split conv_out -> q/k/v and GQA-repeat q,k to num_v_heads:
        #   channels: q [0, key_dim)  k [key_dim, 2*key_dim)  v [2*key_dim, conv_dim)
        #   expanded v-head vh uses k-head kh = vh // gqa (repeat_interleave)
        for b in range(B):
            for vh in range(num_v_heads):
                kh = vh // gqa
                q_bh[b*num_v_heads + vh] = conv_out[b, q_off + kh*head_k_dim : ...]
                k_bh[b*num_v_heads + vh] = conv_out[b, k_off + kh*head_k_dim : ...]
                v_bh[b*num_v_heads + vh] = conv_out[b, v_off + vh*head_v_dim : ...]
                z_bh[b*num_v_heads + vh] = z[b, vh*head_v_dim : ...]

        # gdn_tkg core (validated component, reused verbatim):
        core_flat, state_out = gdn_tkg(q_bh, k_bh, v_bh, b_bh, a_bh, A_log, dt_bias, z_bh, norm_weight, state_bh)

        # scatter core_flat[BH, V] -> core_out[B, 1, value_dim]:
        for b in range(B):
            for vh in range(num_v_heads):
                core_out[b, 0, vh*head_v_dim + d] = core_flat[b*num_v_heads + vh, d]
        return core_out, new_conv_state, new_recurrent_state
    """
    B = hidden.shape[0]
    H = hidden.shape[2]
    conv_dim = W_in_qkv.shape[1]
    value_dim = W_in_z.shape[1]
    num_v_heads = A_log.shape[0]
    head_v_dim = value_dim // num_v_heads
    key_dim = (conv_dim - value_dim) // 2
    head_k_dim = recurrent_state.shape[2]
    num_k_heads = key_dim // head_k_dim
    gqa = num_v_heads // num_k_heads
    BH = B * num_v_heads
    K_win = conv_weight.shape[1]

    kernel_assert(
        head_k_dim <= 128 and head_v_dim <= 128,
        f"head_k_dim and head_v_dim must each be <= 128, got {head_k_dim=}, {head_v_dim=}",
    )
    kernel_assert(H % _P_MAX == 0, f"H must be a multiple of 128, got {H=}")
    # No LNC sharding: the in_proj / conv1d / split / scatter stages are written for a
    # single core (only the nested gdn_tkg shards the BH axis internally). Require grid==1.
    kernel_assert(
        nl.program_ndim() == 0 or nl.num_programs(0) == 1,
        "gdn_block_tkg does not support LNC sharding; launch on a single core (grid size 1), "
        f"got program_ndim={nl.program_ndim()}",
    )
    num_h_tiles = H // _P_MAX

    # ---- Preload hidden^T into SBUF: hidden_sb[128, num_h_tiles, B] ----
    # hidden[b, 0, ht*128 + p] -> hidden_sb[p, ht, b].  (H on partition, tiled.)
    hidden_sb = nl.ndarray((_P_MAX, num_h_tiles, B), dtype=nl.bfloat16, buffer=nl.sbuf)
    nisa.dma_copy(
        dst=hidden_sb,
        src=hidden.ap(
            pattern=[[1, _P_MAX], [_P_MAX, num_h_tiles], [H, B]],
            offset=0,
        ),
    )

    """
    in_proj matmuls (module-level helpers).  For each output feature block:
        result[c, b] = sum_h W[h, c] * hidden[b, h]           (== hidden @ W)
    mixed[B, conv_dim, 1] (channel-on-middle) feeds the conv kernel directly.
    """
    mixed = nl.ndarray((B, conv_dim, 1), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    z_hbm = nl.ndarray((B, value_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    a_bh = nl.ndarray((BH,), dtype=nl.float32, buffer=nl.shared_hbm)
    b_bh = nl.ndarray((BH,), dtype=nl.float32, buffer=nl.shared_hbm)

    _in_proj_channel_partition(W_in_qkv, conv_dim, mixed, hidden_sb, num_h_tiles, B)
    _in_proj_channel_free(W_in_z, value_dim, z_hbm, hidden_sb, num_h_tiles, B)
    _in_proj_flat_bh(W_in_a, a_bh, hidden_sb, num_h_tiles, B, num_v_heads)
    _in_proj_flat_bh(W_in_b, b_bh, hidden_sb, num_h_tiles, B, num_v_heads)

    """conv1d decode (validated component, reused verbatim)."""
    conv_out, new_conv_state = gdn_conv1d_decode(mixed, conv_state, conv_weight)
    #   conv_out: [B, conv_dim, 1] bf16   new_conv_state: [B, conv_dim, K_win] bf16

    """
    split conv_out -> q/k/v; GQA-repeat q,k; build flat [BH, .] scratch for tkg.
        channels:  q [0, key_dim)  k [key_dim, 2*key_dim)  v [2*key_dim, conv_dim)
        expanded v-head vh uses k-head kh = vh // gqa (repeat_interleave).
    """
    q_bh = nl.ndarray((BH, head_k_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    k_bh = nl.ndarray((BH, head_k_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    v_bh = nl.ndarray((BH, head_v_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)
    z_bh = nl.ndarray((BH, head_v_dim), dtype=nl.bfloat16, buffer=nl.shared_hbm)

    q_off = 0
    k_off = key_dim
    v_off = 2 * key_dim
    HK = head_k_dim
    HV = head_v_dim
    for batch_idx in range(B):
        for v_head_idx in range(num_v_heads):
            bh = batch_idx * num_v_heads + v_head_idx
            kh = v_head_idx // gqa
            # q (transpose: conv_out channel-on-partition -> q_bh row on free)
            tmp = nl.ndarray((HK, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=tmp, src=conv_out[batch_idx, q_off + kh * HK : q_off + kh * HK + HK, 0:1])
            nisa.dma_copy(dst=q_bh.ap(pattern=[[1, HK], [0, 1]], offset=bh * HK), src=tmp)
            # k
            nisa.dma_copy(dst=tmp, src=conv_out[batch_idx, k_off + kh * HK : k_off + kh * HK + HK, 0:1])
            nisa.dma_copy(dst=k_bh.ap(pattern=[[1, HK], [0, 1]], offset=bh * HK), src=tmp)
            # v
            tmpv = nl.ndarray((HV, 1), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(
                dst=tmpv,
                src=conv_out[batch_idx, v_off + v_head_idx * HV : v_off + v_head_idx * HV + HV, 0:1],
            )
            nisa.dma_copy(dst=v_bh.ap(pattern=[[1, HV], [0, 1]], offset=bh * HV), src=tmpv)
            # z (already channel-on-free in z_hbm[batch_idx, :] -> straight row copy)
            zt = nl.ndarray((1, HV), dtype=nl.bfloat16, buffer=nl.sbuf)
            nisa.dma_copy(dst=zt, src=z_hbm[batch_idx : batch_idx + 1, v_head_idx * HV : v_head_idx * HV + HV])
            nisa.dma_copy(dst=z_bh[bh : bh + 1, 0:HV], src=zt)

    """gdn_tkg core (validated component, reused verbatim)."""
    state_bh = recurrent_state.reshape((BH, head_k_dim, head_v_dim))
    core_flat, state_out = gdn_tkg(q_bh, k_bh, v_bh, b_bh, a_bh, A_log, dt_bias, z_bh, norm_weight, state_bh)
    #   core_flat: [BH, head_v_dim] bf16   state_out: [BH, head_k_dim, head_v_dim] fp32

    """
    scatter core_flat[BH, V] -> core_out[B, 1, value_dim].
        core_out[b, 0, vh*V + d] = core_flat[b*num_v_heads + vh, d]
    """
    core_out = nl.ndarray((B, 1, value_dim), dtype=core_flat.dtype, buffer=nl.shared_hbm)
    for batch_idx in range(B):
        for v_head_idx in range(num_v_heads):
            bh = batch_idx * num_v_heads + v_head_idx
            ot = nl.ndarray((1, HV), dtype=core_flat.dtype, buffer=nl.sbuf)
            nisa.dma_copy(dst=ot, src=core_flat[bh : bh + 1, 0:HV])
            nisa.dma_copy(dst=core_out[batch_idx, 0:1, v_head_idx * HV : v_head_idx * HV + HV], src=ot)

    new_recurrent_state = state_out.reshape((B, num_v_heads, head_k_dim, head_v_dim))
    return core_out, new_conv_state, new_recurrent_state
