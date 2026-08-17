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
GDN Depthwise Causal Conv1D Kernels

Depthwise causal convolution + silu + conv_state update that precedes the
Qwen3.5 gated-delta-net (GDN) linear attention. Prefill and decode variants.

Mirrors the torch reference in vllm_neuron qwen3_5 model_bf16.py exactly
(bias=False, silu activation, depthwise groups=conv_dim, kernel size K=4).

        +-------------------+   +-------------------+   +-------------------+
        | mixed (q|k|v)     |-->|  causal FIR       |-->|  silu             |-->  conv_out
        | [B, C, T]  (HBM)  |   |  (per-channel)    |   |                   |
        +-------------------+   +-------------------+   +-------------------+
                                        |
                                        v
                                  conv_state (last K tokens)

Layout / mechanics:
    Depthwise conv over conv_dim=C independent channels with a tiny kernel
    (K=4). Instead of the generic implicit-GEMM in experimental/conv, this uses
    a per-channel FIR shift-multiply-accumulate:

        out[c, t] = sum_{j=0..K-1} w[c, j] * in[c, t - (K-1) + j]      (causal)

    Channels map to the partition dimension (128-wide tiles); the time axis T is
    the (contiguous) free dimension. Each tap j is one per-partition-scalar
    multiply-accumulate (tensor_scalar / scalar_tensor_tensor), where w[:, j] is
    a [P, 1] per-partition scalar broadcast across the free axis. No matmul
    needed.
"""

from typing import Tuple

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_helpers import div_ceil
from ...core.utils.logging import get_logger

logger = get_logger("gdn_conv1d")

_PMAX = 128  # Partition dimension size (channels map to the partition axis)


@nki.jit
def gdn_conv1d_prefill(
    mixed: nl.NkiTensor,
    conv_weight: nl.NkiTensor,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Prefill depthwise causal conv1d + silu + conv_state update.

    Mirrors model_bf16.py (fresh-prefill / real_len == T case): the new
    conv_state is the last K tokens of the RAW (pre-conv) mixed tensor.
    TODO: Specify intended usage range (e.g., prefill token count T, batch size B).

    Dimensions:
        B: Batch size
        C: Number of channels (== conv_dim, e.g. 8192)
        T: Number of prefill tokens
        K: Conv kernel width (e.g. 4)

    Args:
        mixed (nl.NkiTensor): [B, C, T] @ HBM, RAW pre-conv input
            (q|k|v concatenated on the channel axis).
        conv_weight (nl.NkiTensor): [C, K] @ HBM, depthwise weights
            (self.conv1d.weight squeezed).

    Returns:
        conv_out (nl.NkiTensor): [B, C, T] @ HBM, silu(causal_conv1d(mixed)).
        new_conv_state (nl.NkiTensor): [B, C, K] @ HBM, last K RAW tokens of mixed.

    Notes:
        - Depthwise (groups == C); each channel is an independent FIR.
        - Channels map to the partition axis (128-wide tiles), T to the free axis.

    Pseudocode:
        for c, t:
            # causal per-channel FIR (K taps), zero-padded on the left
            out[c, t] = sum_{j=0..K-1} conv_weight[c, j] * mixed[c, t - (K-1) + j]
        conv_out = silu(out)
        new_conv_state = mixed[:, :, -K:]   # last K RAW (pre-conv) tokens
    """
    B, C, T = mixed.shape
    K = conv_weight.shape[1]

    conv_out = nl.ndarray((B, C, T), dtype=mixed.dtype, buffer=nl.shared_hbm)
    new_conv_state = nl.ndarray((B, C, K), dtype=mixed.dtype, buffer=nl.shared_hbm)

    num_c_tiles = div_ceil(C, _PMAX)

    for batch_idx in nl.affine_range(B):
        for c_tile_idx in nl.affine_range(num_c_tiles):
            c0 = c_tile_idx * _PMAX
            p_sz = min(_PMAX, C - c0)

            in_sb = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=in_sb[0:p_sz, 0:T], src=mixed[batch_idx, c0 : c0 + p_sz, 0:T])

            w_sb = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_sb[0:p_sz, 0:K], src=conv_weight[c0 : c0 + p_sz, 0:K])

            acc = nl.ndarray((_PMAX, T), dtype=nl.float32, buffer=nl.sbuf)
            _causal_fir(acc, in_sb, w_sb, p_sz, T, K)

            out_sb = nl.ndarray((_PMAX, T), dtype=conv_out.dtype, buffer=nl.sbuf)
            nisa.activation(dst=out_sb[0:p_sz, 0:T], op=nl.silu, data=acc[0:p_sz, 0:T])
            nisa.dma_copy(dst=conv_out[batch_idx, c0 : c0 + p_sz, 0:T], src=out_sb[0:p_sz, 0:T])

            # conv_state = last K RAW (pre-conv) tokens of mixed.
            state_sb = nl.ndarray((_PMAX, K), dtype=new_conv_state.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=state_sb[0:p_sz, 0:K], src=in_sb[0:p_sz, T - K : T])
            nisa.dma_copy(dst=new_conv_state[batch_idx, c0 : c0 + p_sz, 0:K], src=state_sb[0:p_sz, 0:K])

    return conv_out, new_conv_state


@nki.jit
def gdn_conv1d_decode(
    x: nl.NkiTensor,
    conv_state: nl.NkiTensor,
    conv_weight: nl.NkiTensor,
) -> Tuple[nl.NkiTensor, nl.NkiTensor]:
    """
    Decode-step depthwise causal conv1d update + silu.

    Mirrors _torch_causal_conv1d_update (model_bf16.py):
        hs = cat([conv_state, x], dim=-1)[:, :, 1:]   # slide window, [B, C, K]
        out = silu(depthwise_conv1d(hs))              # [B, C, 1]
        new_conv_state = hs

    Processes a single decode token (T == 1) per call.

    Dimensions:
        B: Batch size
        C: Number of channels (== conv_dim, e.g. 8192)
        K: Conv kernel width (e.g. 4)

    Args:
        x (nl.NkiTensor): [B, C, 1] @ HBM, current-token RAW mixed (q|k|v).
        conv_state (nl.NkiTensor): [B, C, K] @ HBM, previous conv window.
        conv_weight (nl.NkiTensor): [C, K] @ HBM, depthwise weights.

    Returns:
        out (nl.NkiTensor): [B, C, 1] @ HBM, silu(conv1d(sliding window)).
        new_conv_state (nl.NkiTensor): [B, C, K] @ HBM, shifted window (== hs).

    Notes:
        - Single output position; the FIR reduces to a K-tap dot product.
        - Depthwise (groups == C); channels map to the partition axis.

    Pseudocode:
        hs = cat([conv_state, x], dim=-1)[:, :, 1:]   # slide window, [B, C, K]
        for c:
            out[c] = sum_{j=0..K-1} conv_weight[c, j] * hs[c, j]
        out = silu(out)
        new_conv_state = hs
    """
    B, C, _ = x.shape
    K = conv_weight.shape[1]

    out = nl.ndarray((B, C, 1), dtype=x.dtype, buffer=nl.shared_hbm)
    new_conv_state = nl.ndarray((B, C, K), dtype=x.dtype, buffer=nl.shared_hbm)

    num_c_tiles = div_ceil(C, _PMAX)

    for batch_idx in nl.affine_range(B):
        for c_tile_idx in nl.affine_range(num_c_tiles):
            c0 = c_tile_idx * _PMAX
            p_sz = min(_PMAX, C - c0)

            cs_sb = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=cs_sb[0:p_sz, 0:K], src=conv_state[batch_idx, c0 : c0 + p_sz, 0:K])

            x_sb = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=x_sb[0:p_sz, 0:1], src=x[batch_idx, c0 : c0 + p_sz, 0:1])

            # hs = [conv_state[:, 1:K], x]  (slide window left by one).
            hs_sb = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_copy(dst=hs_sb[0:p_sz, 0 : K - 1], src=cs_sb[0:p_sz, 1:K])
            nisa.tensor_copy(dst=hs_sb[0:p_sz, K - 1 : K], src=x_sb[0:p_sz, 0:1])

            w_sb = nl.ndarray((_PMAX, K), dtype=nl.float32, buffer=nl.sbuf)
            nisa.dma_copy(dst=w_sb[0:p_sz, 0:K], src=conv_weight[c0 : c0 + p_sz, 0:K])

            # out = sum_j w[:, j] * hs[:, j]   (single output position).
            acc = nl.ndarray((_PMAX, 1), dtype=nl.float32, buffer=nl.sbuf)
            nisa.tensor_scalar(
                dst=acc[0:p_sz, 0:1],
                data=hs_sb[0:p_sz, 0:1],
                op0=nl.multiply,
                operand0=w_sb[0:p_sz, 0:1],
            )
            for tap_idx in range(1, K):
                nisa.scalar_tensor_tensor(
                    dst=acc[0:p_sz, 0:1],
                    data=hs_sb[0:p_sz, tap_idx : tap_idx + 1],
                    op0=nl.multiply,
                    operand0=w_sb[0:p_sz, tap_idx : tap_idx + 1],
                    op1=nl.add,
                    operand1=acc[0:p_sz, 0:1],
                )

            out_sb = nl.ndarray((_PMAX, 1), dtype=out.dtype, buffer=nl.sbuf)
            nisa.activation(dst=out_sb[0:p_sz, 0:1], op=nl.silu, data=acc[0:p_sz, 0:1])
            nisa.dma_copy(dst=out[batch_idx, c0 : c0 + p_sz, 0:1], src=out_sb[0:p_sz, 0:1])

            # new_conv_state = hs (the shifted window).
            state_sb = nl.ndarray((_PMAX, K), dtype=new_conv_state.dtype, buffer=nl.sbuf)
            nisa.tensor_copy(dst=state_sb[0:p_sz, 0:K], src=hs_sb[0:p_sz, 0:K])
            nisa.dma_copy(dst=new_conv_state[batch_idx, c0 : c0 + p_sz, 0:K], src=state_sb[0:p_sz, 0:K])

    return out, new_conv_state


def _causal_fir(acc, in_sb, w_sb, P, T, K):
    """acc[0:P, 0:T] = causal depthwise FIR of in_sb with taps w_sb.

    acc, in_sb: [P, T] SBUF (acc fp32). w_sb: [P, K] SBUF per-channel taps.
    out[:, t] = sum_{j} w[:, j] * in[:, t - (K-1) + j], zero-padded on the left.

    Tap j contributes with left-shift = (K-1) - j:
        acc[:, shift:T] += w[:, j] * in[:, 0:T-shift]
    The full-width tap (shift == 0, j == K-1) initializes the accumulator.
    """
    # Initialize accumulator with the full-width (shift=0) tap j=K-1.
    nisa.tensor_scalar(
        dst=acc[0:P, 0:T],
        data=in_sb[0:P, 0:T],
        op0=nl.multiply,
        operand0=w_sb[0:P, K - 1 : K],
    )
    # Accumulate the remaining (left-shifted) taps.
    for tap_idx in range(K - 1):
        shift = (K - 1) - tap_idx
        if shift >= T:
            # Tap reaches entirely into the zero left-pad for this tile width.
            continue
        nisa.scalar_tensor_tensor(
            dst=acc[0:P, shift:T],
            data=in_sb[0:P, 0 : T - shift],
            op0=nl.multiply,
            operand0=w_sb[0:P, tap_idx : tap_idx + 1],
            op1=nl.add,
            operand1=acc[0:P, shift:T],
        )
