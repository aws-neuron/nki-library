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

"""Pure PyTorch reference for MXFP4 projection sub-kernels (gate_up and down).

Replicates the hardware MX matmul behavior: unpack x4 -> reshape to [P, F//4, 4]
-> apply block scale per 8x1x4 block -> einsum contraction.

These references can be composed into higher-level torch references (e.g., MLP TKG,
MoE TKG) that need MXFP4 projection, or tested standalone.

Note: x4 packed numpy arrays are the only numpy inputs — they arrive as numpy
because torch_ref_wrapper cannot convert them. The unpack functions convert them
to torch immediately. All other inputs are torch tensors.
"""

import math

import numpy as np
import torch

from ...utils.mx_torch_common import (
    mx_matmul,
    quantize_to_mx_fp8,
    unpack_float4_x4,
    unpack_float8_e4m3fn_x4,
    unpack_float8_e5m2_x4,
)

_pmax = 128
_q_width = 4
_q_height = 8


# =============================================================================
# Public torch reference functions
# =============================================================================


def gate_up_proj_mxfp4_torch_ref(
    hidden_qtz: np.ndarray,
    hidden_scale: torch.Tensor,
    weight_qtz: np.ndarray,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    H: int,
    I: int,
    BxS: int,
    hidden_unpack_fn: callable = None,
    weight_unpack_fn: callable = None,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation of MXFP4 gate/up projection.

    Computes: hidden [H, BxS] @ weight [H, I] -> [I, BxS], with MX block-scaled
    quantized matmul, then reshuffles output to tiled [128, n_I512, BxS, 4] layout.

    This is a reference implementation for testing the gate_up_projection_mx_tp_shard_H
    NKI kernel. It prioritizes correctness over performance.

    Args:
        hidden_qtz: numpy x4 [128, H//512, BxS] — pre-quantized hidden
        hidden_scale: torch uint8 [16, H//512, BxS] — hidden block scales
        weight_qtz: numpy float4_e2m1fn_x4 [128, H//512, I] — quantized weight
        weight_scale: torch uint8 [16, H//512, I] — weight block scales
        bias: torch float32 [bias_par_dim, ceil(I/512), 4]
        H, I, BxS: int — dimensions
        hidden_unpack_fn: optional unpack function for hidden (default: _unpack_float8_e5m2_x4)

    Returns: {"out": torch.Tensor float32} with shape [128, ceil(I/512), BxS, 4]
    """
    n_I512 = math.ceil(I / 512)
    unpack_hidden = hidden_unpack_fn or unpack_float8_e5m2_x4
    unpack_weight = weight_unpack_fn or unpack_float4_x4

    # Unpack x4 numpy -> torch at the boundary, then everything is torch
    w_flat = unpack_weight(weight_qtz.transpose(1, 0, 2).reshape(H // _q_width, I))
    h_flat = unpack_hidden(
        hidden_qtz.reshape(_pmax, H // _pmax // _q_width, BxS).transpose(1, 0, 2).reshape(H // _q_width, BxS)
    )
    w_sc = weight_scale.permute(1, 0, 2).reshape(H // _q_height // _q_width, I).float()
    h_sc = (
        hidden_scale.reshape(_pmax // _q_height, H // _pmax // _q_width, BxS)
        .permute(1, 0, 2)
        .reshape(H // _q_height // _q_width, BxS)
        .float()
    )

    # MX matmul: [H/4, I] x [H/4, BxS] -> [I, BxS]
    result = mx_matmul(w_flat, h_flat, w_sc, h_sc)

    # Reshape [I, BxS] -> tiled [128, n_I512, BxS, 4]
    out = torch.zeros(_pmax, n_I512, BxS, 4)
    for i in range(n_I512):
        rows_filled = 512 if not (i == n_I512 - 1 and I % 512 != 0) else I % 512
        cur = result[i * 512 : i * 512 + rows_filled, :]
        rows_padded = math.ceil(rows_filled / 8) * 8
        if rows_padded > rows_filled:
            cur = torch.nn.functional.pad(cur, (0, 0, 0, rows_padded - rows_filled))
        out[: rows_padded // 4, i, :, :] = cur.reshape(4, rows_padded // 4, BxS).permute(1, 2, 0)

    if bias is not None:
        bias_t = bias.float()
        if bias_t.shape[0] < _pmax:
            bias_t = torch.nn.functional.pad(bias_t, (0, 0, 0, 0, 0, _pmax - bias_t.shape[0]))
        out += bias_t.unsqueeze(2)

    return {"out": out}


def down_proj_mxfp4_torch_ref(
    inter: torch.Tensor,
    weight_qtz: np.ndarray,
    weight_scale: torch.Tensor,
    bias: torch.Tensor,
    H: int,
    I: int,
    BxS: int,
    use_stream_shuffle_broadcast: bool = True,
    weight_unpack_fn: callable = None,
) -> dict[str, torch.Tensor]:
    """PyTorch reference implementation of MXFP4 down projection.

    Computes: weight [I, H] @ intermediate [I, BxS] -> [BxS, H], with MX block-scaled
    quantized matmul. Intermediate is quantized to mxfp8 before the matmul.

    This is a reference implementation for testing the down_projection_mx_shard_H
    NKI kernel. It prioritizes correctness over performance.

    Args:
        inter: torch float32 [128, ceil(I/512), BxS, 4] — intermediate activation
        weight_qtz: numpy float4_e2m1fn_x4 [p_size, ceil(I/512), H] — quantized weight
        weight_scale: torch uint8 [p_size//8, ceil(I/512), H] — weight block scales
        bias: torch float32 [1, H]
        H, I, BxS: int — dimensions
        use_stream_shuffle_broadcast: bool — hardware optimization flag, ignored
            (does not affect numerical result)

    Returns: {"out": torch.Tensor float32} with shape [BxS, H]
    """
    I_padded_q = math.ceil(I / _q_width)

    # Flatten intermediate (torch) -> numpy for quantization
    inter_t = inter.float()
    if I < 512:
        inter_t = inter_t[:I_padded_q]
    inter_flat = inter_t.permute(1, 0, 2, 3).reshape(-1, BxS * _q_width).numpy()

    # Reshape weight (numpy x4) and scale (torch)
    weight_np = weight_qtz.transpose(1, 0, 2).reshape(-1, H)
    w_sc = weight_scale.permute(1, 0, 2).reshape(-1, H).float()

    # Align partition dim for MX block alignment
    P = inter_flat.shape[0]
    P_aligned = w_sc.shape[0] * _q_height
    if P < P_aligned:
        inter_flat = np.pad(inter_flat, ((0, P_aligned - P), (0, 0)))
    if weight_np.shape[0] < P_aligned:
        weight_np = np.pad(weight_np, ((0, P_aligned - weight_np.shape[0]), (0, 0)))
    else:
        weight_np = weight_np[:P_aligned]

    # Quantize intermediate to mxfp8, unpack both, then MX matmul
    inter_mx_np, inter_scale_np = quantize_to_mx_fp8(inter_flat)
    unpack_w = weight_unpack_fn or unpack_float4_x4
    result = mx_matmul(
        unpack_float8_e4m3fn_x4(inter_mx_np),
        unpack_w(weight_np),
        torch.from_numpy(inter_scale_np.astype(np.float32)),
        w_sc,
    )

    if bias is not None:
        result = result + bias.float().reshape(1, H)

    return {"out": result}
