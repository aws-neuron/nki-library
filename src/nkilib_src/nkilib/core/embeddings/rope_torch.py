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

import torch


def rope_torch_ref(
    x_in: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    lnc_shard: bool = False,
    contiguous_layout: bool = True,
    relayout_in_sbuf: bool = False,
) -> dict:
    """Torch reference for RoPE kernel.

    Args:
        x_in: [d_head, B, n_heads, S]
        cos: [d_head//2, B, S]
        sin: [d_head//2, B, S]
        lnc_shard, contiguous_layout, relayout_in_sbuf: kernel config flags
    """
    d_head, B, n_heads, S = x_in.shape

    if d_head not in (64, 128):
        raise ValueError(f"[NCC_INKI016] Kernel validation exception: d_head must be 64 or 128, got {d_head}")

    x_out = torch.empty_like(x_in)
    for b in range(B):
        for h in range(n_heads):
            x_out[:, b, h, :] = _rope_single_head(x_in[:, b, h, :], cos[:, b, :], sin[:, b, :], contiguous_layout)

    return {"x_out": x_out}


def _rope_single_head(x_in, cos, sin, contiguous_layout):
    """Apply RoPE to single head: [d_head, S]."""
    d_head = x_in.shape[0]
    x = x_in.T  # [d_head, S] -> [S, d_head]

    if contiguous_layout:
        new_x = torch.empty_like(x)
        new_x[:, ::2] = x[:, : d_head // 2]
        new_x[:, 1::2] = x[:, d_head // 2 :]
        x = new_x

    freqs_cos = cos.T  # [half_d, S] -> [S, half_d]
    freqs_sin = sin.T

    xri = x.reshape(x.shape[:-1] + (-1, 2))
    x_r, x_i = xri[..., 0], xri[..., 1]

    x_out_r = x_r * freqs_cos - x_i * freqs_sin
    x_out_i = x_r * freqs_sin + x_i * freqs_cos

    x_out = torch.stack([x_out_r, x_out_i], dim=-1).reshape(x.shape)

    if contiguous_layout:
        x_out = torch.cat((x_out[:, 0::2], x_out[:, 1::2]), dim=1)

    return x_out.T  # [S, d_head] -> [d_head, S]
