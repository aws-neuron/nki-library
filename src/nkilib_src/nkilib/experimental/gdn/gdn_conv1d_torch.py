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

"""PyTorch reference implementations for the GDN depthwise causal conv1d kernels.

These mirror the qwen3_5 model_bf16.py torch ops EXACTLY (bias=False, silu,
depthwise groups=conv_dim, kernel=4):

- Prefill: model_bf16.py:1503-1527 (F.conv1d padding=K-1, groups=conv_dim,
  slice [:, :, :T], silu; state = last K REAL tokens of RAW mixed).
- Decode : model_bf16.py:_torch_causal_conv1d_update (lines 1189-1206),
  call site 1647-1653.
"""

import torch
import torch.nn.functional as F


def gdn_conv1d_prefill_torch_ref(mixed: torch.Tensor, conv_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Prefill depthwise causal conv1d + silu + conv_state update.

    Mirrors model_bf16.py:1503-1527 for the fresh-prefill (no padding_mask)
    case, where real_len == T so the state is simply the last K tokens of the
    RAW (pre-conv) mixed tensor.

    Args:
        mixed:       [B, conv_dim, T] RAW pre-conv input (q|k|v concatenated).
        conv_weight: [conv_dim, K] depthwise weights (== self.conv1d.weight
                     squeezed on the singleton in_channels/group dim).

    Returns:
        conv_out:       [B, conv_dim, T] silu(causal_conv1d(mixed)).
        new_conv_state: [B, conv_dim, K] last K RAW tokens of mixed.
    """
    B, conv_dim, T = mixed.shape
    K = conv_weight.shape[-1]
    w = conv_weight.reshape(conv_dim, 1, K)  # [conv_dim, 1, K] for groups=conv_dim
    conv_out = F.conv1d(
        mixed,
        w,
        bias=None,
        padding=K - 1,
        groups=conv_dim,
    )[:, :, :T]
    conv_out = F.silu(conv_out)
    new_conv_state = mixed[:, :, -K:].contiguous()
    return conv_out, new_conv_state


def gdn_conv1d_decode_torch_ref(
    x: torch.Tensor, conv_state: torch.Tensor, conv_weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode-step depthwise causal conv1d update + silu.

    Mirrors _torch_causal_conv1d_update (model_bf16.py:1189-1206).

    Args:
        x:           [B, conv_dim, 1] current-token RAW mixed.
        conv_state:  [B, conv_dim, K] previous window.
        conv_weight: [conv_dim, K] depthwise weights.

    Returns:
        out:            [B, conv_dim, 1] silu(conv1d(sliding window)).
        new_conv_state: [B, conv_dim, K] shifted window (== hs).
    """
    B, conv_dim, _ = x.shape
    K = conv_weight.shape[-1]
    hs = torch.cat([conv_state, x], dim=-1)[:, :, 1:]  # [B, conv_dim, K]
    new_conv_state = hs.contiguous()
    w = conv_weight.reshape(conv_dim, 1, K)
    out = F.conv1d(hs, w, bias=None, padding=0, groups=conv_dim)  # [B, conv_dim, 1]
    out = F.silu(out[:, :, -1:])
    return out, new_conv_state
