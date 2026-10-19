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
"""Torch reference for the FlashAttention-2 forward kernel."""

import torch


def flash_attention_2_torch_ref(q, k, v, scale, config=None):
    """Reference for ``O = softmax(Q @ K^T * scale) @ V`` with native GQA.

    q: [B, Sq, D]; k: [Bkv, D, Sk]; v: [Bkv, Sk, D] -> out: [B, Sq, D]

    Deliberately an explicit matmul-softmax-matmul rather than ``scaled_dot_product_attention``.
    SDPA would avoid materializing the ``[Sq, Sk]`` scores, but requires a transposed K [Bkv, Sk, D].
    Resulting in an extra copy that turns out to cost more than a naiive sdpa
    """
    q_heads_group_size = q.shape[0] // k.shape[0]
    k_expanded = torch.repeat_interleave(k.float(), q_heads_group_size, dim=0)
    v_expanded = torch.repeat_interleave(v.float(), q_heads_group_size, dim=0)

    scores = torch.matmul(q.float(), k_expanded) * scale
    probs = torch.softmax(scores, dim=-1)
    return torch.matmul(probs, v_expanded)
