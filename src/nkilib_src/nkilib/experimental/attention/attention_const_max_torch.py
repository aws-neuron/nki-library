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
Pytorch reference for attention_const_max kernel
"""

import torch.nn.functional as F


def attention_const_max_torch_ref(q_hbm, k_hbm, v_hbm, softmax_max, softmax_scale=None):
    """Reference implementation using torch SDPA — mathematically equivalent.

    The softmax_max parameter does not affect the output of a correctly-implemented
    constant-max attention (the output is invariant to the choice of max), so this
    reference simply uses torch's SDPA which computes the exact same result.

    Args:
        q_hbm: [N, d, Sq] bf16 tensor
        k_hbm: [N, d, Sk] bf16 tensor
        v_hbm: [N, Sk, d] bf16 tensor
        softmax_max: unused (output is invariant to max choice)
        softmax_scale: scaling factor, defaults to 1/sqrt(d)

    Returns:
        [N, Sq, d] tensor
    """
    d = q_hbm.shape[1]
    if softmax_scale is None:
        softmax_scale = d ** (-0.5)

    q_t = q_hbm.float().transpose(-2, -1).unsqueeze(0)  # (1, N, Sq, d)
    k_t = k_hbm.float().transpose(-2, -1).unsqueeze(0)  # (1, N, Sk, d)
    v_t = v_hbm.float().unsqueeze(0)  # (1, N, Sk, d)
    out = F.scaled_dot_product_attention(q_t, k_t, v_t, scale=softmax_scale)
    return out.squeeze(0)  # (N, Sq, d)
