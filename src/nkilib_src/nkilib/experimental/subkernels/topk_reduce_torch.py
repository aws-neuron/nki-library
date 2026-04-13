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

"""PyTorch reference implementation for topk_reduce kernel."""

import torch


def topk_reduce_torch_ref(input: torch.Tensor, T: int, K: int) -> torch.Tensor:
    """Gather scattered rows by packed global token index and reduce along K.

    Args:
        input: (TK_padded, H+2) tensor with hidden states and packed int32 token indices
        T: number of output tokens
        K: number of top-K entries per token

    Returns:
        (T, H) tensor of reduced hidden states
    """
    H = input.shape[1] - 2

    # Extract packed int32 global token indices from last 2 bf16 columns
    idx_bf16 = input[:, H:].to(torch.bfloat16).contiguous()
    global_token_indices = idx_bf16.view(torch.int32).squeeze(-1)

    # For each token, gather matching rows and sum
    out = torch.zeros(T, H, dtype=input.dtype)
    for t in range(T):
        mask = global_token_indices == t
        out[t] = input[mask, :H].sum(dim=0)

    return out
