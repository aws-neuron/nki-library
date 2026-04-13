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


def topk_torch_ref(inp: torch.Tensor, config) -> dict[str, torch.Tensor]:
    """TopK torch reference implementation.

    Args:
        inp: Input tensor of shape [BxS, vocab_size]
        config: RotationalTopkConfig with k and sorted attributes

    Returns:
        dict with keys:
            - topk_values: shape [BxS, k] containing topk values
            - topk_indices: shape [BxS, k] containing indices (not validated for exact match)

    Note:
        topk_indices may differ from NKI due to tie-breaking. Validation focuses on
        topk_values correctness, which implicitly validates indices functionality.
    """
    k = config.topk_config.k
    sorted_output = config.topk_config.sorted

    values, indices = torch.topk(inp, k=k, dim=-1, largest=True, sorted=sorted_output)
    return {"topk_values": values, "topk_indices": indices.to(torch.int32)}
