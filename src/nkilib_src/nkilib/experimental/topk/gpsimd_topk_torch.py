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
"""PyTorch reference implementation for the GpSIMD top-k kernel."""

import torch


def gpsimd_topk_torch_ref(inp: torch.Tensor, config) -> dict[str, torch.Tensor]:
    """Top-K torch reference for gpsimd_topk.

    Computes torch.topk over the last dimension, matching the gpsimd_topk kernel
    interface. Uses config.topk_config.k and config.topk_config.sorted.

    Args:
        inp: Input tensor of shape [BxS, vocab_size].
        config: GpsimdTopkConfig with topk_config property exposing k and sorted.

    Returns:
        dict with keys:
            - topk_values: shape [BxS, k] containing top-k values.
            - topk_indices: shape [BxS, k] containing vocab indices (uint32).
    """
    k = config.topk_config.k
    sorted_output = config.topk_config.sorted

    values, indices = torch.topk(inp, k=k, dim=-1, largest=True, sorted=sorted_output)
    return {"topk_values": values, "topk_indices": indices.to(torch.int32)}
