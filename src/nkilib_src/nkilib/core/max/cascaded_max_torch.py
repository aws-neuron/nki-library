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


def cascaded_max_torch_ref(input_tensor: torch.Tensor) -> dict[str, torch.Tensor]:
    """Cascaded max torch reference implementation.

    Args:
        input_tensor: Input tensor of shape [..., vocab_size]

    Returns:
        dict with keys:
            - max_values: shape [..., 1] containing max values
            - max_indices: shape [..., 1] containing indices of max values
    """
    max_values, max_indices = torch.max(input_tensor, dim=-1, keepdim=True)
    return {"max_values": max_values, "max_indices": max_indices.to(torch.int32)}
