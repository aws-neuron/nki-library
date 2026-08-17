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
"""PyTorch reference implementation for the pad kernel."""

import torch


def pad_torch_ref(x_ref, padding, mode="replicate", value=0):
    """Torch reference for pad.

    Equivalent to ``torch.nn.functional.pad(x, padding, mode=mode, value=value)``.

    Args:
        x_ref: Input tensor.
        padding: Padding amounts in PyTorch convention (innermost-first).
        mode: "constant", "replicate", "reflect", or "circular".
        value: Fill value for constant mode (default 0).

    Returns:
        Padded output tensor.
    """
    return torch.nn.functional.pad(x_ref, padding, mode=mode, value=value)
