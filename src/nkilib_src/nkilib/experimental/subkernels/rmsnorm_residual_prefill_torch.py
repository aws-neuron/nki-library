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

"""PyTorch reference implementation for rmsnorm_residual_prefill kernel."""

from typing import Dict, Optional

import torch


def rmsnorm_residual_prefill_torch_ref(
    hidden: torch.Tensor,
    gamma: torch.Tensor,
    residual: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    hidden_actual: Optional[int] = None,
) -> Dict[str, torch.Tensor]:
    """Optional pre-norm residual add then RMSNorm. Keys match output_tensor_descriptor.

    Outputs are flattened to [T, H] to match the kernel, which folds a [B, S, H] input
    into [B*S, H] and returns that rank regardless of the input rank.
    """
    x = hidden.float() if residual is None else hidden.float() + residual.float()
    n = x.shape[-1] if hidden_actual is None else hidden_actual
    rms = torch.sqrt((x * x).sum(dim=-1, keepdim=True) / n + eps)
    out = {"out": ((x / rms) * gamma.float()).reshape(-1, x.shape[-1])}
    if residual is not None:
        out["residual_out"] = x.reshape(-1, x.shape[-1])
    return out
