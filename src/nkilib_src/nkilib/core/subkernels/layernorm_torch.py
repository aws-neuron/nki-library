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

"""PyTorch reference implementations for the layernorm_tkg kernel."""

from typing import Optional

import torch


def layer_norm_torch_ref(
    hidden: torch.Tensor,
    gamma: Optional[torch.Tensor],
    norm_b: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    **_,
) -> torch.Tensor:
    """
    PyTorch reference implementation of Layer normalization.

    Args:
        hidden (torch.Tensor): Input tensor to normalize.
        gamma (torch.Tensor or None): Scale parameter.
        norm_b (torch.Tensor or None): Bias parameter.
        eps (float): Epsilon for numerical stability.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    # All intermediates need to happen in FP32 for numerical precision
    hidden = hidden.to(torch.float32)

    mean = hidden.mean(dim=-1, keepdim=True)
    var = hidden.var(dim=-1, correction=0, keepdim=True)

    norm = (hidden - mean) * (var + eps).sqrt().reciprocal().to(hidden.dtype)
    if gamma is not None:
        norm *= gamma
    if norm_b is not None:
        norm += norm_b
    return norm


def layernorm_tkg_torch_ref_lnc1(
    input: torch.Tensor,
    gamma: torch.Tensor,
    output: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    use_heap_memory: bool = False,
    sbm: Optional[object] = None,
) -> dict[str, torch.Tensor]:
    """Torch reference for layernorm_tkg kernel (LNC1 output layout).

    This is a reference implementation for testing the NKI layernorm_tkg kernel.
    It applies Layer normalization and reshapes the output into the LNC1 tile layout.

    Args:
        input (torch.Tensor): [B, S, H] input hidden states.
        gamma (torch.Tensor): [1, H] LayerNorm weight vector.
        output (torch.Tensor): [128, B*S, H//128] output buffer. Unused, present for interface compatibility.
        beta (torch.Tensor or None): [1, H] LayerNorm bias vector.
        eps (float): Epsilon for numerical stability.
        use_heap_memory (bool): Unused, present for interface compatibility.
        sbm: Unused, present for interface compatibility.

    Returns:
        dict: {"out": torch.Tensor} with shape [128, B*S, H//128].

    Note:
        Hardware-specific parameters (use_heap_memory, sbm) are accepted but ignored
        as they don't affect the mathematical result.
    """
    B, S, H = input.shape
    BxS = B * S
    H0, H1 = 128, H // 128
    dtype = input.dtype

    result = layer_norm_torch_ref(input, gamma, norm_b=beta, eps=eps)
    result = result.reshape(BxS, H0, H1).permute(1, 0, 2)

    return {"out": result.to(dtype)}


def layernorm_tkg_torch_ref(
    input: torch.Tensor,
    gamma: torch.Tensor,
    output: torch.Tensor,
    beta: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
    use_heap_memory: bool = False,
    sbm: Optional[object] = None,
) -> dict[str, torch.Tensor]:
    """Torch reference for layernorm_tkg kernel (LNC2 output layout).

    This is a reference implementation for testing the NKI layernorm_tkg kernel.
    It applies Layer normalization and reshapes the output into the LNC2 tile layout,
    which interleaves two halves of the hidden dimension.

    Args:
        input (torch.Tensor): [B, S, H] input hidden states.
        gamma (torch.Tensor): [1, H] LayerNorm weight vector.
        output (torch.Tensor): [128, B*S, H//128] output buffer. Unused, present for interface compatibility.
        beta (torch.Tensor or None): [1, H] LayerNorm bias vector.
        eps (float): Epsilon for numerical stability.
        use_heap_memory (bool): Unused, present for interface compatibility.
        sbm: Unused, present for interface compatibility.

    Returns:
        dict: {"out": torch.Tensor} with shape [128, B*S, H//128].

    Note:
        Hardware-specific parameters (use_heap_memory, sbm) are accepted but ignored
        as they don't affect the mathematical result.
    """
    B, S, H = input.shape
    BxS = B * S
    H0, H1 = 128, H // 128
    dtype = input.dtype

    result = layer_norm_torch_ref(input, gamma, norm_b=beta, eps=eps)
    result = result.reshape(BxS, -1)

    # LNC2: interleave two halves
    t0 = result[:, 0 : H // 2]
    t1 = result[:, H // 2 :]
    t0 = t0.reshape(BxS, H0, H1 // 2).permute(1, 0, 2)
    t1 = t1.reshape(BxS, H0, H1 // 2).permute(1, 0, 2)
    result = torch.cat([t0, t1], dim=2)

    return {"out": result.to(dtype)}
