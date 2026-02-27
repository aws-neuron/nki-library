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

"""PyTorch reference implementation for cross entropy loss kernel."""

import torch
import torch.nn.functional as F


def cross_entropy_forward_torch_ref(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    PyTorch reference implementation of cross entropy forward pass.

    This is a reference implementation for testing the NKI cross_entropy_forward kernel.
    It implements the same mathematical operation using PyTorch operations.

    Args:
        logits: Input logits tensor of shape [num_positions, vocab_size]
        targets: Target indices tensor of shape [num_positions]

    Returns:
        loss: Cross entropy loss per position [num_positions]
        lse: Log-sum-exp values per position [num_positions]

    Note:
        This implementation prioritizes clarity over performance.
        Hardware-specific parameters (positions_per_batch, chunk_size, etc.)
        are not included as they don't affect the mathematical result.
    """
    # Compute cross entropy loss (no reduction to get per-position loss)
    loss = F.cross_entropy(logits, targets, reduction='none')

    # Compute log-sum-exp for backward pass
    lse = torch.logsumexp(logits, dim=1)

    return loss, lse


def cross_entropy_backward_torch_ref(
    logits: torch.Tensor,
    targets: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """
    PyTorch reference implementation of cross entropy backward pass.

    This is a reference implementation for testing the NKI cross_entropy_backward kernel.
    It uses PyTorch's autograd to compute the gradient of cross entropy loss with respect
    to logits, which is guaranteed to be correct.

    Args:
        logits: Input logits tensor of shape [num_positions, vocab_size]
        targets: Target indices tensor of shape [num_positions]
        reduction: How to reduce the loss. Options:
            - 'mean': Average loss over all positions (most common, matches PyTorch default)
            - 'sum': Sum loss over all positions

    Returns:
        grad_logits: Gradient with respect to logits [num_positions, vocab_size]

    Note:
        This implementation uses PyTorch's autograd for correctness and simplicity.
        Hardware-specific parameters (positions_per_batch, chunk_size, etc.)
        are not included as they don't affect the mathematical result.
    """
    if reduction not in ("mean", "sum"):
        raise ValueError(f"Unknown reduction: {reduction}. Use 'mean' or 'sum'.")

    # Create a copy of logits with gradient tracking enabled
    logits_copy = logits.detach().clone().requires_grad_(True)

    # Forward pass: compute cross entropy loss with specified reduction
    loss = F.cross_entropy(logits_copy, targets, reduction=reduction)

    # Backward pass: compute gradients
    loss.backward()

    return logits_copy.grad
