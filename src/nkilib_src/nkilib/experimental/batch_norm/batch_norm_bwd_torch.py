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

"""PyTorch reference implementation for the batch normalization backward kernel."""

from typing import Optional

import torch


def batch_norm_bwd_torch_ref(
    dy: torch.Tensor,
    x_in: torch.Tensor,
    means: torch.Tensor,
    variances: torch.Tensor,
    gamma: torch.Tensor,
    beta: torch.Tensor,
    eps: float,
    fuse_relu: Optional[bool] = False,
    relu_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of the batch normalization backward kernel.

    Signature mirrors batch_norm_bwd. Rather than hand-deriving the gradients, this
    reconstructs the batchnorm forward pass (plus the fused ReLU) and differentiates it
    with torch.autograd.grad, so the golden is defined by the forward math alone.

    Args:
        dy (torch.Tensor): [B, C, D, H, W], gradient w.r.t. the batchnorm output.
        x_in (torch.Tensor): [B, C, D, H, W], forward-pass input to the batchnorm.
        means (torch.Tensor): [C], per-channel mean saved by the forward pass.
        variances (torch.Tensor): [C], per-channel biased variance (correction 0) saved by
            the forward pass.
        gamma (torch.Tensor): [C, 1], per-channel batchnorm scale.
        beta (torch.Tensor): [C, 1], per-channel batchnorm shift. Its value only affects
            the result when fuse_relu is True and relu_mask is None, where it decides
            where the fused ReLU clipped; dbeta itself does not depend on it. Note the
            forward output y below is built with beta, so (y > 0) already accounts for it.
        eps (float): Added to the saved variance before the reciprocal square root. Required with
            no default, unlike the kernel's: restating the kernel's default here would let the two
            drift silently, since no test exercises either default (the test always passes eps).
        fuse_relu (Optional[bool]): Whether the forward pass applied a ReLU after the
            batchnorm.
        relu_mask (Optional[torch.Tensor]): [B, C, D, H, W], uint8 mask of the fused ReLU
            (1 where it passed its input through, 0 where it clipped). Optional even when
            fuse_relu is True: omitting it recomputes the mask from the forward output, which
            is what the kernel does in that case too.

    Returns:
        dict[str, torch.Tensor]: keys "dx" [B, C, D, H, W], "dgamma" [C, 1] and
            "dbeta" [C, 1].
    """
    C = dy.shape[1]
    # Per-channel params broadcast over the batch and spatial dims.
    view_shape = (1, C, 1, 1, 1)
    reduce_dims = (0, 2, 3, 4)

    # Leaves of the autograd graph, in float32 (the kernel's accumulation dtype).
    x = x_in.float().detach().requires_grad_(True)
    g = gamma.float().reshape(C, 1).detach().requires_grad_(True)
    b = beta.float().reshape(C, 1).detach().requires_grad_(True)

    # Batchnorm forward. Adding a batch statistic minus its detached self is a no-op numerically but
    # keeps d(mean)/dx and d(var)/dx in the graph; without them dx misses the 1/N correction terms.
    batch_mean = x.mean(dim=reduce_dims, keepdim=True)
    batch_var = x.var(dim=reduce_dims, correction=0, keepdim=True)
    mean = means.float().reshape(view_shape) + (batch_mean - batch_mean.detach())
    var = variances.float().reshape(view_shape) + (batch_var - batch_var.detach())
    y = (x - mean) * torch.rsqrt(var + eps) * g.reshape(view_shape) + b.reshape(view_shape)

    grad_out = dy.float()
    if fuse_relu:
        # d(relu(y))/dy is the mask of where the forward ReLU passed its input through.
        mask = (y > 0) if relu_mask == None else relu_mask.float()
        grad_out = grad_out * mask

    dx, dgamma, dbeta = torch.autograd.grad(y, (x, g, b), grad_outputs=grad_out)

    return {
        "dx": dx.to(dy.dtype).detach(),
        "dgamma": dgamma.detach(),
        "dbeta": dbeta.detach(),
    }
