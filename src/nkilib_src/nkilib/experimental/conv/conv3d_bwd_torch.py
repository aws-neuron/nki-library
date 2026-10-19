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

"""PyTorch reference implementation for the 3D convolution backward kernel.

Produces the three gradients of a 3D convolution forward pass:
    y = conv3d(x, w) + b

    dx = grad w.r.t. input x      [B, C_in, D, H, W]
    dw = grad w.r.t. filters w    [K_d, K_h, K_w, C_in, C_out]
    db = grad w.r.t. bias b       [C_out]

The reference uses torch.autograd to provide ground truth, matching the
kernel's filter layout [K_d, K_h, K_w, C_in, C_out] (PyTorch conv weight layout is
[C_out, C_in, K_d, K_h, K_w]).
"""

from typing import Optional

import torch
import torch.nn.functional as F


def conv3d_dx_torch_ref(
    dy: torch.Tensor,
    x_in: torch.Tensor,
    filters: torch.Tensor,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    sbm: Optional[object] = None,
    use_auto_allocation: bool = False,
) -> torch.Tensor:
    """
    PyTorch reference for the input gradient dx of a 3D convolution.

    Args:
        dy (torch.Tensor): Output gradient of shape [B, C_out, D_out, H_out, W_out].
        x_in (torch.Tensor): Forward input of shape [B, C_in, D, H, W].
        filters (torch.Tensor): Forward filters of shape [K_d, K_h, K_w, C_in, C_out].
        stride (tuple[int, int, int]): Forward convolution strides.
        padding (tuple[int, int, int, int, int, int]): Flattened forward padding
            (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right).
        dilation (tuple[int, int, int]): Forward dilation factors.
        sbm (Optional[object]): Unused. Present only for signature parity with the NKI
            kernel's SBUF-manager parameter.
        use_auto_allocation (bool): Unused. Present only for signature parity with the
            NKI kernel.

    Returns:
        torch.Tensor: Input gradient dx of shape [B, C_in, D, H, W].
    """
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
    asymmetric = pad_d_left != pad_d_right or pad_h_top != pad_h_bottom or pad_w_left != pad_w_right

    x = x_in.detach().clone().requires_grad_(True)
    w = filters.detach().clone().permute(4, 3, 0, 1, 2).contiguous()

    if asymmetric:
        x_padded = F.pad(
            x, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_left, pad_d_right), mode="constant", value=0
        )
        conv_padding = (0, 0, 0)
    else:
        x_padded = x
        conv_padding = (pad_d_left, pad_h_top, pad_w_left)

    y = F.conv3d(x_padded, w, bias=None, stride=stride, padding=conv_padding, dilation=dilation)

    (dx,) = torch.autograd.grad(outputs=y, inputs=x, grad_outputs=dy, retain_graph=False)
    return dx.detach()


def conv3d_dw_torch_ref(
    dy: torch.Tensor,
    x_in: torch.Tensor,
    filters: torch.Tensor,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    sbm: Optional[object] = None,
    use_auto_allocation: bool = False,
) -> torch.Tensor:
    """
    PyTorch reference for the filter gradient dw of a 3D convolution.

    Args:
        dy (torch.Tensor): Output gradient of shape [B, C_out, D_out, H_out, W_out].
        x_in (torch.Tensor): Forward input of shape [B, C_in, D, H, W].
        filters (torch.Tensor): Forward filters of shape [K_d, K_h, K_w, C_in, C_out].
        stride (tuple[int, int, int]): Forward convolution strides.
        padding (tuple[int, int, int, int, int, int]): Flattened forward padding
            (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right).
        dilation (tuple[int, int, int]): Forward dilation factors.
        sbm (Optional[object]): Unused. Present only for signature parity with the NKI
            kernel's SBUF-manager parameter.
        use_auto_allocation (bool): Unused. Present only for signature parity with the
            NKI kernel.

    Returns:
        torch.Tensor: Filter gradient dw of shape [K_d, K_h, K_w, C_in, C_out].
    """
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding
    asymmetric = pad_d_left != pad_d_right or pad_h_top != pad_h_bottom or pad_w_left != pad_w_right

    x = x_in.detach().clone()
    w = filters.detach().clone().permute(4, 3, 0, 1, 2).contiguous().requires_grad_(True)

    if asymmetric:
        x_padded = F.pad(
            x, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_left, pad_d_right), mode="constant", value=0
        )
        conv_padding = (0, 0, 0)
    else:
        x_padded = x
        conv_padding = (pad_d_left, pad_h_top, pad_w_left)

    y = F.conv3d(x_padded, w, bias=None, stride=stride, padding=conv_padding, dilation=dilation)

    (dw,) = torch.autograd.grad(outputs=y, inputs=w, grad_outputs=dy, retain_graph=False)
    # Convert PyTorch weight-grad layout [C_out, C_in, K_d, K_h, K_w] to kernel layout.
    return dw.permute(2, 3, 4, 1, 0).contiguous().detach()


def conv3d_db_torch_ref(
    dy: torch.Tensor,
    sbm: Optional[object] = None,
    use_auto_allocation: bool = False,
) -> torch.Tensor:
    """
    PyTorch reference for the bias gradient db of a 3D convolution.

    Matches the conv3d_db kernel signature, which takes only the output gradient dy
    (plus SBUF-manager plumbing). db is independent of x_in / filters / stride /
    padding / dilation.

    Args:
        dy (torch.Tensor): Output gradient of shape [B, C_out, D_out, H_out, W_out].
        sbm (Optional[object]): Unused. Present only for signature parity with the NKI
            kernel's SBUF-manager parameter.
        use_auto_allocation (bool): Unused. Present only for signature parity with the
            NKI kernel.

    Returns:
        torch.Tensor: Bias gradient db of shape [C_out].
    """
    # db is the sum of dy over the batch and all output spatial positions.
    return dy.detach().sum(dim=(0, 2, 3, 4))


def conv3d_bwd_torch_ref(
    dy: torch.Tensor,
    x_in: torch.Tensor,
    filters: torch.Tensor,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    compute_dx: bool = True,
    compute_dw: bool = True,
    compute_db: bool = True,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of the 3D convolution backward pass.

    Args:
        dy (torch.Tensor): Output gradient of shape [B, C_out, D_out, H_out, W_out].
        x_in (torch.Tensor): Forward input of shape [B, C_in, D, H, W].
        filters (torch.Tensor): Forward filters of shape [K_d, K_h, K_w, C_in, C_out].
        stride (tuple[int, int, int]): Forward convolution strides.
        padding (tuple[int, int, int, int, int, int]): Flattened forward padding
            (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right).
        dilation (tuple[int, int, int]): Forward dilation factors.
        compute_dx (bool): Whether to compute the input gradient dx.
        compute_dw (bool): Whether to compute the filter gradient dw.
        compute_db (bool): Whether to compute the bias gradient db.

    Returns:
        dict[str, torch.Tensor]: Subset of {"dx", "dw", "db"} per the compute flags:
            - "dx": [B, C_in, D, H, W]
            - "dw": [K_d, K_h, K_w, C_in, C_out] (kernel filter layout)
            - "db": [C_out]
    """
    result: dict[str, torch.Tensor] = {}
    if compute_dx:
        result["dx"] = conv3d_dx_torch_ref(dy, x_in, filters, stride, padding, dilation)
    if compute_dw:
        result["dw"] = conv3d_dw_torch_ref(dy, x_in, filters, stride, padding, dilation)
    if compute_db:
        result["db"] = conv3d_db_torch_ref(dy)
    return result
