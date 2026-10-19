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

"""PyTorch reference implementation for conv_fuser for targeted conv3d testing."""

from typing import Optional

import torch
import torch.nn as nn

from ...core.utils.allocator import SbufManager
from ...core.utils.common_types import ActFnType
from ...core.utils.kernel_assert import kernel_assert
from .conv3d import BatchNormMode, ResidualAddLoc
from .conv3d_torch import _apply_activation_and_residual


def conv_fuser_torch_ref(
    x_in: torch.Tensor,
    filters: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: tuple[int, int, int] = (1, 1, 1),
    padding: tuple[int, int, int, int, int, int] = (0, 0, 0, 0, 0, 0),
    dilation: tuple[int, int, int] = (1, 1, 1),
    activation_fn: Optional[ActFnType] = None,
    lnc_shard: bool = False,
    batch_norm_mode: BatchNormMode = BatchNormMode.NONE,
    output_pre_norm: Optional[bool] = False,
    batch_norm_eps: Optional[float] = 1e-5,
    gamma: Optional[torch.Tensor] = None,
    beta: Optional[torch.Tensor] = None,
    momentum: Optional[float] = 0.1,
    running_means: Optional[torch.Tensor] = None,
    running_variances: Optional[torch.Tensor] = None,
    sbm: Optional[SbufManager] = None,
    use_auto_allocation: bool = False,
    residual_add_loc: ResidualAddLoc = ResidualAddLoc.NONE,
    residuals_in: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """
    PyTorch reference implementation of ConvFuser model.
    Args match conv3d and are checked for match with a ConvFuser config.

    Args:
        x_in (torch.Tensor): Input tensor of shape [B, C_in, D, H, W]
        filters (torch.Tensor): Filter weights of shape [K_d, K_h, K_w, C_in, C_out]
        bias (Optional[torch.Tensor]): Optional bias tensor of shape [C_out]
        stride (tuple[int, int, int]): Stride for convolution. Default (1, 1, 1).
        padding (tuple[int, int, int, int, int, int]):
            Flattened padding tuple (pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right).
            Default (0, 0, 0, 0, 0, 0).
        dilation (tuple[int, int, int]): Dilation factor for dilated convolution. Default (1, 1, 1).
        activation_fn (Optional[ActFnType]): Optional activation function type. Default None.
        lnc_shard (bool): Whether LNC sharding is enabled (unused in reference). Default False.
        residual_add_loc (ResidualAddLoc): Where residuals_in is added relative to the ReLU: NONE
            (not at all), PRE_ACT (before) or POST_ACT (after). Default ResidualAddLoc.NONE.
        residuals_in (Optional[torch.Tensor]): [B, C_out, 1, H_out, W_out] residual tensor added to
            the output. Required when residual_add_loc is not NONE. Default None.

    Returns:
        dict[str, torch.Tensor]: Dictionary with key "out" containing output tensor of shape
            [B, C_out, D_out, H_out, W_out]
    """

    K_d, K_h, K_w, C_in, C_out = filters.shape
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding

    # Check that this matches the ConvFuser 3x3 2D convolution with a padding of 1 and bias=False
    kernel_assert(
        K_d == 1 and K_h == 3 and K_w == 3, f"Filter must have K_d, K_h, K_w == (1, 3, 3), got {K_d}, {K_h}, {K_w}"
    )
    kernel_assert(stride == (1, 1, 1), f"stride must be (1, 1, 1), got {stride}")
    kernel_assert(padding == (0, 0, 1, 1, 1, 1), f"padding must be (0, 0, 1, 1, 1, 1), got {padding}")
    kernel_assert(dilation == (1, 1, 1), f"dilation must be (1, 1, 1), got {dilation}")
    kernel_assert(activation_fn == ActFnType.ReLU, f"Activation function must be relu, got {activation_fn}")
    kernel_assert(batch_norm_mode.is_fused(), f"batch_norm_mode must be EVAL or TRAINING, got {batch_norm_mode}")
    kernel_assert(bias == None, f"bias must be None, got {bias}")

    # Create the convfuser layers
    conv = nn.Conv2d(C_in, C_out, kernel_size=3, stride=1, padding=1, dilation=1, bias=False)
    norm = nn.BatchNorm2d(C_out)

    # Set weights/params
    # For conv weights: transpose from [K_d, K_h, K_w, C_in, C_out] to [C_out, C_in, K_d, K_h, K_w] and squeeze out K_d
    with torch.no_grad():
        conv.weight.copy_(filters.permute(4, 3, 0, 1, 2).squeeze(2))
        norm.eps = batch_norm_eps
        norm.weight.copy_(gamma.squeeze(1))
        norm.bias.copy_(beta.squeeze(1))
        norm.momentum = momentum
        norm.running_mean.copy_(running_means.squeeze(1))
        norm.running_var.copy_(running_variances.squeeze(1))

    if batch_norm_mode == BatchNormMode.EVAL:
        norm.eval()

    # Perform convolution
    conv_output = conv(x_in.squeeze(2))
    orig_dtype = conv_output.dtype
    norm_output = norm(conv_output.float())

    # residuals_in is [B, C_out, D_out=1, H_out, W_out]; drop the D axis to match the 2D pipeline.
    # The ReLU + residual placement is shared with conv3d_torch_ref so both refs encode the
    # PRE_ACT / POST_ACT contract once (activation_fn is asserted to be ReLU above).
    residual_2d = residuals_in.squeeze(2) if residuals_in != None else None
    relu_output = _apply_activation_and_residual(
        norm_output.to(orig_dtype), activation_fn, residual_add_loc, residual_2d
    )

    if batch_norm_mode == BatchNormMode.EVAL:
        return {"out": relu_output.detach()}

    means = conv_output.float().mean(dim=(0, 2, 3))
    variances = conv_output.float().var(dim=(0, 2, 3), correction=0)
    running_means = norm.running_mean.unsqueeze(1)
    running_variances = norm.running_var.unsqueeze(1)

    # Key insertion order must match the kernel's returned tuple; output_pre_norm inserts the raw
    # pre-batchnorm conv output right after "out".
    outputs = {"out": relu_output.detach()}
    if output_pre_norm:
        outputs["conv_out"] = conv_output.detach()
    outputs["means"] = means.detach()
    outputs["variances"] = variances.detach()
    outputs["updated_running_means"] = running_means.detach()
    outputs["updated_running_variances"] = running_variances.detach()
    return outputs
