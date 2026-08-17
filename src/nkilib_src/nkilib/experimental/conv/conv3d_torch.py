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

"""PyTorch reference implementation for 3D convolution kernel testing."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...core.utils.allocator import SbufManager
from ...core.utils.common_types import ActFnType
from .conv3d import BatchNormMode, ResidualAddLoc


def _apply_activation(output: torch.Tensor, activation_fn: ActFnType) -> torch.Tensor:
    """Apply the given activation function to output, matching the kernel's ActFnType map."""
    if activation_fn == ActFnType.SiLU:
        return F.silu(output)
    elif activation_fn == ActFnType.GELU:
        return F.gelu(output)
    elif activation_fn == ActFnType.GELU_Tanh_Approx:
        return F.gelu(output, approximate="tanh")
    elif activation_fn == ActFnType.Swish:
        return F.silu(output)
    elif activation_fn == ActFnType.ReLU:
        return F.relu(output)
    else:
        raise ValueError(f"Unsupported activation function: {activation_fn}")


def _apply_activation_and_residual(
    output: torch.Tensor,
    activation_fn: Optional[ActFnType],
    residual_add_loc: ResidualAddLoc,
    residuals_in: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Apply the optional activation function and the optional residual add on the requested side of it.

    With no activation function PRE_ACT and POST_ACT are equivalent, matching the kernel.
    """
    if residual_add_loc == ResidualAddLoc.PRE_ACT:
        output = output + residuals_in
    if activation_fn != None:
        output = _apply_activation(output, activation_fn)
    if residual_add_loc == ResidualAddLoc.POST_ACT:
        output = output + residuals_in
    return output


def conv3d_torch_ref(
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
    PyTorch reference implementation of 3D convolution kernel.

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
        lnc_shard (bool): Does nothing. Will be deprecated next release.
        batch_norm_mode (BatchNormMode): Fused-batchnorm mode. NONE applies no batchnorm. TRAINING
            computes the batch statistics, normalizes with them and momentum-updates the running
            statistics, matching torch.nn.BatchNorm2d in train(). EVAL normalizes with the given
            running_means / running_variances and does not momentum-update them, matching
            torch.nn.BatchNorm2d in eval(); only "out" is returned. Default BatchNormMode.NONE.
        output_pre_norm (Optional[bool]): When True (with BatchNormMode.TRAINING), additionally
            return the raw pre-batchnorm convolution output under "conv_out". Default False.
        residual_add_loc (ResidualAddLoc): Where residuals_in is added: NONE (not at all), PRE_ACT
            (before the activation function) or POST_ACT (after it). Equivalent when activation_fn
            is None. Default ResidualAddLoc.NONE.
        residuals_in (Optional[torch.Tensor]): [B, C_out, D_out, H_out, W_out] residual tensor
            added to the output. Required when residual_add_loc is not NONE. Default None.
        sbm (Optional[SbufManager]): Unused in the reference; accepted so the signature matches the
            kernel's. Default None.
        use_auto_allocation (bool): Unused in the reference; accepted for signature parity with the
            kernel. Default False.

    Returns:
        dict[str, torch.Tensor]: Dictionary with key "out" containing output tensor of shape
            [B, C_out, D_out, H_out, W_out]
    """
    K_d, K_h, K_w, C_in, C_out = filters.shape
    pad_d_left, pad_d_right, pad_h_top, pad_h_bottom, pad_w_left, pad_w_right = padding

    # Handle asymmetric padding by manually padding the input
    if pad_d_left != pad_d_right or pad_h_top != pad_h_bottom or pad_w_left != pad_w_right:
        x_in = F.pad(
            x_in, (pad_w_left, pad_w_right, pad_h_top, pad_h_bottom, pad_d_left, pad_d_right), mode="constant", value=0
        )
        conv_padding = (0, 0, 0)
    else:
        conv_padding = (pad_d_left, pad_h_top, pad_w_left)

    # Create nn.Conv3d module
    conv = nn.Conv3d(
        in_channels=C_in,
        out_channels=C_out,
        kernel_size=(K_d, K_h, K_w),
        stride=stride,
        padding=conv_padding,
        dilation=dilation,
        bias=(bias != None),
    )

    # Set weights: transpose from [K_d, K_h, K_w, C_in, C_out] to [C_out, C_in, K_d, K_h, K_w]
    with torch.no_grad():
        conv.weight.copy_(filters.permute(4, 3, 0, 1, 2))
        if bias != None:
            conv.bias.copy_(bias)

    # Perform convolution
    conv_out = conv(x_in)

    # Fuse batchnorm 2d if specified
    if batch_norm_mode.is_fused():
        # Per-C_out-channel statistics over batch and all spatial dims (B, D_out, H_out, W_out).
        if batch_norm_mode == BatchNormMode.EVAL:
            # Eval mode: normalize with the given running statistics and do not update them.
            means = running_means.float().reshape(C_out)
            variances = running_variances.float().reshape(C_out)
        else:
            means = conv_out.float().mean(dim=(0, 2, 3, 4))
            variances = conv_out.float().var(dim=(0, 2, 3, 4), correction=0)
            correct1_variances = conv_out.float().var(dim=(0, 2, 3, 4), correction=1)

            # running_means / running_variances are [C_out, 1]; updated stats keep that shape. means /
            # variances are [C_out], so reshape to [C_out, 1] to broadcast correctly.
            rm = running_means.float().reshape(C_out, 1)
            rv = running_variances.float().reshape(C_out, 1)
            updated_running_means = (1 - momentum) * rm + momentum * means.reshape(C_out, 1)
            updated_running_variances = (1 - momentum) * rv + momentum * correct1_variances.reshape(C_out, 1)

        # gamma / beta [C_out, 1] -> flatten to [C_out], broadcast over output. torch.nn.BatchNorm2d:
        #   out = gamma * (y - mean) / sqrt(var + eps) + beta
        gamma_c = gamma.reshape(C_out).float()
        beta_c = beta.reshape(C_out).float()
        view_shape = (1, C_out, 1, 1, 1)

        orig_dtype = conv_out.dtype
        out_f = conv_out.float()
        normalized = (out_f - means.view(view_shape)) / torch.sqrt(variances.view(view_shape) + batch_norm_eps)
        output = (normalized * gamma_c.view(view_shape) + beta_c.view(view_shape)).to(orig_dtype)

        # Optionally apply the activation (and the residual add) after batchnorm; the statistics
        # above were computed on the raw convolution output, matching the kernel.
        output = _apply_activation_and_residual(output, activation_fn, residual_add_loc, residuals_in)

        # In eval mode don't update means and variances
        if batch_norm_mode == BatchNormMode.EVAL:
            return {"out": output.detach()}
        # Key insertion order must match the kernel's returned tuple; output_pre_norm inserts the
        # raw pre-batchnorm conv output right after "out".
        outputs = {"out": output.detach()}
        if output_pre_norm:
            outputs["conv_out"] = conv_out.detach()
        outputs["means"] = means.detach()
        outputs["variances"] = variances.detach()
        outputs["updated_running_means"] = updated_running_means.detach()
        outputs["updated_running_variances"] = updated_running_variances.detach()
        return outputs

    # Apply activation function and residual add if specified
    output = _apply_activation_and_residual(conv_out, activation_fn, residual_add_loc, residuals_in)

    return {"out": output.detach()}
