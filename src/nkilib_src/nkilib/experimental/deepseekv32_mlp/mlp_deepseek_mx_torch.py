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

"""PyTorch reference for the DeepSeek V3.2 MX MLP kernel (``mlp_deepseek_mx``).

Focused golden for the single path that kernel supports: MX (MXFP8) quantization, CTE mode,
MX-prequantized packed block-scale hidden input, no normalization / bias / fused-add.
"""

import nki.language as nl
import numpy as np
import torch

from ...core.mlp.mlp_torch import mlp_torch_ref
from ...core.utils.common_types import (
    ActFnType,
    ComputationMode,
    DtypeMode,
    MLPGateUpWeightLayout,
    NormType,
    QuantizationType,
)

_LNC = 2
_SCALE_BLOCK_128 = 128


def _compact_gate_up_scale_to_native(compact, H, I):
    """Broadcast a compact block-128 gate/up scale [H/128, I/128] to the native block-32
    physical layout [16, H/512, I/512, 4, 128] the golden consumes.

    Exact inverse of the kernel's in-kernel expand+swizzle: native[kk, ht, it, a, b] samples
    compact[ht*4 + kk//4, it*4 + b//32] (independent of the x4 lane ``a`` and of ``b % 32``).
    """
    compact = compact.cpu().numpy() if isinstance(compact, torch.Tensor) else np.asarray(compact)
    n_H512 = H // 512
    n_I512 = I // 512
    native = np.empty((16, n_H512, n_I512, 4, 128), dtype=np.uint8)
    for kk in range(16):
        for b in range(128):
            native[kk, :, :, :, b] = compact[
                np.arange(n_H512)[:, None] * 4 + kk // 4,
                np.arange(n_I512)[None, :] * 4 + b // 32,
            ][:, :, None]
    return torch.from_numpy(native)


def _compact_down_scale_to_native(compact, H, n_I512):
    """Broadcast a compact block-128 down scale [I/128, H/128] to the native block-32
    layout [16, I/512, H] the golden consumes. Down has no column swizzle: a plain 128x
    repeat on H and 4x repeat on the 32-K sub-block. native[kk, it, h] = compact[it*4 + kk//4, h//128].
    """
    compact = compact.cpu().numpy() if isinstance(compact, torch.Tensor) else np.asarray(compact)
    native = np.empty((16, n_I512, H), dtype=np.uint8)
    for kk in range(16):
        native[kk, :, :] = np.repeat(compact[np.arange(n_I512) * 4 + kk // 4, :], _SCALE_BLOCK_128, axis=1)[:, :H]
    return torch.from_numpy(native)


def mlp_deepseek_mx_torch_ref(
    hidden_tensor,
    gate_proj_weights_tensor,
    up_proj_weights_tensor,
    down_proj_weights_tensor,
    gate_w_scale,
    up_w_scale,
    down_w_scale,
    activation_fn=ActFnType.SiLU,
    output_dtype=None,
    routed_expert_output=None,
    sbm=None,  # kernel-parity only (HBM allocator); unused by the reference
    compact_scales=False,
):
    """Golden for ``mlp_deepseek_mx``.

    Delegates to the shared ``mlp_torch_ref`` with the DeepSeek-fixed config (MX CTE, no norm /
    bias / fused-add, ``H_X4_INNERMOST`` weights, prequantized packed block-scale input) so the
    golden math is identical to what the core MLP kernel is validated against; this wrapper only
    pins the config and maps the DeepSeek kernel's argument names.

    Args:
        hidden_tensor: [B, S, H + scale_region] packed MX-prequantized hidden input.
        gate_proj_weights_tensor / up_proj_weights_tensor: [128, H/512, I/512, 4, 128, 4] fp8.
        down_proj_weights_tensor: [128, I/512, H, 4] fp8.
        gate_w_scale / up_w_scale: [16, H/512, I/512, 4, 128] uint8 MX weight scales.
        down_w_scale: [16, I/512, H] uint8 MX weight scales.
        activation_fn (ActFnType): Gate activation (default SiLU).
        output_dtype: Output dtype (default bf16).
        routed_expert_output: Optional [B, S, H] routed-expert output added to the MLP result
            (shared-experts mode). Summed in fp32 then cast to ``output_dtype`` to mirror the
            kernel's in-DMA fp32 accumulate. When None, the MLP result is returned unchanged.
        sbm: Accepted for kernel signature parity; unused.
        compact_scales: When True, ``gate_w_scale`` / ``up_w_scale`` are compact block-128
            [H/128, I/128] and ``down_w_scale`` is [I/128, H/128]; they are broadcast to the
            native block-32 physical layout here (lossless, since one 128x128 block shares one
            scale) so the delegated golden is identical to the compact_scales=False golden.

    Returns:
        dict: ``{"out": [B, S, H]}`` reference output (summed with the routed output if provided).
    """
    if output_dtype == None:
        output_dtype = nl.bfloat16
    if compact_scales:
        # Weight tensors carry their shapes; derive H (down [128, I/512, H, 4]) and I (gate/up
        # [128, H/512, I/512, 4, 128, 4]) to rebroadcast the compact scales to native block-32.
        H = down_proj_weights_tensor.shape[2]
        n_I512 = down_proj_weights_tensor.shape[1]
        I = gate_proj_weights_tensor.shape[2] * gate_proj_weights_tensor.shape[3] * gate_proj_weights_tensor.shape[4]
        gate_w_scale = _compact_gate_up_scale_to_native(gate_w_scale, H, I)
        up_w_scale = _compact_gate_up_scale_to_native(up_w_scale, H, I)
        down_w_scale = _compact_down_scale_to_native(down_w_scale, H, n_I512)
    ref = mlp_torch_ref[_LNC](
        hidden_tensor=hidden_tensor,
        gate_proj_weights_tensor=gate_proj_weights_tensor,
        up_proj_weights_tensor=up_proj_weights_tensor,
        down_proj_weights_tensor=down_proj_weights_tensor,
        gate_w_scale=gate_w_scale,
        up_w_scale=up_w_scale,
        down_w_scale=down_w_scale,
        activation_fn=activation_fn,
        quantization_type=QuantizationType.MX,
        normalization_type=NormType.NO_NORM,
        gate_up_w_layout=MLPGateUpWeightLayout.H_X4_INNERMOST,
        mode=ComputationMode.PREFILL,
        dtype_mode=DtypeMode.NON_OCP,
        output_dtype=output_dtype,
    )
    if routed_expert_output != None:
        # Mirror the kernel: the store DMA adds the routed output in fp32, then casts to output_dtype.
        summed = ref["out"].to(torch.float32) + routed_expert_output.to(torch.float32)
        ref["out"] = summed.to(ref["out"].dtype)
    return ref
