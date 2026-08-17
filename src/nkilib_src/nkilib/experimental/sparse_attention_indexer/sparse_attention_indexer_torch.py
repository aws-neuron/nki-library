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

"""PyTorch reference implementation for DeepSeek Sparse Attention Indexer."""

import ml_dtypes as mld
import numpy as np
import torch

# fp8_e4m3fn dynamic range bound (= 2^8). Used both by the OCP MX block-32
# scaling (e8m0 = max_block_exponent - 8) and the rare full-row fp8 quant.
FP8_E4M3FN_MAX = 448.0

# OCP MX block size: 1 e8m0 scale per 32 contiguous contraction-dim elements.
MX_BLOCK = 32

# Maximum representable exponent of fp8_e4m3fn (2^8 = 256 fits within 448).
FP8_E4M3FN_MAX_EXP = 8


def _block32_mx_quant_dequant(x):
    """Simulate OCP MX block-32 quant→dequant along the last dim.

    Matches the kernel's ``nisa.quantize_mx`` (block-32 e8m0) semantics
    exactly — uses float32 bit-extraction for the per-block max exponent so
    the result is bit-identical to
    ``test/utils/mx_utils.py:quantize_mx_golden`` (and the simulator's
    ``quantize_mx``) up to the trailing fp8-rounding step.

      For each block of 32 elements along the last dim of x,
          biased_exp     = (uint32_view(block) >> 23) & 0xFF      # float32 biased exp
          max_biased_exp = max(biased_exp over the block)
          e8m0           = max_biased_exp - 8                     # for fp8_e4m3fn
          scale          = 2^(e8m0 - 127)
          fp8            = round_e4m3fn(clip(block / scale, ±448))
          deq            = fp8 * scale

    Last dim of x must be a multiple of 32 (kernel pads K to K_TILE=512
    before quantizing; this ref only models the valid head_dim portion, so
    head_dim must already be a multiple of 32).
    """
    assert x.shape[-1] % MX_BLOCK == 0, (
        f"_block32_mx_quant_dequant requires last dim multiple of {MX_BLOCK}, got shape {tuple(x.shape)}"
    )
    """Compute via numpy so we can bit-extract the float32 exponent the way
    quantize_mx_golden / the simulator do; this avoids the small log2
    rounding mismatch a torch.log2-based path would introduce on
    power-of-two inputs."""
    x_f32 = x.detach().to(torch.float32).cpu().numpy()
    orig_shape = x_f32.shape
    blocked = x_f32.reshape(*orig_shape[:-1], orig_shape[-1] // MX_BLOCK, MX_BLOCK)

    """Float32 biased exponent (bits 23..30). Zero gives 0 → e8m0 = -8 (uint8
    wrap = 248) → scale = 2^(248 - 127) = 2^121, large but harmless because
    the input is zero."""
    exp = (blocked.view(np.uint32) >> 23) & 0xFF
    max_exp = exp.max(axis=-1, keepdims=True)  # uint32
    e8m0 = (max_exp.astype(np.uint8) - FP8_E4M3FN_MAX_EXP).astype(np.int32)
    scale = (2.0 ** (e8m0 - 127)).astype(np.float32)  # broadcast block→element below

    quantized = np.clip(blocked / scale, -FP8_E4M3FN_MAX, FP8_E4M3FN_MAX).astype(mld.float8_e4m3fn)
    dequantized = quantized.astype(np.float32) * scale
    return torch.from_numpy(dequantized.reshape(orig_shape).copy()).to(x.dtype).to(x.device)


def _layernorm_ref(x, gamma, beta, eps=1e-6):
    """LayerNorm reference."""
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    return (x - mean) / torch.sqrt(var + eps) * gamma + beta


def _rope_non_interleaved_ref(x, cos, sin):
    """Non-interleaved RoPE: pairs (x_i, x_{i+d/2})."""
    d = x.shape[-1]
    d_half = d // 2
    x1 = x[..., :d_half]
    x2 = x[..., d_half:]
    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    return torch.cat([y1, y2], dim=-1)
