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
# SPDX-License-Identifier: Apache-2.0
# ty: ignore — vendored GPT-OSS golden; optional safetensors import unresolved for ty
"""
MXFP4 checkpoint loader for the standalone GPT-OSS decode golden.

Loads the **raw HuggingFace GPT-OSS MXFP4 checkpoint** and dequantizes the
expert weights to dense bf16/fp32 in the model's *natural, unpadded* order.
This is deliberately Option 1 from the format analysis: read the raw
``*_blocks`` / ``*_scales`` tensors and apply the canonical
``_dequantize_mxfp4_to_bf16`` recipe (transcribed verbatim from
``vllm_neuron/model/gpt_oss/weight_loaders_bf16.py``). It sidesteps the tiled
hardware layout and the hidden-dim shuffle entirely, and is the exact recipe
the repo itself uses as its HF reference.

Raw checkpoint expert-weight layout (per layer, ``E`` experts):

    gate_up_proj_blocks : uint8  [E, 2*I, H//32, 16]   (16 bytes = 32 FP4 values)
    gate_up_proj_scales : uint8  [E, 2*I, H//32]
    gate_up_proj_bias   : bf16   [E, 2*I]              interleaved [gate0,up0,...]
    down_proj_blocks    : uint8  [E, H, I//32, 16]
    down_proj_scales    : uint8  [E, H, I//32]
    down_proj_bias      : bf16   [E, H]

Everything else (attention q/k/v/o, sinks, router, norms, embed, lm_head) is
already dense bf16 in the checkpoint.

No ``nki`` / ``vllm`` / ``transformers`` imports here.
"""

from __future__ import annotations

import json
import os
import struct
from typing import Dict

import torch

# ---------------------------------------------------------------------------
# Core MXFP4 dequant (ground truth: weight_loaders_bf16.py:264-316)
# ---------------------------------------------------------------------------

# E2M1 FP4 lookup table. Index = 4-bit code; bit 3 is the sign.
_FP4_VALUES = [
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    -0.0,
    -0.5,
    -1.0,
    -1.5,
    -2.0,
    -3.0,
    -4.0,
    -6.0,
]


def dequantize_mxfp4(blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    """Dequantize MXFP4 packed blocks + scales to dense ``dtype``.

    Each byte in ``blocks`` packs two FP4 values (low nibble first, high nibble
    second). Each group of 16 bytes (= 32 FP4 values) shares one ``uint8``
    scale, applied as ``value * 2^(scale - 127)``.

    Args:
        blocks: uint8 ``[..., G, 16]`` packed FP4.
        scales: uint8 ``[..., G]`` biased exponents.
        dtype:  output dtype (bf16 matches production; fp32 for tighter refs).

    Returns:
        Dense tensor ``[..., G*32]``.

    Usage:
        >>> w = dequantize_mxfp4(blocks, scales)   # [..., G*32]
    """
    lut = torch.tensor(_FP4_VALUES, dtype=dtype, device=blocks.device)
    exp = (scales.to(torch.int32) - 127).unsqueeze(-1)  # [..., G, 1]

    out = torch.empty(*blocks.shape[:-1], blocks.shape[-1] * 2, dtype=dtype, device=blocks.device)  # [..., G, 32]
    out[..., 0::2] = lut[(blocks & 0x0F).long()]  # low nibble  -> even positions
    out[..., 1::2] = lut[(blocks >> 4).long()]  # high nibble -> odd positions
    torch.ldexp(out, exp, out=out)  # * 2^(scale - 127)
    return out.flatten(-2)  # [..., G*32]


# ---------------------------------------------------------------------------
# Dense expert-weight reconstruction (natural, unpadded order)
# ---------------------------------------------------------------------------
#
# The torch MoE math in kernels.py expects, per local expert:
#   gate_up_weight : [E, H, 2, I]   (contract H -> 2*I)
#   down_weight    : [E, I, H]      (contract I -> H)
#   gate_up_bias   : [E, 2, I]      (up portion already +1)
#   down_bias      : [E, H]
# where dim 1 of gate_up_* is (gate, up).


def build_gate_up_weight(
    blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Raw gate_up blocks/scales ``[E, 2I, H//32, 16]`` -> dense ``[E, H, 2, I]``.

    De-interleaves the ``2I`` axis (even=gate, odd=up).
    """
    deq = dequantize_mxfp4(blocks, scales, dtype)  # [E, 2I, H]
    gate = deq[:, 0::2, :]  # [E, I, H]  even = gate
    up = deq[:, 1::2, :]  # [E, I, H]  odd  = up
    # -> [E, H, 2, I]: contraction dim H first, then (gate, up), then I
    fused = torch.stack([gate, up], dim=1)  # [E, 2, I, H]
    return fused.permute(0, 3, 1, 2).contiguous()  # [E, H, 2, I]


def build_down_weight(blocks: torch.Tensor, scales: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Raw down blocks/scales ``[E, H, I//32, 16]`` -> dense ``[E, I, H]``."""
    deq = dequantize_mxfp4(blocks, scales, dtype)  # [E, H, I]
    return deq.transpose(1, 2).contiguous()  # [E, I, H]


def build_gate_up_bias(bias: torch.Tensor, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Raw interleaved gate_up bias ``[E, 2I]`` -> ``[E, 2, I]`` with +1 on up.

    Production bakes ``hidden_act_bias=1.0`` into the up bias at load time
    (weight_loaders_mxfp4.py). We keep that convention so the golden's
    ``(up_bias)`` already includes +1; the SwiGLU in kernels.py then does NOT
    re-add it. (Equivalently: HF clamps up to [-limit, limit] and computes
    ``(up + 1) * glu`` — see hf_reference.py — which is algebraically the same.)
    """
    gate = bias[:, 0::2].to(dtype)  # [E, I]
    up = bias[:, 1::2].to(dtype) + 1.0  # [E, I]  <-- +1 baked in
    return torch.stack([gate, up], dim=1)  # [E, 2, I]


# ---------------------------------------------------------------------------
# safetensors reader (no `safetensors.torch` dependency on layout details)
# ---------------------------------------------------------------------------


class SafetensorsIndex:
    """Minimal lazy reader over a sharded safetensors checkpoint directory.

    Reads ``model.safetensors.index.json`` to map tensor names -> shard files,
    and loads individual tensors on demand via ``safetensors.safe_open``.

    Usage:
        >>> idx = SafetensorsIndex("/workplace/qieqingy/gptoss-120b-hf")
        >>> w = idx.get("model.layers.0.self_attn.q_proj.weight")
    """

    def __init__(self, ckpt_dir: str):
        self.dir = ckpt_dir
        index_path = os.path.join(ckpt_dir, "model.safetensors.index.json")
        if os.path.exists(index_path):
            with open(index_path) as f:
                self.weight_map: Dict[str, str] = json.load(f)["weight_map"]
        else:
            # Single-file checkpoint: scan the one safetensors file's header.
            files = [f for f in os.listdir(ckpt_dir) if f.endswith(".safetensors")]
            assert len(files) == 1, f"expected index.json or single file in {ckpt_dir}"
            names = _read_safetensors_header(os.path.join(ckpt_dir, files[0]))
            self.weight_map = dict.fromkeys(names, files[0])
        self._open_files: dict = {}

    def _handle(self, fname: str):
        from safetensors import safe_open

        if fname not in self._open_files:
            self._open_files[fname] = safe_open(os.path.join(self.dir, fname), framework="pt", device="cpu")
        return self._open_files[fname]

    def has(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        return self._handle(self.weight_map[name]).get_tensor(name)


def _read_safetensors_header(path: str) -> list:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
    return [k for k in header if k != "__metadata__"]
