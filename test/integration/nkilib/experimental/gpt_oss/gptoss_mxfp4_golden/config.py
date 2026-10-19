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
"""
GPT-OSS configuration for the standalone MXFP4 decode golden.

Mirrors ``vllm_neuron/model/gpt_oss/config.py`` but with **no HF / vllm
imports** and, critically, **no hidden/intermediate padding**. The production
config pads ``hidden_size`` and ``intermediate_size`` to 3072 for hardware
alignment and shuffles the hidden dim; the golden works entirely in the
model's *natural, unpadded* dimensions (2880 for the 120B), which keeps the
math readable and lets us cross-check against HuggingFace directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import torch


@dataclass
class GptOssConfig:
    """GPT-OSS architecture hyperparameters (unpadded, single device)."""

    # ── Model architecture ────────────────────────────────────────────────
    vocab_size: int = 201088
    hidden_size: int = 2880
    num_hidden_layers: int = 36
    num_attention_heads: int = 64  # Q heads
    num_key_value_heads: int = 8  # KV heads (GQA)
    head_dim: int = 64
    intermediate_size: int = 2880
    rms_norm_eps: float = 1e-5
    torch_dtype: torch.dtype = torch.bfloat16

    # ── MoE ───────────────────────────────────────────────────────────────
    num_local_experts: int = 128
    num_experts_per_tok: int = 4

    # ── Attention features ────────────────────────────────────────────────
    # Sliding window applied on even-indexed ("sliding_attention") layers.
    sliding_window: int | None = 128

    # ── RoPE (YaRN) ───────────────────────────────────────────────────────
    rope_theta: float = 150000.0
    rope_scaling_factor: float = 32.0
    rope_beta_fast: float = 32.0
    rope_beta_slow: float = 1.0
    rope_initial_context_length: int = 4096

    # ── SwiGLU activation ─────────────────────────────────────────────────
    swiglu_limit: float = 7.0
    swiglu_alpha: float = 1.702

    # ── Sequence ──────────────────────────────────────────────────────────
    max_position_embeddings: int = 131072

    # Convenience: which layers slide. GPT-OSS alternates
    # sliding (even idx) / full (odd idx).
    def is_sliding_layer(self, layer_idx: int) -> bool:
        return self.sliding_window is not None and layer_idx % 2 == 0

    @classmethod
    def from_hf_json(cls, config_path: str) -> "GptOssConfig":
        """Build from a HuggingFace ``config.json`` (e.g. the 120B checkpoint).

        Usage:
            >>> cfg = GptOssConfig.from_hf_json(
            ...     "/workplace/qieqingy/gptoss-120b-hf/config.json"
            ... )
            >>> cfg.hidden_size, cfg.num_local_experts
            (2880, 128)
        """
        with open(config_path) as f:
            d = json.load(f)

        rope = d.get("rope_scaling", {}) or {}
        dtype = d.get("torch_dtype", "bfloat16")
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)

        return cls(
            vocab_size=d["vocab_size"],
            hidden_size=d["hidden_size"],
            num_hidden_layers=d["num_hidden_layers"],
            num_attention_heads=d["num_attention_heads"],
            num_key_value_heads=d["num_key_value_heads"],
            head_dim=d["head_dim"],
            intermediate_size=d["intermediate_size"],
            rms_norm_eps=d["rms_norm_eps"],
            torch_dtype=dtype,
            num_local_experts=d["num_local_experts"],
            num_experts_per_tok=d["num_experts_per_tok"],
            sliding_window=d.get("sliding_window"),
            rope_theta=d.get("rope_theta", 150000.0),
            rope_scaling_factor=rope.get("factor", 1.0),
            rope_beta_fast=rope.get("beta_fast", 32.0),
            rope_beta_slow=rope.get("beta_slow", 1.0),
            rope_initial_context_length=rope.get("original_max_position_embeddings", 4096),
            swiglu_limit=d.get("swiglu_limit", 7.0),
            swiglu_alpha=1.702,
            max_position_embeddings=d.get("max_position_embeddings", 131072),
        )

    def tiny(self, **overrides) -> "GptOssConfig":
        """Return a shrunken copy for fast CPU tests (keeps head_dim, ratios).

        Usage:
            >>> small = GptOssConfig().tiny(num_hidden_layers=2, hidden_size=256)
        """
        from dataclasses import replace

        return replace(self, **overrides)
