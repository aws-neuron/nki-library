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
Single-layer flat golden for GPT-OSS MXFP4 decode kernel development.

A kernel environment passes everything as plain tensors, and one decoder layer
(the repeating transformer block) is the realistic unit to validate — not all 36
layers. This module provides a flat entry point ``gptoss_mxfp4_decode_layer``
whose signature is **one explicit plain-tensor argument per weight / cache /
input**, with scalars taken from ``GptOssConfig`` (the config classes stay).

It does **not** flatten the model logic: internally it constructs a
``GptOssDecoderLayer``, assigns the passed-in tensors into its parameters, binds
the paged KV cache, and calls its ``forward_decode``. So the math that runs is
exactly the structured golden (which in turn calls the ``kernels.py`` oracles);
the flat signature is just the boundary a kernel author diffs against.

Tensor contract (one decoder layer, ``B`` sequences, ``S`` active tokens each,
``T = B*S``):

  inputs
    hidden_states        [T, H]        pre-layer residual stream (bf16/fp32)
    positions            [T]           int  absolute position of each active token
    cos, sin             [T, hd//2]    RoPE tables for the active tokens

  attention weights
    input_layernorm_weight   [H]
    qkv_proj_weight          [H, q_size + 2*kv_size]   fused (row=hidden, col=out)
    qkv_proj_bias            [q_size + 2*kv_size]
    o_proj_weight            [q_size, H]
    o_proj_bias              [H]
    sinks                    [num_q_heads]             fp32 per-head sink logit

  MoE weights (dense, already dequantized — see dequantize_expert_weights)
    post_attention_layernorm_weight  [H]
    router_weight            [E, H]
    router_bias              [E]
    gate_up_weight           [E, H, 2, I]   dim2 = (gate, up)
    gate_up_bias             [E, 2, I]      up portion already includes +1
    down_weight              [E, I, H]
    down_bias                [E, H]

  paged KV cache (written in place)
    k_cache, v_cache     [num_blocks, kv_heads, block_size, head_dim]
    block_table          [B, max_blocks_per_seq]  int32, -1 for unused blocks
    slot_mapping         [T]                       int64, block*block_size + off

  q_size  = num_q_heads  * head_dim
  kv_size = num_kv_heads * head_dim

No ``nki`` / ``vllm`` / ``transformers`` imports here.
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import Tensor

from . import loader as _ldr
from .config import GptOssConfig
from .model import GptOssDecoderLayer

# ---------------------------------------------------------------------------
# Optional: raw MXFP4 -> dense expert tensors (so a kernel dev can start from
# the raw checkpoint tensors instead of pre-dequantized ones).
# ---------------------------------------------------------------------------


def dequantize_expert_weights(
    gate_up_blocks: Tensor,  # uint8 [E, 2I, H//32, 16]
    gate_up_scales: Tensor,  # uint8 [E, 2I, H//32]
    gate_up_bias_raw: Tensor,  # bf16  [E, 2I]  interleaved [gate0, up0, ...]
    down_blocks: Tensor,  # uint8 [E, H, I//32, 16]
    down_scales: Tensor,  # uint8 [E, H, I//32]
    down_bias_raw: Tensor,  # bf16  [E, H]
    dtype: torch.dtype = torch.float32,
) -> dict:
    """Raw HF MXFP4 expert tensors -> the dense tensors the flat entry expects.

    Returns a dict with keys ``gate_up_weight`` [E,H,2,I], ``gate_up_bias``
    [E,2,I] (up +1 baked in), ``down_weight`` [E,I,H], ``down_bias`` [E,H].

    Usage:
        >>> dense = dequantize_expert_weights(gub, gus, gu_bias, db, ds, d_bias)
        >>> gptoss_mxfp4_decode_layer(cfg, ..., **dense, ...)
    """
    return {
        "gate_up_weight": _ldr.build_gate_up_weight(gate_up_blocks, gate_up_scales, dtype),
        "gate_up_bias": _ldr.build_gate_up_bias(gate_up_bias_raw, dtype),
        "down_weight": _ldr.build_down_weight(down_blocks, down_scales, dtype),
        "down_bias": down_bias_raw.to(dtype),
    }


# ---------------------------------------------------------------------------
# The one-layer flat entry point
# ---------------------------------------------------------------------------


def gptoss_mxfp4_decode_layer(
    config: GptOssConfig,
    # ── inputs ──────────────────────────────────────────────────────────
    hidden_states: Tensor,  # [T, H]
    positions: Tensor,  # [T] int absolute positions
    cos: Tensor,  # [T, head_dim//2]
    sin: Tensor,  # [T, head_dim//2]
    # ── attention weights ──────────────────────────────────────────────
    input_layernorm_weight: Tensor,  # [H]
    qkv_proj_weight: Tensor,  # [H, q_size + 2*kv_size]
    qkv_proj_bias: Tensor,  # [q_size + 2*kv_size]
    o_proj_weight: Tensor,  # [q_size, H]
    o_proj_bias: Tensor,  # [H]
    sinks: Tensor,  # [num_q_heads]
    # ── MoE weights (dense) ─────────────────────────────────────────────
    post_attention_layernorm_weight: Tensor,  # [H]
    router_weight: Tensor,  # [E, H]
    router_bias: Tensor,  # [E]
    gate_up_weight: Tensor,  # [E, H, 2, I]
    gate_up_bias: Tensor,  # [E, 2, I]  (up +1 baked)
    down_weight: Tensor,  # [E, I, H]
    down_bias: Tensor,  # [E, H]
    # ── paged KV cache (written in place) ───────────────────────────────
    k_cache: Tensor,  # [num_blocks, kv_heads, block_size, head_dim]
    v_cache: Tensor,  # [num_blocks, kv_heads, block_size, head_dim]
    block_table: Tensor,  # [B, max_blocks_per_seq] int32
    slot_mapping: Tensor,  # [T] int64
    # ── per-layer control ───────────────────────────────────────────────
    sliding_window: Optional[int] = None,
    *,
    capture: Optional[dict] = None,
) -> Tensor:
    """Flat single-layer MXFP4 decode oracle. Returns ``[T, H]`` (post-layer
    residual stream), and writes the new K/V into ``k_cache`` / ``v_cache``.

    All tensors are plain (no modules); ``config`` supplies scalars. Internally
    constructs a ``GptOssDecoderLayer``, loads these tensors into it, and calls
    ``forward_decode`` — so the executed math is the structured golden.

    ``sliding_window``: pass the window size for a sliding-attention layer, or
    ``None`` for a full-attention layer. (In GPT-OSS even layers slide, odd
    layers are full; here it is explicit so the kernel controls it per call.)

    ``capture``: pass a dict to receive this layer's attention/MoE kernel-IO
    (``capture["attention"][0]`` / ``capture["moe"][0]`` -> {inputs, output}).

    Usage:
        >>> out = gptoss_mxfp4_decode_layer(cfg, hidden, pos, cos, sin,
        ...     ln_w, qkv_w, qkv_b, o_w, o_b, sinks,
        ...     pa_ln_w, r_w, r_b, gu_w, gu_b, d_w, d_b,
        ...     k_cache, v_cache, block_table, slot_mapping,
        ...     sliding_window=128)
    """
    # A one-layer config so GptOssDecoderLayer's internal sizing matches.
    layer_cfg = config.tiny(num_hidden_layers=1)

    # layer_idx picks sliding vs full inside the module. Even idx slides; use
    # idx 0 when a window is requested, idx 1 (full) otherwise — but only if the
    # config actually enables sliding. This keeps the module's own
    # ``is_sliding_layer`` consistent with the caller's ``sliding_window``.
    if sliding_window is not None:
        layer_cfg = layer_cfg.tiny(sliding_window=sliding_window)
        layer_idx = 0  # even -> sliding
    else:
        layer_idx = 1  # odd -> full attention

    layer = GptOssDecoderLayer(layer_cfg, layer_idx=layer_idx)
    layer.eval()

    # Load the plain tensors into the module's parameters (assign, so dtype
    # follows the passed tensors exactly — the kernel's dtypes, not defaults).
    state = {
        "input_layernorm.weight": input_layernorm_weight,
        "self_attn.qkv_proj_weight": qkv_proj_weight,
        "self_attn.qkv_proj_bias": qkv_proj_bias,
        "self_attn.o_proj_weight": o_proj_weight,
        "self_attn.o_proj_bias": o_proj_bias,
        "self_attn.sinks": sinks,
        "mlp.experts.post_attention_layernorm.weight": post_attention_layernorm_weight,
        "mlp.experts.router_weight": router_weight,
        "mlp.experts.router_bias": router_bias,
        "mlp.experts.gate_up_weight": gate_up_weight,
        "mlp.experts.gate_up_bias": gate_up_bias,
        "mlp.experts.down_weight": down_weight,
        "mlp.experts.down_bias": down_bias,
    }
    layer.load_state_dict(state, strict=True, assign=True)

    # Bind the paged KV cache (written in place by the attention oracle).
    layer.self_attn.k_cache = k_cache
    layer.self_attn.v_cache = v_cache

    # RoPE tables are supplied directly as plain tensors (the kernel owns them).
    position_embeddings = (cos, sin)

    ctx = _maybe_capture(capture)
    with torch.no_grad(), ctx:
        out = layer.forward_decode(
            hidden_states,
            positions,
            position_embeddings,
            block_table,
            slot_mapping,
        )
    return out


def _maybe_capture(capture: Optional[dict]):
    """Return a context manager that records this layer's kernel IO, or a
    null context when capture is not requested. Reuses golden_entry's wrapper."""
    import contextlib

    if capture is None:
        return contextlib.nullcontext()
    from .golden_entry import _capture_kernel_io

    return _capture_kernel_io(capture)


__all__ = ["gptoss_mxfp4_decode_layer", "dequantize_expert_weights"]
