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
# ty: ignore — vendored GPT-OSS golden; torch nn.Module idioms trip ty's type mapping
"""
Flat function entry point for the GPT-OSS MXFP4 decode golden.

For whole-model kernel development you want a single flat call
``weights + inputs + KV caches -> logits`` that mirrors a megakernel's
signature. This module provides exactly that. It does **not** flatten the model:
internally it constructs and drives the ``GptOss*`` ``nn.Module`` hierarchy from
``model.py`` (which in turn calls the ``kernels.py`` oracles). The flat function
is just a thin, stateless boundary around the structured golden.

Two things a kernel author gets here:

  1. ``gptoss_mxfp4_decode`` — the flat one-shot oracle. Give it a config, a
     weights source (a state_dict, a raw HF MXFP4 checkpoint dir, or a prebuilt
     model), the paged KV caches, and the decode inputs; get logits back.

  2. Optional per-layer **kernel-IO capture**. Pass ``capture=True`` (or a dict)
     and every ``attention_decode`` / ``moe_block_tkg`` call's inputs and output
     are recorded (cloned, KV caches snapshotted *before* the in-place write).
     Feed those captured inputs to your kernel and diff against the captured
     output — the tightest possible per-stage oracle.

Typical kernel-dev loop:

    from gptoss_mxfp4_golden.golden_entry import build_golden_model, gptoss_mxfp4_decode

    model = build_golden_model(cfg, ckpt_dir="/workplace/qieqingy/gptoss-120b-hf",
                               num_layers=4)              # build once
    logits, io = gptoss_mxfp4_decode(
        model, input_ids, positions,
        k_caches, v_caches, block_tables, slot_mappings,
        capture=True,
    )
    # io["attention"][layer_idx]["inputs"]  -> dict of tensors your kernel takes
    # io["attention"][layer_idx]["output"]  -> reference output to diff against
    # io["moe"][layer_idx]                  -> same for the MoE kernel

No ``nki`` / ``vllm`` / ``transformers`` imports here.
"""

from __future__ import annotations

import contextlib
from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor

from . import kernels
from .config import GptOssConfig
from .model import GptOssForCausalLM

# ---------------------------------------------------------------------------
# Model construction (build once, decode many times)
# ---------------------------------------------------------------------------


def build_golden_model(
    config: GptOssConfig,
    *,
    weights: Optional[Dict[str, Tensor]] = None,
    ckpt_dir: Optional[str] = None,
    num_layers: Optional[int] = None,
) -> GptOssForCausalLM:
    """Construct a golden ``GptOssForCausalLM`` and populate its weights.

    Exactly one of ``weights`` / ``ckpt_dir`` should be given (if neither, the
    model keeps its random init — handy for shape/wiring smoke tests).

    Args:
        config:     the golden ``GptOssConfig`` (unpadded, natural dims).
        weights:    a flat ``state_dict`` (keys as in ``model.state_dict()``).
        ckpt_dir:   a raw HF GPT-OSS MXFP4 checkpoint dir (dequantized on load).
        num_layers: load only the first N layers (for a layer-slice check).

    Returns:
        An ``eval()``-mode model, ready for ``gptoss_mxfp4_decode``.

    Usage:
        >>> m = build_golden_model(cfg, ckpt_dir="/workplace/qieqingy/gptoss-120b-hf")
    """
    if weights is not None and ckpt_dir is not None:
        raise ValueError("pass at most one of `weights` / `ckpt_dir`")

    model = GptOssForCausalLM(config)
    if ckpt_dir is not None:
        model.load_weights(ckpt_dir, num_layers=num_layers)
    elif weights is not None:
        model.load_state_dict(weights, strict=True, assign=True)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Per-layer kernel-IO capture
# ---------------------------------------------------------------------------


def _snapshot(v):
    """Detach + clone a value for a faithful, side-effect-free capture."""
    if isinstance(v, Tensor):
        return v.detach().clone()
    return v


@contextlib.contextmanager
def _capture_kernel_io(store: dict):
    """Temporarily wrap the two kernel oracles to record their per-call IO.

    ``attention_decode`` writes K/V in place, so its inputs are snapshotted
    (cloned) *before* the real call runs — i.e. the pre-write cache a kernel
    would receive.
    """
    store.setdefault("attention", [])
    store.setdefault("moe", [])

    real_attn = kernels.attention_decode
    real_moe = kernels.moe_block_tkg

    def attn_wrapper(*args, **kwargs):
        assert not args, "attention_decode is called by keyword in model.py"
        inputs = {k: _snapshot(v) for k, v in kwargs.items()}
        out = real_attn(**kwargs)
        store["attention"].append({"inputs": inputs, "output": _snapshot(out)})
        return out

    def moe_wrapper(*args, **kwargs):
        assert not args, "moe_block_tkg is called by keyword in model.py"
        inputs = {k: _snapshot(v) for k, v in kwargs.items()}
        out = real_moe(**kwargs)
        store["moe"].append({"inputs": inputs, "output": _snapshot(out)})
        return out

    kernels.attention_decode = attn_wrapper
    kernels.moe_block_tkg = moe_wrapper
    try:
        yield store
    finally:
        kernels.attention_decode = real_attn
        kernels.moe_block_tkg = real_moe


# ---------------------------------------------------------------------------
# Cache / table normalization helpers
# ---------------------------------------------------------------------------


def _as_per_layer(x: Union[Tensor, List[Tensor], Dict[int, Tensor]], n: int) -> dict:
    """Normalize a per-layer arg to a ``{layer_idx: tensor}`` dict.

    Accepts a single shared tensor, a list/tuple, or an already-keyed dict.
    """
    if isinstance(x, dict):
        return x
    if isinstance(x, (list, tuple)):
        assert len(x) == n, f"expected {n} per-layer entries, got {len(x)}"
        return {i: x[i] for i in range(n)}
    # single shared tensor -> broadcast to every layer
    return dict.fromkeys(range(n), x)


# ---------------------------------------------------------------------------
# The flat entry point
# ---------------------------------------------------------------------------


def gptoss_mxfp4_decode(
    model_or_config: Union[GptOssForCausalLM, GptOssConfig],
    input_ids: Tensor,  # [B*S_decode] token ids
    positions: Tensor,  # [B*S_decode] abs positions
    k_caches: Union[List[Tensor], Dict[int, Tensor]],  # per-layer paged K cache
    v_caches: Union[List[Tensor], Dict[int, Tensor]],  # per-layer paged V cache
    block_tables: Union[Tensor, List[Tensor], Dict[int, Tensor]],
    slot_mappings: Union[Tensor, List[Tensor], Dict[int, Tensor]],
    *,
    weights: Optional[Dict[str, Tensor]] = None,
    ckpt_dir: Optional[str] = None,
    num_layers: Optional[int] = None,
    capture: Union[bool, dict, None] = None,
) -> Union[Tensor, Tuple[Tensor, dict]]:
    """Flat whole-model MXFP4 decode oracle (one decode step).

    Internally constructs / drives the structured ``GptOss*`` classes; the
    parallelism-free, single-device math runs through the ``kernels.py`` oracles.
    This is the reference to diff a whole-model decode kernel against.

    Cache layout (per attention layer), mirroring production:
        K_cache / V_cache : [num_blocks, kv_heads, block_size, head_dim]
        block_table       : [B, max_blocks_per_seq] int32, -1 for unused blocks
        slot_mapping      : [B*S_decode] int64, write slot = block*block_size + off

    Args:
        model_or_config: a prebuilt ``GptOssForCausalLM`` (fast path — build once
            with ``build_golden_model`` and reuse), OR a ``GptOssConfig`` (then
            ``weights`` / ``ckpt_dir`` selects the weight source and a model is
            built for this call).
        input_ids:   ``[B*S_decode]`` token ids for the active tokens.
        positions:   ``[B*S_decode]`` absolute sequence positions.
        k_caches, v_caches: per-layer paged caches (list, dict, or — via the
            block/slot args — bound on the model). Written **in place**.
        block_tables, slot_mappings: per-layer tables; a single tensor is
            broadcast to all layers, or pass a list/dict for SWA-trimmed tables.
        weights / ckpt_dir / num_layers: weight source when a config is passed.
        capture: ``True`` (or a dict to fill) records per-layer kernel IO.

    Returns:
        ``logits`` ``[B*S_decode, vocab]`` — or ``(logits, io)`` when ``capture``.

    Usage:
        >>> model = build_golden_model(cfg, ckpt_dir=CKPT, num_layers=4)
        >>> logits = gptoss_mxfp4_decode(
        ...     model, input_ids, positions, k_caches, v_caches, bt, sm)
        >>> logits, io = gptoss_mxfp4_decode(model, ..., capture=True)
    """
    if isinstance(model_or_config, GptOssForCausalLM):
        model = model_or_config
    else:
        model = build_golden_model(model_or_config, weights=weights, ckpt_dir=ckpt_dir, num_layers=num_layers)

    n = model.config.num_hidden_layers

    # Bind the caches onto each attention layer (in-place written by the kernel).
    kd = _as_per_layer(k_caches, n)
    vd = _as_per_layer(v_caches, n)
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.k_cache = kd[li]
        layer.self_attn.v_cache = vd[li]

    bt = _as_per_layer(block_tables, n)
    sm = _as_per_layer(slot_mappings, n)

    input_ids = input_ids.reshape(-1)
    positions = positions.reshape(-1)

    store: Optional[dict] = None
    if capture:
        store = capture if isinstance(capture, dict) else {}

    ctx = _capture_kernel_io(store) if store is not None else contextlib.nullcontext()
    with torch.no_grad(), ctx:
        logits = model.forward_decode(input_ids, positions, bt, sm)

    if store is not None:
        return logits, store
    return logits


__all__ = ["build_golden_model", "gptoss_mxfp4_decode"]
