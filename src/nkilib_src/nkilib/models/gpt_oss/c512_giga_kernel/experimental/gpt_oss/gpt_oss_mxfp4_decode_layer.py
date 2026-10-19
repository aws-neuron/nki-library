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

"""GPT-OSS MXFP4 decode-layer token-generation kernel (PLACEHOLDER STUB).

One GPT-OSS decoder block for the decode (token-generation) path:

    h -> RMSNorm -> attention(GQA + per-head sinks + optional SWA + YaRN RoPE,
                              paged KV cache)             -> + residual
      -> RMSNorm -> MoE(top-k softmax router + SwiGLU MXFP4 experts) -> + residual  ==> out

This is the repeating transformer block of GPT-OSS-120B (36 layers; even layers
slide, odd layers are full attention). It is the realistic unit for a decode
megakernel, and the piece nkilib does not yet have: ``transformer_tkg`` composes
attention with a *dense* MLP, while GPT-OSS needs sink+SWA+paged-KV attention
fused with an *MXFP4 MoE* block.

Status
------
The body is a **placeholder** that returns a correct-shape ``[T, H]`` output so
the tensor contract, test harness, and reference oracle can be built ahead of
the real kernel. The reference oracle is the standalone pure-torch golden
``gptoss_mxfp4_decode_layer`` in ``private-vllm-neuron/gptoss_mxfp4_golden`` —
bit-validated against HuggingFace on real 120B weights. See the integration
test ``test/integration/nkilib/experimental/gpt_oss/`` for how the two are diffed.

Implementation plan (what replaces the stub body)
-------------------------------------------------
The real kernel composes two existing nkilib kernels, both of which already
expose the GPT-OSS-specific knobs:

  * attention: ``experimental/transformer/attention_block_tkg`` — already accepts
    ``sink=`` (per-head attention sink) and ``rope_contiguous_layout=True`` (the
    NEOX half-split RoPE GPT-OSS uses). Needs the paged block-table / slot-mapping
    KV contract and per-layer sliding-window masking wired through.
  * MoE: ``core/moe_block/moe_block_tkg`` — already matches the GPT-OSS MoE math:
    ``router_bias``, ``top_k`` with SOFTMAX (``router_act_fn``), the ``+1`` on the
    up projection via ``hidden_act_bias``, ``gate``/``up`` clamps, and the
    ``[E, H, 2, I]`` fused gate/up weight layout (MXFP4 scales via
    ``expert_gate_up_weights_scale`` / ``expert_down_weights_scale``).

Tensor contract (mirrors the golden's flat entry, one arg per tensor):
    inputs        hidden_states [T,H], positions [T], cos/sin [T, head_dim//2]
    attention     input_layernorm_weight [H], qkv_proj_weight [H, q+2kv],
                  qkv_proj_bias [q+2kv], o_proj_weight [q_size,H], o_proj_bias [H],
                  sinks [num_q_heads]
    MoE (dense-dequantized MXFP4)
                  post_attention_layernorm_weight [H], router_weight [E,H],
                  router_bias [E], gate_up_weight [E,H,2,I], gate_up_bias [E,2,I]
                  (up +1 baked), down_weight [E,I,H], down_bias [E,H]
    paged KV      k_cache/v_cache [num_blocks, kv_heads, block_size, head_dim],
                  block_table [B, max_blocks_per_seq] int32, slot_mapping [T] int64
    scalars       head_dim, num_q_heads, num_kv_heads, num_experts, top_k, eps,
                  softmax_scale, swiglu_limit, swiglu_alpha, sliding_window

    T = B * S_decode (S_decode = 1 for plain decode).
"""

from typing import Optional

import nki
import nki.isa as nisa
import nki.language as nl

from ...core.utils.kernel_assert import kernel_assert


@nki.jit
def gpt_oss_mxfp4_decode_layer(
    # ── inputs ──────────────────────────────────────────────────────────
    hidden_states: nl.NkiTensor,  # [T, H]
    positions: nl.NkiTensor,  # [T] int absolute positions
    cos: nl.NkiTensor,  # [T, head_dim//2]
    sin: nl.NkiTensor,  # [T, head_dim//2]
    # ── attention weights ──────────────────────────────────────────────
    input_layernorm_weight: nl.NkiTensor,  # [H]
    qkv_proj_weight: nl.NkiTensor,  # [H, q_size + 2*kv_size]
    qkv_proj_bias: nl.NkiTensor,  # [q_size + 2*kv_size]
    o_proj_weight: nl.NkiTensor,  # [q_size, H]
    o_proj_bias: nl.NkiTensor,  # [H]
    sinks: nl.NkiTensor,  # [num_q_heads]
    # ── MoE weights (dense, dequantized MXFP4) ──────────────────────────
    post_attention_layernorm_weight: nl.NkiTensor,  # [H]
    router_weight: nl.NkiTensor,  # [E, H]
    router_bias: nl.NkiTensor,  # [E]
    gate_up_weight: nl.NkiTensor,  # [E, H, 2, I]
    gate_up_bias: nl.NkiTensor,  # [E, 2, I]  (up +1 baked)
    down_weight: nl.NkiTensor,  # [E, I, H]
    down_bias: nl.NkiTensor,  # [E, H]
    # ── paged KV cache ──────────────────────────────────────────────────
    k_cache: nl.NkiTensor,  # [num_blocks, kv_heads, block_size, head_dim]
    v_cache: nl.NkiTensor,  # [num_blocks, kv_heads, block_size, head_dim]
    block_table: nl.NkiTensor,  # [B, max_blocks_per_seq] int32
    slot_mapping: nl.NkiTensor,  # [T] int64
    # ── scalars (config-derived; compile-time constants) ────────────────
    head_dim: int,
    num_q_heads: int,
    num_kv_heads: int,
    num_experts: int,
    top_k: int,
    eps: float,
    softmax_scale: float,
    swiglu_limit: float,
    swiglu_alpha: float,
    sliding_window: Optional[int] = None,
) -> nl.NkiTensor:
    """One GPT-OSS decoder block for token generation. **Placeholder stub.**

    Returns ``[T, H]`` (the post-layer residual stream), matching the standalone
    golden ``gptoss_mxfp4_decode_layer``. The production kernel additionally
    writes the freshly-computed K/V into ``k_cache`` / ``v_cache`` in place via
    ``slot_mapping``; that mutation is not yet modelled by this stub.

    Dimensions:
        B: batch (sequences); S_decode: active tokens/seq (1 for plain decode)
        T = B * S_decode; H: hidden size; E: experts; I: MoE intermediate size
        q_size = num_q_heads * head_dim; kv_size = num_kv_heads * head_dim

    Args:
        hidden_states (nl.NkiTensor): [T, H] pre-layer residual stream on HBM.
        positions (nl.NkiTensor): [T] absolute position of each active token.
        cos (nl.NkiTensor): [T, head_dim//2] YaRN RoPE cosine table.
        sin (nl.NkiTensor): [T, head_dim//2] YaRN RoPE sine table.
        input_layernorm_weight (nl.NkiTensor): [H] pre-attention RMSNorm gamma.
        qkv_proj_weight (nl.NkiTensor): [H, q_size + 2*kv_size] fused QKV weight.
        qkv_proj_bias (nl.NkiTensor): [q_size + 2*kv_size] fused QKV bias ([q|k|v]).
        o_proj_weight (nl.NkiTensor): [q_size, H] output projection weight.
        o_proj_bias (nl.NkiTensor): [H] output projection bias.
        sinks (nl.NkiTensor): [num_q_heads] per-head attention-sink logit.
        post_attention_layernorm_weight (nl.NkiTensor): [H] pre-MoE RMSNorm gamma.
        router_weight (nl.NkiTensor): [E, H] router weight.
        router_bias (nl.NkiTensor): [E] router bias.
        gate_up_weight (nl.NkiTensor): [E, H, 2, I] fused gate/up (dim 2 = gate,up).
        gate_up_bias (nl.NkiTensor): [E, 2, I] fused gate/up bias (up +1 baked in).
        down_weight (nl.NkiTensor): [E, I, H] down projection weight.
        down_bias (nl.NkiTensor): [E, H] down projection bias.
        k_cache (nl.NkiTensor): [num_blocks, kv_heads, block_size, head_dim] paged K.
        v_cache (nl.NkiTensor): [num_blocks, kv_heads, block_size, head_dim] paged V.
        block_table (nl.NkiTensor): [B, max_blocks_per_seq] int32, -1 for unused.
        slot_mapping (nl.NkiTensor): [T] int64 write slot = block*block_size + off.
        head_dim (int): per-head dimension.
        num_q_heads (int): number of query heads.
        num_kv_heads (int): number of key/value heads (GQA).
        num_experts (int): number of local experts (E).
        top_k (int): experts selected per token.
        eps (float): RMSNorm epsilon.
        softmax_scale (float): attention score scale (head_dim ** -0.5).
        swiglu_limit (float): SwiGLU clamp limit.
        swiglu_alpha (float): SwiGLU (Swish) alpha.
        sliding_window (Optional[int]): window size for a sliding-attention layer,
            or ``None`` for a full-attention layer.

    Returns:
        nl.NkiTensor: [T, H] post-layer residual stream on shared HBM.

    Notes:
        - PLACEHOLDER: returns a zero-filled output of the correct shape/dtype.
          Traces/compiles cleanly (``--nki-compilation-mode=tracer``); it does not
          yet compute the layer nor write the KV cache, so it will not match the
          golden numerically until the real body is implemented.
        - The stub reads only ``hidden_states`` for its shape/dtype; all other
          inputs are declared to fix the kernel's tensor contract for the harness.
    """
    T, H = hidden_states.shape
    dtype = hidden_states.dtype
    P_MAX = nl.tile_size.pmax

    # Placeholder covers the decode regime (T = batch * S_decode is small). The
    # real kernel tiles T generally; the stub keeps a single partition tile so it
    # traces without tiling machinery.
    kernel_assert(T <= P_MAX, f"placeholder gpt_oss_mxfp4_decode_layer expects T <= {P_MAX}, got T={T}")

    # TODO(kernel): replace the zero-fill below with the real decoder block:
    #   1. RMSNorm(hidden_states, input_layernorm_weight, eps)
    #   2. attention_block_tkg(... sink=sinks, rope_contiguous_layout=True,
    #                          paged K/V via block_table/slot_mapping, SWA=sliding_window)
    #   3. residual add
    #   4. RMSNorm(., post_attention_layernorm_weight, eps)
    #   5. moe_block_tkg(... router SOFTMAX + router_bias, top_k, hidden_act_bias=+1,
    #                    gate/up clamps from swiglu_limit, MXFP4 experts)
    #   6. residual add  ->  layer_output [T, H]
    # and scatter the new K/V into k_cache/v_cache at slot_mapping.
    layer_output = nl.ndarray((T, H), dtype=dtype, buffer=nl.shared_hbm)
    zeros_sb = nl.ndarray((T, H), dtype=dtype, buffer=nl.sbuf)
    nisa.memset(dst=zeros_sb, value=0)
    nisa.dma_copy(dst=layer_output, src=zeros_sb)
    return layer_output
