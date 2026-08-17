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
GPT-OSS MXFP4 decode golden — structured model.

A standalone, pure-CPU PyTorch transcription of
``vllm_neuron/model/gpt_oss/model_mxfp4.py``, **preserving the class hierarchy
and section layout** of the original. The parallelism collectives (TP/DP/EP),
the NKI megakernels, and the FP8 / DCP / attention-DP / packed-K branches are
removed; each ``forward_decode`` calls the pure-torch oracle in ``kernels.py``
exactly where the original calls ``NF.attention_decode`` / ``NF.moe_block_tkg``.

Sections mirror the original file:
  1. RMS Normalization
  2. Rotary Position Embedding (YaRN)
  3. Attention (decode)
  4. MoE Experts (MXFP4, dequantized)
  5. MLP Wrapper
  6. Decoder Layer
  7. Model Backbone
  8. Language Model Head

Only the decode path is implemented (per project scope); the KV cache is seeded
externally (synthetic or via ``PagedKVManager.prefill_write``).
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import nn

from . import kernels
from . import loader as ldr
from .config import GptOssConfig
from .paged_kv import PagedKVManager

# =============================================================================
# Section 1: RMS Normalization
# =============================================================================


class GptOssRMSNorm(nn.Module):
    """RMSNorm over the (unpadded) hidden dim, fp32 accumulation.

    The golden works in natural unpadded dims, so there is no padded-variance
    correction or shuffled zeroing (unlike the production MXFP4 RMSNorm).
    """

    def __init__(self, hidden_size: int, eps: float):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        x = hidden_states.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * x).to(input_dtype)


# =============================================================================
# Section 2: Rotary Position Embedding (YaRN)
# <-- MODEL-SPECIFIC: transcribed from GptOssRotaryEmbedding (model_mxfp4.py)
# =============================================================================


class GptOssRotaryEmbedding(nn.Module):
    """YaRN rotary embedding, transcribed from the production model."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta
        self.scaling_factor = config.rope_scaling_factor
        self.beta_fast = config.rope_beta_fast
        self.beta_slow = config.rope_beta_slow
        self.initial_context_length = config.rope_initial_context_length

        inv_freq, concentration = self._compute_inv_freq_and_concentration()
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.register_buffer("concentration", concentration, persistent=False)

    def _compute_inv_freq_and_concentration(self):
        freq = self.rope_theta ** (torch.arange(0, self.head_dim, 2, dtype=torch.float) / self.head_dim)
        concentration = 0.1 * math.log(self.scaling_factor) + 1.0
        d_half = self.head_dim / 2
        low = (
            d_half * math.log(self.initial_context_length / (self.beta_fast * 2 * math.pi)) / math.log(self.rope_theta)
        )
        high = (
            d_half * math.log(self.initial_context_length / (self.beta_slow * 2 * math.pi)) / math.log(self.rope_theta)
        )
        interpolation = 1.0 / (self.scaling_factor * freq)
        extrapolation = 1.0 / freq
        ramp = (torch.arange(d_half, dtype=torch.float32) - low) / (high - low)
        mask = 1 - ramp.clamp(0, 1)
        inv_freq = interpolation * (1 - mask) + extrapolation * mask
        return inv_freq, torch.tensor(concentration)

    def forward(self, position_ids: torch.Tensor, dtype: torch.dtype):
        """position_ids: [N] -> (cos, sin) each [N, head_dim//2]."""
        inv_freq_expanded = self.inv_freq[None, :].float()  # [1, hd/2]
        position_ids_expanded = position_ids[:, None].float()  # [N, 1]
        freqs = position_ids_expanded @ inv_freq_expanded  # [N, hd/2]
        cos = freqs.cos() * self.concentration
        sin = freqs.sin() * self.concentration
        return cos.to(dtype), sin.to(dtype)


# =============================================================================
# Section 3: Attention (decode)
# <-- MODEL-SPECIFIC: GQA, sinks, sliding window, YaRN RoPE
# =============================================================================


class GptOssAttention(nn.Module):
    """Multi-head attention (decode path only), single device.

    Weights are stored fused (qkv) exactly like the production model, so a
    kernel author sees the same ``W_qkv`` / ``W_out`` shapes.
    """

    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.dtype = config.torch_dtype
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.scaling = config.head_dim**-0.5
        self.sliding_window = config.sliding_window if config.is_sliding_layer(layer_idx) else None

        q_size = self.num_attention_heads * self.head_dim
        kv_size = self.num_key_value_heads * self.head_dim
        qkv_size = q_size + 2 * kv_size

        # Fused QKV weight [H, q_size + 2*kv_size] (row = hidden, col = out).
        self.qkv_proj_weight = nn.Parameter(torch.empty(self.hidden_size, qkv_size, dtype=self.dtype))
        self.qkv_proj_bias = nn.Parameter(torch.empty(qkv_size, dtype=self.dtype))
        self.o_proj_weight = nn.Parameter(torch.empty(q_size, self.hidden_size, dtype=self.dtype))
        self.o_proj_bias = nn.Parameter(torch.empty(self.hidden_size, dtype=self.dtype))
        self.sinks = nn.Parameter(torch.empty(self.num_attention_heads, dtype=torch.float32))

        self.q_size = q_size
        self.kv_size = kv_size

        # KV cache tensors (bound externally).
        self.k_cache: Optional[torch.Tensor] = None
        self.v_cache: Optional[torch.Tensor] = None

    def forward_decode(
        self,
        hidden_states: torch.Tensor,  # [B*S_decode, H]
        positions: torch.Tensor,  # [B*S_decode] absolute positions
        position_embeddings,  # (cos, sin) each [B*S_decode, head_dim//2]
        block_table: torch.Tensor,  # [B, max_blocks_per_seq]
        slot_mapping: torch.Tensor,  # [B*S_decode]
    ) -> torch.Tensor:
        B = block_table.shape[0]
        tokens, hidden = hidden_states.shape
        S_decode = tokens // B
        hidden_states = hidden_states.to(self.dtype)
        X = hidden_states.view(B, S_decode, hidden)

        cos, sin = position_embeddings

        # >>> KERNEL-ORACLE: attention decode (QKV, RoPE, paged attn, KV write, O) <<<
        output = kernels.attention_decode(
            X=X,
            W_qkv=self.qkv_proj_weight,
            bias_qkv=self.qkv_proj_bias,
            num_q_heads=self.num_attention_heads,
            num_kv_heads=self.num_key_value_heads,
            head_dim=self.head_dim,
            cos=cos,
            sin=sin,
            K_cache=self.k_cache,
            V_cache=self.v_cache,
            block_table=block_table,
            slot_mapping=slot_mapping,
            pos_ids=positions,
            sliding_window=self.sliding_window,
            sink=self.sinks,
            softmax_scale=self.scaling,
            W_out=self.o_proj_weight,
            bias_out=self.o_proj_bias,
            update_cache=True,
        )
        return output


# =============================================================================
# Section 4: MoE Experts (MXFP4, dequantized to dense)
# <-- MODEL-SPECIFIC: 128 experts, top-4, SwiGLU with clamping
# <-- MXFP4: weights dequantized at load time (loader.py) into dense tensors
# =============================================================================


class GptOssExperts(nn.Module):
    """Expert feed-forward with MXFP4 weights dequantized to dense."""

    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.total_num_experts = config.num_local_experts
        self.num_experts_per_token = config.num_experts_per_tok
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.rms_norm_eps = config.rms_norm_eps
        self.alpha = config.swiglu_alpha
        self.limit = config.swiglu_limit

        E, H, I = self.total_num_experts, self.hidden_size, self.intermediate_size

        # Pre-MLP RMSNorm.
        self.post_attention_layernorm = GptOssRMSNorm(H, config.rms_norm_eps)

        # Router (dense bf16).
        self.router_weight = nn.Parameter(torch.empty(E, H, dtype=torch.float32))
        self.router_bias = nn.Parameter(torch.zeros(E, dtype=torch.float32))

        # Dense dequantized expert weights (natural order).
        #   gate_up_weight : [E, H, 2, I]   gate_up_bias : [E, 2, I] (up +1 baked)
        #   down_weight    : [E, I, H]      down_bias    : [E, H]
        self.gate_up_weight = nn.Parameter(torch.empty(E, H, 2, I, dtype=torch.float32), requires_grad=False)
        self.gate_up_bias = nn.Parameter(torch.zeros(E, 2, I, dtype=torch.float32), requires_grad=False)
        self.down_weight = nn.Parameter(torch.empty(E, I, H, dtype=torch.float32), requires_grad=False)
        self.down_bias = nn.Parameter(torch.zeros(E, H, dtype=torch.float32), requires_grad=False)

    def forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """hidden_states: [T, H] -> [T, H]."""
        # >>> KERNEL-ORACLE: fused RMSNorm + router top-k + SwiGLU experts <<<
        return kernels.moe_block_tkg(
            hidden_states=hidden_states,
            gamma=self.post_attention_layernorm.weight,
            eps=self.rms_norm_eps,
            router_weight=self.router_weight,
            router_bias=self.router_bias,
            gate_up_weight=self.gate_up_weight,
            gate_up_bias=self.gate_up_bias,
            down_weight=self.down_weight,
            down_bias=self.down_bias,
            top_k=self.num_experts_per_token,
            swiglu_limit=self.limit,
            swiglu_alpha=self.alpha,
        )


# =============================================================================
# Section 5: MLP Wrapper
# =============================================================================


class GptOssMLP(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.experts = GptOssExperts(config)
        self.dtype = config.torch_dtype

    def forward_decode(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.experts.forward_decode(hidden_states.to(self.dtype))


# =============================================================================
# Section 6: Decoder Layer
# =============================================================================


class GptOssDecoderLayer(nn.Module):
    def __init__(self, config: GptOssConfig, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.input_layernorm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = GptOssAttention(config, layer_idx)
        self.mlp = GptOssMLP(config)

    def forward_decode(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
        position_embeddings,
        block_table: torch.Tensor,
        slot_mapping: torch.Tensor,
    ) -> torch.Tensor:
        # hidden_states -> RMSNorm -> Attention -> residual -> MoE -> residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn.forward_decode(
            hidden_states, positions, position_embeddings, block_table, slot_mapping
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.mlp.forward_decode(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


# =============================================================================
# Section 7: Model Backbone
# =============================================================================


class GptOssModel(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, dtype=config.torch_dtype)
        self.layers = nn.ModuleList([GptOssDecoderLayer(config, i) for i in range(config.num_hidden_layers)])
        self.norm = GptOssRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.rotary_emb = GptOssRotaryEmbedding(config)

    def forward_decode(
        self,
        input_ids: torch.Tensor,  # [B*S_decode]
        positions: torch.Tensor,  # [B*S_decode] absolute positions
        block_tables: dict,  # layer_idx -> [B, max_blocks_per_seq]
        slot_mappings: dict,  # layer_idx -> [B*S_decode]
    ) -> torch.Tensor:
        hidden_states = self.embed_tokens(input_ids)  # [T, H]
        cos, sin = self.rotary_emb(positions, dtype=hidden_states.dtype)
        position_embeddings = (cos, sin)

        for idx, layer in enumerate(self.layers):
            hidden_states = layer.forward_decode(
                hidden_states,
                positions,
                position_embeddings,
                block_tables[idx],
                slot_mappings[idx],
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states


# =============================================================================
# Section 8: Language Model Head
# =============================================================================


class GptOssForCausalLM(nn.Module):
    def __init__(self, config: GptOssConfig):
        super().__init__()
        self.config = config
        self.model = GptOssModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False, dtype=config.torch_dtype)

    def bind_kv_cache(self, mgr: "PagedKVManager") -> None:
        """Attach the paged KV cache tensors from a manager to each attn layer.

        Usage:
            >>> mgr = PagedKVManager(...); model.bind_kv_cache(mgr)
        """
        for li, layer in enumerate(self.model.layers):
            layer.self_attn.k_cache = mgr.k_caches[li]
            layer.self_attn.v_cache = mgr.v_caches[li]

    @torch.no_grad()
    def forward_decode(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_tables: dict,
        slot_mappings: dict,
    ) -> torch.Tensor:
        """Returns logits [B*S_decode, vocab]."""
        hidden_states = self.model.forward_decode(input_ids, positions, block_tables, slot_mappings)
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        return logits

    # ── Weight loading from the raw HF MXFP4 checkpoint ─────────────────────

    @torch.no_grad()
    def load_weights(self, ckpt_dir: str, num_layers: Optional[int] = None) -> None:
        """Load and dequantize weights from a raw HF GPT-OSS MXFP4 checkpoint.

        Only the first ``num_layers`` layers are loaded when given (handy for a
        layer-slice wiring check; note fewer than all layers won't produce
        coherent text).

        Usage:
            >>> m = GptOssForCausalLM(cfg); m.load_weights("/workplace/qieqingy/gptoss-120b-hf")
        """
        idx = ldr.SafetensorsIndex(ckpt_dir)
        cfg = self.config
        n = num_layers if num_layers is not None else cfg.num_hidden_layers

        # Embedding / final norm / lm_head.
        self.model.embed_tokens.weight.copy_(idx.get("model.embed_tokens.weight").to(torch.float32))
        self.model.norm.weight.copy_(idx.get("model.norm.weight").to(torch.float32))
        self.lm_head.weight.copy_(idx.get("lm_head.weight").to(cfg.torch_dtype))

        for li in range(n):
            p = f"model.layers.{li}"
            layer = self.model.layers[li]
            attn = layer.self_attn
            exp = layer.mlp.experts

            # Norms.
            layer.input_layernorm.weight.copy_(idx.get(f"{p}.input_layernorm.weight").to(torch.float32))
            exp.post_attention_layernorm.weight.copy_(idx.get(f"{p}.post_attention_layernorm.weight").to(torch.float32))

            # Attention: fuse q/k/v -> [H, q_size+2*kv_size]; o -> [q_size, H].
            qw = idx.get(f"{p}.self_attn.q_proj.weight")  # [q_size, H]
            kw = idx.get(f"{p}.self_attn.k_proj.weight")  # [kv_size, H]
            vw = idx.get(f"{p}.self_attn.v_proj.weight")  # [kv_size, H]
            qkv = torch.cat([qw, kw, vw], dim=0).to(cfg.torch_dtype)  # [qkv, H]
            attn.qkv_proj_weight.copy_(qkv.t().contiguous())  # [H, qkv]
            qb = idx.get(f"{p}.self_attn.q_proj.bias")
            kb = idx.get(f"{p}.self_attn.k_proj.bias")
            vb = idx.get(f"{p}.self_attn.v_proj.bias")
            attn.qkv_proj_bias.copy_(torch.cat([qb, kb, vb], dim=0).to(cfg.torch_dtype))

            ow = idx.get(f"{p}.self_attn.o_proj.weight")  # [H, q_size]
            attn.o_proj_weight.copy_(ow.t().contiguous().to(cfg.torch_dtype))  # [q_size, H]
            attn.o_proj_bias.copy_(idx.get(f"{p}.self_attn.o_proj.bias").to(cfg.torch_dtype))
            attn.sinks.copy_(idx.get(f"{p}.self_attn.sinks").to(torch.float32))

            # Router.
            exp.router_weight.copy_(idx.get(f"{p}.mlp.router.weight").to(torch.float32))
            exp.router_bias.copy_(idx.get(f"{p}.mlp.router.bias").to(torch.float32))

            # Experts: dequantize MXFP4 -> dense.
            gub = idx.get(f"{p}.mlp.experts.gate_up_proj_blocks")
            gus = idx.get(f"{p}.mlp.experts.gate_up_proj_scales")
            exp.gate_up_weight.copy_(ldr.build_gate_up_weight(gub, gus))
            exp.gate_up_bias.copy_(ldr.build_gate_up_bias(idx.get(f"{p}.mlp.experts.gate_up_proj_bias")))
            db = idx.get(f"{p}.mlp.experts.down_proj_blocks")
            ds = idx.get(f"{p}.mlp.experts.down_proj_scales")
            exp.down_weight.copy_(ldr.build_down_weight(db, ds))
            exp.down_bias.copy_(idx.get(f"{p}.mlp.experts.down_proj_bias").to(torch.float32))


__all__ = [
    "GptOssRMSNorm",
    "GptOssRotaryEmbedding",
    "GptOssAttention",
    "GptOssExperts",
    "GptOssMLP",
    "GptOssDecoderLayer",
    "GptOssModel",
    "GptOssForCausalLM",
]
