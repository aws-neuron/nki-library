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

from typing import Optional

import torch

from ..mlp.mlp_parameters import TKG_BS_SEQLEN_THRESHOLD
from ..utils.kernel_assert import kernel_assert


def silu(x: torch.Tensor) -> torch.Tensor:
    """SiLU activation function: x * sigmoid(x)"""
    return x * torch.sigmoid(x)


def rms_norm_torch(hidden: torch.Tensor, gamma: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    """
    RMSNorm implementation in PyTorch.

    Args:
      hidden: Input tensor of shape (..., H)
      gamma: Scale parameter of shape (H,) or None
      eps: Epsilon for numerical stability

    Returns:
      Normalized tensor of same shape as hidden
    """
    rms = torch.sqrt(torch.mean(hidden**2, dim=-1, keepdim=True) + eps)
    norm = hidden / rms
    if gamma is not None:
        norm = norm * gamma
    return norm


def norm_qkv_torch(
    hidden: torch.Tensor,
    gamma: Optional[torch.Tensor],
    qkv_weights: torch.Tensor,
    head_dim: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    QKV projection with RMSNorm.

    Args:
      hidden: Input tensor of shape (B, S, H)
      gamma: RMSNorm scale parameter of shape (H,) or None
      qkv_weights: QKV projection weights of shape (H, D) where D = (num_q_heads + 2) * head_dim
      head_dim: Dimension per head
      eps: Epsilon for RMSNorm

    Returns:
      QKV output of shape (num_heads+2, B, S, head_dim)
    """
    normed = rms_norm_torch(hidden, gamma, eps)
    qkv_out = torch.matmul(normed, qkv_weights)  # (B, S, D)
    b, s, d = qkv_out.shape
    qkv_out = qkv_out.reshape(b, s, d // head_dim, head_dim)  # (B, S, num_heads+2, head_dim)
    qkv_out = qkv_out.permute(2, 0, 1, 3)  # (num_heads+2, B, S, head_dim)
    return qkv_out


def rope_torch(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """
    Rotary Position Embedding (RoPE) in PyTorch.

    Uses first_second_half_impl=True (interleave first/second halves into even/odd positions).

    Args:
      x: Input tensor of shape (d_head, S)
      cos: Cosine frequencies of shape (d_head//2, S)
      sin: Sine frequencies of shape (d_head//2, S)

    Returns:
      RoPE-transformed tensor of shape (d_head, S)
    """
    d_head, S = x.shape
    half_d = d_head // 2

    x_t = x.T  # (S, d_head)

    # Interleave first/second halves into even/odd positions
    new_x = torch.empty_like(x_t)
    new_x[:, ::2] = x_t[:, :half_d]  # even positions <- first half
    new_x[:, 1::2] = x_t[:, half_d:]  # odd positions <- second half

    cos_t = cos.T  # (S, d_head//2)
    sin_t = sin.T

    # Split into real/imaginary pairs
    xri = new_x.reshape(S, half_d, 2)
    x_r = xri[:, :, 0]  # (S, d_head//2)
    x_i = xri[:, :, 1]

    # Apply rotation
    x_out_r = x_r * cos_t - x_i * sin_t
    x_out_i = x_r * sin_t + x_i * cos_t

    # Stack and flatten
    x_out = torch.stack([x_out_r, x_out_i], dim=-1)  # (S, d_head//2, 2)
    x_out = x_out.reshape(S, d_head)

    # Put even/odd indices back into first/second halves
    result = torch.cat([x_out[:, 0::2], x_out[:, 1::2]], dim=1)

    return result.T  # (d_head, S)


def attention_tkg_torch(
    Q: torch.Tensor,
    K_active: torch.Tensor,
    V_active: torch.Tensor,
    K_prior: torch.Tensor,
    V_prior: torch.Tensor,
    mask_prior: torch.Tensor,
    mask_active: torch.Tensor,
) -> torch.Tensor:
    """
    Attention computation for TKG (token generation) with KV cache.

    Args:
      Q: Query tensor of shape (d_head, num_heads, S_tkg)
      K_active: Active K tensor of shape (d_head, S_tkg)
      V_active: Active V tensor of shape (S_tkg, d_head)
      K_prior: Prior K cache of shape (d_head, S_ctx)
      V_prior: Prior V cache of shape (S_ctx, d_head)
      mask_prior: Prior attention mask of shape (num_heads * S_tkg, S_ctx)
      mask_active: Active attention mask of shape (num_heads * S_tkg, S_tkg)

    Returns:
      Attention output of shape (d_head, num_heads * S_tkg)
    """
    d_head, num_heads, S_tkg = Q.shape
    S_ctx = K_prior.shape[1]

    Q_flat = Q.reshape(d_head, num_heads * S_tkg)  # (d_head, num_heads * S_tkg)

    # Compute attention scores
    prior_scores = Q_flat.T @ K_prior  # (num_heads * S_tkg, S_ctx)
    active_scores = Q_flat.T @ K_active  # (num_heads * S_tkg, S_tkg)

    # Apply masks
    min_val = torch.finfo(torch.float32).min

    # Check if mask shapes match scores shapes
    if mask_prior.shape == prior_scores.shape:
        mask_prior_float = mask_prior.float()
        prior_scores = torch.where(mask_prior_float == 1, prior_scores, torch.full_like(prior_scores, min_val))
    # else: skip masking (shapes don't match, likely during NKI tracing)

    if mask_active.shape == active_scores.shape:
        mask_active_float = mask_active.float()
        active_scores = torch.where(mask_active_float == 1, active_scores, torch.full_like(active_scores, min_val))

    # Softmax: max-reduce
    max_prior = prior_scores.max(dim=-1, keepdim=True).values
    max_active = active_scores.max(dim=-1, keepdim=True).values
    max_score = torch.max(max_prior, max_active)

    # Softmax: exp
    exp_prior = torch.exp(prior_scores - max_score)
    exp_active = torch.exp(active_scores - max_score)
    denominator = exp_prior.sum(dim=-1, keepdim=True) + exp_active.sum(dim=-1, keepdim=True)

    # Attention output
    softmax_prior = exp_prior.to(Q.dtype)
    softmax_active = exp_active.to(Q.dtype)
    attn_prior = softmax_prior @ V_prior  # (num_heads * S_tkg, d_head)
    attn_active = softmax_active @ V_active
    attn_output = (attn_prior + attn_active) / denominator

    return attn_output.T  # (d_head, num_heads * S_tkg)


def attention_block_torch(
    X: torch.Tensor,
    W_gamma: torch.Tensor,
    W_qkv: torch.Tensor,
    W_out: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    K_cache: torch.Tensor,
    V_cache: torch.Tensor,
    mask_cache: torch.Tensor,
    mask_active: torch.Tensor,
    position_ids: torch.Tensor,
    eps: float,
    d_head: int,
    q_heads_per_core: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Full attention block: QKV projection -> RoPE -> Attention -> Output projection -> KV cache update.

    Args:
      X: Input tensor of shape (B, S_tkg, H)
      W_gamma: RMSNorm gamma of shape (H,)
      W_qkv: QKV weights of shape (H, (q_heads_per_core + 2) * d_head)
      W_out: Output projection weights of shape (q_heads_per_core * d_head, H)
      cos: RoPE cosine of shape (d_head//2, B, S_tkg)
      sin: RoPE sine of shape (d_head//2, B, S_tkg)
      K_cache: K cache of shape (B, d_head, S_ctx) - transposed layout
      V_cache: V cache of shape (B, S_ctx, d_head)
      mask_cache: Cache mask of shape (B, q_heads_per_core * S_tkg, S_ctx)
      mask_active: Active mask of shape (B, q_heads_per_core * S_tkg, S_tkg)
      position_ids: Position IDs of shape (B, S_tkg)
      eps: RMSNorm epsilon
      d_head: Head dimension
      q_heads_per_core: Number of Q heads per core

    Returns:
      Tuple of (output, updated_K_cache, updated_V_cache)
    """
    batch, S_tkg, H = X.shape
    S_ctx = K_cache.shape[2]

    # QKV projection with RMSNorm
    qkv_out = norm_qkv_torch(X, W_gamma, W_qkv, d_head, eps)  # (num_heads+2, B, S_tkg, d_head)

    # Apply RoPE to Q and K
    qk_out = torch.zeros((d_head, batch, q_heads_per_core + 1, S_tkg), dtype=X.dtype, device=X.device)
    for b in range(batch):
        for h in range(q_heads_per_core + 1):
            cur_head = qkv_out[h, b, :, :].T  # (d_head, S_tkg)
            qk_out[:, b, h, :] = rope_torch(cur_head, cos[:, b, :], sin[:, b, :])

    q_out = qk_out[:, :, :q_heads_per_core, :]  # (d_head, B, q_heads_per_core, S_tkg)
    k_out = qk_out[:, :, q_heads_per_core, :]  # (d_head, B, S_tkg)
    v = qkv_out[-1]  # (B, S_tkg, d_head)

    # Scale Q
    q_out = q_out / (d_head**0.5)

    # Attention per batch
    attn_out = torch.zeros((batch, d_head, q_heads_per_core * S_tkg), dtype=X.dtype, device=X.device)
    for b in range(batch):
        attn_out[b] = attention_tkg_torch(
            q_out[:, b, :, :],  # (d_head, q_heads_per_core, S_tkg)
            k_out[:, b, :],  # (d_head, S_tkg)
            v[b],  # (S_tkg, d_head)
            K_cache[b, :, :S_ctx],  # (d_head, S_ctx)
            V_cache[b, :S_ctx, :],  # (S_ctx, d_head)
            mask_cache[b],  # (q_heads_per_core * S_tkg, S_ctx)
            mask_active[b],  # (q_heads_per_core * S_tkg, S_tkg)
        )

    # Reshape and output projection
    # (B, d_head, q_heads_per_core * S_tkg) -> (B, S_tkg, q_heads_per_core, d_head) -> (B*S_tkg, q_heads_per_core * d_head)
    attn_out = (
        attn_out.reshape(batch, d_head, q_heads_per_core, S_tkg)
        .permute(0, 3, 2, 1)
        .reshape(batch * S_tkg, q_heads_per_core * d_head)
    )
    out = torch.matmul(attn_out, W_out)  # (B*S_tkg, H)
    out = out.reshape(batch, S_tkg, H)

    # Update KV cache
    K_cache_updated = K_cache.clone()
    V_cache_updated = V_cache.clone()
    S_tkg_local = k_out.shape[2]
    for b in range(batch):
        pos = position_ids[b]  # shape (S_tkg,) from input
        pos_len = pos.shape[0]
        for s in range(min(S_tkg_local, pos_len)):
            p = pos[s].item() if hasattr(pos[s], 'item') else int(pos[s])
            K_cache_updated[b, :, p] = k_out[:, b, s]
            V_cache_updated[b, p, :] = v[b, s, :]

    return out, K_cache_updated, V_cache_updated


def mlp_torch(
    hidden: torch.Tensor,
    gamma: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    eps: float,
    lnc: int,
    down_proj_layout_enabled: bool = True,
) -> torch.Tensor:
    """
    MLP block with RMSNorm (non-quantized path).

    Args:
      hidden: Input tensor of shape (B, S, H)
      gamma: RMSNorm gamma of shape (H,)
      gate: Gate projection weights of shape (H, I)
      up: Up projection weights of shape (H, I)
      down: Down projection weights of shape (I, H)
      eps: RMSNorm epsilon
      lnc: LNC parameter for layout transformation
      down_proj_layout_enabled: Whether down projection has optimized layout

    Returns:
      MLP output of shape (B, S, H)
    """
    # RMSNorm
    normed = rms_norm_torch(hidden, gamma, eps)

    # Handle down projection layout if enabled
    if down_proj_layout_enabled:
        I, H = down.shape
        down = down.reshape(I, lnc, H // 128 // lnc, 128).permute(0, 1, 3, 2).reshape(I, H)

    # Gate and Up projections
    gate_out = torch.matmul(normed, gate)
    up_out = torch.matmul(normed, up)

    # SiLU activation on gate, multiply with up
    gate_act = silu(gate_out)
    mult = gate_act * up_out

    # Down projection
    output = torch.matmul(mult, down)

    return output


def quantized_mlp_torch(
    hidden: torch.Tensor,
    gamma: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    gate_w_scale: torch.Tensor,
    up_w_scale: torch.Tensor,
    down_w_scale: torch.Tensor,
    eps: float,
    clamp_bound: float,
    lnc: int,
    down_proj_layout_enabled: bool = True,
) -> torch.Tensor:
    """
    Quantized MLP block with RMSNorm and FP8 row quantization simulation.

    Args:
      hidden: Input tensor of shape (B, S, H)
      gamma: RMSNorm gamma of shape (H,)
      gate: Gate projection weights (quantized) of shape (H, I)
      up: Up projection weights (quantized) of shape (H, I)
      down: Down projection weights (quantized) of shape (I, H)
      gate_w_scale: Gate weight scale of shape (1, I)
      up_w_scale: Up weight scale of shape (1, I)
      down_w_scale: Down weight scale of shape (1, H)
      eps: RMSNorm epsilon
      clamp_bound: Clipping boundary for row quantization (0 means no clipping)
      lnc: LNC parameter for layout transformation
      down_proj_layout_enabled: Whether down projection has optimized layout

    Returns:
      MLP output of shape (B, S, H)
    """
    FP8_E4M3_MAX = 240.0

    def row_quantize(x: torch.Tensor, clip_bound: float) -> tuple[torch.Tensor, torch.Tensor]:
        """Row-wise quantization simulation."""
        abs_max = x.abs().max(dim=-1, keepdim=True).values
        if clip_bound > 0:
            abs_max = abs_max.clamp(max=clip_bound)
            x = x.clamp(-clip_bound, clip_bound)
        min_scale = torch.tensor(1e-5, device=x.device, dtype=x.dtype)
        quant_scale = torch.max(abs_max / FP8_E4M3_MAX, min_scale)
        return x / quant_scale, quant_scale

    def scale_matmul(x: torch.Tensor, w: torch.Tensor, w_scale: torch.Tensor, x_scale: torch.Tensor) -> torch.Tensor:
        """Scaled matrix multiplication for row quantization."""
        result = torch.matmul(x, w.to(x.dtype))
        # Handle TG case where scale tensor is bigger than result
        if len(result.shape) == 3 and len(w_scale.shape) == 2:
            B, S, I = result.shape
            if B * S <= TKG_BS_SEQLEN_THRESHOLD:
                result = result * w_scale[:S, :]
            else:
                result = result * w_scale  # broadcast
        else:
            result = result * w_scale
        result = result * x_scale  # broadcast (B, S, 1)
        return result

    # RMSNorm
    normed = rms_norm_torch(hidden, gamma, eps)

    # Row quantize input
    quantized_input, input_scale = row_quantize(normed, clamp_bound)

    # Handle down projection layout if enabled
    if down_proj_layout_enabled:
        I, H = down.shape
        down = down.reshape(I, lnc, H // 128 // lnc, 128).permute(0, 1, 3, 2).reshape(I, H)

    # Gate and Up projections with scaling
    gate_out = scale_matmul(quantized_input, gate, gate_w_scale, input_scale)
    up_out = scale_matmul(quantized_input, up, up_w_scale, input_scale)

    # SiLU activation on gate, multiply with up
    gate_act = silu(gate_out)
    mult = gate_act * up_out

    # Row quantize intermediate
    quantized_mult, mult_scale = row_quantize(mult, clamp_bound)

    # Down projection with scaling
    output = scale_matmul(quantized_mult, down, down_w_scale, mult_scale)

    return output


def llama3_transformer_fwd_tkg_torch(
    X: torch.Tensor,
    W_qkvs: list[torch.Tensor],
    W_outs: list[torch.Tensor],
    W_gates: list[torch.Tensor],
    W_gate_scales: list[Optional[torch.Tensor]],
    W_ups: list[torch.Tensor],
    W_up_scales: list[Optional[torch.Tensor]],
    W_downs: list[torch.Tensor],
    W_down_scales: list[Optional[torch.Tensor]],
    W_gamma_qkvs: list[torch.Tensor],
    W_gamma_mlps: list[torch.Tensor],
    RoPE_cos: torch.Tensor,
    RoPE_sin: torch.Tensor,
    mask_cache: torch.Tensor,
    mask_active: torch.Tensor,
    position_ids: torch.Tensor,
    K_caches: list[torch.Tensor],
    V_caches: list[torch.Tensor],
    num_layers: int,
    replica_groups: list[list[int]],
    eps: float = 1e-6,
    clamp_bound: float = 0.0,
    K_cache_transposed: bool = True,
    sbuf_residual_and_cc: bool = True,
    use_cascaded_attn: bool = True,
    use_gpsimd_sb2sb: bool = True,
    output_KV_caches: bool = False,
    mlp_down_proj_layout_enabled: bool = True,
    use_bir_mlp_kernel: bool = True,
    DBG: bool = False,
    DBG_LNC1: bool = False,
):
    """
    Full PyTorch reference implementation of the transformer TKG megakernel.

    Implements the transformer layer loop:
    - For each layer: attention block -> residual add -> MLP -> residual add
    - Simulates all-reduce by multiplying by cc_workers
    """
    B, S_tkg, H = X.shape
    _, _, S_max_ctx, d_head = V_caches[0].shape
    dtype = X.dtype
    device = X.device

    cc_workers = len(replica_groups[0])
    lnc = 1 if DBG_LNC1 else 2

    if W_gate_scales is None:
        nonquantized_layers = list(range(num_layers))
    else:
        nonquantized_layers = {0, num_layers - 1}

    # Only cascaded attention is supported
    kernel_assert(use_cascaded_attn, "use_cascaded_attn must be True")

    # Cascaded: mask_cache shape is (S_ctx, batch, qheads_per_core, S_tkg)
    S_ctx = mask_cache.shape[0]
    q_heads_per_core = mask_cache.shape[2]

    # Reshape masks
    mask_cache_reshaped = mask_cache.reshape(B, q_heads_per_core * S_tkg, S_ctx)

    # Cascaded uses one mask for both prior and active, split them into two
    mask_active_reshaped = mask_cache_reshaped[:, :, S_ctx - S_tkg :].clone()
    mask_cache_reshaped = mask_cache_reshaped.clone()
    mask_cache_reshaped[:, :, S_ctx - S_tkg :] = 0

    # Storage for KV cache outputs
    K_cache_out = []
    V_cache_out = []

    cur_X = X.float()  # upcast to float32 for computation

    for layer in range(num_layers):
        residual = cur_X

        # Remove n_kv_heads dim from KV caches
        K_cache_layer = K_caches[layer].squeeze(1).float()
        V_cache_layer = V_caches[layer].squeeze(1).float()

        # Attention block
        attn_out, K_cache_updated, V_cache_updated = attention_block_torch(
            cur_X,
            W_gamma_qkvs[layer].float(),
            W_qkvs[layer].float(),
            W_outs[layer].float(),
            RoPE_cos.float(),
            RoPE_sin.float(),
            K_cache_layer[:, :, :S_ctx],
            V_cache_layer[:, :S_ctx, :],
            mask_cache_reshaped,
            mask_active_reshaped,
            position_ids,
            eps,
            d_head,
            q_heads_per_core,
        )

        # All-reduce simulation
        attn_out = cc_workers * attn_out

        # Residual add
        attn_out = attn_out + residual
        residual = attn_out

        # MLP - quantized or non-quantized based on layer
        if layer in nonquantized_layers:
            mlp_out = mlp_torch(
                attn_out,
                W_gamma_mlps[layer].float(),
                W_gates[layer].float(),
                W_ups[layer].float(),
                W_downs[layer].float(),
                eps,
                lnc,
                mlp_down_proj_layout_enabled,
            )
        else:
            mlp_out = quantized_mlp_torch(
                attn_out,
                W_gamma_mlps[layer].float(),
                W_gates[layer].float(),
                W_ups[layer].float(),
                W_downs[layer].float(),
                W_gate_scales[layer].float(),
                W_up_scales[layer].float(),
                W_down_scales[layer].float(),
                eps,
                clamp_bound,
                lnc,
                mlp_down_proj_layout_enabled,
            )

        # All-reduce simulation
        mlp_out = cc_workers * mlp_out

        # Residual add
        mlp_out = mlp_out + residual

        cur_X = mlp_out

        # Store updated KV caches (add n_kv_heads dim back)
        K_cache_out.append(K_cache_updated.unsqueeze(1))
        V_cache_out.append(V_cache_updated.unsqueeze(1))

    final_output = cur_X.to(dtype)
    return_values = [final_output]

    if output_KV_caches:
        for i in range(num_layers):
            return_values.append(K_cache_out[i].to(dtype))
        for i in range(num_layers):
            return_values.append(V_cache_out[i].to(dtype))

    # DBG outputs - return zeros for now as they require internal kernel state
    if DBG:
        H0 = 128
        H1 = H // H0
        n_prgs = 1 if DBG_LNC1 else 2
        H1_shard = H1 // n_prgs
        BxS = B * S_tkg

        for _ in range(num_layers):
            return_values.append(torch.zeros((B, S_tkg, H), dtype=dtype, device=device))  # DBG_layer_input
            if sbuf_residual_and_cc:
                return_values.append(torch.zeros((H0, H1_shard * BxS), dtype=dtype, device=device))
                return_values.append(torch.zeros((H0, H1_shard * BxS), dtype=dtype, device=device))
                return_values.append(torch.zeros((H0, H1_shard * BxS), dtype=dtype, device=device))
                return_values.append(torch.zeros((H0, H1_shard * BxS), dtype=dtype, device=device))
                return_values.append(torch.zeros((B, S_tkg, H), dtype=dtype, device=device))
            return_values.append(torch.zeros((B, S_tkg, H), dtype=dtype, device=device))
            if sbuf_residual_and_cc:
                return_values.append(torch.zeros((H0, H1_shard * BxS), dtype=dtype, device=device))
                return_values.append(torch.zeros((H0, H1_shard * BxS), dtype=dtype, device=device))
                return_values.append(torch.zeros((B, S_tkg, H), dtype=dtype, device=device))
            return_values.append(torch.zeros((B, S_tkg, H), dtype=dtype, device=device))

    return tuple(return_values) if len(return_values) > 1 else return_values[0]
