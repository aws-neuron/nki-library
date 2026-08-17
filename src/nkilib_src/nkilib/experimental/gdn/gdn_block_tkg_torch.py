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

"""PyTorch reference for the fused GDN DECODE block (gdn_block_decode_fused).

Composes, matching qwen3_5 model_bf16.py _forward_decode fused path (1636-1690)
MINUS out_proj:
  in_proj matmuls (q|k|v, z, a, b) -> conv1d_decode(+silu+state) -> split+GQA ->
  recurrent gated-delta-rule + RMSNormGated (== gdn_tkg math).

Weights are taken in KERNEL layout [H, out_features] (i.e. nn.Linear.weight.T)
so the reference and the NKI kernel agree exactly. The caller that wires this
into the model would pass self.in_proj_*.weight.T (or store transposed).
"""

import torch
import torch.nn.functional as F

from .gdn_conv1d_torch import gdn_conv1d_decode_torch_ref


def _l2norm(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalize ``x`` along ``dim`` with an ``eps`` floor on the norm."""
    return x * torch.rsqrt((x * x).sum(dim=dim, keepdim=True) + eps)


def gdn_block_decode_fused_torch_ref(
    hidden: torch.Tensor,  # [B, 1, H]
    W_in_qkv: torch.Tensor,  # [H, 2*key_dim + value_dim]
    W_in_z: torch.Tensor,  # [H, value_dim]
    W_in_a: torch.Tensor,  # [H, num_v_heads]
    W_in_b: torch.Tensor,  # [H, num_v_heads]
    conv_weight: torch.Tensor,  # [conv_dim, K_win]
    conv_state: torch.Tensor,  # [B, conv_dim, K_win]
    A_log: torch.Tensor,  # [num_v_heads]
    dt_bias: torch.Tensor,  # [num_v_heads]
    norm_weight: torch.Tensor,  # [head_v_dim]
    recurrent_state: torch.Tensor,  # [B, num_v_heads, head_k_dim, head_v_dim]
    num_k_heads: int,
    num_v_heads: int,
    head_k_dim: int,
    head_v_dim: int,
):
    """PyTorch reference for the fused GDN decode block (gdn_block_decode_fused).

    Reference implementation used to validate the NKI gdn_block_tkg kernel. Composes
    the fused decode path (in_proj -> depthwise causal conv1d + silu -> GQA split ->
    recurrent gated delta rule + RMSNormGated) for a single decode token (T == 1),
    matching qwen3_5 model_bf16.py _forward_decode (minus out_proj). All recurrent /
    normalization math is done in float32 for CPU-reference stability.

    Dimensions:
        B: Batch size
        T: Query sequence length (must be 1 for decode)
        H: Hidden dimension size
        key_dim: num_k_heads * head_k_dim
        value_dim: num_v_heads * head_v_dim
        conv_dim: 2 * key_dim + value_dim
        K_win: Causal conv1d window length

    Args:
        hidden (torch.Tensor): [B, 1, H], decode-step hidden states.
        W_in_qkv (torch.Tensor): [H, 2*key_dim + value_dim], q|k|v in-projection weight
            in kernel layout (nn.Linear.weight.T).
        W_in_z (torch.Tensor): [H, value_dim], gate (z) in-projection weight.
        W_in_a (torch.Tensor): [H, num_v_heads], "a" in-projection weight.
        W_in_b (torch.Tensor): [H, num_v_heads], "b" in-projection weight.
        conv_weight (torch.Tensor): [conv_dim, K_win], depthwise conv1d weights.
        conv_state (torch.Tensor): [B, conv_dim, K_win], previous conv1d window.
        A_log (torch.Tensor): [num_v_heads], log decay parameter per v-head.
        dt_bias (torch.Tensor): [num_v_heads], timestep bias per v-head.
        norm_weight (torch.Tensor): [head_v_dim], RMSNormGated scale.
        recurrent_state (torch.Tensor): [B, num_v_heads, head_k_dim, head_v_dim],
            previous recurrent state.
        num_k_heads (int): Number of key heads.
        num_v_heads (int): Number of value heads.
        head_k_dim (int): Key head dimension.
        head_v_dim (int): Value head dimension.

    Returns:
        core_out (torch.Tensor): [B, 1, value_dim], fused decode-block output.
        new_conv_state (torch.Tensor): [B, conv_dim, K_win], updated conv1d window.
        new_recurrent_state (torch.Tensor): [B, num_v_heads, head_k_dim, head_v_dim],
            updated recurrent state.

    Notes:
        - Decode only: raises ValueError unless T == 1.
        - Weights are expected in KERNEL layout [H, out_features] (nn.Linear.weight.T)
          so the reference and the NKI kernel agree exactly.
        - GQA (num_v_heads > num_k_heads) is handled by repeat_interleave of q, k.

    Pseudocode:
        proj_qkv, z, a, b = hidden @ (W_in_qkv, W_in_z, W_in_a, W_in_b)
        q, k, v = split(proj_qkv into key_dim, key_dim, value_dim)
        mixed = concat(q, k, v) transposed to [B, conv_dim, 1]
        conv_out, new_conv_state = conv1d_decode(mixed, conv_state, conv_weight)  # + silu
        q, k, v = split(conv_out); GQA-repeat q, k to num_v_heads
        qf = l2norm(q); kf = l2norm(k); q_s = qf * (1 / sqrt(head_k_dim))
        beta = sigmoid(b); g = -exp(A_log) * softplus(a + dt_bias)
        S = recurrent_state * exp(g)
        kv = sum(S * kf, dim=-2); delta = (v - kv) * beta
        S = S + kf * delta
        out = sum(S * q_s, dim=-2)
        out = out * rsqrt(mean(out^2) + eps) * norm_weight * silu(z)
        core_out = out reshaped to [B, 1, value_dim]
    """
    B, T, H = hidden.shape
    if T != 1:
        raise ValueError(f"decode expects T == 1, got T={T}")
    key_dim = num_k_heads * head_k_dim
    value_dim = num_v_heads * head_v_dim
    conv_dim = 2 * key_dim + value_dim
    gqa = num_v_heads // num_k_heads

    # ---- in_proj matmuls (hidden @ W, W in [H, out]) ----
    proj_qkv = hidden @ W_in_qkv  # [B, 1, conv_dim]
    z = hidden @ W_in_z  # [B, 1, value_dim]
    a = hidden @ W_in_a  # [B, 1, num_v_heads]
    b = hidden @ W_in_b  # [B, 1, num_v_heads]

    q_flat = proj_qkv[:, :, :key_dim]
    k_flat = proj_qkv[:, :, key_dim : 2 * key_dim]
    v_flat = proj_qkv[:, :, 2 * key_dim :]
    mixed = torch.cat([q_flat, k_flat, v_flat], dim=-1).transpose(1, 2).contiguous()
    #  mixed: [B, conv_dim, 1]

    # ---- depthwise causal conv1d update + silu ----
    conv_out, new_conv_state = gdn_conv1d_decode_torch_ref(mixed, conv_state, conv_weight)
    conv_out = conv_out.transpose(1, 2).contiguous()  # [B, 1, conv_dim]

    q = conv_out[:, :, :key_dim].view(B, T, num_k_heads, head_k_dim)
    k = conv_out[:, :, key_dim : 2 * key_dim].view(B, T, num_k_heads, head_k_dim)
    v = conv_out[:, :, 2 * key_dim :].view(B, T, num_v_heads, head_v_dim)
    z = z.view(B, T, num_v_heads, head_v_dim)

    # ---- GQA repeat q,k to num_v_heads ----
    if gqa > 1:
        q = q.repeat_interleave(gqa, dim=2)
        k = k.repeat_interleave(gqa, dim=2)

    BH = B * num_v_heads
    q_bh = q.reshape(BH, head_k_dim)
    k_bh = k.reshape(BH, head_k_dim)
    v_bh = v.reshape(BH, head_v_dim)
    z_bh = z.reshape(BH, head_v_dim)
    b_bh = b.reshape(BH)
    a_bh = a.reshape(BH)
    state_bh = recurrent_state.reshape(BH, head_k_dim, head_v_dim)

    # ---- gdn_tkg math (folds l2norm/beta/g/RMSNormGated + recurrence) ----
    K = head_k_dim
    V = head_v_dim
    scale = 1.0 / (K**0.5)
    qf = _l2norm(q_bh.float())
    kf = _l2norm(k_bh.float())
    q_s = qf * scale

    beta = torch.sigmoid(b_bh.float())
    A_log_bh = A_log.float().repeat(B)  # v-head cycles fastest per batch
    dt_bh = dt_bias.float().repeat(B)
    g = -A_log_bh.exp() * F.softplus(a_bh.float() + dt_bh)

    exp_g = g.exp().unsqueeze(-1).unsqueeze(-1)
    S = state_bh.float() * exp_g
    kv = (S * kf.unsqueeze(-1)).sum(dim=-2)
    delta = (v_bh.float() - kv) * beta.unsqueeze(-1)
    S = S + kf.unsqueeze(-1) * delta.unsqueeze(-2)
    out = (S * q_s.unsqueeze(-1)).sum(dim=-2)  # [BH, V]

    rms = torch.rsqrt((out * out).mean(dim=-1, keepdim=True) + 1e-6)
    silu_z = z_bh.float() * torch.sigmoid(z_bh.float())
    out = out * rms * norm_weight.float().unsqueeze(0) * silu_z

    core_out = out.reshape(B, T, value_dim)
    new_recurrent_state = S.reshape(B, num_v_heads, head_k_dim, head_v_dim)
    return core_out, new_conv_state, new_recurrent_state
