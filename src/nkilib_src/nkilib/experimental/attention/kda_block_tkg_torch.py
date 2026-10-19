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

"""Torch reference for the full Kimi Delta Attention (KDA) block, token generation.

Mirrors ``kda_block_tkg`` op for op: input RMSNorm, the fused input projection, the
causal short convolutions with SiLU, the low-rank forget gate, the KDA recurrence, the
gated output RMSNorm, and the output projection. The caller keeps the TP collective and
the residual add, so neither appears here.

Structure follows ``experimental/gdn/gdn_block_tkg_torch.py`` (the GDN block reference);
the recurrence and gate math follow ``kda_tkg_torch.py``, which is validated against the
KDA recurrence kernel. All math runs in float32 for CPU-reference stability.
"""

import torch
import torch.nn.functional as F

# Short-convolution streams, in the order they are stacked in conv_w / conv_state.
_CONV_STREAMS = 3


def _l2norm(x: torch.Tensor, l2_eps: float, dim: int = -1) -> torch.Tensor:
    """L2 normalize with the epsilon inside the sqrt, matching the kernel's rsqrt(ss + eps)."""
    return x * torch.rsqrt(x.pow(2).sum(dim=dim, keepdim=True) + l2_eps)


def _conv_decode(x: torch.Tensor, state: torch.Tensor, w: torch.Tensor) -> tuple:
    """One decode step of a depthwise causal conv1d, plus SiLU.

    Args:
        x:     [B, C, 1]      this token's raw projected channels.
        state: [B, C, K - 1]  the previous window, i.e. positions t-(K-1) .. t-1.
        w:     [C, K]         per-channel taps, tap 0 oldest.

    Returns:
        out:       [B, C, 1]      silu(conv1d(window ending at t)).
        new_state: [B, C, K - 1]  the window shifted forward by one token.
    """
    channels, taps = w.shape
    window = torch.cat([state, x], dim=-1)  # [B, C, K] -- already the window ending at t
    out = F.conv1d(window, w.reshape(channels, 1, taps), bias=None, padding=0, groups=channels)
    return F.silu(out), window[:, :, 1:].contiguous()


def kda_block_tkg_torch_ref(
    hidden: torch.Tensor,  # [B, n, H]
    gamma_in: torch.Tensor,  # [1, H]
    W_in: torch.Tensor,  # [H, I]
    W_fb: torch.Tensor,  # [dk, dk * num_head]         head-outer columns
    W_gb: torch.Tensor,  # [dk, dv * num_head]         channel-outer columns
    conv_w: torch.Tensor,  # [3, K, dk * num_head]
    A_log: torch.Tensor,  # [1, num_head]
    dt_bias: torch.Tensor,  # [dk, num_head]
    gamma_o: torch.Tensor,  # [1, dv]
    W_o: torch.Tensor,  # [dv * num_head, H]
    out: torch.Tensor,  # [B, n, H]                    placeholder; kernel destination
    S: torch.Tensor,  # [B, num_head, dk, dv]
    conv_state: torch.Tensor,  # [B, 3, K - 1, dk * num_head]
    scale: torch.Tensor,  # [1, 1]
    lower_bound: torch.Tensor,  # [1, 1]
    rms_norm_eps: float = 1.0e-5,
    l2_eps: float = 1.0e-6,
) -> dict:
    """Full KDA block for token generation: hidden states in, attention output out.

    Layout contract, which must match the kernel's weight preparation exactly:
        W_in columns are q | k | v | g_out | beta | f_a. Within each per-head block the
        order differs, because it is what selects the kernel's post-transpose layout:
          - q and k are CHANNEL-outer (d * num_head + h)
          - v and g_out are HEAD-outer (h * dv + d)
        W_fb's columns are HEAD-outer (h * dk + d) so each head's dk columns are
        contiguous, which is what lets the kernel skip a transpose on the forget gate.
        W_gb's columns are CHANNEL-outer (d * num_head + h): the gated norm wants the
        output gate head-major, so the kernel tiles that matmul over dv instead.
        BOTH gates are low-rank -- the output gate is g_a -> W_gb, not a full-width
        W_in block, matching g_a_proj/g_b_proj in the reference.
        conv_w[i] and conv_state[:, i] follow the same channel order as the W_in block
        they filter, so streams 0/1 are channel-outer and stream 2 is head-outer.
        W_o's rows are head-outer, matching the flattened output of the gated norm.

    Returns:
        dict with
            out:        [B, n, H]                     attention output (TP partial)
            S:          [B, num_head, dk, dv]         updated recurrent state
            conv_state: [B, 3, K - 1, dk * num_head]  updated convolution windows
    """
    del out  # the kernel writes into this; the reference returns fresh tensors

    batch, n_tok, hidden_size = hidden.shape
    if n_tok != 1:
        raise ValueError(f"decode expects n == 1, got n={n_tok}")
    num_head = A_log.shape[-1]
    dk = dt_bias.shape[0]
    dv = S.shape[-1]
    taps = conv_w.shape[1]
    i_qk = dk * num_head
    i_v = dv * num_head

    scale_f = scale.to(torch.float32).reshape(-1)[0]
    lower_bound_f = lower_bound.to(torch.float32).reshape(-1)[0]

    # ---- input RMSNorm ----
    h = hidden.to(torch.float32)
    h = h * torch.rsqrt(h.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps)
    h = h * gamma_in.to(torch.float32)

    # ---- fused input projection ----
    proj = h @ W_in.to(torch.float32)  # [B, n, I]
    off_v = 2 * i_qk
    off_g = off_v + i_v  # g_a latent, dk wide
    off_beta = off_g + dk
    off_fa = off_beta + num_head

    q_flat = proj[..., 0:i_qk]
    k_flat = proj[..., i_qk : 2 * i_qk]
    v_flat = proj[..., off_v : off_v + i_v]
    g_a = proj[..., off_g : off_g + dk]
    beta_logit = proj[..., off_beta : off_beta + num_head]
    f_a = proj[..., off_fa : off_fa + dk]

    # ---- low-rank gates: dk-wide latent -> per-head-per-channel logits ----
    f_gate_flat = f_a @ W_fb.to(torch.float32)  # [B, n, dk * num_head], head-outer
    g_out_flat = g_a @ W_gb.to(torch.float32)  # [B, n, dv * num_head], channel-outer

    # ---- causal short conv + SiLU, on the flat channel vectors ----
    # conv_w is tap-major [3, K, C] and conv_state position-major [B, 3, K-1, C]; both
    # transpose to the [C, K] / [B, C, K-1] shape the depthwise conv1d wants.
    conv_w_f = conv_w.to(torch.float32)
    conv_state_f = conv_state.to(torch.float32)
    conv_out = []
    new_conv_state = []
    for stream, flat in enumerate((q_flat, k_flat, v_flat)):
        y, new_state = _conv_decode(
            flat.transpose(1, 2).contiguous(),  # [B, C, n]
            conv_state_f[:, stream].transpose(1, 2).contiguous(),  # [B, C, K-1]
            conv_w_f[stream].t().contiguous(),  # [C, K]
        )
        conv_out.append(y.transpose(1, 2))  # [B, n, C]
        new_conv_state.append(new_state.transpose(1, 2))  # [B, K-1, C]
    q_flat, k_flat, v_flat = conv_out
    conv_state_out = torch.stack(new_conv_state, dim=1)  # [B, 3, K - 1, C]

    # ---- to head-major [B, num_head, d], honouring each block's column order ----
    def _channel_outer(x, d):
        return x.reshape(batch, d, num_head).transpose(1, 2)  # [B, num_head, d]

    def _head_outer(x, d):
        return x.reshape(batch, num_head, d)  # [B, num_head, d]

    q_hm = _channel_outer(q_flat, dk)
    k_hm = _channel_outer(k_flat, dk)
    f_gate = _head_outer(f_gate_flat, dk)
    v_hm = _head_outer(v_flat, dv)
    g_out = _channel_outer(g_out_flat, dv)
    beta_logit = beta_logit.reshape(batch, num_head, 1)

    # ---- preprocessing (mirrors kda_tkg_torch) ----
    dt_bias_hm = dt_bias.to(torch.float32).t()  # [num_head, dk]
    a_log_hm = A_log.to(torch.float32).reshape(num_head, 1)
    g_act = lower_bound_f * torch.sigmoid(torch.exp(a_log_hm) * (f_gate + dt_bias_hm))
    exp_g = torch.exp(g_act)  # [B, num_head, dk]
    q_n = _l2norm(q_hm, l2_eps) * scale_f
    k_n = _l2norm(k_hm, l2_eps)
    beta_f = torch.sigmoid(beta_logit)

    # ---- recurrence, one token ----
    S_in = S.to(torch.float32)
    kv = torch.einsum("bhk,bhkv->bhv", k_n * exp_g, S_in)
    diff = (v_hm - kv) * beta_f
    S_out = S_in * exp_g.unsqueeze(-1) + k_n.unsqueeze(-1) * diff.unsqueeze(-2)
    o = torch.einsum("bhk,bhkv->bhv", q_n, S_out)  # [B, num_head, dv]

    # ---- gated output RMSNorm: normalize over dv, THEN gate (matches the GDN block ref) ----
    o = o * torch.rsqrt(o.pow(2).mean(dim=-1, keepdim=True) + rms_norm_eps)
    o = o * gamma_o.to(torch.float32).reshape(1, 1, dv) * torch.sigmoid(g_out)

    # ---- output projection; head-outer flatten to match W_o's row order ----
    out_f = o.reshape(batch, n_tok, num_head * dv) @ W_o.to(torch.float32)  # [B, n, H]

    del hidden_size, taps  # derived for documentation of the contract only
    return {"out": out_f, "S": S_out, "conv_state": conv_state_out}
