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

"""PyTorch reference implementation for Flash Attention backward pass kernel."""

from typing import Optional, Tuple, Union

import torch

from ..utils.kernel_assert import kernel_assert

PMAX = 128


def attention_bwd_torch_ref(
    q_ref: torch.Tensor,
    k_ref: torch.Tensor,
    v_ref: torch.Tensor,
    o_ref: torch.Tensor,
    dy_ref: torch.Tensor,
    lse_ref: torch.Tensor,
    sinks_ref: Optional[torch.Tensor] = None,
    use_causal_mask: bool = False,
    mixed_precision: bool = False,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
) -> dict:
    """
    PyTorch reference implementation of Flash Attention backward pass.

    Signature matches the attention_bwd kernel for consistency.

    Args:
        q_ref (torch.Tensor): Query tensor of shape (B, Hq, d, seqlen_q)
        k_ref (torch.Tensor): Key tensor of shape (B, Hkv, d, seqlen_k)
        v_ref (torch.Tensor): Value tensor of shape (B, Hkv, d, seqlen_k)
        o_ref (torch.Tensor): Forward pass output (unused, needed for kernel signature match)
        dy_ref (torch.Tensor): Gradient of output of shape (B, Hq, d, seqlen_q)
        lse_ref (torch.Tensor): Log-sum-exp from forward pass (unused, needed for kernel signature match)
        sinks_ref (Optional[torch.Tensor]): Optional attention sinks of shape (B, Hq) or (B, Hq, num_sinks)
        use_causal_mask (bool): Whether to apply causal masking
        mixed_precision (bool): Whether to use mixed precision for reductions
        softmax_scale (Optional[float]): Scale factor for softmax (default: 1/sqrt(d))
        sliding_window (Optional[int]): Sliding window size

    Returns:
        dict with keys: out_dq_ref, out_dk_ref, out_dv_ref, and optionally out_dsinks_ref
    """
    B, Hq, d, _ = q_ref.shape
    Hkv = k_ref.shape[1]
    k_expanded = expand_kv_heads(k_ref, Hq)
    v_expanded = expand_kv_heads(v_ref, Hq)

    sliding_window = min(sliding_window or 0, k_ref.shape[-1])
    softmax_scale = softmax_scale or (1.0 / (d**0.5))
    q_scaled = q_ref * softmax_scale

    _, _, norm_score = compute_o_lse(
        q_ref,
        k_ref,
        v_ref,
        use_causal_mask,
        mixed_precision,
        softmax_scale=softmax_scale,
        sliding_window=sliding_window,
        sinks=sinks_ref,
    )

    softmax_dy = mixed_precision_matmul(dy_ref.permute(0, 1, 3, 2), v_expanded)

    # Calculate softmax_dx
    if sinks_ref is not None:
        num_sinks = sinks_ref.shape[-1] if len(sinks_ref.shape) == 3 else 1
        softmax_dy_padded = torch.cat([softmax_dy, torch.zeros_like(norm_score[..., -num_sinks:])], axis=-1)
        softmax_dx_golden = softmax_dx(softmax_dy_padded, norm_score, dim=-1, mixed_precision=mixed_precision)
        norm_score = norm_score[..., :-num_sinks]

        softmax_dx_golden, dsinks_golden = (
            softmax_dx_golden[..., :-num_sinks],
            softmax_dx_golden[..., -num_sinks:],
        )
        dsinks_golden = dsinks_golden.sum(dim=2)
    else:
        softmax_dx_golden = softmax_dx(softmax_dy, norm_score, dim=-1, mixed_precision=mixed_precision)

    dv_golden = mixed_precision_matmul(dy_ref, norm_score)
    dq_golden = mixed_precision_matmul(k_expanded, softmax_dx_golden.permute(0, 1, 3, 2)) * softmax_scale
    dk_golden = mixed_precision_matmul(q_scaled, softmax_dx_golden)

    # Reduce gradients for GQA
    dk_golden = reduce_kv_grad(dk_golden, Hkv, mixed_precision)
    dv_golden = reduce_kv_grad(dv_golden, Hkv, mixed_precision)

    dq_golden = dq_golden.to(q_ref.dtype)
    dk_golden = dk_golden.to(k_ref.dtype)
    dv_golden = dv_golden.to(v_ref.dtype)

    result = {
        "out_dq_ref": dq_golden,
        "out_dk_ref": dk_golden,
        "out_dv_ref": dv_golden,
    }
    if sinks_ref is not None:
        result["out_dsinks_ref"] = dsinks_golden.to(sinks_ref.dtype)
    return result


def compute_o_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    use_causal_mask: bool = True,
    mixed_precision: bool = True,
    softmax_scale: Optional[float] = None,
    logit_bias: Optional[torch.Tensor] = None,
    sliding_window: Optional[int] = None,
    dropout_mask: Optional[torch.Tensor] = None,
    sinks: Optional[torch.tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Compute forward attention output, log-sum-exp, and normalized scores.

    Args:
        q: Query tensor of shape (B, Hq, d, seqlen_q)
        k: Key tensor of shape (B, Hkv, d, seqlen_k)
        v: Value tensor of shape (B, Hkv, d, seqlen_k)
        use_causal_mask: Whether to apply causal masking
        mixed_precision: Whether to use float32 for reductions
        softmax_scale: Scale factor (default: 1/sqrt(d))
        logit_bias: Optional additive bias for attention scores
        sliding_window: Sliding window size
        dropout_mask: Optional dropout mask
        sinks: Attention sinks of shape (B, Hq)

    Returns:
        Tuple of (o_proj, lse, norm_score):
            - o_proj: Output, shape (B, Hq, d, seqlen_q)
            - lse: Log-sum-exp, shape (B, Hq, PMAX, seqlen_q // PMAX)
            - norm_score: Attention weights, shape (B, Hq, seqlen_q, seqlen_k)

    Note:
        Assumes seqlen_q is divisible by PMAX (128)
    """

    B, Hq, d, seqlen_q = q.shape
    input_dtype = q.dtype
    Hkv, seqlen_k = k.shape[1], k.shape[3]
    sliding_window = sliding_window or 0

    # Expand (B, Hkv, d, S) -> (B, Hq, d, S)
    k = expand_kv_heads(k, Hq)
    v = expand_kv_heads(v, Hq)

    # Compute golden output
    softmax_scale = softmax_scale or (1.0 / (d**0.5))
    q_scaled = q * softmax_scale
    raw_score = torch.matmul(q_scaled.permute(0, 1, 3, 2).to(torch.float32), k.to(torch.float32))

    # Create combined mask
    mask = torch.zeros(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool)
    if use_causal_mask:
        mask |= torch.triu(torch.ones(seqlen_q, seqlen_k, device=q.device, dtype=torch.bool), diagonal=1)
    if sliding_window > 0:
        q_pos = torch.arange(seqlen_q, device=q.device).unsqueeze(1)
        k_pos = torch.arange(seqlen_k, device=q.device).unsqueeze(0)
        mask |= k_pos < (q_pos - sliding_window + 1)

    # Apply mask once with broadcasting (mask broadcasts from (seqlen_q, seqlen_k) to (B, Hq, seqlen_q, seqlen_k))
    if use_causal_mask or sliding_window > 0:
        raw_score = raw_score.masked_fill(mask, -float("inf"))

    # Assume logit_bias can be broadcasted
    if logit_bias is not None:
        raw_score = raw_score + logit_bias

    if sinks is not None:
        num_sinks = sinks.shape[-1] if len(sinks.shape) == 3 else 1
        sinks_expanded = sinks.reshape(B, Hq, 1, num_sinks).expand(B, Hq, seqlen_q, num_sinks)
        raw_score = torch.cat([raw_score, sinks_expanded], axis=-1)

    norm_score, cached_negative_max, cached_sum_reciprocal = softmax(
        raw_score, dim=-1, mixed_precision=mixed_precision, return_max_reduce=True
    )

    norm_score = norm_score.to(input_dtype)
    norm_score_orig = norm_score
    if sinks is not None:
        norm_score = norm_score[..., :-num_sinks]

    if dropout_mask is not None:  # we assume dropout mask already includes 1/(1-dropout_p) scaling
        norm_score_dropped = norm_score * dropout_mask
    else:
        norm_score_dropped = norm_score

    # Calculate output projection
    o_proj = torch.matmul(norm_score_dropped, v.permute(0, 1, 3, 2)).permute(0, 1, 3, 2)
    o_proj = o_proj.contiguous()

    # Calculate lse
    kernel_assert(
        seqlen_q % PMAX == 0, f"Query sequence length should be divisble by {PMAX} for softmax cache, got {seqlen_q}"
    )
    cached_negative_max = cached_negative_max.squeeze(-1).reshape(B, Hq, seqlen_q // PMAX, PMAX).permute(0, 1, 3, 2)
    cached_sum_reciprocal = cached_sum_reciprocal.squeeze(-1).reshape(B, Hq, seqlen_q // PMAX, PMAX).permute(0, 1, 3, 2)

    lse = -1 * (cached_negative_max + torch.log(cached_sum_reciprocal))
    return o_proj.to(input_dtype), lse, norm_score_orig


def mixed_precision_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Perform matrix multiplication in float32 and cast back to input dtype"""
    input_dtype = a.dtype
    a, b = a.to(torch.float32), b.to(torch.float32)
    c = torch.matmul(a, b)
    return c.to(input_dtype)


def expand_kv_heads(x: torch.Tensor, num_q_heads: int) -> torch.Tensor:
    """Expand KV heads for GQA: (B, Hkv, d, S) -> (B, Hq, d, S)"""
    kernel_assert(
        num_q_heads % x.shape[1] == 0,
        (f"Query heads {num_q_heads} should be equal to or divisible by key/value heads ({x.shape[1]})"),
    )
    num_groups = num_q_heads // x.shape[1]
    return x.repeat_interleave(num_groups, dim=1)


def reduce_kv_grad(dx: torch.Tensor, num_kv_heads: int, mixed_precision: bool) -> torch.Tensor:
    """Sum KV head gradients for GQA: (B, Hq, d, S) -> (B, Hkv, d, S)"""
    B, Hq, d, S = dx.shape
    num_groups = Hq // num_kv_heads
    kernel_assert(
        Hq % num_kv_heads == 0,
        (f"Query heads {Hq} should be equal to or divisible by key/value heads ({num_kv_heads})"),
    )
    if mixed_precision:
        input_dtype = dx.dtype
        return dx.reshape(B, num_kv_heads, num_groups, d, S).float().sum(dim=2).to(input_dtype)
    return dx.reshape(B, num_kv_heads, num_groups, d, S).sum(dim=2)


def softmax(
    x: torch.Tensor,
    dim: Union[int, Tuple[int, ...]],
    zero_max_mode: bool = False,
    mixed_precision: bool = False,
    return_max_reduce: bool = False,
) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """
    Numerically stable softmax with optional statistics caching.

    Args:
        x: Input tensor
        dim: Dimension(s) for softmax computation
        zero_max_mode: Use max(0, max(x)) for numerical stability
        mixed_precision: Compute sum in float32
        return_max_reduce: Return (softmax, -max, 1/sum) for backward

    Returns:
        Softmax tensor, or tuple with cached statistics if return_max_reduce=True
    """
    max_value = torch.amax(x, dim=dim, keepdim=True)
    if zero_max_mode:
        max_value = torch.maximum(torch.zeros_like(max_value), max_value)
    exp = torch.exp(x - max_value)
    if mixed_precision:
        reduce = exp.to(torch.float32).sum(dim=dim, keepdim=True).to(x.dtype)
    else:
        reduce = exp.sum(dim=dim, keepdim=True)
    if return_max_reduce:
        return exp / reduce, -max_value, torch.reciprocal(reduce)
    return exp / reduce


def softmax_dx(
    dy: torch.Tensor,
    y: torch.Tensor,
    dim: Union[int, Tuple[int, ...]],
    mixed_precision: bool = False,
) -> torch.Tensor:
    """
    Compute softmax backward: dx_i = y_i * (dy_i - sum_j(dy_j * y_j)).

    Args:
        dy: Upstream gradient
        y: Softmax output from forward pass
        dim: Dimension(s) of softmax computation
        mixed_precision: Compute reduction in float32

    Returns:
        Gradient with respect to softmax input
    """
    prod = dy * y
    if mixed_precision:
        reduce = prod.to(torch.float32).sum(dim=dim, keepdim=True).to(dy.dtype)
    else:
        reduce = prod.sum(dim=dim, keepdim=True)
    subtract = dy - reduce
    return subtract * y
