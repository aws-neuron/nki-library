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

"""Numpy reference + decode helpers for ``rmsnorm_mx_prefill``.

The kernel emits a packed ``[T, H + scale_region]`` fp8 tensor:
    - ``[:, :H]``               : dense fp8 quantized normalized activations (natural H order)
    - ``[:, H:H+scale_region]`` : MX uint8 scales (folded when ``pack_scales``)

Validation strategy is a **dequant round-trip**: the comparator decodes the kernel's packed
fp8 + scales back to float and compares against the fp32 RMSNorm reference. This avoids having to
bit-match the exact on-chip scale layout, while still exercising the full quant pipeline.
"""

import neuron_dtypes as dt
import nki.language as nl
import numpy as np
import torch

from ..utils.common_types import RouterActFnType

_H0 = 128
_Q_WIDTH = 4
_K_BLOCK = 512
_SCALES_PER_BLOCK = 4
_FP8_E4M3_MAX_EXP = 7  # quantize_to_mx max_exp for float8_e4m3fn


def _to_torch_dtype(dtype) -> torch.dtype:
    """Map a numpy / neuron_dtypes 16-bit dtype to its torch equivalent (bf16/fp16 are IEEE in both).

    Only the round-trip dtypes used here (input dtype, router compute dtype) need mapping; matching by
    name keeps it robust across np.dtype and neuron_dtypes/nl dtype objects.
    """
    name = str(dtype)
    if "bfloat16" in name:
        return torch.bfloat16
    if "float16" in name:
        return torch.float16
    if "float32" in name:
        return torch.float32
    raise ValueError(f"unsupported round-trip dtype for the torch ref: {dtype}")


def swizzle_h_index(H: int) -> np.ndarray:
    """Router-weight permutation: ht-major swizzle slot -> original H index = h512*512 + 4*p + q.

    Pre-permutes the ROUTER WEIGHTS into the kernel's internal swizzle H order (matches the offline
    host weight-prep). Hidden states are NOT permuted; the ref un-permutes the weights with this index
    so it can matmul against natural-layout norm.
    """
    num_h512 = H // _K_BLOCK
    h512 = np.arange(num_h512).reshape(num_h512, 1, 1)
    q = np.arange(_Q_WIDTH).reshape(1, _Q_WIDTH, 1)
    p = np.arange(_H0).reshape(1, 1, _H0)
    return (h512 * _K_BLOCK + _Q_WIDTH * p + q).reshape(-1).astype(np.int64)


def rmsnorm_mx_prefill_torch_ref(
    hidden_states,
    gamma,
    router_weights=None,
    router_bias=None,
    eps: float = 1e-6,
    top_k: int = 1,
    router_act_fn: RouterActFnType = RouterActFnType.SIGMOID,
    n_group: int = 1,
    topk_group: int = 1,
    routed_scaling_factor: float = 1.0,
    qmx_output_dtype=nl.float8_e4m3fn_x4,  # noqa: ARG001 - dtype-only; ref works in fp32, signature parity with kernel
    pack_scales: bool = True,  # noqa: ARG001 - layout-only; does not change the float reference
    pack_affinities: bool = False,  # noqa: ARG001 - output-layout-only; does not change the float reference
    unpadded_hidden_size: int = None,
    residual=None,
    emit_norm_bf16: bool = False,
):
    """Reference for ``rmsnorm_mx_prefill``, with toggles mirroring the kernel's optional outputs.

    Always computes the RMSNorm path; the residual and router halves activate on the same conditions
    as the kernel (residual when ``residual`` is set, router when ``router_weights`` is set), and the
    returned dict carries only the keys those toggles produce -- so callers can match the kernel's
    variable output set directly.

    Args:
        hidden_states (np.ndarray): [B, S, H] bf16 input.
        gamma (np.ndarray): [1, H] or [H] RMSNorm weights.
        eps (float): epsilon.
        pack_scales (bool): folded scale layout flag (does not change the float reference).
        unpadded_hidden_size (int): unpadded hidden size for the mean denominator. The sum-of-squares is
            still taken over the full (zero-padded) H; only the divisor is unpadded_hidden_size. Defaults to H.
        residual (np.ndarray): [B, S, H] optional residual added (in the input dtype) before the norm.
        router_weights (np.ndarray): [H, E] PRE-PERMUTED router weights (swizzle H order), or None to
            skip the router. When set, the ref un-permutes them via swizzle_h_index to matmul natural norm.
        router_bias (np.ndarray): [1, E] optional router bias.
        top_k (int): number of experts selected per token (router path only).
        router_act_fn (RouterActFnType): SOFTMAX (over the top-K logits) or SIGMOID (per-logit).
        qmx_output_dtype / pack_scales / pack_affinities: dtype/layout-only kernel args, accepted for
            signature parity; the fp32 reference does not depend on them.

    The kernel routes from compute_dtype = router_weights.dtype (a 16-bit dtype); the ref rounds norm
    through that dtype before the logits matmul to match the kernel's router precision.

    Returns:
        dict, always with "out" (fp32 [T, H] normalized activations); plus "out_residual" (fp32 [T, H]
        pre-norm sum) when residual is set; plus "expert_index" ([T, top_k] int32) and
        "expert_affinities" ([T, E] fp32) when router_weights is set.

    Pseudocode:
        x = float32(hidden_states.reshape(T, H))
        if residual is not None:
            x = float32(input_dtype(x + residual))        # kernel adds in input dtype; round to match
            out_residual = x                              # pre-norm sum (extra output)
        ss = sum(x * x, axis=H)                            # over full padded H (pad contributes 0)
        norm = x * rsqrt(ss / unpadded_hidden_size + eps) * gamma # mean divides by unpadded_hidden_size
        out = norm

        if router_weights is not None:
            norm_cd = float32(router_compute_dtype(norm))  # kernel routes from the 16-bit compute dtype
            w_nat   = unpermute(router_weights, swizzle_h_index(H))  # back to natural H for the matmul
            logits  = norm_cd @ w_nat (+ router_bias)
            idx     = argsort(-logits)[:, :top_k]          # expert_index
            tk      = logits[idx]
            aff     = softmax(tk) if is_softmax else sigmoid(tk)
            expert_affinities = scatter(aff -> idx) over [T, E]   # zero elsewhere
    """
    # Accept neuron_dtypes/np arrays or torch tensors; the input (16-bit) dtype drives the residual
    # round-trip, so capture it before upcasting to fp32 for the math.
    hidden_np = hidden_states.numpy() if hasattr(hidden_states, "numpy") else np.asarray(hidden_states)
    gamma_np = gamma.numpy() if hasattr(gamma, "numpy") else np.asarray(gamma)
    in_torch_dtype = _to_torch_dtype(hidden_np.dtype)

    H = hidden_np.shape[-1]
    if unpadded_hidden_size is None:
        unpadded_hidden_size = H
    T = int(np.prod(hidden_np.shape[:-1]))
    hidden = torch.from_numpy(dt.static_cast(hidden_np.reshape(T, H), np.float32))
    gamma_t = torch.from_numpy(dt.static_cast(gamma_np.reshape(1, H), np.float32))

    if residual is not None:
        residual_np = residual.numpy() if hasattr(residual, "numpy") else np.asarray(residual)
        residual_t = torch.from_numpy(dt.static_cast(residual_np.reshape(T, H), np.float32))
        # Kernel adds in the input dtype (dma_compute casts the fp32 sum back); round through it to match.
        hidden = (hidden + residual_t).to(in_torch_dtype).float()

    sum_squares = torch.sum(hidden * hidden, dim=-1, keepdim=True)
    inv_rms = torch.rsqrt(sum_squares / unpadded_hidden_size + eps)
    norm = hidden * inv_rms * gamma_t

    result = {"out": norm.numpy().astype(np.float32)}
    if residual is not None:
        result["out_residual"] = hidden.numpy().astype(np.float32)  # pre-norm sum (input + residual)
    if emit_norm_bf16:
        # Token-major bf16 RMSNorm output (natural H) = hidden * inv_rms * gamma, cast to bf16.
        result["norm_bf16"] = dt.static_cast(norm.numpy().astype(np.float32), dt.bfloat16)

    if router_weights is not None:
        wperm = router_weights.numpy() if hasattr(router_weights, "numpy") else np.asarray(router_weights)
        E = wperm.shape[1]
        # Kernel routes from compute_dtype = router_weights.dtype: round norm through it before the matmul.
        norm_cd = norm.to(_to_torch_dtype(wperm.dtype)).float()
        # Un-permute the pre-permuted weights back to natural H to matmul against natural norm.
        perm = swizzle_h_index(H)
        w_nat = np.empty_like(dt.static_cast(wperm, np.float32))
        w_nat[perm] = dt.static_cast(wperm, np.float32)
        logits = norm_cd @ torch.from_numpy(w_nat)
        if router_bias is not None:
            wb = router_bias.numpy() if hasattr(router_bias, "numpy") else np.asarray(router_bias)
            logits = logits + torch.from_numpy(dt.static_cast(wb, np.float32).reshape(1, E))
        if router_act_fn == RouterActFnType.NOAUX_TC:
            # Group-limited noaux_tc: sigmoid(logits) scores, +bias for selection only,
            # per-group top-2 gating, keep top-`topk_group` groups, final top-`top_k` experts by the
            # biased score, then L1-normalize the PRE-bias selected scores and scale.
            experts_per_group = E // n_group
            scores = torch.sigmoid(logits)  # PRE-bias affinity source
            scores_for_choice = scores + torch.from_numpy(dt.static_cast(wb, np.float32).reshape(1, E))
            grp = scores_for_choice.reshape(T, n_group, experts_per_group)
            group_score = grp.topk(2, dim=2).values.sum(dim=2)  # [T, n_group]
            kept_groups = group_score.topk(topk_group, dim=1).indices  # [T, topk_group]
            group_mask = torch.zeros((T, n_group), dtype=torch.float32)
            group_mask.scatter_(1, kept_groups, 1.0)
            expert_mask = group_mask.unsqueeze(-1).expand(T, n_group, experts_per_group).reshape(T, E)
            masked = scores_for_choice.masked_fill(expert_mask == 0, float("-inf"))
            expert_index = masked.topk(top_k, dim=1).indices  # [T, top_k]
            sel = torch.zeros((T, E), dtype=torch.float32)
            sel.scatter_(1, expert_index, torch.gather(scores, 1, expert_index))  # PRE-bias scores
            denom = sel.sum(dim=1, keepdim=True) + 1e-20
            affinities = sel / denom * routed_scaling_factor
            result["expert_affinities"] = affinities.numpy().astype(np.float32)  # dense only, no expert_index
        else:
            # argsort descending; take the top-K expert ids per token.
            expert_index = torch.argsort(-logits, dim=1)[:, :top_k].to(torch.int32)
            topk_logits = torch.gather(logits, 1, expert_index.long())
            is_softmax = router_act_fn != RouterActFnType.SIGMOID
            topk_aff = torch.softmax(topk_logits, dim=1) if is_softmax else torch.sigmoid(topk_logits)
            affinities = torch.zeros((T, E), dtype=torch.float32)
            affinities.scatter_(1, expert_index.long(), topk_aff)
            result["expert_index"] = expert_index.numpy().astype(np.int32)
            result["expert_affinities"] = affinities.numpy().astype(np.float32)

    return result


def reference_mx_dequant(norm_fp32: np.ndarray, round_dtype=dt.bfloat16) -> np.ndarray:
    """Quantize the normalized activations with the kernel's MX block grouping, then dequant.

    The kernel's MX block covers 32 CONTIGUOUS hidden values for a single token (8 swizzle
    partitions x 4 q-lanes map to H indices h512*512 + 32*blk + [0,32)). This produces a
    "quantized golden": comparing the kernel's dequantized output against THIS (rather than the
    raw fp32 RMSNorm) isolates layout correctness from the unavoidable e4m3 quantization loss, so
    a tight tolerance (well under 5%) is meaningful.

    Args:
        norm_fp32 (np.ndarray): [T, H] fp32 normalized activations.
        round_dtype: 16-bit dtype the kernel quantizes from (the swizzle/quant-feed compute_dtype).
            float16 when a router is fused (compute_dtype = router_weights.dtype); bfloat16 otherwise.

    Returns:
        np.ndarray: [T, H] fp32 dequantized (MX e4m3 round-tripped) activations.
    """
    T, H = norm_fp32.shape
    # Round the input through the kernel's compute dtype first (the kernel quantizes from it).
    x = dt.static_cast(dt.static_cast(norm_fp32, round_dtype), np.float32)
    n_blocks = H // 32
    blocks = x.reshape(T, n_blocks, 32)
    # Per 32-wide block: scale exponent = max block exponent - e4m3 max_exp (7).
    exp_field = (blocks.astype(np.float32).view(np.uint32) >> 23) & 0xFF
    block_max_exp = exp_field.max(axis=2, keepdims=True)  # [T, n_blocks, 1]
    scale_uint8 = np.clip(block_max_exp - _FP8_E4M3_MAX_EXP, 0, 255)
    factor = np.power(2.0, scale_uint8.astype(np.float64) - 127.0)
    # Quantize to e4m3 and dequant: round each value to fp8 then scale back.
    q = dt.static_cast(
        np.clip(blocks / np.where(factor == 0, 1.0, factor), -448.0, 448.0).astype(np.float32), dt.float8_e4m3fn
    )
    deq = dt.static_cast(q, np.float32) * factor
    return deq.reshape(T, H).astype(np.float32)


def decode_packed_output(packed_fp8, T: int, H: int, pack_scales: bool = True) -> np.ndarray:
    """Decode the kernel's packed [T, H + scale_region] fp8 output back to fp32 activations.

    Applies the per-MX-block scale to each 4-wide group of fp8 values:
        value = fp8_code * 2^(scale_exp - max_exp - 127 + 127) ... (per quantize_to_mx convention)
    i.e. dequant factor = 2^(scale_uint8 - 0) adjusted by the e4m3 max_exp baked into the scale.

    Args:
        packed_fp8 (np.ndarray): kernel output reinterpreted as fp8 [T, H + scale_region].
        T, H (int): token / hidden dims.
        pack_scales (bool): whether scales are folded.

    Returns:
        np.ndarray: [T, H] fp32 dequantized activations.
    """
    num_h512 = H // _K_BLOCK
    n_packed = (num_h512 + _SCALES_PER_BLOCK - 1) // _SCALES_PER_BLOCK if pack_scales else num_h512
    scale_region = n_packed * _H0

    quant = dt.static_cast(packed_fp8[:, :H], np.float32)  # [T, H] fp8 -> fp32 codes
    scales = packed_fp8[:, H : H + scale_region].view(np.uint8)  # [T, scale_region] uint8 exponents

    # Dequant. h = h512*512 + 4*p + q (swizzle identity); one MX scale covers q_height=8 partitions
    # x q_width=4 lanes, so block index along the 128-partition axis is blk = p // 8. quantize_mx
    # writes scales in HW quadrant layout (4 valid per 32-quadrant) -> column (blk//4)*32 + (blk%4).
    # When folded, tile h512 shares output block (h512//4) and is shifted by (h512%4)*q_width within
    # each quadrant. Dequant factor = 2^(scale - 127).
    _Q_HEIGHT = 8
    _QUADRANT = 32
    out = np.empty((T, H), dtype=np.float32)
    for h512 in range(num_h512):
        if pack_scales:
            base = (h512 // _SCALES_PER_BLOCK) * _H0
            shift = (h512 % _SCALES_PER_BLOCK) * _Q_WIDTH
        else:
            base = h512 * _H0
            shift = 0
        for p in range(_H0):
            blk = p // _Q_HEIGHT
            col = base + (blk // 4) * _QUADRANT + shift + (blk % 4)
            factor = np.power(2.0, scales[:, col].astype(np.float64) - 127.0).astype(np.float32)
            for q in range(_Q_WIDTH):
                h = h512 * _K_BLOCK + _Q_WIDTH * p + q
                out[:, h] = quant[:, h] * factor
    return out
