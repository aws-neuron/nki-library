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

"""
Pytorch reference for attention_cte kernel

"""

import torch

from ..utils.kernel_assert import assert_shape, kernel_assert


def attention_cte_torch_ref(
    q: torch.tensor,
    k: torch.tensor,
    v: torch.tensor,
    scale: float = 1.0,
    causal_mask: bool = True,
    k_prior=None,
    v_prior=None,
    prior_used_len=None,
    sink=None,
    sliding_window=0,
    tp_q=True,
    tp_k=False,
    tp_out=False,
    cache_softmax=False,
    softmax_dtype=torch.float32,
    cp_offset: torch.tensor = None,
    global_cp_deg: int = None,
    cp_strided_q_slicing: bool = False,
):
    """PyTorch reference implementation for attention_cte NKI kernel.

    This function provides a CPU-based reference implementation with identical
    interface to the NKI kernel for validation and testing purposes.

    Summary:
        Computes multi-head attention with support for causal masking, sliding window,
        prefix caching, context parallelism, and GQA. All computations are performed
        in float32 for CPU compatibility.

    Args:
        q (torch.tensor): Query tensor
        k (torch.tensor): Key tensor
        v (torch.tensor): Value tensor
        scale (float, optional): Scaling factor for attention scores. Default: 1.0
        causal_mask (bool, optional): Whether to apply causal mask. Default: True
        k_prior (torch.tensor, optional): Prior key tensor for prefix caching. Default: None
        v_prior (torch.tensor, optional): Prior value tensor for prefix caching. Default: None
        prior_used_len (torch.tensor, optional): Length of prior to use. Default: None
        sink (torch.tensor, optional): Sink token tensor. Default: None
        sliding_window (int, optional): Sliding window size. Default: 0
        tp_q (bool, optional): Query transpose flag. Default: True
        tp_k (bool, optional): Key transpose flag. Default: False
        tp_out (bool, optional): Output transpose flag. Default: False
        cache_softmax (bool, optional): Whether to cache softmax statistics. Default: False
        softmax_dtype (torch.dtype, optional): Data type for softmax outputs. Default: torch.float32
        cp_offset (torch.tensor, optional): Context parallel offset. Default: None
        global_cp_deg (int, optional): Global context parallel degree. Default: None
        cp_strided_q_slicing (bool, optional): Whether Q is strided. Default: False

    Returns:
        torch.tensor: Attention output tensor
        If cache_softmax is True, returns tuple of (output, neg_max, recip)

    Notes:
        - All inputs are converted to float32 for CPU compatibility
        - Supports GQA by replicating K/V tensors
        - Implements flash attention statistics when cache_softmax=True
    """
    # Process shapes and configs
    is_prefix_caching = k_prior is not None

    # Convert to float32 because CPU doesn't support half precision
    q = q.to(torch.float32)
    k = k.to(torch.float32)
    v = v.to(torch.float32)

    # Transpose inputs as per flag
    if not tp_q:
        q = q.transpose(1, 2)  # [bs, seqlen_q, d]
    if tp_k:
        k = k.transpose(1, 2)  # [bs, d, seqlen_k]
        if is_prefix_caching:
            k_prior = k_prior.transpose(1, 2)

    bs = q.shape[0]
    bs_kv, d = v.shape[0], v.shape[2]
    seqlen_q, seqlen_k = q.shape[1], k.shape[2]
    kernel_assert(d > 0, f"d must be postive, got {d}")

    # concatenate K and V with prior if prefix caching
    if is_prefix_caching:
        # Convert to float32 because CPU doesn't support half precision
        k_prior = k_prior.to(torch.float32)
        v_prior = v_prior.to(torch.float32)

        prior_used_len = prior_used_len.item()
        k_prior = k_prior[:, :, :prior_used_len]
        v_prior = v_prior[:, :prior_used_len, :]
        k = torch.cat([k_prior, k], dim=2)  # [bs, d, prior_used_len + seqlen_k]
        v = torch.cat([v_prior, v], dim=1)  # [bs, prior_used_len + seqlen_k, d]
    else:
        prior_used_len = 0

    # apply GQA replication if required
    if bs != bs_kv:
        kernel_assert(bs % bs_kv == 0, "Q batch size must be a multiple of KV batch size")
        k = torch.repeat_interleave(k, repeats=bs // bs_kv, dim=0)
        v = torch.repeat_interleave(v, repeats=bs // bs_kv, dim=0)

    # Build mask (causal, sliding window, prior used length for prefix caching)
    cp_offset_val = 0 if cp_offset is None else int(cp_offset.item())
    mask = q.new_full((seqlen_q, prior_used_len + seqlen_k), 0)

    # Define mask in terms of q/k position tensors
    if cp_offset is not None and cp_strided_q_slicing:
        q_positions = cp_offset_val + torch.arange(seqlen_q, dtype=torch.int32) * global_cp_deg
    else:
        q_positions = cp_offset_val + torch.arange(seqlen_q, dtype=torch.int32)
    kv_positions = torch.arange(prior_used_len + seqlen_k, dtype=torch.int32) - prior_used_len
    pos_diff = kv_positions[None, :] - q_positions[:, None]

    # Some torch versions require torch.where args to be tensors
    minus_inf_t = q.new_full((1,), -float("inf"))
    zero_t = q.new_full((1,), 0.0)
    # Generate mask
    if causal_mask:
        mask += torch.where(pos_diff > 0, minus_inf_t, zero_t)
    if sliding_window > 0:
        mask += torch.where(pos_diff <= -sliding_window, minus_inf_t, zero_t)

    # Compute QK, apply mask
    qk = q @ k  # [bs, seqlen_q, prior_used_len + seqlen_k]
    qk *= scale
    qk += mask[None, :, :]

    # Concat sink
    if sink is not None:
        assert_shape(sink, (bs, 1), "sink")
        sink = sink.reshape(bs, 1, 1).expand(-1, seqlen_q, -1)  # [bs, seqlen_q, 1]
        qk = torch.cat([qk, sink], dim=-1)  # [bs, seqlen_q, seqlen_k+1]

    # Softmax + PV matmul
    if cache_softmax:
        # in this case we need to return intermediate tensors
        tile_size = 128
        kernel_assert(
            seqlen_q % tile_size == 0,
            f"For cache softmax, kernel currently expects seqlen_q multiple of {tile_size}, got {seqlen_q=}",
        )
        # Compute softmax with caching of intermediate statistics
        max_value = torch.max(qk, dim=-1, keepdim=True).values  # [bs, seqlen_q, 1]
        qk_shifted = qk - max_value
        exp_values = torch.exp(qk_shifted)
        sum_exp = torch.sum(exp_values, dim=-1, keepdim=True)  # [bs, seqlen_q, 1]

        w = exp_values / sum_exp  # [bs, seqlen_q, seqlen_k+1] if sink else [bs, seqlen_q, seqlen_k]

        # Cache statistics for backward pass
        neg_max = -max_value  # [bs, seqlen_q, 1]
        recip = torch.reciprocal(sum_exp)  # [bs, seqlen_q, 1]

        if sink is not None:
            w = w[..., :-1]  # [bs, seqlen_q, seqlen_k]

        # Compute out, transpose if needed
        out = w @ v  # [bs, seqlen_q, d]
        out = out.transpose(1, 2) if tp_out else out

        # Reshape neg_max and recip to match the expected output format with tile_size=128
        # The output format is [bs, 128, seq_grps] where seq_grps = seqlen_q // 128)
        seq_grps = seqlen_q // tile_size
        neg_max = neg_max.reshape(bs, seq_grps, tile_size).transpose(1, 2)  # [bs, tile_size, seq_grps]
        recip = recip.reshape(bs, seq_grps, tile_size).transpose(1, 2)  # [bs, tile_size, seq_grps]

        # Cast to softmax_dtype
        neg_max = neg_max.to(softmax_dtype)
        recip = recip.to(softmax_dtype)

        return out, neg_max, recip
    else:
        w = torch.softmax(qk, dim=-1)
        if sink is not None:
            w = w[..., :-1]  # [bs, seqlen_q, seqlen_k]

        # Compute out, transpose if needed
        out = w @ v  # [bs, seqlen_q, d]
        return out.transpose(1, 2) if tp_out else out
