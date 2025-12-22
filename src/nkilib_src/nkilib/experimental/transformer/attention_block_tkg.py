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
Attention Block TKG Kernel

This kernel implements the attention block for token generation (TKG), fusing all stages in SBUF to avoid HBM round-trips.

It performs:
        +-------------------+   +-------------------+   +-------------------+   +-------------------+
        | Input X (HBM)     |-->|   RMSNorm X       |-->| QKV Projection    |-->|  Split QKV → Q,K  |-->
        | [B, S_tkg, H]     |   | (optional)        |   |                   |   |   (transpose)     |
        +-------------------+   +-------------------+   +-------------------+   +-------------------+

        +-------------------+   +-------------------+   +-------------------+   +-----------------------+
    --> | RMSNorm Q/K       |-->|   RoPE Embedding  |-->| RMSNorm Q/K       |-->| Attention TKG         |-->
        | (optional)        |   | (optional)        |   | (optional)        |   | (softmax(Q·Kᵀ/√d) @ V)|
        +-------------------+   +-------------------+   +-------------------+   +-----------------------+

        +-------------------+   +-------------------+
    --> | KV-Cache Update   |-->| Output Projection |
        | (optional)        |   | (optional)        |
        +-------------------+   +-------------------+

Features:
- Supports grouped-query attention (GQA) with a single key/value head
- LNC-2 sharding support
- Operates with or without output projection
- Optimized for small batch_size * sequence length typical in decoding
"""

from typing import Any, Dict, Optional, Tuple

import nki
import nki.isa as nisa
import nki.language as nl
from nki.isa.constants import oob_mode

from ...core.attention.attention_tkg import AttnTKGConfig, attention_tkg
from ...core.embeddings.rope import RoPE_sbuf
from ...core.output_projection.output_projection_tkg import output_projection_tkg
from ...core.qkv.qkv import qkv_tkg
from ...core.utils.allocator import SbufManager, create_auto_alloc_manager
from ...core.utils.common_types import NormType, QuantizationType
from ...core.utils.kernel_assert import kernel_assert
from ...core.utils.kernel_helpers import get_verified_program_sharding_info
from ...core.utils.logging import Logger

# ==================== Main Kernel Function ====================


# TODO(NKI-699): Refactor API to use configuration dataclasses for better clarity
def attention_block_tkg(
    # -- input
    X: nl.ndarray,
    X_hidden_dim_actual: Optional[int],
    # -- rmsnorm X
    rmsnorm_X_enabled: bool,
    rmsnorm_X_eps: Optional[float],
    rmsnorm_X_gamma: Optional[nl.ndarray],
    # -- qkv projections
    W_qkv: nl.ndarray,
    bias_qkv: Optional[nl.ndarray],
    quantization_type_qkv: QuantizationType,
    weight_dequant_scale_qkv: Optional[nl.ndarray],
    input_dequant_scale_qkv: Optional[nl.ndarray],
    # -- Q/K processing: pre-RoPE RMSNorm
    rmsnorm_QK_pre_rope_enabled: bool,
    rmsnorm_QK_pre_rope_eps: float,
    # -- Q/K processing: RoPE
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    # -- Q/K processing: post-RoPE RMSNorm
    rmsnorm_QK_post_rope_enabled: bool,
    rmsnorm_QK_post_rope_eps: float,
    rmsnorm_QK_post_rope_W_Q: Optional[nl.ndarray],
    rmsnorm_QK_post_rope_W_K: Optional[nl.ndarray],
    # -- attention
    K_cache_transposed: bool,
    active_blocks_table: Optional[nl.ndarray],
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    attention_mask: nl.ndarray,
    sink: Optional[nl.ndarray],
    # -- KV cache update
    update_cache: bool,
    kv_cache_update_idx: Optional[nl.ndarray],
    # -- output projection
    W_out: Optional[nl.ndarray],
    bias_out: Optional[nl.ndarray],
    quantization_type_out: QuantizationType,
    weight_dequant_scale_out: Optional[nl.ndarray],
    input_dequant_scale_out: Optional[nl.ndarray],
    transposed_out: bool,
    # -- output
    out_in_sb: bool,
    sbm: Optional[SbufManager] = None,
    skip_attention: bool = False,
):
    """
    Fused Attention Block for Token Generation (TKG).

    Performs end-to-end attention block computation optimized for autoregressive
    decoding with all stages fused in SBUF to avoid HBM round-trips. Intended for
    small batch sizes (B ≤ 16) and short sequence lengths (S_tkg ≤ 16) typical in
    token generation workloads.

    Dimensions:
        B: Batch size (≤ 16 recommended)
        S_tkg: Number of new tokens to generate (≤ 16 recommended)
        S_ctx: KV cache sequence length in current bucket
        S_max_ctx: Maximum KV cache capacity of current bucket
        H: Hidden dimension (must be multiple of 128)
        d_head: Head dimension (must be even)
        q_heads: Number of query heads
        kv_heads: 1 (GQA with single KV head)

    Args:
        X (nl.ndarray): Input hidden states
            Shape:
                [B, S_tkg, H]                                when in HBM
                [H0=pmax, BxS, H1] where H1=lnc x (H//lnc//pmax) when in SBUF

            When in SBUF, the layout is obtained by rearranging HBM data:
                HBM: (BxS, lnc, H0, H1//lnc) -> SBUF: (H0, BxS, (lnc, H1//lnc))
            This interleaves H1//lnc values from each lnc chunk along the H dimension,
            matching qkv_tkg() kernel's expected SBUF input format.
        X_hidden_dim_actual (Optional[int]): Actual hidden dim if X is padded

        rmsnorm_X_enabled (bool): Apply RMSNorm to X before QKV projection
        rmsnorm_X_eps (Optional[float]): RMSNorm epsilon (default 1e-3)
        rmsnorm_X_gamma (Optional[nl.ndarray]): [1, H] @ HBM, RMSNorm weights

        W_qkv (nl.ndarray): [H, d_head*(q_heads+2)] @ HBM, QKV projection weights
        bias_qkv (Optional[nl.ndarray]): [1, d_head*(q_heads+2)] @ HBM, QKV bias

        rmsnorm_QK_pre_rope_enabled (bool): Apply RMSNorm to Q/K before RoPE
        rmsnorm_QK_pre_rope_eps (float): Pre-RoPE RMSNorm epsilon
        cos (Optional[nl.ndarray]): [d_head//2, B, S_tkg] @ HBM, RoPE cosine embeddings (None = skip RoPE)
        sin (Optional[nl.ndarray]): [d_head//2, B, S_tkg] @ HBM, RoPE sine embeddings (None = skip RoPE)
        rope_contiguous_layout (bool): True for contiguous halves, False for interleaved
        rmsnorm_QK_post_rope_enabled (bool): Apply RMSNorm to Q/K after RoPE
        rmsnorm_QK_post_rope_eps (float): Post-RoPE RMSNorm epsilon
        rmsnorm_QK_post_rope_W_Q (Optional[nl.ndarray]): [1, d_head] @ HBM, Post-RoPE Q weights
        rmsnorm_QK_post_rope_W_K (Optional[nl.ndarray]): [1, d_head] @ HBM, Post-RoPE K weights

        K_cache_transposed (bool): K cache layout flag
        active_blocks_table (Optional[nl.ndarray]): [B, num_blocks] @ HBM, Block indices for block KV cache
        K_cache (nl.ndarray): Key cache @ HBM
        V_cache (nl.ndarray): Value cache @ HBM
        attention_mask (nl.ndarray): [S_ctx, B, q_heads, S_tkg] @ HBM, Attention mask
        sink (Optional[nl.ndarray]): [H, 1] @ HBM, Attention sink tokens

        update_cache (bool): Update KV cache with new tokens
        kv_cache_update_idx (Optional[nl.ndarray]): [B, 1], Cache write positions (uint32_max = skip)

        W_out (Optional[nl.ndarray]): [q_heads*d_head, H] @ HBM, Output projection weights
        bias_out (Optional[nl.ndarray]): [1, H] @ HBM, Output projection bias
        transposed_out (bool): Transpose output layout (requires W_out)
        out_in_sb (bool): Return output in SBUF instead of HBM
        sbm (Optional[SbufManager]): SBUF memory manager (otherwise auto-allocated)
        skip_attention (bool): Skip attention computation (for testing)

    Returns:
        out (nl.ndarray): Output tensor with shape depending on projection and output location:
            - Without projection (W_out=None):
                - out_in_sb=False: [B, q_heads, d_head, S_tkg] @ HBM
                - out_in_sb=True: [d_head, B*q_heads*S_tkg] @ SBUF
            - With projection (W_out provided):
                - transposed_out=False, out_in_sb=False: [B*S_tkg, H] @ HBM
                - transposed_out=False, out_in_sb=True: [B*S_tkg, H//lnc] @ SBUF
                - transposed_out=True, out_in_sb=False: [128, lnc, H//lnc//128, B*S_tkg] @ HBM
                - transposed_out=True, out_in_sb=True: [128, H//lnc//128, B*S_tkg] @ SBUF
        K_out (nl.ndarray):
            - If update_cache=True: Updated K cache (shape matches K_cache input)
            - If update_cache=False: New K tokens [d_head, B, S_tkg] @ HBM
        V_out (nl.ndarray):
            - If update_cache=True: Updated V cache (shape matches V_cache input)
            - If update_cache=False: New V tokens [B, 1, S_tkg, d_head] @ HBM

    Notes:
        - Requires NeuronCore v3+
        - d_head must be even
        - H must be multiple of pmax
        - Requires batch * sequence_tkg * q_heads <= pmax (=pmax)
        - Supports grouped-query attention (GQA) with single key/value head
        - LNC-2 sharding support for KV cache updates

    Pseudocode:
        # Stage 1: QKV Projection
        if rmsnorm_X_enabled:
            X_norm = rms_norm(X, rmsnorm_X_gamma, rmsnorm_X_eps)
        QKV = matmul(X_norm, W_qkv) + bias_qkv

        # Stage 2: Q/K Processing
        Q, K = split_and_transpose(QKV)
        if rmsnorm_QK_pre_rope_enabled:
            Q, K = rms_norm(Q), rms_norm(K)
        if cos is not None and sin is not None:
            Q, K = rope(Q, cos, sin), rope(K, cos, sin)
        if rmsnorm_QK_post_rope_enabled:
            Q = rms_norm(Q, rmsnorm_QK_post_rope_W_Q)
            K = rms_norm(K, rmsnorm_QK_post_rope_W_K)
        V = extract_V(QKV)

        # Stage 3: Attention
        Q_scaled = Q / sqrt(d_head)
        attn_out = attention_tkg(Q_scaled, K, V, K_cache, V_cache, attention_mask)

        # Stage 4: KV Cache Update
        if update_cache:
            update_kv_cache(K_cache, V_cache, K, V, kv_cache_update_idx)
            K_out, V_out = K_cache, V_cache  # Return updated caches
        else:
            K_out, V_out = K, V  # Return new tokens

        # Stage 5: Output Projection
        if W_out is not None:
            output = matmul(attn_out, W_out) + bias_out
        else:
            output = attn_out

        return output, K_out, V_out
    """

    # ========== Validation and Setup ==========
    config = _validate_and_extract_config(
        X,
        W_qkv,
        K_cache,
        V_cache,
        attention_mask,
        cos,
        sin,
        rmsnorm_X_gamma,
        K_cache_transposed,
        active_blocks_table,
        W_out,
    )

    B, S_tkg = config['B'], config['S_tkg']
    d_head, q_heads = config['d_head'], config['q_heads']
    S_ctx, S_max_ctx = config['S_ctx'], config['S_max_ctx']
    is_block_kv, blk_len = config['is_block_kv'], config['blk_len']
    cache_had_head_dim = config['cache_had_head_dim']
    do_out_proj = config['do_out_proj']
    K_cache, V_cache = config['K_cache'], config['V_cache']
    kv_heads = 1
    I = d_head * (q_heads + 2 * kv_heads)

    sbm = sbm if sbm is not None else create_auto_alloc_manager(logger=Logger("attn-block-tkg"))
    sbm.open_scope("attn-blk-tkg-scope")

    # ========== QKV Projection ==========
    # Input:  X [B, S_tkg, H] @ HBM
    # Output: QKV_tkg_sb [B*S_tkg, I] @ SBUF where I = d_head * (q_heads + 2)
    rmsnorm_X_eps = 1e-3 if rmsnorm_X_eps is None else rmsnorm_X_eps
    QKV_tkg_sb = qkv_tkg(
        hidden=X,
        qkv_w=W_qkv,
        norm_w=rmsnorm_X_gamma,
        d_head=d_head,
        num_q_heads=q_heads,
        num_kv_heads=kv_heads,
        eps=rmsnorm_X_eps,
        norm_type=NormType.RMS_NORM if rmsnorm_X_enabled else NormType.NO_NORM,
        quantization_type=quantization_type_qkv,
        qkv_w_scale=weight_dequant_scale_qkv,
        qkv_in_scale=input_dequant_scale_qkv,
        output_in_sbuf=True,
        qkv_bias=bias_qkv,
        hidden_actual=X_hidden_dim_actual,
        sbm=sbm,
    )

    # ========== Q/K Processing: Transpose + RMSNorm pre + RoPE + RMSNorm post ==========
    # Input:  QKV_tkg_sb [B*S_tkg, I] @ SBUF
    # Output: Q_tkg_sb [d_head, B*q_heads*S_tkg] @ SBUF, K_tkg_sb [d_head, B*S_tkg] @ SBUF
    Q_tkg_sb, K_tkg_sb = _QK_processing(
        QKV_tkg_sb,
        q_heads,
        B,
        rmsnorm_QK_pre_rope_enabled,
        rmsnorm_QK_pre_rope_eps,
        cos,
        sin,
        rope_contiguous_layout,
        rmsnorm_QK_post_rope_enabled,
        rmsnorm_QK_post_rope_eps,
        rmsnorm_QK_post_rope_W_Q,
        rmsnorm_QK_post_rope_W_K,
        sbm,
    )
    # Extract V: skip Q and K heads
    V_tkg_sb = nl.ndarray((B * S_tkg, d_head), dtype=QKV_tkg_sb.dtype, buffer=nl.sbuf, name="attention_blk_V_tkg_sb")
    nisa.tensor_copy(V_tkg_sb, QKV_tkg_sb[:, nl.ds(d_head * (q_heads + kv_heads), d_head)])

    # ========== Attention Computation ==========
    # Input:  Q_tkg_sb [d_head, B*q_heads*S_tkg] @ SBUF
    #         K_tkg_sb [d_head, B*S_tkg] @ SBUF
    #         V_tkg_hbm [B, 1, S_tkg, d_head] @ HBM
    # Output: attn_out [d_head, B*q_heads*S_tkg] @ SBUF or [B, q_heads, d_head, S_tkg] @ HBM

    V_tkg_hbm = nl.ndarray(
        (B, 1, S_tkg, d_head), dtype=QKV_tkg_sb.dtype, buffer=nl.shared_hbm, name="v_attention_hbm"
    )  # for attention_tkg()
    nisa.dma_copy(V_tkg_hbm.reshape((B * S_tkg, d_head)), V_tkg_sb)

    if skip_attention:
        attn_out = Q_tkg_sb
    else:
        # Scale Q by 1/sqrt(d_head)
        nisa.tensor_scalar(dst=Q_tkg_sb, data=Q_tkg_sb, op0=nl.multiply, operand0=(d_head ** (-0.5)))
        # Allocate attention output buffer
        allocate_attn_out_on_HBM = not do_out_proj and not out_in_sb
        if allocate_attn_out_on_HBM:
            attn_out = nl.ndarray(
                (B, q_heads, d_head, S_tkg), dtype=X.dtype, buffer=nl.shared_hbm, name="attn_v_active_hbm"
            )
        else:  # attn_out @ SBUF
            attn_out = sbm.alloc_stack((d_head, B * q_heads * S_tkg), dtype=X.dtype, buffer=nl.sbuf)

        # Prepare KV cache views for attention
        if is_block_kv:
            k_prior, v_prior = K_cache, V_cache
        else:
            k_shape = (B, 1, d_head, S_max_ctx) if K_cache_transposed else (B, 1, S_max_ctx, d_head)
            k_prior = K_cache.reshape(k_shape)
            v_prior = V_cache.reshape((B, 1, S_max_ctx, d_head))

        attn_cfg = AttnTKGConfig(
            bs=B,
            q_head=q_heads,
            s_active=S_tkg,
            curr_sprior=S_ctx,
            full_sprior=S_max_ctx,
            d_head=d_head,
            block_len=blk_len if is_block_kv else 0,
            tp_k_prior=not K_cache_transposed,
            strided_mm1=not is_block_kv,
            use_pos_id=False,
            fuse_rope=False,
            use_gpsimd_sb2sb=True,
            qk_in_sb=True,
            k_out_in_sb=False,
            out_in_sb=do_out_proj or out_in_sb,
        )

        attention_tkg(
            q=Q_tkg_sb,
            k_active=K_tkg_sb,
            v_active=V_tkg_hbm,  # Attention_tkg() wants V @ HBM
            k_prior=k_prior,
            v_prior=v_prior,
            mask=attention_mask,
            out=attn_out,  # OUT
            cfg=attn_cfg,
            sbm=sbm,
            sink=sink,
            active_blocks_table=active_blocks_table,
        )

    # ========== KV Cache Update ==========
    # Input:  K_tkg_sb [d_head, B*S_tkg] @ SBUF
    #         V_tkg_hbm [B, 1, S_tkg, d_head] @ HBM
    # Output: K_hbm_out, V_hbm_out (updated caches or new tokens) @ HBM
    if update_cache:
        _kv_cache_update(
            K_cache=K_cache,
            V_cache=V_cache,
            K_tkg=K_tkg_sb,
            V_tkg=V_tkg_sb,
            kv_cache_update_idx=kv_cache_update_idx,
            B=B,
            d_head=d_head,
            S_tkg=S_tkg,
            S_max_ctx=S_max_ctx,
            K_cache_transposed=K_cache_transposed,
            is_block_kv=is_block_kv,
        )
        K_cache, V_cache = __internal_unsqueeze_head_dim(K_cache, V_cache, cache_had_head_dim, is_block_kv)
    else:  # No cache update: return new K/V tokens
        K_tkg_hbm = nl.ndarray((d_head, B, S_tkg), dtype=K_tkg_sb.dtype, buffer=nl.shared_hbm, name="K_hbm")
        nisa.dma_copy(K_tkg_hbm.reshape(K_tkg_sb.shape), K_tkg_sb)

    # ========== Output Projection (Optional) ==========
    # Input:  attn_out [d_head, B, q_heads, S_tkg] @ SBUF/HBM
    # Output: kernel_output layout depends on transposed_out and out_in_sb
    if do_out_proj:
        kernel_output = output_projection_tkg(
            attention=attn_out.reshape((d_head, B, q_heads, S_tkg)),
            weight=W_out,
            bias=bias_out,
            quantization_type=quantization_type_out,
            weight_scale=weight_dequant_scale_out,
            input_scale=input_dequant_scale_out,
            TRANSPOSE_OUT=transposed_out,
            OUT_IN_SB=out_in_sb,
        )
    else:
        kernel_assert(not transposed_out, "transposed_out requires output projection (W_out must be provided)")
        kernel_output = attn_out

    # Copy output to HBM if caller expects it on HBM but it's on SBUF. This is only used for debug when skipping both attention and output-projection.
    if out_in_sb == False and kernel_output.buffer == nl.sbuf:
        kernel_output_hbm = nl.ndarray(
            kernel_output.shape, kernel_output.dtype, nl.shared_hbm, name="kernel_output_hbm"
        )
        nisa.dma_copy(kernel_output_hbm, kernel_output)
        kernel_output = kernel_output_hbm

    # ========== Cleanup and Return ==========
    sbm.close_scope()
    if update_cache:
        return kernel_output, K_cache, V_cache
    else:
        return kernel_output, K_tkg_hbm, V_tkg_hbm


############### Internal ###############


def _validate_and_extract_config(
    X: nl.ndarray,
    W_qkv: nl.ndarray,
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    attention_mask: nl.ndarray,
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rmsnorm_X_gamma: Optional[nl.ndarray],
    K_cache_transposed: bool,
    active_blocks_table: Optional[nl.ndarray],
    W_out: Optional[nl.ndarray],
) -> Dict[str, Any]:
    """
    Validate inputs and extract configuration parameters for attention block.

    Args:
        X (nl.ndarray): Input hidden states
        W_qkv (nl.ndarray): QKV projection weights
        K_cache (nl.ndarray): Key cache
        V_cache (nl.ndarray): Value cache
        attention_mask (nl.ndarray): Attention mask
        cos (Optional[nl.ndarray]): RoPE cosine embeddings
        sin (Optional[nl.ndarray]): RoPE sine embeddings
        rmsnorm_X_gamma (Optional[nl.ndarray]): RMSNorm weights
        K_cache_transposed (bool): K cache layout flag
        active_blocks_table (Optional[nl.ndarray]): Block indices for block KV cache
        W_out (Optional[nl.ndarray]): Output projection weights

    Returns:
        Dict[str, Any]: Configuration dictionary with keys: B, S_tkg, H, d_head, half_d,
            q_heads, S_ctx, S_max_ctx, is_block_kv, blk_len, cache_had_head_dim,
            do_out_proj, K_cache, V_cache

    Notes:
        - Validates tensor shapes and dimensions
        - Extracts batch size, sequence lengths, and head dimensions
        - Handles both block and flat KV cache layouts
    """

    kernel_assert(
        nisa.get_nc_version() >= nisa.nc_version.gen3,
        f"Kernel requires nc-version >= gen3, got {nisa.get_nc_version()}",
    )

    _, B, _, _ = attention_mask.shape
    if X.buffer == nl.sbuf:
        # expecting X.shape = (pmax, B*S, H // pmax) @ SBUF
        kernel_assert(len(X.shape) == 3, "SBUF input X must have 3 dimensions")
        kernel_assert(X.shape[0] == nl.tile_size.pmax, f"SBUF input X dim0 must be {nl.tile_size.pmax}")
        kernel_assert(X.shape[1] % B == 0, f"SBUF input X dim1 must be divisible by B={B}")
        H = X.shape[2] * nl.tile_size.pmax
        S_tkg = X.shape[1] // B
    else:
        # expecting X.shape = (B,S,H) @ HBM
        kernel_assert(X.buffer == nl.hbm, "Input X must be in HBM or SBUF")
        B, S_tkg, H = X.shape

    d_head = V_cache.shape[-1]
    I = W_qkv.shape[1]
    kv_heads = 1

    # This limitation can be relaxed
    kernel_assert(B * S_tkg <= nl.tile_size.pmax, f"B * S_tkg must be <= {nl.tile_size.pmax}, got {B * S_tkg}")
    kernel_assert(d_head % 2 == 0, f"d_head must be even, got {d_head}")
    kernel_assert(
        d_head > 0 and I % d_head == 0,
        f"QKV weights must be packed as (q_heads + 2) * d_head, got I={I}, d_head={d_head}",
    )

    q_heads = I // d_head - 2 * kv_heads
    half_d = d_head // 2

    # Process KV cache
    is_block_kv = active_blocks_table is not None
    K_cache, V_cache, cache_had_head_dim = __internal_squeeze_head_dim(K_cache, V_cache, is_block_kv)

    if is_block_kv:
        blk_len = V_cache.shape[1]
        S_ctx = S_max_ctx = active_blocks_table.shape[1] * blk_len
        kernel_assert(
            V_cache.shape == K_cache.shape,
            f"Block KV cache shape mismatch: K={K_cache.shape} vs V={V_cache.shape}",
        )
    else:
        S_ctx = attention_mask.shape[0]
        S_max_ctx = V_cache.shape[1]
        blk_len = 0
        kernel_assert(
            V_cache.shape[0] == B,
            f"V_cache batch mismatch: expected {B}, got {V_cache.shape[0]}",
        )
        expected_K_shape = (B, d_head, S_max_ctx) if K_cache_transposed else (B, S_max_ctx, d_head)
        kernel_assert(
            tuple(K_cache.shape) == expected_K_shape,
            f"K_cache shape mismatch: expected {expected_K_shape}, got {K_cache.shape}",
        )

    # Validate attention mask
    expected_mask_shape = (S_ctx, B, q_heads, S_tkg)
    kernel_assert(
        tuple(attention_mask.shape) == expected_mask_shape,
        f"attention_mask shape mismatch: expected {expected_mask_shape}, got {attention_mask.shape}",
    )

    # Validate RMSNorm weights
    if rmsnorm_X_gamma is not None:
        kernel_assert(
            tuple(rmsnorm_X_gamma.shape) == (1, H),
            f"rmsnorm_X_gamma must be (1, {H}), got {rmsnorm_X_gamma.shape}",
        )

    # Validate RoPE embeddings
    if cos is not None and sin is not None:
        kernel_assert(
            tuple(cos.shape) == (half_d, B, S_tkg),
            f"cos shape mismatch: expected ({half_d}, {B}, {S_tkg}), got {cos.shape}",
        )
        kernel_assert(
            tuple(sin.shape) == (half_d, B, S_tkg),
            f"sin shape mismatch: expected ({half_d}, {B}, {S_tkg}), got {sin.shape}",
        )

    return {
        'B': B,
        'S_tkg': S_tkg,
        'H': H,
        'd_head': d_head,
        'half_d': half_d,
        'q_heads': q_heads,
        'S_ctx': S_ctx,
        'S_max_ctx': S_max_ctx,
        'is_block_kv': is_block_kv,
        'blk_len': blk_len,
        'cache_had_head_dim': cache_had_head_dim,
        'do_out_proj': W_out is not None,
        'K_cache': K_cache,
        'V_cache': V_cache,
    }


def __internal_squeeze_head_dim(
    K_cache: nl.ndarray, V_cache: nl.ndarray, is_block_kv: bool
) -> Tuple[nl.ndarray, nl.ndarray, bool]:
    """
    Remove head dimension from 4D cache tensors.

    Args:
        K_cache (nl.ndarray): Key cache (3D or 4D)
        V_cache (nl.ndarray): Value cache (3D or 4D)
        is_block_kv (bool): Block KV cache flag

    Returns:
        Tuple[nl.ndarray, nl.ndarray, bool]: (K_squeezed, V_squeezed, had_head_dim)

    Notes:
        - If is_block_kv is False, removes dim-1 (N) from BNSd or BNdS
        - If is_block_kv is True, removes dim-2 from (blocks, block_len, heads, d_head)
        - Returns original tensors if already 3D
    """
    if len(K_cache.shape) != 4:
        kernel_assert(len(K_cache.shape) == len(V_cache.shape) == 3, "Expecting KV cache to have 3 or 4 dims")
        return K_cache, V_cache, False

    head_dim = 2 if is_block_kv else 1
    kernel_assert(len(K_cache.shape) == len(V_cache.shape) == 4, "Expecting KV cache to have 3 or 4 dims")
    kernel_assert(K_cache.shape[head_dim] == V_cache.shape[head_dim] == 1, "Expecting single head for KV")
    K_shape = list(K_cache.shape[:head_dim]) + list(K_cache.shape[head_dim + 1 :])
    V_shape = list(V_cache.shape[:head_dim]) + list(V_cache.shape[head_dim + 1 :])
    return K_cache.reshape(tuple(K_shape)), V_cache.reshape(tuple(V_shape)), True


def __internal_unsqueeze_head_dim(
    K_cache: nl.ndarray, V_cache: nl.ndarray, cache_had_head_dim: bool, is_block_kv: bool
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Add back head dimension if cache originally had one.

    Args:
        K_cache (nl.ndarray): Key cache (3D)
        V_cache (nl.ndarray): Value cache (3D)
        cache_had_head_dim (bool): Whether cache originally had head dimension
        is_block_kv (bool): Block KV cache flag

    Returns:
        Tuple[nl.ndarray, nl.ndarray]: (K_cache, V_cache) with head dimension restored

    Notes:
        - Inverse operation of __internal_squeeze_head_dim
        - Returns original tensors if cache_had_head_dim is False
    """
    if not cache_had_head_dim:
        return K_cache, V_cache

    head_dim = 2 if is_block_kv else 1
    K_shape = list(K_cache.shape[:head_dim]) + [1] + list(K_cache.shape[head_dim:])
    V_shape = list(V_cache.shape[:head_dim]) + [1] + list(V_cache.shape[head_dim:])
    return K_cache.reshape(tuple(K_shape)), V_cache.reshape(tuple(V_shape))


def _to_sbuf(buf: nl.ndarray, sbm: SbufManager) -> nl.ndarray:
    """
    Ensure buffer is in SBUF; copy from HBM if needed.

    Args:
        buf (nl.ndarray): Input buffer (HBM or SBUF)
        sbm (SbufManager): SBUF memory manager

    Returns:
        nl.ndarray: Buffer in SBUF

    Notes:
        - Returns original buffer if already in SBUF
        - Allocates and copies if buffer is in HBM
    """
    if buf.buffer == nl.sbuf:
        return buf
    else:
        sb = sbm.alloc_stack(buf.shape, dtype=buf.dtype, buffer=nl.sbuf)
        nisa.dma_copy(sb, buf)
        return sb


def _process_head_group(
    QKV: nl.ndarray,
    qkv_offset: int,
    n_heads: int,
    d: int,
    B: int,
    S: int,
    rmsnorm_pre_enabled: bool,
    rmsnorm_pre_eps: float,
    enable_rope: bool,
    sb_cos: Optional[nl.ndarray],
    sb_sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    rmsnorm_post_enabled: bool,
    rmsnorm_post_eps: float,
    rmsnorm_post_W: Optional[nl.ndarray],
    sbm: SbufManager,
) -> nl.ndarray:
    """
    Process Q or K: extract heads, transpose to [d, B*n_heads*S], apply optional RMSNorm pre/post and RoPE.
    For Q: n_heads=q_heads, qkv_offset=0
    For K: n_heads=1, qkv_offset=d*q_heads
    """
    # Transpose heads from [B*S, n_heads*d] to [d, B*n_heads*S]
    out = sbm.alloc_stack(shape=(d, B * n_heads * S), dtype=QKV.dtype, buffer=nl.sbuf)
    for head_idx in range(n_heads):
        psum = nl.ndarray((d, B * S), dtype=QKV.dtype, buffer=nl.psum)
        nisa.nc_transpose(psum, QKV[:, nl.ds(qkv_offset + head_idx * d, d)])
        nisa.tensor_copy(out.reshape((d, B, n_heads, S))[:, :, head_idx, :], psum.reshape((d, B, S)))

    # Pre-RoPE RMSNorm
    if rmsnorm_pre_enabled:
        _rms_norm_inplace(out, rmsnorm_pre_eps, w=None, sbm=sbm)

    # RoPE
    if enable_rope:
        out_4d = out.reshape((d, B, n_heads, S))
        out_rope = sbm.alloc_stack(out_4d.shape, dtype=out.dtype, buffer=nl.sbuf)
        RoPE_sbuf(out_4d, sb_cos, sb_sin, out_rope, convert_from_interleaved=not rope_contiguous_layout)
        out = out_rope.reshape((d, B * n_heads * S))

    # Post-RoPE RMSNorm
    if rmsnorm_post_enabled:
        _rms_norm_inplace(out, rmsnorm_post_eps, rmsnorm_post_W, sbm)

    return out


def _QK_processing(
    QKV: nl.ndarray,
    q_heads: int,
    B: int,
    rmsnorm_pre_enabled: bool,
    rmsnorm_pre_eps: float,
    cos: Optional[nl.ndarray],
    sin: Optional[nl.ndarray],
    rope_contiguous_layout: bool,
    rmsnorm_post_enabled: bool,
    rmsnorm_post_eps: float,
    rmsnorm_post_W_Q: Optional[nl.ndarray],
    rmsnorm_post_W_K: Optional[nl.ndarray],
    sbm: SbufManager,
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Unified Q/K processing: transpose, optional pre-RoPE RMSNorm, optional RoPE, optional post-RoPE RMSNorm.

    Args:
        QKV: [B*S, I] @ SBUF - concatenated Q/K/V projections where I = d*(q_heads+2)
        q_heads: number of query heads
        B: batch size
        rmsnorm_pre_enabled: Apply RMSNorm before RoPE
        rmsnorm_pre_eps: Pre-RoPE RMSNorm epsilon
        cos: RoPE cosine embeddings (None = skip RoPE)
        sin: RoPE sine embeddings (None = skip RoPE)
        rope_contiguous_layout: True for contiguous halves, False for interleaved
        rmsnorm_post_enabled: Apply RMSNorm after RoPE
        rmsnorm_post_eps: Post-RoPE RMSNorm epsilon
        rmsnorm_post_W_Q: Post-RoPE Q weights (optional)
        rmsnorm_post_W_K: Post-RoPE K weights (optional)
        sbm: SBUF memory manager

    Returns:
        Q: [d, B*q_heads*S] @ SBUF
        K: [d, B*S] @ SBUF
    """
    kernel_assert(QKV.buffer == nl.sbuf, "QKV must be in SBUF")
    kernel_assert(len(QKV.shape) == 2, "Expecting QKV.shape=(BxS, I)")

    BxS, I = QKV.shape
    S = BxS // B
    d = I // (q_heads + 2)

    enable_rope = cos is not None and sin is not None

    # Load RoPE embeddings to SBUF if needed
    sb_cos, sb_sin = None, None
    if enable_rope:
        kernel_assert(cos.shape == sin.shape, "cos and sin must match")
        sb_cos = _to_sbuf(cos, sbm)
        sb_sin = _to_sbuf(sin, sbm)

    Q = _process_head_group(
        QKV,
        qkv_offset=0,
        n_heads=q_heads,
        d=d,
        B=B,
        S=S,
        rmsnorm_pre_enabled=rmsnorm_pre_enabled,
        rmsnorm_pre_eps=rmsnorm_pre_eps,
        enable_rope=enable_rope,
        sb_cos=sb_cos,
        sb_sin=sb_sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_post_enabled=rmsnorm_post_enabled,
        rmsnorm_post_eps=rmsnorm_post_eps,
        rmsnorm_post_W=rmsnorm_post_W_Q,
        sbm=sbm,
    )
    K = _process_head_group(
        QKV,
        qkv_offset=d * q_heads,
        n_heads=1,
        d=d,
        B=B,
        S=S,
        rmsnorm_pre_enabled=rmsnorm_pre_enabled,
        rmsnorm_pre_eps=rmsnorm_pre_eps,
        enable_rope=enable_rope,
        sb_cos=sb_cos,
        sb_sin=sb_sin,
        rope_contiguous_layout=rope_contiguous_layout,
        rmsnorm_post_enabled=rmsnorm_post_enabled,
        rmsnorm_post_eps=rmsnorm_post_eps,
        rmsnorm_post_W=rmsnorm_post_W_K,
        sbm=sbm,
    )

    return Q, K


def _rms_norm_inplace(
    x: nl.ndarray, eps: float, w: Optional[nl.ndarray] = None, sbm: Optional[SbufManager] = None
) -> None:
    """
    RMS normalization in-place: x / sqrt(mean(x^2) + eps), optionally scaled by w.
    Computed in fp32, result written back to x in original dtype.

    Args:
        x: [d_head, BnS] @ SBUF - input tensor (d_head must be nl.tile_size.pmax), modified in-place
        eps: epsilon for numerical stability
        w: [d_head, 1] @ HBM - optional scale weights
        sbm: SBUF memory manager
    """
    d_head, BnS = x.shape
    kernel_assert(d_head == nl.tile_size.pmax, f"d_head must be {nl.tile_size.pmax}, got {d_head}")

    # Setup constants
    ones_sb = sbm.alloc_stack((d_head, d_head), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(ones_sb, 1.0)
    eps_sb = sbm.alloc_stack((d_head, 1), dtype=nl.float32, buffer=nl.sbuf)
    nisa.memset(eps_sb, eps)

    # Compute x^2 in fp32
    x_squared = sbm.alloc_stack((d_head, BnS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(x_squared, x, x, nl.multiply)

    # Compute sum(x^2) via matmul with all-ones matrix
    psum_sb = nl.ndarray((d_head, BnS), dtype=nl.float32, buffer=nl.psum)
    nisa.nc_matmul(psum_sb, stationary=ones_sb, moving=x_squared)

    # Compute rsqrt(mean(x^2) + eps)
    rsqrt_sb = sbm.alloc_stack((d_head, BnS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.activation(dst=rsqrt_sb, op=nl.rsqrt, data=psum_sb, bias=eps_sb, scale=1.0 / d_head)

    # Normalize: x * rsqrt
    out_sb = sbm.alloc_stack((d_head, BnS), dtype=nl.float32, buffer=nl.sbuf)
    nisa.tensor_tensor(out_sb, x, rsqrt_sb, nl.multiply)

    # Optional scaling by weights
    if w is not None:
        w_sb = sbm.alloc_stack((d_head, 1), dtype=nl.float32, buffer=nl.sbuf)
        nisa.dma_copy(w_sb, w.reshape((d_head, 1)))
        nisa.tensor_scalar(dst=out_sb, data=out_sb, op0=nl.multiply, operand0=w_sb)

    # Copy result back to x with original dtype
    nisa.tensor_copy(dst=x, src=out_sb)


############################# KV cache update logic #############################


def _kv_cache_update(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    B: int,
    d_head: int,
    S_tkg: int,
    S_max_ctx: int,
    K_cache_transposed: bool,
    is_block_kv: bool,
) -> Tuple[nl.ndarray, nl.ndarray]:
    """
    Update KV cache with new tokens for token generation.

    Args:
        K_cache: K cache @ HBM
            - Block KV: [num_blocks, block_len, d_head]
            - Flat transposed: [B, d_head, S_max_ctx]
            - Flat: [B, S_max_ctx, d_head]
        V_cache: V cache @ HBM
            - Block KV: [num_blocks, block_len, d_head]
            - Flat: [B, S_max_ctx, d_head]
        K_tkg: [d_head, B*S_tkg] @ SBUF
        V_tkg: [B, S_tkg, d_head] @ SBUF
        kv_cache_update_idx: [B] slot indices for cache writes
        B: batch size
        d_head: head dimension
        S_tkg: number of new tokens
        S_max_ctx: max cache sequence length
        K_cache_transposed: K cache layout flag
        is_block_kv: block KV cache flag

    Returns:
        Updated (K_cache, V_cache) - modified in-place
    """

    # TODO: oob_mode.skip not supported for flat cache. Using oob_mode.skip causes accuracy failures (root cause unknown).

    if is_block_kv:
        _update_block_cache(K_cache, V_cache, K_tkg, V_tkg, kv_cache_update_idx, S_tkg, B)
    elif not K_cache_transposed and S_tkg == 1 and B > 1:
        # one vector DMA of (B, S_tkg, d_head). Bug for S_tkg > 1.
        _update_flat_cache_batched(
            K_cache,
            V_cache,
            K_tkg,
            V_tkg,
            kv_cache_update_idx,
            S_tkg,
            S_max_ctx,
            B,
            d_head,
        )
    else:
        # B scalar DMA of (S_tkg, d_head)
        _update_flat_cache(
            K_cache,
            V_cache,
            K_tkg,
            V_tkg,
            K_cache_transposed,
            kv_cache_update_idx,
            S_tkg,
            S_max_ctx,
            B,
            d_head,
        )


def _update_flat_cache_batched(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    S_max_ctx: int,
    B: int,
    d_head: int,
) -> None:
    """
    Update flat (non-block) KV cache with new tokens using batched DMA operations.

    This optimized version writes all B batches in a single DMA operation using vector_offset
    for indirect addressing. Currently limited to S_tkg=1 due to access pattern bug.

    Args:
        K_cache: [B, S_max_ctx, d_head] K cache in HBM
        V_cache: [B, S_max_ctx, d_head] V cache in HBM
        K_tkg: [d_head, B*S_tkg] new K tokens in SBUF
        V_tkg: [B*S_tkg, d_head] new V tokens in SBUF
        kv_cache_update_idx: [B, 1] per-batch write positions
        S_tkg: number of new tokens per batch
        S_max_ctx: maximum cache sequence length
        B: batch size
        d_head: head dimension
    """
    # Validate sharding configuration
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")
    kernel_assert(S_tkg == 1, f"_update_flat_cache_batched() only supports S_tkg=1, got {S_tkg}")
    kernel_assert(B * S_tkg <= nl.tile_size.pmax, f"B * S_tkg must be <= {nl.tile_size.pmax}, got {B * S_tkg}")

    # Validate tensor shapes
    kernel_assert(
        K_cache.shape == (B, S_max_ctx, d_head),
        f"K_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {K_cache.shape}",
    )
    kernel_assert(
        V_cache.shape == (B, S_max_ctx, d_head),
        f"V_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {V_cache.shape}",
    )
    kernel_assert(
        K_tkg.shape == (d_head, B * S_tkg),
        f"K_tkg shape mismatch: expected {(d_head, B * S_tkg)}, got {K_tkg.shape}",
    )
    kernel_assert(
        V_tkg.shape == (B * S_tkg, d_head),
        f"V_tkg shape mismatch: expected {(B * S_tkg, d_head)}, got {V_tkg.shape}",
    )
    kernel_assert(
        kv_cache_update_idx.shape == (B, 1),
        f"kv_cache_update_idx shape mismatch: expected {(B, 1)}, got {kv_cache_update_idx.shape}",
    )

    # Compute absolute token indices (global flat cache indices):
    #   token_indices[b] = kv_cache_update_idx[b] + b * S_max_ctx
    token_indices = nl.ndarray((B, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.dma_copy(token_indices, kv_cache_update_idx)

    batch_offset = nl.ndarray((B, 1), dtype=nl.uint32, buffer=nl.sbuf)
    nisa.iota(batch_offset, [[0, 1]], offset=0, channel_multiplier=S_max_ctx)
    nisa.tensor_tensor(token_indices, token_indices, batch_offset, nl.add)

    # Vector DMA with indirect addressing:
    # - Reshape cache to (B*S_max_ctx, d_head) so each row has stride d_head
    # - vector_offset provides per-batch row indices: token_indices[b]
    # - DMA engine scales token_indices[b] by d_head (stride of indirect_dim=0)

    # Update V_cache on lnc=0
    if n_prgs == 1 or prg_id == 0:
        nisa.dma_copy(
            dst=V_cache.reshape((B * S_max_ctx, d_head)).ap(
                pattern=[[1, B], [d_head, S_tkg], [1, d_head]],
                offset=0,
                vector_offset=token_indices,
                indirect_dim=0,
            ),
            src=V_tkg.ap(pattern=[[S_tkg * d_head, B], [d_head, S_tkg], [1, d_head]]),
        )

    # Update K_cache on lnc=1
    if n_prgs == 1 or prg_id == 1:
        # Transpose K_tkg from [d_head, B*S_tkg] to [B*S_tkg, d_head] to match V_tkg layout
        K_psum = nl.ndarray((B * S_tkg, d_head), K_tkg.dtype, nl.psum)
        K_reshaped_sb = nl.ndarray(K_psum.shape, K_tkg.dtype, nl.sbuf)
        nisa.nc_transpose(dst=K_psum, data=K_tkg)
        nisa.tensor_copy(dst=K_reshaped_sb, src=K_psum)

        # Write K using same access pattern as V
        nisa.dma_copy(
            dst=K_cache.reshape((B * S_max_ctx, d_head)).ap(
                pattern=[[1, B], [d_head, S_tkg], [1, d_head]],
                offset=0,
                vector_offset=token_indices,
                indirect_dim=0,
            ),
            src=K_reshaped_sb.ap(pattern=[[S_tkg * d_head, B], [d_head, S_tkg], [1, d_head]]),
        )


def _update_flat_cache(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    K_cache_transposed: bool,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    S_max_ctx: int,
    B: int,
    d_head: int,
) -> None:
    """
    Update flat (non-block) KV cache with new tokens using per-batch scalar_offset.

    This version iterates over batches and uses scalar_offset for indirect addressing.
    Supports any B*S_tkg <= PMAX and handles both transposed and non-transposed K cache layouts.

    Args:
        K_cache: [B, d_head, S_max_ctx] if transposed else [B, S_max_ctx, d_head] @ HBM
        V_cache: [B, S_max_ctx, d_head] @ HBM
        K_tkg: [d_head, B*S_tkg] @ SBUF
        V_tkg: [B*S_tkg, d_head] @ SBUF
        K_cache_transposed: K cache layout flag
        kv_cache_update_idx: [B, 1] per-batch write positions
        S_tkg: number of new tokens per batch
        S_max_ctx: maximum cache sequence length
        B: batch size
        d_head: head dimension
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")

    # Validate tensor shapes
    kernel_assert(
        kv_cache_update_idx.shape == (B, 1),
        f"kv_cache_update_idx shape mismatch: expected {(B, 1)}, got {kv_cache_update_idx.shape}",
    )
    kernel_assert(
        V_cache.shape == (B, S_max_ctx, d_head),
        f"V_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {V_cache.shape}",
    )
    kernel_assert(
        V_tkg.shape == (B * S_tkg, d_head),
        f"V_tkg shape mismatch: expected {(B * S_tkg, d_head)}, got {V_tkg.shape}",
    )
    kernel_assert(
        K_tkg.shape == (d_head, B * S_tkg),
        f"K_tkg shape mismatch: expected {(d_head, B * S_tkg)}, got {K_tkg.shape}",
    )

    # Update V_cache on lnc=0
    if n_prgs == 1 or prg_id == 0:
        start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        for batch_idx in range(B):
            nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx])
            nisa.dma_copy(
                dst=V_cache.ap(
                    pattern=[[d_head, S_tkg], [1, d_head]],
                    offset=batch_idx * S_max_ctx * d_head,
                    scalar_offset=start_position,
                    indirect_dim=1,
                ),
                src=V_tkg[nl.ds(batch_idx * S_tkg, S_tkg), :],
            )

    # Update K_cache on lnc=1
    if n_prgs == 1 or prg_id == 1:
        if K_cache_transposed:
            kernel_assert(
                K_cache.shape == (B, d_head, S_max_ctx),
                f"K_cache shape mismatch: expected {(B, d_head, S_max_ctx)}, got {K_cache.shape}",
            )
            # K_tkg is already in correct layout [d_head, B*S_tkg]
            start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
            for batch_idx in range(B):
                nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx])
                nisa.dma_copy(
                    dst=K_cache.ap(
                        pattern=[[S_max_ctx, d_head], [1, S_tkg]],
                        offset=batch_idx * d_head * S_max_ctx,
                        scalar_offset=start_position,
                        indirect_dim=2,
                    ),
                    src=K_tkg[:, nl.ds(batch_idx * S_tkg, S_tkg)],
                )
        else:
            kernel_assert(
                K_cache.shape == (B, S_max_ctx, d_head),
                f"K_cache shape mismatch: expected {(B, S_max_ctx, d_head)}, got {K_cache.shape}",
            )
            kernel_assert(
                B * S_tkg <= nl.tile_size.pmax and d_head <= nl.tile_size.pmax,
                f"Transpose constraints: B*S_tkg={B * S_tkg}, d_head={d_head} (both must be <= {nl.tile_size.pmax})",
            )

            # Transpose K_tkg from [d_head, B*S_tkg] to [B*S_tkg, d_head]
            K_psum = nl.ndarray((B * S_tkg, d_head), K_tkg.dtype, nl.psum)
            K_reshaped_sb = nl.ndarray(K_psum.shape, K_tkg.dtype, nl.sbuf)
            nisa.nc_transpose(dst=K_psum, data=K_tkg)
            nisa.tensor_copy(dst=K_reshaped_sb, src=K_psum)

            # Write transposed K to cache
            start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
            for batch_idx in range(B):
                nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx])
                nisa.dma_copy(
                    dst=K_cache.ap(
                        pattern=[[d_head, S_tkg], [1, d_head]],
                        offset=batch_idx * S_max_ctx * d_head,
                        scalar_offset=start_position,
                        indirect_dim=1,
                    ),
                    src=K_reshaped_sb[nl.ds(batch_idx * S_tkg, S_tkg), :],
                )


def _update_block_cache(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    B: int,
) -> None:
    """
    Update block KV cache with new tokens.

    Routes to batched or scalar_offset implementation based on S_tkg.

    Args:
        K_cache: [num_blocks, block_len, d_head]
        V_cache: [num_blocks, block_len, d_head]
        K_tkg: [d_head, B*S_tkg]
        V_tkg: [B*S_tkg, d_head]
        kv_cache_update_idx: [B, 1] slot indices for cache update (uint32 max = skip)
        S_tkg: number of new tokens
        B: batch size
    """
    # TODO: Use batched case for all S_tkg values once vector DMA access pattern supports S_tkg > 1
    if S_tkg == 1:
        # one vector DMA, S_tkg = 1
        _update_block_cache_batched(K_cache, V_cache, K_tkg, V_tkg, kv_cache_update_idx, S_tkg, B)
    else:
        # B scalar DMAs, with any S_tkg
        _update_block_cache_scalar(K_cache, V_cache, K_tkg, V_tkg, kv_cache_update_idx, S_tkg, B)


def _update_block_cache_batched(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    B: int,
) -> None:
    """
    Update block KV cache with new tokens using batched DMA operations.

    This optimized version writes all B batches in a single DMA operation using vector_offset.
    Currently limited to S_tkg=1 due to access pattern bug.

    Args:
        K_cache: [num_blocks, block_len, d_head]
        V_cache: [num_blocks, block_len, d_head]
        K_tkg: [d_head, B*S_tkg]
        V_tkg: [B*S_tkg, d_head]
        kv_cache_update_idx: [B, 1] slot indices for cache update (uint32 max = skip)
        S_tkg: number of new tokens (must be 1)
        B: batch size
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")
    kernel_assert(S_tkg == 1, f"_update_block_cache_batched() only supports S_tkg=1, got {S_tkg}")
    kernel_assert(B * S_tkg <= nl.tile_size.pmax, f"B * S_tkg must be <= {nl.tile_size.pmax}, got {B * S_tkg}")

    num_blocks, blk_len, d_head = K_cache.shape

    kernel_assert(
        kv_cache_update_idx.shape == (B, 1),
        f"kv_cache_update_idx shape mismatch: expected {(B, 1)}, got {kv_cache_update_idx.shape}",
    )
    kernel_assert(
        K_cache.shape == V_cache.shape,
        f"K/V cache shape mismatch: K={K_cache.shape} vs V={V_cache.shape}",
    )
    kernel_assert(
        V_tkg.shape == (B * S_tkg, d_head),
        f"V_tkg shape mismatch: expected {(B * S_tkg, d_head)}, got {V_tkg.shape}",
    )
    kernel_assert(
        K_tkg.shape == (d_head, B * S_tkg),
        f"K_tkg shape mismatch: expected {(d_head, B * S_tkg)}, got {K_tkg.shape}",
    )

    # Transpose K_tkg on lnc=1
    if n_prgs == 1 or prg_id == 1:
        K_psum = nl.ndarray((B * S_tkg, d_head), K_tkg.dtype, nl.psum)
        K_transposed_sb = nl.ndarray(K_psum.shape, K_tkg.dtype, nl.sbuf)
        nisa.nc_transpose(dst=K_psum, data=K_tkg)
        nisa.tensor_copy(dst=K_transposed_sb, src=K_psum)

    # Copy cache update indices to SBUF
    kv_cache_update_idx_sb = nl.ndarray(kv_cache_update_idx.shape, kv_cache_update_idx.dtype, nl.sbuf)
    nisa.dma_copy(kv_cache_update_idx_sb, kv_cache_update_idx)

    # Vector DMA with indirect addressing:
    # - Reshape cache to (num_blocks*blk_len, d_head) so each row has stride d_head
    # - vector_offset provides per-batch global token indices: kv_cache_update_idx_sb[b]
    # - DMA engine scales by d_head

    # Update V_cache on lnc=0
    if n_prgs == 1 or prg_id == 0:
        nisa.dma_copy(
            dst=V_cache.reshape((num_blocks * blk_len, d_head)).ap(
                pattern=[[1, B], [d_head, S_tkg], [1, d_head]],
                offset=0,
                vector_offset=kv_cache_update_idx_sb,
                indirect_dim=0,
            ),
            src=V_tkg.ap(pattern=[[S_tkg * d_head, B], [d_head, S_tkg], [1, d_head]]),
            oob_mode=oob_mode.skip,  # skip writes for invalid batch (position_id = uint32_max)
        )

    # Update K_cache on lnc=1
    if n_prgs == 1 or prg_id == 1:
        nisa.dma_copy(
            dst=K_cache.reshape((num_blocks * blk_len, d_head)).ap(
                pattern=[[1, B], [d_head, S_tkg], [1, d_head]],
                offset=0,
                vector_offset=kv_cache_update_idx_sb,
                indirect_dim=0,
            ),
            src=K_transposed_sb.ap(pattern=[[S_tkg * d_head, B], [d_head, S_tkg], [1, d_head]]),
            oob_mode=oob_mode.skip,  # skip writes for invalid batch (position_id = uint32_max)
        )


def _update_block_cache_scalar(
    K_cache: nl.ndarray,
    V_cache: nl.ndarray,
    K_tkg: nl.ndarray,
    V_tkg: nl.ndarray,
    kv_cache_update_idx: nl.ndarray,
    S_tkg: int,
    B: int,
) -> None:
    """
    Update block KV cache with new tokens using per-batch scalar_offset.

    This version iterates over batches and uses scalar_offset for indirect addressing.
    Supports S_tkg > 1.

    Args:
        K_cache: [num_blocks, block_len, d_head]
        V_cache: [num_blocks, block_len, d_head]
        K_tkg: [d_head, B*S_tkg]
        V_tkg: [B*S_tkg, d_head]
        kv_cache_update_idx: [B, 1] slot indices for cache update (uint32 max = skip)
        S_tkg: number of new tokens
        B: batch size
    """
    _, n_prgs, prg_id = get_verified_program_sharding_info("kv_cache update", (0, 1), 2)
    kernel_assert(n_prgs <= 2, f"Expected lnc in [1,2], got {n_prgs}")
    kernel_assert(B * S_tkg <= nl.tile_size.pmax, f"B * S_tkg must be <= {nl.tile_size.pmax}, got {B * S_tkg}")

    num_blocks, blk_len, d_head = K_cache.shape

    kernel_assert(
        kv_cache_update_idx.shape == (B, 1),
        f"kv_cache_update_idx shape mismatch: expected {(B, 1)}, got {kv_cache_update_idx.shape}",
    )
    kernel_assert(
        K_cache.shape == V_cache.shape,
        f"K/V cache shape mismatch: K={K_cache.shape} vs V={V_cache.shape}",
    )
    kernel_assert(
        V_tkg.shape == (B * S_tkg, d_head),
        f"V_tkg shape mismatch: expected {(B * S_tkg, d_head)}, got {V_tkg.shape}",
    )
    kernel_assert(
        K_tkg.shape == (d_head, B * S_tkg),
        f"K_tkg shape mismatch: expected {(d_head, B * S_tkg)}, got {K_tkg.shape}",
    )

    # Transpose K_tkg once on lnc=1
    if n_prgs == 1 or prg_id == 1:
        K_psum = nl.ndarray((B * S_tkg, d_head), K_tkg.dtype, nl.psum)
        K_transposed_sb = nl.ndarray(K_psum.shape, K_tkg.dtype, nl.sbuf)
        nisa.nc_transpose(dst=K_psum, data=K_tkg)
        nisa.tensor_copy(dst=K_transposed_sb, src=K_psum)

    # Update cache per batch element using scalar_offset
    for batch_idx in range(B):
        start_position = nl.ndarray((1, 1), dtype=nl.uint32, buffer=nl.sbuf)
        nisa.dma_copy(start_position, kv_cache_update_idx[batch_idx])

        # Update V_cache on lnc=0
        if n_prgs == 1 or prg_id == 0:
            nisa.dma_copy(
                dst=V_cache.ap(
                    pattern=[[d_head, S_tkg], [1, d_head]],
                    offset=0,
                    scalar_offset=start_position,
                    indirect_dim=1,
                ),
                src=V_tkg[nl.ds(batch_idx * S_tkg, S_tkg), :],
                oob_mode=oob_mode.skip,  # skip writes for invalid batch (position_id = uint32_max)
            )

        # Update K_cache on lnc=1
        if n_prgs == 1 or prg_id == 1:
            nisa.dma_copy(
                dst=K_cache.ap(
                    pattern=[[d_head, S_tkg], [1, d_head]],
                    offset=0,
                    scalar_offset=start_position,
                    indirect_dim=1,
                ),
                src=K_transposed_sb[nl.ds(batch_idx * S_tkg, S_tkg), :],
                oob_mode=oob_mode.skip,  # skip writes for invalid batch (position_id = uint32_max)
            )
