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

"""PyTorch reference implementation for the fused attention block TKG kernel.

Pipeline::

    X ─► RMSNorm ─► QKV projection ─► QK norm (pre-RoPE) ─► RoPE
      ─► QK norm (post-RoPE) ─► FP8 KV quantization (optional)
      ─► Attention (with KV cache) ─► KV cache update
      ─► Output projection ─► X_out

Dimensions used throughout:
    B: Batch size
    S: Active sequence length (tokens being generated)
    H: Hidden dimension
    D: Head dimension (d_head)
    N: Number of query heads
    S_ctx: Context length (KV cache length visible to attention)
    S_max: Maximum context length (KV cache allocation size)
"""

import inspect
import math
from typing import Dict, Optional, Tuple

import torch

from ...core.attention.attention_tkg_torch import attention_tkg_torch_ref
from ...core.attention.attention_tkg_utils import AttnTKGConfig
from ...core.embeddings.rope_torch import _rope_single_head
from ...core.output_projection.output_projection_tkg_torch import output_projection_tkg_torch_ref
from ...core.qkv.qkv_tkg_torch import qkv_tkg_torch_ref
from ...core.utils.common_types import NormType, QKVOutputLayout, QuantizationType
from ...core.utils.kernel_assert import kernel_assert

_FP8_E4M3_MAX = 240.0


class AttentionBlockTkgTorchRef(torch.nn.Module):
    """PyTorch reference for the fused attention block TKG kernel.

    Stateful ``nn.Module`` whose ``forward`` implements the full pipeline.
    ``self.lnc`` is set at construction and used by the inner attention ref.

    Usage::

        ref = AttentionBlockTkgTorchRef(lnc=2)
        out = ref(X=..., ...)
    """

    def __init__(self, lnc: int = 2):
        super().__init__()
        self.lnc = lnc
        self.__signature__ = inspect.signature(self.forward)

    def forward(
        self,
        # -- input
        X: torch.Tensor,
        *,
        X_hidden_dim_actual: Optional[int],
        # -- rmsnorm X
        rmsnorm_X_enabled: bool,
        rmsnorm_X_eps: Optional[float],
        rmsnorm_X_gamma: Optional[torch.Tensor],
        # -- qkv projections
        W_qkv: torch.Tensor,
        bias_qkv: Optional[torch.Tensor],
        quantization_type_qkv: QuantizationType,
        weight_dequant_scale_qkv: Optional[torch.Tensor],
        input_dequant_scale_qkv: Optional[torch.Tensor],
        # -- QK rmsnorm pre RoPE
        rmsnorm_QK_enabled: bool,
        rmsnorm_QK_eps: Optional[float],
        W_rmsnorm_Q_pre_rope: Optional[torch.Tensor],
        W_rmsnorm_K_pre_rope: Optional[torch.Tensor],
        # -- RoPE
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        rope_contiguous_layout: bool,
        # -- QK rmsnorm post RoPE
        rmsnorm_QK_post_rope_enabled: bool,
        rmsnorm_QK_post_rope_eps: float,
        W_rmsnorm_Q_post_rope: Optional[torch.Tensor],
        W_rmsnorm_K_post_rope: Optional[torch.Tensor],
        # -- attention
        skip_attention: bool,
        K_cache_transposed: bool,
        active_blocks_table: Optional[torch.Tensor],
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        sink: Optional[torch.Tensor],
        softmax_scale: Optional[float],
        # -- FP8 KV cache quantization
        k_scale: Optional[torch.Tensor],
        v_scale: Optional[torch.Tensor],
        # -- KV cache update
        update_cache: bool,
        kv_cache_update_idx: torch.Tensor,
        # -- output projection
        W_out: Optional[torch.Tensor],
        bias_out: Optional[torch.Tensor],
        quantization_type_out: QuantizationType,
        weight_dequant_scale_out: Optional[torch.Tensor],
        input_dequant_scale_out: Optional[torch.Tensor],
        # -- output
        transposed_out: bool,
        out_in_sb: bool,
        # -- kernel-only params (accepted for signature compatibility, validated below)
        sbm: None = None,
        X_in_sb: bool = False,
        KVDP: int = 1,
        KVDP_replica_group: None = None,
    ) -> Dict[str, torch.Tensor]:
        """PyTorch reference for the fused attention block TKG kernel.

        Implements the full fused attention block pipeline for token generation:
        RMSNorm → QKV projection → QK norm → RoPE → QK norm → FP8 KV quantization
        → Attention (with KV cache) → KV cache update → Output projection.

        Each stage delegates to its standalone torch reference where one exists
        (QKV, RoPE, attention, output projection). Stages without a standalone
        kernel (QK norm, FP8 quantization, KV cache update) are implemented
        inline in this module.

        Dimensions:
            B: Batch size
            S: Active sequence length (tokens being generated)
            H: Hidden dimension
            D: Head dimension (d_head)
            N: Number of query heads
            S_ctx: Context length (KV cache length visible to attention)
            S_max: Maximum context length (KV cache allocation size)

        Args:
            X: Input hidden states. Shape: ``[B, S, H]``.
            X_hidden_dim_actual: Actual hidden dimension when H is padded.
                None means H is the true dimension.

            rmsnorm_X_enabled: Apply RMSNorm to X before QKV projection.
            rmsnorm_X_eps: Epsilon for input RMSNorm.
            rmsnorm_X_gamma: Gamma (scale) for input RMSNorm. Shape: ``[1, H]``.

            W_qkv: QKV projection weight. Shape: ``[H, (N + 2) * D]``.
            bias_qkv: Optional QKV bias. Shape: ``[1, (N + 2) * D]``.
            quantization_type_qkv: Quantization for QKV projection (NONE, ROW, STATIC).
            weight_dequant_scale_qkv: QKV weight dequantization scale.
            input_dequant_scale_qkv: QKV input dequantization scale.

            rmsnorm_QK_enabled: Apply per-head RMSNorm to Q/K before RoPE.
            rmsnorm_QK_eps: Epsilon for pre-RoPE QK norm.
            W_rmsnorm_Q_pre_rope: Gamma for Q pre-RoPE norm. Shape: ``[1, D]``.
            W_rmsnorm_K_pre_rope: Gamma for K pre-RoPE norm. Shape: ``[1, D]``.

            cos: RoPE cosine table. Shape: ``[D//2, B, S]``. None to skip RoPE.
            sin: RoPE sine table. Shape: ``[D//2, B, S]``. None to skip RoPE.
            rope_contiguous_layout: RoPE half-dimension split layout.

            rmsnorm_QK_post_rope_enabled: Apply per-head RMSNorm to Q/K after RoPE.
            rmsnorm_QK_post_rope_eps: Epsilon for post-RoPE QK norm.
            W_rmsnorm_Q_post_rope: Gamma for Q post-RoPE norm. Shape: ``[1, D]``.
            W_rmsnorm_K_post_rope: Gamma for K post-RoPE norm. Shape: ``[1, D]``.

            skip_attention: If True, skip attention and pass Q through directly.
            K_cache_transposed: K cache layout is ``[B, 1, D, S_max]`` vs ``[B, 1, S_max, D]``.
            active_blocks_table: Block-to-slot mapping for block KV cache.
                Shape: ``[B, S_ctx // block_len]``. None for non-block KV.
            K_cache: Key cache tensor (see :func:`_update_kv_cache` for shapes).
            V_cache: Value cache tensor.
            attention_mask: Causal attention mask. Shape: ``[S_ctx, B, N, S]``.
            sink: Optional attention sink tensor. Forwarded to attention ref.
            softmax_scale: Custom softmax scale. None uses ``1/√D``.

            k_scale: FP8 quantization scale for K. Shape: ``[1, 1]`` or ``[P, 1]``.
                None to disable FP8 KV quantization.
            v_scale: FP8 quantization scale for V. Same shape convention as k_scale.

            update_cache: Whether to write new K/V into the cache.
            kv_cache_update_idx: Per-batch cache write position. Shape: ``[B, 1]``.

            W_out: Output projection weight. Shape: ``[N*D, H]``. None to skip.
            bias_out: Output projection bias. Shape: ``[1, H]``.
            quantization_type_out: Quantization for output projection.
            weight_dequant_scale_out: Output projection weight dequantization scale.
            input_dequant_scale_out: Output projection input dequantization scale.

            transposed_out: Transpose the final output.
            out_in_sb: Output in SBUF layout (``[D, B, N, S]`` instead of ``[B, N, D, S]``).

            sbm: Accepted for signature compatibility (kernel-only, SBUF memory handle).
            X_in_sb: Accepted for signature compatibility (kernel-only, input SBUF flag).
            KVDP: Must be 1. This ref computes a single-rank result; KVDP slicing
                is handled by the test harness. Asserted to prevent misuse.
            KVDP_replica_group: Accepted for signature compatibility (kernel-only).
            Dict with keys:
                - ``"X_out"``: Output tensor. Shape depends on ``W_out``, ``transposed_out``,
                  and ``out_in_sb`` settings.
                - ``"K_cache_updated"`` / ``"K_tkg"``: Updated K cache (or raw K if
                  ``update_cache`` is False).
                - ``"V_cache_updated"`` / ``"V_tkg"``: Updated V cache (or raw V).
        """
        kernel_assert(
            KVDP == 1, f"Torch ref is single-rank only, got KVDP={KVDP}. Slice inputs per-rank before calling."
        )

        X = X.float()
        B, S_tkg, H, d_head, q_heads, S_ctx, S_max_ctx, blk_len = self._extract_shapes(
            X,
            W_qkv,
            K_cache,
            K_cache_transposed,
            attention_mask,
            active_blocks_table,
        )

        # -- QKV projection (reuse qkv_tkg_torch_ref)
        qkv_result = qkv_tkg_torch_ref(
            hidden=X,
            qkv_w=W_qkv,
            norm_w=rmsnorm_X_gamma if rmsnorm_X_enabled else None,
            norm_type=NormType.RMS_NORM if rmsnorm_X_enabled else NormType.NO_NORM,
            eps=rmsnorm_X_eps if rmsnorm_X_eps else 1e-6,
            d_head=d_head,
            num_kv_heads=1,
            num_q_heads=q_heads,
            output_layout=QKVOutputLayout.NBSd,
            quantization_type=quantization_type_qkv,
            qkv_w_scale=weight_dequant_scale_qkv,
            qkv_in_scale=input_dequant_scale_qkv,
            qkv_bias=bias_qkv,
            hidden_actual=X_hidden_dim_actual,
        )
        QKV = qkv_result['out'].float()  # [N+2, B, S, D]

        # -- QK norm pre RoPE (in-place on QKV)
        QKV = self._qk_norm_pre_rope(
            QKV,
            q_heads,
            d_head,
            rmsnorm_QK_enabled,
            rmsnorm_QK_eps,
            W_rmsnorm_Q_pre_rope,
            W_rmsnorm_K_pre_rope,
        )

        # -- RoPE → Q [D, B, N, S], K [D, B, S]
        cos_f = cos.float() if cos is not None else None
        sin_f = sin.float() if sin is not None else None
        Q, K = self._apply_rope_to_qk(QKV, cos_f, sin_f, q_heads, d_head, rope_contiguous_layout)

        # -- QK norm post RoPE (Q stays [D, B, N, S], K stays [D, B, S])
        Q, K = self._qk_norm_post_rope(
            Q,
            K,
            d_head,
            rmsnorm_QK_post_rope_enabled,
            rmsnorm_QK_post_rope_eps,
            W_rmsnorm_Q_post_rope,
            W_rmsnorm_K_post_rope,
        )

        V = QKV[-1]  # [B, S, D]

        # -- FP8 KV cache quantization (optional)
        kv_quant = k_scale is not None and v_scale is not None
        K_new = K.clone()
        V_new = V.clone()
        if kv_quant:
            K_new, V_new = self._quantize_kv_to_fp8(K_new, V_new, k_scale, v_scale)

        # -- Attention (reuse attention_tkg_torch_ref)
        # Default output for the skip_attention + no output projection path.
        output = Q.clone()  # [D,B,N,S]
        if skip_attention:
            attn_out = Q.permute(1, 2, 0, 3)  # [D,B,N,S] -> [B,N,D,S]
        else:
            attn_out = self._run_attention(
                Q,
                K_new,
                V_new,
                K_cache,
                V_cache,
                attention_mask,
                active_blocks_table,
                K_cache_transposed,
                blk_len,
                S_ctx,
                S_max_ctx,
                softmax_scale,
                q_heads,
                sink=sink,
            )  # [B, N, D, S]
            output = attn_out.permute(2, 0, 1, 3) if out_in_sb else attn_out  # may be overridden

        # -- KV cache update
        K_out, V_out = self._update_kv_cache(
            K_new,
            V_new,
            K_cache.float(),
            V_cache.float(),
            kv_cache_update_idx,
            update_cache,
            K_cache_transposed,
            blk_len,
            d_head,
        )

        # -- Output projection (reuse output_projection_tkg_torch_ref)
        if W_out is not None:
            attn_D_B_N_S = attn_out.permute(2, 0, 1, 3)  # [B,N,D,S] -> [D,B,N,S]
            out_result = output_projection_tkg_torch_ref(
                attention=attn_D_B_N_S,
                weight=W_out,
                bias=bias_out,
                quantization_type=quantization_type_out,
                weight_scale=weight_dequant_scale_out,
                input_scale=input_dequant_scale_out,
                TRANSPOSE_OUT=transposed_out,
            )
            output = out_result['out']

        # -- Build result dict
        if update_cache:
            return {"X_out": output, "K_cache_updated": K_out, "V_cache_updated": V_out}
        return {"X_out": output, "K_tkg": K_out, "V_tkg": V_out}

    # ── Helper methods ──────────────────────────────────────────────────────

    def _extract_shapes(
        self,
        X: torch.Tensor,
        W_qkv: torch.Tensor,
        K_cache: torch.Tensor,
        K_cache_transposed: bool,
        attention_mask: torch.Tensor,
        active_blocks_table: Optional[torch.Tensor],
    ) -> Tuple[int, int, int, int, int, int, int, int]:
        """Extract and validate tensor dimensions, mirroring the kernel's
        ``_validate_and_extract_config``.

        Expected input shapes:

        - ``X``:  ``(B, S_tkg, H)``
        - ``W_qkv``:  ``(H, (q_heads + 2) * d_head)``
        - ``K_cache`` flat:  ``(B, 1, S_max_ctx, d_head)``  or transposed ``(B, 1, d_head, S_max_ctx)``
        - ``K_cache`` block:  ``(n_blocks, blk_len, d_head)``
        - ``attention_mask``:  ``(S_ctx, B, q_heads, S_tkg)``

        Returns:
            ``(B, S_tkg, H, d_head, q_heads, S_ctx, S_max_ctx, blk_len)``
        """
        B, S_tkg, H = X.shape  # (B, S_tkg, H)

        is_block_kv = active_blocks_table is not None

        if is_block_kv:
            # block: (n_blocks, blk_len, d_head)
            d_head = K_cache.shape[2]
        elif K_cache_transposed:
            # flat transposed: (B, 1, d_head, S_max_ctx)
            d_head = K_cache.shape[2]
        else:
            # flat: (B, 1, S_max_ctx, d_head)
            d_head = K_cache.shape[3]

        # W_qkv packs Q, K, V heads: (H, (q_heads + 2*kv_heads) * d_head), kv_heads=1
        q_heads = W_qkv.shape[1] // d_head - 2
        kernel_assert(q_heads > 0, f"q_heads must be > 0, got {q_heads}")
        kernel_assert(d_head % 2 == 0, f"d_head must be even, got {d_head}")

        S_ctx = attention_mask.shape[0]  # (S_ctx, B, q_heads, S_tkg)
        blk_len = K_cache.shape[1] if is_block_kv else 0

        if is_block_kv:
            S_max_ctx = S_ctx
        elif K_cache_transposed:
            # (B, 1, d_head, S_max_ctx)
            S_max_ctx = K_cache.shape[3]
        else:
            # (B, 1, S_max_ctx, d_head)
            S_max_ctx = K_cache.shape[2]

        return B, S_tkg, H, d_head, q_heads, S_ctx, S_max_ctx, blk_len

    def _rms_norm(self, x: torch.Tensor, axis: int, eps: float, w: Optional[torch.Tensor] = None) -> torch.Tensor:
        """RMSNorm: ``x / sqrt(mean(x², axis) + eps) * w``.

        Args:
            x: Input tensor.
            axis: Dimension to normalize over.
            eps: Epsilon for numerical stability.
            w: Optional affine scale (gamma), broadcastable to *x*.
        """
        normalized = x / torch.sqrt(torch.mean(x**2, dim=axis, keepdim=True) + eps)
        if w is not None:
            normalized = normalized * w
        return normalized

    def _qk_norm_pre_rope(
        self,
        QKV: torch.Tensor,
        num_heads: int,
        d_head: int,
        enabled: bool,
        eps: float,
        W_Q: Optional[torch.Tensor],
        W_K: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """Apply per-head RMSNorm to Q and K before RoPE (in-place).

        Args:
            QKV: Combined QKV tensor. Shape: ``[N+2, B, S, D]``.
            num_heads: Number of query heads (N).
            d_head: Head dimension (D).
            enabled: Whether to apply normalization.
            eps: Epsilon for RMSNorm.
            W_Q: Optional gamma for Q heads. Shape: ``[1, D]``.
            W_K: Optional gamma for K head. Shape: ``[1, D]``.

        Returns:
            QKV tensor (modified in-place). Shape: ``[N+2, B, S, D]``.
        """
        if not enabled:
            return QKV
        w_Q = W_Q.float().reshape(1, 1, 1, d_head) if W_Q is not None else None
        w_K = W_K.float().reshape(1, 1, 1, d_head) if W_K is not None else None
        QKV[:num_heads] = self._rms_norm(QKV[:num_heads], axis=-1, eps=eps, w=w_Q)
        QKV[num_heads] = self._rms_norm(QKV[num_heads], axis=-1, eps=eps, w=w_K)
        return QKV

    def _apply_rope_to_qk(
        self,
        QKV: torch.Tensor,
        cos: Optional[torch.Tensor],
        sin: Optional[torch.Tensor],
        num_heads: int,
        d_head: int,
        rope_contiguous_layout: bool,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Rotary Position Embedding to Q and K heads.

        Transposes from QKV's ``[N+2, B, S, D]`` layout to the ``[D, B, *, S]``
        layout expected by downstream attention, applying RoPE per-head.

        Args:
            QKV: Combined QKV tensor. Shape: ``[N+2, B, S, D]``.
            cos: Cosine table for RoPE. Shape: ``[D//2, B, S]``. None to skip RoPE.
            sin: Sine table for RoPE. Shape: ``[D//2, B, S]``. None to skip RoPE.
            num_heads: Number of query heads (N).
            d_head: Head dimension (D).
            rope_contiguous_layout: Whether RoPE uses contiguous (True) or
                interleaved (False) layout for the half-dimension split.

        Returns:
            Tuple of (Q, K):
                - Q: Shape ``[D, B, N, S]``.
                - K: Shape ``[D, B, S]``.
        """
        _, batch, S_tkg, _ = QKV.shape
        skip_rope = cos is None or sin is None
        QK = torch.zeros(d_head, batch, num_heads + 1, S_tkg)
        for b in range(batch):
            for h in range(num_heads + 1):
                cur_head = QKV[h, b, :, :].T
                if skip_rope:
                    QK[:, b, h, :] = cur_head
                else:
                    QK[:, b, h, :] = _rope_single_head(cur_head, cos[:, b, :], sin[:, b, :], rope_contiguous_layout)
        return QK[:, :, :num_heads, :], QK[:, :, num_heads, :]

    def _qk_norm_post_rope(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        d_head: int,
        enabled: bool,
        eps: float,
        W_Q: Optional[torch.Tensor],
        W_K: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply per-head RMSNorm to Q and K after RoPE.

        Args:
            Q: Query tensor. Shape: ``[D, B, N, S]``.
            K: Key tensor. Shape: ``[D, B, S]``.
            d_head: Head dimension (D).
            enabled: Whether to apply normalization.
            eps: Epsilon for RMSNorm.
            W_Q: Optional gamma for Q. Shape: ``[1, D]``.
            W_K: Optional gamma for K. Shape: ``[1, D]``.

        Returns:
            Tuple of (Q, K) with same shapes as inputs.
        """
        if not enabled:
            return Q, K
        w_Q = W_Q.float().reshape(d_head, 1, 1, 1) if W_Q is not None else None
        w_K = W_K.float().reshape(d_head, 1, 1) if W_K is not None else None
        return self._rms_norm(Q, axis=0, eps=eps, w=w_Q), self._rms_norm(K, axis=0, eps=eps, w=w_K)

    def _quantize_kv_to_fp8(
        self,
        K: torch.Tensor,
        V: torch.Tensor,
        k_scale: torch.Tensor,
        v_scale: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Quantize K and V to FP8-E4M3 range: scale and clamp to [-240, 240].

        Args:
            K: Key tensor (any shape, typically ``[D, B, S]``), in float32.
            V: Value tensor (any shape, typically ``[B, S, D]``), in float32.
            k_scale: Scale for K. Only ``k_scale[0, 0]`` is used (broadcast).
            v_scale: Scale for V. Only ``v_scale[0, 0]`` is used (broadcast).

        Returns:
            Tuple of (K_quantized, V_quantized) in float32, clamped to FP8 range.
        """
        K_scaled = K * k_scale[0, 0].float()
        V_scaled = V * v_scale[0, 0].float()
        k_clip = (K_scaled.abs() > _FP8_E4M3_MAX).float().mean().item()
        v_clip = (V_scaled.abs() > _FP8_E4M3_MAX).float().mean().item()
        kernel_assert(k_clip <= 0.01, f"Too many K values clipped ({k_clip:.1%}), scale may be inappropriate")
        kernel_assert(v_clip <= 0.01, f"Too many V values clipped ({v_clip:.1%}), scale may be inappropriate")
        return (
            torch.clamp(K_scaled, -_FP8_E4M3_MAX, _FP8_E4M3_MAX),
            torch.clamp(V_scaled, -_FP8_E4M3_MAX, _FP8_E4M3_MAX),
        )

    def _run_attention(
        self,
        Q: torch.Tensor,
        K_active: torch.Tensor,
        V_active: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        attention_mask: torch.Tensor,
        active_blocks_table: Optional[torch.Tensor],
        K_cache_transposed: bool,
        block_len: int,
        S_ctx: int,
        S_max_ctx: int,
        softmax_scale: Optional[float],
        num_heads: int,
        sink: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Run scaled dot-product attention via :func:`attention_tkg_torch_ref`.

        Applies softmax scaling to Q, then delegates to the standalone attention
        TKG reference which handles the KV cache lookup and masked attention.

        Args:
            Q: Query tensor. Shape: ``[D, B, N, S]``.
            K_active: Active key tensor. Shape: ``[D, B, S]``.
            V_active: Active value tensor. Shape: ``[B, S, D]``.
            K_cache: Key cache. Shape: ``[blocks, block_len, D]`` (block KV) or
                ``[B, 1, D, S_max]`` (transposed) or ``[B, 1, S_max, D]``.
            V_cache: Value cache. Shape: ``[blocks, block_len, D]`` (block KV) or
                ``[B, 1, S_max, D]``.
            attention_mask: Causal mask. Shape: ``[S_ctx, B, N, S]``.
            active_blocks_table: Block-to-slot mapping for block KV cache.
                Shape: ``[B, S_ctx // block_len]``. None for non-block KV.
            K_cache_transposed: Whether K cache has transposed layout.
            block_len: Block length for block KV cache (0 = non-block).
            S_ctx: Context length visible to attention.
            S_max_ctx: Maximum context length (cache allocation size).
            softmax_scale: Custom softmax scale. None uses ``1/√D``.
            num_heads: Number of query heads (N).
            sink: Optional attention sink tensor. Forwarded to attention ref.

        Returns:
            Attention output. Shape: ``[B, N, D, S]``.
        """
        d_head, batch, _, S_tkg = Q.shape
        q = Q.permute(1, 2, 3, 0)
        k_active_attn = K_active.reshape(d_head, batch, 1, S_tkg).permute(1, 2, 3, 0)
        v_active_attn = V_active.reshape(batch, 1, S_tkg, d_head)

        if softmax_scale is not None:
            q = q * softmax_scale
        else:
            q = q / math.sqrt(d_head)

        is_block_kv = block_len > 0
        cfg = AttnTKGConfig(
            bs=batch,
            q_head=num_heads,
            s_active=S_tkg,
            curr_sprior=S_ctx,
            full_sprior=S_max_ctx,
            d_head=d_head,
            block_len=block_len,
            tp_k_prior=not K_cache_transposed,
            strided_mm1=not is_block_kv,
            use_pos_id=False,
            fuse_rope=False,
        )
        out = torch.zeros(batch, num_heads, d_head, S_tkg, dtype=torch.float32)
        abt = active_blocks_table.to(torch.int32) if active_blocks_table is not None else None

        attention_tkg_torch_ref[self.lnc](
            q=q,
            k_active=k_active_attn,
            v_active=v_active_attn,
            k_prior=K_cache.float(),
            v_prior=V_cache.float(),
            mask=attention_mask.to(torch.uint8),
            out=out,
            cfg=cfg,
            sbm=None,
            sink=sink,
            active_blocks_table=abt if is_block_kv else None,
        )
        return out

    def _update_kv_cache(
        self,
        K_new: torch.Tensor,
        V_new: torch.Tensor,
        K_cache: torch.Tensor,
        V_cache: torch.Tensor,
        kv_cache_update_idx: torch.Tensor,
        update_cache: bool,
        K_cache_transposed: bool,
        block_len: int,
        d_head: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Write newly projected K/V into the KV cache at the given positions.

        When ``update_cache`` is False, returns the new K/V tensors unchanged
        (they are still needed as outputs for the test harness).

        For block KV caches, ``kv_cache_update_idx`` contains *physical* flat
        indices into the block table. An index of ``-1`` (from uint32 0xFFFFFFFF
        converted to int32 by the test wrapper) means "skip this batch element".

        Args:
            K_new: New key tensor from projection. Shape: ``[D, B, S]``.
            V_new: New value tensor from projection. Shape: ``[B, S, D]``.
            K_cache: Key cache to update. Shape: ``[blocks, block_len, D]``
                (block KV) or ``[B, 1, D, S_max]`` / ``[B, 1, S_max, D]``.
            V_cache: Value cache to update. Same shape convention as K_cache.
            kv_cache_update_idx: Per-batch write position. Shape: ``[B, 1]``.
            update_cache: Whether to actually write into the cache.
            K_cache_transposed: Whether K cache uses ``[B, 1, D, S_max]`` layout.
            block_len: Block length (0 = non-block KV cache).
            d_head: Head dimension (D).

        Returns:
            Tuple of (K_cache_updated, V_cache_updated).
        """
        if not update_cache:
            return K_new, V_new

        K_cache = K_cache.clone()
        V_cache = V_cache.clone()
        batch, S_tkg = K_new.shape[1], K_new.shape[2]

        if block_len > 0:
            num_blocks = K_cache.shape[0]
            K_flat = K_cache.reshape(num_blocks * block_len, d_head)
            V_flat = V_cache.reshape(num_blocks * block_len, d_head)
            for b in range(batch):
                idx = int(kv_cache_update_idx[b, 0].item())
                if idx < 0 or idx == 0xFFFFFFFF:
                    continue
                K_flat[idx : idx + S_tkg, :] = K_new[:, b, :].T
                V_flat[idx : idx + S_tkg, :] = V_new[b, :, :]
            return K_flat.reshape(num_blocks, block_len, d_head), V_flat.reshape(num_blocks, block_len, d_head)

        for b in range(batch):
            start_pos = int(kv_cache_update_idx[b, 0].item())
            if K_cache_transposed:
                K_cache[b, 0, :, start_pos : start_pos + S_tkg] = K_new[:, b, :]
            else:
                K_cache[b, 0, start_pos : start_pos + S_tkg, :] = K_new[:, b, :].T
            V_cache[b, 0, start_pos : start_pos + S_tkg, :] = V_new[b, :, :]
        return K_cache, V_cache
